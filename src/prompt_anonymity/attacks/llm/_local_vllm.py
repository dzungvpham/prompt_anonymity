"""A local vLLM judge behind the same interface as :class:`._openrouter.OpenRouterChat`.

The listwise reranker (:mod:`.listwise_llm_rerank`) asks a judge to order five shortlisted authors
and justify each position. This client runs that judge **locally** -- Qwen3.8-27B by default, the
same checkpoint AFR's agent uses -- and is called the way AFR calls it
(:meth:`prompt_anonymity.defenses.afr` ``_engine`` / ``chat_pairs``): one offline ``vllm.LLM``
built lazily, then batched ``engine.chat(conversations, SamplingParams, use_tqdm=False)``. The
attack cannot tell it from the API clients: ``complete`` / ``complete_batch`` / ``complete_stream``
and the reply counters are the same.

Thinking is ON
--------------
AFR leaves the chat template's default alone; this judge asks for thinking explicitly with
``chat_template_kwargs={"enable_thinking": True}`` (EmBad passes the same switch set to ``False``).
The reply then carries the model's reasoning ahead of a ``</think>`` marker, and **only the text
after it is returned** -- see :func:`split_thinking`. That is not cosmetic: the ranking parser slices
from the first ``{`` to the last ``}``, so a brace anywhere in the thinking would corrupt the JSON.

Sampling follows Qwen's published thinking-mode settings (:data:`THINKING_SAMPLING`) rather than
AFR's greedy decoding: greedy decoding with thinking on is prone to repetition loops that run to the
token limit. Verdicts are cached by prompt text, so a re-run still reproduces what was judged.

Serving knobs (environment, outside the cache key -- they change where the model runs, not what it
is asked): ``RERANK_QWEN_GPU_MEM_UTIL`` (0.90; nothing else shares the card, unlike AFR's Harrier),
``RERANK_QWEN_MAX_MODEL_LEN`` (32768), ``RERANK_QWEN_MAX_NUM_SEQS`` (128 -- **required** for this
hybrid Mamba/attention checkpoint, whose default of 256 does not start; see ``afr.AFR_MAX_NUM_SEQS``),
``RERANK_QWEN_CHUNK`` (prompts per ``chat()`` call in :meth:`LocalVLLMChat.complete_stream`).

``vllm``/``torch`` are imported lazily, so importing this module costs nothing.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path

#: ``models.toml`` section and environment override for the judge checkpoint.
MODEL_CONFIG_SECTION = "rerank_qwen"
MODEL_ENV_VAR = "RERANK_QWEN_MODEL"
#: Opt in to fetching a checkpoint that is not already on disk (~54 GB for the default).
ALLOW_DOWNLOAD = os.environ.get("RERANK_QWEN_ALLOW_DOWNLOAD", "") == "1"

RERANK_QWEN_GPU_MEM_UTIL = float(os.environ.get("RERANK_QWEN_GPU_MEM_UTIL", "0.90"))
RERANK_QWEN_MAX_MODEL_LEN = int(os.environ.get("RERANK_QWEN_MAX_MODEL_LEN", "32768"))
RERANK_QWEN_MAX_NUM_SEQS = int(os.environ.get("RERANK_QWEN_MAX_NUM_SEQS", "128"))
#: Prompts per ``chat()`` call when streaming. vLLM batches inside a chunk; the verdict cache banks
#: after every chunk, so this is also how much work a preemption can cost.
RERANK_QWEN_CHUNK = int(os.environ.get("RERANK_QWEN_CHUNK", "128"))

#: Qwen's recommended sampling for thinking mode. Part of the verdict cache key.
THINKING_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"


def split_thinking(text: str, finish_reason: str | None = None) -> tuple[str, str, bool, bool]:
    """``(thinking, answer, reasoned, truncated)`` for one raw completion.

    * ``...</think> answer`` -- the answer is what follows the LAST ``</think>``; any ``<think>``
      before it is dropped (some templates put the opening tag in the prompt, so only the closing
      one appears in the completion). ``reasoned`` is true.
    * no ``</think>`` and the completion stopped at the length limit -- it was cut off while still
      thinking, so there is no answer: ``("<the thinking>", "", False, True)``.
    * no think tags at all -- the whole text is the answer, not counted as reasoned.

    ``truncated`` is also true when an answer exists but the completion still hit the limit (the
    JSON after ``</think>`` may be cut short); the parser repairs what it can of such a reply.
    """
    text = text or ""
    hit_limit = finish_reason == "length"
    if THINK_CLOSE in text:
        thinking, _, answer = text.rpartition(THINK_CLOSE)
        thinking = thinking.replace(THINK_OPEN, "", 1).strip()
        return thinking, answer.strip(), True, hit_limit
    if hit_limit:
        return text.replace(THINK_OPEN, "", 1).strip(), "", False, True
    return "", text.strip(), False, False


def judge_checkpoint() -> str:
    """The judge checkpoint, resolved to a loadable directory, or a loud failure naming the fix.

    Same resolution as AFR (:func:`prompt_anonymity.defenses._backends.model_checkpoint`), with the
    same refusal: a checkpoint that is not already on disk is an error unless
    ``RERANK_QWEN_ALLOW_DOWNLOAD=1``, because silently pulling ~54 GB onto a compute node is worse
    than failing in the first second.
    """
    from ...defenses._backends import model_checkpoint

    path = model_checkpoint(MODEL_CONFIG_SECTION, MODEL_ENV_VAR, local_only=not ALLOW_DOWNLOAD)
    if not ALLOW_DOWNLOAD and not Path(path).is_dir():
        raise RuntimeError(
            f"the rerank judge checkpoint {path!r} is not on disk. Point ${MODEL_ENV_VAR} at a local "
            f"copy, fix [{MODEL_CONFIG_SECTION}] in models.toml, or set "
            f"RERANK_QWEN_ALLOW_DOWNLOAD=1 to fetch it."
        )
    return path


class LocalVLLMChat:
    """``OpenRouterChat``'s interface over a local vLLM model with thinking on.

    Parameters
    ----------
    model : str
        Label for the judge (logged and used in the attack's cache key). The weights loaded are
        :func:`judge_checkpoint`'s.
    system_prompt : str
        Fixed system prompt for every request.
    max_tokens : int
        Completion budget, **thinking included**.
    seed : int
        vLLM engine seed and per-request sampling seed.

    Attributes
    ----------
    n_replies, n_replies_with_reasoning, n_truncated : int
        Completions received, those that closed a thinking block, and those that hit
        ``max_tokens``.
    total_reasoning_tokens : int
        Tokens spent inside thinking blocks, counted with the engine's tokenizer.
    total_cost : float
        Always ``0.0``: local.
    """

    def __init__(self, model: str, system_prompt: str = "", *, max_tokens: int = 16384,
                 seed: int = 47, chunk: int = RERANK_QWEN_CHUNK, **_ignored):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.seed = seed
        self.chunk = max(1, int(chunk))
        self.total_cost = 0.0
        self.n_replies = 0
        self.n_replies_with_reasoning = 0
        self.n_truncated = 0
        self.total_reasoning_tokens = 0
        self._lock = threading.Lock()
        self._llm = None
        self._sampling = None

    # -- the engine, as AFR builds it --
    def _engine(self):
        if self._llm is None:
            from ...defenses._backends import configure_cuda_toolkit

            configure_cuda_toolkit()  # must precede the import: vLLM reads the environment at import

            from vllm import LLM, SamplingParams

            path = judge_checkpoint()
            print(f"LLM judge: loading {path} locally (vLLM), thinking on, "
                  f"max_model_len={RERANK_QWEN_MAX_MODEL_LEN:,}, "
                  f"gpu_memory_utilization={RERANK_QWEN_GPU_MEM_UTIL}, "
                  f"max_num_seqs={RERANK_QWEN_MAX_NUM_SEQS}", flush=True)
            # Prefix caching pays here: the system prompt is identical for every call.
            self._llm = LLM(model=path, dtype="auto",
                            gpu_memory_utilization=RERANK_QWEN_GPU_MEM_UTIL,
                            max_model_len=RERANK_QWEN_MAX_MODEL_LEN,
                            max_num_seqs=RERANK_QWEN_MAX_NUM_SEQS,
                            enable_prefix_caching=True, seed=self.seed)
            self._sampling = SamplingParams(max_tokens=self.max_tokens, seed=self.seed,
                                            **THINKING_SAMPLING)
        return self._llm

    def _count_tokens(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(self._engine().get_tokenizer()(text).input_ids)
        except Exception:  # noqa: BLE001 - a diagnostic count must not fail a run
            return 0

    def _chat(self, texts: list[str]) -> list[str]:
        """One batched ``chat()`` over ``texts``; visible answers back in input order."""
        engine = self._engine()
        conversations = [[{"role": "system", "content": self.system_prompt},
                          {"role": "user", "content": text}] for text in texts]
        outputs = engine.chat(conversations, self._sampling, use_tqdm=False,
                              chat_template_kwargs={"enable_thinking": True})
        answers = []
        for output in outputs:
            completion = output.outputs[0]
            thinking, answer, reasoned, truncated = split_thinking(
                completion.text, getattr(completion, "finish_reason", None))
            tokens = self._count_tokens(thinking)
            with self._lock:
                self.n_replies += 1
                self.n_replies_with_reasoning += int(reasoned)
                self.n_truncated += int(truncated)
                self.total_reasoning_tokens += tokens
            answers.append(answer)
        return answers

    # -- OpenRouterChat's interface --
    def complete(self, text: str, max_tokens: int | None = None) -> str:
        return self._chat([text])[0] if text.strip() else ""

    def complete_batch(self, texts: list[str], max_tokens=None) -> list[str]:
        texts = list(texts)
        replies: list[str] = [""] * len(texts)
        for index, reply in self.complete_stream(texts):
            replies[index] = reply
        return replies

    def complete_stream(self, texts: list[str], max_tokens=None):
        """``(index, answer)`` for every prompt, yielded one chunk of :attr:`chunk` at a time.

        vLLM batches within a chunk; yielding after each lets the verdict cache bank it, so a
        preempted job loses at most one chunk. Blank prompts are answered ``""`` without a call.
        """
        texts = list(texts)
        live = [index for index, text in enumerate(texts) if text.strip()]
        for index, text in enumerate(texts):
            if not text.strip():
                yield index, ""
        for start in range(0, len(live), self.chunk):
            block = live[start:start + self.chunk]
            for index, answer in zip(block, self._chat([texts[i] for i in block])):
                yield index, answer

    def close(self) -> None:
        if self._llm is not None:
            from ...defenses._backends import shutdown_vllm

            shutdown_vllm(self._llm)
            self._llm = None
