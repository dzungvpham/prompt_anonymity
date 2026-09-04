"""Style-convergence rewrite defense: rewrite every prompt into one fixed, neutral style.

Threat model: a user's own writing style fingerprints them across queries. This defense rewrites
each user turn into ONE fixed target style (:data:`REWRITE_PROMPT_HEADER`) with a local instruct
model, so turns written by different people converge to a shared, hard-to-link stylometric identity
while their content is preserved. The featurize stage re-derives features from the rewritten text.

The model is toggleable so the defense can be upgraded to a newer instruct model without touching
code: pass ``model_id`` (vLLM path) / ``gguf_repo``+``gguf_file`` (llama.cpp path), or set the
``QWEN_HF_REPO`` / ``QWEN_BACKEND`` env vars. The active model is part of :meth:`params`, so
swapping it invalidates the cache automatically. Two backends: ``vllm`` (bf16, batched -- the right
tool on an A100/H100) and ``llama_cpp`` (a 4-bit GGUF for a laptop/phone).
"""

from __future__ import annotations

import os

from ._backends import PerTurnBatchRewriteDefense

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ EDIT ME — the style every prompt is forced into ("rewrite like X"). Keep   │
# │ the "preserve all info / output only the rewrite" guardrails or the        │
# │ rewritten prompt will drift from what the user actually asked.             │
# └──────────────────────────────────────────────────────────────────────────┘
REWRITE_PROMPT_HEADER = """You are a prompt-rewriting filter. Rewrite the user's message into a single, neutral, standardized style so that messages written by different people become stylistically indistinguishable.
Rules:
- Preserve ALL information, intent, constraints, and any code, names, numbers, or quotations exactly as given.
- Write in plain, formal English: complete declarative sentences, no slang, no contractions, no emoji, no personal asides or filler.
- Keep the same language as the input message.
- Use active voice and subject-verb-object order only. Do not use passive voice, fronted clauses, cleft constructions ("It is X that..."), or inversions.
- Express every request as a direct imperative beginning with a verb (e.g., 'Write...', 'Summarize...', 'Fix...'). Do not use politeness framings ('Could you', 'I'd like you to', 'Please') or question forms for requests.
- One idea per sentence. Split compound or multi-clause sentences.
- Order the content canonically: (1) the task, (2) constraints and requirements, (3) supporting context or examples.
- Remove hedges and intensifiers ('really', 'very', 'basically', 'just', etc).
- Use only periods and commas. Do not use em dashes, semicolons, colons for asides, ellipses, or parentheticals; rewrite the content into separate sentences instead.
- Use digits for all numbers.
- Do NOT substitute technical terms, domain verbs, or proper nouns for synonyms. Normalize register and grammar only, never denotation.
- Do NOT answer, explain, or comment on the message. Output ONLY the rewritten message and nothing else."""

# Default backend + models. vLLM is the cluster path (bf16, continuous batching); llama.cpp runs a
# 4-bit GGUF on a laptop/phone. Both default to Qwen2.5-3B-Instruct and are env/arg-overridable.
DEFAULT_BACKEND = os.environ.get("QWEN_BACKEND", "vllm").lower()  # "vllm" | "llama_cpp"
DEFAULT_VLLM_MODEL = os.environ.get("QWEN_HF_REPO", "Qwen/Qwen2.5-3B-Instruct")
DEFAULT_GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
DEFAULT_GGUF_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"

# vLLM knobs (env-tunable, matching the DS_env sandbox).
_VLLM_DTYPE = os.environ.get("QWEN_VLLM_DTYPE", "bfloat16")
_VLLM_GPU_MEM_UTIL = float(os.environ.get("QWEN_VLLM_GPU_MEM_UTIL", "0.90"))
_VLLM_MAX_MODEL_LEN = int(os.environ.get("QWEN_VLLM_MAX_MODEL_LEN", "4096"))
# Skip vLLM's nvcc-dependent startup compile (torch.compile + CUDA-graph capture) on a node without
# the CUDA toolkit; a bit slower at decode but runs with only the runtime the wheel bundles.
_VLLM_ENFORCE_EAGER = os.environ.get("QWEN_VLLM_ENFORCE_EAGER", "0") == "1"


class _QwenGGUFRewriter:
    """On-device 4-bit Qwen GGUF backend (llama-cpp-python; CPU-friendly, GPU-offloaded when
    available). Decodes one prompt at a time, so :meth:`rewrite_batch` loops."""

    def __init__(self, repo_id, filename, system_prompt, n_ctx=4096, max_tokens=1024):
        from llama_cpp import Llama

        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        print(f"Loading Qwen GGUF '{repo_id}/{filename}' (4-bit, llama.cpp) for rewrite...")
        # n_gpu_layers=-1 offloads all layers to the GPU when llama-cpp-python was built with CUDA
        # (a harmless no-op otherwise); the single biggest speed lever on a GPU box.
        self.llm = Llama.from_pretrained(
            repo_id=repo_id, filename=filename, n_ctx=n_ctx,
            n_threads=None, n_gpu_layers=-1, verbose=False,
        )

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            # temperature=0 -> greedy/deterministic, so re-runs and cache hits are reproducible.
            resp = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.0, max_tokens=self.max_tokens,
            )
            out.append(resp["choices"][0]["message"]["content"].strip())
        return out


class _QwenVLLMRewriter:
    """Cluster-optimized Qwen backend using vLLM (continuous batching + PagedAttention). Runs the
    whole batch of turns through one shared system prompt, so an A100/H100 stays saturated."""

    def __init__(self, model_id, system_prompt, max_tokens=1024):
        from ._backends import local_checkpoint, resolve_model_path, shared_checkpoint
        from vllm import LLM, SamplingParams

        # Resolved before vLLM sees it. `model_id` is a REPO ID by default, and vLLM cannot tell a
        # repo id from a path that does not exist -- it fetches either. On a cluster that already
        # mirrors these weights that is ~6 GB downloaded into $HF_HOME for nothing, so prefer
        # /datasets/ai and refuse rather than download. Same resolution as afr and loo_unlink.
        if model_id and model_id != DEFAULT_VLLM_MODEL:
            path = resolve_model_path(shared_checkpoint(model_id) or model_id)
            print(f"[qwen_rewrite] checkpoint: {path}")
        else:
            path = local_checkpoint("qwen_rewrite", "QWEN_HF_REPO",
                                    allow_env="QWEN_ALLOW_DOWNLOAD",
                                    size_hint="~6 GB for a 3B model")

        self.system_prompt = system_prompt
        print(f"Loading Qwen (vLLM) '{path}' ({_VLLM_DTYPE}) for rewrite...")
        self.llm = LLM(
            model=path, dtype=_VLLM_DTYPE,
            gpu_memory_utilization=_VLLM_GPU_MEM_UTIL,
            max_model_len=_VLLM_MAX_MODEL_LEN, enforce_eager=_VLLM_ENFORCE_EAGER,
        )
        self.sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        conversations = [
            [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": text}]
            for text in texts
        ]
        outs = self.llm.chat(conversations, self.sampling, use_tqdm=False)
        return [o.outputs[0].text.strip() for o in outs]


class QwenRewriteDefense(PerTurnBatchRewriteDefense):
    """Rewrite every user turn into the fixed :data:`REWRITE_PROMPT_HEADER` style with Qwen.

    Parameters
    ----------
    backend : {"vllm", "llama_cpp"}
        Inference backend; defaults to ``QWEN_BACKEND`` (``"vllm"``).
    model_id : str
        vLLM model repo/path (unquantized); defaults to ``QWEN_HF_REPO`` or Qwen2.5-3B-Instruct.
    gguf_repo, gguf_file : str
        llama.cpp GGUF repo + file for the ``llama_cpp`` backend.
    system_prompt : str
        The convergence-target instructions; defaults to :data:`REWRITE_PROMPT_HEADER`.

    The backend is built lazily on first use, so a fully-cached run loads no model. The active model
    is reported by :meth:`params`, so switching models invalidates the cache automatically.
    """

    name = "qwen_rewrite"
    version = "1"

    def __init__(self, *, backend: str = DEFAULT_BACKEND, model_id: str = DEFAULT_VLLM_MODEL,
                 gguf_repo: str = DEFAULT_GGUF_REPO, gguf_file: str = DEFAULT_GGUF_FILE,
                 system_prompt: str = REWRITE_PROMPT_HEADER):
        self.backend = backend
        self.model_id = model_id
        self.gguf_repo = gguf_repo
        self.gguf_file = gguf_file
        self.system_prompt = system_prompt
        self._backend = None

    def _active_model(self) -> str:
        """String identifying the model actually loaded, for the cache key."""
        return self.model_id if self.backend == "vllm" else f"{self.gguf_repo}/{self.gguf_file}"

    def params(self) -> dict:
        # Everything that determines the rewrite goes in the cache key. The system prompt is a
        # module constant (not part of the class source the logic hash covers), so include it
        # verbatim: editing REWRITE_PROMPT_HEADER or passing a custom prompt then re-caches.
        return {
            "backend": self.backend,
            "model": self._active_model(),
            "system_prompt": self.system_prompt,
        }

    def _get_backend(self):
        if self._backend is None:
            if self.backend == "vllm":
                self._backend = _QwenVLLMRewriter(self.model_id, self.system_prompt)
            else:
                self._backend = _QwenGGUFRewriter(self.gguf_repo, self.gguf_file, self.system_prompt)
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        return self._get_backend().rewrite_batch(texts)
