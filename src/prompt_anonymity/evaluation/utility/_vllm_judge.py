"""Self-hosted judge: the conversation-utility rubric against a local vLLM server.

The same OpenAI-compatible ``/chat/completions`` surface as the hosted DeepSeek judge
(:mod:`._deepseek`), so :class:`VLLMJudge` is only endpoint resolution on top of
:class:`~._deepseek.OpenAICompatibleJudge`: everything that shapes a verdict -- the rubric, the
input format, the score parser, the cache -- is shared, and the two backends differ in *where* a
request goes and *what it costs*.

**Where the server is.** ``$LOCAL_LLM_BASE_URL`` when set (a full ``http://host:port/v1``), else
``http://localhost:$LOCAL_LLM_PORT/v1`` -- the port ``scripts/serve_qwen.sh`` reads from the same
``.env`` key, so the server and this client agree on one value. vLLM needs no key; the SDK insists
on a non-empty one, so ``"EMPTY"`` is sent (vLLM's own convention).

**Which model.** The served model name is part of the judge cache key (it is ``judge_model`` in
:meth:`~.prompt_judge.ConversationUtility.params`), so it must name what the server really runs.
Rather than hardcode it, :func:`served_model_name` asks the server's ``/v1/models`` and takes the
one model it lists; pass a name explicitly when the server hosts several, or to score a fully cached
run with the server down.

**What it costs: nothing, recorded as ``0.0``, not ``nan``.** ``nan`` in ``judge_cost_usd`` means
"a bill exists but its price is unknown"; a self-hosted request has no bill, so :data:`FREE` rates
make it an honest zero. The GPU time is real but is paid in SLURM allocation, like the local
``semantic``/``fluency`` metrics.

**Reasoning and sampling are honoured here, unlike on DeepSeek.** Probed 2026-09-25 against
``qwen3.8-27b`` (Qwen3.8-27B-FP8, ``--reasoning-parser qwen3``):

* Thinking is **on** by default (~345 output tokens on a toy pair, 1.3k reasoning chars).
* ``reasoning_effort`` is **validated** -- ``"bogus"`` and even ``"high"`` are a 400; the chat
  template accepts ``none``, ``low``, ``medium`` and ``xhigh`` (its default). ``"low"`` measured 222
  tokens, ``"none"`` 13 with no reasoning at all -- a real dial, not DeepSeek's coarse one.
* ``temperature``/``top_p`` **do** steer the sampler. As with DeepSeek, the response cache -- not
  the sampler -- is what makes a re-run reproducible.

**Thinking is off by default (``reasoning_effort="none"``); the chain of thought is written into
the answer instead**, as the ``Chain_of_thought`` field of
:data:`~.prompt_judge.JUDGE_RESPONSE_FORMAT`. With thinking on, the model reasoned on a 10-point
scale in its hidden trace and the 1-5 schema then cut "10" to "1" -- see that constant. Sampling is
therefore Qwen's recommended *non-thinking* pair (0.7 / 0.8); greedy decoding is still avoided,
since Qwen's model card warns it causes repetition.
"""

from __future__ import annotations

import json
import os
import urllib.request

from ._deepseek import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT_SECONDS,
    OpenAICompatibleJudge,
    TokenRates,
)

#: Environment variables for the endpoint (process env or the gitignored ``.env``).
LOCAL_LLM_BASE_URL_ENV = "LOCAL_LLM_BASE_URL"
LOCAL_LLM_PORT_ENV = "LOCAL_LLM_PORT"
LOCAL_LLM_HOST = "localhost"

#: Placeholder key: vLLM checks none unless started with ``--api-key``, the SDK requires one.
LOCAL_LLM_API_KEY = "EMPTY"

#: A self-hosted request bills nothing; see the module docstring for why this is 0, not ``None``.
FREE = TokenRates(input=0.0, cached_input=0.0, output=0.0)

#: Qwen's recommended non-thinking sampling (the model card's pair); see the module docstring.
DEFAULT_LOCAL_TEMPERATURE = 0.7
DEFAULT_LOCAL_TOP_P = 0.8

#: Hidden thinking off: the reasoning is written into the answer's ``Chain_of_thought`` field.
DEFAULT_LOCAL_REASONING_EFFORT = "none"

#: Client-side concurrency. The server batches internally (``serve_qwen.sh`` caps it at
#: ``--max-num-seqs 5``) and queues the rest, so a few more in flight than it runs keeps it full.
DEFAULT_LOCAL_MAX_WORKERS = 8


def local_base_url() -> str:
    """The local server's ``/v1`` URL, from ``$LOCAL_LLM_BASE_URL`` or ``$LOCAL_LLM_PORT``."""
    from ...data.config import _load_dotenv_once

    _load_dotenv_once()
    explicit = os.environ.get(LOCAL_LLM_BASE_URL_ENV)
    if explicit:
        return explicit.rstrip("/")
    port = os.environ.get(LOCAL_LLM_PORT_ENV)
    if not port:
        raise RuntimeError(
            f"no local judge endpoint: set {LOCAL_LLM_BASE_URL_ENV} (e.g. http://host:8383/v1) or "
            f"{LOCAL_LLM_PORT_ENV} in .env, or pass --judge-base-url."
        )
    return f"http://{LOCAL_LLM_HOST}:{port.strip()}/v1"


def served_model_name(base_url: str, timeout: float = 10.0) -> str:
    """The one model the server at ``base_url`` serves, read from its ``/v1/models``.

    Raises when the server is unreachable or lists several models, naming the fix either way --
    guessing would put the wrong model name into the cache key.
    """
    url = f"{base_url.rstrip('/')}/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            listing = json.load(response)
    except OSError as error:
        raise RuntimeError(
            f"could not reach the local judge at {url} ({error}). Start it with "
            "scripts/serve_qwen.sh, or pass --judge-model to name the model explicitly "
            "(enough for a fully cached run)."
        ) from None
    names = [entry["id"] for entry in listing.get("data", [])]
    if len(names) != 1:
        raise RuntimeError(f"{url} serves {names or 'no models'}; pass --judge-model to pick one.")
    return names[0]


class VLLMJudge(OpenAICompatibleJudge):
    """:class:`~._deepseek.OpenAICompatibleJudge` against a self-hosted vLLM server.

    Parameters
    ----------
    model : str
        Served model name (``--served-model-name``); see :func:`served_model_name`.
    system_prompt : str
        The rubric.
    base_url : str, optional
        Endpoint; defaults to :func:`local_base_url`.
    **kwargs
        Forwarded to :class:`~._deepseek.OpenAICompatibleJudge` (``max_tokens``,
        ``temperature``, ``top_p``, ``reasoning_effort``, ``max_workers``, ...).
    """

    def __init__(self, model: str, system_prompt: str = "", *, base_url: str | None = None,
                 max_workers: int = DEFAULT_LOCAL_MAX_WORKERS,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS, **kwargs):
        super().__init__(model, system_prompt, base_url=base_url or local_base_url(),
                         api_key=LOCAL_LLM_API_KEY, rates=FREE, max_workers=max_workers,
                         max_retries=max_retries, timeout=timeout, **kwargs)

    def _server_root(self) -> str:
        """The server's root URL: vLLM serves ``/tokenize`` beside ``/v1``, not under it."""
        root = str(self._client.base_url).rstrip("/")
        return root[:-len("/v1")] if root.endswith("/v1") else root

    def input_budget(self) -> int:
        """Most prompt tokens a request can carry: the served context minus ``max_tokens``.

        vLLM rejects a request whose prompt plus ``max_tokens`` exceeds ``max_model_len`` (read
        from ``/v1/models``, so it follows however the server was started).
        """
        with urllib.request.urlopen(f"{self._server_root()}/v1/models", timeout=30) as response:
            listing = json.load(response)
        lengths = {entry["id"]: entry.get("max_model_len") for entry in listing.get("data", [])}
        if not lengths.get(self.model):
            raise RuntimeError(f"the server does not report max_model_len for '{self.model}'.")
        return int(lengths[self.model]) - self.max_tokens

    def input_token_counts(self, texts: list[str]) -> list[int]:
        """Exact prompt length of each request, from the server's own ``/tokenize``.

        Exact because it is the server applying the model's chat template to the same system +
        user messages :meth:`complete` sends -- a local tokenizer would have to reproduce that
        template, and an estimate would either waste context or let an over-long request through.
        Concurrent, since it is one small HTTP call per conversation.
        """
        from concurrent.futures import ThreadPoolExecutor

        def count(text: str) -> int:
            body = json.dumps({"model": self.model, "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ]}).encode()
            request = urllib.request.Request(f"{self._server_root()}/tokenize", data=body,
                                             headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=120) as response:
                return int(json.load(response)["count"])

        with ThreadPoolExecutor(max_workers=16) as pool:
            return list(pool.map(count, texts))
