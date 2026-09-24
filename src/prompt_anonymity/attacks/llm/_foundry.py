"""Claude on Microsoft Foundry, behind the same interface as :class:`._openrouter.OpenRouterChat`.

The listwise Sonnet reranker (:mod:`.listwise_llm_rerank`) was built on OpenRouter, but the project's
Sonnet 5 access is a **Foundry deployment** reached through the Anthropic SDK's
:class:`~anthropic.AnthropicFoundry` client. :class:`FoundryChat` subclasses ``OpenRouterChat`` and
replaces only the transport -- the constructor and :meth:`FoundryChat.complete` -- so the thread
pool, the streaming generator the verdict cache consumes, and the cost/reasoning counters are all
inherited unchanged. The attack cannot tell the two clients apart.

Reasoning
---------
Sonnet 5 reasons through **adaptive thinking**: ``thinking={"type": "adaptive"}`` plus
``output_config={"effort": ...}``. A fixed ``budget_tokens`` and the sampling controls
(``temperature``/``top_p``) are rejected with a 400, so neither is sent. Thinking arrives as
``thinking`` blocks in the response content -- with empty text by default, since the display mode
defaults to omitted, but the blocks are there -- and that is what the reasoning counter counts. The
API reports no separate thinking-token figure, so ``total_reasoning_tokens`` stays 0 here; the
evidence is the block count.

Thinking and the visible answer share ``max_tokens``. A reply cut off there parses as nothing and
silently degrades that row to the distance order, so the caller should budget generously (you pay
only for tokens produced) and :attr:`FoundryChat.n_truncated` records every one that still hits it.

Credentials
-----------
:data:`FOUNDRY_API_KEY_ENV` and :data:`FOUNDRY_ENDPOINT_ENV`, from the environment or the project's
``.env``. **Both are required**: with the key alone the SDK would post an Azure key to Anthropic's own
API and fail in a way that reads like a bad key rather than a misrouted request (the lesson recorded
beside :class:`prompt_anonymity.defenses.embad.HostedMutator`, the other Foundry caller, whose
credential lookup this one copies).

``anthropic`` and ``python-dotenv`` are imported lazily, so importing this module costs nothing.
"""

from __future__ import annotations

import os
import threading

from ._openrouter import OpenRouterChat

#: Environment variable holding the Foundry key. Shared with the OpenRouter unbatched channel of the
#: listwise reranker by the project's choice -- a ``.env`` holding a Foundry key here is not a valid
#: OpenRouter key, and vice versa.
FOUNDRY_API_KEY_ENV = "SONNET_API_KEY"
#: Environment variable holding the Foundry endpoint (the resource's base URL).
FOUNDRY_ENDPOINT_ENV = "FOUNDRY_ENDPOINT"

#: The deployment the project's Sonnet 5 access is published under. **A Foundry deployment name,
#: not an Anthropic model id** -- on Foundry the ``model`` field names the deployment.
DEFAULT_FOUNDRY_MODEL = "claude-sonnet-5-2"

#: Dollars per million (input, output) tokens. Claude on Microsoft Foundry bills at standard
#: Anthropic API rates; same figures as ``embad.HOSTED_MUTATOR_RATES``. Hardcoded and able to go
#: stale -- tokens are the record, dollars are derived -- and a deployment missing from this table
#: reports ``nan`` rather than a confident zero.
FOUNDRY_RATES = {
    "claude-sonnet-5-2": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
}

#: SDK-level retries for 408/409/429/5xx and connection errors (its default is 2, too few for a
#: thousand-request run behind a shared rate limit). Backoff is the SDK's own.
FOUNDRY_MAX_RETRIES = 8
#: Seconds per request. Generous: a reply with high-effort thinking can take a while.
FOUNDRY_TIMEOUT = 600.0


def foundry_credential(name: str) -> str:
    """One credential, from the environment or from the project's ``.env``.

    Reads the single key with ``dotenv_values`` rather than ``load_dotenv()``, which would write
    every key in the file into this process's environment -- see
    :meth:`prompt_anonymity.defenses.embad.HostedMutator.credential` for why that bit before.
    """
    value = os.environ.get(name)
    if value:
        return value
    from dotenv import dotenv_values, find_dotenv

    for path in (find_dotenv(), find_dotenv(usecwd=True)):
        if path:
            found = (dotenv_values(path) or {}).get(name)
            if found:
                return found
    raise RuntimeError(
        f"{name} is not set and is not in the project's .env. The Foundry client needs both "
        f"{FOUNDRY_API_KEY_ENV} (the key) and {FOUNDRY_ENDPOINT_ENV} (the endpoint URL)."
    )


class FoundryChat(OpenRouterChat):
    """``OpenRouterChat``'s interface over Claude on Microsoft Foundry.

    Parameters
    ----------
    model : str
        Foundry deployment name (default :data:`DEFAULT_FOUNDRY_MODEL`).
    system_prompt : str
        Fixed system prompt for every request.
    max_tokens : int
        Output budget per reply, **thinking included**.
    reasoning_effort : str or None
        ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"`` turns on adaptive thinking at that
        effort. ``None`` sends neither ``thinking`` nor ``output_config``.
    max_workers : int
        Concurrent requests for :meth:`complete_batch` / :meth:`complete_stream`.

    Attributes
    ----------
    n_truncated, n_refused : int
        Replies that stopped at ``max_tokens``, and replies declined with ``stop_reason ==
        "refusal"`` (returned as ``""``, which parses as nothing).
    input_tokens, output_tokens : int
        Billed tokens across every reply; ``total_cost`` is derived from them.
    """

    def __init__(self, model: str = DEFAULT_FOUNDRY_MODEL, system_prompt: str = "", *,
                 max_tokens: int = 16000, reasoning_effort: str | None = None,
                 max_workers: int = 8, **_ignored):
        # Deliberately not super().__init__: that reads an OpenRouter key and builds a requests
        # session. Only the attributes the inherited pool/stream/counter code reads are set here.
        from anthropic import AnthropicFoundry

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.total_cost = 0.0
        self.n_replies = 0
        self.n_replies_with_reasoning = 0
        self.total_reasoning_tokens = 0
        self.n_truncated = 0
        self.n_refused = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._cost_lock = threading.Lock()
        self._client = AnthropicFoundry(
            api_key=foundry_credential(FOUNDRY_API_KEY_ENV),
            base_url=foundry_credential(FOUNDRY_ENDPOINT_ENV),
            max_retries=FOUNDRY_MAX_RETRIES,
            timeout=FOUNDRY_TIMEOUT,
        )
        print(f"LLM judge: Microsoft Foundry client using deployment '{model}'"
              + (f", adaptive thinking at effort '{reasoning_effort}'." if reasoning_effort
                 else ", no thinking."))

    def _price(self, input_tokens: int, output_tokens: int) -> float:
        rates = FOUNDRY_RATES.get(self.model)
        if rates is None:
            return float("nan")
        return input_tokens * rates[0] / 1e6 + output_tokens * rates[1] / 1e6

    def _record_message(self, message) -> None:
        """Tokens, estimated dollars, reasoning evidence and stop reason for one reply."""
        usage = getattr(message, "usage", None)
        tokens_in = int(getattr(usage, "input_tokens", 0) or 0)
        tokens_out = int(getattr(usage, "output_tokens", 0) or 0)
        reasoned = any(getattr(block, "type", None) in ("thinking", "redacted_thinking")
                       for block in (getattr(message, "content", None) or []))
        stop = getattr(message, "stop_reason", None)
        with self._cost_lock:
            self.n_replies += 1
            self.input_tokens += tokens_in
            self.output_tokens += tokens_out
            self.total_cost += self._price(tokens_in, tokens_out)
            if reasoned:
                self.n_replies_with_reasoning += 1
            if stop == "max_tokens":
                self.n_truncated += 1
            elif stop == "refusal":
                self.n_refused += 1

    def complete(self, text: str, max_tokens: int | None = None) -> str:
        """The model's visible answer to ``text`` under the fixed system prompt.

        Transient failures are retried by the SDK (:data:`FOUNDRY_MAX_RETRIES`). A non-retryable
        4xx -- a wrong deployment name, a rejected parameter, a bad key -- fails fast with the
        offending prompt's start attached, the same contract as ``OpenRouterChat.complete``.
        """
        import anthropic

        if not text.strip():
            return ""
        request = {
            "model": self.model,
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "system": self.system_prompt,
            "messages": [{"role": "user", "content": text}],
        }
        if self.reasoning_effort:
            request["thinking"] = {"type": "adaptive"}
            request["output_config"] = {"effort": self.reasoning_effort}
        try:
            message = self._client.messages.create(**request)
        except anthropic.APIStatusError as err:
            if 400 <= err.status_code < 500 and err.status_code not in (408, 409, 429):
                snippet = text[:200].replace("\n", " ")
                raise RuntimeError(
                    f"Foundry rejected the request (HTTP {err.status_code}) for deployment "
                    f"{self.model!r}; not retrying. Offending prompt: {len(text)} chars, starts "
                    f"{snippet!r}. Error: {err.message}"
                ) from err
            raise
        self._record_message(message)
        if message.stop_reason == "refusal":
            return ""
        return "".join(block.text for block in message.content if block.type == "text")


def probe(model: str = DEFAULT_FOUNDRY_MODEL, reasoning_effort: str | None = "high") -> dict:
    """One tiny real request with the settings a run will use. Costs a fraction of a cent.

    Confirms what a key check alone cannot: that the endpoint answers, the deployment name exists,
    and adaptive thinking plus this effort are accepted rather than 400'd. Whether *this* reply
    thought is reported, not required -- adaptive thinking may skip a trivial question -- so the
    real evidence is the run's own reply count.
    """
    client = FoundryChat(model, "Answer with the number only.", max_tokens=2000,
                         reasoning_effort=reasoning_effort, max_workers=1)
    reply = client.complete("What is 17 * 23 - 4?")
    return {"reply": reply.strip()[:40], "reasoned": client.n_replies_with_reasoning > 0,
            "truncated": client.n_truncated > 0, "refused": client.n_refused > 0,
            "input_tokens": client.input_tokens, "output_tokens": client.output_tokens,
            "cost_usd": client.total_cost}
