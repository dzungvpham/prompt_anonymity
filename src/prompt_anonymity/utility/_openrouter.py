"""Lean OpenRouter chat client shared by the utility metrics.

The utility metrics make remote chat calls against OpenRouter -- generating a response model's
answer to a prompt, asking a judge to rule PASS/FAIL on two answers, or asking one to score two
whole conversations 1-5. This module holds the one client they all use: :class:`OpenRouterChat`, a
thin wrapper over the chat-completions endpoint: a lazily read ``OPENROUTER_API_KEY`` (from a
``.env``), jittered exponential backoff on transient failures, fail-fast on non-retryable 4xx, and
a thread pool to fan a batch of requests out.

The judges stay remote because a judge is meant to be a stronger, independent model than the one
under test. They are no longer the only remote caller: the OpenAnonymity defense used to be one and
now runs a local model through vLLM (:mod:`prompt_anonymity.defenses.openanonymity`), but the Frame
Shift defense (:mod:`prompt_anonymity.defenses.frame_shift`) reuses this client to rewrite prompts
through a hosted model. That second caller is why :meth:`OpenRouterChat.complete` takes a per-call
``max_tokens``: a judge's reply is a fixed-size verdict, a rewrite's is as long as its input.

One prompt is one request, with no token-budget chunking / context-length re-split (which the
scrubber does do). That is safe for the per-turn callers, whose inputs are turn-sized and bounded;
the conversation-level judge (:mod:`.prompt_judge`), which sends two whole conversations per call,
is *not* inherently bounded, and relies on its own ``max_chars`` cap to stay under a context window
-- overrun there surfaces as a fail-fast 4xx that aborts the batch. ``requests`` and
``python-dotenv`` are imported lazily so importing this module (e.g. to reach the prompt constants
or the verdict parser) never requires the network deps or a key.
"""

from __future__ import annotations

import os
import random
import time

#: OpenRouter chat-completions endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
#: Environment variable holding the OpenRouter API key (loaded from a ``.env`` if present).
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterChat:
    """Minimal OpenRouter chat client: one fixed system prompt, one user message per call.

    Built lazily by the utility scorer on first real need, so a fully-cached scoring run
    constructs no client and needs no API key. ``temperature=0`` by default -> greedy and
    reproducible, so cache hits are stable.

    Parameters
    ----------
    model : str
        OpenRouter chat model id (e.g. ``"openai/gpt-4o"``).
    system_prompt : str
        System message prepended to every request (the response-model persona or the judge rubric).
    temperature, top_p, max_tokens : float / float / int
        Standard sampling controls; ``temperature=0`` keeps decoding deterministic.
    max_workers : int
        Thread-pool width for :meth:`complete_batch` (requests are network I/O-bound).
    max_retries, backoff_cap, timeout : int / float / float
        Retry budget for transient failures, backoff ceiling (seconds), and per-request timeout.
    """

    def __init__(self, model: str, system_prompt: str = "", *, temperature: float = 0.0,
                 top_p: float = 1.0, max_tokens: int = 1024, max_workers: int = 8,
                 max_retries: int = 8, backoff_cap: float = 30.0, timeout: float = 120.0,
                 base_url: str = OPENROUTER_BASE_URL, api_key_env: str = OPENROUTER_API_KEY_ENV):
        import requests
        from dotenv import load_dotenv

        self._requests = requests
        self.model = model
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        self.timeout = timeout
        self.base_url = base_url

        # Walks UP from the working directory: a .env at the repo root (or above it) is found, one
        # in a SUBdirectory is not -- `DS_env/.env` does not work when the job runs from the root.
        load_dotenv()
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Add it to a .env file (e.g. '{api_key_env}=sk-or-...') "
                "so the utility metric can call OpenRouter."
            )
        print(f"Utility: OpenRouter client using model '{model}'.")

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, 2 ** attempt)
        time.sleep(random.uniform(0, delay))  # full jitter de-synchronizes concurrent workers

    def complete(self, text: str, max_tokens: int | None = None) -> str:
        """Return the model's reply to ``text`` under the fixed system prompt.

        A blank input has nothing to answer and 400s some providers, so short-circuit it to ``""``.
        Transient failures (429, 5xx, network/JSON errors) are retried with jittered exponential
        backoff; a non-retryable 4xx (bad model id, malformed/over-limit prompt) fails fast with
        OpenRouter's error body attached (``raise_for_status`` alone carries only the status line).

        ``max_tokens`` overrides the instance default for this one call. The judges leave it unset
        (their replies are a verdict and a sentence, so one fixed cap fits every call); a *rewrite*
        caller needs a budget proportional to its input, since a long turn's rewrite is long too and
        the fixed default would silently truncate it.
        """
        if not text.strip():
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature, "top_p": self.top_p,
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(self.base_url, headers=headers, json=payload,
                                           timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            except self._requests.exceptions.HTTPError as err:
                status = err.response.status_code
                body = err.response.text
                last_err = RuntimeError(f"{status} {err.response.reason}: {body}")
                # 429 (rate-limit) is transient and retryable; other 4xx are client errors that fail
                # identically on retry, so surface the offending prompt and stop.
                if 400 <= status < 500 and status != 429:
                    snippet = text[:200].replace("\n", " ")
                    raise RuntimeError(
                        f"OpenRouter rejected the request (HTTP {status}) for model {self.model!r}; "
                        f"not retrying. Offending prompt: {len(text)} chars, starts {snippet!r}. "
                        f"Response body: {body}"
                    ) from err
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
            except Exception as err:  # noqa: BLE001 - network/JSON errors are all retryable
                last_err = err
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
        raise RuntimeError(f"OpenRouter request failed after {self.max_retries} attempts: {last_err}")

    def complete_batch(self, texts: list[str], max_tokens=None) -> list[str]:
        """Answer a batch of prompts concurrently, preserving input order.

        Network I/O-bound, so a thread pool gives near-linear speedup despite the GIL; a prompt
        that still fails after retries raises, aborting the batch (same contract as the scrubber).

        ``max_tokens`` is either ``None`` (use the instance default for every call), one int applied
        to all of them, or a sequence carrying a per-prompt budget -- which is what a rewrite caller
        wants, its budget scaling with each input's length.
        """
        from concurrent.futures import ThreadPoolExecutor

        texts = list(texts)
        if not texts:
            return []
        if max_tokens is None or isinstance(max_tokens, int):
            budgets = [max_tokens] * len(texts)
        else:
            budgets = [int(b) for b in max_tokens]
            if len(budgets) != len(texts):
                raise ValueError(f"max_tokens has {len(budgets)} entries but texts has {len(texts)}.")
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map over both iterables -> preserves order, re-raises the first failure.
            return list(pool.map(self.complete, texts, budgets))
