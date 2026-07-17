"""Lean OpenRouter chat client shared by the utility-fidelity stage.

The fidelity metric makes two kinds of remote chat call -- generating a response model's answer to
a prompt, and asking a judge to score two answers -- both against OpenRouter. This module holds the
one client both use: :class:`OpenRouterChat`, a thin wrapper over the chat-completions endpoint with
the same production niceties as the OpenAnonymity defense's backend
(:class:`prompt_anonymity.defenses.openanonymity._OpenAnonBackend`) -- a lazily read
``OPENROUTER_API_KEY`` (from a ``.env``), jittered exponential backoff on transient failures,
fail-fast on non-retryable 4xx, and a thread pool to fan a batch of requests out.

It is deliberately **leaner** than the scrubber backend: the response and judge inputs here are
turn-sized and bounded, so there is no token-budget chunking / context-length re-split -- one
prompt is one request. Like the scrubber, ``requests`` and ``python-dotenv`` are imported lazily so
importing this module (e.g. to reach the prompt constants or the verdict parser) never requires the
network deps or a key.
"""

from __future__ import annotations

import os
import random
import time

#: OpenRouter chat-completions endpoint (same as the OpenAnonymity defense).
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
#: Environment variable holding the OpenRouter API key (loaded from a ``.env`` if present).
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterChat:
    """Minimal OpenRouter chat client: one fixed system prompt, one user message per call.

    Built lazily by the fidelity scorer on first real need, so a fully-cached scoring run
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

        load_dotenv()  # walks up from cwd, so the key can live in the repo root or a subdir
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Add it to a .env file (e.g. '{api_key_env}=sk-or-...') "
                "so the fidelity metric can call OpenRouter."
            )
        print(f"Fidelity: OpenRouter client using model '{model}'.")

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, 2 ** attempt)
        time.sleep(random.uniform(0, delay))  # full jitter de-synchronizes concurrent workers

    def complete(self, text: str) -> str:
        """Return the model's reply to ``text`` under the fixed system prompt.

        A blank input has nothing to answer and 400s some providers, so short-circuit it to ``""``.
        Transient failures (429, 5xx, network/JSON errors) are retried with jittered exponential
        backoff; a non-retryable 4xx (bad model id, malformed/over-limit prompt) fails fast with
        OpenRouter's error body attached (``raise_for_status`` alone carries only the status line).
        """
        if not text.strip():
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens,
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

    def complete_batch(self, texts: list[str]) -> list[str]:
        """Answer a batch of prompts concurrently, preserving input order.

        Network I/O-bound, so a thread pool gives near-linear speedup despite the GIL; a prompt
        that still fails after retries raises, aborting the batch (same contract as the scrubber).
        """
        from concurrent.futures import ThreadPoolExecutor

        texts = list(texts)
        if not texts:
            return []
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.complete, texts))  # preserves order, re-raises first failure
