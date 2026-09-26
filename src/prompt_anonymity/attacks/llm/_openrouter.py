"""Lean OpenRouter chat client shared by the LLM-judge attacks.

:class:`OpenRouterChat` is a thin wrapper over the chat-completions endpoint: a lazily read
``OPENROUTER_API_KEY``, jittered exponential backoff on transient failures, fail-fast on
non-retryable 4xx, and a thread pool to fan a batch of requests out.

**Reasoning models are first-class here.** ``reasoning_effort`` sends ``reasoning.effort``, and
``temperature=None``/``top_p=None`` omit those fields entirely rather than sending a default --
Claude Sonnet 5 rejects the sampling controls with a 400 and does its thinking under an effort
level instead.

**Not only the judge attacks call this.** :mod:`prompt_anonymity.defenses.frame_shift` reuses it
to rewrite prompts through a hosted model, so a defense imports it across package boundaries --
worth knowing before moving this module again. That is also why :meth:`OpenRouterChat.complete`
takes a per-call ``max_tokens``: a judge's reply is a fixed-size verdict, a rewrite's is as long as
its input.

One prompt is one request, with no chunking/re-split -- safe here since a judge sees bounded
snippets and the rewriter is handed one turn at a time. ``requests``/``python-dotenv`` are imported
lazily so importing this module never requires the network deps or a key.
"""

from __future__ import annotations

import os
import random
import threading
import time

#: OpenRouter chat-completions endpoint.
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"
#: Environment variable holding the OpenRouter API key (loaded from a ``.env`` if present).
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"


#: Key-introspection endpoint. A GET here authenticates without generating a single token, which
#: is what makes it usable as a preflight.
OPENROUTER_KEY_URL = "https://openrouter.ai/api/v1/key"


def check_credentials(api_key_env: str = OPENROUTER_API_KEY_ENV, *, timeout: float = 30.0):
    """``(ok, message)`` for the key in ``$api_key_env`` -- resolved, then actually used.

    Asks the provider directly rather than just checking the ``.env`` file exists, so a bad or
    missing key is caught before a long job burns time reaching the first request.

    Never returns or logs the key itself, only its length and last four characters.
    """
    import os

    from dotenv import load_dotenv

    load_dotenv()
    key = os.environ.get(api_key_env)
    if not key:
        return False, (f"{api_key_env} is not set, and no .env on the way up from {os.getcwd()} "
                       f"defines it (load_dotenv walks up from the working directory).")
    if key != key.strip():
        return False, (f"{api_key_env} has leading or trailing whitespace ({len(key)} chars); "
                       f"the provider will see a malformed Authorization header.")

    import requests

    shape = f"{len(key)} chars, ending {key[-4:]!r}"
    try:
        response = requests.get(OPENROUTER_KEY_URL, timeout=timeout,
                                headers={"Authorization": f"Bearer {key}"})
    except Exception as err:  # noqa: BLE001 - a preflight must not raise
        return False, f"could not reach OpenRouter to check {api_key_env} ({shape}): {err}"

    if response.status_code == 200:
        body = response.json().get("data", {})
        limit, usage = body.get("limit"), body.get("usage")
        headroom = "no spend limit set" if limit is None else f"limit {limit}, used {usage}"
        return True, f"{api_key_env} authenticates ({shape}); {headroom}"
    if response.status_code == 401:
        return False, (f"{api_key_env} is set ({shape}) but OpenRouter rejects it: 401 "
                       f"{response.text.strip()[:200]}.")
    return False, (f"{api_key_env} ({shape}) got HTTP {response.status_code} from OpenRouter: "
                   f"{response.text.strip()[:200]}")


class OpenRouterChat:
    """Minimal OpenRouter chat client: one fixed system prompt, one user message per call.

    Built lazily on first real need, so a fully-cached run constructs no client and needs no API
    key. ``temperature=0`` by default -> greedy and reproducible, so cache hits are stable.

    Parameters
    ----------
    model : str
        OpenRouter chat model id (e.g. ``"openai/gpt-4o"``).
    system_prompt : str
        System message prepended to every request.
    temperature, top_p, max_tokens : float or None / float or None / int
        Standard sampling controls. ``None`` omits the field from the payload entirely, which a
        reasoning model needs -- Claude Sonnet 5 rejects ``temperature``/``top_p`` outright.
    reasoning_effort : str or None
        ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``, sent as ``reasoning.effort``.
        ``None`` (default) sends no reasoning field.
    max_workers : int
        Thread-pool width for :meth:`complete_batch` (requests are network I/O-bound).
    max_retries, backoff_cap, timeout : int / float / float
        Retry budget for transient failures, backoff ceiling (seconds), and per-request timeout.

    Attributes
    ----------
    total_cost : float
        What OpenRouter reported billing for requests this client actually made, in USD. Best
        effort: a reply with no usage block contributes nothing, so this is a floor on the spend.
    """

    def __init__(self, model: str, system_prompt: str = "", *, temperature: float | None = 0.0,
                 top_p: float | None = 1.0, max_tokens: int = 1024,
                 reasoning_effort: str | None = None, max_workers: int = 8,
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
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        self.timeout = timeout
        self.base_url = base_url
        self.total_cost = 0.0
        # Tracks whether reasoning actually happened, not just whether it was asked for -- see
        # _record_usage.
        self.n_replies = 0
        self.n_replies_with_reasoning = 0
        self.total_reasoning_tokens = 0
        self._cost_lock = threading.Lock()

        load_dotenv()  # walks up from the working directory to find a .env
        self.api_key = os.environ.get(api_key_env)
        if not self.api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Add it to a .env file (e.g. '{api_key_env}=sk-or-...') "
                "so the LLM-judge attack can call OpenRouter."
            )
        print(f"LLM judge: OpenRouter client using model '{model}'.")

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, 2 ** attempt)
        time.sleep(random.uniform(0, delay))  # full jitter de-synchronizes concurrent workers

    def _record_cost(self, reply) -> None:
        """Add one reply's billed cost to :attr:`total_cost`, tolerating any shape it doesn't
        recognise -- a bookkeeping miss should never crash a completed run."""
        try:
            cost = (reply.get("usage") or {}).get("cost")
        except AttributeError:
            return
        if cost is None:
            return
        try:
            cost = float(cost)
        except (TypeError, ValueError):
            return
        with self._cost_lock:
            self.total_cost += cost

    def _record_usage(self, reply) -> None:
        """Count one reply, and whether it reasoned, toward the ``n_replies*`` counters.

        A reply counts as reasoned if its usage reports reasoning tokens or its message carries a
        non-empty ``reasoning``/``reasoning_details`` field. An unrecognised shape counts as a
        reply without reasoning rather than raising.
        """
        tokens, reasoned = 0, False
        try:
            details = (reply.get("usage") or {}).get("completion_tokens_details") or {}
            tokens = int(details.get("reasoning_tokens") or 0)
        except (AttributeError, TypeError, ValueError):
            tokens = 0
        try:
            message = reply["choices"][0]["message"]
            reasoned = bool(message.get("reasoning") or message.get("reasoning_details"))
        except (AttributeError, IndexError, KeyError, TypeError):
            reasoned = False
        with self._cost_lock:
            self.n_replies += 1
            self.total_reasoning_tokens += max(tokens, 0)
            if tokens > 0 or reasoned:
                self.n_replies_with_reasoning += 1

    def complete(self, text: str, max_tokens: int | None = None) -> str:
        """Return the model's reply to ``text`` under the fixed system prompt.

        A blank input has nothing to answer and 400s some providers, so short-circuits to ``""``.
        Transient failures (429, 5xx, network/JSON errors) are retried with jittered exponential
        backoff; a non-retryable 4xx fails fast with OpenRouter's error body attached.

        ``max_tokens`` overrides the instance default for this one call -- useful for a rewrite
        caller whose output length scales with its input, unlike a judge's fixed-size verdict.
        """
        if not text.strip():
            return ""
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            "max_tokens": self.max_tokens if max_tokens is None else int(max_tokens),
            "usage": {"include": True},  # only way to learn the billed cost of a single call
        }
        # Sampling controls are sent only when set -- a reasoning model rejects them outright.
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if self.reasoning_effort:
            payload["reasoning"] = {"effort": self.reasoning_effort}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._requests.post(self.base_url, headers=headers, json=payload,
                                           timeout=self.timeout)
                resp.raise_for_status()
                reply = resp.json()
                self._record_cost(reply)
                self._record_usage(reply)
                return reply["choices"][0]["message"]["content"]
            except self._requests.exceptions.HTTPError as err:
                status = err.response.status_code
                body = err.response.text
                last_err = RuntimeError(f"{status} {err.response.reason}: {body}")
                # 429 is transient; other 4xx fail identically on retry, so stop immediately.
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

        A prompt that still fails after retries raises, aborting the batch.

        ``max_tokens`` is either ``None`` (instance default for every call), one int applied to
        all of them, or a sequence carrying a per-prompt budget.
        """
        from concurrent.futures import ThreadPoolExecutor

        texts = list(texts)
        if not texts:
            return []
        budgets = self._budgets(texts, max_tokens)
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map over both iterables -> preserves order, re-raises the first failure.
            return list(pool.map(self.complete, texts, budgets))

    def _budgets(self, texts: list, max_tokens):
        """One ``max_tokens`` per prompt: the instance default, one int for all, or a sequence."""
        if max_tokens is None or isinstance(max_tokens, int):
            return [max_tokens] * len(texts)
        budgets = [int(b) for b in max_tokens]
        if len(budgets) != len(texts):
            raise ValueError(f"max_tokens has {len(budgets)} entries but texts has {len(texts)}.")
        return budgets

    def complete_stream(self, texts: list[str], max_tokens=None):
        """Answer a batch of prompts concurrently, yielding ``(index, reply)`` as each lands.

        Same work as :meth:`complete_batch`, but results come back in completion order rather than
        input order, so a caller can persist each reply as it arrives instead of waiting on the
        last one -- see :meth:`prompt_anonymity.caching.TransformCache.apply_streaming`.

        A prompt that still fails after retries raises here, ending the generator and cancelling
        what has not started; replies already yielded are unaffected.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        texts = list(texts)
        if not texts:
            return
        budgets = self._budgets(texts, max_tokens)
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(self.complete, text, budget): index
                       for index, (text, budget) in enumerate(zip(texts, budgets))}
            for future in as_completed(futures):
                yield futures[future], future.result()
