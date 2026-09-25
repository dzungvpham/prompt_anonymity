"""Lean client for Claude Sonnet via Azure AI Foundry -- a second judge provider.

Same shape as OpenRouterChat (complete / complete_batch, same construction signature) so it
drops into any judge call site that already takes an OpenRouterChat-like client, on a separate
account and quota from the shared OpenRouter key.

Credentials are read from the environment (ANTHROPIC_FOUNDRY_API_KEY, ANTHROPIC_FOUNDRY_BASE_URL),
never accepted as constructor arguments with a literal default, so a key can never end up
committed by accident.
"""
from __future__ import annotations

import os
import random
import time

DEFAULT_MODEL = "claude-sonnet-5-2"
DEFAULT_MAX_RETRIES = 5
DEFAULT_TIMEOUT = 60.0


class AnthropicFoundryChat:
    def __init__(self, model: str = DEFAULT_MODEL, system_prompt: str = "", *,
                 max_tokens: int = 1024, max_workers: int = 8,
                 max_retries: int = DEFAULT_MAX_RETRIES, timeout: float = DEFAULT_TIMEOUT):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.max_workers = max_workers
        self.max_retries = max_retries
        self.timeout = timeout
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return self._client
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("ANTHROPIC_FOUNDRY_API_KEY")
        base_url = os.environ.get("ANTHROPIC_FOUNDRY_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError(
                "AnthropicFoundryChat needs ANTHROPIC_FOUNDRY_API_KEY and "
                "ANTHROPIC_FOUNDRY_BASE_URL in the environment (a .env at the repo root, "
                "read via load_dotenv())."
            )
        from anthropic import AnthropicFoundry
        self._client = AnthropicFoundry(api_key=api_key, base_url=base_url)
        return self._client

    def _backoff(self, attempt: int) -> None:
        time.sleep(min(30.0, (2 ** attempt)) + random.uniform(0, 1))

    def complete(self, text: str, max_tokens: int | None = None) -> str:
        if not text.strip():
            return ""
        client = self._ensure_client()
        budget = self.max_tokens if max_tokens is None else int(max_tokens)
        last_err = None
        for attempt in range(self.max_retries):
            try:
                message = client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=[{"role": "user", "content": text}],
                    max_tokens=budget,
                )
                return "".join(
                    block.text for block in message.content if getattr(block, "type", None) == "text"
                )
            except Exception as err:
                status = getattr(err, "status_code", None)
                last_err = err
                if status is not None and 400 <= status < 500 and status != 429:
                    raise RuntimeError(
                        f"Anthropic Foundry rejected the request (HTTP {status}) for model "
                        f"{self.model!r}; not retrying. {err}"
                    ) from err
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
        raise RuntimeError(f"Anthropic Foundry request failed after {self.max_retries} attempts: {last_err}")

    def complete_batch(self, texts: list[str], max_tokens=None) -> list[str]:
        from concurrent.futures import ThreadPoolExecutor
        texts = list(texts)
        if not texts:
            return []
        if max_tokens is None or isinstance(max_tokens, int):
            budgets = [max_tokens] * len(texts)
        else:
            budgets = [int(b) for b in max_tokens]
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self.complete, texts, budgets))