"""DeepSeek chat client for the utility judge, over the OpenAI-compatible API.

The utility metric scores whole conversations with an LLM judge -- thousands of independent calls
with no dependency on each other. :class:`DeepSeekJudge` fans them out over a thread pool (the work
is network-bound, so threads scale it nearly linearly despite the GIL) and returns one reply per
input, in input order.

**The provider is reached through the ``openai`` library, not a DeepSeek-specific SDK.** DeepSeek
serves an OpenAI-compatible ``/chat/completions`` endpoint, so ``OpenAI(base_url=..., api_key=...)``
is the supported client -- the library is a transport here, and nothing about it implies an OpenAI
model. Both values come from the gitignored ``.env``: ``DEEPSEEK_BASE_URL`` (the endpoint) and
``DEEPSEEK_API_KEY``, and **both are required** -- the base URL has no default worth guessing, and
the library would otherwise talk to OpenAI's own API, where this key is not a credential.

**This replaced a Microsoft Foundry Claude deployment on 2026-08-11** (`git log --follow` for
``_foundry.py``). The Message Batches API went with it -- already unavailable on Foundry, and not
part of the OpenAI-compatible surface either -- and is not a choice waiting to be revisited.
Anthropic's ``output_config.effort`` knob went too, and that one *did* come back: see
:data:`DEFAULT_REASONING_EFFORT`, added later the same day. In between, ``temperature=0`` stood in
as the reproducibility lever; enabling reasoning ended that, because thinking mode ignores
temperature, so the response cache is what makes a re-run reproducible now. The judging model is
the only thing that differs between the utility axis and the attack judges, which use OpenRouter
(:mod:`prompt_anonymity.attacks.llm._openrouter`).

Retries are the SDK's own (connection errors, 408/409/429 and 5xx, exponential backoff), configured
by ``max_retries`` rather than hand-rolled. A request that still fails raises and aborts the batch
the caller handed over: the caller writes whatever it gets back into a content-addressed cache, and
a cached error is a *permanent* one -- it would never be retried. :mod:`.prompt_judge` bounds the
blast radius by judging in chunks, so an abort loses at most one chunk of successes and a re-run
resumes from the cache.

``openai`` and ``python-dotenv`` are imported lazily so importing this module -- e.g. to reach the
rubric constants or the score parser -- never requires the SDK or a key.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass

#: Environment variables the client reads (from the process env or a ``.env``). Both are required.
DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL_ENV = "DEEPSEEK_BASE_URL"

#: Default judge model, as the configured endpoint names its deployment.
DEFAULT_JUDGE_MODEL = "DeepSeek-V4-Flash"

#: Reasoning budget, sent as the OpenAI-compatible ``reasoning_effort`` field. ``None`` omits it.
#:
#: **How this deployment actually behaves, probed 2026-08-11 (do not re-probe; do not "fix" the
#: call to match DeepSeek's published guide).** The guide
#: (https://api-docs.deepseek.com/guides/thinking_mode/) says to send ``reasoning_effort`` *and*
#: ``extra_body={"thinking": {"type": "enabled"}}``, and that thinking is on by default at effort
#: ``high``. Neither holds on this Azure-hosted deployment:
#:
#: * ``extra_body={"thinking": ...}`` is a hard **400** -- ``unrecognized_request_argument:
#:   thinking``. The field must be sent bare, which works.
#: * **Thinking is OFF here unless the field is sent.** With no ``reasoning_effort`` the model
#:   returns no ``reasoning_content`` at all (measured: 38 output tokens, 0 reasoning characters
#:   over 5 samples). So setting this to ``"low"`` *enables* reasoning rather than reducing it --
#:   it is a ~6x increase in output tokens, which are the dear ones (see :data:`MODEL_RATES`).
#: * ``"none"`` is honoured and turns thinking off (41 tokens, 0 reasoning chars).
#: * **The level is only partly honoured.** Over 5 samples each, mean reasoning was 830 chars at
#:   ``low``, 1,059 at ``high``, 2,108 at ``max`` -- ``max`` separates cleanly, but ``low`` and
#:   ``high`` overlap heavily, and the control settles it: an **invalid** value (``"bogus"``) is
#:   accepted without error and lands at 903 chars, indistinguishable from ``low``. So the value
#:   is not validated, and anything that is not ``"none"`` buys reasoning somewhere in that band.
#:   Treat ``low`` as "reasoning on, modest" rather than as a precise dial.
DEFAULT_REASONING_EFFORT = "low"

#: Sampling controls. **Both are no-ops while reasoning is enabled** -- DeepSeek's guide states
#: thinking mode ignores ``temperature`` and ``top_p`` (accepted for compatibility, no effect) --
#: so 1.0/1.0 is the honest setting: the neutral value that says "not steering the sampler",
#: rather than a 0.0 that reads as determinism the API is not providing. The consequence is real
#: and is the reason this is spelled out: **verdicts are no longer deterministic**, and the
#: response cache, not the temperature, is what makes a re-run reproducible.
DEFAULT_TEMPERATURE = 1.0
DEFAULT_TOP_P = 1.0

#: Thread-pool width. Judging is network-bound, so this is a rate-limit knob rather than a CPU one:
#: raise it if the endpoint's quota allows and the wall-clock matters, lower it if 429s dominate.
DEFAULT_MAX_WORKERS = 8

#: SDK retry budget. Well above the library's default of 2 because a long judge run will meet
#: transient 429s and 5xx, and the alternative to retrying them is aborting a chunk of paid work.
DEFAULT_MAX_RETRIES = 8

#: Per-request timeout (seconds). Generous: a request carries two whole conversations.
DEFAULT_TIMEOUT_SECONDS = 300.0

@dataclass(frozen=True)
class TokenRates:
    """List price in **US dollars per million tokens** for one model.

    ``cached_input`` is the discounted rate for input tokens the provider served from its own
    context cache. It is a separate tier rather than a multiplier because the discount is steep and
    provider-specific -- on the Azure-hosted DeepSeek deployment it is about 15% of the full input
    rate, which is far too large to fold into an approximation.
    """

    input: float
    cached_input: float
    output: float


#: Published list price per model. The DeepSeek deployment is Azure-hosted and priced from its
#: published rate card. **Add a model here rather than guessing**: an invented price produces a
#: plausible dollar figure in the score file that nobody would think to re-check, which is worse
#: than a blank one -- an unpriced model leaves ``judge_cost_usd`` as ``nan``, which reads as
#: "unknown" instead of "free".
MODEL_RATES: dict[str, TokenRates] = {
    "DeepSeek-V4-Flash": TokenRates(input=0.19, cached_input=0.028, output=0.51),
}


def token_rates(model: str) -> TokenRates | None:
    """Published rates for ``model``, or ``None`` when it carries no configured price."""
    return MODEL_RATES.get(model)


@dataclass
class JudgeUsage:
    """Token usage accumulated over the API calls one judge client actually made.

    **Tokens are the record; dollars are a derived estimate.** The counts come back from the API
    and cannot go stale; a price list is hardcoded and eventually will. So everything downstream
    stores the token counts and recomputes cost from them, and any printed figure names the rate it
    used -- or says plainly that no rate is known.

    Note this counts *requests made*, not conversations scored: a verdict served from the cache and
    a conversation the defense left unchanged both cost nothing and appear nowhere here. A run whose
    inputs are all cached ends with ``requests == 0``, which is the correct record of what it spent.
    """

    requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0

    def add(self, usage) -> None:
        """Fold one completion's ``usage`` block in. Absent fields count as zero.

        OpenAI-compatible names: ``prompt_tokens`` / ``completion_tokens``. Cached-prefix hits are
        reported under ``prompt_tokens_details.cached_tokens`` when the provider supports context
        caching; they are **already included** in ``prompt_tokens``, so the field is carried for
        visibility (a high number means the rubric prefix is being reused) and is not added again.
        """
        self.requests += 1
        self.input_tokens += getattr(usage, "prompt_tokens", 0) or 0
        self.output_tokens += getattr(usage, "completion_tokens", 0) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        self.cached_input_tokens += getattr(details, "cached_tokens", 0) or 0

    @property
    def uncached_input_tokens(self) -> int:
        """Input tokens billed at the full rate: total input minus the cached-prefix hits.

        **This subtraction is the whole point.** ``cached_tokens`` is a *subset* of
        ``prompt_tokens``, not an additional bucket, so pricing both at their own rate without
        removing the overlap would bill the cached tokens twice. Confirmed arithmetically on the
        first real run: 2,277 input tokens against 1,792 cached over two requests is ~1,138 per
        request, which is the ~900-token rubric plus a short conversation -- consistent only with
        the cached figure being contained in the total.
        """
        return max(0, self.input_tokens - self.cached_input_tokens)

    def estimated_cost(self, rates: TokenRates | None) -> float | None:
        """Estimated dollars for this usage, or ``None`` when no rate is known."""
        if rates is None:
            return None
        return (
            self.uncached_input_tokens * rates.input
            + self.cached_input_tokens * rates.cached_input
            + self.output_tokens * rates.output
        ) / 1e6

    def summary(self, rates: TokenRates | None = None) -> str:
        """One line: requests, tokens, and either a cost estimate or a note that none is known."""
        if not self.requests:
            return "API usage: 0 requests (everything served from cache or unchanged) -- $0.00"
        line = (f"API usage: {self.requests:,} requests  "
                f"in={self.input_tokens:,} tok  out={self.output_tokens:,} tok")
        if self.cached_input_tokens:
            share = self.cached_input_tokens / self.input_tokens if self.input_tokens else 0.0
            line += (f"  ({self.cached_input_tokens:,} of input, {share:.0%}, served from the "
                     "provider's cache)")
        cost = self.estimated_cost(rates)
        if cost is None:
            return line + ("  cost=unknown (this model has no entry in MODEL_RATES; add one to "
                           "price it -- the token counts above are exact either way)")
        return line + (f"  est. cost=${cost:.4f} (at ${rates.input}/${rates.cached_input}/"
                       f"${rates.output} per Mtok in/cached/out)")


def _reply_text(choice) -> str:
    """The assistant text of one completion, or a diagnostic marker when there is none.

    Two non-answers become markers rather than an empty string: a **content-filter** stop and a
    generation **truncated** by the token budget. Both would otherwise reach the score parser as
    ``""`` and be reported as an unparseable judge reply, which hides why.
    """
    content = (choice.message.content or "") if choice.message is not None else ""
    if choice.finish_reason == "content_filter":
        return "REFUSAL: the provider's content filter blocked this conversation"
    if not content.strip() and choice.finish_reason == "length":
        return "TRUNCATED: max_tokens reached before any answer text was produced"
    return content


class DeepSeekJudge:
    """Score a list of prompts with one chat model: one reply per input, in input order.

    Built lazily by the metric on first real need, so a fully-cached scoring run constructs no
    client and needs no credentials.

    Parameters
    ----------
    model : str
        Model / deployment name as the configured endpoint serves it.
    system_prompt : str
        System message prepended to every request -- the judge's rubric.
    max_tokens : int
        Output budget per request, **shared between the reasoning and the answer**. The verdict
        itself is one line of JSON, but with ``reasoning_effort`` set the chain of thought is spent
        from the same budget; too tight a budget shows up as a truncated (and unparseable) reply
        rather than an error, which is what :func:`_reply_text`'s ``TRUNCATED`` marker names.
        Measured headroom: the dearest setting (``max``) spent ~460 tokens on this workload, so the
        8192 default is ample -- it was sized for reasoning that, before 2026-08-11, was not
        actually happening.
    temperature, top_p : float
        Sampling controls. Both are ignored by the API while reasoning is on -- see
        :data:`DEFAULT_TEMPERATURE`.
    reasoning_effort : str or None
        ``"low"`` / ``"high"`` / ``"max"`` to think, ``"none"`` not to, ``None`` to omit the field
        (which on this deployment also means not thinking). See :data:`DEFAULT_REASONING_EFFORT`
        for what is actually honoured here, which is not what the vendor guide says.
    max_workers, max_retries, timeout : int / int / float
        Concurrency, SDK retry budget, and per-request timeout.
    """

    def __init__(self, model: str = DEFAULT_JUDGE_MODEL, system_prompt: str = "", *,
                 max_tokens: int = 8192, temperature: float = DEFAULT_TEMPERATURE,
                 top_p: float = DEFAULT_TOP_P,
                 reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
                 max_workers: int = DEFAULT_MAX_WORKERS,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 timeout: float = DEFAULT_TIMEOUT_SECONDS,
                 api_key_env: str = DEEPSEEK_API_KEY_ENV,
                 base_url_env: str = DEEPSEEK_BASE_URL_ENV):
        from dotenv import load_dotenv
        from openai import OpenAI

        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.reasoning_effort = reasoning_effort
        self.max_workers = max_workers

        load_dotenv()  # walks up from cwd, so credentials can live in the repo root or a subdir
        api_key = os.environ.get(api_key_env)
        base_url = os.environ.get(base_url_env)
        if not api_key:
            raise RuntimeError(
                f"{api_key_env} not set. Add it to a .env file so the utility metric can call the "
                "judge."
            )
        if not base_url:
            raise RuntimeError(
                f"{base_url_env} not set. Without it the OpenAI client would send this key to "
                "OpenAI's own API, where it is not a credential. Set the DeepSeek endpoint in .env."
            )
        self._client = OpenAI(base_url=base_url, api_key=api_key,
                              max_retries=max_retries, timeout=timeout)
        # Running tally of what this client has actually spent. `complete` runs on a thread pool,
        # so the tally is mutated under a lock; read `usage` after the pool has drained.
        self.usage = JudgeUsage()
        # The same spend, broken out per request and keyed by the prompt that caused it, so a
        # caller can attribute a dollar figure to the individual conversation it judged rather
        # than only to the run as a whole. Keys are references to strings the caller already
        # holds, so this costs a dict entry per request and no copy of the text.
        self.request_usage: dict[str, JudgeUsage] = {}
        self._usage_lock = threading.Lock()
        print(f"Utility: judge using model '{model}' "
              f"(reasoning_effort={reasoning_effort}, temperature={temperature}, top_p={top_p})")

    def complete(self, text: str) -> str:
        """Return the judge's reply to ``text`` under the fixed rubric.

        A blank input has nothing to judge, so it short-circuits to ``""`` without a request.
        """
        if not text.strip():
            return ""
        # `reasoning_effort` is sent bare and only when set: `extra_body={"thinking": ...}`, which
        # the vendor guide pairs it with, is rejected outright here (see DEFAULT_REASONING_EFFORT).
        extra = {"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            **extra,
        )
        # Record the spend before anything can go wrong with the reply: the tokens were billed
        # whether or not the verdict turns out to be parseable, filtered, or truncated.
        with self._usage_lock:
            if completion.usage is not None:
                self.usage.add(completion.usage)
                per_request = JudgeUsage()
                per_request.add(completion.usage)
                self.request_usage[text] = per_request
        if not completion.choices:
            return "EMPTY: the provider returned no choices for this conversation"
        return _reply_text(completion.choices[0])

    def complete_batch(self, texts: list[str]) -> list[str]:
        """Judge a batch of prompts concurrently, preserving input order.

        Network-bound, so a thread pool gives near-linear speedup despite the GIL. A prompt that
        still fails after the SDK's retries raises, aborting the batch -- see the module docstring
        for why a failure is not turned into a placeholder reply.
        """
        from concurrent.futures import ThreadPoolExecutor

        texts = list(texts)
        if not texts:
            return []
        workers = max(1, min(self.max_workers, len(texts)))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            # map preserves order and re-raises the first failure.
            return list(pool.map(self.complete, texts))

    def request_cost(self, text: str) -> float:
        """Dollars **this run** spent judging ``text``, at this model's configured rates.

        Three outcomes, deliberately distinguished: a real figure when the request was made here;
        ``0.0`` when it was not (the verdict came from the cache, or the conversation was never
        judged), because that is what this run spent on it; and ``nan`` when the request *was* made
        but the model carries no price in :data:`MODEL_RATES` -- "unknown", which must not be
        recorded as "free". Cost paid by an *earlier* run is not zero and is not lost: it is
        already in that conversation's row of the score file, which a re-run preserves.
        """
        usage = self.request_usage.get(text)
        if usage is None:
            return 0.0
        cost = usage.estimated_cost(token_rates(self.model))
        return float("nan") if cost is None else cost
