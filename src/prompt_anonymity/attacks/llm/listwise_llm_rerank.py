"""Nearest-neighbor linkage whose whole top-K is reordered by Claude Sonnet 5, with reasons.

The listwise sibling of :mod:`.euclidean_llm_judge`. That attack shortlists the top-K authors by
embedding distance, shows a judge the unknown text plus those K candidates, and asks **which one**
wrote it; the winner is promoted to rank 1 and everything else keeps the distance order. This one
shows the judge exactly the same thing and asks it to **rank all K**, most to least likely, with a
sentence or two per position saying why that candidate belongs there. The full ordering is folded
back into the score matrix by :func:`~prompt_anonymity.attacks.llm.listwise.fold_listwise`, so
top-1 through top-(K-1) all move while top-K stays pinned to the base attack's own number -- see
that module for why the pinned ceiling is what makes the comparison readable.

Two things the single-pick version cannot measure follow from that. The CMC curve between ranks 1
and K becomes an outcome rather than a constant, which is where most of the difference between a
usable attack and an unusable one lives; and because every position carries a justification, the
detail table says *what the model thought it was seeing* -- the raw material for calibrating the
rubric, and the only way to tell a judge that is reading style from one that is reading topic.

The judge
---------
Claude Sonnet 5 over OpenRouter (``anthropic/claude-sonnet-5``) with **reasoning enabled**:
``reasoning.effort``, which OpenRouter maps onto Anthropic's ``output_config.effort`` for Claude 4.6
and newer. The older fixed thinking budget is not available and must not be sent -- ``budget_tokens``
is rejected with a 400 on Sonnet 5, which uses adaptive thinking steered by effort instead.

Requests go through the **Batch API** by default
(:class:`~prompt_anonymity.attacks.llm._openrouter_batch.OpenRouterBatch`), at roughly half the
real-time price with a 24-hour window, resuming rather than resubmitting if the job is interrupted.
Its key is ``SONNET_OR_KEY``, read from a ``.env`` at the repo root.

``batch=False`` takes the synchronous thread-pool client
(:class:`~prompt_anonymity.attacks.llm._openrouter.OpenRouterChat`) instead, under
:data:`SYNC_API_KEY_ENV` (``SONNET_API_KEY``). **It is a full run, not a lesser one**: same model,
same rubric, same ``reasoning.effort``, and the verdicts land in the same cache namespace. What it
trades is money for time -- full real-time price, ~$2/$10 per MTok against the batch tier's
~$1/$5 -- and what it buys is a result in one sitting instead of a submit job, a 24-hour window
and a collect job. Throughput is then just how many requests are in flight, which
``LISTWISE_RERANK_MAX_WORKERS`` sets.

**The discount is carried by the model slug, not by the endpoint.** OpenRouter lists
``anthropic/claude-sonnet-5`` and ``anthropic/claude-sonnet-5:batch`` as two separate models, at
$2/$10 and $1/$5 per MTok; the batch endpoint accepts either, and the plain one simply runs at full
price. ``judge_model`` here is therefore the *logical* model -- the batch client appends the
``:batch`` variant itself when it submits, and prints the slug it actually used. Because the cache is
namespaced by the logical model, a ``batch=False`` smoke run and the full batched run share one
namespace and the smoke run's verdicts are reused rather than re-bought.

**Cost.** At ``snippet_chars=800`` and ``top_k=10`` a row is roughly 2.4k input tokens plus thinking
and ten justifications out, which at batch rates lands near **$0.01-0.015 per unknown document**;
``top_k=5`` is roughly half that. Verdicts are cached by prompt text under ``<cache_dir>/attacks``,
so a re-run -- or a re-swept ambiguity gate -- costs nothing, and a fully-cached run makes no
request and needs no key. Start small and read the reasons before spending a corpus on a rubric
that may turn out to be miscalibrated.

Two knobs are inherited unchanged from the single-pick judge, and for the same reasons:
**candidate shuffling** (on by default -- LLM rankers over-weight the end of a list, and presenting
in distance order would systematically reward the distance metric's worst candidate) and the
**ambiguity gate** (``margin_quantile``, applying the rerank only to rows where the vectors were
close to a coin flip). Every row is still judged, so the cache is complete and re-sweeping the gate
needs no new API calls.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...caching import TransformCache, logic_hash, params_hash
from ...core import AttackData
from ._foundry import DEFAULT_FOUNDRY_MODEL, FoundryChat
from ._openrouter import OpenRouterChat
from ._openrouter_batch import OpenRouterBatch
from .candidates import author_candidates
from .listwise import (
    complete_order,
    detail_table,
    fold_listwise,
    present,
    progress_printer,
    report,
)

#: Authorship-attribution rubric for a forced **full ranking**. The candidate count is filled in per
#: call so the prompt matches ``top_k``. JSON rather than prose because the reply carries two things
#: (an ordering and a reason per position) that have to survive parsing intact; the parser repairs
#: rather than rejects, but a shape the model can hit exactly is worth asking for.
RANK_SYSTEM_PROMPT_TEMPLATE = (
    "You are an authorship-attribution judge. You will be shown one QUERY text and "
    "{n} CANDIDATE texts, labeled 1 through {n}.\n"
    "Task:\n"
    "Rank ALL {n} candidates from most to least likely to have been written by the SAME author as "
    "the QUERY.\n"
    "Base every judgement on writing style -- word choice, sentence structure, punctuation habits, "
    "register, verbosity, quirks of phrasing -- and weight style over topic or subject matter.\n"
    "Rules:\n"
    "- This is a forced ranking: every candidate from 1 to {n} appears exactly once, even if none "
    "is an obvious match. Do NOT refuse, do NOT omit a candidate, and do NOT repeat one.\n"
    "- Give one or two sentences for EVERY position saying why that candidate sits there -- what it "
    "shares with the QUERY, or what rules it out relative to the candidate ranked above it.\n"
    "- Output ONLY a JSON object of exactly this shape, most likely first, and nothing else -- no "
    "preamble, no markdown fences:\n"
    '{{"ranking": [{{"candidate": <int>, "why": "<one or two sentences>"}}, ...]}}'
)

#: Default OpenRouter judge model. Any OpenRouter chat slug works; it is part of the cache key, so a
#: swap re-caches automatically.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"

#: How many of the distance metric's nearest authors the judge reranks per unknown row. The two
#: configurations this attack exists to compare are ``top_k=5`` and ``top_k=10``.
DEFAULT_TOP_K = 5
#: Chars of each conversation shown to the judge, per text -- enough style signal while keeping one
#: query + K candidates + instructions comfortably inside a chat context.
DEFAULT_SNIPPET_CHARS = 800
#: Thinking depth, sent as ``reasoning.effort`` (-> Anthropic's ``output_config.effort``).
DEFAULT_REASONING_EFFORT = "high"
#: Seed for the per-row candidate shuffle -- reproducible, so the judge cache stays stable.
DEFAULT_SEED = 47

#: Output budget: thinking tokens count against it too, so this is deliberately loose. It is a cap,
#: not a spend -- a truncated reply loses the whole JSON object, which costs far more than the
#: headroom does.
MAX_TOKENS_BASE = 2000
#: Additional budget per candidate, covering that position's justification.
MAX_TOKENS_PER_CANDIDATE = 200
#: Output budget on the Foundry provider, thinking included. Far above the OpenRouter figure: with
#: adaptive thinking at high effort, 3,000 tokens is tight enough that a hard row can be cut off
#: mid-JSON and silently fall back to the distance order. It is a cap, not a spend -- only tokens
#: produced are billed. (The OpenRouter budget is left as it was, so that channel's configuration is
#: unchanged.)
FOUNDRY_MAX_TOKENS = 16000

#: Where the unbatched judge is reached. ``openrouter`` is the original channel; ``foundry`` is Claude
#: on Microsoft Foundry through the Anthropic SDK (:mod:`._foundry`), unbatched only.
PROVIDERS = ("openrouter", "foundry")

#: Environment variable holding the key for the **unbatched** path. Deliberately not the batch
#: client's ``SONNET_OR_KEY`` (:mod:`._openrouter_batch`) and not the shared ``OPENROUTER_API_KEY``:
#: this arm's spend is meant to be separable from the defenses' and the other judges'.
SYNC_API_KEY_ENV = "SONNET_API_KEY"

#: Thread-pool width for the unbatched path, where throughput is entirely a function of how many
#: requests are in flight. Read once at import, mirroring
#: :data:`prompt_anonymity.defenses.frame_shift.FRAME_SHIFT_MAX_WORKERS`, so a job script can raise
#: it without a code change. The client retries 429s with jittered backoff, so a value that
#: outruns the account's rate limit costs latency rather than failures.
SYNC_MAX_WORKERS = int(os.environ.get("LISTWISE_RERANK_MAX_WORKERS", "8"))

#: Manual logic version for the judge cache; bump to force a full recompute (see caching.py).
RERANK_VERSION = "1"


def _rank_prompt(query_text: str, candidate_texts: list[str]) -> str:
    """One judge prompt: a QUERY plus its labeled CANDIDATE texts (built by direct string join so
    braces in the conversation text are never mangled)."""
    lines = [f"QUERY:\n{query_text}"]
    for index, candidate in enumerate(candidate_texts, 1):
        lines.append(f"\nCANDIDATE {index}:\n{candidate}")
    n = len(candidate_texts)
    lines.append(
        f"\nRank all {n} candidates from most to least likely to share an author with the QUERY, "
        f"with one or two sentences per position. Reply with the JSON object only."
    )
    return "\n".join(lines)


def _json_object(raw: str):
    """The outermost JSON object in ``raw``, or ``None``.

    Sliced between the first ``{`` and the last ``}`` rather than parsed whole, which handles the
    two ways a model disobeys "no preamble": markdown fences around the object, and a sentence
    before it. A reasoning model's chain of thought arrives in a separate field and never lands
    here, so the content is normally the bare object.
    """
    text = (raw or "").strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        payload = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _parse_ranking(raw: str, k: int) -> tuple[list[int], list[str]]:
    """The judge's ordering as 0-based presented slots, best first, plus a reason per position.

    **Repairs rather than rejects.** Entries that are out of range or repeat a candidate already
    placed are dropped, and anything the judge did not rank is simply absent from the result -- the
    caller completes the permutation in distance order via
    :func:`~prompt_anonymity.attacks.llm.listwise.complete_order`, so a partial answer keeps what
    the judge said and falls back to the embedding below it, and a reply that parses to nothing
    reproduces the base attack exactly.

    That tolerance matters more here than the single-pick judge's "read the last digit" does: a
    K-way permutation is a much larger target than one digit, and discarding a ranking because its
    tenth entry repeated its ninth would throw away nine good judgements.
    """
    payload = _json_object(raw)
    entries = payload.get("ranking") if payload else None
    if not isinstance(entries, list):
        return [], []

    order: list[int] = []
    reasons: list[str] = []
    seen: set[int] = set()
    for entry in entries:
        if isinstance(entry, dict):
            candidate, why = entry.get("candidate"), entry.get("why", "")
        else:  # a bare list of numbers is a shape models fall into; accept it, minus the reasons
            candidate, why = entry, ""
        try:
            slot = int(candidate) - 1
        except (TypeError, ValueError):
            continue
        if not 0 <= slot < k or slot in seen:
            continue
        seen.add(slot)
        order.append(slot)
        # Whitespace collapsed to single spaces: a reason is one CSV field, and a model that answers
        # with an embedded newline would otherwise make the detail table unreadable to every tool
        # that is not a CSV parser. Nothing is lost -- it was asked for one or two sentences.
        reasons.append(" ".join(str(why).split()))
    return order, reasons


class ListwiseLLMRerankAttack:
    """Rerank a nearest-neighbor attack's whole top-K with an LLM authorship judge.

    The judge client is built lazily on first real need, so a fully-cached run makes no API call and
    needs no ``SONNET_OR_KEY``. The model, rubric, effort and presentation params are all part of
    the cache key, so swapping any of them re-caches automatically.

    Parameters
    ----------
    judge_model : str or None
        The judge: an OpenRouter model slug, or a Foundry **deployment name** under
        ``provider="foundry"``. ``None`` (default) picks the provider's default --
        ``anthropic/claude-sonnet-5`` or :data:`~._foundry.DEFAULT_FOUNDRY_MODEL`.
    provider : str
        ``"openrouter"`` (default) or ``"foundry"`` (:mod:`._foundry`: Claude on Microsoft Foundry,
        adaptive thinking, a :data:`FOUNDRY_MAX_TOKENS` budget). Foundry is unbatched only, so it
        requires ``batch=False``. Part of the cache key.
    top_k : int
        How many nearest authors to rerank per unknown row (the headline comparison is 5 vs 10).
    snippet_chars : int
        Chars of each conversation shown to the judge, per text.
    reasoning_effort : str or None
        Thinking depth: ``"low"``/``"medium"``/``"high"``/``"xhigh"``/``"max"``, or ``None`` to send
        no reasoning field at all.
    batch : bool
        ``True`` (default) submits through OpenRouter's Batch API at ~50% of the real-time price,
        with a 24-hour window, under ``SONNET_OR_KEY``. ``False`` uses the synchronous thread-pool
        client under :data:`SYNC_API_KEY_ENV`, at full price and with
        :data:`SYNC_MAX_WORKERS` requests in flight -- for when waiting beats saving. It is **not**
        part of the cache key: it changes the delivery channel, not the prompts, so the two share
        one cache namespace and either one's verdicts are reused by the other.
    wait : bool
        Batch mode only. ``False`` submits, records resume tickets and exits, so the run can be
        collected by re-running the same command later.
    margin_quantile : float
        Ambiguity gate in ``[0, 1]``: apply the rerank only to rows whose top-1/top-2 distance
        margin is at or below this quantile of all rows' margins. ``1.0`` (default) reranks every
        row. Every row is still judged, so re-sweeping this needs no new API calls -- which is why
        it is deliberately absent from the cache key.
    shuffle_candidates : bool
        Show each row's K candidates in a seeded random order to cancel position bias (default
        ``True``).
    seed : int
        Seed for the candidate shuffle (reproducible, so the cache stays stable).
    verbose : bool
        Print per-run diagnostics (pick distributions, parse rates, and what the batch cost).

    Attributes
    ----------
    authors : numpy.ndarray
        Author labels indexing the returned matrix's columns; set by :meth:`attack`.
    detail : pandas.DataFrame
        One row per (unknown document x candidate) with the judge's position and its reason; set by
        :meth:`attack`. See :func:`~prompt_anonymity.attacks.llm.listwise.detail_table`.
    cost_usd : float
        What OpenRouter billed for the requests this run actually made, or ``nan`` when nothing was
        billed through a channel that reports it (a fully-cached run). Both clients report it; the
        synchronous one's figure is a floor, since a reply that arrives without a usage block
        contributes nothing to it. Following
        :mod:`prompt_anonymity.evaluation.utility._deepseek`: a cached row contributes nothing and
        never overwrites what was really paid.
    reasoning_stats : dict
        ``replies``, ``replies_with_reasoning`` and ``reasoning_tokens`` for the replies this run
        actually received -- proof that thinking happened rather than just that it was requested --
        plus ``truncated`` and ``refused`` (Foundry only; ``0`` on OpenRouter, which does not
        report them). Synchronous channel only (the batch client does not count them) and cached
        rows contribute nothing, so a fully-cached run reports zero replies; set by :meth:`attack`.
    """

    def __init__(self, *, judge_model: str | None = None, provider: str = "openrouter",
                 top_k: int = DEFAULT_TOP_K, snippet_chars: int = DEFAULT_SNIPPET_CHARS,
                 reasoning_effort: str | None = DEFAULT_REASONING_EFFORT, batch: bool = True,
                 wait: bool = True, margin_quantile: float = 1.0,
                 shuffle_candidates: bool = True, seed: int = DEFAULT_SEED, verbose: bool = True):
        if not 0.0 <= margin_quantile <= 1.0:
            raise ValueError(f"margin_quantile must be in [0, 1] (got {margin_quantile}).")
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {PROVIDERS} (got {provider!r}).")
        if provider == "foundry" and batch:
            raise ValueError("provider='foundry' is unbatched only; pass batch=False "
                             "(run_rerank.py: --no-batch).")
        if judge_model is None:
            judge_model = DEFAULT_FOUNDRY_MODEL if provider == "foundry" else DEFAULT_JUDGE_MODEL
        self.provider = provider
        self.judge_model = judge_model
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.reasoning_effort = reasoning_effort
        self.batch = batch
        self.wait = wait
        self.margin_quantile = margin_quantile
        self.shuffle_candidates = shuffle_candidates
        self.seed = seed
        self.verbose = verbose
        self._client = None
        self.authors: np.ndarray | None = None
        self.detail: pd.DataFrame | None = None
        self.cost_usd: float = float("nan")
        self.reasoning_stats: dict = {}

    def _max_tokens(self, k: int) -> int:
        if self.provider == "foundry":
            return FOUNDRY_MAX_TOKENS
        return MAX_TOKENS_BASE + MAX_TOKENS_PER_CANDIDATE * k

    def _build_client(self, k: int, ticket_dir) -> None:
        if self._client is None:
            system_prompt = RANK_SYSTEM_PROMPT_TEMPLATE.format(n=k)
            if self.batch:
                self._client = OpenRouterBatch(
                    self.judge_model, system_prompt, max_tokens=self._max_tokens(k),
                    reasoning_effort=self.reasoning_effort, wait=self.wait, ticket_dir=ticket_dir,
                )
            elif self.provider == "foundry":
                # Same rubric and effort, different transport: adaptive thinking at this effort,
                # credentials from FOUNDRY_API_KEY_ENV / FOUNDRY_ENDPOINT_ENV. See _foundry.py.
                self._client = FoundryChat(
                    self.judge_model, system_prompt, max_tokens=self._max_tokens(k),
                    reasoning_effort=self.reasoning_effort, max_workers=SYNC_MAX_WORKERS,
                )
            else:
                # Same model, same rubric, same effort as the batch path -- only the delivery
                # channel differs, which is what makes an unbatched run a measurement rather than
                # a shape check. temperature/top_p are passed as None so the client omits them
                # entirely: Sonnet 5 rejects both with a 400, and a default sent anyway is
                # indistinguishable from this attack asking for them.
                self._client = OpenRouterChat(
                    self.judge_model, system_prompt, max_tokens=self._max_tokens(k),
                    temperature=None, top_p=None, reasoning_effort=self.reasoning_effort,
                    max_workers=SYNC_MAX_WORKERS, api_key_env=SYNC_API_KEY_ENV,
                )

    def _judge(self, prompts: list[str], k: int, ticket_dir) -> list[str]:
        """Every reply at once -- the batch channel, where one submission is the unit of work."""
        self._build_client(k, ticket_dir)
        return self._client.complete_batch(prompts)

    def _judge_stream(self, prompts: list[str], k: int, ticket_dir):
        """``(index, reply)`` as each one lands -- the unbatched channel.

        Same concurrency as :meth:`_judge`; what differs is that the caller learns about a reply
        the moment it arrives and can bank it. At real-time prices a full corpus is hours of paid
        calls, and a job killed at 90% must not have to buy the first 90% again.
        """
        self._build_client(k, ticket_dir)
        yield from self._client.complete_stream(prompts)

    def _cache(self, cache_dir, k: int) -> TransformCache:
        # Keyed by prompt text; namespaced by the judge model + rubric + effort + presentation params
        # so any change that alters the prompts (or the model, or how hard it thinks) re-caches.
        # `batch` and `margin_quantile` are deliberately NOT in the key: neither changes a prompt,
        # so a smoke run's verdicts are reused by the real one and re-sweeping the gate is free. The
        # class source + version guard against silent logic drift (see caching.py).
        return TransformCache(
            Path(cache_dir) / "attacks", "listwise_llm_rerank",
            logic_hash([OpenRouterBatch, ListwiseLLMRerankAttack], version=RERANK_VERSION),
            params_hash({
                "provider": self.provider,
                "judge_model": self.judge_model,
                "judge_system_prompt": RANK_SYSTEM_PROMPT_TEMPLATE.format(n=k),
                "reasoning_effort": self.reasoning_effort,
                "max_tokens": self._max_tokens(k),
                "top_k": self.top_k,
                "snippet_chars": self.snippet_chars,
                "shuffle_candidates": self.shuffle_candidates,
                "seed": self.seed,
            }),
        )

    def attack(self, data: AttackData, *, cache_dir=None) -> pd.DataFrame:
        """Run the listwise-reranked attack on ``data``, returning an ``[n_unknown x n_authors]``
        score matrix (**higher = more likely this author**), with authors in ``self.authors``.

        Parameters
        ----------
        data : AttackData
            The split to attack; must carry ``known_texts`` and ``unknown_texts`` (the judge reads
            raw conversation text, not embeddings). The base ranking uses ``data.metric``.
        cache_dir : str or pathlib.Path or None
            Cache root; verdicts live under ``<cache_dir>/attacks`` and batch resume tickets under
            ``<cache_dir>/attacks/_batches``. ``None`` disables both -- every row hits the API, and
            an interrupted batch is money lost -- so pass one for any run worth paying for.
        """
        if data.known_texts is None or data.unknown_texts is None:
            raise ValueError(
                "ListwiseLLMRerankAttack needs known_texts and unknown_texts on the AttackData "
                "(the judge reads raw conversation text); load the dataset with text."
            )
        known_texts = [str(text) for text in np.asarray(data.known_texts)]
        unknown_texts = [str(text) for text in np.asarray(data.unknown_texts)]

        # Shortlist the most likely AUTHORS, each represented by their own document nearest to this
        # unknown one. See .candidates for why the unit is the author rather than the conversation.
        candidates = author_candidates(
            data.known_embeddings, data.known_labels, data.unknown_embeddings,
            top_k=self.top_k, metric=data.metric,
        )
        self.authors = candidates.authors
        n, k = candidates.author_index.shape
        if k < 2:
            return pd.DataFrame(candidates.scores)  # <2 candidates: nothing to reorder

        layout = present(candidates, seed=self.seed, shuffle=self.shuffle_candidates)
        prompts = [
            _rank_prompt(
                unknown_texts[i][: self.snippet_chars],
                [known_texts[j][: self.snippet_chars] for j in layout.documents[i]],
            )
            for i in range(n)
        ]

        if cache_dir is not None:
            cache = self._cache(cache_dir, k)
            tickets = Path(cache_dir) / "attacks" / "_batches"
            if self.batch:
                # One submission is the unit of work, and the batch client keeps its own resume
                # tickets, so there is nothing finer to bank here.
                replies = cache.apply_batch(prompts, lambda batch: self._judge(batch, k, tickets))
            else:
                # flush_every=1: a verdict is written the moment it arrives. At real-time prices
                # each one is real money, and a preemption a minute later must not re-buy it.
                replies = cache.apply_streaming(
                    prompts, lambda batch: self._judge_stream(batch, k, tickets),
                    flush_every=1,
                    on_progress=progress_printer("judged") if self.verbose else None,
                )
            if self.verbose:
                print(f"  listwise rerank: {cache.hits}/{n} rows served from cache, "
                      f"{cache.misses} judged")
        else:
            replies = self._judge(prompts, k, None)

        # Parse, then complete every ordering in DISTANCE order, so a refusal or an unparseable
        # reply degrades that row to the plain nearest-neighbor ranking rather than to the shuffle.
        orders, reasons, n_parsed = [], [], 0
        for i, reply in enumerate(replies):
            partial, why = _parse_ranking(reply, k)
            if partial:
                n_parsed += 1
            orders.append(complete_order(partial, layout.ranks[i]))
            reasons.append(list(why) + [""] * (k - len(why)))

        # Ambiguity gate: apply the rerank only where the two leading authors were near-tied, which
        # is where a judge can plausibly beat the distance metric. Confident rows keep their own #1.
        gate = np.quantile(candidates.margin, self.margin_quantile)
        applied = candidates.margin <= gate

        boosted = fold_listwise(candidates.scores, layout, orders, apply_mask=applied)
        self.detail = detail_table(candidates, layout, orders, data.unknown_labels,
                                   unknown_ids=data.unknown_ids, reasons=reasons, applied=applied)
        self.cost_usd = getattr(self._client, "total_cost", float("nan"))
        self.reasoning_stats = {
            "replies": getattr(self._client, "n_replies", 0),
            "replies_with_reasoning": getattr(self._client, "n_replies_with_reasoning", 0),
            "reasoning_tokens": getattr(self._client, "total_reasoning_tokens", 0),
            "truncated": getattr(self._client, "n_truncated", 0),
            "refused": getattr(self._client, "n_refused", 0),
        }

        if self.verbose:
            report(orders, layout, n_parsed=n_parsed, n_applied=int(applied.sum()),
                   label="listwise rerank")
            if n_parsed < n:
                print(f"  listwise rerank: {n - n_parsed} rows returned nothing parseable and fell "
                      f"back to the base attack's ordering")
        return pd.DataFrame(boosted)


def listwise_llm_rerank_attack(data: AttackData, *, cache_dir=None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: rerank ``data``'s nearest-neighbor top-K with a default LLM judge.

    Any :class:`ListwiseLLMRerankAttack` constructor argument (``judge_model``, ``top_k``,
    ``reasoning_effort``, ``batch``, ``margin_quantile``, ...) may be passed through ``kwargs``. The
    attack object carries the per-position reasons on its ``detail`` table, which this wrapper
    discards -- construct the class directly when you want them.
    """
    return ListwiseLLMRerankAttack(**kwargs).attack(data, cache_dir=cache_dir)
