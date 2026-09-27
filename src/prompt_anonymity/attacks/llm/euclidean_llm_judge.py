"""Nearest-neighbor linkage reranked by an LLM-as-a-judge.

Plain :class:`~prompt_anonymity.attacks.NearestNeighbor` usually lands the true author
somewhere in an unknown conversation's top-K nearest known rows, but is weaker at picking
which of those K is actually correct. This attack keeps the embedding distance's top-K
membership but reranks within it: for each unknown conversation it shows an LLM judge the
unknown text plus its top-K known candidates and asks which candidate was written by the same
author, from writing style alone. The judge's pick is promoted to rank 1.

Only the within-top-K order changes -- top-K membership is left exactly as the distance metric
found it, so this isolates whatever extra signal the LLM adds on top of the embedding distance
at rank 1; only top-1 can move.

The judge runs on OpenRouter through this package's shared
:class:`~prompt_anonymity.attacks.llm._openrouter.OpenRouterChat` client, and its verdicts are
cached by prompt text and presentation params, so a re-swept ambiguity gate costs no API calls.

Two knobs guard against a naive rerank underperforming the distance baseline:

* **Candidate shuffling** (``shuffle_candidates``, on by default). LLM judges have a
  position/recency bias; feeding candidates in distance-rank order would let that bias
  systematically promote the metric's worst top-K candidate. Each row's K candidates are shown
  in a seeded random order and the judge's positional pick is mapped back to the real known row.
* **Ambiguity gate** (``margin_quantile``). The rerank is applied only on rows whose top-1
  vs. top-2 distance margin is at or below this quantile -- the closest calls, where the
  distance metric was least sure. Confident rows keep their own #1. Every row is still judged
  (so the cache is complete); only the application is gated, so retuning the quantile needs no
  new API calls.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ...caching import TransformCache, logic_hash, params_hash
from ...core import AttackData
from ._openrouter import OpenRouterChat
from .candidates import author_candidates

#: Authorship-attribution judge rubric. A forced K-way choice returning a single digit; the
#: candidate count is filled in per call so the prompt matches ``top_k``.
JUDGE_SYSTEM_PROMPT_TEMPLATE = (
    "You are an authorship-attribution judge. You will be shown one QUERY text and "
    "{n} CANDIDATE texts, labeled 1 through {n}.\n"
    "Task:\n"
    "Choose the ONE candidate most likely written by the SAME author as the QUERY.\n"
    "Base the decision on writing style -- word choice, sentence structure, punctuation "
    "habits, register, verbosity, quirks of phrasing -- and weight style over topic or "
    "subject matter.\n"
    "Rules:\n"
    "- This is a forced choice: you MUST pick exactly one candidate, the single closest "
    "stylistic match. Even if none is an obvious match, pick the best of the {n}. Do NOT "
    "refuse and do NOT answer 0.\n"
    "- Output ONLY the single digit (1-{n}) of your choice and nothing else -- no words, no "
    "punctuation, no explanation."
)

#: Default OpenRouter judge model. Any OpenRouter chat model id works; it is part of the cache
#: key, so a swap re-caches automatically.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"

#: How many of the distance metric's nearest candidates the judge reranks per unknown row.
DEFAULT_TOP_K = 5
#: Chars of each conversation shown to the judge, per text.
DEFAULT_SNIPPET_CHARS = 800
#: Output budget: the judge only ever needs to emit one digit.
DEFAULT_JUDGE_MAX_TOKENS = 8
#: Seed for the per-row candidate shuffle -- reproducible, so the judge cache stays stable.
DEFAULT_SEED = 47

#: Manual logic version for the judge cache; bump to force a full recompute (see caching.py).
JUDGE_VERSION = "1"

_DIGIT_RE = re.compile(r"[0-9]")


def _judge_prompt(query_text: str, candidate_texts: list[str]) -> str:
    """One judge prompt: a QUERY plus its labeled CANDIDATE texts (built by direct string
    join so braces in the text are never mangled)."""
    lines = [f"QUERY:\n{query_text}"]
    for i, cand in enumerate(candidate_texts, 1):
        lines.append(f"\nCANDIDATE {i}:\n{cand}")
    n = len(candidate_texts)
    lines.append(
        f"\nWhich candidate is the closest stylistic match to the QUERY, i.e. most likely "
        f"written by the same author? You MUST pick one. Answer with a single digit 1-{n}."
    )
    return "\n".join(lines)


def _parse_choice(raw: str) -> int:
    """The LAST digit the judge emitted, or 0 if it emitted none. The judge is asked for a
    forced choice, so 0 (or an out-of-range digit) means it disobeyed; callers force such a
    row to the NEAREST candidate (distance rank 0) rather than skipping the rerank, so every
    gated row commits (see :meth:`EuclideanLLMJudgeAttack.attack`).

    Last (not first) digit: with a bare ``"3"`` the two coincide, but if a model leaks a
    preamble before the answer (e.g. ``"answer: 2"``), the decision digit is the trailing one.
    """
    matches = _DIGIT_RE.findall(raw or "")
    return int(matches[-1]) if matches else 0


class EuclideanLLMJudgeAttack:
    """Rerank a nearest-neighbor attack's top-K with an LLM authorship judge.

    The judge client is built lazily on first real need, so a fully-cached run makes no API
    call and needs no ``OPENROUTER_API_KEY``. The active model, system prompt, and
    presentation params are part of the cache key, so swapping any of them re-caches
    automatically.

    Parameters
    ----------
    judge_model : str
        OpenRouter model id for the judge (default ``anthropic/claude-sonnet-5``).
    top_k : int
        How many nearest candidates to rerank per unknown row.
    snippet_chars : int
        Chars of each conversation shown to the judge, per text.
    margin_quantile : float
        Ambiguity gate in ``[0, 1]``: apply the rerank only on rows whose top-1/top-2 distance
        margin is at or below this quantile of all rows' margins. ``1.0`` reranks every row.
    shuffle_candidates : bool
        Show each row's K candidates in a seeded random order to cancel the judge's position
        bias (default ``True``).
    seed : int
        Seed for the candidate shuffle (reproducible, so the cache stays stable).
    judge_max_tokens : int
        Output budget for the judge (it only needs one digit).
    verbose : bool
        Print per-run diagnostics (pick distributions and how many rows were reranked).
    """

    def __init__(self, *, judge_model: str = DEFAULT_JUDGE_MODEL, top_k: int = DEFAULT_TOP_K,
                 snippet_chars: int = DEFAULT_SNIPPET_CHARS,
                 margin_quantile: float = 1.0, shuffle_candidates: bool = True,
                 seed: int = DEFAULT_SEED, judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
                 verbose: bool = True):
        if not 0.0 <= margin_quantile <= 1.0:
            raise ValueError(f"margin_quantile must be in [0, 1] (got {margin_quantile}).")
        self.judge_model = judge_model
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.margin_quantile = margin_quantile
        self.shuffle_candidates = shuffle_candidates
        self.seed = seed
        self.judge_max_tokens = judge_max_tokens
        self.verbose = verbose
        self._client: OpenRouterChat | None = None

    def _judge(self, prompts: list[str], n_candidates: int) -> list[str]:
        if self._client is None:
            self._client = OpenRouterChat(
                self.judge_model,
                JUDGE_SYSTEM_PROMPT_TEMPLATE.format(n=n_candidates),
                max_tokens=self.judge_max_tokens,
            )
        return self._client.complete_batch(prompts)

    def _cache(self, cache_dir, n_candidates: int) -> TransformCache:
        # margin_quantile is deliberately not in the key: it only gates which cached verdicts
        # get applied, never the prompts, so re-sweeping it is free.
        return TransformCache(
            Path(cache_dir) / "attacks", "euclidean_llm_judge",
            logic_hash([OpenRouterChat, EuclideanLLMJudgeAttack], version=JUDGE_VERSION),
            params_hash({
                "judge_model": self.judge_model,
                "judge_system_prompt": JUDGE_SYSTEM_PROMPT_TEMPLATE.format(n=n_candidates),
                "top_k": self.top_k,
                "snippet_chars": self.snippet_chars,
                "shuffle_candidates": self.shuffle_candidates,
                "seed": self.seed,
            }),
        )

    def attack(self, data: AttackData, *, cache_dir=None) -> pd.DataFrame:
        """Run the reranked attack on ``data``, returning an ``[n_unknown x n_authors]`` score
        matrix (**higher = more likely this author**), with authors in ``self.authors``.

        Parameters
        ----------
        data : AttackData
            The split to attack; must carry ``known_texts`` and ``unknown_texts`` (the judge
            reads raw conversation text, not just embeddings) and ``known_labels`` (the
            shortlist is over authors). The base ranking uses ``data.metric``.
        cache_dir : str or pathlib.Path or None
            Cache root; verdicts live under ``<cache_dir>/attacks``. ``None`` disables caching
            (every judged row hits the API), matching the other attacks, which do not cache.
        """
        if data.known_texts is None or data.unknown_texts is None:
            raise ValueError(
                "EuclideanLLMJudgeAttack needs known_texts and unknown_texts on the AttackData "
                "(the judge reads raw conversation text); load the dataset with text."
            )
        known_texts = [str(t) for t in np.asarray(data.known_texts)]
        unknown_texts = [str(t) for t in np.asarray(data.unknown_texts)]

        # Shortlist the most likely authors, each represented by their own document nearest to
        # this unknown one (see attacks.llm.candidates for why the unit is the author).
        candidates = author_candidates(
            data.known_embeddings, data.known_labels, data.unknown_embeddings,
            top_k=self.top_k, metric=data.metric,
        )
        self.authors = candidates.authors
        scores = candidates.scores
        n, k = candidates.author_index.shape
        if k < 2:
            return pd.DataFrame(scores)  # <2 candidates: nothing to rerank

        top_idx = candidates.document_index    # (n, k) known-document rows, slot 0 = best author
        author_idx = candidates.author_index   # (n, k) author columns, slot 0 = best author
        row_max = scores.max(axis=1)           # == the best author's score

        # Per-row best vs second-best author margin (small = ambiguous). gate_thresh is that
        # margin's `margin_quantile` quantile, so ~that fraction of the closest calls pass.
        margins = candidates.margin
        gate_thresh = np.quantile(margins, self.margin_quantile)

        # Presentation order of each row's K candidates. Shuffling decouples the slot a candidate
        # is shown in from its distance rank, cancelling the judge's position bias. present[i] =
        # the real known indices in the order the judge sees them; perms[i] = the corresponding
        # distance ranks (0 = nearest).
        rng = np.random.default_rng(self.seed)
        if self.shuffle_candidates:
            perms = [rng.permutation(k) for _ in range(n)]
        else:
            perms = [np.arange(k) for _ in range(n)]
        present = [top_idx[i][perms[i]] for i in range(n)]            # documents to show
        present_authors = [author_idx[i][perms[i]] for i in range(n)]  # their authors

        prompts = [
            _judge_prompt(
                unknown_texts[i][: self.snippet_chars],
                [known_texts[j][: self.snippet_chars] for j in present[i]],
            )
            for i in range(n)
        ]
        if cache_dir is not None:
            cache = self._cache(cache_dir, k)
            raw_choices = cache.apply_batch(prompts, lambda ps: self._judge(ps, k))
        else:
            raw_choices = self._judge(prompts, k)

        choices = [_parse_choice(raw) for raw in raw_choices]

        # Promote each gated pick to rank 1 by setting its distance strictly below the row's
        # current minimum. Every other entry -- including which rows are in the top-K -- is
        # untouched, so top-K membership (and top-K/top-2K accuracy) is unchanged and only top-1
        # can move relative to the base attack.
        boosted = scores.copy()
        applied = 0
        forced = 0
        for i, choice in enumerate(choices):
            if margins[i] > gate_thresh:
                continue  # confident row: keep the base attack's own #1 author
            if 1 <= choice <= k:
                boosted[i, present_authors[i][choice - 1]] = row_max[i] + 1.0
                applied += 1
            else:
                # A refusal / invalid digit commits to the best candidate (already this row's
                # #1) instead of skipping the rerank, so the row never worsens vs. the baseline.
                boosted[i, author_idx[i, 0]] = row_max[i] + 1.0
                applied += 1
                forced += 1

        if self.verbose:
            pos_dist = {p: choices.count(p) for p in range(k + 1)}
            rank_picks = [int(perms[i][c - 1]) for i, c in enumerate(choices) if 1 <= c <= k]
            rank_dist = {r: rank_picks.count(r) for r in range(k)}
            print(f"  LLM judge: positional picks {pos_dist} (0=refused; want ~uniform if unbiased)")
            print(f"  LLM judge: author-rank of picks {rank_dist} (0=best; want mass on 0-1)")
            print(f"  LLM judge: rerank applied to {applied}/{n} rows "
                  f"(margin_quantile={self.margin_quantile}, gate<= {gate_thresh:.4g}); "
                  f"{forced} refusals forced to nearest")

        return pd.DataFrame(boosted)


def euclidean_llm_judge_attack(data: AttackData, *, cache_dir=None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: rerank ``data``'s nearest-neighbor top-K with a default judge.

    Mirrors :func:`prompt_anonymity.evaluation.utility.conversation_utility`. Any
    :class:`EuclideanLLMJudgeAttack` constructor argument (``judge_model``, ``top_k``,
    ``snippet_chars``, ``margin_quantile``, ``shuffle_candidates``, ``seed``, ...) may be
    passed through ``kwargs``.
    """
    return EuclideanLLMJudgeAttack(**kwargs).attack(data, cache_dir=cache_dir)
