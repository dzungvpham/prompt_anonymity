"""Bradley-Terry tournament rerank of nearest-neighbor linkage (LSOD Algorithm 1).

Port of *Algorithm 1: LLM-based confidence sorting* from "Large-scale online deanonymization
with LLMs" (arXiv 2602.16800), applied per unknown conversation. Where
:class:`~prompt_anonymity.attacks.EuclideanLLMJudgeAttack` asks the judge ONE K-way forced
choice per unknown row, this ranks each row's top-K nearest known candidates with a
Bradley-Terry / Swiss-system TOURNAMENT of *pairwise* comparisons: candidates carry online
Elo-style ratings, each round pairs similarly-rated candidates, the judge picks the more
plausible same-author match, and the ratings update. After ``rounds`` rounds the candidates
are reranked by rating.

Like the K-way judge, only the WITHIN-top-K order changes -- top-K membership (and hence
top-K / top-2K accuracy) is left exactly as the distance metric found it -- so at ``top_k=5``
top-5 and top-10 are identical to the underlying nearest-neighbor attack and only top-1 can
move. This makes it directly comparable to :class:`EuclideanLLMJudgeAttack`; the difference is
how the within-top-K order is decided (a rating tournament vs. a single forced choice).

The judge runs on OpenRouter (default ``anthropic/claude-sonnet-5``, through the same
:class:`~prompt_anonymity.fidelity._openrouter.OpenRouterChat` client) and its one-digit
pairwise verdicts are cached with :class:`~prompt_anonymity.caching.TransformCache` under
``<cache_dir>/attacks``, keyed by the prompt text -- so a Swiss *rematch* of the same pair
(same presentation) is free on re-encounter, and a fully-cached run makes no API call.

Rounds are SEQUENTIAL (round *r*'s pairings depend on round *r-1*'s ratings), but within a
round every active row contributes its comparisons to ONE batched judge call, so the judge
runs in large batches and each round is cached and crash-safe.

Ambiguity gate (``margin_quantile``): only rows whose top-1/top-2 distance margin is at or
below the quantile are tournamented; confident rows keep the distance metric's order. Unlike
the K-way judge's gate (which judges every row and gates only the boost *application*, so
re-sweeping is free), this gates *execution* -- it skips the API calls for confident rows --
so a lower quantile is cheaper but re-widening it later needs fresh comparisons.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ...caching import TransformCache, logic_hash, params_hash
from ...core import AttackData
from ...fidelity._openrouter import OpenRouterChat
from .euclidean_llm_judge import _parse_choice
from .candidates import author_candidates

#: Pairwise authorship-attribution judge rubric: a 1-vs-2 forced choice returning one digit.
PAIRWISE_JUDGE_SYSTEM_PROMPT = (
    "You are an authorship-attribution judge. You will be shown one QUERY text and two "
    "CANDIDATE texts, labeled 1 and 2.\n"
    "Task:\n"
    "Choose the ONE candidate more likely written by the SAME author as the QUERY.\n"
    "Base the decision on writing style -- word choice, sentence structure, punctuation "
    "habits, register, verbosity, quirks of phrasing -- and weight style over topic or "
    "subject matter.\n"
    "Rules:\n"
    "- This is a forced choice: you MUST pick exactly one candidate, the closer stylistic "
    "match. Even if neither is an obvious match, pick the better of the two. Do NOT refuse "
    "and do NOT answer 0.\n"
    "- Output ONLY the single digit (1 or 2) of your choice and nothing else -- no words, no "
    "punctuation, no explanation."
)

#: Default OpenRouter judge model -- Claude Sonnet. Overridable; part of the cache key.
DEFAULT_JUDGE_MODEL = "anthropic/claude-sonnet-5"

#: How many nearest candidates enter each unknown row's tournament.
DEFAULT_TOP_K = 5
#: N in Algorithm 1: Swiss rounds. With K=5 each round plays 2 matches (+1 bye), so a candidate
#: plays up to N matches; 4 settles the top-1 without excess API calls.
DEFAULT_ROUNDS = 4
#: Online Bradley-Terry (Elo-style) step, in logistic units: ``r += elo_k * (S - E)``,
#: ``E = sigmoid(r_a - r_b)``. Larger = ratings move faster (decisive, noisier).
DEFAULT_ELO_K = 0.4
#: Chars of each conversation shown to the judge, per text.
DEFAULT_SNIPPET_CHARS = 800
#: Output budget: the judge only needs one digit (1 or 2).
DEFAULT_JUDGE_MAX_TOKENS = 8
#: Seed for the presentation-order flips -- reproducible, so the pairwise cache stays stable.
DEFAULT_SEED = 47

#: Manual logic version for the pairwise-judge cache; bump to force a full recompute.
JUDGE_VERSION = "1"


def _pairwise_judge_prompt(query_text: str, cand_a_text: str, cand_b_text: str) -> str:
    """One pairwise judge prompt: a QUERY plus exactly two labeled CANDIDATE texts (direct
    string join so braces in the text are never mangled)."""
    return (
        f"QUERY:\n{query_text}\n"
        f"\nCANDIDATE 1:\n{cand_a_text}\n"
        f"\nCANDIDATE 2:\n{cand_b_text}\n"
        "\nWhich candidate is the closer stylistic match to the QUERY, i.e. more likely "
        "written by the same author? You MUST pick one. Answer with a single digit, 1 or 2."
    )


class BradleyTerryTournamentAttack:
    """Rerank a nearest-neighbor attack's top-K with a Bradley-Terry pairwise-judge tournament.

    The judge client is built lazily on first real need, so a fully-cached run makes no API
    call and needs no ``OPENROUTER_API_KEY``. The active model, prompt, and presentation params
    are part of the cache key, so swapping any of them re-caches automatically.

    Parameters
    ----------
    judge_model : str
        OpenRouter model id for the pairwise judge (default ``anthropic/claude-sonnet-5``).
    top_k : int
        How many nearest candidates enter each row's tournament.
    rounds : int
        Swiss rounds (N in Algorithm 1).
    elo_k : float
        Online Bradley-Terry step size in logistic units.
    snippet_chars : int
        Chars of each conversation shown to the judge, per text.
    margin_quantile : float
        Ambiguity gate in ``[0, 1]``: tournament only rows whose top-1/top-2 distance margin
        is at or below this quantile of all rows' margins. ``1.0`` tournaments every row.
    seed : int
        Seed for the 1-vs-2 presentation flips (reproducible, so the cache stays stable).
    judge_max_tokens : int
        Output budget for the judge (one digit).
    verbose : bool
        Print per-run diagnostics (rows tournamented, comparisons, pick balance, top-1 moves).
    """

    def __init__(self, *, judge_model: str = DEFAULT_JUDGE_MODEL, top_k: int = DEFAULT_TOP_K,
                 rounds: int = DEFAULT_ROUNDS, elo_k: float = DEFAULT_ELO_K,
                 snippet_chars: int = DEFAULT_SNIPPET_CHARS, margin_quantile: float = 1.0,
                 seed: int = DEFAULT_SEED, judge_max_tokens: int = DEFAULT_JUDGE_MAX_TOKENS,
                 verbose: bool = True):
        if not 0.0 <= margin_quantile <= 1.0:
            raise ValueError(f"margin_quantile must be in [0, 1] (got {margin_quantile}).")
        self.judge_model = judge_model
        self.top_k = top_k
        self.rounds = rounds
        self.elo_k = elo_k
        self.snippet_chars = snippet_chars
        self.margin_quantile = margin_quantile
        self.seed = seed
        self.judge_max_tokens = judge_max_tokens
        self.verbose = verbose
        self._client: OpenRouterChat | None = None

    def _judge(self, prompts: list[str]) -> list[str]:
        if self._client is None:
            self._client = OpenRouterChat(
                self.judge_model, PAIRWISE_JUDGE_SYSTEM_PROMPT, max_tokens=self.judge_max_tokens
            )
        return self._client.complete_batch(prompts)

    def _cache(self, cache_dir) -> TransformCache:
        # Keyed by prompt text; namespaced by the judge model + rubric + presentation params so a
        # change that alters the prompts (or the model) re-caches. elo_k / rounds are NOT in the
        # key: they only steer which pairs meet, and a given pair's prompt text is identical
        # regardless, so a Swiss rematch hits the cache across settings.
        return TransformCache(
            Path(cache_dir) / "attacks", "bt_tournament",
            logic_hash([OpenRouterChat, BradleyTerryTournamentAttack], version=JUDGE_VERSION),
            params_hash({
                "judge_model": self.judge_model,
                "judge_system_prompt": PAIRWISE_JUDGE_SYSTEM_PROMPT,
                "top_k": self.top_k,
                "snippet_chars": self.snippet_chars,
                "seed": self.seed,
            }),
        )

    def attack(self, data: AttackData, *, cache_dir=None) -> pd.DataFrame:
        """Run the tournament rerank on ``data``, returning an ``[n_unknown x n_known]`` distance
        matrix (smaller = more similar), aligned to ``data`` by position like every attack.

        Parameters
        ----------
        data : AttackData
            The split to attack; must carry ``known_texts`` and ``unknown_texts``. The base
            ranking uses ``data.metric``.
        cache_dir : str or pathlib.Path or None
            Cache root; verdicts live under ``<cache_dir>/attacks``. ``None`` disables caching.
        """
        if data.known_texts is None or data.unknown_texts is None:
            raise ValueError(
                "BradleyTerryTournamentAttack needs known_texts and unknown_texts on the "
                "AttackData (the judge reads raw conversation text); load the dataset with text."
            )
        known_texts = [str(t) for t in np.asarray(data.known_texts)]
        unknown_texts = [str(t) for t in np.asarray(data.unknown_texts)]

        # Shortlist the most likely AUTHORS, each represented by their own document nearest to
        # this unknown one. See prompt_anonymity.attacks.llm.candidates for why the unit is the
        # author rather than the conversation.
        candidates = author_candidates(
            data.known_embeddings, data.known_labels, data.unknown_embeddings,
            top_k=self.top_k, metric=data.metric,
        )
        self.authors = candidates.authors
        scores = candidates.scores
        n, k = candidates.author_index.shape
        if k < 2:
            return pd.DataFrame(scores)  # need >=2 candidates to hold a match

        top_idx = candidates.document_index    # (n, k) known-document rows, slot 0 = best author
        author_idx = candidates.author_index   # (n, k) author columns, slot 0 = best author
        row_max = scores.max(axis=1)           # == the best author's score

        # Ambiguity gate: tournament only the closest calls (small best/second-best AUTHOR margin).
        margins = candidates.margin
        gate_thresh = np.quantile(margins, self.margin_quantile)
        active_rows = np.where(margins <= gate_thresh)[0]

        # BT ratings, (n, k), in logistic units. Seed a TINY author-rank prior (best slot highest)
        # so round-1 Swiss pairing is deterministic (best vs 2nd-best, 3rd vs 4th, ...) while
        # being small enough that a single upset (step ~elo_k) flips it -- the tournament, not
        # the prior, decides.
        ratings = np.tile(((k - 1 - np.arange(k)) * 1e-3).astype(float), (n, 1))

        rng = np.random.default_rng(self.seed)  # seeds 1-vs-2 presentation flips; reproducible
        cache = self._cache(cache_dir) if cache_dir is not None else None
        pos_picks = {0: 0, 1: 0, 2: 0}          # judge's presented-slot pick balance
        total_cmp = 0

        for _ in range(self.rounds):
            # Build this round's Swiss pairings across every active row. meta[t] =
            # (row, slot_shown_as_1, slot_shown_as_2) for prompts[t].
            prompts, meta = [], []
            for i in active_rows:
                slot_order = np.argsort(-ratings[i], kind="stable")  # best-rated first; ties keep dist order
                for p in range(0, k - 1, 2):                         # adjacent pairs; odd last slot = bye
                    a, b = int(slot_order[p]), int(slot_order[p + 1])
                    # Randomize which candidate is shown as 1 vs 2 to cancel the judge's position
                    # bias (same rationale as EuclideanLLMJudgeAttack's candidate shuffle).
                    first, second = (b, a) if rng.integers(2) else (a, b)
                    prompts.append(_pairwise_judge_prompt(
                        unknown_texts[int(i)][: self.snippet_chars],
                        known_texts[top_idx[i, first]][: self.snippet_chars],
                        known_texts[top_idx[i, second]][: self.snippet_chars],
                    ))
                    meta.append((int(i), first, second))
            if not prompts:
                break

            raw = cache.apply_batch(prompts, self._judge) if cache is not None else self._judge(prompts)
            for (i, s1, s2), rawout in zip(meta, raw):
                choice = _parse_choice(rawout)
                pos_picks[choice if choice in (1, 2) else 0] += 1
                s_first = 1.0 if choice == 1 else 0.0 if choice == 2 else 0.5  # refusal -> draw
                expected_first = 1.0 / (1.0 + np.exp(-(ratings[i, s1] - ratings[i, s2])))
                delta = self.elo_k * (s_first - expected_first)
                ratings[i, s1] += delta
                ratings[i, s2] -= delta
            total_cmp += len(meta)

        # Rerank each active row's shortlist by final rating and place them all strictly above
        # the row's maximum (every non-shortlisted author scores <= row_max), so shortlist
        # membership is unchanged and only the internal order (hence top-1) follows the
        # tournament.
        boosted = scores.copy()
        changed_top1 = 0
        for i in active_rows:
            final_order = np.argsort(-ratings[i], kind="stable")  # best slot first
            for pos, slot in enumerate(final_order):
                boosted[i, author_idx[i, slot]] = row_max[i] + (k - pos)  # pos 0 (best) -> largest
            if final_order[0] != 0:  # the base attack's best author is no longer this row's #1
                changed_top1 += 1

        if self.verbose:
            print(f"  BT tournament: {len(active_rows)}/{n} rows tournamented "
                  f"(margin_quantile={self.margin_quantile}, gate<= {gate_thresh:.4g}), "
                  f"{self.rounds} rounds, {total_cmp} pairwise comparisons")
            print(f"  BT tournament: presented-slot picks {pos_picks} "
                  f"(0=refused/draw; want ~equal 1 vs 2 if position bias cancelled)")
            print(f"  BT tournament: top-1 changed vs base on {changed_top1}/{len(active_rows)} "
                  f"tournamented rows")

        return pd.DataFrame(boosted)


def bt_tournament_attack(data: AttackData, *, cache_dir=None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: rerank ``data``'s nearest-neighbor top-K with a default tournament.

    Any :class:`BradleyTerryTournamentAttack` constructor argument (``judge_model``, ``top_k``,
    ``rounds``, ``elo_k``, ``snippet_chars``, ``margin_quantile``, ``seed``, ...) may be passed
    through ``kwargs``.
    """
    return BradleyTerryTournamentAttack(**kwargs).attack(data, cache_dir=cache_dir)
