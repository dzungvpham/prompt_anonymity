"""Shared machinery for the listwise rerankers: order the whole shortlist, not just its head.

:mod:`.euclidean_llm_judge` asks its judge for **one digit** -- which of the K shortlisted authors
wrote this? -- and promotes that one candidate to rank 1. Everything else in the row keeps the
distance metric's order, so at ``top_k=5`` the top-5 and top-10 accuracies are identical to plain
nearest-neighbor *by construction* and only top-1 can move. That is a deliberate, clean isolation
of the rank-1 decision, and it is also most of what the judge knows thrown away: a model that has
read five candidate texts has an opinion about all five, and ranks 2..K are where the rest of the
CMC curve lives.

The attacks built on this module ask instead for the **whole ordering** of the shortlist. Two of
them share it -- :mod:`.listwise_llm_rerank` (a hosted LLM reading the texts) and
:mod:`.listwise_jina_rerank` (a local cross-encoder scoring them) -- and they differ only in how
the order is produced. Shortlisting, presentation, fold-back and reporting are all here.

What moves, and what cannot
---------------------------
:func:`fold_listwise` rewrites only the K shortlisted authors' scores, placing them strictly above
every other author in the reranker's order. So shortlist *membership* is still exactly what the
distance metric found, and:

* at ``top_k=5``: top-1 through top-4 can move; **top-5 is pinned** to the base attack's top-5.
* at ``top_k=10``: top-1 through top-9 can move; **top-10 is pinned**.

That pinned number is the point. It is the recall ceiling the reranker was handed, unchanged by
anything the reranker did, so the gain at every k below it is attributable to the reranking alone
rather than to a shortlist that happened to be better.

Failure degrades to the baseline, never to noise
------------------------------------------------
Candidates are presented in a seeded *shuffle* (see :func:`present`), so a "do nothing" order in
presented-slot terms is a random order, not the distance order -- the opposite of what a refusal
should cost. :func:`complete_order` therefore fills whatever the reranker did not rank by
**distance rank**, nearest first. A row the model refused outright, or whose reply did not parse at
all, comes out as the plain nearest-neighbor ordering: the same guarantee
:mod:`.euclidean_llm_judge` gets by forcing a refusal to the nearest candidate, extended to a whole
permutation.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import numpy as np
import pandas as pd

from .candidates import AuthorCandidates


class Presentation(NamedTuple):
    """How each row's shortlist was laid out in front of the reranker.

    All three arrays are indexed by **presented slot** -- slot 0 is the candidate labelled
    ``CANDIDATE 1`` -- so a reranker's positional answer maps straight back through them.

    Attributes
    ----------
    ranks : list of numpy.ndarray, each of shape (k,)
        ``ranks[i][slot]`` is the *distance* rank (0 = the base attack's best author) of the
        candidate shown in that slot. This is the signal axis: it says where in the distance
        ordering the reranker's choices actually came from.
    documents : list of numpy.ndarray, each of shape (k,)
        ``documents[i][slot]`` is the known-side row index of the text shown in that slot.
    authors : list of numpy.ndarray, each of shape (k,)
        ``authors[i][slot]`` is the score-matrix column of the author shown in that slot.
    """

    ranks: list[np.ndarray]
    documents: list[np.ndarray]
    authors: list[np.ndarray]


def present(candidates: AuthorCandidates, *, seed: int, shuffle: bool = True) -> Presentation:
    """Lay each row's shortlist out for the reranker, shuffled by default.

    Rerankers that read a list over-weight whichever end they saw last (position/recency bias).
    Feeding candidates in distance-rank order then systematically rewards the distance metric's
    *worst* shortlisted candidate, which is how a naive rerank ends up below its own baseline. Each
    row's K candidates are therefore shown in a seeded -- and so reproducible, so cache-stable --
    random order, and the reranker's positional answer is mapped back through
    :attr:`Presentation.documents` / :attr:`Presentation.authors`.

    ``shuffle=False`` presents them nearest-first; it exists to *measure* the bias (compare the two
    runs' positional pick distributions), not as a working configuration.
    """
    n, k = candidates.author_index.shape
    rng = np.random.default_rng(seed)
    perms = [rng.permutation(k) if shuffle else np.arange(k) for _ in range(n)]
    return Presentation(
        ranks=perms,
        documents=[candidates.document_index[i][perms[i]] for i in range(n)],
        authors=[candidates.author_index[i][perms[i]] for i in range(n)],
    )


def complete_order(partial: Sequence[int], ranks: np.ndarray) -> list[int]:
    """Complete a partial slot ordering by appending what is missing in **distance order**.

    ``partial`` is the slots the reranker actually placed, best first; ``ranks`` is one row of
    :attr:`Presentation.ranks`. Slots absent from ``partial`` -- because the reply was truncated,
    skipped a candidate, or did not parse at all -- are appended nearest-first, so:

    * an empty ``partial`` reproduces the base attack's own ordering exactly, and
    * a partial answer keeps what the reranker said and falls back to the distance metric below it.

    Duplicates and out-of-range entries are the parser's job to drop before this is called; this
    function only fills gaps.
    """
    seen = set(int(slot) for slot in partial)
    tail = sorted((slot for slot in range(len(ranks)) if slot not in seen),
                  key=lambda slot: int(ranks[slot]))
    return [int(slot) for slot in partial] + tail


def fold_listwise(scores: np.ndarray, presentation: Presentation, orders: Sequence[Sequence[int]],
                  *, apply_mask: np.ndarray | None = None) -> np.ndarray:
    """Write a per-row shortlist ordering back into the score matrix (**higher = more likely**).

    Each shortlisted author is given ``row_max + (k - rank)``, so the reranker's first choice sits
    at ``row_max + k`` and its last at ``row_max + 1``. Every one of them therefore clears
    ``row_max`` -- the best score anywhere in that row -- while keeping the reranker's order intact
    among themselves. Nothing outside the shortlist is touched, which is what pins top-K accuracy
    at the shortlist size to the base attack's own number (see this module's docstring).

    Parameters
    ----------
    scores : numpy.ndarray of shape (n_unknown, n_authors)
        The base attack's scores; copied, not modified.
    presentation : Presentation
        The layout the orders refer to.
    orders : sequence of sequence of int
        ``orders[i]`` is a permutation of presented slots, best first, length k.
    apply_mask : numpy.ndarray of shape (n_unknown,) or None
        Rows to rerank. ``None`` reranks every row; ``False`` leaves that row exactly as the base
        attack scored it (used by the ambiguity gate).
    """
    boosted = np.array(scores, dtype=float, copy=True)
    row_max = scores.max(axis=1)
    k = len(presentation.authors[0]) if presentation.authors else 0
    for i, order in enumerate(orders):
        if apply_mask is not None and not apply_mask[i]:
            continue
        for rank, slot in enumerate(order):
            boosted[i, presentation.authors[i][slot]] = row_max[i] + (k - rank)
    return boosted


def detail_table(candidates: AuthorCandidates, presentation: Presentation,
                 orders: Sequence[Sequence[int]], unknown_labels, *, unknown_ids=None,
                 reasons: Sequence[Sequence[str]] | None = None,
                 relevance: Sequence[Sequence[float]] | None = None,
                 applied: np.ndarray | None = None) -> pd.DataFrame:
    """One row per (unknown document x shortlisted candidate): what was shown, where it landed, why.

    This is the side-car the score matrix cannot carry. It follows the precedent set by
    :class:`~prompt_anonymity.evaluation.utility.prompt_judge.ConversationUtilityResult`, whose
    free-text ``reason`` lives on a per-item detail table and is deliberately kept out of the
    aggregated columns: a justification is for reading while calibrating the rubric, not for
    averaging.

    ``distance_rank`` is the column that diagnoses the reranker. If the model's ranks 1-3 sit on
    distance ranks 0-1 it is finding the same neighbourhood the vectors did and sharpening it; if
    its lower ranks correlate with nothing, the tail of a long shortlist is noise -- which is a
    result to report, not a bug.

    Parameters
    ----------
    reasons : sequence of sequence of str or None
        ``reasons[i][rank]`` justifies the candidate the reranker placed at that rank. ``None``
        for a reranker that emits no text (the Jina variant), leaving the column empty.
    relevance : sequence of sequence of float or None
        ``relevance[i][slot]`` is a per-**slot** score from the reranker, if it produced one.
    applied : numpy.ndarray of shape (n_unknown,) or None
        Whether the row's rerank was actually folded back (``False`` under the ambiguity gate).
    """
    authors = np.asarray(candidates.authors)
    unknown_labels = np.asarray(unknown_labels)
    records = []
    for i, order in enumerate(orders):
        for rank, slot in enumerate(order):
            author_column = int(presentation.authors[i][slot])
            records.append({
                "row": i,
                "unknown_doc": None if unknown_ids is None else unknown_ids[i],
                "true_author": unknown_labels[i],
                "rerank_position": rank + 1,          # 1 = the reranker's best guess
                "presented_slot": int(slot) + 1,      # the "CANDIDATE n" label it was shown under
                "distance_rank": int(presentation.ranks[i][slot]),  # 0 = base attack's best author
                "author": authors[author_column],
                "known_doc_row": int(presentation.documents[i][slot]),
                "is_true_author": bool(authors[author_column] == unknown_labels[i]),
                "reason": "" if reasons is None else reasons[i][rank],
                "relevance_score": (np.nan if relevance is None
                                    else float(relevance[i][slot])),
                "applied": True if applied is None else bool(applied[i]),
            })
    return pd.DataFrame.from_records(records)


def report(orders: Sequence[Sequence[int]], presentation: Presentation, *, n_parsed: int,
           n_applied: int, label: str = "listwise rerank") -> None:
    """Print the two distributions that say whether the rerank did anything real.

    **Position axis** -- which presented slot the reranker put first. With shuffling on this should
    be ~uniform; a spike on the last slot means position bias survived and the ranking is partly an
    artifact of the layout.

    **Signal axis** -- the *distance* rank of the candidate the reranker put first. Mass on 0-1
    means it is agreeing with the vectors about the neighbourhood and re-ordering within it; a flat
    spread across all K means it found no style signal the embedding had not already found, and the
    rerank is closer to a shuffle than to an attack.
    """
    n = len(orders)
    k = len(presentation.authors[0]) if presentation.authors else 0
    firsts = [int(order[0]) for order in orders if len(order)]
    slot_distribution = {slot + 1: firsts.count(slot) for slot in range(k)}
    rank_distribution = {
        rank: sum(1 for i, order in enumerate(orders)
                  if len(order) and int(presentation.ranks[i][order[0]]) == rank)
        for rank in range(k)
    }
    print(f"  {label}: top pick by presented slot {slot_distribution} (want ~uniform if unbiased)")
    print(f"  {label}: top pick by distance rank {rank_distribution} (0=base best; want mass on 0-1)")
    print(f"  {label}: {n_parsed}/{n} rows returned a usable ordering; "
          f"{n_applied}/{n} folded back into the scores")


# --- self-test ---------------------------------------------------------------

def _synthetic(seed: int = 3, n_authors: int = 12, per_author: int = 5, dim: int = 8):
    """A small author-clustered corpus as an :class:`~prompt_anonymity.core.AttackData`.

    Clustered, so the nearest-neighbor baseline is well above chance without being perfect -- a
    baseline that is already at 100% cannot show a rerank moving anything.
    """
    from ...core import AttackData

    rng = np.random.default_rng(seed)
    n = n_authors * per_author
    centers = rng.normal(size=(n_authors, dim))
    known = np.array([centers[i // per_author] + 0.9 * rng.normal(size=dim) for i in range(n)])
    labels = np.array([f"auth{i // per_author:02d}" for i in range(n)], dtype=object)
    unknown = np.array([centers[i % n_authors] + 0.9 * rng.normal(size=dim) for i in range(20)])
    unknown_labels = np.array([f"auth{i % n_authors:02d}" for i in range(20)], dtype=object)
    return AttackData(
        known_embeddings=known, unknown_embeddings=unknown,
        known_labels=labels, unknown_labels=unknown_labels, metric="cosine",
        known_texts=np.array([f"known text {i}" for i in range(n)], dtype=object),
        unknown_texts=np.array([f"unknown text {i}" for i in range(20)], dtype=object),
        unknown_ids=np.array([f"d{i:03d}" for i in range(20)], dtype=object),
    )


def _selftest() -> None:
    """Every invariant the two listwise rerankers rest on -- offline, no key, no GPU.

    This repo has no test framework and no pytest dependency, so the checks live here rather than
    introducing one (the same choice ``epi``, ``collision_seeding`` and ``frame_shift`` made). Run
    with ``python -m prompt_anonymity.attacks.llm.listwise --selftest``.

    Note that running this module as ``__main__`` gives it a second identity, so the
    ``fold_listwise`` the attacks call is a different function object from the one checked directly
    above it. The behaviour is identical and nothing here hashes the source (the end-to-end checks
    pass ``cache_dir=None``), so the duplication is cosmetic -- but it is why a ``logic_hash`` must
    never be asserted from inside a self-test.
    """
    import json

    from ..similarity import NearestNeighbor
    from .candidates import author_candidates
    from .listwise_llm_rerank import ListwiseLLMRerankAttack, _parse_ranking

    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  -- {detail}'}")
        if not condition:
            failures.append(name)

    # 1. The parser repairs rather than rejects. A K-way permutation is a far larger target than the
    #    single-pick judge's one digit, so every way a model can miss it is a way to lose a whole
    #    row's judgement -- these are the shapes observed in practice.
    k = 5
    good = json.dumps({"ranking": [{"candidate": c, "why": f"reason {c}"} for c in [3, 1, 5, 2, 4]]})
    check("parses a well-formed ranking", _parse_ranking(good, k)[0] == [2, 0, 4, 1, 3])
    check("keeps one reason per position", _parse_ranking(good, k)[1][0] == "reason 3")
    check("survives markdown fences", _parse_ranking(f"```json\n{good}\n```", k)[0] == [2, 0, 4, 1, 3])
    check("survives a preamble", _parse_ranking(f"Here you go:\n{good}\nHTH", k)[0] == [2, 0, 4, 1, 3])
    check("drops a repeated candidate",
          _parse_ranking(json.dumps({"ranking": [{"candidate": c} for c in [3, 1, 3, 2]]}), k)[0]
          == [2, 0, 1])
    check("drops out-of-range candidates",
          _parse_ranking(json.dumps({"ranking": [{"candidate": c} for c in [9, 2, 0, -1, 4]]}), k)[0]
          == [1, 3])
    check("accepts a bare list of numbers",
          _parse_ranking(json.dumps({"ranking": [3, 1, 2]}), k)[0] == [2, 0, 1])
    check("refusal parses to nothing", _parse_ranking("I cannot help with that.", k) == ([], []))
    check("truncated JSON parses to nothing",
          _parse_ranking('{"ranking": [{"candidate": 3, "why": "abc', k) == ([], []))

    # 2. Completion falls back to DISTANCE order, not to the presented order. Candidates are shown
    #    shuffled, so "do nothing" in slot terms is a random ranking -- the opposite of what a
    #    refusal should cost.
    ranks = np.array([3, 0, 4, 1, 2])
    check("empty ordering reproduces the distance order",
          complete_order([], ranks) == [1, 3, 4, 0, 2])
    check("partial ordering keeps its head, fills by distance",
          complete_order([2], ranks) == [2, 1, 3, 4, 0])
    check("complete ordering is left alone",
          complete_order([0, 1, 2, 3, 4], ranks) == [0, 1, 2, 3, 4])

    # 3. Fold-back moves the order inside the shortlist and nothing else. This is the property every
    #    reported number depends on: if it fails, top-K at the shortlist size stops being the base
    #    attack's own recall ceiling and the comparison means nothing.
    data = _synthetic()
    rng = np.random.default_rng(0)
    candidates = author_candidates(data.known_embeddings, data.known_labels,
                                   data.unknown_embeddings, top_k=5, metric="cosine")
    layout = present(candidates, seed=47, shuffle=True)
    n, width = candidates.author_index.shape
    orders = [list(rng.permutation(width)) for _ in range(n)]
    boosted = fold_listwise(candidates.scores, layout, orders)
    check("shortlist membership is unchanged",
          all(set(np.argsort(-candidates.scores[i])[:width])
              == set(np.argsort(-boosted[i])[:width]) for i in range(n)))
    check("order inside the shortlist is the reranker's",
          all(list(np.argsort(-boosted[i])[:width])
              == [int(layout.authors[i][slot]) for slot in orders[i]] for i in range(n)))
    check("a closed gate leaves the scores untouched",
          np.array_equal(fold_listwise(candidates.scores, layout, orders,
                                       apply_mask=np.zeros(n, dtype=bool)), candidates.scores))

    # 4. End to end through the LLM attack, with the judge replaced by a canned reply. The canned
    #    ranking REVERSES whatever it was shown, which is the most disruptive valid permutation
    #    there is -- so if top-K still cannot move, nothing a real judge does will move it either.
    ranker = NearestNeighbor(metric="cosine", linkage="max")
    ranker.fit(data.known_embeddings, data.known_labels)
    base = np.asarray(ranker.score(data.unknown_embeddings), dtype=float)

    attack = ListwiseLLMRerankAttack(top_k=5, verbose=False)
    attack._judge = lambda prompts, width, tickets: [
        json.dumps({"ranking": [{"candidate": c, "why": f"candidate {c} reads similarly"}
                                for c in range(width, 0, -1)]})
        for _ in prompts
    ]
    reranked = attack.attack(data, cache_dir=None).to_numpy()
    for cutoff in (5, 10):
        check(f"top-{cutoff} membership still matches the baseline",
              all(set(np.argsort(-base[i])[:cutoff]) == set(np.argsort(-reranked[i])[:cutoff])
                  for i in range(n)))
    check("top-1 actually moved",
          any(base[i].argmax() != reranked[i].argmax() for i in range(n)))
    check("detail table is one row per candidate", attack.detail.shape[0] == n * 5)
    check("every position carries a reason", bool(attack.detail["reason"].str.len().gt(0).all()))

    # 5. Batching is a delivery channel, not a prompt, so the two must share a cache namespace --
    #    otherwise the cheap real-time smoke run is paid for twice, once again at batch rates.
    sync = ListwiseLLMRerankAttack(top_k=5, batch=False)
    batched = ListwiseLLMRerankAttack(top_k=5, batch=True)
    import tempfile
    with tempfile.TemporaryDirectory() as scratch:
        check("batch and real-time share one cache namespace",
              sync._cache(scratch, 5).dir == batched._cache(scratch, 5).dir)
        check("a different reasoning effort gets its own namespace",
              batched._cache(scratch, 5).dir
              != ListwiseLLMRerankAttack(top_k=5, reasoning_effort="low")._cache(scratch, 5).dir)

    # 6. A judge that refuses everything must land exactly on the baseline -- never worse.
    refusing = ListwiseLLMRerankAttack(top_k=5, verbose=False)
    refusing._judge = lambda prompts, width, tickets: ["I'm sorry, I can't do that." for _ in prompts]
    refused = refusing.attack(data, cache_dir=None).to_numpy()
    check("total refusal reproduces the baseline ordering",
          all(list(np.argsort(-refused[i])) == list(np.argsort(-base[i])) for i in range(n)))

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Invariant checks for the listwise rerank attacks (offline; no key, no GPU).")
    parser.add_argument("--selftest", action="store_true", help="run the checks")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("nothing to do; pass --selftest")
    print("listwise rerank self-test")
    _selftest()


if __name__ == "__main__":
    main()
