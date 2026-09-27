"""Nearest-neighbor linkage whose top-K is reordered by a local Jina reranker.

The free counterpart to :mod:`.listwise_llm_rerank`, and the control it needs. That attack pays a
frontier model to read K candidate conversations and order them; this one hands the identical
shortlist, presented identically, to ``jinaai/jina-reranker-v3.5`` running on the local GPU, and
folds its ordering back the same way. The only thing that differs between the two runs is what
produced the permutation, so the difference between their CMC curves is the value of the frontier
model -- rather than the value of "a reranker", which is what a comparison against the plain
distance baseline alone would actually be measuring.

The expectation going in is not obvious in either direction. jina-reranker-v3.5 is a 0.6B listwise
model (Qwen3-0.6B backbone, query and all candidates scored in one forward pass) trained for
*relevance* -- topical match between a query and a document. Authorship attribution asks the
opposite question: two texts by one author are usually about different things, and two texts on one
topic are usually by different people. A retrieval reranker may therefore score the shortlist
confidently and in exactly the wrong direction, which would make it a strong negative control; or
its language modelling may pick up stylistic regularity anyway, which would make most of the LLM's
bill unnecessary. The detail table's ``distance_rank`` column is what settles it.

No reasons here
---------------
A reranker emits a relevance score, not an argument. The detail table's ``reason`` column is
therefore empty for this attack and ``relevance_score`` carries the model's own number per
candidate, which is the nearest thing it has to a justification and is what a calibration pass
should read instead.

Practicalities
--------------
The checkpoint resolves through ``$JINA_RERANKER_MODEL`` -> ``models.toml`` -> the hub repo id, so a
cluster mirror is configured in one place and an unconfigured machine downloads a 0.6B model rather
than failing. ``torch``/``transformers`` are imported lazily inside :meth:`_ensure_model`, so
importing this module costs nothing, and ``trust_remote_code=True`` is required -- the listwise
``rerank`` interface lives in the model's own Hub-side code, not in ``transformers``. Scores are
cached under ``<cache_dir>/attacks`` like the API judges'; here that saves GPU time rather than
money, so it is a convenience rather than a necessity.

**License.** jina-reranker-v3.5 is CC BY-NC 4.0 -- non-commercial. Fine for the research use this
repository is, worth knowing before it is reused anywhere else.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

from ...caching import TransformCache, logic_hash, params_hash
from ...core import AttackData
from .candidates import author_candidates
from .listwise import detail_table, fold_listwise, present, progress_printer, report

#: The reranker checkpoint. 0.6B, multilingual, listwise -- which matters here: the corpora are not
#: English-only (swe-chat carries Chinese conversations), and an English-only reranker would score
#: those rows as noise.
DEFAULT_MODEL_ID = "jinaai/jina-reranker-v3.5"
#: Environment override for a local copy; see ``defenses/models.toml`` for the resolution ladder.
MODEL_ENV_VAR = "JINA_RERANKER_MODEL"
#: ``models.toml`` section holding the checkpoint default.
MODEL_CONFIG_SECTION = "jina_reranker"

#: How many of the distance metric's nearest authors to rerank per unknown row.
DEFAULT_TOP_K = 5
#: Chars of each conversation handed to the reranker, per text. Kept identical to the LLM judge's
#: default so the two attacks see the same evidence; the model itself supports far more (131k
#: tokens), so raising this is a real experiment rather than a workaround.
DEFAULT_SNIPPET_CHARS = 800
#: Seed for the per-row candidate shuffle.
DEFAULT_SEED = 47

#: How many scored shortlists to bank at a time. This attack runs on a **preemptible** partition,
#: and until a score is written to the cache the GPU time that produced it is lost on requeue --
#: so the number answers "how much work is a preemption allowed to destroy", in shortlists. Small,
#: because a shortlist is a forward pass and the write is one small file either way; not 1, because
#: there is no reason to pay a syscall per item when a few seconds of rework is free.
#: ``JINA_RERANK_FLUSH_EVERY`` overrides it.
FLUSH_EVERY = int(os.environ.get("JINA_RERANK_FLUSH_EVERY", "8"))

#: Manual logic version for the score cache; bump to force a full recompute (see caching.py).
RERANK_VERSION = "1"


class ListwiseJinaRerankAttack:
    """Rerank a nearest-neighbor attack's whole top-K with a local Jina reranker.

    The model is loaded lazily on first real need, so a fully-cached run touches no GPU.

    Parameters
    ----------
    model_id : str
        Reranker checkpoint, resolved through ``$JINA_RERANKER_MODEL`` -> ``models.toml`` -> this
        repo id.
    top_k : int
        How many nearest authors to rerank per unknown row (the headline comparison is 5 vs 10).
    snippet_chars : int
        Chars of each conversation handed to the reranker, per text.
    device : str or None
        ``"cuda"`` / ``"cpu"``; ``None`` picks a GPU when one is visible.
    margin_quantile : float
        Ambiguity gate in ``[0, 1]``, as in :class:`.listwise_llm_rerank.ListwiseLLMRerankAttack`.
    shuffle_candidates : bool
        Present each row's candidates in a seeded random order (default ``True``). A scoring model
        is permutation-invariant so this changes nothing for it, and it is left on anyway: it keeps
        the presented layout -- and therefore the detail tables -- directly comparable with the LLM
        judge's, which is the whole point of running this attack.
    seed : int
        Seed for the candidate shuffle.
    verbose : bool
        Print per-run diagnostics.

    Attributes
    ----------
    authors : numpy.ndarray
        Author labels indexing the returned matrix's columns; set by :meth:`attack`.
    detail : pandas.DataFrame
        One row per (unknown document x candidate), carrying ``relevance_score``; set by
        :meth:`attack`.
    """

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, top_k: int = DEFAULT_TOP_K,
                 snippet_chars: int = DEFAULT_SNIPPET_CHARS, device: str | None = None,
                 margin_quantile: float = 1.0, shuffle_candidates: bool = True,
                 seed: int = DEFAULT_SEED, verbose: bool = True):
        if not 0.0 <= margin_quantile <= 1.0:
            raise ValueError(f"margin_quantile must be in [0, 1] (got {margin_quantile}).")
        self.model_id = model_id
        self.top_k = top_k
        self.snippet_chars = snippet_chars
        self.device = device
        self.margin_quantile = margin_quantile
        self.shuffle_candidates = shuffle_candidates
        self.seed = seed
        self.verbose = verbose
        self._model = None
        self.authors: np.ndarray | None = None
        self.detail: pd.DataFrame | None = None

    def _ensure_model(self):
        """Load the reranker once, on first use.

        ``torch``/``transformers`` and the checkpoint resolver are imported *inside* this method so
        that importing :mod:`prompt_anonymity.attacks` -- which every experiment does -- never drags
        in a deep-learning stack for a run that is not using this attack. Same idiom as
        :mod:`prompt_anonymity.features.luar`.
        """
        if self._model is None:
            from transformers import AutoModel

            from ...evaluation.utility._local import resolve_checkpoint, select_device

            path = resolve_checkpoint(MODEL_CONFIG_SECTION, MODEL_ENV_VAR, self.model_id)
            device = select_device(self.device)
            print(f"listwise rerank: loading {self.model_id} from {path} on {device}")
            # trust_remote_code is required: `rerank` is defined in the model's Hub-side code.
            # dtype="auto" follows the checkpoint's own preference rather than forcing a width.
            self._model = AutoModel.from_pretrained(
                path, dtype="auto", trust_remote_code=True,
            ).to(device).eval()
        return self._model

    def _score(self, items: list[tuple[str, list[str]]]) -> list[list[float]]:
        """Relevance of each candidate to its query, **in presented-slot order**.

        ``rerank`` returns its results sorted by score and carrying the input ``index``, so the
        scores are scattered back into slot order here -- the caller ranks them itself, and a list
        that is already sorted would lose which slot each number belongs to.
        """
        return [row for _, row in self._score_stream(items)]

    def _score_stream(self, items: list[tuple[str, list[str]]]):
        """``(index, scores)`` per item, yielded as each shortlist is scored.

        One forward pass per item either way -- there is no batching knob on this model -- so this
        costs nothing over :meth:`_score` and lets the caller bank each result. On a preemptible
        GPU partition that is the difference between a requeued job resuming and one starting over,
        which for a whole corpus it may never get far enough to finish.
        """
        import torch

        model = self._ensure_model()
        with torch.inference_mode():
            for index, (query, documents) in enumerate(items):
                row = [0.0] * len(documents)
                for result in model.rerank(query, documents):
                    row[int(result["index"])] = float(result["relevance_score"])
                yield index, row

    def _cache(self, cache_dir) -> TransformCache:
        return TransformCache(
            Path(cache_dir) / "attacks", "listwise_jina_rerank",
            logic_hash([ListwiseJinaRerankAttack], version=RERANK_VERSION),
            params_hash({
                "model_id": self.model_id,
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
            Must carry ``known_texts`` and ``unknown_texts``; the reranker reads text, not vectors.
        cache_dir : str or pathlib.Path or None
            Cache root; scores live under ``<cache_dir>/attacks``. ``None`` disables caching, which
            costs GPU time rather than money.
        """
        if data.known_texts is None or data.unknown_texts is None:
            raise ValueError(
                "ListwiseJinaRerankAttack needs known_texts and unknown_texts on the AttackData "
                "(the reranker reads raw conversation text); load the dataset with text."
            )
        known_texts = [str(text) for text in np.asarray(data.known_texts)]
        unknown_texts = [str(text) for text in np.asarray(data.unknown_texts)]

        candidates = author_candidates(
            data.known_embeddings, data.known_labels, data.unknown_embeddings,
            top_k=self.top_k, metric=data.metric,
        )
        self.authors = candidates.authors
        n, k = candidates.author_index.shape
        if k < 2:
            return pd.DataFrame(candidates.scores)

        layout = present(candidates, seed=self.seed, shuffle=self.shuffle_candidates)
        items = [
            (unknown_texts[i][: self.snippet_chars],
             [known_texts[j][: self.snippet_chars] for j in layout.documents[i]])
            for i in range(n)
        ]

        if cache_dir is not None:
            cache = self._cache(cache_dir)
            relevance = cache.apply_streaming(
                items, self._score_stream, key=_cache_key, flush_every=FLUSH_EVERY,
                on_progress=progress_printer("scored") if self.verbose else None,
            )
            if self.verbose:
                print(f"  listwise rerank: {cache.hits}/{n} rows served from cache, "
                      f"{cache.misses} scored")
        else:
            relevance = self._score(items)

        # Highest relevance first. Ties break on the distance rank (`ranks` is the slot's position in
        # the base ordering), so a model that scores two candidates identically leaves that pair as
        # the embedding had it rather than as the shuffle happened to lay it out.
        orders = [sorted(range(k), key=lambda slot: (-relevance[i][slot], layout.ranks[i][slot]))
                  for i in range(n)]

        gate = np.quantile(candidates.margin, self.margin_quantile)
        applied = candidates.margin <= gate

        boosted = fold_listwise(candidates.scores, layout, orders, apply_mask=applied)
        self.detail = detail_table(candidates, layout, orders, data.unknown_labels,
                                   unknown_ids=data.unknown_ids, relevance=relevance,
                                   applied=applied)

        if self.verbose:
            report(orders, layout, n_parsed=n, n_applied=int(applied.sum()),
                   label="listwise rerank (jina)")
        return pd.DataFrame(boosted)


def _cache_key(item: tuple[str, list[str]]) -> str:
    """Content-address a (query, candidates) pair. NUL-joined because it cannot occur in the text
    and so cannot let two different pairs collide onto one key."""
    query, documents = item
    return "\x00".join([query, *documents])


def listwise_jina_rerank_attack(data: AttackData, *, cache_dir=None, **kwargs) -> pd.DataFrame:
    """Convenience wrapper: rerank ``data``'s nearest-neighbor top-K with the local Jina reranker.

    Any :class:`ListwiseJinaRerankAttack` constructor argument may be passed through ``kwargs``. The
    attack object carries per-candidate relevance scores on its ``detail`` table, which this wrapper
    discards -- construct the class directly when you want them.
    """
    return ListwiseJinaRerankAttack(**kwargs).attack(data, cache_dir=cache_dir)
