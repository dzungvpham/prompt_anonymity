"""Base classes shared by the utility metrics.

A utility metric answers one question: *did the defense keep what mattered?* Each one scores a
post-defense :class:`~prompt_anonymity.core.AttackData` against its pre-defense ``reference``, and
they differ in what they compare and what they report. There is one today --
:mod:`.prompt_judge`, which compares the two **conversations** and returns a 1-5 score -- and this
base exists so a second (say, one that compares the *answers* two prompts elicit) is a subclass
rather than a fork.

A developer writing a new metric subclasses :class:`UtilityMetric` and implements
:meth:`~UtilityMetric.score`; the base supplies the three things every metric needs and would
otherwise copy: side loading + validation (:meth:`~UtilityMetric._load_sides`), seeded sampling
for cheap calibration runs (the ``limit`` argument), and a ready
:class:`~prompt_anonymity.caching.TransformCache` whose key is derived from the subclass's
``version`` and :meth:`~UtilityMetric.params` (:meth:`~UtilityMetric._cache`). Results subclass
:class:`UtilityResult`, which supplies the shared ``__str__`` and the
:meth:`~UtilityResult.scores` frame every metric contributes to the one output file.

Cache invalidation deliberately differs from :class:`prompt_anonymity.defenses.base.CachedDefense`.
A defense hashes its whole class hierarchy, so editing a base class invalidates every defense --
affordable there, because recomputing is local GPU time. Here recomputing means **billed API
calls**, and :func:`~prompt_anonymity.caching.source_digest` hashes comments and docstrings too, so
hierarchy hashing would throw away a cache of paid judge replies every time someone reworded a
comment in this file. Metrics therefore hash only the classes named by
:meth:`~UtilityMetric.logic_classes` (by default just the API client), and real logic changes are
declared by bumping :attr:`~UtilityMetric.version`. The trade is explicit: a logic edit *without*
a version bump serves stale results, which is the cheaper mistake of the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ...caching import TransformCache, logic_hash, params_hash
from ._deepseek import DeepSeekJudge

#: Default RNG seed for ``limit`` sampling; matches the experiment driver's ``--seed``.
DEFAULT_SEED = 47


class Sides(NamedTuple):
    """The aligned original/defended text pair a metric scores, plus its provenance.

    Attributes
    ----------
    indices : list of int
        Row positions in the *full* split that ``original`` / ``defended`` came from -- how a
        metric recovers each sampled row's identity (its ``conv_id``) from the bundle it was
        handed, so a sampled run's scores join against a full one's.
    original, defended : list of str
        Pre-defense and post-defense text, paired positionally.
    total : int
        Size of the full split before sampling; equal to ``len(indices)`` when nothing was sampled.
    """

    indices: list[int]
    original: list[str]
    defended: list[str]
    total: int

    @property
    def sampled_from(self) -> int | None:
        """``total`` when this is a sample, else ``None`` -- pass straight to a result's
        ``sampled_from`` so summaries can flag partial runs."""
        return self.total if len(self.indices) < self.total else None


class UtilityResult:
    """Shared behaviour for metric results: string form and the tidy per-conversation score frame.

    Not a dataclass -- subclasses declare their own fields with ``@dataclass`` and inherit these
    methods, which avoids the field-ordering constraints dataclass inheritance imposes. A subclass
    must provide the attributes ``n``, ``sampled_from``, ``table``, a :meth:`summary`, and
    :attr:`score_columns`.
    """

    #: Output column name -> the column of ``table`` it comes from. These are the *only* columns a
    #: run contributes to the shared ``experiments/utility/<source>_<defense>.csv``; everything else
    #: on ``table`` is working detail for interactive use. Names must be unique across metrics,
    #: since every metric writes into the same file -- hence the ``judge_`` prefix on the judge's.
    score_columns: dict[str, str] = {}

    n: int
    sampled_from: int | None
    table: pd.DataFrame

    def summary(self) -> str:
        """One-line human-readable score line. Implemented by subclasses; should begin with
        :meth:`sample_note` so a sampled number is never mistaken for a full-split one."""
        raise NotImplementedError

    def sample_note(self) -> str:
        """``"[SAMPLE 50/2500] "`` when this run scored a subset, else ``""``.

        Prefixed to :meth:`summary` so a cheap ``limit`` calibration number cannot be
        misread as a full-split result.
        """
        if self.sampled_from is None:
            return ""
        return f"[SAMPLE {self.n}/{self.sampled_from}] "

    def __str__(self) -> str:
        return self.summary()

    def scores(self) -> pd.DataFrame:
        """``conv_id`` plus this metric's :attr:`score_columns`, renamed to their output names.

        This is what :mod:`experiments.eval_utility` merges into the one score file a
        (dataset, defense) pair gets. Every metric returns the same shape -- a key column and some
        numbers -- so the driver merges them without knowing which metric ran, and a new metric
        joins the file by declaring :attr:`score_columns` and nothing else.
        """
        return self.table[["conv_id", *self.score_columns.values()]].rename(
            columns={source: output for output, source in self.score_columns.items()}
        )


class UtilityMetric:
    """Base class for utility metrics.

    Subclasses set :attr:`name` (required -- it is the cache namespace) and :attr:`version`, may
    override :meth:`params` to declare configuration that changes model input, and implement
    :meth:`score`. Cache keys are assembled by the base, so a subclass author never touches
    invalidation; see the module docstring for why that differs from
    :class:`prompt_anonymity.defenses.base.CachedDefense`.
    """

    #: Cache namespace under ``<cache_dir>/utility/``. Required, and must be unique across
    #: metrics: :class:`~prompt_anonymity.caching.TransformCache` prunes sibling logic-version
    #: directories under its own name, so a shared name would let one metric delete another's cache.
    name: str = ""
    #: Manual logic version. Bump when behaviour changes in a way :meth:`logic_classes` hashing
    #: cannot see -- which, given the default, means essentially any change to the metric itself.
    version: str = ""

    def params(self) -> dict:
        """Configuration that changes what the model is sent (models, prompts, truncation).

        Included in the cache key, so different configurations coexist rather than overwrite.
        Must be JSON-serializable. Leave *post-hoc* knobs out -- a threshold applied to an already
        cached score belongs outside the key, so re-sweeping it costs nothing.
        """
        return {}

    def logic_classes(self) -> list:
        """Classes whose source is hashed into the cache key. **Empty by default.**

        This used to name the API client, which meant every edit to the client -- a new timeout
        default, a log line, the usage accounting added on 2026-08-11 -- silently discarded every
        cached verdict and re-bought it at full price. None of those edits change what a judge
        replies; what does is the model, the rubric, and the effort, and all three are already in
        :meth:`params` and therefore already in the key.

        So invalidation here is entirely deliberate: bump :attr:`version` when the metric's
        behaviour really changes. The risk that buys the saving is the stated one -- a behavioural
        edit *without* a bump serves stale results -- and it is the cheaper mistake, because a
        stale namespace can be deleted by hand while re-buying a corpus cannot be undone.
        """
        return []

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED):
        """Score ``data`` (post-defense) against ``reference`` (pre-defense). Subclasses implement.

        Parameters
        ----------
        data, reference : AttackData
            The defended split and the loader's original; rows align by position.
        cache_dir : str or pathlib.Path
            Cache root; entries live under ``<cache_dir>/utility/<name>/``.
        side : {"unknown", "known"}
            Which side to score. Defaults to the side a defense rewrites.
        limit : int, optional
            Score only a seeded random sample of this many conversations. Use it to calibrate a
            rubric for a few cents before committing to a full paid run.
        seed : int
            Seed for that sample, so a limited run is reproducible.

        Returns
        -------
        UtilityResult
        """
        raise NotImplementedError

    def _load_sides(self, data, reference, side: str, *, limit: int | None = None,
                    seed: int = DEFAULT_SEED) -> Sides:
        """Validate and extract the aligned ``(original, defended)`` text pair, optionally sampled.

        Sampling lives here rather than in the caller so every metric -- including ones not yet
        written -- inherits ``limit`` for free, and so the sampled row *indices* travel
        with the text instead of being lost.
        """
        if side not in ("unknown", "known"):
            raise ValueError(f"side must be 'unknown' or 'known' (got {side!r}).")
        original = getattr(reference, f"{side}_texts", None)
        defended = getattr(data, f"{side}_texts", None)
        if original is None or defended is None:
            raise ValueError(
                f"utility scoring needs {side}_texts on both reference and data; "
                "load the dataset with text and run the defense first."
            )
        original = [str(t) for t in np.asarray(original)]
        defended = [str(t) for t in np.asarray(defended)]
        if len(original) != len(defended):
            raise ValueError(
                f"reference {side}_texts ({len(original)}) and data {side}_texts ({len(defended)}) "
                "must have the same number of rows."
            )

        total = len(original)
        indices = list(range(total))
        if limit is not None and 0 <= limit < total:
            # Seeded *random* sample, not a head slice: the splits are grouped by identity, so the
            # first N rows can easily be one or two authors and say nothing about how the metric
            # behaves across the split. Sorted so the sampled table reads in split order.
            rng = np.random.default_rng(seed)
            indices = sorted(int(i) for i in rng.choice(total, size=limit, replace=False))
            original = [original[i] for i in indices]
            defended = [defended[i] for i in indices]
        return Sides(indices, original, defended, total)

    def _logic_hash(self) -> str:
        return logic_hash(self.logic_classes(), version=self.version)

    def _cache(self, cache_dir, *, name: str | None = None,
               params: dict | None = None) -> TransformCache:
        """A cache for this metric under ``<cache_dir>/utility/``.

        ``name`` and ``params`` default to :attr:`name` and :meth:`params`; pass them explicitly
        for a metric that needs a second, separately-keyed cache -- e.g. a two-stage metric whose
        first stage (generating a response to each prompt) should be keyed by the prompt alone, so
        it is reused no matter which judge scores it.
        """
        return TransformCache(
            Path(cache_dir) / "utility",
            name or self.name,
            self._logic_hash(),
            params_hash(self.params() if params is None else params),
        )
