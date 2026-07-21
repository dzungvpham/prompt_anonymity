"""Base classes shared by the fidelity metrics.

A fidelity metric answers one question: *did the defense keep what mattered?* Each one scores a
post-defense :class:`~prompt_anonymity.core.AttackData` against its pre-defense ``reference``, and
they differ only in what they compare and what they report -- :mod:`.answer_judge` compares the
**answers** two prompts elicit (per turn, PASS/FAIL), :mod:`.prompt_judge` compares the **prompts**
themselves (per conversation, 1-5).

A developer writing a new metric subclasses :class:`FidelityMetric` and implements
:meth:`~FidelityMetric.score`; the base supplies the three things every metric needs and would
otherwise copy: side loading + validation (:meth:`~FidelityMetric._load_sides`), seeded sampling
for cheap calibration runs (the ``limit`` argument), and a ready
:class:`~prompt_anonymity.caching.TransformCache` whose key is derived from the subclass's
``version`` and :meth:`~FidelityMetric.params` (:meth:`~FidelityMetric._cache`). Results subclass
:class:`FidelityResult`, which supplies the shared ``__str__`` / ``to_csv`` behaviour.

Cache invalidation deliberately differs from :class:`prompt_anonymity.defenses.base.CachedDefense`.
A defense hashes its whole class hierarchy, so editing a base class invalidates every defense --
affordable there, because recomputing is local GPU time. Here recomputing means **billed API
calls**, and :func:`~prompt_anonymity.caching.source_digest` hashes comments and docstrings too, so
hierarchy hashing would throw away a cache of paid judge replies every time someone reworded a
comment in this file. Metrics therefore hash only the classes named by
:meth:`~FidelityMetric.logic_classes` (by default just the API client), and real logic changes are
declared by bumping :attr:`~FidelityMetric.version`. The trade is explicit: a logic edit *without*
a version bump serves stale results, which is the cheaper mistake of the two.
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ..caching import TransformCache, logic_hash, params_hash
from ._openrouter import OpenRouterChat

#: Default RNG seed for ``limit`` sampling; matches the experiment driver's ``--seed``.
DEFAULT_SEED = 47


class Sides(NamedTuple):
    """The aligned original/defended text pair a metric scores, plus its provenance.

    Attributes
    ----------
    indices : list of int
        Row positions in the *full* split that ``original`` / ``defended`` came from. Metrics put
        these in their result table's ``conv_index`` so a sampled run stays joinable against a
        full one.
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


class FidelityResult:
    """Shared behaviour for metric results: string form and a worst-first CSV dump.

    Not a dataclass -- subclasses declare their own fields with ``@dataclass`` and inherit these
    methods, which avoids the field-ordering constraints dataclass inheritance imposes. A subclass
    must provide the attributes ``n``, ``sampled_from``, ``table``, a :meth:`summary`, and
    :attr:`sort_column`.
    """

    #: Column of ``table`` that :meth:`to_csv` sorts on; override in the subclass.
    sort_column: str = ""

    n: int
    sampled_from: int | None
    table: pd.DataFrame

    def summary(self) -> str:
        """One-line human-readable score line. Implemented by subclasses; should begin with
        :meth:`sample_note` so a sampled number is never mistaken for a full-split one."""
        raise NotImplementedError

    def sample_note(self) -> str:
        """``"[SAMPLE 50/2500] "`` when this run scored a subset, else ``""``.

        Prefixed to :meth:`summary` so a cheap ``--fidelity-limit`` calibration number cannot be
        misread as a full-split result.
        """
        if self.sampled_from is None:
            return ""
        return f"[SAMPLE {self.n}/{self.sampled_from}] "

    def _sort_values(self) -> pd.Series:
        """Sort key for :meth:`to_csv`; ascending order must put the *worst* rows first.
        Override when the sort column is not already ordered that way."""
        return self.table[self.sort_column]

    def __str__(self) -> str:
        return self.summary()

    def to_csv(self, path) -> None:
        """Write the per-row table to ``path``, worst rows first so spot-checking starts with the
        rows most likely to reveal a broken defense (or a miscalibrated rubric)."""
        order = self._sort_values()
        self.table.assign(_o=order).sort_values("_o", na_position="first").drop(
            columns="_o"
        ).to_csv(path, index=False)


class FidelityMetric:
    """Base class for fidelity metrics.

    Subclasses set :attr:`name` (required -- it is the cache namespace) and :attr:`version`, may
    override :meth:`params` to declare configuration that changes model input, and implement
    :meth:`score`. Cache keys are assembled by the base, so a subclass author never touches
    invalidation; see the module docstring for why that differs from
    :class:`prompt_anonymity.defenses.base.CachedDefense`.
    """

    #: Cache namespace under ``<cache_dir>/fidelity/``. Required, and must be unique across
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
        """Classes whose source is hashed into the cache key.

        Defaults to the API client alone, deliberately excluding the metric's own class -- see the
        module docstring. Override only if a metric genuinely wants source-level invalidation and
        is willing to pay for the recomputes.
        """
        return [OpenRouterChat]

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED):
        """Score ``data`` (post-defense) against ``reference`` (pre-defense). Subclasses implement.

        Parameters
        ----------
        data, reference : AttackData
            The defended split and the loader's original; rows align by position.
        cache_dir : str or pathlib.Path
            Cache root; entries live under ``<cache_dir>/fidelity/<name>/``.
        side : {"unknown", "known"}
            Which side to score. Defaults to the side a defense rewrites.
        limit : int, optional
            Score only a seeded random sample of this many conversations. Use it to calibrate a
            rubric for a few cents before committing to a full paid run.
        seed : int
            Seed for that sample, so a limited run is reproducible.

        Returns
        -------
        FidelityResult
        """
        raise NotImplementedError

    def _load_sides(self, data, reference, side: str, *, limit: int | None = None,
                    seed: int = DEFAULT_SEED) -> Sides:
        """Validate and extract the aligned ``(original, defended)`` text pair, optionally sampled.

        Sampling lives here rather than in the caller so every metric -- including ones not yet
        written -- inherits ``--fidelity-limit`` for free, and so the sampled row *indices* travel
        with the text instead of being lost.
        """
        if side not in ("unknown", "known"):
            raise ValueError(f"side must be 'unknown' or 'known' (got {side!r}).")
        original = getattr(reference, f"{side}_texts", None)
        defended = getattr(data, f"{side}_texts", None)
        if original is None or defended is None:
            raise ValueError(
                f"fidelity scoring needs {side}_texts on both reference and data; "
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
        """A cache for this metric under ``<cache_dir>/fidelity/``.

        ``name`` and ``params`` default to :attr:`name` and :meth:`params`; pass them explicitly
        for a metric that needs a second, separately-keyed cache (e.g. :mod:`.answer_judge`, whose
        response stage is keyed by prompt text independently of the judge stage).
        """
        return TransformCache(
            Path(cache_dir) / "fidelity",
            name or self.name,
            self._logic_hash(),
            params_hash(self.params() if params is None else params),
        )
