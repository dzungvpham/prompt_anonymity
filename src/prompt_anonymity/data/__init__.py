"""The dataset build pipeline: raw upstream corpora in, the unified public dataset out.

This package used to hold a second family of modules beside the build pipeline -- per-dataset
*loaders* (``splits.py``, ``wildchat.py``, ``swe_chat.py``) that read the pre-unification CSVs,
applied a dataset-specific known/unknown split and returned an
:class:`~prompt_anonymity.core.AttackData` for a defense and an attack. Their only caller was the
fixed-split experiment runner, and both went on 2026-08-04: the split is now a property of the
*experiment* (``experiments/run_experiment.py`` cuts the timeline into known intervals itself,
straight from the built parquets), not of a loader. Recover them from git history if the
per-dataset CSV path is ever needed again.

The pipeline's entry points are meant to be run as scripts, in this order:

    python -m prompt_anonymity.data.build_dataset       # raw sources  -> data/dist/*.parquet
    python -m prompt_anonymity.data.validate_dataset    # integrity checks on what was built
    python -m prompt_anonymity.data.compute_features    # -> data/dist/<split>_<feature>.parquet

with :mod:`~prompt_anonymity.data.sources_wildchat` / :mod:`~prompt_anonymity.data.sources_swe_chat`
adapting one upstream corpus each, and :mod:`~prompt_anonymity.data.text_cleaning`,
:mod:`~prompt_anonymity.data.identity`, :mod:`~prompt_anonymity.data.dedup` and
:mod:`~prompt_anonymity.data.language_detection` implementing the shared stages.
:mod:`~prompt_anonymity.data.download` mirrors the published dataset back down, and
:mod:`~prompt_anonymity.data.find_fragments` is a one-off study of identity fragmentation.

:mod:`~prompt_anonymity.data.apply_defenses` is an optional fourth step, run between the last two:
it rewrites a built split's conversations with one registered defense
(:mod:`prompt_anonymity.defenses`) and writes them back in the *same schema* at
``data/dist/defended/<defense>/<split>.parquet`` -- a drop-in replacement, so
``compute_features --dist-dir data/dist/defended/<defense>`` featurizes the defended text with no
further changes and the attack pipeline can then be run against it.

**Code lives here; data does not.** Where the raw inputs are read from and where the outputs are
written is configuration, not a constant -- see :mod:`prompt_anonymity.data.config`, which resolves
both (falling back to downloading the raw sources from HuggingFace) so the build runs on any
machine.
"""

from __future__ import annotations

# Deliberately empty of re-exports. Every module here is either a script (`python -m
# prompt_anonymity.data.<name>`) or a stage imported by one, so importing this package should cost
# nothing: `build_dataset` alone pulls in pyarrow, the raw-source adapters and language detection,
# which is not a price `import prompt_anonymity.data` should pay to reach `config.data_dir`.
__all__: list[str] = []
