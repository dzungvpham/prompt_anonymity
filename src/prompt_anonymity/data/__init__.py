"""The dataset build pipeline: raw upstream corpora in, the unified public dataset out.

The known/unknown split is a property of the experiment runner, not of a loader here -- this
package only builds and validates the dataset.

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

# Deliberately empty of re-exports: every module here is a script or a stage imported by one, so
# importing this package should not pull in any of their heavy dependencies.
__all__: list[str] = []
