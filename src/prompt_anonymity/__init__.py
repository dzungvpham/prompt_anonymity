"""prompt_anonymity -- linkage (re-identification) attacks and metrics.

Research toolkit for the project "Can Prompt Anonymity Really Hide Your Identity?".
It tests whether stripping explicit identifiers actually anonymizes user prompts by
trying to re-link anonymous ("unknown") conversations back to the labeled ("known")
user who wrote them, purely from prompt content (semantic embeddings or writing
style). Accuracy above random guessing means anonymity has failed.

The work splits into a **build** half, run once per corpus as scripts, and an **experiment**
half that reads what the build wrote. One subpackage each:

``data``
    The build pipeline: raw upstream corpora -> one parquet per split, then feature vectors
    (``compute_features``) and, optionally, defended copies of the text (``apply_defenses``).
    Everything downstream reads its output rather than re-deriving it.
``defenses``
    Rewrite the conversation *text* to resist linkage (the no-op baseline plus a cached
    text-rewrite framework, e.g. round-trip translation, StyleRemix, OpenAnonymity). An
    extension point for anonymization countermeasures; driven by ``data.apply_defenses``.
``features``
    :class:`~prompt_anonymity.features.Featurizer` classes such as
    :class:`~prompt_anonymity.features.GeminiEmbedding2Featurizer` -- cached ``texts -> ndarray``
    transforms, driven by ``data.compute_features``.
``attacks``
    Score each unknown document against the known authors, producing an
    ``[n_documents x n_authors]`` score matrix (e.g.
    :class:`prompt_anonymity.attacks.NearestNeighbor`).
``evaluation``
    Everything that turns an attack's or a defense's output into a number, in three layers:
    ``evaluation.metrics`` (stateless scoring of one score matrix --
    :func:`~prompt_anonymity.evaluation.metrics.top_k_accuracy` and the chance baseline
    :func:`~prompt_anonymity.evaluation.metrics.random_guessing_accuracy`),
    :class:`~prompt_anonymity.evaluation.LinkageRanking` (rank once, reuse it), and
    ``evaluation.utility`` -- the **other axis**, measuring what a defense cost rather than what
    it hid. Driven by ``experiments/eval_utility.py``.

    Absorbed the former top-level ``prompt_anonymity.metrics`` and ``prompt_anonymity.utility``
    packages; those paths no longer exist.

:class:`prompt_anonymity.core.AttackData` is the hand-off object between a defense and the text
it rewrites; it is what ``data.apply_defenses`` hands each defense.

The package deliberately stops at numbers: it has no plotting module. Figures are a
presentation concern that belongs with the experiments, and drawing them from inside a run
means each run can only ever plot itself -- so all of it lives in
``experiments/plot_results.py``, which reads the written CSVs back and can compare runs.
"""

__version__ = "0.1.0"
