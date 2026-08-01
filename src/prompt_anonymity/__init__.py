"""prompt_anonymity -- linkage (re-identification) attacks and metrics.

Research toolkit for the project "Can Prompt Anonymity Really Hide Your Identity?".
It tests whether stripping explicit identifiers actually anonymizes user prompts by
trying to re-link anonymous ("unknown") conversations back to the labeled ("known")
user who wrote them, purely from prompt content (semantic embeddings or writing
style). Accuracy above random guessing means anonymity has failed.

A linkage experiment runs in stages, one subpackage each:

``data``
    Load a dataset and split each user's conversations into a labeled "known" set and
    an anonymous "unknown" set, returning an :class:`~prompt_anonymity.core.AttackData`
    carrying the conversation text and the committed precomputed features.
``defenses``
    Optionally rewrite the conversation *text* to resist linkage (the no-op baseline plus
    a cached text-rewrite framework, e.g. round-trip translation). An extension point for
    anonymization countermeasures.
``features``
    Turn the (possibly defended) text into attack-ready vectors -- :func:`apply_featurizer`
    with a :class:`~prompt_anonymity.features.Featurizer` such as
    :class:`~prompt_anonymity.features.StyloMetrixFeaturizer`. Runs after defenses, reuses the
    loaded features where text is unchanged, and caches recomputed vectors.
``attacks``
    Score each unknown conversation against the known ones, producing a distance matrix
    (e.g. :class:`prompt_anonymity.attacks.NearestNeighbor`).
``metrics``
    Stateless scoring of an attack's output: :func:`~prompt_anonymity.metrics.top_k_accuracy`
    and the chance baseline :func:`~prompt_anonymity.metrics.random_guessing_accuracy`.
``fidelity``
    The utility axis complementing the privacy metrics, via
    :func:`~prompt_anonymity.fidelity.run_fidelity`: ``utility`` scores whether a defended prompt
    still yields an equally-useful answer (the paper's PASS/FAIL predicate over a response model's
    answers), and ``conversation`` scores 1-5 how much of the whole conversation survives. Both
    compare the defended split against the pre-defense ``reference``.
``evaluation``
    Rank once and reuse it: :class:`~prompt_anonymity.evaluation.LinkageRanking`, the
    headline table, and the candidate-pool-size sweep.
``viz``
    Plot helpers for the result tables (optional ``viz`` extra).

The shared hand-off object across stages is :class:`prompt_anonymity.core.AttackData`.
"""

__version__ = "0.1.0"
