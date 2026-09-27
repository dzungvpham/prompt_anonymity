"""Open-set handling: deciding that the best-scoring author is nobody at all.

Identification asks which known author wrote a document. In a real corpus many anonymous
documents were written by somebody the attacker has never seen, so a usable attack also has to
**reject**. This package turns a score matrix into that accept/reject decision;
:mod:`prompt_anonymity.evaluation.metrics.detection` scores how well it did.
"""

from __future__ import annotations

from .rejection import LearnedRejector, cohort_normalize, rejection_features, rejection_score

__all__ = ["cohort_normalize", "rejection_score", "rejection_features", "LearnedRejector"]
