"""Multiclass discriminant attacks: fit a decision function per author on the known side.

These learn *what separates* authors rather than where each one sits, which is worth roughly a
doubling of top-1 over the same features scored by cosine distance -- when there is enough data
per author to fit a separator. See :mod:`prompt_anonymity.attacks.similarity` for the other family
and for when it wins instead.

Training on the known side is legitimate: those documents and their labels belong to the
attacker, so its labels, document counts and covariance structure are all available to them.

Cost scales very differently with the size of the author pool, which decides what is runnable at
corpus scale:

======================  ==============  =====================================================
attack                  cost in T       measured
======================  ==============  =====================================================
``rlsc``                flat            0.21 s at 200 authors, 0.31 s at 8,000
``logistic``            linear          T coupled weight vectors
``xgboost``             linear          0.097 s per author per 20 rounds -> ~8 h at 19,711
``svm``                 quadratic       one-vs-one, so T(T-1)/2 pairwise problems
======================  ==============  =====================================================

Only ``rlsc`` is genuinely usable on a pool of tens of thousands; see
:class:`RegularizedLeastSquares` for how, and for the masking failure that has to be fixed
before a one-vs-all method means anything at that shape.
"""

from __future__ import annotations

from .boosted_trees import GradientBoostedTrees
from .logistic import LogisticAttribution
from .rlsc import RegularizedLeastSquares
from .svm import SupportVectorAttribution

MULTICLASS_ATTACKS = {
    "logistic": LogisticAttribution,
    "svm": SupportVectorAttribution,
    "xgboost": GradientBoostedTrees,
    "rlsc": RegularizedLeastSquares,
}

__all__ = ["MULTICLASS_ATTACKS", "LogisticAttribution", "SupportVectorAttribution",
           "GradientBoostedTrees", "RegularizedLeastSquares"]
