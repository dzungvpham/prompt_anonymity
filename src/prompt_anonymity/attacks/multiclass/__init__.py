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
``logistic_sgd``        linear          same model, minibatched: ~2 TFLOP/epoch at 19,711
``xgboost``             linear          0.097 s per author per 20 rounds -> ~8 h at 19,711
``svm``                 quadratic       one-vs-one, so T(T-1)/2 pairwise problems
======================  ==============  =====================================================

``rlsc`` is the only one whose cost does not grow with the pool at all; see
:class:`RegularizedLeastSquares` for how, and for the masking failure that has to be fixed
before a one-vs-all method means anything at that shape. ``logistic_sgd`` is linear like
``logistic`` but with a constant small enough to run: it is the same objective fitted by
minibatch Adam, which bounds the ``[documents x authors]`` logit matrix that makes lbfgs
impossible past a few thousand authors, and puts the two matrix products on a GPU. See
:class:`MinibatchLogisticAttribution` for the measured boundary and for why it is a separate
registry entry rather than a solver flag on ``logistic``.
"""

from __future__ import annotations

from .boosted_trees import GradientBoostedTrees
from .logistic import LogisticAttribution
from .logistic_sgd import MinibatchLogisticAttribution
from .rlsc import RegularizedLeastSquares
from .svm import SupportVectorAttribution

MULTICLASS_ATTACKS = {
    "logistic": LogisticAttribution,
    "logistic_sgd": MinibatchLogisticAttribution,
    "svm": SupportVectorAttribution,
    "xgboost": GradientBoostedTrees,
    "rlsc": RegularizedLeastSquares,
}

__all__ = ["MULTICLASS_ATTACKS", "LogisticAttribution", "MinibatchLogisticAttribution",
           "SupportVectorAttribution", "GradientBoostedTrees", "RegularizedLeastSquares"]
