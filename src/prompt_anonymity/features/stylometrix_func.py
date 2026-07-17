"""StyloFuncFeaturizer: concatenates two or more featurizers' output vectors.

Lets you test whether stacking a weaker signal (e.g. function words) onto a strong one
(StyloMetrix) improves linkage accuracy, without writing a new attack or touching
evaluation code — it's just a wider feature vector fed into the same nearest-neighbor
attack.
"""

from __future__ import annotations

import numpy as np
from .stylometrix import StyloMetrixFeaturizer
from .function_words import FunctionWordFeaturizer
from .base import Featurizer


class StyloFuncFeaturizer(Featurizer):
    """Concatenates the feature vectors of multiple sub-featurizers, column-wise.

    All sub-featurizers must use the same distance metric family; this class inherits
    the metric from the first sub-featurizer by default (override via constructor).
    """

    name = "stylometrix_func"
    version = "1"

    def __init__(
        self,
        featurizers: list[Featurizer] | None = None,
        metric: str | None = None,
    ):
        if featurizers is None:
            featurizers = [
                StyloMetrixFeaturizer(),
                FunctionWordFeaturizer(),
            ]

        if not featurizers:
            raise ValueError("StyloFuncFeaturizer needs at least one sub-featurizer.")

        self.featurizers = featurizers
        self.metric = metric or featurizers[0].metric

    def params(self) -> dict:
        # Included in the cache key so different sub-featurizer combos cache separately.
        return {"sub_featurizers": [f.name for f in self.featurizers]}

    def featurize(self, texts) -> np.ndarray:
        texts = list(texts)
        parts = [np.asarray(f.featurize(texts), dtype=float) for f in self.featurizers]
        return np.concatenate(parts, axis=1)