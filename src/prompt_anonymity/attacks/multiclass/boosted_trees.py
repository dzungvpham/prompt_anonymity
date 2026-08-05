"""Gradient-boosted decision trees (XGBoost) over the known authors."""

from __future__ import annotations

import numpy as np


class GradientBoostedTrees:
    """Gradient-boosted decision trees (XGBoost) over the known authors.

    The only non-linear, non-metric attack here: every other one ultimately compares documents
    along straight lines in feature space. StyloMetrix features are heterogeneous -- ratios in
    [0, 1] next to raw counts, many near-zero for most documents -- and trees handle that mix
    natively, splitting on thresholds instead of weighting directions, and picking up interactions
    between features that a linear model cannot express.

    The score is the **log** class probability, not the raw probability. It is the same ranking
    either way, but log-space is what the downstream machinery expects: cohort normalisation
    z-scores across authors (meaningful for log-odds-like quantities, not for probabilities that
    sum to 1), and it makes ``softmax(score)`` recover the model's own posterior exactly, so the
    calibration metrics in :mod:`prompt_anonymity.metrics.detection` measure something real for
    this attack.

    This is also the **most expensive** attack in the package, and the cost is driven by the
    feature count in a way the others' is not. A tree has to search for a split *per feature, per
    node*, where a distance-based attack sees the same vectors as one matrix multiply; and the
    multi-class objective trains one tree per author per boosting round. Measured on swe-chat
    (997 documents, 81 authors, 10 rounds, 2 threads): 1.23 s on 196-dimensional StyloMetrix
    against 54.2 s on 3,072-dimensional Gemini embeddings -- 44x for 15.7x the columns. Prefer a
    linear attack on wide dense embeddings, or set ``device="cuda"``.

    ``xgboost`` is imported lazily so the rest of the package works without it installed.
    """

    name = "xgboost"

    def __init__(self, n_estimators: int = 300, max_depth: int = 3, learning_rate: float = 0.3,
                 subsample: float = 1.0, n_jobs: int = -1, device: str = "cpu"):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.device = device

    def resolved_device(self) -> str:
        """``self.device``, with ``"auto"`` resolved to ``"cuda"`` only if a GPU is usable.

        ``"auto"`` is not the default deliberately. The GPU histogram builder sums gradients in a
        different order from the CPU one, so the two can choose different splits and a run is
        then only reproducible on the same kind of machine -- a silent dependency on what hardware
        happened to be free. Ask for ``"cuda"`` when you want it, and record it.

        The published speedups come from datasets with 10^5-10^7 rows and these windows have
        10^3, so per-node launch overhead is a large fraction of the work; the wide feature axis
        is what pays for it instead. Measured on the shape this attack actually sees on swe-chat
        with Gemini embeddings (1,000 documents x 3,072 features, 81 authors, 10 rounds, 8 CPU
        threads against one A100-80GB): 22.1 s on CPU against 1.29 s on CUDA, 17x.
        """
        if self.device != "auto":
            return self.device
        try:
            from xgboost import XGBClassifier, build_info

            if not build_info().get("USE_CUDA"):  # a CPU-only wheel cannot be talked into a GPU
                return "cpu"
            # A CUDA-enabled wheel still says nothing about whether a device is *visible*, and
            # xgboost exposes no device count to ask (there is no ``xgboost.device_count``; an
            # earlier version of this check imported one and so sent every ``auto`` run to the
            # CPU). Fitting two rows is the cheap definitive probe -- milliseconds, and it raises
            # exactly when a real fit would.
            XGBClassifier(n_estimators=1, device="cuda", verbosity=0).fit(
                np.zeros((2, 1), dtype=np.float32), np.array([0, 1]))
            return "cuda"
        except Exception:  # no driver, no visible device, or a device too old for the kernels
            return "cpu"

    def fit(self, embeddings, labels):
        from xgboost import XGBClassifier  # lazy: keeps xgboost an optional runtime dependency

        self.authors, codes = np.unique(labels, return_inverse=True)
        self.model = XGBClassifier(
            n_estimators=self.n_estimators, max_depth=self.max_depth,
            learning_rate=self.learning_rate, subsample=self.subsample,
            tree_method="hist", device=self.resolved_device(),
            objective="multi:softprob", n_jobs=self.n_jobs, verbosity=0,
        ).fit(embeddings, codes)
        return self

    def score(self, embeddings):
        # A cuda-fitted model handed host arrays warns about the device mismatch and copies them
        # over. Left alone: prediction is a rounding error next to fitting, and moving the unknown
        # side to the device here would put the transfer in a different place, not remove it.
        return np.log(np.clip(self.model.predict_proba(embeddings), 1e-12, None))

