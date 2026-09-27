"""Multinomial logistic regression fitted by minibatch gradient descent, for large author pools.

Same *model* as :class:`~prompt_anonymity.attacks.multiclass.logistic.LogisticAttribution` -- one
softmax over the known authors, L2-regularised, optionally class-balanced -- fitted a different
way so that it is runnable when there are tens of thousands of authors.

Why the sklearn path stops working
----------------------------------
``LogisticRegression``'s multinomial loss materialises the full ``[n_documents x n_authors]``
matrix of logits **and** the matrix of probabilities, both in float64, on every lbfgs iteration.
That's affordable with a small author pool but grows with both document and author count until it
exceeds memory outright on a corpus with tens of thousands of authors.

What this does instead
----------------------
Minibatching bounds the logit matrix at ``[batch_size x n_authors]`` regardless of corpus size,
and puts the two matrix products (``X W`` forward, ``X^T dZ`` backward) on a GPU where that shape
is cheap. The weight matrix itself is small enough to fit on any card even at a very large author
pool, so only the per-batch logits need bounding.

The objective is written to be the *same function* sklearn minimises, so ``C`` means what it means
there and :data:`~run_experiment.HYPERPARAMETER_SPACES` transfers unchanged::

    sum_i weight_i * cross_entropy(x_i W + b, y_i)  +  ||W||^2 / (2 C)

with the intercept unregularised, and ``weight_i`` from ``class_weight="balanced"`` as
``n_documents / (n_authors * count(y_i))``. It is optimised rather than solved, so it lands *near*
sklearn's optimum rather than on it -- see :meth:`MinibatchLogisticAttribution.fit` and the note in
:class:`MinibatchLogisticAttribution` on why this is registered under a name of its own rather
than swapped in behind ``logistic``.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

#: Rows scored per pass in :meth:`MinibatchLogisticAttribution.score`. Bounds the device-side
#: logit block; the output matrix is preallocated whole on the host instead.
SCORE_ROW_BATCH = 8192


class MinibatchLogisticAttribution:
    """Multinomial logistic regression over the known authors, fitted by minibatch Adam.

    A drop-in for :class:`LogisticAttribution` at the estimator contract -- ``fit(embeddings,
    labels)`` then ``score(embeddings)``, authors in ``self.authors``, higher is more likely -- and
    the attack to reach for when the known side enrolls more authors than lbfgs can hold a logit
    matrix for. See the module docstring for where that boundary falls.

    **Registered separately from** ``logistic`` **on purpose.** The two minimise the same
    objective but land in different places: an iterative fit under a fixed epoch budget is an
    approximation, and the results directory name is a contract that says what produced the
    numbers in it. Folding this in behind a size threshold would have made one attack name mean
    two different fits depending on the corpus, and a swe-chat number would silently stop being
    comparable to a WildChat one.

    Determinism: the shuffling is seeded, so a CPU fit is reproducible bit for bit. A CUDA fit is
    reproducible only up to the reduction order of the matrix products, which moves the last few
    digits of a logit and, very occasionally, a tie.

    Parameters
    ----------
    C
        Inverse L2 regularisation strength, as in sklearn: the penalty is ``||W||^2 / (2C)``.
    class_weight
        ``"balanced"`` weights each author's documents by ``n / (n_authors * count)``, so a user
        with 300 documents does not dominate the objective over one with 3; ``None`` leaves the
        document counts in as the informative prior they partly are.
    steps
        Total gradient steps. **The budget is in steps rather than epochs**, and that is not
        cosmetic: what an "epoch" means depends on corpus size relative to the batch size, so an
        epoch budget would silently spend a different amount of optimisation on different corpora.
        A fit costs ``steps x batch_size x n_features x n_authors`` and nothing else, so this is
        also the only dial that changes what a fit costs.
    batch_size
        Documents per gradient step. Also what bounds the logit matrix; lower it before lowering
        anything else if a device runs out of memory. Note the two parameters interact: halving
        this halves the cost of a step and the amount of data each one sees.
    learning_rate
        Adam's initial step, decayed to zero on a cosine schedule over ``epochs``. The default is
        tuned for **standardized** features (``run_experiment.py --standardize``, the default);
        on raw StyloMetrix columns, which mix ratios in [0, 1] with raw counts, it is far too
        large for the wide columns and far too small for the narrow ones.
    device
        ``"auto"`` (default) uses CUDA when a device is visible, else the CPU. The CPU path works
        but is substantially slower on a large author pool.
    tf32
        Let CUDA run the two matrix products on Ampere-or-later tensor cores (default: on). This
        is the single largest speed lever here, since a fit is essentially one matmul chain and
        nothing else, and PyTorch otherwise leaves those units idle. It costs mantissa bits, which
        is why it is a named parameter rather than something switched on silently, but the loss of
        precision does not move the fitted model's predictions in practice. Ignored on the CPU,
        and set and restored around the fit rather than left mutated, since it is process-global
        state this class does not own.
    seed
        Seeds the minibatch shuffling.
    """

    name = "logistic_sgd"

    def __init__(self, C: float = 1.0, class_weight: str | None = "balanced",
                 steps: int = 2000, batch_size: int = 8192, learning_rate: float = 0.05,
                 device: str = "auto", tf32: bool = True, seed: int = 0):
        self.C = C
        self.class_weight = class_weight
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.device = device
        self.tf32 = tf32
        self.seed = seed

    @contextmanager
    def _matmul_precision(self, torch, device):
        """Enable TF32 matmuls for the duration of a fit, then put the global flag back."""
        if device.type != "cuda" or not self.tf32:
            yield
            return
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    # torch is imported inside the methods, not at module scope, so that importing
    # `prompt_anonymity.attacks` -- which every run does, for the registry -- does not pull in a
    # deep-learning stack for the attacks that have no use for one.
    def _torch(self):
        import torch
        return torch

    def _resolve_device(self, torch):
        if self.device != "auto":
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fit(self, embeddings, labels):
        """Fit the softmax over the known authors.

        Agrees closely with the exact (lbfgs) fit on corpora small enough to run both, both in
        closed-set accuracy and in out-of-set detection AUROC; residual disagreement is
        concentrated in documents whose top two authors are near-tied, and does not shrink with a
        longer step budget past a point.

        ``C`` is the one setting that matters most and the default is not necessarily the best
        value for a given corpus -- it is worth tuning (``--tune``) rather than trusted as-is.
        """
        torch = self._torch()
        device = self._resolve_device(torch)

        self.authors, codes = np.unique(labels, return_inverse=True)
        n_documents, n_features = embeddings.shape
        n_authors = len(self.authors)

        X = torch.as_tensor(np.ascontiguousarray(embeddings, dtype=np.float32), device=device)
        y = torch.as_tensor(codes.astype(np.int64), device=device)

        if self.class_weight == "balanced":
            counts = np.bincount(codes, minlength=n_authors)
            weights = torch.as_tensor(
                (n_documents / (n_authors * counts)).astype(np.float32), device=device)
        elif self.class_weight is None:
            weights = None
        else:
            raise ValueError(f"class_weight must be 'balanced' or None, got {self.class_weight!r}")

        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        coef = torch.zeros((n_features, n_authors), device=device, requires_grad=True)
        intercept = torch.zeros(n_authors, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([coef, intercept], lr=self.learning_rate)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        # sklearn penalises the summed loss; dividing the whole objective by n_documents makes it
        # a mean loss plus a penalty scaled to match, which is the form a fixed learning rate
        # behaves sensibly under across corpora of very different sizes.
        penalty = 1.0 / (2.0 * self.C * n_documents)
        # A fresh shuffle is drawn whenever the previous one runs out, so the budget is spent as
        # whole passes wherever the known side is larger than a batch and as repeated draws of the
        # whole side where it is not. Nothing here assumes the two are the same thing.
        order, cursor = torch.randperm(n_documents, generator=generator).to(device), 0
        with self._matmul_precision(torch, device):
            for _ in range(self.steps):
                if cursor >= n_documents:
                    order, cursor = torch.randperm(n_documents, generator=generator).to(device), 0
                rows = order[cursor:cursor + self.batch_size]
                cursor += self.batch_size
                logits = X[rows] @ coef + intercept
                loss = torch.nn.functional.cross_entropy(
                    logits, y[rows], weight=weights, reduction="mean")
                loss = loss + penalty * coef.square().sum()
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                schedule.step()
        self.final_loss = float(loss.detach())

        self.coef = coef.detach()
        self.intercept = intercept.detach()
        self.device_ = device
        return self

    def score(self, embeddings):
        """Author logits for each document, ``[n_documents x n_authors]`` float32 on the host.

        Built in row blocks straight into a preallocated host array, so there is never a second
        full copy of the score matrix live on the device at the same time.
        """
        torch = self._torch()
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        out = np.empty((len(embeddings), len(self.authors)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(embeddings), SCORE_ROW_BATCH):
                block = embeddings[start:start + SCORE_ROW_BATCH]
                logits = torch.as_tensor(block, device=self.device_) @ self.coef + self.intercept
                out[start:start + SCORE_ROW_BATCH] = logits.cpu().numpy()
        return out
