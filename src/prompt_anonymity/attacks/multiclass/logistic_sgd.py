"""Multinomial logistic regression fitted by minibatch gradient descent, for large author pools.

Same *model* as :class:`~prompt_anonymity.attacks.multiclass.logistic.LogisticAttribution` -- one
softmax over the known authors, L2-regularised, optionally class-balanced -- fitted a different
way so that it is runnable when there are tens of thousands of authors.

Why the sklearn path stops working
----------------------------------
``LogisticRegression``'s multinomial loss materialises the full ``[n_documents x n_authors]``
matrix of logits **and** the matrix of probabilities, both in float64, on every lbfgs iteration.
That is fine at swe-chat's 124 authors and fatal at WildChat's:

===============  ==========  ==========  =================  ==========================
known config     documents   authors     logit matrix       measured / projected fit
===============  ==========  ==========  =================  ==========================
swe-chat 0075    3,250       124         3 MB               2 s
wildchat 0025    43,127      7,456       2.6 GB             ~30 min, ~8 GB peak
wildchat 0075    129,382     19,711      20.4 GB            ~4 h, >40 GB peak
===============  ==========  ==========  =================  ==========================

The 20.4 GB is one array of two that are live at once, against a 16 GB job cap, so the largest
configuration does not merely run slowly -- it cannot run at all. Measured scaling behind the
projection, on wildchat StyloMetrix with the author pool subsampled: 26 s at 250 authors, 34 s at
500, 64 s at 1,000, 158 s at 2,000, i.e. the per-iteration cost tracks ``n_documents x n_authors``
once the fixed overhead is paid, and both factors grow together as the known side widens.

What this does instead
----------------------
Minibatching bounds the logit matrix at ``[batch_size x n_authors]`` regardless of corpus size --
646 MB at the default 8,192 rows against 19,711 authors, in float32 -- and puts the two matrix
products (``X W`` forward, ``X^T dZ`` backward) somewhere they are cheap. On one A100 the whole
fit is a couple of minutes where lbfgs projects to hours, because 2 TFLOP per epoch is a few
tenths of a second of tensor-core time and nothing about the shape is awkward for a GPU: the
weight matrix is ``[n_features + 1 x n_authors]``, 15 MB at WildChat's largest pool, so the model
and its optimiser state fit in a corner of any card.

The objective is written to be the *same function* sklearn minimises, so ``C`` means what it means
there and :data:`~run_experiment.HYPERPARAMETER_SPACES` transfers unchanged::

    sum_i weight_i * cross_entropy(x_i W + b, y_i)  +  ||W||^2 / (2 C)

with the intercept unregularised, and ``weight_i`` from ``class_weight="balanced"`` as
``n_documents / (n_authors * count(y_i))``. It is optimised rather than solved, so it lands *near*
sklearn's optimum rather than on it -- see :meth:`MinibatchLogisticAttribution.fit` for the
measured agreement and for why this is registered under a name of its own rather than swapped in
behind ``logistic``.
"""

from __future__ import annotations

from contextlib import contextmanager

import numpy as np

#: Rows scored per pass in :meth:`MinibatchLogisticAttribution.score`. Bounds the device-side
#: logit block (rows x n_authors x 4 B, 646 MB at the default against 19,711 authors); the output
#: matrix is preallocated whole on the host, because that one is the caller's documented cost.
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
        cosmetic: an epoch is one step on swe-chat (2,992 known documents, under a single batch)
        and sixteen on WildChat's largest known side, so an epoch budget silently means two
        different amounts of optimisation on the two corpora -- measured, 60 "epochs" reproduced
        the exact fit's predictions 78.8% of the time on swe-chat against 98.5% at 300 steps.
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
        and is what the swe-chat validation below was run on; it is roughly two orders of
        magnitude slower on a WildChat-sized pool.
    tf32
        Let CUDA run the two matrix products on Ampere-or-later tensor cores (default: on).
        **This is the single largest speed lever here**, because a fit is essentially one matmul
        chain and nothing else: PyTorch ships with ``allow_tf32 = False`` for matmul, which leaves
        those units idle -- ~4.5 fp32 TFLOPS against ~9 TF32 on an A16, and 19.5 against 156 on an
        A100. It costs mantissa bits (10 against 23), which is why it is a named parameter rather
        than something switched on silently -- but measured on WildChat's ``known0075`` (129,382
        documents, 19,711 authors, one A16) it is **1.47x faster** (217.5 s -> 147.7 s) and the
        numbers do not move: top-1 0.1440 against 0.1441, **99.91% identical predictions**, final
        loss agreeing to four significant figures. Hence the default. Ignored on the CPU, and set
        and restored around the fit rather than left mutated, since it is process-global state
        this class does not own.
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

        **Measured agreement with the exact fit**, swe-chat / StyloMetrix / standardized, both
        attacks run through ``run_experiment.py`` at matched settings, over all six known
        configurations: closed-set top-1 within **0.002** everywhere (0.2564 against 0.2576 at
        ``known0075``, 0.3965 against 0.3965 at ``known5075``) and out-of-set detection AUROC
        within **0.011**. At the estimator level on ``known0075``, identical top-1 to four
        decimals from 300 steps upward, agreeing with lbfgs on 98.5% of individual predictions at
        300 steps and 99.3% at 1,000, against lbfgs's 28.4 s. The residual disagreement is
        documents whose top two authors are near-tied and does not shrink with a longer budget:
        10,000 steps agrees no better than 1,000. Insensitive to ``learning_rate`` over 0.02-0.1,
        all three landing on the same top-1 and the same final loss to four decimals.

        **``C`` is the one setting that does matter, and the default is not the best value.**
        Swept on WildChat ``known0075`` (19,711 authors): top-1 runs 0.0977 at C = 0.002, 0.1231
        at 0.01, 0.1394 at 0.05, **0.1440 at 0.2**, 0.1387 at the C = 1.0 default and 0.1322 at
        5.0 -- a clean unimodal curve whose peak sits below 1.0, the same direction the search
        picks for :class:`LogisticAttribution` on swe-chat (C = 0.11 in four of six
        configurations, never 1.0). The default costs ~0.005 of top-1 there, so an untuned run is
        a mild floor rather than a wrong answer. Note 0.1440 is an *oracle* figure, read off the
        test set; ``--tune`` picks on the known side and can only do as well or worse.
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

        Built in row blocks straight into a preallocated host array. The result is the caller's
        documented memory cost -- 3.4 GB at WildChat's largest configuration -- and there is no
        reason for a second copy of it to exist on the device at the same time.
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
