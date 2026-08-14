"""Metric learning for the clustering attack: fit a space on labelled history, cluster in it.

A clustering attack needs no labels *at attack time* -- that is what makes it a distinct threat
model. It does not follow that the attacker has no labels **at all**. In this project's own
threat model the attacker already holds a labelled known side (``run_clustering.py`` uses it to
choose hyper-parameters), and a real attacker scraping their own logs holds one too. Everything
here spends that supervision on the *representation* rather than on the algorithm, which is the
one place it can help without turning linkability back into identifiability:

    A learned projection is fitted on documents by authors the collection under attack may not
    contain, and is then applied to every document blind. No document in the collection is
    labelled, no author in it is enrolled, and the number of clusters is still not an input.

Why this and not a better clustering algorithm
-----------------------------------------------
Measured on WildChat's tuning slice, ``connected``'s neighbour graph carries same-author edges at
roughly a third precision, and the giant cluster it produces is not a chain through a handful of
bad edges but a large region at that precision throughout. No cut of such a graph recovers the
partition, because the ordering of the edges is what is wrong. Every transform here changes that
ordering; nothing here changes how the graph is cut.

The three, in increasing order of what they assume
--------------------------------------------------
=====================  =================================================================
:func:`fit_wccn`       Within-class covariance normalisation. Whitens by the covariance
                       of an author's *own* scatter, so directions along which one person
                       varies stop counting as evidence they are two. Closed form, no
                       hyper-parameter but the shrinkage, and it is the standard opener in
                       speaker verification for exactly this reason
:func:`fit_lda`        The same whitening plus a projection onto the directions that
                       separate authors best. Adds an output dimension to choose
:func:`fit_contrastive`  A linear map trained so that cosine after projection *is* a
                       same-author score. Assumes the most, can express the most, and is
                       the only one of the three that needs a GPU to be comfortable
=====================  =================================================================

All three return a :class:`LinearProjection`, so the attack pipeline downstream is identical and
a run records which was used as one string.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..common import inverse_sqrt, unit_rows

#: Authors with a single document contribute nothing to a within-author scatter estimate (their
#: centred rows are exactly zero) but do inflate the class count the estimate divides by. They are
#: dropped from every fit here; they are still clustered like anything else.
MIN_DOCUMENTS_PER_AUTHOR = 2

#: Rows per chunk when accumulating a ``d x d`` scatter matrix. At 3,072 dimensions a full
#: float64 copy of WildChat's history side is 2.1 GB, and the covariance accumulation wants
#: another; chunking holds the peak to this many rows instead, at no cost in accuracy since the
#: accumulator is float64 throughout.
SCATTER_CHUNK = 8192


@dataclass
class LinearProjection:
    """A learned map ``x -> unit(x @ matrix)``, with the name of whatever fitted it.

    Attributes
    ----------
    matrix : numpy.ndarray of shape (n_features, n_components)
        Applied on the right, so ``transform`` is one GEMM.
    name : str
        Recorded in the results table. A projection whose provenance is not in the row it produced
        is not reproducible, and these differ only in a matrix.
    """

    matrix: np.ndarray
    name: str = "identity"

    @property
    def n_components(self) -> int:
        return self.matrix.shape[1]

    def transform(self, embeddings: np.ndarray) -> np.ndarray:
        """Project and L2-normalise, in float32.

        Normalisation is part of the projection rather than the caller's business: every consumer
        compares with cosine, and a projection that changed vector norms would silently change
        what a euclidean fallback means.
        """
        projected = np.asarray(embeddings, dtype=np.float32) @ self.matrix.astype(np.float32)
        return unit_rows(projected).astype(np.float32)


def identity_projection(n_features: int) -> LinearProjection:
    """The no-op, so "no projection" is a value of the same axis rather than a branch."""
    return LinearProjection(np.eye(n_features, dtype=np.float32), "identity")


def _author_codes(author_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Integer-code the authors and return the mask of those with enough documents to fit on."""
    _, codes, counts = np.unique(np.asarray(author_ids), return_inverse=True, return_counts=True)
    codes = codes.ravel()
    return codes, counts[codes] >= MIN_DOCUMENTS_PER_AUTHOR


#: Scatter matrices memoised by the content of what they were computed from. Both closed-form
#: fitters sweep a hyper-parameter (``shrinkage``, ``n_components``) that is applied *after* the
#: accumulation, so without this a six-point sweep pays the 1.6 TFLOP pass six times for six
#: identical results. Keyed by digest rather than by object identity because the caller slices a
#: fresh array out of the corpus on every run.
_SCATTER_CACHE: dict[str, tuple[np.ndarray, np.ndarray, int]] = {}


def _scatter_digest(embeddings: np.ndarray, codes: np.ndarray) -> str:
    """Content key for :data:`_SCATTER_CACHE`."""
    import hashlib

    digest = hashlib.blake2b(digest_size=16)
    for array in (np.ascontiguousarray(embeddings), np.ascontiguousarray(codes)):
        digest.update(str(array.shape).encode())
        digest.update(array.view(np.uint8))
    return digest.hexdigest()


def _scatter_matrices(embeddings: np.ndarray, codes: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, int]:
    """Pooled within-author and between-author scatter, accumulated in chunks.

    Returns ``(within, between, n_authors)``. The between matrix weights each author by their
    document count, which is what makes the pair a decomposition at all: the *unnormalised*
    scatters then satisfy ``S_total = S_within + S_between`` exactly, so a direction is either
    within-author variation or between-author variation and never partly counted twice.

    The two are returned with **different denominators** -- ``N - n_authors`` and ``N - 1``, each
    the unbiased one for its own quantity -- so they do not themselves sum to the total covariance.
    That is deliberate and harmless for both callers: :func:`fit_wccn` uses only ``within``, and
    :func:`fit_lda` takes eigenvectors of ``W^-1/2 B W^-1/2``, which a constant rescaling of ``B``
    cannot move (it scales every eigenvalue alike and leaves the ordering and the directions
    untouched).

    Memoised on the content of its inputs -- see :data:`_SCATTER_CACHE`.
    """
    key = _scatter_digest(embeddings, codes)
    if key in _SCATTER_CACHE:
        return _SCATTER_CACHE[key]
    embeddings = np.asarray(embeddings, dtype=np.float32)
    n_features = embeddings.shape[1]
    n_authors = int(codes.max()) + 1

    sums = np.zeros((n_authors, n_features), dtype=np.float64)
    np.add.at(sums, codes, embeddings)
    counts = np.bincount(codes, minlength=n_authors).astype(np.float64)
    means = sums / counts[:, None]
    grand = sums.sum(axis=0) / counts.sum()

    within = np.zeros((n_features, n_features), dtype=np.float64)
    for start in range(0, len(embeddings), SCATTER_CHUNK):
        block = embeddings[start:start + SCATTER_CHUNK].astype(np.float64)
        block -= means[codes[start:start + SCATTER_CHUNK]]
        within += block.T @ block
    within /= max(len(embeddings) - n_authors, 1)

    centred_means = (means - grand) * np.sqrt(counts)[:, None]
    between = centred_means.T @ centred_means / max(len(embeddings) - 1, 1)
    _SCATTER_CACHE[key] = (within, between, n_authors)
    return within, between, n_authors


def fit_wccn(embeddings: np.ndarray, author_ids: np.ndarray, *,
             shrinkage: float = 0.1) -> LinearProjection:
    """Within-class covariance normalisation, fitted on labelled history.

    The matrix plain cosine ignores. If a single author's documents scatter widely along some
    direction -- topic drift, the language they happened to write in that week, conversation length
    -- then agreement along that direction is weak evidence of shared authorship, and disagreement
    along it is weak evidence against. Whitening by the pooled within-author covariance rescales
    every direction by how much one person moves along it, which is the closest thing to a closed
    form for "discount the nuisance".

    ``shrinkage`` pulls the estimate toward a scaled identity. It is not optional at these shapes:
    3,072 dimensions estimated from 86,255 documents is only a 28:1 ratio, and the smallest
    eigenvalues of such an estimate are badly biased downward -- exactly the ones the inverse
    square root then multiplies up. Sweep it; the useful range is wide.
    """
    codes, usable = _author_codes(author_ids)
    embeddings = np.asarray(embeddings, dtype=np.float32)[usable]
    codes = np.unique(codes[usable], return_inverse=True)[1].ravel()
    within, _, _ = _scatter_matrices(embeddings, codes)
    scale = np.trace(within) / within.shape[0]
    within = (1 - shrinkage) * within + shrinkage * scale * np.eye(within.shape[0])
    return LinearProjection(inverse_sqrt(within).astype(np.float32), f"wccn(shrinkage={shrinkage})")


def fit_lda(embeddings: np.ndarray, author_ids: np.ndarray, *, n_components: int = 256,
            shrinkage: float = 0.1) -> LinearProjection:
    """Whiten by within-author scatter, then keep the directions that separate authors best.

    Solves the generalised eigenproblem ``B v = lambda W v`` the numerically sane way -- whiten
    with ``W^{-1/2}``, take the top eigenvectors of the whitened between-author scatter, and
    compose -- rather than inverting ``W`` and eigendecomposing a non-symmetric product.

    This is :func:`fit_wccn` plus a rank truncation, and the truncation is the point: the whitened
    space has 3,072 directions and only the leading few hundred carry any author separation, so
    keeping all of them means the distance is mostly noise that has been amplified to unit
    variance. ``n_components`` is the one real knob and it wants sweeping.
    """
    codes, usable = _author_codes(author_ids)
    embeddings = np.asarray(embeddings, dtype=np.float32)[usable]
    codes = np.unique(codes[usable], return_inverse=True)[1].ravel()
    within, between, n_authors = _scatter_matrices(embeddings, codes)
    scale = np.trace(within) / within.shape[0]
    within = (1 - shrinkage) * within + shrinkage * scale * np.eye(within.shape[0])

    whitener = inverse_sqrt(within)
    values, vectors = np.linalg.eigh(whitener @ between @ whitener)
    keep = min(n_components, n_authors - 1, embeddings.shape[1])
    leading = vectors[:, np.argsort(values)[::-1][:keep]]
    return LinearProjection((whitener @ leading).astype(np.float32),
                            f"lda(n_components={keep}, shrinkage={shrinkage})")


def _author_centroids(model, features, starts, counts, device, sample: int = 8):
    """Mean projected vector per author, over up to ``sample`` of their documents.

    Used only to *build batches* (see ``hard_negatives``), never to score anything, so a sample is
    enough and the whole point is that it is cheap enough to refresh during training.
    """
    import torch

    offsets = np.minimum(np.arange(sample)[None, :], (counts - 1)[:, None])
    rows = torch.from_numpy((starts[:, None] + offsets).ravel()).to(device)
    with torch.no_grad():
        projected = torch.nn.functional.normalize(model(features[rows]), dim=1)
        projected = projected.view(len(starts), sample, -1).mean(dim=1)
        return torch.nn.functional.normalize(projected, dim=1)


def fit_contrastive(embeddings: np.ndarray, author_ids: np.ndarray, *, n_components: int = 512,
                    steps: int = 3000, authors_per_batch: int = 512, learning_rate: float = 1e-3,
                    temperature: float = 0.05, weight_decay: float = 1e-4,
                    hidden: int = 0, hard_negatives: int = 0, seed: int = 20260813,
                    device: str | None = None, verbose: bool = True) -> LinearProjection:
    """Train a projection so that cosine in the output space *is* a same-author score.

    Supervised contrastive (NT-Xent with in-batch negatives): each step draws
    ``authors_per_batch`` authors, two documents from each, and maximises the agreement of each
    pair against every other document in the batch. That is a direct optimisation of the quantity
    the neighbour graph is built from, where :func:`fit_wccn` and :func:`fit_lda` optimise proxies
    for it.

    Two deliberate limits. **The default is a single linear map**, not a network: the attacker is
    re-weighting an embedding somebody else trained, and 3,072 x 512 is already 1.6M parameters
    from ~13,000 authors. ``hidden > 0`` adds one ReLU layer for the comparison, and it is a
    comparison worth making rather than an obvious upgrade. **Negatives are in-batch only**, which
    with 1,024 documents per step is a 1,023-way problem -- far harder than the pairwise one and
    cheap, but it does mean the loss never sees the full 7,867-author field the attack faces.

    Returns a :class:`LinearProjection` when ``hidden == 0``. A hidden layer is not a linear map,
    so that variant returns its collapsed first layer and warns -- it is for measuring whether
    depth would buy anything, not for use.
    """
    import torch

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = np.random.default_rng(seed)
    torch.manual_seed(seed)

    codes, usable = _author_codes(author_ids)
    embeddings = np.asarray(embeddings, dtype=np.float32)[usable]
    codes = np.unique(codes[usable], return_inverse=True)[1].ravel()
    n_authors = int(codes.max()) + 1

    # Contiguous per-author blocks, so drawing two documents from an author is two integers into a
    # slice rather than a mask over the whole history side once per author per step.
    order = np.argsort(codes, kind="stable")
    embeddings, codes = embeddings[order], codes[order]
    starts = np.searchsorted(codes, np.arange(n_authors))
    counts = np.bincount(codes, minlength=n_authors)

    features = torch.from_numpy(embeddings).to(device)
    n_features = features.shape[1]
    if hidden:
        model = torch.nn.Sequential(torch.nn.Linear(n_features, hidden), torch.nn.ReLU(),
                                    torch.nn.Linear(hidden, n_components)).to(device)
    else:
        model = torch.nn.Linear(n_features, n_components, bias=False).to(device)
        # Start at a scaled identity rather than at random: the input is already a good space, so
        # the run begins at "cosine on the original embedding" and can only be judged against it.
        with torch.no_grad():
            model.weight.copy_(torch.eye(n_components, n_features, device=device))
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    batch_authors = min(authors_per_batch, n_authors)
    started = time.perf_counter()
    neighbor_authors = None
    for step in range(steps):
        # Hard negatives. With random batches, 511 of the 512 negatives are authors the model
        # already separates easily, so almost every step's gradient comes from the handful it does
        # not -- and the attack's actual failure mode is confusing *similar* authors. Building the
        # batch out of one seed author and its nearest neighbours in the current projected space
        # makes every negative a near-miss. The neighbour table is refreshed every
        # ``hard_negatives`` steps rather than every step, because it moves slowly and rebuilding
        # it is a 13,694 x 13,694 matmul.
        if hard_negatives and step % hard_negatives == 0:
            centroids = _author_centroids(model, features, starts, counts, device)
            width = min(batch_authors, n_authors)
            neighbor_authors = torch.topk(centroids @ centroids.T, width, dim=1).indices.cpu().numpy()
            del centroids
        if neighbor_authors is not None:
            # The seed's own row already begins with itself, so the batch is the seed plus its
            # closest rivals -- exactly the cohort a clustering attack has to tell apart.
            chosen = neighbor_authors[generator.integers(0, n_authors)]
        else:
            chosen = generator.choice(n_authors, size=batch_authors, replace=False)
        first = starts[chosen] + generator.integers(0, counts[chosen])
        second = starts[chosen] + generator.integers(0, counts[chosen])
        # Force the pair to differ wherever the author has the documents for it; an author with
        # exactly two is handled by the modular bump rather than by resampling in a loop.
        same = first == second
        second[same] = starts[chosen][same] + (first[same] - starts[chosen][same] + 1) % counts[chosen][same]

        rows = torch.from_numpy(np.concatenate([first, second])).to(device)
        projected = torch.nn.functional.normalize(model(features[rows]), dim=1)
        logits = projected @ projected.T / temperature
        logits.fill_diagonal_(-torch.inf)
        # Document i's positive is its partner: the two halves are aligned by construction.
        targets = torch.arange(2 * batch_authors, device=device)
        targets = (targets + batch_authors) % (2 * batch_authors)
        loss = torch.nn.functional.cross_entropy(logits, targets)

        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if verbose and (step % 250 == 0 or step == steps - 1):
            print(f"    step {step:>5d}/{steps}  loss={loss.item():.4f}  "
                  f"[{time.perf_counter() - started:.0f}s on {device}]", flush=True)

    if hidden:
        import warnings
        warnings.warn("a hidden layer is not a linear projection; returning the first layer only, "
                      "which is a diagnostic and not the model that was trained.")
        matrix = model[0].weight.detach().cpu().numpy().T
        return LinearProjection(matrix, f"contrastive_mlp(hidden={hidden})")
    matrix = model.weight.detach().cpu().numpy().T
    return LinearProjection(matrix, f"contrastive(n_components={n_components}, steps={steps}, "
                                    f"temperature={temperature}, lr={learning_rate})")


#: Name -> fitting function, mirroring the other registries in the package. ``identity`` is
#: registered so "no projection" is selectable by the same flag as every other option.
PROJECTION_FITTERS = {
    "identity": lambda embeddings, author_ids, **kwargs: identity_projection(embeddings.shape[1]),
    "wccn": fit_wccn,
    "lda": fit_lda,
    "contrastive": fit_contrastive,
}


def fit_projection(name: str, embeddings: np.ndarray, author_ids: np.ndarray,
                   **kwargs) -> LinearProjection:
    """Fit a registered projection by name on labelled history."""
    try:
        fitter = PROJECTION_FITTERS[name]
    except KeyError:
        raise ValueError(f"unknown projection {name!r}; "
                         f"available: {sorted(PROJECTION_FITTERS)}") from None
    return fitter(embeddings, author_ids, **kwargs)
