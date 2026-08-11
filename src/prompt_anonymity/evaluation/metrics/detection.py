"""Open-set metrics: *is this document's author known at all*, and can the score be trusted?

Identification asks which of the known users wrote a document. In a real corpus most anonymous
documents were written by somebody the attacker has never seen, so a usable attack also has to
**reject**: decide that the best match, however good it looks, is not good enough. That decision
is a detection problem with its own vocabulary, borrowed from speaker/author verification and
biometrics, and it is scored separately here because it can succeed or fail independently of
identification -- an attack can rank the right author first almost every time and still be
unable to tell an enrolled user from a stranger.

Everything takes a **rejection score, where higher means more likely out-of-set**. See
:func:`prompt_anonymity.evaluation.metrics.detection.max_softmax_confidence` for turning a raw score matrix
into the complementary confidence.

Reading order
-------------
1. :func:`detection_auroc` -- threshold-free: does the score separate the two populations *at
   all*? Below ~0.55 no threshold on it will beat trivially rejecting everything, and nothing
   further down this list is worth reading.
2. :func:`equal_error_rate` -- the standard single-number operating point, no cost assumptions.
3. :func:`detection_identification_rate` -- the one that matters for an attack, because it
   requires the document to be both accepted *and* attributed to the right author.
4. :func:`c_at_1` and :func:`calibration_metrics` -- whether abstaining was used well, and
   whether the reported confidence means anything.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def max_softmax_confidence(scores) -> np.ndarray:
    """Per-document confidence in the top-ranked author: the maximum softmax probability.

    The usual way to read a probability off an attribution model. For a multinomial logistic
    attack whose scores are class logits this is exactly ``predict_proba(...).max(axis=1)``; for
    a distance- or similarity-based scorer it is a monotone rescaling rather than a real
    posterior, which is precisely why :func:`calibration_metrics` is worth running before
    treating it as one.

    Computed in row blocks, because the obvious ``softmax(scores).max(axis=1)`` needs two full
    float64 copies of the score matrix to return one number per document -- over 10 GB at
    86,000 documents against 7,500 authors, which is enough to lose a 16 GB job. Only the top
    probability is wanted, and it has a closed form that never materialises the rest:
    ``max_j softmax(s)_j = 1 / sum_j exp(s_j - max_j s)``.
    """
    scores = np.asarray(scores)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (n_documents, n_candidates); got {scores.shape}.")

    confidence = np.empty(len(scores))
    block_rows = max(1, 2 ** 28 // max(scores.shape[1] * 8, 1))     # ~256 MB per block
    for start in range(0, len(scores), block_rows):
        block = scores[start:start + block_rows].astype(np.float64)
        block -= block.max(axis=1, keepdims=True)
        np.exp(block, out=block)
        confidence[start:start + block_rows] = 1.0 / block.sum(axis=1)
    return confidence


def detection_auroc(rejection_scores, is_out_of_set) -> float:
    """Area under the ROC for separating out-of-set documents from in-set ones.

    Threshold-free, so it is the first thing to read: 0.5 means the score says nothing about
    membership, and **below** 0.5 means it is anti-correlated -- out-of-set documents look
    *more* familiar than in-set ones, which happens when the nearest-candidate distance is
    driven by document length or genericness rather than by authorship.

    Returns NaN when either class is absent, since the quantity is undefined then rather than
    zero. Wraps scikit-learn's ``roc_auc_score`` with out-of-set as the positive class.
    """
    is_out_of_set = np.asarray(is_out_of_set, dtype=bool)
    if is_out_of_set.all() or not is_out_of_set.any():
        return float("nan")
    return float(roc_auc_score(is_out_of_set, np.asarray(rejection_scores, dtype=float)))


def equal_error_rate(rejection_scores, is_out_of_set) -> tuple[float, float]:
    """Equal error rate and the threshold that achieves it.

    The operating point where the two mistakes balance: the false-accept rate (out-of-set
    documents let through) equals the false-reject rate (in-set documents thrown away). The
    verification literature's default summary because it needs no assumption about the relative
    cost of the two errors -- unlike accuracy, which silently assumes the corpus's own
    out-of-set proportion is the one you care about.

    Returns ``(eer, threshold)``, both NaN when either class is absent. Computed from
    scikit-learn's ``roc_curve``: the ROC is a step function, so the two rates rarely cross
    exactly and the value is linearly interpolated between the two points that bracket the
    crossing.
    """
    rejection_scores = np.asarray(rejection_scores, dtype=float)
    is_out_of_set = np.asarray(is_out_of_set, dtype=bool)
    if is_out_of_set.all() or not is_out_of_set.any():
        return float("nan"), float("nan")

    false_accept, true_accept, thresholds = roc_curve(is_out_of_set, rejection_scores)
    false_reject = 1.0 - true_accept
    difference = false_accept - false_reject
    crossing = int(np.argmin(np.abs(difference)))
    # Interpolate between the bracketing points when the curve steps over the crossing.
    neighbour = crossing + (1 if difference[crossing] < 0 else -1)
    if 0 <= neighbour < len(difference) and difference[neighbour] * difference[crossing] < 0:
        weight = abs(difference[crossing]) / abs(difference[crossing] - difference[neighbour])
        eer = ((1 - weight) * (false_accept[crossing] + false_reject[crossing])
               + weight * (false_accept[neighbour] + false_reject[neighbour])) / 2
        threshold = (1 - weight) * thresholds[crossing] + weight * thresholds[neighbour]
    else:
        eer = (false_accept[crossing] + false_reject[crossing]) / 2
        threshold = thresholds[crossing]
    return float(eer), float(threshold)


def detection_identification_rate(rejection_scores, correct, is_out_of_set, far: float = 0.10) -> float:
    """DIR@FAR: in-set documents both accepted **and** correctly attributed, at a fixed
    false-alarm rate.

    The open-set identification summary, and the number that actually describes an attack:
    accepting a document is worthless if the author named is the wrong one, so unlike
    :func:`detection_auroc` this metric fails a document for either mistake. The threshold is
    read off the out-of-set scores, so ``far=0.10`` means "the accuracy an attacker gets if they
    tolerate wrongly accepting 10% of strangers".

    Parameters
    ----------
    rejection_scores : array-like of shape (n_documents,)
        Higher = more likely out-of-set.
    correct : array-like of shape (n_documents,)
        Whether the top-ranked author is the true one. Only read for in-set documents.
    is_out_of_set : array-like of shape (n_documents,)
        Whether each document's author is absent from the candidate set.
    far : float, default 0.10
        Tolerated false-accept rate on the out-of-set documents.
    """
    rejection_scores = np.asarray(rejection_scores, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    is_out_of_set = np.asarray(is_out_of_set, dtype=bool)
    if is_out_of_set.all() or not is_out_of_set.any():
        return float("nan")
    threshold = float(np.percentile(rejection_scores[is_out_of_set], far * 100))
    return float(((rejection_scores <= threshold) & correct)[~is_out_of_set].mean())


def c_at_1(correct, answered) -> float:
    """PAN's c@1: accuracy that rewards abstaining over guessing wrong.

    From the PAN authorship-verification shared tasks. An unanswered document is credited with
    the accuracy achieved on the answered ones::

        c@1 = (n_correct + n_unanswered * n_correct / n) / n

    so declining to answer is worth exactly the system's own average -- better than a wrong
    answer, worse than a right one. That makes it the natural score for a reject option: plain
    accuracy punishes abstention as harshly as error, while open-set accuracy pays for
    abstention on out-of-set documents and can look respectable while re-identifying nobody.

    Apply it to the **in-set** documents, where abstaining is genuinely neither right nor wrong.
    On out-of-set documents rejection is the correct answer rather than an abstention, and
    belongs in an accuracy that scores it as such.

    Parameters
    ----------
    correct : array-like of shape (n_documents,)
        Whether the attributed author is the true one (ignored where ``answered`` is False).
    answered : array-like of shape (n_documents,)
        Whether the attack committed to an author rather than rejecting.
    """
    correct = np.asarray(correct, dtype=bool)
    answered = np.asarray(answered, dtype=bool)
    n = len(correct)
    if n == 0:
        return float("nan")
    n_correct = int((correct & answered).sum())
    n_unanswered = int((~answered).sum())
    return float((n_correct + n_unanswered * n_correct / n) / n)


def selective_classification(confidence, correct,
                             coverages=(0.10, 0.25, 0.50, 0.75, 1.00)) -> pd.DataFrame:
    """Precision when the attack answers only its most confident documents.

    The trade-off Narayanan et al. reported as the real result of their 100,000-author attack:
    top-1 accuracy was ~20%, but *choosing when to speak* lifted precision past 80% while still
    identifying half of the authors it would have identified by always guessing. An attack that
    is usually wrong but knows when it is right is a far sharper privacy threat than its headline
    accuracy suggests, and no single-threshold number shows that -- hence a curve.

    This is the closed-set companion to
    :func:`detection_identification_rate`. DIR@FAR asks "of the documents whose author is
    enrolled, how many are accepted *and* correct, while holding impostor acceptance to a fixed
    rate" -- an open-set question that needs out-of-set documents to define its false alarm rate.
    This function asks the simpler one that applies to in-set documents alone: rank by confidence,
    answer the top fraction, and report how often those answers are right.

    Parameters
    ----------
    confidence : array-like of shape (n_documents,)
        Per-document confidence in the top-ranked author, higher = more confident. Any monotone
        confidence works: :func:`max_softmax_confidence`, or the negated
        :func:`~prompt_anonymity.attacks.rejection.rejection_score`, whose cohort-normalised
        margin is a variant of the "gap statistic" that paper used.
    correct : array-like of shape (n_documents,)
        Whether the top-ranked author is the true one.
    coverages : sequence of float
        Fractions of documents to answer, each in (0, 1].

    Returns
    -------
    pandas.DataFrame
        One row per coverage, with columns ``coverage`` (requested), ``n_answered``,
        ``precision`` (accuracy among answered) and ``recall`` -- correct answers retained as a
        fraction of those made at full coverage, which is the sense in which the original reports
        "50% recall". ``recall`` is 1.0 at full coverage by construction.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if len(confidence) != len(correct):
        raise ValueError(
            f"confidence has {len(confidence)} entries but correct has {len(correct)}."
        )

    # Descending confidence; ties broken arbitrarily but deterministically.
    order = np.argsort(-confidence, kind="stable")
    ordered_correct = correct[order]
    cumulative_correct = np.cumsum(ordered_correct)
    total_correct = int(correct.sum())

    rows = []
    for coverage in coverages:
        n_answered = max(1, int(round(coverage * len(correct)))) if len(correct) else 0
        n_answered = min(n_answered, len(correct))
        if n_answered == 0:
            rows.append({"coverage": coverage, "n_answered": 0,
                         "precision": float("nan"), "recall": float("nan")})
            continue
        n_correct = int(cumulative_correct[n_answered - 1])
        rows.append({
            "coverage": coverage,
            "n_answered": n_answered,
            "precision": n_correct / n_answered,
            "recall": n_correct / total_correct if total_correct else float("nan"),
        })
    return pd.DataFrame(rows)


def calibration_metrics(confidence, correct, n_bins: int = 10) -> dict:
    """Does the reported confidence mean what it says?

    Every threshold in an open-set pipeline assumes the score is meaningful, but a model can
    rank perfectly while being badly overconfident -- and an overconfident model makes a
    threshold calibrated on one window transfer poorly to the next. These numbers say whether
    that assumption holds.

    Parameters
    ----------
    confidence : array-like of shape (n_documents,)
        Claimed probability that the top-ranked author is correct, e.g. from
        :func:`max_softmax_confidence`.
    correct : array-like of shape (n_documents,)
        Whether it actually was.
    n_bins : int, default 10
        Equal-width bins over ``[0, 1]`` for the expected/maximum calibration error.

    Returns
    -------
    dict
        ``brier``
            Brier score, scikit-learn's ``brier_score_loss``: mean squared error of the
            probability. Lower is better; it scores sharpness and calibration together, so a
            model that always says 0.5 scores poorly even though it is perfectly calibrated.
        ``expected_calibration_error``
            Mean absolute gap between confidence and accuracy within a bin, weighted by bin
            occupancy. 0 = perfectly calibrated. scikit-learn has no ECE (its
            ``calibration_curve`` returns the reliability diagram without bin weights), so the
            binning is done here.
        ``max_calibration_error``
            The worst bin's gap -- the tail ECE averages away.
        ``mean_confidence`` / ``accuracy``
            Their difference is the overall bias: positive means overconfident.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=bool)
    if confidence.size == 0:
        return dict.fromkeys(
            ("brier", "expected_calibration_error", "max_calibration_error",
             "mean_confidence", "accuracy"), float("nan"))

    # np.digitize with right=True puts a confidence of exactly 0 in bin 0 and 1.0 in the last.
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)
    gaps, weights = [], []
    for index in range(n_bins):
        in_bin = bin_index == index
        if not in_bin.any():
            continue
        gaps.append(abs(correct[in_bin].mean() - confidence[in_bin].mean()))
        weights.append(in_bin.mean())
    # brier_score_loss needs both outcomes present to infer the positive label; say so explicitly.
    brier = float(brier_score_loss(correct, confidence, pos_label=True))
    return {
        "brier": brier,
        "expected_calibration_error": float(np.average(gaps, weights=weights)) if gaps else float("nan"),
        "max_calibration_error": float(np.max(gaps)) if gaps else float("nan"),
        "mean_confidence": float(confidence.mean()),
        "accuracy": float(correct.mean()),
    }
