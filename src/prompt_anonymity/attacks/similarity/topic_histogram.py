"""Topic-histogram attribution: the attacker for a release that reveals only a topic id.

Built for the content-floor experiment (``run_experiment.py --unknown-snap-k``), where every
unknown conversation is released as the centre of its cluster in a K-topic codebook. A
nearest-neighbour attacker is the wrong tool there: it matches a centre to the single closest
known *document*, which with a public codebook is usually one of the documents the codebook was
built from -- someone else's -- so it scores below chance for a reason unrelated to privacy.

This attack is defense-aware instead. The codebook is public, and since every release is exactly
one codebook centre, it can be read straight off the released vectors: the distinct rows of the
queries *are* the codebook (up to the attacker's own standardization, which maps equal rows to
equal rows). Each known document is assigned to its nearest centre, each author gets a smoothed
topic histogram ``P_u(c)``, and a released conversation in topic ``c`` scores ``log P_u(c)``
against every author -- the likelihood-ratio attacker of the "useful => linkable" argument.
"""

from __future__ import annotations

import numpy as np

from ..common import unit_rows


class TopicHistogram:
    """Score each author by how often their known documents fall in the query's topic."""

    name = "topic_histogram"

    def __init__(self, smoothing: float = 1.0, max_topics: int = 8192):
        self.smoothing = smoothing
        self.max_topics = max_topics

    def fit(self, embeddings, labels):
        self.authors, self._codes = np.unique(labels, return_inverse=True)
        self._known = unit_rows(np.asarray(embeddings, dtype=np.float32))
        return self

    def score(self, embeddings):
        queries = np.asarray(embeddings, dtype=np.float32)
        codebook, topic_of_query = np.unique(queries, axis=0, return_inverse=True)
        if len(codebook) > self.max_topics:
            raise ValueError(
                f"{len(codebook):,} distinct query vectors: topic_histogram expects a topic-only "
                f"release (run_experiment.py --unknown-snap-k), not raw embeddings.")
        centres = unit_rows(codebook)

        # Known documents -> nearest public centre, in blocks to bound memory.
        topic_of_known = np.empty(len(self._known), dtype=np.int64)
        for start in range(0, len(self._known), 8192):
            block = self._known[start:start + 8192] @ centres.T
            topic_of_known[start:start + 8192] = block.argmax(axis=1)

        n_topics, n_authors = len(centres), len(self.authors)
        counts = np.zeros((n_authors, n_topics))
        np.add.at(counts, (self._codes, topic_of_known), 1.0)
        prior = (counts.sum(axis=0) + 1.0) / (counts.sum() + n_topics)
        histogram = ((counts + self.smoothing * prior)
                     / (counts.sum(axis=1, keepdims=True) + self.smoothing))
        log_histogram = np.log(histogram).astype(np.float32)   # (authors, topics)
        return log_histogram[:, np.ravel(topic_of_query)].T     # (queries, authors)
