from __future__ import annotations

import numpy as np

from .base import Featurizer


class StyleDistanceFeaturizer(Featurizer):
    name = "style_distance"
    version = "1"
    metric = "cosine"

    def __init__(self):
        self._model = None

    def featurize(self, texts) -> np.ndarray:
        if self._model is None:
            # Imported lazily so importing the features registry never requires torch/
            # sentence-transformers -- only selecting this featurizer does.
            import torch
            from sentence_transformers import SentenceTransformer

            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(
                "StyleDistance/styledistance",
                device=device
            )

        texts = [t or "" for t in texts]

        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            batch_size=64
        )

        return np.asarray(embeddings)