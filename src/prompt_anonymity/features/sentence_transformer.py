"""Small sentence-transformers encoders, used as EmBad's cross-tokenizer surrogate ensemble.

One generic featurizer covering any repository that ships a sentence-transformers config, plus the
three registered instances built on it. A single class handles all of them because each repository
already declares its own pipeline (pooling, projections, normalization), so reimplementing it here
would only let it drift from the model card.

**Why these three.** EmBad's problem is *transfer*: the search steers local encoders while the
target adversary uses one the defender doesn't have, so an ensemble is only useful if its members
can disagree the way the target might. ``harrier_270m`` and ``embeddinggemma_300m`` share a
tokenizer (Gemma 3); ``jina_v5_nano`` uses an unrelated one (EuroBERT). EmBad's actual third member
is ``harrier`` (the Qwen 3 checkpoint in :mod:`~prompt_anonymity.features.harrier`) -- not
``harrier_270m`` here, which shares only the product name.

**Task conditioning.** These models are task-conditioned via a text prefix. All three are set to
their **clustering** task where available, matching :mod:`~prompt_anonymity.features.gemini_embedding`
(the adversary's encoder) so every vector in the pipeline describes the same task. The prefix lives
inside :meth:`render`, so it's part of the string the search steers.
"""

from __future__ import annotations

import numpy as np

from .base import Featurizer

#: Documents per forward pass. A memory knob only -- padding does not change a vector -- so it is
#: excluded from :meth:`params` and runs at different batch sizes share one cache.
DEFAULT_BATCH_SIZE = 32


class SentenceTransformerFeaturizer(Featurizer):
    """Embed each conversation with a sentence-transformers repository, L2-normalized.

    Subclasses set :attr:`repo`, :attr:`name`, and optionally :attr:`prompt` (a text prefix) and
    :attr:`model_kwargs` (passed to the backing model, e.g. a task selector).

    Pair with ``--metric cosine``: :meth:`featurize` normalizes, so cosine is a plain dot product.
    """

    version = "1"

    #: HuggingFace repository id. Also the cache-key identity.
    repo: str = ""
    #: Text prefix prepended to every input, or ``None``. Part of :meth:`render`, so it is inside
    #: the string the search steers.
    prompt: str | None = None
    #: Extra ``model_kwargs`` for the backing model (e.g. ``{"default_task": "clustering"}``).
    model_kwargs: dict | None = None

    def __init__(self, *, max_tokens: int = 2048, batch_size: int = DEFAULT_BATCH_SIZE):
        #: Truncation window. Kept equal across the ensemble's members so each reads the same
        #: amount of a document.
        self.max_tokens = int(max_tokens)
        self.batch_size = max(1, int(batch_size))
        self._model = None

    def params(self) -> dict:
        return {
            "repo": self.repo,
            "prompt": self.prompt,
            "model_kwargs": self.model_kwargs or {},
            "max_tokens": self.max_tokens,
            "pipeline": "sentence_transformers",
        }

    def checkpoint(self) -> str:
        """The repository id. Resolution and caching are sentence-transformers' own."""
        return self.repo

    def render(self, text: str) -> str:
        """The exact string handed to the model, task prefix included."""
        return f"{self.prompt}{text}" if self.prompt else text

    def load(self):
        """The backing ``SentenceTransformer``, built once."""
        if self._model is None:
            import torch
            from sentence_transformers import SentenceTransformer

            from ..defenses._backends import gpu_dtype

            kwargs = dict(dtype=gpu_dtype(torch))
            kwargs.update(self.model_kwargs or {})
            print(f"[{self.name}] loading {self.repo}")
            self._model = SentenceTransformer(
                self.repo, device="cuda" if torch.cuda.is_available() else "cpu",
                model_kwargs=kwargs, trust_remote_code=True)
            self._model.max_seq_length = self.max_tokens
        return self._model

    def featurize(self, texts) -> np.ndarray:
        import torch

        model = self.load()
        rendered = [self.render(text or "") for text in texts]
        out = []
        for start in range(0, len(rendered), self.batch_size):
            batch = rendered[start:start + self.batch_size]
            with torch.inference_mode():
                v = model.encode(batch, convert_to_tensor=True, show_progress_bar=False)
            out.append(torch.nn.functional.normalize(v.float(), p=2, dim=1).cpu().numpy())
        return np.concatenate(out, axis=0) if out else np.empty((0, 0), dtype=np.float32)


class Harrier270mFeaturizer(SentenceTransformerFeaturizer):
    """Harrier 270m. **A Gemma 3 model**, unlike ``harrier`` (0.6b), which is Qwen 3 -- same product
    name, different backbone and an incompatible tokenizer. Its repository declares no clustering
    prompt, so it runs unconditioned."""

    name = "harrier_270m"
    repo = "microsoft/harrier-oss-v1-270m"


class EmbeddingGemma300mFeaturizer(SentenceTransformerFeaturizer):
    """EmbeddingGemma 300m under its own clustering prompt."""

    name = "embeddinggemma_300m"
    repo = "google/embeddinggemma-300m"
    prompt = "task: clustering | query: "


class JinaV5NanoFeaturizer(SentenceTransformerFeaturizer):
    """Jina Embeddings v5 nano under its clustering task.

    The task is a **model argument** here rather than a prefix, so it cannot ride in
    :meth:`render`; the repository refuses to encode without one.
    """

    name = "jina_v5_nano"
    repo = "jinaai/jina-embeddings-v5-text-nano"
    model_kwargs = {"default_task": "clustering"}
