"""Small sentence-transformers encoders, used as EmBad's cross-tokenizer surrogate ensemble.

One generic featurizer covering any repository that ships a sentence-transformers config, plus the
three registered instances built on it. They are handled by a single class because the
repositories already declare their own pipeline -- pooling, dense projections, normalization -- so
reimplementing any of it here would only be a way for it to drift from what the model card says.

Why these three
---------------

They are deliberately not one family. EmBad's whole problem is *transfer*: the search steers local
encoders and the adversary uses one the defender does not have, so an ensemble is only worth
anything if its members disagree in ways the target might.

======================  =================  ==============  =====================================
\\                        tokenizer          pooling         notes
======================  =================  ==============  =====================================
``harrier_270m``        Gemma 3 (262k)     mean + norm     Gemma3TextModel, 640-d
``embeddinggemma_300m`` Gemma 3 (262k)     mean + 2 dense  Gemma3TextModel, 768-d, task-prefixed
``jina_v5_nano``        EuroBERT (128k)    custom module   task-conditioned, 768-d
======================  =================  ==============  =====================================

The first two share a tokenizer exactly; **Jina's vocabulary is unrelated**, with zero ids in
common. That mattered when EmBad searched over token ids and had to split its members into ones it
could optimize jointly and ones that could only score text. It searches over natural-language
passages now (:class:`~prompt_anonymity.defenses.embad.SearchPool`), which every tokenizer reads,
so the disagreement is only what the ensemble wants it to be: three encoders that can fail
differently.

**Two of the three EmBad actually uses live here.** Its third member is ``harrier`` -- the 0.6b
Qwen 3 checkpoint in :mod:`~prompt_anonymity.features.harrier`, not the Gemma-based ``harrier_270m``
below, which shares the product name and nothing else.

Task conditioning
-----------------

These models are task-conditioned and the task rides in a text prefix. All three are set to their
**clustering** task where they have one, matching the default of
:mod:`~prompt_anonymity.features.gemini_embedding` -- the adversary's encoder -- so every vector in
the pipeline describes the same task. The prefix is part of :meth:`render`, so it is inside the
string the search steers rather than bolted on afterwards.
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
        #: Truncation window. Defaults to the smallest of the three repositories' own limits so the
        #: members read the same amount of each document -- an encoder that sees twice as much text
        #: as its neighbour is not solving the same problem, and the ensemble loss would be summing
        #: two different questions.
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
