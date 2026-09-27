"""LUAR: neural authorship embeddings (Rivera-Soto et al., EMNLP 2021).

LUAR maps a *collection* of documents to a 512-d vector under which two collections by the same
author have higher cosine similarity than two by different authors. Each of the collection's ``w``
excerpts goes through SBERT, and the resulting vectors are fused by self-attention across excerpts,
max-pooling, and a linear projection -- a learned fusion, not a mean of independent embeddings. It
is trained with a supervised contrastive loss on L2-normalized embeddings, so **cosine** is the
metric this space was fit under (Euclidean on the raw projection is not).

Here one conversation is one collection, per the paper's own guidance for input that isn't already
a set of short posts.

Deviations from the paper, and why
----------------------------------
* **Deterministic excerpts**, not the paper's random sampling: feature vectors here are cached
  content-addressed by text, and a random sampler would return different vectors on a cache hit
  than a miss, silently corrupting the feature space. Takes the *first* ``token_length`` tokens of
  each span instead.
* **Token-uniform spans, not paragraphs.** Turns are joined with blank lines and also contain
  them, so splitting on blank lines doesn't recover turn boundaries. Cutting the token stream into
  ``w`` equal spans instead gives whole-document coverage with a fixed, reproducible ``w``.
* **No L2 normalization**: the checkpoint's forward pass ends at the linear projection; cosine is
  scale-invariant, so normalizing would only discard norm information the scale-sensitive attacks
  (``logistic``, ``wccn``, ``plda``) can use.
* **Frozen and zero-shot** -- no fine-tuning on this corpus, matching the paper's own cross-domain
  setting.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import Featurizer

# The Reddit ("Million User Dataset") checkpoint -- the paper's best zero-shot transfer, which
# matches this use case: chat/agent transcripts are a domain the checkpoint has never seen.
DEFAULT_MODEL_ID = "rrivera1849/LUAR-MUD"
# Pinned so `trust_remote_code` runs a fixed revision of the Hub's modeling code, and so a
# checkpoint update invalidates the cache (the revision is in params()) rather than silently
# mixing two embedding spaces.
DEFAULT_REVISION = "f1db50251805ed69b43cf4f72ea2f0e231f36a1c"

# The paper's test-time episode length for its closest domains. `token_length` is the full
# tokenizer window, special tokens included.
DEFAULT_EPISODE_LENGTH = 16
DEFAULT_TOKEN_LENGTH = 32
DEFAULT_BATCH_SIZE = 32


class LuarFeaturizer(Featurizer):
    """Embed each conversation as a LUAR collection, yielding one 512-d vector per conversation.

    Pair with ``--metric cosine``; see the module docstring for why.
    """

    name = "luar"
    version = "1"

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_REVISION,
                 episode_length: int = DEFAULT_EPISODE_LENGTH,
                 token_length: int = DEFAULT_TOKEN_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        self.model_id = model_id
        self.revision = revision
        self.episode_length = int(episode_length)
        self.token_length = int(token_length)
        self.batch_size = max(1, int(batch_size))
        if self.episode_length < 1:
            raise ValueError(f"episode_length must be at least 1, got {self.episode_length}.")
        if self.token_length < 4:
            # A window has to hold the tokenizer's special tokens plus some text to be meaningful.
            raise ValueError(f"token_length must be at least 4, got {self.token_length}.")
        self._tokenizer = None
        self._model = None
        self._device = None

    def params(self) -> dict:
        # `batch_size` excluded: it changes grouping into forward passes, never the vectors.
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "episode_length": self.episode_length,
            "token_length": self.token_length,
        }

    def _ensure_model(self):
        """Load the tokenizer and checkpoint on first use; returns ``(tokenizer, model, device)``."""
        if self._model is None:
            # Imported lazily so importing the features registry never requires torch/transformers.
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # LUAR ships its own modeling code on the Hub, so trust_remote_code is required;
            # `revision` pins which version of that code runs.
            self._model = AutoModel.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            ).to(self._device).eval()
        return self._tokenizer, self._model, self._device

    def _episode(self, tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
        """One document's collection: ``(w, token_length)`` input-id and attention-mask arrays.

        The document's token stream is cut into ``w`` equal contiguous spans, and the first
        ``token_length`` tokens of each become one excerpt, so excerpts cover the whole document.

        ``w`` scales with document length up to ``episode_length``, rather than a fixed value:
        that avoids both an all-zero-padded excerpt (undefined mean pooling) and repeating text to
        fake a larger collection than the author actually wrote. Empty text still yields one
        excerpt so every document gets a well-defined vector.
        """
        budget = self.token_length - tokenizer.num_special_tokens_to_add(pair=False)
        # verbose=False: cutting a long document into short windows is the point, not a mistake.
        ids = tokenizer(text or "", add_special_tokens=False, verbose=False)["input_ids"]
        count = min(self.episode_length, max(1, -(-len(ids) // budget)))

        input_ids = np.full((count, self.token_length), tokenizer.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((count, self.token_length), dtype=np.int64)
        for index, span in enumerate(np.array_split(np.asarray(ids, dtype=np.int64), count)):
            excerpt = tokenizer.build_inputs_with_special_tokens(span[:budget].tolist())
            input_ids[index, :len(excerpt)] = excerpt
            attention_mask[index, :len(excerpt)] = 1
        return input_ids, attention_mask

    def featurize(self, texts) -> np.ndarray:
        import torch

        tokenizer, model, device = self._ensure_model()
        episodes = [self._episode(tokenizer, text) for text in texts]
        embeddings = np.empty((len(texts), model.config.embedding_size), dtype=float)

        # Batch by collection length (w varies per document) so no batch needs a padded-out
        # excerpt.
        by_length: dict[int, list[int]] = defaultdict(list)
        for position, (input_ids, _) in enumerate(episodes):
            by_length[len(input_ids)].append(position)

        for positions in by_length.values():
            for start in range(0, len(positions), self.batch_size):
                batch = positions[start:start + self.batch_size]
                input_ids = torch.as_tensor(
                    np.stack([episodes[position][0] for position in batch]), device=device
                )
                attention_mask = torch.as_tensor(
                    np.stack([episodes[position][1] for position in batch]), device=device
                )
                with torch.inference_mode():
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                # The Hub model returns a bare tensor, or a tuple when attentions are requested.
                if isinstance(output, tuple):
                    output = output[0]
                embeddings[batch] = output.float().cpu().numpy()

        return embeddings
