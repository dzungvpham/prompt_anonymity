"""LUAR: neural authorship embeddings (Rivera-Soto et al., EMNLP 2021).

LUAR maps a *collection* of documents to a 512-d vector under which two collections by the same
author have higher cosine similarity than two by different authors. Each of the collection's ``w``
excerpts goes through SBERT, and the resulting ``w`` vectors are fused by self-attention across
excerpts, max-pooling, and a linear projection -- so the fusion across documents is learned, not a
mean of independent embeddings. It is trained with a supervised contrastive loss on L2-normalized
embeddings, which is why **cosine** is the metric this space was fit under (``--metric cosine``);
Euclidean on the raw projection is not.

Here one conversation is one collection. The paper prescribes exactly this when the input is not
already a set of short posts: "if instead the input is a single long document, we regard it as a
collection by subdividing into paragraphs".

Deviations from the paper, and why
----------------------------------
* **Deterministic excerpts.** The paper samples ``N = 32`` consecutive tokens at random from each
  of ``w`` contiguous documents. Feature vectors here are cached content-addressed by text (see
  :mod:`prompt_anonymity.caching`), so a featurizer that sampled randomly would return different
  vectors on a cache hit than on a miss and silently corrupt the feature space. This takes the
  *first* ``token_length`` tokens of each span instead. Random windows were a training-time
  augmentation, not part of inference.
* **Token-uniform spans, not paragraphs.** Turns reach a featurizer already joined by a blank line
  (``TURN_SEPARATOR`` in ``data/compute_features.py``) and turns themselves contain blank lines, so
  splitting on ``"\\n\\n"`` does not recover turn boundaries -- it over-segments unpredictably and
  makes ``w`` depend on markdown formatting. Cutting the token stream into ``w`` equal spans gives
  the same "collection covering the whole document" effect with a fixed, reproducible ``w``, and
  keeps long agent sessions sampled end-to-end rather than truncated to a prefix.
* **No L2 normalization.** The checkpoint's own forward pass ends at the linear projection, and
  cosine is scale-invariant, so normalizing here would be a no-op for the intended metric while
  discarding norm information that the scale-sensitive attacks (``logistic``, ``wccn``, ``plda``)
  can use.
* **Frozen and zero-shot** -- no fine-tuning on this corpus, which is the paper's own cross-domain
  setting and keeps the adversary reproducible.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .base import Featurizer

# The Reddit ("Million User Dataset") checkpoint. The paper's headline transfer result is that the
# Reddit-trained model generalizes best zero-shot -- over 80% of the in-domain R@8 of the
# domain-specific models on both unseen domains -- which is the situation here: chat and agent
# transcripts are a fourth domain the checkpoint has never seen.
DEFAULT_MODEL_ID = "rrivera1849/LUAR-MUD"
# Pinned so `trust_remote_code` runs a fixed revision of the Hub's modeling code rather than
# whatever `main` points at today, and so that a checkpoint update invalidates this featurizer's
# cache (the revision is in params()) instead of silently mixing two embedding spaces.
DEFAULT_REVISION = "f1db50251805ed69b43cf4f72ea2f0e231f36a1c"

# The paper draws N = 32 consecutive tokens from each of w documents, with w = ceil(1 + 15x),
# x ~ Beta(3, 1) -- so w is at most 16 and skewed toward it -- and fixes w = 16 at test time for
# the Reddit and Amazon domains. `token_length` is the full tokenizer window, special tokens
# included, which is what a `max_length=32` truncating tokenizer produces during training.
DEFAULT_EPISODE_LENGTH = 16
DEFAULT_TOKEN_LENGTH = 32
# Documents per forward pass: 32 documents x 16 segments = 512 short sequences in flight.
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
        # `batch_size` is deliberately excluded: it changes how documents are grouped into forward
        # passes, never the vectors, so runs with different batch sizes share one cache namespace.
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "episode_length": self.episode_length,
            "token_length": self.token_length,
        }

    def _ensure_model(self):
        """Load the tokenizer and checkpoint on first use; returns ``(tokenizer, model, device)``."""
        if self._model is None:
            # Imported lazily (like style_distance and the model-backed defenses) so importing the
            # features registry never requires torch / transformers -- only selecting this
            # featurizer does. Needs the [features] extra (see pyproject.toml / install.sh).
            import torch
            from transformers import AutoModel, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            # LUAR ships its own modeling code on the Hub (config `auto_map` -> `model.LUAR`), so
            # it cannot be loaded without trust_remote_code; `revision` pins which code that is.
            self._model = AutoModel.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            ).to(self._device).eval()
        return self._tokenizer, self._model, self._device

    def _episode(self, tokenizer, text: str) -> tuple[np.ndarray, np.ndarray]:
        """One document's collection: ``(w, token_length)`` input-id and attention-mask arrays.

        The document's token stream is cut into ``w`` equal contiguous spans and the first
        ``token_length`` tokens of each (special tokens included) become one excerpt, so the
        excerpts cover the whole document rather than a prefix of it.

        ``w`` is only as large as there is text to fill: 1 for a short conversation, rising to
        ``episode_length`` once the document is long enough. That keeps ``w`` inside the range the
        checkpoint was trained over and avoids the two bad alternatives -- an entirely padded
        excerpt, whose all-zero attention mask the backbone's mean pooling cannot divide by, and
        repeating text to reach a fixed ``w``, which would fake a larger collection than the author
        actually wrote. Empty text still yields one excerpt (special tokens only), so every
        document gets a well-defined vector rather than an all-zero row that would make cosine
        distance undefined.
        """
        budget = self.token_length - tokenizer.num_special_tokens_to_add(pair=False)
        # `verbose=False` silences the "sequence longer than the model's maximum" notice: reading
        # the whole document and then cutting it into short windows is the point here.
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

        # Short documents get shorter collections, so batch by collection length: every document in
        # a batch then shares one (batch, w, token_length) tensor without any padded-out excerpt.
        # `w` is bounded by `episode_length`, so this is at most that many groups.
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
