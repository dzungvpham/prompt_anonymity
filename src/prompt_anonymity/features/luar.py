"""LUAR: neural authorship embeddings (Rivera-Soto et al., EMNLP 2021).

LUAR maps a *collection* of documents to a 512-d vector under which two collections by the same
author have higher cosine similarity than two by different authors. Each of the collection's ``w``
excerpts goes through SBERT (a RoBERTa encoder, mean-pooled over tokens), and the resulting
vectors are fused by self-attention across excerpts, max-pooling, and a linear projection -- a
learned fusion, not a mean of independent embeddings. It is trained with a supervised contrastive
loss on L2-normalized embeddings, so **cosine** is the metric this space was fit under (Euclidean
on the raw projection is not).

This module serves LUAR two ways, matching the two kinds of collection in the paper's evaluation:

* :class:`LuarFeaturizer` (``--feature luar``) -- **one conversation is one collection**, the
  paper's treatment of a single long document. A per-document vector, precomputed into a parquet
  like every other featurizer, and attacked by any attack.
* :class:`LuarAuthorCollections` (``--feature luar_collection``) -- **one known author's recent
  history is one collection**, the paper's treatment of an author's posts. Built by
  ``experiments/run_experiment.py`` per known configuration, from that configuration's known
  documents only, and scored against the per-conversation vectors of the unknown side. This is
  the only form in which LUAR's fusion ever sees more than one document by the same person.

How a conversation becomes a collection
---------------------------------------
The paper (§2): *"If instead the input is a single long document, we regard it as a collection
by subdividing into paragraphs"*, and on fanfiction, at test time, *"we take w to be the number of
paragraphs, which essentially amounts to encoding the entire work."* Concretely:

1. **Paragraphs.** The text is split on blank lines (:data:`PARAGRAPH_BREAK`). Turns are joined
   by a blank line (``compute_features.TURN_SEPARATOR``), so every turn boundary is also a
   paragraph boundary.
2. **Windows inside paragraphs.** Each paragraph's token stream is cut into
   ``ceil(n / budget)`` near-equal contiguous windows of at most ``budget`` text tokens, where
   ``budget`` is ``token_length`` (32, the checkpoint's training excerpt length) less the
   tokenizer's special tokens. Every window is one excerpt. No window straddles two paragraphs,
   and no token is skipped.
3. **No cap on w** beyond a memory guard (``max_excerpts``, past which an evenly spaced subset is
   kept). The self-attention across excerpts has no parameters and the max-pool takes any count,
   so a long collection is well defined -- the paper ran its fanfiction stories this way.

Step 2 is the one place this goes beyond the paper's text, and it is needed for this data: a
prompt is often one long paragraph (a pasted essay, code with single newlines), and one excerpt
per paragraph would read its first 30 tokens and nothing else. Near-equal windows rather than
fixed 30-token ones keep a paragraph of 31 tokens from producing a one-token excerpt.

How an author becomes a collection
----------------------------------
In the paper's Amazon and Reddit evaluations an author's collection is ``w = 16`` of their
documents. :class:`LuarAuthorCollections` takes each known author's ``collection_size`` (16) most
recent known documents and fills ``collection_size`` excerpts from them **round-robin**, newest
document first: the first excerpt of every document, then the second of every document, and so
on. An author with 16 or more documents therefore contributes the opening of each, as the paper's
did; an author with fewer (WildChat's median is 3) still fills the collection from deeper in their
own documents rather than being embedded from a collection the model was not trained on.

Known limits
------------
* **Short collections are out of the training distribution.** Training drew
  ``w = ceil(1 + 15x)`` with ``x ~ Beta(3, 1)``, so ``P(w <= 2) = (1/15)^3 = 0.03%`` and
  ``P(w <= 4) = 0.2^3 = 0.8%``: the model essentially never saw a collection of one or two
  excerpts, which is what a prompt under ~60 tokens becomes. Nothing faithful to the paper fixes
  that; :meth:`LuarFeaturizer.close` reports how many documents it affects.
* **English only.** The backbone is ``paraphrase-distilroberta-base-v1``, fine-tuned on Reddit.
  Expect it to degrade on WildChat's non-English documents by construction.

Deviations from the paper, and why
----------------------------------
* **Deterministic windows**, not the paper's random 32-token windows. Those are a training-time
  augmentation; and feature vectors here are cached content-addressed by text, so a random
  sampler would return different vectors on a cache hit than a miss.
* **No L2 normalization**: the checkpoint's forward pass ends at the linear projection; cosine is
  scale-invariant, so normalizing would only discard norm information the scale-sensitive attacks
  (``logistic``, ``wccn``, ``plda``) can use.
* **Frozen and zero-shot** -- no fine-tuning on this corpus, matching the paper's own cross-domain
  setting.
* **TF32 matmuls** (``tf32=True``, the default; ``False`` for plain FP32). The encoder is
  compute-bound in FP32, and TF32 is 4.2x faster on an A100 (13.5 s -> 3.2 s for 1,024 WildChat
  conversations). Measured against FP32 on those conversations: worst-case cosine 1.000000 and
  every document keeps the same nearest neighbour. It is in ``params()``, so the two never share
  a cache entry.

Batching
--------
The checkpoint's own forward pass takes a ``(batch, w, token_length)`` block, and because its
attention across excerpts is **unmasked** a padded excerpt would leak into every vector, so it can
only batch collections of equal ``w``. Almost all of the cost is the encoder, though, which sees
each excerpt independently. :meth:`LuarFeaturizer.embed_excerpts` therefore runs the encoder over
every excerpt of every collection in flat batches, then applies the checkpoint's own mean-pooling,
attention, max-pool and projection modules per collection, grouped by ``w``. The arithmetic is the
checkpoint's, regrouped: on 64 swe-chat conversations of 1 to 152 excerpts it agrees with the
stock ``model(input_ids, attention_mask)`` call to 1.1e-6 absolute (minimum cosine 0.9999999).
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from ..caching import TransformCache, logic_hash, params_hash
from .base import Featurizer

# The Reddit ("Million User Dataset") checkpoint -- the paper's best zero-shot transfer, which
# matches this use case: chat/agent transcripts are a domain the checkpoint has never seen.
DEFAULT_MODEL_ID = "rrivera1849/LUAR-MUD"
# Pinned so `trust_remote_code` runs a fixed revision of the Hub's modeling code, and so a
# checkpoint update invalidates the cache (the revision is in params()) rather than silently
# mixing two embedding spaces.
DEFAULT_REVISION = "f1db50251805ed69b43cf4f72ea2f0e231f36a1c"

#: The checkpoint's training excerpt length (the paper's N = 32), special tokens included.
DEFAULT_TOKEN_LENGTH = 32
#: Memory guard on one collection's excerpt count, not a modelling choice: 1,024 excerpts is
#: ~30,000 tokens of text, past what any other featurizer here reads (Gemini stops at 8,192).
DEFAULT_MAX_EXCERPTS = 1024
#: The paper's test-time collection size for its many-short-documents domains (Amazon, Reddit).
DEFAULT_COLLECTION_SIZE = 16
#: Excerpts per encoder call. Only sets how work is grouped onto the GPU, never a vector.
DEFAULT_BATCH_SIZE = 1024
#: Bound on ``collections x w`` per call of the cross-excerpt aggregation, so one very long
#: collection and a thousand short ones cost the same bounded memory.
AGGREGATION_EXCERPTS = 16384

#: A blank line: two line breaks with only whitespace between them.
PARAGRAPH_BREAK = re.compile(r"\n[^\S\n]*\n")

#: Collections cached at once by :meth:`LuarAuthorCollections.author_vectors`; each chunk is
#: written to disk before the next starts, so a long build checkpoints as it goes.
COLLECTION_CHUNK = 1024


class LuarFeaturizer(Featurizer):
    """Embed each conversation as a LUAR collection, yielding one 512-d vector per conversation.

    See the module docstring for how a conversation is cut into excerpts. Pair with
    ``--metric cosine``.
    """

    name = "luar"
    version = "2"

    def __init__(self, *, model_id: str = DEFAULT_MODEL_ID, revision: str = DEFAULT_REVISION,
                 token_length: int = DEFAULT_TOKEN_LENGTH,
                 max_excerpts: int = DEFAULT_MAX_EXCERPTS,
                 batch_size: int = DEFAULT_BATCH_SIZE, tf32: bool = True):
        self.model_id = model_id
        self.revision = revision
        self.token_length = int(token_length)
        self.max_excerpts = int(max_excerpts)
        self.batch_size = max(1, int(batch_size))
        self.tf32 = bool(tf32)
        if self.token_length < 4:
            # A window has to hold the tokenizer's special tokens plus some text to be meaningful.
            raise ValueError(f"token_length must be at least 4, got {self.token_length}.")
        if self.max_excerpts < 1:
            raise ValueError(f"max_excerpts must be at least 1, got {self.max_excerpts}.")
        self._tokenizer = None
        self._model = None
        self._device = None
        # Running totals over every document this instance featurized, reported by close().
        self.n_documents = 0
        self.n_excerpts = 0
        self.n_short = 0
        self.n_capped = 0

    def params(self) -> dict:
        # `batch_size` excluded: it changes grouping into forward passes, never the vectors.
        # `tf32` included: it does change them, slightly (see `_matmul_precision`).
        return {
            "model_id": self.model_id,
            "revision": self.revision,
            "token_length": self.token_length,
            "max_excerpts": self.max_excerpts,
            "excerpts": "paragraph_windows",
            "tf32": self.tf32,
        }

    @contextmanager
    def _matmul_precision(self, torch):
        """Run CUDA matmuls on TF32 tensor cores for the duration, then put the global flag back.

        The encoder is compute-bound in plain FP32 (about 4,800 excerpts/s on an A100, its FP32
        ceiling for this model), and PyTorch leaves the tensor cores idle by default. The flag is
        process-global state this class does not own, so it is restored rather than left set, as
        in :class:`~prompt_anonymity.attacks.multiclass.MinibatchLogisticAttribution`.
        """
        if self._device != "cuda" or not self.tf32:
            yield
            return
        previous = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            yield
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous

    def _ensure_model(self):
        """Load the tokenizer and checkpoint on first use; returns ``(tokenizer, model, device)``."""
        if self._model is None:
            # Imported lazily so importing the features registry never requires torch/transformers.
            import torch
            from transformers import AutoModel, AutoTokenizer

            # LUAR ships its own modeling code on the Hub, so trust_remote_code is required -- by
            # the tokenizer too, which reads the custom config to pick its class; `revision` pins
            # which version of that code runs.
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModel.from_pretrained(
                self.model_id, revision=self.revision, trust_remote_code=True
            ).to(self._device).eval()
            # The special tokens a single sequence is wrapped in (``<s> ... </s>`` for this
            # checkpoint), read off the tokenizer rather than built with
            # `build_inputs_with_special_tokens`, which transformers 5 removed.
            bare = self._tokenizer("a", add_special_tokens=False)["input_ids"]
            wrapped = self._tokenizer("a", add_special_tokens=True)["input_ids"]
            start = next(index for index in range(len(wrapped) - len(bare) + 1)
                         if wrapped[index:index + len(bare)] == bare)
            self._prefix, self._suffix = wrapped[:start], wrapped[start + len(bare):]
        return self._tokenizer, self._model, self._device

    def windows(self, text: str) -> list[np.ndarray]:
        """One document's excerpts as raw token-id windows (no special tokens), in reading order.

        Paragraphs are split on blank lines and each is cut into ``ceil(n / budget)`` near-equal
        contiguous windows, so every token of the document lands in exactly one window. Empty
        text, or text that is only whitespace, has no windows.
        """
        tokenizer, _, _ = self._ensure_model()
        paragraphs = [paragraph for paragraph in PARAGRAPH_BREAK.split(text or "")
                      if paragraph.strip()]
        if not paragraphs:
            return []
        budget = self.token_length - len(self._prefix) - len(self._suffix)
        windows = []
        # verbose=False: a paragraph far longer than the encoder's window is expected here, since
        # it is cut into windows rather than fed whole.
        for ids in tokenizer(paragraphs, add_special_tokens=False, verbose=False)["input_ids"]:
            if ids:
                windows.extend(np.array_split(np.asarray(ids, dtype=np.int64),
                                              -(-len(ids) // budget)))
        return windows

    def excerpts(self, windows: list[np.ndarray]) -> list[list[int]]:
        """Encoder inputs (special tokens added) for one collection's ``windows``.

        Past ``max_excerpts`` an evenly spaced subset is kept, so a capped collection still spans
        the whole document. A collection with no windows becomes one excerpt of special tokens
        only, so every document gets a well-defined vector.
        """
        self._ensure_model()
        if len(windows) > self.max_excerpts:
            keep = np.linspace(0, len(windows) - 1, self.max_excerpts).round().astype(int)
            windows = [windows[index] for index in keep]
        if not windows:
            windows = [np.empty(0, dtype=np.int64)]
        return [self._prefix + window.tolist() + self._suffix for window in windows]

    def embed_excerpts(self, collections: list[list[list[int]]]) -> np.ndarray:
        """One vector per collection, from each collection's encoder inputs (see :meth:`excerpts`).

        The encoder runs over all excerpts of all collections in flat batches; the checkpoint's own
        mean-pooling, cross-excerpt attention, max-pool and projection then run per collection,
        grouped by excerpt count since that attention takes no mask.
        """
        import torch

        tokenizer, model, device = self._ensure_model()
        sizes = np.array([len(collection) for collection in collections])
        if (sizes == 0).any():
            raise ValueError("every collection needs at least one excerpt; see `excerpts`.")
        flat = [excerpt for collection in collections for excerpt in collection]
        input_ids = np.full((len(flat), self.token_length), tokenizer.pad_token_id, dtype=np.int64)
        attention_mask = np.zeros((len(flat), self.token_length), dtype=np.int64)
        for row, excerpt in enumerate(flat):
            input_ids[row, :len(excerpt)] = excerpt
            attention_mask[row, :len(excerpt)] = 1

        embeddings = np.empty((len(collections), model.config.embedding_size), dtype=np.float32)
        with torch.inference_mode(), self._matmul_precision(torch):
            pooled = torch.empty((len(flat), model.hidden_size), device=device)
            for start in range(0, len(flat), self.batch_size):
                ids = torch.as_tensor(input_ids[start:start + self.batch_size], device=device)
                mask = torch.as_tensor(attention_mask[start:start + self.batch_size], device=device)
                hidden = model.transformer(input_ids=ids, attention_mask=mask,
                                           return_dict=True)["last_hidden_state"]
                pooled[start:start + self.batch_size] = model.mean_pooling(hidden, mask)

            offsets = np.concatenate([[0], np.cumsum(sizes)])
            by_size: dict[int, list[int]] = defaultdict(list)
            for position, size in enumerate(sizes):
                by_size[int(size)].append(position)
            for size, members in by_size.items():
                step = max(1, AGGREGATION_EXCERPTS // size)
                for start in range(0, len(members), step):
                    chunk = members[start:start + step]
                    rows = np.concatenate([np.arange(offsets[member], offsets[member] + size)
                                           for member in chunk])
                    block = pooled[torch.as_tensor(rows, device=device)].view(len(chunk), size, -1)
                    attended = model.attn_fn(block, block, block)
                    projected = model.linear(attended.amax(dim=1))
                    embeddings[chunk] = projected.float().cpu().numpy()
        return embeddings

    def featurize(self, texts) -> np.ndarray:
        collections = []
        for text in texts:
            windows = self.windows(text)
            self.n_documents += 1
            self.n_excerpts += min(len(windows), self.max_excerpts)
            self.n_short += len(windows) <= 2
            self.n_capped += len(windows) > self.max_excerpts
            collections.append(self.excerpts(windows))
        return self.embed_excerpts(collections)

    def close(self) -> None:
        """Report how the documents this run embedded were cut. Safe to call more than once."""
        if self.n_documents:
            print(f"[{self.name}] {self.n_documents:,} documents -> {self.n_excerpts:,} excerpts "
                  f"(mean {self.n_excerpts / self.n_documents:.1f}); "
                  f"{self.n_short:,} ({self.n_short / self.n_documents:.1%}) have 2 or fewer, a "
                  f"collection size the checkpoint was essentially never trained on; "
                  f"{self.n_capped:,} capped at {self.max_excerpts:,}")


class LuarAuthorCollections:
    """Each known author's recent history as one LUAR collection (``--feature luar_collection``).

    Not a :class:`~prompt_anonymity.features.base.Featurizer`: its unit is an author on one known
    side, not a document, so it cannot be precomputed into a per-document parquet without fixing
    the known side in advance. ``experiments/run_experiment.py`` builds it per known configuration
    from that configuration's known documents only. The unknown side is the per-conversation
    :class:`LuarFeaturizer` vectors (:attr:`document_feature`), read from their parquet.

    See the module docstring for how the collection is filled.
    """

    name = "luar_collection"
    #: The per-document feature parquet the unknown side is read from.
    document_feature = LuarFeaturizer.name
    #: Manual logic version, as on a featurizer.
    version = "1"

    def __init__(self, collection_size: int = DEFAULT_COLLECTION_SIZE,
                 featurizer: LuarFeaturizer | None = None):
        self.collection_size = int(collection_size)
        if self.collection_size < 1:
            raise ValueError(f"collection_size must be at least 1, got {self.collection_size}.")
        self.featurizer = featurizer if featurizer is not None else LuarFeaturizer()

    def params(self) -> dict:
        return {**self.featurizer.params(), "collection_size": self.collection_size}

    def open_cache(self, cache_dir) -> TransformCache:
        """This class's cache under ``<cache_dir>/features``, namespaced like a featurizer's: by
        name, the source of this class *and* the featurizer's hierarchy, and :meth:`params`."""
        classes = [type(self), *(cls for cls in type(self.featurizer).__mro__ if cls is not object)]
        return TransformCache(Path(cache_dir) / "features", self.name,
                              logic_hash(classes, version=self.version), params_hash(self.params()))

    def collection(self, documents: list[str]) -> list[list[int]]:
        """Encoder inputs for one author, from their documents **newest first**.

        Up to :attr:`collection_size` excerpts, taken round-robin across the documents: the first
        window of each, then the second of each, and so on.
        """
        per_document = [self.featurizer.windows(text) for text in documents]
        picked: list[np.ndarray] = []
        for depth in range(max((len(windows) for windows in per_document), default=0)):
            for windows in per_document:
                if depth < len(windows):
                    picked.append(windows[depth])
                    if len(picked) == self.collection_size:
                        return self.featurizer.excerpts(picked)
        return self.featurizer.excerpts(picked)

    def author_vectors(self, texts, labels, cache_dir) -> tuple[np.ndarray, np.ndarray]:
        """``(authors, vectors)`` for one known side, row ``i`` of ``vectors`` being ``authors[i]``.

        ``texts`` and ``labels`` are the known side's documents **in timeline order**, so an
        author's most recent documents are their last rows. ``authors`` is sorted as
        :func:`numpy.unique` sorts it, which is the column order every attack uses. Each vector is
        cached under a digest of the exact documents it was built from.
        """
        authors, codes = np.unique(np.asarray(labels), return_inverse=True)
        histories: list[list[str]] = [[] for _ in authors]
        for position in range(len(texts) - 1, -1, -1):        # newest first
            history = histories[codes[position]]
            if len(history) < self.collection_size:
                history.append(texts[position] or "")

        def digest(documents: list[str]) -> str:
            # A digest rather than the text itself: the cache stores its key in every entry, and
            # sixteen whole conversations per author per known side would copy the corpus.
            hasher = hashlib.sha256()
            for text in documents:
                hasher.update(text.encode("utf-8"))
                hasher.update(b"\x1e")
            return hasher.hexdigest()

        def embed(batch: list[list[str]]) -> list[list[float]]:
            vectors = self.featurizer.embed_excerpts([self.collection(history) for history in batch])
            return [row.tolist() for row in vectors]

        cache = self.open_cache(cache_dir)
        vectors, computed = [], 0
        for start in range(0, len(histories), COLLECTION_CHUNK):
            vectors.extend(cache.apply_batch(histories[start:start + COLLECTION_CHUNK], embed,
                                             key=digest))
            computed += cache.misses
        print(f"[{self.name}] {len(authors):,} author collections "
              f"(up to {self.collection_size} excerpts from each author's "
              f"{self.collection_size} most recent known documents): {computed:,} embedded, "
              f"{len(authors) - computed:,} served from {cache.dir}")
        return authors, np.asarray(vectors, dtype=np.float32)
