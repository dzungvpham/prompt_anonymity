"""Gemini Embedding 2 semantic embeddings, served through OpenRouter.

The stylometric featurizers describe *how* a document is written; this one describes what it is
**about**, by embedding the text with Google's ``gemini-embedding-2`` (up to 3072 dimensions,
L2-normalized, matryoshka-truncatable). It is the successor to the old
``wildchat/get_embeddings.py``, which called ``gemini-embedding-001`` through Vertex AI with a
service account and a hand-rolled token-rate limiter; going through OpenRouter's OpenAI-compatible
``/embeddings`` endpoint needs nothing but the ``OPENROUTER_API_KEY`` this project already keeps
in its ``.env`` (the same key the fidelity judges and the OpenAnonymity defense use).

Like every :class:`~prompt_anonymity.features.base.Featurizer`, vectors are cached on disk by
text content, so a re-run costs nothing and an interrupted run resumes where it stopped -- which
matters more here than for the local featurizers, because every cache miss is a paid API call.

The task prefix
---------------
Embedding 2 has no ``task_type`` request field -- Google's documentation is explicit that the
field applies to ``gemini-embedding-001`` only. Instead the task is written **into the text**::

    task: sentence similarity | query: <the document>

which is what this featurizer sends (:data:`TASK_PREFIX_TEMPLATE`, :data:`DEFAULT_TASK`). That
detail is what makes the task selectable here at all: a request *field* would be dropped in
transit -- probed 2026-07-31, OpenRouter silently discards ``task_type``/``taskType``/
``input_type``/``provider.*``/``extra_body.*``, returning bit-identical vectors even for an
invalid value -- whereas a prefix is part of ``input`` and reaches the model verbatim.

``sentence similarity`` is the symmetric task: both sides of a comparison are embedded the same
way, which is what linkage does (an unknown document against known documents, not a query against
a corpus). ``clustering`` and ``classification`` are the other symmetric options; the asymmetric
retrieval tasks (``search result``, ``question answering``, ``fact checking``, ``code
retrieval``) would embed the two sides differently and do not fit this pipeline.

How much of a document is read
------------------------------
The model reads at most :data:`MODEL_INPUT_TOKENS` (8,192) tokens. On the previous model, input
past the limit was **silently ignored and still billed** -- embedding a 27,000-token document
returned, to within floating-point equality, the embedding of its first 2,048 tokens, at 13x the
price -- so the client, not the provider, does the cutting.

One document becomes **one** model-sized input and **one** call: no splitting into several
windows, no pooling of several vectors. A document's vector therefore represents its opening
~8,192 tokens, and the rest of a very long session is not represented at all -- a deliberate
simplification. It keeps a vector a real model output rather than an average of outputs, and it
caps the price of any document at :data:`INPUT_TOKENS` tokens whatever its length.

The cut is made *above* the model's limit (:data:`INPUT_TOKENS`, 10,000), because the budget is
measured with ``tiktoken`` -- a different tokenizer than Gemini's, disagreeing by a few percent on
prose and rather more on CJK and code. Overshooting means the provider, not this client, drops the
last tokens, so the model's window is certainly **full**; undershooting would silently leave part
of it empty. The overshoot is billed and discarded, which is what keeps it modest.
"""

from __future__ import annotations

import os
import random
import threading
import time

import numpy as np

from .base import Featurizer

#: OpenRouter's OpenAI-compatible embeddings endpoint.
OPENROUTER_EMBEDDINGS_URL = "https://openrouter.ai/api/v1/embeddings"
#: Environment variable holding the OpenRouter API key (loaded from a ``.env`` if present).
OPENROUTER_API_KEY_ENV = "OPENROUTER_API_KEY"

#: OpenRouter model id. Note the spelling: the Hub serves Gemini Embedding 2 as
#: ``google/gemini-embedding-2``; ``google/gemini-embedding-002`` is rejected as unknown.
MODEL_ID = "google/gemini-embedding-2"

#: How Embedding 2 is told what the embedding is for -- a prefix on the text itself, since the
#: model has no ``task_type`` field (see the module docstring).
TASK_PREFIX_TEMPLATE = "task: {task} | query: {text}"

#: The symmetric task this pipeline wants: both sides of a linkage comparison are embedded the
#: same way. Other symmetric options are ``"clustering"`` and ``"classification"``.
DEFAULT_TASK = "sentence similarity"

#: The model's input limit. Tokens past it are dropped by the provider without an error (and
#: charged for), so the client must cut the text itself -- see the module docstring.
MODEL_INPUT_TOKENS = 8192

#: Tokens of each document actually sent. Above the model's limit on purpose: the budget is
#: measured with a stand-in tokenizer, so the excess guarantees the model's window is full, and
#: it is billed and discarded.
INPUT_TOKENS = 10_000

#: ``tiktoken`` encoding used to measure the budget (a proxy for Gemini's tokenizer, see above).
TOKENIZER_ENCODING = "o200k_base"

#: Characters per token assumed when ``tiktoken`` is unavailable. Deliberately *generous*: with a
#: single truncated input the two failure modes are not symmetric -- overshooting costs a few
#: tokens of billing that the provider discards, while undershooting silently leaves part of the
#: model's window unused. Still only an estimate; ``tiktoken`` is much preferred.
FALLBACK_CHARS_PER_TOKEN = 5

#: The model's native output width. Fewer dimensions can be requested (Google recommends 768 or
#: 1536; anything from 128 to 3072 is accepted): the model is matryoshka-trained, so a prefix of
#: the vector is itself a usable embedding -- but a truncated one comes back **un-normalized**, so
#: this featurizer always re-normalizes (see :meth:`GeminiEmbedding2Featurizer._normalize`).
NATIVE_DIMENSIONS = 3072

#: Documents per HTTP request, and the total tokens one request may carry. Both caps apply: at
#: 8,192-token documents a plain count would build enormous requests, while at this corpus's
#: median document (a few hundred characters) a token-only rule would build huge ones too. Packing
#: to whichever cap binds first keeps requests roughly uniform, and keeps the work lost to one
#: retry bounded.
DOCUMENTS_PER_REQUEST = 32
REQUEST_TOKEN_BUDGET = 60_000

#: Concurrent requests when a call has more than one batch. Requests are network I/O-bound, so
#: threads (not processes) are the right tool.
DEFAULT_WORKERS = 8


class GeminiEmbedding2Featurizer(Featurizer):
    """Embeds documents with ``gemini-embedding-2`` through OpenRouter.

    Each document becomes exactly one API input and one vector: it is cut to
    :attr:`input_tokens` tokens (above what the model reads), wrapped in the task prefix, and
    embedded as-is. Nothing is pooled or averaged.

    Parameters
    ----------
    task : str
        Task written into the prefix (default :data:`DEFAULT_TASK`, ``"sentence similarity"``).
        Pass ``""``/``None`` to send the bare document with no prefix.
    input_tokens : int
        Tokens of each document to send (default :data:`INPUT_TOKENS`). Lower it to spend less
        per document; raising it far past :data:`MODEL_INPUT_TOKENS` only buys tokens the model
        will not read but will bill.
    dimensions : int, optional
        Output width. ``None`` (default) keeps the model's native :data:`NATIVE_DIMENSIONS`;
        768 or 1536 give a smaller vector (and a much smaller feature file) at some quality cost.
    workers : int, optional
        Concurrent HTTP requests (default :data:`DEFAULT_WORKERS`). Named to match the other
        featurizers' ``workers``, though here it is threads rather than processes.
    model : str
        OpenRouter model id, for pointing the same code at another embedding model.
    batch_size, request_token_budget, max_retries, backoff_cap, timeout
        Documents and tokens per request, retry budget for transient failures, backoff ceiling
        (seconds), and per-request timeout.

    An empty or whitespace-only document is given a zero vector without an API call: there is
    nothing to embed, and the endpoint rejects an empty string outright. Note that zero vectors
    have no cosine direction -- a distance to them is undefined -- but the built datasets contain
    no empty documents (the build drops them), so this only guards ad-hoc use.
    """

    name = "gemini_embedding_2"
    version = "1"

    def __init__(self, *, task: str | None = DEFAULT_TASK, input_tokens: int = INPUT_TOKENS,
                 dimensions: int | None = None, workers: int | None = None, model: str = MODEL_ID,
                 batch_size: int = DOCUMENTS_PER_REQUEST,
                 request_token_budget: int = REQUEST_TOKEN_BUDGET, max_retries: int = 8,
                 backoff_cap: float = 30.0, timeout: float = 300.0):
        self.task = task or ""
        self.input_tokens = max(1, int(input_tokens))
        self.dimensions = int(dimensions) if dimensions else None
        self.workers = DEFAULT_WORKERS if workers is None else max(1, int(workers))
        self.model = model
        self.batch_size = max(1, int(batch_size))
        self.request_token_budget = max(1, int(request_token_budget))
        self.max_retries = max_retries
        self.backoff_cap = backoff_cap
        self.timeout = timeout

        self._requests = None       # lazily imported `requests` module
        self._api_key = None        # lazily read from the environment / .env
        self._encoding = None       # lazily built tiktoken encoding (False once known missing)
        self._local = threading.local()  # per-thread requests.Session (connection reuse)
        self._usage_lock = threading.Lock()
        # Running API usage, reported by close(). Cost is OpenRouter's own accounting, in USD.
        self.total_requests = 0
        self.total_documents = 0
        self.total_truncated = 0
        self.total_tokens = 0
        self.total_cost = 0.0

    def params(self) -> dict:
        # Everything that changes the numbers, and nothing that does not: `workers`, `batch_size`,
        # `request_token_budget` and the retry knobs only affect how the calls are scheduled, so a
        # serial and a parallel run share one cache namespace. `task` does change them -- it is
        # part of the text the model sees -- so each task caches separately.
        return {"model": self.model, "task": self.task,
                "dimensions": self.dimensions or NATIVE_DIMENSIONS,
                "input_tokens": self.input_tokens}

    # --- reading one document ------------------------------------------------

    def _tokenizer(self):
        """The ``tiktoken`` encoding, or ``None`` if it is unavailable (reported once)."""
        if self._encoding is None:
            try:
                import tiktoken

                self._encoding = tiktoken.get_encoding(TOKENIZER_ENCODING)
            except Exception:  # noqa: BLE001 - not installed, or no cached encoding to download
                print(f"[{self.name}] tiktoken unavailable; measuring the input budget in "
                      f"characters (~{FALLBACK_CHARS_PER_TOKEN} chars/token).")
                self._encoding = False
        return self._encoding or None

    def _count_tokens(self, text: str) -> int:
        """Token count for ``text``, via ``tiktoken`` when it is installed, else estimated.

        ``tiktoken`` is not Gemini's tokenizer -- it is a stand-in, accurate enough given that
        :data:`INPUT_TOKENS` deliberately overshoots the model's window.
        """
        encoding = self._tokenizer()
        if encoding is None:
            return -(-len(text) // FALLBACK_CHARS_PER_TOKEN)  # ceiling division
        # disallowed_special=() so a literal "<|endoftext|>" in a user's prompt is counted as
        # ordinary text instead of raising.
        return len(encoding.encode(text, disallowed_special=()))

    def _prepare(self, text: str) -> tuple[str, int, bool]:
        """``(request input, its token count, whether the document was cut)``.

        Cuts the document to the input budget, then wraps it in the task prefix (whose handful of
        tokens ride on top of the budget, so a document is never shortened by the prefix). Blank
        text returns ``("", 0, False)`` and is never sent.

        Cutting on token boundaries can in principle split a character that spans several tokens,
        but only at the very end of the text -- inside the overshoot the model never reads -- so
        the cut is made on the token sequence directly rather than approximated in characters.
        """
        text = text or ""
        if not text.strip():
            return "", 0, False

        truncated = False
        encoding = self._tokenizer()
        if encoding is None:
            budget = self.input_tokens * FALLBACK_CHARS_PER_TOKEN
            if len(text) > budget:
                text, truncated = text[:budget], True
        else:
            tokens = encoding.encode(text, disallowed_special=())
            if len(tokens) > self.input_tokens:
                text, truncated = encoding.decode(tokens[:self.input_tokens]), True

        prepared = TASK_PREFIX_TEMPLATE.format(task=self.task, text=text) if self.task else text
        return prepared, self._count_tokens(prepared), truncated

    # --- OpenRouter ----------------------------------------------------------

    def _session(self):
        """This thread's ``requests`` session, built on first use (connections are not shared)."""
        if self._requests is None:
            import requests
            from dotenv import load_dotenv

            self._requests = requests
            load_dotenv()  # walks up from cwd, so the key can live in the repo root or a subdir
            self._api_key = os.environ.get(OPENROUTER_API_KEY_ENV)
            if not self._api_key:
                raise RuntimeError(
                    f"{OPENROUTER_API_KEY_ENV} not set. Add it to a .env file (e.g. "
                    f"'{OPENROUTER_API_KEY_ENV}=sk-or-...') so {self.name} can call OpenRouter."
                )
        session = getattr(self._local, "session", None)
        if session is None:
            session = self._local.session = self._requests.Session()
            session.headers.update({"Authorization": f"Bearer {self._api_key}",
                                    "Content-Type": "application/json"})
        return session

    def _record_usage(self, documents: int, usage: dict) -> None:
        """Add one response's billing to the running totals (thread-safe)."""
        with self._usage_lock:
            self.total_requests += 1
            self.total_documents += documents
            self.total_tokens += int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
            self.total_cost += float(usage.get("cost") or 0.0)

    def _embed_batch(self, documents: list[str]) -> list[list[float]]:
        """Embed one batch of prepared inputs, retrying transient failures; order is preserved.

        Transient failures (429, 5xx, network/JSON errors) are retried with jittered exponential
        backoff; other 4xx are client errors that would fail identically on retry, so they fail
        fast with OpenRouter's error body attached. The response is re-ordered by each item's
        ``index`` rather than trusted to arrive in order.
        """
        payload = {"model": self.model, "input": documents, "encoding_format": "float"}
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        # Built outside the retry loop: a missing API key is not a transient failure, and
        # retrying it would burn the whole backoff budget before reporting the real problem.
        session = self._session()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = session.post(OPENROUTER_EMBEDDINGS_URL, json=payload,
                                        timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                items = sorted(body["data"], key=lambda item: item.get("index", 0))
                if len(items) != len(documents):
                    raise RuntimeError(f"asked for {len(documents)} embeddings, got {len(items)}")
                self._record_usage(len(documents), body.get("usage") or {})
                return [item["embedding"] for item in items]
            except self._requests.exceptions.HTTPError as error:
                status = error.response.status_code
                last_error = RuntimeError(f"{status} {error.response.reason}: {error.response.text}")
                if 400 <= status < 500 and status != 429:
                    longest = max((len(document) for document in documents), default=0)
                    raise RuntimeError(
                        f"OpenRouter rejected the embeddings request (HTTP {status}) for model "
                        f"{self.model!r}; not retrying. Batch: {len(documents)} documents, longest "
                        f"{longest:,} characters. Response body: {error.response.text}"
                    ) from error
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
            except Exception as error:  # noqa: BLE001 - network/JSON errors are all retryable
                last_error = error
                if attempt < self.max_retries - 1:
                    self._backoff(attempt)
        raise RuntimeError(f"OpenRouter embeddings request failed after {self.max_retries} "
                           f"attempts: {last_error}")

    def _backoff(self, attempt: int) -> None:
        delay = min(self.backoff_cap, 2 ** attempt)
        time.sleep(random.uniform(0, delay))  # full jitter de-synchronizes concurrent workers

    def _batches(self, documents: list[str], token_counts: list[int]) -> list[list[str]]:
        """Pack the inputs into requests, closing one when either cap would be exceeded.

        A single input over the token budget still gets its own request rather than being dropped:
        the budget shapes requests, it does not filter documents.
        """
        batches: list[list[str]] = []
        current: list[str] = []
        current_tokens = 0
        for document, tokens in zip(documents, token_counts):
            too_many = len(current) >= self.batch_size
            too_big = current and current_tokens + tokens > self.request_token_budget
            if too_many or too_big:
                batches.append(current)
                current, current_tokens = [], 0
            current.append(document)
            current_tokens += tokens
        if current:
            batches.append(current)
        return batches

    def _embed(self, documents: list[str], token_counts: list[int]) -> list[list[float]]:
        """Embed every input, packing them into requests and running the requests concurrently."""
        batches = self._batches(documents, token_counts)
        if not batches:
            return []
        self._session()  # fail here, in the caller's thread, if there is no usable API key
        if len(batches) == 1:
            return self._embed_batch(batches[0])

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=min(self.workers, len(batches))) as pool:
            vectors = []
            for batch_vectors in pool.map(self._embed_batch, batches):  # order preserved
                vectors.extend(batch_vectors)
        return vectors

    # --- featurizer interface ------------------------------------------------

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        """Return ``vector`` on the unit sphere (a zero vector is left alone).

        The native 3072-dimension output already arrives L2-normalized, but a
        ``dimensions``-truncated matryoshka embedding does not, so normalizing here keeps every
        vector this featurizer returns comparable -- and cosine and euclidean ranking identical.
        """
        norm = float(np.linalg.norm(vector))
        return vector / norm if norm else vector

    def featurize(self, texts) -> np.ndarray:
        prepared = [self._prepare(text) for text in texts]
        width = self.dimensions or NATIVE_DIMENSIONS

        # Blank documents are held out of the request (the endpoint rejects an empty string) and
        # keep their zero row; everything else is one input.
        wanted = [index for index, (document, _, _) in enumerate(prepared) if document]
        vectors = self._embed([prepared[index][0] for index in wanted],
                              [prepared[index][1] for index in wanted])
        with self._usage_lock:
            self.total_truncated += sum(truncated for _, _, truncated in prepared)

        out = np.zeros((len(texts), width), dtype=float)
        for row, vector in zip(wanted, vectors):
            out[row] = self._normalize(np.asarray(vector, dtype=float))
        return out

    def close(self) -> None:
        """Report what the run spent. Safe to call more than once (totals only ever grow)."""
        if self.total_requests:
            print(f"[{self.name}] {self.total_documents:,} documents in "
                  f"{self.total_requests:,} requests "
                  f"({self.total_truncated:,} cut to the first {self.input_tokens:,} tokens), "
                  f"{self.total_tokens:,} tokens, ${self.total_cost:.4f}")
