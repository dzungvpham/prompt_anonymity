r"""Embed VisionArena's images with Cohere Embed v5 Pro on Microsoft Foundry -- one vector per document.

The image counterpart of :mod:`~prompt_anonymity.data.compute_features`: it reads the built
``vision_arena`` split (see :mod:`~prompt_anonymity.data.sources_vision_arena`), embeds each
**distinct** image once (81,111 images behind 92,295 documents), and writes one vector per document
so the image channel can be attacked like any other feature::

    python experiments/run_experiment.py --source vision_arena \
        --feature cohere_embed_v5_pro_image --attacks nearest_neighbor

What is sent, and why (measured 2026-11-11; the probes are in ``tmp/vision_arena_probes/``)
-------------------------------------------------------------------------------------------

* **The model is reached by one route only**, Cohere's own v2 API on the Foundry resource:
  ``POST <host of $ANTHROPIC_ENDPOINT>/providers/cohere/v2/embed`` with ``api-key: $MODEL_API_KEY``.
  Azure's generic embedding routes answer ``api_not_supported`` for it. It accepts PNG, JPEG, WebP
  and GIF, ignores the data URI's MIME label (it sniffs the bytes), returns 2048-d unit vectors, and
  takes at most :data:`MAX_IMAGES_PER_REQUEST` images per request.
* **The rate limiter charges request bytes, not image tokens** -- about
  :data:`LIMITER_TOKENS_PER_BODY_BYTE` per byte of request body, so a flat and a noisy 512x512 PNG
  that bill the same 258 image tokens were charged 594 and 262,068. Throughput is therefore set by
  payload size, and the encoding is chosen to shrink it.
* **The default encoding is WebP quality 25 at full resolution, transparency flattened onto white**
  (:class:`ImageEncoding`). On 250 real images that is 19.2 KiB a mean image against the PNG's 404,
  i.e. ~8.9 h for the split at 1M limiter tokens per minute against 186 h. On 20 real opaque images
  WebP's cosine to the PNG's own vector has a median of 0.989 at q35 and 0.986 at q20 (q25 lies
  between). **Resizing was measured and rejected**: at the same run time, WebP q50 at 768 px left
  12 of 20 images below cos 0.97 against q35's 2 -- the images are mostly text (65% OCR-tagged)
  and already at most 1024 px. Transparency is flattened because the model reads a PNG's colour
  channels without its alpha (a mostly transparent image scored cos 0.17 PNG-vs-JPEG), and white
  is what a light-mode viewer saw.
* **Billing depends on dimensions only** -- ``round(w/32) x round(h/32) + 2`` image tokens -- so the
  encoding changes the run time and not the bill: ~$15.5 at $0.40/M image tokens, or ~$61 if the
  ``billed_units.input_tokens`` figure (a steady 3.92x the image tokens) is the one charged. Both
  are recorded for every request.

The fused variant (``--with-text``, probed 2026-11-12)
------------------------------------------------------

One input per **document**: its user turns joined as everywhere (:data:`TURN_SEPARATOR`) and cut
to :data:`TEXT_MAX_CHARS`, then its image, sent as ``inputs: [{"content": [text, image_url]}]``;
the model returns **one joint vector** per input (:data:`FUSED_FEATURE`). What the probe showed:

* ``input_type`` is ignored for mixed content (four values, bit-identical vectors), so
  :data:`FUSED_INPUT_TYPE` is recorded rather than chosen.
* The text moves the vector: a 38-character question took it to cos 0.915 against the same image
  embedded alone.
* There is no server-side cut at the lengths this split has: all 178,520 characters of the longest
  document were read and billed (47,297 text tokens), so the cap here is ours.
* ``billed_units.input_tokens`` is the text tokens plus the usual 3.92x the image tokens.

The unit of work is a distinct ``(image, capped text)`` pair (:func:`fused_key`), about 14% more
than the distinct images, since one image can come with several texts from its one author. The
request body is the same image plus a little text, so the run time is the image run's scaled by
that, ~10 h at 1M limiter tokens per minute. Its cache namespace and key column (``item_key``) are
its own, and its records go to ``tmp/vision_arena_image_text_embeddings/`` -- a separate ledger,
because it carries two more columns (text tokens and their cost) than the image ledger.

Keeping the limit saturated
---------------------------

Encoding runs in a process pool (one task per upstream parquet row group, so each row group's
image bytes are read once) and fills a bounded queue of batches; :data:`DEFAULT_CONCURRENCY` sender
threads drain it through one :class:`RateGovernor`. The governor keeps a rolling 60 s ledger of
every request it let through, charged at ``body bytes x`` :data:`LIMITER_TOKENS_PER_BODY_BYTE`, and
admits a request only while that ledger stays under ``utilisation x`` the limits -- read from the
``x-ratelimit-limit-*`` response headers, so a quota change is picked up mid-run. The estimate is
exact (a pilot's four requests were charged within 0.3% of it), so the default target is 98%. A
``429`` still pauses every sender for the ``Retry-After`` time and lowers the target by 5% (never
below 70%); every quiet two minutes after it raises the target 1% back, so a systematic
underestimate corrects itself while a one-off does not slow the rest of the run.

**Batches perturb vectors very slightly.** An image embedded beside others is cos 0.99984 to the
same image embedded alone, while two single calls are bit-identical. Batches are formed in a fixed
order (by upstream shard and row), so a run is reproducible; a resumed run re-forms its batches
from what is left, which moves the affected vectors by that much.

Where things go
---------------

* **The feature file** -- ``data/dist/vision_arena_<feature>.parquet``, the feature-file layout
  exactly (``doc_id``, ``author_id``, ``f0000 .. f2047`` as float32), one row per document in the
  split's order. Documents sharing an image share its vector. **An image that could not be embedded
  gives NaN rows**, not missing rows, which would break the one-to-one ``doc_id`` join every runner
  enforces. ``<feature>`` is :data:`FEATURE`, plus a suffix for any non-default encoding
  (``_webp_q50``, ``_jpeg_q90``, ``_max768``), so two encodings never share a file. It is written to
  ``data/dist`` like every stage's output, and the working copy is then moved by hand into
  ``data/hf`` beside the split, where the runners read both (the split is read from there by
  default too). Like the split it is **local-only** (the dataset's license forbids
  redistribution): ``data/hf`` is still a local directory and must never be uploaded, and
  ``download`` deletes both files from it, since the Hub does not hold them -- ``--assemble-only``
  rebuilds this one from the cache in under a minute, at no cost.
* **The vector cache** -- ``data/.cache/vision_arena_image_vectors/<namespace>/part-*.parquet``,
  ``image_name`` + vector, flushed every :data:`FLUSH_EVERY_IMAGES` images or
  :data:`FLUSH_EVERY_SECONDS` and on any exit, so an interrupted run resumes and re-pays only for
  requests in flight. The namespace is a hash of the model, the encoding and :data:`EMBED_VERSION`
  only (written beside it as ``params.json``) and **is never pruned** -- unlike a featurizer's
  cache, which deletes stale versions on open. Bump the version by hand to re-embed.
* **Cost records** -- ``tmp/vision_arena_image_embeddings/``, **never** ``data/``:
  ``requests-<run>.csv`` (one row per request: images, body bytes, status, latency, billed units)
  and the append-only ``spend.csv`` ledger, one row per run that made a request, written even on an
  abort. Summing the ledger is this embedder's whole bill.

Run (from the repo root)::

    python -m prompt_anonymity.data.embed_vision_arena_images --dry-run       # counts, no requests
    python -m prompt_anonymity.data.embed_vision_arena_images --limit 64      # PAID pilot, no output file
    python -m prompt_anonymity.data.embed_vision_arena_images                 # PAID: the whole split
    python -m prompt_anonymity.data.embed_vision_arena_images --assemble-only # cache -> feature file
    sbatch scripts/embed_vision_arena_images_slurm.sh                         # the whole split on SLURM
    sbatch scripts/embed_vision_arena_images_slurm.sh --with-text             # the fused variant
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import queue
import signal
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import requests

from prompt_anonymity.caching import params_hash
from prompt_anonymity.resources import available_cpus

from .compute_features import (TURN_SEPARATOR, build_feature_frame, positional_feature_names, report_written,
                               write_parquet)
from .config import _env, cache_dir, dist_dir, hf_dir, project_root, raw_path

#: The split this embedder reads -- the only one with images.
SOURCE = "vision_arena"

#: The Foundry deployment, the route that serves it, and the credentials, as the topic labeller
#: names them (the same Foundry resource and key).
MODEL = "Cohere-Embed-V5-Pro"
EMBED_ROUTE = "/providers/cohere/v2/embed"
FOUNDRY_ENDPOINT_ENV = "ANTHROPIC_ENDPOINT"
FOUNDRY_API_KEY_ENV = "MODEL_API_KEY"

#: The feature name, i.e. the ``<split>_<feature>.parquet`` part a runner's ``--feature`` takes.
FEATURE = "cohere_embed_v5_pro_image"

#: The fused variant's feature name (``--with-text``): each document's user turns and its image
#: sent as one input, one joint vector per document.
FUSED_FEATURE = "cohere_embed_v5_pro_image_text"

#: ``input_type`` for a fused input. **The server ignores it for mixed content** -- ``clustering``,
#: ``search_document``, ``classification`` and ``image`` gave bit-identical vectors (probed
#: 2026-11-12) -- so this is the project's symmetric-task convention, recorded for provenance.
FUSED_INPUT_TYPE = "clustering"

#: A fused document's text is cut to this many characters, the cap the Gemini text feature reads,
#: so the two see the same text. The model itself read (and billed) all 178,520 characters of the
#: longest document, and its vector moved to cos 0.93 against the first 20,000.
TEXT_MAX_CHARS = 20_000

#: The model's native (and maximum) output width.
DIMENSIONS = 2048

#: Hard API cap; a 97th image is a 400.
MAX_IMAGES_PER_REQUEST = 96

#: The rate limiter's charge per byte of request body (measured 0.19-0.25; the upper end, from the
#: largest requests, so the governor errs towards a 429-free run rather than a faster one).
LIMITER_TOKENS_PER_BODY_BYTE = 0.25

#: The limiter's renewal period (``x-ratelimit-renewalperiod-*`` header).
LIMIT_WINDOW_SECONDS = 60.0

#: Limits used until the first response reports the real ones (the quota as of 2026-11-11).
DEFAULT_TOKENS_PER_MINUTE = 1_000_000
DEFAULT_REQUESTS_PER_MINUTE = 1_000

#: Share of the limits the governor aims at, and its floor after repeated 429s. The byte estimate
#: is exact (a pilot's four requests were charged within 0.3% of ``body x 0.25``), so the target
#: sits close to the limit; a 429 lowers it by 5%, and each quiet :data:`RECOVERY_SECONDS` after
#: one raises it 1% back towards the target, so one bad minute does not slow a nine-hour run.
DEFAULT_UTILISATION = 0.98
MIN_UTILISATION = 0.70
RECOVERY_SECONDS = 120.0

#: Images per request. Small enough that a retry is cheap and the window fills smoothly; the
#: request limit never binds (about ten requests a minute at this size).
DEFAULT_BATCH_IMAGES = 16

#: Requests in flight at once -- enough that request latency never leaves the budget unspent.
DEFAULT_CONCURRENCY = 4

#: Price per million tokens, applied to both billed figures in the records.
PRICE_PER_MTOK = 0.40

#: Manual logic version of the vector cache; bump it to re-embed after a behavioural change.
EMBED_VERSION = "1"

#: This embedder's own cache root under ``--cache-dir`` (never a featurizer's pruning cache).
CACHE_SUBDIR = "vision_arena_image_vectors"

#: Cache flush cadence: whichever comes first.
FLUSH_EVERY_IMAGES = 1000
FLUSH_EVERY_SECONDS = 120.0

#: Cost records, relative to the project root -- kept out of ``data/``. The fused variant keeps its
#: own directory, because its ledger has two more columns (text tokens) than the image ledger,
#: which must stay append-only under its original header.
USAGE_SUBDIR = Path("tmp") / "vision_arena_image_embeddings"
FUSED_USAGE_SUBDIR = Path("tmp") / "vision_arena_image_text_embeddings"
SPEND_LEDGER = "spend.csv"
SPEND_COLUMNS = ["finished_at_utc", "outcome", "run_id", "model", "encoding", "images_embedded",
                 "images_failed", "requests", "image_tokens", "billed_input_tokens",
                 "cost_usd_image_tokens", "cost_usd_billed_input_tokens", "elapsed_s"]
FUSED_SPEND_COLUMNS = SPEND_COLUMNS + ["text_tokens", "cost_usd_text_tokens"]
REQUEST_COLUMNS = ["time_utc", "n_images", "body_bytes", "limiter_estimate", "status", "latency_s",
                   "billed_input_tokens", "billed_image_tokens", "remaining_tokens", "billed_text_tokens"]

#: HTTP statuses that mean the configuration is wrong, not the request: stop the run.
FATAL_STATUSES = frozenset({401, 403, 404})

#: Attempts for a transient failure (5xx, timeout, connection error) before the batch is left for
#: the next run.
TRANSIENT_ATTEMPTS = 6


# --- encoding ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class ImageEncoding:
    """How an upstream PNG is re-encoded before it is sent (see the module docstring for the choice).

    ``fmt`` is ``"webp"`` or ``"jpeg"``; ``quality`` is the codec's 0-100 setting; ``max_side``, if
    set, downscales so the longer side is at most that many pixels (measured to cost far more
    fidelity than lowering quality, so off by default).
    """

    fmt: str = "webp"
    quality: int = 25
    max_side: int | None = None

    def label_suffix(self) -> str:
        """The feature-name suffix of a non-default encoding; empty for the default."""
        default = ImageEncoding()
        suffix = ""
        if (self.fmt, self.quality) != (default.fmt, default.quality):
            suffix += f"_{self.fmt}_q{self.quality}"
        if self.max_side:
            suffix += f"_max{self.max_side}"
        return suffix

    def describe(self) -> str:
        return f"{self.fmt} q{self.quality}" + (f" max{self.max_side}" if self.max_side else " full-res")

    def encode(self, png: bytes) -> bytes:
        """Flatten onto white, optionally downscale, and encode. Pixels are decoded here, once."""
        from PIL import Image

        image = Image.open(io.BytesIO(png)).convert("RGBA")
        white = Image.new("RGBA", image.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, image).convert("RGB")
        if self.max_side and max(image.size) > self.max_side:
            scale = self.max_side / max(image.size)
            image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                                 Image.LANCZOS)
        out = io.BytesIO()
        if self.fmt == "webp":
            image.save(out, "WEBP", quality=self.quality, method=6)  # method 6: smallest files
        elif self.fmt == "jpeg":
            image.save(out, "JPEG", quality=self.quality, optimize=True)
        else:
            raise ValueError(f"unsupported format {self.fmt!r} (webp or jpeg)")
        return out.getvalue()


def _encode_row_group(task) -> list[tuple[str, bytes]]:
    """Worker: encode the requested rows of one upstream row group. Top-level so it pickles.

    ``task`` is ``(raw_dir, shard, row_group, first_row, rows, names, keys, encoding)`` and the
    result is one ``(key, encoded bytes)`` per row -- the key is the image name itself, or a fused
    item's key (:func:`fused_key`). Each row is checked to hold the image its name says, which is
    what makes the locator trustworthy.
    """
    raw_dir, shard, row_group, first_row, rows, names, keys, encoding = task
    images = pq.ParquetFile(Path(raw_dir) / shard).read_row_group(row_group, columns=["images"]) \
        .column("images").to_pylist()
    out = []
    for row, name, key in zip(rows, names, keys):
        entry = images[row - first_row][0]
        if entry["path"] != name:
            raise ValueError(f"{shard} row {row} holds {entry['path']}, expected {name}")
        out.append((key, encoding.encode(entry["bytes"])))
    return out


def encode_tasks(pending: pd.DataFrame, raw_dir: Path, encoding: ImageEncoding) -> list[tuple]:
    """One task per upstream row group holding pending items, in shard-then-row order."""
    tasks = []
    for shard, rows in pending.groupby("image_shard", sort=True):
        meta = pq.ParquetFile(raw_dir / shard).metadata
        starts = np.cumsum([0] + [meta.row_group(g).num_rows for g in range(meta.num_row_groups)])
        group = np.searchsorted(starts, rows["image_row"].to_numpy(), side="right") - 1
        for g in np.unique(group):
            chunk = rows[group == g]
            tasks.append((str(raw_dir), shard, int(g), int(starts[g]), chunk["image_row"].tolist(),
                          chunk["image_name"].tolist(), chunk["key"].tolist(), encoding))
    return tasks


# --- what is embedded --------------------------------------------------------------------------

def fused_text(turns) -> str:
    """A document's user turns as the fused input's text: joined as everywhere, capped."""
    return TURN_SEPARATOR.join(turns)[:TEXT_MAX_CHARS]


def fused_key(image_name: str, text: str) -> str:
    """The cache key of one fused input: its image and a digest of its (capped) text.

    Content-addressed rather than a ``doc_id``, so two documents sending the same text with the
    same image share one vector -- as documents sharing an image do in the image-only variant.
    """
    return f"{image_name}:{hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]}"


def embedding_items(split_path: Path, with_text: bool) -> pd.DataFrame:
    """One row per document: ``doc_id``, ``author_id``, the image locator, and the cache ``key``.

    The key is the ``image_name`` for the image-only variant and :func:`fused_key` for the fused
    one, which also gets the ``text`` it sends.
    """
    columns = ["doc_id", "author_id", "image_name", "image_shard", "image_row"]
    docs = pd.read_parquet(split_path, columns=columns + (["turns"] if with_text else []))
    if not with_text:
        return docs.assign(key=docs["image_name"])
    docs["text"] = docs.pop("turns").map(fused_text)
    docs["key"] = [fused_key(name, text) for name, text in zip(docs["image_name"], docs["text"])]
    return docs


def fused_input(text: str, image_uri: str) -> dict:
    """One ``inputs`` entry: the text (left out when empty) and then the image."""
    parts = [{"type": "text", "text": text}] if text else []
    return {"content": parts + [{"type": "image_url", "image_url": {"url": image_uri}}]}


# --- rate limiting -----------------------------------------------------------------------------

class RateGovernor:
    """Admit requests so that a rolling window of their estimated limiter cost stays under the limits.

    Thread-safe. ``acquire`` blocks until a request of the given estimated cost fits in both the
    token and the request budget (``utilisation x`` the limits, over :data:`LIMIT_WINDOW_SECONDS`)
    and no 429 back-off is in force. ``observe`` adopts the limits a response reports; ``back_off``
    pauses everyone after a 429 and lowers the utilisation.
    """

    def __init__(self, tokens_per_minute: int, requests_per_minute: int, utilisation: float):
        self.tokens_per_minute = tokens_per_minute
        self.requests_per_minute = requests_per_minute
        self.utilisation = self.target_utilisation = utilisation
        self._last_change = 0.0
        self._sent: deque[tuple[float, float]] = deque()
        self._lock = threading.Lock()
        self._paused_until = 0.0

    def _purge(self, now: float) -> None:
        while self._sent and self._sent[0][0] <= now - LIMIT_WINDOW_SECONDS:
            self._sent.popleft()

    def window_usage(self) -> tuple[float, int]:
        """Estimated tokens and requests admitted in the current window."""
        with self._lock:
            self._purge(time.monotonic())
            return sum(c for _, c in self._sent), len(self._sent)

    def acquire(self, cost: float, stop: threading.Event) -> bool:
        """Wait until ``cost`` fits; record it and return True, or return False once ``stop`` is set."""
        while not stop.is_set():
            with self._lock:
                now = time.monotonic()
                self._purge(now)
                if self.utilisation < self.target_utilisation and now - self._last_change >= RECOVERY_SECONDS:
                    self.utilisation = min(self.target_utilisation, self.utilisation + 0.01)
                    self._last_change = now
                used = sum(c for _, c in self._sent)
                fits = (used + cost <= self.utilisation * self.tokens_per_minute
                        and len(self._sent) + 1 <= self.utilisation * self.requests_per_minute)
                if now >= self._paused_until and (fits or not self._sent):  # an empty window admits anything
                    self._sent.append((now, cost))
                    return True
                wait = max(self._paused_until - now,
                           (self._sent[0][0] + LIMIT_WINDOW_SECONDS - now) if self._sent else 0.05)
            stop.wait(min(max(wait, 0.05), 1.0))
        return False

    def observe(self, headers) -> None:
        tokens, reqs = headers.get("x-ratelimit-limit-tokens"), headers.get("x-ratelimit-limit-requests")
        with self._lock:
            if tokens and int(tokens) != self.tokens_per_minute:
                print(f"  rate limit: {int(tokens):,} tokens/min reported (was {self.tokens_per_minute:,})", flush=True)
                self.tokens_per_minute = int(tokens)
            if reqs and int(reqs) != self.requests_per_minute:
                self.requests_per_minute = int(reqs)

    def back_off(self, seconds: float) -> None:
        with self._lock:
            self._paused_until = max(self._paused_until, time.monotonic() + seconds)
            self.utilisation = max(MIN_UTILISATION, self.utilisation * 0.95)
            self._last_change = time.monotonic()
            print(f"  429: pausing {seconds:.0f}s, utilisation now {self.utilisation:.2f}", flush=True)


# --- vector cache ------------------------------------------------------------------------------

class VectorCache:
    """Append-only parquet parts of key -> vector under one never-pruned namespace.

    The key column is ``image_name`` for the image-only variant and ``item_key`` (:func:`fused_key`)
    for the fused one, so a part can never be read as the other variant's.
    """

    def __init__(self, root: Path, params: dict, key_column: str = "image_name"):
        self.dir = root / CACHE_SUBDIR / params_hash(params)
        self.dir.mkdir(parents=True, exist_ok=True)
        meta = self.dir / "params.json"
        if not meta.exists():
            meta.write_text(json.dumps(params, indent=2, sort_keys=True) + "\n")
        self.key_column = key_column
        self._buffer: list[tuple[str, np.ndarray]] = []
        self._lock = threading.Lock()
        self._seq = 0

    def parts(self) -> list[Path]:
        return sorted(self.dir.glob("part-*.parquet"))

    def done_keys(self) -> set[str]:
        return {n for p in self.parts() for n in pq.read_table(p, columns=[self.key_column]).column(0).to_pylist()}

    def add(self, names: list[str], vectors: np.ndarray) -> None:
        with self._lock:
            self._buffer.extend(zip(names, vectors))

    def pending(self) -> int:
        with self._lock:
            return len(self._buffer)

    def flush(self) -> int:
        """Write the buffer as one new part (temp file, then rename); return how many it held."""
        with self._lock:
            buffer, self._buffer = self._buffer, []
            self._seq += 1
            seq = self._seq
        if not buffer:
            return 0
        names = pa.array([n for n, _ in buffer], pa.string())
        flat = pa.array(np.stack([v for _, v in buffer]).astype(np.float32).ravel(), pa.float32())
        table = pa.table({self.key_column: names,
                          "embedding": pa.FixedSizeListArray.from_arrays(flat, DIMENSIONS)})
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        final = self.dir / f"part-{stamp}-{seq:05d}.parquet"
        tmp = final.with_suffix(".parquet.tmp")
        pq.write_table(table, tmp)
        tmp.replace(final)
        return len(buffer)

    def load_all(self) -> tuple[list[str], np.ndarray]:
        """Every cached vector (first occurrence of a key wins), as ``(keys, matrix)``."""
        names, blocks, seen = [], [], set()
        for part in self.parts():
            table = pq.read_table(part)
            part_names = table.column(self.key_column).to_pylist()
            matrix = table.column("embedding").combine_chunks().flatten().to_numpy().reshape(-1, DIMENSIONS)
            keep = [i for i, n in enumerate(part_names) if n not in seen]
            seen.update(part_names[i] for i in keep)
            names += [part_names[i] for i in keep]
            blocks.append(matrix[keep])
        return names, (np.concatenate(blocks) if blocks else np.zeros((0, DIMENSIONS), np.float32))


# --- the run -----------------------------------------------------------------------------------

class UsageLog:
    """Per-request CSV and running totals, thread-safe."""

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self._file = path.open("a", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=REQUEST_COLUMNS)
        if path.stat().st_size == 0:
            self._writer.writeheader()
        self._lock = threading.Lock()
        self.requests = self.images = self.failed = self.image_tokens = self.billed_input_tokens = 0
        self.text_tokens = 0

    def record(self, row: dict, embedded: int = 0, failed: int = 0) -> None:
        with self._lock:
            self._writer.writerow(row)
            self._file.flush()
            self.requests += 1
            self.images += embedded
            self.failed += failed
            self.image_tokens += int(row.get("billed_image_tokens") or 0)
            self.billed_input_tokens += int(row.get("billed_input_tokens") or 0)
            self.text_tokens += int(row.get("billed_text_tokens") or 0)

    def close(self) -> None:
        self._file.close()


class Embedder:
    """Senders, governor, cache and records for one run (see the module docstring)."""

    def __init__(self, host: str, key: str, governor: RateGovernor, cache: VectorCache,
                 usage: UsageLog, stop: threading.Event, texts: dict[str, str] | None = None):
        self.url = host + EMBED_ROUTE
        self.key = key
        self.governor, self.cache, self.usage, self.stop = governor, cache, usage, stop
        self.texts = texts  # key -> text for the fused variant; None sends images alone
        self.fatal: str | None = None
        self.failed: list[tuple[str, str]] = []
        self._local = threading.local()

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "session"):
            self._local.session = requests.Session()
            self._local.session.headers.update({"api-key": self.key, "Content-Type": "application/json"})
        return self._local.session

    def body(self, batch: list[tuple[str, bytes]]) -> bytes:
        # The MIME label is ignored by the server (it sniffs the bytes).
        uris = ["data:image/webp;base64," + base64.b64encode(data).decode() for _, data in batch]
        if self.texts is None:
            payload = {"model": MODEL, "input_type": "image", "embedding_types": ["float"], "images": uris}
        else:
            payload = {"model": MODEL, "input_type": FUSED_INPUT_TYPE, "embedding_types": ["float"],
                       "inputs": [fused_input(self.texts[key], uri) for (key, _), uri in zip(batch, uris)]}
        # Raw UTF-8 rather than \u escapes: the limiter charges body bytes, and an escape is 6 bytes
        # where a Cyrillic or Vietnamese character is 2-3.
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    def send(self, batch: list[tuple[str, bytes]]) -> None:
        """Embed one batch, retrying 429s and transient errors; split a 400 to isolate the bad image."""
        body = self.body(batch)
        estimate = len(body) * LIMITER_TOKENS_PER_BODY_BYTE
        for attempt in range(TRANSIENT_ATTEMPTS):
            if not self.governor.acquire(estimate, self.stop):
                return
            start = time.monotonic()
            row = {"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                   "n_images": len(batch), "body_bytes": len(body), "limiter_estimate": round(estimate)}
            try:
                response = self._session().post(self.url, data=body, timeout=(15, 300))
            except requests.RequestException as error:
                self.usage.record({**row, "status": f"error:{type(error).__name__}",
                                   "latency_s": round(time.monotonic() - start, 2)})
                self.stop.wait(min(60, 2 ** attempt))
                continue
            row.update(status=response.status_code, latency_s=round(time.monotonic() - start, 2),
                       remaining_tokens=response.headers.get("x-ratelimit-remaining-tokens"))
            self.governor.observe(response.headers)
            if response.status_code == 200:
                payload = response.json()
                vectors = np.asarray(payload["embeddings"]["float"], dtype=np.float32)
                if vectors.shape != (len(batch), DIMENSIONS):
                    raise RuntimeError(f"expected {len(batch)} x {DIMENSIONS} vectors, got {vectors.shape}")
                billed = payload.get("meta", {}).get("billed_units", {})
                self.cache.add([name for name, _ in batch], vectors)
                self.usage.record({**row, "billed_input_tokens": billed.get("input_tokens"),
                                   "billed_image_tokens": billed.get("image_tokens"),
                                   "billed_text_tokens": billed.get("text_tokens")}, embedded=len(batch))
                return
            self.usage.record(row)
            if response.status_code == 429:
                self.governor.back_off(float(response.headers.get("Retry-After") or 10))
                continue
            if response.status_code in FATAL_STATUSES:
                self.fatal = f"HTTP {response.status_code}: {response.text[:300]}"
                self.stop.set()
                return
            if response.status_code == 400:
                if len(batch) > 1:  # isolate the image the API refuses; the rest go through
                    half = len(batch) // 2
                    self.send(batch[:half])
                    self.send(batch[half:])
                else:
                    self.failed.append((batch[0][0], response.text[:300]))
                    self.usage.record({**row, "status": "400-single"}, failed=1)
                return
            self.stop.wait(min(60, 2 ** attempt))  # 5xx and the like: transient
        # Out of attempts: leave the batch uncached so the next run retries it.


def run_embedding(pending: pd.DataFrame, raw_dir: Path, encoding: ImageEncoding, embedder: Embedder,
                  *, batch_images: int, concurrency: int, encode_workers: int, unit: str = "images") -> None:
    """Encode ``pending`` items' images in a process pool and send them through ``embedder``'s senders."""
    total = len(pending)
    batches: queue.Queue = queue.Queue(maxsize=4 * concurrency)
    done_sentinel = object()
    stop = embedder.stop

    def produce() -> None:
        tasks = deque(encode_tasks(pending, raw_dir, encoding))
        try:
            with ProcessPoolExecutor(max_workers=encode_workers) as pool:
                futures = deque(pool.submit(_encode_row_group, tasks.popleft())
                                for _ in range(min(len(tasks), 2 * encode_workers)))
                batch: list[tuple[str, bytes]] = []
                while futures and not stop.is_set():
                    items = futures.popleft().result()
                    if tasks:
                        futures.append(pool.submit(_encode_row_group, tasks.popleft()))
                    for item in items:
                        batch.append(item)
                        if len(batch) == batch_images:
                            _put(batch)
                            batch = []
                if batch and not stop.is_set():
                    _put(batch)
                for future in futures:
                    future.cancel()
        finally:
            for _ in range(concurrency):
                batches.put(done_sentinel)

    def _put(batch) -> None:
        while not stop.is_set():
            try:
                batches.put(batch, timeout=1.0)
                return
            except queue.Full:
                continue

    def consume() -> None:
        while True:
            batch = batches.get()
            if batch is done_sentinel:
                return
            if not stop.is_set():
                embedder.send(batch)

    threads = [threading.Thread(target=produce, name="encoder", daemon=True)]
    threads += [threading.Thread(target=consume, name=f"sender-{i}", daemon=True) for i in range(concurrency)]
    for t in threads:
        t.start()

    start = last_report = last_flush = time.monotonic()
    history: deque[tuple[float, int]] = deque([(start, 0)])
    while any(t.is_alive() for t in threads[1:]):
        time.sleep(1.0)
        now = time.monotonic()
        if embedder.cache.pending() >= FLUSH_EVERY_IMAGES or now - last_flush >= FLUSH_EVERY_SECONDS:
            embedder.cache.flush()
            last_flush = now
        if now - last_report >= 60:
            done = embedder.usage.images
            history.append((now, done))
            while len(history) > 1 and history[0][0] < now - 600:
                history.popleft()
            (t0, d0) = history[0]
            rate = (done - d0) / max(now - t0, 1e-9) * 60
            used, n_req = embedder.governor.window_usage()
            eta = (total - done) / rate / 60 if rate > 0 else float("nan")
            print(f"  {done:,}/{total:,} {unit} ({100 * done / total:.1f}%) | {rate:,.0f}/min over the last "
                  f"{(now - t0) / 60:.0f} min | window {used / embedder.governor.tokens_per_minute:.0%} of "
                  f"{embedder.governor.tokens_per_minute:,} tok, {n_req} req | ETA {eta:.1f} h | "
                  f"billed {embedder.usage.image_tokens:,} image / {embedder.usage.billed_input_tokens:,} input tok",
                  flush=True)
            last_report = now
    embedder.cache.flush()


# --- assembly ----------------------------------------------------------------------------------

def assemble(docs: pd.DataFrame, cache: VectorCache, out_path: Path) -> int:
    """Write the per-document feature file from the cache; return how many documents got NaN rows.

    ``docs`` is :func:`embedding_items`' frame, in the split's order, so the output rows are too.
    """
    names, matrix = cache.load_all()
    position = pd.Series(np.arange(len(names)), index=pd.Index(names))
    where = docs["key"].map(position)
    vectors = np.full((len(docs), DIMENSIONS), np.nan, dtype=np.float32)
    have = where.notna().to_numpy()
    vectors[have] = matrix[where[have].astype(int).to_numpy()]
    frame = build_feature_frame(docs["doc_id"], docs["author_id"], vectors, positional_feature_names(DIMENSIONS))
    write_parquet(frame, out_path)
    report_written(frame, out_path)
    missing = int((~have).sum())
    if missing:
        print(f"  {missing:,} documents ({docs.loc[~have, 'key'].nunique():,} distinct inputs) have no vector: NaN rows")
    return missing


# --- records -----------------------------------------------------------------------------------

def append_spend(usage_dir: Path, row: dict, columns: list[str] = SPEND_COLUMNS) -> None:
    path = usage_dir / SPEND_LEDGER
    new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        if new:
            writer.writeheader()
        writer.writerow(row)


def foundry_host() -> tuple[str, str]:
    endpoint, key = _env(FOUNDRY_ENDPOINT_ENV), _env(FOUNDRY_API_KEY_ENV)
    if not endpoint or not key:
        raise SystemExit(f"set {FOUNDRY_ENDPOINT_ENV} (the Foundry endpoint) and {FOUNDRY_API_KEY_ENV} "
                         f"(its key) in the environment or the project's .env")
    parsed = urlparse(endpoint.strip())
    return f"{parsed.scheme}://{parsed.netloc}", key.strip()


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--split-dir", default=None, help="where vision_arena.parquet is (default: data/hf)")
    p.add_argument("--raw", default=None, help="the upstream train-*.parquet directory (default: "
                   "$PROMPT_ANONYMITY_VISION_ARENA_RAW, else the config file)")
    p.add_argument("--out-dir", default=None, help="where the feature file goes (default: data/dist)")
    p.add_argument("--cache-dir", default=None, help="cache root (default: data/.cache)")
    p.add_argument("--format", default="webp", choices=["webp", "jpeg"])
    p.add_argument("--quality", type=int, default=25)
    p.add_argument("--max-side", type=int, default=None,
                   help="downscale the longer side to this many pixels (default: off; measured to cost "
                        "far more fidelity than lowering --quality)")
    p.add_argument("--batch-images", type=int, default=DEFAULT_BATCH_IMAGES)
    p.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    p.add_argument("--encode-workers", type=int, default=available_cpus())
    p.add_argument("--utilisation", type=float, default=DEFAULT_UTILISATION,
                   help="share of the rate limit to aim at (default 0.98)")
    p.add_argument("--tokens-per-minute", type=int, default=DEFAULT_TOKENS_PER_MINUTE,
                   help="starting token limit; replaced by the first response's header")
    p.add_argument("--requests-per-minute", type=int, default=DEFAULT_REQUESTS_PER_MINUTE)
    p.add_argument("--limit", type=int, default=None,
                   help="pilot: embed at most N pending images and write no feature file")
    p.add_argument("--dry-run", action="store_true", help="count cached and pending images; no requests")
    p.add_argument("--assemble-only", action="store_true", help="write the feature file from the cache")
    p.add_argument("--with-text", action="store_true",
                   help=f"the fused variant: each document's user turns (first {TEXT_MAX_CHARS:,} characters) "
                        f"and its image as one input, one joint vector per document ({FUSED_FEATURE})")
    args = p.parse_args()
    if not 1 <= args.batch_images <= MAX_IMAGES_PER_REQUEST:
        p.error(f"--batch-images must be 1..{MAX_IMAGES_PER_REQUEST}")

    encoding = ImageEncoding(args.format, args.quality, args.max_side)
    feature = (FUSED_FEATURE if args.with_text else FEATURE) + encoding.label_suffix()
    unit = "documents" if args.with_text else "images"
    split_path = Path(args.split_dir or hf_dir()) / f"{SOURCE}.parquet"
    out_path = Path(args.out_dir or dist_dir()) / f"{SOURCE}_{feature}.parquet"
    params = {"model": MODEL, "route": EMBED_ROUTE, "input_type": "image", "dimensions": DIMENSIONS,
              "encoding": asdict(encoding), "flatten_alpha": "white", "webp_method": 6,
              "version": EMBED_VERSION}
    if args.with_text:  # the image-only params stay exactly as they were, so their namespace does too
        params.update(input_type=FUSED_INPUT_TYPE, content="text+image", text_max_chars=TEXT_MAX_CHARS,
                      turn_separator=TURN_SEPARATOR)
    cache = VectorCache(Path(args.cache_dir or cache_dir()), params,
                        key_column="item_key" if args.with_text else "image_name")

    docs = embedding_items(split_path, args.with_text)
    if args.assemble_only:
        assemble(docs, cache, out_path)
        return 0

    items = docs.drop_duplicates("key").sort_values(["image_shard", "image_row"])
    done = cache.done_keys()
    pending = items[~items["key"].isin(done)]
    if args.limit is not None:
        pending = pending.head(args.limit)
    what = "distinct (text, image) inputs" if args.with_text else "distinct images"
    print(f"[{SOURCE}] {feature} ({encoding.describe()}): {len(items):,} {what}, "
          f"{len(done & set(items['key'])):,} cached, {len(pending):,} to embed -> cache {cache.dir}")
    if args.with_text:
        print(f"  text capped at {TEXT_MAX_CHARS:,} characters for "
              f"{int((docs['text'].str.len() == TEXT_MAX_CHARS).sum()):,} documents; "
              f"{int((docs['text'] == '').sum()):,} send the image alone")
    if args.dry_run or pending.empty:
        if pending.empty and args.limit is None:
            assemble(docs, cache, out_path)
        return 0

    host, key = foundry_host()
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    usage_dir = project_root() / (FUSED_USAGE_SUBDIR if args.with_text else USAGE_SUBDIR)
    usage = UsageLog(usage_dir / f"requests-{run_id}.csv")
    stop = threading.Event()
    governor = RateGovernor(args.tokens_per_minute, args.requests_per_minute, args.utilisation)
    texts = dict(zip(items["key"], items["text"])) if args.with_text else None
    embedder = Embedder(host, key, governor, cache, usage, stop, texts=texts)

    def on_signal(signum, _frame):
        print(f"  signal {signum}: stopping, flushing the cache", flush=True)
        stop.set()
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    start = time.monotonic()
    raw_dir = Path(args.raw) if args.raw else raw_path(SOURCE)
    outcome = "aborted"
    try:
        run_embedding(pending, raw_dir, encoding, embedder, batch_images=args.batch_images,
                      concurrency=args.concurrency, encode_workers=args.encode_workers, unit=unit)
        outcome = "fatal" if embedder.fatal else ("stopped" if stop.is_set() else "completed")
    finally:
        cache.flush()
        usage.close()
        if usage.requests:
            spend = {
                "finished_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "outcome": outcome, "run_id": run_id, "model": MODEL,
                "encoding": encoding.describe() + (f" + text<={TEXT_MAX_CHARS}" if args.with_text else ""),
                "images_embedded": usage.images, "images_failed": usage.failed, "requests": usage.requests,
                "image_tokens": usage.image_tokens, "billed_input_tokens": usage.billed_input_tokens,
                "cost_usd_image_tokens": round(usage.image_tokens * PRICE_PER_MTOK / 1e6, 4),
                "cost_usd_billed_input_tokens": round(usage.billed_input_tokens * PRICE_PER_MTOK / 1e6, 4),
                "elapsed_s": round(time.monotonic() - start)}
            if args.with_text:
                spend.update(text_tokens=usage.text_tokens,
                             cost_usd_text_tokens=round(usage.text_tokens * PRICE_PER_MTOK / 1e6, 4))
            append_spend(usage_dir, spend, FUSED_SPEND_COLUMNS if args.with_text else SPEND_COLUMNS)
        if embedder.failed:
            with (usage_dir / f"failed-{run_id}.csv").open("w", newline="") as f:
                csv.writer(f).writerows([(cache.key_column, "error"), *embedder.failed])

    text_bill = (f" + {usage.text_tokens:,} text tokens (${usage.text_tokens * PRICE_PER_MTOK / 1e6:.2f})"
                 if args.with_text else "")
    print(f"[{SOURCE}] {outcome}: {usage.images:,} {unit} in {usage.requests:,} requests, "
          f"{(time.monotonic() - start) / 3600:.2f} h; billed {usage.image_tokens:,} image tokens "
          f"(${usage.image_tokens * PRICE_PER_MTOK / 1e6:.2f}){text_bill} / {usage.billed_input_tokens:,} input "
          f"tokens (${usage.billed_input_tokens * PRICE_PER_MTOK / 1e6:.2f}); {usage.failed} {unit} refused",
          flush=True)
    if embedder.fatal:
        print(f"[{SOURCE}] stopped on a configuration error: {embedder.fatal}", file=sys.stderr)
        return 1
    if outcome == "completed" and args.limit is None:
        remaining = set(items["key"]) - cache.done_keys()
        if remaining:
            print(f"[{SOURCE}] {len(remaining):,} {unit} still uncached (transient failures or refusals); "
                  f"re-run to retry them -- writing the feature file with NaN rows meanwhile")
        assemble(docs, cache, out_path)
    return 0 if outcome == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
