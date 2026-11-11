"""VisionArena-Chat source adapter: raw ``lmarena-ai/VisionArena-Chat`` parquet -> normalized documents.

VisionArena-Chat is ~200k conversations between Chatbot Arena users and vision-language models
(2024-06-19 to 2024-11-10). Every conversation carries **exactly one image**, so a document here is
a prompt *and* an image, and the split exists for **multimodal** linkage attacks. A document is one
conversation, represented by its ``user``-role turns (``turns_raw``) plus the image's identity and
where to find its bytes; the bytes themselves are never copied (78.5 GiB of the 78.8 GiB upstream).

**Local-only.** The upstream license forbids transferring the dataset to any third party, so this
split is built into ``data/dist`` and is never part of the published HuggingFace dataset (see
``build_dataset.LOCAL_ONLY_SOURCES``). Its license also forbids attempting to identify
individuals; linkage against the pseudonymous ids is the use this benchmark makes of it.

What the adapter does, in order:

1. **Pass 1 reads only the small columns** (ids, timestamps, the stock-image flag, the image file
   name -- a few MB) and never an image byte.
2. **Stock-image conversations are dropped** (``is_preset``: the arena's "random image" button,
   13.8% of conversations, 1,818 images drawn from NewYorker / TextVQA / ChartQA / WikiArt /
   DocVQA). The image is not the user's, and one stock image is shared by up to 102 users, so it
   would forge links in the image channel. Dropped *before* identities are counted or merged.
3. **Split identities are merged** (:func:`merge_split_identities`). The upstream ``user_id`` is
   IP-based, and one person regularly appears under two ids: an uploaded image used by exactly
   two ids is, in the median case, sent by both **6 seconds apart**, and some id pairs share 93
   images. Two ids are linked when they share an uploaded image that at most
   :data:`MERGE_MAX_IDS_PER_IMAGE` ids use, sent by both within :data:`MERGE_WINDOW_SECONDS`;
   linked ids are unioned, and a group larger than :data:`MERGE_MAX_GROUP_IDS` is left unmerged
   (chaining through a popular image, not one person). Measured on the uploaded conversations:
   62,162 raw ids -> 53,362 identities (8,139 groups, 3 oversized and left alone), and merged ids
   agree on their dominant language 81.3% of the time against 25.2% for random pairs -- evidence
   the rule itself does not use.
4. **Identities below ``min_docs`` are dropped** (counted after the merge), which is safe because
   every downstream stage only removes further rows.
5. **Pass 2 reads the conversation text and category flags** for the kept rows only.

Columns returned, beyond the ones every adapter returns (``doc_id, source, identity, turns_raw,
languages, model, model_owner, agent, started_at, ended_at, repo_id, user_id``):

* ``image_name`` -- the upstream image file name **verbatim** (``<32 hex>.png``). It is the
  upstream's own key for the image, not an MD5 of the stored bytes (0 of 100 matched), and two
  conversations about one image carry the same name. **It is an exact-image identity key**: an
  attack that reads it is a copy detector, so use it deliberately.
* ``image_shard`` / ``image_row`` -- the upstream parquet file and 0-based row the image bytes are
  in, so a featurizer can fetch them (:func:`read_image_bytes`).
* ``n_upstream_ids`` -- how many upstream ``user_id`` values were merged into this identity.
* ``category_*`` -- the upstream question-category flags. ``category_code`` is upstream
  ``is_code``; upstream ``code`` is dropped (12,724 nulls, otherwise ~99% equal to ``is_code``).
  **``category_refusal`` flags the model's reply, not the request**: when one user sent one prompt
  and image to several models, the flag disagrees across the models in 92.1% of such groups.

``turns_raw`` is uncleaned; cleaning, dedup, languages and the image header pass run in
``build_dataset.py``. ``tstamp`` is the only timestamp, so ``started_at == ended_at``; whether it
marks a conversation's start or end is not documented upstream.
"""

from __future__ import annotations

import io
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

from .common import model_owner, normalize_language

SOURCE = "vision_arena"

# --- identity merge (see the module docstring, step 3) ---
#: Two ids are linked only through an image that at most this many ids ever sent. An image used by
#: more is a genuinely popular one, and linking through it would chain strangers together.
MERGE_MAX_IDS_PER_IMAGE = 3
#: ...and only when the two ids sent it this close together. Measured on images used by exactly two
#: ids: the nearest pair of their conversations is a median 6 s apart, 71% under a minute.
MERGE_WINDOW_SECONDS = 600
#: A merged group with more ids than this is left unmerged and reported instead (3 groups of 6).
MERGE_MAX_GROUP_IDS = 5

#: Upstream question-category flags kept, and the column each becomes. ``is_code`` is the complete
#: one of the two near-duplicate code flags; upstream ``code`` has nulls and is not kept.
CATEGORY_COLUMNS = {
    "captioning": "category_captioning",
    "is_code": "category_code",
    "creative_writing": "category_creative_writing",
    "diagram": "category_diagram",
    "entity_recognition": "category_entity_recognition",
    "homework": "category_homework",
    "humor": "category_humor",
    "ocr": "category_ocr",
    "refusal": "category_refusal",
}

#: The columns this source adds to the common schema, in output order.
EXTRA_COLUMNS = [
    "image_name", "image_shard", "image_row", "image_width", "image_height", "image_icc_profile",
    "n_upstream_ids", *CATEGORY_COLUMNS.values(),
]

#: The upstream label for "could not tell" -- treated as no label, so the language stage decides.
UPSTREAM_UNKNOWN_LANGUAGE = "unknown"

#: ``image_icc_profile`` when an image embeds a colour profile whose description cannot be read
#: (distinct from null, which means the image carries no profile at all).
UNREADABLE_ICC_PROFILE = "<unreadable>"

_INDEX_COLUMNS = ["conversation_id", "user_id", "tstamp", "is_preset", "images.list.element.path"]
_TEXT_COLUMNS = ["conversation_id", "conversation", "model", "language", "categories"]


def shard_paths(raw_dir: str | Path) -> list[Path]:
    """The upstream ``train-*.parquet`` shards, in name order (the order ``image_row`` refers to)."""
    paths = sorted(Path(raw_dir).glob("train-*.parquet"))
    if not paths:
        raise SystemExit(f"no VisionArena train-*.parquet shards under {raw_dir}")
    return paths


def read_upstream_index(raw_dir: str | Path) -> pd.DataFrame:
    """Pass 1: one row per upstream conversation, from the small columns only (no image bytes).

    Columns: ``conversation_id, user_id, tstamp, is_preset, image_name, image_shard, image_row``.
    ``image_row`` is the conversation's 0-based row within ``image_shard``. Every upstream
    conversation has exactly one image, which is asserted rather than assumed.
    """
    frames = []
    for path in shard_paths(raw_dir):
        table = pq.ParquetFile(path).read(columns=_INDEX_COLUMNS)
        images = table.column("images").to_pylist()
        if any(len(entry) != 1 for entry in images):
            raise ValueError(f"{path.name}: a conversation without exactly one image")
        frames.append(pd.DataFrame({
            "conversation_id": table.column("conversation_id").to_pylist(),
            "user_id": table.column("user_id").to_numpy(),
            "tstamp": table.column("tstamp").to_numpy(),
            "is_preset": table.column("is_preset").to_numpy(),
            "image_name": [entry[0]["path"] for entry in images],
            "image_shard": path.name,
            "image_row": np.arange(table.num_rows, dtype=np.int64),
        }))
    return pd.concat(frames, ignore_index=True)


def merge_split_identities(
    index: pd.DataFrame, *,
    window_seconds: float = MERGE_WINDOW_SECONDS,
    max_ids_per_image: int = MERGE_MAX_IDS_PER_IMAGE,
    max_group_ids: int = MERGE_MAX_GROUP_IDS,
) -> pd.Series:
    """Map every upstream ``user_id`` in ``index`` to the representative id of its merged group.

    ``index`` must already exclude stock-image conversations. Two ids are linked when they share an
    image that at most ``max_ids_per_image`` ids sent and their nearest conversations on it are at
    most ``window_seconds`` apart; links are unioned (union-find), and the representative is the
    group's smallest id, so the mapping is deterministic. A group of more than ``max_group_ids``
    ids is not merged -- each of its ids maps to itself -- and the count of such groups is printed.
    """
    ids_per_image = index.groupby("image_name")["user_id"].nunique()
    shared = ids_per_image.index[(ids_per_image >= 2) & (ids_per_image <= max_ids_per_image)]
    candidates = index[index["image_name"].isin(set(shared))]

    parent: dict[int, int] = {}

    def find(x: int) -> int:
        root = x
        while parent.get(root, root) != root:
            root = parent[root]
        while parent.get(x, x) != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    n_links = 0
    for _, rows in candidates.groupby("image_name"):
        times = {uid: np.sort(group["tstamp"].to_numpy()) for uid, group in rows.groupby("user_id")}
        uids = sorted(times)
        for i, a in enumerate(uids):
            for b in uids[i + 1:]:
                gap = np.abs(times[a][:, None] - times[b][None, :]).min()
                if gap <= window_seconds:
                    n_links += 1
                    ra, rb = find(int(a)), find(int(b))
                    if ra != rb:
                        parent[max(ra, rb)] = min(ra, rb)

    all_ids = pd.unique(index["user_id"])
    roots = pd.Series({int(uid): find(int(uid)) for uid in all_ids})
    sizes = roots.map(roots.value_counts())
    oversized = sizes > max_group_ids
    representative = roots.where(~oversized, roots.index.to_series())
    n_groups = int((roots.value_counts() > 1).sum())
    n_oversized = int(roots[oversized].nunique())
    print(f"  VisionArena: {n_links:,} id links -> {n_groups:,} merged groups "
          f"({n_oversized:,} over {max_group_ids} ids left unmerged); "
          f"{len(all_ids):,} upstream ids -> {representative.nunique():,} identities")
    return representative


def _user_turns(conversation) -> list[str]:
    """The non-empty ``user`` messages of one upstream conversation, in order.

    Upstream nests each message in a one-element list (``[[{content, role}], ...]``); that holds
    for every row, and a longer inner list is an error rather than something to guess about.
    """
    turns = []
    for message in conversation or []:
        if len(message) != 1:
            raise ValueError(f"unexpected message nesting of length {len(message)}")
        if message[0].get("role") == "user" and message[0].get("content"):
            turns.append(message[0]["content"])
    return turns


def _iso_utc(seconds: float) -> str:
    """A POSIX timestamp in seconds as an ISO-8601 UTC string, at microsecond precision.

    Rounded to the microsecond because a float's nanosecond digits are representation noise
    (``1721637562.4643`` would otherwise print as ``...22.464299917``).
    """
    return pd.Timestamp(round(seconds * 1e6), unit="us", tz="UTC").isoformat()


def load_vision_arena_documents(
    raw_dir: str | Path, *, min_docs: int = 2, merge_identities: bool = True,
) -> pd.DataFrame:
    """Load VisionArena-Chat as a normalized (uncleaned) document frame; see the module docstring.

    ``merge_identities=False`` keeps every upstream ``user_id`` as its own identity
    (``--keep-split-identities``), for measuring what the merge changes. ``image_width``,
    ``image_height`` and ``image_icc_profile`` are returned empty; they are filled by
    :func:`read_image_headers` once the build has decided which rows survive.
    """
    index = read_upstream_index(raw_dir)
    n_all = len(index)
    index = index[~index["is_preset"]].reset_index(drop=True)
    print(f"  VisionArena: {n_all:,} conversations; {n_all - len(index):,} on stock images dropped")

    if merge_identities:
        representative = merge_split_identities(index)
        index["identity_id"] = index["user_id"].map(representative)
    else:
        index["identity_id"] = index["user_id"]
    index["n_upstream_ids"] = index["identity_id"].map(
        index.groupby("identity_id")["user_id"].nunique()).astype("int32")

    sizes = index.groupby("identity_id")["conversation_id"].transform("size")
    index = index[sizes >= max(1, min_docs)].set_index("conversation_id")
    print(f"  VisionArena: {len(index):,} conversations / {index['identity_id'].nunique():,} identities "
          f"with >= {min_docs} conversations")

    records = []
    for path in tqdm(shard_paths(raw_dir), desc="  VisionArena pass 2", unit="shard"):
        table = pq.ParquetFile(path).read(columns=_TEXT_COLUMNS)
        ids = table.column("conversation_id").to_pylist()
        wanted = [i for i, cid in enumerate(ids) if cid in index.index]
        if not wanted:
            continue
        table = table.take(wanted)
        cols = {name: table.column(name).to_pylist() for name in ("conversation_id", "conversation",
                                                                  "model", "language", "categories")}
        for k in range(table.num_rows):
            turns = _user_turns(cols["conversation"][k])
            if not turns:
                continue  # user-prompt dataset: no user text -> drop (18 upstream conversations)
            row = index.loc[cols["conversation_id"][k]]
            language = cols["language"][k]
            language = None if language == UPSTREAM_UNKNOWN_LANGUAGE else normalize_language(language)
            categories = cols["categories"][k] or {}
            model = cols["model"][k]
            records.append({
                "doc_id": f"va-{cols['conversation_id'][k]}",
                "source": SOURCE,
                "identity": f"user:{int(row['identity_id'])}",
                "turns_raw": turns,
                "languages": [language] if language else [],
                "model": model,
                "model_owner": model_owner(model),
                "agent": None,
                "started_at": _iso_utc(row["tstamp"]),
                "ended_at": _iso_utc(row["tstamp"]),
                "repo_id": None,
                "user_id": None,  # the cleaner's SWE-chat scrub token, not the upstream user id
                "image_name": row["image_name"],
                "image_shard": row["image_shard"],
                "image_row": int(row["image_row"]),
                "image_width": None,
                "image_height": None,
                "image_icc_profile": None,
                "n_upstream_ids": int(row["n_upstream_ids"]),
                **{out: bool(categories.get(flag)) for flag, out in CATEGORY_COLUMNS.items()},
            })
    return pd.DataFrame.from_records(records)


# --- image bytes and headers ---------------------------------------------------------------------

def read_image_bytes(raw_dir: str | Path, image_shard: str, image_rows) -> list[bytes]:
    """The PNG bytes of ``image_rows`` (0-based rows of ``image_shard``), in the order given.

    Reads only the row groups that hold those rows, and only the ``images`` column. This is how a
    featurizer gets pixels for a document: by its ``image_shard`` / ``image_row``, never by copying
    the bytes into the split.
    """
    parquet = pq.ParquetFile(Path(raw_dir) / image_shard)
    starts = np.cumsum([0] + [parquet.metadata.row_group(g).num_rows
                              for g in range(parquet.metadata.num_row_groups)])
    rows = np.asarray(list(image_rows), dtype=np.int64)
    groups = np.searchsorted(starts, rows, side="right") - 1
    out: dict[int, bytes] = {}
    for group in np.unique(groups):
        images = parquet.read_row_group(int(group), columns=["images"]).column("images").to_pylist()
        for row in rows[groups == group]:
            out[int(row)] = images[int(row - starts[group])][0]["bytes"]
    return [out[int(row)] for row in rows]


def _png_header(data: bytes) -> tuple[int, int, str | None]:
    """``(width, height, icc_profile_description)`` of a PNG, without decoding its pixels."""
    from PIL import Image, ImageCms

    image = Image.open(io.BytesIO(data))  # lazy: reads the header and ancillary chunks only
    width, height = image.size
    icc = image.info.get("icc_profile")
    if not icc:
        return width, height, None
    try:
        description = ImageCms.getProfileDescription(ImageCms.ImageCmsProfile(io.BytesIO(icc))).strip()
    except Exception:  # noqa: BLE001 - a malformed profile is data, not a build failure
        description = ""
    return width, height, description or UNREADABLE_ICC_PROFILE


def _shard_headers(args) -> pd.DataFrame:
    """Worker: image headers for one shard's requested rows. Top-level so it pickles."""
    raw_dir, image_shard, doc_ids, rows, names = args
    parquet = pq.ParquetFile(Path(raw_dir) / image_shard)
    starts = np.cumsum([0] + [parquet.metadata.row_group(g).num_rows
                              for g in range(parquet.metadata.num_row_groups)])
    rows = np.asarray(rows, dtype=np.int64)
    groups = np.searchsorted(starts, rows, side="right") - 1
    width = np.zeros(len(rows), dtype=np.int32)
    height = np.zeros(len(rows), dtype=np.int32)
    icc: list[str | None] = [None] * len(rows)
    for group in np.unique(groups):
        images = parquet.read_row_group(int(group), columns=["images"]).column("images").to_pylist()
        for k in np.flatnonzero(groups == group):
            entry = images[int(rows[k] - starts[group])][0]
            if entry["path"] != names[k]:  # the locator must point at the image the row names
                raise ValueError(f"{image_shard} row {rows[k]}: holds {entry['path']}, expected {names[k]}")
            width[k], height[k], icc[k] = _png_header(entry["bytes"])
    return pd.DataFrame({"doc_id": doc_ids, "image_width": width, "image_height": height,
                         "image_icc_profile": icc})


def read_image_headers(frame: pd.DataFrame, raw_dir: str | Path, *, workers: int = 1) -> pd.DataFrame:
    """Fill ``image_width`` / ``image_height`` / ``image_icc_profile`` for every row of ``frame``.

    Reads the image bytes of the surviving rows (one process per shard, ``workers`` at a time) but
    decodes no pixels: PIL opens each PNG lazily and only its header and colour profile are read.
    Parquet reads a column a whole row group at a time, so this still reads most of the upstream
    78.5 GiB of image bytes -- it is the one heavy stage of the build and runs last for that reason.
    Each row's ``image_shard`` / ``image_row`` is also verified to hold the image ``image_name``
    names, which is what makes the locator trustworthy for anything that later fetches pixels.
    """
    jobs = [(str(raw_dir), shard, list(g["doc_id"]), list(g["image_row"]), list(g["image_name"]))
            for shard, g in frame.groupby("image_shard")]
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            parts = list(tqdm(pool.map(_shard_headers, jobs), total=len(jobs),
                              desc=f"  image headers ({workers} procs)", unit="shard"))
    else:
        parts = [_shard_headers(job) for job in tqdm(jobs, desc="  image headers", unit="shard")]
    headers = pd.concat(parts, ignore_index=True)
    out = frame.drop(columns=["image_width", "image_height", "image_icc_profile"]).merge(
        headers, on="doc_id", how="left", validate="one_to_one")
    if out["image_width"].isna().any():
        raise ValueError("image header pass left rows without a header")
    out["image_width"] = out["image_width"].astype("int32")
    out["image_height"] = out["image_height"].astype("int32")
    return out
