"""WildChat source adapter: raw WildChat parquet -> normalized per-conversation documents.

Reads the raw ``allenai/WildChat-4.8M`` parquet directly (no dependency on the
``wildchat/`` scripts or their intermediate CSVs), keeping every conversation on one of the
studied models -- *without* the old "identity used >=2 distinct models" requirement. A
document is one conversation, represented by its ``user``-role turns; a conversation with no
user turns is dropped (this is a user-prompt dataset, so we do not fall back to assistant text
as the legacy pipeline did).

Two filters are applied here rather than downstream, because both are cheap at load time and
both shrink what the (memory-hungry) second pass has to hold:

* **Programmatic clients are dropped** -- a conversation posted by ``gradio_client``, ``httpx``,
  ``node``, or any other HTTP library is not one person's writing (see
  :func:`data.identity.is_programmatic_user_agent`). This runs *before* the identity counting
  below and before all deduplication, so bot traffic can neither qualify an identity for the
  ``min_docs`` floor nor influence which affixes look like shared boilerplate.
* **Identities below ``min_docs`` are dropped**, which is safe because every downstream stage
  only ever removes further rows.

This adapter returns the **raw** user turns as a list (column ``turns_raw``) and does *no*
text cleaning -- cleaning is a separate, parallelized stage in ``build_dataset.py`` so it
can use all CPU cores.

Because the raw model slice is millions of conversations of which ~99% come from
single-conversation identities, the loader runs in two passes: pass 1 reads only the tiny
identity columns to find identities with at least ``min_docs`` conversations, and pass 2
streams the (large) ``conversation`` column but keeps only those identities' rows.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm

from .common import model_owner, normalize_language
from .identity import is_programmatic_user_agent, wildchat_device_info, wildchat_identity

# The studied WildChat models. These are pooled with no roles attached: no model is the "known"
# side and no model is the "unknown" side. The old by-model linkage split (gpt-4o = labeled,
# gpt-4.1-mini = anonymous) is gone, so the list is simply which slices of WildChat to include,
# and the downstream split is free to be assigned on any axis (e.g. by day).
WILDCHAT_MODELS = [
    "gpt-4o-2024-08-06",
    "gpt-4.1-mini-2025-04-14",
    "gpt-4o-mini-2024-07-18",
]


def _row_identity(hashed_ip, header, ua_map) -> str:
    """Return the WildChat request-fingerprint identity for one row."""
    header = header or {}
    device_info = wildchat_device_info(header.get("user-agent"), ua_map)
    return wildchat_identity(hashed_ip, header.get("accept-language") or "", device_info)


def _iso_utc(ts) -> str | None:
    """Coerce a naive/aware timestamp to an ISO-8601 UTC string (``None`` passes through)."""
    if ts is None:
        return None
    ts = pd.Timestamp(ts)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def load_wildchat_documents(
    raw_path: str | Path,
    ua_map: dict[str, str],
    *,
    models: list[str] = WILDCHAT_MODELS,
    min_docs: int = 2,
    drop_programmatic: bool = True,
    batch_size: int = 2000,
    max_batches: int | None = None,
) -> pd.DataFrame:
    """Load WildChat as a normalized (uncleaned) document frame.

    Columns: ``doc_id, source, identity, turns_raw, languages, model, model_owner, agent,
    started_at, ended_at, repo_id, user_id`` (``repo_id``/``user_id`` are ``None`` for
    WildChat and exist only so the downstream cleaner has a uniform signature).

    ``drop_programmatic`` (default on) discards conversations posted by HTTP clients rather than
    typed into a browser, in **both** passes -- so a bot's conversations do not count toward its
    identity's ``min_docs`` total either. Pass ``False`` only to measure how much bot traffic the
    corpus carries; keeping it costs the author labels their meaning (see
    :func:`data.identity.is_programmatic_user_agent`).

    ``max_batches`` caps the number of non-empty pass-2 batches (pass 1 is never capped);
    it exists only for quick end-to-end smoke tests on a slice of the data.
    """
    dataset = ds.dataset(str(raw_path), format="parquet")
    model_filter = ds.field("model").isin(models)

    # Millions of rows share a few thousand distinct user-agents, so classify each string once and
    # reuse the verdict rather than re-running the pattern match per row.
    bot_cache: dict[str, bool] = {}

    def is_bot(header) -> bool:
        """True if this row's request came from code, not a browser (always False if disabled)."""
        if not drop_programmatic:
            return False
        user_agent = (header or {}).get("user-agent") or ""
        verdict = bot_cache.get(user_agent)
        if verdict is None:
            verdict = bot_cache[user_agent] = is_programmatic_user_agent(user_agent, ua_map)
        return verdict

    # Pass 1: identities only -> drop programmatic clients, then keep identities with
    # >= min_docs of the *remaining* conversations.
    meta = dataset.scanner(columns=["hashed_ip", "header"], filter=model_filter).to_table()
    headers = meta.column("header").to_pylist()
    identities = [_row_identity(h, hd, ua_map)
                  for h, hd in zip(meta.column("hashed_ip").to_pylist(), headers)
                  if not is_bot(hd)]
    n_programmatic = len(headers) - len(identities)
    counts = pd.Series(identities).value_counts()
    keep = set(counts[counts >= max(1, min_docs)].index)
    print(f"  WildChat: {len(headers):,} convs on {len(models)} model(s); "
          f"{n_programmatic:,} ({n_programmatic / max(len(headers), 1) * 100:.1f}%) from programmatic clients")
    print(f"  WildChat: {len(identities):,} convs / {counts.size:,} identities; "
          f"{len(keep):,} identities with >={min_docs} convs")

    # Pass 2: stream the conversation column, collecting RAW user turns for kept identities.
    columns = ["conversation_hash", "conversation", "hashed_ip", "header",
               "language", "timestamp", "model"]
    scanner = dataset.scanner(columns=columns, filter=model_filter, batch_size=batch_size)
    records = []
    processed = 0  # non-empty batches (the filter yields many empty ones for other-model row groups)
    for batch in tqdm(scanner.to_batches(), desc="  WildChat pass 2", unit="batch"):
        if batch.num_rows == 0:
            continue
        cols = {name: batch.column(name).to_pylist() for name in columns}
        for i in range(batch.num_rows):
            if is_bot(cols["header"][i]):
                continue
            identity = _row_identity(cols["hashed_ip"][i], cols["header"][i], ua_map)
            if identity not in keep:
                continue
            conv = cols["conversation"][i] or []
            turns = [t["content"] for t in conv
                     if t.get("role") == "user" and t.get("content")]
            if not turns:
                continue  # user-prompt dataset: no user turns -> drop
            model = cols["model"][i]
            # WildChat user turns carry no per-turn timestamp, but the assistant replies all do
            # and turns strictly alternate, so the assistant-turn times give the conversation's
            # real span; fall back to the conversation-level timestamp if none is timed.
            asst_ts = [t["timestamp"] for t in conv
                       if t.get("role") == "assistant" and t.get("timestamp") is not None]
            conv_ts = cols["timestamp"][i]
            started_at = min(asst_ts) if asst_ts else conv_ts
            ended_at = max(asst_ts) if asst_ts else conv_ts
            lang = normalize_language(cols["language"][i])
            records.append({
                "doc_id": f"wc-{cols['conversation_hash'][i]}",
                "source": "wildchat",
                "identity": identity,
                "turns_raw": turns,
                "languages": [lang] if lang else [],   # single detected language; primary is first
                "model": model,
                "model_owner": model_owner(model),
                "agent": None,
                "started_at": _iso_utc(started_at),
                "ended_at": _iso_utc(ended_at),
                "repo_id": None,
                "user_id": None,
            })
        processed += 1
        if max_batches is not None and processed >= max_batches:
            break
    return pd.DataFrame.from_records(records)
