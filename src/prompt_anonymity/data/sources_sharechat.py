"""ShareChat source adapter: raw ``tucnguyen/ShareChat`` CSVs -> normalized per-conversation documents.

`ShareChat <https://huggingface.co/datasets/tucnguyen/ShareChat>`_ is a corpus of publicly
*shared* chatbot conversations scraped from five platforms' share links (ChatGPT, Claude, Gemini,
Grok, Perplexity). It ships one CSV per platform, and each **row is one message** rather than one
conversation, so this adapter groups by the share ``url`` -- the conversation's identifier -- and
keeps that conversation's ``role == "user"`` messages in ``message_index`` order.

**ShareChat has no user id, and this adapter invents none.** The share link says which
conversation a message belongs to and nothing about who wrote it: two links may or may not be the
same person, and the upstream data cannot tell us. So every row's ``identity`` is ``None`` and the
built split's ``author_id`` column is null throughout (see
:mod:`~prompt_anonymity.data.build_dataset`, which skips pseudonymization and the ``min_docs``
floor for this source). That makes ShareChat unusable as a *labeled* side of a linkage experiment
and exactly right as a pool of **out-of-set documents** -- traffic whose author is, by
construction, absent from any known side.

Two ShareChat-specific stages have no analogue in the other adapters:

* **Upstream redaction markers are removed** (:func:`strip_upstream_redactions`). ShareChat was
  de-identified with Microsoft Presidio before release, which left ``<REDACTED>`` in ~30% of user
  turns (plus a smaller number of ``<DATE_TIME>``). Neither WildChat nor SWE-chat carries such a
  token, so leaving it in would make ShareChat separable from them on a literal string -- fatal
  here, because this is the out-of-set pool an open-set detector is measured against, and it would
  be detecting the *corpus* rather than a stranger. The markers are deleted and the surrounding
  text kept. ShareChat's third Presidio placeholder, ``<URL>``, is **kept**: it is already this
  project's own placeholder for the same thing (:data:`~prompt_anonymity.data.text_cleaning.URL_PLACEHOLDER`),
  so it is indistinguishable from what our scrubber would have written.
* **Timestamps are per-platform** (:data:`PLATFORM_TIME_COLUMNS`). Each platform exports a
  different time field -- some per message, some per conversation, some in a human-readable format,
  and Claude none at all -- so there is one rule per platform rather than one shared column.

The adapter returns the **raw** (uncleaned) user turns as a list (``turns_raw``); text cleaning is
the parallelized stage in ``build_dataset.py``, identical to every other source.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .common import model_owner, normalize_language, ordered_languages

#: The platforms ShareChat scraped, one CSV each. Every one is built into the single ``sharechat``
#: split (the platform is recorded per document in ``model`` / ``model_owner``), the same way
#: WildChat pools its three models into one split rather than splitting on them.
SHARECHAT_PLATFORMS = ("chatgpt", "claude", "gemini", "grok", "perplexity")

#: platform -> its CSV in the raw ShareChat repo. All five are required; a missing one is an error
#: naming the file, rather than a split that silently omits a platform.
PLATFORM_FILES = {p: f"{p}_results_final_language_filtered.csv" for p in SHARECHAT_PLATFORMS}

#: platform -> the organization serving it, i.e. the party that actually observed the conversation.
#: This is set from the **platform**, not from :func:`~prompt_anonymity.data.common.model_owner`,
#: because the platform is the reliable fact: two platforms ship no model column at all, and the
#: model strings that do exist are display names (``2.0 Flash``) or codenames (``grok-2``) that a
#: model-id lookup does not recognize. Where both are available they agree.
PLATFORM_OWNER = {
    "chatgpt": "OpenAI",
    "claude": "Anthropic",
    "gemini": "Google",
    "grok": "xAI",
    "perplexity": "Perplexity",
}

#: Presidio placeholders left in the text by ShareChat's own de-identification pass, deleted here.
#: ``<URL>`` is deliberately **not** in this list -- see the module docstring.
UPSTREAM_REDACTION_TOKENS = ("<REDACTED>", "<DATE_TIME>")
_REDACTION_RE = re.compile("|".join(re.escape(t) for t in UPSTREAM_REDACTION_TOKENS))

#: Values that appear in a platform's ``model`` column but are not model ids. Two kinds:
#:
#: * **conversation-role markers** -- Grok labels its own rows ``human`` / ``ASSISTANT``;
#: * **ChatGPT export field names that leaked into the value**: ``default_model_slug`` (700,836
#:   rows), ``requested_model_slug`` (6,565) and ``parent_id`` (4) are keys of the ChatGPT share
#:   JSON, not models. The first also carries real meaning -- "the account's default was used" --
#:   but it does not name which model that was.
#:
#: A conversation whose model column holds nothing else falls back to the platform name, which
#: says the same thing plainly. Kept as an explicit list rather than a shape rule (e.g. "contains
#: an underscore"): ChatGPT's own ``gpt4t_1`` is a real model id with an underscore in it, so a
#: rule would have eaten it. The full raw vocabulary of all five platforms was enumerated when
#: this was written, so this list is closed rather than a guess.
NON_MODEL_VALUES = frozenset({
    "human", "assistant", "model", "user", "llm", "none", "nan", "",
    "default_model_slug", "requested_model_slug", "parent_id",
})

#: platform -> ``(granularity, column, fallback_column)`` for the conversation's time span.
#:
#: ``"message"`` means the column carries a per-message time, so the conversation's span is the
#: min and max over its **user** messages (the same rule SWE-chat uses); ``"conversation"`` means
#: one value describes the whole conversation, which then serves as both ends. ``None`` means the
#: platform exports no usable time at all.
#:
#: The choice per platform, where more than one column was available:
#:
#: * **chatgpt** -- ``message_create_time`` (per message, ``ts:<unix seconds>``) over
#:   ``create_time`` (one value per conversation), because a per-message time gives a real span;
#:   ``create_time`` is the fallback for the ~1% of user messages that carry no time of their own.
#: * **grok** -- ``message_create_time`` over ``last_updated``, for the same reason. Grok's
#:   ``last_updated`` is constant per conversation and records when the *share page* was last
#:   touched, not when the conversation happened.
#: * **gemini** -- ``created_at`` over ``published_at``: the first is when the conversation was
#:   held, the second when the user chose to share it, which can be days later and is an act of
#:   publishing rather than of chatting.
#: * **perplexity** -- ``last_updated`` is the only time column, and it is a **date with no
#:   time of day**, so these documents are ordered to the day and no finer.
#: * **claude** -- no time column exists; both ends are ``None``.
PLATFORM_TIME_COLUMNS = {
    "chatgpt": ("message", "message_create_time", "create_time"),
    "claude": (None, None, None),
    "gemini": ("conversation", "created_at", None),
    "grok": ("message", "message_create_time", None),
    "perplexity": ("conversation", "last_updated", None),
}

#: Rows per chunk when streaming a platform CSV. ShareChat's CSVs reach 2.3 GB and the assistant
#: replies are ~8x the user prompts by volume, so each chunk is filtered to user rows before
#: anything is retained -- this bounds peak memory by the *user* text rather than by the file.
READ_CHUNK_ROWS = 200_000

#: ChatGPT writes its per-message time as ``ts:<unix seconds>.<fraction>``.
_UNIX_TS_RE = re.compile(r"^\s*ts:")

#: Timestamps outside this window are treated as corrupt and discarded (the document becomes
#: undated) rather than published as-is. The lower bound is just before ChatGPT's public launch,
#: the earliest moment any of these five products could have been used; the upper is the pinned
#: ShareChat revision's own publication date, since a scraped conversation cannot postdate the
#: scrape. Both are constants rather than "now", so a rebuild is reproducible.
#:
#: This exists because ChatGPT's ``message_create_time`` carries a handful of corrupt unix values
#: -- 33 of its 483,894 user messages land between 2059 and 2282, one of them far enough out to
#: overflow a nanosecond timestamp outright. Every other column of every other platform parses
#: entirely inside the window. A bad value is nulled rather than clamped: it tells us nothing
#: about the real time, and a document dated 2282 would sit at the end of every chronological
#: ordering forever.
PLAUSIBLE_TIME_WINDOW = ("2022-11-01", "2026-05-06")


def strip_upstream_redactions(text: str) -> str:
    """Delete ShareChat's Presidio markers (:data:`UPSTREAM_REDACTION_TOKENS`), keeping the text.

    ``"My name is <REDACTED> and I live in <REDACTED>."`` becomes
    ``"My name is  and I live in ."``; the doubled spaces are squeezed later by
    :func:`~prompt_anonymity.data.text_cleaning.scrub_identifiers`, which every source runs through.

    The token is removed rather than translated into this project's placeholder vocabulary because
    ``<REDACTED>`` stands for a *union* of entity types (names, phone numbers, credit cards,
    addresses) that no single placeholder of ours means, and because neither other corpus redacts
    those categories at all -- a WildChat prompt containing a person's name keeps it. Deleting
    therefore makes ShareChat's text more like the others', not less.
    """
    if not isinstance(text, str):
        return ""
    return _REDACTION_RE.sub("", text)


def _share_slug(url: str) -> str:
    """The conversation's id within its platform: the share URL's last path segment.

    Query strings and fragments are dropped first (Perplexity appends ``?s=u``). The result is the
    upstream's own identifier, passed through the way the other adapters pass through WildChat's
    ``conversation_hash`` and SWE-chat's ``session_id``. Its shape varies by platform -- a UUID, a
    12-hex token, or a topic slug with an id appended -- and the rare collision between two
    different URLs is resolved by ``build_dataset``'s ``_uniquify_doc_ids``.
    """
    return re.sub(r"[?#].*$", "", str(url)).rstrip("/").rsplit("/", 1)[-1]


def _to_utc_iso(series: pd.Series) -> pd.Series:
    """Parse a ShareChat time column to tz-aware UTC timestamps (unparseable -> ``NaT``).

    Handles the two shapes the five platforms use: ChatGPT's ``ts:<unix seconds>`` and everything
    else's date strings -- ISO-8601 with a ``Z`` (Grok), a naive ``YYYY-MM-DD HH:MM:SS`` (ChatGPT's
    conversation-level column), and human-readable ``January 25, 2025 01:58 AM`` / ``August 12,
    2025`` (Gemini, Perplexity). Naive values are read as UTC, matching the other two adapters.
    ``ts:`` seconds are rounded to microseconds, since a float's excess precision would otherwise
    show up as spurious nanosecond digits in the ISO string. Anything unparseable, or outside
    :data:`PLAUSIBLE_TIME_WINDOW`, becomes ``NaT``.

    The window is applied to the *unix seconds* before a timestamp is built, not to the result:
    ChatGPT's worst corrupt value is year 2282, which is past the range a nanosecond timestamp can
    represent at all, so constructing it first and filtering after raises ``OutOfBoundsDatetime``
    instead of yielding the ``NaT`` we want.
    """
    low, high = (pd.Timestamp(bound, tz="UTC") for bound in PLAUSIBLE_TIME_WINDOW)
    text = series.astype("string").str.strip()
    # A plain numpy bool array, not the nullable `boolean` that `.str.match` returns on a `string`
    # column -- pandas rejects the latter as an indexer.
    is_unix = text.str.match(_UNIX_TS_RE).fillna(False).to_numpy(dtype=bool)

    seconds = pd.to_numeric(text.str.removeprefix("ts:"), errors="coerce").where(is_unix)
    seconds = seconds.where(seconds.between(low.timestamp(), high.timestamp()))
    from_unix = pd.to_datetime((seconds * 1e6).round(), unit="us", utc=True)

    from_text = pd.to_datetime(text.where(~is_unix), format="mixed", utc=True, errors="coerce")
    return from_unix.fillna(from_text).where(lambda t: t.between(low, high))


def _iso_or_none(value) -> str | None:
    """An ISO-8601 UTC string for a timestamp, or ``None`` for ``NaT``/missing."""
    return None if value is None or pd.isna(value) else pd.Timestamp(value).isoformat()


def _conversation_times(user_rows: pd.DataFrame, platform: str) -> pd.DataFrame:
    """Per-conversation ``started_at`` / ``ended_at`` (ISO-8601 UTC strings, or ``None``).

    Applies that platform's rule from :data:`PLATFORM_TIME_COLUMNS`: a per-message column is
    reduced to its min and max over the conversation's user messages, a conversation-level column
    serves as both ends, and a platform with no time column yields ``None`` for every document.
    Where a per-message column is present but empty for a whole conversation, the platform's
    conversation-level fallback is used instead.
    """
    urls = user_rows["url"].drop_duplicates()
    granularity, column, fallback = PLATFORM_TIME_COLUMNS[platform]
    if granularity is None:
        return pd.DataFrame({"url": urls, "started_at": None, "ended_at": None})

    parsed = _to_utc_iso(user_rows[column])
    if granularity == "message":
        spans = parsed.groupby(user_rows["url"]).agg(["min", "max"])
    else:  # one value describes the whole conversation -> it is both ends
        first = parsed.groupby(user_rows["url"]).first()
        spans = pd.DataFrame({"min": first, "max": first})

    if fallback is not None:
        backup = _to_utc_iso(user_rows[fallback]).groupby(user_rows["url"]).min()
        spans["min"] = spans["min"].fillna(backup)
        spans["max"] = spans["max"].fillna(backup)

    spans = spans.reindex(urls)
    return pd.DataFrame({
        "url": urls.to_numpy(),
        "started_at": [_iso_or_none(v) for v in spans["min"]],
        "ended_at": [_iso_or_none(v) for v in spans["max"]],
    })


def _dominant_model(models: pd.Series) -> str | None:
    """The most frequent real model id in a conversation's rows, or ``None`` if it has none.

    Role markers and placeholders (:data:`NON_MODEL_VALUES`) are discarded first, so a Grok
    conversation whose every row says ``human`` / ``ASSISTANT`` yields ``None`` rather than a role.
    """
    real = [m for m in models
            if isinstance(m, str) and m.strip().lower() not in NON_MODEL_VALUES]
    return pd.Series(real).mode().iat[0] if real else None


def _read_platform(path: Path, platform: str) -> tuple[pd.DataFrame, pd.Series]:
    """Stream one platform's CSV, returning ``(user rows, per-conversation model)``.

    The CSV is read in :data:`READ_CHUNK_ROWS` chunks and each chunk is reduced immediately: the
    ``model`` column is voted on across **all** rows (Grok records the real model only on its
    assistant rows) while everything else is filtered down to ``role == "user"``. Only those two
    reductions are retained, so peak memory tracks the user prompts rather than the file -- which
    matters because the assistant replies are the bulk of it.
    """
    _, time_column, time_fallback = PLATFORM_TIME_COLUMNS[platform]
    header = pd.read_csv(path, nrows=0).columns
    wanted = ["url", "role", "message_index", "plain_text", "detected_language_final",
              time_column, time_fallback, "model"]
    columns = [c for c in dict.fromkeys(wanted) if c and c in header]
    missing = [c for c in ("url", "role", "message_index", "plain_text") if c not in columns]
    if missing:
        raise SystemExit(f"ShareChat {platform} CSV is missing required column(s) {missing}: {path}")

    user_chunks, model_chunks = [], []
    reader = pd.read_csv(path, usecols=columns, chunksize=READ_CHUNK_ROWS)
    for chunk in tqdm(reader, desc=f"  ShareChat {platform}", unit="chunk"):
        if "model" in chunk.columns:
            model_chunks.append(chunk[["url", "model"]].dropna())
        user_chunks.append(chunk[chunk["role"] == "user"])

    users = pd.concat(user_chunks, ignore_index=True)
    users = users[users["plain_text"].notna()].reset_index(drop=True)
    if model_chunks:
        pairs = pd.concat(model_chunks, ignore_index=True)
        models = pairs.groupby("url")["model"].agg(_dominant_model)
    else:
        models = pd.Series(dtype="object")
    return users, models


def load_sharechat_documents(raw_path: str | Path) -> pd.DataFrame:
    """Load ShareChat as a normalized (uncleaned) document frame, all five platforms pooled.

    ``raw_path`` is the directory holding the five per-platform CSVs (:data:`PLATFORM_FILES`);
    all five are required.

    Columns: ``doc_id, source, identity, turns_raw, languages, model, model_owner, agent,
    started_at, ended_at, repo_id, user_id`` -- the same layout the WildChat adapter returns, so
    the shared build pipeline treats it identically. Three of them are constant for this source:
    ``identity`` is ``None`` (ShareChat publishes no user id -- see the module docstring), and
    ``agent`` / ``repo_id`` / ``user_id`` are ``None`` because it is browser chat, not agent
    sessions.

    One document is one shared conversation, keyed by its share ``url``; its ``turns_raw`` are that
    conversation's user messages in ``message_index`` order, with the upstream Presidio markers
    removed (:func:`strip_upstream_redactions`). ``doc_id`` is ``sh-<platform>-<share slug>``.
    ``model`` is the conversation's own model id where the platform exports one, else the platform
    name; ``model_owner`` always comes from the platform (:data:`PLATFORM_OWNER`).
    """
    raw_path = Path(raw_path)
    records = []
    for platform, filename in PLATFORM_FILES.items():
        path = raw_path / filename
        if not path.exists():
            raise SystemExit(
                f"ShareChat {platform} CSV not found: {path}\n"
                f"Expected all of {sorted(PLATFORM_FILES.values())} in {raw_path}."
            )
        users, models = _read_platform(path, platform)
        users = users.sort_values(["url", "message_index"], kind="stable")
        users["_lang"] = users["detected_language_final"].map(normalize_language)

        conversations = users.groupby("url", sort=False).agg(
            turns_raw=("plain_text", lambda s: [strip_upstream_redactions(str(t)) for t in s]),
            languages=("_lang", ordered_languages),
        ).reset_index()
        conversations = conversations.merge(_conversation_times(users, platform), on="url", how="left")

        conversations["model"] = conversations["url"].map(models).fillna(platform)
        conversations["model_owner"] = PLATFORM_OWNER[platform]
        conversations["doc_id"] = "sh-" + platform + "-" + conversations["url"].map(_share_slug)
        print(f"  ShareChat {platform}: {len(conversations):,} conversations, "
              f"{int(users.shape[0]):,} user turns")
        records.append(conversations)

    frame = pd.concat(records, ignore_index=True)
    frame["source"] = "sharechat"
    # ShareChat publishes no user id, so there is no identity to pseudonymize; build_dataset skips
    # its author stages for this source and the split's author_id column is null throughout.
    frame["identity"] = None
    for column in ("agent", "repo_id", "user_id"):
        frame[column] = None
    return frame[[
        "doc_id", "source", "identity", "turns_raw", "languages", "model", "model_owner",
        "agent", "started_at", "ended_at", "repo_id", "user_id",
    ]]
