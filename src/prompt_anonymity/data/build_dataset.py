"""Build the unified, public-ready prompt dataset from the raw WildChat, SWE-chat and ShareChat sources.

Each source is built **independently** through the same shared stages and written to its own
parquet file, so they are separate HuggingFace splits (``wildchat``, ``swe_chat``, ``sharechat``)
of one dataset rather than a single combined file. Building per source is exactly equivalent to
the old combined run: the only cross-source coupling was boilerplate dedup, and no affix is shared
*solely* across sources, so the split changes the **layout, not the data**.

Pipeline per source (identical treatment for all three, except the source-specific steps noted
below; see the per-module docstrings for detail):

    load one source (raw turns; WildChat: drop programmatic clients; SWE-chat: keep only
      human-authored turns; ShareChat: group per-message rows into conversations and strip its
      upstream Presidio markers)  ->  clean turns in parallel
      ->  [SWE-chat only] collapse consecutive-duplicate turns  ->  drop empty
      ->  pseudonymize author_id  ->  unified dedup  ->  [WildChat only] drop relay authors
      ->  min-docs-per-author filter
      ->  resolve languages (primary + secondary)
      ->  finalize (sort, unique doc_ids, select columns)  ->  one parquet per source

**ShareChat has no author, and every author-keyed stage above is skipped for it** (see
:data:`UNIDENTIFIED_SOURCES`): its ``author_id`` column is null throughout, pseudonymization and
the ``min_docs`` floor do not run, and dedup goes through :func:`run_dedup_unidentified` instead.
That makes it useless as a labeled side of a linkage experiment and exactly right as a pool of
**out-of-set documents** for the open-set attacks -- traffic whose author is, by construction, on
nobody's known side.

The SWE-chat consecutive-duplicate-turn step removes a user turn that exactly repeats the one
before it when the repeat looks like an artifact or boilerplate (same source turn_id, no agent
response in between, very long, or a value that recurs often); see
:func:`dedup_consecutive_turns`. WildChat needs no such step -- its turns strictly alternate
with the model's replies, so it has no consecutive-duplicate-turn artifacts.

The language stage is likewise per-source (see ``prompt_anonymity/data/language_detection.py``): SWE-chat is fully
re-detected because its upstream labels are unreliable, while WildChat keeps its trusted upstream
primary and has only a *secondary* language detected (it ships one language per conversation). That
WildChat secondary pass used to be a separate follow-up script run against the built parquet; it is
a stage of this build now, so one command produces the finished files.

One row = one document (a WildChat conversation or a SWE-chat session), with the user turns
stored as a **list** (``turns``, cleaned/scrubbed) rather than a delimiter-joined string, so
turn boundaries are unambiguous. There is **no length-based filtering** -- documents are kept
regardless of how short they are; the only removals are "has no user turns", empty documents,
exact/boilerplate duplicates, and authors with fewer than ``--min-docs`` documents (a *count*
filter, not a length filter).

Design choices worth knowing:

* **WildChat inclusion.** Every browser-sent conversation on the studied models
  (:data:`prompt_anonymity.data.sources_wildchat.WILDCHAT_MODELS`) is kept. The models are pooled with no roles
  attached -- none is the "known" side and none the "unknown" side -- and the old "identity used
  >=2 distinct models" rule is dropped, so single-model users are retained. Conversations posted
  by HTTP clients rather than typed into a browser are dropped at load time (``--keep-
  programmatic-clients`` disables this); the only per-author floor is ``--min-docs`` (default 2),
  since attribution/clustering needs >=2 documents per author. Set ``--min-docs 1`` to include
  single-document authors.
* **Relay authors are dropped.** A relay posts many different people's messages under one request
  fingerprint, making that ``author_id`` a mixture rather than an author. Two kinds are removed:
  those whose client is programmatic (caught by the load-time filter above, including one whose
  hand-built ``User-Agent:...`` header gave it away) and those that reach the API through a
  browser but inject chat scaffolding into every prompt (:func:`drop_relay_authors`;
  ``--keep-relay-authors`` disables it). This is the one content-based *author* filter in the
  build -- no document is ever dropped for its content alone.
* **Consistency.** Both sources are cleaned by the same scrubber and deduplicated by the same
  rule, so ``source`` does not leak through surface tokens.
* **No split assigned.** The dataset no longer ships a ``split_role`` column; the known/unknown
  linkage split is being redesigned (e.g. SWE-chat by day rather than by last session) and will
  be assigned downstream, not here.

Cleaning is CPU-bound (regex over long pasted texts), so it is parallelized across processes
with ``--workers``, defaulting to the CPUs this job may actually use (see
:mod:`prompt_anonymity.resources` -- on a shared cluster that is well below the machine's
core count).

**Where the data comes from and goes.** Neither location is hard-coded: each source's raw
upstream data is resolved by :func:`prompt_anonymity.data.config.raw_path` -- a local copy if one
is configured (``$PROMPT_ANONYMITY_WILDCHAT_RAW`` / ``$PROMPT_ANONYMITY_SWE_CHAT_RAW``, or a
``path`` in ``datasets.toml``), otherwise downloaded from HuggingFace at the pinned revision --
and is resolved *only for the sources being built*, so a SWE-chat-only run never touches
WildChat. The built parquets go to ``--out-dir``, by default the project's ``data/dist``
(:func:`prompt_anonymity.data.config.dist_dir`).

Run ``python -m prompt_anonymity.data.build_dataset --help`` for options.
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd
import pyarrow as pa
from tqdm import tqdm

from prompt_anonymity.resources import available_cpus

from .config import dist_dir, raw_path
from .dedup import MAX_PER_AFFIX, MIN_AFFIX_LEN, deduplicate
from .identity import hash_author_id, load_ua_device_map
from .language_detection import (
    WILDCHAT_DETECT_CAP,
    WILDCHAT_PRIMARY_MIN_PRESENCE,
    WILDCHAT_SECONDARY_MIN_CONFIDENCE,
    add_secondary_languages,
    resolve_document_languages,
)
from .sources_sharechat import load_sharechat_documents
from .sources_swe_chat import is_scaffolding_turn, load_swe_chat_documents
from .sources_wildchat import WILDCHAT_MODELS, load_wildchat_documents
from .text_cleaning import clean_prompt

FINAL_COLUMNS = [
    "doc_id", "source", "author_id",
    "turns", "num_turns",
    "language_primary", "language_secondary", "model", "model_owner", "agent", "started_at", "ended_at",
]

# Every final column except the ``turns`` list and the ``num_turns`` count is a nullable string, and
# some come out all-``None`` for a given source (WildChat has no ``agent``; most documents have no
# ``language_secondary``). An all-``None`` column would serialize as the Arrow ``null`` type, and a
# ``null`` column cannot cast to its sibling split's ``string`` -- the one thing that breaks loading
# the two parquets as a single HuggingFace dataset. So these are written with an explicit Arrow
# ``string`` type, identical across both splits: it is the lighter, HF-native flavor (``string`` and
# ``large_string`` interoperate across splits regardless). See :func:`_with_arrow_string_columns`.
STRING_COLUMNS = tuple(c for c in FINAL_COLUMNS if c not in ("turns", "num_turns"))

#: The sources, named the way every CLI, config key, split, parquet and results directory in the
#: project names them. HuggingFace split names must match ``^\w+$`` (no hyphens), which is why
#: this corpus is ``swe_chat`` and not ``swe-chat``: one spelling, so a ``--source`` value, a
#: ``[sources.*]`` table, ``swe_chat.parquet`` and ``swe_chat_base_..._nearest_neighbor/`` all say
#: the same word. There is no source -> split mapping any more; the source name *is* the split.
#:
#: **The ``source`` column is a different string and deliberately still ``swe-chat``.** That value
#: is data, not a name: :func:`~prompt_anonymity.data.identity.hash_author_id` hashes it into
#: every ``author_id`` (which is also prefixed with it, ``swe-chat-<16 hex>``), so respelling it
#: would silently change every id in the published dataset, in every feature parquet, and in
#: every results CSV already computed. The adapters set it themselves
#: (:mod:`~prompt_anonymity.data.sources_swe_chat`), so it does not follow this constant.
#: ``sharechat`` was added after that lesson and spells its column the same as its name.
SOURCES = ("wildchat", "swe_chat", "sharechat")

#: Sources that publish no author identity, so every author-keyed stage is skipped for them and
#: their ``author_id`` column is null throughout: no :func:`pseudonymize`, no
#: :func:`filter_min_docs`, no :func:`drop_relay_authors`, and dedup runs the unidentified variant
#: (:func:`run_dedup_unidentified`).
#:
#: ShareChat is the only one. It is a corpus of *shared conversation links*, and a share link
#: identifies the conversation, not the person -- two links may or may not be the same author and
#: the upstream data cannot say. Inventing one author per link would have been convenient (every
#: groupby keeps working) but it asserts something unverified, so the column stays null and the
#: split is what it honestly is: a pool of documents with **no known author**, which is exactly what
#: an out-of-set / distractor population for an open-set attack needs to be.
UNIDENTIFIED_SOURCES = frozenset({"sharechat"})

# Rows per parquet row group. A row group is the smallest unit a reader can skip to, so writing
# one giant group forces any consumer to materialize the whole file: pyarrow's default (1024*1024
# rows) put all of WildChat in a single 1.09 GiB group, which the HuggingFace dataset viewer
# refuses to scan (its per-read limit is 300 MB). WildChat's conversations average ~6 KB, so 5000
# rows is a ~30 MB group -- comfortably inside the viewer's budget with room for the long tail,
# and still large enough that per-group metadata and compression ratios stay negligible.
PARQUET_ROW_GROUP_SIZE = 5000

# Consecutive-duplicate turn dedup (SWE-chat only). A user turn that exactly repeats the one
# immediately before it is dropped when it is long (>= this many chars) or recurs this often as
# a consecutive duplicate across the corpus (in addition to the always-on turn_id / no-agent
# rules); see :func:`dedup_consecutive_turns`.
CONSECUTIVE_DUP_MAX_LEN = 1000
CONSECUTIVE_DUP_MIN_OCC = 5


# --- parallel cleaning ------------------------------------------------------

def _clean_command_turn(text: str, repo_id, user_id, mask_ids) -> str:
    """Clean a reduced slash-command turn (``/name args``): keep the ``/command`` token verbatim
    and scrub identifiers only in the arguments (the command name is not a path/identifier and
    would otherwise be mangled to ``<PATH>``)."""
    parts = text.split(None, 1)
    if len(parts) == 1:
        return parts[0]
    return f"{parts[0]} {clean_prompt(parts[1], repo_id=repo_id, user_id=user_id, mask_ids=mask_ids)}".strip()


def _clean_document(args):
    """Worker: clean one document's raw turns into a list of identifier-scrubbed turns.

    Scrubs identifiers (and, for SWE-chat, the session's repo/user tokens plus opaque
    ids / ``user@host`` logins via ``mask_ids``). ``is_command`` marks reduced slash-command
    invocations, whose ``/command`` token is preserved; it is ``None`` for a source that has no
    slash commands (WildChat), which skips building a per-turn flag list for every document.
    Top-level so it is picklable by ``ProcessPoolExecutor``.
    """
    turns_raw, is_command, repo_id, user_id, mask_ids = args
    if is_command is None:
        return [clean_prompt(t, repo_id=repo_id, user_id=user_id, mask_ids=mask_ids) for t in turns_raw]
    return [
        _clean_command_turn(t, repo_id, user_id, mask_ids) if cmd
        else clean_prompt(t, repo_id=repo_id, user_id=user_id, mask_ids=mask_ids)
        for t, cmd in zip(turns_raw, is_command)
    ]


# Columns consumed by the cleaning stage and dead afterwards. Dropped as soon as ``turns`` exists
# so the corpus text is not held twice (raw + cleaned) through the stages that follow -- at
# WildChat's scale the raw copy is many GiB, and the next stage (dedup) allocates a third copy of
# its own. :mod:`prompt_anonymity.data.validate_dataset` separately asserts none of these reach the output.
_CLEANING_INPUT_COLUMNS = ("turns_raw", "is_command", "repo_id", "user_id")


def clean_documents(frame: pd.DataFrame, workers: int, *, mask_ids: bool = False) -> pd.DataFrame:
    """Replace the raw ``turns_raw`` column with the cleaned ``turns`` list, cleaning in parallel.

    ``mask_ids`` enables the extra opaque-id / ``user@host`` scrubs (SWE-chat only for now).
    The inputs the cleaner consumed (:data:`_CLEANING_INPUT_COLUMNS`) are dropped from the
    returned frame -- nothing downstream reads them, and keeping them doubles the memory the
    corpus text occupies for the rest of the build.
    """
    # A source with no slash commands passes None rather than a per-turn list of False.
    cmd_flags = frame["is_command"] if "is_command" in frame.columns else [None] * len(frame)
    args = list(zip(frame["turns_raw"], cmd_flags, frame["repo_id"], frame["user_id"],
                    [mask_ids] * len(frame)))
    desc = f"cleaning ({workers} procs)" if workers and workers > 1 else "cleaning"
    if workers and workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(tqdm(pool.map(_clean_document, args, chunksize=64), total=len(args), desc=desc))
    else:
        results = [_clean_document(a) for a in tqdm(args, desc=desc)]
    del args  # releases this stage's references to the raw turns before the frame does
    frame = frame.drop(columns=[c for c in _CLEANING_INPUT_COLUMNS if c in frame.columns])
    frame["turns"] = results
    return frame


# --- split ------------------------------------------------------------------

def _uniquify_doc_ids(doc_id: pd.Series) -> pd.Series:
    """Make doc_ids globally unique by suffixing collisions (``-1``, ``-2``, ...).

    Collisions occur when different authors send byte-identical content and thus share a
    WildChat ``conversation_hash``; those are legitimately distinct documents (different
    authors), so both rows are kept and the id is disambiguated deterministically -- the
    first occurrence in sort order keeps the bare id.
    """
    rank = doc_id.groupby(doc_id).cumcount()
    return doc_id.where(rank == 0, doc_id + "-" + rank.astype(str))


# --- shared pipeline stages -------------------------------------------------
# Every source runs through these same stages; only the loader differs. Keeping each stage a
# small named function is what lets the per-source builder share the pipeline end to end.

def load_source(
    source: str, *, wildchat_raw: str | None, swe_raw: str | None, sharechat_raw: str | None,
    ua_map: dict | None, min_docs: int, drop_programmatic: bool, wildchat_max_batches: int | None,
) -> pd.DataFrame:
    """Load one source's raw (uncleaned) documents into the common column layout.

    ``wildchat_raw`` / ``swe_raw`` / ``sharechat_raw`` are optional overrides of where that
    source's raw data is; ``None`` (the default) lets
    :func:`~prompt_anonymity.data.config.raw_path` resolve it from the environment, the config
    file, or a HuggingFace download. Only the source being loaded is resolved, so building one
    source never fetches the others.

    ``drop_programmatic`` applies to WildChat only -- SWE-chat sessions have no user-agent (they
    are all agent-CLI traffic by construction, with their non-human turns removed per turn), and
    ShareChat's conversations were scraped from share *pages*, which carry no client metadata.
    """
    if source == "wildchat":
        if ua_map is None:
            ua_map = load_ua_device_map()
        return load_wildchat_documents(
            raw_path("wildchat", wildchat_raw), ua_map, models=WILDCHAT_MODELS, min_docs=min_docs,
            drop_programmatic=drop_programmatic, max_batches=wildchat_max_batches,
        )
    if source == "swe_chat":
        return load_swe_chat_documents(raw_path("swe_chat", swe_raw))
    if source == "sharechat":
        return load_sharechat_documents(raw_path("sharechat", sharechat_raw))
    raise ValueError(f"unknown source: {source!r}")


def drop_empty_turns(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop individual turns that are empty after cleaning (ShareChat only).

    Removing ShareChat's upstream Presidio markers
    (:func:`~prompt_anonymity.data.sources_sharechat.strip_upstream_redactions`) empties any turn
    that was *entirely* redacted -- a message that was nothing but a name or a phone number. Such
    a turn carries no text but still counts toward ``num_turns``, so it would report a length the
    document does not have. Measured on the raw ShareChat conversations that is 1.4% of turns,
    against 0.003% in WildChat and 0 in SWE-chat, which is why this runs for ShareChat alone:
    those two have no redaction stage to empty a turn, and applying it to them would rewrite
    already-published documents for a handful of rows.

    Documents are never dropped here; one left with no turns at all is removed immediately after
    by :func:`drop_empty_documents`.
    """
    frame = frame.copy()
    before = int(frame["turns"].map(len).sum())
    frame["turns"] = frame["turns"].map(lambda ts: [t for t in ts if t])
    removed = before - int(frame["turns"].map(len).sum())
    print(f"  empty turns removed: {removed:,} of {before:,}")
    return frame


def drop_empty_documents(frame: pd.DataFrame) -> pd.DataFrame:
    """Drop documents whose every turn is empty after cleaning (whitespace-only originals).

    A validity filter, not a length filter -- nothing is dropped merely for being short.
    """
    return frame[frame["turns"].map(lambda ts: any(t for t in ts))].reset_index(drop=True)


def _consecutive_dup_counts(turn_lists) -> tuple[Counter, Counter]:
    """Corpus-wide ``(occurrences, n_docs)`` per turn value for consecutive duplicates.

    For each maximal run of ``L`` identical back-to-back turns in a document, the value gains
    ``L - 1`` *occurrences* (the copies beyond the first); ``n_docs`` counts the distinct
    documents in which the value is a consecutive duplicate at all. These are the ``occurrences``
    and ``n_docs`` columns of the consecutive-dup audit, and together they drive the "recurs
    often *within a single document*" rule in :func:`dedup_consecutive_turns`.
    """
    occ: Counter = Counter()
    ndocs: Counter = Counter()
    for turns in turn_lists:
        seen: set = set()
        i, n = 0, len(turns)
        while i < n:
            j = i
            while j + 1 < n and turns[j + 1] == turns[i]:
                j += 1
            if j > i and turns[i]:
                occ[turns[i]] += j - i
                seen.add(turns[i])
            i = j + 1
        for value in seen:
            ndocs[value] += 1
    return occ, ndocs


def dedup_consecutive_turns(
    frame: pd.DataFrame, *,
    max_len: int = CONSECUTIVE_DUP_MAX_LEN,
    min_occ: int = CONSECUTIVE_DUP_MIN_OCC,
) -> pd.DataFrame:
    """Collapse *consecutive*-duplicate user turns within a document (SWE-chat only).

    A turn that exactly repeats the turn immediately before it is dropped -- keeping the first --
    when ANY of these hold, each a signal the repeat is an artifact or low-value boilerplate
    rather than a distinct authored message:

    1. it shares the previous turn's source ``turn_id`` (a literal duplicate row in the upstream
       log -- the same message recorded twice);
    2. no agent turn ran between the two (``agent_before`` is false) -- a resend with no model
       response in between, not a reply-then-reask; an ``assistant_response`` that is an API error
       (expired auth, 500/529, rate-limit, ...) does *not* count as a response, so failed-request
       retries fall under this rule too;
    3. the message is ``>= max_len`` characters (long pasted templates / skill scaffolding);
    4. the value recurs as a consecutive duplicate ``>= min_occ`` times **within a single
       document** (``occurrences >= min_occ`` and ``n_docs == 1``) -- one session spamming a
       command in a loop (e.g. a monitoring check), as opposed to a short reply like ``yes`` that
       recurs across many authors' sessions and is left alone;
    5. it is a framework-injected scaffolding message (:func:`is_scaffolding_turn` -- slash
       commands, tool I/O markers, skill attachments, ...), which is never authored prose.

    Only *consecutive* duplicates are considered: a repeat separated by any different turn is a
    non-consecutive duplicate and is left untouched. Requires the per-turn ``turn_ids`` and
    ``agent_before`` lists from the SWE-chat loader (aligned with ``turns``). Document rows are
    never dropped here (the first turn of every run is always kept); only turns are removed.
    """
    occ, ndocs = _consecutive_dup_counts(frame["turns"])
    new_turns = []
    for turns, turn_ids, agent_before in zip(frame["turns"], frame["turn_ids"], frame["agent_before"]):
        kept: list[str] = []
        for k, turn in enumerate(turns):
            if k >= 1 and turn and turn == turns[k - 1] and (
                turn_ids[k] == turn_ids[k - 1]              # 1. literal duplicate source row
                or not agent_before[k]                      # 2. no agent turn responded in between
                or len(turn) >= max_len                     # 3. long pasted message
                or (occ[turn] >= min_occ and ndocs[turn] == 1)  # 4. spammed within one document
                or is_scaffolding_turn(turn)                # 5. framework scaffolding, not prose
            ):
                continue                                    # drop this repeat, keep the first
            kept.append(turn)
        new_turns.append(kept)
    frame = frame.copy()
    frame["turns"] = new_turns
    return frame


def pseudonymize(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the opaque, source-prefixed ``author_id`` hashed from ``(source, identity)``."""
    frame = frame.copy()
    frame["author_id"] = [hash_author_id(s, i) for s, i in zip(frame["source"], frame["identity"])]
    return frame


def run_dedup(frame: pd.DataFrame, *, affix_len: int, max_per_affix: int,
              affix_dedup: bool = True) -> pd.DataFrame:
    """Apply the unified dedup, keyed on an internal join of the cleaned turns (not stored).

    Authors are source-prefixed and every author lives in a single source, so running this per
    source is identical to running it on the combined corpus -- verified at a 50-char affix: no
    affix was shared solely across sources, and a longer window can only reduce collisions (a
    shared 100-char prefix implies a shared 50-char one).

    ``affix_dedup=False`` runs exact-duplicate removal only (no whole-document drop for a shared
    prefix/suffix); SWE-chat uses this, WildChat keeps the default. See :func:`prompt_anonymity.data.dedup.deduplicate`.

    The joined text is a full second copy of the corpus, so it is dropped again before returning
    rather than riding along (unused) through every later stage.
    """
    frame = frame.copy()
    frame["_dedup_text"] = frame["turns"].map("\n".join)
    deduped = deduplicate(
        frame, identity_col="author_id", text_col="_dedup_text",
        order_cols=("author_id", "started_at", "doc_id"),
        affix_len=affix_len, max_per_affix=max_per_affix, affix_dedup=affix_dedup,
    )
    return deduped.drop(columns="_dedup_text")


def run_dedup_unidentified(frame: pd.DataFrame, *, affix_len: int) -> pd.DataFrame:
    """Dedup for a source with no author labels (:data:`UNIDENTIFIED_SOURCES`).

    :func:`run_dedup` keys every rule on the author, which ShareChat does not have, so the two
    rules that still mean something are run explicitly and each is given the identity column that
    makes it do its one job:

    1. **Global exact duplicates.** With no author, "the same person said this twice" is not
       expressible; what is, is "this exact text appears twice in the corpus". Run under a single
       constant identity so :func:`~prompt_anonymity.data.dedup.deduplicate`'s step 1 collapses
       every repeat to its earliest occurrence, corpus-wide.
    2. **Shared-affix boilerplate.** Run again with ``doc_id`` as the identity -- every document is
       then its own identity, so the cross-identity rule (step 2) drops any document whose leading
       or trailing ``affix_len``-char slice is shared with *another document*. The per-identity
       steps (1 and 3) are no-ops under unique ids, which is why the first pass is needed at all.

    This is deliberately stricter than WildChat's, which keeps up to ``max_per_affix`` documents
    per (author, affix) on the grounds that a habitual opening is genuine style. Without authors
    that exemption cannot be granted -- there is no way to tell one person's repeated template from
    a template many people pasted -- and for a distractor pool the strict reading is the safe one:
    templated text is exactly what should not be in it.
    """
    frame = frame.copy()
    frame["_dedup_text"] = frame["turns"].map("\n".join)
    frame["_one"] = ""  # a single constant identity, so step 1 runs corpus-wide

    exact = deduplicate(
        frame, identity_col="_one", text_col="_dedup_text",
        order_cols=("started_at", "doc_id"), affix_dedup=False,
    )
    n_exact = len(frame) - len(exact)

    deduped = deduplicate(
        exact, identity_col="doc_id", text_col="_dedup_text",
        order_cols=("started_at", "doc_id"), affix_len=affix_len, max_per_affix=1,
    )
    print(f"  unidentified dedup: {n_exact:,} exact duplicates, "
          f"{len(exact) - len(deduped):,} shared-affix boilerplate documents removed")
    return deduped.drop(columns=["_dedup_text", "_one"])


def filter_min_docs(frame: pd.DataFrame, min_docs: int) -> pd.DataFrame:
    """Keep authors with at least ``min_docs`` documents (a *count* filter, not a length filter).

    Applied after dedup, so duplicates cannot manufacture a qualifying pair.
    """
    sizes = frame.groupby("author_id")["doc_id"].transform("size")
    return frame[sizes >= min_docs].reset_index(drop=True)


# Chat scaffolding that only a *program* puts in a prompt. Two forms, both rare enough (0.14% of
# WildChat documents) to be near-unambiguous:
#   (a) the text ends on a dangling speaker label ("... Assistant:") -- the client is asking the
#       model to complete the next turn, a completion-style API call. Nobody types this into a
#       chat box and hits send.
#   (b) the text opens with a speaker or system label ("System: You are an expert ...") -- a
#       front-end's injected persona, not the person's own words.
# Together these catch relays that replay the whole transcript as one prompt, which is why such
# authors are also ~always 100% single-turn: the client keeps history in the prompt instead of
# using the multi-turn API.
_SCAFFOLD_TRAILING_LABEL = re.compile(r"(?:^|\n|\s)(?:Assistant|AI|ChatGPT|Bot)\s*:\s*$", re.IGNORECASE)
_SCAFFOLD_LEADING_LABEL = re.compile(r"^\s*(?:System|Assistant|AI|ChatGPT|Bot)\s*:\s", re.IGNORECASE)

# An author must show scaffolding at least this many times before the whole author is dropped. One
# occurrence is not enough: a person can paste a transcript once.
MIN_SCAFFOLD_DOCS = 2


def _has_chat_scaffolding(turns) -> bool:
    """True if a document carries a programmatic chat-scaffolding marker (see above)."""
    text = "\n".join(turns)
    return bool(_SCAFFOLD_TRAILING_LABEL.search(text) or _SCAFFOLD_LEADING_LABEL.match(text))


def drop_relay_authors(frame: pd.DataFrame, *, min_scaffold_docs: int = MIN_SCAFFOLD_DOCS) -> pd.DataFrame:
    """Drop authors that are relays -- one ``author_id`` covering many different people.

    A relay (a Discord bridge, a hosted Space front-end, an API app) posts everyone's messages
    under one request fingerprint, so its ``author_id`` is not an author at all. That is fatal for
    attribution: the label is a mixture, and no amount of style modelling can be right about it.

    An author is dropped when **both** hold:

    * at least ``min_scaffold_docs`` of its documents carry chat scaffolding
      (:func:`_has_chat_scaffolding`) -- the client is injecting structure a human would not type; and
    * **every** one of its documents is a single-turn conversation -- the client never uses a
      follow-up, because it re-sends the whole history as a fresh prompt instead.

    The conjunction is what makes this safe, and it is deliberately *not* volume-dependent: it
    fires on an author with as few as two documents, where behavioural statistics (language
    spread, activity hours, turn counts) have no power at all. The corpus-wide rate of
    all-single-turn authors is 13%, so the second condition is real evidence rather than the base
    rate, and each condition alone is far too broad to use by itself.

    Requiring *both* also spares the genuine edge case: one person using a custom front-end that
    injects a system prompt. They trip the scaffolding test, but their conversations have
    follow-up turns, so they are kept -- their injected preamble is a text-cleaning problem, not
    grounds for deleting the person.
    """
    scaffold = frame["turns"].map(_has_chat_scaffolding)
    per_author = pd.DataFrame({
        "scaffold": scaffold.groupby(frame["author_id"]).sum(),
        "all_single": frame["num_turns"].eq(1).groupby(frame["author_id"]).all(),
    })
    relays = set(per_author.index[(per_author["scaffold"] >= min_scaffold_docs) & per_author["all_single"]])
    if relays:
        n_docs = int(frame["author_id"].isin(relays).sum())
        print(f"  relay authors dropped: {len(relays):,} ({n_docs:,} documents)")
    return frame[~frame["author_id"].isin(relays)].reset_index(drop=True)


def finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort deterministically, uniquify doc_ids, select the final columns.

    Documents are grouped by author and ordered in time within each; a source with no authors
    (:data:`UNIDENTIFIED_SOURCES`, whose ``author_id`` is null throughout) drops that key and is
    ordered by time alone. ``doc_id`` is the final tiebreak either way, so the order is total
    even where the timestamps are missing or tie.

    No known/unknown split is assigned -- the dataset ships without a ``split_role`` column
    while the split is being redesigned (see the module docstring).
    """
    frame = frame.copy()
    keys = ["source", "author_id", "started_at", "doc_id"]
    if frame["author_id"].isna().all():
        keys.remove("author_id")
    frame = frame.sort_values(keys).reset_index(drop=True)
    frame["doc_id"] = _uniquify_doc_ids(frame["doc_id"])
    return frame[FINAL_COLUMNS]


# --- per-source build -------------------------------------------------------

def build_source(
    source: str,
    *,
    wildchat_raw: str | None = None,
    swe_raw: str | None = None,
    sharechat_raw: str | None = None,
    ua_map: dict | None = None,
    min_docs: int = 2,
    drop_programmatic: bool = True,
    drop_relays: bool = True,
    affix_len: int = MIN_AFFIX_LEN,
    max_per_affix: int = MAX_PER_AFFIX,
    consec_dup_max_len: int = CONSECUTIVE_DUP_MAX_LEN,
    consec_dup_min_occ: int = CONSECUTIVE_DUP_MIN_OCC,
    secondary_threshold: float = WILDCHAT_SECONDARY_MIN_CONFIDENCE,
    secondary_primary_floor: float = WILDCHAT_PRIMARY_MIN_PRESENCE,
    secondary_cap: int = WILDCHAT_DETECT_CAP,
    workers: int | None = None,
    wildchat_max_batches: int | None = None,
) -> pd.DataFrame:
    """Run the full pipeline for **one** source and return its finished dataset frame.

    All sources share the stage functions above; only :func:`load_source` differs, plus three
    source-specific stages: the SWE-chat consecutive-duplicate-turn dedup (which needs that
    source's per-turn metadata), the WildChat secondary-language pass (the ``secondary_*`` knobs),
    and -- for a source in :data:`UNIDENTIFIED_SOURCES` -- the author-free dedup that stands in for
    every author-keyed stage. ``ua_map`` may be passed in to avoid reloading the user-agent map
    when building multiple sources. The per-stage row counts are attached to
    ``frame.attrs["counts"]``.
    """
    if workers is None:
        workers = available_cpus()
    unidentified = source in UNIDENTIFIED_SOURCES

    print(f"[{source}] loading ...")
    frame = load_source(
        source, wildchat_raw=wildchat_raw, swe_raw=swe_raw, sharechat_raw=sharechat_raw,
        ua_map=ua_map, min_docs=min_docs, drop_programmatic=drop_programmatic,
        wildchat_max_batches=wildchat_max_batches,
    )
    n_loaded = len(frame)

    print(f"[{source}] cleaning {n_loaded:,} documents on {workers} worker(s) ...")
    # SWE-chat additionally masks opaque ids (UUIDs / commit SHAs -> <ID>) and shell logins
    # (user@host -> <HOST>), which pervade agentic/terminal logs. WildChat leaves these off
    # for now, so its committed output is unaffected.
    frame = clean_documents(frame, workers, mask_ids=(source == "swe_chat"))

    consec_removed = 0
    if source == "swe_chat":
        turns_before = int(frame["turns"].map(len).sum())
        frame = dedup_consecutive_turns(frame, max_len=consec_dup_max_len, min_occ=consec_dup_min_occ)
        consec_removed = turns_before - int(frame["turns"].map(len).sum())
        print(f"[{source}] consecutive-dup turns removed: {consec_removed:,}")

    # ShareChat's redaction stripping can empty a whole turn; the other two sources have nothing
    # that does, so they keep their turn lists exactly as published. See drop_empty_turns.
    if source == "sharechat":
        frame = drop_empty_turns(frame)

    frame = drop_empty_documents(frame)
    n_after_empty = len(frame)
    frame["num_turns"] = frame["turns"].map(len)

    # ShareChat publishes no user id, so there is nothing to pseudonymize and author_id stays null;
    # its dedup, relay and min-docs stages are all skipped or replaced below.
    if unidentified:
        frame = frame.copy()
        frame["author_id"] = None
        frame = run_dedup_unidentified(frame, affix_len=affix_len)
    else:
        frame = pseudonymize(frame)
        # SWE-chat: do NOT drop a whole conversation just because it shares a templated prefix/suffix
        # -- injected non-human templates are stripped at the turn level (see sources_swe_chat), so
        # only exact duplicates are removed. WildChat keeps the full affix dedup (heavy cross-author
        # templating, not stripped per turn).
        affix_dedup = source != "swe_chat"
        frame = run_dedup(frame, affix_len=affix_len, max_per_affix=max_per_affix,
                          affix_dedup=affix_dedup)
    n_after_dedup = len(frame)

    # Relays put many people under one author_id. WildChat only: a SWE-chat author is a code-host
    # user account, which cannot be a relay, and its injected scaffolding is already removed per
    # turn. Runs before the min-docs floor so a dropped relay cannot prop up anything downstream.
    n_relay_docs = 0
    if source == "wildchat" and drop_relays:
        before = len(frame)
        frame = drop_relay_authors(frame)
        n_relay_docs = before - len(frame)

    # A count of documents *per author* has no meaning without authors, so an unidentified source
    # skips the floor entirely rather than being filtered against a single null group.
    if not unidentified:
        frame = filter_min_docs(frame, min_docs)
    n_after_mindocs = len(frame)

    # Resolve language_primary / language_secondary. SWE-chat's upstream labels are unreliable
    # (a third empty, some CJK mislabeled English), so we re-detect it with Lingua and fall back to
    # upstream only where the detector abstains; WildChat's and ShareChat's are trusted, so they
    # just split the existing list into the two columns -- a schema update, not a relabel of the
    # primary. ShareChat's labels were checked against Lingua on a 2,000-conversation sample across
    # all five platforms and agreed 98.7% of the time, which is why it gets WildChat's policy and
    # not SWE-chat's.
    frame, lang_stats = resolve_document_languages(frame, redetect=(source == "swe_chat"))
    print(f"[{source}] languages: {lang_stats['n_lingua']:,} by detector, "
          f"{lang_stats['n_fallback']:,} from upstream, {lang_stats['n_default']:,} defaulted to English")

    # WildChat labels one language per conversation, so the split above leaves it no secondary; it
    # is detected here, keeping the trusted primary (SWE-chat already got its secondary from the
    # re-detection). Single-process and Lingua-bound -- a few minutes on the full WildChat corpus.
    #
    # ShareChat labels *per message*, so the split above does hand it an upstream secondary -- and
    # this pass deliberately overwrites it. That upstream signal is per-message detection on short
    # messages and it shows: 5.8% of conversations have turns labelled with different languages,
    # and the lists include things like ('English', 'Latin', 'Malayalam') -- exactly the "a rare
    # confusable steals a short span" failure that `observed_language_detector`'s restricted
    # candidate set exists to prevent. The vetted pass puts it at 1.6%, in line with WildChat's
    # 0.9%, on plausible pairs.
    if source in ("wildchat", "sharechat"):
        frame, _ = add_secondary_languages(
            frame, threshold=secondary_threshold,
            primary_floor=secondary_primary_floor, cap=secondary_cap,
        )
    n_secondary = int(frame["language_secondary"].notna().sum())
    print(f"[{source}] secondary language: {n_secondary:,} of {len(frame):,} documents "
          f"({n_secondary / max(len(frame), 1) * 100:.2f}%)")

    result = finalize(frame)
    result.attrs["counts"] = {
        "loaded": n_loaded, "consecutive_dup_turns_removed": consec_removed,
        "after_drop_empty": n_after_empty, "after_dedup": n_after_dedup,
        "relay_docs_removed": n_relay_docs,
        "after_min_docs": n_after_mindocs, "final": len(result),
        "with_secondary_language": n_secondary,
    }
    print(f"[{source}] done: {len(result):,} documents")
    return result


def build_all(
    sources: tuple[str, ...] = SOURCES,
    *,
    wildchat_raw: str | None = None,
    swe_raw: str | None = None,
    sharechat_raw: str | None = None,
    min_docs: int = 2,
    drop_programmatic: bool = True,
    drop_relays: bool = True,
    affix_len: int = MIN_AFFIX_LEN,
    max_per_affix: int = MAX_PER_AFFIX,
    consec_dup_max_len: int = CONSECUTIVE_DUP_MAX_LEN,
    consec_dup_min_occ: int = CONSECUTIVE_DUP_MIN_OCC,
    secondary_threshold: float = WILDCHAT_SECONDARY_MIN_CONFIDENCE,
    secondary_primary_floor: float = WILDCHAT_PRIMARY_MIN_PRESENCE,
    secondary_cap: int = WILDCHAT_DETECT_CAP,
    workers: int | None = None,
    wildchat_max_batches: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build every requested source and return ``{source: finished frame}`` (insertion order).

    The user-agent map is loaded once and shared across sources that need it.
    """
    ua_map = load_ua_device_map() if "wildchat" in sources else None
    return {
        source: build_source(
            source, wildchat_raw=wildchat_raw, swe_raw=swe_raw, sharechat_raw=sharechat_raw,
            ua_map=ua_map,
            min_docs=min_docs, drop_programmatic=drop_programmatic, drop_relays=drop_relays,
            affix_len=affix_len, max_per_affix=max_per_affix,
            consec_dup_max_len=consec_dup_max_len, consec_dup_min_occ=consec_dup_min_occ,
            secondary_threshold=secondary_threshold,
            secondary_primary_floor=secondary_primary_floor, secondary_cap=secondary_cap,
            workers=workers, wildchat_max_batches=wildchat_max_batches,
        )
        for source in sources
    }


# --- outputs ----------------------------------------------------------------

def _with_arrow_string_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy whose :data:`STRING_COLUMNS` carry an explicit Arrow ``string`` type.

    Keeps the two splits' schemas identical and, crucially, keeps an all-``None`` column (e.g.
    WildChat's ``agent``) from serializing as the Arrow ``null`` type, which cannot cast to the
    sibling split's string when both are loaded as one HuggingFace dataset. ``frame.attrs`` (the
    pipeline-counts provenance) is carried over so ``to_parquet`` still records it.
    """
    out = frame.copy()
    out.attrs = dict(frame.attrs)
    for col in STRING_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(pd.ArrowDtype(pa.string()))
    return out


def write_outputs(frames: dict[str, pd.DataFrame], out_dir: str | Path) -> None:
    """Write one parquet per source (a HuggingFace split) and print a combined summary.

    ``frames`` maps source -> finished frame. Each is written to ``<split>.parquet`` (e.g.
    ``wildchat.parquet``, ``swe_chat.parquet``) with the string columns explicitly Arrow-typed
    (:func:`_with_arrow_string_columns`). A summary over the two combined is printed to the
    console; no sample or stats files are written.

    Files are written in :data:`PARQUET_ROW_GROUP_SIZE`-row groups with a page index, so a
    reader (notably the HuggingFace dataset viewer) can seek to a handful of rows instead of
    decoding the entire file.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Remove stale artifacts from prior layouts so the dist dir carries no orphans: the
    # pre-split combined parquet, and the sample/stats files this builder no longer emits.
    for stale in ("prompt_dataset.parquet", "sample_200.csv", "stats.json"):
        (out_dir / stale).unlink(missing_ok=True)

    for source, frame in frames.items():
        path = out_dir / f"{source}.parquet"
        _with_arrow_string_columns(frame).to_parquet(
            path,
            index=False,
            row_group_size=PARQUET_ROW_GROUP_SIZE,
            write_page_index=True,
        )
        print(f"[{source}] wrote {len(frame):,} documents -> {path}")

    combined = pd.concat(frames.values(), ignore_index=True)
    _print_report(build_stats(frames, combined))


def build_stats(frames: dict[str, pd.DataFrame], combined: pd.DataFrame) -> dict:
    """Aggregate stats for the console report, plus per-source pipeline counts."""
    stats = summarize(combined)
    stats["counts_pipeline"] = {src: f.attrs.get("counts", {}) for src, f in frames.items()}
    stats["splits"] = {src: int(len(f)) for src, f in frames.items()}
    return stats


def summarize(frame: pd.DataFrame) -> dict:
    """Compact statistics for the console report."""
    chars = frame["turns"].map(lambda ts: sum(len(t) for t in ts))
    if "language_primary" in frame.columns:
        primary_lang = frame["language_primary"].fillna("<none>")
    else:  # legacy `languages` list column (a source not yet rebuilt to the two-column schema)
        primary_lang = frame["languages"].map(lambda ls: ls[0] if len(ls) else "<none>")
    # Bilingual documents, as "<primary> + <secondary>" pairs (missing secondaries are None/NA).
    if {"language_primary", "language_secondary"} <= set(frame.columns):
        pairs = Counter(f"{p} + {s}" for p, s in zip(frame["language_primary"], frame["language_secondary"])
                        if isinstance(s, str))
    else:
        pairs = Counter()
    # `nunique`/`groupby` both drop nulls, so a source with no authors (ShareChat) contributes 0
    # authors and no docs-per-author mass rather than one giant null group -- which is the honest
    # reading, but means the author lines of this report describe only the identified sources.
    docs_per_author = frame.groupby("author_id")["doc_id"].size()

    def per_source(fn):
        return {src: fn(g) for src, g in frame.groupby("source")}

    return {
        "counts_pipeline": frame.attrs.get("counts", {}),
        "documents": int(len(frame)),
        "authors": int(frame["author_id"].nunique()),
        "documents_without_author": int(frame["author_id"].isna().sum()),
        "documents_per_source": frame["source"].value_counts().to_dict(),
        "authors_per_source": per_source(lambda g: int(g["author_id"].nunique())),
        "docs_per_author": {k: round(float(v), 2) for k, v in
                            docs_per_author.describe(percentiles=[.5, .9, .99]).items()},
        "num_turns": {k: round(float(v), 2) for k, v in
                      frame["num_turns"].describe(percentiles=[.5, .9, .99]).items()},
        "chars_per_doc": {k: round(float(v), 1) for k, v in
                          chars.describe(percentiles=[.5, .9, .99]).items()},
        "top_primary_languages": primary_lang.value_counts().head(12).to_dict(),
        "documents_with_secondary_language": int(sum(pairs.values())),
        "top_language_pairs": dict(pairs.most_common(12)),
    }


def _print_report(stats: dict) -> None:
    s = stats
    print("\n" + "=" * 64 + "\nUNIFIED PROMPT DATASET — SUMMARY\n" + "=" * 64)
    print("pipeline row counts (per source):", s["counts_pipeline"])
    print("splits (files):", s["splits"])
    print(f"documents: {s['documents']:,}   authors: {s['authors']:,}   "
          f"documents with no author: {s['documents_without_author']:,}")
    print("documents per source:", s["documents_per_source"])
    print("authors per source:", s["authors_per_source"])
    print("docs/author:", s["docs_per_author"])
    print("num_turns:", s["num_turns"])
    print("chars/doc:", s["chars_per_doc"])
    print("top primary languages:", s["top_primary_languages"])
    print(f"documents with a secondary language: {s['documents_with_secondary_language']:,}")
    print("top (primary + secondary) pairs:", s["top_language_pairs"])


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wildchat-raw", default=None,
                   help="directory of raw WildChat parquet shards (default: $PROMPT_ANONYMITY_"
                        "WILDCHAT_RAW, else the config file, else downloaded from HuggingFace)")
    p.add_argument("--swe-raw", default=None,
                   help="raw SWE-chat conversations.parquet (default: $PROMPT_ANONYMITY_"
                        "SWE_CHAT_RAW, else the config file, else downloaded from HuggingFace)")
    p.add_argument("--sharechat-raw", default=None,
                   help="directory of the five raw ShareChat per-platform CSVs (default: "
                        "$PROMPT_ANONYMITY_SHARECHAT_RAW, else the config file, else downloaded "
                        "from HuggingFace)")
    p.add_argument("--sources", nargs="+", default=list(SOURCES), choices=list(SOURCES))
    p.add_argument("--min-docs", type=int, default=2,
                   help="minimum documents per author (default 2; set 1 to keep single-doc authors)")
    p.add_argument("--keep-programmatic-clients", action="store_true",
                   help="WildChat: keep conversations posted by HTTP clients (gradio_client, httpx, "
                        "node, ...) instead of dropping them at load time. For measurement only -- "
                        "these conflate many humans under one author_id")
    p.add_argument("--keep-relay-authors", action="store_true",
                   help="WildChat: keep authors identified as relays (one author_id covering many "
                        "people: chat scaffolding in >=2 documents and no multi-turn conversation "
                        "at all). For measurement only")
    p.add_argument("--affix-len", type=int, default=MIN_AFFIX_LEN,
                   help=f"leading/trailing slice length for near-duplicate dedup (default {MIN_AFFIX_LEN})")
    p.add_argument("--max-per-affix", type=int, default=MAX_PER_AFFIX,
                   help=f"keep this many earliest documents per (author, affix) (default {MAX_PER_AFFIX})")
    p.add_argument("--consec-dup-max-len", type=int, default=CONSECUTIVE_DUP_MAX_LEN,
                   help="SWE-chat: collapse a consecutive duplicate turn this long or longer (chars)")
    p.add_argument("--consec-dup-min-occ", type=int, default=CONSECUTIVE_DUP_MIN_OCC,
                   help="SWE-chat: collapse a consecutive duplicate value recurring this often corpus-wide")
    p.add_argument("--secondary-threshold", type=float, default=WILDCHAT_SECONDARY_MIN_CONFIDENCE,
                   help="WildChat: minimum Lingua confidence for a secondary language (default 0.30)")
    p.add_argument("--secondary-primary-floor", type=float, default=WILDCHAT_PRIMARY_MIN_PRESENCE,
                   help="WildChat: minimum confidence the primary must clear for a secondary to count "
                        "(default 0.05; below it the document is an upstream mislabel, not bilingual)")
    p.add_argument("--secondary-cap", type=int, default=WILDCHAT_DETECT_CAP,
                   help="WildChat: detect the secondary on at most this many chars of prose (default 5000)")
    p.add_argument("--workers", type=int, default=available_cpus(),
                   help="parallel cleaning processes (default: the CPUs this job is allocated, "
                        "which on a shared cluster is fewer than the machine's cores)")
    p.add_argument("--wildchat-max-batches", type=int, default=None, help="testing: cap pass-2 batches")
    p.add_argument("--out-dir", default=None,
                   help="where the built parquets go (default: the project's data/dist)")
    args = p.parse_args()

    frames = build_all(
        tuple(args.sources),
        wildchat_raw=args.wildchat_raw, swe_raw=args.swe_raw, sharechat_raw=args.sharechat_raw,
        min_docs=args.min_docs, drop_programmatic=not args.keep_programmatic_clients,
        drop_relays=not args.keep_relay_authors,
        affix_len=args.affix_len, max_per_affix=args.max_per_affix,
        consec_dup_max_len=args.consec_dup_max_len, consec_dup_min_occ=args.consec_dup_min_occ,
        secondary_threshold=args.secondary_threshold,
        secondary_primary_floor=args.secondary_primary_floor, secondary_cap=args.secondary_cap,
        workers=args.workers, wildchat_max_batches=args.wildchat_max_batches,
    )
    write_outputs(frames, args.out_dir or dist_dir())


if __name__ == "__main__":
    main()
