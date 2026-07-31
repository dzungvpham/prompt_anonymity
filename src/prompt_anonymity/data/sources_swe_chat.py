"""SWE-chat source adapter: raw SALT-NLP/SWE-chat parquet -> normalized per-session documents.

Reads the raw parquet directly (no dependency on the ``swe-chat/`` scripts or their
intermediate CSVs). A document is one *session*, represented by its **human-authored**
``user_prompt`` turns in conversation order. The raw ``user_prompt`` stream also carries a lot the
person did not type -- coding-agent orchestration, tool-I/O echoes, CLI framework wrappers,
/compact summaries, and injected slash-command expansions -- so this adapter keeps only the
genuinely human turns (see :func:`filter_to_human_turns`). It returns the **raw** (uncleaned) turns
as a list (``turns_raw``); text cleaning (which for SWE-chat also scrubs the session's repo/user
tokens) happens in the parallelized stage in ``build_dataset.py``.

Identity is the ``user_id``, with the same repo-based recovery the linkage loader used for
id-less sessions (single-user repo -> that user; orphan repo -> the ``repo_id`` itself;
multi-user repo -> dropped).
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds

from .common import model_owner, normalize_language

# Turn types that represent the agent actually doing something in response to a user prompt
# (an LLM reply, its thinking, a tool call, or that call's result) -- as opposed to bookkeeping
# rows (``progress``, ``queue_operation``, ``file_snapshot``, ``system_event``, ...), which make
# up the bulk of a session log. Used to tell whether a real agent turn sits between two
# consecutive user prompts: a mere ``turn_number`` gap does not, because the gap is usually all
# bookkeeping rows.
AGENT_TURN_TYPES = frozenset({"assistant_response", "assistant_thinking", "tool_use", "tool_result"})

# Framework-injected "scaffolding" turns: coding-agent / CLI control messages that land in the
# user-prompt stream but are not authored by the human -- slash-command invocations and expanded
# command/skill templates, tool I/O markers, skill attachments, interrupt and image
# placeholders. Matched against the *cleaned* turn text (identifiers already scrubbed to
# ``<PATH>`` etc.). Heuristic but conservative, and used only to collapse *consecutive
# duplicates* of such messages, so a false positive can at most drop an exact back-to-back
# repeat. Longer command/skill template bodies not matched here are caught by the length rule.
_SWE_SCAFFOLDING_RE = re.compile(
    r"^\[Request interrupted by user"                      # user-interrupt marker
    r"|^\[Image:"                                          # image placeholder, e.g. [Image: image/png]
    r"|^Tool loaded\.$"                                    # tool-load notice
    r"|^Summarize the task tool output above"              # framework auto-continuation prompt
    r"|^Continue from where you left off\.$"               # framework resume/continuation prompt
    r"|^Base directory for this skill:"                    # skill scaffolding header
    r"|^Caveat: The messages below"                        # injected caveat block
    r"|Execute the following steps non-interactively"      # slash-command template body
    r"|Optionally specify a change name after"             # slash-command template body
    r"|<command-message>|<command-name>|<command-args>"    # slash-command invocation tags
    r"|<bash-input>|<bash-stdout>|<bash-stderr>"           # bash tool tags
    r"|<local-command|<manually_attached_skills>",         # local-command / skill-attachment tags
    re.IGNORECASE,
)


def is_scaffolding_turn(text: str) -> bool:
    """Whether a (cleaned) turn is a framework-injected scaffolding message (see ``_SWE_SCAFFOLDING_RE``)."""
    return bool(_SWE_SCAFFOLDING_RE.search(text))


def _is_error_response(text: str) -> bool:
    """Whether an ``assistant_response`` is an API/transport error rather than a real reply.

    Coding-agent logs record failed requests (expired auth, 500/529, rate-limit, connection
    refused) as ``assistant_response`` turns whose content is an ``API Error: ...`` / JSON error
    payload. These are *not* the agent engaging with the prompt, so a user turn resent after one
    is a retry, not a reply-then-reask; :func:`_cum_agent_by_turn` excludes them.
    """
    if not text:
        return False
    return text.lstrip().startswith("API Error") or '"type":"error"' in text


# --- slash-command / skill normalization (see the two forms below) ----------
# Form A -- a slash-command INVOCATION, recorded as just the tag block (the user typed
# ``/name args``); we reduce it to that literal text. Form B -- an EXPANDED skill/command BODY
# injected verbatim by the agent (skill file / command template), which the user did not author;
# we drop it. See ``load_swe_chat_documents``.
_CMD_TAG_BLOCK_RE = re.compile(r"<command-(message|name|args)>.*?</command-\1>", re.S | re.I)
_CMD_NAME_RE = re.compile(r"<command-name>(.*?)</command-name>", re.S | re.I)
_CMD_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.S | re.I)


def _reduce_command_invocation(text: str) -> str | None:
    """If ``text`` is a slash-command invocation (form A), return the user-typed ``/name args``.

    Only matches a *pure* invocation -- the ``<command-*>`` tag block with no other content --
    so a pasted transcript that merely contains such a tag is not mistaken for one. Returns
    ``None`` when ``text`` is not a pure invocation.
    """
    if "<command-name>" not in text:
        return None
    if _CMD_TAG_BLOCK_RE.sub("", text).strip():   # substantive content beyond the tags
        return None
    m = _CMD_NAME_RE.search(text)
    name = m.group(1).strip() if m else ""
    if not name:
        return None
    if not name.startswith("/"):
        name = "/" + name
    a = _CMD_ARGS_RE.search(text)
    args = a.group(1).strip() if a else ""
    return f"{name} {args}".strip()


def _is_skill_or_command_body(text: str) -> bool:
    """Whether ``text`` is a framework-injected skill/command BODY (form B), not user-authored.

    These are the skill file (``Base directory for this skill: ...``), a manually-attached skill
    block, or an expanded command template -- injected verbatim by Claude Code (often
    auto-triggered mid-agent-loop), never typed by the user.
    """
    return (text.startswith("Base directory for this skill:")
            or "<manually_attached_skills>" in text
            or "Execute the following steps non-interactively" in text)


def _normalize_command_turn(text: str):
    """Classify a raw user turn: ``None`` to drop (form B), ``(reduced, True)`` for a reduced
    slash-command invocation (form A), or ``(text, False)`` for an ordinary prompt."""
    if _is_skill_or_command_body(text):
        return None
    reduced = _reduce_command_invocation(text)
    if reduced is not None:
        return (reduced, True)
    return (text, False)


# ---------------------------------------------------------------------------
# Human-turn extraction -- keep only genuinely human-authored user prompts.
# ---------------------------------------------------------------------------
# The ``user_prompt`` stream carries a lot that the human did not type: coding-agent orchestration
# (``<teammate-message>`` and its untagged continuation chunks), tool-I/O echoes (``<bash-*>``,
# ``<ide_*>``, ``<task-notification>``), CLI framework wrappers (Conductor ``<system_instruction>``),
# ``/compact`` continuation summaries, and -- importantly -- the *expanded body* of a slash command,
# which the CLI injects as the turn(s) right after the human types ``/cmd`` (e.g. ``/retro`` ->
# "Reflect on the work just completed ..."). None of it is authored by the person, and the raw
# metadata cannot tell it apart (it is all ``role=user`` / ``turn_type=user_prompt``, and SWE-chat's
# own intent/pushback classifiers even labelled it like ordinary prompts). So the split is done by
# content plus adjacency, keyed on whether a real agent turn ran between two user prompts
# (``agent_before``): consecutive user turns with no agent reply are one injected block.

HUMAN_TURN_MAX_LEN = 8000    # a user turn at least this long is treated as pasted / agent content
CONTINUATION_MIN_LEN = 500   # a post-injection turn this long (or structured) is a continuation chunk

# Framework wrapper blocks removed in place (keeping any human prose around them), matched on RAW
# text where the tags are intact -- cleaning would mangle ``</system_instruction>`` into ``<PATH>``.
_WRAPPER_BLOCK_RE = re.compile(
    r"<system[_-]instruction>.*?</system[_-]instruction>"  # Conductor / attached-files wrapper (``_`` and ``-`` spellings)
    r"|<teammate-message\b[^>]*>.*?</teammate-message>",   # multi-agent orchestration message
    re.S | re.I,
)
_WRAPPER_OPEN_RE = re.compile(r"<(?:system[_-]instruction|teammate-message)\b.*", re.S | re.I)
# A leading tool marker line that prefixes an otherwise-human turn (stripped, prose kept).
_LEAD_MARKER_RE = re.compile(r"^(?:\[Request interrupted[^\]]*\]|\[Image:[^\]]*\])\s*")
# A turn that opens with markdown/list/table structure looks like a continuation/expansion chunk.
_CONTINUATION_RE = re.compile(r"(#{1,6}\s|\*\*|\||\d+[.)]\s|[-*]\s|```|>|\{)")

# Injected non-human templates removed *in place* so a real request in the same turn survives.
# BMAD-METHOD posts this fixed reminder as its own user turn right after a ``/bmad-*`` slash
# command (e.g. ``/bmad-bmb-edit-module``); the ``{project-root}`` path varies. A turn that is
# only the reminder becomes empty after removal and is dropped downstream (``empty``).
_BMAD_REMINDER_RE = re.compile(
    r"IT IS CRITICAL THAT YOU FOLLOW THIS COMMAND: LOAD the FULL \{project-root\}.*?"
    r"READ its entire contents and follow its directions exactly!",
    re.S | re.I,
)
# The OMX Explore harness prepends a fixed read-only-agent system prompt and labels the real
# user text ``User request:`` (after an ``... END EXPLORE PROMPT ...`` divider); keep only what
# follows that label. Anchored to a turn that opens with the persona. ``(?:\s|\\n)*`` also eats a
# leading literal ``\n`` -- these turns use escaped newlines, normalized later during cleaning.
# A persona turn with no ``User request:`` label is left intact here and dropped by
# :func:`_nonhuman_turn_reason` (``orchestration``).
_OMX_PROMPT_RE = re.compile(r"^You are OMX Explore\b.*?User request:(?:\s|\\n)*", re.S)

# The ``/review`` slash-command posts a fixed "You are a code reviewer. Your job is to review code
# changes ..." template body as a user turn; the only human-authored part is its ``Input:`` field
# (the review target/instruction the user typed after ``/review`` -- often a commit/branch ref, a
# free-text note, or empty). Reduce the turn to just that field, like OMX's ``User request:``. An
# empty ``Input:`` leaves the turn empty and it is dropped downstream; a body whose ``Input:``/``---``
# structure is missing is left intact and dropped by :func:`_nonhuman_turn_reason` (``skill-body``).
_REVIEW_BODY_RE = re.compile(
    r"^You are a code reviewer\. Your job is to review code changes\b.*?Input:\s*(?P<input>.*?)\s*---",
    re.S,
)


def _reduce_review_command_body(text: str) -> str:
    """If ``text`` is the ``/review`` command body, return only its ``Input:`` field; else ``text``."""
    m = _REVIEW_BODY_RE.match(text)
    return m.group("input") if m else text


def _strip_framework_wrappers(text: str) -> str:
    """Remove injected non-human template text *in place*, keeping human prose around it.

    Strips, without dropping the turn: ``<system_instruction>``/``<system-instruction>`` blocks
    (both the ``_`` and ``-`` spellings of the Conductor / attached-files wrapper) and
    ``<teammate-message ...>`` blocks; the OMX Explore read-only-agent persona (keeping only the
    text after its ``User request:`` label); the ``/review`` "You are a code reviewer ..." command
    body (keeping only its ``Input:`` field); the BMAD-METHOD ``IT IS CRITICAL ...`` command
    reminder; and a leading ``[Request interrupted ...]`` / ``[Image: ...]`` marker. A turn that
    is *only* wrapper/template becomes empty (dropped downstream by the ``empty`` reason);
    ``...</system-instruction>\\n\\nCreate a PR`` keeps ``Create a PR``. An unclosed wrapper is
    dropped from its tag to end-of-turn.
    """
    t = _WRAPPER_BLOCK_RE.sub("", text)
    if _WRAPPER_OPEN_RE.search(t):        # an unclosed wrapper -> drop from its tag to end-of-turn
        t = _WRAPPER_OPEN_RE.sub("", t)
    t = _OMX_PROMPT_RE.sub("", t)         # OMX Explore persona -> keep only the User request
    t = _reduce_review_command_body(t)    # /review "You are a code reviewer..." body -> keep only Input
    t = _BMAD_REMINDER_RE.sub("", t)      # BMAD injected command reminder
    t = t.lstrip()
    while True:
        m = _LEAD_MARKER_RE.match(t)
        if not m:
            break
        t = t[m.end():].lstrip()
    return t.strip()


def _nonhuman_turn_reason(text: str) -> str | None:
    """Reason a (wrapper-stripped) turn is not human-authored, or ``None`` if it looks human.

    Catches: ``empty`` (nothing left after stripping), framework ``tool-io`` tags
    (``<bash-*>``, ``<ide_*>``, ``<local-command*>`` at the start, or a ``<task-notification>``
    completion block anywhere in the turn -- leaked agent/tool output), ``env-context`` (an
    injected ``<environment_context>`` block: cwd/shell/date/timezone), ``orchestration``
    (multi-agent tmux-injection status lines carrying the ``[OMX_TMUX_INJECT]`` marker or an ``[OMX
    ...]`` prefix, or an OMX Explore read-only-agent persona that :func:`_strip_framework_wrappers`
    left intact because it had no ``User request:`` label -- all injected as fake user turns),
    ``compaction`` (/compact continuation summaries),
    ``plan-mode`` (``Implement the following plan: ...`` -- an approved plan the CLI injects to open
    an execution session; the plan body is model-generated plan-mode output, not human prose),
    ``skill-body`` (expanded skill/command bodies, including a ``/review`` "You are a code reviewer
    ..." body left unreduced because its ``Input:`` field could not be found), ``scaffolding``
    (standalone framework messages --
    ``Tool loaded.``, ``Continue from where you left off.``, ``Summarize the task tool output ...`` --
    see :data:`_SWE_SCAFFOLDING_RE`; these survived before because that regex was only applied to
    *consecutive* duplicates, so a lone one between two real turns slipped through), ``markdown-body``
    (a multi-paragraph Markdown document -- opens with a ``#`` heading or ``**`` bold *and* contains a
    blank line; these are the expanded skill/command templates the CLI injects as user turns, e.g.
    ``# NW-DESIGN ...\\n\\n**Wave**: ...`` or the reversed ``## Phase 7 ... ## Phase 1`` blocks, which
    reach the stream far from -- or wrapped around -- their triggering command, so adjacency alone
    misses them; the blank-line requirement spares a short single-line human ``## heading`` / ``**note**``),
    and ``too-long`` (>= :data:`HUMAN_TURN_MAX_LEN` chars -- pasted logs or long injected bodies).
    """
    if not text:
        return "empty"
    s = text.lstrip()
    low = s.lower()
    if s.startswith(("<task-notification>", "<bash-input>", "<bash-stdout>", "<bash-stderr>",
                     "<ide_", "<local-command")):
        return "tool-io"
    if "<task-notification>" in text or "</task-notification>" in text:
        return "tool-io"        # a task-completion notification block anywhere in the turn
    if "<environment_context>" in text:
        return "env-context"    # injected environment block (cwd/shell/date/timezone), not authored
    if s.startswith(("[OMX", "You are OMX Explore")) or "[OMX_TMUX_INJECT]" in s:
        return "orchestration"
    if low.startswith("this session is being continued") or low.startswith("caveat: the messages below"):
        return "compaction"
    if s.startswith("Implement the following plan"):
        return "plan-mode"
    if (s.startswith("Base directory for this skill:") or "<manually_attached_skills>" in text
            or "Execute the following steps non-interactively" in text):
        return "skill-body"
    if s.startswith("You are a code reviewer. Your job is to review code changes"):
        return "skill-body"    # unreduced /review command body (its Input: field was not found)
    if is_scaffolding_turn(s):
        return "scaffolding"
    if (s.startswith("#") or s.startswith("**")) and "\n\n" in s:
        return "markdown-body"
    if len(text) >= HUMAN_TURN_MAX_LEN:
        return "too-long"
    return None


def _is_continuation_like(text: str) -> bool:
    """Whether a turn looks like a continuation / expansion chunk (opens structured, or is longish)."""
    return bool(_CONTINUATION_RE.match(text.lstrip())) or len(text) >= CONTINUATION_MIN_LEN


def _is_markdown(text: str) -> bool:
    """Heuristic: a turn that opens with a Markdown heading (``#``) is Markdown document text."""
    return text.lstrip().startswith("#")


def merge_markdown_runs(turns_raw, is_command, agent_before, turn_ids):
    """Merge a run of consecutive Markdown turns (no agent turn between them) into one turn.

    A single Markdown document -- an injected skill reference (e.g. the ``/claude-api`` skill's API
    docs), or a long pasted spec -- is often recorded as many back-to-back ``user_prompt`` turns,
    one per ``##`` section. That inflates the turn count and hides that it is one unit (and lets the
    chunks slip past the human filter individually). A maximal run of turns that each open with
    ``#`` and, after the first, had **no agent turn in between** (``agent_before`` false) is
    concatenated into a single turn (keeping the run's first ``turn_id``/``agent_before``). The
    merged turn is then subject to the normal human-turn filter, so an over-long injected doc is
    dropped by the length cap while a short genuinely-human one survives as a single valid turn.
    Turns that do not open with ``#`` (and slash commands) pass through unchanged. Returns aligned
    ``(turns, is_command, agent_before, turn_ids)``.
    """
    turns, cmds, agents, ids = [], [], [], []
    i, n = 0, len(turns_raw)
    while i < n:
        if not is_command[i] and _is_markdown(turns_raw[i]):
            j = i + 1
            while (j < n and not is_command[j] and _is_markdown(turns_raw[j])
                   and not agent_before[j]):
                j += 1
            turns.append("\n\n".join(turns_raw[i:j]) if j - i > 1 else turns_raw[i])
            cmds.append(False)
            agents.append(agent_before[i])
            ids.append(turn_ids[i])
            i = j
        else:
            turns.append(turns_raw[i]); cmds.append(is_command[i])
            agents.append(agent_before[i]); ids.append(turn_ids[i])
            i += 1
    return turns, cmds, agents, ids


def filter_to_human_turns(turns_raw, is_command, agent_before, turn_ids):
    """Keep only human-authored turns; return aligned ``(turns, is_command, agent_before, turn_ids)``.

    Walks a session's turns in order, tracking whether the previous turn was non-human (an injected
    turn or a slash command, either of which can be followed by injected continuation/expansion
    turns):

    * a reduced slash-command invocation (``is_command``) is the human typing ``/cmd`` -- **kept**
      (unless it too reaches :data:`HUMAN_TURN_MAX_LEN`, i.e. its argument is a big paste), and
      flags that following turns may be its injected expansion;
    * any other turn is wrapper-stripped (:func:`_strip_framework_wrappers`) and **dropped** if
      non-human (:func:`_nonhuman_turn_reason`) -- this removes teammate/agent messages, tool I/O,
      orchestration status, compaction, scaffolding, skill bodies, injected Markdown-document bodies
      (``markdown-body``: the expanded skill/command templates, which open with ``#``/``**`` and span
      multiple paragraphs, wherever they land in the stream), and over-long pastes, while keeping human
      prose that merely sat inside a wrapper;
    * a turn right after a non-human turn, with **no agent turn in between** (``agent_before`` false)
      and that looks like a continuation (:func:`_is_continuation_like`), is an injected
      continuation/expansion chunk -- **dropped**; a short free-form message there (``yes``,
      ``merged``, a quick instruction) is **kept**.

    ``agent_before`` is the key signal: an agent reply between two user turns means the second is the
    human replying; a run of user turns with no agent reply is one injected block (a chunked command
    body, a multi-part agent message). See the module notes above for why metadata alone cannot do
    this. The returned per-turn lists stay aligned for the downstream cleaner and consecutive-dup
    dedup.
    """
    turns, cmds, agents, ids = [], [], [], []
    prev_nonhuman = False
    for raw, cmd, ab, tid in zip(turns_raw, is_command, agent_before, turn_ids):
        if cmd:  # human's slash-command invocation; its expansion (if any) follows
            if len(raw) < HUMAN_TURN_MAX_LEN:  # else the argument is a big paste -> drop
                turns.append(raw); cmds.append(True); agents.append(ab); ids.append(tid)
            prev_nonhuman = True
            continue
        stripped = _strip_framework_wrappers(raw)
        if _nonhuman_turn_reason(stripped) is not None:
            prev_nonhuman = True
            continue
        if prev_nonhuman and not ab and _is_continuation_like(stripped):
            prev_nonhuman = True  # still inside the injected block
            continue
        turns.append(stripped); cmds.append(False); agents.append(ab); ids.append(tid)
        prev_nonhuman = False
    return turns, cmds, agents, ids


def _dominant_model_per_session(dataset) -> pd.Series:
    """Most frequent real model id per session (model only appears on non-user turns)."""
    meta = dataset.scanner(columns=["session_id", "model"]).to_table().to_pandas()
    real = meta[meta["model"].notna() & ~meta["model"].astype(str).str.lower().isin(["none", "nan", ""])]
    return real.groupby("session_id")["model"].agg(lambda s: s.mode().iat[0])


def _recover_identity(sessions: pd.DataFrame) -> pd.Series:
    """Assign each session an identity, recovering id-less ones from their repo (NA if unresolved)."""
    has_uid = sessions["user_id"].notna() & (sessions["user_id"].astype(str).str.lower() != "nan")
    labeled = sessions[has_uid]
    users_per_repo = labeled.groupby("repo_id")["user_id"].nunique()
    sole_user_per_repo = labeled.groupby("repo_id")["user_id"].first()
    repo_user_count = sessions["repo_id"].map(users_per_repo).fillna(0).astype(int)

    idless = ~has_uid
    identity = sessions["user_id"].astype("object").where(has_uid)
    identity = identity.mask(idless & (repo_user_count == 1), sessions["repo_id"].map(sole_user_per_repo))
    identity = identity.mask(idless & (repo_user_count == 0), sessions["repo_id"])
    return identity  # (>1-user-repo id-less sessions stay NA)


def _ordered_languages(series) -> list[str]:
    """Detected languages for a session with the **primary (most frequent) language first**.

    Ties on frequency are broken by first appearance; remaining languages follow, sorted.
    Empty list when no language was detected on any turn.
    """
    vals = [x for x in series if isinstance(x, str) and x]
    if not vals:
        return []
    primary = Counter(vals).most_common(1)[0][0]
    return [primary] + sorted(set(vals) - {primary})


def _cum_agent_by_turn(dataset) -> pd.DataFrame:
    """Cumulative count of agent turns before each turn, keyed by ``(session_id, turn_number)``.

    Scans *all* turn types (a session log is mostly non-agent bookkeeping rows) and counts, in
    global ``turn_number`` order, how many :data:`AGENT_TURN_TYPES` turns precede each turn --
    **excluding ``assistant_response`` turns that are API errors** (:func:`_is_error_response`),
    since a failed request is not the agent engaging. Differencing this between two user prompts
    then tells whether any *real* agent turn ran between them -- robust to bookkeeping rows and to
    skipped (e.g. empty) user prompts, unlike a raw ``turn_number`` gap. The rare duplicate
    ``turn_number`` rows (literal duplicate source rows) carry an equal count, so collapsing to
    one row per key is safe.
    """
    turns = dataset.scanner(columns=["session_id", "turn_number", "turn_type"]).to_table().to_pandas()
    turns = turns.sort_values(["session_id", "turn_number"], kind="stable")
    turns["is_agent"] = turns["turn_type"].isin(AGENT_TURN_TYPES)
    # Demote error ``assistant_response`` turns (failed requests) so they don't count as a reply.
    ar = dataset.scanner(
        columns=["session_id", "turn_number", "content"],
        filter=(ds.field("turn_type") == "assistant_response"),
    ).to_table().to_pandas()
    err_keys = {(r.session_id, r.turn_number) for r in ar.itertuples() if _is_error_response(r.content)}
    if err_keys:
        is_err = pd.Series(list(zip(turns["session_id"], turns["turn_number"])), index=turns.index).isin(err_keys)
        turns.loc[is_err, "is_agent"] = False
    turns["cum_agent"] = turns.groupby("session_id")["is_agent"].cumsum() - turns["is_agent"]
    return (turns[["session_id", "turn_number", "cum_agent"]]
            .drop_duplicates(["session_id", "turn_number"], keep="last"))


def load_swe_chat_documents(raw_path: str | Path) -> pd.DataFrame:
    """Load SWE-chat as a normalized (uncleaned) document frame.

    Columns: ``doc_id, source, identity, turns_raw, turn_ids, agent_before, is_command,
    languages, model, model_owner, agent, started_at, ended_at, repo_id, user_id``. Beyond the
    WildChat adapter's columns this adds three per-turn lists aligned with ``turns_raw`` --
    ``turn_ids`` (the source ``turn_id``), ``agent_before`` (whether an agent turn ran since the
    previous user prompt), and ``is_command`` (whether the turn is a reduced slash-command
    invocation, so the cleaner preserves the ``/command`` token). ``turns_raw`` holds only the
    genuinely **human-authored** turns: framework-injected skill/command bodies are dropped during
    parsing, and :func:`filter_to_human_turns` then removes agent/orchestration turns, tool I/O,
    /compact summaries, and injected command expansions (and strips framework wrappers in place).
    ``repo_id``/``user_id`` are retained so the downstream cleaner can scrub them; all of these
    helper columns are dropped from the final dataset.
    """
    dataset = ds.dataset(str(raw_path), format="parquet")
    sess_model = _dominant_model_per_session(dataset)

    up = dataset.scanner(
        columns=["session_id", "user_id", "repo_id", "agent", "content",
                 "turn_number", "turn_id", "timestamp", "language"],
        filter=(ds.field("turn_type") == "user_prompt"),
    ).to_table().to_pandas()
    up = up[up["content"].notna()].copy()
    # Normalize command/skill turns on the RAW text (before cleaning): drop framework-injected
    # skill/command BODIES (form B -- not user-authored) and reduce slash-command INVOCATIONS
    # (form A) to the literal ``/name args`` the user typed. ``is_command`` marks the reduced
    # ones so the cleaner keeps the ``/command`` token and scrubs only the arguments.
    norm = up["content"].map(_normalize_command_turn)
    keep = norm.map(lambda x: x is not None).to_numpy()
    up = up[keep].copy()
    norm = norm[keep]
    up["content"] = [x[0] for x in norm]
    up["is_command"] = [x[1] for x in norm]
    # Order user prompts by the global ``turn_number`` (identical to conversation order but the
    # canonical key for the agent-between computation below).
    up = up.sort_values(["session_id", "turn_number"], kind="stable")
    up["_lang"] = up["language"].map(normalize_language)

    # ``agent_before[i]`` = an agent turn ran between user prompt i-1 and i in the same session.
    # Merge the cumulative agent count, re-sort (merge may reorder), then difference it between
    # consecutive *non-empty* user prompts.
    up = up.merge(_cum_agent_by_turn(dataset), on=["session_id", "turn_number"], how="left")
    up = up.sort_values(["session_id", "turn_number"], kind="stable")
    up["agent_before"] = (up["cum_agent"] - up.groupby("session_id")["cum_agent"].shift(1)) > 0

    sessions = up.groupby("session_id", sort=False).agg(
        user_id=("user_id", "first"),
        repo_id=("repo_id", lambda s: next((x for x in s if pd.notna(x)), None)),
        agent=("agent", "first"),
        languages=("_lang", _ordered_languages),
        started_at=("timestamp", "min"),    # earliest user-prompt turn (conversation start)
        ended_at=("timestamp", "max"),      # latest user-prompt turn (conversation end)
        turns_raw=("content", list),        # raw user-prompt turns, in conversation order
        turn_ids=("turn_id", list),         # source turn_id per turn (aligned with turns_raw)
        agent_before=("agent_before", list),  # per turn: agent turn ran since the previous prompt
        is_command=("is_command", list),    # per turn: reduced slash-command invocation (form A)
    ).reset_index()

    # First merge Markdown documents that were split across consecutive turns (e.g. an injected
    # skill reference recorded one ``##`` section per turn), then keep only genuinely human-authored
    # turns (strip framework wrappers; drop agent/framework turns, tool I/O, compaction, plan-mode
    # plans, and injected slash-command expansions). Sessions left with no human turns are dropped
    # downstream by ``drop_empty_documents``. See merge_markdown_runs / filter_to_human_turns.
    def _process(tr, ic, ab, ti):
        return filter_to_human_turns(*merge_markdown_runs(tr, ic, ab, ti))

    kept = [
        _process(tr, ic, ab, ti)
        for tr, ic, ab, ti in zip(sessions["turns_raw"], sessions["is_command"],
                                  sessions["agent_before"], sessions["turn_ids"])
    ]
    sessions["turns_raw"] = [k[0] for k in kept]
    sessions["is_command"] = [k[1] for k in kept]
    sessions["agent_before"] = [k[2] for k in kept]
    sessions["turn_ids"] = [k[3] for k in kept]

    sessions["model"] = sessions["session_id"].map(sess_model).fillna("<no_model>")
    sessions["model_owner"] = sessions["model"].map(model_owner)
    sessions["identity"] = _recover_identity(sessions)
    sessions = sessions[sessions["identity"].notna()].copy()
    sessions["identity"] = sessions["identity"].astype(str)

    sessions["doc_id"] = "sc-" + sessions["session_id"].astype(str)
    sessions["source"] = "swe-chat"
    sessions["agent"] = sessions["agent"].where(sessions["agent"].notna(), None)
    for _col in ("started_at", "ended_at"):
        sessions[_col] = pd.to_datetime(sessions[_col], utc=True).map(
            lambda t: t.isoformat() if pd.notna(t) else None
        )
    return sessions[[
        "doc_id", "source", "identity", "turns_raw", "turn_ids", "agent_before", "is_command",
        "languages", "model", "model_owner", "agent", "started_at", "ended_at", "repo_id", "user_id",
    ]]
