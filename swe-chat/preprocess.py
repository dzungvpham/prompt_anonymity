# Preprocess SWE-chat into a cached, scrubbed per-session table for the linkage study
# (analog of wildchat's deduped CSV). One row per user_prompt session (all languages):
#   session_id, user_id, repo_id, agent, model (dominant), model_owner,
#   language, timestamp, num_turns, content
# `content` = the session's user prompts joined with \n===\n, then scrubbed of
# user-identifying tokens so StyloMetrix sees writing style, not identifiers:
#   usernames / emails -> <USER>,  filepaths -> <PATH>,  URLs -> <URL>,
#   repo owner / name -> <REPO>.
# `language` = comma-separated unique languages the source detected across the
# session's user prompts (empty if none detected). The source only tags language on
# user_prompt rows, so it is derived from exactly the rows we keep.
import re
import pandas as pd
import pyarrow.dataset as ds

DATASET_PATH = "/datasets/ai/salt-nlp/hub/datasets--SALT-NLP--SWE-chat/snapshots/f66cca95b14caaa4177f7ed5eaa424608dadcffa/conversations.parquet"
OUTPUT_PATH = "swe_chat_sessions.csv"
TURN_DELIM = "\n===\n"


def model_owner(model):
    """Map a model id to the org that serves it (the data's true observer).
    All of a vendor's variants collapse to one owner: claude-* -> Anthropic, etc.
    Anonymized codenames / <synthetic> / missing -> 'unknown'."""
    if not isinstance(model, str) or not model:
        return "unknown"
    m = model.lower()
    if m.startswith("claude"):
        return "Anthropic"
    if m.startswith("gpt") or "codex" in m:
        return "OpenAI"
    if m.startswith("gemini"):
        return "Google"
    if m.startswith("glm"):
        return "Zhipu"
    return "unknown"


URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Filepaths: windows, unix-absolute/home/tilde, and relative paths that are clearly
# paths (>=3 segments, or ending in a file extension) so prose like "and/or" is spared.
# (?<!\w): match a path even when wrapped in `backticks`/"quotes"/(parens), but NOT a
# bare mid-word slash like "and/or" (where the slash follows a word char).
PATH_RE = re.compile(
    r"""(?<!\w)(?:
          [A-Za-z]:\\[^\s]+                       # C:\Users\...
        | ~?/[\w.\-/]+                             # /Users/x/...  or  ~/...
        | [\w.\-]+/[\w.\-]+/[\w.\-/]+              # a/b/c relative (>=3 segments)
        | [\w.\-]+/[\w.\-/]*\.[A-Za-z0-9]{1,6}     # src/main.py relative w/ extension
    )""",
    re.VERBOSE,
)


COMMON_REPO_WORDS = {
    "agent", "app", "cli", "site", "api", "web", "core", "sdk", "docs", "data",
    "main", "code", "dev", "bot", "chat", "tool", "tools", "server", "client",
    "backend", "frontend", "utils", "common", "config", "demo", "example",
    "examples", "test", "tests", "www", "lib", "ui",
    # placeholder words, so scrubbing a repo token never eats a placeholder we set
    "repo", "path", "url", "user",
}


def scrub(text, user_id, repo_id):
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(" <URL> ", text)
    t = EMAIL_RE.sub(" <USER> ", t)            # emails are user-identifying too
    t = PATH_RE.sub(" <PATH> ", t)             # removes usernames hiding inside paths

    # repo_id is "owner/repo": scrub the full slug, then the owner and repo-name
    # tokens standalone. Skip very short or common-word tokens (e.g. a repo named
    # "app"/"cli") so ordinary prose survives -- the slug + owner carry the signal.
    rid = "" if repo_id is None else str(repo_id).strip()
    if rid and rid.lower() != "nan":
        t = re.sub(rf"(?<!\w){re.escape(rid)}(?!\w)", " <REPO> ", t, flags=re.IGNORECASE)
        for tok in {p for p in rid.split("/") if p}:
            if len(tok) >= 3 and tok.lower() not in COMMON_REPO_WORDS:
                t = re.sub(rf"(?<!\w){re.escape(tok)}(?!\w)", " <REPO> ", t, flags=re.IGNORECASE)

    uid = "" if user_id is None else str(user_id).strip()
    if uid and uid.lower() != "nan":
        tokens = {uid}
        if "@" in uid:
            tokens.add(uid.split("@")[0])
        for tok in tokens:                     # remaining standalone username mentions
            if len(tok) >= 3:
                t = re.sub(rf"(?<!\w){re.escape(tok)}(?!\w)", " <USER> ", t, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", t).strip()


def first_non_null(s):
    s = s.dropna()
    return s.iloc[0] if len(s) else None


def join_langs(s):
    # comma-separated unique detected languages for the session (sorted; "" if none)
    return ",".join(sorted(set(s.dropna())))


# 1. session -> dominant model (model only appears on assistant/tool rows)
dataset = ds.dataset(DATASET_PATH, format="parquet")
meta = dataset.to_table(columns=["session_id", "model"]).to_pandas()
sess_model = (
    meta[meta["model"].notna()].groupby("session_id")["model"].agg(lambda s: s.mode().iat[0])
)

# 2. user prompts (all languages) -> one document per session
up = dataset.to_table(
    columns=["session_id", "user_id", "repo_id", "agent", "content",
             "conversation_turn_number", "timestamp", "language"],
    filter=(ds.field("turn_type") == "user_prompt"),
).to_pandas()
up = up[up["content"].notna()]
sessions = (
    up.sort_values(["session_id", "conversation_turn_number"])
    .groupby("session_id")
    .agg(
        user_id=("user_id", "first"),
        repo_id=("repo_id", first_non_null),
        agent=("agent", "first"),
        language=("language", join_langs),
        timestamp=("timestamp", "min"),
        num_turns=("content", "size"),
        content=("content", TURN_DELIM.join),
    )
    .reset_index()
)

# 3. attach model + owner
sessions["model"] = sessions["session_id"].map(sess_model).fillna("<no_model>")
sessions["model_owner"] = sessions["model"].apply(model_owner)

# 4. scrub user-identifying tokens (usernames, paths, URLs, repo owner/name)
sessions["content"] = [
    scrub(c, u, r)
    for c, u, r in zip(sessions["content"], sessions["user_id"], sessions["repo_id"])
]

sessions = sessions[
    ["session_id", "user_id", "repo_id", "agent", "model", "model_owner",
     "language", "timestamp", "num_turns", "content"]
]
sessions.to_csv(OUTPUT_PATH, index=False)
print(f"Wrote {len(sessions)} sessions to {OUTPUT_PATH}")
print("\n=== sessions per (agent, model_owner) ===")
print(sessions.groupby(["agent", "model_owner"]).size().sort_values(ascending=False).to_string())
print("\n=== sessions per detected-language combo (top 12) ===")
print(sessions["language"].replace("", "<none>").value_counts().head(12).to_string())
