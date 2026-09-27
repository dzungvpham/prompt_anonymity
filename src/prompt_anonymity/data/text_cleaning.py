"""Shared text-cleaning backbone for the unified prompt dataset.

Both sources (WildChat, SWE-chat) run their user prompts through the *same* two
functions here, so the cleaned ``text`` field is processed identically regardless of
origin -- otherwise the ``source`` label would leak through surface tokens (e.g. a real
URL vs a scrubbed ``<URL>``), contaminating both attribution and clustering.

Two stages, kept separate so callers can choose how much to apply:

* :func:`normalize_whitespace` -- structural tidy-up that *preserves* newlines and
  indentation, since those are themselves stylometric signal.
* :func:`scrub_identifiers` -- replaces explicit identifiers (URLs, emails, IPs, file
  paths, and -- when known -- repository/user tokens) with fixed ``<PLACEHOLDER>``
  sentinels. This is the privacy pass, and it is also what makes the project's thesis
  testable: what remains once explicit identifiers are gone is *writing style*. Its
  ``mask_ids`` flag additionally scrubs opaque ids (UUIDs, hex hashes, commit SHAs) to
  ``<ID>`` and shell login strings (``user@host``) to ``<HOST>`` -- common in agentic
  logs, so only SWE-chat enables it for now.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Placeholder vocabulary -- identical across every source.
# ---------------------------------------------------------------------------
URL_PLACEHOLDER = "<URL>"
EMAIL_PLACEHOLDER = "<EMAIL>"
IP_PLACEHOLDER = "<IP>"
PATH_PLACEHOLDER = "<PATH>"
REPO_PLACEHOLDER = "<REPO>"
USER_PLACEHOLDER = "<USER>"
ID_PLACEHOLDER = "<ID>"      # opaque ids: UUIDs, hex hashes, git commit SHAs (mask_ids only)
HOST_PLACEHOLDER = "<HOST>"  # shell login strings: user@bare-hostname (mask_ids only)

# ---------------------------------------------------------------------------
# Identifier patterns.
# ---------------------------------------------------------------------------
URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
# Dotted IPv4 not glued to a word char on either side (so version strings like
# "v1.2.3.4" embedded in a token are left alone, but a bare "192.168.0.1" is caught).
IPV4_RE = re.compile(r"(?<!\w)(?:\d{1,3}\.){3}\d{1,3}(?!\w)")
# Filepaths: windows, unix-absolute, tilde-home, and relative paths. A slash path needs
# **>= 2 ``/`` delimiters** so single-slash things like ``/init`` or ``</tag>`` are left
# alone; a ``~/...`` home path or a ``dir/file.ext`` relative path is unambiguous even
# with one slash. ``(?<!\w)`` lets a path be wrapped in quotes/parens/tags without also
# catching a mid-word slash like "and/or".
PATH_RE = re.compile(
    r"""(?<!\w)(?:
          [A-Za-z]:\\[^\s]+                       # C:\Users\...             (windows)
        | ~/[\w.\-][\w.\-/]*                       # ~/config  ~/Users/x/...  (tilde home)
        | /[\w.\-]+/[\w.\-/]*                      # /Users/x/...             (absolute, >=2 slashes)
        | [\w.\-]+/[\w.\-]+/[\w.\-/]+              # a/b/c relative (>=3 segments)
        | [\w.\-]+/[\w.\-/]*\.[A-Za-z0-9]{1,6}     # src/main.py relative w/ extension
    )""",
    re.VERBOSE,
)
# Slash-separated *numbers* are dates, fractions, ratios or ranges -- never a filesystem
# path -- but would otherwise be swallowed by :data:`PATH_RE`'s relative-path branch.
# Matching on shape (digits joined by ``/ - . :``, starting with a digit) rather than one
# field order spares every date ordering without guessing which field is the month.
_NUMERIC_TOKEN_RE = re.compile(r"^\d+(?:[/\-.:]\d+)+$")

# --- opaque-id patterns (only applied when scrub_identifiers(mask_ids=True)) -----------
# Canonical UUID: 8-4-4-4-12 hex groups. Essentially never a natural-language word.
UUID_RE = re.compile(
    r"(?<![\w-])[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}(?![\w-])"
)
# A bare hex run of >= 7 chars -- a git short/long SHA or an MD5/SHA-1/SHA-256 digest. This
# regex only bounds the run; :func:`_looks_like_hash` decides whether it's really an id, so
# all-letter hex words ("decade") and plain decimal numbers are left alone.
_HEX_RUN_RE = re.compile(r"(?<![\w-])[0-9a-fA-F]{7,}(?![\w-])")
# Hex-valid strings that are recognized technical *names*, not identifiers -- never masked.
_HEX_NOT_IDS = {"ed25519"}
# Shell login strings copy-pasted from a terminal prompt: ``user@host`` where ``host`` is a
# bare, dotless machine name (a dotted ``user@host.tld`` is already caught as <EMAIL>). The
# host must start with a letter and not be a lone version tag, so dependency/action refs
# (``actions/checkout@v4``, ``pkg@5``) are spared.
_HOST_CHANNEL_RE = r"(?:v\d|main|master|latest|stable|beta|alpha|next|canary|nightly|edge|release|dev|prod|staging|HEAD|head)"
USER_HOST_RE = re.compile(
    rf"(?<![\w/@.\-])"                       # left: not glued to word/slash/@/dot/hyphen
    rf"[A-Za-z0-9_][\w.+\-]*"                # local part (no slash)
    rf"@"
    rf"(?!{_HOST_CHANNEL_RE}(?![\w.\-]))"    # host is not a lone version/channel tag (@v4, @main)
    rf"[A-Za-z][A-Za-z0-9\-]*"               # host: bare machine name, letter-initial, dotless
    rf"(?![\w.\-])"                          # right: no trailing word/dot/hyphen (dotted => <EMAIL>)
)

# Repo-name tokens too generic to scrub as identifiers -- scrubbing a repo literally
# named "app"/"cli"/"api" would eat ordinary prose, so we skip these standalone tokens
# (the full ``owner/repo`` slug and the owner token still carry the identifying signal).
# The placeholder words are here too, so scrubbing a repo token never eats a placeholder.
COMMON_REPO_WORDS = {
    "agent", "app", "cli", "site", "api", "web", "core", "sdk", "docs", "data",
    "main", "code", "dev", "bot", "chat", "tool", "tools", "server", "client",
    "backend", "frontend", "utils", "common", "config", "demo", "example",
    "examples", "test", "tests", "www", "lib", "ui",
    "repo", "path", "url", "user", "email", "ip",
}

_CR_RE = re.compile(r"\r\n?")
_TRAILING_WS_RE = re.compile(r"[ \t]+(?=\n)")
_MANY_BLANKLINES_RE = re.compile(r"\n{3,}")


def normalize_whitespace(text: str) -> str:
    """Tidy whitespace while preserving line structure.

    Converts CRLF/CR to LF, strips trailing spaces/tabs at line ends, and collapses runs
    of 3+ newlines to a single blank line. Newlines, indentation, and intra-line spacing
    are otherwise preserved, since those are part of a user's writing style. Also
    unescapes the literal ``\\n`` two-character sequence the legacy WildChat CSV used in
    place of real newlines.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\\n", "\n")
    text = _CR_RE.sub("\n", text)
    text = _TRAILING_WS_RE.sub("", text)
    text = _MANY_BLANKLINES_RE.sub("\n\n", text)
    return text.strip()


def _scrub_known_token(text: str, token: str, placeholder: str, *, min_len: int = 3) -> str:
    """Replace standalone whole-word occurrences of a known identifier token."""
    token = "" if token is None else str(token).strip()
    if len(token) < min_len or token.lower() in COMMON_REPO_WORDS:
        return text
    return re.sub(rf"(?<!\w){re.escape(token)}(?!\w)", f" {placeholder} ", text, flags=re.IGNORECASE)


def _tokens_longest_first(tokens) -> list[str]:
    """Deduplicate known-identifier ``tokens`` and order them longest first (ties alphabetically).

    Order matters: for a repo slug like ``foo/foo-new-map``, replacing the owner token
    ``foo`` first leaves ``<REPO> -new-map`` still identifying, while replacing the longer
    ``foo-new-map`` first yields a clean ``<REPO>``. Longest-first is deterministic and the
    more specific match.
    """
    return sorted({t for t in tokens if t}, key=lambda t: (-len(t), t))


def _looks_like_hash(tok: str) -> bool:
    """True if a bare hex run should be treated as an opaque id (hash / commit SHA).

    A run of >= 32 hex chars always qualifies. A shorter run (7..31) qualifies only if it
    mixes a hex *letter* (a-f) and a *digit*, sparing English words spelled from a-f only
    (``decade``) and plain decimal numbers. Recognized hex-valid names (``ed25519``) are
    always kept.
    """
    if tok.lower() in _HEX_NOT_IDS:
        return False
    if len(tok) >= 32:
        return True
    has_alpha = any(c in "abcdefABCDEF" for c in tok)
    has_digit = any(c.isdigit() for c in tok)
    return has_alpha and has_digit


def _mask_paths(text: str) -> str:
    """Replace file paths with ``<PATH>``, leaving date-shaped numeric tokens intact.

    Everything :data:`PATH_RE` matches becomes ``<PATH>`` *except* an all-numeric slash token
    (see :data:`_NUMERIC_TOKEN_RE`) -- a date in any field order, a fraction, or a numeric range.
    Applied to both sources, since users of either paste dates.
    """
    return PATH_RE.sub(
        lambda m: m.group(0) if _NUMERIC_TOKEN_RE.match(m.group(0)) else f" {PATH_PLACEHOLDER} ",
        text,
    )


def _mask_hashes(text: str) -> str:
    """Replace UUIDs and hash-like hex runs (see :func:`_looks_like_hash`) with ``<ID>``."""
    text = UUID_RE.sub(f" {ID_PLACEHOLDER} ", text)
    return _HEX_RUN_RE.sub(
        lambda m: f" {ID_PLACEHOLDER} " if _looks_like_hash(m.group(0)) else m.group(0), text
    )


def scrub_identifiers(
    text: str, *, repo_id: str | None = None, user_id: str | None = None, mask_ids: bool = False,
) -> str:
    """Replace explicit identifiers in ``text`` with fixed placeholder sentinels.

    Generic patterns scrubbed on every source: URLs -> ``<URL>``, emails -> ``<EMAIL>``,
    IPv4 addresses -> ``<IP>``, file paths -> ``<PATH>`` (see :func:`_mask_paths`). When
    known, per-document identifiers are scrubbed too: ``repo_id`` (an ``owner/repo`` slug
    plus its tokens) -> ``<REPO>``, and ``user_id`` -> ``<USER>``. WildChat has no such
    tokens and passes ``None`` for both.

    ``mask_ids`` turns on two extra scrubs for identifier-heavy agentic/terminal logs
    (SWE-chat only, for now): shell login strings -> ``<HOST>``, and opaque ids -- UUIDs
    and hash-like hex runs -> ``<ID>`` (see :func:`_looks_like_hash`).

    Whitespace/newline structure is preserved; run :func:`normalize_whitespace` first for
    structural tidy-up too.
    """
    if not isinstance(text, str):
        return ""
    t = URL_RE.sub(f" {URL_PLACEHOLDER} ", text)
    t = EMAIL_RE.sub(f" {EMAIL_PLACEHOLDER} ", t)
    if mask_ids:  # dotless user@host that <EMAIL> (which needs a dotted domain) does not catch
        t = USER_HOST_RE.sub(f" {HOST_PLACEHOLDER} ", t)
    t = IPV4_RE.sub(f" {IP_PLACEHOLDER} ", t)
    t = _mask_paths(t)
    if mask_ids:  # after PATH, so a hash embedded in a path stays part of the single <PATH>
        t = _mask_hashes(t)

    rid = "" if repo_id is None else str(repo_id).strip()
    if rid and rid.lower() != "nan":
        # Scrub the full owner/repo slug first, then the owner and repo-name tokens, longest
        # first so the most specific match wins (see :func:`_tokens_longest_first`).
        t = re.sub(rf"(?<!\w){re.escape(rid)}(?!\w)", f" {REPO_PLACEHOLDER} ", t, flags=re.IGNORECASE)
        for tok in _tokens_longest_first(rid.split("/")):
            t = _scrub_known_token(t, tok, REPO_PLACEHOLDER)

    uid = "" if user_id is None else str(user_id).strip()
    if uid and uid.lower() != "nan":
        for tok in _tokens_longest_first([uid, uid.split("@")[0]]):
            t = _scrub_known_token(t, tok, USER_PLACEHOLDER)

    # Squeeze inline runs of spaces/tabs left by substitution, but keep line-leading
    # indentation intact -- it's itself an authorship signal.
    t = re.sub(r"(?<=\S)[ \t]{2,}", " ", t)
    t = _TRAILING_WS_RE.sub("", t)
    return t.strip()


def clean_prompt(
    text: str, *, repo_id: str | None = None, user_id: str | None = None, mask_ids: bool = False,
) -> str:
    """Full cleaning of one user prompt: normalize whitespace, then scrub identifiers."""
    return scrub_identifiers(
        normalize_whitespace(text), repo_id=repo_id, user_id=user_id, mask_ids=mask_ids,
    )
