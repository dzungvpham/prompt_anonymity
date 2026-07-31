"""Author-identity construction and pseudonymization for the unified dataset.

An *author* means different things in the two sources, and that difference is documented
rather than hidden:

* **WildChat** has no user accounts, so an author is the request-fingerprint tuple
  ``hashed_ip | accept_language | device_info`` (the same identity the linkage study
  used). ``device_info`` is a coarse browser/OS/device string derived from the request
  user-agent via a lookup table (see :func:`load_ua_device_map`).
* **SWE-chat** has a ``user_id``, with repo-based recovery for id-less sessions
  (handled in ``sources_swe_chat.py``).

The same WildChat user-agent lookup also answers a second question the pipeline needs before
anything else: *was this conversation typed into a browser at all?* See
:func:`is_programmatic_user_agent` and the "Programmatic clients" note below.

Whatever the raw identity, :func:`hash_author_id` turns it into an opaque, salted,
source-prefixed pseudonym for public release: the same author always maps to the same
``author_id``, but the raw fingerprint / user handle is not published. The salt is a
fixed project constant (not a secret) -- its only job is to make the ids opaque and
stable, not to provide cryptographic anonymity (the original public source datasets
already contain the underlying text).
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

# Fixed, non-secret salt so author_ids are stable across rebuilds and not a bare hash of
# the raw identity. Bump the version suffix if you ever want to re-pseudonymize.
AUTHOR_ID_SALT = "prompt-anonymity-dataset-v1"
_UA_MAP_PATH = Path(__file__).with_name("resources") / "ua_device_map.json"


def load_ua_device_map(path: str | Path = _UA_MAP_PATH) -> dict[str, str]:
    """Load the precomputed ``user-agent string -> device_info`` lookup.

    The map is derived once from WildChat's ``parsed_user_agents.json`` (see
    ``resources/`` and the generating snippet in the dataset README) so this package does
    not depend on the ``wildchat/`` directory at build time. ``device_info`` is a
    ``browser;os;device;cpu`` style string, or the leading token of the user-agent when
    the parser could not identify a browser.
    """
    with open(path) as f:
        return json.load(f)


def wildchat_device_info(user_agent: str | None, ua_map: dict[str, str]) -> str:
    """Map a raw user-agent to its coarse ``device_info`` string (``"N/A"`` if unknown)."""
    if not user_agent:
        return "N/A"
    return ua_map.get(user_agent, "N/A")


def wildchat_identity(hashed_ip: str | None, accept_language: str | None, device_info: str | None) -> str:
    """Build the WildChat request-fingerprint identity ``hashed_ip|accept_language|device_info``."""
    return "|".join([str(hashed_ip or ""), str(accept_language or ""), str(device_info or "")])


# ---------------------------------------------------------------------------
# Programmatic clients
#
# A large share of WildChat traffic never came from a person at a keyboard: it was posted by
# code. On the three studied models, 9.0% of conversations carry an HTTP-library user-agent
# (``gradio_client`` alone is 7.5%), and among the identities that survive the >=2-documents
# floor -- the only ones an authorship study can use -- the share is far higher still, because
# one script generates thousands of conversations under a single request fingerprint while a
# person generates a handful.
#
# These are poison for authorship work, and not primarily because the *text* is synthetic --
# ``gradio_client`` traffic is often real humans typing into a HuggingFace Space. The problem is
# that one ``author_id`` then conflates *many* humans: a relay bot forwarding a Discord channel's
# requests looks like one extraordinarily prolific, stylistically incoherent author. Content
# heuristics do not catch this; the client string does, which is why the filter lives here at the
# identity layer rather than in the text pipeline.
# ---------------------------------------------------------------------------

# Clients the user-agent parser *does* name, but which are HTTP libraries, headless automation, or
# crawlers rather than interactive browsers. Compared lower-cased against the browser field of the
# device map. ``Mozilla`` is here because it is what a bare, product-less ``Mozilla/5.0`` parses to
# -- every genuine browser appends its own product token, so the bare string is a spoof or a stub.
PROGRAMMATIC_CLIENT_NAMES = frozenset({
    "adsbot-google-mobile", "axios", "bun", "chrome headless", "curl", "go-http-client",
    "google-read-aloud", "mozilla", "node-fetch", "okhttp", "postmanruntime", "python-httpx",
    "python-requests", "undici",
})

# Library / automation / crawler tokens matched against the *raw* user-agent string. This is the
# belt to the device map's braces: it catches a programmatic client whose user-agent is missing
# from the map (the map covers every user-agent in the current snapshot, but a rebuilt or extended
# snapshot may add one) without waiting for the map to be regenerated. Word-bounded so a token
# cannot fire on a substring of a device or browser name.

# A user-agent that begins with an HTTP *header name* was assembled by hand -- the client wrote
# ``{"User-Agent": "User-Agent:Mozilla/5.0 ..."}``, prepending the field name to its own value.
# No browser emits this. It is worth a rule of its own because the one client doing it in WildChat
# is a high-volume relay whose forged Chrome string otherwise passes every other check: 62,365
# conversations under a single request fingerprint, spanning 71 primary languages, every one of
# them a single turn, active around the clock for 276 days. See the "Programmatic clients" note --
# this is the identity-conflation failure in its purest form, and the malformed header is the only
# metadata that betrays it. Measured across all 3,199,860 conversations of the raw snapshot, this
# pattern matches exactly one user-agent string from exactly one IP: no collateral.
_MALFORMED_UA_RE = re.compile(
    r"^\s*(?:user-?agent|accept(?:-language|-encoding)?|host|referer|connection)\s*[:=]", re.I
)

_PROGRAMMATIC_UA_RE = re.compile(
    r"""(?ix) \b(?:
          python | python-requests | python-httpx | httpx | aiohttp | urllib | requests
        | curl | libcurl | wget | okhttp | axios | node | node-fetch | undici | bun | deno
        | go-http-client | java | jakarta | apache-httpclient | httpclient | dart | guzzle
        | libwww-perl | lwp | restsharp | scrapy | postmanruntime | insomnia | reqwest | ktor
        | gradio_client | gradio_client_php | huggingface_hub | hf_hub | openai | langchain
        | dalvik | cpp-httplib
        | headlesschrome | phantomjs | selenium | puppeteer | playwright
        | bot | crawler | spider | scraper
    )\b""",
)


def is_programmatic_user_agent(user_agent: str | None, ua_map: dict[str, str]) -> bool:
    """True if ``user_agent`` belongs to code (an HTTP client, script, or crawler), not a browser.

    The rule is an **allowlist**: a conversation is treated as human only when its user-agent
    parses to a recognizable interactive browser. Three ways to fail, in order:

    1. **No user-agent at all, or a malformed one.** Every real browser sends one; an empty field
       is a bare HTTP request (0.6% of the three-model slice). A value starting with ``{`` is a
       client that dumped its whole header dict into the field, and one beginning with a header
       *name* (``User-Agent:Mozilla/5.0 ...``, :data:`_MALFORMED_UA_RE`) is one that prepended the
       field name to its own value -- neither is a browser signature. The second case is what
       exposes WildChat's largest relay identity, whose forged Chrome string is otherwise clean.
    2. **A library/automation token in the raw string** -- ``httpx``, ``gradio_client``,
       ``okhttp``, ``HeadlessChrome``, ... (:data:`_PROGRAMMATIC_UA_RE`). Independent of the
       device map, so it still fires on a user-agent the map has never seen.
    3. **No browser identified, or a non-browser client named.** ``ua_map`` yields the
       ``browser;os;device;cpu`` device string only when the parser recognized a browser; the
       fallback form (no ``;``) means it did not, and those are clients like ``gradio_client``,
       ``node``, or a raw header blob. A parsed name in :data:`PROGRAMMATIC_CLIENT_NAMES` (an
       HTTP library the parser happens to name) fails the same way.

    In-app webviews (WeChat, Line, Instagram, ...) and desktop app shells (Electron, VS Code)
    **pass** -- a person is typing in all of them.
    """
    if (not user_agent or not user_agent.strip() or user_agent.lstrip().startswith("{")
            or _MALFORMED_UA_RE.match(user_agent)):
        return True
    if _PROGRAMMATIC_UA_RE.search(user_agent):
        return True
    device_info = ua_map.get(user_agent)
    if not device_info or ";" not in device_info:
        return True
    return device_info.split(";")[0].strip().lower() in PROGRAMMATIC_CLIENT_NAMES


def hash_author_id(source: str, raw_identity: str, *, salt: str = AUTHOR_ID_SALT) -> str:
    """Return an opaque, stable, source-prefixed pseudonym for a raw identity.

    Same ``(source, raw_identity)`` always yields the same id; different authors collide
    only with negligible probability (16 hex chars = 64 bits). Format: ``"{source}-{hex16}"``.
    """
    digest = hashlib.sha256(f"{salt}|{source}|{raw_identity}".encode("utf-8")).hexdigest()
    return f"{source}-{digest[:16]}"


def hash_token(source: str, raw_value: str, *, salt: str = AUTHOR_ID_SALT) -> str:
    """Opaque salted hash for an auxiliary identifier (e.g. a SWE-chat repo slug)."""
    digest = hashlib.sha256(f"{salt}|{source}|repo|{raw_value}".encode("utf-8")).hexdigest()
    return digest[:16]
