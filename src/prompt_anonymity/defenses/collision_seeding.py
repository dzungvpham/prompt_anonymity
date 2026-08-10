r"""Collision seeding: manufacture *shared* style so authors collide instead of separating.

Every other defense in this package removes signal -- push each author toward one neutral style
(``styleremix``, ``qwen_rewrite``), or add per-word DP noise (``dp_mlm``). Collision seeding is the
additive counterpart. It picks a small set of unusual-but-natural writing quirks and gives the *same*
quirk to a group of otherwise-unrelated authors, so an attacker who latches onto the quirk lands on a
**group** rather than a person. The goal is not to make an author unrecognizable; it is to make them
confusable with the ~N/K others who share their profile.

Three properties make this work, and each is easy to get wrong:

**1. Profiles, not independent markers.** If each author drew markers independently, ``M`` markers
with ``s`` per author would give ``C(M, s)`` distinct signatures -- at M=40, s=4 that is ~91,000, far
more than the author count, so every author would get a *unique* fingerprint and the defense would
make attribution strictly easier. A fixed codebook of ``K`` profiles caps the number of distinct
signatures at ``K`` by construction. ``K`` is the real privacy knob: expected collision group is
``N/K`` authors. The failure mode is available as an ablation (``independent=True``) precisely
because it is instructive.

**2. Inconsistency.** A marker applied to 100% of an author's documents is a *cleaner* signal than
any real habit -- perfectly reliable, and visibly synthetic. Real quirks are inconsistent: people
misspell a word most of the time, not always. So each (author, marker) pair draws its own rate from
``U[rate_min, rate_max]`` (default 40-70%), and the coin is flipped per document. Varying the rate
per author matters as much as the rate itself: a fixed 55% across all authors would make "55%"
the tell instead. With ~10 documents per author, an attacker's estimate of the rate has standard
error ~0.157 against a prior standard deviation of only ~0.087, so the rate is essentially
unidentifiable at this corpus's documents-per-author.

**3. Natural base rates.** A marker that appears *nowhere* in the corpus naturally becomes a perfect
group indicator -- no background noise to hide in. Run ``--audit`` before fixing a marker set and
drop the zero-base-rate ones. Several whitespace/layout quirks fail this test on this corpus because
:func:`~prompt_anonymity.data.text_cleaning.normalize_whitespace` and ``scrub_identifiers`` already
squeezed those patterns out at build time (runs of 3+ newlines are collapsed; runs of 2+ spaces after
a non-space are squeezed), so e.g. "double space after a period" has a base rate of exactly zero.

Two structural facts about this pipeline shape the implementation:

**No caching.** :meth:`~prompt_anonymity.caching.IndexedRowCache.apply` computes *distinct source
strings* and maps results back by source text. Every other defense is a pure function of its text, so
that has never mattered; this one is a function of ``(text, author_id, doc_id)``, and routing it
through the cache would hand two authors who both wrote "thanks!" the same output -- destroying
exactly the property the defense exists to create. Since the transform is pure Python string work
(microseconds per turn, versus GPU-hours for ``dp_mlm``), the cache buys nothing here. This class
subclasses :class:`~prompt_anonymity.defenses.base.CachedDefense` for the registry contract and
overrides :meth:`transform` directly, ignoring the cache -- the same escape hatch
:mod:`~prompt_anonymity.defenses.styleremix_openanon` uses. Reproducibility comes from seeded
determinism instead of from disk.

**No global author view.** ``apply_defenses --num-shards`` gives each SLURM task every N-th
*document*, so no task ever sees all of an author's work or the full author list; a
shuffle-and-partition assignment could not agree across shards. Every decision here is therefore a
pure function of a keyed hash of ``(seed, author_id, marker_key, doc_id)``. Shard layout is
irrelevant, re-runs are bit-identical, and the assignment manifest can be rebuilt offline from the
author list alone, without re-running the defense.

Stdlib only -- no new dependency. ``pandas``/``pyarrow`` are imported lazily inside the CLI helpers
so importing this module (which the registry does at package import) stays cheap.

Command line::

    python -m prompt_anonymity.defenses.collision_seeding --selftest
    python -m prompt_anonymity.defenses.collision_seeding --audit    --source swe_chat
    python -m prompt_anonymity.defenses.collision_seeding --manifest --source swe_chat
"""

from __future__ import annotations

import hashlib
import random
import re
import unicodedata
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..caching import IndexedRowCache, params_hash
from ..core import AttackData
from .base import CachedDefense

# Separator between a document id and a turn's position, forming a turn's cache id
# (``<doc_id>#<n>``). This is ``prompt_anonymity.data.apply_defenses.TURN_ID_SEPARATOR``, redefined
# here rather than imported: `data` imports `defenses`, so importing it back would be circular.
# `--selftest` asserts the two still agree.
TURN_ID_SEPARATOR = "#"

#: Probability that a first-turn-only marker (an opener or a closer) also fires on one of a
#: document's *other* turns. Openers and closers belong at the edges of a conversation -- prepending
#: "quick question --" to all 14 turns of a session would read as tampering, not habit -- but a
#: person who opens with "hey," does sometimes do it twice in a session. Applied per turn.
EDGE_MARKER_REPEAT_PROB = 0.35

#: Share of *letters* in a turn that must be Latin-script before universal (non-trigger-dependent)
#: markers are allowed to fire. WildChat is multilingual, and English quirks pasted onto Chinese or
#: Russian text would be a glaring artifact rather than camouflage. Trigger-dependent markers
#: self-gate: English trigger words simply do not occur in non-English text.
DEFAULT_MIN_LATIN_RATIO = 0.6


# --- deterministic randomness ------------------------------------------------

def _rng(*parts) -> random.Random:
    """A ``random.Random`` seeded by the keyed hash of ``parts``.

    Every stochastic decision in this defense goes through here, which is what makes the whole
    thing a pure function of ``(seed, author_id, marker_key, doc_id)``: shard layout cannot change
    an outcome, a re-run is bit-identical, and the assignment can be reconstructed offline. BLAKE2b
    rather than :func:`hash` because Python's string hashing is salted per process.
    """
    digest = hashlib.blake2b("\x00".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "big"))


# --- protected spans ---------------------------------------------------------

#: Spans a marker must never touch. Corrupting these is the main utility risk, and it is
#: concentrated in SWE-chat, whose prompts are largely code and terminal output. Note that the build
#: stage has already replaced URLs/emails/paths/hashes with ``<URL>``/``<PATH>``/``<ID>``-style
#: placeholders (see :func:`~prompt_anonymity.data.text_cleaning.scrub_identifiers`), so those
#: placeholders are what mostly needs protecting -- lowercasing ``<URL>`` to ``<url>`` would corrupt
#: a sentinel the rest of the pipeline matches on. The residual URL and path patterns stay as a
#: belt-and-braces guard for text that reached here unscrubbed.
_PROTECTED_RE = re.compile(
    r"```.*?```"           # fenced code block (DOTALL: spans lines)
    r"|~~~.*?~~~"          # alternate fence
    r"|`[^`\n]*`"          # inline code span
    r"|<[A-Z][A-Z_]*>"     # build-stage placeholders: <URL> <EMAIL> <PATH> <ID> <REPO> <USER> ...
    r"|https?://\S+"       # residual URL
    r"|\]\([^)\s]*\)"      # markdown link target
    r"|\S{25,}",           # long unbroken token: identifier, hash, base64 blob
    re.DOTALL,
)


def split_protected(text: str) -> list[tuple[str, bool]]:
    """Cut ``text`` into ``(chunk, is_free)`` segments; markers only ever rewrite free chunks.

    Concatenating the chunks reproduces the input exactly, so protection can never lose or reorder
    text -- only decline to touch it.
    """
    segments: list[tuple[str, bool]] = []
    last = 0
    for match in _PROTECTED_RE.finditer(text):
        if match.start() > last:
            segments.append((text[last:match.start()], True))
        segments.append((match.group(0), False))
        last = match.end()
    if last < len(text):
        segments.append((text[last:], True))
    return segments


def on_free_text(text: str, transform: Callable[[str], str]) -> str:
    """Apply ``transform`` to every free segment of ``text``, leaving protected spans byte-identical."""
    return "".join(chunk if not free else transform(chunk) for chunk, free in split_protected(text))


def latin_ratio(text: str) -> float:
    """Share of this text's *letters* that are Latin-script; 1.0 when it contains no letters.

    Counting letters (rather than characters) keeps punctuation, digits and whitespace from diluting
    the measure, so a mostly-code English turn still reads as Latin. Returning 1.0 for letterless
    text means a turn of pure punctuation is not gratuitously excluded from universal markers.
    """
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 1.0
    latin = sum(1 for c in letters if "LATIN" in unicodedata.name(c, ""))
    return latin / len(letters)


# --- marker representation ---------------------------------------------------

@dataclass(frozen=True)
class Marker:
    """One writing quirk.

    Attributes
    ----------
    key : str
        Stable identifier. It is part of the RNG seed for rate and coin draws, so **renaming a key
        silently reassigns every author's rate for it**. Treat keys as append-only.
    pool : str
        Which class-pool this marker is drawn from when profiles are built. Mutually exclusive
        markers (``ellipsis_2dot``/``ellipsis_4dot``, ``bullet_star``/``bullet_dash``,
        ``dialect_uk``/``dialect_us``) share a pool, so a profile can never contain both -- pool
        membership *is* the conflict rule (see :func:`build_profiles`).
    universal : bool
        True when the marker fires on essentially any English text. Every profile is built to hold
        at least one, so no author ends up undefended just because their documents happen to contain
        none of the trigger words.
    whole_text : bool
        True for openers/closers, which apply to the whole turn (prepend/append). Everything else is
        applied per free segment, so it cannot reach inside a code block.
    edge : str
        ``"first"`` for openers, ``"last"`` for closers, ``""`` otherwise. An edge marker fires on
        its edge turn of the document, and elsewhere only with
        :data:`EDGE_MARKER_REPEAT_PROB`.
    apply : callable
        ``(text, rng) -> text``. **Must return the input unchanged when no trigger is present** --
        that is how the caller distinguishes an *eligible* document from an *applied* one.
    detect : str
        Regex matching this marker's signature in text, used by ``--audit`` to measure the quirk's
        natural base rate in the undefended corpus. Empty means "not auditable".
    """

    key: str
    pool: str
    universal: bool
    apply: Callable[[str, random.Random], str]
    detect: str = ""
    whole_text: bool = False
    edge: str = ""

    #: Application order within a document. Substitutions run before layout, and openers/closers run
    #: last, so a closer is not itself typo'd by a substitution marker in the same profile.
    @property
    def stage(self) -> int:
        return 2 if self.edge else (1 if self.pool == "layout" else 0)


def _match_case(source: str, target: str) -> str:
    """Give ``target`` the capitalization of ``source`` (``Definitely`` -> ``Definately``)."""
    if len(source) > 1 and source.isupper():
        return target.upper()
    if source[:1].isupper():
        return target[:1].upper() + target[1:]
    return target


def word_swap(trigger: str, replacement: str, *, key: str, pool: str = "misspelling") -> Marker:
    """A case-preserving, word-boundary-anchored substitution (``definitely`` -> ``definately``).

    Trigger-dependent by construction: text without the trigger comes back unchanged, which is also
    the "not eligible" signal the caller reads.
    """
    pattern = re.compile(rf"(?<!\w){re.escape(trigger)}(?!\w)", re.IGNORECASE)
    detect = rf"(?<!\w){re.escape(replacement)}(?!\w)"

    def apply(text: str, rng: random.Random) -> str:
        return pattern.sub(lambda m: _match_case(m.group(0), replacement), text)

    return Marker(key=key, pool=pool, universal=False, apply=apply, detect=detect)


def word_swap_group(pairs: tuple[tuple[str, str], ...], *, key: str, pool: str) -> Marker:
    """One marker covering a *family* of substitutions, applied together.

    Used for dialect: a British speller writes ``colour`` *and* ``organise`` *and* ``behaviour``.
    Splitting those into separate markers would let one author draw ``colour`` while writing
    ``organize``, which is not how a dialect habit looks.
    """
    patterns = [
        (re.compile(rf"(?<!\w){re.escape(trigger)}(?!\w)", re.IGNORECASE), replacement)
        for trigger, replacement in pairs
    ]
    detect = "|".join(rf"(?<!\w){re.escape(r)}(?!\w)" for _, r in pairs)

    def apply(text: str, rng: random.Random) -> str:
        for pattern, replacement in patterns:
            text = pattern.sub(lambda m, r=replacement: _match_case(m.group(0), r), text)
        return text

    return Marker(key=key, pool=pool, universal=False, apply=apply, detect=detect)


def _edge_is_free(text: str, edge: str) -> bool:
    """Whether the turn's first (or last) segment is ordinary prose rather than a protected span.

    Openers and closers are the only markers applied to the whole turn rather than per free segment,
    so they are the only ones that could land *against* a protected span. Prepending
    ``"quick question -- "`` to a turn that opens with a fenced code block would push the fence off
    the line start and stop it rendering as code -- a formatting corruption, not a writing habit. So
    an edge marker declines a turn whose relevant edge is not free prose.
    """
    segments = split_protected(text)
    if not segments:
        return False
    chunk, free = segments[0] if edge == "first" else segments[-1]
    return free and bool(chunk.strip())


def opener(phrase: str, *, key: str) -> Marker:
    """A question lead prepended to a turn (``quick question -- ``)."""
    def apply(text: str, rng: random.Random) -> str:
        stripped = text.lstrip()
        if not stripped or not _edge_is_free(text, "first"):
            return text
        lead = text[: len(text) - len(stripped)]
        return f"{lead}{phrase}{stripped}"

    return Marker(key=key, pool="edge", universal=True, apply=apply,
                  detect=re.escape(phrase.strip()), whole_text=True, edge="first")


def closer(phrase: str, *, key: str) -> Marker:
    """A sign-off appended to a turn (`` thanks!``)."""
    def apply(text: str, rng: random.Random) -> str:
        stripped = text.rstrip()
        if not stripped or not _edge_is_free(text, "last"):
            return text
        return f"{stripped}{phrase}{text[len(stripped):]}"

    return Marker(key=key, pool="edge", universal=True, apply=apply,
                  detect=re.escape(phrase.strip()), whole_text=True, edge="last")


_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
#: Same boundary, but *capturing* the separator so a rejoin restores the original whitespace
#: verbatim. Splitting on the non-capturing form and rejoining on " " would silently flatten every
#: paragraph break in the turn -- a far bigger style edit than the marker itself.
_SENTENCE_KEEP_RE = re.compile(r"((?<=[.!?])\s+)")


def _decapitalize(text: str) -> str:
    """Lowercase the first word when a hedge has been pushed in front of it.

    Someone who writes "honestly, the build is broken" does not then capitalize "The" -- leaving it
    capitalized is the giveaway that something was inserted mechanically. Applied only to an
    ordinary Capitalized word: ``I`` and all-caps tokens (``API``, ``SQL``) keep their case, since
    lowercasing those would be a different and much more visible edit.
    """
    head = text.split(" ", 1)[0].rstrip(".,;:!?")
    if not head or head == "I" or not head[:1].isupper() or head[1:] != head[1:].lower():
        return text
    return text[:1].lower() + text[1:]


def hedge(word: str, *, key: str) -> Marker:
    """A discourse filler inserted at one sentence boundary per turn (``honestly``, ``tbh``).

    One insertion per turn, at a position drawn from ``rng`` -- inserting into every sentence would
    be a caricature rather than a habit. Only sentences long enough to carry a hedge are eligible,
    so it never lands on a one-word line.
    """
    detect = rf"(?<!\w){re.escape(word)}(?!\w)"

    def apply(text: str, rng: random.Random) -> str:
        # Alternating [sentence, separator, sentence, ...]; even indices are sentences.
        parts = _SENTENCE_KEEP_RE.split(text)
        candidates = [i for i in range(0, len(parts), 2) if len(parts[i].strip()) > 20]
        if not candidates:
            return text
        target = candidates[rng.randrange(len(candidates))]
        sentence = parts[target]
        lead = sentence[: len(sentence) - len(sentence.lstrip())]
        parts[target] = f"{lead}{word}, {_decapitalize(sentence.lstrip())}"
        return "".join(parts)

    return Marker(key=key, pool="hedge", universal=True, apply=apply, detect=detect)


def _regex_marker(pattern: str, replacement, *, key: str, pool: str, detect: str,
                  whole_text: bool = False) -> Marker:
    """A punctuation/layout habit expressed directly as a regex substitution.

    ``whole_text`` is for habits anchored to the *end of the turn* (see ``no_terminal_period``):
    applied per free segment, an end-anchored pattern would fire at the end of every segment, which
    is mid-sentence wherever a code span interrupts the text.
    """
    compiled = re.compile(pattern)

    def apply(text: str, rng: random.Random) -> str:
        return compiled.sub(replacement, text)

    return Marker(key=key, pool=pool, universal=True, apply=apply, detect=detect,
                  whole_text=whole_text)


# --- the marker inventory ----------------------------------------------------

_CONTRACTIONS = [
    "don't", "can't", "won't", "I'm", "I've", "I'd", "I'll", "you're", "you've", "that's",
    "isn't", "doesn't", "didn't", "wasn't", "aren't", "it's", "there's", "haven't", "hasn't",
    "wouldn't", "couldn't", "shouldn't", "let's", "they're", "we're", "he's", "she's", "what's",
]
_NO_APOSTROPHE_RE = re.compile(
    "|".join(rf"(?<!\w){re.escape(c)}(?!\w)" for c in _CONTRACTIONS), re.IGNORECASE
)
_EXPANSIONS = {
    "don't": "do not", "can't": "cannot", "won't": "will not", "isn't": "is not",
    "doesn't": "does not", "didn't": "did not", "wasn't": "was not", "aren't": "are not",
    "haven't": "have not", "hasn't": "has not", "wouldn't": "would not",
    "couldn't": "could not", "shouldn't": "should not", "I'm": "I am", "I've": "I have",
    "you're": "you are", "you've": "you have", "they're": "they are", "we're": "we are",
}
_EXPANSION_RE = re.compile(
    "|".join(rf"(?<!\w){re.escape(c)}(?!\w)" for c in _EXPANSIONS), re.IGNORECASE
)


def _drop_apostrophes(text: str, rng: random.Random) -> str:
    return _NO_APOSTROPHE_RE.sub(lambda m: m.group(0).replace("'", "").replace("’", ""), text)


def _expand_contractions(text: str, rng: random.Random) -> str:
    lookup = {k.lower(): v for k, v in _EXPANSIONS.items()}
    return _EXPANSION_RE.sub(lambda m: _match_case(m.group(0), lookup[m.group(0).lower()]), text)


def _caps_emphasis(text: str, rng: random.Random) -> str:
    """Shout exactly one word per turn -- the ``i need this to be FAST`` habit."""
    words = [m for m in re.finditer(r"(?<!\w)[a-z]{5,}(?!\w)", text)]
    if not words:
        return text
    chosen = words[rng.randrange(len(words))]
    return text[: chosen.start()] + chosen.group(0).upper() + text[chosen.end():]


def _blank_line_between_sentences(text: str, rng: random.Random) -> str:
    return _SENTENCE_SPLIT_RE.sub("\n\n", text)


def _line_break_between_sentences(text: str, rng: random.Random) -> str:
    return _SENTENCE_SPLIT_RE.sub("\n", text)


def _all_lowercase(text: str, rng: random.Random) -> str:
    return text.lower()


#: British spellings, as one bundled habit. Deliberately a single marker: see :func:`word_swap_group`.
_UK_PAIRS = (
    ("color", "colour"), ("colors", "colours"), ("organize", "organise"),
    ("organized", "organised"), ("realize", "realise"), ("realized", "realised"),
    ("behavior", "behaviour"), ("center", "centre"), ("analyze", "analyse"),
    ("analyzed", "analysed"), ("license", "licence"), ("defense", "defence"),
    ("traveling", "travelling"), ("favorite", "favourite"), ("gray", "grey"),
    ("optimize", "optimise"), ("optimized", "optimised"), ("initialize", "initialise"),
)
_US_PAIRS = tuple((uk, us) for us, uk in _UK_PAIRS)


def _build_markers() -> dict[str, Marker]:
    """The full inventory. ``--audit`` cuts it down to what actually has a base rate on a corpus."""
    markers: list[Marker] = []

    # A. Lexical misspellings -- trigger-dependent.
    for trigger, replacement in [
        ("definitely", "definately"), ("separate", "seperate"), ("receive", "recieve"),
        ("occurred", "occured"), ("necessary", "neccessary"), ("until", "untill"),
        ("a lot", "alot"), ("weird", "wierd"), ("believe", "beleive"),
        ("calendar", "calender"), ("recommend", "reccomend"), ("accommodate", "accomodate"),
        ("beginning", "begining"), ("successful", "succesful"), ("environment", "enviroment"),
        ("length", "lenght"), ("publicly", "publically"), ("consistent", "consistant"),
        ("existence", "existance"), ("maintenance", "maintainance"), ("occasion", "occassion"),
        ("privilege", "priviledge"), ("rhythm", "rythm"), ("tomorrow", "tommorow"),
        ("truly", "truely"), ("immediately", "immediatly"), ("queue", "que"),
        ("argument", "arguement"), ("dependent", "dependant"), ("referring", "refering"),
    ]:
        markers.append(word_swap(trigger, replacement, key=f"ms_{replacement}"))

    # B. Transposition typos -- trigger-dependent, low natural rate, expect the audit to thin these.
    # "from"->"form" and "for"->"fro" are deliberately excluded: both produce a different real word,
    # so they change meaning rather than reading as a slip.
    for trigger, replacement in [
        ("the", "teh"), ("and", "adn"), ("that", "taht"), ("with", "wiht"),
        ("this", "tihs"), ("just", "jsut"), ("what", "waht"), ("because", "becuase"),
    ]:
        markers.append(word_swap(trigger, replacement, key=f"tp_{replacement}",
                                 pool="transposition"))

    # C. Apostrophe & contraction habits -- universal.
    markers += [
        Marker(key="no_apostrophe", pool="contraction", universal=True, apply=_drop_apostrophes,
               detect=r"(?<!\w)(dont|cant|wont|im|ive|youre|thats|isnt|doesnt|didnt)(?!\w)"),
        Marker(key="expand_contractions", pool="contraction", universal=True,
               apply=_expand_contractions, detect=r"(?<!\w)(do not|cannot|will not|is not)(?!\w)"),
    ]
    for trigger, replacement, key in [
        ("going to", "gonna", "gonna"), ("want to", "wanna", "wanna"),
        ("kind of", "kinda", "kinda"), ("got to", "gotta", "gotta"),
        ("because", "cuz", "cuz"), ("because", "b/c", "bc"),
    ]:
        markers.append(word_swap(trigger, replacement, key=f"cn_{key}", pool="contraction"))

    # D. Punctuation habits -- universal.
    markers += [
        _regex_marker(r"\.\.\.", "..", key="ellipsis_2dot", pool="punctuation",
                      detect=r"(?<!\.)\.\.(?!\.)"),
        _regex_marker(r"\.\.\.", "....", key="ellipsis_4dot", pool="punctuation",
                      detect=r"(?<!\.)\.\.\.\.(?!\.)"),
        _regex_marker(r"(?<=[a-zA-Z0-9])!(?!!)", "!!", key="double_bang", pool="punctuation",
                      detect=r"!!"),
        _regex_marker(r"[ \t]*([?!])", r" \1", key="space_before_punct", pool="punctuation",
                      detect=r"\S[ ]+[?!]"),
        _regex_marker(r",[ \t]+", ",", key="no_space_after_comma", pool="punctuation",
                      detect=r"\w,\w"),
        _regex_marker(r"\s+[—–]\s+|(?<=\s)-(?=\s)", " -- ", key="dash_double",
                      pool="punctuation", detect=r"\s--\s"),
        # Anchored to the very end of the turn, hence whole_text: applied per free segment this
        # would strip the period wherever a code span happens to follow it, mid-sentence.
        _regex_marker(r"\.[ \t]*\Z", "", key="no_terminal_period", pool="punctuation",
                      detect=r"(?s)[a-z][ \t]*\Z", whole_text=True),
    ]

    # E. Capitalization habits -- universal. All in one pool: they are mutually exclusive habits.
    markers += [
        _regex_marker(r"(?<!\w)I(?!\w)", "i", key="lowercase_i", pool="capitalization",
                      detect=r"(?<!\w)i(?!\w)"),
        Marker(key="all_lowercase", pool="capitalization", universal=True, apply=_all_lowercase,
               detect=r"(?s)\A[^A-Z]*\Z"),
        Marker(key="caps_emphasis", pool="capitalization", universal=True, apply=_caps_emphasis,
               detect=r"(?<!\w)[A-Z]{5,}(?!\w)"),
        _regex_marker(r"(?<!\w)(API|SQL|JSON|HTTP|CSS|HTML)(?!\w)", lambda m: m.group(0).lower(),
                      key="lower_acronyms", pool="capitalization",
                      detect=r"(?<!\w)(api|sql|json|http|css|html)(?!\w)"),
    ]

    # F. Layout quirks -- universal. Expect `blank_line_sentences` to fail the base-rate audit on
    # this corpus: normalize_whitespace collapses 3+ newlines, so the pattern is rare by construction.
    markers += [
        _regex_marker(r"(?m)^(\s*)-(?=\s)", r"\1*", key="bullet_star", pool="layout",
                      detect=r"(?m)^\s*\*\s"),
        _regex_marker(r"(?m)^(\s*)\*(?=\s)", r"\1-", key="bullet_dash", pool="layout",
                      detect=r"(?m)^\s*-\s"),
        _regex_marker(r"(?m)^(\s*\d+)\.(?=\s)", r"\1)", key="numbered_paren", pool="layout",
                      detect=r"(?m)^\s*\d+\)\s"),
        Marker(key="blank_line_sentences", pool="layout", universal=True,
               apply=_blank_line_between_sentences, detect=r"[.!?]\n\n"),
        Marker(key="line_break_sentences", pool="layout", universal=True,
               apply=_line_break_between_sentences, detect=r"[.!?]\n(?!\n)"),
        _regex_marker(r"->", "=>", key="arrow_fat", pool="layout", detect=r"=>"),
    ]

    # G/H. Openers and closers -- universal, edge-anchored.
    for phrase, key in [
        ("quick question -- ", "quick_question"), ("quick q: ", "quick_q"), ("hey, ", "hey"),
        ("ok so ", "ok_so"), ("just wondering, ", "just_wondering"),
        ("so basically, ", "so_basically"), ("context: ", "context_label"),
        ("small ask: ", "small_ask"),
    ]:
        markers.append(opener(phrase, key=f"op_{key}"))
    for phrase, key in [
        (" thanks!", "thanks"), (" thanks in advance", "thanks_advance"), (" thx", "thx"),
        (" appreciate it", "appreciate"), (" pls", "pls"), (" let me know", "lmk"),
        (" does that make sense?", "make_sense"), (" no rush", "no_rush"),
    ]:
        markers.append(closer(phrase, key=f"cl_{key}"))

    # I. Hedges & discourse fillers -- universal, one insertion per turn.
    for word in ["honestly", "basically", "tbh", "imo", "fwiw", "to be fair", "actually"]:
        markers.append(hedge(word, key=f"hg_{word.replace(' ', '_')}"))

    # J. Abbreviation & register -- trigger-dependent.
    for trigger, replacement, key in [
        ("you", "u", "u"), ("your", "ur", "ur"), ("please", "pls", "pls"),
        ("with", "w/", "with"), ("without", "w/o", "without"), ("probably", "prolly", "prolly"),
        ("though", "tho", "tho"), ("through", "thru", "thru"), ("thanks", "thx", "thanks"),
        ("something", "smth", "smth"),
    ]:
        markers.append(word_swap(trigger, replacement, key=f"ab_{key}", pool="abbreviation"))

    # K. Dialect -- trigger-dependent, one bundled habit per direction, same pool so a profile can
    # never hold both.
    markers += [
        word_swap_group(_UK_PAIRS, key="dialect_uk", pool="dialect"),
        word_swap_group(_US_PAIRS, key="dialect_us", pool="dialect"),
    ]

    return {marker.key: marker for marker in markers}


MARKERS: dict[str, Marker] = _build_markers()

#: Pools a profile draws from, one marker each. Grouping mutually exclusive habits into one pool is
#: what prevents a profile from containing e.g. both ellipsis styles. The first four are drawn
#: always; ``dialect`` is drawn only sometimes (see :data:`DIALECT_PROBABILITY`) because a dialect
#: flip is the most visible -- and the most contradiction-prone -- marker in the inventory.
PROFILE_POOLS: tuple[tuple[str, ...], ...] = (
    ("misspelling", "transposition", "abbreviation"),   # a trigger-dependent lexical habit
    ("punctuation",),                                   # a punctuation habit
    ("capitalization", "layout"),                       # a casing or layout habit
    ("edge", "hedge", "contraction"),                   # a discourse habit
)

#: Chance a profile also carries a dialect habit. Kept below 1 so dialect is not a universal tell,
#: and capped at one per profile by pool structure.
DIALECT_PROBABILITY = 0.4

#: The markers that survived ``--audit`` on **SWE-chat** (4,334 documents / 157 authors), i.e. those
#: whose quirk occurs naturally but not universally (base rate in ``(0, 0.25]``) and whose trigger
#: appears in at least 5% of documents. 47 of the inventory's 98.
#:
#: **This set is corpus-specific and must not be reused for WildChat.** SWE-chat prose is short and
#: technical, so every one of the 30 lexical misspellings failed for lack of coverage -- words like
#: "definitely", "separate" and "environment" barely appear -- leaving the lexical slot to
#: transposition typos and abbreviations. WildChat's longer prose should revive that class and drop
#: others, so re-run the audit and add a ``WILDCHAT_MARKERS`` beside this one::
#:
#:     python -m prompt_anonymity.defenses.collision_seeding --audit --source wildchat
#:
#: Until then the registry's collision-seeding entries are wired to *this* set, which is the right
#: default only while SWE-chat is the corpus under study.
SWE_CHAT_MARKERS: tuple[str, ...] = (
    'ab_pls', 'ab_smth', 'ab_thru', 'ab_u', 'ab_ur', 'ab_with', 'ab_without', 'all_lowercase',
    'blank_line_sentences', 'bullet_star', 'caps_emphasis', 'cl_lmk', 'cl_make_sense', 'cl_pls',
    'cl_thanks', 'cl_thx', 'cn_cuz', 'cn_wanna', 'dash_double', 'dialect_uk', 'double_bang',
    'ellipsis_2dot', 'ellipsis_4dot', 'expand_contractions', 'hg_actually', 'hg_basically',
    'hg_honestly', 'hg_imo', 'hg_tbh', 'hg_to_be_fair', 'lower_acronyms', 'lowercase_i',
    'no_apostrophe', 'no_space_after_comma', 'numbered_paren', 'op_context_label', 'op_hey',
    'op_ok_so', 'space_before_punct', 'tp_adn', 'tp_becuase', 'tp_jsut', 'tp_taht', 'tp_teh',
    'tp_tihs', 'tp_waht', 'tp_wiht',
)

#: Number of profiles when ``independent=True`` is *not* used. The privacy knob: expected collision
#: group is ``n_authors / n_profiles``.
#:
#: Note this is a *large* K for a small corpus. On SWE-chat's 157 authors it makes groups of ~13 and
#: hands an attacker log2(12) = 3.58 bits of the 7.29 that identify an author -- and collision
#: seeding is purely additive, so it never removes the natural style they would use to separate the
#: 13. Whether that trade pays off is what the K sweep measures (``_k4`` gives 2.00 bits and groups
#: of ~39); do not assume the default is on the right side of it.
DEFAULT_N_PROFILES = 12

#: Expected markers per author in the ``independent=True`` ablation, matched to the codebook's
#: typical profile size so the two differ only in *structure*, not in dose.
INDEPENDENT_MARKERS_PER_AUTHOR = 4.4


def build_profiles(marker_keys: tuple[str, ...], *, n_profiles: int, seed) -> tuple[tuple[str, ...], ...]:
    """Construct the profile codebook: ``n_profiles`` marker bundles, deterministically.

    Built rather than hardcoded so that cutting markers after an ``--audit`` reshapes the codebook
    automatically instead of silently leaving dangling keys. One marker is drawn per entry of
    :data:`PROFILE_POOLS`, plus a dialect marker with probability :data:`DIALECT_PROBABILITY`, so
    every profile mixes classes -- a plausible *person*, not a list of tics -- and holds at least
    three universal markers, guaranteeing coverage for an author whose documents contain none of the
    trigger words.

    Profiles may share individual markers; that is fine, and mildly helpful, because it blurs the
    boundary between groups. What must not happen is two *identical* profiles, which would silently
    halve ``K``; that is checked and raised.
    """
    by_pool: dict[str, list[str]] = {}
    for key in marker_keys:
        by_pool.setdefault(MARKERS[key].pool, []).append(key)

    profiles: list[tuple[str, ...]] = []
    for index in range(n_profiles):
        rng = _rng(seed, "profile_def", index)
        chosen: list[str] = []
        for pools in PROFILE_POOLS:
            candidates = sorted({k for pool in pools for k in by_pool.get(pool, [])})
            if candidates:
                chosen.append(candidates[rng.randrange(len(candidates))])
        dialect = sorted(by_pool.get("dialect", []))
        if dialect and rng.random() < DIALECT_PROBABILITY:
            chosen.append(dialect[rng.randrange(len(dialect))])
        if not chosen:
            raise ValueError("no markers survived filtering; cannot build a profile codebook.")
        profiles.append(tuple(sorted(set(chosen))))

    if len(set(profiles)) != len(profiles):
        raise ValueError(
            f"profile codebook has duplicates ({len(set(profiles))} distinct of {n_profiles}); "
            "the marker pool is too small for this many profiles -- lower n_profiles or keep more "
            "markers."
        )
    return tuple(profiles)


# --- the defense -------------------------------------------------------------

@dataclass
class _Coverage:
    """Per-marker tally, printed at the end of a run: is this marker actually firing?

    Units are stated in the field names because they differ: assignment is per *document* (the coin
    is flipped per document), while scheduling and firing are per *turn*.
    """
    docs_assigned: int = 0      # documents whose author carries this marker
    docs_scheduled: int = 0     # ... and whose per-document coin came up heads
    turns_scheduled: int = 0    # turns of those documents where the marker was allowed to act
    turns_changed: int = 0      # ... and where the text actually changed (a trigger was present)


class CollisionSeedingDefense(CachedDefense):
    """Seed shared, inconsistently-applied writing quirks across groups of unrelated authors.

    Parameters
    ----------
    seed : int
        Master seed. Every draw derives from it, so changing it reshuffles the entire assignment.
    n_profiles : int
        Size of the profile codebook -- the privacy knob. Expected collision group is
        ``n_authors / n_profiles``. Ignored when ``independent=True``.
    rate_min, rate_max : float
        Bounds of the per-(author, marker) application rate, drawn uniformly. The default
        ``0.4-0.7`` keeps every author visibly inconsistent while leaving the rate itself
        unidentifiable at ~10 documents per author. Set both to 1.0 for the "perfectly consistent"
        ablation.
    independent : bool
        **Ablation.** Draw markers per author independently instead of from the codebook, producing
        a near-unique fingerprint per author. Expected to perform *worse than no defense*; included
        because demonstrating that is the point.
    marker_keys : tuple of str, optional
        Restrict the inventory (e.g. to what survived ``--audit``). Defaults to everything.
    min_latin_ratio : float
        Latin-script share below which universal markers are suppressed for a turn.
    """

    name = "collision_seeding"
    version = "1"

    def __init__(self, *, seed: int = 0, n_profiles: int = DEFAULT_N_PROFILES,
                 rate_min: float = 0.4, rate_max: float = 0.7, independent: bool = False,
                 marker_keys: tuple[str, ...] | None = None,
                 min_latin_ratio: float = DEFAULT_MIN_LATIN_RATIO):
        if not 0.0 <= rate_min <= rate_max <= 1.0:
            raise ValueError(f"need 0 <= rate_min <= rate_max <= 1 (got {rate_min}, {rate_max}).")
        self.seed = seed
        self.n_profiles = n_profiles
        self.rate_min = rate_min
        self.rate_max = rate_max
        self.independent = independent
        self.min_latin_ratio = min_latin_ratio
        self.marker_keys = tuple(marker_keys) if marker_keys else tuple(sorted(MARKERS))
        unknown = set(self.marker_keys) - set(MARKERS)
        if unknown:
            raise ValueError(f"unknown marker keys: {sorted(unknown)}")
        self.profiles = (
            () if independent
            else build_profiles(self.marker_keys, n_profiles=n_profiles, seed=seed)
        )

    def params(self) -> dict:
        # The marker set is hashed rather than listed: it is ~90 keys, and what matters downstream
        # is only that a different set separates results directories and provenance.
        return {
            "seed": self.seed,
            "n_profiles": self.n_profiles,
            "rate_min": self.rate_min,
            "rate_max": self.rate_max,
            "independent": self.independent,
            "min_latin_ratio": self.min_latin_ratio,
            "markers": params_hash({"keys": list(self.marker_keys)}),
        }

    # --- assignment ---------------------------------------------------------

    def profile_index(self, author_id: str) -> int:
        """Which profile this author carries. Pure function of ``(seed, author_id)``."""
        if self.independent:
            raise ValueError("independent=True has no profile index; use author_markers().")
        return _rng(self.seed, "profile", author_id).randrange(len(self.profiles))

    def author_markers(self, author_id: str) -> tuple[str, ...]:
        """The marker keys this author carries."""
        if not self.independent:
            return self.profiles[self.profile_index(author_id)]
        # Ablation: independent Bernoulli per marker, tuned to the codebook's typical dose so the
        # two arms differ in structure rather than in how much text is touched.
        probability = min(1.0, INDEPENDENT_MARKERS_PER_AUTHOR / max(1, len(self.marker_keys)))
        return tuple(
            key for key in self.marker_keys
            if _rng(self.seed, "indep", author_id, key).random() < probability
        )

    def marker_rate(self, author_id: str, marker_key: str) -> float:
        """This author's application rate for this marker, drawn once and stable forever."""
        if self.rate_min == self.rate_max:
            return self.rate_min
        return _rng(self.seed, "rate", author_id, marker_key).uniform(self.rate_min, self.rate_max)

    def _applies_to_document(self, author_id: str, marker_key: str, doc_id: str) -> bool:
        rate = self.marker_rate(author_id, marker_key)
        return _rng(self.seed, "doc", author_id, marker_key, doc_id).random() < rate

    # --- rewriting ----------------------------------------------------------

    def _apply_marker(self, marker: Marker, text: str, rng: random.Random) -> str:
        if marker.whole_text:
            return marker.apply(text, rng)
        return on_free_text(text, lambda chunk: marker.apply(chunk, rng))

    def rewrite_turn(self, text: str, author_id: str, doc_id: str, turn_index: int,
                     last_turn_index: int, coverage: dict[str, _Coverage] | None = None) -> str:
        """Apply this author's active markers to one turn."""
        if not text.strip():
            return text
        allow_universal = latin_ratio(text) >= self.min_latin_ratio
        markers = [MARKERS[key] for key in self.author_markers(author_id)]
        for marker in sorted(markers, key=lambda m: (m.stage, m.key)):
            if marker.universal and not allow_universal:
                continue
            tally = coverage.setdefault(marker.key, _Coverage()) if coverage is not None else None
            if not self._applies_to_document(author_id, marker.key, doc_id):
                continue
            # An edge marker is scheduled for the document but acts only on its edge turn (plus the
            # occasional repeat), so counting it as scheduled on every turn would make its hit rate
            # look artificially poor.
            if marker.edge and not self._fires_on_turn(marker, author_id, doc_id, turn_index,
                                                       last_turn_index):
                continue
            if tally is not None:
                tally.turns_scheduled += 1
            rng = _rng(self.seed, "turn", author_id, marker.key, doc_id, turn_index)
            rewritten = self._apply_marker(marker, text, rng)
            if rewritten != text and tally is not None:
                tally.turns_changed += 1
            text = rewritten
        return text

    def _fires_on_turn(self, marker: Marker, author_id: str, doc_id: str, turn_index: int,
                       last_turn_index: int) -> bool:
        """Edge markers belong at the edges: an opener on turn 0, a closer on the last turn.

        Elsewhere they fire only with :data:`EDGE_MARKER_REPEAT_PROB` -- prepending "quick question"
        to all 14 turns of a session would read as tampering, but a person who opens with "hey,"
        does sometimes do it twice.
        """
        at_edge = turn_index == 0 if marker.edge == "first" else turn_index == last_turn_index
        if at_edge:
            return True
        return _rng(self.seed, "edge", author_id, marker.key, doc_id, turn_index).random() \
            < EDGE_MARKER_REPEAT_PROB

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        """Rewrite every unknown-side turn. ``cache`` is deliberately unused -- see the module docstring.

        ``apply_defenses`` hands the whole corpus in on the unknown side, one row per turn, with
        ``unknown_labels`` carrying ``author_id`` and ``unknown_ids`` carrying ``<doc_id>#<n>``.

        **Invariant: a document must arrive whole.** Everything else here keys on ``author_id`` and
        ``doc_id`` alone, but a closer fires on a document's *last* turn, which can only be derived
        from the batch. ``select_shard`` slices the document frame (``documents.iloc[i::n]``), so
        every turn of a document always travels together and the derived last index is stable across
        any shard layout. The ``--selftest`` shard-invariance check is what would catch a future
        change to turn-level sharding.
        """
        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts; load the dataset with text.")
        if data.unknown_ids is None:
            raise ValueError(
                f"defense {self.name!r} needs unknown_ids to recover each turn's document; "
                "apply_defenses supplies them as '<doc_id>#<n>'."
            )
        texts = [str(t) for t in data.unknown_texts]
        authors = [str(a) for a in data.unknown_labels]
        ids = [str(i) for i in data.unknown_ids]

        parsed = [_parse_turn_id(turn_id) for turn_id in ids]
        last_index: dict[str, int] = {}
        for doc_id, turn_index in parsed:
            last_index[doc_id] = max(last_index.get(doc_id, 0), turn_index)

        coverage: dict[str, _Coverage] = {}
        seen_documents: set[tuple[str, str]] = set()
        out: list[str] = []
        for text, author_id, (doc_id, turn_index) in zip(texts, authors, parsed):
            if (author_id, doc_id) not in seen_documents:
                seen_documents.add((author_id, doc_id))
                for key in self.author_markers(author_id):
                    tally = coverage.setdefault(key, _Coverage())
                    tally.docs_assigned += 1
                    if self._applies_to_document(author_id, key, doc_id):
                        tally.docs_scheduled += 1
            out.append(self.rewrite_turn(text, author_id, doc_id, turn_index,
                                         last_index[doc_id], coverage))

        self._report_coverage(coverage, len(seen_documents), len(texts))
        return replace(data, unknown_texts=np.asarray(out, dtype=object))

    def _report_coverage(self, coverage: dict[str, _Coverage], n_documents: int,
                         n_turns: int) -> None:
        """Print realized coverage per marker: the check that a marker is not silently never firing.

        Two numbers matter. ``rate`` is scheduled documents over assigned documents -- it should sit
        inside ``[rate_min, rate_max]``, and a value outside it means the coin is wrong. ``hit`` is
        changed turns over scheduled turns -- it measures *trigger availability*, so a marker with a
        healthy rate but a near-zero hit is trigger-starved on this corpus and should be cut. That is
        what ``--audit`` predicts ahead of time; this confirms it after the fact.
        """
        profiles = "independent" if self.independent else f"{len(self.profiles)} profiles"
        print(f"[{self.name}] {n_documents:,} documents / {n_turns:,} turns; {profiles}")
        rows = sorted(coverage.items(), key=lambda kv: -kv[1].turns_changed)
        for key, tally in rows:
            if not tally.docs_assigned:
                continue
            rate = tally.docs_scheduled / tally.docs_assigned
            hit = tally.turns_changed / tally.turns_scheduled if tally.turns_scheduled else 0.0
            print(f"  {key:<26} docs={tally.docs_assigned:>7,} rate={rate:.2f}  "
                  f"turns_changed={tally.turns_changed:>7,}/{tally.turns_scheduled:<7,} hit={hit:.0%}")
        silent = [key for key, t in rows if t.turns_scheduled and not t.turns_changed]
        if silent:
            print(f"  [warning] scheduled but never fired (no trigger in this corpus): "
                  f"{', '.join(sorted(silent))}")


def _parse_turn_id(turn_id: str) -> tuple[str, int]:
    """Recover ``(doc_id, turn_index)`` from a ``<doc_id>#<n>`` cache id.

    Falls back to treating the whole id as the document with index 0, so the defense still runs
    (per document = per row) if it is ever handed ids without the separator.
    """
    doc_id, separator, position = turn_id.rpartition(TURN_ID_SEPARATOR)
    if not separator or not position.isdigit():
        return turn_id, 0
    return doc_id, int(position)


# --- offline manifest --------------------------------------------------------

def collision_manifest(author_ids, defense: CollisionSeedingDefense):
    """``DataFrame`` of ``author_id, profile, markers, rates`` -- the assignment, rebuilt offline.

    Because every decision is a pure function of the author id, nothing has to survive the defense
    run: the analysis joins against this. Used by the within-group confusion metric, which asks what
    fraction of *misattributed* documents were assigned to an author sharing the true author's
    profile (chance is ``1/K``).
    """
    import pandas as pd

    rows = []
    for author_id in dict.fromkeys(str(a) for a in author_ids):
        markers = defense.author_markers(author_id)
        rows.append({
            "author_id": author_id,
            "profile": -1 if defense.independent else defense.profile_index(author_id),
            "markers": "|".join(markers),
            "rates": "|".join(f"{defense.marker_rate(author_id, k):.4f}" for k in markers),
        })
    return pd.DataFrame(rows)


# --- corpus audit ------------------------------------------------------------

#: A marker whose quirk never appears naturally has no background to hide in: any author carrying it
#: is instantly separable from everyone who does not, and a human reader would spot it as synthetic.
#: Several layout markers fail this on purpose-built corpora -- ``normalize_whitespace`` collapses
#: runs of 3+ newlines and ``scrub_identifiers`` squeezes runs of 2+ spaces, so those patterns are
#: absent by construction.
MIN_BASE_RATE = 0.0

#: The mirror of :data:`MIN_BASE_RATE`, and just as necessary. A quirk a large share of the corpus
#: *already* has cannot make a group cohesive, because everyone outside the group has it too: it
#: carries no signal to collide on, while still costing naturalness and utility. Measured on
#: SWE-chat, ``no_terminal_period`` (55% of documents already end without one) and
#: ``line_break_sentences`` (52%) are majority behaviour rather than quirks. 0.25 is set to catch
#: those without touching the genuinely-uncommon-but-present band (10-22%) the design wants.
MAX_BASE_RATE = 0.25

#: A marker whose trigger appears in too few documents cannot cover an author even when assigned.
MIN_TRIGGER_RATE = 0.05


def audit_markers(documents: list[str], *, marker_keys=None) -> list[dict]:
    """Measure each marker's natural base rate and trigger coverage on a corpus.

    ``base_rate`` -- share of documents where the quirk *already* occurs, i.e. the background noise a
    seeded marker would blend into. ``trigger_rate`` -- share of documents the marker would actually
    change if applied, i.e. how much coverage it can give an author who carries it.

    Run this before fixing a marker set: the two rates are what
    :func:`surviving_markers` cuts on, and both are corpus-specific.
    """
    keys = tuple(marker_keys) if marker_keys else tuple(sorted(MARKERS))
    total = max(1, len(documents))
    rows = []
    for key in keys:
        marker = MARKERS[key]
        detector = re.compile(marker.detect) if marker.detect else None
        base = sum(1 for d in documents if detector and detector.search(d))
        rng = _rng(0, "audit", key)
        triggered = sum(1 for d in documents if _would_change(marker, d, rng))
        rows.append({
            "marker": key,
            "pool": marker.pool,
            "universal": marker.universal,
            "auditable": detector is not None,
            "base_rate": base / total,
            "trigger_rate": triggered / total,
        })
    return rows


def _would_change(marker: Marker, text: str, rng: random.Random) -> bool:
    """Whether this marker would alter ``text`` at all -- the eligibility test, protection included."""
    if marker.whole_text:
        return marker.apply(text, rng) != text
    return on_free_text(text, lambda chunk: marker.apply(chunk, rng)) != text


def surviving_markers(rows: list[dict], *, min_base_rate: float = MIN_BASE_RATE,
                      max_base_rate: float = MAX_BASE_RATE,
                      min_trigger_rate: float = MIN_TRIGGER_RATE) -> tuple[str, ...]:
    """Marker keys that clear all three audit thresholds, sorted.

    The base rate has to land in a *band*, not just above a floor. Too low (default: seen zero
    times) and the quirk has no background to hide in, so it is a perfect group indicator. Too high
    (default: more than a quarter of documents) and it is the corpus norm rather than a quirk, so it
    cannot distinguish a group from everyone else. The floor is a strict inequality -- a quirk seen
    even once is kept -- while the ceiling is inclusive.
    """
    return tuple(sorted(
        row["marker"] for row in rows
        if min_base_rate < row["base_rate"] <= max_base_rate
        and row["trigger_rate"] >= min_trigger_rate
    ))


def _load_documents(source: str, dist_dir=None, limit=None) -> tuple[list[str], list[str]]:
    """``(document_texts, author_ids)`` from a built split, one string of joined turns per document.

    Imports are local: this module is imported by the defense registry at package import, and
    ``data.config`` would be a circular import at module level (``data`` imports ``defenses``).
    """
    import pyarrow.parquet as pq

    from ..data.config import hf_dir

    path = Path(dist_dir) if dist_dir else hf_dir()
    frame = pq.read_table(path / f"{source}.parquet", columns=["author_id", "turns"]).to_pandas()
    if limit:
        frame = frame.head(limit)
    documents = ["\n".join(str(t) for t in turns) for turns in frame["turns"]]
    return documents, [str(a) for a in frame["author_id"]]


# --- self-test ---------------------------------------------------------------

def _selftest() -> None:
    """The plan's verification checks, as a runnable command.

    This repo has no test framework and no pytest dependency, so the checks live here rather than
    introducing one. Every check is a property the defense's correctness actually rests on.
    """
    from ..data.apply_defenses import TURN_ID_SEPARATOR as PIPELINE_SEPARATOR

    defense = CollisionSeedingDefense(seed=7)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  -- {detail}'}")
        if not condition:
            failures.append(name)

    # 1. The turn-id separator still agrees with the pipeline that produces the ids.
    check("turn id separator matches apply_defenses", TURN_ID_SEPARATOR == PIPELINE_SEPARATOR,
          f"{TURN_ID_SEPARATOR!r} != {PIPELINE_SEPARATOR!r}")

    # 2. Determinism: the same inputs give the same output, every time.
    sample = "I definitely think the environment is weird. Can you check it?"
    first = defense.rewrite_turn(sample, "author-a", "doc-1", 0, 2)
    again = defense.rewrite_turn(sample, "author-a", "doc-1", 0, 2)
    check("deterministic across calls", first == again)

    # 3. Shard invariance: a turn's rewrite must not depend on which other rows shared the batch.
    #    Sharded exactly as the pipeline does -- `select_shard` slices the *document* frame
    #    (`documents.iloc[i::n]`), so every turn of a document always travels together. That is a
    #    real invariant this defense depends on, not an incidental one: a closer fires on a
    #    document's LAST turn, which is only knowable from the whole document. Splitting a document
    #    across shards would change its output, so if `apply_defenses` ever shards by turn, this
    #    check is what should fail.
    n_documents, turns_per_document = 20, 3
    texts, authors, ids, documents = [], [], [], []
    for d in range(n_documents):
        documents.append([])
        for t in range(turns_per_document):
            documents[-1].append(len(texts))
            texts.append(f"I definitely think item {d}.{t} is weird. Can you check the environment?")
            authors.append(f"author-{d % 7}")
            ids.append(f"doc-{d}#{t}")
    whole = _run(defense, texts, authors, ids)
    sharded_rows: list[tuple[int, str]] = []
    for offset in range(4):
        rows = [row for d in range(offset, n_documents, 4) for row in documents[d]]
        sharded_rows.extend(zip(rows, _run(defense, [texts[i] for i in rows],
                                           [authors[i] for i in rows], [ids[i] for i in rows])))
    sharded = [text for _, text in sorted(sharded_rows)]
    check("shard-invariant (documents kept whole)", whole == sharded)

    # 4. One output per input, in order -- regroup_turns raises otherwise.
    check("turn count preserved", len(whole) == len(texts))

    # 5. Protected spans are byte-identical. A turn that is *nothing but* a code block has no free
    #    prose at either edge, so openers/closers decline it too and it survives verbatim.
    code = "```\nif (x) { return DEFINITELY; }\n```"
    protected_ok = all(
        defense.rewrite_turn(code, f"author-{i}", f"doc-{i}", 0, 0) == code for i in range(60)
    )
    check("code-only turn untouched", protected_ok)

    #    With prose around it, the code block itself must still come through byte-identical.
    mixed = f"can you fix this please. it is definitely wrong\n{code}\nthanks"
    embedded_ok = all(
        code in defense.rewrite_turn(mixed, f"author-{i}", f"doc-{i}", 0, 0) for i in range(60)
    )
    check("embedded code block untouched", embedded_ok)
    inline = "use `definitely_flag` and <PATH> here"
    inline_ok = all(
        "`definitely_flag`" in defense.rewrite_turn(inline, f"author-{i}", f"doc-{i}", 0, 0)
        and "<PATH>" in defense.rewrite_turn(inline, f"author-{i}", f"doc-{i}", 0, 0)
        for i in range(60)
    )
    check("inline code and placeholders untouched", inline_ok)

    # 6. Realized rate tracks the drawn rate within sampling error.
    author = "rate-probe"
    key = defense.author_markers(author)[0]
    target = defense.marker_rate(author, key)
    trials = 4000
    hits = sum(defense._applies_to_document(author, key, f"doc-{i}") for i in range(trials))
    realized = hits / trials
    check("realized rate matches drawn rate", abs(realized - target) < 0.03,
          f"target={target:.3f} realized={realized:.3f}")
    check("drawn rate inside configured bounds", defense.rate_min <= target <= defense.rate_max)

    # 7. Profile cardinality is exactly K -- the property the whole design rests on.
    authors_many = [f"a-{i}" for i in range(5000)]
    signatures = {defense.author_markers(a) for a in authors_many}
    check("codebook caps distinct signatures at K", len(signatures) <= defense.n_profiles,
          f"{len(signatures)} distinct signatures for n_profiles={defense.n_profiles}")

    # ... and that the ablation genuinely blows past it, which is the ablation's entire point.
    loose = CollisionSeedingDefense(seed=7, independent=True)
    loose_signatures = {loose.author_markers(a) for a in authors_many}
    check("independent ablation produces many more signatures",
          len(loose_signatures) > 20 * defense.n_profiles, f"{len(loose_signatures)} signatures")

    # 8. Non-Latin text is left alone by universal markers.
    chinese = "请帮我写一个程序来处理这些数据，谢谢你的帮助。"
    china_ok = all(
        defense.rewrite_turn(chinese, f"author-{i}", f"doc-{i}", 0, 0) == chinese for i in range(60)
    )
    check("non-Latin turn untouched", china_ok)

    # 9. Empty and whitespace-only turns survive untouched (regroup_turns still expects them back).
    check("blank turn untouched", defense.rewrite_turn("   ", "a", "d", 0, 0) == "   ")

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


def _run(defense: CollisionSeedingDefense, texts, authors, ids) -> list[str]:
    """Drive ``transform`` the way ``apply_defenses`` does, for the self-test."""
    import contextlib
    import io

    data = AttackData(
        known_embeddings=np.zeros((0, 0), dtype=np.float32),
        unknown_embeddings=np.zeros((len(texts), 0), dtype=np.float32),
        known_labels=np.empty(0, dtype=object),
        unknown_labels=np.asarray(authors, dtype=object),
        unknown_texts=np.asarray(texts, dtype=object),
        unknown_ids=np.asarray(ids, dtype=object),
    )
    with contextlib.redirect_stdout(io.StringIO()):  # silence the coverage report
        defended = defense.transform(data, cache=None)
    return [str(t) for t in defended.unknown_texts]


# --- command line ------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="run the correctness checks and exit (needs no data)")
    p.add_argument("--audit", action="store_true",
                   help="measure every marker's natural base rate and trigger coverage on a corpus, "
                        "and print the set that survives the cut thresholds")
    p.add_argument("--manifest", action="store_true",
                   help="write the author -> profile assignment parquet the evaluation joins against")
    p.add_argument("--source", default="swe_chat", help="built split to read (default: swe_chat)")
    p.add_argument("--dist-dir", default=None, help="directory holding the built parquets")
    p.add_argument("--out-dir", default=None, help="where --manifest writes (default: data/dist)")
    p.add_argument("--limit", type=int, default=None, help="use only the first N documents")
    p.add_argument("--seed", type=int, default=0, help="master seed (default: 0)")
    p.add_argument("--n-profiles", type=int, default=DEFAULT_N_PROFILES,
                   help=f"profile codebook size (default: {DEFAULT_N_PROFILES})")
    args = p.parse_args()

    if args.selftest:
        print("collision_seeding self-test")
        _selftest()
        return

    if not (args.audit or args.manifest):
        p.error("choose one of --selftest, --audit, --manifest")

    documents, authors = _load_documents(args.source, args.dist_dir, args.limit)
    print(f"[{args.source}] {len(documents):,} documents / {len(set(authors)):,} authors")

    if args.audit:
        rows = audit_markers(documents)
        keep = set(surviving_markers(rows))
        print(f"\n{'marker':<26} {'pool':<15} {'base':>8} {'trigger':>9}  verdict")
        for row in sorted(rows, key=lambda r: (-r["trigger_rate"], r["marker"])):
            if row["marker"] in keep:
                verdict = "keep"
            elif row["base_rate"] <= MIN_BASE_RATE:
                verdict = "DROP (no base rate)"
            elif row["base_rate"] > MAX_BASE_RATE:
                verdict = "DROP (already the norm)"
            else:
                verdict = "drop (no coverage)"
            print(f"{row['marker']:<26} {row['pool']:<15} {row['base_rate']:>7.2%} "
                  f"{row['trigger_rate']:>8.2%}  {verdict}")
        print(f"\n{len(keep)} of {len(rows)} markers survive on {args.source}.")
        print("MARKER_KEYS = (\n" + "".join(f"    {k!r},\n" for k in sorted(keep)) + ")")

    if args.manifest:
        from ..data.config import dist_dir

        defense = CollisionSeedingDefense(seed=args.seed, n_profiles=args.n_profiles)
        manifest = collision_manifest(authors, defense)
        out = Path(args.out_dir) if args.out_dir else dist_dir()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.source}_collision_manifest.parquet"
        manifest.to_parquet(path, index=False)
        sizes = manifest["profile"].value_counts()
        print(f"wrote {len(manifest):,} authors -> {path}")
        print(f"collision groups: {len(sizes)} profiles, "
              f"{sizes.min():,}-{sizes.max():,} authors each (median {int(sizes.median()):,})")


__all__ = [
    "CollisionSeedingDefense",
    "Marker",
    "MARKERS",
    "audit_markers",
    "build_profiles",
    "collision_manifest",
    "latin_ratio",
    "on_free_text",
    "split_protected",
    "surviving_markers",
]


if __name__ == "__main__":
    main()
