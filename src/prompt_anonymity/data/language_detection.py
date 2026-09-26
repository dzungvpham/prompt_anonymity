"""Resolve per-document language labels (primary + secondary) with a language detector.

**Why.** SWE-chat's upstream language labels are unreliable: many documents ship with no label (the
detector abstained on prompts that open with a ``/slash-command`` or a placeholder, or are short /
code-heavy), and some are mislabeled ``English`` where the prose is actually CJK. So we
**re-detect every SWE-chat document** with a dedicated detector and fall back to the upstream label
only where the detector abstains.

**Tool.** `Lingua <https://github.com/pemistahl/lingua-py>`_, chosen for its short-text accuracy and
built-in abstention. It exposes both a whole-text confidence ranking (for the *primary* language)
and a span segmenter (for a *secondary* language).

**Primary vs. secondary (script-first).** In coding conversations English is often *pasted* content
(errors, logs, code, paths), so a naive whole-text vote calls a document English even when the human
is writing in another language. Detection is instead by **script**: a substantial non-Latin share of
the letters (``MINORITY_MIN_SHARE``) becomes the **primary** language, and **English** becomes the
**secondary** only when the Latin content is itself a substantial, confidently-English share (other
Latin-language guesses on that residual are jargon, not a real secondary). Latin-dominant documents
fall back to Lingua's whole-text top language and get no secondary. See :func:`detect_languages`.

**Output schema.** :func:`resolve_document_languages` replaces the upstream ``languages`` list with
``language_primary`` / ``language_secondary`` columns. ``redetect=True`` runs the Lingua pass above
(SWE-chat); ``redetect=False`` just splits the existing ``languages`` list without re-detecting (for
a source whose upstream labels are trusted, e.g. WildChat).

**A second policy: trusted primary, detected secondary (WildChat).** WildChat's upstream primary
label is trusted, so instead of the re-detection above, :func:`add_secondary_languages` keeps
``language_primary`` verbatim and detects only a *second* language, over the corpus's own observed
primary vocabulary (which keeps rare confusable languages from stealing short spans) -- with a
guard requiring the primary to itself be present in the text, so an upstream mislabel yields no
secondary rather than a spurious one.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

import pandas as pd
from lingua import Language, LanguageDetector, LanguageDetectorBuilder
from tqdm import tqdm

from .config import hf_dir


# Candidate languages: the ones that appear in SWE-chat's existing labels, plus common world
# languages an unlabeled prompt might be in. Restricting the set (vs. all ~75 Lingua supports)
# sharpens accuracy on the short prompts that dominate the empties -- fewer confusable candidates.
DETECT_LANGUAGES = [
    Language.ENGLISH, Language.CHINESE, Language.JAPANESE, Language.KOREAN, Language.RUSSIAN,
    Language.PORTUGUESE, Language.SPANISH, Language.FRENCH, Language.GERMAN, Language.ITALIAN,
    Language.TURKISH, Language.PERSIAN, Language.ARABIC, Language.HINDI, Language.VIETNAMESE,
    Language.INDONESIAN, Language.DUTCH, Language.POLISH, Language.UKRAINIAN, Language.THAI,
]

MIN_CHARS = 2                    # fewer than this many letters is too little to identify (CJK is dense)
MIN_CONFIDENCE = 0.70            # Lingua top-candidate confidence floor; below it -> undetermined
MINORITY_MIN_SHARE = 0.20        # a script must cover >= this share of the letters to count as a language
SECONDARY_MIN_CONFIDENCE = 0.50  # confidence floor for a secondary (Latin) language
DEFAULT_LANGUAGE = "English"     # primary label for a document neither the detector nor upstream covers

# Trusted-primary secondary pass (WildChat; see the module docstring). Separate knobs from the
# script-first path above -- this one keeps the upstream primary and only looks for a second
# language, over the corpus's own observed language vocabulary.
WILDCHAT_SECONDARY_MIN_CONFIDENCE = 0.30  # a non-primary language needs this confidence to be a secondary
WILDCHAT_PRIMARY_MIN_PRESENCE = 0.05      # the primary must clear this, else the doc is an upstream mislabel
WILDCHAT_DETECT_CAP = 5000                # detect on at most this many chars of stripped prose (runtime bound)

# Scaffolding removed before detecting: fenced code, inline code, scrubbing placeholders,
# /slash-commands, and @path / @mention refs -- none of it is natural-language prose, and its
# presence at the start of a prompt is exactly what made the upstream detector abstain.
_STRIP_RES = [
    re.compile(r"```.*?```", re.S),          # fenced code block
    re.compile(r"`[^`]*`"),                   # inline code span
    re.compile(r"<[A-Z_]+>"),                 # scrubbing placeholder: <PATH>, <URL>, <ID>, ...
    re.compile(r"(?<!\w)/[A-Za-z][\w:.-]*"),  # /slash-command (not a mid-word slash like and/or)
    re.compile(r"@\S+"),                      # @path / @mention ref
]


def strip_for_detection(text: str) -> str:
    """Remove code, scrubbing placeholders, slash-commands and ``@`` refs; return the prose remainder."""
    for rx in _STRIP_RES:
        text = rx.sub(" ", text)
    return " ".join(text.split())


@lru_cache(maxsize=1)
def _detector():
    """The shared Lingua detector over :data:`DETECT_LANGUAGES` (built once, lazily; ~seconds)."""
    return LanguageDetectorBuilder.from_languages(*DETECT_LANGUAGES).build()


def _char_script(ch: str) -> str | None:
    """Unicode script of a character, or ``None`` for non-letters. CJK/Cyrillic/etc. ranges are
    tested before the Latin fallback because those characters are also ``isalpha``."""
    o = ord(ch)
    if 0x3040 <= o <= 0x30FF:                            return "Kana"        # Hiragana + Katakana
    if 0x4E00 <= o <= 0x9FFF or 0x3400 <= o <= 0x4DBF:   return "Han"
    if 0xAC00 <= o <= 0xD7A3 or 0x1100 <= o <= 0x11FF:   return "Hangul"
    if 0x0400 <= o <= 0x04FF:                            return "Cyrillic"
    if 0x0600 <= o <= 0x06FF or 0x0750 <= o <= 0x077F:   return "Arabic"
    if 0x0900 <= o <= 0x097F:                            return "Devanagari"
    if 0x0E00 <= o <= 0x0E7F:                            return "Thai"
    return "Latin" if ch.isalpha() else None


def _script_runs(prose: str, scripts: set[str]) -> str:
    """Contiguous runs of characters whose script is in ``scripts``, joined by spaces.

    Joining runs (rather than concatenating stray characters) preserves word boundaries so the
    detector sees real words -- important for Cyrillic/Latin, harmless for space-less CJK.
    """
    runs: list[str] = []
    cur: list[str] = []
    for ch in prose:
        if _char_script(ch) in scripts:
            cur.append(ch)
        elif cur:
            runs.append("".join(cur)); cur = []
    if cur:
        runs.append("".join(cur))
    return " ".join(runs)


def _confident_language(text: str, floor: float) -> str | None:
    """Lingua's top language for ``text`` if its confidence clears ``floor``, else ``None``."""
    conf = _detector().compute_language_confidence_values(text)
    if conf and conf[0].value >= floor:
        return conf[0].language.name.title()
    return None


def detect_languages(text: str, *, min_chars: int = MIN_CHARS, min_confidence: float = MIN_CONFIDENCE,
                     minority_min_share: float = MINORITY_MIN_SHARE,
                     secondary_min_confidence: float = SECONDARY_MIN_CONFIDENCE,
                     ) -> tuple[str | None, str | None]:
    """Detect a document's ``(primary, secondary)`` languages, ``(None, None)`` if undetermined.

    Strips scaffolding first (:func:`strip_for_detection`), then decides by **script** (see the
    module docstring for why): a substantial non-Latin share of the letters makes that script's
    language the **primary**, with **English** as the **secondary** only when the Latin content is
    itself a substantial, confidently-English share. Otherwise the document is Latin-dominant: the
    primary is Lingua's whole-text top language and there is no secondary.

    Returns ``(None, None)`` when there are fewer than ``min_chars`` letters or the language is not
    identified confidently (``min_confidence``), so the caller can fall back. Labels are Title-cased
    to match the existing vocabulary (``"English"``, ``"Japanese"``, ...).
    """
    prose = strip_for_detection(text)
    scripts = [s for s in map(_char_script, prose) if s]
    n = len(scripts)
    if n < min_chars:
        return None, None
    n_latin = scripts.count("Latin")

    if (n - n_latin) / n >= minority_min_share:
        # Substantial non-Latin script -> the user's writing language. Latin is a secondary only
        # if also substantial, and only when it's confidently English (a bag of technical
        # identifiers can mis-tag as German/Dutch/etc., so other guesses are dropped as jargon).
        primary = _confident_language(_script_runs(prose, set(scripts) - {"Latin"}), min_confidence)
        if primary is not None:
            secondary = None
            if n_latin / n >= minority_min_share:
                if _confident_language(_script_runs(prose, {"Latin"}), secondary_min_confidence) == "English":
                    secondary = "English"
            return primary, secondary
        # non-Latin text not confidently identifiable -> fall through to the whole-text vote

    return _confident_language(prose, min_confidence), None


def resolve_document_languages(
    frame: pd.DataFrame, *, redetect: bool, min_chars: int = MIN_CHARS,
    min_confidence: float = MIN_CONFIDENCE, minority_min_share: float = MINORITY_MIN_SHARE,
) -> tuple[pd.DataFrame, dict]:
    """Add ``language_primary`` / ``language_secondary`` columns, replacing the ``languages`` list.

    ``redetect=True`` re-detects every document with Lingua (:func:`detect_languages`) and, where
    it abstains, keeps only the upstream primary -- its secondary there is unverifiable and dropped
    (the SWE-chat policy). ``redetect=False`` just splits the existing list without re-detecting
    (the schema-only update for a trusted-label source such as WildChat). A document neither the
    detector nor the upstream list can label falls back to :data:`DEFAULT_LANGUAGE`, so
    ``language_primary`` is **never null**.

    Leaves the intermediate ``languages`` column in place. Returns ``(new_frame, stats)`` with
    ``n_lingua``, ``n_fallback`` and ``n_default`` counts.
    """
    upstream = [list(ls) for ls in frame["languages"]]
    turns = list(frame["turns"])
    primaries: list[str] = []
    secondaries: list[str | None] = []
    n_lingua = n_fallback = n_default = 0

    for i, up in enumerate(upstream):
        primary = secondary = None
        if redetect:
            primary, secondary = detect_languages(
                "\n".join(turns[i]), min_chars=min_chars, min_confidence=min_confidence,
                minority_min_share=minority_min_share,
            )
        if primary is not None:
            n_lingua += 1
        elif len(up) >= 1:                           # fall back to the upstream label
            primary = up[0]
            # Keep the upstream secondary only when we trust the upstream labels wholesale
            # (redetect=False). After a Lingua abstention the secondary is the unverifiable part.
            if not redetect:
                secondary = up[1] if len(up) >= 2 else None
            n_fallback += 1
        else:                                        # neither detector nor upstream had a label
            primary = DEFAULT_LANGUAGE
            n_default += 1
        primaries.append(primary)
        secondaries.append(secondary)

    frame = frame.copy()
    frame["language_primary"] = primaries
    frame["language_secondary"] = secondaries
    stats = {"n_docs": len(frame), "n_lingua": n_lingua, "n_fallback": n_fallback, "n_default": n_default}
    return frame, stats


# --- trusted-primary secondary pass (WildChat) ------------------------------
# Keeps an already-resolved ``language_primary`` and detects only a *second* language, over the
# languages the corpus itself uses as primaries. See the module docstring for why WildChat gets
# this instead of the script-first re-detection above.

def observed_language_detector(primary_labels: pd.Series) -> LanguageDetector | None:
    """Build a high-accuracy Lingua detector over the languages appearing as primaries in a corpus.

    Restricting the candidate set to the corpus's own vocabulary (vs. all ~75 Lingua languages) is
    what keeps rare, confusable languages from grabbing short spans of an otherwise monolingual
    prompt. Labels that are not Lingua languages (notably ``"Nolang"``, the upstream abstention
    marker) are dropped from the candidate set; documents carrying them simply get no secondary.

    Returns ``None`` when fewer than two candidates remain -- a detector needs at least two
    languages to choose between, and a single-language corpus has no secondary to find anyway.
    """
    by_name = {lang.name.title(): lang for lang in Language.all()}
    candidates = [by_name[name] for name in primary_labels.dropna().unique() if name in by_name]
    if len(candidates) < 2:
        return None
    return LanguageDetectorBuilder.from_languages(*candidates).build()


def detect_secondary_language(
    text: str, primary: str, detector: LanguageDetector, *,
    threshold: float = WILDCHAT_SECONDARY_MIN_CONFIDENCE,
    primary_floor: float = WILDCHAT_PRIMARY_MIN_PRESENCE,
    cap: int = WILDCHAT_DETECT_CAP,
) -> str | None:
    """Return a document's secondary language, or ``None`` if it is (effectively) monolingual.

    ``primary`` is the kept label, ``detector`` an :func:`observed_language_detector`. The secondary
    is the most confident language that is not the primary and clears ``threshold`` -- but only when
    the primary is itself present in the text (confidence >= ``primary_floor``), so an upstream
    *mislabel* (text wholly in a different language than ``primary``) yields ``None`` rather than a
    spurious "secondary". Detection reads at most ``cap`` characters of the scaffolding-stripped
    prose (:func:`strip_for_detection`) to bound runtime; a substantial second language surfaces
    well within that.
    """
    prose = strip_for_detection(text)[:cap]
    if len(prose) < MIN_CHARS:
        return None
    confidences = detector.compute_language_confidence_values(prose)  # all candidates, descending
    by_name = {c.language.name.title(): c.value for c in confidences}
    if by_name.get(primary, 0.0) < primary_floor:
        return None  # primary not genuinely present -> treat as a mislabel, not a bilingual doc
    for c in confidences:  # sorted by confidence descending
        name = c.language.name.title()
        if name == primary:
            continue
        return name if c.value >= threshold else None
    return None


def add_secondary_languages(
    frame: pd.DataFrame, *,
    threshold: float = WILDCHAT_SECONDARY_MIN_CONFIDENCE,
    primary_floor: float = WILDCHAT_PRIMARY_MIN_PRESENCE,
    cap: int = WILDCHAT_DETECT_CAP,
) -> tuple[pd.DataFrame, dict]:
    """Fill ``language_secondary`` for a frame whose ``language_primary`` is already resolved.

    Runs :func:`detect_secondary_language` over every document, *replacing* the existing
    ``language_secondary`` column -- this is for a source whose upstream data carries no second
    language of its own (WildChat), so there is nothing to preserve. ``language_primary`` is never
    touched.

    Returns ``(new_frame, stats)`` with ``n_docs`` and ``n_secondary`` (documents that got one).
    """
    detector = observed_language_detector(frame["language_primary"])
    if detector is None:  # <2 candidate languages: nothing can be a *second* language
        secondaries: list[str | None] = [None] * len(frame)
    else:
        secondaries = [
            detect_secondary_language("\n".join(turns), primary, detector,
                                      threshold=threshold, primary_floor=primary_floor, cap=cap)
            for turns, primary in tqdm(zip(frame["turns"], frame["language_primary"]),
                                       total=len(frame), desc="  detecting secondary language")
        ]
    frame = frame.copy()
    frame["language_secondary"] = pd.Series(secondaries, index=frame.index, dtype=object)
    return frame, {"n_docs": len(frame), "n_secondary": sum(s is not None for s in secondaries)}


def main() -> None:
    """Report the language columns of ``hf/swe_chat.parquet`` (post-build), or a dry-run resolve."""
    swe = pd.read_parquet(hf_dir() / "swe_chat.parquet")
    if "language_primary" in swe.columns:
        prim = swe["language_primary"].fillna("<none>").value_counts()
        n_sec = int(swe["language_secondary"].notna().sum())
        print(f"swe_chat.parquet language_primary distribution ({len(swe):,} docs):")
        print(prim.to_string())
        print(f"\ndocuments with a secondary language: {n_sec:,}")
        if n_sec:
            print(swe.loc[swe["language_secondary"].notna(), "language_secondary"]
                  .value_counts().head(10).to_string())
    elif "languages" in swe.columns:
        _, stats = resolve_document_languages(swe, redetect=True)
        print(f"dry-run resolve (redetect=True): {stats}")
    else:
        print("swe_chat.parquet has neither `languages` nor `language_primary`.")


if __name__ == "__main__":
    main()
