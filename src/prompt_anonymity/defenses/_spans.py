"""Sentence-level span segmentation with exact character offsets.

The leave-one-out defense (:mod:`.loo_unlink`) scores *spans* of a prompt rather than whole turns:
delete one, re-embed, and see how much same-author similarity moves. That needs a segmenter with
three properties:

* **Exact offsets.** A span carries ``(turn_index, start, end)`` into its own turn, so an edit can be
  spliced back in place and audited against the original text.
* **A total partition.** The spans of a turn concatenate back to the turn byte-for-byte
  (:func:`spans_cover`), so deleting a span is unambiguous string surgery and a "delete nothing" run
  is provably a pass-through.
* **Protected regions are never cut.** These corpora are full of fenced code blocks, inline code,
  URLs and placeholders (``<URL>``, ``<PATH>``); a boundary landing inside one would let the defense
  delete half a code fence.

The protected-region machinery is borrowed from :mod:`.collision_seeding` rather than reimplemented,
so the two defenses can't drift apart on "which parts of this text are safe to rewrite". The import
direction points into that module (rather than the other way) because it already has committed
results and a cache keyed on its own source.

Sentence boundaries come from NLTK's Punkt (installed for :mod:`.dp_mlm`, the ``[dpmlm]`` extra).
Punkt is English-trained, so a non-Latin or Punkt-unavailable turn degrades to
:data:`FALLBACK_SPLIT_RE` instead of failing; :attr:`Span.fallback` records which.

Run ``python -m prompt_anonymity.defenses._spans --selftest`` for the invariant checks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

# The single definition of "which parts of this text are safe to rewrite", owned by
# collision_seeding -- see the module docstring for why the import points that way. `_PROTECTED_RE`
# is private there only because it had no second caller until now; nothing else about it is internal.
from .collision_seeding import _PROTECTED_RE as PROTECTED_RE
from .collision_seeding import latin_ratio, on_free_text, split_protected

#: Used when Punkt is unavailable or the turn is not Latin-script enough for an English model to be
#: meaningful. Deliberately crude: splits after sentence-final punctuation and after a blank line.
#: The CJK branch is zero-width on purpose -- CJK sentence punctuation isn't followed by a space, so
#: requiring ``\s+`` after it would silently return a whole Chinese/Japanese turn as one span.
FALLBACK_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|(?<=[。！？])|\n{2,}")

#: Word boundaries, used when a turn turns out to be a single sentence. Cutting after a whitespace
#: run (rather than before a word) keeps each span's trailing space attached to it, exactly as the
#: sentence splitters do, so the two granularities tile a turn the same way.
WORD_SPLIT_RE = re.compile(r"\s+")

#: A turn shorter than this many words is left as one span even when it is a single sentence.
#: Word-splitting "thanks!" produces spans that cannot be scored (deleting the only content word
#: deletes the prompt) and cannot be generalized upward into anything.
MIN_WORDS_FOR_WORD_SPANS = 4

#: A span with less free (non-protected, non-whitespace) text than this is not worth editing: there
#: is nothing for a rewriter to generalize, and asking it to try is how a bare ``</code>`` becomes a
#: paragraph of invented prose. Such spans are still *scored*, because "the code block is what
#: identifies this author" is a finding, and still counted in the span-kind aggregate.
MIN_EDITABLE_CHARS = 12

#: The same floor for word spans, where 12 characters would rule out most English words and silently
#: turn the word-granularity path back into a no-op. Two characters keeps bare punctuation and
#: stray whitespace out while letting real words through.
MIN_EDITABLE_WORD_CHARS = 2

#: Share of a turn's letters that must be Latin-script before Punkt is trusted over the fallback.
LATIN_THRESHOLD = 0.5


@dataclass(frozen=True)
class Span:
    """One candidate span: a sentence-ish slice of one user turn, located by character offset.

    Attributes
    ----------
    turn_index : int
        Which turn of the document this span belongs to. Turn boundaries survive the defense
        (``apply_defenses`` requires one output turn per input turn), so an edit is always applied
        within a single turn and never across two.
    start, end : int
        Half-open character offsets into ``turns[turn_index]``. ``end`` of one span is ``start`` of
        the next, so the spans of a turn tile it completely -- see :func:`spans_cover`.
    text : str
        ``turns[turn_index][start:end]``, carried so scoring code does not have to keep the turn
        list alongside every span.
    editable : bool
        Whether the edit loop may rewrite this span. False for spans that are essentially all code,
        placeholder or whitespace (see :data:`MIN_EDITABLE_CHARS`). Such spans are still scored for
        linkage; they are simply never chosen as an edit target.
    granularity : {"sentence", "word"}
        Which splitter produced this span. ``"word"`` means the turn was a single sentence and was
        re-cut by word -- see :func:`segment_turns`. Carried into the edit log and the span-kind
        aggregate, because "the identifying spans were all single words" is a different finding from
        "the identifying spans were all sentences".
    fallback : bool
        True when this span came from :data:`FALLBACK_SPLIT_RE` rather than Punkt. Recorded so a run
        can report how much of its segmentation was heuristic instead of silently mixing the two.
    """

    turn_index: int
    start: int
    end: int
    text: str
    editable: bool
    granularity: str
    fallback: bool

    def __len__(self) -> int:
        return self.end - self.start


# --- protected regions -------------------------------------------------------
#
# `split_protected`, `on_free_text` and `latin_ratio` are imported from collision_seeding above and
# re-exported here, so a caller that only needs spans has one module to import. The two functions
# below are additions this module needs and that one does not: it rewrites *within* free chunks and
# never has to know where they sit in the original string, whereas segmentation is entirely about
# offsets.

def protected_ranges(text: str) -> list[tuple[int, int]]:
    """Half-open ``(start, end)`` offsets of every protected region, in order."""
    return [(match.start(), match.end()) for match in PROTECTED_RE.finditer(text)]


def free_char_count(text: str) -> int:
    """Non-whitespace characters of ``text`` that sit outside any protected region.

    The measure :data:`MIN_EDITABLE_CHARS` is applied to. Whitespace is excluded so an indented
    code block surrounded by newlines does not read as editable prose.
    """
    return sum(len("".join(chunk.split())) for chunk, free in split_protected(text) if free)


# --- sentence boundaries -----------------------------------------------------

_PUNKT_UNAVAILABLE = False


def _punkt_starts(text: str, language: str) -> list[int] | None:
    """Sentence start offsets from NLTK Punkt, or ``None`` when it is unavailable.

    Offsets are recovered by walking ``sent_tokenize``'s output through the source with ``str.index``
    rather than ``span_tokenize``, since ``sent_tokenize`` is the stable public API across NLTK
    versions. It only ever strips surrounding whitespace, never rewrites characters, so the walk is
    exact -- and if a sentence can't be located, the whole turn falls back rather than producing
    offsets that don't point at their own text.
    """
    global _PUNKT_UNAVAILABLE
    if _PUNKT_UNAVAILABLE:
        return None
    try:
        # Imported lazily: nltk arrives with the [dpmlm] extra, and importing this module must not
        # require it (the fallback splitter is pure stdlib and good enough for a selftest).
        from nltk.tokenize import sent_tokenize

        sentences = sent_tokenize(text, language=language)
    except (ImportError, LookupError):
        # LookupError is NLTK's "the punkt data is not downloaded" signal. Remember it so a corpus
        # run does not pay the exception on every one of a million turns.
        _PUNKT_UNAVAILABLE = True
        return None

    starts: list[int] = []
    cursor = 0
    for sentence in sentences:
        if not sentence:
            continue
        try:
            position = text.index(sentence, cursor)
        except ValueError:
            return None  # cannot align -- refuse rather than emit offsets that lie
        starts.append(position)
        cursor = position + len(sentence)
    return starts


def _fallback_starts(text: str) -> list[int]:
    """Sentence start offsets from :data:`FALLBACK_SPLIT_RE`."""
    return [match.end() for match in FALLBACK_SPLIT_RE.finditer(text)]


def _word_starts(text: str) -> list[int]:
    """Offsets just past each run of whitespace -- i.e. where each word after the first begins."""
    return [match.end() for match in WORD_SPLIT_RE.finditer(text)]


def _admissible(text: str, starts) -> list[int]:
    """Drop cut points that are redundant or would split a protected region.

    A cut at 0 or at ``len(text)`` produces an empty span; a cut strictly inside a fenced code block,
    URL or ``<PLACEHOLDER>`` would let the defense delete half of it. Both are dropped here so every
    splitter gets the same protection without repeating the check.
    """
    blocked = protected_ranges(text)
    return sorted({
        start for start in starts
        if 0 < start < len(text)
        and not any(begin < start < end for begin, end in blocked)
    })


def _cut_points(text: str, language: str) -> tuple[list[int], str, bool]:
    """``(sorted cut offsets, granularity, used_fallback)`` for one turn.

    Sentence boundaries first. If they yield no cut at all -- the turn is a single sentence -- the
    turn is re-cut by word instead: leave-one-out on a single span would just delete the entire
    prompt, carrying no information about *which part* identifies the author. Word granularity
    applies only in that case; :attr:`Span.granularity` tells the two regimes apart.
    """
    used_fallback = False
    starts = None
    if latin_ratio(text) >= LATIN_THRESHOLD:
        starts = _punkt_starts(text, language)
    if starts is None:
        starts = _fallback_starts(text)
        used_fallback = True

    cuts = _admissible(text, starts)
    if cuts:
        return cuts, "sentence", used_fallback

    # One sentence. Fall back to words, unless the turn is too short for that to mean anything.
    if len(text.split()) >= MIN_WORDS_FOR_WORD_SPANS:
        word_cuts = _admissible(text, _word_starts(text))
        if word_cuts:
            return word_cuts, "word", used_fallback
    return [], "sentence", used_fallback


# --- the public entry point --------------------------------------------------

def segment_turns(turns: Sequence[str], *, language: str = "english") -> list[Span]:
    """Segment a document's user turns into sentence-level spans, in document order.

    Every turn is tiled completely: for a given ``turn_index`` the spans are contiguous, start at 0
    and end at ``len(turn)``. An empty turn contributes no spans (there is nothing to score and
    nothing to edit), which is why callers must reassemble edited documents through
    :func:`apply_edits` rather than by concatenating spans.

    A turn that is a single sentence is segmented by **word** instead -- see :func:`_cut_points` for
    why, and check :attr:`Span.granularity` rather than assuming one regime.
    """
    spans: list[Span] = []
    for turn_index, turn in enumerate(turns):
        turn = "" if turn is None else str(turn)
        if not turn:
            continue
        cuts, granularity, used_fallback = _cut_points(turn, language)
        floor = MIN_EDITABLE_WORD_CHARS if granularity == "word" else MIN_EDITABLE_CHARS
        bounds = [0, *cuts, len(turn)]
        for start, end in zip(bounds, bounds[1:]):
            text = turn[start:end]
            spans.append(Span(
                turn_index=turn_index,
                start=start,
                end=end,
                text=text,
                editable=free_char_count(text) >= floor,
                granularity=granularity,
                fallback=used_fallback,
            ))
    return spans


def spans_cover(turns: Sequence[str], spans: Sequence[Span]) -> bool:
    """Whether ``spans`` tile ``turns`` exactly -- the invariant every caller relies on.

    Used by the selftest and cheap enough to assert in a debug run over real data. Turns that
    produced no spans (empty ones) are excluded from the check rather than counted as a failure.
    """
    rebuilt: dict[int, str] = {}
    for span in spans:
        if span.text != str(turns[span.turn_index])[span.start:span.end]:
            return False
        rebuilt[span.turn_index] = rebuilt.get(span.turn_index, "") + span.text
    return all(rebuilt[index] == str(turns[index]) for index in rebuilt)


# --- applying edits ----------------------------------------------------------

def apply_edits(turns: Sequence[str], edits: Sequence[tuple[Span, str]]) -> list[str]:
    """Return ``turns`` with each span replaced by its new text; turn count is preserved.

    Edits are applied **back to front within each turn** so that an earlier edit never invalidates a
    later span's offsets. Two edits overlapping the same span are a programming error and raise:
    silently letting the second win would make the edit log a lie about what was applied.

    Passing an empty ``edits`` returns the turns unchanged and byte-identical, which is what makes a
    zero-budget run a provable pass-through.
    """
    by_turn: dict[int, list[tuple[Span, str]]] = {}
    for span, replacement in edits:
        by_turn.setdefault(span.turn_index, []).append((span, replacement))

    result = [str(turn) for turn in turns]
    for turn_index, turn_edits in by_turn.items():
        turn_edits.sort(key=lambda item: item[0].start, reverse=True)
        previous_start = None
        for span, replacement in turn_edits:
            if previous_start is not None and span.end > previous_start:
                raise ValueError(
                    f"overlapping edits on turn {turn_index}: span [{span.start}, {span.end}) "
                    f"runs into an already-applied edit starting at {previous_start}."
                )
            text = result[turn_index]
            result[turn_index] = text[:span.start] + replacement + text[span.end:]
            previous_start = span.start
    return result


def without_span(turns: Sequence[str], span: Span) -> list[str]:
    """``turns`` with ``span`` deleted -- the leave-one-out variant scored in
    :mod:`.loo_unlink`.

    Deletion (rather than replacement by a placeholder) is what makes the linkage score measure the
    *information* the span carries: a placeholder would leave a token for the encoder to key on and
    understate the span's contribution.
    """
    return apply_edits(turns, [(span, "")])


# --- selftest ----------------------------------------------------------------

_SELFTEST_TURNS = [
    "I'm building a Django inventory app for a llama farm in Reykjavik. "
    "It needs to track feed batches. Can you sketch the models?",
    "Here is what I have so far:\n\n```python\nclass Llama(models.Model):\n"
    "    name = models.CharField(max_length=64)\n```\n\n"
    "The migration fails with `django.db.utils.OperationalError`. See <URL> for the traceback. "
    "What am I missing?",
    "",
    "谢谢！这个方法很有用。我还有一个问题。",
]


def _selftest() -> None:
    """Invariant checks over :data:`_SELFTEST_TURNS`. Exits non-zero on the first failure."""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("segment_turns:")
    spans = segment_turns(_SELFTEST_TURNS)
    check(bool(spans), "produced at least one span")
    check(spans_cover(_SELFTEST_TURNS, spans), "spans tile every turn byte-for-byte")
    check(all(span.text == _SELFTEST_TURNS[span.turn_index][span.start:span.end] for span in spans),
          "every span's text matches its own offsets")
    check(all(span.turn_index != 2 for span in spans), "an empty turn contributes no spans")
    check(len([span for span in spans if span.turn_index == 3]) > 1,
          "a CJK turn splits on its own punctuation, which carries no trailing whitespace")

    print("protected regions:")
    fenced = [span for span in spans if "```" in span.text]
    check(len(fenced) == 1, f"the fenced code block is not split across spans (got {len(fenced)})")
    check(all(span.text.count("```") % 2 == 0 for span in spans),
          "no span holds an unbalanced code fence")
    check(all("<URL>" not in span.text or span.text.count("<URL>") == 1 for span in spans),
          "a placeholder is never cut in half")

    print("single-sentence turns fall back to words:")
    single = segment_turns(["Can you help me debug my Django inventory app for a llama farm?"])
    check(len(single) > 1, f"a one-sentence turn is split (got {len(single)} spans)")
    check(all(span.granularity == "word" for span in single), "...and it is split by word")
    check(spans_cover(["Can you help me debug my Django inventory app for a llama farm?"], single),
          "word spans still tile the turn byte-for-byte")
    check(sum(span.editable for span in single) > 1, "word spans are editable (the 12-char floor "
                                                     "would have excluded nearly every word)")
    check([span.text for span in single][:3] == ["Can ", "you ", "help "],
          "each word span carries its own trailing whitespace")

    multi = segment_turns([_SELFTEST_TURNS[0]])
    check(all(span.granularity == "sentence" for span in multi),
          "a multi-sentence turn stays at sentence granularity")
    short = segment_turns(["thanks!"])
    check(len(short) == 1, "a turn under the word-span floor is left whole")

    print("editable flag:")
    code_only = segment_turns(["```\nx = 1\n```"])
    check(all(not span.editable for span in code_only), "a pure-code turn has no editable span")
    prose = segment_turns(["This is an ordinary sentence about a project. And a second one."])
    check(all(span.editable for span in prose), "an ordinary prose turn is editable")

    print("apply_edits:")
    check(apply_edits(_SELFTEST_TURNS, []) == [str(t) for t in _SELFTEST_TURNS],
          "no edits is a byte-identical pass-through")
    first = next(span for span in spans if span.editable)
    edited = apply_edits(_SELFTEST_TURNS, [(first, "REPLACED. ")])
    check(len(edited) == len(_SELFTEST_TURNS), "turn count survives an edit")
    check("REPLACED. " in edited[first.turn_index], "the replacement landed in the right turn")

    turn_zero = [span for span in spans if span.turn_index == 0]
    check(len(without_span(_SELFTEST_TURNS, turn_zero[0])[0]) < len(_SELFTEST_TURNS[0]),
          "without_span shortens the turn it targets")

    overlapping = [(turn_zero[0], "a"), (turn_zero[0], "b")]
    try:
        apply_edits(_SELFTEST_TURNS, overlapping)
        check(False, "overlapping edits raise")
    except ValueError:
        check(True, "overlapping edits raise")

    print(f"\n{len(failures)} failure(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true", help="run the invariant checks")
    arguments = parser.parse_args()
    if arguments.selftest:
        _selftest()
    else:
        parser.error("nothing to do; pass --selftest")
