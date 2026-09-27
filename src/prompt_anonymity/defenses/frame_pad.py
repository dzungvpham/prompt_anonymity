r"""Frame pad: leave the prompt alone and bolt a shared, off-topic turn onto the end of it.

:mod:`~prompt_anonymity.defenses.frame_shift` both **adds** a heavy topical frame and **rewrites**
the user's request into that frame's register. This defense isolates the first half: the user's
turns are returned **byte-identical**, and the document simply gains one extra turn of dense,
content-loaded prose from one of the same 50 scenes (:data:`~.frame_shift.FRAMINGS`), drawn by a
keyed hash of its ``doc_id``::

    turns: ["how can I create a slurm script", "no, with a GPU"]
    ->     ["how can I create a slurm script", "no, with a GPU",
            "The orangery has not been warm since Michaelmas. Lady Ashcombe took her chocolate ..."]

So it answers one question cleanly: **does adding shared off-topic content dilute the authorial
signal by itself, with nothing rewritten?** Read next to ``frame_shift`` it decomposes that defense
into dilution versus rewriting; read next to ``collision_seeding`` it is the same collision idea moved
from the character-n-gram channel (spelling, punctuation) to whole paragraphs of shared topical text.

**The padding text is generated once, not per document.** A bank of
:data:`FRAME_PAD_PASSAGES_PER_FRAME` passages is written for each of the 50 scenes, on the same model
``frame_shift`` rewrites with, and every document draws one passage from its scene's bank. Corpus
scale is therefore free. It also means the pad **repeats** across documents that draw the same scene
and passage index -- deliberately: shared text is collision material, while unique-per-document
padding would only add length.

**The pad is uncorrelated with the document by construction.** It is drawn from the ``doc_id`` alone
(:func:`~._keying.keyed_rng`) and the defense never reads the document's text -- ``apply_defenses``
calls :meth:`FramePadDefense.extra_turns`, which takes an identifier and nothing else, so there is no
path by which the padding could be chosen to suit, echo or summarise what the user wrote. A pad that
tracked the topic would reinforce the topical signal instead of burying it. The same keying also
makes the assignment survive sharding under a SLURM array and be reconstructible offline
(``--manifest``), and it shares its draw namespace and seed with ``frame_shift``, so a document lands
in the same scene under both defenses and the two arms are comparable document by document.

**Nothing in the codebook is software-adjacent** -- a property of :data:`~.frame_shift.FRAMINGS` --
and the passage prompt's rule 3 forbids computing vocabulary outright, because the corpora are
software chat and assistant chat and a pad sharing their words would blend into the text it is meant
to sit apart from. That rule is *not* machine-enforced, so ``--show-bank`` is meant to be read by a
person once the bank is built.

**This defense adds a turn, which no other defense here does.** ``apply_defenses`` normally requires
exactly one output turn per input turn (see
:func:`~prompt_anonymity.data.apply_defenses.regroup_turns`), and it hands a defense per-turn rows
rather than documents, so a turn cannot be added from inside the usual rewrite path at all. Instead
this class declares :attr:`~FramePadDefense.appends_turns` and exposes :meth:`FramePadDefense.extra_turns`,
which ``apply_defenses`` calls per document *after* regrouping. It is not a
:class:`~prompt_anonymity.caching.CachedDefense` for the same reason it is cheap: the transform is a
dictionary lookup, and caching it would buy nothing while costing a million-row table write.

**Where this can silently do nothing.** ``gemini_embedding_2`` reads only a document's first 8,192
tokens and discards the rest (see :mod:`prompt_anonymity.features.gemini_embedding`), and a document's
text is its turns joined by a blank line -- so for any document already longer than that window, the
appended turn is never embedded. ``--preview`` reports what share of the sampled documents are in
that state *before* padding, which is what separates "the pad did nothing" from "the pad was not
read".

**The bank is English**, while WildChat is not: a Russian document gets an English pad. That is a
strong shared signal (good for collision) and a conspicuous one (bad for plausibility); it is a
property of this arm, not a bug.

Run it::

    python -m prompt_anonymity.defenses.frame_pad --selftest             # offline, no API key
    python -m prompt_anonymity.defenses.frame_pad --build-bank           # ~2 cents, once
    python -m prompt_anonymity.defenses.frame_pad --show-bank
    python -m prompt_anonymity.defenses.frame_pad --preview --limit 5    # free, reads the bank
    python -m prompt_anonymity.defenses.frame_pad --manifest --source swe_chat
    python -m prompt_anonymity.data.apply_defenses --source swe_chat --defense frame_pad
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from ._backends import TURN_DELIM, document_id
from ._keying import keyed_rng
from .frame_shift import FRAMING_KEYS, FRAMINGS, SINGLE_FRAMING_KEY, Framing

#: The bank writer, an OpenRouter chat model id. Shared with ``frame_shift``'s rewriter so both arms
#: are written by the same model -- removing "one arm had a better writer" as an explanation of any
#: gap between them.
#:
#: Cost is not a reason to pick a cheaper model here: the bank is a one-time build, and what it buys
#: is prose density, which is the property this defense lives on (see :func:`passage_density`). A
#: cheaper model writes thinner, more generic passages, and a thin bank reduces the whole arm to "the
#: documents got longer". Read the bank with ``--show-bank`` before building a corpus on it.
FRAME_PAD_MODEL = os.environ.get("FRAME_PAD_MODEL", "deepseek/deepseek-v4-flash-0731")
#: Master seed for the scene and passage draws. Shared with ``frame_shift`` by design (see the module
#: docstring), so at equal seeds a document draws the same scene under both defenses.
FRAME_PAD_SEED = int(os.environ.get("FRAME_PAD_SEED", "0"))
#: Passages generated per scene. This is the collision knob: K scenes x P passages is the number of
#: distinct pads in circulation, so P=1 would make the pad a perfect indicator of the scene (and the
#: scene a perfect indicator of a group of documents), while a large P dilutes toward per-document
#: uniqueness and stops being collision material at all. Part of the bank, so changing it rebuilds.
FRAME_PAD_PASSAGES_PER_FRAME = int(os.environ.get("FRAME_PAD_PASSAGES_PER_FRAME", "8"))
#: Target length of one passage, in words. Fixed length is the design: every document gets the same
#: amount of padding whatever its own size, so a short document is diluted far more than a long one.
FRAME_PAD_TARGET_WORDS = int(os.environ.get("FRAME_PAD_TARGET_WORDS", "180"))
#: Passages this short (characters) are rejected at build time: the model returned a stub rather than
#: a passage, and a stub pads nothing.
FRAME_PAD_MIN_PASSAGE_CHARS = 200

FRAME_PAD_TEMPERATURE = 0.0   # greedy -> the bank is reproducible from the prompt and the codebook
FRAME_PAD_TOP_P = 1.0
FRAME_PAD_OUTPUT_TAG = "passage"
#: Concurrent requests during a bank build. Higher than other defenses use because this job is many
#: short, network-bound calls at temperature 0, so concurrency only changes how long the wait is.
#: The client already backs off with full jitter on a 429.
FRAME_PAD_MAX_WORKERS = int(os.environ.get("FRAME_PAD_MAX_WORKERS", "24"))
#: Calls per progress line, so a slow build and a hung one don't look identical on stdout.
FRAME_PAD_PROGRESS_EVERY = int(os.environ.get("FRAME_PAD_PROGRESS_EVERY", "48"))
FRAME_PAD_MAX_RETRIES = int(os.environ.get("FRAME_PAD_MAX_RETRIES", "8"))
FRAME_PAD_TIMEOUT = float(os.environ.get("FRAME_PAD_TIMEOUT", "180"))

#: Filename of the passage bank inside the dist directory.
FRAME_PAD_BANK_FILENAME = "frame_pad_bank.json"
#: The bank committed alongside this module -- the default, and the reason a normal run needs no API
#: key and no generation step. See :func:`bank_path` for the precedence, and ``--build-bank`` for
#: making a new one (which lands in ``data/dist`` and takes precedence over this).
#:
#: **CSV, not JSON**, because this file is meant to be read and edited by people, with a passage's
#: length and density beside it. :meth:`PassageBank.load` dispatches on the extension, and
#: :meth:`PassageBank.to_csv` round-trips, so editing or deleting a row changes the bank with no
#: rebuild step.
PACKAGED_BANK = Path(__file__).with_name("frame_pad_bank.csv")
#: Environment variable pointing at a bank file directly (wins over the dist-directory default).
#: Point it at a copy to freeze a bank against a rebuild, the way ``$PROMPT_ANONYMITY_MODELS_CONFIG``
#: pins a models config.
FRAME_PAD_BANK_ENV = "FRAME_PAD_BANK"

#: Tokens ``gemini_embedding_2`` reads from a document before discarding the rest. Not used to make
#: any decision -- only to report, in ``--preview``, how many documents are already past it and so
#: cannot have their pad embedded at all. Kept here rather than imported so this module never pulls
#: in the featurizer package.
EMBEDDING_WINDOW_TOKENS = 8192
#: Characters per token when estimating that window without a tokenizer.
CHARS_PER_TOKEN = 4


FRAME_PAD_SYSTEM_PROMPT = """
You are FramePad, a generator of dense, inert scene prose.

# Task
You are given a SCENE and an ASPECT of it. Write ONE passage of heavily detailed prose from inside
that scene, about that aspect. The passage will be appended to an unrelated document, so it must
stand completely on its own, must be packed with concrete subject matter, and must never address,
instruct or question a reader.

# Rules

1. BE DENSE WITH CONTENT. This is the most important rule after rule 2. Every sentence must carry
   specific, checkable substance from inside the scene: named people, named places, dates, times,
   quantities, measurements, prices, part numbers, case numbers, procedures, the technical
   vocabulary the scene's people actually use, and the concrete details of what was done, recorded,
   measured or decided. Aim for several proper nouns and several numbers in every passage.

   Atmosphere is NOT content. Do not write mood, weather, feelings or scene-setting for its own
   sake. "The afternoon light fell across the room and she felt uneasy" is worthless here; "The
   alkalinity read 8.4 dKH on Tuesday, down from 9.1 the week before, and the second dosing pump was
   found to have salt-crept its way out of calibration again" is what this wants. A passage that
   could be dropped into a different scene with three words changed has failed.

2. NO REQUESTS AND NO QUESTIONS. Not one. The passage must not ask for anything, must not tell anyone
   to do anything, and must not end on a question. This is the rule that matters most: the text will
   sit inside a real message to an assistant, and anything that reads as an instruction would be
   acted on. Declarative sentences only. No imperative openings ("Consider...", "Note that...",
   "Imagine..."), no second person addressed to the reader.

3. NOTHING TO DO WITH COMPUTERS. Not one word. No code, commands, file paths, URLs, error messages,
   identifiers, programming, software, applications, servers, databases, APIs, repositories,
   terminals, scripts, algorithms, data, debugging, engineering or IT of any kind, and no modern
   digital devices. The documents this text is appended to are software and assistant chat; the pad
   exists precisely because it shares NO vocabulary with them, and one stray technical word undoes
   that for the passage it appears in. Period-appropriate machinery, tools and the scene's own
   domain jargon are fine and encouraged -- a caliper, a protein skimmer, a ley-line, a quorum.

4. WRITE TO THE GIVEN ASPECT. You are told which aspect of the scene to write about. Stay on it: it
   is what keeps this passage from repeating another one drawn from the same scene.

5. LENGTH. About {{TARGET_WORDS}} words. A passage much shorter than that will be rejected. Use the
   length for more substance, never for more atmosphere.

6. ENGLISH, and plain prose. No headings, no lists, no markdown, no stage directions, no titles.

# Output contract
Return ONE passage wrapped in <passage> tags, and nothing else:

<passage>
...the passage...
</passage>

No preamble, no numbering, no commentary before or after it.

# Example
SCENE: The minutes of a municipal planning and zoning board: attendance, quorum established, a
variance requested for parcel 14-227-03, a neighbour's objection about setbacks, and an item where
the board asks staff to explain a matter fully for the record.
ASPECT: a dispute or disagreement, and how it was settled

<passage>
The November meeting ran forty minutes past its scheduled close, largely on account of item 4(b).
Chair Ndiaye recorded the attendance at seven of nine members, Member Okonkwo having sent regrets and
Member Whitlock arriving after the roll. The applicant for the Larchmere Avenue variance appeared
without counsel and with a survey dated some eleven years earlier, which the Assessor's office
declined to accept for the purpose of establishing the rear setback. A neighbour spoke for the
allotted three minutes about a hedge, and then for two more about drainage, which the Chair permitted
on the grounds that the two were, in her phrase, the same complaint wearing different hats. The
Board's counsel reminded members that the hardship standard is not satisfied by inconvenience alone.
The matter was laid over to the December calendar with the survey to be refreshed at the applicant's
expense, the vote being five in favour, one opposed, and one abstention entered without stated
reason.
</passage>
""".strip()


FRAME_PAD_INPUT_TEMPLATE = """
SCENE: {{SCENE}}
ASPECT: {{ASPECT}}
""".strip()

#: One aspect per passage, cycled over a scene's :data:`FRAME_PAD_PASSAGES_PER_FRAME` requests.
#:
#: Diversity within a scene has to come from the *prompt*, because temperature is 0: asking the same
#: model the same question eight times returns the same passage eight times. Asking for all eight in
#: one reply was the first design and it failed on the cluster -- half the scenes came back with no
#: parseable passage at all, and one long multi-passage completion gives no way to tell a
#: contract-ignoring model from a truncated one. One passage per call is the robust shape: each
#: request is short, independently retried by the client, and independently diagnosable.
FRAME_PAD_ASPECTS = (
    "an inventory, a set of measurements, or a record of quantities taken on one particular day",
    "a dispute or disagreement, and how it was settled",
    "an accident, a breakage, or a near miss, and what it cost",
    "a visitor or newcomer, who they were and what they brought with them",
    "a repair, a restoration, or a piece of routine maintenance done properly",
    "a record of who was present, what was decided, and by what margin",
    "a departure, an ending, or a handover from one person to another",
    "an unusual find, purchase or acquisition, and what was paid for it",
    "a delay, a postponement, or a deadline that was missed",
    "a procedure carried out step by step, exactly as it should be done",
    "a correction to an earlier record, and who noticed the error",
    "a season's or a year's worth of the work, summarised with figures",
)


# --- the passage bank --------------------------------------------------------

def openrouter_chat_class():
    """The shared OpenRouter client class, imported lazily and from exactly one place.

    Lazy so that importing this module -- which the defense registry does at package import --
    never pulls in ``requests``/``python-dotenv`` or asks for an API key; a fully-cached run and the
    offline selftest both need neither.

    In one place because the module has already moved once
    (``prompt_anonymity.utility`` -> :mod:`prompt_anonymity.attacks.llm._openrouter`, when the
    utility judge left OpenRouter), and a defense importing it across package boundaries is exactly
    the kind of caller such a move forgets. The selftest resolves this function, so the next move
    breaks a check that runs in seconds with no key rather than stage 0b of a cluster job that has
    already queued.
    """
    from ..attacks.llm._openrouter import OpenRouterChat

    return OpenRouterChat


def parse_passages(raw_text: str) -> list[str]:
    """Pull the ``<passage>`` blocks out of one reply, in order.

    Unlike :func:`~._backends.extract_tagged_output` this expects *several* tagged blocks and takes
    only closed ones: a final block whose closing tag is missing was cut off by the token cap, and
    half a passage is not one. Blocks that survive are stripped; the caller filters short ones.
    """
    text = (raw_text or "").strip()
    if not text:
        return []
    blocks = re.findall(rf"<{FRAME_PAD_OUTPUT_TAG}>\s*([\s\S]*?)\s*</{FRAME_PAD_OUTPUT_TAG}>",
                        text, re.IGNORECASE)
    return [block.strip() for block in blocks if block.strip()]


def accept_passage(raw_text: str) -> tuple[str | None, str]:
    """One reply -> ``(passage, reason)``; ``passage`` is ``None`` when nothing is usable.

    Three ways in, in order of preference:

    * the text inside ``<passage>`` tags, as the contract asks;
    * failing that, the **whole reply**, when it is long enough to be a passage and carries no stray
      tag. Models drop the wrapper often enough that throwing away good prose over it is silly, and
      here the wrapper carries no information -- one call returns one passage, so there is nothing to
      delimit;
    * a reply with an *opening* tag and no closing one was truncated mid-passage, and half a passage
      is not one. Rejected as ``"stray tag"``.

    The reason string is the diagnosis, and it exists because the first two cluster runs failed with
    no way to tell "the model returned nothing" from "the model ignored the tags".
    """
    text = (raw_text or "").strip()
    if not text:
        return None, "empty"
    tagged = parse_passages(text)
    if tagged:
        best = max(tagged, key=len)
        if len(best) < FRAME_PAD_MIN_PASSAGE_CHARS:
            return None, "too short"
        return best, "tagged"
    if f"<{FRAME_PAD_OUTPUT_TAG}" in text.lower():
        return None, "stray tag"      # opened and never closed -> truncated
    if len(text) < FRAME_PAD_MIN_PASSAGE_CHARS:
        return None, "too short"
    return text, "untagged"


def _save_partial(bank: "PassageBank") -> Path | None:
    """Write the passages generated so far to ``<bank>.partial.json``, best-effort.

    Insurance against losing paid work to a kill, a preemption or a timeout: a build that dies
    mid-way leaves this behind, and ``mv frame_pad_bank.partial.json frame_pad_bank.json`` makes it
    the bank (the defense is happy with a partial one -- see
    :meth:`FramePadDefense.active_framings`). Overwritten each chunk, unlike
    :meth:`PassageBank.save`, which deliberately refuses to clobber a finished bank.
    """
    path = bank_path().with_suffix(".partial.json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(bank.to_json(), handle, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
        return path
    except OSError as error:  # noqa: BLE001 - losing the safety net must not kill the build
        print(f"[frame_pad] could not write the partial bank ({error})")
        return None


def _dump_rejected(rejected: list[dict], model: str, framings, passages_per_frame: int,
                   target_words: int) -> Path:
    """Write the rejected replies beside the bank, so a failed build can be read rather than guessed.

    Best-effort: a build that cannot write its diagnosis still raises with the message, since losing
    the real error to a permissions problem would be the worst of both.
    """
    path = bank_path().with_suffix(".rejected.json")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump({"model": model, "scenes": len(framings),
                       "passages_per_frame": passages_per_frame, "target_words": target_words,
                       "rejected": rejected}, handle, ensure_ascii=False, indent=1)
    except OSError as error:  # noqa: BLE001 - the raise that follows carries the real failure
        print(f"[frame_pad] could not write the rejected replies ({error})")
    return path


class PassageBank:
    """The generated padding text: ``framing key -> passages``, plus what produced it.

    Small enough to hold in memory and to read (``--show-bank``), which is the point -- this is the
    text that will be appended to a whole corpus, so a human should be able to look at all of it.

    :attr:`digest` is a sha256 over the passages alone, in canonical form. It goes into
    :meth:`FramePadDefense.params`, so a run records exactly which bank it padded with and two runs
    can be told apart even if both used "the default bank".
    """

    def __init__(self, passages: dict[str, list[str]], *, model: str = FRAME_PAD_MODEL,
                 target_words: int = FRAME_PAD_TARGET_WORDS):
        # Stripped at construction so a passage has ONE canonical form. Without this the JSON and the
        # CSV disagree on trailing whitespace (the CSV reader strips, the writer does not), the
        # digest changes depending on which file a bank was loaded from, and a bank that round-trips
        # through the readable form is silently not the bank that was written.
        self.passages = {key: tuple(t.strip() for t in texts if t and t.strip())
                         for key, texts in passages.items()
                         if any(t and t.strip() for t in texts)}
        self.model = model
        self.target_words = target_words

    def __len__(self) -> int:
        return sum(len(texts) for texts in self.passages.values())

    @property
    def digest(self) -> str:
        canonical = json.dumps({k: list(v) for k, v in sorted(self.passages.items())},
                               ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]

    def passages_for(self, key: str) -> tuple[str, ...] | None:
        """This scene's passages, or ``None`` when the bank has never heard of it.

        ``None`` rather than a raise so a bank built under an older codebook degrades to "this
        document is not padded" instead of killing a corpus run partway through; the caller reports
        the count.
        """
        return self.passages.get(key)

    def covers(self, framings: tuple[Framing, ...]) -> list[str]:
        """Framing keys this bank has no passages for."""
        return [f.key for f in framings if not self.passages.get(f.key)]

    def to_json(self) -> dict:
        return {
            "model": self.model,
            "target_words": self.target_words,
            "passages_per_frame": max((len(v) for v in self.passages.values()), default=0),
            "digest": self.digest,
            "passages": {key: list(texts) for key, texts in sorted(self.passages.items())},
        }

    @classmethod
    def from_json(cls, payload: dict) -> "PassageBank":
        return cls(payload.get("passages") or {},
                   model=payload.get("model", FRAME_PAD_MODEL),
                   target_words=int(payload.get("target_words", FRAME_PAD_TARGET_WORDS)))

    def to_csv(self, path: Path) -> Path:
        """Write the bank as a CSV: one row per passage, with what it measures beside it.

        The JSON is what the code loads; this is what a person reads and eyeballs -- sort by density
        to find the thin passages, by scene to check diversity.

        Passages contain newlines, which is what CSV quoting is for; ``newline=""`` on the file is
        required for that to round-trip on every platform.
        """
        import csv

        labels = {f.key: f.label for f in FRAMINGS}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["framing", "label", "index", "chars", "words", "density", "passage"])
            for key, texts in sorted(self.passages.items()):
                for index, text in enumerate(texts):
                    writer.writerow([key, labels.get(key, ""), index, len(text),
                                     len(re.findall(r"[\w'’-]+", text)),
                                     f"{passage_density(text):.1f}", text])
        return path

    @classmethod
    def from_csv(cls, path: Path, **kwargs) -> "PassageBank":
        """Read a bank back from the CSV form, so a hand-edited one can be used as-is.

        Rows are grouped by ``framing`` in the order they appear -- the ``index`` column is
        informational, so deleting a row does not leave a hole.
        """
        import csv

        passages: dict[str, list[str]] = {}
        with open(path, encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                text = (row.get("passage") or "").strip()
                if text:
                    passages.setdefault(row["framing"], []).append(text)
        return cls(passages, **kwargs)

    @classmethod
    def load(cls, path: Path) -> "PassageBank":
        """Load a bank from ``.json`` or ``.csv``, by extension."""
        if Path(path).suffix.lower() == ".csv":
            return cls.from_csv(Path(path))
        with open(path, encoding="utf-8") as handle:
            return cls.from_json(json.load(handle))

    def save(self, path: Path) -> None:
        """Write the bank, atomically and without clobbering a bank another process just wrote.

        Both halves matter under a SLURM array: every task resolves the bank, so several may try to
        build one at once, and two tasks padding a corpus from *different* banks would be a silently
        inconsistent dataset. A builder that loses the race keeps the winner's file (the caller
        re-reads from disk afterwards), and the temp-plus-rename means a reader never sees a partial
        file.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            print(f"[frame_pad] {path.name} already exists; keeping it and discarding this build.")
            return
        temporary = path.with_suffix(f".{os.getpid()}.tmp")
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(self.to_json(), handle, ensure_ascii=False, indent=1)
        os.replace(temporary, path)
        # The readable copy, always written beside the canonical one, so "what is in the bank?" is
        # answered by opening a spreadsheet rather than by running a command.
        self.to_csv(path.with_suffix(".csv"))


def bank_path(explicit: str | os.PathLike | None = None) -> Path:
    """Where the bank lives, in precedence order.

    1. an explicit path (``--bank``);
    2. ``$FRAME_PAD_BANK``;
    3. ``<dist>/frame_pad_bank.json`` -- a bank built on this machine, which wins over the shipped
       one so ``--build-bank --force`` still means something;
    4. :data:`PACKAGED_BANK`, the bank committed to the repo.

    Point 4 is the one that matters: the padding text is a fixed codebook, identical for every user,
    corpus and run, exactly like :data:`~.frame_shift.FRAMINGS`. Shipping it means no API key, no
    generation step, and bit-identical padding for anyone who checks out the repo.

    ``data.config`` is imported lazily to avoid a circular import with the defense registry.
    """
    if explicit:
        return Path(explicit).expanduser()
    override = os.environ.get(FRAME_PAD_BANK_ENV)
    if override:
        return Path(override).expanduser()
    from ..data.config import dist_dir

    local = dist_dir() / FRAME_PAD_BANK_FILENAME
    if local.exists() or not PACKAGED_BANK.exists():
        return local          # a locally built bank, or nowhere to build one but here
    return PACKAGED_BANK


def build_bank(model: str = FRAME_PAD_MODEL, *, framings: tuple[Framing, ...] = FRAMINGS,
               passages_per_frame: int = FRAME_PAD_PASSAGES_PER_FRAME,
               target_words: int = FRAME_PAD_TARGET_WORDS) -> PassageBank:
    """Generate the padding passages: one request per scene, all of them concurrently.

    **One call per passage**, not per scene: asking for a whole scene's passages in one reply gives no
    way to tell a contract-ignoring model from a truncated one, so a failure loses the whole scene
    instead of one passage. One short request per passage means each is retried independently.

    Diversity within a scene comes from :data:`FRAME_PAD_ASPECTS`, one aspect per request, because at
    temperature 0 the same prompt returns the same passage every time.

    A reply is accepted three ways, in order: the text inside ``<passage>`` tags; failing that, the
    whole reply, if it is long enough to be a passage and carries no stray tag (models drop the
    wrapper often enough that discarding a good passage over it would be silly); failing that,
    nothing. Whatever is rejected is written to ``<bank>.rejected.json`` so a failure can be read
    rather than guessed at.

    Passages under :data:`FRAME_PAD_MIN_PASSAGE_CHARS` are dropped as stubs. Nothing is rejected on
    *content*: an earlier version filtered on a software-vocabulary word list and lost scenes to their
    own ordinary English. Keeping the pad clear of software is the system prompt's job, checked by a
    human reading ``--show-bank``.

    Raises if a scene ends with no passage at all -- a bank missing a scene would leave every
    document assigned to it unpadded, a silent hole in the arm.
    """
    system_prompt = FRAME_PAD_SYSTEM_PROMPT.replace("{{TARGET_WORDS}}", str(target_words))
    client = openrouter_chat_class()(
        model, system_prompt, temperature=FRAME_PAD_TEMPERATURE, top_p=FRAME_PAD_TOP_P,
        max_workers=FRAME_PAD_MAX_WORKERS, max_retries=FRAME_PAD_MAX_RETRIES,
        timeout=FRAME_PAD_TIMEOUT,
    )
    # One job per (scene, passage), ordered ROUND-ROBIN (every scene's first passage, then every
    # scene's second, ...) so a build killed partway leaves a uniform bank rather than a prefix of
    # fully-covered scenes and untouched ones. The aspect list is cycled; past its length the repeat
    # is numbered so the prompt still differs, since an identical prompt at temperature 0 returns
    # identical text.
    jobs = [(framing, index) for index in range(passages_per_frame) for framing in framings]
    prompts = []
    for framing, index in jobs:
        aspect = FRAME_PAD_ASPECTS[index % len(FRAME_PAD_ASPECTS)]
        if index >= len(FRAME_PAD_ASPECTS):
            aspect += f" (a second, different occasion of this -- number {index + 1})"
        prompts.append(FRAME_PAD_INPUT_TEMPLATE.replace("{{SCENE}}", framing.scene)
                                               .replace("{{ASPECT}}", aspect))
    # ~1.4 tokens per word, tripled for tag overhead and the model's own verbosity. Over-budgeting a
    # completion is free (only generated tokens are billed); under-budgeting truncates the passage.
    budget = int(target_words * 3) + 512

    print(f"[frame_pad] building a bank: {len(framings)} scenes x {passages_per_frame} passages "
          f"~{target_words} words = {len(jobs)} calls at {FRAME_PAD_MAX_WORKERS} concurrent, "
          f"model '{model}'", flush=True)

    passages: dict[str, list[str]] = {f.key: [] for f in framings}
    rejected: list[dict] = []
    untagged = 0
    # In chunks, purely so there is progress on stdout: one all-or-nothing batch of 400 prints
    # nothing for as long as it takes, and a slow build then looks exactly like a hung one.
    chunk_size = max(1, FRAME_PAD_PROGRESS_EVERY)
    started = time.time()
    for offset in range(0, len(jobs), chunk_size):
        chunk = jobs[offset:offset + chunk_size]
        replies = client.complete_batch(prompts[offset:offset + chunk_size], budget)
        for (framing, index), reply in zip(chunk, replies):
            passage, reason = accept_passage(reply)
            if passage is None:
                rejected.append({"framing": framing.key, "aspect_index": index, "reason": reason,
                                 "reply": (reply or "")[:2000]})
                continue
            untagged += reason == "untagged"
            passages[framing.key].append(passage)
        done = offset + len(chunk)
        elapsed = time.time() - started
        remaining = (len(jobs) - done) * elapsed / done  # seconds, at the rate so far
        # The reason breakdown is on every line, not just at the end: "25 rejected" tells you
        # something is wrong and nothing about what, and waiting out a 20-minute build to find out
        # is exactly the loop this reporting exists to break. 'empty'/'stray tag' both mean the
        # model spent its budget without producing a closed passage -- raise the budget or drop to
        # a non-reasoning model; 'too short' means it wrote a stub.
        why = Counter(entry["reason"] for entry in rejected)
        print(f"[frame_pad] {done}/{len(jobs)} calls | "
              f"{sum(len(v) for v in passages.values())} passages, {len(rejected)} rejected"
              + (" (" + ", ".join(f"{r}: {n}" for r, n in why.most_common()) + ")" if why else "")
              + f" | {elapsed / 60:.1f} min elapsed"
              + (f", ~{remaining / 60:.1f} min left" if done < len(jobs) else ""), flush=True)
        if rejected and offset == 0:
            # One real example, once, so the first chunk already shows what a bad reply looks like.
            first = rejected[0]
            print(f"[frame_pad] first rejection ({first['reason']}, {first['framing']}): "
                  f"{first['reply'][:300]!r}", flush=True)
        # Keep what has been paid for. Two builds have now lost several hundred generated passages
        # -- one to an exception, one to a kill -- and every call here is money already spent, so the
        # work in hand goes to disk after every chunk. Written beside the bank rather than to it, so
        # a completed build's atomic write is still the thing that creates the real file.
        _save_partial(PassageBank(passages, model=model, target_words=target_words))

    bank = PassageBank(passages, model=model, target_words=target_words)
    missing = bank.covers(framings)
    # Always keep the rejected replies, not only when a scene ends up empty: a build that "worked"
    # while throwing away half its calls is a build whose replies someone needs to read.
    dump = _dump_rejected(rejected, model, framings, passages_per_frame, target_words) if rejected \
        else None
    if rejected:
        why = Counter(entry["reason"] for entry in rejected)
        print(f"[frame_pad] {len(rejected)} of {len(jobs)} calls produced nothing usable "
              + ", ".join(f"{reason}: {n}" for reason, n in why.most_common()))
    if untagged:
        print(f"[frame_pad] {untagged} reply/replies omitted the <passage> tags and were kept whole")
    if not len(bank):
        raise SystemExit(
            f"every one of the {len(jobs)} calls failed -- not one usable passage.\n"
            f"The replies are in {dump} -- READ THEM. 'empty' means the model returned nothing at "
            f"all, 'too short' a stub, 'stray tag' a reply truncated mid-passage (raise the token "
            f"budget, or the model is spending it on reasoning). Then try another --model."
        )
    if missing:
        # NOT fatal, and this used to be: a build that lost some scenes raised before saving and
        # threw away every passage it had paid for. A partial bank is usable -- the defense restricts
        # its draw to the covered scenes (see FramePadDefense.active_framings), so every document is
        # still padded and only pad diversity suffers -- so it is written and the gap is reported.
        print(f"[frame_pad] {len(missing)} of {len(framings)} scene(s) got no usable passage: "
              f"{', '.join(missing)}")
        print(f"[frame_pad] keeping the bank anyway -- the draw restricts to the "
              f"{len(framings) - len(missing)} covered scenes, so every document is still padded. "
              f"The rejected replies are in {dump} if you want to know why they failed.")
    short = [f"{f.key} ({len(passages[f.key])})" for f in framings
             if len(passages[f.key]) < passages_per_frame]
    if short:
        print(f"[frame_pad] note: {len(short)} scene(s) hold fewer than {passages_per_frame} "
              f"passages: {', '.join(short)}")
    scores = bank_quality(bank)
    print(f"[frame_pad] bank built: {len(bank):,} passages, digest {bank.digest}; "
          f"density {scores['density']:.1f} content markers/100 words "
          f"({scores['thin']} thin), {scores['questions']} with a question, "
          f"{scores['imperatives']} imperative, {scores['technical']} technical")
    if any(scores[k] for k in ("thin", "questions", "imperatives", "technical")):
        print("[frame_pad] some passages break the contract -- read them with --show-bank before "
              "padding a corpus. A thin bank makes this arm indistinguishable from 'documents got "
              "longer', and a passage that asks for something will be answered.")
    return bank


def resolve_bank(path: Path, *, framings: tuple[Framing, ...] = FRAMINGS,
                 model: str = FRAME_PAD_MODEL,
                 passages_per_frame: int = FRAME_PAD_PASSAGES_PER_FRAME,
                 target_words: int = FRAME_PAD_TARGET_WORDS, force: bool = False) -> PassageBank:
    """The bank at ``path``, built and written first if it is not there (or ``force``).

    Building here is a convenience for local use; a cluster run should build it explicitly in a setup
    stage (``--build-bank``) so that the paid step happens once, visibly, before an array fans out.
    Whatever happens, the bank is re-read from disk at the end, so the returned object is the one on
    disk -- including when another process won the race to write it.
    """
    if force or not path.exists():
        build_bank(model, framings=framings, passages_per_frame=passages_per_frame,
                   target_words=target_words).save(path)
    bank = PassageBank.load(path)
    # Print the source path so a coverage gap is traceable to a specific bank file, not just a count.
    print(f"[frame_pad] bank: {len(bank):,} passages over {len(bank.passages)} scene(s), "
          f"digest {bank.digest} <- {path}")
    missing = bank.covers(framings)
    if missing:
        print(f"[frame_pad] note: {path} has no passages for {len(missing)} of {len(framings)} "
              f"scene(s) ({', '.join(missing[:3])}{'...' if len(missing) > 3 else ''}). The draw is "
              f"restricted to the covered scenes, so every document is still padded -- the cost is "
              f"pad diversity, not coverage. `--build-bank --force` rebuilds the whole bank.")
    return bank


# --- rewrite quality, measured ------------------------------------------------

#: Openings that would make a passage read as an instruction to the assistant rather than as prose.
IMPERATIVE_OPENERS = ("please ", "consider ", "note that", "imagine ", "write ", "explain ",
                      "describe ", "list ", "tell ", "give ", "make ", "create ", "help ")

#: Markup that has no business in a passage: fenced or inline code, URLs, function calls, source
#: filenames. This is a *markup* pattern rather than a software-vocabulary word list on purpose: an
#: ordinary word like "code", "data", "variable" or "terminal" has many non-software senses, so a word
#: list produces false positives that delete a whole scene's passages instead of catching a leak.
#: Keeping software vocabulary out of the pad is the system prompt's job (rule 3), checked by a human
#: reading ``--show-bank``.
CODEISH_PATTERN = re.compile(r"`|https?://|\w+\(\)|\w+\.(?:py|js|sh|json)\b")

#: Content markers per 100 words below which a passage counts as *thin*: mood and scene-setting
#: rather than substance. Calibrated so that ordinary prose passes and only genuinely atmospheric
#: writing ("the light fell across the room and she felt uneasy") is flagged.
DENSITY_FLOOR = 5.0


#: Numbers written as words. Without these the measure is a register test rather than a content
#: test: a zoning board writes "41 pounds" and a Socratic dialogue writes "some fourteen feet", and
#: only the first has a digit in it -- so genuinely dense prose in a narrative register would
#: otherwise be scored as thin.
NUMBER_WORDS = (
    "one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|"
    "sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|"
    "hundred|thousand|million|dozen|score|half|quarter|third|fourth|fifth|sixth|seventh|eighth|"
    "ninth|tenth|first|second|twelfth|twentieth|thirtieth"
)
NUMBER_WORD_PATTERN = re.compile(rf"\b(?:{NUMBER_WORDS})(?:s|th|ths)?\b", re.IGNORECASE)


def passage_density(text: str) -> float:
    """Content markers per 100 words: proper nouns plus numbers, written either way.

    A stand-in for "how much substance is in here", and a deliberately crude one -- what it has to
    separate is a passage full of named people, parcel numbers, dosages and dates from a passage of
    weather and feelings, and those two are far apart on this measure. Words capitalised anywhere but
    at the start of a sentence count as proper nouns; anything containing a digit counts as a number.

    Density is the property this defense's padding lives or dies by. The pad exists to put *topical
    content* between the reader and the author's own words, the way each of the 50 scenes carries
    real subject matter; a pad made of mood adds length and nothing else, and length alone is the
    null hypothesis this arm is supposed to beat.
    """
    words = re.findall(r"[\w'’-]+", text)
    if not words:
        return 0.0
    proper = 0
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        tokens = re.findall(r"[\w'’-]+", sentence)
        proper += sum(1 for token in tokens[1:] if token[:1].isupper())
    numeric = sum(1 for word in words if any(character.isdigit() for character in word))
    numeric += len(NUMBER_WORD_PATTERN.findall(text))
    return 100.0 * (proper + numeric) / len(words)


def bank_quality(bank: PassageBank) -> dict:
    """Mechanical checks over a whole bank, for ``--build-bank``/``--show-bank`` to print.

    Each one counts violations of a rule that decides whether this arm measures what it claims:

    * ``questions`` -- passages containing a question mark. A padded document that asks a question is
      a document whose *utility* judgement changes, because the assistant will answer the padding.
    * ``imperatives`` -- passages opening in the imperative, same failure by a different route.
    * ``technical`` -- passages carrying code markup: backticks, URLs, function calls, source
      filenames (:data:`CODEISH_PATTERN`). A pointer to passages worth reading first; software
      *vocabulary* is not machine-checked (see :data:`CODEISH_PATTERN`), so ``--show-bank`` is the
      real check.
    * ``thin`` / ``density`` -- passages under :data:`DENSITY_FLOOR` content markers per 100 words
      (:func:`passage_density`). A thin bank turns this defense into "documents got longer", which is
      the one result it must not be confounded with.
    """
    texts = [p for passages in bank.passages.values() for p in passages]
    if not texts:
        return {"passages": 0, "questions": 0, "imperatives": 0, "technical": 0, "thin": 0,
                "density": 0.0, "min_chars": 0, "median_chars": 0, "max_chars": 0}
    lengths = sorted(len(t) for t in texts)
    densities = [passage_density(t) for t in texts]
    return {
        "passages": len(texts),
        "questions": sum("?" in t for t in texts),
        "imperatives": sum(t.lstrip().lower().startswith(IMPERATIVE_OPENERS) for t in texts),
        "technical": sum(bool(CODEISH_PATTERN.search(t)) for t in texts),
        "thin": sum(density < DENSITY_FLOOR for density in densities),
        "density": sum(densities) / len(densities),
        "min_chars": lengths[0],
        "median_chars": lengths[len(lengths) // 2],
        "max_chars": lengths[-1],
    }


# --- defense ----------------------------------------------------------------

class FramePadDefense:
    """Append one shared, off-topic turn to each document, drawn by a keyed hash of its ``doc_id``.

    A **plain** defense, not a :class:`~prompt_anonymity.defenses.base.CachedDefense`: padding is a
    dictionary lookup, so a disk cache would cost a table write per corpus and save nothing.

    It is also the only defense here that changes a document's turn *count*, which the per-turn
    rewrite path structurally cannot do (its rows are turns, and it must return one output per row).
    :attr:`appends_turns` is how it says so; ``apply_defenses`` reads that flag and calls
    :meth:`extra_turns` per document instead of running the per-turn path. Applied directly to an
    :class:`~prompt_anonymity.core.AttackData` whose rows are whole conversation *cells*
    (:data:`~._backends.TURN_DELIM`-joined), :meth:`__call__` appends the same turn inside the cell,
    which is the same operation in the other encoding.

    Parameters
    ----------
    model, passages_per_frame, target_words : str, int, int
        Recorded, and used only when a bank has to be built. They do not change an existing bank.
    seed : int
        Master seed for both draws. Shared with ``frame_shift``'s ``"framing"`` namespace on purpose:
        at equal seeds a document lands in the same scene under both defenses.
    single_framing : str or None
        Force one scene on the whole corpus -- the ``frame_pad_single`` ablation, the
        convergence-vs-dilution control. With one scene the corpus draws from P pads instead of
        K x P, so it is also the "how much does pad diversity matter" arm.
    framings : tuple of Framing
        The scene codebook, shared with ``frame_shift``.
    bank : PassageBank or None
        A pre-built bank, mostly for tests. ``None`` resolves (and if necessary builds) one lazily on
        first use, so importing the registry never reads a file or spends money.
    bank_path : path or None
        Override the bank location (see :func:`bank_path`).
    """

    name = "frame_pad"
    version = "1"
    #: Read by ``apply_defenses``: this defense adds turns to a document rather than rewriting them,
    #: so the per-turn rewrite path does not apply to it.
    appends_turns = True

    def __init__(self, *, model: str = FRAME_PAD_MODEL, seed: int = FRAME_PAD_SEED,
                 single_framing: str | None = None,
                 passages_per_frame: int = FRAME_PAD_PASSAGES_PER_FRAME,
                 target_words: int = FRAME_PAD_TARGET_WORDS,
                 framings: tuple[Framing, ...] = FRAMINGS,
                 bank: PassageBank | None = None, bank_path=None):
        if not framings:
            raise ValueError("frame_pad needs at least one framing.")
        if single_framing is not None and single_framing not in {f.key for f in framings}:
            raise ValueError(f"unknown framing {single_framing!r}; "
                             f"available: {sorted(f.key for f in framings)}")
        self.model = model
        self.seed = seed
        self.single_framing = single_framing
        self.passages_per_frame = passages_per_frame
        self.target_words = target_words
        self.framings = framings
        self._by_key = {f.key: f for f in framings}
        self._bank = bank
        self._bank_path = bank_path
        self._active: tuple[Framing, ...] | None = None   # see active_framings()
        self.padded = 0
        self.unpadded = 0

    def params(self) -> dict:
        """What this run padded with. Not a cache key (nothing here is cached) -- a record, printed
        by the CLI and stored beside a manifest, so a defended corpus can be traced to its bank."""
        return {
            "model": self.model,
            "seed": self.seed,
            "single_framing": self.single_framing,
            "passages_per_frame": self.passages_per_frame,
            "target_words": self.target_words,
            "framings": [f.key for f in self.framings],
            "bank_digest": self.bank().digest,
        }

    # --- the bank ---

    def bank(self) -> PassageBank:
        """The passage bank, resolved (and built if absent) on first use."""
        if self._bank is None:
            self._bank = resolve_bank(
                bank_path(self._bank_path), framings=self.framings, model=self.model,
                passages_per_frame=self.passages_per_frame, target_words=self.target_words,
            )
        return self._bank

    # --- assignment ---

    def active_framings(self) -> tuple[Framing, ...]:
        """The scenes documents are actually assigned to: the codebook, minus any the bank misses.

        A partial bank is the normal case, not an error -- a build where some calls fail writes what
        it got -- and the alternative to restricting here is worse than it looks: documents drawn to
        an unbanked scene would get **no pad at all**, so the defended arm would silently contain a
        slice of undefended documents and under-report the defense's effect.

        Restricting changes the draw's modulus, so the scene assignment no longer matches
        ``frame_shift``'s at the same seed. That is a real loss (it is what made the two arms
        comparable document by document) and it is why this says so out loud, once.
        """
        if self._active is None:
            bank = self.bank()
            covered = tuple(f for f in self.framings if bank.passages_for(f.key))
            if not covered:
                raise SystemExit(
                    f"the passage bank has no passages for any of this defense's "
                    f"{len(self.framings)} scenes; build one with "
                    f"`python -m prompt_anonymity.defenses.frame_pad --build-bank`."
                )
            if len(covered) != len(self.framings):
                print(f"[frame_pad] the bank covers {len(covered)} of {len(self.framings)} scenes; "
                      f"assignment is restricted to those, so every document still gets a pad. "
                      f"Note the scene draw no longer lines up with frame_shift's at this seed.")
            self._active = covered
        return self._active

    def framing_for(self, doc_id) -> Framing:
        """The scene this document's pad comes from: a keyed hash of ``(seed, "framing", doc_id)``.

        The same key as :meth:`~.frame_shift.FrameShiftDefense.framing_for`, so with a **complete**
        bank the two defenses agree document by document and a comparison between them holds the
        scene fixed. See :meth:`active_framings` for what a partial bank costs.
        """
        if self.single_framing is not None:
            return self._by_key[self.single_framing]
        active = self.active_framings()
        return active[keyed_rng(self.seed, "framing", doc_id).randrange(len(active))]

    def passage_for(self, doc_id) -> str | None:
        """The padding text for this document, or ``None`` when its scene is missing from the bank.

        A second, independent draw (namespace ``"passage"``) over the scene's passages: independent
        so that two documents sharing a scene do not automatically share a passage, which is what
        keeps the pad from being a perfect indicator of the scene.
        """
        framing = self.framing_for(doc_id)
        passages = self.bank().passages_for(framing.key)
        if not passages:
            return None
        return passages[keyed_rng(self.seed, "passage", doc_id).randrange(len(passages))]

    # --- the two application paths ---

    def extra_turns(self, doc_id) -> list[str]:
        """Turns to append to this document -- exactly one, or none if its scene is unbanked.

        Called by ``apply_defenses`` once per document, after the (here, non-existent) per-turn pass.
        The document's own turns are never seen and never touched.
        """
        passage = self.passage_for(document_id(doc_id))
        if passage is None:
            self.unpadded += 1
            return []
        self.padded += 1
        return [passage]

    def __call__(self, data):
        """Pad an :class:`~prompt_anonymity.core.AttackData` whose rows are conversation cells.

        The other application path (``apply_defenses`` uses :meth:`extra_turns`). A cell is turns
        joined by :data:`~._backends.TURN_DELIM`, so appending a turn is appending the delimiter and
        the passage. Only the unknown side is padded -- the same threat model every rewrite defense
        here uses: the adversary's known documents are already released.
        """
        from dataclasses import replace

        if data.unknown_texts is None:
            raise ValueError("defense 'frame_pad' needs unknown_texts; load the dataset with text.")
        ids = (data.unknown_ids if data.unknown_ids is not None
               else np.arange(len(data.unknown_texts)))
        padded = []
        for text, row_id in zip(data.unknown_texts, ids):
            extra = self.extra_turns(row_id)
            padded.append(f"{text}{TURN_DELIM}{extra[0]}" if extra else str(text))
        return replace(data, unknown_texts=np.asarray(padded, dtype=object))

    def report(self) -> None:
        """Print what was padded. Called by ``apply_defenses`` when the pass finishes."""
        total = self.padded + self.unpadded
        if not total:
            return
        print(f"[frame_pad] appended a turn to {self.padded:,} of {total:,} documents"
              + (f"; {self.unpadded:,} were assigned a scene missing from the bank and were NOT "
                 f"padded" if self.unpadded else ""))


def frame_pad_manifest(doc_ids, defense: FramePadDefense):
    """``doc_id -> (scene, pad)`` for a corpus, rebuilt from the identifier list alone.

    The analysis counterpart of :func:`~.frame_shift.frame_shift_manifest`, with the passage's own
    hash as well as the scene: the question this arm has to answer is whether an attacker is
    recovering the *pad* rather than the author, and the pad -- not just the scene -- is the group
    that would make that happen (K x P groups, not K).
    """
    import pandas as pd

    rows = []
    for doc_id in doc_ids:
        framing = defense.framing_for(doc_id)
        passage = defense.passage_for(doc_id)
        rows.append((str(doc_id), framing.key, framing.label,
                     hashlib.sha256((passage or "").encode("utf-8")).hexdigest()[:12],
                     len(passage or "")))
    return pd.DataFrame(rows, columns=["doc_id", "framing", "label", "pad_id", "pad_chars"])


# --- self-test ---------------------------------------------------------------

def _fake_bank(framings: tuple[Framing, ...] = FRAMINGS, per_frame: int = 4) -> PassageBank:
    """A synthetic bank, so every offline check runs with no API key and no network.

    Passages are numbered by (scene, position) rather than built from the scene's own label: they
    have to be distinct from each other for the assignment checks to mean anything, while carrying
    no real vocabulary that the quality measures could trip over.
    """
    return PassageBank({f.key: [f"Scene {n} passage {i}. " + "filler " * 40
                                for i in range(per_frame)]
                        for n, f in enumerate(framings)})


def _cells(texts: list[str], ids: list[str]):
    """An :class:`~prompt_anonymity.core.AttackData` of conversation cells, for the checks below."""
    from ..core import AttackData

    return AttackData(
        known_embeddings=np.zeros((0, 0), dtype=np.float32),
        unknown_embeddings=np.zeros((len(texts), 0), dtype=np.float32),
        known_labels=np.empty(0, dtype=object),
        unknown_labels=np.asarray([f"author-{i}" for i in range(len(texts))], dtype=object),
        unknown_texts=np.asarray(texts, dtype=object),
        unknown_ids=np.asarray(ids, dtype=object),
    )


def _selftest() -> None:
    """The invariant checks, as a runnable command -- offline, no API key needed.

    This repo has no test framework and no pytest dependency, so the checks live here rather than
    introducing one (the same choice ``collision_seeding`` and ``frame_shift`` made). Every check is
    a property this defense's correctness rests on.
    """
    from ..data.apply_defenses import TURN_ID_SEPARATOR as PIPELINE_SEPARATOR
    from ._backends import TURN_ID_SEPARATOR
    from .frame_shift import FrameShiftDefense

    bank = _fake_bank()
    defense = FramePadDefense(seed=7, bank=bank)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  -- {detail}'}")
        if not condition:
            failures.append(name)

    # 1. The turn-id separator still agrees with the pipeline that produces the ids.
    check("turn id separator matches apply_defenses", TURN_ID_SEPARATOR == PIPELINE_SEPARATOR,
          f"{TURN_ID_SEPARATOR!r} != {PIPELINE_SEPARATOR!r}")

    # 1b. The OpenRouter client is still where this module thinks it is. Resolving it costs
    #     milliseconds and needs no key, and it is the difference between finding out here and
    #     finding out in stage 0b of a queued cluster job -- which is exactly how this broke once,
    #     when the client moved out of prompt_anonymity.utility.
    try:
        client_class = openrouter_chat_class()
    except Exception as error:  # noqa: BLE001 - reporting the failure IS the check
        check("the OpenRouter client resolves", False, f"{type(error).__name__}: {error}")
    else:
        check("the OpenRouter client resolves", hasattr(client_class, "complete_batch"),
              f"{client_class!r} has no complete_batch")

    # 2. The defense declares itself to the pipeline. Without this flag apply_defenses would send it
    #    down the per-turn path, where adding a turn raises -- so the flag IS the integration.
    check("declares appends_turns", FramePadDefense.appends_turns is True)
    check("exposes extra_turns", callable(getattr(defense, "extra_turns", None)))

    # 3. Exactly one turn is appended, and it is real text.
    turns = defense.extra_turns("doc-1")
    check("appends exactly one turn", len(turns) == 1, f"{len(turns)}")
    check("the appended turn is substantial", len(turns[0]) >= 100, f"{len(turns[0])} chars")

    # 4. Assignment is deterministic, seed-sensitive, and depends only on the doc_id.
    check("assignment is deterministic", defense.extra_turns("doc-1") == defense.extra_turns("doc-1"))
    check("assignment depends on the seed",
          FramePadDefense(seed=8, bank=bank).extra_turns("doc-1") != defense.extra_turns("doc-1")
          or FramePadDefense(seed=9, bank=bank).extra_turns("doc-1") != defense.extra_turns("doc-1"))

    # 5. Every turn id of one document resolves to the same document, and so to one pad. This is what
    #    makes the pad a property of the document rather than of whichever row was seen.
    pads = {tuple(defense.extra_turns(f"doc-42{TURN_ID_SEPARATOR}{t}")) for t in range(9)}
    check("all rows of a document give one pad", len(pads) == 1, f"{len(pads)} pads")

    # 5b. The pad is a function of the ID ONLY -- never of the text. Two documents that say exactly
    #     the same thing must get unrelated pads, or the padding would reinforce the topical signal
    #     it exists to bury. Checked at the cell path, the only path that sees text at all.
    same_text = ["identical text here" for _ in range(2)]
    cell_pads = [str(out).split(TURN_DELIM)[-1] for out in
                 FramePadDefense(seed=7, bank=bank)(_cells(same_text, ["doc-a", "doc-b"])).unknown_texts]
    check("identical documents get different pads", cell_pads[0] != cell_pads[1])
    differing = [str(out).split(TURN_DELIM)[-1] for out in
                 FramePadDefense(seed=7, bank=bank)(_cells(["one thing", "a completely other thing"],
                                                          ["doc-a", "doc-a"])).unknown_texts]
    check("the same id gives the same pad whatever the text", differing[0] == differing[1])

    # 6. Documents spread evenly over the codebook, and over the passages within a scene -- an
    #    unreachable passage would shrink the collision space without saying so.
    counts = Counter(defense.framing_for(f"doc-{i}").key for i in range(20_000))
    expected = 20_000 / len(FRAMINGS)
    check("all framings are reachable", len(counts) == len(FRAMINGS), f"{len(counts)} used")
    check("assignment is roughly uniform",
          max(abs(n - expected) for n in counts.values()) < 0.25 * expected,
          f"expected {expected:.0f} each, range {min(counts.values())}-{max(counts.values())}")
    single = FramePadDefense(seed=7, single_framing=SINGLE_FRAMING_KEY, bank=bank)
    drawn = {single.passage_for(f"doc-{i}") for i in range(2_000)}
    check("every passage of a scene is reachable",
          len(drawn) == len(bank.passages_for(SINGLE_FRAMING_KEY)), f"{len(drawn)} distinct pads")

    # 7. Shard invariance: a document's pad does not depend on which shard it landed in, which is
    #    what lets apply_defenses run as a SLURM array.
    every = [defense.passage_for(f"doc-{i}") for i in range(500)]
    sharded = {i: defense.passage_for(f"doc-{i}")
               for offset in range(4) for i in range(offset, 500, 4)}
    check("shard-invariant", every == [sharded[i] for i in range(500)])

    # 8. The scene draw agrees with frame_shift's at the same seed -- the property that makes the two
    #    arms comparable document by document.
    shift = FrameShiftDefense(seed=7)
    check("scene assignment matches frame_shift",
          all(defense.framing_for(f"doc-{i}") == shift.framing_for(f"doc-{i}") for i in range(500)))

    # 9. The single-scene ablation collapses the codebook, and an unknown key is rejected at
    #    construction rather than hours into a run.
    check("single ablation uses one scene",
          len({single.framing_for(f"doc-{i}").key for i in range(1000)}) == 1)
    try:
        FramePadDefense(single_framing="not_a_frame", bank=bank)
    except ValueError:
        check("unknown framing key is rejected", True)
    else:
        check("unknown framing key is rejected", False, "no ValueError raised")

    # 10. A PARTIAL bank still pads every document. This is the normal case -- a build where some
    #     calls fail writes what it got -- and the failure it guards against is the quiet one: if
    #     documents drawn to an unbanked scene went unpadded, the defended arm would carry a slice of
    #     undefended documents and under-report the defense.
    covered = {FRAMINGS[0].key: ["x" * 300], FRAMINGS[3].key: ["y" * 300]}
    partial = FramePadDefense(seed=7, bank=PassageBank(covered))
    outputs = [partial.extra_turns(f"doc-{i}") for i in range(2_000)]
    check("a partial bank still pads every document",
          all(len(o) == 1 for o in outputs), f"{sum(1 for o in outputs if not o)} unpadded")
    check("a partial bank draws only from covered scenes",
          {partial.framing_for(f"doc-{i}").key for i in range(2_000)} == set(covered))
    empty = FramePadDefense(seed=7, bank=PassageBank({}))
    try:
        empty.extra_turns("doc-1")
    except SystemExit:
        check("an empty bank fails loudly", True)
    else:
        check("an empty bank fails loudly", False, "no SystemExit raised")

    # 11. The conversation-cell path appends a turn in the cell encoding, leaving the original text
    #     an exact prefix -- "the user's words are untouched" is the whole claim of this defense.
    from ._backends import split_turns

    cells = ["first turn" + TURN_DELIM + "second turn", "only turn"]
    padded = FramePadDefense(seed=7, bank=bank)(_cells(cells, ["doc-1", "doc-2"]))
    check("cell path preserves the original text exactly",
          all(str(out).startswith(cell) for cell, out in zip(cells, padded.unknown_texts)))
    check("cell path adds exactly one turn",
          all(len(split_turns(str(out))) == len(split_turns(cell)) + 1
              for cell, out in zip(cells, padded.unknown_texts)))
    check("cell path keeps the row count", len(padded.unknown_texts) == len(cells))

    # 12. Reply parsing, including the truncated last block a token cap produces.
    reply = "<passage>\none\n</passage>\n<passage>\ntwo\n</passage>"
    check("passages parse", parse_passages(reply) == ["one", "two"])
    check("a truncated final passage is dropped",
          parse_passages(reply + "<passage>\nhalf a pas") == ["one", "two"])
    check("an empty reply parses to nothing", parse_passages("") == [])

    # 12b. accept_passage is where a build succeeds or silently loses a scene. The untagged branch is
    #      the important one: without it, a model dropping its output tags would look identical to a
    #      model producing nothing at all.
    body = "The Assessor recorded nine entries on 14 November, of which three were later struck. " * 4
    cases = {
        "tagged": f"<passage>\n{body}\n</passage>",
        "untagged": body,
        "stray tag": f"<passage>\n{body[:80]}",
        "empty": "   ",
        "too short": "<passage>\nstub\n</passage>",
    }
    for expected, reply in cases.items():
        passage, reason = accept_passage(reply)
        check(f"accept_passage: {expected}",
              reason == expected and (passage is None) == (expected in ("stray tag", "empty",
                                                                        "too short")),
              f"got ({'None' if passage is None else f'{len(passage)} chars'}, {reason!r})")
    check("accept_passage keeps the untagged text verbatim",
          accept_passage(body)[0] == body.strip())

    # 12c. One aspect per passage is the only thing making a scene's passages differ at temperature 0,
    #      so there must be enough of them to cover a bank's worth, and they must be distinct.
    check("there are at least as many aspects as passages per scene",
          len(FRAME_PAD_ASPECTS) >= FRAME_PAD_PASSAGES_PER_FRAME,
          f"{len(FRAME_PAD_ASPECTS)} aspects, {FRAME_PAD_PASSAGES_PER_FRAME} passages")
    check("aspects are distinct", len(set(FRAME_PAD_ASPECTS)) == len(FRAME_PAD_ASPECTS))

    # 13. The bank round-trips through JSON with a stable digest -- the digest is what identifies
    #     which text a defended corpus was padded with.
    reloaded = PassageBank.from_json(bank.to_json())
    check("bank round-trips", reloaded.passages == bank.passages)
    check("digest is stable", reloaded.digest == bank.digest)
    check("digest tracks the passages",
          PassageBank({"a": ["x" * 300]}).digest != PassageBank({"a": ["y" * 300]}).digest)

    # 14. The quality measures discriminate -- they are what a human reads before trusting a bank.
    bad = PassageBank({"a": ["Please consider the ledger? See `df.head()` at https://x.example " * 9]})
    scores = bank_quality(bad)
    check("quality: a question is caught", scores["questions"] == 1)
    check("quality: an imperative opening is caught", scores["imperatives"] == 1)
    check("quality: technical content is caught", scores["technical"] == 1)

    # 14b. The code-markup check flags markup and NOTHING ELSE -- a vocabulary-based filter would
    #      catch ordinary English that happens to use software-adjacent words, so every one of these
    #      sentences must pass.
    scene_english = ("The operator tapped out the message in Morse code.",
                     "The data from the 2011 record is still disputed at the reservoir hide.",
                     "Solve for the variable x in the third exercise.",
                     "The branch line ends at the terminal beside the goods shed.",
                     "She sat at the keyboard and played the third inversion.",
                     "The host went off script for a full minute.",
                     "The server brought the second course.",
                     "The framework of the altarpiece is original.",
                     "The regiment was ordered to deploy at first light.",
                     "The patent application was filed in 1911, and the concert program lists four.")
    flagged = [s for s in scene_english if CODEISH_PATTERN.search(s)]
    check("ordinary scene English is never flagged", not flagged, str(flagged))
    check("code markup is flagged",
          all(CODEISH_PATTERN.search(s) for s in
              ("see `df.head()`", "at https://x.example", "in config.py", "call render()")))
    clean = bank_quality(bank)
    check("quality: a clean bank scores clean",
          clean["questions"] == 0 and clean["imperatives"] == 0 and clean["technical"] == 0,
          f"{clean}")

    # 15. Density separates content from atmosphere. This is the measure that decides whether the
    #     pad is doing what the arm claims (adding topical CONTENT) or merely adding length, so a
    #     measure that scored mood-writing as content would endorse exactly the wrong bank.
    dense = ("Chair Ndiaye recorded attendance at seven of nine members. The Larchmere Avenue "
             "variance for parcel 14-227-03 was laid over to the December calendar, the survey of "
             "2014 having been refused by the Assessor, on a vote of five to one with one "
             "abstention.")
    airy = ("The light fell slowly across the room and everything felt very still. She waited for a "
            "while, thinking about nothing in particular, and then she waited a little longer while "
            "the quiet settled over everything around her again.")
    check("density: content-heavy prose scores well above the floor",
          passage_density(dense) > DENSITY_FLOOR * 1.5, f"{passage_density(dense):.1f}")
    check("density: atmosphere scores below the floor",
          passage_density(airy) < DENSITY_FLOOR, f"{passage_density(airy):.1f}")
    check("density: empty text is safe", passage_density("") == 0.0)
    check("quality: thin passages are counted",
          bank_quality(PassageBank({"a": [airy], "b": [dense]}))["thin"] == 1)
    # The prompt's own worked example must pass the bar the prompt sets -- it is the only concrete
    # picture the model gets of what "dense" means. (The LAST tagged block: the earlier ones belong
    # to the output-contract template.)
    example = parse_passages(FRAME_PAD_SYSTEM_PROMPT)[-1]
    check("the shipped example passage clears the floor",
          passage_density(example) > DENSITY_FLOOR, f"{passage_density(example):.1f}")
    shipped = bank_quality(PassageBank({"a": [example]}))
    check("the shipped example breaks none of the bank rules",
          not any(shipped[rule] for rule in ("questions", "imperatives", "technical", "thin")),
          f"{ {rule: shipped[rule] for rule in ('questions', 'imperatives', 'technical', 'thin')} }")

    # 15b. The SHIPPED bank, if there is one. This is the text that actually gets appended to a
    #      corpus, so it is checked here rather than trusted: full scene coverage (a gap silently
    #      shrinks the codebook), enough passages to be a collision space rather than a fingerprint,
    #      and the four contract properties -- a pad that asks a question gets answered, and a thin
    #      one reduces this defense to "the documents got longer".
    if PACKAGED_BANK.exists():
        shipped = PassageBank.load(PACKAGED_BANK)
        gaps = shipped.covers(FRAMINGS)
        quality = bank_quality(shipped)
        check("shipped bank covers every scene", not gaps,
              f"missing: {', '.join(gaps[:5])}")
        check("shipped bank has at least 3 passages per scene",
              all(len(v) >= 3 for v in shipped.passages.values()),
              f"min {min((len(v) for v in shipped.passages.values()), default=0)}")
        check("shipped bank asks nothing", quality["questions"] == 0, f"{quality['questions']}")
        check("shipped bank instructs nothing", quality["imperatives"] == 0,
              f"{quality['imperatives']}")
        check("shipped bank carries no code markup", quality["technical"] == 0,
              f"{quality['technical']}")
        check("shipped bank is content, not atmosphere",
              quality["thin"] == 0 and quality["density"] >= DENSITY_FLOOR,
              f"{quality['thin']} thin, mean density {quality['density']:.1f}")
        check("shipped bank has no duplicate passages",
              len({p for v in shipped.passages.values() for p in v}) == len(shipped),
              f"{len(shipped)} passages")
    else:
        print("  --    no shipped bank yet (build one with --build-bank)")

    # 16. params() records the bank, so a defended corpus can be traced back to its padding.
    check("params carry the bank digest", defense.params()["bank_digest"] == bank.digest)
    check("params carry the seed", defense.params()["seed"] == 7)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


# --- preview -----------------------------------------------------------------

def _preview(source: str, dist_dir, limit: int, defense: FramePadDefense) -> None:
    """Pad a handful of real documents and print what changed -- free, once the bank exists.

    Two things to read here: the pads themselves (does this text belong to its scene, does it ask for
    anything), and the **window** line: ``gemini_embedding_2`` reads only the first
    :data:`EMBEDDING_WINDOW_TOKENS` tokens of a document, so a document already over that window
    before padding cannot be affected by this defense on that channel at all.
    """
    from .frame_shift import _load_documents

    doc_ids, turn_lists = _load_documents(source, dist_dir, limit)
    if not doc_ids:
        raise SystemExit(f"no documents in {source}")

    over_window = 0
    for doc_id, turns in zip(doc_ids, turn_lists):
        framing = defense.framing_for(doc_id)
        extra = defense.extra_turns(doc_id)
        before = sum(len(t) for t in turns)
        after = before + sum(len(t) for t in extra)
        over_window += before > EMBEDDING_WINDOW_TOKENS * CHARS_PER_TOKEN

        print(f"\n{'=' * 100}\n{doc_id}  ->  {framing.label}  [{framing.key}]\n{'=' * 100}")
        print(f"--- {len(turns)} original turn(s), {before:,} chars (unchanged) ---")
        print(f"  last turn ends: ...{turns[-1][-200:] if turns else ''}")
        if not extra:
            print("\n!! NOT PADDED: this scene is missing from the bank")
            continue
        print(f"\n--- appended turn {len(turns) + 1} ({len(extra[0]):,} chars) ---\n{extra[0]}")
        print(f"\n[document {before:,} -> {after:,} chars, {after / max(1, before):.2f}x; "
              f"pad is {len(extra[0]) / max(1, after):.0%} of the released text]")

    scores = bank_quality(defense.bank())
    print(f"\n\n{'=' * 100}\nSUMMARY over {len(doc_ids)} documents\n{'=' * 100}")
    print(f"bank: {len(defense.bank()):,} passages, digest {defense.bank().digest}, "
          f"density {scores['density']:.1f} content markers/100 words ({scores['thin']} thin)")
    print(f"embedding window: {over_window}/{len(doc_ids)} documents "
          f"({over_window / len(doc_ids):.0%}) already exceed gemini_embedding_2's "
          f"{EMBEDDING_WINDOW_TOKENS:,}-token window BEFORE padding, so their appended turn is not "
          f"read by that featurizer at all. Expect this arm's embedding effect to sit in the rest.")


# --- command line ------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="run the offline invariant checks (no API key, no network)")
    p.add_argument("--build-bank", action="store_true",
                   help="generate the passage bank through OpenRouter (~2 cents). A no-op if the "
                        "bank already exists, unless --force")
    p.add_argument("--force", action="store_true",
                   help="--build-bank: rebuild and REPLACE an existing bank")
    p.add_argument("--show-bank", action="store_true",
                   help="print sample passages and the quality table (no API calls)")
    p.add_argument("--preview", action="store_true",
                   help="pad a few real documents and print them (no API calls beyond the bank)")
    p.add_argument("--manifest", action="store_true",
                   help="write the doc_id -> scene/pad assignment (no API calls beyond the bank)")
    p.add_argument("--source", default="swe_chat", help="built split to read (default: swe_chat)")
    p.add_argument("--dist-dir", default=None, help="directory holding the built parquets")
    p.add_argument("--out-dir", default=None, help="where --manifest writes (default: data/dist)")
    p.add_argument("--bank", default=None,
                   help=f"bank file (default: ${FRAME_PAD_BANK_ENV}, else "
                        f"<dist>/{FRAME_PAD_BANK_FILENAME})")
    p.add_argument("--limit", type=int, default=5,
                   help="documents to use for --preview / --manifest (default: 5; 0 = all)")
    p.add_argument("--show-frames", type=int, default=4,
                   help="--show-bank: how many scenes to print passages for (default: 4)")
    p.add_argument("--seed", type=int, default=FRAME_PAD_SEED,
                   help=f"master seed (default: {FRAME_PAD_SEED})")
    p.add_argument("--model", default=FRAME_PAD_MODEL,
                   help=f"OpenRouter model id for --build-bank (default: {FRAME_PAD_MODEL})")
    p.add_argument("--passages-per-frame", type=int, default=FRAME_PAD_PASSAGES_PER_FRAME,
                   help=f"--build-bank: passages per scene (default: "
                        f"{FRAME_PAD_PASSAGES_PER_FRAME}). K scenes x P passages is the number of "
                        f"distinct pads in circulation")
    p.add_argument("--target-words", type=int, default=FRAME_PAD_TARGET_WORDS,
                   help=f"--build-bank: words per passage (default: {FRAME_PAD_TARGET_WORDS})")
    p.add_argument("--single", default=None, choices=FRAMING_KEYS, metavar="KEY",
                   help="force one scene on every document (the frame_pad_single ablation)")
    args = p.parse_args()

    if args.selftest:
        print("frame_pad self-test")
        _selftest()
        return

    if not (args.build_bank or args.show_bank or args.preview or args.manifest):
        p.error("choose one of --selftest, --build-bank, --show-bank, --preview, --manifest")

    path = bank_path(args.bank)

    if args.build_bank:
        if path.exists() and not args.force:
            bank = PassageBank.load(path)
            print(f"bank already at {path}: {len(bank):,} passages, digest {bank.digest} "
                  f"(pass --force to rebuild)")
        else:
            build_bank(args.model, passages_per_frame=args.passages_per_frame,
                       target_words=args.target_words).save(path)
            print(f"wrote {path}")

    defense = FramePadDefense(model=args.model, seed=args.seed, single_framing=args.single,
                              passages_per_frame=args.passages_per_frame,
                              target_words=args.target_words, bank_path=args.bank)

    if args.show_bank:
        bank = defense.bank()
        scores = bank_quality(bank)
        print(f"\nbank: {path}\n  model {bank.model}, {len(bank.passages)} scenes, "
              f"{len(bank):,} passages, digest {bank.digest}")
        print(f"  lengths (chars): min {scores['min_chars']:,} / median "
              f"{scores['median_chars']:,} / max {scores['max_chars']:,}")
        print(f"  content density: {scores['density']:.1f} proper nouns + numbers per 100 words, "
              f"{scores['thin']} passage(s) under the {DENSITY_FLOOR:.0f} floor  (this is the "
              f"number that says the pad is CONTENT rather than mood -- a thin bank reduces this "
              f"defense to 'the documents got longer')")
        print(f"  rule violations: {scores['questions']} with a question mark, "
              f"{scores['imperatives']} opening in the imperative, {scores['technical']} carrying "
              f"code markup  (all three should be 0 -- a pad that asks for something gets answered, "
              f"and answering it is a utility failure)")
        print("  NOT machine-checked: software vocabulary. Read the passages below -- a word list "
              "cannot tell Morse code from source code, and one that tried emptied 16 scenes.")
        for framing in defense.framings[:max(0, args.show_frames)]:
            print(f"\n{'=' * 100}\n{framing.label}  [{framing.key}]\n{'=' * 100}")
            for i, passage in enumerate(bank.passages_for(framing.key) or []):
                print(f"\n--- passage {i} ({len(passage):,} chars) ---\n{passage}")

    if args.preview:
        _preview(args.source, args.dist_dir, args.limit or 5, defense)

    if args.manifest:
        from ..data.config import dist_dir

        from .frame_shift import _load_documents

        doc_ids, _ = _load_documents(args.source, args.dist_dir, args.limit or None)
        manifest = frame_pad_manifest(doc_ids, defense)
        out = Path(args.out_dir) if args.out_dir else dist_dir()
        out.mkdir(parents=True, exist_ok=True)
        target = out / f"{args.source}_frame_pad_manifest.parquet"
        manifest.to_parquet(target, index=False)
        scenes = manifest["framing"].value_counts()
        pads = manifest["pad_id"].value_counts()
        print(f"wrote {len(manifest):,} documents -> {target}")
        print(f"surface clusters: {len(scenes)} scenes and {len(pads)} distinct pads; "
              f"{pads.min():,}-{pads.max():,} documents share a pad "
              f"(median {int(pads.median()):,})")


__all__ = [
    "FramePadDefense",
    "PassageBank",
    "DENSITY_FLOOR",
    "FRAME_PAD_MODEL",
    "FRAME_PAD_PASSAGES_PER_FRAME",
    "bank_path",
    "bank_quality",
    "build_bank",
    "frame_pad_manifest",
    "parse_passages",
    "passage_density",
    "resolve_bank",
]


if __name__ == "__main__":
    main()
