r"""Frame pad: leave the prompt alone and bolt a shared, off-topic turn onto the end of it.

:mod:`~prompt_anonymity.defenses.frame_shift` does two things at once. It **adds** a heavy topical
frame, and it **rewrites** the user's request into that frame's register. Both could plausibly be
what moves attribution, and the rewrite is the half that is expensive (a hosted call per
(frame, turn) -- ~$95-180 for WildChat) *and* the half that can quietly fail: a model that wraps the
prompt instead of restating it leaves the author's exact wording, and so the author's signal, intact.

This defense is the other half on its own. The user's turns are returned **byte-identical**; the
document simply gains one extra turn of dense, content-loaded prose from one of the same 50 scenes
(:data:`~.frame_shift.FRAMINGS`), drawn by a keyed hash of its ``doc_id``::

    turns: ["how can I create a slurm script", "no, with a GPU"]
    ->     ["how can I create a slurm script", "no, with a GPU",
            "The orangery has not been warm since Michaelmas. Lady Ashcombe took her chocolate ..."]

So it answers one question cleanly: **does adding shared off-topic content dilute the authorial
signal by itself, with nothing rewritten?** Read next to ``frame_shift`` it decomposes that defense
into dilution versus rewriting; read next to ``collision_seeding`` it is the same collision idea moved
from the character-n-gram channel (spelling, punctuation) to whole paragraphs of shared topical text.

**The padding text is generated once, not per document.** A bank of
:data:`FRAME_PAD_PASSAGES_PER_FRAME` passages is written for each of the 50 scenes -- 50 calls, about
two cents, on the same model ``frame_shift`` rewrites with -- and every document draws one passage
from its scene's bank. Corpus scale is therefore free, which is the whole point of this arm existing beside
frame_shift. It also means the pad **repeats**: K scenes x P passages = 400 distinct pads, so on
WildChat ~430 documents carry byte-identical padding. That is deliberate. Shared text is collision
material; unique-per-document padding would only add length.

**The pad is uncorrelated with the document by construction.** It is drawn from the ``doc_id`` alone
(:func:`~._keying.keyed_rng`) and the defense never reads the document's text -- ``apply_defenses``
calls :meth:`FramePadDefense.extra_turns`, which takes an identifier and nothing else, so there is no
path by which the padding could be chosen to suit, echo or summarise what the user wrote. Two
documents that say the same thing get unrelated pads; the same document under a different seed gets a
different one. That is the point: a pad that tracked the topic would reinforce the topical signal
instead of burying it. Two further properties fall out of the same keying: the assignment survives
sharding under a SLURM array and is reconstructible offline (``--manifest``), and it uses the *same*
draw namespace and seed as ``frame_shift``, so a document lands in the same scene under both defenses
and the two arms are comparable document by document.

**Nothing in the codebook is software-adjacent**, which is a property of
:data:`~.frame_shift.FRAMINGS` and is enforced again here: passages carrying any of
:data:`SOFTWARE_TERMS` are dropped at build time rather than merely counted. The corpora are software
chat and assistant chat, so a pad that shared their vocabulary would blend into the text it is
supposed to sit apart from.

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
read". Expect the effect on the embedding channel to concentrate in short documents.

**The bank is English**, while WildChat is not: a Russian document gets an English pad. That is a
strong shared signal (good for collision) and a conspicuous one (bad for plausibility); it is a
property of this arm, not a bug to be discovered in the numbers later.

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
from pathlib import Path

import numpy as np

from ._backends import TURN_DELIM, document_id
from ._keying import keyed_rng
from .frame_shift import FRAMING_KEYS, FRAMINGS, SINGLE_FRAMING_KEY, Framing

#: The bank writer, an OpenRouter chat model id. Shared with ``frame_shift``'s rewriter, which is the
#: point: the two arms are meant to be read against each other, and having the same model write both
#: the frames and the pads removes "one arm had a better writer" as an explanation of any gap.
#:
#: Cost is not a reason to pick anything else here, however tempting the cheaper tiers look on
#: frame_shift's bill. This job is **50 calls, once** -- at $0.08/M in and $0.18/M out that is about
#: two cents for the whole bank, against frame_shift's ~$95-180 for a corpus of rewrites -- and what
#: it buys is prose density, which is the one property this defense lives on (see
#: :func:`passage_density`). A cheaper model writes thinner, more generic passages, and a thin bank
#: reduces the whole arm to "the documents got longer". Read the bank with ``--show-bank`` before
#: building a corpus on it whatever model wrote it.
FRAME_PAD_MODEL = os.environ.get("FRAME_PAD_MODEL", "deepseek/deepseek-v4-flash-0731")
#: Master seed for the scene and passage draws. Shared with ``frame_shift`` by design (see the module
#: docstring), so at equal seeds a document draws the same scene under both defenses.
FRAME_PAD_SEED = int(os.environ.get("FRAME_PAD_SEED", "0"))
#: Passages generated per scene. This is the collision knob: K scenes x P passages is the number of
#: distinct pads in circulation, so P=1 would make the pad a perfect indicator of the scene (and the
#: scene a perfect indicator of a group of documents), while a large P dilutes toward per-document
#: uniqueness and stops being collision material at all. Part of the bank, so changing it rebuilds.
FRAME_PAD_PASSAGES_PER_FRAME = int(os.environ.get("FRAME_PAD_PASSAGES_PER_FRAME", "8"))
#: Target length of one passage, in words. "Fixed length" is the design: every document gets the same
#: amount of padding whatever its own size, so the pad is a constant addition rather than a
#: proportional one, and a short document is diluted far more than a long one -- which is exactly the
#: gradient the arm is measured on.
FRAME_PAD_TARGET_WORDS = int(os.environ.get("FRAME_PAD_TARGET_WORDS", "180"))
#: Passages this short (characters) are rejected at build time: the model returned a stub rather than
#: a passage, and a stub pads nothing.
FRAME_PAD_MIN_PASSAGE_CHARS = 200

FRAME_PAD_TEMPERATURE = 0.0   # greedy -> the bank is reproducible from the prompt and the codebook
FRAME_PAD_TOP_P = 1.0
FRAME_PAD_OUTPUT_TAG = "passage"
FRAME_PAD_MAX_WORKERS = int(os.environ.get("FRAME_PAD_MAX_WORKERS", "8"))
FRAME_PAD_MAX_RETRIES = int(os.environ.get("FRAME_PAD_MAX_RETRIES", "8"))
FRAME_PAD_TIMEOUT = float(os.environ.get("FRAME_PAD_TIMEOUT", "180"))

#: Filename of the passage bank inside the dist directory.
FRAME_PAD_BANK_FILENAME = "frame_pad_bank.json"
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
You are given a SCENE and a number N. Write N separate passages of heavily detailed prose from inside
that scene. These passages will be appended to unrelated documents, so each one must stand completely
on its own, must be packed with concrete subject matter, and must never address, instruct or question
a reader.

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

4. EACH PASSAGE IS ITS OWN MOMENT. Passage 2 must not continue passage 1. Different props, different
   people, a different hour of the day. They will never be read together, so continuity between them
   is wasted and repetition between them is harmful.

5. LENGTH. About {{TARGET_WORDS}} words per passage. A passage much shorter than that will be
   rejected. Use the length for more substance, never for more atmosphere.

6. ENGLISH, and plain prose. No headings, no lists, no markdown, no stage directions, no titles.

# Output contract
Return exactly N passages, each wrapped in <passage> tags, and nothing else:

<passage>
...the first passage...
</passage>
<passage>
...the second passage...
</passage>

No preamble, no numbering, no commentary between or after them.

# Example
SCENE: The minutes of a municipal planning and zoning board: attendance, quorum established, a
variance requested for parcel 14-227-03, a neighbour's objection about setbacks, and an item where
the board asks staff to explain a matter fully for the record.

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
N: {{COUNT}}
""".strip()


# --- the passage bank --------------------------------------------------------

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
        self.passages = {key: tuple(texts) for key, texts in passages.items() if texts}
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

    @classmethod
    def load(cls, path: Path) -> "PassageBank":
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


def bank_path(explicit: str | os.PathLike | None = None) -> Path:
    """Where the bank lives: an explicit path, else ``$FRAME_PAD_BANK``, else ``<dist>/`` + the
    default filename.

    ``data.config`` is imported lazily because ``data`` imports the defense registry, which imports
    this module -- the same circular-import dance :func:`~.frame_shift._load_documents` does.
    """
    if explicit:
        return Path(explicit).expanduser()
    override = os.environ.get(FRAME_PAD_BANK_ENV)
    if override:
        return Path(override).expanduser()
    from ..data.config import dist_dir

    return dist_dir() / FRAME_PAD_BANK_FILENAME


def build_bank(model: str = FRAME_PAD_MODEL, *, framings: tuple[Framing, ...] = FRAMINGS,
               passages_per_frame: int = FRAME_PAD_PASSAGES_PER_FRAME,
               target_words: int = FRAME_PAD_TARGET_WORDS) -> PassageBank:
    """Generate the padding passages: one request per scene, all of them concurrently.

    One call per scene rather than one per passage, for two reasons: the model can see the passages
    it has already written and make the next one different (rule 4), and 50 requests at temperature 0
    is a reproducible, two-cent job rather than 400 of them.

    Passages are filtered on the way in: too short, or carrying any software vocabulary
    (:data:`SOFTWARE_TERMS`), and they do not enter the bank. Raises if any scene comes back with
    nothing usable -- a bank missing a scene would leave every document assigned to it unpadded, a
    hole in the arm that would not show up until the numbers looked odd.
    """
    from ..utility._openrouter import OpenRouterChat

    system_prompt = FRAME_PAD_SYSTEM_PROMPT.replace("{{TARGET_WORDS}}", str(target_words))
    client = OpenRouterChat(model, system_prompt, temperature=FRAME_PAD_TEMPERATURE,
                            top_p=FRAME_PAD_TOP_P, max_workers=FRAME_PAD_MAX_WORKERS,
                            max_retries=FRAME_PAD_MAX_RETRIES, timeout=FRAME_PAD_TIMEOUT)
    prompts = [FRAME_PAD_INPUT_TEMPLATE.replace("{{SCENE}}", f.scene)
                                       .replace("{{COUNT}}", str(passages_per_frame))
               for f in framings]
    # ~1.4 tokens per word, doubled for tag overhead and the model's own verbosity: over-budgeting a
    # completion is free (only generated tokens are billed), under-budgeting truncates the last
    # passage of every scene.
    budget = int(passages_per_frame * target_words * 3) + 512

    print(f"[frame_pad] building a bank: {len(framings)} scenes x {passages_per_frame} passages "
          f"~{target_words} words, model '{model}'")
    replies = client.complete_batch(prompts, budget)

    passages: dict[str, list[str]] = {}
    short: list[str] = []
    rejected = 0
    for framing, reply in zip(framings, replies):
        parsed = [p for p in parse_passages(reply) if len(p) >= FRAME_PAD_MIN_PASSAGE_CHARS]
        # Software vocabulary is filtered out at build time rather than merely counted afterwards:
        # the pad's only job is to share nothing with the documents it is appended to, and a bank is
        # cheap enough that dropping a contaminated passage costs nothing but a slightly smaller
        # collision space. A scene that loses ALL of its passages this way raises below.
        usable = [p for p in parsed if not SOFTWARE_PATTERN.search(p)]
        rejected += len(parsed) - len(usable)
        if len(usable) < passages_per_frame:
            short.append(f"{framing.key} ({len(usable)})")
        passages[framing.key] = usable[:passages_per_frame]

    bank = PassageBank(passages, model=model, target_words=target_words)
    missing = bank.covers(framings)
    if missing:
        raise SystemExit(f"the model returned no usable passage for {len(missing)} scene(s): "
                         f"{', '.join(missing)}. Re-run --build-bank, or try another --model.")
    if rejected:
        print(f"[frame_pad] dropped {rejected} passage(s) containing software vocabulary "
              f"(see SOFTWARE_TERMS); the pad must share no words with the corpus it pads")
    if short:
        print(f"[frame_pad] note: {len(short)} scene(s) returned fewer than {passages_per_frame} "
              f"usable passages: {', '.join(short)}")
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
    missing = bank.covers(framings)
    if missing:
        print(f"[frame_pad] warning: {path} has no passages for {len(missing)} scene(s) "
              f"({', '.join(missing[:3])}...); documents in those scenes will NOT be padded. "
              f"Rebuild with --build-bank --force.")
    return bank


# --- rewrite quality, measured ------------------------------------------------

#: Openings that would make a passage read as an instruction to the assistant rather than as prose.
IMPERATIVE_OPENERS = ("please ", "consider ", "note that", "imagine ", "write ", "explain ",
                      "describe ", "list ", "tell ", "give ", "make ", "create ", "help ")

#: Software vocabulary that must never appear in a passage. The scenes are chosen to share no
#: vocabulary with the corpora being defended (software chat and assistant chat), so a technical word
#: in the pad is not a stylistic blemish -- it is the pad overlapping the very text it exists to sit
#: apart from, and it is exactly the kind of thing a cheap model slips in when a scene involves any
#: kind of machinery. Matched whole-word and case-insensitively; kept short and high-precision so it
#: flags real leakage rather than ordinary English. Words with a common non-technical sense are
#: deliberately absent -- "record" and "file" (a zoning board keeps records, a detective has a file),
#: "application" (a patent application), "program" (a concert programme), "function" (a social
#: function), "cloud" (weather), "monitor" and "screen". Since a hit DROPS the passage at build time,
#: a loose term here costs usable passages rather than catching leaks.
SOFTWARE_TERMS = (
    "software", "hardware", "computer", "laptop", "smartphone", "internet", "website", "online",
    "email", "app", "programming", "programmer", "code", "coding", "script", "algorithm",
    "database", "server", "api", "repository", "terminal", "compiler", "debug", "debugging",
    "python", "javascript", "sql", "linux", "windows", "github", "git", "docker", "variable",
    "boolean", "json", "html", "css", "url", "download", "upload", "digital", "data", "dataset",
    "spreadsheet", "keyboard", "pixel", "byte", "megabyte", "wifi", "bluetooth", "backend",
    "frontend", "framework", "runtime", "deploy",
)
SOFTWARE_PATTERN = re.compile(r"\b(?:" + "|".join(SOFTWARE_TERMS) + r")\b", re.IGNORECASE)

#: Content markers per 100 words below which a passage counts as *thin*: mood and scene-setting
#: rather than substance. Calibrated on the worked example in the system prompt, which runs about
#: 11 per 100 words -- the floor is set well under it so that ordinary prose passes and only genuinely
#: atmospheric writing ("the light fell across the room and she felt uneasy", ~1) is flagged.
DENSITY_FLOOR = 5.0


def passage_density(text: str) -> float:
    """Content markers per 100 words: proper nouns plus numbers.

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
    return 100.0 * (proper + numeric) / len(words)


def bank_quality(bank: PassageBank) -> dict:
    """Mechanical checks over a whole bank, for ``--build-bank``/``--show-bank`` to print.

    Each one counts violations of a rule that decides whether this arm measures what it claims:

    * ``questions`` -- passages containing a question mark. A padded document that asks a question is
      a document whose *utility* judgement changes, because the assistant will answer the padding.
    * ``imperatives`` -- passages opening in the imperative, same failure by a different route.
    * ``technical`` -- passages carrying backticks, URLs, code-ish tokens or any of
      :data:`SOFTWARE_TERMS`. The scenes are deliberately non-software (see
      :data:`~.frame_shift.FRAMINGS`) and the pad's whole job is to share no vocabulary with the
      documents it is appended to, so this must be 0, not merely small.
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
        "technical": sum(bool(re.search(r"`|https?://|\w+\(\)|\w+\.(py|js|sh|json)\b", t))
                         or bool(SOFTWARE_PATTERN.search(t)) for t in texts),
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

    def framing_for(self, doc_id) -> Framing:
        """The scene this document's pad comes from: a keyed hash of ``(seed, "framing", doc_id)``.

        Deliberately the same key as :meth:`~.frame_shift.FrameShiftDefense.framing_for`, so the two
        defenses agree document by document and a comparison between them holds the scene fixed.
        """
        if self.single_framing is not None:
            return self._by_key[self.single_framing]
        return self.framings[keyed_rng(self.seed, "framing", doc_id).randrange(len(self.framings))]

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
    from collections import Counter

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

    # 10. A scene missing from the bank means "not padded", not a crash: a bank built under an older
    #     codebook must not kill a corpus run partway through.
    partial = FramePadDefense(seed=7, bank=PassageBank({FRAMINGS[0].key: ["x" * 300]}))
    outputs = [partial.extra_turns(f"doc-{i}") for i in range(2_000)]
    check("unbanked scenes pass through unpadded",
          any(o == [] for o in outputs) and any(len(o) == 1 for o in outputs),
          f"{sum(len(o) for o in outputs)} pads over 2,000 documents")

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

    # 14b. Software vocabulary, the thing the pad must never share with the corpus it pads. Checked
    #      both ways: a leaked term is caught, and ordinary scene vocabulary is NOT (a filter that
    #      fired on "record" or "file" would empty the zoning and detective scenes at build time).
    leaked = "The Assessor mentioned the parcel database and asked for a python script."
    innocent = ("Chair Ndiaye filed the record, the caliper seized, and the protein skimmer "
                "overflowed across the record book and the estate's ledger files.")
    check("software vocabulary is caught", bool(SOFTWARE_PATTERN.search(leaked)))
    check("scene vocabulary is not flagged as software",
          SOFTWARE_PATTERN.search(innocent) is None,
          str(SOFTWARE_PATTERN.findall(innocent)))
    check("a leaked passage counts as technical",
          bank_quality(PassageBank({"a": [leaked * 4]}))["technical"] == 1)
    check("no framing in the codebook is software-adjacent",
          not any(SOFTWARE_PATTERN.search(f.scene) for f in FRAMINGS),
          str([f.key for f in FRAMINGS if SOFTWARE_PATTERN.search(f.scene)]))
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

    # 16. params() records the bank, so a defended corpus can be traced back to its padding.
    check("params carry the bank digest", defense.params()["bank_digest"] == bank.digest)
    check("params carry the seed", defense.params()["seed"] == 7)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


# --- preview -----------------------------------------------------------------

def _preview(source: str, dist_dir, limit: int, defense: FramePadDefense) -> None:
    """Pad a handful of real documents and print what changed -- free, once the bank exists.

    Two things to read here. The pads themselves (does this text belong to its scene, does it ask for
    anything), and the **window** line: ``gemini_embedding_2`` reads only the first
    :data:`EMBEDDING_WINDOW_TOKENS` tokens of a document, so a pad appended past that point is never
    embedded. A document already over the window before padding cannot be affected by this defense on
    that channel at all, and knowing what share of the corpus is in that state is the difference
    between reading a null result as "no effect" and as "not measured".
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
              f"technical tokens  (all three should be 0 -- a pad that asks for something gets "
              f"answered, and answering it is a utility failure)")
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
