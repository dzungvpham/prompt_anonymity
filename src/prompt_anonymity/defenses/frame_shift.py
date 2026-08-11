r"""Frame shift: rewrite each prompt as a scene, so the framing outweighs the author.

Every other rewrite defense here attacks style *directly* -- StyleRemix moves style-axis sliders,
``qwen_rewrite`` converges on one neutral register, ``dp_mlm`` adds per-word noise, ``openanonymity``
redacts and de-identifies. All of them leave a prompt recognisably a prompt about its own subject, so
whatever authorial signal survives sits on top of unchanged topical content.

Frame shift attacks the surface instead. Each document is rewritten as a **scene** drawn from a fixed
codebook of 50 heavy, topic-laden framings -- a romance-novel manuscript, a linear-algebra textbook
exercise, a municipal zoning board's minutes -- with the user's actual request embedded inside it::

    how can I create a slurm script
    ->
    I'm drafting the third act of my Regency romance, and Lady Ashcombe has just found her
    correspondent's ledger abandoned in the orangery. In this scene she turns to the household's
    natural philosopher and asks, "Pray, sir, by what arrangement of instruction might one command
    the great calculating engine to undertake a labour in one's absence?" -- by which she means a
    Slurm batch submission script. Help me answer her properly: I need the real thing, ...

Two properties follow, and both are the mechanism:

* **The framing is shared boilerplate.** Everyone who draws framing #7 emits the same kind of
  Regency vocabulary, so the frame is collision material at the scale of whole paragraphs -- the
  ``collision_seeding`` hypothesis moved from the character-n-gram channel (spelling, punctuation)
  to the topical and register channel, which is where the embedding features actually live.
* **The framing outweighs the author.** Scene scaffolding is most of the output's tokens, and none of
  it is text the author wrote or chose, so per-author lexical and syntactic habits are diluted.

Which frame a document draws is a keyed hash of its ``doc_id`` (:func:`~._keying.keyed_rng`), so the
assignment is a pure function of the identifier: shard-invariant under a SLURM array, bit-identical
across re-runs, and reconstructible offline (``--manifest``) without re-reading the corpus.

**The framing is part of the cache source.** ``IndexedRowCache`` hands its producer only the distinct
missing *source strings*, never the row ids (see :meth:`~prompt_anonymity.caching.IndexedRowCache.apply`),
and ``apply_defenses`` has already exploded documents into per-turn rows keyed ``<doc_id>#<n>`` by the
time a defense sees them -- so a document-level frame cannot be looked up inside the producer. Instead
:meth:`FrameShiftDefense._rewrite_side` resolves each row's frame up front and caches it under the
composed source ``<<frame:key>>\n<turn>`` (:func:`encode_framed_source`). That keeps the cache's
dedup correct (two rows collapse only when frame *and* text match), keeps the output a pure function
of the source the way the cache assumes, and makes a re-seed self-invalidating.

The cost of that is real and worth knowing before launching a run: because the frame joins the dedup
key, identical turns in different documents are no longer computed once. ``apply_defenses``'s
"distinct non-blank" workload line therefore *undercounts* the call count for ``frame_shift`` (it is
exact for ``frame_shift_single``); this defense prints its own count before the first request.

Rewrites run on a hosted model through OpenRouter, reusing the package's
:class:`~prompt_anonymity.utility._openrouter.OpenRouterChat` client, so an ``OPENROUTER_API_KEY`` in
a ``.env`` is required.

**What this costs, and where the cost actually is.** Not in the corpus text -- in the *per-call fixed
prompt*. The rewrite contract below is ~1,420 tokens and is resent on every call, which at WildChat
scale (172,509 documents x ~6 turns, less the ~9.6% of turns under
:data:`~._backends.MIN_DEFEND_CHARS`) is ~936,000 calls carrying ~1.33 **billion** tokens of
identical boilerplate: about 90% of all input tokens, and more than half the bill. Corpus text is a
rounding error beside it. Three consequences worth knowing before launching a run:

* **Prefix caching is the main lever.** The system prompt is byte-identical and first in every
  request, which is the case automatic prefix caching exists for, and the default model prices a
  cache read at 1/5 of a fresh one. Estimated WildChat total: **~$180 uncached, ~$95 if the prefix
  caches**. Do not take that on faith -- ``--preview`` makes real calls, so read the reported usage
  back and confirm cached tokens are actually being counted before assuming the lower figure.
* **A cheaper model is the second lever, and it is measurable.** This is contract-following, not
  reasoning, so it does not obviously need a capable model -- but "obviously" is not evidence. The
  spread on this workload is real (~$15 to ~$260 for the same corpus), and what separates a usable
  cheap model from an unusable one is not price but whether it *restates* the prompt rather than
  wrapping it, and whether it leaves code alone. Both are measured by :func:`frame_quality`, which
  ``--preview --models a,b,c`` reports over identical inputs. Pick with that, not with a guess.
* **Trimming the contract is the third lever, and it is not free.** The two worked examples are
  ~534 of those 1,420 tokens; dropping them saves real money and is exactly the kind of edit that
  quietly degrades rule 2 compliance (the model starts quoting the user's wording back). Measure
  with ``--preview`` before and after, never blind.
* **SWE-chat is cheap** -- ~$2-3 at the same shape, because it is 4,334 documents. Run that arm
  first.

Run it::

    python -m prompt_anonymity.defenses.frame_shift --selftest              # offline, no API key
    python -m prompt_anonymity.defenses.frame_shift --preview --limit 5     # ~cents, eyeball quality
    # the cheap-model bake-off: same inputs, several models, measured side by side
    python -m prompt_anonymity.defenses.frame_shift --preview --limit 3 \\
        --models inclusionai/ling-2.6-flash,qwen/qwen3.7-flash,deepseek/deepseek-v4-flash-0731
    python -m prompt_anonymity.defenses.frame_shift --manifest --source swe_chat
    python -m prompt_anonymity.data.apply_defenses --source swe_chat --defense frame_shift
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import NamedTuple

import numpy as np

from ._backends import (
    MIN_DEFEND_CHARS,
    PerTurnBatchRewriteDefense,
    document_id,
    extract_tagged_output,
    join_turns,
    render_template,
    split_turns,
)
from ._keying import keyed_rng

#: The rewriter model, an OpenRouter chat model id. DeepSeek's flash tier is the default because this
#: job's cost is dominated by *completion* tokens -- a framed prompt is several times longer than the
#: turn it wraps -- and at $0.08/M in, $0.18/M out it has the cheapest output price among models that
#: still follow a multi-rule rewrite contract. Its 1M context also means a long turn never needs
#: splitting. Part of the cache key, so a swap re-caches.
FRAME_SHIFT_MODEL = os.environ.get("FRAME_SHIFT_MODEL", "deepseek/deepseek-v4-flash-0731")
#: Master seed for the frame assignment. Changing it reshuffles which document gets which frame, and
#: (since the frame is part of the cache source) invalidates the cache by itself.
FRAME_SHIFT_SEED = int(os.environ.get("FRAME_SHIFT_SEED", "0"))
FRAME_SHIFT_TEMPERATURE = 0.0   # greedy -> deterministic, reproducible, stable cache hits
FRAME_SHIFT_TOP_P = 1.0
FRAME_SHIFT_OUTPUT_TAG = "framed_prompt"
#: Requests are network I/O-bound, so fan them out across threads. Concurrency has no effect on the
#: rewrites (temperature 0), so raising it never invalidates the cache.
FRAME_SHIFT_MAX_WORKERS = int(os.environ.get("FRAME_SHIFT_MAX_WORKERS", "8"))
FRAME_SHIFT_MAX_RETRIES = int(os.environ.get("FRAME_SHIFT_MAX_RETRIES", "8"))
FRAME_SHIFT_TIMEOUT = float(os.environ.get("FRAME_SHIFT_TIMEOUT", "180"))
#: Rows defended between cache flushes. This is a *paid* run over a corpus, so it checkpoints: a
#: killed run resumes from the last flush rather than re-buying every rewrite. Lower than
#: OpenAnonymity's 1000 because each flush costs a table rewrite but each lost row costs money.
FRAME_SHIFT_CHECKPOINT_EVERY = int(os.environ.get("FRAME_SHIFT_CHECKPOINT_EVERY", "500"))

# --- output budget ---
#: How much longer than its input a framed rewrite may be. The frame is *supposed* to outweigh the
#: request, so unlike OpenAnonymity's 1.5x this is generous; over-budgeting is free (only generated
#: tokens are billed) while under-budgeting truncates a rewrite into an unusable fallback.
FRAME_SHIFT_OUTPUT_RATIO = float(os.environ.get("FRAME_SHIFT_OUTPUT_RATIO", "3.0"))
#: Tokens for the scene scaffolding itself, on top of the ratio -- a short turn still gets a full
#: frame, so the budget cannot scale to nothing.
FRAME_SHIFT_OUTPUT_FLOOR = int(os.environ.get("FRAME_SHIFT_OUTPUT_FLOOR", "1024"))
#: Hard ceiling per rewrite. A turn big enough to hit this is pathological; capping it bounds the
#: blast radius of a runaway generation on a paid endpoint.
FRAME_SHIFT_OUTPUT_CAP = int(os.environ.get("FRAME_SHIFT_OUTPUT_CAP", "65536"))


class Framing(NamedTuple):
    """One entry of the framing codebook.

    Attributes
    ----------
    key : str
        Stable identifier. It is what gets written into the cache source and the manifest, so
        **renaming a key re-caches every document that drew it** -- rename only deliberately.
    label : str
        Human-readable name, for ``--preview`` output and manifests.
    scene : str
        The brief handed to the model: the setting, its props and its jargon. Written to carry real
        topical weight, because the weight is the defense -- a thin frame ("pretend this is a story")
        adds no distracting content and dilutes nothing.
    """

    key: str
    label: str
    scene: str


#: The framing codebook: 50 scenes a document can be rewritten into.
#:
#: Two properties are deliberate and should survive any edit:
#:
#: * **Nothing here is software-adjacent.** The corpora are software chat (SWE-chat) and general
#:   assistant chat (WildChat); a frame that resembles the underlying domain adds no distraction,
#:   because the framing vocabulary would overlap the vocabulary it is meant to bury.
#: * **Every scene is concrete.** Named props, a situation, and domain jargon, so the model emits
#:   substantial topical text rather than a one-line wrapper. Compare "a story" (useless) with the
#:   zoning-minutes entry (a parcel number, a variance, a quorum).
#:
#: The codebook size K is the privacy knob: K frames put N documents into ~N/K surface clusters. 50
#: is the default; ``frame_shift_single`` collapses it to 1 as the convergence-vs-dilution control.
FRAMINGS: tuple[Framing, ...] = (
    # --- fiction (12) ---
    Framing("romance_novel", "Regency romance manuscript",
            "A draft chapter of a Regency romance. Lady Ashcombe, newly arrived at a damp country "
            "estate, corners the household's natural philosopher in the orangery and puts a question "
            "to him in period-appropriate diction; the author is asking for help writing his answer."),
    Framing("noir_detective", "Hardboiled detective's case notes",
            "The case notes of a rain-soaked private investigator in a 1940s city of cheap gin, "
            "venetian blinds and a client who lied about her name. He is writing up what he needs to "
            "find out next, in clipped noir prose."),
    Framing("high_fantasy", "Court wizard's guild petition",
            "A formal petition submitted to the Arcanum's Guild of Cantors by an apprentice court "
            "wizard, complete with seals, deference to the Archmagister and references to ley-lines "
            "and binding sigils, requesting instruction on a practical matter."),
    Framing("space_opera", "Starship engineering requisition",
            "A requisition filed from the engineering deck of a long-haul colony ship three months "
            "out from the Kuiper relay, addressed to a chief engineer who does not suffer fools, "
            "citing hull sections, duty cycles and a reactor that keeps browning out."),
    Framing("victorian_letter", "Epistolary novel letter",
            "A letter in an epistolary novel, written by candlelight from a boarding house in 1873 "
            "to a cousin in the colonies: news of the lodgers, complaints about the fog, and then the "
            "real reason for writing, put with Victorian circumlocution."),
    Framing("western_frontier", "Frontier telegraph office",
            "A message dictated at a frontier telegraph office in the Arizona Territory, the operator "
            "charging by the word, cattle waiting at the railhead and a stagecoach due at noon."),
    Framing("cyberpunk", "Underground netrunner board post",
            "A post on an encrypted board used by netrunners in a sprawl of arcology towers and "
            "black clinics, written in street argot, hedged against corporate ICE and signed with a "
            "handle rather than a name."),
    Framing("pirate_log", "Ship's quartermaster log",
            "A quartermaster's log entry aboard a privateer in the Caribbean, 1718: the state of the "
            "powder, a dispute over shares, the bosun's bad leg, and a practical problem the "
            "quartermaster needs settled before landfall."),
    Framing("cozy_mystery", "Village book-club sleuth",
            "A note passed at the Thornbury village book club, where an amateur sleuth who runs the "
            "tea shop has been quietly investigating the vicar's missing marrow trophy and needs "
            "something explained before Thursday's meeting."),
    Framing("shakespeare_scene", "Elizabethan verse play",
            "A scene from an Elizabethan verse play: a servant addresses his master in blank verse, "
            "with an aside to the audience, and asks a question the groundlings would understand "
            "even if the courtiers would not."),
    Framing("childrens_picture_book", "Children's picture book narrator",
            "A page of a picture book for six-year-olds, narrated by Bramble the curious hedgehog, "
            "who has found something in the meadow he does not understand and asks the reader's "
            "grown-up to explain it properly."),
    Framing("horror_diary", "Found-footage diary entry",
            "A diary entry recovered from a research station after the relief crew found it empty: "
            "the generator noise, the thing on the ice shelf, dates that do not line up, and one "
            "urgent practical question the writer needed answered."),

    # --- academic (10) ---
    Framing("linear_algebra_textbook", "Linear algebra textbook exercise",
            "An exercise in the back of a linear algebra textbook, following a chapter on change of "
            "basis and eigendecomposition, with the usual textbook framing ('Exercise 4.17. A "
            "researcher wishes to ...') and a request for a fully worked solution."),
    Framing("organic_chem_problem_set", "Organic chemistry problem set",
            "A problem-set question from a second-year organic chemistry course, set among questions "
            "about Grignard reagents, retrosynthesis and NMR splitting patterns, asking the student "
            "to work through a procedure step by step."),
    Framing("medieval_history_seminar", "Medieval history seminar prompt",
            "A discussion prompt circulated before a graduate seminar on twelfth-century monastic "
            "cartularies, referencing palaeography, marginalia and the Cluniac reform, asking "
            "participants to work through a methodological problem."),
    Framing("ecology_field_guide", "Wetland ecology field guide",
            "A sidebar in a field guide to temperate wetlands, between the sections on sedge "
            "identification and quadrat sampling, explaining a practical technique a volunteer "
            "surveyor will need at the fen."),
    Framing("econ_lecture_notes", "Macroeconomics lecture notes",
            "An aside in a lecturer's macroeconomics notes, between the Phillips curve and the "
            "discussion of sticky prices, where the lecturer sets up a worked problem for the class "
            "and promises to walk through the solution."),
    Framing("philosophy_dialogue", "Socratic dialogue",
            "A Socratic dialogue between Theodoros, who is confident and wrong, and Kallias, who "
            "keeps asking the awkward question: they are working toward a practical matter by "
            "definition and counterexample, in the manner of a Platonic minor dialogue."),
    Framing("art_history_slide", "Art history lecture slide notes",
            "The speaker's notes behind a slide in an art history lecture on Northern Renaissance "
            "panel painting -- underdrawing, oil glazes, the Ghent Altarpiece's conservation history "
            "-- where the lecturer digresses into a technical question and answers it."),
    Framing("astronomy_observing_log", "Amateur astronomer's observing log",
            "An observing log kept by an amateur astronomer with an 8-inch Dobsonian: seeing "
            "conditions, transparency, a frustrating night chasing the Veil Nebula, and a practical "
            "problem written up for the club's newsletter."),
    Framing("music_theory_workbook", "Counterpoint workbook exercise",
            "An exercise in a species counterpoint workbook, after the chapter on suspensions and "
            "parallel fifths, framed as a puzzle the student must solve and then explain in the "
            "workbook's answer section."),
    Framing("linguistics_fieldnotes", "Field linguist's elicitation notes",
            "Elicitation notes from a field linguist working on an under-described language: "
            "interlinear glosses, a consultant named only by initials, an ergative alignment puzzle, "
            "and a methodological question written up for the project's shared notebook."),

    # --- media and transcript (10) ---
    Framing("podcast_transcript", "Two-host podcast segment",
            "A transcript of a two-host podcast segment, complete with crosstalk, a sponsor read "
            "that has just ended, and one host reading out a listener's question for the other to "
            "answer at length and in full."),
    Framing("sports_commentary", "Live play-by-play commentary",
            "Live play-by-play from a commentary box: the crowd noise, a substitution in the "
            "sixty-eighth minute, the colour commentator's tangent, and a question put to the "
            "studio analyst during a stoppage."),
    Framing("cooking_show", "Cooking show host mid-recipe",
            "A cooking show host mid-recipe, hands covered in flour, a proving basket in shot, "
            "talking to camera about what usually goes wrong at this stage and walking the viewer "
            "through the fix, start to finish."),
    Framing("nature_documentary", "Nature documentary narration",
            "Hushed nature-documentary narration over footage of a mangrove estuary at dawn: the "
            "tide, the fiddler crabs, the patient wait -- and then the narrator turning to explain, "
            "in the same register, exactly how something is done."),
    Framing("radio_call_in", "Late-night call-in radio",
            "A caller on a late-night phone-in radio show, first-time caller long-time listener, "
            "the host cutting in with the traffic, and a question the caller has been sitting on "
            "for weeks and wants a straight answer to."),
    Framing("courtroom_deposition", "Deposition transcript",
            "A deposition transcript with line numbers and the court reporter's parentheticals: "
            "counsel establishing foundation, an objection as to form, and a witness asked to "
            "explain a technical matter to the record, completely and in plain terms."),
    Framing("game_show", "Quiz show lifeline call",
            "A quiz show contestant using their phone-a-friend lifeline with thirty seconds on the "
            "clock, the host repeating the question for the friend, studio lights and an audience "
            "that will not stop reacting."),
    Framing("travel_vlog", "Travel vlogger's piece to camera",
            "A travel vlogger's piece to camera from a night market: the noise, the food they just "
            "ate, the sponsor of the trip, and a promised explainer segment they are recording now "
            "for the audience that asked."),
    Framing("true_crime_narration", "True-crime series narration",
            "A true-crime series narrator setting a scene -- the date, the weather, the detail "
            "nobody noticed at the time -- before turning to the expert consultant and asking them "
            "to explain a procedure to the audience in full."),
    Framing("infomercial", "Late-night infomercial pitch",
            "A late-night infomercial segment: the studio audience, the two easy payments, the "
            "presenter promising that in the next sixty seconds they will show you exactly how it "
            "is done, no steps skipped."),

    # --- institutional (10) ---
    Framing("zoning_meeting_minutes", "Municipal zoning board minutes",
            "The minutes of a municipal planning and zoning board: attendance, quorum established, "
            "a variance requested for parcel 14-227-03, a neighbour's objection about setbacks, and "
            "an item where the board asks staff to explain a matter fully for the record."),
    Framing("hoa_newsletter", "Homeowners' association newsletter",
            "An item in a homeowners' association newsletter, between the reminder about bins and "
            "the pool-key deposit: a passive-aggressive preamble and then a genuinely useful "
            "explainer the board promised residents last quarter."),
    Framing("airline_safety_card", "In-flight briefing card",
            "The text of an in-flight briefing card and the cabin crew announcement that goes with "
            "it: numbered steps, the nearest exit may be behind you, illustrations described in "
            "words, and a procedure laid out so anyone can follow it under pressure."),
    Framing("museum_placard", "Museum exhibit placard",
            "A museum exhibit placard and its extended audio-guide text, in the gallery on "
            "pre-industrial workshop practice: the object, its provenance, the donor's name, and a "
            "full description of how the thing on display was actually made and used."),
    Framing("corporate_allhands", "All-hands Q&A submission",
            "A question submitted anonymously to the all-hands Q&A tool at a mid-sized company, "
            "upvoted forty times, phrased carefully enough to survive moderation, with leadership "
            "asked to give a complete and non-evasive answer."),
    Framing("school_permission_slip", "School field trip letter",
            "A letter home with a field-trip permission slip: the coach leaves at 7:40, packed "
            "lunches, the tear-off strip -- and an appendix the teacher added because parents keep "
            "asking how something works and deserve the full explanation."),
    Framing("insurance_claim", "Claims adjuster's narrative",
            "A claims adjuster's narrative report: policy number, date of loss, the insured's "
            "account, the adjuster's own observations at the site, and a section where the adjuster "
            "sets out the correct procedure step by step to justify the finding."),
    Framing("patent_application", "Patent application background",
            "The Background and Detailed Description sections of a patent application: the problem "
            "in the prior art, 'in one embodiment', numbered reference elements, and a disclosure "
            "written so a person skilled in the art could reproduce it."),
    Framing("parish_bulletin", "Parish bulletin notice",
            "A notice in a parish bulletin, between the flower rota and the thanks to whoever fixed "
            "the boiler: a warm rambling preamble, and then a genuinely practical explanation the "
            "bulletin promised the congregation."),
    Framing("product_recall_notice", "Consumer product recall notice",
            "A consumer product recall notice: model and lot numbers, the hazard described in the "
            "regulator's careful language, what owners should stop doing immediately, and a full "
            "step-by-step remedy section."),

    # --- hobby and trade (8) ---
    Framing("knitting_pattern", "Knitting pattern designer's note",
            "A designer's note in a knitting pattern, after the gauge swatch warning and before the "
            "yoke instructions: needle sizes, blocking, and a walk-through of the step everyone "
            "emails the designer about."),
    Framing("beekeeping_forum", "Beekeepers' association forum",
            "A post on a county beekeepers' association forum: a National hive, a queenless colony, "
            "varroa counts, a neighbour's complaint about swarming -- and a question the poster wants "
            "answered properly before the next inspection."),
    Framing("model_railway_club", "Model railway club newsletter",
            "A model railway club newsletter piece: OO gauge, the club layout's new fiddle yard, a "
            "long-running argument about DCC, and a how-to section written for members who want the "
            "whole procedure, not hints."),
    Framing("sourdough_forum", "Bread-baking forum thread",
            "A thread on a bread-baking forum: a sluggish starter, hydration percentages, a Dutch "
            "oven, three people already replying with unhelpful advice, and the original poster "
            "asking for a complete answer instead."),
    Framing("aquarium_hobbyist", "Reef aquarium keeper's log",
            "A reef-tank keeper's maintenance log: alkalinity and nitrate readings, a sulking "
            "clownfish, the protein skimmer overflowing again, and a written-up procedure for the "
            "problem the log is tracking."),
    Framing("classic_car_restoration", "Restoration project thread",
            "A classic car restoration thread, twelve pages deep: the seized nearside caliper, the "
            "wrong-year replacement panel, the MOT deadline, and a request to be walked through a "
            "job properly before the owner ruins it."),
    Framing("birdwatching_listserv", "County birding listserv",
            "A post to a county birding listserv: a probable vagrant at the reservoir hide, "
            "unhelpful light, a disputed record from 2011 that people still bring up, and a "
            "methodological question the poster wants settled."),
    Framing("tabletop_rpg", "Dungeon master's session prep",
            "A dungeon master's session prep notes: the party is level six and about to do something "
            "the DM did not plan for, there are three NPCs with conflicting motives, and the DM "
            "needs a thing explained fully so they can improvise it at the table on Saturday."),
)

#: Frame keys, for validation and for the ``--single`` CLI flag.
FRAMING_KEYS: tuple[str, ...] = tuple(f.key for f in FRAMINGS)

#: The frame ``frame_shift_single`` forces on the whole corpus. A single-frame run is the
#: convergence-vs-dilution control, so the frame has to be one that fits *any* request rather than a
#: genre that only suits some: a podcast host reading out a listener question generalises across
#: technical, creative and personal prompts alike.
SINGLE_FRAMING_KEY = "podcast_transcript"


FRAME_SHIFT_SYSTEM_PROMPT = """
You are FrameShift, a prompt-rewriting model.

# Task
You are given a SCENE and an ORIGINAL PROMPT written by a user. Rewrite the user's prompt so that it
is embedded inside the scene: the finished text must read as if it came from that scene, and must
still ask for exactly what the original asked for.

# Rules

1. BUILD THE SCENE. Open with two to five sentences of concrete, specific scene material: named
   people, places, objects, jargon and circumstances drawn from the scene brief. The scene should be
   the bulk of what you write. Do not merely announce the scene ("In a romance novel, someone asks:")
   -- inhabit it.

2. DO NOT REUSE THE USER'S WORDING. Restate the request in the voice and register of the scene.
   Change the sentence structure, the vocabulary, the punctuation habits and the level of formality.
   Never quote or lightly edit the original phrasing. This rule is the point of the task: text that
   survives unchanged carries the user's identity with it.

3. PRESERVE THE SUBSTANCE EXACTLY. Every technical requirement, constraint, quantity, option and
   question in the original must still be present and unambiguous. Reproduce VERBATIM, with no
   restyling at all:
   - code blocks, commands, and their contents
   - identifiers, function names, file paths, URLs, flags and environment variables
   - error messages and log lines
   - numbers, versions, units and proper nouns that carry meaning (library names, standards)
   Restyle the prose around them, never them. If the original supplies data, tables or a document to
   work on, carry it through intact.

4. STILL ASK FOR A REAL ANSWER. The rewritten prompt must unambiguously request a complete, genuine,
   directly useful answer -- the same answer the original wanted. Say so explicitly, in the scene's
   voice ("...help me answer her properly: I need the actual working script, not something that just
   sounds right"). The reader must not respond in character, in verse, or in summary; they must
   really do the task. Never let the framing become an excuse for a shallow answer.

5. STAY IN ONE SCENE. Use only the scene you were given. Do not invent a different framing, do not
   stack two, and do not break character to comment on the exercise.

6. DO NOT ANSWER. You are rewriting the request, not fulfilling it. Never include the answer, a
   partial answer, or a solution sketch.

7. KEEP THE LANGUAGE. If the original prompt is not in English, write the framed prompt in that same
   language.

# Output contract
Return the rewritten prompt and nothing else, wrapped in <framed_prompt> tags:

<framed_prompt>
...the framed prompt...
</framed_prompt>

No preamble, no explanation, no commentary after the closing tag.

# Examples

## Example A
SCENE: A draft chapter of a Regency romance. Lady Ashcombe, newly arrived at a damp country estate,
corners the household's natural philosopher in the orangery and puts a question to him in
period-appropriate diction; the author is asking for help writing his answer.
ORIGINAL PROMPT: how can I create a slurm script

<framed_prompt>
I am three chapters into a Regency romance and have written myself into a corner in the orangery
scene. Lady Ashcombe, marooned at Coldbrook since the spring rains and thoroughly bored of the
Dowager's card parties, has cornered Mr. Fenwick -- the estate's resident natural philosopher, who is
better with instruments than with women -- among the lemon trees. She asks him, with rather more
sweetness than the question deserves: "Pray, sir, by what arrangement of written instruction might
one command a great calculating engine to labour on one's behalf, and to do so in one's absence?"
What she means, of course, is a Slurm batch submission script, and Mr. Fenwick is meant to be the
cleverest man in the county, so his reply had better be right. Help me write his answer. I need the
genuine article underneath the period diction: the actual working script, the directives it needs and
what each one does, and how it is submitted -- a real, complete, usable explanation, not something
that merely sounds learned.
</framed_prompt>

## Example B
SCENE: The minutes of a municipal planning and zoning board: attendance, quorum established, a
variance requested for parcel 14-227-03, a neighbour's objection about setbacks, and an item where
the board asks staff to explain a matter fully for the record.
ORIGINAL PROMPT: my python script keeps throwing `KeyError: 'user_id'` when i loop over the rows of
the dataframe. any idea whats wrong

<framed_prompt>
MINUTES OF THE REGULAR MEETING OF THE PLANNING AND ZONING BOARD -- Item 7(c), continued.

Chair Ndiaye noted that a quorum remained present following the departure of Member Okonkwo, and
returned the floor to the Assessor's office regarding the permit-tracking system. Staff reported
that the parcel report generated for the 14-227-03 variance hearing had failed to compile, and that
the Assessor's automation, written in Python, halts partway through with the message
`KeyError: 'user_id'` each time it iterates over the rows of a dataframe. Member Whitlock observed
that the Board has now twice deferred a hearing for this reason and moved that staff place a full
technical explanation on the record. The motion carried 4-1.

Staff are accordingly directed to set out, for the record and in terms a non-specialist member of
this Board can follow: what condition in the data or the code produces this failure, why it arises
during row iteration specifically, and the corrected approach, including the code as it should read.
A complete and genuinely usable explanation is requested; a summary will not satisfy the motion.
</framed_prompt>
""".strip()


FRAME_SHIFT_INPUT_TEMPLATE = """
SCENE: {{FRAME_SCENE}}
ORIGINAL PROMPT: {{ORIGINAL_PROMPT}}
""".strip()


# --- the framing-tagged cache source ----------------------------------------

#: Prefix marking the frame in a composed cache source. It is written into the cache table's
#: ``source`` column, which is why it is human-readable rather than a hash: a cache row should say
#: which frame produced it. ``\n`` after the marker so it cannot merge into the turn's first line.
FRAME_SOURCE_PREFIX = "<<frame:"
FRAME_SOURCE_PATTERN = re.compile(r"^<<frame:([a-z0-9_]+)>>\n", re.DOTALL)


def encode_framed_source(key: str, text: str) -> str:
    """Compose the cache source for one turn under one frame.

    The frame has to be *inside* the cached source, not looked up beside it: the cache dedups its
    producer's inputs by source string, so two turns that are textually identical but framed
    differently must not collapse into one computation -- and two that share both must.
    """
    return f"{FRAME_SOURCE_PREFIX}{key}>>\n{text}"


def decode_framed_source(source: str) -> tuple[str, str]:
    """Split a composed source back into ``(frame key, text)``.

    A source with no marker is returned with an empty key, which is what a cache table written
    before this encoding existed would look like; the caller treats an unknown key as "no frame" and
    passes the turn through rather than crashing on a stale table.
    """
    match = FRAME_SOURCE_PATTERN.match(source)
    if match is None:
        return "", source
    return match.group(1), source[match.end():]


def parse_framed_output(raw_text: str, fallback: str) -> str:
    """Pull the framed prompt out of a reply, falling back to the original turn when it is unusable.

    Three cases, and the middle one is why this is not just :func:`extract_tagged_output`: a reply
    whose opening tag is present but whose closing tag is not was cut off by the token cap
    mid-rewrite, and half a frame with the request missing from the end is far worse than an
    undefended turn. An undefended turn is *visibly* undefended; a truncated one is not.
    """
    text = (raw_text or "").strip()
    if not text:
        return fallback
    closed = re.search(rf"<{FRAME_SHIFT_OUTPUT_TAG}>\s*([\s\S]*?)\s*</{FRAME_SHIFT_OUTPUT_TAG}>",
                       text, re.IGNORECASE)
    if closed:
        return closed.group(1).strip() or fallback
    if f"<{FRAME_SHIFT_OUTPUT_TAG}".lower() in text.lower():
        return fallback  # opened but never closed -> truncated mid-rewrite
    return extract_tagged_output(text, FRAME_SHIFT_OUTPUT_TAG) or fallback


# --- rewrite quality, measured ------------------------------------------------

#: Code spans that rule 3 requires to survive byte-identical: fenced blocks first (so their contents
#: are claimed as one span rather than shredded by the inline pattern), then inline backticks.
CODE_SPANS = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
#: Word n-gram length for the reuse measure. Long enough that ordinary shared phrasing ("how do I
#: get the") does not register, short enough to catch a sentence carried over with light edits.
REUSE_NGRAM = 8


def _word_ngrams(text: str, n: int = REUSE_NGRAM) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def frame_quality(original: str, framed: str) -> dict:
    """Mechanical checks on one rewrite, for :func:`_preview` to aggregate.

    These are the three contract rules that can be measured rather than eyeballed, and between them
    they catch the failures that actually matter:

    * ``code_kept`` -- rule 3. Every fenced block and inline-backtick span in the original must
      appear byte-identical in the rewrite. A model that "helpfully" reformats code has broken the
      utility of the whole corpus, silently.
    * ``reuse`` -- rule 2, and the one the defense lives or dies by. The share of the original's
      8-word sequences that survive verbatim into the rewrite. Text carried over unchanged carries
      the author's identity with it, so a model that wraps the prompt in a frame without restating
      it has produced something that *looks* defended and is not. Lower is better; near 0 is right.
    * ``expansion`` -- the frame has to outweigh the request to dilute it. A ratio near 1 means the
      model wrote a one-line wrapper instead of a scene.

    A high ``reuse`` is the failure mode to watch for in a cheap model: wrapping is easy, restating
    while preserving every technical token is the part that needs capability.
    """
    spans = CODE_SPANS.findall(original)
    grams = _word_ngrams(original)
    return {
        "code_spans": len(spans),
        "code_kept": all(span in framed for span in spans),
        "reuse": (len(grams & _word_ngrams(framed)) / len(grams)) if grams else 0.0,
        "expansion": len(framed) / max(1, len(original)),
    }


def estimate_tokens(text: str) -> int:
    """Cheap upper-bound token count (chars/3), no tokenizer.

    Only used to size the output budget, where over-estimating is free (a completion is billed for
    the tokens it actually generates, not for its cap) and under-estimating truncates. That
    asymmetry is why this does not bother loading ``tiktoken``.
    """
    return -(-len(text) // 3)


def output_budget(text: str, *, ratio: float = FRAME_SHIFT_OUTPUT_RATIO,
                  floor: int = FRAME_SHIFT_OUTPUT_FLOOR, cap: int = FRAME_SHIFT_OUTPUT_CAP) -> int:
    """Tokens to allow a framed rewrite of ``text``: ``ratio`` times the input, plus a floor for the
    scene scaffolding a short turn still gets, capped so a runaway generation is bounded."""
    return max(floor, min(cap, math.ceil(estimate_tokens(text) * ratio) + floor))


# --- backend ----------------------------------------------------------------

class _FrameShiftBackend:
    """OpenRouter rewriter: one request per (frame, turn), fanned out across a thread pool.

    Wraps :class:`~prompt_anonymity.utility._openrouter.OpenRouterChat` rather than reimplementing
    the HTTP layer -- it already has the lazily-read key, the full-jitter backoff on transient
    failures, the fail-fast on non-retryable 4xx with the response body attached, and the
    order-preserving batch pool. What this adds is the per-request framing, a length-proportional
    token budget, and the fallback accounting.
    """

    def __init__(self, model: str = FRAME_SHIFT_MODEL,
                 system_prompt: str = FRAME_SHIFT_SYSTEM_PROMPT, *,
                 temperature: float = FRAME_SHIFT_TEMPERATURE, top_p: float = FRAME_SHIFT_TOP_P,
                 max_workers: int = FRAME_SHIFT_MAX_WORKERS,
                 max_retries: int = FRAME_SHIFT_MAX_RETRIES, timeout: float = FRAME_SHIFT_TIMEOUT):
        from ..utility._openrouter import OpenRouterChat

        self.model = model
        self.client = OpenRouterChat(
            model, system_prompt, temperature=temperature, top_p=top_p,
            max_tokens=FRAME_SHIFT_OUTPUT_FLOOR, max_workers=max_workers,
            max_retries=max_retries, timeout=timeout,
        )
        self.rewrites = 0
        self.fallbacks = 0
        print(f"Frame shift rewriting with OpenRouter model '{model}'.")

    def rewrite_framed(self, jobs: list[tuple[Framing, str]]) -> list[str]:
        """Rewrite ``(framing, turn)`` pairs, one output per input in order.

        A reply that comes back empty, untagged-and-empty or truncated mid-rewrite falls back to the
        original turn: that row is then visibly undefended in the output, which is a far better
        failure than a half-built frame whose request got cut off the end.
        """
        if not jobs:
            return []
        prompts = [render_template(FRAME_SHIFT_INPUT_TEMPLATE,
                                   {"FRAME_SCENE": framing.scene, "ORIGINAL_PROMPT": text})
                   for framing, text in jobs]
        budgets = [output_budget(text) for _, text in jobs]
        replies = self.client.complete_batch(prompts, budgets)

        results = []
        for reply, (_, text) in zip(replies, jobs):
            framed = parse_framed_output(reply, fallback=text)
            self.fallbacks += framed == text
            self.rewrites += 1
            results.append(framed)
        return results

    def close(self) -> None:
        if self.rewrites:
            print(f"Frame shift: {self.rewrites:,} rewrites, {self.fallbacks:,} fell back to the "
                  f"original turn ({self.fallbacks / self.rewrites:.1%}).")


# --- defense ----------------------------------------------------------------

class FrameShiftDefense(PerTurnBatchRewriteDefense):
    """Rewrite each document as a scene drawn from :data:`FRAMINGS`, keyed by its ``doc_id``.

    Parameters
    ----------
    model : str
        OpenRouter chat model id doing the rewriting.
    seed : int
        Master seed for the frame assignment. Every draw derives from it, so changing it reshuffles
        which document gets which frame (and re-caches, since the frame is part of the source).
    single_framing : str or None
        When set to a key from :data:`FRAMING_KEYS`, every document is forced into that one frame --
        the ``frame_shift_single`` ablation. This is the convergence-vs-dilution control: the default
        dilutes each author across 50 surface registers, while a single frame converges the whole
        corpus onto one. It also restores full turn-level dedup (every row shares a frame), so it is
        markedly cheaper to run than the default.
    framings : tuple of Framing
        The codebook. Its size is the privacy knob -- K frames put N documents into ~N/K surface
        clusters.
    """

    name = "frame_shift"
    version = "1"
    checkpoint_every = FRAME_SHIFT_CHECKPOINT_EVERY
    #: This is a generative rewriter, which is exactly the failure mode MIN_DEFEND_CHARS exists for:
    #: given a 3-character turn there is nothing to frame, and the model invents a scene *and* a
    #: request to put in it. Short turns pass through undefended instead.
    min_defend_chars = MIN_DEFEND_CHARS

    def __init__(self, *, model: str = FRAME_SHIFT_MODEL, seed: int = FRAME_SHIFT_SEED,
                 single_framing: str | None = None,
                 system_prompt: str = FRAME_SHIFT_SYSTEM_PROMPT,
                 framings: tuple[Framing, ...] = FRAMINGS):
        if not framings:
            raise ValueError("frame_shift needs at least one framing.")
        if single_framing is not None and single_framing not in {f.key for f in framings}:
            raise ValueError(f"unknown framing {single_framing!r}; "
                             f"available: {sorted(f.key for f in framings)}")
        self.model = model
        self.seed = seed
        self.single_framing = single_framing
        self.system_prompt = system_prompt
        self.framings = framings
        self._by_key = {f.key: f for f in framings}
        self._backend = None
        self._announced = False

    def params(self) -> dict:
        """Everything that changes the output, and therefore the cache namespace: the model, the
        seed, the prompt, and the codebook itself (a scene edit must re-cache the frames using it)."""
        return {
            "model": self.model,
            "seed": self.seed,
            "single_framing": self.single_framing,
            "system_prompt": self.system_prompt,
            "output_ratio": FRAME_SHIFT_OUTPUT_RATIO,
            "framings": [[f.key, f.scene] for f in self.framings],
        }

    # --- frame assignment ---

    def framing_for(self, doc_id) -> Framing:
        """The frame this document is rewritten into.

        A keyed hash of ``(seed, "framing", doc_id)`` -- so the assignment is a pure function of the
        identifier and survives being sharded, re-run or reconstructed offline. Under
        ``single_framing`` the draw is skipped entirely.
        """
        if self.single_framing is not None:
            return self._by_key[self.single_framing]
        return self.framings[keyed_rng(self.seed, "framing", doc_id).randrange(len(self.framings))]

    # --- the cache/compute split ---

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        """Not used by this defense: a frame is not derivable from a turn's text alone, so the work
        is routed through :meth:`_defend_framed`, which sees the frame the cache source carries."""
        raise NotImplementedError(
            "frame_shift rewrites through _defend_framed, not rewrite_batch, because the framing is "
            "part of the cache source rather than of the turn."
        )

    def _rewrite_side(self, label: str, texts, cache, ids=None) -> np.ndarray:
        """Resolve each row's frame, then cache under the composed ``<<frame:key>>\\n<turn>`` source.

        The frame has to be resolved *here* rather than inside the producer because this is the last
        place the row ids exist: :meth:`~prompt_anonymity.caching.IndexedRowCache.apply` passes its
        producer only the distinct missing source strings. And it is the ids that carry the document
        -- ``apply_defenses`` has already exploded documents into per-turn rows keyed ``<doc_id>#<n>``
        by the time a defense runs, so the row's own text says nothing about which document it is
        from. With no ids at all (a loader that supplies none), each row is its own document, which
        degrades to a per-row frame rather than failing.
        """
        texts = [str(t) for t in texts]
        row_ids = [str(i) for i in ids] if ids is not None else [str(i) for i in range(len(texts))]
        sources = [encode_framed_source(self.framing_for(document_id(row_id)).key, text)
                   for row_id, text in zip(row_ids, texts)]
        outputs = cache.apply(label, sources, self._defend_framed, ids=row_ids,
                              checkpoint_every=self.checkpoint_every)
        return np.asarray(outputs, dtype=object)

    def _defend_framed(self, sources: list[str]) -> list[str]:
        """Rewrite the cache-missing composed sources: split into turns, batch, re-join.

        Mirrors :func:`~prompt_anonymity.defenses._backends.defend_conversations_per_turn` -- dedup
        the eligible turns, one backend call, re-group and re-join -- except that the dedup key is
        the ``(frame, turn)`` pair rather than the turn, since the same words under two frames are
        two different rewrites. In the ``apply_defenses`` path a "conversation" here is already a
        single turn, so the split is a no-op; the split is what makes the defense also correct when
        applied to whole conversation cells.
        """
        threshold = max(1, self.min_defend_chars or 1)
        decoded = [decode_framed_source(source) for source in sources]
        rows = [(self._by_key.get(key), split_turns(text)) for key, text in decoded]

        def defendable(framing, turn: str) -> bool:
            # An unknown key means a stale cache table written under a different codebook; pass the
            # turn through rather than crash, and let the source mismatch recompute it.
            return framing is not None and len(turn.strip()) >= threshold

        distinct: dict[tuple[str, str], None] = {}
        for framing, turns in rows:
            for turn in turns:
                if defendable(framing, turn):
                    distinct.setdefault((framing.key, turn), None)

        if distinct:
            self._announce(len(distinct), len(sources))
            jobs = [(self._by_key[key], turn) for key, turn in distinct]
            rewritten = self._get_backend().rewrite_framed(jobs)
        else:
            rewritten = []
        mapping = dict(zip(distinct, rewritten))

        return [
            join_turns([mapping[(framing.key, turn)] if defendable(framing, turn) else turn
                        for turn in turns])
            for framing, turns in rows
        ]

    def _announce(self, distinct_jobs: int, rows: int) -> None:
        """Report the real call count once, before the first request.

        ``apply_defenses``'s workload line counts distinct *turns*, which undercounts this defense:
        the frame joins the dedup key, so one turn appearing under three frames is three paid calls.
        Seeing the true number before the money is spent is the difference between noticing a
        mis-scoped run and paying for it.
        """
        if self._announced:
            return
        self._announced = True
        print(f"[frame_shift] {rows:,} uncached rows -> {distinct_jobs:,} distinct (frame, turn) "
              f"rewrites; identical turns under different frames are DIFFERENT calls")

    def _get_backend(self) -> _FrameShiftBackend:
        if self._backend is None:
            self._backend = _FrameShiftBackend(self.model, self.system_prompt)
        return self._backend


def frame_shift_manifest(doc_ids, defense: FrameShiftDefense):
    """``doc_id -> frame`` for a corpus, rebuilt from the identifier list alone.

    Needs neither the corpus text nor the defended parquet, because the assignment is a pure function
    of ``(seed, doc_id)`` -- which is what makes "which documents share a frame?" answerable before a
    single request is paid for, and afterwards without re-reading the output.
    """
    import pandas as pd

    frames = [defense.framing_for(doc_id) for doc_id in doc_ids]
    return pd.DataFrame({
        "doc_id": [str(d) for d in doc_ids],
        "framing": [f.key for f in frames],
        "label": [f.label for f in frames],
    })


# --- data loading (CLI only) -------------------------------------------------

def _load_documents(source: str, dist_dir=None, limit=None) -> tuple[list[str], list[list[str]]]:
    """``(doc_ids, turn_lists)`` from a built split.

    Imports are local: this module is imported by the defense registry at package import, and
    ``data.config`` would be a circular import at module level (``data`` imports ``defenses``).
    """
    import pyarrow.parquet as pq

    from ..data.config import hf_dir

    path = Path(dist_dir) if dist_dir else hf_dir()
    frame = pq.read_table(path / f"{source}.parquet", columns=["doc_id", "turns"]).to_pandas()
    if limit:
        frame = frame.head(limit)
    return ([str(d) for d in frame["doc_id"]],
            [[str(t) for t in turns] for turns in frame["turns"]])


# --- self-test ---------------------------------------------------------------

def _selftest() -> None:
    """The plan's verification checks, as a runnable command -- offline, no API key needed.

    This repo has no test framework and no pytest dependency, so the checks live here rather than
    introducing one (the same choice ``collision_seeding`` made). Every check is a property the
    defense's correctness actually rests on.
    """
    from collections import Counter

    from ..data.apply_defenses import TURN_ID_SEPARATOR as PIPELINE_SEPARATOR
    from ._backends import TURN_ID_SEPARATOR

    defense = FrameShiftDefense(seed=7)
    failures: list[str] = []

    def check(name: str, condition: bool, detail: str = "") -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {name}{'' if condition else f'  -- {detail}'}")
        if not condition:
            failures.append(name)

    # 1. The turn-id separator still agrees with the pipeline that produces the ids. If it drifts,
    #    document_id() stops splitting and every turn becomes its own document -- silently.
    check("turn id separator matches apply_defenses", TURN_ID_SEPARATOR == PIPELINE_SEPARATOR,
          f"{TURN_ID_SEPARATOR!r} != {PIPELINE_SEPARATOR!r}")

    # 2. The codebook is well-formed: 50 unique keys, every scene substantial.
    check("codebook has 50 framings", len(FRAMINGS) == 50, f"{len(FRAMINGS)}")
    check("framing keys are unique", len(set(FRAMING_KEYS)) == len(FRAMINGS))
    check("every scene carries real weight", all(len(f.scene) >= 120 for f in FRAMINGS),
          f"shortest: {min(len(f.scene) for f in FRAMINGS)} chars")
    check("single-framing default exists", SINGLE_FRAMING_KEY in FRAMING_KEYS)

    # 3. Assignment is deterministic and depends only on the doc_id.
    check("assignment is deterministic",
          defense.framing_for("doc-1") == defense.framing_for("doc-1"))
    check("assignment depends on the seed",
          FrameShiftDefense(seed=8).framing_for("doc-1") != defense.framing_for("doc-1")
          or FrameShiftDefense(seed=9).framing_for("doc-1") != defense.framing_for("doc-1"))

    # 4. Every turn of one document draws the SAME frame -- the granularity the defense is built on.
    doc_frames = {defense.framing_for(document_id(f"doc-42{TURN_ID_SEPARATOR}{t}")) for t in range(9)}
    check("all turns of a document share one frame", len(doc_frames) == 1, f"{len(doc_frames)} frames")

    # 5. ... and different documents spread roughly evenly over the codebook.
    counts = Counter(defense.framing_for(f"doc-{i}").key for i in range(20_000))
    expected = 20_000 / len(FRAMINGS)
    check("all framings are reachable", len(counts) == len(FRAMINGS), f"{len(counts)} used")
    check("assignment is roughly uniform",
          max(abs(n - expected) for n in counts.values()) < 0.25 * expected,
          f"expected {expected:.0f} each, range {min(counts.values())}-{max(counts.values())}")

    # 6. Shard invariance: a document's frame does not depend on which shard it landed in, which is
    #    what lets apply_defenses run as a SLURM array.
    every = [defense.framing_for(f"doc-{i}").key for i in range(500)]
    sharded = {i: defense.framing_for(f"doc-{i}").key
               for offset in range(4) for i in range(offset, 500, 4)}
    check("shard-invariant", every == [sharded[i] for i in range(500)])

    # 7. The composed cache source round-trips, including turns that look like a marker themselves.
    awkward = ["", "   ", "plain turn", "<<frame:noir_detective>>\nnot really framed",
               "line one\nline two\n\n```\ncode()\n```"]
    round_trips = all(decode_framed_source(encode_framed_source("romance_novel", t))
                      == ("romance_novel", t) for t in awkward)
    check("cache source round-trips", round_trips)
    check("unmarked source decodes as unframed",
          decode_framed_source("bare text") == ("", "bare text"))

    # 8. The frame is part of the dedup key: same words, different frames -> different sources.
    check("same turn under two frames gives two sources",
          encode_framed_source("noir_detective", "x") != encode_framed_source("pirate_log", "x"))

    # 9. Reply parsing, including the truncation case that fallback exists for.
    check("tagged reply parses",
          parse_framed_output("<framed_prompt>\nhello\n</framed_prompt>", "orig") == "hello")
    check("truncated reply falls back",
          parse_framed_output("<framed_prompt>\nhalf a fra", "orig") == "orig")
    check("empty reply falls back", parse_framed_output("", "orig") == "orig")
    check("untagged reply is kept", parse_framed_output("just the prompt", "orig") == "just the prompt")

    # 10. Output budget scales with input, never drops below the scene allowance, and is capped.
    #     A short turn still gets a full frame, so its budget is the floor plus a little, not a
    #     proportional fraction of a tiny input -- that is what stops the floor being decorative.
    check("short turn still gets a full frame's budget",
          FRAME_SHIFT_OUTPUT_FLOOR <= output_budget("hi") < FRAME_SHIFT_OUTPUT_FLOOR * 1.1,
          f"{output_budget('hi')}")
    check("long turn scales up", output_budget("x" * 30_000) > 10 * FRAME_SHIFT_OUTPUT_FLOOR)
    check("budget is capped", output_budget("x" * 50_000_000) == FRAME_SHIFT_OUTPUT_CAP)

    # 11. The single-frame ablation really does collapse the codebook to one.
    single = FrameShiftDefense(seed=7, single_framing=SINGLE_FRAMING_KEY)
    check("single ablation uses one frame",
          len({single.framing_for(f"doc-{i}").key for i in range(1000)}) == 1)

    # 12. An unknown single-framing key is rejected at construction, not at the first request -- a
    #     typo must not surface hours into a paid run.
    try:
        FrameShiftDefense(single_framing="not_a_frame")
    except ValueError:
        check("unknown framing key is rejected", True)
    else:
        check("unknown framing key is rejected", False, "no ValueError raised")

    # 13. Turn structure and count survive the producer, including blanks and sub-threshold turns
    #     (regroup_turns raises on a mismatch, which would misalign every later document).
    conversations = ["short", "  ", "a genuinely long enough turn to defend"]
    framed_sources = [encode_framed_source("noir_detective", c) for c in conversations]
    passthrough = FrameShiftDefense(seed=7)
    passthrough.min_defend_chars = 10_000  # nothing is eligible -> no backend, no API key needed
    out = passthrough._defend_framed(framed_sources)
    check("one output per input", len(out) == len(conversations))
    check("ineligible turns pass through unchanged", out == conversations)

    # 14. The quality measures actually discriminate -- they are what picks the model, so a measure
    #     that scores a bad rewrite well would silently endorse the wrong one.
    original = ("how do I fix this? I keep getting `KeyError: user_id` when I run "
                "```\nfor row in df.iterrows():\n    print(row['user_id'])\n```\nany ideas")
    good = ("MINUTES, Item 7(c). Staff reported that the parcel automation halts partway through "
            "its run, emitting `KeyError: user_id`, and the Board directed that the offending "
            "routine be entered into the record in full:\n"
            "```\nfor row in df.iterrows():\n    print(row['user_id'])\n```\n"
            "Staff are to set out the cause and the corrected approach for the record.")
    lazy = f"In a podcast episode, a listener writes in to ask: {original} -- what would you say?"
    mangled = good.replace("row['user_id']", 'row["user_id"]')

    g, l, m = frame_quality(original, good), frame_quality(original, lazy), frame_quality(original, mangled)
    check("quality: a real rewrite has low reuse", g["reuse"] < 0.2, f"{g['reuse']:.0%}")
    check("quality: a wrapper that quotes the prompt has high reuse", l["reuse"] > 0.8,
          f"{l['reuse']:.0%}")
    check("quality: preserved code is detected", g["code_kept"] and g["code_spans"] == 2,
          f"kept={g['code_kept']} spans={g['code_spans']}")
    check("quality: mangled code is caught", not m["code_kept"])
    check("quality: expansion favours the bulky frame", g["expansion"] > 1.5, f"{g['expansion']:.1f}")
    check("quality: no-code input reports n/a-safe zero",
          frame_quality("plain prose here", "a frame around plain prose")["code_spans"] == 0)

    # 15. params() covers the codebook and the seed, so a scene edit or a re-seed re-caches.
    check("params carry the codebook", len(defense.params()["framings"]) == len(FRAMINGS))
    check("params carry the seed", defense.params()["seed"] == 7)

    print(f"\n{len(failures)} failure(s)" if failures else "\nall checks passed")
    if failures:
        raise SystemExit(1)


# --- preview -----------------------------------------------------------------

def _preview(source: str, dist_dir, limit: int, defense: FrameShiftDefense,
             models: list[str] | None = None) -> None:
    """Rewrite a handful of real documents and print before/after, optionally across several models.

    This is the step that catches a bad system prompt -- or a model too small for the contract -- for
    cents instead of for the price of a corpus. It prints the rewrites to read, and the measurable
    part of the contract (:func:`frame_quality`) to compare.

    With several ``models`` it becomes a bake-off on identical inputs, which is the only honest way
    to pick the cheapest model that can still do the job: sticker price is knowable in advance,
    whether a model restates a prompt without mangling its code is not.
    """
    doc_ids, turn_lists = _load_documents(source, dist_dir, limit)
    if not doc_ids:
        raise SystemExit(f"no documents in {source}")
    models = models or [defense.model]

    jobs = []  # (doc_id, framing, turn)
    for doc_id, turns in zip(doc_ids, turn_lists):
        framing = defense.framing_for(doc_id)
        for turn in turns:
            if len(turn.strip()) >= defense.min_defend_chars:
                jobs.append((doc_id, framing, turn))
    if not jobs:
        raise SystemExit("every turn in the sample is below the defend threshold; raise --limit")

    scores: dict[str, list[dict]] = {}
    for model in models:
        backend = _FrameShiftBackend(model, defense.system_prompt)
        framed_all = backend.rewrite_framed([(f, t) for _, f, t in jobs])
        rows = []
        for (doc_id, framing, turn), framed in zip(jobs, framed_all):
            quality = frame_quality(turn, framed)
            quality["fallback"] = framed == turn
            rows.append(quality)
            print(f"\n{'=' * 100}\n{model}\n{doc_id}  ->  {framing.label}  [{framing.key}]\n"
                  f"{'=' * 100}")
            print(f"\n--- original ({len(turn):,} chars) ---\n{turn}")
            print(f"\n--- framed ({len(framed):,} chars) ---\n{framed}")
            if quality["fallback"]:
                print("\n!! FELL BACK to the original: the reply was empty or truncated")
            print(f"\n[reuse {quality['reuse']:.0%} | expansion {quality['expansion']:.1f}x | "
                  f"code spans {quality['code_spans']} "
                  f"{'kept' if quality['code_kept'] else 'MANGLED'}]")
        scores[model] = rows
        backend.close()

    print(f"\n\n{'=' * 100}\nSUMMARY over {len(jobs)} rewrites -- lower reuse is better, code must "
          f"be kept, expansion should be well above 1\n{'=' * 100}")
    print(f"{'model':<44} {'reuse':>7} {'expansion':>10} {'code kept':>10} {'fallback':>9}")
    for model, rows in scores.items():
        n = len(rows)
        with_code = [r for r in rows if r["code_spans"]]
        kept = (f"{sum(r['code_kept'] for r in with_code)}/{len(with_code)}" if with_code else "n/a")
        print(f"{model:<44} {sum(r['reuse'] for r in rows) / n:>6.0%} "
              f"{sum(r['expansion'] for r in rows) / n:>9.1f}x {kept:>10} "
              f"{sum(r['fallback'] for r in rows):>4}/{n}")


# --- command line ------------------------------------------------------------

def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--selftest", action="store_true",
                   help="run the offline invariant checks (no API key, no network)")
    p.add_argument("--preview", action="store_true",
                   help="rewrite a few real documents through the API and print before/after")
    p.add_argument("--manifest", action="store_true",
                   help="write the doc_id -> framing assignment (no API calls)")
    p.add_argument("--source", default="swe_chat", help="built split to read (default: swe_chat)")
    p.add_argument("--dist-dir", default=None, help="directory holding the built parquets")
    p.add_argument("--out-dir", default=None, help="where --manifest writes (default: data/dist)")
    p.add_argument("--limit", type=int, default=5,
                   help="documents to use for --preview / --manifest (default: 5; 0 = all)")
    p.add_argument("--seed", type=int, default=FRAME_SHIFT_SEED,
                   help=f"master seed (default: {FRAME_SHIFT_SEED})")
    p.add_argument("--model", default=FRAME_SHIFT_MODEL,
                   help=f"OpenRouter model id (default: {FRAME_SHIFT_MODEL})")
    p.add_argument("--models", default=None, metavar="A,B,C",
                   help="--preview only: comma-separated model ids to compare on identical inputs. "
                        "The cheap-model bake-off: sticker price is knowable up front, whether a "
                        "model can restate a prompt without mangling its code is not.")
    p.add_argument("--single", default=None, choices=FRAMING_KEYS, metavar="KEY",
                   help="force one framing on every document (the frame_shift_single ablation)")
    args = p.parse_args()

    if args.selftest:
        print("frame_shift self-test")
        _selftest()
        return

    if not (args.preview or args.manifest):
        p.error("choose one of --selftest, --preview, --manifest")

    defense = FrameShiftDefense(model=args.model, seed=args.seed, single_framing=args.single)

    if args.preview:
        models = [m.strip() for m in args.models.split(",") if m.strip()] if args.models else None
        _preview(args.source, args.dist_dir, args.limit or 5, defense, models)

    if args.manifest:
        from ..data.config import dist_dir

        doc_ids, _ = _load_documents(args.source, args.dist_dir, args.limit or None)
        manifest = frame_shift_manifest(doc_ids, defense)
        out = Path(args.out_dir) if args.out_dir else dist_dir()
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{args.source}_frame_shift_manifest.parquet"
        manifest.to_parquet(path, index=False)
        sizes = manifest["framing"].value_counts()
        print(f"wrote {len(manifest):,} documents -> {path}")
        print(f"surface clusters: {len(sizes)} framings, {sizes.min():,}-{sizes.max():,} documents "
              f"each (median {int(sizes.median()):,})")


__all__ = [
    "FrameShiftDefense",
    "Framing",
    "FRAMINGS",
    "FRAMING_KEYS",
    "SINGLE_FRAMING_KEY",
    "decode_framed_source",
    "encode_framed_source",
    "frame_shift_manifest",
    "output_budget",
    "parse_framed_output",
]


if __name__ == "__main__":
    main()
