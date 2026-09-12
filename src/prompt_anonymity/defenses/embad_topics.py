"""Build EmBad's decoy-subject pool from Library of Congress Subject Headings.

EmBad appends one short trigger turn to a document, and every trigger names a **decoy subject** --
the thing it asserts the document is really about (see :data:`~prompt_anonymity.defenses.embad
.DECOY_TOPIC`). Until now that subject was a single hardcoded string, which makes every measured
result conditional on one arbitrary choice of topic. This module builds the pool the search draws
from instead, so a trigger can be optimised over many subjects and *validated on subjects it never
saw*.

Two stages, deliberately separable
==================================
``--harvest`` needs the network and no GPU; ``--expand`` needs a GPU and no network. They are one
pipeline but never one job, because on a cluster they belong on different nodes.

1. **Harvest** (:func:`harvest_seeds`) streams the Library of Congress's bulk MADS-RDF export and
   keeps the headings typed ``madsrdf:Topic``. LCSH is a controlled vocabulary of *what a document
   can be about*, maintained by librarians for exactly the question being asked here, so the
   diversity comes from the vocabulary rather than from a language model's sampler -- which is the
   whole point. An LLM asked to invent 100,000 topics mode-collapses; an LLM handed 100,000
   externally-supplied subjects cannot.

2. **Expand** (:class:`TopicExpander`) rewrites each heading into the pool's format with a local
   Qwen. This is a constrained rewrite of a supplied subject, not open-ended generation: normalise
   the heading's cataloguing conventions into plain English, then name three concrete things inside
   it. A 9B model is comfortable with that and would not be trusted with the first stage.

Why the MADS-RDF export and not the SKOS one
============================================
LC publishes both. **Take MADS.** The SKOS export (``subjects.skosrdf.jsonld.gz``) types every
heading as a bare ``skos:Concept``, which leaves the ~50% of LCSH that is place names, family
names and building names to be separated from real subject matter by pattern-matching the label --
measured here as regex whack-a-mole that still leaked ``Toppenish Creek (Wash.)`` and
``Maquoketa River (Iowa)``. MADS types each heading structurally, so the filter is a field lookup.
Measured over a 4 MB slice (12,736 typed records):

===========================  ======  ====================================================
type                          share   disposition
===========================  ======  ====================================================
``madsrdf:Topic``               41%   kept -- this is the pool's source
``madsrdf:Geographic`` (both)   35%   dropped: place names
``madsrdf:ComplexSubject``      15%   dropped for now; see :data:`COMPLEX_SUBJECTS_NOTE`
``FamilyName``/``CorporateName`` 9%   dropped: families, buildings, institutions
===========================  ======  ====================================================

The full export is ~140 MB gzipped, which projects to roughly **447,000 headings, ~181,000 of them
``Topic``** -- more than the search can consume, which is the point of a sampling frame.

What the format has to satisfy
==============================
Two things, and **length is deliberately not one of them.**

* Every entry keeps the *shape* ``head, facet, facet and facet``. That is what
  :data:`TOPIC_PATTERN` enforces, and it matters for a concrete reason: EmBad's render guard is
  ``topic.split(",")[0] in rendered``, so an entry needs comma structure with a **distinctive head
  term** or the guard proves nothing. A bare ``"cooking"`` would match incidental document text.
* A decoy has to be *far* from the corpora being defended (software chat, general assistant chat).
  LCSH's bias toward what books are written about -- humanities, history, material culture,
  natural history -- helps here rather than hurting.

An earlier version of this module banded entries tightly (45-80 characters) on the grounds that
``embad.MAX_TOKENS`` capped a rendered trigger, so a long subject could truncate away
and make a candidate's viability a function of the topic draw. **That reasoning is retired, and the cap itself was removed on 2026-09-07**: the
64-token cap is an inheritance from the GCG-style search this work started from, not a property of
the evolutionary search, which is free to evolve a trigger of whatever length it wants. The band
that remains (:data:`TOPIC_MIN_CHARS`--:data:`TOPIC_MAX_CHARS`) is a **sanity bound** catching a
degenerate or runaway generation, not a design constraint -- so it is wide, and nothing should be
tuned against it.

Reproducibility
===============
The harvest is a deterministic filter over a versioned export. The expansion decodes **greedily**
(temperature 0), as :mod:`~prompt_anonymity.defenses.styleremix` does and for the same reason: a
pool that changes between builds would silently split the arms of any comparison drawn across it.
Sampling from the pool is the search's job and is seeded there.

Usage
=====

.. code-block:: shell

    # stage 1 -- network, no GPU (~140 MB download, cached)
    python -m prompt_anonymity.defenses.embad_topics --harvest

    # stage 2 -- GPU, no network. Ten seeds, which is the smoke test.
    python -m prompt_anonymity.defenses.embad_topics --expand --limit 10 --show

    # the real build
    python -m prompt_anonymity.defenses.embad_topics --expand --limit 0

    python -m prompt_anonymity.defenses.embad_topics --selftest   # offline, no GPU, no network

The built pool lands at :data:`PACKAGED_POOL` and is committed. ``.gitignore`` ignores ``*.csv``
globally, so adding it needs ``git add -f``, exactly as ``frame_pad_bank.csv`` did.
"""

from __future__ import annotations

import csv
import gzip
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------------------------
# The source
# --------------------------------------------------------------------------------------------

#: The Library of Congress's bulk MADS-RDF export of LCSH, as gzipped JSON-LD, one record per line.
#: ~140 MB. Served as a 303 redirect to S3, which ``urllib`` follows on its own.
LCSH_URL = "https://id.loc.gov/download/authorities/subjects.madsrdf.jsonld.gz"

#: Local filename for the cached export. Kept under ``<data>/.cache/`` because it is a downloaded
#: build input, regenerable and never committed.
LCSH_FILENAME = "subjects.madsrdf.jsonld.gz"

#: Every heading's IRI starts with this. Records in the export also mention *other* authorities
#: (broader terms, sources, collections) inside the same ``@graph``, so the subject of a record is
#: identified by its IRI rather than by being first.
SUBJECT_IRI_PREFIX = "http://id.loc.gov/authorities/subjects/sh"

#: The MADS type that means "a topical subject" as opposed to a place, a family or an institution.
#: Types arrive compacted against the record's ``@context`` (``["madsrdf:Authority",
#: "madsrdf:Topic"]``), so this is compared as a string rather than resolved as an IRI.
TOPIC_TYPE = "madsrdf:Topic"

#: Where a record keeps the heading itself, as ``{"@language": "en", "@value": ...}``.
LABEL_KEY = "madsrdf:authoritativeLabel"

#: LCSH retires headings without deleting them. A deprecated one is still valid English but is no
#: longer the vocabulary's answer for its subject, and there is no reason to seed from it.
DEPRECATED_KEY = "owl:deprecated"

#: Why ``madsrdf:ComplexSubject`` is not harvested, though it is 15% of the export and holds some of
#: its best material (``Choral singing--Studies and exercises``, ``String craft--Japan``, ``Art,
#: Chinese--Western influences``). A complex subject is a *precoordinated string* of facets joined
#: by ``--``, and a large share of those facets are geographic (``Marketplaces--Poland``,
#: ``Faults (Geology)--Maine``), which is the material the MADS typing exists to exclude. Harvesting
#: them means re-deriving that judgement per facet -- a second filter with its own error rate -- for
#: subjects the ``Topic`` type already supplies ~181,000 of. Revisit only if the pool turns out to
#: be too narrow, which a sample audit would show before a search ever did.
COMPLEX_SUBJECTS_NOTE = "complex subjects are precoordinated facet strings; many facets are places"

# --------------------------------------------------------------------------------------------
# What a usable seed looks like
# --------------------------------------------------------------------------------------------

#: Seed length band, in characters. The floor drops single words too generic to anchor a decoy
#: (``Art``, ``Law``); the ceiling drops headings so specific that the expansion has nothing left to
#: add and simply restates them.
SEED_MIN_CHARS = 6
SEED_MAX_CHARS = 60

#: Headings the MADS typing keeps but that are not subject matter. Small on purpose: the structural
#: filter does nearly all the work, and every pattern here is a residue observed in a sample rather
#: than a category guessed at in advance.
SEED_REJECTIONS = (
    ("fictitious character",
     re.compile(r"\((?:Fictitious|Legendary|Mythical|Biblical|Greek myth|Norse myth)[^)]*\)", re.I)),
    ("named award or vessel",
     re.compile(r"\b(Awards?|Prize|Trophy|Medal|Scholarship|Fellowship)\b")),
    ("cataloguing artifact",
     re.compile(r"^(?:Miscellanea|Terms and phrases|Names,|Abbreviations of)\b")),
)


def is_non_ascii(heading: str) -> bool:
    """Whether a heading carries any character outside ASCII.

    In LCSH this is overwhelmingly one thing: a romanised non-English title written with combining
    marks and half-rings (``Inspekt͡sii͡a medit͡sinskai͡a germenevtika``, ``Konakŭt na Salikh
    aga``). Those are legitimate headings and useless decoys -- the subject is unreadable to anyone
    who does not already know the work -- and they were 4.8% of a measured slice.

    **The rule is deliberately blunter than the intent.** An NFKD fold was tried first, to keep an
    ordinary English heading that merely carries an accent (``Cafe`` from ``Café``) while dropping
    the transliterations; it does not work, because a transliteration folds to ASCII base characters
    just as cleanly as an accent does -- the combining marks are exactly what NFKD strips. There is
    no cheap test that separates the two, so the whole class goes. The cost is a handful of accented
    English headings out of ~181,000; the benefit is that it matches what the pool itself requires,
    since :func:`topic_rejection` rejects a non-ASCII entry anyway.
    """
    return any(ord(character) > 0x7F for character in heading)


def seed_rejection(heading: str) -> str | None:
    """Why this heading is not usable as a seed, or ``None`` when it is.

    Returns the *reason* rather than a boolean so a build can report what it dropped and how much
    of each -- a filter whose rejection profile nobody looks at is a filter nobody can tune.
    """
    if not (SEED_MIN_CHARS <= len(heading) <= SEED_MAX_CHARS):
        return "length band"
    if is_non_ascii(heading):
        return "non-ASCII heading"
    for reason, pattern in SEED_REJECTIONS:
        if pattern.search(heading):
            return reason
    return None


# --------------------------------------------------------------------------------------------
# Harvesting
# --------------------------------------------------------------------------------------------

def lcsh_path(cache: Path | None = None) -> Path:
    """Where the downloaded export is cached."""
    from ..data.config import cache_dir
    root = Path(cache) if cache is not None else cache_dir() / "embad_topics"
    return root / LCSH_FILENAME


def download_lcsh(destination: Path, *, max_bytes: int = 0, force: bool = False) -> Path:
    """Fetch the LCSH export to ``destination``, or reuse what is already there.

    Downloads through a ``.part`` file and renames on success, so an interrupted fetch never leaves
    a truncated file that looks complete to the next run. ``max_bytes`` requests a byte range
    instead of the whole export, which is the fast path for a smoke test: a partial gzip stream
    decodes fine until it runs out, and :func:`harvest_seeds` treats that as the end of the file
    rather than as an error.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        print(f"[embad_topics] using the cached export at {destination} "
              f"({destination.stat().st_size / 1e6:.1f} MB); pass --force to re-download")
        return destination
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        LCSH_URL, headers={"User-Agent": "prompt-anonymity/embad_topics (research)"})
    if max_bytes:
        request.add_header("Range", f"bytes=0-{max_bytes - 1}")
    print(f"[embad_topics] downloading {LCSH_URL}"
          f"{f' (first {max_bytes / 1e6:.1f} MB)' if max_bytes else ' (~140 MB)'}...")
    with urllib.request.urlopen(request, timeout=120) as response, open(partial, "wb") as out:
        copied = 0
        while chunk := response.read(1 << 20):
            out.write(chunk)
            copied += len(chunk)
    partial.replace(destination)
    print(f"[embad_topics] wrote {destination} ({copied / 1e6:.1f} MB)")
    return destination


def iter_headings(path: Path) -> "list[tuple[str, str]]":
    """Every ``(iri, heading)`` typed :data:`TOPIC_TYPE` in the export, in file order.

    Streams line by line: the export is one JSON object per line and never has to be held whole.
    A truncated file -- which is what ``--max-bytes`` produces on purpose -- ends the stream with a
    decompression error, and that is reported and treated as the end rather than raised, so the
    smoke path and the real build run the same code.
    """
    found: list[tuple[str, str]] = []
    try:
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue  # the last line of a truncated slice
                for node in record.get("@graph", []):
                    iri = node.get("@id", "")
                    if not iri.startswith(SUBJECT_IRI_PREFIX):
                        continue
                    types = node.get("@type", [])
                    types = [types] if isinstance(types, str) else types
                    if TOPIC_TYPE not in types or node.get(DEPRECATED_KEY):
                        continue
                    label = node.get(LABEL_KEY)
                    label = label.get("@value") if isinstance(label, dict) else label
                    if isinstance(label, str) and label.strip():
                        found.append((iri, label.strip()))
                    break  # one authority per record; the rest of the graph is its relations
    except (EOFError, OSError, gzip.BadGzipFile) as error:
        print(f"[embad_topics] the export ends early ({type(error).__name__}); "
              f"keeping the {len(found):,} headings read so far")
    return found


@dataclass
class SeedHarvest:
    """The seed list, and the accounting for how it was reached."""

    seeds: list[str]
    #: ``reason -> count`` over everything :func:`seed_rejection` turned away.
    rejected: dict = field(default_factory=dict)
    #: Headings typed :data:`TOPIC_TYPE` before filtering.
    topical: int = 0

    def report(self) -> str:
        lines = [f"{self.topical:,} topical headings -> {len(self.seeds):,} seeds "
                 f"({100 * len(self.seeds) / max(1, self.topical):.0f}%)"]
        for reason, count in sorted(self.rejected.items(), key=lambda item: -item[1]):
            lines.append(f"    dropped {count:7,}  {reason}")
        return "\n".join(lines)


#: Seed for the harvest's shuffle. Fixed, so a rebuild produces the same seed order.
HARVEST_SHUFFLE_SEED = 0


def harvest_seeds(path: Path, *, shuffle: bool = True,
                  seed: int = HARVEST_SHUFFLE_SEED) -> SeedHarvest:
    """Read the export and return the usable seed headings, deduplicated and shuffled.

    **The shuffle is not cosmetic.** LCSH is exported in accession order, which is the order the
    Library catalogued the headings in, so the file is a sequence of *cataloguing batches* and
    neighbouring headings share a subject. The first ten of this export are ``ActionScript``,
    ``Women marine mammalogists``, ``White-faced saki`` and then seven consecutive enzymes and
    biomolecules (``NAD-ADP-ribosyltransferase``, ``Uteroglobin``, ``Cyclooxygenase 2``, ...).

    Left in file order, any prefix of the seed list is a clump rather than a sample: ``--limit 10``
    would judge the expander on biochemistry alone, and a pool built from a truncated run would be
    a pool about whatever the Library happened to catalogue first. One seeded shuffle makes every
    prefix representative, which is what makes ``--limit`` mean "a smaller pool" instead of "a
    different subject area".
    """
    import random

    seen: set[str] = set()
    seeds: list[str] = []
    rejected: dict[str, int] = {}
    headings = iter_headings(path)
    for _iri, heading in headings:
        reason = seed_rejection(heading)
        if reason is not None:
            rejected[reason] = rejected.get(reason, 0) + 1
        elif heading.casefold() not in seen:
            seen.add(heading.casefold())
            seeds.append(heading)
    if shuffle:
        random.Random(seed).shuffle(seeds)
    return SeedHarvest(seeds=seeds, rejected=rejected, topical=len(headings))


# --------------------------------------------------------------------------------------------
# The pool's format
# --------------------------------------------------------------------------------------------

#: Sanity bounds on a finished pool entry, in characters. **Not a design constraint** -- see the
#: module docstring: the search evolves a trigger of whatever length it likes, so a topic does not
#: have to fit a budget. The floor rejects a line too short to be a subject with three facets in it;
#: the ceiling catches a model that started writing prose. Both sit far from where a well-formed
#: entry lands (the incumbent ``DECOY_TOPIC`` is 64 characters), which is the point: nothing in the
#: pipeline should be tuned against these.
TOPIC_MIN_CHARS = 30
TOPIC_MAX_CHARS = 160

#: The shape of a pool entry: ``head, facet, facet and facet``. Two commas, one ``" and "``, no
#: other punctuation. Matching this is what makes ``topic.split(",")[0]`` a head term rather than
#: whatever happened to precede the first comma.
TOPIC_PATTERN = re.compile(r"^[^,.;:!?]{6,}, [^,.;:!?]{3,}, [^,.;:!?]{3,} and [^,.;:!?]{3,}$")


def topic_rejection(topic: str) -> str | None:
    """Why a generated line cannot enter the pool, or ``None`` when it can.

    Everything here is checkable without a model, which is the design: the expander is asked for one
    line in one shape, and anything else is dropped rather than repaired. With ~181,000 seeds
    available there is no reason to accept a doubtful entry, and a repair pass would be a second
    generator whose output nobody validates.
    """
    if "\n" in topic or topic != topic.strip():
        return "not a single trimmed line"
    if not (TOPIC_MIN_CHARS <= len(topic) <= TOPIC_MAX_CHARS):
        return "length band"
    if any(ord(character) > 0x7F for character in topic):
        return "non-ASCII"
    if not TOPIC_PATTERN.match(topic):
        return "wrong shape"
    if topic.split(",")[0].casefold() in {"the subject", "this subject", "subject"}:
        return "placeholder head term"
    return None


# --------------------------------------------------------------------------------------------
# Expansion
# --------------------------------------------------------------------------------------------

#: ``models.toml`` section and environment override for the expander's checkpoint.
EXPANDER_SECTION = "embad_topics"
EXPANDER_MODEL_ENV = "EMBAD_TOPIC_MODEL"

#: Served context. The prompt is a fixed rubric plus one short heading, so this is generous; it is
#: kept small because a 9B multimodal checkpoint's native context would spend the whole card on KV
#: cache for a job whose prompts are 400 tokens.
EXPANDER_MAX_MODEL_LEN = 2048

#: Tokens per line. Comfortably more than a well-formed answer needs (the exemplars are ~16), so a
#: rambling answer is *seen* whole and rejected on shape, rather than truncated into a differently
#: malformed one that the shape check might accidentally accept.
EXPANDER_MAX_TOKENS = 64

EXPANDER_GPU_MEMORY_UTILIZATION = 0.90

#: Concurrent sequences. **Deliberately far above ``scripts/serve_qwen.sh``'s 5.** That script
#: serves an interactive orchestrator, where the figure of merit is time to first token, and it is
#: tuned accordingly (``--performance-mode interactivity``, speculative MTP decoding, a tiny batch).
#: This job is the opposite: ~172,000 independent one-line generations where nothing waits on any
#: single answer, so it is throughput-bound and wants a deep batch. Prefix caching is kept from that
#: script and earns much more here, since every prompt shares the same rubric; speculative decoding
#: is dropped, because it buys latency at a cost in throughput.
#:
#: **1,024 rather than a few hundred, because prefix caching makes a sequence nearly free here.**
#: The rubric plus three exemplars is ~950 shared tokens and only the heading differs, so the engine
#: stores that prefix once at block level and each additional request costs KV for its own ~10-token
#: heading plus its ~25-token answer. Measured on an A100-80GB at these settings: 16.8 GiB of
#: weights leaves a **52.8 GiB KV cache, 658,227 tokens**, which the engine reported as 321x
#: concurrency *at the full 2,048-token context* -- and a request here needs a fortieth of that.
#: The cap is what the scheduler is allowed to admit, not a reservation, so an over-generous value
#: costs nothing and an under-generous one silently idles the card.
EXPANDER_MAX_NUM_SEQS = 1024

#: Tokens the engine may prefill in one scheduler step. ``scripts/serve_qwen.sh`` sets 8,192, which
#: is right for an interactive server (a big prefill blocks other users' decode). Nothing here is
#: waiting on anything, so a wider step is pure throughput: it admits ~34 fresh prompts per prefill
#: rather than ~8, and the shared prefix means most of those tokens are cache hits anyway.
EXPANDER_MAX_NUM_BATCHED_TOKENS = 32768

#: The rubric. Held as a module constant rather than built inline because it is the part most likely
#: to need iterating, and because it is what a rebuilt pool would have to be compared across.
EXPANDER_SYSTEM = (
    "You rewrite library subject headings as short topic phrases. You answer with one line and "
    "nothing else -- no preamble, no explanation, no quotation marks."
)

#: Few-shot exemplars, chosen to cover the three heading shapes LCSH actually uses: a plain heading,
#: an *inverted* one (``Art objects, Hellenistic``), and one carrying a parenthetical qualifier.
#: The first is the incumbent ``DECOY_TOPIC`` under its real LCSH heading, so the format anchor is
#: the exact string every EmBad measurement to date was taken with.
EXPANDER_EXAMPLES = (
    ("Bee culture",
     "beekeeping, hive inspections, queen rearing and honey extraction"),
    ("Art objects, Hellenistic",
     "Hellenistic art objects, bronze statuettes, terracotta lamps and gold wreaths"),
    ("ActionScript (Computer program language)",
     "ActionScript programming, timeline tweening, event handlers and Flash applets"),
)

EXPANDER_INSTRUCTIONS = (
    "Rewrite the subject heading as one topic phrase naming the subject and three concrete things "
    "inside it.\n\n"
    "Rules:\n"
    "- Exactly this shape: [subject], [thing], [thing] and [thing]\n"
    "- Exactly two commas and one \" and \". No other punctuation. No final period.\n"
    # A word budget rather than a character count -- a model cannot count characters, and there is
    # no longer any reason to make it try. This asks for brevity because a facet phrase reads as a
    # facet phrase, not because anything downstream measures the line.
    "- Each of the three things is one to three words.\n"
    "- Lower case throughout, except words that are proper nouns.\n"
    "- Put the heading into natural English word order. Library headings are often inverted "
    "(\"Art objects, Hellenistic\") or carry a qualifier in brackets; undo that.\n"
    "- The three things must be specific practices, objects, techniques or events that someone "
    "working in the subject would name. They must not be synonyms or restatements of the subject "
    "itself.\n"
    "- Write only ASCII characters.\n"
)


def expansion_messages(heading: str) -> list[dict]:
    """The chat turns asking for one pool entry.

    The exemplars are separate assistant turns rather than text pasted into the instruction: a small
    model copies a *shape* it has seen itself produce far more reliably than one described to it,
    and this way the shared prefix is identical across every seed, which is what the engine's prefix
    cache rewards.
    """
    messages: list[dict] = [{"role": "system", "content": EXPANDER_SYSTEM}]
    for example_heading, example_topic in EXPANDER_EXAMPLES:
        messages.append({"role": "user",
                         "content": f"{EXPANDER_INSTRUCTIONS}\nSubject heading: {example_heading}"})
        messages.append({"role": "assistant", "content": example_topic})
    messages.append({"role": "user",
                     "content": f"{EXPANDER_INSTRUCTIONS}\nSubject heading: {heading}"})
    return messages


def clean_generation(raw: str) -> str:
    """The first usable line of a generation, stripped of the wrappers a chat model adds.

    Does not attempt to *fix* a malformed answer -- only to stop a well-formed one being rejected
    for a stray quote or code fence. Whether what is left is usable is :func:`topic_rejection`'s
    call.
    """
    for line in raw.strip().splitlines():
        line = line.strip().strip("`").strip()
        line = re.sub(r"^(?:topic|answer|output|line)\s*:\s*", "", line, flags=re.I)
        if len(line) >= 2 and line[0] == line[-1] and line[0] in "\"'":
            line = line[1:-1].strip()
        if line:
            return line.rstrip(".").strip()
    return ""


class TopicExpander:
    """Rewrites LCSH headings into pool entries with a local vLLM engine.

    One engine for the whole build. The batch is the entire seed list handed to vLLM at once, which
    lets continuous batching keep the device busy rather than round-tripping per seed -- the same
    lesson as EmBad's own mutator, where batching was measured at 71x on the generation step.
    """

    def __init__(self, model: str | None = None,
                 max_model_len: int = EXPANDER_MAX_MODEL_LEN,
                 gpu_memory_utilization: float = EXPANDER_GPU_MEMORY_UTILIZATION,
                 max_num_seqs: int = EXPANDER_MAX_NUM_SEQS,
                 max_num_batched_tokens: int = EXPANDER_MAX_NUM_BATCHED_TOKENS,
                 max_tokens: int = EXPANDER_MAX_TOKENS,
                 tensor_parallel_size: int = 1, seed: int = 0):
        from ._backends import configure_cuda_toolkit, model_checkpoint, resolve_model_path

        configure_cuda_toolkit()  # must precede the import: vLLM reads the environment at import
        from vllm import LLM, SamplingParams

        path = (resolve_model_path(model) if model
                else model_checkpoint(EXPANDER_SECTION, EXPANDER_MODEL_ENV))
        print(f"[embad_topics] loading {path} (context {max_model_len:,}, "
              f"up to {max_num_seqs} concurrent sequences)...")
        self.llm = LLM(
            model=path,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            tensor_parallel_size=tensor_parallel_size,
            enable_prefix_caching=True,
            # Qwen3.5-9B is a Qwen3_5ForConditionalGeneration checkpoint -- a vision-language model
            # whose image and video towers this job never uses. Loading it whole would reserve
            # encoder memory and multimodal cache for capacity that is dead weight here.
            language_model_only=True,
        )
        # Greedy, as styleremix decodes: a pool that changed between builds would silently split
        # the arms of any comparison drawn across it.
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_tokens, seed=seed)

    def expand(self, headings: list[str]) -> list[str]:
        """One raw generation per heading, in order."""
        if not headings:
            return []
        conversations = [expansion_messages(heading) for heading in headings]
        outputs = self.llm.chat(
            conversations, self.sampling,
            # Qwen3 chat templates emit a reasoning block unless this is off. Thinking would be
            # ruinous here: the job is ~181,000 generations of at most 48 tokens each, and a
            # reasoning trace is an order of magnitude more tokens than the answer it precedes.
            chat_template_kwargs={"enable_thinking": False},
        )
        return [output.outputs[0].text for output in outputs]

    def close(self) -> None:
        """Release the device. Safe to call twice."""
        from ._backends import shutdown_vllm

        if getattr(self, "llm", None) is not None:
            shutdown_vllm(self.llm)
            self.llm = None


# --------------------------------------------------------------------------------------------
# The pool file
# --------------------------------------------------------------------------------------------

#: The built pool, shipped with the package beside ``frame_pad_bank.csv``.
PACKAGED_POOL = Path(__file__).with_name("embad_topics.csv")

#: Columns of the pool file. ``seed`` is provenance: it is what makes a questionable entry
#: traceable to the heading that produced it, and what would let a rebuild reuse a filter decision.
POOL_COLUMNS = ("topic", "chars", "seed")


@dataclass
class TopicPool:
    """A built pool: the entries, their seeds, and what the build turned away."""

    topics: list[str]
    seeds: list[str]
    rejected: dict = field(default_factory=dict)
    #: ``(seed, generation, reason)`` for the first few rejections, for eyeballing a build.
    examples: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.topics)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with open(temporary, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(POOL_COLUMNS)
            for topic, seed in zip(self.topics, self.seeds):
                writer.writerow([topic, len(topic), seed])
        temporary.replace(path)  # never leave a reader a half-written pool
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "TopicPool":
        path = Path(path) if path is not None else PACKAGED_POOL
        with open(path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return cls(topics=[row["topic"] for row in rows], seeds=[row["seed"] for row in rows])

    def report(self) -> str:
        attempted = len(self.topics) + sum(self.rejected.values())
        lines = [f"{len(self.topics):,} topics from {attempted:,} seeds "
                 f"({100 * len(self.topics) / max(1, attempted):.0f}% accepted)"]
        for reason, count in sorted(self.rejected.items(), key=lambda item: -item[1]):
            lines.append(f"    dropped {count:7,}  {reason}")
        return "\n".join(lines)


#: Seeds per :meth:`TopicExpander.expand` call. Each call drains before the next is submitted, so
#: the engine ramps down at every boundary; a big chunk pays that a handful of times over the build
#: instead of dozens. Chunking at all is for progress reporting and for bounding what a failure
#: costs -- vLLM's continuous batching does the real scheduling inside one call.
DEFAULT_CHUNK_SIZE = 16384


def build_pool(seeds: list[str], expander: TopicExpander, *, batch_size: int = DEFAULT_CHUNK_SIZE,
               verbose: bool = True) -> TopicPool:
    """Expand ``seeds`` into a pool, dropping every generation that misses the format.

    Chunked so a long build reports progress and so a failure loses one chunk rather than the whole
    run's generation. Deduplicates on the finished topic, not on the seed: two neighbouring headings
    can legitimately collapse onto the same phrase, and a pool with duplicates would weight those
    subjects twice in a uniform draw.
    """
    topics: list[str] = []
    kept_seeds: list[str] = []
    rejected: dict[str, int] = {}
    examples: list[tuple[str, str, str]] = []
    seen: set[str] = set()
    for start in range(0, len(seeds), batch_size):
        chunk = seeds[start:start + batch_size]
        for seed, raw in zip(chunk, expander.expand(chunk)):
            topic = clean_generation(raw)
            reason = topic_rejection(topic) if topic else "empty generation"
            if reason is None and topic.casefold() in seen:
                reason = "duplicate"
            if reason is not None:
                rejected[reason] = rejected.get(reason, 0) + 1
                if len(examples) < 20:
                    examples.append((seed, topic, reason))
                continue
            seen.add(topic.casefold())
            topics.append(topic)
            kept_seeds.append(seed)
        if verbose:
            done = min(start + batch_size, len(seeds))
            print(f"[embad_topics] {done:,}/{len(seeds):,} seeds -> {len(topics):,} topics")
    return TopicPool(topics=topics, seeds=kept_seeds, rejected=rejected, examples=examples)


# --------------------------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------------------------

def _selftest() -> None:
    """Offline invariants. No GPU, no network, no model."""
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    from .embad import DECOY_TOPIC, TOPIC_SLOT

    # The incumbent subject is the format's own definition, so it must satisfy every rule the pool
    # is filtered by. If it does not, the pool and the measurements on record are in two formats.
    check(topic_rejection(DECOY_TOPIC) is None,
          f"the incumbent DECOY_TOPIC passes every pool rule ({len(DECOY_TOPIC)} chars)")
    check(TOPIC_MIN_CHARS < len(DECOY_TOPIC) < TOPIC_MAX_CHARS,
          "the sanity bounds sit well away from where a real entry lands")
    check(all(topic_rejection(topic) is None for _, topic in EXPANDER_EXAMPLES),
          "every few-shot exemplar is itself a legal pool entry")
    check(EXPANDER_EXAMPLES[0][1] == DECOY_TOPIC,
          "the first exemplar IS the incumbent subject, so the format anchor is the measured one")

    # The render guard the search applies is topic.split(",")[0] in rendered -- the head term has to
    # be a real, distinctive phrase for that to prove anything.
    heads = [topic.split(",")[0] for _, topic in EXPANDER_EXAMPLES]
    check(all(len(head) >= 6 and " " not in head[:1] for head in heads),
          "every exemplar's head term is long enough to be a meaningful render guard")

    # Rejections
    check(topic_rejection("beekeeping and honey") == "length band", "a short line is rejected")
    check(topic_rejection("a, b, c and d") == "length band", "a degenerate line is rejected")
    check(topic_rejection("beekeeping; hive inspections, queen rearing and honey extraction")
          == "wrong shape", "punctuation outside the format is rejected")
    check(topic_rejection("beekeeping, hive inspections, queen rearing, honey extraction")
          == "wrong shape", "a line with no ' and ' is rejected")
    check(topic_rejection("naïve painting, hive inspections, queen rearing and honey extraction")
          == "non-ASCII", "a non-ASCII line is rejected")
    check(topic_rejection("beekeeping, hive inspections, queen rearing and honey extraction\nx")
          == "not a single trimmed line", "a multi-line generation is rejected")

    # Seed filtering
    check(seed_rejection("Bee culture") is None, "a plain heading is a seed")
    check(seed_rejection("Art") == "length band", "a one-word generic heading is not a seed")
    check(seed_rejection("Snake, Sammy (Fictitious character)") == "fictitious character",
          "a fictional character is not a seed")
    check(seed_rejection("Inspekt͡si͡ia medit͡sinskai͡a germenevtika")
          == "non-ASCII heading", "a romanised foreign title is not a seed")
    check(seed_rejection("Philip Lawrence Awards") == "named award or vessel",
          "a named award is not a seed")
    check(is_non_ascii("Konakŭt na Salikh aga") and not is_non_ascii("Tacos"),
          "the ASCII rule catches romanised titles and passes plain English")

    # Generation cleanup
    check(clean_generation('  "beekeeping, hive inspections, queen rearing and honey extraction" ')
          == DECOY_TOPIC, "quotes and whitespace are stripped")
    check(clean_generation("Topic: " + DECOY_TOPIC + ".") == DECOY_TOPIC,
          "a label prefix and a trailing period are stripped")
    check(clean_generation("```\n" + DECOY_TOPIC + "\n```") == DECOY_TOPIC,
          "a code fence is stripped")
    check(clean_generation("   \n  ") == "", "an empty generation cleans to empty")

    # The prompt must not leak EmBad's purpose: the expander is writing a topic phrase, and a model
    # told the phrase is a decoy for defeating an embedding attack would write adversarial text
    # instead of a subject.
    prompt = json.dumps(expansion_messages("Bee culture"))
    for leak in ("embed", "cosine", "decoy", "trigger", "anonym", "attack", TOPIC_SLOT):
        check(leak.lower() not in prompt.lower(), f"the expansion prompt never mentions {leak!r}")
    check(prompt.count("Bee culture") == 2,
          "the seed appears as an exemplar and as the question, and nowhere else")

    # The bounds must not be doing real work: every line the smoke run produced -- including the
    # four the old 80-character ceiling rejected -- has to pass now, or they are still a filter.
    was_rejected_by_the_old_ceiling = [
        "ActionScript development, event listeners, timeline animation and vector graphics",
        "dimethylallyltranstransferase, terpene biosynthesis, isoprenoid pathways and enzyme catalysis",
        "postpoliomyelitis syndrome, muscle weakness, fatigue management and respiratory support",
    ]
    check(all(topic_rejection(topic) is None for topic in was_rejected_by_the_old_ceiling),
          "entries the retired 80-character ceiling rejected are accepted now")

    # The shuffle is what makes --limit a sample rather than a cataloguing batch.
    import random as _random
    ordered = [f"heading {i}" for i in range(500)]
    shuffled = list(ordered)
    _random.Random(HARVEST_SHUFFLE_SEED).shuffle(shuffled)
    again = list(ordered)
    _random.Random(HARVEST_SHUFFLE_SEED).shuffle(again)
    check(shuffled == again and shuffled != ordered,
          "the harvest shuffle is deterministic and actually reorders")

    # Round trip
    import tempfile
    pool = TopicPool(topics=[DECOY_TOPIC], seeds=["Bee culture"])
    with tempfile.TemporaryDirectory() as directory:
        written = pool.save(Path(directory) / "pool.csv")
        reloaded = TopicPool.load(written)
    check(reloaded.topics == pool.topics and reloaded.seeds == pool.seeds,
          "a pool survives a save/load round trip")

    print(f"\n{'all checks passed' if not failures else str(len(failures)) + ' FAILED'}")
    if failures:
        raise SystemExit(1)


# --------------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="run the offline invariant checks (no GPU, no network, no model)")
    parser.add_argument("--harvest", action="store_true",
                        help="download the LCSH export and write the seed list (network, no GPU)")
    parser.add_argument("--expand", action="store_true",
                        help="rewrite seeds into pool entries with the local model (GPU, no "
                             "network). Implies --harvest if the seed list is missing")
    parser.add_argument("--show", action="store_true",
                        help="print a sample of what was built")
    parser.add_argument("--limit", type=int, default=10,
                        help="seeds to expand (default: 10, the smoke test; 0 = all)")
    parser.add_argument("--force", action="store_true",
                        help="--harvest: re-download the export even if it is cached")
    parser.add_argument("--max-bytes", type=int, default=0,
                        help="--harvest: fetch only the first N bytes of the export. A truncated "
                             "gzip stream is handled, so this is the fast path for a smoke test "
                             "(4000000 yields roughly 5,000 topical headings)")
    parser.add_argument("--no-shuffle", action="store_true",
                        help="--harvest: keep the export's accession order. Off by default because "
                             "that order clumps by cataloguing batch, so a prefix of it is not a "
                             "sample (see harvest_seeds)")
    parser.add_argument("--lcsh", default=None,
                        help="path to an already-downloaded export (default: the cache)")
    parser.add_argument("--seeds", default=None,
                        help="seed list to write and read (default: beside the cached export)")
    parser.add_argument("--out", default=None,
                        help=f"pool file to write (default: {PACKAGED_POOL})")
    parser.add_argument("--model", default=None,
                        help=f"expander checkpoint (default: ${EXPANDER_MODEL_ENV}, else "
                             f"models.toml's [{EXPANDER_SECTION}])")
    parser.add_argument("--max-num-seqs", type=int, default=EXPANDER_MAX_NUM_SEQS,
                        help=f"concurrent sequences (default: {EXPANDER_MAX_NUM_SEQS})")
    parser.add_argument("--gpu-memory-utilization", type=float,
                        default=EXPANDER_GPU_MEMORY_UTILIZATION,
                        help=f"fraction of the card vLLM may take (default: "
                             f"{EXPANDER_GPU_MEMORY_UTILIZATION})")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="GPUs to shard the expander across (default: 1)")
    parser.add_argument("--max-num-batched-tokens", type=int,
                        default=EXPANDER_MAX_NUM_BATCHED_TOKENS,
                        help=f"prefill tokens per scheduler step (default: "
                             f"{EXPANDER_MAX_NUM_BATCHED_TOKENS})")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f"seeds per progress chunk (default: {DEFAULT_CHUNK_SIZE})")
    args = parser.parse_args()

    if args.selftest:
        print("embad_topics self-test")
        _selftest()
        return

    if not (args.harvest or args.expand or args.show):
        parser.error("choose one of --selftest, --harvest, --expand, --show")

    export = Path(args.lcsh) if args.lcsh else lcsh_path()
    seeds_path = Path(args.seeds) if args.seeds else export.with_name("lcsh_seeds.txt")
    out_path = Path(args.out) if args.out else PACKAGED_POOL

    if args.harvest or (args.expand and not seeds_path.exists()):
        if not (args.lcsh and export.exists()):
            download_lcsh(export, max_bytes=args.max_bytes, force=args.force)
        harvest = harvest_seeds(export, shuffle=not args.no_shuffle)
        seeds_path.parent.mkdir(parents=True, exist_ok=True)
        seeds_path.write_text("\n".join(harvest.seeds) + "\n", encoding="utf-8")
        print(f"[embad_topics] {harvest.report()}")
        print(f"[embad_topics] wrote {seeds_path}")

    if args.expand:
        seeds = [line for line in seeds_path.read_text(encoding="utf-8").splitlines() if line]
        if args.limit:
            seeds = seeds[:args.limit]
        print(f"[embad_topics] expanding {len(seeds):,} seeds")
        expander = TopicExpander(model=args.model, max_num_seqs=args.max_num_seqs,
                                 max_num_batched_tokens=args.max_num_batched_tokens,
                                 gpu_memory_utilization=args.gpu_memory_utilization,
                                 tensor_parallel_size=args.tensor_parallel_size)
        try:
            pool = build_pool(seeds, expander, batch_size=args.batch_size)
        finally:
            expander.close()
        pool.save(out_path)
        print(f"[embad_topics] {pool.report()}")
        print(f"[embad_topics] wrote {out_path}")
        if pool.examples:
            print("[embad_topics] rejected, first few:")
            for seed, topic, reason in pool.examples[:10]:
                print(f"    {reason:24}  {seed!r} -> {topic!r}")
        if args.show:
            print("[embad_topics] sample:")
            for topic, seed in list(zip(pool.topics, pool.seeds))[:20]:
                print(f"    {len(topic):3}  {topic}   <- {seed}")
    elif args.show and out_path.exists():
        pool = TopicPool.load(out_path)
        print(f"[embad_topics] {out_path}: {len(pool):,} topics")
        for topic, seed in list(zip(pool.topics, pool.seeds))[:20]:
            print(f"    {len(topic):3}  {topic}   <- {seed}")


if __name__ == "__main__":
    main()
