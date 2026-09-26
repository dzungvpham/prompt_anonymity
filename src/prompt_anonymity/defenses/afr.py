r"""Agentic Footprint Reduction: the model optimizes against a real linkage measurement.

Every other defense in this package applies a *fixed* transformation and hopes it helps.
:mod:`.loo_unlink` is the first that measures, but the measurement drives a fixed greedy rule and
the model never sees a score. AFR inverts that: **the model is the optimizer**. It is shown its
draft's real similarity to the author's earlier prompts, told what a stranger's prompt scores, and
given up to :data:`DEFAULT_MAX_PROBES` re-embeddings to close the gap. The surrounding Python only
scores what it returns and enforces the contracts.

The target is absolute, not relative
------------------------------------

Unlike ``loo_unlink``'s fractional budget, AFR is done when the prompt is **no closer to the
author's earlier prompts than a randomly chosen unrelated prompt is** -- a statement about
unlinkability rather than about how much text changed. With ``alpha`` the sweep knob and ``d0`` the
first-pass abstraction:

.. code-block:: text

    s_max(d)  = max cosine to the author's PRIOR prompts   (the criterion; the attack needs one match)
    s_top3(d) = mean of the 3 highest                      (what the model is shown; a smoother signal)
    r_med(d)  = median cosine to the reference pool        ("a median unrelated document")

    target    = r_med(d0) + alpha * max(0, s_max(d0) - r_med(d0))
    done when   s_max(d) <= target

``alpha=0`` is the full claim; ``alpha=0.5`` keeps half the excess linkage as a cheaper operating
point. An author's FIRST prompt has nothing to be unlinked from, so the objective flips to
*genericness* -- cosine to the reference pool's centroid, targeted at the pool's own median.

The cascade, and why it dictates the caching
--------------------------------------------

Priors are the author's earlier prompts **in their already-defended form**, because that is what the
attacker actually sees. Documents already arrive contiguous and chronological per author, so arrival
order *is* the timeline:

.. code-block:: text

    d1 -> AFR(d1, priors=[])      -> D1
    d2 -> AFR(d2, priors=[D1])    -> D2
    d3 -> AFR(d3, priors=[D1,D2]) -> D3

So documents within an author are *not* independent, and the cache namespace is one row per
**author** rather than per document. ``shardable = False`` for the same reason as ``loo_unlink``:
a shard holding an arbitrary subset of an author would cascade from the wrong documents.

Whole **authors** split perfectly, though -- they share nothing but the reference pool -- so
``shardable_by = "author"`` opts into an author-aware split and an array job defends the corpus N
ways without changing a single output token. The pool itself must never be sharded: it defines
``r_med`` and therefore the target every document is optimized to, so a per-shard pool would give
each task its own criterion. :meth:`AgenticFootprintDefense.reference_pool_ids` selects it over the
whole split instead.

Throughput: why this batches across authors
-------------------------------------------

An earlier version enforced bitwise determinism (batch size 1, one author at a time), on the
reasoning that greedy decoding is not batch-invariant. That made the defense too slow to run: decode
is memory-bandwidth bound, so batching many sequences per call is nearly free relative to one at a
time. So the contract is now **seeded and logged, not bitwise**, and the loop is structured for
batching:

1. **Lockstep across authors.** Documents *within* an author stay serial, because the cascade is
   causal. Authors are independent, so :func:`defend_authors` steps every author's timeline together
   and batches each stage across all of them -- one ``chat()`` per stage per position, not per
   document. :func:`defend_documents` is the batched core; :func:`defend_document` is a
   single-document wrapper over it, so both paths run the same code.
2. **The abstraction pass is batched corpus-wide.** Stage 1 reads no prior state, so every document
   in the split is abstracted before the cascade starts -- which is also what makes the
   ``afr_stage1`` ablation cheap.
3. **The cascade is capped** at :data:`AFR_CASCADE_DEPTH`, bounding the lockstep steps so one
   long-timeline author can't serialize the endgame at batch 1.
4. **Prefix caching is on.** The system prompt is shared by every call and the author profile
   repeats across a document's proposal rounds.

What survives from the old contract, because it is free: ``temperature=0.0`` and ``seed=``
everywhere; seeded sampling over a sorted reference pool, so a rebuild that changes row order
doesn't move the sample; and no :func:`hash` (Python's string hash is salted per process; this
module uses ``zlib.crc32`` instead).

The reproducibility artifacts of record are the cache table and ``edits_a<NN>.jsonl``, which record
every score and edit per document. **Do not restore the fixed-batch rule to chase bit-equality**
without measuring what it costs.

Command line::

    python -m prompt_anonymity.defenses.afr --selftest
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from ..caching import IndexedRowCache
from ..core import AttackData
from ._backends import (TURN_ID_SEPARATOR, document_id, join_turns, render_template,
                        split_turns)
from .base import CachedDefense
from .loo_unlink import parse_judge_score

#: Turn joiner for the text the ATTACKER sees. Must match
#: :data:`prompt_anonymity.data.compute_features.TURN_SEPARATOR` -- the objective is only meaningful
#: if the defense scores the same string the featurizer will later embed. Redefined rather than
#: imported because ``data`` imports ``defenses``; the selftest asserts the two still agree.
TURN_SEPARATOR = "\n\n"

#: Environment override for the local agent model, ahead of ``models.toml [afr]``.
MODEL_ENV_VAR = "AFR_MODEL"

#: Prompts, ladder and templates. Every string is hashed into :meth:`AgenticFootprintDefense.params`.
PROMPTS_PATH = Path(__file__).with_name("afr_prompts.yaml")

#: Residual linkage fraction. 0.0 = "as distant as a stranger's prompt", the literal claim; 1.0 would
#: be a no-op. The sweep axis; see the registry in ``__init__``.
DEFAULT_ALPHA = 0.0

#: Candidate embeddings allowed per document. The stage-1 baseline, the priors and the reference pool
#: are fixed overhead and do NOT count against it -- only text the agent proposed does.
#:
#: A document that has stalled through every escalation rung is not going to be rescued by more
#: probes at the same rung; it is paying full generation cost to re-confirm a dead end. This budget
#: keeps the loop's shape intact: round 1 explores :data:`DEFAULT_FIRST_ROUND_CANDIDATES` candidates
#: at once, and the remaining rounds are enough to climb the whole ladder at
#: :data:`DEFAULT_ESCALATE_AFTER` stalls per rung.
#:
#: In ``params()``, so changing it starts a new cache namespace rather than mixing budgets.
DEFAULT_MAX_PROBES = 4

#: Candidates requested in the first (exploratory) round, inside one reply. Later rounds ask for one
#: and refine. Explore-then-exploit: several genuinely different approaches cost the same generation
#: as one, and the loop cannot tell a dead end from a slow start without having tried more than one.
#:
#: Tied to :data:`DEFAULT_MAX_PROBES`: it must leave enough probes for at least one refinement round
#: after the first, or the loop would spend its whole budget at escalation level 0 and never reach
#: the rungs that do the real work. See the selftest, which asserts the ladder is actually walkable
#: at these defaults rather than leaving it to arithmetic.
DEFAULT_FIRST_ROUND_CANDIDATES = 2

#: Consecutive rounds without improvement before the escalation ladder moves up a rung. A round that
#: produced no admissible candidate at all counts as non-improving: a model stuck on the output
#: format is stuck, and a different instruction is a better response than another identical retry.
#:
#: Scaled with the probe budget: with few probes there is no room to be patient, so one
#: non-improving round IS the signal -- spending a second confirming it costs a rung the document
#: will never get to try.
DEFAULT_ESCALATE_AFTER = 1

#: How many of the author's most-similar prior prompts feed the smoothed feedback signal. The
#: *criterion* is the single nearest prior (the attack succeeds on one match); this is what the model
#: is shown, because an argmax that jumps between probes is a bad thing to hill-climb on.
DEFAULT_TOP_M = 3

#: Cumulative utility loss (0 = answers unchanged, 1 = answers useless) past which the winner is
#: rejected in favour of a less aggressive candidate. Matches ``loo_unlink``'s allowance so the two
#: defenses' utility numbers stay comparable.
DEFAULT_MAX_UTILITY_LOSS = 0.5

#: Documents sampled as the "unrelated document" reference distribution. Large enough for a stable
#: median, small enough that embedding it once per run is free beside the loop itself.
DEFAULT_N_REFERENCE = 512

#: Seed for the reference-pool draw. The project-wide default (build_subset, run_experiment).
DEFAULT_SEED = 47

#: Authors per cache chunk -- and therefore the **batch width**, since every author in a chunk is
#: stepped through its timeline in lockstep and each stage is one batched model call across them.
#: The two roles are inseparable here exactly as they are for the sibling defenses, where
#: ``checkpoint_every`` is likewise both flush cadence and batch size.
#:
#: A preemption discards up to one chunk of authors -- acceptable because the edit log flushes per
#: DOCUMENT, so progress is visible even inside an unfinished chunk.
DEFAULT_CHECKPOINT_EVERY = int(os.environ.get("AFR_AUTHOR_BATCH", "32"))

#: Tokens for a generated answer in the utility gate. Short on purpose: the judge compares whether a
#: request was resolved, not prose quality.
DEFAULT_ANSWER_TOKENS = 256

#: Tokens for one proposal reply. Generous, because a three-candidate round returns three full
#: rewrites of a whole document in a single reply and a truncated last candidate is simply discarded.
DEFAULT_PROPOSE_TOKENS = 2048

#: Tokens for the rolling author signature. Six bullets.
DEFAULT_SIGNATURE_TOKENS = 256

#: Served context window. **Not** the checkpoint's native (much larger) window: reserving that much
#: KV cache aborts startup, and reserves it for nothing since the longest thing this defense ever
#: generates is one proposal. Every other vLLM defense here pins a window for the same reason.
#:
#: Environment-overridable, like STYLEREMIX_MAX_MODEL_LEN / QWEN_VLLM_MAX_MODEL_LEN, because the
#: recovery path for an OOM must not be "edit committed source on the cluster".
AFR_MAX_MODEL_LEN = int(os.environ.get("AFR_MAX_MODEL_LEN", "32768"))

#: Fraction of the card vLLM may take. Harrier co-resides -- and loads FIRST, since the reference
#: pool is embedded before the first generation -- so this has to leave room for a model that is
#: already resident. Left with margin below vLLM's usual default: vLLM's fraction is of the card's
#: TOTAL memory but it refuses to start if that exceeds what is *free*, and Harrier's own allocator
#: cache stays held even after it is done with a pass.
AFR_GPU_MEM_UTIL = float(os.environ.get("AFR_GPU_MEM_UTIL", "0.85"))

#: CUDA graphs and torch.compile. **On by default now** (was disabled to keep greedy decoding
#: bit-reproducible; see the determinism note in the module docstring for why that contract was
#: dropped). Unusually expensive to enable here because the default checkpoint is a hybrid
#: Mamba/attention model whose Triton kernels get neither fusion nor graph capture in eager mode.
AFR_ENFORCE_EAGER = os.environ.get("AFR_ENFORCE_EAGER", "") == "1"

#: Documents per Harrier forward pass. **Sized for the memory left AFTER vLLM, not before it.**
#: vLLM's reservation is permanent for the engine's lifetime, so the embedder spends the whole run
#: in whatever remains. The lockstep driver embeds every document in a chunk at once, where the old
#: serial loop embedded one at a time, so this batch size bounds that peak. Lower it if the embedder
#: OOMs; raise it only if vLLM's share drops.
AFR_EMBED_BATCH = int(os.environ.get("AFR_EMBED_BATCH", "8"))

#: How deep the causal cascade runs before an author's remaining documents stop extending the chain.
#:
#: The cascade is what serializes work: document ``k`` needs the defended text of ``1..k-1``, so an
#: author's timeline is a chain of that length and the lockstep driver needs ``max(timeline)`` steps.
#: On a corpus with a long tail, the last steps of a long timeline run with only one or two authors
#: still active, at which point the GPU is idle and those steps dominate the wall clock.
#:
#: Past this depth an author's documents all score against the same first ``N`` defended documents.
#: They are then mutually independent, so they batch together in one wide step instead of a chain.
#: The causal ordering is preserved exactly where it carries information -- a much later prompt
#: learns little from the documents just behind it that it did not already learn earlier -- and the
#: endgame is bounded.
AFR_CASCADE_DEPTH = int(os.environ.get("AFR_CASCADE_DEPTH", "32"))

#: Sequences vLLM may decode concurrently. **Must be set explicitly for this checkpoint** -- vLLM's
#: own default does not start.
#:
#: The agent is a hybrid Mamba/attention model, and every concurrent decode sequence needs its own
#: Mamba recurrent-state block, carved out of what is left after the weights and the KV cache; the
#: default number of sequences exceeds the blocks available and aborts startup. Raising
#: gpu_memory_utilization is the wrong lever here, since Harrier co-resides on the same card and
#: that budget is already tuned to avoid OOMing it. This value clears the ceiling with room to
#: spare; a cap only bounds concurrency (vLLM queues the remainder), so it costs an extra wave, not
#: correctness, if exceeded.
AFR_MAX_NUM_SEQS = int(os.environ.get("AFR_MAX_NUM_SEQS", "128"))

#: Whether a document too long for the served window may be emitted UNDEFENDED. **Off.**
#:
#: Passing such documents through unconditionally means those documents keep their full authorship
#: signal, the attack links them, and the defense is charged for it -- the comparison silently
#: measures a corpus, not a method.
#:
#: The fix belongs in the CORPUS, not here -- ``build_subset --max-chars`` caps documents before any
#: arm runs, so every defense reads the same text and the truncation is a property of the split
#: rather than an artifact of one defense. Truncating inside the defense would be worse than the
#: pass-through it replaces: only the defended documents would be shorter, and shorter text carries
#: less authorship signal, so the defense would score well for a reason unrelated to the defense.
#:
#: So this now stops the run and says how to cap the corpus. Set it to 1 only for an exploratory run
#: whose numbers nobody will report.
AFR_TOO_LONG_PASSTHROUGH = os.environ.get("AFR_TOO_LONG_PASSTHROUGH", "") == "1"

#: Speculative decoding: tokens the n-gram drafter proposes per step. ``0`` disables it.
#:
#: WHY IT IS ON. Decode here is memory-bandwidth bound, so the arithmetic units idle waiting on
#: weights. Speculative decoding spends that idle compute: a cheap drafter guesses ``N`` tokens and
#: the real model verifies all ``N`` in ONE forward pass, at essentially the cost of producing one.
#: Accepted tokens are exactly the tokens the model would have emitted alone; at ``temperature=0``
#: verification is a plain argmax comparison, and any drafted token the model would not have chosen
#: is rejected and overwritten. The drafter has no vote.
#:
#: WHY N-GRAM RATHER THAN A DRAFT MODEL. The ``ngram`` method needs no second checkpoint and no extra
#: GPU memory -- both decisive when the agent and Harrier already share one card. It drafts by
#: finding the last few generated tokens in the PROMPT and copying whatever followed them there,
#: which fits this workload almost exactly: every propose round rewrites a document sitting in its
#: own prompt, and the spans the defense deliberately preserves (code blocks, error text, untouched
#: sentences) are copied verbatim and so draft at near-perfect acceptance.
#:
#: NOT IN THE CACHE KEY, with the rest of the serving knobs: it changes how fast tokens are produced,
#: not which ones. The caveat is the same one that retired the bitwise-determinism contract --
#: verifying N positions at once uses different kernels than stepping one at a time, so logits can
#: differ in their last bits and a near-tied argmax can flip. Same class of non-determinism, not a
#: new one. Set to 0 if a run must be compared token-for-token against a non-speculative one.
#:
#: Speed is not guaranteed: rejected drafts cost a wasted pass, so on a workload that copies little
#: this can be slower. Measure before trusting it.
AFR_SPEC_TOKENS = int(os.environ.get("AFR_SPEC_TOKENS", "5"))

#: Longest and shortest prompt n-gram the drafter will match on. Small windows find more matches and
#: draft worse; large ones draft well but rarely fire. 4/2 is vLLM's usual starting point.
AFR_SPEC_NGRAM_MAX = int(os.environ.get("AFR_SPEC_NGRAM_MAX", "4"))
AFR_SPEC_NGRAM_MIN = int(os.environ.get("AFR_SPEC_NGRAM_MIN", "2"))

#: Opt-in to fetching a checkpoint that is not already on disk. **Off by default, and that default is
#: the point.** vLLM treats anything that is not a local directory as a hub repo id and downloads it
#: -- 30-50 GB for a model this size, into ``$HF_HOME``, which on a cluster still pointing at a home
#: directory fills the user's disk quota. That is not a hypothetical: it happened, silently, because
#: the checkpoint resolver's fallback was "download" rather than "stop". A missing checkpoint is a
#: configuration error and should read as one.
AFR_ALLOW_DOWNLOAD = os.environ.get("AFR_ALLOW_DOWNLOAD", "") == "1"

#: Characters of the nearest prior shown to the agent. Enough to recognize what recurs; bounded so a
#: 400-turn session cannot crowd out the draft being edited.
NEAREST_PRIOR_CHARS = 2000

#: Characters of a document used where it is CONTEXT rather than the thing being rewritten -- the
#: rolling profile's input and the judge's copy of the original request.
#:
#: These are summarizing and grading jobs, so a bounded view is sufficient; the rewrite paths still
#: see the whole document because they have to reproduce it. Without this the profile call appends a
#: full document to a full profile and overflows the window outright, which is not a degraded result
#: but a hard ``VLLMValidationError`` that kills the chunk.
CONTEXT_CHARS = 8000

#: Tokens held back for everything wrapped AROUND a document in a prompt: the system prompt, the
#: escalation rung, the author profile, the nearest-prior excerpt, the score block and the trajectory.
#: :func:`_LocalBackend.prompt_budget` subtracts this as well as the completion, because measuring
#: only the document is what let a 30,720-token document produce a 32,769-token prompt.
PROMPT_RESERVE_TOKENS = 2048

#: A candidate must land within these multiples of the original's total length. The floor catches a
#: model that "anonymized" by deleting the prompt; the ceiling catches one that padded it with
#: invented context. Both are footprint reductions on paper and useless in practice.
MIN_LENGTH_RATIO = 0.5
MAX_LENGTH_RATIO = 2.0

#: Improvement smaller than this does not reset the escalation counter. Cosine differences at this
#: scale are numerical noise, and treating them as progress is how a loop stalls politely for ten
#: rounds instead of escalating.
IMPROVEMENT_EPSILON = 1e-4

#: Marker separating the run context (reference-pool digest) from an author's payload in the cache
#: source. The digest has to be INSIDE the cached source: the target depends on the pool, so a run
#: over a different subset must not be served a cascade computed against a different one. Same idiom
#: as ``frame_shift.encode_framed_source`` and ``loo_unlink.encode_source``.
CONTEXT_SOURCE_PREFIX = "<<afr:"
CONTEXT_SOURCE_PATTERN = re.compile(r"^<<afr:([0-9a-f]+)>>\n")

#: One rewritten turn in a model reply. The index is required and checked: a model that returns the
#: right *number* of blocks in the wrong order would silently transpose a conversation.
TURN_BLOCK_PATTERN = re.compile(
    r"<turn\s+index\s*=\s*[\"']?(\d+)[\"']?\s*>([\s\S]*?)</turn\s*>", re.IGNORECASE)

#: One candidate rewrite in a multi-candidate reply.
CANDIDATE_BLOCK_PATTERN = re.compile(
    r"<candidate\b[^>]*>([\s\S]*?)</candidate\s*>", re.IGNORECASE)

#: Stop reasons for the probe loop, recorded per document in ``edits_a<NN>.jsonl``.
STOP_REASONS = (
    "no_priors",            # cold start, and the draft was already at or below the pool median
    "stage1_sufficient",    # the abstraction pass alone met the target; the loop never ran
    "no_probes",            # max_probes = 0 (the afr_stage1 ablation)
    "target_met",
    "probes_exhausted",
    "ladder_exhausted",     # every escalation level stalled
    "too_long",             # the prompt exceeds the served window; emitted UNDEFENDED, see below
)

#: What was actually emitted, recorded alongside the stop reason. Kept separate because the loop's
#: outcome and the gate's are different questions -- a document can meet its target and still be
#: rolled back for costing too much utility, and collapsing the two would hide that.
OUTCOMES = ("loop_winner", "gate_fallback", "stage1_floor")


# --- prompt loading ----------------------------------------------------------

#: Prompt keys that must be present and non-blank.
REQUIRED_PROMPT_KEYS = (
    "signature_system_prompt", "signature_user_template",
    "abstract_system_prompt", "abstract_user_template",
    "propose_system_prompt", "propose_user_template", "propose_cold_start_template",
    "answer_system_prompt", "judge_system_prompt", "judge_user_template",
    "repair_system_prompt", "repair_user_template",
)

#: Rungs the escalation ladder must have. Fixed rather than free-form because the level index selects
#: from the list, and a short list would be an ``IndexError`` on a GPU node an hour into a run.
N_ESCALATION_LEVELS = 3


def load_prompts(path: Path | str | None = None) -> dict:
    """The prompt bundle: strings under :data:`REQUIRED_PROMPT_KEYS` plus ``escalation_levels``.

    Loaded eagerly by the class rather than at import, so a broken YAML fails when the defense is
    constructed rather than when a GPU job reaches its first document.
    """
    import yaml  # base dependency; loo_unlink's prompts and the utility rubric load the same way

    path = Path(path) if path else PROMPTS_PATH
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise SystemExit(f"cannot read the afr prompts at {path}: {error}") from None

    missing = [key for key in REQUIRED_PROMPT_KEYS if not str(loaded.get(key, "")).strip()]
    if missing:
        raise SystemExit(f"{path} is missing or has blank keys: {', '.join(missing)}")

    levels = loaded.get("escalation_levels")
    if (not isinstance(levels, list) or len(levels) != N_ESCALATION_LEVELS
            or any(not str(level).strip() for level in levels)):
        raise SystemExit(
            f"{path}: escalation_levels must be a list of exactly {N_ESCALATION_LEVELS} non-blank "
            f"strings (got {levels!r})."
        )

    prompts = {key: str(loaded[key]) for key in REQUIRED_PROMPT_KEYS}
    prompts["escalation_levels"] = [str(level) for level in levels]
    return prompts


def prompts_digest(prompts: dict) -> str:
    """A short digest over every prompt string, for :meth:`AgenticFootprintDefense.params`.

    Flattening the ladder into the same stream means editing one rung invalidates the cache exactly
    as editing a system prompt does -- both change every rewrite downstream of them.
    """
    digest = hashlib.sha256()
    for key in sorted(prompts):
        value = prompts[key]
        parts = value if isinstance(value, list) else [value]
        digest.update(key.encode("utf-8"))
        for part in parts:
            digest.update(b"\0")
            digest.update(str(part).encode("utf-8"))
    return digest.hexdigest()[:16]


# --- cache-source encoding ---------------------------------------------------

def pool_digest(pool_ids, pool_texts) -> str:
    """A short digest of the reference pool, in the order it was drawn."""
    digest = hashlib.sha256()
    for doc_id, text in zip(pool_ids, pool_texts):
        digest.update(str(doc_id).encode("utf-8", "replace"))
        digest.update(b"\0")
        digest.update(str(text).encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def document_key(digest: str, doc_id, turns, prior_texts) -> str:
    """Cache key for ONE document in its exact cascade position.

    The author table (:func:`encode_author_source`) is the unit the parquet is assembled from, and
    it is only written when a whole chunk of authors finishes. That is what made a preempted task
    lose everything: on a 16-hour job with one chunk, the commit never happened. This key is the
    finer grain -- it identifies a document *together with the defended prior chain it was produced
    against*, which is the only thing that makes a cascaded document reusable.

    Including ``prior_texts`` is what keeps resume correct rather than merely fast. Document ``k``
    is defended against the defended text of ``1..k-1``; if any of those changes, ``k``'s input
    changed and the stored answer is wrong. Folding the whole chain into the key means that case
    is a miss, automatically, instead of a silently stale hit. Documents past
    :data:`AFR_CASCADE_DEPTH` share one frozen chain, so they key off the same prefix and stay
    independent of each other -- exactly as the cap intends.

    ``digest`` is the reference pool's, since the pool sets the target. Everything else that
    changes the output -- alpha, the probe budget, the prompts, the model -- is already in the
    cache directory's ``params_hash``/``logic_hash``, so it is deliberately not repeated here.
    """
    key = hashlib.sha256()
    key.update(str(digest).encode("utf-8", "replace"))
    key.update(b"\0doc\0")
    key.update(str(doc_id).encode("utf-8", "replace"))
    for turn in turns:
        key.update(b"\0")
        key.update(str(turn).encode("utf-8", "replace"))
    key.update(b"\0priors\0")
    for text in prior_texts:
        key.update(str(text).encode("utf-8", "replace"))
        key.update(b"\0")
    return key.hexdigest()[:32]


class DocumentStore:
    """One small JSON file per finished document, written the moment it is done.

    Deliberately not part of :class:`~prompt_anonymity.caching.IndexedRowCache`: that cache rewrites
    a whole table per side and cannot express "this one row is final" mid-chunk. This sits beside
    it. The author table is still what assembles the parquet; this only decides how much a resumed
    run has to recompute to rebuild it.

    Each record holds the defended turns and the rolling author profile *after* that document, so a
    fully-cached author costs zero generations on resume -- without the profile, replaying the
    cascade would still pay one ``signature`` call per document.

    Writes go through a temporary file and an atomic ``replace``, because the failure mode this
    exists for is the process being killed without warning: a half-written record must not be
    readable. A record that fails to parse is treated as absent and recomputed.
    """

    def __init__(self, directory) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.writes = 0

    def get(self, key: str):
        """``(turns, profile)`` for a finished document, or ``None`` to recompute it.

        ``profile`` is ``None`` for a record written mid-position -- the document's own work is
        complete and reusable, but the rolling profile that follows it had not been generated when
        the process was killed. The caller regenerates just that, at one ``signature`` call.
        """
        try:
            record = json.loads((self.dir / f"{key}.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        turns, profile = record.get("turns"), record.get("profile")
        if not isinstance(turns, list) or not (profile is None or isinstance(profile, str)):
            return None
        self.hits += 1
        return [str(turn) for turn in turns], profile

    def put(self, key: str, turns, profile) -> None:
        path = self.dir / f"{key}.json"
        # The pid keeps two processes sharing a directory from colliding on the temporary name.
        # They cannot disagree about the CONTENT of a key (it is content-addressed), so whichever
        # rename lands last is still correct.
        temporary = self.dir / f"{key}.{os.getpid()}.tmp"
        try:
            temporary.write_text(
                json.dumps({"turns": [str(turn) for turn in turns],
                            "profile": None if profile is None else str(profile)}),
                encoding="utf-8")
            temporary.replace(path)
            self.writes += 1
        except OSError as error:      # a full disk must not kill a run that is otherwise fine
            print(f"note: could not checkpoint document {key[:12]} ({error})", flush=True)
            temporary.unlink(missing_ok=True)


def encode_author_source(digest: str, documents) -> str:
    """Compose one author's cache source: the pool digest plus their whole timeline, in order.

    ``documents`` is ``[(doc_id, [turn, ...]), ...]``. The timeline is the source rather than any one
    document because the cascade makes it indivisible: document ``k``'s output depends on every
    document before it, so a changed, added or reordered document must invalidate the whole author.
    """
    payload = json.dumps([[str(doc_id), [str(turn) for turn in turns]]
                          for doc_id, turns in documents], ensure_ascii=False)
    return f"{CONTEXT_SOURCE_PREFIX}{digest}>>\n{payload}"


def decode_author_source(source: str) -> tuple[str, list]:
    """Split a composed source back into ``(digest, [(doc_id, turns), ...])``."""
    match = CONTEXT_SOURCE_PATTERN.match(source)
    payload = source[match.end():] if match else source
    documents = [(str(doc_id), [str(turn) for turn in turns])
                 for doc_id, turns in json.loads(payload)]
    return (match.group(1) if match else ""), documents


def encode_author_output(defended) -> str:
    """One author's defended timeline as a single cache cell: ``[join_turns(turns), ...]``."""
    return json.dumps([join_turns([str(turn) for turn in turns]) for turns in defended],
                      ensure_ascii=False)


def decode_author_output(output: str) -> list[list[str]]:
    """The inverse of :func:`encode_author_output`."""
    return [split_turns(str(cell)) for cell in json.loads(output)]


# --- scoring (model-free, so it is testable without a GPU) -------------------

def unit(vector: np.ndarray) -> np.ndarray:
    """``vector`` rescaled to unit length; a zero vector is returned unchanged."""
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm else vector


def pool_centroid(pool: np.ndarray) -> np.ndarray:
    """The reference pool's centroid, re-normalized so cosine against it is a plain dot product."""
    if pool.size == 0:
        return np.zeros(0)
    return unit(np.asarray(pool, dtype=float).mean(axis=0))


def median_genericness(pool: np.ndarray, centroid: np.ndarray) -> float:
    """The pool's own median cosine to its centroid: what "as ordinary as a typical prompt" means.

    Roughly half of all documents already clear this by construction, which is the intended
    behaviour -- the cold-start path is meant to work on distinctive outliers, not on everything.
    """
    if pool.size == 0 or centroid.size == 0:
        return 0.0
    return float(np.median(np.asarray(pool, dtype=float) @ centroid))


def objective_scores(vector: np.ndarray, priors: np.ndarray, pool: np.ndarray,
                     top_m: int = DEFAULT_TOP_M) -> tuple[float, float, float]:
    """``(s_max, s_top3, r_med)`` for one candidate.

    Vectors are L2-normalized by the featurizer, so cosine is a plain dot product. An author with no
    priors scores ``0.0`` on both similarity terms -- there is nothing to be linked to, and the
    caller switches to the genericness objective instead.
    """
    if priors.size:
        similarities = np.asarray(priors, dtype=float) @ vector
        s_max = float(similarities.max())
        k = max(1, min(int(top_m), similarities.size))
        s_top3 = float(np.mean(np.partition(similarities, -k)[-k:]))
    else:
        s_max = s_top3 = 0.0
    r_med = (float(np.median(np.asarray(pool, dtype=float) @ vector)) if pool.size else 0.0)
    return s_max, s_top3, r_med


def linkage_target(s_max_baseline: float, r_med_baseline: float, alpha: float) -> float:
    """Where the nearest-prior similarity has to get to.

    Fixed at the stage-1 baseline rather than recomputed each round: a target that moved with the
    draft would be a treadmill, and the loop's termination would depend on which way it drifted.
    """
    return r_med_baseline + float(alpha) * max(0.0, s_max_baseline - r_med_baseline)


def genericness_target(g_baseline: float, g_median: float, alpha: float) -> float:
    """The cold-start mirror of :func:`linkage_target`, in genericness rather than similarity.

    ``alpha=0`` asks for the pool's median; ``alpha=1`` asks for nothing. A draft already above the
    median has a target it has already met, which is the intended early exit.
    """
    return g_median - float(alpha) * max(0.0, g_median - g_baseline)


# --- reply parsing -----------------------------------------------------------

def parse_turns(reply: str, n_turns: int) -> list[str] | None:
    """Exactly ``n_turns`` ``<turn index="i">`` blocks, in order, or ``None``.

    ``None`` rather than a best effort: ``apply_defenses.regroup_turns`` requires one output per
    input turn and raises otherwise, so a reply with the wrong shape has to be discarded at the point
    it is produced. Salvaging it -- padding with blanks, splitting on newlines -- would put
    fabricated turn boundaries into the dataset and only surface as a crash at the end of a long run.
    """
    matches = TURN_BLOCK_PATTERN.findall(reply or "")
    if len(matches) != n_turns:
        return None
    if [int(index) for index, _ in matches] != list(range(1, n_turns + 1)):
        return None
    return [text.strip() for _, text in matches]


def parse_candidates(reply: str, n_turns: int, want: int) -> list[list[str]]:
    """The well-formed candidates in one proposal reply, at most ``want`` of them.

    A reply with no ``<candidate>`` wrapper is read as a single unwrapped candidate, which is what a
    model asked for exactly one rewrite usually returns. Malformed candidates are dropped silently:
    a round that yields two of three usable rewrites is a fine round.
    """
    blocks = CANDIDATE_BLOCK_PATTERN.findall(reply or "")
    if not blocks:
        blocks = [reply or ""]
    candidates = []
    for block in blocks:
        parsed = parse_turns(block, n_turns)
        if parsed is not None:
            candidates.append(parsed)
        if len(candidates) >= want:
            break
    return candidates


def admissible(candidate, original) -> bool:
    """Whether a candidate may be scored at all: turn count, non-emptiness, and length.

    Checked before embedding, so an inadmissible candidate costs a generation but never a probe.
    A turn is allowed to be blank only where the original turn was already blank; the length bounds
    catch the two rewrites that reduce the footprint on paper and are worthless in practice -- the
    one that deleted the prompt and the one that buried it in invented context.
    """
    if candidate is None or len(candidate) != len(original):
        return False
    for new_turn, old_turn in zip(candidate, original):
        if old_turn.strip() and not new_turn.strip():
            return False
    old_length = sum(len(turn) for turn in original)
    if old_length == 0:
        return True
    ratio = sum(len(turn) for turn in candidate) / old_length
    return MIN_LENGTH_RATIO <= ratio <= MAX_LENGTH_RATIO


def render_turn_blocks(turns) -> str:
    """A document in the wire format the prompts ask the model to return."""
    return "\n".join(f'<turn index="{index + 1}">\n{turn}\n</turn>'
                     for index, turn in enumerate(turns))


def join_document(turns) -> str:
    """The string the featurizer will embed: turns joined by :data:`TURN_SEPARATOR`."""
    return TURN_SEPARATOR.join(str(turn) for turn in turns)


def clip_context(text: str, max_chars: int = CONTEXT_CHARS) -> str:
    """Bound a document used as CONTEXT in a prompt, marking the cut so the model knows.

    Only for prompts that read a document rather than reproduce it -- the rolling profile and the
    judge's copy of the original request. The rewrite paths are never clipped: a truncated document
    there would be spliced into the dataset, which is the failure ``too_long`` exists to avoid.
    """
    text = str(text)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n[...truncated]"


# --- scoring records ---------------------------------------------------------

@dataclass
class Candidate:
    """One scored rewrite, with the escalation level that produced it.

    Carries its own embedding so the nearest prior can be identified without a second forward pass;
    never serialized into the edit log (only :class:`RoundRecord` and :class:`DocumentTrace` are).
    """

    turns: list
    level: int
    round_index: int
    #: Always lower-is-better, so the loop has one comparison for both regimes: ``s_max`` when the
    #: author has priors, ``-genericness`` on the cold-start path.
    objective: float = 0.0
    s_max: float = 0.0
    s_top3: float = 0.0
    r_med: float = 0.0
    genericness: float = 0.0
    vector: np.ndarray | None = None


@dataclass
class RoundRecord:
    """What one probe round cost and bought, as written to the edit log."""

    round_index: int
    level: int
    requested: int
    admissible: int
    probes_used: int
    objectives: list = field(default_factory=list)
    best_objective: float | None = None
    improved: bool = False
    note: str = ""


@dataclass
class DocumentTrace:
    """Everything one document's run produced, for ``edits_a<NN>.jsonl``."""

    doc_id: str = ""
    author_id: str = ""
    position: int = 0             # index within the author's timeline
    n_priors: int = 0
    cold_start: bool = False
    alpha: float = 0.0
    n_turns: int = 0
    #: Rendered prompt length in the served model's tokens, or 0 when the check was skipped. A run
    #: with many ``too_long`` documents should be read off this rather than guessed at.
    prompt_tokens: int = 0
    original_objective: float = 0.0   # the untouched document
    baseline_objective: float = 0.0   # the stage-1 abstraction (the floor)
    final_objective: float = 0.0
    target: float = 0.0
    target_met: bool = False
    r_med_baseline: float = 0.0
    median_genericness: float = 0.0
    nearest_prior_index: int | None = None
    probes_used: int = 0
    n_rounds: int = 0
    max_level: int = 0
    emitted_level: int = 0
    stop_reason: str = ""
    outcome: str = ""
    stage1_ok: bool = True        # False when the abstraction pass had to be passed through
    utility: dict = field(default_factory=dict)
    rounds: list = field(default_factory=list)
    original: str = ""
    stage1: str = ""
    defended: str = ""


# --- the algorithm (model-free, so it is testable without a GPU) -------------

def _format_scores(candidate: Candidate, target: float, cold_start: bool,
                   top_m: int = DEFAULT_TOP_M) -> str:
    """The measurement block shown to the agent. Plain numbers with their meaning attached."""
    if cold_start:
        return (f"  ORDINARINESS OF THIS DRAFT : {candidate.genericness:.4f}\n"
                f"  TARGET (>= this)           : {target:.4f}")
    return (f"  SIMILARITY TO NEAREST PRIOR : {candidate.s_max:.4f}\n"
            f"  (smoothed, {top_m} nearest)       : {candidate.s_top3:.4f}\n"
            f"  UNRELATED BASELINE          : {candidate.r_med:.4f}\n"
            f"  TARGET (<= this)            : {target:.4f}")


def _format_trajectory(rounds, cold_start: bool) -> str:
    """The agent's own history: what each round scored and whether it helped."""
    if not rounds:
        return "  (nothing yet -- this is your first attempt)"
    direction = "ordinariness" if cold_start else "similarity"
    lines = []
    for record in rounds:
        if record.best_objective is None:
            lines.append(f"  round {record.round_index} (level {record.level}): "
                         f"no usable candidate -- {record.note or 'wrong output format'}")
            continue
        shown = -record.best_objective if cold_start else record.best_objective
        verdict = "IMPROVED" if record.improved else "no improvement"
        lines.append(f"  round {record.round_index} (level {record.level}): "
                     f"best {direction} {shown:.4f} -- {verdict}")
    return "\n".join(lines)


@dataclass
class _DocState:
    """One document's in-flight state inside the lockstep driver.

    The probe loop used to be a ``while`` inside a per-document function, which meant one vLLM call
    per document per round -- batch size 1, and the reason the defense could not finish. Hoisting
    that loop's variables into a record lets :func:`defend_documents` advance many documents through
    the *same* round together and issue one batched call per stage. See the module docstring.
    """

    turns: list
    priors: np.ndarray
    prior_texts: list
    pool: np.ndarray
    centroid: np.ndarray
    g_median: float
    profile: str
    cold_start: bool
    trace: DocumentTrace
    top_m: int = DEFAULT_TOP_M
    floor: Candidate | None = None
    best: Candidate | None = None
    history: list = field(default_factory=list)
    rounds: list = field(default_factory=list)
    target: float = 0.0
    loop_target: float = 0.0
    level: int = 0
    stalled: int = 0
    probes: int = 0
    round_index: int = 0
    stop_reason: str = ""
    finished: bool = False
    pending: list = field(default_factory=list)   # this round's admissible candidates

    def score(self, vector: np.ndarray, level: int, round_index: int) -> Candidate:
        s_max, s_top3, r_med = objective_scores(vector, self.priors, self.pool, self.top_m)
        generic = float(vector @ self.centroid) if self.centroid.size else 0.0
        candidate = Candidate(turns=[], level=level, round_index=round_index, s_max=s_max,
                              s_top3=s_top3, r_med=r_med, genericness=generic, vector=vector)
        # One lower-is-better number for both regimes, so the loop has a single comparison.
        candidate.objective = -generic if self.cold_start else s_max
        return candidate

    def finish(self, reason: str) -> None:
        self.stop_reason = reason
        self.finished = True


def _new_state(turns, priors, prior_texts, pool, *, profile, alpha, top_m,
               centroid=None, g_median=None) -> _DocState:
    """A :class:`_DocState` with its scoring context resolved but nothing computed yet."""
    turns = [str(turn) for turn in turns]
    priors = np.asarray(priors, dtype=float) if priors is not None else np.zeros((0, 0))
    pool = np.asarray(pool, dtype=float) if pool is not None else np.zeros((0, 0))
    if centroid is None:
        centroid = pool_centroid(pool)
    if g_median is None:
        g_median = median_genericness(pool, centroid)
    cold_start = priors.size == 0
    trace = DocumentTrace(n_priors=int(priors.shape[0]) if priors.size else 0,
                          cold_start=cold_start, alpha=float(alpha), n_turns=len(turns),
                          median_genericness=float(g_median), original=join_document(turns))
    return _DocState(turns=turns, priors=priors, prior_texts=list(prior_texts or []), pool=pool,
                     centroid=centroid, g_median=float(g_median), profile=profile or "",
                     cold_start=cold_start, trace=trace, top_m=top_m)


def _round_payload(state: _DocState, max_probes: int, first_round_candidates: int) -> dict:
    """The prompt context for one proposal round. Pure formatting; no model call."""
    want = first_round_candidates if state.round_index == 0 else 1
    want = min(want, max_probes - state.probes)
    return {
        "turns": state.best.turns,
        "n_turns": len(state.turns),
        "want": want,
        "profile": state.profile,
        "nearest_prior": nearest_prior_text(state.best, state.priors, state.prior_texts),
        "scores": _format_scores(state.best, state.target, state.cold_start, state.top_m),
        "trajectory": _format_trajectory(state.rounds, state.cold_start),
        "probes_left": max_probes - state.probes,
        "level": state.level,
        "cold_start": state.cold_start,
    }


def _absorb_round(state: _DocState, vectors: np.ndarray, escalate_after: int) -> None:
    """Score this round's candidates, update the best, and advance the escalation ladder."""
    state.round_index += 1
    record = RoundRecord(round_index=state.round_index, level=state.level,
                         requested=len(state.pending) or 1, admissible=len(state.pending),
                         probes_used=0)
    if len(state.pending) == 0:
        record.note = "no admissible candidate"
        state.stalled += 1
    else:
        state.probes += len(state.pending)
        record.probes_used = len(state.pending)
        scored = []
        for position, candidate_turns in enumerate(state.pending):
            candidate = state.score(vectors[position], state.level, state.round_index)
            candidate.turns = list(candidate_turns)
            scored.append(candidate)
            state.history.append(candidate)
        round_best = min(scored, key=lambda item: item.objective)
        record.objectives = [round(item.objective, 6) for item in scored]
        record.best_objective = round_best.objective
        if round_best.objective < state.best.objective - IMPROVEMENT_EPSILON:
            state.best = round_best
            record.improved = True
            state.stalled = 0
        else:
            state.stalled += 1
    state.rounds.append(record)
    state.pending = []

    if state.best.objective <= state.loop_target:
        state.finish("target_met")
    elif state.stalled >= escalate_after:
        state.level += 1
        state.stalled = 0
        if state.level >= N_ESCALATION_LEVELS:
            state.finish("ladder_exhausted")


def _finalize(state: _DocState) -> tuple[Candidate, Candidate, list, DocumentTrace]:
    trace = state.trace
    trace.stop_reason = state.stop_reason
    trace.rounds = state.rounds
    trace.probes_used = state.probes
    trace.n_rounds = len(state.rounds)
    trace.max_level = max((candidate.level for candidate in state.history), default=0)
    trace.final_objective = state.best.objective
    trace.target_met = state.best.objective <= state.loop_target
    trace.nearest_prior_index = nearest_prior_index(state.best, state.priors, state.prior_texts)
    return state.best, state.floor, state.history, trace


def defend_documents(items, *, embed, abstract, propose, alpha: float = DEFAULT_ALPHA,
                     max_probes: int = DEFAULT_MAX_PROBES,
                     escalate_after: int = DEFAULT_ESCALATE_AFTER,
                     top_m: int = DEFAULT_TOP_M,
                     first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                     fits=None) -> list:
    """Defend many **independent** documents in lockstep, one batched model call per stage.

    ``items`` is a list of dicts with ``turns``, ``priors``, ``prior_texts``, ``pool`` and optionally
    ``profile``/``centroid``/``g_median``. Independence is the caller's contract: documents from the
    same author at different cascade positions must NOT be passed together, because a later one is
    conditioned on an earlier one's output. :func:`defend_authors` is what enforces that.

    The injected callables are **batched**, mirroring
    :meth:`~._backends.PerTurnBatchRewriteDefense.rewrite_batch`:

    * ``embed(texts) -> (n, d)`` unit rows
    * ``abstract(list_of_turnlists) -> list of turnlist-or-None``
    * ``propose(list_of_payloads) -> list of list-of-candidate-turnlists``

    ``fits(text) -> (bool, n_tokens)`` reports whether a rendered document leaves room for a reply in
    the served window. A document that does not fit is emitted **unchanged**, with
    ``stop_reason="too_long"``, rather than being left to raise inside a batched ``chat`` call and
    take every other document in that batch with it. ``None`` disables the check (the fakes).
    """
    states = [_new_state(item["turns"], item.get("priors"), item.get("prior_texts"),
                         item.get("pool"), profile=item.get("profile", ""), alpha=alpha,
                         top_m=top_m, centroid=item.get("centroid"),
                         g_median=item.get("g_median"))
              for item in items]

    # --- window check, before any generation (no model call) ------------------
    for state in states:
        if fits is None:
            continue
        ok, n_tokens = fits(render_turn_blocks(state.turns))
        state.trace.prompt_tokens = int(n_tokens)
        if not ok:
            if AFR_TOO_LONG_PASSTHROUGH:
                passthrough = Candidate(turns=list(state.turns), level=0, round_index=0)
                state.floor = state.best = passthrough
                state.history = [passthrough]
                state.trace.outcome = "stage1_floor"
                state.trace.defended = state.trace.original
                state.finish("too_long")
            else:
                raise SystemExit(
                    f"afr: document {state.trace.doc_id or '<unknown>'} renders to "
                    f"{int(n_tokens):,} tokens, over the {AFR_MAX_MODEL_LEN:,}-token window, and "
                    f"AFR_TOO_LONG_PASSTHROUGH is off.\n"
                    f"Emitting it undefended would put untouched text inside the defended split, "
                    f"where it inflates the attack against a defense that never saw it -- so this "
                    f"stops instead.\n"
                    f"Cap the CORPUS rather than this defense, so every arm reads the same text:\n"
                    f"    python -m prompt_anonymity.data.build_subset --max-chars 40000 "
                    f"--out-source <name> ...\n"
                    f"Or raise AFR_MAX_MODEL_LEN (costs KV cache), or set "
                    f"AFR_TOO_LONG_PASSTHROUGH=1 to accept undefended pass-throughs."
                )

    live = [state for state in states if not state.finished]

    # --- stage 1: one batched abstraction pass over every live document -------
    if live:
        proposals = abstract([state.turns for state in live])
        for state, proposed in zip(live, proposals):
            if not admissible(proposed, state.turns):
                # A pass-through is visibly undefended, which is a far better failure than splicing
                # a truncated or fabricated rewrite into the dataset. Recorded so a run where it is
                # common is diagnosable as a prompt problem rather than read as a weak defense.
                proposed = list(state.turns)
                state.trace.stage1_ok = False
            state.pending = [proposed]

        # Floors and originals in ONE embedding call rather than two per document.
        floor_texts = [join_document(state.pending[0]) for state in live]
        original_texts = [join_document(state.turns) for state in live]
        vectors = embed(floor_texts + original_texts)
        for index, state in enumerate(live):
            floor = state.score(vectors[index], 0, 0)
            floor.turns = list(state.pending[0])
            state.pending = []
            state.floor = state.best = floor
            state.history = [floor]
            state.trace.stage1 = join_document(floor.turns)
            state.trace.baseline_objective = floor.objective
            state.trace.r_med_baseline = floor.r_med
            state.trace.original_objective = state.score(
                vectors[len(live) + index], 0, 0).objective

            target = (genericness_target(floor.genericness, state.g_median, alpha)
                      if state.cold_start else linkage_target(floor.s_max, floor.r_med, alpha))
            state.trace.target = float(target)
            state.target = target
            # Both regimes compare lower-is-better, so the cold-start target is negated alongside
            # its objective rather than special-casing every comparison.
            state.loop_target = -target if state.cold_start else target

            if state.cold_start and state.pool.size == 0:
                state.finish("no_priors")
            elif floor.objective <= state.loop_target:
                state.finish("no_priors" if state.cold_start else "stage1_sufficient")
            elif max_probes <= 0:
                state.finish("no_probes")

    # --- stage 2: the probe loop, all live documents advancing together -------
    while True:
        active = []
        for state in states:
            if state.finished:
                continue
            if state.probes >= max_probes:
                state.finish("probes_exhausted")
                continue
            active.append(state)
        if not active:
            break

        payloads = [_round_payload(state, max_probes, first_round_candidates)
                    for state in active]
        # A payload asking for zero candidates means the budget ran out mid-round; treat it as
        # exhausted rather than sending an empty request.
        exhausted = [state for state, payload in zip(active, payloads) if payload["want"] <= 0]
        for state in exhausted:
            state.finish("probes_exhausted")
        active = [state for state, payload in zip(active, payloads) if payload["want"] > 0]
        payloads = [payload for payload in payloads if payload["want"] > 0]
        if not active:
            break

        for state, candidates in zip(active, propose(payloads)):
            state.pending = [candidate for candidate in candidates
                             if admissible(candidate, state.turns)]

        # Every candidate from every document, embedded in one call.
        flat = [join_document(candidate) for state in active for candidate in state.pending]
        vectors = embed(flat) if flat else np.zeros((0, 0))
        offset = 0
        for state in active:
            count = len(state.pending)
            _absorb_round(state, vectors[offset:offset + count], escalate_after)
            offset += count

    return [_finalize(state) for state in states]


def defend_document(turns, priors, prior_texts, pool, *, embed, abstract, propose,
                    profile: str = "", alpha: float = DEFAULT_ALPHA,
                    max_probes: int = DEFAULT_MAX_PROBES,
                    escalate_after: int = DEFAULT_ESCALATE_AFTER,
                    top_m: int = DEFAULT_TOP_M,
                    first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                    centroid=None, g_median: float | None = None, fits=None,
                    ) -> tuple[Candidate, Candidate, list, DocumentTrace]:
    """One document through :func:`defend_documents`. Returns ``(winner, floor, history, trace)``.

    A thin wrapper, kept because a single document is the readable unit to reason about and to test.
    It takes the **unbatched** callables (``abstract(turns)``, ``propose(payload)``) and adapts them,
    so a caller with one document does not have to think in lists -- but the code underneath is the
    same batched path production uses, which is what keeps the two from drifting apart.
    """
    results = defend_documents(
        [{"turns": turns, "priors": priors, "prior_texts": prior_texts, "pool": pool,
          "profile": profile, "centroid": centroid, "g_median": g_median}],
        embed=embed,
        abstract=lambda batch: [abstract(item) for item in batch],
        propose=lambda batch: [propose(item) for item in batch],
        alpha=alpha, max_probes=max_probes, escalate_after=escalate_after, top_m=top_m,
        first_round_candidates=first_round_candidates, fits=fits)
    return results[0]


def nearest_prior_index(candidate: Candidate, priors, prior_texts) -> int | None:
    """Which of the author's earlier prompts this candidate is closest to, or ``None``.

    One matrix-vector product against vectors already in memory -- no model call, no probe. Recorded
    in the trace so a cascade can be audited after the fact: document ``k``'s nearest prior must
    resolve to a *defended* earlier document of the same author.
    """
    priors = np.asarray(priors, dtype=float)
    if not priors.size or not prior_texts or candidate.vector is None:
        return None
    return int(np.argmax(priors @ candidate.vector))


def nearest_prior_text(candidate: Candidate, priors, prior_texts) -> str:
    """The author's most similar earlier prompt, truncated, or a note when there is none.

    Truncated rather than summarized: the agent is looking for *which specifics recur*, and a
    paraphrase would launder away exactly the surface detail it needs to spot. The bound keeps a
    400-turn session from crowding the draft being edited out of the context window.
    """
    index = nearest_prior_index(candidate, priors, prior_texts)
    if index is None:
        return "(none -- this is their first prompt)"
    text = str(prior_texts[min(index, len(prior_texts) - 1)])
    if len(text) <= NEAREST_PRIOR_CHARS:
        return text
    return text[:NEAREST_PRIOR_CHARS] + "\n[...truncated]"


def gate_and_select(winner: Candidate, floor: Candidate, history, original_document: str, *,
                    gate, max_utility_loss: float, trace: DocumentTrace) -> Candidate:
    """Apply the utility gate to the winner, falling back down the escalation ladder if it fails.

    Escalation is what makes this load-bearing rather than a formality: level 2 is allowed to
    restate a prompt from scratch, and that can cost more utility than the linkage it buys. So a
    rejected winner falls back to the *least aggressive* candidate that still improved on the floor,
    and only then to the floor itself. At most two candidates are ever judged, so the ladder cannot
    turn the gate into the dominant cost of the run.

    The floor is never itself rejected: it is the abstraction pass, the same operation every other
    rewriting defense in this package performs, and having nothing to emit is not an option.
    """
    if winner is floor or not winner.turns:
        trace.outcome = "stage1_floor"
        trace.emitted_level = floor.level
        trace.utility = {"checked": False, "reason": "loop kept the stage-1 output"}
        return floor

    losses = gate(original_document, [join_document(winner.turns)])
    loss = float(losses[0]) if losses else 1.0
    trace.utility = {"checked": True, "winner_loss": round(loss, 4),
                     "winner_level": winner.level, "max_utility_loss": max_utility_loss}
    if loss <= max_utility_loss:
        trace.outcome = "loop_winner"
        trace.emitted_level = winner.level
        return winner

    # Least aggressive first, then best-scoring within that level.
    alternatives = sorted(
        (item for item in history
         if item is not winner and item is not floor and item.turns
         and item.objective < floor.objective - IMPROVEMENT_EPSILON),
        key=lambda item: (item.level, item.objective),
    )
    if alternatives:
        fallback = alternatives[0]
        fallback_losses = gate(original_document, [join_document(fallback.turns)])
        fallback_loss = float(fallback_losses[0]) if fallback_losses else 1.0
        trace.utility["fallback_loss"] = round(fallback_loss, 4)
        trace.utility["fallback_level"] = fallback.level
        if fallback_loss <= max_utility_loss:
            trace.outcome = "gate_fallback"
            trace.emitted_level = fallback.level
            trace.final_objective = fallback.objective
            return fallback

    trace.outcome = "stage1_floor"
    trace.emitted_level = floor.level
    trace.final_objective = floor.objective
    return floor


#: Documents per batch once the cascade cap is passed. Bounds the size of one embedding call and one
#: proposal batch; vLLM schedules within it, so larger mostly costs memory rather than buying speed.
DEFAULT_DOCUMENT_BATCH = 256


def gate_and_select_batch(results, *, gate, max_utility_loss: float) -> list[Candidate]:
    """The utility gate over many documents at once: at most **two** batched judging passes.

    ``gate(pairs) -> list of per-variant losses``, where each pair is
    ``(original_document, [variant, ...])``. Run per document this was four generations each and the
    single largest avoidable cost in the loop; run as two passes over the whole batch it is two
    calls total, regardless of how many documents are in flight.

    Pass 1 judges every winner. Pass 2 judges a fallback only for the winners that failed, walking
    *down* the escalation ladder to the least aggressive candidate that still beat the floor -- so a
    level-2 structural rewrite that cost too much utility falls back to a level-0 edit rather than
    all the way to the abstraction pass.
    """
    chosen: list[Candidate | None] = [None] * len(results)
    judged: list[int] = []
    for index, (winner, floor, _history, trace) in enumerate(results):
        if winner is floor or not winner.turns:
            trace.outcome = "stage1_floor"
            trace.emitted_level = floor.level
            trace.utility = {"checked": False, "reason": "loop kept the stage-1 output"}
            chosen[index] = floor
        else:
            judged.append(index)

    if judged:
        losses = gate([(results[i][3].original, [join_document(results[i][0].turns)])
                       for i in judged])
        retry: list[int] = []
        for index, per_variant in zip(judged, losses):
            winner, floor, _history, trace = results[index]
            loss = float(per_variant[0]) if per_variant else 1.0
            trace.utility = {"checked": True, "winner_loss": round(loss, 4),
                             "winner_level": winner.level, "max_utility_loss": max_utility_loss}
            if loss <= max_utility_loss:
                trace.outcome = "loop_winner"
                trace.emitted_level = winner.level
                chosen[index] = winner
            else:
                retry.append(index)

        fallbacks: dict[int, Candidate] = {}
        for index in retry:
            winner, floor, history, _trace = results[index]
            alternatives = sorted(
                (item for item in history
                 if item is not winner and item is not floor and item.turns
                 and item.objective < floor.objective - IMPROVEMENT_EPSILON),
                key=lambda item: (item.level, item.objective))
            if alternatives:
                fallbacks[index] = alternatives[0]

        if fallbacks:
            order = list(fallbacks)
            losses = gate([(results[i][3].original, [join_document(fallbacks[i].turns)])
                           for i in order])
            for index, per_variant in zip(order, losses):
                loss = float(per_variant[0]) if per_variant else 1.0
                trace = results[index][3]
                trace.utility["fallback_loss"] = round(loss, 4)
                trace.utility["fallback_level"] = fallbacks[index].level
                if loss <= max_utility_loss:
                    trace.outcome = "gate_fallback"
                    trace.emitted_level = fallbacks[index].level
                    trace.final_objective = fallbacks[index].objective
                    chosen[index] = fallbacks[index]

        for index in retry:
            if chosen[index] is None:
                _winner, floor, _history, trace = results[index]
                trace.outcome = "stage1_floor"
                trace.emitted_level = floor.level
                trace.final_objective = floor.objective
                chosen[index] = floor

    return [candidate for candidate in chosen]


def defend_authors(timelines, pool_for, *, embed, abstract, propose, gate, signature,
                   alpha: float = DEFAULT_ALPHA, max_probes: int = DEFAULT_MAX_PROBES,
                   escalate_after: int = DEFAULT_ESCALATE_AFTER, top_m: int = DEFAULT_TOP_M,
                   max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                   first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                   cascade_depth: int = AFR_CASCADE_DEPTH,
                   document_batch: int = DEFAULT_DOCUMENT_BATCH,
                   fits=None, on_document=None, store=None, pool_key: str = "") -> dict:
    """Cascade many authors' timelines **in lockstep**, batching every stage across them.

    ``timelines`` is ``[(author, [(doc_id, turns), ...]), ...]`` in the order the documents were
    written; ``pool_for(author)`` returns that author's reference matrix. Returns
    ``{author: [defended turn list, ...]}``.

    Within an author the cascade is causal and therefore serial: document ``k`` is defended against
    the *defended* text of ``1..k-1``. Across authors nothing is shared, so position ``p`` of every
    author is one batch. That is the whole throughput story -- one model call per stage per position
    instead of one per document (see the module docstring).

    Two phases:

    **Phase A, positions below** ``cascade_depth``: true lockstep, one batch per position, batch
    width equal to the number of authors still that long.

    **Phase B, everything past the cap**: those documents all score against the same frozen first
    ``cascade_depth`` defended documents, so they are mutually independent and run as a few wide
    batches rather than as a long thin tail. Without this a single 450-document author would
    serialize ~400 steps at batch 1 and dominate the wall clock.

    ``on_document(author, doc_id, position, trace)`` is called as each document completes, so a
    caller can flush its log continuously rather than after a whole author.

    ``store`` is an optional :class:`DocumentStore`. When given, every finished document is written
    to it immediately and a resumed run replays the cascade from it without touching the model. The
    replay is exact rather than approximate: a document's key carries the defended prior chain it
    was produced against (see :func:`document_key`), so a hit is only ever a document whose input is
    identical, and anything downstream of a change recomputes.
    """
    timelines = [(author, list(documents)) for author, documents in timelines]
    pools = {author: np.asarray(pool_for(author), dtype=float) for author, _ in timelines}
    centroids = {author: pool_centroid(pools[author]) for author, _ in timelines}
    medians = {author: median_genericness(pools[author], centroids[author])
               for author, _ in timelines}

    prior_texts: dict = {author: [] for author, _ in timelines}
    prior_vectors: dict = {author: [] for author, _ in timelines}
    profiles: dict = {author: "" for author, _ in timelines}
    defended: dict = {author: [None] * len(documents) for author, documents in timelines}

    def run(batch) -> list[int]:
        """Defend one batch of ``(author, position, doc_id, turns)``.

        Returns the positions **within this batch** that were actually computed, so phase A knows
        which authors still need a ``signature`` call and which had their profile restored from the
        store. A cached document costs no generation at all.
        """
        pending, keys = [], {}
        for index, (author, position, doc_id, turns) in enumerate(batch):
            if store is None:
                pending.append(index)
                continue
            key = document_key(pool_key, doc_id, turns, prior_texts[author])
            keys[index] = key
            record = store.get(key)
            if record is None:
                pending.append(index)
                continue
            turns_out, profile = record
            defended[author][position] = turns_out
            if profile is None:
                # Committed mid-position: the expensive work is here, but the profile that follows
                # it was never written. Restoring costs one `signature` call instead of the ~11
                # generations the document itself took.
                profileless.append(index)
            else:
                restored[author] = profile
        if not pending:
            return []

        items = [{"turns": batch[index][3],
                  "priors": (np.vstack(prior_vectors[batch[index][0]])
                             if prior_vectors[batch[index][0]] else np.zeros((0, 0))),
                  "prior_texts": prior_texts[batch[index][0]],
                  "pool": pools[batch[index][0]],
                  "profile": profiles[batch[index][0]],
                  "centroid": centroids[batch[index][0]],
                  "g_median": medians[batch[index][0]]}
                 for index in pending]
        results = defend_documents(
            items, embed=embed, abstract=abstract, propose=propose, alpha=alpha,
            max_probes=max_probes, escalate_after=escalate_after, top_m=top_m,
            first_round_candidates=first_round_candidates, fits=fits)
        winners = gate_and_select_batch(results, gate=gate, max_utility_loss=max_utility_loss)
        for index, winner, result in zip(pending, winners, results):
            author, position, doc_id, _turns = batch[index]
            trace = result[3]
            turns_out = [str(turn) for turn in winner.turns]
            trace.doc_id, trace.author_id, trace.position = str(doc_id), str(author), position
            trace.defended = join_document(turns_out)
            defended[author][position] = turns_out
            # COMMIT NOW, before anything else in this position runs. This is the whole point of the
            # store: a document that took ~11 generations must survive the next preemption, and on a
            # 2-hour preempt window a position of 40 documents does not reliably finish. The profile
            # is not known yet (it needs the signature call that ends the position), so it is written
            # as null and filled in below; a resume that finds a null profile pays one signature
            # call rather than redoing the document.
            if store is not None:
                store.put(keys[index], turns_out, None)
            if on_document is not None:
                on_document(author, doc_id, position, trace)
        return pending

    #: Profiles recovered from the store this position, so phase A can skip their signature call.
    restored: dict = {}
    #: Documents restored from a mid-position commit, whose profile still has to be generated.
    profileless: list = []

    # --- phase A: lockstep over the cascaded prefix ---------------------------
    depth = max(0, int(cascade_depth))
    for position in range(depth):
        batch = [(author, position, documents[position][0], documents[position][1])
                 for author, documents in timelines if position < len(documents)]
        if not batch:
            break
        restored, profileless = {}, []
        need = run(batch)
        # The cascade step: what the attacker will see becomes the next document's prior. Only the
        # embeddings and profiles of the capped prefix are ever extended.
        fresh = [(author, join_document(defended[author][position]))
                 for author, _position, _doc_id, _turns in batch]
        vectors = embed([text for _author, text in fresh])
        for index, (author, text) in enumerate(fresh):
            prior_texts[author].append(text)
            prior_vectors[author].append(vectors[index])
        # A profile is generated for a document computed this position, and for one restored from a
        # mid-position commit that never got its profile written. A restored record that HAS a
        # profile costs nothing at all -- that is the fully-cached path.
        advance = sorted(set(need) | set(profileless))
        if advance:
            updated = signature([(profiles[fresh[index][0]], fresh[index][1]) for index in advance])
            for index, profile in zip(advance, updated):
                profiles[fresh[index][0]] = profile
        for author, profile in restored.items():
            profiles[author] = profile
        # Re-write those records now that the profile exists, completing the entries `run` wrote
        # with a null profile. Same key, so this replaces rather than duplicates.
        if store is not None:
            for index in advance:
                author, position_, doc_id, turns = batch[index]
                store.put(document_key(pool_key, doc_id, turns,
                                       prior_texts[author][:position_]),
                          defended[author][position_], profiles[author])

    # --- phase B: everything past the cap, in wide independent batches --------
    tail = [(author, position, documents[position][0], documents[position][1])
            for author, documents in timelines
            for position in range(depth, len(documents))]
    for start in range(0, len(tail), max(1, document_batch)):
        slice_ = tail[start:start + document_batch]
        restored, profileless = {}, []
        computed = run(slice_)
        # Phase B never extends the chain, so the profile is unchanged. `run` already committed each
        # document as it finished; this only completes those records with the (unchanged) profile so
        # a later resume reads them as fully cached rather than profile-less.
        if store is not None:
            for index in sorted(set(computed) | set(profileless)):
                author, position, doc_id, turns = slice_[index]
                store.put(document_key(pool_key, doc_id, turns, prior_texts[author]),
                          defended[author][position], profiles[author])

    return defended


def defend_author(documents, pool, *, embed, abstract, propose, gate, signature,
                  alpha: float = DEFAULT_ALPHA, max_probes: int = DEFAULT_MAX_PROBES,
                  escalate_after: int = DEFAULT_ESCALATE_AFTER, top_m: int = DEFAULT_TOP_M,
                  max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                  first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                  cascade_depth: int = AFR_CASCADE_DEPTH, fits=None,
                  ) -> tuple[list, list]:
    """One author through :func:`defend_authors`, with the unbatched callables. See that function.

    Kept because a single author's timeline is the readable unit to reason about and to test; the
    code underneath is the batched path production uses.
    """
    traces: list = []
    defended = defend_authors(
        [("_", documents)], lambda _author: pool,
        embed=embed,
        abstract=lambda batch: [abstract(item) for item in batch],
        propose=lambda batch: [propose(item) for item in batch],
        gate=lambda pairs: [gate(original, variants) for original, variants in pairs],
        signature=lambda pairs: [signature(profile, text) for profile, text in pairs],
        alpha=alpha, max_probes=max_probes, escalate_after=escalate_after, top_m=top_m,
        max_utility_loss=max_utility_loss, first_round_candidates=first_round_candidates,
        cascade_depth=cascade_depth, fits=fits,
        on_document=lambda _a, _d, _p, trace: traces.append(trace))
    return defended["_"], traces


# --- the reference pool ------------------------------------------------------

def select_reference_pool(doc_ids, *, n_reference: int = DEFAULT_N_REFERENCE,
                          seed: int = DEFAULT_SEED) -> list[int]:
    """Positions of the reference documents, drawn seeded over a **sorted** ``doc_id`` list.

    Sorting before drawing is what makes the sample survive a rebuild that changes row order -- the
    same reasoning as ``build_subset.select_authors``. The chosen positions come back in sorted
    order so the pool's digest, and therefore every author's cache key, is stable.
    """
    order = sorted(range(len(doc_ids)), key=lambda index: str(doc_ids[index]))
    if len(order) <= n_reference:
        return order
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(order), size=int(n_reference), replace=False)
    return [order[int(index)] for index in sorted(chosen.tolist())]


# --- the local backend -------------------------------------------------------

class _LocalBackend:
    """One vLLM engine serving every generative job, plus the Harrier embedder.

    Held together because they share a GPU. Unlike ``loo_unlink``'s 3B generator, the agent here is a
    30B-class model, so ``gpu_memory_utilization`` is raised and the job needs a card that can hold
    both -- see ``experiments/run_afr.sbatch``.

    Every ``chat`` call is made at a batch size fixed by its call site (see the module docstring's
    determinism note). Nothing here batches across documents or authors: a cascade that produced
    different text depending on what else happened to be in flight would be uncacheable.
    """

    def __init__(self, prompts: dict, *, model: str | None = None,
                 answer_tokens: int = DEFAULT_ANSWER_TOKENS,
                 propose_tokens: int = DEFAULT_PROPOSE_TOKENS,
                 seed: int = DEFAULT_SEED, featurizer=None, checkpoint=None):
        self.prompts = prompts
        self.model = model
        #: Callable returning the resolved checkpoint, supplied by the defense so the
        #: download-refusal guard cannot be bypassed by reaching the engine another way.
        self._checkpoint = checkpoint
        self.answer_tokens = answer_tokens
        self.propose_tokens = propose_tokens
        self.seed = seed
        self._featurizer = featurizer
        self._llm = None
        self._tokenizer = None
        self._sampling = {}

    @property
    def prompt_budget(self) -> int:
        """Tokens a DOCUMENT may occupy: the window, minus the longest completion, minus the wrapper.

        Three terms, and the third was missing at first. The longest completion is a ``propose``
        round returning up to three full rewrites in one reply; the wrapper is everything that
        surrounds the document in that prompt (system prompt, escalation rung, author profile,
        nearest-prior excerpt, scores, trajectory). Measuring only the document is how a 30,720-token
        document became a 32,769-token prompt and hard-failed the request.
        """
        return max(1, AFR_MAX_MODEL_LEN - self.propose_tokens - PROMPT_RESERVE_TOKENS)

    def count_tokens(self, text: str) -> int:
        """Length of ``text`` in the served model's tokens, via the engine's own tokenizer.

        The engine's, not a separately loaded one: the whole point is to measure against what vLLM
        will actually accept. Building the engine is what makes the tokenizer available, so this
        loads it on first use exactly as :meth:`chat` does."""
        if self._tokenizer is None:
            self._tokenizer = self._engine().get_tokenizer()
        return len(self._tokenizer(text).input_ids)

    # -- embedding --
    def featurizer(self):
        if self._featurizer is None:
            from ..features.harrier import HarrierFeaturizer

            # batch_size is explicit because the default is sized for a card the embedder has to
            # itself, and here it shares one with a 27B model whose reservation is permanent.
            # It never changes a vector, so it stays out of the featurizer's params() and cache key.
            self._featurizer = HarrierFeaturizer(batch_size=AFR_EMBED_BATCH)
        return self._featurizer

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a batch, halving the forward-pass size and retrying if the GPU is momentarily full.

        The embedder shares a card with a model that has already reserved most of it, and the amount
        left over moves with vLLM's in-flight work. A transient OOM here is a reason to take smaller
        bites, not to lose the job -- which is what it did before this: one OOM inside a forward pass
        killed the run and discarded the chunk. The reduced size sticks, so the run settles at
        whatever fits rather than rediscovering the limit on every call.
        """
        texts = list(texts)
        if not texts:
            return np.zeros((0, 0))
        featurizer = self.featurizer()
        while True:
            try:
                return np.asarray(featurizer.featurize(texts), dtype=float)
            except Exception as error:  # noqa: BLE001 - torch.OutOfMemoryError without importing torch
                if "out of memory" not in str(error).lower() or featurizer.batch_size <= 1:
                    raise
                featurizer.batch_size = max(1, featurizer.batch_size // 2)
                print(f"[afr] embedder OOM; retrying at batch_size={featurizer.batch_size} "
                      f"(set AFR_EMBED_BATCH lower to start there)", flush=True)
                try:
                    import torch

                    torch.cuda.empty_cache()
                except Exception:  # noqa: BLE001 - the retry matters, the reclaim is a bonus
                    pass

    # -- generation --
    def _engine(self):
        if self._llm is None:
            from ._backends import (configure_cuda_toolkit, model_checkpoint, resolve_model_path,
                                    shared_checkpoint)

            configure_cuda_toolkit()  # must precede the import: vLLM reads the environment at import

            from vllm import LLM, SamplingParams

            # Resolved by the defense when it owns this backend, so the download-refusal guard in
            # `AgenticFootprintDefense.agent_checkpoint` is on the only path that reaches vLLM.
            # The fallback here covers a backend built directly (tests, a REPL).
            if self._checkpoint is not None:
                path = self._checkpoint()
            else:
                path = (resolve_model_path(shared_checkpoint(self.model) or self.model)
                        if self.model else model_checkpoint("afr", MODEL_ENV_VAR,
                                                            local_only=not AFR_ALLOW_DOWNLOAD))
            # Hand back what the embedder's allocator is hoarding, BEFORE vLLM profiles the device.
            #
            # Harrier runs first -- it embeds the whole reference pool before any generation -- and
            # PyTorch's caching allocator keeps the blocks from that peak instead of returning them
            # to the driver, which can starve vLLM's memory request enough to fail startup.
            # `empty_cache` releases the cached-but-unused blocks; Harrier's weights stay resident,
            # which is what we want, since it is used again between rounds.
            try:
                import torch

                if torch.cuda.is_available():
                    free_before = torch.cuda.mem_get_info()[0] / 2**30
                    torch.cuda.empty_cache()
                    free_after = torch.cuda.mem_get_info()[0] / 2**30
                    print(f"[afr] released the embedder's cached blocks: "
                          f"{free_before:.1f} -> {free_after:.1f} GiB free")
            except Exception as error:  # noqa: BLE001 - a diagnostic must not break the run
                print(f"note: could not reclaim cached GPU memory ({type(error).__name__}: {error})")

            print(f"[afr] loading agent {path} (vLLM), max_model_len={AFR_MAX_MODEL_LEN:,}, "
                  f"gpu_memory_utilization={AFR_GPU_MEM_UTIL}, "
                  f"max_num_seqs={AFR_MAX_NUM_SEQS}, "
                  f"enforce_eager={AFR_ENFORCE_EAGER}")
            # `enable_prefix_caching` is a real saving here rather than a default worth copying: the
            # system prompt and escalation ladder are identical for every call in the corpus, and
            # within one document the author profile and the nearest-prior excerpt repeat across all
            # of its proposal rounds. Reusing that KV prefix removes the re-prefill each round.
            options = dict(model=path, dtype="auto", gpu_memory_utilization=AFR_GPU_MEM_UTIL,
                           max_model_len=AFR_MAX_MODEL_LEN, enforce_eager=AFR_ENFORCE_EAGER,
                           max_num_seqs=AFR_MAX_NUM_SEQS,
                           enable_prefix_caching=True, seed=self.seed)
            # Speculative decoding (see AFR_SPEC_TOKENS). Passed as a nested dict because the flat
            # `speculative_model=` / `ngram_prompt_lookup_max=` spelling was removed in vLLM v1; the
            # retry below covers the versions that still want the old one, or that were built
            # without ngram support, since neither is worth failing a 10-hour job over.
            speculative = None
            if AFR_SPEC_TOKENS > 0:
                speculative = {"method": "ngram",
                               "num_speculative_tokens": AFR_SPEC_TOKENS,
                               "prompt_lookup_max": AFR_SPEC_NGRAM_MAX,
                               "prompt_lookup_min": AFR_SPEC_NGRAM_MIN}
                print(f"[afr] speculative decoding: ngram, {AFR_SPEC_TOKENS} draft tokens, "
                      f"lookup {AFR_SPEC_NGRAM_MIN}-{AFR_SPEC_NGRAM_MAX}")
            try:
                self._llm = LLM(**options, **({"speculative_config": speculative}
                                              if speculative else {}))
            except (TypeError, ValueError) as error:
                if speculative is None:
                    raise
                # Do NOT swallow a real startup failure (a bad checkpoint, a missing file) as "no
                # speculation": retry only when the complaint is about the argument itself, or about
                # the cache blocks speculation is what reserves extra of. That second case is not
                # hypothetical -- drafting costs one Mamba recurrent-state block per drafted position
                # on this hybrid checkpoint, which is how a default max_num_seqs of 256 stopped
                # fitting. AFR_MAX_NUM_SEQS is the real fix; this is the seatbelt for the next
                # checkpoint that budgets differently, since a slower run beats a dead one.
                blame = str(error).lower()
                if not any(word in blame for word in
                           ("speculative", "ngram", "cache block", "max_num_seqs")):
                    raise
                print(f"note: this vLLM rejected the speculative config, continuing without it "
                      f"({type(error).__name__}: {error})")
                self._llm = LLM(**options)
            self._sampling = {
                # Temperature 0 everywhere: the defense must be a deterministic function of its
                # input, or the content-addressed cache would return a different cascade on a hit
                # than on a miss.
                "signature": SamplingParams(temperature=0.0, max_tokens=DEFAULT_SIGNATURE_TOKENS),
                "abstract": SamplingParams(temperature=0.0, max_tokens=self.propose_tokens),
                "propose": SamplingParams(temperature=0.0, max_tokens=self.propose_tokens),
                "answer": SamplingParams(temperature=0.0, max_tokens=self.answer_tokens),
                "judge": SamplingParams(temperature=0.0, max_tokens=32),
            }
        return self._llm

    def chat_pairs(self, pairs, kind: str) -> list[str]:
        """One batched call over ``(system, user)`` pairs; replies come back in input order.

        Pairs rather than one shared system prompt because a proposal round's system prompt carries
        the escalation rung, which differs per document -- and putting those in separate calls to
        keep the system prompt uniform is exactly the batch-1 mistake this file was rewritten to
        undo. vLLM batches heterogeneous prompts fine; prefix caching still shares whatever leading
        tokens they do have in common.
        """
        pairs = list(pairs)
        if not pairs:
            return []
        engine = self._engine()
        conversations = [[{"role": "system", "content": system},
                          {"role": "user", "content": user}] for system, user in pairs]
        outputs = engine.chat(conversations, self._sampling[kind], use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

    def chat(self, system: str, users: list[str], kind: str) -> list[str]:
        """One batched chat call: same system prompt, many user messages, replies in input order."""
        return self.chat_pairs([(system, user) for user in users], kind)

    def close(self) -> None:
        if self._llm is not None:
            from ._backends import shutdown_vllm

            shutdown_vllm(self._llm)
            self._llm = None


# --- the defense -------------------------------------------------------------

class AgenticFootprintDefense(CachedDefense):
    """Agentic footprint reduction at one residual-linkage level.

    Parameters
    ----------
    alpha : float
        Residual linkage fraction. ``0.0`` targets "as distant from your earlier prompts as a
        stranger's prompt is"; ``0.5`` keeps half the excess. The sweep axis.
    max_probes : int
        Candidate embeddings allowed per document. ``0`` runs the abstraction pass and stops --
        the ``afr_stage1`` ablation, which isolates the loop rather than the model.
    escalate_after, top_m, max_utility_loss, n_reference, seed
        See the module constants.
    """

    name = "afr"
    version = "1"

    #: Never by DOCUMENT: ``select_shard`` interleaves rows, and a shard holding an arbitrary subset
    #: of an author would cascade document ``k`` against the wrong priors. See the module docstring.
    shardable = False

    #: But splitting whole AUTHORS across tasks is exact -- authors share nothing except the
    #: reference pool, and :meth:`reference_pool_ids` keeps that identical in every shard. This is
    #: the only lever that shortens the wall clock without changing a single output token.
    #: ``apply_defenses`` reads it and switches to ``select_author_shard``.
    shardable_by = "author"

    #: AUTHORS between cache checkpoints -- an author is the atomic unit of the cascade.
    checkpoint_every = DEFAULT_CHECKPOINT_EVERY

    def __init__(self, *, alpha: float = DEFAULT_ALPHA, max_probes: int = DEFAULT_MAX_PROBES,
                 escalate_after: int = DEFAULT_ESCALATE_AFTER, top_m: int = DEFAULT_TOP_M,
                 max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                 first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                 n_reference: int = DEFAULT_N_REFERENCE, seed: int = DEFAULT_SEED,
                 model: str | None = None, prompts_path: str | None = None,
                 answer_tokens: int = DEFAULT_ANSWER_TOKENS,
                 log_dir: str | None = None, backend=None):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}.")
        if max_probes < 0:
            raise ValueError(f"max_probes must be >= 0, got {max_probes}.")
        if escalate_after < 1:
            raise ValueError(f"escalate_after must be >= 1, got {escalate_after}.")
        self.alpha = float(alpha)
        self.max_probes = int(max_probes)
        self.escalate_after = int(escalate_after)
        self.top_m = int(top_m)
        self.max_utility_loss = float(max_utility_loss)
        self.first_round_candidates = int(first_round_candidates)
        self.n_reference = int(n_reference)
        self.seed = int(seed)
        self.model = model
        self.answer_tokens = int(answer_tokens)
        self.prompts = load_prompts(prompts_path)
        self.log_dir = log_dir
        self._backend = backend
        self._log_handle = None

    def params(self) -> dict:
        """Everything that changes the output, prompts and pool draw included.

        The prompt strings are hashed rather than embedded so the key stays short, but they ARE in
        the key: editing the ladder or the propose contract changes every rewrite, and a run that
        silently mixed two contracts in one parquet would be uninterpretable.
        """
        return {
            "alpha": round(self.alpha, 4),
            "max_probes": self.max_probes,
            "escalate_after": self.escalate_after,
            "top_m": self.top_m,
            "max_utility_loss": round(self.max_utility_loss, 4),
            "first_round_candidates": self.first_round_candidates,
            "n_reference": self.n_reference,
            "seed": self.seed,
            "answer_tokens": self.answer_tokens,
            "model": self.model or os.environ.get(MODEL_ENV_VAR) or "models.toml:afr",
            "prompts_sha": prompts_digest(self.prompts),
            "turn_separator": TURN_SEPARATOR,
        }

    # -- model-backed callables handed to the loop --

    def backend(self):
        if self._backend is None:
            self._backend = _LocalBackend(self.prompts, model=self.model,
                                          answer_tokens=self.answer_tokens, seed=self.seed,
                                          checkpoint=self.agent_checkpoint)
        return self._backend

    def _repair_batch(self, failures, kind: str) -> dict:
        """One batched format-repair pass. ``failures`` is ``[(key, n_turns, error, reply)]``.

        The repair is a separate call rather than a longer prompt because the failure it fixes is
        not a reasoning failure: the model produced a fine rewrite in the wrong wrapper, and showing
        it its own reply is the shortest path to the same rewrite in the right one. Batched for the
        same reason everything else here is -- a corpus-wide pass may have hundreds of malformed
        replies, and repairing them one at a time would reintroduce the batch-1 cost.
        """
        failures = list(failures)
        if not failures:
            return {}
        pairs = [(render_template(self.prompts["repair_system_prompt"], {"N_TURNS": str(n_turns)}),
                  render_template(self.prompts["repair_user_template"],
                                  {"ERROR": error, "REPLY": reply}))
                 for _key, n_turns, error, reply in failures]
        replies = self.backend().chat_pairs(pairs, kind)
        repaired = {}
        for (key, n_turns, _error, _reply), reply in zip(failures, replies):
            parsed = parse_turns(reply, n_turns)
            if parsed is not None:
                repaired[key] = parsed
        return repaired

    def _abstract(self, batch) -> list:
        """Stage 1, batched: generalize each whole document upward.

        Returns one turn list (or ``None``) per input. This stage reads no prior state, so the
        caller is free to hand it every document in the corpus at once -- which is what
        :meth:`transform` does before the cascade starts.
        """
        batch = [list(turns) for turns in batch]
        if not batch:
            return []
        pairs = []
        for turns in batch:
            n_turns = len(turns)
            pairs.append((
                render_template(self.prompts["abstract_system_prompt"],
                                {"N_TURNS": str(n_turns)}),
                render_template(self.prompts["abstract_user_template"],
                                {"DOCUMENT": render_turn_blocks(turns),
                                 "N_TURNS": str(n_turns)})))
        replies = self.backend().chat_pairs(pairs, "abstract")
        results: list = [None] * len(batch)
        failures = []
        for index, (turns, reply) in enumerate(zip(batch, replies)):
            n_turns = len(turns)
            parsed = parse_turns(reply, n_turns)
            if parsed is not None:
                results[index] = parsed
            else:
                failures.append((index, n_turns,
                                 f"expected exactly {n_turns} <turn> block(s), "
                                 f"numbered 1..{n_turns}", reply))
        for index, parsed in self._repair_batch(failures, "abstract").items():
            results[index] = parsed
        return results

    def _propose(self, batch) -> list:
        """Stage 2, batched: one proposal round for each in-flight document.

        Each document asks for its candidates inside ONE reply (cheaper than one call per
        candidate), and every document's request goes in the SAME batch. The escalation rung lives
        in the system prompt, so the prompts differ per document -- which is why this goes through
        :meth:`_LocalBackend.chat_pairs` rather than a shared-system ``chat``.
        """
        batch = list(batch)
        if not batch:
            return []
        pairs = []
        for payload in batch:
            n_turns = int(payload["n_turns"])
            want = int(payload["want"])
            level = min(int(payload["level"]), N_ESCALATION_LEVELS - 1)
            template = (self.prompts["propose_cold_start_template"] if payload["cold_start"]
                        else self.prompts["propose_user_template"])
            pairs.append((
                render_template(self.prompts["propose_system_prompt"],
                                {"N_TURNS": str(n_turns), "N_CANDIDATES": str(want),
                                 "ESCALATION": self.prompts["escalation_levels"][level]}),
                render_template(template, {
                    "DOCUMENT": render_turn_blocks(payload["turns"]),
                    "N_TURNS": str(n_turns),
                    "N_CANDIDATES": str(want),
                    "PROFILE": payload["profile"] or "(nothing recorded yet)",
                    "NEAREST_PRIOR": payload["nearest_prior"],
                    "SCORES": payload["scores"],
                    "TRAJECTORY": payload["trajectory"],
                    "PROBES_LEFT": str(payload["probes_left"]),
                })))
        replies = self.backend().chat_pairs(pairs, "propose")
        results: list = [[] for _ in batch]
        failures = []
        for index, (payload, reply) in enumerate(zip(batch, replies)):
            n_turns, want = int(payload["n_turns"]), int(payload["want"])
            candidates = parse_candidates(reply, n_turns, want)
            if candidates:
                results[index] = candidates
            else:
                failures.append((index, n_turns,
                                 f"expected {want} candidate(s), each with exactly {n_turns} "
                                 f"<turn> block(s)", reply))
        for index, parsed in self._repair_batch(failures, "propose").items():
            results[index] = [parsed]
        return results

    def agent_checkpoint(self) -> str:
        """The resolved agent checkpoint, refusing a download unless explicitly opted in.

        vLLM cannot tell "a path that does not exist" from "a hub repo id" -- both are just strings
        it will happily fetch. So the check has to happen here: if the resolved checkpoint is not a
        directory on disk and :data:`AFR_ALLOW_DOWNLOAD` is unset, stop. A 30-50 GB download nobody
        asked for is worse than a failed job, and on a cluster whose ``$HF_HOME`` still points at a
        home directory it takes the quota with it.
        """
        from ._backends import model_checkpoint, resolve_model_path, shared_checkpoint

        if self.model:
            path = resolve_model_path(shared_checkpoint(self.model) or self.model)
            print(f"[{self.name}] agent checkpoint: {path}")
        else:
            path = model_checkpoint(self.name, MODEL_ENV_VAR, local_only=not AFR_ALLOW_DOWNLOAD)
        if not AFR_ALLOW_DOWNLOAD and not Path(path).is_dir():
            raise SystemExit(
                f"[{self.name}] {path!r} is not a directory on this machine, so vLLM would try to "
                f"DOWNLOAD it (30-50 GB for a model this size, into $HF_HOME -- check that it is "
                f"not your home quota).\n"
                f"  - point $AFR_MODEL at a checkpoint already on disk, or\n"
                f"  - fix the [afr] path in models.toml, or\n"
                f"  - set AFR_ALLOW_DOWNLOAD=1 (and $HF_HOME to scratch) if you really do want it "
                f"fetched."
            )
        return path

    def _resolve_checkpoints(self) -> None:
        """Fail now, with a message, if either checkpoint cannot be located.

        Neither model is loaded here -- only the paths are resolved, which is filesystem work. The
        point is that a mis-configured run dies in the first second naming ``$AFR_MODEL`` or
        ``$HARRIER_MODEL``, instead of an hour later with an ``OSError`` from inside the cascade --
        or, worse, silently downloading its way through a disk quota.
        """
        from ..features.harrier import HarrierFeaturizer

        self.agent_checkpoint()
        print(f"[{self.name}] embedder checkpoint: {HarrierFeaturizer().checkpoint()}")

    def _fits(self, rendered_document: str) -> tuple[bool, int]:
        """``(fits, n_tokens)`` for a rendered document against the served window.

        Measured on the document alone rather than on the fully assembled prompt: the prompt also
        carries a system prompt, the profile and a bounded prior excerpt, so this is an
        under-estimate by a fixed few hundred tokens. That is deliberate -- the budget already
        subtracts a whole ``propose`` completion, which is far larger than the shortfall, so the
        check stays conservative without having to re-render every prompt variant to measure it.
        """
        backend = self.backend()
        counter = getattr(backend, "count_tokens", None)
        if counter is None:
            # An injected backend (the tests, or a caller supplying their own) has no tokenizer to
            # measure against. Not being able to measure is a reason to skip the guard, never a
            # reason to skip the document -- so it fits by default.
            return True, 0
        n_tokens = counter(rendered_document)
        return n_tokens <= backend.prompt_budget, n_tokens

    def _signature(self, batch) -> list:
        """Stage 0, batched: fold each newly defended document into its author's rolling profile.

        ``batch`` is ``[(profile, defended_document), ...]``; returns the updated profiles in order.
        One call for every author advancing a cascade step, not one per author.
        """
        from ._backends import extract_tagged_output

        batch = list(batch)
        if not batch:
            return []
        # Both fields are clipped: an unbounded profile plus an unbounded document is exactly the
        # pair that overflowed the window and hard-failed a whole chunk.
        users = [render_template(self.prompts["signature_user_template"],
                                 {"PROFILE": clip_context(profile or "(empty)", CONTEXT_CHARS // 4),
                                  "DOCUMENT": clip_context(document)})
                 for profile, document in batch]
        replies = self.backend().chat(self.prompts["signature_system_prompt"], users, "signature")
        updated = []
        for (profile, _document), reply in zip(batch, replies):
            extracted = extract_tagged_output(reply, "profile")
            # An unusable reply keeps the previous profile rather than clearing it: a stale profile
            # is weaker guidance, an empty one is none at all.
            updated.append(extracted.strip() if extracted and extracted.strip() else profile)
        return updated

    def _gate(self, pairs) -> list:
        """Stage 3, batched: utility loss in [0, 1] per variant, judged on the ANSWER.

        ``pairs`` is ``[(original_document, [variant, ...]), ...]``; returns one loss list per pair.
        **Two model calls for the whole batch** -- one answer pass, one judge pass -- where the
        per-document version cost four generations each. On a corpus this is the difference between
        the gate being a rounding error and being a third of the run.

        Both sides are judged, so a reference answer that was itself poor is not charged to the
        rewrite. An unreadable verdict is the maximum loss rather than zero: a candidate whose cost
        cannot be established must not become the cheapest one to accept. Identical in construction
        to ``loo_unlink._utility`` so the two defenses' utility numbers can be read against each
        other.
        """
        pairs = [(original, list(variants)) for original, variants in pairs]
        if not pairs:
            return []

        # One flat answer batch: each document contributes its original plus each of its variants,
        # and `spans` remembers which replies belong to which document.
        # The answer prompts are bounded too. A document that passed `_fits` sits just under the
        # window on its own, so appending a completion to it can still overflow.
        prompts, spans = [], []
        for original, variants in pairs:
            start = len(prompts)
            prompts.append(clip_context(original, CONTEXT_CHARS * 2))
            prompts.extend(clip_context(variant, CONTEXT_CHARS * 2) for variant in variants)
            spans.append((start, len(variants)))
        answers = self.backend().chat(self.prompts["answer_system_prompt"], prompts, "answer")
        if len(answers) != len(prompts):
            return [[1.0] * len(variants) for _original, variants in pairs]

        # One flat judge batch over every answer, each against ITS OWN original request.
        judge_users = []
        for (original, _variants), (start, count) in zip(pairs, spans):
            for offset in range(count + 1):
                judge_users.append(render_template(
                    self.prompts["judge_user_template"],
                    {"REQUEST": clip_context(original),
                     "ANSWER": clip_context(answers[start + offset], CONTEXT_CHARS // 2)}))
        judged = self.backend().chat(self.prompts["judge_system_prompt"], judge_users, "judge")
        if len(judged) != len(judge_users):
            return [[1.0] * len(variants) for _original, variants in pairs]

        losses, cursor = [], 0
        for _original, variants in pairs:
            reference_score = parse_judge_score(judged[cursor])
            if reference_score is None:
                reference_score = 5.0
            per_variant = []
            for offset in range(1, len(variants) + 1):
                score = parse_judge_score(judged[cursor + offset])
                if score is None:
                    per_variant.append(1.0)
                    continue
                # Normalized by the 4-point span of the 1-5 scale, clamped at 0: a rewrite that
                # somehow improves the answer is free, not negative-cost.
                per_variant.append(max(0.0, (reference_score - score) / 4.0))
            losses.append(per_variant)
            cursor += len(variants) + 1
        return losses

    # -- the reference pool under sharding --

    def reference_pool_ids(self, doc_ids) -> list[str]:
        """Which documents form the reference pool, chosen over the WHOLE split's ``doc_id``s.

        Exists for :mod:`~prompt_anonymity.data.apply_defenses`. An author-sharded task only holds
        its own authors' rows, and the pool drawn from those would be a *different* pool per shard
        -- which would be quietly fatal, because the pool defines ``r_med`` and therefore the target
        every document is optimized to (``r_med + alpha * (s_max_0 - r_med)``). Each shard would
        succeed against its own criterion and the arms would not be comparable.

        So the caller selects here, over the full frame, reads those documents' turns, and passes
        them on the *known* side; :meth:`_pool_from_known` picks them up. Because
        :func:`select_reference_pool` samples over **sorted** ``doc_id``s with a fixed seed, the
        result is identical to what an unsharded run picks for itself -- so ``pool_digest``, which
        is part of every author's cache key, matches across shard layouts. The selftest asserts it
        rather than trusting it.
        """
        doc_ids = [str(doc) for doc in doc_ids]
        chosen = select_reference_pool(doc_ids, n_reference=self.n_reference, seed=self.seed)
        return [doc_ids[index] for index in chosen]

    def _pool_from_known(self, data: AttackData):
        """``(ids, texts, authors)`` of a caller-supplied pool, or ``None`` if there is not one.

        The known side arrives as the same per-TURN stream as the unknown side, so the documents are
        rebuilt and joined with :func:`join_document` here -- byte-identical to how the unsharded
        path builds ``document_text``, which is what keeps the digest stable. Sorted by ``doc_id``
        for the same reason: :func:`select_reference_pool` returns its choice in that order, and the
        rows may arrive in split order instead.
        """
        if data.known_texts is None or len(data.known_texts) == 0:
            return None
        if data.known_ids is None:
            raise ValueError(f"defense {self.name!r}: a known-side reference pool needs known_ids.")
        labels = ([str(label) for label in data.known_labels]
                  if data.known_labels is not None and len(data.known_labels)
                  else [""] * len(data.known_texts))
        turns: dict[str, list[str]] = {}
        author: dict[str, str] = {}
        for row_id, text, label in zip(data.known_ids, data.known_texts, labels):
            doc = document_id(str(row_id))
            turns.setdefault(doc, []).append(str(text))
            author.setdefault(doc, str(label))
        pool_ids = sorted(turns)
        return pool_ids, [join_document(turns[doc]) for doc in pool_ids], \
            [author[doc] for doc in pool_ids]

    # -- edit log --

    def _open_log(self, cache_dir) -> None:
        """Open the per-alpha edit log in append mode.

        Append, not truncate: a preempted job resumes and only recomputes what it lost, so the lines
        already written describe authors still in the cache. An author recomputed after a cache
        invalidation appears twice, newest last -- readers should keep the last line per ``doc_id``.
        """
        directory = Path(self.log_dir) if self.log_dir else Path(cache_dir) / "afr_logs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"edits_a{int(round(self.alpha * 100)):02d}.jsonl"
        print(f"[{self.name}] edit log -> {path}")
        self._log_handle = open(path, "a", encoding="utf-8")

    def _write_trace(self, trace: DocumentTrace) -> None:
        if self._log_handle is None:
            return
        record = asdict(trace)
        record["rounds"] = [asdict(item) if not isinstance(item, dict) else item
                            for item in trace.rounds]
        self._log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_handle.flush()   # a preempted job should not lose the lines it already produced

    # -- the CachedDefense contract --

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        """Defend every author's timeline, then return the defended text one row per input TURN.

        ``apply_defenses`` hands this defense the split exploded into turns -- ``unknown_texts`` are
        turns, ``unknown_ids`` are ``<doc_id>#<n>``, ``unknown_labels`` are author ids. That is
        enough to rebuild documents and group them by author in arrival order, which
        ``build_dataset.finalize``'s sort makes chronological. Using the author's own earlier prompts
        is legitimate here: a person defending their own prompts knows what they previously wrote.
        """
        from dataclasses import replace as dataclass_replace

        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts.")
        ids = ([str(i) for i in data.unknown_ids] if data.unknown_ids is not None
               else [str(i) for i in range(len(data.unknown_texts))])
        texts = [str(text) for text in data.unknown_texts]
        authors = [str(label) for label in data.unknown_labels]

        # Turn stream -> documents, preserving first-appearance (= chronological) order.
        order: list[str] = []
        doc_turns: dict[str, list[str]] = {}
        doc_author: dict[str, str] = {}
        positions: dict[str, list[int]] = {}
        for position, (row_id, text, author) in enumerate(zip(ids, texts, authors)):
            doc = document_id(row_id)
            if doc not in doc_turns:
                order.append(doc)
                doc_turns[doc], positions[doc] = [], []
                doc_author[doc] = author
            doc_turns[doc].append(text)
            positions[doc].append(position)

        by_author: dict[str, list[str]] = {}
        for doc in order:
            by_author.setdefault(doc_author[doc], []).append(doc)

        document_text = {doc: join_document(doc_turns[doc]) for doc in order}

        # --- the reference pool: "a median unrelated document" -----------------
        # From the KNOWN side when the caller supplied one, which is how an author-sharded run keeps
        # every shard calibrated against the same "unrelated". See `reference_pool_ids`.
        supplied = self._pool_from_known(data)
        if supplied is not None:
            pool_ids, pool_texts, pool_authors = supplied
        else:
            pool_positions = select_reference_pool(order, n_reference=self.n_reference,
                                                   seed=self.seed)
            pool_ids = [order[index] for index in pool_positions]
            pool_texts = [document_text[doc] for doc in pool_ids]
            pool_authors = [doc_author[doc] for doc in pool_ids]
        digest = pool_digest(pool_ids, pool_texts)

        # Authors in SORTED order, one at a time -- see the determinism note.
        author_order = sorted(by_author)
        sources = [encode_author_source(digest, [(doc, doc_turns[doc]) for doc in by_author[author]])
                   for author in author_order]
        source_author = {}
        for author, source in zip(author_order, sources):
            source_author.setdefault(source, author)

        print(f"[{self.name}] {len(order):,} documents / {len(author_order):,} authors, "
              f"alpha={self.alpha:.2f}, probes<={self.max_probes}, "
              f"reference pool {len(pool_ids):,}")

        # Resolve BOTH checkpoints before anything expensive starts. Without this the first failure
        # of a mis-pointed Harrier is a bare OSError from `from_pretrained`, raised deep inside
        # `compute` -- after the log is opened, which is what makes the symptom "an empty
        # edits_a<NN>.jsonl and no explanation" rather than a message naming the variable to set.
        # Skipped when a backend was injected (the tests), which has no checkpoints to resolve.
        if self._backend is None:
            self._resolve_checkpoints()

        self._open_log(cache.dir)

        # Per-DOCUMENT durability, beside the per-author table. The author table is only written
        # when a whole chunk finishes, so on a preemptible job a killed task could commit nothing at
        # all. This makes every finished document survive independently, so a resumed run pays only
        # for what it had not already done. It lives under the cache directory, so it is scoped by
        # logic_hash and params_hash exactly like the table and is discarded by the same invalidation.
        documents_store = DocumentStore(Path(cache.dir) / "afr_docs")

        # The pool is embedded ONCE for the whole run and then sliced per author, rather than once
        # per author for vectors that would be identical every time.
        pool_matrix: list = [None]

        def pool_for(author: str) -> np.ndarray:
            """The reference pool with this author's own documents removed.

            An author whose own documents were in the pool would be calibrating "unrelated" partly
            against themselves, which biases the target toward being trivially met.
            """
            if pool_matrix[0] is None:
                pool_matrix[0] = (self.backend().embed(pool_texts) if pool_texts
                                  else np.zeros((0, 0)))
            keep = [index for index, owner in enumerate(pool_authors) if owner != author]
            if not keep or pool_matrix[0].size == 0:
                return np.zeros((0, 0))
            return pool_matrix[0][np.asarray(keep, dtype=int)]

        def compute(missing: list[str]) -> list[str]:
            """Defend a whole chunk of authors AT ONCE, in lockstep.

            This is where the throughput lives. ``cache.apply`` hands over
            ``checkpoint_every`` authors, and every one of them is stepped through its timeline
            together so each stage is a single batched model call across the chunk -- rather than
            the author-at-a-time loop this used to be, which produced batch-1 calls and could not
            finish. ``checkpoint_every`` is therefore both the flush cadence AND the batch width,
            exactly as it is for the sibling defenses (see ``_backends`` and the module docstring).
            """
            authors = [source_author[source] for source in missing]
            timelines = [(author, decode_author_source(source)[1])
                         for author, source in zip(authors, missing)]
            documents = sum(len(entries) for _author, entries in timelines)
            print(f"[{self.name}] chunk: {len(timelines):,} authors / {documents:,} documents "
                  f"(cascade depth {AFR_CASCADE_DEPTH})", flush=True)

            def record(author, doc_id, position, trace) -> None:
                trace.author_id = str(author)
                self._write_trace(trace)   # per DOCUMENT: the progress signal, flushed immediately

            defended = defend_authors(
                timelines, pool_for,
                embed=self.backend().embed, abstract=self._abstract, propose=self._propose,
                gate=self._gate, signature=self._signature,
                alpha=self.alpha, max_probes=self.max_probes,
                escalate_after=self.escalate_after, top_m=self.top_m,
                max_utility_loss=self.max_utility_loss,
                first_round_candidates=self.first_round_candidates,
                cascade_depth=AFR_CASCADE_DEPTH, fits=self._fits, on_document=record,
                store=documents_store, pool_key=digest)
            if documents_store is not None:
                print(f"[{self.name}] chunk done: {documents_store.hits:,} documents restored from "
                      f"the document store, {documents_store.writes:,} newly written", flush=True)
            return [encode_author_output(defended[author]) for author in authors]

        try:
            defended_authors = cache.apply("unknown", sources, compute, ids=author_order,
                                           checkpoint_every=self.checkpoint_every)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            if self._backend is not None:
                self._backend.close()

        # Authors -> the per-turn stream apply_defenses expects, back in input order.
        rebuilt = list(texts)
        for author, encoded in zip(author_order, defended_authors):
            documents = decode_author_output(encoded)
            author_docs = by_author[author]
            if len(documents) != len(author_docs):
                raise ValueError(
                    f"{self.name}: author {author} came back with {len(documents)} documents but "
                    f"was given {len(author_docs)}; the cascade must preserve the timeline."
                )
            for doc, defended_turns in zip(author_docs, documents):
                if len(defended_turns) != len(positions[doc]):
                    raise ValueError(
                        f"{self.name}: document {doc} came back with {len(defended_turns)} turns "
                        f"but was given {len(positions[doc])}; turn boundaries must survive the "
                        f"rewrite."
                    )
                for position, turn in zip(positions[doc], defended_turns):
                    rebuilt[position] = turn

        return dataclass_replace(data, unknown_texts=np.asarray(rebuilt, dtype=object))


#: The residual-linkage levels registered as their own defenses, mirroring ``LOO_UNLINK_BUDGETS``.
#: Each writes its own parquet and its own results directory, and each caches separately (alpha is in
#: ``params()``), which is what turns "a curve, not a point" into ordinary registry entries.
#: Keep in sync with experiments/run_afr.sbatch's ALPHAS.
AFR_RESIDUALS = (0.0, 0.25, 0.5)


# --- selftest ----------------------------------------------------------------

def _fake_backend(identifying=("Reykjavik", "llama")):
    """Deterministic stand-ins for the embedder, the agent and the judge.

    The embedder puts the identifying tokens on their own axes and everything else on a **crc32**
    axis -- not :func:`hash`, which is salted per process and would make this selftest produce
    different results on every run (see the module docstring's determinism note, and
    ``loo_unlink._fake_backend``, which has exactly that problem).

    The fake agent honours the escalation ladder: at level 0 it generalizes only the first
    identifying token, at level 1 both, at level 2 it also drops non-essential words. That is enough
    structure for the ladder, the stall detector and the utility gate's walk-back to be exercised.
    """
    dimensions = 32

    def embed(texts):
        matrix = np.zeros((len(texts), dimensions))
        for row, text in enumerate(texts):
            for axis, token in enumerate(identifying):
                matrix[row, axis] = 3.0 * text.count(token)
            for word in text.split():
                cleaned = word.lower().strip(".,!?").encode("utf-8", "replace")
                matrix[row, 2 + zlib.crc32(cleaned) % (dimensions - 2)] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def _generalize(turns, level):
        out = []
        for turn in turns:
            text = turn
            for position, token in enumerate(identifying):
                if level >= 1 or position == 0:
                    text = text.replace(token, "a small business")
            if level >= 2:
                text = " ".join(word for word in text.split() if len(word) > 3)
            out.append(text)
        return out

    def abstract(turns):
        return _generalize(turns, 0)

    def propose(payload):
        level = int(payload["level"])
        want = int(payload["want"])
        base = list(payload["turns"])
        candidates = [_generalize(base, level)]
        # A second, slightly different approach so a multi-candidate round has something to choose
        # between; a third that is deliberately inadmissible (a deletion), to exercise the filter.
        if want > 1:
            candidates.append(_generalize(base, level + 1))
        if want > 2:
            candidates.append(["" for _ in base])
        return candidates[:want]

    def signature(profile, document):
        return (profile + "\n- mentions " + document.split()[0]).strip() if document else profile

    def gate(original, variants):
        # Cost proportional to how much text the variant dropped: additive, monotone, and enough
        # structure for the walk-back down the ladder to be exercised.
        return [min(1.0, max(0.0, (len(original) - len(variant)) / max(len(original), 1) * 2.0))
                for variant in variants]

    return embed, abstract, propose, signature, gate


def _selftest() -> None:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("constants agree with the rest of the pipeline:")
    from ..data.apply_defenses import TURN_ID_SEPARATOR as PIPELINE_TURN_ID
    from ..data.compute_features import TURN_SEPARATOR as PIPELINE_SEPARATOR
    check(TURN_SEPARATOR == PIPELINE_SEPARATOR,
          "TURN_SEPARATOR matches compute_features (the objective scores what the attacker embeds)")
    check(TURN_ID_SEPARATOR == PIPELINE_TURN_ID, "TURN_ID_SEPARATOR matches apply_defenses")

    print("prompts:")
    prompts = load_prompts()
    check(len(prompts) == len(REQUIRED_PROMPT_KEYS) + 1,
          f"every prompt key loads (got {len(prompts)})")
    check(len(prompts["escalation_levels"]) == N_ESCALATION_LEVELS,
          "the escalation ladder has exactly three rungs")
    check(prompts_digest(prompts) == prompts_digest(load_prompts()),
          "the prompt digest is stable across loads")

    print("turn parsing:")
    check(parse_turns('<turn index="1">a</turn><turn index="2">b</turn>', 2) == ["a", "b"],
          "reads two well-formed turns")
    check(parse_turns('<turn index="1">a</turn>', 2) is None, "too few turns -> None")
    check(parse_turns('<turn index="2">b</turn><turn index="1">a</turn>', 2) is None,
          "out-of-order indices -> None, never a silently transposed conversation")
    check(len(parse_candidates('<candidate id="1"><turn index="1">a</turn></candidate>'
                               '<candidate id="2"><turn index="1">b</turn></candidate>', 1, 3)) == 2,
          "reads two candidates from one reply")
    check(parse_candidates('<turn index="1">a</turn>', 1, 1) == [["a"]],
          "an unwrapped single candidate is still read")

    print("admissibility:")
    check(admissible(["hello there friend"], ["hello there friend"]), "an identical rewrite passes")
    check(not admissible([""], ["hello there friend"]), "a deletion is rejected")
    check(not admissible(["x"], ["hello there friend"]), "an over-short rewrite is rejected")
    check(not admissible(["a", "b"], ["a"]), "a changed turn count is rejected")

    print("cache source encoding:")
    encoded = encode_author_source("deadbeef", [("d1", ["a", "b"]), ("d2", ["c"])])
    digest, documents = decode_author_source(encoded)
    check(digest == "deadbeef" and documents == [("d1", ["a", "b"]), ("d2", ["c"])],
          "an author source round-trips")
    check(encode_author_source("a", [("d1", ["x"])]) != encode_author_source("b", [("d1", ["x"])]),
          "a different reference pool is a different cache entry")
    check(encode_author_source("a", [("d1", ["x"]), ("d2", ["y"])])
          != encode_author_source("a", [("d2", ["y"]), ("d1", ["x"])]),
          "a reordered timeline is a different cache entry (the cascade depends on order)")
    round_tripped = decode_author_output(encode_author_output([["a", "b"], ["c"]]))
    check(round_tripped == [["a", "b"], ["c"]], "an author's defended output round-trips")

    print("checkpoint resolution (prefer the cluster mirror over a download):")
    import tempfile

    from ._backends import model_path, shared_checkpoint
    with tempfile.TemporaryDirectory() as root:
        mirror = Path(root) / "qwen3" / "hub" / "models--Qwen--Qwen3-30B-A3B-Instruct-2507-FP8"
        mirror.mkdir(parents=True)
        check(shared_checkpoint("Qwen/Qwen3-30B-A3B-Instruct-2507-FP8", root) == str(mirror),
              "a mirrored repo id resolves to the checkpoint on disk")
        check(shared_checkpoint("Qwen/Not-Mirrored", root) is None,
              "an unmirrored repo id falls through to the download path")
        check(shared_checkpoint("/an/explicit/path", root) is None,
              "an absolute path is passed over, never rewritten")
        check(shared_checkpoint("bare-name", root) is None, "a bare name is passed over")
    check(shared_checkpoint("Qwen/Anything", "/nonexistent-root") is None,
          "a machine with no mirror root degrades rather than raising")
    configured = model_path("afr", "AFR_MODEL_UNSET_FOR_SELFTEST")
    check(Path(configured).is_absolute(),
          f"models.toml [afr] points at a checkpoint on disk, not a repo id that would download "
          f"({configured!r})")
    check(not AFR_ALLOW_DOWNLOAD,
          "downloading is OFF unless AFR_ALLOW_DOWNLOAD=1 is set explicitly")
    try:
        AgenticFootprintDefense(model="/definitely/not/here").agent_checkpoint()
        check(False, "a non-existent checkpoint is refused rather than downloaded")
    except SystemExit as error:
        check("DOWNLOAD" in str(error) and "AFR_ALLOW_DOWNLOAD" in str(error),
              "a non-existent checkpoint raises, naming the download risk and the opt-out")
    with tempfile.TemporaryDirectory() as real:
        check(AgenticFootprintDefense(model=real).agent_checkpoint() == real,
              "a checkpoint that IS on disk passes through untouched")

    # The embedder resolves the same way, and all three of its variants must reach a checkpoint --
    # they key on one [harrier] section because the A/B varies the instruction, not the weights.
    from ..features import get_featurizer
    sections = {name: get_featurizer(name).config_section
                for name in ("harrier", "harrier_imperative", "harrier_plain")}
    check(set(sections.values()) == {"harrier"},
          f"every Harrier variant reads the [harrier] section (got {sections})")
    check(all(get_featurizer(name).checkpoint() for name in sections),
          "...so none of them dies on a missing models.toml section")

    print("reference pool:")
    doc_ids = [f"d{index:03d}" for index in range(100)]
    first = select_reference_pool(doc_ids, n_reference=10, seed=7)
    shuffled = list(reversed(doc_ids))
    second = select_reference_pool(shuffled, n_reference=10, seed=7)
    check(first == select_reference_pool(doc_ids, n_reference=10, seed=7),
          "the same inputs draw the same pool")
    check(sorted(doc_ids[index] for index in first)
          == sorted(shuffled[index] for index in second),
          "row order does not move the sample (sorted before drawing)")
    check(select_reference_pool(doc_ids[:5], n_reference=10) == list(range(5)),
          "a pool smaller than the request is taken whole")

    print("scoring:")
    matrix = np.eye(4)
    check(objective_scores(matrix[0], matrix[:2], matrix, top_m=2)[0] == 1.0,
          "s_max finds an identical prior")
    check(objective_scores(matrix[0], np.zeros((0, 0)), matrix)[0] == 0.0,
          "no priors -> zero similarity, not a crash")
    check(abs(linkage_target(0.8, 0.4, 0.0) - 0.4) < 1e-9, "alpha=0 targets the unrelated baseline")
    check(abs(linkage_target(0.8, 0.4, 0.5) - 0.6) < 1e-9, "alpha=0.5 keeps half the excess")
    check(abs(genericness_target(0.2, 0.6, 0.0) - 0.6) < 1e-9, "cold start targets the pool median")

    print("defend_document:")
    embed, abstract, propose, signature, gate = _fake_backend()
    turns = ["I run a llama farm in Reykjavik. I need help with a Django inventory app. "
             "What models should I define?"]
    prior_texts = ["My llama farm in Reykjavik needs a stock tracker. Any advice on Django?"]
    priors = embed(prior_texts)
    pool = embed(["How do I center a div in CSS?",
                  "What is the difference between a list and a tuple in Python?",
                  "Recommend a book about the history of cartography.",
                  "My React build fails with an out of memory error."])

    winner, floor, history, trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract, propose=propose,
        alpha=0.0)
    check(len(winner.turns) == len(turns), "turn count is preserved")
    check(winner.objective <= floor.objective,
          f"the winner never scores worse than the stage-1 floor "
          f"({winner.objective:.4f} <= {floor.objective:.4f})")
    check(trace.probes_used <= DEFAULT_MAX_PROBES,
          f"the probe budget is respected ({trace.probes_used} <= {DEFAULT_MAX_PROBES})")
    check(trace.stop_reason in STOP_REASONS, f"stop_reason is known ({trace.stop_reason!r})")
    check(all(record.probes_used == record.admissible for record in trace.rounds),
          "an inadmissible candidate costs a generation but never a probe")

    stage1_only, stage1_floor, _, stage1_trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract, propose=propose,
        alpha=0.0, max_probes=0)
    check(stage1_trace.probes_used == 0 and stage1_trace.stop_reason in ("no_probes",
                                                                        "stage1_sufficient"),
          "max_probes=0 runs the abstraction pass and stops")
    check(stage1_only.turns == stage1_floor.turns,
          "...and emits exactly the stage-1 output (the afr_stage1 ablation)")

    cold, _, _, cold_trace = defend_document(
        turns, np.zeros((0, 0)), [], pool, embed=embed, abstract=abstract, propose=propose,
        alpha=0.0)
    check(cold_trace.cold_start and cold_trace.n_priors == 0,
          "an author's first prompt takes the cold-start path")
    check(len(cold.turns) == len(turns), "...and still preserves turn count")

    print("engine startup (the defects that killed the first cluster run):")
    import inspect as _inspect

    # EVERY vLLM-backed defense, not just this one. `qwen_rewrite` was missing the call and died in
    # `warmup_kernels` -> `worker_sample_tokens` when the FlashInfer sampler JIT ran on a node with
    # no CUDA toolkit -- a whole arm lost to a defect this check already covered for two modules.
    # Enumerated from the loaders themselves so a NEW vLLM defense is caught by the same net.
    from . import loo_unlink as _loo
    from . import openanonymity as _oa
    from . import qwen_rewrite as _qwen
    from . import styleremix as _sr

    loaders = [
        ("afr", _LocalBackend._engine),
        ("loo_unlink", _loo._LocalBackend._engine),
        ("openanonymity", _oa._OpenAnonBackend.__init__),
        ("qwen_rewrite", _qwen._QwenVLLMRewriter.__init__),
        ("styleremix", _sr.load_styleremix_model if hasattr(_sr, "load_styleremix_model") else None),
    ]
    for module, engine in [(name, fn) for name, fn in loaders if fn is not None]:
        source = _inspect.getsource(engine)
        if "from vllm import" not in source:
            continue        # not a vLLM loader (styleremix uses transformers/PEFT)

        toolkit = source.find("configure_cuda_toolkit()")
        vllm_import = source.find("from vllm import")
        check(toolkit != -1 and toolkit < vllm_import,
              f"{module}: configure_cuda_toolkit() runs BEFORE `from vllm import` "
              f"(vLLM reads its environment at import; without it the FlashInfer sampler JIT "
              f"can kill startup)")
    engine_source = _inspect.getsource(_LocalBackend._engine)
    check("max_model_len=" in engine_source,
          "afr pins max_model_len (the checkpoint's native 262k window does not fit the KV cache)")
    check(AFR_MAX_MODEL_LEN > DEFAULT_PROPOSE_TOKENS,
          f"the window leaves room for a reply ({AFR_MAX_MODEL_LEN} > {DEFAULT_PROPOSE_TOKENS})")
    knobs = AgenticFootprintDefense().params()
    check("max_model_len" not in knobs and "gpu_memory_utilization" not in knobs,
          "the serving knobs stay OUT of params(): they change where it runs, not what it emits")

    print("oversized documents:")
    long_turns = ["word " * 5000]

    # BY DEFAULT an over-long document stops the run. Emitting it undefended would leave untouched
    # text in the defended split, where it keeps its full authorship signal and is charged to the
    # defense that never saw it. The fix is a corpus-level cap (build_subset --max-chars), which
    # keeps every arm on the same text; the error says so.
    refused = None
    try:
        defend_document(long_turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
                        propose=propose, alpha=0.0, fits=lambda text: (False, 99_999))
    except SystemExit as error:
        refused = str(error)
    check(refused is not None, "an over-long document STOPS the run rather than passing through")
    check(refused is not None and "build_subset --max-chars" in refused,
          "...and the error names the corpus-level fix, not just the failure")

    # The old behaviour stays reachable, explicitly, for exploratory runs.
    global AFR_TOO_LONG_PASSTHROUGH
    AFR_TOO_LONG_PASSTHROUGH = True
    try:
        over, over_floor, over_history, over_trace = defend_document(
            long_turns, priors, prior_texts, pool, embed=embed, abstract=abstract, propose=propose,
            alpha=0.0, fits=lambda text: (False, 99_999))
    finally:
        AFR_TOO_LONG_PASSTHROUGH = False
    check(over_trace.stop_reason == "too_long",
          "with AFR_TOO_LONG_PASSTHROUGH=1 it stops at 'too_long' instead")
    check(over.turns == long_turns and over_trace.defended == over_trace.original,
          "...and is emitted UNDEFENDED rather than truncated")
    check(over_trace.prompt_tokens == 99_999, "...with its measured token count recorded")
    check(over is over_floor and over_history == [over],
          "...and the gate has nothing to walk back to, so it cannot be 'improved' into a crash")
    fitted, _, _, fitted_trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract, propose=propose,
        alpha=0.0, fits=lambda text: (True, 123))
    check(fitted_trace.stop_reason != "too_long" and fitted_trace.prompt_tokens == 123,
          "a document that fits is defended normally and still records its length")
    check("too_long" in STOP_REASONS, "'too_long' is a known stop reason, so the report counts it")

    class _NoTokenizer:
        prompt_budget = 10
    unmeasurable = AgenticFootprintDefense(backend=_NoTokenizer())
    check(unmeasurable._fits("anything at all") == (True, 0),
          "a backend with no tokenizer skips the guard rather than skipping every document")

    print("prompt window arithmetic:")
    budget = _LocalBackend({}, propose_tokens=DEFAULT_PROPOSE_TOKENS).prompt_budget
    check(budget == AFR_MAX_MODEL_LEN - DEFAULT_PROPOSE_TOKENS - PROMPT_RESERVE_TOKENS,
          f"the document budget reserves the completion AND the wrapper around it ({budget:,} of "
          f"{AFR_MAX_MODEL_LEN:,})")
    check(budget + DEFAULT_PROPOSE_TOKENS < AFR_MAX_MODEL_LEN,
          "...so a document at the budget plus its reply still fits the window")
    long_context = "word " * 20_000
    check(len(clip_context(long_context)) < len(long_context),
          "a long document is clipped where it is CONTEXT (profile input, judge request)")
    check(clip_context("short") == "short", "a short one is untouched")
    check(clip_context(long_context).endswith("[...truncated]"),
          "...and the cut is marked, so the model is not shown a sentence that just stops")
    # The rewrite paths must NEVER clip -- a truncated document there enters the dataset.
    rewrite_source = _inspect.getsource(AgenticFootprintDefense._abstract) + \
        _inspect.getsource(AgenticFootprintDefense._propose)
    check("clip_context" not in rewrite_source,
          "the abstract and propose paths never clip: over-long documents take the too_long exit")

    print("the escalation ladder:")

    def stubborn_propose(payload):
        # Never improves: the exact draft it was given, returned unchanged.
        return [list(payload["turns"])] * int(payload["want"])

    # max_probes is explicit because this exercises the LADDER, not the budget: walking all three
    # rungs at escalate_after=2 needs six rounds, and the first spends
    # DEFAULT_FIRST_ROUND_CANDIDATES probes at once, so the default budget of 4 runs out first and
    # the document would stop at 'probes_exhausted' without ever reaching the top.
    _, _, _, stalled_trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
        propose=stubborn_propose, alpha=0.0, escalate_after=2, max_probes=10)
    # THE BUDGET MUST BE ABLE TO WALK THE LADDER. This is the check that would have caught cutting
    # max_probes 10 -> 4 while leaving first_round_candidates at 3: the loop then spent every probe
    # at level 0 and the structural rung was unreachable, quietly removing the defense's most
    # aggressive mode. Run at the REAL defaults, so any future change to any of the three constants
    # has to keep them consistent with each other.
    _, _, _, budget_trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
        propose=stubborn_propose, alpha=0.0)
    check(budget_trace.max_level == N_ESCALATION_LEVELS - 1,
          f"the DEFAULT probe budget reaches the top escalation rung "
          f"(reached {budget_trace.max_level}, top is {N_ESCALATION_LEVELS - 1}; "
          f"max_probes={DEFAULT_MAX_PROBES}, first_round={DEFAULT_FIRST_ROUND_CANDIDATES}, "
          f"escalate_after={DEFAULT_ESCALATE_AFTER})")
    check(budget_trace.stop_reason == "ladder_exhausted",
          f"...and exhausts the ladder rather than running out of probes first "
          f"({budget_trace.stop_reason!r})")
    check(budget_trace.probes_used <= DEFAULT_MAX_PROBES,
          f"...without exceeding the budget ({budget_trace.probes_used} <= {DEFAULT_MAX_PROBES})")

    check(stalled_trace.stop_reason == "ladder_exhausted",
          f"a stalled agent climbs the ladder and stops at the top "
          f"({stalled_trace.stop_reason!r})")
    check(stalled_trace.n_rounds <= N_ESCALATION_LEVELS * 2 + 1,
          f"...after escalate_after rounds per rung, not the whole budget "
          f"({stalled_trace.n_rounds} rounds)")

    levels_seen = []

    def recording_propose(payload):
        levels_seen.append(int(payload["level"]))
        return [list(payload["turns"])]

    defend_document(turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
                    propose=recording_propose, alpha=0.0, escalate_after=1)
    check(levels_seen[:3] == [0, 1, 2],
          f"escalate_after=1 walks every rung in order (saw {levels_seen[:3]})")

    print("the utility gate:")
    generous_trace = DocumentTrace()
    kept = gate_and_select(winner, floor, history, join_document(turns),
                           gate=lambda original, variants: [0.0] * len(variants),
                           max_utility_loss=0.5, trace=generous_trace)
    check(kept.turns == winner.turns and generous_trace.outcome in ("loop_winner", "stage1_floor"),
          "a costless rewrite is kept")
    strict_trace = DocumentTrace()
    rolled_back = gate_and_select(winner, floor, history, join_document(turns),
                                  gate=lambda original, variants: [1.0] * len(variants),
                                  max_utility_loss=0.5, trace=strict_trace)
    check(rolled_back.turns == floor.turns and strict_trace.outcome == "stage1_floor",
          "a rewrite that destroys the answer falls back to the stage-1 floor")

    print("batched core (the throughput fix):")
    widths: list = []

    def counting_abstract(batch):
        widths.append(len(batch))
        return [abstract(item) for item in batch]

    def counting_propose(batch):
        widths.append(len(batch))
        return [propose(item) for item in batch]

    many = [{"turns": turns, "priors": priors, "prior_texts": prior_texts, "pool": pool}
            for _ in range(8)]
    batched = defend_documents(many, embed=embed, abstract=counting_abstract,
                               propose=counting_propose, alpha=0.0)
    check(len(batched) == 8, "defend_documents returns one result per input document")
    check(max(widths) == 8,
          f"...and advances all of them together (max batch width {max(widths)} of 8)")
    check(all(result[3].probes_used <= DEFAULT_MAX_PROBES for result in batched),
          "every document respects its own probe budget")

    solo = defend_document(turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
                           propose=propose, alpha=0.0)
    check(solo[0].turns == batched[0][0].turns,
          "the single-document wrapper and the batched core agree on the same input")
    check(abs(solo[3].final_objective - batched[0][3].final_objective) < 1e-12,
          "...including the score, so the two paths cannot drift apart")

    print("the cascade cap:")
    long_timeline = [(f"d{i}", [f"I run a llama farm in Reykjavik, note {i}."]) for i in range(6)]
    capped, capped_traces = defend_author(long_timeline, pool, embed=embed, abstract=abstract,
                                          propose=propose, gate=gate, signature=signature,
                                          alpha=0.0, cascade_depth=2)
    check(len(capped) == 6, "a capped cascade still returns every document")
    check([t.n_priors for t in capped_traces][:3] == [0, 1, 2],
          f"priors accumulate up to the cap ({[t.n_priors for t in capped_traces]})")
    check(all(t.n_priors == 2 for t in capped_traces[2:]),
          "...and freeze there, so documents past the cap are mutually independent")
    uncapped, uncapped_traces = defend_author(long_timeline, pool, embed=embed, abstract=abstract,
                                              propose=propose, gate=gate, signature=signature,
                                              alpha=0.0, cascade_depth=99)
    check([t.n_priors for t in uncapped_traces] == [0, 1, 2, 3, 4, 5],
          "an uncapped cascade keeps extending the chain")

    print("per-document resume (the store):")
    import tempfile

    with tempfile.TemporaryDirectory() as temporary:
        two_authors = [("a", [(f"a{i}", [f"I run a llama farm in Reykjavik, note {i}."])
                              for i in range(4)]),
                       ("b", [(f"b{i}", [f"Our team of 4 at Acme Robotics, item {i}."])
                              for i in range(3)])]
        calls: list = []

        # defend_authors takes the BATCHED callables (see defend_author for the same adapters).
        def counted_abstract(batch):
            batch = list(batch)
            calls.append(len(batch))
            return [abstract(item) for item in batch]

        def batched_propose(batch):
            batch = list(batch)
            calls.append(len(batch))
            return [propose(item) for item in batch]

        def batched_gate(pairs):
            return [gate(original, variants) for original, variants in pairs]

        def batched_signature(pairs):
            calls.append(len(list(pairs)))
            return [signature(profile, text) for profile, text in pairs]

        def cascade(timelines, store, key="pooldigest"):
            return defend_authors(timelines, lambda _author: pool, embed=embed,
                                  abstract=counted_abstract, propose=batched_propose,
                                  gate=batched_gate, signature=batched_signature,
                                  alpha=0.0, cascade_depth=2, store=store, pool_key=key)

        store = DocumentStore(Path(temporary) / "docs")
        cold = cascade(two_authors, store)
        cold_calls = sum(calls)
        # Distinct FILES, not writes: each document is now written twice -- once the instant it
        # finishes (profile null) and once when the position's profile is known. Same key, so the
        # second replaces the first.
        stored = len(list((Path(temporary) / "docs").glob("*.json")))
        check(cold_calls > 0 and stored == 7,
              f"a cold run computes every document and stores all 7 ({stored})")

        calls.clear()
        warm_store = DocumentStore(Path(temporary) / "docs")
        warm = cascade(two_authors, warm_store)
        check(sum(calls) == 0,
              f"a resumed run makes NO generation calls at all ({sum(calls)} calls)")
        check(warm_store.hits == 7, f"...because all 7 documents were restored ({warm_store.hits})")
        check(warm == cold, "and it reproduces the cold run's output exactly")

        # The point of folding the prior chain into the key: an edit upstream must invalidate
        # everything downstream of it, not just the document that changed.
        edited = [("a", [("a0", ["Something else entirely."])] + two_authors[0][1][1:]),
                  two_authors[1]]
        calls.clear()
        edited_store = DocumentStore(Path(temporary) / "docs")
        cascade(edited, edited_store)
        check(edited_store.hits == 3,
              f"editing author a's first document invalidates a's whole chain, and only b's 3 "
              f"documents still hit ({edited_store.hits})")

        # A different reference pool means a different target, so nothing may be reused.
        other_store = DocumentStore(Path(temporary) / "docs")
        cascade(two_authors, other_store, key="a different pool")
        check(other_store.hits == 0, "a different reference pool shares nothing")

        # The resume checks above all resume a run that FINISHED, which is exactly the case where
        # everything happens to be committed -- they'd pass even if the store only committed once
        # per lockstep POSITION, in which case a job preempted mid-position would save nothing at
        # all. This kills the run partway through a position instead, to catch that.
        class _Preempted(Exception):
            pass

        class _DyingStore(DocumentStore):
            def __init__(self, directory, die_after):
                super().__init__(directory)
                self.die_after, self.n = die_after, 0

            def put(self, key, turns, profile):
                super().put(key, turns, profile)
                self.n += 1
                if self.n >= self.die_after:
                    raise _Preempted()

        killed_dir = Path(temporary) / "killed"
        try:
            cascade(two_authors, _DyingStore(killed_dir, die_after=2))
        except _Preempted:
            pass
        # Position 0 holds one document per author (2 here), so dying on the 2nd commit means both
        # of that position's documents are already durable -- with the OLD per-position commit,
        # nothing at all would have been.
        survived = len(list(killed_dir.glob("*.json")))
        check(survived == 2,
              f"a run killed mid-position leaves its finished documents on disk ({survived})")

        calls.clear()
        after_kill = DocumentStore(killed_dir)
        recovered = cascade(two_authors, after_kill)
        check(after_kill.hits == survived,
              f"...and the resumed run reuses every one of them ({after_kill.hits} hits)")
        check(recovered == cold,
              "...and still reproduces the uninterrupted run's output exactly")
        check(sum(calls) < cold_calls,
              f"...having done strictly less work ({sum(calls)} vs {cold_calls} calls)")

        # A truncated record (a task killed mid-write) must read as absent, never as a hit.
        victim = next(iter((Path(temporary) / "docs").glob("*.json")))
        victim.write_text('{"turns": [', encoding="utf-8")
        check(DocumentStore(Path(temporary) / "docs").get(victim.stem) is None,
              "a half-written record is treated as missing rather than trusted")

    print("the batched utility gate:")
    gate_calls: list = []

    def counting_gate(pairs):
        pairs = list(pairs)
        gate_calls.append(len(pairs))
        return [[0.0] * len(variants) for _original, variants in pairs]

    results = defend_documents(many, embed=embed, abstract=counting_abstract,
                               propose=counting_propose, alpha=0.0)
    picked = gate_and_select_batch(results, gate=counting_gate, max_utility_loss=0.5)
    check(len(picked) == len(results), "the gate returns one choice per document")
    check(len(gate_calls) <= 2,
          f"...in at most two batched passes, not one per document ({len(gate_calls)})")
    strict_results = defend_documents(many, embed=embed, abstract=counting_abstract,
                                      propose=counting_propose, alpha=0.0)
    rolled = gate_and_select_batch(strict_results,
                                   gate=lambda pairs: [[1.0] * len(v) for _o, v in pairs],
                                   max_utility_loss=0.5)
    check(all(choice.turns == result[1].turns
              for choice, result in zip(rolled, strict_results)),
          "a batch the judge rejects falls back to each document's own stage-1 floor")

    print("defend_author (the cascade):")
    documents = [("d1", ["I run a llama farm in Reykjavik. What Django models do I need?"]),
                 ("d2", ["My llama farm in Reykjavik needs a stock tracker. Advice on Django?"]),
                 ("d3", ["How should I model feed batches for the llama farm in Django?"])]
    defended, traces = defend_author(documents, pool, embed=embed, abstract=abstract,
                                     propose=propose, gate=gate, signature=signature, alpha=0.0)
    check(len(defended) == 3 and all(len(turns) == 1 for turns in defended),
          "every document comes back with its own turn count")
    check(traces[0].n_priors == 0 and traces[1].n_priors == 1 and traces[2].n_priors == 2,
          f"priors accumulate along the timeline "
          f"({[trace.n_priors for trace in traces]})")
    check(traces[0].cold_start and not traces[2].cold_start,
          "only the first document takes the cold-start path")

    # The cascade must score against DEFENDED priors, not the originals. Defending the same author
    # again but with the middle document's ORIGINAL text as the prior would give a different
    # baseline; the check below is that the prior actually used was the defended one.
    defended_second = join_document(defended[1])
    original_second = join_document(documents[1][1])
    if defended_second != original_second:
        prior_vector = embed([defended_second])[0]
        third_vector = embed([join_document(defended[2])])[0]
        against_defended = float(prior_vector @ third_vector)
        against_original = float(embed([original_second])[0] @ third_vector)
        check(abs(against_defended - against_original) > 1e-9,
              "defended and original priors are distinguishable, so the cascade is testable")

    print("determinism:")
    first_run, first_traces = defend_author(documents, pool, embed=embed, abstract=abstract,
                                            propose=propose, gate=gate, signature=signature,
                                            alpha=0.0)
    second_run, second_traces = defend_author(documents, pool, embed=embed, abstract=abstract,
                                              propose=propose, gate=gate, signature=signature,
                                              alpha=0.0)
    check(first_run == second_run, "the same author cascaded twice is byte-identical")
    check([trace.stop_reason for trace in first_traces]
          == [trace.stop_reason for trace in second_traces],
          "...and produces identical traces")
    check(embed(["Reykjavik llama"]).tolist() == embed(["Reykjavik llama"]).tolist(),
          "the fake embedder is stable within a process (crc32, not hash())")

    print("registry:")
    from . import DEFENSES
    sweep = sorted(name for name in DEFENSES if name.startswith("afr"))
    check(len(sweep) == len(AFR_RESIDUALS) + 2,
          f"the residual sweep and both controls are registered ({len(sweep)}: {sweep})")
    check(all(not DEFENSES[name].shardable for name in sweep),
          "every registered variant refuses DOCUMENT sharding")
    check(all(DEFENSES[name].shardable_by == "author" for name in sweep),
          "every registered variant opts in to AUTHOR sharding")

    print("author sharding:")
    from ..data.compute_features import select_author_shard
    import pandas as _pd
    frame = _pd.DataFrame({
        "doc_id": [f"d{index:03d}" for index in range(60)],
        # deliberately lopsided: one author with 25 documents, a tail of singletons
        "author_id": (["a"] * 25 + ["b"] * 12 + ["c"] * 8 + ["d"] * 5
                      + [f"e{index}" for index in range(10)]),
    })
    shards = [select_author_shard(frame, index, 4) for index in range(4)]
    check(sum(len(part) for part in shards) == len(frame),
          "the author shards partition the frame exactly (no row lost or duplicated)")
    check(sorted(sum((list(part["doc_id"]) for part in shards), [])) == sorted(frame["doc_id"]),
          "and they cover every doc_id once")
    owners = [set(part["author_id"]) for part in shards]
    check(all(not (left & right) for index, left in enumerate(owners)
              for right in owners[index + 1:]),
          "no author appears in two shards -- the cascade is never cut")
    check(max(len(part) for part in shards) - min(len(part) for part in shards) <= 25,
          "longest-processing-time placement keeps the shards within one big author of each other")
    check(list(select_author_shard(frame, 0, 1)["doc_id"]) == list(frame["doc_id"]),
          "num_shards=1 is the whole frame, unchanged")
    check(all(list(select_author_shard(frame, index, 4).index)
              == list(select_author_shard(frame, index, 4).index) for index in range(4)),
          "the assignment is deterministic")

    print("the reference pool survives sharding:")
    defense = DEFENSES["afr"]
    all_ids = [f"d{index:03d}" for index in range(300)]
    # What an UNSHARDED transform picks for itself, by the same call it makes internally...
    positions = select_reference_pool(all_ids, n_reference=defense.n_reference, seed=defense.seed)
    unsharded = [all_ids[index] for index in positions]
    # ...must equal what a sharded run is told to hand every task.
    check(defense.reference_pool_ids(all_ids) == unsharded,
          "reference_pool_ids reproduces the unsharded pool exactly (so pool_digest matches)")
    check(defense.reference_pool_ids(list(reversed(all_ids))) == unsharded,
          "and it does not depend on the order the split hands over its doc_ids")

    pool_turns = {doc: [f"{doc} turn one", f"{doc} turn two"] for doc in unsharded[:5]}
    known_ids, known_texts, known_authors = [], [], []
    for doc, turns_ in pool_turns.items():
        for index, turn in enumerate(turns_):
            known_ids.append(f"{doc}#{index}")
            known_texts.append(turn)
            known_authors.append("someone")
    supplied = defense._pool_from_known(AttackData(
        known_embeddings=np.zeros((len(known_texts), 0)),
        unknown_embeddings=np.zeros((0, 0)),
        known_labels=np.asarray(known_authors, dtype=object),
        unknown_labels=np.empty(0, dtype=object),
        known_texts=np.asarray(known_texts, dtype=object),
        known_ids=np.asarray(known_ids, dtype=object)))
    check(supplied is not None and supplied[0] == sorted(pool_turns),
          "a known-side pool is rebuilt into documents, sorted by doc_id")
    check(supplied[1] == [join_document(pool_turns[doc]) for doc in sorted(pool_turns)],
          "and its texts are joined exactly as the unsharded path joins them")
    check(defense._pool_from_known(AttackData(
        known_embeddings=np.zeros((0, 0)), unknown_embeddings=np.zeros((0, 0)),
        known_labels=np.empty(0, dtype=object),
        unknown_labels=np.empty(0, dtype=object))) is None,
          "no known side means the defense samples its own pool, as before")

    print("serving knobs stay out of the cache key:")
    keys = set(DEFENSES["afr"].params())
    check(not (keys & {"speculative", "spec_tokens", "enforce_eager", "prefix_caching",
                       "gpu_memory_utilization", "max_model_len", "cascade_depth"}),
          f"params() carries no serving knob ({sorted(keys)})")
    check(DEFENSES["afr_stage1"].max_probes == 0,
          "afr_stage1 is the loop ablation, not the model ablation")
    check(DEFENSES["afr"].params()["alpha"] == DEFENSES["afr_a00"].params()["alpha"],
          "afr and afr_a00 share a cache entry")

    print(f"\n{len(failures)} failure(s).")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--selftest", action="store_true",
                        help="run the loop against fake models (no GPU, no network)")
    arguments = parser.parse_args()
    if arguments.selftest:
        _selftest()
    else:
        parser.error("nothing to do; pass --selftest")
