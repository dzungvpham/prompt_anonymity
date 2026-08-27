r"""Agentic Footprint Reduction: the model optimizes against a real linkage measurement.

Every other defense in this package applies a *fixed* transformation and hopes it helps -- neutralize
style (``styleremix``, ``qwen_rewrite``), add per-word DP noise (``dp_mlm``), manufacture shared
quirks (``collision_seeding``), reframe the scene (``frame_shift``). None of them ever check whether
the edit worked. :mod:`.loo_unlink` is the first that measures, but the measurement drives a fixed
greedy rule and the model is demoted to a span rewriter that never sees a score.

AFR inverts that. **The model is the optimizer.** It is shown its draft's real similarity to the
author's earlier prompts, told what a stranger's prompt scores, and given up to
:data:`DEFAULT_MAX_PROBES` re-embeddings to close the gap. The surrounding Python only scores what it
returns and enforces the contracts.

The target is absolute, not relative
------------------------------------

``loo_unlink`` chases a fractional budget ("30% less similar than you were"), which is unanchored:
30% off a prompt that was never identifiable is wasted utility, and 30% off a glaring one is not
enough. AFR is done when the prompt is **no closer to the author's earlier prompts than a randomly
chosen unrelated prompt is** -- a statement about unlinkability rather than about how much text
changed. With ``alpha`` the sweep knob and ``d0`` the first-pass abstraction:

.. code-block:: text

    s_max(d)  = max cosine to the author's PRIOR prompts   (the criterion; the attack needs one match)
    s_top3(d) = mean of the 3 highest                      (what the model is shown; a smoother signal)
    r_med(d)  = median cosine to the reference pool        ("a median unrelated document")

    target    = r_med(d0) + alpha * max(0, s_max(d0) - r_med(d0))
    done when   s_max(d) <= target

``alpha=0`` is the full claim; ``alpha=0.5`` keeps half the excess linkage as a cheaper operating
point. An author's FIRST prompt has nothing to be unlinked from, so the objective flips to
*genericness* -- cosine to the reference pool's centroid, targeted at the pool's own median. A first
prompt that is a distinctive outlier is precisely what a later prompt gets recognized against.

The cascade, and why it dictates the caching
--------------------------------------------

Priors are the author's earlier prompts **in their already-defended form**, because that is what the
attacker actually sees. ``build_dataset.finalize`` sorts by ``(source, author_id, started_at,
doc_id)``, so an author's documents already arrive contiguous and chronological -- arrival order *is*
the timeline and no extra column has to be plumbed through ``apply_defenses``.

.. code-block:: text

    d1 -> AFR(d1, priors=[])      -> D1
    d2 -> AFR(d2, priors=[D1])    -> D2
    d3 -> AFR(d3, priors=[D1,D2]) -> D3

So documents within an author are *not* independent, and the cache namespace is one row per **author**
rather than per document (see :meth:`AgenticFootprintDefense.transform`). ``shardable = False`` for
the same reason as ``loo_unlink``, one level stronger: a shard holding an arbitrary subset of an
author would not merely mis-measure the baseline, it would cascade from the wrong documents.

Determinism is engineered, not incidental
-----------------------------------------

A cascade compounds drift: one flipped token in document 2 means documents 3..k are computed from a
different input, and a content-addressed cache that returns different text on a hit than on a miss
puts two regimes in one parquet. Every source is closed deliberately:

1. **The cascade unit is the cache unit.** An author is always recomputed from its first document.
   There is no partial state to restore inconsistently.
2. **vLLM greedy decoding is not batch-invariant.** ``temperature=0`` fixes the sampling rule, not
   the arithmetic -- continuous batching changes reduction order, which occasionally flips a token.
   So: ``seed=`` and ``enforce_eager=True`` on the engine; **every** generation is a fixed-size
   ``chat()`` call (1 for signature/abstract/propose/repair -- a multi-candidate round asks for its
   candidates inside *one* reply -- and 2 for the answer and judge pairs), never "however many are
   pending"; and authors are processed strictly one at a time in sorted ``author_id`` order.
3. **Seeded sampling over a sorted list.** The reference pool follows ``build_subset.select_authors``:
   sort by ``doc_id`` *before* drawing, so a rebuild that changes row order does not move the sample.
4. **No** :func:`hash`. Python's string hash is salted per process. (``loo_unlink._fake_backend``
   uses it and is, as a result, not reproducible across processes; this module uses ``zlib.crc32``.)
5. What is *not* claimed: bitwise equality across different GPU models, drivers or vLLM builds. The
   reproducibility artifacts of record are the cache table and ``edits_a<NN>.jsonl``. Bump
   :attr:`AgenticFootprintDefense.version` when the serving stack changes.

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
DEFAULT_MAX_PROBES = 10

#: Candidates requested in the first (exploratory) round, inside one reply. Later rounds ask for one
#: and refine. Explore-then-exploit: three genuinely different approaches cost the same generation as
#: one, and the loop cannot tell a dead end from a slow start without having tried more than one.
DEFAULT_FIRST_ROUND_CANDIDATES = 3

#: Consecutive rounds without improvement before the escalation ladder moves up a rung. A round that
#: produced no admissible candidate at all counts as non-improving: a model stuck on the output
#: format is stuck, and a different instruction is a better response than another identical retry.
DEFAULT_ESCALATE_AFTER = 2

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

#: Authors between cache checkpoints. An author is the atomic unit of the cascade, so this is also
#: the granularity a preemption can cost -- roughly 25-100 documents at ``wildchat_small``'s shape.
DEFAULT_CHECKPOINT_EVERY = 5

#: Tokens for a generated answer in the utility gate. Short on purpose: the judge compares whether a
#: request was resolved, not prose quality.
DEFAULT_ANSWER_TOKENS = 256

#: Tokens for one proposal reply. Generous, because a three-candidate round returns three full
#: rewrites of a whole document in a single reply and a truncated last candidate is simply discarded.
DEFAULT_PROPOSE_TOKENS = 2048

#: Tokens for the rolling author signature. Six bullets.
DEFAULT_SIGNATURE_TOKENS = 256

#: Characters of the nearest prior shown to the agent. Enough to recognize what recurs; bounded so a
#: 400-turn session cannot crowd out the draft being edited.
NEAREST_PRIOR_CHARS = 2000

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


def defend_document(turns, priors, prior_texts, pool, *, embed, abstract, propose,
                    profile: str = "", alpha: float = DEFAULT_ALPHA,
                    max_probes: int = DEFAULT_MAX_PROBES,
                    escalate_after: int = DEFAULT_ESCALATE_AFTER,
                    top_m: int = DEFAULT_TOP_M,
                    first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                    centroid=None, g_median: float | None = None,
                    ) -> tuple[Candidate, Candidate, list, DocumentTrace]:
    """Run the abstraction pass and the probe loop over one document.

    Returns ``(winner, floor, history, trace)``: the best-scoring candidate, the stage-1 abstraction
    that the utility gate falls back to, every admissible candidate seen (so the gate can walk back
    down the escalation ladder), and the trace. The gate itself runs in
    :func:`gate_and_select`, one level up, because it is the only part that needs a second round of
    generation *after* the loop has finished.

    ``embed(texts) -> (n, d)`` unit rows, ``abstract(turns) -> [turn]`` and ``propose(payload) ->
    [[turn], ...]`` are injected rather than constructed here, so the loop can be exercised against
    fakes with no GPU -- see :func:`_selftest`.
    """
    turns = [str(turn) for turn in turns]
    n_turns = len(turns)
    priors = np.asarray(priors, dtype=float) if priors is not None else np.zeros((0, 0))
    pool = np.asarray(pool, dtype=float) if pool is not None else np.zeros((0, 0))
    cold_start = priors.size == 0
    if centroid is None:
        centroid = pool_centroid(pool)
    if g_median is None:
        g_median = median_genericness(pool, centroid)

    def score(vector: np.ndarray, level: int, round_index: int) -> Candidate:
        s_max, s_top3, r_med = objective_scores(vector, priors, pool, top_m)
        generic = float(vector @ centroid) if centroid.size else 0.0
        candidate = Candidate(turns=[], level=level, round_index=round_index, s_max=s_max,
                              s_top3=s_top3, r_med=r_med, genericness=generic, vector=vector)
        # One lower-is-better number for both regimes, so the loop has a single comparison.
        candidate.objective = -generic if cold_start else s_max
        return candidate

    trace = DocumentTrace(n_priors=int(priors.shape[0]) if priors.size else 0,
                          cold_start=cold_start, alpha=float(alpha), n_turns=n_turns,
                          median_genericness=float(g_median), original=join_document(turns))

    # --- stage 1: the abstraction pass, which is also the floor ---------------
    proposed_floor = abstract(turns)
    if not admissible(proposed_floor, turns):
        # A pass-through here is visibly undefended, which is a far better failure than splicing a
        # truncated or fabricated rewrite into the dataset. Recorded so a run where it is common is
        # diagnosable as a prompt problem rather than read as a weak defense.
        proposed_floor = list(turns)
        trace.stage1_ok = False
    floor_vector = embed([join_document(proposed_floor)])[0]
    floor = score(floor_vector, level=0, round_index=0)
    floor.turns = list(proposed_floor)
    trace.stage1 = join_document(proposed_floor)

    original_candidate = score(embed([join_document(turns)])[0], level=0, round_index=0)
    trace.original_objective = original_candidate.objective
    trace.baseline_objective = floor.objective
    trace.r_med_baseline = floor.r_med

    target = (genericness_target(floor.genericness, g_median, alpha) if cold_start
              else linkage_target(floor.s_max, floor.r_med, alpha))
    # Both regimes are compared as lower-is-better, so the cold-start target is negated with its
    # objective rather than being special-cased at every comparison below.
    trace.target = float(target)
    loop_target = -target if cold_start else target

    best = floor
    history: list[Candidate] = [floor]
    rounds: list[RoundRecord] = []
    level = 0
    stalled = 0
    probes = 0
    round_index = 0

    if cold_start and pool.size == 0:
        stop_reason = "no_priors"
    elif best.objective <= loop_target:
        stop_reason = "no_priors" if cold_start else "stage1_sufficient"
    elif max_probes <= 0:
        stop_reason = "no_probes"
    else:
        stop_reason = "probes_exhausted"
        while probes < max_probes:
            round_index += 1
            want = first_round_candidates if round_index == 1 else 1
            want = min(want, max_probes - probes)
            if want <= 0:
                break

            payload = {
                "turns": best.turns,
                "n_turns": n_turns,
                "want": want,
                "profile": profile,
                "nearest_prior": nearest_prior_text(best, priors, prior_texts),
                "scores": _format_scores(best, target, cold_start, top_m),
                "trajectory": _format_trajectory(rounds, cold_start),
                "probes_left": max_probes - probes,
                "level": level,
                "cold_start": cold_start,
            }
            proposals = propose(payload)
            usable = [candidate for candidate in proposals if admissible(candidate, turns)]
            record = RoundRecord(round_index=round_index, level=level, requested=want,
                                 admissible=len(usable), probes_used=0)

            if not usable:
                record.note = "no admissible candidate"
                stalled += 1
            else:
                vectors = embed([join_document(candidate) for candidate in usable])
                probes += len(usable)
                record.probes_used = len(usable)
                scored = []
                for position, candidate_turns in enumerate(usable):
                    candidate = score(vectors[position], level=level, round_index=round_index)
                    candidate.turns = list(candidate_turns)
                    scored.append(candidate)
                    history.append(candidate)
                round_best = min(scored, key=lambda item: item.objective)
                record.objectives = [round(item.objective, 6) for item in scored]
                record.best_objective = round_best.objective
                if round_best.objective < best.objective - IMPROVEMENT_EPSILON:
                    best = round_best
                    record.improved = True
                    stalled = 0
                else:
                    stalled += 1
            rounds.append(record)

            if best.objective <= loop_target:
                stop_reason = "target_met"
                break
            if stalled >= escalate_after:
                level += 1
                stalled = 0
                if level >= N_ESCALATION_LEVELS:
                    stop_reason = "ladder_exhausted"
                    break

    trace.stop_reason = stop_reason
    trace.rounds = rounds
    trace.probes_used = probes
    trace.n_rounds = len(rounds)
    trace.max_level = max((candidate.level for candidate in history), default=0)
    trace.final_objective = best.objective
    trace.target_met = best.objective <= loop_target
    trace.nearest_prior_index = nearest_prior_index(best, priors, prior_texts)
    return best, floor, history, trace


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


def defend_author(documents, pool, *, embed, abstract, propose, gate, signature,
                  alpha: float = DEFAULT_ALPHA, max_probes: int = DEFAULT_MAX_PROBES,
                  escalate_after: int = DEFAULT_ESCALATE_AFTER, top_m: int = DEFAULT_TOP_M,
                  max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                  first_round_candidates: int = DEFAULT_FIRST_ROUND_CANDIDATES,
                  ) -> tuple[list, list]:
    """Cascade one author's whole timeline. ``documents`` is ``[(doc_id, [turn, ...]), ...]``.

    Each document is defended against the **defended** text of the ones before it, and the rolling
    profile is updated from the defended text too -- so nothing the attacker cannot see ever enters
    the objective. This is the unit of both the cascade and the cache: an author is always computed
    from its first document, which is what makes a cache hit and a cache miss the same text.
    """
    centroid = pool_centroid(np.asarray(pool, dtype=float))
    g_median = median_genericness(np.asarray(pool, dtype=float), centroid)

    prior_texts: list[str] = []
    prior_vectors: list[np.ndarray] = []
    profile = ""
    defended_documents: list[list[str]] = []
    traces: list[DocumentTrace] = []

    for position, (doc_id, turns) in enumerate(documents):
        priors = (np.vstack(prior_vectors) if prior_vectors else np.zeros((0, 0)))
        winner, floor, history, trace = defend_document(
            turns, priors, prior_texts, pool, embed=embed, abstract=abstract, propose=propose,
            profile=profile, alpha=alpha, max_probes=max_probes, escalate_after=escalate_after,
            top_m=top_m, first_round_candidates=first_round_candidates,
            centroid=centroid, g_median=g_median)
        chosen = gate_and_select(winner, floor, history, join_document(turns),
                                 gate=gate, max_utility_loss=max_utility_loss, trace=trace)

        defended_turns = [str(turn) for turn in chosen.turns]
        defended_text = join_document(defended_turns)
        trace.doc_id = str(doc_id)
        trace.position = position
        trace.defended = defended_text

        defended_documents.append(defended_turns)
        traces.append(trace)

        # The cascade step: what the attacker will see becomes the next document's prior.
        if position + 1 < len(documents):
            prior_texts.append(defended_text)
            prior_vectors.append(embed([defended_text])[0])
            profile = signature(profile, defended_text)

    return defended_documents, traces


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
                 seed: int = DEFAULT_SEED, featurizer=None):
        self.prompts = prompts
        self.model = model
        self.answer_tokens = answer_tokens
        self.propose_tokens = propose_tokens
        self.seed = seed
        self._featurizer = featurizer
        self._llm = None
        self._sampling = {}

    # -- embedding --
    def featurizer(self):
        if self._featurizer is None:
            from ..features.harrier import HarrierFeaturizer

            self._featurizer = HarrierFeaturizer()
        return self._featurizer

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.featurizer().featurize(list(texts)), dtype=float)

    # -- generation --
    def _engine(self):
        if self._llm is None:
            from vllm import LLM, SamplingParams

            from ._backends import model_path, resolve_model_path

            path = resolve_model_path(self.model or model_path("afr", MODEL_ENV_VAR))
            print(f"[afr] loading agent {path} (vLLM)")
            # `seed` and `enforce_eager` are both load-bearing for reproducibility, not tuning:
            # see the determinism note in the module docstring. Harrier co-resides, so the
            # utilization is left below what a 30B FP8 model would otherwise take.
            self._llm = LLM(model=path, dtype="auto", gpu_memory_utilization=0.75,
                            enforce_eager=True, seed=self.seed)
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

    def chat(self, system: str, users: list[str], kind: str) -> list[str]:
        """One batched chat call: same system prompt, many user messages, replies in input order."""
        if not users:
            return []
        engine = self._engine()
        conversations = [[{"role": "system", "content": system},
                          {"role": "user", "content": user}] for user in users]
        outputs = engine.chat(conversations, self._sampling[kind], use_tqdm=False)
        return [output.outputs[0].text.strip() for output in outputs]

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

    #: See the module docstring. Checked by ``apply_defenses``.
    shardable = False

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
                                          answer_tokens=self.answer_tokens, seed=self.seed)
        return self._backend

    def _turns_with_repair(self, system: str, user: str, kind: str, n_turns: int):
        """One generation, plus at most one format-repair retry. ``None`` if both fail.

        The repair is a separate call rather than a longer prompt because the failure it fixes is
        not a reasoning failure: the model produced a fine rewrite in the wrong wrapper, and showing
        it its own reply is the shortest path to the same rewrite in the right one.
        """
        backend = self.backend()
        replies = backend.chat(system, [user], kind)
        reply = replies[0] if replies else ""
        parsed = parse_turns(reply, n_turns)
        if parsed is not None:
            return parsed
        repair_system = render_template(self.prompts["repair_system_prompt"],
                                        {"N_TURNS": str(n_turns)})
        repair_user = render_template(
            self.prompts["repair_user_template"],
            {"ERROR": f"expected exactly {n_turns} <turn> block(s), numbered 1..{n_turns}",
             "REPLY": reply})
        repaired = backend.chat(repair_system, [repair_user], kind)
        return parse_turns(repaired[0] if repaired else "", n_turns)

    def _abstract(self, turns) -> list[str] | None:
        """Stage 1: generalize the whole document upward. Returns ``None`` on an unusable reply."""
        n_turns = len(turns)
        system = render_template(self.prompts["abstract_system_prompt"],
                                 {"N_TURNS": str(n_turns)})
        user = render_template(self.prompts["abstract_user_template"],
                               {"DOCUMENT": render_turn_blocks(turns), "N_TURNS": str(n_turns)})
        return self._turns_with_repair(system, user, "abstract", n_turns)

    def _propose(self, payload: dict) -> list[list[str]]:
        """Stage 2: one proposal round. Returns the well-formed candidates in the reply.

        A multi-candidate round is ONE call asking for several rewrites, not several calls -- both
        because it is cheaper and because a fixed batch size of 1 is what keeps greedy decoding
        reproducible (see the module docstring).
        """
        n_turns = int(payload["n_turns"])
        want = int(payload["want"])
        level = min(int(payload["level"]), N_ESCALATION_LEVELS - 1)
        system = render_template(
            self.prompts["propose_system_prompt"],
            {"N_TURNS": str(n_turns), "N_CANDIDATES": str(want),
             "ESCALATION": self.prompts["escalation_levels"][level]})
        template = (self.prompts["propose_cold_start_template"] if payload["cold_start"]
                    else self.prompts["propose_user_template"])
        user = render_template(template, {
            "DOCUMENT": render_turn_blocks(payload["turns"]),
            "N_TURNS": str(n_turns),
            "N_CANDIDATES": str(want),
            "PROFILE": payload["profile"] or "(nothing recorded yet)",
            "NEAREST_PRIOR": payload["nearest_prior"],
            "SCORES": payload["scores"],
            "TRAJECTORY": payload["trajectory"],
            "PROBES_LEFT": str(payload["probes_left"]),
        })
        replies = self.backend().chat(system, [user], "propose")
        reply = replies[0] if replies else ""
        candidates = parse_candidates(reply, n_turns, want)
        if candidates:
            return candidates
        # Nothing parsed: give the format one repair attempt, which recovers the common case of a
        # single well-reasoned rewrite returned without the wrapper.
        repaired = self._turns_with_repair(
            render_template(self.prompts["repair_system_prompt"], {"N_TURNS": str(n_turns)}),
            render_template(self.prompts["repair_user_template"],
                            {"ERROR": f"expected {want} candidate(s), each with exactly {n_turns} "
                                      f"<turn> block(s)", "REPLY": reply}),
            "propose", n_turns)
        return [repaired] if repaired else []

    def _signature(self, profile: str, defended_document: str) -> str:
        """Stage 0: fold one newly defended document into the rolling author profile."""
        from ._backends import extract_tagged_output

        user = render_template(self.prompts["signature_user_template"],
                               {"PROFILE": profile or "(empty)", "DOCUMENT": defended_document})
        replies = self.backend().chat(self.prompts["signature_system_prompt"], [user], "signature")
        extracted = extract_tagged_output(replies[0] if replies else "", "profile")
        # An unusable reply keeps the previous profile rather than clearing it: a stale profile is
        # weaker guidance, an empty one is none at all.
        return extracted.strip() if extracted and extracted.strip() else profile

    def _gate(self, original_document: str, variants: list[str]) -> list[float]:
        """Stage 3: utility loss in [0, 1] per variant, judged on the ANSWER, not the prompt.

        Both sides are judged, so a reference answer that was itself poor is not charged to the
        rewrite. An unreadable verdict is the maximum loss rather than zero: a candidate whose cost
        cannot be established must not become the cheapest one to accept. Identical in construction
        to ``loo_unlink._utility`` so the two defenses' utility numbers can be read against each
        other.
        """
        backend = self.backend()
        answers = backend.chat(self.prompts["answer_system_prompt"],
                               [original_document, *variants], "answer")
        if not answers:
            return [1.0] * len(variants)
        reference_answer, variant_answers = answers[0], answers[1:]

        judged = backend.chat(
            self.prompts["judge_system_prompt"],
            [render_template(self.prompts["judge_user_template"],
                             {"REQUEST": original_document, "ANSWER": answer})
             for answer in (reference_answer, *variant_answers)],
            "judge")
        reference_score = parse_judge_score(judged[0]) if judged else None
        if reference_score is None:
            reference_score = 5.0

        losses = []
        for reply in judged[1:]:
            score = parse_judge_score(reply)
            if score is None:
                losses.append(1.0)
                continue
            # Normalized by the 4-point span of the 1-5 scale, clamped at 0: a rewrite that somehow
            # improves the answer is free, not negative-cost.
            losses.append(max(0.0, (reference_score - score) / 4.0))
        return losses

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
        pool_positions = select_reference_pool(order, n_reference=self.n_reference, seed=self.seed)
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
        self._open_log(cache.dir)

        # The pool is embedded ONCE for the whole run and then sliced per author. Re-embedding it
        # per author would be ~n_authors x n_reference forward passes -- 100k on wildchat_small --
        # to produce vectors that are identical every time.
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
            outputs = []
            for source in missing:
                author = source_author[source]
                _, documents = decode_author_source(source)
                defended, traces = defend_author(
                    documents, pool_for(author),
                    embed=self.backend().embed, abstract=self._abstract, propose=self._propose,
                    gate=self._gate, signature=self._signature,
                    alpha=self.alpha, max_probes=self.max_probes,
                    escalate_after=self.escalate_after, top_m=self.top_m,
                    max_utility_loss=self.max_utility_loss,
                    first_round_candidates=self.first_round_candidates)
                for trace in traces:
                    trace.author_id = author
                    self._write_trace(trace)
                outputs.append(encode_author_output(defended))
            return outputs

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

    print("the escalation ladder:")

    def stubborn_propose(payload):
        # Never improves: the exact draft it was given, returned unchanged.
        return [list(payload["turns"])] * int(payload["want"])

    _, _, _, stalled_trace = defend_document(
        turns, priors, prior_texts, pool, embed=embed, abstract=abstract,
        propose=stubborn_propose, alpha=0.0, escalate_after=2)
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
          "every registered variant refuses sharding")
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
