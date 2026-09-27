r"""Leave-one-out content unlinkability: remove what identifies the author, keep what answers.

Every other defense in this package perturbs *style* -- neutralize it (``styleremix``,
``qwen_rewrite``), add per-word DP noise (``dp_mlm``), or manufacture shared quirks
(``collision_seeding``). Against a frontier embedding model that is the wrong axis. What links two
prompts by the same person is **content**: the problem domain, the project specifics, the library
combination, the named entities, the way the task is framed. This defense goes after that.

The method, per prompt:

1. Segment it into spans (:mod:`._spans` -- sentences, or words when the turn is a single sentence).
2. **Linkage.** Delete a span, re-embed, and measure how much the prompt's similarity to its
   author's *other* prompts drops. Against the top-``m`` most similar siblings rather than the mean,
   because the attack succeeds on the nearest neighbour and the metric should reflect what the
   attack actually does.
3. **Utility.** Delete the same span, generate an answer to the modified prompt, and have a judge
   rate whether that answer still resolves the *original* request. Utility is measured downstream on
   the answer, never as similarity to the original prompt -- similarity to the original is exactly
   what step 2 is trying to reduce, so using it here would make the objective fight itself.
4. Edit the best ``linkage / utility`` span, **generalizing it upward rather than deleting it**, then
   re-score linkage on what remains and repeat until a linkage budget is met.

Why bother with something this blunt before trying gradients or adversarial rewriting: it **removes
information** instead of perturbing it. An adversarial perturbation exploits one surrogate's decision
boundary and transfers poorly to a stronger encoder. Deleted content is gone for every model.
Surrogate mismatch costs ranking quality here, not soundness -- which is what makes the Gemini
transfer test in the evaluation stage a real test rather than self-grading.

Everything in the loop is local
-------------------------------

Scoring uses :class:`~prompt_anonymity.features.harrier.HarrierFeaturizer` (local, offline); the
rewrite, the answers and the rating all run on one local vLLM engine
(:data:`MODEL_ENV_VAR` -> ``models.toml [loo_unlink]``). Nothing here calls a metered API, which is
the only reason the loop is affordable given it is roughly quadratic in span count and the budget
sweep multiplies that by the number of operating points. The paid channels (Gemini embeddings, the
DeepSeek judge) grade the finished text afterwards and never touch this file.

Two departures from the design spec, both deliberate
----------------------------------------------------

**Utility is scored once per span; linkage is re-scored every iteration.** The spec re-scores both.
Its argument for re-scoring is about *linkage* -- two individually weak spans can be jointly
identifying, which leave-one-out in isolation cannot see -- and that argument does not carry to
utility, whose loss under span removal is close to additive. Re-scoring utility would multiply the
generation count (the dominant cost of the whole pipeline) by the number of edits per prompt.
``rescore_utility=True`` does it the strict way for a validation run.

**Word-granularity spans for single-sentence turns.** See :func:`._spans._cut_points`.

Sharding
--------

``shardable = False``. Linkage is measured against the author's other documents, and
``apply_defenses --num-shards`` splits by *document*, so a shard holds an arbitrary subset of each
author -- the baseline would be computed against a truncated author and would differ from shard to
shard. ``apply_defenses`` refuses a sharded run for this defense; checkpointing every
:data:`DEFAULT_CHECKPOINT_EVERY` documents is what makes a preempted job resumable instead.

Command line::

    python -m prompt_anonymity.defenses.loo_unlink --selftest
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

from ..caching import IndexedRowCache
from ..core import AttackData
from ._backends import (TURN_ID_SEPARATOR, document_id, extract_tagged_output, join_turns,
                        render_template, split_turns)
from ._spans import Span, apply_edits, segment_turns
from .base import CachedDefense

#: Turn joiner for the text the ATTACKER sees. Must match
#: :data:`prompt_anonymity.data.compute_features.TURN_SEPARATOR` -- linkage is only meaningful if the
#: defense scores the same string the featurizer will later embed. Redefined rather than imported
#: because ``data`` imports ``defenses`` and importing it back would be circular; the selftest
#: asserts the two still agree.
TURN_SEPARATOR = "\n\n"

#: Environment override for the local generator, ahead of ``models.toml [loo_unlink]``.
MODEL_ENV_VAR = "LOO_UNLINK_MODEL"

#: Prompts, rubric and templates. Every string in it is hashed into :meth:`LOOUnlinkDefense.params`.
PROMPTS_PATH = Path(__file__).with_name("loo_unlink_prompts.yaml")

#: How many of the author's most-similar other prompts define the linkage baseline. The attack
#: succeeds on the nearest neighbour, so a global mean would understate what is at stake.
DEFAULT_TOP_M = 3

#: Target fractional reduction in that top-m similarity. The sweep knob; see the registry below.
DEFAULT_BUDGET = 0.3

#: Cumulative utility loss (0 = answers unchanged, 1 = answers useless) past which a prompt stops
#: being edited even if its linkage budget is unmet. A prompt is allowed to FAIL its budget; it is
#: not allowed to be destroyed reaching for one.
DEFAULT_MAX_UTILITY_LOSS = 0.5

#: Floor on the denominator of ``linkage / utility``, so a span the judge scored as costless does not
#: divide by zero and win by infinity.
UTILITY_EPSILON = 0.05

#: Hard cap on edits per prompt, independent of the budgets. Protection against a pathological
#: document (hundreds of word spans, each buying a sliver of linkage) monopolizing a run.
DEFAULT_MAX_EDITS = 12

#: Documents between cache checkpoints. Matches the granularity a preemption can cost.
DEFAULT_CHECKPOINT_EVERY = 25

#: Tokens allowed for a generated answer. The single biggest lever on how long a budget point takes:
#: the loop generates one answer per span. The judge compares whether a request was resolved, not
#: prose quality, so a short answer is not a handicap.
DEFAULT_ANSWER_TOKENS = 256

#: Tokens allowed for one span rewrite. Generous relative to a sentence so a long span is not cut
#: mid-clause, which would splice a truncated fragment into the prompt.
DEFAULT_REWRITE_TOKENS = 256

#: Marker separating a document's author-context digest from its text in the cache source. The digest
#: has to be INSIDE the cached source, not looked up beside it: this defense's output depends on the
#: author's other prompts, so two runs over different subsets must not share a cache entry for the
#: same document. Same idiom as ``frame_shift.encode_framed_source``.
CONTEXT_SOURCE_PREFIX = "<<loo:"
CONTEXT_SOURCE_PATTERN = re.compile(r"^<<loo:([0-9a-f]+)>>\n")


# --- prompt loading ----------------------------------------------------------

def load_prompts(path: Path | str | None = None) -> dict:
    """The prompt bundle as a dict of strings. Missing or blank keys are a hard error.

    Loaded eagerly by the class rather than at import, so a broken YAML fails when the defense is
    constructed rather than when a GPU job reaches its first document.
    """
    import yaml  # base dependency; the utility judge's rubric loads the same way

    path = Path(path) if path else PROMPTS_PATH
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as error:
        raise SystemExit(f"cannot read the loo_unlink prompts at {path}: {error}") from None

    required = ("rewrite_system_prompt", "rewrite_user_template", "answer_system_prompt",
                "judge_system_prompt", "judge_user_template")
    missing = [key for key in required if not str(loaded.get(key, "")).strip()]
    if missing:
        raise SystemExit(f"{path} is missing or has blank keys: {', '.join(missing)}")
    return {key: str(loaded[key]) for key in required}


# --- cache-source encoding ---------------------------------------------------

def context_digest(sibling_texts) -> str:
    """A short digest of the author's other prompts, in a canonical order.

    This is what makes the cache correct rather than merely fast. The defended text is a function of
    ``(document, the author's other documents)``, so a run over a different subset -- a different
    author sample, a re-run with a different ``--k-min`` -- must not be served a cached rewrite
    computed against a different sibling set. Sorting makes the digest independent of row order.
    """
    digest = hashlib.sha256()
    for text in sorted(str(text) for text in sibling_texts):
        digest.update(text.encode("utf-8", "replace"))
        digest.update(b"\0")
    return digest.hexdigest()[:16]


def encode_source(digest: str, text: str) -> str:
    """Compose the cache source for one document under one author context."""
    return f"{CONTEXT_SOURCE_PREFIX}{digest}>>\n{text}"


def decode_source(source: str) -> tuple[str, str]:
    """Split a composed source back into ``(digest, text)``; an unmarked source yields an empty
    digest, which is what a cache table written before this encoding existed would look like."""
    match = CONTEXT_SOURCE_PATTERN.match(source)
    if match is None:
        return "", source
    return match.group(1), source[match.end():]


# --- scoring records ---------------------------------------------------------

@dataclass
class SpanScore:
    """One span's scores, as written to the edit log."""

    index: int
    turn_index: int
    start: int
    end: int
    text: str
    granularity: str
    editable: bool
    linkage: float = 0.0
    utility_loss: float = 0.0
    ratio: float = 0.0
    applied_order: int | None = None   # None = never edited
    replacement: str | None = None


@dataclass
class DocumentTrace:
    """Everything one document's run produced, for ``edits.jsonl``."""

    doc_id: str
    author_id: str
    baseline_similarity: float
    final_similarity: float
    budget: float
    budget_met: bool
    cumulative_utility_loss: float
    stop_reason: str
    n_spans: int
    n_edits: int
    spans: list = field(default_factory=list)
    original: str = ""
    defended: str = ""


# --- the algorithm (model-free, so it is testable without a GPU) -------------

def top_m_similarity(vector: np.ndarray, sibling_matrix: np.ndarray, top_m: int) -> float:
    """Mean cosine similarity to the ``top_m`` most similar siblings.

    Vectors are L2-normalized by the featurizer, so cosine is a plain dot product. An author with no
    siblings scores 0.0: there is nothing to be linked to, so no span can reduce linkage and the
    document is passed through untouched.
    """
    if sibling_matrix.size == 0:
        return 0.0
    similarities = sibling_matrix @ vector
    if similarities.size > top_m:
        similarities = np.partition(similarities, -top_m)[-top_m:]
    return float(np.mean(similarities))


def defend_document(turns, sibling_texts, *, embed, rewrite, utility, budget: float,
                    top_m: int = DEFAULT_TOP_M,
                    max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                    max_edits: int = DEFAULT_MAX_EDITS,
                    rescore_utility: bool = False,
                    language: str = "english") -> tuple[list[str], DocumentTrace]:
    """Run the edit loop over one document. Returns ``(defended turns, trace)``.

    ``embed(texts) -> (n, d)`` unit-norm rows, ``rewrite(document, span_text) -> str`` and
    ``utility(document_variants) -> [loss]`` are injected rather than constructed here, so the loop
    can be exercised against fakes with no GPU -- see :func:`_selftest`.

    Turn count is preserved throughout: every edit is a splice inside one turn, and the loop
    reassembles from the ORIGINAL turns each iteration (``apply_edits`` against accumulated edits)
    rather than mutating in place, so span offsets stay valid for the whole run.
    """
    turns = [str(turn) for turn in turns]
    spans = [span for span in segment_turns(turns, language=language)]
    records = [SpanScore(index=i, turn_index=s.turn_index, start=s.start, end=s.end, text=s.text,
                         granularity=s.granularity, editable=s.editable)
               for i, s in enumerate(spans)]

    def render(edits, extra=None) -> list[str]:
        return apply_edits(turns, list(edits) + ([extra] if extra else []))

    def document_of(turn_list) -> str:
        return TURN_SEPARATOR.join(turn_list)

    original_document = document_of(turns)
    trace = DocumentTrace(doc_id="", author_id="", baseline_similarity=0.0, final_similarity=0.0,
                          budget=budget, budget_met=False, cumulative_utility_loss=0.0,
                          stop_reason="", n_spans=len(spans), n_edits=0, spans=records,
                          original=original_document, defended=original_document)

    siblings = [str(text) for text in sibling_texts]
    if not siblings:
        trace.stop_reason = "no_siblings"
        trace.budget_met = True   # nothing to unlink from; the budget is vacuously satisfied
        return turns, trace

    sibling_matrix = embed(siblings)
    baseline = top_m_similarity(embed([original_document])[0], sibling_matrix, top_m)
    trace.baseline_similarity = baseline
    trace.final_similarity = baseline
    target = (1.0 - budget) * baseline

    candidates = [i for i, span in enumerate(spans) if span.editable]
    if not candidates:
        trace.stop_reason = "no_editable_spans"
        return turns, trace

    # --- utility, once per span, against the original document ---------------
    def score_utility(edits, indices) -> dict:
        variants = [document_of(render(edits, (spans[i], ""))) for i in indices]
        return dict(zip(indices, utility(document_of(render(edits)), variants)))

    utility_loss = score_utility([], candidates)
    for index, loss in utility_loss.items():
        records[index].utility_loss = float(loss)

    edits: list[tuple[Span, str]] = []
    cumulative = 0.0
    remaining = list(candidates)
    stop_reason = "budget_met"

    while True:
        current_turns = render(edits)
        current_similarity = top_m_similarity(
            embed([document_of(current_turns)])[0], sibling_matrix, top_m)
        trace.final_similarity = current_similarity
        if current_similarity <= target:
            stop_reason = "budget_met"
            break
        if not remaining:
            stop_reason = "no_spans_left"
            break
        if len(edits) >= max_edits:
            stop_reason = "max_edits"
            break

        # --- linkage, re-scored against the CURRENT document ------------------
        variants = [document_of(render(edits, (spans[i], ""))) for i in remaining]
        vectors = embed(variants)
        linkage = {
            index: current_similarity - top_m_similarity(vectors[position], sibling_matrix, top_m)
            for position, index in enumerate(remaining)
        }
        if rescore_utility and edits:
            for index, loss in score_utility(edits, remaining).items():
                utility_loss[index] = float(loss)

        for index in remaining:
            records[index].linkage = float(linkage[index])
            records[index].ratio = float(
                linkage[index] / max(utility_loss[index], UTILITY_EPSILON))

        useful = [index for index in remaining if linkage[index] > 0]
        if not useful:
            # Nothing left whose removal reduces linkage at all. Editing further would spend
            # utility for no privacy, which is worse than failing the budget.
            stop_reason = "no_positive_linkage"
            break

        best = max(useful, key=lambda index: records[index].ratio)
        if cumulative + utility_loss[best] > max_utility_loss:
            stop_reason = "utility_budget"
            break

        replacement = rewrite(document_of(current_turns), spans[best].text)
        records[best].applied_order = len(edits)
        records[best].replacement = replacement
        edits.append((spans[best], replacement))
        cumulative += utility_loss[best]
        remaining.remove(best)

    defended = render(edits)
    trace.stop_reason = stop_reason
    trace.budget_met = trace.final_similarity <= target
    trace.cumulative_utility_loss = cumulative
    trace.n_edits = len(edits)
    trace.defended = document_of(defended)
    return defended, trace


# --- the local backend -------------------------------------------------------

class _LocalBackend:
    """One vLLM engine serving all three generative jobs, plus the Harrier embedder.

    Held together because they share a GPU: Harrier-0.6B and a 3B generator co-reside comfortably,
    and loading them separately per stage would mean paying the load twice per job.
    """

    def __init__(self, prompts: dict, *, model: str | None = None,
                 answer_tokens: int = DEFAULT_ANSWER_TOKENS,
                 rewrite_tokens: int = DEFAULT_REWRITE_TOKENS,
                 featurizer=None):
        self.prompts = prompts
        self.model = model
        self.answer_tokens = answer_tokens
        self.rewrite_tokens = rewrite_tokens
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
            from ._backends import (configure_cuda_toolkit, local_checkpoint, resolve_model_path,
                                    shared_checkpoint)

            configure_cuda_toolkit()  # must precede the import: vLLM reads the environment at import

            from vllm import LLM, SamplingParams

            # `local_checkpoint` prefers an already-mirrored copy of the weights over letting vLLM
            # silently download the repo id, same resolution `afr` uses.
            if self.model:
                path = resolve_model_path(shared_checkpoint(self.model) or self.model)
                print(f"[loo_unlink] generator checkpoint: {path}")
            else:
                path = local_checkpoint("loo_unlink", MODEL_ENV_VAR,
                                        allow_env="LOO_UNLINK_ALLOW_DOWNLOAD",
                                        size_hint="~6 GB for a 3B model")
            print(f"[loo_unlink] loading generator {path} (vLLM)")
            # gpu_memory_utilization is left well below the default: Harrier is on the same card,
            # and vLLM pre-allocates its KV cache greedily enough to OOM a co-resident model.
            self._llm = LLM(model=path, dtype="auto", gpu_memory_utilization=0.55,
                            enforce_eager=True)
            self._sampling = {
                # Temperature 0: the defense must be a deterministic function of its input, or the
                # content-addressed cache would return a different rewrite on a hit than on a miss.
                "answer": SamplingParams(temperature=0.0, max_tokens=self.answer_tokens),
                "rewrite": SamplingParams(temperature=0.0, max_tokens=self.rewrite_tokens),
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


def parse_judge_score(reply: str) -> float | None:
    """The 1-5 score out of a judge reply, or ``None`` when it cannot be read.

    Accepts the requested JSON and also a bare number, because a small local model asked for JSON
    returns a bare number often enough that treating it as a parse failure would throw away real
    verdicts. Anything else is None, and the caller decides what an unreadable verdict means -- it
    must not silently become a zero, which would read as "this edit cost nothing" and make the span
    maximally attractive to edit.
    """
    text = (reply or "").strip()
    if not text:
        return None
    match = re.search(r'"score"\s*:\s*([0-9]+(?:\.[0-9]+)?)', text)
    if match is None:
        match = re.search(r"\b([1-5])(?:\.[0-9]+)?\b", text)
    if match is None:
        return None
    score = float(match.group(1))
    return score if 1.0 <= score <= 5.0 else None


# --- the defense -------------------------------------------------------------

class LOOUnlinkDefense(CachedDefense):
    """Leave-one-out content unlinkability at one linkage budget.

    Parameters
    ----------
    budget : float
        Target fractional reduction in top-m author similarity. The sweep axis: 0.1 is a light
        touch, 0.7 removes most of what links the prompt to its author and costs accordingly.
    top_m, max_utility_loss, max_edits, rescore_utility
        See the module constants and the departures note in the module docstring.
    """

    name = "loo_unlink"
    version = "1"

    #: See the module docstring. Checked by ``apply_defenses``.
    shardable = False

    #: Documents between cache checkpoints, read by :meth:`transform`.
    checkpoint_every = DEFAULT_CHECKPOINT_EVERY

    def __init__(self, *, budget: float = DEFAULT_BUDGET, top_m: int = DEFAULT_TOP_M,
                 max_utility_loss: float = DEFAULT_MAX_UTILITY_LOSS,
                 max_edits: int = DEFAULT_MAX_EDITS, rescore_utility: bool = False,
                 model: str | None = None, prompts_path: str | None = None,
                 answer_tokens: int = DEFAULT_ANSWER_TOKENS,
                 log_dir: str | None = None, backend=None):
        if not 0.0 <= budget <= 1.0:
            raise ValueError(f"budget must be in [0, 1], got {budget}.")
        self.budget = float(budget)
        self.top_m = int(top_m)
        self.max_utility_loss = float(max_utility_loss)
        self.max_edits = int(max_edits)
        self.rescore_utility = bool(rescore_utility)
        self.model = model
        self.answer_tokens = int(answer_tokens)
        self.prompts = load_prompts(prompts_path)
        self.log_dir = log_dir
        self._backend = backend
        self._log_handle = None

    def params(self) -> dict:
        """Everything that changes the output, prompts included.

        The prompt strings are hashed rather than embedded so the key stays short, but they ARE in
        the key: editing the rubric or the rewrite contract changes every rewrite, and a run that
        silently mixed two contracts in one parquet would be uninterpretable.
        """
        prompt_digest = hashlib.sha256(
            "\0".join(self.prompts[key] for key in sorted(self.prompts)).encode("utf-8")
        ).hexdigest()[:16]
        return {
            "budget": round(self.budget, 4),
            "top_m": self.top_m,
            "max_utility_loss": round(self.max_utility_loss, 4),
            "max_edits": self.max_edits,
            "rescore_utility": self.rescore_utility,
            "answer_tokens": self.answer_tokens,
            "model": self.model or os.environ.get(MODEL_ENV_VAR) or "models.toml:loo_unlink",
            "prompts_sha": prompt_digest,
            "turn_separator": TURN_SEPARATOR,
        }

    # -- model-backed callables handed to the loop --

    def backend(self):
        if self._backend is None:
            self._backend = _LocalBackend(self.prompts, model=self.model,
                                          answer_tokens=self.answer_tokens)
        return self._backend

    def _rewrite(self, document: str, span_text: str) -> str:
        """Generalize one span upward; fall back to the original span if the reply is unusable.

        An unusable reply must not become a deletion. A truncated or empty rewrite spliced into the
        prompt is worse than an unedited prompt, and *visibly* undefended text is easier to account
        for in the results than silently corrupted text.
        """
        user = render_template(self.prompts["rewrite_user_template"],
                               {"DOCUMENT": document, "SPAN": span_text})
        replies = self.backend().chat(self.prompts["rewrite_system_prompt"], [user], "rewrite")
        extracted = extract_tagged_output(replies[0] if replies else "", "rewritten")
        if not extracted or not extracted.strip():
            return span_text
        # The span's own leading/trailing whitespace is structural -- it is what keeps the splice
        # reading continuously with the text on either side -- so it is restored around whatever the
        # model returned rather than trusting the model to have reproduced it.
        leading = span_text[:len(span_text) - len(span_text.lstrip())]
        trailing = span_text[len(span_text.rstrip()):]
        return f"{leading}{extracted.strip()}{trailing}"

    def _utility(self, original_document: str, variants: list[str]) -> list[float]:
        """Utility loss in [0, 1] for each variant: how much worse its answer serves the ORIGINAL
        request than the original prompt's own answer does.

        Both sides are judged, so a reference answer that was itself poor does not get charged to
        the edits. An unreadable verdict is treated as the maximum loss rather than zero: a span
        whose cost cannot be established should not be the cheapest one to cut.
        """
        backend = self.backend()
        answers = backend.chat(self.prompts["answer_system_prompt"],
                               [original_document, *variants], "answer")
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
            # Normalized by the 4-point span of the 1-5 scale, and clamped at 0: an edit that
            # somehow improves the answer is free, not negative-cost.
            losses.append(max(0.0, (reference_score - score) / 4.0))
        return losses

    # -- edit log --

    def _open_log(self, cache_dir) -> None:
        """Open ``edits.jsonl`` in append mode under the log directory.

        Append, not truncate, because a preempted job resumes and only recomputes what it lost: the
        lines already written describe documents that are still in the cache. A document recomputed
        after a cache invalidation appears twice, newest last -- readers should keep the last line
        per ``doc_id``.
        """
        directory = Path(self.log_dir) if self.log_dir else Path(cache_dir) / "loo_unlink_logs"
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"edits_b{int(round(self.budget * 100)):02d}.jsonl"
        print(f"[{self.name}] edit log -> {path}")
        self._log_handle = open(path, "a", encoding="utf-8")

    def _write_trace(self, trace: DocumentTrace) -> None:
        if self._log_handle is None:
            return
        record = asdict(trace)
        record["spans"] = [asdict(span) if not isinstance(span, dict) else span
                           for span in trace.spans]
        self._log_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._log_handle.flush()   # a preempted job should not lose the lines it already produced

    # -- the CachedDefense contract --

    def transform(self, data: AttackData, cache: IndexedRowCache) -> AttackData:
        """Defend every document, then return the defended text one row per input TURN.

        ``apply_defenses`` hands this defense the split exploded into turns -- ``unknown_texts`` are
        turns, ``unknown_ids`` are ``<doc_id>#<n>``, ``unknown_labels`` are author ids. That is
        enough to rebuild documents and group them by author, which is what the linkage baseline
        needs. Using the author's own prior prompts is legitimate here: a person defending their own
        prompts knows what they previously wrote.
        """
        from dataclasses import replace as dataclass_replace

        if data.unknown_texts is None:
            raise ValueError(f"defense {self.name!r} needs unknown_texts.")
        ids = ([str(i) for i in data.unknown_ids] if data.unknown_ids is not None
               else [str(i) for i in range(len(data.unknown_texts))])
        texts = [str(text) for text in data.unknown_texts]
        authors = [str(label) for label in data.unknown_labels]

        # Turn stream -> documents, preserving first-appearance order.
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

        document_text = {doc: TURN_SEPARATOR.join(doc_turns[doc]) for doc in order}
        siblings = {doc: [document_text[other] for other in by_author[doc_author[doc]]
                          if other != doc]
                    for doc in order}
        digests = {doc: context_digest(siblings[doc]) for doc in order}

        sources = [encode_source(digests[doc], join_turns(doc_turns[doc])) for doc in order]
        source_doc = {}
        for doc, source in zip(order, sources):
            source_doc.setdefault(source, doc)

        print(f"[{self.name}] {len(order):,} documents / {len(by_author):,} authors, "
              f"budget={self.budget:.2f}")
        self._open_log(cache.dir)

        def compute(missing: list[str]) -> list[str]:
            outputs = []
            for source in missing:
                doc = source_doc[source]
                _, packed = decode_source(source)
                defended, trace = defend_document(
                    split_turns(packed), siblings[doc],
                    embed=self.backend().embed, rewrite=self._rewrite, utility=self._utility,
                    budget=self.budget, top_m=self.top_m,
                    max_utility_loss=self.max_utility_loss, max_edits=self.max_edits,
                    rescore_utility=self.rescore_utility)
                trace.doc_id, trace.author_id = doc, doc_author[doc]
                self._write_trace(trace)
                outputs.append(join_turns(defended))
            return outputs

        try:
            defended_documents = cache.apply("unknown", sources, compute, ids=order,
                                             checkpoint_every=self.checkpoint_every)
        finally:
            if self._log_handle is not None:
                self._log_handle.close()
                self._log_handle = None
            if self._backend is not None:
                self._backend.close()

        # Documents -> the per-turn stream apply_defenses expects, back in input order.
        rebuilt = list(texts)
        for doc, defended in zip(order, defended_documents):
            turns = split_turns(defended)
            if len(turns) != len(positions[doc]):
                raise ValueError(
                    f"{self.name}: document {doc} came back with {len(turns)} turns but was given "
                    f"{len(positions[doc])}; turn boundaries must survive the rewrite."
                )
            for position, turn in zip(positions[doc], turns):
                rebuilt[position] = turn

        return dataclass_replace(data, unknown_texts=np.asarray(rebuilt, dtype=object))


# --- selftest ----------------------------------------------------------------

def _fake_backend(identifying=("Reykjavik", "llama")):
    """Deterministic stand-ins for the embedder, rewriter and judge.

    The embedder puts the identifying tokens on their own axes and everything else on a hashed
    axis, so deleting a span that carries one of those tokens measurably reduces similarity to a
    sibling that also carries it -- which is exactly the signal the real loop is looking for, with
    no GPU involved.
    """
    dimensions = 32

    def embed(texts):
        matrix = np.zeros((len(texts), dimensions))
        for row, text in enumerate(texts):
            for axis, token in enumerate(identifying):
                matrix[row, axis] = 3.0 * text.count(token)
            for word in text.split():
                matrix[row, 2 + hash(word.lower().strip(".,!?")) % (dimensions - 2)] += 1.0
            norm = np.linalg.norm(matrix[row])
            if norm:
                matrix[row] /= norm
        return matrix

    def rewrite(document, span_text):
        replaced = span_text
        for token in identifying:
            replaced = replaced.replace(token, "a small business")
        return replaced

    def utility(original, variants):
        # Cost proportional to how much text the variant dropped -- additive, monotone, and enough
        # structure for the ratio ranking to be exercised.
        return [min(1.0, (len(original) - len(variant)) / max(len(original), 1) * 2.0)
                for variant in variants]

    return embed, rewrite, utility


def _selftest() -> None:
    failures: list[str] = []

    def check(condition: bool, message: str) -> None:
        print(f"  {'ok  ' if condition else 'FAIL'}  {message}")
        if not condition:
            failures.append(message)

    print("constants agree with the rest of the pipeline:")
    from ..data.compute_features import TURN_SEPARATOR as PIPELINE_SEPARATOR
    from ..data.apply_defenses import TURN_ID_SEPARATOR as PIPELINE_TURN_ID
    check(TURN_SEPARATOR == PIPELINE_SEPARATOR,
          "TURN_SEPARATOR matches compute_features (linkage scores the text the attacker embeds)")
    check(TURN_ID_SEPARATOR == PIPELINE_TURN_ID, "TURN_ID_SEPARATOR matches apply_defenses")

    print("prompts:")
    prompts = load_prompts()
    check(len(prompts) == 5, f"all five prompt keys load (got {len(prompts)})")

    print("judge parsing:")
    check(parse_judge_score('{"score": 4}') == 4.0, "reads the requested JSON")
    check(parse_judge_score("4") == 4.0, "reads a bare number")
    check(parse_judge_score("nonsense") is None, "unreadable -> None, never a silent zero")
    check(parse_judge_score('{"score": 9}') is None, "out-of-range -> None")

    print("cache source encoding:")
    encoded = encode_source("deadbeef", "hello")
    check(decode_source(encoded) == ("deadbeef", "hello"), "round-trips")
    check(decode_source("unmarked")[0] == "", "an unmarked source degrades rather than crashing")
    check(context_digest(["a", "b"]) == context_digest(["b", "a"]), "digest ignores sibling order")
    check(context_digest(["a"]) != context_digest(["a", "b"]),
          "a different sibling set is a different cache entry")

    print("defend_document:")
    embed, rewrite, utility = _fake_backend()
    turns = ["I run a llama farm in Reykjavik. I need help with a Django inventory app. "
             "What models should I define?"]
    sibling = ["My llama farm in Reykjavik needs a stock tracker. Any advice on Django?"]

    defended, trace = defend_document(turns, sibling, embed=embed, rewrite=rewrite,
                                      utility=utility, budget=0.3)
    check(len(defended) == len(turns), "turn count is preserved")
    check(trace.n_edits > 0, f"at least one edit was applied (got {trace.n_edits})")
    check(trace.final_similarity < trace.baseline_similarity,
          f"similarity dropped ({trace.baseline_similarity:.3f} -> {trace.final_similarity:.3f})")
    check("Reykjavik" not in "".join(defended) or "llama" not in "".join(defended),
          "an identifying token was generalized away")
    check(all(record.applied_order is None or record.replacement is not None
              for record in trace.spans), "every applied edit recorded its replacement")

    zero, zero_trace = defend_document(turns, sibling, embed=embed, rewrite=rewrite,
                                       utility=utility, budget=0.0)
    check(zero == [str(t) for t in turns], "budget 0 is a byte-identical pass-through")
    check(zero_trace.n_edits == 0, "...and applies no edits")

    lone, lone_trace = defend_document(turns, [], embed=embed, rewrite=rewrite,
                                       utility=utility, budget=0.5)
    check(lone == [str(t) for t in turns] and lone_trace.stop_reason == "no_siblings",
          "an author with no other prompts is passed through untouched")

    capped, capped_trace = defend_document(turns, sibling, embed=embed, rewrite=rewrite,
                                           utility=utility, budget=1.0, max_utility_loss=0.0)
    check(capped_trace.n_edits == 0 and capped_trace.stop_reason == "utility_budget",
          "a zero utility allowance stops before the first edit rather than destroying the prompt")

    strict, strict_trace = defend_document(turns, sibling, embed=embed, rewrite=rewrite,
                                           utility=utility, budget=0.3, rescore_utility=True)
    check(strict_trace.n_edits > 0, "rescore_utility=True still terminates and edits")

    print("multi-turn documents:")
    many = ["I run a llama farm in Reykjavik.", "How do I model feed batches in Django?"]
    out, many_trace = defend_document(many, sibling, embed=embed, rewrite=rewrite,
                                      utility=utility, budget=0.4)
    check(len(out) == 2, "a two-turn document comes back as two turns")
    check(many_trace.n_spans >= 2, f"spans were found in both turns ({many_trace.n_spans})")

    print("registry:")
    from . import DEFENSES
    sweep = [name for name in DEFENSES if name.startswith("loo_unlink")]
    check(len(sweep) == len(LOO_UNLINK_BUDGETS) + 1,
          f"the budget sweep is registered ({len(sweep)} entries: {sorted(sweep)})")
    check(all(not DEFENSES[name].shardable for name in sweep),
          "every registered variant refuses sharding")

    print(f"\n{len(failures)} failure(s).")
    if failures:
        raise SystemExit(1)


#: The linkage budgets registered as their own defenses, mirroring ``DPMLM_SWEEP_EPSILONS``. Each
#: writes its own parquet and its own results directory, and each caches separately (the budget is in
#: ``params()``), which is what turns "a curve, not a point" into ordinary registry entries.
#: Keep in sync with experiments/run_loo_unlink.sbatch's BUDGETS.
LOO_UNLINK_BUDGETS = (0.1, 0.2, 0.3, 0.5, 0.7)


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
