"""Semantic preservation without a judge: multilingual NLI entailment and BERTScore-recall.

The conversation judge (:mod:`.prompt_judge`) is the accurate measurement but the expensive one --
one API call per conversation. This metric is the complement: local models, no API, so it can run
over every document in a split. The intended workflow is to score the whole corpus here, score a
sample with the judge, and correlate the two, turning the judge into a calibration set for the
free metric.

Two scores, both deliberately **recall-oriented**.

``entailment``
    Multilingual NLI with **premise = the rewrite, hypothesis = the original**, reported as
    ``P(entailment)``: does the rewritten text still support everything the original said?

    **This is the opposite direction from PAN's "soundness"**, which asks whether the obfuscated
    text is entailed *by* the original -- a precision check that fires when a rewrite adds
    unsupported content. That's the wrong test here: material the rewrite invents is outside the
    judgement (see ``conversation_judge.yaml``'s third invariant), so only dropped or altered
    content should count, and running it this direction encodes that in the metric itself.

``bertscore_recall``
    Greedy-matched cosine similarity over contextual token embeddings, in the recall direction: for
    each *original* token, its best match anywhere in the rewrite. Extra tokens in the rewrite are
    never visited, so additions cannot cost points.

**Why two scores and not one.** They fail differently. NLI is sensitive to negation and dropped
constraints but degrades on long technical turns; BERTScore is robust to length but is a pure
similarity measure, so it marks a rewrite down for a style shift as much as for a content loss.
Together, a defense that scores well on both is preserving content, and a split between them is
worth inspecting by hand.

**Raw BERTScore is not interpretable on its own**, since unrelated text in the same language still
scores high. The run also computes a **corpus-local baseline** over deliberately mismatched pairs
and reports ``bertscore_rescaled = (raw - baseline) / (1 - baseline)``, putting "no better than an
unrelated conversation" at 0 and "identical" at 1. The baseline is corpus-specific, so rescaled
scores from different splits aren't comparable unless their baselines match.

**Scoring is per turn where the structure allows it.** Per-turn defenses preserve turn count, so
turn *i* of the original is compared against turn *i* of the rewrite, weighted by original turn
length. That keeps each NLI pair inside the model's token window and localises a loss to the turn
that caused it. A defense that doesn't preserve turn count falls back to a single whole-cell
comparison, recorded in the ``aligned`` column.

**Results are not cached, unlike the LLM judge's.** A miss here is GPU-minutes, not a re-billed API
call, so caching paid verdicts is not the same argument. The cost is resumability: shard a large
split into a subset per run rather than scoring it in one long job.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._local import batched, resolve_checkpoint, select_device, torch_dtype
from ._parsing import render_turns
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult
from ...defenses._backends import split_turns

#: Multilingual NLI checkpoint. mDeBERTa-v3 fine-tuned on XNLI + multilingual-NLI covers the
#: languages these corpora actually contain; an English-only model (``roberta-large-mnli``, the
#: locally cached ``microsoft/deberta-base-mnli``) would score every Chinese conversation as a
#: total loss. Override with ``$UTILITY_NLI_MODEL`` or a ``[semantic_nli]`` models.toml entry.
DEFAULT_NLI_MODEL = "MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"
NLI_MODEL_ENV = "UTILITY_NLI_MODEL"

#: Multilingual encoder for BERTScore. XLM-R large is the standard multilingual choice and is
#: already mirrored on this machine. Override with ``$UTILITY_EMBED_MODEL`` / ``[semantic_embed]``.
DEFAULT_EMBED_MODEL = "xlm-roberta-large"
EMBED_MODEL_ENV = "UTILITY_EMBED_MODEL"

#: Which transformer layer to take token embeddings from. BERTScore uses a tuned intermediate
#: layer rather than the (too task-specialised) final one; 17 is the reference implementation's
#: choice for XLM-R large. The single knob that most changes the absolute numbers, which is also
#: why the corpus-local baseline matters.
DEFAULT_EMBED_LAYER = 17

#: Token cap per NLI pair and per embedded text. 512 is mDeBERTa's window; turns longer than this
#: are truncated, which the ``truncated_turns`` column records per conversation rather than
#: leaving silent -- see :meth:`_Scorer.truncated` for why that matters more than it sounds.
DEFAULT_MAX_LENGTH = 512

#: Pairs scored per forward pass. Memory, not correctness -- lower it if a long-turn corpus OOMs.
DEFAULT_BATCH_SIZE = 16

#: Mismatched pairs drawn to estimate the BERTScore baseline. Enough to be stable to ~0.01 without
#: costing a meaningful share of the run; the estimate is reported so it can be judged.
DEFAULT_BASELINE_PAIRS = 200

#: Bump to invalidate cached scores. ``"1"`` is the initial implementation.
SEMANTIC_UTILITY_VERSION = "1"


def _entailment_index(config) -> int:
    """Which output logit is the *entailment* class for this checkpoint.

    Read from ``config.id2label`` rather than assumed: NLI checkpoints genuinely disagree about
    label order (MNLI-trained models commonly use contradiction/neutral/entailment, others the
    reverse), and guessing wrong silently reports the contradiction probability as if it were
    faithfulness -- a number that looks plausible and is exactly backwards.
    """
    labels = {index: str(label).lower() for index, label in (config.id2label or {}).items()}
    for index, label in labels.items():
        if "entail" in label:
            return int(index)
    raise RuntimeError(
        f"cannot find an entailment label in the NLI model's id2label ({labels}); "
        "point $UTILITY_NLI_MODEL at a standard 3-way NLI checkpoint."
    )


class _Scorer:
    """Holds the two loaded models. Built once per run, on first real need."""

    def __init__(self, nli_model: str, embed_model: str, embed_layer: int,
                 max_length: int, batch_size: int, device: str | None = None):
        import torch
        from transformers import (AutoConfig, AutoModel, AutoModelForSequenceClassification,
                                  AutoTokenizer)

        self.torch = torch
        self.device = select_device(device)
        self.dtype = torch_dtype(self.device)
        self.max_length = max_length
        self.batch_size = batch_size
        self.embed_layer = embed_layer

        print(f"Utility(semantic): loading NLI '{nli_model}' and encoder '{embed_model}' "
              f"on {self.device}")
        self.nli_tokenizer = AutoTokenizer.from_pretrained(nli_model)
        self.nli = AutoModelForSequenceClassification.from_pretrained(
            nli_model, dtype=self.dtype).to(self.device).eval()
        self.entail_index = _entailment_index(AutoConfig.from_pretrained(nli_model))

        self.embed_tokenizer = AutoTokenizer.from_pretrained(embed_model)
        self.embed = AutoModel.from_pretrained(
            embed_model, dtype=self.dtype, output_hidden_states=True).to(self.device).eval()

    def truncated(self, premises: list[str], hypotheses: list[str]) -> list[bool]:
        """Which pairs do not fit in the NLI window, and are therefore only partly compared.

        Worth its own pass (tokenizing twice is cheap next to a forward pass) because truncation is
        not a neutral loss of precision here: it cuts the premise and the hypothesis at *different*
        points in the content, so the surviving rewrite may genuinely not contain the surviving
        original, and the model correctly reports low entailment for a comparison that was never
        made. Entailment is far more sensitive to this than BERTScore recall, so a long-document
        corpus needs this column read before its entailment figure is believed.
        """
        flags: list[bool] = []
        for premise, hypothesis in zip(premises, hypotheses):
            length = len(self.nli_tokenizer(premise, hypothesis)["input_ids"])
            flags.append(length > self.max_length)
        return flags

    def entailment(self, premises: list[str], hypotheses: list[str]) -> list[float]:
        """``P(entailment)`` that each premise supports its hypothesis."""
        scores: list[float] = []
        for start in range(0, len(premises), self.batch_size):
            batch_p = premises[start:start + self.batch_size]
            batch_h = hypotheses[start:start + self.batch_size]
            encoded = self.nli_tokenizer(batch_p, batch_h, truncation=True, padding=True,
                                         max_length=self.max_length,
                                         return_tensors="pt").to(self.device)
            with self.torch.no_grad():
                logits = self.nli(**encoded).logits.float()
            probs = self.torch.softmax(logits, dim=-1)[:, self.entail_index]
            scores.extend(probs.cpu().tolist())
        return scores

    def _token_embeddings(self, texts: list[str]):
        """Per-text L2-normalised token embeddings from ``embed_layer``, special tokens dropped.

        Special tokens are removed because BERTScore matches *content*: ``<s>`` in the original
        would otherwise always find ``<s>`` in the rewrite and score a free 1.0, inflating every
        document by roughly one token's worth and inflating short ones most.
        """
        encoded = self.embed_tokenizer(texts, truncation=True, padding=True,
                                       max_length=self.max_length,
                                       return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            hidden = self.embed(**encoded).hidden_states[self.embed_layer].float()
        hidden = self.torch.nn.functional.normalize(hidden, dim=-1)
        special = self.torch.tensor(
            self.embed_tokenizer.all_special_ids, device=hidden.device)
        keep = encoded["attention_mask"].bool() & ~self.torch.isin(encoded["input_ids"], special)
        return [hidden[i][keep[i]] for i in range(len(texts))]

    def bertscore_recall(self, references: list[str], candidates: list[str]) -> list[float]:
        """For each reference token, its best cosine match in the candidate; averaged per pair.

        Recall only. Precision (the same thing from the candidate's side) is what penalises a
        candidate for containing material the reference lacks, which this project counts as
        neither a loss nor a gain -- so computing it would only invite reporting it.
        """
        scores: list[float] = []
        for ref_batch, cand_batch in zip(batched(references, self.batch_size),
                                         batched(candidates, self.batch_size)):
            ref_embeddings = self._token_embeddings(ref_batch)
            cand_embeddings = self._token_embeddings(cand_batch)
            for reference, candidate in zip(ref_embeddings, cand_embeddings):
                if reference.numel() == 0 or candidate.numel() == 0:
                    scores.append(float("nan"))
                    continue
                similarity = reference @ candidate.T
                scores.append(float(similarity.max(dim=1).values.mean()))
        return scores


@dataclass
class SemanticUtilityResult(UtilityResult):
    """Reference-free semantic preservation for one (original, defended) comparison.

    Attributes
    ----------
    mean_entailment : float
        Mean ``P(rewrite entails original)`` over conversations, in ``[0, 1]``.
    mean_bertscore_recall : float
        Mean raw BERTScore recall. **Not interpretable alone** -- see ``bertscore_baseline``.
    bertscore_baseline : float
        The same score over deliberately mismatched pairs from this run: the value a rewrite gets
        for being the same language and register as the original while sharing none of its content.
    mean_bertscore_rescaled : float
        ``(raw - baseline) / (1 - baseline)``, so 0 is "no better than an unrelated conversation"
        and 1 is "identical". Corpus-local, so compare it only within a split.
    n, n_unchanged, n_aligned, n_truncated : int
        Conversations scored; how many the defense left byte-identical (scored 1.0 without a
        forward pass); how many were compared turn-by-turn rather than as a single cell; and how
        many had at least one turn pair too long for the NLI window. **Read the last one before
        trusting a low entailment figure** -- a truncated pair is a comparison that was only partly
        made, and entailment is the score that suffers for it (see :meth:`_Scorer.truncated`).
    table : pandas.DataFrame
        Per-conversation detail: ``conv_id``, ``entailment``, ``bertscore_recall``, ``n_turns``,
        ``truncated_turns`` and ``aligned``. Only the two scores reach the output file; the rest
        are here for reading a surprising number back to its cause.
    """

    mean_entailment: float
    mean_bertscore_recall: float
    bertscore_baseline: float
    mean_bertscore_rescaled: float
    n: int
    n_unchanged: int
    n_aligned: int
    n_truncated: int
    table: pd.DataFrame
    sampled_from: int | None = None

    #: Both scores go to the file, because they fail differently and a split between them is the
    #: signal to inspect by hand -- reporting either alone would hide exactly that.
    score_columns = {"entailment": "entailment", "bertscore_recall": "bertscore_recall"}

    def summary(self) -> str:
        return (
            f"{self.sample_note()}Semantic utility  n={self.n}  "
            f"entailment={self.mean_entailment:.3f}  "
            f"bertscore_recall={self.mean_bertscore_recall:.3f} "
            f"(baseline={self.bertscore_baseline:.3f}, "
            f"rescaled={self.mean_bertscore_rescaled:.3f})  "
            f"(unchanged={self.n_unchanged}, turn-aligned={self.n_aligned}/{self.n}, "
            f"truncated={self.n_truncated}/{self.n})"
        )


class SemanticUtility(UtilityMetric):
    """Score semantic preservation with local NLI and embedding models -- no API, no cost per row.

    Parameters
    ----------
    nli_model, embed_model : str
        Checkpoints for the two scorers; both default to multilingual models (see the module
        docstring for why that is a requirement rather than a preference).
    embed_layer : int
        Which layer to take BERTScore embeddings from.
    max_length, batch_size : int
        Token cap per pair and pairs per forward pass.
    baseline_pairs : int
        Mismatched pairs used to estimate the BERTScore baseline; ``0`` skips it, leaving the
        rescaled score undefined.
    device : str, optional
        Force ``"cuda"`` / ``"cpu"``; otherwise a GPU is used when visible.
    """

    name = "semantic_utility"
    version = SEMANTIC_UTILITY_VERSION

    def __init__(self, *, nli_model: str | None = None, embed_model: str | None = None,
                 embed_layer: int = DEFAULT_EMBED_LAYER, max_length: int = DEFAULT_MAX_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE,
                 baseline_pairs: int = DEFAULT_BASELINE_PAIRS, device: str | None = None):
        self.nli_model = nli_model or resolve_checkpoint("semantic_nli", NLI_MODEL_ENV,
                                                         DEFAULT_NLI_MODEL)
        self.embed_model = embed_model or resolve_checkpoint("semantic_embed", EMBED_MODEL_ENV,
                                                             DEFAULT_EMBED_MODEL)
        self.embed_layer = embed_layer
        self.max_length = max_length
        self.batch_size = batch_size
        self.baseline_pairs = baseline_pairs
        self.device = device
        self._scorer: _Scorer | None = None

    def params(self) -> dict:
        # Everything that changes a score. batch_size and baseline_pairs are absent: the first is
        # pure memory management, and the second only affects a figure derived after the fact.
        return {"nli_model": self.nli_model, "embed_model": self.embed_model,
                "embed_layer": self.embed_layer, "max_length": self.max_length}

    def _get_scorer(self) -> _Scorer:
        if self._scorer is None:
            self._scorer = _Scorer(self.nli_model, self.embed_model, self.embed_layer,
                                   self.max_length, self.batch_size, self.device)
        return self._scorer

    def _pairs(self, original: str, defended: str) -> tuple[list[str], list[str], bool]:
        """The (original, defended) text pairs to score, and whether turns were aligned.

        Rendered through :func:`~._parsing.render_turns` without labels: the ``[Turn n]`` prefixes
        the LLM judge needs are noise to a sentence-pair model, and would be matched against each
        other by BERTScore as if they were content.
        """
        original_turns = [t for t in split_turns(original) if t.strip()]
        defended_turns = [t for t in split_turns(defended) if t.strip()]
        if len(original_turns) == len(defended_turns) and original_turns:
            return original_turns, defended_turns, True
        return ([render_turns(original, drop_blank=True)],
                [render_turns(defended, drop_blank=True)], False)

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED) -> SemanticUtilityResult:
        """Score ``data`` (post-defense) against ``reference`` (pre-defense)."""
        sides = self._load_sides(data, reference, side, limit=limit, seed=seed)
        ids = getattr(data, f"{side}_ids", None)

        units = []  # (conv_index, conv_id, original, defended)
        for position, row in enumerate(sides.indices):
            original, defended = sides.original[position], sides.defended[position]
            if not original.strip():
                continue
            units.append((row, ids[row] if ids is not None else row, original, defended))
        n = len(units)

        # A byte-identical rewrite preserves everything by construction; scoring it would spend a
        # forward pass to rediscover 1.0 and, worse, would land slightly below it (a model's
        # entailment probability for a text against itself is not exactly 1).
        entailments: list[float] = [1.0] * n
        recalls: list[float] = [1.0] * n
        aligned: list[bool] = [True] * n
        turn_counts: list[int] = [0] * n
        truncated_counts: list[int] = [0] * n
        changed = [u for u, (_, _, original, defended) in enumerate(units) if defended != original]

        flat_original: list[str] = []
        flat_defended: list[str] = []
        spans: list[tuple[int, int]] = []
        for u in changed:
            original_parts, defended_parts, is_aligned = self._pairs(units[u][2], units[u][3])
            spans.append((len(flat_original), len(original_parts)))
            aligned[u] = is_aligned
            turn_counts[u] = len(original_parts)
            flat_original.extend(original_parts)
            flat_defended.extend(defended_parts)

        if flat_original:
            scorer = self._get_scorer()
            print(f"Utility(semantic): scoring {len(changed):,} changed conversations "
                  f"({len(flat_original):,} turn pairs)")
            # premise = rewrite, hypothesis = original: "does the rewrite still support this?"
            part_entail = scorer.entailment(flat_defended, flat_original)
            part_recall = scorer.bertscore_recall(flat_original, flat_defended)
            part_truncated = scorer.truncated(flat_defended, flat_original)

            for k, u in enumerate(changed):
                start, count = spans[k]
                # Weight turns by their length in characters: a one-word turn losing its meaning
                # matters less to the conversation than a paragraph doing so, and an unweighted
                # mean lets a corpus of short acknowledgements ("ok", "thanks") dominate.
                weights = np.array([max(1, len(flat_original[start + i])) for i in range(count)],
                                   dtype=float)
                weights /= weights.sum()
                entailments[u] = float(np.dot(weights, part_entail[start:start + count]))
                recalls[u] = float(np.nansum(weights * np.array(part_recall[start:start + count])))
                truncated_counts[u] = int(sum(part_truncated[start:start + count]))

        baseline = self._baseline(flat_original, flat_defended, seed)
        finite = np.array([r for r in recalls if np.isfinite(r)], dtype=float)
        mean_recall = float(finite.mean()) if finite.size else float("nan")
        rescaled = ((mean_recall - baseline) / (1 - baseline)
                    if np.isfinite(baseline) and baseline < 1 else float("nan"))

        # No conversation text: it is the bulk of the dataset, it is already in the parquet pair
        # this was loaded from, and holding two copies of a whole split here is what made a
        # full-corpus run expensive in memory for no reader.
        table = pd.DataFrame({
            "conv_id": [u[1] for u in units],
            "entailment": entailments,
            "bertscore_recall": recalls,
            "n_turns": turn_counts,
            "truncated_turns": truncated_counts,
            "aligned": aligned,
        })
        return SemanticUtilityResult(
            mean_entailment=float(np.mean(entailments)) if entailments else float("nan"),
            mean_bertscore_recall=mean_recall,
            bertscore_baseline=baseline,
            mean_bertscore_rescaled=rescaled,
            n=n, n_unchanged=n - len(changed), n_aligned=int(sum(aligned)),
            n_truncated=int(sum(1 for c in truncated_counts if c)),
            table=table, sampled_from=sides.sampled_from,
        )

    def _baseline(self, originals: list[str], defendeds: list[str], seed: int) -> float:
        """Mean BERTScore recall over deliberately mismatched pairs from this run.

        Estimated from the run's own text rather than a fixed constant so it reflects this
        corpus's language mix and register -- which is the whole point, since the number it
        calibrates is "how much better than an unrelated conversation is this rewrite?".
        """
        if not self.baseline_pairs or len(originals) < 2:
            return float("nan")
        rng = np.random.default_rng(seed)
        count = min(self.baseline_pairs, len(originals))
        left = rng.choice(len(originals), size=count, replace=False)
        # Rotate by one so no index is paired with itself: a "mismatched" pair that happens to be
        # a real pair would drag the baseline up toward the real score and shrink every rescaled
        # number toward zero.
        right = np.roll(left, 1)
        scores = self._get_scorer().bertscore_recall(
            [originals[i] for i in left], [defendeds[j] for j in right])
        finite = np.array([s for s in scores if np.isfinite(s)], dtype=float)
        return float(finite.mean()) if finite.size else float("nan")


def semantic_utility(data, *, cache_dir, reference, side: str = "unknown",
                     limit: int | None = None, seed: int = DEFAULT_SEED,
                     **kwargs) -> SemanticUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default :class:`SemanticUtility`."""
    return SemanticUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
