"""Is the rewrite still well-formed text? Perplexity under a small multilingual language model.

This is PAN's third obfuscation axis. Their triad is **safe** (no longer attributable), **sound**
(the content survives) and **sensible** (the output is well-formed and inconspicuous); the attacks
measure the first, :mod:`.prompt_judge` and :mod:`.semantic` the second, and nothing measured the
third until this module. It is worth having separately because a rewrite can be perfectly faithful
and still unusable -- degenerate repetition, a collapsed sentence, a model that lost the plot --
and a content metric will happily call that preserved.

**The score is a ratio, not an absolute.** Perplexity is dominated by what a text is *about*: a
dense stack trace scores far worse than small talk under any language model, so an absolute number
mostly ranks the corpus by topic. What is meaningful is each document against *itself*:
``ppl_ratio = perplexity(rewrite) / perplexity(original)``. Content cancels, and what is left is
what the rewrite did to the text's well-formedness. Above 1 is less fluent than the original,
below 1 is more (which happens, and is not automatically good -- a defense that replaces a
specific technical question with a bland paraphrase lowers perplexity while destroying utility,
which is precisely why this is reported *beside* the content metrics and never instead of them).

**Aggregated as a geometric mean**, because the quantity is a ratio: a document that doubles its
perplexity (2.0) and one that halves it (0.5) should cancel, and under an arithmetic mean they
average to 1.25 -- a fictitious 25% degradation. The geometric mean is equivalently the arithmetic
mean of the log ratios, which is how it is computed.

**Multilingual by requirement.** The default is a small Qwen3 *base* checkpoint: these corpora
contain Chinese conversations, and an English-only LM would report every one of them as
catastrophically disfluent in both versions -- the ratio would partly cancel that, but not
reliably. See :data:`DEFAULT_FLUENCY_MODEL` for why it is a base model rather than an instruct one.

**CoLA acceptability is deliberately not implemented**, although the StyleRemix defense ships one
upstream. CoLA is an English grammatical-acceptability corpus; running it on this corpus would
score the Chinese half as unacceptable regardless of what the defense did to it. Perplexity under
a multilingual LM measures the same underlying property without that failure. If an English-only
split is ever scored in isolation, adding CoLA becomes reasonable.

**Results are not cached, unlike the LLM judge's.** A cache exists on the base class and these
metrics deliberately do not use it: a cache miss here is GPU-minutes, not a re-billed API call, so
the argument that justifies caching paid verdicts does not carry over. The cost of that choice is
**resumability**, and it has a concrete limit -- 50 conversations take well under a minute on an
A16, so a full swe-chat split (4,334) is minutes, but WildChat (172,509) is hours and will not fit
one ``--qos=short`` allocation. Shard it by passing a subset, or add caching, before running the
larger corpus.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._local import resolve_checkpoint, select_device, torch_dtype
from ._parsing import render_turns
from .base import DEFAULT_SEED, UtilityMetric, UtilityResult

#: Small multilingual causal LM used as the fluency reference. Deliberately small: perplexity
#: *ranking* is stable across model sizes, and this runs over whole corpora.
#:
#: **A base model, not an instruct one.** Perplexity here is meant to answer "is this text
#: well-formed?", and an instruction-tuned checkpoint answers a subtly different question: its
#: distribution has been reshaped toward assistant-style replies, so it finds polished prose
#: unusually likely and terse developer shorthand unusually surprising -- which is exactly the
#: contrast this metric measures, now partly attributable to the scorer. The base model is the
#: neutral language model the ratio assumes.
#:
#: This is a HuggingFace repo id on purpose: it is committed code, so it must work on a machine
#: with no local mirror. Point ``$UTILITY_FLUENCY_MODEL`` (in ``.env``) or a ``[fluency]`` entry in
#: a ``models.toml`` of your own at a copy already on disk -- on this machine that is
#: ``/datasets/ai/qwen3/hub/models--Qwen--Qwen3-0.6B-Base``, and
#: :func:`~prompt_anonymity.defenses._backends.resolve_model_path` follows a hub cache directory
#: like that one into its snapshot.
DEFAULT_FLUENCY_MODEL = "Qwen/Qwen3-0.6B-Base"
FLUENCY_MODEL_ENV = "UTILITY_FLUENCY_MODEL"

#: Token cap per scored text. Perplexity is a per-token average, so a truncated document is still
#: a valid (if partial) measurement -- unlike a content metric, where truncation loses content.
DEFAULT_MAX_LENGTH = 1024

#: Texts per forward pass. Small because the memory cost is ``batch x sequence x vocab``, and a
#: multilingual vocabulary makes that the dominant allocation of the whole metric.
DEFAULT_BATCH_SIZE = 4

#: Sequence positions whose logits are upcast to fp32 at once. The one knob that decides whether
#: this metric fits on a card: see :meth:`_PerplexityScorer.perplexity` for why the obvious
#: unchunked implementation needs ~10 GB of scratch and this one needs a few hundred MB.
LOGIT_CHUNK_TOKENS = 128

#: A rewrite whose perplexity exceeds the original's by more than this is counted in
#: ``degraded_rate``. 1.5x is well outside ordinary paraphrase noise while not firing on every
#: reworded sentence. Post-hoc, so deliberately **not** in the cache key.
DEGRADED_RATIO_THRESHOLD = 1.5

#: Bump to invalidate cached scores. ``"1"`` is the initial implementation.
FLUENCY_UTILITY_VERSION = "1"


class _PerplexityScorer:
    """A causal LM used only to score, never to generate."""

    def __init__(self, model: str, max_length: int, batch_size: int, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = select_device(device)
        self.max_length = max_length
        self.batch_size = batch_size
        print(f"Utility(fluency): loading '{model}' on {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        if self.tokenizer.pad_token is None:
            # Needed only to make batching legal; padded positions are masked out of the loss
            # below, so the choice of pad token cannot affect a score.
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model, dtype=torch_dtype(self.device)).to(self.device).eval()

    def perplexity(self, texts: list[str]) -> list[float]:
        """Token-level perplexity of each text; ``nan`` for text too short to score.

        Computed by hand rather than through the model's own ``labels=`` loss because that averages
        over the whole batch and returns one number -- useless here, where the point is the
        per-document value.

        **The arithmetic is chunked along the sequence, and that is not an optimisation.** The
        logits are ``[batch, sequence, vocab]``, and a modern multilingual vocabulary is enormous
        (151,936 for Qwen3): at batch 4 and 1,024 tokens that single tensor is 1.2 GB in fp16. The
        obvious implementation -- upcast it and take ``log_softmax`` -- allocates two *more* copies
        at fp32, about 5 GB each, and OOMs a 15 GB card on the first batch (observed). Instead the
        sequence is walked in slices, and only a slice is ever upcast; and rather than a full
        ``log_softmax`` (which materialises a probability for all 152k tokens when 1 is wanted),
        each target's log-probability comes from ``logit[target] - logsumexp(logits)``, which is
        the same value with no vocabulary-sized intermediate.
        """
        scores: list[float] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start:start + self.batch_size]
            encoded = self.tokenizer(batch, truncation=True, padding=True,
                                     max_length=self.max_length,
                                     return_tensors="pt").to(self.device)
            input_ids = encoded["input_ids"]
            mask = encoded["attention_mask"]
            with self.torch.no_grad():
                logits = self.model(**encoded).logits
            # Shift: position t's logits predict token t+1, so the first token is never predicted
            # and never scored.
            targets = input_ids[:, 1:]
            target_mask = mask[:, 1:].float()
            total = self.torch.zeros(targets.shape[0], device=logits.device, dtype=self.torch.float32)
            for begin in range(0, targets.shape[1], LOGIT_CHUNK_TOKENS):
                window = slice(begin, begin + LOGIT_CHUNK_TOKENS)
                chunk = logits[:, :-1][:, window].float()
                chunk_targets = targets[:, window]
                token_log_prob = (chunk.gather(-1, chunk_targets.unsqueeze(-1)).squeeze(-1)
                                  - chunk.logsumexp(dim=-1))
                total += (token_log_prob * target_mask[:, window]).sum(dim=1)
                del chunk, token_log_prob
            counts = target_mask.sum(dim=1)
            for value, count in zip(total.cpu().tolist(), counts.cpu().tolist()):
                scores.append(float(np.exp(-value / count)) if count > 0 else float("nan"))
        return scores


@dataclass
class FluencyUtilityResult(UtilityResult):
    """Well-formedness of the rewrite relative to the original.

    Attributes
    ----------
    geometric_mean_ratio : float
        Geometric mean of ``perplexity(rewrite) / perplexity(original)``. 1.0 means the rewrite is
        exactly as well-formed as what it replaced; above 1 is worse.
    median_ratio : float
        The median of the same quantity -- read it next to the mean, since a handful of degenerate
        rewrites can move a mean a long way and this metric exists to find exactly those.
    degraded_rate : float
        Share of conversations whose perplexity ratio exceeds
        :data:`DEGRADED_RATIO_THRESHOLD`.
    mean_original_ppl, mean_defended_ppl : float
        Absolute perplexities, reported only as a sanity check on the model and the corpus -- they
        are dominated by subject matter, which is why the ratio is the headline.
    n, n_unchanged : int
        Conversations scored, and how many were byte-identical (ratio 1.0, no forward pass).
    table : pandas.DataFrame
        Per-conversation detail: ``conv_id``, ``original_ppl``, ``defended_ppl`` and ``ppl_ratio``.
        Only the ratio reaches the output file -- the two absolutes are dominated by subject
        matter, so they belong next to a specific row being investigated, not in a comparison
        table.
    """

    geometric_mean_ratio: float
    median_ratio: float
    degraded_rate: float
    mean_original_ppl: float
    mean_defended_ppl: float
    n: int
    n_unchanged: int
    table: pd.DataFrame
    sampled_from: int | None = None

    #: The ratio is the measurement; the two absolute perplexities it is built from stay on
    #: ``table``, since a number dominated by subject matter is not comparable across rows.
    score_columns = {"ppl_ratio": "ppl_ratio"}

    def summary(self) -> str:
        return (
            f"{self.sample_note()}Fluency utility  n={self.n}  "
            f"ppl_ratio geo-mean={self.geometric_mean_ratio:.3f}  "
            f"median={self.median_ratio:.3f}  "
            f"degraded(>{DEGRADED_RATIO_THRESHOLD}x)={self.degraded_rate:.4f}  "
            f"(orig ppl={self.mean_original_ppl:.1f}, defended ppl={self.mean_defended_ppl:.1f}, "
            f"unchanged={self.n_unchanged})"
        )


class FluencyUtility(UtilityMetric):
    """Score how well-formed a rewrite is relative to its original, by perplexity ratio.

    Parameters
    ----------
    model : str
        Causal LM checkpoint; defaults to a small multilingual one.
    max_length, batch_size : int
        Token cap per text and texts per forward pass.
    device : str, optional
        Force ``"cuda"`` / ``"cpu"``; otherwise a GPU is used when visible.
    """

    name = "fluency_utility"
    version = FLUENCY_UTILITY_VERSION

    def __init__(self, *, model: str | None = None, max_length: int = DEFAULT_MAX_LENGTH,
                 batch_size: int = DEFAULT_BATCH_SIZE, device: str | None = None):
        self.model = model or resolve_checkpoint("fluency", FLUENCY_MODEL_ENV,
                                                 DEFAULT_FLUENCY_MODEL)
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device
        self._scorer: _PerplexityScorer | None = None

    def params(self) -> dict:
        return {"model": self.model, "max_length": self.max_length}

    def score(self, data, *, cache_dir, reference, side: str = "unknown",
              limit: int | None = None, seed: int = DEFAULT_SEED) -> FluencyUtilityResult:
        """Score ``data`` (post-defense) against ``reference`` (pre-defense)."""
        sides = self._load_sides(data, reference, side, limit=limit, seed=seed)
        ids = getattr(data, f"{side}_ids", None)

        units = []
        for position, row in enumerate(sides.indices):
            original, defended = sides.original[position], sides.defended[position]
            if not original.strip():
                continue
            units.append((row, ids[row] if ids is not None else row, original, defended))
        n = len(units)

        original_ppl: list[float] = [float("nan")] * n
        defended_ppl: list[float] = [float("nan")] * n
        ratios: list[float] = [1.0] * n
        changed = [u for u, (_, _, original, defended) in enumerate(units) if defended != original]

        if changed:
            scorer = self._scorer or _PerplexityScorer(self.model, self.max_length,
                                                       self.batch_size, self.device)
            self._scorer = scorer
            print(f"Utility(fluency): scoring {len(changed):,} changed conversations")
            # Rendered without [Turn n] labels: they are identical on both sides, so they add the
            # same highly-predictable tokens to each and compress every ratio toward 1.
            originals = [render_turns(units[u][2], drop_blank=True) for u in changed]
            defendeds = [render_turns(units[u][3], drop_blank=True) for u in changed]
            for u, value in zip(changed, scorer.perplexity(originals)):
                original_ppl[u] = value
            for u, value in zip(changed, scorer.perplexity(defendeds)):
                defended_ppl[u] = value
            for u in changed:
                if np.isfinite(original_ppl[u]) and original_ppl[u] > 0:
                    ratios[u] = defended_ppl[u] / original_ppl[u]
                else:
                    ratios[u] = float("nan")

        finite = np.array([r for r in ratios if np.isfinite(r) and r > 0], dtype=float)
        geometric = float(np.exp(np.mean(np.log(finite)))) if finite.size else float("nan")
        # No conversation text, for the reason in `semantic.py`: it is already on disk and nothing
        # downstream reads it back from here.
        table = pd.DataFrame({
            "conv_id": [u[1] for u in units],
            "original_ppl": original_ppl,
            "defended_ppl": defended_ppl,
            "ppl_ratio": ratios,
        })
        return FluencyUtilityResult(
            geometric_mean_ratio=geometric,
            median_ratio=float(np.median(finite)) if finite.size else float("nan"),
            degraded_rate=(float((finite > DEGRADED_RATIO_THRESHOLD).mean())
                           if finite.size else float("nan")),
            mean_original_ppl=float(np.nanmean(original_ppl)) if n else float("nan"),
            mean_defended_ppl=float(np.nanmean(defended_ppl)) if n else float("nan"),
            n=n, n_unchanged=n - len(changed),
            table=table, sampled_from=sides.sampled_from,
        )


def fluency_utility(data, *, cache_dir, reference, side: str = "unknown",
                    limit: int | None = None, seed: int = DEFAULT_SEED,
                    **kwargs) -> FluencyUtilityResult:
    """Convenience wrapper: score ``data`` vs. ``reference`` with a default :class:`FluencyUtility`."""
    return FluencyUtility(**kwargs).score(
        data, cache_dir=cache_dir, reference=reference, side=side, limit=limit, seed=seed
    )
