"""Harrier: an instruction-conditioned embedder, run locally and offline.

Harrier is the *attacker* inside the leave-one-out defense (:mod:`prompt_anonymity.defenses.loo_unlink`).
The defense deletes a span, re-embeds, and measures how far the prompt moved away from its author's
other prompts -- which needs an embedding call per span per prompt, re-run after every edit. That is
roughly quadratic in span count, and the budget sweep multiplies it by the number of operating
points, so a single prompt can cost hundreds of forward passes. Metered pricing makes the method
unaffordable at exactly the scale where it becomes useful; local inference makes the call count
irrelevant. **Every embedding in the defense pipeline is this featurizer.** The paid Gemini channel
(:mod:`.gemini_embedding`) grades the finished text and never touches the scoring loop.

Registered three ways, which is how the instruction A/B gets run:

``harrier``
    The task-description phrasing. Instruction-tuned embedders are generally trained on task
    *descriptions* rather than imperatives, so this is expected to condition more effectively.
``harrier_imperative``
    The bare imperative. Kept because it is the phrasing the design spec proposed, and "the obvious
    phrasing is worse" is only demonstrable if both are measured.
``harrier_plain``
    No instruction at all -- the control that says whether conditioning bought anything.

Run all three against undefended text and keep the strongest. This matters because the instruction
defines the *attacker's* strength, and a defense paper wants the strongest attacker available: a weak
attacker makes the defense look better than it is.

Two things to know before trusting a number from this file
----------------------------------------------------------

**The instruction template is a guess until someone checks the model card.** Instruction-tuned
decoder embedders use a specific wrapper -- typically an ``Instruct:``/``Query:`` structure -- and
concatenating the instruction as plain text conditions the model far less than the trained format
does. :data:`INSTRUCT_TEMPLATE` encodes the common convention and is recorded in :meth:`params`, so a
correction invalidates the cache rather than silently mixing two embedding spaces. **Verify it
against the published card before the first real run.**

**Instruction asymmetry is not expressible in this pipeline, and that is a real deviation.** These
models conventionally instruct the query side and leave documents unconditioned. Same-author
retrieval is symmetric -- both sides are user prompts of the same kind -- so instructing both is the
defensible choice here anyway. But it is also the only available one: ``compute_features`` writes one
feature parquet per split and ``run_experiment.load_documents_and_features`` slices *both* attack
sides out of that single file, so there is no seam at which a known-side and an unknown-side encoding
could differ. Supporting the asymmetric setup would mean teaching the runner to read two feature
files. Report the symmetric result as what it is.

Offline by construction
-----------------------

The weights are expected to be on the cluster already. The checkpoint resolves through the same
``$ENV`` -> ``models.toml`` ladder the vLLM defenses use
(:func:`~prompt_anonymity.defenses._backends.model_path`), and every ``from_pretrained`` call passes
``local_files_only=True`` so a mis-specified path fails immediately and loudly instead of silently
attempting a download and hanging on a compute node with no outbound network. Set ``$HARRIER_MODEL``
in your ``.env``; do not put a machine-specific path in the committed ``models.toml``.
"""

from __future__ import annotations

import numpy as np

from .base import Featurizer

#: Hub id, used only as the ``models.toml`` default and as cache-key identity. Nothing downloads it:
#: see the module docstring.
DEFAULT_MODEL_ID = "microsoft/harrier-oss-v1-0.6b"

#: Environment override for the checkpoint location, read before ``models.toml``.
MODEL_ENV_VAR = "HARRIER_MODEL"

#: ⚠️ VERIFY AGAINST THE PUBLISHED MODEL CARD BEFORE THE FIRST RUN. This is the convention
#: instruction-tuned decoder embedders share, not a value read off Harrier's own card. It is part of
#: :meth:`HarrierFeaturizer.params`, so correcting it invalidates every cached vector instead of
#: mixing two embedding spaces in one feature file -- which is the failure this constant exists to
#: make impossible rather than merely unlikely.
INSTRUCT_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"

#: Task-description phrasing (registry name ``harrier``).
TASK_INSTRUCTION = "Given a chat prompt, retrieve other prompts written by the same author"

#: Bare imperative (registry name ``harrier_imperative``).
IMPERATIVE_INSTRUCTION = "attempt to relink author ID"

#: Truncation window, in tokens. Matched to :mod:`.gemini_embedding`'s 8,192-token window on purpose:
#: the headline claim is a comparison between the two attackers, and an encoder that reads twice as
#: much of each document as the other is not measuring the same task. Raise it only if the Gemini
#: side moves too.
DEFAULT_MAX_TOKENS = 8192

#: Documents per forward pass. A memory knob only -- it never changes a vector, so it is excluded
#: from :meth:`HarrierFeaturizer.params` and runs with different batch sizes share one cache.
DEFAULT_BATCH_SIZE = 32


class HarrierFeaturizer(Featurizer):
    """Embed each conversation with a local Harrier checkpoint under a fixed instruction.

    Pair with ``--metric cosine``: the vectors are L2-normalized (see :meth:`featurize`), so cosine
    is a plain dot product and Euclidean distance is a monotone function of it.
    """

    name = "harrier"
    version = "1"

    def __init__(self, *, instruction: str | None = TASK_INSTRUCTION,
                 model: str | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        #: ``None`` means "no conditioning at all" -- a genuinely different input string, not an
        #: empty instruction, which would still wrap the text in the template's scaffolding.
        self.instruction = instruction
        self.model = model
        self.max_tokens = int(max_tokens)
        self.batch_size = max(1, int(batch_size))
        if self.max_tokens < 16:
            raise ValueError(f"max_tokens must be at least 16, got {self.max_tokens}.")
        self._tokenizer = None
        self._model = None
        self._device = None

    def params(self) -> dict:
        """Everything that changes a vector. ``batch_size`` is excluded (see its constant).

        ``model_id`` is the *configured* identity rather than the resolved filesystem path: two
        machines mirroring the same checkpoint at different paths should share a cache namespace,
        and a path is not a fact about the embedding space. The template is included because a
        correction to it must invalidate every cached vector.
        """
        return {
            "model_id": self.model or DEFAULT_MODEL_ID,
            "instruction": self.instruction,
            "instruct_template": INSTRUCT_TEMPLATE if self.instruction else None,
            "max_tokens": self.max_tokens,
            "pooling": "last_token",
        }

    # --- model loading -------------------------------------------------------

    def checkpoint(self) -> str:
        """The resolved local checkpoint directory. Raises rather than falling back to a download."""
        # Imported here rather than at module scope so importing the features registry does not pull
        # in the defenses package; `model_path` reads .env and models.toml, nothing heavier.
        from ..defenses._backends import model_path, resolve_model_path

        return resolve_model_path(self.model or model_path(self.name, MODEL_ENV_VAR))

    def _ensure_model(self):
        """Load tokenizer and checkpoint on first use; returns ``(tokenizer, model, device)``."""
        if self._model is None:
            # Lazy, like luar/style_distance: only selecting this featurizer needs torch.
            import torch
            from transformers import AutoModel, AutoTokenizer

            from ..defenses._backends import gpu_dtype

            path = self.checkpoint()
            print(f"[{self.name}] loading {path} (local_files_only)")
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            # Last-token pooling requires the real final token to sit at index -1, which is only
            # true when padding is on the LEFT. Getting this wrong does not crash -- it silently
            # pools the pad token for every sequence shorter than the batch maximum, which looks
            # like a model that mysteriously cannot distinguish short documents.
            self._tokenizer.padding_side = "left"
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = AutoModel.from_pretrained(
                path, local_files_only=True, dtype=gpu_dtype(torch),
            ).to(self._device).eval()
        return self._tokenizer, self._model, self._device

    # --- encoding ------------------------------------------------------------

    def render(self, text: str) -> str:
        """The exact string handed to the tokenizer, instruction wrapper included."""
        if not self.instruction:
            return text
        return INSTRUCT_TEMPLATE.format(instruction=self.instruction, text=text)

    def featurize(self, texts) -> np.ndarray:
        import torch

        tokenizer, model, device = self._ensure_model()
        rendered = [self.render(text or "") for text in texts]
        vectors: list[np.ndarray] = []

        # Longest-first batching: sequences are padded to the batch maximum, so grouping similar
        # lengths together keeps the padding (and therefore the wasted FLOPs) small. Order is
        # restored afterwards, because the caller maps rows back to documents by position.
        order = sorted(range(len(rendered)), key=lambda i: len(rendered[i]), reverse=True)
        for start in range(0, len(order), self.batch_size):
            batch = [rendered[i] for i in order[start:start + self.batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=self.max_tokens, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
            # Left padding puts the final real token last for every row in the batch.
            pooled = hidden[:, -1]
            # Normalize once. The checkpoint is documented as emitting L2-normalized vectors, in
            # which case this is exactly a no-op; doing it anyway costs nothing and means a
            # checkpoint that does NOT normalize cannot quietly hand the scale-sensitive attacks
            # (logistic, wccn, plda) a magnitude that varies with document length.
            pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=1)
            vectors.append(pooled.cpu().numpy())

        stacked = np.concatenate(vectors, axis=0) if vectors else np.empty((0, 0), dtype=float)
        result = np.empty_like(stacked)
        result[np.asarray(order, dtype=int)] = stacked
        return result


class HarrierImperativeFeaturizer(HarrierFeaturizer):
    """``harrier`` under the bare imperative instruction. See the module docstring's A/B note."""

    name = "harrier_imperative"

    def __init__(self, **options):
        options.setdefault("instruction", IMPERATIVE_INSTRUCTION)
        super().__init__(**options)


class HarrierPlainFeaturizer(HarrierFeaturizer):
    """``harrier`` with no instruction: the unconditioned control for the A/B."""

    name = "harrier_plain"

    def __init__(self, **options):
        options["instruction"] = None
        super().__init__(**options)
