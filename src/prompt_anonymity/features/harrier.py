"""Harrier: an instruction-conditioned embedder, run locally and offline.

Harrier is the *attacker* inside the leave-one-out defense
(:mod:`prompt_anonymity.defenses.loo_unlink`), which re-embeds after every edit -- far too many
calls to run against a metered API. Local inference makes the call count irrelevant. **Every
embedding in the defense pipeline is this featurizer**; the paid Gemini channel
(:mod:`.gemini_embedding`) only grades the finished text.

Registered three ways, to A/B the conditioning instruction:

``harrier``
    Task-description phrasing (instruction-tuned embedders are typically trained on descriptions
    rather than imperatives).
``harrier_imperative``
    The bare imperative, kept so "the obvious phrasing is worse" is demonstrable rather than
    assumed.
``harrier_plain``
    No instruction -- the control for whether conditioning helps at all.

Run all three against undefended text and keep the strongest: the instruction defines the
*attacker's* strength, and a weak attacker makes a defense look better than it is.

**The instruction template is a guess until checked against the model card** -- instruction-tuned
decoder embedders use a specific wrapper, and plain concatenation conditions the model far less
than the trained format. :data:`INSTRUCT_TEMPLATE` is in :meth:`params`, so correcting it
invalidates the cache instead of silently mixing two embedding spaces.

**Instruction asymmetry is not expressible in this pipeline.** These models conventionally
instruct the query side only; here both known and unknown documents are user prompts of the same
kind, and both come from one feature parquet with no seam to instruct them differently. Report the
symmetric result as what it is.

Offline by construction
-----------------------

The checkpoint resolves through :func:`~prompt_anonymity.defenses._backends.model_checkpoint`
(``$HARRIER_MODEL`` -> the cluster's local mirror -> the HuggingFace cache), with
``local_files_only=True`` throughout so a cache miss fails loudly instead of hanging on a node
with no outbound network. Set ``$HARRIER_MODEL`` in ``.env`` rather than editing ``models.toml``
with a machine-specific path.

All three variants resolve through the **same** ``[harrier]`` section
(:attr:`HarrierFeaturizer.config_section`): the A/B varies the instruction, not the weights.
"""

from __future__ import annotations

import numpy as np

from .base import Featurizer

#: Hub id, used only as the ``models.toml`` default and as cache-key identity. Nothing downloads it:
#: see the module docstring.
DEFAULT_MODEL_ID = "microsoft/harrier-oss-v1-0.6b"

#: Environment override for the checkpoint location, read before ``models.toml``.
MODEL_ENV_VAR = "HARRIER_MODEL"

#: ⚠️ Verify against the published model card before the first run -- this is the shared convention,
#: not a value read off Harrier's own card. Part of :meth:`HarrierFeaturizer.params`, so correcting
#: it invalidates the cache rather than silently mixing two embedding spaces.
INSTRUCT_TEMPLATE = "Instruct: {instruction}\nQuery: {text}"

#: Task-description phrasing (registry name ``harrier``).
TASK_INSTRUCTION = "Given a chat prompt, retrieve other prompts written by the same author"

#: Bare imperative (registry name ``harrier_imperative``).
IMPERATIVE_INSTRUCTION = "attempt to relink author ID"

#: Truncation window, in tokens. Matched to :mod:`.gemini_embedding`'s window so the two attackers
#: read the same amount of each document; raise it only if the Gemini side moves too.
DEFAULT_MAX_TOKENS = 8192

#: Documents per forward pass. A memory knob only -- excluded from :meth:`HarrierFeaturizer.params`.
DEFAULT_BATCH_SIZE = 32


class HarrierFeaturizer(Featurizer):
    """Embed each conversation with a local Harrier checkpoint under a fixed instruction.

    Pair with ``--metric cosine``: the vectors are L2-normalized (see :meth:`featurize`), so cosine
    is a plain dot product and Euclidean distance is a monotone function of it.
    """

    name = "harrier"
    version = "1"

    #: Which ``models.toml`` section holds the checkpoint. Fixed across all three variants since the
    #: A/B varies only the instruction, not the weights.
    config_section = "harrier"

    def __init__(self, *, instruction: str | None = TASK_INSTRUCTION,
                 model: str | None = None,
                 max_tokens: int = DEFAULT_MAX_TOKENS,
                 batch_size: int = DEFAULT_BATCH_SIZE):
        #: ``None`` means no conditioning at all -- distinct from an empty string, which would still
        #: wrap the text in the template.
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
        """Everything that changes a vector. ``model_id`` is the configured identity rather than the
        resolved filesystem path, so two machines mirroring the same checkpoint share a cache."""
        return {
            "model_id": self.model or DEFAULT_MODEL_ID,
            "instruction": self.instruction,
            "instruct_template": INSTRUCT_TEMPLATE if self.instruction else None,
            "max_tokens": self.max_tokens,
            "pooling": "last_token",
        }

    # --- model loading -------------------------------------------------------

    def checkpoint(self) -> str:
        """The resolved local checkpoint directory. Never downloads; see the module docstring."""
        # Imported here so importing the features registry doesn't pull in the defenses package.
        from ..defenses._backends import model_checkpoint, resolve_model_path, shared_checkpoint

        if self.model:
            return resolve_model_path(shared_checkpoint(self.model) or self.model)
        return model_checkpoint(self.config_section, MODEL_ENV_VAR, local_only=True)

    def _ensure_model(self):
        """Load tokenizer and checkpoint on first use; returns ``(tokenizer, model, device)``."""
        if self._model is None:
            # Lazy: only selecting this featurizer needs torch.
            import torch
            from transformers import AutoModel, AutoTokenizer

            from ..defenses._backends import gpu_dtype

            path = self.checkpoint()
            print(f"[{self.name}] loading {path} (local_files_only)")
            self._tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
            # Last-token pooling needs the real final token at index -1, which requires LEFT
            # padding -- getting this wrong silently pools the pad token instead, with no crash.
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

        # Longest-first batching keeps padding waste down; order is restored below.
        order = sorted(range(len(rendered)), key=lambda i: len(rendered[i]), reverse=True)
        for start in range(0, len(order), self.batch_size):
            batch = [rendered[i] for i in order[start:start + self.batch_size]]
            encoded = tokenizer(batch, padding=True, truncation=True,
                                max_length=self.max_tokens, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state
            # Left padding puts the final real token last for every row in the batch.
            pooled = hidden[:, -1]
            # Normalize defensively: a no-op if the checkpoint already does, but guards against a
            # length-varying magnitude reaching the scale-sensitive attacks.
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
