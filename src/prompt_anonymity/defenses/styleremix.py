"""StyleRemix authorship-obfuscation defense (Fisher et al., EMNLP 2024).

StyleRemix rewrites text along interpretable *style axes*, each backed by a LoRA adapter over a
Llama-3-8B base; the selected adapters are merged with PEFT's weighted "cat" combination and that
merged adapter rewrites each prompt. The paper randomizes per-document weights to *evade* an
attributor; our goal is the opposite -- *convergence* -- so we drive every prompt toward ONE FIXED
slider configuration (:data:`STYLEREMIX_SLIDERS`), collapsing per-user style toward a shared
identity. This is the LoRA-steering analogue of the Qwen rewrite defense. The featurize stage
re-derives features from the rewritten text.

Base model + LoRA collection: https://huggingface.co/collections/hallisky/authorship-obfuscation-66564c1c1d59bb62eaaf954f
The base repo is gated (needs ``huggingface-cli login``). Set ``STYLEREMIX_4BIT=1`` to fit a
memory-constrained GPU (fp16 is faster on an A100, where 4-bit only adds dequant overhead).
"""

from __future__ import annotations

import os

from ._backends import PerTurnBatchRewriteDefense, gpu_dtype

STYLEREMIX_BASE_MODEL = "meta-llama/Meta-Llama-3-8B"

#: Per-axis LoRA adapters from the paper's authorship-obfuscation HF collection.
STYLEREMIX_ADAPTERS = {
    "length_more": "hallisky/lora-length-long-llama-3-8b",
    "length_less": "hallisky/lora-length-short-llama-3-8b",
    "function_more": "hallisky/lora-function-more-llama-3-8b",
    "function_less": "hallisky/lora-function-less-llama-3-8b",
    "grade_more": "hallisky/lora-grade-highschool-llama-3-8b",
    "grade_less": "hallisky/lora-grade-elementary-llama-3-8b",
    "formality_more": "hallisky/lora-formality-formal-llama-3-8b",
    "formality_less": "hallisky/lora-formality-informal-llama-3-8b",
    "sarcasm_more": "hallisky/lora-sarcasm-more-llama-3-8b",
    "sarcasm_less": "hallisky/lora-sarcasm-less-llama-3-8b",
    "voice_passive": "hallisky/lora-voice-passive-llama-3-8b",
    "voice_active": "hallisky/lora-voice-active-llama-3-8b",
    "type_persuasive": "hallisky/lora-type-persuasive-llama-3-8b",
    "type_expository": "hallisky/lora-type-expository-llama-3-8b",
    "type_narrative": "hallisky/lora-type-narrative-llama-3-8b",
    "type_descriptive": "hallisky/lora-type-descriptive-llama-3-8b",
}

#: Fixed convergence target. Bipolar axes take [-1, 1] (+ = more/advanced/formal/active/longer); the
#: four one-directional writing-type axes take [0, 1] and are mutually exclusive. 0 disables an axis.
STYLEREMIX_SLIDERS = {
    "length": 0.0,
    "function_words": 0.0,
    "grade_level": 0.0,
    "formality": 1.0,    # -> formal register
    "sarcasm": 0.0,
    "voice": 1.0,        # -> active voice
    "persuasive": 0.0,   # type_* axes: keep at most one of these four nonzero
    "descriptive": 0.0,
    "narrative": 0.0,
    "expository": 0.0,
}
STYLEREMIX_MAX_NEW_TOKENS = 1024
STYLEREMIX_LOAD_IN_4BIT = os.environ.get("STYLEREMIX_4BIT", "0") == "1"
STYLEREMIX_BATCH_SIZE = int(os.environ.get("STYLEREMIX_BATCH_SIZE", "8"))


class _StyleRemixBackend:
    """Llama-3-8B steered by a fixed merged LoRA adapter. The combo adapter is built ONCE (our
    weights never change) and reused for every prompt; :meth:`rewrite_batch` decodes in
    left-padded sub-batches to keep a GPU busy."""

    # slider name -> (positive-direction adapter, negative-direction adapter).
    _AXIS_ADAPTERS = {
        "length": ("length_more", "length_less"),
        "function_words": ("function_more", "function_less"),
        "grade_level": ("grade_more", "grade_less"),
        "formality": ("formality_more", "formality_less"),
        "sarcasm": ("sarcasm_more", "sarcasm_less"),
        "voice": ("voice_active", "voice_passive"),
        "persuasive": ("type_persuasive", None),
        "descriptive": ("type_descriptive", None),
        "narrative": ("type_narrative", None),
        "expository": ("type_expository", None),
    }

    def __init__(self, base_model, adapters, sliders, load_in_4bit, max_new_tokens, batch_size):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size

        active = self._resolve_sliders(sliders)
        types = [k for k in ("persuasive", "descriptive", "narrative", "expository") if sliders.get(k, 0)]
        assert len(types) <= 1, (
            f"At most one writing-type axis may be nonzero (got {types}); they are mutually exclusive."
        )
        if not active:
            raise ValueError("STYLEREMIX_SLIDERS are all zero -- no style to apply.")
        self._adapter_names = list(active.keys())

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading StyleRemix base '{base_model}' on {self.device} "
              f"({'4-bit' if load_in_4bit else 'fp16/fp32'})...")

        # A dedicated pad token is added (Llama-3 ships none) and base embeddings resized to match.
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, add_bos_token=True, add_eos_token=False, padding_side="left"
        )
        self.tokenizer.add_special_tokens({"pad_token": "<padding_token>"})

        load_kwargs = {}
        if load_in_4bit and self.device == "cuda":
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_compute_dtype=gpu_dtype(torch), bnb_4bit_quant_type="nf4",
            )
        else:
            load_kwargs["torch_dtype"] = gpu_dtype(torch)
        base = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)
        base.resize_token_embeddings(len(self.tokenizer))
        if not load_in_4bit or self.device != "cuda":
            base = base.to(self.device)

        # Load only the adapters the configured sliders use, then merge into one weighted "cat"
        # adapter that stays active for every prompt.
        first = self._adapter_names[0]
        model = PeftModel.from_pretrained(base, adapters[first], adapter_name=first)
        for name in self._adapter_names[1:]:
            model.load_adapter(adapters[name], adapter_name=name)
        combo = "styleremix_" + "-".join(f"{n}{int(100 * active[n])}" for n in self._adapter_names)
        model.add_weighted_adapter(
            self._adapter_names, weights=list(active.values()),
            adapter_name=combo, combination_type="cat",
        )
        model.set_adapter(combo)
        model.eval()
        self.model = model
        print(f"  StyleRemix target style: {active}")

    @classmethod
    def _resolve_sliders(cls, sliders):
        """{slider: value in [-1,1]} -> {adapter_name: magnitude}, dropping zeros. Positive values
        select the '+' adapter, negative the '-' adapter."""
        active = {}
        for name, value in sliders.items():
            if not value:
                continue
            pos, neg = cls._AXIS_ADAPTERS[name]
            adapter = pos if value > 0 else neg
            if adapter is None:
                raise ValueError(f"Axis '{name}' is one-directional; use a value in [0, 1].")
            active[adapter] = abs(value)
        return active

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        results = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            # Same "### Original: ... ### Rewrite:" template the LoRA adapters were trained on.
            # Greedy decode -> deterministic convergence. Left padding (set in __init__) keeps every
            # row's generated span at the same offset for a uniform slice.
            prompts = [f"### Original: {t}\n ### Rewrite:" for t in chunk]
            inputs = self.tokenizer(
                prompts, return_tensors="pt", max_length=2048, truncation=True, padding=True,
            ).to(self.model.device)
            input_length = inputs.input_ids.shape[1]
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            for row in outputs[:, input_length:]:
                results.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return results


class StyleRemixDefense(PerTurnBatchRewriteDefense):
    """Steer every user turn toward the fixed :data:`STYLEREMIX_SLIDERS` style with Llama-3-8B +
    merged per-axis LoRA. The backend is built lazily, so a fully-cached run loads no model."""

    name = "styleremix"
    version = "1"

    def __init__(self, *, base_model: str = STYLEREMIX_BASE_MODEL, sliders: dict | None = None,
                 load_in_4bit: bool = STYLEREMIX_LOAD_IN_4BIT,
                 max_new_tokens: int = STYLEREMIX_MAX_NEW_TOKENS,
                 batch_size: int = STYLEREMIX_BATCH_SIZE):
        self.base_model = base_model
        self.sliders = dict(STYLEREMIX_SLIDERS if sliders is None else sliders)
        self.load_in_4bit = load_in_4bit
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self._backend = None

    def params(self) -> dict:
        # Base model + slider target determine the rewrite; batch size / quantization do not change
        # the (greedy) output, so they stay out of the key.
        return {"base_model": self.base_model, "sliders": self.sliders}

    def _get_backend(self) -> _StyleRemixBackend:
        if self._backend is None:
            self._backend = _StyleRemixBackend(
                self.base_model, STYLEREMIX_ADAPTERS, self.sliders,
                self.load_in_4bit, self.max_new_tokens, self.batch_size,
            )
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        return self._get_backend().rewrite_batch(texts)
