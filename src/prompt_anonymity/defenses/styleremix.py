"""StyleRemix authorship-obfuscation defense (Fisher et al., EMNLP 2024).

StyleRemix rewrites text along interpretable *style axes*, each backed by a LoRA adapter over a
Llama-3-8B base; the selected adapters are merged with PEFT's weighted "cat" combination and that
merged adapter rewrites each prompt. The paper randomizes per-document weights to *evade* an
attributor; our goal is the opposite -- *convergence* -- so we drive every prompt toward ONE FIXED
slider configuration (:data:`STYLEREMIX_SLIDERS`), collapsing per-user style toward a shared
identity. This is the LoRA-steering analogue of the Qwen rewrite defense. The featurize stage
re-derives features from the rewritten text.

Base model + LoRA collection: https://huggingface.co/collections/hallisky/authorship-obfuscation-66564c1c1d59bb62eaaf954f
Upstream code: https://github.com/jfisher52/StyleRemix

Serving: merge once, then vLLM
------------------------------

vLLM applies **one** LoRA per request and has no equivalent of PEFT's ``add_weighted_adapter``, so
the weighted composition that *is* StyleRemix cannot happen at request time. It does not need to:
our slider configuration is fixed by design, so the merge is done **once, offline**
(:func:`build_merged_adapter`) into a single adapter that vLLM then serves for every prompt. The
merged adapter is cached on disk and keyed by the slider configuration, so it is built on the first
run and reused thereafter.

(The paper's evasion mode, which redraws weights per document, is therefore *not* expressible on
this backend -- it would need a fresh merge per document.)

Two details of the released artifacts matter for this:

* The adapters are **r=16, alpha=32**, not the ``r = 32`` the paper's Appendix C.3 reports. The
  weights win. ``cat`` sums ranks, so a merge of *n* axes has rank ``16n`` -- 32 for the two axes
  enabled below -- which is what :data:`STYLEREMIX_MAX_LORA_RANK` must cover.
* Each adapter file is **4.2 GB** rather than the ~50 MB its config implies, because the upstream
  code resizes the embedding matrix to add a padding token and PEFT then auto-saves
  ``embed_tokens`` and ``lm_head``. Those tensors are verifiably redundant -- every one of the
  128,256 original rows is bit-identical to the base model's, the only difference being one
  untrained pad row -- so the merge drops them (``save_embedding_layers=False``). vLLM could not
  consume them anyway, and it needs no pad token, since it does not pad inside a sequence.

Fidelity to the upstream implementation
---------------------------------------

Matching upstream: the adapter set, the ``### Original: ... ### Rewrite:`` format (which Appendix
C.3 confirms is also the *training* format), the ``cat`` merge, and cutting the generation at the
next ``###`` -- upstream's ``process_output``, which is the only thing stopping a base model from
running on past the rewrite. Here that is a vLLM ``stop`` string, so generation *halts* rather than
being trimmed after the fact.

Deliberately different, and why:

* **Greedy decoding.** Upstream samples (``do_sample=True, top_p=0.95``, and temperature 0.6
  inherited from Llama-3's ``generation_config.json``). We decode greedily: convergence wants the
  same input to land on the same rewrite, and a cached defense whose output is not reproducible
  cannot be re-run.
* **Long turns are split, not truncated.** The adapters were trained at a 512-token sequence
  length, and upstream's merge pipeline truncates its input at 256 tokens. Chat turns are routinely
  longer than either, and silently dropping their tails would make the defended dataset a lie, so a
  long turn is cut into in-distribution fragments, each rewritten, then rejoined
  (:data:`STYLEREMIX_INPUT_TOKENS`).
* **No ``unidecode``.** Upstream ASCII-folds every prompt, which is harmless for its English
  corpora and destructive for a multilingual one.
* Only the *active* adapters are merged, where upstream merges all 16 with zero weights for the
  inactive ones. The zero-weight blocks contribute nothing, so the delta is identical while the
  rank stays 16n instead of 256.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

from ._backends import (
    MIN_DEFEND_CHARS,
    PerTurnBatchRewriteDefense,
    configure_cuda_toolkit,
    resolve_model_path,
    shutdown_vllm,
)

#: EDIT ME — the base model the adapters steer. A checkpoint directory, a HuggingFace hub *cache*
#: entry (resolved to its snapshot), or a repo id to download. Part of the cache key, so a swap
#: re-caches. The adapters are trained on Llama-3-8B and mean nothing on top of anything else.
STYLEREMIX_BASE_MODEL = os.environ.get(
    "STYLEREMIX_BASE_MODEL", "/datasets/ai/llama3/hub/models--meta-llama--Meta-Llama-3-8B")

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

#: The prompt format the adapters were trained on. Not a chat template -- the base model has none.
STYLEREMIX_PROMPT_TEMPLATE = "### Original: {text}\n ### Rewrite:"
#: Upstream cuts every generation at the next ``###``; as a vLLM stop string it also halts decoding.
STYLEREMIX_STOP = "###"

#: Input tokens per fragment. The adapters were trained at a 512-token sequence length, so this is
#: what they have actually seen; a longer turn is split rather than truncated.
STYLEREMIX_INPUT_TOKENS = int(os.environ.get("STYLEREMIX_INPUT_TOKENS", "512"))
#: Generation cap per fragment. Upstream uses 512 (merge pipeline) or 1024 (quickstart); a fragment
#: is at most 512 tokens in and the length axis can lengthen it, so 1024 leaves room.
STYLEREMIX_MAX_NEW_TOKENS = int(os.environ.get("STYLEREMIX_MAX_NEW_TOKENS", "1024"))
#: Context window to serve: one fragment in, one rewrite out, plus the template.
STYLEREMIX_MAX_MODEL_LEN = int(os.environ.get("STYLEREMIX_MAX_MODEL_LEN", "2048"))
#: Largest merged rank vLLM will accept. ``cat`` gives 16 per active axis; vLLM's allowed values are
#: 1, 8, 16, 32, 64, 128, 256, 320, 512, and the merge rounds up to one of them.
STYLEREMIX_MAX_LORA_RANK = int(os.environ.get("STYLEREMIX_MAX_LORA_RANK", "0")) or None
#: Fraction of GPU memory vLLM may hold. Lower it when another model shares the GPU -- notably the
#: combined StyleRemix+OpenAnonymity defense, which loads this *and* the scrubber.
STYLEREMIX_GPU_MEM_UTIL = float(os.environ.get("STYLEREMIX_GPU_MEM_UTIL", "0.85"))
#: Skip vLLM's startup compile: faster to start, slower per token. Worth it for a smoke test.
STYLEREMIX_ENFORCE_EAGER = os.environ.get("STYLEREMIX_ENFORCE_EAGER", "0") == "1"
#: Where merged adapters are cached. Regenerable, so it belongs with the other build artifacts.
STYLEREMIX_ADAPTER_DIR = os.environ.get("STYLEREMIX_ADAPTER_DIR", "")
#: Conversations defended between cache flushes, so a long run resumes where it stopped.
STYLEREMIX_CHECKPOINT_EVERY = int(os.environ.get("STYLEREMIX_CHECKPOINT_EVERY", "1000"))
#: Shortest turn worth restyling; below this the model has nothing to work from and invents content
#: (see :data:`~prompt_anonymity.defenses._backends.MIN_DEFEND_CHARS` for the measurement). Style is
#: the one thing a 5-character turn does not carry, so skipping it costs the defense almost nothing.
STYLEREMIX_MIN_CHARS = int(os.environ.get("STYLEREMIX_MIN_CHARS", str(MIN_DEFEND_CHARS)))

# slider name -> (positive-direction adapter, negative-direction adapter).
AXIS_ADAPTERS = {
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

#: vLLM's permitted ``max_lora_rank`` values; a merged rank is rounded up to one of these.
VLLM_LORA_RANKS = (1, 8, 16, 32, 64, 128, 256, 320, 512)


def resolve_sliders(sliders: dict) -> dict:
    """``{slider: value in [-1, 1]}`` -> ``{adapter_name: magnitude}``, dropping zeros.

    Positive values select the '+' adapter of the axis, negative the '-' adapter. The four
    writing-type axes are one-directional and mutually exclusive, which is checked here.
    """
    types = [name for name in ("persuasive", "descriptive", "narrative", "expository")
             if sliders.get(name, 0)]
    if len(types) > 1:
        raise ValueError(f"At most one writing-type axis may be nonzero (got {types}); "
                         f"they are mutually exclusive.")
    active = {}
    for name, value in sliders.items():
        if not value:
            continue
        if name not in AXIS_ADAPTERS:
            raise ValueError(f"unknown style axis {name!r}; available: {sorted(AXIS_ADAPTERS)}")
        positive, negative = AXIS_ADAPTERS[name]
        adapter = positive if value > 0 else negative
        if adapter is None:
            raise ValueError(f"Axis {name!r} is one-directional; use a value in [0, 1].")
        active[adapter] = abs(value)
    if not active:
        raise ValueError("STYLEREMIX_SLIDERS are all zero -- no style to apply.")
    return active


def slider_tag(active: dict) -> str:
    """Directory-safe name for a merged adapter, e.g. ``formality_more100-voice_active100``.

    Upstream's combo-adapter naming (name + weight x100), which makes the cached artifact
    self-describing and gives two slider configurations two different directories.
    """
    return "-".join(f"{name}{int(round(100 * weight))}" for name, weight in sorted(active.items()))


def merged_adapter_root() -> Path:
    """Where merged adapters are cached: ``$STYLEREMIX_ADAPTER_DIR``, else ``<data>/.cache/styleremix``."""
    if STYLEREMIX_ADAPTER_DIR:
        return Path(STYLEREMIX_ADAPTER_DIR)
    from ..data.config import cache_dir  # local import: the defense does not otherwise need the data pkg
    return cache_dir() / "styleremix"


def vllm_lora_rank(rank: int) -> int:
    """The smallest ``max_lora_rank`` vLLM accepts that still holds ``rank``."""
    for allowed in VLLM_LORA_RANKS:
        if rank <= allowed:
            return allowed
    raise ValueError(f"merged LoRA rank {rank} exceeds vLLM's largest supported rank "
                     f"({VLLM_LORA_RANKS[-1]}); enable fewer style axes.")


def build_merged_adapter(base_model: str, adapters: dict, active: dict, out_dir: Path) -> Path:
    """Merge the active per-axis adapters into one LoRA on disk, and return its directory.

    Runs PEFT's own ``add_weighted_adapter(..., combination_type="cat")`` -- the upstream
    composition, not a reimplementation -- which concatenates the components into a single adapter
    of rank ``sum(r_i)`` whose delta is ``sum_i weight_i * (alpha_i / r_i) * B_i @ A_i``.

    Cached: an existing directory is reused, so the (slow, one-off) merge happens on the first run
    for a given slider configuration and never again. The base model is loaded on **CPU** purely
    because PEFT needs something to attach the adapters to -- no base weight is read or written by
    the merge, and no GPU is touched. ``save_embedding_layers=False`` keeps the redundant 4 GB
    embedding tensors out of the result (see the module docstring).
    """
    if (out_dir / "adapter_config.json").exists():
        return out_dir

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"StyleRemix: merging {len(active)} adapter(s) {list(active)} -> {out_dir} "
          f"(one-off; loading the base on CPU to host them)")
    path = resolve_model_path(base_model)
    base = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, low_cpu_mem_usage=True)
    # The adapters were saved against a base whose embeddings had been resized for an added padding
    # token, so their files carry 128,257-row embed_tokens/lm_head and PEFT refuses to load them
    # into an unresized 128,256-row base. Reproduce upstream's resize purely to satisfy that shape
    # check: the extra row is untrained, it is dropped again by save_embedding_layers=False below,
    # and the base vLLM serves is the original unresized checkpoint.
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.add_special_tokens({"pad_token": "<padding_token>"})
    base.resize_token_embeddings(len(tokenizer))
    names = list(active)
    model = PeftModel.from_pretrained(base, adapters[names[0]], adapter_name=names[0])
    for name in names[1:]:
        model.load_adapter(adapters[name], adapter_name=name)
    combo = slider_tag(active)
    model.add_weighted_adapter(names, weights=[active[n] for n in names],
                               adapter_name=combo, combination_type="cat")
    model.set_adapter(combo)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out_dir.parent), selected_adapters=[combo],
                          save_embedding_layers=False)
    print(f"StyleRemix: merged adapter written to {out_dir}")
    return out_dir


def merged_adapter_rank(adapter_dir: Path) -> int:
    """The merged adapter's rank, read back from what was actually written."""
    import json
    return int(json.loads((adapter_dir / "adapter_config.json").read_text())["r"])


def strip_leading_marker(text: str) -> str:
    """Drop a leading ``###`` marker the model sometimes echoes before its rewrite (upstream does
    the same), then trim."""
    return re.sub(r"^\s*###\s*(Rewrite:)?", "", text).strip()


class _StyleRemixBackend:
    """Llama-3-8B steered by one pre-merged LoRA, served by vLLM.

    The merge is built (or reused) before the engine starts, so the running engine holds exactly one
    adapter and every request uses it. Long turns are split into in-distribution fragments, rewritten
    in one batched call, and rejoined.
    """

    def __init__(self, base_model, adapters, sliders, *, max_new_tokens=STYLEREMIX_MAX_NEW_TOKENS,
                 input_tokens=STYLEREMIX_INPUT_TOKENS, max_model_len=STYLEREMIX_MAX_MODEL_LEN,
                 gpu_memory_utilization=STYLEREMIX_GPU_MEM_UTIL,
                 enforce_eager=STYLEREMIX_ENFORCE_EAGER, max_lora_rank=STYLEREMIX_MAX_LORA_RANK):
        configure_cuda_toolkit()  # must precede the import: vLLM reads the environment at import
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        active = resolve_sliders(sliders)
        adapter_dir = build_merged_adapter(
            base_model, adapters, active, merged_adapter_root() / slider_tag(active))
        rank = merged_adapter_rank(adapter_dir)
        self.max_new_tokens = max_new_tokens
        self.input_tokens = input_tokens

        path = resolve_model_path(base_model)
        print(f"StyleRemix: loading base '{path}' with vLLM (LoRA rank {rank}, "
              f"context {max_model_len:,}); target style {active}")
        self.llm = LLM(
            model=path, enable_lora=True, max_lora_rank=max_lora_rank or vllm_lora_rank(rank),
            max_model_len=max_model_len, gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=enforce_eager,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora_request = LoRARequest(slider_tag(active), 1, str(adapter_dir))
        # Greedy, and stopped at the next "###" so a base model cannot run on past the rewrite.
        self.sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=max_new_tokens,
                                       stop=[STYLEREMIX_STOP])

    def _fragments(self, text: str) -> list[str]:
        """``text`` split into pieces of at most :attr:`input_tokens` tokens, concatenating back to
        the original. Returns ``[text]`` for anything already short enough (the common case)."""
        if len(text) <= self.input_tokens:  # a token is >=1 char, so this cannot exceed the budget
            return [text]
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.input_tokens:
            return [text]
        return [self.tokenizer.decode(ids[i:i + self.input_tokens])
                for i in range(0, len(ids), self.input_tokens)]

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        """Restyle a batch of turns, preserving input order. Blank turns pass through."""
        texts = list(texts)
        fragments: list[str] = []
        owners: list[int] = []
        for position, text in enumerate(texts):
            if not text.strip():
                continue
            for fragment in self._fragments(text):
                fragments.append(fragment)
                owners.append(position)
        if not fragments:
            return texts

        prompts = [STYLEREMIX_PROMPT_TEMPLATE.format(text=fragment) for fragment in fragments]
        outputs = self.llm.generate(prompts, self.sampling, lora_request=self.lora_request)

        rejoined: dict[int, list[str]] = {}
        empty = 0
        for position, output, fragment in zip(owners, outputs, fragments):
            rewritten = strip_leading_marker(output.outputs[0].text)
            if not rewritten:
                # The model emitted the stop marker immediately, so there is no rewrite to use.
                # Keeping the original leaves the turn visibly undefended rather than blank.
                rewritten = fragment
                empty += 1
            rejoined.setdefault(position, []).append(rewritten)
        if empty:
            print(f"StyleRemix: {empty:,}/{len(fragments):,} fragments produced no rewrite and were "
                  f"left unchanged.")
        return [" ".join(rejoined[i]) if i in rejoined else text
                for i, text in enumerate(texts)]

    def close(self) -> None:
        """Shut the engine down and hand its GPU memory back (see :func:`shutdown_vllm`)."""
        if getattr(self, "llm", None) is not None:
            shutdown_vllm(self.llm)
            self.llm = None


class StyleRemixDefense(PerTurnBatchRewriteDefense):
    """Steer every user turn toward the fixed :data:`STYLEREMIX_SLIDERS` style with Llama-3-8B +
    a pre-merged per-axis LoRA, served by vLLM.

    The backend (and the merge behind it) is built lazily, so a fully-cached run loads no model.
    """

    name = "styleremix"
    # 2: backend moved from Transformers to vLLM with a pre-merged adapter, generation is now
    # stopped at "###" (as upstream post-processes), and long turns are fragmented rather than
    # truncated. The rewrites differ from version 1's.
    version = "2"
    checkpoint_every = STYLEREMIX_CHECKPOINT_EVERY
    min_defend_chars = STYLEREMIX_MIN_CHARS

    def __init__(self, *, base_model: str = STYLEREMIX_BASE_MODEL, sliders: dict | None = None,
                 max_new_tokens: int = STYLEREMIX_MAX_NEW_TOKENS,
                 input_tokens: int = STYLEREMIX_INPUT_TOKENS,
                 min_defend_chars: int = STYLEREMIX_MIN_CHARS):
        self.base_model = base_model
        self.sliders = dict(STYLEREMIX_SLIDERS if sliders is None else sliders)
        self.max_new_tokens = max_new_tokens
        self.input_tokens = input_tokens
        self.min_defend_chars = min_defend_chars
        self._backend = None

    def params(self) -> dict:
        # What determines the (greedy) rewrite: the base model, the style target, how much text the
        # model sees per call -- the fragment size changes where a long turn is cut, and so its
        # rewrite -- and which turns are rewritten at all. Engine knobs (memory fraction, eager
        # mode) do not change the output.
        return {"base_model": self.base_model, "sliders": self.sliders,
                "input_tokens": self.input_tokens, "max_new_tokens": self.max_new_tokens,
                "min_defend_chars": self.min_defend_chars}

    def _get_backend(self) -> _StyleRemixBackend:
        if self._backend is None:
            self._backend = _StyleRemixBackend(
                self.base_model, STYLEREMIX_ADAPTERS, self.sliders,
                max_new_tokens=self.max_new_tokens, input_tokens=self.input_tokens,
            )
        return self._backend

    def rewrite_batch(self, texts: list[str]) -> list[str]:
        return self._get_backend().rewrite_batch(texts)
