import hashlib
import os
import random
import re

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- Config ---
LANG = ("English", "en")
KNOWN_MODEL = "gpt-4o-2024-08-06"
UNKNOWN_MODEL = "gpt-4.1-mini-2025-04-14"
# Shared inputs live in DS_env/ root (also consumed by analyze_wildchat.ipynb).
DATA_CSV = "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv"
EMBEDDINGS_CSV = f"wildchat_filtered_{LANG[1]}_2048_stylometrix.csv"
N_SIM = 100
SAMPLE_STEP = 25
RANDOM_SEED = 47

# --- Round-trip-translation (RTT) defense config ---
MAX_LEN = 2048                        # char truncation, matches wildchat/stylometrix.py
STYLO_LANGCODE = "en"                 # final text is English -> embed with the English model

# Defense-generated artifacts are organized under defense_data/{cache,embeddings,output}.
DEFENSE_DATA_DIR = "defense_data"
CACHE_DIR = os.path.join(DEFENSE_DATA_DIR, "cache")
EMB_DIR = os.path.join(DEFENSE_DATA_DIR, "embeddings")
OUTPUT_DIR = os.path.join(DEFENSE_DATA_DIR, "output")
for _d in (CACHE_DIR, EMB_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)  # ensure dirs exist on a fresh checkout

OUTPUT_CSV = os.path.join(OUTPUT_DIR, "wildchat_analysis_stylometrix_results.csv")

# Argos backend caches (source -> translated CSV, plus re-embedded .npz per side).
RTT_CACHE_CSV = os.path.join(CACHE_DIR, "wildchat_rtt_translation_cache.csv")
RTT_EMB_KNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_known_emb.npz")
RTT_EMB_UNKNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_unknown_emb.npz")

# NLLB backend gets its own cache files so it never clobbers the Argos results.
NLLB_MODEL = "facebook/nllb-200-distilled-600M"         # ~600M params; fits an 8GB GPU in fp16
NLLB_CACHE_CSV = os.path.join(CACHE_DIR, "wildchat_rtt_nllb_translation_cache.csv")
NLLB_EMB_KNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_nllb_known_emb.npz")
NLLB_EMB_UNKNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_nllb_unknown_emb.npz")

# --- Rewrite defense config -------------------------------------------------
# Threat model: a USER's own writing style fingerprints them across queries.
# This defense rewrites every prompt into ONE fixed target style locally (4-bit
# Qwen via llama.cpp) BEFORE it is ever sent to a chatbot provider, so prompts
# from different users converge to a shared, indistinguishable stylometric
# identity while keeping each query's content intact.

# ┌──────────────────────────────────────────────────────────────────────────┐
# │ EDIT ME — this is the style every prompt is forced into ("rewrite like X").│
# │ Tweak the instructions to experiment with different convergence targets.   │
# │ KEEP the "preserve all info / output only the rewrite" guardrails, or the  │
# │ rewritten prompt will drift from what the user actually asked.             │
# └──────────────────────────────────────────────────────────────────────────┘
REWRITE_PROMPT_HEADER = """You are a prompt-rewriting filter. Rewrite the user's message into a single, neutral, standardized style so that messages written by different people become stylistically indistinguishable.
Rules:
- Preserve ALL information, intent, constraints, and any code, names, numbers, or quotations exactly as given.
- Write in plain, formal English: complete declarative sentences, no slang, no contractions, no emoji, no personal asides or filler.
- Keep the same language as the input message.
- Use active voice and subject-verb-object order only. Do not use passive voice, fronted clauses, cleft constructions ("It is X that..."), or inversions.
- Express every request as a direct imperative beginning with a verb (e.g., 'Write...', 'Summarize...', 'Fix...'). Do not use politeness framings ('Could you', 'I'd like you to', 'Please') or question forms for requests.
- One idea per sentence. Split compound or multi-clause sentences.
- Order the content canonically: (1) the task, (2) constraints and requirements, (3) supporting context or examples.
- Remove hedges and intensifiers ('really', 'very', 'basically', 'just', etc).
- Use only periods and commas. Do not use em dashes, semicolons, colons for asides, ellipses, or parentheticals; rewrite the content into separate sentences instead.
- Use digits for all numbers.
- Do NOT substitute technical terms, domain verbs, or proper nouns for synonyms. Normalize register and grammar only, never denotation.
- Do NOT answer, explain, or comment on the message. Output ONLY the rewritten message and nothing else."""

# Qwen 4-bit GGUF, auto-downloaded from the Hugging Face Hub on first use and
# cached locally (offline thereafter), mirroring the Argos/NLLB download pattern.
# Swap to "...1.5b..." for slower phones or "...7b..." on a laptop for fidelity.
QWEN_GGUF_REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"        # on-device rewriter
QWEN_GGUF_FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"      # ~2GB, 4-bit, CPU-friendly
REWRITE_CACHE_CSV = os.path.join(CACHE_DIR, "wildchat_rewrite_cache.csv")
REWRITE_EMB_KNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rewrite_known_emb.npz")
REWRITE_EMB_UNKNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rewrite_unknown_emb.npz")

# --- Qwen round-trip-translation defense config -----------------------------
# Like the NLLB RTT defense, but the EN->ZH->JA->EN chain is driven by the SAME
# on-device 4-bit Qwen GGUF as the rewrite defense (GPU-offloaded via llama.cpp).
# Each hop is a separate, discrete chat completion so the language pivots are
# explicit and independently inspectable, rather than one fused prompt.
QWEN_RTT_HOPS = [
    ("English", "Chinese"),
    ("Chinese", "Japanese"),
    ("Japanese", "English"),
]
QWEN_RTT_CACHE_CSV = os.path.join(CACHE_DIR, "wildchat_rtt_qwen_translation_cache.csv")
QWEN_RTT_EMB_KNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_qwen_known_emb.npz")
QWEN_RTT_EMB_UNKNOWN_NPZ = os.path.join(EMB_DIR, "wildchat_rtt_qwen_unknown_emb.npz")


# ---------------------------------------------------------------------------
# Attack functions
# Each takes (known_emb, unknown_emb) as np.ndarray and returns a similarity
# matrix of shape (n_unknown, n_known) where higher = more similar.
# ---------------------------------------------------------------------------

def euclidean_style_attack(known_emb: np.ndarray, unknown_emb: np.ndarray) -> np.ndarray:
    return -cdist(unknown_emb, known_emb, metric="euclidean")


def cosine_style_attack(known_emb: np.ndarray, unknown_emb: np.ndarray) -> np.ndarray:
    known_norm = known_emb / np.linalg.norm(known_emb, axis=1, keepdims=True)
    unknown_norm = unknown_emb / np.linalg.norm(unknown_emb, axis=1, keepdims=True)
    return unknown_norm @ known_norm.T


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    df = df[df["language"] == LANG[0]]
    df = df[
        df.groupby(["hashed_ip", "accept_language", "device_info"])["model"]
        .transform("nunique")
        .ge(2)
    ]
    df = df.sort_values(["hashed_ip", "accept_language", "device_info", "timestamp"])
    df["identity"] = df["hashed_ip"] + "|" + df["accept_language"] + "|" + df["device_info"]
    return df.reset_index(drop=True)


def load_embeddings(df: pd.DataFrame):
    known_filter = df["model"] == KNOWN_MODEL
    unknown_filter = df["model"] == UNKNOWN_MODEL

    embeddings = pd.read_csv(EMBEDDINGS_CSV).drop(columns="text")
    known_emb = embeddings[known_filter].to_numpy()
    unknown_emb = embeddings[unknown_filter].to_numpy()

    known = df[known_filter].reset_index(drop=True)
    unknown = df[unknown_filter].reset_index(drop=True)

    return known, unknown, known_emb, unknown_emb


# ---------------------------------------------------------------------------
# Ranking — shared utility, attack-agnostic
# ---------------------------------------------------------------------------

def build_rankings(known, unknown, known_emb, unknown_emb, attack_fn):
    known_ids = known["identity"]
    unknown_ids = unknown["identity"]

    similarity = attack_fn(known_emb, unknown_emb)
    sim_rank = np.argsort(-similarity, axis=1)

    sim_rank_correct = []
    sim_rank_ids = []
    for i, rank in enumerate(sim_rank):
        target_id = unknown_ids.iloc[i]
        rank_ids = known_ids.iloc[rank]
        sim_rank_correct.append((rank_ids == target_id).to_list())
        sim_rank_ids.append(list({id: True for id in rank_ids}.keys()))

    return sim_rank_correct, sim_rank_ids


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_trial(
    sim_rank_correct,
    sim_rank_ids,
    unknown,
    unknown_ids,
    unknown_ids_to_conv_idx,
    unknown_ids_conv_count,
    sample=None,
):
    all_ids = set(unknown_ids)
    if sample is None:
        valid_ids = all_ids
        sample = len(valid_ids)
    else:
        valid_ids = set(random.sample(sorted(all_ids), sample))

    valid_idx = [idx for vid in valid_ids for idx in unknown_ids_to_conv_idx[vid]]
    s_correct = [sim_rank_correct[idx] for idx in valid_idx]
    s_ids = [
        [iden for iden in sim_rank_ids[idx] if iden in valid_ids]
        for idx in valid_idx
    ]
    unknown_sub = unknown.iloc[valid_idx]
    conv_count = {k: v for k, v in unknown_ids_conv_count.items() if k in valid_ids}
    num_ids = len(valid_ids)

    res = []
    for top in [1, 5, 10]:
        conv_acc = float(np.mean([np.any(l[:top]) for l in s_correct]))

        correct_ids = {
            unknown_sub.iloc[i]["identity"]
            for i, l in enumerate(s_ids)
            if unknown_sub.iloc[i]["identity"] in l[:top]
        }
        id_acc = len(correct_ids) / num_ids

        random_guessing = sum(
            1 - (1 - min(top, num_ids) / num_ids) ** cnt
            for cnt in conv_count.values()
        ) / num_ids

        id_correct_cnt = {}
        for i, l in enumerate(s_ids):
            target = unknown_sub.iloc[i]["identity"]
            if target in l[:top]:
                id_correct_cnt[target] = id_correct_cnt.get(target, 0) + 1
        correct_ids_all = sum(
            1 for iden in id_correct_cnt if id_correct_cnt[iden] == conv_count[iden]
        )
        id_acc_all = correct_ids_all / num_ids

        random_guessing_all = sum(
            (min(top, num_ids) / num_ids) ** cnt for cnt in conv_count.values()
        ) / num_ids

        res.append({
            "sample": sample,
            "top": top,
            "conv_acc": conv_acc,
            "id_acc": id_acc,
            "random_id": random_guessing,
            "advantage": id_acc - random_guessing,
            "id_acc_all_conv": id_acc_all,
            "random_id_all_conv": random_guessing_all,
            "advantage_all_conv": id_acc_all - random_guessing_all,
        })

    return res


def run_experiment(attack_fn, known, unknown, known_emb, unknown_emb, output_csv=OUTPUT_CSV):
    unknown_ids = unknown["identity"]
    num_unique_ids = unknown_ids.nunique()

    unknown_ids_to_conv_idx = {}
    for uid in unknown_ids:
        unknown_ids_to_conv_idx[uid] = unknown[unknown["identity"] == uid].index.tolist()
    unknown_ids_conv_count = unknown.groupby("identity").size().to_dict()

    print(f"Running attack: {attack_fn.__name__}")
    sim_rank_correct, sim_rank_ids = build_rankings(known, unknown, known_emb, unknown_emb, attack_fn)

    random.seed(RANDOM_SEED)
    all_res = []

    for sample in range(SAMPLE_STEP, num_unique_ids, SAMPLE_STEP):
        trial_res = []
        for _ in tqdm(range(N_SIM), desc=f"  n={sample}", leave=False):
            res = run_trial(
                sim_rank_correct,
                sim_rank_ids,
                unknown,
                unknown_ids,
                unknown_ids_to_conv_idx,
                unknown_ids_conv_count,
                sample=sample,
            )
            trial_res.extend(res)

        top1 = np.mean([r["id_acc"] for r in trial_res if r["top"] == 1])
        top5 = np.mean([r["id_acc"] for r in trial_res if r["top"] == 5])
        print(f"  n={sample:4d} | top-1: {top1:.3f} | top-5: {top5:.3f}")
        all_res.extend(trial_res)

    pd.DataFrame(all_res).to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")


# ---------------------------------------------------------------------------
# Round-trip-translation (RTT) defense
#
# Idea: scramble per-user stylometric signal by translating each prompt through
# a language chain and back to English, then re-embedding the result. We then
# ask whether euclidean_style_attack can still link prompts after this
# perturbation — i.e. does the defense lower identity accuracy toward random?
#
# Intended chain: EN -> ZH -> JA -> EN.
# NOTE / ROADBLOCK: Argos Translate has no direct zh->ja model, so the zh->ja
# hop is auto-pivoted through English by the library. The *effective* path is
# therefore EN -> ZH -> (EN) -> JA -> EN. This is still a strong multi-hop
# translationese perturbation; it is just not a "true" cross-lingual zh->ja hop.
# The Translator backend is swappable so a future NLLBTranslator (facebook/
# nllb-200, which supports a genuine zh->ja) can drop in without other changes.
# ---------------------------------------------------------------------------


class ArgosTranslator:
    """Round-trip translator backed by argos-translate (offline, CPU-friendly).

    HOPS is the logical chain we request. REQUIRED is the set of *direct*
    packages that must be installed for every hop to resolve: zh->ja needs
    zh->en + en->ja installed because Argos pivots that hop through English.
    """

    HOPS = [("en", "zh"), ("zh", "ja"), ("ja", "en")]
    REQUIRED = [("en", "zh"), ("zh", "en"), ("en", "ja"), ("ja", "en")]

    def __init__(self):
        # Imported lazily so the rest of this module runs without argostranslate.
        import argostranslate.package as package

        package.update_package_index()
        available = package.get_available_packages()
        installed = {(p.from_code, p.to_code) for p in package.get_installed_packages()}

        # Figure out which direct packages are still missing, then install them.
        # Downloads happen once (later runs are fully offline); the progress bar
        # makes the otherwise-silent model download visible.
        missing = [pair for pair in self.REQUIRED if pair not in installed]
        for from_code, to_code in tqdm(missing, desc="Downloading argos models", unit="pkg"):
            match = next(
                (p for p in available if p.from_code == from_code and p.to_code == to_code),
                None,
            )
            if match is None:
                raise RuntimeError(f"No argos package available for {from_code}->{to_code}")
            package.install_from_path(match.download())

    def roundtrip(self, text: str) -> str:
        import argostranslate.translate as translate

        # Walk the chain; the zh->ja hop is auto-pivoted through English.
        for from_code, to_code in self.HOPS:
            text = translate.translate(text, from_code, to_code)
        return text


class NLLBTranslator:
    """Round-trip translator using Meta's NLLB-200 (HuggingFace transformers).

    Unlike Argos, NLLB is many-to-many, so zh->ja is a *real* direct hop with no
    English pivot — the chain is a true EN->ZH->JA->EN. Higher quality than Argos
    at the cost of a heavier model. Runs on GPU if one is visible, else CPU.

    HOPS use FLORES-200 language codes. Text is split into sentences and
    translated in a single batched generate() call per hop, both to stay within
    the model's context window and to keep the GPU busy.
    """

    HOPS = [("eng_Latn", "zho_Hans"), ("zho_Hans", "jpn_Jpan"), ("jpn_Jpan", "eng_Latn")]
    # Sentence terminators across the languages we pass through (Latin + CJK).
    _SENT_SPLIT = re.compile(r"(?<=[.!?。！？])\s*")

    def __init__(self, model_name=NLLB_MODEL, max_length=512):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.max_length = max_length
        print(f"Loading NLLB model '{model_name}' on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

    def _split(self, text):
        parts = [p for p in self._SENT_SPLIT.split(text) if p.strip()]
        return parts or [text]  # always translate at least the whole string

    def _translate(self, sentences, src, tgt):
        self.tokenizer.src_lang = src
        inputs = self.tokenizer(
            sentences, return_tensors="pt", padding=True, truncation=True, max_length=self.max_length
        ).to(self.device)
        bos = self.tokenizer.convert_tokens_to_ids(tgt)  # forced target-language BOS
        with self._torch.no_grad():
            out = self.model.generate(**inputs, forced_bos_token_id=bos, max_length=self.max_length)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    def roundtrip(self, text: str) -> str:
        sentences = self._split(text)
        for src, tgt in self.HOPS:
            sentences = self._translate(sentences, src, tgt)
        return " ".join(sentences)


class QwenRewriter:
    """On-device prompt-rewriting defense backed by a 4-bit Qwen GGUF via
    llama-cpp-python (CPU-friendly; runs on a slow laptop or phone).

    Contract: exposes `.roundtrip(text) -> text`, the SAME duck-typed interface
    the translators use, so it drops straight into `_rtt_defense` with no other
    changes. Here the "roundtrip" is a style-normalizing rewrite: every prompt is
    rewritten into one fixed target style (REWRITE_PROMPT_HEADER), collapsing
    per-user stylometric signal toward a shared identity while preserving content.

    The model is auto-downloaded from the HF Hub on first use (cached offline
    afterwards), the same first-run-only cost the Argos/NLLB backends pay.
    """

    def __init__(self, repo_id=QWEN_GGUF_REPO, filename=QWEN_GGUF_FILE,
                 system_prompt=REWRITE_PROMPT_HEADER, n_ctx=4096,
                 n_threads=None, max_tokens=1024, n_gpu_layers=-1):
        from llama_cpp import Llama

        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        # n_gpu_layers=-1 offloads ALL transformer layers to the GPU. This is the
        # single biggest speed lever: without it llama.cpp keeps everything on CPU.
        # It only takes effect if llama-cpp-python was built with CUDA support
        # (llama_cpp.llama_supports_gpu_offload() must be True) — otherwise it is a
        # harmless no-op and inference stays on CPU. Qwen2.5-3B q4 (~2GB) fits the
        # 3070's 8GB VRAM with room to spare.
        print(f"Loading Qwen GGUF '{repo_id}/{filename}' (4-bit, llama.cpp)...")
        # from_pretrained downloads+caches the GGUF via huggingface_hub; n_threads
        #=None lets llama.cpp pick a sensible count from the available cores.
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    def roundtrip(self, text: str) -> str:
        # temperature=0 -> greedy/deterministic. Same model + same prompt + greedy
        # decoding is exactly what drives every user's text to converge in style.
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return out["choices"][0]["message"]["content"].strip()


class QwenTranslator:
    """Round-trip translator that drives the SAME 4-bit Qwen GGUF as QwenRewriter
    through an EN->ZH->JA->EN chain, GPU-offloaded via llama.cpp.

    Contract: exposes `.roundtrip(text) -> text`, the same duck-typed interface
    the other translators use, so it drops straight into `_rtt_defense`.

    Unlike NLLBTranslator (one batched generate per hop), each hop here is a
    SEPARATE, discrete chat completion: English->Chinese, then Chinese->Japanese,
    then Japanese->English. The model sees only the previous hop's output, so the
    three translation steps are explicit and independently inspectable rather than
    fused into a single prompt. This is the round-trip analogue of the rewrite
    defense — it perturbs per-user stylometric signal via translationese instead
    of via style normalization.
    """

    def __init__(self, repo_id=QWEN_GGUF_REPO, filename=QWEN_GGUF_FILE,
                 hops=QWEN_RTT_HOPS, n_ctx=4096, n_threads=None,
                 max_tokens=1024, n_gpu_layers=-1):
        from llama_cpp import Llama

        self.hops = hops
        self.max_tokens = max_tokens
        # n_gpu_layers=-1 offloads ALL transformer layers to the GPU, exactly as
        # QwenRewriter does; it is a harmless no-op if llama.cpp lacks CUDA. The
        # ~2GB q4 model fits the 3070's 8GB VRAM with room to spare.
        print(f"Loading Qwen GGUF '{repo_id}/{filename}' (4-bit, llama.cpp) for RTT...")
        self.llm = Llama.from_pretrained(
            repo_id=repo_id,
            filename=filename,
            n_ctx=n_ctx,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
        )

    # Conversational preambles the model adds despite the guardrail, e.g.
    # "Here's the text in English:" or the Chinese/Japanese equivalents the
    # intermediate hops emit. Anchored at the start and required to both carry a
    # translation/language keyword AND end in a colon, so a legitimate leading
    # clause without that shape is left untouched. Stripped at EVERY hop so an
    # intermediate-language preamble does not compound down the chain.
    _PREAMBLE_RE = re.compile(
        r"^\s*"
        r"(?:sure|certainly|of course|okay|ok)?[,!.]?\s*"          # optional filler opener
        r"(?:here(?:'s| is| are)|here you go|below is|the following is|"
        r"the\s+\w+\s+translation\s+is|"
        r"以下は|以下が|これは|翻訳は|日本語訳|"                     # Japanese lead-ins
        r"以下是|这是|翻译如下|中文翻译|英文翻译|英语翻译)"        # Chinese lead-ins
        r"[^\n:：]*[:：]\s*",                                       # rest of clause + colon
        re.IGNORECASE,
    )

    @classmethod
    def _strip_preamble(cls, text: str) -> str:
        text = text.strip()
        # Unwrap a fully fenced ```...``` block if the model wrapped the output.
        fence = re.match(r"^```[^\n]*\n(.*)\n```$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()
        # Drop a leading "Here's the text in English:"-style preamble.
        text = cls._PREAMBLE_RE.sub("", text, count=1).strip()
        # Strip a single pair of wrapping quotes the model sometimes adds.
        if len(text) >= 2 and text[0] in "\"'“「" and text[-1] in "\"'”」":
            text = text[1:-1].strip()
        return text

    def _translate(self, text: str, src_lang: str, tgt_lang: str) -> str:
        # One discrete hop. temperature=0 -> greedy/deterministic so re-runs and
        # cache hits are reproducible. The guardrails reduce (but never fully
        # eliminate) conversational preambles, so _strip_preamble cleans up the
        # output afterward.
        system_prompt = (
            f"You are a professional translation engine. Translate the user's "
            f"message from {src_lang} to {tgt_lang}.\n"
            f"Rules:\n"
            f"- Preserve ALL information, intent, code, names, numbers, and quotations.\n"
            f"- Do NOT answer, explain, or comment on the message.\n"
            f"- Do NOT add any preamble, prefix, label, or note such as "
            f"'Here is the translation:'. Begin directly with the translated text.\n"
            f"- Output ONLY the {tgt_lang} translation and nothing else."
        )
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            temperature=0.0,
            max_tokens=self.max_tokens,
        )
        return self._strip_preamble(out["choices"][0]["message"]["content"])

    def roundtrip(self, text: str) -> str:
        # Walk the chain one discrete hop at a time: EN->ZH->JA->EN.
        for src_lang, tgt_lang in self.hops:
            text = self._translate(text, src_lang, tgt_lang)
        return text


def _save_cache(cache, cache_csv):
    pd.DataFrame(
        {"source": list(cache.keys()), "translated": list(cache.values())}
    ).to_csv(cache_csv, index=False)


def round_trip_translate(texts, translator, cache_csv=RTT_CACHE_CSV, label="", flush_every=25):
    """Translate a list of texts through the translator, preserving order.

    Dedupes identical prompts and caches results to disk (keyed by source text),
    so re-runs — and any overlap between the known/unknown sets — never
    re-translate the same string. Translation is the slow, CPU-bound step.

    The cache is flushed every `flush_every` translations so an interrupt or
    crash never loses more than that many prompts (the prior version wrote only
    once at the very end, so a mid-run crash lost the whole batch).
    """
    cache = {}
    if cache_csv and os.path.exists(cache_csv):
        cached = pd.read_csv(cache_csv).dropna(subset=["source"])
        cache = dict(zip(cached["source"], cached["translated"].fillna("")))

    # Only translate strings we have not already cached.
    unique = list(dict.fromkeys(texts))
    pending = [t for t in unique if t not in cache]
    # Honest accounting: say how many were served from cache vs. actually run.
    print(f"  {label or 'RTT'}: {len(unique) - len(pending)} cached, {len(pending)} to translate")

    for i, src in enumerate(tqdm(pending, desc=f"  RTT translate {label}".rstrip(), leave=False), 1):
        cache[src] = translator.roundtrip(src)
        if cache_csv and i % flush_every == 0:
            _save_cache(cache, cache_csv)  # incremental flush -> crash-safe

    if cache_csv and pending:
        _save_cache(cache, cache_csv)  # final flush for the remainder

    return [cache[t] for t in texts]


def embed_texts(texts, reference_columns, langcode=STYLO_LANGCODE, max_len=MAX_LEN):
    """Re-embed text with StyloMetrix, matching the original embedding pipeline
    (English model, 2048-char truncation — see wildchat/stylometrix.py).

    Asserts the resulting feature columns line up with the original embeddings
    CSV so the attack's distance computation compares like-for-like features.
    """
    import spacy
    import stylo_metrix as sm

    spacy.prefer_gpu()  # use GPU if available, else fall back to CPU (non-fatal)
    stylo = sm.StyloMetrix(langcode)
    emb = stylo.transform([t[:max_len] for t in texts]).drop(columns="text")

    assert list(emb.columns) == list(reference_columns), (
        "StyloMetrix feature columns do not match the original embeddings CSV; "
        "cannot compare original vs. translated embeddings."
    )
    return emb.to_numpy()


def _texts_digest(texts):
    """Stable hash of an ordered list of texts, used as the embedding-cache key."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")  # delimiter so ["ab","c"] != ["a","bc"]
    return h.hexdigest()


def cached_embed(texts, reference_columns, npz_path):
    """embed_texts() with an on-disk cache for the expensive transformer pass.

    The cache is keyed by a digest of the exact input texts, so it is reused only
    when the identical (translated) texts are embedded again. This means a crash
    later in the pipeline never forces a recompute of an already-finished side.
    """
    digest = _texts_digest(texts)
    if npz_path and os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=False)
        if str(data["digest"]) == digest and data["emb"].shape[0] == len(texts):
            print(f"  loaded cached embeddings <- {npz_path}")
            return data["emb"]
        # Digest/row mismatch -> stale cache (texts changed); fall through to recompute.

    emb = embed_texts(texts, reference_columns)
    if npz_path:
        np.savez(npz_path, emb=emb, digest=digest)
        print(f"  saved embeddings -> {npz_path}")
    return emb


def _model_texts(df, model, max_len=MAX_LEN):
    """Conversation text for one model's rows, in the SAME order load_embeddings
    uses (df[mask].reset_index), truncated to match the embedding pipeline. This
    ordering is what keeps the re-embeddings row-aligned with the identities."""
    sub = df[df["model"] == model].reset_index(drop=True)
    return sub["conversation"].str[:max_len].tolist()


# ---------------------------------------------------------------------------
# Defenses
#
# A defense takes the loaded data + the original (undefended) embeddings and
# returns the (known_emb, unknown_emb) the attack should run against. Keeping
# this signature uniform is what makes defenses pluggable: register one in the
# DEFENSES dict below and it shows up in the menu automatically.
#   signature: (df, known, unknown, known_emb, unknown_emb)
#              -> (known_emb, unknown_emb)
# ---------------------------------------------------------------------------

def no_defense(df, known, unknown, known_emb, unknown_emb):
    """Baseline: attack the original embeddings, unchanged."""
    return known_emb, unknown_emb


def _rtt_defense(df, translator, cache_csv, known_npz, unknown_npz):
    """Round-trip-translation defense: BOTH sets are translated and re-embedded.

    Defending known and unknown together is the point — it models the real
    deployment where every prompt passes through the defense before it leaves
    the client. Each side's prompts are translated EN->ZH->JA->EN and re-embedded
    with StyloMetrix. Translations and embeddings are cached to disk (per-backend
    files), so the run is resumable and only the first pass pays the full cost.

    `translator` is any object with a `.roundtrip(text) -> text` method, which is
    what lets the Argos and NLLB backends share this exact pipeline.
    """
    reference_columns = pd.read_csv(EMBEDDINGS_CSV, nrows=0).drop(columns="text").columns

    print("Defending KNOWN set (translate -> embed)...")
    known_trans = round_trip_translate(_model_texts(df, KNOWN_MODEL), translator, cache_csv=cache_csv, label="KNOWN")
    known_emb = cached_embed(known_trans, reference_columns, known_npz)

    print("Defending UNKNOWN set (translate -> embed)...")
    unknown_trans = round_trip_translate(_model_texts(df, UNKNOWN_MODEL), translator, cache_csv=cache_csv, label="UNKNOWN")
    unknown_emb = cached_embed(unknown_trans, reference_columns, unknown_npz)

    return known_emb, unknown_emb


def rtt_argos_defense(df, known, unknown, known_emb, unknown_emb):
    """RTT via Argos (offline, fast CPU; zh->ja silently pivots through English)."""
    return _rtt_defense(df, ArgosTranslator(), RTT_CACHE_CSV, RTT_EMB_KNOWN_NPZ, RTT_EMB_UNKNOWN_NPZ)


def rtt_nllb_defense(df, known, unknown, known_emb, unknown_emb):
    """RTT via NLLB-200 (higher quality, true direct zh->ja; wants a GPU)."""
    return _rtt_defense(df, NLLBTranslator(), NLLB_CACHE_CSV, NLLB_EMB_KNOWN_NPZ, NLLB_EMB_UNKNOWN_NPZ)


def rewrite_qwen_defense(df, known, unknown, known_emb, unknown_emb):
    """Style-convergence defense: rewrite every prompt on-device with a 4-bit
    Qwen into one fixed style (REWRITE_PROMPT_HEADER), then re-embed. Reuses the
    RTT pipeline since QwenRewriter exposes the same .roundtrip(text)->text
    contract as the translators."""
    return _rtt_defense(df, QwenRewriter(), REWRITE_CACHE_CSV, REWRITE_EMB_KNOWN_NPZ, REWRITE_EMB_UNKNOWN_NPZ)


def rtt_qwen_defense(df, known, unknown, known_emb, unknown_emb):
    """RTT via the on-device 4-bit Qwen GGUF, driven through EN->ZH->JA->EN as
    three discrete translation steps (GPU-offloaded). Reuses the RTT pipeline
    since QwenTranslator exposes the same .roundtrip(text)->text contract."""
    return _rtt_defense(df, QwenTranslator(), QWEN_RTT_CACHE_CSV, QWEN_RTT_EMB_KNOWN_NPZ, QWEN_RTT_EMB_UNKNOWN_NPZ)


# ---------------------------------------------------------------------------
# Registries — the one place to plug in new modules. Add an entry to either
# dict and it is automatically listed in the menu and runnable. Keys are the
# human-readable names shown to the user.
# ---------------------------------------------------------------------------

ATTACKS = {
    "Euclidean Style": euclidean_style_attack,
    "Cosine Style": cosine_style_attack,
}

DEFENSES = {
    "None": no_defense,
    "RTT (Argos)": rtt_argos_defense,
    "RTT (NLLB)": rtt_nllb_defense,
    "RTT (Qwen 4-bit)": rtt_qwen_defense,
    "Rewrite (Qwen 4-bit)": rewrite_qwen_defense,
}


# ---------------------------------------------------------------------------
# Entry point — interactive menu driven by the registries above
# ---------------------------------------------------------------------------

def choose(title, options):
    """Print a numbered menu for `options` (a dict) and return the chosen key."""
    keys = list(options)
    print(f"\n{title}\n")
    for i, key in enumerate(keys, 1):
        print(f"  {i}. {key}")
    while True:
        raw = input("> ").strip()
        if raw.isdigit() and 1 <= int(raw) <= len(keys):
            return keys[int(raw) - 1]
        print(f"Please enter a number from 1 to {len(keys)}.")


def main():
    print("Loading data...")
    df = load_data()

    print("Loading embeddings...")
    known, unknown, known_emb, unknown_emb = load_embeddings(df)

    defense_name = choose("What defense do you want?", DEFENSES)
    attack_name = choose("What attack?", ATTACKS)

    # Apply the chosen defense (may translate + re-embed), then run the attack.
    known_emb, unknown_emb = DEFENSES[defense_name](df, known, unknown, known_emb, unknown_emb)

    # Encode both choices in the filename so runs don't overwrite each other.
    def slug(name):
        return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")

    output_csv = os.path.join(OUTPUT_DIR, f"wildchat_analysis_{slug(attack_name)}_{slug(defense_name)}.csv")
    print(f"\nRunning [{attack_name}] attack with [{defense_name}] defense -> {output_csv}")
    run_experiment(
        attack_fn=ATTACKS[attack_name],
        known=known,
        unknown=unknown,
        known_emb=known_emb,
        unknown_emb=unknown_emb,
        output_csv=output_csv,
    )


if __name__ == "__main__":
    main()
