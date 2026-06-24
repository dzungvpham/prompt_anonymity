import hashlib
import os
import random

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from tqdm import tqdm

# --- Config ---
LANG = ("English", "en")
KNOWN_MODEL = "gpt-4o-2024-08-06"
UNKNOWN_MODEL = "gpt-4.1-mini-2025-04-14"
DATA_CSV = "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv"
EMBEDDINGS_CSV = f"wildchat_filtered_{LANG[1]}_2048_stylometrix.csv"
OUTPUT_CSV = "wildchat_analysis_stylometrix_results.csv"
N_SIM = 100
SAMPLE_STEP = 25
RANDOM_SEED = 47

# --- Round-trip-translation (RTT) defense config ---
MAX_LEN = 2048                        # char truncation, matches wildchat/stylometrix.py
STYLO_LANGCODE = "en"                 # final text is English -> embed with the English model
RTT_CACHE_CSV = "wildchat_rtt_translation_cache.csv"    # source -> translated, shared & resumable
RTT_OUTPUT_CSV = "wildchat_analysis_euclidean_rtt.csv"  # defended attack results
RTT_EMB_KNOWN_NPZ = "wildchat_rtt_known_emb.npz"        # cached StyloMetrix embeddings (known, translated)
RTT_EMB_UNKNOWN_NPZ = "wildchat_rtt_unknown_emb.npz"    # cached StyloMetrix embeddings (unknown, translated)


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


def run_rtt_defense(
    attack_fn=euclidean_style_attack,
    defend_known=True,
    defend_unknown=True,
    translator=None,
    output_csv=RTT_OUTPUT_CSV,
):
    """Apply the round-trip-translation defense, then run the linkage attack.

    defend_known / defend_unknown toggle which side is translated:
      - both True (default) -> realistic "everyone runs the extension" deployment
      - defend_unknown only -> worst case: attacker holds a pristine reference set

    Compare the resulting CSV against the undefended baseline (main()): a working
    defense lowers top-1/top-5 id_acc toward the random_id column.
    """
    print("Loading data...")
    df = load_data()

    print("Loading embeddings...")
    known, unknown, known_emb, unknown_emb = load_embeddings(df)

    # Original feature column order, used to validate the re-embeddings.
    reference_columns = pd.read_csv(EMBEDDINGS_CSV, nrows=0).drop(columns="text").columns

    if translator is None:
        translator = ArgosTranslator()

    # Re-embed each defended side from its translated text. Sides that are not
    # defended keep their original embeddings loaded above. round_trip_translate
    # reports its own cached-vs-to-translate counts, so no misleading print here.
    if defend_known:
        print("Defending KNOWN set (translate -> embed)...")
        known_trans = round_trip_translate(_model_texts(df, KNOWN_MODEL), translator, label="KNOWN")
        known_emb = cached_embed(known_trans, reference_columns, RTT_EMB_KNOWN_NPZ)

    if defend_unknown:
        print("Defending UNKNOWN set (translate -> embed)...")
        unknown_trans = round_trip_translate(_model_texts(df, UNKNOWN_MODEL), translator, label="UNKNOWN")
        unknown_emb = cached_embed(unknown_trans, reference_columns, RTT_EMB_UNKNOWN_NPZ)

    run_experiment(
        attack_fn=attack_fn,
        known=known,
        unknown=unknown,
        known_emb=known_emb,
        unknown_emb=unknown_emb,
        output_csv=output_csv,
    )


# ---------------------------------------------------------------------------
# Entry point — swap attack_fn here to test a different attack
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df = load_data()

    print("Loading embeddings...")
    known, unknown, known_emb, unknown_emb = load_embeddings(df)

    run_experiment(
        attack_fn=euclidean_style_attack,
        known=known,
        unknown=unknown,
        known_emb=known_emb,
        unknown_emb=unknown_emb,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Stylometric linkage attack / RTT defense")
    parser.add_argument(
        "--defense",
        action="store_true",
        help="run the round-trip-translation defense instead of the undefended baseline",
    )
    parser.add_argument(
        "--defend-side",
        choices=["both", "unknown"],
        default="both",
        help="which set to translate: both (default, deployment model) or unknown-only (worst case)",
    )
    args = parser.parse_args()

    if args.defense:
        run_rtt_defense(defend_known=(args.defend_side == "both"), defend_unknown=True)
    else:
        main()
