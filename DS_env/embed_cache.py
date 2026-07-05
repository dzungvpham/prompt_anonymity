# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "numpy",
#   "pandas",
#   "click",
#   "transformers>=4.36,<4.37",
#   "spacy>=3.8,<3.9",
#   "spacy-transformers>=1.3.4,<1.4",
#   "stylo_metrix",
#   "en-core-web-trf @ https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.8.0/en_core_web_trf-3.8.0-py3-none-any.whl",
# ]
# ///
"""Embed any defense's cached (defended) text with StyloMetrix, for the attack.

Runs via `uv run --no-project embed_cache.py <cache.csv>`. StyloMetrix's English
pipeline is `en_core_web_trf`, which runs through `spacy-transformers` and needs
`transformers < 4.37` — that conflicts with the A100 defense-generation stack
(transformers 5.x), so this script pins its own deps in the PEP 723 header above.
`uv run` builds and caches an isolated env from them; there is no venv to create or
activate, and the A100 `.venv` is left untouched. It imports NONE of
stylometric_attacks.py (whose backends want the heavy stack); the pipeline pieces
it needs are duplicated below and kept faithful to their originals (names noted).

Given a per-turn {source, translated} cache CSV, it:
  1. rebuilds each conversation by splitting the ORIGINAL text on TURN_DELIM and
     re-joining the DEFENDED turns (turns are recombined into a full conversation
     BEFORE embedding — StyloMetrix sees whole conversations, never lone turns),
  2. embeds each rebuilt conversation with StyloMetrix (2048-char truncation, same
     as the reference corpus), and
  3. writes the .npz files stylometric_attacks.py loads — <stem>_{known,unknown}_
     emb.npz next to the cache under defense_data/embeddings/ — including the
     `digest` cached_embed() checks.
A turn absent from the cache is kept as its original text, so a one-sided defense
(e.g. UNKNOWN only) degrades to skipping the undefended side instead of crashing.

Examples:
  uv run --no-project embed_cache.py wildchat_styleremix_cache.csv
  uv run --no-project embed_cache.py defense_data/cache/wildchat_openanon_cache.csv
"""

import argparse
import hashlib
import os

# stylometric_attacks.py uses relative cache/embedding paths, so anchor to this
# file's dir (DS_env) — otherwise a launch from elsewhere reads the wrong files.
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

# --- Config (mirrors stylometric_attacks.py) -------------------------------
LANG = ("English", "en")
KNOWN_MODEL = "gpt-4o-2024-08-06"
UNKNOWN_MODEL = "gpt-4.1-mini-2025-04-14"
DATA_CSV = "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv"
EMBEDDINGS_CSV = f"wildchat_filtered_{LANG[1]}_2048_stylometrix.csv"
MAX_LEN = 2048            # char truncation, matches wildchat/stylometrix.py
STYLO_LANGCODE = "en"     # final text is English -> embed with the English model

# On-disk turn separator: the literal string "\n===\n" (real newlines were stored
# as the two-char sequence "\n"), NOT real newlines. See stylometric_attacks.py.
TURN_DELIM = "\\n===\\n"

# Defense artifacts live under defense_data/{cache,embeddings}.
CACHE_DIR = os.path.join("defense_data", "cache")
EMB_DIR = os.path.join("defense_data", "embeddings")


def npz_paths_for(cache_csv):
    """Derive the (known_npz, unknown_npz) output paths from a cache filename,
    reproducing stylometric_attacks.py's naming: strip a trailing
    `_translation_cache`/`_cache` from the stem, then add `_{known,unknown}_emb.npz`
    under defense_data/embeddings/. E.g.
      wildchat_styleremix_cache.csv            -> wildchat_styleremix_{known,unknown}_emb.npz
      wildchat_rtt_translation_cache.csv       -> wildchat_rtt_{known,unknown}_emb.npz
    """
    stem = os.path.splitext(os.path.basename(cache_csv))[0]
    for suffix in ("_translation_cache", "_cache"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return (os.path.join(EMB_DIR, f"{stem}_known_emb.npz"),
            os.path.join(EMB_DIR, f"{stem}_unknown_emb.npz"))


def load_data() -> pd.DataFrame:
    """Mirror of stylometric_attacks.load_data(): English rows, identities with
    >=2 distinct models, stable-sorted so row order matches the identities."""
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


def model_texts(df, model):
    """Mirror of stylometric_attacks._model_texts(): one model's conversations in
    load_embeddings' order (df[mask].reset_index), keeping rows identity-aligned."""
    return df[df["model"] == model].reset_index(drop=True)["conversation"].tolist()


def texts_digest(texts):
    """Mirror of stylometric_attacks._texts_digest(): stable hash of the ordered
    texts, used as cached_embed's key. Must match byte-for-byte so the attack
    accepts this .npz as its own."""
    h = hashlib.sha256()
    for t in texts:
        h.update(str(t).encode("utf-8", "replace"))
        h.update(b"\x00")  # delimiter so ["ab","c"] != ["a","bc"]
    return h.hexdigest()


def embed_texts(texts, reference_columns, langcode=STYLO_LANGCODE, max_len=MAX_LEN):
    """Mirror of stylometric_attacks.embed_texts(): StyloMetrix English model,
    2048-char truncation, with a column-alignment assert so the re-embeddings are
    like-for-like with the reference corpus. `texts` are WHOLE recombined
    conversations (see rebuild), not individual turns."""
    import spacy
    import stylo_metrix as sm

    spacy.prefer_gpu()  # GPU if available, else CPU (non-fatal)
    stylo = sm.StyloMetrix(langcode)
    emb = stylo.transform([t[:max_len] for t in texts]).drop(columns="text")

    assert list(emb.columns) == list(reference_columns), (
        "StyloMetrix feature columns do not match the original embeddings CSV; "
        "cannot compare original vs. translated embeddings."
    )
    return emb.to_numpy()


def rebuild(model, lut):
    """Recombine each conversation from its DEFENDED turns, in load_data() order so
    rows stay aligned with the identities. Splits the original conversation on
    TURN_DELIM, swaps in the cached defended turn per part (keeping the original
    when a turn is absent from the cache), then re-joins with TURN_DELIM so the
    result is one whole conversation string ready to embed."""
    texts, hit, miss = [], 0, 0
    for conv in model_texts(load_data(), model):
        parts = []
        for turn in str(conv).split(TURN_DELIM):
            if turn in lut:
                parts.append(lut[turn]); hit += 1
            else:
                parts.append(turn); miss += 1 if turn.strip() else 0
        texts.append(TURN_DELIM.join(parts))  # <- turns recombined before embedding
    print(f"  {model}: {hit} turns from cache, {miss} non-empty turns missing (kept original)")
    return texts, hit


def main():
    ap = argparse.ArgumentParser(description="Embed a defense's cached defended text with StyloMetrix.")
    ap.add_argument("cache_csv", help="per-turn {source, translated} cache CSV "
                                       "(bare name is resolved under defense_data/cache/)")
    ap.add_argument("--sides", choices=["unknown", "known", "both"], default="unknown",
                    help="which side(s) the defense was actually run on, i.e. which to embed. "
                         "Default 'unknown'. Only these sides are written; the other keeps its "
                         "original precomputed embedding. This is EXPLICIT on purpose: trivial "
                         "turns ('hi', 'thanks') collide across sides, so cache hits alone cannot "
                         "tell a defended side from incidental overlap.")
    args = ap.parse_args()

    cache_csv = args.cache_csv
    if not os.path.exists(cache_csv):
        cache_csv = os.path.join(CACHE_DIR, os.path.basename(cache_csv))
    known_npz, unknown_npz = npz_paths_for(cache_csv)

    ref_cols = pd.read_csv(EMBEDDINGS_CSV, nrows=0).drop(columns="text").columns

    # turn -> defended turn, loaded exactly as the attack does (drop null source,
    # null translated -> "") so a complete cache reproduces the attack's own texts.
    cache = pd.read_csv(cache_csv).dropna(subset=["source"])
    lut = dict(zip(cache["source"], cache["translated"].fillna("")))
    print(f"{os.path.basename(cache_csv)}: {len(lut)} defended turns | sides={args.sides}")

    todo = {"unknown": [(UNKNOWN_MODEL, unknown_npz)],
            "known": [(KNOWN_MODEL, known_npz)],
            "both": [(KNOWN_MODEL, known_npz), (UNKNOWN_MODEL, unknown_npz)]}[args.sides]

    for model, npz in todo:
        texts, hit = rebuild(model, lut)
        if hit == 0:
            print(f"  -> no cached turns for {model}; skipping (side not defended).")
            continue
        emb = embed_texts(texts, ref_cols)
        np.savez(npz, emb=emb, digest=texts_digest(texts))
        print(f"  saved {emb.shape} -> {npz}")


if __name__ == "__main__":
    main()
