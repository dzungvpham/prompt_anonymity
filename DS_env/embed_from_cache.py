"""Embed an already-translated defense straight from its per-turn cache.

Why this exists: the StyloMetrix embedder needs spaCy's `en_core_web_trf`, which
pulls in `spacy-transformers`. That package only supports the transformers 4.x
line, while the defense backends (NLLB-200, Llama-3-8B StyleRemix) need a modern
transformers — so the embed step and the defense step cannot share one env.

Because every text-rewriting defense caches its work as a per-turn
{source, translated} CSV (see round_trip_translate / round_trip_translate_by_turn),
embedding no longer needs the LLM at all once translation is done. This script
re-joins the cached turns into conversations and writes the SAME .npz embedding
files the main pipeline expects (identical digests), so a later attack run loads
them straight from cache. Run it in an env with a spacy-transformers-compatible
transformers (e.g. transformers<4.37) plus `en_core_web_trf` installed.

Usage:
    python embed_from_cache.py "StyleRemix (Llama-3-8B LoRA)"            # both sides
    python embed_from_cache.py "StyleRemix (Llama-3-8B LoRA)" unknown   # unknown only
    python embed_from_cache.py "StyleRemix (Llama-3-8B LoRA)" known
    python embed_from_cache.py                                          # list defenses

Pick the side(s) you actually ran the defense on. An undefended side keeps its
ORIGINAL embeddings at attack time (see _rtt_defense), so it needs no cache and
should be skipped here. Only translation for the chosen side must be COMPLETE:
any turn missing from the cache aborts with an error (nothing is silently
re-translated here — this script never loads a model).
"""

import sys

import pandas as pd

import stylometric_attacks as s


class _CacheOnlyTranslator:
    """Stand-in backend that must never be called: if round_trip_translate asks it
    to translate a turn, that turn is missing from the cache, so we fail loudly
    instead of silently loading a model this env can't run."""

    def roundtrip(self, text):
        raise RuntimeError(
            "Uncached turn encountered — translation is incomplete for this "
            f"defense. Finish the translate pass first. Turn was:\n{text[:200]!r}"
        )


# side keyword -> which (model, npz-key) pairs to embed.
_SIDES = {
    "both": [("known", "known_npz"), ("unknown", "unknown_npz")],
    "known": [("known", "known_npz")],
    "unknown": [("unknown", "unknown_npz")],
}
_MODELS = {"known": s.KNOWN_MODEL, "unknown": s.UNKNOWN_MODEL}


def embed_defense(name, side="both"):
    spec = s.DEFENSE_SPECS[name]
    df = s.load_data()
    reference_columns = pd.read_csv(s.EMBEDDINGS_CSV, nrows=0).drop(columns="text").columns

    for which, npz_key in _SIDES[side]:
        model, npz = _MODELS[which], spec[npz_key]
        print(f"\n=== {name} :: {which} ({model}) ===")
        # All turns for this side are already cached, so the stand-in translator is
        # never called; this only re-joins the cached turns back into conversations.
        joined = s.round_trip_translate_by_turn(
            s._model_texts(df, model), _CacheOnlyTranslator(),
            cache_csv=spec["cache_csv"], label=model,
        )
        s.cached_embed(joined, reference_columns, npz)
        print(f"  embedded {len(joined)} conversations -> {npz}")


def main():
    args = sys.argv[1:]
    name = args[0] if args else None
    side = args[1].lower() if len(args) > 1 else "both"
    if name not in s.DEFENSE_SPECS or side not in _SIDES:
        print("Usage: python embed_from_cache.py <defense> [both|known|unknown]\n")
        print("Defenses:")
        for n in s.DEFENSE_SPECS:
            print(f"  {n!r}")
        sys.exit(1)
    embed_defense(name, side)


if __name__ == "__main__":
    main()
