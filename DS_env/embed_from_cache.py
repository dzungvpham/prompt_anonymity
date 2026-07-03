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
    python embed_from_cache.py "StyleRemix (Llama-3-8B LoRA)"
    python embed_from_cache.py            # lists the available defense names

Prereq: the defense's translation must be COMPLETE for both sides. Any turn that
is missing from the cache aborts with an error (nothing is silently re-translated
here — this script never loads a model).
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


def embed_defense(name):
    spec = s.DEFENSE_SPECS[name]
    df = s.load_data()
    reference_columns = pd.read_csv(s.EMBEDDINGS_CSV, nrows=0).drop(columns="text").columns

    for model, npz in [(s.KNOWN_MODEL, spec["known_npz"]),
                       (s.UNKNOWN_MODEL, spec["unknown_npz"])]:
        print(f"\n=== {name} :: {model} ===")
        # All turns are already cached, so the stand-in translator is never called;
        # this only re-joins the cached turns back into conversations.
        joined = s.round_trip_translate_by_turn(
            s._model_texts(df, model), _CacheOnlyTranslator(),
            cache_csv=spec["cache_csv"], label=model,
        )
        s.cached_embed(joined, reference_columns, npz)
        print(f"  embedded {len(joined)} conversations -> {npz}")


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in s.DEFENSE_SPECS:
        print("Pick one defense to embed from its cache:\n")
        for n in s.DEFENSE_SPECS:
            print(f"  {n!r}")
        sys.exit(1)
    embed_defense(sys.argv[1])


if __name__ == "__main__":
    main()
