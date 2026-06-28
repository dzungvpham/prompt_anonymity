#!/usr/bin/env python3
"""Token-count the prompt column of a dataset CSV.

The prompt text in this project lives in the `conversation` column (there is no
`prompt` column), so that is the default; override with --column.

Uses tiktoken's o200k_base encoding by default, which is the BPE used by the
gpt-4o / gpt-oss family — i.e. the same tokenizer that bills the OpenAnonymity
(OpenRouter) defense. Install it with:  uv pip install tiktoken

Examples:
  python count_tokens.py                       # default CSV + conversation column
  python count_tokens.py --group-by model      # per-model breakdown
  python count_tokens.py --max-len 2048        # match the defense's char truncation
  python count_tokens.py --csv other.csv --column text
"""
import argparse
import sys

import numpy as np
import pandas as pd

# Default CSV matches the one the attack/defense pipeline uses.
try:
    import stylometric_attacks as sa
    DEFAULT_CSV = sa.DATA_CSV
    DEFAULT_MAX_LEN = sa.MAX_LEN
except Exception:
    DEFAULT_CSV = "wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv"
    DEFAULT_MAX_LEN = 2048


def get_tokenizer(encoding):
    try:
        import tiktoken
    except ImportError:
        sys.exit(
            "tiktoken is not installed. Install it with:\n"
            "    uv pip install tiktoken\n"
            "(o200k_base is the gpt-4o / gpt-oss tokenizer used for billing.)"
        )
    enc = tiktoken.get_encoding(encoding)
    # encode_ordinary skips special-token handling -> faster and safe for raw text.
    return lambda s: len(enc.encode_ordinary(s))


def summarize(label, token_counts):
    tc = np.asarray(token_counts)
    n = len(tc)
    total = int(tc.sum())
    print(f"\n{label}")
    print(f"  rows           : {n:,}")
    print(f"  total tokens   : {total:,}")
    print(f"  mean / row     : {tc.mean():.1f}")
    print(f"  median / row   : {np.median(tc):.0f}")
    print(f"  min / max      : {tc.min()} / {tc.max()}")
    print(f"  p90 / p95 / p99: {np.percentile(tc,90):.0f} / "
          f"{np.percentile(tc,95):.0f} / {np.percentile(tc,99):.0f}")
    return total


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=DEFAULT_CSV, help="dataset CSV path")
    ap.add_argument("--column", default="conversation",
                    help="text column to token-count (default: conversation)")
    ap.add_argument("--encoding", default="o200k_base",
                    help="tiktoken encoding (default: o200k_base = gpt-4o/oss)")
    ap.add_argument("--max-len", type=int, default=None,
                    help=f"truncate each text to N chars before counting "
                         f"(use {DEFAULT_MAX_LEN} to match the defense pipeline)")
    ap.add_argument("--group-by", default=None,
                    help="also break totals down by this column (e.g. model)")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    if args.column not in df.columns:
        sys.exit(f"Column {args.column!r} not in CSV. Available: {list(df.columns)}")

    texts = df[args.column].fillna("").astype(str)
    if args.max_len is not None:
        texts = texts.str[:args.max_len]

    tok = get_tokenizer(args.encoding)
    counts = texts.map(tok)

    trunc = f" (truncated to {args.max_len} chars)" if args.max_len is not None else ""
    print(f"CSV      : {args.csv}")
    print(f"Column   : {args.column}{trunc}")
    print(f"Encoding : {args.encoding}")

    grand = summarize("ALL ROWS", counts)

    if args.group_by:
        if args.group_by not in df.columns:
            sys.exit(f"--group-by column {args.group_by!r} not in CSV.")
        for key, idx in df.groupby(args.group_by).groups.items():
            summarize(f"{args.group_by} = {key}", counts.loc[idx])

    print(f"\nGRAND TOTAL tokens in {args.column!r}: {grand:,}")


if __name__ == "__main__":
    main()
