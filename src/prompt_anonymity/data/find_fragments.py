"""Find `author_id`s that are fragments of the same person, for manual verification.

A WildChat identity is ``hashed_ip | accept_language | device_info``. On a mobile carrier the IP
rotates while the other two components stay fixed, so **one person forks into several
``author_id``s**. That is the opposite of the relay problem (one label covering many people), and
it is arguably worse for the linkage study: a fragmented author makes a *correct* attack look
wrong, because the attacker links an unknown document to the right human under a different label
and is scored as a miss.

This is a **ground-truth cleaning** tool, not part of the attack. ``accept_language`` /
``device_info`` / ``country`` are label-construction metadata the attacker never sees, and content
similarity here decides whether two labels denote the same person rather than performing an
attack -- so both are fair game. **Metadata for recall, content for precision.**

Method
------
1. **Block** on ``(accept_language, device_info, country)`` -- the components that survive an IP
   change.
2. **Constrain.** Within a block, keep pairs whose activity intervals are **disjoint** (an overlap
   means two people sharing a fingerprint, never one person) with a gap under ``--max-gap`` days.
3. **Score with content.** Build a char-n-gram TF-IDF centroid per author and take the cosine
   between the two candidates, calibrated against a null (the same author's mean similarity to
   random authors). ``lift = sim / null_sim`` is the headline number.
4. **Assemble chains** by union-find over accepted pairs, rejecting any chain whose members'
   intervals overlap.

Timing alone cannot call individual pairs -- naming *which* authors needs the content signal, and
even then the output is a **ranked shortlist for manual verification**, not a decided merge. See
``data/fragmentation_findings.md`` for the precision estimate.

Usage
-----
    python -m prompt_anonymity.data.find_fragments                      # defaults
    python -m prompt_anonymity.data.find_fragments --min-lift 3 --max-gap 45

Writes ``data/fragment_chains.csv`` (one row per fragment, grouped into chains, with a sample
prompt for eyeballing) and ``data/fragment_pairs.csv`` (every scored pair, for tuning the
threshold yourself).
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import data_dir, hf_dir, raw_path
from .identity import (
    hash_author_id,
    is_programmatic_user_agent,
    load_ua_device_map,
    wildchat_device_info,
    wildchat_identity,
)
from .sources_wildchat import WILDCHAT_MODELS

BLOCK_COLUMNS = ["accept_language", "device_info", "country"]


def scan_identities(raw_dir: str | Path, models: list[str], min_convs: int = 2) -> pd.DataFrame:
    """One linear pass over the raw metadata -> one row per identity.

    Only the small metadata columns are read (never ``conversation``), so this is cheap. Rows from
    programmatic clients are skipped, matching the dataset build.
    """
    ua_map = load_ua_device_map()
    dataset = ds.dataset(str(raw_dir), format="parquet")
    columns = ["hashed_ip", "header", "country", "state", "timestamp"]
    agg: dict = {}
    bot_cache: dict = {}
    scanner = dataset.scanner(columns=columns, filter=ds.field("model").isin(models), batch_size=4000)
    for batch in scanner.to_batches():
        if batch.num_rows == 0:
            continue
        cols = {name: batch.column(name).to_pylist() for name in columns}
        for i in range(batch.num_rows):
            header = cols["header"][i] or {}
            user_agent = header.get("user-agent")
            if user_agent not in bot_cache:
                bot_cache[user_agent] = is_programmatic_user_agent(user_agent, ua_map)
            if bot_cache[user_agent]:
                continue
            accept_language = header.get("accept-language") or ""
            device_info = wildchat_device_info(user_agent, ua_map)
            key = (cols["hashed_ip"][i], accept_language, device_info)
            timestamp = cols["timestamp"][i]
            record = agg.get(key)
            if record is None:
                agg[key] = [1, timestamp, timestamp, collections.Counter([cols["state"][i]]), cols["country"][i]]
            else:
                record[0] += 1
                record[1] = min(record[1], timestamp)
                record[2] = max(record[2], timestamp)
                record[3][cols["state"][i]] += 1
    rows = [
        {
            "author_id": hash_author_id("wildchat", wildchat_identity(*key)),
            "accept_language": key[1], "device_info": key[2], "country": value[4],
            "state": value[3].most_common(1)[0][0], "n_convs": value[0],
            "first": value[1], "last": value[2],
        }
        for key, value in agg.items() if value[0] >= min_convs
    ]
    frame = pd.DataFrame(rows)
    frame["first"] = pd.to_datetime(frame["first"])
    frame["last"] = pd.to_datetime(frame["last"])
    return frame


def candidate_pairs(identities: pd.DataFrame, max_gap_days: float, max_block: int) -> pd.DataFrame:
    """Within-block pairs with disjoint intervals and a gap under ``max_gap_days``.

    Blocks larger than ``max_block`` are skipped: a fingerprint shared by hundreds of identities
    carries too little information for the timing constraint to mean anything.
    """
    out = []
    for _, group in identities.groupby(BLOCK_COLUMNS, sort=False):
        size = len(group)
        if size < 2 or size > max_block:
            continue
        group = group.sort_values("first")
        first = group["first"].to_numpy()
        last = group["last"].to_numpy()
        author = group["author_id"].to_numpy()
        state = group["state"].to_numpy()
        n_convs = group["n_convs"].to_numpy()
        for i in range(size):
            for j in range(i + 1, size):
                gap = (first[j] - last[i]) / np.timedelta64(1, "D")
                if gap < 0:      # overlapping intervals -> two people, never one
                    continue
                if gap > max_gap_days:   # sorted by start, so no later j can qualify
                    break
                out.append((author[i], author[j], round(float(gap), 2), size,
                            int(n_convs[i]), int(n_convs[j]), bool(state[i] == state[j])))
    return pd.DataFrame(out, columns=["a", "b", "gap_days", "block_size", "n_a", "n_b", "same_state"])


def score_with_content(pairs: pd.DataFrame, documents: pd.DataFrame, *, n_null: int = 40,
                       seed: int = 0) -> pd.DataFrame:
    """Add ``sim``, ``null_sim`` and ``lift`` to each candidate pair.

    ``sim`` is the cosine between the two authors' char-n-gram TF-IDF centroids. ``null_sim`` is
    the same author's mean cosine against ``n_null`` random authors, which calibrates for authors
    who are simply generic. ``lift = sim / null_sim`` is what to threshold on.
    """
    involved = sorted(set(pairs["a"]) | set(pairs["b"]))
    text = (documents[documents["author_id"].isin(involved)]
            .groupby("author_id")["_text"].apply(lambda s: "\n".join(s)[:200_000]))
    text = text.reindex(involved).fillna("")
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                 max_features=300_000, sublinear_tf=True)
    matrix = vectorizer.fit_transform(text.to_numpy())      # L2-normalized rows
    index = {author: i for i, author in enumerate(text.index)}

    ia = pairs["a"].map(index).to_numpy()
    ib = pairs["b"].map(index).to_numpy()
    sim = np.asarray(matrix[ia].multiply(matrix[ib]).sum(axis=1)).ravel()

    rng = np.random.default_rng(seed)
    sample = rng.choice(matrix.shape[0], size=min(n_null, matrix.shape[0]), replace=False)
    null_by_row = np.asarray((matrix @ matrix[sample].T).mean(axis=1)).ravel()
    null_sim = (null_by_row[ia] + null_by_row[ib]) / 2

    scored = pairs.copy()
    scored["sim"] = sim.round(4)
    scored["null_sim"] = null_sim.round(4)
    scored["lift"] = (sim / np.maximum(null_sim, 1e-9)).round(2)
    return scored.sort_values("lift", ascending=False).reset_index(drop=True)


def assemble_chains(accepted: pd.DataFrame, identities: pd.DataFrame) -> pd.DataFrame:
    """Greedily grow chains from the strongest links, keeping every chain temporally disjoint.

    Plain union-find is wrong here: transitive closure over a permissive similarity threshold
    pulls unrelated identities into one blob, and a single overlapping member then invalidates
    the whole chain. Instead, links are considered strongest-first and a merge is only accepted
    when the union stays temporally disjoint (no two members overlap, since one person cannot be
    active as two identities at once) -- a link that would violate that is skipped rather than
    poisoning the cluster.
    """
    meta = identities.set_index("author_id")
    spans = {a: (r.first, r.last) for a, r in meta.iterrows()}

    chain_of: dict = {}                       # author -> chain key
    members: dict = {}                        # chain key -> list of authors

    def disjoint(authors) -> bool:
        iv = sorted((spans[a] for a in authors if a in spans))
        return all(iv[i + 1][0] >= iv[i][1] for i in range(len(iv) - 1))

    for row in accepted.sort_values("sim", ascending=False).itertuples():
        if row.a not in spans or row.b not in spans:
            continue
        ca, cb = chain_of.get(row.a), chain_of.get(row.b)
        if ca is not None and ca == cb:
            continue
        group = set(members.get(ca, [row.a])) | set(members.get(cb, [row.b]))
        if not disjoint(group):
            continue                          # would put one person in two places at once
        key = ca or cb or row.a
        members.pop(ca, None)
        members.pop(cb, None)
        members[key] = sorted(group)
        for author in group:
            chain_of[author] = key

    rows = []
    for chain_id, (_, group) in enumerate(sorted(members.items()), start=1):
        if len(group) < 2:
            continue
        sub = meta.loc[group].sort_values("first")
        prev_last = None
        for author, r in sub.iterrows():
            rows.append({
                "chain_id": chain_id, "author_id": author, "n_convs": r.n_convs,
                "first": r.first, "last": r.last, "state": r.state, "country": r.country,
                "gap_from_prev": None if prev_last is None else round((r.first - prev_last).total_seconds() / 86400, 2),
                "accept_language": r.accept_language, "device_info": r.device_info,
            })
            prev_last = r.last
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw", default=None,
                   help="directory of raw WildChat parquet shards (default: $PROMPT_ANONYMITY_"
                        "WILDCHAT_RAW, else the config file, else downloaded from HuggingFace)")
    p.add_argument("--built", default=None,
                   help="the built wildchat.parquet (default: the project's data/hf)")
    p.add_argument("--max-gap", type=float, default=45.0, help="max days between fragments (default 45)")
    p.add_argument("--max-block", type=int, default=200, help="skip fingerprints shared by more identities")
    p.add_argument("--min-sim", type=float, default=0.6,
                   help="cosine between the two authors' char-n-gram TF-IDF centroids (default 0.6); "
                        "this is a shortlist to verify, not a decided answer")
    p.add_argument("--out-dir", default=None,
                   help="where the two CSVs go (default: the project's data/)")
    args = p.parse_args()

    print("scanning raw metadata ...")
    identities = scan_identities(raw_path("wildchat", args.raw), WILDCHAT_MODELS)
    print(f"  identities: {len(identities):,}   blocks: {identities.groupby(BLOCK_COLUMNS).ngroups:,}")

    pairs = candidate_pairs(identities, args.max_gap, args.max_block)
    print(f"candidate pairs (disjoint, gap<={args.max_gap}d): {len(pairs):,}")

    documents = pd.read_parquet(args.built or hf_dir() / "wildchat.parquet",
                                columns=["author_id", "turns"])
    documents["_text"] = documents["turns"].map(lambda t: "\n".join(t))
    known = set(documents["author_id"])
    pairs = pairs[pairs["a"].isin(known) & pairs["b"].isin(known)].reset_index(drop=True)
    print(f"  ...both fragments present in the built dataset: {len(pairs):,}")

    print("scoring with content similarity ...")
    scored = score_with_content(pairs, documents)
    accepted = scored[scored["sim"] >= args.min_sim]
    print(f"  accepted at sim>={args.min_sim}: {len(accepted):,} pairs")

    chains = assemble_chains(accepted, identities)
    n_chains = chains["chain_id"].nunique() if len(chains) else 0
    print(f"  chains: {n_chains:,} covering {len(chains):,} author_ids")

    # attach a sample prompt per fragment so the output can be eyeballed directly
    sample = (documents.groupby("author_id")["_text"]
              .apply(lambda s: " ".join(s.iloc[0].split())[:160]))
    if len(chains):
        chains["sample_prompt"] = chains["author_id"].map(sample)
        chains["docs_in_dataset"] = chains["author_id"].map(documents["author_id"].value_counts())

    out = Path(args.out_dir) if args.out_dir else data_dir()
    scored.to_csv(out / "fragment_pairs.csv", index=False)
    chains.to_csv(out / "fragment_chains.csv", index=False)
    print(f"wrote {out/'fragment_pairs.csv'} and {out/'fragment_chains.csv'}")


if __name__ == "__main__":
    main()
