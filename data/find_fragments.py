"""Find `author_id`s that are fragments of the same person, for manual verification.

A WildChat identity is ``hashed_ip | accept_language | device_info``. On a mobile carrier the IP
rotates while the other two components stay fixed, so **one person forks into several
``author_id``s**. That is the opposite of the relay problem (one label covering many people), and
it is arguably worse for the linkage study: a fragmented author makes a *correct* attack look
wrong, because the attacker links an unknown document to the right human under a different label
and is scored as a miss.

This is a **ground-truth cleaning** tool, not part of the attack. The distinction matters for what
evidence is fair game:

* ``accept_language`` / ``device_info`` / ``country`` are *label-construction* metadata. The
  attacker never sees them, so using them here does not leak anything into the experiment.
* **Content similarity is fair game too** -- it is being used to decide whether two labels denote
  the same person, not to perform the attack. (It would be circular only if the merged labels were
  then used to score a content-based attack *as though* the merge were independent evidence.)

So: **metadata for recall, content for precision.**

Method
------
1. **Block** on ``(accept_language, device_info, country)`` -- the components that survive an IP
   change. O(N) hash grouping, so this never materializes an N^2 comparison over the corpus.
2. **Constrain.** Within a block, keep pairs whose activity intervals are **disjoint** (an overlap
   means two people sharing a fingerprint, never one person) with a gap under ``--max-gap`` days.
3. **Score with content.** Build a char-n-gram TF-IDF centroid per author and take the cosine
   between the two candidates. Each pair is also compared against a null: the same author's mean
   similarity to random authors in the corpus. ``lift = sim / null_sim`` is the headline number.
4. **Assemble chains** by union-find over accepted pairs, rejecting any chain whose members'
   intervals overlap.

Precision, measured honestly: against a control of pairs matched on primary language and forced to
be temporally disjoint but drawn from *different* fingerprint blocks, same-fingerprint candidates
are only ~2.7x enriched at ``sim >= 0.5``. So roughly **half** the accepted pairs are expected to
be coincidence. Treat the output as a **ranked shortlist for manual verification**, not a decided
merge. (The control is itself conservative: someone who changes phone gets a new ``device_info``
and lands in the control while genuinely being a fragment, so true precision is somewhat better.)

Why content is required: metadata alone cannot call individual pairs. Scoring pairs purely on how
surprising their timing is, given block density, and correcting for multiple testing across ~10^6
candidate pairs accepts *zero* pairs -- in a block of 82 identities over 359 days a 5-day gap is
unremarkable. The aggregate excess over a permutation null is real (~200x), so fragmentation is
pervasive, but naming *which* authors needs the content signal. See
``data/fragmentation_findings.md``.

Usage
-----
    PYTHONPATH=. python -m data.find_fragments                      # defaults
    PYTHONPATH=. python -m data.find_fragments --min-lift 3 --max-gap 45

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

from .build_dataset import WILDCHAT_RAW
from .identity import (
    hash_author_id,
    is_programmatic_user_agent,
    load_ua_device_map,
    wildchat_device_info,
    wildchat_identity,
)
from .sources_wildchat import WILDCHAT_MODELS

BLOCK_COLUMNS = ["accept_language", "device_info", "country"]
DIST = Path(__file__).with_name("dist")


def scan_identities(raw_path: str, models: list[str], min_convs: int = 2) -> pd.DataFrame:
    """One linear pass over the raw metadata -> one row per identity.

    Only the small metadata columns are read (never ``conversation``), so this is cheap. Rows from
    programmatic clients are skipped, matching the dataset build.
    """
    ua_map = load_ua_device_map()
    dataset = ds.dataset(raw_path, format="parquet")
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

    Plain union-find is wrong here. Transitive closure over a permissive similarity threshold
    pulls unrelated identities into one blob -- the verified four-fragment Korean chain ended up
    inside a 54-member cluster -- and a single overlapping member then invalidates the whole thing,
    discarding the good chain along with the bad links.

    Instead: consider links strongest-first and merge two chains only when the union stays
    temporally disjoint (no two members overlap, since one person cannot be active as two
    identities at once). A link that would violate that is skipped rather than poisoning the
    cluster, so a strong chain survives a weak neighbour.
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
    p.add_argument("--raw", default=WILDCHAT_RAW)
    p.add_argument("--built", default=str(DIST / "wildchat.parquet"))
    p.add_argument("--max-gap", type=float, default=45.0, help="max days between fragments (default 45)")
    p.add_argument("--max-block", type=int, default=200, help="skip fingerprints shared by more identities")
    p.add_argument("--min-sim", type=float, default=0.6,
                   help="cosine between the two authors' char-n-gram TF-IDF centroids (default 0.6). "
                        "A language-matched control of different-fingerprint pairs reaches this "
                        "level 1.0%% of the time, so expect roughly half the accepted pairs to be "
                        "coincidence -- this is a shortlist to verify, not a decided answer")
    p.add_argument("--out-dir", default=str(Path(__file__).parent))
    args = p.parse_args()

    print("scanning raw metadata ...")
    identities = scan_identities(args.raw, WILDCHAT_MODELS)
    print(f"  identities: {len(identities):,}   blocks: {identities.groupby(BLOCK_COLUMNS).ngroups:,}")

    pairs = candidate_pairs(identities, args.max_gap, args.max_block)
    print(f"candidate pairs (disjoint, gap<={args.max_gap}d): {len(pairs):,}")

    documents = pd.read_parquet(args.built, columns=["author_id", "turns"])
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

    out = Path(args.out_dir)
    scored.to_csv(out / "fragment_pairs.csv", index=False)
    chains.to_csv(out / "fragment_chains.csv", index=False)
    print(f"wrote {out/'fragment_pairs.csv'} and {out/'fragment_chains.csv'}")


if __name__ == "__main__":
    main()
