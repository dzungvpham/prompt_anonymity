# SWE-chat stylometric linkage (re-identification) attack -- script form of analyze_swe_chat.ipynb.
#
# Each session's scrubbed user prompts (preprocess.py) are embedded with StyloMetrix
# (stylometrix.py). For each identity we hold out the chronologically LAST session as the
# anonymous "unknown" and rank the earlier "known" sessions by negative Euclidean distance --
# the same attack as analyze_wildchat.ipynb.
#
# Identity (user_id is the username; repo_id is "owner/repo"):
#   - user_id when present;
#   - id-less session on a single-user repo        -> that repo's known user_id (merge);
#   - id-less session on an orphan repo (0 users)   -> repo_id (its own identity);
#   - id-less session on a repo shared by >1 user   -> dropped (no valid identifier).
# Then, per identity, dedup sessions by exact content and by MIN_LEN-char prefix/suffix,
# keeping only the FIRST (chronological) session of each duplicate group (unlike wildchat,
# which keeps up to three).
#
# Two adversaries: Global (one pool) and Per-provider (partition by model_owner).
# Run preprocess.py then stylometrix.py (GPU) before this.

import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial.distance import cdist
from tqdm import tqdm

CACHE = "swe_chat_sessions.csv"
FEATS = "swe_chat_stylometrix_en_2048.csv"
MIN_LEN = 50  # prefix/suffix length for dedup (matches wildchat filter.py)


def topk_conv_hit_prob(known_count, total_known, k):
    """Probability that ranking all `total_known` known conversations uniformly at random places
    at least one of a target user's `known_count` conversations in the top k. Equivalent to
    drawing k conversations without replacement and hitting the target at least once:
        1 - prod_{i=0}^{k-1} (total_known - known_count - i) / (total_known - i).
    k is capped at total_known (can't rank more conversations than exist)."""
    k = min(k, total_known)
    p_miss = 1.0  # probability none of the target's conversations land in the top k
    for i in range(k):
        p_miss *= (total_known - known_count - i) / (total_known - i)
    return 1.0 - p_miss


def build_ranking(sessions_df, n_known=None):
    """Hold out each identity's last session as 'unknown', keep earlier ones as 'known',
    then rank the pool's known sessions per unknown by negative Euclidean StyloMetrix distance.

    The single last session of each identity is always the unknown. n_known (ablation over the
    amount of labeled history) restricts the known set to that identity's first n_known earliest
    sessions and drops the ones in between, so known always precedes the unknown; identities with
    <= n_known sessions keep all but their last as known. None keeps every earlier session.

    Returns a dict bundling everything analyze() needs:
      - unknown                    : DataFrame of the held-out unknown sessions
      - unknown_ids                : their identities (Series, aligned to `unknown`)
      - unknown_rows_by_identity   : identity -> row positions of its unknown sessions in `unknown`
      - unknown_count_by_identity  : identity -> number of unknown sessions it has
      - known_count_by_identity    : identity -> number of known sessions it has
      - ranked_correct             : per unknown, booleans over the known-session ranking
                                     (True where that ranked known session is a true match)
      - ranked_identities          : per unknown, the distinct candidate identities in ranked order
    """
    # Sort chronologically within each identity, then split into known (labeled) and unknown.
    sessions_df = sessions_df.sort_values(["identity", "timestamp", "session_id"]).reset_index(drop=True)
    if n_known is None:
        # Default: hold out the last session of each identity as its unknown, keep all earlier.
        sessions_df["is_unknown"] = False
        sessions_df.loc[sessions_df.groupby("identity").tail(1).index, "is_unknown"] = True
    else:
        # Ablation: the last session is the (single) unknown and the first min(n_known, S-1)
        # earliest sessions are the known history; any sessions in between are dropped. So known
        # always precedes the unknown, and a user with <= n_known sessions keeps all but its last.
        within_user = sessions_df.groupby("identity").cumcount()
        n_sessions = sessions_df.groupby("identity")["session_id"].transform("size")
        is_known = within_user < np.minimum(n_known, n_sessions - 1)
        is_unknown = within_user == n_sessions - 1
        keep = is_known | is_unknown
        kept_is_unknown = is_unknown[keep].to_numpy()
        sessions_df = sessions_df[keep].reset_index(drop=True)
        sessions_df["is_unknown"] = kept_is_unknown

    # StyloMetrix feature matrix, one row per session in sessions_df order. StyloMetrix can
    # emit NaN for some features (e.g. ratios with a zero denominator); treat those as 0.
    feature_matrix = np.nan_to_num(
        feature_df.loc[sessions_df["session_id"], feature_cols].to_numpy(dtype=np.float32),
        nan=0.0,
    )
    known_mask = (~sessions_df["is_unknown"]).to_numpy()
    unknown_mask = sessions_df["is_unknown"].to_numpy()
    known_sessions = sessions_df[known_mask].reset_index(drop=True)
    unknown_sessions = sessions_df[unknown_mask].reset_index(drop=True)
    known_ids, unknown_ids = known_sessions["identity"], unknown_sessions["identity"]

    # Similarity = negative Euclidean distance, so closer in style space => higher similarity.
    # similarity[u, k] scores unknown session u against known session k.
    similarity = -cdist(feature_matrix[unknown_mask], feature_matrix[known_mask], metric="euclidean")
    # For each unknown, the known-session column indices ordered most- to least-similar.
    ranked_known_order = np.argsort(-similarity, axis=1)

    ranked_correct, ranked_identities = [], []
    for unknown_idx, known_order in enumerate(ranked_known_order):
        ranked_known_ids = known_ids.iloc[known_order]
        # Per ranked known session: does it belong to this unknown's true identity?
        ranked_correct.append((ranked_known_ids == unknown_ids.iloc[unknown_idx]).to_list())
        # Collapse the per-session ranking into a ranking of distinct identities. Dict keys
        # preserve first-seen order, so each identity keeps the rank of its best known session.
        ranked_identities.append(list({identity: True for identity in ranked_known_ids}.keys()))

    # Per-identity bookkeeping used by analyze(). With one held-out session per identity the
    # unknown counts are trivial (one row, count 1), but both stay general -- the unknown counts
    # and the known counts feed the random baselines (random_id and random_conv respectively).
    unknown_rows_by_identity = {
        uid: unknown_sessions[unknown_sessions["identity"] == uid].index.tolist()
        for uid in unknown_ids
    }
    unknown_count_by_identity = unknown_sessions.groupby("identity").size().to_dict()
    known_count_by_identity = known_sessions.groupby("identity").size().to_dict()
    return dict(
        unknown=unknown_sessions,
        unknown_ids=unknown_ids,
        unknown_rows_by_identity=unknown_rows_by_identity,
        unknown_count_by_identity=unknown_count_by_identity,
        known_count_by_identity=known_count_by_identity,
        ranked_correct=ranked_correct,
        ranked_identities=ranked_identities,
    )


def analyze(ranking, sample_size=None, top_ks=(1, 5, 10)):
    """Top-k conversation- and identity-level accuracy vs two random-guessing baselines.
    Each identity contributes one unknown session, so conv_acc == id_acc at top-1.

    Baselines: `random_id` guesses identities uniformly; `random_conv` guesses in proportion to
    each user's number of known conversations (a random adversary ranking known conversations).

    `sample_size` optionally restricts the candidate pool to a random subset of identities
    (used by the pool-size sweep); None means use every identity.
    """
    unknown_ids = ranking["unknown_ids"]
    if sample_size is None:
        candidate_ids = set(unknown_ids)            # full candidate pool
        sample_size = len(candidate_ids)
    else:
        # Subsample a candidate pool of the requested size (for the pool-size sweep). Sort the
        # unique ids first so the draw is reproducible across runs: iterating a set of strings has
        # process-dependent (hash-randomized) order, which would otherwise make random.sample pick
        # a different subset each run despite the fixed seed.
        candidate_ids = set(random.sample(sorted(set(unknown_ids.to_list())), sample_size))

    # Row positions (in `unknown`) of every unknown session belonging to a candidate identity.
    unknown_rows = [row for cand_id in candidate_ids for row in ranking["unknown_rows_by_identity"][cand_id]]
    ranked_correct = [ranking["ranked_correct"][row] for row in unknown_rows]
    # Re-rank within the sampled pool: drop candidate identities that aren't in this subset so
    # the ranking reflects a pool of exactly `sample_size` candidates.
    ranked_ids = [[cid for cid in ranking["ranked_identities"][row] if cid in candidate_ids] for row in unknown_rows]
    unknown_subset = ranking["unknown"].iloc[unknown_rows]
    unknown_count_by_identity = {uid: c for uid, c in ranking["unknown_count_by_identity"].items() if uid in candidate_ids}
    known_count_by_identity = {uid: c for uid, c in ranking["known_count_by_identity"].items() if uid in candidate_ids}
    total_known = sum(known_count_by_identity.values())   # total known conversations in the candidate pool
    n_candidates = len(candidate_ids)

    results = []
    for top in top_ks:
        # Conversation level: is any of the top-k ranked known sessions a true match?
        conv_acc = float(np.mean([np.any(matches[:top]) for matches in ranked_correct]))
        # Identity level: an identity is re-identified if any of its unknown conversations
        # places the true identity within the top-k ranked candidate identities.
        reidentified_ids = {
            true_id
            for i, ranked in enumerate(ranked_ids)
            if (true_id := unknown_subset.iloc[i]["identity"]) in ranked[:top]
        }
        id_acc = len(reidentified_ids) / n_candidates
        # Random baseline: a uniform-random top-k guess out of n candidates contains the true
        # identity with probability min(top, n)/n. An identity with c unknown conversations is
        # re-identified if at least one of them succeeds: 1 - (1 - min(top, n)/n)^c. Averaging
        # that success probability over all n identities gives the expected random accuracy.
        random_id = sum(
            1 - (1 - min(top, n_candidates) / n_candidates) ** c
            for c in unknown_count_by_identity.values()
        ) / n_candidates
        # Conversation-count-weighted baseline: a random adversary that ranks known conversations
        # uniformly, so users with more known conversations are likelier to be guessed (unlike
        # random_id, which weights every user equally). For a user with t known conversations out
        # of total_known, one unknown lands a correct known conversation in the top-k with
        # topk_conv_hit_prob(t, total_known, top); an identity with c unknowns is hit if any
        # succeed. (At top-1 this collapses to (1/n)*sum(t/total_known) = 1/n, i.e. it equals
        # random_id; the conversation weighting only changes the baseline for k > 1.)
        random_conv = sum(
            1 - (1 - topk_conv_hit_prob(known_count_by_identity[uid], total_known, top)) ** c
            for uid, c in unknown_count_by_identity.items()
        ) / n_candidates
        results.append({"sample": sample_size, "top": top, "conv_acc": conv_acc,
                        "id_acc": id_acc, "random_id": random_id, "random_conv": random_conv,
                        "advantage": id_acc - random_id, "advantage_conv": id_acc - random_conv})
    return results


# ---------- 1. load ----------
sessions = pd.read_csv(CACHE)
sessions = sessions[sessions["model_owner"] == "Anthropic"]   # restrict the global pool to one agent provider
sessions["timestamp"] = pd.to_datetime(sessions["timestamp"], format="ISO8601", utc=True)
sessions["content"] = sessions["content"].fillna("").astype(str)
feature_df = pd.read_csv(FEATS).set_index("session_id")       # StyloMetrix features, indexed by session_id
feature_df = feature_df[feature_df.index.isin(sessions["session_id"])]
feature_cols = feature_df.columns.tolist()


# ---------- 2. identity, with repo-based recovery for id-less sessions ----------
has_user_id = sessions["user_id"].notna() & (sessions["user_id"].astype(str).str.lower() != "nan")
labeled_sessions = sessions[has_user_id]
users_per_repo = labeled_sessions.groupby("repo_id")["user_id"].nunique()   # distinct known users per repo
sole_user_per_repo = labeled_sessions.groupby("repo_id")["user_id"].first()  # only meaningful for single-user repos
# Distinct known users on each session's repo (0 if the repo never appears with a user_id).
repo_user_count = sessions["repo_id"].map(users_per_repo).fillna(0).astype(int)

is_idless = ~has_user_id
identity = sessions["user_id"].astype("object").where(has_user_id)                              # user_id, else NaN
identity = identity.mask(is_idless & (repo_user_count == 1), sessions["repo_id"].map(sole_user_per_repo))  # merge into the repo's sole user
identity = identity.mask(is_idless & (repo_user_count == 0), sessions["repo_id"])               # orphan repo -> repo_id
sessions["identity"] = identity                                                                 # (>1-user repos stay NaN -> dropped below)

n_total = len(sessions)
n_dropped = int(sessions["identity"].isna().sum())
sessions = sessions[sessions["identity"].notna()].copy()
sessions["identity"] = sessions["identity"].astype(str)
print(f"{n_total} sessions | {len(feature_cols)} StyloMetrix features")
print(f"  identity: {int(has_user_id.sum())} labeled (user_id) | "
      f"{int((is_idless & (repo_user_count == 1)).sum())} id-less recovered via single-user repo | "
      f"{int((is_idless & (repo_user_count == 0)).sum())} id-less on orphan repo (-> repo_id)")
print(f"  dropped {n_dropped} id-less sessions on repos shared by >1 user (no valid identity)")

# ---------- 3. per-identity dedup: exact content + prefix/suffix, keep first (chronological) ----------
data = sessions.sort_values(["identity", "timestamp", "session_id"]).reset_index(drop=True)
n_before_dedup = len(data)
data = data.drop_duplicates(["identity", "content"], keep="first")                   # exact content
# Key each session by its leading/trailing MIN_LEN chars to catch near-duplicate templates.
affix_keys = data.assign(_prefix=data["content"].str[:MIN_LEN],
                         _suffix=data["content"].str[-MIN_LEN:],
                         _length=data["content"].str.len())
long_enough = affix_keys["_length"] >= MIN_LEN                                       # skip very short prompts
# Within an identity, a later session sharing a prefix OR suffix with an earlier one is a dup.
is_affix_dup = (long_enough & affix_keys.duplicated(["identity", "_prefix"], keep="first")) | \
               (long_enough & affix_keys.duplicated(["identity", "_suffix"], keep="first"))
data = data[~is_affix_dup].reset_index(drop=True)
print(f"  dedup (content + {MIN_LEN}-char prefix/suffix, keep first): {n_before_dedup} -> {len(data)} sessions\n")

# ---------- 4. global adversary (one pool) ----------
# Keep only identities with >=2 sessions: one is held out as unknown, >=1 remains known.
global_sessions = data[data.groupby("identity")["session_id"].transform("size") >= 2].copy()
global_ranking = build_ranking(global_sessions)
n_global_users = global_sessions["identity"].nunique()
print(f"{n_global_users} users, {len(global_sessions)} sessions")
global_rows = analyze(global_ranking)
for row in global_rows:
    print(f"  top {row['top']:>2}: conv_acc={row['conv_acc']:.3f}  id_acc={row['id_acc']:.3f}  "
          f"random_id={row['random_id']:.3f}  random_conv={row['random_conv']:.3f}  "
          f"advantage={row['advantage']:.3f}  advantage_conv={row['advantage_conv']:.3f}")
pd.DataFrame(global_rows).to_csv("swe_chat_global_results.csv", index=False)

# ---------- 5. global sweep: re-identification vs candidate-pool size (top-1) ----------
random.seed(47)
n_simulations = 100   # random subsamples per pool size, to get a distribution
sweep_rows = []
# Pool sizes 25, 50, 75, ... up to (but excluding) the total, plus the full-size point.
for pool_size in list(range(25, n_global_users, 25)) + [n_global_users]:
    for _ in tqdm(range(n_simulations), desc=f"sweep n={pool_size}", leave=False):
        sweep_rows.extend({**result, "n": pool_size}
                          for result in analyze(global_ranking, sample_size=pool_size, top_ks=(1,)))
sweep = pd.DataFrame(sweep_rows)
sweep.to_csv("swe_chat_global_sweep.csv", index=False)

# Long format for seaborn: one row per (pool size, metric) so id_acc and the random baseline
# can be drawn as paired boxplots. Only random_id is plotted (the sweep is top-1, where
# random_conv equals it); random_conv/advantage_conv remain in the CSV for k>1 comparisons.
melted_sweep = sweep.melt(id_vars="n", value_vars=["id_acc", "random_id"], var_name="metric", value_name="value")
melted_sweep["metric"] = melted_sweep["metric"].map({"id_acc": "StyloMetrix", "random_id": "Random guessing"})
plt.figure(figsize=(7, 4))
sns.boxplot(data=melted_sweep, x="n", y="value", hue="metric", order=sorted(sweep["n"].unique()),
            showfliers=False, width=0.5, palette=sns.color_palette(n_colors=2))
plt.xlabel("Number of candidate users")
plt.ylabel("Top-1 identification accuracy")
plt.legend(title="")
plt.grid(linestyle="--", color="lightgray")
plt.tight_layout()
plt.savefig("swe_chat_global_top1.pdf")
plt.close()

# ---------- 6. ablation: vary the number of (earliest) known sessions per user ----------
# Each user's single last session is the unknown; its first n_known earliest sessions are the
# known/labeled history (sessions in between are dropped), so known always precedes the unknown
# and a user with <= n_known sessions keeps all but its last as known. Every attackable user is
# kept, so the candidate pool is fixed (the same users as section 4) for all n_known, and the
# setup converges to the global adversary once n_known >= max_known. One unknown per user isolates
# the effect of known-history size (no confound from how many unknowns a user has).
sessions_per_user = global_sessions.groupby("identity")["session_id"].size()
max_known = int(sessions_per_user.max()) - 1   # known sessions held by the most active user
n_known_grid = sorted({v for v in [1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 75, 100, 150, max_known]
                       if 1 <= v <= max_known})

ablation_rows = []
for n_known in n_known_grid:
    ablation_ranking = build_ranking(global_sessions, n_known=n_known)
    for r in analyze(ablation_ranking):
        ablation_rows.append({"n_known": n_known, "users": n_global_users, **r})
ablation = pd.DataFrame(ablation_rows)
ablation.to_csv("swe_chat_known_ablation.csv", index=False)

print("\nKnown-session ablation (id_acc vs number of known sessions per user):")
for k in (1, 5, 10):
    print(f"--- top-{k} ---")
    print(ablation[ablation["top"] == k][
        ["n_known", "users", "id_acc", "random_id", "advantage"]
    ].to_string(index=False))

# Plot id_acc (solid) and its random baseline (dashed) vs n_known, one color per top-k.
plt.figure(figsize=(7, 4.5))
for color, k in zip(sns.color_palette(n_colors=3), (1, 5, 10)):
    sub = ablation[ablation["top"] == k].sort_values("n_known")
    plt.plot(sub["n_known"], sub["id_acc"], marker="o", color=color, label=f"StyloMetrix top-{k}")
    plt.plot(sub["n_known"], sub["random_id"], linestyle="--", color=color, alpha=0.7, label=f"Random top-{k}")
plt.xlabel("Number of known sessions per user")
plt.ylabel("Top-k identification accuracy")
plt.grid(linestyle="--", color="lightgray")
# Legend below the axes, 3 columns x 2 rows (matplotlib fills column-major, so each column is one
# top-k: StyloMetrix on the first row, its Random baseline on the second).
plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8)
plt.tight_layout()
plt.savefig("swe_chat_known_ablation.pdf", bbox_inches="tight")
plt.close()

# ---------- 7. per agent-provider adversary (partition by model_owner) ----------
# rows = []
# for (owner,), part in data.groupby(["model_owner"]):
#     part = part[part.groupby("identity")["session_id"].transform("size") >= 2]
#     n = part["identity"].nunique()
#     if n < 2:
#         continue
#     R = build_ranking(part)
#     for r in analyze(R):
#         rows.append({"model_owner": owner, "users": n, **r})

# per_owner = pd.DataFrame(rows).sort_values(["users", "model_owner"], ascending=[False, True])
# per_owner.to_csv("swe_chat_per_owner_results.csv", index=False)
# print("\nPer-model_owner attack (partitions with >=2 attackable users):")
# for k in (1, 5, 10):
#     print(f"--- top-{k} ---")
#     print(per_owner[per_owner["top"] == k][
#         ["model_owner", "users", "conv_acc", "id_acc", "random_id", "advantage"]
#     ].to_string(index=False))

# top1 = per_owner[per_owner["top"] == 1]
# labels = top1["model_owner"] + "\n(n=" + top1["users"].astype(str) + ")"
# x = np.arange(len(top1))
# w = 0.38
# plt.figure(figsize=(8, 4))
# plt.bar(x - w / 2, top1["id_acc"], w, label="StyloMetrix")
# plt.bar(x + w / 2, top1["random_id"], w, label="Random guessing")
# plt.xticks(x, labels, fontsize=8)
# plt.ylabel("Top-1 identification accuracy")
# plt.legend()
# plt.grid(axis="y", linestyle="--", color="lightgray")
# plt.tight_layout()
# plt.savefig("swe_chat_per_owner_top1.pdf")
# plt.close()

# print("\nSaved: swe_chat_global_results.csv, swe_chat_global_sweep.csv, "
#       "swe_chat_per_owner_results.csv, swe_chat_global_top1.pdf, swe_chat_per_owner_top1.pdf")
