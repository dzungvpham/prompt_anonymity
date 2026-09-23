#!/bin/bash
#
# run_rerank_sonnet_swe_chat.sbatch -- rerank the top-5 nearest-neighbor author shortlist on SWE-chat
# (gemini_embedding_2 vectors) with Claude Sonnet 5 over OpenRouter, reasoning ON, candidates
# SCRAMBLED, then prove both of those from the output.
#
#   sbatch experiments/run_rerank_sonnet_swe_chat.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# THE PAID TWIN of run_rerank_jina_swe_chat.sbatch: same split, same shortlist, same presentation,
# same results tree (this writes .../llm, that one .../jina), so the gap between the two prices the
# frontier model rather than the idea of reranking. Standalone like that one: it calls run_rerank.py
# directly, and every stage prints a numbered, timestamped header and fails loudly.
#
# KEY: SONNET_API_KEY, in a .env AT THE REPO ROOT (load_dotenv walks UP from the working directory,
# so DS_env/.env is never found). This job always runs UNBATCHED (--no-batch), which is the channel
# that reads SONNET_API_KEY; the Batch API path reads SONNET_OR_KEY instead and is not used here.
#
# COST: full real-time price, roughly $0.012 per unknown document at top-5. Start with LIMIT=20 (~25
# cents), read the reasons and the audit, then run the lot. Verdicts are cached by prompt text the
# moment they arrive, so the full run re-pays nothing and a preempted/requeued job resumes.
#
# REASONING IS ON AND CANNOT BE SWITCHED OFF HERE. EFFORT defaults to high and must be one of
# low|medium|high|xhigh|max. It is checked three ways: the attack is built exactly as the run builds
# it and its effort asserted (step 3); the effort is part of the verdict cache key, so no verdict
# from a non-reasoning run can be served; and after the run, the replies' own usage/reasoning fields
# are counted (step 6) -- a request that asked for thinking but came back without any fails the job.
#
# SCRAMBLING IS ON AND CHECKED TWICE. Each row's K candidates are shown in a seeded random order
# (list rerankers favour whichever end they read last, which fed nearest-first rewards the metric's
# WORST candidate). Step 3 asserts the attack is configured to shuffle and that the permutation is
# really random; step 6 reads back, row by row, the distance order Sonnet was actually shown.
#
# Top-5 at k=5 equals the baseline by construction (run_rerank.py asserts it); top-1/2/3 are the
# reranker's contribution.
#
# Preflight on a login node (no money spent, nothing reranked):
#
#     CHECK_ONLY=1 bash experiments/run_rerank_sonnet_swe_chat.sbatch
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#   LIMIT=20 sbatch ...                    # a seeded sample of the unknown side; bounds the bill
#   EFFORT=xhigh sbatch ...                # thinking depth: low/medium/high/xhigh/max (never off)
#   WORKERS=24 sbatch ...                  # concurrent requests (default 16; 429s back off)
#   TOP_K=10 sbatch ...                    # a different shortlist size (~2x the cost)
#   MODEL=anthropic/claude-sonnet-5 ...    # OpenRouter slug for the judge
#   FEATURE=luar / WINDOW=5075 / SAMPLE=8 / OUT_DIR=...
#
#SBATCH --job-name=rrk-sonnet-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=0-12:00:00
#SBATCH --requeue

set -euo pipefail

# Resolve the repo root, then CHECK it -- $SLURM_SUBMIT_DIR is inherited garbage in an Open
# OnDemand shell, so it is a candidate rather than an answer.
looks_like_repo() { [ -d "$1/src/prompt_anonymity" ] && [ -d "$1/experiments" ]; }

PROJECT=""
for candidate in "${PROMPT_ANONYMITY_PROJECT:-}" "${SLURM_SUBMIT_DIR:-}" "$PWD"; do
    if [ -n "$candidate" ] && looks_like_repo "$candidate"; then
        PROJECT="$candidate"
        break
    fi
done
if [ -z "$PROJECT" ]; then
    echo "ERROR: cannot find the repo root. Tried \$PROMPT_ANONYMITY_PROJECT" \
         "(${PROMPT_ANONYMITY_PROJECT:-unset}), \$SLURM_SUBMIT_DIR (${SLURM_SUBMIT_DIR:-unset})" \
         "and \$PWD ($PWD)." >&2
    echo "       Run this from the repo root, or set PROMPT_ANONYMITY_PROJECT=/path/to/repo." >&2
    exit 1
fi
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-none}"
WINDOW="${WINDOW:-0075}"
TOP_K="${TOP_K:-5}"
SEED="${SEED:-47}"
SAMPLE="${SAMPLE:-3}"
EFFORT="${EFFORT:-high}"
MODEL="${MODEL:-}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
CACHE_DIR="${CACHE_DIR:-$PROJECT/data/.cache}"
CHECK_ONLY="${CHECK_ONLY:-0}"
export LISTWISE_RERANK_MAX_WORKERS="${WORKERS:-16}"

# Reasoning is the point of this arm; there is no spelling of EFFORT that turns it off.
case "$EFFORT" in
    low|medium|high|xhigh|max) ;;
    *) echo "ERROR: EFFORT='$EFFORT' -- must be one of low|medium|high|xhigh|max. This job does" \
            "not run Sonnet without reasoning." >&2
       exit 1 ;;
esac

# `none` is spelled `base` in results/ (run_experiment.NO_DEFENSE_TAG); the split stem is the
# source alone for `none`, else <source>_<defense>.
if [ "$DEFENSE" = "none" ]; then TAG=base; STEM="$SOURCE"; else TAG="$DEFENSE"; STEM="${SOURCE}_$DEFENSE"; fi

# The `llm` subdirectory is the one submit_rerank_swe_chat.sh gives this arm, so
# finish_rerank.sbatch can merge it with the jina arm without racing on rerank_summary.csv.
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/${SOURCE}_${TAG}_${FEATURE}_known${WINDOW}/llm}"

# The Python checks below read their settings from the environment.
export TOP_K SEED EFFORT MODEL OUT_DIR

# Bare `python` when checking on a login node; the env bootstrap only when running as a job.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

echo "=================================================================="
echo "listwise rerank (Sonnet 5, unbatched) -- $SOURCE / $FEATURE / known$WINDOW / top-$TOP_K"
echo "  reasoning effort: $EFFORT"
echo "  concurrency:      $LISTWISE_RERANK_MAX_WORKERS requests in flight"
if [ -n "${LIMIT:-}" ]; then
    echo "  unknown side:     a seeded sample of $LIMIT documents"
else
    echo "  unknown side:     all of it (full real-time price)"
fi
echo "  output:           $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. environment --------------------------------------------------------
echo ""
echo "[1/6] environment ($(date))"
run python -c '
import sys
print("  interpreter:", sys.executable)
import requests, dotenv  # the OpenRouter client needs both
print("  requests", requests.__version__)
'

# --- 2. inputs and self-test -----------------------------------------------
echo ""
echo "[2/6] inputs ($(date))"
if [ ! -f "$DIST_DIR/${STEM}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}.parquet -- the $SOURCE split is missing." >&2
    exit 1
fi
if [ ! -f "$DIST_DIR/${STEM}_${FEATURE}.parquet" ]; then
    echo "ERROR: no $DIST_DIR/${STEM}_${FEATURE}.parquet" >&2
    echo "  Compute the vectors first with:" >&2
    echo "    python -m prompt_anonymity.data.compute_features --source $SOURCE \\" >&2
    echo "        --feature $FEATURE --dist-dir $DIST_DIR" >&2
    exit 1
fi
echo "  split:    $DIST_DIR/${STEM}.parquet"
echo "  features: $DIST_DIR/${STEM}_${FEATURE}.parquet"
echo "  attack self-test:"
run python -m prompt_anonymity.attacks.llm.listwise --selftest

# --- 3. configuration: reasoning, scramble, key -----------------------------
# Builds the attack through run_rerank.build_attack -- the exact call the real run makes -- and
# asserts on the object, so a default flipped anywhere upstream fails here, before any money.
echo ""
echo "[3/6] configuration check: reasoning, scramble, key ($(date))"
PRE_CHECK=$(cat <<'PY'
import math, os, sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, "experiments")
from run_rerank import build_attack
from prompt_anonymity.attacks.llm.candidates import AuthorCandidates
from prompt_anonymity.attacks.llm.listwise import present
from prompt_anonymity.attacks.llm.listwise_llm_rerank import (
    MAX_TOKENS_BASE, MAX_TOKENS_PER_CANDIDATE, SYNC_API_KEY_ENV)

k, seed, effort = int(os.environ["TOP_K"]), int(os.environ["SEED"]), os.environ["EFFORT"]
attack = build_attack("llm", k, Namespace(
    margin_quantile=1.0, seed=seed, batch=False, wait=True,
    judge_model=os.environ.get("MODEL") or None, reasoning_effort=effort))

problems = []
if attack.batch:
    problems.append("attack.batch is True -- the batch channel reads SONNET_OR_KEY, not SONNET_API_KEY")
if not attack.reasoning_effort or attack.reasoning_effort != effort:
    problems.append(f"reasoning_effort is {attack.reasoning_effort!r}, expected {effort!r}")
if attack.shuffle_candidates is not True:
    problems.append(f"shuffle_candidates is {attack.shuffle_candidates!r} -- candidates would be "
                    f"shown nearest-first")
if SYNC_API_KEY_ENV != "SONNET_API_KEY":
    problems.append(f"the unbatched channel reads {SYNC_API_KEY_ENV}, not SONNET_API_KEY")

# Scramble, on a synthetic shortlist: exactly the call the attack makes, with its own settings.
n = 1000
grid = np.tile(np.arange(k), (n, 1))
layout = present(AuthorCandidates(authors=np.arange(k), scores=np.zeros((n, k)), author_index=grid,
                                  document_index=grid, margin=np.zeros(n)),
                 seed=seed, shuffle=attack.shuffle_candidates)
identity = float(np.mean([np.array_equal(r, np.arange(k)) for r in layout.ranks]))
first = np.bincount([int(r[0]) for r in layout.ranks], minlength=k) / n
if identity > 0.05:
    problems.append(f"{identity:.1%} of synthetic rows presented nearest-first "
                    f"(a real shuffle gives ~{1 / math.factorial(k):.1%})")
if (first == 0).any():
    problems.append(f"slot 1 never shows distance ranks {np.flatnonzero(first == 0).tolist()}")

print(f"  judge model:      {attack.judge_model}")
print(f"  reasoning effort: {attack.reasoning_effort}")
print(f"  channel:          unbatched, key {SYNC_API_KEY_ENV}")
print(f"  output budget:    {MAX_TOKENS_BASE + MAX_TOKENS_PER_CANDIDATE * k} tokens per reply "
      f"(thinking included)")
print(f"  scramble:         shuffle_candidates={attack.shuffle_candidates}; "
      f"{identity:.1%} of {n} synthetic rows nearest-first; slot-1 distance ranks "
      + ", ".join(f"{rank}:{share:.0%}" for rank, share in enumerate(first)))
if problems:
    for problem in problems:
        print("ERROR:", problem, file=sys.stderr)
    sys.exit(1)
print("  ok")
PY
)
run python -c "$PRE_CHECK"

# --- 4. credentials --------------------------------------------------------
echo ""
echo "[4/6] credentials ($(date))"
if [ -z "${SONNET_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: SONNET_API_KEY is not set and there is no $PROJECT/.env for load_dotenv to read." >&2
    echo "  Add it at the REPO ROOT:  echo 'SONNET_API_KEY=sk-or-...' >> $PROJECT/.env" >&2
    exit 1
fi
# Asks OpenRouter about the key without generating tokens; prints its length and last four
# characters, never the key.
run python -c '
import sys
from prompt_anonymity.attacks.llm._openrouter import check_credentials
from prompt_anonymity.attacks.llm.listwise_llm_rerank import SYNC_API_KEY_ENV

ok, message = check_credentials(SYNC_API_KEY_ENV)
print("  " + message)
sys.exit(0 if ok else 1)
'

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, nothing sent to Sonnet)"
    exit 0
fi

# --- 5. the attack ---------------------------------------------------------
ARGS=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
      --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
      --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR"
      --variant llm --top-k "$TOP_K" --no-batch --reasoning-effort "$EFFORT")
if [ -n "${LIMIT:-}" ]; then
    ARGS+=(--limit "$LIMIT")
fi
if [ -n "$MODEL" ]; then
    ARGS+=(--judge-model "$MODEL")
fi

echo ""
echo "[5/6] reranking ($(date))"
run python -u experiments/run_rerank.py "${ARGS[@]}"

# --- 6. audit: what Sonnet was actually shown, and whether it actually thought ----------------
echo ""
echo "[6/6] auditing the output ($(date))"
POST_CHECK=$(cat <<'PY'
import os, sys

import pandas as pd

out, k = os.environ["OUT_DIR"], int(os.environ["TOP_K"])
detail = pd.read_csv(f"{out}/rerank_detail_llm_k{k}.csv")
summary = pd.read_csv(f"{out}/rerank_summary.csv")
problems = []

# Scramble, as shown: per row, the distance ranks in presented-slot order ARE the order Sonnet read.
shown = (detail.sort_values(["row", "presented_slot"]).groupby("row")["distance_rank"]
         .apply(lambda ranks: tuple(int(r) for r in ranks)))
nearest_first = int(shown.map(lambda seq: seq == tuple(range(k))).sum())
share = nearest_first / max(len(shown), 1)
print(f"  scramble:  {nearest_first}/{len(shown)} rows ({share:.1%}) shown nearest-first")
slot1 = detail.loc[detail["presented_slot"] == 1, "distance_rank"].value_counts(normalize=True)
print("  slot 1 held distance rank: "
      + ", ".join(f"{int(r)}:{s:.0%}" for r, s in slot1.sort_index().items()))
picks = detail.loc[detail["rerank_position"] == 1, "presented_slot"].value_counts(normalize=True)
print("  Sonnet's top pick by presented slot (want ~uniform; a spike = position bias): "
      + ", ".join(f"{int(s)}:{p:.0%}" for s, p in picks.sort_index().items()))
if share > 0.05 and nearest_first >= 3:
    problems.append(f"{share:.1%} of rows were shown nearest-first -- the candidates were not scrambled")

# Reasoning, from the replies themselves. Cached rows carry no evidence either way.
row = summary[(summary["variant"] == "llm") & (summary["top_k"] == k)]
replies = float(row["judge_replies"].iloc[0]) if "judge_replies" in row and len(row) else float("nan")
if pd.isna(replies) or replies == 0:
    print("  reasoning: every row was served from the cache -- no new evidence this run "
          "(reasoning_effort is part of the cache key, so these came from a reasoning run)")
else:
    reasoned = float(row["judge_replies_with_reasoning"].iloc[0])
    tokens = float(row["judge_reasoning_tokens"].iloc[0])
    print(f"  reasoning: {reasoned:.0f}/{replies:.0f} fresh replies reasoned, "
          f"{tokens / replies:.0f} reasoning tokens per reply on average")
    if reasoned == 0:
        problems.append("reasoning was requested but NO reply reported any -- neither "
                        "usage.completion_tokens_details.reasoning_tokens nor message.reasoning")
    elif reasoned < 0.9 * replies:
        print(f"  WARNING: only {reasoned / replies:.0%} of replies reasoned")

# Truncation: a row with no reasons at all parsed as nothing and fell back to distance order.
blank = detail.groupby("row")["reason"].apply(
    lambda reasons: reasons.fillna("").astype(str).str.strip().eq("").all())
print(f"  unparsed:  {int(blank.sum())}/{len(blank)} rows carry no reasons (fell back to distance order)")
if blank.mean() > 0.10:
    problems.append(f"{blank.mean():.0%} of rows returned nothing usable -- if replies were cut off, "
                    f"thinking may be eating the output budget (MAX_TOKENS_BASE)")

cost = row["judge_cost_usd"].iloc[0] if len(row) else float("nan")
print(f"  cost:      ${cost:.2f} billed this run" if pd.notna(cost) else "  cost:      nothing billed")
if problems:
    for problem in problems:
        print("ERROR:", problem, file=sys.stderr)
    sys.exit(1)
print("  ok")
PY
)
run python -c "$POST_CHECK"

run python experiments/check_rerank_output.py "$OUT_DIR" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date). Wrote, under $OUT_DIR:"
ls -l "$OUT_DIR"
echo ""
echo "rerank_summary.csv: top_1..3 against base_top_1..3 is Sonnet's gain; top_$TOP_K must equal"
echo "base_top_$TOP_K. Compare with the jina arm at ${OUT_DIR%/llm}/jina. Read the reasons above."
