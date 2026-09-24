#!/bin/bash
#
# run_rerank_qwen_swe_chat.sbatch -- rerank the top-5 nearest-neighbor author shortlist on SWE-chat
# (gemini_embedding_2 vectors) with Qwen3.8-27B running LOCALLY, thinking ON, candidates SCRAMBLED,
# one short reason per rank -- then prove all of that from the output.
#
#   sbatch experiments/run_rerank_qwen_swe_chat.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
# One-time: mkdir -p scripts/logs
#
# THE LOCAL TWIN of run_rerank_sonnet_swe_chat.sbatch: the same split, shortlist, presentation and
# ranking rubric, judged by an open model on this node's GPU instead of an API. Results go to .../qwen
# beside .../jina and .../llm. Standalone like the others: it calls run_rerank.py directly, and every
# stage prints a numbered, timestamped header and fails loudly.
#
# CALLED THE WAY AFR CALLS IT: one offline vLLM engine over the same checkpoint
# (models.toml [rerank_qwen], = [afr]: /datasets/ai/qwen3/hub/models--Qwen--Qwen3.8-27B, bf16 ~54 GB),
# then batched engine.chat(...) calls. Needs the environment AFR runs in (vllm from the [qwen] extra)
# and an 80 GB card. FREE -- no API, no key.
#
# ONLY THE TOP 5 MOVE. Qwen is shown the five shortlisted candidates and nothing else, and its
# ordering is written back onto exactly those five; run_rerank.py checks row by row that nothing
# outside the shortlist moved. top-1/2/3 are the judge's contribution; top-5 is pinned.
#
# THINKING IS ON (chat_template_kwargs enable_thinking=True), with Qwen's thinking-mode sampling
# (temperature 0.6, top_p 0.95, top_k 20, fixed seed). Only the text after </think> is parsed.
# Step 5 counts replies that actually closed a thinking block and fails if none did.
#
# A REASON PER RANK. The rubric asks for a short "why" at every position; step 5 fails if more than
# 5% of rows are missing one, and the checker prints three whole rankings with their reasons.
#
# THE BUDGET IS 16,384 TOKENS per reply, thinking included. A reply cut off mid-thought has no
# answer and falls back to the distance order, so step 5 fails above 5% truncated.
#
# PREEMPTION IS SURVIVABLE: prompts are judged in chunks (RERANK_QWEN_CHUNK, 128) and every chunk is
# cached as it finishes, so a requeue redoes at most one chunk.
#
# Preflight on a login node (no GPU, no model load):
#
#     CHECK_ONLY=1 bash experiments/run_rerank_qwen_swe_chat.sbatch
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#   LIMIT=20 sbatch ...                        # a seeded sample of the unknown side
#   TOP_K=10 sbatch ...                        # a different shortlist size
#   RERANK_QWEN_MODEL=/path/to/checkpoint ...  # a different local checkpoint
#   RERANK_QWEN_GPU_MEM_UTIL=0.85 ...          # vLLM's share of the card (default 0.90)
#   RERANK_QWEN_MAX_NUM_SEQS=64 ...            # concurrent sequences (default 128; hybrid-model cap)
#   RERANK_QWEN_CHUNK=64 ...                   # prompts per chat() call / per cache flush
#   FEATURE=luar / WINDOW=5075 / SAMPLE=8 / OUT_DIR=...
#
#SBATCH --job-name=rrk-qwen-swe
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=a100-80g|h100|h200
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0-08:00:00
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
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
CACHE_DIR="${CACHE_DIR:-$PROJECT/data/.cache}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# `none` is spelled `base` in results/ (run_experiment.NO_DEFENSE_TAG); the split stem is the
# source alone for `none`, else <source>_<defense>.
if [ "$DEFENSE" = "none" ]; then TAG=base; STEM="$SOURCE"; else TAG="$DEFENSE"; STEM="${SOURCE}_$DEFENSE"; fi
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/${SOURCE}_${TAG}_${FEATURE}_known${WINDOW}/qwen}"

# The Python checks below read their settings from the environment.
export TOP_K SEED OUT_DIR CHECK_ONLY

# Bare `python` when checking on a login node; the env bootstrap only when running as a job.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

echo "=================================================================="
echo "listwise rerank (Qwen3.8-27B local, thinking on) -- $SOURCE / $FEATURE / known$WINDOW / top-$TOP_K"
if [ -n "${LIMIT:-}" ]; then
    echo "  unknown side: a seeded sample of $LIMIT documents"
else
    echo "  unknown side: all of it"
fi
echo "  output:       $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. environment --------------------------------------------------------
# The judge is a 27B model on vLLM: no GPU means no run, and vLLM missing means the wrong
# environment. Both are checked before anything is loaded. vLLM is only required for a real run.
echo ""
echo "[1/5] environment ($(date))"
run python -c '
import os, sys
print("  interpreter:", sys.executable)
import torch
cuda = torch.cuda.is_available()
print("  torch", torch.__version__, "| cuda:", cuda, "|",
      torch.cuda.get_device_name(0) if cuda else "no device")
try:
    import vllm
    print("  vllm", vllm.__version__)
except ImportError as error:
    if os.environ.get("CHECK_ONLY") != "1":
        sys.exit(f"ERROR: {error}. The local judge needs vLLM -- run this in the environment AFR "
                 f"uses (install.sh installs the [qwen] extra on a CUDA machine).")
    print("  vllm: not importable here (fine for CHECK_ONLY; required for the run)")
if os.environ.get("CHECK_ONLY") != "1" and not cuda:
    sys.exit("ERROR: no GPU visible to torch inside the job.")
'

# --- 2. inputs, self-test, checkpoint ---------------------------------------
echo ""
echo "[2/5] inputs ($(date))"
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
# Resolved and checked on disk now, so a wrong path fails in seconds rather than after the queue,
# and never turns into a 54 GB download.
run python -c '
import sys
from prompt_anonymity.attacks.llm._local_vllm import judge_checkpoint
try:
    print("  judge checkpoint:", judge_checkpoint())
except Exception as error:
    sys.exit(f"ERROR: {error}")
'

# --- 3. configuration: provider, thinking, sampling, budget, scramble --------
# Builds the attack through run_rerank.build_attack -- the exact call the real run makes -- and
# asserts on it. No model is loaded here.
echo ""
echo "[3/5] configuration check ($(date))"
PRE_CHECK=$(cat <<'PY'
import math, os, sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, "experiments")
from run_rerank import build_attack
from prompt_anonymity.attacks.llm import _local_vllm
from prompt_anonymity.attacks.llm.candidates import AuthorCandidates
from prompt_anonymity.attacks.llm.listwise import present
from prompt_anonymity.attacks.llm.listwise_llm_rerank import RANK_SYSTEM_PROMPT_TEMPLATE

k, seed = int(os.environ["TOP_K"]), int(os.environ["SEED"])
attack = build_attack("qwen", k, Namespace(margin_quantile=1.0, seed=seed, batch=True, wait=True,
                                           judge_model=None, reasoning_effort=None))
problems = []
if attack.provider != "local" or attack.batch:
    problems.append(f"built provider={attack.provider!r}, batch={attack.batch} -- expected local, unbatched")
if not attack.reasoning_effort:
    problems.append("thinking would be off")
if attack.shuffle_candidates is not True:
    problems.append(f"shuffle_candidates is {attack.shuffle_candidates!r}")
budget = attack._max_tokens(k)
if budget < 16000:
    problems.append(f"budget {budget} tokens leaves too little room for thinking")
if budget >= _local_vllm.RERANK_QWEN_MAX_MODEL_LEN - 2048:
    problems.append(f"budget {budget} leaves no room for the prompt in a "
                    f"{_local_vllm.RERANK_QWEN_MAX_MODEL_LEN}-token window")
if '"why"' not in RANK_SYSTEM_PROMPT_TEMPLATE:
    problems.append("the ranking rubric no longer asks for a reason per position")

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

print(f"  judge:        {attack.judge_model} (local vLLM, unbatched)")
print(f"  thinking:     on; sampling {_local_vllm.THINKING_SAMPLING}, seed {seed}")
print(f"  budget:       {budget} tokens per reply (thinking included), "
      f"window {_local_vllm.RERANK_QWEN_MAX_MODEL_LEN}")
print(f"  serving:      gpu_memory_utilization {_local_vllm.RERANK_QWEN_GPU_MEM_UTIL}, "
      f"max_num_seqs {_local_vllm.RERANK_QWEN_MAX_NUM_SEQS}, chunk {_local_vllm.RERANK_QWEN_CHUNK}")
print(f"  scramble:     shuffle_candidates={attack.shuffle_candidates}; "
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

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, no model loaded, nothing reranked)"
    exit 0
fi

# --- 4. the attack ---------------------------------------------------------
ARGS=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
      --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
      --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR"
      --variant qwen --top-k "$TOP_K" --no-batch)
if [ -n "${LIMIT:-}" ]; then
    ARGS+=(--limit "$LIMIT")
fi

echo ""
echo "[4/5] reranking ($(date))"
run python -u experiments/run_rerank.py "${ARGS[@]}"

# --- 5. audit: what Qwen was shown, whether it thought, and whether every rank has a reason ------
echo ""
echo "[5/5] auditing the output ($(date))"
POST_CHECK=$(cat <<'PY'
import os, sys

import pandas as pd

out, k = os.environ["OUT_DIR"], int(os.environ["TOP_K"])
detail = pd.read_csv(f"{out}/rerank_detail_qwen_k{k}.csv")
summary = pd.read_csv(f"{out}/rerank_summary.csv")
row = summary[(summary["variant"] == "qwen") & (summary["top_k"] == k)]
problems = []


def stat(name):
    """One summary number for this configuration, or nan if the column/row is absent."""
    return float(row[name].iloc[0]) if name in row and len(row) else float("nan")


# Scramble, as shown: per row, the distance ranks in presented-slot order ARE the order Qwen read.
shown = (detail.sort_values(["row", "presented_slot"]).groupby("row")["distance_rank"]
         .apply(lambda ranks: tuple(int(r) for r in ranks)))
nearest_first = int(shown.map(lambda seq: seq == tuple(range(k))).sum())
share = nearest_first / max(len(shown), 1)
print(f"  scramble:  {nearest_first}/{len(shown)} rows ({share:.1%}) shown nearest-first")
picks = detail.loc[detail["rerank_position"] == 1, "presented_slot"].value_counts(normalize=True)
print("  top pick by presented slot (want ~uniform; a spike = position bias): "
      + ", ".join(f"{int(s)}:{p:.0%}" for s, p in picks.sort_index().items()))
if share > 0.05 and nearest_first >= 3:
    problems.append(f"{share:.1%} of rows were shown nearest-first -- the candidates were not scrambled")

# Thinking, from the completions themselves. Cached rows carry no evidence either way.
replies = stat("judge_replies")
if pd.isna(replies) or replies == 0:
    print("  thinking:  every row was served from the cache -- no new evidence this run")
else:
    reasoned, tokens = stat("judge_replies_with_reasoning"), stat("judge_reasoning_tokens")
    truncated = stat("judge_truncated")
    print(f"  thinking:  {reasoned:.0f}/{replies:.0f} fresh replies closed a thinking block, "
          f"~{tokens / replies:.0f} thinking tokens per reply")
    print(f"  budget:    {truncated:.0f}/{replies:.0f} replies hit the token limit")
    if reasoned == 0:
        problems.append("thinking was requested but NO reply contained a thinking block")
    elif reasoned < 0.9 * replies:
        print(f"  WARNING: only {reasoned / replies:.0%} of replies thought")
    if truncated > 0.05 * replies:
        problems.append(f"{truncated:.0f}/{replies:.0f} replies were cut off -- raise LOCAL_MAX_TOKENS "
                        f"(and RERANK_QWEN_MAX_MODEL_LEN with it)")

# A reason per rank: every one of a row's K positions must carry a non-empty "why".
filled = detail["reason"].fillna("").astype(str).str.strip().ne("")
complete_rows = filled.groupby(detail["row"]).all()
missing = int((~complete_rows).sum())
print(f"  reasons:   {int(complete_rows.sum())}/{len(complete_rows)} rows have a reason at all "
      f"{k} ranks; median reason {detail.loc[filled, 'reason'].astype(str).str.split().str.len().median():.0f} words")
if missing > 0.05 * len(complete_rows):
    problems.append(f"{missing}/{len(complete_rows)} rows are missing a reason at one or more ranks")

ties = stat("n_boundary_ties")
if pd.notna(ties) and ties:
    print(f"  ceiling:   {ties:.0f} row(s) tie at the shortlist boundary; top-{k} is pinned to "
          f"shortlist_ceiling rather than base_top_{k} (expected, not a leak)")
if problems:
    for problem in problems:
        print("ERROR:", problem, file=sys.stderr)
    sys.exit(1)
print("  ok")
PY
)
run python -c "$POST_CHECK"

# Structural checks, then SAMPLE whole rankings printed with each rank's reason.
run python experiments/check_rerank_output.py "$OUT_DIR" --sample "$SAMPLE" --seed "$SEED"

echo ""
echo "Done $(date). Wrote, under $OUT_DIR:"
ls -l "$OUT_DIR"
echo ""
echo "rerank_summary.csv: top_1..3 against base_top_1..3 is Qwen's gain; top_$TOP_K equals"
echo "shortlist_ceiling. rerank_detail_qwen_k$TOP_K.csv has every rank's reason in the 'reason' column."
