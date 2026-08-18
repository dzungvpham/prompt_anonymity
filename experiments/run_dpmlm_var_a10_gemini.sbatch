#!/bin/bash
#
# run_dpmlm_var_a10_gemini.sbatch -- attack an ALREADY-DEFENDED dp_mlm_var_a10 split on Gemini
# embeddings, then draw the figures: featurize -> attack (logistic) -> plot.
#
#   sbatch experiments/run_dpmlm_var_a10_gemini.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from. One-time: mkdir -p scripts/logs
#
# HOW THIS DIFFERS FROM run_dpmlm_var_a10.sbatch: that script runs all three stages, and its first
# one -- the DP-MLM pass itself -- is the multi-hour GPU stage. This script NEVER defends. It
# starts from the parquet that run already wrote and fails immediately if it is missing, because
# re-running DP-MLM to get a second feature view of the same defended text would be hours of GPU
# time to reproduce a file that is already on disk.
#
# REQUIRES an OPENROUTER_API_KEY the job can reach -- not for the defense (nothing is rewritten
# here), but for gemini_embedding_2, which is an OpenRouter embeddings call per document.
# load_dotenv walks UP from the working directory, so the key must be in a .env at the REPO ROOT:
# cp DS_env/.env .env
#
# COST is entirely stage 1: one embedding call per document at $0.20/M tokens. SWE-chat is 4,334
# documents, so this arm is cents; WildChat is 172,509 and is not. The stage caches by text
# content, so a re-run or a requeue costs nothing for what it already embedded.
#
# NO GPU. gemini_embedding_2 is a network call, the logistic attack is scikit-learn, and
# plot_results is matplotlib -- so this asks for a CPU node, which is also what gets it scheduled
# quickly. Only put it back on a GPU partition if you add stylometrix to $FEATURE (it runs a spaCy
# transformer); copy the #SBATCH block from run_dpmlm_var_a10.sbatch.
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   SOURCE=wildchat sbatch experiments/run_dpmlm_var_a10_gemini.sbatch    # default swe_chat
#   DEFENSE=dp_mlm_var_a25 sbatch experiments/run_dpmlm_var_a10_gemini.sbatch  # the A=0.25 variant
#   PLOT=0 sbatch experiments/run_dpmlm_var_a10_gemini.sbatch             # attack only, no figures
#
# Anything after the script name is forwarded to run_experiment.py only, and lands AFTER this
# script's own --attacks, so it wins:
#
#   sbatch experiments/run_dpmlm_var_a10_gemini.sbatch --attacks logistic nearest_neighbor
#
# READ THIS BEFORE INTERPRETING A NULL RESULT, and it bites this defense harder than most.
# gemini_embedding_2 reads only each document's first 8,192 tokens and silently discards (but
# still bills for) the rest. The whole point of the adaptive-length variant is that it changes
# document LENGTH -- and on any document already past that window, the length change downstream of
# the cut is invisible to the embedding. So on Gemini, this arm mostly tests what the add/delete
# coins did to the surviving prefix, not what they did to the length signature. StyloMetrix (the
# run_dpmlm_var_a10.sbatch arm) reads the whole document and is where the length story lives; the
# two are complementary readings, not a repeat.
#
#SBATCH --job-name=dp_mlm_var_a10_gemini
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu-preempt
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-08:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-swe_chat}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-dp_mlm_var_a10}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
PLOT="${PLOT:-1}"

DEFENDED="$DIST_DIR/${SOURCE}_${DEFENSE}.parquet"

# Same env bootstrap the experiment launcher uses, so how a batch shell finds conda stays defined
# in exactly one file. It ends in `exec "$@"`, so calling it as a command runs the stage.
#
# Deliberately NOT `srun`: since SLURM 22.05 a step does not inherit --cpus-per-task from the batch
# allocation, so `srun <stage>` would hand these stages a single CPU -- which here would throttle
# both the embedding client's concurrency and plot_results' worker pool.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "DP-MLM adaptive length on Gemini embeddings: $DEFENSE"
echo "  source:   $SOURCE"
echo "  feature:  $FEATURE"
echo "  defended: $DEFENDED"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), started $(date)"
echo "=================================================================="

# 0a. Fail before spending anything if the defended split is not there. This script does not
#     defend, by design -- see the header.
if [ ! -f "$DEFENDED" ]; then
    echo "ERROR: no defended split at $DEFENDED"
    echo "This script attacks an existing one; it will not run DP-MLM. Produce it first with:"
    echo "  DEFENSE=$DEFENSE sbatch experiments/run_dpmlm_var_a10.sbatch"
    echo "or, if it exists somewhere else, point this job at that directory with DIST_DIR=..."
    exit 1
fi

# 0b. And fail before stage 1 if the key is unreachable, rather than after the runner has walked
#     the corpus. Checking for the file as well as the variable because load_dotenv is what
#     usually supplies it, and it reads a .env at the repo root.
if [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: no OPENROUTER_API_KEY in the environment and no $PROJECT/.env for load_dotenv."
    echo "  cp DS_env/.env .env"
    exit 1
fi

# 1. Featurize the defended text. THIS is the stage that spends money, and it caches by text
#    content -- so a re-run costs nothing and a preempted one resumes where it stopped.
echo ""
echo "------------------------------------------------------------------"
echo "[1/3] featurize  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$DIST_DIR"

# 2. Attack it. --data-dir is data/dist because that is where stage 1 (and the defense run before
#    it) writes; the runner's own default is data/hf, the published mirror, which lacks these
#    files. "$@" comes last so a forwarded --attacks overrides the one here.
echo ""
echo "------------------------------------------------------------------"
echo "[2/3] attack  ($(date))"
echo "------------------------------------------------------------------"
"${RUN[@]}" python experiments/run_experiment.py \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
    --data-dir "$DIST_DIR" --attacks logistic "$@"

# 3. Draw the figures. Cheap and idempotent: it recomputes no attack, reads the result CSVs on
#    disk, and its cache skips every figure nothing has moved under -- so it redraws this run's
#    panels and leaves the rest of the tree alone. --jobs is pinned to the allocation because its
#    own default reads the node's affinity mask, and each worker holds a replicate weight matrix.
if [ "$PLOT" != "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[3/3] plot  ($(date))"
    echo "------------------------------------------------------------------"
    "${RUN[@]}" python experiments/plot_results.py --jobs "${SLURM_CPUS_PER_TASK:-8}"
fi

TAG="${SOURCE}_${DEFENSE}_${FEATURE}"
echo ""
echo "Done $(date)."
echo "  results:  experiments/results/${TAG}_logistic/"
echo "  figures:  experiments/plots/${SOURCE}/"
echo ""
echo "What to compare, and what each comparison answers:"
echo "  vs ${SOURCE}_base_${FEATURE}_logistic       -- how much attribution the defense removed"
echo "                                                 on this channel at all"
echo "  vs ${SOURCE}_${DEFENSE}_stylometrix_logistic -- THE comparison for this variant. Gemini"
echo "                                                 reads a truncated prefix; StyloMetrix reads"
echo "                                                 the whole document and so sees the length"
echo "                                                 change. A defense that only works on one of"
echo "                                                 them has not broken the author signal."
echo "  vs ${SOURCE}_dp_mlm_eps100_${FEATURE}_logistic -- the fixed-length run at the same eps:"
echo "                                                 what breaking the length signature bought."
