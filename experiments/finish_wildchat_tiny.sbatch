#!/bin/bash
#
# finish_wildchat_tiny.sbatch -- bring every defense that has already been DEFENDED through to a
# figure: check which ones are missing Gemini embeddings, embed those, attack those, then plot.
#
#   sbatch experiments/finish_wildchat_tiny.sbatch
#
# SUBMIT FROM THE REPO ROOT (--output is relative and SLURM resolves it against the submit dir).
#
# NO GPU. Gemini embeddings are an OpenRouter API call, the logistic attack is CPU, and plotting is
# CPU -- so this asks for a CPU partition and starts far sooner than the defend jobs did. If your
# cluster's CPU partition is not called "cpu", override it rather than editing this file:
#
#     sbatch --partition=<name> experiments/finish_wildchat_tiny.sbatch
#     sinfo -s                                  # to see what the partitions are called
#
# SEE WHAT IS MISSING WITHOUT DOING ANYTHING -- runs in a second, on the login node:
#
#     CHECK_ONLY=1 bash experiments/finish_wildchat_tiny.sbatch
#
# WHY THE ATTACK STAGE IS HERE, when you asked for "embed and plot": `plot_results.py` draws from
# `experiments/results/<...>/rolling_results.csv`, which `run_experiment.py` writes. Embeddings
# alone produce no figure. So each arm goes embed -> attack -> (finally) plot, and any arm that
# already has results is skipped.
#
# WHAT COUNTS AS "ALREADY DONE", per arm:
#   defended split   data/dist/<source>[_<defense>].parquet          -- absent => the arm never ran
#   embeddings       data/dist/<source>[_<defense>]_<feature>.parquet
#   attack results   experiments/results/<source>_<tag>_<feature>_<attack>/rolling_results.csv
# where <tag> is the defense name, except `none`, which is spelled `base` downstream.
#
# COST. One OpenRouter embedding call per document at $0.20/M tokens, and it caches by text content,
# so re-running this is free for anything already embedded. ~900 documents per arm is cents.
#
# Overrides (sbatch exports your submitting environment):
#   SOURCE=wildchat_small sbatch ...
#   FEATURE=harrier sbatch ...        # local + free, but needs a GPU: add --gpus=1
#   DEFENSES="afr afr_stage1" sbatch ...
#   PLOT=0 sbatch ...                 # embed and attack only
#
#SBATCH --job-name=finish-wct
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=0-04:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-wildchat_tiny}"
FEATURE="${FEATURE:-gemini_embedding_2}"
ATTACK="${ATTACK:-logistic}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"
RESULTS_DIR="${RESULTS_DIR:-$PROJECT/experiments/results}"
PLOT="${PLOT:-1}"
CHECK_ONLY="${CHECK_ONLY:-0}"

# Every arm the grid can produce. `none` first because it is the baseline every figure needs.
read -r -a DEFENSE_LIST <<< "${DEFENSES:-none example_normalization collision_seeding frame_shift frame_pad qwen_rewrite styleremix styleremix_openanon loo_unlink_b30 afr_stage1 afr}"

# Bare `bash`/`python` when checking on a login node; the env bootstrap only when running as a job.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

# `none` is spelled `base` in results/ and on figures (run_experiment.NO_DEFENSE_TAG).
tag_of() { [ "$1" = "none" ] && echo base || echo "$1"; }
# split_stem(): the source alone for `none`, else <source>_<defense>.
stem_of() { [ "$1" = "none" ] && echo "$SOURCE" || echo "${SOURCE}_$1"; }

echo "=================================================================="
echo "finish $SOURCE / $FEATURE / $ATTACK"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. status ------------------------------------------------------------
# Printed as a table first, so the log opens with the whole picture rather than making you infer it
# from what did or did not run afterwards.
NEED_EMBED=()
NEED_ATTACK=()
printf "\n%-22s %-9s %-11s %-9s\n" "arm" "defended" "embeddings" "results"
printf "%-22s %-9s %-11s %-9s\n" "----------------------" "---------" "-----------" "---------"
for defense in "${DEFENSE_LIST[@]}"; do
    stem="$(stem_of "$defense")"
    split="$DIST_DIR/${stem}.parquet"
    features="$DIST_DIR/${stem}_${FEATURE}.parquet"
    results="$RESULTS_DIR/${SOURCE}_$(tag_of "$defense")_${FEATURE}_${ATTACK}/rolling_results.csv"

    if [ ! -f "$split" ]; then
        printf "%-22s %-9s %-11s %-9s   (never defended -- skipped)\n" "$defense" "no" "-" "-"
        continue
    fi
    have_f="no"; [ -f "$features" ] && have_f="yes"
    have_r="no"; [ -f "$results" ]  && have_r="yes"
    printf "%-22s %-9s %-11s %-9s\n" "$defense" "yes" "$have_f" "$have_r"
    [ "$have_f" = "no" ] && NEED_EMBED+=("$defense")
    [ "$have_r" = "no" ] && NEED_ATTACK+=("$defense")
done

echo ""
echo "to embed: ${#NEED_EMBED[@]}   to attack: ${#NEED_ATTACK[@]}"

if [ "$CHECK_ONLY" = "1" ]; then
    echo "(CHECK_ONLY=1 -- nothing run)"
    exit 0
fi

# 0. A missing key would otherwise surface one arm into the run rather than at the start.
if [ "$FEATURE" = "gemini_embedding_2" ] && [ "${#NEED_EMBED[@]}" -gt 0 ] \
   && [ -z "${OPENROUTER_API_KEY:-}" ] && [ ! -f "$PROJECT/.env" ]; then
    echo "ERROR: $FEATURE needs OPENROUTER_API_KEY, and there is none in the environment nor a" >&2
    echo "       $PROJECT/.env for load_dotenv to read." >&2
    exit 1
fi

# --- 2. embed what is missing --------------------------------------------
for defense in "${NEED_EMBED[@]+"${NEED_EMBED[@]}"}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[embed] $defense  ($(date))"
    echo "------------------------------------------------------------------"
    # `none` is the undefended split itself, so it takes no --defense flag.
    if [ "$defense" = "none" ]; then
        run python -m prompt_anonymity.data.compute_features \
            --source "$SOURCE" --feature "$FEATURE" --dist-dir "$DIST_DIR"
    else
        run python -m prompt_anonymity.data.compute_features \
            --source "$SOURCE" --defense "$defense" --feature "$FEATURE" --dist-dir "$DIST_DIR"
    fi
done

# --- 3. attack what has no results ---------------------------------------
# `|| true` per arm: one arm failing (too few authors for a window, say) must not cost the figure
# for the ten that worked. Failures are listed at the end.
FAILED=()
for defense in "${NEED_ATTACK[@]+"${NEED_ATTACK[@]}"}"; do
    echo ""
    echo "------------------------------------------------------------------"
    echo "[attack] $defense  ($(date))"
    echo "------------------------------------------------------------------"
    if ! run python experiments/run_experiment.py \
            --source "$SOURCE" --defense "$defense" --feature "$FEATURE" \
            --data-dir "$DIST_DIR" --attacks "$ATTACK"; then
        echo "WARNING: attack failed for $defense -- continuing" >&2
        FAILED+=("$defense")
    fi
done

# --- 4. plot --------------------------------------------------------------
if [ "$PLOT" != "0" ]; then
    echo ""
    echo "------------------------------------------------------------------"
    echo "[plot]  ($(date))"
    echo "------------------------------------------------------------------"
    # --force because plot_results caches figures whose inputs have not moved; new results should
    # always redraw. --jobs pinned to the allocation rather than the node's affinity mask.
    run python experiments/plot_results.py --jobs "${SLURM_CPUS_PER_TASK:-8}" --force
fi

echo ""
echo "Done $(date)."
if [ "${#FAILED[@]}" -gt 0 ]; then
    echo "ARMS THAT FAILED THEIR ATTACK: ${FAILED[*]}"
fi
echo "Figures:"
echo "  experiments/plots/${SOURCE}/accuracy/doc/by_defense/${FEATURE}_${ATTACK}.pdf"
echo "  experiments/plots/${SOURCE}/accuracy/author/by_defense/${FEATURE}_${ATTACK}.pdf"
echo ""
echo "One line per defense. If an arm is missing from the figure, plot_results skipped its"
echo "directory -- most often because its name is not in plot_results.DEFENSES (rtt_argos is"
echo "deliberately absent). The plot stage prints every directory it skips, and why."
