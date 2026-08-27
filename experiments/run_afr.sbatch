#!/bin/bash
#
# run_afr.sbatch -- one GPU job running the full three-stage pipeline for Agentic Footprint
# Reduction at one residual-linkage level.
#
#   sbatch experiments/run_afr.sbatch
#
# SUBMIT FROM THE REPO ROOT. --output below is a relative path, and SLURM resolves it against the
# directory you submit from.
#
# ONE-TIME SETUP -- SLURM fails a job whose --output directory does not exist:
#
#   mkdir -p scripts/logs
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#
#   DEFENSE=afr_a50 sbatch experiments/run_afr.sbatch        # a different point on the sweep
#   DEFENSE=afr_stage1 sbatch experiments/run_afr.sbatch     # the loop ablation (see below)
#   FEATURE=harrier sbatch experiments/run_afr.sbatch        # the surrogate-overfit check
#   SOURCE=swe_chat sbatch experiments/run_afr.sbatch        # default wildchat_small
#
# No model paths are needed on Unity: both checkpoints resolve to the cluster's mirror under
# /datasets/ai automatically. Set AFR_MODEL / HARRIER_MODEL in .env only if your copies are filed
# somewhere that lookup will not find, or to swap the agent's quantization.
#
# Anything after the script name is forwarded to run_experiment.py only (not to the build stages):
#
#   sbatch experiments/run_afr.sbatch --attacks nearest_neighbor logistic
#
# THE TWO RUNS THAT MAKE THIS INTERPRETABLE, and neither is optional:
#
#   DEFENSE=afr_stage1 sbatch experiments/run_afr.sbatch
#       Same model, same abstraction pass, zero probes. `afr` vs `afr_stage1` is what isolates the
#       measurement loop; `afr` vs `none` only shows that a 30B model rewrote the text.
#
#   FEATURE=harrier sbatch experiments/run_afr.sbatch
#       Harrier is what the defense optimized against, so a large drop here with no drop on
#       gemini_embedding_2 is surrogate overfitting. Report it as such rather than burying it.
#
# WHY gpu_large RESOURCES. Unlike the other defense sbatch files, this one does NOT mirror
# [profiles.gpu] in scripts/slurm.toml. The agent is a 30B-class model in FP8 (~33 GB of weights
# plus a KV cache) and Harrier-0.6B co-resides on the same card, so a 48 GB L40S or a 40 GB A100
# will OOM during engine startup. The constraint below asks for an 80 GB card. If your cluster
# names those differently, this line and scripts/slurm.toml are the only places to change.
#
# The wall clock is 16h rather than 12: AFR spends roughly 11 generations per document against a
# large model, and unlike the DP-MLM jobs it cannot be sharded (see below).
#
# NO SHARDING. `apply_defenses --num-shards` splits by document, and this defense cascades each
# author's timeline in order -- a shard holding an arbitrary subset of an author would defend
# document k against the wrong priors. `apply_defenses` refuses the run outright. The subset this
# is built for (`wildchat_small`, ~200 authors) does not need it.
#
# PREEMPTION IS SAFE. gpu-preempt jobs get killed; --requeue puts this back in the queue and all
# three stages resume rather than restart -- apply_defenses checkpoints its cache every 5 AUTHORS
# (the cascade's atomic unit, so a resumed job never restarts an author mid-timeline),
# compute_features caches per text, and run_experiment skips finished output.
#
#SBATCH --job-name=afr
#SBATCH --output=scripts/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --nodes=1
#SBATCH --partition=gpu-preempt
#SBATCH --gpus=1
#SBATCH --constraint=sm_90|a100-80g
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=0-16:00:00
#SBATCH --requeue

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$PROJECT"

SOURCE="${SOURCE:-wildchat_small}"
FEATURE="${FEATURE:-gemini_embedding_2}"
DEFENSE="${DEFENSE:-afr}"
DIST_DIR="${DIST_DIR:-$PROJECT/data/dist}"

# Each stage goes through scripts/slurm_job.sh, the same env bootstrap the experiment launcher uses,
# so how a batch shell here finds conda stays defined in exactly one file. It ends in `exec "$@"`.
#
# Deliberately NOT `srun`: since SLURM 22.05 a step does not inherit --cpus-per-task from the batch
# allocation, so `srun <stage>` would hand these stages a single CPU.
RUN=(bash scripts/slurm_job.sh)

echo "=================================================================="
echo "Agentic Footprint Reduction: $DEFENSE  (source=$SOURCE feature=$FEATURE)"
echo "Job $SLURM_JOB_ID on $(hostname), started $(date)"
echo "=================================================================="

# Neither $AFR_MODEL nor $HARRIER_MODEL is required: both resolve through
# _backends.model_checkpoint, which prefers the cluster's own mirror of the repo id in models.toml
# (/datasets/ai/*/hub/models--<org>--<name>) and falls back only when there is none. Resolve both
# here so the log says where the weights are coming from BEFORE the queue time is spent -- the
# failure this catches is a defend stage that stalls for twenty minutes downloading 33 GB, or dies
# in from_pretrained because Harrier is not where it was assumed to be.
"${RUN[@]}" python -c "
from prompt_anonymity.defenses._backends import model_checkpoint
model_checkpoint('afr', 'AFR_MODEL')
model_checkpoint('harrier', 'HARRIER_MODEL', local_only=True)" || {
    echo "FATAL: could not resolve a checkpoint. Set AFR_MODEL / HARRIER_MODEL in .env." >&2
    exit 1
}

# vLLM has to be recent enough to serve an FP8 MoE. If this line reports something old and the
# defend stage dies in engine startup, the documented fallback is AFR_MODEL=<Qwen3-32B-AWQ> with
# gpu_memory_utilization dropped to 0.55 in defenses/afr.py's _LocalBackend.
"${RUN[@]}" python -c "import vllm; print(f'vllm {vllm.__version__}')"

# 1. Defend: the long stage. ~11 generations per document against a 30B model, cascaded per author.
#    Watch the printed workload line for the document/author counts and the reference-pool size.
"${RUN[@]}" python -m prompt_anonymity.data.apply_defenses \
    --source "$SOURCE" --defense "$DEFENSE" --dist-dir "$DIST_DIR"

# 2. Featurize the defended text. This is the ATTACKER's encoder and is deliberately not the one the
#    defense optimized against -- that mismatch is what makes the number a transfer result rather
#    than self-grading.
"${RUN[@]}" python -m prompt_anonymity.data.compute_features \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" --dist-dir "$DIST_DIR"

# 3. Attack it. --data-dir is data/dist because that is where the two stages above write; the
#    runner's own default is data/hf, the published mirror, which will not have these files.
"${RUN[@]}" python experiments/run_experiment.py \
    --source "$SOURCE" --defense "$DEFENSE" --feature "$FEATURE" \
    --data-dir "$DIST_DIR" "$@"

echo ""
echo "Done $(date). Results in experiments/results/${SOURCE}_${DEFENSE}_${FEATURE}_*/"
echo ""
echo "Read the edit log before the results -- it says whether the defense did what it claims:"
echo "  jq -s 'group_by(.stop_reason)|map({(.[0].stop_reason): length})|add' \\"
echo "     data/.cache/defended/${SOURCE}/afr_logs/edits_a*.jsonl"
echo "  target_met dominating  -> the loop is working"
echo "  ladder_exhausted       -> the agent stalls; the propose prompt needs work"
echo "  max_level always 0     -> escalation never fires; the trigger is mistuned"
echo ""
echo "Then compare against the controls:"
echo "  ${SOURCE}_afr_stage1_${FEATURE}_*   the loop ablation (same model, no probes)"
echo "  ${SOURCE}_loo_unlink_b30_${FEATURE}_*   the greedy-rule alternative at a similar target"
echo "  ${SOURCE}_base_${FEATURE}_*             undefended"
echo "Utility (free metrics first, the paid judge only once those look sane):"
echo "  python experiments/eval_utility.py --source $SOURCE --defense $DEFENSE"
echo "Draw figures with: python experiments/plot_results.py"
