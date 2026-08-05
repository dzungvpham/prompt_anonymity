#!/bin/bash
#
# The batch wrapper `run_all_experiments.py --slurm` submits every cell through:
#
#     sbatch <resource flags> scripts/slurm_job.sh python experiments/run_experiment.py --source ...
#
# sbatch forwards everything after the script name to the script, so this file's whole job is to
# put the environment together and hand over. It runs whatever it is given -- it is not specific
# to the experiment runner.
#
# THIS IS THE SECOND (AND LAST) FILE TO EDIT FOR A DIFFERENT CLUSTER: it holds how a batch shell
# here gets a working conda env. Resources live in scripts/slurm.toml, not here.
#
# There are deliberately NO #SBATCH directives in this file. Every resource flag comes from
# slurm.toml via the sbatch command line, so there is exactly one place to look when a job asks
# for the wrong thing -- and no in-file directive that the command line silently overrides.
#
# Nothing here is hardcoded to one checkout. The launcher passes --chdir=<project root>, so a job
# it submits already starts there; $SLURM_SUBMIT_DIR and then $PWD are the fallbacks for running
# this file by hand. The env is an `env/` beside the project unless $PROMPT_ANONYMITY_CONDA_ENV
# says otherwise; sbatch exports the submitting environment by default, so that variable (and
# $PROMPT_ANONYMITY_PROJECT) can just be set in your shell before launching.
#
# Do not switch the project root to this script's own path: sbatch copies the batch script to a
# spool directory on the node, so $BASH_SOURCE at run time is not where this file lives.

set -euo pipefail

PROJECT="${PROMPT_ANONYMITY_PROJECT:-${SLURM_SUBMIT_DIR:-$PWD}}"
CONDA_ENV="${PROMPT_ANONYMITY_CONDA_ENV:-$PROJECT/env}"
CONDA_MODULE="${PROMPT_ANONYMITY_CONDA_MODULE:-conda/latest}"

cd "$PROJECT"

# Python block-buffers stdout when it is a file rather than a terminal, which is exactly what
# --output makes it: a run's progress then sits in an 8 KB buffer and lands in the log all at once
# when the process exits. That turns `tail -f` into "nothing is happening" for an hour, and a job
# killed by its time limit loses the buffer entirely -- the two moments a log is most needed.
# Unbuffering costs nothing at this volume (a few hundred lines per run).
export PYTHONUNBUFFERED=1

# Each step is conditional because how conda arrives differs per cluster, and an unconditional
# `module load` is a hard failure on a site that has no Lmod at all. sbatch exports the submitting
# environment by default, so on Unity `module` is usually already here; ~/.bashrc is the fallback
# that defines it, and is sourced with -u off because other people's rc files are not -u clean.
if ! command -v module >/dev/null 2>&1 && [[ -f ~/.bashrc ]]; then
    set +u
    # shellcheck disable=SC1090
    source ~/.bashrc
    set -u
fi
if ! command -v conda >/dev/null 2>&1 && command -v module >/dev/null 2>&1; then
    module load "$CONDA_MODULE"
fi
if ! command -v conda >/dev/null 2>&1; then
    echo "no conda on PATH: set PROMPT_ANONYMITY_CONDA_MODULE, or edit this file for your site" >&2
    exit 1
fi

# The hook rather than a bare `conda activate`: activation is a shell function, and whether it is
# defined in a batch shell depends on whether conda's init block ran in a file this shell read.
eval "$(conda shell.bash hook)"
echo "Activating $CONDA_ENV"
conda activate "$CONDA_ENV"

# Only when this job actually asked for one, so the CPU cells' logs are not a page of error text.
if [[ -n "${SLURM_JOB_GPUS:-${SLURM_GPUS_ON_NODE:-}}" ]]; then
    nvidia-smi
fi

echo "Running: $*"
exec "$@"
