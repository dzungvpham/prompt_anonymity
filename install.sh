#!/usr/bin/env bash
#
# install.sh -- one-command setup for the prompt_anonymity project.
#
# Installs the prompt_anonymity package with all its Python dependencies (pulled from
# pyproject.toml) and, on a CUDA machine, its GPU-backed extras.
#
# Run it from inside your activated Python 3.12 environment, e.g.:
#
#   conda create -p ./env python=3.12 && conda activate ./env
#   ./install.sh                          # auto-detect the accelerator
#   ./install.sh --accelerator cuda12x    # or force a specific one

set -euo pipefail

usage() {
  cat <<'EOF'
install.sh -- one-command setup for the prompt_anonymity project.

Usage:
  ./install.sh [--accelerator <cuda12x|cuda13x|apple|cpu>]

Run from inside your activated Python 3.12 virtual environment.

If --accelerator is omitted it is auto-detected (NVIDIA -> cuda12x/cuda13x,
Apple Silicon -> apple) and falls back to cuda12x when no GPU is visible. You
may also set the PROMPT_ANONYMITY_ACCEL environment variable instead of the
flag. The accelerator only decides whether the GPU-backed extras are installed:
--accelerator apple or cpu installs the base package alone.
EOF
}

# --- locations -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

# --- helpers ---------------------------------------------------------------
log()  { printf '\n\033[1;34m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }
pip_install() { "$PYTHON" -m pip install "$@"; }

# --- parse arguments -------------------------------------------------------
ACCEL="${PROMPT_ANONYMITY_ACCEL:-}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --accelerator)
      [[ $# -ge 2 ]] || die "--accelerator requires a value"
      ACCEL="$2"; shift 2 ;;
    --accelerator=*) ACCEL="${1#*=}"; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1 (try --help)" ;;
  esac
done

# --- detect accelerator if not specified -----------------------------------
detect_accel() {
  # Apple Silicon.
  if [[ "$(uname -s)" == "Darwin" && "$(uname -m)" == "arm64" ]]; then
    echo "apple"; return
  fi
  # NVIDIA: read the CUDA major version reported by the driver.
  if command -v nvidia-smi >/dev/null 2>&1; then
    local cuda_major
    cuda_major="$(nvidia-smi 2>/dev/null \
      | sed -n 's/.*CUDA Version: \([0-9]*\).*/\1/p' | head -1 || true)"
    if [[ "$cuda_major" == "13" ]]; then echo "cuda13x"; return; fi
    if [[ -n "$cuda_major" ]]; then echo "cuda12x"; return; fi
  fi
  echo ""  # unknown
}

if [[ -z "$ACCEL" ]]; then
  ACCEL="$(detect_accel)"
  if [[ -z "$ACCEL" ]]; then
    warn "could not detect a GPU; defaulting to --accelerator cuda12x."
    warn "re-run with --accelerator <cuda12x|cuda13x|apple|cpu> if that is wrong."
    ACCEL="cuda12x"
  else
    log "Auto-detected accelerator: $ACCEL"
  fi
fi

# --- sanity checks ---------------------------------------------------------
command -v "$PYTHON" >/dev/null 2>&1 || die "python interpreter '$PYTHON' not found"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  warn "no active virtualenv/conda env detected; packages will install into"
  warn "  $("$PYTHON" -c 'import sys; print(sys.prefix)')"
  warn "see README.md for creating and activating an env first."
fi

case "$ACCEL" in
  cuda12x|cuda13x|apple|cpu) ;;
  *) die "invalid accelerator '$ACCEL' (expected cuda12x, cuda13x, apple, or cpu)" ;;
esac

# Pin numpy to the version already in the environment, for every later install, so that no extra
# drags it up to 2.x and silently breaks a CUDA build compiled against 1.x. Those builds are not
# in the package's dependency tree, so pip can't see the conflict on its own; a constraints file
# makes the requirement explicit.
NUMPY_VERSION="$("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
printf 'numpy==%s\n' "$NUMPY_VERSION" > "$CONSTRAINTS_FILE"
log "Pinning numpy==$NUMPY_VERSION for the remaining installs."

# --- prompt_anonymity package + its dependencies ------------------------
# The GPU model-backed defenses live behind two extras rather than the base dependencies, since the
# default --defense none pass-through needs neither:
#   [styleremix] torch/transformers/peft/accelerate/bitsandbytes -- plain wheels, no compiler needed.
#   [qwen]       vllm/llama-cpp-python -- vllm only builds on Linux+CUDA, and falls back to a source
#                build (needing CUDA_HOME / the CUDA toolkit, not just a driver) when no prebuilt
#                wheel matches your CUDA version. Installed separately and non-fatally so a vllm
#                build failure can't take peft/transformers down with it.
#   [dpmlm]      nltk (on top of [styleremix]'s torch/transformers) for dp_mlm.py (--defense dp_mlm,
#                the dp_mlm_eps<eps> sweep, and the adaptive-length dp_mlm_var_a<A> variants -- all
#                one extra; the variants add no dependency of their own).
#                Plain wheels; the NLTK data files it needs are fetched right after.
#   [dpmlm-pii]  presidio-analyzer + spaCy for the --defense dp_mlm_pii variant's PII detection.
#                Heavier (Presidio needs a spaCy pipeline; we fetch en_core_web_lg, its default);
#                installed separately and non-fatally so a failure can't take the rest down. Only
#                needed for dp_mlm_pii -- plain dp_mlm does not use it.
#   [rerank]     transformers (on top of [styleremix]'s torch) for the listwise_jina_rerank attack's
#                local reranker. Plain wheels; the 0.6B checkpoint is fetched right after. The
#                listwise_llm_rerank attack beside it needs no extra -- it talks HTTP (OpenRouter) or
#                the Anthropic SDK (Foundry), and `anthropic` is a core dependency, so every branch
#                below installs it with the package itself.
log "Installing the prompt_anonymity package (editable) and its dependencies ..."
STYLEREMIX_EXTRA="without"
QWEN_EXTRA="without"
DPMLM_EXTRA="without"
DPMLM_PII_EXTRA="without"
FEATURES_EXTRA="without"
ARGOS_EXTRA="without"
RERANK_EXTRA="without"
case "$ACCEL" in
  cuda12x|cuda13x)
    pip_install -e "$SCRIPT_DIR[styleremix]" -c "$CONSTRAINTS_FILE"
    STYLEREMIX_EXTRA="with"
    if pip_install -e "$SCRIPT_DIR[qwen]" -c "$CONSTRAINTS_FILE"; then
      QWEN_EXTRA="with"
    else
      warn "the [qwen] extra (vllm/llama-cpp-python) failed to install -- qwen_rewrite.py won't work."
      warn "vllm often needs CUDA_HOME set to your CUDA toolkit to build from source; see its docs."
      warn "styleremix.py is unaffected: its [styleremix] extra installed separately above."
    fi
    pip_install -e "$SCRIPT_DIR[dpmlm]" -c "$CONSTRAINTS_FILE"
    DPMLM_EXTRA="with"
    log "Downloading NLTK data for dp_mlm.py (punkt, stopwords, wordnet) ..."
    "$PYTHON" -m nltk.downloader -q punkt punkt_tab stopwords wordnet || \
      warn "NLTK data download failed; dp_mlm.py will retry it lazily on first use (needs network)."
    # style_distance and luar featurizers: sentence-transformers + transformers + einops (all
    # reusing torch from [styleremix]). einops is there for LUAR's Hub-side modeling code, not for
    # ours -- see the [features] comment in pyproject.toml.
    pip_install -e "$SCRIPT_DIR[features]" -c "$CONSTRAINTS_FILE"
    FEATURES_EXTRA="with"
    # Fetch LUAR's weights, tokenizer and modeling code now rather than on first use, so a compute
    # node with no outbound network can run --feature luar. The model id and pinned revision are
    # read from the featurizer itself to keep one source of truth; importing it costs nothing here
    # because it imports torch/transformers lazily.
    log "Downloading the LUAR checkpoint for the luar featurizer ..."
    "$PYTHON" -c 'from huggingface_hub import snapshot_download as fetch; from prompt_anonymity.features.luar import DEFAULT_MODEL_ID as model, DEFAULT_REVISION as revision; fetch(model, revision=revision)' || \
      warn "LUAR checkpoint download failed; the luar featurizer will retry it lazily on first use (needs network)."
    # rtt_argos defense: argostranslate. Non-fatal -- other defenses are unaffected.
    if pip_install -e "$SCRIPT_DIR[argos]" -c "$CONSTRAINTS_FILE"; then
      ARGOS_EXTRA="with"
    else
      warn "the [argos] extra (argostranslate) failed to install -- --defense rtt_argos won't work."
    fi
    # dp_mlm_pii only: Presidio + its default spaCy model. Non-fatal -- plain dp_mlm is unaffected.
    if pip_install -e "$SCRIPT_DIR[dpmlm-pii]" -c "$CONSTRAINTS_FILE" \
       && "$PYTHON" -m spacy download en_core_web_lg; then
      DPMLM_PII_EXTRA="with"
    else
      warn "the [dpmlm-pii] extra (presidio-analyzer/spaCy en_core_web_lg) failed to install --"
      warn "  --defense dp_mlm_pii won't work, but --defense dp_mlm is unaffected."
    fi
    # listwise_jina_rerank attack: transformers on top of [styleremix]'s torch. Non-fatal -- the
    # attacks that do not use a local reranker are unaffected.
    if pip_install -e "$SCRIPT_DIR[rerank]" -c "$CONSTRAINTS_FILE"; then
      RERANK_EXTRA="with"
      # Fetch the reranker now rather than on first use, so a compute node with no outbound network
      # can run the attack. The repo id is read from the attack itself to keep one source of truth;
      # importing it costs nothing here because it imports torch/transformers lazily.
      log "Downloading the jina-reranker checkpoint for the listwise_jina_rerank attack ..."
      "$PYTHON" -c 'from huggingface_hub import snapshot_download as fetch; from prompt_anonymity.attacks.llm.listwise_jina_rerank import DEFAULT_MODEL_ID as model; fetch(model)' || \
        warn "jina-reranker download failed; the attack will retry it lazily on first use (needs network)."
    else
      warn "the [rerank] extra (transformers) failed to install -- the listwise_jina_rerank attack"
      warn "  won't work. listwise_llm_rerank is unaffected: it calls an API, not a local model."
    fi
    ;;
  *)
    warn "accelerator '$ACCEL' has no CUDA; skipping the GPU extras ([styleremix]/[qwen]/[dpmlm]/[dpmlm-pii]/[features]/[argos]/[rerank])."
    warn "run 'pip install -e .[all] -c <constraints>' manually if you need them here."
    pip_install -e "$SCRIPT_DIR" -c "$CONSTRAINTS_FILE" ;;
esac

log "Done. Installed prompt_anonymity (editable, $STYLEREMIX_EXTRA [styleremix], $QWEN_EXTRA [qwen], $DPMLM_EXTRA [dpmlm], $DPMLM_PII_EXTRA [dpmlm-pii], $FEATURES_EXTRA [features], $ARGOS_EXTRA [argos], $RERANK_EXTRA [rerank])."
log "See README.md for running the pipeline."
