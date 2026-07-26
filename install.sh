#!/usr/bin/env bash
#
# install.sh -- one-command setup for the prompt_anonymity project.
#
# Installs the full stack in the order the StyloMetrix + GPU-spaCy combination
# requires:
#   1. spaCy with GPU support (CUDA or Apple Metal)
#   2. the en_core_web_trf transformer model
#   3. StyloMetrix from source  (the PyPI package pins a broken spaCy version)
#   4. the prompt_anonymity package itself, with all its Python dependencies
#      (pulled from pyproject.toml)
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
flag. Note: StyloMetrix uses the GPU when available and otherwise falls back to
CPU (much slower), so the cpu option works but is best paired with reusing the
committed feature vectors.
EOF
}

# --- locations -------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"
STYLOMETRIX_DIR="${STYLOMETRIX_DIR:-$SCRIPT_DIR/StyloMetrix}"
STYLOMETRIX_REPO="https://github.com/NASK-NLP/StyloMetrix"
SPACY_MODEL="en_core_web_trf"

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
command -v git       >/dev/null 2>&1 || die "git is required to fetch StyloMetrix"

if [[ -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  warn "no active virtualenv/conda env detected; packages will install into"
  warn "  $("$PYTHON" -c 'import sys; print(sys.prefix)')"
  warn "see README.md for creating and activating an env first."
fi

# --- 1. spaCy (with GPU support) -------------------------------------------
log "Installing spaCy ($ACCEL) ..."
case "$ACCEL" in
  cuda12x) pip_install -U "spacy[cuda12x]" ;;
  # spaCy has no cuda13x extra: install plain spaCy, then the CUDA 13 cupy build.
  cuda13x) pip_install -U "spacy" && pip_install -U "cupy-cuda13x[ctk]" ;;
  apple)   pip_install -U "spacy[apple]" ;;
  cpu)     warn "installing CPU-only spaCy; StyloMetrix feature computation will be slow without a GPU."
           pip_install -U "spacy" ;;
  *) die "invalid accelerator '$ACCEL' (expected cuda12x, cuda13x, apple, or cpu)" ;;
esac

# Pin numpy to the version spaCy just selected, for every later install. spaCy with the
# cuda12x extra resolves numpy 1.x (cupy-cuda12x is built against numpy 1.x); installing
# StyloMetrix or the package next can otherwise drag numpy up to 2.x and silently break that
# build. cupy is not in their dependency trees, so pip can't see the conflict on its own; a
# constraints file makes the requirement explicit. (cuda13x/apple/cpu just keep their numpy.)
NUMPY_VERSION="$("$PYTHON" -c 'import numpy; print(numpy.__version__)')"
CONSTRAINTS_FILE="$(mktemp)"
trap 'rm -f "$CONSTRAINTS_FILE"' EXIT
printf 'numpy==%s\n' "$NUMPY_VERSION" > "$CONSTRAINTS_FILE"
log "Pinning numpy==$NUMPY_VERSION for the remaining installs."

# --- 2. spaCy transformer model --------------------------------------------
log "Downloading spaCy model: $SPACY_MODEL ..."
pip_install -U click
"$PYTHON" -m spacy download "$SPACY_MODEL"

# --- 3. StyloMetrix from source --------------------------------------------
log "Setting up StyloMetrix ..."
if [[ -d "$STYLOMETRIX_DIR/.git" || -f "$STYLOMETRIX_DIR/setup.cfg" ]]; then
  log "Using existing StyloMetrix checkout: $STYLOMETRIX_DIR"
else
  log "Cloning StyloMetrix into $STYLOMETRIX_DIR ..."
  git clone "$STYLOMETRIX_REPO" "$STYLOMETRIX_DIR"
fi

# Patch the two known issues in-place. Both edits are idempotent, so re-running
# the script (or pointing it at an already-patched checkout) is harmless.
# `-i.bak` + rm keeps this portable across GNU sed and BSD/macOS sed.
if [[ -f "$STYLOMETRIX_DIR/requirements.txt" ]]; then
  # Drop the broken spaCy version pin (e.g. "spacy==3.7.2" -> "spacy"), while
  # leaving spacymoji / spacy_syllables untouched.
  sed -i.bak -E 's/^spacy[[:space:]]*([<>=!~].*)?$/spacy/' \
    "$STYLOMETRIX_DIR/requirements.txt"
  rm -f "$STYLOMETRIX_DIR/requirements.txt.bak"
fi
if [[ -f "$STYLOMETRIX_DIR/setup.cfg" ]]; then
  # Fill in the version placeholder so setuptools can build the package.
  sed -i.bak 's/{{VERSION_PLACEHOLDER}}/1.0.0/g' "$STYLOMETRIX_DIR/setup.cfg"
  rm -f "$STYLOMETRIX_DIR/setup.cfg.bak"
fi

pip_install -e "$STYLOMETRIX_DIR" -c "$CONSTRAINTS_FILE"

# --- 4. prompt_anonymity package + its dependencies ------------------------
# The GPU model-backed defenses live behind two extras rather than the base dependencies, since the
# default --defense none pass-through needs neither:
#   [styleremix] torch/transformers/peft/accelerate/bitsandbytes -- plain wheels, no compiler needed.
#   [qwen]       vllm/llama-cpp-python -- vllm only builds on Linux+CUDA, and falls back to a source
#                build (needing CUDA_HOME / the CUDA toolkit, not just a driver) when no prebuilt
#                wheel matches your CUDA version. Installed separately and non-fatally so a vllm
#                build failure can't take peft/transformers down with it.
#   [dpmlm]      nltk (on top of [styleremix]'s torch/transformers) for dp_mlm.py. Plain wheels;
#                the NLTK data files it needs are fetched right after. The dp_mlm_pii variant's
#                Presidio/spaCy deps ([dpmlm-pii]) are NOT auto-installed here -- install manually if
#                you need them: pip install -e .[dpmlm-pii] && python -m spacy download en_core_web_lg
log "Installing the prompt_anonymity package (editable) and its dependencies ..."
STYLEREMIX_EXTRA="without"
QWEN_EXTRA="without"
DPMLM_EXTRA="without"
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
    ;;
  *)
    warn "accelerator '$ACCEL' has no CUDA; skipping the [styleremix]/[qwen]/[dpmlm] extras."
    warn "run 'pip install -e .[styleremix] -c <constraints>' manually if you need styleremix here."
    pip_install -e "$SCRIPT_DIR" -c "$CONSTRAINTS_FILE" ;;
esac

log "Done. Installed spaCy[$ACCEL], $SPACY_MODEL, StyloMetrix (editable), and"
log "prompt_anonymity (editable, $STYLEREMIX_EXTRA [styleremix], $QWEN_EXTRA [qwen], $DPMLM_EXTRA [dpmlm])."
log "See README.md for running the pipeline."
