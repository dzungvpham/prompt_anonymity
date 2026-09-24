#!/bin/bash
#
# run_rerank_sonnet_swe_chat.sbatch -- rerank the top-5 nearest-neighbor author shortlist on SWE-chat
# (gemini_embedding_2 vectors) with Claude Sonnet 5 on MICROSOFT FOUNDRY, reasoning ON, candidates
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
# CREDENTIALS, both in a .env AT THE REPO ROOT (or exported):
#   SONNET_API_KEY=<the Foundry key>
#   FOUNDRY_ENDPOINT=<the Foundry endpoint URL>
# BOTH are required: with the key alone the Anthropic SDK would post an Azure key to Anthropic's own
# API and fail like a bad key. The judge is the Foundry DEPLOYMENT claude-sonnet-5-2 (MODEL=...), via
# the Anthropic SDK's AnthropicFoundry client (src/prompt_anonymity/attacks/llm/_foundry.py). Needs
# the `anthropic` package: re-run install.sh, or `pip install anthropic`, if step 1 says it is missing.
# PROVIDER=openrouter runs the old OpenRouter channel instead (SONNET_API_KEY must then be an
# OpenRouter key -- the two channels share the variable).
#
# COST: full price, no batch discount -- roughly $0.012 per unknown document at top-5 at Sonnet 5's
# $2/$10 per MTok (Foundry bills at standard Anthropic rates; the cost column is an estimate from
# token counts, as Foundry replies carry no price). Start with LIMIT=20 (~25 cents), read the
# reasons and the audit, then run the lot. Verdicts are cached by prompt text the moment they
# arrive, so the full run re-pays nothing and a preempted/requeued job resumes.
#
# REASONING IS ON AND CANNOT BE SWITCHED OFF HERE. Every request carries adaptive thinking at EFFORT
# (default high; low|medium|high|xhigh|max, nothing else accepted). Checked three ways: the attack is
# built exactly as the run builds it and its effort asserted (step 3); the preflight probe sends one
# real request with thinking on and must not be rejected (step 4); and after the run, every fresh
# reply is inspected for thinking blocks -- a run where none of them thought fails (step 6). Effort
# is also part of the verdict cache key, so no verdict from a non-reasoning run can be served.
#
# THE OUTPUT BUDGET IS 16,000 TOKENS, thinking included. A reply cut off at the budget parses as
# nothing and silently falls back to the distance order, so step 6 counts truncated replies and
# fails above 5%. Only tokens produced are billed; the budget is a ceiling, not a spend.
#
# SCRAMBLING IS ON AND CHECKED TWICE. Each row's K candidates are shown in a seeded random order
# (list rerankers favour whichever end they read last, which fed nearest-first rewards the metric's
# WORST candidate). Step 3 asserts the attack is configured to shuffle and that the permutation is
# really random; step 6 reads back, row by row, the distance order Sonnet was actually shown.
#
# Top-5 at k=5 is pinned to the shortlist's own recall (run_rerank.py checks it row by row);
# top-1/2/3 are the reranker's contribution.
#
# Preflight on a login node (nothing reranked; the Foundry probe costs a fraction of a cent):
#
#     CHECK_ONLY=1 bash experiments/run_rerank_sonnet_swe_chat.sbatch
#
# Overrides (sbatch exports your submitting environment, so these are read at run time):
#   LIMIT=20 sbatch ...                    # a seeded sample of the unknown side; bounds the bill
#   EFFORT=xhigh sbatch ...                # thinking depth: low/medium/high/xhigh/max (never off)
#   WORKERS=24 sbatch ...                  # concurrent requests (default 16; 429s back off)
#   TOP_K=10 sbatch ...                    # a different shortlist size
#   MODEL=claude-sonnet-5-2 ...            # Foundry deployment name (or an OpenRouter slug)
#   PROVIDER=openrouter sbatch ...         # the old OpenRouter channel
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
PROVIDER="${PROVIDER:-foundry}"
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
case "$PROVIDER" in
    foundry|openrouter) ;;
    *) echo "ERROR: PROVIDER='$PROVIDER' -- must be foundry or openrouter." >&2
       exit 1 ;;
esac

# `none` is spelled `base` in results/ (run_experiment.NO_DEFENSE_TAG); the split stem is the
# source alone for `none`, else <source>_<defense>.
if [ "$DEFENSE" = "none" ]; then TAG=base; STEM="$SOURCE"; else TAG="$DEFENSE"; STEM="${SOURCE}_$DEFENSE"; fi

# The `llm` subdirectory is the one submit_rerank_swe_chat.sh gives this arm, so
# finish_rerank.sbatch can merge it with the jina arm without racing on rerank_summary.csv.
OUT_DIR="${OUT_DIR:-$PROJECT/experiments/results/rerank/${SOURCE}_${TAG}_${FEATURE}_known${WINDOW}/llm}"

# The Python checks below read their settings from the environment.
export TOP_K SEED EFFORT PROVIDER MODEL OUT_DIR

# Bare `python` when checking on a login node; the env bootstrap only when running as a job.
if [ "$CHECK_ONLY" = "1" ]; then
    RUN=()
else
    RUN=(bash scripts/slurm_job.sh)
fi
run() { if [ "${#RUN[@]}" -eq 0 ]; then "$@"; else "${RUN[@]}" "$@"; fi; }

echo "=================================================================="
echo "listwise rerank (Sonnet 5 via $PROVIDER, unbatched) -- $SOURCE / $FEATURE / known$WINDOW / top-$TOP_K"
echo "  reasoning effort: $EFFORT"
echo "  concurrency:      $LISTWISE_RERANK_MAX_WORKERS requests in flight"
if [ -n "${LIMIT:-}" ]; then
    echo "  unknown side:     a seeded sample of $LIMIT documents"
else
    echo "  unknown side:     all of it (full price)"
fi
echo "  output:           $OUT_DIR"
echo "Job ${SLURM_JOB_ID:-local} on $(hostname), $(date)"
echo "=================================================================="

# --- 1. environment --------------------------------------------------------
echo ""
echo "[1/6] environment ($(date))"
run python -c '
import os, sys
print("  interpreter:", sys.executable)
import dotenv, requests
if os.environ["PROVIDER"] == "foundry":
    try:
        import anthropic
        from anthropic import AnthropicFoundry  # noqa: F401
    except ImportError as error:
        sys.exit(f"ERROR: {error}. The Foundry provider needs the anthropic SDK: re-run install.sh "
                 f"or `pip install anthropic` in this environment.")
    print("  anthropic", anthropic.__version__)
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

# --- 3. configuration: provider, reasoning, budget, scramble -----------------
# Builds the attack through run_rerank.build_attack -- the exact call the real run makes -- and
# asserts on the object, so a default flipped anywhere upstream fails here, before any money.
echo ""
echo "[3/6] configuration check: provider, reasoning, budget, scramble ($(date))"
PRE_CHECK=$(cat <<'PY'
import math, os, sys
from argparse import Namespace

import numpy as np

sys.path.insert(0, "experiments")
from run_rerank import build_attack
from prompt_anonymity.attacks.llm import _foundry
from prompt_anonymity.attacks.llm.candidates import AuthorCandidates
from prompt_anonymity.attacks.llm.listwise import present
from prompt_anonymity.attacks.llm.listwise_llm_rerank import SYNC_API_KEY_ENV

k, seed, effort = int(os.environ["TOP_K"]), int(os.environ["SEED"]), os.environ["EFFORT"]
provider = os.environ["PROVIDER"]
attack = build_attack("llm", k, Namespace(
    margin_quantile=1.0, seed=seed, batch=False, wait=True, provider=provider,
    judge_model=os.environ.get("MODEL") or None, reasoning_effort=effort))

problems = []
if attack.batch:
    problems.append("attack.batch is True -- this job only runs the unbatched channel")
if attack.provider != provider:
    problems.append(f"attack.provider is {attack.provider!r}, expected {provider!r}")
if not attack.reasoning_effort or attack.reasoning_effort != effort:
    problems.append(f"reasoning_effort is {attack.reasoning_effort!r}, expected {effort!r}")
if attack.shuffle_candidates is not True:
    problems.append(f"shuffle_candidates is {attack.shuffle_candidates!r} -- candidates would be "
                    f"shown nearest-first")
budget = attack._max_tokens(k)
if provider == "foundry":
    keys = f"{_foundry.FOUNDRY_API_KEY_ENV} + {_foundry.FOUNDRY_ENDPOINT_ENV}"
    if (_foundry.FOUNDRY_API_KEY_ENV, _foundry.FOUNDRY_ENDPOINT_ENV) != ("SONNET_API_KEY",
                                                                        "FOUNDRY_ENDPOINT"):
        problems.append(f"the Foundry client reads {keys}, not SONNET_API_KEY + FOUNDRY_ENDPOINT")
    if budget < 16000:
        problems.append(f"output budget is {budget} tokens; thinking needs >= 16000")
else:
    keys = SYNC_API_KEY_ENV

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

print(f"  provider:         {attack.provider} (unbatched), credentials {keys}")
print(f"  judge model:      {attack.judge_model}")
print(f"  reasoning:        adaptive thinking, effort {attack.reasoning_effort}"
      if provider == "foundry" else f"  reasoning:        reasoning.effort {attack.reasoning_effort}")
print(f"  output budget:    {budget} tokens per reply (thinking included)")
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
if [ "$PROVIDER" = "foundry" ]; then
    # One real request with the run's own settings: proves the endpoint answers, the deployment name
    # exists, and adaptive thinking + this effort are accepted rather than 400'd. ~$0.001. The key
    # itself is never printed.
    PROBE=$(cat <<'PY'
import os, sys
from prompt_anonymity.attacks.llm._foundry import DEFAULT_FOUNDRY_MODEL, probe

model = os.environ.get("MODEL") or DEFAULT_FOUNDRY_MODEL
try:
    result = probe(model, os.environ["EFFORT"])
except Exception as error:
    sys.exit(f"ERROR: the Foundry probe failed -- {type(error).__name__}: {error}")
print(f"  deployment {model} answered {result['reply']!r}: "
      f"{result['input_tokens']}+{result['output_tokens']} tokens, ~${result['cost_usd']:.4f}; "
      f"thinking block present: {result['reasoned']} (adaptive thinking may skip a trivial "
      f"question -- the run itself is what must show it)")
if result["refused"] or result["truncated"]:
    sys.exit("ERROR: the probe was refused or truncated -- check the deployment before paying for a run")
PY
)
    run python -c "$PROBE"
else
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
fi

if [ "$CHECK_ONLY" = "1" ]; then
    echo ""
    echo "(CHECK_ONLY=1 -- preflight only, nothing reranked)"
    exit 0
fi

# --- 5. the attack ---------------------------------------------------------
ARGS=(--source "$SOURCE" --feature "$FEATURE" --defense "$DEFENSE"
      --data-dir "$DIST_DIR" --known-window "$WINDOW" --seed "$SEED"
      --cache-dir "$CACHE_DIR" --output-dir "$OUT_DIR"
      --variant llm --top-k "$TOP_K" --no-batch --provider "$PROVIDER"
      --reasoning-effort "$EFFORT")
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
row = summary[(summary["variant"] == "llm") & (summary["top_k"] == k)]
problems = []


def stat(name):
    """One summary number for this configuration, or nan if the column/row is absent."""
    return float(row[name].iloc[0]) if name in row and len(row) else float("nan")


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
replies = stat("judge_replies")
if pd.isna(replies) or replies == 0:
    print("  reasoning: every row was served from the cache -- no new evidence this run "
          "(the effort is part of the cache key, so these came from a reasoning run)")
else:
    reasoned = stat("judge_replies_with_reasoning")
    tokens = stat("judge_reasoning_tokens")
    per_reply = f", {tokens / replies:.0f} reasoning tokens per reply" if tokens > 0 else ""
    print(f"  reasoning: {reasoned:.0f}/{replies:.0f} fresh replies contained thinking{per_reply}")
    if reasoned == 0:
        problems.append("reasoning was requested but NO reply contained any thinking")
    elif reasoned < 0.9 * replies:
        print(f"  WARNING: only {reasoned / replies:.0%} of replies contained thinking")
    truncated, refused = stat("judge_truncated"), stat("judge_refused")
    if pd.notna(truncated):
        print(f"  budget:    {truncated:.0f} replies hit max_tokens, {refused:.0f} refused")
        if truncated > 0.05 * replies:
            problems.append(f"{truncated:.0f}/{replies:.0f} replies were cut off at max_tokens -- "
                            f"thinking is eating the budget (FOUNDRY_MAX_TOKENS)")

# Rows that parsed as nothing (truncated, refused, or malformed) fell back to distance order.
blank = detail.groupby("row")["reason"].apply(
    lambda reasons: reasons.fillna("").astype(str).str.strip().eq("").all())
print(f"  unparsed:  {int(blank.sum())}/{len(blank)} rows carry no reasons (fell back to distance order)")
if blank.mean() > 0.10:
    problems.append(f"{blank.mean():.0%} of rows returned nothing usable")

ties = stat("n_boundary_ties")
if pd.notna(ties) and ties:
    print(f"  ceiling:   {ties:.0f} row(s) tie at the shortlist boundary; top-{k} is pinned to "
          f"shortlist_ceiling rather than base_top_{k} (expected, not a leak)")
cost = stat("judge_cost_usd")
print(f"  cost:      ~${cost:.2f} this run (estimated from tokens)" if pd.notna(cost) and cost
      else "  cost:      nothing billed")
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
echo "shortlist_ceiling. Compare with the jina arm at ${OUT_DIR%/llm}/jina. Read the reasons above."
