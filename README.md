# Can Prompt Anonymity Really Hide Your Identity?

Research code for **linkage (re-identification) attacks on large chat logs**: split each user's
conversations into a labeled "known" set and an anonymous "unknown" set, then try to re-link the
unknown conversations to the right user from prompt content alone — writing style or semantic
embeddings. Accuracy above random guessing means the prompts are not anonymous.

The repo also implements **defenses** (rewriting prompts to break that link) so an attack can be
evaluated with and without one.

## Quickstart

After `./install.sh` (below), these are the commands that matter. Everything reads and writes the
`data/` folder, which holds **outputs only** — never code.

```bash
# 1. Download prepared datasets to data/hf
python -m prompt_anonymity.data.download

# 2. Run the attack (rolling chronological windows, open candidate set)
python experiments/run_experiment_v2.py --source swe-chat --feature stylometrix --attacks nearest_neighbor
```

## Building data from scratch (Optional)

```bash
# 1. Download
hf auth login
hf download allenai/WildChat-4.8M --repo-type dataset --local-dir /datasets/ai/
hf download SALT-NLP/SWE-chat --repo-type dataset --local-dir /datasets/ai/

# 2. Filter and Preprocess
python -m prompt_anonymity.data.build_dataset            # -> data/dist/<split>.parquet
python -m prompt_anonymity.data.validate_dataset         # integrity checks on what was built

# 3. Turn documents into attack-ready vectors
python -m prompt_anonymity.data.compute_features --source swe-chat --feature stylometrix
python -m prompt_anonymity.data.compute_features --source swe-chat --feature gemini_embedding_2

# 4. Defend the prompts, then attack the defended version
python -m prompt_anonymity.data.apply_defenses   --source swe-chat --defense openanonymity
python -m prompt_anonymity.data.compute_features --source swe-chat --defense openanonymity \
    --feature stylometrix
python experiments/run_experiment_v2.py --source swe-chat --feature openanonymity_stylometrix
```

Every stage is a name→implementation **registry**, so `--feature`, `--defense` and `--attacks`
accept anything registered, and adding one makes it selectable with no change to the scripts:

| registry | names |
| --- | --- |
| `FEATURIZERS` | `stylometrix`, `function_words`, `character_statistics`, `char_ngram_tfidf`, `style_distance`, `gemini_embedding_2`, `gemini_embedding_001` |
| `ATTRIBUTION_ATTACKS` | `nearest_neighbor`, `cosine`, `wccn`, `lda`, `plda`, `logistic`, `svm`, `rlsc`, `xgboost` |
| `DEFENSES` | `none`, `openanonymity`, `styleremix`, `styleremix_openanon`, `qwen_rewrite`, `dp_mlm` (+ `dp_mlm_eps<ε>` sweep), `rtt_argos`, `example_normalization` |

## Installation

- Clone this repo and `cd` into it.
- Set up a new virtual environment and activate it (we used Python 3.12). E.g., `conda create -p ./env python=3.12` and `conda activate ./env`
- Run the install script: `./install.sh`

This single command installs the whole stack: GPU-enabled spaCy, the `en_core_web_trf` model, StyloMetrix from source (the PyPI package pins a broken spaCy version), and the `prompt_anonymity` package together with all of its Python dependencies.

The script auto-detects your accelerator and falls back to CUDA 12 when no GPU is visible. Override it when needed:

- NVIDIA CUDA 12 (default): `./install.sh --accelerator cuda12x`
- NVIDIA CUDA 13: `./install.sh --accelerator cuda13x`
- Apple M-series: `./install.sh --accelerator apple`
- CPU only: `./install.sh --accelerator cpu` (works, but computing StyloMetrix features falls back to CPU and can be very slow — a GPU is strongly recommended)

On `cuda12x`/`cuda13x` the script also installs the extras the model-backed defenses need: `styleremix` (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`) and `qwen` (`vllm`, `llama-cpp-python`), installed as separate `pip` calls so a `vllm` build failure can't take `peft` down with it — `vllm` sometimes builds from source instead of using a prebuilt wheel, which needs `CUDA_HOME` pointed at a full CUDA toolkit (not just a driver). Both are skipped on `apple`/`cpu` because `vllm` needs Linux+CUDA. **Attacks and featurization need none of this**; only the defenses do.

<details>
<summary>What <code>install.sh</code> does, as manual steps</summary>

- Install spacy with GPU support: https://spacy.io/usage. E.g.: `pip install -U 'spacy[cuda12x]'` for CUDA on x86 machines (make sure to choose the right CUDA version. If you need version 13 instead of 12, omit the `[cuda12x]` option, run `pip install cupy-cuda13x[ctk]` afterwards installing spacy) or `pip install -U 'spacy[apple]'` for Apple's M-series.
- After spacy is installed, run `python -m spacy download en_core_web_trf` to download the large English model.
- Clone the StyloMetrix repo at https://github.com/NASK-NLP/StyloMetrix (do not run `pip install stylometrix` because its spacy requirement is broken).
- Modify the repo's requirements.txt file by removing the version pin for spacy (e.g., remove the ==3.7.2)
- Modify the repo's setup.cfg file by replacing {{VERSION_PLACEHOLDER}} with 1.0.0
- Now, run `pip install -e .` in the repo. (If you installed spaCy with the `cuda12x` extra, StyloMetrix will otherwise try to upgrade numpy from 1.x to 2.x and break the CUDA 12 build — prevent it with a constraints file, e.g. `pip install -e . -c constraints.txt` where `constraints.txt` contains `numpy<2`.)
- Finally, install the `prompt_anonymity` package and its remaining dependencies (`datasets`, `huggingface_hub`, `matplotlib`, `python-dotenv`, `seaborn`, `tqdm`, etc.) with `pip install -e .` from the repo root.

</details>

## The dataset pipeline

`data/dist/` holds one parquet per split, plus everything derived from it. The naming is flat and
positional — `<split>`, then a defense and/or a featurizer:

| file | written by | contents |
| --- | --- | --- |
| `swe_chat.parquet` | `build_dataset` | documents: `doc_id`, `author_id`, `turns`, timestamps, language |
| `swe_chat_stylometrix.parquet` | `compute_features` | `doc_id`, `author_id`, one column per feature |
| `swe_chat_openanonymity.parquet` | `apply_defenses` | `doc_id`, `author_id`, `turns` rewritten |
| `swe_chat_openanonymity_stylometrix.parquet` | `compute_features --defense` | features of the defended text |

Defended splits and feature files share the `<split>_<name>.parquet` namespace, so keep defense and
featurizer names disjoint. On disk you tell them apart by their columns: a defended split has
`turns`, a feature file has feature columns.

A defended file carries **only** `doc_id`, `author_id` and the rewritten `turns` — a defense changes
nothing else, so timestamps, language and model stay in `<split>.parquet` instead of being
duplicated. Join them back on `doc_id` when you need them; `compute_features --defense` does exactly
that for its `--language` filter.

**Where the raw data comes from is configuration, not a constant.** `src/prompt_anonymity/data/datasets.toml` pins each source's HuggingFace repo and revision, and with nothing else set the build downloads them (WildChat-4.8M is gated: accept its terms and `hf auth login` first). If you already have local copies, point at them without editing the committed config:

```bash
# in .env, or exported in your shell
PROMPT_ANONYMITY_WILDCHAT_RAW=/path/to/WildChat-4.8M/data          # directory of parquet shards
PROMPT_ANONYMITY_SWE_CHAT_RAW=/path/to/SWE-chat/conversations.parquet
PROMPT_ANONYMITY_DATA_DIR=/path/for/outputs                        # default: <repo>/data
```

A `datasets.toml` of your own — in the repo root, or wherever `$PROMPT_ANONYMITY_DATASETS_CONFIG` points — overrides the packaged defaults; see the comments in that file for the schema.

Both `compute_features` and `apply_defenses` shard for SLURM job arrays (`--num-shards` /
`--shard-index`, filled in automatically from `SLURM_ARRAY_TASK_ID`; see
`scripts/compute_features_slurm.sh` and `scripts/apply_defenses_slurm.sh`) and cache every result,
so re-runs and resumed runs recompute nothing. That matters most for `--feature gemini_embedding_2`,
where each cache miss is a paid OpenRouter call (~$0.32 for all of SWE-chat; the run prints what it
spent). That featurizer embeds each document once, from its first 8,192 tokens (the model's window),
prefixed with `task: sentence similarity | query: ` — so its vectors describe a document's opening
rather than all of it. `--task clustering` (or `classification`) changes that prefix and writes a
separate file, so tasks can be compared side by side.

## Defenses

A defense rewrites the prompt text a user would release, **one user turn at a time**, so turn count
and boundaries survive and the defended document stays aligned with the original. Apply one to a
whole split with `apply_defenses`; the result is a drop-in replacement for the split it came from.

| defense | what it does | needs |
| --- | --- | --- |
| `none` | pass-through baseline | — |
| `openanonymity` | redacts identifiers behind stable placeholders (`[PERSON_1]`, `[ORG_1]`) **and** de-identifies writing style. Port of OpenAnonymity's `scrubberService.js` REDACT step, system prompt byte-identical to upstream | 80 GB GPU (gpt-oss-safeguard-120b via vLLM) |
| `styleremix` | steers text along interpretable style axes (formality, voice, length, …) with per-axis LoRA adapters merged into one, over Llama-3-8B (Fisher et al., EMNLP 2024) | GPU (~20 GB) |
| `styleremix_openanon` | both, strictly in sequence: restyle everything, release the GPU, then scrub the restyled text | 80 GB GPU |
| `qwen_rewrite` | rewrites every prompt into one fixed neutral style | GPU, or CPU via GGUF |
| `dp_mlm`, `dp_mlm_eps<ε>` | differentially-private word-level rewriting at a given ε | GPU |
| `rtt_argos` | round-trip translation | — |

**The model-backed defenses run locally**, through vLLM, against checkpoints on disk — no API keys,
no data leaving the machine. Point them elsewhere with `OPENANON_MODEL` / `STYLEREMIX_BASE_MODEL`,
which accept a checkpoint directory, a HuggingFace hub cache entry, or a repo id.

Two behaviours worth knowing:

- **Long turns are split, not truncated.** A turn too long for the model's window is cut into
  fragments, each rewritten, then rejoined — so no text is silently dropped.
- **Very short turns pass through undefended** by the style rewriters (under 16 characters). A
  contentless turn gives a rewriter nothing to work from and it invents something: measured on 7,764
  rewrites, a 1–5 character input came back more than 3× longer 11.6% of the time. Such turns are
  ~1% of the corpus text. This is deliberately **off** for `openanonymity`, where skipping short
  turns would be a privacy hole — a 13-character turn can be an email address.

Long runs are slow (SWE-chat is ~26K generations, roughly an hour on an A100) but **resumable**: the
cache checkpoints as it goes, so a preempted or killed run picks up where it stopped when
re-submitted unchanged.

## Running re-identification experiments

**`experiments/run_experiment_v2.py` is the canonical runner.** It reads the built parquets and
sweeps a grid of **rolling chronological windows**: documents are ordered by `ended_at` and cut at
fractions of that timeline, so the attacker never sees the future.

```bash
# defaults: 0.25/0.50/0.75 known fractions x 0.10/0.25/0.50 windows = 8 independent attacks
python experiments/run_experiment_v2.py --source wildchat --feature stylometrix \
    --attacks nearest_neighbor rlsc --output-dir experiments/results/wc-stylo

# semantic embeddings instead of style, one window, with hyper-parameter tuning
python experiments/run_experiment_v2.py --source swe-chat --feature gemini_embedding_2 \
    --known-fractions 0.5 --window 0.25 --attacks rlsc --tune

# open-set: score everything, with an explicit reject class
python experiments/run_experiment_v2.py --source swe-chat --feature stylometrix --ood reject
```

Two things to read carefully in the output:

- **The candidate set is open.** By default an unknown document whose author is absent from the
  known side is counted (`n_ood` / `ood_rate`) but not scored. Out-of-set rates are high and grow
  with the window — 48%–72% across WildChat's eight windows — so always read the in-set counts next
  to any accuracy.
- **Micro top-1 hides most of the story.** It is carried by a handful of prolific users: at
  50%/50%, 18 of 2,178 target users had >50% of their documents attributed while 1,973 were never
  identified once. Read the macro numbers and `selective_classification` (precision when the attack
  answers only its most confident documents) in `rolling_results.csv`.

Results land in `experiments/results/<tag>/`: `rolling_results.csv` (one row per window per attack,
all metrics), `cmc_results.csv` (document-level top-k at every k), per-window `predictions_*.csv`
and `author_report_*.csv`, plus `topk_accuracy_*.pdf`, `window_sweep_top1_*.pdf`, `cmc_curve_*.pdf`.

<details>
<summary>The older <code>run_experiment.py</code> (single fixed split, defense + fidelity pipeline)</summary>

`experiments/run_experiment.py` predates the unified dataset: it reads the older per-dataset CSVs
with `--dataset`, does one fixed known/unknown split instead of rolling windows, and is the only
path that runs the **fidelity judges** (`--fidelity`, which ask a remote model whether a defended
prompt still gets the same answer — the one part of the package that still calls a hosted API).

```bash
python experiments/run_experiment.py --dataset wildchat
python experiments/run_experiment.py --dataset swe-chat --model-owner Anthropic
python experiments/run_experiment.py --dataset wildchat --feature stylometrix function_words
```

It writes `headline_results.csv`, `sweep_results.csv` and two plots to `experiments/results/<tag>/`.
`advantage` (`id_acc − random_id`) is the adversary's edge over random guessing; anything above zero
means the prompts are not fully anonymous.

</details>

## How it works: the `prompt_anonymity` package

The runners are thin drivers over the installable package in `src/`. One experiment is a fixed
pipeline, regardless of dataset, attack, or defense:

```
load documents → (defend) → featurize → attack → rank → metrics → CSVs + plots
```

The object handed between stages in the older pipeline is `AttackData` (`core.py`): the known
(labeled) and unknown (anonymous) conversations, one feature vector and one true-author label per
conversation on each side, plus the distance metric (`--metric`, default `cosine` — an attack-level
choice, not tied to the featurizer).

- **`data/`** — two families that share a package: the **loaders** (`splits.py`, `wildchat.py`, `swe_chat.py`) that feed the attack pipeline, and the **build modules** (`build_dataset.py`, `compute_features.py`, `apply_defenses.py`, `text_cleaning.py`, `identity.py`, `dedup.py`, …) that produce the dataset. WildChat splits *by model*; SWE-chat splits *by time* (each user's last session is the unknown). Registry: `DATASET_LOADERS`.
- **`defenses/`** — text rewrites applied before the attack. Expensive ones subclass `CachedDefense` and get automatic, auto-invalidating disk caching: edit a defense's code and its cache namespace changes. Registry: `DEFENSES`.
- **`features/`** — text → vectors, cached per document by content hash. Registry: `FEATURIZERS`.
- **`attacks/`** — every attack returns an `[n_documents × n_authors]` score matrix, higher = more likely. Four families: `similarity/` (summarise each author, score the match), `multiclass/` (a decision function per author), `llm/` (shortlist cheaply, let a judge reorder), `verification/` (learned same-author scoring over pairs), plus `ood/` for accept-or-reject. Registry: `ATTRIBUTION_ATTACKS`.
- **`metrics/`** — top-k and macro/micro accuracy, CMC curves, selective classification, and open-set scoring (`detection_auroc`, `equal_error_rate`, `c_at_1`, calibration).
- **`evaluation/`**, **`viz/`** — ranking helpers and the plot functions behind the result figures.
- **`fidelity/`** — does a defended prompt still get the same answer? Remote judges, the only part that needs `OPENROUTER_API_KEY`.

Scale is the constraint that shapes the attack code: WildChat is 172,509 documents / 25,357 authors,
and jobs here run under a 16 GB memory cap. Attacks whose training cost is flat in the number of
authors (`cosine`, `wccn`, `lda`, `plda`, `rlsc`) stay usable there; `xgboost` and `svm` do not.
Distances are computed by one blocked, BLAS-backed kernel (`attacks/similarity/kernel.py`) — never
call `scipy.spatial.distance.cdist` directly, it is ~150× slower for cosine and forces float64.

To add a dataset, attack, defense, or featurizer, implement the small interface documented in that
subpackage's `__init__.py` and add it to the registry; both runners pick it up automatically.
