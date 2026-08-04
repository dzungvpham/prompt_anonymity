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
python experiments/run_experiment.py --source swe_chat --feature stylometrix --attacks nearest_neighbor

# 3. Draw the figures from every result on disk
python experiments/plot_results.py
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
python -m prompt_anonymity.data.compute_features --source swe_chat --feature stylometrix
python -m prompt_anonymity.data.compute_features --source swe_chat --feature gemini_embedding_2

# 4. Defend the prompts, then attack the defended version
python -m prompt_anonymity.data.apply_defenses   --source swe_chat --defense openanonymity
python -m prompt_anonymity.data.compute_features --source swe_chat --defense openanonymity \
    --feature stylometrix
python experiments/run_experiment.py --source swe_chat --defense openanonymity \
    --feature stylometrix

# 5. Draw every figure from the CSVs the runs left behind (run it any time; --window to re-cut)
python experiments/plot_results.py
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

**`experiments/run_experiment.py` is the canonical runner.** It reads the built parquets and
sweeps a grid of **rolling chronological windows**: documents are ordered by `ended_at` and cut at
fractions of that timeline, so the attacker never sees the future.

```bash
# defaults: 0.25/0.50/0.75 known fractions x 0.10/0.25/0.50 windows = 8 independent attacks
python experiments/run_experiment.py --source wildchat --feature stylometrix \
    --attacks nearest_neighbor

# the same attack against text a defense rewrote (the vectors must already exist)
python experiments/run_experiment.py --source wildchat --feature stylometrix \
    --defense openanonymity --attacks nearest_neighbor

# semantic embeddings instead of style, one window (hyper-parameters are tuned by default,
# per window, on that window's own known side; --no-tune uses the defaults instead)
python experiments/run_experiment.py --source swe_chat --feature gemini_embedding_2 \
    --known-fractions 0.5 --window 0.25 --attacks rlsc

# open-set: score everything, with an explicit reject class
python experiments/run_experiment.py --source swe_chat --feature stylometrix --ood reject
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

Results land in `experiments/results/<dataset>_<defense>_<feature>_<attack>/`: `rolling_results.csv`
(one row per window per attack, all metrics), `cmc_results.csv` (document-level top-k at every k),
and `predictions_*.csv` and `author_report_*.csv`, one file per known side with a `window` column
distinguishing the windows run from it. **The runner draws no figures** — see the next section.

### Figures

`experiments/plot_results.py` draws every figure in the project, from the CSVs the runs left
behind. Run it any time; `--window` re-cuts the windows without re-running an experiment:

```bash
python experiments/plot_results.py         # -> experiments/plots/<dataset>/
```

It picks up each results directory whose name is `<dataset>_<defense>_<feature>_<attack>` (an
undefended run spells its defense `base`; anything else — an ad-hoc directory, or a run carrying
extra qualifiers such as `_langaware` — is skipped, and says so). Per dataset it writes:

Four curve types, each with the same layout — `by_defense/<feature>_<attack>.pdf` holds the attack
fixed (**does the defense cost the attacker anything?**) and `by_method/<defense>.pdf` holds the
defense fixed (**which attack is strongest against it?**):

- `accuracy/by_{defense,method}/` — **CMC**: top-k accuracy against k.
- `risk_coverage/by_{defense,method}/` — **precision when the attack answers only its most
  confident documents.** An attack that is usually wrong but knows when it is right is a sharper
  threat than its headline accuracy suggests: on SWE-chat, Gemini embeddings go from 0.57 top-1 to
  **1.00 precision at 10% coverage**.
- `author_risk/by_{defense,method}/` — **each user's own accuracy, most exposed first.** Anonymity
  fails unevenly: on WildChat **89% of users are never identified once** under StyloMetrix (68%
  under Gemini embeddings), and the mean is carried by the few percent who are identified every
  time.
- `scaling/by_{defense,method}/` — **accuracy against the size of the candidate pool**: is the
  threat an artefact of a small pool? Each window is interpolated down to smaller galleries, so
  the curve is continuous rather than three measured points. Drawn only for `nearest_neighbor`
  and `cosine`, the attacks whose scores do not depend on which other authors are enrolled —
  anything that refits against the gallery would be understated by the interpolation.
- `per_run/<run>/` — the per-window detail for one run on its own (CMC per window, the window
  sweep, top-k bars).

Two figures have no `by_defense`/`by_method` split. `accuracy/macro_micro.pdf` shows every run's
top-1 counted three ways — per document, per user, per identity — which differ by up to 3× on the
same run.
`plots/cross_dataset/scaling.pdf` puts every undefended run on one log axis from 2 to 19,711
candidate users; matched there, WildChat turns out to be *more* identifiable than SWE-chat, the
opposite of what the raw headline numbers suggest. `plots/cross_dataset/` is where anything
spanning both corpora goes, since filing it under either would imply it belonged to that one.

Both comparison figures plot the CMC curve **averaged over the run's rolling windows** — one curve
per run is what makes several runs comparable on one axes — inside a shaded 95% interval for that
mean, and each is written next to a `.csv` of the exact numbers plotted. Two things to read
correctly: the average is truncated to the k values *every* window reached, so a curve stops short
of 1.0 at its right edge even though each individual window's CMC reaches 1.0 at its own pool size
(`per_run/` shows those); and the band describes how much the result moves across windows, not
sampling error, since the windows are nested slices of one corpus. Keeping the figures out of the
runners is what lets them compare runs at all: a script that plots its own output can only ever
plot one.

The single fixed-split runner that used to sit beside this one — `--dataset`, the older
per-dataset CSVs, one known/unknown split, a defense applied to the text inside the run, and the
`--fidelity` judges — was **removed on 2026-08-04**; what is now `run_experiment.py` is the file
that was `run_experiment_v2.py`. Defending is a build step of its own now (`apply_defenses` +
`compute_features --defense`, above). The utility measurement has no command-line entry point at
present: `src/prompt_anonymity/utility/` (called `fidelity/` until the same day) is still in the
package, but nothing calls it. Removed with the runner, as its only callers: the per-dataset CSV
loaders (`data/splits.py`, `data/wildchat.py`, `data/swe_chat.py`, `load_dataset`,
`DATASET_LOADERS`), `features.apply_featurizer`, and `evaluation.pool_size_sweep`.

## How it works: the `prompt_anonymity` package

The runner is a thin driver over the installable package in `src/`. One experiment is a fixed
pipeline, regardless of dataset, attack, or defense:

```
load documents → (defend) → featurize → attack → rank → metrics → CSVs
```

`AttackData` (`core.py`) is what a defense is handed: conversations on a known and an unknown
side, one feature vector and one true-author label each, plus the distance metric (`--metric`,
default `cosine` — an attack-level choice, not tied to the featurizer). `data/apply_defenses.py`
builds one per defense run; nothing else constructs it now that the loaders are gone.

- **`data/`** — the build pipeline: `build_dataset.py`, `compute_features.py`, `apply_defenses.py`, `validate_dataset.py`, `download.py`, and the shared stages (`text_cleaning.py`, `identity.py`, `dedup.py`, `language_detection.py`, `sources_*.py`). Each is a `python -m` entry point or a stage of one; the package exports nothing, so importing it is free. The per-dataset loaders that used to live here went with the fixed-split runner — the known/unknown split is a property of the experiment now, not of a loader.
- **`defenses/`** — text rewrites applied before the attack. Expensive ones subclass `CachedDefense` and get automatic, auto-invalidating disk caching: edit a defense's code and its cache namespace changes. Registry: `DEFENSES`.
- **`features/`** — text → vectors, cached per document by content hash. Registry: `FEATURIZERS`.
- **`attacks/`** — every attack returns an `[n_documents × n_authors]` score matrix, higher = more likely. Four families: `similarity/` (summarise each author, score the match), `multiclass/` (a decision function per author), `llm/` (shortlist cheaply, let a judge reorder), `verification/` (learned same-author scoring over pairs), plus `ood/` for accept-or-reject. Registry: `ATTRIBUTION_ATTACKS`.
- **`metrics/`** — top-k and macro/micro accuracy, CMC curves, selective classification, and open-set scoring (`detection_auroc`, `equal_error_rate`, `c_at_1`, calibration).
- **`evaluation/`** — ranking helpers: rank once, reuse it for the headline table.
  There is deliberately no plotting module in the package; figures live in
  `experiments/plot_results.py`.
- **`utility/`** — does a defended prompt still get the same answer? Two remote judges (`answer`, per turn; `conversation`, whole), the only part that needs `OPENROUTER_API_KEY`. **Currently has no caller** — see the note at the top of the package.

Scale is the constraint that shapes the attack code: WildChat is 172,509 documents / 25,357 authors,
and jobs here run under a 16 GB memory cap. Attacks whose training cost is flat in the number of
authors (`cosine`, `wccn`, `lda`, `plda`, `rlsc`) stay usable there; `xgboost` and `svm` do not.
Distances are computed by one blocked, BLAS-backed kernel (`attacks/similarity/kernel.py`) — never
call `scipy.spatial.distance.cdist` directly, it is ~150× slower for cosine and forces float64.

To add a dataset, attack, defense, or featurizer, implement the small interface documented in that
subpackage's `__init__.py` and add it to the registry; both runners pick it up automatically.
