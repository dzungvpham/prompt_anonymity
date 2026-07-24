# Can Prompt Anonymity Really Hide Your Identity?

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

On `cuda12x`/`cuda13x` the script also installs two extras needed by the GPU model-backed defenses: `styleremix` (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`, for `styleremix.py`) and `qwen` (`vllm`, `llama-cpp-python`, for `qwen_rewrite.py`), installed as separate `pip` calls so a `vllm` build failure can't take `peft` down with it — `vllm` sometimes builds from source instead of using a prebuilt wheel, which needs `CUDA_HOME` pointed at a full CUDA toolkit (not just a driver). Both are skipped on `apple`/`cpu` because `vllm` needs Linux+CUDA; install `styleremix` manually there with `pip install -e ".[styleremix]"` if you need it. The default `--defense none` pass-through needs none of this.

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

## Running re-identification experiments

`experiments/run_experiment.py` is the main entry point. The filtered WildChat conversations and their StyloMetrix feature vectors are included in the repo, so after `./install.sh` the WildChat attack runs with no extra data prep (SWE-chat reads the CSVs under `swe-chat/`):

```bash
# WildChat (English), nearest-neighbour attack, no defense — the defaults
python experiments/run_experiment.py --dataset wildchat

# WildChat, Russian subset
python experiments/run_experiment.py --dataset wildchat --language Russian

# SWE-chat, restricted to one agent provider (default: Anthropic)
python experiments/run_experiment.py --dataset swe-chat --model-owner Anthropic

# SWE-chat across every provider
python experiments/run_experiment.py --dataset swe-chat --model-owner all

# Combine featurizers: concatenate StyloMetrix with function-word frequencies
python experiments/run_experiment.py --dataset wildchat --feature stylometrix function_words
```

The default run (`--defense none`) reuses the committed StyloMetrix vectors and is a pure, **GPU-free** pass-through that reproduces the published numbers exactly. StyloMetrix is invoked only when a defense rewrites the prompt text (the rewritten conversations are re-featurized); that step uses the GPU when one is available and otherwise falls back to CPU, which can be very slow.

Each run prints a headline table to the console and writes four files to `experiments/results/<tag>/` (override with `--output-dir`):

- `headline_results.csv` — top-k accuracy on the full candidate pool. Columns: `top`, `n_identities`, `conv_acc` (conversation-level), `id_acc` (identity-level), `random_id` (chance baseline), `advantage` (`id_acc − random_id`).
- `sweep_results.csv` — top-k identity accuracy vs. candidate-pool size. Columns: `n`, `top`, `id_acc`, `random_id`.
- `topk_accuracy.pdf`, `poolsize_sweep_top<k>.pdf` — the corresponding plots.

`advantage` is the adversary's edge over random guessing; anything above zero means the prompts are not fully anonymous.

Common options (run with `--help` for the full list):

| Option | Default | Meaning |
| --- | --- | --- |
| `--dataset` | (required) | `wildchat` or `swe-chat`. |
| `--attack` | `nearest_neighbor` | Attack to run (any key in the `ATTACKS` registry). |
| `--metric` | `cosine` | Distance metric the attack uses to compare vectors (`cosine`, `euclidean`, or any scipy `cdist` metric). Decoupled from the featurizer. |
| `--defense` | `none` | Defense applied before the attack (`none`, `example_normalization`, …). |
| `--feature` | `stylometrix` | Conversation representation(s). Name several (e.g. `--feature stylometrix function_words`) to concatenate their feature vectors into one. Gemini embeddings are planned. |
| `--language` | `English` | WildChat language subset (`English` / `Russian`). |
| `--model-owner` | `Anthropic` | SWE-chat: restrict to one agent provider, or `all`. |
| `--top-ks` | `1 5 10` | k values for the headline table. |
| `--sweep-top-k` | `1` | k used for the pool-size sweep. |
| `--pool-step` / `--n-sims` / `--seed` | `25` / `100` / `47` | Pool-size sweep granularity, repeats, and RNG seed. |
| `--cache-dir` | `experiments/.cache` | On-disk cache for defense rewrites and recomputed features. |

`--attack` and `--defense` are read straight from the package registries, so registering a new attack or defense (see below) makes it selectable here with no change to the script.

## How it works: the `prompt_anonymity` package

`run_experiment.py` is a thin driver over the installable `prompt_anonymity` package in `src/`. One experiment is a fixed pipeline, regardless of dataset, attack, or defense:

```
load_dataset → apply_defense → apply_featurizer → run_attack
            → LinkageRanking → headline_accuracy + pool_size_sweep → CSVs + plots
```

The object handed between stages is `AttackData` (`core.py`): the known (labeled) and unknown (anonymous) conversations, with one feature vector and one true-author label per conversation on each side, plus the distance metric the attack uses (a `--metric` choice, default `cosine` — not tied to the featurizer).

Each stage is its own subpackage, and most expose a name→implementation registry so the pieces are pluggable:

- **`data/`** — dataset loaders and split rules. `load_dataset(name, dir, **opts)` returns an `AttackData`. WildChat splits *by model* (one model labeled, another anonymous); SWE-chat splits *by time* (each user's last session is the unknown). Registry: `DATASET_LOADERS`.
- **`defenses/`** — optional transforms that rewrite the conversation *text* to resist linkage, run before the attack. `none` is the baseline; costly rewrites (e.g. round-trip translation) use a disk-cached `CachedDefense`. Registry: `DEFENSES`; entry point `apply_defense`.
- **`features/`** — turn the (possibly defended) text into attack-ready vectors. `StyloMetrixFeaturizer`, `FunctionWordFeaturizer`, `CharacterStatisticsFeaturizer`. Runs after the defense, and reuses the committed vectors wherever the text is unchanged — so a no-defense run never invokes StyloMetrix (and needs no GPU); GPU is used for any recompute when available, else CPU (much slower). `apply_featurizer` also accepts several featurizers at once and concatenates their vectors column-wise, each one reusing its own cache. Registry: `FEATURIZERS`; entry point `apply_featurizer`.
- **`attacks/`** — score each unknown conversation against all known ones, producing an `[n_unknown × n_known]` distance matrix. `nearest_neighbor`. Registry: `ATTACKS`; entry point `run_attack`.
- **`evaluation/`** — `LinkageRanking` sorts each unknown's candidates once and then scores any sub-pool cheaply; `headline_accuracy` and `pool_size_sweep` build the result tables.
- **`metrics/`** — stateless `top_k_accuracy` and the `random_guessing_accuracy` chance baseline.
- **`viz/`** — Matplotlib/Seaborn plot helpers for the result tables.

To add a dataset, attack, defense, or featurizer, implement the small interface documented in that subpackage's `__init__.py` and add it to the registry; `run_experiment.py` picks it up automatically.

## (Optional) Regenerating the WildChat data

The filtered conversation CSV and StyloMetrix feature CSVs are already included, so the experiments above run without this. This section only rebuilds the WildChat data from scratch; `analyze_wildchat.ipynb` remains for exploratory stats and clustering, but `run_experiment.py` is the canonical way to run the linkage attack.

- Download WildChat-4.8M dataset, e.g.,: `hf download allenai/WildChat-4.8M --repo-type dataset --local-dir /datasets/ai/` (might need to run `hf auth login` first)
- Now, cd into `wildchat/` and run the following scripts in order (it will take a while):
  - `preprocess.py`: (Optional) This will create 320 files in the `wildchat_preprocessed/`.
  - `filter.py`: (Optional) This will filter the dataset into a `wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv`
  - `get_embeddings.py`: (Optional) If you want Gemini embedding, you will need to set up a credential file for Google Cloud. This script will try to get Gemini embedding for the **ENTIRE** unfiltered WildChat-4.8M dataset.  - 
  - `stylometrix.py`: (Optional) Compute Stylometrix features for the filtered dataset `wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv` into `wildchat_embeddings/wildchat_filtered_en_2048_stylometrix.csv`. Check the script to see how you can change the language model.
  - `analyze_wildchat.ipynb`: Notebook for plotting some stats and running the linkage attack using the generated data.
