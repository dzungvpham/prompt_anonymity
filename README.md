# Can Prompt Anonymity Really Hide Your Identity?

Research code for our paper **Can Prompt Anonymity Really Hide Your Identity?**

## Quickstart

After `./install.sh` (below), these are the commands that matter. Everything reads and writes the
`data/` folder, which holds **outputs only** — never code.

```bash
# 1. Download prepared datasets to data/hf
python -m prompt_anonymity.data.download

# 2. Run a specific experiment
python experiments/run_experiment.py --source swe_chat --feature gemini_embedding_2 --attacks nearest_neighbor

# 2b. Run all eligible experiments (skipping the ones already on disk)
python experiments/run_all_experiments.py

# 2c. ...or submit one SLURM job per experiment, GPU only where it helps.
#     --dry-run prints the sbatch command lines without submitting anything.
python experiments/run_all_experiments.py --slurm --dry-run

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
python -m prompt_anonymity.data.compute_features --source swe_chat --feature function_words
python -m prompt_anonymity.data.compute_features --source swe_chat --feature gemini_embedding_2

# 4. Defend the prompts, then attack the defended version
python -m prompt_anonymity.data.apply_defenses   --source swe_chat --defense openanonymity
python -m prompt_anonymity.data.compute_features --source swe_chat --defense openanonymity \
    --feature gemini_embedding_2
python experiments/run_experiment.py --source swe_chat --defense openanonymity \
    --feature gemini_embedding_2

# 5. Draw every figure from the CSVs the runs left behind (run it any time; --window to re-cut)
python experiments/plot_results.py
```

Every stage is a name→implementation **registry**, so `--feature`, `--defense` and `--attacks`
accept anything registered, and adding one makes it selectable with no change to the scripts.

## Installation

- Clone this repo and `cd` into it.
- Set up a new virtual environment and activate it (we used Python 3.12). E.g., `conda create -p ./env python=3.12` and `conda activate ./env`
- Run the install script: `./install.sh`

This single command installs the `prompt_anonymity` package together with all of its Python dependencies.

The script auto-detects your accelerator (it decides only whether the GPU extras below are installed) and falls back to CUDA 12 when no GPU is visible. Override it when needed:

- NVIDIA CUDA 12 (default): `./install.sh --accelerator cuda12x`
- NVIDIA CUDA 13: `./install.sh --accelerator cuda13x`
- Apple M-series: `./install.sh --accelerator apple`
- CPU only: `./install.sh --accelerator cpu`

On `cuda12x`/`cuda13x` the script also installs the extras the model-backed defenses need: `styleremix` (`torch`, `transformers`, `peft`, `accelerate`, `bitsandbytes`) and `qwen` (`vllm`, `llama-cpp-python`), installed as separate `pip` calls so a `vllm` build failure can't take `peft` down with it — `vllm` sometimes builds from source instead of using a prebuilt wheel, which needs `CUDA_HOME` pointed at a full CUDA toolkit (not just a driver). Both are skipped on `apple`/`cpu` because `vllm` needs Linux+CUDA. **Attacks and featurization need none of this**; only the defenses do.

<details>
<summary>What <code>install.sh</code> does, as manual steps</summary>

- Install the `prompt_anonymity` package and its dependencies (`datasets`, `huggingface_hub`, `matplotlib`, `python-dotenv`, `seaborn`, `tqdm`, etc.) with `pip install -e .` from the repo root.

</details>
