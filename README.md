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
accept anything registered, and adding one makes it selectable with no change to the scripts.

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
