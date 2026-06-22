# Can Prompt Anonymity Really Hide Your Identity?

## Installation

- Clone this repo and cd into it.
- Set up a new virtual environment and activate it (we used Python 3.12). E.g., `conda create -p ./env python=3.12`.
- Install Stylometrix (follow these steps precisely to avoid issues):
  - Install spacy with GPU support: https://spacy.io/usage. E.g.: `pip install -U 'spacy[cuda12x]'` for CUDA (make sure to choose the right CUDA version. If you need version 13 instead of 12, omit the `[cuda12x]` option, run `pip install cupy-cuda13x[ctk]` afterwards).
  - After spacy is installed, run `python -m spacy download en_core_web_trf` to download the large English model.
  - Clone the StyloMetrix repo at https://github.com/NASK-NLP/StyloMetrix (do not run `pip install stylometrix` because its spacy requirement is broken).
  - Modify the repo's requirements.txt file by removing the version pin for spacy (e.g., remove the ==3.7.2)
  - Modify the repo's setup.cfg file by replacing {{VERSION_PLACEHOLDER}} with 1.0.0
  - Now, run `pip install -e .` in the repo
- Next, install the following: `pip install datasets google-genai huggingface_hub matplotlib python-dotenv seaborn tqdm`

## WildChat Data Preparation (Optional) and Analysis

- Download WildChat-4.8M dataset, e.g.,: `hf download allenai/WildChat-4.8M --repo-type dataset --local-dir /datasets/ai/` (might need to run `hf auth login` first)
- Now, cd into `wildchat/` and run the following scripts in order (it will take a while):
  - `preprocess.py`: (Optional) This will create 320 files in the `wildchat_preprocessed/`.
  - `filter.py`: (Optional) This will filter the dataset into a `wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv`
  - `get_embeddings.py`: (Optional) If you want Gemini embedding, you will need to set up a credential file for Google Cloud. This script will try to get Gemini embedding for the **ENTIRE** unfiltered WildChat-4.8M dataset.  - 
  - `stylometrix.py`: (Optional) Compute Stylometrix features for the filtered dataset `wildchat_filtered_4o20240806_41mini20250414_device_deduped.csv` into `wildchat_embeddings/wildchat_filtered_en_2048_stylometrix.csv`. Check the script to see how you can change the language model.
  - `analyze_wildchat.ipynb`: Notebook for plotting some stats and running the linkage attack using the generated data.
