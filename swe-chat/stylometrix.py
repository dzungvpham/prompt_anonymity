# Compute StyloMetrix features for the scrubbed SWE-chat sessions produced by preprocess.py.
# Mirrors wildchat/stylometrix.py: read the cached session table, compute stylometric features
# over each session's (scrubbed) text truncated to MAX_LEN, and save them keyed by session_id
# so analyze_swe_chat.ipynb can align features to sessions by merge.
import pandas as pd
import spacy
import stylo_metrix as sm

MAX_LEN = 2048
langcode = "en"
INPUT_PATH = "swe_chat_sessions.csv"
OUTPUT_PATH = f"swe_chat_stylometrix_{langcode}_{MAX_LEN}.csv"

assert spacy.require_gpu(), "Spacy cannot use GPU!"

sessions = pd.read_csv(INPUT_PATH)
sessions["content"] = sessions["content"].fillna("").astype(str)
# A handful of sessions can be empty after scrubbing; give StyloMetrix a harmless token
# so transform never sees an empty document (their features end up ~0 and are ignored).
texts = [(c[:MAX_LEN] if c.strip() else "n a") for c in sessions["content"].to_list()]

stylo = sm.StyloMetrix(langcode)
style_embeddings = stylo.transform(texts).drop(columns="text").reset_index(drop=True)
style_embeddings.insert(0, "session_id", sessions["session_id"].to_list())
style_embeddings.to_csv(OUTPUT_PATH, index=False)
print(f"Saved {style_embeddings.shape[0]} session feature vectors to {OUTPUT_PATH}", flush=True)
