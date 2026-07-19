# function words are often used in stylometry and authorship attribution, as they are less likely to be consciously controlled by the author and can reveal patterns in writing style.

from __future__ import annotations
import numpy as np
import re
from .base import Featurizer

# we can extend this list with more function words if needed
FUNCTION_WORDS = ["the","a","an","and","but","or","if","because","of","to","in",
    "on","at","with","however","although","actually","just","really","very",
    "so","then","also","that","this","these","those","which","who","what"]

# featurizer that counts the frequency of function words in a text
class FunctionWordFeaturizer(Featurizer):
    name = "function_words"
    version = "1"

    def featurize(self, texts) -> np.ndarray:
        rows = []
        for text in texts:
            words = re.findall(r"\b\w+\b", (text or "").lower())
            n = len(words) or 1
            rows.append([words.count(w) / n for w in FUNCTION_WORDS])
        return np.asarray(rows, dtype=float)