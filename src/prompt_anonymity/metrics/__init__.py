"""Metrics for scoring linkage attacks.

``top_k_accuracy`` measures how often an attack re-identifies the true author within
its top-k guesses; ``random_guessing_accuracy`` gives the corresponding chance
baseline, so the difference is the adversary's advantage over guessing.
"""

from .accuracy import random_guessing_accuracy, top_k_accuracy

__all__ = ["random_guessing_accuracy", "top_k_accuracy"]
