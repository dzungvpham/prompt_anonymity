"""Example / template defenses.

:class:`ExampleTextNormalizationDefense` is a runnable, self-contained demonstration of the
caching machinery (no external models needed). :class:`RoundTripTranslationDefense` is the
template a real text defense follows; it is not registered because it needs a translation
model supplied by the caller.

Both only rewrite text -- features are computed afterwards by
:mod:`prompt_anonymity.features`, so a defense never needs to know how conversations are
featurized.
"""

from __future__ import annotations

import re
import string

from .base import CachedTextRewriteDefense

_PUNCTUATION = set(string.punctuation)


class ExampleTextNormalizationDefense(CachedTextRewriteDefense):
    """Runnable example: a cheap text rewrite (lowercase, strip punctuation, collapse
    whitespace) standing in for an expensive text defense such as translation. It exists to
    exercise the caching end-to-end without external models.

    A production defense subclasses :class:`CachedTextRewriteDefense` and implements
    :meth:`rewrite_text` with the costly op (a translation / LLM call); the featurize stage
    then re-derives features from the rewritten text.
    """

    name = "example_normalization"
    version = "1"

    def rewrite_text(self, text: str) -> str:
        stripped = "".join(char for char in (text or "").lower() if char not in _PUNCTUATION)
        return re.sub(r"\s+", " ", stripped).strip()


class RoundTripTranslationDefense(CachedTextRewriteDefense):
    """Template for a real round-trip-translation defense: translate each unknown prompt to a
    pivot language and back to wash out stylistic signal. The featurize stage re-derives
    features from the translated text, so this defense only supplies the rewrite.

    Not registered in ``DEFENSES`` because it needs a model supplied by the caller: pass a
    ``translate(text, pivot_language) -> text`` callable. The expensive translation is cached
    per prompt. Bump :attr:`version` whenever the translation model changes (source hashing
    cannot see an external model swap), so the cache invalidates.

    Example
    -------
    >>> defense = RoundTripTranslationDefense(pivot_language="fr", translate=my_translate)
    >>> defended = defense(data, cache_dir="experiments/.cache")
    """

    name = "round_trip_translation"
    version = "1"

    def __init__(self, *, pivot_language: str = "fr", translate):
        self.pivot_language = pivot_language
        self._translate = translate

    def params(self) -> dict:
        return {"pivot_language": self.pivot_language}

    def rewrite_text(self, text: str) -> str:
        return self._translate(text, self.pivot_language)
