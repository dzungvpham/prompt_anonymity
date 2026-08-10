"""Small shared helpers used by the source adapters (model ownership, language names)."""

from __future__ import annotations

from collections import Counter


def model_owner(model: str | None) -> str:
    """Map a model id to the organization that serves it (the data's true observer).

    All of a vendor's variants collapse to one owner: ``claude-*`` -> ``Anthropic``,
    ``gpt-*``/``codex`` -> ``OpenAI``, ``gemini-*`` -> ``Google``, ``glm-*`` -> ``Zhipu``.
    Anonymized codenames, ``<synthetic>``, ``None``, or missing -> ``"unknown"``.
    (Adapted from the original ``swe-chat/preprocess.py``; WildChat's two gpt models both
    resolve to ``OpenAI``.)
    """
    if not isinstance(model, str) or not model or model.lower() in {"none", "nan", "<no_model>"}:
        return "unknown"
    m = model.lower()
    if m.startswith("claude"):
        return "Anthropic"
    if m.startswith("gpt") or "codex" in m:
        return "OpenAI"
    if m.startswith("gemini"):
        return "Google"
    if m.startswith("glm"):
        return "Zhipu"
    return "unknown"


def normalize_language(name: str | None) -> str | None:
    """Normalize a detected-language name to Title case (``"ENGLISH"`` -> ``"English"``).

    WildChat emits Title-case names and SWE-chat emits upper-case ones; Title-casing both
    yields a single consistent vocabulary. Returns ``None`` for empty/missing input.
    """
    if not name or not isinstance(name, str) or name.strip().lower() in {"", "nan", "none"}:
        return None
    return name.strip().title()


def ordered_languages(series) -> list[str]:
    """Detected languages for a document with the **primary (most frequent) language first**.

    Both SWE-chat and ShareChat label language *per turn*, so a document's label is a vote over
    its turns: ties on frequency are broken by first appearance, and the remaining languages
    follow, sorted. Returns an empty list when no turn carried a language, which is what
    :func:`~prompt_anonymity.data.language_detection.resolve_document_languages` reads as
    "upstream has no label here".
    """
    values = [x for x in series if isinstance(x, str) and x]
    if not values:
        return []
    primary = Counter(values).most_common(1)[0][0]
    return [primary] + sorted(set(values) - {primary})
