"""Small shared helpers used by both source adapters (model ownership, language names)."""

from __future__ import annotations


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
