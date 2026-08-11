"""Shared plumbing for the utility metrics that run a model locally rather than calling an API.

:mod:`.prompt_judge` sends conversations to a hosted judge. The metrics here --
:mod:`.semantic` and :mod:`.fluency` -- instead load a checkpoint and score on the machine they
run on, which changes what the surrounding code has to care about:

* **Cost becomes GPU time, not dollars.** A judge run is bounded by a bill; these are bounded by
  a SLURM allocation. Caching still pays (a re-run of a 172k-document corpus is hours), but a
  cache miss is recoverable rather than re-billed, so ``logic_classes()`` staying empty matters
  less here than it does for the judge.
* **Checkpoints are machine-specific paths.** Both metrics resolve their models through the same
  ``$ENV_VAR`` → ``models.toml`` → hub-repo-id ladder the vLLM defenses use
  (:func:`~prompt_anonymity.defenses._backends.model_path`), so a cluster mirror is configured in
  one place and a fresh clone still works by downloading.
* **They are multilingual by requirement, not by preference.** The corpora are not English-only --
  swe-chat carries Chinese conversations, and StyleRemix has been observed *translating* a turn
  into English. An English-only scorer would read that as catastrophic content loss when the
  conversation-level judge correctly calls it faithful, so every default checkpoint below is a
  multilingual one and swapping in an English model is a real methodological choice.

``torch`` and ``transformers`` are imported lazily, inside the functions that need them, so that
importing :mod:`prompt_anonymity.evaluation.utility` -- which ``experiments/eval_utility.py`` does
unconditionally -- never drags in a deep-learning stack for a run that only wants the API judge.
"""

from __future__ import annotations

import os


def select_device(prefer: str | None = None) -> str:
    """``"cuda"`` when a GPU is visible, else ``"cpu"``; ``prefer`` overrides both.

    Printed rather than silent, because the difference is not a detail: these scorers are minutes
    on a GPU and hours on CPU, and a job that quietly fell back is one that looks hung.
    """
    import torch

    if prefer:
        return prefer
    if torch.cuda.is_available():
        return "cuda"
    print("note: no GPU visible -- scoring on CPU, which is far slower. "
          "Submit through SLURM (scripts/eval_utility_slurm.sh) for a GPU.")
    return "cpu"


def torch_dtype(device: str):
    """Half precision on GPU, float32 on CPU.

    Scoring is inference-only and the numbers here are read to two or three decimals, so fp16 on
    the GPU costs nothing that matters and roughly halves both memory and time. CPU fp16 is
    emulated and slower than fp32, hence the split.
    """
    import torch

    return torch.float16 if device == "cuda" else torch.float32


def resolve_checkpoint(name: str, env_var: str, default: str) -> str:
    """The checkpoint to load: ``$env_var``, else ``models.toml``'s ``[name].model``, else
    ``default`` -- then followed into its snapshot if it is a hub cache directory.

    Unlike the defenses' :func:`~prompt_anonymity.defenses._backends.model_path`, a missing
    ``models.toml`` entry is not fatal here: these metrics ship a working public repo id as
    ``default``, so an unconfigured machine downloads rather than exiting. Configure a local
    mirror when the download is the problem, not to make the metric run at all.
    """
    from ...data.config import _load_dotenv_once
    from ...defenses._backends import resolve_model_path, _load_models_config

    # The machine-local path lives in the gitignored `.env`, so read that before the environment:
    # otherwise the override is only seen when something else in the process happened to load it
    # first, and the metric silently downloads a second copy of a checkpoint already on disk.
    _load_dotenv_once()
    override = os.environ.get(env_var)
    if override:
        return resolve_model_path(override)
    section = _load_models_config().get(name) or {}
    return resolve_model_path(section.get("model") or default)


def batched(items: list, size: int):
    """Yield ``items`` in lists of at most ``size``. Batch size is a memory knob: these models are
    quadratic in sequence length, and the corpora contain some very long conversations."""
    for start in range(0, len(items), size):
        yield items[start:start + size]
