"""Where the dataset build reads its raw sources and writes its outputs.

Locations come from a small TOML file plus environment overrides, with downloading the raw source
from HuggingFace as the fallback, so a fresh clone can rebuild the dataset without editing
anything.

Two things are configured:

* :func:`raw_path` -- where one source's *raw* upstream data is (WildChat's parquet directory,
  SWE-chat's ``conversations.parquet``). Resolution order, first hit wins:

  1. an explicit argument (typically a ``--wildchat-raw`` / ``--swe-raw`` CLI flag);
  2. ``$PROMPT_ANONYMITY_<SOURCE>_RAW`` (``PROMPT_ANONYMITY_SWE_CHAT_RAW``), read from the
     process environment or a ``.env`` -- the place for machine-specific paths, since ``.env``
     is not committed;
  3. ``path`` for that source in the config file;
  4. a HuggingFace download of ``hf_repo`` at the pinned ``hf_revision``, cached by
     ``huggingface_hub`` in the usual place (``$HF_HOME``).

* :func:`data_dir` -- the project's ``data/`` folder, where everything the build *produces*
  goes: ``dist/`` parquets, the featurizer cache, the published-dataset mirror. It is a plain
  directory next to the code rather than part of the package, so the package stays importable
  from anywhere while the outputs stay in the working copy. ``$PROMPT_ANONYMITY_DATA_DIR``
  overrides it; otherwise it is ``data_dir`` from the config file (default ``data``), resolved
  against :func:`project_root`.

The config file itself is found in this order: ``$PROMPT_ANONYMITY_DATASETS_CONFIG``, then a
``datasets.toml`` in the working directory or any parent of it, then the packaged default
(:data:`PACKAGED_CONFIG`, ``prompt_anonymity/data/datasets.toml``) -- which is HuggingFace-only
and machine-independent, and is the file to read for the schema.
"""

from __future__ import annotations

import os
import tomllib
from functools import lru_cache
from pathlib import Path

#: Name of a user-supplied config file, searched for from the working directory upwards.
CONFIG_FILENAME = "datasets.toml"
#: Machine-independent defaults shipped with the package (HuggingFace repos, no local paths).
PACKAGED_CONFIG = Path(__file__).with_name(CONFIG_FILENAME)

#: Environment variable pointing at a config file directly (wins over the search).
CONFIG_PATH_ENV = "PROMPT_ANONYMITY_DATASETS_CONFIG"
#: Environment variable overriding the output directory.
DATA_DIR_ENV = "PROMPT_ANONYMITY_DATA_DIR"
#: Template for the per-source raw-path override, e.g. ``PROMPT_ANONYMITY_SWE_CHAT_RAW``.
RAW_PATH_ENV = "PROMPT_ANONYMITY_{source}_RAW"

#: Files whose presence marks the project root, for resolving relative paths in the packaged
#: config (a config file of your own resolves them against its own directory instead).
ROOT_MARKERS = ("pyproject.toml", ".git")

#: Fallback output directory name, if the config file sets none.
DEFAULT_DATA_DIR = "data"


@lru_cache(maxsize=1)
def _load_dotenv_once() -> None:
    """Merge a ``.env`` into the environment, once per process.

    ``python-dotenv`` walks up from the working directory, the same way the rest of the project
    finds its API keys, and never overwrites a variable already set in the real environment.
    """
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:  # optional at runtime: the real environment still works
        pass


def _env(name: str) -> str | None:
    """An environment variable's value, or ``None`` if unset or blank (``.env`` included)."""
    _load_dotenv_once()
    value = (os.environ.get(name) or "").strip()
    return value or None


def project_root(start: Path | None = None) -> Path:
    """The directory holding ``pyproject.toml`` / ``.git``, searching upwards from ``start``.

    Falls back to the working directory when nothing is found (an installed package run outside
    a checkout), which keeps ``data/`` predictable: it is always next to the project you are
    working in, whichever subdirectory you happen to run the command from.
    """
    start = (start or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        if any((directory / marker).exists() for marker in ROOT_MARKERS):
            return directory
    return start


def find_config_file() -> Path:
    """Locate the config file: the env override, else a ``datasets.toml`` at or above the
    working directory, else the packaged default."""
    override = _env(CONFIG_PATH_ENV)
    if override:
        path = Path(override).expanduser()
        if not path.exists():
            raise SystemExit(f"{CONFIG_PATH_ENV}={override} does not exist.")
        return path
    here = Path.cwd().resolve()
    for directory in (here, *here.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return PACKAGED_CONFIG


@lru_cache(maxsize=None)
def load_config() -> tuple[dict, Path]:
    """``(parsed config, its path)``, read once per process."""
    path = find_config_file()
    with open(path, "rb") as handle:
        return tomllib.load(handle), path


def _base_dir(config_path: Path) -> Path:
    """What a relative path in the config file is relative to.

    Its own directory for a config file of your own -- so a checked-in config keeps working
    wherever the checkout lives -- but the project root for the packaged default, whose directory
    is inside ``site-packages`` / ``src`` and is not where outputs belong.
    """
    return project_root() if config_path == PACKAGED_CONFIG else config_path.parent


def data_dir() -> Path:
    """The project's ``data/`` directory: everything the build writes goes under here.

    Not created here -- writers create the subdirectory they need.
    """
    override = _env(DATA_DIR_ENV)
    if override:
        return Path(override).expanduser()
    config, config_path = load_config()
    configured = Path((config.get("paths") or {}).get("data_dir") or DEFAULT_DATA_DIR).expanduser()
    if configured.is_absolute():
        return configured
    return (_base_dir(config_path) / configured).resolve()


def hf_dir() -> Path:
    """Where dataset parquets are READ from (``<data_dir>/hf``).

    :mod:`~prompt_anonymity.data.download` mirrors the published HF dataset here; every stage
    that reads a base split (or another stage's output) looks here by default, so a downloaded
    checkout works with no local build. Not written to by this project's own scripts -- see
    :func:`dist_dir` for where they write -- so it stays an exact mirror of what was downloaded.
    """
    return data_dir() / "hf"


def dist_dir() -> Path:
    """Where this project's own scripts WRITE their outputs (``<data_dir>/dist``): a freshly
    built split, computed features, a defended split. Kept separate from :func:`hf_dir` so running
    the pipeline locally never mutates the downloaded mirror -- read from ``hf/``, write to
    ``dist/``, and point a later stage's ``--dist-dir`` at ``dist/`` explicitly to chain onto a
    local output rather than the mirror.
    """
    return data_dir() / "dist"


def cache_dir() -> Path:
    """Where computed feature vectors are cached (``<data_dir>/.cache``, regenerable)."""
    return data_dir() / ".cache"


def source_config(source: str) -> dict:
    """The config table for one source (``wildchat``, ``swe_chat``)."""
    config, config_path = load_config()
    sources = config.get("sources") or {}
    if source not in sources:
        raise SystemExit(
            f'no [sources."{source}"] section in {config_path}; it configures '
            f"{sorted(sources) or 'nothing'}. Add one (see {PACKAGED_CONFIG}), point at the data "
            f"with ${RAW_PATH_ENV.format(source=_env_source(source))}, or pass the path explicitly."
        )
    return sources[source]


def _download_from_hub(source: str, settings: dict) -> Path:
    """Fetch one source from HuggingFace and return the local path the loaders should read.

    ``hf_file`` downloads that one file; ``hf_dir`` (or neither) downloads the repo -- or just
    that subdirectory of it -- and returns the directory. Both go through ``huggingface_hub``'s
    cache, so a second run re-uses the first run's download. A gated repo (WildChat is one) fails
    here until the terms are accepted and a token is available, so that case gets its own message
    rather than a bare traceback.
    """
    repo = settings.get("hf_repo")
    if not repo:
        raise SystemExit(
            f"nowhere to read raw {source} from: its config has no `path` and no `hf_repo`. Set "
            f"{RAW_PATH_ENV.format(source=_env_source(source))} in your .env, or add a path to "
            f"the config file."
        )
    revision = settings.get("hf_revision")
    hf_file, hf_dir = settings.get("hf_file"), settings.get("hf_dir")
    print(f"[{source}] no local copy configured; fetching {repo}"
          f"{f'@{revision[:8]}' if revision else ''} from HuggingFace "
          f"(cached, so this is a one-off download)")
    try:
        from huggingface_hub import hf_hub_download, snapshot_download

        if hf_file:
            return Path(hf_hub_download(repo, hf_file, repo_type="dataset", revision=revision))
        patterns = [f"{hf_dir.rstrip('/')}/**"] if hf_dir else None
        snapshot = Path(snapshot_download(repo, repo_type="dataset", revision=revision,
                                          allow_patterns=patterns))
        return snapshot / hf_dir if hf_dir else snapshot
    except Exception as error:  # noqa: BLE001 - gated repo, no token, no network, ...
        raise SystemExit(
            f"could not download raw {source} from HuggingFace ({repo}): {error}\n"
            f"If the repo is gated, accept its terms on the dataset page and log in "
            f"(`hf auth login`, or set HF_TOKEN). If you already have a local copy, point at it "
            f"with {RAW_PATH_ENV.format(source=_env_source(source))}=<path>."
        ) from error


def _env_source(source: str) -> str:
    """Source name as it appears in an environment variable (``swe_chat`` -> ``SWE_CHAT``)."""
    return source.upper().replace("-", "_")


def raw_path(source: str, explicit: str | Path | None = None) -> Path:
    """Local path to one source's raw upstream data, downloading it if necessary.

    Parameters
    ----------
    source : str
        ``"wildchat"`` or ``"swe_chat"`` (any key in the config's ``[sources]`` table).
    explicit : str or pathlib.Path, optional
        A caller-supplied path (a CLI flag), which wins over everything else.

    Returns the directory or file the source adapter expects (see
    :mod:`prompt_anonymity.data.sources_wildchat` / :mod:`~prompt_anonymity.data.sources_swe_chat`).
    A configured path that does not exist is an error naming the setting that produced it, rather
    than a confusing failure inside pyarrow later.
    """
    def use(value, origin: str) -> Path:
        path = Path(str(value)).expanduser()
        if not path.exists():
            raise SystemExit(f"raw {source} path from {origin} does not exist: {path}")
        return path

    if explicit:
        return use(explicit, "the command line")
    # Checked before the config is consulted, so an environment override works even for a source
    # the config file has never heard of.
    variable = RAW_PATH_ENV.format(source=_env_source(source))
    from_env = _env(variable)
    if from_env:
        return use(from_env, f"${variable}")

    settings = source_config(source)
    if settings.get("path"):
        return use(settings["path"], f"`path` in {load_config()[1]}")
    return _download_from_hub(source, settings)
