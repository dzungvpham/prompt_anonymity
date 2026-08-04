"""Download the published dataset repo from HuggingFace into ``data/hf/``.

Mirrors the whole HuggingFace dataset repo (every file, current ``main`` revision) into a local
folder as real files -- no symlinks into the HF cache -- so the parquet shards can be read
directly, e.g. ``pd.read_parquet("data/hf/swe_chat.parquet")``.

Re-running *overwrites the destination with the latest revision*: changed files are re-fetched,
and files that no longer exist upstream are deleted, so the folder always equals the Hub state
rather than accumulating stale artifacts from older revisions. Unchanged files are skipped via
the download metadata HuggingFace keeps in ``<out>/.cache/``, so a refresh only pays for what
actually moved; ``--force`` throws that away and re-downloads everything from scratch.

Usage (from the repo root)::

    python -m prompt_anonymity.data.download                       # -> data/hf/, mirroring main
    python -m prompt_anonymity.data.download --force               # ignore local copies, re-download all
    python -m prompt_anonymity.data.download --include '*.parquet' # only the data files (no pruning)
    python -m prompt_anonymity.data.download --repo-id other/repo --out /tmp/hf
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from .config import hf_dir

# Default dataset repo. The destination is the project's ``data/hf/`` -- resolved at call time
# (:func:`prompt_anonymity.data.config.data_dir`) rather than from this file's location, since
# the code lives in the installed package and the outputs live in the working copy.
REPO_ID = "pavidu/PromptAnonBench"

# HuggingFace's own bookkeeping inside ``local_dir`` (download metadata + locks). It is not repo
# content, so it is never a pruning candidate and is hidden from the printed file listing.
HF_METADATA_DIR = ".cache"


def prune_stale_files(
    out_dir: Path,
    repo_id: str,
    revision: str | None = None,
    token: str | None = None,
) -> list[Path]:
    """Delete files under ``out_dir`` that the upstream repo no longer contains; return them.

    :func:`snapshot_download` only ever *adds* files, so a file renamed or dropped upstream would
    otherwise linger locally and silently shadow the real dataset (an old ``wildchat.parquet``
    next to its replacement is indistinguishable from a current one on disk). Listing the repo
    and removing everything else makes the folder an exact mirror. HuggingFace's own
    :data:`HF_METADATA_DIR` bookkeeping is left alone, and empty directories are cleaned up after.
    """
    tracked = set(HfApi(token=token).list_repo_files(repo_id, repo_type="dataset", revision=revision))
    removed = []
    for path in sorted(out_dir.rglob("*")):
        if not path.is_file() or HF_METADATA_DIR in path.relative_to(out_dir).parts:
            continue
        if path.relative_to(out_dir).as_posix() not in tracked:
            path.unlink()
            removed.append(path)

    # Directories emptied by the deletions above (deepest first, so parents collapse too).
    for directory in sorted(out_dir.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir() and HF_METADATA_DIR not in directory.relative_to(out_dir).parts:
            if not any(directory.iterdir()):
                directory.rmdir()
    return removed


def download(
    repo_id: str = REPO_ID,
    out_dir: str | Path | None = None,
    revision: str | None = None,
    include: list[str] | None = None,
    token: str | None = None,
    force: bool = False,
) -> tuple[Path, list[Path]]:
    """Mirror ``repo_id`` (a HuggingFace *dataset* repo) into ``out_dir``; overwrites what is there.

    Returns ``(path, removed)`` -- the destination, and the stale files pruned from it.

    ``revision`` pins a branch/tag/commit (default: the repo's default branch). ``include`` is an
    optional list of glob patterns (e.g. ``["*.parquet"]``) limiting which files are fetched.
    ``token`` overrides the cached/``HF_TOKEN`` credential.

    Refreshing is cheap: a file whose upstream hash matches the local download metadata is left
    alone, everything else is re-fetched and overwritten. ``force=True`` deletes the destination
    first, so every file is downloaded again -- the way to recover from a locally edited or
    truncated copy, which an incremental refresh would keep.

    Pruning is skipped when ``include`` is set: a filtered download is a partial copy by design,
    so the unmatched files already on disk are not stale, they are simply not being refreshed.

    ``out_dir`` defaults to the project's ``data/hf/``.
    """
    out_dir = Path(out_dir) if out_dir else hf_dir()
    if force and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    path = Path(snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=out_dir,
        allow_patterns=include,
        token=token,
        force_download=force,
    ))
    removed = [] if include else prune_stale_files(path, repo_id, revision, token)
    return path, removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo-id", default=REPO_ID, help=f"HF dataset repo (default: {REPO_ID})")
    parser.add_argument("--out", default=None, help="destination folder (default: the project's data/hf/)")
    parser.add_argument("--revision", default=None, help="branch, tag, or commit to download")
    parser.add_argument(
        "--include", nargs="+", default=None, metavar="GLOB",
        help="only download files matching these glob patterns (default: everything)",
    )
    parser.add_argument(
        "--token", default=None,
        help="HF access token; defaults to HF_TOKEN or the token cached by `hf auth login`",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="wipe the destination and re-download every file instead of reusing local copies",
    )
    args = parser.parse_args()

    path, removed = download(
        args.repo_id, args.out, args.revision, args.include, args.token, args.force,
    )
    for stale in removed:
        print(f"Removed (no longer in the repo): {stale.relative_to(path)}")

    files = sorted(
        p for p in path.rglob("*")
        if p.is_file() and HF_METADATA_DIR not in p.relative_to(path).parts
    )
    print(f"Downloaded {len(files)} file(s) from {args.repo_id} to {path}:")
    for f in files:
        print(f"  {f.relative_to(path)}  ({f.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
