r"""Export a defended split to a plain CSV a human can read -- one row per turn, original beside
defended.

``apply_defenses`` writes parquet (``data/dist/<split>_<defense>.parquet``), which is the right
format for the pipeline and the wrong one for *reading the text*: a list-of-strings column in a
columnar file is not something you open and eyeball. This script is the bridge:

    python experiments/export_defended_csv.py --source wildchat --defense dp_mlm_eps100
    # -> data/dist/csv/wildchat_dp_mlm_eps100.csv

EVERY turn of every defended document lands in the file -- this is an export, not a sample.

What a row is
-------------

One **turn** by default, which is the unit the defense actually rewrote (see
``apply_defenses``: each user turn is defended on its own and the turn count is preserved), so an
original and its rewrite sit side by side in one row and a diff is a glance rather than a join:

    doc_id, author_id, turn_index, n_turns, original_text, defended_text

``--level document`` instead writes one row per document with the turns joined by a blank line,
which is what you want when reading the defended text as prose rather than auditing per-turn edits.
``--no-original`` drops the original column (and stops reading the undefended split at all), for
when the file is going somewhere the source text should not.

Memory
------

Streamed, a parquet batch at a time, so this script stays runnable on a login node even on a large
corpus. The undefended split is walked *alongside* the defended one rather than loaded into a
lookup dict, which works because both files carry the split's row order (see ``merge_shards``) and
a defended file is a subset of the split in that same order. Documents the defended file skips --
a ``--language`` or ``--limit`` run -- are passed over and dropped, so memory stays flat.

If the two files are *not* in a compatible order the forward scan runs off the end of the split and
this exits with an error naming the document, rather than emitting a file whose "original" column
belongs to some other conversation. ``--no-original`` always works.

Readability
-----------

Text is written with full quoting, so the embedded newlines in a prompt survive and
``pd.read_csv`` / Excel / LibreOffice read the file back exactly. Two knobs for when that is not
what you want:

* ``--newlines escape`` writes ``\n`` as a two-character escape, making every record one physical
  line -- the form to use for ``grep``, ``head`` and ``wc -l``.
* ``--encoding utf-8-sig`` prepends the byte-order mark Excel wants before it will treat a UTF-8
  CSV as UTF-8. Everything else should stay on the ``utf-8`` default.

Other uses
----------

``--defended PATH`` exports any single file rather than the assembled split -- point it at one
array task's shard (``data/dist/shards/<split>_<defense>.0003-of-0064.parquet``) to read that
shard's text before the rest of the array has finished. ``--out`` names the output; the default is
``<dist>/csv/<stem>.csv``, keeping bulky text exports out of the directory the pipeline globs for
parquets.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import pyarrow.parquet as pq

from prompt_anonymity.data.config import dist_dir, hf_dir

# Rows per parquet batch. Small, because a row here carries a whole conversation's text: this
# bounds how much of the corpus is decoded into Python objects at once, which is the only thing
# standing between this script and WildChat's several-GiB `turns` column.
READ_BATCH_ROWS = 256

# How turns are joined in `--level document`. The same separator the rest of the pipeline uses to
# make a document's text out of its turns (`compute_features.TURN_SEPARATOR`), duplicated here
# rather than imported so this script pulls in no featurizer or defense registry -- it has to stay
# a light, login-node-runnable tool.
TURN_SEPARATOR = "\n\n"


def defended_stem(source: str, defense: str) -> str:
    """Name a defended split goes by: ``wildchat_dp_mlm_eps100``.

    Mirrors ``apply_defenses.defended_stem``; not imported from it because that module builds the
    defense registry on import, which drags in every defense's dependencies.
    """
    return f"{source}_{defense}"


def iter_rows(path: Path, columns: list[str]):
    """Yield the parquet's rows as dicts, one batch of :data:`READ_BATCH_ROWS` decoded at a time."""
    for batch in pq.ParquetFile(path).iter_batches(batch_size=READ_BATCH_ROWS, columns=columns):
        decoded = {name: batch.column(name).to_pylist() for name in columns}
        for position in range(batch.num_rows):
            yield {name: decoded[name][position] for name in columns}


class OriginalTurns:
    """The undefended turns of each defended document, read by a single forward pass over the split.

    :meth:`get` must be called in the defended file's order -- which is the split's order, so a
    forward scan finds every document exactly once and never holds more than one batch. Documents
    the defended file does not contain are skipped and discarded (counted in :attr:`skipped`), so a
    language-filtered export costs no extra memory.
    """

    def __init__(self, path: Path):
        self._path = path
        self._rows = iter_rows(path, ["doc_id", "turns"])
        self.skipped = 0

    def get(self, doc_id: str) -> list[str]:
        for row in self._rows:
            if row["doc_id"] == doc_id:
                return [str(turn) for turn in row["turns"]]
            self.skipped += 1
        raise SystemExit(
            f"reached the end of {self._path} while looking for document {doc_id!r}.\n"
            f"That means the defended file is not in the split's row order (or holds documents "
            f"the split does not), so originals cannot be matched up by a single pass. "
            f"Re-export with --no-original, or re-merge the shards.")


def escape_newlines(text: str) -> str:
    r"""Collapse a turn onto one physical line: CRLF/CR/LF all become a literal ``\n``."""
    return text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "\\n")


def write_csv(defended_path: Path, out_path: Path, *, original_path: Path | None,
              level: str, newlines: str, encoding: str, limit: int | None) -> None:
    """Stream the defended parquet into ``out_path``; print what landed there."""
    originals = OriginalTurns(original_path) if original_path else None
    transform = escape_newlines if newlines == "escape" else (lambda text: text)

    header = ["doc_id", "author_id"]
    header += ["turn_index", "n_turns"] if level == "turn" else ["n_turns"]
    if originals:
        header.append("original_text")
    header.append("defended_text")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    documents = rows = 0
    original_chars = defended_chars = 0
    sample = None

    # newline="" is required of any csv.writer: the module emits its own line terminators, and
    # letting Python also translate them corrupts every quoted field containing a newline -- which
    # here is most of them.
    with open(out_path, "w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_ALL, lineterminator="\n")
        writer.writerow(header)

        for document in iter_rows(defended_path, ["doc_id", "author_id", "turns"]):
            if limit is not None and documents >= limit:
                break
            doc_id = str(document["doc_id"])
            author_id = str(document["author_id"])
            defended = [str(turn) for turn in document["turns"]]
            # A turn-ADDING defense (frame_pad) emits more turns than the split records, so the
            # tail has no original to sit beside; "" says that plainly rather than misaligning the
            # column. DP-MLM preserves the count exactly, so this never fires for it.
            original = originals.get(doc_id) if originals else []
            # How many turns the document really had, before the padding below. A turn-ADDING
            # defense's extra turns must not lengthen the original side of a `--level document`
            # row with empty separators.
            original_count = len(original)
            original += [""] * max(0, len(defended) - original_count)

            documents += 1
            defended_chars += sum(len(turn) for turn in defended)
            original_chars += sum(len(turn) for turn in original)
            if sample is None and any(turn.strip() for turn in defended):
                sample = next(turn for turn in defended if turn.strip())

            if level == "document":
                row = [doc_id, author_id, len(defended)]
                if originals:
                    row.append(transform(TURN_SEPARATOR.join(original[:original_count])))
                row.append(transform(TURN_SEPARATOR.join(defended)))
                writer.writerow(row)
                rows += 1
                continue

            for index, turn in enumerate(defended):
                row = [doc_id, author_id, index, len(defended)]
                if originals:
                    row.append(transform(original[index]))
                row.append(transform(turn))
                writer.writerow(row)
                rows += 1

    print(f"wrote {rows:,} rows / {documents:,} documents -> {out_path} "
          f"({out_path.stat().st_size / 1e6:.1f} MB)")
    if originals:
        # The length ratio is the cheapest read on whether the defense did what it claims: plain
        # DP-MLM is word-count preserving, so ~1.0 here, while its adaptive-length variants and the
        # turn-adding defenses should visibly differ.
        ratio = defended_chars / original_chars if original_chars else float("nan")
        print(f"  characters: {original_chars:,} original -> {defended_chars:,} defended "
              f"({ratio:.2f}x); {originals.skipped:,} split documents skipped (not in this file)")
    if sample:
        excerpt = " ".join(sample.split())[:200]
        print(f"  first non-blank defended turn: {excerpt}{'...' if len(excerpt) == 200 else ''}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="wildchat",
                   help="the split the defended file was made from (default: wildchat)")
    p.add_argument("--defense", default="dp_mlm_eps100",
                   help="the defense it was defended with; names the input file together with "
                        "--source (default: dp_mlm_eps100)")
    p.add_argument("--defended", default=None,
                   help="export this parquet instead of <out-dir>/<source>_<defense>.parquet -- "
                        "e.g. one array task's shard under data/dist/shards/")
    p.add_argument("--out", default=None,
                   help="CSV to write (default: <out-dir>/csv/<input stem>.csv)")
    p.add_argument("--dist-dir", default=None,
                   help="where the UNDEFENDED split is read from, for the original_text column "
                        "(default: the project's data/hf)")
    p.add_argument("--out-dir", default=None,
                   help="where the defended parquet was written (default: the project's "
                        "data/dist)")
    p.add_argument("--no-original", action="store_true",
                   help="write only the defended text, and never open the undefended split")
    p.add_argument("--level", default="turn", choices=("turn", "document"),
                   help="one row per turn -- the unit the defense rewrote -- or one per document "
                        "with the turns joined by a blank line (default: turn)")
    p.add_argument("--newlines", default="keep", choices=("keep", "escape"),
                   help=r"'keep' quotes the text so real newlines survive (pandas/Excel read it "
                        r"back exactly); 'escape' writes them as \n so every record is one "
                        r"physical line, for grep (default: keep)")
    p.add_argument("--encoding", default="utf-8",
                   help="output encoding; use utf-8-sig if Excel must open it (default: utf-8)")
    p.add_argument("--limit", type=int, default=None,
                   help="export only the first N documents (for a quick look)")
    args = p.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else dist_dir()
    dist = Path(args.dist_dir) if args.dist_dir else hf_dir()

    defended_path = (Path(args.defended) if args.defended
                     else out_dir / f"{defended_stem(args.source, args.defense)}.parquet")
    if not defended_path.exists():
        raise SystemExit(
            f"{defended_path} not found -- defend the split first:\n"
            f"  python -m prompt_anonymity.data.apply_defenses "
            f"--source {args.source} --defense {args.defense}\n"
            f"(a sharded array writes shards under {out_dir / 'shards'} and merges them once the "
            f"last task finishes; `--merge` assembles them by hand.)")

    original_path = None
    if not args.no_original:
        original_path = dist / f"{args.source}.parquet"
        if not original_path.exists():
            raise SystemExit(f"{original_path} not found, so there is no original text to put "
                             f"beside the defended text. Point --dist-dir at the split, or pass "
                             f"--no-original.")

    out_path = Path(args.out) if args.out else out_dir / "csv" / f"{defended_path.stem}.csv"
    write_csv(defended_path, out_path, original_path=original_path, level=args.level,
              newlines=args.newlines, encoding=args.encoding, limit=args.limit)


if __name__ == "__main__":
    main()
