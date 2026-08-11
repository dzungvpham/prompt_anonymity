"""The keyed-hash RNG every deterministic defense draws from.

A defense that assigns something -- a marker profile to an author, a trigger to a document -- must
make that assignment a *pure function* of the thing's identifier. Three properties follow, and all
three are load-bearing for this pipeline:

* **Shard invariance.** ``apply_defenses`` runs as a SLURM array over document slices. If an
  assignment depended on batch position or on a stateful generator, the same document would get a
  different answer under a different ``--num-shards``, and the defended split would stop being a
  function of its input.
* **Bit-identical re-runs.** No state has to survive a run for it to be reproducible.
* **Offline reconstruction.** The manifest a defense writes for downstream analysis is rebuilt from
  the identifier list alone, without re-reading the corpus or the defended parquet.

:func:`hash` cannot be used for this: Python salts string hashing per process, so the same author id
hashes differently in two runs of the same script. BLAKE2b is keyless, fast, and stable across
versions and platforms.
"""

from __future__ import annotations

import hashlib
import random


def keyed_rng(*parts) -> random.Random:
    """A ``random.Random`` seeded by the BLAKE2b digest of ``parts``.

    Parts are stringified and joined with a NUL, which cannot occur in the ids this package builds,
    so ``("ab", "c")`` and ``("a", "bc")`` are different keys. By convention the first part is the
    defense's master seed and the second a short string literal naming the *kind* of draw
    (``"profile"``, ``"trigger"``, ``"rate"``); that namespace is what keeps two independent
    decisions about the same document from being perfectly correlated.
    """
    digest = hashlib.blake2b("\x00".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "big"))


__all__ = ["keyed_rng"]
