"""The keyed-hash RNG every deterministic defense draws from.

A defense that assigns something -- a marker profile to an author, a trigger to a document -- must
make that assignment a *pure function* of the thing's identifier, so the same id always draws the
same answer regardless of sharding, run order, or process state, and a manifest can be rebuilt
offline from the identifier list alone.

:func:`hash` cannot be used for this: Python salts string hashing per process. BLAKE2b is keyless,
fast, and stable across versions and platforms.
"""

from __future__ import annotations

import hashlib
import random


def keyed_rng(*parts) -> random.Random:
    """A ``random.Random`` seeded by the BLAKE2b digest of ``parts``.

    Parts are stringified and joined with a NUL (which can't occur in these ids), so ``("ab", "c")``
    and ``("a", "bc")`` hash to different keys. By convention the first part is the defense's master
    seed and the second names the kind of draw (``"profile"``, ``"trigger"``, ``"rate"``), which
    keeps independent decisions about the same document from correlating.
    """
    digest = hashlib.blake2b("\x00".join(str(p) for p in parts).encode("utf-8"), digest_size=8)
    return random.Random(int.from_bytes(digest.digest(), "big"))


__all__ = ["keyed_rng"]
