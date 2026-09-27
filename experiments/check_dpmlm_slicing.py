"""Verify DP-MLM's sliced/streamed sampling before committing a long run to it.

    srun --partition=gpu-preempt --gpus=1 --mem=24g --time=00:25:00 --pty \
        python experiments/check_dpmlm_slicing.py

Runs in a few minutes on any GPU (or slowly on a CPU node) and exits non-zero if anything fails,
so it can gate a submission:

    python experiments/check_dpmlm_slicing.py && sbatch experiments/run_dpmlm_wildchat.sbatch

Background: sampling in :mod:`~prompt_anonymity.defenses.dp_mlm` materializes
``[positions x vocab]`` in float32 plus a softmax of the same shape, per conversation -- a long
enough turn can OOM. ``DPMLM_SAMPLE_SLICE`` caps the rows per sampling call and turns longer than
it are streamed in waves, which bounds peak memory by the wave instead of by the turn.

The three properties that has to have, and that this checks:

1. **Equivalence below the threshold.** A turn no longer than the slice must produce exactly what
   the pre-slicing code produced -- one sampling call on one generator state. This is what keeps
   every cached turn on disk valid. ``sample_slice = 0`` selects the old path, so the check is a
   direct A/B on the same input.
2. **Determinism when streamed.** A turn longer than the slice is drawn wave by wave; the result
   must still be reproducible, or the cache would serve one text and a re-run produce another.
3. **Structure preserved.** Streaming must not drop, duplicate or reorder words: plain DP-MLM is
   word-count preserving, so the streamed output must have the same word count as the unstreamed
   one, and an empty turn must stay empty.

It deliberately does NOT assert that a streamed turn equals an unstreamed one -- it cannot. Waves
consume the RNG differently, so the sampled words differ. That is the documented trade for being
able to process such turns at all.
"""

from __future__ import annotations

import sys

from prompt_anonymity.defenses.dp_mlm import DPMLM_SAMPLE_SLICE, DPMLMDefense

SENTENCE = "the quick brown fox jumps over the lazy dog"
SHORT = " ".join([SENTENCE] * 30)    # below any threshold tested here
LONG = " ".join([SENTENCE] * 300)    # long enough to force streaming at slice=64


def main() -> int:
    print(f"DPMLM_SAMPLE_SLICE default: {DPMLM_SAMPLE_SLICE}")
    backend = DPMLMDefense(epsilon=100)._get_backend()
    print(f"device: {backend.device}")
    failures = []

    # 1. Below the threshold: the new default must reproduce the old path exactly.
    backend.sample_slice = 0
    old = backend.rewrite_batch([SHORT])[0]
    backend.sample_slice = DPMLM_SAMPLE_SLICE
    new = backend.rewrite_batch([SHORT])[0]
    ok = old == new
    print(f"[1] equivalence below threshold (slice 0 vs {DPMLM_SAMPLE_SLICE}): {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("a short turn changed output -- cached turns would no longer match")
        print(f"    old: {old[:120]}")
        print(f"    new: {new[:120]}")

    # 2/3. Force the streaming path with a small slice, and run it twice.
    backend.sample_slice = 64
    first = backend.rewrite_batch([LONG, "Hello there, can you help me?", ""])
    second = backend.rewrite_batch([LONG])[0]

    ok = first[0] == second
    print(f"[2] streamed output is deterministic: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("streaming is not reproducible -- the cache would be inconsistent")

    # The unstreamed run of the same text is the reference for structure (not for content).
    backend.sample_slice = 0
    reference = backend.rewrite_batch([LONG])[0]
    streamed_words, reference_words = len(first[0].split()), len(reference.split())
    ok = streamed_words == reference_words
    print(f"[3] streamed word count {streamed_words} == unstreamed {reference_words}: "
          f"{'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append("streaming changed the turn's length -- words dropped, added or reordered")

    ok = first[2] == ""
    print(f"[4] empty turn passes through: {'PASS' if ok else 'FAIL'}")
    if not ok:
        failures.append(f"an empty turn came back as {first[2]!r}")

    print()
    print(f"short sample: {' '.join(first[1].split()[:14])}")
    print(f"long sample : {' '.join(first[0].split()[:14])}")
    print()
    if failures:
        print("FAILED:")
        for failure in failures:
            print(f"  - {failure}")
        return 1
    print("All checks passed. The sliced and streamed paths behave, and short turns are unchanged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
