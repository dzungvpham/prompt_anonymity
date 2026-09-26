"""What this process may actually use: CPU cores, memory, and GPU memory.

On a shared batch node the obvious answers are wrong: ``os.cpu_count()`` and ``/proc/meminfo``
describe the *machine*, not the slice of it the scheduler handed this job. Sizing a worker pool
from the machine's numbers oversubscribes the allocation -- harmless for CPU, but exceeding the
memory cap gets the job OOM-killed, usually deep into a long run.

Three sources, in decreasing reliability, all consulted and the smallest answer taken:

* **Scheduler affinity** (``os.sched_getaffinity``) -- authoritative for CPUs, since a cpuset
  cgroup shows up directly as the process's affinity mask.
* **cgroup limits** -- authoritative for memory. The limit is frequently set on an *ancestor*
  cgroup rather than the process's own: SLURM caps memory at the job level while the task's leaf
  cgroup reads ``max``, so every level from the leaf up to the root has to be read (see
  :func:`_cgroup_limits`). Both cgroup v2 (``memory.max``, ``cpu.max``) and v1
  (``memory.limit_in_bytes``, ``cpu.cfs_quota_us``) layouts are handled.
* **Scheduler environment** (``SLURM_CPUS_PER_TASK``, ``SLURM_MEM_PER_NODE`` / ``_PER_CPU``) --
  a cross-check for the cases where the cgroup is not visible from inside the job.

Everything degrades safely: a detector that finds nothing returns ``None`` and simply does not
constrain the result, so this works unchanged on a laptop with no cgroups and no scheduler.
"""

from __future__ import annotations

import os
from pathlib import Path

CGROUP_ROOT = Path("/sys/fs/cgroup")

# cgroup v1 writes a sentinel near 2**63 for "no limit" rather than a word like "max"; anything
# at or above this is not a real cap. (The exact value is PAGE_COUNTER_MAX, kernel-dependent.)
_CGROUP_V1_UNLIMITED = 2 ** 62

MIB = 2 ** 20
GIB = 2 ** 30


def _read_int(path: Path) -> int | None:
    """First whitespace-separated token of ``path`` as an int, or ``None``.

    ``None`` covers every "no limit here" case: a missing/unreadable file, the cgroup v2 literal
    ``max``, and cgroup v1's near-2**63 sentinel.
    """
    try:
        token = path.read_text().split()[0]
    except (OSError, IndexError):
        return None
    if token == "max":
        return None
    try:
        value = int(token)
    except ValueError:
        return None
    return None if value < 0 or value >= _CGROUP_V1_UNLIMITED else value


def _cgroup_self_path() -> Path | None:
    """Filesystem path of this process's cgroup v2 node, or ``None`` if it cannot be determined."""
    try:
        entries = Path("/proc/self/cgroup").read_text().splitlines()
    except OSError:
        return None
    for entry in entries:
        fields = entry.split(":", 2)  # v2 lines are "0::/relative/path"
        if len(fields) == 3 and fields[0] == "0":
            return CGROUP_ROOT / fields[2].lstrip("/")
    return None


def _cgroup_limits(filename: str) -> list[int]:
    """Every numeric value of ``filename`` from this process's cgroup up to the root.

    Walking upward is the point: a SLURM job's memory cap lives on the job-level cgroup while the
    task's own leaf reads ``max``, so reading only the leaf finds no limit at all.
    """
    values: list[int] = []
    node = _cgroup_self_path()
    while node is not None:
        value = _read_int(node / filename)
        if value is not None:
            values.append(value)
        if node == CGROUP_ROOT or CGROUP_ROOT not in node.parents:
            break
        node = node.parent
    return values


def _env_int(name: str) -> int | None:
    """Environment variable as a positive int, or ``None`` when unset/malformed."""
    try:
        value = int(os.environ[name])
    except (KeyError, ValueError):
        return None
    return value if value > 0 else None


def available_cpus() -> int:
    """CPU cores this process may actually run on (at least 1).

    The scheduler affinity mask is the primary signal (a cpuset cgroup appears there directly),
    narrowed by a cgroup CPU *quota* if one is set (``cpu.max`` / ``cfs_quota_us``, which limit
    CPU time rather than which cores are usable) and by ``SLURM_CPUS_PER_TASK``.
    """
    try:
        limits = [len(os.sched_getaffinity(0))]
    except AttributeError:  # not Linux; no affinity mask to consult
        limits = [os.cpu_count() or 1]

    for quota_line in _cgroup_limits("cpu.max"):  # v2: "<quota> <period>", quota already parsed
        limits.append(max(1, quota_line // 100_000))
    quota = _read_int(CGROUP_ROOT / "cpu" / "cpu.cfs_quota_us")   # v1
    period = _read_int(CGROUP_ROOT / "cpu" / "cpu.cfs_period_us") or 100_000
    if quota:
        limits.append(max(1, quota // period))

    slurm_cpus = _env_int("SLURM_CPUS_PER_TASK") or _env_int("SLURM_CPUS_ON_NODE")
    if slurm_cpus:
        limits.append(slurm_cpus)
    return max(1, min(limits))


def available_memory_bytes() -> int | None:
    """Memory this process may use, in bytes, or ``None`` if no limit can be determined.

    The smallest of: every cgroup ``memory.max`` from this process's cgroup up to the root (v2),
    the v1 ``memory.limit_in_bytes``, SLURM's ``SLURM_MEM_PER_NODE`` / ``SLURM_MEM_PER_CPU``, and
    the machine's physical memory. ``None`` means nothing was detectable -- callers should treat
    that as "unconstrained" rather than as zero.
    """
    limits = list(_cgroup_limits("memory.max"))                       # cgroup v2, leaf upward
    v1 = _read_int(CGROUP_ROOT / "memory" / "memory.limit_in_bytes")  # cgroup v1
    if v1:
        limits.append(v1)

    slurm_mb = _env_int("SLURM_MEM_PER_NODE")
    if slurm_mb is None:
        per_cpu = _env_int("SLURM_MEM_PER_CPU")
        slurm_mb = per_cpu * available_cpus() if per_cpu else None
    if slurm_mb:
        limits.append(slurm_mb * MIB)

    try:  # the machine's own memory is still an upper bound
        limits.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    return min(limits) if limits else None


def gpu_memory_bytes() -> int | None:
    """Free memory on the active GPU in bytes, or ``None`` when there is no usable GPU.

    Reports *free* rather than total memory, since another process on a shared GPU may already
    hold part of it. Requires torch; any failure (no torch, no CUDA, a driver error) is reported
    as ``None`` so callers fall back to their no-GPU path rather than crashing.

    Note that querying this initializes a CUDA context in **this** process (a few hundred MiB).
    Call it once, before deciding how to distribute work.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return int(free)
    except Exception:
        return None


def describe_budget() -> str:
    """One-line, human-readable summary of the detected budget, for run logs."""
    memory = available_memory_bytes()
    memory_text = f"{memory / GIB:.1f} GiB" if memory else "unknown"
    gpu = gpu_memory_bytes()
    gpu_text = f", {gpu / GIB:.1f} GiB free GPU memory" if gpu else ""
    return f"{available_cpus()} CPU(s), {memory_text} memory{gpu_text}"


__all__ = ["available_cpus", "available_memory_bytes", "gpu_memory_bytes", "describe_budget"]
