"""Host CPU counters shared by allocation and per-batch sampling."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HostCpuTimes:
    total: int
    idle: int
    iowait: int


def read_host_cpu_times() -> HostCpuTimes | None:
    try:
        fields = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()
        if not fields or fields[0] != "cpu":
            return None
        values = [int(value) for value in fields[1:]]
    except (OSError, ValueError, IndexError):
        return None
    if len(values) < 5:
        return None
    # Linux reports guest time again inside user/nice, so only the first eight
    # non-guest counters belong in the total.
    return HostCpuTimes(total=sum(values[:8]), idle=values[3], iowait=values[4])


def host_cpu_percentages(
    previous: HostCpuTimes | None, current: HostCpuTimes | None
) -> tuple[float | None, float | None]:
    if previous is None or current is None:
        return None, None
    total = current.total - previous.total
    if total <= 0:
        return None, None
    idle = max(0, current.idle - previous.idle)
    iowait = max(0, current.iowait - previous.iowait)
    busy = max(0, total - idle - iowait)
    return 100 * busy / total, 100 * iowait / total
