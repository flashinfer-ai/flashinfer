"""Per-kernel GPU-idle gap analysis for an nsys SQLite export of the EP bench.

Reads a .sqlite produced by `nsys stats --report cuda_gpu_kern_sum --force-export`
and, per kernel, reports the median GPU-idle gap *before* it (time since the previous
kernel ended) and its median duration. A large gap before a per-iter kernel = host
launch path executed as GPU idle (the FI-vs-ep_bench host-call overhead).

Usage:  python fi_gap_analysis.py <trace>.sqlite
"""

import sqlite3
import sys
from statistics import median

_KNOWN = (
    "ht_dispatch_kernel",
    "ht_combine_kernel",
    "ht_scan",
    "dense_to_sparse_prob",
    "sparse_to_dense_prob",
    "convert_topk",
    "AllReduce_Sum_f32",
    "AllReduce_Sum_u32",
    "AllGather",
    "reduce_kernel",
    "elementwise",
)


def short_name(name):
    for key in _KNOWN:
        if key in name:
            return key
    return name.split("(")[0].split("<")[0][:24]


def med_us(values):
    return median(values) / 1000.0 if values else 0.0


def main():
    conn = sqlite3.connect(sys.argv[1])
    kernels = list(
        conn.execute(
            "SELECT k.start, k.end, COALESCE(sd.value, ss.value) "
            "FROM CUPTI_ACTIVITY_KIND_KERNEL k "
            "LEFT JOIN StringIds sd ON k.demangledName = sd.id "
            "LEFT JOIN StringIds ss ON k.shortName = ss.id "
            "ORDER BY k.start"
        )
    )

    gap_before = {}
    duration = {}
    for i, (start, end, name) in enumerate(kernels):
        nm = short_name(name)
        duration.setdefault(nm, []).append(end - start)
        if i > 0:
            gap_before.setdefault(nm, []).append(start - kernels[i - 1][1])

    print(f"{'kernel':24s} {'n':>4} {'med_gap_before_us':>17} {'med_dur_us':>11}")
    for nm in sorted(duration, key=lambda k: -sum(duration[k])):
        print(
            f"{nm:24s} {len(duration[nm]):4d} "
            f"{med_us(gap_before.get(nm, [])):17.1f} {med_us(duration[nm]):11.1f}"
        )


if __name__ == "__main__":
    main()
