# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize every rank and preserve source/protocol boundaries."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics


def stats(values):
    values = sorted(values)

    def p(q):
        x = (len(values) - 1) * q
        i = math.floor(x)
        j = math.ceil(x)
        return values[i] + (values[j] - values[i]) * (x - i)

    return dict(
        n=len(values),
        median_us=statistics.median(values),
        p10_us=p(0.1),
        p90_us=p(0.9),
        cv=statistics.stdev(values) / statistics.mean(values),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("job", type=Path)
    a = p.parse_args()
    meta = [json.loads(f.read_text()) for f in a.job.glob("*/rank-*-runtime.json")]
    impls = sorted(set(m["impl"] for m in meta))
    assert impls and "donor" in impls
    assert len(meta) == len(impls) * 4
    assert all(m.get("complete") and m["validation"] == "bitwise PASS" for m in meta)
    for rank in range(4):
        payload = [m["payload"] for m in meta if m["rank"] == rank]
        assert len(payload) == len(impls) and all(v == payload[0] for v in payload)
    flat = []
    g = collections.defaultdict(list)
    for impl in impls:
        for phase in ["standard", "conditioned"]:
            for rid in range(5):
                path = a.job / impl / f"rank-0-{phase}-round-{rid}.jsonl"
                records = [json.loads(line) for line in path.read_text().splitlines()]
                assert len(records) == 80
                for r in records:
                    assert {v["rank"] for v in r["ranks"]} == set(range(4))
                    metric = "gpu_us" if r["scope"] == "raw_pipeline" else "wall_us"
                    critical = max(r["ranks"], key=lambda v: v[metric])
                    assert all(
                        math.isfinite(v[k]) and v[k] > 0
                        for v in r["ranks"]
                        for k in ["gpu_us", "wall_us", "enqueue_us"]
                    )
                    row = dict(
                        phase=phase,
                        impl=impl,
                        round=rid,
                        cache=r["cache"],
                        scope=r["scope"],
                        iteration=r["iteration"],
                        critical_rank=critical["rank"],
                        primary_us=critical[metric],
                        gpu_us=critical["gpu_us"],
                        wall_us=critical["wall_us"],
                        enqueue_us=critical["enqueue_us"],
                    )
                    flat.append(row)
                    g[phase, r["cache"], r["scope"], impl].append(row)
    rows = []
    for k, values in sorted(g.items()):
        rs = collections.defaultdict(list)
        for v in values:
            rs[v["round"]].append(v["primary_us"])
        assert set(rs) == set(range(5)) and all(len(v) == 20 for v in rs.values())
        row = dict(zip(["phase", "cache", "scope", "impl"], k, strict=True))
        row.update(stats([v["primary_us"] for v in values]))
        row["rounds"] = {str(rid): stats(v) for rid, v in rs.items()}
        round_med = [statistics.median(v) for v in rs.values()]
        row["round_cv"] = statistics.stdev(round_med) / statistics.mean(round_med)
        rows.append(row)
    comparisons = []
    for r in rows:
        if r["impl"] == "donor":
            continue
        b = next(
            v
            for v in rows
            if v["impl"] == "donor"
            and all(v[k] == r[k] for k in ["phase", "cache", "scope"])
        )
        ratio = r["median_us"] / b["median_us"]
        status = (
            "different-host-wrapper-scopes"
            if r["scope"] == "full_operator"
            else (
                "inconclusive-baseline-dispersion"
                if b["round_cv"] > 0.05
                else "PASS"
                if ratio <= 1.05
                else "REGRESSION"
            )
        )
        comparisons.append(
            dict(
                phase=r["phase"],
                cache=r["cache"],
                scope=r["scope"],
                impl=r["impl"],
                baseline_us=b["median_us"],
                port_us=r["median_us"],
                ratio=ratio,
                status=status,
                baseline_round_cv=b["round_cv"],
            )
        )
    profiles = []
    for impl in impls:
        for cache in ["warm", "cold"]:
            ranks = []
            for m in meta:
                if m["impl"] != impl:
                    continue
                profile = m["profiles"][cache]
                events = profile["events"]
                mega = [
                    e["duration_us"]
                    for e in events
                    if "kernel_cutlass" in e["name"]
                    and "kernel_cutlass__reduce_" not in e["name"]
                ]
                reduce = [
                    e["duration_us"]
                    for e in events
                    if "kernel_cutlass__reduce_" in e["name"]
                ]
                assert len(mega) == len(reduce) == 5
                ranks.append(
                    dict(
                        rank=m["rank"], mega=mega, reduce=reduce, trace=profile["trace"]
                    )
                )
            assert len(ranks) == 4
            profiles.append(
                dict(
                    impl=impl,
                    cache=cache,
                    fused=stats([max(v["mega"][i] for v in ranks) for i in range(5)]),
                    reduce=stats(
                        [max(v["reduce"][i] for v in ranks) for i in range(5)]
                    ),
                    ranks=ranks,
                )
            )
    compiler_audits = []
    if "before" in impls and "port" in impls:
        for rank in range(4):
            result = dict(rank=rank)
            for suffix in [".ptx", ".cubin"]:
                result[suffix] = (
                    hashlib.sha256(
                        (a.job / "before" / f"rank-{rank}{suffix}").read_bytes()
                    ).hexdigest()
                    == hashlib.sha256(
                        (a.job / "port" / f"rank-{rank}{suffix}").read_bytes()
                    ).hexdigest()
                )
            compiler_audits.append(result)
    result = dict(
        job=str(a.job.resolve()),
        implementations=impls,
        rows=rows,
        comparisons=comparisons,
        profiles=profiles,
        runtime=meta,
        payload_identity="all ranks match",
        compiler_audits=compiler_audits,
        sample_count=len(flat),
    )
    (a.job / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    with (a.job / "critical-rank-samples.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    print(
        "Validated",
        len(rows),
        "groups;",
        len(flat),
        "critical-rank samples; all payloads, bit checks and profiles pass",
    )
    for r in comparisons:
        print(r)


if __name__ == "__main__":
    main()
