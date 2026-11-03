# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Summarize immutable critical-rank samples; retain per-rank provenance."""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    position = (len(values) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def stats(values):
    return dict(
        n=len(values),
        median_us=statistics.median(values),
        p10_us=percentile(values, 0.1),
        p90_us=percentile(values, 0.9),
        mean_us=statistics.mean(values),
        cv=statistics.stdev(values) / statistics.mean(values) if len(values) > 1 else 0,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--on-root", type=Path)
    parser.add_argument("--ep4-root", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    grouped = defaultdict(list)
    manifests = []
    flattened = []
    incomplete = []

    def source_paths(pattern):
        for mode, run_root in [("off", args.root), ("on", args.on_root or args.root)]:
            for path in (run_root / "measurements" / mode).glob(pattern):
                if (
                    mode == "off"
                    and args.ep4_root
                    and path.relative_to(run_root / "measurements" / mode).parts[0]
                    == "ep4"
                ):
                    continue
                yield path
        if args.ep4_root:
            yield from (args.ep4_root / "measurements" / "off").glob(
                pattern.replace("ep*", "ep4")
            )

    for path in sorted(source_paths("ep*/round-*/*/rank-0-manifest.json")):
        meta = json.loads(path.read_text())
        if not meta.get("complete"):
            incomplete.append(str(path))
            continue
        manifests.append(meta)
        sample_path = path.with_name("rank-0-samples.jsonl")
        records = [json.loads(line) for line in sample_path.read_text().splitlines()]
        for record in records:
            assert {row["rank"] for row in record["ranks"]} == set(range(meta["world"]))
            assert len(record["ranks"]) == meta["world"]
            assert 0 <= record["iteration"] < 20
            assert all(
                math.isfinite(row[key]) and row[key] > 0
                for row in record["ranks"]
                for key in ("gpu_us", "wall_us", "enqueue_us")
            )
            metric = "gpu_us" if record["scope"] == "raw_pipeline" else "wall_us"
            critical = max(record["ranks"], key=lambda row: row[metric])
            row = dict(
                mode=meta["mode"],
                ep=meta["world"],
                impl=meta["impl"],
                round=meta["round"],
                tokens=record["tokens"],
                cache=record["cache"],
                scope=record["scope"],
                iteration=record["iteration"],
                critical_rank=critical["rank"],
                primary_us=critical[metric],
                gpu_us=critical["gpu_us"],
                wall_us=critical["wall_us"],
                enqueue_us=critical.get("enqueue_us"),
            )
            flattened.append(row)
            key = tuple(
                row[name] for name in ("mode", "ep", "tokens", "cache", "scope", "impl")
            )
            grouped[key].append(row)
    results = []
    for key, rows in sorted(grouped.items()):
        item = dict(
            zip(("mode", "ep", "tokens", "cache", "scope", "impl"), key, strict=True)
        )
        rounds = defaultdict(list)
        for row in rows:
            rounds[row["round"]].append(row["primary_us"])
        round_medians = [statistics.median(v) for v in rounds.values()]
        item.update(stats([row["primary_us"] for row in rows]))
        item["rounds"] = {str(k): stats(v) for k, v in rounds.items()}
        item["round_cv"] = (
            statistics.stdev(round_medians) / statistics.mean(round_medians)
            if len(round_medians) > 1
            else None
        )
        item["complete"] = len(rounds) == 5 and all(
            len(v) == 20 for v in rounds.values()
        )
        results.append(item)
    comparisons = []
    for port in [row for row in results if row["impl"] == "port"]:
        control = "base" if port["mode"] == "off" else "donor"
        matches = [
            row
            for row in results
            if row["impl"] == control
            and all(
                row[k] == port[k] for k in ("mode", "ep", "tokens", "cache", "scope")
            )
        ]
        if not matches:
            continue
        baseline = matches[0]
        status = "incomplete"
        ratio = port["median_us"] / baseline["median_us"]
        if port["complete"] and baseline["complete"]:
            if port["mode"] == "off":
                status = (
                    "inconclusive-baseline-dispersion"
                    if baseline["round_cv"] > 0.05
                    else ("PASS" if ratio <= 1.05 else "REGRESSION")
                )
            else:
                status = "measured-no-speedup-gate"
        comparisons.append(
            dict(
                mode=port["mode"],
                ep=port["ep"],
                tokens=port["tokens"],
                cache=port["cache"],
                scope=port["scope"],
                baseline_us=baseline["median_us"],
                port_us=port["median_us"],
                ratio=ratio,
                baseline_round_cv=baseline["round_cv"],
                status=status,
            )
        )
    # Cross-source identity checks are mandatory, not inferred from equal seeds.
    identities = defaultdict(list)
    all_rank_manifests = []
    all_rank_paths = []
    for path in sorted(source_paths("ep*/round-*/*/rank-*-manifest.json")):
        manifest = json.loads(path.read_text())
        if not manifest.get("complete"):
            if str(path) not in incomplete:
                incomplete.append(str(path))
            continue
        all_rank_manifests.append(manifest)
        all_rank_paths.append(str(path.resolve()))
    for manifest in all_rank_manifests:
        for case in manifest["cases"]:
            identities[
                (manifest["mode"], manifest["world"], case["tokens"], manifest["rank"])
            ].append((manifest["impl"], case["payload"]))
    payload_checks = []
    for key, values in sorted(identities.items()):
        expected = values[0][1]
        payload_checks.append(
            dict(
                mode=key[0],
                ep=key[1],
                tokens=key[2],
                rank=key[3],
                records=len(values),
                match=all(value == expected for _, value in values),
            )
        )
    profile_groups = defaultdict(list)
    for path in sorted(source_paths("ep*/round-0/*/rank-*-manifest.json")):
        meta = json.loads(path.read_text())
        if not meta.get("complete"):
            continue
        for case in meta["cases"]:
            for cache, profile in case.get("profiles", {}).items():
                mega = [
                    event["duration_us"]
                    for event in profile["events"]
                    if "kernel_cutlass" in event["name"]
                    and "kernel_cutlass__reduce_" not in event["name"]
                ]
                reduce = [
                    event["duration_us"]
                    for event in profile["events"]
                    if "kernel_cutlass__reduce_" in event["name"]
                ]
                key = (meta["mode"], meta["world"], case["tokens"], cache, meta["impl"])
                profile_groups[key].append(
                    dict(
                        rank=meta["rank"],
                        mega=mega,
                        reduce=reduce,
                        trace=profile["trace"],
                        names=sorted(set(event["name"] for event in profile["events"])),
                        expected_calls=profile["calls"],
                    )
                )
    profiles = []
    for key, values in sorted(profile_groups.items()):
        item = dict(zip(("mode", "ep", "tokens", "cache", "impl"), key, strict=True))
        valid = (
            len(values) == key[1]
            and {row["rank"] for row in values} == set(range(key[1]))
            and all(
                len(row["mega"]) == row["expected_calls"]
                and len(row["reduce"]) == row["expected_calls"]
                for row in values
            )
        )
        item.update(valid=valid, ranks=values)
        if valid:
            calls = values[0]["expected_calls"]
            item["fused_critical_rank"] = stats(
                [max(row["mega"][i] for row in values) for i in range(calls)]
            )
            item["reduce_critical_rank"] = stats(
                [max(row["reduce"][i] for row in values) for i in range(calls)]
            )
        profiles.append(item)
    output = dict(
        run_roots=dict(
            off=str(args.root),
            on=str(args.on_root or args.root),
            off_ep4=str(args.ep4_root) if args.ep4_root else None,
        ),
        rows=results,
        comparisons=comparisons,
        payload_checks=payload_checks,
        profiles=profiles,
        incomplete_manifests=incomplete,
        manifests=manifests,
        all_rank_manifest_paths=all_rank_paths,
    )
    (args.root / "summary.json").write_text(json.dumps(output, indent=2) + "\n")
    if flattened:
        with (args.root / "critical-rank-samples.csv").open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flattened[0]))
            writer.writeheader()
            writer.writerows(flattened)
    print(
        "Complete source points:",
        len(manifests),
        "sample groups:",
        len(results),
        "comparisons:",
        len(comparisons),
    )
    for row in comparisons:
        print(
            f"{row['mode']:3} EP{row['ep']} T{row['tokens']:4} {row['cache']:4} {row['scope']:14} {row['baseline_us']:10.2f} -> {row['port_us']:10.2f} us {row['ratio']:6.3f}x {row['status']}"
        )
    assert all(row["match"] for row in payload_checks), (
        "Quantized payload/config identity mismatch"
    )
    if args.require_complete:
        assert not incomplete, incomplete
        assert len(manifests) == 40, len(manifests)
        assert len(all_rank_manifests) == 110, len(all_rank_manifests)
        assert len(results) == 56 and all(row["complete"] for row in results), (
            "Incomplete primary matrix"
        )
        assert len(comparisons) == 28, len(comparisons)
        assert len(flattened) == 5600, len(flattened)
        assert len(payload_checks) == 18 and all(
            row["records"] == 10 for row in payload_checks
        )
        assert len(profiles) == 28 and all(row["valid"] for row in profiles), (
            "Incomplete profiler matrix"
        )
        assert all(
            case["validation"] == "bitwise PASS"
            for meta in all_rank_manifests
            for case in meta["cases"]
        )
        print(
            "PASS: full matrix coverage, payload identity, bitwise validation and profiler attribution"
        )


if __name__ == "__main__":
    main()
