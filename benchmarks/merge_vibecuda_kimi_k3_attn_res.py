"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""Merge the two-process VibeCUDA Kimi-K3 AttnRes benchmark JSONs.

``bench_vibecuda_kimi_k3_attn_res.py`` runs in two clean processes: the PR
checkout (candidate arms ``vibecuda``/``vibecuda_nopdl`` plus the checkout-HEAD
CAKE diagnostic and the pinned SGLang arm) and the pinned CAKE revision
(``cake`` at PR #5916 head, the eager FP32 ``torch`` reference, and the pinned
SGLang arm re-measured). This tool joins the per-row JSONs of the two
processes row by row and reports per-workload ratios plus aggregate means.

Denominator roles (labels only; row measurements are never rewritten):

  primary      : ``sglang`` — SGLang PR32890 standalone op, pinned clean
                 checkout at :data:`SGLANG_COMMIT`; the configured fast_impl
                 denominator of the fixed benchmark.
  diagnostic   : ``cake_pinned`` — CAKE at PR #5916 head :data:`CAKE_COMMIT`
                 (pure pinned arm; registers only the K=4 unfused cells, so
                 every other row keeps an explicit ``unsupported`` status and
                 is excluded from that baseline's means, never from the
                 matrix).
  diagnostic   : ``cake_latest`` — CAKE at the PR checkout's HEAD.
  informational: ``torch`` — eager FP32 reference.

Each side accepts one or more JSON files, so a matrix split across several
chunk invocations (same hardware, identical rows/seeds/iters) merges exactly
like a single full run. The tool verifies that device metadata, accuracy
policy, timing configuration, workload labels, shapes, and input families
agree across every input file, and that no measured arm exceeds the public
accuracy tolerance; any accuracy failure makes the exit status nonzero.

Usage::

    python benchmarks/merge_vibecuda_kimi_k3_attn_res.py \
        --candidate vc_cand.json --baseline vc_pinned.json \
        -o vc_bench_merged.json
"""

import argparse
import json
import sys

SGLANG_COMMIT = "fb207b72b02a2adfab3a4a18f5e3b5fa57bb1049"
CAKE_COMMIT = "44b9ef010ad527efeb007d413ccd550f1e641eca"
ATOL, RTOL = 8e-2, 3e-2

CANONICAL_13 = [
    "m1_k1",
    "m8_k2",
    "m16_k4",
    "m32_k4",
    "m64_k8",
    "m128_k8",
    "m256_k8",
    "m512_k8",
    "m1024_k8",
    "m2048_k8",
    "m4096_k8",
    "m8192_k8",
    "m9364_k8",
]


def load_side(paths, side):
    rows, corr = [], []
    meta = None
    accuracy = None
    timing = None
    for path in paths:
        with open(path) as f:
            data = json.load(f)
        m = {k: data[k] for k in ("device", "capability", "torch")}
        if meta is None:
            meta = m
            accuracy = data.get("accuracy_policy")
            timing = data.get("timing")
        else:
            assert m == meta, f"{side}: device metadata mismatch in {path}"
            assert data.get("accuracy_policy") == accuracy, (
                f"{side}: accuracy policy mismatch in {path}"
            )
            assert data.get("timing") == timing, f"{side}: timing mismatch in {path}"
        for row in data["rows"]:
            assert row["label"] not in {r["label"] for r in rows}, (
                f"{side}: duplicate row {row['label']} across input files"
            )
            rows.append(row)
        corr.extend(data.get("correctness_rows", []))
        commit = data.get("sglang_commit")
        if commit is not None:
            assert commit == SGLANG_COMMIT, (
                f"{side}: unexpected sglang_commit {commit} in {path}"
            )
    assert meta is not None, f"{side}: no input files"
    return meta, accuracy, timing, rows, corr


def geo(values):
    g = 1.0
    for value in values:
        g *= value
    return g ** (1.0 / len(values))


def merge_rows(crows, brows):
    """Join candidate/baseline process rows and compute per-row ratios."""
    crow = {r["label"]: r for r in crows}
    brow = {r["label"]: r for r in brows}
    assert set(crow) == set(brow), (
        "workload matrix mismatch between the candidate and baseline processes"
    )
    merged = []
    ratios = {
        "sglang": [],
        "sglang13": [],
        "cake_pinned": [],
        "cake_pinned13": [],
        "cake_latest": [],
        "cake_latest13": [],
        "torch": [],
    }
    acc_fail = []
    for label in crow:
        cand, base = crow[label], brow[label]
        assert cand["M"] == base["M"] and cand["K"] == base["K"], label
        assert cand["input_family"] == base["input_family"], label
        row = dict(
            label=label, M=cand["M"], K=cand["K"], input_family=cand["input_family"]
        )
        vc = cand.get("vibecuda_ms")
        assert vc is not None, (
            f"{label}: candidate process did not measure the vibecuda arm"
        )
        row["vibecuda_ms"] = vc
        row["vibecuda_nopdl_ms"] = cand.get("vibecuda_nopdl_ms")
        row["vibecuda_max_abs_err"] = cand.get("vibecuda_max_abs_err")
        row["vibecuda_nopdl_max_abs_err"] = cand.get("vibecuda_nopdl_max_abs_err")
        row["sglang_ms"] = cand.get("sglang_ms")
        if row["sglang_ms"] is None:
            row["sglang_ms"] = base.get("sglang_ms")
        row["sglang_status"] = cand.get("sglang_status") or base.get("sglang_status")
        row["sglang_max_abs_err"] = cand.get("sglang_max_abs_err")
        if row["sglang_max_abs_err"] is None:
            row["sglang_max_abs_err"] = base.get("sglang_max_abs_err")
        row["cake_pinned_ms"] = base.get("cake_ms")
        row["cake_pinned_status"] = base.get("cake_status")
        row["cake_pinned_max_abs_err"] = base.get("cake_max_abs_err")
        row["cake_latest_ms"] = cand.get("cake_ms")
        row["cake_latest_status"] = cand.get("cake_status")
        row["cake_latest_max_abs_err"] = cand.get("cake_max_abs_err")
        row["torch_ms"] = base.get("torch_ms")
        row["torch_max_abs_err"] = base.get("torch_max_abs_err")
        for key, tag in (
            ("sglang_ms", "sglang"),
            ("cake_pinned_ms", "cake_pinned"),
            ("cake_latest_ms", "cake_latest"),
            ("torch_ms", "torch"),
        ):
            val = row[key]
            if val:
                ratio = val / vc
                row[f"speedup_vibecuda_vs_{key[:-3]}"] = ratio
                ratios[tag].append(ratio)
                if f"{tag}13" in ratios and label in CANONICAL_13:
                    ratios[f"{tag}13"].append(ratio)
        for name, src in (
            ("vibecuda", cand),
            ("vibecuda_nopdl", cand),
            ("sglang", cand),
            ("cake_latest", cand),
            ("cake_pinned", base),
            ("torch", base),
        ):
            key = (
                "cake_max_abs_err"
                if name.startswith("cake_")
                else f"{name}_max_abs_err"
            )
            err = src.get(key)
            if err is None and name == "sglang":
                err = base.get(key)
            if err is not None and not err < ATOL:
                acc_fail.append(f"{label}/{name} err={err}")
        merged.append(row)
        print(json.dumps(row))
    return merged, ratios, acc_fail


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--candidate",
        nargs="+",
        required=True,
        help="JSON(s) from the PR-checkout process (vibecuda arms)",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        help="JSON(s) from the pinned-CAKE process (baseline arms)",
    )
    parser.add_argument("-o", "--out", default="vc_bench_merged.json")
    args = parser.parse_args(argv)

    cmeta, cacc, ctim, crows, ccorr = load_side(args.candidate, "candidate")
    bmeta, bacc, btim, brows, bcorr = load_side(args.baseline, "baseline")
    assert cmeta == bmeta, "device metadata mismatch between processes"
    assert cacc == bacc, "accuracy policy mismatch between processes"
    assert ctim == btim, "timing configuration mismatch between processes"

    merged, ratios, acc_fail = merge_rows(crows, brows)

    summary = dict(
        device=cmeta["device"],
        capability=cmeta["capability"],
        torch=cmeta["torch"],
        rows=len(merged),
        correctness_rows=ccorr + bcorr,
        accuracy_failures=acc_fail,
        accuracy_policy=dict(atol=ATOL, rtol=RTOL),
        timing=ctim,
        primary_denominator=(
            f"sglang_b200 @ {SGLANG_COMMIT} (configured fast_impl; SGLang "
            "PR32890 standalone op, pinned clean checkout)"
        ),
        cake_pinned_denominator=(
            f"CAKE @ PR#5916 head {CAKE_COMMIT} (DIAGNOSTIC: pure pinned CAKE "
            "arm; unfused K=4 rows only, all other rows explicit unsupported "
            "status)"
        ),
        cake_latest_denominator="CAKE @ checkout HEAD (diagnostic only)",
        torch_reference="eager FP32 reference (informational)",
    )
    for tag, values in ratios.items():
        if values:
            summary[f"vs_{tag}_arith_mean"] = sum(values) / len(values)
            summary[f"vs_{tag}_geomean"] = geo(values)
            summary[f"vs_{tag}_min"] = min(values)
            summary[f"vs_{tag}_n"] = len(values)
    print(json.dumps(summary))
    with open(args.out, "w") as f:
        json.dump(dict(summary=summary, rows=merged), f, indent=1)
    print(f"merged -> {args.out}", file=sys.stderr)
    if acc_fail:
        print(f"accuracy failures: {acc_fail}", file=sys.stderr)
    return 1 if acc_fail else 0


if __name__ == "__main__":
    sys.exit(main())
