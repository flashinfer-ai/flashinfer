"""Render PR markdown tables from caac.py summary.json files.

Usage: python make_pr_tables.py BASE_SUMMARY.json [--override JOB=SUMMARY.json ...] [--repeat JOB=SUMMARY.json]
Rows of an overridden job come from its override summary (e.g. a rerun on the final source).
"""

import argparse
import ast
import collections
import json
import math

ORDER = ["tp-v3", "tp-kimi-k3", "tp-v41-flash", "tp-nemo3-ultra", "ep-v3", "ep-kimi-k3", "ep-v41-flash", "ep-nemo3-ultra"]
NAMES = {
    "tp-v3": "V3 TP8", "tp-kimi-k3": "K3 TP8", "tp-v41-flash": "V41F TP4", "tp-nemo3-ultra": "N3U TP8",
    "ep-v3": "V3 EP8", "ep-kimi-k3": "K3 EP8", "ep-v41-flash": "V41F EP4", "ep-nemo3-ultra": "N3U EP8",
}
FC2_K = {"tp-v3": 256, "tp-kimi-k3": 384, "tp-v41-flash": 576, "tp-nemo3-ultra": 640,
         "ep-v3": 2048, "ep-kimi-k3": 3072, "ep-v41-flash": 2304, "ep-nemo3-ultra": 5120}
FC2_M = {"v3": 7168, "kimi-k3": 3584, "v41-flash": 5120, "nemo3-ultra": 2048}


def unit_size(job, tactic_list):
    """m_tiles_per_unit the candidate selects for this cell's GEMM2 tactic (mirrors the kernel rule)."""
    if not tactic_list or tactic_list == ["None"]:
        return 1
    (m, n, kt), (cm, cn), _ = ast.literal_eval(tactic_list[0])[1]
    if m != 128:
        return 1
    model = job.split("-", 1)[1]
    hidden = FC2_M[model]
    if hidden % (128 * cm):
        return 1
    k_tiles = -(-FC2_K[job] // kt)
    cluster_tiles = hidden // (128 * cm)
    g = 1
    for t in range(2, 4 // k_tiles + 1):
        if cluster_tiles % t == 0 and 2 * t * n + 2 * 128 <= 512:
            g = t
    return g


def fmt_tactic(t):
    if not t or t == ["None"]:
        return "-1"
    out = []
    for gemm in ast.literal_eval(t[0]):
        (m, n, _), (cm, cn), _ = gemm
        out.append(f"{m}/{n} c{cm}{cn}")
    return out[0] if out[0] == out[1] else f"{out[0]} ; {out[1]}"


def geo(xs):
    return 100 * (math.exp(sum(math.log(r["control_us"] / r["candidate_us"]) for r in xs) / len(xs)) - 1) if xs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--repeat", action="append", default=[])
    a = p.parse_args()
    rows = [r for r in json.load(open(a.base))]
    for spec in a.override:
        job, path = spec.split("=", 1)
        rows = [r for r in rows if r["job"] != job] + [r for r in json.load(open(path)) if r["job"] == job]
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["campaign"], r["job"])].append(r)

    print("| Job (GEMM2 K) | Units | Frozen 4-512 | Native 4-512 | Frozen 4096-16384 | Native 4096-16384 | 1 / 2 tokens (frozen) |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for job in ORDER:
        fr, na = sorted(by[("frozen", job)], key=lambda r: r["tokens"]), sorted(by[("native", job)], key=lambda r: r["tokens"])
        units = sorted({unit_size(job, r["candidate_tactic"]) for r in fr if 4 <= r["tokens"] <= 512})
        t12 = " / ".join(f"{r['speedup_pct']:+.2f}%" for r in fr if r["tokens"] <= 2)
        print(f"| {NAMES[job]} ({FC2_K[job]}) | {','.join(map(str, units))} | {geo([r for r in fr if 4 <= r['tokens'] <= 512]):+.2f}% | "
              f"{geo([r for r in na if 4 <= r['tokens'] <= 512]):+.2f}% | {geo([r for r in fr if r['tokens'] >= 4096]):+.2f}% | "
              f"{geo([r for r in na if r['tokens'] >= 4096]):+.2f}% | {t12} |")
    for camp in ("frozen", "native"):
        sel = [r for r in rows if r["campaign"] == camp]
        for scope, pred in (("TP", lambda r: r["job"].startswith("tp")), ("EP", lambda r: r["job"].startswith("ep"))):
            d = [r for r in sel if pred(r) and 4 <= r["tokens"] <= 512]
            pf = [r for r in sel if pred(r) and r["tokens"] >= 4096]
            print(f"\n{camp} {scope}: decode 4-512 {geo(d):+.2f}% ({len(d)} cells), prefill {geo(pf):+.2f}% ({len(pf)} cells)", end="")
    print("\n")
    # Null cells: frozen, one-tile units in the candidate (byte-identical kernels).
    null = [r for r in rows if r["campaign"] == "frozen" and r["tokens"] >= 4 and unit_size(r["job"], r["candidate_tactic"]) == 1]
    for name, xs in (("decode 4-512", [r for r in null if r["tokens"] <= 512]), ("prefill", [r for r in null if r["tokens"] >= 4096])):
        s = sorted(r["speedup_pct"] for r in xs)
        print(f"null cells (frozen, one-tile units) {name}: n={len(xs)}, geomean {geo(xs):+.2f}%, range {s[0]:+.2f}% .. {s[-1]:+.2f}%")
    for spec in a.repeat:
        job, path = spec.split("=", 1)
        rep = [r for r in json.load(open(path)) if r["job"] == job]
        for camp in ("frozen", "native"):
            d = [r for r in rep if r["campaign"] == camp and 4 <= r["tokens"] <= 512]
            print(f"repeat {job} {camp} decode 4-512: {geo(d):+.2f}%")
    for camp in ("frozen", "native"):
        print(f"\n<details><summary>All cells, {camp} tactics (median us per phase; speedup = sqrt(C1*C2)/sqrt(A1*A2) - 1)</summary>\n")
        print("| Job | Tokens | GEMM1 ; GEMM2 tactic (candidate) | Units | Control us | Candidate us | Speedup | C1 / A1 / A2 / C2 us |")
        print("|---|---:|---|---:|---:|---:|---:|---|")
        for job in ORDER:
            for r in sorted(by[(camp, job)], key=lambda r: r["tokens"]):
                print(f"| {NAMES[job]} | {r['tokens']} | {fmt_tactic(r['candidate_tactic'])} | {unit_size(job, r['candidate_tactic'])} | "
                      f"{r['control_us']:.3f} | {r['candidate_us']:.3f} | {r['speedup_pct']:+.2f}% | "
                      f"{r['c1']:.3f} / {r['a1']:.3f} / {r['a2']:.3f} / {r['c2']:.3f} |")
        print("\n</details>\n")


if __name__ == "__main__":
    main()
