"""PR 7290-style same-GPU C->A->A->C W4A16 campaign on one 8-GPU devbox.

prepare   Snapshot the control and candidate trees into the campaign directory (immutable
          copies, own flashinfer/data links) and hash their Python sources.
job       Run one job (model, ep, tp) on one GPU: tune control and candidate (fresh autotune,
          persistent cache), then C->A->A->C for two campaigns:
            frozen: both roles replay the control's tuned winners (causal screen);
            native: each role replays its own tuned winners (what users get).
          Every phase is a fresh process with --no-autotune and a private copy of the cache.
summary   Aggregate: control = sqrt(C1*C2), candidate = sqrt(A1*A2) of per-phase medians;
          speedup = control / candidate - 1; geomeans over tokens >= 4 by band.
"""

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path("/scratch/ziangli/flashinfer-w4a16-tp-low-concurrency")
TOKENS = "1,2,4,8,16,32,64,128,256,512,4096,8192,16384"
JOBS = {
    "ep-v3": ("v3", 8, 1),
    "tp-v3": ("v3", 1, 8),
    "ep-v41-flash": ("v41-flash", 4, 1),
    "tp-v41-flash": ("v41-flash", 1, 4),
    "ep-kimi-k3": ("kimi-k3", 8, 1),
    "tp-kimi-k3": ("kimi-k3", 1, 8),
    "ep-nemo3-ultra": ("nemo3-ultra", 8, 1),
    "tp-nemo3-ultra": ("nemo3-ultra", 1, 8),
}
# (role, tactic source). Both candidate pairs are symmetric about the center:
# frozen = phases 1 and 4 (control's winners), native = phases 2 and 3 (own winners).
PHASES = (
    ("control", "control"),
    ("candidate", "control"),
    ("candidate", "candidate"),
    ("candidate", "candidate"),
    ("candidate", "control"),
    ("control", "control"),
)
CAMPAIGNS = {"frozen": (0, 1, 4, 5), "native": (0, 2, 3, 5)}


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def tree_hash(tree: Path):
    h = hashlib.sha256()
    for p in sorted((tree / "flashinfer").rglob("*.py")):
        h.update(str(p.relative_to(tree)).encode())
        h.update(p.read_bytes())
    for p in sorted((tree / "benchmarks").glob("bench_moe_deepseek.py")):
        h.update(p.read_bytes())
    return h.hexdigest()


def prepare(campaign: Path):
    campaign.mkdir(parents=True, exist_ok=True)
    meta = {}
    for role, src in (("control", ROOT / "control/flashinfer"), ("candidate", ROOT / "flashinfer")):
        dst = campaign / role / "flashinfer"
        dst.mkdir(parents=True, exist_ok=True)
        sh(["rsync", "-a", "--delete", "--exclude=__pycache__", "--exclude=*.egg-info",
            "--exclude=flashinfer/data", f"{src}/", f"{dst}/"])
        data = dst / "flashinfer" / "data"
        data.mkdir(exist_ok=True)
        for name, target in (("csrc", "csrc"), ("include", "include"),
                             ("cccl", "3rdparty/cccl"), ("cutlass", "3rdparty/cutlass"),
                             ("spdlog", "3rdparty/spdlog")):
            link = data / name
            if link.is_symlink() or link.exists():
                link.unlink()
            link.symlink_to(dst / target)
        meta[role] = {"tree": str(dst), "sha256": tree_hash(dst)}
    (campaign / "sources.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


def env_for(campaign: Path, role: str, gpu: int, phase_dir: Path):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("FLASHINFER_", "TRTLLM_", "CUTE_DSL_", "FI_", "CUDA_VISIBLE"))}
    tree = campaign / role / "flashinfer"
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        PYTHONPATH=str(tree),
        FLASHINFER_CUTE_DSL_STRICT_CACHE_KEY="1",
        FLASHINFER_CUDA_ARCH_LIST="10.3a",
        FLASHINFER_DIST_AWARE_AUTOTUNE="0",
        TRTLLM_ENABLE_PDL="1",
        FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH="0",
        TRTLLM_DISABLE_FP4_QUANT_FAST_MATH="0",
        MAX_JOBS="4",
        OMP_NUM_THREADS="1",
        PYTHONUNBUFFERED="1",
        FLASHINFER_WORKSPACE_BASE=str(campaign / "cache" / role / "ws"),
        CUTE_DSL_CACHE_DIR=str(campaign / "cache" / role / f"cute-{gpu}"),
        TRITON_CACHE_DIR=str(phase_dir / "triton"),
        BENCH_RECORD=str(phase_dir / "record.json"),
        # Shared SHA-verified artifact cache (populated earlier); never download during
        # timing. The artifact server is not part of the measurement.
        FLASHINFER_CUBIN_DIR="/root/.cache/flashinfer/cubins",
        FLASHINFER_NO_DOWNLOAD="1",
    )
    return env, tree


def bench(campaign, role, gpu, phase_dir: Path, model, ep, tp, cache: Path, tune: bool):
    phase_dir.mkdir(parents=True, exist_ok=True)
    env, tree = env_for(campaign, role, gpu, phase_dir)
    cmd = [sys.executable, "-u", str(ROOT / "tools/bench_w4a16_only.py"),
           "--functional-api", "--model", model, "--ep", str(ep), "--tp", str(tp),
           "--quant-family", "nvfp4", "--swap-ab", "--no-fused-finalize",
           "--include-activation-quant", "--num-tokens", TOKENS, "--warmup", "10",
           "--iters", "100", "--output-format", "both", "--cache", str(cache)]
    if not tune:
        cmd.append("--no-autotune")
    with open(phase_dir / "stdout.log", "w") as out:
        rc = subprocess.run(cmd, cwd=tree, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
    (phase_dir / "exit").write_text(f"{rc}\n")
    if rc != 0:
        raise SystemExit(f"phase failed rc={rc}: {phase_dir}")


def warm(campaign: Path, gpu: int):
    """Serially build each role's JIT modules and fetch artifacts before parallel jobs."""
    for role in ("control", "candidate"):
        for model, ep, tp in (("v3", 1, 8), ("kimi-k3", 1, 8)):
            pd = campaign / "warm" / f"{role}-{model}"
            pd.mkdir(parents=True, exist_ok=True)
            env, tree = env_for(campaign, role, gpu, pd)
            cmd = [sys.executable, "-u", str(ROOT / "tools/bench_w4a16_only.py"),
                   "--functional-api", "--model", model, "--ep", str(ep), "--tp", str(tp),
                   "--quant-family", "nvfp4", "--swap-ab", "--no-fused-finalize",
                   "--include-activation-quant", "--num-tokens", "1", "--warmup", "1",
                   "--iters", "2", "--no-autotune", "--output-format", "csv"]
            with open(pd / "stdout.log", "w") as out:
                rc = subprocess.run(cmd, cwd=tree, env=env, stdout=out, stderr=subprocess.STDOUT).returncode
            print("warm", role, model, "rc", rc, flush=True)
            if rc:
                raise SystemExit(rc)


def run_job(campaign: Path, job: str, gpu: int):
    model, ep, tp = JOBS[job]
    jd = campaign / "jobs" / job
    jd.mkdir(parents=True, exist_ok=True)
    caches = {}
    for role in ("control", "candidate"):
        cache = jd / f"tuned-cache-{role}"
        if not (jd / f"tune-{role}" / "exit").exists():
            shutil.rmtree(cache, ignore_errors=True)
            bench(campaign, role, gpu, jd / f"tune-{role}", model, ep, tp, cache, tune=True)
        caches[role] = cache
    for i, (role, tactics) in enumerate(PHASES):
        pd = jd / "phases" / f"{i}-{role}-{tactics}-winners"
        if (pd / "exit").exists() and (pd / "exit").read_text().strip() == "0":
            continue
        src = caches[tactics]
        cache = pd / "cache"
        shutil.rmtree(cache, ignore_errors=True)
        shutil.copytree(src, cache)
        bench(campaign, role, gpu, pd, model, ep, tp, cache, tune=False)
        # Replay must not rewrite the cache.
        same = subprocess.run(["diff", "-r", str(src), str(cache)], capture_output=True)
        (pd / "cache-unchanged").write_text("yes\n" if same.returncode == 0 else same.stdout.decode())
    print("job done", job, flush=True)


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def phase_medians(pd: Path):
    rec = json.loads((pd / "record.json").read_text())
    out = {}
    for n, lists in rec["samples_ms"].items():
        samples = [s for lst in lists for s in lst]
        out[int(n)] = (median(samples), len(samples))
    return out, rec["tactics"]


def summary(campaign: Path):
    rows = []
    for job in JOBS:
        jd = campaign / "jobs" / job
        all_phases = [jd / "phases" / f"{i}-{role}-{tactics}-winners" for i, (role, tactics) in enumerate(PHASES)]
        for camp, idx in CAMPAIGNS.items():
            phases = [all_phases[i] for i in idx]
            if not all((p / "record.json").exists() for p in phases):
                continue
            med = [phase_medians(p) for p in phases]
            for n in sorted(med[0][0]):
                c1, c2 = med[0][0][n][0], med[3][0][n][0]
                a1, a2 = med[1][0][n][0], med[2][0][n][0]
                ctrl = math.sqrt(c1 * c2)
                cand = math.sqrt(a1 * a2)
                rows.append(dict(
                    job=job, campaign=camp, tokens=n, control_us=1000 * ctrl,
                    candidate_us=1000 * cand, speedup_pct=100 * (ctrl / cand - 1),
                    c1=1000 * c1, a1=1000 * a1, a2=1000 * a2, c2=1000 * c2,
                    samples=[med[i][0][n][1] for i in range(4)],
                    control_tactic=med[0][1].get(str(n)), candidate_tactic=med[1][1].get(str(n)),
                ))
    (campaign / "summary.json").write_text(json.dumps(rows, indent=1) + "\n")
    for camp in ("frozen", "native"):
        sel = [r for r in rows if r["campaign"] == camp]
        if not sel:
            continue
        print(f"\n### {camp}")
        print("| job | tokens | control us | candidate us | speedup | C1 / A1 / A2 / C2 us |")
        print("|---|---:|---:|---:|---:|---|")
        for r in sel:
            print(f"| {r['job']} | {r['tokens']} | {r['control_us']:.3f} | {r['candidate_us']:.3f} | "
                  f"{r['speedup_pct']:+.2f}% | {r['c1']:.3f} / {r['a1']:.3f} / {r['a2']:.3f} / {r['c2']:.3f} |")
        for band, lo, hi in (("decode 4-512", 4, 512), ("prefill 4096-16384", 4096, 16384), ("all >=4", 4, 10**9)):
            for scope in ("tp", "ep", "all"):
                xs = [r for r in sel if lo <= r["tokens"] <= hi and (scope == "all" or r["job"].startswith(scope))]
                if xs:
                    g = math.exp(sum(math.log(r["control_us"] / r["candidate_us"]) for r in xs) / len(xs))
                    print(f"geomean {camp} {scope} {band}: {100*(g-1):+.2f}% over {len(xs)} cells")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare"); a.add_argument("campaign", type=Path)
    b = sub.add_parser("job"); b.add_argument("campaign", type=Path); b.add_argument("job"); b.add_argument("gpu", type=int)
    c = sub.add_parser("summary"); c.add_argument("campaign", type=Path)
    w = sub.add_parser("warm"); w.add_argument("campaign", type=Path); w.add_argument("gpu", type=int)
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.campaign)
    elif args.cmd == "warm":
        warm(args.campaign, args.gpu)
    elif args.cmd == "job":
        run_job(args.campaign, args.job, args.gpu)
    else:
        summary(args.campaign)
