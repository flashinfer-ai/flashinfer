"""End-to-end smoke test for the FlashInfer LLM example.

Runs ``generate.py`` in two fresh subprocesses and verifies integration-level
invariants that kernel unit tests cannot see:

1. **JIT cache works across processes** — the second run must compile zero
   modules (``jit_builds_total == 0``, measured as "built artifact changed";
   ninja dependency scans still run by design). If this fails, the on-disk
   kernel cache is broken and every user process pays full recompilation.
2. **No steady-state recompiles** — decode steps after the first must never
   trigger a JIT build (``jit_builds_steady == 0``), in both runs.
3. **Greedy determinism** — both runs must produce identical token ids.
4. **Liveness** — every request produced at least one token.

Usage:
    python smoke_test.py --model-id Qwen/Qwen3-0.6B --max-tokens 16

Exit code 0 on pass, 1 on failure (with both runs' output dumped).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def run_generate(args: argparse.Namespace, label: str, sampling: bool = False) -> Dict:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate.py"),
        "--model-id",
        args.model_id,
        "--max-tokens",
        str(args.max_tokens),
    ]
    if sampling:
        cmd += [
            "--temperature",
            str(args.temperature),
            "--top-k",
            str(args.top_k),
            "--top-p",
            str(args.top_p),
            "--seed",
            str(args.seed),
            "--check-sampling",
        ]
    else:
        cmd += ["--temperature", "0.0"]
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        print(f"=== {label}: generate.py failed (rc={proc.returncode}) ===")
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        sys.exit(1)
    smoke: Dict = {"elapsed": elapsed, "stdout": proc.stdout}
    tokens: List[str] = []
    for line in proc.stdout.splitlines():
        if not line.startswith("[smoke] "):
            continue
        key, _, value = line[len("[smoke] ") :].partition("=")
        if key.startswith("tokens_"):
            tokens.append(value)
        else:
            smoke[key] = value
    smoke["tokens"] = tokens
    print(
        f"{label}: {elapsed:.1f}s, jit_builds_total={smoke.get('jit_builds_total')}, "
        f"jit_build_calls={smoke.get('jit_build_calls')}, "
        f"jit_builds_steady={smoke.get('jit_builds_steady')}, "
        f"{len(tokens)} completions"
    )
    return smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--skip-sampling",
        action="store_true",
        help="Skip the top-k/top-p sampling runs (greedy checks only)",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Per-run timeout in seconds (cold JIT compile can be slow)",
    )
    args = parser.parse_args()

    print(f"smoke test: {args.model_id}, greedy, {args.max_tokens} new tokens")
    first = run_generate(args, "run 1 (warmup, compiles allowed)")
    second = run_generate(args, "run 2 (must be fully cached)")

    failures: List[str] = []

    if second.get("jit_builds_total") != "0":
        failures.append(
            "JIT cache miss on warm run: second process compiled "
            f"{second.get('jit_builds_total')} modules "
            f"({second.get('jit_builds_names')}) — the on-disk kernel cache "
            "is not being reused"
        )
    for label, run in (("run 1", first), ("run 2", second)):
        if run.get("jit_builds_steady") != "0":
            failures.append(
                f"{label}: {run.get('jit_builds_steady')} JIT builds during "
                "steady-state decode — kernels are recompiling across steps"
            )
    if first["tokens"] != second["tokens"]:
        failures.append(
            "greedy decode is not deterministic across runs:\n"
            f"  run 1: {first['tokens']}\n  run 2: {second['tokens']}"
        )
    if not first["tokens"] or any(not t for t in first["tokens"]):
        failures.append("empty completion(s) in run 1")

    # ---- top-k/top-p sampling ----
    # Greedy never touches flashinfer.sampling at all (sample_tokens
    # short-circuits to argmax), so without these runs the sampling and topk
    # JIT modules are never even compiled.
    third = fourth = None
    if not args.skip_sampling:
        print(
            f"\nsampling: temperature={args.temperature} top_k={args.top_k} "
            f"top_p={args.top_p} seed={args.seed}"
        )
        third = run_generate(args, "run 3 (sampling, compiles allowed)", sampling=True)
        fourth = run_generate(args, "run 4 (sampling, must be cached)", sampling=True)

        if int(third.get("sample_draws", 0)) == 0:
            failures.append("sampling runs produced no sampled tokens to audit")
        for label, run in (("run 3", third), ("run 4", fourth)):
            if run.get("sample_violations") != "0":
                failures.append(
                    f"{label}: {run.get('sample_violations')} of "
                    f"{run.get('sample_draws')} sampled tokens fell OUTSIDE the "
                    "top-k ∩ nucleus support — the sampling kernel is wrong"
                )
            if run.get("sample_out_of_range") != "0":
                failures.append(
                    f"{label}: {run.get('sample_out_of_range')} sampled token ids "
                    "were outside [0, vocab)"
                )
            if run.get("sample_replay_match") != "1":
                failures.append(
                    f"{label}: re-sampling the same logits with a freshly seeded "
                    "generator gave different tokens — seeded sampling is not "
                    "reproducible"
                )
            if run.get("sample_perreq_violations") != "0":
                failures.append(
                    f"{label}: {run.get('sample_perreq_violations')} violations with "
                    "per-request tensor top_k/top_p (non-fast-path kernels)"
                )
            # Guard against a vacuous membership check: with top_k=0/top_p=1.0
            # the admissible set is the whole vocabulary and it can never fail.
            mean_allowed = float(run.get("sample_allowed_mean", 0))
            if args.top_k > 0 and mean_allowed > 2 * args.top_k:
                failures.append(
                    f"{label}: mean admissible set {mean_allowed:.1f} is far wider "
                    f"than top_k={args.top_k} — the support check is not selective"
                )
        # A sampler degenerated to argmax stays inside the support, so
        # membership cannot see it; require divergence when enough is expected.
        expected = float(third.get("sample_expected_divergences", 0))
        got = int(third.get("sample_divergences", 0))
        if expected >= 10.0 and got == 0:
            failures.append(
                f"run 3: sampling never diverged from greedy in {third.get('sample_draws')} "
                f"draws though ~{expected:.1f} divergences were expected — the "
                "sampler looks degenerate (dead RNG or unconditional argmax)"
            )
        elif expected < 10.0:
            print(
                f"  note: anti-argmax check INCONCLUSIVE "
                f"(only ~{expected:.1f} divergences expected; raise --temperature)"
            )
        if fourth.get("jit_builds_total") != "0":
            failures.append(
                "JIT cache miss on warm sampling run: "
                f"{fourth.get('jit_builds_names')} recompiled"
            )
        if third["tokens"] != fourth["tokens"]:
            failures.append(
                "seeded sampling is not deterministic across processes:\n"
                f"  run 3: {third['tokens']}\n  run 4: {fourth['tokens']}"
            )

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        print("\n=== run 1 output ===\n" + first["stdout"])
        print("\n=== run 2 output ===\n" + second["stdout"])
        if third is not None:
            print("\n=== run 3 output ===\n" + third["stdout"])
            print("\n=== run 4 output ===\n" + fourth["stdout"])
        sys.exit(1)
    print(
        f"\nPASS ({first['elapsed']:.1f}s cold -> {second['elapsed']:.1f}s warm; "
        "cache reuse, steady-state stability, and determinism verified"
        + ("" if args.skip_sampling else "; sampling support audited")
        + ")"
    )


if __name__ == "__main__":
    main()
