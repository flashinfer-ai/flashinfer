"""End-to-end smoke test for the FlashInfer LLM example.

Runs ``generate.py`` in two fresh subprocesses and verifies integration-level
invariants that kernel unit tests cannot see:

1. **JIT cache works across processes** — the second run must compile zero
   modules (``jit_builds_total == 0``). If this fails, the on-disk kernel
   cache is broken and every user process pays full recompilation.
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


def run_generate(args: argparse.Namespace, label: str) -> Dict:
    cmd = [
        sys.executable,
        str(Path(__file__).parent / "generate.py"),
        "--model-id",
        args.model_id,
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        "0.0",
    ]
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
        f"jit_builds_steady={smoke.get('jit_builds_steady')}, "
        f"{len(tokens)} completions"
    )
    return smoke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="Qwen/Qwen3-0.6B")
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

    if failures:
        print("\nFAIL:")
        for f in failures:
            print(f"  - {f}")
        print("\n=== run 1 output ===\n" + first["stdout"])
        print("\n=== run 2 output ===\n" + second["stdout"])
        sys.exit(1)
    print(
        f"\nPASS ({first['elapsed']:.1f}s cold -> {second['elapsed']:.1f}s warm; "
        "cache reuse, steady-state stability, and determinism verified)"
    )


if __name__ == "__main__":
    main()
