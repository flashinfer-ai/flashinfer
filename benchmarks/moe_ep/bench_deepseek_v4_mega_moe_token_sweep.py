#!/usr/bin/env python3
"""Sweep DeepSeek V4 MegaMoE warm runtime across token sizes (vLLM vs flashinfer).

Runs ``bench_deepseek_v4_mega_moe_vLLM.py`` and
``bench_deepseek_v4_mega_moe_flashinfer.py`` via ``torchrun`` for each token
count, parses ``steady_avg_ms`` (post-warmup average), and prints a summary table.

Example (4 GPUs on one node):
    python benchmarks/moe_ep/bench_deepseek_v4_mega_moe_token_sweep.py \\
        --nproc-per-node 4

    python benchmarks/moe_ep/bench_deepseek_v4_mega_moe_token_sweep.py \\
        --nproc-per-node 4 --token-sizes 2048 4096 8192 16384 \\
        --warmup 10 --repeat 50 --verbose
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent.parent

DEFAULT_TOKEN_SIZES = (2048, 4096, 8192, 16384)

BACKEND_SCRIPTS = {
    "vllm": "bench_deepseek_v4_mega_moe_vLLM.py",
    "flashinfer": "bench_deepseek_v4_mega_moe_flashinfer.py",
}

STEADY_AVG_MS_RE = re.compile(r"steady_avg_ms=([\d.]+)")


@dataclass(frozen=True)
class SweepResult:
    backend: str
    num_tokens: int
    steady_avg_ms: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep MegaMoE warm runtime (steady_avg_ms) for vLLM vs flashinfer "
            "across token sizes."
        ),
    )
    parser.add_argument(
        "--token-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_TOKEN_SIZES),
        help="Token counts to benchmark (default: 2048 4096 8192 16384).",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=4,
        help="GPUs per node passed to torchrun (default: 4).",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Warmup iterations forwarded to each backend benchmark.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=20,
        help="Timed iterations forwarded to each backend benchmark.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=tuple(BACKEND_SCRIPTS),
        default=("vllm", "flashinfer"),
        help="Backends to run (default: vllm flashinfer).",
    )
    parser.add_argument(
        "--torchrun",
        default="torchrun",
        help="torchrun executable (default: torchrun).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print full stdout/stderr from each benchmark subprocess.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )
    return parser.parse_args()


def _run_benchmark(
    *,
    backend: str,
    num_tokens: int,
    nproc_per_node: int,
    warmup: int,
    repeat: int,
    torchrun: str,
    verbose: bool,
    dry_run: bool,
) -> SweepResult:
    script = _BENCH_DIR / BACKEND_SCRIPTS[backend]
    cmd = [
        torchrun,
        f"--nproc_per_node={nproc_per_node}",
        str(script),
        "--num-tokens",
        str(num_tokens),
        "--num-max-tokens",
        str(num_tokens),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--no-cold-start",
    ]

    print(
        f"[sweep] backend={backend} num_tokens={num_tokens}\n"
        f"        cmd: {' '.join(cmd)}",
        flush=True,
    )

    if dry_run:
        return SweepResult(backend=backend, num_tokens=num_tokens, steady_avg_ms=0.0)

    proc = subprocess.run(
        cmd,
        cwd=_REPO_ROOT,
        capture_output=True,
        text=True,
    )

    if verbose:
        if proc.stdout:
            print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
        if proc.stderr:
            print(proc.stderr, end="" if proc.stderr.endswith("\n") else "\n", file=sys.stderr)

    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Benchmark failed: backend={backend} num_tokens={num_tokens} "
            f"exit_code={proc.returncode}"
        )

    match = STEADY_AVG_MS_RE.search(proc.stdout)
    if match is None:
        print(proc.stdout)
        print(proc.stderr, file=sys.stderr)
        raise RuntimeError(
            f"Could not parse steady_avg_ms from output: "
            f"backend={backend} num_tokens={num_tokens}"
        )

    return SweepResult(
        backend=backend,
        num_tokens=num_tokens,
        steady_avg_ms=float(match.group(1)),
    )


def _print_summary_table(
    results: list[SweepResult],
    *,
    token_sizes: list[int],
    backends: list[str],
    nproc_per_node: int,
    warmup: int,
    repeat: int,
) -> None:
    by_key = {(r.backend, r.num_tokens): r.steady_avg_ms for r in results}

    print()
    print("DeepSeek V4 MegaMoE warm runtime sweep (steady_avg_ms)")
    print(
        f"  world_size={nproc_per_node} warmup={warmup} repeat={repeat} "
        f"num_max_tokens=num_tokens"
    )
    print()

    header = ["num_tokens"]
    for backend in backends:
        header.append(f"{backend} (ms)")
    if "vllm" in backends and "flashinfer" in backends:
        header.append("flashinfer/vLLM")

    col_widths = [max(len(h), 10) for h in header]
    for idx, num_tokens in enumerate(token_sizes):
        col_widths[0] = max(col_widths[0], len(str(num_tokens)))

    def fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.rjust(width) for cell, width in zip(cells, col_widths))

    print(fmt_row(header))
    print(fmt_row(["-" * width for width in col_widths]))

    for num_tokens in token_sizes:
        row = [str(num_tokens)]
        vllm_ms = by_key.get(("vllm", num_tokens))
        flashinfer_ms = by_key.get(("flashinfer", num_tokens))

        for backend in backends:
            ms = by_key.get((backend, num_tokens))
            row.append(f"{ms:.3f}" if ms is not None else "n/a")

        if "vllm" in backends and "flashinfer" in backends:
            if vllm_ms is not None and flashinfer_ms is not None and vllm_ms > 0:
                row.append(f"{flashinfer_ms / vllm_ms:.3f}")
            else:
                row.append("n/a")

        print(fmt_row(row))


def main() -> int:
    args = parse_args()
    token_sizes = list(dict.fromkeys(args.token_sizes))
    backends = list(dict.fromkeys(args.backends))

    results: list[SweepResult] = []
    for num_tokens in token_sizes:
        for backend in backends:
            results.append(
                _run_benchmark(
                    backend=backend,
                    num_tokens=num_tokens,
                    nproc_per_node=args.nproc_per_node,
                    warmup=args.warmup,
                    repeat=args.repeat,
                    torchrun=args.torchrun,
                    verbose=args.verbose,
                    dry_run=args.dry_run,
                )
            )

    if args.dry_run:
        print("\n[dry-run] Skipping summary table (no benchmark output).")
        return 0

    _print_summary_table(
        results,
        token_sizes=token_sizes,
        backends=backends,
        nproc_per_node=args.nproc_per_node,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
