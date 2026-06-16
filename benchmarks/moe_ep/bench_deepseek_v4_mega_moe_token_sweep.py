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

Example (16 GPUs = 4 nodes x 4 GPUs, inside a Slurm allocation):
    python benchmarks/moe_ep/bench_deepseek_v4_mega_moe_token_sweep.py \\
        --nnodes 4 --nproc-per-node 4

Example (nested inside ``srun --overlap --container-image=...`` on one node):
    export MASTER_ADDR="$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)"

    python benchmarks/moe_ep/bench_deepseek_v4_mega_moe_token_sweep.py \\
        --nnodes 4 --nproc-per-node 4 \\
        --srun-extra-args \\
            --container-image="$IMG" \\
            --container-mounts=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik:/lustre/fsw/coreai_libraries_cudnn/mhoqueanik \\
            --container-workdir=/lustre/fsw/coreai_libraries_cudnn/mhoqueanik
"""

from __future__ import annotations

import argparse
import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

_BENCH_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _BENCH_DIR.parent.parent

DEFAULT_TOKEN_SIZES = (2048, 4096, 8192, 16384)
DEFAULT_MASTER_PORT = 29500

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


@dataclass(frozen=True)
class LaunchConfig:
    nnodes: int
    nproc_per_node: int
    master_addr: str
    master_port: int
    use_srun: bool
    srun_overlap: bool
    srun_extra_args: tuple[str, ...]
    torchrun: str
    srun: str

    @property
    def world_size(self) -> int:
        return self.nnodes * self.nproc_per_node


def _resolve_master_addr(explicit: str | None) -> str:
    if explicit:
        return explicit
    if master := os.environ.get("MASTER_ADDR"):
        return master
    nodelist = os.environ.get("SLURM_NODELIST")
    if nodelist:
        try:
            result = subprocess.run(
                ["scontrol", "show", "hostnames", nodelist],
                capture_output=True,
                text=True,
                check=True,
            )
            first_node = result.stdout.strip().split("\n")[0]
            if first_node:
                return first_node
        except (subprocess.CalledProcessError, FileNotFoundError):
            if "[" in nodelist:
                base = nodelist.split("[", maxsplit=1)[0]
                nums = nodelist.split("[", maxsplit=1)[1].split("]", maxsplit=1)[0]
                first_num = nums.split(",", maxsplit=1)[0].split("-", maxsplit=1)[0]
                return f"{base}{first_num}"
            return nodelist.split(",", maxsplit=1)[0]
    return "127.0.0.1"


def _resolve_srun(explicit: str) -> str:
    candidates: list[str] = []
    if explicit_env := os.environ.get("SRUN"):
        candidates.append(explicit_env)
    candidates.extend([explicit, "srun", "/usr/bin/srun"])

    seen: set[str] = set()
    for cand in candidates:
        if not cand or cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
        if found := shutil.which(cand):
            return found

    raise SystemExit(
        "ERROR: srun not found inside this environment.\n"
        "  Multi-node sweeps need the Slurm client. Either:\n"
        "    1) Exit the container and re-run ./open_docker.sh (bind-mounts srun), or\n"
        "    2) Pass --srun /path/to/srun if Slurm is mounted elsewhere."
    )


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
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes for torchrun (default: 1).",
    )
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=4,
        help="GPUs per node passed to torchrun (default: 4).",
    )
    parser.add_argument(
        "--master-addr",
        default=None,
        help=(
            "torchrun master address. Defaults to $MASTER_ADDR, then the first "
            "host in $SLURM_NODELIST, then 127.0.0.1."
        ),
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=DEFAULT_MASTER_PORT,
        help=f"torchrun master port (default: {DEFAULT_MASTER_PORT}).",
    )
    parser.add_argument(
        "--use-srun",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Launch multi-node benchmarks via srun (default: on when --nnodes > 1)."
        ),
    )
    parser.add_argument(
        "--srun-overlap",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Pass --overlap --jobid=$SLURM_JOB_ID to nested srun launches "
            "(default: on when SLURM_JOB_ID is set)."
        ),
    )
    parser.add_argument(
        "--srun-extra-args",
        nargs=argparse.REMAINDER,
        default=(),
        help=(
            "Extra srun flags inserted before --nodes (e.g. container-image, "
            "container-mounts). Use '--' before this flag if needed."
        ),
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
        "--srun",
        default="srun",
        help="srun executable for multi-node launches (default: srun).",
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


def _launch_config_from_args(args: argparse.Namespace) -> LaunchConfig:
    if args.nnodes < 1:
        raise SystemExit("ERROR: --nnodes must be >= 1")
    if args.nproc_per_node < 1:
        raise SystemExit("ERROR: --nproc-per-node must be >= 1")

    master_addr = _resolve_master_addr(args.master_addr)
    use_srun = args.nnodes > 1 if args.use_srun is None else args.use_srun
    srun_extra_args = tuple(args.srun_extra_args)
    if srun_extra_args[:1] == ("--",):
        srun_extra_args = srun_extra_args[1:]

    slurm_job_id = os.environ.get("SLURM_JOB_ID")
    if args.srun_overlap is None:
        srun_overlap = slurm_job_id is not None
    else:
        srun_overlap = args.srun_overlap

    if args.nnodes > 1 and master_addr in {"127.0.0.1", "localhost"}:
        print(
            "WARNING: multi-node launch with loopback master address; "
            "pass --master-addr or set MASTER_ADDR / SLURM_NODELIST.",
            file=sys.stderr,
            flush=True,
        )

    srun_bin = args.srun
    if use_srun and args.nnodes > 1:
        srun_bin = _resolve_srun(args.srun)

    return LaunchConfig(
        nnodes=args.nnodes,
        nproc_per_node=args.nproc_per_node,
        master_addr=master_addr,
        master_port=args.master_port,
        use_srun=use_srun,
        srun_overlap=srun_overlap,
        srun_extra_args=srun_extra_args,
        torchrun=args.torchrun,
        srun=srun_bin,
    )


def _benchmark_script_args(
    *,
    script: Path,
    num_tokens: int,
    warmup: int,
    repeat: int,
) -> list[str]:
    return [
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


def _build_launch_command(
    *,
    launch: LaunchConfig,
    script: Path,
    num_tokens: int,
    warmup: int,
    repeat: int,
) -> list[str]:
    bench_args = _benchmark_script_args(
        script=script,
        num_tokens=num_tokens,
        warmup=warmup,
        repeat=repeat,
    )

    if launch.nnodes == 1:
        return [
            launch.torchrun,
            f"--nproc_per_node={launch.nproc_per_node}",
            *bench_args,
        ]

    torchrun_parts = [
        shlex.quote(launch.torchrun),
        f"--nnodes={launch.nnodes}",
        f"--nproc_per_node={launch.nproc_per_node}",
        "--node_rank=$SLURM_NODEID",
        f"--master_addr={shlex.quote(launch.master_addr)}",
        f"--master_port={launch.master_port}",
        *(shlex.quote(part) for part in bench_args),
    ]
    inner = " ".join(torchrun_parts)

    if launch.use_srun:
        srun_cmd = [launch.srun]
        if launch.srun_overlap:
            job_id = os.environ.get("SLURM_JOB_ID")
            if job_id is None:
                raise RuntimeError(
                    "--srun-overlap requires SLURM_JOB_ID in the environment"
                )
            srun_cmd.extend(["--overlap", f"--jobid={job_id}"])
        srun_cmd.extend(launch.srun_extra_args)
        srun_cmd.extend(
            [
                f"--nodes={launch.nnodes}",
                "--ntasks-per-node=1",
                "bash",
                "-c",
                inner,
            ]
        )
        return srun_cmd

    return [
        launch.torchrun,
        f"--nnodes={launch.nnodes}",
        f"--nproc_per_node={launch.nproc_per_node}",
        f"--master_addr={launch.master_addr}",
        f"--master_port={launch.master_port}",
        *bench_args,
    ]


def _run_benchmark(
    *,
    backend: str,
    num_tokens: int,
    launch: LaunchConfig,
    warmup: int,
    repeat: int,
    verbose: bool,
    dry_run: bool,
) -> SweepResult:
    script = _BENCH_DIR / BACKEND_SCRIPTS[backend]
    cmd = _build_launch_command(
        launch=launch,
        script=script,
        num_tokens=num_tokens,
        warmup=warmup,
        repeat=repeat,
    )

    print(
        f"[sweep] backend={backend} num_tokens={num_tokens} "
        f"world_size={launch.world_size}\n"
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
    launch: LaunchConfig,
    warmup: int,
    repeat: int,
) -> None:
    by_key = {(r.backend, r.num_tokens): r.steady_avg_ms for r in results}

    print()
    print("DeepSeek V4 MegaMoE warm runtime sweep (steady_avg_ms)")
    print(
        f"  nnodes={launch.nnodes} nproc_per_node={launch.nproc_per_node} "
        f"world_size={launch.world_size}"
    )
    print(
        f"  master_addr={launch.master_addr} master_port={launch.master_port} "
        f"use_srun={launch.use_srun} srun_overlap={launch.srun_overlap}"
    )
    print(
        f"  warmup={warmup} repeat={repeat} num_max_tokens=num_tokens"
    )
    print()

    header = ["num_tokens"]
    for backend in backends:
        header.append(f"{backend} (ms)")
    if "vllm" in backends and "flashinfer" in backends:
        header.append("flashinfer/vLLM")

    col_widths = [max(len(h), 10) for h in header]
    for num_tokens in token_sizes:
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
    launch = _launch_config_from_args(args)
    token_sizes = list(dict.fromkeys(args.token_sizes))
    backends = list(dict.fromkeys(args.backends))

    results: list[SweepResult] = []
    for num_tokens in token_sizes:
        for backend in backends:
            results.append(
                _run_benchmark(
                    backend=backend,
                    num_tokens=num_tokens,
                    launch=launch,
                    warmup=args.warmup,
                    repeat=args.repeat,
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
        launch=launch,
        warmup=args.warmup,
        repeat=args.repeat,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
