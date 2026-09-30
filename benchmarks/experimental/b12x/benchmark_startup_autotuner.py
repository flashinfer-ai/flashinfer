"""Cold and cached TP2 startup of model-derived native b12x kernel workloads.

Timing covers process/import/device setup, native preparation, complete candidate
compilation and timing, selection persistence, and priming. Independent complete
quality races and cosine checks run after both ranks finish startup.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback

_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_ROOT))


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, choices=("qwen3.8-flash-next-180b", "glm-5.3-flash")
    )
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument(
        "--devices", required=True, help="Two explicitly assigned GPU ordinals or UUIDs for TP2"
    )
    parser.add_argument(
        "--rows",
        default="1,4,128",
        help="Concrete planned/capture row counts, not a search range",
    )
    parser.add_argument("--max-seqs", type=int, default=1)
    parser.add_argument("--cache-tokens", type=int, default=4096)
    parser.add_argument("--spec-tokens", type=int, default=3)
    parser.add_argument(
        "--compile-workers",
        type=int,
        choices=(8, 16),
        default=8,
        help="Total serial compiler processes across both TP ranks",
    )
    parser.add_argument(
        "--startup-target", type=float, default=120,
        help="Advisory startup target in seconds; never prunes candidates or fails acceptance",
    )
    parser.add_argument(
        "--process-timeout", type=float, default=0,
        help="Operational phase timeout in seconds; 0 waits for complete runs without a deadline",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--groups", help="Diagnostic component subset; never passes full acceptance"
    )
    parser.add_argument("--cosine", type=float, default=0.998)
    parser.add_argument(
        "--startup-only", action="store_true",
        help="Measure startup/restart without the separate post-startup quality rerace",
    )
    parser.add_argument("--worker-rank", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--cached", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--phase-dir", type=Path, help=argparse.SUPPRESS)
    return parser




def _prepare_world(session, requests):
    """Drive every rank, including finished ranks, through complete control rounds."""
    import torch.distributed as dist

    size, rank = dist.get_world_size(), dist.get_rank()
    job = session.begin(requests)
    authorization = None
    try:
        while True:
            error = None
            try:
                progress = job.advance(collective_key=authorization)
                local = {
                    "rank": rank, "done": progress.done, "error": None,
                    "ready": tuple((item.key, item.ranks) for item in progress.ready_collectives),
                }
            except BaseException as failure:
                error = failure
                local = {
                    "rank": rank, "done": False, "ready": (),
                    "error": f"{type(failure).__name__}: {failure}",
                }
            gathered = [None] * size
            dist.all_gather_object(gathered, local)
            if {entry["rank"] for entry in gathered} != set(range(size)):
                raise RuntimeError("benchmark collective control domain is inconsistent")
            failures = [entry for entry in gathered if entry["error"] is not None]
            if failures:
                if error is not None:
                    raise error
                raise RuntimeError(f"peer preparation failed: {failures}")
            if all(entry["done"] for entry in gathered):
                return job.result()
            participants, reporters = {}, {}
            for entry in gathered:
                for key, ranks in entry["ready"]:
                    ranks = tuple(ranks)
                    if participants.setdefault(key, ranks) != ranks:
                        raise RuntimeError("benchmark collective participants disagree")
                    if not set(ranks) <= set(range(size)) or entry["rank"] not in ranks:
                        raise RuntimeError("benchmark collective names an invalid participant")
                    reporters.setdefault(key, set()).add(entry["rank"])
            choices = [
                key for key, ranks in participants.items()
                if set(ranks) <= reporters[key]
            ]
            selected = min(choices) if choices else None
            authorization = (
                selected if selected is not None and rank in participants[selected] else None
            )
            if progress.pending_compilation and session._pool is not None:
                session._pool.wait_for_progress(timeout=0.05)
    except BaseException:
        job.close()
        raise


def _worker(args):
    from dataclasses import asdict
    from datetime import timedelta
    import torch
    import torch.distributed as dist
    from b12x.preparation import PreparationSession, detect_device
    from b12x.preparation._cache import digest
    from b12x.preparation._measurement import no_compilation
    from b12x.preparation.types import _close_all
    from b12x.testing.startup import make_benchmark_requests, model_metadata
    from benchmarks.experimental.b12x.startup_quality import check_quality, flatten_requests

    rank = args.worker_rank
    torch.set_num_threads(1)
    torch.cuda.set_device(rank)
    torch.manual_seed(42)
    device = torch.device("cuda", rank)
    metadata = model_metadata(
        args.model, tp=args.tp, checkpoint=args.checkpoint,
        max_seqs=args.max_seqs, cache_tokens=args.cache_tokens,
        spec_tokens=args.spec_tokens,
    )
    metadata["_tp_rank"] = rank
    rows = tuple(sorted({int(value) for value in args.rows.split(",")}))
    if not rows or min(rows) <= 0:
        raise ValueError("planned rows must be positive concrete counts")
    if max(rows) < args.max_seqs * (args.spec_tokens + 1):
        raise ValueError("maximum planned rows must cover the declared verifier capacity")
    metadata["_rows"] = rows
    groups = None if args.groups is None else tuple(args.groups.split(","))
    use_comm = groups is None or "comm" in groups
    if use_comm:
        dist.init_process_group(
            "gloo", init_method=f"file://{args.phase_dir / 'rendezvous'}",
            rank=rank, world_size=args.tp,
            timeout=timedelta(seconds=args.process_timeout) if args.process_timeout else timedelta(days=365),
        )
    result = session = None
    collective_calls = []
    startup_path = args.phase_dir / f"rank-{rank}.startup.json"
    final_path = args.phase_dir / f"rank-{rank}.json"
    try:
        with torch.inference_mode():
            requests, owners = make_benchmark_requests(
                metadata, device=device, rows=rows,
                groups=None if groups is None else tuple(group for group in groups if group != "comm"),
            )
            if use_comm:
                from b12x.testing.startup import comm
                collective = comm.make_benchmark_requests(metadata, device=device, rows=rows)
                requests.extend(collective)
                owners.update({request.name: comm for request in collective})
            detected = detect_device(device)
            session = PreparationSession(
                device=detected, cache_dir=args.run_dir / "choices",
                namespace={
                    "model": args.model, "tp": args.tp, "config": digest(metadata),
                    "rows": rows, "groups": groups,
                },
                compile_workers=args.compile_workers // args.tp,
                cache_only=args.cached,
            )
            result = _prepare_world(session, requests) if use_comm else session.prepare(requests)
            collective_calls = [
                call for name, call in result.benchmark_calls.items() if name.startswith("comm.")
            ]
            torch.cuda.synchronize(device)
            completed = time.monotonic()
            leaves, _ = flatten_requests(requests, owners, detected)
            summary = {
                "rank": rank, "model": args.model, "cached": args.cached,
                "startup_completed": completed,
                "program_counts": dict(result.program_counts),
                "device_ordinal": rank, "requests": len(leaves), "names": list(leaves),
                "cache_hits": result.cache_hits,
                "benchmarked_candidates": result.benchmarked_candidates,
                "parent_cute_compilations": result.parent_cute_compilations,
                "parent_triton_compilations": result.parent_triton_compilations,
                "compilation": None if result.compilation is None else asdict(result.compilation),
                "overlapped_benchmark_samples": result.overlapped_benchmark_samples,
                "tuner_seconds": result.elapsed_seconds,
                "choices": {
                    name: leaves[name].plan.contract.config_payload(selection.config).to_dict()
                    for name, selection in result.selections.items()
                },
                "selection_sources": {name: selection.source for name, selection in result.selections.items()},
                "coverage": {name: dict(value) for name, value in result.coverage.items()},
            }
            startup_path.write_text(json.dumps(summary, indent=2) + "\n")
            # No rank's independent recheck competes with its peer's preparation.
            other_startup = args.phase_dir / f"rank-{1 - rank}.startup.json"
            while not other_startup.exists():
                time.sleep(0.02)
            summary["quality"] = None
            if not args.startup_only:
                summary["quality"] = check_quality(
                    requests, owners, result, device=detected, device_ordinal=rank,
                    cosine=args.cosine, cached=args.cached,
                    prohibit_compilation=no_compilation,
                    prepare_collective_batch=_prepare_world if use_comm else None,
                )
            final_path.write_text(json.dumps(summary, indent=2) + "\n")
            print(json.dumps({
                "rank": rank, "cache_hits": result.cache_hits, "requests": len(leaves),
                "quality": None if summary["quality"] is None else summary["quality"]["aggregate"],
            }), flush=True)
    except BaseException as error:
        failure = {
            "rank": rank, "error": f"{type(error).__name__}: {error}",
            "traceback": traceback.format_exc(),
        }
        (args.phase_dir / f"rank-{rank}.failure.json").write_text(json.dumps(failure, indent=2) + "\n")
        raise
    finally:
        closers = []
        if result is not None:
            closers.append(result.close)
        if session is not None:
            closers.append(session.close)
        if collective_calls:
            from b12x.testing.startup import comm
            closers.append(lambda: comm.close_calls(collective_calls))
        if dist.is_initialized():
            closers.append(dist.destroy_process_group)
        _close_all(closers)


def _phase(args, cached):
    phase = args.run_dir / ("cached" if cached else "cold")
    phase.mkdir(parents=True, exist_ok=False)
    (args.run_dir / "xdg" / "torch" / "kernels").mkdir(parents=True, exist_ok=True)
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": args.devices,
        "PYTHONPATH": os.pathsep.join(filter(None, (str(_ROOT), os.environ.get("PYTHONPATH")))),
        "PYTHONDONTWRITEBYTECODE": "1",
        "B12X_COMPILE_CACHE_DIR": str(args.run_dir / "cute"),
        "TRITON_CACHE_DIR": str(args.run_dir / "triton"),
        "CUTE_DSL_CACHE_DIR": str(args.run_dir / "dsl"),
        "TORCHINDUCTOR_CACHE_DIR": str(args.run_dir / "inductor"),
        "XDG_CACHE_HOME": str(args.run_dir / "xdg"),
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if not environment.get("CUDA_HOME"):
        nvcc = shutil.which("nvcc")
        if nvcc is not None:
            environment["CUDA_HOME"] = str(Path(nvcc).resolve().parent.parent)
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--model",
        args.model,
        "--tp",
        str(args.tp),
        "--devices",
        args.devices,
        "--rows",
        args.rows,
        "--max-seqs",
        str(args.max_seqs),
        "--cache-tokens",
        str(args.cache_tokens),
        "--spec-tokens",
        str(args.spec_tokens),
        "--compile-workers",
        str(args.compile_workers),
        "--startup-target",
        str(args.startup_target),
        "--process-timeout",
        str(args.process_timeout),
        "--run-dir",
        str(args.run_dir),
        "--phase-dir",
        str(phase),
        "--cosine",
        str(args.cosine),
    ]
    if cached:
        command.append("--cached")
    if args.groups:
        command.extend(("--groups", args.groups))
    if args.checkpoint:
        command.extend(("--checkpoint", str(args.checkpoint)))
    if args.startup_only:
        command.append("--startup-only")
    children, logs = [], []
    started = time.monotonic()
    try:
        for rank in range(args.tp):
            log = (phase / f"rank-{rank}.log").open("w")
            logs.append(log)
            children.append(
                subprocess.Popen(
                    [*command, "--worker-rank", str(rank)],
                    cwd=_ROOT,
                    env=environment,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            )
        deadline = started + args.process_timeout if args.process_timeout else None
        while any(child.poll() is None for child in children):
            if any(child.returncode not in (None, 0) for child in children):
                raise RuntimeError(
                    f"native startup rank failed; see {phase}/rank-*.log"
                )
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"native startup acceptance timed out; see {phase}")
            time.sleep(0.05)
        if any(child.returncode for child in children):
            raise RuntimeError(f"native startup rank failed; see {phase}/rank-*.log")
    finally:
        for child in children:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
    ranks = [
        json.loads((phase / f"rank-{rank}.json").read_text()) for rank in range(args.tp)
    ]

    startup_seconds = max(row["startup_completed"] for row in ranks) - started
    failures = []
    quality = None
    if not args.startup_only:
        from benchmarks.experimental.b12x.startup_quality import aggregate_quality
        quality = aggregate_quality([row["quality"] for row in ranks], cached=cached)
        if not quality["passed"]:
            failures.extend(quality["failures"])
        for row in ranks:
            if not row["quality"]["aggregate"]["passed"]:
                failures.append({"rank": row["rank"], "quality": row["quality"]["aggregate"]["failures"]})
    if cached and any(
        row["benchmarked_candidates"]
        or row["parent_cute_compilations"]
        or row["parent_triton_compilations"]
        or row["compilation"] is not None
        or (row["quality"] is not None and (
            row["quality"]["aggregate"]["measured_count"]
            or row["quality"]["performance_recheck"]
        ))
        for row in ranks
    ):
        failures.append("cached startup performed compilation or benchmarking")
    return {
        "startup_seconds": startup_seconds,
        "startup_target_seconds": args.startup_target,
        "startup_target_met": startup_seconds <= args.startup_target,
        "startup_target_advisory": True,
        "quality": quality,
        "passed": not failures,
        "failures": failures,
        "ranks": ranks,
    }


def main():
    args = _parser().parse_args()
    devices = tuple(value.strip() for value in args.devices.split(","))
    if args.tp != 2 or len(devices) != 2 or len(set(devices)) != 2 or any(not value for value in devices):
        raise SystemExit("TP2 startup requires two distinct assigned GPU ordinals or UUIDs")

    if not 0 < args.startup_target < float("inf") or not 0 <= args.process_timeout < float("inf"):
        raise SystemExit("startup target must be positive finite; process timeout must be nonnegative finite")
    if not 0 < args.cosine <= 1:
        raise SystemExit("cosine threshold must be in (0, 1]")
    if args.worker_rank is not None:
        _worker(args)
        return
    if args.run_dir is None:
        args.run_dir = Path(tempfile.mkdtemp(prefix="b12x-startup-"))
    else:
        args.run_dir = args.run_dir.resolve()
        args.run_dir.mkdir(parents=True, exist_ok=False)
    report = {
        "command": sys.argv,
        "worktree": str(_ROOT),
        "rows": args.rows,
        "tp": args.tp,
        "devices": devices,
        "compile_workers": args.compile_workers,
        "full_model_workload": args.groups is None,
        "startup_target_seconds": args.startup_target,
        "startup_target_advisory": True,
        "acceptance_scope": "startup_timing_only" if args.startup_only else (
            "diagnostic_subset" if args.groups is not None else "full_model"
        ),
        "full_acceptance": False,
        "results": {},
    }
    output = args.run_dir / "results.json"
    try:
        for cached in (False, True):
            name = "cached" if cached else "cold"
            report["results"][name] = _phase(args, cached)
            output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "phase": name,
                        "startup_seconds": report["results"][name]["startup_seconds"],
                        "startup_target_met": report["results"][name]["startup_target_met"],
                        "quality": report["results"][name]["quality"],
                        "passed": report["results"][name]["passed"],
                        "result": str(output),
                    }
                ),
                flush=True,
            )
        report["passed"] = all(phase["passed"] for phase in report["results"].values())
        report["full_acceptance"] = report["passed"] and args.groups is None and not args.startup_only
        output.write_text(json.dumps(report, indent=2) + "\n")
        if not report["passed"]:
            raise AssertionError("startup acceptance failed; full coverage/quality failures are recorded in results.json")
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        report["passed"] = False
        report["full_acceptance"] = False
        report["worker_failures"] = {
            str(path.relative_to(args.run_dir)): path.read_text(errors="replace")
            for path in sorted(args.run_dir.glob("*/rank-*.failure.json"))
        }
        report["worker_reports"] = {
            str(path.relative_to(args.run_dir)): path.read_text(errors="replace")
            for path in sorted(args.run_dir.glob("*/rank-*.json"))
            if not path.name.endswith((".failure.json", ".startup.json"))
        }
        report["worker_logs"] = {
            str(path.relative_to(args.run_dir)): path.read_text(errors="replace")
            for path in sorted(args.run_dir.glob("*/rank-*.log"))
        }
        output.write_text(json.dumps(report, indent=2) + "\n")
        raise
    if args.startup_only:
        print("Startup timing only; no post-startup quality qualification.", flush=True)
    if args.groups:
        print("Diagnostic subset only; not full model startup acceptance.", flush=True)


if __name__ == "__main__":
    from b12x.testing.memory import absorb_small_page_fragments

    absorb_small_page_fragments()
    main()
