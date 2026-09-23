#!/usr/bin/env python3
"""Strict Rubin qualification with per-rank JUnit logs and a job timeout.

Run from the checkout with the desired Python environment, for example:
python tests/moe_ep/qualify_sm107.py --suite all --world-size 4 --output-dir /tmp/sm107-ep4
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

SUITES = {
    "host": [
        "tests/moe_ep/test_sm107_block_scaled_config.py",
        "tests/moe_ep/test_sm107_block_scaled_contracts.py",
        "tests/moe_ep/test_sm107_block_scaled_weights.py",
        "tests/moe_ep/test_sm107_torch_staging.py",
        "tests/moe_ep/test_sm107_tuning.py",
        "tests/moe_ep/test_runtime_contracts.py",
        "tests/moe_ep/test_sm107_qualification.py",
    ],
    "single": [
        "tests/moe_ep/test_sm107_block_scaled_kernel_vs_reference.py",
        "tests/moe_ep/test_sm107_kernel_boundaries.py",
    ],
    "multi": ["tests/moe_ep/test_moe_ep_sm107_block_scaled_mega_multirank.py"],
}


def _preflight(world_size):
    import torch

    if torch.cuda.device_count() < world_size:
        raise RuntimeError(
            f"qualification needs {world_size} visible GPUs; found {torch.cuda.device_count()}"
        )
    capabilities = [torch.cuda.get_device_capability(i) for i in range(world_size)]
    if any(cc != (10, 7) for cc in capabilities):
        raise RuntimeError(
            f"qualification requires native SM107 on every rank; found {capabilities}"
        )
    target = os.environ.setdefault("CUTE_DSL_ARCH", "sm_107a")
    if target not in ("sm_107", "sm_107a"):
        raise RuntimeError("export CUTE_DSL_ARCH=sm_107a before starting qualification")
    os.environ["FLASHINFER_STRICT_MOE_EP_TESTS"] = "1"


class _NoSkips:
    def __init__(self):
        self.skipped = []
        self.passed = 0

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed += 1

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_sessionfinish(self, session, exitstatus):
        if self.skipped or not self.passed:
            session.exitstatus = 1
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(
                    f"SM107 qualification FAILED: {self.passed} passed, {len(self.skipped)} skipped; skips and empty runs are forbidden",
                    red=True,
                )


def _worker(args):
    import pytest
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1")) if args.suite == "multi" else 1
    _preflight(world_size)
    torch.cuda.set_device(local_rank)
    if args.suite == "multi":
        if world_size < 2 or os.environ.get("MEGA_NO_DIST") == "1":
            raise RuntimeError(
                "multi-rank qualification requires torchrun and MEGA_NO_DIST unset"
            )
    else:
        os.environ["MEGA_NO_DIST"] = "1"
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.installed_package:
        sys.path[:] = [
            entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != ROOT
        ]
        os.chdir(output)
    import flashinfer
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        require_sm107_dsl,
    )

    package_file = Path(flashinfer.__file__).resolve()
    if args.installed_package and ROOT in package_file.parents:
        raise RuntimeError(
            f"installed-package qualification imported the source checkout: {package_file}"
        )
    require_sm107_dsl()
    (output / f"package-rank-{local_rank}.json").write_text(
        json.dumps(dict(package_file=str(package_file)), indent=2)
    )
    pytest_args = [
        *(str(ROOT / path) for path in SUITES[args.suite]),
        "-v",
        "-s",
        "--strict-markers",
        "--import-mode=importlib",
        f"--junitxml={output / f'rank-{local_rank}.xml'}",
    ]
    if args.filter:
        pytest_args += ["-k", args.filter]
    rc = int(pytest.main(pytest_args, plugins=[_NoSkips()]))
    sys.stdout.flush()
    sys.stderr.flush()
    # Avoid interpreter-finalizer ordering between CUDA/NVSHMEM extensions.
    os._exit(rc)


def _run(command, log_path, timeout):
    print(f"Running {command!r}; log: {log_path}", flush=True)
    with log_path.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return 124


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=[*SUITES, "all"], default="all")
    parser.add_argument("--world-size", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=3600, help="seconds per suite")
    parser.add_argument(
        "--filter", help="pytest -k expression, recorded in the manifest"
    )
    parser.add_argument(
        "--sanitizer-tool", choices=["memcheck", "initcheck", "racecheck", "synccheck"]
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--installed-package",
        action="store_true",
        help="require importing a wheel outside this source checkout",
    )
    args = parser.parse_args()
    if args.worker:
        return _worker(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    suites = list(SUITES) if args.suite == "all" else [args.suite]
    manifest = dict(arguments=vars(args), commands=[], status="running")
    manifest_path = output / "qualification.json"
    try:
        _preflight(args.world_size if args.suite in ("multi", "all") else 1)
    except Exception as exc:
        manifest.update(status="preflight_failed", error=str(exc), exit_code=1)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        raise
    for suite in suites:
        worker = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            "--suite",
            suite,
            "--output-dir",
            str(output / suite),
        ]
        if args.filter:
            worker += ["--filter", args.filter]
        if args.installed_package:
            worker += ["--installed-package"]
        if args.sanitizer_tool:
            worker = [
                "compute-sanitizer",
                "--tool",
                args.sanitizer_tool,
                "--error-exitcode",
                "1",
                *worker,
            ]
        command = (
            worker
            if suite != "multi"
            else [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={args.world_size}",
                "--no-python",
                *worker,
            ]
        )
        manifest["commands"].append(command)
        manifest_path.write_text(json.dumps(manifest, indent=2))
        rc = _run(command, output / f"{suite}.log", args.timeout)
        if rc:
            manifest.update(status="failed", failed_suite=suite, exit_code=rc)
            manifest_path.write_text(json.dumps(manifest, indent=2))
            raise SystemExit(rc)
    manifest.update(status="passed", exit_code=0)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"SM107 qualification passed: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
