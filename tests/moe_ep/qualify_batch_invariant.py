#!/usr/bin/env python3
# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Focused MegaMoE BI proof, end-to-end and physical EP qualification.

Run from a provisioned SM100/SM103/SM107 environment. Unsupported numerical
classes are explicitly deselected; missing dependencies, skips, empty runs and
missing Lean are failures. Ordinary pytest suites retain their optional gates.
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

ROOT = Path(__file__).resolve().parents[2]
LEAN_DIR = ROOT / "flashinfer/batch_invariant/lean"
FAMILIES = ("bf16", "nvfp4", "mxfp4", "mxfp8", "a16")


def suite_files(suite, families):
    if suite == "host":
        return [
            *(
                (
                    "tests/batch_invariant/test_megamoe_lean.py"
                    if family == "bf16"
                    else f"tests/batch_invariant/test_{family}_megamoe_lean.py"
                )
                for family in families
            ),
            "tests/batch_invariant/test_megamoe_numerical_variants_lean.py",
            "tests/cpu/moe_ep/test_legacy_autotune_collective_failures.py",
            "tests/cpu/moe_ep/test_blackwell_bi_outer_autoconfig.py",
        ]
    files = [
        f"tests/moe_ep/test_{family}_megamoe_batch_invariant.py" for family in families
    ]
    if any(family in ("nvfp4", "mxfp4", "mxfp8") for family in families):
        files += ["tests/moe_ep/test_megamoe_quantized_tiles_batch_invariant.py"]
    if any(family in ("bf16", "nvfp4", "mxfp4", "mxfp8") for family in families):
        files += ["tests/moe_ep/test_megamoe_numerical_variants.py"]
    if any(family in ("bf16", "nvfp4", "a16") for family in families):
        files += ["tests/moe_ep/test_blackwell_bi_outer_autoconfig.py"]
    return files


def require_lean():
    lake = shutil.which("lake")
    if lake is None:
        raise RuntimeError(
            "BI qualification requires the pinned Lean toolchain; lake is missing"
        )
    pinned = (LEAN_DIR / "lean-toolchain").read_text().strip().rsplit(":v", 1)[-1]
    result = subprocess.run(
        [lake, "--version"], cwd=LEAN_DIR, capture_output=True, text=True, timeout=30
    )
    if result.returncode or f"Lean version {pinned}" not in result.stdout:
        raise RuntimeError(
            f"BI qualification requires Lean {pinned}: {result.stdout} {result.stderr}"
        )
    return lake


def preflight_gpu(world_size, expected_arch=None):
    import torch

    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f"BI qualification needs {world_size} physical visible GPUs")
    capabilities = [torch.cuda.get_device_capability(i) for i in range(world_size)]
    if len(set(capabilities)) != 1 or capabilities[0] not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        raise RuntimeError(
            f"BI qualification needs one homogeneous supported architecture: {capabilities}"
        )
    for source in range(world_size):
        for destination in range(world_size):
            if source != destination and not torch.cuda.can_device_access_peer(
                source, destination
            ):
                raise RuntimeError(
                    f"BI EP qualification needs full peer access: {source}->{destination}"
                )
    cc = capabilities[0]
    arch = f"sm_{cc[0]}{cc[1]}"
    if expected_arch and arch != expected_arch:
        raise RuntimeError(f"expected {expected_arch}, found {arch}")
    target = os.environ.setdefault("CUTE_DSL_ARCH", arch + "a")
    if target not in (arch, arch + "a"):
        raise RuntimeError(f"BI qualification needs native {arch}a, found {target}")
    return arch


def supported_case(nodeid, params, arch):
    if not any(
        name in nodeid
        for name in (
            "megamoe_batch_invariant.py",
            "megamoe_quantized_tiles_batch_invariant.py",
            "test_megamoe_numerical_variants.py",
            "test_blackwell_bi_outer_autoconfig.py",
        )
    ):
        return True
    source, _, case = nodeid.partition("::")
    # The shared A16 provider exposes this ABI on Rubin. Block-scaled Rubin
    # providers do not; admit only the two cases qualified on native SM107.
    a16_persistent = Path(
        source
    ).name == "test_a16_megamoe_batch_invariant.py" and case in (
        "test_w8a16_batch_invariant_persistent_live_count[w8a16_e4m3]",
        "test_w8a16_batch_invariant_persistent_live_count[w8a16_e5m2]",
    )
    if arch == "sm_107" and (
        params.get("provider") == "legacy"
        or ("persistent_live_count" in nodeid and not a16_persistent)
        or "blackwell_only" in nodeid
    ):
        return False
    if arch != "sm_107" and params.get("kind") == "mxfp4":
        return False
    if (
        ("test_mxfp8_" in nodeid or str(params.get("kind", "")).startswith("mxfp8_"))
        and params.get("provider") == "legacy"
        and params.get("early") is False
    ):
        return False
    return True


class QualificationGate:
    def __init__(self, arch=None, expected_sources=()):
        self.arch = arch
        self.passed = 0
        self.skipped = []
        self.deselected = []
        self.expected_sources = set(expected_sources)
        self.passed_sources = set()

    def pytest_collection_modifyitems(self, config, items):
        if self.arch is None:
            return
        retained, excluded = [], []
        for item in items:
            params = getattr(getattr(item, "callspec", None), "params", {})
            (
                retained if supported_case(item.nodeid, params, self.arch) else excluded
            ).append(item)
        if excluded:
            self.deselected.extend(item.nodeid for item in excluded)
            config.hook.pytest_deselected(items=excluded)
            items[:] = retained

    def pytest_collectreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)

    def pytest_runtest_logreport(self, report):
        if report.skipped:
            self.skipped.append(report.nodeid)
        elif report.when == "call" and report.passed:
            self.passed += 1
            self.passed_sources.add(report.nodeid.split("::", 1)[0])

    def pytest_sessionfinish(self, session, exitstatus):
        missing = self.expected_sources - self.passed_sources
        if self.skipped or not self.passed or missing:
            session.exitstatus = 1
            reporter = session.config.pluginmanager.get_plugin("terminalreporter")
            if reporter:
                reporter.write_line(
                    f"BI qualification FAILED: {self.passed} passed, {len(self.skipped)} skipped, missing sources={sorted(missing)}; skips/empty runs forbidden"
                )


def worker(args):
    import pytest

    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    arch = None
    if args.suite == "host":
        require_lean()
        # Keep generated build products outside the checked-out package.
        lean_copy = output / "lean"
        shutil.copytree(
            LEAN_DIR,
            lean_copy,
            ignore=shutil.ignore_patterns(".lake"),
            dirs_exist_ok=True,
        )
        from flashinfer.moe_ep.batch_invariant import lean_cases

        lean_cases.LEAN_DIR = str(lean_copy)
    else:
        import torch

        world = int(os.environ.get("WORLD_SIZE", "1")) if args.suite == "multi" else 1
        arch = preflight_gpu(world, args.arch)
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        if args.suite == "multi":
            if world < 2 or os.environ.get("MEGA_NO_DIST") == "1":
                raise RuntimeError(
                    "BI EP qualification requires torchrun and MEGA_NO_DIST unset"
                )
        else:
            os.environ["MEGA_NO_DIST"] = "1"
    rank = int(os.environ.get("RANK", "0"))
    files = suite_files(args.suite, args.families)
    gate = QualificationGate(arch, files)
    arguments = [
        *files,
        "-v",
        "-s",
        "-x",
        "--strict-markers",
        f"--junitxml={output / f'rank-{rank}.xml'}",
    ]
    if args.suite != "host":
        arguments += ["-m", "gpu_2" if args.suite == "multi" else "not gpu_2"]
    rc = int(pytest.main(arguments, plugins=[gate]))
    (output / f"rank-{rank}.json").write_text(
        json.dumps(
            dict(
                arch=arch,
                suite=args.suite,
                passed=gate.passed,
                skipped=gate.skipped,
                architecture_deselected=gate.deselected,
                exit_code=rc,
            ),
            indent=2,
        )
    )
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


def run(command, log, timeout, env):
    with log.open("w") as stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
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


def worker_command(stage, args, output):
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--suite",
        stage,
        "--families",
        *args.families,
        "--output-dir",
        str(output / stage),
    ]
    if args.arch:
        command += ["--arch", args.arch]
    if stage == "multi":
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc-per-node={args.world_size}",
            "--no-python",
            *command,
        ]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite", choices=("host", "single", "multi", "all"), default="all"
    )
    parser.add_argument(
        "--families", nargs="+", choices=FAMILIES, default=list(FAMILIES)
    )
    parser.add_argument("--arch", choices=("sm_100", "sm_103", "sm_107"))
    parser.add_argument("--world-size", choices=(2, 4, 8), type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=7200, help="seconds per suite")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument(
        "--plan",
        action="store_true",
        help="print commands without installing dependencies or running tests",
    )
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT))
    if args.worker:
        return worker(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    stages = ("host", "single", "multi") if args.suite == "all" else (args.suite,)
    manifest = dict(arguments=vars(args), status="running", stages=[])
    path = output / "qualification.json"
    if args.plan:
        manifest.update(
            status="planned",
            stages=[
                dict(suite=stage, command=worker_command(stage, args, output))
                for stage in stages
            ],
        )
        path.write_text(json.dumps(manifest, indent=2))
        print(json.dumps(manifest, indent=2))
        return
    try:
        if "host" in stages:
            require_lean()
        if args.suite != "host":
            manifest["arch"] = preflight_gpu(
                args.world_size if "multi" in stages else 1, args.arch
            )
        for stage in stages:
            command = worker_command(stage, args, output)
            env = os.environ.copy()
            if stage == "host":
                env["CUDA_VISIBLE_DEVICES"] = ""
            elif stage == "multi":
                env.pop("MEGA_NO_DIST", None)
                env["FAIL_ON_DISTRIBUTED_SKIP"] = "1"
            record = dict(suite=stage, command=command, status="running")
            manifest["stages"].append(record)
            path.write_text(json.dumps(manifest, indent=2))
            print(
                f"BI qualification: {stage}; log={output / (stage + '.log')}",
                flush=True,
            )
            rc = run(command, output / (stage + ".log"), args.timeout, env)
            record.update(exit_code=rc, status="passed" if rc == 0 else "failed")
            if rc:
                raise RuntimeError(f"BI {stage} qualification failed with exit {rc}")
        manifest["status"] = "passed"
    except Exception as error:
        manifest.update(status="failed", error=str(error))
        raise
    finally:
        path.write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
