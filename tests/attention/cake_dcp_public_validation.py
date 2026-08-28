"""Promotion hook for the generated Cake FMHA DCP source bundle.

This script is intentionally public-repository owned.  A downstream promotion
job may provide a freshly generated source directory, but it cannot decide
which public API test or which checked-in destination defines success.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import torch


EXPECTED_ARTIFACT_COUNT = 37
_TWO_GPU_API_TEST = (
    "tests/attention/test_dcp_spec_fp8.py::"
    "test_fp8_page64_d256_initializes_dynamic_smem_on_each_device"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_export(
    generated_root: Path,
    public_root: Path,
    *,
    expected_artifact_count: int = EXPECTED_ARTIFACT_COUNT,
) -> dict[str, object]:
    """Compare the generated flat source bundle with ``csrc/dcp``."""

    generated = {path.name: path for path in generated_root.iterdir() if path.is_file()}
    public = {path.name: path for path in public_root.iterdir() if path.is_file()}
    generated_names = set(generated)
    public_names = set(public)
    common_names = sorted(generated_names & public_names)
    mismatched = [
        {
            "name": name,
            "generated_sha256": _sha256(generated[name]),
            "public_sha256": _sha256(public[name]),
        }
        for name in common_names
        if generated[name].read_bytes() != public[name].read_bytes()
    ]
    matched_count = len(common_names) - len(mismatched)
    passed = (
        len(generated) == expected_artifact_count
        and len(public) == expected_artifact_count
        and generated_names == public_names
        and matched_count == expected_artifact_count
    )
    return {
        "export_parity_passed": passed,
        "expected_artifact_count": expected_artifact_count,
        "generated_artifact_count": len(generated),
        "public_artifact_count": len(public),
        "matched_artifact_count": matched_count,
        "missing_from_public": sorted(generated_names - public_names),
        "unexpected_in_public": sorted(public_names - generated_names),
        "mismatched_artifacts": mismatched,
    }


def _run_two_gpu_public_api(repo_root: Path) -> tuple[bool, int]:
    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count < 2:
        return False, visible_gpu_count
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", _TWO_GPU_API_TEST],
        cwd=repo_root,
        check=False,
    )
    return completed.returncode == 0, visible_gpu_count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the public Cake FMHA DCP export and two-GPU API"
    )
    parser.add_argument("--cake-export-dir", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    generated_root = args.cake_export_dir.resolve()
    public_root = repo_root / "csrc" / "dcp"
    if not generated_root.is_dir():
        parser.error(f"Cake export directory does not exist: {generated_root}")
    if not public_root.is_dir():
        parser.error(f"public DCP source directory does not exist: {public_root}")

    parity = compare_export(generated_root, public_root)
    api_passed, visible_gpu_count = _run_two_gpu_public_api(repo_root)
    payload = {
        "schema": "cake-fmha-dcp-public-validation-v1",
        "two_gpu_api_passed": api_passed,
        "visible_gpu_count": visible_gpu_count,
        **parity,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if api_passed and bool(parity["export_parity_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
