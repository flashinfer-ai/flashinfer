#!/usr/bin/env python3
"""Reconstruct and refresh deterministic pytest duration estimates from JUnit runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_sharding.observations import EstimateRefresh, refresh_estimates
from scripts.test_sharding.models import Plan
from scripts.test_sharding.scanner import (
    scan_cleaned_artifact_dirs,
    scan_observation_inputs,
    write_scan_outputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("scan", "write reviewable observations without modifying estimates"),
        ("refresh", "scan observations and update tracked estimate files"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument(
            "inputs",
            type=Path,
            nargs="*",
            help="runner JUnit directories or XML files defining the observation window",
        )
        child.add_argument(
            "--cleaned-artifact-dir",
            type=Path,
            action="append",
            default=[],
            help=(
                "directory containing GitLab-cleaned ZIP artifacts and sibling "
                "job logs; may be repeated"
            ),
        )
        child.add_argument(
            "--reconstruction-manifest",
            type=Path,
            help=(
                "compatible runner manifest used to restore node IDs truncated "
                "in cleaned artifacts"
            ),
        )
        child.add_argument(
            "--output-dir",
            type=Path,
            default=Path.cwd(),
            help="directory for observed CSV and JSON review artifacts",
        )
        if command == "refresh":
            child.add_argument(
                "--duration-file",
                type=Path,
                default=REPO_ROOT
                / "tests"
                / "data"
                / "unit_test_duration_estimates.csv.gz",
            )
            child.add_argument(
                "--overhead-file",
                type=Path,
                default=REPO_ROOT
                / "tests"
                / "data"
                / "unit_test_overhead_estimates.csv",
            )
            child.add_argument(
                "--summary-file",
                type=Path,
                default=REPO_ROOT
                / "tests"
                / "data"
                / "unit_test_duration_estimates_summary.csv",
            )
            child.add_argument(
                "--prune",
                action="store_true",
                help="remove obsolete nodes proven absent by a complete collection manifest",
            )
            child.add_argument(
                "--complete-collection-manifest",
                type=Path,
                help="manifest proving a complete, unsampled tests/ collection for --prune",
            )
    return parser


def _prune_scope(manifest_path: Path) -> tuple[set[str], str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("selection", {}).get("sanity_test"):
        raise ValueError("a sampled manifest cannot authorize pruning")
    test_path = Path(manifest["test_path"]).resolve()
    if test_path != (REPO_ROOT / "tests").resolve():
        raise ValueError("pruning requires a complete collection rooted at tests/")
    plan = Plan.from_dict(manifest["plan"])
    return {node.nodeid for node in plan.nodes}, plan.options.profile


def _validate_inputs(args: argparse.Namespace) -> None:
    if not args.inputs and not args.cleaned_artifact_dir:
        raise ValueError("provide runner JUnit inputs or --cleaned-artifact-dir")
    if args.cleaned_artifact_dir and args.reconstruction_manifest is None:
        raise ValueError("--cleaned-artifact-dir requires --reconstruction-manifest")
    if args.reconstruction_manifest is not None and not args.cleaned_artifact_dir:
        raise ValueError("--reconstruction-manifest requires --cleaned-artifact-dir")


def _scan_inputs(args: argparse.Namespace):
    observations, overheads, diagnostics = scan_observation_inputs(args.inputs)
    if args.cleaned_artifact_dir:
        artifact_observations, artifact_overheads, artifact_diagnostics = (
            scan_cleaned_artifact_dirs(
                args.cleaned_artifact_dir,
                args.reconstruction_manifest,
            )
        )
        observations.extend(artifact_observations)
        overheads.extend(artifact_overheads)
        diagnostics.extend(artifact_diagnostics)
    return observations, overheads, diagnostics


def _refresh_scope(args: argparse.Namespace) -> tuple[set[str] | None, str | None]:
    if args.prune:
        if args.complete_collection_manifest is None:
            raise ValueError("--prune requires --complete-collection-manifest")
        return _prune_scope(args.complete_collection_manifest)
    if args.complete_collection_manifest is not None:
        raise ValueError("--complete-collection-manifest is only valid with --prune")
    return None, None


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    _validate_inputs(args)
    observations, overheads, diagnostics = _scan_inputs(args)
    write_scan_outputs(args.output_dir, observations, overheads, diagnostics)
    for diagnostic in diagnostics:
        print(f"WARNING: {diagnostic}", file=sys.stderr)
    print(
        f"Scanned {len(observations)} testcase observations and "
        f"{len(overheads)} batch overhead observations"
    )
    if args.command == "scan":
        return 0
    if not observations:
        raise ValueError(
            "refresh requires at least one eligible real testcase observation"
        )
    keep_nodeids, prune_profile = _refresh_scope(args)
    refresh_estimates(
        observations,
        overheads,
        EstimateRefresh(
            duration_file=args.duration_file,
            overhead_file=args.overhead_file,
            summary_file=args.summary_file,
            keep_nodeids=keep_nodeids,
            prune_profile=prune_profile,
        ),
    )
    print(f"Updated {args.duration_file}")
    print(f"Updated {args.overhead_file}")
    print(f"Updated {args.summary_file}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(3) from None
