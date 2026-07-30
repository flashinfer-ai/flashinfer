#!/usr/bin/env python3
"""Deterministic, duration-balanced, resumable pytest coordinator."""

from __future__ import annotations

import argparse
import math
import os
import shlex
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.test_sharding.models import (
    DEFAULT_CHECKPOINT_SECONDS,
    DEFAULT_TARGET_UNIT_SECONDS,
    Plan,
    PlanningOptions,
)
from scripts.test_sharding.runner import (
    CollectionTimeoutError,
    DeadlineClock,
    ExecutionSettings,
    ManifestPreparation,
    SelectionSettings,
    execute_shard,
    finalize_latest,
    plan_description,
    prepare_manifest,
    visible_devices,
)
from scripts.test_sharding.planner import capacity_metrics
from scripts.test_sharding.state import (
    AttemptSettings,
    RunnerStateError,
    load_manifest,
)
from scripts.test_sharding.summary import exit_code_for_summary, publish_summary


EXIT_HELP = (
    "exit codes: 0=complete without failures; "
    "1=complete with real or synthetic failures; "
    "2=incomplete and resumable; "
    "3=configuration, manifest, collection, or infrastructure error"
)
DEFAULT_DEADLINE_SECONDS = 0
DEFAULT_UNIT_TIMEOUT_SECONDS = 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return int(value) if value is not None else default


def _env_positive_float(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return value


def _pytest_command_prefix() -> tuple[str, ...]:
    try:
        return tuple(shlex.split(os.environ.get("PYTEST_COMMAND_PREFIX", "")))
    except ValueError as error:
        raise RunnerStateError(f"invalid PYTEST_COMMAND_PREFIX: {error}") from error


def _configure_output() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(line_buffering=True)


def _positive(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _nonnegative(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return parsed


def _timeout_policy(value: str) -> str:
    if value not in {"resume", "skip", "fail"}:
        raise argparse.ArgumentTypeError("must be one of: resume, skip, fail")
    return value


def _auto_profile() -> str:
    cuda = "unknown"
    gpu = "unknown"
    try:
        import torch

        if torch.version.cuda:
            cuda = torch.version.cuda.split(".", 1)[0]
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability(0)
            name = torch.cuda.get_device_name(0).lower()
            gpu = next(
                (
                    model
                    for model in ("b100", "b200", "h100", "h200", "a100", "l40")
                    if model in name
                ),
                f"sm{major}{minor}",
            )
    except Exception:
        pass
    return f"{gpu}-cuda{cuda}"


def _add_common(parser: argparse.ArgumentParser) -> None:
    timing_profile = os.environ.get("UNIT_TEST_TIMING_PROFILE") or None
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--junit-dir",
        type=Path,
        default=Path(os.environ.get("JUNIT_DIR", "./junit")),
        help="shared result and persistent state directory (JUNIT_DIR)",
    )
    parser.add_argument(
        "--target-unit-seconds",
        type=_positive,
        default=os.environ.get("UNIT_TEST_TARGET_SECONDS", DEFAULT_TARGET_UNIT_SECONDS),
        help="soft logical-unit target (UNIT_TEST_TARGET_SECONDS)",
    )
    parser.add_argument(
        "--checkpoint-seconds",
        type=_positive,
        default=os.environ.get(
            "UNIT_TEST_CHECKPOINT_SECONDS", DEFAULT_CHECKPOINT_SECONDS
        ),
        help="source-local checkpoint target (UNIT_TEST_CHECKPOINT_SECONDS)",
    )
    parser.add_argument(
        "--unknown-case-seconds",
        type=_positive,
        default=os.environ.get("UNIT_TEST_UNKNOWN_CASE_SECONDS", 1),
        help="unknown-node estimate floor (UNIT_TEST_UNKNOWN_CASE_SECONDS)",
    )
    parser.add_argument(
        "--timing-profile",
        default=timing_profile if timing_profile is not None else argparse.SUPPRESS,
        help=(
            "stable timing profile (UNIT_TEST_TIMING_PROFILE)"
            if timing_profile is not None
            else (
                "stable timing profile "
                "(UNIT_TEST_TIMING_PROFILE; default: auto-detected)"
            )
        ),
    )
    parser.add_argument(
        "--shard-count",
        type=_positive,
        default=os.environ.get("UNIT_TEST_SHARD_COUNT", 1),
        help="number of deterministic external shards (UNIT_TEST_SHARD_COUNT)",
    )
    parser.add_argument(
        "--shard-index",
        type=_nonnegative,
        default=os.environ.get("UNIT_TEST_SHARD_INDEX", 0),
        help="zero-based external shard (UNIT_TEST_SHARD_INDEX)",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path(os.environ.get("TEST_PATH") or "tests/"),
        help="pytest collection scope (TEST_PATH)",
    )
    parser.add_argument(
        "--sanity-test",
        action="store_true",
        help=(
            "globally select every SAMPLE_RATE-th collected node using a zero-based "
            "SAMPLE_OFFSET (defaults: SAMPLE_RATE=5, SAMPLE_OFFSET=0)"
        ),
    )
    parser.add_argument(
        "--workers",
        type=_positive,
        default=os.environ.get("UNIT_TEST_WORKERS", max(1, len(visible_devices()))),
        help="local workers, at most visible GPU count (UNIT_TEST_WORKERS)",
    )
    parser.add_argument(
        "--unit-timeout-seconds",
        type=_nonnegative,
        default=os.environ.get(
            "UNIT_TEST_TIMEOUT_SECONDS", DEFAULT_UNIT_TIMEOUT_SECONDS
        ),
        help="cumulative unit timeout; 0 disables (UNIT_TEST_TIMEOUT_SECONDS)",
    )
    parser.add_argument(
        "--timeout-grace-seconds",
        type=_nonnegative,
        default=os.environ.get("UNIT_TEST_TIMEOUT_GRACE_SECONDS", 300),
        help="SIGTERM-to-SIGKILL grace (UNIT_TEST_TIMEOUT_GRACE_SECONDS)",
    )
    parser.add_argument(
        "--timeout-policy",
        type=_timeout_policy,
        choices=("resume", "skip", "fail"),
        default=os.environ.get("UNIT_TEST_TIMEOUT_POLICY", "resume"),
        help="handling for unexecuted nodes (UNIT_TEST_TIMEOUT_POLICY)",
    )
    parser.add_argument(
        "--deadline-seconds",
        type=_nonnegative,
        default=os.environ.get("UNIT_TEST_DEADLINE_SECONDS", DEFAULT_DEADLINE_SECONDS),
        help="shared attempt deadline; 0 disables (UNIT_TEST_DEADLINE_SECONDS)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=EXIT_HELP,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("run", "create/resume a run and execute one external shard"),
        ("plan", "create/verify a plan without executing tests"),
    ):
        child = subparsers.add_parser(
            command,
            help=help_text,
            epilog=EXIT_HELP,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        _add_common(child)
    for command, help_text in (
        ("finalize", "idempotently close the latest attempt and regenerate artifacts"),
        ("summarize", "regenerate derived artifacts from authoritative batch XML"),
    ):
        child = subparsers.add_parser(
            command,
            help=help_text,
            epilog=EXIT_HELP,
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        child.add_argument(
            "--junit-dir",
            type=Path,
            default=Path(os.environ.get("JUNIT_DIR", "./junit")),
            help="shared result and persistent state directory (JUNIT_DIR)",
        )
    return parser


def _shell_settings(argv: list[str]) -> int:
    operation = "plan" if "--dry-run" in argv else "run"
    parser = _parser()
    args = parser.parse_args([operation, *argv])
    try:
        sample_rate = _env_int("SAMPLE_RATE", 5)
        sample_offset = _env_int("SAMPLE_OFFSET", 0)
        if sample_rate <= 0:
            raise ValueError("SAMPLE_RATE must be positive")
        if not 0 <= sample_offset < sample_rate:
            raise ValueError("SAMPLE_OFFSET must be in [0, SAMPLE_RATE)")
        _pytest_command_prefix()
        _env_positive_float("MEMORY_MONITOR_INTERVAL", 2)
    except (RunnerStateError, ValueError) as error:
        parser.error(str(error))
    print(operation)
    print(args.test_path)
    return 0


def _load_frozen_plan(junit_dir: Path) -> Plan:
    manifest = load_manifest(junit_dir)
    if manifest is None:
        raise RunnerStateError(f"no runner manifest in {junit_dir}")
    return Plan.from_dict(manifest["plan"])


def _execute_command(args: argparse.Namespace, operation_started_at: float) -> int:
    junit_dir = args.junit_dir.resolve()
    if args.command in {"finalize", "summarize"}:
        plan = _load_frozen_plan(junit_dir)
        if args.command == "finalize":
            return finalize_latest(junit_dir, plan)
        return exit_code_for_summary(publish_summary(junit_dir, plan))

    if "PYTEST_FILE_TIMEOUT_SECONDS" in os.environ:
        raise RunnerStateError(
            "PYTEST_FILE_TIMEOUT_SECONDS is obsolete; use UNIT_TEST_TIMEOUT_SECONDS "
            "or --unit-timeout-seconds"
        )
    if "PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS" in os.environ:
        raise RunnerStateError(
            "PYTEST_FILE_TIMEOUT_KILL_AFTER_SECONDS is obsolete; use "
            "UNIT_TEST_TIMEOUT_GRACE_SECONDS or --timeout-grace-seconds"
        )

    pytest_command_prefix = _pytest_command_prefix()
    profile = getattr(args, "timing_profile", None) or _auto_profile()
    planning = PlanningOptions(
        profile=profile,
        checkpoint_seconds=args.checkpoint_seconds,
        target_unit_seconds=args.target_unit_seconds,
        unknown_case_seconds=args.unknown_case_seconds,
        shard_count=args.shard_count,
    )
    sample_rate = _env_int("SAMPLE_RATE", 5)
    sample_offset = _env_int("SAMPLE_OFFSET", 0)
    selection = SelectionSettings(
        test_path=args.test_path.resolve(),
        sanity_test=args.sanity_test,
        sample_rate=sample_rate,
        sample_offset=sample_offset,
    )
    collection_timeout = (
        max(0.001, args.deadline_seconds - (time.time() - operation_started_at))
        if args.deadline_seconds > 0
        else None
    )
    attempt = AttemptSettings(
        deadline_seconds=args.deadline_seconds,
        unit_timeout_seconds=args.unit_timeout_seconds,
        timeout_grace_seconds=args.timeout_grace_seconds,
        timeout_policy=args.timeout_policy,
    )
    print(
        f"RUNNER STATUS: state=collecting command={args.command} "
        f"test_path={selection.test_path}",
        flush=True,
    )
    _, plan, created = prepare_manifest(
        ManifestPreparation(
            repo_root=REPO_ROOT,
            junit_dir=junit_dir,
            selection=selection,
            planning=planning,
            collection_timeout_seconds=collection_timeout,
            collection_grace_seconds=args.timeout_grace_seconds,
            attempt_settings=attempt if args.command == "run" else None,
            operation_started_at=operation_started_at,
            pytest_command_prefix=pytest_command_prefix,
        )
    )
    print(("Created" if created else "Using") + f" plan in {junit_dir}", flush=True)
    print(
        plan_description(
            plan,
            workers=args.workers,
            deadline_seconds=args.deadline_seconds,
        ),
        flush=True,
    )
    print(
        f"RUNNER STATUS: state=plan-ready nodes={len(plan.nodes)} "
        f"units={len(plan.units)} shards={plan.options.shard_count}",
        flush=True,
    )
    capacity = capacity_metrics(
        plan,
        {index: args.workers for index in range(plan.options.shard_count)},
        deadline_seconds=args.deadline_seconds,
    )
    if (
        args.deadline_seconds > 0
        and capacity["estimated_makespan_ms"] > args.deadline_seconds * 1000
    ):
        print(
            "WARNING: estimated per-shard load exceeds the attempt deadline at "
            f"{args.workers} local worker(s); configure more workers/shards or expect resume",
            file=sys.stderr,
        )
    if args.unit_timeout_seconds > 0:
        over_timeout = [
            batch
            for unit in plan.units
            for batch in unit.batches
            if batch.estimated_ms > args.unit_timeout_seconds * 1000
        ]
        if over_timeout:
            print(
                f"WARNING: {len(over_timeout)} atomic/checkpoint batch(es) are estimated "
                "to exceed the unit timeout",
                file=sys.stderr,
            )
    if args.command == "plan":
        return 0
    execution = ExecutionSettings(
        workers=args.workers,
        shard_index=args.shard_index,
        attempt=attempt,
        monitor_memory=os.environ.get("MONITOR_TEST_MEMORY", "true").lower()
        not in {"0", "false", "no"},
        memory_interval=_env_positive_float("MEMORY_MONITOR_INTERVAL", 2),
        pytest_command_prefix=pytest_command_prefix,
    )
    return execute_shard(
        repo_root=REPO_ROOT,
        junit_dir=junit_dir,
        plan=plan,
        execution=execution,
        operation_started_at=operation_started_at,
    )


def main(argv: list[str] | None = None) -> int:
    _configure_output()
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["__shell-settings"]:
        return _shell_settings(arguments[1:])
    operation_started_at = time.time()
    deadline_clock = DeadlineClock(
        started_at=operation_started_at,
        limit_seconds=0,
    )
    try:
        args = _parser().parse_args(arguments)
        deadline_clock = DeadlineClock(
            started_at=operation_started_at,
            limit_seconds=getattr(args, "deadline_seconds", 0),
        )
        return _execute_command(args, operation_started_at)
    except CollectionTimeoutError as error:
        pending = "unknown" if error.pending_nodes is None else str(error.pending_nodes)
        deadline_fields = (
            error.deadline_clock.status_fields()
            if error.deadline_clock is not None
            else deadline_clock.status_fields()
        )
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "PYTEST KILLED phase=collection reason=attempt-deadline "
            f"signal={error.termination_signal} "
            f"elapsed={error.elapsed_seconds:.1f}s scope={error.result_scope} "
            f"finalized={error.finalized_nodes} passed={error.passed} "
            f"failed={error.failed} skipped={error.skipped} "
            f"pending={pending} node=unknown",
            file=sys.stderr,
            flush=True,
        )
        print(
            "RUNNER STATUS: state=killed phase=collection "
            "reason=attempt-deadline "
            f"signal={error.termination_signal} scope={error.result_scope} "
            f"finalized={error.finalized_nodes} passed={error.passed} "
            f"failed={error.failed} skipped={error.skipped} pending={pending} "
            f"{deadline_fields}",
            file=sys.stderr,
            flush=True,
        )
        return 3
    except (RunnerStateError, ValueError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        print(
            "RUNNER STATUS: state=failed "
            "reason=configuration-or-infrastructure-error "
            f"{deadline_clock.status_fields()}",
            file=sys.stderr,
            flush=True,
        )
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
