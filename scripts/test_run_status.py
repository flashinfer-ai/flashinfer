"""Write optional CI completion reports for the communication-test wrappers.

Each pytest invocation owns a receipt, so concurrent MPI processes and shell
workers never overwrite one another. Only reaching the wrapper's normal end
marks the overall operation complete. A killed invocation retains its pending
receipt even if the legacy wrapper otherwise returns success.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import uuid


def write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as output:
        json.dump(report, output)
        temporary = output.name
    os.replace(temporary, path)


class PytestCompletion:
    """Observe collection and session completion inside each distributed worker."""

    def __init__(self) -> None:
        self.complete = False
        self.errors: list[str] = []
        self.tests_executed = 0

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.errors.append(f"pytest collection failed: {report.nodeid}")

    def pytest_internalerror(self, excrepr) -> None:
        self.errors.append(f"pytest internal error: {excrepr}")

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "call" and not report.skipped:
            self.tests_executed += 1

    def pytest_sessionfinish(self) -> None:
        self.complete = True


def local_worker_identity() -> tuple[int, int]:
    for count_name, rank_name in (
        ("LOCAL_WORLD_SIZE", "LOCAL_RANK"),
        ("OMPI_COMM_WORLD_LOCAL_SIZE", "OMPI_COMM_WORLD_LOCAL_RANK"),
        ("MPI_LOCALNRANKS", "MPI_LOCALRANKID"),
    ):
        if count_name in os.environ:
            return int(os.environ[count_name]), int(os.environ[rank_name])
    return 1, 0


def run_pytest(receipt: Path, arguments: list[str]) -> int:
    worker = receipt.parent / receipt.stem / f"worker-{uuid.uuid4().hex}.json"
    write_report(worker, {"complete": False})
    # Each CI container owns its result directory; require all local workers,
    # including ranks that failed before they could create a pending receipt.
    worker_count, worker_rank = local_worker_identity()
    # Import only after writing a pending receipt, including import failures in
    # the incomplete-run check. A segfault leaves this receipt pending too.
    import pytest

    completion = PytestCompletion()
    status = int(pytest.main(arguments, plugins=[completion]))
    write_report(
        worker,
        {
            "complete": completion.complete,
            "exit_code": status,
            "infrastructure_errors": completion.errors,
            "worker_count": worker_count,
            "worker_rank": worker_rank,
            "tests_executed": completion.tests_executed,
        },
    )
    return status


def complete_script(directory: Path, status: int) -> None:
    errors = []
    complete = True
    test_failures = False
    tests_executed = 0
    for path in sorted(directory.glob("pytest-*.json")):
        try:
            receipt = json.loads(path.read_text())
            if receipt["complete"] is not True:
                complete = False
                errors.append(f"pytest invocation did not finish: {path.name}")
            workers = list((directory / path.stem).glob("worker-*.json"))
            if not workers:
                complete = False
                errors.append(f"launcher produced no worker results: {path.name}")
            worker_failures = False
            expected_counts = set()
            worker_ranks = []
            for worker in workers:
                outcome = json.loads(worker.read_text())
                if outcome["complete"] is not True:
                    complete = False
                    errors.append(f"pytest worker did not finish: {worker.name}")
                    continue
                expected_counts.add(outcome["worker_count"])
                worker_ranks.append(outcome["worker_rank"])
                worker_errors = outcome["infrastructure_errors"]
                count = outcome["tests_executed"]
                if type(count) is not int or count < 0:
                    raise ValueError("tests executed must be a nonnegative integer")
                tests_executed += count
                if not isinstance(worker_errors, list):
                    raise ValueError("worker infrastructure errors must be a list")
                errors.extend(worker_errors)
                if outcome["exit_code"] not in (0, 1):
                    errors.append(f"pytest infrastructure exit {outcome['exit_code']}")
                worker_failures |= outcome["exit_code"] == 1
            if (
                len(expected_counts) != 1
                or next(iter(expected_counts)) < 1
                or sorted(worker_ranks) != list(range(next(iter(expected_counts))))
            ):
                complete = False
                errors.append(f"launcher is missing expected worker ranks: {path.name}")
            if receipt.get("exit_code") not in (0, 1) or (
                receipt.get("exit_code") == 1 and not worker_failures
            ):
                errors.append(
                    f"launcher exit is not explained by pytest assertions: {path.name}"
                )
            test_failures |= worker_failures
        except (OSError, ValueError, TypeError, KeyError):
            complete = False
            errors.append(f"invalid pytest receipt: {path.name}")
    if status not in (0, 1) or (status == 1 and not test_failures):
        errors.append(f"script exit {status} is not explained by pytest assertions")
    write_report(
        directory / "script-result.json",
        {
            "schema_version": 1,
            "complete": complete,
            "exit_code": status,
            "infrastructure_errors": errors,
            "tests_executed": tests_executed,
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument(
        "operation", choices=("begin", "start", "finish", "complete", "run")
    )
    parser.add_argument("--status", type=int, default=0)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, default=[])
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    if args.operation == "begin":
        write_report(
            args.directory / "script-result.json",
            {
                "schema_version": 1,
                "complete": False,
                "exit_code": None,
                "infrastructure_errors": [],
                "tests_executed": 0,
            },
        )
    elif args.operation == "start":
        path = args.directory / f"pytest-{uuid.uuid4().hex}.json"
        write_report(path, {"complete": False})
        print(path)
    elif args.operation in ("finish", "run"):
        if args.receipt is None or args.receipt.parent != args.directory:
            parser.error("receipt must belong to the result directory")
        if args.operation == "run":
            return run_pytest(args.receipt, args.pytest_args)
        write_report(args.receipt, {"complete": True, "exit_code": args.status})
    else:
        complete_script(args.directory, args.status)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
