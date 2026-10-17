#!/usr/bin/env python3
"""Run one finite RoCE model configuration and record a checked result."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", help="Case name from cases.json, or 'list'.")
    parser.add_argument("--quint", default="quint", help="Quint executable path.")
    parser.add_argument("--node", help="Node executable when --quint names cli.js.")
    parser.add_argument("--output", type=Path, default=Path("formal-results"))
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--include-expensive", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "cases.json").read_text())
    cases = dict(manifest["cases"])
    if args.include_expensive:
        cases.update(manifest.get("optional_cases", {}))
    if args.case == "list":
        print("\n".join(cases))
        return 0
    case = cases.get(args.case)
    if case is None:
        parser.error(f"Unknown case: {args.case}")
    executable = ([args.node] if args.node else []) + [args.quint]
    version = subprocess.run(
        executable + ["--version"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    ).stdout.strip()
    if version != manifest["quint_version"]:
        parser.error(f"Expected Quint {manifest['quint_version']}, found {version!r}")
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output.resolve()
    command = executable + ["verify"]
    command += [str(root / case["file"]), "--main", case["module"]]
    command += ["--backend", case["backend"], "--invariant", case["invariant"]]
    command += ["--apalache-version", manifest["apalache_version"], "--verbosity", "3"]
    if case["backend"] == "tlc":
        command += ["--tlc-config", str(root / "tlc.json")]
    if "temporal" in case:
        command += ["--temporal", case["temporal"]]
    if "max_steps" in case:
        command += ["--max-steps", str(case["max_steps"])]
    if "step" in case:
        command += ["--step", case["step"]]
    environment = dict(os.environ)
    environment.setdefault("JAVA_TOOL_OPTIONS", "-Xmx2g -XX:ActiveProcessorCount=4")
    environment.setdefault("JVM_ARGS", "-Xmx2g -XX:ActiveProcessorCount=4")
    start = time.monotonic()
    log_path = output / f"{args.case}.log"
    timed_out = False
    with log_path.open("w") as log:
        proc = subprocess.Popen(
            command,
            cwd=output,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                returncode = proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                returncode = proc.wait()
    text = log_path.read_text()
    passed = returncode == 0 and "[ok] No violation found" in text
    # Each negative case requests exactly one invariant. TLC renames that
    # expression q_inv; backend failures and temporal failures do not match.
    violated = (
        returncode != 0
        and "[violation]" in text
        and (
            "Error: Invariant q_inv is violated." in text
            if case["backend"] == "tlc"
            else "Invariant q_inv is violated" in text
            or bool(
                re.search(
                    r"State \d+: state invariant 0 (?:is )?violated",
                    text,
                    re.IGNORECASE,
                )
            )
        )
    )
    accepted = not timed_out and (passed if case["expected"] == "pass" else violated)
    counts = re.findall(
        r"([\d,]+) states generated, ([\d,]+) distinct states found", text
    )
    result = {
        "case": args.case,
        "quint_version": version,
        "configuration": case,
        "command": command,
        "elapsed_seconds": time.monotonic() - start,
        "returncode": returncode,
        "timed_out": timed_out,
        "outcome": "pass" if passed else "counterexample" if violated else "error",
        "accepted": accepted,
        "states": counts[-1] if counts else None,
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.glob("*.qnt"))
            + [root / "cases.json", root / "tlc.json", root / "run.py"]
        },
        "log": str(log_path),
    }
    (output / f"{args.case}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if accepted else 1


if __name__ == "__main__":
    sys.exit(main())
