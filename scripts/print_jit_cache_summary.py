#!/usr/bin/env python3
"""
Print aggregated JIT cache coverage summary from multiple pytest runs.

This script reads the JSON file written by pytest (via conftest.py) and
prints a unified summary of all missing JIT cache modules across all test runs.

Usage:
    python scripts/print_jit_cache_summary.py /path/to/aggregate.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, Set


def print_jit_cache_summary(aggregate_file: str):
    """Read and print JIT cache coverage summary from aggregate file"""
    aggregate_path = Path(aggregate_file)

    if not aggregate_path.exists():
        print("No JIT cache coverage data found.")
        print(f"Expected file: {aggregate_file}")
        return

    # Read all entries from the file
    missing_modules: Set[tuple] = set()
    with open(aggregate_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            missing_modules.add(
                (entry["test_name"], entry["module_name"], entry["spec_info"])
            )

    if not missing_modules:
        print("✅ All tests passed - no missing JIT cache modules!")
        return

    # Print summary
    print("=" * 80)
    print("flashinfer-jit-cache Package Coverage Report")
    print("=" * 80)
    print()
    print("This report shows the coverage of the flashinfer-jit-cache package.")
    print(
        "Tests are skipped when required modules are not found in the installed JIT cache."
    )
    print()
    print(
        f"⚠️  {len(missing_modules)} test(s) skipped due to missing JIT cache modules:"
    )
    print()

    # Group by module name
    module_to_tests: Dict[str, Dict] = {}
    for test_name, module_name, spec_info in missing_modules:
        if module_name not in module_to_tests:
            module_to_tests[module_name] = {"tests": [], "spec_info": spec_info}
        module_to_tests[module_name]["tests"].append(test_name)

    for module_name in sorted(module_to_tests.keys()):
        info = module_to_tests[module_name]
        print(f"Module: {module_name}")
        print(f"  Spec: {info['spec_info']}")
        print(f"  Affected tests ({len(info['tests'])}):")
        for test in sorted(info["tests"]):
            print(f"    - {test}")
        print()

    print("These tests require JIT compilation but FLASHINFER_DISABLE_JIT=1 was set.")
    print(
        "To improve coverage, add the missing modules to the flashinfer-jit-cache build configuration."
    )
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/print_jit_cache_summary.py <aggregate_file>")
        sys.exit(1)

    print_jit_cache_summary(sys.argv[1])
