#!/usr/bin/env python3
"""Report which requirements the current environment does not already satisfy.

Prints one unsatisfied requirement per line on stdout, ready to pass to
`pip install`. Exit codes: 0 all satisfied, 1 some printed, 2 the check could
not run, so callers can fall back to a full dependency sync.
"""

import sys
from importlib.metadata import PackageNotFoundError, version

try:
    from packaging.requirements import InvalidRequirement, Requirement
except ImportError as exc:  # pragma: no cover - packaging ships in every CI image
    print(f"cannot check requirements: {exc}", file=sys.stderr)
    raise SystemExit(2) from exc


def parse_requirements(path):
    """Read a requirements file, ignoring comments, blanks, and pip options."""
    requirements = []
    try:
        with open(path, encoding="utf-8") as handle:
            lines = handle.read().splitlines()
    except OSError as exc:
        print(f"skipping {path}: {exc}", file=sys.stderr)
        return requirements

    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        try:
            requirements.append(Requirement(line))
        except InvalidRequirement as exc:
            print(f"{path}: cannot parse {line!r}: {exc}", file=sys.stderr)
            raise SystemExit(2) from exc
    return requirements


def main(paths):
    if not paths:
        print("usage: check_requirements.py REQUIREMENTS_FILE...", file=sys.stderr)
        return 2

    unsatisfied = []
    for path in paths:
        for req in parse_requirements(path):
            if req.marker is not None and not req.marker.evaluate():
                continue
            try:
                installed = version(req.name)
            except PackageNotFoundError:
                print(f"{req.name}: not installed", file=sys.stderr)
                unsatisfied.append(str(req))
                continue
            if not req.specifier.contains(installed, prereleases=True):
                print(
                    f"{req.name}: installed {installed} does not satisfy "
                    f"{req.specifier}",
                    file=sys.stderr,
                )
                unsatisfied.append(str(req))

    for req in unsatisfied:
        print(req)
    return 1 if unsatisfied else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
