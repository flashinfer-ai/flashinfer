#!/usr/bin/env python3
"""Check every provider wheel in a wheelhouse for foreign CUDA targets and PTX.

The release and nightly builds run this inspection through
``verify_jit_cache_provider_wheelhouse.py`` on one provider at a time. The PR
AOT lane builds all providers into a single wheelhouse, so this walks it:
each ``flashinfer-jit-cache-<provider>`` wheel is validated and its modules are
listed with ``cuobjdump``; a module carrying SASS another architecture cannot run,
or any PTX, fails the run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from jit_cache_provider_validation import (
    Wheel,
    inspect_cuda_architectures,
    validate_provider,
)

_PROVIDER_PREFIX = "flashinfer-jit-cache-"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--wheelhouse", type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--cuobjdump", type=Path, required=True)
    args = parser.parse_args()

    failures = []
    providers = []
    for path in sorted(args.wheelhouse.glob("*.whl")):
        wheel = Wheel.open(path)
        distribution = wheel.distribution.replace("_", "-").lower()
        provider = distribution.removeprefix(_PROVIDER_PREFIX)
        # Skip the shim (no trailing dash) and every other non-provider wheel.
        if not distribution.startswith(_PROVIDER_PREFIX) or not provider.startswith(
            "sm"
        ):
            continue
        providers.append(provider)
        try:
            _, module_paths = validate_provider(wheel, provider, args.version)
            inspect_cuda_architectures(
                wheel, module_paths, provider, args.cuobjdump, strict=True
            )
        except ValueError as error:
            failures.append(f"{provider}: {error}")
            continue
        print(f"{provider}: {len(module_paths)} modules, targets and PTX OK")

    if not providers:
        print(f"ERROR: no provider wheels found in {args.wheelhouse}")
        return 1
    if failures:
        print("ERROR: provider architecture validation failed:")
        print("\n".join(f"  {failure}" for failure in failures))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
