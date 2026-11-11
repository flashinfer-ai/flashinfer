#!/usr/bin/env python3
"""Fail fast when an AOT spec's explicit nvcc targets break a provider wheel.

Release and nightly builds inspect each built provider wheel with ``cuobjdump``
and reject modules that carry SASS for another architecture or any PTX. This
script catches the same mistake for the whole AOT spec list before anything is
compiled: it generates the specs for one provider architecture exactly as the
provider build does, then checks the ``-gencode`` flags each spec names. Specs
without explicit gencodes use the compilation context's targets, which are
already limited to ``FLASHINFER_CUDA_ARCH_LIST``, so only explicit flags can
cause a mismatch.

Needs torch and a CUDA toolkit (spec generation queries ``nvcc``), not a GPU.
Run one provider per process: the target list is fixed when ``flashinfer`` is
first imported.

    python scripts/check_jit_spec_gencodes.py --provider 8.9
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def spec_flag_lists(spec) -> list[tuple[str, list[str]]]:
    """Return ``(label, flags)`` for every nvcc flag list a spec compiles with."""
    flag_lists = []
    shared = getattr(spec, "extra_cuda_cflags", None)
    if shared:
        flag_lists.append((spec.name, list(shared)))
    for source, flags in (
        getattr(spec, "extra_cuda_cflags_by_source", None) or {}
    ).items():
        flag_lists.append((f"{spec.name} ({Path(source).name})", list(flags)))
    return flag_lists


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--provider",
        required=True,
        help="Provider architecture, e.g. 8.9, 9.0a or sm120f",
    )
    args = parser.parse_args()

    package_config = _load_module(
        "_flashinfer_provider_package_config",
        REPO_ROOT / "flashinfer-jit-cache-provider" / "package_config.py",
    )
    nvcc_arch, provider_tag = package_config.normalize_cuda_architecture(args.provider)
    # Must be set before flashinfer is imported: the compilation context reads it then.
    os.environ["FLASHINFER_CUDA_ARCH_LIST"] = nvcc_arch

    cuda_architecture = _load_module(
        "_flashinfer_jit_cuda_architecture",
        REPO_ROOT / "flashinfer" / "jit" / "_cuda_architecture.py",
    )
    from flashinfer import aot

    with tempfile.TemporaryDirectory(prefix="flashinfer-spec-lint-") as build_dir:
        specs = aot.generate_jit_specs(REPO_ROOT, Path(build_dir))

    checked = 0
    problems: list[str] = []
    for spec in specs:
        for label, flags in spec_flag_lists(spec):
            issues = cuda_architecture.provider_flag_issues(flags, provider_tag)
            checked += bool(cuda_architecture.cuda_targets_from_nvcc_flags(flags)[0])
            problems.extend(f"  {label}: {issue}" for issue in issues)

    print(
        f"Provider {provider_tag}: {len(specs)} specs, {checked} with explicit "
        f"-gencode targets checked"
    )
    if problems:
        print(f"ERROR: {len(problems)} spec(s) break the {provider_tag} provider rule:")
        print("\n".join(problems))
        print(
            "Register such a module only for the architectures it can serve, or "
            "derive its targets from current_compilation_context "
            "(see gen_ulysses_lowp_module in flashinfer/jit/comm.py)."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
