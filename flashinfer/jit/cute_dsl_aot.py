"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

"""GPU-free compilation and packaging shared by registered CuTe-DSL operations."""

import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from . import env as jit_env
from .cute_dsl_core import (
    JitSpecCuteDsl,
    cap_cute_dsl_compile_workers,
    sanitize_symbol_name,
)


@dataclass(frozen=True)
class CuteDslAOTOperation:
    """An operation owns its targets, named configurations, and spec factory.

    Configurations must be picklable, and factories must be module-level
    functions so CPU builds can use spawned processes. Tactics, tensor layouts,
    and launch arguments stay inside the operation's implementation.
    """

    name: str
    architectures: frozenset[str]
    configs: Callable[[str], Mapping[str, Any]]
    make_spec: Callable[[Any], JitSpecCuteDsl]


def get_aot_operations() -> tuple[CuteDslAOTOperation, ...]:
    """Register operation adapters here; imports must not initialize a GPU."""
    from .gemm.cute_dsl_fp4 import MM_FP4_OPERATION

    return (MM_FP4_OPERATION,)


def _compile_aot_kernel(operation, name, config, arch, jit_dir) -> Path:
    """Export one registered specialization; never load or execute its GPU code."""
    os.environ["CUTE_DSL_ARCH"] = arch.replace("sm", "sm_", 1)
    jit_env.FLASHINFER_JIT_DIR = jit_dir
    # Installed providers must not satisfy an incremental build's local outputs.
    jit_env.FLASHINFER_AOT_PROVIDERS = ()
    spec = operation.make_spec(config)
    if (
        spec.module_name != operation.name
        or spec.kernel_name != name
        or spec.expected_meta["arch"] != arch
    ):
        raise RuntimeError(
            f"CuTe-DSL spec does not match registered configuration {operation.name}/{arch}/{name}"
        )
    if not spec.is_compiled:
        spec.compile_and_persist()
    if not spec.is_compiled:
        raise RuntimeError(f"CuTe-DSL artifact was not persisted: {spec.object_path}")
    return spec.object_path


def compile_and_package_cute_dsl(
    arch: str,
    out_dir: Optional[Path],
    operations: Optional[Sequence[CuteDslAOTOperation]] = None,
) -> None:
    """Build registered operations for one target with strict artifact exports."""
    if operations is None:
        operations = get_aot_operations()
    selected = [op for op in operations if arch in op.architectures]
    if not selected:
        return
    if len({op.name for op in selected}) != len(selected):
        raise ValueError("Duplicate CuTe-DSL AOT operation names")
    previous_arch = os.environ.get("CUTE_DSL_ARCH")
    os.environ["CUTE_DSL_ARCH"] = arch.replace("sm", "sm_", 1)
    try:
        workers = max(
            1,
            cap_cute_dsl_compile_workers(
                int(os.environ.get("FLASHINFER_CUTE_DSL_AOT_WORKERS", "4"))
            ),
        )
        for operation in selected:
            configs = operation.configs(arch)
            if not configs:
                raise RuntimeError(
                    f"CuTe-DSL operation {operation.name} has no configurations for {arch}"
                )
            if sanitize_symbol_name(operation.name) != operation.name or any(
                sanitize_symbol_name(name) != name or not name for name in configs
            ):
                raise ValueError(
                    f"Unsafe CuTe-DSL operation/kernel name: {operation.name}"
                )
            print(
                f"CuTe-DSL {arch}/{operation.name}: exporting {len(configs)} kernels with {min(workers, len(configs))} workers",
                flush=True,
            )
            pool = ProcessPoolExecutor(
                max_workers=min(workers, len(configs)), mp_context=get_context("spawn")
            )
            artifacts = []
            try:
                futures = [
                    pool.submit(
                        _compile_aot_kernel,
                        operation,
                        name,
                        config,
                        arch,
                        jit_env.FLASHINFER_JIT_DIR,
                    )
                    for name, config in configs.items()
                ]
                for index, future in enumerate(futures, 1):
                    path = future.result()
                    artifacts.append(path)
                    print(
                        f"CuTe-DSL {arch}/{operation.name}: {index}/{len(configs)} {path.stem}",
                        flush=True,
                    )
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
            # Verify the entire inventory after workers finish, including when
            # no packaging directory is requested. Incompatible sibling specs
            # must not wipe artifacts exported earlier in the same build.
            for path in artifacts:
                if not path.is_file() or not (path.parent / "meta.json").is_file():
                    raise RuntimeError(f"Missing CuTe-DSL AOT artifact: {path}")
            if out_dir is not None:
                module_dirs = {path.parent for path in artifacts}
                for source in module_dirs:
                    destination = out_dir / source.name
                    if source.resolve() == destination.resolve():
                        raise ValueError(
                            "CuTe-DSL output directory must differ from the build cache"
                        )
                    if destination.exists():
                        shutil.rmtree(destination)
                    destination.mkdir(parents=True, exist_ok=True)
                    for path in artifacts:
                        if path.parent == source:
                            shutil.copy2(path, destination)
                    shutil.copy2(source / "meta.json", destination)
    finally:
        if previous_arch is None:
            os.environ.pop("CUTE_DSL_ARCH", None)
        else:
            os.environ["CUTE_DSL_ARCH"] = previous_arch


if __name__ == "__main__":
    import argparse
    import json

    operations = get_aot_operations()
    parser = argparse.ArgumentParser(
        description="Build registered CuTe-DSL provider artifacts without a GPU."
    )
    parser.add_argument("--operation", choices=[op.name for op in operations])
    parser.add_argument(
        "--arch",
        choices=sorted(set().union(*(op.architectures for op in operations))),
        required=True,
    )
    parser.add_argument("--build-dir", type=Path)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the kernel inventory without compiling",
    )
    args = parser.parse_args()
    if args.operation:
        operations = tuple(op for op in operations if op.name == args.operation)
    if args.build_dir is not None:
        jit_env.FLASHINFER_JIT_DIR = args.build_dir / "cached_ops"
    if args.list:
        print(
            json.dumps(
                [
                    name
                    for op in operations
                    if args.arch in op.architectures
                    for name in op.configs(args.arch)
                ],
                indent=2,
            )
        )
    else:
        compile_and_package_cute_dsl(args.arch, args.out_dir, operations)
