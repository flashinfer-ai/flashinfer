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

import functools
import hashlib
import json
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .fused_moe import trtllm_gen_fused_moe_build_inputs

CakeStepFunTarget = Literal["sm_100a", "sm_103a"]

_TARGET_FLAGS: dict[CakeStepFunTarget, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
_TARGET_MINOR: dict[CakeStepFunTarget, int] = {"sm_100a": 0, "sm_103a": 3}
_MODULE_URI: dict[CakeStepFunTarget, str] = {
    "sm_100a": "fused_moe_cake_stepfun_sm100",
    "sm_103a": "fused_moe_cake_stepfun_sm103",
}
_INVENTORY = "cake_stepfun_inventory.json"
_INVENTORY_SCHEMA = "flashinfer.cake_stepfun.inventory.v2"
_MANIFEST = "cake_stepfun_generated_manifest.cuh"
_RUNNER_SOURCE = "cake_stepfun_fc1_runner.cu"
_RUNNER_HEADER = "cake_stepfun_fc1_runner.cuh"
_BINDING_SOURCE = "cake_stepfun_moe_binding.cu"


def _get_cake_stepfun_csrc_dir() -> Path:
    """Locate the Cake StepFun sources in installed and source checkouts."""
    checkout = Path(__file__).resolve().parents[2] / "csrc" / "fused_moe" / "cake_stepfun"
    if checkout.exists():
        return checkout
    installed = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "cake_stepfun"
    if installed.exists():
        return installed
    raise FileNotFoundError(
        "Cake StepFun CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _load_inventory(csrc_dir: Path, target: CakeStepFunTarget) -> tuple[list[Path], dict[Path, list[str]]]:
    """Validate the generated inventory and return the exact-architecture device units."""
    generated_dir = csrc_dir / "generated"
    inventory_path = generated_dir / _INVENTORY
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Cake StepFun generated inventory was not found at {inventory_path}; "
            "install the generated StepFun FC1 kernel inventory"
        )
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cake StepFun inventory could not be read: {inventory_path}") from error
    if not isinstance(inventory, dict) or inventory.get("schema") != _INVENTORY_SCHEMA:
        raise ValueError(f"Cake StepFun inventory schema must be {_INVENTORY_SCHEMA}")
    kernels = inventory.get("kernels")
    files = inventory.get("files")
    if not isinstance(kernels, list) or not kernels or not isinstance(files, dict) or not files:
        raise ValueError("Cake StepFun inventory kernels and files must be non-empty")
    program = {
        key: inventory[key] for key in ("kernels", "families", "files") if key in inventory
    }
    program_bytes = (
        json.dumps(program, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if hashlib.sha256(program_bytes).hexdigest() != inventory.get("program_hash"):
        raise ValueError("Cake StepFun inventory program hash mismatch")
    repo_root = csrc_dir.parents[2].resolve()
    for relative, digest in files.items():
        source = (repo_root / relative).resolve()
        if not source.is_file() or not source.is_relative_to(repo_root):
            raise FileNotFoundError(f"Cake StepFun inventory source not found: {relative}")
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cake StepFun inventory file hash mismatch: {relative}")
    manifest_relative = inventory.get("manifest")
    if manifest_relative not in files or Path(manifest_relative).name != _MANIFEST:
        raise ValueError("Cake StepFun inventory must list the generated manifest header")
    families = inventory.get("families")
    if not isinstance(families, dict) or not families:
        raise ValueError("Cake StepFun inventory families must be a non-empty mapping")
    device_sources: list[Path] = []
    compile_flags: dict[Path, list[str]] = {}
    seen: set[tuple[str, int]] = set()
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict) or kernel.get("arch") not in _TARGET_FLAGS:
            raise ValueError(f"Cake StepFun inventory kernels[{index}] is invalid")
        device = kernel.get("device")
        flags = kernel.get("compile_flags")
        if device not in files or not isinstance(flags, list) or not all(isinstance(f, str) and f for f in flags):
            raise ValueError(f"Cake StepFun inventory kernels[{index}] device or compile_flags are invalid")
        if kernel["arch"] != target:
            continue
        family = kernel.get("family")
        tile = kernel.get("tile_n")
        if family not in families or not isinstance(tile, int) or tile not in families[family].get("tiles", ()):
            raise ValueError(f"Cake StepFun inventory kernels[{index}] family or tile is invalid")
        if (family, tile) in seen:
            raise ValueError(f"Cake StepFun inventory kernels[{index}] duplicates ({family}, tile {tile})")
        seen.add((family, tile))
        source = (repo_root / device).resolve()
        device_sources.append(source)
        compile_flags[source] = list(flags)
    for family, spec in families.items():
        missing = sorted(set(spec.get("tiles", ())) - {tile for fam, tile in seen if fam == family})
        if missing:
            raise ValueError(f"Cake StepFun inventory lacks {family} tiles {missing} for target {target}")
    if not device_sources:
        raise ValueError(f"Cake StepFun inventory has no kernels for target {target}")
    return device_sources, compile_flags


def get_cake_stepfun_fused_moe_uri(target: CakeStepFunTarget) -> str:
    """Return the exact-architecture Cake StepFun fused-MoE JIT module key."""
    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake StepFun target: {target}")
    return _MODULE_URI[target]


@functools.cache
def gen_cake_stepfun_fused_moe_module(target: CakeStepFunTarget) -> JitSpec:
    """Generate the exact-architecture Cake StepFun fused-MoE module.

    The module is the trtllm-gen fused-MoE host pipeline (routing, GEMM2, finalize
    and the batched-GEMM runner over the published cubins) compiled with
    ``-DCAKE_STEPFUN_FC1``, which replaces the GEMM1 stage by the exported Cake
    StepFun FC1 kernels of ``csrc/fused_moe/cake_stepfun/``. It exports the same
    TVM-FFI operations as ``fused_moe_trtllm_sm100`` under its own module name plus
    the standalone FC1 entry points ``cake_stepfun_fc1_families``,
    ``cake_stepfun_fc1_tiles`` and ``cake_stepfun_fc1``.
    """
    uri = get_cake_stepfun_fused_moe_uri(target)
    csrc_dir = _get_cake_stepfun_csrc_dir()
    device_sources, device_flags = _load_inventory(csrc_dir, target)
    for source in (
        csrc_dir / _RUNNER_SOURCE,
        csrc_dir / _RUNNER_HEADER,
        csrc_dir / _BINDING_SOURCE,
        csrc_dir / "generated" / _MANIFEST,
    ):
        if not source.is_file():
            raise FileNotFoundError(f"Cake StepFun source not found: {source}")
    target_flags = [
        *_TARGET_FLAGS[target],
        "-DCAKE_STEPFUN_FC1",
        f"-DFLASHINFER_CAKE_STEPFUN_TARGET_MINOR={_TARGET_MINOR[target]}",
        # This module defines the trtllm-gen fused-MoE host symbols with a GEMM1
        # runner of a different layout than the public module. Hide every host
        # symbol so the two libraries never bind to each other's definitions
        # (the TVM-FFI entry points and the cubin-loader hooks declare default
        # visibility explicitly).
        "-Xcompiler=-fvisibility=hidden",
        "-Xcompiler=-fvisibility-inlines-hidden",
    ]
    sources, cflags, include_paths = trtllm_gen_fused_moe_build_inputs(
        uri, enable_rubin=False, nvcc_flags=target_flags
    )
    spec = gen_jit_spec(
        uri,
        [*sources, csrc_dir / _RUNNER_SOURCE, csrc_dir / _BINDING_SOURCE, *device_sources],
        extra_cuda_cflags=cflags,
        extra_cuda_cflags_by_source={
            source: [*target_flags, *flags] for source, flags in device_flags.items()
        },
        extra_include_paths=[*include_paths, csrc_dir],
    )
    logger.info(f"Generated Cake StepFun fused-MoE {target} JIT spec: {spec.name}")
    return spec


__all__ = [
    "CakeStepFunTarget",
    "gen_cake_stepfun_fused_moe_module",
    "get_cake_stepfun_fused_moe_uri",
]
