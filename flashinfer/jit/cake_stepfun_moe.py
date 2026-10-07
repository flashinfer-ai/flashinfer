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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, logger, sm100a_nvcc_flags, sm103a_nvcc_flags
from .fused_moe import trtllm_gen_fused_moe_build_inputs

CakeStepFunTarget = Literal["sm_100a", "sm_103a"]
CakeStepFunStage = Literal["routing", "fc1", "requant", "fc2", "finalize"]

#: Pipeline stages of the fused-MoE forward, in execution order. ``fc1`` is mandatory
#: (it is the module's reason to exist); the other four form the full Cake path.
CAKE_STEPFUN_STAGES: tuple[CakeStepFunStage, ...] = (
    "routing",
    "fc1",
    "requant",
    "fc2",
    "finalize",
)
#: Environment switch of the full path: ``auto`` (default: full path when the
#: inventory covers every stage for the target), ``1`` (require it) or ``0``
#: (FC1-only module over the trtllm-gen routing, GEMM2 and finalize kernels).
CAKE_STEPFUN_FULL_PATH_ENV = "FLASHINFER_CAKE_STEPFUN_FULL_PATH"

_TARGET_FLAGS: dict[CakeStepFunTarget, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
_TARGET_MINOR: dict[CakeStepFunTarget, int] = {"sm_100a": 0, "sm_103a": 3}
_MODULE_URI: dict[tuple[CakeStepFunTarget, bool], str] = {
    ("sm_100a", False): "fused_moe_cake_stepfun_sm100",
    ("sm_103a", False): "fused_moe_cake_stepfun_sm103",
    ("sm_100a", True): "fused_moe_cake_stepfun_full_sm100",
    ("sm_103a", True): "fused_moe_cake_stepfun_full_sm103",
}
_INVENTORY = "cake_stepfun_inventory.json"
_INVENTORY_SCHEMA = "flashinfer.cake_stepfun.inventory.v4"
_MANIFEST = "cake_stepfun_generated_manifest.cuh"
_FC1_SOURCE = "cake_stepfun_fc1_runner.cu"
_FC1_HEADER = "cake_stepfun_fc1_runner.cuh"
_STAGES_SOURCE = "cake_stepfun_stages.cu"
_STAGES_HEADER = "cake_stepfun_stages.cuh"
_ABI_HEADER = "cake_stepfun_abi.cuh"
_BINDING_SOURCE = "cake_stepfun_moe_binding.cu"
# Generated-manifest kernel table every stage must define when the inventory lists it.
_STAGE_TABLE: dict[str, str] = {
    "routing": "kRoutingKernels",
    "fc1": "kFc1Kernels",
    "requant": "kRequantKernels",
    "fc2": "kFc2Kernels",
    "finalize": "kFinalizeKernels",
}
# Stages whose records are (family, tile) pairs declared in a families mapping.
_TILED_STAGES: dict[str, str] = {"fc1": "families", "fc2": "fc2_families"}
_HASHED_KEYS = ("kernels", "families", "fc2_families", "files")
#: Routing input kinds a routing record declares (``RoutingInput`` of the ABI header).
CAKE_STEPFUN_ROUTING_INPUTS: tuple[str, ...] = ("scores", "topk_ids")
#: Expert-weight dtypes a finalize record declares (``FinalizeKernelSpec.expert_weights_dtype``).
CAKE_STEPFUN_FINALIZE_WEIGHT_DTYPES: tuple[str, ...] = ("float32", "bfloat16")


def _get_cake_stepfun_csrc_dir() -> Path:
    """Locate the Cake StepFun sources in installed and source checkouts."""
    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "fused_moe" / "cake_stepfun"
    )
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


@dataclass(frozen=True)
class CakeStepFunInventory:
    """Validated view of the generated inventory (all targets)."""

    path: Path
    repo_root: Path
    manifest: Path
    #: Per target: the stages that have at least one exported kernel.
    stages: dict[CakeStepFunTarget, frozenset[str]]
    #: Per target: device translation units and their extra compile flags.
    device_sources: dict[CakeStepFunTarget, list[Path]]
    compile_flags: dict[CakeStepFunTarget, dict[Path, list[str]]]
    #: Per target: routing input kinds (``scores`` / ``topk_ids``) with a kernel.
    routing_inputs: dict[CakeStepFunTarget, frozenset[str]]
    #: Per target: expert-weight dtypes (``float32`` / ``bfloat16``) with a finalize kernel.
    finalize_weight_dtypes: dict[CakeStepFunTarget, frozenset[str]]
    #: Per target: FC2 kernel symbol -> trtllm-gen configuration (function name) it ports.
    fc2_native_configs: dict[CakeStepFunTarget, dict[str, str]]

    def missing_stages(self, target: CakeStepFunTarget) -> tuple[str, ...]:
        present = self.stages.get(target, frozenset())
        return tuple(stage for stage in CAKE_STEPFUN_STAGES if stage not in present)


def _validate_tiled_record(
    kernel: dict, index: int, stage: str, declared: dict, seen: set
) -> None:
    family = kernel.get("family")
    tile = kernel.get("tile_n")
    if (
        family not in declared
        or not isinstance(tile, int)
        or tile not in declared[family].get("tiles", ())
    ):
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] ({stage}) family or tile is "
            f"not declared in {_TILED_STAGES[stage]}"
        )
    key = (kernel["arch"], stage, family, tile)
    if key in seen:
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] duplicates ({stage}, {family}, "
            f"tile {tile}) for {kernel['arch']}"
        )
    seen.add(key)
    if stage == "fc2":
        native = kernel.get("native_config")
        if not isinstance(native, str) or not native.startswith("bmm_"):
            raise ValueError(
                f"Cake StepFun inventory kernels[{index}] (fc2) must name the trtllm-gen "
                "configuration its kernel is a port of (native_config = bmm_* function "
                f"name), got {native!r}"
            )


def _validate_variant_record(kernel: dict, index: int, stage: str, seen: set) -> None:
    variant = kernel.get("variant")
    if not isinstance(variant, str) or not variant:
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] ({stage}) needs a non-empty variant"
        )
    key = (kernel["arch"], stage, variant)
    if key in seen:
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] duplicates ({stage}, {variant}) "
            f"for {kernel['arch']}"
        )
    seen.add(key)
    if stage == "routing" and kernel.get("input") not in CAKE_STEPFUN_ROUTING_INPUTS:
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] (routing) input must be one of "
            f"{CAKE_STEPFUN_ROUTING_INPUTS}, got {kernel.get('input')!r}"
        )
    if (
        stage == "finalize"
        and kernel.get("expert_weights_dtype")
        not in CAKE_STEPFUN_FINALIZE_WEIGHT_DTYPES
    ):
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] (finalize) expert_weights_dtype must "
            f"be one of {CAKE_STEPFUN_FINALIZE_WEIGHT_DTYPES}, got "
            f"{kernel.get('expert_weights_dtype')!r}"
        )


def _pre_kernel_device(kernel: dict, index: int, files: dict) -> str | None:
    """Device unit of a routing record's leading kernel (``pre_kernel``), if any."""
    pre = kernel.get("pre_kernel")
    if pre is None:
        return None
    if (
        not isinstance(pre, dict)
        or not isinstance(pre.get("kernel_symbol"), str)
        or not pre["kernel_symbol"]
        or pre.get("device") not in files
    ):
        raise ValueError(
            f"Cake StepFun inventory kernels[{index}] (routing) pre_kernel needs a "
            "kernel_symbol and a device unit listed in files"
        )
    return pre["device"]


@functools.cache
def load_cake_stepfun_inventory(csrc_dir: Path | None = None) -> CakeStepFunInventory:
    """Read and validate the generated inventory.

    The inventory (schema ``flashinfer.cake_stepfun.inventory.v4``) lists every
    exported device translation unit with its ``stage`` (one of
    :data:`CAKE_STEPFUN_STAGES`), ``arch``, ``device`` path, ``compile_flags`` and
    launch metadata. ``fc1`` / ``fc2`` records are ``(family, tile_n)`` pairs declared
    in ``families`` / ``fc2_families``; ``routing``, ``requant`` and ``finalize``
    records carry a unique ``variant`` (routing records also their ``input`` kind
    and, for the two-kernel large-token path, a ``pre_kernel`` unit; finalize
    records their ``expert_weights_dtype``). Every ``fc2`` record names the
    trtllm-gen configuration its kernel is a port of (``native_config``, a ``bmm_*``
    function name; :func:`cake_stepfun_fc2_native_config`). ``program_hash`` seals
    ``kernels``, ``families``, ``fc2_families`` and the per-file ``files`` digests,
    and the generated manifest header must define the kernel table of every listed
    stage.
    """
    csrc_dir = _get_cake_stepfun_csrc_dir() if csrc_dir is None else csrc_dir
    generated_dir = csrc_dir / "generated"
    inventory_path = generated_dir / _INVENTORY
    if not inventory_path.is_file():
        raise FileNotFoundError(
            f"Cake StepFun generated inventory was not found at {inventory_path}; "
            "install the generated StepFun kernel inventory"
        )
    try:
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cake StepFun inventory could not be read: {inventory_path}"
        ) from error
    if not isinstance(inventory, dict) or inventory.get("schema") != _INVENTORY_SCHEMA:
        raise ValueError(f"Cake StepFun inventory schema must be {_INVENTORY_SCHEMA}")
    kernels = inventory.get("kernels")
    files = inventory.get("files")
    if (
        not isinstance(kernels, list)
        or not kernels
        or not isinstance(files, dict)
        or not files
    ):
        raise ValueError("Cake StepFun inventory kernels and files must be non-empty")
    program = {key: inventory[key] for key in _HASHED_KEYS if key in inventory}
    program_bytes = (
        json.dumps(program, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")
    if hashlib.sha256(program_bytes).hexdigest() != inventory.get("program_hash"):
        raise ValueError("Cake StepFun inventory program hash mismatch")
    repo_root = csrc_dir.parents[2].resolve()
    for relative, digest in files.items():
        source = (repo_root / relative).resolve()
        if not source.is_file() or not source.is_relative_to(repo_root):
            raise FileNotFoundError(
                f"Cake StepFun inventory source not found: {relative}"
            )
        if hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Cake StepFun inventory file hash mismatch: {relative}")
    manifest_relative = inventory.get("manifest")
    if manifest_relative not in files or Path(manifest_relative).name != _MANIFEST:
        raise ValueError(
            "Cake StepFun inventory must list the generated manifest header"
        )
    declared: dict[str, dict] = {}
    for stage, key in _TILED_STAGES.items():
        mapping = inventory.get(key)
        if stage == "fc1" and (not isinstance(mapping, dict) or not mapping):
            raise ValueError(
                "Cake StepFun inventory families must be a non-empty mapping"
            )
        if mapping is not None and not isinstance(mapping, dict):
            raise ValueError(f"Cake StepFun inventory {key} must be a mapping")
        declared[stage] = mapping or {}

    stages: dict[CakeStepFunTarget, set[str]] = {t: set() for t in _TARGET_FLAGS}
    device_sources: dict[CakeStepFunTarget, list[Path]] = {t: [] for t in _TARGET_FLAGS}
    compile_flags: dict[CakeStepFunTarget, dict[Path, list[str]]] = {
        t: {} for t in _TARGET_FLAGS
    }
    routing_inputs: dict[CakeStepFunTarget, set[str]] = {
        t: set() for t in _TARGET_FLAGS
    }
    finalize_dtypes: dict[CakeStepFunTarget, set[str]] = {
        t: set() for t in _TARGET_FLAGS
    }
    fc2_native_configs: dict[CakeStepFunTarget, dict[str, str]] = {
        t: {} for t in _TARGET_FLAGS
    }
    seen: set = set()
    for index, kernel in enumerate(kernels):
        if not isinstance(kernel, dict) or kernel.get("arch") not in _TARGET_FLAGS:
            raise ValueError(f"Cake StepFun inventory kernels[{index}] is invalid")
        stage = kernel.get("stage")
        if stage not in CAKE_STEPFUN_STAGES:
            raise ValueError(
                f"Cake StepFun inventory kernels[{index}] stage must be one of "
                f"{CAKE_STEPFUN_STAGES}, got {stage!r}"
            )
        device = kernel.get("device")
        flags = kernel.get("compile_flags")
        symbol = kernel.get("kernel_symbol")
        if (
            device not in files
            or not isinstance(flags, list)
            or not all(isinstance(f, str) and f for f in flags)
            or not isinstance(symbol, str)
            or not symbol
        ):
            raise ValueError(
                f"Cake StepFun inventory kernels[{index}] device, compile_flags or "
                "kernel_symbol are invalid"
            )
        if stage in _TILED_STAGES:
            _validate_tiled_record(kernel, index, stage, declared[stage], seen)
        else:
            _validate_variant_record(kernel, index, stage, seen)
        target: CakeStepFunTarget = kernel["arch"]
        stages[target].add(stage)
        if stage == "fc2":
            fc2_native_configs[target][symbol] = kernel["native_config"]
        source = (repo_root / device).resolve()
        device_sources[target].append(source)
        compile_flags[target][source] = list(flags)
        if stage == "routing":
            routing_inputs[target].add(kernel["input"])
            pre_device = _pre_kernel_device(kernel, index, files)
            if pre_device is not None:
                pre_source = (repo_root / pre_device).resolve()
                if pre_source not in compile_flags[target]:
                    device_sources[target].append(pre_source)
                    compile_flags[target][pre_source] = list(flags)
        elif stage == "finalize":
            finalize_dtypes[target].add(kernel["expert_weights_dtype"])
    for target in _TARGET_FLAGS:
        if not device_sources[target]:
            continue
        if "fc1" not in stages[target]:
            raise ValueError(
                f"Cake StepFun inventory has no fc1 kernels for target {target}"
            )
        for stage in _TILED_STAGES:
            if stage not in stages[target]:
                continue
            for family, spec in declared[stage].items():
                missing = sorted(
                    tile
                    for tile in spec.get("tiles", ())
                    if (target, stage, family, tile) not in seen
                )
                if missing:
                    raise ValueError(
                        f"Cake StepFun inventory lacks {stage} {family} tiles {missing} "
                        f"for target {target}"
                    )
    manifest = (repo_root / manifest_relative).resolve()
    table_sources = {manifest_relative: manifest.read_text(encoding="utf-8")}
    # The tables of the stages other than fc1 live in the second generated header the inventory
    # names (``stages_manifest``, included at the end of the hand-written ABI header).
    stages_relative = inventory.get("stages_manifest")
    if stages_relative is not None:
        if not isinstance(stages_relative, str) or stages_relative not in files:
            raise ValueError(
                "Cake StepFun inventory stages_manifest must name a generated file listed in files"
            )
        table_sources[stages_relative] = (
            (repo_root / stages_relative).resolve().read_text(encoding="utf-8")
        )
    for stage in CAKE_STEPFUN_STAGES:
        if any(stage in present for present in stages.values()):
            table = _STAGE_TABLE[stage]
            if not any(table in text for text in table_sources.values()):
                raise ValueError(
                    f"Cake StepFun inventory lists {stage} kernels but the generated "
                    f"manifest(s) {', '.join(table_sources)} define no {table} table"
                )
    return CakeStepFunInventory(
        path=inventory_path,
        repo_root=repo_root,
        manifest=manifest,
        stages={t: frozenset(s) for t, s in stages.items()},
        device_sources=device_sources,
        compile_flags=compile_flags,
        routing_inputs={t: frozenset(s) for t, s in routing_inputs.items()},
        finalize_weight_dtypes={t: frozenset(s) for t, s in finalize_dtypes.items()},
        fc2_native_configs=fc2_native_configs,
    )


def cake_stepfun_target(name: str) -> CakeStepFunTarget:
    """Narrow an architecture string to the exact Cake StepFun JIT target.

    The Cake StepFun kernels are exported per exact target; every other
    architecture string (including ``sm_120a`` and the generic ``sm_100``)
    raises instead of being silently mapped onto a neighbouring target.
    """
    if name == "sm_100a":
        return "sm_100a"
    if name == "sm_103a":
        return "sm_103a"
    raise ValueError(
        f"unsupported Cake StepFun target: {name!r} (expected one of "
        + ", ".join(repr(t) for t in _TARGET_FLAGS)
        + ")"
    )


def _require_target(target: CakeStepFunTarget) -> None:
    cake_stepfun_target(target)


def cake_stepfun_stages(target: CakeStepFunTarget) -> frozenset[str]:
    """Return the pipeline stages with an exported Cake kernel for ``target``."""
    _require_target(target)
    return load_cake_stepfun_inventory().stages[target]


def cake_stepfun_missing_stages(target: CakeStepFunTarget) -> tuple[str, ...]:
    """Return the stages (in pipeline order) the inventory does not cover for ``target``."""
    _require_target(target)
    return load_cake_stepfun_inventory().missing_stages(target)


def cake_stepfun_routing_inputs(target: CakeStepFunTarget) -> frozenset[str]:
    """Return the routing input kinds (``scores``, ``topk_ids``) exported for ``target``."""
    _require_target(target)
    return load_cake_stepfun_inventory().routing_inputs[target]


def cake_stepfun_finalize_weight_dtypes(target: CakeStepFunTarget) -> frozenset[str]:
    """Return the expert-weight dtypes (``float32``, ``bfloat16``) the exported finalize
    kernels of ``target`` read."""
    _require_target(target)
    return load_cake_stepfun_inventory().finalize_weight_dtypes[target]


def cake_stepfun_fc2_native_config(
    target: CakeStepFunTarget, kernel_symbol: str
) -> str:
    """Return the trtllm-gen configuration (function name) the exported FC2 kernel
    ``kernel_symbol`` of ``target`` is a port of."""
    _require_target(target)
    configs = load_cake_stepfun_inventory().fc2_native_configs[target]
    if kernel_symbol not in configs:
        raise KeyError(
            f"Cake StepFun inventory has no fc2 kernel {kernel_symbol!r} for {target}"
        )
    return configs[kernel_symbol]


def resolve_cake_stepfun_full_path(
    target: CakeStepFunTarget, full_path: bool | None = None
) -> bool:
    """Decide whether the module for ``target`` runs the full Cake path.

    ``full_path=None`` consults :data:`CAKE_STEPFUN_FULL_PATH_ENV` (``auto`` by
    default: full path exactly when the inventory covers every stage). Requesting
    the full path (``True`` or ``1``) with a stage missing raises ``ValueError``
    naming the missing stages; ``False`` / ``0`` selects the FC1-only module.
    """
    _require_target(target)
    if full_path is None:
        setting = os.environ.get(CAKE_STEPFUN_FULL_PATH_ENV, "auto").strip().lower()
        if setting in ("", "auto"):
            full_path = None
        elif setting in ("1", "true", "on"):
            full_path = True
        elif setting in ("0", "false", "off"):
            full_path = False
        else:
            raise ValueError(
                f"{CAKE_STEPFUN_FULL_PATH_ENV} must be auto, 0 or 1, got {setting!r}"
            )
    missing = cake_stepfun_missing_stages(target)
    if full_path is None:
        return not missing
    if full_path and missing:
        raise ValueError(
            "Cake StepFun full path requested but the generated inventory has no "
            f"{', '.join(missing)} kernel(s) for {target}; export the missing stage(s) "
            f"or select the FC1-only module ({CAKE_STEPFUN_FULL_PATH_ENV}=0)"
        )
    return bool(full_path)


def get_cake_stepfun_fused_moe_uri(
    target: CakeStepFunTarget, full_path: bool = False
) -> str:
    """Return the exact-architecture Cake StepFun fused-MoE JIT module key."""
    _require_target(target)
    return _MODULE_URI[(target, bool(full_path))]


@functools.cache
def _gen_module(target: CakeStepFunTarget, full_path: bool) -> JitSpec:
    uri = get_cake_stepfun_fused_moe_uri(target, full_path)
    csrc_dir = _get_cake_stepfun_csrc_dir()
    inventory = load_cake_stepfun_inventory(csrc_dir)
    device_sources = inventory.device_sources[target]
    device_flags = inventory.compile_flags[target]
    if not device_sources:
        raise ValueError(f"Cake StepFun inventory has no kernels for target {target}")
    required = [
        csrc_dir / _FC1_SOURCE,
        csrc_dir / _FC1_HEADER,
        csrc_dir / _BINDING_SOURCE,
        csrc_dir / "generated" / _MANIFEST,
    ]
    host_sources = [csrc_dir / _FC1_SOURCE, csrc_dir / _BINDING_SOURCE]
    if full_path:
        missing = inventory.missing_stages(target)
        if missing:
            raise ValueError(
                f"Cake StepFun full path for {target} lacks stage(s): {', '.join(missing)}"
            )
        required += [
            csrc_dir / _STAGES_SOURCE,
            csrc_dir / _STAGES_HEADER,
            csrc_dir / _ABI_HEADER,
        ]
        host_sources.append(csrc_dir / _STAGES_SOURCE)
    for source in required:
        if not source.is_file():
            raise FileNotFoundError(f"Cake StepFun source not found: {source}")
    target_flags = [
        *_TARGET_FLAGS[target],
        "-DCAKE_STEPFUN_FC1",
        *(["-DCAKE_STEPFUN_FULL"] if full_path else []),
        f"-DFLASHINFER_CAKE_STEPFUN_TARGET_MINOR={_TARGET_MINOR[target]}",
        # This module defines the trtllm-gen fused-MoE host symbols with stage
        # runners of a different layout than the public module. Hide every host
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
        [*sources, *host_sources, *device_sources],
        extra_cuda_cflags=cflags,
        extra_cuda_cflags_by_source={
            source: [*target_flags, *flags] for source, flags in device_flags.items()
        },
        extra_include_paths=[*include_paths, csrc_dir],
    )
    logger.info(
        f"Generated Cake StepFun fused-MoE {target} JIT spec: {spec.name} "
        f"(stages: {', '.join(s for s in CAKE_STEPFUN_STAGES if s in inventory.stages[target])}; "
        f"full path: {full_path})"
    )
    return spec


def gen_cake_stepfun_fused_moe_module(
    target: CakeStepFunTarget, full_path: bool | None = None
) -> JitSpec:
    """Generate the exact-architecture Cake StepFun fused-MoE module.

    The module is the trtllm-gen fused-MoE host pipeline compiled with
    ``-DCAKE_STEPFUN_FC1``, which replaces the GEMM1 stage by the exported Cake
    StepFun FC1 kernels of ``csrc/fused_moe/cake_stepfun/``. When the generated
    inventory covers every stage of :data:`CAKE_STEPFUN_STAGES` for ``target`` (or
    ``full_path`` requests it, see :func:`resolve_cake_stepfun_full_path`), the
    module is also compiled with ``-DCAKE_STEPFUN_FULL`` and the routing, GEMM2,
    NVFP4 per-token requantization and finalize stages run exported Cake kernels
    as well. Both variants export the TVM-FFI operations of
    ``fused_moe_trtllm_sm100`` under their own module names plus the standalone
    Cake entry points (``cake_stepfun_fc1_families``, ``cake_stepfun_fc1_tiles``,
    ``cake_stepfun_fc1``, ``cake_stepfun_stages``, ``cake_stepfun_full_path`` and,
    on the full path, the per-stage operations).
    """
    return _gen_module(target, resolve_cake_stepfun_full_path(target, full_path))


__all__ = [
    "CAKE_STEPFUN_FULL_PATH_ENV",
    "CAKE_STEPFUN_STAGES",
    "CakeStepFunInventory",
    "CakeStepFunStage",
    "CakeStepFunTarget",
    "cake_stepfun_missing_stages",
    "cake_stepfun_stages",
    "gen_cake_stepfun_fused_moe_module",
    "get_cake_stepfun_fused_moe_uri",
    "load_cake_stepfun_inventory",
    "resolve_cake_stepfun_full_path",
]
