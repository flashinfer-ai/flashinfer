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

from __future__ import annotations

import functools
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal

import torch

from . import env as jit_env
from .core import JitSpec, gen_jit_spec, sm100a_nvcc_flags, sm103a_nvcc_flags


CakeMoeFinalizeArch = Literal["sm_100a", "sm_103a"]
CakeMoeFinalizeDType = Literal["float16", "bfloat16"]
CakeMoeFinalizeOutputProfile = Literal["110", "111"]

_SOURCE_PACKAGE = "cake_moe_finalize_allreduce_fusion"
_MANIFEST_NAME = f"{_SOURCE_PACKAGE}_import_manifest.json"
_SOURCE_PREFIX = ("csrc", _SOURCE_PACKAGE)
_ARCH_BY_CAPABILITY: dict[tuple[int, int], CakeMoeFinalizeArch] = {
    (10, 0): "sm_100a",
    (10, 3): "sm_103a",
}
_NVCC_FLAGS = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
}
_DTYPE_NAME = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
}
_MAX_COMM_SIZE = 2147483647 & ~((1 << 21) - 1)
_CONTRACT = {
    "operator": "trtllm_moe_finalize_allreduce_fusion",
    "architectures": ["sm_100a", "sm_103a"],
    "dtypes": ["float16", "bfloat16"],
    "world_sizes": [2, 4, 8],
    "hidden_dim": 7168,
    "max_lamport_comm_bytes": _MAX_COMM_SIZE,
    "top_k": [4, 8],
    "output_profiles": ["110", "111"],
    "shared_expert": [False, True],
    "pdl": [False, True],
    "workspace_abi": "flashinfer_pointer_table",
    "launch_grid": {"grid_y": 1, "grid_z": 1},
}
_BUILD_CONTRACT = {
    "translation_unit_model": "separate_device_and_binding",
    "architecture_source": "modules[].arch",
    "binary_payloads": False,
    "target_infrastructure": {
        "binding_runtime": "flashinfer_tvm_ffi_utils",
        "headers_owned_by_target": True,
        "required_headers": ["tvm_ffi_utils.h"],
    },
}
_ARG_PLAN = (
    ("buffer", "allreduce_in"),
    ("buffer", "inverse_indices"),
    ("buffer", "expert_scales"),
    ("buffer", "shared_expert_output"),
    ("buffer", "residual"),
    ("buffer", "norm_weight"),
    ("buffer", "residual_out"),
    ("buffer", "norm_out"),
    ("buffer", "quant_out"),
    ("buffer", "scale_out"),
    ("buffer", "workspace_tensor"),
    ("parameter", "world_rank"),
    ("parameter", "tokens"),
    ("parameter", "top_k"),
    ("parameter", "has_shared_expert"),
    ("parameter", "routed_scaling_factor"),
    ("parameter", "epsilon"),
    ("parameter", "weight_bias"),
    ("parameter", "scale_factor"),
    ("grid", "grid_x"),
    ("grid", "grid_y"),
    ("grid", "grid_z"),
)


@dataclass(frozen=True)
class CakeMoeFinalizeLaunchGrid:
    """Fixed non-X launch dimensions required by every generated binding."""

    grid_y: Literal[1] = 1
    grid_z: Literal[1] = 1


_LAUNCH_GRID = CakeMoeFinalizeLaunchGrid()


@dataclass(frozen=True)
class CakeMoeFinalizeModuleSpec:
    """One verified generated module and its exact source closure."""

    arch: CakeMoeFinalizeArch
    dtype: CakeMoeFinalizeDType
    world_size: int
    output_profile: CakeMoeFinalizeOutputProfile
    use_pdl: bool
    name: str
    module_ident: str
    ffi_entry: str
    device_path: Path
    binding_path: Path
    closure_sha256: str
    arg_plan: tuple[tuple[str, str], ...]
    launch_grid: CakeMoeFinalizeLaunchGrid


def _require_manifest(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"invalid Cake MoE finalize import manifest: {message}")


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        _require_manifest(key not in result, f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _compact_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / _SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / _SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake MoE finalize CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def get_cake_moe_finalize_manifest_path() -> Path:
    """Return the verified source package's import-manifest path."""

    return _source_dir() / _MANIFEST_NAME


def _resolve_source_path(source_dir: Path, value: object, label: str) -> Path:
    _require_manifest(isinstance(value, str) and bool(value), f"{label} missing")
    assert isinstance(value, str)
    relative = PurePosixPath(value)
    _require_manifest(
        not relative.is_absolute()
        and ".." not in relative.parts
        and "." not in relative.parts
        and relative.parts[:2] == _SOURCE_PREFIX
        and len(relative.parts) == 4
        and relative.parts[2] in ("sm_100a", "sm_103a"),
        f"{label} must name one architecture-owned source file",
    )
    path = source_dir.joinpath(*relative.parts[2:])
    _require_manifest(path.suffix == ".cu", f"{label} must name a .cu file")
    _require_manifest(
        path.name.startswith("cake_trtllm_moe_finalize_"),
        f"{label} must use the public Cake finalize filename prefix",
    )
    _require_manifest(path.is_file() and not path.is_symlink(), f"{label} missing")
    return path


def _verify_file_record(
    source_dir: Path, record: object, label: str
) -> tuple[str, Path]:
    _require_manifest(isinstance(record, dict), f"{label} must be an object")
    assert isinstance(record, dict)
    _require_manifest(
        set(record)
        in ({"path", "sha256", "bytes"}, {"path", "kind", "sha256", "bytes"}),
        f"{label} keys mismatch",
    )
    path = _resolve_source_path(source_dir, record.get("path"), f"{label}.path")
    digest = record.get("sha256")
    size = record.get("bytes")
    _require_manifest(
        isinstance(digest, str)
        and len(digest) == 64
        and all(character in "0123456789abcdef" for character in digest),
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    _require_manifest(
        isinstance(size, int) and not isinstance(size, bool) and size >= 0,
        f"{label}.bytes must be a nonnegative integer",
    )
    payload = path.read_bytes()
    _require_manifest(len(payload) == size, f"{label}.bytes mismatch")
    _require_manifest(
        hashlib.sha256(payload).hexdigest() == digest,
        f"{label}.sha256 mismatch",
    )
    return str(record["path"]), path


def _expected_routes() -> set[tuple[str, str, int, str, bool]]:
    return {
        (arch, dtype, world_size, output_profile, use_pdl)
        for arch in ("sm_100a", "sm_103a")
        for dtype in ("float16", "bfloat16")
        for world_size in (2, 4, 8)
        for output_profile in ("110", "111")
        for use_pdl in (False, True)
    }


@functools.cache
def get_cake_moe_finalize_module_specs() -> tuple[CakeMoeFinalizeModuleSpec, ...]:
    """Load and verify the complete 48-module source-only export."""

    source_dir = _source_dir()
    manifest_path = get_cake_moe_finalize_manifest_path()
    _require_manifest(
        manifest_path.is_file() and not manifest_path.is_symlink(),
        f"missing {_MANIFEST_NAME}",
    )
    payload = json.loads(
        manifest_path.read_text(encoding="utf-8"),
        object_pairs_hook=_reject_duplicate_keys,
    )
    _require_manifest(isinstance(payload, dict), "root must be an object")
    assert isinstance(payload, dict)
    _require_manifest(
        set(payload)
        == {
            "schema",
            "producer",
            "producer_revision",
            "artifact_kind",
            "build_contract",
            "library",
            "name",
            "contract",
            "modules",
            "files",
        },
        "top-level keys mismatch",
    )
    _require_manifest(payload.get("schema") == "cake.library_export.v1", "schema")
    _require_manifest(payload.get("producer") == "cake", "producer")
    _require_manifest(
        payload.get("producer_revision") is None,
        "producer_revision must be omitted from the public source package",
    )
    _require_manifest(payload.get("artifact_kind") == "source_only", "artifact_kind")
    _require_manifest(
        payload.get("build_contract") == _BUILD_CONTRACT, "build_contract"
    )
    _require_manifest(payload.get("library") == "flashinfer", "library")
    _require_manifest(payload.get("name") == _SOURCE_PACKAGE, "name")
    _require_manifest(payload.get("contract") == _CONTRACT, "contract")

    files = payload.get("files")
    _require_manifest(isinstance(files, list), "files must be a list")
    inventory: dict[str, Path] = {}
    for index, record in enumerate(files):
        path_value, path = _verify_file_record(source_dir, record, f"files[{index}]")
        _require_manifest(path_value not in inventory, f"duplicate file {path_value}")
        assert isinstance(record, dict)
        expected_kind = (
            "device_source" if path.name.endswith("_device.cu") else "tvm_ffi_binding"
        )
        _require_manifest(
            record.get("kind") == expected_kind,
            f"files[{index}].kind mismatch",
        )
        inventory[path_value] = path

    actual_files = {
        path.relative_to(source_dir).as_posix()
        for path in source_dir.rglob("*")
        if path.is_file()
    }
    expected_files = {
        PurePosixPath(path).relative_to(*_SOURCE_PREFIX).as_posix()
        for path in inventory
    } | {_MANIFEST_NAME}
    _require_manifest(
        actual_files == expected_files,
        "source tree mismatch: "
        f"missing={sorted(expected_files - actual_files)}, "
        f"extra={sorted(actual_files - expected_files)}",
    )

    modules = payload.get("modules")
    _require_manifest(isinstance(modules, list), "modules must be a list")
    observed_routes: set[tuple[str, str, int, str, bool]] = set()
    observed_names: set[tuple[str, str]] = set()
    specs: list[CakeMoeFinalizeModuleSpec] = []
    module_keys = {
        "arch",
        "name",
        "role",
        "translation_units",
        "kernel_symbol",
        "module_ident",
        "ffi_entry",
        "binding_infrastructure",
        "arg_plan",
        "compile_flags",
        "tma_abi",
        "launch",
        "route",
        "closure",
        "arg_plan_sha256",
        "closure_sha256",
    }
    for index, item in enumerate(modules):
        label = f"modules[{index}]"
        _require_manifest(isinstance(item, dict), f"{label} must be an object")
        assert isinstance(item, dict)
        _require_manifest(set(item) == module_keys, f"{label} keys mismatch")
        arch = item.get("arch")
        route = item.get("route")
        _require_manifest(arch in ("sm_100a", "sm_103a"), f"{label}.arch")
        _require_manifest(
            isinstance(route, dict)
            and set(route)
            == {"dtype", "world_size", "hidden_dim", "output_profile", "use_pdl"},
            f"{label}.route keys mismatch",
        )
        assert isinstance(arch, str) and isinstance(route, dict)
        dtype = route.get("dtype")
        world_size = route.get("world_size")
        output_profile = route.get("output_profile")
        use_pdl = route.get("use_pdl")
        _require_manifest(dtype in ("float16", "bfloat16"), f"{label}.route.dtype")
        _require_manifest(world_size in (2, 4, 8), f"{label}.route.world_size")
        _require_manifest(route.get("hidden_dim") == 7168, f"{label}.route.hidden_dim")
        _require_manifest(
            output_profile in ("110", "111"), f"{label}.route.output_profile"
        )
        _require_manifest(type(use_pdl) is bool, f"{label}.route.use_pdl")
        assert isinstance(dtype, str) and isinstance(world_size, int)
        assert isinstance(output_profile, str) and isinstance(use_pdl, bool)
        route_key = (arch, dtype, world_size, output_profile, use_pdl)
        _require_manifest(
            route_key not in observed_routes, f"duplicate route {route_key}"
        )
        observed_routes.add(route_key)

        kernel_stem = (
            f"cake_trtllm_moe_finalize_{dtype}_ws{world_size}_o{output_profile}"
        )
        expected_name = f"{kernel_stem}_pdl{int(use_pdl)}"
        name = item.get("name")
        _require_manifest(name == expected_name, f"{label}.name")
        assert isinstance(name, str)
        name_key = (arch, name)
        _require_manifest(
            name_key not in observed_names,
            f"duplicate module name {name} for architecture {arch}",
        )
        observed_names.add(name_key)
        _require_manifest(
            item.get("role") == f"finalize_pdl{int(use_pdl)}", f"{label}.role"
        )
        _require_manifest(
            item.get("kernel_symbol") == f"kernel_{kernel_stem}",
            f"{label}.kernel_symbol",
        )
        _require_manifest(item.get("module_ident") == name, f"{label}.module_ident")
        _require_manifest(item.get("ffi_entry") == "run", f"{label}.ffi_entry")
        _require_manifest(
            item.get("binding_infrastructure")
            == {
                "runtime": "flashinfer_tvm_ffi_utils",
                "target_owned_headers": ["tvm_ffi_utils.h"],
            },
            f"{label}.binding_infrastructure",
        )
        arg_plan_raw = item.get("arg_plan")
        _require_manifest(
            isinstance(arg_plan_raw, list)
            and all(
                isinstance(entry, list)
                and len(entry) == 2
                and all(isinstance(value, str) for value in entry)
                for entry in arg_plan_raw
            ),
            f"{label}.arg_plan",
        )
        assert isinstance(arg_plan_raw, list)
        arg_plan = tuple(tuple(entry) for entry in arg_plan_raw)
        _require_manifest(arg_plan == _ARG_PLAN, f"{label}.arg_plan contract mismatch")
        _require_manifest(
            item.get("arg_plan_sha256")
            == hashlib.sha256(_compact_json(arg_plan_raw)).hexdigest(),
            f"{label}.arg_plan_sha256 mismatch",
        )
        _require_manifest(
            item.get("compile_flags") == ["--use_fast_math"],
            f"{label}.compile_flags",
        )
        _require_manifest(item.get("tma_abi") == "pointer", f"{label}.tma_abi")
        launch = item.get("launch")
        _require_manifest(
            isinstance(launch, dict)
            and launch.get("block") == [224, 1, 1]
            and launch.get("dynamic_smem_bytes") == 256
            and launch.get("cluster") == [4, 1, 1]
            and launch.get("cooperative") is False
            and launch.get("use_pdl") is use_pdl
            and launch.get("cluster_scheduling_policy") == "default"
            and set(launch)
            == {
                "block",
                "dynamic_smem_bytes",
                "cluster",
                "cooperative",
                "use_pdl",
                "cluster_scheduling_policy",
            },
            f"{label}.launch",
        )

        units = item.get("translation_units")
        _require_manifest(
            isinstance(units, dict)
            and set(units) == {"device", "binding", "compile_separately"}
            and units.get("compile_separately") is True,
            f"{label}.translation_units",
        )
        assert isinstance(units, dict)
        device_value = units.get("device")
        binding_value = units.get("binding")
        expected_device = f"csrc/{_SOURCE_PACKAGE}/{arch}/{kernel_stem}_device.cu"
        expected_binding = f"csrc/{_SOURCE_PACKAGE}/{arch}/{expected_name}_binding.cu"
        _require_manifest(
            device_value == expected_device,
            f"{label}.translation_units.device",
        )
        _require_manifest(
            binding_value == expected_binding,
            f"{label}.translation_units.binding",
        )
        _require_manifest(
            device_value in inventory, f"{label} device absent from inventory"
        )
        _require_manifest(
            binding_value in inventory, f"{label} binding absent from inventory"
        )

        closure = item.get("closure")
        _require_manifest(
            isinstance(closure, list) and len(closure) == 2,
            f"{label}.closure must contain device and binding",
        )
        assert isinstance(closure, list)
        closure_paths = []
        for closure_index, record in enumerate(closure):
            path_value, _ = _verify_file_record(
                source_dir, record, f"{label}.closure[{closure_index}]"
            )
            _require_manifest(
                path_value in inventory,
                f"{label}.closure[{closure_index}] absent from inventory",
            )
            assert isinstance(record, dict)
            inventory_record = next(
                value for value in files if value.get("path") == path_value
            )
            _require_manifest(
                record
                == {key: inventory_record[key] for key in ("path", "sha256", "bytes")},
                f"{label}.closure[{closure_index}] inventory mismatch",
            )
            closure_paths.append(path_value)
        _require_manifest(
            closure_paths == [device_value, binding_value],
            f"{label}.closure order mismatch",
        )
        identity = {
            key: value
            for key, value in item.items()
            if key not in {"arg_plan_sha256", "closure_sha256"}
        }
        closure_sha256 = item.get("closure_sha256")
        _require_manifest(
            closure_sha256 == hashlib.sha256(_compact_json(identity)).hexdigest(),
            f"{label}.closure_sha256 mismatch",
        )
        assert isinstance(closure_sha256, str)
        specs.append(
            CakeMoeFinalizeModuleSpec(
                arch=arch,  # type: ignore[arg-type]
                dtype=dtype,  # type: ignore[arg-type]
                world_size=world_size,
                output_profile=output_profile,  # type: ignore[arg-type]
                use_pdl=use_pdl,
                name=name,
                module_ident=name,
                ffi_entry="run",
                device_path=inventory[device_value],
                binding_path=inventory[binding_value],
                closure_sha256=closure_sha256,
                arg_plan=arg_plan,  # type: ignore[arg-type]
                launch_grid=_LAUNCH_GRID,
            )
        )

    expected_routes = _expected_routes()
    _require_manifest(
        observed_routes == expected_routes,
        "route set mismatch: "
        f"missing={sorted(expected_routes - observed_routes)}, "
        f"extra={sorted(observed_routes - expected_routes)}",
    )
    _require_manifest(
        len(inventory) == 72,
        "file inventory must contain 24 device and 48 binding sources",
    )
    specs.sort(
        key=lambda spec: (
            spec.arch,
            spec.dtype,
            spec.world_size,
            spec.output_profile,
            spec.use_pdl,
        )
    )
    return tuple(specs)


def target_arch(device_index: int) -> CakeMoeFinalizeArch:
    """Return the exact generated architecture for one CUDA device."""

    capability = torch.cuda.get_device_capability(device_index)
    arch = _ARCH_BY_CAPABILITY.get(capability)
    if arch is None:
        raise ValueError(
            "Cake MoE finalize requires SM100 or SM103, got "
            f"SM{capability[0]}{capability[1]}"
        )
    return arch


def get_cake_moe_finalize_module_spec(
    *,
    arch: CakeMoeFinalizeArch,
    dtype: CakeMoeFinalizeDType,
    world_size: int,
    output_profile: CakeMoeFinalizeOutputProfile,
    use_pdl: bool,
) -> CakeMoeFinalizeModuleSpec:
    """Select one exact route from the verified export."""

    route = (arch, dtype, world_size, output_profile, use_pdl)
    for spec in get_cake_moe_finalize_module_specs():
        if (
            spec.arch,
            spec.dtype,
            spec.world_size,
            spec.output_profile,
            spec.use_pdl,
        ) == route:
            return spec
    raise ValueError(f"unsupported Cake MoE finalize route: {route}")


@functools.cache
def gen_cake_moe_finalize_module(
    arch: CakeMoeFinalizeArch,
    dtype: CakeMoeFinalizeDType,
    world_size: int,
    output_profile: CakeMoeFinalizeOutputProfile,
    use_pdl: bool,
) -> JitSpec:
    """Create a JIT spec for one verified generated module."""

    spec = get_cake_moe_finalize_module_spec(
        arch=arch,
        dtype=dtype,
        world_size=world_size,
        output_profile=output_profile,
        use_pdl=use_pdl,
    )
    return gen_jit_spec(
        name=f"{spec.module_ident}_{spec.closure_sha256[:16]}",
        sources=[spec.device_path, spec.binding_path],
        extra_cuda_cflags=[*_NVCC_FLAGS[arch], "--use_fast_math"],
        extra_include_paths=[_source_dir().parent],
    )


@functools.cache
def load_cake_moe_finalize_module(
    arch: CakeMoeFinalizeArch,
    dtype: CakeMoeFinalizeDType,
    world_size: int,
    output_profile: CakeMoeFinalizeOutputProfile,
    use_pdl: bool,
):
    """Build and load one verified generated module."""

    return gen_cake_moe_finalize_module(
        arch, dtype, world_size, output_profile, use_pdl
    ).build_and_load()


def get_cake_moe_finalize_library_path(
    arch: CakeMoeFinalizeArch,
    dtype: CakeMoeFinalizeDType,
    world_size: int,
    output_profile: CakeMoeFinalizeOutputProfile,
    use_pdl: bool,
) -> Path:
    """Return the route-owned TVM-FFI shared-library cache path."""

    return gen_cake_moe_finalize_module(
        arch, dtype, world_size, output_profile, use_pdl
    ).get_library_path()


def _check_cuda_tensor(
    tensor: object,
    name: str,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
    shape: tuple[int, ...] | None = None,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device.type != "cuda":
        raise ValueError(f"{name} must be a CUDA tensor")
    if device is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if shape is not None and tuple(tensor.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")
    return tensor


def _fp4_scale_storage_elements(tokens: int, hidden_dim: int) -> int:
    padded_rows = ((tokens + 127) // 128) * 128
    scale_columns = hidden_dim // 16
    padded_columns = ((scale_columns + 3) // 4) * 4
    return padded_rows * padded_columns


def _lamport_comm_size_bytes(
    tokens: int,
    hidden_dim: int,
    element_size: int,
    world_size: int,
) -> int:
    return tokens * hidden_dim * element_size * world_size


def validate_cake_moe_finalize_args(
    *,
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    quant_out: torch.Tensor | None,
    scale_out: torch.Tensor | None,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: torch.Tensor | None,
    expert_scale_factor: torch.Tensor | None,
    routed_scaling_factor: float | None,
    weight_bias: float | None,
) -> tuple[CakeMoeFinalizeArch, CakeMoeFinalizeDType, CakeMoeFinalizeOutputProfile]:
    """Validate the frozen source route without synchronizing the device."""

    if world_size not in (2, 4, 8):
        raise ValueError(
            f"Cake MoE finalize world_size must be 2, 4, or 8, got {world_size}"
        )
    if type(world_rank) is not int or not 0 <= world_rank < world_size:
        raise ValueError(f"world_rank must be in [0, {world_size}), got {world_rank}")
    if type(launch_with_pdl) is not bool:
        raise TypeError("launch_with_pdl must be bool")
    if not isinstance(eps, (int, float)) or not float(eps) > 0:
        raise ValueError(f"eps must be positive, got {eps}")
    if routed_scaling_factor is not None and not isinstance(
        routed_scaling_factor, (int, float)
    ):
        raise TypeError("routed_scaling_factor must be a number or None")
    if weight_bias is not None and float(weight_bias) not in (0.0, 1.0):
        raise ValueError("Cake MoE finalize weight_bias must be None, 0.0, or 1.0")

    allreduce_in = _check_cuda_tensor(allreduce_in, "allreduce_in")
    if (
        allreduce_in.ndim != 2
        or allreduce_in.shape[1] != 7168
        or allreduce_in.shape[0] <= 0
    ):
        raise ValueError(
            "allreduce_in must have shape [num_permuted_rows, 7168] with at least one row"
        )
    if allreduce_in.dtype not in _DTYPE_NAME:
        raise ValueError("Cake MoE finalize supports float16 and bfloat16 inputs")
    dtype = allreduce_in.dtype
    device = allreduce_in.device
    residual_in = _check_cuda_tensor(
        residual_in, "residual_in", device=device, dtype=dtype
    )
    if residual_in.ndim != 2 or residual_in.shape[1] != 7168:
        raise ValueError("residual_in must have shape [token_num, 7168]")
    tokens = residual_in.shape[0]
    if tokens < 1:
        raise ValueError(f"token_num must be positive, got {tokens}")
    required_lamport_comm_size = _lamport_comm_size_bytes(
        tokens,
        7168,
        residual_in.element_size(),
        world_size,
    )
    if required_lamport_comm_size > _MAX_COMM_SIZE:
        raise ValueError(
            f"required_lamport_comm_size {required_lamport_comm_size} is greater "
            f"than MAX_COMM_SIZE {_MAX_COMM_SIZE}"
        )
    _check_cuda_tensor(
        norm_weight, "norm_weight", device=device, dtype=dtype, shape=(7168,)
    )
    indices = _check_cuda_tensor(
        expanded_idx_to_permuted_idx,
        "expanded_idx_to_permuted_idx",
        device=device,
        dtype=torch.int32,
    )
    if (
        indices.ndim != 2
        or indices.shape[0] != tokens
        or indices.shape[1] not in (4, 8)
    ):
        raise ValueError(
            "expanded_idx_to_permuted_idx must have shape [token_num, top_k] with top_k 4 or 8"
        )
    if expert_scale_factor is None:
        raise ValueError('expert_scale_factor is required when backend="cake"')
    _check_cuda_tensor(
        expert_scale_factor,
        "expert_scale_factor",
        device=device,
        dtype=dtype,
        shape=tuple(indices.shape),
    )
    if shared_expert_output is not None:
        _check_cuda_tensor(
            shared_expert_output,
            "shared_expert_output",
            device=device,
            dtype=dtype,
            shape=tuple(residual_in.shape),
        )
    if norm_out is None or residual_out is None:
        raise ValueError('norm_out and residual_out are required when backend="cake"')
    _check_cuda_tensor(
        norm_out, "norm_out", device=device, dtype=dtype, shape=tuple(residual_in.shape)
    )
    _check_cuda_tensor(
        residual_out,
        "residual_out",
        device=device,
        dtype=dtype,
        shape=tuple(residual_in.shape),
    )
    if (quant_out is None) != (scale_out is None):
        raise ValueError("quant_out and scale_out must be provided together")
    output_profile: CakeMoeFinalizeOutputProfile = "110"
    if quant_out is not None and scale_out is not None:
        output_profile = "111"
        quant_out = _check_cuda_tensor(
            quant_out, "quant_out", device=device, dtype=dtype
        )
        scale_out = _check_cuda_tensor(
            scale_out, "scale_out", device=device, dtype=dtype
        )
        required_quant_elements = tokens * 7168 // 4
        if quant_out.numel() < required_quant_elements:
            raise ValueError(
                f"quant_out requires at least {required_quant_elements} elements"
            )
        required_scale_elements = _fp4_scale_storage_elements(tokens, 7168)
        if scale_out.numel() < required_scale_elements:
            raise ValueError(
                "scale_out is smaller than the padded SWIZZLED_128x4 ABI "
                f"({required_scale_elements} elements)"
            )
    workspace_ptrs = _check_cuda_tensor(
        workspace_ptrs, "workspace_ptrs", device=device, dtype=torch.int64
    )
    expected_workspace_ptrs = 3 * world_size + 1
    if workspace_ptrs.ndim != 1 or workspace_ptrs.numel() != expected_workspace_ptrs:
        raise ValueError(
            f"workspace_ptrs must contain exactly {expected_workspace_ptrs} pointers"
        )
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    arch = target_arch(device_index)
    return arch, _DTYPE_NAME[dtype], output_profile  # type: ignore[return-value]


@functools.cache
def _dummy(device_index: int, dtype: torch.dtype) -> torch.Tensor:
    return torch.empty(1, dtype=dtype, device=torch.device("cuda", device_index))


def run_cake_moe_finalize(
    *,
    backend: Literal["cake"] = "cake",
    allreduce_in: torch.Tensor,
    residual_in: torch.Tensor,
    norm_weight: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    norm_out: torch.Tensor | None,
    residual_out: torch.Tensor | None,
    quant_out: torch.Tensor | None,
    scale_out: torch.Tensor | None,
    workspace_ptrs: torch.Tensor,
    launch_with_pdl: bool,
    world_rank: int,
    world_size: int,
    eps: float,
    shared_expert_output: torch.Tensor | None,
    expert_scale_factor: torch.Tensor | None,
    routed_scaling_factor: float | None,
    weight_bias: float | None,
) -> None:
    """Validate, select, and launch one Cake finalize route."""

    if backend != "cake":
        raise ValueError(f"backend must be 'cake', got {backend!r}")

    arch, dtype_name, output_profile = validate_cake_moe_finalize_args(
        allreduce_in=allreduce_in,
        residual_in=residual_in,
        norm_weight=norm_weight,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        norm_out=norm_out,
        residual_out=residual_out,
        quant_out=quant_out,
        scale_out=scale_out,
        workspace_ptrs=workspace_ptrs,
        launch_with_pdl=launch_with_pdl,
        world_rank=world_rank,
        world_size=world_size,
        eps=eps,
        shared_expert_output=shared_expert_output,
        expert_scale_factor=expert_scale_factor,
        routed_scaling_factor=routed_scaling_factor,
        weight_bias=weight_bias,
    )
    assert norm_out is not None and residual_out is not None
    assert expert_scale_factor is not None
    device_index = allreduce_in.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    dummy = _dummy(device_index, allreduce_in.dtype)
    tokens, top_k = expanded_idx_to_permuted_idx.shape
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    grid_x = min(sm_count, tokens * 4) // 4 * 4
    if grid_x <= 0:
        raise ValueError("Cake MoE finalize requires one complete four-CTA cluster")
    spec = get_cake_moe_finalize_module_spec(
        arch=arch,
        dtype=dtype_name,
        world_size=world_size,
        output_profile=output_profile,
        use_pdl=launch_with_pdl,
    )
    values: dict[str, object] = {
        "allreduce_in": allreduce_in,
        "inverse_indices": expanded_idx_to_permuted_idx,
        "expert_scales": expert_scale_factor,
        "shared_expert_output": (
            shared_expert_output if shared_expert_output is not None else dummy
        ),
        "residual": residual_in,
        "norm_weight": norm_weight,
        "residual_out": residual_out,
        "norm_out": norm_out,
        "quant_out": quant_out if quant_out is not None else dummy,
        "scale_out": scale_out if scale_out is not None else dummy,
        "workspace_tensor": workspace_ptrs,
        "world_rank": world_rank,
        "tokens": tokens,
        "top_k": top_k,
        "has_shared_expert": int(shared_expert_output is not None),
        "routed_scaling_factor": float(
            1.0 if routed_scaling_factor is None else routed_scaling_factor
        ),
        "epsilon": float(eps),
        "weight_bias": float(0.0 if weight_bias is None else weight_bias),
        "scale_factor": 1.0,
        "grid_x": grid_x,
        "grid_y": spec.launch_grid.grid_y,
        "grid_z": spec.launch_grid.grid_z,
    }
    module = load_cake_moe_finalize_module(
        arch, dtype_name, world_size, output_profile, launch_with_pdl
    )
    getattr(module, spec.ffi_entry)(*(values[name] for _, name in spec.arg_plan))


__all__ = [
    "CakeMoeFinalizeLaunchGrid",
    "CakeMoeFinalizeModuleSpec",
    "gen_cake_moe_finalize_module",
    "get_cake_moe_finalize_library_path",
    "get_cake_moe_finalize_module_spec",
    "get_cake_moe_finalize_module_specs",
    "load_cake_moe_finalize_module",
    "run_cake_moe_finalize",
    "target_arch",
    "validate_cake_moe_finalize_args",
]
