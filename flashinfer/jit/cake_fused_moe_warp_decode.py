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
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from . import env as jit_env
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)

CakeWarpDecodeTarget = Literal["sm100a", "sm103a"]

_TARGET_FLAGS: dict[CakeWarpDecodeTarget, list[str]] = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_TARGET_ARCH: dict[CakeWarpDecodeTarget, str] = {
    "sm100a": "sm_100a",
    "sm103a": "sm_103a",
}
_TARGET_MINOR: dict[CakeWarpDecodeTarget, int] = {"sm100a": 0, "sm103a": 3}
_MODULE_URI: dict[CakeWarpDecodeTarget, str] = {
    "sm100a": "cake_fused_moe_warp_decode_sm100a",
    "sm103a": "cake_fused_moe_warp_decode_sm103a",
}
_LEGACY_SM103A_GENERATED_SOURCE = "cake_adaptive_warp_decode_kernels.cu"
_EXPORTED_MANIFEST = "cake_warp_decode_inventory.json"
_BINDING_SOURCE = "cake_warp_decode_binding.cu"
_GENERATED_MANIFEST = "cake_warp_decode_generated_manifest.cuh"
_CONTRACT_HEADER = "cake_warp_decode_contract.cuh"


def _get_cake_fused_moe_warp_decode_csrc_dir() -> Path:
    """Locate Cake warp-decode sources in installed and source checkouts."""

    checkout = (
        Path(__file__).resolve().parents[2] / "csrc" / "fused_moe" / "warp_decode"
    )
    if checkout.exists():
        return checkout

    installed = jit_env.FLASHINFER_CSRC_DIR / "fused_moe" / "warp_decode"
    if installed.exists():
        return installed

    raise FileNotFoundError(
        "Cake warp-decode CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_include_dir() -> Path:
    """Locate FlashInfer headers in installed and source checkouts."""

    if jit_env.FLASHINFER_INCLUDE_DIR.exists():
        return jit_env.FLASHINFER_INCLUDE_DIR

    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "FlashInfer headers were not found. Checked:\n"
        f"  - {jit_env.FLASHINFER_INCLUDE_DIR}\n"
        f"  - {checkout}"
    )


def _require_dict(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(
            f"Cake warp-decode export manifest {context} must be an object"
        )
    return value


def _resolve_export_path(csrc_dir: Path, raw_path: Any, context: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path or "\\" in raw_path:
        raise ValueError(
            f"Cake warp-decode export manifest {context} must be a non-empty POSIX path"
        )
    parts = raw_path.split("/")
    posix_path = PurePosixPath(raw_path)
    if posix_path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(
            f"Cake warp-decode export manifest {context} is not a safe relative path: "
            f"{raw_path!r}"
        )

    repo_root = csrc_dir.parents[2].resolve()
    resolved = (repo_root / Path(*posix_path.parts)).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as error:
        raise ValueError(
            f"Cake warp-decode export manifest {context} escapes the package root: "
            f"{raw_path!r}"
        ) from error
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Cake warp-decode export manifest {context} source not found: {resolved}"
        )
    return resolved


def _load_exported_device_sources(
    csrc_dir: Path, target: CakeWarpDecodeTarget
) -> tuple[list[Path], bool, dict[Path, list[str]]]:
    """Validate the target-owned inventory and resolve exact-architecture devices."""

    generated_dir = csrc_dir / "generated"
    manifest_path = generated_dir / _EXPORTED_MANIFEST
    if not manifest_path.is_file():
        if target == "sm103a":
            legacy = generated_dir / _LEGACY_SM103A_GENERATED_SOURCE
            if legacy.is_file():
                return [legacy], False, {}
        raise FileNotFoundError(
            f"Cake warp-decode {target} generated inventory was not found at "
            f"{manifest_path}; install the generated warp-decode kernel inventory"
        )
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cake warp-decode inventory could not be read: {manifest_path}"
        ) from error
    manifest = _require_dict(payload, "root")
    if manifest.get("schema") != "flashinfer.warp_decode.inventory.v1":
        raise ValueError(
            "Cake warp-decode inventory schema must be flashinfer.warp_decode.inventory.v1"
        )
    modules = manifest.get("modules")
    sequences = manifest.get("sequences")
    routes = manifest.get("routes")
    files = _require_dict(manifest.get("files"), "files")
    if not all(
        isinstance(rows, list) and rows for rows in (modules, sequences, routes)
    ):
        raise ValueError(
            "Cake warp-decode inventory modules, sequences, and routes must be non-empty arrays"
        )
    program = {
        "modules": modules,
        "sequences": sequences,
        "routes": routes,
        "files": files,
    }
    program_bytes = (
        json.dumps(program, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n"
    ).encode("utf-8")
    if hashlib.sha256(program_bytes).hexdigest() != manifest.get("program_hash"):
        raise ValueError("Cake warp-decode inventory program hash mismatch")

    file_paths: dict[str, Path] = {}
    for raw_path, digest in files.items():
        source = _resolve_export_path(csrc_dir, raw_path, f"files[{raw_path!r}]")
        if (
            not isinstance(digest, str)
            or hashlib.sha256(source.read_bytes()).hexdigest() != digest
        ):
            raise ValueError(
                f"Cake warp-decode inventory file hash mismatch: {raw_path}"
            )
        file_paths[raw_path] = source
    used_files: set[str] = set()

    def inventory_file(raw_path: Any, context: str) -> Path:
        if not isinstance(raw_path, str) or raw_path not in file_paths:
            raise ValueError(f"Cake warp-decode inventory {context} is not in files")
        used_files.add(raw_path)
        return file_paths[raw_path]

    def identity(row: dict[str, Any], context: str) -> tuple[str, str, str]:
        arch, name, role = (row.get(key) for key in ("arch", "name", "role"))
        if arch not in _TARGET_ARCH.values() or not all(
            isinstance(value, str) and value for value in (name, role)
        ):
            raise ValueError(
                f"Cake warp-decode inventory {context} identity is invalid"
            )
        return arch, name, role

    module_sources: dict[tuple[str, str, str], Path] = {}
    module_devices: dict[tuple[str, str, str], str] = {}
    module_symbols: dict[tuple[str, str, str], str] = {}
    device_compile_flags: dict[Path, list[str]] = {}
    for index, raw_module in enumerate(modules):
        context = f"modules[{index}]"
        module = _require_dict(raw_module, context)
        key = identity(module, context)
        if key in module_sources:
            raise ValueError(f"Cake warp-decode inventory duplicates module {key!r}")
        if not all(
            isinstance(module.get(field), str) and module[field]
            for field in ("kernel_symbol", "ffi_entry")
        ):
            raise ValueError(f"Cake warp-decode inventory {context} symbol is invalid")
        device = inventory_file(module.get("device"), f"{context}.device")
        if device.parent.name != key[0]:
            raise ValueError(
                f"Cake warp-decode inventory {context} device architecture differs"
            )
        inventory_file(module.get("binding"), f"{context}.binding")
        compile_flags = module.get("compile_flags")
        if not isinstance(compile_flags, list) or not all(
            isinstance(flag, str) and flag for flag in compile_flags
        ):
            raise ValueError(
                f"Cake warp-decode inventory {context}.compile_flags is invalid"
            )
        if (
            device in device_compile_flags
            and device_compile_flags[device] != compile_flags
        ):
            raise ValueError(
                f"Cake warp-decode inventory has conflicting compile flags for {device}"
            )
        device_compile_flags[device] = compile_flags
        module_sources[key] = device
        module_devices[key] = module["device"]
        module_symbols[key] = module["kernel_symbol"]

    sequence_modules: dict[tuple[str, str, str], tuple[tuple[str, str, str], ...]] = {}
    sequence_module_union: set[tuple[str, str, str]] = set()
    for index, raw_sequence in enumerate(sequences):
        context = f"sequences[{index}]"
        sequence = _require_dict(raw_sequence, context)
        key = identity(sequence, context)
        if key in sequence_modules:
            raise ValueError(f"Cake warp-decode inventory duplicates sequence {key!r}")
        members = sequence.get("modules")
        if not isinstance(members, list) or not members:
            raise ValueError(
                f"Cake warp-decode inventory {context}.modules must be non-empty"
            )
        module_keys = []
        for member_index, raw_member in enumerate(members):
            member = _require_dict(raw_member, f"{context}.modules[{member_index}]")
            member_key = (key[0], member.get("name"), member.get("role"))
            if member_key not in module_sources:
                raise ValueError(
                    f"Cake warp-decode inventory {context} references missing module {member_key!r}"
                )
            module_keys.append(member_key)
        expected_devices = list(
            dict.fromkeys(module_devices[member] for member in module_keys)
        )
        if sequence.get("devices") != expected_devices:
            raise ValueError(
                f"Cake warp-decode inventory {context} device and module inventories differ"
            )
        if not isinstance(sequence.get("ffi_entry"), str) or not sequence["ffi_entry"]:
            raise ValueError(
                f"Cake warp-decode inventory {context}.ffi_entry is invalid"
            )
        inventory_file(sequence.get("binding"), f"{context}.binding")
        sequence_modules[key] = tuple(module_keys)
        sequence_module_union.update(module_keys)
    if sequence_module_union != set(module_sources):
        raise ValueError(
            "Cake warp-decode inventory module and sequence closure differs"
        )

    generated_header = generated_dir / _GENERATED_MANIFEST
    repo_root = csrc_dir.parents[2].resolve()
    for source in (
        generated_header,
        csrc_dir / _BINDING_SOURCE,
        csrc_dir / _CONTRACT_HEADER,
    ):
        inventory_file(source.resolve().relative_to(repo_root).as_posix(), source.name)
    if used_files != set(files):
        raise ValueError("Cake warp-decode inventory files exceed its source closure")
    header_text = generated_header.read_text(encoding="utf-8")
    for key, symbol in module_symbols.items():
        if symbol not in header_text:
            raise ValueError(
                f"Cake warp-decode generated C++ manifest is missing {key[0]} symbol {symbol}"
            )

    target_arch = _TARGET_ARCH[target]
    route_module_union: set[tuple[str, str, str]] = set()
    route_sequences: set[tuple[str, str, str]] = set()
    seen_routes: set[tuple[str, str]] = set()
    silu_tokens: dict[str, set[int]] = {
        "fc1_silu_static": set(),
        "fc1_silu_persistent": set(),
    }
    for index, raw_route in enumerate(routes):
        context = f"routes[{index}]"
        route = _require_dict(raw_route, context)
        arch = route.get("arch")
        args = _require_dict(route.get("args"), f"{context}.args")
        route_key = (
            arch,
            json.dumps(args, sort_keys=True, separators=(",", ":"), allow_nan=False),
        )
        if route_key in seen_routes:
            raise ValueError(
                f"Cake warp-decode inventory duplicates route {route_key!r}"
            )
        seen_routes.add(route_key)
        sequence = _require_dict(route.get("sequence"), f"{context}.sequence")
        sequence_key = (arch, sequence.get("name"), sequence.get("role"))
        if sequence_key not in sequence_modules:
            raise ValueError(
                f"Cake warp-decode inventory {context} references missing sequence"
            )
        stages = route.get("stages")
        if not isinstance(stages, list) or not stages:
            raise ValueError(
                f"Cake warp-decode inventory {context}.stages must be non-empty"
            )
        stage_keys = []
        for stage_index, raw_stage in enumerate(stages):
            stage = _require_dict(raw_stage, f"{context}.stages[{stage_index}]")
            member = _require_dict(
                stage.get("module"), f"{context}.stages[{stage_index}].module"
            )
            member_key = (arch, member.get("name"), member.get("role"))
            stage_keys.append(member_key)
            if (
                arch == target_arch
                and args.get("activation") == "silu"
                and tuple(
                    args.get(name)
                    for name in (
                        "hidden_size",
                        "intermediate_size",
                        "num_experts",
                        "local_num_experts",
                        "top_k",
                    )
                )
                == (6144, 1536, 192, 192, 4)
                and stage.get("name") == "fc1"
                and stage.get("template") in silu_tokens
            ):
                tokens = args.get("num_tokens")
                if not isinstance(tokens, int) or isinstance(tokens, bool):
                    raise ValueError(
                        f"Cake warp-decode inventory {context} num_tokens is invalid"
                    )
                silu_tokens[stage["template"]].add(tokens)
        if tuple(stage_keys) != sequence_modules[sequence_key]:
            raise ValueError(
                f"Cake warp-decode inventory {context} stage and sequence order differs"
            )
        route_module_union.update(stage_keys)
        route_sequences.add(sequence_key)
    if route_module_union != set(module_sources) or route_sequences != set(
        sequence_modules
    ):
        raise ValueError("Cake warp-decode inventory route closure differs")
    device_sources = list(
        dict.fromkeys(
            source for key, source in module_sources.items() if key[0] == target_arch
        )
    )
    if not device_sources:
        raise ValueError(
            f"Cake warp-decode inventory has no modules for exact target {target_arch}"
        )
    has_silu = silu_tokens["fc1_silu_static"] == {1} and silu_tokens[
        "fc1_silu_persistent"
    ] == set(range(2, 33))
    return (
        device_sources,
        has_silu,
        {source: device_compile_flags[source] for source in device_sources},
    )


def get_cake_fused_moe_warp_decode_uri(
    target: CakeWarpDecodeTarget = "sm103a",
) -> str:
    """Return the exact-architecture Cake warp-decode JIT module key."""

    if target not in _TARGET_FLAGS:
        raise ValueError(f"unsupported Cake warp-decode target: {target}")
    return _MODULE_URI[target]


@functools.cache
def gen_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
) -> JitSpec:
    """Generate one exact-architecture Cake warp-decode JIT module."""

    uri = get_cake_fused_moe_warp_decode_uri(target)
    csrc_dir = _get_cake_fused_moe_warp_decode_csrc_dir()
    generated_dir = csrc_dir / "generated"
    generated_sources, has_silu, compile_flags = _load_exported_device_sources(
        csrc_dir, target
    )
    required_files = (
        csrc_dir / _BINDING_SOURCE,
        generated_dir / _GENERATED_MANIFEST,
        csrc_dir / _CONTRACT_HEADER,
    )
    for source in required_files:
        if not source.is_file():
            raise FileNotFoundError(f"Cake warp-decode source not found: {source}")

    # Only exported device TUs are compiled. Their generated FFI bindings each
    # own an entry named by the export manifest and would conflict with the
    # production receipt/Graph-aware binding's single typed `run` alias.
    target_flags = [
        *_TARGET_FLAGS[target],
        f"-DFLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR={_TARGET_MINOR[target]}",
        f"-DFLASHINFER_CAKE_WARP_DECODE_HAS_SILU={int(has_silu)}",
    ]
    spec = gen_jit_spec(
        name=uri,
        sources=[
            *generated_sources,
            csrc_dir / _BINDING_SOURCE,
        ],
        extra_cuda_cflags=target_flags,
        extra_cuda_cflags_by_source={
            source: [*target_flags, *flags] for source, flags in compile_flags.items()
        },
        extra_ldflags=["-lcuda"],
        extra_include_paths=[
            csrc_dir,
            generated_dir,
            csrc_dir.parents[1],
            _get_include_dir(),
        ],
    )
    logger.info(f"Generated Cake warp-decode {target} JIT spec: {spec.name}")
    return spec


@functools.cache
def _build_and_load_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
) -> Any:
    module = gen_cake_fused_moe_warp_decode_module(target).build_and_load()
    logger.info(f"Loaded Cake warp-decode {target} module")
    return module


def _get_compute_capability(device: Any = None) -> tuple[int, int]:
    # Keep the heavyweight runtime dependency out of module import. This JIT
    # module is also imported by source-only packaging and AOT tooling.
    import torch  # noqa: PLC0415

    from ..utils import get_compute_capability  # noqa: PLC0415

    resolved_device = torch.device("cuda") if device is None else torch.device(device)
    return get_compute_capability(resolved_device)


def _check_exact_target(target: CakeWarpDecodeTarget, device: Any = None) -> None:
    major, minor = _get_compute_capability(device)
    expected_minor = _TARGET_MINOR[target]
    if (major, minor) != (10, expected_minor):
        raise RuntimeError(
            f"Cake warp decode target {target} requires exact compute capability "
            f"10.{expected_minor}, "
            f"got {major}.{minor}"
        )


def load_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
    *,
    device: Any = None,
) -> Any:
    """Build or load the module after checking the requested CUDA device."""

    get_cake_fused_moe_warp_decode_uri(target)
    _check_exact_target(target, device)
    return _build_and_load_cake_fused_moe_warp_decode_module(target)


def get_cake_fused_moe_warp_decode_module(
    target: CakeWarpDecodeTarget = "sm103a",
    *,
    device: Any = None,
) -> Any:
    """Return the module exporting size, prepare, launch, and receipt release."""

    return load_cake_fused_moe_warp_decode_module(target, device=device)


__all__ = [
    "CakeWarpDecodeTarget",
    "gen_cake_fused_moe_warp_decode_module",
    "get_cake_fused_moe_warp_decode_module",
    "get_cake_fused_moe_warp_decode_uri",
    "load_cake_fused_moe_warp_decode_module",
]
