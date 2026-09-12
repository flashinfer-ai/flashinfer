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
_EXPORTED_MANIFEST = "cake_warp_decode_manifest.json"
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


def _resolve_export_binding_paths(
    csrc_dir: Path, raw_binding: Any, context: str
) -> tuple[Path, ...]:
    """Resolve an original binding TU or its target-owned shared replacement."""

    if isinstance(raw_binding, str):
        return (_resolve_export_path(csrc_dir, raw_binding, context),)

    binding = _require_dict(raw_binding, context)
    if binding.get("delivery") != "target_owned_shared":
        raise ValueError(
            f"Cake warp-decode export manifest {context}.delivery must be "
            "target_owned_shared"
        )
    paths = binding.get("paths")
    if not isinstance(paths, list) or not paths:
        raise ValueError(
            f"Cake warp-decode export manifest {context}.paths must be a non-empty array"
        )
    return tuple(
        _resolve_export_path(csrc_dir, raw_path, f"{context}.paths[{index}]")
        for index, raw_path in enumerate(paths)
    )


def _load_exported_device_sources(
    csrc_dir: Path, target: CakeWarpDecodeTarget
) -> tuple[list[Path], bool]:
    """Resolve one exact-architecture generated inventory without its FFI bindings."""

    generated_dir = csrc_dir / "generated"
    manifest_path = generated_dir / _EXPORTED_MANIFEST
    if not manifest_path.is_file():
        if target == "sm103a":
            legacy = generated_dir / _LEGACY_SM103A_GENERATED_SOURCE
            if legacy.is_file():
                return [legacy], False
        raise FileNotFoundError(
            f"Cake warp-decode {target} generated inventory was not found at "
            f"{manifest_path}; install the generated warp-decode kernel inventory"
        )

    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Cake warp-decode export manifest could not be read: {manifest_path}"
        ) from error
    manifest = _require_dict(payload, "root")
    if manifest.get("schema") != "cake.library_export.v4":
        raise ValueError(
            "Cake warp-decode export manifest schema must be cake.library_export.v4"
        )

    modules = manifest.get("modules")
    sequences = manifest.get("sequences")
    contract = _require_dict(manifest.get("contract"), "contract")
    routes = contract.get("routes")
    if not isinstance(modules, list) or not isinstance(sequences, list):
        raise ValueError(
            "Cake warp-decode export manifest modules and sequences must be arrays"
        )
    if not isinstance(routes, list):
        raise ValueError(
            "Cake warp-decode export manifest contract.routes must be an array"
        )

    target_arch = _TARGET_ARCH[target]
    module_sources: dict[tuple[str, str, str], Path] = {}
    module_symbols: dict[tuple[str, str, str], str] = {}
    for index, raw_module in enumerate(modules):
        module = _require_dict(raw_module, f"modules[{index}]")
        if module.get("arch") != target_arch:
            continue
        name = module.get("name")
        role = module.get("role")
        symbol = module.get("kernel_symbol")
        ffi_entry = module.get("ffi_entry")
        if not all(
            isinstance(value, str) and value
            for value in (name, role, symbol, ffi_entry)
        ):
            raise ValueError(
                f"Cake warp-decode export manifest modules[{index}] identity is incomplete"
            )
        key = (target_arch, name, role)
        if key in module_sources:
            raise ValueError(
                f"Cake warp-decode export manifest duplicates module {key!r}"
            )
        translation_units = _require_dict(
            module.get("translation_units"), f"modules[{index}].translation_units"
        )
        if translation_units.get("compile_separately") is not True:
            raise ValueError(
                f"Cake warp-decode export manifest modules[{index}] must compile separately"
            )
        module_sources[key] = _resolve_export_path(
            csrc_dir,
            translation_units.get("device"),
            f"modules[{index}].translation_units.device",
        )
        module_symbols[key] = symbol
        # Exported per-module bindings intentionally are not compiled: they all
        # own an FFI entry, while the production binding below owns the single
        # typed run alias plus receipt and CUDA Graph lifecycle entrypoints.
        _resolve_export_binding_paths(
            csrc_dir,
            translation_units.get("binding"),
            f"modules[{index}].translation_units.binding",
        )
    if not module_sources:
        raise ValueError(
            f"Cake warp-decode export manifest has no modules for exact target {target_arch}"
        )

    device_sources: list[Path] = []
    seen_device_sources: set[Path] = set()
    for index, raw_sequence in enumerate(sequences):
        sequence = _require_dict(raw_sequence, f"sequences[{index}]")
        if sequence.get("arch") != target_arch:
            continue
        ffi_entry = sequence.get("ffi_entry")
        if not isinstance(ffi_entry, str) or not ffi_entry:
            raise ValueError(
                f"Cake warp-decode export manifest sequences[{index}].ffi_entry is invalid"
            )
        translation_units = _require_dict(
            sequence.get("translation_units"), f"sequences[{index}].translation_units"
        )
        if translation_units.get("compile_separately") is not True:
            raise ValueError(
                f"Cake warp-decode export manifest sequences[{index}] must compile separately"
            )
        devices = translation_units.get("devices")
        if not isinstance(devices, list) or not devices:
            raise ValueError(
                f"Cake warp-decode export manifest sequences[{index}] devices must be non-empty"
            )
        for device_index, raw_device in enumerate(devices):
            source = _resolve_export_path(
                csrc_dir,
                raw_device,
                f"sequences[{index}].translation_units.devices[{device_index}]",
            )
            if source not in seen_device_sources:
                seen_device_sources.add(source)
                device_sources.append(source)
        _resolve_export_binding_paths(
            csrc_dir,
            translation_units.get("binding"),
            f"sequences[{index}].translation_units.binding",
        )
    if not device_sources:
        raise ValueError(
            f"Cake warp-decode export manifest has no sequences for exact target {target_arch}"
        )
    if set(module_sources.values()) != seen_device_sources:
        raise ValueError(
            f"Cake warp-decode {target_arch} module and sequence device inventories differ"
        )

    generated_header = generated_dir / _GENERATED_MANIFEST
    if not generated_header.is_file():
        raise FileNotFoundError(
            f"Cake warp-decode generated C++ manifest not found: {generated_header}"
        )
    header_text = generated_header.read_text(encoding="utf-8")
    missing_symbols = [
        symbol for symbol in module_symbols.values() if symbol not in header_text
    ]
    if missing_symbols:
        raise ValueError(
            f"Cake warp-decode generated C++ manifest is missing {target_arch} symbols: "
            + ", ".join(sorted(missing_symbols))
        )

    silu_tokens: dict[str, set[int]] = {
        "fc1_silu_static": set(),
        "fc1_silu_persistent": set(),
    }
    for index, raw_route in enumerate(routes):
        route = _require_dict(raw_route, f"contract.routes[{index}]")
        if route.get("arch") != target_arch:
            continue
        args = _require_dict(route.get("args"), f"contract.routes[{index}].args")
        if not (
            args.get("activation") == "silu"
            and args.get("hidden_size") == 6144
            and args.get("intermediate_size") == 1536
            and args.get("num_experts") == 192
            and args.get("local_num_experts") == 192
            and args.get("top_k") == 4
        ):
            continue
        num_tokens = args.get("num_tokens")
        if not isinstance(num_tokens, int) or isinstance(num_tokens, bool):
            raise ValueError(
                f"Cake warp-decode export manifest contract.routes[{index}] num_tokens is invalid"
            )
        stages = route.get("stages")
        if not isinstance(stages, list):
            raise ValueError(
                f"Cake warp-decode export manifest contract.routes[{index}].stages must be an array"
            )
        for stage_index, raw_stage in enumerate(stages):
            stage = _require_dict(
                raw_stage, f"contract.routes[{index}].stages[{stage_index}]"
            )
            template = stage.get("template")
            if stage.get("name") != "fc1" or template not in silu_tokens:
                continue
            module = _require_dict(
                stage.get("module"),
                f"contract.routes[{index}].stages[{stage_index}].module",
            )
            module_key = (target_arch, module.get("name"), module.get("role"))
            if module_key not in module_sources:
                raise ValueError(
                    f"Cake warp-decode SiLU route references missing module {module_key!r}"
                )
            if module_sources[module_key] not in seen_device_sources:
                raise ValueError(
                    f"Cake warp-decode SiLU route module {module_key!r} is not in its sequence"
                )
            silu_tokens[template].add(num_tokens)

    has_silu = silu_tokens["fc1_silu_static"] == {1} and silu_tokens[
        "fc1_silu_persistent"
    ] == set(range(2, 33))
    return device_sources, has_silu


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
    generated_sources, has_silu = _load_exported_device_sources(csrc_dir, target)
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
    spec = gen_jit_spec(
        name=uri,
        sources=[
            *generated_sources,
            csrc_dir / _BINDING_SOURCE,
        ],
        extra_cuda_cflags=[
            *_TARGET_FLAGS[target],
            f"-DFLASHINFER_CAKE_WARP_DECODE_TARGET_MINOR={_TARGET_MINOR[target]}",
            f"-DFLASHINFER_CAKE_WARP_DECODE_HAS_SILU={int(has_silu)}",
        ],
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
