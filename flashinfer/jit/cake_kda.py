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
from dataclasses import dataclass
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

CakeKDAVariant = Literal[
    "m128_unbounded_softplus",
    "m128_bt64_unbounded_softplus",
]
CakeKDATarget = Literal["sm100a", "sm103a"]
CakeKDAAffineRole = Literal["main", "map", "scan", "correction"]

CAKE_KDA_VARIANTS: tuple[CakeKDAVariant, ...] = (
    "m128_unbounded_softplus",
    "m128_bt64_unbounded_softplus",
)
CAKE_KDA_AFFINE_ROLES: tuple[CakeKDAAffineRole, ...] = (
    "main",
    "map",
    "scan",
    "correction",
)

_CAKE_KDA_AFFINE_MANIFEST = (
    "cake_kda_bf16_affine_unbounded_softplus_import_manifest.json"
)
_CAKE_KDA_AFFINE_CONTRACT = {
    "batch_size": 1,
    "beta_layout": "contiguous",
    "checkpoint_mode": "none",
    "dtype": "bfloat16",
    "gate_kind": "unbounded_softplus",
    "head_dim": 128,
    "head_relationship": "equal_q_kv",
    "initial_state": "indexed_bfloat16_pool",
    "max_local_heads": 32,
    "min_parts": 2,
    "min_tokens": 8192,
    "state_publication": "final_only",
    "targets": ["sm100a", "sm103a"],
    "token_multiple": 32,
}

_CAKE_KDA_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
}
_CAKE_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
}

# Keep the frozen cache key tied to the complete generated-plus-integration
# implementation so an installed cache cannot satisfy a refreshed export.
_CAKE_KDA_MODULE_IDENTS = {
    "m128_unbounded_softplus": "d7a7b33c69",
    "m128_bt64_unbounded_softplus": "8f5147c17f",
}


@dataclass(frozen=True)
class CakeKDAAffineModuleSpec:
    """One verified target-and-role source closure from the sealed export."""

    target: CakeKDATarget
    role: CakeKDAAffineRole
    module_ident: str
    binding_path: Path
    sources: tuple[Path, ...]


def _get_cake_kda_csrc_dir() -> Path:
    """Locate frozen CakeKDA sources in installed and source checkouts."""

    installed = jit_env.FLASHINFER_CSRC_DIR / "kda"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "kda"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "CakeKDA CUDA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _get_cake_kda_include_dir() -> Path:
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


def _require_affine_manifest(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(f"invalid Cake KDA affine import manifest: {message}")


def _resolve_affine_manifest_file(
    csrc_dir: Path, value: object, label: str
) -> Path:
    _require_affine_manifest(isinstance(value, str) and bool(value), f"{label} missing")
    relative = PurePosixPath(value)
    _require_affine_manifest(
        not relative.is_absolute()
        and ".." not in relative.parts
        and relative.parts[:2] == ("csrc", "kda")
        and len(relative.parts) == 3,
        f"{label} must name one csrc/kda file",
    )
    path = csrc_dir / relative.name
    _require_affine_manifest(
        path.name.startswith("cake_kda_") and path.suffix in (".cu", ".cuh"),
        f"{label} must use a public cake_kda CUDA filename",
    )
    _require_affine_manifest(path.is_file(), f"{label} does not exist: {path}")
    return path


def _verify_affine_manifest_file(
    csrc_dir: Path,
    *,
    path_value: object,
    sha256_value: object,
    label: str,
) -> Path:
    path = _resolve_affine_manifest_file(csrc_dir, path_value, f"{label}.path")
    _require_affine_manifest(
        isinstance(sha256_value, str)
        and len(sha256_value) == 64
        and all(character in "0123456789abcdef" for character in sha256_value),
        f"{label}.sha256 must be one full lowercase SHA-256",
    )
    actual = hashlib.sha256(path.read_bytes()).hexdigest()
    _require_affine_manifest(
        actual == sha256_value,
        f"{label}.sha256 mismatch: {actual} != {sha256_value}",
    )
    return path


@functools.cache
def get_cake_kda_affine_module_specs() -> tuple[CakeKDAAffineModuleSpec, ...]:
    """Read and verify all affine modules, or return empty while pending.

    The checked-in pending manifest intentionally keeps the route disabled.
    The sealed importer replaces it atomically only after all eight generated
    target/role closures and their full SHA-256 identities are available.
    """

    csrc_dir = _get_cake_kda_csrc_dir()
    manifest_path = csrc_dir / _CAKE_KDA_AFFINE_MANIFEST
    payload: Any = json.loads(manifest_path.read_text())
    _require_affine_manifest(isinstance(payload, dict), "root must be an object")
    _require_affine_manifest(payload.get("schema_version") == 1, "unsupported schema")
    _require_affine_manifest(
        payload.get("contract") == _CAKE_KDA_AFFINE_CONTRACT,
        "contract mismatch",
    )
    status = payload.get("status")
    modules = payload.get("modules")
    _require_affine_manifest(isinstance(modules, list), "modules must be a list")
    if status == "pending_generated_sources":
        _require_affine_manifest(not modules, "pending manifest must not list modules")
        return ()
    _require_affine_manifest(status == "complete", f"unsupported status {status!r}")

    expected = {
        (target, role)
        for target in _CAKE_KDA_NVCC_FLAGS
        for role in CAKE_KDA_AFFINE_ROLES
    }
    observed: set[tuple[str, str]] = set()
    specs: list[CakeKDAAffineModuleSpec] = []
    for index, item in enumerate(modules):
        label = f"modules[{index}]"
        _require_affine_manifest(isinstance(item, dict), f"{label} must be an object")
        target = item.get("target")
        role = item.get("role")
        _require_affine_manifest(target in _CAKE_KDA_NVCC_FLAGS, f"{label}.target")
        _require_affine_manifest(role in CAKE_KDA_AFFINE_ROLES, f"{label}.role")
        key = (target, role)
        _require_affine_manifest(key not in observed, f"duplicate module {target}/{role}")
        observed.add(key)

        module_ident = item.get("module_ident")
        _require_affine_manifest(
            isinstance(module_ident, str)
            and module_ident.startswith("cake_kda_")
            and module_ident.replace("_", "").isalnum()
            and module_ident == module_ident.lower(),
            f"{label}.module_ident must be a public cake_kda symbol",
        )
        binding_path = _verify_affine_manifest_file(
            csrc_dir,
            path_value=item.get("binding_path"),
            sha256_value=item.get("binding_sha256"),
            label=f"{label}.binding",
        )
        _require_affine_manifest(
            binding_path.suffix == ".cu", f"{label}.binding must be a .cu file"
        )
        source_items = item.get("sources")
        _require_affine_manifest(
            isinstance(source_items, list) and bool(source_items),
            f"{label}.sources must be non-empty",
        )
        source_paths = tuple(
            _verify_affine_manifest_file(
                csrc_dir,
                path_value=source.get("path") if isinstance(source, dict) else None,
                sha256_value=(
                    source.get("sha256") if isinstance(source, dict) else None
                ),
                label=f"{label}.sources[{source_index}]",
            )
            for source_index, source in enumerate(source_items)
        )
        specs.append(
            CakeKDAAffineModuleSpec(
                target=target,
                role=role,
                module_ident=module_ident,
                binding_path=binding_path,
                sources=source_paths,
            )
        )

    _require_affine_manifest(
        observed == expected,
        f"target/role set mismatch: missing={sorted(expected - observed)}, "
        f"extra={sorted(observed - expected)}",
    )
    specs.sort(key=lambda spec: (spec.target, CAKE_KDA_AFFINE_ROLES.index(spec.role)))
    return tuple(specs)


def cake_kda_affine_is_available() -> bool:
    """Return whether the complete sealed affine export is installed."""

    return len(get_cake_kda_affine_module_specs()) == (
        len(_CAKE_KDA_NVCC_FLAGS) * len(CAKE_KDA_AFFINE_ROLES)
    )


def get_cake_kda_affine_module_spec(
    target: CakeKDATarget, role: CakeKDAAffineRole
) -> CakeKDAAffineModuleSpec:
    """Return one verified affine source closure."""

    for spec in get_cake_kda_affine_module_specs():
        if spec.target == target and spec.role == role:
            return spec
    raise RuntimeError(
        "Cake KDA affine generated sources are not installed for "
        f"{target}/{role}; run tools/import-cake-kda-prefill-affine with the "
        "complete sealed bundle"
    )


def get_cake_kda_affine_uri(
    target: CakeKDATarget, role: CakeKDAAffineRole
) -> str:
    """Return the exact target-and-role cache identity from the sealed export."""

    spec = get_cake_kda_affine_module_spec(target, role)
    return f"{spec.module_ident}_{target}_{role}"


@functools.cache
def gen_cake_kda_affine_module(
    target: CakeKDATarget, role: CakeKDAAffineRole
) -> JitSpec:
    """Generate one verified affine target-and-role JIT module."""

    spec = get_cake_kda_affine_module_spec(target, role)
    csrc_dir = _get_cake_kda_csrc_dir()
    jit_spec = gen_jit_spec(
        name=get_cake_kda_affine_uri(target, role),
        sources=[spec.binding_path],
        extra_cuda_cflags=[
            *_CAKE_KDA_NVCC_FLAGS[target],
            _CAKE_KDA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            _get_cake_kda_include_dir(),
        ],
    )
    logger.info(f"Generated Cake KDA affine {role} {target} JIT spec: {jit_spec.name}")
    return jit_spec


@functools.cache
def load_cake_kda_affine_module(
    target: CakeKDATarget, role: CakeKDAAffineRole
):
    """Build or load one verified affine target-and-role module."""

    module = gen_cake_kda_affine_module(target, role).build_and_load()
    logger.info(f"Loaded Cake KDA affine {role} {target} module")
    return module


def get_cake_kda_affine_module(target: CakeKDATarget, role: CakeKDAAffineRole):
    """Return one loaded affine module for the host-side composite."""

    return load_cake_kda_affine_module(target, role)


def get_cake_kda_uri(variant: CakeKDAVariant, target: CakeKDATarget) -> str:
    """Return the target-specific JIT/AOT key for one schedule."""

    if variant not in CAKE_KDA_VARIANTS:
        raise ValueError(f"unsupported CakeKDA variant: {variant}")
    if target not in _CAKE_KDA_NVCC_FLAGS:
        raise ValueError(f"unsupported CakeKDA target: {target}")
    module_ident = _CAKE_KDA_MODULE_IDENTS[variant]
    return f"cake_kda_bf16_fused_{variant}_{module_ident}_{target}"


@functools.cache
def gen_cake_kda_module(variant: CakeKDAVariant, target: CakeKDATarget) -> JitSpec:
    """Generate one exact-SM100a or exact-SM103a JIT module.

    Each physical schedule is compiled in its own translation unit because the
    checked-in frozen sources intentionally retain generated helper names and
    macros. ``gen_jit_spec`` supplies FlashInfer's standard ``-use_fast_math``
    flag. B200 and B300 use separate exact targets and therefore separate
    cubins and cache identities.
    """

    csrc_dir = _get_cake_kda_csrc_dir()
    include_dir = _get_cake_kda_include_dir()
    uri = get_cake_kda_uri(variant, target)
    binding = csrc_dir / f"cake_kda_bf16_fused_{variant}_binding.cu"
    if not binding.exists():
        raise FileNotFoundError(f"CakeKDA binding source not found: {binding}")

    spec = gen_jit_spec(
        name=uri,
        sources=[binding],
        extra_cuda_cflags=[
            *_CAKE_KDA_NVCC_FLAGS[target],
            _CAKE_KDA_TARGET_DEFINE[target],
        ],
        extra_include_paths=[
            csrc_dir,
            csrc_dir.parent,
            include_dir,
        ],
    )
    logger.info(f"Generated CakeKDA {variant} {target} JIT spec: {spec.name}")
    return spec


def gen_cake_kda_m128_unbounded_softplus_module(target: CakeKDATarget) -> JitSpec:
    """Generate the native unbounded-softplus M128 module."""

    return gen_cake_kda_module("m128_unbounded_softplus", target)


def gen_cake_kda_m128_bt64_unbounded_softplus_module(
    target: CakeKDATarget,
) -> JitSpec:
    """Generate the checkpoint-aligned native unbounded-softplus BT64 module."""

    return gen_cake_kda_module("m128_bt64_unbounded_softplus", target)


@functools.cache
def load_cake_kda_module(variant: CakeKDAVariant, target: CakeKDATarget):
    """Build or load one physical, target-specific CakeKDA module."""

    module = gen_cake_kda_module(variant, target).build_and_load()
    logger.info(f"Loaded CakeKDA {variant} {target} module")
    return module


def load_cake_kda_m128_unbounded_softplus_module(target: CakeKDATarget):
    """Load the native unbounded-softplus M128 module."""

    return load_cake_kda_module("m128_unbounded_softplus", target)


def load_cake_kda_m128_bt64_unbounded_softplus_module(target: CakeKDATarget):
    """Load the checkpoint-aligned native unbounded-softplus BT64 module."""

    return load_cake_kda_module("m128_bt64_unbounded_softplus", target)


def get_cake_kda_prefill_module(variant: CakeKDAVariant, target: CakeKDATarget):
    """Return the loaded module used by the recurrent-KDA prefill dispatcher."""

    return load_cake_kda_module(variant, target)


__all__ = [
    "CAKE_KDA_AFFINE_ROLES",
    "CAKE_KDA_VARIANTS",
    "CakeKDAAffineModuleSpec",
    "CakeKDAAffineRole",
    "CakeKDATarget",
    "CakeKDAVariant",
    "cake_kda_affine_is_available",
    "gen_cake_kda_affine_module",
    "gen_cake_kda_m128_bt64_unbounded_softplus_module",
    "gen_cake_kda_m128_unbounded_softplus_module",
    "gen_cake_kda_module",
    "get_cake_kda_affine_module",
    "get_cake_kda_affine_module_spec",
    "get_cake_kda_affine_module_specs",
    "get_cake_kda_affine_uri",
    "get_cake_kda_prefill_module",
    "get_cake_kda_uri",
    "load_cake_kda_m128_bt64_unbounded_softplus_module",
    "load_cake_kda_m128_unbounded_softplus_module",
    "load_cake_kda_affine_module",
    "load_cake_kda_module",
]
