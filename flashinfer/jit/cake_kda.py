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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ._kda_jit_common import (
    gen_kda_jit_spec,
    get_kda_csrc_dir as _get_cake_kda_csrc_dir,
    get_flashinfer_include_dir as _get_cake_kda_include_dir,
)
from .core import JitSpec, logger

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

_CAKE_KDA_TARGETS: tuple[CakeKDATarget, ...] = ("sm100a", "sm103a")
_CAKE_KDA_TARGET_DEFINE = {
    "sm100a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=0",
    "sm103a": "-DFLASHINFER_CAKE_KDA_TARGET_MINOR=3",
}

# Headers shared by every affine translation unit.
_CAKE_KDA_AFFINE_COMMON_HEADERS = (
    "cake_kda_affine_binding_common.cuh",
    "cake_kda_binding_common.cuh",
)
# Role -> (binding translation unit, frozen generated body, role launcher header).
_CAKE_KDA_AFFINE_ROLE_SOURCES: dict[CakeKDAAffineRole, tuple[str, str, str]] = {
    "main": (
        "cake_kda_affine_unbounded_softplus_main_binding.cu",
        "cake_kda_affine_unbounded_softplus_main.cu",
        "cake_kda_affine_direct_m128_binding.cuh",
    ),
    "map": (
        "cake_kda_affine_unbounded_softplus_map_binding.cu",
        "cake_kda_affine_unbounded_softplus_map.cu",
        "cake_kda_affine_direct_m128_binding.cuh",
    ),
    "scan": (
        "cake_kda_affine_unbounded_softplus_scan_binding.cu",
        "cake_kda_affine_unbounded_softplus_scan.cu",
        "cake_kda_affine_scan_binding.cuh",
    ),
    "correction": (
        "cake_kda_affine_unbounded_softplus_correction_binding.cu",
        "cake_kda_affine_unbounded_softplus_correction.cu",
        "cake_kda_affine_direct_m128_binding.cuh",
    ),
}


@dataclass(frozen=True)
class CakeKDAAffineModuleSpec:
    """One target-and-role source closure of the affine composite."""

    target: CakeKDATarget
    role: CakeKDAAffineRole
    module_ident: str
    binding_path: Path
    sources: tuple[Path, ...]


def _cake_kda_source(csrc_dir: Path, name: str) -> Path:
    path = csrc_dir / name
    if not path.is_file():
        raise FileNotFoundError(f"Cake KDA source not found: {path}")
    return path


@functools.cache
def get_cake_kda_affine_module_specs() -> tuple[CakeKDAAffineModuleSpec, ...]:
    """Return the eight target-and-role affine module closures.

    JIT rebuilds follow the sources through ninja's dependency scan and AOT
    artifacts are built from the same tree, so the module identity is the
    plain role name.
    """

    csrc_dir = _get_cake_kda_csrc_dir()
    specs: list[CakeKDAAffineModuleSpec] = []
    for target in _CAKE_KDA_TARGETS:
        for role in CAKE_KDA_AFFINE_ROLES:
            binding, body, role_header = _CAKE_KDA_AFFINE_ROLE_SOURCES[role]
            specs.append(
                CakeKDAAffineModuleSpec(
                    target=target,
                    role=role,
                    module_ident=f"cake_kda_affine_unbounded_softplus_{role}",
                    binding_path=_cake_kda_source(csrc_dir, binding),
                    sources=tuple(
                        _cake_kda_source(csrc_dir, name)
                        for name in (body, role_header, *_CAKE_KDA_AFFINE_COMMON_HEADERS)
                    ),
                )
            )
    return tuple(specs)


def cake_kda_affine_is_available() -> bool:
    """Return whether every affine target-and-role closure is present."""

    return len(get_cake_kda_affine_module_specs()) == (
        len(_CAKE_KDA_TARGETS) * len(CAKE_KDA_AFFINE_ROLES)
    )


def get_cake_kda_affine_module_spec(
    target: CakeKDATarget, role: CakeKDAAffineRole
) -> CakeKDAAffineModuleSpec:
    """Return one affine source closure."""

    for spec in get_cake_kda_affine_module_specs():
        if spec.target == target and spec.role == role:
            return spec
    raise ValueError(f"unsupported Cake KDA affine module: {target}/{role}")


def get_cake_kda_affine_uri(target: CakeKDATarget, role: CakeKDAAffineRole) -> str:
    """Return the target-and-role cache identity."""

    spec = get_cake_kda_affine_module_spec(target, role)
    return f"{spec.module_ident}_{target}"


@functools.cache
def gen_cake_kda_affine_module(
    target: CakeKDATarget, role: CakeKDAAffineRole
) -> JitSpec:
    """Generate one affine target-and-role JIT module."""

    spec = get_cake_kda_affine_module_spec(target, role)
    jit_spec = gen_kda_jit_spec(
        name=get_cake_kda_affine_uri(target, role),
        sources=[spec.binding_path],
        target=target,
        target_define=_CAKE_KDA_TARGET_DEFINE[target],
        csrc_dir=_get_cake_kda_csrc_dir(),
        include_dir=_get_cake_kda_include_dir(),
    )
    logger.info(f"Generated Cake KDA affine {role} {target} JIT spec: {jit_spec.name}")
    return jit_spec


@functools.cache
def load_cake_kda_affine_module(target: CakeKDATarget, role: CakeKDAAffineRole):
    """Build or load one affine target-and-role module."""

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
    if target not in _CAKE_KDA_TARGETS:
        raise ValueError(f"unsupported CakeKDA target: {target}")
    return f"cake_kda_bf16_fused_{variant}_{target}"


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
    uri = get_cake_kda_uri(variant, target)
    binding = _cake_kda_source(csrc_dir, f"cake_kda_bf16_fused_{variant}_binding.cu")
    spec = gen_kda_jit_spec(
        name=uri,
        sources=[binding],
        target=target,
        target_define=_CAKE_KDA_TARGET_DEFINE[target],
        csrc_dir=csrc_dir,
        include_dir=_get_cake_kda_include_dir(),
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
