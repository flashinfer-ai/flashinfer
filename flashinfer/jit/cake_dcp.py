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
from collections.abc import Mapping
from pathlib import Path
import json
from typing import Any, Literal, Optional

from . import env as jit_env
from .cake_fmha import CAKE_FMHA_JIT_TAG, get_cake_fmha_csrc_dir
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
)

DcpSpecTarget = Literal["sm100a", "sm103a", "sm100f"]

_DCP_SPEC_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
# The DCP registry shares source bodies between SM100 and SM103. Compile those
# bodies with the family target for SM107.
_TARGET_MANIFEST_ARCH = {
    "sm100a": "sm_100a",
    "sm103a": "sm_103a",
    "sm100f": "sm_100a",
}
_DCP_JIT_BINDINGS = {
    "dcp_spec_bf16_balanced": "jit/cake_fmha_dcp_spec_bf16_balanced_jit_binding.cu",
    "dcp_spec_bf16_fp8_balanced": (
        "jit/cake_fmha_dcp_spec_bf16_fp8_balanced_jit_binding.cu"
    ),
    "dcp_spec_bf16_fp8_d256_balanced": (
        "jit/cake_fmha_dcp_spec_bf16_fp8_d256_balanced_jit_binding.cu"
    ),
}


@functools.cache
def get_dcp_spec_registry() -> dict[str, Any]:
    """Load the checked-in DCP speculative-decode registry."""

    registry_path = get_cake_fmha_csrc_dir() / "cuda" / "dcp_spec" / "registry.json"
    registry = json.loads(registry_path.read_text())
    if registry.get("name") != "cake_fmha_dcp_spec":
        raise RuntimeError("Cake FMHA DCP registry has an invalid product identifier")
    return registry


def _get_dcp_family(name: str) -> Mapping[str, Any]:
    try:
        return get_dcp_spec_registry()["families"][name]
    except KeyError as exc:
        raise RuntimeError(f"Cake FMHA DCP family is missing: {name}") from exc


def _get_dcp_member(family_name: str, selector: Mapping[str, int]) -> Mapping[str, Any]:
    """The one ``source_family`` member of a DCP family with ``selector``."""

    family = _get_dcp_family(family_name)
    matches = [
        entry
        for entry in family["source_family"]
        if entry.get("selector") == dict(selector)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Cake FMHA DCP selector is not unique: {family_name} {dict(selector)!r}"
        )
    return matches[0]


def _get_dcp_sources(
    family_name: str,
    target: DcpSpecTarget,
    selector: Mapping[str, int],
) -> tuple[Path, Path]:
    member = _get_dcp_member(family_name, selector)
    csrc_dir = get_cake_fmha_csrc_dir()
    body = csrc_dir / member["sources"][_TARGET_MANIFEST_ARCH[target]]
    binding = csrc_dir / _DCP_JIT_BINDINGS[family_name]
    for source in (body, binding):
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA DCP source not found: {source}")
    return body, binding


# ---------------------------------------------------------------------------
# Static DCP programs (one program per route instance and architecture)
# ---------------------------------------------------------------------------
#
# ``registry["programs"]`` lists every delivered program once (device and
# binding sources, nvcc flags, FFI entry, argument plan, architectures, the
# compile-line constants its text leaves to the loader) and
# ``registry["static_routes"]`` maps ``family -> instance -> architecture ->
# program`` (an instance whose text is the same for both architectures binds
# one program for both; the head_dim-256 FP8 instances bind one per
# architecture, their MMA issue form differs per target).  The
# batch size and rank are runtime kernel arguments; speculative length, heads,
# world and (split-KV families) the split count are defined on the compile line
# (``-DNAME=value``), so one module per program, target and constant set.  The
# program name is the delivered-text identity the exporter assigned.

DcpStaticFamily = Literal["bf16_v1", "bf16_v4", "fp8_d128", "fp8_d256"]
DCP_STATIC_FAMILIES: tuple[str, ...] = ("bf16_v1", "bf16_v4", "fp8_d128", "fp8_d256")


def _static_program(
    family: str, instance: str, target: str
) -> tuple[str, Mapping[str, Any]]:
    registry = get_dcp_spec_registry()
    try:
        name = registry["static_routes"][family][instance][
            _TARGET_MANIFEST_ARCH[target]
        ]
    except KeyError as exc:
        raise ValueError(
            f"unknown static DCP route {family}/{instance} for target {target}"
        ) from exc
    return name, registry["programs"][name]


def _static_defines(
    program: Mapping[str, Any], specializations: Mapping[str, int]
) -> tuple[tuple[str, int], ...]:
    """``program``'s compile-line constants from ``specializations`` (exactly its declared names)."""

    names = list(program["specializations"])
    missing = sorted(set(names) - set(specializations))
    extra = sorted(set(specializations) - set(names))
    if missing or extra:
        raise ValueError(
            f"static DCP program expects the constants {names}; "
            f"missing {missing}, unexpected {extra}"
        )
    return tuple((name, int(specializations[name])) for name in sorted(names))


def get_dcp_spec_static_uri(
    family: DcpStaticFamily,
    instance: str,
    target: DcpSpecTarget,
    specializations: Mapping[str, int],
) -> str:
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    name, program = _static_program(family, instance, target)
    suffix = "".join(
        f"_{define}{value}"
        for define, value in _static_defines(program, specializations)
    )
    return f"cake_fmha_dcp_spec_{family}_{instance}_{target}_{name}{suffix}"


def gen_dcp_spec_static_module(
    family: DcpStaticFamily,
    instance: str,
    target: DcpSpecTarget,
    specializations: Mapping[str, int],
) -> JitSpec:
    """Generate one static DCP module (one per program, target and compile-line constant set)."""

    uri = get_dcp_spec_static_uri(family, instance, target, specializations)
    _name, program = _static_program(family, instance, target)
    if _TARGET_MANIFEST_ARCH[target] not in program["arches"]:
        raise ValueError(f"{family}/{instance} is not delivered for {target}")
    csrc_dir = get_cake_fmha_csrc_dir()
    sources = [csrc_dir / source for source in program["sources"]]
    for source in sources:
        if not source.is_file():
            raise FileNotFoundError(f"Cake FMHA DCP source not found: {source}")
    spec = gen_jit_spec(
        name=uri,
        sources=sources,
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            *program["compile_flags"],
            *[
                f"-D{define}={value}"
                for define, value in _static_defines(program, specializations)
            ],
        ],
        extra_include_paths=[
            csrc_dir / "cuda" / "dcp_spec",
            csrc_dir,
            jit_env.FLASHINFER_CSRC_DIR,
        ],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated static DCP speculative FMHA JIT spec: {spec.name}")
    return spec


def load_dcp_spec_static_module(
    family: DcpStaticFamily,
    instance: str,
    target: DcpSpecTarget,
    specializations: Mapping[str, int],
):
    return _load_dcp_spec_static_module(
        family,
        instance,
        target,
        tuple(sorted((str(k), int(v)) for k, v in specializations.items())),
    )


@functools.cache
def _load_dcp_spec_static_module(
    family: DcpStaticFamily,
    instance: str,
    target: DcpSpecTarget,
    specializations: tuple[tuple[str, int], ...],
):
    module = gen_dcp_spec_static_module(
        family, instance, target, dict(specializations)
    ).build_and_load()
    logger.info(f"Loaded static DCP speculative FMHA module: {module}")
    return module


# ---------------------------------------------------------------------------
# On-device load-balanced DCP families (CAKE-685 round 3)
# ---------------------------------------------------------------------------
#
# Each family ships shape-independent programs for both architectures
# (``cuda/dcp_spec/<family>/kernel.cu``, or one ``kernel_<key><value>.cu`` per
# program of a family with ``program_variants``; the shared-base prologue is
# switched by ``__CUDA_ARCH__``).  Its 32- and 64-row packed instances are the
# manifest members' ``defines`` (``-DN_ROWS=32`` / ``64``), selected by the
# packed-row tile the request's speculative rows need; batch, heads, lengths,
# rank and world are runtime kernel arguments.

DcpBalancedFamily = Literal[
    "dcp_spec_bf16_balanced",
    "dcp_spec_bf16_fp8_balanced",
    "dcp_spec_bf16_fp8_d256_balanced",
]
DCP_BALANCED_FAMILIES: tuple[str, ...] = (
    "dcp_spec_bf16_balanced",
    "dcp_spec_bf16_fp8_balanced",
    "dcp_spec_bf16_fp8_d256_balanced",
)
DCP_BALANCED_N_ROWS: tuple[int, ...] = (32, 64)
# Manifest selector of one packed-row instance inside a balanced family.
_DCP_BALANCED_SELECTOR_KEY = "n_rows"


def dcp_balanced_program_variants(family: str) -> Optional[Mapping[str, Any]]:
    """The ``program_variants`` block of a balanced family, or ``None`` when the family ships one program.

    A family with program variants ships one traced program per value of its
    selector ``key`` (each a ``-DN_ROWS`` body of its own) and the host selects
    the program from launch metadata (:func:`flashinfer.cake_dcp.dcp_balanced_program`).
    """

    addon = get_cake_fmha_manifest()["add_ons"]["cake_fmha_dcp_spec"]
    route = addon["manifest"].get("balanced_routes", {}).get(family)
    return None if route is None else route.get("program_variants")


def _validate_balanced_specialization(
    family: str, target: DcpSpecTarget, n_rows: int, program: Optional[int] = None
) -> None:
    if family not in DCP_BALANCED_FAMILIES:
        raise ValueError(f"unsupported balanced DCP family: {family}")
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if n_rows not in DCP_BALANCED_N_ROWS:
        raise ValueError(
            f"balanced DCP n_rows must be one of {DCP_BALANCED_N_ROWS}, got {n_rows}"
        )
    variants = dcp_balanced_program_variants(family)
    if variants is None:
        if program is not None:
            raise ValueError(
                f"balanced DCP family {family} ships one program; got program={program!r}"
            )
    elif program not in [int(value) for value in variants["values"]]:
        raise ValueError(
            f"balanced DCP family {family} program ({variants['key']}) must be one of "
            f"{list(variants['values'])}, got {program!r}"
        )


def _balanced_selector(
    family: str, n_rows: int, program: Optional[int]
) -> dict[str, int]:
    selector = {_DCP_BALANCED_SELECTOR_KEY: n_rows}
    variants = dcp_balanced_program_variants(family)
    if variants is not None:
        selector[str(variants["key"])] = int(program)  # type: ignore[arg-type]
    return selector


def get_dcp_spec_balanced_uri(
    family: DcpBalancedFamily,
    target: DcpSpecTarget,
    n_rows: int,
    program: Optional[int] = None,
) -> str:
    _validate_balanced_specialization(family, target, n_rows, program)
    variants = dcp_balanced_program_variants(family)
    program_tag = "" if variants is None else f"_{variants['key']}{int(program)}"  # type: ignore[arg-type]
    return f"cake_fmha_{family}_n{n_rows}{program_tag}_{target}_{CAKE_FMHA_JIT_TAG}"


def _get_dcp_balanced_sources(
    family: DcpBalancedFamily,
    target: DcpSpecTarget,
    n_rows: int,
    program: Optional[int] = None,
) -> tuple[Path, Path, Path, Mapping[str, int]]:
    """``(program body, exported launch binding, FlashInfer adapter, instance defines)`` of one instance."""

    selector = _balanced_selector(family, n_rows, program)
    body, api_binding = _get_dcp_sources(family, target, selector)
    defines = dict(_get_dcp_member(family, selector).get("defines", {}))
    if defines.get("N_ROWS") != n_rows:
        raise RuntimeError(
            f"Cake FMHA DCP member {family} {selector!r} does not define N_ROWS={n_rows}"
        )
    launch_binding = (
        get_cake_fmha_csrc_dir() / _get_dcp_family(family)["binding_source"]
    )
    if not launch_binding.is_file():
        raise FileNotFoundError(f"Cake FMHA DCP source not found: {launch_binding}")
    return body, launch_binding, api_binding, defines


@functools.cache
def gen_dcp_spec_balanced_module(
    family: DcpBalancedFamily,
    target: DcpSpecTarget,
    n_rows: int,
    program: Optional[int] = None,
) -> JitSpec:
    """Generate one shape-independent balanced DCP module (one per packed tile and program).

    The family's program (``program`` selects it for families with
    ``program_variants``; see :func:`dcp_balanced_program_variants`) is
    instantiated with the manifest member's defines (``-DN_ROWS``).  The
    exported launch binding owns the kernel's thread count and dynamic shared
    memory; the FlashInfer adapter encodes the tensor maps, carves the
    caller-owned scratch and calls it as ``CAKE_FMHA_DCP_BALANCED_LAUNCH``.
    """

    uri = get_dcp_spec_balanced_uri(family, target, n_rows, program)
    body, launch_binding, api_binding, defines = _get_dcp_balanced_sources(
        family, target, n_rows, program
    )
    manifest_family = _get_dcp_family(family)
    csrc_dir = get_cake_fmha_csrc_dir()

    spec = gen_jit_spec(
        name=uri,
        sources=[body, launch_binding, api_binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            *(f"-D{name}={value}" for name, value in sorted(defines.items())),
            f"-DCAKE_FMHA_DCP_BALANCED_LAUNCH={manifest_family['launch_binding']}",
        ],
        extra_include_paths=[csrc_dir, jit_env.FLASHINFER_CSRC_DIR],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated balanced DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def load_dcp_spec_balanced_module(
    family: DcpBalancedFamily,
    target: DcpSpecTarget,
    n_rows: int,
    program: Optional[int] = None,
):
    module = gen_dcp_spec_balanced_module(
        family, target, n_rows, program
    ).build_and_load()
    logger.info(f"Loaded balanced DCP speculative FMHA module: {module}")
    return module


__all__ = [
    "DCP_BALANCED_FAMILIES",
    "DCP_BALANCED_N_ROWS",
    "DCP_STATIC_FAMILIES",
    "DcpBalancedFamily",
    "DcpSpecTarget",
    "DcpStaticFamily",
    "dcp_balanced_program_variants",
    "gen_dcp_spec_balanced_module",
    "gen_dcp_spec_static_module",
    "get_dcp_spec_balanced_uri",
    "get_dcp_spec_registry",
    "get_dcp_spec_static_uri",
    "load_dcp_spec_balanced_module",
    "load_dcp_spec_static_module",
]
