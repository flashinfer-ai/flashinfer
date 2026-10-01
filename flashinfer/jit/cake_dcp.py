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
from typing import Any, Literal, Mapping, Optional

from . import env as jit_env
from .cake_fmha import (
    CAKE_FMHA_FLASHINFER_BINDINGS_SHA256,
    CAKE_FMHA_MANIFEST_SHA256,
    get_cake_fmha_csrc_dir,
    get_cake_fmha_manifest,
)
from .core import (
    JitSpec,
    gen_jit_spec,
    logger,
    sm100a_nvcc_flags,
    sm100f_nvcc_flags,
    sm103a_nvcc_flags,
)

DcpSpecVariant = Literal["v1", "v4"]
DcpSpecTarget = Literal["sm100a", "sm103a", "sm100f"]

_DCP_SPEC_NVCC_FLAGS = {
    "sm100a": sm100a_nvcc_flags,
    "sm103a": sm103a_nvcc_flags,
    "sm100f": sm100f_nvcc_flags,
}
# The DCP manifest shares source bodies between SM100 and SM103. Compile those
# authenticated bodies with the family target for SM107.
_TARGET_MANIFEST_ARCH = {
    "sm100a": "sm_100a",
    "sm103a": "sm_103a",
    "sm100f": "sm_100a",
}
_DCP_JIT_BINDINGS = {
    "dcp_spec_bf16_v1": "jit/cake_fmha_dcp_spec_bf16_v1_jit_binding.cu",
    "dcp_spec_bf16_v4": "jit/cake_fmha_dcp_spec_bf16_v4_jit_binding.cu",
    "dcp_spec_bf16_fp8": "jit/cake_fmha_dcp_spec_bf16_fp8_jit_binding.cu",
    "dcp_spec_bf16_balanced": "jit/cake_fmha_dcp_spec_bf16_balanced_jit_binding.cu",
    "dcp_spec_bf16_fp8_balanced": (
        "jit/cake_fmha_dcp_spec_bf16_fp8_balanced_jit_binding.cu"
    ),
    "dcp_spec_bf16_fp8_d256_balanced": (
        "jit/cake_fmha_dcp_spec_bf16_fp8_d256_balanced_jit_binding.cu"
    ),
}
_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 8)
_FP8_D256_SUPPORTED_Q_LENS = (1, 2, 3, 4, 5, 6, 7, 8)
_FP8_D256_SUPPORTED_SPLITS = (1, 2, 3, 4, 8, 16)
_SUPPORTED_CP_WORLDS = (1, 2, 4, 8)


def _get_dcp_family(name: str) -> Mapping[str, Any]:
    addon = get_cake_fmha_manifest()["add_ons"]["cake_fmha_dcp_spec"]
    if addon.get("installed") is not True:
        raise RuntimeError("the authenticated Cake FMHA DCP add-on is not installed")
    families = addon["manifest"]["families"]
    try:
        return families[name]
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


def _validate_specialization(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> None:
    if variant not in ("v1", "v4"):
        raise ValueError(f"unsupported DCP speculative FMHA variant: {variant}")
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if q_len not in _SUPPORTED_Q_LENS:
        raise ValueError(f"q_len must be one of {_SUPPORTED_Q_LENS}, got {q_len}")
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(
            f"cp_world must be one of {_SUPPORTED_CP_WORLDS}, got {cp_world}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for GQA: "
            f"got {num_q_heads} and {num_kv_heads}"
        )
    group_ratio = num_q_heads // num_kv_heads
    if not 1 <= group_ratio <= 8:
        raise ValueError(f"head group ratio must be in [1, 8], got {group_ratio}")
    if variant == "v1" and route_param not in (0, 1):
        raise ValueError(f"v1 retain_kv_l2 must be 0 or 1, got {route_param}")
    if variant == "v4" and not 2 <= route_param <= 16:
        raise ValueError(f"v4 num_split must be in [2, 16], got {route_param}")


def get_dcp_spec_uri(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> str:
    _validate_specialization(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    )
    route_name = "retain" if variant == "v1" else "split"
    return (
        f"cake_fmha_dcp_spec_bf16_{variant}_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}"
        f"_cp{cp_world}_{route_name}{route_param}_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


def _validate_fp8_specialization(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> None:
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if q_len not in _FP8_SUPPORTED_Q_LENS:
        raise ValueError(f"q_len must be one of {_FP8_SUPPORTED_Q_LENS}, got {q_len}")
    if cp_world not in _SUPPORTED_CP_WORLDS:
        raise ValueError(
            f"cp_world must be one of {_SUPPORTED_CP_WORLDS}, got {cp_world}"
        )
    if num_q_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("num_q_heads and num_kv_heads must be positive")
    if num_q_heads % num_kv_heads != 0:
        raise ValueError(
            "num_q_heads must be divisible by num_kv_heads for GQA: "
            f"got {num_q_heads} and {num_kv_heads}"
        )
    group_ratio = num_q_heads // num_kv_heads
    if not 1 <= group_ratio <= 8:
        raise ValueError(f"head group ratio must be in [1, 8], got {group_ratio}")
    if not 1 <= num_split <= 4:
        raise ValueError(f"FP8 num_split must be in [1, 4], got {num_split}")
    if retain_kv_l2 not in (0, 1):
        raise ValueError(f"FP8 retain_kv_l2 must be 0 or 1, got {retain_kv_l2}")


def get_dcp_spec_fp8_uri(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> str:
    _validate_fp8_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    )
    return (
        f"cake_fmha_dcp_spec_bf16_fp8_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}_cp{cp_world}"
        f"_split{num_split}_retain{retain_kv_l2}_{CAKE_FMHA_MANIFEST_SHA256[:12]}_"
        f"{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


@functools.cache
def gen_dcp_spec_module(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
) -> JitSpec:
    """Generate one source-specialized, one-launch DCP speculative FMHA module."""

    uri = get_dcp_spec_uri(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    )
    selector = (
        {"retain_kv_l2": route_param} if variant == "v1" else {"num_split": route_param}
    )
    body, binding = _get_dcp_sources(f"dcp_spec_bf16_{variant}", target, selector)
    csrc_dir = get_cake_fmha_csrc_dir()

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCP_WORLD={cp_world}",
        ],
        extra_include_paths=[csrc_dir],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def gen_dcp_spec_fp8_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
) -> JitSpec:
    """Generate one BF16-Q/FP8-KV, HND-page64 Cake FMHA module."""

    uri = get_dcp_spec_fp8_uri(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    )
    body, binding = _get_dcp_sources(
        "dcp_spec_bf16_fp8",
        target,
        {"num_split": num_split, "retain_kv_l2": retain_kv_l2},
    )
    csrc_dir = get_cake_fmha_csrc_dir()

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCP_WORLD={cp_world}",
        ],
        extra_include_paths=[csrc_dir],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated FP8 DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def load_dcp_spec_module(
    variant: DcpSpecVariant,
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    route_param: int,
):
    module = gen_dcp_spec_module(
        variant,
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        route_param,
    ).build_and_load()
    logger.info(f"Loaded DCP speculative FMHA module: {module}")
    return module


@functools.cache
def load_dcp_spec_fp8_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
    retain_kv_l2: int,
):
    module = gen_dcp_spec_fp8_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
        retain_kv_l2,
    ).build_and_load()
    logger.info(f"Loaded FP8 DCP speculative FMHA module: {module}")
    return module


def _get_d256_csrc_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / "dcp"
    if installed.exists():
        return installed

    checkout = Path(__file__).resolve().parents[2] / "csrc" / "dcp"
    if checkout.exists():
        return checkout

    raise FileNotFoundError(
        "DCP speculative FMHA sources were not found. Checked:\n"
        f"  - {installed}\n"
        f"  - {checkout}"
    )


def _validate_fp8_d256_specialization(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
) -> None:
    if target not in _DCP_SPEC_NVCC_FLAGS:
        raise ValueError(f"unsupported DCP speculative FMHA target: {target}")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if q_len not in _FP8_D256_SUPPORTED_Q_LENS:
        raise ValueError(
            f"D256 q_len must be one of {_FP8_D256_SUPPORTED_Q_LENS}, got {q_len}"
        )
    if (num_q_heads, num_kv_heads) != (16, 1):
        raise ValueError(
            "D256 production specialization requires num_q_heads=16 and "
            f"num_kv_heads=1, got {num_q_heads} and {num_kv_heads}"
        )
    if cp_world not in (1, 4):
        raise ValueError(f"D256 cp_world must be 1 or 4, got {cp_world}")
    if num_split not in _FP8_D256_SUPPORTED_SPLITS:
        raise ValueError(
            f"D256 num_split must be one of {_FP8_D256_SUPPORTED_SPLITS}, "
            f"got {num_split}"
        )


def get_dcp_spec_fp8_d256_uri(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
) -> str:
    _validate_fp8_d256_specialization(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
    )
    return (
        f"cake_fmha_dcp_spec_bf16_fp8_d256_{target}"
        f"_b{batch_size}_q{q_len}_hq{num_q_heads}_hkv{num_kv_heads}_cp{cp_world}"
        f"_split{num_split}_retain0"
    )


@functools.cache
def gen_dcp_spec_fp8_d256_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
) -> JitSpec:
    """Generate one D256/ratio16 BF16-Q/FP8-KV Cake FMHA module."""

    uri = get_dcp_spec_fp8_d256_uri(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
    )
    csrc_dir = _get_d256_csrc_dir()
    body = csrc_dir / (f"cake_fmha_dcp_spec_bf16_fp8_d256_split{num_split}_retain0.cu")
    binding = csrc_dir / "cake_fmha_dcp_spec_bf16_fp8_d256_binding.cu"
    for source in (body, binding):
        if not source.exists():
            raise FileNotFoundError(
                f"D256 DCP speculative FMHA source not found: {source}"
            )

    spec = gen_jit_spec(
        name=uri,
        sources=[body, binding],
        extra_cuda_cflags=[
            *_DCP_SPEC_NVCC_FLAGS[target],
            f"-DBATCH_SIZE={batch_size}",
            f"-DQ_LEN={q_len}",
            f"-DNUM_Q_HEADS={num_q_heads}",
            f"-DNUM_KV_HEADS={num_kv_heads}",
            f"-DCP_WORLD={cp_world}",
        ],
        extra_include_paths=[csrc_dir],
        extra_ldflags=["-lcuda"],
    )
    logger.info(f"Generated D256 FP8 DCP speculative FMHA JIT spec: {spec.name}")
    return spec


@functools.cache
def load_dcp_spec_fp8_d256_module(
    target: DcpSpecTarget,
    batch_size: int,
    q_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    cp_world: int,
    num_split: int,
):
    module = gen_dcp_spec_fp8_d256_module(
        target,
        batch_size,
        q_len,
        num_q_heads,
        num_kv_heads,
        cp_world,
        num_split,
    ).build_and_load()
    logger.info(f"Loaded D256 FP8 DCP speculative FMHA module: {module}")
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
    return (
        f"cake_fmha_{family}_n{n_rows}{program_tag}_{target}"
        f"_{CAKE_FMHA_MANIFEST_SHA256[:12]}_{CAKE_FMHA_FLASHINFER_BINDINGS_SHA256[:12]}"
    )


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
    "DcpBalancedFamily",
    "dcp_balanced_program_variants",
    "gen_dcp_spec_balanced_module",
    "get_dcp_spec_balanced_uri",
    "load_dcp_spec_balanced_module",
    "gen_dcp_spec_fp8_d256_module",
    "get_dcp_spec_fp8_d256_uri",
    "load_dcp_spec_fp8_d256_module",
    "DcpSpecTarget",
    "DcpSpecVariant",
    "gen_dcp_spec_fp8_module",
    "gen_dcp_spec_module",
    "get_dcp_spec_fp8_uri",
    "get_dcp_spec_uri",
    "load_dcp_spec_fp8_module",
    "load_dcp_spec_module",
]
