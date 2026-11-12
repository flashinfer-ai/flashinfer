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

# JIT loaders of the MOSCA MegaMoE kernels (SM100 family).
#
# Nothing is prebuilt: no cubin, no shared library, no AOT registration
# (flashinfer/aot.py). Two kinds of tvm-ffi modules are compiled from source by
# FlashInfer's JIT on first use and cached in its JIT cache:
#
# * the host planner (``csrc/megamoe/sm100_fp8_fp4_planner.cu``), one per
#   process: the symmetric-buffer layout and the per-call launch configuration;
# * one kernel module per launch configuration
#   (``csrc/megamoe/sm100_fp8_fp4_launcher.cu`` plus a generated
#   ``mosca_megamoe_kernel_config.h`` that fixes the template arguments and the
#   MOSCA feature macros of the kernel header
#   ``csrc/megamoe/sm100_fp8_fp4_mega_moe.cuh``).
#
# Module names carry a digest of every source they compile (the kernel header
# and every file of the vendored ``deep_gemm`` tree included) and of the
# configuration, so a changed source or configuration never reuses a stale
# library.
#
# Compile flags reproduce upstream DeepJIT's command line for this kernel
# (../VENDOR.md): ``sm_100f``, C++20, ``--register-usage-level=10``, extended
# lambdas, and NO ``-use_fast_math`` (numerics are set by the kernel's
# ``kFastMath`` template argument, as upstream). FlashInfer's JIT adds -O3 and
# its usual defines.

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from ....jit import env as jit_env
from ....jit.core import gen_jit_spec, logger
from ....jit.utils import write_if_different
from .._vendor import CSRC_DIR, VENDOR_INCLUDE_DIR, vendored_tree

VENDORED_TREE = "deep_gemm"
FAMILY_DIR = CSRC_DIR / "megamoe"
PLANNER_SOURCE = FAMILY_DIR / "sm100_fp8_fp4_planner.cu"
LAUNCHER_SOURCE = FAMILY_DIR / "sm100_fp8_fp4_launcher.cu"
#: The MOSCA kernel (derived from upstream DeepGEMM; ../VENDOR.md).
KERNEL_HEADER = FAMILY_DIR / "sm100_fp8_fp4_mega_moe.cuh"
SHIM_HEADER = CSRC_DIR / "common" / "deep_gemm_host_shim.h"
CONFIG_HEADER = "mosca_megamoe_kernel_config.h"
DIGEST_HEX_DIGITS = 20
SUPPORTED_MAJOR = 10  # sm_100f: B200 / GB200 (10.0) and B300 / GB300 (10.3)

NVCC_FLAGS = [
    "-gencode=arch=compute_100f,code=sm_100f",
    "-std=c++20",
    "--ptxas-options=--register-usage-level=10",
    "--compiler-options=-fconcepts",
    "--expt-extended-lambda",
    "--diag-suppress=39,161,174,177,186,940",
    "--compiler-options=-Wno-deprecated-declarations,-Wno-abi",
]
INCLUDE_DIRS = [CSRC_DIR, VENDOR_INCLUDE_DIR]


def compiled_sources() -> list[Path]:
    """Every file the MegaMoE modules compile, in a stable order."""
    return [
        *vendored_tree(VENDORED_TREE).files,
        KERNEL_HEADER,
        SHIM_HEADER,
        PLANNER_SOURCE,
        LAUNCHER_SOURCE,
    ]


@functools.cache
def source_digest() -> str:
    """SHA-256 (20 hex digits) over every compiled source: path and bytes."""
    files = compiled_sources()
    if not all(f.is_file() for f in files):
        raise FileNotFoundError(
            "MOSCA MegaMoE sources are not installed under "
            f"{CSRC_DIR} (PyPI release wheels exclude experimental kernel "
            "sources; use a source install or a nightly wheel)"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.relative_to(CSRC_DIR).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:DIGEST_HEX_DIGITS]


def require_supported_device(device: Any = None) -> None:
    """Raise unless ``device`` is an SM100-family GPU (compute capability 10.x)."""
    import torch

    from ....utils import get_compute_capability

    resolved = torch.device("cuda") if device is None else torch.device(device)
    if resolved.index is None:
        resolved = torch.device("cuda", torch.cuda.current_device())
    major, minor = get_compute_capability(resolved)
    if major != SUPPORTED_MAJOR:
        raise RuntimeError(
            "MOSCA MegaMoE (experimental) runs on SM100-family GPUs "
            f"(compute capability 10.x, sm_100f); got {major}.{minor}"
        )


@functools.cache
def host_module() -> Any:
    """Build (once) and load the host planner module."""
    spec = gen_jit_spec(
        name=f"mosca_megamoe_host_{source_digest()}",
        sources=[PLANNER_SOURCE],
        extra_cuda_cflags=list(NVCC_FLAGS),
        extra_include_paths=list(INCLUDE_DIRS),
        use_fast_math=False,
    )
    logger.info(f"Loading MOSCA MegaMoE host planner: {spec.name}")
    return spec.build_and_load()


@dataclasses.dataclass(frozen=True)
class KernelParams:
    """Every compile-time argument of one kernel module."""

    num_max_tokens_per_rank: int
    hidden: int
    intermediate_hidden: int
    num_experts: int
    num_topk: int
    num_ranks: int
    num_sms: int
    block_m: int
    block_n: int
    block_k: int
    store_block_m: int
    sf_block_m: int
    sf_block_n: int
    gran_k: int
    num_ring_tokens: int
    num_sf_ring_tokens: int
    swizzle_acts_mode: int
    swizzle_weights_mode: int
    num_stages: int
    smem_size: int
    num_dispatch_threads: int
    num_non_epilogue_threads: int
    num_epilogue_threads: int
    num_bytes_per_pull: int
    activation_clamp: float
    fast_math: bool
    small_t: bool
    sparse_count: bool
    gather: bool

    def module_name(self) -> str:
        payload = json.dumps(
            {"sources": source_digest(), "params": dataclasses.asdict(self)},
            sort_keys=True,
            separators=(",", ":"),
        )
        return (
            "mosca_megamoe_"
            + hashlib.sha256(payload.encode()).hexdigest()[:DIGEST_HEX_DIGITS]
        )


#: MOSCA feature macros of the kernel header, keyed by the
#: ``KernelParams`` switch that turns them on (catalog.py lists the features).
FEATURE_MACROS = {
    "small_t": "DGX_SMALL_T",
    "sparse_count": "DGX_SPARSE_COUNT",
    "gather": "DGX_BCAST",
}


def _float_literal(value: float) -> str:
    # Upstream renders the clamp as a hexadecimal float literal. "No clamp" is
    # +inf upstream, which this backend cannot spell: the launcher instantiates
    # the kernel in host code, and nvcc's host stub prints a float template
    # argument of +inf as the undeclared identifier `inf`. The config therefore
    # requires a finite clamp.
    if value is None or not math.isfinite(value):
        raise ValueError(
            f"MOSCA MegaMoE needs a finite activation clamp, got {value!r}"
        )
    return f"{float(value).hex()}f"


def render_config_header(p: KernelParams) -> str:
    defines = {
        "MOSCA_NUM_MAX_TOKENS_PER_RANK": p.num_max_tokens_per_rank,
        "MOSCA_HIDDEN": p.hidden,
        "MOSCA_INTERMEDIATE_HIDDEN": p.intermediate_hidden,
        "MOSCA_NUM_EXPERTS": p.num_experts,
        "MOSCA_NUM_TOPK": p.num_topk,
        "MOSCA_NUM_RANKS": p.num_ranks,
        "MOSCA_NUM_SMS": p.num_sms,
        "MOSCA_BLOCK_M": p.block_m,
        "MOSCA_BLOCK_N": p.block_n,
        "MOSCA_BLOCK_K": p.block_k,
        "MOSCA_STORE_BLOCK_M": p.store_block_m,
        "MOSCA_SF_BLOCK_M": p.sf_block_m,
        "MOSCA_SF_BLOCK_N": p.sf_block_n,
        "MOSCA_GRAN_K": p.gran_k,
        "MOSCA_NUM_RING_TOKENS": p.num_ring_tokens,
        "MOSCA_NUM_SF_RING_TOKENS": p.num_sf_ring_tokens,
        "MOSCA_SWIZZLE_ACTS_MODE": p.swizzle_acts_mode,
        "MOSCA_SWIZZLE_WEIGHTS_MODE": p.swizzle_weights_mode,
        "MOSCA_NUM_STAGES": p.num_stages,
        "MOSCA_SMEM_SIZE": p.smem_size,
        "MOSCA_NUM_DISPATCH_THREADS": p.num_dispatch_threads,
        "MOSCA_NUM_NON_EPILOGUE_THREADS": p.num_non_epilogue_threads,
        "MOSCA_NUM_EPILOGUE_THREADS": p.num_epilogue_threads,
        "MOSCA_NUM_BYTES_PER_PULL": p.num_bytes_per_pull,
        "MOSCA_ACTIVATION_CLAMP": _float_literal(p.activation_clamp),
        "MOSCA_FAST_MATH": "true" if p.fast_math else "false",
    }
    lines = [
        "// Generated by flashinfer.experimental.mosca.megamoe.jit; do not edit.",
        "#pragma once",
    ]
    lines += [f"#define {name} {value}" for name, value in defines.items()]
    # MOSCA feature macros, read by the kernel header.
    lines += [
        f"#define {macro}"
        for switch, macro in FEATURE_MACROS.items()
        if getattr(p, switch)
    ]
    return "\n".join(lines) + "\n"


_LOADED: dict[KernelParams, Any] = {}


def kernel_spec(params: KernelParams) -> Any:
    """The JIT spec of one kernel module (writes its generated config header)."""
    name = params.module_name()
    gen_dir = jit_env.FLASHINFER_GEN_SRC_DIR / name
    write_if_different(gen_dir / CONFIG_HEADER, render_config_header(params))
    return gen_jit_spec(
        name=name,
        sources=[LAUNCHER_SOURCE],
        extra_cuda_cflags=list(NVCC_FLAGS),
        extra_ldflags=["-lcuda"],
        extra_include_paths=[gen_dir, *INCLUDE_DIRS],
        use_fast_math=False,
    )


def is_loaded(params: KernelParams) -> bool:
    """Whether the kernel module of ``params`` is loaded in this process."""
    return params in _LOADED


def kernel_module(params: KernelParams) -> Any:
    """Build (once per configuration) and load the kernel module for ``params``.

    After loading, the module must describe exactly ``params`` and the device
    must hold every CTA of the persistent grid at once (``num_sms / 2``
    resident 2-CTA clusters); otherwise the kernel would wait forever for CTAs
    that never start, so this raises instead.
    """
    module = _LOADED.get(params)
    if module is not None:
        return module
    spec = kernel_spec(params)
    logger.info(f"Loading MOSCA MegaMoE kernel module {spec.name}")
    module = spec.build_and_load()
    described = tuple(int(v) for v in module.describe())
    expected = (
        params.num_max_tokens_per_rank,
        params.hidden,
        params.intermediate_hidden,
        params.num_experts,
        params.num_topk,
        params.num_ranks,
        params.num_sms,
        params.block_m,
        params.store_block_m,
        params.num_stages,
        params.smem_size,
        int(params.gather),
        int(params.small_t),
        int(params.sparse_count),
    )
    if described != expected:
        raise RuntimeError(
            f"MOSCA MegaMoE module {spec.name} describes {described}, expected {expected}"
        )
    clusters = int(module.max_active_clusters())
    if 2 * clusters < params.num_sms:
        raise RuntimeError(
            f"MOSCA MegaMoE module {spec.name}: the device holds {clusters} resident "
            f"2-CTA clusters of this configuration, but the persistent grid needs "
            f"{params.num_sms // 2} (num_sms={params.num_sms}); lower num_sms"
        )
    _LOADED[params] = module
    return module


def build_kernel_modules(
    params: list[KernelParams], shard: int = 0, num_shards: int = 1
) -> None:
    """Compile (without loading) every ``params[i]`` with ``i % num_shards == shard``.

    Each module builds under its own lock in its own build directory, which is
    where ``kernel_module`` later finds it up to date. EP ranks pass their rank
    as the shard so the builds run in parallel across ranks; FlashInfer's
    combined ``build_jit_specs`` is not used, because it keeps a separate ninja
    dependency log and relinks modules that other processes are loading.
    """
    pending = [p for p in dict.fromkeys(params) if p not in _LOADED]
    for index, p in enumerate(pending):
        if index % num_shards == shard:
            kernel_spec(p).build(need_lock=True)


__all__ = [
    "FEATURE_MACROS",
    "KernelParams",
    "NVCC_FLAGS",
    "build_kernel_modules",
    "compiled_sources",
    "host_module",
    "is_loaded",
    "kernel_module",
    "kernel_spec",
    "render_config_header",
    "require_supported_device",
    "source_digest",
]
