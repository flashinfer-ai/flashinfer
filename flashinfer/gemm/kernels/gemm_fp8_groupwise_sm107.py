# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM107 (Rubin) FP8 groupwise GEMM on the CuTe-DSL ``SM107BlockwiseGemmKernel``.

Split-K reuses the kernel's batch mode: K is cut into ``S`` slices that are passed as
``S`` batch entries through strided views (no copies); the FP32 partials are then summed.
Only ``scale_major_mode == "MN"`` is supported.
"""

import functools
import math
from typing import Callable, NamedTuple, Optional, Tuple

import torch

_SCALE_BLOCK = 128


class SM107BlockwiseTactic(NamedTuple):
    """(use_2cta, split_k); use_2cta selects the 256x128 tile on a (2, 1) cluster."""

    use_2cta: bool
    split_k: int


def _cfg(use_2cta: bool):
    if use_2cta:
        return (256, 128, 128), (256, 128, 64), (2, 1)
    return (128, 128, 128), (128, 128, 64), (1, 1)


@functools.cache
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _kernel_disk_name(
    use_2cta: bool, c_dtype_name: str, max_active_clusters: int
) -> str:
    """On-disk kernel name; must cover every codegen argument of ``_get_compiled``."""
    return (
        f"blockwise_{2 if use_2cta else 1}cta_{c_dtype_name}_mac{max_active_clusters}"
    )


def _cache_key_files() -> Tuple[str, ...]:
    """Source files whose content invalidates the on-disk kernels."""
    from . import blockwise_gemm_blackwell, blockwise_gemm_rubin

    return (
        __file__,
        blockwise_gemm_blackwell.__file__,
        blockwise_gemm_rubin.__file__,
    )


@functools.cache
def _get_compiled(use_2cta: bool, out_is_fp32: bool, out_dtype_name: str) -> Callable:
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils

    from flashinfer.cute_dsl.utils import is_rubin_cute_dsl_available
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    if not is_rubin_cute_dsl_available():
        raise NotImplementedError(
            "The SM107 CuTe-DSL FP8 groupwise GEMM requires CuTe DSL >= 4.8 "
            "(cutlass.utils.rubin_helpers)."
        )
    from .blockwise_gemm_rubin import SM107BlockwiseGemmKernel

    mma_tiler, mma_inst, cluster = _cfg(use_2cta)
    c_dtype = (
        cutlass.Float32
        if out_is_fp32
        else {"bfloat16": cutlass.BFloat16, "float16": cutlass.Float16}[out_dtype_name]
    )
    sym_m, sym_n, sym_k, sym_l = (cute.sym_int() for _ in range(4))
    sym_kb, sym_nb = cute.sym_int(), cute.sym_int()
    mk = cute.runtime.make_fake_compact_tensor
    a_f = mk(
        cutlass.Float8E4M3FN,
        (sym_m, sym_k, sym_l),
        stride_order=(2, 0, 1),
        assumed_align=16,
    )
    b_f = mk(
        cutlass.Float8E4M3FN,
        (sym_n, sym_k, sym_l),
        stride_order=(2, 0, 1),
        assumed_align=16,
    )
    c_f = mk(c_dtype, (sym_m, sym_n, sym_l), stride_order=(1, 0, 2), assumed_align=16)
    sfa_f = mk(
        cutlass.Float32,
        (sym_m, sym_kb, sym_l),
        stride_order=(0, 1, 2),
        assumed_align=16,
    )
    sfb_f = mk(
        cutlass.Float32,
        (sym_nb, sym_kb, sym_l),
        stride_order=(0, 1, 2),
        assumed_align=4,
    )
    stream_f = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    gemm = SM107BlockwiseGemmKernel(
        cutlass.Float32, use_2cta, mma_tiler, mma_inst, cluster
    )
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        cluster[0] * cluster[1]
    )
    # max_active_clusters is baked into the kernel, so it is part of the disk name.
    return build_and_load_cute_dsl_kernel(
        "gemm_fp8_groupwise_sm107",
        _kernel_disk_name(
            use_2cta, "float32" if out_is_fp32 else out_dtype_name, max_active_clusters
        ),
        lambda: cute.compile(
            gemm,
            a_f,
            b_f,
            c_f,
            sfa_f,
            sfb_f,
            max_active_clusters,
            stream_f,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=_cache_key_files(),
    )


# Relative wave cost model (unit: one 128-wide K block of one tile) for the split-K /
# 2-CTA choice, fitted to tactic sweeps.
_C_TILE = 4.3478
_C_RED2_FIXED = 17.391
_C_REDN_FIXED = 34.783
_C_RED_BYTES = 1.84e6
_MAX_SPLIT_K = 8


def _estimate_cost(
    m: int, n: int, k: int, tactic: SM107BlockwiseTactic, num_sms: int
) -> float:
    kb = k // _SCALE_BLOCK
    s = tactic.split_k
    tile_m = 256 if tactic.use_2cta else 128
    # in SM units: a 2-CTA cluster tile occupies two SMs
    units = math.ceil(m / tile_m) * math.ceil(n / 128) * (2 if tactic.use_2cta else 1)
    waves = math.ceil(units * s / num_sms)
    t = waves * (kb / s + _C_TILE)
    if s == 2:
        t += _C_RED2_FIXED + m * n * 10 / _C_RED_BYTES
    elif s > 2:
        t += _C_REDN_FIXED + m * n * (4 * s + 6) / _C_RED_BYTES
    return t


def valid_tactics(m: int, n: int, k: int) -> Tuple[SM107BlockwiseTactic, ...]:
    kb = k // _SCALE_BLOCK
    out = []
    # 2-CTA first for M >= 256 so that it wins model ties there.
    for use_2cta in (True, False) if m >= 256 else (False,):
        for s in range(1, _MAX_SPLIT_K + 1):
            if kb % s == 0 and (s == 1 or kb // s >= 2):
                out.append(SM107BlockwiseTactic(use_2cta, s))
    return tuple(out)


def default_tactic(
    m: int, n: int, k: int, device: torch.device
) -> SM107BlockwiseTactic:
    sms = _num_sms(
        device.index if device.index is not None else torch.cuda.current_device()
    )
    return min(valid_tactics(m, n, k), key=lambda t: _estimate_cost(m, n, k, t, sms))


def check_problem(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    scale_major_mode: Optional[str],
    scale_granularity_mnk: Tuple[int, int, int],
    out_dtype: torch.dtype,
) -> None:
    m, k = a.shape
    n = b.shape[0]
    if scale_major_mode != "MN":
        raise ValueError(
            "The cute-dsl SM107 backend supports scale_major_mode='MN' only."
        )
    if tuple(scale_granularity_mnk) != (1, 128, 128):
        raise ValueError(
            "The cute-dsl SM107 backend supports scale_granularity_mnk=(1, 128, 128) only."
        )
    if a.dtype != torch.float8_e4m3fn or b.dtype != torch.float8_e4m3fn:
        raise ValueError(
            "The cute-dsl SM107 backend supports float8_e4m3fn inputs only."
        )
    if out_dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("The cute-dsl SM107 backend supports bf16/fp16 output only.")
    if k % _SCALE_BLOCK or n % _SCALE_BLOCK:
        raise ValueError(
            "The cute-dsl SM107 backend requires k and n to be multiples of 128."
        )
    if m % 4:
        raise ValueError("The cute-dsl SM107 backend requires m to be a multiple of 4.")
    if not (a.is_contiguous() and b.is_contiguous()):
        raise ValueError("a and b must be contiguous (K-major).")
    kb, nb = k // _SCALE_BLOCK, n // _SCALE_BLOCK
    if tuple(a_scale.shape) != (kb, m) or not a_scale.is_contiguous():
        raise ValueError(f"a_scale must be a contiguous ({kb}, {m}) tensor in MN mode.")
    if tuple(b_scale.shape) != (kb, nb) or not b_scale.is_contiguous():
        raise ValueError(
            f"b_scale must be a contiguous ({kb}, {nb}) tensor in MN mode."
        )
    if a_scale.dtype != torch.float32 or b_scale.dtype != torch.float32:
        raise ValueError("a_scale / b_scale must be float32.")


def gemm_fp8_nt_groupwise_sm107(
    a: torch.Tensor,
    b: torch.Tensor,
    a_scale: torch.Tensor,
    b_scale: torch.Tensor,
    out: torch.Tensor,
    tactic: Optional[SM107BlockwiseTactic] = None,
) -> torch.Tensor:
    """``a`` (m, k), ``b`` (n, k) fp8; MN-major fp32 scales; ``out`` (m, n) bf16/fp16."""
    m, k = a.shape
    n = b.shape[0]
    if tactic is None:
        tactic = default_tactic(m, n, k, a.device)
    use_2cta, s = tactic
    kb, nb = k // _SCALE_BLOCK, n // _SCALE_BLOCK
    if kb % s:
        raise ValueError(f"split_k={s} must divide k/128={kb}")
    ks, kbs = k // s, kb // s
    a3 = a.as_strided((m, ks, s), (k, 1, ks))
    b3 = b.as_strided((n, ks, s), (k, 1, ks))
    sfa3 = a_scale.as_strided((m, kbs, s), (1, m, m * kbs))
    sfb3 = b_scale.as_strided((nb, kbs, s), (1, nb, nb * kbs))
    if s == 1:
        c3 = out.as_strided((m, n, 1), (n, 1, m * n))
        _get_compiled(use_2cta, False, str(out.dtype).split(".")[-1])(
            a3, b3, c3, sfa3, sfb3
        )
        return out
    partial = torch.empty((s, m, n), dtype=torch.float32, device=a.device)
    c3 = partial.permute(1, 2, 0)
    _get_compiled(use_2cta, True, "float32")(a3, b3, c3, sfa3, sfb3)
    if s == 2:
        torch.add(partial[0], partial[1], out=out)  # fp32 add, one cast on store
    else:
        out.copy_(partial.sum(dim=0))
    return out
