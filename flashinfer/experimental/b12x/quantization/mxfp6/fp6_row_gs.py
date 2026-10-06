"""Per-row activation global-scale kernel for large-M FP6 quantization.

The serving per-row global-scale recipe (see ``fp6_dense_weights``) needs, for
every activation row, ``gs_r = numerator(fmt) / max(amax_r, 1e-6)`` plus the
bf16 output correction ``inv_gs_r = bf16(1/gs_r)`` and the GEMM epilogue
``alpha = 1/(1*w_gs)``. On the small-M decode path the whole recipe is fused
into :class:`~b12x.quantization.mxfp6.bf16_to_fp6_small_m.SmallMQuantKernel`
(every CTA redundantly rescans the tiny input). At prefill M (thousands of
rows) a redundant scan is unaffordable, so this kernel computes the row scales
in ONE bandwidth-bound pass (one CTA per row) and hands them to the TMA
quantizer's ``per_row`` mode, which applies the bf16 pre-scale in-registers.

The two-kernel formulation matches the host arithmetic bit-for-bit:

* the f32 max over exact ``|bf16|`` values is order-independent and therefore
  bit-identical to ``x.abs().amax(dim=1).float()``;
* ``div.rn.f32`` is correctly rounded, provably equal to the host chain's
  f64 divide + f32 cast (the 2p+2 double-rounding rule);
* ``inv_gs_r`` uses div.rn.f32 + cvt.rn.bf16.f32, bit-identical to
  ``(1.0 / gs.double()).float().to(torch.bfloat16)``;
* ``alpha`` uses div.rn.f32, bit-identical to ``torch.reciprocal`` (both
  correctly rounded) of ``1.0 * w_gs``.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
import torch
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import (
    block_reduce,
    cvt_f32_to_bf16_bits,
    div_rn_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_f32,
    st_global_u16,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.fp6 import mx_gs_numerator
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream

_THREADS = 128

# Fmt-aware global-scale numerator (e3m2: 12544.0, e2m3: 3360.0, e4m3:
# 200704.0 — all exactly representable in f32), resolved at trace time from
# ``self.fmt`` exactly like the small-M kernel.
_GS_NUMERATOR = {f: mx_gs_numerator(f) for f in ("e3m2", "e2m3", "e4m3")}

_KERNEL_CACHE: Dict[Tuple, object] = {}


class RowGsKernel:
    """One 128-thread CTA per activation row (grid = padded M).

    Vectorized 128-bit loads (``K % 128 == 0`` keeps every row start 16-byte
    aligned), warp + block max-reduce, then thread 0 stores ``gs_r`` (f32) and
    ``inv_gs_r`` (bf16). Thread 0 of CTA 0 additionally stores
    ``alpha = 1/(1*w_gs)`` for the GEMM epilogue. Zero padding rows reduce to
    amax 0 -> ``gs = numerator/1e-6`` (finite); their pre-scaled values stay
    exactly 0.0, matching the host chain's zero-padded quantizer input.
    """

    def __init__(self, fmt: str = "e4m3"):
        assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act fmt: {fmt}"
        self.fmt = fmt

    @cute.jit
    def __call__(
        self,
        bf16_input: cute.Tensor,
        w_gscale: cute.Tensor,
        gs_out: cute.Tensor,
        inv_gs_out: cute.Tensor,
        alpha_out: cute.Tensor,
        grid: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        self.kernel(bf16_input, w_gscale, gs_out, inv_gs_out, alpha_out).launch(
            grid=(grid, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mWgs: cute.Tensor,
        mGs: cute.Tensor,
        mInvGs: cute.Tensor,
        mAlpha: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        k = Int32(mX.shape[1])
        nvec_row = k // Int32(8)

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1, _THREADS // 32)),
            byte_alignment=16,
        )

        # Row amax: f32 max of the exact |bf16| values (each u32 word holds two
        # bf16; abs is the bits shifted high with the sign cleared). Max is
        # order-independent -> bit-identical to x.abs().amax(dim=1).float().
        # Row id * stride in Int64 per the 64-bit offset guideline.
        row_base = Int64(bidx) * Int64(k)
        local = cutlass.Float32(0.0)
        i = Int32(tidx)
        while i < nvec_row:
            w0, w1, w2, w3 = ld_global_v4_u32(
                get_ptr_as_int64(mX, row_base + Int64(i * Int32(8)))
            )
            for w in (w0, w1, w2, w3):
                hi = u32_as_f32(w & Uint32(0x7FFF0000))
                lo = u32_as_f32((w << Uint32(16)) & Uint32(0x7FFF0000))
                local = fmax_f32(local, fmax_f32(hi, lo))
            i += Int32(_THREADS)
        local = warp_reduce(local, fmax_f32)
        amax_r = block_reduce(local, fmax_f32, red_buf, cutlass.Float32(0.0))

        if tidx == Int32(0):
            # gs_r with a correctly rounded f32 division (div.rn.f32): equal to
            # the host chain's f64 divide + f32 cast (see module docstring).
            gs_r = div_rn_f32(
                cutlass.Float32(_GS_NUMERATOR[self.fmt]),
                fmax_f32(amax_r, cutlass.Float32(1e-6)),
            )
            st_global_f32(get_ptr_as_int64(mGs, bidx), gs_r)
            # inv_gs_r = bf16(1/gs_r): div.rn.f32 + cvt.rn.bf16.f32, matching
            # (1.0 / gs.double()).float().to(torch.bfloat16) bit-for-bit.
            st_global_u16(
                get_ptr_as_int64(mInvGs, bidx),
                cvt_f32_to_bf16_bits(div_rn_f32(cutlass.Float32(1.0), gs_r)),
            )
            if bidx == Int32(0):
                # alpha = 1/(1*w_gs): the per-row path quantizes with a UNIT
                # activation gs, so alpha only undoes the weight global scale.
                st_global_f32(
                    get_ptr_as_int64(mAlpha, Int32(0)),
                    div_rn_f32(
                        cutlass.Float32(1.0),
                        cutlass.Float32(mWgs[Int32(0)]),
                    ),
                )


def compile_fp6_row_gs(m: int, k: int, fmt: str = "e4m3"):
    """Compile the per-row global-scale kernel for a padded ``(m, k)`` input.

    Returns ``launch(bf16_input, w_gscale, gs_out, inv_gs_out, alpha_out)``
    where ``gs_out`` is f32 ``(m,)``, ``inv_gs_out`` bf16 ``(m,)`` and
    ``alpha_out`` f32 ``(1,)``. ``m`` is the PADDED row count (every row the
    TMA quantizer will process needs a scale); callers slice ``inv_gs_out`` to
    the true row count for the post-GEMM correction.
    """
    assert m >= 1, f"m must be >= 1, got {m}"
    assert k % 128 == 0, f"K must be a multiple of 128, got {k}"
    assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act fmt: {fmt}"
    cache_key = (m, k, fmt)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    bf16_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    wgs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (m,), assumed_align=4
    )
    inv_gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m,), assumed_align=2
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    kernel = RowGsKernel(fmt)
    raise_if_kernel_resolution_frozen(
        "cute.compile", target=kernel, cache_key=cache_key
    )
    raw = b12x_compile(
        kernel,
        bf16_fake,
        wgs_fake,
        gs_fake,
        inv_gs_fake,
        alpha_fake,
        m,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.fp6_row_gs",
            1,
            cache_key,
        ),
    )

    def launch(
        bf16_input: torch.Tensor,
        w_gscale: torch.Tensor,
        gs_out: torch.Tensor,
        inv_gs_out: torch.Tensor,
        alpha_out: torch.Tensor,
    ) -> None:
        expected = (
            ("bf16_input", bf16_input, (m, k), torch.bfloat16, 16),
            ("w_gscale", w_gscale, (1,), torch.float32, 4),
            ("gs_out", gs_out, (m,), torch.float32, 4),
            ("inv_gs_out", inv_gs_out, (m,), torch.bfloat16, 2),
            ("alpha_out", alpha_out, (1,), torch.float32, 4),
        )
        for name, tensor, shape, dtype, alignment in expected:
            if (
                tensor.shape != shape
                or tensor.dtype != dtype
                or tensor.device.type != "cuda"
                or not tensor.is_contiguous()
                or tensor.data_ptr() % alignment != 0
            ):
                raise ValueError(
                    f"{name} must be a contiguous CUDA {dtype} tensor with "
                    f"shape {shape} and {alignment}-byte alignment; got shape "
                    f"{tuple(tensor.shape)}, dtype {tensor.dtype}, device "
                    f"{tensor.device}, contiguous {tensor.is_contiguous()}, "
                    f"data_ptr % {alignment} = {tensor.data_ptr() % alignment}"
                )
        if any(tensor.device != bf16_input.device for _, tensor, *_ in expected[1:]):
            raise ValueError("all row-scale kernel tensors must be on one device")
        raw(
            bf16_input,
            w_gscale,
            gs_out,
            inv_gs_out,
            alpha_out,
            current_cuda_stream(),
        )

    launch = attach_programs(launch, raw)
    _KERNEL_CACHE[cache_key] = launch
    return launch
