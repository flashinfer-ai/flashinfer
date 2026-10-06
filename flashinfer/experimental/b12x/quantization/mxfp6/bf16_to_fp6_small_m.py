"""BF16 -> MX-FP6 quantizer specialized for small M (decode/MTP activations).

The TMA quantizer (:mod:`b12x.quantization.mxfp6.bf16_to_fp6_tma`) tiles
M in 128, so an M=1 decode activation pays for 128 quantized rows (~99% wasted
work: ~29 us per linear, larger than the decode GEMM itself). This kernel
quantizes ONLY the real rows — one thread per 32-element block, plain global
loads, no TMA/smem — and writes the exact same outputs for those rows:

* byte-container codes (one FP6 code per byte, the ``mxf8f6f4`` GEMM operand
  layout) at the same ``(row, K)`` offsets, and
* UE8M0 scale bytes at the same swizzled blockscaled-SFA offsets,

using the *same* ``quantize_block_fp6_*_bytes`` math and the same store
helpers, so results are bit-identical to the TMA path for every written row.
Rows ``m..127`` of the padded buffers stay unwritten — the GEMM runs at the
true ``m`` and never depends on them (same contract as the uninitialized
``x_pad`` rows on the large-M path).

The ENTIRE activation global-scale pipeline is fused in as well: EVERY CTA
first computes the bf16 amax of the whole input itself (vectorized 128-bit
loads; abs-max of bf16 is order-independent and therefore bit-identical to
``torch.linalg.vector_norm(x, inf)`` and identical across CTAs), block-reduces
it, then computes ``gs = numerator(fmt) / max(amax, 1e-6)`` per-thread (the
fmt-aware numerator from :func:`b12x._lib.fp6.mx_gs_numerator`); thread 0
of CTA 0 emits ``alpha = 1 / (gs * w_gscale)`` for the GEMM epilogue. That
removes the separate ``vector_norm`` reduce kernel (~1 ms/step at 256
linears/step in the serving profile) plus the f32-convert/clamp/div/mul/
reciprocal launches from the decode hot path. All divisions use div.rn.f32
(correctly rounded); the host-side chain divides in f64 and casts to f32 —
provably the same bits — so codes/scales/alpha stay bit-identical. (A raw
torch f32 scalar/tensor divide is NOT always correctly rounded on CUDA.)

The amax scan is deliberately REDUNDANT per CTA rather than single-CTA or
cross-CTA synced: at small-M sizes the scan is a sub-microsecond L2 read
(<=442 KB) while quantization wants the full grid (up to 54 CTAs at m=16,
K=13824). A single-CTA variant was tried and cost ~1.2 ms/step in serving by
serializing the MTP-verify quants — redundancy is the cheaper trade by far.
"""
from __future__ import annotations

from typing import Dict, Tuple

import cutlass
import cutlass.cute as cute
import cuda.bindings.driver as cuda
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int32, Int64, Uint32

from b12x._lib.compiler import KernelCompileSpec, compile as b12x_compile
from b12x._lib.compile_plan import attach_programs
from b12x._lib.intrinsics import (
    block_reduce,
    cvt_f32_to_bf16_bits,
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    get_ptr_as_int64,
    ld_global_v4_u32,
    st_global_f32,
    st_global_u8,
    st_global_u16,
    u32_as_f32,
    warp_reduce,
)
from b12x._lib.fp6 import (
    mx_gs_numerator,
    quantize_block_fp6_e2m3_bytes,
    quantize_block_fp6_e3m2_bytes,
    quantize_block_fp8_e4m3_bytes,
)
from b12x._lib.runtime_control import raise_if_kernel_resolution_frozen
from b12x._lib.utils import current_cuda_stream, make_ptr
from .bf16_to_fp6_tma import moe_mxfp6_store_bytes_u64_global

_FP6_BLOCK_ELEMS = 32
_THREADS = 128

# Numerator of the activation global scale, per sub-format: maps amax into the
# format's own range exactly like the host-side recipe (e3m2: 12544.0,
# e2m3: 3360.0 — both exactly representable in f32). Resolved at kernel trace
# time from ``self.fmt``, so codes/scales/alpha stay bit-identical to the host
# path in b12x.quantization.mxfp6.fp6_dense_weights.
_GS_NUMERATOR = {f: mx_gs_numerator(f) for f in ("e3m2", "e2m3", "e4m3")}

#: Largest M routed to this kernel (decode is 1, MTP verify a handful). Above
#: this the 128-row TMA quantizer amortizes fine and its grid parallelism wins.
SMALL_M_MAX = 16

_KERNEL_CACHE: Dict[Tuple, object] = {}


class SmallMQuantKernel:
    """One thread per 32-element block over the ``m * K/32`` real blocks.

    ``per_row=True`` fuses the ENTIRE per-row activation global-scale recipe
    from ``fp6_dense_weights`` (the serving decode path) in-kernel:

    * per-row bf16 amax (order-independent max -> bit-identical to
      ``x.abs().amax(dim=1)``),
    * ``gs_r = numerator(fmt) / max(amax_r, 1e-6)`` in f32,
    * the pre-scale ``bf16(f32(x) * gs_r)`` per element (cvt.rn.bf16.f32 is
      bit-identical to ``(x.float() * gs).to(torch.bfloat16)``),
    * quantization with a UNIT global scale and ``alpha = 1/(1*w_gs)``
      (the pre-scaled row amax is exactly the fmt numerator, so the
      per-tensor gs the host chain's kernel re-derived collapses to exactly
      1.0 — same argument as the host-path comment in fp6_dense_weights),
    * the per-row output correction ``inv_gs_r = bf16(1/gs_r)`` written to
      ``mInvGs`` for the caller's post-GEMM ``result.mul_``.

    This replaces the ~12 eager torch launches per linear (abs/amax/clamp/
    div/mul/cast/fill/copy) that the host-side per-row chain cost on the
    decode hot path (~8 ms/step at 27B serving scale) with zero extra
    launches, while keeping every written bit identical.
    """

    def __init__(self, fmt: str = "e3m2", per_row: bool = False):
        assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act fmt: {fmt}"
        self.fmt = fmt
        self.per_row = per_row

    @cute.jit
    def __call__(
        self,
        bf16_input: cute.Tensor,
        w_gscale: cute.Tensor,
        codes_flat: cute.Tensor,
        sfa_ptr: cute.Pointer,
        alpha_out: cute.Tensor,
        inv_gs_out: cute.Tensor,
        grid: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        # Flat uint8 view over the swizzled SFA buffer; one 512-byte group per
        # 128-wide K-tile (m-tile index is always 0 here since m <= 16 < 128).
        sf_size = (bf16_input.shape[1] // 128) * 512
        sfa_flat = cute.make_tensor(sfa_ptr, cute.make_layout(sf_size))
        # Full quant grid; each CTA redundantly re-derives the (identical)
        # amax in its prologue, so no cross-CTA synchronization is needed.
        self.kernel(
            bf16_input, codes_flat, sfa_flat, w_gscale, alpha_out, inv_gs_out
        ).launch(
            grid=(grid, 1, 1),
            block=[_THREADS, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mCodes: cute.Tensor,
        mSFA: cute.Tensor,
        mWgs: cute.Tensor,
        mAlpha: cute.Tensor,
        mInvGs: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        m_c = mX.shape[0]  # static at trace time (compiled per (m, k))
        m = Int32(m_c)
        k = Int32(mX.shape[1])
        blocks_per_row = k // Int32(_FP6_BLOCK_ELEMS)
        total = m * blocks_per_row

        smem = cutlass.utils.SmemAllocator()
        red_buf = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=cute.make_layout((1, _THREADS // 32)),
            byte_alignment=16,
        )

        gs_value = cutlass.Float32(1.0)
        gs_pr = cute.make_rmem_tensor((m_c,), cutlass.Float32)
        if cutlass.const_expr(self.per_row):
            # Phase 0 (per-row mode): per-CTA amax of EACH row (every CTA
            # computes the same values — see module docstring). Same
            # vectorized 128-bit loads as the per-tensor scan; k % 128 == 0
            # keeps every row start 16-byte aligned. f32 max over exact
            # |bf16| values is bit-identical to x.abs().amax(dim=1).float().
            nvec_row = k // Int32(8)
            for r in cutlass.range_constexpr(m_c):
                if cutlass.const_expr(r > 0):
                    # red_buf is reused across rows: all warps must finish
                    # reading row r-1's partials before row r overwrites them.
                    cute.arch.barrier()
                local = cutlass.Float32(0.0)
                i = Int32(tidx)
                while i < nvec_row:
                    w0, w1, w2, w3 = ld_global_v4_u32(
                        get_ptr_as_int64(
                            mX, Int32(r) * k + i * Int32(8)
                        )
                    )
                    for w in (w0, w1, w2, w3):
                        hi = u32_as_f32(w & Uint32(0x7FFF0000))
                        lo = u32_as_f32((w << Uint32(16)) & Uint32(0x7FFF0000))
                        local = fmax_f32(local, fmax_f32(hi, lo))
                    i += Int32(_THREADS)
                local = warp_reduce(local, fmax_f32)
                amax_r = block_reduce(
                    local, fmax_f32, red_buf, cutlass.Float32(0.0)
                )
                # gs_r = numerator(fmt) / max(amax_r, 1e-6) with a correctly
                # rounded f32 division (div.rn.f32, not the DSL's ``/`` which
                # may lower to an approximate division). NOTE: torch's CUDA
                # f32 scalar/tensor division is itself NOT always correctly
                # rounded (200704/2.625 lands 1 ulp high), so the host chain
                # in fp6_dense_weights divides in f64 and casts — provably
                # equal to div.rn.f32 — to stay bit-identical to this kernel.
                gs_pr[r] = div_rn_f32(
                    cutlass.Float32(_GS_NUMERATOR[self.fmt]),
                    fmax_f32(amax_r, cutlass.Float32(1e-6)),
                )
            # Per-row output correction for the caller's post-GEMM multiply:
            # inv_gs_r = bf16(1/gs_r), bit-identical to the host chain's
            # (1.0 / a_gs_pr).to(torch.bfloat16) (div.rn.f32 + cvt.rn.bf16).
            if bidx == Int32(0):
                for r in cutlass.range_constexpr(m_c):
                    if tidx == Int32(r):
                        inv_r = div_rn_f32(cutlass.Float32(1.0), gs_pr[r])
                        st_global_u16(
                            get_ptr_as_int64(mInvGs, Int32(r)),
                            cvt_f32_to_bf16_bits(inv_r),
                        )
            # Quantization below uses the pre-scaled values with a UNIT gs
            # (gs_value stays 1.0): the pre-scaled row amax is exactly the
            # fmt numerator, so the host chain's re-derived per-tensor gs is
            # exactly 1.0 as well — bit-identical codes/scales/alpha.
        else:
            # Phase 0 (per-tensor mode): per-CTA amax over all m*K bf16
            # values (every CTA computes the same value — see module
            # docstring), 128-bit loads (k % 128 == 0 so m*K % 8 == 0 and x
            # is 16-byte aligned). abs of a bf16 in f32 is just its bits
            # shifted high with the sign cleared, so each u32 word costs two
            # LOP3s + two fmax. max is order-independent -> bit-identical to
            # torch.linalg.vector_norm(x, inf).
            nvec = (m * k) // Int32(8)
            local = cutlass.Float32(0.0)
            i = Int32(tidx)
            while i < nvec:
                w0, w1, w2, w3 = ld_global_v4_u32(
                    get_ptr_as_int64(mX, i * Int32(8))
                )
                for w in (w0, w1, w2, w3):
                    hi = u32_as_f32(w & Uint32(0x7FFF0000))
                    lo = u32_as_f32((w << Uint32(16)) & Uint32(0x7FFF0000))
                    local = fmax_f32(local, fmax_f32(hi, lo))
                i += Int32(_THREADS)
            local = warp_reduce(local, fmax_f32)
            amax_val = block_reduce(
                local, fmax_f32, red_buf, cutlass.Float32(0.0)
            )

            # Activation global scale, fused: gs = numerator(fmt) /
            # max(amax, 1e-6) with a correctly rounded f32 division
            # (div.rn, not the DSL's ``/``). The host-side A/B recipe
            # divides in f64 and casts to f32, which is bit-identical to
            # div.rn.f32 (see the per-row branch comment).
            amax_f32 = fmax_f32(amax_val, cutlass.Float32(1e-6))
            gs_value = div_rn_f32(
                cutlass.Float32(_GS_NUMERATOR[self.fmt]), amax_f32
            )

        # Phase 1: one 32-element block per thread across the full grid —
        # same parallelism as the pre-fusion kernel. Same per-block math
        # (load order, f32 convert, sequential fmax-of-fabs) as the TMA
        # kernel -> bit-identical codes/scales.
        idx = bidx * Int32(_THREADS) + tidx
        if idx == Int32(0):
            # alpha = 1/(a_gs * w_gs) for the GEMM epilogue; div.rn.f32
            # matches torch.reciprocal (both correctly rounded). In per-row
            # mode gs_value is the unit scale (1.0) — alpha = 1/(1*w_gs),
            # same as the host chain's torch.reciprocal(gs_unit * w_gs).
            alpha = div_rn_f32(
                cutlass.Float32(1.0),
                gs_value * cutlass.Float32(mWgs[Int32(0)]),
            )
            st_global_f32(get_ptr_as_int64(mAlpha, Int32(0)), alpha)
        if idx < total:
            row = idx // blocks_per_row
            blk = idx % blocks_per_row
            col0 = blk * Int32(_FP6_BLOCK_ELEMS)
            vals = cute.make_rmem_tensor((_FP6_BLOCK_ELEMS,), cutlass.Float32)
            bmax = cutlass.Float32(0.0)
            if cutlass.const_expr(self.per_row):
                # Select this block's row scale (m <= 16, fully unrolled).
                gs_row = cutlass.Float32(0.0)
                for r in cutlass.range_constexpr(m_c):
                    if row == Int32(r):
                        gs_row = gs_pr[r]
                for e in cutlass.range_constexpr(_FP6_BLOCK_ELEMS):
                    # Pre-scale with the SAME rounding chain as the host:
                    # f32 multiply (RN), round through bf16 (RN), widen back
                    # (exact). The quantizer below then sees the identical
                    # bf16 values the host chain handed it.
                    v_raw = cutlass.Float32(mX[row, col0 + Int32(e)])
                    pre_bits = cvt_f32_to_bf16_bits(v_raw * gs_row)
                    v = u32_as_f32(pre_bits << Uint32(16))
                    vals[e] = v
                    bmax = fmax_f32(bmax, fabs_f32(v))
            else:
                for e in cutlass.range_constexpr(_FP6_BLOCK_ELEMS):
                    v = cutlass.Float32(mX[row, col0 + Int32(e)])
                    vals[e] = v
                    bmax = fmax_f32(bmax, fabs_f32(v))
            if cutlass.const_expr(self.fmt == "e4m3"):
                q0, q1, q2, q3, sbyte = quantize_block_fp8_e4m3_bytes(
                    vals, bmax, gs_value
                )
            elif cutlass.const_expr(self.fmt == "e2m3"):
                q0, q1, q2, q3, sbyte = quantize_block_fp6_e2m3_bytes(
                    vals, bmax, gs_value
                )
            else:
                q0, q1, q2, q3, sbyte = quantize_block_fp6_e3m2_bytes(
                    vals, bmax, gs_value
                )
            # Byte-container row stride is the full K width (one byte per code).
            # Bounded by m<=16 today, but row*stride offsets stay Int64 per the
            # coding guideline so the width is safe if the small-M cap grows.
            byte_offset = Int64(row) * Int64(k) + Int64(col0)
            moe_mxfp6_store_bytes_u64_global(mCodes, byte_offset, q0, q1, q2, q3)
            # Swizzled blockscaled-SFA offset, identical to the TMA kernel with
            # mt == 0:  kt*512 + (row%32)*16 + (row//32)*4 + sf_block.
            kt = col0 // Int32(128)
            sf_block = (col0 % Int32(128)) // Int32(_FP6_BLOCK_ELEMS)
            sf_offset = (
                Int64(kt) * Int64(512)
                + Int64((row % Int32(32)) * Int32(16))
                + Int64((row // Int32(32)) * Int32(4))
                + Int64(sf_block)
            )
            st_global_u8(get_ptr_as_int64(mSFA, sf_offset), sbyte)


def compile_bf16_to_fp6_small_m(m: int, k: int, fmt: str = "e3m2", per_row: bool = False):
    """Compile the small-M BF16->MX-FP6 bytes quantizer for ``(m, k)``.

    Returns ``launch(bf16_input, w_gscale, codes_flat, scale_flat,
    alpha_out, inv_gs_out)`` where ``codes_flat``/``scale_flat`` are the flat
    views from :func:`allocate_bf16_to_fp6_tma_outputs` (``emit="bytes"``),
    ``w_gscale`` the f32 ``(1,)`` weight global scale, and ``alpha_out`` an
    f32 ``(1,)`` buffer that receives ``1/(a_gs*w_gs)``. The activation amax
    is computed IN-KERNEL (no separate ``vector_norm`` launch). Only the
    first ``m`` rows of codes and their scale bytes are written.

    With ``per_row=True`` the whole per-row global-scale recipe runs
    in-kernel (see :class:`SmallMQuantKernel`); ``inv_gs_out`` must be a bf16
    tensor with at least ``m`` leading elements and receives the per-row
    output correction ``bf16(1/gs_r)``. With ``per_row=False`` pass any bf16
    tensor (it is never written).
    """
    assert 1 <= m <= SMALL_M_MAX, f"small-M quantizer requires m<={SMALL_M_MAX}, got {m}"
    assert k % 128 == 0, f"K must be a multiple of 128, got {k}"
    assert fmt in ("e3m2", "e2m3", "e4m3"), f"unsupported act fmt: {fmt}"
    cache_key = (m, k, fmt, per_row)
    cached = _KERNEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    sf = cutlass.Float8E8M0FNU
    bf16_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m, k), stride_order=(1, 0), assumed_align=16
    )
    wgs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    codes_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Uint8, (m * k,), assumed_align=16
    )
    alpha_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (1,), assumed_align=4
    )
    inv_gs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.BFloat16, (m,), assumed_align=2
    )
    sfa_fake = make_ptr(sf, 16, AddressSpace.gmem, assumed_align=16)
    total_blocks = m * (k // _FP6_BLOCK_ELEMS)
    grid = (total_blocks + _THREADS - 1) // _THREADS
    kernel = SmallMQuantKernel(fmt, per_row=per_row)
    raise_if_kernel_resolution_frozen("cute.compile", target=kernel, cache_key=cache_key)
    raw = b12x_compile(
        kernel,
        bf16_fake,
        wgs_fake,
        codes_fake,
        sfa_fake,
        alpha_fake,
        inv_gs_fake,
        grid,
        current_cuda_stream(),
        compile_spec=KernelCompileSpec.from_key(
            "quantization.bf16_to_fp6_small_m",
            6,
            cache_key,
        ),
    )

    def launch(bf16_input, w_gscale, codes_flat, scale_flat, alpha_out, inv_gs_out):
        sfa_p = make_ptr(
            sf, scale_flat.data_ptr(), AddressSpace.gmem, assumed_align=16
        )
        raw(
            bf16_input,
            w_gscale,
            codes_flat[: m * k],
            sfa_p,
            alpha_out,
            inv_gs_out[:m],
            current_cuda_stream(),
        )

    launch = attach_programs(launch, raw)
    _KERNEL_CACHE[cache_key] = launch
    return launch
