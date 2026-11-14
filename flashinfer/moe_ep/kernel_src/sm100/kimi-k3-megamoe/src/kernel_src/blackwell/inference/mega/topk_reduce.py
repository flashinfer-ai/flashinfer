# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device combine reduce: collapse per-(token, topk) fc2 cells into one row.

The combine step writes one fc2 output per ``(token, topk)`` cell; this reduces
over the topk axis into the token-centric ``(token, hidden)`` output. The wire
format is described by ``CombineFormat``:

    bf16          -- no staging: bf16 terms reduced directly.
    32e4m3xe8m0   -- MXFP8: fp8 e4m3 data + per-32 e8m0 (power-of-2) scale.
    16e2m1xbf16   -- fp4 e2m1 data + per-16 bf16 amax (one level, no global);
                     dequant per element x = fp4 * (amax * (1 / 6)).

Task partition: each worker owns one ``(token, hidden_tile)`` and loops topk; the
flat worker index decodes into ``(token_idx, hidden_tile_idx)`` via a constant
divide by ``hidden_tiles``. The per-block scale is broadcast to a logical
per-hidden view (stride 0) so it tiles by the same worker index as the data. The
activation load stays in the topk loop (too large to hoist); the small scale and
score loads are hoisted ahead of the loop when topk is small.
"""
from .....helpers import codegen_options as _kimi_options

import os
from typing import ClassVar, Dict, Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Float32, Int32, Int64, T

from .....helpers.constants import Nvfp4E2M1RcpLimit
from .....helpers.dsl_helpers import mark_alignment
from .....quant_def import CombineFormat

# ---------------------------------------------------------------------------
# fp4 (e2m1) -> fp32 register decode.
#
# Blackwell has no e2m1->f32 upconvert: the framework's ``term.load().to(f32)``
# lowers to an ALU subnormal-normalization path. Both helpers
# below force a table-driven decode instead; ``e2m1_reg`` (N e2m1 codes, N % 8
# == 0) is read as packed b32 words and N fp32 values are written into
# ``fp32_reg`` in code order. The 16 e2m1 values are exact in fp32, so the two
# decoders are bit-for-bit identical (cross-check the optimal one against the
# cvt one over all 16 codes).
# ---------------------------------------------------------------------------


@cute.jit
def cvt_e2m1_to_fp32_cvt_ptx(e2m1_reg: cute.Tensor, fp32_reg: cute.Tensor) -> None:
    """Decode via the e2m1->f16 HW cvt (``cvt.rn.f16x2.e2m1x2``) then widen f16->f32.

    Safe baseline: the per-pair ``cvt`` instruction is itself a HW PRMT+F2FP, so
    the e2m1->f16 step already avoids ALU normalization; f16->f32 is one cheap
    ``cvt.f32.f16`` per element.
    """
    src_words = cute.recast_tensor(e2m1_reg, Int32)  # (N,) e2m1 -> (N/8,) b32
    for w in cutlass.range_constexpr(cute.size(src_words)):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.f32()] * 8),
            [src_words[w].ir_value()],
            "{\n"
            "  .reg .b8  b0, b1, b2, b3;\n"
            "  .reg .b32 p0, p1, p2, p3;\n"
            "  .reg .b16 c0, d0, c1, d1, c2, d2, c3, d3;\n"
            "  mov.b32 {b0, b1, b2, b3}, $8;\n"
            "  cvt.rn.f16x2.e2m1x2 p0, b0;\n"
            "  cvt.rn.f16x2.e2m1x2 p1, b1;\n"
            "  cvt.rn.f16x2.e2m1x2 p2, b2;\n"
            "  cvt.rn.f16x2.e2m1x2 p3, b3;\n"
            "  mov.b32 {c0, d0}, p0;\n"
            "  mov.b32 {c1, d1}, p1;\n"
            "  mov.b32 {c2, d2}, p2;\n"
            "  mov.b32 {c3, d3}, p3;\n"
            "  cvt.f32.f16 $0, c0;\n"
            "  cvt.f32.f16 $1, d0;\n"
            "  cvt.f32.f16 $2, c1;\n"
            "  cvt.f32.f16 $3, d1;\n"
            "  cvt.f32.f16 $4, c2;\n"
            "  cvt.f32.f16 $5, d2;\n"
            "  cvt.f32.f16 $6, c3;\n"
            "  cvt.f32.f16 $7, d3;\n"
            "}",
            "=f,=f,=f,=f,=f,=f,=f,=f,r",
            has_side_effects=False,
        )
        for i in cutlass.range_constexpr(8):
            fp32_reg[w * 8 + i] = Float32(llvm.extractvalue(T.f32(), res, [i]))


@cute.jit
def cvt_e2m1_to_fp32_optimal_ptx(e2m1_reg: cute.Tensor, fp32_reg: cute.Tensor) -> None:
    """Decode via a register-resident bf16 LUT + PRMT, landing fp32 directly.

    The 8 e2m1 magnitudes ``{0,.5,1,1.5,2,3,4,6}`` are exact bf16, so a PRMT
    byte-gather of the per-magnitude hi/lo bytes builds the bf16; ``fp32 =
    bf16 << 16`` makes the widen free. Sign (nibble bit3) is spread to the 4
    output-byte MSBs with one ``prmt`` of ``{word<<4, word}`` (selector 0x5140),
    avoiding the non-uniform shift the 4-bit-vs-8-bit stride would otherwise need.
    """
    src_words = cute.recast_tensor(e2m1_reg, Int32)  # (N,) e2m1 -> (N/8,) b32
    dst_words = cute.recast_tensor(fp32_reg, Int32)  # write fp32 bit patterns
    for w in cutlass.range_constexpr(cute.size(src_words)):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 8),
            [src_words[w].ir_value()],
            "{\n"
            "  .reg .b32 ha, hb, la, lb, inh, wl, ih, il, hl, ll, hh, lh, sl, sh, p0, p1, p2, p3;\n"
            "  mov.b32 ha, 0x3F3F3F00;\n"  # hi byte LUT, magnitudes 0..3
            "  mov.b32 hb, 0x40404040;\n"  # hi byte LUT, magnitudes 4..7
            "  mov.b32 la, 0xC0800000;\n"  # lo byte LUT, magnitudes 0..3
            "  mov.b32 lb, 0xC0804000;\n"  # lo byte LUT, magnitudes 4..7
            "  shr.b32 inh, $8, 16;\n"  # high 4 elements -> low 16 bits
            "  and.b32 il, $8, 0x00007777;\n"  # low 4 magnitude indices (clear sign)
            "  and.b32 ih, inh, 0x00007777;\n"  # high 4 magnitude indices
            "  prmt.b32 hl, ha, hb, il;\n"  # hi bytes for e0..e3
            "  prmt.b32 ll, la, lb, il;\n"  # lo bytes for e0..e3
            "  prmt.b32 hh, ha, hb, ih;\n"  # hi bytes for e4..e7
            "  prmt.b32 lh, la, lb, ih;\n"  # lo bytes for e4..e7
            "  shl.b32 wl, $8, 4;\n"
            "  prmt.b32 sl, wl, $8, 0x5140;\n"  # gather s0..s3 to byte MSBs
            "  and.b32 sl, sl, 0x80808080;\n"
            "  or.b32  hl, hl, sl;\n"
            "  shl.b32 wl, inh, 4;\n"
            "  prmt.b32 sh, wl, inh, 0x5140;\n"  # gather s4..s7 to byte MSBs
            "  and.b32 sh, sh, 0x80808080;\n"
            "  or.b32  hh, hh, sh;\n"
            "  prmt.b32 p0, ll, hl, 0x5140;\n"  # {bf16(e0), bf16(e1)}
            "  prmt.b32 p1, ll, hl, 0x7362;\n"  # {bf16(e2), bf16(e3)}
            "  prmt.b32 p2, lh, hh, 0x5140;\n"  # {bf16(e4), bf16(e5)}
            "  prmt.b32 p3, lh, hh, 0x7362;\n"  # {bf16(e6), bf16(e7)}
            "  shl.b32 $0, p0, 16;\n"
            "  and.b32 $1, p0, 0xFFFF0000;\n"
            "  shl.b32 $2, p1, 16;\n"
            "  and.b32 $3, p1, 0xFFFF0000;\n"
            "  shl.b32 $4, p2, 16;\n"
            "  and.b32 $5, p2, 0xFFFF0000;\n"
            "  shl.b32 $6, p3, 16;\n"
            "  and.b32 $7, p3, 0xFFFF0000;\n"
            "}",
            "=r,=r,=r,=r,=r,=r,=r,=r,r",
            has_side_effects=False,
        )
        for i in cutlass.range_constexpr(8):
            dst_words[w * 8 + i] = Int32(llvm.extractvalue(T.i32(), res, [i]))


S22_MX_V2_MIN_T = 16384  # [s22-mxvar] MX buckets (max tokens / rank) from which TopkReduce v2 is the default


class TopkReduce:
    """Combine reduce for a fixed ``(hidden, num_topk, combine_format)``.

    ``__init__`` pins the static shape and format (and the derived launch
    geometry); ``__call__`` (a ``@cute.jit`` launcher) sizes a 1D grid from the
    runtime token count and dispatches the format's kernel. The caller owns the
    torch->cute conversion and the ``cute.compile`` / ``aot_compile``.
    """

    _threads: ClassVar[int] = 128
    # combine_format.name -> hidden elements per worker (one LDG of data:
    # bf16 8*2B=16B, e4m3 16*1B=16B, e2m1 16*0.5B=8B). For quantized formats this
    # stays <= the scale block, so each worker reads exactly one scale entry.
    _hidden_per_thread: ClassVar[Dict[str, int]] = {"bf16": 8, "32e4m3xe8m0": 16, "16e2m1xbf16": 16}
    # topk count at/below which the scale + score loads are hoisted ahead of the
    # topk loop (small enough to not bloat registers; a CTA-broadcast read).
    _prefetch_limit: ClassVar[int] = 16
    # per-format v2 launch config (threads per CTA, load kind, store kind, workers per thread)
    _s18_defaults: ClassVar[Dict[str, Dict[str, int]]] = {
        "32e4m3xe8m0": {"tpb": 32, "ld": 2, "st": 0, "wpt": 1},
        "bf16": {"tpb": 32, "ld": 2, "st": 0, "wpt": 1},
    }

    def __init__(
        self, hidden: int, num_topk: int, combine_format: CombineFormat, quant_kind: Optional[str] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self.hidden = int(hidden)
        self.num_topk = int(num_topk)
        self.combine_format = combine_format
        self.hidden_per_thread = self._hidden_per_thread[combine_format.name]
        # hidden must tile cleanly both into worker slices and into scale blocks.
        align = max(combine_format.scale_block or self.hidden_per_thread, self.hidden_per_thread)
        if self.hidden % align != 0:
            raise ValueError(
                f"hidden ({self.hidden}) must be divisible by max(scale_block, "
                f"hidden_per_thread) = {align} for combine_format {combine_format}."
            )
        self.hidden_tiles = self.hidden // self.hidden_per_thread
        # tail guard only needed when the worker count per token is not a whole
        # number of CTAs; prefetch only when topk is small enough to hoist.
        self.require_predicate = self.hidden_tiles % self._threads != 0
        self.prefetch = self.num_topk <= self._prefetch_limit
        # MEGAMOE_S18=1: v2 kernels for the FP8 (32e4m3xe8m0) and BF16 wire formats. Same worker partition
        # and the same per-element instruction sequence (bit-identical output); only the data path changes:
        #   MEGAMOE_S18_TPB   threads per CTA (default 128)
        #   MEGAMOE_S18_LD    0 = DSL 128-bit copies, 1 = ld.global.nc.L1::no_allocate.v4 (default), 2 = same, kept in order
        #   MEGAMOE_S18_ST    0 = DSL store, 1 = st.global.cs (streaming) (default 0)
        #   MEGAMOE_S18_WPT   workers per thread (grid-stride loop; > 1 double-buffers the next worker's loads)
        # All slot loads of a worker are issued before its arithmetic. e2m1 (16e2m1xbf16) keeps the reference kernel.
        # Default (MEGAMOE_S18 unset): v2 for NVFP4 MegaMoE only. MX (mxfp4_mxfp8) keeps the reference reduce: cold
        # -0.55% but sustained (power-capped) +0.23..+0.55% at u2k (docs/optimizations.md "s18-red2"). MEGAMOE_S18=0 / 1 forces off / on.
        # MX uses v2 from the S22_MX_V2_MIN_T bucket up (MEGAMOE_S18_MX_MIN_T overrides; 0 = never): at large token
        # counts the reduce sits on the critical path of the heaviest rank (variable-token steps) and v2 wins in both
        # regimes there (docs/optimizations.md "s22-mxvar").
        forced = _kimi_options.environ.get("MEGAMOE_S18", "")
        kind = str(getattr(quant_kind, "value", quant_kind))
        mx_min_t = int(_kimi_options.environ.get("MEGAMOE_S18_MX_MIN_T", str(S22_MX_V2_MIN_T)) or "0")
        mx_v2 = kind == "mxfp4_mxfp8" and mx_min_t > 0 and max_tokens is not None and max_tokens >= mx_min_t
        wanted = forced == "1" if forced in ("0", "1") else (kind == "nvfp4" or mx_v2)
        self.s18 = wanted and combine_format.name in ("32e4m3xe8m0", "bf16")
        # Defaults per wire format = best measured GB300 config (docs/optimizations.md "s18-red2"); env vars override.
        default = self._s18_defaults.get(combine_format.name, self._s18_defaults["bf16"])
        self.s18_tpb = int(_kimi_options.environ.get("MEGAMOE_S18_TPB") or default["tpb"])
        self.s18_ld = int(_kimi_options.environ.get("MEGAMOE_S18_LD") or default["ld"])
        self.s18_st = int(_kimi_options.environ.get("MEGAMOE_S18_ST") or default["st"])
        self.s18_wpt = int(_kimi_options.environ.get("MEGAMOE_S18_WPT") or default["wpt"])
        if self.s18 and (self.s18_tpb % 32 or not 32 <= self.s18_tpb <= 1024 or self.s18_wpt < 1):
            raise ValueError("MEGAMOE_S18_TPB must be a multiple of 32 in [32, 1024]; MEGAMOE_S18_WPT >= 1.")

    def s18_tag(self) -> str:
        """Compile-cache tag of the s18 configuration ('' when off)."""
        if not self.s18:
            return ""
        return f"_s18t{self.s18_tpb}l{self.s18_ld}s{self.s18_st}w{self.s18_wpt}"

    # -- launcher -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],  # (token, topk, hidden)
        reduced_output: cute.Tensor,  # (token, hidden)
        topk_score: Optional[cute.Tensor],  # (token, topk)
        stream: cuda.CUstream,
    ):
        threads = self.s18_tpb if cutlass.const_expr(self.s18) else self._threads
        total_workers = reduced_output.shape[0] * self.hidden_tiles
        per_cta = threads * self.s18_wpt if cutlass.const_expr(self.s18) else threads
        grid = [(total_workers + per_cta - 1) // per_cta, 1, 1]
        block = [threads, 1, 1]

        combine_quant = cute.make_tensor(
            combine_quant.iterator,
            cute.make_layout(
                (combine_quant.shape[0], self.num_topk, self.hidden), stride=combine_quant.stride
            ),
        )
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden), stride=reduced_output.stride),
        )
        if cutlass.const_expr(topk_score is not None):
            topk_score = cute.make_tensor(
                topk_score.iterator,
                cute.make_layout((topk_score.shape[0], self.num_topk), stride=topk_score.stride),
            )

        if cutlass.const_expr(not self.combine_format.is_quantized):
            if cutlass.const_expr(self.s18):
                self._reduce_v2(combine_quant, None, topk_score, reduced_output, total_workers).launch(
                    grid=grid, block=block, stream=stream
                )
            else:
                self._reduce_bf16(combine_quant, topk_score, reduced_output).launch(
                    grid=grid, block=block, stream=stream
                )
            return

        # The mega kernel hands sf in already as the depth-2 broadcast layout; a
        # plain (torch) sf is depth-1 and gets its hidden mode split into
        # (sf_vec, hidden/sf_vec):(0, s_h) so logical hidden h reads block h//sf_vec.
        sf_vec = self.combine_format.scale_block
        if cutlass.const_expr(cute.depth(combine_sf.layout) >= 2):
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout(
                    (combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)),
                    stride=combine_sf.stride,
                ),
            )
        else:
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout(
                    (combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)),
                    stride=(combine_sf.stride[0], combine_sf.stride[1], (0, combine_sf.stride[2])),
                ),
            )

        if cutlass.const_expr(self.s18):
            self._reduce_v2(combine_quant, sf, topk_score, reduced_output, total_workers).launch(
                grid=grid, block=block, stream=stream
            )
        elif cutlass.const_expr(self.combine_format.act_dtype is cutlass.Float8E4M3FN):
            self._reduce_mxfp8(combine_quant, sf, topk_score, reduced_output).launch(
                grid=grid, block=block, stream=stream
            )
        else:
            self._reduce_fp4(combine_quant, sf, topk_score, reduced_output).launch(
                grid=grid, block=block, stream=stream
            )

    # -- kernels --------------------------------------------------------------

    @cute.kernel
    def _reduce_bf16(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = (
            topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        )
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            terms = cute.zipped_divide(
                combine_output[token_idx, None, None], (num_topk, hidden_per_thread)
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[
                (None,), (hidden_tile_idx,)
            ]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread,), cutlass.BFloat16)
                cute.copy(
                    load_atom,
                    mark_alignment(terms[k, None], hidden_per_thread * cutlass.BFloat16.width // 8),
                    term,
                )
                if cutlass.const_expr(topk_score is not None and not prefetch):
                    score_reg[k] = topk_score[token_idx, Int32(k)]
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    value_pair = (Float32(term[i]), Float32(term[i + 1]))
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            value_pair, score_pair, (acc[i], acc[i + 1])
                        )
                    else:
                        if cutlass.const_expr(topk_score is not None):
                            acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(value_pair, score_pair)
                        else:
                            acc[i] = value_pair[0]
                            acc[i + 1] = value_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_mxfp8(
        self,
        combine_quant: cute.Tensor,
        combine_sf: cute.Tensor,  # depth-2 broadcast view: logical (token, topk, hidden) e8m0
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = (
            topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        )
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk,), cutlass.Float8E8M0FNU)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None], (num_topk, hidden_per_thread)
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None], (num_topk, hidden_per_thread)
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[
                (None,), (hidden_tile_idx,)
            ]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(
                    sf[None, 0], scale_reg
                )  # one scale per topk slot (stride-0 broadcast)

            fp8_dtype = self.combine_format.act_dtype
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), fp8_dtype, num_bits_per_copy=128
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread,), fp8_dtype)
                cute.copy(
                    load_atom,
                    mark_alignment(codes[k, None], hidden_per_thread * fp8_dtype.width // 8),
                    term,
                )
                value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
                value.store(term.load().to(cutlass.Float32))

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                scale = Float32(scale_reg[k])  # e8m0 -> f32
                scale_pair = (scale, scale)
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = cute.arch.mul_packed_f32x2((value[i], value[i + 1]), scale_pair)
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            dequant_pair, score_pair, (acc[i], acc[i + 1])
                        )
                    else:
                        if cutlass.const_expr(topk_score is not None):
                            acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(
                                dequant_pair, score_pair
                            )
                        else:
                            acc[i] = dequant_pair[0]
                            acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=256),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_fp4(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden) e2m1 (logical)
        combine_sf: cute.Tensor,  # depth-2 broadcast view: logical (token, topk, hidden) bf16 amax
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        threads = self._threads
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        worker_idx = cute.arch.block_idx()[0] * Int32(threads) + cute.arch.thread_idx()[0]
        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        score_dtype = (
            topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        )
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk,), cutlass.BFloat16)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None], (num_topk, hidden_per_thread)
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None], (num_topk, hidden_per_thread)
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(reduced_output[token_idx, None], (hidden_per_thread,))[
                (None,), (hidden_tile_idx,)
            ]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(sf[None, 0], scale_reg)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN, num_bits_per_copy=64
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float4E2M1FN)
                cute.copy(
                    load_atom,
                    mark_alignment(
                        codes[k, None], hidden_per_thread * cutlass.Float4E2M1FN.width // 8
                    ),
                    term,
                )
                value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
                # Dev-only knob (MEGA_F4CVT_USE_MANUAL): manual LUT/PRMT decode vs
                # the HW cvt path, so both SASS forms can be compared on device;
                # one is kept once chosen. Read inline on purpose -- never a
                # customer-facing option. Both decoders are bit-exact.
                if cutlass.const_expr(_kimi_options.environ.get("MEGA_F4CVT_USE_MANUAL", "0") == "1"):
                    cvt_e2m1_to_fp32_optimal_ptx(term, value)
                else:
                    cvt_e2m1_to_fp32_cvt_ptx(term, value)

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                # amax (bf16) -> per-element scale; (1/6) folds the fp4 grid max.
                scale = Float32(scale_reg[k]) * Float32(Nvfp4E2M1RcpLimit)
                scale_pair = (scale, scale)
                score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = cute.arch.mul_packed_f32x2((value[i], value[i + 1]), scale_pair)
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                            dequant_pair, score_pair, (acc[i], acc[i + 1])
                        )
                    elif cutlass.const_expr(topk_score is not None):
                        acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(dequant_pair, score_pair)
                    else:
                        acc[i] = dequant_pair[0]
                        acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=256),
                out,
                mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    # -- v2 kernel ---------------------------------------------------------------------------------------

    @cute.jit
    def _s18_load16(self, src: cute.Tensor, dst: cute.Tensor) -> None:
        """16-byte load of ``src`` (16 B of codes) into register tensor ``dst`` (same element type)."""
        if cutlass.const_expr(self.s18_ld >= 1):
            # LD=2: side-effecting asm keeps program order, so every slot load of the worker stays ahead of its
            # arithmetic (ptxas cannot sink a load next to its first use to save registers).
            res = llvm.inline_asm(
                llvm.StructType.get_literal([T.i32()] * 4),
                [src.iterator.toint().ir_value()],
                "ld.global.nc.L1::no_allocate.v4.b32 {$0, $1, $2, $3}, [$4];",
                "=r,=r,=r,=r,l",
                has_side_effects=self.s18_ld == 2,
            )
            words = cute.recast_tensor(dst, Int32)
            for i in cutlass.range_constexpr(4):
                words[i] = Int32(llvm.extractvalue(T.i32(), res, [i]))
        else:
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), dst.element_type, num_bits_per_copy=128),
                mark_alignment(src, 16),
                dst,
            )

    @cute.jit
    def _s18_store(self, src: cute.Tensor, dst: cute.Tensor) -> None:
        """Store the bf16 register row ``src`` (hidden_per_thread elements) to ``dst``."""
        nbytes = cute.size(src) * src.element_type.width // 8
        if cutlass.const_expr(self.s18_st == 1):
            words = cute.recast_tensor(src, Int32)
            for c in cutlass.range_constexpr(nbytes // 16):
                llvm.inline_asm(
                    None,
                    [
                        (dst.iterator.toint() + Int64(c * 16)).ir_value(),
                        words[4 * c].ir_value(),
                        words[4 * c + 1].ir_value(),
                        words[4 * c + 2].ir_value(),
                        words[4 * c + 3].ir_value(),
                    ],
                    "st.global.cs.v4.b32 [$0], {$1, $2, $3, $4};",
                    "l,r,r,r,r",
                    has_side_effects=True,
                )
        else:
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(), src.element_type, num_bits_per_copy=min(256, nbytes * 8)
                ),
                src,
                mark_alignment(dst, nbytes),
            )

    @cute.kernel
    def _reduce_v2(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden): fp8 e4m3 codes or bf16 terms
        combine_sf: Optional[cute.Tensor],  # depth-2 broadcast e8m0 view (FP8) or None (BF16)
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        total_workers: Int32,
    ):
        threads = self.s18_tpb
        wpt = self.s18_wpt
        hpt = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        prefetch = self.prefetch
        quantized = self.combine_format.is_quantized
        act_dtype = self.combine_format.act_dtype
        out_dtype = reduced_output.element_type
        score_dtype = (
            topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        )

        first = cute.arch.block_idx()[0] * Int32(threads * wpt) + cute.arch.thread_idx()[0]
        for wi in cutlass.range_constexpr(wpt):
            worker_idx = first + Int32(wi * threads)
            if worker_idx < total_workers:
                token_idx = worker_idx // hidden_tiles
                hidden_tile_idx = worker_idx % hidden_tiles
                codes = cute.zipped_divide(
                    combine_quant[token_idx, None, None], (num_topk, hpt)
                )[(None, None), (0, hidden_tile_idx)]
                dst = cute.zipped_divide(reduced_output[token_idx, None], (hpt,))[(None,), (hidden_tile_idx,)]

                # ---- issue every slot load of this worker before any arithmetic ----
                terms = []
                for k in cutlass.range_constexpr(num_topk):
                    t = cute.make_rmem_tensor((hpt,), act_dtype)
                    self._s18_load16(codes[k, None], t)
                    terms.append(t)

                score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
                if cutlass.const_expr(topk_score is not None):
                    if cutlass.const_expr(prefetch):
                        cute.autovec_copy(topk_score[token_idx, None], score_reg)
                else:
                    for k in cutlass.range_constexpr(num_topk):
                        score_reg[k] = score_dtype(1)
                scale_reg = cute.make_rmem_tensor((num_topk,), cutlass.Float8E8M0FNU)
                sf = None
                if cutlass.const_expr(quantized):
                    sf = cute.zipped_divide(combine_sf[token_idx, None, None], (num_topk, hpt))[
                        (None, None), (0, hidden_tile_idx)
                    ]
                    if cutlass.const_expr(prefetch):
                        cute.autovec_copy(sf[None, 0], scale_reg)

                # ---- arithmetic: the reference instruction sequence, k ascending ----
                acc = cute.make_rmem_tensor((hpt,), cutlass.Float32)
                for k in cutlass.range_constexpr(num_topk):
                    if cutlass.const_expr(not prefetch):
                        if cutlass.const_expr(quantized):
                            scale_reg[k] = sf[k, 0]
                        if cutlass.const_expr(topk_score is not None):
                            score_reg[k] = topk_score[token_idx, Int32(k)]
                    score_pair = (Float32(score_reg[k]), Float32(score_reg[k]))
                    if cutlass.const_expr(quantized):
                        value = cute.make_rmem_tensor((hpt,), cutlass.Float32)
                        value.store(terms[k].load().to(cutlass.Float32))
                        scale = Float32(scale_reg[k])  # e8m0 -> f32
                        scale_pair = (scale, scale)
                        for i in cutlass.range_constexpr(0, hpt, 2):
                            dequant_pair = cute.arch.mul_packed_f32x2((value[i], value[i + 1]), scale_pair)
                            if cutlass.const_expr(k != 0):
                                acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                                    dequant_pair, score_pair, (acc[i], acc[i + 1])
                                )
                            else:
                                if cutlass.const_expr(topk_score is not None):
                                    acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(dequant_pair, score_pair)
                                else:
                                    acc[i] = dequant_pair[0]
                                    acc[i + 1] = dequant_pair[1]
                    else:
                        term = terms[k]
                        for i in cutlass.range_constexpr(0, hpt, 2):
                            value_pair = (Float32(term[i]), Float32(term[i + 1]))
                            if cutlass.const_expr(k != 0):
                                acc[i], acc[i + 1] = cute.arch.fma_packed_f32x2(
                                    value_pair, score_pair, (acc[i], acc[i + 1])
                                )
                            else:
                                if cutlass.const_expr(topk_score is not None):
                                    acc[i], acc[i + 1] = cute.arch.mul_packed_f32x2(value_pair, score_pair)
                                else:
                                    acc[i] = value_pair[0]
                                    acc[i + 1] = value_pair[1]

                out = cute.make_rmem_tensor((hpt,), out_dtype)
                out.store(acc.load().to(out_dtype))
                self._s18_store(out, dst)
