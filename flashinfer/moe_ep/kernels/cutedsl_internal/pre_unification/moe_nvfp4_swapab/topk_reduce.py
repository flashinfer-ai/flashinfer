# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Device combine reduce: collapse per-(token, topk) fc2 cells into one row.

The combine step writes one fc2 output per ``(token, topk)`` cell; this reduces
over the topk axis into the token-centric ``(token, hidden)`` output. The wire
format is a :class:`~pre_unification.comm.token_comm.CombineFormat`:

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

from __future__ import annotations

from typing import ClassVar, Dict, Optional, Tuple, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as cutlass_utils
from cutlass.cutlass_dsl import Float32, Int32, T
from cutlass._mlir.dialects import llvm

from ..common.megamoe_constants import Nvfp4E2M1RcpLimit
from ..comm.token_comm import CombineFormat


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
    src_words = cute.recast_tensor(e2m1_reg, Int32)          # (N,) e2m1 -> (N/8,) b32
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
    src_words = cute.recast_tensor(e2m1_reg, Int32)          # (N,) e2m1 -> (N/8,) b32
    dst_words = cute.recast_tensor(fp32_reg, Int32)           # write fp32 bit patterns
    for w in cutlass.range_constexpr(cute.size(src_words)):
        res = llvm.inline_asm(
            llvm.StructType.get_literal([T.i32()] * 8),
            [src_words[w].ir_value()],
            "{\n"
            "  .reg .b32 ha, hb, la, lb, inh, wl, ih, il, hl, ll, hh, lh, sl, sh, p0, p1, p2, p3;\n"
            "  mov.b32 ha, 0x3F3F3F00;\n"      # hi byte LUT, magnitudes 0..3
            "  mov.b32 hb, 0x40404040;\n"      # hi byte LUT, magnitudes 4..7
            "  mov.b32 la, 0xC0800000;\n"      # lo byte LUT, magnitudes 0..3
            "  mov.b32 lb, 0xC0804000;\n"      # lo byte LUT, magnitudes 4..7
            "  shr.b32 inh, $8, 16;\n"          # high 4 elements -> low 16 bits
            "  and.b32 il, $8, 0x00007777;\n"   # low 4 magnitude indices (clear sign)
            "  and.b32 ih, inh, 0x00007777;\n"  # high 4 magnitude indices
            "  prmt.b32 hl, ha, hb, il;\n"      # hi bytes for e0..e3
            "  prmt.b32 ll, la, lb, il;\n"      # lo bytes for e0..e3
            "  prmt.b32 hh, ha, hb, ih;\n"      # hi bytes for e4..e7
            "  prmt.b32 lh, la, lb, ih;\n"      # lo bytes for e4..e7
            "  shl.b32 wl, $8, 4;\n"
            "  prmt.b32 sl, wl, $8, 0x5140;\n"  # gather s0..s3 to byte MSBs
            "  and.b32 sl, sl, 0x80808080;\n"
            "  or.b32  hl, hl, sl;\n"
            "  shl.b32 wl, inh, 4;\n"
            "  prmt.b32 sh, wl, inh, 0x5140;\n" # gather s4..s7 to byte MSBs
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
    _hidden_per_thread: ClassVar[Dict[str, int]] = {
        "bf16": 8,
        "32e4m3xe8m0": 16,
        "32e5m2xe8m0": 16,
        "16e2m1xbf16": 16,
    }
    # topk count at/below which the scale + score loads are hoisted ahead of the
    # topk loop (small enough to not bloat registers; a CTA-broadcast read).
    _prefetch_limit: ClassVar[int] = 16

    _Float32Value = Union[Tuple[Float32, Float32], Float32]

    def __init__(
        self,
        hidden: int,
        num_topk: int,
        combine_format: CombineFormat,
        *,
        sm_arch: str,
        num_remote_nodes: int = 0,
        ranks_per_node: int = 1,
        num_experts_per_rank: int = 1,
        combine_slot_indexed: bool = False,
        f4cvt_use_manual: bool = False,
        persistent: bool = False,
        num_sms: Optional[int] = None,
        ctas_per_sm: Optional[int] = None,
        max_fetch: int = 8,
        fetch_rounds: int = 4,
    ) -> None:
        self.hidden = int(hidden)
        # Diagnostic decoder choice (MegaMoEDiagnostics.f4cvt_use_manual):
        # manual LUT/PRMT FP4 decode instead of the hardware cvt path; both
        # are bit-exact. Plain bool so cutlass.const_expr folds the branch.
        self.f4cvt_use_manual = bool(f4cvt_use_manual)
        # Slot-indexed landing: the remote partial for (token, node) lives at
        # internode_combine[phase][node][token_node_slot[node][token]] rather
        # than at the token row. Requires the token_node_slot map at call time.
        self.combine_slot_indexed = bool(combine_slot_indexed)
        self.num_topk = int(num_topk)
        self.combine_format = combine_format
        self.num_remote_nodes = int(num_remote_nodes)
        self.ranks_per_node = int(ranks_per_node)
        self.num_experts_per_rank = int(num_experts_per_rank)
        if self.num_remote_nodes < 0:
            raise ValueError("num_remote_nodes must be non-negative")
        arch_code = sm_arch.removeprefix("sm_")
        if arch_code[-1:] in ("a", "f"):
            arch_code = arch_code[:-1]
        if not sm_arch.startswith("sm_") or not arch_code.isdigit():
            raise ValueError(
                f"sm_arch must have the form 'sm_XX', got {sm_arch!r}."
            )
        arch_number = int(arch_code)
        if arch_number < 90:
            raise ValueError(
                f"sm_arch must target SM90 or newer, got {sm_arch!r}."
            )
        self.use_scalar_math = arch_number < 100
        if self.use_scalar_math and combine_format.name != "bf16":
            raise ValueError(
                f"sm_arch={sm_arch!r} only supports BF16 combine, "
                f"got {combine_format.name!r}."
            )
        self.sm_arch = sm_arch
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

        self.persistent = bool(persistent)
        if self.persistent:
            if combine_format.is_quantized or self.num_remote_nodes:
                raise ValueError("Persistent reduction requires local BF16 combine.")
            load_regs = self.hidden_per_thread * combine_format.act_dtype.width // 32
            regs_needed = self.hidden_per_thread + self.num_topk * load_regs + 16
            self.ctas_per_sm = (
                int(ctas_per_sm) if ctas_per_sm is not None
                else max(4, min(8, 65536 // (self._threads * regs_needed)))
            )
            self.max_fetch = int(max_fetch)
            self.fetch_rounds = int(fetch_rounds)
            self.num_sms = int(
                num_sms if num_sms else cutlass_utils.HardwareInfo().get_device_multiprocessor_count()
            )
            if min(self.ctas_per_sm, self.max_fetch, self.fetch_rounds, self.num_sms) < 1:
                raise ValueError("Persistent scheduler parameters must be positive.")
            self.num_ctas = self.ctas_per_sm * self.num_sms
            self.warps_per_cta = self._threads // 32

    @cute.jit
    def _fma(
        self,
        lhs: _Float32Value,
        rhs: Float32,
        acc: _Float32Value,
    ) -> _Float32Value:
        """Architecture-selected FP32 FMA with a scalar multiplier."""
        if cutlass.const_expr(self.use_scalar_math):
            # CUTLASS DSL 4.6 exposes explicit scalar math through cute.math;
            # the CI-pinned 4.5.2 wheel needs the equivalent PTX fallback.
            if cutlass.const_expr(hasattr(cute.math, "fma")):
                return cute.math.fma(lhs, rhs, acc)
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [lhs.ir_value(), rhs.ir_value(), acc.ir_value()],
                    "fma.rn.f32 $0, $1, $2, $3;",
                    "=f,f,f,f",
                    has_side_effects=False,
                )
            )
        return cute.arch.fma_packed_f32x2(lhs, (rhs, rhs), acc)

    @cute.jit
    def _fmul(
        self,
        lhs: _Float32Value,
        rhs: Float32,
    ) -> _Float32Value:
        """Architecture-selected FP32 multiply with a scalar multiplier."""
        if cutlass.const_expr(self.use_scalar_math):
            # Keep the same 4.6 public-API / 4.5.2 PTX split as _fma.
            if cutlass.const_expr(hasattr(cute.math, "mul")):
                return cute.math.mul(lhs, rhs)
            return Float32(
                llvm.inline_asm(
                    T.f32(),
                    [lhs.ir_value(), rhs.ir_value()],
                    "mul.rn.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                )
            )
        return cute.arch.mul_packed_f32x2(lhs, (rhs, rhs))

    # -- launcher -------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        combine_quant: cute.Tensor,         # (token, topk, hidden)
        combine_sf: Optional[cute.Tensor],  # (token, topk, hidden)
        reduced_output: cute.Tensor,        # (token, hidden)
        topk_score: Optional[cute.Tensor],  # (token, topk)
        stream: cuda.CUstream,
        internode_combine: Optional[cute.Tensor] = None,
        combine_phase_snapshot: Optional[cute.Tensor] = None,
        topk_idx: Optional[cute.Tensor] = None,
        local_rank: Optional[Int32] = None,
        token_node_slot: Optional[cute.Tensor] = None,
        counters: Optional[cute.Tensor] = None,
        num_tokens_dev: Optional[cute.Tensor] = None,
    ):
        """Counters must be zeroed before each persistent launch; the live count is read-only."""
        if cutlass.const_expr(self.persistent):
            if cutlass.const_expr(internode_combine is not None):
                raise ValueError("Persistent reduction does not accept internode partials.")
            if cutlass.const_expr(counters is None or num_tokens_dev is None):
                raise ValueError("Persistent reduction requires counters and num_tokens_dev.")
        elif cutlass.const_expr(num_tokens_dev is not None):
            raise ValueError("num_tokens_dev requires persistent=True.")
        threads = self._threads
        total_workers = reduced_output.shape[0] * self.hidden_tiles
        grid = [(total_workers + threads - 1) // threads, 1, 1]
        block = [threads, 1, 1]

        if cutlass.const_expr(internode_combine is not None):
            combine_quant = cute.make_tensor(
                combine_quant.iterator,
                cute.make_layout(
                    (2, combine_quant.shape[1], self.num_topk, self.hidden),
                    stride=combine_quant.stride,
                ),
            )
            internode_combine = cute.make_tensor(
                internode_combine.iterator,
                cute.make_layout(
                    (
                        2,
                        self.num_remote_nodes,
                        internode_combine.shape[2],
                        self.hidden,
                    ),
                    stride=internode_combine.stride,
                ),
            )
            topk_idx = cute.make_tensor(
                topk_idx.iterator,
                cute.make_layout(
                    (topk_idx.shape[0], self.num_topk),
                    stride=topk_idx.stride,
                ),
            )
            if cutlass.const_expr(self.combine_slot_indexed):
                if cutlass.const_expr(token_node_slot is None):
                    raise ValueError(
                        "combine_slot_indexed requires token_node_slot"
                    )
                token_node_slot = cute.make_tensor(
                    token_node_slot.iterator,
                    cute.make_layout(
                        (self.num_remote_nodes, token_node_slot.shape[1]),
                        stride=token_node_slot.stride,
                    ),
                )
            else:
                token_node_slot = None
        else:
            combine_quant = cute.make_tensor(
                combine_quant.iterator,
                cute.make_layout(
                    (combine_quant.shape[0], self.num_topk, self.hidden),
                    stride=combine_quant.stride,
                ),
            )
        reduced_output = cute.make_tensor(
            reduced_output.iterator,
            cute.make_layout((reduced_output.shape[0], self.hidden), stride=reduced_output.stride))
        if cutlass.const_expr(topk_score is not None):
            topk_score = cute.make_tensor(
                topk_score.iterator,
                cute.make_layout((topk_score.shape[0], self.num_topk), stride=topk_score.stride))

        if cutlass.const_expr(self.persistent):
            self._reduce_persistent(
                combine_quant, topk_score, reduced_output, counters, num_tokens_dev
            ).launch(grid=[self.num_ctas, 1, 1], block=block, stream=stream)
            return

        if cutlass.const_expr(not self.combine_format.is_quantized):
            if cutlass.const_expr(internode_combine is not None):
                self._reduce_bf16_internode(
                    combine_quant,
                    internode_combine,
                    combine_phase_snapshot,
                    topk_idx,
                    topk_score,
                    reduced_output,
                    local_rank,
                    token_node_slot,
                ).launch(grid=grid, block=block, stream=stream)
            else:
                self._reduce_bf16(
                    combine_quant, topk_score, reduced_output
                ).launch(grid=grid, block=block, stream=stream)
            return

        # The mega kernel hands sf in already as the depth-2 broadcast layout; a
        # plain (torch) sf is depth-1 and gets its hidden mode split into
        # (sf_vec, hidden/sf_vec):(0, s_h) so logical hidden h reads block h//sf_vec.
        sf_vec = self.combine_format.scale_block
        if cutlass.const_expr(cute.depth(combine_sf.layout) >= 2):
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout((combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)), stride=combine_sf.stride))
        else:
            sf = cute.make_tensor(
                combine_sf.iterator,
                cute.make_layout(
                    (combine_sf.shape[0], self.num_topk, (sf_vec, self.hidden // sf_vec)),
                    stride=(combine_sf.stride[0], combine_sf.stride[1], (0, combine_sf.stride[2])),
                ),
            )

        if cutlass.const_expr(self.combine_format.act_dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)):
            self._reduce_mxfp8(combine_quant, sf, topk_score, reduced_output).launch(
                grid=grid, block=block, stream=stream,
            )
        else:
            self._reduce_fp4(combine_quant, sf, topk_score, reduced_output).launch(
                grid=grid, block=block, stream=stream,
            )

    @cute.jit
    def _mark_alignment(self, tensor: cute.Tensor, align_bytes: int) -> cute.Tensor:
        p = tensor.iterator
        return cute.make_tensor(
            cute.make_ptr(p.dtype, p.toint(), p.memspace, assumed_align=align_bytes),
            tensor.layout,
        )

    # -- kernels --------------------------------------------------------------

    @cute.kernel
    def _reduce_bf16_internode(
        self,
        combine_output: cute.Tensor,
        internode_combine: cute.Tensor,
        combine_phase_snapshot: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        local_rank,
        token_node_slot: Optional[cute.Tensor],
    ):
        worker_idx = (
            cute.arch.block_idx()[0] * Int32(self._threads)
            + cute.arch.thread_idx()[0]
        )
        self._reduce_bf16_internode_worker(
            combine_output, internode_combine, combine_phase_snapshot,
            topk_idx, topk_score, reduced_output, local_rank,
            token_node_slot, worker_idx,
        )

    @cute.jit
    def reduce_internode_tail(
        self,
        combine_output: cute.Tensor,
        internode_combine: cute.Tensor,
        combine_phase_snapshot: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        local_rank,
        token_node_slot: Optional[cute.Tensor],
        worker_idx: Int32,
        worker_stride: Int32,
    ):
        """Same ordered worker, grid-strided by the tail's dispatch threads.

        The caller must have drained GIN arrivals and the first tail NVLink
        barrier. No phase advance or combine-buffer reuse may precede this call.
        Restore the same static modes as the standalone host launcher; this
        creates views only, with no additional allocations or data movement.
        """
        combine_output = cute.make_tensor(
            combine_output.iterator,
            cute.make_layout(
                (2, combine_output.shape[1], self.num_topk, self.hidden),
                stride=combine_output.stride,
            ),
        )
        internode_combine = cute.make_tensor(
            internode_combine.iterator,
            cute.make_layout(
                (2, self.num_remote_nodes, internode_combine.shape[2], self.hidden),
                stride=internode_combine.stride,
            ),
        )
        topk_idx = cute.make_tensor(
            topk_idx.iterator,
            cute.make_layout((topk_idx.shape[0], self.num_topk), stride=topk_idx.stride),
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
        if cutlass.const_expr(self.combine_slot_indexed):
            if cutlass.const_expr(token_node_slot is None):
                raise ValueError("combine_slot_indexed requires token_node_slot")
            token_node_slot = cute.make_tensor(
                token_node_slot.iterator,
                cute.make_layout(
                    (self.num_remote_nodes, token_node_slot.shape[1]),
                    stride=token_node_slot.stride,
                ),
            )
        else:
            token_node_slot = None

        total_workers = Int32(reduced_output.shape[0] * self.hidden_tiles)
        while worker_idx < total_workers:
            self._reduce_bf16_internode_worker(
                combine_output, internode_combine, combine_phase_snapshot,
                topk_idx, topk_score, reduced_output, local_rank,
                token_node_slot, worker_idx,
            )
            worker_idx = worker_idx + worker_stride

    @cute.jit
    def _reduce_bf16_internode_worker(
        self,
        combine_output: cute.Tensor,          # (phase, token, topk, hidden)
        internode_combine: cute.Tensor,       # (phase, node, token, hidden)
        combine_phase_snapshot: cute.Tensor,  # (1,)
        topk_idx: cute.Tensor,                 # (token, topk)
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        local_rank,
        token_node_slot: Optional[cute.Tensor],
        worker_idx: Int32,
    ):
        """FP32 sum of local top-k cells plus one reduced partial per node."""
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        needs_guard = self.require_predicate
        out_dtype = reduced_output.element_type

        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            phase = Int32(
                cute.arch.load(
                    combine_phase_snapshot.iterator,
                    Int32,
                    sem="acquire",
                    scope="gpu",
                )
            ) & Int32(1)
            local_terms = cute.zipped_divide(
                combine_output[phase, token_idx, None, None],
                (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            remote_terms = cute.zipped_divide(
                internode_combine[phase, None, token_idx, None],
                (self.num_remote_nodes, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            dst = cute.zipped_divide(
                reduced_output[token_idx, None], (hidden_per_thread,),
            )[(None,), (hidden_tile_idx,)]

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                cutlass.BFloat16,
                num_bits_per_copy=128,
            )
            acc = cute.make_rmem_tensor(
                (hidden_per_thread,), cutlass.Float32,
            )
            for i in cutlass.range_constexpr(hidden_per_thread):
                acc[i] = Float32(0.0)

            my_node = Int32(local_rank) // Int32(self.ranks_per_node)
            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor(
                    (hidden_per_thread,), cutlass.BFloat16,
                )
                cute.copy(load_atom, local_terms[k, None], term)
                expert_id = Int32(topk_idx[token_idx, Int32(k)])
                expert_node = Int32(-1)
                if expert_id >= Int32(0):
                    expert_node = (
                        expert_id
                        // Int32(self.num_experts_per_rank)
                        // Int32(self.ranks_per_node)
                    )
                if expert_node == my_node:
                    score = Float32(1.0)
                    if cutlass.const_expr(topk_score is not None):
                        score = Float32(topk_score[token_idx, Int32(k)])
                    if cutlass.const_expr(self.use_scalar_math):
                        for i in cutlass.range_constexpr(hidden_per_thread):
                            acc[i] = self._fma(
                                Float32(term[i]), score, acc[i],
                            )
                    else:
                        for i in cutlass.range_constexpr(
                            0, hidden_per_thread, 2
                        ):
                            acc[i], acc[i + 1] = self._fma(
                                (Float32(term[i]), Float32(term[i + 1])),
                                score,
                                (acc[i], acc[i + 1]),
                            )

            # Each remote node already reduced all of this token's hits in
            # FP32 and cast once to BF16. Add exactly one row for each distinct
            # node present in topk_idx; stale rows/cells in the selected phase
            # are therefore never consumed and need no cross-node clear race.
            for remote_slot_off in cutlass.range_constexpr(
                0, self.num_remote_nodes, 1
            ):
                remote_slot = Int32(remote_slot_off)
                remote_node = remote_slot
                if remote_slot >= my_node:
                    remote_node = remote_slot + Int32(1)
                has_remote = Int32(0)
                for k in cutlass.range_constexpr(0, num_topk, 1):
                    expert_id = Int32(topk_idx[token_idx, Int32(k)])
                    expert_node = Int32(-1)
                    if expert_id >= Int32(0):
                        expert_node = (
                            expert_id
                            // Int32(self.num_experts_per_rank)
                            // Int32(self.ranks_per_node)
                        )
                    if expert_node == remote_node:
                        has_remote = Int32(1)

                term = cute.make_rmem_tensor(
                    (hidden_per_thread,), cutlass.BFloat16,
                )
                if cutlass.const_expr(token_node_slot is not None):
                    # Slot-indexed landing plane: dispatch_prep recorded this
                    # token's message slot for each remote node it was sent to.
                    # A token without a hit on that node has no valid slot, so
                    # the row is loaded only under has_remote.
                    if has_remote != Int32(0):
                        landing_slot = Int32(
                            token_node_slot[remote_slot, token_idx]
                        )
                        remote_row = cute.zipped_divide(
                            internode_combine[
                                phase, remote_slot, landing_slot, None
                            ],
                            (hidden_per_thread,),
                        )[(None,), (hidden_tile_idx,)]
                        cute.copy(load_atom, remote_row, term)
                else:
                    cute.copy(load_atom, remote_terms[remote_slot, None], term)
                if has_remote != Int32(0):
                    if cutlass.const_expr(self.use_scalar_math):
                        for i in cutlass.range_constexpr(hidden_per_thread):
                            acc[i] = self._fma(
                                Float32(term[i]), Float32(1.0), acc[i],
                            )
                    else:
                        for i in cutlass.range_constexpr(
                            0, hidden_per_thread, 2
                        ):
                            acc[i], acc[i + 1] = self._fma(
                                (Float32(term[i]), Float32(term[i + 1])),
                                Float32(1.0),
                                (acc[i], acc[i + 1]),
                            )

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    out_dtype,
                    num_bits_per_copy=128,
                ),
                out,
                self._mark_alignment(
                    dst, hidden_per_thread * out_dtype.width // 8,
                ),
            )

    @cute.kernel
    def _reduce_bf16(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
    ):
        worker_idx = cute.arch.block_idx()[0] * Int32(self._threads) + cute.arch.thread_idx()[0]
        self._reduce_bf16_cell(
            combine_output, topk_score, reduced_output, worker_idx,
            reduced_output.shape[0], self.require_predicate,
        )

    @cute.kernel
    def _reduce_persistent(
        self,
        combine_quant: cute.Tensor,  # (token, topk, hidden)
        topk_score: Optional[cute.Tensor],  # (token, topk)
        reduced_output: cute.Tensor,  # (token, hidden)
        counters: cute.Tensor,  # int32 x1: work cursor, zeroed by the caller before every launch
        num_tokens_dev: cute.Tensor,  # int32 x1: valid token count
    ):
        """Read the live count and distribute BF16 cells across a fixed grid."""
        lanes = 32
        hidden_tiles = self.hidden_tiles

        tid = cute.arch.thread_idx()[0]
        lane = tid % Int32(lanes)
        warp = tid // Int32(lanes)
        bid = cute.arch.block_idx()[0]

        num_tokens = max(Int32(0), min(Int32(reduced_output.shape[0]), num_tokens_dev[0]))

        num_units = (num_tokens * Int32(hidden_tiles) + Int32(lanes - 1)) // Int32(lanes)

        counter_ptr = counters.iterator

        # Seed one unit per warp, STRIDED. Blocked seeding (warp w takes units
        # [w*fetch, (w+1)*fetch)) idles most warps when num_units is small --
        # exactly the decode case this path exists for.
        total_warps = Int32(self.num_ctas * self.warps_per_cta)
        unit = bid * Int32(self.warps_per_cta) + warp
        preclaimed = total_warps

        rounds = Int32(self.fetch_rounds)
        leftover = num_units - preclaimed
        fetch = (leftover + total_warps * rounds - Int32(1)) // (total_warps * rounds)
        fetch = max(Int32(1), min(Int32(self.max_fetch), fetch))

        res_lo = Int32(0)
        res_hi = Int32(0)

        has_leftover = leftover > Int32(0)  # kernel-uniform

        while unit < num_units:
            # Issue the refill BEFORE this unit's work and consume it after, so
            # the atomic's L2 round trip overlaps the loads/FMAs.
            need_fetch = has_leftover and res_lo >= res_hi  # warp-uniform
            base = Int32(0)
            if need_fetch and lane == Int32(0):
                base = cute.arch.atomic_add(counter_ptr, fetch)

            self._reduce_bf16_cell(
                combine_quant, topk_score, reduced_output,
                unit * Int32(lanes) + lane, num_tokens,
            )

            if need_fetch:
                base = cute.arch.shuffle_sync(base, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31)
                res_lo = preclaimed + base
                res_hi = res_lo + fetch
            if has_leftover:
                unit = res_lo
                res_lo = res_lo + Int32(1)
            else:
                unit = num_units  # seed unit was this warp's only unit


    @cute.jit
    def _reduce_bf16_cell(
        self,
        combine_output: cute.Tensor,
        topk_score: Optional[cute.Tensor],
        reduced_output: cute.Tensor,
        worker_idx: Int32,
        num_tokens: Int32,
        needs_guard: cutlass.Constexpr[bool] = True,
    ):
        hidden_per_thread = self.hidden_per_thread
        hidden_tiles = self.hidden_tiles
        num_topk: cutlass.Constexpr[int] = self.num_topk
        prefetch = self.prefetch
        out_dtype = reduced_output.element_type

        token_idx = worker_idx // hidden_tiles
        hidden_tile_idx = worker_idx % hidden_tiles

        if (not needs_guard) or token_idx < num_tokens:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            terms = cute.zipped_divide(
                combine_output[token_idx, None, None], (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None], (hidden_per_thread,),
            )[(None,), (hidden_tile_idx,)]

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.BFloat16, num_bits_per_copy=128,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            score_dtype = (
                topk_score.dtype
                if cutlass.const_expr(topk_score is not None)
                else cutlass.Float32
            )
            score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor(
                    (hidden_per_thread,), cutlass.BFloat16,
                )
                cute.copy(load_atom, terms[k, None], term)
                if cutlass.const_expr(topk_score is not None and not prefetch):
                    score_reg[k] = topk_score[token_idx, Int32(k)]
                score = Float32(score_reg[k])

                if cutlass.const_expr(self.use_scalar_math):
                    for i in cutlass.range_constexpr(hidden_per_thread):
                        value = Float32(term[i])
                        if cutlass.const_expr(k == 0):
                            if cutlass.const_expr(topk_score is not None):
                                acc[i] = self._fmul(value, score)
                            else:
                                acc[i] = value
                        else:
                            acc[i] = self._fma(value, score, acc[i])
                else:
                    for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                        value_pair = (Float32(term[i]), Float32(term[i + 1]))
                        if cutlass.const_expr(k != 0):
                            acc[i], acc[i + 1] = self._fma(
                                value_pair,
                                score,
                                (acc[i], acc[i + 1]),
                            )
                        else:
                            if cutlass.const_expr(topk_score is not None):
                                acc[i], acc[i + 1] = self._fmul(
                                    value_pair, score,
                                )
                            else:
                                acc[i] = value_pair[0]
                                acc[i + 1] = value_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=128),
                out, self._mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_mxfp8(
        self,
        combine_quant: cute.Tensor,
        combine_sf: cute.Tensor,            # depth-2 broadcast view: logical (token, topk, hidden) e8m0
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

        score_dtype = topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk,), cutlass.Float8E8M0FNU)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None], (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None], (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None], (hidden_per_thread,),
            )[(None,), (hidden_tile_idx,)]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(sf[None, 0], scale_reg)       # one scale per topk slot (stride-0 broadcast)

            fp8_dtype = self.combine_format.act_dtype
            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), fp8_dtype, num_bits_per_copy=128,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread,), fp8_dtype)
                cute.copy(load_atom, codes[k, None], term)
                value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
                value.store(term.load().to(cutlass.Float32))

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                scale = Float32(scale_reg[k])                   # e8m0 -> f32
                score = Float32(score_reg[k])

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = self._fmul(
                        (value[i], value[i + 1]), scale,
                    )
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = self._fma(
                            dequant_pair, score, (acc[i], acc[i + 1]),
                        )
                    else:
                        if cutlass.const_expr(topk_score is not None):
                            acc[i], acc[i + 1] = self._fmul(
                                dequant_pair, score,
                            )
                        else:
                            acc[i] = dequant_pair[0]
                            acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=256),
                out, self._mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )

    @cute.kernel
    def _reduce_fp4(
        self,
        combine_quant: cute.Tensor,         # (token, topk, hidden) e2m1 (logical)
        combine_sf: cute.Tensor,            # depth-2 broadcast view: logical (token, topk, hidden) bf16 amax
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

        score_dtype = topk_score.dtype if cutlass.const_expr(topk_score is not None) else cutlass.Float32
        score_reg = cute.make_rmem_tensor((num_topk,), score_dtype)
        scale_reg = cute.make_rmem_tensor((num_topk,), cutlass.BFloat16)

        if (not needs_guard) or token_idx < reduced_output.shape[0]:
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            codes = cute.zipped_divide(
                combine_quant[token_idx, None, None], (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, topk, hidden) -> (topk, hidden_per_thread)
            sf = cute.zipped_divide(
                combine_sf[token_idx, None, None], (num_topk, hidden_per_thread),
            )[(None, None), (0, hidden_tile_idx)]
            # (token, hidden) -> (hidden_per_thread)
            dst = cute.zipped_divide(
                reduced_output[token_idx, None], (hidden_per_thread,),
            )[(None,), (hidden_tile_idx,)]

            if cutlass.const_expr(topk_score is not None):
                if cutlass.const_expr(prefetch):
                    cute.autovec_copy(topk_score[token_idx, None], score_reg)
            else:
                for k in cutlass.range_constexpr(num_topk):
                    score_reg[k] = score_dtype(1)
            if cutlass.const_expr(prefetch):
                cute.autovec_copy(sf[None, 0], scale_reg)

            load_atom = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN, num_bits_per_copy=64,
            )
            acc = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)

            for k in cutlass.range_constexpr(0, num_topk, 1):
                term = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float4E2M1FN)
                cute.copy(load_atom, codes[k, None], term)
                value = cute.make_rmem_tensor((hidden_per_thread,), cutlass.Float32)
                # Diagnostic decoder choice: manual LUT/PRMT decode vs the HW
                # cvt path, so both SASS forms can be compared on device.
                if cutlass.const_expr(self.f4cvt_use_manual):
                    cvt_e2m1_to_fp32_optimal_ptx(term, value)
                else:
                    cvt_e2m1_to_fp32_cvt_ptx(term, value)

                if cutlass.const_expr(not prefetch):
                    scale_reg[k] = sf[k, 0]
                    if cutlass.const_expr(topk_score is not None):
                        score_reg[k] = topk_score[token_idx, Int32(k)]

                # amax (bf16) -> per-element scale; (1/6) folds the fp4 grid max.
                scale = Float32(scale_reg[k]) * Float32(Nvfp4E2M1RcpLimit)
                score = Float32(score_reg[k])

                for i in cutlass.range_constexpr(0, hidden_per_thread, 2):
                    dequant_pair = self._fmul(
                        (value[i], value[i + 1]), scale,
                    )
                    if cutlass.const_expr(k != 0):
                        acc[i], acc[i + 1] = self._fma(
                            dequant_pair, score, (acc[i], acc[i + 1]),
                        )
                    elif cutlass.const_expr(topk_score is not None):
                        acc[i], acc[i + 1] = self._fmul(
                            dequant_pair, score,
                        )
                    else:
                        acc[i] = dequant_pair[0]
                        acc[i + 1] = dequant_pair[1]

            out = cute.make_rmem_tensor((hidden_per_thread,), out_dtype)
            out.store(acc.load().to(out_dtype))
            cute.copy(
                cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), out_dtype, num_bits_per_copy=256),
                out, self._mark_alignment(dst, hidden_per_thread * out_dtype.width // 8),
            )
