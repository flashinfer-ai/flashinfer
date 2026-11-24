# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Autonomous epilogue for the fused FC1+FC2 swap-AB MegaMoE kernel.

Per-thread RMEM tensors flow between the transpose / SwiGLU / quantize / fc2
store steps as bare ``cute.Tensor`` fragments; their thread distribution is a
fixed physical property of the surrounding atom sequence and is documented in
local comments. FC2 store mappings are finite ``FunctionMapping`` objects
evaluated at runtime to drive metadata lookup and destination pointer math.
"""
from .....helpers import codegen_options as _kimi_options

import dataclasses
import os
from typing import Any, Callable, ClassVar, List, Literal, Optional, Tuple, Type, Union

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64, T

from .....api import (
    ImplDesc,
    KernelComponent,
    OptionalRequirement,
    ProblemDesc,
    StaticOrRuntimeIntegerType,
)
from .....communication.nvlink_domain.symmetric_buffer import SymmetricBufferDevice
from .....communication.token_protocol import TokenSrcMetadata
from .....helpers.constants import Fp8E4M3RcpLimit, Fp8E5M2RcpLimit, Fp32Max, Nvfp4E2M1RcpLimit
from .....helpers.cute_py_helpers import tcgen05_block_scaled_acc_dtype
from .....helpers.dsl_helpers import mark_alignment
from .....helpers.flag_batch import (
    GpuAsyncReleaseFlagBatchTracker,
    GpuFc2AsyncReleaseFlagBatchTracker,
    GpuReleaseFlagBatchTracker,
)
from .....helpers import prod_config as _prod_config  # prod1
from .....helpers.iket_compat import iket
from .....helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
    cvt_f32_to_fp8_to_f32,
    movmatrix_b16,
    red_add_relaxed_sys_v2_bf16x2,
)
from .....helpers.smem_workspace import SmemRegion, SmemWorkspace
from .....quant_def import CombineFormat, QuantKind
from ....function_mapping import CoordinateSpace, FunctionMapping
from ....schedulers.base import SchedulerConsumer
from ....schedulers.fc12_mapping import BlockPhase, SwapAbFc12WorkTileInfo
from .block_scaled_swap_ab_fc12_extension import BlockScaledSwapAbFc12Extension


# Codegen-time epilogue experiment switches. Each defaults to off, which
# reproduces the v4 code path exactly. They are read when the epilogue is constructed (trace time) and
# folded into the kernel name, so toggling one recompiles instead of hitting a stale in-process kernel.
#   MEGAMOE_EPI_FC1_ACT_FIRST=1   FC1: activation on both token halves before one 32x32 TMEM transpose
#                                 (6 STTM + 10 LDTM per subtile instead of 9 + 15; same math, same order).
#   MEGAMOE_EPI_FC2_NO_TILE_DRAIN=1  FC2 (UBLK, no fc2 done flags): skip the full bulk-store drain between
#                                 consecutive FC2 tiles; read-drain before an FC1 tile; full drain at exit.
#   MEGAMOE_EPI_FC2_PREFETCH=1    FC2 (UBLK BF16): issue the next subtile's TMEM load before the current
#                                 subtile's SMEM staging / bulk issue (fully unrolled subtile loop).
#   MEGAMOE_EPI_IKET_FINE=1       extra IKET sub-ranges inside the FC1 / FC2 subtile bodies.
# 
#   MEGAMOE_EPI_FC1_PIPE_FULL=1   as FC1_PIPE, but only for full tiles (valid_tokens == tile N); partial
#                                 tiles run the default loop, so no out-of-range subtile is computed.
#   MEGAMOE_EPI_FC1_PIPE=1        FC1: software-pipelined subtile loop (fully unrolled). Straight-line
#                                 block k holds LDTM(k+1) | TMEM transpose(k) | activation(k+1) |
#                                 quant + STS + TMA store(k-1), so the dependent TMEM transpose chain
#                                 overlaps MUFU/ALU work of the neighbouring subtiles. Same per-element
#                                 math as without it (bit-identical output); pure scheduling change.
# The v2-situ switches (MEGAMOE_SITU_FAST / MEGAMOE_SITU_FOLD_ALPHA / MEGAMOE_FC1_TOPK_FOLD) are also
# folded into the kernel-name tag so an in-process toggle recompiles.
_EPI_OPTION_ENVS = (
    ("fc1_act_first", "MEGAMOE_EPI_FC1_ACT_FIRST", "a"),
    ("fc2_no_tile_drain", "MEGAMOE_EPI_FC2_NO_TILE_DRAIN", "d"),
    ("fc2_prefetch", "MEGAMOE_EPI_FC2_PREFETCH", "p"),
    ("iket_fine", "MEGAMOE_EPI_IKET_FINE", "i"),
    ("fc1_pipe", "MEGAMOE_EPI_FC1_PIPE", "f"),
    ("fc1_pipe_full", "MEGAMOE_EPI_FC1_PIPE_FULL", "F"),
    ("situ_fast", "MEGAMOE_SITU_FAST", "s"),
    ("situ_fold_alpha", "MEGAMOE_SITU_FOLD_ALPHA", "g"),
    ("fc1_topk_fold", "MEGAMOE_FC1_TOPK_FOLD", "t"),
    # FC1 epilogue instruction-count cuts (each bit-identical; see SwapABGatedActEpilogue)
    ("lean_smem_trans", "MEGAMOE_LEAN_SMEM_TRANS", "T"),
    ("lean_qmul2", "MEGAMOE_LEAN_QMUL2", "m"),
    ("lean_amax_tree", "MEGAMOE_LEAN_AMAX_TREE", "x"),
    ("lean_sfidx", "MEGAMOE_LEAN_SFIDX", "k"),
    ("lean_tmem_uniform", "MEGAMOE_LEAN_TMEM_UNIFORM", "u"),
    # remaining epilogue instruction cuts (each bit-identical; see SwapABGatedActEpilogue)
    ("lean_fc2_stsm", "MEGAMOE_LEAN_FC2_STSM", "S"),
    ("lean_fc2_quant", "MEGAMOE_LEAN_FC2_QUANT", "Q"),
    ("lean_fc1_quant", "MEGAMOE_LEAN_FC1_QUANT", "q"),
    ("lean_fc2_addr", "MEGAMOE_LEAN_FC2_ADDR", "A"),
    ("lean_tile", "MEGAMOE_LEAN_TILE_DRAIN", "O"),
    ("lean_sched_fence", "MEGAMOE_LEAN_SCHED_FENCE", "F"),
    # MX (ue8m0 / 32, paired amax exchange) epilogue switches (each bit-identical, see __init__)
    ("mx_lean_t", "MEGAMOE_MX_LEAN_T", "M"),
    # port of v11's MEGAMOE_EPI_ASYNC_FLAG (bit-identical): see __init__
    ("epi_async_flag", "MEGAMOE_EPI_ASYNC_FLAG", "Y"),
    # async release for the FC2 done flags only (bit-identical): on for NVFP4 + quantized combine at N = 240, see __init__
    ("epi_fc2_async_flag", "MEGAMOE_EPI_FC2_ASYNC_FLAG", "Z"),
    # two FC1 output staging stages instead of one per subtile (frees SMEM for an AB stage)
    ("fc1_stage2", "MEGAMOE_MX_FC1_STAGE2", "2"),
)


def epi_options(quant_kind=None, combine_format=None) -> dict:
    """Resolved epilogue switches. With a config (quant kind + combine format) the adopted production
    switches (helpers/prod_config.py) default to that config's adopted value and the env only overrides them; the
    diagnostic / parked switches keep their plain env read (default off). Without a config: env only (legacy)."""
    options = {}
    for key, env, _ in _EPI_OPTION_ENVS:
        if quant_kind is not None and env in _prod_config._ENV_KIND:
            options[key] = bool(_prod_config.flag(env, quant_kind, combine_format))
        else:
            options[key] = _kimi_options.environ.get(env, "0") == "1"
    return options


def epi_options_tag(options: Optional[dict] = None) -> str:
    """Kernel-name tag of the resolved switches (prod1: pass the epilogue's ``epi_options_resolved``)."""
    if options is None:
        options = epi_options()
    tag = "".join(letter for key, _, letter in _EPI_OPTION_ENVS if options[key])
    return f"_epiopt{tag}" if tag else ""


def _fine_push(enabled: bool, name: str) -> None:
    if enabled:
        iket.range_push(name)


def _fine_pop(enabled: bool) -> None:
    if enabled:
        iket.range_pop()


@dataclasses.dataclass(frozen=True)
class QuantImpl:
    """Register-level block quantizer shared by the fc1 / fc2 epilogues.

    Returns ``(data_regs, sf_regs)`` ONLY: the caller pre-multiplies the topk
    weight / global scale into ``prequant_reg`` beforehand and owns the data /
    sf plane stores afterwards. ``sf_vec_direction`` selects the per-block amax
    reduction:

      * ``regs_in_thread``            -- a block is ``sf_vec`` contiguous regs of
        one thread; amax is thread-local (packed bf16x2 abs-max for combine,
        fp32 ``fmax`` for orthodox).
      * ``threads_with_the_same_reg`` -- a block is one reg across ``sf_vec`` warp
        lanes; amax is a warp CREDUX (full warp for vec=32, lane-predicated
        halves for vec=16) and is fp32-only, so bf16 is upconverted first.
      * ``regs_in_pair_threads``      -- a block is ``sf_vec / 2`` regs in each of
        two paired warps; amax is thread-local over the half, then exchanged with
        the partner warp through SMEM. Orthodox mx only: fc1's TMEM transpose
        hands one warp exactly 16 intermediate values per token, so a 32-wide
        scale block necessarily straddles warps ``w`` and ``w ^ 1``.
    ``prequant_reg`` must be 1D with size divisible by the per-thread block share
    (``sf_vec``, or ``sf_vec / 2`` for the paired direction). Combine inputs are
    bf16 (fc2's bf16 reorder regs); orthodox input is fp32 (swiglu).
    """

    quant_kind: Union[QuantKind, CombineFormat]
    sf_vec_direction: Literal["regs_in_thread", "threads_with_the_same_reg", "regs_in_pair_threads"]
    lane_idx: Optional[Any] = None
    warp_idx: Optional[Any] = None
    pair_exchange_barrier: Optional[Any] = None
    # v9-fc1lean (nvfp4 regs_in_thread only; both bit-identical, default off):
    packed_scale: bool = False  # MEGAMOE_LEAN_QMUL2: data scaling as packed f32x2 multiplies
    amax_tree: bool = False  # MEGAMOE_LEAN_AMAX_TREE: abs-max as a tree instead of a chain
    # v14-epi3 (both bit-identical, default off):
    lean_combine: bool = False  # MEGAMOE_LEAN_FC2_QUANT: nvfp4 combine regs_in_thread, see _lean_combine
    lean_nvfp4: bool = False  # MEGAMOE_LEAN_FC1_QUANT: orthodox nvfp4 regs_in_thread, see _lean_nvfp4

    _directions: ClassVar[Tuple[str, ...]] = (
        "regs_in_thread",
        "threads_with_the_same_reg",
        "regs_in_pair_threads",
    )

    # -- config / validation --------------------------------------------------

    def __post_init__(self):
        if isinstance(self.quant_kind, CombineFormat):
            if not self.quant_kind.is_quantized:
                raise ValueError(
                    f"QuantImpl combine path needs a quantized CombineFormat, got {self.quant_kind}."
                )
        elif not isinstance(self.quant_kind, QuantKind):
            raise ValueError(
                f"quant_kind must be a QuantKind or a CombineFormat, got {self.quant_kind!r}."
            )
        if self.sf_vec_direction not in self._directions:
            raise ValueError(
                f"sf_vec_direction must be one of {self._directions}, got {self.sf_vec_direction!r}."
            )
        if self.sf_vec_direction == "threads_with_the_same_reg" and self.lane_idx is None:
            raise ValueError("across-lane quant needs lane_idx for the CREDUX half-warp predicate.")
        if self.sf_vec_direction == "regs_in_pair_threads":
            if isinstance(self.quant_kind, CombineFormat):
                raise ValueError(
                    "The paired direction is orthodox-only; combine blocks never straddle warps."
                )
            if self.sf_vec_size % 2 != 0:
                raise ValueError(
                    f"The paired direction needs an even sf_vec, got {self.sf_vec_size}."
                )
            if self.lane_idx is None or self.warp_idx is None or self.pair_exchange_barrier is None:
                raise ValueError(
                    "The paired direction needs lane_idx, warp_idx and pair_exchange_barrier."
                )
        elif not isinstance(self.quant_kind, CombineFormat) and self.sf_vec_size != 16:
            # A wider orthodox block cannot be reduced inside one thread: see the paired direction.
            raise ValueError(f"Orthodox sf_vec {self.sf_vec_size} needs the paired direction.")

    @property
    def data_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.act_dtype
        # Orthodox output feeds fc2 as its activation, so it is the kind's activation type.
        return self.quant_kind.activation_dtype

    @property
    def scale_dtype(self):
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_dtype
        return self.quant_kind.sf_dtype

    @property
    def sf_vec_size(self) -> int:
        if isinstance(self.quant_kind, CombineFormat):
            return self.quant_kind.scale_block
        return self.quant_kind.sf_vec_size

    @property
    def _is_combine(self) -> bool:
        return isinstance(self.quant_kind, CombineFormat)

    @property
    def _data_rcp_limit(self) -> float:
        # 1 / max representable magnitude of the data element type.
        dt = self.data_dtype
        if dt is cutlass.Float4E2M1FN:
            return Nvfp4E2M1RcpLimit  # 1/6
        if dt is cutlass.Float8E4M3FN:
            return Fp8E4M3RcpLimit  # 1/448
        return Fp8E5M2RcpLimit  # 1/57344

    @property
    def _block_share_per_thread(self) -> int:
        """How many of a block's elements this thread holds."""
        return (
            self.sf_vec_size // 2
            if self.sf_vec_direction == "regs_in_pair_threads"
            else self.sf_vec_size
        )

    # -- dispatch -------------------------------------------------------------

    @cute.jit
    def __call__(
        self, prequant_reg: cute.Tensor, *, norm_const=None, smem_intermediate=None, block_scales=None
    ):
        if cutlass.const_expr(cute.size(prequant_reg) % self._block_share_per_thread != 0):
            raise ValueError(
                "prequant_reg size must be divisible by this thread's share of a block."
            )
        # Combine quantizes fc2's bf16 reorder regs; orthodox the fp32 swiglu.
        expected_in = cutlass.BFloat16 if self._is_combine else cutlass.Float32
        if cutlass.const_expr(prequant_reg.element_type is not expected_in):
            raise TypeError(
                f"QuantImpl({self.quant_kind}) expects {expected_in} prequant input, got {prequant_reg.element_type}."
            )
        needs_smem = cutlass.const_expr(self.sf_vec_direction == "regs_in_pair_threads")
        if cutlass.const_expr(needs_smem != (smem_intermediate is not None)):
            raise ValueError(
                "smem_intermediate must be supplied for the paired direction and only for it."
            )
        if cutlass.const_expr(not self._is_combine):
            if cutlass.const_expr(self.sf_vec_direction == "regs_in_pair_threads"):
                return self.mx_quant_regs_in_pair_threads_impl(prequant_reg, smem_intermediate)
            if cutlass.const_expr(block_scales is not None):
                return self.nvfp4_quant_impl(
                    prequant_reg, norm_const=norm_const, block_scales=block_scales
                )
            return self.nvfp4_quant_impl(prequant_reg, norm_const=norm_const)
        if cutlass.const_expr(self.data_dtype is cutlass.Float4E2M1FN):
            if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
                return self.nvfp4_combine_quant_regs_in_thread_impl(prequant_reg)
            return self.nvfp4_combine_quant_threads_with_the_same_reg_impl(prequant_reg)
        if cutlass.const_expr(self.sf_vec_direction == "regs_in_thread"):
            return self.mxfp8_combine_quant_regs_in_thread_impl(prequant_reg)
        return self.mxfp8_combine_quant_threads_with_the_same_reg_impl(prequant_reg)

    # -- impls ----------------------------------------------------------------

    # regs_in_thread only; fc1 promises the vec direction via its TMEM transpose.
    @cute.jit
    def nvfp4_quant_impl(
        self,
        prequant_reg: cute.Tensor,
        *,
        norm_const: Optional[cutlass.Float32] = None,
        block_scales=None,
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # v2-situ (MEGAMOE_FC1_TOPK_FOLD=1): ``block_scales[b]`` is a per-block multiplier (the
        # token's topk weight) applied as if every input of block b had been pre-multiplied by it.
        # amax(|x*t|) == amax(|x|)*|t| exactly under round-to-nearest (monotone), so the e4m3
        # scale is bit-identical to the pre-multiplied path; the data multiply becomes
        # x * (t * acc_scale) instead of (x * t) * acc_scale (one fp32 rounding moved, before
        # the e2m1 conversion). Saves one FMUL per element.
        # fp32 in -> e2m1 data + e4m3 sfc. Mirrors the prior nvfp4_quant scale
        # math (sfc -> capped/masked acc_scale); topk pre-mult + sf store are the
        # caller's job now.
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        if cutlass.const_expr(
            self.lean_nvfp4
            and block_scales is None
            and n_blocks % 2 == 0
            and self.data_dtype is cutlass.Float4E2M1FN
        ):
            return self._lean_nvfp4(prequant_reg, norm_const)
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.Float8E4M3FN)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        rcp_limit = cutlass.Float32(self._data_rcp_limit)
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            if cutlass.const_expr(self.amax_tree):
                amax = self._amax_thread_fp32_tree(block)
            else:
                amax = self._amax_thread_fp32(block)
            if cutlass.const_expr(block_scales is not None):
                amax = amax * _abs_f32(block_scales[vec_block_idx])
            if cutlass.const_expr(norm_const is not None):
                sfc_fp32 = amax * rcp_limit * norm_const
            else:
                sfc_fp32 = amax * rcp_limit
            sfc_e4m3 = sfc_fp32.to(cutlass.Float8E4M3FN)
            sfc_rt = cutlass.Float32(sfc_e4m3)
            if cutlass.const_expr(norm_const is not None):
                acc_scale = norm_const * cute.arch.rcp_approx(sfc_rt)
            else:
                acc_scale = cute.arch.rcp_approx(sfc_rt)
            acc_scale = cute.arch.fmin(acc_scale, Fp32Max)
            mask = cute.arch.fmin(sfc_rt * cutlass.Float32(1e30), cutlass.Float32(1.0))
            acc_scale = acc_scale * mask
            if cutlass.const_expr(block_scales is not None):
                acc_scale = acc_scale * cutlass.Float32(block_scales[vec_block_idx])
            sf_values.append(sfc_e4m3)
            if cutlass.const_expr(self.packed_scale):
                data_blocks.append(self._scale_to_data_packed_ssa(block, acc_scale))
            else:
                data_blocks.append(self._scale_to_data_ssa(block, acc_scale))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E4M3FN))
        return data, sf

    @cute.jit
    def mx_quant_regs_in_pair_threads_impl(
        self,
        prequant_reg: cute.Tensor,  # (token_2_intermeidate_x)
        smem_intermediate: cute.Tensor,  # (token_blocks, epi_threads)
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        half = self._block_share_per_thread
        n_blocks = cute.size(prequant_reg) // half
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), self.data_dtype)
        sf = cute.make_rmem_tensor((n_blocks,), self.scale_dtype)
        in_blocks = cute.zipped_divide(prequant_reg, (half,))  # ((half,), (n_blocks,))

        if cutlass.const_expr(cute.size(smem_intermediate, mode=[0]) != n_blocks):
            raise ValueError(
                f"The paired amax exchange needs {n_blocks} rows, got {cute.size(smem_intermediate, mode=[0])}."
            )
        if cutlass.const_expr(smem_intermediate.stride[0] != 1):
            # Thread-major would still be correct but would silently split the exchange into
            # per-block scalar accesses.
            raise ValueError(
                "The paired amax exchange must be block-major to stay one vector access."
            )

        exchange_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            cutlass.Float32,
            num_bits_per_copy=n_blocks * cutlass.Float32.width,
        )
        # Which warps pair up is epilogue knowledge: warp w owns one contiguous run of
        # intermediate outputs, so a block that is twice that run wide joins adjacent warps.
        partner_warp = self.warp_idx ^ cutlass.Int32(1)
        own_thread = self.warp_idx * cutlass.Int32(32) + self.lane_idx
        partner_thread = partner_warp * cutlass.Int32(32) + self.lane_idx

        half_amax = cute.make_rmem_tensor((n_blocks,), cutlass.Float32)
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            half_amax[vec_block_idx] = self._amax_thread_fp32(in_blocks[None, vec_block_idx])
        cute.copy(
            exchange_atom,
            cute.coalesce(half_amax),
            cute.coalesce(smem_intermediate[None, own_thread]),
        )

        self.pair_exchange_barrier.arrive_and_wait()

        partner_amax = cute.make_rmem_tensor((n_blocks,), cutlass.Float32)
        cute.copy(
            exchange_atom,
            cute.coalesce(smem_intermediate[None, partner_thread]),
            cute.coalesce(partner_amax),
        )

        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block_amax = cute.arch.fmax(half_amax[vec_block_idx], partner_amax[vec_block_idx])
            scale_e8m0, scale_f32 = self._e8m0(block_amax)
            sf_values.append(scale_e8m0)
            data_blocks.append(
                self._scale_to_data_ssa(in_blocks[None, vec_block_idx], self._enc_mxfp8(scale_f32))
            )
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, self.scale_dtype))
        return data, sf

    @cute.jit
    def nvfp4_combine_quant_regs_in_thread_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        if cutlass.const_expr(
            self.lean_combine
            and self.sf_vec_size == 16
            and (cute.size(prequant_reg) // self.sf_vec_size) % 2 == 0
        ):
            return self._lean_combine(prequant_reg)
        # bf16 in -> e2m1 data + per-16 bf16 amax. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.BFloat16)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            amax = self._amax_thread_bf16(block)
            sf_values.append(amax)
            decode_scale = cutlass.Float32(amax) * cutlass.Float32(self._data_rcp_limit)
            data_blocks.append(self._scale_to_data_ssa(block, self._enc_nvfp4(decode_scale)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.BFloat16))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx % 16 + selected_sf_idx * 16
    # hidden_idx = lane_idx // 16 * 16
    @cute.jit
    def nvfp4_combine_quant_threads_with_the_same_reg_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # UBLK has lane == hidden, so the warp's 32 lanes are 32 consecutive
        # hidden. A scale block = sf_vec hidden, so the lanes split along hidden
        # into 32 // sf_vec blocks of sf_vec lanes each (warp = blocks_per_warp *
        # lanes_per_block, the EP x TP split). Only the sf_vec lanes inside a
        # block share its CREDUX scale, so they pool the subtile tokens: sf_vec
        # == 32 pools the whole warp, sf_vec < 32 pools fewer (more per lane).
        lanes_per_block = self.sf_vec_size
        n_tokens = cute.size(prequant_reg)
        lane_in_block = self.lane_idx % cutlass.Int32(lanes_per_block)
        data = cute.make_rmem_tensor((n_tokens,), cutlass.Float4E2M1FN)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block,), cutlass.BFloat16)
        scaled_vec = cute.full((n_tokens,), cutlass.Float32(0.0), cutlass.Float32)
        for token_idx in cutlass.range_constexpr(n_tokens):
            value = cutlass.Float32(prequant_reg[token_idx])
            amax_bf16 = self._amax_lane(value).to(cutlass.BFloat16)
            slot = token_idx // lanes_per_block
            if (token_idx % lanes_per_block) == lane_in_block:
                selected_sf[slot] = amax_bf16
            else:
                selected_sf[slot] = selected_sf[slot]
            decode_scale = cutlass.Float32(amax_bf16) * cutlass.Float32(self._data_rcp_limit)
            scaled_value = value * self._enc_nvfp4(decode_scale)
            scaled_vec = cute.TensorSSA(
                vector.insert(scaled_value.ir_value(), scaled_vec.ir_value(), [], [token_idx]),
                (n_tokens,),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(cutlass.Float4E2M1FN))
        return data, selected_sf

    @cute.jit
    def mxfp8_combine_quant_regs_in_thread_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # bf16 in -> e4m3/e5m2 data + per-32 e8m0. amax found on bf16 (packed).
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), self.data_dtype)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.Float8E8M0FNU)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        data_blocks = []
        sf_values = []
        for vec_block_idx in cutlass.range_constexpr(n_blocks):
            block = in_blocks[None, vec_block_idx]
            # widen the native-bf16 amax to fp32 for the e8m0 round-up math.
            scale_e8m0, scale_f32 = self._e8m0(cutlass.Float32(self._amax_thread_bf16(block)))
            sf_values.append(scale_e8m0)
            data_blocks.append(self._scale_to_data_ssa(block, self._enc_mxfp8(scale_f32)))
        self._store_packed_blocks(data, data_blocks)
        sf.store(self._values_to_ssa(sf_values, cutlass.Float8E8M0FNU))
        return data, sf

    # Mapping: (lane_idx, selected_sf_idx) -> (token_64, hidden_32)
    # token_idx = lane_idx + selected_sf_idx * 32
    # hidden_idx = 0
    @cute.jit
    def mxfp8_combine_quant_threads_with_the_same_reg_impl(
        self, prequant_reg: cute.Tensor
    ) -> Tuple[cute.Tensor, cute.Tensor]:
        # UBLK has lane == hidden, so the warp's 32 lanes are 32 consecutive
        # hidden. A scale block = sf_vec hidden, so the lanes split along hidden
        # into 32 // sf_vec blocks of sf_vec lanes each (warp = blocks_per_warp *
        # lanes_per_block, the EP x TP split). Only the sf_vec lanes inside a
        # block share its CREDUX scale, so they pool the subtile tokens. mxfp8
        # sf_vec == 32 -> the whole warp is one block, all 32 lanes pool.
        lanes_per_block = self.sf_vec_size
        n_tokens = cute.size(prequant_reg)
        lane_in_block = self.lane_idx % cutlass.Int32(lanes_per_block)
        data = cute.make_rmem_tensor((n_tokens,), self.data_dtype)
        selected_sf = cute.make_rmem_tensor((n_tokens // lanes_per_block,), cutlass.Float8E8M0FNU)
        scaled_vec = cute.full((n_tokens,), cutlass.Float32(0.0), cutlass.Float32)
        for token_idx in cutlass.range_constexpr(n_tokens):
            value = cutlass.Float32(prequant_reg[token_idx])
            scale_e8m0, scale_f32 = self._e8m0(self._amax_lane(value))
            slot = token_idx // lanes_per_block
            if (token_idx % lanes_per_block) == lane_in_block:
                selected_sf[slot] = scale_e8m0
            else:
                selected_sf[slot] = selected_sf[slot]
            scaled_value = value * self._enc_mxfp8(scale_f32)
            scaled_vec = cute.TensorSSA(
                vector.insert(scaled_value.ir_value(), scaled_vec.ir_value(), [], [token_idx]),
                (n_tokens,),
                cutlass.Float32,
            )
        self._store_packed_data(data, scaled_vec.to(self.data_dtype))
        return data, selected_sf

    # -- v14-epi3 lean quantizers ----------------------------------------------
    #
    # _lean_combine (MEGAMOE_LEAN_FC2_QUANT), same values as nvfp4_combine_quant_regs_in_thread_impl:
    #   * bf16x2 word -> two f32: lo = w << 16, hi = w & 0xffff0000 (cvt.f32.bf16 is exact; the
    #     default lowering spends PRMT + IMAD per high half).
    #   * block abs-max over the 16 exact f32 values as max3(|a|, |b|, |c|) (FMNMX3): max of the
    #     magnitudes is order-free and NaN-ignoring like max.xorsign.abs.bf16x2, so it equals the f32
    #     image of the masked bf16 amax; its upper 16 bits are that bf16 (the wire scale).
    #   * decode = amax * (1/6) and the enc / mask products of two blocks as mul.rn.f32x2 (lane-wise
    #     identical to the scalar mul.f32), rcp.approx.ftz / fmin unchanged; data = x * enc as before.
    #   * the two bf16 scales of a block pair are one PRMT (upper halves of the two f32 amaxes).

    @cute.jit
    def _lean_combine(self, prequant_reg: cute.Tensor) -> Tuple[cute.Tensor, cute.Tensor]:
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        words = cute.recast_tensor(prequant_reg, cutlass.Int32)  # block b = words[8b : 8b + 8]
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.BFloat16)
        sf_words = cute.recast_tensor(sf, cutlass.Int32)
        rcp_limit = cutlass.Float32(self._data_rcp_limit)
        data_blocks = []
        for pair in cutlass.range_constexpr(n_blocks // 2):
            vals = []
            amaxes = []
            for b in cutlass.range_constexpr(2):
                block_vals = []
                for i in cutlass.range_constexpr(vec // 2):
                    w = cutlass.Int32(words[(2 * pair + b) * (vec // 2) + i])
                    block_vals.append(_i32_as_f32(w << cutlass.Int32(16)))
                    block_vals.append(_i32_as_f32(w & cutlass.Int32(-65536)))
                vals.append(block_vals)
                amaxes.append(_abs_max_f32(block_vals))
            d0, d1 = cute.arch.mul_packed_f32x2((amaxes[0], amaxes[1]), (rcp_limit, rcp_limit))
            big = cutlass.Float32(1e30)
            m0, m1 = cute.arch.mul_packed_f32x2((d0, d1), (big, big))
            r0 = cute.arch.fmin(cute.arch.rcp_approx(d0), Fp32Max)
            r1 = cute.arch.fmin(cute.arch.rcp_approx(d1), Fp32Max)
            m0 = cute.arch.fmin(m0, cutlass.Float32(1.0))
            m1 = cute.arch.fmin(m1, cutlass.Float32(1.0))
            e0, e1 = cute.arch.mul_packed_f32x2((r0, r1), (m0, m1))
            encs = (e0, e1)
            for b in cutlass.range_constexpr(2):
                scaled = cute.make_rmem_tensor((vec,), cutlass.Float32)
                for i in cutlass.range_constexpr(0, vec, 2):
                    p0, p1 = cute.arch.mul_packed_f32x2(
                        (vals[b][i], vals[b][i + 1]), (encs[b], encs[b])
                    )
                    scaled[i] = p0
                    scaled[i + 1] = p1
                data_blocks.append(scaled.load().to(self.data_dtype))
            sf_words[pair] = _prmt_b32(
                _f32_as_i32(amaxes[0]), _f32_as_i32(amaxes[1]), 0x7632
            )
        self._store_packed_blocks(data, data_blocks)
        return data, sf

    # _lean_nvfp4 (MEGAMOE_LEAN_FC1_QUANT), same values as nvfp4_quant_impl (no block_scales):
    #   * abs-max as FMNMX3 over |x| (equals the masked max.xorsign.abs chain, NaN-ignoring alike);
    #   * the two blocks' amax * (1/6) [* norm] and the 1e30 mask / enc products as mul.rn.f32x2;
    #   * both UE4M3 scales from one cvt.rn.satfinite.e4m3x2.f32 (low byte = block 0) and read back by one
    #     cvt.rn.f16x2.e4m3x2 + two exact f16 -> f32 (the scalar path issues the same cvts per block).

    @cute.jit
    def _lean_nvfp4(self, prequant_reg: cute.Tensor, norm_const) -> Tuple[cute.Tensor, cute.Tensor]:
        vec = self.sf_vec_size
        n_blocks = cute.size(prequant_reg) // vec
        data = cute.make_rmem_tensor((cute.size(prequant_reg),), cutlass.Float4E2M1FN)
        sf = cute.make_rmem_tensor((n_blocks,), cutlass.Float8E4M3FN)
        sf_pairs = cute.recast_tensor(sf, cutlass.Int16)
        in_blocks = cute.zipped_divide(prequant_reg, (vec,))  # ((vec,), (n_blocks,))
        rcp_limit = cutlass.Float32(self._data_rcp_limit)
        big = cutlass.Float32(1e30)
        data_blocks = []
        for pair in cutlass.range_constexpr(n_blocks // 2):
            blocks = (in_blocks[None, 2 * pair], in_blocks[None, 2 * pair + 1])
            a0 = _abs_max_f32([blocks[0][i] for i in range(vec)])
            a1 = _abs_max_f32([blocks[1][i] for i in range(vec)])
            s0, s1 = cute.arch.mul_packed_f32x2((a0, a1), (rcp_limit, rcp_limit))
            if cutlass.const_expr(norm_const is not None):
                n = cutlass.Float32(norm_const)
                s0, s1 = cute.arch.mul_packed_f32x2((s0, s1), (n, n))
            sf_bits, rt0, rt1 = _cvt_e4m3x2_roundtrip(s0, s1)
            sf_pairs[pair] = sf_bits
            r0 = cute.arch.rcp_approx(rt0)
            r1 = cute.arch.rcp_approx(rt1)
            if cutlass.const_expr(norm_const is not None):
                r0 = norm_const * r0
                r1 = norm_const * r1
            r0 = cute.arch.fmin(r0, Fp32Max)
            r1 = cute.arch.fmin(r1, Fp32Max)
            m0, m1 = cute.arch.mul_packed_f32x2((rt0, rt1), (big, big))
            m0 = cute.arch.fmin(m0, cutlass.Float32(1.0))
            m1 = cute.arch.fmin(m1, cutlass.Float32(1.0))
            e0, e1 = cute.arch.mul_packed_f32x2((r0, r1), (m0, m1))
            encs = (e0, e1)
            for b in cutlass.range_constexpr(2):
                # Explicit f32x2 (= MEGAMOE_LEAN_QMUL2's multiply): same per-element IEEE products.
                data_blocks.append(self._scale_to_data_packed_ssa(blocks[b], encs[b]))
        self._store_packed_blocks(data, data_blocks)
        return data, sf

    # -- shared sub-steps -----------------------------------------------------

    @cute.jit
    def _scale_to_data_ssa(self, block: cute.Tensor, enc: cutlass.Float32) -> cute.TensorSSA:
        block_f32 = block.load().to(cutlass.Float32)
        enc_vec = cute.full_like(block_f32, enc, cutlass.Float32)
        return (block_f32 * enc_vec).to(self.data_dtype)

    @cute.jit
    def _scale_to_data_packed_ssa(self, block: cute.Tensor, enc: cutlass.Float32) -> cute.TensorSSA:
        """v9-fc1lean: same per-element IEEE multiply as _scale_to_data_ssa, issued as f32x2 pairs."""
        n: cutlass.Constexpr[int] = cute.size(block)
        scaled = cute.make_rmem_tensor((n,), cutlass.Float32)
        enc_pair = (cutlass.Float32(enc), cutlass.Float32(enc))
        for i in cutlass.range_constexpr(0, n, 2):
            p = cute.arch.mul_packed_f32x2(
                (cutlass.Float32(block[i]), cutlass.Float32(block[i + 1])), enc_pair
            )
            scaled[i] = p[0]
            scaled[i + 1] = p[1]
        return scaled.load().to(self.data_dtype)

    @cute.jit
    def _concat_blocks_ssa(self, blocks, dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            block = blocks[block_idx]
            for elem_idx in cutlass.range_constexpr(cute.size(block.shape)):
                values.append(block[elem_idx].ir_value())
        vec = vector.from_elements(T.vector(len(values), dtype.mlir_type), values)
        return cute.TensorSSA(vec, (len(values),), dtype)

    @cute.jit
    def _values_to_ssa(self, values, dtype: Type[cutlass.Numeric]) -> cute.TensorSSA:
        vec = vector.from_elements(
            T.vector(len(values), dtype.mlir_type),
            [values[i].ir_value() for i in range(len(values))],
        )
        return cute.TensorSSA(vec, (len(values),), dtype)

    @cute.jit
    def _concat_i32_blocks_ssa(self, blocks) -> cute.TensorSSA:
        values = []
        for block_idx in cutlass.range_constexpr(len(blocks)):
            packed_block = blocks[block_idx].bitcast(cutlass.Int32)
            for elem_idx in cutlass.range_constexpr(cute.size(packed_block.shape)):
                values.append(packed_block[elem_idx].ir_value())
        vec = vector.from_elements(T.vector(len(values), cutlass.Int32.mlir_type), values)
        return cute.TensorSSA(vec, (len(values),), cutlass.Int32)

    @cute.jit
    def _store_packed_blocks(self, data: cute.Tensor, blocks) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(self._concat_i32_blocks_ssa(blocks))

    @cute.jit
    def _store_packed_data(self, data: cute.Tensor, data_ssa: cute.TensorSSA) -> None:
        packed_data = cute.recast_tensor(data, cutlass.Int32)
        packed_data.store(data_ssa.bitcast(cutlass.Int32))

    @cute.jit
    def _amax_thread_fp32(self, block: cute.Tensor) -> cutlass.Float32:
        # max.xorsign.abs reduces |.| in one op per element; the result sign is
        # the xor of the inputs (junk for an amax), so clear it at the end.
        def max_abs(lhs: cutlass.Float32, rhs: cutlass.Float32) -> cutlass.Float32:
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [cutlass.Float32(lhs).ir_value(), cutlass.Float32(rhs).ir_value()],
                    "max.xorsign.abs.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        acc = block[0]
        for elem_idx in cutlass.range_constexpr(1, cute.size(block)):
            acc = max_abs(acc, block[elem_idx])
        mag_bits = cutlass.Int32(
            llvm.bitcast(T.i32(), cutlass.Float32(acc).ir_value())
        ) & cutlass.Int32(0x7FFFFFFF)
        return cutlass.Float32(llvm.bitcast(T.f32(), mag_bits.ir_value()))

    def _amax_thread_fp32_tree(self, block: cute.Tensor) -> cutlass.Float32:
        """v9-fc1lean: _amax_thread_fp32 as a balanced tree. max(|.|) is associative and commutative
        (NaN inputs are ignored either way, all-NaN gives NaN), and the junk sign is cleared, so the
        result bits are identical; the dependent chain drops from 15 to 4. Plain Python (no
        @cute.jit) so the trace-time while / comprehensions are not turned into device loops."""

        def max_abs(lhs: cutlass.Float32, rhs: cutlass.Float32) -> cutlass.Float32:
            return cutlass.Float32(
                llvm.inline_asm(
                    T.f32(),
                    [cutlass.Float32(lhs).ir_value(), cutlass.Float32(rhs).ir_value()],
                    "max.xorsign.abs.f32 $0, $1, $2;",
                    "=f,f,f",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        level = [block[i] for i in range(cute.size(block))]
        while len(level) > 1:
            nxt = [max_abs(level[i], level[i + 1]) for i in range(0, len(level) - 1, 2)]
            if len(level) % 2 == 1:
                nxt.append(level[-1])
            level = nxt
        acc = level[0]
        mag_bits = cutlass.Int32(
            llvm.bitcast(T.i32(), cutlass.Float32(acc).ir_value())
        ) & cutlass.Int32(0x7FFFFFFF)
        return cutlass.Float32(llvm.bitcast(T.f32(), mag_bits.ir_value()))

    @cute.jit
    def _amax_thread_bf16(self, block: cute.Tensor) -> cutlass.BFloat16:
        # Packed bf16x2 abs-max: tree-reduce the pairs, then fold the survivor's
        # two halves (high shifted into low). max.xorsign.abs leaves a junk sign,
        # so the low bf16 is masked before being read back. The amax is natively
        # bf16 -- exactly what the wire format stores.
        def max_abs(lhs: cutlass.Int32, rhs: cutlass.Int32) -> cutlass.Int32:
            return cutlass.Int32(
                llvm.inline_asm(
                    T.i32(),
                    [cutlass.Int32(lhs).ir_value(), cutlass.Int32(rhs).ir_value()],
                    "max.xorsign.abs.bf16x2 $0, $1, $2;",
                    "=r,r,r",
                    has_side_effects=False,
                    is_align_stack=False,
                    asm_dialect=llvm.AsmDialect.AD_ATT,
                )
            )

        pairs = cute.recast_tensor(block, cutlass.Int32)  # (vec/2,) bf16x2
        acc = cutlass.Int32(pairs[0])
        for pair_idx in cutlass.range_constexpr(1, cute.size(pairs)):
            acc = max_abs(acc, pairs[pair_idx])
        acc = max_abs(acc, acc >> cutlass.Int32(16))
        amax_bits = cute.make_rmem_tensor((1,), cutlass.Int32)
        amax_bits[0] = acc & cutlass.Int32(0x7FFF)
        return cute.recast_tensor(amax_bits, cutlass.BFloat16)[0]

    @cute.jit
    def _amax_lane(self, v: cutlass.Float32) -> cutlass.Float32:
        if cutlass.const_expr(self.sf_vec_size == 32):
            return cute.arch.warp_redux_sync(v, "fmax", abs=True)
        first_half = (self.lane_idx % cutlass.Int32(32)) < cutlass.Int32(16)
        vsel = cutlass.Float32(0.0)
        if first_half:
            vsel = v
        amax = cute.arch.warp_redux_sync(vsel, "fmax", abs=True)
        if not first_half:
            amax = cute.arch.warp_redux_sync(v, "fmax", abs=True)
        return amax

    @cute.jit
    def _e8m0(self, amax: cutlass.Float32) -> Tuple[cutlass.Float8E8M0FNU, cutlass.Float32]:
        candidate = amax * cutlass.Float32(self._data_rcp_limit)
        scale_f32 = cutlass.Float32(cvt_f32_to_fp8_to_f32(candidate, cutlass.Float8E8M0FNU))
        return scale_f32.to(cutlass.Float8E8M0FNU), scale_f32

    @cute.jit
    def _enc_nvfp4(self, decode_scale: cutlass.Float32) -> cutlass.Float32:
        # rcp.approx.ftz with the fc1 cap+mask idiom (amax==0 -> 0, no inf*0 NaN).
        enc = cute.arch.fmin(cute.arch.rcp_approx(decode_scale), Fp32Max)
        mask = cute.arch.fmin(decode_scale * cutlass.Float32(1e30), cutlass.Float32(1.0))
        return enc * mask

    @cute.jit
    def _enc_mxfp8(self, scale_f32: cutlass.Float32) -> cutlass.Float32:
        # Skip nan
        enc = cute.arch.fmin(cute.arch.rcp_approx(scale_f32), Fp32Max)
        mask = cute.arch.fmin(scale_f32 * cutlass.Float32(1e30), cutlass.Float32(1.0))
        return enc * mask


# =============================================================================
# Region tag
# =============================================================================


class Region:
    """Codegen-time region tag for a 16x32 sub-region within a 32x32 tile."""

    Top = 0
    Bottom = 1


# =============================================================================
# TmemTranspose16x32
# =============================================================================


class _TmemTranspose16x32Core:
    """Physical implementation of the 16x32 -> 32x16 TMEM in-place transpose.

    The transpose is a fixed sequence of tcgen05 32-bit element atoms; each
    32-bit slot is an fp32 SwiGLU-fold value for FC1. The (thread, reg) ->
    (tmem_dp, tmem_col) input / output mapping is documented on the
    ``TmemTranspose16x32`` subclass, which is the public entry point.

    Per-thread RMEM coordinate convention:

      - ``lane_idx`` -- warp lane id (= thread index within warp), in [0, 32).
      - ``elem_idx`` -- per-thread reg index, in [0, 16).
    """

    _PermR1 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)
    _PermR3 = (0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15)
    _PermR4 = (0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15)

    _TmemRowStride = 1 << 16
    _io_dtype = cutlass.Float32

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((_TmemTranspose16x32Core._TmemRowStride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor, num_regs: int, offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset, cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),))
        )

    @staticmethod
    def load_subtile_raw_acc(
        tmem_subtile_tensor: cute.Tensor,
    ) -> Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor]:
        """LDTM the entire 32-lane x 64-col raw acc region of one epi
        subtile into 4 independent (16,) fp32 RMEM tensors.

        Used by the FC1 overlap-acc unroll path to extract all raw acc data
        of the first 2 subtiles up front, so that the acc TMEM can be released
        after the first subtile's 4 LDTMs.

        ``tmem_subtile_tensor`` is the (32 lanes, 64 cols) view onto a
        single epi subtile's acc TMEM region (already offset by
        ``warp_lane_offset + acc_stage_col_offset + subtile_col_offset``;
        see ``SwapABGatedActEpilogue._subtile_local_tmem_tensor``).

        Returns a 4-tuple of (16,) fp32 RMEM tensors carrying the FC1 raw
        LDTM distribution:

          [0] gate_lo / first-half top   -- subtile cols 0..31, lanes 0..15
          [1] up_lo   / first-half bot   -- subtile cols 0..31, lanes 16..31
          [2] raw_top / second-half top  -- subtile cols 32..63, lanes 0..15
          [3] raw_bot / second-half bot  -- subtile cols 32..63, lanes 16..31

        4 atom calls of ``Ld16x64bOp(Repetition.x16) Float32`` -- the same
        atom used by the per-subtile entry LDTM.  Each output is in the
        raw-LDTM input distribution consumed by ``TmemTranspose16x32``.
        """
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), _TmemTranspose16x32Core._io_dtype
        )

        ptr = tmem_subtile_tensor.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride

        # 4 source 16-lane x 32-col views over the (32, 64) subtile region:
        #   first  half (cols 0..31): top  lanes 0..15  / bot lanes 16..31
        #   second half (cols 32..63): top lanes 0..15  / bot lanes 16..31
        # All offsets are Python ints (compile-time const) so cute can
        # const-fold them and infer the correct (>= 8 B / 2 col) ptr
        # alignment that the LDTM atom requires.  Using ``cutlass.Int32``
        # offsets here would wrap them as SSA values that cute treats as
        # alignment-unknown, tripping the atom's verifier.
        first_top_view = cute.make_tensor(ptr, _TmemTranspose16x32Core._tmem_layout(16, 32))
        first_bot_view = cute.make_tensor(
            ptr + half_lane_off, _TmemTranspose16x32Core._tmem_layout(16, 32)
        )
        second_top_view = cute.make_tensor(ptr + 32, _TmemTranspose16x32Core._tmem_layout(16, 32))
        second_bot_view = cute.make_tensor(
            ptr + 32 + half_lane_off, _TmemTranspose16x32Core._tmem_layout(16, 32)
        )

        first_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        first_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_top = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)
        second_bot = cute.make_rmem_tensor((16,), _TmemTranspose16x32Core._io_dtype)

        cute.copy(
            atom_ld16x64, first_top_view, _TmemTranspose16x32Core._rmem_copy_view(first_top, 16)
        )
        cute.copy(
            atom_ld16x64, first_bot_view, _TmemTranspose16x32Core._rmem_copy_view(first_bot, 16)
        )
        cute.copy(
            atom_ld16x64, second_top_view, _TmemTranspose16x32Core._rmem_copy_view(second_top, 16)
        )
        cute.copy(
            atom_ld16x64, second_bot_view, _TmemTranspose16x32Core._rmem_copy_view(second_bot, 16)
        )

        return (first_top, first_bot, second_top, second_bot)

    def __init__(self, tmem_ptr, region: int, reg_tensor: Optional[cute.Tensor] = None) -> None:
        # The whole transpose is built from 32-bit element atoms; _io_dtype
        # drives _src_regs / output / every LDTM/STTM atom below, so guard the
        # invariant once here (tautological today, defensive against future
        # dtype edits).
        if cutlass.const_expr(self._io_dtype.width != 32):
            raise TypeError(
                f"{type(self).__name__} requires a 32-bit _io_dtype (the "
                f"transpose uses 32-bit element atoms), got {self._io_dtype} "
                f"(width {self._io_dtype.width})."
            )

        half_lane_off = 16 * self._TmemRowStride
        if region == Region.Top:
            src_ptr = tmem_ptr
            dst_ptr = tmem_ptr
        elif region == Region.Bottom:
            src_ptr = tmem_ptr + half_lane_off
            dst_ptr = tmem_ptr + 16
        else:
            raise ValueError("region must be Region.Top or Region.Bottom")

        self.region = region

        self._tmem_src_full = cute.make_tensor(src_ptr, self._tmem_layout(16, 32))
        self._tmem_dst_full = cute.make_tensor(dst_ptr, self._tmem_layout(32, 16))
        self._tmem_dst_top = cute.make_tensor(dst_ptr, self._tmem_layout(16, 16))
        self._tmem_dst_bot = cute.make_tensor(dst_ptr + half_lane_off, self._tmem_layout(16, 16))

        self._atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_st16x128 = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), self._io_dtype
        )
        self._atom_st32x32 = cute.make_copy_atom(
            tcgen05.St32x32bOp(tcgen05.Repetition.x16), self._io_dtype
        )
        self._atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), self._io_dtype
        )
        self._atom_ld16x128 = cute.make_copy_atom(
            tcgen05.Ld16x128bOp(tcgen05.Repetition.x4), self._io_dtype
        )

        self._src_regs = cute.make_rmem_tensor((16,), self._io_dtype)
        # ``output`` is a bare (16,) RMEM fragment; its (lane_idx, elem_idx)
        # distribution after all four rounds is the transpose output mapping
        # documented on ``TmemTranspose16x32``.
        self.output = cute.make_rmem_tensor((16,), self._io_dtype)

        # skip-R1.Load mode: ``reg_tensor`` must already be in the transpose
        # input distribution (see ``TmemTranspose16x32`` / produced by
        # ``load_subtile_raw_acc``); we copy it in lieu of the R1 LDTM.
        # Weak entry guard (replaces the removed input contract): the transpose
        # atoms are 32-bit element atoms over exactly 16 regs/lane, so the fed
        # tensor must be a 32-bit element type of size 16.
        self._reg_tensor = reg_tensor
        if reg_tensor is not None:
            if cutlass.const_expr(reg_tensor.element_type.width != 32):
                raise TypeError(
                    f"{type(self).__name__} reg_tensor must be a 32-bit element "
                    f"type, got element type "
                    f"{reg_tensor.element_type} (width {reg_tensor.element_type.width})."
                )
            if cutlass.const_expr(cute.size(reg_tensor) != 16):
                raise ValueError(
                    f"{type(self).__name__} reg_tensor must hold exactly 16 elements, got {cute.size(reg_tensor)}."
                )
            for r in range(16):
                self._src_regs[r] = reg_tensor[r]

    # -- R1 ------------------------------------------------------------------

    def r1_load(self) -> None:
        """LDTM src region -> ``_src_regs``.  No-op in skip-R1.Load mode."""
        if self._reg_tensor is not None:
            return
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r1_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR1[r]]

    def r1_store(self) -> None:
        cute.copy(self._atom_st16x128, self._rmem_copy_view(self.output, 16), self._tmem_src_full)

    # -- R2 ------------------------------------------------------------------

    def r2_load(self) -> None:
        cute.copy(self._atom_ld16x64, self._tmem_src_full, self._rmem_copy_view(self._src_regs, 16))

    def r2_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self._src_regs, 16), self._tmem_dst_full)

    # -- R3 ------------------------------------------------------------------

    def r3_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r3_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x256,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r3_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR3[r]]

    def r3_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

    # -- R4 ------------------------------------------------------------------

    def r4_load_top(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_top,
            self._rmem_copy_view(self._src_regs, 8, offset=0),
        )

    def r4_load_bot(self) -> None:
        cute.copy(
            self._atom_ld16x128,
            self._tmem_dst_bot,
            self._rmem_copy_view(self._src_regs, 8, offset=8),
        )

    def r4_perm(self) -> None:
        for r in range(16):
            self.output[r] = self._src_regs[self._PermR4[r]]

    def r4_store(self) -> None:
        cute.copy(self._atom_st32x32, self._rmem_copy_view(self.output, 16), self._tmem_dst_full)

    def from_r1_perm_until_last_store(self) -> cute.Tensor:
        self.r1_perm()
        self.r1_store()
        self.r2_load()
        self.r2_store()
        self.r3_load_top()
        self.r3_load_bot()
        self.r3_perm()
        self.r3_store()
        self.r4_load_top()
        self.r4_load_bot()
        self.r4_perm()
        return self.output


class TmemTranspose16x32(_TmemTranspose16x32Core):
    """FC1 16x32 -> 32x16 TMEM in-place transpose.

    The per-thread RMEM ``(lane_idx, elem_idx) -> (tmem_dp, tmem_col)`` mapping
    is fixed by the underlying atom sequence. Each slot is an fp32 SwiGLU-fold
    value and ``tmem_col`` is the intermediate-output index.

    Input distribution -- what each (lane_idx, elem_idx) reg holds on entry
    (i.e. straight after the 16-dp x 32-col source LDTM, or as fed in via
    ``reg_tensor`` / ``load_subtile_raw_acc`` for skip-R1.Load mode):

        tmem_dp  = elem_idx * 2 + (lane_idx // 2) % 2          # in [0, 32)
        tmem_col = (lane_idx % 2) * 8 + lane_idx // 4          # in [0, 16)

    Output distribution -- after all four rounds, the 32-dp x 16-col result has
    each lane owning one full dp-row of 16 cols:

        tmem_dp  = lane_idx                                    # in [0, 32)
        tmem_col = elem_idx                                    # in [0, 16)
    """


# =============================================================================
# TmemTranspose32x32Inplace
# =============================================================================


class TmemTranspose32x32Inplace:
    """fc1 epi 32x32 in-place TMEM transpose: two ``TmemTranspose16x32``
    sub-instances (``top`` = lanes 0..15, ``bot`` = lanes 16..31).

    Optional ``reg_tensor_top`` / ``reg_tensor_bot`` enable skip-R1.Load mode
    for both halves; they must be provided or omitted together.
    """

    def __init__(
        self,
        tmem_ptr,
        reg_tensor_top: Optional[cute.Tensor] = None,
        reg_tensor_bot: Optional[cute.Tensor] = None,
    ) -> None:
        if (reg_tensor_top is None) != (reg_tensor_bot is None):
            raise ValueError(
                "TmemTranspose32x32Inplace: reg_tensor_top and reg_tensor_bot "
                "must be provided or omitted together (both halves either "
                "skip-R1.Load or do R1.Load)."
            )
        self.top = TmemTranspose16x32(tmem_ptr, Region.Top, reg_tensor=reg_tensor_top)
        self.bot = TmemTranspose16x32(tmem_ptr, Region.Bottom, reg_tensor=reg_tensor_bot)

    def from_r1_perm_until_last_store(self) -> Tuple[cute.Tensor, cute.Tensor]:
        self.bot.r1_perm()
        self.top.r1_perm()
        self.bot.r1_store()
        self.top.r1_store()

        self.bot.r2_load()
        self.top.r2_load()
        self.top.r2_store()
        self.bot.r2_store()

        self.top.r3_load_top()
        self.top.r3_load_bot()
        self.bot.r3_load_top()
        self.bot.r3_load_bot()
        self.top.r3_perm()
        self.bot.r3_perm()
        self.top.r3_store()
        self.bot.r3_store()

        self.top.r4_load_top()
        self.top.r4_load_bot()
        self.bot.r4_load_top()
        self.bot.r4_load_bot()
        self.top.r4_perm()
        self.bot.r4_perm()
        return self.top.output, self.bot.output


class TmemTranspose32x64B16Movm:
    """FC2 warp-local 32-hidden x 64-token BF16 transpose using MOVM.

    Input is the flat ``[top, bottom]`` distribution produced by two
    16dp256bit accumulator loads followed by ``fc2_f2fp``. Output
    ``(lane_idx, elem_idx)`` coordinates are:

        token  = lane_idx + 32 * (elem_idx // 32)
        hidden = elem_idx % 32

    The fixed register permutation between MOVM and STTM is an SSA rename. It
    keeps each thread's two complete hidden-32 rows without any lane exchange.
    """

    _tmem_row_stride = 1 << 16
    _store_reg_source_indices = (
        0,
        2,
        1,
        3,
        16,
        18,
        17,
        19,
        8,
        10,
        9,
        11,
        24,
        26,
        25,
        27,
        4,
        6,
        5,
        7,
        20,
        22,
        21,
        23,
        12,
        14,
        13,
        15,
        28,
        30,
        29,
        31,
    )

    @staticmethod
    def _tmem_layout(num_lanes: int, num_cols: int) -> cute.Layout:
        return cute.make_layout(
            (((num_lanes, num_cols), 1),),
            stride=(((TmemTranspose32x64B16Movm._tmem_row_stride, 1), 0),),
        )

    @staticmethod
    def _rmem_copy_view(rmem: cute.Tensor, num_regs: int, offset: int = 0) -> cute.Tensor:
        return cute.make_tensor(
            rmem.iterator + offset, cute.make_layout((((num_regs,), 1),), stride=(((1,), 0),))
        )

    @cute.jit
    def __init__(self, tmem_ptr, reg_tensor: cute.Tensor) -> None:
        if cutlass.const_expr(reg_tensor.element_type is not cutlass.BFloat16):
            raise TypeError(
                f"{type(self).__name__} expects BF16 input after f2fp, got {reg_tensor.element_type}."
            )
        if cutlass.const_expr(cute.size(reg_tensor) != 64):
            raise ValueError(
                f"{type(self).__name__} expects 64 BF16 elements, got {cute.size(reg_tensor)}."
            )

        movm_words = movmatrix_b16(cute.recast_tensor(reg_tensor, cutlass.Int32))
        self._store_words = cute.make_rmem_tensor(movm_words.layout, movm_words.element_type)
        for store_reg in cutlass.range_constexpr(32):
            self._store_words[store_reg] = movm_words[self._store_reg_source_indices[store_reg]]

        half_lane_offset = 16 * self._tmem_row_stride
        self._tmem_top = cute.make_tensor(tmem_ptr, self._tmem_layout(16, 32))
        self._tmem_bottom = cute.make_tensor(tmem_ptr + half_lane_offset, self._tmem_layout(16, 32))
        self._tmem_full = cute.make_tensor(tmem_ptr, self._tmem_layout(32, 32))
        self._store_atom = cute.make_copy_atom(
            tcgen05.St16x128bOp(tcgen05.Repetition.x8), cutlass.Float32
        )
        self._load_atom = cute.make_copy_atom(
            tcgen05.Ld32x32bOp(tcgen05.Repetition.x32), cutlass.Float32
        )

    @cute.jit
    def __call__(self) -> cute.Tensor:
        movm_words_f32 = cute.recast_tensor(self._store_words, cutlass.Float32)
        cute.copy(self._store_atom, self._rmem_copy_view(movm_words_f32, 16), self._tmem_top)
        cute.copy(
            self._store_atom, self._rmem_copy_view(movm_words_f32, 16, offset=16), self._tmem_bottom
        )

        output_words = cute.make_rmem_tensor((32,), cutlass.Float32)
        cute.copy(self._load_atom, self._tmem_full, self._rmem_copy_view(output_words, 32))
        return cute.recast_tensor(output_words, cutlass.BFloat16)


@dataclasses.dataclass(frozen=True)
class GatedActEpilogueArgs:
    """Optional runtime tensors used by the gated-activation epilogue."""

    fc1_alpha: Optional[cute.Tensor]
    fc2_alpha: Optional[cute.Tensor]
    fc1_norm_const: Optional[cute.Tensor]
    # -----------------------------------
    # MoE domain (token, topk), deepgemm graph only? for transformer graph, we want reduce kernel to perform the
    # score mul.
    topk_scores: Optional[cute.Tensor]


class SwapABGatedActEpilogue(KernelComponent):
    """Autonomous epilogue for the swap-AB SwiGLU NVFP4 kernel.

    ``run()`` is the single entry point the kernel calls inside the epi
    warp body.  The kernel's responsibility is reduced to:

      - allocate / free TMEM and build ``acc_tensor``
      - construct the AB / acc pipelines
      - obtain the scheduler consumer

    Everything else (acc consumer state, task-tile loop, overlap rotation,
    early release, TMA store commit / drain, per-subtile dispatch) lives
    inside this class.
    """

    _EpilogueSyncWaitBarId = 1  # Arrive and wait only
    # v10-epi2 named barriers (free ids on the Blackwell kernel: 3, 8, 14, 15).
    _EpilogueWg1SyncWaitBarId = 14  # WG1's tile-boundary rendezvous (WG0 keeps id 1)
    _EpilogueWgHandoffBarId = 15  # WG1 has loaded the "second" subtile -> WG0 may use its TMEM
    EpilogueWgTmemDoneBarId = 3  # WG1 done with TMEM -> WG0 may free it
    _EpilogueAsyncBarIdBase = 4  # Some arrive, the others arrive and wait
    _EpilogueFc1GateUpInterleave = 16
    _EpilogueTokenTileSize = 64  # Fundamentally the epi_tile_n
    _EpilogueUnitTokens = 32  # token granularity of the N = 224 epilogue (7 units per tile)
    _EpilogueFc1IntermediateGateUpTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueFc1IntermediateDownTileSize = 64  # Fundamentally epi_tile_m // 2
    _EpilogueFc2HiddenTileSize = 128  # Fundamentally epi_tile_m
    _EpilogueWarpCnt = 4
    # One warp owns this many intermediate_down outputs per token: the TMEM transpose gives each
    # warp 32 accumulator rows, which the gate/up interleave halves.
    _EpilogueFc1IntermediateDownPerWarp = 16
    smem_scratch_overlay: ClassVar[str] = "blackwell.swap_ab_gated_act_epilogue.scratch"
    fc1_staging_region: ClassVar[str] = "blackwell.swap_ab_gated_act_epilogue.fc1_staging"
    fc2_staging_region: ClassVar[str] = "blackwell.swap_ab_gated_act_epilogue.fc2_staging"
    # A 144 B FP8 stride rotates each row by four SMEM banks.
    _Fc2UblkFp8RowStrideBytes = 144
    _ScratchByteAlignment = 128

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            # The accumulator follows from the instruction family; see __init__.
            "quant_kind": str,
            "hidden_size": StaticOrRuntimeIntegerType,
            "intermediate_gateup_size": StaticOrRuntimeIntegerType,
            "combine_format": CombineFormat,
            "gate_up_clamp": Optional[float],
            "swiglu_alpha": OptionalRequirement(Optional[float]),
            "swiglu_beta": OptionalRequirement(Optional[float]),
            # SiTU (Kimi K3) selects a different gated-activation core; optional so
            # existing SwiGLU descriptors stay valid unchanged. See _resolve_situ_betas.
            "situ_beta": OptionalRequirement(Optional[float]),
            "situ_linear_beta": OptionalRequirement(Optional[float]),
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, object]:
        return {
            "mma_tiler_mnk": tuple,
            "cluster_shape_mn": tuple,
            "use_2cta_instrs": bool,
            "fc2_use_bulk": bool,
            "communication_enabled": bool,
            "fc1_epi_flag_batch": int,
            "fc2_epi_flag_batch": int,
            "fc2_tma_stages": OptionalRequirement(int),
            "reduce_topk_in_kernel": OptionalRequirement(bool),
            "token_back_push_data": OptionalRequirement(bool),
        }

    @staticmethod
    def _resolve_situ_betas(problem_desc: ProblemDesc) -> Tuple[Optional[float], Optional[float]]:
        """Resolve the SiTU (Kimi K3) betas from the ProblemDesc; ``(None, None)`` means SwiGLU.

        Both betas must be given together, and SiTU excludes ``gate_up_clamp`` -- matching DeepGEMM's
        ``DG_HOST_ASSERT(not use_situ or not activation_clamp_opt.has_value())``. They are baked in at
        codegen time exactly like ``gate_up_clamp``, so the enclosing KernelClass must also fold them
        into its ``name()`` cache key.
        """
        beta = problem_desc.get("situ_beta")
        linear_beta = problem_desc.get("situ_linear_beta")
        if (beta is None) != (linear_beta is None):
            raise ValueError(
                f"situ_beta and situ_linear_beta must be set together, got {beta} and {linear_beta}."
            )
        if beta is None:
            return None, None
        if beta <= 0 or linear_beta <= 0:
            raise ValueError(
                f"SiTU beta parameters must be positive, got {beta} and {linear_beta}."
            )
        if problem_desc["gate_up_clamp"] is not None:
            raise ValueError(
                "SiTU does not support gate_up_clamp; the two activation variants are exclusive."
            )
        if problem_desc.get("swiglu_alpha") is not None:
            raise ValueError("SiTU does not support SwiGLUBias constants.")
        return float(beta), float(linear_beta)

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.acc_dtype = tcgen05_block_scaled_acc_dtype
        self.hidden_size = problem_desc["hidden_size"]
        self.intermediate_gateup_size = problem_desc["intermediate_gateup_size"]
        self.combine_format = problem_desc["combine_format"]
        self.gate_up_clamp = problem_desc["gate_up_clamp"]
        self.swiglu_alpha = problem_desc.get("swiglu_alpha")
        self.swiglu_beta = problem_desc.get("swiglu_beta")
        if (self.swiglu_alpha is None) != (self.swiglu_beta is None):
            raise ValueError("swiglu_alpha and swiglu_beta must be set together.")
        if self.swiglu_alpha is not None:
            self.swiglu_alpha = float(self.swiglu_alpha)
            self.swiglu_beta = float(self.swiglu_beta)
        self.situ_beta, self.situ_linear_beta = self._resolve_situ_betas(problem_desc)
        # v2-situ A/B switches (trace-time constants; default off = base kernel). Use a distinct
        # VARIANT_CACHE_TAG per combination: the on-disk CuTe DSL cache may not see env vars.
        #   MEGAMOE_SITU_FAST=1        3-MUFU SiTU core (_situ_fast_pair) instead of 6 MUFU.
        #   MEGAMOE_SITU_FOLD_ALPHA=1  (with SITU_FAST) fold the fc1 alpha dequant into the exp
        #                              constants; drops the per-element dequant multiplies.
        #   MEGAMOE_FC1_TOPK_FOLD=1    fold the topk weight into the per-block NVFP4 scale instead
        #                              of pre-multiplying every element.
        # epilogue switches resolved once per config (prod_config defaults + env overrides).
        _options = epi_options(self.quant_kind, self.combine_format)
        self.epi_options_resolved = _options
        self.situ_fast = self.situ_beta is not None and _options["situ_fast"]
        self.situ_fold_alpha = (
            self.situ_fast and _kimi_options.environ.get("MEGAMOE_SITU_FOLD_ALPHA", "0") == "1"
        )
        self.fc1_topk_fold = _kimi_options.environ.get("MEGAMOE_FC1_TOPK_FOLD", "0") == "1"
        # v10-epi2 (MEGAMOE_EPI_WG2=1, decided by the kernel and passed as impl_desc["epi_wg2"]): a
        # second epilogue warpgroup (warps 12-15) splits every work tile's four token subtiles with
        # warps 0-3. See SwapABFc1Epilogue.call_wg2 for the TMEM / SMEM / barrier protocol.
        self.epi_wg2 = bool(impl_desc.get("epi_wg2", False))
        self.mma_tiler_mnk = impl_desc["mma_tiler_mnk"]
        self.cluster_shape_mn = impl_desc["cluster_shape_mn"]
        self.use_2cta_instrs = impl_desc["use_2cta_instrs"]
        self.fc2_use_bulk = impl_desc["fc2_use_bulk"]
        self.communication_enabled = impl_desc["communication_enabled"]
        self.fc1_epi_flag_batch = impl_desc["fc1_epi_flag_batch"]
        self.fc2_epi_flag_batch = impl_desc["fc2_epi_flag_batch"]
        self.opt_fc1_act_first = _options["fc1_act_first"]
        self.opt_fc2_no_tile_drain = _options["fc2_no_tile_drain"]
        self.opt_fc2_prefetch = _options["fc2_prefetch"]
        self.opt_iket_fine = _options["iket_fine"]
        self.opt_fc1_pipe = _options["fc1_pipe"] or _options["fc1_pipe_full"]
        self.fc1_pipe_full_only = _options["fc1_pipe_full"]
        # v9-fc1lean switches (codegen-time; all default off = v6 code):
        #   MEGAMOE_LEAN_SMEM_TRANS=1   FC1: activation on the raw TMEM-load distribution (ACT_FIRST math),
        #                               then a warp-private SMEM transpose (32 STS.32 + 8 LDS.128 per subtile,
        #                               swizzled, bank-conflict free) instead of the TMEM STTM/LDTM chain and
        #                               its register permutations. Pure data movement: bit-identical.
        #   MEGAMOE_LEAN_QMUL2=1        FC1 quant: x * acc_scale as packed FMUL2 (8 per token, not 16 FMUL).
        #   MEGAMOE_LEAN_AMAX_TREE=1    FC1 quant: 16-wide abs-max as a 4-level tree, not a 15-long chain.
        #   MEGAMOE_LEAN_SFIDX=1        FC1 scale-factor STG: index the ((32,4),rest) token mode with the
        #                               hierarchical coordinate ((lane, q), r) the tile already knows, instead
        #                               of letting cute divide/mod the flat token index (cta_tile_n % 128 == 0).
        #   MEGAMOE_LEAN_TMEM_UNIFORM=1 FC1: mark the per-subtile TMEM address warp-uniform (fewer R2UR).
        self.lean_smem_trans = _options["lean_smem_trans"]
        self.lean_qmul2 = _options["lean_qmul2"]
        self.lean_amax_tree = _options["lean_amax_tree"]
        self.lean_sfidx = _options["lean_sfidx"]
        self.lean_tmem_uniform = _options["lean_tmem_uniform"]
        # v14-epi3 switches (codegen-time; all default off = v12 code):
        #   MEGAMOE_LEAN_FC2_STSM=1   FC2 (STG / quantized) BF16 transpose: 8 STSM.x4.trans + 8 LDS.128 through a
        #                             warp-private SMEM scratch instead of 32 MOVM + the TMEM STTM/LDTM round trip.
        #                             Same output distribution; no TMEM workspace, so WG2 drops the FC2 handoff
        #                             barrier. Pure data movement: bit-identical.
        #   MEGAMOE_LEAN_FC2_QUANT=1  FC2 e2m1 combine quant: bf16 -> f32 by SHL / LOP3 (no PRMT), 16-wide abs-max
        #                             as FMNMX3 in f32, the two blocks of a token share packed f32x2 scale math and
        #                             one PRMT builds the bf16 scale pair. Same values: bit-identical.
        #   MEGAMOE_LEAN_FC1_QUANT=1  FC1 NVFP4 quant: abs-max as FMNMX3, both tokens' UE4M3 scale through one
        #                             cvt e4m3x2 + one e4m3x2 -> f16x2, packed f32x2 scale math. Bit-identical.
        #   MEGAMOE_LEAN_FC2_ADDR=1   FC2 STG router: per-tile per-thread data / scale base addresses, per-subtile
        #                             offsets only (no per-STG 64-bit pointer rebuild). Same addresses.
        #   MEGAMOE_LEAN_TILE_DRAIN=1 per-tile tail: no bulk-group wait after an FC2 tile in STG mode (it issued no
        #                             bulk copies and every FC1 tile drains its own), so no DEPBAR / L1 invalidate.
        #   MEGAMOE_LEAN_SCHED_FENCE=1  epilogue work-tile consume: no fence.acq_rel.cta (MEMBAR.ALL.CTA, which also
        #                             waits for the warp's in-flight global stores) before the slot release; the
        #                             mbarrier.arrive is .release.cta by default and already orders the slot reads.
        self.lean_fc2_stsm = _options["lean_fc2_stsm"]
        self.lean_fc2_quant = _options["lean_fc2_quant"]
        self.lean_fc1_quant = _options["lean_fc1_quant"]
        self.lean_fc2_addr = _options["lean_fc2_addr"]
        self.lean_tile = _options["lean_tile"]
        self.lean_sched_fence = _options["lean_sched_fence"]
        # MEGAMOE_MX_LEAN_T=1: let the paired-exchange (sf_vec 32) FC1 path use the v9 lean SMEM
        # transpose too (with MEGAMOE_LEAN_SMEM_TRANS=1). The lean transpose hands fc1_quant the same
        # (lane = token, 16 intermediate values) registers as the TMEM transpose chain, its scratch is
        # warp-private and disjoint from the amax-exchange slots (which borrow the subtile's staging stage),
        # so the values are bit-identical. Default off keeps the measured mx1 kernel unchanged.
        self.mx_lean_t = _options["mx_lean_t"]
        # MEGAMOE_EPI_ASYNC_FLAG=1 (from v11-epi2defer): publish the per-tile done flags with
        # red.async.release.gpu from one elected lane of warp 0 (fire-and-forget) instead of a blocking
        # red.release.gpu by thread 0. The mx1 IKET puts 40% of the MMA's acc_acquire time in cases where
        # the epilogue is still in drain/signal of the previous tile (signal p90 3.5 us): the blocking
        # release keeps warp 0 -- and through the pair-exchange / subtile barriers the whole WG -- from
        # starting the next tile and releasing the shared accumulator columns. Same flags, same values.
        self.epi_async_flag = _options["epi_async_flag"]
        self.epi_fc2_async_flag = _options["epi_fc2_async_flag"]

        if self.communication_enabled:
            for field_name in ("reduce_topk_in_kernel", "token_back_push_data"):
                if field_name not in impl_desc:
                    raise KeyError(
                        f"Communication-enabled Epilogue requires ImplDesc field {field_name!r}."
                    )
        self.reduce_topk_in_kernel = impl_desc.get("reduce_topk_in_kernel", False)
        self.token_back_push_data = impl_desc.get("token_back_push_data", False)
        if not self.communication_enabled and (
            self.reduce_topk_in_kernel or self.token_back_push_data
        ):
            raise ValueError(
                "A communication-disabled Epilogue cannot enable communication policies."
            )

        # FC1 emits what FC2 consumes as its activation, in the kind's own scale format.
        self.fc1_output_dtype = self.quant_kind.activation_dtype
        self.fc1_output_sf_dtype = self.quant_kind.sf_dtype
        self.sf_vec_size = self.quant_kind.sf_vec_size
        # A 32-wide scale block spans two epilogue warps; see QuantImpl's paired direction.
        self.needs_pair_amax_exchange = self.sf_vec_size > self._EpilogueFc1IntermediateDownPerWarp
        self.token_back_push_sf = self.communication_enabled and self.combine_format.is_quantized
        self.token_back_enabled = self.token_back_push_data or self.token_back_push_sf
        # v16-fc2blk (MEGAMOE_FC2_BLOCK_DONE=1, decided by the kernel and passed as impl_desc["fc2_block_done"]).
        self.fc2_block_done = bool(impl_desc.get("fc2_block_done", False)) and self.token_back_enabled
        # v26-tbtail (MEGAMOE_FC2_TAIL_BLOCKS=K): (K, local expert count) or None.
        self.fc2_tail_blocks = impl_desc.get("fc2_tail_blocks", None) if self.token_back_enabled else None
        self.fc2_output_is_local = not self.communication_enabled or self.token_back_push_data
        self.fc2_use_tma = self.fc2_use_bulk and self.fc2_output_is_local
        self.fc2_use_ublk = self.fc2_use_bulk and not self.fc2_output_is_local
        if self.fc2_use_ublk and self.combine_format.act_dtype.width == 4:
            raise ValueError("FC2 UBLK does not support an FP4 combine payload.")
        if self.reduce_topk_in_kernel and self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("In-kernel top-k reduction requires a BF16 combine format.")
        self.reduce_topk_in_epilogue = self.reduce_topk_in_kernel and not self.token_back_push_data
        # v5-epi switches only take effect where their preconditions hold; elsewhere they are no-ops.
        # No-tile-drain: FC2 publishes no done flag (BF16 epi_warps token-back writes peers directly),
        # so between two FC2 tiles only the staging-ring reuse needs ordering, which the per-subtile
        # read wait already provides once every FC2 iteration commits exactly one bulk group.
        self.fc2_skip_tile_drain = (
            self.opt_fc2_no_tile_drain and self.fc2_use_ublk and not self.token_back_enabled
        )
        # v6-fc1pipe: the pipelined FC1 loop runs out-of-range subtiles' math unpredicated (only their
        # stores are predicated), which is only safe for the in-thread quant (no pair amax exchange
        # through the subtile's staging stage / barrier inside the quant).
        self.fc1_pipe = self.opt_fc1_pipe and not self.needs_pair_amax_exchange
        self.fc2_ldtm_prefetch = (
            self.opt_fc2_prefetch and self.fc2_use_ublk and not self.combine_format.is_quantized
        )
        if not 1 <= self.fc1_epi_flag_batch <= 32 or not 1 <= self.fc2_epi_flag_batch <= 32:
            raise ValueError("Epilogue flag batch sizes must be in [1, 32].")
        self.cluster_tile_intermediate_downproj = (
            self._EpilogueFc1IntermediateDownTileSize * self.cluster_shape_mn[0]
        )

        atom_thr_size = 2 if self.use_2cta_instrs else 1
        self.cta_tile_m = self._EpilogueFc2HiddenTileSize
        self.cta_tile_n = self.mma_tiler_mnk[1]
        self.cta_tile_k = self.mma_tiler_mnk[2]
        assert self.mma_tiler_mnk[0] // atom_thr_size == self.cta_tile_m
        # mma tile N = 224 with two full accumulators (2 x 224 + 64 SF = 512 TMEM columns): the
        # epilogue works in 32-token units (7 per tile), see SwapABFc1Epilogue.call_n224.
        # N = 240 (two full 240-column accumulators, SF staged per K half) reuses the 32-token unit path:
        # 8 units, the last one half (tokens 224..239); its FC1 stores are 16-token boxes (experts padded to 16).
        self.n224 = self.cta_tile_n in (224, 240)
        self.n240 = self.cta_tile_n == 240
        # NVFP4 + quantized combine at N = 240 (the 4096 / 8192 / 16384 buckets: two full accumulators + early token-back)
        # publishes its FC2 done flags asynchronously unless MEGAMOE_EPI_FC2_ASYNC_FLAG is set. With two full
        # accumulators the epilogue, not the MMA, set the pace, and the blocking FC2 release (MEMBAR.ALL.GPU: the SM's
        # in-flight NVLink / TMA traffic drains first; the tile's own ~30 KB of STG FC2 rows too) was 2.3-2.9 us of a
        # ~5 us epilogue cycle (docs/optimizations.md "s24-u4k"). FC1 flags stay blocking (they gate FC2 operand loads). N = 224 (2048
        # bucket, MMA-bound) keeps both blocking.
        if (
            self.n240
            and _prod_config.env_override("MEGAMOE_EPI_FC2_ASYNC_FLAG") is None
            and _prod_config.defaults_active()
            and _prod_config.config_key(self.quant_kind, self.combine_format) == ("nv", "lp")
        ):
            self.epi_fc2_async_flag = True
        if self.epi_async_flag:
            self.epi_fc2_async_flag = False  # every flag is already asynchronous
        self.epi_options_resolved = {**self.epi_options_resolved, "epi_fc2_async_flag": self.epi_fc2_async_flag}
        # N = 224 / 240 on the bulk (UBLK / TMA) FC2 store with ONE epilogue WG: FC1 runs the 32-token unit
        # loop on a single WG (call_n224_1wg, unit pairs -> the 4 FC1 staging stages), FC2 the regular 64-token bulk
        # subtile loop (subtile 3 partial; rows past the tile's valid tokens are predicated off by the output router).
        self.n224_1wg = self.n224 and self.fc2_use_bulk and not self.epi_wg2
        if self.n224:
            assert self.cta_tile_n % 16 == 0
        else:
            assert self.cta_tile_n % self._EpilogueTokenTileSize == 0
        tmem_plan = impl_desc["tmem_plan"]
        self.num_sfa_tmem_cols = tmem_plan.sfa_columns
        self.num_sfb_tmem_cols = tmem_plan.sfb_columns
        self.num_sf_tmem_cols = tmem_plan.sfa_columns + tmem_plan.sfb_columns
        self.num_tmem_alloc_cols = tmem_plan.allocation_columns
        self.num_accumulator_stages = tmem_plan.accumulator_stage_count
        self.num_accumulator_pipeline_stages = tmem_plan.accumulator_pipeline_stages
        self.accumulator_overlap_columns = (
            tmem_plan.accumulator_stage_columns - tmem_plan.accumulator_stage_stride_columns
        )
        self.num_accumulator_tmem_cols = tmem_plan.accumulator_columns
        self.overlapping_accum = self.accumulator_overlap_columns > 0
        self.accumulator_shape = (
            self.cta_tile_m,
            self.cta_tile_n,
            tmem_plan.accumulator_stage_count,
        )
        self.accumulator_stride = (1 << 16, 1, tmem_plan.accumulator_stage_stride_columns)

        if (
            isinstance(self.hidden_size, int)
            and self.hidden_size % (self.cta_tile_m * self.cluster_shape_mn[0]) == 0
        ):
            self.fc2_hidden_needs_predicate: bool = False
        else:
            self.fc2_hidden_needs_predicate: bool = True

        if isinstance(self.intermediate_gateup_size, int):
            self.intermediate_downproj: Optional[int] = self.intermediate_gateup_size // 2
        else:
            self.intermediate_downproj: Optional[int] = None

        self.subtile_cnt = self.cta_tile_n // self._EpilogueTokenTileSize
        if self.n224:
            # [s5-n224, from v30] 4 FC1 staging stages (2 per epilogue WG), each a 64-token x 64-intermediate tile
            # holding two 32-token units; 7 units per tile.
            self.subtile_cnt = 4
            self.unit_cnt = -(-self.cta_tile_n // self._EpilogueUnitTokens)
        # MEGAMOE_MX_FC1_STAGE2=1: the FC1 output staging ring has 2 stages, used in processing order
        # (stage = position % 2) instead of one stage per token subtile. Positions >= 2 reuse the stage of position
        # p - 2: every epilogue thread waits for its own outstanding bulk groups' SMEM reads (only the issuer of
        # p - 2's TMA store has one) and the WG meets at its barrier before anything -- amax-exchange slots, the
        # quantized STS -- touches the stage. Same data, same stores: bit-identical. Frees 2 stages (8 KiB for
        # an e4m3 FC1 output), which with MEGAMOE_LEAN_FC2_STSM off (no 18 KiB scratch) lets a 6th K=128 AB
        # stage fit (5248 B short before). Single-WG default FC1 loop only (not WG2 / FC1_PIPE).
        self.fc1_stage2 = (
            _options["fc1_stage2"]
            and not self.epi_wg2
            and not (self.opt_fc1_pipe and not self.needs_pair_amax_exchange)
            and self.overlapping_accum
            and self.subtile_cnt == 4
        )
        self.fc1_staging_stage_cnt = 2 if self.fc1_stage2 else self.subtile_cnt

        # One staging stage per token subtile, each an (epi_tile_n, epi_tile_m // 2) quantized tile.
        self.fc1_staging_stage_bytes = (
            self._EpilogueTokenTileSize
            * self._EpilogueFc1IntermediateDownTileSize
            * self.fc1_output_dtype.width
            // 8
        )
        self.fc1_staging_bytes = self.fc1_staging_stage_cnt * self.fc1_staging_stage_bytes
        # v9-fc1lean SMEM transpose scratch: per epilogue warp, 64 tokens x 16 fp32 + padding (4.5 KiB). It shares the
        # fc1 staging lifetime of the epilogue scratch overlay (free whenever the FC2 staging is larger).
        self.fc1_lean_scratch_bytes = (
            self._EpilogueWarpCnt * 1152 * 4  # == _LeanScratchWordsPerWarp words per warp
            # v10-epi2: same bytes, as 8 warps x 576 words (one token half at a time, see
            # _fc1_smem_transpose_wg2), so the AB pipeline keeps its stage count.
            # N = 224 always uses the lean (warp-private) transpose, MX included.
            if self.lean_smem_trans and (not self.needs_pair_amax_exchange or self.mx_lean_t or self.n224)
            else 0
        )
        # v14-epi3 (MEGAMOE_LEAN_FC2_STSM): the FC2 BF16 transpose borrows the same per-warp scratch slots
        # (2048 B per 32-token pass; WG2 slots are 2304 B, single-WG slots 4608 B hold both passes at once).
        # STG FC2 stores only: with a bulk FC2 store the overlay's fc2 staging lifetime aliases the scratch
        # while bulk copies may still read it.
        # N = 224 needs the STSM transpose (its FC2 unit path); BF16 combine / MX defaults leave it off.
        self.fc2_stsm = (
            (self.lean_fc2_stsm or self.n224) and not self.reduce_topk_in_epilogue and not self.fc2_use_bulk
        )
        self.fc2_stsm_scratch_bytes = self._EpilogueWarpCnt * 1152 * 4 if self.fc2_stsm else 0
        # The amax exchange plane is (token chunk, epilogue thread), block-major so each thread's
        # column is one vector access. A thread holds one token per 32-lane chunk of the subtile.
        # These slots borrow the current subtile's staging stage rather than costing their own
        # bytes -- see the lifetime argument in fc1_quant.
        self.fc1_amax_token_chunks = self._EpilogueTokenTileSize // 32
        self.fc1_amax_slot_count = (
            self.fc1_amax_token_chunks * self._EpilogueWarpCnt * 32
            if self.needs_pair_amax_exchange
            else 0
        )
        if self.fc1_amax_slot_count * 4 > self.fc1_staging_stage_bytes:
            raise ValueError("The paired amax exchange does not fit in one fc1 staging stage.")

        requested_fc2_tma_stages = impl_desc.get("fc2_tma_stages")
        if (
            requested_fc2_tma_stages is not None
            and not 1 <= requested_fc2_tma_stages <= self.subtile_cnt
        ):
            raise ValueError(
                f"fc2_tma_stages must be in [1, {self.subtile_cnt}], got {requested_fc2_tma_stages}."
            )
        if self.fc2_use_bulk:
            single_stage_region = self._make_fc2_single_stage_region()
            if self.fc2_use_ublk and single_stage_region.nbytes % 16 != 0:
                raise ValueError("Each FC2 UBLK staging stage must occupy a multiple of 16 bytes.")
            # Additional stages trade mainloop SMEM for store overlap.
            self.fc2_tma_stages = (
                requested_fc2_tma_stages if requested_fc2_tma_stages is not None else 1
            )
            self.fc2_staging_spec: Optional[SmemRegion] = self._make_fc2_staging_region(
                self.fc2_tma_stages
            )
        else:
            self.fc2_tma_stages = 0
            self.fc2_staging_spec = None

        if self.n224:
            # two disjoint accumulators, both epilogue WGs, warp-private SMEM transposes only
            # (no TMEM workspace exists: both accumulator stages belong to the MMA / the other tile).
            # or ONE epilogue WG with the bulk FC2 store (n224_1wg; FC2 keeps its 64-token bulk loop).
            if not self.epi_wg2 and not self.n224_1wg:
                raise ValueError("mma_tiler N = 224 requires MEGAMOE_EPI_WG2=1.")
            if self.overlapping_accum or self.num_accumulator_pipeline_stages != 2:
                raise ValueError("mma_tiler N = 224 needs the two-stage disjoint accumulator TMEM plan.")
            if self.fc1_lean_scratch_bytes == 0 or not (self.fc2_stsm or self.n224_1wg):
                raise ValueError(
                    "mma_tiler N = 224 requires MEGAMOE_LEAN_SMEM_TRANS=1 and MEGAMOE_LEAN_FC2_STSM=1."
                )
            if self.fc1_pipe or self.opt_fc1_act_first or (self.fc2_ldtm_prefetch and not self.n224_1wg):
                raise ValueError("mma_tiler N = 224 does not support the v5 / v6 FC1 / FC2 loop options.")
        if self.epi_wg2:
            # The split relies on the SM100 overlapped two-stage accumulator (one shared overlap
            # subtile per tile), four 64-token subtiles, register-only FC2 stores and in-thread
            # NVFP4 scale blocks (the paired amax exchange uses the single-WG sync barrier).
            if (not self.overlapping_accum and not self.n224) or self.subtile_cnt != 4:
                raise ValueError("MEGAMOE_EPI_WG2 needs the overlapped N=256 accumulator.")
            if self.fc2_use_bulk:
                raise ValueError("MEGAMOE_EPI_WG2 supports only the STG FC2 store (fc2_use_bulk=False).")
            # the paired amax exchange now runs on each WG's own 128-thread barrier
            # (SwapABFc1Epilogue._pair_barrier), so sf_vec 32 kinds may use WG2.
            if self.reduce_topk_in_epilogue:
                raise ValueError("MEGAMOE_EPI_WG2 does not support REDG.")

    @classmethod
    def epilogue_sync_barrier(cls) -> pipeline.NamedBarrier:
        """The one barrier every epilogue rendezvous uses: all four warps, arrive-and-wait.

        Reused rather than split per purpose because the participant set is always the same 128
        threads and no two uses are ever in flight together -- the tile-boundary rendezvous sits
        outside the subtile loop, FC1's amax exchange and FC2's bulk-store handshake belong to
        different work tiles.
        """
        return pipeline.NamedBarrier(
            barrier_id=cls._EpilogueSyncWaitBarId, num_threads=32 * cls._EpilogueWarpCnt
        )

    def register_smem_regions(self, smem_workspace: SmemWorkspace) -> None:
        """Declare the epilogue scratch as one allocation shared by two exclusive lifetimes.

        FC1's staging tile and FC2's store tile belong to different work tiles, separated by the
        tile-boundary TMA drain and rendezvous in ``run()``, so they never coexist.
        """
        overlay = smem_workspace.create_overlay(self.smem_scratch_overlay)
        overlay.add_lifetime("fc1_staging").register_tensor(
            self.fc1_staging_region,
            cutlass.Int8,
            (
                self.fc1_staging_bytes
                + (
                    max(self.fc1_lean_scratch_bytes, self.fc2_stsm_scratch_bytes)
                    if self.fc2_stsm_scratch_bytes
                    else self.fc1_lean_scratch_bytes
                ),
            ),
            byte_alignment=128,
        )
        if self.fc2_staging_spec is not None:
            overlay.add_lifetime("fc2_staging").register_tensor(
                self.fc2_staging_region,
                self.fc2_staging_spec.dtype,
                self.fc2_staging_spec.shape,
                stride=self.fc2_staging_spec.stride,
                swizzle=self.fc2_staging_spec.swizzle,
                byte_alignment=self.fc2_staging_spec.byte_alignment,
            )

    def fc1_staged_smem_layout(
        self, n_stages: int, without_stage_mode: bool = False, token_rows: Optional[int] = None
    ) -> Union[cute.Layout, cute.ComposedLayout]:
        layout = sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            utils.LayoutEnum.ROW_MAJOR,
            (
                self._EpilogueTokenTileSize if token_rows is None else token_rows,
                self._EpilogueFc1IntermediateDownTileSize,
            ),
            n_stages,
        )
        if without_stage_mode:
            return cute.select(layout, mode=[0, 1])
        return layout

    def fc2_tma_staged_smem_spec(self, n_stages: int) -> Tuple[Tuple, Tuple, Tuple[int, int, int]]:
        """Return the bank-conflict-free token-major FC2 TMA staging layout."""
        stage_stride = self._EpilogueTokenTileSize * self._EpilogueFc2HiddenTileSize
        wire_dtype = self.combine_format.act_dtype
        if wire_dtype is cutlass.BFloat16:
            shape = ((32, 2), (64, 2), n_stages)
            stride = ((64, 2048), (1, 4096), stage_stride if n_stages > 1 else 0)
            swizzle = (3, 4, 3)
        elif wire_dtype.width == 8:
            shape = ((32, 2), 128, n_stages)
            stride = ((128, 4096), 1, stage_stride if n_stages > 1 else 0)
            swizzle = (3, 4, 3)
        elif wire_dtype.width == 4:
            shape = ((32, 2), 128, n_stages)
            stride = ((128, 4096), 1, stage_stride if n_stages > 1 else 0)
            swizzle = (2, 4, 3)
        else:
            raise ValueError(f"Unsupported FC2 TMA staging dtype {wire_dtype}.")
        return shape, stride, swizzle

    def fc2_tma_staged_smem_layout(
        self, n_stages: int, without_stage_mode: bool = False
    ) -> cute.ComposedLayout:
        shape, stride, swizzle = self.fc2_tma_staged_smem_spec(n_stages)
        layout = cute.make_composed_layout(
            cute.make_swizzle(*swizzle), 0, cute.make_layout(shape, stride=stride)
        )
        if without_stage_mode:
            return cute.select(layout, mode=[0, 1])
        return layout

    def _make_fc2_single_stage_region(self) -> SmemRegion:
        """Describe one FC2 bulk-staging subtile for TMA or UBLK."""
        if self.fc2_use_tma:
            shape, stride, swizzle = self.fc2_tma_staged_smem_spec(1)
            return SmemRegion(
                name="",
                kind="tensor",
                dtype=self.combine_format.act_dtype,
                shape=shape[:-1],
                stride=stride[:-1],
                swizzle=swizzle,
                byte_alignment=128,
            )
        row_stride_elements = (
            self._Fc2UblkFp8RowStrideBytes
            if self.combine_format.act_dtype.width == 8
            else self._EpilogueFc2HiddenTileSize
        )
        return SmemRegion(
            name="",
            kind="tensor",
            dtype=self.combine_format.act_dtype,
            shape=(self._EpilogueTokenTileSize, self._EpilogueFc2HiddenTileSize),
            stride=(row_stride_elements, 1),
            swizzle=None,
            byte_alignment=16,
        )

    def _make_fc2_staging_region(self, stage_count: int) -> SmemRegion:
        single_stage_region = self._make_fc2_single_stage_region()
        return SmemRegion(
            name="",
            kind="tensor",
            dtype=single_stage_region.dtype,
            shape=(*single_stage_region.shape, stage_count),
            stride=(
                *single_stage_region.stride,
                single_stage_region.cosize if stage_count > 1 else 0,
            ),
            swizzle=single_stage_region.swizzle,
            byte_alignment=single_stage_region.byte_alignment,
        )

    def _fc1_box_tokens(self) -> int:
        """token rows per FC1 TMA store of the unit path: 32 (N = 224) or 16 (N = 240)."""
        return 16 if self.n240 else self._EpilogueUnitTokens

    def _unit_tmem_view(self, tmem_acc_tensor: cute.Tensor) -> cute.Tensor:
        """(cta_tile_m, N) accumulator viewed 32-token-unit-divisible: N = 240 is widened to 256 columns
        (unit 7 also reads columns 240..255 = the next stage / SF columns; only its tokens 224..239 are ever
        stored: per-token predicates / 16-token boxes)."""
        if cutlass.const_expr(self.n240):
            return cute.make_tensor(
                tmem_acc_tensor.iterator,
                cute.make_layout((self.cta_tile_m, 256), stride=tmem_acc_tensor.layout.stride),
            )
        return tmem_acc_tensor

    def prepare_tma_store_params(
        self, fc1_output_template: cute.Tensor, fc2_output_template: cute.Tensor
    ) -> Tuple[cute.CopyAtom, cute.Tensor, Optional[cute.CopyAtom], Optional[cute.Tensor]]:
        """Build the FC1 TMA store and the optional local FC2 TMA store."""
        fc1_operation = cpasync.CopyBulkTensorTileS2GOp()
        fc1_smem_layout = self.fc1_staged_smem_layout(1, without_stage_mode=True)
        fc1_tile = (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize)
        if cutlass.const_expr(self.n224):
            # one TMA store per 32-token unit (one half of a 64-token staging stage).
            # N = 240: one per 16-token half unit (tiles start at 240 i, 16-row granular).
            box_rows = self._fc1_box_tokens()
            fc1_smem_layout = self.fc1_staged_smem_layout(1, without_stage_mode=True, token_rows=box_rows)
            fc1_tile = (box_rows, self._EpilogueFc1IntermediateDownTileSize)
        fc1_atom, fc1_tensor = cpasync.make_tiled_tma_atom(
            fc1_operation, fc1_output_template, fc1_smem_layout, fc1_tile
        )

        if cutlass.const_expr(self.fc2_use_tma):
            # Keep tiled rest modes dynamic so a single static tile cannot collapse to stride zero.
            runtime_token_extent = cutlass.Int32(fc2_output_template.shape[0])
            runtime_hidden_extent = cutlass.Int32(fc2_output_template.shape[2])
            fc2_token_major_template = cute.make_tensor(
                fc2_output_template.iterator,
                cute.make_layout(
                    (runtime_token_extent, runtime_hidden_extent, cutlass.Int32(1)),
                    stride=(fc2_output_template.stride[0], fc2_output_template.stride[2], 0),
                ),
            )
            fc2_operation = cpasync.CopyBulkTensorTileS2GOp()
            fc2_smem_layout = self.fc2_tma_staged_smem_layout(1, without_stage_mode=True)
            fc2_tile = (self._EpilogueTokenTileSize, self._EpilogueFc2HiddenTileSize)
            fc2_atom, fc2_tensor = cpasync.make_tiled_tma_atom(
                fc2_operation, fc2_token_major_template, fc2_smem_layout, fc2_tile
            )
        else:
            fc2_atom = None
            fc2_tensor = None
        return fc1_atom, fc1_tensor, fc2_atom, fc2_tensor

    @cute.jit
    def run(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        tmem_ptr: cute.Pointer,
        acc_pipeline,
        # ── Sched ────────────────────────────────────────────────────────
        sched_consumer: SchedulerConsumer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        # ── tensors ──────────────────────────────────
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # Domain of fake (m, n, l)
        fc1_output_sf: cute.Tensor,  # Domain of fake (m, n, l)
        tma_atom_fc2_output: Optional[cute.CopyAtom],
        fc2_tma_output: Optional[cute.Tensor],  # Domain (physical_token, hidden, l=1)
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        fc1_done_counter: cute.Tensor,  # 1D tensor
        tidx: cutlass.Int32,
        token_src_metadata: Optional[cute.Tensor],
        fc2_done_counter: Optional[cute.Tensor],
        fc2_output_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        optional_epi_args: Optional[GatedActEpilogueArgs] = None,
    ):
        if cutlass.const_expr(not smem_workspace.finalized):
            raise RuntimeError("SwapABGatedActEpilogue.run requires a finalized SmemWorkspace.")
        if cutlass.const_expr(optional_epi_args is None):
            optional_epi_args = GatedActEpilogueArgs(
                fc1_alpha=None, fc2_alpha=None, fc1_norm_const=None, topk_scores=None
            )
        fc1_staging_pointer = smem_workspace.ptr(self.fc1_staging_region, smem_base)
        if cutlass.const_expr(self.fc2_tma_stages > 0):
            fc2_smem_tensor = smem_workspace.tensor(self.fc2_staging_region, smem_base)
        else:
            fc2_smem_tensor = None
        if cutlass.const_expr(
            self.fc2_use_tma and (tma_atom_fc2_output is None or fc2_tma_output is None)
        ):
            raise ValueError("FC2 TMA store requires a TMA atom and token-major output tensor.")
        if cutlass.const_expr(
            self.communication_enabled
            and (token_src_metadata is None or peer_rank_ptr_mapper is None)
        ):
            raise ValueError(
                "Communication-enabled Epilogue requires token metadata and a peer pointer mapper."
            )
        if cutlass.const_expr(self.token_back_enabled and fc2_done_counter is None):
            raise ValueError("Token-back requires an FC2 done counter.")
        if cutlass.const_expr(self.token_back_push_sf and fc2_output_sf is None):
            raise ValueError("Quantized token-back requires an FC2 output scale tensor.")
        tmem_acc = cute.make_tensor(
            cute.recast_ptr(tmem_ptr, dtype=cutlass.Float32),
            cute.make_layout(self.accumulator_shape, stride=self.accumulator_stride),
        )

        # v10-epi2: warps 0-3 -> WG 0, warps 12-15 (threads 384..511) -> WG 1. Warp-uniform.
        epi_wg = tidx // cutlass.Int32(384) if cutlass.const_expr(self.epi_wg2) else None
        fc1_epi = SwapABFc1Epilogue(
            self,
            tidx,
            fc1_staging_pointer,
            kernel_extension,
            tma_atom_fc1_output,
            fc1_output,
            fc1_output_sf,
            fc1_done_counter,
            optional_epi_args,
            epi_wg=epi_wg,
        )
        fc2_epi = SwapABFc2Epilogue(
            self,
            tidx,
            fc2_smem_tensor,
            tma_atom_fc2_output,
            fc2_tma_output,
            fc2_output,
            token_src_metadata,
            fc2_done_counter,
            fc2_output_sf,
            peer_rank_ptr_mapper,
            optional_epi_args,
            epi_wg=epi_wg,
            scratch_pointer=fc1_staging_pointer if cutlass.const_expr(self.fc2_stsm) else None,
        )

        acc_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_accumulator_pipeline_stages
        )
        if cutlass.const_expr(self.epi_wg2):
            # Each WG drains / rendezvouses / signals on its own: WG0 on id 1, WG1 on id 14.
            wait_only_named_barrier = pipeline.NamedBarrier(
                barrier_id=cutlass.Int32(self._EpilogueSyncWaitBarId)
                + epi_wg
                * cutlass.Int32(self._EpilogueWg1SyncWaitBarId - self._EpilogueSyncWaitBarId),
                num_threads=32 * self._EpilogueWarpCnt,
            )
        else:
            wait_only_named_barrier = self.epilogue_sync_barrier()
        is_odd_turn = cutlass.Int32(1)
        if cutlass.const_expr(self.lean_sched_fence):
            work_tile_info = sched_consumer.consume_work(lean_release=True)
        else:
            work_tile_info = sched_consumer.consume_work()

        if cutlass.const_expr(self.epi_async_flag):
            flag_tracker = GpuAsyncReleaseFlagBatchTracker(
                flag_address=Int64(0),
                accumulated_flags=cutlass.Int32(0),
                phase=cutlass.Int32(work_tile_info.phase),
                warp_idx=cute.arch.make_warp_uniform(
                    (tidx % (self._EpilogueWarpCnt * 32)) // cutlass.Int32(32)
                ),
            )
        elif cutlass.const_expr(self.epi_fc2_async_flag):
            flag_tracker = GpuFc2AsyncReleaseFlagBatchTracker(
                flag_address=Int64(0),
                accumulated_flags=cutlass.Int32(0),
                phase=cutlass.Int32(work_tile_info.phase),
                thread_idx=tidx % (self._EpilogueWarpCnt * 32),
            )
        else:
            flag_tracker = GpuReleaseFlagBatchTracker(
                flag_address=Int64(0),
                accumulated_flags=cutlass.Int32(0),
                phase=cutlass.Int32(work_tile_info.phase),
                thread_idx=tidx % (self._EpilogueWarpCnt * 32),
            )

        while work_tile_info.is_valid_tile:
            if cutlass.const_expr(self.overlapping_accum):
                tmem_stage_idx = acc_consumer_state.phase
            else:
                tmem_stage_idx = acc_consumer_state.index
            tmem_acc_current = tmem_acc[None, None, tmem_stage_idx]
            if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                iket.range_push("epi.fc1_tile")
                if cutlass.const_expr(self.opt_iket_fine):
                    # v6-fc1pipe diag: split FC1 tiles into full (all subtiles valid) and partial.
                    if work_tile_info.valid_tokens_in_cta_tile >= cutlass.Int32(self.cta_tile_n):
                        iket.range_push("epi.fc1_full")
                    else:
                        iket.range_push("epi.fc1_part")
                if cutlass.const_expr(self.n224_1wg):
                    # single-WG unit loop (bulk FC2 store configs).
                    fc1_epi.call_n224_1wg(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=self._unit_tmem_view(tmem_acc_current),
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                    )
                elif cutlass.const_expr(self.n224):
                    fc1_epi.call_n224(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=self._unit_tmem_view(tmem_acc_current),
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        tile_parity=tmem_stage_idx,
                    )
                elif cutlass.const_expr(self.epi_wg2):
                    fc1_epi.call_wg2(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=tmem_acc_current,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        is_odd_turn=is_odd_turn,
                    )
                else:
                    fc1_epi(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=tmem_acc_current,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        is_odd_turn=is_odd_turn,
                    )
                if cutlass.const_expr(self.opt_iket_fine):
                    iket.range_pop()
                iket.range_pop()
            else:
                # The __call__ args should only take the while loop args, leave all loop irrevalent args to the init.
                iket.range_push("epi.fc2_tile")
                if cutlass.const_expr(self.n224_1wg):
                    # the regular bulk FC2 loop over a 256-column view of this 224 / 240-column stage:
                    # subtile 3 is partial (its columns past the stage are read but never stored: the router
                    # predicates every row by the tile's valid tokens).
                    fc2_epi(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=cute.make_tensor(
                            tmem_acc_current.iterator,
                            cute.make_layout(
                                (self.cta_tile_m, -(-self.cta_tile_n // 64) * 64), stride=(1 << 16, 1)
                            ),
                        ),
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        is_odd_turn=is_odd_turn,
                    )
                elif cutlass.const_expr(self.n224):
                    fc2_epi.call_n224(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=self._unit_tmem_view(tmem_acc_current),
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        tile_parity=tmem_stage_idx,
                    )
                elif cutlass.const_expr(self.epi_wg2):
                    fc2_epi.call_wg2(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=tmem_acc_current,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        is_odd_turn=is_odd_turn,
                    )
                else:
                    fc2_epi(
                        work_tile_info=work_tile_info,
                        tmem_acc_tensor=tmem_acc_current,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                        is_odd_turn=is_odd_turn,
                    )
                iket.range_pop()
            prev_work_tile_info = work_tile_info
            cur_was_linear1 = prev_work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)

            acc_consumer_state.advance()
            if cutlass.const_expr(self.overlapping_accum):
                is_odd_turn = cutlass.Int32(1) - is_odd_turn

            if cutlass.const_expr(self.lean_sched_fence):
                work_tile_info = sched_consumer.consume_work(lean_release=True)
            else:
                work_tile_info = sched_consumer.consume_work()

            # Every asynchronous store commits at issue; drain before completion or scratch reuse.
            iket.range_push("epi.drain")
            if cutlass.const_expr(self.lean_tile and not self.fc2_use_bulk):
                # v14-epi3: STG FC2 tiles issue no bulk copies and every FC1 tile is drained here at its
                # own end, so after an FC2 tile there is no bulk group to wait for.
                if cur_was_linear1:
                    cute.arch.cp_async_bulk_wait_group(0)
            elif cutlass.const_expr(self.fc2_skip_tile_drain):
                if cur_was_linear1:
                    cute.arch.cp_async_bulk_wait_group(0)
                else:
                    if work_tile_info.is_valid_tile:
                        if work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1):
                            # FC1 staging overlays the FC2 ring: only the SMEM reads must be done.
                            cute.arch.cp_async_bulk_wait_group(0, read=True)
                    else:
                        cute.arch.cp_async_bulk_wait_group(0)
            else:
                cute.arch.cp_async_bulk_wait_group(0)
            # _fence_rel_gpu()
            wait_only_named_barrier.arrive_and_wait()
            iket.range_pop()
            iket.range_push("epi.signal")

            # Publish completion for the work tile snapshotted above.
            if cur_was_linear1:
                flag_tracker = fc1_epi.signal_fc1_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            else:
                flag_tracker = fc2_epi.signal_fc2_done(
                    prev_work_tile_info, work_tile_info, flag_tracker
                )
            iket.range_pop()
        # Tail flush
        flag_tracker.fire()


class _ImmutableAfterInit:
    """Froze at the point calling `_freeze()`"""

    def __setattr__(self, name, value):
        if self.__dict__.get("_frozen_", False):
            raise AttributeError(
                f"{type(self).__name__} is immutable after __init__ (cannot set {name!r})."
            )
        object.__setattr__(self, name, value)

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen_", True)


# Device only object
class SwapABFc1Epilogue(_ImmutableAfterInit):
    def __init__(
        self,
        base: SwapABGatedActEpilogue,
        tidx: cutlass.Int32,
        staging_pointer: cute.Pointer,
        kernel_extension: BlockScaledSwapAbFc12Extension,
        tma_atom_fc1_output: cute.CopyAtom,
        fc1_output: cute.Tensor,  # fake (m,n,l) domain
        fc1_output_sf: cute.Tensor,  # fake (m,n,l) domain
        fc1_done_counter: cute.Tensor,  # 1D tensor
        optional_epi_args: GatedActEpilogueArgs,
        epi_wg: Optional[cutlass.Int32] = None,
    ):
        self.base = base
        self.epi_wg = epi_wg
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        # (token64, intermediate, stage). The swizzle travels with the layout instead of being
        # spelled out here, so an 8-bit fc1 output picks up its own atom without a code change.
        staged_layout = base.fc1_staged_smem_layout(base.fc1_staging_stage_cnt)
        self.smem_tensor = cute.make_tensor(
            cute.recast_ptr(staging_pointer, staged_layout.inner, dtype=base.fc1_output_dtype),
            staged_layout.outer,
        )
        # Kept unswizzled and untyped so fc1_quant can carve an fp32 amax-exchange view out of one
        # staging stage without going through the staging tensor's swizzle.
        self.staging_pointer = staging_pointer
        self.kernel_extension = kernel_extension
        self.fc1_tma_atom = tma_atom_fc1_output
        self.fc1_output = fc1_output
        self.fc1_output_sf = fc1_output_sf
        self.fc1_done_counter = fc1_done_counter
        self.optional_epi_args = optional_epi_args
        self._freeze()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # This object is a loop-invariant Python context wrapper, not a
        # dynamic value.  Keep it out of scf.while iter_args and reconstruct by
        # identity across region boundaries.  Any field that becomes a
        # loop-carried SSA value must be passed explicitly to __call__ instead
        # of being stored here.
        return []

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "SwapABFc1Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc1_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        # Only in-bound intermediate_downproj tiles signal; OOB -> null slot.
        needs_intermediate_guard = (
            self.intermediate_downproj is None
            or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
        )
        if cutlass.const_expr(needs_intermediate_guard):
            in_bound = (
                work_tile_info.tile_m_idx * self._EpilogueFc1IntermediateDownTileSize
                < self.fc1_output.shape[1]
            )
        else:
            in_bound = True
        slot = work_tile_info.cumulative_token_block_count + work_tile_info.tile_n_idx
        flag_address = Int64(0)
        if in_bound:
            flag_address = (self.fc1_done_counter.iterator + slot).toint()
        return flag_tracker.accumulate(
            next_work_tile_info.phase, self.fc1_epi_flag_batch, flag_address
        )

    @cute.jit
    def __call__(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, cta_tile_n)
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ):
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output, _ = self.kernel_extension.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info
        )
        # (tokens_this_expert, intermediate_down, 1)
        real_fc1_output_sf, _ = self.kernel_extension.get_gmem_tensor(
            "sfc", self.fc1_output_sf, work_tile_info
        )
        # subtile-irrevalent hoist out here.
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[work_tile_info.expert_idx]
        else:
            norm_const = None
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc1IntermediateGateUpTileSize, self._EpilogueTokenTileSize),
        )[None, None, 0, None]

        iket.range_push("epi.acc_wait")
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_pop()
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        if cutlass.const_expr(self.fc1_pipe and not self.fc1_pipe_full_only):
            self._run_subtiles_pipelined(
                work_tile_info=work_tile_info,
                tmem_acc_tensor_tiled_by_epi_tile=tmem_acc_tensor_tiled_by_epi_tile,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                is_odd_turn=is_odd_turn,
                valid_tokens=valid_tokens,
                fc1_output=real_fc1_output,
                fc1_output_sf=real_fc1_output_sf,
                alpha_val=alpha_val,
                norm_const=norm_const,
            )
        elif cutlass.const_expr(self.fc1_pipe):
            # MEGAMOE_EPI_FC1_PIPE_FULL=1: pipelined path only for full tiles (no dead subtile work);
            # partial tiles take the default loop.
            if valid_tokens >= cutlass.Int32(self.cta_tile_n):
                self._run_subtiles_pipelined(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor_tiled_by_epi_tile=tmem_acc_tensor_tiled_by_epi_tile,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    valid_tokens=valid_tokens,
                    fc1_output=real_fc1_output,
                    fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                )
            else:
                self._run_subtiles_default(
                    work_tile_info=work_tile_info,
                    tmem_acc_tensor_tiled_by_epi_tile=tmem_acc_tensor_tiled_by_epi_tile,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                    is_odd_turn=is_odd_turn,
                    valid_tokens=valid_tokens,
                    real_fc1_output=real_fc1_output,
                    real_fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                )
        else:
            self._run_subtiles_default(
                work_tile_info=work_tile_info,
                tmem_acc_tensor_tiled_by_epi_tile=tmem_acc_tensor_tiled_by_epi_tile,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                is_odd_turn=is_odd_turn,
                valid_tokens=valid_tokens,
                real_fc1_output=real_fc1_output,
                real_fc1_output_sf=real_fc1_output_sf,
                alpha_val=alpha_val,
                norm_const=norm_const,
            )

    @cute.jit
    def _run_subtiles_default(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor_tiled_by_epi_tile: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
        valid_tokens: cutlass.Int32,
        real_fc1_output: cute.Tensor,
        real_fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
    ):
        """The v5-epi FC1 subtile loop, unchanged (moved into a method for the FULL dispatch)."""
        if cutlass.const_expr(self.lean_tmem_uniform):
            # v9-fc1lean: TMEM base and subtile indices are warp-uniform; say so, so ptxas keeps the
            # LDTM / STTM addresses in uniform registers instead of re-deriving them per op (R2UR).
            tmem_base = tmem_acc_tensor_tiled_by_epi_tile.iterator
            tmem_acc_tensor_tiled_by_epi_tile = cute.make_tensor(
                cute.make_ptr(
                    tmem_base.dtype,
                    cute.arch.make_warp_uniform(tmem_base.toint()),
                    tmem_base.memspace,
                    assumed_align=16,
                ),
                tmem_acc_tensor_tiled_by_epi_tile.layout,
            )
        # Overlap path preloads two subtiles before releasing acc TMEM.
        unroll_tile_cnt = 2 if cutlass.const_expr(self.overlapping_accum) else 0
        remain_subtile_cnt = self.subtile_cnt - unroll_tile_cnt

        if cutlass.const_expr(unroll_tile_cnt > 0):
            subtile_idx_first = (cutlass.Int32(self.subtile_cnt) - is_odd_turn) % cutlass.Int32(
                self.subtile_cnt
            )
            subtile_idx_second = (
                cutlass.Int32(self.subtile_cnt + 1) - is_odd_turn
            ) % cutlass.Int32(self.subtile_cnt)
            if cutlass.const_expr(self.lean_tmem_uniform):
                subtile_idx_first = cute.arch.make_warp_uniform(subtile_idx_first)
                subtile_idx_second = cute.arch.make_warp_uniform(subtile_idx_second)

            # preload_subtile_first: subtile_idx_first's raw PRE-transpose acc, LDTM'd by
            # all 128 epi threads into 4 reg tensors == the 4 quadrants of the subtile's
            # (128 tmem_dp x 64 tmem_col) footprint. Only these raw-TMEM offsets are
            # guaranteed:
            #   reg[0]/reg[1], reg[2]/reg[3] : top vs bot      -> 16 apart in tmem_dp
            #   reg[0]/reg[2], reg[1]/reg[3] : 1st vs 2nd half -> 32 apart in tmem_col
            # (so reg[0..1] = the first 128x32, reg[2..3] = the second 128x32 of the 128x64.)
            # The per-lane (lane_idx, elem_idx) -> (tmem_dp, tmem_col) layout INSIDE each
            # reg tensor is opaque -- do not assume it; it only becomes well-defined once
            # the tmem transpose consumes them.
            preload_subtile_first: Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor] = (
                _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx_first]
                )
            )

            # Release acc to next MMA unconditionally.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

            # preload_subtile_second: same 128 tmem_dp x 64 tmem_col footprint, but for
            # subtile_idx_second (the other token subtile, not the 2nd col-half). Same
            # quadrant/offset invariants and opaque per-lane layout as above.
            preload_subtile_second: Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor] = (
                _TmemTranspose16x32Core.load_subtile_raw_acc(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx_second]
                )
            )

            # Both unrolled subtiles borrow tmem_subtile_second as workspace.
            preload_pair = (preload_subtile_first, preload_subtile_second)
            subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
            for i in cutlass.range_constexpr(unroll_tile_cnt):
                if subtile_idx_pair[i] * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                    self.run_subtile(
                        work_tile_info=work_tile_info,
                        subtile_idx=subtile_idx_pair[i],
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                            None, None, subtile_idx_second
                        ],
                        preload_acc=preload_pair[i],
                        fc1_output=real_fc1_output,
                        fc1_output_sf=real_fc1_output_sf,
                        alpha_val=alpha_val,
                        norm_const=norm_const,
                        stage_idx=cutlass.Int32(i) if cutlass.const_expr(self.fc1_stage2) else None,
                    )

        for i in cutlass.range(remain_subtile_cnt, unroll=1):
            real_i = i + unroll_tile_cnt
            if cutlass.const_expr(self.overlapping_accum):
                subtile_idx = (
                    cutlass.Int32(real_i + self.subtile_cnt) - is_odd_turn
                ) % cutlass.Int32(self.subtile_cnt)
            else:
                subtile_idx = cutlass.Int32(real_i)
            if cutlass.const_expr(self.lean_tmem_uniform):
                subtile_idx = cute.arch.make_warp_uniform(subtile_idx)

            if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                    preload_acc=None,
                    fc1_output=real_fc1_output,
                    fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                    stage_idx=(
                        cutlass.Int32(real_i % 2) if cutlass.const_expr(self.fc1_stage2) else None
                    ),
                    stage_reuse=self.fc1_stage2,
                )

        # Non-overlap-path release: at the natural task-tile boundary.
        if cutlass.const_expr(not self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    # -- v6-fc1pipe: software-pipelined FC1 subtile loop (MEGAMOE_EPI_FC1_PIPE=1) ----------------
    #
    # Per subtile k the FC1 epilogue is   L(k)  4x LDTM of the raw gate/up accumulator
    #                                     A(k)  activation work that precedes any TMEM transpose
    #                                     T(k)  the dependent STTM/LDTM transpose chain (+ the
    #                                           mid-chain activation of the default ordering)
    #                                     Q(k)  quant + scale-factor STG + STS, then the TMA store.
    # The default loop runs them back to back per subtile, and every subtile ends in a branch (TMA
    # issue by warp == subtile, valid-token check), so ptxas can never overlap one subtile's TMEM
    # latency with another's math. Here the loop is fully unrolled and skewed:
    #
    #     prologue : L(0) [release acc if overlapping] A(0)
    #     block k  : L(k+1)  T(k)  [last T: release acc if not overlapping]  A(k+1)  Q(k-1)
    #     tail     : Q(n-1)
    #
    # so each straight-line block carries three independent streams: the T(k) chain (TMEM latency),
    # A(k+1) (MUFU / FMA) and Q(k-1) (ALU + stores). Out-of-range subtiles are loaded, activated and
    # quantized unconditionally (garbage, never stored); only their scale-factor STG and TMA store
    # are predicated, which keeps the blocks branch-free. Their STS lands in their own (dead)
    # staging stage, which no TMA reads.
    #
    # TMEM workspace: T(k) transposes in place inside a subtile region whose raw data is already in
    # registers and that the next tile's MMA does not touch -- subtile k's own region, except for
    # the overlapping first subtile, which (exactly like the default path) borrows the second
    # subtile's region. L(k+1) is issued before T(k) in program order, so that borrowed region has
    # been read before T(0) overwrites it (TMEM ops of one warp execute in order, as the default
    # path already relies on). Values per element are computed by the same op sequence as the
    # default path, so the output is bit-identical.

    @cute.jit
    def _run_subtiles_pipelined(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor_tiled_by_epi_tile: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
        valid_tokens: cutlass.Int32,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
    ):
        subtile_cnt: cutlass.Constexpr[int] = self.subtile_cnt
        fine = self.opt_iket_fine
        if cutlass.const_expr(self.overlapping_accum and subtile_cnt < 2):
            raise ValueError("FC1 pipelined epilogue: overlapping accumulator needs >= 2 subtiles.")

        # Same subtile order as the default path (first / second, then the rest).
        sub_idx = []
        for i in cutlass.range_constexpr(subtile_cnt):
            if cutlass.const_expr(self.overlapping_accum):
                sub_idx.append(
                    (cutlass.Int32(i + subtile_cnt) - is_odd_turn) % cutlass.Int32(subtile_cnt)
                )
            else:
                sub_idx.append(cutlass.Int32(i))

        # Prologue: L(0), early release (overlap), A(0).
        _fine_push(fine, "fc1p.pro")
        raw_next = self._fc1_ldtm_subtile(tmem_acc_tensor_tiled_by_epi_tile[None, None, sub_idx[0]])
        if cutlass.const_expr(self.overlapping_accum):
            # The overlapping first subtile is in registers: hand the accumulator to the next MMA.
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)
        # A(0) sits behind a (uniform) branch on subtile 0's validity: besides skipping dead work,
        # the branch is the only thing that keeps ptxas from hoisting A(0)'s math between the four
        # L(0) LDTMs (it then reuses their destination registers, serialising the loads and pushing
        # the accumulator release back by ~150 instructions). Neither bar.sync nor bar.warp.sync
        # stops that hoisting. An out-of-range subtile 0 transposes / quantizes stale registers
        # that are never stored.
        pre_cur = self._fc1_pre_transpose_alloc()
        # Seed with the raw registers (no undefined phi inputs -> no spill at the merge).
        self._fc1_copy_pre(pre_cur, self._fc1_raw_as_pre(raw_next))
        if sub_idx[0] * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
            self._fc1_copy_pre(pre_cur, self._fc1_pre_transpose(raw_next, alpha_val))
        _fine_pop(fine)

        pending = None
        for k in cutlass.range_constexpr(subtile_cnt):
            _fine_push(fine, "fc1p.blk")
            # L(k+1)
            if cutlass.const_expr(k + 1 < subtile_cnt):
                raw_next = self._fc1_ldtm_subtile(
                    tmem_acc_tensor_tiled_by_epi_tile[None, None, sub_idx[k + 1]]
                )
            # T(k)
            if cutlass.const_expr(self.overlapping_accum and k == 0):
                workspace_idx = sub_idx[1]
            else:
                workspace_idx = sub_idx[k]
            two_token = self._fc1_transpose(
                pre_cur, tmem_acc_tensor_tiled_by_epi_tile[None, None, workspace_idx], alpha_val
            )
            if cutlass.const_expr(k + 1 == subtile_cnt and not self.overlapping_accum):
                # Last TMEM access of the tile: release before the remaining quant / stores.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
            # A(k+1)
            if cutlass.const_expr(k + 1 < subtile_cnt):
                pre_cur = self._fc1_pre_transpose(raw_next, alpha_val)
            # Q(k-1)
            if cutlass.const_expr(pending is not None):
                self._fc1_finish_subtile(
                    work_tile_info=work_tile_info,
                    subtile_idx=pending[1],
                    two_token=pending[0],
                    valid_tokens=valid_tokens,
                    fc1_output=fc1_output,
                    fc1_output_sf=fc1_output_sf,
                    norm_const=norm_const,
                )
            pending = (two_token, sub_idx[k])
            _fine_pop(fine)

        _fine_push(fine, "fc1p.tail")
        self._fc1_finish_subtile(
            work_tile_info=work_tile_info,
            subtile_idx=pending[1],
            two_token=pending[0],
            valid_tokens=valid_tokens,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
            norm_const=norm_const,
        )
        _fine_pop(fine)

    @cute.jit
    def _fc1_ldtm_subtile(self, tmem_subtile_tensor: cute.Tensor):
        """L(k): the default path's per-subtile raw LDTM (4x Ld16x64b.x16), verbatim."""
        gate_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
        up_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
        gate_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
        up_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
        tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[
            (None, 0), None
        ]
        tmem_subtile_tensor_in_first_load_view = cute.logical_divide(
            cute.zipped_divide(tmem_subtile_tensor_per_warp, (16, 32)), ((16, 32), 1)
        )
        atom = cute.make_copy_atom(tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32)
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 0]),
            wrap_into_copy_standard_layout(gate_token_0_32),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 1]),
            wrap_into_copy_standard_layout(up_token_0_32),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 2]),
            wrap_into_copy_standard_layout(gate_token_32_64),
        )
        cute.copy(
            atom,
            wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 3]),
            wrap_into_copy_standard_layout(up_token_32_64),
        )
        return (gate_token_0_32, up_token_0_32, gate_token_32_64, up_token_32_64)

    @cute.jit
    def _fc1_pre_transpose(self, raw, alpha_val: Optional[cutlass.Float32]):
        """A(k): the activation work that needs no TMEM transpose (same calls as run_subtile)."""
        gate_token_0_32, up_token_0_32, gate_token_32_64, up_token_32_64 = raw
        if cutlass.const_expr(self.opt_fc1_act_first):
            return (
                self.alpha_swiglu_clamp(gate_token_0_32, up_token_0_32, alpha_val),
                self.alpha_swiglu_clamp(gate_token_32_64, up_token_32_64, alpha_val),
            )
        return (
            self.alpha_swiglu_clamp(gate_token_0_32, up_token_0_32, alpha_val),
            gate_token_32_64,
            up_token_32_64,
        )

    def _fc1_pre_transpose_alloc(self):
        """Register tensors shaped like _fc1_pre_transpose's result."""
        n_out: cutlass.Constexpr[int] = 2 if self.opt_fc1_act_first else 3
        return tuple(cute.make_rmem_tensor((16,), cutlass.Float32) for _ in range(n_out))

    def _fc1_raw_as_pre(self, raw):
        if cutlass.const_expr(self.opt_fc1_act_first):
            return (raw[0], raw[2])
        return (raw[0], raw[2], raw[3])

    @cute.jit
    def _fc1_copy_pre(self, dst, src):
        for t in cutlass.range_constexpr(len(dst)):
            for r in cutlass.range_constexpr(16):
                dst[t][r] = src[t][r]

    @cute.jit
    def _fc1_transpose(self, pre, tmem_workspace: cute.Tensor, alpha_val: Optional[cutlass.Float32]):
        """T(k): TMEM transpose chain(s) of run_subtile, same TMEM op sequence and region."""
        if cutlass.const_expr(self.opt_fc1_act_first):
            act_tmem_trans = TmemTranspose32x32Inplace(
                tmem_workspace.iterator, reg_tensor_top=pre[0], reg_tensor_bot=pre[1]
            )
            return act_tmem_trans.from_r1_perm_until_last_store()
        token_32_64_tmem_trans = TmemTranspose32x32Inplace(
            tmem_workspace.iterator, reg_tensor_top=pre[1], reg_tensor_bot=pre[2]
        )
        gate_trans, up_trans = token_32_64_tmem_trans.from_r1_perm_until_last_store()
        token_32_64_pre_quant = self.alpha_swiglu_clamp(gate_trans, up_trans, alpha_val)
        token_0_32_tmem_trans = TmemTranspose16x32(
            tmem_workspace.iterator, Region.Top, reg_tensor=pre[0]
        )
        token_0_32_pre_quant = token_0_32_tmem_trans.from_r1_perm_until_last_store()
        return (token_0_32_pre_quant, token_32_64_pre_quant)

    @cute.jit
    def _fc1_finish_subtile(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        subtile_idx: cutlass.Int32,
        two_token,
        valid_tokens: cutlass.Int32,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        norm_const: Optional[cutlass.Float32],
    ):
        """Q(k): run_subtile's quant + SF store + STS + TMA store, stores predicated on validity."""
        valid = subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            topk_score_tensor, _ = self.kernel_extension.get_gmem_tensor(
                "topk", self.optional_epi_args.topk_scores, work_tile_info
            )
            token_base = (
                work_tile_info.tile_n_idx * self.cta_tile_n
                + subtile_idx * self._EpilogueTokenTileSize
                + self.lane_idx
            )
            # Out-of-range subtiles may sit past the topk tensor: read a dummy instead.
            topk_0 = cutlass.Float32(0.0)
            topk_1 = cutlass.Float32(0.0)
            if valid:
                topk_0 = cutlass.Float32(topk_score_tensor[token_base])
                topk_1 = cutlass.Float32(topk_score_tensor[token_base + 32])
            topk_scores = (topk_0, topk_1)
        else:
            topk_scores = None

        self.fc1_quant(
            work_tile_info=work_tile_info,
            two_token=two_token,
            topk_scores=topk_scores,
            norm_const=norm_const,
            intermediate_output_size=cute.size(fc1_output, 1),
            fc1_output_sf=fc1_output_sf,
            subtile_idx=subtile_idx,
            sf_store_pred=valid,
        )

        if valid:
            fc1_smem = self.smem_tensor[None, None, subtile_idx]
            fc1_gmem_cta_view = cute.flat_divide(
                fc1_output, (self.cta_tile_n, self.cta_tile_m // 2)
            )[None, None, work_tile_info.tile_n_idx, work_tile_info.tile_m_idx, 0]
            fc1_gmem_subtile_view = cute.flat_divide(
                fc1_gmem_cta_view,
                (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize),
            )[None, None, subtile_idx, 0]
            tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
                self.fc1_tma_atom,
                0,
                cute.make_layout(1),
                cute.group_modes(fc1_smem, 0, 2),
                cute.group_modes(fc1_gmem_subtile_view, 0, 2),
            )
            subtile_bar_id = subtile_idx + cutlass.Int32(
                SwapABGatedActEpilogue._EpilogueAsyncBarIdBase
            )
            tma_ready_to_read_smem_named_barrier = pipeline.NamedBarrier(
                barrier_id=subtile_bar_id, num_threads=self._EpilogueWarpCnt * 32
            )
            cute.arch.fence_proxy("async.shared", space="cta")
            if self.warp_idx == subtile_idx:
                tma_ready_to_read_smem_named_barrier.arrive_and_wait()
                with cute.arch.elect_one():
                    cute.copy(self.fc1_tma_atom, tma_smem_src, tma_gmem_dst)
                    cute.arch.cp_async_bulk_commit_group()
            else:
                tma_ready_to_read_smem_named_barrier.arrive()

    # -- v9-fc1lean: warp-private SMEM transpose (MEGAMOE_LEAN_SMEM_TRANS=1) ---------------------
    #
    # After the raw Ld16x64b.x16 loads (and the elementwise activation), lane L / reg e of token half
    # h holds (token = 32h + 2e + b, intermediate i) with b = (L >> 1) & 1, i = (L & 1) * 8 + (L >> 2)
    # (the TmemTranspose16x32 input distribution). The quant wants lane L = token 32h + L, reg e =
    # intermediate e (the TMEM transpose's output distribution). Scratch per warp, fp32 words:
    #     row(t) = (t >> 1) * 36 + (t & 1) * 16,  word(h, t, i) = h * 576 + row(t) + i
    # Stores (fixed e): rows 2e / 2e+1 are 16 words apart mod 32, so the 32 lanes hit 32 distinct
    # banks. Loads: lane L reads row L as 4 x LDS.128; within each 8-lane phase the rows start in 8
    # distinct 16-byte bank groups. Each side needs one per-thread base register (everything else is
    # an immediate). 2 x 576 words = 4.5 KiB per warp. Pure data movement: bit-identical.

    _LeanScratchWordsPerWarp: ClassVar[int] = 1152

    @cute.jit
    def _fc1_smem_transpose_wg2(self, act_0_32: cute.Tensor, act_32_64: cute.Tensor):
        """v10-epi2: _fc1_smem_transpose with half the scratch per warp (576 words): the two token
        halves go through the same 576-word plane one after the other. Same swizzle / addresses as
        the h = 0 plane of the single-WG version, so it stays bank-conflict free; pure data movement
        (bit-identical). Warp slot = epi_wg * 4 + warp_idx."""
        lane = self.lane_idx
        b = (lane >> cutlass.Int32(1)) & cutlass.Int32(1)
        inter = (lane & cutlass.Int32(1)) * cutlass.Int32(8) + (lane >> cutlass.Int32(2))
        store_base = b * cutlass.Int32(16) + inter
        load_base = (lane >> cutlass.Int32(1)) * cutlass.Int32(36) + (lane & cutlass.Int32(1)) * cutlass.Int32(
            16
        )
        half_words: cutlass.Constexpr[int] = self._LeanScratchWordsPerWarp // 2
        scratch_ptr = cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32) + (
            cutlass.Int32(self.fc1_staging_bytes // 4)
            + (self.epi_wg * cutlass.Int32(self._EpilogueWarpCnt) + self.warp_idx)
            * cutlass.Int32(half_words)
        )
        scratch = cute.make_tensor(scratch_ptr, cute.make_layout(half_words))
        acts = (act_0_32, act_32_64)
        outs = []
        for h in cutlass.range_constexpr(2):
            # Previous loads (other subtile, or the other half) by other lanes must finish first.
            cute.arch.sync_warp()
            for e in cutlass.range_constexpr(16):
                scratch[store_base + cutlass.Int32(e * 36)] = acts[h][e]
            cute.arch.sync_warp()
            out = cute.make_rmem_tensor((16,), cutlass.Float32)
            for c in cutlass.range_constexpr(4):
                src = mark_alignment(
                    cute.make_tensor(
                        scratch.iterator + (load_base + cutlass.Int32(4 * c)),
                        cute.make_layout(4),
                    ),
                    16,
                )
                vals = src.load()
                for j in cutlass.range_constexpr(4):
                    out[4 * c + j] = vals[j]
            outs.append(out)
        return outs[0], outs[1]

    @cute.jit
    def _fc1_smem_transpose(self, act_0_32: cute.Tensor, act_32_64: cute.Tensor):
        lane = self.lane_idx
        b = (lane >> cutlass.Int32(1)) & cutlass.Int32(1)
        inter = (lane & cutlass.Int32(1)) * cutlass.Int32(8) + (lane >> cutlass.Int32(2))
        store_base = b * cutlass.Int32(16) + inter
        load_base = (lane >> cutlass.Int32(1)) * cutlass.Int32(36) + (lane & cutlass.Int32(1)) * cutlass.Int32(
            16
        )
        scratch_ptr = cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32) + (
            cutlass.Int32(self.fc1_staging_bytes // 4)
            + self.warp_idx * cutlass.Int32(self._LeanScratchWordsPerWarp)
        )
        scratch = cute.make_tensor(scratch_ptr, cute.make_layout(self._LeanScratchWordsPerWarp))

        # The previous subtile's loads by other lanes must finish before this warp overwrites.
        cute.arch.sync_warp()
        acts = (act_0_32, act_32_64)
        for h in cutlass.range_constexpr(2):
            for e in cutlass.range_constexpr(16):
                scratch[store_base + cutlass.Int32(h * 576 + e * 36)] = acts[h][e]
        cute.arch.sync_warp()
        outs = []
        for h in cutlass.range_constexpr(2):
            out = cute.make_rmem_tensor((16,), cutlass.Float32)
            for c in cutlass.range_constexpr(4):
                src = mark_alignment(
                    cute.make_tensor(
                        scratch.iterator + (load_base + cutlass.Int32(h * 576 + 4 * c)),
                        cute.make_layout(4),
                    ),
                    16,
                )
                vals = src.load()
                for j in cutlass.range_constexpr(4):
                    out[4 * c + j] = vals[j]
            outs.append(out)
        return outs[0], outs[1]

    # -- v10-epi2: two epilogue warpgroups split one work tile --------------------------------
    #
    # TMEM: the SM100 overlapped accumulator gives each tile one "first" subtile whose columns the
    # NEXT tile's MMA overwrites, and the single-WG path also borrows the "second" subtile's TMEM as
    # transpose workspace (cols [0, 32) of a 64-col subtile; every transpose here only touches those
    # 32 columns of its 32 lanes). With two WGs:
    #   WG0: preloads "first" (overlap) into registers, transposes it in second[32, 64), then runs
    #        loop subtile (first + 2) in place.
    #   WG1: loads "second" into registers, transposes it in second[0, 32), then runs loop subtile
    #        (first + 3) in place.
    # WG0 may write second[32, 64) only after WG1 has read all of "second": WG1 bar.arrive(15) after
    # its LDTM wait, WG0 bar.sync(15). WG0 syncs BEFORE releasing the accumulator, so WG1's arrive
    # for tile i+1 (which needs MMA i+1, which needs this release) can never pair with WG0's sync for
    # tile i. The acc release counts all 8 epilogue warps, so the stage is reused only once both WGs
    # are past it; MMA i+1 only overwrites "first", which WG0 holds in registers.
    # With MEGAMOE_LEAN_SMEM_TRANS (v9) FC1 uses no TMEM workspace, so FC1 skips the handoff; each
    # warp of both WGs has its own 4.5 KiB SMEM transpose scratch (doubled scratch region).
    # SMEM staging: WG w owns stages {2w, 2w+1} (and TMA barriers 4+2w, 5+2w), so a stage is only
    # rewritten by the WG whose tile-boundary bulk wait drained it.

    @cute.jit
    def call_wg2(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, cta_tile_n)
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ):
        epi_wg = self.epi_wg
        real_fc1_output, _ = self.kernel_extension.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info
        )
        real_fc1_output_sf, _ = self.kernel_extension.get_gmem_tensor(
            "sfc", self.fc1_output_sf, work_tile_info
        )
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[work_tile_info.expert_idx]
        else:
            norm_const = None
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc1IntermediateGateUpTileSize, self._EpilogueTokenTileSize),
        )[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        subtile_cnt = cutlass.Int32(self.subtile_cnt)
        subtile_idx_first = (subtile_cnt - is_odd_turn) % subtile_cnt
        subtile_idx_second = (subtile_cnt + 1 - is_odd_turn) % subtile_cnt
        head_subtile_idx = (subtile_idx_first + epi_wg) % subtile_cnt
        loop_subtile_idx = (subtile_idx_first + 2 + epi_wg) % subtile_cnt

        preload_head = _TmemTranspose16x32Core.load_subtile_raw_acc(
            tmem_acc_tensor_tiled_by_epi_tile[None, None, head_subtile_idx]
        )
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(self.fc1_lean_scratch_bytes == 0):
            # TMEM-transpose path: WG0 will scribble on second[32, 64) -> wait for WG1's load.
            # (v9 lean SMEM transpose touches no TMEM after the loads: no handoff needed.)
            _epi_wg2_handoff(epi_wg)
        acc_pipeline.consumer_release(acc_consumer_state)

        # WG0 -> second[32, 64), WG1 -> second[0, 32) (unused by the lean SMEM transpose).
        workspace = cute.flat_divide(
            tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx_second],
            (self._EpilogueFc1IntermediateGateUpTileSize, 32),
        )[None, None, 0, cutlass.Int32(1) - epi_wg]
        stage_base = epi_wg * cutlass.Int32(2)
        if head_subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
            self.run_subtile(
                work_tile_info=work_tile_info,
                subtile_idx=head_subtile_idx,
                tmem_subtile_tensor=workspace,
                preload_acc=preload_head,
                fc1_output=real_fc1_output,
                fc1_output_sf=real_fc1_output_sf,
                alpha_val=alpha_val,
                norm_const=norm_const,
                stage_idx=stage_base,
            )
        if loop_subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
            self.run_subtile(
                work_tile_info=work_tile_info,
                subtile_idx=loop_subtile_idx,
                tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, loop_subtile_idx],
                preload_acc=None,
                fc1_output=real_fc1_output,
                fc1_output_sf=real_fc1_output_sf,
                alpha_val=alpha_val,
                norm_const=norm_const,
                stage_idx=stage_base + cutlass.Int32(1),
            )

    # -- v30-n224: two full accumulators, 32-token units (mma tile N = 224) ----------------------
    #
    # TMEM: 2 x 224 accumulator columns + 64 SF columns = 512. The two stages are disjoint, so the MMA
    # of tile i+1 never touches tile i's columns: there is no head / overlap subtile and no early
    # release; each WG releases the stage after its last LDTM of the tile (the release counts all 8
    # epilogue warps). A tile is 7 units of 32 tokens. WG w owns the units u with (u + w + p) even,
    # p = the tile's accumulator stage (alternates every tile): WG0 gets {0,2,4,6} on even stages and
    # {1,3,5} on odd stages, WG1 the complement, so the odd unit alternates between the WGs and a
    # partial tile's leading units are split across both WGs.
    # Units are processed in pairs (two tokens per thread: the FC1 quant keeps its two-token packed
    # path); a WG owning {1,3,5} runs the pair (1,3) and then unit 5 alone. A pair uses one 64-token
    # staging stage (rows 0..31 = first unit, 32..63 = second) and issues one 32-token TMA store per
    # valid unit; WG w's stages are 2w (first pair) and 2w + 1 (second pair or the single unit), so
    # each stage is written once per tile and drained by its WG's tile-boundary bulk wait.
    # Per element the math is the default path's (same activation, quant, scale rounding); only the
    # token grouping changes. Bit-identical.

    @cute.jit
    def _n224_ldtm_unit(self, tmem_unit: cute.Tensor):
        """Raw gate / up accumulator of one 32-token unit (warp rows 0..15 / 16..31), the
        load_subtile_raw_acc first-half loads."""
        atom_ld16x64 = cute.make_copy_atom(
            tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), _TmemTranspose16x32Core._io_dtype
        )
        ptr = tmem_unit.iterator
        half_lane_off = 16 * _TmemTranspose16x32Core._TmemRowStride
        gate = cute.make_rmem_tensor((16,), cutlass.Float32)
        up = cute.make_rmem_tensor((16,), cutlass.Float32)
        cute.copy(
            atom_ld16x64,
            cute.make_tensor(ptr, _TmemTranspose16x32Core._tmem_layout(16, 32)),
            _TmemTranspose16x32Core._rmem_copy_view(gate, 16),
        )
        cute.copy(
            atom_ld16x64,
            cute.make_tensor(ptr + half_lane_off, _TmemTranspose16x32Core._tmem_layout(16, 32)),
            _TmemTranspose16x32Core._rmem_copy_view(up, 16),
        )
        return gate, up

    @cute.jit
    def _n224_smem_transpose(self, acts):
        """_fc1_smem_transpose_wg2 for one or two 32-token halves (same slot, same swizzle)."""
        lane = self.lane_idx
        b = (lane >> cutlass.Int32(1)) & cutlass.Int32(1)
        inter = (lane & cutlass.Int32(1)) * cutlass.Int32(8) + (lane >> cutlass.Int32(2))
        store_base = b * cutlass.Int32(16) + inter
        load_base = (lane >> cutlass.Int32(1)) * cutlass.Int32(36) + (lane & cutlass.Int32(1)) * cutlass.Int32(
            16
        )
        half_words: cutlass.Constexpr[int] = self._LeanScratchWordsPerWarp // 2
        if cutlass.const_expr(self.epi_wg is None):
            warp_slot = self.warp_idx  # single epilogue WG: slots 0..3
        else:
            warp_slot = self.epi_wg * cutlass.Int32(self._EpilogueWarpCnt) + self.warp_idx
        scratch_ptr = cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32) + (
            cutlass.Int32(self.fc1_staging_bytes // 4) + warp_slot * cutlass.Int32(half_words)
        )
        scratch = cute.make_tensor(scratch_ptr, cute.make_layout(half_words))
        outs = []
        for h in cutlass.range_constexpr(len(acts)):
            cute.arch.sync_warp()
            for e in cutlass.range_constexpr(16):
                scratch[store_base + cutlass.Int32(e * 36)] = acts[h][e]
            cute.arch.sync_warp()
            out = cute.make_rmem_tensor((16,), cutlass.Float32)
            for c in cutlass.range_constexpr(4):
                src = mark_alignment(
                    cute.make_tensor(
                        scratch.iterator + (load_base + cutlass.Int32(4 * c)),
                        cute.make_layout(4),
                    ),
                    16,
                )
                vals = src.load()
                for j in cutlass.range_constexpr(4):
                    out[4 * c + j] = vals[j]
            outs.append(out)
        return outs

    @cute.jit
    def _n224_units(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        raws,  # per unit: (gate, up) raw accumulator registers
        units,  # per unit: runtime unit index in [0, 7)
        valid_tokens: cutlass.Int32,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
        stage_idx: cutlass.Int32,
    ):
        """Activation + SMEM transpose + quant + stores of one unit pair (or a single unit)."""
        n_units: cutlass.Constexpr[int] = len(units)
        unit_tokens: cutlass.Constexpr[int] = self._EpilogueUnitTokens
        fine = self.opt_iket_fine
        _fine_push(fine, "fc1.act")
        acts = [self.alpha_swiglu_clamp(raws[h][0], raws[h][1], alpha_val) for h in range(n_units)]
        _fine_pop(fine)
        _fine_push(fine, "fc1.trans")
        tokens = self._n224_smem_transpose(acts)
        _fine_pop(fine)
        unit_valid = [units[h] * cutlass.Int32(unit_tokens) < valid_tokens for h in range(n_units)]
        token_idx = [
            work_tile_info.tile_n_idx * self.cta_tile_n + units[h] * cutlass.Int32(unit_tokens) + self.lane_idx
            for h in range(n_units)
        ]
        topk_scores = None
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            topk_score_tensor, _ = self.kernel_extension.get_gmem_tensor(
                "topk", self.optional_epi_args.topk_scores, work_tile_info
            )
            scores = []
            for h in cutlass.range_constexpr(n_units):
                score = cutlass.Float32(0.0)
                if unit_valid[h]:
                    score = cutlass.Float32(topk_score_tensor[token_idx[h]])
                scores.append(score)
            topk_scores = tuple(scores)

        _fine_push(fine, "fc1.quant")
        self._n224_quant(
            work_tile_info=work_tile_info,
            tokens=tokens,
            units=units,
            unit_valid=unit_valid,
            token_idx=token_idx,
            topk_scores=topk_scores,
            norm_const=norm_const,
            intermediate_output_size=cute.size(fc1_output, 1),
            fc1_output_sf=fc1_output_sf,
            stage_idx=stage_idx,
        )
        _fine_pop(fine)
        _fine_push(fine, "fc1.store")
        stage = self.smem_tensor[None, None, stage_idx]
        # (token, intermediate_down, l=1) -> box_tokens x 64-intermediate boxes (32 for N = 224; 16 for
        # N = 240, two boxes per unit: tile i starts at token 240 i = box 15 i, and a 32-row box could cross into the
        # next tile / expert since experts are only padded to 16 rows).
        box_tokens: cutlass.Constexpr[int] = self._fc1_box_tokens()
        boxes_per_unit: cutlass.Constexpr[int] = unit_tokens // box_tokens
        fc1_gmem_units = cute.flat_divide(
            fc1_output, (box_tokens, self._EpilogueFc1IntermediateDownTileSize)
        )
        stage_halves = cute.flat_divide(
            stage, (box_tokens, self._EpilogueFc1IntermediateDownTileSize)
        )
        tma_args = []
        box_valid = []
        for h in cutlass.range_constexpr(n_units):
            for q in cutlass.range_constexpr(boxes_per_unit):
                tma_args.append(
                    cpasync.tma_partition(
                        self.fc1_tma_atom,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(stage_halves[None, None, h * boxes_per_unit + q, 0], 0, 2),
                        cute.group_modes(
                            fc1_gmem_units[
                                None,
                                None,
                                work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n // box_tokens)
                                + units[h] * cutlass.Int32(boxes_per_unit)
                                + cutlass.Int32(q),
                                work_tile_info.tile_m_idx,
                                0,
                            ],
                            0,
                            2,
                        ),
                    )
                )
                if cutlass.const_expr(boxes_per_unit == 1):
                    box_valid.append(unit_valid[h])
                else:
                    box_valid.append(
                        units[h] * cutlass.Int32(unit_tokens) + cutlass.Int32(q * box_tokens) < valid_tokens
                    )
        stage_bar = pipeline.NamedBarrier(
            barrier_id=stage_idx + cutlass.Int32(SwapABGatedActEpilogue._EpilogueAsyncBarIdBase),
            num_threads=self._EpilogueWarpCnt * 32,
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        if self.warp_idx == stage_idx:
            stage_bar.arrive_and_wait()
            with cute.arch.elect_one():
                for b in cutlass.range_constexpr(len(tma_args)):
                    if box_valid[b]:
                        cute.copy(self.fc1_tma_atom, tma_args[b][0], tma_args[b][1])
                cute.arch.cp_async_bulk_commit_group()
        else:
            stage_bar.arrive()
        _fine_pop(fine)

    @cute.jit
    def _n224_quant(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tokens,  # per unit: (16,) fp32, lane = token, reg = intermediate
        units,
        unit_valid,
        token_idx,
        topk_scores,
        norm_const: Optional[cutlass.Float32],
        intermediate_output_size: cutlass.Int32,
        fc1_output_sf: cute.Tensor,
        stage_idx: cutlass.Int32,
    ):
        """fc1_quant (in-thread NVFP4) for one or two 32-token units; same per-token math."""
        n_units: cutlass.Constexpr[int] = len(tokens)
        values_per_token = cute.size(tokens[0])
        amax_exchange = None
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            # MX (sf_vec 32): a scale block straddles warps w / w ^ 1 of this WG, exactly as in fc1_quant
            # (same QuantImpl direction, same exchange plane in this unit pair's own staging stage -- dead now: the
            # previous tile's TMA store of it was drained at the tile boundary, and the STS below follows the
            # retire barrier). The plane is (2 token chunks, 128 threads); a single unit quantizes a copy of its
            # token as the second chunk (never stored) so every thread runs the same exchange.
            amax_exchange = mark_alignment(
                cute.make_tensor(
                    cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32)
                    + stage_idx * cutlass.Int32(self.fc1_staging_stage_bytes // 4),
                    cute.make_layout(
                        (self.fc1_amax_token_chunks, self._EpilogueWarpCnt * 32),
                        stride=(1, self.fc1_amax_token_chunks),
                    ),
                ),
                16,
            )
            quant = QuantImpl(
                self.quant_kind,
                "regs_in_pair_threads",
                lane_idx=self.lane_idx,
                warp_idx=self.warp_idx,
                pair_exchange_barrier=self._pair_barrier(),
            )
        elif cutlass.const_expr(self.lean_fc1_quant):
            quant = QuantImpl(
                self.quant_kind,
                "regs_in_thread",
                packed_scale=self.lean_qmul2,
                amax_tree=self.lean_amax_tree,
                lean_nvfp4=True,
            )
        else:
            quant = QuantImpl(
                self.quant_kind,
                "regs_in_thread",
                packed_scale=self.lean_qmul2,
                amax_tree=self.lean_amax_tree,
            )
        intermediate_idx = (
            work_tile_info.tile_m_idx * (self.cta_tile_m // 2)
            + self.warp_idx * self._EpilogueFc1IntermediateDownPerWarp
        )
        # SF rows are laid out 256 per 224-token tile (helpers.utils.sf_row_in_expert): token 32 u + lane of tile i
        # sits at SF row 256 i + 32 u + lane, i.e. 32-row chunk 8 i + u of the ((32, 4), rest) token mode.
        sf_chunks = [
            work_tile_info.tile_n_idx * cutlass.Int32(8) + units[h] for h in range(n_units)
        ]
        if cutlass.const_expr(self.lean_sfidx):
            sf_token_coords = [
                ((self.lane_idx, c & cutlass.Int32(3)), c >> cutlass.Int32(2)) for c in sf_chunks
            ]
        else:
            sf_token_coords = [c * cutlass.Int32(32) + self.lane_idx for c in sf_chunks]
        smem_stage = self.smem_tensor[None, None, stage_idx]
        smem_tiled = cute.zipped_divide(smem_stage, (1, values_per_token))
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=values_per_token * self.fc1_output_dtype.width,
        )
        fold_topk = self._fc1_topk_fold_ok(topk_scores, quant, values_per_token)
        # the paired exchange plane always has 2 token chunks: pad a single unit with a copy of itself.
        quant_units: cutlass.Constexpr[int] = 2 if self.needs_pair_amax_exchange else n_units
        weighted = cute.make_rmem_tensor((quant_units * values_per_token,), cutlass.Float32)
        for half in cutlass.range_constexpr(quant_units):
            src_half = min(half, n_units - 1)
            tok = tokens[src_half]
            base = half * values_per_token
            if cutlass.const_expr(topk_scores is not None and not fold_topk):
                topk_pair = (topk_scores[src_half], topk_scores[src_half])
                for i in cutlass.range_constexpr(0, values_per_token, 2):
                    w0, w1 = cute.arch.mul_packed_f32x2((tok[i], tok[i + 1]), topk_pair)
                    weighted[base + i] = w0
                    weighted[base + i + 1] = w1
            else:
                for i in cutlass.range_constexpr(0, values_per_token):
                    weighted[base + i] = tok[i]
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            data_regs, sf_regs = quant(weighted, norm_const=norm_const, smem_intermediate=amax_exchange)
            # Retire the exchange before any thread writes this stage (the slots are the stage's own bytes).
            self._pair_barrier().arrive_and_wait()
        elif cutlass.const_expr(fold_topk):
            data_regs, sf_regs = quant(weighted, norm_const=norm_const, block_scales=topk_scores)
        else:
            data_regs, sf_regs = quant(weighted, norm_const=norm_const)
        data_by_token = cute.zipped_divide(data_regs, (values_per_token,))
        # Both warps of an MX pair derive the same scale; only the even one stores it (as fc1_quant).
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            stores_scale_factor = self.warp_idx % 2 == 0
        else:
            stores_scale_factor = True
        needs_intermediate_guard: cutlass.Constexpr[bool] = (
            self.intermediate_downproj is None
            or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
        )
        for half in cutlass.range_constexpr(n_units):
            if unit_valid[half] and stores_scale_factor:
                if cutlass.const_expr(needs_intermediate_guard):
                    if intermediate_idx < intermediate_output_size:
                        fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]
                else:
                    fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]
            smem_thread_row = smem_tiled[(0, None), (self.lane_idx + 32 * half, self.warp_idx)]
            cute.copy(
                store_atom, cute.coalesce(data_by_token[None, half]), cute.coalesce(smem_thread_row)
            )

    @cute.jit
    def call_n224(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, 224)
        acc_pipeline,
        acc_consumer_state,
        tile_parity: cutlass.Int32,
    ):
        epi_wg = self.epi_wg
        real_fc1_output, _ = self.kernel_extension.get_gmem_tensor(
            "c", self.fc1_output, work_tile_info
        )
        real_fc1_output_sf, _ = self.kernel_extension.get_gmem_tensor(
            "sfc", self.fc1_output_sf, work_tile_info
        )
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[work_tile_info.expert_idx]
        else:
            norm_const = None
        # (cta_tile_m, 224) -> (128, 32, unit)
        tmem_units = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc1IntermediateGateUpTileSize, self._EpilogueUnitTokens),
        )[None, None, 0, None]

        iket.range_push("epi.acc_wait")
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_pop()
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        unit_tokens = cutlass.Int32(self._EpilogueUnitTokens)

        first = (tile_parity + epi_wg) & cutlass.Int32(1)
        stage_base = epi_wg * cutlass.Int32(2)
        pair_cnt = cutlass.Int32(2) - first
        if cutlass.const_expr(self.unit_cnt == 8):
            # N = 240: 8 units, both WGs run two pairs ({0,2},{4,6} / {1,3},{5,7}), no single unit.
            pair_cnt = cutlass.Int32(2)
        for k in cutlass.range(pair_cnt, unroll=1):
            unit_a = first + cutlass.Int32(4) * k
            unit_b = unit_a + cutlass.Int32(2)
            if unit_a * unit_tokens < valid_tokens:
                raw_a = self._n224_ldtm_unit(tmem_units[None, None, unit_a])
                raw_b = self._n224_ldtm_unit(tmem_units[None, None, unit_b])
                if k == cutlass.Int32(1):
                    # Units 4 and 6: this WG's last accumulator read of the tile.
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                self._n224_units(
                    work_tile_info=work_tile_info,
                    raws=(raw_a, raw_b),
                    units=(unit_a, unit_b),
                    valid_tokens=valid_tokens,
                    fc1_output=real_fc1_output,
                    fc1_output_sf=real_fc1_output_sf,
                    alpha_val=alpha_val,
                    norm_const=norm_const,
                    stage_idx=stage_base + k,
                )
            else:
                if k == cutlass.Int32(1):
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
        if cutlass.const_expr(self.unit_cnt % 2 == 1):
            if first == cutlass.Int32(1):
                unit_c = cutlass.Int32(self.unit_cnt - 2)
                if unit_c * unit_tokens < valid_tokens:
                    raw_c = self._n224_ldtm_unit(tmem_units[None, None, unit_c])
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    self._n224_units(
                        work_tile_info=work_tile_info,
                        raws=(raw_c,),
                        units=(unit_c,),
                        valid_tokens=valid_tokens,
                        fc1_output=real_fc1_output,
                        fc1_output_sf=real_fc1_output_sf,
                        alpha_val=alpha_val,
                        norm_const=norm_const,
                        stage_idx=stage_base + cutlass.Int32(1),
                    )
                else:
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def call_n224_1wg(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,  # (cta_tile_m, 224 | 240)
        acc_pipeline,
        acc_consumer_state,
    ):
        """call_n224 for ONE epilogue WG (bulk FC2 store configs): units in pairs (0,1), (2,3), (4,5),
        (6[,7]) -> FC1 staging stage 0..3 (each stage used once per tile; run() drains them at the tile boundary). The
        accumulator is released after the last valid unit pair's TMEM loads (the other stage feeds the next MMA)."""
        real_fc1_output, _ = self.kernel_extension.get_gmem_tensor("c", self.fc1_output, work_tile_info)
        real_fc1_output_sf, _ = self.kernel_extension.get_gmem_tensor(
            "sfc", self.fc1_output_sf, work_tile_info
        )
        if cutlass.const_expr(self.optional_epi_args.fc1_alpha is not None):
            alpha_val = self.optional_epi_args.fc1_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        if cutlass.const_expr(self.optional_epi_args.fc1_norm_const is not None):
            norm_const = self.optional_epi_args.fc1_norm_const[work_tile_info.expert_idx]
        else:
            norm_const = None
        tmem_units = cute.flat_divide(
            tmem_acc_tensor,
            (self._EpilogueFc1IntermediateGateUpTileSize, self._EpilogueUnitTokens),
        )[None, None, 0, None]

        iket.range_push("epi.acc_wait")
        acc_pipeline.consumer_wait(acc_consumer_state)
        iket.range_pop()
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        unit_tokens = cutlass.Int32(self._EpilogueUnitTokens)
        unit_cnt: cutlass.Constexpr[int] = self.unit_cnt
        pair_cnt: cutlass.Constexpr[int] = (unit_cnt + 1) // 2
        # Number of unit pairs holding a valid token; the release follows that pair's TMEM loads.
        valid_pairs = (valid_tokens + cutlass.Int32(2 * self._EpilogueUnitTokens - 1)) // cutlass.Int32(
            2 * self._EpilogueUnitTokens
        )
        for p in cutlass.range_constexpr(pair_cnt):
            unit_a = cutlass.Int32(2 * p)
            if unit_a * unit_tokens < valid_tokens:
                if cutlass.const_expr(2 * p + 1 < unit_cnt):
                    unit_b = cutlass.Int32(2 * p + 1)
                    raw_a = self._n224_ldtm_unit(tmem_units[None, None, unit_a])
                    raw_b = self._n224_ldtm_unit(tmem_units[None, None, unit_b])
                    if valid_pairs == cutlass.Int32(p + 1):
                        cute.arch.fence_view_async_tmem_load()
                        acc_pipeline.consumer_release(acc_consumer_state)
                    self._n224_units(
                        work_tile_info=work_tile_info,
                        raws=(raw_a, raw_b),
                        units=(unit_a, unit_b),
                        valid_tokens=valid_tokens,
                        fc1_output=real_fc1_output,
                        fc1_output_sf=real_fc1_output_sf,
                        alpha_val=alpha_val,
                        norm_const=norm_const,
                        stage_idx=cutlass.Int32(p),
                    )
                else:
                    raw_a = self._n224_ldtm_unit(tmem_units[None, None, unit_a])
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    self._n224_units(
                        work_tile_info=work_tile_info,
                        raws=(raw_a,),
                        units=(unit_a,),
                        valid_tokens=valid_tokens,
                        fc1_output=real_fc1_output,
                        fc1_output_sf=real_fc1_output_sf,
                        alpha_val=alpha_val,
                        norm_const=norm_const,
                        stage_idx=cutlass.Int32(p),
                    )
        if valid_pairs == cutlass.Int32(0):
            # No valid token (cannot happen for a scheduled tile; keeps the release unconditional).
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        subtile_idx: cutlass.Int32,
        # (intermedaite_gateup_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        # Rmems preloaded from tmem, contract with downstream tmem trans. Do not assume mapping here.
        preload_acc: Tuple[cute.Tensor, cute.Tensor, cute.Tensor, cute.Tensor],
        # (tokens_this_expert, intermediate_down, 1)
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        alpha_val: Optional[cutlass.Float32],
        norm_const: Optional[cutlass.Float32],
        stage_idx: Optional[cutlass.Int32] = None,
        stage_reuse: bool = False,
    ):
        # v10-epi2: the SMEM staging stage / TMA-store barrier / issuing warp follow ``stage_idx``
        # (WG-owned slots) instead of the token subtile; single-WG mode keeps them equal.
        if cutlass.const_expr(stage_idx is None):
            stage_idx = subtile_idx
        if cutlass.const_expr(self.optional_epi_args.topk_scores is not None):
            # This means we need to perform DeepGEMM computation graph, topk_score at fc1 pre-quant
            topk_score_tensor, _ = self.kernel_extension.get_gmem_tensor(
                "topk", self.optional_epi_args.topk_scores, work_tile_info
            )  # (tokens_this_expert)
        else:
            topk_score_tensor = None

        # Mapping of the transposed accumulator for orthodox NVFP4 output:
        # (epi_tid, val_id) -> (token_idx, intermediate_down_idx)
        # token_idx = epi_tid % 32 + val_id // 16 * 32
        # intermediate_down_idx = val_id % 16 + epi_tid // 32 * 16
        # Each thread holds (intermediate_down_16, token_2):(1, 16)

        # Step -1: preload topk scores.
        current_two_token_idices = (
            work_tile_info.tile_n_idx * self.cta_tile_n
            + subtile_idx * self._EpilogueTokenTileSize
            + self.lane_idx,
            work_tile_info.tile_n_idx * self.cta_tile_n
            + subtile_idx * self._EpilogueTokenTileSize
            + self.lane_idx
            + 32,
        )
        if cutlass.const_expr(topk_score_tensor is not None):
            topk_scores = (
                topk_score_tensor[current_two_token_idices[0]],
                topk_score_tensor[current_two_token_idices[1]],
            )
        else:
            topk_scores = None

        # Step 0: load tmem
        if cutlass.const_expr(preload_acc is not None):
            gate_token_0_32, up_token_0_32, gate_token_32_64, up_token_32_64 = preload_acc
        else:
            gate_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
            up_token_0_32 = cute.make_rmem_tensor((16,), cutlass.Float32)
            gate_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
            up_token_32_64 = cute.make_rmem_tensor((16,), cutlass.Float32)
            # Although hardcode is not right, but since the whole tmem transpose is too tricky, I have to hardcode...
            # (epi_tile_m, epi_tile_n) -> (warp_local_epi_tile_m, epi_tile_n)
            # tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[(None,
            # self.warp_idx), None]
            tmem_subtile_tensor_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[
                (None, 0), None
            ]
            # (warp_local_epi_tile_m, epi_tile_n) -> (((16, 32), 1), (2, 2))
            tmem_subtile_tensor_in_first_load_view = cute.logical_divide(
                cute.zipped_divide(tmem_subtile_tensor_per_warp, (16, 32)), ((16, 32), 1)
            )
            atom = cute.make_copy_atom(tcgen05.Ld16x64bOp(tcgen05.Repetition.x16), cutlass.Float32)
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 0]),
                wrap_into_copy_standard_layout(gate_token_0_32),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 1]),
                wrap_into_copy_standard_layout(up_token_0_32),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 2]),
                wrap_into_copy_standard_layout(gate_token_32_64),
            )
            cute.copy(
                atom,
                wrap_into_copy_standard_layout(tmem_subtile_tensor_in_first_load_view[None, 3]),
                wrap_into_copy_standard_layout(up_token_32_64),
            )

        fine = self.opt_iket_fine
        if cutlass.const_expr(self.fc1_lean_scratch_bytes > 0):
            # v9-fc1lean (MEGAMOE_LEAN_SMEM_TRANS): ACT_FIRST math, then a warp-private SMEM transpose.
            _fine_push(fine, "fc1.act")
            token_0_32_act = self.alpha_swiglu_clamp(gate_token_0_32, up_token_0_32, alpha_val)
            token_32_64_act = self.alpha_swiglu_clamp(
                gate_token_32_64, up_token_32_64, alpha_val
            )
            _fine_pop(fine)
            _fine_push(fine, "fc1.trans")
            if cutlass.const_expr(self.epi_wg2):
                token_0_32_pre_quant, token_32_64_pre_quant = self._fc1_smem_transpose_wg2(
                    token_0_32_act, token_32_64_act
                )
            else:
                token_0_32_pre_quant, token_32_64_pre_quant = self._fc1_smem_transpose(
                    token_0_32_act, token_32_64_act
                )
            _fine_pop(fine)
        elif cutlass.const_expr(self.opt_fc1_act_first):
            # v5-epi: the activation is elementwise and gate / up share one register distribution
            # in both token halves, so run it on both halves BEFORE transposing and move only the
            # 16 activations per half. One 32x32 in-place transpose (top = tokens 0..31 through
            # the Top region, bot = tokens 32..63 through the Bottom region) replaces the 32x32
            # raw gate/up transpose plus the separate 16x32 activation transpose. Top and Bottom
            # apply the same register permutation (the default path relies on it to pair gate
            # with up), so the outputs land in the default path's (lane = token, elem =
            # intermediate) layout; the values are bit-identical (same per-element math, and a
            # TMEM transpose only moves 32-bit words).
            _fine_push(fine, "fc1.act")
            token_0_32_act = self.alpha_swiglu_clamp(gate_token_0_32, up_token_0_32, alpha_val)
            token_32_64_act = self.alpha_swiglu_clamp(
                gate_token_32_64, up_token_32_64, alpha_val
            )
            _fine_pop(fine)
            _fine_push(fine, "fc1.trans")
            act_tmem_trans = TmemTranspose32x32Inplace(
                tmem_subtile_tensor.iterator,
                reg_tensor_top=token_0_32_act,
                reg_tensor_bot=token_32_64_act,
            )
            token_0_32_pre_quant, token_32_64_pre_quant = (
                act_tmem_trans.from_r1_perm_until_last_store()
            )
            _fine_pop(fine)
        else:
            _fine_push(fine, "fc1.act_trans")
            # Step 1: perform swiglu on the first part, interleave with the second's 32x32 tmem transpose.
            token_0_32_pre_quant_pre_trans = self.alpha_swiglu_clamp(
                gate_token_0_32, up_token_0_32, alpha_val
            )

            # gate_token_32_64 / up_token_32_64 are already in the transpose input
            # distribution (see TmemTranspose16x32 / load_subtile_raw_acc).
            token_32_64_tmem_trans = TmemTranspose32x32Inplace(
                tmem_subtile_tensor.iterator,
                reg_tensor_top=gate_token_32_64,
                reg_tensor_bot=up_token_32_64,
            )

            # Transpose output: each lane holds (token_1, intermediate_16); tmem_dp
            # = lane_idx (token), tmem_col = elem_idx (intermediate output idx).
            gate_token_32_64_trans_pre_act, up_token_32_64_trans_pre_act = (
                token_32_64_tmem_trans.from_r1_perm_until_last_store()
            )

            token_32_64_pre_quant = self.alpha_swiglu_clamp(
                gate_token_32_64_trans_pre_act, up_token_32_64_trans_pre_act, alpha_val
            )

            token_0_32_tmem_trans = TmemTranspose16x32(
                tmem_subtile_tensor.iterator, Region.Top, reg_tensor=token_0_32_pre_quant_pre_trans
            )
            token_0_32_pre_quant = token_0_32_tmem_trans.from_r1_perm_until_last_store()
            _fine_pop(fine)

        # Step 2: Quant
        if cutlass.const_expr(stage_reuse):
            # MEGAMOE_MX_FC1_STAGE2: this staging stage still feeds position p - 2's TMA store.
            cute.arch.cp_async_bulk_wait_group(0, read=True)
            self._pair_barrier().arrive_and_wait()
        _fine_push(fine, "fc1.quant")
        self.fc1_quant(
            work_tile_info=work_tile_info,
            two_token=(token_0_32_pre_quant, token_32_64_pre_quant),
            topk_scores=topk_scores,
            norm_const=norm_const,
            intermediate_output_size=cute.size(fc1_output, 1),
            fc1_output_sf=fc1_output_sf,
            subtile_idx=subtile_idx,
            stage_idx=stage_idx,
        )

        _fine_pop(fine)
        _fine_push(fine, "fc1.store")
        # Step 3: TMASTG
        # (token_64, intermeidate_64)
        fc1_smem = self.smem_tensor[None, None, stage_idx]
        # (token, intermediate_down, l=1) -> (cta_token, cta_intermediate_down)
        fc1_gmem_cta_view = cute.flat_divide(fc1_output, (self.cta_tile_n, self.cta_tile_m // 2))[
            None, None, work_tile_info.tile_n_idx, work_tile_info.tile_m_idx, 0
        ]
        # (cta_token, cta_intermediate_down) -> (token_64, intermediate_64)
        fc1_gmem_subtile_view = cute.flat_divide(
            fc1_gmem_cta_view,
            (self._EpilogueTokenTileSize, self._EpilogueFc1IntermediateDownTileSize),
        )[None, None, subtile_idx, 0]
        tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
            self.fc1_tma_atom,
            0,
            cute.make_layout(1),
            cute.group_modes(fc1_smem, 0, 2),
            cute.group_modes(fc1_gmem_subtile_view, 0, 2),
        )

        subtile_bar_id = stage_idx + cutlass.Int32(SwapABGatedActEpilogue._EpilogueAsyncBarIdBase)
        tma_ready_to_read_smem_named_barrier = pipeline.NamedBarrier(
            barrier_id=subtile_bar_id, num_threads=self._EpilogueWarpCnt * 32
        )
        cute.arch.fence_proxy("async.shared", space="cta")
        if self.warp_idx == stage_idx:
            tma_ready_to_read_smem_named_barrier.arrive_and_wait()
            with cute.arch.elect_one():
                # if work_tile_info.tile_m_idx * (self.cta_tile_m // 2) < cute.size(fc1_output, 1):
                cute.copy(self.fc1_tma_atom, tma_smem_src, tma_gmem_dst)
                cute.arch.cp_async_bulk_commit_group()
        else:
            tma_ready_to_read_smem_named_barrier.arrive()
        _fine_pop(fine)

    @cute.jit
    def alpha_swiglu_clamp(
        self,
        gate_rmem: cute.Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        up_rmem: cute.Tensor,  # Raw fc1 acc (pre-dequant); even-size 1D fp32 rmem
        alpha_val: Optional[cutlass.Float32],
    ) -> cute.Tensor:
        # ── Input contract checks (compile-time): fp32, 1D, even-count, rmem ──
        # Wrapped in const_expr so the DSL evaluates them at trace time and the
        # raise fires during compilation rather than emitting a runtime branch.
        for _name, _t in (("gate_rmem", gate_rmem), ("up_rmem", up_rmem)):
            if cutlass.const_expr(_t.element_type is not cutlass.Float32):
                raise TypeError(
                    f"alpha_swiglu_clamp: {_name} must be Float32, got {_t.element_type}"
                )
            if cutlass.const_expr(_t.memspace != AddressSpace.rmem):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} must be a register (rmem) tensor, got address space {_t.memspace}"
                )
            if cutlass.const_expr(cute.rank(_t) != 1):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} must be 1D, got rank {cute.rank(_t)}"
                )
            if cutlass.const_expr(cute.size(_t) % 2 != 0):
                raise ValueError(
                    f"alpha_swiglu_clamp: {_name} element count must be even, got {cute.size(_t)}"
                )
        if cutlass.const_expr(cute.size(gate_rmem) != cute.size(up_rmem)):
            raise ValueError(
                "alpha_swiglu_clamp: gate_rmem and up_rmem must have equal size, got "
                f"{cute.size(gate_rmem)} vs {cute.size(up_rmem)}"
            )

        # gate_rmem / up_rmem are the RAW fc1 fp32 accumulator (pre-dequant).
        # Order follows the NVFP4 -> fp32 -> SwiGLU contract and MUST be:
        #
        #   1. dequant:  gate = alpha * gate_raw ; up = alpha * up_raw
        #      (alpha = expert-wise global scale on the acc; None => alpha == 1.)
        #   2. clamp the DEQUANTED (real) values, gpt-oss ``_apply_gate`` style:
        #        gate = min(gate, +limit)           (upper bound only)
        #        up   = clamp(up, -limit, +limit)   (symmetric)
        #   3. gated activation, either SwiGLU or SiTU (mutually exclusive; SiTU has no clamp):
        #        SwiGLU: out = (up + swiglu_beta) * gate * sigmoid(swiglu_alpha * gate)
        #                Standard SwiGLU uses swiglu_alpha=1 and swiglu_beta=0.
        #        SiTU:   out = beta*tanh(gate/beta)*sigmoid(gate) * linear_beta*tanh(up/linear_beta)
        #                sigmoid(x) = rcp(1 + exp2(-x * log2e))
        #
        # The symmetric up-clamp is a single ``min.xorsign.abs.f32`` (magnitude
        # min(|up|, limit), sign = sign(up)^sign(limit) = sign(up) since limit>=0);
        # the gate-clamp is a plain ``min.f32``. ``.xorsign.abs`` has no f32x2 form,
        # so dequant+clamp run scalar while the swiglu core stays packed f32x2.
        n = cute.size(gate_rmem)
        out = cute.make_rmem_tensor((n,), cutlass.Float32)
        log2_e = 1.4426950408889634

        activation_alpha = 1.0 if self.swiglu_alpha is None else self.swiglu_alpha
        neg_activation_alpha_log2e = -activation_alpha * log2_e
        neg_log2e_pair = (
            cutlass.Float32(neg_activation_alpha_log2e),
            cutlass.Float32(neg_activation_alpha_log2e),
        )
        one_pair = (cutlass.Float32(1.0), cutlass.Float32(1.0))
        if cutlass.const_expr(self.swiglu_beta is not None):
            beta_pair = (cutlass.Float32(self.swiglu_beta), cutlass.Float32(self.swiglu_beta))
        if cutlass.const_expr(self.gate_up_clamp is not None):
            limit = cutlass.Float32(self.gate_up_clamp)

        if cutlass.const_expr(self.situ_fast):
            situ_consts = self._situ_fast_consts(alpha_val)

        for i in cutlass.range_constexpr(0, n, 2):
            g0 = gate_rmem[i]
            g1 = gate_rmem[i + 1]
            u0 = up_rmem[i]
            u1 = up_rmem[i + 1]

            # 1) dequant raw acc to real values (skip entirely when alpha is None, or when the
            #    SiTU fast core folds alpha into its exp constants -- SiTU has no clamp).
            if cutlass.const_expr(alpha_val is not None and not self.situ_fold_alpha):
                alpha_pair = (alpha_val, alpha_val)
                g0, g1 = cute.arch.mul_packed_f32x2((g0, g1), alpha_pair)
                u0, u1 = cute.arch.mul_packed_f32x2((u0, u1), alpha_pair)

            # 2) clamp the real values (skip when no clamp configured).
            if cutlass.const_expr(self.gate_up_clamp is not None):
                # gate upper-clamp: min(gate, +limit)
                g0 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [g0.ir_value(), limit.ir_value()],
                        "min.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                g1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [g1.ir_value(), limit.ir_value()],
                        "min.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                # up symmetric-clamp: clamp(up, -limit, +limit) in one instruction
                u0 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [u0.ir_value(), limit.ir_value()],
                        "min.xorsign.abs.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )
                u1 = cutlass.Float32(
                    llvm.inline_asm(
                        cutlass.Float32.mlir_type,
                        [u1.ir_value(), limit.ir_value()],
                        "min.xorsign.abs.f32 $0, $1, $2;",
                        "=f,f,f",
                        has_side_effects=True,
                        is_align_stack=False,
                        asm_dialect=llvm.AsmDialect.AD_ATT,
                    )
                )

            # 3) gated activation on the dequanted (and clamped) real values.
            #    sigmoid(x) = rcp(1 + exp2(-x * log2e)) -- shared by both cores.
            def _sigmoid(p0, p1):
                neg = cute.arch.mul_packed_f32x2((p0, p1), neg_log2e_pair)
                e = (cute.math.exp2(neg[0], fastmath=True), cute.math.exp2(neg[1], fastmath=True))
                d = cute.arch.add_packed_f32x2(e, one_pair)
                return (cute.arch.rcp_approx(d[0]), cute.arch.rcp_approx(d[1]))

            if cutlass.const_expr(self.situ_fast):
                # v2-situ: 3-MUFU SiTU core (MEGAMOE_SITU_FAST=1); see _situ_fast_pair.
                out_pair = self._situ_fast_pair(g0, g1, u0, u1, situ_consts)
            else:
                sigmoid_pair = _sigmoid(g0, g1)

                if cutlass.const_expr(self.situ_beta is None):
                    # The SwiGLUBias offset follows the up clamp.
                    if cutlass.const_expr(self.swiglu_beta is not None):
                        u0, u1 = cute.arch.add_packed_f32x2((u0, u1), beta_pair)
                    ug = cute.arch.mul_packed_f32x2((u0, u1), (g0, g1))
                    out_pair = cute.arch.mul_packed_f32x2(ug, sigmoid_pair)
                else:
                    # SiTU (Kimi K3), matching HF ``modeling_kimi.py``:
                    #     situ_gate = beta        * tanh(gate / beta) * sigmoid(gate)
                    #     situ_up   = linear_beta * tanh(up   / linear_beta)
                    #     out       = situ_gate * situ_up
                    #
                    # ``tanh(z) = 2 * sigmoid(2z) - 1`` keeps the whole core on the packed f32x2 path --
                    # there is no packed tanh, so calling one would force this loop back to scalar.
                    #
                    # beta * tanh(x/beta) = beta * (2*sigmoid(2x/beta) - 1) = 2*beta*sigmoid(2x/beta) - beta
                    # so the reciprocals and the 2*beta factors fold at trace time.
                    inv_2beta = cutlass.Float32(2.0 / self.situ_beta)
                    two_beta = cutlass.Float32(2.0 * self.situ_beta)
                    neg_beta = cutlass.Float32(-self.situ_beta)
                    inv_2lbeta = cutlass.Float32(2.0 / self.situ_linear_beta)
                    two_lbeta = cutlass.Float32(2.0 * self.situ_linear_beta)
                    neg_lbeta = cutlass.Float32(-self.situ_linear_beta)

                    gs = _sigmoid(*cute.arch.mul_packed_f32x2((g0, g1), (inv_2beta, inv_2beta)))
                    tanh_g = cute.arch.add_packed_f32x2(
                        cute.arch.mul_packed_f32x2(gs, (two_beta, two_beta)), (neg_beta, neg_beta)
                    )

                    us = _sigmoid(*cute.arch.mul_packed_f32x2((u0, u1), (inv_2lbeta, inv_2lbeta)))
                    tanh_u = cute.arch.add_packed_f32x2(
                        cute.arch.mul_packed_f32x2(us, (two_lbeta, two_lbeta)), (neg_lbeta, neg_lbeta)
                    )

                    situ_gate = cute.arch.mul_packed_f32x2(tanh_g, sigmoid_pair)
                    out_pair = cute.arch.mul_packed_f32x2(situ_gate, tanh_u)

            out[i] = out_pair[0]
            out[i + 1] = out_pair[1]

        return out

    # -- v2-situ: 3-MUFU SiTU core (MEGAMOE_SITU_FAST=1) ------------------------------------
    #
    # SiTU (Kimi K3): out = beta*tanh(g/beta) * sigmoid(g) * lbeta*tanh(u/lbeta), g/u dequanted.
    # With y = e^(-2|g|/beta), z = e^(-2|u|/lbeta) (both in (0, 1], so nothing can overflow) and
    # e^(-|g|) = y^(beta/2) (= y*y for Kimi's beta = 4, no extra MUFU):
    #     beta*tanh(g/beta)   = sgn(g) * beta*(1 - y) / (1 + y)
    #     sigmoid(g)          = (g >= 0 ? 1 : y^2) / (1 + y^2)
    #     lbeta*tanh(u/lbeta) = sgn(u) * lbeta*(1 - z) / (1 + z)
    #     out = sel(g >= 0, P, -P*y^2) * rcp((1 + y) * (1 + y^2) * (1 + z)),
    #           P = beta*(1 - y) * copysign(lbeta*(1 - z), u)
    # MUFU per output: 2 ex2 + 1 rcp = 3 (base: 3 ex2 + 3 rcp = 6). The rcp argument is in [1, 8].
    # check_situ.py (this variant dir) runs exactly this sequence against the base sequence on GPU:
    # max abs error ~1/3 of the base path's on every input set, same NaN/inf/zero behavior.
    # With situ_fold_alpha the dequant scale is folded into the exp constants (alpha > 0: it is the
    # product of the positive NVFP4 global scales), so g/u here are the raw accumulators and only
    # their signs and |.| are used.

    def _situ_fast_consts(self, alpha_val):
        log2_e = 1.4426950408889634
        beta = self.situ_beta
        lbeta = self.situ_linear_beta
        cg = cutlass.Float32(-2.0 * log2_e / beta)
        cu = cutlass.Float32(-2.0 * log2_e / lbeta)
        cs = cutlass.Float32(-log2_e)  # only for beta/2 not in {1, 2}
        if self.situ_fold_alpha and alpha_val is not None:  # trace-time
            cg = alpha_val * cg
            cu = alpha_val * cu
            cs = alpha_val * cs
        return dict(
            cg=(cg, cg),
            cu=(cu, cu),
            cs=(cs, cs),
            neg_beta=(cutlass.Float32(-beta), cutlass.Float32(-beta)),
            beta=(cutlass.Float32(beta), cutlass.Float32(beta)),
            neg_lbeta=(cutlass.Float32(-lbeta), cutlass.Float32(-lbeta)),
            lbeta=(cutlass.Float32(lbeta), cutlass.Float32(lbeta)),
            one=(cutlass.Float32(1.0), cutlass.Float32(1.0)),
        )

    def _situ_fast_pair(self, g0, g1, u0, u1, c):
        # e^(-|g|) = y^k with k = beta/2: k in {1, 2} reuses y; otherwise one extra ex2.
        k = self.situ_beta / 2.0
        ag = cute.arch.mul_packed_f32x2((_abs_f32(g0), _abs_f32(g1)), c["cg"])
        au = cute.arch.mul_packed_f32x2((_abs_f32(u0), _abs_f32(u1)), c["cu"])
        y = (cute.math.exp2(ag[0], fastmath=True), cute.math.exp2(ag[1], fastmath=True))
        z = (cute.math.exp2(au[0], fastmath=True), cute.math.exp2(au[1], fastmath=True))
        # numerators: beta*(1 - y) >= 0 and lbeta*(1 - z) >= 0 (one FFMA each), sign of u copied in.
        pg = cute.arch.fma_packed_f32x2(y, c["neg_beta"], c["beta"])
        pu = cute.arch.fma_packed_f32x2(z, c["neg_lbeta"], c["lbeta"])
        pu = (_copysign_f32(u0, pu[0]), _copysign_f32(u1, pu[1]))
        # e^(-|g|) (ys), 1 + e^(-|g|) (d2) and -e^(-|g|) (nys).
        if k == 2.0:  # trace-time
            d2 = cute.arch.fma_packed_f32x2(y, y, c["one"])
            nys = cute.arch.mul_packed_f32x2(y, (-y[0], -y[1]))
        elif k == 1.0:
            d2 = cute.arch.add_packed_f32x2(y, c["one"])
            nys = (-y[0], -y[1])
        else:
            as_ = cute.arch.mul_packed_f32x2((_abs_f32(g0), _abs_f32(g1)), c["cs"])
            ys = (cute.math.exp2(as_[0], fastmath=True), cute.math.exp2(as_[1], fastmath=True))
            d2 = cute.arch.add_packed_f32x2(ys, c["one"])
            nys = (-ys[0], -ys[1])
        d1 = cute.arch.add_packed_f32x2(y, c["one"])
        d3 = cute.arch.add_packed_f32x2(z, c["one"])
        den = cute.arch.mul_packed_f32x2(cute.arch.mul_packed_f32x2(d1, d2), d3)
        r = (cute.arch.rcp_approx(den[0]), cute.arch.rcp_approx(den[1]))
        t = cute.arch.mul_packed_f32x2(pg, pu)
        tn = cute.arch.mul_packed_f32x2(t, nys)
        m = (_select_ge0_f32(g0, t[0], tn[0]), _select_ge0_f32(g1, t[1], tn[1]))
        return cute.arch.mul_packed_f32x2(m, r)

    def _fc1_topk_fold_ok(self, topk_scores, quant, values_per_token) -> bool:
        """Trace-time (plain Python) gate for MEGAMOE_FC1_TOPK_FOLD."""
        return bool(
            self.fc1_topk_fold
            and topk_scores is not None
            and not self.needs_pair_amax_exchange
            and not quant._is_combine
            and quant.data_dtype is cutlass.Float4E2M1FN
            and quant.sf_vec_size == values_per_token
        )

    def _pair_barrier(self) -> pipeline.NamedBarrier:
        """The 128-thread barrier of this epilogue WG (the paired amax exchange stays within a WG:
        partner warp = warp ^ 1 of the same WG). WG0 / single WG: id 1; WG1 (MEGAMOE_EPI_WG2): id 14, the
        same per-WG ids the tile-boundary rendezvous uses (never in flight together with a subtile's quant)."""
        if cutlass.const_expr(self.epi_wg is None):
            return SwapABGatedActEpilogue.epilogue_sync_barrier()
        base = SwapABGatedActEpilogue
        return pipeline.NamedBarrier(
            barrier_id=cutlass.Int32(base._EpilogueSyncWaitBarId)
            + self.epi_wg * cutlass.Int32(base._EpilogueWg1SyncWaitBarId - base._EpilogueSyncWaitBarId),
            num_threads=32 * base._EpilogueWarpCnt,
        )

    @cute.jit
    def fc1_quant(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        two_token: Tuple[
            cute.Tensor, cute.Tensor
        ],  # two rmem tensor, each fp32 @ (token_1, intermediate_16)
        topk_scores: Optional[Tuple[cutlass.Float32, cutlass.Float32]],
        norm_const: Optional[cutlass.Float32],
        intermediate_output_size: cutlass.Int32,
        fc1_output_sf: cute.Tensor,  # MoE domain (token_this_rank, intermediate_down, 1)
        subtile_idx: cutlass.Int32,
        sf_store_pred=None,
        stage_idx: Optional[cutlass.Int32] = None,
    ):
        if cutlass.const_expr(stage_idx is None):
            stage_idx = subtile_idx
        # v6-fc1pipe: ``sf_store_pred`` (a runtime Boolean) predicates the scale-factor global store;
        # the pipelined loop quantizes out-of-range subtiles too and must not store their scales.
        # ``two_token`` are the two post-swiglu, transposed token rmem tensors; each lane holds one
        # token's 16 intermediate-output values. half 0 -> token (lane), half 1 -> (lane+32).
        #
        # Those 16 values are a whole scale block at sf_vec 16 (nvfp4) but only half of one at
        # sf_vec 32 (mx), where the other half sits in the paired warp -- hence the two directions
        # below. Both are handed the two tokens in one call so the paired path needs a single
        # exchange barrier per subtile rather than one per token.
        #
        # Per token (ported from PostSwigluHalf._gen_sfc_quantize + stg_sfc + r2s):
        #   1. (Path A) pre-multiply topk weight into the values, if present.
        #   2. block quant -> data regs + one scale factor (QuantImpl's job).
        #   3. write the scale factor to fc1_output_sf[token, intermediate_idx, 0]
        #      (plain scalar store; predicated unless statically in-bound).
        #   4. STS the quantized values into this subtile's shared output stage.
        # norm_const is treated like alpha_val: None => behaves as 1.0 (factors const-elided, not
        # multiplied by 1.0). It only exists for nvfp4; an e8m0 scale absorbs the rescale itself.
        values_per_token = cute.size(two_token[0])
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            # The exchange slots live in THIS subtile's staging stage. That stage is dead right
            # now -- the previous tile's copy of it was drained at the tile boundary, and this
            # tile writes it only after the retire barrier below -- whereas every other stage may
            # still have a TMA store reading it. Borrowing any other stage would corrupt it.
            amax_exchange = mark_alignment(
                cute.make_tensor(
                    cute.recast_ptr(self.staging_pointer, dtype=cutlass.Float32)
                    + stage_idx * cutlass.Int32(self.fc1_staging_stage_bytes // 4),
                    cute.make_layout(
                        (self.fc1_amax_token_chunks, self._EpilogueWarpCnt * 32),
                        stride=(1, self.fc1_amax_token_chunks),
                    ),
                ),
                16,
            )
            quant = QuantImpl(
                self.quant_kind,
                "regs_in_pair_threads",
                lane_idx=self.lane_idx,
                warp_idx=self.warp_idx,
                pair_exchange_barrier=self._pair_barrier(),
            )
        else:
            amax_exchange = None
            if cutlass.const_expr(self.lean_fc1_quant):
                quant = QuantImpl(
                    self.quant_kind,
                    "regs_in_thread",
                    packed_scale=self.lean_qmul2,
                    amax_tree=self.lean_amax_tree,
                    lean_nvfp4=True,
                )
            else:
                quant = QuantImpl(
                    self.quant_kind,
                    "regs_in_thread",
                    packed_scale=self.lean_qmul2,
                    amax_tree=self.lean_amax_tree,
                )

        # Both warps of a pair derive the same scale, so only the even one stores it. The scale
        # plane folds any coordinate inside a block onto that block's slot, so the per-warp base
        # addresses the right entry for either vec size.
        intermediate_idx = (
            work_tile_info.tile_m_idx * (self.cta_tile_m // 2)
            + self.warp_idx * self._EpilogueFc1IntermediateDownPerWarp
        )
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            stores_scale_factor = self.warp_idx % 2 == 0
        else:
            stores_scale_factor = True
        subtile_token_start = (
            work_tile_info.tile_n_idx * self.cta_tile_n + subtile_idx * self._EpilogueTokenTileSize
        )
        token_idx_pair = (
            subtile_token_start + self.lane_idx,
            subtile_token_start + self.lane_idx + 32,
        )
        # v9-fc1lean (MEGAMOE_LEAN_SFIDX): the scale-factor plane's token mode is ((32, 4), rest),
        # i.e. row = r0 + 32 * r1 + 128 * r2. With a 128-multiple tile N the tile base is 128-aligned,
        # so token = lane + 32 * (2 * subtile + half) + tile_n * N decomposes without any div / mod.
        if cutlass.const_expr(self.lean_sfidx and self.cta_tile_n % 128 == 0):
            sf_token_coords = []
            for half in cutlass.range_constexpr(2):
                chunk = subtile_idx * cutlass.Int32(2) + cutlass.Int32(half)
                sf_token_coords.append(
                    (
                        (self.lane_idx, chunk & cutlass.Int32(3)),
                        work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n // 128)
                        + (chunk >> cutlass.Int32(2)),
                    )
                )
        else:
            sf_token_coords = token_idx_pair

        # This subtile's (token, intermediate) shared output stage, tiled into (1, 16) blocks so
        # each thread's cells slice out directly (zipped_divide + slice; avoids the ambiguous
        # local_tile surface).
        smem_stage = self.smem_tensor[None, None, stage_idx]
        # (token_64, intermediate_down_64) -> ((1, 16), (token_tile_size, warp_cnt))
        smem_tiled = cute.zipped_divide(smem_stage, (1, values_per_token))
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.fc1_output_dtype,
            num_bits_per_copy=values_per_token * self.fc1_output_dtype.width,
        )

        # v2-situ (MEGAMOE_FC1_TOPK_FOLD=1): fold the topk weight into the per-block NVFP4 scale
        # (one block per token here) instead of multiplying each element. Only for the in-thread
        # nvfp4 path where a token's values are exactly one scale block.
        fold_topk = self._fc1_topk_fold_ok(topk_scores, quant, values_per_token)

        # 1) topk-weight pre-multiply (Path A) into a weighted scratch, both tokens back to back.
        weighted = cute.make_rmem_tensor((2 * values_per_token,), cutlass.Float32)
        for half in cutlass.range_constexpr(2):
            tok = two_token[half]
            base = half * values_per_token
            if cutlass.const_expr(topk_scores is not None and not fold_topk):
                topk_pair = (topk_scores[half], topk_scores[half])
                for i in cutlass.range_constexpr(0, values_per_token, 2):
                    w0, w1 = cute.arch.mul_packed_f32x2((tok[i], tok[i + 1]), topk_pair)
                    weighted[base + i] = w0
                    weighted[base + i + 1] = w1
            else:
                for i in cutlass.range_constexpr(0, values_per_token):
                    weighted[base + i] = tok[i]

        # 2) Core block quant. One scale factor per token either way; the paired direction spends
        #    its exchange barrier inside this call.
        if cutlass.const_expr(fold_topk):
            data_regs, sf_regs = quant(
                weighted,
                norm_const=norm_const,
                smem_intermediate=amax_exchange,
                block_scales=(topk_scores[0], topk_scores[1]),
            )
        else:
            data_regs, sf_regs = quant(
                weighted, norm_const=norm_const, smem_intermediate=amax_exchange
            )
        data_by_token = cute.zipped_divide(data_regs, (values_per_token,))

        # Retire the exchange: its slots are this stage's own bytes, so no thread may start
        # writing the stage until every thread has read its partner's amax. Sitting after the
        # quantization rather than before it puts the sync latency behind work that is already in
        # flight (the e8m0 path alone goes through MUFU).
        if cutlass.const_expr(self.needs_pair_amax_exchange):
            self._pair_barrier().arrive_and_wait()

        for half in cutlass.range_constexpr(2):
            # 3) scale-factor store (predicate const-elided when statically in-bound, mirroring
            #    signal_fc1_done's intermediate predicate).
            if cutlass.const_expr(sf_store_pred is not None):
                if stores_scale_factor and sf_store_pred:
                    if cutlass.const_expr(
                        self.intermediate_downproj is None
                        or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
                    ):
                        if intermediate_idx < intermediate_output_size:
                            fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]
                    else:
                        fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]
            else:
                if stores_scale_factor:
                    if cutlass.const_expr(
                        self.intermediate_downproj is None
                        or self.intermediate_downproj % self.cluster_tile_intermediate_downproj != 0
                    ):
                        if intermediate_idx < intermediate_output_size:
                            fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]
                    else:
                        fc1_output_sf[sf_token_coords[half], intermediate_idx, 0] = sf_regs[half]

            # 4) STS the quantized values into this subtile's shared output stage.
            # ((1, 16), (token_tile_size, warp_cnt)) -> (16)
            smem_thread_row = smem_tiled[(0, None), (self.lane_idx + 32 * half, self.warp_idx)]
            cute.copy(
                store_atom, cute.coalesce(data_by_token[None, half]), cute.coalesce(smem_thread_row)
            )


# -- v10-epi2 helpers ------------------------------------------------------------------------


def _tcgen05_fence(kind: str) -> None:
    llvm.inline_asm(
        None,
        [],
        f"tcgen05.fence::{kind};",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def _epi_wg2_handoff(epi_wg: cutlass.Int32):
    """WG1 has finished reading its head subtile (the tile's "second") -> WG0 may use second[32, 64).

    Callers have already issued tcgen05.wait::ld. WG1 arrives without waiting; WG0 waits.
    """
    handoff = pipeline.NamedBarrier(
        barrier_id=SwapABGatedActEpilogue._EpilogueWgHandoffBarId,
        num_threads=2 * 32 * SwapABGatedActEpilogue._EpilogueWarpCnt,
    )
    _tcgen05_fence("before_thread_sync")
    if epi_wg == cutlass.Int32(0):
        handoff.arrive_and_wait()
    else:
        handoff.arrive()
    _tcgen05_fence("after_thread_sync")


# -- v14-epi3 helpers -------------------------------------------------------------------------


def _i32_as_f32(x) -> cutlass.Float32:
    return cutlass.Float32(llvm.bitcast(T.f32(), cutlass.Int32(x).ir_value()))


def _f32_as_i32(x) -> cutlass.Int32:
    return cutlass.Int32(llvm.bitcast(T.i32(), cutlass.Float32(x).ir_value()))


def _prmt_b32(a, b, selector: int) -> cutlass.Int32:
    return cutlass.Int32(
        llvm.inline_asm(
            T.i32(),
            [cutlass.Int32(a).ir_value(), cutlass.Int32(b).ir_value()],
            f"prmt.b32 $0, $1, $2, {selector:#x};",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _max3_f32(a, b, c, abs_a: bool) -> cutlass.Float32:
    """max(a', |b|, |c|) with a' = |a| if abs_a else a (sm_100+ three-input max -> FMNMX3)."""
    if abs_a:
        body = "{\n\t.reg .f32 a, b, c;\n\tabs.f32 a, $1;\n\tabs.f32 b, $2;\n\tabs.f32 c, $3;\n\tmax.f32 $0, a, b, c;\n\t}"
    else:
        body = "{\n\t.reg .f32 b, c;\n\tabs.f32 b, $2;\n\tabs.f32 c, $3;\n\tmax.f32 $0, $1, b, c;\n\t}"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(a).ir_value(),
                cutlass.Float32(b).ir_value(),
                cutlass.Float32(c).ir_value(),
            ],
            body,
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _max2_f32(a, b, abs_a: bool) -> cutlass.Float32:
    if abs_a:
        body = "{\n\t.reg .f32 a, b;\n\tabs.f32 a, $1;\n\tabs.f32 b, $2;\n\tmax.f32 $0, a, b;\n\t}"
    else:
        body = "{\n\t.reg .f32 b;\n\tabs.f32 b, $2;\n\tmax.f32 $0, $1, b;\n\t}"
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(a).ir_value(), cutlass.Float32(b).ir_value()],
            body,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _cvt_e4m3x2_roundtrip(s0, s1):
    """(e4m3x2 bits [lo = s0, hi = s1] as i16, f32(e4m3(s0)), f32(e4m3(s1))), satfinite, rn."""
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i16(), T.f32(), T.f32()]),
        [cutlass.Float32(s0).ir_value(), cutlass.Float32(s1).ir_value()],
        "{\n\t.reg .b32 h2;\n\t.reg .b16 hl, hh;\n\t"
        "cvt.rn.satfinite.e4m3x2.f32 $0, $4, $3;\n\t"
        "cvt.rn.f16x2.e4m3x2 h2, $0;\n\t"
        "mov.b32 {hl, hh}, h2;\n\t"
        "cvt.f32.f16 $1, hl;\n\t"
        "cvt.f32.f16 $2, hh;\n\t}",
        "=h,=f,=f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return (
        cutlass.Int16(llvm.extractvalue(T.i16(), res, [0])),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [1])),
        cutlass.Float32(llvm.extractvalue(T.f32(), res, [2])),
    )


def _abs_max_f32(values) -> cutlass.Float32:
    """max_i |values[i]| (+0 for all-zero, NaN only if all NaN) as a FMNMX3 chain.

    A chain, not a tree: each step folds two fresh inputs (abs as free operand modifiers), so n inputs
    take ceil((n - 1) / 2) instructions either way and the chain needs no extra abs."""
    n = len(values)
    if n == 1:
        return _max2_f32(values[0], values[0], True)
    if n == 2:
        return _max2_f32(values[0], values[1], True)
    acc = _max3_f32(values[0], values[1], values[2], True)
    i = 3
    while i + 1 < n:
        acc = _max3_f32(acc, values[i], values[i + 1], False)
        i += 2
    if i < n:
        acc = _max2_f32(acc, values[i], False)
    return acc


def _stsm_x4_trans_b16(addr: cutlass.Int32, offset: int, r0, r1, r2, r3) -> None:
    """stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [addr + offset], {r0, r1, r2, r3}."""
    llvm.inline_asm(
        None,
        [
            cutlass.Int32(addr).ir_value(),
            cutlass.Int32(r0).ir_value(),
            cutlass.Int32(r1).ir_value(),
            cutlass.Int32(r2).ir_value(),
            cutlass.Int32(r3).ir_value(),
        ],
        f"stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [$0+{offset}], {{$1, $2, $3, $4}};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


def _lds128_b32_off(addr: cutlass.Int32, offset: int):
    """ld.shared.v4.b32 from [addr + offset] (ordered after the side-effecting STSM asm)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32()] * 4),
        [cutlass.Int32(addr).ir_value()],
        f"ld.shared.v4.b32 {{$0, $1, $2, $3}}, [$4+{offset}];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )
    return tuple(cutlass.Int32(llvm.extractvalue(T.i32(), result, [i])) for i in range(4))


# -- v2-situ scalar helpers (pure; no side effects so they schedule freely) ---------------------


def _abs_f32(x) -> cutlass.Float32:
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(x).ir_value()],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _copysign_f32(sign_src, mag) -> cutlass.Float32:
    """|mag| with the sign bit of ``sign_src`` (PTX copysign: sign of a into b)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [cutlass.Float32(sign_src).ir_value(), cutlass.Float32(mag).ir_value()],
            "copysign.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def _select_ge0_f32(x, a, b) -> cutlass.Float32:
    """``x >= 0 ? a : b`` (NaN x selects b; -0.0 selects a)."""
    return cutlass.Float32(
        llvm.inline_asm(
            T.f32(),
            [
                cutlass.Float32(x).ir_value(),
                cutlass.Float32(a).ir_value(),
                cutlass.Float32(b).ir_value(),
            ],
            "{\n\t.reg .pred p;\n\tsetp.ge.f32 p, $1, 0f00000000;\n\tselp.f32 $0, $2, $3, p;\n\t}",
            "=f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dataclasses.dataclass(frozen=True)
class Fc2ProcessPipeline:
    tmem_acc_load: Callable
    f2fp: Callable
    post_f2fp_reorder: Callable
    store_function: Callable
    # Kept as a finer-grained, elem-level reading aid for the store-out layout
    # (never evaluated); ``store_out_mapping`` is the per-issue form that the
    # router actually evaluates at runtime to drive metadata / pointer math.
    fc2_cta_tile_mapping: FunctionMapping
    store_out_mapping: FunctionMapping
    require_tmem_trans: bool
    # SF plane per-issue mapping; None for the bf16 (unquantized) paths.
    sf_store_out_mapping: Optional[FunctionMapping] = None


# Device only object
class SwapABFc2Epilogue(_ImmutableAfterInit):
    def __init__(
        self,
        base: SwapABGatedActEpilogue,
        tidx: cutlass.Int32,
        smem_tensor: Optional[cute.Tensor],
        tma_atom_fc2_output: Optional[cute.CopyAtom],
        fc2_tma_output: Optional[cute.Tensor],
        fc2_output: cute.Tensor,  # MoE domain (token, topk, hidden)
        token_src_metadata: Optional[cute.Tensor],
        fc2_done_counter: Optional[cute.Tensor],
        fc2_output_sf: Optional[cute.Tensor],
        peer_rank_ptr_mapper: Optional[SymmetricBufferDevice],
        optional_epi_args: GatedActEpilogueArgs,
        epi_wg: Optional[cutlass.Int32] = None,
        scratch_pointer: Optional[cute.Pointer] = None,
    ):
        self.base = base
        self.epi_wg = epi_wg
        self.tidx = tidx % (base._EpilogueWarpCnt * 32)
        self.warp_idx = self.tidx // 32
        self.lane_idx = self.tidx % 32
        # v14-epi3 (MEGAMOE_LEAN_FC2_STSM): epilogue scratch base (the fc1 staging overlay); None when off.
        # The lane's STSM row address and LDS.128 base are loop invariants, computed once here.
        self.scratch_pointer = scratch_pointer
        if cutlass.const_expr(scratch_pointer is not None):
            self.stsm_addrs = self._fc2_stsm_addrs()
        else:
            self.stsm_addrs = None
        self.tma_atom_fc2_output = tma_atom_fc2_output
        self.fc2_tma_output = fc2_tma_output
        self.fc2_output = fc2_output
        self.token_src_metadata = token_src_metadata
        self.fc2_done_counter = fc2_done_counter
        self.fc2_output_sf = fc2_output_sf
        self.peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self.optional_epi_args = optional_epi_args
        if cutlass.const_expr(base.fc2_use_tma):
            self.smem_tensor = smem_tensor
            self.process_pipeline = make_fc2_tma_process_pipeline(
                combine_format=base.combine_format,
                cta_token_tile_size=base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        elif cutlass.const_expr(base.fc2_use_ublk):
            self.smem_tensor = smem_tensor
            self.process_pipeline = make_fc2_ublk_process_pipeline(
                combine_format=base.combine_format,
                # N = 224 / 240: map the 64-token subtile grid of a 256-token tile; tokens past the tile's
                # valid count (always <= cta_tile_n) are predicated off by Fc2OutputRouter.
                cta_token_tile_size=-(-base.cta_tile_n // 64) * 64 if base.n224_1wg else base.cta_tile_n,
                cta_hidden_tile_size=base.cta_tile_m,
            )
        else:
            self.smem_tensor = None
            if cutlass.const_expr(base.reduce_topk_in_epilogue):
                self.process_pipeline = make_fc2_redg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
            else:
                self.process_pipeline = make_fc2_stg_process_pipeline(
                    combine_format=base.combine_format,
                    cta_token_tile_size=base.cta_tile_n,
                    cta_hidden_tile_size=base.cta_tile_m,
                )
        self._freeze()

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "base"), name)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        # See SwapABFc1Epilogue.__extract_mlir_values__: this helper carries
        # only loop-invariant Python context.  It intentionally serializes no
        # MLIR values, so changing it to store loop-carried state would be a
        # correctness bug.
        return []

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "SwapABFc2Epilogue":
        assert len(values) == 0
        return self

    @cute.jit
    def signal_fc2_done(self, work_tile_info, next_work_tile_info, flag_tracker):
        publish: cutlass.Constexpr = self.token_back_enabled
        flag_address = Int64(0)
        if cutlass.const_expr(publish and self.fc2_block_done):
            # v16-fc2blk: per 256-token block (= the FC1-done slot of this tile), see TokenComm.fc2_block_done.
            flag_address = (
                self.fc2_done_counter.iterator
                + work_tile_info.cumulative_token_block_count
                + work_tile_info.tile_n_idx
            ).toint()
        elif cutlass.const_expr(publish and self.fc2_tail_blocks is not None):
            # v26-tbtail: the last K local experts signal their 256-token block's counter (after the E per-expert
            # counters); the others their expert's counter.
            flag_address = (self.fc2_done_counter.iterator + work_tile_info.expert_idx).toint()
            if work_tile_info.expert_idx >= cutlass.Int32(
                self.fc2_tail_blocks[1] - self.fc2_tail_blocks[0]
            ):
                flag_address = (
                    self.fc2_done_counter.iterator
                    + cutlass.Int32(self.fc2_tail_blocks[1])
                    + work_tile_info.cumulative_token_block_count
                    + work_tile_info.tile_n_idx
                ).toint()
        elif cutlass.const_expr(publish):
            flag_address = (self.fc2_done_counter.iterator + work_tile_info.expert_idx).toint()
        no_fire: cutlass.Constexpr = not publish
        return flag_tracker.accumulate(
            next_work_tile_info.phase, self.fc2_epi_flag_batch, flag_address, no_fire
        )

    @cute.jit
    def _make_output_router(self, work_tile_info: SwapAbFc12WorkTileInfo) -> "Fc2OutputRouter":
        task_tile_data_row_start = (
            work_tile_info.cumulative_data_physical_row
            + work_tile_info.tile_n_idx * cutlass.Int32(self.cta_tile_n)
        )
        hidden_base_this_cta_tile = work_tile_info.tile_m_idx * cutlass.Int32(self.cta_tile_m)
        valid_hidden_this_cta_tile = (
            cutlass.Int32(self.fc2_output.shape[2]) - hidden_base_this_cta_tile
        )
        if valid_hidden_this_cta_tile < 0:
            valid_hidden_this_cta_tile = 0
        if valid_hidden_this_cta_tile > self._EpilogueFc2HiddenTileSize:
            valid_hidden_this_cta_tile = self._EpilogueFc2HiddenTileSize

        metadata = None
        peer_rank_ptr_mapper = None
        data_token_base = task_tile_data_row_start
        if cutlass.const_expr(
            self.token_src_metadata is not None and not self.token_back_push_data
        ):
            metadata = cute.domain_offset((task_tile_data_row_start,), self.token_src_metadata)
            peer_rank_ptr_mapper = self.peer_rank_ptr_mapper
            data_token_base = None

        if cutlass.const_expr(self.combine_format.is_quantized):
            base_outputs = (self.fc2_output, self.fc2_output_sf)
            token_bases = (data_token_base, task_tile_data_row_start)
            output_mappings = (
                self.process_pipeline.store_out_mapping,
                self.process_pipeline.sf_store_out_mapping,
            )
        else:
            base_outputs = self.fc2_output
            token_bases = data_token_base
            output_mappings = self.process_pipeline.store_out_mapping

        # v14-epi3 (MEGAMOE_LEAN_FC2_ADDR): local STG of fp4 / fp8 wire data -> per-tile address bases.
        lean_addr: cutlass.Constexpr[bool] = (
            self.lean_fc2_addr
            and metadata is None
            and self.combine_format.is_quantized
            and self.combine_format.act_dtype.width <= 8
            and self.process_pipeline.store_function is fc2_stg_store_function
        )
        router = Fc2OutputRouter(
            metadata=metadata,
            token_bases=token_bases,
            base_outputs=base_outputs,
            hidden_base_this_cta_tile=hidden_base_this_cta_tile,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            valid_tokens_this_cta_tile=work_tile_info.valid_tokens_in_cta_tile,
            valid_hidden_this_cta_tile=valid_hidden_this_cta_tile,
            reduce_topk_in_epilogue=self.reduce_topk_in_epilogue,
            output_mappings=output_mappings,
            epi_tid=self.tidx,
            combine_format=self.combine_format,
        ).prefetch()
        if cutlass.const_expr(lean_addr):
            router = router.lean_prefetch()
        return router

    @cute.jit
    def __call__(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ):
        # subtile-irrelevant hoist: fc2 alpha scales raw fc2 accumulators before f2fp.
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        # (cta_tile_m, cta_tile_n) -> (epi_tile_m, epi_tile_n, iters)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize, self._EpilogueTokenTileSize)
        )[None, None, 0, None]

        iket.range_push("epi.acc_wait")
        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        iket.range_pop()
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        if cutlass.const_expr(self.fc2_ldtm_prefetch):
            self._run_subtiles_prefetched(
                work_tile_info=work_tile_info,
                tmem_acc_tensor_tiled_by_epi_tile=tmem_acc_tensor_tiled_by_epi_tile,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
                is_odd_turn=is_odd_turn,
                fc2_output_router=fc2_output_router,
                alpha_val=alpha_val,
                valid_tokens=valid_tokens,
            )
        else:
            # Overlap path preloads two subtiles before releasing acc TMEM.
            unroll_tile_cnt = (
                2
                if cutlass.const_expr(
                    self.overlapping_accum and self.process_pipeline.require_tmem_trans
                )
                else 0
            )
            remain_subtile_cnt = self.subtile_cnt - unroll_tile_cnt

            if cutlass.const_expr(unroll_tile_cnt > 0):
                subtile_idx_first = (cutlass.Int32(self.subtile_cnt) - is_odd_turn) % cutlass.Int32(
                    self.subtile_cnt
                )
                subtile_idx_second = (
                    cutlass.Int32(self.subtile_cnt + 1) - is_odd_turn
                ) % cutlass.Int32(self.subtile_cnt)

                # Each warp preloads its local 32-hidden x 64-token accumulator as two
                # (32,) FP32 tensors:
                #
                #   preload[0]: hidden  0..15 x token 0..63
                #   preload[1]: hidden 16..31 x token 0..63
                #
                # One LDTM.16dp256bit.x8 produces each tensor. Within it, every
                # consecutive register pair holds adjacent token columns at one
                # hidden coordinate. After F2FP, that pair is one BF16x2 MOVM input.
                preload_subtile_first: Tuple[cute.Tensor, ...] = self.process_pipeline.tmem_acc_load(
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                        None, None, subtile_idx_first
                    ],
                    epi=self,
                )

                # Release acc to next MMA unconditionally.
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

                # Same two-half mapping for the other token subtile. Both preloaded
                # subtiles can now use this second subtile's TMEM as transpose workspace.
                preload_subtile_second: Tuple[cute.Tensor, ...] = self.process_pipeline.tmem_acc_load(
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                        None, None, subtile_idx_second
                    ],
                    epi=self,
                )

                # Both unrolled subtiles borrow tmem_subtile_second as workspace.
                preload_pair = (preload_subtile_first, preload_subtile_second)
                subtile_idx_pair = (subtile_idx_first, subtile_idx_second)
                for i in cutlass.range_constexpr(unroll_tile_cnt):
                    if subtile_idx_pair[i] * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                        self.run_subtile(
                            work_tile_info=work_tile_info,
                            epilogue_iter_idx=cutlass.Int32(i),
                            subtile_idx=subtile_idx_pair[i],
                            tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                                None, None, subtile_idx_second
                            ],
                            preload_acc=preload_pair[i],
                            fc2_output_router=fc2_output_router,
                            alpha_val=alpha_val,
                            release_after_ldtm=False,
                            acc_pipeline=acc_pipeline,
                            acc_consumer_state=acc_consumer_state,
                        )
                    else:
                        if cutlass.const_expr(self.fc2_skip_tile_drain):
                            cute.arch.cp_async_bulk_commit_group()

            if cutlass.const_expr(self.overlapping_accum and unroll_tile_cnt == 0):
                release_after_ldtm = True
            else:
                release_after_ldtm = False
            for i in cutlass.range(remain_subtile_cnt, unroll=1):
                # for i in cutlass.range_constexpr(remain_subtile_cnt):
                real_i = i + unroll_tile_cnt
                if cutlass.const_expr(self.overlapping_accum):
                    subtile_idx = (
                        cutlass.Int32(real_i + self.subtile_cnt) - is_odd_turn
                    ) % cutlass.Int32(self.subtile_cnt)
                else:
                    subtile_idx = cutlass.Int32(real_i)

                if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                    self.run_subtile(
                        work_tile_info=work_tile_info,
                        epilogue_iter_idx=real_i,
                        subtile_idx=subtile_idx,
                        tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                        preload_acc=None,
                        fc2_output_router=fc2_output_router,
                        alpha_val=alpha_val,
                        release_after_ldtm=release_after_ldtm,
                        acc_pipeline=acc_pipeline,
                        acc_consumer_state=acc_consumer_state,
                    )
                    release_after_ldtm = False
                else:
                    if cutlass.const_expr(self.fc2_skip_tile_drain):
                        # Keep one bulk group per FC2 iteration so the staging ring's read wait stays
                        # exact across tiles once the tile-boundary drain is skipped.
                        cute.arch.cp_async_bulk_commit_group()

            # Non-overlap-path release: at the natural task-tile boundary.
            if cutlass.const_expr(not self.overlapping_accum):
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def _run_subtiles_prefetched(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor_tiled_by_epi_tile: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
        valid_tokens: cutlass.Int32,
    ):
        """v5-epi FC2 (UBLK BF16, no TMEM transpose) subtile loop with a one-deep TMEM prefetch.

        Same subtile order, staging stages and stores as the default loop. Differences: the loop is
        fully unrolled, and subtile i+1's LDTM.x64 is issued before subtile i's staging wait / STS /
        bulk issue, so its latency hides behind them. TMEM of out-of-range subtiles is loaded but
        never used. With the overlapping accumulator, the accumulator is released right after the
        first (overlapping) subtile is in registers -- the default path releases after the first
        *valid* subtile's load, which is never earlier.
        """
        subtile_cnt: cutlass.Constexpr[int] = self.subtile_cnt

        def subtile_at(real_i: int):
            if cutlass.const_expr(self.overlapping_accum):
                return (cutlass.Int32(real_i + subtile_cnt) - is_odd_turn) % cutlass.Int32(
                    subtile_cnt
                )
            return cutlass.Int32(real_i)

        next_subtile_idx = subtile_at(0)
        next_loaded = self.process_pipeline.tmem_acc_load(
            tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, next_subtile_idx],
            epi=self,
        )
        if cutlass.const_expr(self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

        for real_i in cutlass.range_constexpr(subtile_cnt):
            subtile_idx = next_subtile_idx
            loaded = next_loaded
            if cutlass.const_expr(real_i + 1 < subtile_cnt):
                next_subtile_idx = subtile_at(real_i + 1)
                next_loaded = self.process_pipeline.tmem_acc_load(
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[
                        None, None, next_subtile_idx
                    ],
                    epi=self,
                )
            if subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
                self.run_subtile(
                    work_tile_info=work_tile_info,
                    epilogue_iter_idx=cutlass.Int32(real_i),
                    subtile_idx=subtile_idx,
                    tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx],
                    preload_acc=loaded,
                    fc2_output_router=fc2_output_router,
                    alpha_val=alpha_val,
                    release_after_ldtm=False,
                    acc_pipeline=acc_pipeline,
                    acc_consumer_state=acc_consumer_state,
                )
            else:
                if cutlass.const_expr(self.fc2_skip_tile_drain):
                    cute.arch.cp_async_bulk_commit_group()

        if cutlass.const_expr(not self.overlapping_accum):
            cute.arch.fence_view_async_tmem_load()
            acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def call_wg2(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        is_odd_turn: cutlass.Int32,
    ):
        # v10-epi2 FC2 split; same TMEM / barrier protocol as SwapABFc1Epilogue.call_wg2. The STG
        # reorder (TmemTranspose32x64B16Movm) also only uses cols [0, 32) of its workspace view.
        epi_wg = self.epi_wg
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        tmem_acc_tensor_tiled_by_epi_tile = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize, self._EpilogueTokenTileSize)
        )[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile

        subtile_cnt = cutlass.Int32(self.subtile_cnt)
        subtile_idx_first = (subtile_cnt - is_odd_turn) % subtile_cnt
        subtile_idx_second = (subtile_cnt + 1 - is_odd_turn) % subtile_cnt
        head_subtile_idx = (subtile_idx_first + epi_wg) % subtile_cnt
        loop_subtile_idx = (subtile_idx_first + 2 + epi_wg) % subtile_cnt

        preload_head = self.process_pipeline.tmem_acc_load(
            tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, head_subtile_idx],
            epi=self,
        )
        cute.arch.fence_view_async_tmem_load()
        if cutlass.const_expr(not self.fc2_stsm):
            # TMEM-transpose path: WG0 will scribble on second[32, 64) -> wait for WG1's load.
            # (v14 STSM transpose touches no TMEM after the loads: no handoff needed.)
            _epi_wg2_handoff(epi_wg)
        acc_pipeline.consumer_release(acc_consumer_state)

        workspace = cute.flat_divide(
            tmem_acc_tensor_tiled_by_epi_tile[None, None, subtile_idx_second],
            (self._EpilogueFc2HiddenTileSize, 32),
        )[None, None, 0, cutlass.Int32(1) - epi_wg]
        if head_subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
            self.run_subtile(
                work_tile_info=work_tile_info,
                epilogue_iter_idx=epi_wg,
                subtile_idx=head_subtile_idx,
                tmem_subtile_tensor=workspace,
                preload_acc=preload_head,
                fc2_output_router=fc2_output_router,
                alpha_val=alpha_val,
                release_after_ldtm=False,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
            )
        if loop_subtile_idx * cutlass.Int32(self._EpilogueTokenTileSize) < valid_tokens:
            self.run_subtile(
                work_tile_info=work_tile_info,
                epilogue_iter_idx=epi_wg + cutlass.Int32(2),
                subtile_idx=loop_subtile_idx,
                tmem_subtile_tensor=tmem_acc_tensor_tiled_by_epi_tile[None, None, loop_subtile_idx],
                preload_acc=None,
                fc2_output_router=fc2_output_router,
                alpha_val=alpha_val,
                release_after_ldtm=False,
                acc_pipeline=acc_pipeline,
                acc_consumer_state=acc_consumer_state,
            )

    # -- v14-epi3: FC2 BF16 transpose through warp-private SMEM (MEGAMOE_LEAN_FC2_STSM=1) -----------
    #
    # Input: fc2_f2fp of the two Ld16x256b.x8 loads. Word w of the top half (w < 16) is the m8n8 b16
    # fragment of matrix (g = w & 1, j = w >> 1), word 16 + w of the bottom half is (g = 2 + (w & 1),
    # j = w >> 1): lane L holds hidden 8g + L/4 at tokens 8j + 2(L%4) + {0, 1} (the MOVM input layout of
    # TmemTranspose32x64B16Movm). stmatrix .trans writes each such matrix as 8 token rows of 8 hiddens,
    # so the scratch holds [token][hidden] BF16, 64 B per token row, with the 16 B chunk g of row t at
    # chunk g ^ ((t >> 1) & 3) (conflict-free for both the STSM row phases and the LDS.128 phases).
    # Lane L then reads token L (and L + 32) as 4 x LDS.128 each: exactly the TMEM path's output
    # (token = lane + 32 * (elem // 32), hidden = elem % 32). STSM.x4 j stores matrices (0..3, j); lane
    # t addresses row t & 7 of matrix t >> 3. WG2 slots (576 words) hold one 32-token pass at a time.
    # Pure data movement: bit-identical.

    def _fc2_stsm_addrs(self):
        """(STSM row address, LDS.128 chunk-0 address) of this lane; both 32-bit shared addresses.

        The slot base is 64 B aligned (128 B staging alignment, 8 KiB fc1 staging, 2304 / 4608 B slots),
        so the chunk swizzle only touches address bits 4-5 and chunk g of the lane's row is
        ``ld0 ^ (16 * g)``."""
        lane = self.lane_idx
        if cutlass.const_expr(self.epi_wg2):
            slot = self.epi_wg * cutlass.Int32(self._EpilogueWarpCnt) + self.warp_idx
            slot_bytes = 576 * 4
        else:
            slot = self.warp_idx
            slot_bytes = 1152 * 4
        scratch_base = cutlass.Int32(
            cute.recast_ptr(self.scratch_pointer, dtype=cutlass.Int8).toint()
        ) + (cutlass.Int32(self.fc1_staging_bytes) + slot * cutlass.Int32(slot_bytes))
        # STSM row address of lane t: row t & 7 of matrix (chunk) t >> 3.
        st_addr = scratch_base + (
            (lane & cutlass.Int32(7)) * cutlass.Int32(64)
            + ((lane >> cutlass.Int32(3)) ^ ((lane >> cutlass.Int32(1)) & cutlass.Int32(3)))
            * cutlass.Int32(16)
        )
        # LDS.128 of chunk 0 of token row `lane`: chunk g ^ ((lane >> 1) & 3) in memory.
        ld0 = scratch_base + (
            lane * cutlass.Int32(64) + ((lane >> cutlass.Int32(1)) & cutlass.Int32(3)) * cutlass.Int32(16)
        )
        return st_addr, ld0

    @cute.jit
    def _fc2_stsm_transpose(self, casted: cute.Tensor) -> cute.Tensor:
        if cutlass.const_expr(cute.size(casted) != 64 or casted.element_type is not cutlass.BFloat16):
            raise ValueError("FC2 STSM transpose expects the 64 BF16 f2fp registers.")
        words = cute.recast_tensor(casted, cutlass.Int32)
        passes: cutlass.Constexpr[int] = 2 if self.epi_wg2 else 1
        st_addr, ld0 = self.stsm_addrs
        ld_addr = [ld0] + [ld0 ^ cutlass.Int32(16 * g) for g in range(1, 4)]
        out_words = cute.make_rmem_tensor((32,), cutlass.Int32)
        groups_per_pass: cutlass.Constexpr[int] = 8 // passes
        for p in cutlass.range_constexpr(passes):
            # Other lanes' loads of the previous subtile / pass (or FC1's) must finish first.
            cute.arch.sync_warp()
            for jj in cutlass.range_constexpr(groups_per_pass):
                j = p * groups_per_pass + jj
                _stsm_x4_trans_b16(
                    st_addr,
                    jj * 512,
                    words[2 * j],
                    words[2 * j + 1],
                    words[16 + 2 * j],
                    words[16 + 2 * j + 1],
                )
            cute.arch.sync_warp()
            for half in cutlass.range_constexpr(2 // passes):
                token_half = p * (2 // passes) + half  # 0: tokens 0..31, 1: tokens 32..63
                for g in cutlass.range_constexpr(4):
                    v = _lds128_b32_off(ld_addr[g], half * 2048)
                    for k in cutlass.range_constexpr(4):
                        out_words[16 * token_half + 4 * g + k] = v[k]
        return cute.recast_tensor(out_words, cutlass.BFloat16)

    # -- v30-n224: FC2 in 32-token units (see SwapABFc1Epilogue.call_n224 for the ownership) ------
    #
    # Per unit: two Ld16x256b.x4 (warp hidden rows 0..15 / 16..31 x 32 tokens), f2fp, one 32-token pass
    # of the STSM transpose (= the WG2 pass), the combine quant of one token per lane, then the STG of
    # data / scale at router iter = unit (32 tokens per iter). Pure regrouping of the 64-token subtile
    # path: bit-identical.

    @cute.jit
    def _n224_ldtm_unit(self, tmem_unit: cute.Tensor):
        atom_ld16x256 = cute.make_copy_atom(
            tcgen05.Ld16x256bOp(tcgen05.Repetition.x4), cutlass.Float32
        )
        ptr = tmem_unit.iterator
        half_lane_offset = 16 * TmemTranspose32x64B16Movm._tmem_row_stride
        top = cute.make_rmem_tensor((16,), cutlass.Float32)
        bottom = cute.make_rmem_tensor((16,), cutlass.Float32)
        cute.copy(
            atom_ld16x256,
            cute.make_tensor(ptr, TmemTranspose32x64B16Movm._tmem_layout(16, 32)),
            TmemTranspose32x64B16Movm._rmem_copy_view(top, 16),
        )
        cute.copy(
            atom_ld16x256,
            cute.make_tensor(ptr + half_lane_offset, TmemTranspose32x64B16Movm._tmem_layout(16, 32)),
            TmemTranspose32x64B16Movm._rmem_copy_view(bottom, 16),
        )
        return top, bottom

    @cute.jit
    def _n224_stsm_transpose(self, casted: cute.Tensor) -> cute.Tensor:
        """One 32-token pass of _fc2_stsm_transpose: top words 0..7 / bottom words 8..15 (j = w >> 1)."""
        if cutlass.const_expr(cute.size(casted) != 32 or casted.element_type is not cutlass.BFloat16):
            raise ValueError("FC2 unit STSM transpose expects the 32 BF16 f2fp registers.")
        words = cute.recast_tensor(casted, cutlass.Int32)
        st_addr, ld0 = self.stsm_addrs
        ld_addr = [ld0] + [ld0 ^ cutlass.Int32(16 * g) for g in range(1, 4)]
        out_words = cute.make_rmem_tensor((16,), cutlass.Int32)
        cute.arch.sync_warp()
        for j in cutlass.range_constexpr(4):
            _stsm_x4_trans_b16(
                st_addr, j * 512, words[2 * j], words[2 * j + 1], words[8 + 2 * j], words[8 + 2 * j + 1]
            )
        cute.arch.sync_warp()
        for g in cutlass.range_constexpr(4):
            v = _lds128_b32_off(ld_addr[g], 0)
            for k in cutlass.range_constexpr(4):
                out_words[4 * g + k] = v[k]
        return cute.recast_tensor(out_words, cutlass.BFloat16)

    @cute.jit
    def _n224_unit(
        self,
        loaded,
        unit: cutlass.Int32,
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
    ):
        fine = self.opt_iket_fine
        _fine_push(fine, "fc2.ldtm_cvt")
        casted = fc2_f2fp(*loaded, alpha_val=alpha_val)
        pre_store = self._n224_stsm_transpose(casted)
        _fine_pop(fine)
        if cutlass.const_expr(self.combine_format.is_quantized):
            if cutlass.const_expr(self.lean_fc2_quant):
                data_unit, sf_regs = QuantImpl(
                    self.combine_format, "regs_in_thread", lean_combine=True
                )(pre_store)
            else:
                data_unit, sf_regs = QuantImpl(self.combine_format, "regs_in_thread")(pre_store)
        else:
            data_unit = pre_store
            sf_regs = None
        stg_width_elems: cutlass.Constexpr[int] = min(32, 256 // data_unit.element_type.width)
        stg_bits: cutlass.Constexpr[int] = stg_width_elems * data_unit.element_type.width
        copy_atom_vec = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=stg_bits
        )
        elems_per_thread: cutlass.Constexpr[int] = cute.size(data_unit)
        if cutlass.const_expr(sf_regs is not None):
            sf_scales_per_stg: cutlass.Constexpr[int] = 32 // self.combine_format.scale_block
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(unit)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[scale_idx]
        iters_per_unit: cutlass.Constexpr[int] = elems_per_thread // stg_width_elems
        copy_src = cute.zipped_divide(data_unit, (stg_width_elems,))
        single_copy_layout = cute.make_layout(((stg_width_elems, 1),), stride=((1, 0),))
        unit_iter_base = unit * cutlass.Int32(iters_per_unit)
        for local_iter in cutlass.range_constexpr(iters_per_unit):
            dst_ptr, pred = fc2_output_router.get_data_dst(unit_iter_base + cutlass.Int32(local_iter))
            if pred != cutlass.Int32(0):
                src_i = cute.make_tensor(copy_src[None, local_iter].iterator, single_copy_layout)
                dst_i = cute.make_tensor(dst_ptr, single_copy_layout)
                cute.copy(
                    copy_atom_vec,
                    cute.recast_tensor(src_i, cutlass.Int32),
                    cute.recast_tensor(dst_i, cutlass.Int32),
                )

    @cute.jit
    def call_n224(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        tmem_acc_tensor: cute.Tensor,
        acc_pipeline,
        acc_consumer_state,
        tile_parity: cutlass.Int32,
    ):
        epi_wg = self.epi_wg
        if cutlass.const_expr(self.optional_epi_args.fc2_alpha is not None):
            alpha_val = self.optional_epi_args.fc2_alpha[work_tile_info.expert_idx]
        else:
            alpha_val = None
        acc_ready = False
        if not work_tile_info.peek_ready:
            acc_ready = True
            acc_pipeline.consumer_wait(acc_consumer_state)
        fc2_output_router = self._make_output_router(work_tile_info)
        tmem_units = cute.flat_divide(
            tmem_acc_tensor, (self._EpilogueFc2HiddenTileSize, self._EpilogueUnitTokens)
        )[None, None, 0, None]

        acc_pipeline.consumer_wait(acc_consumer_state, acc_ready)
        valid_tokens = work_tile_info.valid_tokens_in_cta_tile
        unit_tokens = cutlass.Int32(self._EpilogueUnitTokens)

        first = (tile_parity + epi_wg) & cutlass.Int32(1)
        pair_cnt = cutlass.Int32(2) - first
        if cutlass.const_expr(self.unit_cnt == 8):
            pair_cnt = cutlass.Int32(2)  # N = 240: two pairs per WG, no single unit
        for k in cutlass.range(pair_cnt, unroll=1):
            unit_a = first + cutlass.Int32(4) * k
            unit_b = unit_a + cutlass.Int32(2)
            if unit_a * unit_tokens < valid_tokens:
                loaded_a = self._n224_ldtm_unit(tmem_units[None, None, unit_a])
                loaded_b = self._n224_ldtm_unit(tmem_units[None, None, unit_b])
                if k == cutlass.Int32(1):
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                self._n224_unit(loaded_a, unit_a, fc2_output_router, alpha_val)
                if unit_b * unit_tokens < valid_tokens:
                    self._n224_unit(loaded_b, unit_b, fc2_output_router, alpha_val)
            else:
                if k == cutlass.Int32(1):
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
        if cutlass.const_expr(self.unit_cnt % 2 == 1):
            if first == cutlass.Int32(1):
                unit_c = cutlass.Int32(self.unit_cnt - 2)
                if unit_c * unit_tokens < valid_tokens:
                    loaded_c = self._n224_ldtm_unit(tmem_units[None, None, unit_c])
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)
                    self._n224_unit(loaded_c, unit_c, fc2_output_router, alpha_val)
                else:
                    cute.arch.fence_view_async_tmem_load()
                    acc_pipeline.consumer_release(acc_consumer_state)

    @cute.jit
    def run_subtile(
        self,
        work_tile_info: SwapAbFc12WorkTileInfo,
        epilogue_iter_idx: cutlass.Int32,
        subtile_idx: cutlass.Int32,
        # (hidden_tile, token_subtile), fundamentally (epi_tile_m, epi_tile_n)
        tmem_subtile_tensor: cute.Tensor,
        preload_acc: Optional[Tuple[cute.Tensor, ...]],
        fc2_output_router: "Fc2OutputRouter",
        alpha_val: Optional[cutlass.Float32],
        release_after_ldtm: Union[cutlass.Boolean, bool],
        acc_pipeline,
        acc_consumer_state,
    ):
        process_pipeline = self.process_pipeline
        fine = self.opt_iket_fine
        _fine_push(fine, "fc2.ldtm_cvt")
        if cutlass.const_expr(preload_acc is None):
            loaded = process_pipeline.tmem_acc_load(
                tmem_subtile_tensor=tmem_subtile_tensor, epi=self
            )
            if release_after_ldtm:
                cute.arch.fence_view_async_tmem_load()
                acc_pipeline.consumer_release(acc_consumer_state)
        else:
            loaded = preload_acc

        casted = process_pipeline.f2fp(*loaded, alpha_val=alpha_val)
        # reorder returns a bare RMEM fragment in the store's expected pre-store
        # distribution; reorder + store are paired 1:1 inside the pipeline.
        if cutlass.const_expr(
            self.fc2_stsm and process_pipeline.post_f2fp_reorder is fc2_stg_post_f2fp_reorder
        ):
            pre_store = self._fc2_stsm_transpose(casted)
        else:
            pre_store = process_pipeline.post_f2fp_reorder(
                casted=casted, tmem_subtile_view=tmem_subtile_tensor
            )
        _fine_pop(fine)
        process_pipeline.store_function(
            epi=self,
            subtile=pre_store,
            work_tile_info=work_tile_info,
            epilogue_iter_idx=epilogue_iter_idx,
            subtile_idx=subtile_idx,
            fc2_output_router=fc2_output_router,
        )


@dataclasses.dataclass(frozen=True)
class Fc2OutputRouter:
    # One packed i64 TokenSrcMetadata record per pool token. None means a local write.
    metadata: Optional[cute.Tensor]
    # token + possible sf
    token_bases: Union[Tuple[Optional[cutlass.Int32], cutlass.Int32], Optional[cutlass.Int32]]
    base_outputs: Union[Tuple[cute.Tensor, cute.Tensor], cute.Tensor]  # (token, topk, hidden)
    hidden_base_this_cta_tile: Union[cutlass.Int32, int]
    peer_rank_ptr_mapper: Optional[SymmetricBufferDevice]
    valid_tokens_this_cta_tile: cutlass.Int32
    valid_hidden_this_cta_tile: Union[cutlass.Int32, int]
    reduce_topk_in_epilogue: bool
    # Per-issue (epi_tid, iter_idx) -> (token_cta_tile, hidden_cta_tile). Data
    # mapping, or (data mapping, sf mapping) when quantized.
    output_mappings: Union[Tuple[FunctionMapping, FunctionMapping], FunctionMapping]
    epi_tid: cutlass.Int32
    combine_format: CombineFormat
    # After metadata prefetch
    dst_ptrs: Optional[cute.Tensor] = (
        None  # i64 x (copy_iters_this_thread_cta_tile), fundamentally the pointers.
    )
    valid: Optional[cute.Tensor] = None  # (copy_iters_this_thread_cta_tile)
    # v14-epi3 (MEGAMOE_LEAN_FC2_ADDR, local STG with one STG per 32 hidden): per-tile address bases and
    # per-iter byte steps, and the remaining valid tokens of this lane (0 when its hidden is out of range).
    lean_data_base: Optional[Int64] = None
    lean_data_step: Optional[Int64] = None
    lean_sf_base: Optional[Int64] = None
    lean_sf_step: Optional[Int64] = None
    lean_rem_tokens: Optional[cutlass.Int32] = None

    @property
    def data_output(self) -> cute.Tensor:
        return self.base_outputs[0] if isinstance(self.base_outputs, tuple) else self.base_outputs

    @property
    def sf_output(self) -> Optional[cute.Tensor]:
        # Present iff quantized; (pool_token, 1, hidden // sf_vec) rank-local.
        return self.base_outputs[1] if isinstance(self.base_outputs, tuple) else None

    @property
    def data_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[0] if isinstance(self.token_bases, tuple) else self.token_bases

    @property
    def sf_token_base(self) -> Optional[cutlass.Int32]:
        return self.token_bases[1] if isinstance(self.token_bases, tuple) else None

    @property
    def data_mapping(self) -> FunctionMapping:
        return (
            self.output_mappings[0]
            if isinstance(self.output_mappings, tuple)
            else self.output_mappings
        )

    @property
    def sf_mapping(self) -> Optional[FunctionMapping]:
        return self.output_mappings[1] if isinstance(self.output_mappings, tuple) else None

    def __post_init__(self) -> None:
        if (self.metadata is None) == (self.data_token_base is None):
            raise ValueError(
                "Fc2OutputRouter requires exactly one of metadata or a (data) token base."
            )
        if (self.metadata is None) != (self.peer_rank_ptr_mapper is None):
            raise ValueError("Fc2OutputRouter requires peer_rank_ptr_mapper iff metadata is set.")
        if self.reduce_topk_in_epilogue and self.metadata is None:
            raise ValueError("Fc2OutputRouter reduction requires metadata routing.")

    @cute.jit
    def prefetch(self) -> "Fc2OutputRouter":
        # Only the metadata (comm) path prefetches a pointer array: its
        # metadata-derived address has long-latency LDGs worth issuing early.
        # The local (no-comm) path computes its affine address on demand in
        # get_dst() -- no array, hence no runtime-indexed local-memory spill.
        if cutlass.const_expr(self.metadata is None):
            return self
        copy_iters: cutlass.Constexpr[int] = self.data_mapping.domain.axis_size("iter_idx")

        valid = cute.make_rmem_tensor((copy_iters,), cutlass.Int32)
        dst_ptrs = cute.make_rmem_tensor((copy_iters,), cutlass.Int64)

        # Compiler should be able to optimize the same token_copy_group's offset add. (Fundamental cse +
        # strength_reduce)
        # We should check the SASS to ensure this happens.
        for iter_idx in cutlass.range_constexpr(copy_iters):
            coord = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])

            valid[iter_idx] = cutlass.Int32(0)
            dst_ptrs[iter_idx] = cutlass.Int64(0)

            token_valid = token_in_tile < self.valid_tokens_this_cta_tile
            hidden_valid = hidden_in_tile < cutlass.Int32(self.valid_hidden_this_cta_tile)
            if token_valid and hidden_valid:
                valid[iter_idx] = cutlass.Int32(1)
                if cutlass.const_expr(self.metadata is None):
                    dst_tokens = self.data_token_base + token_in_tile
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    # Int64 token coord: dst_tokens*K*H overflows int32 once
                    # T*K*H exceeds 2^31 (data_output is (token, topk, hidden)).
                    dst_ptrs[iter_idx] = self.data_output[
                        Int64(dst_tokens), None, dst_hidden
                    ].iterator.toint()

                else:
                    md = TokenSrcMetadata.load(
                        self.metadata.iterator.toint()
                        + Int64(token_in_tile) * Int64(TokenSrcMetadata.nbytes)
                    )
                    dst_rank = md.src_rank
                    dst_token = md.src_token
                    dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                    if cutlass.const_expr(not self.reduce_topk_in_epilogue):
                        dst_topk = md.src_topk
                    else:
                        dst_topk = 0
                    # Int64 token coord: domain_offset on (token, topk, hidden)
                    # computes dst_token*K*H, which overflows int32 once T*K*H > 2^31.
                    dst_ptrs[iter_idx] = self.peer_rank_ptr_mapper.map_pointer(
                        cute.domain_offset(
                            (Int64(dst_token), dst_topk, dst_hidden), self.data_output
                        ).iterator,
                        dst_rank,
                        byte_alignment=32,
                    ).toint()

        return dataclasses.replace(self, dst_ptrs=dst_ptrs, valid=valid)

    @cute.jit
    def lean_prefetch(self) -> "Fc2OutputRouter":
        """v14-epi3: hoist the local (no-metadata) STG addressing to once per tile.

        Valid only when every data issue covers 32 hidden of one token (token = lane + 32 * iter,
        hidden = 32 * warp, as for fp4 / fp8 wire data) and the scale plane uses the same (token,
        hidden) per iter. Then address(iter) = address(0) + iter * (address(token + 32) - address(token))
        exactly: the element offsets are non-negative and even (fp4 hidden bases are multiples of 32,
        rows are whole bytes), so the 4-bit byte rounding is affine. The predicate is the original
        ``token < valid_tokens and hidden < valid_hidden``."""
        coord0 = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=0)
        token0 = cutlass.Int32(coord0["token_in_cta_tile"])
        hidden0 = cutlass.Int32(coord0["hidden_in_cta_tile"])
        dst_hidden = hidden0 + self.hidden_base_this_cta_tile
        data_t0 = self.data_output[Int64(self.data_token_base + token0), None, dst_hidden].iterator.toint()
        data_t32 = self.data_output[
            Int64(self.data_token_base + token0 + cutlass.Int32(32)), None, dst_hidden
        ].iterator.toint()
        sf_t0 = self.sf_output[Int64(self.sf_token_base + token0), None, dst_hidden].iterator.toint()
        sf_t32 = self.sf_output[
            Int64(self.sf_token_base + token0 + cutlass.Int32(32)), None, dst_hidden
        ].iterator.toint()
        rem = self.valid_tokens_this_cta_tile - token0
        if hidden0 >= cutlass.Int32(self.valid_hidden_this_cta_tile):
            rem = cutlass.Int32(0)
        return dataclasses.replace(
            self,
            lean_data_base=Int64(data_t0),
            lean_data_step=Int64(data_t32) - Int64(data_t0),
            lean_sf_base=Int64(sf_t0),
            lean_sf_step=Int64(sf_t32) - Int64(sf_t0),
            lean_rem_tokens=rem,
        )

    @cute.jit
    def get_data_dst(
        self, iter_idx: Union[int, cutlass.Int32]
    ) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue DATA destination: gmem pointer + validity predicate.

        The router owns ``data_output`` so the caller never re-assembles a
        pointer from a raw int; it just builds its own copy tensor (STG) or
        feeds the pointer to inline asm (REDG/UBLK).

        Alignment is unified at 32 B: only STG feeds this pointer to a real
        ``cute.copy`` (256 b vector store, genuinely 32 B aligned); REDG/UBLK
        only ``ptrtoint`` it for inline-asm issue, where the hint is inert.
        """
        if cutlass.const_expr(self.lean_data_base is not None):
            pred = cutlass.Int32(0)
            if cutlass.Int32(iter_idx) * cutlass.Int32(32) < self.lean_rem_tokens:
                pred = cutlass.Int32(1)
            addr = self.lean_data_base + Int64(iter_idx) * self.lean_data_step
        elif cutlass.const_expr(self.metadata is None):
            # no-comm: on-demand affine address (no prefetched array). The
            # invariant base hoists out of the caller's loop via CSE; a
            # constexpr iter folds the per-issue offset into the store.
            coord = self.data_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            pred = cutlass.Int32(0)
            addr = cutlass.Int64(0)
            if token_in_tile < self.valid_tokens_this_cta_tile and hidden_in_tile < cutlass.Int32(
                self.valid_hidden_this_cta_tile
            ):
                pred = cutlass.Int32(1)
                dst_tokens = self.data_token_base + token_in_tile
                dst_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                # Int64 token coord: dst_tokens*K*H overflows int32 once T*K*H > 2^31.
                addr = self.data_output[Int64(dst_tokens), None, dst_hidden].iterator.toint()
        else:
            # comm: read the pointer / validity prefetched by prefetch().
            addr = self.dst_ptrs[iter_idx]
            pred = self.valid[iter_idx]
        ptr = cute.make_ptr(
            self.data_output.element_type, addr, AddressSpace.gmem, assumed_align=32
        )
        return ptr, pred

    @cute.jit
    def get_sf_dst(self, iter_idx: Union[int, cutlass.Int32]) -> Tuple[cute.Pointer, cutlass.Int32]:
        """Per-issue SF destination: rank-local gmem pointer + validity predicate.

        SF never goes to a peer (it is staged locally and pushed token-contiguously
        by the dispatch / standalone warps), so this is always the affine local
        address -- no metadata routing, no prefetch. ``sf_output`` is the broadcast
        plane ``(pool_token, 1, (sf_vec, hidden//sf_vec)):(., ., (0, 1))``, so the
        logical hidden coordinate folds to its scale block on indexing.
        """
        if cutlass.const_expr(self.lean_sf_base is not None):
            pred = cutlass.Int32(0)
            if cutlass.Int32(iter_idx) * cutlass.Int32(32) < self.lean_rem_tokens:
                pred = cutlass.Int32(1)
            addr = self.lean_sf_base + Int64(iter_idx) * self.lean_sf_step
        else:
            coord = self.sf_mapping.evaluate(epi_tid=self.epi_tid, iter_idx=iter_idx)
            token_in_tile = cutlass.Int32(coord["token_in_cta_tile"])
            hidden_in_tile = cutlass.Int32(coord["hidden_in_cta_tile"])
            pred = cutlass.Int32(0)
            addr = cutlass.Int64(0)
            if token_in_tile < self.valid_tokens_this_cta_tile and hidden_in_tile < cutlass.Int32(
                self.valid_hidden_this_cta_tile
            ):
                pred = cutlass.Int32(1)
                sf_row = self.sf_token_base + token_in_tile
                sf_hidden = hidden_in_tile + self.hidden_base_this_cta_tile
                addr = self.sf_output[Int64(sf_row), None, sf_hidden].iterator.toint()
        # Per-block scale offsets are element-granular; claim the scale dtype's
        # natural element alignment (e8m0 1 B / bf16 2 B).
        sf_ptr = cute.make_ptr(
            self.sf_output.element_type, addr, AddressSpace.gmem, assumed_align=4
        )
        return sf_ptr, pred


def make_fc2_stg_cta_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0 or cta_token_tile_size in (224, 240)  # 7 x 32 / 
    # N = 240 maps 8 whole 32-token units (tokens 240..255 fall outside valid_tokens: predicated off).
    cta_token_tile_size = -(-cta_token_tile_size // 32) * 32
    wire_dtype = combine_format.act_dtype
    assert wire_dtype.width in (4, 8, 16), "fc2 STG wire dtype must be fp4/fp8/bf16."
    elems_per_stg = min(256 // wire_dtype.width, 32)
    stgs_per_hidden32 = 32 // elems_per_stg
    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": epi_tid % 32 + elem_idx // 32 * 32,
            "hidden_in_cta_tile": elem_idx % 32 + epi_tid // 32 * 32,
        },
    )
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(
            ("epi_tid", "iter_idx"), (128, stgs_per_hidden32 * cta_token_tile_size // 32)
        ),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": epi_tid % 32 + iter_idx // stgs_per_hidden32 * 32,
            "hidden_in_cta_tile": (iter_idx % stgs_per_hidden32) * elems_per_stg
            + epi_tid // 32 * 32,
        },
    )
    sf_store_out_mapping = None
    if combine_format.is_quantized:

        def stg_sf_mapping(epi_tid, iter_idx):
            lane = epi_tid % 32
            warp = epi_tid // 32
            return {"token_in_cta_tile": lane + iter_idx * 32, "hidden_in_cta_tile": warp * 32}

        sf_store_out_mapping = FunctionMapping(
            domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, cta_token_tile_size // 32)),
            codomain=CoordinateSpace(
                ("token_in_cta_tile", "hidden_in_cta_tile"),
                (cta_token_tile_size, cta_hidden_tile_size),
            ),
            function=stg_sf_mapping,
        )
    return store_out_mapping, sf_store_out_mapping, fundamental_mapping


def make_fc2_redg_cta_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    # In-kernel reduce is bf16-only and never quantized, so there is no SF plane.
    assert combine_format.act_dtype.width == 16
    assert not combine_format.is_quantized

    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": (
                ((elem_idx // 4) // 16) * 64
                + (((elem_idx // 4) % 16) // 8) * 32
                + (((elem_idx // 4) % 8) // 4) * 16
                + (((elem_idx // 4) % 4) % 2) * 8
                + (epi_tid % 32) // 4
            ),
            "hidden_in_cta_tile": (
                (epi_tid // 32) * 32
                + (epi_tid % 4) * 4
                + (((elem_idx // 4) % 4) // 2) * 16
                + elem_idx % 4
            ),
        },
    )
    # SIMT REDG emits one 8B red.v2.bf16x2 per 4 hidden elements. Each
    # 64-token subtile contributes two token rows per lane and 8 hidden
    # segments per token row.
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, cta_token_tile_size // 64 * 16)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (cta_token_tile_size, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": (
                (iter_idx // 16) * 64
                + ((iter_idx % 16) // 8) * 32
                + ((iter_idx % 8) // 4) * 16
                + ((iter_idx % 4) % 2) * 8
                + (epi_tid % 32) // 4
            ),
            "hidden_in_cta_tile": (
                (epi_tid // 32) * 32 + (epi_tid % 4) * 4 + ((iter_idx % 4) // 2) * 16
            ),
        },
    )
    return store_out_mapping, None, fundamental_mapping


def make_fc2_ublk_store_out_mapping(
    combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
):
    assert cta_hidden_tile_size == 128
    assert cta_token_tile_size % 64 == 0
    # UBLK pushes whole hidden rows by byte count, so the token/hidden mapping
    # is element-indexed and dtype-independent (wire dtype only sets copy bytes).
    assert combine_format.act_dtype.width in (4, 8, 16), "fc2 UBLK wire dtype must be fp4/fp8/bf16."
    assert cta_token_tile_size <= 256
    max_token_cta_tile = 256
    fundamental_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "elem_idx"), (128, cta_token_tile_size)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (max_token_cta_tile, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, elem_idx: {
            "token_in_cta_tile": elem_idx // cta_hidden_tile_size * 32
            + epi_tid % 8
            + epi_tid // 32 * 8
            + ((epi_tid % 32) // 8) * 64,
            "hidden_in_cta_tile": elem_idx % cta_hidden_tile_size,
        },
    )
    # One row per thread covers two 64-row subtiles per iteration.
    copy_iters = (cta_token_tile_size + 127) // 128
    store_out_mapping = FunctionMapping(
        domain=CoordinateSpace(("epi_tid", "iter_idx"), (128, copy_iters)),
        codomain=CoordinateSpace(
            ("token_in_cta_tile", "hidden_in_cta_tile"), (max_token_cta_tile, cta_hidden_tile_size)
        ),
        function=lambda epi_tid, iter_idx: {
            "token_in_cta_tile": (
                iter_idx * 128 + ((epi_tid % 32) // 16) * 64 + (epi_tid // 32) * 16 + epi_tid % 16
            ),
            "hidden_in_cta_tile": 0,
        },
    )
    if combine_format.is_quantized:
        # Quantized UBLK reuses the hidden-contiguous STG mappings for 128-bit R2S.
        _, sf_store_out_mapping, fundamental_mapping = make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    else:
        sf_store_out_mapping = None
    return store_out_mapping, sf_store_out_mapping, fundamental_mapping


# (...) -> ((atom_v, 1))
@cute.jit
def wrap_into_copy_standard_layout(tensor: cute.Tensor):
    tensor = cute.coalesce(cute.flatten(tensor))
    tensor = cute.append_ones(tensor, cute.rank(tensor) + 1)
    tensor = cute.group_modes(tensor, 0, cute.rank(tensor) - 1)
    tensor = cute.group_modes(tensor, 0, cute.rank(tensor))
    return tensor


@cute.jit
def fc2_f2fp(*tensors, alpha_val: Optional[cutlass.Float32] = None, **_) -> cute.Tensor:
    reorder_dtype = cutlass.BFloat16
    total_size = 0
    for t in tensors:
        total_size += cute.size(t)
    converted_acc = cute.make_rmem_tensor((total_size,), reorder_dtype)
    elems_processed = 0
    for t in tensors:
        current_tensor_size = cute.size(t)
        dst = cute.make_tensor(
            converted_acc.iterator + elems_processed, cute.make_layout((current_tensor_size,))
        )
        if cutlass.const_expr(alpha_val is None):
            dst.store(t.load().to(reorder_dtype))
        else:
            if cutlass.const_expr(current_tensor_size % 2 != 0):
                raise ValueError("fc2_f2fp expects even elements for each input tensor.")
            scaled = cute.make_rmem_tensor((current_tensor_size,), cutlass.Float32)
            for i in cutlass.range_constexpr(0, current_tensor_size, 2):
                # scaled[i] = t[i] * alpha_val
                s0, s1 = cute.arch.mul_packed_f32x2((t[i], t[i + 1]), (alpha_val, alpha_val))
                scaled[i] = s0
                scaled[i + 1] = s1
            dst.store(scaled.load().to(reorder_dtype))
        elems_processed += current_tensor_size
    return converted_acc


@cute.jit
def post_f2fp_reorder_identity(*, casted: cute.Tensor, **_):
    # UBLK: the f2fp output is already in the pre-store distribution (each lane
    # owns one hidden element across the 64 subtile tokens); no reorder needed.
    return casted


@cute.jit
def fc2_stg_tmem_acc_load(*, tmem_subtile_tensor: cute.Tensor, **_):
    atom_ld16x256 = cute.make_copy_atom(tcgen05.Ld16x256bOp(tcgen05.Repetition.x8), cutlass.Float32)
    ptr = tmem_subtile_tensor.iterator
    half_lane_offset = 16 * TmemTranspose32x64B16Movm._tmem_row_stride
    top_view = cute.make_tensor(ptr, TmemTranspose32x64B16Movm._tmem_layout(16, 64))
    bottom_view = cute.make_tensor(
        ptr + half_lane_offset, TmemTranspose32x64B16Movm._tmem_layout(16, 64)
    )
    top = cute.make_rmem_tensor((32,), cutlass.Float32)
    bottom = cute.make_rmem_tensor((32,), cutlass.Float32)
    cute.copy(atom_ld16x256, top_view, TmemTranspose32x64B16Movm._rmem_copy_view(top, 32))
    cute.copy(atom_ld16x256, bottom_view, TmemTranspose32x64B16Movm._rmem_copy_view(bottom, 32))
    return top, bottom


@cute.jit
def fc2_ublk_tmem_acc_load(*, tmem_subtile_tensor: cute.Tensor, epi, **_):
    # UBLK consumes a warp-local 32-hidden x 64-token slice.  The caller passes
    # the CTA-level 128-hidden x 64-token subtile view, so select this epi
    # warp's hidden block before issuing LDTM.x64.
    tmem_subtile_per_warp = cute.logical_divide(tmem_subtile_tensor, (32, None))[
        (None, epi.warp_idx), None
    ]
    raw_regs = cute.make_rmem_tensor((64,), cutlass.Float32)
    atom_ld32x32_x64 = cute.make_copy_atom(
        tcgen05.Ld32x32bOp(tcgen05.Repetition.x64), cutlass.Float32
    )
    cute.copy(
        atom_ld32x32_x64,
        wrap_into_copy_standard_layout(tmem_subtile_per_warp),
        wrap_into_copy_standard_layout(raw_regs),
    )
    return (raw_regs,)


@cute.jit
def fc2_stg_post_f2fp_reorder(
    *,
    casted: cute.Tensor,  # (subtile_cnt,)
    tmem_subtile_view: cute.Tensor,  # (epi_tile_m, epi_tile_n)
    **_,
):
    if cutlass.const_expr(cute.size(casted) != 64):
        raise NotImplementedError("fc2 stg pass expects 64 BF16 registers before store reorder.")
    return TmemTranspose32x64B16Movm(tmem_subtile_view.iterator, casted)()


@cute.jit
def fc2_redg_post_f2fp_reorder(*, casted: cute.Tensor, tmem_subtile_view: cute.Tensor, **_):
    # (epi_tid, elem_idx) -> (token_64, hidden_128), each thread hold token_2 x hidden_32
    natural = fc2_stg_post_f2fp_reorder(casted=casted, tmem_subtile_view=tmem_subtile_view)
    core_matrix_reorder_sttm_atom = cute.make_copy_atom(
        tcgen05.St32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
    )
    core_matrix_reorder_ldtm_atom = cute.make_copy_atom(
        tcgen05.Ld16x256bOp(tcgen05.Repetition.x2), cutlass.Float32
    )
    # ((16, 2), token_32_group)
    token_groups = cute.logical_divide(cute.zipped_divide(natural, (32,)), (16, None))
    out = cute.make_rmem_tensor(token_groups.shape, casted.dtype)
    out_as_i32 = cute.recast_tensor(out, cutlass.Float32)
    # (32, 64)
    tmem_warp = cute.flat_divide(tmem_subtile_view, (32, cute.size(tmem_subtile_view, 1)))[
        None, None, 0, 0
    ]
    # (16, 16, 16dp_group, token_32_groups). Note, this tmem can provide 2x cols since the original is bf16.
    tmem_groups = cute.flat_divide(tmem_warp, (16, 16))
    for group_idx in cutlass.range_constexpr(cute.size(token_groups, 1)):
        sttm_source = cute.recast_tensor(token_groups[None, group_idx], cutlass.Float32)
        sttm_destination = tmem_groups[None, None, None, group_idx]
        cute.copy(
            core_matrix_reorder_sttm_atom,
            wrap_into_copy_standard_layout(sttm_source),
            wrap_into_copy_standard_layout(sttm_destination),
        )
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(tmem_groups[None, None, 0, group_idx]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 0), group_idx]),
        )
        cute.copy(
            core_matrix_reorder_ldtm_atom,
            wrap_into_copy_standard_layout(tmem_groups[None, None, 1, group_idx]),
            wrap_into_copy_standard_layout(out_as_i32[(None, 1), group_idx]),
        )
    return cute.coalesce(out)


@cute.jit
def fc2_stg_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always BF16 before quantization
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    if cutlass.const_expr(epi.combine_format.is_quantized):
        if cutlass.const_expr(epi.lean_fc2_quant):
            data_subtile, sf_regs = QuantImpl(
                epi.combine_format, "regs_in_thread", lean_combine=True
            )(subtile)
        else:
            data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
    else:
        data_subtile = subtile
        sf_regs = None
    stg_width_elems: cutlass.Constexpr[int] = min(32, 256 // data_subtile.element_type.width)
    stg_bits: cutlass.Constexpr[int] = stg_width_elems * data_subtile.element_type.width
    copy_atom_vec = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=stg_bits
    )
    elems_per_thread: cutlass.Constexpr[int] = cute.size(data_subtile)
    if cutlass.const_expr(elems_per_thread % stg_width_elems != 0):
        raise ValueError(
            "fc2 STG store requires pre-store elems per thread to be divisible "
            f"by STG issue width, got {elems_per_thread} and {stg_width_elems}."
        )

    if cutlass.const_expr(sf_regs is not None):
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]

    iters_per_subtile: cutlass.Constexpr[int] = elems_per_thread // stg_width_elems
    copy_src = cute.zipped_divide(data_subtile, (stg_width_elems,))
    single_copy_layout = cute.make_layout(((stg_width_elems, 1),), stride=((1, 0),))
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(iters_per_subtile)
    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            src_i = cute.make_tensor(copy_src[None, local_iter].iterator, single_copy_layout)
            dst_i = cute.make_tensor(dst_ptr, single_copy_layout)
            cute.copy(
                copy_atom_vec,
                cute.recast_tensor(src_i, cutlass.Int32),
                cute.recast_tensor(dst_i, cutlass.Int32),
            )


@cute.jit
def fc2_tma_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always BF16 before quantization
    work_tile_info: SwapAbFc12WorkTileInfo,
    epilogue_iter_idx: cutlass.Int32,
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    if cutlass.const_expr(
        epi.smem_tensor is None or epi.tma_atom_fc2_output is None or epi.fc2_tma_output is None
    ):
        raise ValueError(
            "FC2 TMA store requires staged SMEM, a TMA atom, and a token-major output tensor."
        )

    if cutlass.const_expr(epi.combine_format.is_quantized):
        data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
    else:
        data_subtile = subtile
        sf_regs = None

    if cutlass.const_expr(sf_regs is not None):
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]

    vector_elements: cutlass.Constexpr[int] = 128 // data_subtile.element_type.width
    if cutlass.const_expr(
        cute.rank(data_subtile) != 1
        or cute.size(data_subtile) != 2 * 32
        or data_subtile.stride[0] != 1
        or 32 % vector_elements != 0
    ):
        raise ValueError(
            "FC2 TMA store requires a contiguous two-token by 32-hidden register fragment."
        )

    stage_idx = epilogue_iter_idx % cutlass.Int32(epi.fc2_tma_stages)
    smem_stage = epi.smem_tensor[None, None, stage_idx]
    store_atom = cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), data_subtile.element_type, num_bits_per_copy=128
    )
    tiler_mn = (64, 128)
    layout_copy_tv = cute.make_layout(((32, 4), (32, 2)), stride=((1, 2048), (64, 32)))
    tiled_store = cute.make_tiled_copy(store_atom, layout_copy_tv, tiler_mn)
    thread_store = tiled_store.get_slice(epi.tidx)
    smem_partition = thread_store.partition_D(smem_stage)
    register_partition = cute.composition(data_subtile, cute.make_layout(smem_partition.shape))
    cute.copy(tiled_store, register_partition, smem_partition)

    token_tile_idx = (
        work_tile_info.cumulative_data_physical_row
        + work_tile_info.tile_n_idx * cutlass.Int32(epi.cta_tile_n)
        + subtile_idx * cutlass.Int32(epi._EpilogueTokenTileSize)
    ) // cutlass.Int32(epi._EpilogueTokenTileSize)
    tiled_output = cute.flat_divide(
        epi.fc2_tma_output, (epi._EpilogueTokenTileSize, epi._EpilogueFc2HiddenTileSize)
    )
    gmem_subtile = tiled_output[None, None, token_tile_idx, work_tile_info.tile_m_idx, 0]
    tma_smem_src, tma_gmem_dst = cpasync.tma_partition(
        epi.tma_atom_fc2_output,
        0,
        cute.make_layout(1),
        cute.group_modes(smem_stage, 0, 2),
        cute.group_modes(gmem_subtile, 0, 2),
    )

    epilogue_barrier = SwapABGatedActEpilogue.epilogue_sync_barrier()
    cute.arch.fence_proxy("async.shared", space="cta")
    epilogue_barrier.arrive_and_wait()
    if epi.warp_idx == cutlass.Int32(0):
        cute.copy(epi.tma_atom_fc2_output, tma_smem_src, tma_gmem_dst)
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(epi.fc2_tma_stages - 1, read=True)
    epilogue_barrier.arrive_and_wait()


@cute.jit
def fc2_ublk_store_function_impl(
    *,
    epi,
    subtile: cute.Tensor,  # Always bf16 pre-quant tensor
    epilogue_iter_idx: cutlass.Int32,
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    smem_tensor = epi.smem_tensor
    if cutlass.const_expr(smem_tensor is None):
        raise ValueError("fc2 UBLK store requires epi.smem_tensor.")
    quantized: cutlass.Constexpr[bool] = epi.combine_format.is_quantized
    if cutlass.const_expr(quantized):
        # The transposed fragment gives each thread complete hidden-axis scale blocks.
        data_subtile, sf_regs = QuantImpl(epi.combine_format, "regs_in_thread")(subtile)
        sf_scales_per_stg: cutlass.Constexpr[int] = 32 // epi.combine_format.scale_block
        token_groups_per_subtile: cutlass.Constexpr[int] = epi._EpilogueTokenTileSize // 32
        for token_group in cutlass.range_constexpr(token_groups_per_subtile):
            sf_iter = (
                cutlass.Int32(subtile_idx) * cutlass.Int32(token_groups_per_subtile) + token_group
            )
            sf_ptr, sf_pred = fc2_output_router.get_sf_dst(sf_iter)
            if sf_pred != cutlass.Int32(0):
                sf_dst = cute.make_tensor(sf_ptr, cute.make_layout((sf_scales_per_stg,)))
                for scale_idx in cutlass.range_constexpr(sf_scales_per_stg):
                    sf_dst[scale_idx] = sf_regs[token_group * sf_scales_per_stg + scale_idx]
    else:
        data_subtile = subtile

    smem_read_write_bar = SwapABGatedActEpilogue.epilogue_sync_barrier()
    warp_idx = epi.warp_idx
    lane_idx = epi.lane_idx
    stage_cnt: cutlass.Constexpr[int] = epi.fc2_tma_stages
    stage_idx = epilogue_iter_idx % cutlass.Int32(stage_cnt)
    smem_stage = smem_tensor[None, None, stage_idx]
    fine = epi.opt_iket_fine
    _fine_push(fine, "fc2.stage_wait")
    # Reuse waits here so the previous copy overlaps the next subtile; run() drains the final commit.
    cute.arch.cp_async_bulk_wait_group(stage_cnt - 1, read=True)
    smem_read_write_bar.arrive_and_wait()
    _fine_pop(fine)
    _fine_push(fine, "fc2.sts")

    if cutlass.const_expr(quantized):
        vector_elements: cutlass.Constexpr[int] = 128 // data_subtile.element_type.width
        if cutlass.const_expr(
            cute.rank(data_subtile) != 1
            or cute.size(data_subtile) != 2 * 32
            or data_subtile.stride[0] != 1
            or 32 % vector_elements != 0
        ):
            raise ValueError(
                "FC2 UBLK store requires a contiguous two-token by 32-hidden register fragment."
            )
        store_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), data_subtile.element_type, num_bits_per_copy=128
        )
        tiler_mn = (64, 128)
        layout_copy_tv = cute.make_layout(((32, 4), (32, 2)), stride=((1, 2048), (64, 32)))
        tiled_store = cute.make_tiled_copy(store_atom, layout_copy_tv, tiler_mn)
        thread_store = tiled_store.get_slice(epi.tidx)
        smem_partition = thread_store.partition_D(smem_stage)
        register_partition = cute.composition(data_subtile, cute.make_layout(smem_partition.shape))
        cute.copy(tiled_store, register_partition, smem_partition)
    else:
        if cutlass.const_expr(cute.size(data_subtile) != epi._EpilogueTokenTileSize):
            raise ValueError(
                "BF16 UBLK staging requires one register per token in the 64-token subtile."
            )
        warp_hidden_base = cutlass.Int32(warp_idx * 32)
        for token_idx in cutlass.range_constexpr(epi._EpilogueTokenTileSize):
            smem_stage[token_idx, warp_hidden_base + lane_idx] = data_subtile[token_idx]

    cute.arch.fence_proxy("async.shared", space="cta")
    smem_read_write_bar.arrive_and_wait()
    _fine_pop(fine)
    _fine_push(fine, "fc2.issue")

    copy_elems = cutlass.Int32(epi._EpilogueFc2HiddenTileSize)
    if cutlass.const_expr(epi.fc2_hidden_needs_predicate):
        copy_elems = cutlass.Int32(fc2_output_router.valid_hidden_this_cta_tile)
    copy_bytes = copy_elems * epi.combine_format.act_dtype.width // 8
    scratch_row = warp_idx * cutlass.Int32(16) + lane_idx % cutlass.Int32(16)
    copy_iters: cutlass.Constexpr[int] = fc2_output_router.data_mapping.domain.axis_size("iter_idx")
    for ublk_iter_idx in cutlass.range_constexpr(copy_iters):
        owned_subtile = cutlass.Int32(ublk_iter_idx * 2) + lane_idx // cutlass.Int32(16)
        if owned_subtile == subtile_idx:
            dst_ptr, pred = fc2_output_router.get_data_dst(ublk_iter_idx)
            if pred != cutlass.Int32(0):
                src_row = cute.slice_(smem_stage, (scratch_row, None))
                if cutlass.const_expr(epi.reduce_topk_in_epilogue):
                    cp_reduce_async_bulk_add_bf16_s2g(dst_ptr, src_row.iterator, copy_bytes)
                else:
                    cp_async_bulk_s2g(dst_ptr, src_row.iterator, copy_bytes)

    cute.arch.cp_async_bulk_commit_group()
    _fine_pop(fine)


@cute.jit
def fc2_redg_store_function(
    *,
    epi,
    subtile: cute.Tensor,  # Always bf16; in-kernel reduce never quantizes
    subtile_idx: cutlass.Int32,
    fc2_output_router: Fc2OutputRouter,
    **_,
):
    redg_width_elems: cutlass.Constexpr[int] = 4
    elems_per_thread: cutlass.Constexpr[int] = cute.size(subtile)
    if cutlass.const_expr(elems_per_thread % redg_width_elems != 0):
        raise ValueError(
            "fc2 REDG store requires pre-store elems per thread to be divisible "
            f"by REDG issue width, got {elems_per_thread} and {redg_width_elems}."
        )
    iters_per_subtile: cutlass.Constexpr[int] = elems_per_thread // redg_width_elems
    subtile_iter_base = cutlass.Int32(subtile_idx) * cutlass.Int32(iters_per_subtile)
    subtile_by_redg_issue = cute.zipped_divide(subtile, (redg_width_elems,))

    for local_iter in cutlass.range_constexpr(iters_per_subtile):
        global_iter = subtile_iter_base + cutlass.Int32(local_iter)
        dst_ptr, pred = fc2_output_router.get_data_dst(global_iter)
        if pred != cutlass.Int32(0):
            bf16x4 = subtile_by_redg_issue[None, local_iter]
            packed_bf16x2 = cute.recast_tensor(bf16x4, cutlass.Float32)
            red_add_relaxed_sys_v2_bf16x2(
                dst_ptr, cutlass.Float32(packed_bf16x2[0]), cutlass.Float32(packed_bf16x2[1])
            )


def make_fc2_stg_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_stg_post_f2fp_reorder,
        store_function=fc2_stg_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=True,
    )


def make_fc2_tma_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_stg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_stg_post_f2fp_reorder,
        store_function=fc2_tma_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=True,
    )


def make_fc2_redg_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = (
        make_fc2_redg_cta_store_out_mapping(
            combine_format, cta_token_tile_size, cta_hidden_tile_size
        )
    )
    return Fc2ProcessPipeline(
        tmem_acc_load=fc2_stg_tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=fc2_redg_post_f2fp_reorder,
        store_function=fc2_redg_store_function,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=True,
    )


def make_fc2_ublk_process_pipeline(
    *, combine_format: CombineFormat, cta_token_tile_size: int, cta_hidden_tile_size: int
) -> Fc2ProcessPipeline:
    store_out_mapping, sf_store_out_mapping, fundamental_mapping = make_fc2_ublk_store_out_mapping(
        combine_format, cta_token_tile_size, cta_hidden_tile_size
    )
    if combine_format.is_quantized:
        if combine_format.act_dtype.width != 8:
            raise ValueError("FC2 UBLK quantization requires an FP8 combine payload.")
        tmem_acc_load = fc2_stg_tmem_acc_load
        post_f2fp_reorder = fc2_stg_post_f2fp_reorder
    else:
        tmem_acc_load = fc2_ublk_tmem_acc_load
        post_f2fp_reorder = post_f2fp_reorder_identity
    return Fc2ProcessPipeline(
        tmem_acc_load=tmem_acc_load,
        f2fp=fc2_f2fp,
        post_f2fp_reorder=post_f2fp_reorder,
        store_function=fc2_ublk_store_function_impl,
        fc2_cta_tile_mapping=fundamental_mapping,
        store_out_mapping=store_out_mapping,
        sf_store_out_mapping=sf_store_out_mapping,
        require_tmem_trans=combine_format.is_quantized,
    )


__all__ = [
    "Fc2OutputRouter",
    "Fc2ProcessPipeline",
    "GatedActEpilogueArgs",
    "QuantImpl",
    "SwapABGatedActEpilogue",
    "make_fc2_redg_process_pipeline",
    "make_fc2_stg_process_pipeline",
    "make_fc2_tma_process_pipeline",
    "make_fc2_ublk_process_pipeline",
]
