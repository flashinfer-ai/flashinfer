"""SM100 mixed-MMA tile geometry, descriptors and K128 issue sequence."""

from __future__ import annotations

from dataclasses import dataclass

import cutlass
import cutlass.cute as cute
from cutlass import Uint16, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import arith, llvm

from ..ptx.tcgen05 import (
    mma_mxf8f6f4_2cta,
    mma_mxf8f6f4_k128_2cta,
    tcgen05_after_thread_sync,
    tcgen05_commit_2cta,
    utccp_sf_4x32_2cta,
)


@cute.jit
def blockscaled_instr_desc_hi(umma_n: Uint32, weight_format: Uint32,
                              weight_sf_byte: Uint32,
                              activation_sf_byte: Uint32) -> Uint32:
    """Encode the reference swapped-operand MXF8F6F4 instruction descriptor.

    The descriptor's A operand is the *weight* (E2M1=5 or E4M3=0) and B is
    the FP8 activation. `umma_n` is align(valid_m, 16), not BLOCK_M. SF byte
    selectors map to bits 29:30 for weight A and 4:5 for activation B.
    See the CUTLASS InstrDescriptorBlockScaled bitfield definition.
    """
    return (Uint32(16) << 24) | (Uint32(1) << 23) | \
        ((umma_n >> 3) << 17) | (weight_format << 7) | \
        (weight_sf_byte << 29) | (activation_sf_byte << 4)


@cute.jit
def kmajor_smem_desc(smem_ptr: cute.Pointer, stride_bytes: Uint32,
                     swizzle_128: cutlass.Constexpr,
                     byte_offset: Uint32) -> Uint64:
    """SM100 descriptor for reference K-major 128B data or UTCCP SF tile."""
    address = Uint64(smem_ptr.toint()) + Uint64(byte_offset)
    layout = Uint64(2) if cutlass.const_expr(swizzle_128) else Uint64(0)
    return ((address >> 4) & Uint64(0x3fff)) | \
        (Uint64(stride_bytes >> 4) << 32) | (Uint64(1) << 46) | \
        (layout << 61)


@cute.jit
def kmajor_smem_desc_from_start(start_address_16b: Uint32,
                                stride_bytes: Uint32,
                                swizzle_128: cutlass.Constexpr) -> Uint64:
    """Keep descriptor flags fixed and replace only the reference low address."""
    layout = Uint64(2) if cutlass.const_expr(swizzle_128) else Uint64(0)
    return Uint64(start_address_16b & Uint32(0x3fff)) | \
        (Uint64(stride_bytes >> 4) << 32) | (Uint64(1) << 46) | \
        (layout << 61)


@dataclass(frozen=True)
class MixedMmaTile:
    block_m: int
    hidden: int
    intermediate: int
    block_k: int = 128
    block_n: int = 128
    epilogue_stages: int = 2

    def __post_init__(self) -> None:
        if self.block_m not in (8, 16, 32, 64, 96, 128, 192, 240) or self.block_m % 16:
            raise ValueError('2CTA mixed MMA requires a 16-aligned routed BM')
        if self.block_n != 128 or self.block_k != 128:
            raise ValueError('GC mixed MMA uses BN128 and BK128')
        if self.hidden % 256 or self.intermediate % 128:
            raise ValueError('GC two-CTA N geometry is not integral')
        if self.l1_n_blocks > 64:
            raise ValueError('L2 dependency mask exceeds 64 bits')

    @property
    def sf_block_m(self) -> int:
        return (self.block_m + 127) // 128 * 128

    @property
    def l1_n_blocks(self) -> int:
        return 2 * self.intermediate // self.block_n

    @property
    def l2_n_blocks(self) -> int:
        return self.hidden // self.block_n

    @property
    def l1_clusters(self) -> int:
        return self.l1_n_blocks // 2

    @property
    def l2_clusters(self) -> int:
        return self.l2_n_blocks // 2

    @property
    def accum_tmem_cols(self) -> int:
        return self.block_m * self.epilogue_stages

    @property
    def sfa_tmem_cols(self) -> int:
        return self.sf_block_m // 32

    @property
    def sfb_tmem_cols(self) -> int:
        return self.block_n // 32

    @property
    def tmem_cols(self) -> int:
        needed = self.accum_tmem_cols + self.sfa_tmem_cols + self.sfb_tmem_cols
        result = max(32, 1 << (needed - 1).bit_length())
        if result > 512:
            raise ValueError('GC TMEM allocation exceeds 512 columns')
        return result

    @property
    def smem_a_stage_bytes(self) -> int:
        return self.block_m // 2 * self.block_k  # FP8 activation

    @property
    def smem_b_stage_bytes(self) -> int:
        return self.block_n * self.block_k  # packed FP4 TMA-unpacked to 8b SMEM

    @property
    def smem_sfa_stage_bytes(self) -> int:
        return self.sf_block_m * (self.block_k // 128) * 4

    @property
    def smem_sfb_stage_bytes(self) -> int:
        return self.block_n * (self.block_k // 128) * 4

    def l2_k_mask(self, k_block: int) -> int:
        """Two completed W1 N128 tiles feed one W2 K128 tile."""
        width = self.block_k // (self.block_n // 2)
        if k_block < 0 or (k_block + 1) * width > self.l1_n_blocks:
            raise ValueError('invalid L2 K block')
        return ((1 << width) - 1) << (k_block * width)


@dsl_user_op
def _select_sfb_id(is_shared: Uint32, shared_id: Uint32, routed_id: Uint32,
                   *, loc=None, ip=None) -> Uint32:
    """Pure reference ternary for the shared/routed SFB byte ID."""
    is_shared_pred = (is_shared != Uint32(0)).ir_value(loc=loc, ip=ip)
    return Uint32(arith.select(
        is_shared_pred, shared_id.ir_value(loc=loc, ip=ip),
        routed_id.ir_value(loc=loc, ip=ip), loc=loc, ip=ip))


@cute.jit
def issue_mixed_k128(weight_desc_start: Uint32, activation_desc_start: Uint32,
                     weight_sf_smem: cute.Pointer,
                     activation_sf_smem: cute.Pointer, stage_idx: Uint32,
                     data_desc_hi: Uint64, sf_desc_hi: Uint64,
                     accum_tmem: Uint32, weight_sf_tmem: Uint32,
                     activation_sf_tmem: Uint32,
                     stage_empty_bar: cute.Pointer, accum_full_bar: cute.Pointer,
                     instr_desc_base: Uint32, k_block: Uint32,
                     last_k_block: Uint32, is_shared: Uint32,
                     sf_block_m: cutlass.Constexpr,
                     shared_weight_gran_k: cutlass.Constexpr,
                     has_shared: cutlass.Constexpr,
                     reuse_descriptors: cutlass.Constexpr) -> None:
    """Issue the reference swapped-A/B 2CTA mixed MMA for one K128 stage.

    The caller has already completed the stage's TMA G2S and mbarrier wait.
    It gives this function the per-stage A/B descriptor bases, the two SF
    shared-memory bases, and descriptor high fields retained by the leader
    warp across tasks and K stages.
    A=weight, B=activation in the instruction;
    routed FP4 uses format 5, shared FP8 format 0. Every K128 commits to the
    stage-empty barrier; the last K128 also commits to accumulator-full.
    """
    tcgen05_after_thread_sync()
    # The leader warp has already selected one task slot and pipeline stage.
    # Only stage A/B descriptor bases are prepared in the full warp. The
    # reference computes SF addresses, SF selection and K32 descriptors in
    # the single elected issuer after the stage-full wait.
    weight_start = weight_desc_start
    activation_start = activation_desc_start

    with cute.arch.elect_one():
        activation_sf_start = Uint32(
            (Uint32((activation_sf_smem + stage_idx * sf_block_m).toint()) >>
             Uint32(4)))
        activation_sf_desc = sf_desc_hi | Uint64(activation_sf_start)
        utccp_sf_4x32_2cta(activation_sf_desc, activation_sf_tmem)
        if cutlass.const_expr(sf_block_m > 128):
            activation_sf_desc = sf_desc_hi | Uint64(
                activation_sf_start + Uint32(32))
            utccp_sf_4x32_2cta(activation_sf_desc,
                               activation_sf_tmem + Uint32(4))
        copy_weight_sf = True
        if cutlass.const_expr(has_shared and shared_weight_gran_k == 128):
            copy_weight_sf = (is_shared == Uint32(0) or
                              k_block % Uint32(4) == Uint32(0))
        if copy_weight_sf:
            weight_sf_start = Uint32(
                (Uint32((weight_sf_smem + stage_idx * 128).toint()) >>
                 Uint32(4)))
            weight_sf_desc = sf_desc_hi | Uint64(weight_sf_start)
            utccp_sf_4x32_2cta(weight_sf_desc, weight_sf_tmem)

        # FP4 16U4_ALIGN16B TMA expands sixteen nibbles into an 8-byte
        # payload and 8-byte padding. K32 advances each descriptor low by
        # two 16-byte address units. Keep every mutation in the elected lane.
        runtime_instr_descs = ()
        for k in cutlass.range_constexpr(4):
            weight_sf_id = Uint32(k)
            if cutlass.const_expr(has_shared and shared_weight_gran_k == 128):
                weight_sf_id = _select_sfb_id(
                    is_shared, k_block % Uint32(4), Uint32(k))
            runtime_instr_desc = (
                instr_desc_base | (weight_sf_id << Uint32(29)) |
                Uint32(16 * k))
            runtime_instr_descs += (runtime_instr_desc,)
        if cutlass.const_expr(reuse_descriptors):
            weight_desc = data_desc_hi | Uint64(weight_start)
            activation_desc = data_desc_hi | Uint64(activation_start)
            first_accumulate = Uint32(1) if k_block > Uint32(0) else Uint32(0)
            mma_mxf8f6f4_k128_2cta(
                weight_desc, activation_desc, accum_tmem, first_accumulate,
                runtime_instr_descs[0], runtime_instr_descs[1],
                runtime_instr_descs[2], runtime_instr_descs[3],
                weight_sf_tmem, activation_sf_tmem)
        else:
            for k in cutlass.range_constexpr(4):
                weight_desc = data_desc_hi | Uint64(
                    weight_start + Uint32(2 * k))
                activation_desc = data_desc_hi | Uint64(
                    activation_start + Uint32(2 * k))
                accumulate = (Uint32(1) if k > 0 else
                              Uint32(1) if k_block > Uint32(0) else Uint32(0))
                mma_mxf8f6f4_2cta(
                    weight_desc, activation_desc, accum_tmem, accumulate,
                    runtime_instr_descs[k], weight_sf_tmem, activation_sf_tmem)
        tcgen05_commit_2cta(stage_empty_bar, Uint16(3))
        if last_k_block != Uint32(0):
            tcgen05_commit_2cta(accum_full_bar, Uint16(3))
    cute.arch.sync_warp()


@dsl_user_op
def select_weight_format(is_shared: Uint32, *, loc=None, ip=None) -> Uint32:
    """Keep the reference's shared/routed descriptor choice in one MMA path."""
    return Uint32(llvm.inline_asm(
        T.i32(), [is_shared.ir_value(loc=loc, ip=ip)],
        '{ .reg .pred p; setp.ne.u32 p, $1, 0; '
        'selp.b32 $0, 0, 5, p; }', '=r,r', has_side_effects=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))
