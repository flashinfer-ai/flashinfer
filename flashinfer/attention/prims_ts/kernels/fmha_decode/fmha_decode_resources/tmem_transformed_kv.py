# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformed NVFP4 K/V resource backed by tensor memory."""

from dataclasses import dataclass
from typing import ClassVar

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Uint8
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.enums import WorkAttr
from cutlass.experimental.task_scheduling.memory import (
    ResourceContext,
    SmemAllocation,
    TmemAllocation,
)
from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
    TaskLocalVariable,
    consumer_work,
    producer_work,
)

from ......cute_dsl.availability import is_rubin_cute_dsl_available
from ...placeholder_helpers import _placeholder_local_array
from ..fmha_decode_config import FmhaDecodeConfig
from .helpers_common import (
    _TASK_CACHE_TMEM_BASE_OFFSET,
    _TASK_CACHE_WARP_GRP_THREAD_IDX,
    Constexpr,
    DecodeGenResourceBase,
    ResourceVars,
    _decode_gen_task_cache,
)
from .smem_resources import SmemKvResource, SmemKvTileResource


def _has_packed_e4m3_e2m1_mul() -> bool:
    """Whether the compile target has the packed E4M3 x E2M1 multiply.

    PTX 9.4 (CUDA 13.4) adds it for SM100 and SM103; ptxas rejects it for
    SM107, which takes the f16x2 path instead.
    """
    if not cutlass.target_version(min_version="13.4"):
        return False
    if not is_rubin_cute_dsl_available():
        return True
    arch = BaseDSL._get_dsl().get_arch_enum()
    return not (Arch.sm_107 <= arch <= Arch.sm_107f)


@cute.jit
def _mul_e2m1x4_e4m3x4(packed_fp4: Int32, packed_sf: Int32) -> Int32:
    """Apply four E4M3 scale factors to four unpacked E2M1 values.

    Each FP4 value occupies the low nibble of one byte in the 32-bit input;
    byte i scales by byte i of ``packed_sf`` and the products round to E4M3.
    Both paths ignore the high nibble of every input byte: the packed
    ``e2m1p4`` operand defines it as padding (checked exhaustively on SM103
    for all byte and scale values), and the f16x2 fallback extracts nibbles
    itself. Callers may therefore pass packed payload bytes, or the same word
    shifted right by four, without masking.
    """
    if cutlass.const_expr(_has_packed_e4m3_e2m1_mul()):
        return cute.arch.inline_ptx(
            "mul.e4m3x4.e2m1p4x4.e4m3x4.satfinite {$w0}, {$r0}, {$r1};",
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )
    else:
        return cute.arch.inline_ptx(
            """
            {
                .reg .b8 fp4_01, fp4_23, unused0, unused1, unused2;
                .reg .b16 sf_01, sf_23, e4m3_01, e4m3_23;
                .reg .b32 tmp, h_01, h_23, sf_h_01, sf_h_23;
                and.b32 tmp, {$r0}, 0x0000000f;
                bfe.u32 h_01, {$r0}, 8, 4;
                shl.b32 h_01, h_01, 4;
                or.b32 h_01, h_01, tmp;
                mov.b32 {fp4_01, unused0, unused1, unused2}, h_01;
                bfe.u32 h_23, {$r0}, 16, 4;
                bfe.u32 tmp, {$r0}, 24, 4;
                shl.b32 tmp, tmp, 4;
                or.b32 h_23, h_23, tmp;
                mov.b32 {fp4_23, unused0, unused1, unused2}, h_23;
                mov.b32 {sf_01, sf_23}, {$r1};
                cvt.rn.f16x2.e2m1x2 h_01, fp4_01;
                cvt.rn.f16x2.e2m1x2 h_23, fp4_23;
                cvt.rn.f16x2.e4m3x2 sf_h_01, sf_01;
                cvt.rn.f16x2.e4m3x2 sf_h_23, sf_23;
                mul.rn.f16x2 h_01, h_01, sf_h_01;
                mul.rn.f16x2 h_23, h_23, sf_h_23;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_01, h_01;
                cvt.rn.satfinite.e4m3x2.f16x2 e4m3_23, h_23;
                mov.b32 {$w0}, {e4m3_01, e4m3_23};
            }
            """,
            write_only_types=[Int32],
            read_only_args=[packed_fp4, packed_sf],
        )


def has_direct_bf16_fp4_fp8_cvt() -> bool:
    """Whether PTX converts E2M1x2 and E4M3x2 directly to BF16x2 (CUDA 13.2+)."""
    return cutlass.target_version(min_version="13.2")


@cute.jit
def e4m3x2_to_bf16x2(packed_pair: Int32) -> Int32:
    """Convert the low two E4M3 bytes of a word to BF16x2 exactly."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            "{ .reg .b16 lo, hi; mov.b32 {lo, hi}, {$r0}; "
            "cvt.rn.bf16x2.e4m3x2 {$w0}, lo; }",
            write_only_types=[Int32],
            read_only_args=[packed_pair],
        )
    return cute.arch.inline_ptx(
        "{ .reg .b16 lo, hi, h0, h1, b0, b1; .reg .b32 half2; "
        "mov.b32 {lo, hi}, {$r0}; cvt.rn.f16x2.e4m3x2 half2, lo; "
        "mov.b32 {h0, h1}, half2; cvt.rn.bf16.f16 b0, h0; cvt.rn.bf16.f16 b1, h1; "
        "mov.b32 {$w0}, {b0, b1}; }",
        write_only_types=[Int32],
        read_only_args=[packed_pair],
    )


@cute.jit
def broadcast_byte(word: Int32, byte_idx: Int32) -> Int32:
    """Replicate byte ``byte_idx`` (runtime, 0..3) of ``word`` into all bytes."""
    return cute.arch.inline_ptx(
        "prmt.b32 {$w0}, {$r0}, {$r0}, {$r1};",
        write_only_types=[Int32],
        read_only_args=[word, byte_idx * Int32(0x1111)],
    )


@cute.jit
def broadcast_byte_const(word: Int32, byte_idx: Constexpr[int]) -> Int32:
    """``broadcast_byte`` with a compile-time byte, as an immediate selector."""
    imm = ("0x0000", "0x1111", "0x2222", "0x3333")[byte_idx]
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r0}}, {imm};",
        write_only_types=[Int32],
        read_only_args=[word],
    )


@cute.jit
def scale_e2m1x4_unpacked_to_bf16x2x2(
    unpacked_fp4: Int32, sf_lo: Int32, sf_hi: Int32
) -> tuple[Int32, Int32]:
    """Scale four unpacked E2M1 values into two exact BF16x2 words.

    ``unpacked_fp4`` holds one E2M1 value in the low nibble of each byte (the
    ``B4X16_P64`` LDSM output). Values 0/1 use ``sf_lo`` and values 2/3 use
    ``sf_hi``. Every E2M1 x E4M3 product has at most five significant bits,
    so the BF16 multiply is exact; nothing is rounded through E4M3.
    """
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b32 x, y;
                .reg .b8 b0, b1, b2, b3;
                and.b32 x, {$r0}, 0x0F0F0F0F;
                shr.u32 y, x, 4;
                or.b32 x, x, y;
                mov.b32 {b0, b1, b2, b3}, x;
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b2;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            }""",
            write_only_types=[Int32, Int32],
            read_only_args=[unpacked_fp4, sf_lo, sf_hi],
        )
    return cute.arch.inline_ptx(
        """{
            .reg .b32 x, y, h01, h23;
            .reg .b8 b0, b1, b2, b3;
            .reg .b16 a0, a1, a2, a3, c0, c1, c2, c3;
            and.b32 x, {$r0}, 0x0F0F0F0F;
            shr.u32 y, x, 4;
            or.b32 x, x, y;
            mov.b32 {b0, b1, b2, b3}, x;
            cvt.rn.f16x2.e2m1x2 h01, b0;
            cvt.rn.f16x2.e2m1x2 h23, b2;
            mov.b32 {a0, a1}, h01;
            mov.b32 {a2, a3}, h23;
            cvt.rn.bf16.f16 c0, a0;
            cvt.rn.bf16.f16 c1, a1;
            cvt.rn.bf16.f16 c2, a2;
            cvt.rn.bf16.f16 c3, a3;
            mov.b32 {$w0}, {c0, c1};
            mov.b32 {$w1}, {c2, c3};
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
        }""",
        write_only_types=[Int32, Int32],
        read_only_args=[unpacked_fp4, sf_lo, sf_hi],
    )


@cute.jit
def scale_e2m1x8_packed_to_bf16x2x4(packed: Int32, sf_pair: Int32):
    """Scale eight packed E2M1 values (two per byte) into four exact BF16x2."""
    if cutlass.const_expr(has_direct_bf16_fp4_fp8_cvt()):
        return cute.arch.inline_ptx(
            """{
                .reg .b8 b0, b1, b2, b3;
                mov.b32 {b0, b1, b2, b3}, {$r0};
                cvt.rn.bf16x2.e2m1x2 {$w0}, b0;
                cvt.rn.bf16x2.e2m1x2 {$w1}, b1;
                cvt.rn.bf16x2.e2m1x2 {$w2}, b2;
                cvt.rn.bf16x2.e2m1x2 {$w3}, b3;
                mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
                mul.rn.bf16x2 {$w1}, {$w1}, {$r1};
                mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
                mul.rn.bf16x2 {$w3}, {$w3}, {$r1};
            }""",
            write_only_types=[Int32, Int32, Int32, Int32],
            read_only_args=[packed, sf_pair],
        )
    lo = scale_e2m1x4_unpacked_to_bf16x2x2(packed & Int32(0x0F0F0F0F), sf_pair, sf_pair)
    hi = scale_e2m1x4_unpacked_to_bf16x2x2(
        (packed >> Int32(4)) & Int32(0x0F0F0F0F), sf_pair, sf_pair
    )
    # Unpacked order is (b0lo, b1lo), (b2lo, b3lo); restore (lo, hi) per byte.
    return (
        byte_permute(lo[0], hi[0], 0x5410),
        byte_permute(lo[0], hi[0], 0x7632),
        byte_permute(lo[1], hi[1], 0x5410),
        byte_permute(lo[1], hi[1], 0x7632),
    )


@cute.jit
def scale_e2m1_token_quad_to_bf16x2x4(word: Int32, sf01: Int32, sf23: Int32):
    """Scale a byte column of four tokens into token-pair BF16x2 words.

    Byte i of ``word`` is token i, holding head-dim rows d (low nibble) and
    d+1 (high nibble). Returns (row d: tokens 0/1, row d: tokens 2/3,
    row d+1: tokens 0/1, row d+1: tokens 2/3); ``sf01``/``sf23`` are the
    per-token BF16 scale pairs. Two byte-selects pair the tokens' nibbles of
    each row so one conversion yields a token pair; products are exact.
    """
    return cute.arch.inline_ptx(
        """{
            .reg .b32 s4, s8, lo, hi;
            .reg .b8 l0, l1, l2, l3, h0, h1, h2, h3;
            shr.u32 s4, {$r0}, 4;
            shr.u32 s8, {$r0}, 8;
            lop3.b32 lo, {$r0}, s4, 0x0F0F0F0F, 0xE4;
            lop3.b32 hi, s4, s8, 0x0F0F0F0F, 0xE4;
            mov.b32 {l0, l1, l2, l3}, lo;
            mov.b32 {h0, h1, h2, h3}, hi;
            cvt.rn.bf16x2.e2m1x2 {$w0}, l0;
            cvt.rn.bf16x2.e2m1x2 {$w1}, l2;
            cvt.rn.bf16x2.e2m1x2 {$w2}, h0;
            cvt.rn.bf16x2.e2m1x2 {$w3}, h2;
            mul.rn.bf16x2 {$w0}, {$w0}, {$r1};
            mul.rn.bf16x2 {$w1}, {$w1}, {$r2};
            mul.rn.bf16x2 {$w2}, {$w2}, {$r1};
            mul.rn.bf16x2 {$w3}, {$w3}, {$r2};
        }""",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[word, sf01, sf23],
    )


@cute.jit
def byte_permute(a: Int32, b: Int32, selector: Constexpr[int]) -> Int32:
    """Pick four bytes of the pair (``a`` = bytes 0-3, ``b`` = bytes 4-7).

    Nibble i of ``selector`` (least significant first) names the source byte
    of result byte i, as in PTX ``prmt.b32``. For example ``0x5140`` packs
    a0, b0, a1, b1: it interleaves the even/odd E4M3 products of two words.
    """
    return cute.arch.inline_ptx(
        f"prmt.b32 {{$w0}}, {{$r0}}, {{$r1}}, {selector:#x};",
        write_only_types=[Int32],
        read_only_args=[a, b],
    )


# The shared-memory helpers below take 32-bit shared-window byte addresses.
# Transform loops compute per-token row addresses arithmetically (fragment
# slots, token rows, swizzled segments), so raw vector loads and stores avoid
# building a CuTe tensor view per row inside the rolled loops.


@cute.jit
def ld_shared_v4_b32(addr: Int32):
    """Load four 32-bit words from a 16-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v4.b32 {{$w0}, {$w1}, {$w2}, {$w3}}, [{$r0}];",
        write_only_types=[Int32, Int32, Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def ld_shared_v2_b32(addr: Int32):
    """Load two 32-bit words from an 8-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.v2.b32 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


@cute.jit
def ld_shared_b32(addr: Int32) -> Int32:
    """Load one 32-bit word from a 4-byte-aligned shared address."""
    return cute.arch.inline_ptx(
        "ld.shared.b32 {$w0}, [{$r0}];",
        write_only_types=[Int32],
        read_only_args=[addr],
    )


@cute.jit
def st_shared_v4_b32(addr: Int32, w0: Int32, w1: Int32, w2: Int32, w3: Int32) -> None:
    """Store four 32-bit words to a 16-byte-aligned shared address."""
    cute.arch.inline_ptx(
        "st.shared.v4.b32 [{$r0}], {{$r1}, {$r2}, {$r3}, {$r4}};",
        read_only_args=[addr, w0, w1, w2, w3],
    )


@cute.jit
def ldsm_m16n16_trans_b8(addr: Int32):
    """Transposing 16x16-byte LDSM: thread t gets byte columns t/4 and t/4+8
    for rows 4*(t%4) .. 4*(t%4)+3 (rows are supplied by threads 0..15)."""
    return cute.arch.inline_ptx(
        "ldmatrix.sync.aligned.m16n16.x1.trans.shared.b8 {{$w0}, {$w1}}, [{$r0}];",
        write_only_types=[Int32, Int32],
        read_only_args=[addr],
    )


@dataclass(kw_only=True)
class TmemTransformedKvResource(DecodeGenResourceBase):
    """Pipeline transformed NVFP4 K/V through TMEM.

    The transform warp groups dequantize a raw NVFP4 stage from
    :class:`SmemKvResource` (an unpacking-TMA stage with separate scale
    tensors, or packed interleaved rows) into E4M3 or BF16 operands, by Q's
    compute dtype. K is stored token-major and V is transposed, so both BMMs
    consume the transformed tile as operand A.
    """

    _task_local_specs: ClassVar[tuple[tuple, ...]] = (
        (
            "kv_desc_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed K consumed by QK MMA.",
        ),
        (
            "v_desc_0_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 0.",
        ),
        (
            "v_desc_1_slot",
            Int32,
            Int32(0),
            "TMEM address for transformed V consumed by VP MMA instance 1.",
        ),
    )
    cfg: Constexpr[FmhaDecodeConfig] = None
    src_smem_kv: Constexpr[SmemKvResource] = None
    _alloc: Constexpr[TmemAllocation | None] = None
    kv_desc_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_0_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    v_desc_1_slot: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    def get_smem_requirements(self) -> list[SmemAllocation]:
        """Return no data allocation; the pipeline barriers still use SMEM."""
        return []

    def get_tmem_requirements(self) -> list[TmemAllocation]:
        """Allocate the staged transformed K/V TMEM ring."""
        if self._alloc is None:
            self._alloc = TmemAllocation(
                name=f"{self.name}",
                num_columns=(
                    self.cfg.tmem_transformed_kv_stage_cols
                    * self.pipeline_config.num_stages
                ),
            )
        return [self._alloc]

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize the stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @cute.jit
    def _create_work_tile_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Initialize one work tile's stage-address task locals."""
        _ = context
        return {
            "kv_desc": Int32(0),
            "v_desc_0": Int32(0),
            "v_desc_1": Int32(0),
        }

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_load_state(self, stage_info: StageInfo) -> None:
        """Initialize producer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @consumer_work(work_attrs=WorkAttr.AUXILIARY)
    @cute.jit
    def init_descriptor_state(self, stage_info: StageInfo) -> None:
        """Initialize consumer-side TMEM stage state."""
        self._create_initial_task_locals(stage_info.context)

    @cute.jit
    def _source_stage(self) -> tuple[cutlass.Array, cutlass.Array]:
        """Return the raw stage's NVFP4 payload and scale arrays.

        Separate scale tensors stage an unpacking-TMA payload plus a scale
        ring; interleaved rows keep packed payload and scales in one stage.
        """
        cfg = self.cfg
        src_stage_idx = self.src_smem_kv.state_src.consumer_work_stage
        src_byte_base = self.src_smem_kv._smem_base_kv.subview(
            src_stage_idx * cfg.smem_kv_tile_elements
        )
        src_bytes = cutlass.Array(
            src_byte_base.data_ptr(),
            dtype=Uint8,
            shape=(cfg.smem_kv_tile_elements,),
            addrspace=3,
        )
        if cutlass.const_expr(cfg.use_nvfp4_interleaved_kv):
            return src_bytes, None
        return src_bytes, self.src_smem_kv._sf_stage_base(src_stage_idx)

    @cute.jit
    def interleaved_row_offset(self, token: Int32) -> Int32:
        """Byte offset of tile token ``token``'s interleaved row in a raw stage."""
        return self.cfg.nvfp4_interleaved_row_offset(token)

    @cute.jit
    def store_k_interleaved(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_chunk: Constexpr[int],
        num_chunks: Constexpr[int],
    ) -> None:
        """Convert one D128 slice of packed interleaved K rows into TMEM.

        Each thread owns one token row, which is also its TMEM lane, so packed
        bytes convert in place without unpacking loads: byte-selected E2M1x2
        pairs scale exactly to BF16x2 for BF16 Q, or round through E4M3 for
        FP8 Q. Each row stores sixteen columns at a time.
        """
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        token = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        src_bytes, _ = self._source_stage()
        row_addr = Int32(src_bytes.data_ptr().toint()) + self.interleaved_row_offset(
            token
        )
        payload_addr = row_addr + Int32(
            head_dim_stage_idx * (cfg.head_dim_kv_stage // 2)
        )
        scale_addr = (
            row_addr
            + Int32(cfg.headdim // 2)
            + Int32(head_dim_stage_idx * (cfg.head_dim_kv_stage // 16))
        )
        sf_lo_word, sf_hi_word = ld_shared_v2_b32(scale_addr)
        stage_addr = self._destination_stage_addr(stage_info)
        cols_per_chunk = 32 * cfg.qk_dtype_bytes // 4
        # Each 16-byte chunk holds 32 values: two scale blocks, 16 BF16 or
        # 8 FP8 columns.
        # FP8 output is short enough to unroll, overlapping each chunk's load
        # and conversion latency; BF16's longer body stays rolled (I-cache).
        for chunk_offset in cutlass.range(
            num_chunks, unroll=num_chunks if cfg.qk_dtype_bytes == 1 else 1
        ):
            chunk = Int32(first_chunk) + chunk_offset
            words = ld_shared_v4_b32(payload_addr + chunk * Int32(16))
            sf_word = cutlass.select_(chunk >= Int32(2), sf_hi_word, sf_lo_word)
            block_base = (chunk & Int32(1)) * Int32(2)
            dst_regs = []
            # Words 0/1 and 2/3 each share one 16-value scale block.
            block_scales = (
                broadcast_byte(sf_word, block_base),
                broadcast_byte(sf_word, block_base + Int32(1)),
            )
            if cutlass.const_expr(cfg.qk_dtype_bytes == 2):
                block_scales = (
                    e4m3x2_to_bf16x2(block_scales[0]),
                    e4m3x2_to_bf16x2(block_scales[1]),
                )
            for word_idx in cutlass.range_constexpr(4):
                sf_scalar = block_scales[word_idx // 2]
                if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                    # Even and odd nibbles scale separately (E4M3 rounding),
                    # then interleave back to natural order.
                    even = _mul_e2m1x4_e4m3x4(words[word_idx], sf_scalar)
                    odd = _mul_e2m1x4_e4m3x4(words[word_idx] >> Int32(4), sf_scalar)
                    dst_regs.append(byte_permute(even, odd, 0x5140))
                    dst_regs.append(byte_permute(even, odd, 0x7362))
                else:
                    dst_regs.extend(
                        scale_e2m1x8_packed_to_bf16x2x4(words[word_idx], sf_scalar)
                    )
            prims.tcgen05_st(
                "32x32b",
                prims.make_tmem_ptr(stage_addr + chunk * Int32(cols_per_chunk), Int32),
                cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
            )

    @cute.jit
    def store_v_interleaved(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert and transpose one D128 slice of packed interleaved V rows.

        Each warp owns 32 head-dimension rows (16 packed bytes per token). A
        transposing LDSM gives thread t byte column t/4 (rows 2c, 2c+1) and
        t/4+8 for four tokens. Instead of shuffling bytes into identity lane
        order, every 16-lane block stores row 2l at lane l and row 2l+1 at
        lane l+8 (l < 8): the low and high nibbles of the thread's own byte.
        Rows stay inside their 16-row scale block, PV carries the order into
        O, and the correction tail restores it once per output tile.
        """
        cfg = self.cfg
        assert has_direct_bf16_fp4_fp8_cvt() or cfg.pv_dtype_bytes == 1
        task_cache = _decode_gen_task_cache(stage_info)
        thread = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp = thread >> Int32(5)
        lane = thread & Int32(31)
        src_bytes, _ = self._source_stage()
        stage_base = Int32(src_bytes.data_ptr().toint())
        column_offset = Int32(
            head_dim_stage_idx * (cfg.head_dim_kv_stage // 2)
        ) + warp * Int32(16)
        # The 16 scale bytes of 16 rows form a 16x16 byte tile. Its transposed
        # LDSM gives byte column (block) b of four tokens to threads 4b..4b+3;
        # this warp's blocks are 2*warp and 2*warp+1 of the D128 slice.
        scale_src_a = warp * Int32(8) + (lane & Int32(3))
        scale_src_b = scale_src_a + Int32(4)
        stage_addr = self._destination_stage_addr(stage_info)
        cols_per_slice = 16 * cfg.pv_dtype_bytes // 4
        for slice_offset in cutlass.range(num_slices, unroll=num_slices):
            slice_idx = Int32(first_slice) + slice_offset
            slice_token = slice_idx * Int32(16)
            ldsm_row = self.interleaved_row_offset(slice_token + (lane & Int32(15)))
            payload = ldsm_m16n16_trans_b8(stage_base + ldsm_row + column_offset)
            scale_lo, scale_hi = ldsm_m16n16_trans_b8(
                stage_base + ldsm_row + Int32(cfg.headdim // 2)
            )
            # Scales of this thread's four tokens for its two blocks.
            scale_cols = scale_lo if head_dim_stage_idx == 0 else scale_hi
            block_scales = (
                cute.arch.shuffle_sync(scale_cols, scale_src_a),
                cute.arch.shuffle_sync(scale_cols, scale_src_b),
            )
            tmem_addr = stage_addr + slice_idx * Int32(cols_per_slice)
            for block in cutlass.range_constexpr(2):
                word = payload[block]
                dst_regs = []
                if cutlass.const_expr(cfg.pv_dtype_bytes == 1):
                    # Byte i is token i, matching the per-token scale word.
                    dst_regs.append(_mul_e2m1x4_e4m3x4(word, block_scales[block]))
                    dst_regs.append(
                        _mul_e2m1x4_e4m3x4(word >> Int32(4), block_scales[block])
                    )
                else:
                    dst_regs.extend(
                        scale_e2m1_token_quad_to_bf16x2x4(
                            word,
                            e4m3x2_to_bf16x2(block_scales[block]),
                            e4m3x2_to_bf16x2(block_scales[block] >> Int32(16)),
                        )
                    )
                prims.tcgen05_st(
                    "16x256b" if cfg.pv_dtype_bytes == 2 else "16x128b",
                    prims.make_tmem_ptr(
                        tmem_addr + (Int32(block * 16) << Int32(16)), Int32
                    ),
                    cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                )

    @cute.jit
    def _destination_stage_addr(self, stage_info: StageInfo) -> Int32:
        """Return the raw TMEM address of one transformed pipeline stage."""
        task_cache = _decode_gen_task_cache(stage_info)
        return (
            task_cache[_TASK_CACHE_TMEM_BASE_OFFSET]
            + Int32(self._alloc.offset)
            + stage_info.stage_idx * Int32(self.cfg.tmem_transformed_kv_stage_cols)
        )

    @cute.jit
    def _store_k(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Dequantize K and store token-major TMEM.

        Separate-scale stages use unpacking LDSM; interleaved rows convert
        packed bytes in place (see ``store_k_interleaved``).

        Each transform warp group covers all 128 TMEM lanes. With two groups,
        each converts half of the stage's eight 16-column slices.
        """
        if cutlass.const_expr(self.cfg.use_nvfp4_interleaved_kv):
            chunks = self.cfg.head_dim_kv_stage // 32
            if cutlass.const_expr(self.num_transform_groups() == 1):
                self.store_k_interleaved(stage_info, head_dim_stage_idx, 0, chunks)
            elif self.transform_group_idx() == Int32(0):
                self.store_k_interleaved(stage_info, head_dim_stage_idx, 0, chunks // 2)
            else:
                self.store_k_interleaved(
                    stage_info, head_dim_stage_idx, chunks // 2, chunks // 2
                )
        elif cutlass.const_expr(self.num_transform_groups() == 1):
            self.store_k_slices(stage_info, head_dim_stage_idx, 0, 8)
        else:
            if self.transform_group_idx() == Int32(0):
                self.store_k_slices(stage_info, head_dim_stage_idx, 0, 4)
            else:
                self.store_k_slices(stage_info, head_dim_stage_idx, 4, 4)
        cute.arch.fence_view_async_tmem_store()

    def num_transform_groups(self) -> int:
        """Return the number of complete warp groups in the transform task."""
        assert self.cfg.transform_kv_num_warps % 4 == 0
        return self.cfg.transform_kv_num_warps // 4

    @cute.jit
    def transform_group_idx(self) -> Int32:
        """Return this warp's warp-group rank inside the transform task."""
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        return (warp_idx - Int32(self.cfg.transform_kv_warp_idx)) // Int32(4)

    @cute.jit
    def ldsm_k_pair(self, src_bytes, thread_row, swizzle_mask, first) -> list:
        """Issue the four unpacking LDSMs of K slices ``first`` and ``first+1``.

        Returns eight words: (slice, row block, matrix) in row-major order.
        """
        head_dim_kv_stage = self.cfg.head_dim_kv_stage
        regs = []
        for col_offset in cutlass.range_constexpr(2):
            for row_block in cutlass.range_constexpr(2):
                local_offset = Int32(
                    row_block * 16 * head_dim_kv_stage + col_offset * 16
                ) + first * Int32(16)
                src_ptr = src_bytes.data_ptr() + (
                    thread_row * Int32(head_dim_kv_stage)
                    + (local_offset ^ swizzle_mask)
                )
                unpacked = prims.ldmatrix(
                    src_ptr,
                    2,
                    prims.MMALayout.ROW,
                    shape=prims.LoadShape.M8N16,
                    src_format=prims.LoadSrcFormat.B4X16_P64,
                )
                regs.append(Int32(unpacked[0]))
                regs.append(Int32(unpacked[1]))
        return regs

    @cute.jit
    def store_k_slices(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert K column slices ``[first_slice, first_slice + num_slices)``."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        # The cached index counts from the task's first warp group; each
        # transform group covers its own 128 TMEM lanes.
        transform_thread_idx = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_qwords = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int64,
            shape=(cfg.smem_kv_sf_tile_bytes // 8,),
            addrspace=3,
        )

        # Each lane owns four K rows. Staged heads load the complete D/16-byte
        # SF row, then select the qword matching the active 128-column slice.
        sf_rows = []
        sf_row_base = warp_idx * Int32(32) + lane_idx // Int32(4)
        for row_group in cutlass.range_constexpr(4):
            sf_row = sf_row_base + Int32(row_group * 8)
            sf_qword = src_sf_qwords[
                cfg.smem_kv_sf_row_offset(sf_row) // Int32(8)
                + Int32(head_dim_stage_idx)
            ]
            sf_rows.append((Int32(sf_qword), Int32(sf_qword >> Int64(32))))

        thread_row = warp_idx * Int32(32) + lane_idx
        swizzle_mask = (thread_row & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        # Sixteen K values form one column slice and share one scale byte.
        # FP8 packs them in four TMEM columns; BF16 packs them in eight.
        # A rolled loop keeps the transform small: its body is replicated at
        # every schedule site, and a fully unrolled stage thrashed the
        # instruction cache. Each iteration still issues four LDSMs first.
        cols_per_slice = 16 * cfg.qk_dtype_bytes // 4
        num_pairs = num_slices // 2
        # Software-pipeline the loop: issue the next pair's LDSMs before
        # converting the current pair so their latency overlaps the ALU work.
        # The final iteration reloads its own (still resident) pair.
        loaded = self.ldsm_k_pair(
            src_bytes, thread_row, swizzle_mask, Int32(first_slice)
        )
        for pair in cutlass.range(num_pairs, unroll=1):
            first = Int32(first_slice) + pair * Int32(2)
            current = loaded
            next_first = cutlass.select_(
                pair + Int32(1) < Int32(num_pairs), first + Int32(2), first
            )
            loaded = self.ldsm_k_pair(src_bytes, thread_row, swizzle_mask, next_first)
            for col_offset in cutlass.range_constexpr(2):
                col_stage = first + Int32(col_offset)
                use_hi_word = col_stage >= Int32(4)
                sf_byte_idx = col_stage & Int32(3)
                tmem_addr = stage_addr + col_stage * Int32(cols_per_slice)
                for row_block in cutlass.range_constexpr(2):
                    reg_idx = (col_offset * 2 + row_block) * 2
                    unpacked = (current[reg_idx], current[reg_idx + 1])
                    dst_regs = []
                    for half in cutlass.range_constexpr(2):
                        sf_lo_word, sf_hi_word = sf_rows[row_block * 2 + half]
                        sf_scalar = broadcast_byte(
                            cutlass.select_(use_hi_word, sf_hi_word, sf_lo_word),
                            sf_byte_idx,
                        )
                        if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                            dst_regs.append(
                                _mul_e2m1x4_e4m3x4(unpacked[half], sf_scalar)
                            )
                        else:
                            sf_pair = e4m3x2_to_bf16x2(sf_scalar)
                            dst_regs.extend(
                                scale_e2m1x4_unpacked_to_bf16x2x2(
                                    unpacked[half], sf_pair, sf_pair
                                )
                            )
                    lane_offset = Int32(row_block * 16) << Int32(16)
                    if cutlass.const_expr(cfg.qk_dtype_bytes == 1):
                        prims.tcgen05_st(
                            "16x128b",
                            prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                            cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                        )
                    else:
                        # 16x256b: registers 0/1 hold lane t/4 and 2/3 hold
                        # lane t/4 + 8, at BF16 columns 2*(t%4) and 2*(t%4)+1.
                        prims.tcgen05_st(
                            "16x256b",
                            prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                            cutlass.Vector.from_elements(
                                (dst_regs[0], dst_regs[1], dst_regs[2], dst_regs[3]),
                                dtype=Int32,
                            ),
                        )

    @cute.jit
    def _store_v(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Dequantize and transpose V with LDSM into TMEM operand A.

        With two transform warp groups, each converts half of the stage's
        eight 16-token slices.
        """
        if cutlass.const_expr(self.cfg.use_nvfp4_interleaved_kv):
            slices = self.cfg.tile_size_kv // 16
            if cutlass.const_expr(self.num_transform_groups() == 1):
                self.store_v_interleaved(stage_info, head_dim_stage_idx, 0, slices)
            elif self.transform_group_idx() == Int32(0):
                self.store_v_interleaved(stage_info, head_dim_stage_idx, 0, slices // 2)
            else:
                self.store_v_interleaved(
                    stage_info, head_dim_stage_idx, slices // 2, slices // 2
                )
        elif cutlass.const_expr(self.num_transform_groups() == 1):
            self.store_v_slices(stage_info, head_dim_stage_idx, 0, 8)
        else:
            if self.transform_group_idx() == Int32(0):
                self.store_v_slices(stage_info, head_dim_stage_idx, 0, 4)
            else:
                self.store_v_slices(stage_info, head_dim_stage_idx, 4, 4)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def store_v_slices(
        self,
        stage_info: StageInfo,
        head_dim_stage_idx: Constexpr[int],
        first_slice: Constexpr[int],
        num_slices: Constexpr[int],
    ) -> None:
        """Convert V token slices ``[first_slice, first_slice + num_slices)``."""
        cfg = self.cfg
        task_cache = _decode_gen_task_cache(stage_info)
        # The cached index counts from the task's first warp group; each
        # transform group covers its own 128 TMEM lanes.
        transform_thread_idx = (
            task_cache[_TASK_CACHE_WARP_GRP_THREAD_IDX]
            - Int32((cfg.transform_kv_warp_idx % 4) * 32)
        ) & Int32(127)
        warp_idx = transform_thread_idx >> Int32(5)
        lane_idx = transform_thread_idx & Int32(31)
        src_bytes, src_sf = self._source_stage()
        src_sf_words = cutlass.Array(
            src_sf.data_ptr(),
            dtype=Int32,
            shape=(cfg.smem_kv_sf_tile_bytes // 4,),
            addrspace=3,
        )

        swizzle_mask = (lane_idx & Int32(7)) * Int32(16)
        stage_addr = self._destination_stage_addr(stage_info)
        sf_head_stage_offset = head_dim_stage_idx * (cfg.head_dim_kv_stage // 16)
        # Sixteen tokens form one slice. Each unpacked word holds four
        # consecutive tokens of one head-dimension row, scaled per token. A
        # rolled two-slice loop keeps this replicated body instruction-cache
        # friendly; each iteration issues its LDSMs and scale loads first.
        cols_per_slice = 16 * cfg.pv_dtype_bytes // 4
        # Scale rows of a 16-token slice start at a fixed stride because page
        # fragments (four tokens or more) divide the slice. Hoist the address
        # math: the rolled loop must not touch the Python config object.
        assert 16 % cfg.num_tokens_per_page == 0 or cfg.num_tokens_per_page % 16 == 0
        sf_slice_stride = cfg.smem_kv_sf_row_offset(16)
        lane_sf_byte_offset = (
            cfg.smem_kv_sf_row_offset((lane_idx % Int32(4)) * Int32(4))
            + Int32(sf_head_stage_offset * 4)
            + warp_idx * Int32(8)
        )
        head_dim_kv_stage = cfg.head_dim_kv_stage
        for pair in cutlass.range(num_slices // 2, unroll=1):
            first = Int32(first_slice) + pair * Int32(2)
            unpacked_tiles = []
            sf_words = []
            for slice_offset in cutlass.range_constexpr(2):
                token_stage = first + Int32(slice_offset)
                # V scale factors use TRT-LLM's 4-token interleaved layout, so
                # the two packed scale vectors are adjacent 32-bit words.
                sf_byte_offset = (
                    token_stage * Int32(sf_slice_stride) + lane_sf_byte_offset
                )
                sf_word_idx = sf_byte_offset // Int32(4)
                sf_words.append(
                    (src_sf_words[sf_word_idx], src_sf_words[sf_word_idx + Int32(1)])
                )
                for col_block in cutlass.range_constexpr(2):
                    local_offset = (
                        token_stage * Int32(16 * head_dim_kv_stage)
                        + Int32(col_block * 16)
                        + warp_idx * Int32(32)
                    )
                    src_ptr = src_bytes.data_ptr() + (
                        lane_idx * Int32(head_dim_kv_stage)
                        + (local_offset ^ swizzle_mask)
                    )
                    unpacked_tiles.append(
                        prims.ldmatrix(
                            src_ptr,
                            2,
                            prims.MMALayout.COL,
                            shape=prims.LoadShape.M16N16,
                            src_format=prims.LoadSrcFormat.B4X16_P64,
                        )
                    )
            for slice_offset in cutlass.range_constexpr(2):
                tmem_addr = stage_addr + (first + Int32(slice_offset)) * Int32(
                    cols_per_slice
                )
                self.store_v_slice_blocks(
                    tmem_addr,
                    unpacked_tiles[slice_offset * 2 : slice_offset * 2 + 2],
                    sf_words[slice_offset],
                )

    @cute.jit
    def store_v_slice_blocks(self, tmem_addr, unpacked_tiles, sf_words) -> None:
        """Scale and store both 16-row head-dimension blocks of one V slice.

        Called inside a rolled runtime loop: do not bind Python-only values
        such as the config to local names here.
        """
        for col_block in cutlass.range_constexpr(2):
            unpacked = unpacked_tiles[col_block]
            packed_sf = sf_words[col_block]
            dst_regs = []
            if cutlass.const_expr(self.cfg.pv_dtype_bytes == 1):
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[0], packed_sf))
                dst_regs.append(_mul_e2m1x4_e4m3x4(unpacked[1], packed_sf))
            else:
                sf_lo = e4m3x2_to_bf16x2(packed_sf)
                sf_hi = e4m3x2_to_bf16x2(packed_sf >> Int32(16))
                for half in cutlass.range_constexpr(2):
                    dst_regs.extend(
                        scale_e2m1x4_unpacked_to_bf16x2x2(unpacked[half], sf_lo, sf_hi)
                    )
            lane_offset = Int32(col_block * 16) << Int32(16)
            if cutlass.const_expr(self.cfg.pv_dtype_bytes == 1):
                prims.tcgen05_st(
                    "16x128b",
                    prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                    cutlass.Vector.from_elements(tuple(dst_regs), dtype=Int32),
                )
            else:
                prims.tcgen05_st(
                    "16x256b",
                    prims.make_tmem_ptr(tmem_addr + lane_offset, Int32),
                    cutlass.Vector.from_elements(
                        (dst_regs[0], dst_regs[1], dst_regs[2], dst_regs[3]),
                        dtype=Int32,
                    ),
                )

    @producer_work
    @cute.jit
    def transform_k0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 0 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_k1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform K instance 1 into the acquired TMEM stage."""
        self._store_k(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v0(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 0 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @producer_work
    @cute.jit
    def transform_v1(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Transform V instance 1 into the acquired TMEM stage."""
        self._store_v(stage_info, head_dim_stage_idx)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=kv_desc_slot)
    @cute.jit
    def k_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed K TMEM address for QK instance 1."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_0_slot)
    @cute.jit
    def v_desc_0(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 0."""
        return self._destination_stage_addr(stage_info)

    @consumer_work(returns=v_desc_1_slot)
    @cute.jit
    def v_desc_1(self, stage_info: StageInfo) -> Int32:
        """Publish the transformed V TMEM address for VP MMA instance 1."""
        return self._destination_stage_addr(stage_info)


@dataclass(kw_only=True)
class SmemNvfp4RawKvResource(SmemKvResource):
    """Raw interleaved-NVFP4 ring of the NVFP4-fed Keeps profiles.

    TransformKvTask converts each stage into the FP8 or BF16 K/V rings. With
    register capture (the BF16 single-raw-stage profile), it first copies its
    share of the tile into registers (``capture_tile``) and releases the
    stage before waiting for the BF16 destination, so the loader refills it
    while the transform writes. Each thread then holds one (token,
    64-channel slab) unit per D128 head-dim stage: eight payload words and
    one word of four block scales.
    """

    # FmhaDecodeConfig.uses_nvfp4_raw_register_capture of the profile; this
    # ring's own cfg is the raw NVFP4 view, which cannot tell.
    uses_register_capture: Constexpr[bool] = False
    captured_words: cutlass.Array = None
    captured_scales: cutlass.Array = None

    @property
    def captured_units(self) -> int:
        """(token, slab) units per thread across the tile's head-dim stages."""
        cfg = self.cfg
        return (2 * cfg.tile_size_kv * cfg.num_head_dim_stages_kv) // (
            cfg.transform_kv_num_warps * 32
        )

    def _init_placeholder_state(self) -> None:
        """Add the capture registers to the raw ring's placeholder state."""
        super()._init_placeholder_state()
        if self.uses_register_capture:
            self.captured_words = _placeholder_local_array(
                Int32, 8 * self.captured_units
            )
            self.captured_scales = _placeholder_local_array(Int32, self.captured_units)

    @cute.jit
    def _create_initial_task_locals(
        self, context: ResourceContext | None = None
    ) -> ResourceVars:
        """Bind the raw ring, then allocate the capture registers."""
        task_locals = SmemKvResource._create_initial_task_locals(self, context)
        if cutlass.const_expr(self.uses_register_capture):
            self.captured_words = cutlass.Array(
                Int32, 8 * self.captured_units, space=cutlass.AddressSpace.rmem
            )
            self.captured_scales = cutlass.Array(
                Int32, self.captured_units, space=cutlass.AddressSpace.rmem
            )
        return task_locals

    @consumer_work
    @cute.jit
    def capture_tile(self, stage_info: StageInfo) -> None:
        """Copy this thread's units of the raw tile into registers.

        Every raw SMEM read finishes here, so the stage can be released
        before the transform waits for its destination; the registers stay
        owned by the transform until its last destination commit. Unit
        ``idx`` covers head-dim stage ``idx * threads // (2 * tile_size_kv)``
        in the order ``SmemKvFromNvfp4Resource.bf16_from_raw`` reads them.
        """
        assert self.uses_register_capture
        cfg = self.cfg
        num_threads = cfg.transform_kv_num_warps * 32
        units_per_stage = 2 * cfg.tile_size_kv
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.transform_kv_warp_idx
        )
        thread = warp * Int32(32) + cute.arch.lane_idx()
        stage_base = Int32(self._smem_base_kv.data_ptr().toint()) + (
            stage_info.stage_idx * Int32(cfg.smem_kv_tile_elements)
        )
        for idx in cutlass.range_constexpr(self.captured_units):
            head_dim_stage_idx = idx * num_threads // units_per_stage
            unit = thread + Int32(idx * num_threads % units_per_stage)
            token = unit % Int32(cfg.tile_size_kv)
            slab = unit // Int32(cfg.tile_size_kv)
            row = stage_base + cfg.nvfp4_interleaved_row_offset(token)
            payload = row + Int32(64 * head_dim_stage_idx) + slab * Int32(32)
            for chunk in cutlass.range_constexpr(2):
                words = ld_shared_v4_b32(payload + Int32(chunk * 16))
                for word_idx in cutlass.range_constexpr(4):
                    self.captured_words[8 * idx + 4 * chunk + word_idx] = words[
                        word_idx
                    ]
            self.captured_scales[idx] = ld_shared_b32(
                row + Int32(cfg.headdim // 2 + 8 * head_dim_stage_idx) + slab * Int32(4)
            )


@dataclass(kw_only=True)
class SmemKvFromNvfp4Resource(SmemKvTileResource):
    """FP8 or BF16 K or V tile ring written by TransformKvTask from NVFP4.

    MMA consumes it exactly like a TMA-loaded ``SmemKvTileResource`` of the
    matching KeepsMmaAb profile: 128-row slabs of 128-byte SW128 rows,
    token-major for both K (K-major B) and V (MN-major B), so neither operand
    needs a transpose. An FP8 D256 stage holds two 128-channel slabs; a BF16
    D128 head-dim stage holds two 64-channel slabs, so one raw tile fills two
    BF16 stages. ``raw_src`` is the raw interleaved ring whose current
    consumer stage the transform converts.
    """

    raw_src: Constexpr[SmemKvResource | None] = None

    @producer_work
    @cute.jit
    def transform_from_raw(
        self, stage_info: StageInfo, *, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Convert the raw ring's current tile into this ring's stage.

        ``head_dim_stage_idx`` selects the BF16 D128 stage; FP8 has one.
        """
        if cutlass.const_expr(self.cfg.k_dtype_bytes == 2):
            self.bf16_from_raw(stage_info, head_dim_stage_idx)
        else:
            assert head_dim_stage_idx == 0
            self.fp8_from_raw(stage_info)

    @cute.jit
    def raw_row_addr(self, token: Int32) -> Int32:
        """Shared address of ``token``'s interleaved row in the raw stage."""
        raw_cfg = self.raw_src.cfg
        src_stage = self.raw_src.state_src.consumer_work_stage
        stage_base = Int32(self.raw_src._smem_base_kv.data_ptr().toint()) + (
            src_stage * Int32(raw_cfg.smem_kv_tile_elements)
        )
        return stage_base + raw_cfg.nvfp4_interleaved_row_offset(token)

    @cute.jit
    def bf16_from_raw(
        self, stage_info: StageInfo, head_dim_stage_idx: Constexpr[int]
    ) -> None:
        """Convert one 128-channel slice of the raw tile into a BF16 stage.

        Each thread owns (token, 64-channel slab) rows: 32 payload bytes and
        their four block scales become one 128-byte SW128 row. E2M1 x E4M3
        products are exact in BF16, so this matches the dequantized cache.
        With register capture the bytes come from ``raw_src``'s registers
        (see ``SmemNvfp4RawKvResource.capture_tile``) instead of its stage.
        """
        cfg = self.cfg
        assert cfg.k_dtype_bytes == 2 and cfg.v_dtype_bytes == 2
        assert cfg.headdim == 256 and cfg.head_dim_kv_stage == 128
        assert cfg.tile_size_kv == 128
        num_threads = cfg.transform_kv_num_warps * 32
        units = 2 * cfg.tile_size_kv
        assert units % num_threads == 0
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.transform_kv_warp_idx
        )
        thread = warp * Int32(32) + cute.arch.lane_idx()
        dst_base = Int32(self._stage_base(stage_info).data_ptr().toint())
        slab_bytes = cfg.tile_size_kv * 128
        captured = self.raw_src.uses_register_capture
        for rep in cutlass.range_constexpr(units // num_threads):
            unit = thread + Int32(rep * num_threads)
            token = unit % Int32(cfg.tile_size_kv)
            slab = unit // Int32(cfg.tile_size_kv)
            # This unit's capture index (see capture_tile).
            idx = (head_dim_stage_idx * units + rep * num_threads) // num_threads
            if cutlass.const_expr(captured):
                sf_word = self.raw_src.captured_scales[idx]
            else:
                row = self.raw_row_addr(token)
                # Channels 128 * stage + 64 * slab: 32 payload bytes, 4 scales.
                payload = row + Int32(64 * head_dim_stage_idx) + slab * Int32(32)
                sf_word = ld_shared_b32(
                    row
                    + Int32(cfg.headdim // 2 + 8 * head_dim_stage_idx)
                    + slab * Int32(4)
                )
            # One BF16x2 pair per 16-channel scale block.
            sf_pairs = (
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 0)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 1)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 2)),
                e4m3x2_to_bf16x2(broadcast_byte_const(sf_word, 3)),
            )
            dst_row = dst_base + slab * Int32(slab_bytes) + token * Int32(128)
            # SW128: 16-byte segment s of row r lands at segment s ^ (r % 8).
            swizzle = token % Int32(8)
            for chunk in cutlass.range_constexpr(2):
                if cutlass.const_expr(captured):
                    base = 8 * idx + 4 * chunk
                    words = (
                        self.raw_src.captured_words[base],
                        self.raw_src.captured_words[base + 1],
                        self.raw_src.captured_words[base + 2],
                        self.raw_src.captured_words[base + 3],
                    )
                else:
                    words = ld_shared_v4_b32(payload + Int32(chunk * 16))
                for word_idx in cutlass.range_constexpr(4):
                    # Eight values per word, sixteen per scale block; each
                    # word becomes one 16-byte BF16 segment in natural order.
                    out = scale_e2m1x8_packed_to_bf16x2x4(
                        words[word_idx], sf_pairs[(4 * chunk + word_idx) // 2]
                    )
                    seg_idx = Int32(4 * chunk + word_idx)
                    st_shared_v4_b32(
                        dst_row + ((seg_idx ^ swizzle) << Int32(4)),
                        out[0],
                        out[1],
                        out[2],
                        out[3],
                    )
        # Generic-proxy SMEM writes must be visible to the async-proxy MMA.
        cute.arch.fence_view_async_shared()

    @cute.jit
    def fp8_from_raw(self, stage_info: StageInfo) -> None:
        """Convert the raw ring's current tile into this ring's E4M3 stage.

        Each thread owns (token, head-dim half) rows: 64 payload bytes and
        their eight block scales become one 128-byte SW128 row. Even and odd
        nibbles are scaled separately (rounding through E4M3, as the FP8-MMA
        variant does everywhere) and interleaved back to natural order.
        """
        cfg = self.cfg
        assert cfg.k_dtype_bytes == 1 and cfg.v_dtype_bytes == 1
        assert cfg.headdim == 256 and cfg.head_dim_kv_stage == 256
        assert cfg.tile_size_kv == 128
        num_threads = cfg.transform_kv_num_warps * 32
        units = 2 * cfg.tile_size_kv
        assert units % num_threads == 0
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx()) - Int32(
            cfg.transform_kv_warp_idx
        )
        thread = warp * Int32(32) + cute.arch.lane_idx()
        dst_base = Int32(self._stage_base(stage_info).data_ptr().toint())
        # One head-dim half: 128 values = 64 payload bytes and 8 scale bytes;
        # its FP8 slab holds tile_size_kv rows of 128 bytes.
        half_payload_bytes = cfg.headdim // 4
        half_scale_bytes = cfg.headdim // 32
        slab_bytes = cfg.tile_size_kv * 128
        for rep in cutlass.range_constexpr(units // num_threads):
            unit = thread + Int32(rep * num_threads)
            token = unit % Int32(cfg.tile_size_kv)
            half = unit // Int32(cfg.tile_size_kv)
            row = self.raw_row_addr(token)
            payload = row + half * Int32(half_payload_bytes)
            sf_lo, sf_hi = ld_shared_v2_b32(
                row + Int32(cfg.headdim // 2) + half * Int32(half_scale_bytes)
            )
            dst_row = dst_base + half * Int32(slab_bytes) + token * Int32(128)
            # SW128: 16-byte segment s of row r lands at segment s ^ (r % 8).
            swizzle = token % Int32(8)
            # Each 16-byte payload chunk holds 32 values: two scale blocks.
            for chunk in cutlass.range_constexpr(4):
                words = ld_shared_v4_b32(payload + Int32(chunk * 16))
                sf_word = sf_lo if chunk < 2 else sf_hi
                scales = (
                    broadcast_byte_const(sf_word, (2 * chunk) % 4),
                    broadcast_byte_const(sf_word, (2 * chunk + 1) % 4),
                )
                out = []
                for word_idx in cutlass.range_constexpr(4):
                    sf = scales[word_idx // 2]
                    # Low nibbles are the even head-dim values, high nibbles
                    # the odd ones; interleave the two product words back.
                    even = _mul_e2m1x4_e4m3x4(words[word_idx], sf)
                    odd = _mul_e2m1x4_e4m3x4(words[word_idx] >> Int32(4), sf)
                    out.append(byte_permute(even, odd, 0x5140))
                    out.append(byte_permute(even, odd, 0x7362))
                for seg in cutlass.range_constexpr(2):
                    seg_idx = Int32(chunk * 2 + seg)
                    st_shared_v4_b32(
                        dst_row + ((seg_idx ^ swizzle) << Int32(4)),
                        out[4 * seg],
                        out[4 * seg + 1],
                        out[4 * seg + 2],
                        out[4 * seg + 3],
                    )
        # Generic-proxy SMEM writes must be visible to the async-proxy MMA.
        cute.arch.fence_view_async_shared()
