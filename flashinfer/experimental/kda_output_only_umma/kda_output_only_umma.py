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

---------------------------------------------------------------------------
KDA output-only (frozen-state) decode on tcgen05 / UMMA / TMEM (SM10x): the
``backend="umma"`` arm of ``flashinfer.kda_kernels.kda_decode_wy_output_only``
(``kda_wy_output_only`` and the vLLM ``kda_recoverssm_verify`` drop-in).

The ReplaySSM UMMA kernel (KDA ucache flush) with an empty history: the state is
read, never written, and there is no ring. Per (sequence b, value head hv), with
T <= 16 tokens, per-channel cumulative log-decay cum_t and k̂ / q̂ the L2-normalized
key / (scaled) query of the query head i_h = hv / (HV / H):

    khat_t = k̂_t e^{cum_t},   ktil_s = k̂_s e^{-cum_s},   qhat_t = q̂_t e^{cum_t}
    u_t    = beta_t (v_t - S0 khat_t - sum_{s<t} (khat_t . ktil_s) u_s)
    y_t    = S0 qhat_t + sum_{s<=t} (qhat_t . ktil_s) u_s

u_t is also the RecoverSSM correction (emit mode).

Work item = (sequence b, value head hv), item = b * HV + hv; 1-D grid of 160-thread
CTAs, one item per CTA (a constant one-trip loop) or (``persistent``) min(items,
SMs * CTAs/SM) CTAs striding over the items (removes the ~1 us per-CTA launch gap;
picked per T / mode on the host; ``prefetch_next`` loads the next item's q / k / gate /
beta words into registers during the current item). Per item:
  warp 0   : mbarrier inits + TMEM allocation (32 columns) at entry; TMA of the bf16
             state (two 16 KB SW128 K halves), v-row bulk copies, the UMMA issue.
  warps 1-4: consumers: row-layout L2 norm (q / k and the gate rows as 16-B chunks,
             gate rows transposed through SMEM), channel-layout gate (X_r / Ktil
             tiles), the mma.sync Gram X_r . Ktil^T written straight into the
             coefficient tables, the per-V-row forward substitution epilogue (TMEM
             lane = V row) in f32x2 fmas (each lane one token's chain, bitwise equal
             to the scalar recurrence).
UMMA: readout R = S0 . X_r^T, M128 N{16, 32} K128 (kind::f16, BF16 A / B, fp32 TMEM);
X_r = [khat; qhat] (2T rows, zero rows after), so R holds S0 khat_t in column t and
S0 qhat_t in column T + t.

The kernel takes ``KdaDecodeWyOutputOnlyKernel``'s argument list (dense
[B, T, ...] tensors, or the vLLM drop-in's packed tokens with runtime token
strides, ragged lengths, null slots and slot-indexed caches).
"""

from __future__ import annotations

from typing import Optional

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int32, Int64, Uint32, Uint64
from cutlass._mlir.dialects import llvm, nvvm
from cutlass.cute.nvgpu import cpasync
from cutlass.cutlass_dsl import T as _T
from cutlass.cutlass_dsl import dsl_user_op

from flashinfer.gdn_kernels import _umma_helpers as U

K_DIM = 128
V_DIM = 128
XC_PITCH = 272  # bytes per row of the row-major X_c / Ktil tiles (136 bf16)
LOG2_E = 1.4426950408889634
_EPS = 1e-6  # inside the L2-norm rsqrt, as the WY kernel
_TMEM_COLS = 32

GATE_PRECOMPUTED = 0
GATE_LOWER_BOUND = 1
GATE_SOFTPLUS = 2


def _bf16_idesc(m: int, n: int):
    """kind::f16 instruction descriptor with BF16 A and B (bits 7 and 10), fp32 D."""
    return U.f16_idesc(m, n) | Uint32((1 << 7) | (1 << 10))


def _sel(cond, a, b, ty):
    """Branch-free typed select."""
    return ty(cutlass.select_(cond, ty(a), ty(b)))


@dsl_user_op
def _ldmatrix_x4(addr: Int32, *, loc=None, ip=None):
    """ldmatrix .x4 (m16n8k16 A fragment of a row-major tile)."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x4.m8n8.shared.b16 {$0,$1,$2,$3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _ldmatrix_x2(addr: Int32, *, loc=None, ip=None):
    """ldmatrix .x2 (m16n8k16 B fragment of an [n][k] row-major tile)."""
    st = llvm.StructType.get_literal([_T.i32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.x2.m8n8.shared.b16 {$0,$1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _ex2(x: Float32, *, loc=None, ip=None) -> Float32:
    """ex2.approx.ftz.f32 (the WY kernel's _exp2_approx_f32)."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "ex2.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _rsqrt_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """rsqrt.approx.ftz.f32."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(x).ir_value(loc=loc, ip=ip)],
        "rsqrt.approx.ftz.f32 $0, $1;",
        "=f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _bar_sync_ua(barrier_id: int, number_of_threads: int, *, loc=None, ip=None) -> None:
    """Named-barrier sync without ``.aligned`` (multi-site barriers; synccheck)."""
    nvvm.barrier_cta_sync(
        Int32(barrier_id).ir_value(loc=loc, ip=ip),
        thread_count=Int32(number_of_threads).ir_value(loc=loc, ip=ip),
        aligned=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _ldg_u16_if(pred: Int32, addr: Int64, *, loc=None, ip=None) -> Uint32:
    """Predicated ld.global.u16 (zero-extended); 0 where pred == 0."""
    r = llvm.inline_asm(
        _T.i32(),
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\t.reg .b16 h;\n\tmov.u32 $0, 0;\n\t"
        "setp.ne.s32 p, $1, 0;\n\t@p ld.global.u16 h, [$2];\n\t"
        "@p cvt.u32.u16 $0, h;\n\t}",
        "=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Uint32(r)


@dsl_user_op
def _ldg_v4_or0(pred: Int32, addr: Int64, *, loc=None, ip=None):
    """Predicated ld.global.v4.b32: four words, 0 where pred == 0."""
    st = llvm.StructType.get_literal([_T.i32()] * 4)
    r = llvm.inline_asm(
        st,
        [Int32(pred).ir_value(loc=loc, ip=ip), Int64(addr).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\n\tmov.b32 $0, 0;\n\tmov.b32 $1, 0;\n\tmov.b32 $2, 0;\n\t"
        "mov.b32 $3, 0;\n\tsetp.ne.s32 p, $4, 0;\n\t"
        "@p ld.global.v4.b32 {$0,$1,$2,$3}, [$5];\n\t}",
        "=r,=r,=r,=r,r,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Uint32(llvm.extractvalue(_T.i32(), r, [i], loc=loc, ip=ip)) for i in range(4)
    ]


@dsl_user_op
def _stg_u16(addr: Int64, w: Uint32, *, loc=None, ip=None) -> None:
    """st.global.u16 of the low 16 bits of ``w``."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Uint32(w).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.u16.u32 h, $1;\n\tst.global.u16 [$0], h;\n\t}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_f32(addr: Int64, x: Float32, *, loc=None, ip=None) -> None:
    """st.global.f32."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(x).ir_value(loc=loc, ip=ip)],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _bulk_g2s_if(
    pred: Int32,
    dst_smem: Int32,
    src_gmem: Int64,
    nbytes: Int32,
    mbar_smem: Int32,
    *,
    loc=None,
    ip=None,
):
    """Predicated cp.async.bulk global -> shared (complete_tx on ``mbar_smem``)."""
    llvm.inline_asm(
        _T.i32(),
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(dst_smem).ir_value(loc=loc, ip=ip),
            Int64(src_gmem).ir_value(loc=loc, ip=ip),
            Int32(nbytes).ir_value(loc=loc, ip=ip),
            Int32(mbar_smem).ir_value(loc=loc, ip=ip),
        ],
        "{\n\t.reg .pred p;\n\tsetp.ne.s32 p, $1, 0;\n\t"
        "@p cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [$2], [$3], $4, [$5];\n\tmov.u32 $0, 0;\n\t}\n",
        "=r,r,r,l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _sts_bf16(addr: Int32, x: Float32, *, loc=None, ip=None) -> None:
    """st.shared.b16 of ``x`` rounded to bf16 (RN)."""
    llvm.inline_asm(
        None,
        [Int32(addr).ir_value(loc=loc, ip=ip), Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.shared.b16 [$0], h;\n\t}",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _stg_bf16(addr: Int64, x: Float32, *, loc=None, ip=None) -> None:
    """st.global.b16 of ``x`` rounded to bf16 (RN)."""
    llvm.inline_asm(
        None,
        [Int64(addr).ir_value(loc=loc, ip=ip), Float32(x).ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .b16 h;\n\tcvt.rn.bf16.f32 h, $1;\n\tst.global.b16 [$0], h;\n\t}",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _pack2(lo: Float32, hi: Float32, *, loc=None, ip=None) -> Int64:
    """Two fp32 as one .b64 f32x2 operand (``lo`` in the low word)."""
    r = llvm.inline_asm(
        _T.i64(),
        [Float32(lo).ir_value(loc=loc, ip=ip), Float32(hi).ir_value(loc=loc, ip=ip)],
        "mov.b64 $0, {$1, $2};",
        "=l,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


@dsl_user_op
def _unpack2(x: Int64, *, loc=None, ip=None):
    """The two fp32 words of a .b64 f32x2 value (low, high)."""
    st = llvm.StructType.get_literal([_T.f32()] * 2)
    r = llvm.inline_asm(
        st,
        [Int64(x).ir_value(loc=loc, ip=ip)],
        "mov.b64 {$0, $1}, $2;",
        "=f,=f,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Float32(llvm.extractvalue(_T.f32(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _ffma2(a: Int64, b: Int64, c: Int64, *, loc=None, ip=None) -> Int64:
    """fma.rn.f32x2: two independent fp32 fmas (each rounded once, as fma.rn.f32)."""
    r = llvm.inline_asm(
        _T.i64(),
        [Int64(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
        "fma.rn.f32x2 $0, $1, $2, $3;",
        "=l,l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


@dsl_user_op
def _fmul2(a: Int64, b: Int64, *, loc=None, ip=None) -> Int64:
    """mul.rn.f32x2."""
    r = llvm.inline_asm(
        _T.i64(),
        [Int64(v).ir_value(loc=loc, ip=ip) for v in (a, b)],
        "mul.rn.f32x2 $0, $1, $2;",
        "=l,l,l",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


@dsl_user_op
def _ffma(a: Float32, b: Float32, c: Float32, *, loc=None, ip=None) -> Float32:
    """fma.rn.f32."""
    r = llvm.inline_asm(
        _T.f32(),
        [Float32(v).ir_value(loc=loc, ip=ip) for v in (a, b, c)],
        "fma.rn.f32 $0, $1, $2, $3;",
        "=f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Float32(r)


@dsl_user_op
def _lds_b64x2(addr: Int32, *, loc=None, ip=None):
    """ld.shared.v2.b64 (16 B aligned): two f32x2 operands."""
    st = llvm.StructType.get_literal([_T.i64()] * 2)
    r = llvm.inline_asm(
        st,
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.b64 {$0,$1}, [$2];",
        "=l,=l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return [
        Int64(llvm.extractvalue(_T.i64(), r, [i], loc=loc, ip=ip)) for i in range(2)
    ]


@dsl_user_op
def _lds_b64(addr: Int32, *, loc=None, ip=None) -> Int64:
    """ld.shared.b64 (8 B aligned): one f32x2 operand."""
    r = llvm.inline_asm(
        _T.i64(),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.b64 $0, [$1];",
        "=l,r",
        has_side_effects=True,
        is_align_stack=False,
        loc=loc,
        ip=ip,
    )
    return Int64(r)


def _x_sw128_off(row, c, half_bytes):
    """Byte offset of element (row, k = c) in a K-major SW128 operand tile: two
    64-element K halves ``half_bytes`` apart, 128 B rows, 16 B chunks XOR-swizzled
    with (row % 8)."""
    return (
        (c // 64) * half_bytes
        + row * 128
        + ((((c % 64) // 8) ^ (row % 8)) * 16)
        + (c % 8) * 2
    )


# SMEM: 32 KB state + X_r + v rows + Ktil + Gram + tables, ~43 KB at T <= 8 (5 CTAs
# per SM) and ~54 KB above (4)
def ctas_per_sm(T: int) -> int:
    """Launch bound (CTAs per SM) of the build for T tokens."""
    return 5 if T <= 8 else 4


class KdaOutputOnlyUmma:
    """UMMA output-only KDA decode; the argument list of ``KdaDecodeWyOutputOnlyKernel``."""

    def __init__(
        self,
        T: int,
        gate_mode: int = GATE_PRECOMPUTED,
        has_dt_bias: bool = False,
        beta_is_logit: bool = False,
        emit_corrections: bool = False,
        vllm_dropin: bool = False,
        null_min: int = 1,
        kg_v4: bool = True,
        min_blocks: Optional[int] = None,
        g_v4: bool = True,
        hv_heads: Optional[int] = None,
        q_heads: Optional[int] = None,
        num_sms: Optional[int] = None,
        persistent: bool = False,
        prefetch_next: bool = True,
    ):
        assert 1 <= T <= 16, T
        self.T = T
        self.gate_mode = int(gate_mode)
        self.has_dt_bias = bool(has_dt_bias)
        self.beta_is_logit = bool(beta_is_logit)
        self.dropin = bool(vllm_dropin)
        self.emit = bool(emit_corrections) or self.dropin
        self.null_min = int(null_min)
        # drop-in kg cache rows 16-B aligned (every kg stride a multiple of 8
        # elements): 16-B stores of the key half, else 2-B stores (separate build)
        self.kg_v4 = bool(kg_v4)
        # gate rows read as 16-B row chunks (token stride a multiple of 8 elements)
        # and transposed through SMEM, else per-thread scalar loads (separate build)
        self.g_v4 = bool(g_v4)
        # head counts baked in (the launch caches per (HV, H)): constant index math
        self.hv_heads = hv_heads
        self.q_heads = q_heads
        # persistent: min(items, num_sms * min_blocks) CTAs striding over the items,
        # else one item per CTA (same kernel body, every CTA runs one iteration)
        self.persistent = bool(persistent)
        self.num_sms = num_sms
        assert not self.persistent or num_sms is not None
        self.NX = 16 if T <= 8 else 32  # X_r / X_c rows (2T live)
        self.KT = 8 if T <= 8 else 16  # Ktil rows (T live)
        self.num_threads = 160  # warp 0: TMA / UMMA / TMEM; warps 1-4: consumers
        self.min_blocks = ctas_per_sm(T) if min_blocks is None else int(min_blocks)
        # persistent: prefetch the next item's row inputs during this item's work
        self.prefetch_next = bool(prefetch_next)

    @cute.jit
    def __call__(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gBeta: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0: cute.Tensor,
        gH0idx: cute.Tensor,
        gOut: cute.Tensor,
        gCorr: cute.Tensor,
        gKg: cute.Tensor,
        gQsl: cute.Tensor,
        scale: Float32,
        lower_bound: Float32,
        s_q_tok: Int32,
        s_k_tok: Int32,
        s_v_tok: Int32,
        s_g_tok: Int32,
        s_b_tok: Int32,
        s_o_tok: Int32,
        s_c_blk: Int32,
        s_c_head: Int32,
        s_c_pos: Int32,
        s_kg_blk: Int32,
        s_kg_head: Int32,
        s_kg_pos: Int32,
        s_h0_blk: Int64,
        HV: Int32,
        V_DIM_: Int32,
        H: Int32,
        stream,
    ):
        B_val = gH0idx.layout.shape[0]
        # pool rebuilt with static inner strides and the runtime (possibly padded) block
        # stride (the WY kernel's scheme); TMA box = (V rows, 64 K) per half
        blk = cute.assume(s_h0_blk, divby=8)
        state = cute.make_tensor(
            gH0.iterator,
            cute.make_layout(
                (gH0.layout.shape[0], HV, V_DIM, K_DIM),
                stride=(blk, V_DIM * K_DIM, K_DIM, 1),
            ),
        )
        elems = 64
        smem_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (1, 1, V_DIM, (elems, 1), 1),
                stride=(0, 0, elems, (1, V_DIM * elems), V_DIM * elems),
            ),
        )
        tma_ld = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            cute.logical_divide(state, (None, None, None, elems)),
            smem_layout,
            cta_tiler=(1, 1, V_DIM, elems),
        )
        n_items = HV * B_val
        grid_x = n_items
        if cutlass.const_expr(self.persistent):
            cap = Int32(self.num_sms * self.min_blocks)
            grid_x = _sel(n_items < cap, n_items, cap, Int32)
        self.kernel(
            gQ,
            gK,
            gV,
            gG,
            gBeta,
            gAlog,
            gDtbias,
            gH0idx,
            gOut,
            gCorr,
            gKg,
            gQsl,
            scale,
            lower_bound,
            s_q_tok,
            s_k_tok,
            s_v_tok,
            s_g_tok,
            s_b_tok,
            s_o_tok,
            s_c_blk,
            s_c_head,
            s_c_pos,
            s_kg_blk,
            s_kg_head,
            s_kg_pos,
            HV,
            H,
            tma_ld,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(self.num_threads, 1, 1),
            min_blocks_per_mp=self.min_blocks,
            stream=stream,
        )

    @cute.jit
    def _load_inputs(self, item, ctx):
        """Global loads of one work item's row inputs for this consumer thread, as a
        flat list: the row-layout k / q words (X row r8 + 8 p, 8 k values at l16 * 8;
        dead rows and tokens past the sequence length read as 0), the raw gate of
        channel ct for every token, and the beta of token ct."""
        T = self.T
        T2 = 2 * T
        NP = (T2 + 7) // 8
        PARTIAL = T2 % 8 != 0
        HV = ctx["HV"]
        hv = item % HV
        brow = item // HV
        i_h = hv // ctx["GRP"]
        r8 = ctx["r8"]
        l16 = ctx["l16"]
        ct = ctx["ct"]
        bos = Int32(0)
        seq_len = Int32(T)
        if cutlass.const_expr(self.dropin):
            bos = Int32(ctx["qsl"][brow])
            seq_len = Int32(ctx["qsl"][brow + 1]) - bos
        tok0 = bos if self.dropin else brow * T
        vals = []
        for p in cutlass.range_constexpr(NP):
            xr = r8 + 8 * p
            is_q = xr >= T
            t_k = _sel(is_q, 0, xr, Int32)
            t_q = _sel(is_q, xr - T, 0, Int32)
            if cutlass.const_expr(PARTIAL and p == NP - 1):
                t_q = _sel(xr >= T2, 0, t_q, Int32)
            tt_ = _sel(is_q, t_q, t_k, Int32)
            live = Int32(tt_ < seq_len)
            if cutlass.const_expr(PARTIAL and p == NP - 1):
                live = Int32((tt_ < seq_len) & (xr < T2))
            ak = ctx["k"] + (
                Int64(tok0 + t_k) * Int64(ctx["sk"]) + Int64(i_h * K_DIM + l16 * 8)
            ) * Int64(2)
            aq = ctx["q"] + (
                Int64(tok0 + t_q) * Int64(ctx["sq"]) + Int64(i_h * K_DIM + l16 * 8)
            ) * Int64(2)
            vals += _ldg_v4_or0(live, _sel(is_q, aq, ak, Int64))
        if cutlass.const_expr(self.g_v4):
            # gate rows t = r8 + 8 i, 8 channels at l16 * 8 (row layout)
            for i in cutlass.range_constexpr((T + 7) // 8):
                tg = r8 + 8 * i
                vals += _ldg_v4_or0(
                    Int32((tg < T) & (tg < seq_len)),
                    ctx["g"]
                    + (
                        Int64(tok0 + tg) * Int64(ctx["sg"])
                        + Int64(hv * K_DIM + l16 * 8)
                    )
                    * Int64(2),
                )
        for t in cutlass.range_constexpr(0 if self.g_v4 else T):
            vals.append(
                _ldg_u16_if(
                    Int32(t < seq_len),
                    ctx["g"]
                    + (Int64(tok0 + t) * Int64(ctx["sg"]) + Int64(hv * K_DIM + ct))
                    * Int64(2),
                )
            )
        tb = _sel(ct < T, ct, 0, Int32)
        vals.append(
            _ldg_u16_if(
                Int32((ct < T) & (tb < seq_len)),
                ctx["b"] + (Int64(tok0 + tb) * Int64(ctx["sb"]) + Int64(hv)) * Int64(2),
            )
        )
        return vals

    @cute.kernel
    def kernel(
        self,
        gQ: cute.Tensor,
        gK: cute.Tensor,
        gV: cute.Tensor,
        gG: cute.Tensor,
        gBeta: cute.Tensor,
        gAlog: cute.Tensor,
        gDtbias: cute.Tensor,
        gH0idx: cute.Tensor,
        gOut: cute.Tensor,
        gCorr: cute.Tensor,
        gKg: cute.Tensor,
        gQsl: cute.Tensor,
        scale: Float32,
        lower_bound: Float32,
        s_q_tok: Int32,
        s_k_tok: Int32,
        s_v_tok: Int32,
        s_g_tok: Int32,
        s_b_tok: Int32,
        s_o_tok: Int32,
        s_c_blk: Int32,
        s_c_head: Int32,
        s_c_pos: Int32,
        s_kg_blk: Int32,
        s_kg_head: Int32,
        s_kg_pos: Int32,
        HV: Int32,
        H: Int32,
        tma_ld: cpasync.TmaInfo,
    ):
        T = self.T
        T2 = 2 * T
        NX = self.NX
        KT = self.KT
        XH = NX * 128  # bytes per K half of the X_r tile
        NP = (T2 + 7) // 8  # row-layout passes (8 X rows per pass)
        PARTIAL = T2 % 8 != 0
        DROPIN = self.dropin
        EMIT = self.emit
        GM = self.gate_mode
        TT = 16  # coefficient-table row pitch
        VB = V_DIM * 2  # bytes per v row

        tid, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        G_, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(tid // 32)
        lane = tid % 32
        if cutlass.const_expr(self.hv_heads is not None):
            HV = self.hv_heads
            H = self.q_heads
        GRP = HV // H  # value heads per query head (GQA)
        # persistent: CTA bid takes items bid, bid + G, ... (item = b * HV + hv)
        n_items = Int32(gH0idx.layout.shape[0]) * HV
        n_it = Int32(1)  # one item per CTA: a constant trip count (loop folds away)
        if cutlass.const_expr(self.persistent):
            n_it = (n_items - Int32(bid) + Int32(G_) - 1) // Int32(G_)

        # ------------------------------------------------------------------ SMEM
        smem = cutlass.utils.SmemAllocator()
        st_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout((V_DIM, (64, 2)), stride=(64, (1, V_DIM * 64))),
        )
        s_state = smem.allocate_tensor(
            BFloat16, st_layout.outer, byte_alignment=1024, swizzle=st_layout.inner
        )
        # X_r = [khat; qhat]: readout B operand and the Gram A operand, K-major SW128
        s_x = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(NX * K_DIM), byte_alignment=1024
        )
        # v rows [t][V] bf16, bulk-copied by warp 0 (read in the epilogue)
        s_v = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(T * V_DIM), byte_alignment=128
        )
        s_ktil = smem.allocate_tensor(
            cutlass.Uint16, cute.make_layout(KT * XC_PITCH // 2), byte_alignment=128
        )
        # coefficient tables (written from the Gram fragments in P4) + beta; the raw
        # gate rows [t][K] bf16 (staged in P1) alias the tables until the end of P2
        # (beta lies past them)
        GST = ((T + 7) // 8) * 8 * K_DIM // 2  # fp32 words of the gate staging
        CO_N = max(2 * TT * TT, GST)
        s_coef = smem.allocate_tensor(
            Float32, cute.make_layout(CO_N + TT), byte_alignment=128
        )
        st_full = [smem.allocate_array(cutlass.Int64, 1) for _ in range(2)]
        x_ready = smem.allocate_array(cutlass.Int64, 1)
        v_full = smem.allocate_array(cutlass.Int64, 1)
        mma_done = smem.allocate_array(cutlass.Int64, 1)
        tmem_done = smem.allocate_array(cutlass.Int64, 1)
        v_free = smem.allocate_array(cutlass.Int64, 1)
        tmem_addr = smem.allocate(Int32, 4)
        _CO_A = 0
        _CO_BQ = TT * TT
        _CO_BETA = CO_N

        if warp == 0:
            U.tmem_alloc(tmem_addr, _TMEM_COLS)
            cute.arch.relinquish_tmem_alloc_permit()
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(st_full[0], 1)
                cute.arch.mbarrier_init(st_full[1], 1)
                cute.arch.mbarrier_init(x_ready, 128)
                cute.arch.mbarrier_init(v_full, 1)
                cute.arch.mbarrier_init(mma_done, 1)
                cute.arch.mbarrier_init(tmem_done, 128)
                cute.arch.mbarrier_init(v_free, 128)
                cute.arch.mbarrier_init_fence()
            cute.arch.fence_acq_rel_cta()
            U.fence_before_sync()
            cute.arch.barrier_arrive(barrier_id=2, number_of_threads=160)

        if cutlass.const_expr(DROPIN):
            sq_t = s_q_tok
            sk_t = s_k_tok
            sv_t = s_v_tok
            sg_t = s_g_tok
            sb_t = s_b_tok
            so_t = s_o_tok
        else:
            sq_t = H * K_DIM
            sk_t = H * K_DIM
            sv_t = HV * V_DIM
            sg_t = HV * K_DIM
            sb_t = HV
            so_t = HV * V_DIM
        st_addr = s_state.iterator.toint()
        x_addr = s_x.iterator.toint()
        v_addr = s_v.iterator.toint()
        ktil_addr = s_ktil.iterator.toint()
        coef_addr = s_coef.iterator.toint()
        gst_addr = coef_addr
        q_base = gQ.iterator.toint()
        k_base = gK.iterator.toint()
        v_base = gV.iterator.toint()
        g_base = gG.iterator.toint()
        b_base = gBeta.iterator.toint()
        o_base = gOut.iterator.toint()
        c_base = gCorr.iterator.toint()
        kg_base = gKg.iterator.toint()

        if warp == 0:
            # ============================================================ producer
            cpasync.prefetch_descriptor(tma_ld.atom)
            U.fence_before_sync()
            cute.arch.sync_warp()
            U.fence_after_sync()
            tbase = tmem_addr[0]
            sd = U.sdesc_sw128(V_DIM * 128)
            x_desc = sd + Uint64(x_addr >> 4)
            stage_desc = [
                sd + Uint64(st_addr >> 4),
                sd + Uint64((st_addr + 16384) >> 4),
            ]
            idesc = _bf16_idesc(128, NX)
            for it in cutlass.range(n_it, unroll=1):
                item = Int32(bid) + it * Int32(G_)
                ph = it & 1
                hv = item % HV
                brow = item // HV
                raw_idx = Int32(gH0idx[brow])
                # dense: a negative slot reads a zero state (TMA out-of-bounds fill), as
                # the WY kernel; drop-in null slots load slot 0 (their output is zeroed)
                sidx = raw_idx
                if cutlass.const_expr(DROPIN):
                    sidx = _sel(raw_idx < 0, 0, raw_idx, Int32)
                bos = Int32(0)
                seq_len = Int32(T)
                if cutlass.const_expr(DROPIN):
                    bos = Int32(gQsl[brow])
                    seq_len = Int32(gQsl[brow + 1]) - bos
                tok0 = bos if DROPIN else brow * T
                # state buffer free once the previous item's readout completed
                if it > 0:
                    cute.arch.mbarrier_wait(mma_done, ph ^ 1)
                for gk in cutlass.range_constexpr(2):
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(st_full[gk], 16384)
                    U.simple_tma_copy(
                        tma_ld.atom,
                        tma_ld.tma_tensor[sidx, hv, None, (None, gk)],
                        s_state[None, (None, gk)],
                        st_full[gk],
                    )
                # v rows free once the consumers read the previous item's
                if it > 0:
                    cute.arch.mbarrier_wait(v_free, ph ^ 1)
                with cute.arch.elect_one():
                    nv = _sel(seq_len < T, seq_len, T, Int32)
                    nv = _sel(nv < 0, 0, nv, Int32)
                    cute.arch.mbarrier_arrive_and_expect_tx(v_full, nv * VB)
                    for t in cutlass.range_constexpr(T):
                        _bulk_g2s_if(
                            Int32(Int32(t) < seq_len),
                            v_addr + t * VB,
                            v_base
                            + (Int64(tok0 + t) * Int64(sv_t) + Int64(hv * V_DIM))
                            * Int64(2),
                            Int32(VB),
                            v_full.toint(),
                        )
                # TMEM free once the consumers loaded the previous readout
                if it > 0:
                    cute.arch.mbarrier_wait(tmem_done, ph ^ 1)
                cute.arch.mbarrier_wait(x_ready, ph)
                U.fence_after_sync()
                for gk in cutlass.range_constexpr(2):
                    cute.arch.mbarrier_wait(st_full[gk], ph)
                    U.fence_after_sync()
                    for ka in cutlass.range_constexpr(4):
                        U.umma_ss(
                            tbase,
                            stage_desc[gk] + Uint64((ka * 32) >> 4),
                            x_desc + Uint64((gk * XH + ka * 32) >> 4),
                            idesc,
                            (gk > 0) or (ka > 0),
                        )
                U.umma_commit(mma_done)
            cute.arch.mbarrier_wait(tmem_done, (n_it - 1) & 1)
            U.fence_after_sync()
            U.tmem_dealloc(tbase, _TMEM_COLS)
        else:
            # ============================================================ consumers
            cw = warp - 1
            ct = tid - 32  # consumer index: channel / (row, lane)
            tq = (warp & 3) << 5  # TMEM lane quadrant of this warp (tcgen05.ld base)
            row = tq | lane  # V row of this consumer (= TMEM lane)
            r8 = ct >> 4
            l16 = ct & 15
            grp = lane >> 2
            c4 = lane & 3
            ctx = dict(
                idx=gH0idx,
                qsl=gQsl,
                HV=HV,
                GRP=GRP,
                k=k_base,
                q=q_base,
                g=g_base,
                b=b_base,
                sk=sk_t,
                sq=sq_t,
                sg=sg_t,
                sb=sb_t,
                r8=r8,
                l16=l16,
                ct=ct,
            )
            # persistent grid: the next item's row inputs are loaded into registers
            # while this item's UMMA / Gram / epilogue run (software pipelined)
            PF_NEXT = self.persistent and self.prefetch_next
            if cutlass.const_expr(PF_NEXT):
                pf = self._load_inputs(Int32(bid), ctx)
            for it in cutlass.range(n_it, unroll=1):
                item = Int32(bid) + it * Int32(G_)
                if cutlass.const_expr(PF_NEXT):
                    cur = pf
                else:
                    cur = self._load_inputs(item, ctx)
                ph = it & 1
                hv = item % HV
                brow = item // HV
                i_h = hv // GRP  # query head (GQA)
                raw_idx = Int32(gH0idx[brow])
                bos = Int32(0)
                seq_len = Int32(T)
                if cutlass.const_expr(DROPIN):
                    bos = Int32(gQsl[brow])
                    seq_len = Int32(gQsl[brow + 1]) - bos
                tok0 = bos if DROPIN else brow * T
                is_null = raw_idx < Int32(self.null_min)
                kg_live = Int32(1)
                if cutlass.const_expr(DROPIN):
                    kg_live = Int32(raw_idx >= Int32(self.null_min))
                # ---- inputs (loads issued before the barriers)
                if it == 0:
                    cute.arch.barrier(
                        barrier_id=2, number_of_threads=160
                    )  # inits + TMEM
                    U.fence_after_sync()
                else:
                    # previous item's epilogue done with the coefficient / beta block
                    _bar_sync_ua(1, 128)
                tbase = tmem_addr[0]
                xw = [cur[4 * p_ : 4 * p_ + 4] for p_ in range(NP)]
                NGW = 4 * ((T + 7) // 8) if self.g_v4 else T
                graw = cur[4 * NP : 4 * NP + NGW]
                braw = cur[4 * NP + NGW]
                if cutlass.const_expr(self.g_v4):
                    # stage the gate rows [t][K] for the channel-layout P2
                    for i in cutlass.range_constexpr((T + 7) // 8):
                        U.sts_v4(
                            gst_addr + (r8 + 8 * i) * VB + l16 * 16,
                            graw[4 * i],
                            graw[4 * i + 1],
                            graw[4 * i + 2],
                            graw[4 * i + 3],
                        )
                eA = Float32(0.0)
                dtb = Float32(0.0)
                if cutlass.const_expr(GM != GATE_PRECOMPUTED):
                    eA = _ex2(Float32(gAlog[i_h]) * Float32(LOG2_E))
                    if cutlass.const_expr(self.has_dt_bias):
                        dtb = Float32(gDtbias[i_h * K_DIM + ct])

                # -------- P1: L2 norm (row layout), raw k̂ / q̂ into X_r, kg key half.
                for p in cutlass.range_constexpr(NP):
                    xr = r8 + 8 * p
                    ss = Float32(0.0)
                    for i in cutlass.range_constexpr(4):
                        lo = U.bf16x2_lo_f32(xw[p][i])
                        hi = U.bf16x2_hi_f32(xw[p][i])
                        ss = ss + lo * lo
                        ss = ss + hi * hi
                    for off in (8, 4, 2, 1):
                        ss = ss + cute.arch.shuffle_sync_bfly(ss, off)
                    inv = _rsqrt_approx(ss + Float32(_EPS))
                    inv = _sel(xr >= T, inv * scale, inv, Float32)
                    nw = []
                    for i in cutlass.range_constexpr(4):
                        nw.append(
                            U.pack_bf16x2(
                                U.bf16x2_lo_f32(xw[p][i]) * inv,
                                U.bf16x2_hi_f32(xw[p][i]) * inv,
                            )
                        )
                    xa = (
                        x_addr
                        + (l16 // 8) * XH
                        + xr * 128
                        + (((l16 % 8) ^ (xr % 8)) * 16)
                    )
                    if cutlass.const_expr(PARTIAL and p == NP - 1):
                        # dead rows xr >= 2T are zero
                        zw = [_sel(xr < T2, w_, Uint32(0), Uint32) for w_ in nw]
                        U.sts_v4(xa, zw[0], zw[1], zw[2], zw[3])
                    else:
                        U.sts_v4(xa, nw[0], nw[1], nw[2], nw[3])
                    if cutlass.const_expr(EMIT):
                        if xr < T:
                            # kg key half: L2-normed k (dense emit) or raw k (drop-in)
                            if cutlass.const_expr(DROPIN):
                                if (kg_live != 0) & (xr < seq_len):
                                    kga = kg_base + (
                                        Int64(raw_idx) * Int64(s_kg_blk)
                                        + Int64(
                                            hv * s_kg_head + xr * s_kg_pos + l16 * 8
                                        )
                                    ) * Int64(2)
                                    if cutlass.const_expr(self.kg_v4):
                                        U.stg_v4(
                                            kga, xw[p][0], xw[p][1], xw[p][2], xw[p][3]
                                        )
                                    else:  # kg strides not multiples of 8: 2-B stores
                                        for i in cutlass.range_constexpr(4):
                                            _stg_u16(kga + 4 * i, xw[p][i])
                                            _stg_u16(kga + 4 * i + 2, xw[p][i] >> 16)
                            else:
                                kga = kg_base + (
                                    (Int64(tok0 + xr) * Int64(HV) + Int64(hv))
                                    * Int64(2 * K_DIM)
                                    + Int64(l16 * 8)
                                ) * Int64(2)
                                U.stg_v4(kga, nw[0], nw[1], nw[2], nw[3])
                z = Uint32(0)
                if True:  # rows 8 NP.. of X_r and T.. of Ktil are zero
                    for p in cutlass.range_constexpr(NP, NX // 8):
                        zr = r8 + 8 * p
                        U.sts_v4(
                            x_addr
                            + (l16 // 8) * XH
                            + zr * 128
                            + (((l16 % 8) ^ (zr % 8)) * 16),
                            z,
                            z,
                            z,
                            z,
                        )
                    if cutlass.const_expr(T < KT):
                        if ct < (KT - T) * 16:
                            U.sts_v4(
                                ktil_addr + (T + r8) * XC_PITCH + l16 * 16, z, z, z, z
                            )
                _bar_sync_ua(1, 128)

                # -------- P2: gate + operand tiles (channel layout, c = ct)
                c = ct
                cum = Float32(0.0)
                if cutlass.const_expr(self.g_v4):
                    graw = [U.lds_u16(gst_addr + t * VB + c * 2) for t in range(T)]
                for t in cutlass.range_constexpr(T):
                    gx = U.u16_as_f32(graw[t], True)
                    if cutlass.const_expr(EMIT):
                        if cutlass.const_expr(DROPIN):
                            if (kg_live != 0) & (Int32(t) < seq_len):
                                _stg_u16(
                                    kg_base
                                    + (
                                        Int64(raw_idx) * Int64(s_kg_blk)
                                        + Int64(
                                            hv * s_kg_head + t * s_kg_pos + K_DIM + c
                                        )
                                    )
                                    * Int64(2),
                                    graw[t],
                                )
                        else:
                            _stg_u16(
                                kg_base
                                + (
                                    (Int64(tok0 + t) * Int64(HV) + Int64(hv))
                                    * Int64(2 * K_DIM)
                                    + Int64(K_DIM + c)
                                )
                                * Int64(2),
                                graw[t],
                            )
                    glog = gx
                    if cutlass.const_expr(GM == GATE_LOWER_BOUND):
                        xg = gx
                        if cutlass.const_expr(self.has_dt_bias):
                            xg = gx + dtb
                        sig = Float32(1.0) / (
                            Float32(1.0)
                            + _ex2((Float32(0.0) - eA) * xg * Float32(LOG2_E))
                        )
                        glog = lower_bound * sig
                    elif cutlass.const_expr(GM == GATE_SOFTPLUS):
                        xg = gx
                        if cutlass.const_expr(self.has_dt_bias):
                            xg = gx + dtb
                        sp = cute.math.log(Float32(1.0) + _ex2(xg * Float32(LOG2_E)))
                        glog = (Float32(0.0) - eA) * sp
                    if cutlass.const_expr(DROPIN):
                        glog = _sel(Int32(t) < seq_len, glog, 0.0, Float32)
                    cum = cum + glog
                    ep = _ex2(cum * Float32(LOG2_E))
                    en = _ex2(cum * Float32(-LOG2_E))  # == (0 - cum) * log2(e)
                    ka_ = x_addr + _x_sw128_off(t, c, XH)
                    qa_ = x_addr + _x_sw128_off(T + t, c, XH)
                    kh = U.u16_as_f32(U.lds_u16(ka_), True)
                    qh = U.u16_as_f32(U.lds_u16(qa_), True)
                    _sts_bf16(ka_, kh * ep)
                    _sts_bf16(qa_, qh * ep)
                    _sts_bf16(ktil_addr + t * XC_PITCH + c * 2, kh * en)
                if ct < T:
                    bv = U.u16_as_f32(braw, True)
                    if cutlass.const_expr(self.beta_is_logit):
                        bv = Float32(1.0) / (
                            Float32(1.0) + _ex2((Float32(0.0) - bv) * Float32(LOG2_E))
                        )
                    s_coef[_CO_BETA + ct] = bv
                # generic X_r stores -> async-proxy fence -> tcgen05 fence -> x_ready:
                # the readout UMMA (async proxy) sees the finished X_r tile
                cute.arch.fence_view_async_shared()
                U.fence_before_sync()
                cute.arch.mbarrier_arrive(x_ready)
                if cutlass.const_expr(PF_NEXT):
                    # every input word of this item is consumed (X_r / Ktil / gate
                    # staging / beta written): load the next item's (clamped to the
                    # last item past the end; the values are then unused)
                    nxt = item + Int32(G_)
                    pf = self._load_inputs(
                        _sel(nxt < n_items, nxt, n_items - 1, Int32), ctx
                    )
                _bar_sync_ua(1, 128)

                # -------- P4: mma.sync Gram X_r . Ktil^T (fp32) -> coefficient tables; A
                # fragments straight from the SW128 X_r tile
                NTILE = (NX // 16) * (KT // 8)
                if cw < NTILE:
                    mt = cw >> 1 if KT == 16 else cw
                    nt = cw & 1 if KT == 16 else 0
                    ar = mt * 16 + lane % 16
                    b_ln = (
                        ktil_addr
                        + (nt * 8 + lane % 8) * XC_PITCH
                        + ((lane // 8) % 2) * 16
                    )
                    acc = [Float32(0.0), Float32(0.0), Float32(0.0), Float32(0.0)]
                    for ks in cutlass.range_constexpr(8):
                        chunk = ks * 2 + lane // 16
                        fa = _ldmatrix_x4(
                            x_addr
                            + (chunk // 8) * XH
                            + ar * 128
                            + (((chunk % 8) ^ (ar % 8)) * 16)
                        )
                        fb = _ldmatrix_x2(b_ln + ks * 32)
                        acc = U.mma_bf16_16816(
                            fa[0], fa[1], fa[2], fa[3], fb[0], fb[1], acc
                        )
                    # coefficient tables straight from the fragments: Gram row r < T
                    # is k̂_r (A[r][s] = beta_r G, s < r), T <= r < 2T is q̂_{r-T}
                    # (Bq[r-T][s] = G, s <= r-T); rows >= 2T are dead
                    r0 = mt * 16 + grp
                    c0 = nt * 8 + 2 * c4
                    for h_ in cutlass.range_constexpr(2):
                        rr = r0 + 8 * h_
                        is_k = rr < T
                        tr = _sel(is_k, rr, rr - T, Int32)
                        bt = s_coef[_CO_BETA + _sel(is_k, rr, 0, Int32)]
                        for e_ in cutlass.range_constexpr(2):
                            cc = c0 + e_
                            gv = acc[2 * h_ + e_]
                            av = _sel(cc < tr, -(bt * gv), 0.0, Float32)
                            bq = _sel(cc <= tr, gv, 0.0, Float32)
                            if rr < T2:
                                # token pairs interleaved: (t, s) at
                                # ((t >> 1) TT + s) 2 + (t & 1)
                                s_coef[
                                    _sel(is_k, _CO_A, _CO_BQ, Int32)
                                    + ((tr >> 1) * TT + cc) * 2
                                    + (tr & 1)
                                ] = _sel(is_k, av, bq, Float32)
                _bar_sync_ua(1, 128)

                # -------- epilogue: per V row forward substitution
                cute.arch.mbarrier_wait(mma_done, ph)
                U.fence_after_sync()
                nld = 1 << (T - 1).bit_length()
                hk = U.tmem_ld(tq, tbase, nld)
                U.tmem_wait_ld()
                cute.arch.mbarrier_wait(v_full, ph)
                # forward substitution two tokens at a time with f32x2 fmas: each
                # lane runs one token's chain in the original order, so every value
                # is rounded exactly as the scalar recurrence (A is stored negated:
                # fma(-A, u, acc) == acc - A u)

                def _co(tab, t_, s_):
                    return coef_addr + (tab + ((t_ >> 1) * TT + s_) * 2 + (t_ & 1)) * 4

                def _ld_v(t_):
                    return U.u16_as_f32(U.lds_u16(v_addr + t_ * VB + row * 2), True)

                # u_t = beta_t (v_t - hk_t) - sum_s A_ts u_s; the first term is
                # fused as fma(beta_t, v_t - hk_t, -A_t0 u_0) (the scalar code's
                # contraction), the rest as fma(-A_ts, u_s, acc)
                uu: list = []
                ud: list = []  # (u_s, u_s) pairs
                for t0 in cutlass.range_constexpr(0, T, 2):
                    if cutlass.const_expr(t0 + 1 < T):
                        bt2 = _pack2(s_coef[_CO_BETA + t0], s_coef[_CO_BETA + t0 + 1])
                        d2 = _pack2(_ld_v(t0) - hk[t0], _ld_v(t0 + 1) - hk[t0 + 1])
                        if cutlass.const_expr(t0 == 0):
                            b0, b1 = _unpack2(bt2)
                            e0, e1 = _unpack2(d2)
                            u0 = b0 * e0
                            u1 = _ffma(b1, e1, s_coef[_CO_A + 1] * u0)
                        else:
                            cf = _lds_b64x2(_co(_CO_A, t0, 0))
                            acc = _ffma2(bt2, d2, _fmul2(cf[0], ud[0]))
                            acc = _ffma2(cf[1], ud[1], acc)
                            for s2 in cutlass.range_constexpr(2, t0, 2):
                                cf = _lds_b64x2(_co(_CO_A, t0, s2))
                                acc = _ffma2(cf[0], ud[s2], acc)
                                acc = _ffma2(cf[1], ud[s2 + 1], acc)
                            u0, a1 = _unpack2(acc)
                            u1 = _ffma(
                                s_coef[_CO_A + ((t0 >> 1) * TT + t0) * 2 + 1], u0, a1
                            )
                        uu += [u0, u1]
                        ud += [_pack2(u0, u0), _pack2(u1, u1)]
                    else:
                        bt = s_coef[_CO_BETA + t0]
                        d0 = _ld_v(t0) - hk[t0]
                        if cutlass.const_expr(t0 == 0):
                            a0 = bt * d0
                        else:
                            a0 = _ffma(
                                bt, d0, s_coef[_CO_A + ((t0 >> 1) * TT) * 2] * uu[0]
                            )
                            for s_ in cutlass.range_constexpr(1, t0):
                                a0 = _ffma(
                                    s_coef[_CO_A + ((t0 >> 1) * TT + s_) * 2],
                                    uu[s_],
                                    a0,
                                )
                        uu.append(a0)
                # v reads (generic proxy) ordered before warp 0's next bulk copy into
                # the rows (async proxy): proxy fence, then the release arrive
                cute.arch.fence_view_async_shared()
                cute.arch.mbarrier_arrive(v_free)  # v rows read
                hq = U.tmem_ld(tq, tbase + T, nld)
                U.tmem_wait_ld()
                U.fence_before_sync()
                cute.arch.mbarrier_arrive(tmem_done)  # readout read out of TMEM
                yy: list = []
                for t0 in cutlass.range_constexpr(0, T, 2):
                    if cutlass.const_expr(t0 + 1 < T):
                        acc = _pack2(hq[t0], hq[t0 + 1])
                        for s2 in cutlass.range_constexpr(0, t0, 2):
                            cf = _lds_b64x2(_co(_CO_BQ, t0, s2))
                            acc = _ffma2(cf[0], ud[s2], acc)
                            acc = _ffma2(cf[1], ud[s2 + 1], acc)
                        acc = _ffma2(_lds_b64(_co(_CO_BQ, t0, t0)), ud[t0], acc)
                        y0, y1 = _unpack2(acc)
                        y1 = _ffma(
                            s_coef[_CO_BQ + ((t0 >> 1) * TT + t0 + 1) * 2 + 1],
                            uu[t0 + 1],
                            y1,
                        )
                        yy += [y0, y1]
                    else:
                        y0 = hq[t0]
                        for s_ in cutlass.range_constexpr(t0 + 1):
                            y0 = _ffma(
                                s_coef[_CO_BQ + ((t0 >> 1) * TT + s_) * 2], uu[s_], y0
                            )
                        yy.append(y0)
                for t in cutlass.range_constexpr(T):
                    y = yy[t]
                    if cutlass.const_expr(DROPIN):
                        y = _sel(is_null, 0.0, y, Float32)
                        if Int32(t) < seq_len:
                            _stg_bf16(
                                o_base
                                + (
                                    Int64(tok0 + t) * Int64(so_t)
                                    + Int64(hv * V_DIM + row)
                                )
                                * Int64(2),
                                y,
                            )
                            if kg_live != 0:
                                _stg_f32(
                                    c_base
                                    + (
                                        Int64(raw_idx) * Int64(s_c_blk)
                                        + Int64(hv * s_c_head + t * s_c_pos + row)
                                    )
                                    * Int64(4),
                                    uu[t],
                                )
                    else:
                        oa = (
                            Int64(tok0 + t) * Int64(so_t) + Int64(hv * V_DIM + row)
                        ) * Int64(2)
                        _stg_bf16(o_base + oa, y)
                        if cutlass.const_expr(EMIT):
                            _stg_bf16(c_base + oa, uu[t])
