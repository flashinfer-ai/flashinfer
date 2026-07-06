# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Chunked Gated Delta Net 2 (GDN-2) prefill kernel for Blackwell SM100.

GDN-2 generalizes GDN to channel-wise gates: the decay gate is per
(token, key-channel), the erase gate beta per (token, key-channel), and the
write gate w per (token, value-channel).  The scalar T-pairwise matrix of
GDN no longer exists; the per-channel decays are folded directly into the
GEMM operands (K_bar / E / q_gamma below).

Algorithm overview (per chunk c, tokens [cC, (c+1)C)):
  Inputs : Q[BT,DK], K[BT,DK], V[BT,DV], gate[BT,DK] (decay), beta[BT,DK]
           (erase LR), w[BT,DV] (write gate)
  State  : S_prev[DK,DV]  (recurrent state, held in TMEM)

  Preprocessing (compute warp group 0; L is negative, exp2(L) decays):
    L[t,c]   = sum_{l=0}^{t} log2(gate[l,c])     per-channel cumulative log-decay
    K_bar    = K * exp2(-L)                       decay-unfolded keys
    E        = exp2(L) * beta * K                 erase-folded keys
    q_gamma  = exp2(L) * Q                        decay-folded queries
    cumprod_total[c] = exp2(L[BT-1,c])            per-channel chunk-total decay

  GEMM 1 - kk      : W_kk[BT,BT]  = E @ K_bar^T        (lower-triangular intra scores)
  GEMM 2 - qk      : W_qk[BT,BT]  = q_gamma @ K_bar^T  (output attention scores)
  GEMM 3 - k*state : KS[BT,DV]    = E @ S_prev         (erase keys applied to state)
  GEMM 4 - q*state : QS[BT,DV]    = q_gamma @ S_prev   (inter-chunk output)
  GEMM 5 - new v   : NV[BT,DV]    = A_inv @ (w*V - KS) (corrected value vectors)
                      where A_inv = (I + M_kk)^{-1},  M_kk = tril(W_kk)  (hierarchical blockwise inverse)
  GEMM 6 - qkv     : O_intra[BT,DV] = W_qkv @ NV       (intra-chunk output)
                      where W_qkv = tril(W_qk) * scale
  GEMM 7 - kv update : dS[DK,DV]  = K_bar^T @ NV       (state update, BT contraction)

  Epilogue:
    O[BT,DV]  = O_intra + scale * QS               (combine intra + inter)
    S_next    = S_prev + cumprod_total * dS         (per-key-channel state update;
                the decay of S_prev is carried by the exp2(L) folds in E/q_gamma)

SMEM layout (226 KB total):
  Buffer                           Size (B)  Stages
  q                                32768     2      <-- raw Q TMA-lands, overwritten with q_gamma
  k                                32768     2      <-- K_bar (written by CG0)
  e                                32768     2      <-- raw K TMA-lands, overwritten with E
  v                                32768     2      <-- double-buffered (prefetch next chunk)
  A_inverse / new_v                 8192     1      <-- A_inv result, then overwritten with fp16 NV
  QK output                         8192     1      <-- W_qkv scores
  O store                          16384     1      <-- O epilogue staging
  raw gate / cumsumlog             32768     1      <-- BT x DK fp32; TMA-landed gate,
                                                        overwritten in place with L
  cumprod_total                     1024     2      <-- DK x fp32 scalars
  beta                             16384     1      <-- BT x DK bf16
  w                                16384     1      <-- BT x DV bf16

TMEM layout (256 KB total, 512 columns):
  Buffer                  Size (B)  Stages
  state (S)               65536     1      <-- DKxDV fp32 = 128x128x4B
  q*state acc             32768     1      <-- BTxDV fp32 accumulator
  state input             32768     1      <-- w*V - KS staging (GEMM 5 A operand)
  shared acc              65536     2      <-- shared accumulator for GEMMs 1/2/5/6/7
  shared input            65536     2      <-- W_qkv / NV staging (GEMM 6 operands)

Warp assignments (12 warps = 384 threads):
  warps 0-3     : compute group 0 - gate cumsum, K_bar/E/q_gamma folds,
                                    kk_epi, qk_epi, inverse
  warps 4-7     : compute group 1 - w*V - k*state staging, state*q_epi,
                                    new_v_epi, dS combine, qkv_epilogue
  warp  8       : MMA warp       - issues all 7 GEMMs
  warp  9       : TMA load warp  - loads q, v, raw k (into e, double-buf)
  warp  10      : TMA gate warp  - loads gate, beta, w
  warp  11      : epilogue warp   - store O to global memory
"""

import math
from typing import Optional, Type, Tuple

import cuda.bindings.driver as cuda


import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.utils import TensorMapManager, TensorMapUpdateMode
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cute.nvgpu import cpasync, tcgen05, OperandMajorMode
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.cute.testing as testing

# ---------------------------------------------------------------------------
# cutlass-dsl 4.4.2 compatibility: TmaInfo was removed; make_tiled_tma_atom_*
# now returns a plain (CopyAtom, Tensor) tuple instead of TmaInfo.
# ---------------------------------------------------------------------------
try:
    from cutlass.cute.nvgpu.cpasync import TmaInfo
except ImportError:
    from cutlass.base_dsl import (
        extract_mlir_values as _emv,
        new_from_mlir_values as _nfmv,
        extract_mlir_attributes as _ema,
        get_mlir_types as _gmt,
    )

    class TmaInfo:  # type: ignore[no-redef]
        """Compatibility shim replacing cpasync.TmaInfo for cutlass-dsl >= 4.4.2."""

        def __init__(self, atom, tma_tensor, smem_layout=None):
            self._atom = atom
            self._tma_tensor = tma_tensor

        @property
        def atom(self):
            return self._atom

        @property
        def tma_tensor(self):
            return self._tma_tensor

        def __extract_mlir_values__(self):
            return _emv(self._atom) + _emv(self._tma_tensor)

        def __extract_mlir_attributes__(self):
            return _ema(self._atom) + _ema(self._tma_tensor)

        def __new_from_mlir_values__(self, values):
            n = len(_gmt(self._atom))
            return TmaInfo(
                _nfmv(self._atom, values[:n]),
                _nfmv(self._tma_tensor, values[n:]),
            )

        def __iter__(self):
            yield self._atom
            yield self._tma_tensor

        def __getitem__(self, i):
            return (self._atom, self._tma_tensor)[i]

        def __len__(self):
            return 2


def _wrap_tma(ret):
    """Wrap make_tiled_tma_atom_* return value in TmaInfo if not already."""
    if isinstance(ret, TmaInfo):
        return ret
    # 4.4.2: returns (CopyAtom, Tensor) tuple
    return TmaInfo(ret[0], ret[1])


from .gated_delta_net_tile_scheduler import (
    GDNTileSchedulerParams,
    GDNTileScheduler,
)


def _bf16x2_to_f32x2(packed_i32):
    """Upcast a packed bf16x2 (one b32 register) to two fp32 via MOV/pack."""
    return cute.arch.inline_ptx(
        "{\n"
        ".reg .b16 lo, hi, z;\n"
        "mov.b16 z, 0;\n"
        "mov.b32 {lo, hi}, $2;\n"
        "mov.b32 $0, {z, lo};\n"
        "mov.b32 $1, {z, hi};\n"
        "}",
        write_only_types=[cutlass.Float32, cutlass.Float32],
        read_only_args=[packed_i32],
    )


def _f32x2_to_bf16x2(lo_f32, hi_f32):
    """Pack two fp32 into one bf16x2 register (rn round), one cvt op."""
    return cute.arch.inline_ptx(
        "cvt.rn.bf16x2.f32 $0, $2, $1;",
        write_only_types=[cutlass.Int32],
        read_only_args=[lo_f32, hi_f32],
    )


def _upcast_bf16_frag_to_f32(dst_f32, src_bf16):
    """Fill dst_f32 (fp32 rmem frag) from src_bf16 (bf16 rmem frag) via MOV-pack pairs."""
    src_i32 = cute.recast_tensor(src_bf16, cutlass.Int32)
    for p in range(cute.size(src_i32)):
        f0, f1 = _bf16x2_to_f32x2(src_i32[p])
        dst_f32[2 * p] = f0
        dst_f32[2 * p + 1] = f1


# ---------------------------------------------------------------------------
# Combined configuration + execution class
# ---------------------------------------------------------------------------


class GatedDeltaNetChunkedKernel2:
    """
    Configuration and execution class for the Chunked GDN kernel.

    Follows the same class-based structure:
      - __init__    : warp IDs, barriers, tile shapes, SMEM/TMEM sizes
      - __call__    : @cute.jit host entry point (TMA setup, kernel launch)
      - kernel      : @cute.kernel device entry point (warp dispatch)
      - per-warp methods called from kernel's chunk loop

    Args:
        io_dtype   : input/output dtype (Float16 or BFloat16)
        acc_dtype  : accumulator dtype  (Float32)
        BT         : chunk size / block tile  (64)
        DK         : key/query hidden dim     (128)
        DV         : value hidden dim         (128)
    """

    # TMA descriptor size in bytes
    bytes_per_tensormap = 128
    # Slots: Q=0, K=1, V=2, O=3, beta=4, w=5  (beta/w added for GDN-2 TMA loads)
    num_tensormaps = 6

    def __init__(
        self,
        io_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        state_dtype: Type[cutlass.Numeric],
        mma_tiler_qk: Tuple[int, int, int],
        mma_tiler_qs: Tuple[int, int, int],
        mma_tiler_qkv: Tuple[int, int, int],
        mma_tiler_kv: Tuple[int, int, int],
        max_active_clusters: int,
        num_sm: int,
        is_GQA: bool,
        use_initial_state: bool,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        is_persistent: bool = True,
    ):
        self.io_dtype = io_dtype
        self.acc_dtype = acc_dtype
        self.state_dtype = state_dtype
        self.mma_tiler_qk = mma_tiler_qk
        self.mma_tiler_qs = mma_tiler_qs
        self.mma_tiler_qkv = mma_tiler_qkv
        self.mma_tiler_kv = mma_tiler_kv
        self.max_active_clusters = max_active_clusters
        self.num_sm = num_sm
        self.is_GQA = is_GQA
        self.use_initial_state = use_initial_state
        self.store_final_state = store_final_state
        self.enable_checkpoints = enable_checkpoints
        self.is_persistent = is_persistent

        # ------------------------------------------------------------------
        # Warp assignments  (12 warps total)
        # ------------------------------------------------------------------
        # T-pairwise / kk_epi / qk_epi / inverse
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # kv_decay_v / v-k*state / epi ops
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        self.mma_warp_id = 8
        self.tma_qkv_warp_id = 9
        self.load_gate_beta_warp_id = 10
        # store O
        self.epilogue_warp_id = 11

        self.num_regs_compute_group_0 = 232  # sum with CG1 + 24 donor must stay 504
        self.num_regs_compute_group_1 = 248
        self.num_regs_other = 24

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_warp_id,
                    self.tma_qkv_warp_id,
                    self.load_gate_beta_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32

        # ------------------------------------------------------------------
        # Named barriers - only TmemAllocator requires a NamedBarrier;
        # all other inter-warp synchronization uses mbarrier-based pipelines
        # created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                )
            ),
        )
        self.inverse_barrier = pipeline.NamedBarrier(
            barrier_id=2,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )
        self.inverse_barrier_inner = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=self.threads_per_warp * 2,
        )
        self.init_state_store_barrier = pipeline.NamedBarrier(
            barrier_id=4,
            num_threads=self.threads_per_warp * len(self.compute_group_1_warp_ids),
        )
        # CG0-wide sync between the per-channel cumprod write to sCumprod and the K/Q
        # cumprod gather (which reads other threads' channels).
        self.cumprod_barrier = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=self.threads_per_warp * len(self.compute_group_0_warp_ids),
        )

    def _setup_attributes(self):
        # ------------------------------------------------------------------
        # SMEM sizes (bytes per stage) and stage counts
        # ------------------------------------------------------------------
        self.smem_q_stages = 2
        self.smem_k_stages = 2
        self.smem_v_stages = 2
        self.smem_ainv_stages = 1
        self.smem_qk_stages = 1
        self.smem_o_stages = 1
        self.smem_group_order_stages = 1
        # Cumulative gate buffers - placed last in SMEM
        self.smem_gate_stages = 1
        self.smem_cumprod_total_stages = 2
        self.smem_beta_stages = 1

        # ------------------------------------------------------------------
        # TMEM column offsets and buffer sizes (fp32, 32B per column)
        # ------------------------------------------------------------------
        self.tmem_kv_acc_stages = 1
        self.tmem_q_state_acc_stages = 1
        self.tmem_state_inp_stages = 1
        self.tmem_shared_inp_stages = 2
        self.tmem_shared_acc_stages = 2

        self.tmem_state_offset = 0
        self.tmem_q_state_offset = (
            self.tmem_state_offset + self.tmem_kv_acc_stages * 128
        )
        self.tmem_state_inp_offset = (
            self.tmem_q_state_offset + self.tmem_q_state_acc_stages * 64
        )
        self.tmem_shared_acc_offset = (
            self.tmem_state_inp_offset + self.tmem_state_inp_stages * 64
        )
        self.tmem_shared_inp_offset = (
            self.tmem_shared_acc_offset + self.tmem_shared_acc_stages * 64
        )

        self.buffer_align_bytes = 1024

    # -----------------------------------------------------------------------
    # Capability check
    # -----------------------------------------------------------------------

    @staticmethod
    def can_implement(
        io_dtype,
        acc_dtype,
        mma_tiler_qk,
        mma_tiler_qs,
        mma_tiler_qkv,
        mma_tiler_kv,
    ):
        """Raise CantImplementError if this configuration is not supported."""
        if io_dtype not in [cutlass.Float16, cutlass.BFloat16]:
            raise testing.CantImplementError(
                f"io_dtype={io_dtype} not supported; only Float16 and BFloat16 are supported"
            )
        if acc_dtype != cutlass.Float32:
            raise testing.CantImplementError(
                f"acc_dtype={acc_dtype} not supported; only Float32 is supported"
            )
        if mma_tiler_qk != (64, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qk={mma_tiler_qk} not supported; only (64, 64, 128) is supported"
            )
        if mma_tiler_qs != (128, 64, 128):
            raise testing.CantImplementError(
                f"mma_tiler_qs={mma_tiler_qs} not supported; only (128, 64, 128) is supported"
            )
        if mma_tiler_qkv != (128, 64, 64):
            raise testing.CantImplementError(
                f"mma_tiler_qkv={mma_tiler_qkv} not supported; only (128, 64, 64) is supported"
            )
        if mma_tiler_kv != (128, 128, 64):
            raise testing.CantImplementError(
                f"mma_tiler_kv={mma_tiler_kv} not supported; only (128, 128, 64) is supported"
            )

    # -----------------------------------------------------------------------
    # Host entry point
    # -----------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        gate: cute.Tensor,
        beta: cute.Tensor,
        w: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        s_in: Optional[cute.Tensor],
        s_out: Optional[cute.Tensor],
        s_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        tensormap_workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # chunk size
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        batch_size = cu_seqlens.shape[0] - 1

        self._setup_attributes()

        # gate/beta are channel-wise on the key axis (d_k); w on the value axis (d_v)
        self.d_k = q.shape[2]
        self.d_v = v.shape[2]

        if cutlass.const_expr(self.is_GQA):
            h_r = h_q // h_v
            h_qv = h_v
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_v)),
                    stride=(q.stride[0], q.stride[2], (q.stride[1], h_r * q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_v)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_v)),
                    stride=(v.stride[2], v.stride[0], (0, v.stride[1])),
                ),
            )
            # write gate w mirrors V (value-major [DV, token])
            w = cute.make_tensor(
                w.iterator,
                cute.make_layout(
                    (w.shape[2], w.shape[0], (h_r, h_v)),
                    stride=(w.stride[2], w.stride[0], (0, w.stride[1])),
                ),
            )
        else:
            h_r = h_v // h_q
            h_qv = h_q
            q = cute.make_tensor(
                q.iterator,
                cute.make_layout(
                    (q.shape[0], q.shape[2], (h_r, h_q)),
                    stride=(q.stride[0], q.stride[2], (0, q.stride[1])),
                ),
            )
            k = cute.make_tensor(
                k.iterator,
                cute.make_layout(
                    (k.shape[0], k.shape[2], (h_r, h_q)),
                    stride=(k.stride[0], k.stride[2], (0, k.stride[1])),
                ),
            )
            v = cute.make_tensor(
                v.iterator,
                cute.make_layout(
                    (v.shape[2], v.shape[0], (h_r, h_q)),
                    stride=(v.stride[2], v.stride[0], (v.stride[1], h_r * v.stride[1])),
                ),
            )
            w = cute.make_tensor(
                w.iterator,
                cute.make_layout(
                    (w.shape[2], w.shape[0], (h_r, h_q)),
                    stride=(w.stride[2], w.stride[0], (w.stride[1], h_r * w.stride[1])),
                ),
            )

        # Channel-wise gate/beta: (token, d, (h_r, h_qv))
        gate = cute.make_tensor(
            gate.iterator,
            cute.make_layout(
                (gate.shape[0], gate.shape[2], (h_r, h_qv)),
                stride=(
                    gate.stride[0],
                    gate.stride[2],
                    (gate.stride[1], h_r * gate.stride[1]),
                ),
            ),
        )
        beta = cute.make_tensor(
            beta.iterator,
            cute.make_layout(
                (beta.shape[0], beta.shape[2], (h_r, h_qv)),
                stride=(
                    beta.stride[0],
                    beta.stride[2],
                    (beta.stride[1], h_r * beta.stride[1]),
                ),
            ),
        )
        o = cute.make_tensor(
            o.iterator,
            cute.make_layout(
                (o.shape[2], o.shape[0], (h_r, h_qv)),
                stride=(o.stride[2], o.stride[0], (o.stride[1], h_r * o.stride[1])),
            ),
        )
        if cutlass.const_expr(s_in is not None):
            s_in = cute.make_tensor(
                s_in.iterator,
                cute.make_layout(
                    (s_in.shape[2], s_in.shape[3], (h_r, h_qv), s_in.shape[0]),
                    stride=(
                        s_in.stride[2],
                        s_in.stride[3],
                        (s_in.stride[1], h_r * s_in.stride[1]),
                        s_in.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(s_out is not None):
            s_out = cute.make_tensor(
                s_out.iterator,
                cute.make_layout(
                    (s_out.shape[2], s_out.shape[3], (h_r, h_qv), s_out.shape[0]),
                    stride=(
                        s_out.stride[2],
                        s_out.stride[3],
                        (s_out.stride[1], h_r * s_out.stride[1]),
                        s_out.stride[0],
                    ),
                ),
            )
        if cutlass.const_expr(self.enable_checkpoints):
            s_checkpoints = cute.make_tensor(
                s_checkpoints.iterator,
                cute.make_layout(
                    (
                        s_checkpoints.shape[2],
                        s_checkpoints.shape[3],
                        (h_r, h_qv),
                        s_checkpoints.shape[0],
                    ),
                    stride=(
                        s_checkpoints.stride[2],
                        s_checkpoints.stride[3],
                        (s_checkpoints.stride[1], h_r * s_checkpoints.stride[1]),
                        s_checkpoints.stride[0],
                    ),
                ),
            )

        # ------------------------------------------------------------------
        # Build tiled MMAs  (one per logical GEMM group, differing in operand major modes)
        # ------------------------------------------------------------------
        def _mma_op(mma_tiler, a_major, b_major, OperandSourceA):
            # Derive MMA atom (M, N) from the first two dims of the tile shape;
            # K=16 is the hardware fp16 atom depth (fixed for SM100 tcgen05).
            return tcgen05.MmaF16BF16Op(
                self.io_dtype,
                self.acc_dtype,
                (mma_tiler[0], mma_tiler[1], 16),
                self.cta_group,
                OperandSourceA,
                a_major,
                b_major,
            )

        # GEMM 1 (kk: K@K^T) + GEMM 2 (qk: Q@K^T)          - KK-major: A=K, B=K
        tiled_mma_qk = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qk,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 3 (k*state: K@S) + GEMM 4 (q*state: Q@S)     - KN-major: A=K, B=MN
        tiled_mma_qs = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qs,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # GEMM 5 (new_v: A_inv@V) + GEMM 6 (qkv: W_qkv@NV)  - KN-major: A=K, B=MN
        tiled_mma_qkv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.K,
                OperandMajorMode.K,
                tcgen05.OperandSource.TMEM,
            )
        )
        # for v_smem_layout_staged
        tiled_mma_qkv_ss = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_qkv,
                OperandMajorMode.MN,
                OperandMajorMode.K,
                tcgen05.OperandSource.SMEM,
            )
        )
        # GEMM 7 (kv_update: K^T@delta -> dS)                    - MN-major: A=MN, B=MN
        tiled_mma_kv = cute.make_tiled_mma(
            _mma_op(
                self.mma_tiler_kv,
                OperandMajorMode.K,
                OperandMajorMode.MN,
                tcgen05.OperandSource.TMEM,
            )
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # ------------------------------------------------------------------
        # SMEM layouts - computed before SharedStorage so cosize() is available
        # ------------------------------------------------------------------
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(self.cta_group)
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        # Q is A operand for tiled_mma_qk (GEMM 2: qk), K is B operand (GEMM 1+2: kk/qk)
        q_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_q_stages
        )
        k_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.io_dtype, self.smem_k_stages
        )
        k_trans_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_kv, self.mma_tiler_kv, self.io_dtype, self.smem_k_stages
        )
        # V is A operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv)
        v_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss, self.mma_tiler_qkv, self.io_dtype, self.smem_v_stages
        )
        # A_inv is B operand for tiled_mma_qkv (GEMM 5: new_v: V @ A_inv);
        ainv_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_ainv_stages
        )
        # W_qkv is A operand for tiled_mma_qkv (GEMM 6: qkv: NV @ qk)
        qk_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_qk_stages
        )

        o_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.io_dtype,
            utils.LayoutEnum.from_tensor(o),
            self.mma_tiler_qkv[:2],
            self.smem_o_stages,
        )
        # Gate buffer: canonical fp32 SW128 epilogue layout; the raw gate
        # TMA-lands already swizzled and is overwritten in place by the cumsum.
        cumsumlog_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            cutlass.Float32,
            utils.LayoutEnum.ROW_MAJOR,
            (self.b_t, self.d_k),
            self.smem_gate_stages,
        )
        gate_raw_smem_layout_staged = cumsumlog_smem_layout_staged
        # beta (erase gate, [BT, d_k]) mirrors K: B-operand SWIZZLED SMEM layout so it can
        # be ldmatrix-loaded (LDSM) into the same fragment as K, instead of LDS-gathered.
        beta_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, cutlass.BFloat16, self.smem_beta_stages
        )
        # w mirrors V: A-operand SMEM layout ([DV, BT])
        w_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma_qkv_ss,
            self.mma_tiler_qkv,
            cutlass.BFloat16,
            self.smem_beta_stages,
        )

        # ------------------------------------------------------------------
        # Shared memory struct  (defined here to capture layout cosizes)
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers - one entry per stage, 2 Int64 words per barrier
            # TMA load warp -> MMA warp (K double-buffered)
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # TMA load warp -> MMA warp (E double-buffered; mirror of K)
            load_e_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # TMA load warp -> MMA warp
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            # TMA load warp -> MMA warp
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            # CG0 -> CG1  (cumprod ready in sCumprod)
            cumprod_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_cumprod_total_stages * 2
            ]
            # TMA warp -> CG0  (raw gate TMA load into sGate)
            load_rawgate_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_gate_stages * 2
            ]
            # TMA warp -> CG0  (beta; independent barrier from w)
            load_beta_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_beta_stages * 2
            ]
            load_w_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_beta_stages * 2
            ]
            # MMA warp -> CG1 (Q*state acc ready in TMEM)
            q_state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_q_state_acc_stages * 2
            ]
            # MMA warp -> CG1 (GEMM 7 done)
            kv_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_kv_acc_stages * 2
            ]
            # MMA warp -> CG0/CG1 (GEMM 1-6 done)
            shared_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_shared_acc_stages * 2
            ]
            # CG0 -> MMA warp (A_inv ready in SMEM)
            ainv_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # CG0 -> MMA warp (QK ready in SMEM)
            qk_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_qk_stages * 2
            ]
            # CG0 -> MMA warp (q_gamma ready in SMEM; consumed by QK and QS)
            q_gamma_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_q_stages * 2
            ]
            # CG0 -> MMA warp (E ready in SMEM)
            e_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_k_stages * 2
            ]
            # CG0 -> MMA warp (k_bar ready in SMEM)
            k_bar_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_k_stages * 2
            ]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_inp_stages * 2
            ]
            # CG1 -> MMA warp (state input ready in SMEM)
            shared_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_shared_inp_stages * 2
            ]
            # CG1 -> epilogue warp
            o_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_o_stages * 2
            ]
            # CG0 -> CG1 (group order)
            group_order_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_group_order_stages * 2
            ]
            # TMEM allocation token
            tmem_holding_buf: cutlass.Int32
            # SMEM tensor buffers (aligned, in SMEM layout order)
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(q_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # sE: identical to sK (same stages, sizes, layout).
            sE: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(k_smem_layout_staged)],
                self.buffer_align_bytes,
            ]

            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # A_inv result, then overwritten with fp16 NV
            sAinv: cute.struct.Align[
                cute.struct.MemRange[
                    self.io_dtype, cute.cosize(ainv_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # W_qk scores
            sQk: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(qk_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(o_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            # Gate buffer [BT, d_k] fp32 - placed last in SMEM.  Raw gate
            # TMA-lands here, then overwritten in place with cumprod.  TMA-aligned.
            cumprod: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, cute.cosize(gate_raw_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            # cumprod_total = last-token cumprod per d_k channel (separate
            # buffer, not aliased with sGate).  CG0 writes, CG1 reads.
            cumprod_total: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, self.d_k * self.smem_cumprod_total_stages
                ],
                self.buffer_align_bytes,
            ]
            # beta (erase) and w (write) gates are bf16.  TMA-aligned.
            beta: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(beta_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            w: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.BFloat16, cute.cosize(w_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Build TMA atoms
        # ------------------------------------------------------------------
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])

        tma_q = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                q,
                q_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_k = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                k,
                k_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_v = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                v,
                v_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv_ss,
                self.cluster_layout_vmnk.shape,
            )
        )

        beta_smem_layout = cute.select(beta_smem_layout_staged, mode=[0, 1, 2])

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.mma_tiler_qkv[:2],
            )
        )

        # beta B-operand TMA atom (mirrors K) -> lands in the swizzled SMEM layout.
        tma_beta = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                beta,
                beta_smem_layout,
                self.mma_tiler_qk,
                tiled_mma_qk,
                self.cluster_layout_vmnk.shape,
            )
        )
        w_smem_layout = cute.select(w_smem_layout_staged, mode=[0, 1, 2])
        tma_w = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_A(
                tma_load_op,
                w,
                w_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv_ss,
                self.cluster_layout_vmnk.shape,
            )
        )

        # raw gate TMA load atom (fp32, [BT, d_k]).
        gate_tma_smem_layout = cute.select(gate_raw_smem_layout_staged, mode=[0, 1])
        tma_gate = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_load_op,
                gate,
                gate_tma_smem_layout,
                (self.b_t, self.d_k),
            )
        )

        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)
        self.tma_beta_bytes = cute.size_in_bytes(cutlass.BFloat16, beta_smem_layout)
        self.tma_w_bytes = cute.size_in_bytes(cutlass.BFloat16, w_smem_layout)
        self.tma_gate_bytes = cute.size_in_bytes(cutlass.Float32, gate_tma_smem_layout)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        scheduler_params = GDNTileSchedulerParams(
            num_seqs=batch_size,
            num_q_heads=h_q,
            num_v_heads=h_v,
            is_GQA=self.is_GQA,
            is_persistent=self.is_persistent,
        )
        grid_shape = GDNTileScheduler.get_grid_shape(
            scheduler_params, self.max_active_clusters
        )

        self.kernel(
            tiled_mma_qk,
            tiled_mma_qs,
            tiled_mma_qkv,
            tiled_mma_qkv_ss,
            tiled_mma_kv,
            tma_q,
            tma_k,
            tma_v,
            tma_beta,
            tma_w,
            tma_gate,
            gate,
            beta,
            w,
            tma_o,
            cu_seqlens,
            s_in,
            s_out,
            s_checkpoints,
            cu_checkpoints,
            checkpoint_every_n_tokens,
            scale,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            v_smem_layout_staged,
            cumsumlog_smem_layout_staged,
            gate_raw_smem_layout_staged,
            beta_smem_layout_staged,
            w_smem_layout_staged,
            ainv_smem_layout_staged,
            qk_smem_layout_staged,
            o_smem_layout_staged,
            scheduler_params,
            q,
            k,
            v,
            o,
            tensormap_workspace,
        ).launch(
            grid=grid_shape,
            block=(self.threads_per_cta, 1, 1),
            cluster=self.cluster_shape_mnk,
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
        )

    # -----------------------------------------------------------------------
    # Device kernel
    # -----------------------------------------------------------------------

    @cute.kernel
    def kernel(
        self,
        # Tiled MMAs (one per logical GEMM group)
        # GEMM 1 (kk) + GEMM 2 (qk)
        tiled_mma_qk: cute.TiledMma,
        # GEMM 3 (k*state) + GEMM 4 (q*state)
        tiled_mma_qs: cute.TiledMma,
        # GEMM 5 (new_v) + GEMM 6 (qkv)
        tiled_mma_qkv: cute.TiledMma,
        # GEMM 5 (new_v: A_inv@V) first tile
        tiled_mma_qkv_ss: cute.TiledMma,
        # GEMM 7 (kv_update)
        tiled_mma_kv: cute.TiledMma,
        # TMA descriptors and cute tensors
        tma_q: TmaInfo,
        tma_k: TmaInfo,
        tma_v: TmaInfo,
        tma_beta: TmaInfo,
        tma_w: TmaInfo,
        tma_gate: TmaInfo,
        mGate: cute.Tensor,
        mBeta: cute.Tensor,
        mW: cute.Tensor,
        tma_o: TmaInfo,
        # (B+1,)  int32  cumulative seq lengths
        cu_seqlens: cute.Tensor,
        # initial state (fp32) from GMEM; None if not used
        mS_init: Optional[cute.Tensor],
        # final state output (fp32) to GMEM; None if not stored
        mS_out: Optional[cute.Tensor],
        mS_checkpoints: Optional[cute.Tensor],
        cu_checkpoints: Optional[cute.Tensor],
        checkpoint_every_n_tokens: cutlass.Int32,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.ComposedLayout,
        gate_raw_smem_layout_staged: cute.ComposedLayout,
        beta_smem_layout_staged: cute.ComposedLayout,
        w_smem_layout_staged: cute.ComposedLayout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        # Scheduler
        scheduler_params: GDNTileSchedulerParams,
        # TMA descriptor workspace in GMEM (one set of 5 slots per CTA)
        # Slots: Q=0, K[0]=1, K[1]=2, V=3, O=4  (each slot = bytes_per_tensormap = 16xInt64)
        mQ,
        mK,
        mV,
        # used for TMA descriptor update
        mO,
        # (num_ctas, 3+smem_k_stages, 16) Int64
        tensormap_workspace: cute.Tensor,
    ):
        """
        Main GDN chunked kernel.

        Warp specialization is the outermost control flow: each warp role owns
        its own persistent tile-scheduler loop, iterating over (batch, head)
        tiles and then over chunks within each tile.
        """
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.use_initial_state):
            assert mS_init is not None, (
                "mS_init must be provided if use_initial_state is True"
            )
        else:
            assert mS_init is None, "mS_init must be None if use_initial_state is False"
        if cutlass.const_expr(self.store_final_state):
            assert mS_out is not None, (
                "mS_out must be provided if store_final_state is True"
            )
        else:
            assert mS_out is None, "mS_out must be None if store_final_state is False"

        # ------------------------------------------------------------------
        # TMA descriptor GMEM workspace - one set of 5 ptrs per CTA
        # Slots: Q=0, K[0]=1, K[1]=2, V=3, O=4
        # ------------------------------------------------------------------
        cta_linear_idx = bidz * grid_dim[1] * grid_dim[0] + bidy * grid_dim[0] + bidx

        tensormap_manager = TensorMapManager(
            TensorMapUpdateMode.GMEM, self.bytes_per_tensormap
        )

        tensormap_workspace = self.initialize_workspace(tensormap_workspace, grid_dim)
        tensormap_q_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 0, None)].iterator
        )
        tensormap_k_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 1, None)].iterator
        )
        tensormap_v_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 2, None)].iterator
        )
        tensormap_o_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 3, None)].iterator
        )
        tensormap_beta_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 4, None)].iterator
        )
        tensormap_w_ptr = tensormap_manager.get_tensormap_ptr(
            tensormap_workspace[(cta_linear_idx, 5, None)].iterator
        )

        # ------------------------------------------------------------------
        # 1. Allocate SMEM / TMEM, prefetch TMA descriptors
        # ------------------------------------------------------------------
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sK_trans = storage.sK.get_tensor(
            k_trans_smem_layout_staged.outer, swizzle=k_trans_smem_layout_staged.inner
        )
        # sE: identical to sK (same layouts), separate buffer.
        sE = storage.sE.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        # A_inverse / new_v  (A_inv written first, then overwritten with fp16 NV)
        sAinv = storage.sAinv.get_tensor(
            ainv_smem_layout_staged.outer, swizzle=ainv_smem_layout_staged.inner
        )
        # QK output / O store  (W_qk first, then O epilogue staging)
        sQk = storage.sQk.get_tensor(
            qk_smem_layout_staged.outer, swizzle=qk_smem_layout_staged.inner
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )
        # ONE fp32 buffer: raw gate (TMA-loaded) and cumprod alias the same storage.
        sCumprod = storage.cumprod.get_tensor(
            cumsumlog_smem_layout_staged.outer,
            swizzle=cumsumlog_smem_layout_staged.inner,
        )
        # sGate is the same swizzled view: the TMA lands the raw gate through
        # the identical mapping, and CG0 overwrites it in place with cumsumlog.
        sGate = sCumprod
        # cumprod_total (per d_k channel) -- separate buffer; CG0 writes, CG1 reads.
        cumprod_total_smem_layout_staged = cute.make_layout(
            (self.d_k, self.smem_cumprod_total_stages)
        )
        sCumprodTotal = storage.cumprod_total.get_tensor(
            cumprod_total_smem_layout_staged
        )
        sBeta = storage.beta.get_tensor(
            beta_smem_layout_staged.outer, swizzle=beta_smem_layout_staged.inner
        )
        sW = storage.w.get_tensor(
            w_smem_layout_staged.outer, swizzle=w_smem_layout_staged.inner
        )

        if warp_idx == self.mma_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_beta.atom)
            cpasync.prefetch_descriptor(tma_w.atom)
            cpasync.prefetch_descriptor(tma_gate.atom)
            cpasync.prefetch_descriptor(tma_o.atom)

        # TMEM allocator object - CG1 will issue the actual allocation
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            # Correction warp is the last one that accesses tmem
            allocator_warp_id=self.compute_group_1_warp_ids[0],
        )

        # ------------------------------------------------------------------
        # mbarrier-based pipelines
        # Each pipeline is created by all threads; barrier_storage points into SMEM.
        # defer_sync=True means pipeline_init_arrive() flushes all at once below.
        # ------------------------------------------------------------------
        def _cg(num_threads):
            return pipeline.CooperativeGroup(pipeline.Agent.Thread, num_threads)

        # 1 thread (TMA issuer)
        cg_tma = _cg(len([self.tma_qkv_warp_id]))
        # 1 thread (UMMA issuer)
        cg_mma = _cg(len([self.mma_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # 128 threads (CG1)
        cg_cg1 = _cg(self.threads_per_warp * len(self.compute_group_1_warp_ids))
        # 4 threads (one per CG1 warp, used for V load signaling)
        cg_cg1_v = _cg(len(self.compute_group_1_warp_ids))
        # one per CG0 warp (TMA-pipeline consumer signaling for beta/w, mirrors cg_cg1_v)
        cg_cg0_v = _cg(len(self.compute_group_0_warp_ids))
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # TMA load -> MMA:  K (double-buffered), Q, V
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # E (double-buffered; mirror of K) -- full/empty barriers for sE.
        load_e_producer, load_e_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_e_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # TMA warp -> CG0  beta (own pipeline, independent of w)
        load_beta_producer, load_beta_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_beta_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_v,  # warp-count (mirrors V's cg_cg1_v)
            tx_count=self.tma_beta_bytes,
            barrier_storage=storage.load_beta_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_w_producer, load_w_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_beta_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg1_v,  # consumed in CG1 (w applied elementwise to V)
            tx_count=self.tma_w_bytes,
            barrier_storage=storage.load_w_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # TMA warp -> CG0  raw gate (CG0 computes per-channel cumsum)
        load_rawgate_producer, load_rawgate_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_v,  # warp-count (CG0 only)
            tx_count=self.tma_gate_bytes,
            barrier_storage=storage.load_rawgate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> CG1:  cumprod_ready
        cumprod_ready_producer, cumprod_ready_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_cumprod_total_stages,
            producer_group=cg_cg0,
            consumer_group=cg_cg1,
            barrier_storage=storage.cumprod_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG1:  q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg1,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> CG0/CG1:  shared_acc
        shared_acc_producer, shared_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_shared_acc_stages,
            producer_group=cg_mma,
            consumer_group=cg_cg0,
            barrier_storage=storage.shared_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  q_gamma_done (CG0 writes q_gamma in sQ)
        q_gamma_ready_producer, q_gamma_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.smem_q_stages,
                producer_group=cg_cg0,
                consumer_group=cg_mma,
                barrier_storage=storage.q_gamma_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG0 -> MMA warp:  e_done (CG0 writes E in sE)
        e_ready_producer, e_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.e_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  k_bar_done (CG0 writes k_bar in sK)
        k_bar_ready_producer, k_bar_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma,
            barrier_storage=storage.k_bar_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG1 -> MMA warp:  state_inp_ready
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> MMA warp:  shared_inp_ready
        shared_inp_ready_producer, shared_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_shared_inp_stages,
                producer_group=cg_cg1,
                consumer_group=cg_mma,
                barrier_storage=storage.shared_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # CG1 -> epilogue warp:  output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_cg1,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        group_order_producer, group_order_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_group_order_stages,
            producer_group=cg_cg0,
            consumer_group=cg_cg1,
            barrier_storage=storage.group_order_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        # Peek-only clones for CG0: wait()/advance() in lockstep, never release
        # (MMA stays the sole consumer that frees Q / sE).
        load_q_consumer_cg0 = load_q_consumer.clone()
        load_k_consumer_cg0 = load_k_consumer.clone()

        # ------------------------------------------------------------------
        # 2. Warp specialization - each warp role owns its own scheduler loop
        # ------------------------------------------------------------------

        # ==============================================================
        # COMPUTE WARP GROUP 0 (warps 0-3)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_0_warp_ids[0]
            and warp_idx <= self.compute_group_0_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                sQk_pisl = self._transform_to_position_independent_layout(
                    sQk, qk_smem_layout_staged.inner
                )
                # First chunk: no previous state (S_prev = 0), skip GEMMs 3/4
                for chunk_idx in cutlass.range(num_chunks_b):
                    (
                        load_rawgate_consumer,
                        load_beta_consumer,
                        shared_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                        group_order_producer,
                        load_k_consumer_cg0,
                        load_q_consumer_cg0,
                        e_ready_producer,
                        k_bar_ready_producer,
                        q_gamma_ready_producer,
                        cumprod_ready_producer,
                    ) = self.compute_group_0(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_qk,),
                        (sBeta, sAinv, sQk_pisl, sCumprod, sCumprodTotal, sE, sK, sQ),
                        (
                            load_rawgate_consumer,
                            load_beta_consumer,
                            shared_acc_consumer,
                            a_inv_ready_producer,
                            qk_ready_producer,
                            group_order_producer,
                            load_k_consumer_cg0,
                            load_q_consumer_cg0,
                            e_ready_producer,
                            k_bar_ready_producer,
                            q_gamma_ready_producer,
                            cumprod_ready_producer,
                        ),
                        (chunk_idx == 0, seqlen_b - chunk_idx * self.b_t),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()
            group_order_producer.tail()
            e_ready_producer.tail()
            k_bar_ready_producer.tail()
            q_gamma_ready_producer.tail()
            cumprod_ready_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 1 (warps 4-7)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            # Total TMEM columns: state(128) + q_state_acc(128) + shared_acc(128x2) = 512
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                checkpoint_offset = 0
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                if num_chunks_b > 0:
                    if cutlass.const_expr(self.enable_checkpoints):
                        checkpoint_offset = cu_checkpoints[batch_idx]
                    if cutlass.const_expr(self.use_initial_state):
                        kv_acc_producer = self._load_initial_state(
                            tidx,
                            mS_init,
                            head_idx,
                            batch_idx,
                            tmem_ptr,
                            tiled_mma_kv,
                            kv_acc_producer,
                        )
                    (
                        load_v_consumer,
                        load_w_consumer,
                        cumprod_ready_consumer,
                        shared_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        group_order_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_idx,
                    ) = self.compute_group_1(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sW,
                            sCumprodTotal,
                            sBeta,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_w_consumer,
                            cumprod_ready_consumer,
                            shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            group_order_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (True, 0, head_idx),
                    )
                for chunk_idx in cutlass.range(1, num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    (
                        load_v_consumer,
                        load_w_consumer,
                        cumprod_ready_consumer,
                        shared_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        group_order_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                    ) = self.compute_group_1(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sW,
                            sCumprodTotal,
                            sBeta,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_w_consumer,
                            cumprod_ready_consumer,
                            shared_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            group_order_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (False, chunk_idx, head_idx),
                    )
                if num_chunks_b > 0:
                    if cutlass.const_expr(
                        self.store_final_state or self.enable_checkpoints
                    ):
                        kv_acc_consumer = self._store_final_state(
                            tidx,
                            mS_out,
                            head_idx,
                            batch_idx,
                            tmem_ptr,
                            tiled_mma_kv,
                            kv_acc_consumer,
                            seqlen_b,
                            mS_checkpoints,
                            checkpoint_offset,
                            checkpoint_every_n_tokens,
                        )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            shared_inp_ready_producer.tail()
            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # MMA WARP (warp 8)
        # ==============================================================
        elif warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                seqlen_b = cu_seqlens[batch_idx + 1] - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                # First chunk: no previous state (S_prev = 0), skip GEMMs 3/4.
                chunk_offset = batch_start
                if num_chunks_b > 0:
                    (
                        shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                        q_gamma_ready_consumer,
                        e_ready_consumer,
                        k_bar_ready_consumer,
                    ) = self.mma_warp(
                        tmem_ptr,
                        (
                            tiled_mma_qk,
                            tiled_mma_qs,
                            tiled_mma_qkv,
                            tiled_mma_qkv_ss,
                            tiled_mma_kv,
                        ),
                        (sQ, sE, sK_trans, sV, sAinv, sQk, sK),
                        (
                            shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            load_v_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            shared_inp_ready_consumer,
                            q_gamma_ready_consumer,
                            e_ready_consumer,
                            k_bar_ready_consumer,
                        ),
                        (True,),
                    )
                # Main loop: chunks 1..num_chunks_b-1 with previous state.
                for chunk_idx in cutlass.range(1, num_chunks_b):  # noqa: B007
                    (
                        shared_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                        q_gamma_ready_consumer,
                        e_ready_consumer,
                        k_bar_ready_consumer,
                    ) = self.mma_warp(
                        tmem_ptr,
                        (
                            tiled_mma_qk,
                            tiled_mma_qs,
                            tiled_mma_qkv,
                            tiled_mma_qkv_ss,
                            tiled_mma_kv,
                        ),
                        (sQ, sE, sK_trans, sV, sAinv, sQk, sK),
                        (
                            shared_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            load_v_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            shared_inp_ready_consumer,
                            q_gamma_ready_consumer,
                            e_ready_consumer,
                            k_bar_ready_consumer,
                        ),
                        (False,),
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            shared_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # ==============================================================
        # TMA LOAD WARP (warp 9)
        # ==============================================================
        elif warp_idx == self.tma_qkv_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init base descriptors once into GMEM (copies embedded atom descriptor)
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_q.atom, tensormap_q_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_k.atom, tensormap_k_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_v.atom, tensormap_v_ptr, self.tma_qkv_warp_id
                )
                # beta uses its embedded descriptor; w uses a per-tile GMEM tensormap
                tensormap_manager.init_tensormap_from_atom(
                    tma_w.atom, tensormap_w_ptr, self.tma_qkv_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                # Build bounded tensors: same ptr/strides, token dim capped to batch_end
                bounded_q = cute.make_tensor(
                    mQ.iterator,
                    cute.make_layout(
                        (batch_end, mQ.shape[1], mQ.shape[2]),
                        stride=(mQ.stride[0], mQ.stride[1], mQ.stride[2]),
                    ),
                )
                bounded_k = cute.make_tensor(
                    mK.iterator,
                    cute.make_layout(
                        (batch_end, mK.shape[1], mK.shape[2]),
                        stride=(mK.stride[0], mK.stride[1], mK.stride[2]),
                    ),
                )
                bounded_v = cute.make_tensor(
                    mV.iterator,
                    cute.make_layout(
                        (mV.shape[0], batch_end, mV.shape[2]),
                        stride=(mV.stride[0], mV.stride[1], mV.stride[2]),
                    ),
                )
                # w is value-major [DV, token] like V: token is mode 1.
                bounded_w = cute.make_tensor(
                    mW.iterator,
                    cute.make_layout(
                        (mW.shape[0], batch_end, mW.shape[2]),
                        stride=(mW.stride[0], mW.stride[1], mW.stride[2]),
                    ),
                )
                if num_chunks_b > 0:
                    # Update K/Q/V + w descriptors (beta uses its embedded descriptor)
                    tensormap_manager.update_tensormap(
                        (bounded_q, bounded_k, bounded_v, bounded_w),
                        (tma_q.atom, tma_k.atom, tma_v.atom, tma_w.atom),
                        (
                            tensormap_q_ptr,
                            tensormap_k_ptr,
                            tensormap_v_ptr,
                            tensormap_w_ptr,
                        ),
                        self.tma_qkv_warp_id,
                        (None, None, None, None),
                    )

                for chunk_idx in cutlass.range(num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    (
                        load_q_producer,
                        load_k_producer,
                        load_v_producer,
                        load_beta_producer,
                        load_w_producer,
                        load_rawgate_producer,
                    ) = self.tma_qkv_warp(
                        (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                        (tma_q, tma_k, tma_v, tma_beta, tma_w, tma_gate),
                        # K loads into sE (sE takes the place of sK).
                        (sQ, sE, sV, sBeta, sW, sGate),
                        (
                            load_q_producer,
                            load_k_producer,
                            load_v_producer,
                            load_beta_producer,
                            load_w_producer,
                            load_rawgate_producer,
                        ),
                        (chunk_offset, chunk_idx, batch_idx, head_idx),
                        (
                            tensormap_manager,
                            tensormap_q_ptr,
                            tensormap_k_ptr,
                            tensormap_v_ptr,
                            tensormap_beta_ptr,
                            tensormap_w_ptr,
                        ),
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            load_beta_producer.tail()
            load_w_producer.tail()
            load_rawgate_producer.tail()

        # ==============================================================
        # GATE/BETA LOAD WARP (warp 10) -- IDLE (raw gate loaded by warp 9, cumsum in CG0)
        # ==============================================================
        if warp_idx == self.load_gate_beta_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
        # ==============================================================
        # EPILOGUE WARP (warp 11)
        # ==============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params, (bidx, bidy, bidz), grid_dim
            )
            work = scheduler.initial_work_tile_info()

            # Init O descriptor once into GMEM
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                batch_idx, head_idx, _ = work.tile_idx
                batch_start = cu_seqlens[batch_idx]
                batch_end = cu_seqlens[batch_idx + 1]
                seqlen_b = batch_end - batch_start
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                # Build bounded O tensor: token dim capped to batch_end
                bounded_o = cute.make_tensor(
                    mO.iterator,
                    cute.make_layout(
                        (mO.shape[0], batch_end, mO.shape[2]),
                        stride=(mO.stride[0], mO.stride[1], mO.stride[2]),
                    ),
                )

                if num_chunks_b > 0:
                    # Update O descriptor independently
                    tensormap_manager.update_tensormap(
                        (bounded_o,),
                        (tma_o.atom,),
                        (tensormap_o_ptr,),
                        self.epilogue_warp_id,
                        (None,),
                    )
                    tensormap_manager.fence_tensormap_update(tensormap_o_ptr)

                for chunk_idx in cutlass.range(num_chunks_b):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    o_store_consumer = self.epilogue_warp(
                        (sO,),
                        (tma_o,),
                        (o_store_consumer,),
                        (head_idx, chunk_offset),
                        (tensormap_manager, tensormap_o_ptr),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

    # -----------------------------------------------------------------------
    # Per-warp methods  (called from kernel's chunk loop)
    # -----------------------------------------------------------------------
    @cute.jit
    def tma_qkv_warp(
        self,
        mma_args: tuple,
        tma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        tensormap_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warp 9: load Q, K (double-buffered), V, beta, w, raw gate for the chunk.

        Pattern (following fmha.py / dense_gemm_persistent.py):
          1. domain_offset the TMA tensor to (chunk_offset, head_idx, 0) so that
             the logical tile (0, ...) maps to the current chunk.
          2. flat_divide to obtain the tiled global view.
          3. thr_mma.partition_{A,B} to get the TMA-compatible per-thread view.
          4. cpasync.tma_partition -> (tXsX, tXgX) SMEM/global pairs.
          5. acquire pipeline stage, issue cute.copy, signal mbarrier.

        Note on head coordinate: head_idx is the flat KV-head index in [0, h_qv).
        For the hierarchical head layout (h_r, h_qv) with h_r having stride 0
        (broadcast), the flat index maps correctly as long as head_idx < h_qv.
        """
        tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv = mma_args
        tma_q, tma_k, tma_v, tma_beta, tma_w, tma_gate = tma_args
        sQ, sK, sV, sBeta, sW, sGate = smem_args
        (
            load_q_producer,
            load_k_producer,
            load_v_producer,
            load_beta_producer,
            load_w_producer,
            load_rawgate_producer,
        ) = pipeline_args
        chunk_offset, chunk_idx, batch_idx, head_idx = work_args
        (
            tensormap_manager,
            tensormap_q_ptr,
            tensormap_k_ptr,
            tensormap_v_ptr,
            tensormap_beta_ptr,
            tensormap_w_ptr,
        ) = tensormap_args

        # Single-CTA mode: no multicast, cta_v = 0.
        cta_layout = cute.make_layout(1)

        # Per-thread MMA slices (cta_v=0 for ONE-CTA mode).
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_qkv_ss = tiled_mma_qkv_ss.get_slice(0)
        thr_mma_kv = tiled_mma_kv.get_slice(0)  # noqa: F841

        # Tile shapes from the MMA tiler (128, 128, 128):
        #   mode[0,2] = (BT, DK) - M,K tile for A (Q) and for B (K) of tiled_mma_qk
        #   mode[1,2] = (BT, DV) - tile shape for loading V (B operand in GEMMs 5/6)
        # (BT, DK)
        qk_tile = cute.select(self.mma_tiler_qk, mode=[0, 2])
        # (DV, BT)
        v_tile = cute.select(self.mma_tiler_qkv, mode=[0, 2])

        # ------------------------------------------------------------------
        # K  (B operand of GEMM-kk / GEMM-qk, double-buffered)
        # Tensor shape: (total_tokens, H_hier, DK)
        # TMA tile:     (BT, DK)
        # ------------------------------------------------------------------
        mK = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_k.tma_tensor[None, None, head_idx]
        )
        # (..., num_k_tiles, ...)
        gK = cute.flat_divide(mK, qk_tile)
        tCgK = thr_mma_qk.partition_B(gK)
        tKsK, tKgK = cpasync.tma_partition(
            tma_k.atom,
            0,
            cta_layout,
            cute.group_modes(sK, 0, 3),
            cute.group_modes(tCgK, 0, 3),
        )

        # Load K for the current chunk into the next available pipeline stage.
        k_handle = load_k_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_k_ptr)

        cute.copy(
            tma_k.atom,
            tKgK[(None, 0, 0)],
            tKsK[(None, k_handle.index)],
            tma_bar_ptr=k_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_k_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # Q  (A operand of GEMM-qk, single-buffered)
        # ------------------------------------------------------------------
        mQ = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_q.tma_tensor[None, None, head_idx]
        )
        gQ = cute.flat_divide(mQ, qk_tile)
        tCgQ = thr_mma_qk.partition_A(gQ)
        tQsQ, tQgQ = cpasync.tma_partition(
            tma_q.atom,
            0,
            cta_layout,
            cute.group_modes(sQ, 0, 3),
            cute.group_modes(tCgQ, 0, 3),
        )

        q_handle = load_q_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_q_ptr)
        cute.copy(
            tma_q.atom,
            tQgQ[(None, 0, 0)],
            tQsQ[(None, q_handle.index)],
            tma_bar_ptr=q_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_q_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # V  (B operand of GEMM-new_v / GEMM-qkv, single-buffered)
        # ------------------------------------------------------------------
        mV = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset), tma_v.tma_tensor[None, None, head_idx]
        )
        gV = cute.flat_divide(mV, v_tile)
        tCgV = thr_mma_qkv_ss.partition_A(gV)
        tVsV, tVgV = cpasync.tma_partition(
            tma_v.atom,
            0,
            cta_layout,
            cute.group_modes(sV, 0, 3),
            cute.group_modes(tCgV, 0, 3),
        )

        v_handle = load_v_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_v_ptr)
        cute.copy(
            tma_v.atom,
            tVgV[(None, 0, 0)],
            tVsV[(None, v_handle.index)],
            tma_bar_ptr=v_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_v_ptr, cute.AddressSpace.generic
            ),
        )

        # ------------------------------------------------------------------
        # raw gate (decay, d_k): channel-wise [BT, d_k] fp32 tile TMA-loaded into
        # sGate FIRST (before beta/w) -- CG0's per-channel cumsum is critical-path.
        # ------------------------------------------------------------------
        gate_tile = (self.b_t, self.d_k)
        mGate_t = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_gate.tma_tensor[None, None, head_idx]
        )
        gGate = cute.flat_divide(mGate_t, gate_tile)
        tGsG, tGgG = cpasync.tma_partition(
            tma_gate.atom,
            0,
            cta_layout,
            cute.group_modes(sGate, 0, 2),
            cute.group_modes(gGate, 0, 2),
        )
        rawgate_handle = load_rawgate_producer.acquire_and_advance()
        cute.copy(
            tma_gate.atom,
            tGgG[(None, 0, 0)],
            tGsG[(None, rawgate_handle.index)],
            tma_bar_ptr=rawgate_handle.barrier,
        )

        # ------------------------------------------------------------------
        # beta (erase, d_k) and w (write, d_v): channel-wise [BT, d] tiles TMA-loaded
        # into sBeta / sW.
        # ------------------------------------------------------------------
        # beta mirrors K: B-operand partition over the (BT, DK) tile.
        mBeta_t = cute.domain_offset(
            (chunk_offset, cutlass.Int32(0)), tma_beta.tma_tensor[None, None, head_idx]
        )
        gBeta = cute.flat_divide(mBeta_t, qk_tile)
        tCgBeta = thr_mma_qk.partition_B(gBeta)
        tBsB, tBgB = cpasync.tma_partition(
            tma_beta.atom,
            0,
            cta_layout,
            cute.group_modes(sBeta, 0, 3),
            cute.group_modes(tCgBeta, 0, 3),
        )

        # w mirrors V exactly: value-major [DV, token], A-operand partition.
        mW_t = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset), tma_w.tma_tensor[None, None, head_idx]
        )
        gW = cute.flat_divide(mW_t, v_tile)
        tCgW = thr_mma_qkv_ss.partition_A(gW)
        tWsW, tWgW = cpasync.tma_partition(
            tma_w.atom,
            0,
            cta_layout,
            cute.group_modes(sW, 0, 3),
            cute.group_modes(tCgW, 0, 3),
        )

        # beta and w each on their own pipeline/barrier (independent).
        beta_handle = load_beta_producer.acquire_and_advance()
        cute.copy(
            tma_beta.atom,
            tBgB[(None, 0, 0)],
            tBsB[(None, beta_handle.index)],
            tma_bar_ptr=beta_handle.barrier,
        )
        w_handle = load_w_producer.acquire_and_advance()
        if chunk_idx == 0:
            tensormap_manager.fence_tensormap_update(tensormap_w_ptr)
        cute.copy(
            tma_w.atom,
            tWgW[(None, 0, 0)],
            tWsW[(None, w_handle.index)],
            tma_bar_ptr=w_handle.barrier,
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_w_ptr, cute.AddressSpace.generic
            ),
        )

        return (
            load_q_producer,
            load_k_producer,
            load_v_producer,
            load_beta_producer,
            load_w_producer,
            load_rawgate_producer,
        )

    @cute.jit
    def mma_warp(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Warp 8: issue all 7 GEMMs in dependency order."""
        tiled_mma_qk, tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv = (
            mma_args
        )
        sQ_gamma, sE, sK_trans, sV, sAinv, sQk, sK_bar = smem_args
        (
            shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            shared_inp_ready_consumer,
            q_gamma_ready_consumer,
            e_ready_consumer,
            k_bar_ready_consumer,
        ) = pipeline_args
        (is_first_chunk,) = work_args

        valid_state = not is_first_chunk or self.use_initial_state

        # ------------------------------------------------------------------
        # Build TMEM accumulator views
        # ------------------------------------------------------------------
        # Shared acc (GEMMs 1/2/3/5/6) - 2 stages, layout from tiled_mma_qk
        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_shared_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tCtAcc_fake.layout
        )

        shared_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(shared_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )

        # q*state acc (GEMM 4 only) - 1 stage, layout from tiled_mma_qs
        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )

        # state acc (GEMM 7 only) - 1 stage, layout from tiled_mma_kv
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        # dS scratch for GEMM 7: the raw update accumulates into the shared_acc
        # columns; CG1 then combines S += cumprod_total * dS.
        tCtDs = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tCtState_fake.layout
        )

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )

        # ------------------------------------------------------------------
        # Pre-create operand fragments (stage dim preserved; sliced at GEMM time)
        # tiled_mma_qk operands (GEMMs 1, 2)
        # E as A for GEMM 1 (kk = E @ K_bar^T)
        tCrK_A = tiled_mma_qk.make_fragment_A(sE)
        # K_bar as B for GEMMs 1+2
        tCrKbar_B = tiled_mma_qk.make_fragment_B(sK_bar)
        # Q_gamma as A for GEMM 2 (qk = Q_gamma @ K_bar^T)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ_gamma)
        # tiled_mma_qs operands (GEMMs 3, 4)
        # S_prev as A for GEMMs 3+4 (from TMEM)
        tCrS_A = tCtState_inp
        # Q_gamma as B for GEMM 4 (q*state = S_prev @ Q_gamma)
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ_gamma)
        # E as B for GEMM 3 (k*state = S_prev @ E)
        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sE)
        # tiled_mma_qkv operands (GEMMs 5, 6)
        # w*V - KS as A for GEMM 5 (staged to TMEM by CG1 in every chunk)
        tCrV_A = tCtShared_inp
        # A_inv as B for GEMM 5
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)

        # W_qkv as A for GEMM 6
        tCrQkv_A = tCtShared_inp
        # NV as B for GEMM 6
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        # tiled_mma_kv operands (GEMM 7)
        # K_bar^T as B for GEMM 7 (A = the NV shared_inp slot GEMM 6 consumes)
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)

        # ---- GEMM 1: kk  (E @ K_bar^T -> shared acc) ------------------------
        # M_kk = E @ K_bar^T, E (= cumprod*beta*K) and K_bar (= K/cumprod) from CG0.
        # Waits e_ready + k_bar_ready, not the raw-K load.
        e_ready_handle = e_ready_consumer.wait_and_advance()
        k_bar_ready_handle = k_bar_ready_consumer.wait_and_advance()
        k_handle = load_k_consumer.wait_and_advance()
        # Acquire kv_acc BEFORE KK^T: waits the previous chunk's combine to
        # finish reading dS out of the shared_acc columns.
        if cutlass.const_expr(self.use_initial_state and is_first_chunk):
            kv_acc_producer.advance()
        kv_acc_handle = kv_acc_producer.acquire_and_advance()

        kk_handle = shared_acc_producer.acquire_and_advance()

        num_kphases = cute.size(tCrK_A, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtShared[None, None, None, kk_handle.index],
                tCrK_A[None, None, kphase_idx, e_ready_handle.index],
                tCrKbar_B[None, None, kphase_idx, k_bar_ready_handle.index],
                tCtShared[None, None, None, kk_handle.index],
            )

        # Signal W_kk ready -> CG0 kk_epi
        kk_handle.commit()

        # ---- GEMM 2: qk  (q_gamma @ K_bar^T -> shared acc) ------------------
        # Intra-chunk scores: A[i,j] = q_gamma_i . K_bar_j (operands from CG0).
        q_gamma_handle = q_gamma_ready_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()
        qk_handle = shared_acc_producer.acquire_and_advance()

        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtShared[None, None, None, qk_handle.index],
                tCrQ_A[None, None, kphase_idx, q_gamma_handle.index],
                tCrKbar_B[None, None, kphase_idx, k_bar_ready_handle.index],
                tCtShared[None, None, None, qk_handle.index],
            )

        # Signal W_qk ready -> CG0 qk_epi
        qk_handle.commit()

        # ---- GEMMs 3+4: k*state and q*state  (S_prev @ E / S_prev @ Q_gamma) ---
        # Skipped on the first chunk when use_initial_state is False.
        if valid_state:
            s_handle = state_inp_ready_consumer.wait_and_advance()

            # GEMM 3: S_prev @ E -> shared acc (K * state)
            ks_handle = shared_acc_producer.acquire_and_advance()
            num_kphases_qs = cute.size(tCrS_A, mode=[2])
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtShared[None, None, None, ks_handle.index],
                    tCrS_A[None, None, kphase_idx, s_handle.index],
                    tCrK_B_qs[None, None, kphase_idx, e_ready_handle.index],
                    tCtShared[None, None, None, ks_handle.index],
                )
            ks_handle.commit()
            # E fully consumed by GEMMs 1+3; release slot immediately.
            e_ready_handle.release()
            k_handle.release()

            # GEMM 4: S_prev @ Q_gamma -> tmem_q_state (Q * state)
            q_state_acc_handle = q_state_acc_producer.acquire_and_advance()
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtQState[None, None, None, q_state_acc_handle.index],
                    tCrS_A[None, None, kphase_idx, s_handle.index],
                    tCrQ_B_qs[None, None, kphase_idx, q_gamma_handle.index],
                    tCtQState[None, None, None, q_state_acc_handle.index],
                )
            q_state_acc_handle.commit()
            s_handle.release()
            # Q_gamma fully consumed by GEMM 4; release slot.
            q_gamma_handle.release()
            q_handle.release()
        else:
            # No valid state: E consumed only by GEMM 1; Q_gamma consumed only by GEMM 2.
            e_ready_handle.release()
            k_handle.release()
            q_gamma_handle.release()
            q_handle.release()

        # ---- GEMM 5: new_v  (A_inv @ (V-KS) -> shared acc) ------------------
        # A_inv from CG0 (a_inv_ready); V-KS from CG1 (v_ks_ready, stored in sV).
        vks_handle = shared_inp_ready_consumer.wait_and_advance()
        ainv_handle = a_inv_ready_consumer.wait_and_advance()
        nv_handle = shared_acc_producer.acquire_and_advance()

        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        cur_tiled_mma_qkv = tiled_mma_qkv
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            cur_tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                cur_tiled_mma_qkv,
                tCtShared[None, None, None, nv_handle.index],
                tCrV_A[None, None, kphase_idx, vks_handle.index],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtShared[None, None, None, nv_handle.index],
            )

        nv_handle.commit()
        ainv_handle.release()
        vks_handle.release()

        # ---- GEMM 7: kv_update  (delta @ K_bar^T -> dS scratch) -----------------
        # Issued BEFORE GEMM 6 so CG1's dS combine overlaps GEMM 6.
        # First chunk: zero-init on kphase 0. Subsequent chunks: always accumulate.
        qkv_nv_handle = shared_inp_ready_consumer.wait_and_advance()

        num_kphases_kv = cute.size(tCrKt_B, mode=[2])
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            # Fresh accumulation every chunk: dS is raw (pre-decay); CG1 folds
            # it into S with the per-channel chunk-total scale.
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_kv,
                tCtDs[None, None, None, kv_acc_handle.index],
                tCrQkv_A[None, None, kphase_idx, qkv_nv_handle.index],
                tCrKt_B[None, None, kphase_idx, k_bar_ready_handle.index],
                tCtDs[None, None, None, kv_acc_handle.index],
            )
        kv_acc_handle.commit()
        # After KV MMA: free k_bar (CG0->MMA).
        k_bar_ready_handle.release()

        # ---- GEMM 6: qkv  (W_qkv @ NV -> q * state acc) ------------------------
        # W_qkv from CG0 (qk_ready, stored in sQk); NV from the same shared_inp
        # slot GEMM 7 read (released after this GEMM, its last consumer).
        qkv_qk_handle = qk_ready_consumer.wait_and_advance()
        q_state_acc_handle = q_state_acc_producer.acquire_and_advance()

        num_kphases_qkv = cute.size(tCrQkv_A, mode=[2])
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            tiled_mma_qkv.set(
                tcgen05.Field.ACCUMULATE, valid_state or (kphase_idx != 0)
            )
            cute.gemm(
                tiled_mma_qkv,
                tCtQState[None, None, None, q_state_acc_handle.index],
                tCrQkv_A[None, None, kphase_idx, qkv_nv_handle.index],
                tCrNv_B[None, None, kphase_idx, qkv_qk_handle.index],
                tCtQState[None, None, None, q_state_acc_handle.index],
            )

        qkv_qk_handle.release()
        qkv_nv_handle.release()
        q_state_acc_handle.commit()

        return (  # type: ignore[return-value]
            shared_acc_producer,
            q_state_acc_producer,
            kv_acc_producer,
            load_k_consumer,
            load_q_consumer,
            load_v_consumer,
            a_inv_ready_consumer,
            qk_ready_consumer,
            state_inp_ready_consumer,
            shared_inp_ready_consumer,
            q_gamma_ready_consumer,
            e_ready_consumer,
            k_bar_ready_consumer,
        )

    @cute.jit
    def compute_group_0(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 0-3: T-pairwise, kk_epi, inverse, qk_epi."""
        (tiled_mma_qk,) = mma_args
        sBeta, sAinv, sQk, sCumprod, sCumprodTotal, sE, sK, sQ = smem_args
        (
            load_rawgate_consumer,
            load_beta_consumer,
            shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
            group_order_producer,
            load_k_consumer_cg0,
            load_q_consumer_cg0,
            e_ready_producer,
            k_bar_ready_producer,
            q_gamma_ready_producer,
            cumprod_ready_producer,
        ) = pipeline_args
        (is_first_chunk, valid_len) = work_args

        # ------------------------------------------------------------------
        # Preamble: per-thread ID within CG0 and TMEM copy setup
        # ------------------------------------------------------------------
        # Local thread ID within CG0 (0..127)
        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0

        # Build TMEM tensor view of shared_acc (both stages):
        #   shape = (per_thread_acc_shape..., tmem_shared_acc_stages)
        tAcc_shape = tiled_mma_qk.partition_shape_C(
            (self.mma_tiler_qk[0], self.mma_tiler_qk[1])
        )
        tAcc_wo_stages = tiled_mma_qk.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(
            tAcc_wo_stages.iterator,
            cute.flat_product(
                tAcc_wo_stages.layout,
                cute.make_layout((self.tmem_shared_acc_stages,), stride=(1,)),
            ),
        )
        tStS_staged = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tAcc.layout
        )
        tStS_staged_mn_view = self.transform_partitioned_tensor_layout(tStS_staged)
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        # TMEM load copy atom and tiled copy (loads fp32 accum from TMEM -> registers)
        tStS_for_t2r = tStS_staged[(None, None), 0, 0, 0]
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tStS_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg0_tidx)

        # Per-thread TMEM source (staged).
        # tTR_tStS shape: (A, B, NUM_SUBS, NUM_STAGES)
        # Each subtile is tTR_tStS[None, None, sub, stage] with shape (A, B).
        tTR_tStS = thr_shared_t2r.partition_S(tStS_staged_mn_view)
        tTR_tScS = thr_shared_t2r.partition_D(cS)

        sub_tile_size = 32

        # SMEM store copies: fp32 registers -> fp16 SMEM (A-operands for GEMM 5 and 6).
        atom_ainv_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_ainv_r2s = cute.make_tiled_copy_D(atom_ainv_r2s, tiled_shared_t2r)
        sAinv_mn_view = self.transform_partitioned_tensor_layout(sAinv)
        thr_ainv_r2s = tiled_ainv_r2s.get_slice(cg0_tidx)
        tCsAI = thr_ainv_r2s.partition_D(sAinv_mn_view)

        atom_qk_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_qk_r2s = cute.make_tiled_copy_D(atom_qk_r2s, tiled_shared_t2r)
        sQk_mn_view = self.transform_partitioned_tensor_layout(sQk)
        tCsQK = tiled_qk_r2s.get_slice(cg0_tidx).partition_D(sQk_mn_view)

        # ------------------------------------------------------------------
        # Gate-fold preamble: K/Q/E/K_bar SMEM<->register copies.
        # ------------------------------------------------------------------
        atom_inp_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        atom_inp_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_inp_s2r = cute.make_tiled_copy_D(atom_inp_s2r, tiled_shared_t2r)
        tiled_inp_r2s = cute.make_tiled_copy_D(atom_inp_r2s, tiled_shared_t2r)
        thr_inp_s2r = tiled_inp_s2r.get_slice(cg0_tidx)
        thr_inp_r2s = tiled_inp_r2s.get_slice(cg0_tidx)
        sE_mn_view = self.transform_partitioned_tensor_layout(sE)
        sK_mn_view = self.transform_partitioned_tensor_layout(sK)
        sQ_mn_view = self.transform_partitioned_tensor_layout(sQ)

        # ------------------------------------------------------------------
        # Step 1: gate cumsum.  CG0 consumes the raw gate, computes the
        # per-channel cumsumlog into sCumprod, then signals CG1.
        # ------------------------------------------------------------------
        gate_handle = load_rawgate_consumer.wait_and_advance()
        gate_stage = gate_handle.index

        rGate = cute.make_rmem_tensor((self.b_t,), self.acc_dtype)
        # Column copies: thread cg0_tidx owns channel column cg0_tidx (all 64
        # tokens), for both the raw-gate read and the cumsumlog write-back.
        tiled_cum_col = cute.make_tiled_copy_tv(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
            cute.make_layout((1, num_threads_cg0)),
            cute.make_layout((self.b_t, 1)),
        )
        thr_cum_col = tiled_cum_col.get_slice(cg0_tidx)
        tCsCumCol = thr_cum_col.partition_D(sCumprod[None, None, gate_stage])
        rGate_v = cute.make_tensor(rGate.iterator, cute.make_layout(tCsCumCol.shape))
        cute.copy(tiled_cum_col, tCsCumCol, rGate_v)
        # cumprod slot acquired up front (index only feeds the sCumprodTotal
        # store below; the slot was freed by CG1 two chunks ago): keeps the
        # outlined mbarrier spin from splitting the log2/chain stream and the
        # write-back into separate basic blocks mid-gate-pass.
        cumprod_ready_handle = cumprod_ready_producer.acquire_and_advance()
        # Tail mask BEFORE the log2 pass (cold branch: only each seq's last
        # chunk): 1.0 + 1e-10 == 1.0 in fp32, so log2 yields exactly 0.0 --
        # bitwise-identical to masking after -- keeping the hot loop below a
        # single straight-line block.
        if valid_len < self.b_t:
            for t in cutlass.range_constexpr(self.b_t):
                if t >= valid_len:
                    rGate[t] = 1.0
        # log2 stream with the serial prefix-sum chain interleaved: the chain
        # step for element t-_K issues alongside log2(t), so the whole chain
        # (~63 FADD latencies) hides inside the much longer MUFU.LG2 stream
        # (64 LG2s at quarter-rate) instead of serializing after it.  _K is
        # the lookahead covering the MUFU latency.
        # eps adds as packed FADD2 pairs: halves the eps instruction count;
        # the serial chain cannot pack (dependent links).
        for t in cutlass.range_constexpr(0, self.b_t, 2):
            e0, e1 = cute.arch.add_packed_f32x2(
                (rGate[t], rGate[t + 1]), (1e-10, 1e-10)
            )
            rGate[t] = e0
            rGate[t + 1] = e1
        _K = 8
        for t in cutlass.range_constexpr(_K + 1):
            rGate[t] = cute.math.log2(rGate[t], fastmath=True)
        for t in cutlass.range_constexpr(_K + 1, self.b_t):
            rGate[t] = cute.math.log2(rGate[t], fastmath=True)
            rGate[t - _K] = rGate[t - _K] + rGate[t - _K - 1]
        for t in cutlass.range_constexpr(self.b_t - _K, self.b_t):
            rGate[t] = rGate[t] + rGate[t - 1]
        # rGate holds CUMSUMLOG (L); consumers apply exp2(+L)/exp2(-L).
        # Write the cumsumlog back through the same swizzled view (in place over
        # the consumed raw gate; stays in this thread's own 32-chan band).
        cute.copy(tiled_cum_col, rGate_v, tCsCumCol)
        # Publish cumprod_total (last-token cumprod) for this channel to the separate
        # cumprod_total buffer (CG1 reads only this).
        sCumprodTotal[cg0_tidx, cumprod_ready_handle.index] = cute.math.exp2(
            rGate[self.b_t - 1], fastmath=True
        )

        # Intra-CG0 sync: K/Q cumprod scale below gathers other threads' channels, so
        # all CG0 threads must see all sCumprod writes first.
        self.cumprod_barrier.arrive_and_wait()

        # Fence the sCumprod / cumprod_total writes, then signal CG1.
        # gate_handle.release() is deferred to after the K/Q cumprod gather below.
        cute.arch.fence_view_async_shared()
        cumprod_ready_handle.commit()

        # ------------------------------------------------------------------
        # Gate folds, fragment form (LDSM/STSM I/O).  Per-slot (token, channel)
        # coordinates come from partition_D over an identity tensor.
        # ------------------------------------------------------------------
        cE_inp = cute.make_identity_tensor((self.b_t, self.d_k))
        tCcE = thr_inp_s2r.partition_D(cE_inp)
        n_slots = cute.size(tCcE)
        # Fragments are FLAT storage with a compact-colex view for the copies,
        # so register slot == tCcE[i]'s linear index.
        rCumprod = cute.make_rmem_tensor((n_slots,), self.acc_dtype)
        # Gather the cumsumlog in FRAGMENT order via a universal-atom copy built
        # on the same reference TV layout as the K/Q fragment copies: its D-side
        # order == tCcE's == the fragments', so rCumprod pairs by linear index.
        tiled_cum_s2r = cute.make_tiled_copy_D(
            cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), cutlass.Float32),
            tiled_shared_t2r,
        )
        tCsCumG = tiled_cum_s2r.get_slice(cg0_tidx).partition_S(
            sCumprod[None, None, gate_stage]
        )
        assert cute.size(tCsCumG) == n_slots
        rCumprod_v = cute.make_tensor(
            rCumprod.iterator, cute.make_layout(tCsCumG.layout.shape)
        )
        cute.copy(tiled_cum_s2r, tCsCumG, rCumprod_v)

        # Phase 1: K -> K_bar = K * exp2(-L)
        k_bar_ready_handle = k_bar_ready_producer.acquire_and_advance()

        # PEEK the K->sE TMA load (wait-only; never release).
        k_handle_cg0 = load_k_consumer_cg0.wait_and_advance()
        e_stage = k_handle_cg0.index

        tCsE = thr_inp_s2r.partition_S(sE_mn_view)[None, None, None, e_stage]
        _frag_view = cute.make_layout(tCsE.layout.shape)  # compact colex
        tRT_rK_flat = cute.make_rmem_tensor((n_slots,), self.io_dtype)
        tRT_rK = cute.make_tensor(tRT_rK_flat.iterator, _frag_view)
        cute.copy(tiled_inp_s2r, tCsE, tRT_rK)

        rKbar_flat = cute.make_rmem_tensor((n_slots,), self.io_dtype)
        rKbar = cute.make_tensor(rKbar_flat.iterator, _frag_view)
        tRT_rK_i32 = cute.recast_tensor(tRT_rK_flat, cutlass.Int32)
        rKbar_i32 = cute.recast_tensor(rKbar_flat, cutlass.Int32)
        for p in cutlass.range_constexpr(cute.size(tRT_rK_i32)):
            k0, k1 = _bf16x2_to_f32x2(tRT_rK_i32[p])
            e0 = cute.math.exp2(-rCumprod[2 * p], fastmath=True)
            e1 = cute.math.exp2(-rCumprod[2 * p + 1], fastmath=True)
            r0, r1 = cute.arch.mul_packed_f32x2((k0, k1), (e0, e1))
            rKbar_i32[p] = _f32x2_to_bf16x2(r0, r1)
        tCsK_d = thr_inp_r2s.partition_D(sK_mn_view)[None, None, None, e_stage]
        cute.copy(tiled_inp_r2s, rKbar, tCsK_d)

        cute.arch.fence_view_async_shared()
        k_bar_ready_handle.commit()
        # Raw-gate release deferred past the K_bar fold (register-dep order).
        gate_handle.release()

        # Phase 2: K -> E = exp2(L) * beta * K  (raw K still in tRT_rK)
        e_ready_handle = e_ready_producer.acquire_and_advance()

        beta_handle = load_beta_consumer.wait_and_advance()
        sBeta_mn_view = self.transform_partitioned_tensor_layout(sBeta)
        tCsBeta = thr_inp_s2r.partition_S(sBeta_mn_view)[
            None, None, None, beta_handle.index
        ]
        tRT_rBetaE_flat = cute.make_rmem_tensor((n_slots,), self.io_dtype)
        tRT_rBetaE = cute.make_tensor(tRT_rBetaE_flat.iterator, _frag_view)
        cute.copy(tiled_inp_s2r, tCsBeta, tRT_rBetaE)

        tRT_rBetaE_i32 = cute.recast_tensor(tRT_rBetaE_flat, cutlass.Int32)
        # exp2(+L) computed ONCE here and kept in registers for the q_gamma
        # fold below; recomputing it there would double the MUFU work.
        rScaleP = cute.make_rmem_tensor((n_slots,), self.acc_dtype)
        for p in cutlass.range_constexpr(cute.size(tRT_rK_i32)):
            k0, k1 = _bf16x2_to_f32x2(tRT_rK_i32[p])
            b0, b1 = _bf16x2_to_f32x2(tRT_rBetaE_i32[p])
            e0 = cute.math.exp2(rCumprod[2 * p], fastmath=True)
            e1 = cute.math.exp2(rCumprod[2 * p + 1], fastmath=True)
            rScaleP[2 * p] = e0
            rScaleP[2 * p + 1] = e1
            t0, t1 = cute.arch.mul_packed_f32x2((k0, k1), (e0, e1))
            r0, r1 = cute.arch.mul_packed_f32x2((t0, t1), (b0, b1))
            tRT_rK_i32[p] = _f32x2_to_bf16x2(r0, r1)
        tCsE_d = thr_inp_r2s.partition_D(sE_mn_view)[None, None, None, e_stage]
        cute.copy(tiled_inp_r2s, tRT_rK, tCsE_d)

        cute.arch.fence_view_async_shared()
        e_ready_handle.commit()
        # beta release deferred past the commit (register-dependency form).
        beta_handle.release()

        # Phase 3: Q -> q_gamma = exp2(L) * Q  (GEMM1 is already running)
        q_gamma_ready_handle = q_gamma_ready_producer.acquire_and_advance()

        # PEEK the Q TMA load (wait-only; never release).
        q_handle_cg0 = load_q_consumer_cg0.wait_and_advance()
        q_stage = q_handle_cg0.index

        tCsQ = thr_inp_s2r.partition_S(sQ_mn_view)[None, None, None, q_stage]
        tRT_rQ_flat = cute.make_rmem_tensor((n_slots,), self.io_dtype)
        tRT_rQ = cute.make_tensor(tRT_rQ_flat.iterator, _frag_view)
        cute.copy(tiled_inp_s2r, tCsQ, tRT_rQ)

        tRT_rQ_i32 = cute.recast_tensor(tRT_rQ_flat, cutlass.Int32)
        for p in cutlass.range_constexpr(cute.size(tRT_rQ_i32)):
            q0, q1 = _bf16x2_to_f32x2(tRT_rQ_i32[p])
            r0, r1 = cute.arch.mul_packed_f32x2(
                (q0, q1), (rScaleP[2 * p], rScaleP[2 * p + 1])
            )
            tRT_rQ_i32[p] = _f32x2_to_bf16x2(r0, r1)
        tCsQ_d = thr_inp_r2s.partition_D(sQ_mn_view)[None, None, None, q_stage]
        cute.copy(tiled_inp_r2s, tRT_rQ, tCsQ_d)

        cute.arch.fence_view_async_shared()
        q_gamma_ready_handle.commit()

        # decay + erase gate folded into the K operands before GEMM 1, so the
        # tril is applied explicitly here: 1.0 for i>=j (lower tri incl diagonal), else 0.0.
        rTril = cute.make_rmem_tensor((2, 16), self.acc_dtype)
        tGrTril = thr_shared_t2r.partition_D(rTril)
        for k in cutlass.range_constexpr(cute.size(tTR_tScS)):
            coord = tTR_tScS[k]
            tGrTril[k] = 1.0 if coord[0] >= coord[1] else 0.0

        # ------------------------------------------------------------------
        # Step 2: kk_epi - load W_kk (GEMM 1 result from TMEM) -> M_kk
        #   Depends on: shared_acc stage 0 (MMA warp GEMM 1 kk done)
        # ------------------------------------------------------------------
        # Acquire sAinvNv slot, fence, signal
        ainv_handle = a_inv_ready_producer.acquire_and_advance()
        # Acquire sQkOstore slot, write W_qkv (fp16), fence, signal
        qk_ready_handle = qk_ready_producer.acquire_and_advance()
        group_order_handle = group_order_producer.acquire_and_advance()
        kk_handle = shared_acc_consumer.wait_and_advance()

        # tKKrKK: full-size register buffer for M_kk (inverse step), filled
        # subtile by subtile; tKKrKK[j] = M_kk[cg0_tidx, j] after the loop.
        tKKrKK = cute.make_rmem_tensor_like(tTR_tScS, self.acc_dtype)

        tKKrKK_out = cute.make_rmem_tensor_like(tKKrKK, self.io_dtype)
        tCrAI = tiled_ainv_r2s.retile(tKKrKK_out)
        for sub in cutlass.range(tKKrKK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, kk_handle.index],
                tKKrKK[None, 0, sub],
            )
            for k in cutlass.range(sub_tile_size):
                tKKrKK[k, 0, sub] = tKKrKK[k, 0, sub] * tGrTril[k, 0, sub]
            tKKrKK_out[None, 0, sub].store(
                tKKrKK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_ainv_r2s,
                tCrAI[None, 0, sub],
                tCsAI[None, 0, sub, ainv_handle.index],
            )
        # Release shared_acc stage 0 (CG1 also releases its side - collective
        # barrier).  The tril multiply above consumed the TMEM loads, so the
        # compiler scoreboard already orders them before this arrive.
        kk_handle.release()

        # ------------------------------------------------------------------
        # Step 3: qk_epi - load W_qk (GEMM 2 result from TMEM), scale -> W_qkv
        #   Depends on: shared_acc stage 1 (MMA warp GEMM 2 qk done)
        #   W_qk[i,j] *= T[i,j] * scale  where T[i,j] = exp2(cumsumlog[i]-cumsumlog[j])
        #   Same subtile structure as kk_epi (num_kk_subs / kk_sub_size reused).
        #   Done before inverse so that qk_ready signals GEMM 6 early, letting the
        #   MMA warp run GEMM 6 in parallel with the inverse computation below.
        # ------------------------------------------------------------------
        qk_handle = shared_acc_consumer.wait_and_advance()

        tStS_for_t2r = tStS_staged[(None, None), 0, 0, qk_handle.index]
        # tTR_tStS = thr_shared_t2r.partition_S(tStS_for_t2r)

        # tQKrQK: full-size register buffer for W_qkv (SMEM write), filled subtile by subtile.
        tQKrQK = cute.make_rmem_tensor_like(tTR_tScS, self.acc_dtype)

        # Convert fp32 tQKrQK -> fp16 and store to sQk, one subtile at a time.
        tQKrQK_out = cute.make_rmem_tensor_like(tQKrQK, self.io_dtype)
        tCrQK = tiled_qk_r2s.retile(tQKrQK_out)
        for sub in cutlass.range(tQKrQK.shape[2]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tStS[None, 0, sub, qk_handle.index],
                tQKrQK[None, 0, sub],
            )
            for k in cutlass.range(sub_tile_size):
                tQKrQK[k, 0, sub] = tQKrQK[k, 0, sub] * tGrTril[k, 0, sub] * scale
            tQKrQK_out[None, 0, sub].store(
                tQKrQK[None, 0, sub].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_qk_r2s, tCrQK[None, 0, sub], tCsQK[None, 0, sub, qk_handle.index]
            )
        cute.arch.fence_view_async_shared()
        # Release shared_acc stage 1 (loads consumed by the scale multiply above)
        qk_handle.release()
        group_order_handle.commit()

        qk_ready_handle.commit()

        # Advance past shared_acc stages CG0 does not read (KS/NV): they form
        # a collective barrier with CG1, so CG0 must advance for CG1 to proceed.
        shared_acc_consumer.advance()
        valid_state = not is_first_chunk or self.use_initial_state
        if valid_state:
            shared_acc_consumer.advance()

        # ------------------------------------------------------------------
        # Step 4: inverse - compute A_inv = (I + M_kk)^{-1}, write to sAinvNv
        #   Done after qk_epi so the inverse computation overlaps with GEMM 6
        #   (qkv) running in the MMA warp.
        # ------------------------------------------------------------------

        # -- Hierarchical blockwise inverse: A_inv = (I + M_kk)^{-1} ----------
        # Thread cg0_tidx owns row cg0_tidx of the BTxBT matrix.
        # sAinv is reinterpreted as row-major (BTxBT) fp16 for the algorithm;
        # the MMA-ready A-operand layout is written back via tiled_store_ainv after.
        # NOTE: assumes io_dtype == Float16 (algorithm uses fp16 SMEM + fp32 accumulators).
        warp_id = cg0_tidx // 32
        lane_id = cg0_tidx % 32

        sA = sAinv_mn_view[None, None, ainv_handle.index]
        # Stage 1: Gauss-Jordan inversion of BT//8 = 16 diagonal 8x8 blocks.
        sM_8x8 = cute.flat_divide(sA, (8, 8))
        self.inverse_barrier.arrive_and_wait()
        if warp_id < 2:
            self._invert_diagonal_NxN(
                sM_8x8[None, None, cg0_tidx // 8, cg0_tidx // 8], cg0_tidx, 8
            )
        self.inverse_barrier.arrive_and_wait()

        # Stage 2: off-diagonal correction 8x8 -> 16x16.
        # 8 diagonal 16x16 tiles; each warp handles 2 tiles sequentially.
        sM_16x16 = cute.flat_divide(sA, (16, 16))
        self._blockwise_diagonal_8x8_to_16x16(
            sM_16x16[None, None, warp_id, warp_id], lane_id
        )
        self.inverse_barrier.arrive_and_wait()

        # Stage 3: off-diagonal correction 16x16 -> 32x32.
        # 4 diagonal 32x32 tiles; one tile per warp.
        sM_32x32 = cute.flat_divide(sA, (32, 32))
        if warp_id < 2:
            self._blockwise_diagonal_16x16_to_32x32(
                sM_32x32[None, None, warp_id, warp_id], lane_id
            )
        self.inverse_barrier.arrive_and_wait()

        # Stage 4: off-diagonal correction 32x32 -> 64x64.
        # 2 diagonal 64x64 tiles; warps 0,1 on tile 0, warps 2,3 on tile 1.
        sM_64x64 = cute.flat_divide(sA, (64, 64))
        if warp_id < 2:
            self._blockwise_diagonal_32x32_to_64x64(
                sM_64x64[None, None, warp_id // 2, warp_id // 2], warp_id, lane_id
            )
        self.inverse_barrier.arrive_and_wait()

        # Fence SMEM writes and signal A_inv ready to MMA warp (GEMM 5 can start)
        cute.arch.fence_view_async_shared()

        ainv_handle.commit()

        return (  # type: ignore[return-value]
            load_rawgate_consumer,
            load_beta_consumer,
            shared_acc_consumer,
            a_inv_ready_producer,
            qk_ready_producer,
            group_order_producer,
            load_k_consumer_cg0,
            load_q_consumer_cg0,
            e_ready_producer,
            k_bar_ready_producer,
            q_gamma_ready_producer,
            cumprod_ready_producer,
        )

    # ------------------------------------------------------------------
    # Hierarchical blockwise inverse helpers (ported from gdn_inverse_verify.py)
    # Compute X = (I + M)^{-1} for a 128x128 unit lower-triangular matrix in-place
    # on a row-major fp16 SMEM buffer.  5-stage algorithm:
    #   Stage 1: Gauss-Jordan inversion of 16 diagonal 8x8 blocks (warp shuffle)
    #   Stage 2: 8x8 -> 16x16 via warp MMA  (SM80_16x8x8)
    #   Stage 3: 16x16 -> 32x32 via warp MMA (SM80_16x8x16)
    #   Stage 4: 32x32 -> 64x64 via warp MMA, 2 warps per 64x64 tile
    #   Stage 5: 64x64 -> 128x128 via warp MMA, 4 warps on full matrix
    # ------------------------------------------------------------------

    def _make_acc_tensor_into_a_view(self, acc: cute.Tensor) -> cute.Tensor:
        """Reinterpret accumulator tensor as an A-operand tensor for the next MMA.

        For SM80_16x8x8 (ratio=1) the layout is unchanged; for SM80_16x8x16 (ratio=2)
        the C-frag atom size differs from the A-frag atom size and requires a reshape.
        """
        acc_layout_divided = cute.logical_divide(acc.layout, (None, None, 2))
        acc_layout_a = cute.make_layout(
            (
                (acc_layout_divided.shape[0], acc_layout_divided.shape[2][0]),
                acc_layout_divided.shape[1],
                acc_layout_divided.shape[2][1],
            ),
            stride=(
                (acc_layout_divided.stride[0], acc_layout_divided.stride[2][0]),
                acc_layout_divided.stride[1],
                acc_layout_divided.stride[2][1],
            ),
        )
        return cute.make_tensor(acc.iterator, acc_layout_a)

    @cute.jit
    def _invert_diagonal_NxN(self, mat_NxN, tidx, N: int = 8):
        """Stage 1: Gauss-Jordan inversion of one diagonal NxN block in-place (fp16 SMEM).

        Each thread owns one row (tidx_in_group = tidx % N).
        Uses warp shuffle to broadcast pivot values; no __syncthreads inside.
        N-1 pivot steps, compile-time unrolled.
        """
        tidx_in_group = tidx % N
        row_f16 = cute.make_rmem_tensor((N,), self.io_dtype)
        cute.autovec_copy(mat_NxN[tidx_in_group, None], row_f16)
        row = cute.make_rmem_tensor_like(row_f16, self.acc_dtype)
        row.store(row_f16.load().to(cutlass.Float32))
        for i in cutlass.range_constexpr(N):
            row[i] = 1.0 if tidx_in_group == i else row[i]
        for src_row in cutlass.range_constexpr(N - 1):
            row_scale = -row[src_row]
            for i in cutlass.range_constexpr(src_row):
                shfl_val = cute.arch.shuffle_sync(
                    row[i], src_row, mask=0xFFFFFFFF, mask_and_clamp=0b1100000011111
                )
                row[i] = (
                    row[i] + row_scale * shfl_val if tidx_in_group > src_row else row[i]
                )
            row[src_row] = row_scale if tidx_in_group > src_row else row[src_row]

        row_f16.store(row.load().to(self.io_dtype))
        cute.autovec_copy(row_f16, mat_NxN[tidx_in_group, None])

    @cute.jit
    def _blockwise_diagonal_8x8_to_16x16(self, mat_16x16, lane_id):
        """Stage 2: off-diagonal correction for one 16x16 diagonal tile (8x8 -> 16x16).

        After Stage 1 each diagonal 8x8 is inverted.  Computes the bottom-left 8x8
        correction block: C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x8_F32F16F16F32_TN, single warp.  D^{-1} broadcast 8x8 -> 16x8.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 8)
        )
        tiled_mma = cute.make_tiled_mma(mma_atom, cute.make_layout((1, 1, 1)))
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=1, transpose=True),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=1, transpose=False),
                mat_16x16.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_8x8_2x2 = cute.flat_divide(mat_16x16, (8, 8))
        sDinv = mat_8x8_2x2[None, None, 1, 1]
        sC = mat_8x8_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_8x8_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_8x8_2x2[None, None, 1, 0]

        sDinv_m_bcast = cute.make_tensor(
            sDinv.iterator,
            cute.logical_product(sDinv.layout, (cute.make_layout((2,), stride=(0,)),)),
        )
        sO_m_bcast = cute.make_tensor(
            sO.iterator,
            cute.logical_product(sO.layout, (cute.make_layout((2,), stride=(0,)),)),
        )

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv_m_bcast))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 8)))

        tOsDinv = D_thr_copy.partition_S(sDinv_m_bcast)
        tOsDinv = cute.logical_divide(tOsDinv, (tOsDinv.shape[0], None, None))
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOrDinv_cv = cute.logical_divide(tOrDinv_cv, (tOrDinv_cv.shape[0], None, None))
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO_m_bcast)
        tOsO = cute.logical_divide(tOsO, (tOsO.shape[0], None, None))
        tOrO_cv = O_thr_copy.retile(tOrO)
        tOrO_cv = cute.logical_divide(tOrO_cv, (tOrO_cv.shape[0], None, None))
        cute.copy(
            D_tiled_copy,
            tOsDinv[(None, 0), None, None],
            tOrDinv_cv[(None, 0), None, None],
        )
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, self.io_dtype)
        tOrDC.store(tDCrDC_a.load().to(self.io_dtype))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(
            tOrO_cv[(None, 0), None, None], self.io_dtype
        )
        tOrO_cv_cvt.store(tOrO_cv[(None, 0), None, None].load().to(self.io_dtype))
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO[(None, 0), None, None])

    @cute.jit
    def _blockwise_diagonal_16x16_to_32x32(self, mat_32x32, lane_id):
        """Stage 3: off-diagonal correction for one 32x32 diagonal tile (16x16 -> 32x32).

        After Stage 2 each diagonal 16x16 is inverted.  Computes C <-- -D^{-1} C A^{-1}.
        MMA: SM80_16x8x16_F32F16F16F32_TN, TileShape (16,16,16), single warp.
        make_acc_into_op ratio=2: A-frag atom size (8) / C-frag atom size (4).
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 16, 16)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_32x32.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_16x16_2x2 = cute.flat_divide(mat_32x32, (16, 16))
        sDinv = mat_16x16_2x2[None, None, 1, 1]
        sC = mat_16x16_2x2[None, None, 1, 0]
        sC = cute.make_tensor(sC.iterator, cute.select(sC.layout, mode=[1, 0]))
        sAinv = mat_16x16_2x2[None, None, 0, 0]
        sAinv = cute.make_tensor(sAinv.iterator, cute.select(sAinv.layout, mode=[1, 0]))
        sO = mat_16x16_2x2[None, None, 1, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 16)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_32x32.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_32x32.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_32x32.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_32x32.element_type))
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _blockwise_diagonal_32x32_to_64x64(self, mat_64x64, warp_id, lane_id):
        """Stage 4: off-diagonal correction for one 64x64 diagonal tile (32x32 -> 64x64).

        4 warps collaborate (warp_id in {0,1,2,3}); x = warp_id//2, y = warp_id%2.
        MMA: SM80_16x8x16 TileShape (16,32,32), permutation_mnk=(16,32,32).
        Ends with sync_threads() to protect the sO write from races.
        """
        mma_atom = cute.nvgpu.warp.MmaF16BF16Op(
            self.io_dtype, self.acc_dtype, (16, 8, 16)
        )
        tiled_mma = cute.make_tiled_mma(
            mma_atom, cute.make_layout((1, 1, 1)), permutation_mnk=(16, 32, 32)
        )
        thr_mma = tiled_mma.get_slice(lane_id)

        D_tiled_copy = cute.make_tiled_copy_A(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        C_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        A_tiled_copy = cute.make_tiled_copy_B(
            cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )
        O_tiled_copy = cute.make_tiled_copy_C(
            cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
                mat_64x64.element_type,
            ),
            tiled_mma,
        )

        D_thr_copy = D_tiled_copy.get_slice(lane_id)
        C_thr_copy = C_tiled_copy.get_slice(lane_id)
        A_thr_copy = A_tiled_copy.get_slice(lane_id)
        O_thr_copy = O_tiled_copy.get_slice(lane_id)

        mat_32x32_2x2 = cute.flat_divide(mat_64x64, (32, 32))
        sDinv_full = mat_32x32_2x2[None, None, 1, 1]
        sC_full = mat_32x32_2x2[None, None, 1, 0]
        sAinv_full = mat_32x32_2x2[None, None, 0, 0]

        sDinv = cute.flat_divide(sDinv_full, (16, 32))[None, None, warp_id % 2, 0]
        sC = cute.make_tensor(
            sC_full.iterator, cute.select(sC_full.layout, mode=[1, 0])
        )
        sAinv = cute.make_tensor(
            sAinv_full.iterator, cute.select(sAinv_full.layout, mode=[1, 0])
        )
        sO = cute.flat_divide(sC_full, (16, 32))[None, None, warp_id % 2, 0]

        tOrDinv = tiled_mma.make_fragment_A(thr_mma.partition_A(sDinv))
        tOrC = tiled_mma.make_fragment_B(thr_mma.partition_B(sC))
        tOrAinv = tiled_mma.make_fragment_B(thr_mma.partition_B(sAinv))
        tDCrDC = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))
        tOrO = tiled_mma.make_fragment_C(tiled_mma.partition_shape_C((16, 32)))

        tOsDinv = D_thr_copy.partition_S(sDinv)
        tOrDinv_cv = D_thr_copy.retile(tOrDinv)
        tOsC = C_thr_copy.partition_S(sC)
        tOrC_cv = C_thr_copy.retile(tOrC)
        tOsAinv = A_thr_copy.partition_S(sAinv)
        tOrAinv_cv = A_thr_copy.retile(tOrAinv)
        tOsO = O_thr_copy.partition_D(sO)
        tOrO_cv = O_thr_copy.retile(tOrO)

        cute.copy(D_tiled_copy, tOsDinv, tOrDinv_cv)
        cute.copy(C_tiled_copy, tOsC, tOrC_cv)
        tDCrDC.fill(0.0)
        cute.gemm(tiled_mma, tDCrDC, tOrDinv, tOrC, tDCrDC)
        tDCrDC.store(-tDCrDC.load())

        tDCrDC_a = self._make_acc_tensor_into_a_view(tDCrDC)
        tOrDC = cute.make_rmem_tensor_like(tDCrDC_a, mat_64x64.element_type)
        tOrDC.store(tDCrDC_a.load().to(mat_64x64.element_type))

        cute.copy(A_tiled_copy, tOsAinv, tOrAinv_cv)
        tOrO.fill(0.0)
        cute.gemm(tiled_mma, tOrO, tOrDC, tOrAinv, tOrO)

        tOrO_cv_cvt = cute.make_rmem_tensor_like(tOrO_cv, mat_64x64.element_type)
        tOrO_cv_cvt.store(tOrO_cv.load().to(mat_64x64.element_type))
        self.inverse_barrier_inner.arrive_and_wait()
        cute.copy(O_tiled_copy, tOrO_cv_cvt, tOsO)

    @cute.jit
    def _load_initial_state(
        self,
        tidx,
        mS_init,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        kv_acc_producer,
    ) -> pipeline.PipelineProducer:
        """Load S_init from GMEM into state TMEM (fp32).

        Two steps:
          1. GMEM fp32 -> registers
          2. registers -> state TMEM (fp32), signal kv_acc so MMA can start GEMM 7
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM store copy (registers -> state TMEM)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        cState = cute.make_identity_tensor((self.mma_tiler_kv[0], self.mma_tiler_kv[1]))
        tCtState_for_r2t = tCtState[(None, None), 0, 0, 0]
        tiled_state_r2t = tcgen05.make_tmem_copy(state_r2t_atom, tCtState_for_r2t)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        tRT_tCcState = thr_state_r2t.partition_S(cState)
        tRT_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, self.acc_dtype)
        tGR_tCrState = cute.make_rmem_tensor_like(tRT_tCcState, self.state_dtype)

        gS_init = cute.flat_divide(
            mS_init[None, None, head_idx, batch_idx],
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
        )[None, None, 0, 0]
        tGR_tCgState = thr_state_r2t.partition_S(gS_init)
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        for sub in cutlass.range(tRT_tCrState.shape[2]):
            # 1. Load S_init fp32 GMEM -> fp32 registers
            cute.autovec_copy(
                tGR_tCgState[None, 0, sub],
                tGR_tCrState[None, 0, sub],
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )
            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                tRT_tCrState[None, 0, sub].store(
                    tGR_tCrState[None, 0, sub].load().to(self.state_dtype)
                )
            else:
                tRT_tCrState = tGR_tCrState

            # 2. fp32 registers -> state TMEM; signal kv_acc (GEMM 7 accumulates dS on top)
            cute.copy(
                tiled_state_r2t,
                tRT_tCrState[None, 0, sub],
                tRT_tCtState[None, 0, sub, kv_acc_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()

        # Manually sync before committing - CG1 is not the MMA warp so uses mbarrier_arrive.
        self.init_state_store_barrier.arrive_and_wait()
        if cg1_tidx == 0:
            cute.arch.mbarrier_arrive(kv_acc_handle.barrier)

        return kv_acc_producer

    @cute.jit
    def _store_final_state(
        self,
        tidx,
        # full output-state GMEM tensor (DK, DV, (h_r, h_qv), B) fp32
        mS_out,
        head_idx,
        batch_idx,
        tmem_ptr,
        tiled_mma_kv,
        # MMA -> CG1 consumer; waited+released inside this method
        kv_acc_consumer,
        seqlen_b,
        mS_checkpoints,
        checkpoint_offset,
        checkpoint_every_n_tokens,
    ):
        """Store final recurrent state from TMEM (fp32) to GMEM mS_out.

        Waits for the last GEMM-7 (kv_acc) to complete, reads state TMEM -> registers,
        writes registers -> GMEM fp32, then releases the consumer handle.
        """
        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # Build state TMEM layout (mirrors compute_group_1 setup)
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )

        # TMEM -> registers  (Ld32x32b)
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        # S is final: the per-chunk dS combine (same warp group) consumed the
        # last GEMM-7 commit and folded it into S before this call.

        for sub in cutlass.range(tTR_rState.shape[2]):
            # Read state TMEM -> fp32 registers
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, sub, 0],
                tTR_rState[None, 0, sub],
            )
            if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                tRG_rState[None, 0, sub].store(
                    tTR_rState[None, 0, sub].load().to(self.state_dtype)
                )
            else:
                tRG_rState = tTR_rState
            if cutlass.const_expr(self.enable_checkpoints):
                if seqlen_b % checkpoint_every_n_tokens == 0:
                    gS_checkpoints = cute.make_tensor(
                        mS_checkpoints[
                            None, None, head_idx, checkpoint_offset
                        ].iterator,
                        cute.make_ordered_layout(
                            (self.mma_tiler_kv[0], self.mma_tiler_kv[1]), order=(1, 0)
                        ),
                    )
                    tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                    cute.autovec_copy(
                        tRG_rState[None, 0, sub],
                        tSgCheckpoints[None, 0, sub],
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                    )
            if cutlass.const_expr(self.store_final_state):
                gS_out = cute.flat_divide(
                    mS_out[None, None, head_idx, batch_idx],
                    (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                )[None, None, 0, 0]
                tRG_tCgState = thr_state_t2r.partition_D(gS_out)
                cute.autovec_copy(
                    tRG_rState,
                    tRG_tCgState,
                    l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                )
        return kv_acc_consumer

    @cute.jit
    def compute_group_1(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        checkpoint_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
        pipeline.PipelineProducer,
    ]:
        """Warps 4-7: v-k*state, state*q_epi, new_v_epi, qkv_epilogue, dS combine."""
        sV, sW, sCumprodTotal, sBeta, sO = smem_args
        mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens = checkpoint_args
        (
            load_v_consumer,
            load_w_consumer,
            cumprod_ready_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            group_order_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        (is_first_chunk, chunk_idx, head_idx) = work_args

        num_threads_cg1 = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        cg1_tidx = tidx % num_threads_cg1

        # -- State TMEM tensor (DKxDV, layout from tiled_mma_kv) --------------
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn_view = self.transform_partitioned_tensor_layout(tCtState)
        tCcState = cute.make_identity_tensor(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        atom_state_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        atom_state_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tCtState_for_t2r = tCtState[(None, None), 0, 0, 0]
        tiled_state_t2r = tcgen05.make_tmem_copy(atom_state_t2r, tCtState_for_t2r)
        tiled_state_r2t = tcgen05.make_tmem_copy(atom_state_r2t, tCtState_for_t2r)
        thr_state_t2r = tiled_state_t2r.get_slice(cg1_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(cg1_tidx)
        tTR_tCtState = thr_state_t2r.partition_S(tCtState_mn_view)
        tTR_tCcState = thr_state_t2r.partition_D(tCcState)
        tRT_tCtState = thr_state_r2t.partition_D(tCtState_mn_view)
        # dS scratch (GEMM 7 output) lives in the shared_acc columns with the
        # same fragment layout as S; read by the end-of-chunk combine.
        tCtDs_cg1 = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset, tCtState_fake.layout
        )
        tCtDs_mn_view = self.transform_partitioned_tensor_layout(tCtDs_cg1)
        tTR_tCtDs = thr_state_t2r.partition_S(tCtDs_mn_view)
        tTR_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.acc_dtype)
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtState_inp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtState_inp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtState_inp_fake.layout,
        )
        tCtState_inp_mn_view = self.transform_partitioned_tensor_layout(tCtState_inp)
        tCcState_inp = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        atom_state_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.io_dtype
        )
        tCtState_inp_for_r2t = tCtState_inp_mn_view[None, None, 0]
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            atom_state_inp_r2t, tCtState_inp_for_r2t
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcState_inp = thr_state_inp_r2t.partition_S(tCcState_inp)
        tRT_tCtState_inp = thr_state_inp_r2t.partition_D(tCtState_inp_mn_view)
        tRT_rState_inp = cute.make_rmem_tensor_like(tRT_tCcState_inp, self.io_dtype)

        # -- Shared acc TMEM tensor (BTxDV, layout from tiled_mma_qkv) ----------
        # Needed for v-k*state (K*S, stage 0) and new_v_epi (NV, stage 1).
        # qkv_epilogue reads O_intra from q_state TMEM, not from this buffer.
        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_shared_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout, cute.make_layout((self.tmem_shared_acc_stages,))
            ),
        )
        tCtShared_mn_view = self.transform_partitioned_tensor_layout(tCtShared)
        # use qk here to construct shared as it has the biggest tiler (BT, BT)
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtShared_for_t2r = tCtShared[(None, None), 0, 0, 0]
        tiled_shared_t2r = tcgen05.make_tmem_copy(atom_shared_t2r, tCtShared_for_t2r)
        thr_shared_t2r = tiled_shared_t2r.get_slice(cg1_tidx)
        tTR_tCtShared = thr_shared_t2r.partition_S(tCtShared_mn_view)
        tTR_tCcShared = thr_shared_t2r.partition_D(tCcShared)

        qkv_inp_shape = tiled_mma_qkv.partition_shape_A(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        tCtShared_inp_fake = tiled_mma_qkv.make_fragment_A(
            cute.append(qkv_inp_shape, self.tmem_shared_inp_stages)
        )
        tCtShared_inp = cute.make_tensor(
            cute.recast_ptr(
                tmem_ptr + self.tmem_shared_inp_offset, dtype=self.io_dtype
            ),
            tCtShared_inp_fake.layout,
        )
        tCtShared_inp_mn_view = self.transform_partitioned_tensor_layout(tCtShared_inp)
        tCcShared_inp = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[2])
        )
        atom_shared_inp_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x128bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tCtShared_inp_for_r2t = tCtShared_inp_mn_view[None, None, 0]
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_for_r2t
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(cg1_tidx)
        tRT_tCcShared_inp = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        tRT_tCtShared_inp = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)
        tRT_rShared_inp = cute.make_rmem_tensor_like(tRT_tCcShared_inp, self.io_dtype)  # noqa: F841

        # -- Q-state TMEM tensor (BTxDV, layout from tiled_mma_qs) --------------
        # Used by state*q_epi (scale Q*S result) and qkv_epilogue (read final O).
        qs_acc_shape = tiled_mma_qs.partition_shape_C(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        tCtQState_fake = tiled_mma_qs.make_fragment_C(
            cute.append(qs_acc_shape, self.tmem_q_state_acc_stages)
        )
        tCtQState = cute.make_tensor(
            tmem_ptr + self.tmem_q_state_offset, tCtQState_fake.layout
        )
        tCtQState_mn_view = self.transform_partitioned_tensor_layout(tCtQState)
        tCcQState = cute.make_identity_tensor(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[1])
        )
        atom_qs_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        atom_qs_r2t = cute.make_copy_atom(
            tcgen05.copy.St16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tCtQState_for_t2r = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_t2r = tcgen05.make_tmem_copy(atom_qs_t2r, tCtQState_for_t2r)
        tCtQState_for_r2t = tCtQState[(None, None), 0, 0, 0]
        tiled_qs_r2t = tcgen05.make_tmem_copy(atom_qs_r2t, tCtQState_for_r2t)
        thr_qs_t2r = tiled_qs_t2r.get_slice(cg1_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(cg1_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        tTR_rQS = cute.make_rmem_tensor_like(tTR_tCcQS, self.acc_dtype)

        # -- SMEM V tiled copy: sV has domain (DV, BT); threads over BT (dim 1) --
        # Thread cg1_tidx owns all DV features for BT token cg1_tidx.
        # Beta scaling: sBeta[cg1_tidx] - one scalar per thread.

        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        tRT_tCtV = thr_shared_inp_r2t.partition_D(tCtShared_inp_mn_view)  # noqa: F841
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(
            atom_v_s2r,
            tiled_shared_inp_r2t,
        )
        thr_v_s2r = tiled_v_s2r.get_slice(cg1_tidx)

        # -- SMEM store: fp32 TMEM (tiled_mma_qs layout) -> fp16 sO --
        # Used by qkv_epilogue to write final O result.
        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_for_t2r)
        thr_o_t2r = tiled_o_t2r.get_slice(cg1_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_mn_view)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(cg1_tidx)
        tCsO = thr_o_r2s.partition_D(sO)

        sub_tile_size = 32

        gate_handle = cumprod_ready_consumer.wait_and_advance()

        valid_state = not is_first_chunk or self.use_initial_state
        if cutlass.const_expr(valid_state):
            if cutlass.const_expr(self.use_initial_state):
                kv_acc_producer.advance()
            if cutlass.const_expr(is_first_chunk):
                # Consume the initial-state load's kv_acc commit (S0 in TMEM).
                # Non-first chunks need no wait: S was last written by this
                # warp group's own dS combine at the end of the previous chunk.
                kv_handle = kv_acc_consumer.wait_and_advance()

            state_inp_ready_handle = state_inp_ready_producer.acquire_and_advance()
            for sub in cutlass.range(tRT_rState_inp.shape[2]):
                cute.copy(
                    tiled_state_t2r,
                    tTR_tCtState[None, 0, sub, 0],
                    tTR_rState[None, 0, sub],
                )
                if cutlass.const_expr(self.enable_checkpoints and not is_first_chunk):
                    if (self.b_t * chunk_idx) % checkpoint_every_n_tokens == 0:
                        gS_checkpoints = cute.make_tensor(
                            mS_checkpoints[
                                None, None, head_idx, checkpoint_offset
                            ].iterator,
                            cute.make_ordered_layout(
                                (self.mma_tiler_kv[0], self.mma_tiler_kv[1]),
                                order=(1, 0),
                            ),
                        )
                        tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                        if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                            tRG_rState[None, 0, sub].store(
                                tTR_rState[None, 0, sub].load().to(self.state_dtype)
                            )
                        else:
                            tRG_rState = tTR_rState
                        cute.autovec_copy(
                            tRG_rState[None, 0, sub],
                            tSgCheckpoints[None, 0, sub],
                            l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                        )

                tRT_rState_inp[None, 0, sub].store(
                    tTR_rState[None, 0, sub].load().to(self.io_dtype)
                )
                cute.copy(
                    tiled_state_inp_r2t,
                    tRT_rState_inp[None, 0, sub],
                    tRT_tCtState_inp[None, 0, sub, state_inp_ready_handle.index],
                )
            cute.arch.fence_view_async_tmem_store()
            state_inp_ready_handle.commit()

            # Load S_prev -> scale by the per-d_k chunk decay -> write back to
            # the TMEM slot; decay gathered per sub-tile.
            rCumprodS = cute.make_rmem_tensor(
                cute.make_layout((sub_tile_size,)), self.acc_dtype
            )
            for sub in cutlass.range(tTR_rState.shape[2]):
                for k in cutlass.range(sub_tile_size):
                    coord = tTR_tCcState[k, 0, sub]
                    rCumprodS[k] = sCumprodTotal[coord[1], gate_handle.index]
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rState[k, 0, sub] = tTR_rState[k, 0, sub] * rCumprodS[k]
                cute.copy(
                    tiled_state_r2t,
                    tTR_rState[None, 0, sub],
                    tRT_tCtState[None, 0, sub, 0],
                )
            if cutlass.const_expr(self.enable_checkpoints and not is_first_chunk):
                if (self.b_t * chunk_idx) % checkpoint_every_n_tokens == 0:
                    checkpoint_offset += 1
            cute.arch.fence_view_async_tmem_store()

            if cutlass.const_expr(is_first_chunk):
                # Release the initial-state slot -> MMA can acquire kv_acc for
                # this chunk's GEMM 7.
                kv_handle.release()

        # wait for kk and qk epilogue to finish
        shared_acc_consumer.advance()
        shared_acc_consumer.advance()

        # gate_handle (sCumprodTotal) stays held until the dS combine below.
        # ---- v - k*state  (ALU) -----------------------------------------------
        # delta[bt, dv] = w*V - (K*S), staged to the shared_inp TMEM slot
        # (GEMM 5 A operand).
        vks_handle = shared_inp_ready_producer.acquire_and_advance()
        v_handle = load_v_consumer.wait_and_advance()
        # write gate: wait for the TMA-loaded w (consumed every chunk).
        w_handle = load_w_consumer.wait_and_advance()

        # The w write gate applies in EVERY chunk: stage w*V (minus K*S when a
        # carried state exists) to the shared_inp TMEM slot unconditionally;
        # GEMM 5 always sources its A operand from TMEM.
        sV_vt_view = self.transform_partitioned_tensor_layout(sV)
        tCsV = thr_v_s2r.partition_S(sV_vt_view)

        tRT_rV = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
        tCrV = tiled_v_s2r.retile(tRT_rV)
        cute.copy(tiled_v_s2r, tCsV[None, None, None, v_handle.index], tCrV)
        # write gate: w uses the same LdMatrix s2r copy as V; V <- w * V.
        sW_vt_view = self.transform_partitioned_tensor_layout(sW)
        tCsW = thr_v_s2r.partition_S(sW_vt_view)
        tRT_rW = cute.make_rmem_tensor_like(tRT_tCcV, self.io_dtype)
        tCrW = tiled_v_s2r.retile(tRT_rW)
        cute.copy(tiled_v_s2r, tCsW[None, None, None, w_handle.index], tCrW)
        # NOTE: leave this as the plain bf16 multiply -- the compiler FUSES it with the
        # V-KS subtract below into ONE native HFMA2.BF16_V2 (V*w - KS) per pair: zero
        # conversions, single rounding.  MOV-pack/fp32 rewrites here are strictly worse.
        for k in cutlass.range(cute.size(tRT_rV), vectorize=True):
            tRT_rV[k] = tRT_rV[k] * tRT_rW[k]
        group_order_handle = group_order_consumer.wait_and_advance()
        if cutlass.const_expr(valid_state):
            tTR_rKS = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
            # Wait for GEMM 3 (K*S) result in shared_acc
            ks_acc_handle = shared_acc_consumer.wait_and_advance()
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, None, None, ks_acc_handle.index],
                tTR_rKS,
            )
            # Release only AFTER the subtract consumes the async tcgen05.ld
            # results: releasing first lets the next chunk's GEMM 1 overwrite
            # the stage mid-load.
            for k in cutlass.range(cute.size(tTR_rKS), vectorize=True):
                tRT_rV[k] = tRT_rV[k] - tTR_rKS[k].to(self.io_dtype)
            ks_acc_handle.release()
        cute.copy(
            tiled_shared_inp_r2t,
            tRT_rV,
            tRT_tCtShared_inp[None, None, None, vks_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        vks_handle.commit()
        # w release deferred past the vks staging commit (register-dep form).
        w_handle.release()

        # ---- state*q_epi (ALU) ------------------------------------------------
        # Q*S_prev cross-chunk contribution.  GDN-2: decay folds into q_gamma (CG0),
        # so only the attention scale is applied here.  Write back to q_state TMEM.
        if cutlass.const_expr(valid_state):
            qs_handle = q_state_acc_consumer.wait_and_advance()
            for sub in cutlass.range(tTR_rQS.shape[1]):
                cute.copy(
                    tiled_qs_t2r,
                    tTR_tCtQS[None, sub, 0, qs_handle.index],
                    tTR_rQS[None, sub, 0],
                )
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rQS[k, sub, 0] = tTR_rQS[k, sub, 0] * scale
                cute.copy(
                    tiled_qs_r2t,
                    tTR_rQS[None, sub, 0],
                    tRT_tCtQS[None, sub, 0, qs_handle.index],
                )
            cute.arch.fence_view_async_tmem_store()
            qs_handle.release()

        # ---- new_v_epi --------------------------------------------------------
        # NV = A_inv @ delta (GEMM 5 result) in shared_acc TMEM; fp32 -> fp16 -> sAinvNv.
        nv_handle = shared_acc_consumer.wait_and_advance()
        v_handle.release()
        nv_ready_handle = shared_inp_ready_producer.acquire_and_advance()

        tTR_rNv = cute.make_rmem_tensor_like(tTR_tCcShared, self.acc_dtype)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        for sub in cutlass.range(tTR_rNv.shape[1]):
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, sub, 0, nv_handle.index],
                tTR_rNv[None, sub, 0],
            )
            tTR_rNv_inp[None, sub, 0].store(
                tTR_rNv[None, sub, 0].load().to(self.io_dtype)
            )
            cute.copy(
                tiled_shared_inp_r2t,
                tTR_rNv_inp[None, sub, 0],
                tRT_tCtShared_inp[None, sub, 0, nv_ready_handle.index],
            )
        cute.arch.fence_view_async_tmem_store()
        nv_ready_handle.commit()

        # ---- dS combine (ALU) --------------------------------------------------
        # Fold GEMM 7's raw dS into S: S += cumprod_total[d_k] * dS.  Runs
        # before the qkv_epilogue so it overlaps GEMM 6.
        kv_handle2 = kv_acc_consumer.wait_and_advance()

        rCumprodC = cute.make_rmem_tensor(
            cute.make_layout((sub_tile_size,)), self.acc_dtype
        )
        tTR_rDs = cute.make_rmem_tensor_like(tTR_rState[None, 0, 0], self.acc_dtype)
        for sub in cutlass.range(tTR_rState.shape[2]):
            cute.copy(
                tiled_state_t2r,
                tTR_tCtDs[None, 0, sub, 0],
                tTR_rDs,
            )
            for k in cutlass.range(sub_tile_size):
                coord = tTR_tCcState[k, 0, sub]
                rCumprodC[k] = sCumprodTotal[coord[1], gate_handle.index]
            if cutlass.const_expr(is_first_chunk and not self.use_initial_state):
                # First chunk without an initial state: S is uninitialized;
                # write instead of accumulate.
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rState[k, 0, sub] = rCumprodC[k] * tTR_rDs[k]
            else:
                cute.copy(
                    tiled_state_t2r,
                    tTR_tCtState[None, 0, sub, 0],
                    tTR_rState[None, 0, sub],
                )
                for k in cutlass.range(sub_tile_size, vectorize=True):
                    tTR_rState[k, 0, sub] = (
                        tTR_rState[k, 0, sub] + rCumprodC[k] * tTR_rDs[k]
                    )
            cute.copy(
                tiled_state_r2t,
                tTR_rState[None, 0, sub],
                tRT_tCtState[None, 0, sub, 0],
            )
        # The dS reads were consumed by the FFMAs above (scoreboard-ordered)
        # -> releasing kv_acc HERE frees the next chunk's KK^T without waiting
        # for the S write-back below to land.
        kv_handle2.release()
        # NV-stage release deferred past the combine (register-dep form).
        nv_handle.release()
        cute.arch.fence_view_async_tmem_store()
        gate_handle.release()

        # ---- qkv_epilogue -----------------------------------------------------
        # GEMM 6 accumulated W_qkv@NV into q_state TMEM on top of the scaled Q*S.
        # q_state_acc second wait (same 1-stage pipeline, wraps back to stage 0).
        o_handle = o_store_producer.acquire_and_advance()
        qs_handle2 = q_state_acc_consumer.wait_and_advance()

        tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, self.acc_dtype)
        tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
        tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
        cute.copy(
            tiled_o_t2r,
            tTR_tOtO[None, None, None, qs_handle2.index],
            tTR_tOrO,
        )
        group_order_handle.release()
        tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
        cute.copy(tiled_o_r2s, tRS_tOrO, tCsO[None, None, None, o_handle.index])
        cute.arch.fence_view_async_shared()
        qs_handle2.release()

        # O in sQkOstore ready for epilogue warp TMA store
        o_handle.commit()

        # ---- kv_update_epi ----------------------------------------------------
        # None, do state update at the beginning of the next chunk.
        return (  # type: ignore[return-value]
            load_v_consumer,
            load_w_consumer,
            cumprod_ready_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            group_order_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
            checkpoint_offset,
        )

    @cute.jit
    def epilogue_warp(
        self,
        smem_args,
        tma_args,
        pipeline_args,
        work_args,
        tensormap_args,
    ) -> pipeline.PipelineConsumer:
        """Warp 11: TMA bulk-store O from SMEM staging buffer to global memory.

        Steps:
          1. Wait for CG1 to signal O is ready in sO (via o_store_consumer).
          2. Domain-offset the TMA tensor to (chunk_offset, head_idx), flat-divide
             into tiles, tma_partition -> (tOsO, tOgO).
          3. Issue TMA S2G bulk copy using the per-work-tile updated descriptor.
          4. Commit the async group and wait for the store to land in GMEM.
          5. Release the pipeline slot back to CG1.
        """
        (sO,) = smem_args
        (tma_o,) = tma_args
        (o_store_consumer,) = pipeline_args
        head_idx, chunk_offset = work_args
        tensormap_manager, tensormap_o_ptr = tensormap_args

        o_handle = o_store_consumer.wait_and_advance()

        cta_layout = cute.make_layout(1)
        # (BT, DV)
        o_tile = cute.select(self.mma_tiler_qkv, mode=[0, 1])

        # Position global O tile at current chunk / head
        mO = cute.domain_offset(
            (cutlass.Int32(0), chunk_offset),
            tma_o.tma_tensor[None, None, head_idx],
        )
        # (BT, DV, num_o_tiles, ...)
        gO = cute.flat_divide(mO, o_tile)

        # TMA partition: tOsO = SMEM source, tOgO = GMEM destination
        tOsO, tOgO = cpasync.tma_partition(
            tma_o.atom,
            0,
            cta_layout,
            cute.group_modes(sO, 0, 2),
            cute.group_modes(gO, 0, 2),
        )

        # TMA bulk store SMEM -> GMEM using the descriptor updated per work tile
        cute.copy(
            tma_o.atom,
            tOsO[(None, o_handle.index)],
            tOgO[(None, 0, 0)],
            tma_desc_ptr=tensormap_manager.get_tensormap_ptr(
                tensormap_o_ptr, cute.AddressSpace.generic
            ),
        )

        # Wait for the store to complete before releasing the SMEM slot
        cute.arch.cp_async_bulk_commit_group()
        cute.arch.cp_async_bulk_wait_group(0)

        o_handle.release()

        return o_store_consumer

    def transform_partitioned_tensor_layout(self, tensor: cute.Tensor) -> cute.Tensor:
        """
        Transform MMA layout from ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        to ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest).

        This groups MMA_ATOM_M with MMA_M and MMA_ATOM_N with MMA_N.

        :param tensor: Input tensor with layout ((MMA_ATOM_M, MMA_ATOM_N), MMA_M, MMA_N, ...rest)
        :type tensor: cute.Tensor
        :return: Transformed tensor with layout ((MMA_ATOM_M, MMA_M), (MMA_ATOM_N, MMA_N), ...rest)
        :rtype: cute.Tensor
        """
        layout = tensor.layout
        # Save original layout in case it is a composed layout
        stored_layout = layout

        if isinstance(stored_layout, cute.ComposedLayout):
            # For composed layouts, we only modify the outer layout
            layout = layout.outer

        shape = layout.shape
        stride = layout.stride

        # Build new shape: ((shape[0][0], shape[1]), (shape[0][1], shape[2]), ...rest)
        new_shape = ((shape[0][0], shape[1]), (shape[0][1], shape[2]), *shape[3:])

        # Build new stride: ((stride[0][0], stride[1]), (stride[0][1], stride[2]), ...rest)
        new_stride = ((stride[0][0], stride[1]), (stride[0][1], stride[2]), *stride[3:])

        new_layout = cute.make_layout(shape=new_shape, stride=new_stride)

        if isinstance(stored_layout, cute.ComposedLayout):
            # Recreate the composed layout
            new_layout = cute.make_composed_layout(
                stored_layout.inner, stored_layout.offset, new_layout
            )

        return cute.make_tensor(tensor.iterator, new_layout)

    @cute.jit
    def _transform_to_position_independent_layout(
        self, tensor: cute.Tensor, swizzle_inner: cute.Swizzle
    ) -> cute.Tensor:
        wo_swizzle_iter = cute.recast_ptr(tensor.iterator, swizzle_=None)
        pisl_swizzle_base = int(math.log2(self.io_dtype.width)) - 1
        pisl_swizzle = cute.make_swizzle(
            swizzle_inner.num_bits, pisl_swizzle_base, swizzle_inner.num_shift
        )
        tensor_pisl = cute.make_composed_layout(pisl_swizzle, 0, tensor.layout)
        return cute.make_tensor(wo_swizzle_iter, tensor_pisl)

    @staticmethod
    def get_workspace_size(num_sm: int, B: int, HQ: int, HV: int, is_persistent: bool):
        # q, k, v, o
        if is_persistent:
            return (
                GatedDeltaNetChunkedKernel2.bytes_per_tensormap
                * GatedDeltaNetChunkedKernel2.num_tensormaps
                * num_sm
            )
        HO = HQ if HQ >= HV else HV
        return (
            GatedDeltaNetChunkedKernel2.bytes_per_tensormap
            * GatedDeltaNetChunkedKernel2.num_tensormaps
            * (B * HO)
        )

    @cute.jit
    def initialize_workspace(
        self, workspace: cute.Tensor, grid_dim: Tuple[int, int, int]
    ):
        workspace = cute.make_tensor(
            workspace.iterator,
            cute.make_layout(
                (
                    grid_dim[0] * grid_dim[1] * grid_dim[2],
                    GatedDeltaNetChunkedKernel2.num_tensormaps,
                    GatedDeltaNetChunkedKernel2.bytes_per_tensormap,
                ),
                stride=(
                    GatedDeltaNetChunkedKernel2.num_tensormaps
                    * GatedDeltaNetChunkedKernel2.bytes_per_tensormap,
                    GatedDeltaNetChunkedKernel2.bytes_per_tensormap,
                    1,
                ),
            ),
        )
        return workspace
