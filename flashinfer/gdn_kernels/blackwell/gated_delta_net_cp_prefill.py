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

"""
Context-parallel Gated Delta Net (GDN) prefill kernel for Blackwell SM100.

Each CTA processes one CP chunk as a recurrence of BT=64 token blocks. DK=DV=128.
The CTA starts from a fixed-up state in registers and uses the signed, beta-folded T
tiles produced by CP preprocessing, so this kernel does not form KK or invert it.

Per-block tcgen05 math:
  QK[64,64] = Q[64,128] @ K.T[128,64]
  KS[64,128] = K[64,128] @ S[128,128]
  QS[64,128] = Q[64,128] @ S[128,128]
  NewV[64,128] = T_gamma[64,64] @ (V - gamma * KS)[64,128]
  O[64,128] = gamma * QS + QK_gamma[64,64] @ NewV[64,128]
  S_next[128,128] = gamma_end * S + K.T[128,64] @ delta[64,128]

The principal fp16/bf16 SMEM buffers are Q=16 KiB, K=32 KiB (two stages),
V=16 KiB, raw T=16 KiB (two stages), transformed T=8 KiB, QK=8 KiB,
and O=16 KiB. TMEM holds QK, state/output accumulators, and the staged operands
shared by the two register-resident state halves.

Warp assignments (16 warps = 512 threads):
  warps 0-3     : transform precomputed T and scale QK
  warps 4-7     : keep the left state half in registers and run low-half epilogues
  warps 8-11    : keep the right state half in registers and run high-half epilogues
  warp 12       : issue QK
  warp 13       : issue KS, QS, NewV, output, and state-update UTCMMA operations
  warp 14       : load Q, K, V, gate, and T
  warp 15       : store output tiles

The two state groups follow the latest non-CP SM100 pipeline design. They stage
their register-resident state halves into TMEM operands, independently consume
the two halves of each accumulator, and add the final state increment back to
registers. The auxiliary QK ring is disjoint from the state accumulator ring.
The state-input and shared-input operands alias the final 64 TMEM columns and
are reused serially for state, delta, NewV, and the decayed state update.
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
from ..delta_rule_dsl.varlen_helper import (
    chunks_for_len,
    varlen_chunk_idx,
    varlen_chunk_valid_len,
)
from ..delta_rule_dsl.custom_compile_cache import KeyedCompileMixin


# ---------------------------------------------------------------------------
# Combined configuration + execution class
# ---------------------------------------------------------------------------


class CPDeltaRulePrefillTcgen05Sm100(KeyedCompileMixin):
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
    num_tensormaps = 5

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
        head_ratio: int,
        use_initial_state: bool,
        store_final_state: bool = True,
        enable_checkpoints: bool = False,
        is_persistent: bool = True,
        order_state_groups: bool = False,
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
        self.head_ratio = head_ratio
        self.needs_initial_state = use_initial_state
        # Every CP CTA starts from a fixed-up state in TMEM. For the first CP
        # chunk this is either the public initial state or an explicit zero.
        self.use_initial_state = True
        self.store_final_state = store_final_state
        self.enable_checkpoints = False
        self.is_persistent = False
        self.order_state_groups = order_state_groups

        # ------------------------------------------------------------------
        # Warp assignments  (16 warps total)
        # ------------------------------------------------------------------
        # Precomputed-T transform and QK scaling.
        self.compute_group_0_warp_ids = [0, 1, 2, 3]
        # Left State half plus low token-half epilogues.
        self.compute_group_1_warp_ids = [4, 5, 6, 7]
        # Right State half plus high token-half epilogues.
        self.compute_group_2_warp_ids = [8, 9, 10, 11]
        self.mma_cg0_warp_id = 12
        self.mma_cg1_warp_id = 13
        self.load_warp_id = 14
        # store O
        self.epilogue_warp_id = 15

        self.num_regs_compute_group_0 = 104
        self.num_regs_compute_group_1 = 176
        self.num_regs_compute_group_2 = 176
        self.num_regs_other = 48

        self.threads_per_cta = 32 * (
            len(
                (
                    self.mma_cg0_warp_id,
                    self.mma_cg1_warp_id,
                    self.load_warp_id,
                    self.epilogue_warp_id,
                )
            )
            + len(self.compute_group_0_warp_ids)
            + len(self.compute_group_1_warp_ids)
            + len(self.compute_group_2_warp_ids)
        )

        self.use_2cta_instrs = False
        self.cluster_shape_mnk = (1, 1, 1)
        self.cta_group = (
            tcgen05.CtaGroup.TWO if self.use_2cta_instrs else tcgen05.CtaGroup.ONE
        )
        self.occupancy = 1
        self.threads_per_warp = 32

        # ------------------------------------------------------------------
        # Named barriers for TMEM allocation, T/state exchange, and the
        # ordering handoff between the two TCGEN05 issuer warps. Other inter-warp
        # synchronization uses mbarrier-based pipelines created inside kernel().
        # ------------------------------------------------------------------
        self.tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=self.threads_per_warp
            * len(
                (
                    self.mma_cg0_warp_id,
                    self.mma_cg1_warp_id,
                    *self.compute_group_0_warp_ids,
                    *self.compute_group_1_warp_ids,
                    *self.compute_group_2_warp_ids,
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
        num_threads_state = self.threads_per_warp * len(self.compute_group_1_warp_ids)
        self.state_order_barrier_0 = pipeline.NamedBarrier(
            barrier_id=4, num_threads=num_threads_state * 2
        )
        self.state_order_barrier_1 = pipeline.NamedBarrier(
            barrier_id=5, num_threads=num_threads_state * 2
        )
        self.manual_cache_key(
            "io_dtype",
            "acc_dtype",
            "state_dtype",
            "mma_tiler_qk",
            "mma_tiler_qs",
            "mma_tiler_qkv",
            "mma_tiler_kv",
            "is_GQA",
            "head_ratio",
            "needs_initial_state",
            "order_state_groups",
        )

    @cute.jit
    def _state_order_wait(self, half_idx):
        if cutlass.const_expr(half_idx == 0):
            self.state_order_barrier_0.arrive_and_wait()
        else:
            self.state_order_barrier_1.arrive_and_wait()

    @cute.jit
    def _state_order_notify(self, half_idx):
        if cutlass.const_expr(half_idx == 0):
            self.state_order_barrier_1.arrive()
        else:
            self.state_order_barrier_0.arrive()

    def _setup_attributes(self):
        # ------------------------------------------------------------------
        # SMEM sizes (bytes per stage) and stage counts
        # ------------------------------------------------------------------
        self.smem_q_stages = 1
        self.smem_k_stages = 2
        self.smem_v_stages = 1
        self.smem_t_stages = 2
        self.smem_ainv_stages = 1
        self.smem_qk_stages = 1
        self.smem_o_stages = 1
        self.smem_group_order_stages = 1
        # Cumulative gate buffers - placed last in SMEM
        self.smem_gate_stages = 1

        # ------------------------------------------------------------------
        # TMEM column offsets and buffer sizes (fp32, 32B per column)
        # ------------------------------------------------------------------
        self.tmem_kv_acc_stages = 1
        self.tmem_q_state_acc_stages = 1
        self.tmem_state_inp_stages = 1
        self.tmem_shared_inp_stages = 1
        self.tmem_aux_acc_stages = 2
        self.tmem_state_acc_stages = 2

        # TMEM is addressed in 128-row columns. Accumulator tensors use one
        # fp32 value per column, while fp16/bf16 operands pack two values into
        # each column. BT is 64; only the recurrent state dimensions are 128.
        tmem_io_pack = self.acc_dtype.width // self.io_dtype.width
        self.tmem_state_cols = self.mma_tiler_kv[1]
        self.tmem_q_state_cols = self.mma_tiler_qs[1]
        self.tmem_state_inp_cols = self.mma_tiler_qs[2] // tmem_io_pack
        self.tmem_shared_acc_cols = self.mma_tiler_qkv[1]
        self.tmem_shared_inp_cols = self.mma_tiler_qkv[2] // tmem_io_pack

        self.tmem_state_offset = 0
        self.tmem_q_state_offset = (
            self.tmem_state_offset + self.tmem_kv_acc_stages * self.tmem_state_cols
        )
        self.tmem_aux_acc_offset = (
            self.tmem_q_state_offset + self.tmem_q_state_acc_stages * 64
        )
        self.tmem_state_acc_offset = (
            self.tmem_aux_acc_offset + self.tmem_aux_acc_stages * 64
        )
        self.tmem_state_inp_offset = (
            self.tmem_state_acc_offset + self.tmem_state_acc_stages * 64
        )
        self.tmem_shared_inp_offset = self.tmem_state_inp_offset

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
        t: cute.Tensor,
        o: cute.Tensor,
        cu_seqlens: cute.Tensor,
        fixed_state: cute.Tensor,
        initial_state: Optional[cute.Tensor],
        state_out: cute.Tensor,
        cp_chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_seqs: cutlass.Int32,
        scale: cutlass.Float32,
        tensormap_workspace: cute.Tensor,
        stream: cuda.CUstream,
    ):
        # chunk size
        self.b_t = 64
        h_q = q.shape[1]
        h_v = v.shape[1]
        self._setup_attributes()

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

        gate = cute.make_tensor(
            gate.iterator,
            cute.make_layout(
                (gate.shape[0], (h_r, h_qv)),
                stride=(gate.stride[0], (gate.stride[1], h_r * gate.stride[1])),
            ),
        )
        t = cute.make_tensor(
            t.iterator,
            cute.make_layout(
                (t.shape[0], t.shape[1], (h_r, h_qv), t.shape[3]),
                stride=(
                    t.stride[0],
                    t.stride[1],
                    (t.stride[2], h_r * t.stride[2]),
                    t.stride[3],
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
        fixed_state = cute.make_tensor(
            fixed_state.iterator,
            cute.make_layout(
                (
                    fixed_state.shape[2],
                    fixed_state.shape[3],
                    (h_r, h_qv),
                    fixed_state.shape[0],
                ),
                stride=(
                    fixed_state.stride[2],
                    fixed_state.stride[3],
                    (fixed_state.stride[1], h_r * fixed_state.stride[1]),
                    fixed_state.stride[0],
                ),
            ),
        )
        if cutlass.const_expr(initial_state is not None):
            initial_state = cute.make_tensor(
                initial_state.iterator,
                cute.make_layout(
                    (
                        initial_state.shape[2],
                        initial_state.shape[3],
                        (h_r, h_qv),
                        initial_state.shape[0],
                    ),
                    stride=(
                        initial_state.stride[2],
                        initial_state.stride[3],
                        (initial_state.stride[1], h_r * initial_state.stride[1]),
                        initial_state.stride[0],
                    ),
                ),
            )
        state_out = cute.make_tensor(
            state_out.iterator,
            cute.make_layout(
                (
                    state_out.shape[2],
                    state_out.shape[3],
                    (h_r, h_qv),
                    state_out.shape[0],
                ),
                stride=(
                    state_out.stride[2],
                    state_out.stride[3],
                    (state_out.stride[1], h_r * state_out.stride[1]),
                    state_out.stride[0],
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
        t_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma_qkv, self.mma_tiler_qkv, self.io_dtype, self.smem_t_stages
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
        # Gate scalar arrays (1D Float32, flat layout - no swizzle needed)
        cumsumlog_smem_layout_staged = cute.make_layout(
            (self.b_t, 1, self.smem_gate_stages)
        )

        # ------------------------------------------------------------------
        # Shared memory struct  (defined here to capture layout cosizes)
        # ------------------------------------------------------------------
        @cute.struct
        class SharedStorage:
            # Pipeline mbarriers - one entry per stage, 2 Int64 words per barrier
            # Unified load warp -> both MMA issuers
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_k_stages * 2]
            # Unified load warp -> both MMA issuers
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_q_stages * 2]
            # Unified load warp -> both State groups
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_v_stages * 2]
            # Unified load warp -> CG0 and both State groups
            load_gate_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_gate_stages * 2
            ]
            # TMA T load -> CG0
            load_t_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.smem_t_stages * 2]
            # State MMA issuer -> both State groups (Q*state acc ready in TMEM)
            q_state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_q_state_acc_stages * 2
            ]
            # State MMA issuer -> both State groups (GEMM 7 done)
            kv_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_kv_acc_stages * 2
            ]
            # Auxiliary MMA issuer -> CG0 (KK/QK accumulator ready)
            aux_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_aux_acc_stages * 2
            ]
            # State MMA issuer -> both State groups (SK/NewV accumulator ready)
            state_acc_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_acc_stages * 2
            ]
            # CG0 -> State MMA issuer (A_inv ready in SMEM)
            ainv_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_ainv_stages * 2
            ]
            # CG0 -> State MMA issuer (QK ready in SMEM)
            qk_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_qk_stages * 2
            ]
            state_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_state_inp_stages * 2
            ]
            # Both State groups -> MMA warp (state input ready in TMEM)
            shared_inp_ready_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.tmem_shared_inp_stages * 2
            ]
            # Both State groups -> epilogue warp
            o_store_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.smem_o_stages * 2
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

            sV: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(v_smem_layout_staged)],
                self.buffer_align_bytes,
            ]
            sT: cute.struct.Align[
                cute.struct.MemRange[self.io_dtype, cute.cosize(t_smem_layout_staged)],
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
            # Cumulative gate scalars - placed last in SMEM
            cumsumlog: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]
            cumprod: cute.struct.MemRange[
                cutlass.Float32, cute.cosize(cumsumlog_smem_layout_staged)
            ]

        self.shared_storage = SharedStorage

        # ------------------------------------------------------------------
        # Build TMA atoms
        # ------------------------------------------------------------------
        q_smem_layout = cute.select(q_smem_layout_staged, mode=[0, 1, 2])
        k_smem_layout = cute.select(k_smem_layout_staged, mode=[0, 1, 2])
        v_smem_layout = cute.select(v_smem_layout_staged, mode=[0, 1, 2])
        t_smem_layout = cute.select(t_smem_layout_staged, mode=[0, 1, 2])

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
        tma_t = _wrap_tma(
            cute.nvgpu.make_tiled_tma_atom_B(
                tma_load_op,
                t,
                t_smem_layout,
                self.mma_tiler_qkv,
                tiled_mma_qkv,
                self.cluster_layout_vmnk.shape,
            )
        )

        cumsumlog_smem_layout = cute.select(cumsumlog_smem_layout_staged, mode=[0])  # noqa: F841

        o_smem_layout = cute.select(o_smem_layout_staged, mode=[0, 1])
        tma_o = _wrap_tma(
            cpasync.make_tiled_tma_atom(
                tma_store_op,
                o,
                o_smem_layout,
                self.mma_tiler_qkv[:2],
            )
        )

        self.tma_q_bytes = cute.size_in_bytes(self.io_dtype, q_smem_layout)
        self.tma_k_bytes = cute.size_in_bytes(self.io_dtype, k_smem_layout)
        self.tma_v_bytes = cute.size_in_bytes(self.io_dtype, v_smem_layout)
        self.tma_t_bytes = cute.size_in_bytes(self.io_dtype, t_smem_layout)
        self.tma_o_bytes = cute.size_in_bytes(self.io_dtype, o_smem_layout)

        # ------------------------------------------------------------------
        # Launch
        # ------------------------------------------------------------------
        scheduler_params = GDNTileSchedulerParams(
            num_seqs=num_seqs,
            num_q_heads=h_r * h_qv * max_cp_chunks_per_seq,
            num_v_heads=h_r * h_qv * max_cp_chunks_per_seq,
            is_GQA=True,
            is_persistent=False,
        )
        grid_shape = (h_r * h_qv * max_cp_chunks_per_seq, num_seqs, 1)

        self.kernel(
            tiled_mma_qk,
            tiled_mma_qs,
            tiled_mma_qkv,
            tiled_mma_qkv_ss,
            tiled_mma_kv,
            tma_q,
            tma_k,
            tma_v,
            tma_t,
            gate,
            tma_o,
            cu_seqlens,
            fixed_state,
            initial_state,
            state_out,
            cp_chunk_len,
            total_cp_chunks,
            max_cp_chunks_per_seq,
            h_r * h_qv,
            num_seqs,
            scale,
            q_smem_layout_staged,
            k_smem_layout_staged,
            k_trans_smem_layout_staged,
            v_smem_layout_staged,
            cumsumlog_smem_layout_staged,
            t_smem_layout_staged,
            ainv_smem_layout_staged,
            qk_smem_layout_staged,
            o_smem_layout_staged,
            scheduler_params,
            q,
            k,
            v,
            t,
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

    @cute.jit
    def get_cp_work(
        self,
        cu_seqlens: cute.Tensor,
        seq_idx: cutlass.Int32,
        flat_work_idx: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        cp_chunk_len: cutlass.Int32,
    ):
        head_idx = flat_work_idx % num_sab_heads
        chunk_idx = flat_work_idx // num_sab_heads
        seq_start = cutlass.Int32(cu_seqlens[seq_idx])
        seq_end = cutlass.Int32(cu_seqlens[seq_idx + 1])
        seq_len = seq_end - seq_start
        num_cp_chunks = chunks_for_len(seq_len, cp_chunk_len)
        valid_chunk_len = cutlass.Int32(0)
        if chunk_idx < num_cp_chunks:
            valid_chunk_len = varlen_chunk_valid_len(seq_len, chunk_idx, cp_chunk_len)
        tok_offset = seq_start + chunk_idx * cp_chunk_len
        cp_chunk_idx = varlen_chunk_idx(seq_idx, seq_start, chunk_idx, cp_chunk_len)
        t_blocks_per_cp_chunk = cute.ceil_div(cp_chunk_len, self.b_t)
        t_block_start = varlen_chunk_idx(
            seq_idx, seq_start, chunk_idx * t_blocks_per_cp_chunk, self.b_t
        )
        return (
            head_idx,
            chunk_idx,
            tok_offset,
            valid_chunk_len,
            num_cp_chunks,
            cp_chunk_idx,
            t_block_start,
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
        tma_t: TmaInfo,
        mGate: cute.Tensor,
        tma_o: TmaInfo,
        cu_seqlens: cute.Tensor,
        mFixedState: cute.Tensor,
        mInitialState: Optional[cute.Tensor],
        mStateOut: cute.Tensor,
        cp_chunk_len: cutlass.Int32,
        total_cp_chunks: cutlass.Int32,
        max_cp_chunks_per_seq: cutlass.Int32,
        num_sab_heads: cutlass.Int32,
        num_seqs: cutlass.Int32,
        scale: cutlass.Float32,
        # SMEM staged layouts (needed to view shared_storage tensor buffers)
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        k_trans_smem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        cumsumlog_smem_layout_staged: cute.Layout,
        t_smem_layout_staged: cute.ComposedLayout,
        ainv_smem_layout_staged: cute.ComposedLayout,
        qk_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        scheduler_params: GDNTileSchedulerParams,
        # TMA descriptor workspace in GMEM (one set of 5 slots per CTA)
        # Slots: Q=0, K[0]=1, K[1]=2, V=3, O=4  (each slot = bytes_per_tensormap = 16xInt64)
        mQ,
        mK,
        mV,
        mT,
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
        warp_group_idx = cute.arch.make_warp_uniform(warp_idx // 4)
        bidx, bidy, bidz = cute.arch.block_idx()
        grid_dim = cute.arch.grid_dim()

        if cutlass.const_expr(self.needs_initial_state):
            assert mInitialState is not None, (
                "mInitialState must be provided if needs_initial_state is True"
            )
        else:
            assert mInitialState is None, (
                "mInitialState must be None if needs_initial_state is False"
            )

        # Checkpointing is not part of the CP prefill contract. These aliases keep
        # the source-equivalent CG1 epilogue specialized on the disabled path.
        mS_checkpoints = mStateOut
        checkpoint_every_n_tokens = cutlass.Int32(0)

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
            tensormap_workspace[(cta_linear_idx, 4, None)].iterator
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
        sV = storage.sV.get_tensor(
            v_smem_layout_staged.outer, swizzle=v_smem_layout_staged.inner
        )
        sT = storage.sT.get_tensor(
            t_smem_layout_staged.outer, swizzle=t_smem_layout_staged.inner
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
        # Gate scalar arrays (1D Float32, flat - no swizzle)
        sCumsumlog = storage.cumsumlog.get_tensor(cumsumlog_smem_layout_staged)
        sCumprod = storage.cumprod.get_tensor(cumsumlog_smem_layout_staged)

        if warp_idx == self.mma_cg0_warp_id:
            cpasync.prefetch_descriptor(tma_q.atom)
            cpasync.prefetch_descriptor(tma_k.atom)
            cpasync.prefetch_descriptor(tma_v.atom)
            cpasync.prefetch_descriptor(tma_t.atom)
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
        cg_tma = _cg(len([self.load_warp_id]))
        # 1 warp (gate/beta ldg/sts path of the unified load warp)
        cg_gate = _cg(self.threads_per_warp * len([self.load_warp_id]))
        # One thread per UMMA issuer; both consume Q and K.
        cg_mma_cg0 = _cg(len([self.mma_cg0_warp_id]))
        cg_mma_cg1 = _cg(len([self.mma_cg1_warp_id]))
        cg_mma_both = _cg(len([self.mma_cg0_warp_id, self.mma_cg1_warp_id]))
        # 128 threads (CG0)
        cg_cg0 = _cg(self.threads_per_warp * len(self.compute_group_0_warp_ids))
        # One elected thread per CG0 warp releases each TMA T stage.
        cg_cg0_t = _cg(len(self.compute_group_0_warp_ids))
        # Both fixed State halves.
        cg_state = _cg(
            self.threads_per_warp
            * (len(self.compute_group_1_warp_ids) + len(self.compute_group_2_warp_ids))
        )
        # One thread per State warp, used for V load signaling.
        cg_state_v = _cg(
            len(self.compute_group_1_warp_ids) + len(self.compute_group_2_warp_ids)
        )
        # CG0 plus both State halves.
        cg_both = _cg(
            self.threads_per_warp * len(self.compute_group_0_warp_ids)
            + self.threads_per_warp
            * (len(self.compute_group_1_warp_ids) + len(self.compute_group_2_warp_ids))
        )
        # 32 threads (epilogue warp)
        cg_epi = _cg(self.threads_per_warp * len([self.epilogue_warp_id]))

        # Unified load warp -> MMA: K, Q, V.
        load_k_producer, load_k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_k_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_k_bytes,
            barrier_storage=storage.load_k_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_q_producer, load_q_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=self.smem_q_stages,
            producer_group=cg_tma,
            consumer_group=cg_mma_both,
            tx_count=self.tma_q_bytes,
            barrier_storage=storage.load_q_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        load_v_producer, load_v_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_v_stages,
            producer_group=cg_tma,
            consumer_group=cg_state_v,
            tx_count=self.tma_v_bytes,
            barrier_storage=storage.load_v_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()
        # Unified load warp -> CG0 / State groups: gate/beta (PipelineAsync, software-signaled).
        # ldg/ldgsts paths do not use TMA barriers; producer calls commit() after writes.
        load_gate_producer, load_gate_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_gate_stages,
            producer_group=cg_gate,
            consumer_group=cg_both,
            barrier_storage=storage.load_gate_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        load_t_producer, load_t_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=self.smem_t_stages,
            producer_group=cg_tma,
            consumer_group=cg_cg0_t,
            tx_count=self.tma_t_bytes,
            barrier_storage=storage.load_t_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> both State groups: kv_acc
        kv_acc_producer, kv_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_kv_acc_stages,
            producer_group=cg_mma_cg1,
            consumer_group=cg_state,
            barrier_storage=storage.kv_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # MMA warp -> both State groups: q_state_acc
        q_state_acc_producer, q_state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_q_state_acc_stages,
            producer_group=cg_mma_cg1,
            consumer_group=cg_state,
            barrier_storage=storage.q_state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # Auxiliary MMA warp -> CG0: KK/QK accumulators.
        aux_acc_producer, aux_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_aux_acc_stages,
            producer_group=cg_mma_cg0,
            consumer_group=cg_cg0,
            barrier_storage=storage.aux_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # State MMA warp -> both State groups: SK/NewV accumulators.
        state_acc_producer, state_acc_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=self.tmem_state_acc_stages,
            producer_group=cg_mma_cg1,
            consumer_group=cg_state,
            barrier_storage=storage.state_acc_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  a_inv_done
        a_inv_ready_producer, a_inv_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_ainv_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma_cg1,
            barrier_storage=storage.ainv_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # CG0 -> MMA warp:  qk_done
        qk_ready_producer, qk_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=self.smem_qk_stages,
            producer_group=cg_cg0,
            consumer_group=cg_mma_cg1,
            barrier_storage=storage.qk_ready_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # Both State groups -> MMA warp: state_inp_ready
        state_inp_ready_producer, state_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_state_inp_stages,
                producer_group=cg_state,
                consumer_group=cg_mma_cg1,
                barrier_storage=storage.state_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # Both State groups -> MMA warp: shared_inp_ready
        shared_inp_ready_producer, shared_inp_ready_consumer = (
            pipeline.PipelineAsyncUmma.create(
                num_stages=self.tmem_shared_inp_stages,
                producer_group=cg_state,
                consumer_group=cg_mma_cg1,
                barrier_storage=storage.shared_inp_ready_mbar_ptr.data_ptr(),
                defer_sync=True,
            ).make_participants()
        )

        # Both State groups -> epilogue warp: output_ready
        o_store_producer, o_store_consumer = pipeline.PipelineAsync.create(
            num_stages=self.smem_o_stages,
            producer_group=cg_state,
            consumer_group=cg_epi,
            barrier_storage=storage.o_store_mbar_ptr.data_ptr(),
            defer_sync=True,
        ).make_participants()

        # Seed the ordered State-group handoff so the left half owns the first copy phase.
        if cutlass.const_expr(self.order_state_groups):
            if warp_group_idx == 2:
                self.state_order_barrier_0.arrive()

        pipeline_init_arrive(is_relaxed=True)

        pipeline_init_wait()

        if warp_group_idx == 3:
            cute.arch.setmaxregister_decrease(self.num_regs_other)

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
            cute.arch.setmaxregister_decrease(self.num_regs_compute_group_0)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                head_idx, _, batch_start, seqlen_b, _, _, _ = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)

                sQk_pisl = self._transform_to_position_independent_layout(
                    sQk, qk_smem_layout_staged.inner
                )
                # First chunk: no previous state (S_prev = 0), skip GEMMs 3/4
                for chunk_idx in cutlass.range(num_chunks_b):
                    (
                        load_gate_consumer,
                        load_t_consumer,
                        aux_acc_consumer,
                        a_inv_ready_producer,
                        qk_ready_producer,
                    ) = self.compute_group_0_cp(
                        tidx,
                        tmem_ptr,
                        scale,
                        (tiled_mma_qk,),
                        (sCumsumlog, sT, sAinv, sQk_pisl),
                        (
                            load_gate_consumer,
                            load_t_consumer,
                            aux_acc_consumer,
                            a_inv_ready_producer,
                            qk_ready_producer,
                        ),
                        (
                            chunk_idx == num_chunks_b - 1,
                            seqlen_b - chunk_idx * self.b_t,
                        ),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            a_inv_ready_producer.tail()
            qk_ready_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 1 (warps 4-7)
        # ==============================================================
        if (
            warp_idx >= self.compute_group_1_warp_ids[0]
            and warp_idx <= self.compute_group_1_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_1)
            # The layouts use all 512 TMEM columns.
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            _, _, _, _, tTR_cState, _, _, _ = self._state_half_tmem_copies(
                tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, 0
            )
            rState = cute.make_rmem_tensor_like(tTR_cState, self.acc_dtype)

            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                (
                    head_idx,
                    cp_chunk_idx_in_seq,
                    batch_start,
                    seqlen_b,
                    num_cp_chunks,
                    cp_chunk_idx,
                    _,
                ) = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                checkpoint_offset = 0
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                if num_chunks_b > 0:
                    self._initialize_cp_state_half(
                        tidx,
                        mFixedState,
                        mInitialState,
                        head_idx,
                        seq_idx,
                        cp_chunk_idx_in_seq,
                        cp_chunk_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        tiled_mma_qs,
                        rState,
                        0,
                    )
                    (
                        load_v_consumer,
                        load_gate_consumer,
                        state_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                        rState,
                    ) = self.compute_state_group(
                        tidx,
                        tmem_ptr,
                        scale,
                        rState,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sCumsumlog,
                            sCumprod,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            state_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (True, 0, head_idx),
                        0,
                    )
                for chunk_idx in cutlass.range(1, num_chunks_b):
                    (
                        load_v_consumer,
                        load_gate_consumer,
                        state_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                        rState,
                    ) = self.compute_state_group(
                        tidx,
                        tmem_ptr,
                        scale,
                        rState,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sCumsumlog,
                            sCumprod,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            state_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (False, chunk_idx, head_idx),
                        0,
                    )
                if num_chunks_b > 0 and cp_chunk_idx_in_seq == num_cp_chunks - 1:
                    self._store_state_half(
                        tidx,
                        mStateOut,
                        head_idx,
                        seq_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        tiled_mma_qs,
                        rState,
                        0,
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

            shared_inp_ready_producer.tail()
            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # COMPUTE WARP GROUP 2 (warps 8-11): right State / high token half
        # ==============================================================
        elif (
            warp_idx >= self.compute_group_2_warp_ids[0]
            and warp_idx <= self.compute_group_2_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.num_regs_compute_group_2)
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            _, _, _, _, tTR_cState, _, _, _ = self._state_half_tmem_copies(
                tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, 1
            )
            rState = cute.make_rmem_tensor_like(tTR_cState, self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                (
                    head_idx,
                    cp_chunk_idx_in_seq,
                    batch_start,
                    seqlen_b,
                    num_cp_chunks,
                    cp_chunk_idx,
                    _,
                ) = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                checkpoint_offset = 0
                sV_pisl = self._transform_to_position_independent_layout(
                    sV, v_smem_layout_staged.inner
                )
                sO_pisl = self._transform_to_position_independent_layout(
                    sO, o_smem_layout_staged.inner
                )
                if num_chunks_b > 0:
                    self._initialize_cp_state_half(
                        tidx,
                        mFixedState,
                        mInitialState,
                        head_idx,
                        seq_idx,
                        cp_chunk_idx_in_seq,
                        cp_chunk_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        tiled_mma_qs,
                        rState,
                        1,
                    )
                    (
                        load_v_consumer,
                        load_gate_consumer,
                        state_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                        rState,
                    ) = self.compute_state_group(
                        tidx,
                        tmem_ptr,
                        scale,
                        rState,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sCumsumlog,
                            sCumprod,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            state_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (True, 0, head_idx),
                        1,
                    )
                for chunk_idx in cutlass.range(1, num_chunks_b):
                    (
                        load_v_consumer,
                        load_gate_consumer,
                        state_acc_consumer,
                        kv_acc_consumer,
                        q_state_acc_consumer,
                        kv_acc_producer,
                        state_inp_ready_producer,
                        shared_inp_ready_producer,
                        o_store_producer,
                        checkpoint_offset,
                        rState,
                    ) = self.compute_state_group(
                        tidx,
                        tmem_ptr,
                        scale,
                        rState,
                        (tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv),
                        (
                            sV_pisl,
                            sCumsumlog,
                            sCumprod,
                            sO_pisl,
                        ),
                        (mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens),
                        (
                            load_v_consumer,
                            load_gate_consumer,
                            state_acc_consumer,
                            kv_acc_consumer,
                            q_state_acc_consumer,
                            kv_acc_producer,
                            state_inp_ready_producer,
                            shared_inp_ready_producer,
                            o_store_producer,
                        ),
                        (False, chunk_idx, head_idx),
                        1,
                    )
                if num_chunks_b > 0 and cp_chunk_idx_in_seq == num_cp_chunks - 1:
                    self._store_state_half(
                        tidx,
                        mStateOut,
                        head_idx,
                        seq_idx,
                        tmem_ptr,
                        tiled_mma_kv,
                        tiled_mma_qs,
                        rState,
                        1,
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            shared_inp_ready_producer.tail()
            o_store_producer.tail()
            state_inp_ready_producer.tail()

        # ==============================================================
        # CG0 MMA ISSUER (warp 12): GEMMs 1-2
        # ==============================================================
        elif warp_idx == self.mma_cg0_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()
            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                head_idx, _, _, seqlen_b, _, _, _ = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                for _ in cutlass.range(num_chunks_b):
                    (
                        aux_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                    ) = self.mma_cg0_warp(
                        tmem_ptr,
                        (tiled_mma_qk, tiled_mma_qkv),
                        (sQ, sK),
                        (aux_acc_producer, load_k_consumer, load_q_consumer),
                        (),
                    )

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()
            aux_acc_producer.tail()

        # ==============================================================
        # CG1 MMA ISSUER (warp 13): GEMMs 3-7
        # ==============================================================
        elif warp_idx == self.mma_cg1_warp_id:
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()
            v_stage_idx = cutlass.Int32(0)
            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                head_idx, _, batch_start, seqlen_b, _, _, _ = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                num_chunks_b = cute.ceil_div(seqlen_b, self.b_t)
                if num_chunks_b > 0:
                    (
                        state_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                    ) = self.mma_cg1_warp(
                        tmem_ptr,
                        (tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv),
                        (sQ, sK, sK_trans, sV, sAinv, sQk),
                        (
                            state_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            load_v_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            shared_inp_ready_consumer,
                        ),
                        (True, v_stage_idx),
                    )
                    v_stage_idx = (v_stage_idx + 1) % self.smem_v_stages
                for chunk_idx in cutlass.range(1, num_chunks_b):  # noqa: B007
                    (
                        state_acc_producer,
                        q_state_acc_producer,
                        kv_acc_producer,
                        load_k_consumer,
                        load_q_consumer,
                        load_v_consumer,
                        a_inv_ready_consumer,
                        qk_ready_consumer,
                        state_inp_ready_consumer,
                        shared_inp_ready_consumer,
                    ) = self.mma_cg1_warp(
                        tmem_ptr,
                        (tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv),
                        (sQ, sK, sK_trans, sV, sAinv, sQk),
                        (
                            state_acc_producer,
                            q_state_acc_producer,
                            kv_acc_producer,
                            load_k_consumer,
                            load_q_consumer,
                            load_v_consumer,
                            a_inv_ready_consumer,
                            qk_ready_consumer,
                            state_inp_ready_consumer,
                            shared_inp_ready_consumer,
                        ),
                        (False, v_stage_idx),
                    )
                    v_stage_idx = (v_stage_idx + 1) % self.smem_v_stages

                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            state_acc_producer.tail()
            q_state_acc_producer.tail()
            kv_acc_producer.tail()

        # ==============================================================
        # UNIFIED LOAD WARP (warp 14)
        # ==============================================================
        elif warp_idx == self.load_warp_id:
            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()

            # Init base descriptors once into GMEM (copies embedded atom descriptor)
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_q.atom, tensormap_q_ptr, self.load_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_k.atom, tensormap_k_ptr, self.load_warp_id
                )
                tensormap_manager.init_tensormap_from_atom(
                    tma_v.atom, tensormap_v_ptr, self.load_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                head_idx, _, batch_start, seqlen_b, _, _, t_block_start = (
                    self.get_cp_work(
                        cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                    )
                )
                batch_end = batch_start + seqlen_b
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
                if num_chunks_b > 0:
                    # Update K/Q/V descriptors
                    tensormap_manager.update_tensormap(
                        (bounded_q, bounded_k, bounded_v),
                        (tma_q.atom, tma_k.atom, tma_v.atom),
                        (tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr),
                        self.load_warp_id,
                        (None, None, None),
                    )

                # Full tiles: load K, Q, V, T, and gate in order.
                for chunk_idx in cutlass.range(num_chunks_b - 1):
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    load_q_producer, load_k_producer, load_v_producer = (
                        self.tma_qkv_warp(
                            (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                            (tma_q, tma_k, tma_v),
                            (sQ, sK, sV),
                            (load_q_producer, load_k_producer, load_v_producer),
                            (chunk_offset, chunk_idx, seq_idx, head_idx),
                            (
                                tensormap_manager,
                                tensormap_q_ptr,
                                tensormap_k_ptr,
                                tensormap_v_ptr,
                            ),
                        )
                    )
                    load_t_producer = self.tma_t_warp(
                        tiled_mma_qkv,
                        tma_t,
                        sT,
                        load_t_producer,
                        t_block_start + chunk_idx,
                        head_idx,
                    )
                    load_gate_producer = self.load_gate_warp(
                        tidx,
                        mGate,
                        (sCumsumlog, sCumprod),
                        load_gate_producer,
                        (chunk_offset, head_idx, False, batch_end),
                    )

                if num_chunks_b > 0:
                    chunk_idx = num_chunks_b - 1
                    chunk_offset = batch_start + chunk_idx * self.b_t
                    load_q_producer, load_k_producer, load_v_producer = (
                        self.tma_qkv_warp(
                            (tiled_mma_qk, tiled_mma_qkv_ss, tiled_mma_kv),
                            (tma_q, tma_k, tma_v),
                            (sQ, sK, sV),
                            (load_q_producer, load_k_producer, load_v_producer),
                            (chunk_offset, chunk_idx, seq_idx, head_idx),
                            (
                                tensormap_manager,
                                tensormap_q_ptr,
                                tensormap_k_ptr,
                                tensormap_v_ptr,
                            ),
                        )
                    )
                    load_t_producer = self.tma_t_warp(
                        tiled_mma_qkv,
                        tma_t,
                        sT,
                        load_t_producer,
                        t_block_start + chunk_idx,
                        head_idx,
                    )
                    load_gate_producer = self.load_gate_warp(
                        tidx,
                        mGate,
                        (sCumsumlog, sCumprod),
                        load_gate_producer,
                        (chunk_offset, head_idx, True, batch_end),
                    )
                scheduler.advance_to_next_work()
                work = scheduler.get_current_work()

            load_q_producer.tail()
            load_k_producer.tail()
            load_v_producer.tail()
            load_t_producer.tail()
            load_gate_producer.tail()

        # ==============================================================
        # EPILOGUE WARP (warp 15)
        # ==============================================================
        if warp_idx == self.epilogue_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            scheduler = GDNTileScheduler.create(
                scheduler_params,
                (bidy, bidx, bidz),
                (grid_dim[1], grid_dim[0], grid_dim[2]),
            )
            work = scheduler.initial_work_tile_info()

            # Init O descriptor once into GMEM
            if work.is_valid_tile:
                tensormap_manager.init_tensormap_from_atom(
                    tma_o.atom, tensormap_o_ptr, self.epilogue_warp_id
                )
                tensormap_manager.fence_tensormap_initialization()

            while work.is_valid_tile:
                seq_idx, flat_work_idx, _ = work.tile_idx
                head_idx, _, batch_start, seqlen_b, _, _, _ = self.get_cp_work(
                    cu_seqlens, seq_idx, flat_work_idx, num_sab_heads, cp_chunk_len
                )
                batch_end = batch_start + seqlen_b
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
    def tma_t_warp(
        self,
        tiled_mma_qkv: cute.TiledMma,
        tma_t: TmaInfo,
        sT: cute.Tensor,
        load_t_producer: pipeline.PipelineProducer,
        t_block_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
    ) -> pipeline.PipelineProducer:
        """Load one signed, beta-folded T tile into its MMA-B SMEM layout."""
        t_handle = load_t_producer.acquire_and_advance()
        mT = tma_t.tma_tensor[None, None, head_idx, t_block_idx]
        gT = cute.flat_divide(
            mT,
            (self.b_t, self.b_t),
        )
        tCgT = tiled_mma_qkv.get_slice(0).partition_B(gT)
        tTsT, tTgT = cpasync.tma_partition(
            tma_t.atom,
            0,
            cute.make_layout(1),
            cute.group_modes(sT, 0, 3),
            cute.group_modes(tCgT, 0, 3),
        )
        cute.copy(
            tma_t.atom,
            tTgT[(None, 0, 0)],
            tTsT[(None, t_handle.index)],
            tma_bar_ptr=t_handle.barrier,
        )
        return load_t_producer

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
        pipeline.PipelineProducer, pipeline.PipelineProducer, pipeline.PipelineProducer
    ]:
        """Unified load warp: load K, Q, and V for the current chunk.

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
        tma_q, tma_k, tma_v = tma_args
        sQ, sK, sV = smem_args
        load_q_producer, load_k_producer, load_v_producer = pipeline_args
        chunk_offset, chunk_idx, batch_idx, head_idx = work_args
        tensormap_manager, tensormap_q_ptr, tensormap_k_ptr, tensormap_v_ptr = (
            tensormap_args
        )

        # Single-CTA mode: no multicast, cta_v = 0.
        cta_layout = cute.make_layout(1)

        # Per-thread MMA slices (cta_v=0 for ONE-CTA mode).
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_qkv_ss = tiled_mma_qkv_ss.get_slice(0)
        thr_mma_kv = tiled_mma_kv.get_slice(0)  # noqa: F841

        # Tile domains are Q/K=(BT=64, DK=128), V=(DV=128, BT=64).
        #   mode[0,2] = (BT, DK) - M,K tile for A (Q) and for B (K) of tiled_mma_qk
        #   mode[1,2] = (BT, DV) - tile shape for loading V (B operand in GEMMs 5/6)
        # (BT, DK)
        qk_tile = cute.select(self.mma_tiler_qk, mode=[0, 2])
        # (DV, BT)
        v_tile = cute.select(self.mma_tiler_qkv, mode=[0, 2])

        # ------------------------------------------------------------------
        # K  (B operand of GEMM-kk / GEMM-qk, multi-stage buffered)
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
        # Q  (A operand of GEMM-qk, multi-stage buffered)
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
        # V  (B operand of GEMM-new_v / GEMM-qkv, multi-stage buffered)
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

        return load_q_producer, load_k_producer, load_v_producer

    @cute.jit
    def load_gate_warp(
        self,
        tidx: cutlass.Int32,
        gate: cute.Tensor,
        smem_args: tuple,
        load_gate_producer: pipeline.PipelineProducer,
        work_args: tuple,
    ) -> pipeline.PipelineProducer:
        """Warp 10: load and prefix-process gate[BT] for the current chunk.

        Gate is loaded via ldg (sync G->R), then preprocessed into cumsum-log
        and cumprod channels in SMEM.

        The last tile uses predicated copies: elements with linear index >= valid_tokens
        are out-of-bounds and receive the neutral gate value one.

        Thread tidx (lane 0..31) owns positions tidx, tidx+32, tidx+64, tidx+96.
        """
        sCumsumlog, sCumprod = smem_args
        chunk_offset, head_idx, is_last_tile, batch_end = work_args

        # lane index
        lidx = tidx % self.threads_per_warp

        gGate = cute.domain_offset((chunk_offset,), gate[None, head_idx])
        cGate = cute.domain_offset(
            (chunk_offset,), cute.make_identity_tensor(gate[None, head_idx].shape)
        )
        gGate = cute.flat_divide(gGate, (self.b_t,))[None, 0]
        cGate = cute.flat_divide(cGate, (self.b_t,))[None, 0]

        # Tiled copy: 1D thread/value layouts; partition_S/D handle element mapping.
        # thread_layout (32,): each of the 32 lanes maps to one row of the b_t tile.
        # value_layout  (4,) : each lane owns 4 elements strided by threads_per_warp.
        thread_layout = cute.make_layout((self.threads_per_warp,), stride=(1,))
        value_layout = cute.make_layout((1,), stride=(1,))

        # Gate: sync G->R (ldg), apply ln + prefix sum, then R->S (sts)
        atom_gate_g2r = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Float32, num_bits_per_copy=32
        )
        tiled_copy_gate_g2r = cute.make_tiled_copy_tv(
            atom_gate_g2r, thread_layout, value_layout
        )

        # Per-thread partitions (1D tensors; no manual 2D reshaping needed)
        thr_copy_gate_g2r = tiled_copy_gate_g2r.get_slice(lidx)
        tGgGate = thr_copy_gate_g2r.partition_S(gGate)
        tGsCumsumlog = thr_copy_gate_g2r.partition_D(sCumsumlog)
        tGsCumprod = thr_copy_gate_g2r.partition_D(sCumprod)

        gate_handle = load_gate_producer.acquire_and_advance()

        rGate = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrGate = tiled_copy_gate_g2r.retile(rGate)
        rCumprod = cute.make_rmem_tensor_like(tGgGate, self.acc_dtype)
        tGrCumprod = tiled_copy_gate_g2r.retile(rCumprod)

        # --- Predicate (last tile only): compute once, reuse for gate and beta ---
        if cutlass.const_expr(is_last_tile):
            valid_tokens = batch_end  # noqa: F841
            tGcGate = thr_copy_gate_g2r.partition_S(cGate)
            tGpGate = cute.make_rmem_tensor(
                ((tGcGate.shape[0][1],), tGcGate.shape[1]), cutlass.Boolean
            )
            for i in range(cute.size(tGpGate)):
                tGpGate[i] = cute.elem_less(tGcGate[i][0], batch_end)

        # --- Gate load ---
        if cutlass.const_expr(is_last_tile):
            # OOB neutral: 1.0 -> log2 ~= 0.0 (no decay contribution)
            tGrGate.fill(1.0)
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate, pred=tGpGate)
        else:
            cute.copy(tiled_copy_gate_g2r, tGgGate, tGrGate)

        # --- log2 + warp-wide inclusive prefix sum + SMEM store (always) ---
        for i in range(cute.size(tGrGate)):
            tGrGate[i] = cute.math.log2(tGrGate[i] + 1e-10, fastmath=True)
        for offset in [1, 2, 4, 8, 16]:
            for col in range(cute.size(tGrGate)):
                n = cute.arch.shuffle_sync_up(
                    tGrGate[col], offset, mask=0xFFFFFFFF, mask_and_clamp=0
                )
                if lidx >= offset:
                    tGrGate[col] = tGrGate[col] + n
        sum_v = 0.0  # noqa: F841
        for col in range(1, cute.size(tGrGate)):
            last_v = cute.arch.shuffle_sync(
                tGrGate[col - 1],
                self.threads_per_warp - 1,
                mask=0xFFFFFFFF,
                mask_and_clamp=self.threads_per_warp - 1,
            )
            tGrGate[col] += last_v
        for col in range(cute.size(tGrGate)):
            tGrCumprod[col] = cute.math.exp2(tGrGate[col], fastmath=True)
        cute.copy(
            tiled_copy_gate_g2r, tGrGate, tGsCumsumlog[None, None, 0, gate_handle.index]
        )
        cute.copy(
            tiled_copy_gate_g2r,
            tGrCumprod,
            tGsCumprod[None, None, 0, gate_handle.index],
        )
        gate_handle.commit()

        return load_gate_producer

    @cute.jit
    def mma_cg0_warp(
        self,
        tmem_ptr: cutlass.Int64,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ) -> tuple[
        pipeline.PipelineProducer,
        pipeline.PipelineConsumer,
        pipeline.PipelineConsumer,
    ]:
        """Warp 12: issue Q @ K.T for the CP score path."""
        tiled_mma_qk, tiled_mma_qkv = mma_args
        sQ, sK = smem_args
        shared_acc_producer, load_k_consumer, load_q_consumer = pipeline_args

        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_aux_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_aux_acc_offset, tCtAcc_fake.layout
        )

        tCrK_B = tiled_mma_qk.make_fragment_B(sK)
        tCrQ_A = tiled_mma_qk.make_fragment_A(sQ)

        k_handle = load_k_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()
        qk_handle = shared_acc_producer.acquire_and_advance()
        num_kphases = cute.size(tCrQ_A, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            tiled_mma_qk.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_qk,
                tCtShared[None, None, None, qk_handle.index],
                tCrQ_A[None, None, kphase_idx, q_handle.index],
                tCrK_B[None, None, kphase_idx, k_handle.index],
                tCtShared[None, None, None, qk_handle.index],
            )
        qk_handle.commit()
        q_handle.release()
        k_handle.release()

        return shared_acc_producer, load_k_consumer, load_q_consumer

    @cute.jit
    def mma_cg1_warp(
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
    ]:
        """Warp 13: issue GEMMs 3-7 for compute group 1."""
        tiled_mma_qs, tiled_mma_qkv, tiled_mma_qkv_ss, tiled_mma_kv = mma_args
        sQ, sK, sK_trans, sV, sAinv, sQk = smem_args
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
        ) = pipeline_args
        is_first_chunk, v_stage_idx = work_args

        valid_state = not is_first_chunk or self.use_initial_state

        # ------------------------------------------------------------------
        # Build TMEM accumulator views
        # ------------------------------------------------------------------
        # Dedicated two-stage state accumulator ring for G3/G5. G1/G2 use a
        # separate two-stage ring and therefore progress independently.
        acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtAcc_fake = tiled_mma_qkv.make_fragment_C(
            cute.append(acc_shape, self.tmem_state_acc_stages)
        )
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_state_acc_offset, tCtAcc_fake.layout
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
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
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
        # Pre-create operand fragments (stage dim preserved; sliced at GEMM time).
        # K as A for GEMM 3 (k*state)
        tCrS_A = tCtState_inp
        # Q as A for GEMM 4 (q*state)
        tCrQ_B_qs = tiled_mma_qs.make_fragment_B(sQ)
        # S_prev as B for GEMMs 3+4
        tCrK_B_qs = tiled_mma_qs.make_fragment_B(sK)
        # tiled_mma_qkv operands (GEMMs 5, 6)
        # V-KS as A for GEMM 5
        if cutlass.const_expr(valid_state):
            tCrV_A = tCtShared_inp
        else:
            tCrV_A = tiled_mma_qkv_ss.make_fragment_A(sV)
        # A_inv as B for GEMM 5
        tCrAinv_B = tiled_mma_qkv.make_fragment_B(sAinv)

        # W_qkv as A for GEMM 6
        tCrQkv_A = tCtShared_inp
        # NV as B for GEMM 6
        tCrNv_B = tiled_mma_qkv.make_fragment_B(sQk)
        # tiled_mma_kv operands (GEMM 7)
        # delta as A for GEMM 7
        tCrDecayV_A = tCtShared_inp
        # K^T as B for GEMM 7
        tCrKt_B = tiled_mma_kv.make_fragment_B(sK_trans)

        k_handle = load_k_consumer.wait_and_advance()
        q_handle = load_q_consumer.wait_and_advance()

        # ---- GEMM 3: k*state  (K @ S_prev -> shared acc) --------------------
        # ---- GEMM 4: q*state  (Q @ S_prev -> tmem_q_state) ------------------
        # Skipped on the first chunk when use_initial_state is False (S_prev = 0, outputs are zero).
        if valid_state:
            s_handle = state_inp_ready_consumer.wait_and_advance()
            ks_handle = shared_acc_producer.acquire_and_advance()

            num_kphases_qs = cute.size(tCrS_A, mode=[2])
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtShared[None, None, None, ks_handle.index],
                    tCrS_A[None, None, kphase_idx, s_handle.index],
                    tCrK_B_qs[None, None, kphase_idx, k_handle.index],
                    tCtShared[None, None, None, ks_handle.index],
                )
            ks_handle.commit()

            q_state_acc_handle = q_state_acc_producer.acquire_and_advance()
            # S_prev still loaded (same s_handle.index as GEMM 3).
            num_kphases_qs = cute.size(tCrS_A, mode=[2])
            for kphase_idx in cutlass.range(num_kphases_qs, unroll_full=True):
                tiled_mma_qs.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
                cute.gemm(
                    tiled_mma_qs,
                    tCtQState[None, None, None, q_state_acc_handle.index],
                    tCrS_A[None, None, kphase_idx, s_handle.index],
                    tCrQ_B_qs[None, None, kphase_idx, q_handle.index],
                    tCtQState[None, None, None, q_state_acc_handle.index],
                )
            q_state_acc_handle.commit()
            # Release state SMEM (S_prev fully consumed by GEMMs 3 + 4).
            s_handle.release()

        q_handle.release()

        # ---- GEMM 5: new_v  (A_inv @ (V-KS) -> shared acc) ------------------
        # A_inv from CG0 (a_inv_ready); V-KS from CG1 (v_ks_ready, stored in sV).
        vks_handle = shared_inp_ready_consumer.wait_and_advance()
        nv_handle = shared_acc_producer.acquire_and_advance()
        ainv_handle = a_inv_ready_consumer.wait_and_advance()

        num_kphases_qkv = cute.size(tCrAinv_B, mode=[2])
        cur_tiled_mma_qkv = tiled_mma_qkv if valid_state else tiled_mma_qkv_ss
        v_operand_stage = vks_handle.index if valid_state else v_stage_idx
        for kphase_idx in cutlass.range(num_kphases_qkv, unroll_full=True):
            cur_tiled_mma_qkv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                cur_tiled_mma_qkv,
                tCtShared[None, None, None, nv_handle.index],
                tCrV_A[None, None, kphase_idx, v_operand_stage],
                tCrAinv_B[None, None, kphase_idx, ainv_handle.index],
                tCtShared[None, None, None, nv_handle.index],
            )

        nv_handle.commit()
        ainv_handle.release()
        vks_handle.release()

        # ---- GEMM 6: qkv  (W_qkv @ NV -> q * state acc) ------------------------
        # W_qkv from CG0 (qk_ready, stored in sQk); NV from CG1 (new_v_ready, stored in sNv).
        qkv_qk_handle = qk_ready_consumer.wait_and_advance()
        qkv_nv_handle = shared_inp_ready_consumer.wait_and_advance()
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

        # ---- GEMM 7: kv_update  (K^T @ delta -> state-increment TMEM) -----------
        # delta from CG1 (decay_v_ready, stored in sDecayV); K^T reuses k_handle slot.
        kv_acc_handle = kv_acc_producer.acquire_and_advance()
        dv_handle = shared_inp_ready_consumer.wait_and_advance()

        num_kphases_kv = cute.size(tCrKt_B, mode=[2])
        for kphase_idx in cutlass.range(num_kphases_kv, unroll_full=True):
            tiled_mma_kv.set(tcgen05.Field.ACCUMULATE, kphase_idx != 0)
            cute.gemm(
                tiled_mma_kv,
                tCtState[None, None, None, kv_acc_handle.index],
                tCrDecayV_A[None, None, kphase_idx, dv_handle.index],
                tCrKt_B[None, None, kphase_idx, k_handle.index],
                tCtState[None, None, None, kv_acc_handle.index],
            )
        kv_acc_handle.commit()
        dv_handle.release()
        # K SMEM slot now free for next chunk
        k_handle.release()

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
        )

    @cute.jit
    def compute_group_0_cp(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        mma_args: tuple,
        smem_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
    ):
        """Materialize gamma-scaled QK and signed-T operands for tcgen05."""
        (tiled_mma_qk,) = mma_args
        sCumsumlog, sT, sAinv, sQk = smem_args
        (
            load_gate_consumer,
            load_t_consumer,
            aux_acc_consumer,
            t_ready_producer,
            qk_ready_producer,
        ) = pipeline_args
        is_final_block, valid_tokens = work_args

        num_threads_cg0 = self.threads_per_warp * len(self.compute_group_0_warp_ids)
        cg0_tidx = tidx % num_threads_cg0
        tAcc_shape = tiled_mma_qk.partition_shape_C(
            (self.mma_tiler_qk[0], self.mma_tiler_qk[1])
        )
        tAcc = tiled_mma_qk.make_fragment_C(tAcc_shape)
        tAcc = cute.make_tensor(
            tAcc.iterator,
            cute.flat_product(
                tAcc.layout,
                cute.make_layout((self.tmem_aux_acc_stages,), stride=(1,)),
            ),
        )
        tStS = cute.make_tensor(tmem_ptr + self.tmem_aux_acc_offset, tAcc.layout)
        tStS_mn = self.transform_partitioned_tensor_layout(tStS)
        cS = cute.make_identity_tensor((self.mma_tiler_qk[0], self.mma_tiler_qk[1]))
        atom_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_t2r = tcgen05.make_tmem_copy(atom_t2r, tStS[(None, None), 0, 0, 0])
        thr_t2r = tiled_t2r.get_slice(cg0_tidx)
        tTR_tStS = thr_t2r.partition_S(tStS_mn)
        tTR_cS = thr_t2r.partition_D(cS)

        sT_mn = self.transform_partitioned_tensor_layout(sT)
        sAinv_mn = self.transform_partitioned_tensor_layout(sAinv)

        gate_handle = load_gate_consumer.wait_and_advance()
        t_handle = load_t_consumer.wait_and_advance()
        t_ready_handle = t_ready_producer.acquire_and_advance()
        copy_mma = cute.make_tiled_mma(
            cute.nvgpu.warp.MmaF16BF16Op(self.io_dtype, self.acc_dtype, (16, 8, 16)),
            cute.make_layout((4, 1, 1)),
            permutation_mnk=(self.b_t, self.b_t, 16),
        )
        atom_t_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_t_s2r = cute.make_tiled_copy_C(atom_t_s2r, copy_mma)
        thr_t_s2r = tiled_t_s2r.get_slice(cg0_tidx)
        tTsT = thr_t_s2r.partition_S(sT_mn)
        rT = cute.make_rmem_tensor(
            copy_mma.partition_shape_C((self.b_t, self.b_t)),
            self.io_dtype,
        )
        tTrT = thr_t_s2r.retile(rT)
        cute.copy(
            tiled_t_s2r,
            tTsT[None, None, None, t_handle.index],
            tTrT,
        )

        cT = cute.make_identity_tensor((self.b_t, self.b_t))
        tTcT = thr_t_s2r.partition_D(cT)
        tTcT = thr_t_s2r.retile(tTcT)
        for i in cutlass.range_constexpr(cute.size(tTrT)):
            t, s = tTcT[i]
            pred = s >= t
            if is_final_block:
                pred = pred and s < valid_tokens and t < valid_tokens
            gamma = cutlass.Float32(0.0)
            if pred:
                gamma = cute.math.exp2(
                    sCumsumlog[s, 0, gate_handle.index]
                    - sCumsumlog[t, 0, gate_handle.index],
                    fastmath=True,
                )
            tTrT[i] = self.io_dtype(-gamma * cutlass.Float32(tTrT[i]))

        sAinv_t = cute.make_tensor(
            sAinv_mn.iterator,
            cute.select(sAinv_mn.layout, mode=[1, 0, 2]),
        )
        atom_t_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_t_r2s = cute.make_tiled_copy_C(atom_t_r2s, copy_mma)
        thr_t_r2s = tiled_t_r2s.get_slice(cg0_tidx)
        tTsAinv = thr_t_r2s.partition_D(sAinv_t)
        tTsAinv = cute.make_tensor(tTsAinv.iterator.align(16), tTsAinv.layout)
        tTrT_r2s = thr_t_r2s.retile(rT)
        cute.copy(
            tiled_t_r2s,
            tTrT_r2s,
            tTsAinv[None, None, None, t_ready_handle.index],
        )
        cute.arch.fence_view_async_shared()
        self.inverse_barrier.arrive_and_wait()
        t_handle.release()
        t_ready_handle.commit()

        qk_ready_handle = qk_ready_producer.acquire_and_advance()
        qk_handle = aux_acc_consumer.wait_and_advance()
        tQKrQK = cute.make_rmem_tensor_like(tTR_cS, self.acc_dtype)
        tQKrQK_out = cute.make_rmem_tensor_like(tQKrQK, self.io_dtype)
        atom_qk_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=4, transpose=False),
            self.io_dtype,
        )
        tiled_qk_r2s = cute.make_tiled_copy_D(atom_qk_r2s, tiled_t2r)
        tQsQK = tiled_qk_r2s.get_slice(cg0_tidx).partition_D(
            self.transform_partitioned_tensor_layout(sQk)
        )
        tQrQK = tiled_qk_r2s.retile(tQKrQK_out)
        for sub in cutlass.range_constexpr(tQKrQK.shape[2]):
            cute.copy(
                tiled_t2r,
                tTR_tStS[None, 0, sub, qk_handle.index],
                tQKrQK[None, 0, sub],
            )
            for i in cutlass.range(32):
                s, t = tTR_cS[i, 0, sub]
                pred = s >= t
                if is_final_block:
                    pred = pred and s < valid_tokens and t < valid_tokens
                gamma = cutlass.Float32(0.0)
                if pred:
                    gamma = cute.math.exp2(
                        sCumsumlog[s, 0, gate_handle.index]
                        - sCumsumlog[t, 0, gate_handle.index],
                        fastmath=True,
                    )
                tQKrQK_out[i, 0, sub] = self.io_dtype(tQKrQK[i, 0, sub] * gamma * scale)
            cute.copy(
                tiled_qk_r2s,
                tQrQK[None, 0, sub],
                tQsQK[None, 0, sub, qk_ready_handle.index],
            )
        cute.arch.fence_view_async_shared()
        qk_handle.release()
        qk_ready_handle.commit()
        gate_handle.release()
        return (
            load_gate_consumer,
            load_t_consumer,
            aux_acc_consumer,
            t_ready_producer,
            qk_ready_producer,
        )

    @cute.jit
    def _state_half_tmem_copies(
        self, tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, half_idx
    ):
        state_tidx = tidx % 128
        state_acc_shape = tiled_mma_kv.partition_shape_C(
            (self.mma_tiler_kv[0], self.mma_tiler_kv[1])
        )
        tCtState_fake = tiled_mma_kv.make_fragment_C(
            cute.append(state_acc_shape, self.tmem_kv_acc_stages)
        )
        tCtState = cute.make_tensor(
            tmem_ptr + self.tmem_state_offset, tCtState_fake.layout
        )
        tCtState_mn = self.transform_partitioned_tensor_layout(tCtState)
        tCtState_half = cute.local_tile(tCtState_mn, (128, 64), (0, half_idx, None))
        cState_half = cute.local_tile(
            cute.make_identity_tensor((128, 128)), (128, 64), (0, half_idx)
        )
        state_t2r_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype
        )
        state_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), self.acc_dtype
        )
        tiled_state_t2r = tcgen05.make_tmem_copy(
            state_t2r_atom, tCtState_half[None, None, 0]
        )
        tiled_state_r2t = tcgen05.make_tmem_copy(
            state_r2t_atom, tCtState_half[None, None, 0]
        )
        thr_state_t2r = tiled_state_t2r.get_slice(state_tidx)
        thr_state_r2t = tiled_state_r2t.get_slice(state_tidx)
        tTR_tState = thr_state_t2r.partition_S(tCtState_half)
        tTR_cState = thr_state_t2r.partition_D(cState_half)
        tRT_tState = thr_state_r2t.partition_D(tCtState_half)

        state_inp_shape = tiled_mma_qs.partition_shape_A(
            (self.mma_tiler_qs[0], self.mma_tiler_qs[2])
        )
        tCtStateInp_fake = tiled_mma_qs.make_fragment_A(
            cute.append(state_inp_shape, self.tmem_state_inp_stages)
        )
        tCtStateInp = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_state_inp_offset, dtype=self.io_dtype),
            tCtStateInp_fake.layout,
        )
        tCtStateInp_mn = self.transform_partitioned_tensor_layout(tCtStateInp)
        tCtStateInp_half = cute.local_tile(
            tCtStateInp_mn, (128, 64), (0, half_idx, None)
        )
        cStateInp_half = cute.local_tile(
            cute.make_identity_tensor((128, 128)), (128, 64), (0, half_idx)
        )
        state_inp_r2t_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(8)), self.io_dtype
        )
        tiled_state_inp_r2t = tcgen05.make_tmem_copy(
            state_inp_r2t_atom, tCtStateInp_half[None, None, 0]
        )
        thr_state_inp_r2t = tiled_state_inp_r2t.get_slice(state_tidx)
        tRT_cStateInp = thr_state_inp_r2t.partition_S(cStateInp_half)
        tRT_tStateInp = thr_state_inp_r2t.partition_D(tCtStateInp_half)
        return (
            tiled_state_t2r,
            tiled_state_r2t,
            tiled_state_inp_r2t,
            tTR_tState,
            tTR_cState,
            tRT_tState,
            tRT_cStateInp,
            tRT_tStateInp,
        )

    @cute.jit
    def _load_state_half(
        self,
        tidx,
        source_state,
        head_idx,
        state_idx,
        tmem_ptr,
        tiled_mma_kv,
        tiled_mma_qs,
        rState,
        half_idx,
    ):
        (
            _,
            tiled_state_r2t,
            _,
            _,
            tTR_cState,
            _,
            _,
            _,
        ) = self._state_half_tmem_copies(
            tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, half_idx
        )
        thr_state_r2t = tiled_state_r2t.get_slice(tidx % 128)
        gState = cute.local_tile(
            source_state[None, None, head_idx, state_idx],
            (128, 64),
            (0, half_idx),
        )
        tRgState = thr_state_r2t.partition_S(gState)
        rStateG = cute.make_rmem_tensor_like(tTR_cState, self.state_dtype)
        for sub in cutlass.range(rState.shape[2]):
            cute.autovec_copy(tRgState[None, 0, sub], rStateG[None, 0, sub])
            rState[None, 0, sub].store(rStateG[None, 0, sub].load().to(self.acc_dtype))

    @cute.jit
    def _initialize_cp_state_half(
        self,
        tidx,
        mFixedState,
        mInitialState,
        head_idx,
        seq_idx,
        cp_chunk_idx_in_seq,
        cp_chunk_idx,
        tmem_ptr,
        tiled_mma_kv,
        tiled_mma_qs,
        rState,
        half_idx,
    ):
        if cp_chunk_idx_in_seq == 0:
            if cutlass.const_expr(self.needs_initial_state):
                self._load_state_half(
                    tidx,
                    mInitialState,
                    head_idx,
                    seq_idx,
                    tmem_ptr,
                    tiled_mma_kv,
                    tiled_mma_qs,
                    rState,
                    half_idx,
                )
            else:
                rState.fill(0.0)
        else:
            self._load_state_half(
                tidx,
                mFixedState,
                head_idx,
                cp_chunk_idx - 1,
                tmem_ptr,
                tiled_mma_kv,
                tiled_mma_qs,
                rState,
                half_idx,
            )

    @cute.jit
    def _store_state_half(
        self,
        tidx,
        mS_out,
        head_idx,
        seq_idx,
        tmem_ptr,
        tiled_mma_kv,
        tiled_mma_qs,
        rState,
        half_idx,
    ):
        (
            tiled_state_t2r,
            _,
            _,
            _,
            tTR_cState,
            _,
            _,
            _,
        ) = self._state_half_tmem_copies(
            tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, half_idx
        )
        thr_state_t2r = tiled_state_t2r.get_slice(tidx % 128)
        rStateG = cute.make_rmem_tensor_like(tTR_cState, self.state_dtype)
        if cutlass.const_expr(self.state_dtype != self.acc_dtype):
            rStateG.store(rState.load().to(self.state_dtype))
        else:
            rStateG = rState

        if cutlass.const_expr(self.store_final_state):
            gState = cute.local_tile(
                mS_out[None, None, head_idx, seq_idx], (128, 64), (0, half_idx)
            )
            tSgState = thr_state_t2r.partition_D(gState)
            tSgState = cute.make_tensor(tSgState.iterator.align(16), tSgState.layout)
            atom_r2g = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.state_dtype, num_bits_per_copy=128
            )
            for sub in cutlass.range(rStateG.shape[2]):
                cute.copy(atom_r2g, rStateG[None, 0, sub], tSgState[None, 0, sub])

    @cute.jit
    def compute_state_group(
        self,
        tidx: cutlass.Int32,
        tmem_ptr: cutlass.Int64,
        scale: cutlass.Float32,
        rState: cute.Tensor,
        mma_args: tuple,
        smem_args: tuple,
        checkpoint_args: tuple,
        pipeline_args: tuple,
        work_args: tuple,
        half_idx,
    ):
        """Maintain one State half and process the corresponding token-feature half."""
        sV, sCumsumlog, sCumprod, sO = smem_args
        mS_checkpoints, checkpoint_offset, checkpoint_every_n_tokens = checkpoint_args
        (
            load_v_consumer,
            load_gate_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
        ) = pipeline_args
        tiled_mma_kv, tiled_mma_qs, tiled_mma_qkv = mma_args
        (is_first_chunk, chunk_idx, head_idx) = work_args

        num_threads_state_group = self.threads_per_warp * len(
            self.compute_group_1_warp_ids
        )
        state_group_tidx = tidx % num_threads_state_group

        # -- State half (V x K view, split along K) --------------------------
        (
            tiled_state_t2r,
            tiled_state_r2t,
            tiled_state_inp_r2t,
            tTR_tCtState,
            tTR_tCcState,
            tRT_tCtState,
            tRT_tCcState_inp,
            tRT_tCtState_inp,
        ) = self._state_half_tmem_copies(
            tidx, tmem_ptr, tiled_mma_kv, tiled_mma_qs, half_idx
        )
        thr_state_t2r = tiled_state_t2r.get_slice(state_group_tidx)
        tTR_rState = rState
        tRG_rState = cute.make_rmem_tensor_like(tTR_tCcState, self.state_dtype)

        # -- Shared acc TMEM tensor (BTxDV, layout from tiled_mma_qkv) ----------
        # Needed for the G3 K*S and G5 new-V pipeline slots.
        # qkv_epilogue reads O_intra from q_state TMEM, not from this buffer.
        qkv_acc_shape = tiled_mma_qkv.partition_shape_C(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        tCtShared_fake = tiled_mma_qkv.make_fragment_C(qkv_acc_shape)
        tCtShared = cute.make_tensor(
            tmem_ptr + self.tmem_state_acc_offset,
            cute.flat_product(
                tCtShared_fake.layout, cute.make_layout((self.tmem_state_acc_stages,))
            ),
        )
        tCtShared_mn_view = self.transform_partitioned_tensor_layout(tCtShared)
        tCcShared = cute.make_identity_tensor(
            (self.mma_tiler_qkv[0], self.mma_tiler_qkv[1])
        )
        atom_shared_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(8)), self.acc_dtype
        )
        tiled_shared_t2r = tcgen05.make_tmem_copy(
            atom_shared_t2r, tCtShared[(None, None), 0, 0, 0]
        )
        thr_shared_t2r = tiled_shared_t2r.get_slice(state_group_tidx)
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
        tiled_shared_inp_r2t = tcgen05.make_tmem_copy(
            atom_shared_inp_r2t, tCtShared_inp_mn_view[None, None, 0]
        )
        thr_shared_inp_r2t = tiled_shared_inp_r2t.get_slice(state_group_tidx)
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
        tiled_qs_t2r = tcgen05.make_tmem_copy(
            atom_qs_t2r, tCtQState[(None, None), 0, 0, 0]
        )
        tiled_qs_r2t = tcgen05.make_tmem_copy(
            atom_qs_r2t, tCtQState[(None, None), 0, 0, 0]
        )
        thr_qs_t2r = tiled_qs_t2r.get_slice(state_group_tidx)
        thr_qs_r2t = tiled_qs_r2t.get_slice(state_group_tidx)
        tTR_tCtQS = thr_qs_t2r.partition_S(tCtQState_mn_view)
        tTR_tCcQS = thr_qs_t2r.partition_D(tCcQState)
        tRT_tCtQS = thr_qs_r2t.partition_D(tCtQState_mn_view)
        # -- SMEM V tiled copy: sV has domain (DV, BT) ----------------------
        tRT_tCcV = thr_shared_inp_r2t.partition_S(tCcShared_inp)
        atom_v_s2r = cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x8x16bOp(num_matrices=4, transpose=True),
            self.io_dtype,
        )
        tiled_v_s2r = cute.make_tiled_copy_S(atom_v_s2r, tiled_shared_inp_r2t)
        thr_v_s2r = tiled_v_s2r.get_slice(state_group_tidx)
        sV_vt_view = self.transform_partitioned_tensor_layout(sV)
        tCsV = thr_v_s2r.partition_S(sV_vt_view)

        # -- SMEM store: fp32 TMEM (tiled_mma_qs layout) -> fp16 sO --
        # Used by qkv_epilogue to write final O result.
        atom_o_t2r = cute.make_copy_atom(
            tcgen05.copy.Ld16x256bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        tCtQState_o = cute.local_tile(tCtQState_mn_view, (128, 32), (0, half_idx, None))
        tCcQState_o = cute.local_tile(
            cute.make_identity_tensor((128, 64)), (128, 32), (0, half_idx)
        )
        sO_half = cute.local_tile(sO, (128, 32), (0, half_idx, None))
        tiled_o_t2r = tcgen05.make_tmem_copy(atom_o_t2r, tCtQState_o[None, None, 0])
        thr_o_t2r = tiled_o_t2r.get_slice(state_group_tidx)
        tTR_tOtO = thr_o_t2r.partition_S(tCtQState_o)
        tTR_tOcO = thr_o_t2r.partition_D(tCcQState_o)
        atom_o_r2s = cute.make_copy_atom(
            cute.nvgpu.warp.StMatrix8x8x16bOp(num_matrices=2, transpose=True),
            self.io_dtype,
        )
        tiled_o_r2s = cute.make_tiled_copy_D(atom_o_r2s, tiled_o_t2r)
        thr_o_r2s = tiled_o_r2s.get_slice(state_group_tidx)
        tCsO = thr_o_r2s.partition_D(sO_half)

        token_half_size = cute.size(tTR_tCcShared, mode=[0])
        state_sub_tile_size = cute.size(tTR_rState, mode=[0])

        gate_handle = load_gate_consumer.wait_and_advance()

        cumprod_total = sCumprod[sCumprod.shape[0] - 1, 0, gate_handle.index]

        valid_state = not is_first_chunk or self.use_initial_state
        if cutlass.const_expr(valid_state):
            state_inp_ready_handle = state_inp_ready_producer.acquire_and_advance()
            tRT_rState_inp = cute.make_rmem_tensor_like(
                tRT_tCcState_inp[None, 0, 0], self.io_dtype
            )
            for sub in cutlass.range(tRT_tCcState_inp.shape[2]):
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
                        gS_checkpoints = cute.local_tile(
                            gS_checkpoints, (128, 64), (0, half_idx)
                        )
                        tSgCheckpoints = thr_state_t2r.partition_D(gS_checkpoints)
                        if cutlass.const_expr(self.state_dtype != self.acc_dtype):
                            tRG_rState[None, 0, sub].store(
                                tTR_rState[None, 0, sub].load().to(self.state_dtype)
                            )
                        else:
                            tRG_rState = tTR_rState
                        tSgCheckpoints = cute.make_tensor(
                            tSgCheckpoints.iterator.align(16),
                            tSgCheckpoints.layout,
                        )
                        atom_r2g = cute.make_copy_atom(
                            cute.nvgpu.CopyUniversalOp(),
                            self.state_dtype,
                            num_bits_per_copy=128,
                        )
                        cute.copy(
                            atom_r2g,
                            tRG_rState[None, 0, sub],
                            tSgCheckpoints[None, 0, sub],
                        )

                tRT_rState_inp.store(tTR_rState[None, 0, sub].load().to(self.io_dtype))
                cute.copy(
                    tiled_state_inp_r2t,
                    tRT_rState_inp,
                    tRT_tCtState_inp[None, 0, sub, state_inp_ready_handle.index],
                )
            cute.arch.fence_view_async_tmem_store()
            state_inp_ready_handle.commit()

            # Keep the recurrent state in registers and apply the block decay in place.
            for sub in cutlass.range(tTR_rState.shape[2]):
                for k in cutlass.range(state_sub_tile_size, vectorize=True):
                    tTR_rState[k, 0, sub] = tTR_rState[k, 0, sub] * cumprod_total
            if cutlass.const_expr(self.enable_checkpoints and not is_first_chunk):
                if (self.b_t * chunk_idx) % checkpoint_every_n_tokens == 0:
                    checkpoint_offset += 1
            cute.arch.fence_view_async_tmem_store()
        tTR_cShared_half = tTR_tCcShared[None, half_idx, 0]
        rCumprod = cute.make_rmem_tensor_like(tTR_cShared_half, self.acc_dtype)
        for k in cutlass.range(token_half_size, unroll_full=True):
            coord = tTR_cShared_half[k]
            rCumprod[k] = sCumprod[coord[1], 0, gate_handle.index]
        # ---- v - k*state  (ALU) -----------------------------------------------
        # delta[bt, dv] = V[bt, dv] - (K*S)[bt, dv]
        # Beta is fused into A_beta in CG0; no beta scaling here.
        # Write fp16 delta to sV (GEMM 5 via v_ks_ready) and sStateDv (GEMM 7 via decay_v_ready).
        vks_handle = shared_inp_ready_producer.acquire_and_advance()
        v_handle = load_v_consumer.wait_and_advance()

        if cutlass.const_expr(valid_state):
            ks_acc_handle = shared_acc_consumer.wait_and_advance()
            # State input aliases shared input. Wait for G4 to release it before
            # publishing delta for G5.
            state_inp_ready_producer.acquire()
            tTR_rKS = cute.make_rmem_tensor_like(tTR_cShared_half, self.acc_dtype)
            tRT_rV_half = cute.make_rmem_tensor_like(
                tRT_tCcV[None, half_idx, 0], self.io_dtype
            )
            # Retile expects the full logical destination. Broadcast the half-sized storage over the
            # feature-half and singleton modes, then copy only the half owned by this State group.
            tRT_rV_cv = cute.make_tensor(
                tRT_rV_half.iterator,
                cute.append(
                    cute.append(
                        tRT_rV_half.layout, cute.make_layout((2,), stride=(0,))
                    ),
                    cute.make_layout((1,), stride=(0,)),
                ),
            )
            tCrV = tiled_v_s2r.retile(tRT_rV_cv)
            if cutlass.const_expr(self.order_state_groups):
                self._state_order_wait(half_idx)
            cute.copy(
                tiled_v_s2r,
                tCsV[None, half_idx, 0, v_handle.index],
                tCrV[None, half_idx, 0],
            )
            cute.copy(
                tiled_shared_t2r,
                tTR_tCtShared[None, half_idx, 0, ks_acc_handle.index],
                tTR_rKS,
            )
            if cutlass.const_expr(self.order_state_groups):
                self._state_order_notify(half_idx)
            for k in cutlass.range(token_half_size, vectorize=True):
                tTR_rKS[k] = tTR_rKS[k] * rCumprod[k]
                tRT_rV_half[k] = tRT_rV_half[k] - tTR_rKS[k].to(self.io_dtype)
            cute.copy(
                tiled_shared_inp_r2t,
                tRT_rV_half,
                tRT_tCtShared_inp[None, half_idx, 0, vks_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()
            ks_acc_handle.release()
        vks_handle.commit()

        # ---- state*q_epi (ALU) ------------------------------------------------
        # Scale Q*S_prev cross-chunk contribution: QS[bt, *] *= cumprod[bt]
        # Write scaled result back to same q_state TMEM slot so GEMM 6 accumulates on top.
        if cutlass.const_expr(valid_state):
            qs_handle = q_state_acc_consumer.wait_and_advance()
            tTR_rQS = cute.make_rmem_tensor_like(
                tTR_tCcQS[None, half_idx, 0], self.acc_dtype
            )
            for k in cutlass.range(token_half_size, unroll_full=True):
                coord = tTR_cShared_half[k]
                rCumprod[k] = sCumprod[coord[1], 0, gate_handle.index] * scale
            cute.copy(
                tiled_qs_t2r,
                tTR_tCtQS[None, half_idx, 0, qs_handle.index],
                tTR_rQS,
            )
            for k in cutlass.range(token_half_size, vectorize=True):
                tTR_rQS[k] = tTR_rQS[k] * rCumprod[k]
            cute.copy(
                tiled_qs_r2t,
                tTR_rQS,
                tRT_tCtQS[None, half_idx, 0, qs_handle.index],
            )
            cute.arch.fence_view_async_tmem_store()

            qs_handle.release()
        rDecayScale = cute.make_rmem_tensor_like(tTR_cShared_half, self.acc_dtype)
        last_cumsumlog = sCumsumlog[self.b_t - 1, 0, gate_handle.index]
        for k in cutlass.range(token_half_size, unroll_full=True):
            coord = tTR_cShared_half[k]
            rDecayScale[k] = cute.math.exp2(
                last_cumsumlog - sCumsumlog[coord[1], 0, gate_handle.index],
                fastmath=True,
            )
        gate_handle.release()

        # ---- new_v_epi --------------------------------------------------------
        # NV = A_inv @ delta (GEMM 5 result) in shared_acc TMEM; fp32 -> fp16 -> sAinvNv.
        nv_handle = shared_acc_consumer.wait_and_advance()
        v_handle.release()
        nv_ready_handle = shared_inp_ready_producer.acquire_and_advance()

        tTR_rNv = cute.make_rmem_tensor_like(tTR_cShared_half, self.acc_dtype)
        tTR_rNv_inp = cute.make_rmem_tensor_like(tTR_rNv, self.io_dtype)
        cute.copy(
            tiled_shared_t2r,
            tTR_tCtShared[None, half_idx, 0, nv_handle.index],
            tTR_rNv,
        )
        tTR_rNv_inp.store(tTR_rNv.load().to(self.io_dtype))
        cute.copy(
            tiled_shared_inp_r2t,
            tTR_rNv_inp,
            tRT_tCtShared_inp[None, half_idx, 0, nv_ready_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        nv_handle.release()
        nv_ready_handle.commit()

        # Write decay_scale * delta to sStateDv -> GEMM 7 B-operand
        # -- kv_decay_v: S_prev *= Phi = exp(cumsumlog[BT-1]) ------------------
        # Always wait for gate (collective barrier shared with CG0).
        decay_v_handle = shared_inp_ready_producer.acquire_and_advance()
        tTR_rDv = tTR_rNv
        tRT_rDv_inp = cute.make_rmem_tensor_like(tTR_rDv, self.io_dtype)
        for k in cutlass.range(token_half_size, vectorize=True):
            tTR_rDv[k] = tTR_rDv[k] * rDecayScale[k]
        tRT_rDv_inp.store(tTR_rDv.load().to(self.io_dtype))
        cute.copy(
            tiled_shared_inp_r2t,
            tRT_rDv_inp,
            tRT_tCtShared_inp[None, half_idx, 0, decay_v_handle.index],
        )
        cute.arch.fence_view_async_tmem_store()
        decay_v_handle.commit()

        # ---- qkv_epilogue -----------------------------------------------------
        # GEMM 6 accumulated W_qkv@NV into q_state TMEM on top of the scaled Q*S.
        # q_state_acc second wait (same 1-stage pipeline, wraps back to stage 0).
        qs_handle2 = q_state_acc_consumer.wait_and_advance()
        o_handle = o_store_producer.acquire_and_advance()
        tTR_tOrO = cute.make_rmem_tensor_like(tTR_tOcO, self.acc_dtype)
        tTR_rO_out = cute.make_rmem_tensor_like(tTR_tOrO, self.io_dtype)
        tRS_tOrO = tiled_o_r2s.retile(tTR_rO_out)
        cute.copy(
            tiled_o_t2r,
            tTR_tOtO[None, None, None, qs_handle2.index],
            tTR_tOrO,
        )
        tTR_rO_out.store(tTR_tOrO.load().to(self.io_dtype))
        cute.copy(tiled_o_r2s, tRS_tOrO, tCsO[None, None, None, o_handle.index])
        cute.arch.fence_view_async_shared()
        o_handle.commit()
        qs_handle2.release()

        # ---- kv_update_epi ----------------------------------------------------
        # Add the current chunk's state increment to the register-resident state.
        kv_handle = kv_acc_consumer.wait_and_advance()
        tTR_rStateInc = cute.make_rmem_tensor_like(
            tTR_tCcState[None, 0, 0], self.acc_dtype
        )
        for sub in cutlass.range(tTR_rState.shape[2]):
            cute.copy(
                tiled_state_t2r,
                tTR_tCtState[None, 0, sub, kv_handle.index],
                tTR_rStateInc,
            )
            for k in cutlass.range(state_sub_tile_size, vectorize=True):
                tTR_rState[k, 0, sub] = tTR_rState[k, 0, sub] + tTR_rStateInc[k]
        kv_handle.release()
        return (  # type: ignore[return-value]
            load_v_consumer,
            load_gate_consumer,
            shared_acc_consumer,
            kv_acc_consumer,
            q_state_acc_consumer,
            kv_acc_producer,
            state_inp_ready_producer,
            shared_inp_ready_producer,
            o_store_producer,
            checkpoint_offset,
            tTR_rState,
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
        """Warp 15: TMA bulk-store O from SMEM staging buffer to global memory.

        Steps:
          1. Wait for both State groups to signal O is ready in sO.
          2. Domain-offset the TMA tensor to (chunk_offset, head_idx), flat-divide
             into tiles, tma_partition -> (tOsO, tOgO).
          3. Issue TMA S2G bulk copy using the per-work-tile updated descriptor.
          4. Commit the async group and wait for the store to land in GMEM.
          5. Release the pipeline slot back to both State groups.
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
                CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap
                * CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
                * num_sm
            )
        HO = HQ if HQ >= HV else HV
        return (
            CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap
            * CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
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
                    CPDeltaRulePrefillTcgen05Sm100.num_tensormaps,
                    CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                ),
                stride=(
                    CPDeltaRulePrefillTcgen05Sm100.num_tensormaps
                    * CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                    CPDeltaRulePrefillTcgen05Sm100.bytes_per_tensormap,
                    1,
                ),
            ),
        )
        return workspace
