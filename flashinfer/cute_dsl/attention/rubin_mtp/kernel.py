# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# ruff: noqa: F841, SIM201, SIM300
# Retain the pinned CuTe source expressions and staging temporaries.

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

# {$nv-internal-release file}

# ============================================================================
# Mixed-CGA MLA decode: preferred 4-CTA and fallback 2-CTA, CLC scheduling.
# Integrated from the validated mixed-CGA implementation (4afb39e5e8c).
# Keep role-local register budgets, independent query warp, and early compute
# consumer release unchanged. Debug instrumentation lives in the frozen copy.
#
# P transfer: softmax quantizes P into a LOCAL SMEM staging buffer (byte-
# identical PISL layout to the PV pair's smem_p), then ONE warp per group
# ships the whole 16KB stage with a single cp.async.bulk S2S (official
# CopyBulkS2SOp atom; SASS: UBLKCP.S.S) posting ONE transaction on the peer
# PV CTA's p_full — replacing the 1024 x 16B st.async micro-transactions of
# the STAS scheme. Producer fencing follows the memory-model guide's Optimal
# pattern (GCA "A Practical Guide for Working with the Memory Model" §5):
#   STS -> fence.proxy.async.shared::cta -> bar.sync -> elect issue.
#
# Pair-readiness: the p_pair_ready mbarrier is REMOVED. The leader's p_full
# expects tx (its own 16KB half) + ONE arrive, and that arrive comes remotely
# from the non-leader PV CTA (mapa + mbarrier.arrive release.cluster) after
# ITS half landed — the leader's single p_full wait doubles as the pair
# rendezvous. Per guide §4/§8 the leader waits with acquire.cluster.
#
# KNOWN BLOCKER: AModel does not model UBLKCP.S.S's mbarrier complete_tx
# (data delivered, tx never posted) — the repo's own
# test/python/api/sm_90a/test_bulk_copy_s2s.py hangs on AModel too. Verify on
# silicon only (run_test_mla_fp8_2p2_fixed_ublkcp.sh --no-amodel). Evidence:
# build/logs_ublkcp_evidence/EVIDENCE.txt
# ============================================================================

import math
from typing import Type, Tuple, Optional
from types import SimpleNamespace

import cuda.bindings.driver as cuda

import cutlass
import cutlass.utils as utils
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.rubin_helpers as sm107_utils
from cutlass.cute.arch import Arch
from cutlass.cutlass_dsl import BaseDSL, if_generate

# Production path: diagnostics remain in the frozen baseline bundle.


@cute.jit
def decode_work_y(y, logical_q, work_q_fdd):
    """Decode the unchanged packed CLC coordinate; divisor is prepared on host."""
    if cutlass.const_expr(isinstance(work_q_fdd, cute.FastDivmodDivisor)):
        batch_idx, query_idx = divmod(y, work_q_fdd)
    else:
        batch_idx = y // logical_q
        query_idx = y % logical_q
    return batch_idx, query_idx


from ._blackwell_helpers import (
    _page_index,
    ceil_div,
    MAX_SPLITS,
    LOG2_E,
)
from ._blackwell import (
    BlackwellMultiHeadLatentAttentionForwardFP8 as _BlackwellMLAFP8,
)
from ._rubin_helpers import (
    pack_f16x2,
    add_packed_f16x2_u32,
    reduce_sum_packed_f16x2_to_f32,
    softmax_f32x4_to_f16x2x2_and_e4m3x4,
)


@cute.jit
def _rescale_factor(previous_max, next_max, scale):
    # Both maxima can be -inf for a fully causal-masked split/query row.
    return (
        cutlass.Float32(1.0)
        if previous_max == next_max
        else cute.math.exp2((previous_max - next_max) * scale, fastmath=True)
    )


@cute.jit
def _rescale_factor_min_before_exp(previous_max, next_max, scale):
    # The PV consumer benefits from this form; softmax retains guarded rescaling.
    # Monotone maxima and positive scale make valid exponents nonpositive.
    # Both -inf produces NaN, which plain min.f32 maps to the numeric zero.
    exponent = cute.arch.fmin(
        (previous_max - next_max) * scale,
        cutlass.Float32(0.0),
        nan=False,
        ftz=False,
    )
    return cute.math.exp2(exponent, fastmath=True)


@cute.jit
def _store_empty_output(mO, mLSE, first_head, rows, query, batch, tidx, threads):
    # Only the correction owner calls this, in the workspace-free K=0 case.
    thread = tidx % threads
    for offset in range(thread, rows * 512, threads):
        mO[first_head + offset // 512, offset % 512, query, batch] = (
            cutlass.Float8E4M3FN(0.0)
        )
    if thread < rows:
        mLSE[first_head + thread, query, batch] = -cutlass.Float32.inf


@cute.jit
def _wait_cluster_mbarrier(mbar_ptr: cute.Pointer, phase: cutlass.Int32) -> None:
    """Block on a local mbarrier (plain CTA-scope parity wait).

    Feynman sm140_mla_decode pattern: the P data's visibility is already
    established by the bulk copy's complete_tx on this mbarrier (async-proxy
    mechanism, same as every PipelineTmaUmma consumer in this kernel), so the
    pair-rendezvous vote only needs the barrier COUNT — no cluster-acquire
    fence. The earlier CLUSTER/ACQUIRE variant emitted a MEMBAR.ALL.GPU per
    wait (SASS has no cluster membar scope).
    """
    cute.arch.mbarrier_wait(mbar_ptr, phase)


@cute.jit
def _arrive_cluster_mbarrier(mbar_ptr: cute.Pointer, peer_rank: cutlass.Int32) -> None:
    """Arrive a peer CTA's mbarrier (plain remote arrive, CTA scope).

    Count-only notification per the Feynman DSMEM pattern; see
    _wait_cluster_mbarrier for why no cluster-release fence is needed. The
    DSL's remote-arrive default is deliberately CTA scope ("cluster scope is
    measurably slower", cute/arch/mbar.py).
    """
    cute.arch.mbarrier_arrive(mbar_ptr, peer_cta_rank_in_cluster=peer_rank)


"""
A Multi-Head Latent Attention (MLA) 2+2-CTA example using fp8 as input/output for the NVIDIA Rubin
SM107 architecture using CUTE DSL.

New design ("2+2"): one (4,1,1) cluster processes one work item. The four CTAs are split into two
2-CTA UMMA pairs with disjoint roles:

- QK pair (cluster ranks 0/1): TMA-loads Q/K, runs the 2-CTA QK UMMA (S into TMEM), and the softmax
  warps. Softmax produces the FP8 P tile in registers and pushes it over DSMEM (st.async) directly
  into the PV pair's P SMEM buffers, together with the per-row correction metadata
  (row_sum/row_max/correction_factor/no_correction).
- PV pair (cluster ranks 2/3): TMA-loads V (both latent halves), runs the 2-CTA PV UMMA with the
  A operand (P) sourced from SMEM, and the correction/epilogue warps which rescale O in TMEM and
  write the full 512-latent output.

Compared to the previous MTP kernel (pv_n_splits=2, where two independent clusters each recomputed
the full QK for one latent half), this design computes QK exactly once and runs QK and PV on
different SMs' tensor cores in parallel, removing the bmm1-recompute mathSOL ceiling.

Cross-CTA dataflow (per k-tile, per QK CTA r in {0,1}):
- P (128x128 fp8, 16KB) : STAS from softmax registers -> PV CTA (r+2) P SMEM stage,
  transaction tracked on the same PV CTA's local p_full mbarrier (16KB local tx). Rank 2 waits for
  rank 3's release.cluster ready vote with acquire.cluster ordering before issuing the 2-CTA PV UMMA.
- corr metadata (128 rows x 16B) : STAS -> PV CTA (r+2) corr SMEM stage, tracked on that CTA's
  cor_full mbarrier.
- p_empty release: PV leader issues tcgen05.commit with a {0,1} multicast mask after the PV UMMA
  consumed the P stage. cor_empty release: correction warps remote-arrive rank r's mbarrier.

The kernel keeps the MTP kernel's split-KV support (reduction kernel unchanged).

To run this example:

.. code-block:: bash

    python examples/cute/rubin/kernel/attention/mla/mla_decode_fp8_MTP.py       \
      --batch_size 4 --latent_dim 512 --rope_dim 64                      \
      --num_heads 256 --seq_len_q 1 --seq_len_k 1024                     \
      --in_dtype Float8E4M3FN --out_dtype Float8E4M3FN                   \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --is_var_seq --is_var_split_kv                                     \
      --is_persistent

The above example runs Multi-Head Latent Attention (MLA) with the following configuration:
- Batch size: 4
- Sequence length of Q: 1
- Sequence length of K: 1024
- Latent dimension: 512
- RoPE dimension: 64
- Number of heads: 256
- Data types: Float8E4M3FN (input), Float8E4M3FN (output), Float32 (accumulation and LSE)

It utilizes page table storage for the KV cache and enables both variable-length KV cache sequences
and variable split KV processing with persistent scheduling.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/cute/rubin/kernel/attention/mla/mla_decode_fp8_MTP.py   \
      --batch_size 4 --latent_dim 512 --rope_dim 64                      \
      --num_heads 128 --seq_len_q 1 --seq_len_k 1024                     \
      --in_dtype Float8E4M3FN --out_dtype Float8E4M3FN                   \
      --acc_dtype Float32 --lse_dtype Float32                            \
      --is_var_seq --is_var_split_kv                                     \
      --is_persistent --warmup_iterations 3                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Data type requirements:
  - Input/output: Float8E4M3FN
  - Accumulation and LSE: Float32
* Fixed architecture parameters:
  - Number of attention heads: 128
  - Latent dimension: 512
  - RoPE dimension: 64
* Input query modes should be (NumHeads, LatentDim/RopeDim, SeqLenQ, BatchSize)
* Input kv latent/rope modes should be (SeqLenK, LatentDim/RopeDim, BatchSize)
* Query sequence length must be 1-4
* Only supports 2-CTA instructions
* Variable sequence length requires page table storage enabled
"""


class RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo:
    # The launcher sets these original-query coordinates before compilation.
    causal_num_heads: int
    causal_seq_len_q: int
    causal_fold_ratio: int
    _fb: "_MixedFallbackPrep"

    def __init__(
        self,
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        max_active_clusters: int,
        page_size: int,
        skip_correction_threshold: float,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        qk_acc_dtype: Optional[Type[cutlass.Numeric]] = None,
        pv_acc_dtype: Optional[Type[cutlass.Numeric]] = None,
        use_fp16_softmax: bool = False,
        force_branch: str = "auto",
    ):
        """Initializes the configuration for a Rubin Multi-Head Latent Attention (MLA) kernel.

        :param acc_dtype: Data type for accumulation S and O
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: Data type for output LSE
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_s_tiler: The (H, K) tile shape of the MMA instruction for S
        :type mma_s_tiler: Tuple[int, int]
        :param mma_p_tiler: The (H, D) tile shape of the MMA instruction for P
        :type mma_p_tiler: Tuple[int, int]
        :param max_active_clusters: Maximum number of active clusters
        :type max_active_clusters: int
        :param page_size: The page size
        :type page_size: int
        :param skip_correction_threshold: Threshold to skip correction
        :type skip_correction_threshold: float
        :param is_persistent: Whether to use persistent kernel mode
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split KV
        :type is_var_split_kv: bool
        """

        self.latent_dim = 512
        self.rope_dim = 64
        self.acc_dtype = acc_dtype
        self.qk_acc_dtype = qk_acc_dtype if qk_acc_dtype is not None else acc_dtype
        self.pv_acc_dtype = pv_acc_dtype if pv_acc_dtype is not None else acc_dtype
        self.lse_dtype = lse_dtype
        self.mma_qk_tiler_mn = mma_qk_tiler_mn
        self.mma_pv_tiler_mn = mma_pv_tiler_mn
        self.max_active_clusters = max_active_clusters
        self.skip_correction_threshold = skip_correction_threshold
        self.use_fp16_softmax = use_fp16_softmax
        assert force_branch in ("auto", "preferred", "fallback")
        # mixed-CGA branch pin: "preferred"/"fallback" compile a single
        # branch and launch plain clusters of that shape (isolated
        # testing); "auto" launches preferred+fallback_cluster and
        # branches on the runtime cluster shape.
        self.force_branch = force_branch
        # The prepared launcher attaches the paired fallback before compilation.
        self.is_persistent = is_persistent
        self.page_size = page_size
        self.is_var_seq = is_var_seq
        self.is_var_split_kv = is_var_split_kv
        # 2+2 design: one (4,1,1) cluster = QK pair (ranks 0/1) + PV pair (ranks 2/3).
        self.cluster_shape_mnk = (4, 1, 1)
        self.cta_pair_size = 2  # CTAs per 2-CTA UMMA pair
        self.use_2cta_instrs = True
        self.num_compute_warps = 4
        self.threads_per_warp = 32
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        # W9 pipelines the next CLC query with its current Q/K TMA work in
        # both cluster shapes; W12-W15 remain register donors.
        mma_qk_tiler_k = self.rope_dim * 2
        self.mma_qk_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_qk_tiler_k,
        )
        self.mma_qk_rope_tiler = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            self.rope_dim,
        )
        self.mma_pv_tiler = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            self.mma_qk_tiler[1],
        )
        self.iterations_qk_latent = self.latent_dim // self.mma_qk_tiler[2]
        self.iterations_qk_rope = 1
        self.iterations_qk = self.iterations_qk_latent + self.iterations_qk_rope
        self.iterations_pv_k = self.mma_qk_tiler[1] // self.mma_pv_tiler[2]
        # No PV N split: the PV pair computes the full latent dim (QK is never recomputed).
        self.pv_n_splits = 1
        self.latent_dim_per_group = self.latent_dim  # 512
        self.iterations_pv_n = self.latent_dim_per_group // self.mma_pv_tiler[1]

        # Auto-only N16 rescale experiment: keep four preloads but halve each
        # transfer width. Native widths and all register budgets stay unchanged.
        self.correction_subtile_n = 16 if force_branch == "auto" else 32
        self.correction_n_subtiles = self.mma_pv_tiler[1] // self.correction_subtile_n
        # Isolate repeated rescale width from final output readout width.
        self.epilogue_subtile_n = 32
        self.epilogue_n_subtiles = self.mma_pv_tiler[1] // self.epilogue_subtile_n
        self.num_correction_groups = 2

        # Warp IDs are role-dependent inside a physical four-CTA cluster:
        # warps 0-3:  QK softmax g0 / PV correction g0
        # warps 4-7:  QK softmax g1 / PV correction g1
        # warp 8:     QK MMA / PV TMEM-pointer participant
        # warp 9:     Q+K TMA / idle on PV
        # warp 10:    idle on QK / V TMA
        # warp 11:    TMEM owner on every CTA / PV MMA
        self.compute_warp_ids = (0, 1, 2, 3)
        self.second_compute_warp_ids = (4, 5, 6, 7)
        self.num_total_compute_warps = 8
        self.mma_qk_warp_id = 8
        self.load_tma_k_warp_id = 9
        self.load_tma_v_warp_id = 10
        self.mma_pv_warp_id = 11
        self.mma_warp_id = self.mma_pv_warp_id
        self.load_warp_id = self.load_tma_k_warp_id
        self.correction_warp_ids = self.compute_warp_ids
        self.second_correction_warp_ids = self.second_compute_warp_ids
        self.tmem_allocator_warp_id = self.mma_pv_warp_id
        # Only the legacy auto mega-entry needs a common 512-thread ABI.
        # A separately compiled preferred entry keeps its native 12-warps /
        # 384-thread launch so ptxas never sees the fallback warp layout.
        self.num_active_warps = 12
        self.num_dummy_warps = 4 if force_branch == "auto" else 0
        self.threads_per_cta = self.threads_per_warp * (
            self.num_active_warps + self.num_dummy_warps
        )

        # Native preferred settings. The separate preferred entry relies on
        # its 384-thread launch bound; these values remain available to its
        # role bodies but no auto-compatible setmax protocol is injected.
        self.softmax_reg_num = 192
        self.correction_reg_num = 256
        self.other_reg_num = 48
        self.dummy_reg_num = 48
        # Branch-local targets for the 512-thread mixed entry. The fallback
        # body retains its own native 160/32 protocol.
        # Keep the two preferred CTA roles on the same 208/48 protocol. Each
        # 512-thread CTA exactly consumes the 64K-register CTA budget:
        # 8 * 208 + 8 * 48 = 2048 registers per lane across its 16 warps.
        self.mixed_preferred_softmax_reg_num = 208
        self.mixed_preferred_correction_reg_num = 208
        self.mixed_preferred_other_reg_num = 48
        # Auto launches 512 threads. Runtime setmaxnreg donation redistributes
        # the CTA register pool after common initialization.
        # Keep the softmax TMEM load chunked so its score fragment does not
        # unnecessarily raise the active-warp target.
        # Ten warps retrieve the TMEM pointer on either role. QK uses g0, g1,
        # W8 and W11; PV uses g0, W8, W11 and correction.
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=1,
            num_threads=(
                self.threads_per_warp * 2
                + self.threads_per_warp * self.num_total_compute_warps
            ),
        )
        self.softmax_exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=2, num_threads=(self.threads_per_warp * self.num_compute_warps)
        )
        self.epilogue_exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=3,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        self.tmem_free_sync_bar = pipeline.NamedBarrier(
            barrier_id=4, num_threads=(self.threads_per_warp * 2)
        )
        # Pingpong order barriers for the two softmax warp groups (QK CTAs).
        # OrderedSequenceBarrier<1,2> pattern: each group arrive_and_waits on its
        # own bar; the OTHER group signals via split-phase .arrive(). num_threads
        # covers both groups (256) so the bar releases only after both arrived.
        self.softmax_order_bar_0 = pipeline.NamedBarrier(
            barrier_id=5,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        self.softmax_order_bar_1 = pipeline.NamedBarrier(
            barrier_id=6,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        self.softmax_warps_initial_sync_bar = pipeline.NamedBarrier(
            barrier_id=7,
            num_threads=(self.threads_per_warp * self.num_total_compute_warps),
        )
        # TMEM dealloc guard: W11 (allocator) must not free TMEM until all 8
        # compute warps are done with their out-of-pipeline TMEM accesses (the
        # softmax cross-group mailbox reads happen AFTER the mma_s consumer
        # release, so the pipelines alone do NOT order them against the free).
        self.tmem_dealloc_sync_bar = pipeline.NamedBarrier(
            barrier_id=8,
            num_threads=(
                self.threads_per_warp
                + self.threads_per_warp * self.num_total_compute_warps
            ),
        )
        # UBLKCP variant: per-softmax-group barrier ordering the group's 128
        # STS threads before the single-warp cp.async.bulk P send (g0 -> 9,
        # g1 -> 10). Pairs with the thread-scope proxy fence per the
        # memory-model guide's Optimal STS->bulk pattern.
        self.p_src_write_bar_0 = pipeline.NamedBarrier(
            barrier_id=9,
            num_threads=(self.threads_per_warp * self.num_compute_warps),
        )
        self.p_src_write_bar_1 = pipeline.NamedBarrier(
            barrier_id=10,
            num_threads=(self.threads_per_warp * self.num_compute_warps),
        )
        # Cross-group (row_max, row_sum) handoff seed for online softmax.
        self.init_row_max = -float("inf")

    def _setup_attributes(self):
        """Set up configurations and parameters for the MLA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the multi-head latent attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        # loopQ: the 1-stage load_q pipeline is kept for the ROPE half of Q
        # only (8KB dedicated home, produce/consume per work tile makes the
        # rope home multi-wave safe automatically). Q LATENT has no fixed
        # home: it enters the K ring as two ordinary slot productions per
        # work tile (gr150 pattern), which is what makes multi-wave (several
        # work tiles per cluster) safe by construction.
        self.load_q_stage = 1
        # K ring = 7 dedicated stages of 36,864B (latent 32,768 + rope
        # 4,096). No Q aliasing: at each work tile start W9 producer_acquires
        # the next TWO ring slots for the two Q latent halves (4 K-iterations
        # = 32,768B each, expected_tx override since Q halves carry no rope),
        # W8 consumes them via S2T and releases; the K stream then reuses the
        # slots like any others. Ring wrap (slot 6 -> 0) is safe: the halves
        # use independent per-slot views.
        self.load_k_stage = 7
        # k_armed ABA fix (Mengyu): the count-only notification ring must be
        # deeper than the K data ring. W9's lead over W10 is bounded by the 7
        # data slots plus the 2 self-produced Q-half notifications per work
        # tile (the Q fills bypass W10, so the data ring alone cannot bound
        # the lead below its own depth -> parity ABA on persistent tile
        # churn). Notification cursors are independent 9-deep states; the K
        # data/TMA barriers keep using the 7-deep cursors.
        self.k_armed_stage = self.load_k_stage + 2
        # V stage holds both latent halves (iterations_pv_n=2) => 32KB/stage/CTA.
        # Traded one V stage (was 9) for 2 extra p_full stages on the PV role
        # (net 0: -32KB V, +32KB P), keeping the role at ~333.8KB.
        self.load_v_stage = 8
        # Two S stages: one per softmax warp group (pingpong).
        self.mma_s_stage = 2
        # Cross-CTA (DSMEM) stages: P SMEM stages on the PV pair and correction
        # metadata stages. Global k-tile t maps to P stage t % p_stage (the
        # group advance logic tracks absolute tile positions, so any even
        # p_stage works): with 4 stages g0 owns {0,2} and g1 owns {1,3},
        # giving each softmax group double buffering of its own sends —
        # funded by dropping one K stage and one V stage.
        self.p_stage = 4
        # Correction metadata pipeline: 4 stages at zero SMEM growth (the
        # 2-word protocol halves the per-stage footprint; 4 x 1,024B = the old
        # 2 x 2,048B).
        self.p_cor_stage = 4
        self.mma_o_stage = self.iterations_pv_n

        # Role-dependent TMEM layouts (offsets of the two roles may overlap, they
        # live in different CTAs):
        #   QK CTA: | S (128 f32 cols) | Q latent (128 fp8-packed cols) | Q rope (16) |
        #   PV CTA: | ---- O (512 f32 cols, both latent halves) ---- |
        self.tmem_s_offset = 0
        self.tmem_s_stage_cols = self.mma_qk_tiler[1]  # 128 FP32 cols per S stage
        # Q-in-TMEM: all latent iterations and Q rope live in TMEM (QK CTAs).
        self.q_tmem_iters = self.iterations_qk_latent
        self.tmem_q_cols = self.q_tmem_iters * self.mma_qk_tiler[2] // 4
        self.tmem_q_offset = self.mma_s_stage * self.tmem_s_stage_cols
        self.tmem_q_rope_offset = self.tmem_q_offset + self.tmem_q_cols
        self.tmem_q_rope_cols = self.rope_dim // 4
        # Softmax cross-group (row_sum, row_max) TMEM mailbox (baseline-style
        # STTM/LDTM). 448 is above the Q-rope S2T's real write footprint (AModel
        # write-watch shows fp8-packed data through col ~440) while keeping all
        # four 4-column mailbox stages inside GR100's 512 allocatable columns.
        # The mailbox is read OUTSIDE any pipeline, so
        # the allocator warp gates its tmem.free on tmem_dealloc_sync_bar —
        # without that guard the dealloc races the last cross-group exchange
        # and the LDTM returns cleared TMEM.
        self.tmem_group_mbox_stage_cols = 4
        self.tmem_group_mbox_offset = 448
        # PV CTA: O occupies the full 512 columns.
        self.tmem_o_offset = 0
        # Correction metadata words per row: (row_sum, row_max). The PV side
        # recomputes correction_factor and no_correction from consecutive
        # row_max values (the sender's rollback keeps row_max unchanged on
        # no-correction tiles, so the receiver's recompute is exact).
        self.corr_words_per_row = 2

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [num_head, rope_dim, seq_len_q, batch_size]
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [seq_len_k, latent_dim, batch_size]
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [seq_len_k, rope_dim, batch_size]
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [page_count, batch_size]
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, seq_len_q, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )
        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1 or q_rope.stride[1] != 1):
            raise ValueError("q_latent and q_rope must have leading dimension 1")
        if cutlass.const_expr(c_latent.stride[1] != 1 or c_rope.stride[1] != 1):
            raise ValueError("c_latent and c_rope must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")

        acc_o, acc_lse = self.initialize_workspace(
            q_latent.shape[0],
            q_latent.shape[1],
            q_latent.shape[2],
            q_latent.shape[3],
            split_kv,
            self.acc_dtype,
            workspace,
        )

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        mma_inst_k = 64
        qk_mma_inst_shape = (
            self.mma_qk_tiler_mn[0],
            self.mma_qk_tiler_mn[1],
            mma_inst_k,
        )
        qk_tiled_mma = sm107_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            qk_mma_inst_shape,
        )
        # QK MMA with A from TMEM (for Q-in-TMEM S2T copy layout)
        qk_tiled_mma_tmem = sm107_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.qk_acc_dtype,
            cta_group,
            qk_mma_inst_shape,
            a_source=tcgen05.OperandSource.TMEM,
        )

        pv_mma_inst_shape = (
            self.mma_pv_tiler_mn[0],
            self.mma_pv_tiler_mn[1],
            mma_inst_k,
        )
        # 2+2: the PV pair's A operand (P) is sourced from SMEM, filled over
        # DSMEM by the QK pair's softmax warps.
        pv_tiled_mma = sm107_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.pv_acc_dtype,
            cta_group,
            pv_mma_inst_shape,
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )
        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )
        q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_rope_tiler,
            self.q_dtype,
            self.load_q_stage,
        )
        # loopQ: half-Q latent layout (4 of the 8 K-iterations, 32,768B) —
        # byte-identical to either half of the full staged Q layout (the
        # iteration mode is compact, 8KB per iteration), sized to exactly one
        # K latent ring slot.
        q_latent_half_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent // 2),
        )
        q_latent_half_smem_layout_staged = cute.logical_divide(
            q_latent_half_smem_layout_staged,
            (None, None, None, self.iterations_qk_latent // 2),
        )

        # Q-in-TMEM layout (using TMEM MMA, all latent iterations)
        q_tmem_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.q_tmem_iters * self.load_q_stage),
        )
        q_tmem_smem_layout = cute.logical_divide(
            q_tmem_smem_layout, (None, None, None, self.q_tmem_iters)
        )
        q_latent_tmem_layout = q_tmem_smem_layout.outer
        # loopQ: TMEM-side half layout for the per-half S2T partitions.
        q_tmem_half_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.q_tmem_iters // 2),
        )
        q_tmem_half_smem_layout = cute.logical_divide(
            q_tmem_half_smem_layout, (None, None, None, self.q_tmem_iters // 2)
        )
        q_latent_half_tmem_layout = q_tmem_half_smem_layout.outer

        q_rope_tmem_smem_layout = sm100_utils.make_smem_layout_a(
            qk_tiled_mma_tmem,
            self.mma_qk_rope_tiler,
            self.q_dtype,
            self.load_q_stage,
        )
        q_rope_tmem_layout = q_rope_tmem_smem_layout.outer

        kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )

        kc_page_tile_size = min(
            self.page_size, qk_tiled_mma.op.shape_mnk[1] // qk_tiled_mma.thr_id.shape
        )
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[1] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )

        kc_latent_smem_layout_for_tma = cute.tiled_divide(
            kc_latent_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_tiler[2])
        )
        kc_latent_smem_layout_for_tma = cute.logical_divide(
            kc_latent_smem_layout_for_tma, (None, None, None, self.iterations_qk_latent)
        )

        kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.k_dtype,
            self.load_k_stage,
        )
        kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (
                self.mma_qk_rope_tiler[1] // qk_tiled_mma.thr_id.shape,
                self.mma_qk_rope_tiler[2],
            ),
            self.k_dtype,
            (self.iterations_qk_rope * self.load_k_stage),
        )
        kc_rope_smem_layout_for_tma = cute.tiled_divide(
            kc_rope_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_rope_tiler[2])
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_staged = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_staged,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        vc_page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_for_tma = cute.tiled_divide(
            vc_smem_layout_for_tma,
            (
                pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
                vc_page_tile_size,
            ),
        )
        vc_smem_layout_for_tma = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )

        tStS_shape = qk_tiled_mma.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        # O accumulator shape in tmem (partition_shape_C of PV MMA)
        tOtO_shape = pv_tiled_mma.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )

        # TMA load for Q latent and rope
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma_tmem,
            cta_layout_vmnk.shape,
        )
        q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_rope,
            q_rope_smem_layout,
            self.mma_qk_rope_tiler,
            qk_tiled_mma_tmem,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent and k rope.
        # tmaKV: K is issued by the PV CTAs as a single-target TMA MULTICAST
        # to the paired QK CTA (mask = 1 << pair_rank, issuer NOT in the
        # mask). cta_group::2 keeps the tx aggregation on the receiving
        # pair's leader mbar, so the QK-side PipelineTmaUmma accounting is
        # unchanged. The issuer must stay out of the mask: under the
        # QK/PV union SMEM layout, K landing in PV smem would fall inside
        # the live V ring.
        tma_mcast_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp(
            cta_group
        )
        kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent, tma_tensor_c_latent = self.make_paged_tiled_tma_atom(
            tma_mcast_load_op,
            c_latent,
            kc_smem_layout,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )

        kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
        tma_atom_c_rope, tma_tensor_c_rope = self.make_paged_tiled_tma_atom(
            tma_mcast_load_op,
            c_rope,
            kc_rope_smem_layout,
            (self.mma_qk_rope_tiler[1], self.mma_qk_rope_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )

        # TMA load for c latent transpose
        vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_transpose,
                vc_smem_layout,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        q_rope_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_rope_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        kc_latent_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        kc_rope_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        vc_copy_size = (
            cute.size_in_bytes(
                self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
            )
            * cute.size(pv_tiled_mma.thr_id.shape)
            * self.iterations_pv_n
            * self.iterations_pv_k
        )

        self.tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size
        # loopQ byte counts. Pair-aggregated (x thr_id.shape) like every
        # other pipeline tx: the load_q pipeline now carries ROPE only; each
        # Q latent half arms one K ring slot with an expected_tx override.
        self.tma_copy_q_rope_bytes = q_rope_copy_size
        self.tma_copy_q_half_bytes = q_latent_copy_size // 2
        # Per-CTA byte stride of one K latent ring slot (pointer math for the
        # dynamic half views).
        self.q_ring_slot_bytes = kc_latent_copy_size // cute.size(
            qk_tiled_mma.thr_id.shape
        )
        self.tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size
        self.tma_copy_vc_bytes = vc_copy_size

        # DSMEM transaction byte counts. An st.async DSMEM store and its
        # completion mbarrier must target the same CTA, so each PV CTA owns the
        # p_full transaction count for the half of P sent by its QK peer. A
        # separate two-arrival barrier below rendezvouses both PV CTAs before
        # the leader issues the 2-CTA PV UMMA.
        p_bytes_per_cta = (
            cute.size_in_bytes(
                self.q_dtype, cute.select(p_smem_layout_staged, mode=[0, 1, 2])
            )
            * self.iterations_pv_k
        )
        self.p_tx_bytes_per_cta = p_bytes_per_cta
        # corr metadata: per-CTA 128 rows x 4 words, tracked on each PV CTA's own
        # cor_full mbarrier.
        self.corr_rows = self.mma_qk_tiler[0] // self.cta_pair_size  # 128
        self.cor_tx_bytes = self.corr_rows * self.corr_words_per_row * 4

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
            self.pv_n_splits,
        )

        # Two role-specific storage structs sharing an IDENTICAL mbarrier header
        # (same field sequence => same offsets). Every CTA constructs both views
        # on the same SMEM base ("union"); the header is initialized once via the
        # QK view, and cross-CTA DSMEM references resolve because offsets match
        # across all CTAs in the cluster.
        corr_smem_words = self.corr_rows * self.corr_words_per_row * self.p_cor_stage

        @cute.struct
        class SplitKVKernelSharedStorageQK:
            # ---- shared mbarrier header (must match PV struct exactly) ----
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4],
                16,
            ]
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_v_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]
            p_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_stage]
            p_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_stage]
            cor_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage]
            cor_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage]
            k_armed_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_armed_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # ---- QK-role data ----
            # UBLKCP variant: local P staging, placed at the FRONT of the
            # role data (offset 1024, right after the shared mbar header).
            # Softmax writes P here (same PISL-swizzled layout as the PV
            # pair's smem_p), then one warp bulk-copies the 16KB stage to the
            # peer PV CTA in a single cp.async.bulk transaction.
            smem_p_src: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            # K ring stage aliasing — field ORDER is load-bearing: the
            # K ring layouts span contiguously from smem_kc_latent into
            # loopQ: no Q aliasing — the kc fields hold the FULL ring and
            # Q latent transits through ring slots. Only the 8KB rope home
            # lays the full ring over the base pointer.
            smem_kc_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_kc_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_rope_smem_layout_staged)
                ],
                1024,
            ]
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

        @cute.struct
        class SplitKVKernelSharedStoragePV:
            # ---- shared mbarrier header (must match QK struct exactly) ----
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4],
                16,
            ]
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_v_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]
            p_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_stage]
            p_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_stage]
            cor_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage]
            cor_empty_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage]
            k_armed_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.k_armed_stage]
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            # ---- PV-role data ----
            # P receive buffer placed at the FRONT of the PV role data
            # (offset 1KB, inside the first 128KB window); V ring follows.
            # Functionally neutral: both sides address P through this field's
            # union offset (sender mapa's storage_pv.smem_p.data_ptr()).
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_vc: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(vc_smem_layout_staged)],
                1024,
            ]
            smem_corr: cute.struct.Align[
                cute.struct.MemRange[self.acc_dtype, corr_smem_words],
                16,
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_total_compute_warps * self.threads_per_warp
            ]

        smem_size_bytes = max(
            SplitKVKernelSharedStorageQK.size_in_bytes(),  # type: ignore[attr-defined]  # cute.struct supplies this method
            SplitKVKernelSharedStoragePV.size_in_bytes(),  # type: ignore[attr-defined]  # cute.struct supplies this method
        )

        softmax_scale_log2 = softmax_scale * LOG2_E

        # mixed-CGA: build the fallback (1xfp8 baseline) argument core on
        # its own (2,1,1)-cluster atoms/layouts; work walk and acc views
        # are shared with the preferred side.
        _fb_core, _fb_shared_storage = self._fb.prepare_args(
            q_latent,
            q_rope,
            c_latent,
            c_rope,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale,
            output_scale,
            lse_scale,
        )
        _mixed_smem_bytes = max(smem_size_bytes, _fb_shared_storage.size_in_bytes())
        # Auto uses the fallback body's 512-thread launch ABI; W12-W15 are
        # register donors in the preferred body. Forced preferred remains at
        # its native 384 threads because const_expr removes the fallback body.
        if cutlass.const_expr(self.force_branch == "fallback"):
            _mixed_cluster = (self.cta_pair_size, 1, 1)
            _mixed_fallback = None
            _mixed_block = self._fb.threads_per_cta
        elif cutlass.const_expr(self.force_branch == "preferred"):
            _mixed_cluster = self.cluster_shape_mnk
            _mixed_fallback = None
            _mixed_block = self.threads_per_cta
        else:
            _mixed_cluster = self.cluster_shape_mnk
            _mixed_fallback = (self.cta_pair_size, 1, 1)
            _mixed_block = max(self.threads_per_cta, self._fb.threads_per_cta)
        self.mixed_split_kv_kernel(
            qk_tiled_mma,
            pv_tiled_mma,
            qk_tiled_mma_tmem,
            q_latent_tmem_layout,
            q_latent_half_tmem_layout,
            q_rope_tmem_layout,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_q_rope,
            tma_tensor_q_rope,
            tma_atom_c_latent,
            tma_tensor_c_latent,
            tma_atom_c_rope,
            tma_tensor_c_rope,
            tma_atom_c_latent_transpose,
            tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lse_scale,
            q_latent_smem_layout_staged,
            q_latent_half_smem_layout_staged,
            q_rope_smem_layout_staged,
            kc_latent_smem_layout_staged,
            kc_rope_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_latent_smem_layout_for_tma,
            kc_rope_smem_layout_for_tma,
            vc_smem_layout_for_tma,
            cta_layout_vmnk,
            tile_sched_params,
            SplitKVKernelSharedStorageQK,
            SplitKVKernelSharedStoragePV,
            smem_size_bytes,
            *_fb_core,
            tile_sched_params,
            _fb_shared_storage,
            cute.fast_divmod_create_divisor(cute.size(o.shape[2]))
            if cutlass.const_expr(self.force_branch == "auto")
            else 0,
        ).launch(
            grid=grid,
            block=[_mixed_block, 1, 1],
            cluster=_mixed_cluster,
            fallback_cluster=_mixed_fallback,
            smem=_mixed_smem_bytes,
            smem_merge_branch_allocs=True,
            stream=stream,
            min_blocks_per_mp=1,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
                lse_scale,
            ).launch(
                grid=(q_latent.shape[0], q_latent.shape[2], q_latent.shape[3]),
                block=[self.threads_per_warp * self.num_compute_warps, 1, 1],
                smem=MAX_SPLITS * self.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.kernel
    def mixed_split_kv_kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_qk_tmem: cute.TiledMma,
        q_latent_tmem_layout: cute.Layout,
        q_latent_half_tmem_layout: cute.Layout,
        q_rope_tmem_layout: cute.Layout,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_latent_half_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: Optional[cute.ComposedLayout],
        kc_rope_smem_layout_for_tma: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma: Optional[cute.ComposedLayout],
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        SharedStorageQK: cutlass.Constexpr,
        SharedStoragePV: cutlass.Constexpr,
        smem_size_bytes: cutlass.Constexpr,
        fb_tiled_mma_qk: cute.TiledMma,
        fb_tiled_mma_pv: cute.TiledMma,
        fb_tma_atom_q_latent: Optional[cute.CopyAtom],
        fb_mQL: cute.Tensor,
        fb_tma_atom_q_rope: Optional[cute.CopyAtom],
        fb_mQR: cute.Tensor,
        fb_tma_atom_c_latent: Optional[cute.CopyAtom],
        fb_mCL: cute.Tensor,
        fb_tma_atom_c_rope: Optional[cute.CopyAtom],
        fb_mKR: cute.Tensor,
        fb_tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        fb_mCLT: cute.Tensor,
        fb_mPT: cute.Tensor,
        fb_mO: Optional[cute.Tensor],
        fb_mLSE: Optional[cute.Tensor],
        fb_mAccO: Optional[cute.Tensor],
        fb_mAccLSE: Optional[cute.Tensor],
        fb_split_kv: cutlass.Int32,
        fb_cache_seqs: cute.Tensor,
        fb_block_split_kvs: cute.Tensor,
        fb_softmax_scale_log2: cutlass.Float32,
        fb_output_scale: cutlass.Float32,
        fb_lse_scale: cutlass.Float32,
        fb_q_latent_smem_layout_staged: cute.ComposedLayout,
        fb_q_rope_smem_layout_staged: cute.ComposedLayout,
        fb_kc_latent_smem_layout_staged: cute.ComposedLayout,
        fb_kc_rope_smem_layout_staged: cute.ComposedLayout,
        fb_p_smem_layout_staged: cute.ComposedLayout,
        fb_vc_smem_layout_staged: cute.ComposedLayout,
        fb_kc_latent_smem_layout_for_tma: Optional[cute.ComposedLayout],
        fb_kc_rope_smem_layout_for_tma: Optional[cute.ComposedLayout],
        fb_vc_smem_layout_for_tma: Optional[cute.ComposedLayout],
        fb_cta_layout_vmnk: cute.Layout,
        fb_tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        fb_SharedStorage: cutlass.Constexpr,
        work_q_fdd,
    ):
        """Mixed-CGA mega kernel.

        Preferred (4,1,1) clusters run the 2+2 loopQ body; fallback (2,1,1)
        clusters run the 1xfp8 baseline body, pair p covering the s = p row
        block of its unit. The branch pins to ``force_branch`` when set, else
        follows the runtime cluster shape (flexible-CGA launch). Both bodies
        allocate their SMEM from offset 0 — the launch merges the exclusive
        branches (``smem_merge_branch_allocs``).
        """
        if cutlass.const_expr(self.force_branch == "preferred"):
            self.preferred_kernel_body(
                tiled_mma_qk,
                tiled_mma_pv,
                tiled_mma_qk_tmem,
                q_latent_tmem_layout,
                q_latent_half_tmem_layout,
                q_rope_tmem_layout,
                tma_atom_q_latent,
                mQL,
                tma_atom_q_rope,
                mQR,
                tma_atom_c_latent,
                mCL,
                tma_atom_c_rope,
                mKR,
                tma_atom_c_latent_transpose,
                mCLT,
                mPT,
                mO,
                mLSE,
                mAccO,
                mAccLSE,
                split_kv,
                cache_seqs,
                block_split_kvs,
                softmax_scale_log2,
                output_scale,
                lse_scale,
                q_latent_smem_layout_staged,
                q_latent_half_smem_layout_staged,
                q_rope_smem_layout_staged,
                kc_latent_smem_layout_staged,
                kc_rope_smem_layout_staged,
                p_smem_layout_staged,
                vc_smem_layout_staged,
                kc_latent_smem_layout_for_tma,
                kc_rope_smem_layout_for_tma,
                vc_smem_layout_for_tma,
                cta_layout_vmnk,
                tile_sched_params,
                SharedStorageQK,
                SharedStoragePV,
                smem_size_bytes,
                work_q_fdd,
            )
        elif cutlass.const_expr(self.force_branch == "fallback"):
            self._fb.fallback_kernel_body(
                fb_tiled_mma_qk,
                fb_tiled_mma_pv,
                fb_tma_atom_q_latent,
                fb_mQL,
                fb_tma_atom_q_rope,
                fb_mQR,
                fb_tma_atom_c_latent,
                fb_mCL,
                fb_tma_atom_c_rope,
                fb_mKR,
                fb_tma_atom_c_latent_transpose,
                fb_mCLT,
                fb_mPT,
                fb_mO,
                fb_mLSE,
                fb_mAccO,
                fb_mAccLSE,
                fb_split_kv,
                fb_cache_seqs,
                fb_block_split_kvs,
                fb_softmax_scale_log2,
                fb_output_scale,
                fb_lse_scale,
                fb_q_latent_smem_layout_staged,
                fb_q_rope_smem_layout_staged,
                fb_kc_latent_smem_layout_staged,
                fb_kc_rope_smem_layout_staged,
                fb_p_smem_layout_staged,
                fb_vc_smem_layout_staged,
                fb_kc_latent_smem_layout_for_tma,
                fb_kc_rope_smem_layout_for_tma,
                fb_vc_smem_layout_for_tma,
                fb_cta_layout_vmnk,
                fb_tile_sched_params,
                fb_SharedStorage,
                work_q_fdd,
            )
        else:
            _cb_x, _, _ = cute.arch.block_in_cluster_dim()
            if _cb_x == self.cluster_shape_mnk[0]:
                self.preferred_kernel_body(
                    tiled_mma_qk,
                    tiled_mma_pv,
                    tiled_mma_qk_tmem,
                    q_latent_tmem_layout,
                    q_latent_half_tmem_layout,
                    q_rope_tmem_layout,
                    tma_atom_q_latent,
                    mQL,
                    tma_atom_q_rope,
                    mQR,
                    tma_atom_c_latent,
                    mCL,
                    tma_atom_c_rope,
                    mKR,
                    tma_atom_c_latent_transpose,
                    mCLT,
                    mPT,
                    mO,
                    mLSE,
                    mAccO,
                    mAccLSE,
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    softmax_scale_log2,
                    output_scale,
                    lse_scale,
                    q_latent_smem_layout_staged,
                    q_latent_half_smem_layout_staged,
                    q_rope_smem_layout_staged,
                    kc_latent_smem_layout_staged,
                    kc_rope_smem_layout_staged,
                    p_smem_layout_staged,
                    vc_smem_layout_staged,
                    kc_latent_smem_layout_for_tma,
                    kc_rope_smem_layout_for_tma,
                    vc_smem_layout_for_tma,
                    cta_layout_vmnk,
                    tile_sched_params,
                    SharedStorageQK,
                    SharedStoragePV,
                    smem_size_bytes,
                    work_q_fdd,
                )
            else:
                self._fb.fallback_kernel_body(
                    fb_tiled_mma_qk,
                    fb_tiled_mma_pv,
                    fb_tma_atom_q_latent,
                    fb_mQL,
                    fb_tma_atom_q_rope,
                    fb_mQR,
                    fb_tma_atom_c_latent,
                    fb_mCL,
                    fb_tma_atom_c_rope,
                    fb_mKR,
                    fb_tma_atom_c_latent_transpose,
                    fb_mCLT,
                    fb_mPT,
                    fb_mO,
                    fb_mLSE,
                    fb_mAccO,
                    fb_mAccLSE,
                    fb_split_kv,
                    fb_cache_seqs,
                    fb_block_split_kvs,
                    fb_softmax_scale_log2,
                    fb_output_scale,
                    fb_lse_scale,
                    fb_q_latent_smem_layout_staged,
                    fb_q_rope_smem_layout_staged,
                    fb_kc_latent_smem_layout_staged,
                    fb_kc_rope_smem_layout_staged,
                    fb_p_smem_layout_staged,
                    fb_vc_smem_layout_staged,
                    fb_kc_latent_smem_layout_for_tma,
                    fb_kc_rope_smem_layout_for_tma,
                    fb_vc_smem_layout_for_tma,
                    fb_cta_layout_vmnk,
                    fb_tile_sched_params,
                    fb_SharedStorage,
                    work_q_fdd,
                )

    @cute.jit
    def make_paged_tiled_tma_atom(
        self,
        tma_load_op: cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp,
        gmem: cute.Tensor,
        smem_layout: cute.Layout,
        mma_tiler,
        tiled_mma: cute.TiledMma,
        is_k_load: bool,
    ):
        ident = cute.make_identity_layout(gmem.shape)
        g_tile = cute.composition(ident, mma_tiler)
        cta_mn = mma_tiler[0] // tiled_mma.thr_id.shape
        cta_v_map = cute.flat_divide(g_tile, (cta_mn,))
        cta_v_map = cute.select(cta_v_map, mode=[0, 2])
        page_tile_size = (
            min(self.page_size, cta_mn)
            if is_k_load
            else min(self.page_size, mma_tiler[1])
        )
        cta_v_map = cute.zipped_divide(
            cta_v_map,
            (page_tile_size, mma_tiler[1]) if is_k_load else (cta_mn, page_tile_size),
        )
        cta_v_map = cute.select(cta_v_map, mode=[0])
        from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir

        res = _cute_nvgpu_ir.atom_make_non_exec_tiled_tma_load(
            gmem.value,
            smem_layout.value,
            cta_v_map,
            tma_load_op._to_ir(),
            num_multicast=1,
        )

        trait_cls = (
            cpasync.CopyBulkTensorTileG2SMulticastNonExecTrait
            if isinstance(tma_load_op, cpasync.CopyBulkTensorTileG2SMulticastOp)
            else cpasync.CopyBulkTensorTileG2SNonExecTrait
        )
        return cute.CopyAtom(tma_load_op, trait_cls(res[0])), res[1]

    @cute.jit
    def preferred_kernel_body(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tiled_mma_qk_tmem: cute.TiledMma,
        q_latent_tmem_layout: cute.Layout,
        q_latent_half_tmem_layout: cute.Layout,
        q_rope_tmem_layout: cute.Layout,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_latent_half_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: Optional[cute.ComposedLayout],
        kc_rope_smem_layout_for_tma: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma: Optional[cute.ComposedLayout],
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        SharedStorageQK: cutlass.Constexpr,
        SharedStoragePV: cutlass.Constexpr,
        smem_size_bytes: cutlass.Constexpr,
        work_q_fdd,
    ):
        """The device split_kv kernel implementation of the Multi-Head Latent Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the MLA computation:
        1. Load warp: Loads Q/C latent/rope data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the MLA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param tma_atom_q_rope: TMA copy atom for query rope tensor
        :type tma_atom_q_rope: cute.CopyAtom
        :param mKR: Compressed rope tensor
        :type mKR: cute.Tensor
        :param tma_atom_c_latent: TMA copy atom for c latent tensor
        :type tma_atom_c_latent: cute.CopyAtom
        :param mCL: Compressed latent tensor
        :type mCL: cute.Tensor
        :param tma_atom_c_rope: TMA copy atom for c rope tensor
        :type tma_atom_c_rope: cute.CopyAtom
        :param mCLT: Compressed latent transpose tensor
        :type mCLT: cute.Tensor
        :param mPT: Page table tensor
        :type mPT: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The variable sequence length tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param q_latent_smem_layout_staged: Shared memory layout for query tensor
        :type q_latent_smem_layout_staged: cute.ComposedLayout
        :param q_rope_smem_layout_staged: Shared memory layout for query rope tensor
        :type q_rope_smem_layout_staged: cute.ComposedLayout
        :param kc_latent_smem_layout_staged: Shared memory layout for key tensor
        :type kc_latent_smem_layout_staged: cute.ComposedLayout
        :param kc_rope_smem_layout_staged: Shared memory layout for key rope tensor
        :type kc_rope_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        # DSMEM peers: QK CTA r sends P/corr data to PV CTA r+2, tracking each
        # transaction on that same destination CTA. The PV pair rendezvouses
        # on rank 2, and PV CTA r+2 releases P back to QK CTA r.
        pv_pair_leader_rank = self.cta_pair_size
        qk_pair_commit_mask = 0b0011  # cluster ranks 0 and 1

        # Prefetch only descriptors owned by the physical CTA role. Compute the
        # cluster rank inside each specialist branch so the role value is not
        # kept live across the entire kernel.
        if warp_idx == self.mma_qk_warp_id:
            prefetch_bidx, _, _ = cute.arch.block_idx()
            prefetch_qk_cta_rank = cute.arch.make_warp_uniform(
                prefetch_bidx % self.cluster_shape_mnk[0]
            )
            if prefetch_qk_cta_rank < self.cta_pair_size:
                cpasync.prefetch_descriptor(tma_atom_q_latent)
                cpasync.prefetch_descriptor(tma_atom_q_rope)
        if warp_idx == self.load_tma_v_warp_id:
            prefetch_bidx, _, _ = cute.arch.block_idx()
            prefetch_v_cta_rank = cute.arch.make_warp_uniform(
                prefetch_bidx % self.cluster_shape_mnk[0]
            )
            if prefetch_v_cta_rank >= self.cta_pair_size:
                cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)
                # tmaKV: PV issues the K multicast, prefetch K descriptors here
                cpasync.prefetch_descriptor(tma_atom_c_latent)
                cpasync.prefetch_descriptor(tma_atom_c_rope)

        # Alloc: both role structs are views ("union") over the same SMEM base.
        # Their mbarrier headers have identical offsets by construction; barrier
        # init below goes through the QK view only.
        smem = cutlass.memory.SmemAllocator()
        smem_base_ptr = smem.allocate(smem_size_bytes, 1024)
        storage = SharedStorageQK(smem_base_ptr)
        storage_pv = SharedStoragePV(smem_base_ptr)

        # Forty data-role warps consume responses. Auto adds CTA0/W12 as the
        # independent scheduler consumer; native-only modes retain CTA0/W9.
        preferred_clc_consumer_warps = 41 if self.force_branch == "auto" else 40
        shared_clc_pipeline = self.make_and_init_clc_pipeline(
            storage.clc_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            preferred_clc_consumer_warps * self.threads_per_warp,
        )
        clc_response_ptr = storage.clc_response.data_ptr()

        # Tensor memory dealloc barrier init. W11 allocates and frees TMEM on
        # every CTA; W8 hands QK completion to W11 through barrier 4.
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.tmem_allocator_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch="sm_107",
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            # loopQ: rope only — Q latent rides the K ring.
            self.tma_copy_q_rope_bytes,
        )
        load_k_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_k_stage,
            self.tma_copy_kc_bytes,
        )
        load_v_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_v_stage,
            self.tma_copy_vc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )

        # Cross-CTA (DSMEM) S2S mbarriers for P and correction metadata.
        # full barriers: armed with expect_tx by the consumer side; STAS
        # transactions complete them (plus one elected consumer arrive).
        # empty barriers: producer waits; consumer releases remotely
        # (tcgen05.commit multicast for P, per-thread remote arrives for corr).
        one_thread_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        corr_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            self.num_compute_warps * self.num_correction_groups * self.threads_per_warp,
        )
        p_full_mbar = pipeline.MbarrierArray(
            barrier_storage=storage.p_full_mbar_ptr.data_ptr().align(min_align=8),
            num_stages=self.p_stage,
            agent=(pipeline.PipelineOp.AsyncThread, one_thread_group),
            tx_count=self.p_tx_bytes_per_cta,
        )
        p_empty_mbar = pipeline.MbarrierArray(
            barrier_storage=storage.p_empty_mbar_ptr.data_ptr().align(min_align=8),
            num_stages=self.p_stage,
            agent=(pipeline.PipelineOp.AsyncThread, one_thread_group),
        )
        # tmaKV: per-stage "QK stage armed" signal. Each QK CTA's W9 remote-
        # arrives its paired PV CTA's k_armed[stage] after producer_acquire
        # (own empty seen + leader expect_tx armed); the PV W10 waits it
        # before issuing the K TMA multicast into that stage. One arrive per
        # stage slot.
        k_armed_mbar = pipeline.MbarrierArray(
            barrier_storage=storage.k_armed_mbar_ptr.data_ptr().align(min_align=8),
            num_stages=self.k_armed_stage,
            agent=(pipeline.PipelineOp.AsyncThread, one_thread_group),
        )
        # UBLKCP variant: no p_pair_ready mbarrier. Pair-readiness is folded
        # into the leader's p_full arrive count (the non-leader votes on the
        # leader's p_full after its own local P half has landed).
        cor_full_mbar = pipeline.MbarrierArray(
            barrier_storage=storage.cor_full_mbar_ptr.data_ptr().align(min_align=8),
            num_stages=self.p_cor_stage,
            agent=(pipeline.PipelineOp.AsyncThread, one_thread_group),
            tx_count=self.cor_tx_bytes,
        )
        cor_empty_mbar = pipeline.MbarrierArray(
            barrier_storage=storage.cor_empty_mbar_ptr.data_ptr().align(min_align=8),
            num_stages=self.p_cor_stage,
            agent=(pipeline.PipelineOp.AsyncThread, corr_consumer_group),
        )
        # Initial arming of the receive-side transaction counts (harmless on the
        # CTAs whose full barriers are never targeted).
        if warp_idx == 0:
            with cute.arch.elect_one():
                for _stage in cutlass.range_constexpr(self.p_stage):
                    cute.arch.mbarrier_expect_tx(
                        p_full_mbar.get_barrier(_stage), self.p_tx_bytes_per_cta
                    )
                for _stage in cutlass.range_constexpr(self.p_cor_stage):
                    cute.arch.mbarrier_expect_tx(
                        cor_full_mbar.get_barrier(_stage), self.cor_tx_bytes
                    )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        # Generate smem tensors. QK-role views:
        # (MMA, MMA_H, MMA_R, PIPE)
        # loopQ: no fixed Q latent view; per-tile half views are built
        # over the K ring slots (see load_tma_qk / mma_qk_warp_body).
        kc_latent_base_ptr = storage.smem_kc_latent.data_ptr()
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer, swizzle=kc_rope_smem_layout_staged.inner
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer, swizzle=kc_rope_smem_layout_for_tma.inner
        )
        # PV-role views:
        # (MMA, MMA_D, MMA_K, PIPE)
        sVC = storage_pv.smem_vc.get_tensor(
            vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
        )
        sVC_for_tma = storage_pv.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer, swizzle=vc_smem_layout_for_tma.inner
        )
        # (MMA, MMA_H, MMA_K, (PV_K, PIPE))
        sP = storage_pv.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # UBLKCP variant: local P staging view on the QK role (byte-identical
        # PISL layout to the PV pair's smem_p, so the bulk copy is a raw
        # same-offset block move).
        sP_src = storage.smem_p_src.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (rows, words, PIPE) — row-major words within a row to match the
        # producer's per-thread st.async.v4 slot layout.
        smem_corr = storage_pv.smem_corr.get_tensor(
            cute.make_layout(
                (self.corr_rows, self.corr_words_per_row, self.p_cor_stage),
                stride=(
                    self.corr_words_per_row,
                    1,
                    self.corr_rows * self.corr_words_per_row,
                ),
            )
        )
        # (compute_threads,)
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage_pv.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_total_compute_warps * self.threads_per_warp)
        )
        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        # W12-W15 retain their original donation before compute requests its
        # budget. Auto's CTA0/W12 then schedules independently of Q/K TMA.
        if cutlass.const_expr(self.force_branch == "auto"):
            if warp_idx >= self.num_active_warps:
                cute.arch.setmaxregister_decrease(self.dummy_reg_num)
            if warp_idx == self.num_active_warps:
                scheduler_bidx, _, _ = cute.arch.block_idx()
                scheduler_rank = cute.arch.make_warp_uniform(
                    scheduler_bidx % self.cluster_shape_mnk[0]
                )
                if scheduler_rank == 0:
                    self.preferred_clc_producer(
                        shared_clc_pipeline,
                        tile_sched_params,
                        clc_response_ptr,
                    )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Role-separated load warps: W9 loads Q+K, W10 loads V.
        # ///////////////////////////////////////////////////////////////////////////////
        # One aligned donor warpgroup executes one dec before its per-warp work.
        if (
            warp_idx >= self.num_total_compute_warps
            and warp_idx < self.num_active_warps
        ):
            if cutlass.const_expr(self.force_branch == "auto"):
                cute.arch.setmaxregister_decrease(self.mixed_preferred_other_reg_num)
            if warp_idx == self.load_tma_k_warp_id:
                role_bidx, _, _ = cute.arch.block_idx()
                role_cta_rank = cute.arch.make_warp_uniform(
                    role_bidx % self.cluster_shape_mnk[0]
                )
                role_pair_rank = role_cta_rank % self.cta_pair_size
                if role_cta_rank < self.cta_pair_size:
                    load_q_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.load_q_stage
                    )
                    load_k_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.load_k_stage
                    )
                    # ABA fix: independent 9-deep notification cursor, created
                    # once and carried across persistent work tiles (never reset
                    # at a work boundary).
                    k_armed_notify_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.k_armed_stage
                    )
                    clc_pipeline, work_tile, clc_consumer_state = (
                        self.make_clc_consumer(
                            shared_clc_pipeline,
                        )
                    )
                    tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                        tile_sched_params,
                        cute.arch.block_idx(),
                        cute.arch.grid_dim(),
                        clc_response_ptr,
                    )
                    clc_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
                    )
                    while work_tile.is_valid_tile:
                        if (
                            cutlass.const_expr(self.force_branch != "auto")
                            and role_cta_rank == 0
                        ):
                            clc_pipeline.producer_acquire(clc_producer_state)
                            mbarrier_addr = clc_pipeline.producer_get_barrier(
                                clc_producer_state
                            )
                            tile_sched.advance_to_next_work(mbarrier_addr)
                            clc_producer_state.advance()
                        _raw = work_tile.tile_idx
                        # CLC coordinates are (cluster-rank, batch, split-kv).
                        _batch_idx, _query_idx = decode_work_y(
                            _raw[1],
                            cute.size(mO.shape[2]),
                            work_q_fdd,
                        )
                        blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                        role_blk_coord = (
                            role_pair_rank,
                            blk_coord[1],
                            blk_coord[2],
                            blk_coord[3],
                            blk_coord[4],
                        )
                        k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                            split_kv,
                            cache_seqs,
                            block_split_kvs,
                            role_blk_coord,
                        )
                        if k_tile_count > 0:
                            # Construct fixed common/tma_qk params for load_tma
                            tma_common_params = SimpleNamespace(
                                blk_coord=role_blk_coord,
                                local_split_kv=local_split_kv,
                                load_q_pipeline=load_q_pipeline,
                                load_k_pipeline=load_k_pipeline,
                                load_v_pipeline=load_v_pipeline,
                                mPT=mPT,
                            )
                            # tmaKV: W9 no longer issues K (moved to the PV
                            # W10 multicast); it only paces the ring and signals
                            # k_armed to the paired PV CTA.
                            tma_qk_params = SimpleNamespace(
                                tiled_mma_qk=tiled_mma_qk,
                                tiled_mma_qk_tmem=tiled_mma_qk_tmem,
                                tma_atom_q_latent=tma_atom_q_latent,
                                tma_atom_q_rope=tma_atom_q_rope,
                                mQL=mQL,
                                mQR=mQR,
                                kc_latent_base_ptr=kc_latent_base_ptr,
                                q_latent_half_smem_layout_staged=(
                                    q_latent_half_smem_layout_staged
                                ),
                                sQ_rope=sQ_rope,
                                k_armed_mbar=k_armed_mbar,
                                k_armed_peer=role_pair_rank + self.cta_pair_size,
                            )
                            # Load tma
                            (
                                load_q_producer_state,
                                load_k_producer_state,
                                k_armed_notify_state,
                            ) = self.load_tma_qk(
                                tma_common_params,
                                tma_qk_params,
                                k_index,
                                k_tile_count,
                                load_q_producer_state,
                                load_k_producer_state,
                                k_armed_notify_state,
                            )
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()

                    if (
                        cutlass.const_expr(self.force_branch != "auto")
                        and role_cta_rank == 0
                    ):
                        clc_pipeline.producer_tail(clc_producer_state)
                    load_q_pipeline.producer_tail(load_q_producer_state)
                    load_k_pipeline.producer_tail(load_k_producer_state)
            if warp_idx == self.load_tma_v_warp_id:
                role_bidx, _, _ = cute.arch.block_idx()
                role_cta_rank = cute.arch.make_warp_uniform(
                    role_bidx % self.cluster_shape_mnk[0]
                )
                role_pair_rank = role_cta_rank % self.cta_pair_size
                if role_cta_rank >= self.cta_pair_size:
                    load_v_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.load_v_stage
                    )
                    # tmaKV: PV-side K issue cursor over the QK ring. Consumer
                    # semantics: each wait blocks until the paired QK CTA's W9
                    # arms the stage and remote-arrives k_armed.
                    load_k_issue_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.load_k_stage
                    )
                    # ABA fix: independent 9-deep notification wait cursor,
                    # mirroring W9's notify producer; carried across work tiles.
                    k_armed_notify_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.k_armed_stage
                    )
                    clc_pipeline, work_tile, clc_consumer_state = (
                        self.make_clc_consumer(
                            shared_clc_pipeline,
                        )
                    )
                    while work_tile.is_valid_tile:
                        _raw = work_tile.tile_idx
                        # CLC coordinates are (cluster-rank, batch, split-kv).
                        _batch_idx, _query_idx = decode_work_y(
                            _raw[1],
                            cute.size(mO.shape[2]),
                            work_q_fdd,
                        )
                        blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                        role_blk_coord = (
                            role_pair_rank,
                            blk_coord[1],
                            blk_coord[2],
                            blk_coord[3],
                            blk_coord[4],
                        )
                        k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                            split_kv,
                            cache_seqs,
                            block_split_kvs,
                            role_blk_coord,
                        )
                        if k_tile_count > 0:
                            # Construct fixed common/tma_pv params for load_tma
                            tma_common_params = SimpleNamespace(
                                blk_coord=role_blk_coord,
                                local_split_kv=local_split_kv,
                                load_v_pipeline=load_v_pipeline,
                                mPT=mPT,
                                pv_n_split_idx=0,
                            )
                            tma_pv_params = SimpleNamespace(
                                tiled_mma_pv=tiled_mma_pv,
                                tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                                mCLT=mCLT,
                                sVC=sVC_for_tma,
                                # tmaKV: K issue moved here from the QK W9.
                                tiled_mma_qk=tiled_mma_qk,
                                tma_atom_c_latent=tma_atom_c_latent,
                                tma_atom_c_rope=tma_atom_c_rope,
                                mCL=mCL,
                                mKR=mKR,
                                sKC=sKC_for_tma,
                                sKC_rope=sKC_rope_for_tma,
                                k_armed_mbar=k_armed_mbar,
                                load_k_pipeline=load_k_pipeline,
                            )
                            # Load tma (K multicast to the paired QK CTA + V)
                            (
                                load_v_producer_state,
                                load_k_issue_state,
                                k_armed_notify_state,
                            ) = self.load_tma_v(
                                tma_common_params,
                                tma_pv_params,
                                k_index,
                                k_tile_count,
                                load_v_producer_state,
                                load_k_issue_state,
                                k_armed_notify_state,
                            )
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
                    load_v_pipeline.producer_tail(load_v_producer_state)

            # ///////////////////////////////////////////////////////////////////////////////
            #  W8 runs QK UMMA and participates in the TMEM-pointer rendezvous on
            #  PV. W11 allocates/frees TMEM on every CTA and runs PV UMMA.
            # ///////////////////////////////////////////////////////////////////////////////
            if warp_idx == self.mma_qk_warp_id:
                role_bidx, _, _ = cute.arch.block_idx()
                role_cta_rank = cute.arch.make_warp_uniform(
                    role_bidx % self.cluster_shape_mnk[0]
                )
                role_pair_rank = role_cta_rank % self.cta_pair_size
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

                if role_cta_rank < self.cta_pair_size:
                    # Q-in-TMEM: create TMEM tensor and S2T copy setup (gr150 pattern)
                    tQ_tmem = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + self.tmem_q_offset, dtype=self.q_dtype
                        ),
                        q_latent_tmem_layout,
                    )
                    tTrQ_fake = tiled_mma_qk_tmem.make_fragment_A(tQ_tmem)
                    tTrQ_tmem = cute.make_tensor(tQ_tmem.iterator, tTrQ_fake.layout)
                    # loopQ: per-half TMEM tensors for the S2T (the SMEM half
                    # views are dynamic ring slots, built per work tile). Half h
                    # covers K-iterations h*4..h*4+3 = 64 TMEM columns.
                    q_half_tmem_cols = (
                        (self.q_tmem_iters // 2) * self.mma_qk_tiler[2] // 4
                    )
                    # Unrolled pair (a mutated python list is a type-unstable
                    # value under the DSL's staged control flow).
                    tQ_tmem_h0 = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + self.tmem_q_offset, dtype=self.q_dtype
                        ),
                        q_latent_half_tmem_layout,
                    )
                    tTrQ_fake_h0 = tiled_mma_qk_tmem.make_fragment_A(tQ_tmem_h0)
                    tTrQ_tmem_half0 = cute.make_tensor(
                        tQ_tmem_h0.iterator, tTrQ_fake_h0.layout
                    )
                    tQ_tmem_h1 = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + self.tmem_q_offset + q_half_tmem_cols,
                            dtype=self.q_dtype,
                        ),
                        q_latent_half_tmem_layout,
                    )
                    tTrQ_fake_h1 = tiled_mma_qk_tmem.make_fragment_A(tQ_tmem_h1)
                    tTrQ_tmem_half1 = cute.make_tensor(
                        tQ_tmem_h1.iterator, tTrQ_fake_h1.layout
                    )
                    tTrQ_tmem_halves = (tTrQ_tmem_half0, tTrQ_tmem_half1)

                    tQ_rope_tmem = cute.make_tensor(
                        cute.recast_ptr(
                            tmem_ptr + self.tmem_q_rope_offset, dtype=self.q_dtype
                        ),
                        q_rope_tmem_layout,
                    )
                    tTrQ_rope_fake = tiled_mma_qk_tmem.make_fragment_A(tQ_rope_tmem)
                    tTrQ_rope_tmem = cute.make_tensor(
                        tQ_rope_tmem.iterator, tTrQ_rope_fake.layout
                    )
                    (
                        s2t_tiled_copy_q_rope,
                        tCsQ_rope_s2t,
                        tCtQ_rope_s2t,
                    ) = self._s2t_copy_and_partition(sQ_rope, tTrQ_rope_tmem)

                    load_q_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.load_q_stage
                    )
                    load_k_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.load_k_stage
                    )
                    mma_s_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.mma_s_stage
                    )
                    clc_pipeline, work_tile, clc_consumer_state = (
                        self.make_clc_consumer(
                            shared_clc_pipeline,
                        )
                    )
                    while work_tile.is_valid_tile:
                        _raw = work_tile.tile_idx
                        # CLC coordinates are (cluster-rank, batch, split-kv).
                        _batch_idx, _query_idx = decode_work_y(
                            _raw[1],
                            cute.size(mO.shape[2]),
                            work_q_fdd,
                        )
                        blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                        role_blk_coord = (
                            role_pair_rank,
                            blk_coord[1],
                            blk_coord[2],
                            blk_coord[3],
                            blk_coord[4],
                        )
                        k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                            split_kv, cache_seqs, block_split_kvs, role_blk_coord
                        )
                        if k_tile_count > 0:
                            mma_common_params = SimpleNamespace(
                                blk_coord=role_blk_coord,
                                local_split_kv=local_split_kv,
                                load_q_pipeline=load_q_pipeline,
                                load_k_pipeline=load_k_pipeline,
                                load_v_pipeline=load_v_pipeline,
                                tmem_ptr=tmem_ptr,
                                is_leader_cta=role_pair_rank == 0,
                                L=self.latent_dim_per_group,
                            )
                            mma_qk_params = SimpleNamespace(
                                mma_s_pipeline=mma_s_pipeline,
                                s2t_tiled_copy_q_rope=s2t_tiled_copy_q_rope,
                                tCsQ_rope_s2t=tCsQ_rope_s2t,
                                tCtQ_rope_s2t=tCtQ_rope_s2t,
                                tiled_mma_qk_tmem=tiled_mma_qk_tmem,
                                tTrQ_tmem=tTrQ_tmem,
                                tTrQ_rope_tmem=tTrQ_rope_tmem,
                                tTrQ_tmem_halves=tTrQ_tmem_halves,
                                kc_latent_base_ptr=kc_latent_base_ptr,
                                q_latent_half_smem_layout_staged=(
                                    q_latent_half_smem_layout_staged
                                ),
                                sQ_rope=sQ_rope,
                                sKC=sKC,
                                sKC_rope=sKC_rope,
                            )
                            (
                                tiled_mma_qk,
                                tiled_mma_qk_tmem,
                                load_q_consumer_state,
                                load_k_consumer_state,
                                mma_s_producer_state,
                            ) = self.mma_qk_warp_body(
                                mma_common_params,
                                mma_qk_params,
                                k_tile_count,
                                tiled_mma_qk,
                                tiled_mma_qk_tmem,
                                load_q_consumer_state,
                                load_k_consumer_state,
                                mma_s_producer_state,
                            )
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()

                    mma_s_pipeline.producer_tail(mma_s_producer_state)
                    self.tmem_free_sync_bar.arrive_and_wait()

            if warp_idx == self.mma_pv_warp_id:
                role_bidx, _, _ = cute.arch.block_idx()
                role_cta_rank = cute.arch.make_warp_uniform(
                    role_bidx % self.cluster_shape_mnk[0]
                )
                role_pair_rank = role_cta_rank % self.cta_pair_size
                tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_107"))
                tmem.wait_for_alloc()
                tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

                if role_cta_rank < self.cta_pair_size:
                    # W8 is the final QK TMEM user; W11 owns deallocation.
                    self.tmem_free_sync_bar.arrive_and_wait()
                else:
                    load_v_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.load_v_stage
                    )
                    p_consumer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Consumer, self.p_stage
                    )
                    mma_o_producer_state = pipeline.make_pipeline_state(
                        pipeline.PipelineUserType.Producer, self.mma_o_stage
                    )
                    clc_pipeline, work_tile, clc_consumer_state = (
                        self.make_clc_consumer(
                            shared_clc_pipeline,
                        )
                    )
                    while work_tile.is_valid_tile:
                        _raw = work_tile.tile_idx
                        # CLC coordinates are (cluster-rank, batch, split-kv).
                        _batch_idx, _query_idx = decode_work_y(
                            _raw[1],
                            cute.size(mO.shape[2]),
                            work_q_fdd,
                        )
                        blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                        role_blk_coord = (
                            role_pair_rank,
                            blk_coord[1],
                            blk_coord[2],
                            blk_coord[3],
                            blk_coord[4],
                        )
                        k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                            split_kv, cache_seqs, block_split_kvs, role_blk_coord
                        )
                        if k_tile_count > 0:
                            mma_pv_common_params = SimpleNamespace(
                                blk_coord=role_blk_coord,
                                local_split_kv=local_split_kv,
                                load_v_pipeline=load_v_pipeline,
                                tmem_ptr=tmem_ptr,
                                is_leader_cta=role_pair_rank == 0,
                                L=self.latent_dim_per_group,
                            )
                            mma_pv_params = SimpleNamespace(
                                mma_o_pipeline=mma_o_pipeline,
                                sVC=sVC,
                                sP=sP,
                                p_full_mbar=p_full_mbar,
                                p_empty_mbar=p_empty_mbar,
                                pv_pair_leader_rank=pv_pair_leader_rank,
                                qk_pair_commit_mask=qk_pair_commit_mask,
                            )
                            (
                                tiled_mma_pv,
                                load_v_consumer_state,
                                p_consumer_state,
                                mma_o_producer_state,
                            ) = self.mma_pv_warp_body(
                                mma_pv_common_params,
                                mma_pv_params,
                                k_tile_count,
                                tiled_mma_pv,
                                load_v_consumer_state,
                                p_consumer_state,
                                mma_o_producer_state,
                            )
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()

                    mma_o_pipeline.producer_tail(mma_o_producer_state)

                # Wait for all compute warps' TMEM accesses (incl. the softmax
                # group-mailbox LDTMs, which are outside any pipeline) before
                # deallocating — freeing early lets the dealloc clear TMEM while
                # the last cross-group exchange is still in flight.
                self.tmem_dealloc_sync_bar.arrive_and_wait()
                tmem.relinquish_alloc_permit()
                tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        # Auto QK W0-W7 share softmax; PV correction retains its own scope.
        # ///////////////////////////////////////////////////////////////////////////////
        compute_cta_rank = cute.arch.make_warp_uniform(
            cute.arch.block_idx()[0] % self.cluster_shape_mnk[0]
        )
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx
            <= (
                self.compute_warp_ids[-1]
                + (self.num_correction_groups - 1) * self.num_compute_warps
            )
            and (
                cutlass.const_expr(self.force_branch == "auto")
                or compute_cta_rank >= self.cta_pair_size
                or warp_idx <= self.compute_warp_ids[-1]
            )
        ):
            role_bidx, _, _ = cute.arch.block_idx()
            role_cta_rank = cute.arch.make_warp_uniform(
                role_bidx % self.cluster_shape_mnk[0]
            )
            role_pair_rank = role_cta_rank % self.cta_pair_size
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            if role_cta_rank < self.cta_pair_size:
                # Keep allocation and the full work loop in the same role scope;
                # do not merge this budget with donor warps before computation.
                if cutlass.const_expr(self.force_branch == "auto"):
                    cute.arch.setmaxregister_increase(
                        self.mixed_preferred_softmax_reg_num
                    )
                softmax_second_group = False
                if cutlass.const_expr(self.force_branch == "auto"):
                    softmax_second_group = (
                        cute.arch.make_warp_uniform(warp_idx) >= self.num_compute_warps
                    )
                mma_s_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_s_stage
                )
                p_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_stage
                )
                p_cor_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_cor_stage
                )
                if softmax_second_group:
                    # Preserve g1's original odd-stage initialization.
                    mma_s_consumer_state.advance()
                    p_producer_state.advance()
                    p_cor_producer_state.advance()
                clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                    shared_clc_pipeline,
                )
                while work_tile.is_valid_tile:
                    if cutlass.const_expr(self.force_branch == "auto"):
                        # Consume metadata early; compute still uses current work.
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        next_work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
                    _raw = work_tile.tile_idx
                    # CLC coordinates are (cluster-rank, batch, split-kv).
                    _batch_idx, _query_idx = decode_work_y(
                        _raw[1],
                        cute.size(mO.shape[2]),
                        work_q_fdd,
                    )
                    blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                    role_blk_coord = (
                        role_pair_rank,
                        blk_coord[1],
                        blk_coord[2],
                        blk_coord[3],
                        blk_coord[4],
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, role_blk_coord
                    )
                    if k_tile_count > 0:
                        compute_common_params = SimpleNamespace(
                            blk_coord=role_blk_coord,
                            split_kv=split_kv,
                            local_split_kv=local_split_kv,
                            smem_exchange=softmax_smem_exchange,
                            mAccO=mAccO,
                            mO=mO,
                            K=cache_seqs[role_blk_coord[3]],
                            L=self.latent_dim_per_group,
                            tmem_ptr=tmem_ptr,
                            tidx=tidx,
                        )
                        compute_softmax_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            mma_s_pipeline=mma_s_pipeline,
                            softmax_scale_log2=softmax_scale_log2,
                            # DSMEM P/corr producer context
                            sP_src=sP_src,
                            smem_p_dst_ptr=storage_pv.smem_p.data_ptr(),
                            smem_p_src_ptr=storage.smem_p_src.data_ptr(),
                            smem_corr=smem_corr,
                            p_full_mbar=p_full_mbar,
                            p_empty_mbar=p_empty_mbar,
                            cor_full_mbar=cor_full_mbar,
                            cor_empty_mbar=cor_empty_mbar,
                            dsmem_data_peer=(role_pair_rank + self.cta_pair_size),
                        )
                        (
                            mma_s_consumer_state,
                            p_producer_state,
                            p_cor_producer_state,
                        ) = self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=softmax_second_group,
                        )
                    if cutlass.const_expr(self.force_branch == "auto"):
                        work_tile = next_work_tile
                    else:
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
            else:
                if cutlass.const_expr(self.force_branch == "auto"):
                    cute.arch.setmaxregister_increase(
                        self.mixed_preferred_correction_reg_num
                    )
                p_cor_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.p_cor_stage
                )
                mma_o_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_o_stage
                )

                clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                    shared_clc_pipeline,
                )
                while work_tile.is_valid_tile:
                    if cutlass.const_expr(self.force_branch == "auto"):
                        # Keep current coordinates live while releasing the next
                        # CLC response before correction. An invalid next work
                        # terminates only after the current work is completed.
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        next_work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()
                    _raw = work_tile.tile_idx
                    # CLC coordinates are (cluster-rank, batch, split-kv).
                    _batch_idx, _query_idx = decode_work_y(
                        _raw[1],
                        cute.size(mO.shape[2]),
                        work_q_fdd,
                    )
                    blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                    role_blk_coord = (
                        role_pair_rank,
                        blk_coord[1],
                        blk_coord[2],
                        blk_coord[3],
                        blk_coord[4],
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, role_blk_coord
                    )
                    if k_tile_count > 0:
                        compute_common_params = SimpleNamespace(
                            blk_coord=role_blk_coord,
                            split_kv=split_kv,
                            local_split_kv=local_split_kv,
                            smem_exchange=epilogue_smem_exchange,
                            mAccO=mAccO,
                            mO=mO,
                            K=cache_seqs[role_blk_coord[3]],
                            L=self.latent_dim_per_group,
                            H=mQL.shape[0],
                            tmem_ptr=tmem_ptr,
                            tidx=tidx,
                            tiled_mma_pv=tiled_mma_pv,
                            mma_o_pipeline=mma_o_pipeline,
                            pv_n_split_idx=0,
                            # DSMEM corr consumer context
                            smem_corr=smem_corr,
                            cor_full_mbar=cor_full_mbar,
                            cor_empty_mbar=cor_empty_mbar,
                            qk_peer_rank=role_pair_rank,
                            softmax_scale_log2=softmax_scale_log2,
                        )
                        compute_epilogue_params = SimpleNamespace(
                            output_scale=output_scale,
                            lse_scale=lse_scale,
                            softmax_scale_log2=softmax_scale_log2,
                            mAccLSE=mAccLSE,
                            mLSE=mLSE,
                        )
                        p_cor_consumer_state, mma_o_consumer_state = self.correction(
                            compute_common_params,
                            compute_epilogue_params,
                            k_tile_count=k_tile_count,
                            p_cor_consumer_state=p_cor_consumer_state,
                            mma_o_consumer_state=mma_o_consumer_state,
                            correction_group_idx=warp_idx // self.num_compute_warps,
                        )
                    elif cutlass.const_expr(mAccO is None):
                        _store_empty_output(
                            mO,
                            mLSE,
                            role_pair_rank * 128,
                            128,
                            role_blk_coord[2],
                            role_blk_coord[3],
                            tidx,
                            256,
                        )
                    if cutlass.const_expr(self.force_branch == "auto"):
                        # Event 2 remains the work transition, not release time.
                        work_tile = next_work_tile
                    else:
                        clc_pipeline.consumer_wait(clc_consumer_state)
                        work_tile = self.get_clc_work(clc_response_ptr)
                        clc_pipeline.consumer_release(clc_consumer_state)
                        clc_consumer_state.advance()

            # Release the TMEM dealloc guard (both roles; see W11).
            self.tmem_dealloc_sync_bar.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        # QK W4-W7 retain their independently specialized softmax g1 body.
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            cutlass.const_expr(self.force_branch != "auto")
            and warp_idx >= self.second_compute_warp_ids[0]
            and warp_idx <= self.second_compute_warp_ids[-1]
            and compute_cta_rank < self.cta_pair_size
        ):
            role_bidx, _, _ = cute.arch.block_idx()
            role_cta_rank = cute.arch.make_warp_uniform(
                role_bidx % self.cluster_shape_mnk[0]
            )
            role_pair_rank = role_cta_rank % self.cta_pair_size
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            if role_cta_rank < self.cta_pair_size:
                # As in g0, allocation must dominate the entire role work loop.
                if cutlass.const_expr(self.force_branch == "auto"):
                    cute.arch.setmaxregister_increase(
                        self.mixed_preferred_softmax_reg_num
                    )
                mma_s_consumer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.mma_s_stage
                )
                p_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_stage
                )
                p_cor_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.p_cor_stage
                )
                # g1 starts one stage ahead on every pingponged resource
                # (stage index tracks the global k-tile position; g1 owns the
                # odd positions).
                mma_s_consumer_state.advance()
                p_producer_state.advance()
                p_cor_producer_state.advance()

                clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                    shared_clc_pipeline,
                )
                while work_tile.is_valid_tile:
                    _raw = work_tile.tile_idx
                    # CLC coordinates are (cluster-rank, batch, split-kv).
                    _batch_idx, _query_idx = decode_work_y(
                        _raw[1],
                        cute.size(mO.shape[2]),
                        work_q_fdd,
                    )
                    blk_coord = (_raw[0], 0, _query_idx, _batch_idx, _raw[2])
                    role_blk_coord = (
                        role_pair_rank,
                        blk_coord[1],
                        blk_coord[2],
                        blk_coord[3],
                        blk_coord[4],
                    )
                    k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                        split_kv, cache_seqs, block_split_kvs, role_blk_coord
                    )
                    if k_tile_count > 0:
                        compute_common_params = SimpleNamespace(
                            blk_coord=role_blk_coord,
                            split_kv=split_kv,
                            local_split_kv=local_split_kv,
                            smem_exchange=softmax_smem_exchange,
                            mAccO=mAccO,
                            mO=mO,
                            K=cache_seqs[role_blk_coord[3]],
                            L=self.latent_dim_per_group,
                            tmem_ptr=tmem_ptr,
                            tidx=tidx,
                        )
                        compute_softmax_params = SimpleNamespace(
                            tiled_mma_qk=tiled_mma_qk,
                            mma_s_pipeline=mma_s_pipeline,
                            softmax_scale_log2=softmax_scale_log2,
                            # DSMEM P/corr producer context
                            sP_src=sP_src,
                            smem_p_dst_ptr=storage_pv.smem_p.data_ptr(),
                            smem_p_src_ptr=storage.smem_p_src.data_ptr(),
                            smem_corr=smem_corr,
                            p_full_mbar=p_full_mbar,
                            p_empty_mbar=p_empty_mbar,
                            cor_full_mbar=cor_full_mbar,
                            cor_empty_mbar=cor_empty_mbar,
                            dsmem_data_peer=(role_pair_rank + self.cta_pair_size),
                        )
                        (
                            mma_s_consumer_state,
                            p_producer_state,
                            p_cor_producer_state,
                        ) = self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=True,
                        )
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
            # Release the TMEM dealloc guard (both roles; see W11).
            self.tmem_dealloc_sync_bar.arrive()

        # Cluster-wide exit sync: keep every CTA's SMEM alive until all CTAs
        # have finished their cross-CTA DSMEM traffic (remote mbarrier arrives
        # from the PV pair target QK CTA SMEM right up to the last tile).
        cute.arch.cluster_arrive()
        cute.arch.cluster_wait()

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        lse_scale: cutlass.Float32,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        """
        bidx, bidy, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, bidy, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, max(1, k_tile_per_cta))

        # Alloc shared memory
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[1], blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.threads_per_warp)

            local_lse = cute.make_rmem_tensor(
                cute.make_layout(lse_per_thread), self.lse_dtype
            )
            lse_max = -self.lse_dtype.inf
            # find the max lse
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in cutlass.range_constexpr(lse_per_thread):
                sum_lse += cute.math.exp2(local_lse[i] - lse_max, fastmath=True)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse, fastmath=True)
                if not sum_lse == self.lse_dtype(0.0) or sum_lse != sum_lse
                else -self.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[1], blk_coord[2]] = global_lse * lse_scale
            # store the scale to shared memory
            for i in cutlass.range_constexpr(lse_per_thread):
                split_kv_idx = tidx + i * self.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = (
                        cute.math.exp2(local_lse[i] - global_lse, fastmath=True)
                        if local_lse[i] != -self.lse_dtype.inf
                        else 0.0
                    )

        pipeline.sync(barrier_id=4)

        elements_per_thread = cute.ceil_div(
            self.latent_dim, self.threads_per_warp * self.num_compute_warps
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[1], blk_coord[2]]
        rAccO = cute.make_rmem_tensor(
            cute.make_layout(elements_per_thread), self.acc_dtype
        )
        rO = cute.make_rmem_tensor(cute.make_layout(elements_per_thread), self.o_dtype)
        rAccO.fill(0.0)
        for i in range(local_split_kv):
            for j in cutlass.range_constexpr(elements_per_thread):
                element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
                rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
        rO.store(rAccO.load().to(self.o_dtype))
        for j in cutlass.range_constexpr(elements_per_thread):
            element_idx = tidx + j * self.threads_per_warp * self.num_compute_warps
            mO[blk_coord[0], element_idx, blk_coord[1], blk_coord[2]] = rO[j]
        return

    @staticmethod
    def get_split_kv(
        B: int, S: int, K: int, mma_qk_tiler_mn: tuple, max_active_blocks: int
    ) -> int:
        """Get the proper split_kv value for the MLA kernel based on parameters.

        :param B: Batch size
        :type B: int
        :param S: Sequence length
        :type S: int
        :param K: Sequence length
        :type K: int
        :param mma_qk_tiler_mn: MLA tiling parameters
        :type mma_qk_tiler_mn: tuple
        :param max_active_blocks: Maximum number of active blocks
        :type max_active_blocks: int
        :return: Split_kv value
        :rtype: int
        """
        max_splits = ceil_div(K, mma_qk_tiler_mn[1])
        blocks_per_batch = max(1, max_active_blocks // B // (S * 2))
        split_heur = min(max_splits, blocks_per_batch)
        k_waves = ceil_div(max_splits, split_heur)
        split_wave_aware = ceil_div(max_splits, k_waves)
        max_split_kv = 32
        return min(split_wave_aware, max_split_kv)

    @cute.jit
    def get_k_tile_count(
        self,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        blk_coord: cute.Coord,
    ) -> tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]:
        """Get the current k_index, k_tile_count, and local split_kv value for the MLA kernel.

        :param split_kv: Split_kv value
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param blk_coord: Block coordinate
        :type blk_coord: cute.Coord
        :return: k_index, k_tile_count, split_kv
        :rtype: tuple[cutlass.Int32, cutlass.Int32, cutlass.Int32]
        """
        K = cache_seqs[blk_coord[3]]
        if cutlass.const_expr(self.is_var_split_kv):
            split_kv = block_split_kvs[blk_coord[3]]

        k_tile_total = cute.ceil_div(K, self.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, split_kv)
        k_index = blk_coord[4] * k_tile_per_cta
        k_tile_count = max(0, min(k_tile_total, k_index + k_tile_per_cta) - k_index)

        return k_index, k_tile_count, split_kv

    @cute.jit
    def load_tma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState | None = None,
        load_k_producer_state: pipeline.PipelineState | None = None,
        k_armed_notify_state: pipeline.PipelineState | None = None,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Load wrap to load Q/K tensors. Updates the load qk producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_k_producer_state: The load k producer state
        :type load_k_producer_state: pipeline.PipelineState

        :return: The load q producer state and load k producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[3]]

        # Flatten divide and partition global tensors for QK TMA load
        # (bM, bK, rM, rK, rL)
        mma_qk_tiler_mk = cute.select(self.mma_qk_tiler, mode=[0, 2])
        gQL = cute.flat_divide(qk_params.mQL, mma_qk_tiler_mk)
        mma_qk_tiler_mk_rope = cute.select(self.mma_qk_rope_tiler, mode=[0, 2])
        gQR = cute.flat_divide(qk_params.mQR, mma_qk_tiler_mk_rope)

        thr_mma_q = qk_params.tiled_mma_qk_tmem.get_slice(
            common_params.blk_coord[0] % cute.size(qk_params.tiled_mma_qk_tmem.thr_id)
        )
        tSgQL = thr_mma_q.partition_A(gQL)
        tSgQR = thr_mma_q.partition_A(gQR)

        # tma partition for q (K issue moved to the PV W10, tmaKV).
        # loopQ: the latent partition is rebuilt per half over the dynamic
        # ring-slot views inside _loopq_produce_q below; only the gmem side
        # is fixed. A throwaway half view at slot 0 supplies the smem-side
        # shape for the gmem partition.
        _sQ_shape_probe = cute.make_tensor(
            cute.recast_ptr(
                qk_params.kc_latent_base_ptr,
                swizzle_=qk_params.q_latent_half_smem_layout_staged.inner,
                dtype=self.q_dtype,
            ),
            qk_params.q_latent_half_smem_layout_staged.outer,
        )
        _, tQLgQL_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_latent,
            0,
            cute.make_layout(1),
            cute.group_modes(_sQ_shape_probe, 0, 3),
            cute.group_modes(tSgQL, 0, 3),
        )

        tQsQ_rope, tQRgQR_mkl = cpasync.tma_partition(
            qk_params.tma_atom_q_rope,
            0,
            cute.make_layout(1),
            cute.group_modes(qk_params.sQ_rope, 0, 3),
            cute.group_modes(tSgQR, 0, 3),
        )

        tQLgQL = tQLgQL_mkl[
            None, None, None, common_params.blk_coord[2], common_params.blk_coord[3]
        ]
        tQRgQR = tQRgQR_mkl[
            None, None, None, common_params.blk_coord[2], common_params.blk_coord[3]
        ]

        # set extra params
        common_params.mPT = mPT
        qk_params.tQLgQL = tQLgQL
        qk_params.tQRgQR = tQRgQR
        qk_params.tQsQ_rope = tQsQ_rope

        # ---- loopQ: Q production at work-tile start ----
        # Rope goes to its dedicated 8KB home through the 1-stage load_q
        # pipeline (its produce/acquire handshake is what makes the rope
        # home multi-wave safe: the next wave's acquire waits the previous
        # S2T's release).
        rope_bar_ptr = common_params.load_q_pipeline.producer_get_barrier(
            load_q_producer_state
        )
        common_params.load_q_pipeline.producer_acquire(load_q_producer_state)
        for i in cutlass.range_constexpr(self.iterations_qk_rope):
            cute.copy(
                qk_params.tma_atom_q_rope,
                qk_params.tQRgQR[None, 0, i],
                qk_params.tQsQ_rope[None, i],
                tma_bar_ptr=rope_bar_ptr,
            )
        load_q_producer_state.advance()
        # Q latent: two ordinary productions on the K ring (gr150 pattern,
        # multi-wave safe: producer_acquire waits until the slot's previous
        # K/Q occupant was consumed). expected_tx override because a Q half
        # (32,768B x pair) differs from a K stage (latent+rope). W9 still
        # remote-arrives k_armed for these slots — W10 waits it and SKIPS
        # the fill (see load_tma_v) so the k_armed phase bookkeeping stays
        # symmetric across rounds with and without Q occupancy.
        for _h in cutlass.range_constexpr(2):
            half_bar_ptr = common_params.load_k_pipeline.producer_get_barrier(
                load_k_producer_state
            )
            common_params.load_k_pipeline.producer_acquire(
                load_k_producer_state,
                expected_tx=cutlass.Int32(self.tma_copy_q_half_bytes),
            )
            _slot = load_k_producer_state.index
            _sQ_half = cute.make_tensor(
                cute.recast_ptr(
                    qk_params.kc_latent_base_ptr + _slot * self.q_ring_slot_bytes,
                    swizzle_=qk_params.q_latent_half_smem_layout_staged.inner,
                    dtype=self.q_dtype,
                ),
                qk_params.q_latent_half_smem_layout_staged.outer,
            )
            _tQsQ_h, _ = cpasync.tma_partition(
                qk_params.tma_atom_q_latent,
                0,
                cute.make_layout(1),
                cute.group_modes(_sQ_half, 0, 3),
                cute.group_modes(tSgQL, 0, 3),
            )
            for i in cutlass.range_constexpr(self.iterations_qk_latent // 2):
                cute.copy(
                    qk_params.tma_atom_q_latent,
                    qk_params.tQLgQL[
                        None, 0, _h * (self.iterations_qk_latent // 2) + i
                    ],
                    _tQsQ_h[None, (i, 0)],
                    tma_bar_ptr=half_bar_ptr,
                )
            with cute.arch.elect_one():
                # ABA fix: the notification mbar is indexed by the 9-deep
                # notify cursor, decoupled from the 7-deep data cursor.
                _arrive_cluster_mbarrier(
                    qk_params.k_armed_mbar.get_barrier(k_armed_notify_state.index),
                    qk_params.k_armed_peer,
                )
            load_k_producer_state.advance()
            k_armed_notify_state.advance()

        while k_tile_count > 0:
            (
                load_q_producer_state,
                load_k_producer_state,
                k_armed_notify_state,
            ) = self.load_tma_qk_one_k_tile(
                common_params,
                qk_params,
                k_index,
                k_tile_count,
                load_q_producer_state,
                load_k_producer_state,
                k_armed_notify_state,
                load_q=False,
            )
            k_index += 1
            k_tile_count -= 1

        return load_q_producer_state, load_k_producer_state, k_armed_notify_state

    @cute.jit
    def load_tma_v(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
        load_k_issue_state: pipeline.PipelineState = None,
        k_armed_notify_state: pipeline.PipelineState = None,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Load wrap to load V tensors. Updates the load v producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load v producer state
        :rtype: pipeline.PipelineState
        """
        # page table
        mPT = common_params.mPT[None, common_params.blk_coord[3]]

        # Flatten divide and partition global tensors for V TMA load
        page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        gCLT = cute.flat_divide(v_params.mCLT, (self.mma_pv_tiler[1], page_tile_size))
        cta_n = self.mma_pv_tiler[1] // v_params.tiled_mma_pv.thr_id.shape

        gCLT = cute.logical_divide(gCLT, (cta_n,))[
            (None, common_params.blk_coord[0]), None, None, None, None
        ]
        tOgCLT = cute.tiled_divide(gCLT, (cta_n, page_tile_size))
        tOgCLT = tOgCLT[None, 0, 0, None, None, None]
        # tma partition for vc
        # smem: ((atom_v, rest_v), STAGE)
        # gmem: ((atom_v, rest_v), RestM, RestK, RestL)
        tVCsVC, tCLTgCLT = cpasync.tma_partition(
            v_params.tma_atom_c_latent_transpose,
            0,
            cute.make_layout(1),
            v_params.sVC,
            tOgCLT,
        )

        # tmaKV: K partitions, PV-side. blk_coord[0] here is the PV CTA's
        # pair rank == the target QK CTA's pair rank, so the page-index and
        # slicing math is identical to the original QK-side W9 code. The
        # smem views come from THIS CTA's union storage: identical offsets
        # on the receiving QK CTA (shared header design), which is what the
        # multicast delivers to.
        k_thr_shape = v_params.tiled_mma_qk.thr_id.shape
        k_cta_n = min(
            v_params.tiled_mma_qk.op.shape_mnk[1] // k_thr_shape,
            self.page_size,
        )
        k_page_tile_size = min(self.page_size, k_cta_n)
        gCL = cute.tiled_divide(v_params.mCL, (k_page_tile_size, self.mma_qk_tiler[2]))
        tSgCL = (
            gCL[None, common_params.blk_coord[0] % k_thr_shape, None, None]
            if k_cta_n < self.page_size
            else gCL[None, 0, None, None]
        )
        gKR = cute.tiled_divide(
            v_params.mKR, (k_page_tile_size, self.mma_qk_rope_tiler[2])
        )
        tSgKR = (
            gKR[None, common_params.blk_coord[0] % k_thr_shape, None, None]
            if k_cta_n < self.page_size
            else gKR[None, 0, None, None]
        )
        tKCsKC, tCLgCL = cpasync.tma_partition(
            v_params.tma_atom_c_latent,
            0,
            cute.make_layout(1),
            v_params.sKC,
            tSgCL,
        )
        tKCsKC_rope, tKRgKR = cpasync.tma_partition(
            v_params.tma_atom_c_rope,
            0,
            cute.make_layout(1),
            v_params.sKC_rope,
            tSgKR,
        )
        # Single-target multicast mask: the paired QK CTA only (issuer NOT
        # included - K must not land in PV smem, see atom construction note).
        k_mcast_mask = cutlass.Int32(1) << common_params.blk_coord[0]
        k_page_per_tile = ceil_div(self.mma_qk_tiler[1] // self.page_size, k_thr_shape)
        # Hoist every Meta-bearing object out of the dynamic while below (the
        # DSL cannot carry Meta values like a CtaGroup through staged control
        # flow): locals only inside the loop, and the per-stage full-mbar
        # pointer is computed by base + index arithmetic instead of
        # producer_get_barrier(state).
        k_armed_mbar_l = v_params.k_armed_mbar
        k_atom_latent_l = v_params.tma_atom_c_latent
        k_atom_rope_l = v_params.tma_atom_c_rope
        # loopQ: the mbar ARRAY base must not depend on the entry cursor
        # (multi-tile calls enter with index != 0); probe with a fresh
        # zero-index state.
        _zero_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.load_k_stage
        )
        k_bar_base = v_params.load_k_pipeline.producer_get_barrier(_zero_state)

        # loopQ: mirror the two Q slot occupancies at each work-tile start.
        # W9 armed these slots for Q and remote-arrived k_armed; wait it (so
        # the k_armed phase bookkeeping stays symmetric on every round) and
        # advance WITHOUT filling K.
        for _h in cutlass.range_constexpr(2):
            # ABA fix: parity wait on the 9-deep notify cursor; the 7-deep
            # issue cursor still tracks the data-ring slot to skip.
            k_armed_mbar_l.wait(k_armed_notify_state.index, k_armed_notify_state.phase)
            load_k_issue_state.advance()
            k_armed_notify_state.advance()

        # set extra params
        common_params.mPT = mPT
        v_params.tCLTgCLT = tCLTgCLT
        v_params.tVCsVC = tVCsVC

        while k_tile_count > 0:
            # ---- K issue (before V, same warp; user design) ----
            k_pidx = cute.make_rmem_tensor(
                cute.make_layout(k_page_per_tile), cutlass.Int32
            )
            for i in cutlass.range_constexpr(k_page_per_tile):
                k_pidx[i] = (
                    _page_index(common_params.mPT, k_index)
                    if self.mma_qk_tiler[1] // self.page_size == 1
                    else _page_index(
                        common_params.mPT,
                        (k_index * k_thr_shape + common_params.blk_coord[0])
                        * k_page_per_tile
                        + i,
                    )
                )
            # Wait for the paired QK CTA's W9 to arm the stage (local empty
            # observed + leader expect_tx armed) before writing into it.
            # ABA fix: wait on the 9-deep notify cursor; K data barrier and
            # SMEM stage addressing stay on the 7-deep issue cursor.
            k_armed_mbar_l.wait(k_armed_notify_state.index, k_armed_notify_state.phase)
            k_tma_bar_ptr = k_bar_base + load_k_issue_state.index
            for i in range(self.iterations_qk_latent):
                for k in range(k_page_per_tile):
                    cute.copy(
                        k_atom_latent_l,
                        tCLgCL[None, i, k_pidx[k]],
                        tKCsKC[None, k, 0, (i, load_k_issue_state.index)],
                        tma_bar_ptr=k_tma_bar_ptr,
                        mcast_mask=k_mcast_mask,
                    )
            for i in cutlass.range_constexpr(self.iterations_qk_rope):
                for k in cutlass.range_constexpr(k_page_per_tile):
                    cute.copy(
                        k_atom_rope_l,
                        tKRgKR[None, i, k_pidx[k]],
                        tKCsKC_rope[None, k, 0, load_k_issue_state.index],
                        tma_bar_ptr=k_tma_bar_ptr,
                        mcast_mask=k_mcast_mask,
                    )
            load_k_issue_state.advance()
            k_armed_notify_state.advance()
            # ---- V issue (unchanged) ----
            load_v_producer_state = self.load_tma_v_one_k_tile(
                common_params,
                v_params,
                k_index,
                load_v_producer_state,
            )
            k_index += 1
            k_tile_count -= 1
        return load_v_producer_state, load_k_issue_state, k_armed_notify_state

    @cute.jit
    def load_tma_qk_one_k_tile(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        load_q_producer_state: pipeline.PipelineState,
        load_k_producer_state: pipeline.PipelineState,
        k_armed_notify_state: pipeline.PipelineState,
        load_q: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Load one k-tile of Q/C latent/rope tensors. Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param k_tile_count: The k tile count
        :type k_tile_count: cutlass.Int32
        :param load_q_producer_state: The load q producer state
        :type load_q_producer_state: pipeline.PipelineState
        :param load_k_producer_state: The load kv producer state
        :type load_k_producer_state: pipeline.PipelineState
        :param load_q: Whether to load q
        :type load_q: bool

        :return: The load q producer state and load kv producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """
        # loopQ: Q production happens in load_tma_qk at work-tile start
        # (two ring-slot productions + rope); nothing Q-related per k-tile,
        # and no alias gate — the ring's own produce/consume protocol covers
        # multi-wave reuse of every slot.
        # tmaKV: the K data itself is issued by the paired PV CTA's W10 as a
        # TMA multicast into this CTA's stage. W9 keeps the ring pacing:
        # producer_acquire waits the LOCAL stage-empty and (leader) arms the
        # expect_tx on the pair-leader full mbar - exactly the accounting the
        # cta_group::2 multicast tx aggregation targets. Then ONE lane
        # remote-arrives the paired PV CTA's k_armed[stage] to release the
        # issue.
        common_params.load_k_pipeline.producer_acquire(load_k_producer_state)
        with cute.arch.elect_one():
            # ABA fix: notify via the 9-deep cursor (data ring stays 7-deep).
            _arrive_cluster_mbarrier(
                qk_params.k_armed_mbar.get_barrier(k_armed_notify_state.index),
                qk_params.k_armed_peer,
            )
        load_k_producer_state.advance()
        k_armed_notify_state.advance()

        return load_q_producer_state, load_k_producer_state, k_armed_notify_state

    @cute.jit
    def load_tma_v_one_k_tile(
        self,
        common_params: SimpleNamespace,
        v_params: SimpleNamespace,
        k_index: cutlass.Int32,
        load_v_producer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Load one k-tile of compressed latent transpose tensor(v). Updates the load qkv producer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param v_params: The load tma v parameters
        :type v_params: SimpleNamespace
        :param k_index: The k index
        :type k_index: cutlass.Int32
        :param load_v_producer_state: The load v producer state
        :type load_v_producer_state: pipeline.PipelineState

        :return: The load qkv producer state
        :rtype: pipeline.PipelineState
        """
        page_per_tile = self.mma_pv_tiler[2] * self.iterations_pv_k // self.page_size
        page_per_subtile = ceil_div(page_per_tile, self.iterations_pv_k)
        k_idx = cute.make_rmem_tensor(cute.make_layout(page_per_tile), cutlass.Int32)
        for i in cutlass.range_constexpr(page_per_tile):
            k_idx[i] = (
                _page_index(common_params.mPT, k_index)
                if page_per_tile == 1
                else _page_index(common_params.mPT, k_index * page_per_tile + i)
            )
        # get the mbar ptr from pipeline.
        tma_bar_ptr = common_params.load_v_pipeline.producer_get_barrier(
            load_v_producer_state
        )
        common_params.load_v_pipeline.producer_acquire(load_v_producer_state)
        n_offset = common_params.pv_n_split_idx * self.iterations_pv_n
        for j in cutlass.range_constexpr(self.iterations_pv_n):
            for i in cutlass.range_constexpr(self.iterations_pv_k):
                if cutlass.const_expr(page_per_tile > 1):
                    for k in cutlass.range_constexpr(page_per_subtile):
                        k_idx_i = k_idx[k + i * page_per_subtile]
                        cute.copy(
                            v_params.tma_atom_c_latent_transpose,
                            v_params.tCLTgCLT[None, j + n_offset, 0, k_idx_i],
                            v_params.tVCsVC[
                                None, 0, k, ((j, i), load_v_producer_state.index)
                            ],
                            tma_bar_ptr=tma_bar_ptr,
                        )
                else:
                    cute.copy(
                        v_params.tma_atom_c_latent_transpose,
                        v_params.tCLTgCLT[None, j + n_offset, i, k_idx[0]],
                        v_params.tVCsVC[
                            None, 0, 0, ((j, i), load_v_producer_state.index)
                        ],
                        tma_bar_ptr=tma_bar_ptr,
                    )
        load_v_producer_state.advance()
        return load_v_producer_state

    @cute.jit
    def mma_qk_warp_body(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_qk_tmem: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """QK-only MMA warp body. Produces one S tile per k-tile."""

        tSrKC = tiled_mma_qk.make_fragment_B(qk_params.sKC)
        tSrKC_rope = tiled_mma_qk.make_fragment_B(qk_params.sKC_rope)
        tStS_shape = tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        # use real tmem ptr for tStS
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)

        # set more parameters
        qk_params.tSrKC = tSrKC
        qk_params.tSrKC_rope = tSrKC_rope
        qk_params.tStS_staged = tStS_staged
        load_q_pipeline = common_params.load_q_pipeline
        load_k_pipeline_l = common_params.load_k_pipeline
        if common_params.is_leader_cta:
            load_q_release_state = load_q_consumer_state.clone()
            # loopQ: Q latent arrives as two K-ring slot productions. For
            # each half: wait the (pair-aggregated) full, build the dynamic
            # slot view, S2T its 4 K-iterations into the TMEM half, then
            # consumer_release (tcgen05.commit tracks the UTCCPs and
            # multicasts the empty to both QK CTAs) — after which the K
            # stream reuses the slot. Both W9s used the same ring cursor, so
            # the collective UTCCP reads the same local offset on both CTAs.
            for _h in cutlass.range_constexpr(2):
                load_k_pipeline_l.consumer_wait(load_k_consumer_state)
                _slot = load_k_consumer_state.index
                _sQ_half = cute.make_tensor(
                    cute.recast_ptr(
                        qk_params.kc_latent_base_ptr + _slot * self.q_ring_slot_bytes,
                        swizzle_=qk_params.q_latent_half_smem_layout_staged.inner,
                        dtype=self.q_dtype,
                    ),
                    qk_params.q_latent_half_smem_layout_staged.outer,
                )
                (
                    _s2t_copy_h,
                    _tCsQ_h,
                    _tCtQ_h,
                ) = self._s2t_copy_and_partition(
                    _sQ_half, qk_params.tTrQ_tmem_halves[_h]
                )
                for q_stage in range(self.q_tmem_iters // 2):
                    for k_block in cutlass.range_constexpr(cute.size(_tCsQ_h.shape[3])):
                        if cutlass.const_expr(self.mma_qk_tiler[0] == 128):
                            cta_m_idx = common_params.blk_coord[0] % 2
                            src_m_parts = cute.size(_tCsQ_h.shape[1])
                            for m_part in cutlass.range_constexpr(src_m_parts):
                                cute.copy(
                                    _s2t_copy_h,
                                    _tCsQ_h[None, m_part, None, k_block, (q_stage, 0)],
                                    _tCtQ_h[
                                        None,
                                        m_part + cta_m_idx * src_m_parts,
                                        None,
                                        k_block,
                                        (q_stage, 0),
                                    ],
                                )
                        else:
                            cute.copy(
                                _s2t_copy_h,
                                _tCsQ_h[None, None, None, k_block, (q_stage, 0)],
                                _tCtQ_h[None, None, None, k_block, (q_stage, 0)],
                            )
                load_k_pipeline_l.consumer_release(load_k_consumer_state)
                load_k_consumer_state.advance()
            # Rope: dedicated home through the 1-stage load_q pipeline.
            load_q_pipeline.consumer_wait(load_q_consumer_state)
            load_q_consumer_state.advance()
            for k_block in cutlass.range_constexpr(
                cute.size(qk_params.tCsQ_rope_s2t.shape[3])
            ):
                if cutlass.const_expr(self.mma_qk_tiler[0] == 128):
                    cta_m_idx = common_params.blk_coord[0] % 2
                    src_m_parts = cute.size(qk_params.tCsQ_rope_s2t.shape[1])
                    for m_part in cutlass.range_constexpr(src_m_parts):
                        cute.copy(
                            qk_params.s2t_tiled_copy_q_rope,
                            qk_params.tCsQ_rope_s2t[None, m_part, None, k_block, 0],
                            qk_params.tCtQ_rope_s2t[
                                None,
                                m_part + cta_m_idx * src_m_parts,
                                None,
                                k_block,
                                0,
                            ],
                        )
                else:
                    cute.copy(
                        qk_params.s2t_tiled_copy_q_rope,
                        qk_params.tCsQ_rope_s2t[None, None, None, k_block, 0],
                        qk_params.tCtQ_rope_s2t[None, None, None, k_block, 0],
                    )
            # Release the rope home as soon as the rope S2T is tracked
            # (tcgen05.commit) — the next wave's rope producer_acquire waits
            # on this, which is the rope home's multi-wave safety.
            load_q_pipeline.consumer_release(load_q_release_state)
            load_q_release_state.advance()
            (
                tiled_mma_qk,
                tiled_mma_qk_tmem,
                load_q_consumer_state,
                load_k_consumer_state,
                mma_s_producer_state,
            ) = self.mma_qk(
                common_params,
                qk_params,
                tiled_mma_qk,
                tiled_mma_qk_tmem,
                load_q_consumer_state,
                load_k_consumer_state,
                mma_s_producer_state,
                wait_q=False,  # loopQ: Q waits handled in the S2T section
            )
            k_tile_count -= 1

            while k_tile_count > 0:
                (
                    tiled_mma_qk,
                    tiled_mma_qk_tmem,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                ) = self.mma_qk(
                    common_params,
                    qk_params,
                    tiled_mma_qk,
                    tiled_mma_qk_tmem,
                    load_q_consumer_state,
                    load_k_consumer_state,
                    mma_s_producer_state,
                    wait_q=False,
                )
                k_tile_count -= 1
            # (Q consumer release moved up, right after the S2T loops.)
            # PV mainloop runs on the dedicated mma_pv warp.

        return (
            tiled_mma_qk,
            tiled_mma_qk_tmem,
            load_q_consumer_state,
            load_k_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def mma_pv_warp_body(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """PV-only MMA warp body. Runs one mma_pv call per k-tile.

        The A operand (P) lives in this CTA's SMEM, filled over DSMEM by the QK
        pair's softmax warps. The k-tile handshake is the S2S protocol on
        p_full/p_empty mbarriers instead of an intra-CTA pipeline.
        """

        tOrP = tiled_mma_pv.make_fragment_A(pv_params.sP)
        tOrVC = tiled_mma_pv.make_fragment_B(pv_params.sVC)

        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1],
            ),
        )
        tOtO_staged = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )

        pv_params.tOrP = tOrP
        pv_params.tOrVC = tOrVC
        pv_params.tOtO_staged = tOtO_staged

        # O accumulates across k-tiles, so start PV accumulation only once.
        tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, False)

        # Both CTAs execute the P-readiness protocol, but keep the CtaGroup
        # GEMM entirely out of a runtime branch inside mma_pv: CuTe Meta
        # objects cannot be SSA-merged across such a branch.
        if common_params.is_leader_cta:
            while k_tile_count > 0:
                (
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                ) = self.mma_pv(
                    common_params,
                    pv_params,
                    tiled_mma_pv,
                    load_v_consumer_state,
                    p_mma_consumer_state,
                    mma_o_producer_state,
                )
                k_tile_count -= 1
        else:
            while k_tile_count > 0:
                p_mma_consumer_state = self.mma_pv_peer(pv_params, p_mma_consumer_state)
                k_tile_count -= 1

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_qk(
        self,
        common_params: SimpleNamespace,
        qk_params: SimpleNamespace,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_qk_tmem: cute.TiledMma,
        load_q_consumer_state: pipeline.PipelineState,
        load_k_consumer_state: pipeline.PipelineState,
        mma_s_producer_state: pipeline.PipelineState,
        wait_q: bool,
    ) -> tuple[
        cute.TiledMma,
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for Q*K^T. Updates the tiled MMA QK and pipeline states.

        :param qk_params: The qk parameters
        :type qk_params: SimpleNamespace
        :param tiled_mma_qk: The tiled mma qk
        :type tiled_mma_qk: cute.TiledMma
        :param load_q_consumer_state: The load q consumer state
        :type load_q_consumer_state: pipeline.PipelineState
        :param load_k_consumer_state: The load k consumer state
        :type load_k_consumer_state: pipeline.PipelineState
        :param mma_s_producer_state: The mma s producer state
        :type mma_s_producer_state: pipeline.PipelineState

        :return: The tiled mma qk, the load q consumer state, the load k consumer state, and the mma s producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """
        tStS = qk_params.tStS_staged[None, None, None, mma_s_producer_state.index]

        load_q_pipeline = common_params.load_q_pipeline
        load_k_pipeline = common_params.load_k_pipeline

        if cutlass.const_expr(wait_q):
            load_q_pipeline.consumer_wait(load_q_consumer_state)
        load_k_pipeline.consumer_wait(load_k_consumer_state)

        qk_params.mma_s_pipeline.producer_acquire(mma_s_producer_state)

        # All latent Q iterations are in TMEM.
        kc_stage = load_k_consumer_state.index
        tiled_mma_qk_tmem.set(tcgen05.Field.ACCUMULATE, False)
        for q_stage in range(self.q_tmem_iters):
            for k_block in cutlass.range_constexpr(
                cute.size(qk_params.tTrQ_tmem.shape[2])
            ):
                cute.gemm(
                    tiled_mma_qk_tmem,
                    tStS,
                    qk_params.tTrQ_tmem[None, None, k_block, (q_stage, 0)],
                    qk_params.tSrKC[None, None, k_block, (q_stage, kc_stage)],
                    tStS,
                )
                tiled_mma_qk_tmem.set(tcgen05.Field.ACCUMULATE, True)

        # Q rope is also in TMEM.
        for q_stage in range(self.iterations_qk_rope):
            kc_stage = load_k_consumer_state.index
            for k_block in cutlass.range_constexpr(
                self.rope_dim // tiled_mma_qk_tmem.shape_mnk[2]
            ):
                cute.gemm(
                    tiled_mma_qk_tmem,
                    tStS,
                    qk_params.tTrQ_rope_tmem[None, None, k_block, q_stage],
                    qk_params.tSrKC_rope[None, None, k_block, kc_stage],
                    tStS,
                )
                tiled_mma_qk_tmem.set(tcgen05.Field.ACCUMULATE, True)
        load_k_pipeline.consumer_release(load_k_consumer_state)
        load_k_consumer_state.advance()
        if cutlass.const_expr(wait_q):
            load_q_consumer_state.advance()

        qk_params.mma_s_pipeline.producer_commit(mma_s_producer_state)
        mma_s_producer_state.advance()
        return (
            tiled_mma_qk,
            tiled_mma_qk_tmem,
            load_q_consumer_state,
            load_k_consumer_state,
            mma_s_producer_state,
        )

    @cute.jit
    def wait_p_local_ready(
        self,
        pv_params: SimpleNamespace,
        p_mma_consumer_state: pipeline.PipelineState,
        arrive_self: bool,
    ) -> None:
        """Wait for and rearm this PV CTA's local P receive slot.

        p_full completes on (expected tx bytes) + (arrive count 1). On the
        NON-LEADER the arrive is its own elect_one vote (arrive_self=True) and
        a CTA-scope wait suffices. On the LEADER the single arrive comes
        REMOTELY from the non-leader (release.cluster) after ITS half landed,
        so the leader's one wait doubles as the pair rendezvous and — per the
        memory-model guide §4/§8 (mbarrier pairs carry implicit async fencing
        at CTA scope only) — uses acquire.cluster semantics.
        """
        # Each QK peer bulk-copies into this PV CTA's P SMEM and completes
        # this same CTA's p_full mbarrier (same-CTA data/mbar pairing is
        # required by the ISA for cluster async stores).
        if cutlass.const_expr(arrive_self):
            with cute.arch.elect_one():
                pv_params.p_full_mbar.arrive_mbarrier(p_mma_consumer_state.index)
            pv_params.p_full_mbar.wait(
                p_mma_consumer_state.index, p_mma_consumer_state.phase
            )
        else:
            _wait_cluster_mbarrier(
                pv_params.p_full_mbar.get_barrier(p_mma_consumer_state.index),
                p_mma_consumer_state.phase,
            )

        # Arm this local receive slot for its next phase. The QK producer
        # cannot reuse P until the UMMA completion releases p_empty, so the
        # current stage remains stable meanwhile.
        with cute.arch.elect_one():
            cute.arch.mbarrier_expect_tx(
                pv_params.p_full_mbar.get_barrier(p_mma_consumer_state.index),
                self.p_tx_bytes_per_cta,
            )

    @cute.jit
    def mma_pv(
        self,
        common_params: SimpleNamespace,
        pv_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        load_v_consumer_state: pipeline.PipelineState,
        p_mma_consumer_state: pipeline.PipelineState,
        mma_o_producer_state: pipeline.PipelineState,
    ) -> tuple[
        cute.TiledMma,
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
    ]:
        """Compute one k-tile of mma for P*V. Updates the tiled mma pv and pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param pv_params: The pv parameters
        :type pv_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param load_v_consumer_state: The load v consumer state
        :type load_v_consumer_state: pipeline.PipelineState
        :param p_mma_consumer_state: The P MMA consumer state
        :type p_mma_consumer_state: pipeline.PipelineState
        :param mma_o_producer_state: The MMA o producer state
        :type mma_o_producer_state: pipeline.PipelineState

        :return: The tiled mma pv, the load v consumer state, the P MMA consumer state, and the MMA o producer state
        :rtype: tuple[cute.TiledMma, pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        load_v_pipeline = common_params.load_v_pipeline
        accumulate_flag = tiled_mma_pv.get(tcgen05.Field.ACCUMULATE)
        mma_o_pipeline = pv_params.mma_o_pipeline

        # Leader: the single p_full wait IS the pair rendezvous — its arrive
        # count comes remotely from the non-leader after that CTA's P half
        # landed, and its tx count from the QK bulk copy into this CTA.
        self.wait_p_local_ready(pv_params, p_mma_consumer_state, arrive_self=False)

        load_v_pipeline.consumer_wait(load_v_consumer_state)
        vc_stage = load_v_consumer_state.index
        for acc_stage in range(self.iterations_pv_n):
            mma_o_pipeline.producer_acquire(mma_o_producer_state)
            tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, accumulate_flag)
            for p_stage in range(self.iterations_pv_k):
                tOtO = pv_params.tOtO_staged[None, None, None, acc_stage]
                for k_block in cutlass.range_constexpr(pv_params.tOrP.shape[2]):
                    cute.gemm(
                        tiled_mma_pv,
                        tOtO,
                        pv_params.tOrP[
                            None,
                            None,
                            k_block,
                            (p_stage, p_mma_consumer_state.index),
                        ],
                        pv_params.tOrVC[
                            None, None, k_block, ((acc_stage, p_stage), vc_stage)
                        ],
                        tOtO,
                    )
                    tiled_mma_pv.set(tcgen05.Field.ACCUMULATE, True)

            mma_o_pipeline.producer_commit(mma_o_producer_state)
            mma_o_producer_state.advance()
        load_v_pipeline.consumer_release(load_v_consumer_state)
        load_v_consumer_state.advance()
        # Release the P stage only after every PV UMMA that reads it has
        # committed. The multicast targets the two QK p_empty barriers.
        with cute.arch.elect_one():
            tcgen05.commit(
                pv_params.p_empty_mbar.get_barrier(p_mma_consumer_state.index),
                pv_params.qk_pair_commit_mask,
                tcgen05.CtaGroup.TWO,
            )
        p_mma_consumer_state.advance()

        return (
            tiled_mma_pv,
            load_v_consumer_state,
            p_mma_consumer_state,
            mma_o_producer_state,
        )

    @cute.jit
    def mma_pv_peer(
        self,
        pv_params: SimpleNamespace,
        p_mma_consumer_state: pipeline.PipelineState,
    ) -> pipeline.PipelineState:
        """Vote the nonleader's P readiness directly on the leader's p_full."""
        self.wait_p_local_ready(pv_params, p_mma_consumer_state, arrive_self=True)
        # The leader's p_full expects exactly one arrive: this remote vote.
        # It can only fire after THIS CTA's P half has landed, so the
        # leader's single p_full wait covers both halves of the pair.
        # Plain CTA-scope remote arrive (Feynman DSMEM pattern): the vote is a
        # count-only notification — this CTA's P half's visibility to the
        # 2-CTA UMMA is established by the bulk copy's complete_tx on OUR
        # p_full, not by this arrive's ordering. The earlier CLUSTER/release
        # variant cost a MEMBAR.ALL.CTA+MEMBAR.ALL.GPU pair per tile.
        with cute.arch.elect_one():
            _arrive_cluster_mbarrier(
                pv_params.p_full_mbar.get_barrier(p_mma_consumer_state.index),
                pv_params.pv_pair_leader_rank,
            )
        p_mma_consumer_state.advance()
        return p_mma_consumer_state

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        is_second_compute_warp: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState

        :return: The MMA s consumer state, the P MMA producer state, and the P correction producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        # 2softmax: the running (row_max, row_sum) is inherited from the peer
        # group every tile; the seed below provides the online-softmax identity
        # for the first tile.
        row_max = self.acc_dtype(self.init_row_max)
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)
        odd_k_tile = k_tile_count % 2 == 1
        # lazySum: the split's serially-final tile belongs to g0 iff the
        # ORIGINAL tile count is odd (g0 takes even ordinals + the extra).
        is_serial_last = odd_k_tile
        if is_second_compute_warp:
            is_serial_last = k_tile_count % 2 == 0
        else:
            is_serial_last = odd_k_tile
        # g0 takes even k-tiles, g1 odd k-tiles (g0 gets the extra tile if odd).
        if is_second_compute_warp:
            k_index = k_index + 1
            k_tile_count = k_tile_count // 2
        else:
            k_tile_count = (k_tile_count + 1) // 2
        valid_k_tile_count = k_tile_count > 0

        # Both softmax groups must enter the same work item before g1 seeds and
        # arrives on order_bar_0. Without this boundary, a faster group can
        # re-enter the named barrier for the next persistent work item before
        # its peer has retired the previous phase.
        self.softmax_warps_initial_sync_bar.arrive_and_wait()

        if is_second_compute_warp:
            # Seed g1's home group-exchange slot so g0's FIRST peer-read gets
            # the online-softmax identity, then pre-arrive order bar 0 so g0's
            # first arrive_and_wait falls through (init-phase trick).
            tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
            # lazySum: seed the merge-exchange slot so a zero-tile g1
            # (k_tile_count==1) still yields a valid peer partial (0) for
            # g0's end-of-split merge. Ordered before the pre-arrive below.
            common_params.smem_exchange[tidx] = self.acc_dtype(0.0)
            self._store_group_mbox(
                common_params,
                softmax_params,
                self.acc_dtype(0.0),
                self.acc_dtype(self.init_row_max),
                p_cor_producer_state.index % 2,
                tidx,
            )
            self.softmax_order_bar_0.arrive()

            def _zero_tile_rendezvous() -> None:
                # critShort: g1 has no tiles (original count == 1) so it never
                # reaches a last-tile publish; stand in for its id3 arrive so
                # g0's end-merge wait completes (the zero-seeded exchange slot
                # above is ordered before this arrive).
                self.epilogue_exchange_sync_bar.arrive()

            if_generate(k_tile_count == 0, _zero_tile_rendezvous)

        # Earliest causal boundary over the original (unfolded) query tokens.
        # Group-final is independent: a split can finish before the masked region.
        first_mask_tile_idx = (
            common_params.K - self.causal_seq_len_q + 1
        ) // self.mma_qk_tiler[1]
        # Auto FP16 keeps the group-final tile in the existing masked tail.
        # Its mask is also valid for a complete tile (all predicates pass),
        # so the hot full-tile body can specialize away local-final handling
        # without adding another softmax body or dropping any causal checks.
        static_nonfinal = self.force_branch == "auto" and self.use_fp16_softmax
        full_tile_reserve = 1 if cutlass.const_expr(static_nonfinal) else 0
        while k_tile_count > full_tile_reserve and k_index < first_mask_tile_idx:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                is_second_compute_warp,
                False,
                False if cutlass.const_expr(static_nonfinal) else k_tile_count == 1,
                is_serial_last,
            )
            k_index = k_index + 2
            k_tile_count = k_tile_count - 1
            # Skip the peer group's stage on every pingponged resource
            # (stage index tracks the global k-tile position mod num_stages).
            if k_tile_count > 0:
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()

        # At most two global K tiles are causal-masked when Q <= tile_N,
        # including a possible partial final K tile.
        # Each ping-pong group then has at most one remaining tile, including
        # the reserved all-valid final tile. Larger Q retains dynamic final
        # handling; no masked tile or split-final exchange is skipped.
        static_final = static_nonfinal and self.causal_seq_len_q <= self.mma_qk_tiler[1]
        # Masked region uses a compile-time mask flag, not a predicate over S.
        while k_tile_count > 0:
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                is_second_compute_warp,
                True,
                True if cutlass.const_expr(static_final) else k_tile_count == 1,
                is_serial_last,
            )
            k_index = k_index + 2
            k_tile_count = k_tile_count - 1
            if k_tile_count > 0:
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()

        # Tail parity fixups: align each group's stage parity with the serial
        # stage sequence of the NEXT work item (odd tile counts swap the
        # stage↔group mapping; see the baseline kernel for the derivation).
        if odd_k_tile and valid_k_tile_count:
            if is_second_compute_warp:
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
        else:
            p_mma_producer_state.advance()
            mma_s_consumer_state.advance()
            p_cor_producer_state.advance()
        # Re-align the pingpong order-bar parity across work items.
        if is_second_compute_warp:
            if odd_k_tile:
                self.softmax_order_bar_1.arrive_and_wait()
        else:
            if not odd_k_tile:
                self.softmax_order_bar_0.arrive_and_wait()

        return mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state

    @cute.jit
    def correction(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        k_tile_count: cutlass.Int32,
        p_cor_consumer_state: pipeline.PipelineState,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_group_idx: int,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState
        :param mma_o_consumer_state: The MMA o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_group_idx: The PV correction warpgroup index (0 or 1)
        :type correction_group_idx: int

        :return: The P correction consumer state, and the MMA o consumer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_count_init = k_tile_count
        prev_row_max = self.acc_dtype(self.init_row_max)
        while k_tile_count > 0:
            (
                p_cor_consumer_state,
                row_sum,
                row_max,
                correction_factor,
                no_correction,
            ) = self.get_correction_factor(
                common_params, p_cor_consumer_state, prev_row_max
            )
            prev_row_max = row_max
            if k_tile_count_init != k_tile_count:
                mma_o_consumer_state = self.rescale(
                    common_params,
                    mma_o_consumer_state,
                    correction_factor,
                    no_correction,
                    correction_group_idx,
                )
            k_tile_count = k_tile_count - 1
            if k_tile_count == 0:
                # Final metadata/rescale finished; PV may still be pending.
                mma_o_consumer_state = self.epilogue(
                    common_params,
                    epilogue_params,
                    mma_o_consumer_state,
                    row_sum,
                    row_max,
                    correction_group_idx,
                )
        return p_cor_consumer_state, mma_o_consumer_state

    @cute.jit
    def exchange_p_cor_metadata(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        correction_factor: cutlass.Float32,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        row_max_new: cutlass.Float32,
        tAcc: cute.Tensor,
        tidx: cutlass.Int32,
        p_cor_producer_state: pipeline.PipelineState,
    ) -> tuple[pipeline.PipelineState, cutlass.Float32]:
        """Send the per-row correction metadata over DSMEM to the paired PV CTA.

        Each of the 128 softmax threads owns one row and sends 2 b32 words
        (row_sum, row_max) as a single st.async.v2 to its slot in the PV CTA's
        corr SMEM stage; the transfer is tracked on the PV CTA's cor_full
        mbarrier (1KB tx per stage). correction_factor / no_correction are NOT
        sent: the PV side recomputes them from consecutive row_max values.
        """
        # critShort: the no_correction decision + row_max_new rollback happen
        # in softmax() BEFORE the mailbox publish; row_max_new arrives here
        # already rolled back (single source of truth).

        rCor = cute.make_rmem_tensor(
            cute.make_layout(self.corr_words_per_row), self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        rCor[0] = row_sum
        rCor[1] = row_max_new

        # Wait acquire.cluster for the PV CTA's release.cluster arrival on this
        # local cor_empty barrier. The producer state starts with phase=1 so the
        # first p_cor_stage acquires fall through.
        _wait_cluster_mbarrier(
            softmax_params.cor_empty_mbar.get_barrier(p_cor_producer_state.index),
            p_cor_producer_state.phase,
        )

        # Per-thread slot pointer inside the (local-union) PV corr buffer; the
        # STAS wrapper maps both data and mbar pointers to the peer rank.
        corr_word_ptr = cute.recast_ptr(
            softmax_params.smem_corr.iterator, dtype=cutlass.Int32
        )
        slot_words = (
            p_cor_producer_state.index * self.corr_rows + tidx
        ) * self.corr_words_per_row
        slot_ptr = (corr_word_ptr + slot_words).align(min_align=8)
        cute.arch.store_async_dsmem(
            slot_ptr,
            (rCor_int[0], rCor_int[1]),
            softmax_params.cor_full_mbar.get_barrier(p_cor_producer_state.index),
            softmax_params.dsmem_data_peer,
        )
        p_cor_producer_state.advance()
        return (p_cor_producer_state, row_max_new)

    @cute.jit
    def _group_mbox_tensor(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        stage_idx: cutlass.Int32,
    ) -> cute.Tensor:
        """(rows, 4, 1) TMEM mailbox stage holding (row_sum, row_max, 0, 0).

        Cross-group handoff storage for the two softmax warp groups. Built as
        a single-stage tensor at base + stage_idx * stage_cols: the MTP
        kernel's verified corr-slot partition shape is (rows, 4, 1); a
        stage>1 mode in the partitioned tensor scrambles slot addressing.
        """
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_layout = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        ).layout
        tStS = cute.make_tensor(common_params.tmem_ptr, tStS_layout)
        tAcc = tStS[(None, None), 0, 0, 0]
        mbox_layout = cute.make_layout(
            (tAcc.shape[0], 4, 1),
            stride=(tAcc.stride[0], 1, self.tmem_group_mbox_stage_cols),
        )
        return cute.make_tensor(
            common_params.tmem_ptr
            + self.tmem_group_mbox_offset
            + stage_idx * self.tmem_group_mbox_stage_cols,
            mbox_layout,
        )

    @cute.jit
    def _store_group_mbox(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        stage_idx: cutlass.Int32,
        tidx: cutlass.Int32,
    ) -> None:
        """STTM this group's (row_sum, row_max) into its mailbox stage."""
        tMbox = self._group_mbox_tensor(common_params, softmax_params, stage_idx)
        cMbox = cute.make_identity_tensor(tMbox.shape)
        store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        tiled_copy = tcgen05.make_tmem_copy(store_atom, tMbox)
        thr_copy = tiled_copy.get_slice(tidx)
        cMbox_part = thr_copy.partition_S(cMbox)
        tMbox_part = thr_copy.partition_D(tMbox)
        rMbox = cute.make_fragment_like(cMbox_part[None, None, None, 0], self.acc_dtype)
        rMbox[0] = row_sum
        rMbox[1] = row_max
        rMbox[2] = self.acc_dtype(0.0)
        rMbox[3] = self.acc_dtype(0.0)
        cute.copy(tiled_copy, rMbox, tMbox_part[None, None, None, 0])
        # Order the TMEM store before the pingpong order-bar arrive so the
        # peer group's LDTM observes it.
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def _load_group_mbox(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        stage_idx: cutlass.Int32,
        tidx: cutlass.Int32,
    ) -> tuple[cutlass.Float32, cutlass.Float32]:
        """LDTM the peer group's (row_max, row_sum) from its mailbox stage."""
        tMbox = self._group_mbox_tensor(common_params, softmax_params, stage_idx)
        cMbox = cute.make_identity_tensor(tMbox.shape)
        load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(4)), self.acc_dtype
        )
        tiled_copy = tcgen05.make_tmem_copy(load_atom, tMbox)
        thr_copy = tiled_copy.get_slice(tidx)
        tMbox_part = thr_copy.partition_S(tMbox)
        cMbox_part = thr_copy.partition_D(cMbox)
        rMbox = cute.make_fragment_like(cMbox_part[None, None, None, 0], self.acc_dtype)
        cute.copy(tiled_copy, tMbox_part[None, None, None, 0], rMbox)
        cute.arch.fence_view_async_tmem_load()
        return rMbox[1], rMbox[0]

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_second_compute_warp: bool,
        is_last_tile: bool,
        is_local_last_tile: cutlass.Boolean,
        is_serial_last: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Softmax for one k-tile. Updates the related pipeline states and returns the computed results.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param is_last_tile: Whether the last tile
        :type is_last_tile: bool
        :param is_local_last_tile: Whether the last tile is local
        :type is_local_last_tile: cutlass.Boolean

        :return: The MMA s consumer state, the P MMA producer state, the P correction producer state, the row max, the row sum, and the correction factor
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32]
        """

        # Keep the S wait early, but defer the P producer acquire until right
        # before STTM P so rmem-only softmax work can overlap with the prior
        # tile's PV consumption of the P stage.
        mma_s_pipeline = softmax_params.mma_s_pipeline
        peek_mma_s_full = mma_s_pipeline.consumer_try_wait(mma_s_consumer_state)
        mma_s_pipeline.consumer_wait(mma_s_consumer_state, peek_mma_s_full)

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cta_pair_size,
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        # Retain the complete score fragment returned by the reducing load;
        # no second TMEM read is needed for exp after the row-max merge.
        _s_load_rep = 64 if self.force_branch == "auto" else 128
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(_s_load_rep)),
            self.acc_dtype,
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(arch >= Arch.sm_100 and arch <= Arch.sm_100f):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                if is_last_tile:
                    q_tok = (
                        common_params.blk_coord[2] * self.causal_fold_ratio
                        + (common_params.blk_coord[0] * cta_qk_tiler[0] + tTR_tS[i][0])
                        // self.causal_num_heads
                    )
                    k_bound = common_params.K - self.causal_seq_len_q + 1 + q_tok
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else -self.acc_dtype.inf
                    )
            # reduction for row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
        elif cutlass.const_expr(
            (arch >= Arch.sm_103 and arch <= Arch.sm_103f)
            or (arch >= Arch.sm_107 and arch <= Arch.sm_107f)
        ):
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(_s_load_rep),
                    redOp=tcgen05.TmemLoadRedOp.MAX,
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )

            cute.arch.fence_view_async_tmem_load()
            softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)
            mma_s_consumer_state.advance()

            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if is_last_tile:
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    q_tok = (
                        common_params.blk_coord[2] * self.causal_fold_ratio
                        + (common_params.blk_coord[0] * cta_qk_tiler[0] + tTR_tS[i][0])
                        // self.causal_num_heads
                    )
                    k_bound = common_params.K - self.causal_seq_len_q + 1 + q_tok
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else -self.acc_dtype.inf
                    )
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    row_max_new = cute.arch.fmax(row_max_new, tTR_rAcc[i], nan=True)
            else:
                for _mi in cutlass.range_constexpr(cute.size(tTR_rMax)):
                    row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[_mi])

        # === 2softmax cross-group merge (pingpong) ===
        # Wait for the peer group BEFORE the mailbox LDTM below: the peer's
        # mailbox STTM + tmem store fence is ordered before its order-bar
        # arrive, so this wait provides the acquire for the peer-read.
        # (tuneOrderBarrier had moved this wait after the read - that races
        # the peer's store and fails AModel ref-check; restored here.)
        if cutlass.const_expr(self.force_branch == "auto"):
            # Select the same pingpong resource without splitting the shared body.
            order_group = cute.arch.make_warp_uniform(
                common_params.tidx // (self.num_compute_warps * self.threads_per_warp)
            )
            cute.arch.barrier(
                barrier_id=5 + order_group,
                number_of_threads=self.threads_per_warp * self.num_total_compute_warps,
            )
        elif is_second_compute_warp:
            self.softmax_order_bar_1.arrive_and_wait()
        else:
            self.softmax_order_bar_0.arrive_and_wait()
        # Serial inheritance: the peer's previous tile IS the serially previous
        # tile, so its (row_max, row_sum) is the global running state entering
        # this tile.
        # The 2-slot TMEM group mailbox tracks tile parity; with p_cor_stage=4
        # the p_cor index runs 0..3, so the mailbox slot is its mod-2.
        saved_p_cor_idx = p_cor_producer_state.index % 2
        peer_slot = (saved_p_cor_idx + 1) % 2
        # lazySum: my_prev_max = this GROUP's global-max view at ITS previous
        # tile (loop-carried row_max). The partial row_sum is rescaled by
        # exp2(my_prev_max - row_max_new) instead of the serial tile-to-tile
        # factor; peer_row_sum is no longer inherited (mailbox row_sum word is
        # dead, layout kept for the verified Rep(4) mbox machinery).
        my_prev_max = row_max
        peer_row_max, peer_row_sum = self._load_group_mbox(
            common_params, softmax_params, peer_slot, tidx
        )
        row_max_new = cute.arch.fmax(row_max_new, peer_row_max)
        row_max = peer_row_max
        # (lazySum) row_sum stays group-local; peer_row_sum intentionally unused.

        # critShort: no_correction decision + row_max_new rollback HOISTED
        # above the mailbox publish (the published max must be the
        # post-rollback value). Single source of truth: the exchange helper
        # no longer recomputes/rolls back.
        no_correction = cutlass.Int32(0)
        if (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
            no_correction = cutlass.Int32(1)
            row_max_new = row_max

        # critShort: publish this tile's row_max and release the peer group
        # IMMEDIATELY after the merge/rollback - the serial critical section
        # ends here. Everything below (cf, cor exchange, MUFU exp2, F2FP,
        # row_sum reduction, P staging/send) overlaps the peer's critical
        # section instead of extending the serial chain.
        self._store_group_mbox(
            common_params,
            softmax_params,
            row_sum,
            row_max_new,
            saved_p_cor_idx,
            tidx,
        )
        if cutlass.const_expr(self.force_branch == "auto"):
            # Peer release uses the complementary resource with the same count.
            cute.arch.barrier_arrive(
                barrier_id=6 - order_group,
                number_of_threads=self.threads_per_warp * self.num_total_compute_warps,
                aligned=False,
            )
        elif is_second_compute_warp:
            self.softmax_order_bar_0.arrive()
        else:
            self.softmax_order_bar_1.arrive()

        # find correction factor (row_max = inherited previous global max).
        # nc=1 -> row_max_new==row_max -> cf==1.0; cf is not sent (2-word cor
        # protocol, PV recomputes its own) and lazySum rescales by cf_local,
        # so the value is chain-vestigial either way.
        correction_factor = _rescale_factor(
            row_max, row_max_new, softmax_params.softmax_scale_log2
        )
        # split kv case
        if not is_local_last_tile:
            (p_cor_producer_state, row_max_new) = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        if cutlass.const_expr(self.use_fp16_softmax):
            # FP16 softmax path (restored, mirroring the 2p2/ublkcp donor):
            # fused f32x4 -> f16x2x2 -> fma -> exp2 -> e4m3x4 pipeline
            n_f16_elems = cute.size(tTR_rAcc)
            n_packed = n_f16_elems // 2  # pairs of f16x2
            n_quads = n_packed // 2  # quads of fp8 (4 fp8 per Uint32)

            fma_b = cutlass.Float16(softmax_params.softmax_scale_log2)
            fma_c = cutlass.Float16(0.0 - row_max_new) * fma_b
            packed_b = pack_f16x2(fma_b, fma_b)
            packed_c = pack_f16x2(fma_c, fma_c)

            tTR_rAcc_f16x2 = cute.make_rmem_tensor(
                cute.make_layout(n_packed), cutlass.Uint32
            )
            # Uint32 fragment: each holds 4 packed fp8, directly STTM-ready
            tTR_p_fp8x4 = cute.make_rmem_tensor(
                cute.make_layout(n_quads), cutlass.Uint32
            )
            for i in cutlass.range_constexpr(n_quads):
                tTR_rAcc_f16x2[i * 2], tTR_rAcc_f16x2[i * 2 + 1], tTR_p_fp8x4[i] = (
                    softmax_f32x4_to_f16x2x2_and_e4m3x4(
                        tTR_rAcc[i * 4],
                        tTR_rAcc[i * 4 + 1],
                        tTR_rAcc[i * 4 + 2],
                        tTR_rAcc[i * 4 + 3],
                        packed_b,
                        packed_c,
                    )
                )
            # Reinterpret Uint32 fp8x4 as fp8 fragment for the P staging path
            tTR_p_fp8 = cute.make_tensor(
                cute.recast_ptr(tTR_p_fp8x4.iterator, dtype=self.q_dtype),
                cute.make_layout(n_f16_elems),
            )
        else:
            # FP32 softmax path
            fma_b = softmax_params.softmax_scale_log2
            fma_c = (
                0.0 - row_max_new if row_max_new != -self.acc_dtype.inf else 0.0
            ) * softmax_params.softmax_scale_log2

            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=True, unroll_full=True
            ):
                tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
                tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

            tTR_p_fp8 = cute.make_fragment_like(tTR_tS, self.q_dtype)
            tTR_p_fp8.store(tTR_rAcc.load().to(self.q_dtype))

        # row_sum is rmem-only; run it before the deferred P acquire.
        # lazySum: rescale the GROUP-LOCAL partial by this group's own max
        # delta (not the serial correction_factor, which stays global for the
        # PV cor protocol). First tile: my_prev_max=-inf -> cf_local=0, and
        # partial is 0, so the product is harmlessly 0.
        cf_local = _rescale_factor(
            my_prev_max, row_max_new, softmax_params.softmax_scale_log2
        )
        row_sum = row_sum * cf_local
        if cutlass.const_expr(self.use_fp16_softmax):
            # FP16 path: tree-reduce in packed f16x2, then convert to f32
            row_sum_f16x2 = tTR_rAcc_f16x2[0]
            for i in cutlass.range_constexpr(1, n_packed):
                row_sum_f16x2 = add_packed_f16x2_u32(row_sum_f16x2, tTR_rAcc_f16x2[i])
            row_sum = row_sum + reduce_sum_packed_f16x2_to_f32(row_sum_f16x2)
        else:
            # FP32 row_sum: `add_packed_f32x2` reduces the instruction count
            row_sum_vec = (0.0, 0.0)
            for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
                row_sum_vec = cute.arch.add_packed_f32x2(
                    row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
                )
            row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        # lazySum end-of-split merge (group-final tiles only):
        #  - the NON-serial-last group publishes its raw partial into the
        #    (otherwise unused) softmax_smem_exchange slot for its rows; its
        #    mailbox store + order-bar arrive below order the write before the
        #    serial-last group's read (bar.sync memory semantics).
        #  - the SERIAL-last group (owner of the split's final tile) folds the
        #    peer partial in, correcting it from the peer's final max view
        #    (= peer_row_max just read from the mailbox this tile) to the
        #    final max. exp rescaling telescopes, so this equals the serial
        #    chain exactly.
        if is_local_last_tile:
            _exch = common_params.smem_exchange
            _scale_log2 = softmax_params.softmax_scale_log2
            _res = cute.make_rmem_tensor(cute.make_layout(1), self.acc_dtype)
            _res[0] = row_sum

            def _lazy_merge() -> None:
                # critShort: acquire via the dedicated end-of-split
                # rendezvous bar (id3, 256 thr = both groups). The peer's
                # raw-partial store is ordered before its arrive; the old
                # ordering backer (tail mbox store + order-bar arrive) no
                # longer exists after the hoist.
                self.epilogue_exchange_sync_bar.arrive_and_wait()
                _res[0] = row_sum + _exch[tidx] * _rescale_factor(
                    peer_row_max, row_max_new, _scale_log2
                )

            def _lazy_publish() -> None:
                _exch[tidx] = row_sum
                # critShort: non-blocking release; this group's P tail
                # proceeds while the serial-last group merges.
                self.epilogue_exchange_sync_bar.arrive()

            if_generate(is_serial_last, _lazy_merge, _lazy_publish)
            row_sum = _res[0]

        # Split-kv correction metadata for the LAST tile: issued right after
        # the row_sum reduction (the reduction kernel consumes this row_sum as
        # the split's final sum, so it cannot be sent any earlier), but before
        # the mailbox/P staging/send so the PV pair's O rescale overlaps them.
        # lazySum: the non-serial-last group's exchange carries its PARTIAL;
        # harmless - the PV epilogue only consumes the FINAL cor stage's
        # row_sum (mid-stream row_sums are dead, see correction()).
        if is_local_last_tile:
            (p_cor_producer_state, row_max_new) = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # critShort: mailbox publish + order-bar release were hoisted to right
        # after the merge (see above); nothing to publish here.

        # UBLKCP 2+2: stage P locally, then push the whole 16KB stage to the
        # paired PV CTA with a single cp.async.bulk (SASS: UBLKCP) posting one
        # transaction on the peer's p_full — replacing the 1024 x 16B
        # st.async micro-transactions of the STAS scheme.
        # Build the per-stage (M,K) PISL-swizzled view over the LOCAL staging
        # buffer (byte-identical layout to the PV pair's smem_p).
        sP = softmax_params.sP_src[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=128,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        trP = cute.make_fragment_like(tTR_tS, self.q_dtype)
        tSTS_rP = cute.make_tensor(tTR_p_fp8.iterator, trP.layout)
        rP_copy_view = smem_thr_copy.retile(tSTS_rP)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)

        # P acquire BEFORE touching the staging buffer: the previous bulk
        # copy of this group's stage has fully read the staging only once the
        # PV pair consumed it (p_empty released via tcgen05.commit multicast).
        softmax_params.p_empty_mbar.wait(
            p_mma_producer_state.index, p_mma_producer_state.phase
        )
        # Guide-optimal STS -> bulk sequence (memory-model guide §5):
        # STS -> fence.proxy.async.shared::cta -> bar.sync -> elect issue.
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)
        cute.arch.fence_view_async_shared()
        if cutlass.const_expr(self.force_branch == "auto"):
            # Keep each group's staging fence on its own 128-thread resource.
            cute.arch.barrier(
                barrier_id=9 + order_group,
                number_of_threads=self.threads_per_warp * self.num_compute_warps,
            )
        elif is_second_compute_warp:
            self.p_src_write_bar_1.arrive_and_wait()
        else:
            self.p_src_write_bar_0.arrive_and_wait()
        # One warp ships the whole stage via the official bulk S2S atom (the
        # atom lowering elect_syncs internally). tx (16KB) lands on the PEER
        # PV CTA's p_full — same-CTA data/mbar pairing per the ISA.
        _p_stage_off = p_mma_producer_state.index * self.p_tx_bytes_per_cta
        _p_flat_layout = cute.make_layout(self.p_tx_bytes_per_cta)
        _p_src_flat = cute.make_tensor(
            (softmax_params.smem_p_src_ptr + _p_stage_off).align(min_align=16),
            _p_flat_layout,
        )
        _p_dst_flat = cute.make_tensor(
            (softmax_params.smem_p_dst_ptr + _p_stage_off).align(min_align=16),
            _p_flat_layout,
        )
        _bulk_s2s_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyBulkS2SOp(), self.q_dtype
        )

        # Keep address preparation outside the issuing-warp condition so
        # that the conditional region contains only the bulk copy.
        # Preserve the cluster-encoded barrier address used by AModel.
        _p_full_local = softmax_params.p_full_mbar.get_barrier(
            p_mma_producer_state.index
        )
        _p_full_peer_s3 = cute.make_ptr(
            cutlass.Int64,
            cute.arch.map_dsmem_ptr(
                _p_full_local, softmax_params.dsmem_data_peer
            ).toint(),
            cute.AddressSpace.smem,
            assumed_align=8,
        )

        def _send_p_bulk() -> None:
            with cute.arch.elect_one():
                cute.copy(
                    _bulk_s2s_atom,
                    _p_src_flat,
                    _p_dst_flat,
                    mbar_ptr=_p_full_peer_s3,
                    cta_rank=softmax_params.dsmem_data_peer,
                )

        # Select the first warp of each four-warp softmax group (W0 or W4).
        warp_idx = tidx // self.threads_per_warp
        if_generate(warp_idx == 0, _send_p_bulk)
        p_mma_producer_state.advance()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            p_cor_producer_state,
            row_max_new,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def _s2t_copy_and_partition(
        self,
        sQ: cute.Tensor,
        tQ: cute.Tensor,
    ):
        """Setup S2T (UTCCP) copy for SMEM→TMEM Q transfer (gr150 pattern)."""
        tCsQ_compact = cute.filter_zeros(sQ)
        tCtQ_compact_staged = cute.filter_zeros(tQ)
        tCtQ_compact = cute.filter_zeros(tQ[None, None, None, 0])

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp128x128bOp(tcgen05.CtaGroup.TWO),
            self.q_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtQ_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsQ_s2t_ = thr_copy_s2t.partition_S(tCsQ_compact)
        tCsQ_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsQ_s2t_)
        tCtQ_s2t = thr_copy_s2t.partition_D(tCtQ_compact_staged)

        return tiled_copy_s2t, tCsQ_s2t, tCtQ_s2t

    @cute.jit
    def _tmem_load_partition(
        self,
        common_params: SimpleNamespace,
        tiled_mma_pv: cute.TiledMma,
        iter_n: int,
        subtile_n: cutlass.Constexpr[int] = 0,
    ) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
        """Tensor memory load partition for rescale and epilogue.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param tiled_mma_pv: The tiled mma pv
        :type tiled_mma_pv: cute.TiledMma
        :param iter_n: The iteration number
        :type iter_n: int

        :return: The tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv, the tiled mma pv
        :rtype: tuple[cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma, cute.TiledMma]
        """

        if cutlass.const_expr(subtile_n == 0):
            subtile_n = self.correction_subtile_n
        n_subtiles = self.mma_pv_tiler[1] // subtile_n
        tOtO_shape = tiled_mma_pv.partition_shape_C(
            cute.select(self.mma_pv_tiler, mode=[0, 1])
        )
        tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
        tOtO_layout = cute.append(
            tOtO.layout,
            cute.make_layout(
                common_params.L // self.mma_pv_tiler[1],
                stride=self.mma_pv_tiler[1],
            ),
        )
        tOtO = cute.make_tensor(
            common_params.tmem_ptr + self.tmem_o_offset, tOtO_layout
        )
        tOtO = tOtO[None, None, None, iter_n // n_subtiles]

        # flat_divide by subtile to split N dimension
        epi_tile = (
            self.mma_pv_tiler[0] // self.cta_pair_size,
            subtile_n,
        )
        tAcc_full = tOtO[(None, None), 0, 0]
        tAcc_epi = cute.flat_divide(tAcc_full, epi_tile)
        sub_n = iter_n % n_subtiles
        tAcc = tAcc_epi[(None, None, 0, sub_n)]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(subtile_n)),
            self.acc_dtype,
        )
        tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )

        tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
        tTR_gO_layout_ref, _ = self._gmem_output_partition(
            common_params, tmem_load_tiled_copy, iter_n, subtile_n
        )
        tTR_rAcc = cute.make_fragment_like(tTR_gO_layout_ref, self.acc_dtype)
        return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_rAcc

    @cute.jit
    def _gmem_output_partition(
        self,
        common_params: SimpleNamespace,
        tmem_load_tiled_copy: cute.TiledCopy,
        iter_n: int,
        subtile_n: cutlass.Constexpr[int] = 0,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Build output partitions immediately before the global store."""

        if cutlass.const_expr(subtile_n == 0):
            subtile_n = self.correction_subtile_n
        n_subtiles = self.mma_pv_tiler[1] // subtile_n
        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cta_pair_size,
            subtile_n,
            self.mma_pv_tiler[2],
        )
        # Flatten divide and partition global tensors for O
        cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

        gmem_iter_n = (
            iter_n + common_params.pv_n_split_idx * self.iterations_pv_n * n_subtiles
        )
        gO = None
        if cutlass.const_expr(common_params.mAccO is not None):
            gO = cute.local_tile(
                common_params.mAccO[None, common_params.blk_coord[4], None, None, None],
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    gmem_iter_n,
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(
                    common_params.mAccO[
                        None, common_params.blk_coord[4], None, None, None
                    ].shape
                ),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    gmem_iter_n,
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
            )
        else:
            gO = cute.local_tile(
                common_params.mO,
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    gmem_iter_n,
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
            )
            cO = cute.local_tile(
                cute.make_identity_tensor(common_params.mO.shape),
                cta_pv_tiler_mn,
                (
                    common_params.blk_coord[0],
                    gmem_iter_n,
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
            )
        tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
            common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        )
        return (
            tmem_load_thr_copy.partition_D(gO),
            tmem_load_thr_copy.partition_D(cO),
        )

    @cute.jit
    def get_correction_factor(
        self,
        common_params: SimpleNamespace,
        p_cor_consumer_state: pipeline.PipelineState,
        prev_row_max: cutlass.Float32,
    ) -> tuple[
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Int32,
    ]:
        """Get the correction factor from the P correction consumer state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param p_cor_consumer_state: The P correction consumer state
        :type p_cor_consumer_state: pipeline.PipelineState

        :return: The P correction consumer state, the row_sum, the row_max, and the correction factor
        :rtype: tuple[pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32, cutlass.Int32]
        """
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        # elect_one() is warp-scoped: restrict single-arrive/re-arm operations to
        # one of the four correction warps or the barrier over-arrives.
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        is_sync_warp = warp_idx == self.correction_warp_ids[0]

        # Wait for the paired QK CTA's corr STAS transaction: one elected arrive
        # plus the expected tx bytes complete the phase; all threads wait.
        def _arrive_cor_full() -> None:
            with cute.arch.elect_one():
                common_params.cor_full_mbar.arrive_mbarrier(p_cor_consumer_state.index)

        if_generate(is_sync_warp, _arrive_cor_full)
        common_params.cor_full_mbar.wait(
            p_cor_consumer_state.index, p_cor_consumer_state.phase
        )

        # Each thread reads its own row's metadata from the corr SMEM stage.
        rCor = cute.make_rmem_tensor(
            cute.make_layout(self.corr_words_per_row), self.acc_dtype
        )
        rCor_int = cute.make_tensor(
            cute.recast_ptr(rCor.iterator, dtype=cutlass.Int32), rCor.layout
        )
        for w in range(self.corr_words_per_row):
            rCor[w] = common_params.smem_corr[tidx, w, p_cor_consumer_state.index]
        row_sum = rCor[0]
        row_max = rCor[1]
        # Recompute correction_factor / no_correction locally from the
        # received row_max stream. The sender rolls row_max back on
        # no-correction tiles, so row_max == prev_row_max there and both the
        # flag ((diff <= threshold) with diff == 0) and the factor (exp2(0),
        # unused when skipped) reproduce the old sent values exactly. The
        # first tile's factor (prev = -inf) is never consumed (the correction
        # loop skips rescale on its first iteration).
        correction_factor = _rescale_factor_min_before_exp(
            prev_row_max, row_max, common_params.softmax_scale_log2
        )
        no_correction = cutlass.Int32(0)
        if (
            row_max - prev_row_max
        ) * common_params.softmax_scale_log2 <= self.skip_correction_threshold:
            no_correction = cutlass.Int32(1)

        # Release: one elected thread re-arms this CTA's cor_full transaction
        # count first, then every thread remote-arrives the QK peer's cor_empty
        # barrier with release.cluster ordering after its read (arrive count =
        # 256 threads). The producer can only unblock after all arrives, which
        # orders it after the re-arm.
        def _rearm_cor_full() -> None:
            with cute.arch.elect_one():
                cute.arch.mbarrier_expect_tx(
                    common_params.cor_full_mbar.get_barrier(p_cor_consumer_state.index),
                    self.cor_tx_bytes,
                )

        if_generate(is_sync_warp, _rearm_cor_full)
        _arrive_cluster_mbarrier(
            common_params.cor_empty_mbar.get_barrier(p_cor_consumer_state.index),
            common_params.qk_peer_rank,
        )
        p_cor_consumer_state.advance()
        return (
            p_cor_consumer_state,
            row_sum,
            row_max,
            correction_factor,
            no_correction,
        )

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
        correction_group_idx: int,
    ) -> pipeline.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param no_correction: Whether to apply correction factor
        :type no_correction: cutlass.Int32
        :param correction_group_idx: The PV correction warpgroup index (0 or 1)
        :type correction_group_idx: int

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        # Auto-only latency-overlap experiment: load four disjoint N16 fragments
        # before consuming any. Native-only modes keep their original order.
        rescale_step = 4 if cutlass.const_expr(self.force_branch == "auto") else 1
        assert (
            self.correction_n_subtiles % (self.num_correction_groups * rescale_step)
            == 0
        ), "Rescale preloads must cover complete correction-group batches"
        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
            if not skip_correction:
                for group_subtile in cutlass.range_constexpr(
                    0,
                    self.correction_n_subtiles // self.num_correction_groups,
                    rescale_step,
                ):
                    sub_n = (
                        correction_group_idx
                        + group_subtile * self.num_correction_groups
                    )
                    actual_n = iter_n * self.correction_n_subtiles + sub_n
                    tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_rAcc = (
                        self._tmem_load_partition(
                            common_params, common_params.tiled_mma_pv, actual_n
                        )
                    )

                    tmem_store_atom = cute.make_copy_atom(
                        tcgen05.copy.St32x32bOp(
                            tcgen05.copy.Repetition(self.correction_subtile_n)
                        ),
                        self.acc_dtype,
                    )
                    tmem_store_tiled_copy = tcgen05.make_tmem_copy(
                        tmem_store_atom, tAcc
                    )

                    if cutlass.const_expr(self.force_branch == "auto"):
                        next_n = actual_n + self.num_correction_groups
                        next_load, next_acc, next_src, next_regs = (
                            self._tmem_load_partition(
                                common_params, common_params.tiled_mma_pv, next_n
                            )
                        )
                        next_store = tcgen05.make_tmem_copy(tmem_store_atom, next_acc)
                        third_load, third_acc, third_src, third_regs = (
                            self._tmem_load_partition(
                                common_params,
                                common_params.tiled_mma_pv,
                                actual_n + 2 * self.num_correction_groups,
                            )
                        )
                        third_store = tcgen05.make_tmem_copy(tmem_store_atom, third_acc)
                        fourth_load, fourth_acc, fourth_src, fourth_regs = (
                            self._tmem_load_partition(
                                common_params,
                                common_params.tiled_mma_pv,
                                actual_n + 3 * self.num_correction_groups,
                            )
                        )
                        fourth_store = tcgen05.make_tmem_copy(
                            tmem_store_atom, fourth_acc
                        )
                    cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
                    if cutlass.const_expr(self.force_branch == "auto"):
                        cute.copy(next_load, next_src, next_regs)
                        cute.copy(third_load, third_src, third_regs)
                        cute.copy(fourth_load, fourth_src, fourth_regs)
                    for i in cutlass.range(
                        cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                    ):
                        tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

                    cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)
                    if cutlass.const_expr(self.force_branch == "auto"):
                        for i in cutlass.range(
                            cute.size(next_regs), vectorize=True, unroll_full=True
                        ):
                            next_regs[i] = next_regs[i] * correction_factor
                        cute.copy(next_store, next_regs, next_src)
                        for i in cutlass.range(
                            cute.size(third_regs), vectorize=True, unroll_full=True
                        ):
                            third_regs[i] = third_regs[i] * correction_factor
                        cute.copy(third_store, third_regs, third_src)
                        for i in cutlass.range(
                            cute.size(fourth_regs), vectorize=True, unroll_full=True
                        ):
                            fourth_regs[i] = fourth_regs[i] * correction_factor
                        cute.copy(fourth_store, fourth_regs, fourth_src)

            cute.arch.fence_view_async_tmem_store()
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        correction_group_idx: int,
    ) -> pipeline.PipelineState:
        """Epilogue for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param correction_group_idx: The PV correction warpgroup index (0 or 1)
        :type correction_group_idx: int

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        for iter_n in cutlass.range_constexpr(self.iterations_pv_n):
            # mma_o pipeline consumer wait
            common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)

            for group_subtile in cutlass.range_constexpr(
                self.epilogue_n_subtiles // self.num_correction_groups
            ):
                sub_n = (
                    correction_group_idx + group_subtile * self.num_correction_groups
                )
                actual_n = iter_n * self.epilogue_n_subtiles + sub_n
                # tmem load tiled copy and partition results.
                tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_rAcc = (
                    self._tmem_load_partition(
                        common_params,
                        common_params.tiled_mma_pv,
                        actual_n,
                        self.epilogue_subtile_n,
                    )
                )

                # load o
                cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

                # Hoist the empty-row guard out of the vectorized epilogue.

                inverse_row_sum = (
                    cute.arch.rcp_approx(row_sum) if row_sum != 0.0 else 0.0
                )

                # apply output scale and normalize by row_sum
                for i in cutlass.range(
                    cute.size(tTR_rAcc), vectorize=True, unroll_full=True
                ):
                    tTR_rAcc[i] = (
                        tTR_rAcc[i] * epilogue_params.output_scale * inverse_row_sum
                    )

                # store o to global memory
                tR2G_rO_src = None
                if cutlass.const_expr(common_params.mAccO is None):
                    tR2G_rO_src = cute.make_fragment_like(tTR_rAcc, self.o_dtype)
                    tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
                else:
                    tR2G_rO_src = tTR_rAcc

                # Build address-rich global partitions only after the TMEM
                # fragment math and output conversion are complete.
                tTR_gO, tTR_cO = self._gmem_output_partition(
                    common_params,
                    tmem_load_tiled_copy,
                    actual_n,
                    self.epilogue_subtile_n,
                )
                if cute.elem_less(tTR_cO[0][0], common_params.H):
                    cute.autovec_copy(
                        tR2G_rO_src,
                        tTR_gO,
                        l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
                    )

            # Both correction groups and both N iterations see the same
            # metadata. Store LSE exactly once from group 0's first N tile.
            if cutlass.const_expr(iter_n == 0):
                if correction_group_idx == 0:
                    self._store_lse(common_params, epilogue_params, row_sum, row_max)

            cute.arch.fence_view_async_tmem_load()
            common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
            mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def _store_lse(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> None:
        """Store the per-row LSE once for the completed output tile."""

        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cta_pair_size,
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        gLSE = None
        cLSE = None
        if cutlass.const_expr(epilogue_params.mAccLSE is None):
            gLSE = cute.local_tile(
                epilogue_params.mLSE,
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(epilogue_params.mLSE.shape),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
                (1, 1, 1),
            )
        else:
            gLSE = cute.local_tile(
                epilogue_params.mAccLSE[None, common_params.blk_coord[4], None, None],
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(
                    epilogue_params.mAccLSE[
                        None, common_params.blk_coord[4], None, None
                    ].shape
                ),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[2],
                    common_params.blk_coord[3],
                ),
                (1, 1, 1),
            )
        lse = (
            cute.math.log2(row_sum, fastmath=True)
            + epilogue_params.softmax_scale_log2 * row_max
        )
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        if cute.elem_less(cLSE[tidx][0], common_params.H):
            gLSE[tidx] = (
                lse * epilogue_params.lse_scale
                if cutlass.const_expr(epilogue_params.mAccLSE is None)
                else lse
            )

    def make_and_init_load_qkv_pipeline(
        self, load_qkv_mbar_ptr, cta_layout_vmnk, load_stages, tx_count
    ) -> pipeline.PipelineTmaUmma:
        """Create and initialize the tma load qkv pipeline.

        :param load_qkv_mbar_ptr: The load qkv mbar pointer
        :type load_qkv_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]
        :param load_stages: The load stages
        :type load_stages: list[int]
        :param tx_count: The tx count
        :type tx_count: int

        :return: The tma load qkv pipeline
        :rtype: pipeline.PipelineTmaUmma
        """
        load_qkv_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.load_tma_k_warp_id])
        )
        load_qkv_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        # mcast_mode_mn=(1,0): restrict the consumer-release arrival mask to the
        # 2-CTA UMMA pair (the V mode). With the default (1,1) the mask would
        # also span the cluster M mode (both pairs of the 4-CTA cluster).
        return pipeline.PipelineTmaUmma.create(
            barrier_storage=load_qkv_mbar_ptr,
            num_stages=load_stages,
            producer_group=load_qkv_producer_group,
            consumer_group=load_qkv_consumer_group,
            tx_count=tx_count,
            cta_layout_vmnk=cta_layout_vmnk,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )

    def make_and_init_mma_s_pipeline(
        self, mma_s_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma s pipeline.

        :param mma_s_mbar_ptr: The mma s mbar pointer
        :type mma_s_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma s pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_s_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_qk_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp * len(self.compute_warp_ids) * self.cta_pair_size
        )
        mma_s_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_s_mbar_ptr,
            num_stages=self.mma_s_stage,
            producer_group=mma_s_producer_group,
            consumer_group=mma_s_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    def make_and_init_mma_o_pipeline(
        self, mma_o_mbar_ptr, cta_layout_vmnk
    ) -> pipeline.PipelineUmmaAsync:
        """Create and initialize the mma o pipeline.

        :param mma_o_mbar_ptr: The mma o mbar pointer
        :type mma_o_mbar_ptr: cute.Tensor
        :param cta_layout_vmnk: The cta layout vmnk
        :type cta_layout_vmnk: tuple[int, int, int]

        :return: The mma o pipeline
        :rtype: pipeline.PipelineUmmaAsync
        """

        mma_o_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len([self.mma_pv_warp_id])
        )
        consumer_thread_size = (
            self.threads_per_warp
            * self.num_compute_warps
            * self.num_correction_groups
            * self.cta_pair_size
        )
        mma_o_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread,
            consumer_thread_size,
        )
        return pipeline.PipelineUmmaAsync.create(
            barrier_storage=mma_o_mbar_ptr,
            num_stages=self.mma_o_stage,
            producer_group=mma_o_producer_group,
            consumer_group=mma_o_consumer_group,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def preferred_clc_producer(
        self,
        clc_pipeline,
        tile_sched_params,
        clc_response_ptr,
    ):
        """CTA0/W12 produces and consumes responses without blocking Q/K TMA."""
        clc_pipeline, work_tile, consumer_state = self.make_clc_consumer(
            clc_pipeline,
        )
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.ProducerConsumer, self.num_clc_stage
        )
        while work_tile.is_valid_tile:
            clc_pipeline.producer_acquire(producer_state)
            barrier = clc_pipeline.producer_get_barrier(producer_state)
            tile_sched.advance_to_next_work(barrier)
            producer_state.advance()
            clc_pipeline.consumer_wait(consumer_state)
            work_tile = self.get_clc_work(clc_response_ptr)
            clc_pipeline.consumer_release(consumer_state)
            consumer_state.advance()
        # Drain the final invalid response before any CTA can retire its SMEM.
        clc_pipeline.producer_tail(producer_state)

    def make_and_init_clc_pipeline(
        self,
        clc_mbar_ptr: cute.Pointer,
        cta_layout_vmnk: cute.Layout,
        num_consumer_threads: int,
    ) -> pipeline.PipelineClcFetchAsync:
        """Create the one-stage CLC response broadcast pipeline."""
        return pipeline.PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.num_clc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_consumer_threads
            ),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def make_clc_consumer(self, clc_pipeline):
        """Reuse initialized barriers; keep each role's progress independent."""
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_clc_stage
        )
        work_tile = utils.WorkTileInfo(cute.arch.block_idx(), cutlass.Boolean(True))
        return clc_pipeline, work_tile, consumer_state

    @cute.jit
    def get_clc_work(self, clc_response_ptr):
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(clc_response_ptr)
        cute.arch.fence_proxy("async.shared", space="cta")
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(bidx % self.cluster_shape_mnk[0])
        return utils.WorkTileInfo((m_idx + cta_rank, n_idx, l_idx), is_valid)

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
        pv_n_splits: int = 1,
    ) -> Tuple[utils.ClcDynamicPersistentTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[ClcDynamicPersistentTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        assert pv_n_splits == 1, "CLC scheduler requires unsplit PV-N in mixed MLA"
        problem_shape_ntile_mnl = (
            cluster_shape_mnk[0],
            cute.size(o_shape[3]) * cute.size(o_shape[2]),
            split_kv,
        )
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams(
            problem_shape_ntile_mnl,
            cluster_shape_mnk,
            fallback_cluster_shape_mnk=(2, 1, 1),
        )
        grid = tile_sched_params.get_grid_shape()

        return tile_sched_params, grid

    @staticmethod
    def get_workspace_size(
        H: int,
        S: int,
        D: int,
        B: int,
        split_kv: int,
        acc_dtype: Type[cutlass.Numeric],
    ) -> int:
        """Get the extra workspace(device memory) size for the MLA kernel when split_kv is not 1.

        :param H: The height of the output tensor C
        :type H: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param D: The depth of the output tensor C
        :type D: int
        :param B: The batch size of the output tensor C
        :type B: int
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]

        :return: The workspace size for the MLA kernel
        :rtype: int
        """
        if split_kv == 1:
            return 0
        return B * H * S * split_kv * (D + 1) * acc_dtype.width // 8

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param S: The sequence length of the output tensor C
        :type S: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            # Offset the typed pointer in elements, widening before address math.
            acc_lse_iter = cute.recast_ptr(
                acc_o_iter + cutlass.Int64(cute.cosize(acc_o_layout)),
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        """Check if the MLA kernel can be implemented.

        :param B: The batch size of the output tensor C
        :type B: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param K: The width of the output tensor KV
        :type K: int
        :param H: The number of heads of the output tensor C
        :type H: int
        :param L: The number of latent dimensions of the tensor KV
        :type L: int
        :param R: The number of rope dimensions of the tensor C_rope
        :type R: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param page_size: The page size of the page table
        :type page_size: int

        :return: Whether the MLA kernel can be implemented
        :rtype: bool
        """
        if L != 512 or R != 64:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if out_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        # page size equals 1 is prohibited by tma specification, not 128B aligned.
        if mma_qk_tiler_mn[1] % page_size != 0 or page_size == 1:
            return False
        # QK M tile must cover the full head count: the 2x1 cluster splits M
        # across CTAs so per-CTA M = tile_M / 2 = H / 2.
        if mma_qk_tiler_mn[0] != H:
            return False
        # Only (256, 256) is currently supported for the PV tile. Other shapes
        # require retuning the coupled 4-CTA QK/PV topology and its N=32
        # correction-subtile partition.
        if mma_pv_tiler_mn != (256, 256):
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if S <= 0 or S > 4:
            return False
        if K <= 0:
            return False
        return True


class _MixedFallbackBase(_BlackwellMLAFP8):
    """MTP-local fallback overrides; standalone decode stays unchanged.

    Keep original-query causal coordinates and the validated softmax, correction
    and epilogue implementations here. Unchanged Blackwell machinery is inherited.
    These methods are scoped to MTP rather than modifying the shared decode class.
    """

    causal_num_heads: int

    arch_str = "sm_107"
    arch_name = "Rubin SM107"

    def __init__(self, *args, use_fp16_softmax=False, **kwargs):
        _BlackwellMLAFP8.__init__(self, *args, **kwargs)
        self.use_fp16_softmax = use_fp16_softmax
        self.softmax_reg_num = 160
        self.correction_reg_num = 160
        self.other_reg_num = 32

    def _setup_attributes(self):
        _BlackwellMLAFP8._setup_attributes(self)
        self.load_k_stage = 4
        self.load_v_stage = 4

    @cute.jit
    def compute(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        k_tile_count: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        is_second_compute_warp: bool,
    ) -> tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]:
        """Compute warp to compute the result of softmax, rescale, and epilogue. Updates the related pipeline states.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param k_tile_count: The number of k-tiles
        :type k_tile_count: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState
        :param is_second_compute_warp: True for g1 (odd k-tiles), False for g0
        :type is_second_compute_warp: bool

        :return: The MMA s consumer state, the P MMA producer state, and the P correction producer state
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState]
        """

        k_tile_total = cute.ceil_div(common_params.K, self.mma_qk_tiler[1])

        # 2softmax: row_max initialised from the init seed (-inf) — same value
        # that init_p_cor_metadata writes to peer-readable TMEM below, so the
        # first tile's load_other_group_metadata + fmax(row_max_new, peer) gives
        # the correct identity for online softmax.
        row_max = self.acc_dtype(self.init_row_max)
        row_sum = self.acc_dtype(0)
        correction_factor = self.acc_dtype(1)
        odd_k_tile = k_tile_count % 2 == 1
        # 2softmax: g0 takes even k-tiles, g1 takes odd k-tiles. g1 advances all
        # pipeline states once at entry to start on stage 1 (one stage per group).
        if cutlass.const_expr(is_second_compute_warp):
            k_index = k_index + 1
            k_tile_count = k_tile_count // 2
        else:
            k_tile_count = (k_tile_count + 1) // 2  # g0 takes the extra tile if odd
        valid_k_tile_count = k_tile_count > 0
        # wait for 2 softmax warp groups to arrive at the barrier to avoid divergent across waves.
        self.softmax_warps_initial_sync_bar.arrive_and_wait()
        common_params.p_cor_pipeline.producer_acquire(p_cor_producer_state)

        # Seed this group's home p_cor TMEM stage so the FIRST tile's peer-read
        # in load_other_group_metadata returns (row_max=-inf, row_sum=0) instead
        # of uninitialised TMEM. Must run after producer_acquire so this group
        # owns its stage write.
        if cutlass.const_expr(is_second_compute_warp):
            self.init_p_cor_metadata(
                common_params, softmax_params, p_cor_producer_state
            )
            self.softmax_order_bar_0.arrive()
        # Number of tiles from the global-K end that may contain causal-masked
        # positions. Min k_bound = K - (S_q-1), which can span up to
        # ceil((seq_len_q-2)/tile_N)+1 tiles (tile-boundary-crossing case). For
        # S_q=1 this reduces to 1 tile — identical to a plain K-bound check.
        tile_n = self.mma_qk_tiler[1]
        mask_tile_count = (self.seq_len_q - 2 + tile_n - 1) // tile_n + 1

        # first_mask_tile_idx is the global index of the first tile that may
        # need masking. Runtime because it depends on K (per-batch in
        # var-seq / split-KV).
        first_mask_tile_idx = k_tile_total - mask_tile_count

        # Phase 1: pure unmasked bulk tiles (all columns strictly < min k_bound).
        # 2softmax: each group steps by 2 k-tiles; phase boundaries respect this.
        while k_tile_count > 0 and (
            cutlass.const_expr(getattr(self, "merge_softmax_loops", False))
            or k_index < first_mask_tile_idx
        ):
            is_local_last_tile = (
                False
                if cutlass.const_expr(common_params.mAccO is None)
                else k_tile_count == 1
            )
            apply_mask = False
            if cutlass.const_expr(getattr(self, "merge_softmax_loops", False)):
                apply_mask = k_index >= first_mask_tile_idx
                # Preserve the old masked phase's unconditional local-last flag.
                is_local_last_tile = apply_mask or is_local_last_tile
            (
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
            ) = self.softmax(
                common_params,
                softmax_params,
                k_index,
                mma_s_consumer_state,
                p_mma_producer_state,
                p_cor_producer_state,
                row_max,
                row_sum,
                correction_factor,
                is_second_compute_warp,
                apply_mask,
                is_local_last_tile,
            )
            k_index = k_index + 2
            k_tile_count = k_tile_count - 1
            if k_tile_count > 0:
                p_mma_producer_state, mma_s_consumer_state, p_cor_producer_state = (
                    self.softmax_advance_to_next_group(
                        common_params,
                        p_mma_producer_state,
                        mma_s_consumer_state,
                        p_cor_producer_state,
                    )
                )

        # Phase 2: remaining tiles that overlap the causal / K-bound region,
        # including this work-split's final tile.
        if cutlass.const_expr(not getattr(self, "merge_softmax_loops", False)):
            while k_tile_count > 0:
                (
                    mma_s_consumer_state,
                    p_mma_producer_state,
                    p_cor_producer_state,
                    row_max,
                    row_sum,
                    correction_factor,
                ) = self.softmax(
                    common_params,
                    softmax_params,
                    k_index,
                    mma_s_consumer_state,
                    p_mma_producer_state,
                    p_cor_producer_state,
                    row_max,
                    row_sum,
                    correction_factor,
                    is_second_compute_warp,
                    True,
                    True,
                )
                k_index = k_index + 2
                k_tile_count = k_tile_count - 1
                if k_tile_count > 0:
                    p_mma_producer_state, mma_s_consumer_state, p_cor_producer_state = (
                        self.softmax_advance_to_next_group(
                            common_params,
                            p_mma_producer_state,
                            mma_s_consumer_state,
                            p_cor_producer_state,
                        )
                    )

        if odd_k_tile and valid_k_tile_count:
            if cutlass.const_expr(is_second_compute_warp):
                # next first compute warp in this wave
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
                # first compute warp in next wave
                p_mma_producer_state.advance()
                mma_s_consumer_state.advance()
                p_cor_producer_state.advance()
        else:
            p_mma_producer_state.advance()
            mma_s_consumer_state.advance()
            p_cor_producer_state.advance()
        if cutlass.const_expr(is_second_compute_warp):
            if odd_k_tile:
                self.softmax_order_bar_1.arrive_and_wait()
        else:
            if not odd_k_tile:
                self.softmax_order_bar_0.arrive_and_wait()
        return mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state

    @cute.jit
    def softmax(
        self,
        common_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        k_index: cutlass.Int32,
        mma_s_consumer_state: pipeline.PipelineState,
        p_mma_producer_state: pipeline.PipelineState,
        p_cor_producer_state: pipeline.PipelineState,
        row_max: cutlass.Float32,
        row_sum: cutlass.Float32,
        correction_factor: cutlass.Float32,
        is_second_compute_warp: bool,
        apply_mask: bool,
        is_local_last_tile: cutlass.Boolean,
    ) -> tuple[
        pipeline.PipelineState,
        pipeline.PipelineState,
        pipeline.PipelineState,
        cutlass.Float32,
        cutlass.Float32,
        cutlass.Float32,
    ]:
        """Softmax for one k-tile. Updates the related pipeline states and returns the computed results.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param softmax_params: The softmax parameters
        :type softmax_params: SimpleNamespace
        :param k_index: The index of the k-tile
        :type k_index: cutlass.Int32
        :param mma_s_consumer_state: The MMA s consumer state
        :type mma_s_consumer_state: pipeline.PipelineState
        :param p_mma_producer_state: The P MMA producer state
        :type p_mma_producer_state: pipeline.PipelineState
        :param p_cor_producer_state: The P correction producer state
        :type p_cor_producer_state: pipeline.PipelineState
        :param row_max: The row max
        :type row_max: cutlass.Float32
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param apply_mask: Whether the tile needs K-bound / causal masking (Python bool
            for the unmasked/masked bulk loops; runtime cutlass.Boolean for the
            split-KV final iter where mask only applies on the global last tile).
        :type apply_mask: bool | cutlass.Boolean
        :param is_local_last_tile: Whether the last tile is local
        :type is_local_last_tile: cutlass.Boolean

        :return: The MMA s consumer state, the P MMA producer state, the P correction producer state, the row max, the row sum, and the correction factor
        :rtype: tuple[pipeline.PipelineState, pipeline.PipelineState, pipeline.PipelineState, cutlass.Float32, cutlass.Float32, cutlass.Float32]
        """

        softmax_exchange_sync_bar = (
            self.softmax_exchange_sync_bar_1
            if is_second_compute_warp
            else self.softmax_exchange_sync_bar_0
        )

        softmax_params.mma_s_pipeline.consumer_wait(mma_s_consumer_state)

        # load S from tmem
        tStS_shape = softmax_params.tiled_mma_qk.partition_shape_C(
            cute.select(self.mma_qk_tiler, mode=[0, 1])
        )
        tStS_staged_fake = softmax_params.tiled_mma_qk.make_fragment_C(
            cute.append(tStS_shape, self.mma_s_stage)
        )
        tStS_staged = cute.make_tensor(common_params.tmem_ptr, tStS_staged_fake.layout)
        tStS = tStS_staged[None, None, None, mma_s_consumer_state.index]

        tAcc = tStS[(None, None), 0, 0]
        cta_qk_tiler = (
            self.mma_qk_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_qk_tiler[1],
            self.mma_qk_tiler[2],
        )
        cS = cute.make_identity_tensor(cute.select(cta_qk_tiler, mode=[0, 1]))

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
        )
        tmem_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        tmem_thr_copy = tmem_tiled_copy.get_slice(tidx)
        tTR_tAcc = tmem_thr_copy.partition_S(tAcc)
        tTR_tS = tmem_thr_copy.partition_D(cS)

        tTR_rAcc = cute.make_fragment_like(tTR_tS, self.acc_dtype)

        row_max_new = row_max
        # Spec-decoding (MTP) causal mask: each row represents one (q_token, head)
        # pair; row r's effective K bound is K - (S_q - 1 - q_tok(r)).
        # With fold factor F = self.fold_sq_ratio (fold_sq=True), the M tile is
        # laid out as [F sub_q_tok][num_heads heads] and there are S_q/F outer
        # chunks indexed by blk_coord[1]:
        #   q_tok(r) = blk_coord[1] * F + (r_global // num_heads)
        # r_global = row_in_cta + cluster_idx * (M_tile / cluster_m)
        # When fold_sq=False this reduces to q_tok = blk_coord[1]. For S_q=1
        # this further reduces to k_bound = K (plain K-bound check).
        # True -inf masks have zero probability at every admitted QK scale.
        # Empty-row normalization below preserves the zero-sum identity.
        cta_m_rows = self.mma_qk_tiler[0] // self.cluster_shape_mnk[0]
        arch = BaseDSL._get_dsl().get_arch_enum()
        if cutlass.const_expr(arch >= Arch.sm_100 and arch <= Arch.sm_100f):
            cute.copy(tmem_tiled_copy, tTR_tAcc, tTR_rAcc)
            for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                if apply_mask:
                    # This independent mixed copy always uses original-query causality.
                    # Fallback s enumerates half-M row blocks, not query tokens.
                    q_tok = (
                        common_params.blk_coord[1] * self.mma_qk_tiler[0]
                        + common_params.blk_coord[0] * cta_m_rows
                        + tTR_tS[i][0]
                    ) // self.causal_num_heads
                    k_bound = common_params.K - (self.seq_len_q - 1) + q_tok
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else -self.acc_dtype.inf
                    )
            # reduction for row_max
            row_max_new = tTR_rAcc.load().reduce(cute.ReductionOp.MAX, row_max_new, 0)
        elif cutlass.const_expr(
            (arch >= Arch.sm_101 and arch <= Arch.sm_101f)
            or (arch >= Arch.sm_103 and arch <= Arch.sm_103f)
            # {$nv-internal-release begin}
            or (arch >= Arch.sm_107 and arch <= Arch.sm_107f)
            # {$nv-internal-release end}
            or (arch >= Arch.sm_110 and arch <= Arch.sm_110f)
        ):
            tmem_load_red_atom = cute.make_copy_atom(
                tcgen05.copy.LdRed32x32bOp(
                    tcgen05.copy.Repetition(64), redOp=tcgen05.TmemLoadRedOp.MAX
                ),
                self.acc_dtype,
            )
            tmem_red_tiled_copy = tcgen05.make_tmem_copy(tmem_load_red_atom, tAcc)
            tmem_red_thr_copy = tmem_red_tiled_copy.get_slice(tidx)
            tTR_tAcc_red = tmem_red_thr_copy.partition_S(tAcc)
            tTR_tS_red = tmem_red_thr_copy.partition_D(cS)
            tTR_rAcc_red = cute.make_fragment_like(tTR_tS_red, self.acc_dtype)
            tTR_rMax = cute.make_rmem_tensor(
                cute.make_layout((1, tTR_tS_red.shape[1], tTR_tS_red.shape[2])),
                self.acc_dtype,
            )
            cute.copy(
                tmem_red_tiled_copy,
                tTR_tAcc_red,
                (tTR_rAcc_red, tTR_rMax),
            )
            tTR_rAcc = cute.make_tensor(tTR_rAcc_red.iterator, tTR_rAcc.layout)
            if apply_mask:
                for i in cutlass.range_constexpr(cute.size(tTR_rAcc)):
                    # This independent mixed copy always uses original-query causality.
                    # Fallback s enumerates half-M row blocks, not query tokens.
                    q_tok = (
                        common_params.blk_coord[1] * self.mma_qk_tiler[0]
                        + common_params.blk_coord[0] * cta_m_rows
                        + tTR_tS[i][0]
                    ) // self.causal_num_heads
                    k_bound = common_params.K - (self.seq_len_q - 1) + q_tok
                    tTR_rAcc[i] = (
                        tTR_rAcc[i]
                        if cute.elem_less(
                            tTR_tS[i][1] + self.mma_qk_tiler[1] * k_index,
                            k_bound,
                        )
                        else -self.acc_dtype.inf
                    )
                # reduction for row_max after manual masking
                row_max_new = tTR_rAcc.load().reduce(
                    cute.ReductionOp.MAX, row_max_new, 0
                )
            else:
                # sm_101+ pre-computed max via reduction is valid here because
                # tTR_rAcc is unmodified (no mask applied to this tile).
                row_max_new = cute.arch.fmax(row_max_new, tTR_rMax[0])
        # fence between tmem load and mma s
        cute.arch.fence_view_async_tmem_load()

        softmax_params.mma_s_pipeline.consumer_release(mma_s_consumer_state)

        # Intra-group warps_in_n=2 exchange across warps (0,1)↔(2,3) within
        # the group. Each group writes into its own half of softmax_smem_exchange:
        # g0 → slots [0, 128), g1 → slots [128, 256). The named barrier covers
        # both groups (one arrive_and_wait serves both the intra-group reduce
        # AND the cross-group Sync #1 below — see the second smem_exchange read
        # which gives this thread the PEER GROUP's row_max).
        _group_offset = self.num_compute_warps * self.threads_per_warp
        if cutlass.const_expr(is_second_compute_warp):
            _my_base = _group_offset
            _peer_base = 0
        else:
            _my_base = 0
            _peer_base = _group_offset
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[_my_base + tidx] = row_max_new
            softmax_exchange_sync_bar.arrive_and_wait()
            row_max_new = cute.arch.fmax(
                row_max_new,
                common_params.smem_exchange[
                    _my_base
                    + (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ],
            )

        # === 2softmax cross-group merge via TMEM peer-read (tunePerf pattern) ===
        # Pingpong A: wait for the OTHER group to release us. The peer's
        # exchange_p_cor_metadata TMEM store on its prev tile is happens-before
        # the .arrive() it issues at the bottom of its prev iter, so this
        # arrive_and_wait gives us the acquire memory ordering needed for the
        # load_other_group_metadata read below. g0's first wait is satisfied
        # by g1's pre-arrive of bar_0 at warp setup (init-phase trick).
        if cutlass.const_expr(is_second_compute_warp):
            self.softmax_order_bar_1.arrive_and_wait()
        else:
            self.softmax_order_bar_0.arrive_and_wait()
        # cute.nvgpu.cfence()

        # Serial inheritance: peer's prev-tile metadata IS the GLOBAL running
        # state right before THIS tile in serial tile order (pingpong serializes
        # tiles 0,1,2,3,... across g0/g1). Override (row_max, row_sum) with
        # peer's prev values so the subsequent correction = exp2(prev_global -
        # this_global) and the row_sum update gives running_sum_after_this_tile
        # = GLOBAL state.
        other_row_max, other_row_sum = self.load_other_group_metadata(
            common_params, softmax_params, p_cor_producer_state
        )
        row_max_new = cute.arch.fmax(row_max_new, other_row_max)
        row_max = other_row_max
        row_sum = other_row_sum

        # Choose the retained normalization base before computing the factor.
        # Otherwise metadata exchange rolls max back but row_sum is still
        # rescaled by the pre-rollback factor (see skipcorr_fix_20260916).
        if (
            row_max_new - row_max
        ) * softmax_params.softmax_scale_log2 <= self.skip_correction_threshold:
            row_max_new = row_max

        # find correction factor (uses inherited row_max from peer = prev global max)
        correction_factor = _rescale_factor(
            row_max, row_max_new, softmax_params.softmax_scale_log2
        )
        saved_p_cor_idx = p_cor_producer_state.index
        # Early store of (row_max_new, correction_factor, no_correction) — row_sum
        # field carries the inherited peer-prev value (the global sum BEFORE this
        # tile), which is a placeholder; the real updated row_sum is patched in
        # via store_p_cor_row_sum below at saved_p_cor_idx. Safe because the
        # correction warp uses row_sum only at the LAST tile path (separate is_local_last_tile branch).
        if not is_local_last_tile:
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )

        # softmax
        fma_b = softmax_params.softmax_scale_log2
        fma_c = (
            0.0 - row_max_new if row_max_new != -self.acc_dtype.inf else 0.0
        ) * softmax_params.softmax_scale_log2

        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * fma_b + fma_c
            tTR_rAcc[i] = cute.math.exp2(tTR_rAcc[i], fastmath=True)

        tTR_rS = cute.make_fragment_like(tTR_tS, self.q_dtype)

        # quantize
        tTR_rS.store(tTR_rAcc.load().to(self.q_dtype))

        # create sP
        sP = softmax_params.sP[None, None, None, (None, p_mma_producer_state.index)]
        sP_mk_view = cute.make_tensor(
            sP.iterator,
            cute.make_layout(
                (
                    (sP.shape[0][0], sP.shape[1]),
                    (sP.shape[0][1], sP.shape[2], sP.shape[3]),
                ),
                stride=(
                    (sP.stride[0][0], sP.stride[1]),
                    (sP.stride[0][1], sP.stride[2], sP.stride[3]),
                ),
            ),
        )
        # {$nv-internal-release begin}
        # TODO: figure out if we could use A tmem for pv.
        # {$nv-internal-release end}
        # change to PISL
        sP_wo_swizzle_iter = cute.recast_ptr(sP.iterator, swizzle_=None)
        swizzle_bits = (
            int(math.log2(self.mma_pv_tiler[2] * self.q_dtype.width // 8 // 32)) + 1
        )
        swizzle_base = 3 if self.q_dtype.width == 16 else 4
        sP_swizzle = cute.make_swizzle(swizzle_bits, swizzle_base, 3)
        sP_mk_view = cute.make_tensor(
            sP_wo_swizzle_iter,
            cute.make_composed_layout(sP_swizzle, 0, sP_mk_view.layout),
        )
        universal_copy_bits = 128
        smem_copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.q_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        smem_tiled_copy = cute.make_tiled_copy_D(smem_copy_atom, tmem_tiled_copy)
        smem_thr_copy = smem_tiled_copy.get_slice(tidx)
        rP_copy_view = smem_thr_copy.retile(tTR_rS)
        sP_copy_view = smem_thr_copy.partition_D(sP_mk_view)

        softmax_params.p_mma_pipeline.producer_acquire(p_mma_producer_state)
        cute.copy(smem_tiled_copy, rP_copy_view, sP_copy_view)

        # fence between smem store and mma o
        cute.arch.fence_view_async_shared()
        softmax_params.p_mma_pipeline.producer_commit(p_mma_producer_state)
        p_mma_producer_state.advance()

        # row_sum update: tunePerf peer-inheritance pattern.
        # row_sum entering this block = peer's prev-tile row_sum (the global
        # running sum BEFORE this tile, inherited via load_other_group_metadata).
        # Apply correction (rescale prev to new row_max base) and add this tile's
        # locally reduced exp sum → running_sum_after_this_tile = GLOBAL state.
        # No cross-group SMEM exchange needed; the next iter's peer-read picks
        # up this group's freshly written row_sum from TMEM corr.
        row_sum = row_sum * correction_factor
        row_sum_vec = (0.0, 0.0)
        for i in cutlass.range_constexpr(0, cute.size(tTR_rAcc), 2):
            row_sum_vec = cute.arch.add_packed_f32x2(
                row_sum_vec, (tTR_rAcc[i], tTR_rAcc[i + 1])
            )
        row_sum = row_sum_vec[0] + row_sum_vec[1] + row_sum

        # Late store of row_sum ONLY — patches in the real row_sum at the slot
        # committed by the early store above so peer's next-iter
        # load_other_group_metadata sees the updated global row_sum. No commit
        # / advance (already done by early store).
        if not is_local_last_tile:
            self.store_p_cor_row_sum(
                common_params,
                row_sum,
                saved_p_cor_idx,
                tAcc,
                tidx,
            )

        # Pingpong B — signal the OTHER group to start its critical section.
        # Split-phase .arrive() contributes this group's threads to the other
        # group's bar and falls through immediately. cfence: control-flow fence
        # to prevent ptxas from hoisting subsequent work above bar.arrive.

        # split kv case
        if is_local_last_tile:
            p_cor_producer_state, row_max_new = self.exchange_p_cor_metadata(
                common_params,
                softmax_params,
                correction_factor,
                row_sum,
                row_max,
                row_max_new,
                tAcc,
                tidx,
                p_cor_producer_state,
            )
        # cute.nvgpu.cfence()
        if cutlass.const_expr(is_second_compute_warp):
            self.softmax_order_bar_0.arrive()  # g1 → g0
        else:
            self.softmax_order_bar_1.arrive()  # g0 → g1
        # cute.nvgpu.cfence()

        mma_s_consumer_state.advance()

        return (
            mma_s_consumer_state,
            p_mma_producer_state,
            p_cor_producer_state,
            row_max_new,
            row_sum,
            correction_factor,
        )

    @cute.jit
    def rescale(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        no_correction: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Rescale for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param correction_factor: The correction factor
        :type correction_factor: cutlass.Float32
        :param no_correction: Whether to apply correction factor
        :type no_correction: cutlass.Int32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """
        skip_correction = cute.arch.vote_all_sync(no_correction == 1)
        # Share the fallback correction body across PV-N tiles. Preserve the
        # per-tile ownership protocol even when correction arithmetic is skipped.
        for iter_n in cutlass.range(self.iterations_pv_n, unroll=1):
            mma_o_consumer_state = self._rescale_one_n(
                common_params,
                mma_o_consumer_state,
                correction_factor,
                skip_correction,
                iter_n,
            )
        return mma_o_consumer_state

    @cute.jit
    def _rescale_one_n(
        self,
        common_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        correction_factor: cutlass.Float32,
        skip_correction: cutlass.Boolean,
        iter_n: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Keep iteration-local copy descriptors out of runtime loop state."""
        common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
        if not skip_correction:
            # tmem load tiled copy and partition results.
            tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
                self._tmem_load_partition(
                    common_params, common_params.tiled_mma_pv, iter_n
                )
            )

            # tmem store tiled copy
            tmem_store_atom = cute.make_copy_atom(
                tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)), self.acc_dtype
            )
            tmem_store_tiled_copy = tcgen05.make_tmem_copy(tmem_store_atom, tAcc)

            # load o
            cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)
            # rescale, using `mul_packed_f32x2` to reduce the number of instructions
            for i in cutlass.range(
                cute.size(tTR_rAcc), vectorize=True, unroll_full=True
            ):
                tTR_rAcc[i] = tTR_rAcc[i] * correction_factor

            # store o to tensor memory for next k tile
            cute.copy(tmem_store_tiled_copy, tTR_rAcc, tTR_tAcc)

        cute.arch.fence_view_async_tmem_store()
        common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()

        return mma_o_consumer_state

    @cute.jit
    def epilogue(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
    ) -> pipeline.PipelineState:
        """Epilogue for one k-tile. Updates the related pipeline state.

        :param common_params: The common parameters
        :type common_params: SimpleNamespace
        :param epilogue_params: The epilogue parameters
        :type epilogue_params: SimpleNamespace
        :param mma_o_consumer_state: The mma o consumer state
        :type mma_o_consumer_state: pipeline.PipelineState
        :param row_sum: The row sum
        :type row_sum: cutlass.Float32
        :param row_max: The row max
        :type row_max: cutlass.Float32

        :return: The MMA o consumer state
        :rtype: pipeline.PipelineState
        """

        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)

        # exchange row_sum between warps (0, 1) and (2, 3)
        if cutlass.const_expr(self.warps_in_n == 2):
            common_params.smem_exchange[tidx] = row_sum
            self.epilogue_exchange_sync_bar.wait()
            # (64, 2)
            row_sum = (
                row_sum
                + common_params.smem_exchange[
                    (tidx + 64) % (self.num_compute_warps * self.threads_per_warp)
                ]
            )
        # Keep iteration-local copy objects out of the runtime loop state.
        for iter_n in cutlass.range(self.iterations_pv_n, unroll=1):
            mma_o_consumer_state = self._epilogue_one_n(
                common_params,
                epilogue_params,
                mma_o_consumer_state,
                row_sum,
                row_max,
                iter_n,
            )
        return mma_o_consumer_state

    @cute.jit
    def _epilogue_one_n(
        self,
        common_params: SimpleNamespace,
        epilogue_params: SimpleNamespace,
        mma_o_consumer_state: pipeline.PipelineState,
        row_sum: cutlass.Float32,
        row_max: cutlass.Float32,
        iter_n: cutlass.Int32,
    ) -> pipeline.PipelineState:
        """Process one PV N tile, preserving its wait/store/release sequence."""
        tidx = common_params.tidx % (self.num_compute_warps * self.threads_per_warp)
        common_params.mma_o_pipeline.consumer_wait(mma_o_consumer_state)
        # tmem load tiled copy and partition results.
        tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc = (
            self._tmem_load_partition(common_params, common_params.tiled_mma_pv, iter_n)
        )

        # load o
        cute.copy(tmem_load_tiled_copy, tTR_tAcc, tTR_rAcc)

        # Hoist the empty-row guard out of the vectorized epilogue.

        inverse_row_sum = cute.arch.rcp_approx(row_sum) if row_sum != 0.0 else 0.0

        # apply output scale and normalize by row_sum
        for i in cutlass.range(cute.size(tTR_rAcc), vectorize=True, unroll_full=True):
            tTR_rAcc[i] = tTR_rAcc[i] * epilogue_params.output_scale * inverse_row_sum

        # store o to global memory
        tR2G_rO_src = None
        tR2G_rO_dst = tTR_gO
        if cutlass.const_expr(common_params.mAccO is None):
            tR2G_rO_src = cute.make_fragment_like(tTR_gO, self.o_dtype)
            # using final output dtype for o
            tR2G_rO_src.store(tTR_rAcc.load().to(self.o_dtype))
        else:
            # using accumulate dtype for o
            tR2G_rO_src = tTR_rAcc

        if cute.elem_less(tTR_cO[0][0], common_params.H):
            cute.autovec_copy(
                tR2G_rO_src,
                tR2G_rO_dst,
                l1c_evict_priority=cute.nvgpu.CacheEvictionPriority.NO_ALLOCATE,
            )

        # store the lse to global memory
        cta_pv_tiler = (
            self.mma_pv_tiler[0] // self.cluster_shape_mnk[0],
            self.mma_pv_tiler[1],
            self.mma_pv_tiler[2],
        )
        gLSE = None
        cLSE = None
        if cutlass.const_expr(epilogue_params.mAccLSE is None):
            gLSE = cute.local_tile(
                epilogue_params.mLSE,
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(epilogue_params.mLSE.shape),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )

        else:
            gLSE = cute.local_tile(
                epilogue_params.mAccLSE[None, common_params.blk_coord[3], None, None],
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
            cLSE = cute.local_tile(
                cute.make_identity_tensor(
                    epilogue_params.mAccLSE[
                        None, common_params.blk_coord[3], None, None
                    ].shape
                ),
                (cta_pv_tiler[0], 1, 1),
                (
                    common_params.blk_coord[0],
                    common_params.blk_coord[1],
                    common_params.blk_coord[2],
                ),
                (1, 1, 1),
            )
        lse = (
            cute.math.log2(row_sum, fastmath=True)
            + epilogue_params.softmax_scale_log2 * row_max
        )
        if cutlass.const_expr(self.warps_in_n == 2):
            if cute.elem_less(cLSE[tidx][0], common_params.H):
                gLSE[tidx] = (
                    lse * epilogue_params.lse_scale
                    if cutlass.const_expr(epilogue_params.mAccLSE is None)
                    else lse
                )

        cute.arch.fence_view_async_tmem_load()
        common_params.mma_o_pipeline.consumer_release(mma_o_consumer_state)
        mma_o_consumer_state.advance()
        return mma_o_consumer_state

    @cute.jit
    def initialize_workspace(
        self,
        H: cutlass.Int32,
        D: cutlass.Int32,
        S: cutlass.Int32,
        B: cutlass.Int32,
        split_kv: cutlass.Int32,
        acc_dtype: Type[cutlass.Numeric],
        workspace: cute.Tensor,
    ) -> tuple[cute.Tensor, cute.Tensor]:
        """Initialize the workspace for the MLA kernel. Construct the intermediate tensors
        acc_o and acc_lse.

        :param H: The height of the output tensor C
        :type H: cutlass.Int32
        :param D: The depth of the output tensor C
        :type D: cutlass.Int32
        :param S: The sequence length of the output tensor C
        :type S: cutlass.Int32
        :param B: The batch size of the output tensor C
        :type B: cutlass.Int32
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: cutlass.Int32
        :param acc_dtype: The data type of the output tensor C
        :type acc_dtype: Type[cutlass.Numeric]
        :param workspace: The workspace tensor
        :type workspace: cute.Tensor

        :return: The output tensor C and the workspace tensor
        :rtype: tuple[cute.Tensor, cute.Tensor]
        """
        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            align = 256 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, S, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(split_kv * H * D, align),
                    cute.assume(H * split_kv * S * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, S, B),
                stride=(split_kv, 1, H * split_kv, H * split_kv * S),
            )
            # Offset the typed pointer in elements, widening before address math.
            acc_lse_iter = cute.recast_ptr(
                acc_o_iter + cutlass.Int64(cute.cosize(acc_o_layout)),
                dtype=acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)
        return acc_o, acc_lse

    @staticmethod
    def can_implement(
        B: int,
        S: int,
        K: int,
        H: int,
        L: int,
        R: int,
        in_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        acc_dtype: Type[cutlass.Numeric],
        lse_dtype: Type[cutlass.Numeric],
        mma_qk_tiler_mn: Tuple[int, int],
        mma_pv_tiler_mn: Tuple[int, int],
        split_kv: int,
        is_persistent: bool,
        is_var_seq: bool,
        is_var_split_kv: bool,
        page_size: int,
    ) -> bool:
        """Check if the MLA kernel can be implemented.

        :param B: The batch size of the output tensor C
        :type B: int
        :param S: The sequence length of the output tensor C
        :type S: int
        :param K: The width of the output tensor KV
        :type K: int
        :param H: The number of heads of the output tensor C
        :type H: int
        :param L: The number of latent dimensions of the tensor KV
        :type L: int
        :param R: The number of rope dimensions of the tensor C_rope
        :type R: int
        :param in_dtype: The data type of the input tensor
        :type in_dtype: Type[cutlass.Numeric]
        :param out_dtype: The data type of the output tensor
        :type out_dtype: Type[cutlass.Numeric]
        :param acc_dtype: The data type of the accumulator
        :type acc_dtype: Type[cutlass.Numeric]
        :param lse_dtype: The data type of the log-sum-exp
        :type lse_dtype: Type[cutlass.Numeric]
        :param mma_qk_tiler_mn: The tile shape of the query-key matrix multiplication
        :type mma_qk_tiler_mn: Tuple[int, int]
        :param mma_pv_tiler_mn: The tile shape of the probability-value matrix multiplication
        :type mma_pv_tiler_mn: Tuple[int, int]
        :param split_kv: The split key-value of the output tensor C
        :type split_kv: int
        :param is_persistent: Whether to use persistent kernel optimization
        :type is_persistent: bool
        :param is_var_seq: Whether to use variable sequence length
        :type is_var_seq: bool
        :param is_var_split_kv: Whether to use variable split_kv
        :type is_var_split_kv: bool
        :param page_size: The page size of the page table
        :type page_size: int

        :return: Whether the MLA kernel can be implemented
        :rtype: bool
        """
        if L != 512 or R != 64:
            return False
        if in_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if out_dtype not in [cutlass.Float8E4M3FN]:
            return False
        if acc_dtype != cutlass.Float32 or lse_dtype != cutlass.Float32:
            return False
        # page size equals 1 is prohibited by tma specification, not 128B aligned.
        if mma_qk_tiler_mn[1] % page_size != 0 or page_size == 1:
            return False
        if mma_qk_tiler_mn[0] != mma_pv_tiler_mn[0] or mma_qk_tiler_mn[0] != 128:
            return False
        if is_var_split_kv and not is_var_seq:
            return False
        if H > mma_qk_tiler_mn[0]:
            return False
        # When H < M tile, fold up to F tokens of S into H (M_eff = H*F ≤ M_tile).
        # F is auto-picked by run() as the largest divisor of S with H*F ≤ M_tile.
        # F=1 always works, so any (H ≤ M_tile, S ≥ 1) is implementable.
        if S <= 0:
            return False
        if K <= 0:
            return False
        return True


class _MixedFallbackPrep(_MixedFallbackBase):
    """1xfp8 baseline (Blackwell pingpong body + Rubin hooks) refactored for
    the mixed-CGA mega-kernel.

    Softmax dispatch goes directly to the MTP-local fallback implementation.
    It preserves original-query causal coordinates and the validated FP16/FP32
    paths without changing the standalone Blackwell or Rubin decode classes.

    * ``prepare_args`` is the Blackwell ``__call__`` with the workspace init
      and the launch removed. It re-views the row tensors as
      (M rows, s = 2 row blocks) and builds the (2,1,1)-cluster TMA
      atoms/layouts; the caller supplies the shared (4,1,1) tile-scheduler
      params and the acc views. Its internal grid/scheduler computations are
      dead code kept for textual fidelity.
    * ``fallback_kernel_body`` is the Blackwell ``split_kv_kernel`` body
      re-scheduled onto the (4,1,1) linear work walk: fallback pair
      ``p = cluster_idx4 // 2`` runs the baseline natively on the s = p row
    block of its unit (full K range, own acc-slot indexing).
    """

    early_compute_clc: bool
    merge_softmax_loops: bool

    num_clc_stage = 1
    num_clc_response_bytes = 16

    @cute.jit
    def softmax(self, *args):
        return _MixedFallbackBase.softmax(self, *args)

    def make_and_init_clc_pipeline(
        self,
        clc_mbar_ptr: cute.Pointer,
        cta_layout_vmnk: cute.Layout,
        num_consumer_threads: int,
    ) -> pipeline.PipelineClcFetchAsync:
        return pipeline.PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=1,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, num_consumer_threads
            ),
            tx_count=16,
            cta_layout_vmnk=cta_layout_vmnk,
            defer_sync=True,
        )

    @cute.jit
    def make_clc_consumer(self, clc_pipeline):
        """Reuse initialized barriers; keep each role's progress independent."""
        consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, 1
        )
        work_tile = utils.WorkTileInfo(cute.arch.block_idx(), cutlass.Boolean(True))
        return clc_pipeline, work_tile, consumer_state

    @cute.jit
    def get_clc_work(self, clc_response_ptr):
        m_idx, n_idx, l_idx, is_valid = cute.arch.clc_response(clc_response_ptr)
        cute.arch.fence_proxy("async.shared", space="cta")
        bidx, _, _ = cute.arch.block_idx()
        cta_rank = cute.arch.make_warp_uniform(bidx % self.cluster_shape_mnk[0])
        return utils.WorkTileInfo((m_idx + cta_rank, n_idx, l_idx), is_valid)

    @staticmethod
    def _halve_rows(t, s_mode: int):
        """(2M, ..., S, ...) -> (M, ..., 2*S, ...) without copying.

        The half-head block is the fastest submode: s_new = 2*s + half.
        Multi-query inputs must pack head blocks contiguously across queries.
        """
        _shape = tuple(t.shape)
        _stride = tuple(t.stride)
        # H and Q are contiguous in the harness's (B,Q,H,D) allocation.
        # Shapes/strides are runtime values when marked dynamic, so do not
        # feed their comparisons into a compile-time const_expr assertion.
        _new_shape = tuple(
            (v // 2 if i == 0 else (2 * v if i == s_mode else v))
            for i, v in enumerate(_shape)
        )
        _new_stride = tuple(
            ((_shape[0] // 2) * _stride[0] if i == s_mode else v)
            for i, v in enumerate(_stride)
        )
        return cute.make_tensor(
            t.iterator, cute.make_layout(_new_shape, stride=_new_stride)
        )

    @cute.jit
    def prepare_args(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        acc_o: Optional[cute.Tensor],
        acc_lse: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [num_head, rope_dim, seq_len_q, batch_size]
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [seq_len_k, latent_dim, batch_size]
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [seq_len_k, rope_dim, batch_size]
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [page_count, batch_size]
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, seq_len_q, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, seq_len_q, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # mixed-CGA: present the 2M-row unit as (M rows, s = 2 row
        # blocks) — the baseline's native multi-token form (pure stride
        # re-view; the kernel walk hands pair p the s = p block).
        # Preserve every query while folding each into two half-head blocks.
        q_latent = self._halve_rows(q_latent, 2)
        q_rope = self._halve_rows(q_rope, 2)
        o = self._halve_rows(o, 2)
        lse = self._halve_rows(lse, 1)
        if cutlass.const_expr(acc_o is not None):
            acc_o = self._halve_rows(acc_o, 3)
            acc_lse = self._halve_rows(acc_lse, 2)
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )
        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1 or q_rope.stride[1] != 1):
            raise ValueError("q_latent and q_rope must have leading dimension 1")
        if cutlass.const_expr(c_latent.stride[1] != 1 or c_rope.stride[1] != 1):
            raise ValueError("c_latent and c_rope must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")

        # When num_heads < M tile, fold up to F = fold_sq_ratio tokens of
        # seq_len_q into the head dimension so M_eff = num_heads * F (≤ M_tile).
        # E.g., H=32, S_q=4 → F=4, M_eff=128, S_q_eff=1
        # E.g., H=32, S_q=8 → F=4, M_eff=128, S_q_eff=2
        # This works because MLA shares KV across all heads/queries independently.
        # Tensor layout: [H, D, S_q, B] → [H*F, D, S_q/F, B]; relies on
        # stride_S == stride_H * H (always true for tensors created by run()).
        if cutlass.const_expr(self.fold_sq):
            F = self.fold_sq_ratio

            def _fold_sq_4d(t):
                return cute.make_tensor(
                    t.iterator,
                    cute.make_layout(
                        (
                            t.shape[0] * F,
                            t.shape[1],
                            t.shape[2] // F,
                            t.shape[3],
                        ),
                        stride=(
                            t.stride[0],
                            t.stride[1],
                            t.stride[2] * F,
                            t.stride[3],
                        ),
                    ),
                )

            q_latent = _fold_sq_4d(q_latent)
            q_rope = _fold_sq_4d(q_rope)
            o = _fold_sq_4d(o)
            # LSE: [H, S_q, B] → [H*F, S_q/F, B]
            lse = cute.make_tensor(
                lse.iterator,
                cute.make_layout(
                    (lse.shape[0] * F, lse.shape[1] // F, lse.shape[2]),
                    stride=(lse.stride[0], lse.stride[1] * F, lse.stride[2]),
                ),
            )

        # mixed-CGA: acc_o / acc_lse are the caller's S-slot views
        # (already row-block re-viewed above).

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self.q_major_mode = OperandMajorMode.K
        self.k_major_mode = OperandMajorMode.K
        self.v_major_mode = OperandMajorMode.MN

        self._setup_attributes()

        cta_group = tcgen05.CtaGroup.TWO
        # the intermediate tensor p is from smem & k-major
        p_major_mode = OperandMajorMode.K
        qk_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_qk_tiler[:2],
        )
        pv_tiled_mma = sm100_utils.make_trivial_tiled_mma(
            self.v_dtype,
            self.v_dtype,
            p_major_mode,
            self.v_major_mode,
            self.acc_dtype,
            cta_group,
            self.mma_pv_tiler[:2],
        )

        cta_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (qk_tiled_mma.thr_id.shape,),
        )

        self.epi_tile = self.mma_pv_tiler[:2]

        q_latent_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.q_dtype,
            (self.iterations_qk_latent * self.load_q_stage),
        )
        q_latent_smem_layout_staged = cute.logical_divide(
            q_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )
        q_rope_smem_layout_staged = sm100_utils.make_smem_layout_a(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.q_dtype,
            self.load_q_stage,
        )

        kc_latent_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_tiler,
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_page_tile_size = min(
            self.page_size, qk_tiled_mma.op.shape_mnk[0] // qk_tiled_mma.thr_id.shape
        )
        kc_latent_smem_layout_staged = cute.logical_divide(
            kc_latent_smem_layout_staged, (None, None, None, self.iterations_qk_latent)
        )

        kc_latent_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (self.mma_qk_tiler[0] // qk_tiled_mma.thr_id.shape, self.mma_qk_tiler[2]),
            self.k_dtype,
            (self.iterations_qk_latent * self.load_k_stage),
        )
        kc_latent_smem_layout_for_tma = cute.tiled_divide(
            kc_latent_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_tiler[2])
        )
        kc_latent_smem_layout_for_tma = cute.logical_divide(
            kc_latent_smem_layout_for_tma, (None, None, None, self.iterations_qk_latent)
        )

        kc_rope_smem_layout_staged = sm100_utils.make_smem_layout_b(
            qk_tiled_mma,
            self.mma_qk_rope_tiler,
            self.k_dtype,
            self.load_k_stage,
        )
        kc_rope_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.K,
            (
                self.mma_qk_rope_tiler[0] // qk_tiled_mma.thr_id.shape,
                self.mma_qk_rope_tiler[2],
            ),
            self.k_dtype,
            (self.iterations_qk_rope * self.load_k_stage),
        )
        kc_rope_smem_layout_for_tma = cute.tiled_divide(
            kc_rope_smem_layout_for_tma, (kc_page_tile_size, self.mma_qk_rope_tiler[2])
        )

        p_smem_layout_staged = sm100_utils.make_smem_layout_a(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.q_dtype,
            (self.iterations_pv_k * self.p_mma_stage),
        )
        p_smem_layout_staged = cute.logical_divide(
            p_smem_layout_staged, (None, None, None, self.iterations_pv_k)
        )

        vc_smem_layout_staged = sm100_utils.make_smem_layout_b(
            pv_tiled_mma,
            self.mma_pv_tiler,
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_staged = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_staged,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        vc_page_tile_size = min(self.page_size, self.mma_pv_tiler[2])
        vc_smem_layout_for_tma = sm100_utils.make_smem_layout(
            OperandMajorMode.MN,
            (self.mma_pv_tiler[1] // pv_tiled_mma.thr_id.shape, self.mma_pv_tiler[2]),
            self.v_dtype,
            (self.iterations_pv_k * self.iterations_pv_n * self.load_v_stage),
        )
        vc_smem_layout_for_tma = cute.tiled_divide(
            vc_smem_layout_for_tma,
            (
                pv_tiled_mma.op.shape_mnk[1] // pv_tiled_mma.thr_id.shape,
                vc_page_tile_size,
            ),
        )
        vc_smem_layout_for_tma = cute.logical_divide(
            cute.logical_divide(
                vc_smem_layout_for_tma,
                (None, None, None, self.iterations_pv_k * self.iterations_pv_n),
            ),
            (None, None, None, (self.iterations_pv_n, None)),
        )
        # TMA load for Q latent and rope
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp(cta_group)

        q_smem_layout = cute.select(q_latent_smem_layout_staged, mode=[0, 1, 2])

        tma_atom_q_latent, tma_tensor_q_latent = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_latent,
            q_smem_layout,
            self.mma_qk_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        q_rope_smem_layout = cute.select(q_rope_smem_layout_staged, mode=[0, 1, 2])
        tma_atom_q_rope, tma_tensor_q_rope = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            q_rope,
            q_rope_smem_layout,
            self.mma_qk_rope_tiler,
            qk_tiled_mma,
            cta_layout_vmnk.shape,
        )
        # TMA load for c latent and k rope
        kc_smem_layout = cute.select(kc_latent_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent, tma_tensor_c_latent = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_latent,
            kc_smem_layout,
            (self.mma_qk_tiler[1], self.mma_qk_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )
        kc_rope_smem_layout = cute.select(kc_rope_smem_layout_for_tma, mode=[0])
        tma_atom_c_rope, tma_tensor_c_rope = self.make_paged_tiled_tma_atom(
            tma_load_op,
            c_rope,
            kc_rope_smem_layout,
            (self.mma_qk_rope_tiler[1], self.mma_qk_rope_tiler[2]),
            qk_tiled_mma,
            is_k_load=True,
        )

        # TMA load for c latent transpose
        vc_smem_layout = cute.select(vc_smem_layout_for_tma, mode=[0])
        tma_atom_c_latent_transpose, tma_tensor_c_latent_transpose = (
            self.make_paged_tiled_tma_atom(
                tma_load_op,
                c_latent_transpose,
                vc_smem_layout,
                (self.mma_pv_tiler[1], self.mma_pv_tiler[2]),
                pv_tiled_mma,
                is_k_load=False,
            )
        )

        q_latent_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        q_rope_copy_size = (
            cute.size_in_bytes(self.q_dtype, q_rope_smem_layout)
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        kc_latent_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_latent_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_latent
        )
        kc_rope_copy_size = (
            cute.size_in_bytes(
                self.k_dtype,
                cute.select(kc_rope_smem_layout_staged, mode=[0, 1, 2]),
            )
            * cute.size(qk_tiled_mma.thr_id.shape)
            * self.iterations_qk_rope
        )
        vc_copy_size = (
            cute.size_in_bytes(
                self.v_dtype, cute.select(vc_smem_layout_staged, mode=[0, 1, 2])
            )
            * cute.size(pv_tiled_mma.thr_id.shape)
            * self.iterations_pv_n
            * self.iterations_pv_k
        )

        self.tma_copy_q_bytes = q_latent_copy_size + q_rope_copy_size
        self.tma_copy_kc_bytes = kc_latent_copy_size + kc_rope_copy_size
        self.tma_copy_vc_bytes = vc_copy_size

        tile_sched_params, grid = self._compute_grid(
            o,
            split_kv,
            self.cluster_shape_mnk,
            self.max_active_clusters,
            self.is_persistent,
        )

        @cute.struct
        class SplitKVKernelSharedStorage:
            # Pipeline barriers
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_clc_stage * 2]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.num_clc_response_bytes // 4],
                16,
            ]
            load_q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_q_stage * 2]
            load_k_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_k_stage * 2]
            load_v_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.load_v_stage * 2]
            mma_s_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_s_stage * 2]
            p_mma_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_mma_stage * 2]
            p_cor_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.p_cor_stage * 2]
            mma_o_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.mma_o_stage * 2]

            # Smem tensors
            smem_p: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, cute.cosize(p_smem_layout_staged)],
                1024,
            ]
            smem_kc_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_latent_smem_layout_staged)
                ],
                1024,
            ]

            smem_kc_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.k_dtype, cute.cosize(kc_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_latent: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_latent_smem_layout_staged)
                ],
                1024,
            ]
            smem_q_rope: cute.struct.Align[
                cute.struct.MemRange[
                    self.q_dtype, cute.cosize(q_rope_smem_layout_staged)
                ],
                1024,
            ]
            smem_vc: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(vc_smem_layout_staged)],
                1024,
            ]
            # 2softmax: doubled so both compute groups can write simultaneously.
            # g0 slots [0, 128); g1 slots [128, 256).
            softmax_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, 2 * self.num_compute_warps * self.threads_per_warp
            ]
            epilogue_smem_exchange: cute.struct.MemRange[
                self.acc_dtype, self.num_compute_warps * self.threads_per_warp
            ]

            # Tmem dealloc cluster barrier
            tmem_dealloc_mbar: cutlass.Int64

            # Tmem holding buffer
            tmem_holding_buf: cutlass.Int32

        softmax_scale_log2 = softmax_scale * LOG2_E

        return (
            qk_tiled_mma,
            pv_tiled_mma,
            tma_atom_q_latent,
            tma_tensor_q_latent,
            tma_atom_q_rope,
            tma_tensor_q_rope,
            tma_atom_c_latent,
            tma_tensor_c_latent,
            tma_atom_c_rope,
            tma_tensor_c_rope,
            tma_atom_c_latent_transpose,
            tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lse_scale,
            q_latent_smem_layout_staged,
            q_rope_smem_layout_staged,
            kc_latent_smem_layout_staged,
            kc_rope_smem_layout_staged,
            p_smem_layout_staged,
            vc_smem_layout_staged,
            kc_latent_smem_layout_for_tma,
            kc_rope_smem_layout_for_tma,
            vc_smem_layout_for_tma,
            cta_layout_vmnk,
        ), SplitKVKernelSharedStorage

    @cute.jit
    def fallback_kernel_body(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: Optional[cute.CopyAtom],
        mQL: cute.Tensor,
        tma_atom_q_rope: Optional[cute.CopyAtom],
        mQR: cute.Tensor,
        tma_atom_c_latent: Optional[cute.CopyAtom],
        mCL: cute.Tensor,
        tma_atom_c_rope: Optional[cute.CopyAtom],
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: Optional[cute.CopyAtom],
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        lse_scale: cutlass.Float32,
        q_latent_smem_layout_staged: cute.ComposedLayout,
        q_rope_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_staged: cute.ComposedLayout,
        kc_rope_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        kc_latent_smem_layout_for_tma: Optional[cute.ComposedLayout],
        kc_rope_smem_layout_for_tma: Optional[cute.ComposedLayout],
        vc_smem_layout_for_tma: Optional[cute.ComposedLayout],
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
        work_q_fdd,
    ):
        """The device split_kv kernel implementation of the Multi-Head Latent Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the MLA computation:
        1. Load warp: Loads Q/C latent/rope data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Compute warps: Compute softmax and do rescaling on accumulators, and store the intermediate/final results
        to global memory

        The kernel produces either intermediate or final results of the MLA computation based on the split_kv parameter.
        When split_kv is 1, the kernel generates the final results directly. Otherwise, it produces intermediate results
        that will later be combined by a reduction kernel.

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases.

        :param tiled_mma_qk: Tiled MMA for Q*K^T
        :type tiled_mma_qk: cute.TiledMma
        :param tiled_mma_pv: Tiled MMA for P*V
        :type tiled_mma_pv: cute.TiledMma
        :param tma_atom_q_latent: TMA copy atom for query latent tensor
        :type tma_atom_q_latent: cute.CopyAtom
        :param mQL: query latent tensor
        :type mQL: cute.Tensor
        :param tma_atom_q_rope: TMA copy atom for query rope tensor
        :type tma_atom_q_rope: cute.CopyAtom
        :param mKR: Compressed rope tensor
        :type mKR: cute.Tensor
        :param tma_atom_c_latent: TMA copy atom for c latent tensor
        :type tma_atom_c_latent: cute.CopyAtom
        :param mCL: Compressed latent tensor
        :type mCL: cute.Tensor
        :param tma_atom_c_rope: TMA copy atom for c rope tensor
        :type tma_atom_c_rope: cute.CopyAtom
        :param mCLT: Compressed latent transpose tensor
        :type mCLT: cute.Tensor
        :param mPT: Page table tensor
        :type mPT: cute.Tensor
        :param mO: Output tensor
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor
        :type mLSE: cute.Tensor
        :param mAccO: Intermediate accumulator output tensor
        :type mAccO: cute.Tensor
        :param mAccLSE: Intermediate accumulator log-sum-exp tensor
        :type mAccLSE: cute.Tensor
        :param split_kv: The split_kv parameter
        :type split_kv: cutlass.Int32
        :param cache_seqs: The variable sequence length tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The per-block split_kv values tensor
        :type block_split_kvs: cute.Tensor
        :param softmax_scale_log2: The log2 scale factor for softmax
        :type softmax_scale_log2: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param q_latent_smem_layout_staged: Shared memory layout for query tensor
        :type q_latent_smem_layout_staged: cute.ComposedLayout
        :param q_rope_smem_layout_staged: Shared memory layout for query rope tensor
        :type q_rope_smem_layout_staged: cute.ComposedLayout
        :param kc_latent_smem_layout_staged: Shared memory layout for key tensor
        :type kc_latent_smem_layout_staged: cute.ComposedLayout
        :param kc_rope_smem_layout_staged: Shared memory layout for key rope tensor
        :type kc_rope_smem_layout_staged: cute.ComposedLayout
        :param p_smem_layout_staged: Shared memory layout for probability matrix
        :type p_smem_layout_staged: cute.ComposedLayout
        :param vc_smem_layout_staged: Shared memory layout for value tensor
        :type vc_smem_layout_staged: cute.ComposedLayout
        :param cta_layout_vmnk: Layout for compute threads
        :type cta_layout_vmnk: cute.Layout
        :param tile_sched_params: Scheduling parameters for work distribution
        :type tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams
        :param SharedStorage: Shared storage for the kernel
        :type SharedStorage: cutlass.Constexpr
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Prefetch tma descriptor
        if warp_idx == self.mma_qk_warp_id:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = cutlass.memory.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # All sixteen warps in each of the two fallback CTAs consume the CLC
        # response. W9 in CTA rank zero also produces the next query.
        shared_clc_pipeline = self.make_and_init_clc_pipeline(
            storage.clc_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            32 * self.threads_per_warp,
        )
        clc_response_ptr = storage.clc_response.data_ptr()

        # Tensor memory dealloc barrier init.
        # TMEM lifetime is owned by mma_pv warp (W11) — the LAST TMEM user.
        # W8 (mma_qk) finishes earlier via mma_s pipeline back-pressure, but W11
        # keeps reading P / writing O until its mma_o.producer_tail. Putting
        # allocate + free on W11 avoids the race where W8 frees TMEM while
        # W11 still has in-flight mma_pv. See OPT#14.
        tmem = cutlass.memory.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_bar,
            allocator_warp_id=self.mma_pv_warp_id,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch_str,
        )

        load_q_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_q_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_q_stage,
            self.tma_copy_q_bytes,
        )
        load_k_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_k_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_k_stage,
            self.tma_copy_kc_bytes,
        )
        load_v_pipeline = self.make_and_init_load_qkv_pipeline(
            storage.load_v_mbar_ptr.data_ptr(),
            cta_layout_vmnk,
            self.load_v_stage,
            self.tma_copy_vc_bytes,
        )
        mma_s_pipeline = self.make_and_init_mma_s_pipeline(
            storage.mma_s_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_mma_pipeline = self.make_and_init_p_mma_pipeline(
            storage.p_mma_mbar_ptr.data_ptr(), cta_layout_vmnk
        )
        p_cor_pipeline = self.make_and_init_p_cor_pipeline(
            storage.p_cor_mbar_ptr.data_ptr()
        )
        mma_o_pipeline = self.make_and_init_mma_o_pipeline(
            storage.mma_o_mbar_ptr.data_ptr(), cta_layout_vmnk
        )

        # Cluster arrive after barrier init
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mnk, is_relaxed=True)

        # Generate smem tensor Q/KC/VC/exchange
        # (MMA, MMA_H, MMA_R, PIPE)
        sQ = storage.smem_q_latent.get_tensor(
            q_latent_smem_layout_staged.outer, swizzle=q_latent_smem_layout_staged.inner
        )
        sQ_rope = storage.smem_q_rope.get_tensor(
            q_rope_smem_layout_staged.outer, swizzle=q_rope_smem_layout_staged.inner
        )
        # (MMA, MMA_K, MMA_R, PIPE)
        sKC = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_staged.outer,
            swizzle=kc_latent_smem_layout_staged.inner,
        )
        sKC_rope = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_staged.outer, swizzle=kc_rope_smem_layout_staged.inner
        )
        sKC_for_tma = storage.smem_kc_latent.get_tensor(
            kc_latent_smem_layout_for_tma.outer,
            swizzle=kc_latent_smem_layout_for_tma.inner,
        )
        sKC_rope_for_tma = storage.smem_kc_rope.get_tensor(
            kc_rope_smem_layout_for_tma.outer, swizzle=kc_rope_smem_layout_for_tma.inner
        )
        # (MMA, MMA_D, MMA_K, PIPE)
        sVC = storage.smem_vc.get_tensor(
            vc_smem_layout_staged.outer, swizzle=vc_smem_layout_staged.inner
        )
        sVC_for_tma = storage.smem_vc.get_tensor(
            vc_smem_layout_for_tma.outer, swizzle=vc_smem_layout_for_tma.inner
        )
        # (MMA, MMA_H, MMA_K)
        sP = storage.smem_p.get_tensor(
            p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner
        )
        # (compute_threads,) — doubled for 2softmax (both groups exchange concurrently).
        softmax_smem_exchange = storage.softmax_smem_exchange.get_tensor(
            cute.make_layout(2 * self.num_compute_warps * self.threads_per_warp)
        )
        epilogue_smem_exchange = storage.epilogue_smem_exchange.get_tensor(
            cute.make_layout(self.num_compute_warps * self.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mnk)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps, including page table and data tensors
        # ///////////////////////////////////////////////////////////////////////////////
        # Note: warp 11 (formerly empty filler) is now mma_pv_warp_id — handled below.

        if warp_idx == self.load_tma_k_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_q_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_q_stage
            )
            load_k_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_k_stage
            )
            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
                tile_sched_params,
                cute.arch.block_idx(),
                cute.arch.grid_dim(),
                clc_response_ptr,
            )
            clc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.ProducerConsumer, 1
            )
            while work_tile.is_valid_tile:
                if is_leader_cta:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(
                        clc_producer_state
                    )
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                # CLC's M coordinate spans four logical rows. A fallback pair
                # maps rows 0/1 to s=0 and rows 2/3 to s=1; M parity is the
                # rank inside the native 2-CTA baseline pair.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Construct fixed common/tma_qk/tma_pv params for load_tma
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_k_pipeline=load_k_pipeline,
                        load_v_pipeline=load_v_pipeline,
                        mPT=mPT,
                    )
                    tma_qk_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        tma_atom_q_latent=tma_atom_q_latent,
                        tma_atom_q_rope=tma_atom_q_rope,
                        tma_atom_c_latent=tma_atom_c_latent,
                        tma_atom_c_rope=tma_atom_c_rope,
                        mQL=mQL,
                        mQR=mQR,
                        mCL=mCL,
                        mKR=mKR,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        sKC=sKC_for_tma,
                        sKC_rope=sKC_rope_for_tma,
                    )
                    # Load tma
                    load_q_producer_state, load_k_producer_state = self.load_tma_qk(
                        tma_common_params,
                        tma_qk_params,
                        k_index,
                        k_tile_count,
                        load_q_producer_state,
                        load_k_producer_state,
                    )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = self.get_clc_work(clc_response_ptr)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            if is_leader_cta:
                clc_pipeline.producer_tail(clc_producer_state)
            load_q_pipeline.producer_tail(load_q_producer_state)
            load_k_pipeline.producer_tail(load_k_producer_state)

        if warp_idx == self.load_tma_v_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            load_v_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.load_v_stage
            )
            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    # Construct fixed common/tma_qk/tma_pv params for load_tma
                    tma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_v_pipeline=load_v_pipeline,
                        mPT=mPT,
                    )
                    tma_pv_params = SimpleNamespace(
                        tiled_mma_pv=tiled_mma_pv,
                        tma_atom_c_latent_transpose=tma_atom_c_latent_transpose,
                        mCLT=mCLT,
                        sVC=sVC_for_tma,
                    )
                    # Load tma
                    load_v_producer_state = self.load_tma_v(
                        tma_common_params,
                        tma_pv_params,
                        k_index,
                        k_tile_count,
                        load_v_producer_state,
                    )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = self.get_clc_work(clc_response_ptr)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            load_v_pipeline.producer_tail(load_v_producer_state)

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA-QK warp (W8): issues all mma_qk, produces S via mma_s pipeline.
        #  Does NOT allocate or free TMEM (W11 owns TMEM lifetime — OPT#14).
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_qk_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            # TMEM allocation done by W11; W8 just waits and retrieves the ptr.
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            load_q_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_q_stage
            )
            load_k_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_k_stage
            )
            mma_s_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_s_stage
            )
            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    mma_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_q_pipeline=load_q_pipeline,
                        load_k_pipeline=load_k_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                    )
                    mma_qk_params = SimpleNamespace(
                        mma_s_pipeline=mma_s_pipeline,
                        sQ=sQ,
                        sQ_rope=sQ_rope,
                        sKC=sKC,
                        sKC_rope=sKC_rope,
                    )
                    (
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        mma_s_producer_state,
                    ) = self.mma(
                        mma_common_params,
                        mma_qk_params,
                        k_tile_count,
                        tiled_mma_qk,
                        load_q_consumer_state,
                        load_k_consumer_state,
                        mma_s_producer_state,
                    )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = self.get_clc_work(clc_response_ptr)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            mma_s_pipeline.producer_tail(mma_s_producer_state)
            # TMEM relinquish/free done by W11 (mma_pv warp, allocator).

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA-PV warp (W11): owns TMEM lifetime. Issues all mma_pv.
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_pv_warp_id:
            cute.arch.setmaxregister_decrease(self.other_reg_num)
            # W11 (mma_pv) owns TMEM lifetime: allocate here, free after the loop.
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            load_v_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.load_v_stage
            )
            p_mma_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_mma_stage
            )
            mma_o_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.mma_o_stage
            )
            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if k_tile_count > 0:
                    mma_pv_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        local_split_kv=local_split_kv,
                        load_v_pipeline=load_v_pipeline,
                        tmem_ptr=tmem_ptr,
                        is_leader_cta=is_leader_cta,
                        L=mCL.shape[1],
                    )
                    mma_pv_only_params = SimpleNamespace(
                        p_mma_pipeline=p_mma_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                        sP=sP,
                        sVC=sVC,
                    )
                    (
                        tiled_mma_pv,
                        load_v_consumer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    ) = self.mma_pv_warp_body(
                        mma_pv_common_params,
                        mma_pv_only_params,
                        k_tile_count,
                        tiled_mma_pv,
                        load_v_consumer_state,
                        p_mma_consumer_state,
                        mma_o_producer_state,
                    )
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = self.get_clc_work(clc_response_ptr)
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()

            mma_o_pipeline.producer_tail(mma_o_producer_state)
            # W11 is the allocator; safe to free now that all mma_pv has retired.
            tmem.relinquish_alloc_permit()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.compute_warp_ids[0]
            and warp_idx <= self.compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if cutlass.const_expr(self.early_compute_clc):
                    # Empty tasks also release their CLC response.
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    next_work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        L=mCL.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=False,
                        )
                    )
                if cutlass.const_expr(self.early_compute_clc):
                    work_tile = next_work_tile
                else:
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp — second group (g1, warps 12-15, odd k-tiles).
        #  2softmax Strategy A: alternates k-tiles with g0; cross-group merges
        #  inside softmax() via softmax_exchange_sync_bar.
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.second_compute_warp_ids[0]
            and warp_idx <= self.second_compute_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.softmax_reg_num)
            mma_s_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_s_stage
            )
            p_mma_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_mma_stage
            )
            p_cor_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            tmem.wait_for_alloc()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            mma_s_consumer_state.advance()
            p_mma_producer_state.advance()
            p_cor_producer_state.advance()
            # Pingpong init-phase trick: g1 pre-arrives softmax_order_bar_0 once
            # so g0's FIRST softmax_order_bar_0.arrive_and_wait() completes
            # without waiting for g1's first loop arrive. Replaces a per-iter
            # is_first_tile guard. Bar is num_total_compute_warps (256 threads)
            # so wait-side 128 + signal-side 128 = 256 → release.
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if cutlass.const_expr(self.early_compute_clc):
                    # Empty tasks also release their CLC response.
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    next_work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=softmax_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        L=mCL.shape[1],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        p_cor_pipeline=p_cor_pipeline,
                    )
                    compute_softmax_params = SimpleNamespace(
                        tiled_mma_qk=tiled_mma_qk,
                        sP=sP,
                        mma_s_pipeline=mma_s_pipeline,
                        p_mma_pipeline=p_mma_pipeline,
                        softmax_scale_log2=softmax_scale_log2,
                    )
                    mma_s_consumer_state, p_mma_producer_state, p_cor_producer_state = (
                        self.compute(
                            compute_common_params,
                            compute_softmax_params,
                            k_index=k_index,
                            k_tile_count=k_tile_count,
                            mma_s_consumer_state=mma_s_consumer_state,
                            p_mma_producer_state=p_mma_producer_state,
                            p_cor_producer_state=p_cor_producer_state,
                            is_second_compute_warp=True,
                        )
                    )
                if cutlass.const_expr(self.early_compute_clc):
                    work_tile = next_work_tile
                else:
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
            # NOTE: g1 skips p_cor_pipeline.producer_tail — g0 already does it.

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.correction_warp_ids[0]
            and warp_idx <= self.correction_warp_ids[-1]
        ):
            cute.arch.setmaxregister_increase(self.correction_reg_num)
            p_cor_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.p_cor_stage
            )
            mma_o_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.mma_o_stage
            )
            # sync with mma warp before retrieving tmem ptr
            tmem.wait_for_alloc()

            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)

            clc_pipeline, work_tile, clc_consumer_state = self.make_clc_consumer(
                shared_clc_pipeline,
            )
            while work_tile.is_valid_tile:
                # Decode the four logical CLC rows into the fallback pair's
                # native rank and the re-viewed s=2 row block.
                _raw = work_tile.tile_idx
                _batch_idx, _query_idx = decode_work_y(
                    _raw[1],
                    cute.size(mO.shape[2]) // 2,
                    work_q_fdd,
                )
                blk_coord = (
                    _raw[0] % 2,
                    2 * _query_idx + _raw[0] // 2,
                    _batch_idx,
                    _raw[2],
                )
                k_index, k_tile_count, local_split_kv = self.get_k_tile_count(
                    split_kv,
                    cache_seqs,
                    block_split_kvs,
                    blk_coord,
                )
                if cutlass.const_expr(self.early_compute_clc):
                    # Empty tasks also release their CLC response.
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    next_work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if k_tile_count > 0:
                    compute_common_params = SimpleNamespace(
                        blk_coord=blk_coord,
                        split_kv=split_kv,
                        local_split_kv=local_split_kv,
                        smem_exchange=epilogue_smem_exchange,
                        mAccO=mAccO,
                        mO=mO,
                        K=cache_seqs[blk_coord[2]],
                        L=mCL.shape[1],
                        H=mQL.shape[0],
                        tmem_ptr=tmem_ptr,
                        tidx=tidx,
                        tiled_mma_pv=tiled_mma_pv,
                        p_cor_pipeline=p_cor_pipeline,
                        mma_o_pipeline=mma_o_pipeline,
                    )
                    compute_epilogue_params = SimpleNamespace(
                        output_scale=output_scale,
                        lse_scale=lse_scale,
                        softmax_scale_log2=softmax_scale_log2,
                        mAccLSE=mAccLSE,
                        mLSE=mLSE,
                    )
                    p_cor_consumer_state, mma_o_consumer_state = self.correction(
                        compute_common_params,
                        compute_epilogue_params,
                        k_tile_count=k_tile_count,
                        p_cor_consumer_state=p_cor_consumer_state,
                        mma_o_consumer_state=mma_o_consumer_state,
                    )
                elif cutlass.const_expr(mAccO is None):
                    _store_empty_output(
                        mO,
                        mLSE,
                        blk_coord[0] * 64,
                        64,
                        blk_coord[1],
                        blk_coord[2],
                        tidx,
                        128,
                    )
                if cutlass.const_expr(self.early_compute_clc):
                    work_tile = next_work_tile
                else:
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = self.get_clc_work(clc_response_ptr)
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
        return


# Preserve the MTP module's class export; run() configures its mixed fallback.
RubinMultiHeadLatentAttentionForwardFP8 = (
    RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo
)
