# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Shared MoE persistent scheduler utilities."""

from abc import ABC, abstractmethod
from enum import IntEnum
from typing import List, Optional, Tuple, Literal

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import Boolean, Int32, extract_mlir_values, new_from_mlir_values, const_expr, dsl_user_op
from cutlass._mlir import ir
import cutlass.pipeline as pipeline

# =============================================================================
# Work Tile State
# =============================================================================


class WorkTileState(IntEnum):
    """State encoding for MoEWorkTileInfo via expert_idx sentinel values."""

    DONE = -1  # All work tiles have been processed.


# =============================================================================
# Work Tile Info
# =============================================================================


class MoEWorkTileInfo:
    """CTA-level scheduler tile plus expert_idx sentinel state."""

    BaseFields = 4
    BaseBytes = 16  # 4 * sizeof(Int32)
    TotalFields = BaseFields  # Subclasses override with BaseFields + extra

    def __init__(
        self,
        expert_idx: Int32,  # >=0 valid, or WorkTileState sentinel
        tile_m_idx: Int32,
        tile_n_idx: Int32,
        k_tile_cnt: Int32,
    ):
        self.expert_idx = expert_idx
        self.tile_m_idx = tile_m_idx
        self.tile_n_idx = tile_n_idx
        self.k_tile_cnt = k_tile_cnt

    @property
    def is_valid_tile(self) -> Boolean:
        """Check if this is a valid work tile (expert_idx >= 0)."""
        return self.expert_idx >= Int32(0)


    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = extract_mlir_values(self.expert_idx)
        values.extend(extract_mlir_values(self.tile_m_idx))
        values.extend(extract_mlir_values(self.tile_n_idx))
        values.extend(extract_mlir_values(self.k_tile_cnt))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoEWorkTileInfo":
        assert len(values) == 4
        return MoEWorkTileInfo(
            expert_idx=new_from_mlir_values(self.expert_idx, [values[0]]),
            tile_m_idx=new_from_mlir_values(self.tile_m_idx, [values[1]]),
            tile_n_idx=new_from_mlir_values(self.tile_n_idx, [values[2]]),
            k_tile_cnt=new_from_mlir_values(self.k_tile_cnt, [values[3]]),
        )

    # =========================================================================
    # Serialization layer (subclasses override to_rmem / from_rmem for extra fields)
    # =========================================================================

    def to_rmem(self) -> cute.Tensor:
        """Pack fields into an rmem tensor for vectorized smem copy.
        Subclasses override to include extra fields."""
        rmem = cute.make_rmem_tensor((self.BaseFields,), Int32)
        rmem[0] = self.expert_idx
        rmem[1] = self.tile_m_idx
        rmem[2] = self.tile_n_idx
        rmem[3] = self.k_tile_cnt
        return rmem

    @classmethod
    def from_rmem(cls, rmem: cute.Tensor) -> "MoEWorkTileInfo":
        """Unpack from rmem tensor. Subclasses override to read extra fields."""
        return cls(
            expert_idx=rmem[0],  # type: ignore[arg-type]
            tile_m_idx=rmem[1],  # type: ignore[arg-type]
            tile_n_idx=rmem[2],  # type: ignore[arg-type]
            k_tile_cnt=rmem[3],  # type: ignore[arg-type]
        )

    # =========================================================================
    # Communication layer (handles pipeline + smem transfer)
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def write_to_smem(
        self,
        smem_buf_tensor: cute.Tensor,
        dependency,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Write work tile to smem with full pipeline management.
        dependency = (pipeline, producer_state)."""
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        pipe.producer_acquire(state)
        rmem = self.to_rmem()
        cute.copy(copy_atom, rmem, smem_buf_tensor[(None, state.index)])
        cute.arch.fence_proxy("async.shared", space="cta")
        pipe.producer_commit(state)
        state.advance()

    @classmethod
    @dsl_user_op
    @cute.jit
    def read_from_smem(
        cls,
        smem_buf_tensor: cute.Tensor,
        dependency,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEWorkTileInfo":
        """Read work tile from smem with full pipeline management.
        dependency = (pipeline, consumer_state)."""
        pipe, state = dependency
        copy_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(), cutlass.Int32, num_bits_per_copy=128
        )
        pipe.consumer_wait(state)
        rmem = cute.make_rmem_tensor((cls.TotalFields,), Int32)
        cute.copy(copy_atom, smem_buf_tensor[(None, state.index)], rmem)
        work = cls.from_rmem(rmem)
        cute.arch.fence_acq_rel_cta()
        pipe.consumer_release(state)
        state.advance()
        return work


# =============================================================================
# Scheduler Extension — Base
# =============================================================================


class MoESchedExtension:
    """
    Base class for MoE scheduler extensions.

    Bridges the scheduler with tensor-level domain conversion and TMA
    descriptor management. Each kernel type (grouped_mm, scaled_grouped_mm,
    etc.) provides its own subclass with:

    - WorkTileInfo subclass with extra precomputed fields (token_offset, etc.)
    - enrich_work_tile_info(): called by scheduler to populate extra fields
    - get_gmem_tensor(): called by consumer warps for domain conversion
    - prefetch_for_expert(): called by scheduler warp for TMA desc prefetch

    This base class provides identity/no-op defaults so the scheduler can
    always call through self._ext without None checks.
    """

    WorkTileInfo = MoEWorkTileInfo

    def __init__(self, workspace=None):
        self.workspace = workspace

    def enrich_work_tile_info(self, base_work: MoEWorkTileInfo) -> MoEWorkTileInfo:
        """Enrich base work tile with domain-specific extra fields.

        Default: identity (returns base_work unchanged).
        Subclasses override to compute and attach extra fields such as
        token_offset, tokens_i, padded_token_offset, padded_tokens_i.
        Must handle invalid tiles (return enriched type with dummy extras).
        """
        return base_work

    def get_gmem_tensor(self, tensor_name, gmem_tensor_in_moe_view, work_tile_info):
        """Convert MoE-view tensor to per-expert tensor for domain conversion.

        Subclasses must override. Reads precomputed extra fields directly
        from work_tile_info instead of recomputing from offs.
        """
        raise NotImplementedError(
            "Subclass must implement get_gmem_tensor for domain conversion"
        )

    def prefetch_for_expert(self, expert_idx: Int32) -> None:
        """Prefetch expert-wise TMA descriptors. Default: no-op."""
        pass


_DEFAULT_SCHED_EXT = MoESchedExtension()


# =============================================================================
# Scheduler Parameters — Base
# =============================================================================


class MoESchedulerParamsBase(ABC):
    """
    Abstract base class for MoE tile scheduler parameters.

    Uses unified semantics for both scenarios:
    - expert_shape: (expert_cnt, intermediate, hidden)

    For 2Dx3D: GEMM is (M=tokens_i, N=intermediate, K=hidden) per expert
    For 2Dx2D: GEMM is (M=hidden, N=intermediate, K=tokens_i) per expert

    ``intermediate`` is the scheduler's per-expert full axis.  Concrete
    kernels may give that axis a narrower meaning.  For example, the current
    fused fc12 swap-AB inference kernel binds it to ``intermediate_gateup``
    (gate + up concatenated); a future non-swap or training kernel should
    document its own interpretation at its params layer instead of changing
    this shared base contract.

    Tile hierarchy:
    - cta_tile_shape_mnk: Single CTA tile shape (tile_m, tile_n, tile_k)
    - cluster_shape_mn: CTAs per cluster (cluster_m, cluster_n)
    - cluster_tile_shape_mn: Cluster tile shape = cta_tile_shape * cluster_shape

    This class is used both on host (for grid shape calculation) and on device
    (stored in scheduler). Codegen-time constants (scenario, cta_tile_shape_mnk,
    cluster_shape_mn, num_sched_stages, is_swap_ab) are NOT serialized to MLIR
    values.

    Coordinate convention:
        The scheduler body uses an internal M-slot for the axis grouped by
        ``offs`` and an internal N-slot for the per-expert full axis.  Callers
        still pass tile and cluster shapes in GEMM-domain order.  When
        ``is_swap_ab`` is true, the constructor swaps M/N once on entry and
        concrete decoders swap tile indices back on exit, so consumers always
        see GEMM-domain ``tile_m_idx`` / ``tile_n_idx``.
    """

    DEFAULT_NUM_SCHED_STAGES = 2

    def __init__(
        self,
        scenario: Literal["2Dx3D", "2Dx2D"],
        expert_shape: Tuple[
            int | Int32, int | Int32, int | Int32
        ],  # (expert_cnt, intermediate, hidden)
        cta_tile_shape_mnk: Tuple[int, int, int],  # (tile_m, tile_n, tile_k)
        cluster_shape_mn: Tuple[int, int],  # (cluster_m, cluster_n)
        override_num_stages: Optional[int] = None,
        is_swap_ab: bool = False,
    ):
        if is_swap_ab and scenario == "2Dx2D":
            # Weight-grad path is an entirely different problem shape and is
            # not in v1 scope for swap-AB.  Reject loudly rather than silently
            # producing nonsense work tiles.
            raise ValueError(
                "is_swap_ab=True is incompatible with scenario='2Dx2D' "
                "(weight-grad path); v1 only supports forward 2Dx3D swap-AB."
            )

        self.scenario = scenario
        self.is_swap_ab = is_swap_ab
        e, i, h = expert_shape
        # Preserve Python ints as codegen-time constants; Int32 stays runtime.
        self.expert_cnt = e
        self.intermediate = i
        self.hidden = h

        # When is_swap_ab is True, the user supplies tuples in GEMM-domain
        # (M, N, K) order but the scheduler body uses "grouped-axis-as-M"
        # internally; swap once here so the rest of the body stays oblivious.
        if is_swap_ab:
            cta_tile_shape_mnk = (
                cta_tile_shape_mnk[1],
                cta_tile_shape_mnk[0],
                cta_tile_shape_mnk[2],
            )
            cluster_shape_mn = (cluster_shape_mn[1], cluster_shape_mn[0])
        self.cta_tile_shape_mnk = cta_tile_shape_mnk
        self.cluster_shape_mn = cluster_shape_mn

        self.num_sched_stages = (
            override_num_stages
            if override_num_stages is not None
            else self.DEFAULT_NUM_SCHED_STAGES
        )
        if self.num_sched_stages <= 0:
            raise ValueError(
                f"num_sched_stages must be positive, got {self.num_sched_stages}"
            )

    @property
    def cluster_tile_m(self) -> int:
        """Cluster tile size along M = cta_tile_m * cluster_m."""
        return self.cta_tile_shape_mnk[0] * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        """Cluster tile size along N = cta_tile_n * cluster_n."""
        return self.cta_tile_shape_mnk[1] * self.cluster_shape_mn[1]

    @property
    def cta_tile_k(self) -> int:
        """CTA tile size along K (same as cluster since cluster_k = 1)."""
        return self.cta_tile_shape_mnk[2]

    @abstractmethod
    def get_scheduler_type(self) -> type:
        """Return the concrete scheduler class bound to this params type."""
        ...

    @abstractmethod
    def get_grid_shape(
        self,
        max_active_clusters: int,
    ) -> Tuple[int, int, int]:
        """Compute grid shape for kernel launch."""
        ...


# =============================================================================
# Scheduler — Base (Device-side)
# =============================================================================


class MoESchedulerBase(ABC):
    """
    Abstract base class for MoE persistent tile schedulers.

    Provides shared tile iteration helpers that convert linear cluster indices
    to (expert_idx, tile_m_idx, tile_n_idx, k_tile_cnt). Subclasses implement
    gen_next_work() to define how work is produced.

    Required members (set by concrete __init__ / create()):
        params: MoESchedulerParamsBase
        offs: cute.Tensor           — (experts,) cumsum of token counts
        _ext: MoESchedExtension     — extension (Python ref, not MLIR-serialized)
        cta_id_in_cluster: cute.Coord
        current_expert_idx: Int32
        expert_tile_start: Int32
        expert_tile_end: Int32
        current_work: MoEWorkTileInfo (or subclass defined by _ext.WorkTileInfo)

    SMEM communication members (set by concrete create()):
        _pipeline                   — PipelineAsync for work tile broadcast
        _smem_buf_tensor            — SMEM tensor for work tile stages
        _num_sched_stages: int      — number of pipeline stages
        _producer_state             — pipeline producer state (MLIR-serialized)
    """

    # =========================================================================
    # Abstract interface
    # =========================================================================

    @abstractmethod
    def gen_next_work(self) -> None:
        """Advance internal state to the next work tile. Sets self.current_work."""
        ...

    # =========================================================================
    # SMEM communication (shared by all scheduler subclasses)
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def publish_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Write current_work to SMEM and advance the producer pipeline."""
        self.current_work.write_to_smem(
            self._smem_buf_tensor,
            (self._pipeline, self._producer_state),
            loc=loc,
            ip=ip,
        )

    @dsl_user_op
    @cute.jit
    def produce_tail(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Ensure all published stages are fully consumed, then release resources."""
        self._pipeline.producer_tail(self._producer_state)

    def make_consumer(self) -> "MoESchedConsumer":
        """Create a consumer handle for non-scheduler warps."""
        return MoESchedConsumer(
            self._pipeline,
            self._smem_buf_tensor,
            self._num_sched_stages,
            work_tile_cls=self._ext.WorkTileInfo,
        )

    # =========================================================================
    # Convenience accessors for params
    # =========================================================================

    @property
    def scenario(self) -> Literal["2Dx3D", "2Dx2D"]:
        return self.params.scenario

    @property
    def expert_cnt(self) -> Int32:
        return self.params.expert_cnt

    @property
    def intermediate(self) -> Int32:
        return self.params.intermediate

    @property
    def hidden(self) -> Int32:
        return self.params.hidden

    @property
    def cta_tile_shape_mnk(self) -> Tuple[int, int, int]:
        return self.params.cta_tile_shape_mnk

    @property
    def cluster_shape_mn(self) -> Tuple[int, int]:
        """Cluster shape used to size cta_id_in_cluster."""
        return self.params.cluster_shape_mn

    @property
    def cluster_tile_m(self) -> int:
        """Tile-partitioning granularity along M."""
        return self.params.cluster_tile_m

    @property
    def cluster_tile_n(self) -> int:
        """Tile-partitioning granularity along N."""
        return self.params.cluster_tile_n

    @property
    def cta_tile_k(self) -> int:
        return self.params.cta_tile_k

    # =========================================================================
    # Shared tile iteration helpers
    # =========================================================================

    @dsl_user_op
    @cute.jit
    def _get_work_tile_for_linear_idx(
        self,
        cluster_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """
        Convert a linear cluster index to MoEWorkTileInfo.

        Uses cached expert tracking state for O(1) fast path when staying
        within the same expert. Advances expert state when needed.

        Returns an invalid tile (expert_idx = -1) if cluster_linear_idx is out of range.
        """
        self._advance_expert_to_contain(cluster_linear_idx, loc=loc, ip=ip)

        is_valid = self.current_expert_idx < self.expert_cnt

        work_tile_info = MoEWorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            k_tile_cnt=Int32(0),
        )

        if is_valid:
            local_idx = cluster_linear_idx - self.expert_tile_start
            cluster_tile_m_idx, cluster_tile_n_idx = self._decompose_local_idx(
                local_idx, self.current_expert_idx, loc=loc, ip=ip
            )

            cta_tile_m_idx = (
                cluster_tile_m_idx * self.cluster_shape_mn[0]
                + self.cta_id_in_cluster[0]  # type: ignore[index]
            )
            cta_tile_n_idx = (
                cluster_tile_n_idx * self.cluster_shape_mn[1]
                + self.cta_id_in_cluster[1]  # type: ignore[index]
            )

            k_tile_cnt = self._compute_k_tile_cnt(
                self.current_expert_idx, loc=loc, ip=ip
            )

            # Swap-AB: re-express the tile indices in GEMM-domain (M, N) on
            # the way out — see MoESchedulerParamsBase docstring.  The body
            # above produced "scheduler-internal M-slot / N-slot" indices
            # (M = grouped axis, N = full axis); when is_swap_ab is True the
            # GEMM-domain mapping is flipped.  Codegen-time const_expr makes
            # this a zero-cost branch.
            if const_expr(self.params.is_swap_ab):
                cta_tile_m_idx, cta_tile_n_idx = cta_tile_n_idx, cta_tile_m_idx

            work_tile_info = MoEWorkTileInfo(
                expert_idx=self.current_expert_idx,
                tile_m_idx=cta_tile_m_idx,
                tile_n_idx=cta_tile_n_idx,
                k_tile_cnt=k_tile_cnt,
            )
        return work_tile_info

    @dsl_user_op
    @cute.jit
    def _advance_expert_to_contain(
        self,
        cluster_linear_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """
        Advance expert tracking state until current expert contains cluster_linear_idx,
        or we run out of experts.

        Fast path: If already in correct expert, no work needed.
        """
        if self.expert_tile_end == Int32(0):
            tiles_for_expert_0 = self._compute_tiles_for_expert(
                Int32(0), loc=loc, ip=ip
            )
            self.expert_tile_end = tiles_for_expert_0

        while (
            cluster_linear_idx >= self.expert_tile_end
            and self.current_expert_idx < self.expert_cnt
        ):
            self.current_expert_idx = self.current_expert_idx + 1
            self.expert_tile_start = self.expert_tile_end

            if self.current_expert_idx < self.expert_cnt:
                tiles_for_expert = self._compute_tiles_for_expert(
                    self.current_expert_idx, loc=loc, ip=ip
                )
                self.expert_tile_end = self.expert_tile_end + tiles_for_expert

    @dsl_user_op
    @cute.jit
    def _compute_tiles_for_expert(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Compute total cluster tiles for a given expert.

        Uses self.cluster_tile_m / self.cluster_tile_n — see the properties'
        docstrings for the preferred vs actual distinction (relevant once
        mix CGA lands).
        """
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt
        else:  # 2Dx3D
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx - 1]  # type: ignore[operator]
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
            return cluster_tile_m_cnt * cluster_tile_n_cnt

    @dsl_user_op
    @cute.jit
    def _decompose_local_idx(
        self,
        local_idx: Int32,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32]:
        """
        Decompose local cluster tile index within expert to (cluster_tile_m_idx, cluster_tile_n_idx).

        Uses short-side-first raster; outputs preferred-granularity indices.
        """
        cluster_tile_m_cnt, cluster_tile_n_cnt = self._get_cluster_tile_counts(
            expert_idx, loc=loc, ip=ip
        )
        cluster_tile_m_idx = -1
        cluster_tile_n_idx = -1

        if cluster_tile_m_cnt <= cluster_tile_n_cnt:
            cluster_tile_m_idx = local_idx % cluster_tile_m_cnt
            cluster_tile_n_idx = local_idx // cluster_tile_m_cnt
        else:
            cluster_tile_n_idx = local_idx % cluster_tile_n_cnt
            cluster_tile_m_idx = local_idx // cluster_tile_n_cnt

        return (cluster_tile_m_idx, cluster_tile_n_idx)

    @dsl_user_op
    @cute.jit
    def _get_cluster_tile_counts(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Tuple[Int32, Int32]:
        """Get (cluster_tile_m_cnt, cluster_tile_n_cnt) for a given expert.

        Uses self.cluster_tile_m / self.cluster_tile_n — see their docstrings
        for the preferred vs actual distinction.
        """
        if const_expr(self.scenario == "2Dx2D"):
            cluster_tile_m_cnt = (
                self.hidden + self.cluster_tile_m - 1
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
        else:  # 2Dx3D
            tokens_i = self.offs[expert_idx]
            if expert_idx > 0:
                tokens_i = tokens_i - self.offs[expert_idx - 1]  # type: ignore[operator]
            cluster_tile_m_cnt = (
                tokens_i + self.cluster_tile_m - 1  # type: ignore[operator]
            ) // self.cluster_tile_m
            cluster_tile_n_cnt = (
                self.intermediate + self.cluster_tile_n - 1
            ) // self.cluster_tile_n
        return (cluster_tile_m_cnt, cluster_tile_n_cnt)

    @dsl_user_op
    @cute.jit
    def _compute_k_tile_cnt(
        self,
        expert_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """
        Compute the number of K tiles for this expert.

        2Dx3D: K = hidden (fixed) -> k_tile_cnt = ceil(hidden / cta_tile_k)
        2Dx2D: K = tokens_i (variable) -> k_tile_cnt = ceil(tokens_i / cta_tile_k)
        """
        if const_expr(self.scenario == "2Dx3D"):
            return (self.hidden + self.cta_tile_k - 1) // self.cta_tile_k
        else:  # 2Dx2D
            tokens_i = self.offs[expert_idx]
            if expert_idx > cutlass.Int32(0):
                tokens_i = tokens_i - self.offs[expert_idx - 1]  # type: ignore[operator]
            return (tokens_i + self.cta_tile_k - 1) // self.cta_tile_k  # type: ignore[return-value, operator]


# =============================================================================
# Scheduler Consumer Handle
# =============================================================================


class MoESchedConsumer:
    """
    Consumer handle for non-scheduler warps to read work tiles from SMEM.

    Each consumer warp creates its own instance (via scheduler.make_consumer())
    inside its warp_idx branch, giving each warp independent pipeline state.

    The work_tile_cls parameter determines which WorkTileInfo subclass is used
    for deserialization, enabling polymorphic extra fields defined by the
    MoESchedExtension.
    """

    def __init__(
        self,
        sched_pipeline,
        smem_buf_tensor: cute.Tensor,
        num_stages: int,
        work_tile_cls=MoEWorkTileInfo,
    ):
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._work_tile_cls = work_tile_cls
        self._consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, num_stages
        )

    def __extract_mlir_values__(self) -> List[ir.Value]:
        values = []
        values.extend(extract_mlir_values(self._consumer_state))
        return values

    def __new_from_mlir_values__(self, values: List[ir.Value]) -> "MoESchedConsumer":
        cs_len = len(extract_mlir_values(self._consumer_state))
        new_cs = new_from_mlir_values(self._consumer_state, values[:cs_len])
        new_obj = MoESchedConsumer.__new__(MoESchedConsumer)
        new_obj._pipeline = self._pipeline
        new_obj._smem_buf_tensor = self._smem_buf_tensor
        new_obj._work_tile_cls = self._work_tile_cls
        new_obj._consumer_state = new_cs
        return new_obj

    @dsl_user_op
    @cute.jit
    def consume_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """Read the next work tile from SMEM. Blocks until data is available."""
        return self._work_tile_cls.read_from_smem(
            self._smem_buf_tensor,
            (self._pipeline, self._consumer_state),
            loc=loc,
            ip=ip,
        )
