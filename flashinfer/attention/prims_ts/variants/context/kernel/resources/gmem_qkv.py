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

"""GMEM Q/K/V TensorMaps and per-work-tile coordinates."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.cutlass_dsl import Constexpr
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    StageInfo,
    TaskLocalVariable,
    consumer_work,
)
from dataclasses import dataclass, field
from ..config import FmhaConfig
from ..device.windows import (
    bottom_right_window_tile_start,
    variable_window_cta_min_start,
)
from ..device.paged_kv import load_block_table_row_bounds, load_runtime_seq_len_kv
from ..device.common import resolve_work_tile_coords
from typing import Any, Optional


# ---------------------------------------------------------------------------
# GmemQKVResource -- global memory Q/K/V source (no pipeline)
# ---------------------------------------------------------------------------


@dataclass(kw_only=True)
class GmemQKVResource(MemoryResource):
    """Provides TMA descriptors and per-tile coordinates for Q/K/V loads.

    Consumer side resolves batch/head/seq coordinates from the work tile
    so that downstream SmemQ/SmemKV producer_work can issue TMA loads.
    """

    # Static configuration
    batch_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cfg: Constexpr[FmhaConfig] = field(init=False, default=None)
    cuseqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    cuseqlen_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    head_coord_kv: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_head_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_page_idx_lb: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_page_idx_ub: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_request_begin: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    kv_tile_start: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seq_coord: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seq_coord_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_k: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()
    seqlen_q: Constexpr[TaskLocalVariable] = TaskLocalVariable.uninitialized()

    # Runtime scalars
    block_table_row_stride: int | Int32 = field(init=False, default=0)
    max_seq_len_kv: Optional[Int32 | int] = field(init=False, default=None)
    q_offset_default: int | Int32 = field(init=False, default=0)
    variable_window_q_stride: int | Int32 = field(init=False, default=0)

    # GMEM pointers and tensors
    cum_seqlen_k: cute.Tensor | None = field(init=False, default=None)
    cum_seqlen_q: cute.Tensor | None = field(init=False, default=None)
    seqlens_kv: cute.Pointer | None = field(init=False, default=None)
    tma_k_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_q_desc: cutlass.Pointer | None = field(init=False, default=None)
    tma_v_desc: cutlass.Pointer | None = field(init=False, default=None)
    variable_window_cta_starts: cute.Tensor | None = field(init=False, default=None)
    variable_window_token_starts: cute.Tensor | None = field(init=False, default=None)

    def __init__(
        self,
        tma_q_desc: cutlass.Pointer | None,
        tma_k_desc: cutlass.Pointer | None,
        tma_v_desc: cutlass.Pointer | None,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        q_offset: int | Int32,
        cfg: FmhaConfig,
        seqlens_kv: cute.Pointer | None = None,
        block_table_row_stride: int | Int32 = 0,
        max_seq_len_kv: Int32 | int | None = None,
        variable_window_token_starts: cute.Tensor | None = None,
        variable_window_cta_starts: cute.Tensor | None = None,
        variable_window_q_stride: int | Int32 = 0,
        **kwargs: Any,
    ) -> None:
        """Bind Q/K/V descriptors, optional varlen metadata, and FMHA config."""
        super().__init__(**kwargs)
        self.tma_q_desc = tma_q_desc
        self.tma_k_desc = tma_k_desc
        self.tma_v_desc = tma_v_desc
        self.cum_seqlen_q = cum_seqlen_q
        self.cum_seqlen_k = cum_seqlen_k
        self.q_offset_default = q_offset
        self.seqlens_kv = seqlens_kv
        self.block_table_row_stride = block_table_row_stride
        self.max_seq_len_kv = max_seq_len_kv
        self.variable_window_token_starts = variable_window_token_starts
        self.variable_window_cta_starts = variable_window_cta_starts
        self.variable_window_q_stride = variable_window_q_stride
        self.cfg = cfg
        self.seq_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/K/V tile sequence coordinate.",
        )
        self.head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O head coordinate for the current work tile.",
        )
        self.kv_head_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K/V head coordinate for the current work tile.",
        )
        self.head_coord_kv = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K/V head coordinate mirrored for master FMHA context schedules.",
        )
        self.batch_coord = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Batch coordinate for the current work tile.",
        )
        self.seq_coord_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q/O row coordinate for the current work tile.",
        )
        self.cuseqlen_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q sequence cumulative offset for variable-length FMHA.",
        )
        self.cuseqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K sequence cumulative offset for variable-length FMHA.",
        )
        self.seqlen_q = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Q sequence length for variable-length FMHA.",
        )
        self.seqlen_k = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="K sequence length for variable-length FMHA.",
        )
        self.kv_tile_start = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="First K/V loop tile for sliding-window FMHA.",
        )
        self.kv_request_begin = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Element offset of the request's block-table row.",
        )
        self.kv_page_idx_lb = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="First logical page containing a visible sliding-window key.",
        )
        self.kv_page_idx_ub = TaskLocalVariable(
            dtype=Int32,
            default=Int32(0),
            docs="Inclusive logical-page upper bound for the request.",
        )

    @consumer_work(
        returns=(
            seq_coord,
            head_coord,
            kv_head_coord,
            head_coord_kv,
            batch_coord,
            seq_coord_q,
            cuseqlen_q,
            cuseqlen_k,
            seqlen_q,
            seqlen_k,
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_lb,
            kv_page_idx_ub,
        )
    )
    @cute.jit
    def compute_coords(
        self, stage_info: StageInfo
    ) -> tuple[
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
        Int32,
    ]:
        """Resolve per-tile coordinates from work_tile for downstream use.

        Populates consumer variables with the batch/head/seq coordinates
        that SmemQ, SmemK, and SmemV producer_work methods need for TMA loads.
        GQA: head_coord indexes Q/O heads (h_q), kv_head_coord indexes K/V
        heads (h_kv). For MHA (h_r=1) they are identical.
        """
        seq_coord, head_coord, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )

        kv_head_coord = (head_coord * self.cfg.work_tile_q_heads) // self.cfg.h_r
        seq_coord_q = seq_coord * self.cfg.q_tile_m * self.cfg.work_tile_q_seq_tiles
        head_coord_kv = kv_head_coord
        cuseqlen_q = Int32(0)
        cuseqlen_k = Int32(0)
        seqlen_q = Int32(0)
        seqlen_k = Int32(0)
        window_q_offset = Int32(self.q_offset_default)
        kv_tile_start = Int32(0)
        kv_request_begin = Int32(0)
        kv_page_idx_lb = Int32(0)
        kv_page_idx_ub = Int32(0)
        if cutlass.const_expr(self.cfg.has_varlen):
            if cutlass.const_expr(self.cfg.has_uniform_varlen):
                seqlen_q = Int32(self.cfg.uniform_seq_len_q)
                cuseqlen_q = batch_coord * seqlen_q
            else:
                cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                next_cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
                seqlen_q = next_cuseqlen_q - cuseqlen_q
            if cutlass.const_expr(self.cfg.use_paged_kv):
                # Paged K/V is addressed through a block table rather than a
                # packed token buffer, so it has no cumulative token offset.
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    seqlen_k = Int32(self.cfg.uniform_seq_len_k)
                else:
                    seqlen_k = load_runtime_seq_len_kv(
                        self.seqlens_kv, self.max_seq_len_kv, batch_coord
                    )
            elif cutlass.const_expr(self.cfg.has_uniform_varlen):
                seqlen_k = Int32(self.cfg.uniform_seq_len_k)
                cuseqlen_k = batch_coord * seqlen_k
            else:
                cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord])
                next_cuseqlen_k = Int32(self.cum_seqlen_k[batch_coord + Int32(1)])
                seqlen_k = next_cuseqlen_k - cuseqlen_k
            seq_coord_q = cuseqlen_q + seq_coord_q
            # Each packed request uses its own bottom-right window origin. For
            # mixed causal plans the task manager also derives the request's
            # K-loop extent from these live Q/K lengths.
            window_q_offset = seqlen_k - seqlen_q
            if cutlass.const_expr(
                self.cfg.use_paged_kv and not self.cfg.stages_page_offsets_in_smem
            ):
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    kv_request_begin = batch_coord * Int32(self.block_table_row_stride)
                    kv_page_idx_ub = Int32(self.cfg.max_num_pages_per_seq_kv - 1)
                else:
                    kv_request_begin, kv_page_idx_ub = load_block_table_row_bounds(
                        Int32(self.block_table_row_stride),
                        self.cfg.num_tokens_per_page,
                        seqlen_k,
                        batch_coord,
                    )
        if cutlass.const_expr(self.cfg.kv_tile_start_window_size_left > 0):
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                kv_tile_start = bottom_right_window_tile_start(
                    seq_coord=seq_coord,
                    q_tile_m=self.cfg.q_tile_m,
                    kv_tile_n=self.cfg.seq_tile_n,
                    q_offset=window_q_offset,
                    window_size_left=self.cfg.kv_tile_start_window_size_left,
                )
            else:
                # Preserve the minimal fixed equal-length specialization: its
                # bottom-right offset is statically zero.
                kv_tile_start = cute.math.max(
                    Int32(0),
                    (
                        seq_coord * self.cfg.q_tile_m
                        - self.cfg.kv_tile_start_window_size_left
                    )
                    // self.cfg.seq_tile_n,
                )
        if cutlass.const_expr(
            self.cfg.use_paged_kv and self.cfg.kv_tile_start_window_size_left > 0
        ):
            first_key = cute.math.max(
                Int32(0),
                seq_coord * self.cfg.q_tile_m
                + window_q_offset
                - self.cfg.kv_tile_start_window_size_left,
            )
            kv_page_idx_lb = first_key // Int32(self.cfg.num_tokens_per_page)
        if cutlass.const_expr(self.cfg.has_variable_window):
            min_window_start = variable_window_cta_min_start(
                self.variable_window_cta_starts,
                batch_coord=batch_coord,
                seq_coord=seq_coord,
                q_stride=self.variable_window_q_stride,
                tile_size_q=self.cfg.cta_tiler[0],
            )
            kv_tile_start = min_window_start // self.cfg.kv_tile_n
        return (
            seq_coord,
            head_coord,
            kv_head_coord,
            head_coord_kv,
            batch_coord,
            seq_coord_q,
            cuseqlen_q,
            cuseqlen_k,
            seqlen_q,
            seqlen_k,
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_lb,
            kv_page_idx_ub,
        )

    @consumer_work(
        returns=(
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )
    )
    @cute.jit
    def compute_page_coords(self, stage_info: StageInfo) -> tuple[Int32, Int32, Int32]:
        """Resolve only the coordinates needed by paged-KV prefetch.

        The page-offset warp does not consume Q/head coordinates. Keeping its
        coordinate path narrow avoids materializing the full Q/K/V coordinate
        tuple once per persistent work tile.
        """
        seq_coord, _head_coord, batch_coord = resolve_work_tile_coords(
            self.cfg, stage_info.work_tile.tile_idx
        )

        if cutlass.const_expr(self.cfg.has_uniform_varlen):
            cached_seqlen_kv = Int32(self.cfg.uniform_seq_len_k)
            kv_request_begin = batch_coord * Int32(self.block_table_row_stride)
            kv_page_idx_ub = Int32(self.cfg.max_num_pages_per_seq_kv - 1)
        else:
            cached_seqlen_kv = load_runtime_seq_len_kv(
                self.seqlens_kv, self.max_seq_len_kv, batch_coord
            )
            kv_request_begin, kv_page_idx_ub = load_block_table_row_bounds(
                Int32(self.block_table_row_stride),
                self.cfg.num_tokens_per_page,
                cached_seqlen_kv,
                batch_coord,
            )
        window_q_offset = Int32(self.q_offset_default)
        kv_tile_start = Int32(0)
        if cutlass.const_expr(self.cfg.kv_tile_start_window_size_left > 0):
            if cutlass.const_expr(self.cfg.has_varlen):
                if cutlass.const_expr(self.cfg.has_uniform_varlen):
                    seqlen_q = Int32(self.cfg.uniform_seq_len_q)
                else:
                    cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord])
                    next_cuseqlen_q = Int32(self.cum_seqlen_q[batch_coord + Int32(1)])
                    seqlen_q = next_cuseqlen_q - cuseqlen_q
                window_q_offset = cached_seqlen_kv - seqlen_q
            if cutlass.const_expr(self.cfg.has_varlen or self.cfg.has_q_offset):
                kv_tile_start = bottom_right_window_tile_start(
                    seq_coord=seq_coord,
                    q_tile_m=self.cfg.q_tile_m,
                    kv_tile_n=self.cfg.seq_tile_n,
                    q_offset=window_q_offset,
                    window_size_left=self.cfg.kv_tile_start_window_size_left,
                )
            else:
                kv_tile_start = cute.math.max(
                    Int32(0),
                    (
                        seq_coord * self.cfg.q_tile_m
                        - self.cfg.kv_tile_start_window_size_left
                    )
                    // self.cfg.seq_tile_n,
                )

        return (
            kv_tile_start,
            kv_request_begin,
            kv_page_idx_ub,
        )
