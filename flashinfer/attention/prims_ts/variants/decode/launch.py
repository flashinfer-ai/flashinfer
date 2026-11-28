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

"""Decode launch specs, compilation, workspace layout and runtime launch.

Builds go through ``compile_cache.build_and_load_attention_kernel``; see its
module docstring for how concurrent builds share the cache.
"""

import functools
import math
import numbers
import torch
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from ..block_sparse.common import num_sparse_pattern_heads
from ...common.attention_args import validate_layout
from ...common.paged_kv import (
    PagedKVCache,
    PagedKVScaleFactors,
    normalize_native_paged_kv_cache,
    normalize_paged_kv_scale_factors,
)
from ...common.scales import GlobalScale, fake_scale, normalize_scale
from ...common.tensors import (
    attention_dtype_key,
    cutlass_dtype,
    resolve_cuda_device,
    validate_runtime_device,
)
from ...common.workspace import (
    WORKSPACE_ALIGNMENT,
    WorkspaceSection,
    align_up,
    append_workspace_section,
    workspace_section_view,
)
from ...compile_cache import build_and_load_attention_kernel
from .validation import (
    decode_output_shape,
    read_packed_q_plan_metadata,
    validate_decode_policy_kv_tile_size,
    validate_decode_query_head_extent,
    validate_head_geometry,
    validate_max_kv_len,
    validate_out,
    validate_q,
    validate_seq_len_q,
    validate_storage_page_size,
    validate_window_left,
)
from flashinfer.utils import get_device_sm_count
from typing import Literal, Optional, Union, cast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel.config import FmhaDecodeConfig


COMPILE_OPTIONS = "--enable-tvm-ffi --opt-level 3"


@dataclass(frozen=True)
class DecodeLaunchSpec:
    """Automatic policy and scratch geometry for one planned shape."""

    config: "FmhaDecodeConfig"
    max_active_clusters: int
    policy: tuple[tuple[str, object], ...]
    scratch_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]
    requested_splits: Optional[int] = None
    # Split-counter entries reserved at the buffer tail (at least two device
    # waves), so plans sharing one buffer keep counters at the same bytes.
    scratch_reserve: int = 0


@dataclass(frozen=True)
class DecodeRuntime:
    """Validated runtime tensors and scalar arguments for one launch."""

    q: torch.Tensor
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    k_sf_cache: torch.Tensor
    v_sf_cache: torch.Tensor
    out: torch.Tensor
    num_physical_pages: int
    k_page_stride: int
    k_head_stride: int
    k_token_stride: int
    v_page_stride: int
    v_head_stride: int
    v_token_stride: int
    k_sf_page_stride: int
    v_sf_page_stride: int
    k_sf_head_stride: int
    k_sf_token_stride: int
    v_sf_head_stride: int
    v_sf_token_stride: int
    softmax_scale: GlobalScale
    output_scale: GlobalScale


@dataclass(frozen=True)
class DecodeWorkspaceLayout:
    """Private FMHA scratch layout; only ``total_bytes`` is public."""

    partial_o: WorkspaceSection
    partial_stats: WorkspaceSection
    split_kv_counter: WorkspaceSection
    cu_seqlens_q: WorkspaceSection
    attention_sinks: WorkspaceSection
    uses_split_kv: bool
    total_bytes: int


@dataclass(frozen=True)
class DecodeWorkspaceViews:
    """Typed zero-copy views bound to one validated workspace buffer."""

    partial_o: torch.Tensor
    partial_stats: torch.Tensor
    split_kv_counter: torch.Tensor
    cu_seqlens_q: torch.Tensor
    attention_sinks: torch.Tensor


@dataclass(frozen=True)
class DirectQ1CompileSpec:
    """Sparse format identity; index and storage extents are runtime operands."""

    position_dtype_key: str
    max_seq_len_kv: int = field(compare=False, hash=False)
    block_topk: int = field(compare=False, hash=False)
    kv_block_size: int
    page_capacity: int = field(compare=False, hash=False)
    count_current_block_in_topk: bool = True
    fixed_query_len: int = field(default=0, compare=False, hash=False)
    fixed_query_layout: bool = field(init=False)

    # Key fixed versus packed metadata layout, without keying the Q token count.
    def __post_init__(self):
        object.__setattr__(self, "fixed_query_layout", self.fixed_query_len > 0)


@dataclass(frozen=True)
class DecodeCompileSpec:
    """Batch-independent static identity for one compiled decode callable."""

    device_index: int
    config_items: tuple[tuple[str, object], ...]
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    page_size: int
    q_dtype_key: str
    k_dtype_key: str
    v_dtype_key: str
    out_dtype_key: str
    packed_query: bool
    max_active_clusters: int
    kv_prefix_mode: Literal["dynamic", "planned_full"]
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"]
    direct_q1_spec: Optional[DirectQ1CompileSpec] = None
    runtime_kv_lengths: bool = False
    scale_types: tuple[bool, bool] = (False, False)


def bind_decode_workspace(
    workspace_buffer: torch.Tensor, layout: DecodeWorkspaceLayout
) -> DecodeWorkspaceViews:
    # Anchor the zero-at-rest state (Q offsets, sinks, counters) to the buffer
    # end so plans sharing one buffer keep it at the same bytes; partials stay
    # at offset 0.
    shift = max(
        workspace_buffer.numel() * workspace_buffer.element_size() - layout.total_bytes,
        0,
    )
    shift -= shift % WORKSPACE_ALIGNMENT

    def state_view(section: WorkspaceSection) -> torch.Tensor:
        return workspace_section_view(
            workspace_buffer,
            replace(section, byte_offset=section.byte_offset + shift),
        )

    return DecodeWorkspaceViews(
        partial_o=workspace_section_view(workspace_buffer, layout.partial_o),
        partial_stats=workspace_section_view(workspace_buffer, layout.partial_stats),
        split_kv_counter=state_view(layout.split_kv_counter),
        cu_seqlens_q=state_view(layout.cu_seqlens_q),
        attention_sinks=state_view(layout.attention_sinks),
    )


def decode_launch_spec_from_config(
    cfg: "FmhaDecodeConfig",
    *,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
    max_active_clusters: int,
    requested_splits: Optional[int] = None,
) -> DecodeLaunchSpec:
    """Derive policy and scratch geometry from one finalized FMHA config."""

    scratch_shapes = decode_scratch_shapes(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
        requested_splits=requested_splits,
    )
    # Every plan, split or not, reserves counters for two device-wide waves of
    # logical Q CTAs, so plans sharing one buffer keep the same state bytes.
    # Larger counts keep their exact size; outside the sparse route (own slice),
    # only separate-reduction plans reach them, and those never read the counters.
    scratch_reserve = max(2 * int(max_active_clusters), math.prod(scratch_shapes[2]))

    return DecodeLaunchSpec(
        config=cfg,
        max_active_clusters=int(max_active_clusters),
        policy=decode_policy_from_config(cfg),
        scratch_shapes=scratch_shapes,
        scratch_reserve=scratch_reserve,
    )


def decode_policy_from_config(
    config: "FmhaDecodeConfig",
) -> tuple[tuple[str, object], ...]:
    """Return the stable private policy record for one resolved FMHA config."""

    seq_len_q = int(config.max_seq_len_q)
    uses_packed_q = bool(config.use_variable_seqlens_q)
    query_layout = (
        "TOTAL_Q_Hq_D"
        if uses_packed_q
        else ("B_Hq_D" if seq_len_q == 1 else "B_SQ_Hq_D")
    )
    return (
        ("seq_len_q", seq_len_q),
        ("max_q_len", seq_len_q),
        ("packed_query", uses_packed_q),
        ("query_layout", query_layout),
        ("output_layout", query_layout),
        (
            "window_left",
            int(config.attention_window_size) - 1
            if config.use_sliding_window_causal
            else -1,
        ),
        (
            "mma_variant",
            "keeps_mma_ab" if config.use_keeps_mma_ab else "swaps_mma_ab",
        ),
        ("tile_size_q", int(config.tile_size_q)),
        ("tile_size_kv", int(config.tile_size_kv)),
        ("num_insts_kv", int(config.num_insts_kv)),
        ("use_split_kv", bool(config.use_split_kv)),
        ("splits_kv", int(config.splits_kv)),
        ("max_splits_kv", int(config.max_splits_kv)),
        (
            "use_separate_reduction_kernel",
            bool(config.use_separate_reduction_kernel),
        ),
        ("use_cluster_smem_reduction", bool(config.use_cluster_smem_reduction)),
        ("use_persistent_scheduler", bool(config.use_persistent_scheduler)),
        ("groups_tokens_heads_q", bool(config.groups_tokens_heads_q)),
    )


def decode_scratch_shapes(
    cfg: "FmhaDecodeConfig",
    *,
    batch_size,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len_q: int,
    requested_splits: Optional[int] = None,
):
    """Return split-workspace shapes for a static config and batch extent."""

    if not cfg.use_split_kv:
        return (
            (1, 1, 1, 1, 1),
            (1, 1, 1, 1, 2),
            (1, 1, 1),
        )

    from .kernel.config import make_q_tile_geometry

    head_ratio = num_qo_heads // num_kv_heads
    geometry = make_q_tile_geometry(
        rows_per_cta=cfg.tile_size_q,
        heads_q_per_kv=head_ratio,
        groups_tokens_heads_q=cfg.groups_tokens_heads_q,
    )
    if isinstance(seq_len_q, int):
        num_q_groups = max(int(geometry.num_q_ctas(seq_len_q)), 1)
    elif cfg.groups_tokens_heads_q:
        import cutlass.cute as cute

        num_q_groups = cute.sym_int()
    else:
        num_q_groups = seq_len_q * geometry.head_ctas_per_token
    partial_o_shape: tuple[int, ...]
    if cfg.uses_compact_split_grid:
        # Partials are packed by the launch fanout: slot (b * Hkv + h) *
        # requested + split. The slot count is a runtime extent, so the
        # compiled kernel does not depend on the plan's batch or fanout.
        if requested_splits is None:
            import cutlass.cute as cute

            num_slots = cute.sym_int()
        else:
            num_slots = batch_size * num_kv_heads * int(requested_splits)
        partial_o_shape = (num_slots, head_ratio * seq_len_q, head_dim)
    else:
        partial_o_shape = (
            batch_size,
            num_kv_heads,
            int(cfg.max_splits_kv),
            head_ratio * seq_len_q,
            head_dim,
        )
    partial_stats_shape = (
        partial_o_shape[:-1]
        if cfg.use_separate_reduction_kernel
        else partial_o_shape[:-1] + (2,)
    )
    counter_shape = (batch_size, num_kv_heads, num_q_groups)
    return partial_o_shape, partial_stats_shape, counter_shape


@functools.cache
def get_compiled_decode(
    compile_spec: DecodeCompileSpec,
):
    """Compile and cache one batch-dynamic TS decode topology."""

    device_index = compile_spec.device_index
    num_qo_heads = compile_spec.num_qo_heads
    num_kv_heads = compile_spec.num_kv_heads
    head_dim = compile_spec.head_dim
    q_dtype_key = compile_spec.q_dtype_key
    k_dtype_key = compile_spec.k_dtype_key
    v_dtype_key = compile_spec.v_dtype_key
    out_dtype_key = compile_spec.out_dtype_key
    # Both layouts retain the flattened tensor ABI; only the offset source differs.
    packed_query = True
    uniform_q = not compile_spec.packed_query
    max_active_clusters = compile_spec.max_active_clusters
    kv_prefix_mode = compile_spec.kv_prefix_mode
    kv_lengths_mode = compile_spec.kv_lengths_mode

    if kv_prefix_mode not in ("dynamic", "planned_full"):
        raise ValueError(f"unsupported KV-prefix compile mode {kv_prefix_mode!r}")
    if kv_lengths_mode not in ("dynamic", "planned_uniform_max"):
        raise ValueError(f"unsupported KV-length compile mode {kv_lengths_mode!r}")
    static_full_split_prefix = kv_prefix_mode == "planned_full"
    static_native_uniform_kv = kv_lengths_mode == "planned_uniform_max"

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_drv

    from .kernel.config import FmhaDecodeConfig
    from .kernel.kernel import launch_decode_kernel
    from .kernel.device.direct_sparse_metadata import DirectSparseMetadataView
    from .kernel.device.q_rows import UniformQueryOffsets

    seq_len_q = cute.sym_int()
    direct_q1_spec = compile_spec.direct_q1_spec

    q_dtype = cutlass_dtype(q_dtype_key)
    k_dtype = cutlass_dtype(k_dtype_key)
    v_dtype = cutlass_dtype(v_dtype_key)
    out_dtype = cutlass_dtype(out_dtype_key)
    cfg = FmhaDecodeConfig(**dict(compile_spec.config_items))
    k_storage_dtype = cutlass.Uint8 if k_dtype == cutlass.Float4E2M1FN else k_dtype
    v_storage_dtype = cutlass.Uint8 if v_dtype == cutlass.Float4E2M1FN else v_dtype
    storage_page_size = int(cfg.effective_storage_tokens_per_page)
    partial_dtype = out_dtype
    if cfg.use_separate_reduction_kernel and out_dtype in (
        cutlass.BFloat16,
        cutlass.Float8E4M3FN,
    ):
        partial_dtype = cutlass.BFloat16
    elif out_dtype == cutlass.Float8E4M3FN or not cfg.use_split_kv:
        partial_dtype = cutlass.Float16

    Int32 = cutlass.Int32
    Int64 = cutlass.Int64
    Float32 = cutlass.Float32

    @cute.jit
    def main_tensor_adapter(
        q: cute.Tensor,
        k_cache: cute.Tensor,
        v_cache: cute.Tensor,
        k_sf_cache: cute.Tensor,
        v_sf_cache: cute.Tensor,
        out: cute.Tensor,
        lse: cute.Tensor | None,
        seq_lens_kv: cute.Tensor,
        cu_seqlens_q: cute.Tensor,
        block_table: cute.Tensor,
        q_token_kv_block_sparse_page_memberships: cute.Tensor,
        partial_o: cute.Tensor,
        partial_stats: cute.Tensor,
        split_kv_counter: cute.Tensor,
        attention_sinks: cute.Tensor,
        num_physical_kv_pages: cutlass.Int64,
        k_page_stride: cutlass.Int64,
        k_head_stride: cutlass.Int64,
        k_token_stride: cutlass.Int64,
        v_page_stride: cutlass.Int64,
        v_head_stride: cutlass.Int64,
        v_token_stride: cutlass.Int64,
        k_sf_page_stride: cutlass.Int64,
        v_sf_page_stride: cutlass.Int64,
        k_sf_head_stride: cutlass.Int64,
        k_sf_token_stride: cutlass.Int64,
        v_sf_head_stride: cutlass.Int64,
        v_sf_token_stride: cutlass.Int64,
        softmax_scale: cutlass.Float32 | cute.Tensor,
        output_scale: cutlass.Float32 | cute.Tensor,
        requested_splits: cutlass.Int32,
        max_q_len: cutlass.Int32,
        direct_q1_inputs: tuple,
        runtime_max_kv_len: cutlass.Int32,
        stream: cuda_drv.CUstream,
        static_cfg: cutlass.Constexpr[FmhaDecodeConfig],
        static_num_qo_heads: cutlass.Constexpr[int],
        static_num_kv_heads: cutlass.Constexpr[int],
        static_head_dim: cutlass.Constexpr[int],
        static_max_active_clusters: cutlass.Constexpr[int],
        static_full_split_prefix: cutlass.Constexpr[bool],
        static_native_uniform_kv: cutlass.Constexpr[bool],
    ) -> None:
        """Adapt TVM-FFI tensors to the dense block-table pointer launcher."""

        pattern_heads = num_sparse_pattern_heads(
            static_num_kv_heads, static_cfg.shares_sparse_pattern
        )
        batch_size = Int32(cute.size(seq_lens_kv)) // Int32(pattern_heads)
        q_offsets_iter = cu_seqlens_q.iterator
        if cutlass.const_expr(uniform_q):
            q_offsets_iter = UniformQueryOffsets(Int32(q.shape[0]) // batch_size)
        total_q_tokens = Int32(q.shape[0])
        if cutlass.const_expr(not static_cfg.use_variable_seqlens_q):
            # Fixed-Q is a distinct specialization. Keep a uniform TVM-FFI
            # wrapper signature, but pass a real null pointer to the kernel so
            # fixed launches have no Q-offset metadata semantics.
            q_offsets_iter = cute.make_ptr(Int32, 0)
        else:
            total_q_tokens = Int32(q.shape[0])

        lengths_iter = seq_lens_kv.iterator
        pages_iter = block_table.iterator
        if cutlass.const_expr(direct_q1_spec is not None):
            lengths_iter = DirectSparseMetadataView(
                direct_q1_inputs,
                cu_seqlens_q,
                packed=static_cfg.use_variable_seqlens_q,
                model_len=direct_q1_inputs[4],
                page_size=storage_page_size,
                page_capacity=direct_q1_inputs[5],
                sparse_block_size=direct_q1_spec.kv_block_size,
                count_current_block_in_topk=direct_q1_spec.count_current_block_in_topk,
                fixed_query_len=direct_q1_inputs[6],
                fixed_query_layout=direct_q1_spec.fixed_query_layout,
                fragment_size=static_cfg.num_tokens_per_page,
                lengths=True,
            )
            pages_iter = DirectSparseMetadataView(
                direct_q1_inputs,
                cu_seqlens_q,
                packed=static_cfg.use_variable_seqlens_q,
                model_len=direct_q1_inputs[4],
                page_size=storage_page_size,
                page_capacity=direct_q1_inputs[5],
                sparse_block_size=direct_q1_spec.kv_block_size,
                count_current_block_in_topk=direct_q1_spec.count_current_block_in_topk,
                fixed_query_len=direct_q1_inputs[6],
                fixed_query_layout=direct_q1_spec.fixed_query_layout,
                fragment_size=static_cfg.num_tokens_per_page,
                lengths=False,
            )

        launch_decode_kernel(
            (
                batch_size,
                Int32(static_num_qo_heads),
                Int32(static_num_kv_heads),
                runtime_max_kv_len,
                Int32(static_head_dim),
            ),
            q.iterator,
            k_cache.iterator,
            v_cache.iterator,
            k_sf_cache.iterator,
            v_sf_cache.iterator,
            out.iterator,
            lengths_iter,
            q_offsets_iter,
            total_q_tokens,
            pages_iter,
            q_token_kv_block_sparse_page_memberships.iterator,
            partial_o.iterator,
            partial_stats.iterator,
            split_kv_counter.iterator,
            attention_sinks.iterator,
            softmax_scale.iterator if compile_spec.scale_types[0] else softmax_scale,
            output_scale.iterator if compile_spec.scale_types[1] else output_scale,
            Int32(0),
            Int32(static_max_active_clusters),
            stream,
            static_cfg,
            None,
            False,
            True,
            Int64(block_table.stride[0]),
            Int32(block_table.shape[1]),
            Int32(q_token_kv_block_sparse_page_memberships.stride[0]),
            num_physical_kv_pages,
            k_page_stride,
            k_head_stride,
            k_token_stride,
            v_page_stride,
            v_head_stride,
            v_token_stride,
            k_sf_page_stride,
            v_sf_page_stride,
            k_sf_head_stride,
            k_sf_token_stride,
            v_sf_head_stride,
            v_sf_token_stride,
            static_full_split_prefix,
            static_native_uniform_kv,
            lse_ptr=lse.iterator if lse is not None else None,
            requested_splits=requested_splits if static_cfg.runtime_split_kv else None,
            max_q_len=max_q_len,
            runtime_q_capacity_rows=(
                Int32(partial_o.shape[-2])
                if static_cfg.use_split_kv
                else max_q_len * Int32(static_num_qo_heads // static_num_kv_heads)
            ),
        )

    reduction_tensor_adapter = None
    if cfg.use_separate_reduction_kernel:
        from .kernel.reduction import launch_separate_reduction

        @cute.jit
        def reduction_tensor_adapter(
            out: cute.Tensor,
            lse: cute.Tensor | None,
            seq_lens_kv: cute.Tensor,
            cu_seqlens_q: cute.Tensor,
            partial_o: cute.Tensor,
            partial_stats: cute.Tensor,
            attention_sinks: cute.Tensor,
            requested_splits: cutlass.Int32,
            max_q_len: cutlass.Int32,
            direct_q1_inputs: tuple,
            runtime_max_kv_len: cutlass.Int32,
            stream: cuda_drv.CUstream,
            static_cfg: cutlass.Constexpr[FmhaDecodeConfig],
            static_num_qo_heads: cutlass.Constexpr[int],
            static_num_kv_heads: cutlass.Constexpr[int],
            static_head_dim: cutlass.Constexpr[int],
            static_full_split_prefix: cutlass.Constexpr[bool],
        ) -> None:
            """Adapt TVM-FFI tensors to the raw standalone split reducer."""

            pattern_heads = num_sparse_pattern_heads(
                static_num_kv_heads, static_cfg.shares_sparse_pattern
            )
            batch_size = Int32(cute.size(seq_lens_kv)) // Int32(pattern_heads)
            q_offsets_iter = cu_seqlens_q.iterator
            if cutlass.const_expr(uniform_q):
                q_offsets_iter = UniformQueryOffsets(Int32(out.shape[0]) // batch_size)
            if cutlass.const_expr(not static_cfg.use_variable_seqlens_q):
                q_offsets_iter = cute.make_ptr(Int32, 0)

            lengths_iter = seq_lens_kv.iterator
            if cutlass.const_expr(direct_q1_spec is not None):
                lengths_iter = DirectSparseMetadataView(
                    direct_q1_inputs,
                    cu_seqlens_q,
                    packed=static_cfg.use_variable_seqlens_q,
                    model_len=direct_q1_inputs[4],
                    page_size=storage_page_size,
                    page_capacity=direct_q1_inputs[5],
                    sparse_block_size=direct_q1_spec.kv_block_size,
                    count_current_block_in_topk=direct_q1_spec.count_current_block_in_topk,
                    fixed_query_len=direct_q1_inputs[6],
                    fixed_query_layout=direct_q1_spec.fixed_query_layout,
                    fragment_size=static_cfg.num_tokens_per_page,
                    lengths=True,
                )
            launch_separate_reduction(
                (
                    batch_size,
                    Int32(static_num_qo_heads),
                    Int32(static_num_kv_heads),
                    runtime_max_kv_len,
                    Int32(static_head_dim),
                ),
                out.iterator,
                lengths_iter,
                q_offsets_iter,
                partial_o.iterator,
                partial_stats.iterator,
                attention_sinks.iterator,
                stream,
                static_cfg,
                static_full_split_prefix,
                lse_ptr=lse.iterator if lse is not None else None,
                requested_splits=requested_splits
                if static_cfg.runtime_split_kv
                else None,
                runtime_q_capacity_rows=Int32(partial_o.shape[-2]),
            )

    physical_pages = cute.sym_int()
    logical_pages = cute.sym_int()
    block_table_row_stride = cute.sym_int64(divisibility=1)
    membership_words = cute.sym_int()
    k_outer_stride = cute.sym_int64(divisibility=1)
    k_head_stride = cute.sym_int64(divisibility=1)
    k_token_stride = cute.sym_int64(divisibility=1)
    v_outer_stride = cute.sym_int64(divisibility=1)
    v_head_stride = cute.sym_int64(divisibility=1)
    v_token_stride = cute.sym_int64(divisibility=1)
    k_sf_outer_stride = cute.sym_int64(divisibility=1)
    k_sf_head_stride = cute.sym_int64(divisibility=1)
    k_sf_token_stride = cute.sym_int64(divisibility=1)
    v_sf_outer_stride = cute.sym_int64(divisibility=1)
    v_sf_head_stride = cute.sym_int64(divisibility=1)
    v_sf_token_stride = cute.sym_int64(divisibility=1)
    batch_size = cute.sym_int()
    total_q_tokens = cute.sym_int()
    runtime_num_q_offsets = cute.sym_int()
    q_shape = (
        (total_q_tokens, num_qo_heads, head_dim)
        if packed_query
        else (
            (batch_size, num_qo_heads, head_dim)
            if seq_len_q == 1
            else (batch_size, seq_len_q, num_qo_heads, head_dim)
        )
    )
    q_fake = cute.runtime.make_fake_compact_tensor(
        q_dtype,
        q_shape,
        stride_order=tuple(reversed(range(len(q_shape)))),
        assumed_align=16,
    )
    k_storage_head_dim = head_dim // 2 if k_dtype == cutlass.Float4E2M1FN else head_dim
    v_storage_head_dim = head_dim // 2 if v_dtype == cutlass.Float4E2M1FN else head_dim
    if cfg.use_nvfp4_interleaved_kv:
        k_storage_head_dim = v_storage_head_dim = cfg.nvfp4_interleaved_row_bytes
    k_fake = cute.runtime.make_fake_tensor(
        k_storage_dtype,
        (physical_pages, num_kv_heads, storage_page_size, k_storage_head_dim),
        stride=(
            k_outer_stride,
            k_head_stride,
            k_token_stride,
            1,
        ),
        assumed_align=16,
    )
    v_fake = cute.runtime.make_fake_tensor(
        v_storage_dtype,
        (physical_pages, num_kv_heads, storage_page_size, v_storage_head_dim),
        stride=(
            v_outer_stride,
            v_head_stride,
            v_token_stride,
            1,
        ),
        assumed_align=16,
    )
    out_shape = decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=total_q_tokens if packed_query else None,
    )
    out_fake = cute.runtime.make_fake_compact_tensor(
        out_dtype,
        out_shape,
        stride_order=tuple(reversed(range(len(out_shape)))),
        assumed_align=16,
    )

    # Describe row-major symbolic storage; assumed_align is in bytes.
    def fake_compact(dtype, shape, assumed_align):
        return cute.runtime.make_fake_compact_tensor(
            dtype,
            shape,
            stride_order=tuple(reversed(range(len(shape)))),
            assumed_align=assumed_align,
        )

    lse_fake = (
        fake_compact(Float32, q_shape[:-1], 4) if cfg.lse_mode != "none" else None
    )
    pattern_heads = num_sparse_pattern_heads(num_kv_heads, cfg.shares_sparse_pattern)
    metadata_rows = batch_size * pattern_heads
    seq_lens_kv_fake = fake_compact(Int32, (metadata_rows,), 4)
    cu_seqlens_q_fake = fake_compact(
        Int32, (runtime_num_q_offsets,) if packed_query else (1,), 4
    )
    block_table_fake = cute.runtime.make_fake_tensor(
        Int32,
        (metadata_rows, logical_pages),
        stride=(block_table_row_stride, 1),
        assumed_align=4,
    )
    q_token_kv_block_sparse_page_memberships_fake = fake_compact(
        Int32, (metadata_rows, membership_words), 4
    )
    partial_o_shape, partial_stats_shape, counter_shape = decode_scratch_shapes(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
    )
    partial_o_fake = fake_compact(partial_dtype, partial_o_shape, 16)
    partial_stats_fake = fake_compact(Float32, partial_stats_shape, 16)
    counter_fake = fake_compact(Int32, counter_shape, 4)
    attention_sinks_fake = fake_compact(
        Float32, (num_qo_heads,) if cfg.use_attention_sinks else (1,), 4
    )
    if cfg.uses_nvfp4_scale_tensors:
        # QToken-route scale tensors may be strided HND views, like the packed
        # K/V (FmhaDecodeConfig.sf_tma_reshape_factor).
        sf_shape = (physical_pages, num_kv_heads, storage_page_size, head_dim // 16)
        k_sf_fake = cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN,
            sf_shape,
            stride=(k_sf_outer_stride, k_sf_head_stride, k_sf_token_stride, 1),
            assumed_align=16,
        )
        v_sf_fake = cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN,
            sf_shape,
            stride=(v_sf_outer_stride, v_sf_head_stride, v_sf_token_stride, 1),
            assumed_align=16,
        )
    else:
        k_sf_fake = fake_compact(cutlass.Uint8, (1,), 1)
        v_sf_fake = fake_compact(cutlass.Uint8, (1,), 1)
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    direct_q1_fake: tuple = ()
    if direct_q1_spec is not None:
        rows = cute.sym_int()
        direct_q1_fake = (
            cute.runtime.make_fake_tensor(
                Int32,
                (
                    (rows, cute.sym_int())
                    if cfg.shares_sparse_pattern
                    else (rows, num_kv_heads, cute.sym_int())
                ),
                stride=(
                    (cute.sym_int64(), 1)
                    if cfg.shares_sparse_pattern
                    else (cute.sym_int64(), cute.sym_int64(), 1)
                ),
                assumed_align=4,
            ),
            cute.runtime.make_fake_tensor(
                Int32,
                (cute.sym_int(), cute.sym_int()),
                stride=(cute.sym_int64(), 1),
                assumed_align=4,
            ),
            (
                None
                if direct_q1_spec.fixed_query_layout
                else fake_compact(Int32, (rows,), 4)
            ),
            fake_compact(
                Int64 if direct_q1_spec.position_dtype_key == "int64" else Int32,
                (cute.sym_int(),) if direct_q1_spec.fixed_query_layout else (rows,),
                4,
            ),
            Int32(0),
            Int32(0),
            Int32(0),
        )

    with torch.cuda.device(device_index):
        compiled_main = build_and_load_attention_kernel(
            "decode",
            "main",
            compile_spec,
            COMPILE_OPTIONS,
            lambda: cute.compile(
                main_tensor_adapter,
                q_fake,
                k_fake,
                v_fake,
                k_sf_fake,
                v_sf_fake,
                out_fake,
                lse_fake,
                seq_lens_kv_fake,
                cu_seqlens_q_fake,
                block_table_fake,
                q_token_kv_block_sparse_page_memberships_fake,
                partial_o_fake,
                partial_stats_fake,
                counter_fake,
                attention_sinks_fake,
                *(Int64(1),) * 13,
                fake_scale(compile_spec.scale_types[0]),
                fake_scale(compile_spec.scale_types[1]),
                Int32(1),
                Int32(0),
                direct_q1_fake,
                Int32(0),
                stream_fake,
                cfg,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                max_active_clusters,
                static_full_split_prefix,
                static_native_uniform_kv,
                options=COMPILE_OPTIONS,
            ),
        )
        compiled_reducer = None
        if cfg.use_separate_reduction_kernel:
            assert reduction_tensor_adapter is not None
            compiled_reducer = build_and_load_attention_kernel(
                "decode",
                "reducer",
                replace(compile_spec, scale_types=(False, False)),
                COMPILE_OPTIONS,
                lambda: cute.compile(
                    reduction_tensor_adapter,
                    out_fake,
                    lse_fake,
                    seq_lens_kv_fake,
                    cu_seqlens_q_fake,
                    partial_o_fake,
                    partial_stats_fake,
                    attention_sinks_fake,
                    Int32(1),
                    Int32(0),
                    direct_q1_fake,
                    Int32(0),
                    stream_fake,
                    cfg,
                    num_qo_heads,
                    num_kv_heads,
                    head_dim,
                    static_full_split_prefix,
                    options=COMPILE_OPTIONS,
                ),
            )

    return compiled_main, compiled_reducer


def launch_decode(
    runtime: DecodeRuntime,
    *,
    seq_lens_kv: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    block_tables: torch.Tensor,
    workspace: DecodeWorkspaceViews,
    compiled_main: Callable[..., object],
    compiled_reducer: Optional[Callable[..., object]],
    lse: Optional[torch.Tensor] = None,
    attention_sinks: Optional[torch.Tensor] = None,
    requested_splits: int = 1,
    max_kv_len: int,
    max_q_len: int,
) -> torch.Tensor:
    """Launch the compiled main kernel and its optional standalone reducer."""

    sinks = workspace.attention_sinks if attention_sinks is None else attention_sinks
    q_offsets = workspace.cu_seqlens_q if qo_indptr is None else qo_indptr
    compiled_main(
        runtime.q.view(-1, runtime.q.shape[-2], runtime.q.shape[-1]),
        runtime.k_cache,
        runtime.v_cache,
        runtime.k_sf_cache,
        runtime.v_sf_cache,
        runtime.out.view(-1, runtime.out.shape[-2], runtime.out.shape[-1]),
        None if lse is None else lse.view(-1, lse.shape[-1]),
        seq_lens_kv,
        q_offsets,
        block_tables,
        block_tables,  # Dense decode constexpr-elides the membership table.
        workspace.partial_o,
        workspace.partial_stats,
        workspace.split_kv_counter,
        sinks,
        runtime.num_physical_pages,
        runtime.k_page_stride,
        runtime.k_head_stride,
        runtime.k_token_stride,
        runtime.v_page_stride,
        runtime.v_head_stride,
        runtime.v_token_stride,
        runtime.k_sf_page_stride,
        runtime.v_sf_page_stride,
        runtime.k_sf_head_stride,
        runtime.k_sf_token_stride,
        runtime.v_sf_head_stride,
        runtime.v_sf_token_stride,
        runtime.softmax_scale,
        runtime.output_scale,
        requested_splits,
        max_q_len,
        (),
        max_kv_len,
    )
    if compiled_reducer is not None:
        compiled_reducer(
            runtime.out.view(-1, runtime.out.shape[-2], runtime.out.shape[-1]),
            None if lse is None else lse.view(-1, lse.shape[-1]),
            seq_lens_kv,
            q_offsets,
            workspace.partial_o,
            workspace.partial_stats,
            sinks,
            requested_splits,
            max_q_len,
            (),
            max_kv_len,
        )
    return runtime.out


def make_decode_compile_spec(
    launch_spec: DecodeLaunchSpec,
    *,
    device_index: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    out_dtype_key: str,
    direct_q1_spec: Optional[DirectQ1CompileSpec] = None,
    flat_native_kv_tma: bool = False,
    flat_nvfp4_scale_num_heads: int = 0,
    lse_mode: str = "none",
    runtime_kv_lengths: bool = False,
    scale_types: tuple[bool, bool] = (False, False),
    packed_query: Optional[bool] = None,
    kv_prefix_mode: Literal["dynamic", "planned_full"] = "dynamic",
    kv_lengths_mode: Literal["dynamic", "planned_uniform_max"] = "dynamic",
) -> DecodeCompileSpec:
    """Freeze the resolved topology while leaving batch and lengths in runtime tensors.

    The config's compile signature already drops every length-derived field, and
    all requests launch through the flattened-Q ABI. Uniform inputs derive
    their offsets from the runtime tensor extent; packed inputs read metadata.
    """

    if launch_spec.config.fixed_seqlen:
        raise ValueError("cached TS decode compiles runtime sequence lengths only")
    config_items = dict(launch_spec.config.compile_signature())
    # Public wrapper plans use runtime KV metadata in compiled kernels. False is retained only
    # for internal prepared/sparse recipes with a fixed native layout.
    if runtime_kv_lengths:
        # Odd, non-aligned geometry keeps static full-tile shortcuts disabled.
        # Native paged requests obtain every real bound from live metadata.
        cfg = launch_spec.config
        max_kv_len = int(cfg.tile_size_kv * cfg.num_insts_kv * 2 + 1)
        config_items["total_kv_tiles"] = (
            max_kv_len + cfg.tile_size_kv - 1
        ) // cfg.tile_size_kv
        config_items["static_seq_len_kv"] = max_kv_len
        config_items["max_num_pages_per_seq_kv"] = (
            max_kv_len + cfg.num_tokens_per_page - 1
        ) // cfg.num_tokens_per_page
        config_items["static_num_skipped_kv_tiles"] = 0
        config_items["static_window_start_idx"] = 0
    config_items["lse_mode"] = lse_mode
    config_items["use_flat_native_kv_tma"] = flat_native_kv_tma
    config_items["flat_native_kv_num_heads"] = num_kv_heads if flat_native_kv_tma else 1
    config_items["flat_nvfp4_scale_num_heads"] = flat_nvfp4_scale_num_heads
    return DecodeCompileSpec(
        device_index=device_index,
        config_items=tuple(config_items.items()),
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        page_size=page_size,
        q_dtype_key=q_dtype_key,
        k_dtype_key=k_dtype_key,
        v_dtype_key=v_dtype_key,
        out_dtype_key=out_dtype_key,
        packed_query=(
            launch_spec.config.use_variable_seqlens_q
            if packed_query is None
            else packed_query
        )
        or direct_q1_spec is not None,
        max_active_clusters=launch_spec.max_active_clusters,
        kv_prefix_mode=kv_prefix_mode,
        kv_lengths_mode=kv_lengths_mode,
        direct_q1_spec=direct_q1_spec,
        runtime_kv_lengths=runtime_kv_lengths,
        scale_types=scale_types,
    )


def make_decode_workspace_layout(
    scratch_shapes: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    out_dtype: torch.dtype,
    *,
    use_separate_reduction_kernel: bool,
    use_split_kv: bool = False,
    scratch_reserve: int = 0,
) -> DecodeWorkspaceLayout:
    """Lay out partials first, then zero-at-rest state.

    The state (Q offsets, sinks, ``scratch_reserve`` split counters) comes
    last; binding anchors it to the buffer end (see ``_bind_decode_workspace``).
    """
    partial_o_shape, partial_stats_shape, counter_shape = scratch_shapes
    partial_o_dtype = out_dtype
    if use_separate_reduction_kernel and out_dtype in (
        torch.bfloat16,
        torch.float8_e4m3fn,
    ):
        partial_o_dtype = torch.bfloat16
    elif out_dtype == torch.float8_e4m3fn or partial_o_shape == (1, 1, 1, 1, 1):
        partial_o_dtype = torch.float16

    byte_end = 0
    partial_o, byte_end = append_workspace_section(
        byte_end, partial_o_shape, partial_o_dtype
    )
    partial_stats, byte_end = append_workspace_section(
        byte_end, partial_stats_shape, torch.float32
    )
    cu_seqlens_q, byte_end = append_workspace_section(byte_end, (1,), torch.int32)
    attention_sinks, byte_end = append_workspace_section(byte_end, (1,), torch.float32)
    split_kv_counter, byte_end = append_workspace_section(
        byte_end, counter_shape, torch.int32, scratch_reserve
    )
    return DecodeWorkspaceLayout(
        partial_o=partial_o,
        partial_stats=partial_stats,
        split_kv_counter=split_kv_counter,
        cu_seqlens_q=cu_seqlens_q,
        attention_sinks=attention_sinks,
        uses_split_kv=use_split_kv,
        total_bytes=align_up(byte_end),
    )


def normalize_plan_seq_lens_kv(
    seq_lens_kv: Optional[Union[Sequence[int], torch.Tensor]],
    *,
    batch_size: int,
    max_kv_len: int,
) -> Optional[tuple[int, ...]]:
    """Validate optional host-only lengths retained by the plan."""

    if seq_lens_kv is None:
        return None
    if isinstance(seq_lens_kv, torch.Tensor):
        if seq_lens_kv.device.type != "cpu":
            raise ValueError("plan seq_lens_kv must be on CPU")
        if seq_lens_kv.ndim != 1:
            raise ValueError("plan seq_lens_kv must be 1D")
        if seq_lens_kv.dtype not in (torch.int32, torch.int64):
            raise TypeError("plan seq_lens_kv must have int32 or int64 dtype")
        raw_values: Sequence[object] = seq_lens_kv.tolist()
    elif isinstance(seq_lens_kv, Sequence) and not isinstance(
        seq_lens_kv, (str, bytes, bytearray)
    ):
        raw_values = seq_lens_kv
    else:
        raise TypeError(
            "plan seq_lens_kv must be a host sequence of integers or a CPU tensor"
        )

    if len(raw_values) != batch_size:
        raise ValueError(
            "plan seq_lens_kv must contain exactly "
            f"batch_size ({batch_size}) values, got {len(raw_values)}"
        )
    normalized = []
    for request_idx, value in enumerate(raw_values):
        if isinstance(value, bool) or not isinstance(value, numbers.Integral):
            raise TypeError(
                "plan seq_lens_kv must contain integers; "
                f"request {request_idx} has {value!r}"
            )
        seq_len = max(int(value), 0)
        if seq_len > max_kv_len:
            raise ValueError(
                "plan seq_lens_kv values must be within "
                f"[0, {max_kv_len}]; request {request_idx} has {seq_len}"
            )
        normalized.append(seq_len)
    return tuple(normalized)


def prepare_decode_runtime(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    device: torch.device,
    batch_size: int,
    seq_len_q: int,
    packed_query: bool,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
    softmax_scale: Optional[GlobalScale],
    output_scale: GlobalScale,
    out: Optional[torch.Tensor],
    kv_layout: str = "HND",
) -> DecodeRuntime:
    """Validate runtime tensors and normalize zero-copy K/V views."""

    storage_page_size = page_size

    validate_q(
        q,
        seq_len_q=seq_len_q,
        packed_query=packed_query,
        device=device,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        q_dtype=q_dtype,
    )
    normalized_cache = normalize_native_paged_kv_cache(
        paged_kv_cache,
        expected_device=device,
        kv_layout=kv_layout,
    )
    k_cache = normalized_cache.k_cache
    v_cache = normalized_cache.v_cache
    if (
        normalized_cache.num_kv_heads != num_kv_heads
        or normalized_cache.storage_page_size != storage_page_size
        or normalized_cache.head_dim != head_dim
    ):
        raise ValueError(
            "paged_kv_cache geometry does not match the launch: expected "
            f"Hkv/storage_page/D=({num_kv_heads}, {storage_page_size}, "
            f"{head_dim}), got "
            f"({normalized_cache.num_kv_heads}, "
            f"{normalized_cache.storage_page_size}, {normalized_cache.head_dim})"
        )
    if k_cache.dtype != k_dtype or v_cache.dtype != v_dtype:
        raise ValueError(
            f"K/V dtype must match the launch (K {k_dtype}, V {v_dtype}), got K "
            f"{k_cache.dtype} and V {v_cache.dtype}"
        )
    k_sf_cache, v_sf_cache, k_sf_strides, v_sf_strides = (
        normalize_paged_kv_scale_factors(
            kv_scale_factors,
            k_cache=k_cache,
            logical_head_dim=head_dim,
        )
    )
    effective_softmax_scale = normalize_scale(
        1.0 / math.sqrt(head_dim) if softmax_scale is None else softmax_scale,
        "softmax_scale",
        device,
    )
    effective_output_scale = normalize_scale(output_scale, "output_scale", device)
    output_shape = decode_output_shape(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        seq_len_q=seq_len_q,
        head_dim=head_dim,
        total_q_tokens=int(q.shape[0]) if packed_query else None,
    )
    if out is None:
        out = torch.empty(output_shape, device=device, dtype=out_dtype)
    else:
        validate_out(
            out,
            q=q,
            expected_shape=output_shape,
            seq_len_q=seq_len_q,
            packed_query=packed_query,
            out_dtype=out_dtype,
        )
    return DecodeRuntime(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_sf_cache=k_sf_cache,
        v_sf_cache=v_sf_cache,
        out=out,
        num_physical_pages=normalized_cache.num_physical_pages,
        k_page_stride=normalized_cache.k_page_stride,
        k_head_stride=normalized_cache.k_head_stride,
        k_token_stride=normalized_cache.k_token_stride,
        v_page_stride=normalized_cache.v_page_stride,
        v_head_stride=normalized_cache.v_head_stride,
        v_token_stride=normalized_cache.v_token_stride,
        k_sf_page_stride=k_sf_strides[0],
        v_sf_page_stride=v_sf_strides[0],
        k_sf_head_stride=k_sf_strides[1],
        k_sf_token_stride=k_sf_strides[2],
        v_sf_head_stride=v_sf_strides[1],
        v_sf_token_stride=v_sf_strides[2],
        softmax_scale=effective_softmax_scale,
        output_scale=effective_output_scale,
    )


def prepare_decode_runtime_unchecked(
    q: torch.Tensor,
    paged_kv_cache: PagedKVCache,
    kv_scale_factors: Optional[PagedKVScaleFactors],
    *,
    out_dtype: torch.dtype,
    softmax_scale: Optional[GlobalScale],
    output_scale: GlobalScale,
    out: Optional[torch.Tensor],
    kv_layout: str = "HND",
) -> DecodeRuntime:
    """Canonicalize one trusted run without invoking explicit validators."""

    if isinstance(paged_kv_cache, torch.Tensor):
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
    else:
        k_cache, v_cache = paged_kv_cache
    if kv_layout == "NHD":
        k_cache = k_cache.transpose(1, 2)
        v_cache = v_cache.transpose(1, 2)
    if k_cache.dtype == torch.uint8:
        k_sf_cache, v_sf_cache = cast(PagedKVScaleFactors, kv_scale_factors)
        k_sf_strides = tuple(int(stride) for stride in k_sf_cache.stride()[:3])
        v_sf_strides = tuple(int(stride) for stride in v_sf_cache.stride()[:3])
    else:
        placeholder = k_cache[0, 0, 0, :1].view(torch.uint8)[:1]
        k_sf_cache, v_sf_cache = placeholder, placeholder
        k_sf_strides = v_sf_strides = (0, 0, 0)
    if out is None:
        out = torch.empty(
            q.shape,
            device=q.device,
            dtype=out_dtype,
        )
    return DecodeRuntime(
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        k_sf_cache=k_sf_cache,
        v_sf_cache=v_sf_cache,
        out=out,
        num_physical_pages=int(k_cache.shape[0]),
        k_page_stride=int(k_cache.stride(0)),
        k_head_stride=int(k_cache.stride(1)),
        k_token_stride=int(k_cache.stride(2)),
        v_page_stride=int(v_cache.stride(0)),
        v_head_stride=int(v_cache.stride(1)),
        v_token_stride=int(v_cache.stride(2)),
        k_sf_page_stride=k_sf_strides[0],
        v_sf_page_stride=v_sf_strides[0],
        k_sf_head_stride=k_sf_strides[1],
        k_sf_token_stride=k_sf_strides[2],
        v_sf_head_stride=v_sf_strides[1],
        v_sf_token_stride=v_sf_strides[2],
        softmax_scale=(
            1.0 / math.sqrt(int(q.shape[-1]))
            if softmax_scale is None
            else softmax_scale
        ),
        output_scale=output_scale,
    )


# Share CPU policy selection across equivalent plans/layers.
@functools.cache
def resolve_decode_launch_spec(
    device_index: int,
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    seq_len_q: int,
    q_dtype_key: str,
    k_dtype_key: str,
    v_dtype_key: str,
    out_dtype_key: str,
    kv_layout: str,
    mask_type: str,
    packed_query: bool,
    window_left: int,
    storage_page_size: Optional[int] = None,
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    use_attention_sinks: bool = False,
) -> DecodeLaunchSpec:
    """Resolve automatic policy and workspace geometry without compiling."""

    seq_len_q = validate_seq_len_q(seq_len_q)
    validate_head_geometry(num_qo_heads, num_kv_heads)
    validate_decode_query_head_extent(
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        max_q_len=seq_len_q,
    )
    max_kv_len = validate_max_kv_len(max_kv_len, "max_kv_len")
    window_left = validate_window_left(window_left, mask_type)
    storage_page_size = validate_storage_page_size(
        page_size,
        page_size if storage_page_size is None else storage_page_size,
    )

    from .kernel.device.constants import MIN_LOOP_ITERS_PER_SPLIT
    from .kernel.config import (
        arch_config_args,
        get_max_active_clusters_for_cluster_size,
        make_q_tile_geometry,
    )
    from .kernel.selection import make_decode_config

    validate_layout(kv_layout)
    q_dtype = cutlass_dtype(q_dtype_key)
    k_dtype = cutlass_dtype(k_dtype_key)
    v_dtype = cutlass_dtype(v_dtype_key)
    out_dtype = cutlass_dtype(out_dtype_key)
    arch_args = arch_config_args(torch.cuda.get_device_capability(device_index))

    # Resolve a candidate against this request geometry and device policy.
    def make_config(
        args: object | None = None,
        *,
        split_kv_mode: str = "disabled",
        splits_kv: int = -1,
        max_splits_kv: int | None = None,
        min_loop_iters_per_split: int = MIN_LOOP_ITERS_PER_SPLIT,
    ) -> "FmhaDecodeConfig":
        return make_decode_config(
            headdim=head_dim,
            args=(arch_args, args),
            seq_len_q=seq_len_q,
            seq_len_kv=max_kv_len,
            batch_size=batch_size,
            num_heads_q=num_qo_heads,
            num_heads_kv=num_kv_heads,
            q_dtype=q_dtype,
            k_dtype=k_dtype,
            v_dtype=v_dtype,
            o_dtype=out_dtype,
            qkv_layout="pagedKv",
            num_tokens_per_page=page_size,
            storage_tokens_per_page=storage_page_size,
            split_kv_mode=split_kv_mode,
            splits_kv=splits_kv,
            max_splits_kv=max_splits_kv,
            min_loop_iters_per_split=min_loop_iters_per_split,
            mask_type=mask_type,
            sliding_window_causal=window_left >= 0,
            attention_window_size=window_left + 1 if window_left >= 0 else 0,
            auto_tuner=True,
            split_kv=split_kv,
            use_attention_sinks=use_attention_sinks,
        )

    # Count logical Q CTAs per KV head, before split-KV fanout.
    def q_ctas(config: "FmhaDecodeConfig") -> int:
        geometry = make_q_tile_geometry(
            rows_per_cta=config.tile_size_q,
            heads_q_per_kv=num_qo_heads // num_kv_heads,
            groups_tokens_heads_q=config.groups_tokens_heads_q,
        )
        return max(int(geometry.num_q_ctas(seq_len_q)), 1)

    # Check CTA or cluster occupancy for the complete nonpersistent grid.
    def fits_one_service_wave(config: "FmhaDecodeConfig", num_q_ctas: int) -> bool:
        logical_grid = batch_size * num_kv_heads * num_q_ctas
        if config.use_persistent_scheduler:
            return False
        if config.use_cluster_smem_reduction:
            cluster_capacity = get_max_active_clusters_for_cluster_size(
                int(config.splits_kv)
            )
            return cluster_capacity > 0 and logical_grid <= cluster_capacity
        split_fanout = int(config.splits_kv) if config.use_split_kv else 1
        return logical_grid * split_fanout <= max_active_clusters

    # Device capacity participates in automatic selection. Resolve it in the
    # target device context without introducing caller-visible policy knobs.
    with torch.cuda.device(device_index):
        max_active_clusters = get_max_active_clusters_for_cluster_size(1)
        if use_q_token_kv_block_sparse_route:
            # Imported here: the QToken package imports decode at module load.
            from ..q_token_kv_block_sparse_metadata.grouping import (
                resolve_decode_config,
            )

            cfg = resolve_decode_config(
                make_config,
                batch_size=batch_size,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                page_size=page_size,
                max_kv_len=max_kv_len,
                group_size=seq_len_q,
                q_dtype_key=q_dtype_key,
                k_dtype_key=k_dtype_key,
                v_dtype_key=v_dtype_key,
                out_dtype_key=out_dtype_key,
                mask_type=mask_type,
                packed_query=packed_query,
                window_left=window_left,
                max_active_clusters=max_active_clusters,
                use_pdl=use_pdl,
                split_kv=split_kv,
                share_pattern_across_kv_heads=share_pattern_across_kv_heads,
            )
        else:
            config_overrides = {"use_pdl": use_pdl}
            if packed_query:
                config_overrides["use_variable_seqlens_q"] = True
            cfg = make_config(config_overrides)

            # A grouped fixed-Q launch can leave most of the first service wave
            # idle. In that regime, evaluate the narrowest supported Swaps head
            # band. Keep it only when the extra head-band CTAs fit in the same
            # resident wave without reducing KV fanout or changing topology.
            if (
                seq_len_q == 1
                and not packed_query
                and cfg.groups_tokens_heads_q
                and not cfg.use_keeps_mma_ab
            ):
                grouped_q_ctas = q_ctas(cfg)
                head_band_geometry = make_q_tile_geometry(
                    rows_per_cta=8,
                    heads_q_per_kv=num_qo_heads // num_kv_heads,
                    groups_tokens_heads_q=False,
                )
                head_band_q_ctas = max(int(head_band_geometry.num_q_ctas(seq_len_q)), 1)
                head_band_cfg = None
                if head_band_q_ctas > grouped_q_ctas and fits_one_service_wave(
                    cfg, head_band_q_ctas
                ):
                    try:
                        head_band_cfg = make_config(
                            {
                                "groups_tokens_heads_q": False,
                                "tile_size_q": 8,
                                "use_pdl": use_pdl,
                            }
                        )
                    except ValueError:
                        head_band_cfg = None
                if head_band_cfg is not None:
                    same_launch_topology = all(
                        getattr(cfg, field) == getattr(head_band_cfg, field)
                        for field in (
                            "use_split_kv",
                            "splits_kv",
                            "max_splits_kv",
                            "use_cluster_smem_reduction",
                            "use_separate_reduction_kernel",
                            "use_persistent_scheduler",
                        )
                    )
                    if same_launch_topology and fits_one_service_wave(
                        head_band_cfg, q_ctas(head_band_cfg)
                    ):
                        cfg = head_band_cfg

            q_token_kv_block_sparse_scattered_fp8 = (
                page_size == 4
                and storage_page_size > page_size
                and q_dtype_key == "float8_e4m3fn"
            )
            unsafe_direct_swaps = not cfg.use_split_kv and not cfg.use_keeps_mma_ab
            unsafe_split_publisher = (
                cfg.use_split_kv and not cfg.use_separate_reduction_kernel
            )
            if q_token_kv_block_sparse_scattered_fp8 and (
                unsafe_direct_swaps or unsafe_split_publisher
            ):
                if not split_kv:
                    raise NotImplementedError(
                        "this FP8 encoded-subpage profile requires split-KV; "
                        "no qualified direct recipe is available with split_kv=False"
                    )
                # Direct, fused-GMEM, and cluster-SMEM publication are not
                # qualified with encoded subpage locators on the FP8 pipeline.
                safe_splits = int(cfg.splits_kv) if cfg.use_split_kv else 2
                safe_tile_size_q = max(int(cfg.tile_size_q), 16)
                cfg = make_config(
                    {
                        "groups_tokens_heads_q": cfg.groups_tokens_heads_q,
                        "tile_size_q": safe_tile_size_q,
                        "use_variable_seqlens_q": packed_query,
                        "use_pdl": use_pdl,
                    },
                    split_kv_mode="gmem_reduction_with_separate_kernel",
                    splits_kv=safe_splits,
                    max_splits_kv=safe_splits,
                )

    requested_splits = int(cfg.splits_kv)
    if cfg.use_split_kv and not use_q_token_kv_block_sparse_route:
        sm_count = get_device_sm_count(torch.device("cuda", device_index))
        requested_splits = min(requested_splits, sm_count)
        # GMEM capacity is a device bound, independent of the runtime fanout.
        # It only bounds the fanout: compact partials are strided by the
        # runtime fanout, so capacity does not size scratch.
        # Keep Q/batch fusion within CUDA's grid-Z limit; larger grids retain
        # the original exact-fanout launch. Cluster sizes remain resident topology.
        compact_grid = batch_size * q_ctas(cfg) <= 65535
        capacity = (
            requested_splits
            if cfg.use_cluster_smem_reduction or not compact_grid
            else sm_count
        )
        cfg = replace(
            cfg,
            splits_kv=capacity,
            max_splits_kv=capacity,
            runtime_split_kv=compact_grid,
        )
    validate_decode_policy_kv_tile_size(cfg)
    resolved = decode_launch_spec_from_config(
        cfg,
        batch_size=batch_size,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_q=seq_len_q,
        max_active_clusters=int(max_active_clusters),
        requested_splits=requested_splits if cfg.use_split_kv else None,
    )
    # Fused GMEM split plans publish through the counters, so they must fit the
    # shared reserve. Separate-reduction plans (e.g. FP8 page-4 encoded-subpage,
    # which always splits) never touch the counters and may exceed it.
    assert (
        use_q_token_kv_block_sparse_route
        or not cfg.use_split_kv
        or cfg.use_separate_reduction_kernel
        or resolved.scratch_reserve == 2 * int(max_active_clusters)
    ), "fused GMEM split counters exceed the shared workspace reserve"
    # Report the runtime fanout separately from the SM-bound workspace capacity.
    policy = dict(resolved.policy)
    policy["splits_kv"] = requested_splits
    return replace(
        resolved, policy=tuple(policy.items()), requested_splits=requested_splits
    )


def resolve_decode_workspace_layout(
    batch_size: int,
    num_qo_heads: int,
    num_kv_heads: int,
    head_dim: int,
    page_size: int,
    max_kv_len: int,
    seq_len_q: int,
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
    kv_layout: str,
    mask_type: str,
    packed_query: bool,
    window_left: int,
    storage_page_size: int,
    device: Optional[Union[int, str, torch.device]],
    use_q_token_kv_block_sparse_route: bool = False,
    use_pdl: bool = False,
    split_kv: bool = True,
    share_pattern_across_kv_heads: bool = True,
    use_attention_sinks: bool = False,
    validate: bool = True,
) -> DecodeWorkspaceLayout:
    """Resolve the byte layout for one already-validated semantic key."""

    resolved_device, device_index = resolve_cuda_device(device)
    if validate:
        validate_runtime_device(resolved_device)
    spec = resolve_decode_launch_spec(
        device_index,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_kv_len,
        seq_len_q,
        attention_dtype_key(q_dtype),
        attention_dtype_key(k_dtype),
        attention_dtype_key(v_dtype),
        attention_dtype_key(out_dtype),
        kv_layout,
        mask_type,
        packed_query,
        window_left,
        storage_page_size,
        use_q_token_kv_block_sparse_route,
        use_pdl,
        split_kv,
        share_pattern_across_kv_heads,
        use_attention_sinks,
    )
    return make_decode_workspace_layout(
        spec.scratch_shapes,
        out_dtype,
        use_separate_reduction_kernel=spec.config.use_separate_reduction_kernel,
        use_split_kv=spec.config.use_split_kv,
        scratch_reserve=spec.scratch_reserve,
    )


def validate_decode_run_metadata_values(
    runtime: DecodeRuntime,
    *,
    seq_lens_kv: torch.Tensor,
    block_tables: torch.Tensor,
    qo_indptr: Optional[torch.Tensor],
    max_kv_len: int,
    page_size: int,
    packed_query: bool,
    seq_len_q: int,
    batch_size: int,
    mask_type: str,
    planned_seq_lens_kv_host: Optional[tuple[int, ...]] = None,
) -> tuple[int, int]:
    """Validate metadata and return live maxima from the same readback."""

    runtime_seq_lens_kv = planned_seq_lens_kv_host
    if runtime_seq_lens_kv is None:
        runtime_seq_lens_kv = tuple(int(value) for value in seq_lens_kv.tolist())
    table_capacity = int(block_tables.shape[1])
    for request_idx, seq_len in enumerate(runtime_seq_lens_kv):
        if seq_len <= 0:
            continue
        if seq_len > max_kv_len:
            raise ValueError(
                f"seq_lens_kv values must be within [1, {max_kv_len}]; "
                f"request {request_idx} has {seq_len}"
            )
        required_pages = (seq_len + page_size - 1) // page_size
        if table_capacity < required_pages:
            raise ValueError(
                "block_tables does not have enough columns for "
                f"seq_lens_kv[{request_idx}]={seq_len}: requires {required_pages}, "
                f"got {table_capacity}"
            )

    block_table_rows = block_tables.tolist()
    locator_capacity = runtime.num_physical_pages * (
        int(runtime.k_cache.shape[2]) // page_size
    )
    for request_idx, (row, seq_len) in enumerate(
        zip(block_table_rows, runtime_seq_lens_kv, strict=True)
    ):
        if seq_len <= 0:
            continue
        required_pages = (seq_len + page_size - 1) // page_size
        if any(
            int(page_id) < 0 or int(page_id) >= locator_capacity
            for page_id in row[:required_pages]
        ):
            raise ValueError(
                "block_tables values for active pages must index the physical "
                f"K/V cache through locators in [0, {locator_capacity}); request "
                f"{request_idx} contains an invalid page ID"
            )

    if packed_query:
        assert qo_indptr is not None
        runtime_max_q_len, total_q, q_lengths = read_packed_q_plan_metadata(qo_indptr)
        if runtime_max_q_len > seq_len_q:
            raise ValueError(
                "qo_indptr contains a per-request Q length larger than "
                f"max_q_len ({seq_len_q}): got {runtime_max_q_len}"
            )
        if total_q != int(runtime.q.shape[0]):
            raise ValueError(
                "the final qo_indptr offset must equal the packed q token count: "
                f"expected {runtime.q.shape[0]}, got {total_q}"
            )
    else:
        q_lengths = (seq_len_q,) * batch_size
        runtime_max_q_len = seq_len_q

    if mask_type == "causal":
        for request_idx, (q_len, kv_len) in enumerate(
            zip(q_lengths, runtime_seq_lens_kv, strict=True)
        ):
            if kv_len <= 0:
                continue
            if q_len > kv_len:
                raise ValueError(
                    "causal decode requires every per-request Q length to be "
                    "no greater than its K/V length; request "
                    f"{request_idx} has Q={q_len} and K/V={kv_len}"
                )
    return runtime_max_q_len, max(0, max(runtime_seq_lens_kv))
