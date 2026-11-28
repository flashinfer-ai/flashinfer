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

"""Query-row indexing: logical/physical output rows, Q groups and sequence bounds."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.experimental.task_scheduling.resources import StageInfo
from ..config import FmhaDecodeConfig
from .common import Constexpr


class UniformQueryOffsets:
    """Affine offsets for fixed-layout Q, with a runtime token count."""

    def __init__(self, seq_len_q):
        self.values = (seq_len_q,)

    def __extract_mlir_values__(self):
        return cutlass.extract_mlir_values(self.values)

    @cute.jit
    def __getitem__(self, index):
        return Int32(index) * Int32(self.seq_len_q)

    def __new_from_mlir_values__(self, values):
        return UniformQueryOffsets(cutlass.new_from_mlir_values(self.values, values)[0])

    @property
    def seq_len_q(self):
        return self.values[0]


@cute.jit
def attention_sink_head_stride(cfg: Constexpr[FmhaDecodeConfig], h_r: Int32) -> Int32:
    """Return the per-token local-head stride for attention-sink indexing."""
    if cutlass.const_expr(cfg.heads_q_per_kv != 0):
        return Int32(cfg.heads_q_per_kv)
    return h_r


@cute.jit
def local_head_from_q_output_row(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    output_row_idx: Int32,
) -> Int32:
    """Recover the local Q head from a packed output row."""
    _, local_head_idx = q_logical_output_row_token_and_local_head(
        cfg, h_r, output_row_idx
    )
    return local_head_idx


@cute.jit
def logical_head_batch(
    stage_info: StageInfo, fallback_h_k_idx: Int32, fallback_b_idx: Int32
) -> tuple[Int32, Int32]:
    """Resolve logical KV head and batch from work tile or static launch."""
    if cutlass.const_expr(stage_info.work_tile is not None):
        tile_idx = stage_info.work_tile.tile_idx
        return Int32(tile_idx[1]), Int32(tile_idx[2])
    return fallback_h_k_idx, fallback_b_idx


@cute.jit
def q_group_token_base(cfg: Constexpr[FmhaDecodeConfig], q_group_idx: Int32) -> Int32:
    """Return the first Q token owned by a logical Q CTA."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        return q_group_idx * Int32(cfg.q_tokens_per_cta)
    if cutlass.const_expr(cfg.is_multi_token_q):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        return q_group_idx // head_ctas_per_token
    return Int32(0)


@cute.jit
def q_logical_output_row_is_valid_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    logical_output_row_idx: Int32,
    seq_len_q: Int32,
) -> cutlass.Boolean:
    """Return whether a batch-local split scratch row owns packed output."""
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        return logical_output_row_idx < seq_len_q * Int32(cfg.heads_q_per_kv)
    return logical_output_row_idx < h_r


@cute.jit
def q_logical_output_row_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    logical_output_row_idx: Int32,
) -> tuple[Int32, Int32]:
    """Decompose one batch-local scratch row into token and local Q head.

    Split-KV scratch stores logical rows densely as ``[token, local_head]``.
    Packed final output needs that pair to replace the scratch buffer's padded
    batch stride with the public cumulative-token offset.
    """
    heads_q_per_kv = h_r
    if cutlass.const_expr(cfg.heads_q_per_kv != 0):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
    heads_q_per_kv_fdd = cute.fast_divmod_create_divisor(heads_q_per_kv)
    return divmod(logical_output_row_idx, heads_q_per_kv_fdd)


@cute.jit
def q_physical_output_row(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map a CTA-local Q row to the public output tensor's flat row.

    Fixed output is token-major ``[B, SQ, Hq, D]`` and packed variable-Q output
    is token-major ``[sumQ, Hq, D]``. Resolve the common token/local-head pair
    once so direct, split-GMEM, separate, and cluster final stores share the same
    ABI.
    """
    q_token_idx, local_head_idx = q_row_token_and_local_head(
        cfg, h_r, q_group_idx, tile_row_idx
    )
    return q_physical_output_row_from_token_and_local_head(
        cfg,
        h_r,
        num_heads_kv,
        batch_idx,
        kv_head_idx,
        q_token_idx,
        local_head_idx,
        q_token_offset,
    )


@cute.jit
def q_physical_output_row_from_logical(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    logical_output_row_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map a batch-local split scratch row to the physical output tensor."""
    q_token_idx, local_head_idx = q_logical_output_row_token_and_local_head(
        cfg, h_r, logical_output_row_idx
    )
    return q_physical_output_row_from_token_and_local_head(
        cfg,
        h_r,
        num_heads_kv,
        batch_idx,
        kv_head_idx,
        q_token_idx,
        local_head_idx,
        q_token_offset,
    )


@cute.jit
def q_physical_output_row_from_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    num_heads_kv: Int32,
    batch_idx: Int32,
    kv_head_idx: Int32,
    q_token_idx: Int32,
    local_head_idx: Int32,
    q_token_offset: Int32,
) -> Int32:
    """Map one token/local-head pair to the selected public O ABI."""
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
        num_heads_q = num_heads_kv * heads_q_per_kv
        global_head_idx = kv_head_idx * heads_q_per_kv + local_head_idx
        return (q_token_offset + q_token_idx) * num_heads_q + global_head_idx
    if cutlass.const_expr(cfg.is_multi_token_q):
        heads_q_per_kv = Int32(cfg.heads_q_per_kv)
        num_heads_q = num_heads_kv * heads_q_per_kv
        global_head_idx = kv_head_idx * heads_q_per_kv + local_head_idx
        return (
            batch_idx * Int32(cfg.max_seq_len_q) + q_token_idx
        ) * num_heads_q + global_head_idx
    return (batch_idx * num_heads_kv + kv_head_idx) * h_r + local_head_idx


@cute.jit
def q_row_is_valid_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
    seq_len_q: Int32,
) -> cutlass.Boolean:
    """Return whether a CTA-local Q row is valid for a runtime Q length."""
    return tile_row_idx < q_tile_valid_rows_for_seq(cfg, h_r, q_group_idx, seq_len_q)


@cute.jit
def q_row_token_and_local_head(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    tile_row_idx: Int32,
) -> tuple[Int32, Int32]:
    """Map a valid CTA-local Q row to its token and KV-local Q head."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        heads_q_per_kv_value = cfg.heads_q_per_kv
        heads_q_per_kv = Int32(heads_q_per_kv_value)
        if cutlass.const_expr(
            heads_q_per_kv_value > 0
            and (heads_q_per_kv_value & (heads_q_per_kv_value - 1)) == 0
        ):
            # Ratio-32 grouped decode lowers directly to shift/mask. Keep a
            # FastDivmod fallback for future non-power-of-two grouped profiles.
            shift = heads_q_per_kv_value.bit_length() - 1
            token_offset = tile_row_idx >> Int32(shift)
            local_head = tile_row_idx & Int32(heads_q_per_kv_value - 1)
        else:
            heads_q_per_kv_fdd = cute.fast_divmod_create_divisor(heads_q_per_kv)
            token_offset, local_head = divmod(tile_row_idx, heads_q_per_kv_fdd)
        return (
            q_group_idx * Int32(cfg.q_tokens_per_cta) + token_offset,
            local_head,
        )
    if cutlass.const_expr(cfg.is_multi_token_q):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        return (
            token_idx,
            head_cta_idx * Int32(cfg.tile_size_q) + tile_row_idx,
        )
    return (
        Int32(0),
        q_group_idx * Int32(cfg.tile_size_q) + tile_row_idx,
    )


@cute.jit
def q_seq_bounds(
    cfg: Constexpr[FmhaDecodeConfig],
    cu_seqlens_q: cute.Pointer | UniformQueryOffsets | None,
    batch_idx: Int32,
) -> tuple[Int32, Int32]:
    """Return packed Q token offset/length, or fixed-SQ neutral bounds."""
    if cutlass.const_expr(isinstance(cu_seqlens_q, UniformQueryOffsets)):
        length = Int32(cu_seqlens_q.seq_len_q)
        return Int32(batch_idx) * length, length
    if cutlass.const_expr(cfg.use_variable_seqlens_q):
        q_begin = Int32(cu_seqlens_q[batch_idx])
        q_end = Int32(cu_seqlens_q[batch_idx + Int32(1)])
        return q_begin, q_end - q_begin
    return Int32(0), Int32(cfg.max_seq_len_q)


@cute.jit
def q_tile_output_row_base(
    cfg: Constexpr[FmhaDecodeConfig], q_group_idx: Int32
) -> Int32:
    """Return the first packed output row owned by a logical Q CTA."""
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        return q_group_idx * Int32(cfg.q_tma_rows_per_cta)
    if cutlass.const_expr(cfg.is_multi_token_q):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        return token_idx * Int32(cfg.heads_q_per_kv) + head_cta_idx * Int32(
            cfg.tile_size_q
        )
    return q_group_idx * Int32(cfg.tile_size_q)


@cute.jit
def q_tile_valid_rows_for_seq(
    cfg: Constexpr[FmhaDecodeConfig],
    h_r: Int32,
    q_group_idx: Int32,
    seq_len_q: Int32,
) -> Int32:
    """Return runtime-valid MMA rows in one logical Q CTA.

    ``h_r`` is the total packed output-row count supplied to correction. Grouped
    profiles advance by complete tokens, so structural padding and the final
    partial token group are excluded independently from that global bound.
    """
    if cutlass.const_expr(cfg.uses_nontrivial_grouped_q_layout):
        token_base = q_group_idx * Int32(cfg.q_tokens_per_cta)
        remaining_tokens = cute.math.max(seq_len_q - token_base, Int32(0))
        valid_tokens = cute.math.min(remaining_tokens, Int32(cfg.q_tokens_per_cta))
        return valid_tokens * Int32(cfg.heads_q_per_kv)
    if cutlass.const_expr(cfg.is_multi_token_q):
        head_ctas_per_token = Int32(
            (cfg.heads_q_per_kv + cfg.tile_size_q - 1) // cfg.tile_size_q
        )
        token_idx = q_group_idx // head_ctas_per_token
        head_cta_idx = q_group_idx - token_idx * head_ctas_per_token
        head_base = head_cta_idx * Int32(cfg.tile_size_q)
        return cute.math.min(
            cute.math.max(Int32(cfg.heads_q_per_kv) - head_base, Int32(0)),
            Int32(cfg.tile_size_q),
        )
    return cute.math.min(
        cute.math.max(
            h_r - q_group_idx * Int32(cfg.tile_size_q),
            Int32(0),
        ),
        Int32(cfg.tile_size_q),
    )


@cute.jit
def resolve_logical_q_group_idx(
    cfg: Constexpr[FmhaDecodeConfig],
    stage_info: StageInfo,
    fallback_q_group_idx: Int32,
) -> Int32:
    """Resolve the q-group from the persistent work tile or static launch."""
    if cutlass.const_expr(cfg.uses_compact_split_grid):
        return Int32(cute.arch.block_idx()[2]) % Int32(cfg.num_q_ctas)
    if cutlass.const_expr(cfg.has_single_q_cta):
        # A split coordinate may still vary in grid X, but every physical CTA
        # maps to logical Q group zero. State this explicitly because the
        # release compiler does not infer the range from the launch geometry.
        return Int32(0)
    if cutlass.const_expr(stage_info.work_tile is not None):
        q_group_cta_idx = Int32(stage_info.work_tile.tile_idx[0])
        if cutlass.const_expr(cfg.use_split_kv):
            return q_group_cta_idx // Int32(cfg.splits_kv)
        return q_group_cta_idx
    return fallback_q_group_idx
