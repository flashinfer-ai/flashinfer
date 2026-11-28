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

"""Primitives shared by the decode kernel's resources and tasks.

Holds the type aliases, packed-f32x2 partials, task-cache offsets and
accessors, warp/barrier primitives and small index helpers. The resource base
class is ``resources.common.DecodeGenResourceBase``.
"""

from functools import partial


import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op
from cutlass.experimental import primitives as prims

from cutlass.experimental.task_scheduling.resources import (
    StageInfo,
)

from ..config import FmhaDecodeConfig

Constexpr = cutlass.Constexpr
fadd2 = partial(cute.arch.add_packed_f32x2, ftz=False, rnd="rn")
fmul2 = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd="rn")
ffma2 = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd="rn")

TaskCache = tuple[
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
]
DescriptorValue = prims.Tcgen05SmemDesc | cutlass.Int64 | Int32
ResourceVarValue = (
    Int32 | Float32 | Uint32 | cutlass.Int64 | cutlass.Array | DescriptorValue
)
ResourceVars = dict[str, ResourceVarValue]


# Offsets into DecodeGenTask.make_task_cache(). Keeping these symbolic makes
# resource code explicit about which task-local lane or address value it needs.
TASK_CACHE_TMEM_BASE_OFFSET = 0
TASK_CACHE_WARP_GRP_THREAD_IDX = 1
TASK_CACHE_WARP_IDX = 2
TASK_CACHE_LANE_IDX = 3
TASK_CACHE_SEQ_LEN_KV = 4
TASK_CACHE_KV_REQUEST_BEGIN = 5
TASK_CACHE_KV_PAGE_IDX_UB = 6
TASK_CACHE_KV_RAW_TILE_BASE = 7
TASK_CACHE_KV_VALID_TILE_END = 8
TASK_CACHE_KV_WINDOW_START = 9
# Block-sparse rows share the two generic KV-span words with paged decode.
# Semantic aliases keep sparse consumers independent of page-table naming
# without extending the stable ten-word task-cache ABI.
TASK_CACHE_SPARSE_ROUTE_BEGIN = TASK_CACHE_KV_REQUEST_BEGIN
TASK_CACHE_SPARSE_ROUTE_COUNT = TASK_CACHE_KV_PAGE_IDX_UB


# Keeps softmax staging reserves bit 5 for the prepared route kind; the low
# four structural validity bits and bit 4 keep their meaning.
SOFTMAX_ROUTE_IS_PROXY_FLAG = 1 << 5


@dsl_user_op
def assume_nonnegative_i32(value: Int32, *, loc=None, ip=None) -> Int32:
    """Express a caller-guaranteed nonnegative Int32 contract to codegen."""

    condition = cutlass.Boolean(value >= Int32(0))
    llvm.intr_assume(
        condition.ir_value(loc=loc, ip=ip),
        [],
        [],
        loc=loc,
        ip=ip,
    )
    return value


def clamp_valid_tile_idx(cfg: FmhaDecodeConfig, tile_idx: Int32) -> Int32:
    """Clamp a static K/V tile index to the last valid tile."""
    return cute.math.min(tile_idx, Int32(cfg.total_kv_tiles - 1))


@cute.jit
def gather_byte_of_words(
    w0: Int32, w1: Int32, w2: Int32, w3: Int32, byte_idx: Int32
) -> Int32:
    """Return byte ``byte_idx`` (0-3) of each word, packed w0 low to w3 high.

    Two PRMTs pick the byte from (w0, w1) and (w2, w3); a third packs the
    resulting halves.
    """
    pair_sel = byte_idx | ((byte_idx + Int32(4)) << Int32(4))
    return cute.arch.inline_ptx(
        "{ .reg .b32 lo, hi; "
        "prmt.b32 lo, {$r0}, {$r1}, {$r4}; "
        "prmt.b32 hi, {$r2}, {$r3}, {$r4}; "
        "prmt.b32 {$w0}, lo, hi, 0x5410; }",
        write_only_types=[Int32],
        read_only_args=[w0, w1, w2, w3, pair_sel],
    )


@cute.jit
def get_task_cache(stage_info: StageInfo) -> TaskCache:
    """Return the task cache or a zero-filled placeholder cache."""
    if cutlass.const_expr(stage_info.task_cache is None):
        zero = Int32(0)
        return (
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
            zero,
        )
    return stage_info.task_cache


def is_last_loop_iteration(stage_info: StageInfo) -> cutlass.Boolean:
    """Return whether the current schedule loop iteration is the final one."""
    return stage_info.loop_offset + Int32(1) == stage_info.loop_end


@cute.jit
def is_proxy_route(route_flags: Int32) -> cutlass.Boolean:
    """Return whether staged Softmax route flags mark a proxy route.

    Keeps stages the flags as an ``Int32`` word. SWAP forwards them as a
    bit-preserving ``Uint32`` dataflow token, which the consumer bitcasts
    before calling.
    """
    return cutlass.Boolean(
        (route_flags & Int32(SOFTMAX_ROUTE_IS_PROXY_FLAG)) != Int32(0)
    )


@cute.jit
def load_nvfp4_row_scale_words(
    cfg: Constexpr[FmhaDecodeConfig],
    src_sf_base: cutlass.Array,
    row: Int32,
    head_dim_stage_idx: Int32 | int,
    *,
    is_v: Constexpr[bool],
) -> tuple[Int32, Int32]:
    """Load eight E4M3 scales for one 128-value row slice as two words.

    Scale rows sit at ``cfg.smem_kv_sf_row_offset``, which skips the padding
    of page fragments below 128 bytes.
    """
    sf_tile_bytes = cfg.smem_kv_sf_tile_bytes
    if cutlass.const_expr(cfg.uses_stage_sliced_nvfp4_sf):
        # Stage-sliced scales (dense page fragments): K keeps the 16 B half
        # row per token, V the stage's 32 B per 4-token group.
        if cutlass.const_expr(not is_v):
            k_scales = cutlass.Array(
                src_sf_base.data_ptr(),
                dtype=Int64,
                shape=(cfg.sf_stage_tile_bytes(False) // 8,),
                addrspace=3,
            )
            pair = k_scales[row * Int32(2) + (Int32(head_dim_stage_idx) & Int32(1))]
            return Int32(pair), Int32(pair >> Int64(32))
        v_scales = cutlass.Array(
            src_sf_base.data_ptr(),
            dtype=Int32,
            shape=(cfg.sf_stage_tile_bytes(True) // 4,),
            addrspace=3,
        )
        v_base = (row >> Int32(2)) * Int32(8)
        v_words = [v_scales[v_base + Int32(i)] for i in range(8)]
        v_token = row & Int32(3)
        return (
            gather_byte_of_words(
                v_words[0], v_words[1], v_words[2], v_words[3], v_token
            ),
            gather_byte_of_words(
                v_words[4], v_words[5], v_words[6], v_words[7], v_token
            ),
        )
    if cutlass.const_expr(not is_v):
        scales = cutlass.Array(
            src_sf_base.data_ptr(),
            dtype=Int64,
            shape=(sf_tile_bytes // 8,),
            addrspace=3,
        )
        # Scale rows of D128+ are whole 8-byte words.
        pair = scales[
            (cfg.smem_kv_sf_row_offset(row) >> Int32(3)) + Int32(head_dim_stage_idx)
        ]
        return Int32(pair), Int32(pair >> Int64(32))
    else:
        # V interleaves four tokens: each lane gathers its byte from eight words.
        scales = cutlass.Array(
            src_sf_base.data_ptr(),
            dtype=Int32,
            shape=(sf_tile_bytes // 4,),
            addrspace=3,
        )
        base = (cfg.smem_kv_sf_row_offset(row & Int32(~3)) >> Int32(2)) + Int32(
            head_dim_stage_idx * 8
        )
        words = [scales[base + Int32(i)] for i in range(8)]
        token = row & Int32(3)
        return (
            gather_byte_of_words(words[0], words[1], words[2], words[3], token),
            gather_byte_of_words(words[4], words[5], words[6], words[7], token),
        )


@cute.jit
def named_barrier_arrive(
    number_of_threads: Constexpr[int], barrier_id: Constexpr[int]
) -> None:
    """Arrive at a named barrier from the configured warp subset."""
    # The non-aligned primitive is required here: ordered softmax uses four
    # participating warps rather than a CTA-wide converged barrier.
    prims.barrier_cta_arrive(barrier_id, number_of_threads)


@cute.jit
def named_barrier_sync(
    number_of_threads: Constexpr[int], barrier_id: Constexpr[int]
) -> None:
    """Wait at a named barrier from the configured warp subset."""
    prims.barrier_cta_sync(barrier_id, thread_count=number_of_threads)


@cute.jit
def sparse_task_cache_route_begin(task_cache: TaskCache) -> Int32:
    """Load the first prepared-route ordinal for one sparse row."""

    return Int32(task_cache[TASK_CACHE_SPARSE_ROUTE_BEGIN])


@cute.jit
def sparse_task_cache_route_count(task_cache: TaskCache) -> Int32:
    """Load the live prepared-route count for one sparse row."""

    return Int32(task_cache[TASK_CACHE_SPARSE_ROUTE_COUNT])


@cute.jit
def split_kv_grid_coord(
    cfg: Constexpr[FmhaDecodeConfig], coord: cute.Coord
) -> cute.Coord:
    """Map compact split/Q/batch launches to capacity-strided task coordinates."""
    if cutlass.const_expr(cfg.uses_compact_split_grid):
        q_group = Int32(coord[2]) % Int32(cfg.num_q_ctas)
        batch = Int32(coord[2]) // Int32(cfg.num_q_ctas)
        return (q_group * Int32(cfg.splits_kv) + Int32(coord[0]), coord[1], batch)
    return coord


@cute.jit
def wait_for_mbarrier_phase(barrier, phase: Int32) -> None:
    """Wait for one parity of a CTA-local reusable mbarrier."""

    while not prims.mbarrier_try_wait_parity(
        barrier,
        phase,
        time_limit=10_000_000,
    ):
        pass


@cute.jit
def warp_broadcast_i32(value: Int32, source_lane: Constexpr[int]) -> Int32:
    """Broadcast one source-lane scalar as a warp-uniform Int32 value."""

    return cute.arch.make_warp_uniform(
        Int32(
            prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=value,
                offset=source_lane,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.IDX,
            )
        )
    )
