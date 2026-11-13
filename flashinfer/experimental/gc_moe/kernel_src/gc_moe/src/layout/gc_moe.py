"""Byte-exact symmetric GC MoE allocation layout for the SM100 implementation.

Signals, routing workspace and tensor buffers share one symmetric allocation.
Ring slots retain generation counters until the finalize role clears them.
"""

from __future__ import annotations

from dataclasses import dataclass


CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192, 240)
TOKEN_ALIGNMENT = 1920
MAX_RANKS = 72
MAX_EXPERTS = 2048
MAX_EXPERTS_PER_RANK = 512
MAX_RING_BLOCKS = 1 << 20
MAX_SHARED_L2_BLOCKS = 1 << 15
# Accumulator barriers reserved here; the compute role indexes them by
# tile.epilogue_stages, so the two depths must agree.
ACCUM_BARRIER_STAGES = 2
# Role 3 uses one dispatch warpgroup and two combine warpgroups.
COMBINE_WARPS = 8
FINALIZE_THREADS = (4 + COMBINE_WARPS) * 32


def align(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class SharedStorageLayout:
    expert_counts: int
    pull: int
    cd: int
    data_a: int
    data_b: int
    data_sfa: int
    data_sfb: int
    amax: int
    tasks: int
    dispatch_bars: int
    data_full: int
    data_empty: int
    accum_full: int
    accum_empty: int
    combine_bars: int
    task_full: int
    task_empty: int
    tmem_ptr: int
    reusable_bytes: int
    field_end: int


def get_role_shared_layout(role: int, block_m: int, num_experts: int,
                           pull_bytes: int, stages: int, *,
                           store_m_l1: int, store_m_l2: int, dispatch_warps: int,
                           block_n: int, block_k: int,
                           sf_block_m: int) -> SharedStorageLayout:
    """Lay out one role's shared data and control fields in byte order."""
    if role not in (1, 2, 3) or block_m % 16 or block_m < 16 or stages < 2:
        raise ValueError('invalid shared-storage role geometry')
    values = {}
    cursor = 0

    def reserve(name: str, size: int, alignment: int) -> int:
        nonlocal cursor
        cursor = align(cursor, alignment)
        values[name] = cursor
        cursor += size
        return values[name]

    if role == 2:
        values['expert_counts'] = values['pull'] = 0
        cursor = 1
    else:
        reserve('expert_counts', num_experts * 4, 1024)
        reserve('pull', dispatch_warps * pull_bytes, 1024)
    reserve('cd', max(2 * 2 * store_m_l1 * (block_n // 2),
                      2 * store_m_l2 * block_n * 2), 1024)
    if role == 1:
        # Dispatch only pulls remote tokens into the ring; MMA stage planes
        # belong to the compute role.
        values['data_a'] = values['data_b'] = 0
        values['data_sfa'] = values['data_sfb'] = 0
    else:
        reserve('data_a', stages * (block_m // 2) * block_k, 1024)
        reserve('data_b', stages * block_n * block_k, 1024)
        reserve('data_sfa', stages * sf_block_m * 4, 4)
        reserve('data_sfb', stages * block_n * 4, 4)
    reserve('amax', 8 * store_m_l1 * 4, 8)
    reserve('tasks', 2 * 32, 16)
    reserve('dispatch_bars', dispatch_warps * 8, 8)
    reserve('data_full', stages * 8, 8)
    reserve('data_empty', stages * 8, 8)
    reserve('accum_full', ACCUM_BARRIER_STAGES * 8, 8)
    reserve('accum_empty', ACCUM_BARRIER_STAGES * 8, 8)
    reserve('combine_bars', COMBINE_WARPS * 8, 8)
    reserve('task_full', 2 * 8, 8)
    reserve('task_empty', 2 * 8, 8)
    reserve('tmem_ptr', 4, 4)
    return SharedStorageLayout(**values, reusable_bytes=values['dispatch_bars'],
                               field_end=cursor)


def _signal_offsets() -> dict[str, int]:
    fields = (
        ('grid_sync_count', 4, 4),
        ('l1_task_count', 2, 4),
        ('l2_task_count', 2, 4),
        ('shared_l1_task_count', 2, 4),
        ('shared_l2_task_count', 2, 4),
        # Persistent receive-generation parity: replay N reads/writes
        # generation parity & 1 of expert_recv_count/expert_recv_count_sum
        # (slot range [gen * count, (gen + 1) * count) inside each field's
        # dual-generation region); role 3 flips this word once per replay.
        # Lives in the header padding.  Role 1 publishes the role 3 epoch
        # at its entry, so role 3 reads it instead of a grid barrier.
        ('recv_parity', 1, 4),
        ('role3_epoch', 1, 4),
        ('routed_pool_blocks', 1, 4),
        ('combine_ready_grid_idx', MAX_RANKS, 8),
        ('expert_send_count', MAX_EXPERTS, 8),
        ('expert_recv_count', 2 * MAX_EXPERTS, 8),
        ('expert_recv_count_sum', 2 * MAX_EXPERTS_PER_RANK, 8),
        ('l1_full_count', MAX_RING_BLOCKS, 4),
        ('l1_empty_count', MAX_RING_BLOCKS, 4),
        ('l2_full_mask', MAX_RING_BLOCKS, 8),
        ('l2_empty_count', MAX_RING_BLOCKS, 4),
        ('shared_l2_full_count', MAX_SHARED_L2_BLOCKS, 4),
    )
    offset = 0
    result = {}
    for name, count, element_bytes in fields:
        field_alignment = 128 if name in (
            'combine_ready_grid_idx', 'expert_send_count'
        ) else element_bytes
        offset = align(offset, field_alignment)
        result[name] = offset
        offset += count * element_bytes
    result['size'] = align(offset, 128)
    return result


SIGNAL_OFFSETS = _signal_offsets()


@dataclass(frozen=True)
class BufferView:
    offset: int
    nbytes: int
    shape: tuple[int, ...]
    dtype: str
    stride: tuple[int, ...] | None = None


@dataclass(frozen=True)
class GCBufferLayout:
    num_ranks: int
    num_experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    ring_tokens: int
    sf_ring_tokens: int
    num_shared_experts: int = 0

    def __post_init__(self) -> None:
        ranks, experts, capacity = self.num_ranks, self.num_experts, self.capacity
        if not (0 < ranks <= MAX_RANKS and 0 < experts <= MAX_EXPERTS and
                experts % ranks == 0 and experts // ranks <= MAX_EXPERTS_PER_RANK):
            raise ValueError('invalid rank/expert configuration')
        if not (capacity > 0 and self.topk > 0 and
                self.topk + bool(self.num_shared_experts) <= 32):
            raise ValueError('invalid capacity/topk/shared configuration')
        if not (self.hidden > 0 and self.intermediate > 0 and
                self.hidden % 128 == 0 and self.intermediate % 128 == 0):
            raise ValueError('hidden dimensions must be positive multiples of 128')
        if self.ring_tokens <= 0 or self.ring_tokens % TOKEN_ALIGNMENT != 0:
            raise ValueError('ring token capacity must be positive and 1920-aligned')
        required_sf_ring_tokens = max(
            (self.ring_tokens // block_m) * align(block_m, 128)
            for block_m in CANDIDATE_BLOCK_M)
        if (self.sf_ring_tokens < required_sf_ring_tokens or
                self.sf_ring_tokens % 4 != 0):
            raise ValueError('invalid scale-factor ring capacity')
        if self.num_shared_experts < 0 or self.intermediate * self.num_shared_experts % 128:
            raise ValueError('invalid shared expert configuration')
        if self.ring_tokens > MAX_RING_BLOCKS * min(CANDIDATE_BLOCK_M):
            raise ValueError('ring signal capacity exceeded')
        if ceil_div(capacity, 8) > MAX_SHARED_L2_BLOCKS:
            raise ValueError('shared L2 signal capacity exceeded')

    @property
    def num_max_pool_tokens(self) -> int:
        local_experts = self.num_experts // self.num_ranks
        return align(self.num_ranks * self.capacity * min(self.topk, local_experts)
                     + local_experts * (max(CANDIDATE_BLOCK_M) - 1), TOKEN_ALIGNMENT)

    @property
    def workspace_offsets(self) -> dict[str, int]:
        source = SIGNAL_OFFSETS['size']
        metadata = source + self.num_experts * self.capacity * 4
        end = align(metadata + self.num_max_pool_tokens * 12, 16)
        return {'signals': 0, 'src_token_topk_idx': source,
                'token_src_metadata': metadata, 'end': end}

    @property
    def views(self) -> dict[str, BufferView]:
        c, h, i, s = self.capacity, self.hidden, self.intermediate, self.num_shared_experts
        r, sr = self.ring_tokens, self.sf_ring_tokens
        shared_sf = ceil_div(c, 8) * 128 if s else 0
        offset = self.workspace_offsets['end']
        result: dict[str, BufferView] = {}

        def add(name: str, nbytes: int, shape: tuple[int, ...], dtype: str,
                stride: tuple[int, ...] | None = None) -> None:
            nonlocal offset
            result[name] = BufferView(offset, nbytes, shape, dtype, stride)
            offset += nbytes

        add('x', c * h, (c, h), 'float8_e4m3fn')
        add('x_sf', c * h // 32, (c, h // 128), 'int32')
        add('topk_idx', c * self.topk * 8, (c, self.topk), 'int64')
        add('topk_weights', c * self.topk * 4, (c, self.topk), 'float32')
        result['shared_l1_acts'] = result['x']
        add('shared_l1_acts_sf', shared_sf * h // 32,
            (shared_sf, h // 128) if s else (0, h // 128), 'int32',
            (1, shared_sf) if s else None)
        add('shared_l2_acts', c * s * i, (c, s * i) if s else (0, 0), 'float8_e4m3fn')
        add('shared_l2_acts_sf', shared_sf * s * i // 32,
            (shared_sf, s * i // 128) if s else (0, 0), 'int32',
            (1, shared_sf) if s else None)
        add('l1_acts', r * h, (r, h), 'float8_e4m3fn')
        add('l1_acts_sf', sr * h // 32, (sr, h // 128), 'int32', (1, sr))
        add('l1_topk_weights', r * 4, (r,), 'float32')
        add('l2_acts', r * i, (r, i), 'float8_e4m3fn')
        add('l2_acts_sf', sr * i // 32, (sr, i // 128), 'int32', (1, sr))
        add('combine_tokens', (self.topk + bool(s)) * c * h * 2,
            (self.topk + bool(s), c, h), 'bfloat16')
        return result

    @property
    def nbytes(self) -> int:
        v = self.views['combine_tokens']
        return v.offset + v.nbytes
