"""SM120-owned workspace ABI, modeled on the SM90 layout.

Activation scale views hold packed int32 words of four UE8M0 bytes, one
byte per K32 block inside a K128 row, as required by the SM120 block-scaled
``mma.sync kind::mxf8f6f4`` path. ``x_sf`` keeps the SM100 packed-UE8M0
public contract (capacity, hidden//128) int32.
"""
from dataclasses import dataclass
from types import MappingProxyType


BATCHED_MIN_ROWS_PER_EXPERT = 32
MAX_RELAY_BYTES = 256 * 1024 * 1024


def align(n, a=128):
    return (n + a - 1) // a * a


@dataclass(frozen=True)
class View:
    offset: int
    shape: tuple
    dtype: str
    nbytes: int


@dataclass(frozen=True)
class SM120GCBufferLayout:
    ranks: int
    experts: int
    capacity: int
    topk: int
    hidden: int
    intermediate: int
    shared_experts: int = 0

    def __post_init__(self):
        ranks, experts, capacity, topk = self.ranks, self.experts, self.capacity, self.topk
        hidden, intermediate, shared_experts = self.hidden, self.intermediate, self.shared_experts
        if not isinstance(shared_experts, int) or shared_experts < 0:
            raise ValueError('shared_experts must be a nonnegative integer')
        views = {}
        cursor = 0
        def reserve(name, shape, dtype, itemsize):
            nonlocal cursor
            cursor = align(cursor)
            size = itemsize
            for dim in shape:
                size *= dim
            views[name] = View(cursor, shape, dtype, size)
            object.__setattr__(self, name, cursor)
            cursor += size
        local = experts // ranks
        rows = ranks * capacity
        # epoch, input_ready, input_tokens, task_queue, finalize_count,
        # compute_done, finalized; all epochs monotonically increase.
        reserve('header', (32,), 'int32', 4)
        reserve('peer_table', (ranks,), 'int64', 8)
        reserve('expert_ready', (local,), 'int32', 4)
        reserve('expert_count', (local,), 'int32', 4)
        reserve('l1_done', (local,), 'int32', 4)
        reserve('active_expert', (local,), 'int32', 4)
        reserve('active_ready', (local,), 'int32', 4)
        reserve('compute_ready', (ranks,), 'int32', 4)
        reserve('combine_block_done', (1024,), 'int32', 4)
        reserve('send_count', (experts,), 'int32', 4)
        reserve('source_index', (local, ranks, capacity), 'int32', 4)
        reserve('x', (capacity, hidden), 'float8_e4m3fn', 1)
        reserve('x_sf', (capacity, hidden // 128), 'int32', 4)
        reserve('topk_idx', (capacity, topk), 'int64', 8)
        reserve('topk_weights', (capacity, topk), 'float32', 4)
        reserve('route_meta', (local, rows, 4), 'int32', 4)
        reserve('route_weight', (local, rows), 'float32', 4)
        reserve('l1_acts', (local, rows, hidden), 'float8_e4m3fn', 1)
        reserve('l1_acts_sf', (local, hidden // 128, rows), 'int32', 4)
        reserve('l2_acts', (local, rows, intermediate), 'float8_e4m3fn', 1)
        reserve('l2_acts_sf', (local, intermediate // 128, rows), 'int32', 4)
        # The direct path uses 64-by-64 BF16 cells. Relay returns use
        # position-major rows in the same arena.
        posblocks = align(capacity * topk, 64) // 64
        colblocks = hidden // 64
        reserve('combine', (posblocks * colblocks * 64 * 64 * 2,), 'uint8', 1)
        reserve('expert_base', (experts,), 'int32', 4)
        # Global handoff pool: N32 uses 16KB per CTA; N96 uses 32KB.
        # Each cell carries two f32/BF16 source tiles instead of shared tiles.
        reserve('handoff', (512, 2, 128, 32), 'float32', 4)
        reserve('pos_base', (local, ranks), 'int32', 4)
        reserve('slot_table', (capacity, topk), 'int32', 4)
        reserve('pos_table', (capacity, topk), 'int32', 4)
        # Streaming-dispatch scratch: the per-expert per-source row counts
        # staged once from the peers' send_count (block 0 reads the whole
        # matrix after scan-done so the expert sweep never re-crosses PCIe
        # for metadata), and the per-expert CTA arrival counters that pick
        # the command publisher.
        reserve('counts', (local, ranks), 'int32', 4)
        reserve('expert_arrive', (local,), 'int32', 4)
        if ranks > 1 and capacity * topk >= BATCHED_MIN_ROWS_PER_EXPERT * local:
            reserve('batched_routes_out', (capacity * topk,), 'int32', 4)
            reserve('batched_routes_in', (ranks, capacity * topk), 'int32', 4)
            if topk < ranks:
                reserve('batched_scales_in', (ranks, capacity, hidden // 128), 'int32', 4)
                reserve('batched_weights_in', (ranks, capacity, topk), 'float32', 4)
            relay_shape = (ranks, capacity * topk, hidden * 2)
            # At least one expected route per token and destination makes
            # batched returns worthwhile. Shared work can also hide the copy.
            if ((topk >= ranks or shared_experts)
                    and relay_shape[0] * relay_shape[1] * relay_shape[2] <= MAX_RELAY_BYTES):
                # FC2 stages rows by (source, source position, hidden column).
                # Size for every route reaching this rank, including skewed routing.
                reserve('relay_output', relay_shape, 'uint8', 1)
                reserve('relay_done', (local,), 'int32', 4)
        # Append shared storage so every routed offset remains identical.
        if shared_experts:
            shared_i = shared_experts * intermediate
            reserve('shared_fc1_done', ((capacity + 7) // 8,), 'int32', 4)
            reserve('shared_l2_acts', (capacity, shared_i), 'float8_e4m3fn', 1)
            reserve('shared_l2_acts_sf', (capacity, shared_i // 128), 'int32', 4)
            reserve('shared_y', (capacity, hidden), 'bfloat16', 2)
        object.__setattr__(self, 'views', MappingProxyType(views))
        object.__setattr__(self, 'nbytes', align(cursor))
        object.__setattr__(self, 'local', local)
        object.__setattr__(self, 'rows', rows)
