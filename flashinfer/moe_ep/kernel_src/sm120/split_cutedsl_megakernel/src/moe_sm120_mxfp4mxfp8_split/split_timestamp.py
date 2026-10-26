"""Low-overhead cross-Green-Context timestamp tracing."""

from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import Int64, T, dsl_user_op


TRACE_CTA_CAPACITY = 256
TRACE_FIELDS = 17
TRACE_ROLES = 4
TRACE_TMA_WAIT_SAMPLE_STRIDE = 8

# Optional dense K2 trace.  The normal CTA-level trace remains tiny and is
# suitable for repeated profiling.  MEGA_SPLIT_K2_TILE_TRACE=1 appends this
# per-tile region for one diagnostic launch.
K2_TILE_TRACE_CTA_CAPACITY = TRACE_CTA_CAPACITY
K2_TILE_TRACE_TILES_PER_CTA = 2048
K2_TILE_TRACE_FIELDS = 11

ROLE_K1 = 0
ROLE_K2 = 1
ROLE_K2_DRAIN = 2
ROLE_K2_FINALIZER = 3

FIELD_KERNEL_ENTRY = 0
FIELD_FIRST_WORK = 1
FIELD_LAST_WORK = 2
FIELD_KERNEL_EXIT = 3
FIELD_TILE_COUNT = 4
FIELD_FIRST_TILE_ID = 5
FIELD_LAST_TILE_ID = 6
FIELD_READY_WAIT_CALLS = 7
FIELD_READY_WAIT_NS = 8
FIELD_TMA_A_WAIT_CALLS = 9
FIELD_TMA_A_WAIT_NS = 10
FIELD_TMA_B_WAIT_CALLS = 11
FIELD_TMA_B_WAIT_NS = 12
FIELD_MAINLOOP_NS = 13
FIELD_STORE_NS = 14
FIELD_TMA_A_TIMED_CALLS = 15
FIELD_TMA_B_TIMED_CALLS = 16

K2_TILE_FIELD_TILE_ID = 0
K2_TILE_FIELD_DEQUEUE_BEGIN = 1
K2_TILE_FIELD_DEQUEUE_END = 2
K2_TILE_FIELD_TILE_BEGIN = 3
K2_TILE_FIELD_TMA_A_WAIT_NS = 4
K2_TILE_FIELD_TMA_B_WAIT_NS = 5
K2_TILE_FIELD_LDSM_QMMA_NS = 6
K2_TILE_FIELD_BF16_PACK_NS = 7
K2_TILE_FIELD_PEER_STORE_NS = 8
K2_TILE_FIELD_PHASE_ADVANCE_NS = 9
K2_TILE_FIELD_TILE_END = 10

TRACE_BASE_WORDS = TRACE_ROLES * TRACE_CTA_CAPACITY * TRACE_FIELDS
K2_TILE_TRACE_WORDS = (
    K2_TILE_TRACE_CTA_CAPACITY
    * K2_TILE_TRACE_TILES_PER_CTA
    * K2_TILE_TRACE_FIELDS
)

TILE_ID_COORD_BITS = 20
TILE_ID_COORD_BASE = 1 << TILE_ID_COORD_BITS
TILE_ID_EXPERT_BASE = 1 << (2 * TILE_ID_COORD_BITS)


@dsl_user_op
def read_globaltimer(*, loc=None, ip=None) -> Int64:
    """Read the GPU-global nanosecond timer."""
    return Int64(
        llvm.inline_asm(
            T.i64(),
            [],
            "mov.u64 $0, %globaltimer;",
            "=l",
            has_side_effects=True,
            asm_dialect=0,
            loc=loc,
            ip=ip,
        )
    )


def trace_word(role, cta_linear_id, field):
    return (
        role * TRACE_CTA_CAPACITY * TRACE_FIELDS
        + cta_linear_id * TRACE_FIELDS
        + field
    )


def k2_tile_trace_word(cta_linear_id, tile_seq, field):
    return (
        TRACE_BASE_WORDS
        + (
            cta_linear_id * K2_TILE_TRACE_TILES_PER_CTA
            + tile_seq
        )
        * K2_TILE_TRACE_FIELDS
        + field
    )


def pack_tile_id(expert_idx, tile_n_idx, tile_m_idx):
    """Pack scheduler coordinates into one host-decodable int64."""
    return (
        Int64(expert_idx) * TILE_ID_EXPERT_BASE
        + Int64(tile_n_idx) * TILE_ID_COORD_BASE
        + Int64(tile_m_idx)
    )
