"""Compile-time plans for the SM90 NVFP4 and MXFP4 MegaMoE kernels.

Every field returned here becomes a template parameter of the CUDA kernel, so
the plan must be resolved on the Python side before the JIT module is built
(FlashInfer renders them into ``megamoe_sm90_nvfp4_config.inc`` via Jinja).

Tuning buckets select tile shapes and pipeline depths for H20 and H200.
Shapes outside those buckets use the generic scheduling policy.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

# SM90 shared-memory capacity, and the mailbox the interleaved scheduler
# appends after the dev-m barriers.
SMEM_CAPACITY = 232448
INTERLEAVED_SCHEDULER_SMEM_BYTES = 96

H20_NUM_SMS = 78
H200_NUM_SMS = 132

# Token-pool block-size candidates (the pool is padded so that any of these
# BLOCK_M values can address it).
CANDIDATE_BLOCK_M = (8, 16, 32, 64, 96, 128, 192)
MIN_CANDIDATE_BLOCK_M = 8
MAX_CANDIDATE_BLOCK_M = 192
LCM_CANDIDATE_BLOCK_M = 384

# (hidden, min_tokens, max_tokens, block_m, experts_per_wave, stages,
#  swap_ab, mode2_row_decoder, single_active_dispatch_warp)
_H20_FUSED_BUCKETS = (
    (4096, 1, 8, 8, 16, 6, True, True, True),
    (4096, 9, 16, 8, 32, 6, True, True, True),
    (4096, 17, 64, 24, 32, 6, True, True, True),
    (4096, 65, 128, 24, 16, 6, True, True, True),
    (7168, 1, 8, 8, 48, 6, True, True, True),
    (7168, 9, 16, 8, 24, 6, True, True, True),
    (7168, 17, 64, 24, 48, 6, True, True, True),
    (7168, 65, 191, 24, 16, 6, True, True, True),
    (6144, 1, 8, 8, 48, 6, True, True, True),
    (6144, 9, 16, 8, 24, 6, True, True, True),
    (6144, 17, 64, 24, 48, 6, True, True, True),
    (6144, 65, 144, 24, 16, 6, True, True, True),
)

# Counter-based dispatch policy: push dispatch replaces the first NVLink
# barrier, and the parity-rotated pool additionally drops the workspace-clean
# barrier. A step is enabled only in the M range where it was measured to win
# on the random router without losing on the all-experts router.
# (hidden, num_experts, num_topk, min_tokens, max_tokens, push, rotated_pool)
_H20_COUNTER_BUCKETS = (
    (4096, 256, 6, 1, 16, True, False),
    (4096, 256, 6, 17, 64, True, True),
    (7168, 384, 6, 1, 16, True, True),
    (6144, 384, 8, 1, 16, True, True),
    (6144, 384, 8, 17, 32, True, False),
)

# All-M arm table: which fused schedule to use. Every arm here runs the same
# fused kernel; they differ in scheduler mode and the RS epilogue.
#   "dynamic-rs"  -> interleaved scheduler + RS swap-AB (mode5)
#   "devm-dynamic"-> interleaved scheduler, no RS
#   "static-ss"   -> static scheduler, no RS
# (num_sms, hidden, num_experts, num_topk, min_tokens, max_tokens, arm)
_ALLM_ARMS = (
    (78, 4096, 256, 6, 1, 128, "dynamic-rs"),
    (78, 4096, 256, 6, 129, 1 << 30, "devm-dynamic"),
    (78, 7168, 384, 6, 1, 191, "dynamic-rs"),
    (78, 7168, 384, 6, 192, 1 << 30, "devm-dynamic"),
    (78, 6144, 384, 8, 1, 144, "dynamic-rs"),
    (78, 6144, 384, 8, 145, 1 << 30, "devm-dynamic"),
    (132, 4096, 256, 6, 1, 35, "dynamic-rs"),
    (132, 4096, 256, 6, 36, 64, "static-ss"),
    (132, 4096, 256, 6, 65, 1 << 30, "devm-dynamic"),
    (132, 7168, 384, 6, 1, 128, "dynamic-rs"),
    (132, 7168, 384, 6, 129, 1 << 30, "devm-dynamic"),
    (132, 6144, 384, 8, 1, 16, "dynamic-rs"),
    (132, 6144, 384, 8, 17, 32, "devm-dynamic"),
    (132, 6144, 384, 8, 33, 96, "dynamic-rs"),
    (132, 6144, 384, 8, 97, 1 << 30, "devm-dynamic"),
)


def align(x: int, a: int) -> int:
    return (x + a - 1) // a * a


def get_num_max_pool_tokens(
    num_ranks: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    num_experts_per_rank: int,
) -> int:
    """Worst-case token-pool capacity, padded so every BLOCK_M candidate fits."""
    num_max_recv = num_ranks * num_max_tokens_per_rank
    max_experts_per_token = min(num_topk, num_experts_per_rank)
    return align(
        num_max_recv * max_experts_per_token
        + num_experts_per_rank * (MAX_CANDIDATE_BLOCK_M - 1),
        LCM_CANDIDATE_BLOCK_M,
    )


def get_num_padded_sf_pool_tokens(num_max_pool_tokens: int, block_m: int) -> int:
    return (num_max_pool_tokens // block_m) * align(block_m, 128)


@dataclass(frozen=True)
class FusedPlan:
    block_m: int
    block_n: int
    num_experts_per_wave: int
    num_stages: int
    smem_size: int
    swap_ab: bool
    use_mode2_row_decoder: bool
    single_active_dispatch_warp: bool
    use_interleaved_scheduler: bool
    rs_swap_ab: bool
    push_dispatch: bool
    no_clean_barrier: bool
    num_max_pool_tokens: int
    num_padded_sf_pool_tokens: int


def select_mxfp4_fused_plan(
    num_sms: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
) -> FusedPlan:
    """Select MXFP4 tile geometry, pipeline depth and decode-warp partition.

    Chunk-major weight tiles are independent of the selected BLOCK_N.
    """
    if num_sms <= 1:
        raise ValueError("num_sms must be greater than one")
    if num_ranks <= 0 or num_ranks > 32 or num_ranks & (num_ranks - 1):
        raise ValueError("num_ranks must be a power of two in [1, 32]")
    if num_experts <= 0 or num_experts % num_ranks:
        raise ValueError("num_experts must divide evenly across ranks")
    if not 0 < num_topk <= min(32, num_experts):
        raise ValueError("num_topk must be in [1, min(32, num_experts)]")
    if (
        hidden <= 0
        or hidden % 256
        or intermediate_hidden <= 0
        or intermediate_hidden % 128
    ):
        raise ValueError(
            "hidden must be a multiple of 256 and intermediate_hidden of 128"
        )
    if not 0 < num_tokens <= num_max_tokens_per_rank:
        raise ValueError("num_tokens must be in (0, num_max_tokens_per_rank]")
    epr = num_experts // num_ranks

    def max_tokens(bm):
        return bm * num_experts // (num_ranks * num_topk)

    limit = max_tokens(24) * H20_NUM_SMS // num_sms
    bm = next((b for b in (8, 24, 32) if num_tokens <= max_tokens(b) * 7 // 10), 32)
    if num_tokens <= 1:
        epw, stages = 24, 8
    elif num_tokens <= 8:
        epw, stages = 16, 8
    elif num_tokens <= 16:
        epw, stages = 24, 8
    else:
        epw, stages = 48, 8
    swap = num_tokens <= max(16, limit)
    rs = swap
    if num_sms > H20_NUM_SMS and bm < 16 and num_tokens <= 16:
        rs = False
    if swap and bm >= 16:
        rs = False
    if not swap:
        bm, stages = 64, 3
    elif not rs:
        stages = 4 if bm == 8 else 3
    epw = min(epw, epr)
    while epr % epw:
        epw -= 1
    pool = get_num_max_pool_tokens(num_ranks, num_max_tokens_per_rank, num_topk, epr)
    return FusedPlan(
        block_m=bm,
        block_n=256,
        num_experts_per_wave=epw,
        num_stages=stages,
        smem_size=SMEM_CAPACITY,
        swap_ab=swap,
        use_mode2_row_decoder=True,
        single_active_dispatch_warp=swap,
        use_interleaved_scheduler=True,
        rs_swap_ab=rs,
        push_dispatch=False,
        no_clean_barrier=False,
        num_max_pool_tokens=pool,
        num_padded_sf_pool_tokens=max(
            get_num_padded_sf_pool_tokens(pool, b) for b in CANDIDATE_BLOCK_M
        ),
    )


def _select_arm(
    num_sms: int, hidden: int, num_experts: int, num_topk: int, m: int
) -> str:
    for sms, h, e, k, lo, hi, arm in _ALLM_ARMS:
        if (
            sms == num_sms
            and h == hidden
            and e == num_experts
            and k == num_topk
            and lo <= m <= hi
        ):
            return arm
    return "devm-dynamic"


def _select_counter_policy(
    num_sms: int, hidden: int, num_experts: int, num_topk: int, m: int
) -> Tuple[bool, bool]:
    """(push_dispatch, no_clean_barrier). H200 is unmeasured and keeps barriers.

    Matching on (hidden, num_experts, num_topk) rather than hidden alone: the
    payoff depends on the pushed row count (m * num_topk) and the per-expert
    pool pressure, so a model that merely shares a hidden size must not inherit
    a measured range. Unmatched shapes keep the always-correct barrier path.
    """
    if num_sms != H20_NUM_SMS:
        return False, False
    for h, e, k, lo, hi, push, rotated in _H20_COUNTER_BUCKETS:
        if h == hidden and e == num_experts and k == num_topk and lo <= m <= hi:
            return push, push and rotated
    return False, False


def select_fused_plan(
    num_sms: int,
    num_ranks: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    intermediate_hidden: int,
    num_max_tokens_per_rank: int,
    num_tokens: int,
    force_push_dispatch: Optional[bool] = None,
    force_no_clean_barrier: Optional[bool] = None,
) -> FusedPlan:
    if num_sms not in (H20_NUM_SMS, H200_NUM_SMS):
        raise ValueError(f"MegaMoE supports SM90 H20/H200 only, got num_sms={num_sms}")
    if num_ranks not in (4, 8):
        raise ValueError("NVFP4 MegaMoE requires 4 or 8 ranks")
    num_experts_per_rank = num_experts // num_ranks
    if num_experts_per_rank * num_ranks != num_experts:
        raise ValueError("num_experts must be divisible by num_ranks")
    supported_experts_per_rank = (32, 48) if num_ranks == 8 else (64, 96)
    if num_experts_per_rank not in supported_experts_per_rank:
        raise ValueError(f"unsupported experts per rank: {num_experts_per_rank}")
    if not 0 < num_tokens <= num_max_tokens_per_rank:
        raise ValueError("num_tokens must be in (0, num_max_tokens_per_rank]")

    # (block_m, block_n, experts per wave, stages, swap_ab, mode2, single warp)
    tuning: Optional[Tuple[int, int, int, int, bool, bool, bool]] = None
    if num_ranks == 8 and num_sms == H20_NUM_SMS:
        for h, lo, hi, bm, epw, stages, swap, mode2, single in _H20_FUSED_BUCKETS:
            if h == hidden and lo <= num_tokens <= hi:
                tuning = (bm, 256, epw, stages, swap, mode2, single)
                break
    if tuning is None:
        # Generic dev-m schedule: correct for any geometry, not individually tuned.
        if num_tokens <= 1:
            tuning = (8, 256, 24, 4, True, True, True)
        elif num_tokens <= 8:
            tuning = (8, 256, 16, 4, True, True, True)
        elif num_tokens <= 16:
            tuning = (8, 256, 24, 4, True, True, True)
        elif num_tokens <= 32:
            tuning = (16, 256, 48, 3, True, True, False)
        elif num_tokens <= 64:
            tuning = (24, 256, 48, 3, True, False, True)
        elif num_tokens <= 256:
            tuning = (64, 256, 48, 3, False, True, False)
        else:
            tuning = (128, 128, 48, 6, False, True, False)

    block_m, block_n, epw, stages, swap_ab, mode2, single_warp = tuning

    # An expert wave must divide the per-rank expert count.
    epw = min(epw, num_experts_per_rank)
    while epw < num_experts_per_rank and num_experts_per_rank % epw != 0:
        epw += 1

    # Pro at BLOCK_M=8 does not fit 4 stages.
    is_pro = (
        num_experts == 384
        and num_topk == 6
        and hidden == 7168
        and intermediate_hidden == 3072
    )
    if is_pro and block_m == 8 and stages == 4:
        stages = 3

    # Deployment arm and counter ranges were measured at EP8.
    arm = (
        _select_arm(num_sms, hidden, num_experts, num_topk, num_tokens)
        if num_ranks == 8
        else "devm-dynamic"
    )
    use_interleaved = arm != "static-ss"
    rs_swap_ab = arm == "dynamic-rs" and swap_ab

    push, no_clean = (
        _select_counter_policy(num_sms, hidden, num_experts, num_topk, num_tokens)
        if num_ranks == 8
        else (False, False)
    )
    if force_push_dispatch is not None:
        push = force_push_dispatch
    if force_no_clean_barrier is not None:
        no_clean = force_no_clean_barrier
    push = push and use_interleaved
    no_clean = no_clean and push

    num_max_pool_tokens = get_num_max_pool_tokens(
        num_ranks, num_max_tokens_per_rank, num_topk, num_experts_per_rank
    )
    # Push dispatch addresses the pool with a fixed stride per local expert, so
    # the worst case (every routed row of every rank landing on one expert)
    # must fit; otherwise fall back to the pull path.
    if push:
        pool_blocks = num_max_pool_tokens // block_m
        slots = 2 if no_clean else 1
        blocks_per_expert = pool_blocks // (num_experts_per_rank * slots)
        if num_tokens * num_ranks > blocks_per_expert * block_m:
            push = False
            no_clean = False

    smem_size = SMEM_CAPACITY
    if use_interleaved:
        smem_size = min(SMEM_CAPACITY + INTERLEAVED_SCHEDULER_SMEM_BYTES, SMEM_CAPACITY)

    return FusedPlan(
        block_m=block_m,
        block_n=block_n,
        num_experts_per_wave=epw,
        num_stages=stages,
        smem_size=smem_size,
        swap_ab=swap_ab,
        use_mode2_row_decoder=rs_swap_ab or mode2,
        single_active_dispatch_warp=single_warp,
        use_interleaved_scheduler=use_interleaved,
        rs_swap_ab=rs_swap_ab,
        push_dispatch=push,
        no_clean_barrier=no_clean,
        num_max_pool_tokens=num_max_pool_tokens,
        # The SF region is sized once, when the symmetric buffer is allocated, so
        # it must hold the largest layout any BLOCK_M candidate can produce --
        # not the one this plan happens to use.
        num_padded_sf_pool_tokens=max(
            get_num_padded_sf_pool_tokens(num_max_pool_tokens, bm)
            for bm in CANDIDATE_BLOCK_M
        ),
    )
