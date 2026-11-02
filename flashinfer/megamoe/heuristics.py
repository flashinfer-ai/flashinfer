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

# H200 (EP8): push wins by about 10 us per call at small M (1.03-1.16x kernel time);
# at BM64 it loses (0.89-0.95x at M=256). Bounds are the largest M that does not lose on
# any measured router (uniform, global Zipf skew, hot experts on one rank). The rotated
# pool only pays on DeepSeek-V4-Pro above M=32; the clean barrier is ~1 us on H200.
_H200_COUNTER_BUCKETS = (
    (4096, 256, 6, 1, 32, True, False),
    (6144, 384, 8, 1, 32, True, False),
    (7168, 384, 6, 1, 32, True, False),
    (7168, 384, 6, 33, 64, True, True),
)

# Counter dispatch is validated only on the small-M plans (BLOCK_M <= 24, M <= 64); larger
# plans keep the barrier path even when forced. Unmeasured geometries push up to M=32,
# where every measured geometry wins on both GPUs.
_PUSH_MAX_TOKENS = 64
_PUSH_MAX_BLOCK_M = 24
_PUSH_DEFAULT_MAX_TOKENS = 32

# The H20 tile buckets were tuned (and are only legal) together with the RS arm of these
# geometries; any other geometry with the same hidden size uses the generic schedule.
_H20_TUNED_GEOMETRIES = {(4096, 256, 6), (7168, 384, 6), (6144, 384, 8)}


# Legal generic tiles from largest to smallest BLOCK_M (the kernel fixes the
# pipeline depth per BLOCK_M), used when the chosen tile overflows shared memory.
_GENERIC_FALLBACK_TUNINGS = (
    (64, 256, 48, 3, False, True, False),
    (24, 256, 48, 3, True, False, True),
    (16, 256, 48, 3, True, True, False),
    (8, 256, 24, 3, True, True, True),
)


def _fused_smem_bytes(plan, num_sms: int, num_experts: int, hidden: int) -> int:
    """Shared memory the fused kernel needs for ``plan``.

    Mirrors the layout in sm90_nvfp4_mega_moe_fused_body.inl (expert-count
    table, FP8 send rows, LUT, CD, the A / packed-B / decoded-B / SFA rings,
    the combine scratch it overlaps with, barriers and the scheduler mailbox);
    keep the two in sync.
    """
    align = lambda x, a: -(-x // a) * a  # noqa: E731
    bm, bn, stages = plan.block_m, plan.block_n, plan.num_stages
    split_m = bm == 128 and bn == 128
    wg_bm = bm // 2 if split_m else bm
    wg_bn = bn if split_m else bn // 2
    dispatch_warps = 1 if plan.single_active_dispatch_warp else 2
    decoded_b_stages = (
        0 if num_sms == 78 and plan.rs_swap_ab else (2 if split_m else stages)
    )
    cd_out = max(2 * wg_bm * (wg_bn // 2), bm * bn * 2 if plan.swap_ab else 0)
    cd_extra = max(2 * bm, bm * 8 if plan.swap_ab else 0) * 4
    gemm_end = (
        align(num_experts * 4, 1024)
        + align(hidden * dispatch_warps, 1024)
        + 1024
        + align(cd_out + cd_extra, 1024)
        + stages * (bm * 128 + bn * 80)
        + decoded_b_stages * bn * 128
        + stages * (2 if split_m else 1) * align(bm * 4, 128)
    )
    combine = 3 * 8 * hidden * 2 // (1 if hidden <= 4096 else 2)
    barriers = (2 + 2 * stages + 16) * 8
    scheduler = (
        INTERLEAVED_SCHEDULER_SMEM_BYTES if plan.use_interleaved_scheduler else 0
    )
    return max(gemm_end, combine) + barriers + scheduler


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
    """(push_dispatch, no_clean_barrier) from the measured per-GPU counter buckets.

    Matching on (hidden, num_experts, num_topk) rather than hidden alone: the
    payoff depends on the pushed row count (m * num_topk) and the per-expert
    pool pressure, so a model that merely shares a hidden size must not inherit
    a measured range. A measured geometry uses only its measured range; an
    unmeasured one pushes (without the rotated pool) up to M=32.
    """
    buckets = [
        b
        for b in {
            H20_NUM_SMS: _H20_COUNTER_BUCKETS,
            H200_NUM_SMS: _H200_COUNTER_BUCKETS,
        }.get(num_sms, ())
        if b[:3] == (hidden, num_experts, num_topk)
    ]
    if not buckets:
        return m <= _PUSH_DEFAULT_MAX_TOKENS, False
    for _, _, _, lo, hi, push, rotated in buckets:
        if lo <= m <= hi:
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
    if num_experts_per_rank < 1:
        raise ValueError("every rank needs at least one local expert")
    if not 0 < num_tokens <= num_max_tokens_per_rank:
        raise ValueError("num_tokens must be in (0, num_max_tokens_per_rank]")
    # Compile-time shape requirements of the fused kernel (scheduler / combine static asserts).
    if hidden % (512 if hidden > 4096 else 256) != 0:
        raise ValueError("hidden must be a multiple of 256 (512 above 4096)")
    if intermediate_hidden % 128 != 0 or 2 * intermediate_hidden > hidden:
        raise ValueError(
            "intermediate_hidden must be a multiple of 128 and at most hidden / 2"
        )

    # (block_m, block_n, experts per wave, stages, swap_ab, mode2, single warp)
    tuning: Optional[Tuple[int, int, int, int, bool, bool, bool]] = None
    if (
        num_ranks == 8
        and num_sms == H20_NUM_SMS
        and (hidden, num_experts, num_topk) in _H20_TUNED_GEOMETRIES
    ):
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

    def build(tuning) -> FusedPlan:
        block_m, block_n, epw, stages, swap_ab, mode2, single_warp = tuning

        # An expert wave must divide the per-rank expert count.
        epw = min(epw, num_experts_per_rank)
        while epw < num_experts_per_rank and num_experts_per_rank % epw != 0:
            epw += 1

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
        push = (
            push
            and use_interleaved
            and num_tokens <= _PUSH_MAX_TOKENS
            and block_m <= _PUSH_MAX_BLOCK_M
            and hidden % 1024 == 0  # push row copy: two 512-element chunks per lane
            and num_experts_per_rank
            <= num_sms  # SM e publishes local expert e's arrival counts
        )
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
            smem_size = min(
                SMEM_CAPACITY + INTERLEAVED_SCHEDULER_SMEM_BYTES, SMEM_CAPACITY
            )

        plan = FusedPlan(
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
        return plan

    # Large hidden sizes or expert counts (e.g. DeepSeek-V4-Pro, Kimi-K2) can
    # overflow SM90 shared memory with the chosen tile; fall back to the legal
    # BM8 3-stage pipeline or to smaller generic tiles until the layout fits.
    candidates = [tuning]
    if tuning[0] == 8 and tuning[3] == 4:
        candidates.append(tuning[:3] + (3,) + tuning[4:])
    candidates += [t for t in _GENERIC_FALLBACK_TUNINGS if t[0] < tuning[0]]
    for candidate in candidates:
        plan = build(candidate)
        if _fused_smem_bytes(plan, num_sms, num_experts, hidden) <= SMEM_CAPACITY:
            return plan
    raise ValueError(
        "MegaMoE fused kernel does not fit SM90 shared memory for "
        f"hidden={hidden}, num_experts={num_experts}"
    )
