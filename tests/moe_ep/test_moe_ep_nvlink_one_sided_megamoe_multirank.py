"""W4A8 NVLinkOneSided MegaMoE (``sm103_mxfp8_mxfp4_bf16_nvlink_one_sided_cutedsl``) vs a Torch oracle.

Launch one rank per SM103 (B300) or SM100 (B200) GPU at the EP size under
test, with ``CUTE_DSL_ARCH`` naming the device's architecture (``sm_100a`` on
B200)::

    CUTE_DSL_ARCH=sm_103a torchrun --standalone --nproc_per_node=4 -m pytest \\
        tests/moe_ep/test_moe_ep_nvlink_one_sided_megamoe_multirank.py -v

(``run_tests.sh mega_sm103_nvlink_one_sided`` runs it at EP2 and EP4, and at EP8 when
eight such GPUs are visible. The dispatch-round cases need at most 32 local
experts, so they run at EP8 only and skip at smaller EP sizes, as do the
Kimi-K3 cases.)

Problem: DeepSeek-V4-Flash routed experts (hidden 4096, intermediate 2048, 256
experts, top-6, SwiGLU with clamp 10), MXFP4 weights and MXFP8 activations
with power-of-two UE8M0 scales per 32 values. The oracle all-gathers every
rank's quantized activations and routing over the EP group and evaluates its
local experts in two numerics:

- The kernel's rounding points: FC1 and SwiGLU in FP32, routing weight applied
  before MXFP8 requantization of the activation, FC2 partials rounded to BF16,
  top-k sum in FP32. The kernel agrees to a relative L2 error of about 2e-3,
  so one misrouted (token, expert) pair out of a full batch already fails the
  bound.
- Exact: every intermediate in FP32 (no rounding, no requantization). The
  kernel's own error is a relative L2 of about 3e-2.

Each case resolves to a different engine configuration: capacity 128 per rank
runs the Sentinel (data-as-flag) dispatch and combine, capacity 16 the same with
just-in-time work claims, capacity 256 the counter dispatch with the Sentinel
combine, and, pinned by a knob, both counter protocols; capacity 1024 runs the
schedule in groups of 256 FC1 tiles (at EP4 on the N128 token tile, at EP8
under the pull dispatch), a case pinned to the N240 token tile covers the widest
tile, at EP8 a case splits the counter dispatch into four rounds of local
experts, two cases pin the Sentinel protocols at 512 and 1024 tokens per rank,
where the top-k planners poll and scatter in several passes, and two cases pin
the pull dispatch at 1024 tokens per rank (at EP8 on N240 token tiles, whose
routes are spread evenly over each expert's tiles), fetched by walking CTAs of
the communication grid or by the main kernel's pull warps. The walking CTAs
store the rows into the landing slots that FC1 gathers from, or, in a case
pinned by a knob, into the token pool that FC1 loads by tiled TMA. A separate EP8 test
delays every round after the first and checks that the per-round gate still
matches the oracle while a negative control, which gates every expert on round 0
only, does not.

Die-local weights (each die's half of every weight matrix in its own HBM) are
on by default from 64 expected routes per local expert on a device with two
locality domains, so at EP4 and EP8 the grouped and 1024-token cases above run
them. A separate test sets them on and off through the backend config and
requires bit-identical outputs, with weights prepared by the backend, by the
public helper in the die-local layout, and in the plain layout (converted on
the first launch).

The in-kernel top-k reduce (``reduce_topk_in_kernel``, opt-in) adds every FC2
partial into its source token's BF16 output row as it is pushed. A separate test
runs it under the counter combine (pinned at 128 under the Sentinel dispatch and
at 256 under the counter dispatch, the grouped capacity, and the pull dispatch at
the grouped capacity, its walking CTAs storing rows into the landing slots or,
pinned by a knob, into the token pool) and bounds it against both oracles and
against the separate reduce's output of the same batch, over launches that
reuse the output with other token counts.

A shared expert (``shared_expert="fp8_in_kernel"``: one MXFP8 E4M3 expert of the
routed intermediate width that every token runs, fused into the main kernel)
adds its output to every token's sum; a separate test checks it against both
oracles (which add it with the routed experts' rounding points: SwiGLU in FP32
without a clamp, MXFP8 requantization, FC2 rounded to BF16) under every dispatch
and combine protocol, with its BF16 rounding knob, with every route masked so
the output is the shared expert's alone, under the in-kernel reduce, across
reruns and CUDA graph replays bit for bit.

Routing weights applied after FC2 (``routing_weights_after_fc2``, opt-in) leave
each route's SwiGLU output unweighted before its requantization and multiply its
FC2 partial instead: in FP32 on its BF16 combine slot in the top-k reduce, or on
its FP32 accumulator in the FC2 epilogue under the in-kernel reduce. A separate
test checks it against the requantized oracle in that order, and against the
default order's oracle as a control it must miss by more, under every dispatch
and combine protocol, with the shared expert, and across reruns and CUDA graph
replays.

Kimi-K3's routed experts (hidden 3584, intermediate 3072, 896 experts, top-16)
run in a separate EP8 test under perfect and random routing at 1, 16, 256, 1024
and 4096 tokens per rank: the Sentinel protocols, the counter dispatch with the
Sentinel combine, and the counter protocols on N192 and N240 token tiles, the
last with its top-k plane split over more exchange CTAs than the list-segment
bound allows, since 16 routes per token among 112 local experts would otherwise
outgrow SMEM. Each output is checked against both oracles, and reruns on both
landing sets and a CUDA graph replay reproduce it bit for bit.

Every case asserts the engine path it resolves to at EP2, EP4 and EP8 (see
``PATHS``), so a change to the capacity defaults that moves a case off its path
fails instead of leaving that path untested. Host-only registration and config
tests are in ``tests/cpu/moe_ep/test_moe_ep_nvlink_one_sided_megamoe_host.py``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import pytest

# Device capabilities the backend admits: SM100 (B200) and SM103 (B300).
BLACKWELL_CAPABILITIES = ((10, 0), (10, 3))
HIDDEN = 4096
INTERMEDIATE = 2048
EXPERTS = 256
TOPK = 6
CLAMP = 10.0
SF_VEC = 32


@dataclass(frozen=True)
class _Problem:
    """Routed experts: hidden size, intermediate size (after SwiGLU), experts, top-k."""

    hidden: int
    intermediate: int
    experts: int
    topk: int


V4_FLASH = _Problem(HIDDEN, INTERMEDIATE, EXPERTS, TOPK)
# Kimi-K3's routed experts: a hidden size of seven 512-byte runs, top-16 of 896
# experts (112 local experts at EP8, about 2.26 routes per reached rank).
KIMI_K3 = _Problem(hidden=3584, intermediate=3072, experts=896, topk=16)
# Relative L2 bounds over each rank's valid rows.
REQUANTIZED_TOLERANCE = 1e-2
EXACT_TOLERANCE = 6e-2
# The in-kernel top-k reduce rounds to BF16 at every add (about 2e-3 relative
# per route) instead of once per token, so its output sits a few 1e-3 from the
# separate reduce's.
IN_KERNEL_REDUCE_VS_SEPARATE_TOLERANCE = 1e-2
SENTINEL_CAPACITY = 128
# The largest capacity at which the Sentinel dispatch compiles just-in-time work
# claims by default (``JUST_IN_TIME_CLAIMS_MAX_TOKENS``, an even capacity, so
# padding for top-6 leaves it unchanged).
JUST_IN_TIME_CAPACITY = 16
# The counter dispatch with the Sentinel combine (by default at EP2 and EP4; at
# EP8 the Sentinel dispatch runs there, so COUNTER_DISPATCH_KNOBS pins it); the
# knob below switches the combine to the counter protocol as well.
COUNTER_DISPATCH_CAPACITY = 256
COUNTER_DISPATCH_KNOBS = {"dispatch_protocol": "counter"}
COUNTER_COMBINE_KNOBS = {"combine_protocol": "counter"}
COUNTER_PROTOCOLS_KNOBS = {**COUNTER_DISPATCH_KNOBS, **COUNTER_COMBINE_KNOBS}
# Above 256 tokens per rank FC1 tiles are scheduled in groups of 256. At EP4
# the expected expert load (96 tokens) also selects the N128 token tile with
# one FC2 TMA stage.
GROUPED_CAPACITY = 1024
# The token tile ``nvlink_one_sided_tiling`` selects while an expert expects about 172
# to 216 tokens (EP8 at 1024 tokens per rank expects 192).
N240_KNOBS = {"mma_tiler_mnk": (256, 240, 128), "fc2_tma_stages": 1}
# The Sentinel protocols above their default capacity. The W id planes of more
# than 2048 vectors (above about 170 tokens per rank at EP8) are polled in
# several passes, a score plane of more than 512 vectors is scattered in rounds
# beyond the ones held in registers, and at EP8 1024 tokens per rank take the
# N240 token tile, whose slots are strided.
SENTINEL_KNOBS = {
    "dispatch_protocol": "sentinel",
    "combine_protocol": "sentinel",
    "dispatch_rounds": 1,
}
SENTINEL_FORCED_CAPACITY = 512
# Target ranks fetch their rows from the source ranks' staged activations, under
# the counter combine in one round: walking CTAs of the communication grid, or the
# main kernel's pull warps.
PULL_KNOBS = {"dispatch_protocol": "pull", "dispatch_rounds": 1}
PULL_WARP_KNOBS = {**PULL_KNOBS, "pull_warps": 4}
# The walking CTAs storing rows at their routes' pool rows, which FC1 loads by
# tiled TMA (the default from 256 expected routes per expert).
PULL_POOL_KNOBS = {**PULL_KNOBS, "fc1_token_pool": True}
# The token pool under the strided N240 tile at every EP size, where its
# capacity applies: token tiles whose routes end past it fall back to Gather4
# from the landing slots. A capacity of one row sends every tile there, about
# half the routes a rank receives at 1024 tokens per rank sends about half of
# them, and 0 sizes the pool for any routing.
PULL_POOL_N240_KNOBS = {**PULL_POOL_KNOBS, **N240_KNOBS}
TOKEN_POOL_PARTIAL_ROWS = 3000
# The walking CTAs claiming landing slots, so a token routed to several local
# experts is fetched about once (at EP2 most tokens are), and the walking CTAs
# acquiring the expert-id releases themselves instead of the index-copy CTAs.
PULL_DEDUP_KNOBS = {**PULL_KNOBS, "pull_walk_dedup": True}
PULL_WALKER_ACQUIRE_KNOBS = {**PULL_KNOBS, "pull_acquire_in_copy": False}
# The walking CTAs counting landed routes per expert, with whole claims for the
# first wave: FC1 tiles wait for their whole expert.
PULL_EXPERT_GATE_KNOBS = {
    **PULL_KNOBS,
    "pull_tile_ready": False,
    "comm_pull_lead_rows": 8,
}
# Counter dispatch in four rounds of local experts. The processing order is
# ranked inside one warp, so it needs at most 32 local experts (EP8 and up).
ROUNDS_KNOBS = {"dispatch_protocol": "counter", "dispatch_rounds": 4}
# Fault injection for the round gate: every pusher holds the rounds after the
# first back by 50 us, long after the refine has reached the second round's
# experts. A refine block that does not wait for its own round then stages
# scale factors and releases rows that have not landed.
ROUND_DELAY_KNOBS = {**ROUNDS_KNOBS, "round_publish_delay_ns": 50_000}


# ----------------------------------------------------------------------------
# Quantization (power-of-two UE8M0 scales rounded up)
#
# The block maximum is multiplied by the reciprocal of the format maximum, as
# the frontend's quantizers do, so a maximum whose scale lands exactly on a
# power of two rounds the same way and both weight and input paths give the
# same bytes.
# ----------------------------------------------------------------------------


def _ceil_to_ue8m0(x):
    import torch

    bits = x.abs().float().view(torch.int32)
    exponent = ((bits >> 23) & 0xFF) + (bits & 0x7FFFFF).bool().int()
    return (exponent.clamp(1, 254) << 23).view(torch.float32)


def _quantize_mxfp8(x):
    """[M, K] -> E4M3 [M, K] and FP32 power-of-two scales [M, K/32]."""
    import torch

    m, k = x.shape
    blocks = x.float().view(m, k // SF_VEC, SF_VEC)
    sf = _ceil_to_ue8m0(blocks.abs().amax(dim=2).clamp(min=1e-4) * (1.0 / 448.0))
    data = (blocks / sf.unsqueeze(2)).to(torch.float8_e4m3fn).view(m, k)
    return data, sf


_E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _quantize_mxfp4(w):
    """[N, K] -> packed E2M1 uint8 [N, K/2] (low nibble = even k), FP32 scales [N, K/32]."""
    import torch

    n, k = w.shape
    blocks = w.float().view(n, k // SF_VEC, SF_VEC)
    sf = _ceil_to_ue8m0(blocks.abs().amax(dim=2).clamp(min=1e-4) * (1.0 / 6.0))
    scaled = (blocks / sf.unsqueeze(2)).view(n, k)
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.uint8)
    # Round to nearest, ties to the even code, at the midpoints of the E2M1 grid.
    for boundary, inclusive in zip(
        (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0),
        (False, True, False, True, False, True, False),
        strict=True,
    ):
        code += (magnitude >= boundary if inclusive else magnitude > boundary).to(
            torch.uint8
        )
    code |= ((scaled < 0) & (code != 0)).to(torch.uint8) << 3
    pairs = code.view(n, k // 2, 2)
    return pairs[..., 0] | (pairs[..., 1] << 4), sf


def _dequantize_mxfp8(data, sf):
    m, k = data.shape
    return (data.float().view(m, k // SF_VEC, SF_VEC) * sf.unsqueeze(2)).view(m, k)


def _dequantize_mxfp4(packed, sf):
    import torch

    n, half_k = packed.shape
    table = torch.tensor(_E2M1_VALUES, device=packed.device)
    codes = torch.stack((packed & 0x0F, packed >> 4), dim=-1).view(n, 2 * half_k).long()
    values = table[codes & 0x7] * torch.where(codes >= 8, -1.0, 1.0)
    return (values.view(n, -1, SF_VEC) * sf.unsqueeze(2)).view(n, 2 * half_k)


def _e8m0_bytes(sf):
    """FP32 power-of-two scales -> E8M0 codes (uint8)."""
    import torch

    return (sf.view(torch.int32) >> 23).to(torch.uint8)


def _e8m0_to_f32(codes):
    """E8M0 codes -> FP32 scales; code 255 is NaN."""
    import torch

    bits = codes.to(torch.int32) << 23
    return torch.where(codes == 255, 0x7FC00000, bits).view(torch.float32)


# ----------------------------------------------------------------------------
# Problem
# ----------------------------------------------------------------------------


@dataclass
class _Weights:
    """One rank's local experts in canonical layout (FC1 rows are [gate; up])."""

    w13_bf16: object  # [E, 2I, H] BF16 source weights
    w2_bf16: object  # [E, H, I]
    w13: object  # [E, 2I, H/2] packed E2M1
    w13_sf: object  # [E, 2I, H/32] FP32 power-of-two scales
    w2: object  # [E, H, I/2]
    w2_sf: object  # [E, H, I/32]
    w13_dequant: object  # [E, 2I, H] BF16, exact
    w2_dequant: object  # [E, H, I] BF16, exact


@dataclass
class _Batch:
    count: int
    x_bf16: object  # [T, H]
    x_fp8: object  # [T, H] float8_e4m3fn
    x_sf_codes: object  # [T, H/32] uint8 E8M0
    topk_ids: object  # [T, K] int64, -1 masks a route
    topk_weights: object  # [T, K] float32


def _make_weights(
    local_experts: int, seed: int, problem: _Problem = V4_FLASH
) -> _Weights:
    import torch

    hidden, intermediate = problem.hidden, problem.intermediate
    generator = torch.Generator(device="cuda").manual_seed(seed)
    # FC1 outputs have a standard deviation of about 4, so the clamp at 10 is
    # active on a few percent of the gate and up values.
    w13 = (
        torch.randn(
            local_experts,
            2 * intermediate,
            hidden,
            device="cuda",
            generator=generator,
        )
        * (4.0 / hidden**0.5)
    ).bfloat16()
    w2 = (
        torch.randn(
            local_experts, hidden, intermediate, device="cuda", generator=generator
        )
        / intermediate**0.5
    ).bfloat16()
    q13, s13, d13, q2, s2, d2 = [], [], [], [], [], []
    for expert in range(local_experts):
        q, s = _quantize_mxfp4(w13[expert])
        q13.append(q)
        s13.append(s)
        d13.append(_dequantize_mxfp4(q, s).bfloat16())
        q, s = _quantize_mxfp4(w2[expert])
        q2.append(q)
        s2.append(s)
        d2.append(_dequantize_mxfp4(q, s).bfloat16())
    return _Weights(
        w13,
        w2,
        torch.stack(q13),
        torch.stack(s13),
        torch.stack(q2),
        torch.stack(s2),
        torch.stack(d13),
        torch.stack(d2),
    )


@dataclass
class _SharedExpert:
    """The shared expert, identical on every rank: MXFP8 E4M3 weights with UE8M0 scales per 32 values."""

    w13: object  # [2I, H] float8_e4m3fn, gate rows then up rows
    w13_sf: object  # [2I, H/32] FP32 power-of-two scales
    w2: object  # [H, I] float8_e4m3fn
    w2_sf: object  # [H, I/32]
    w13_dequant: object  # [2I, H] FP32
    w2_dequant: object  # [H, I] FP32
    # The layer's shared_gate_up_clamp, which the oracles apply.
    clamp: Optional[float] = None
    # The layer's shared_bf16_rounding, which the requantized oracle follows.
    bf16_rounding: bool = False


def _make_shared_expert(seed: int) -> _SharedExpert:
    import torch

    generator = torch.Generator(device="cuda").manual_seed(seed)
    w13 = torch.randn(2 * INTERMEDIATE, HIDDEN, device="cuda", generator=generator) * (
        4.0 / HIDDEN**0.5
    )
    w2 = (
        torch.randn(HIDDEN, INTERMEDIATE, device="cuda", generator=generator)
        / INTERMEDIATE**0.5
    )
    q13, s13 = _quantize_mxfp8(w13)
    q2, s2 = _quantize_mxfp8(w2)
    return _SharedExpert(
        q13, s13, q2, s2, _dequantize_mxfp8(q13, s13), _dequantize_mxfp8(q2, s2)
    )


def _shared_pack(weights: _Weights, shared: _SharedExpert):
    import torch

    from flashinfer.moe_ep import PrequantizedMoEWeightsWithSharedExpert

    # Typed and raw-byte spellings of the shared expert's planes are both accepted.
    return PrequantizedMoEWeightsWithSharedExpert(
        weights.w13,
        weights.w2.view(torch.int8),
        _e8m0_bytes(weights.w13_sf).view(torch.float8_e8m0fnu),
        _e8m0_bytes(weights.w2_sf),
        shared_w13=shared.w13,
        shared_w2=shared.w2.view(torch.uint8),
        shared_w13_scale=_e8m0_bytes(shared.w13_sf).view(torch.float8_e8m0fnu),
        shared_w2_scale=_e8m0_bytes(shared.w2_sf),
    )


def _make_batch(
    count: int,
    seed: int,
    *,
    holes: bool = False,
    skewed: bool = False,
    unrouted: bool = False,
    problem: _Problem = V4_FLASH,
    perfect: Optional[tuple[int, int]] = None,
) -> _Batch:
    """Random routing, with ``skewed`` every route on experts ``0..topk-1``, or
    with ``unrouted`` every route masked.

    ``perfect=(rank, world)`` sends route slot ``s`` (``token * topk + k``) of
    that EP rank to rank ``(s + rank) % world`` and there to local expert
    ``(s + rank) // world`` modulo the local experts, so every source rank's
    routes spread evenly over the ranks and walk each rank's experts in turn.
    """
    import torch

    topk = problem.topk
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        count, problem.hidden, device="cuda", generator=generator
    ).bfloat16()
    data, sf = _quantize_mxfp8(x)
    scores = torch.randn(count, problem.experts, device="cuda", generator=generator)
    values, ids = torch.topk(scores, topk, dim=-1)
    if perfect is not None:
        rank, world = perfect
        local = problem.experts // world
        slots = torch.arange(count * topk, device="cuda") + rank
        ids = ((slots % world) * local + (slots // world) % local).view(count, topk)
        values = torch.gather(scores, 1, ids)
    weights = torch.softmax(values, dim=-1).float()
    if skewed:
        # Experts 0..topk-1 all live on EP rank 0; each token visits them in
        # its own slot order.
        ids = torch.argsort(
            torch.rand(count, topk, device="cuda", generator=generator), dim=-1
        )
    if holes:
        ids[::3, -1] = -1
        ids[1::4, 0] = -1
    if unrouted:
        ids.fill_(-1)
    return _Batch(count, x, data, _e8m0_bytes(sf), ids.contiguous(), weights)


def _tensors(batch: _Batch, *, quantize_input: bool):
    from flashinfer.moe_ep import MoEEpTensors

    return MoEEpTensors(
        hidden_states=batch.x_bf16 if quantize_input else batch.x_fp8,
        scales=None if quantize_input else batch.x_sf_codes,
        topk_ids=batch.topk_ids.clone(),
        topk_weights=batch.topk_weights.clone(),
    )


# ----------------------------------------------------------------------------
# Oracle
# ----------------------------------------------------------------------------


def _reference(
    batch: _Batch,
    weights: _Weights,
    capacity: int,
    group,
    shared=None,
    weight_after_fc2=False,
    problem: _Problem = V4_FLASH,
):
    """(requantized-numerics, exact) FP32 outputs of this rank's rows (collective).

    With ``weight_after_fc2`` the requantized oracle applies each route's
    routing weight to its BF16-rounded FC2 partial instead of to its SwiGLU
    output before the requantization.

    With ``shared`` (a ``_SharedExpert``) every row adds the shared expert's
    output, in the routed experts' numerics with its own clamp and no weight;
    with its ``bf16_rounding`` the requantized oracle rounds the shared gate,
    up and SwiGLU product and the routes' sum to BF16 before the shared output
    joins it.
    """
    import torch
    import torch.distributed as dist

    world = dist.get_world_size(group)
    rank = dist.get_rank(group)
    hidden, intermediate, topk = problem.hidden, problem.intermediate, problem.topk
    local = problem.experts // world
    count = batch.count

    def padded(tensor, fill):
        out = torch.full(
            (capacity, *tensor.shape[1:]), fill, dtype=tensor.dtype, device="cuda"
        )
        out[:count].copy_(tensor)
        return out

    def gathered(tensor):
        parts = [torch.empty_like(tensor) for _ in range(world)]
        dist.all_gather(parts, tensor, group=group)
        return torch.cat(parts)

    data = gathered(padded(batch.x_fp8.view(torch.uint8), 0))
    codes = gathered(padded(batch.x_sf_codes, 127))
    ids = gathered(padded(batch.topk_ids, -1))
    routing = gathered(padded(batch.topk_weights, 0.0))
    x = _dequantize_mxfp8(data.view(torch.float8_e4m3fn), _e8m0_to_f32(codes))
    partial = torch.zeros(2, world * capacity, topk, hidden, device="cuda")
    for expert in range(local):
        token, slot = torch.nonzero(ids == rank * local + expert, as_tuple=True)
        if token.numel() == 0:
            continue
        w1 = weights.w13_dequant[expert].float()
        w2 = weights.w2_dequant[expert].float()
        fc1 = x[token] @ w1.T
        scale = routing[token, slot].unsqueeze(1)
        gate = fc1[:, :intermediate].clamp(max=CLAMP)
        up = fc1[:, intermediate:].clamp(-CLAMP, CLAMP)
        act = gate * torch.sigmoid(gate) * up
        if weight_after_fc2:
            requantized = _dequantize_mxfp8(*_quantize_mxfp8(act)) @ w2.T
            partial[0, token, slot] = requantized.bfloat16().float() * scale
        else:
            act = act * scale
            requantized = _dequantize_mxfp8(*_quantize_mxfp8(act)) @ w2.T
            partial[0, token, slot] = requantized.bfloat16().float()
        partial[1, token, slot] = (act @ w2.T) * (scale if weight_after_fc2 else 1.0)
    dist.all_reduce(partial, group=group)
    mine = partial[:, rank * capacity : rank * capacity + count].sum(dim=2)
    if shared is not None and count:
        local_x = x[rank * capacity : rank * capacity + count]
        fc1 = local_x @ shared.w13_dequant.T

        def activation(fc1, rounded):
            gate, up = fc1[:, :intermediate], fc1[:, intermediate:]
            if rounded:
                gate, up = gate.bfloat16().float(), up.bfloat16().float()
            if shared.clamp is not None:
                gate, up = (
                    gate.clamp(max=shared.clamp),
                    up.clamp(-shared.clamp, shared.clamp),
                )
            act = gate * torch.sigmoid(gate) * up
            return act.bfloat16().float() if rounded else act

        act = activation(fc1, shared.bf16_rounding)
        if shared.bf16_rounding:
            mine[0] = mine[0].bfloat16().float()
        mine[0] += (
            (_dequantize_mxfp8(*_quantize_mxfp8(act)) @ shared.w2_dequant.T)
            .bfloat16()
            .float()
        )
        mine[1] += activation(fc1, False) @ shared.w2_dequant.T
    return mine[0], mine[1]


def _relative_l2(actual, expected) -> float:
    actual, expected = actual.float(), expected.float()
    return float((actual - expected).norm() / expected.norm().clamp_min(1e-30))


def _assert_all_ranks(condition: bool, group, message: str) -> None:
    """Assert ``condition`` on every rank of ``group`` together, so none hangs."""
    import torch
    import torch.distributed as dist

    flag = torch.tensor([float(bool(condition))], device="cuda")
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=group)
    assert flag.item() == 1.0, message


def _oracle_errors(
    output,
    batch,
    weights,
    capacity,
    group,
    *,
    skip_rows=0,
    label="",
    shared=None,
    weight_after_fc2=False,
    problem: _Problem = V4_FLASH,
):
    """(nonfinite, relL2 vs requantized, relL2 vs exact) over every rank's valid rows.

    Each value is the maximum over the ranks of ``group`` (collective).
    """
    import torch
    import torch.distributed as dist

    requantized, exact = _reference(
        batch, weights, capacity, group, shared, weight_after_fc2, problem
    )
    rows = slice(skip_rows, batch.count)
    error = torch.zeros(3, device="cuda")
    if batch.count > skip_rows:
        actual = output[: batch.count][rows]
        error[0] = float(not bool(torch.isfinite(actual).all()))
        error[1] = _relative_l2(actual, requantized[rows])
        error[2] = _relative_l2(actual, exact[rows])
    dist.all_reduce(error, op=dist.ReduceOp.MAX, group=group)
    nonfinite, requantized_error, exact_error = error.tolist()
    if dist.get_rank(group) == 0:
        print(
            f"[EP{dist.get_world_size(group)} capacity {capacity}] {label}: relL2 "
            f"{requantized_error:.2e} vs requantized, {exact_error:.2e} vs exact"
        )
    return nonfinite, requantized_error, exact_error


def _within_bounds(errors) -> bool:
    nonfinite, requantized_error, exact_error = errors
    return (
        nonfinite == 0
        and requantized_error < REQUANTIZED_TOLERANCE
        and exact_error < EXACT_TOLERANCE
    )


def _check(
    output,
    batch,
    weights,
    capacity,
    group,
    *,
    skip_rows=0,
    label="",
    shared=None,
    weight_after_fc2=False,
    problem: _Problem = V4_FLASH,
):
    """Collectively assert both oracle bounds on every rank's valid rows."""
    nonfinite, requantized_error, exact_error = _oracle_errors(
        output,
        batch,
        weights,
        capacity,
        group,
        skip_rows=skip_rows,
        label=label,
        shared=shared,
        weight_after_fc2=weight_after_fc2,
        problem=problem,
    )
    assert nonfinite == 0, f"{label}: nonfinite output"
    assert requantized_error < REQUANTIZED_TOLERANCE, (
        f"{label}: relL2 {requantized_error:.3e} vs requantized numerics"
    )
    assert exact_error < EXACT_TOLERANCE, (
        f"{label}: relL2 {exact_error:.3e} vs exact FP32"
    )


# ----------------------------------------------------------------------------
# Harness
# ----------------------------------------------------------------------------


def _launcher_ranks() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _init_distributed():
    import torch
    import torch.distributed as dist

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    if torch.cuda.get_device_capability() not in BLACKWELL_CAPABILITIES:
        pytest.skip("the NVLinkOneSided MegaMoE requires SM100 or SM103")
    rank, world = _launcher_ranks()
    if world < 2 or EXPERTS % world:
        pytest.skip("needs a torchrun launch with an EP size dividing 256")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl", device_id=torch.device("cuda", local_rank)
        )
    return rank, world


def _layer(
    capacity: int,
    weights,
    *,
    rank: int,
    world: int,
    quantize_input: bool,
    process_group=None,
    transformed=None,
    knobs=None,
    die_local_weights=None,
    reduce_topk_in_kernel=False,
    fc1_token_pool_capacity=None,
    problem: _Problem = V4_FLASH,
    **shared_fields,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig,
    )

    return MoEEpLayer(
        BootstrapConfig(world_size=world, rank=rank, process_group=process_group),
        FleetParams(
            num_experts=problem.experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=problem.hidden,
        ),
        weights,
        backend=MegaConfig(
            megakernel=Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig(
                intermediate_size=problem.intermediate,
                top_k=problem.topk,
                activation_clamp=CLAMP,
                knobs=knobs,
                die_local_weights=die_local_weights,
                reduce_topk_in_kernel=reduce_topk_in_kernel,
                fc1_token_pool_capacity=fc1_token_pool_capacity,
                **shared_fields,
            ),
            quantize_input=quantize_input,
            preprocess_weights=transformed is None,
            transformed_weights=transformed,
        ),
    )


def _prequantized_pack(weights: _Weights):
    import torch

    from flashinfer.moe_ep import MoEWeightPack

    # One typed and one raw-byte scale plane: both spellings are accepted.
    return MoEWeightPack(
        weights.w13,
        weights.w2.view(torch.int8),
        _e8m0_bytes(weights.w13_sf).view(torch.float8_e8m0fnu),
        _e8m0_bytes(weights.w2_sf),
    )


# ----------------------------------------------------------------------------
# Engine paths
# ----------------------------------------------------------------------------

EP_SIZES = (2, 4, 8)

# The engine path every case resolves to at EP2, EP4 and EP8 (None where the
# case skips), keyed by test and case id, as ``_path`` spells it: dispatch/combine
# protocols, token tile N (and its N128 alternate), dispatch rounds, schedule
# policy and hint (with pipelined groups), then just-in-time work claims, the
# pull dispatch's fetchers (walking CTAs or main-kernel pull warps) and FC1's
# token pool where they apply.


def _same(path: str) -> tuple[str, str, str]:
    return (path, path, path)


# Up to 256 tokens per rank one schedule group holds every FC1 tile.
SENTINEL_PATH = _same("sentinel/sentinel N64 rounds1 grouped4096")
JIT_PATH = _same("sentinel/sentinel N64 rounds1 grouped4096 jit")
COUNTER_DISPATCH_PATH = _same("counter/sentinel N64 rounds1 grouped4096")
SENTINEL_DISPATCH_PATH = _same("sentinel/counter N64 rounds1 grouped4096")
COUNTER_PROTOCOLS_PATH = _same("counter/counter N64 rounds1 grouped4096")
# 256 tokens per rank: the counter dispatch at EP2 and EP4, the Sentinel one at EP8.
CAPACITY_256_PATH = (*COUNTER_DISPATCH_PATH[:2], SENTINEL_PATH[2])
N240_PATH = (
    "counter/sentinel N240|N128 rounds1 grouped4096",
    "counter/sentinel N240|N128 rounds1 grouped4096",
    "sentinel/sentinel N240|N128 rounds1 grouped4096",
)
SENTINEL_512_PATH = (
    "sentinel/sentinel N64 rounds1 grouped256 jit",
    "sentinel/sentinel N64 rounds1 grouped256 jit",
    "sentinel/sentinel N128 rounds1 grouped256 jit",
)
# 1024 tokens per rank: pipelined groups of FC1 tiles and just-in-time claims,
# with a token tile that grows with the EP size (N240 and its N128 alternate at
# EP8); by default the counter protocols at EP2 and EP4, the pull dispatch at EP8.
GROUPED_PATH = (
    "counter/counter N64 rounds1 grouped256+pipelined jit",
    "counter/counter N128 rounds1 grouped256+pipelined jit",
    "pull/counter N240|N128 rounds1 grouped128+pipelined jit walkers",
)
SENTINEL_1024_PATH = (
    "sentinel/sentinel N64 rounds1 grouped256+pipelined jit",
    "sentinel/sentinel N128 rounds1 grouped256+pipelined jit",
    "sentinel/sentinel N240|N128 rounds1 grouped128+pipelined jit",
)
PULL_PATH = (
    "pull/counter N64 rounds1 grouped256+pipelined jit walkers",
    "pull/counter N128 rounds1 grouped256+pipelined jit walkers",
    "pull/counter N240|N128 rounds1 grouped128+pipelined jit walkers",
)
PULL_WARPS_PATH = tuple(path.replace("walkers", "pull_warps4") for path in PULL_PATH)
PULL_POOL_PATH = tuple(f"{path} pool" for path in PULL_PATH)
PULL_N240_PATH = (
    "pull/counter N240|N128 rounds1 grouped256+pipelined jit walkers",
    "pull/counter N240|N128 rounds1 grouped256+pipelined jit walkers",
    "pull/counter N240|N128 rounds1 grouped128+pipelined jit walkers",
)
PULL_POOL_N240_PATH = tuple(f"{path} pool" for path in PULL_N240_PATH)
# Dispatch rounds need at most 32 local experts, so EP8 only.
ROUNDS_PATH = (None, None, "counter/counter N240|N128 rounds4 grouped128+pipelined jit")
# Kimi-K3 runs at EP8 only. Its top-16 reaches a rank with about 2.26 routes,
# which keeps the pull dispatch out: the Sentinel protocols up to 128 tokens per
# rank, the counter dispatch with the Sentinel combine at 256, then the counter
# protocols in one round (112 local experts) on token tiles that grow with the load.
KIMI_K3_PATHS = {
    1: (None, None, "sentinel/sentinel N64 rounds1 grouped4096 jit"),
    16: (None, None, "sentinel/sentinel N64 rounds1 grouped4096 jit"),
    256: (None, None, "counter/sentinel N64 rounds1 grouped4096"),
    1024: (None, None, "counter/counter N192|N128 rounds1 grouped256+pipelined jit"),
    4096: (None, None, "counter/counter N240|N128 rounds1 grouped256+pipelined jit"),
}
KIMI_K3_ROUTINGS = ("perfect", "random")

PATHS: dict[str, dict[str, tuple[Optional[str], ...]]] = {
    "test_layer_vs_oracle": {
        "sentinel_protocols": SENTINEL_PATH,
        "just_in_time_claims": JIT_PATH,
        "capacity_256": CAPACITY_256_PATH,
        "counter_dispatch": COUNTER_DISPATCH_PATH,
        "counter_protocols": COUNTER_PROTOCOLS_PATH,
        "grouped_schedule": GROUPED_PATH,
        "n240_tile": N240_PATH,
        "dispatch_rounds": ROUNDS_PATH,
        "sentinel_512": SENTINEL_512_PATH,
        "sentinel_1024": SENTINEL_1024_PATH,
        "pull_1024": PULL_PATH,
        "pull_warps_1024": PULL_WARPS_PATH,
        "pull_token_pool_1024": PULL_POOL_PATH,
        "pull_dedup_1024": PULL_PATH,
        "pull_walker_acquire_1024": PULL_PATH,
        "pull_expert_gate_1024": PULL_PATH,
    },
    "test_die_local_weights": {"sentinel": SENTINEL_PATH, "grouped": GROUPED_PATH},
    "test_reduce_topk_in_kernel": {
        "sentinel_dispatch": SENTINEL_DISPATCH_PATH,
        "counter_protocols": COUNTER_PROTOCOLS_PATH,
        "grouped": GROUPED_PATH,
        "pull": PULL_PATH,
        "pull_token_pool": PULL_POOL_PATH,
    },
    "test_token_pool_capacity": {
        "every_tile_gathers": PULL_POOL_N240_PATH,
        "tiles_past_capacity": PULL_POOL_N240_PATH,
        "launch_past_capacity": PULL_POOL_N240_PATH,
    },
    "test_dispatch_round_gate": {
        "round_gate": ROUNDS_PATH,
        "round0_gate_control": ROUNDS_PATH,
    },
    "test_layer_on_ep_subgroup": {"": (SENTINEL_PATH[0], None, None)},
    "test_shared_expert": {
        "just_in_time_claims": JIT_PATH,
        "sentinel_protocols": SENTINEL_PATH,
        "sentinel_bf16_rounding": SENTINEL_PATH,
        "capacity_256": CAPACITY_256_PATH,
        "counter_dispatch": COUNTER_DISPATCH_PATH,
        "n240_tile": N240_PATH,
        "grouped_shared_clamp": GROUPED_PATH,
        "dispatch_rounds": ROUNDS_PATH,
        "pull_1024": PULL_PATH,
        "pull_token_pool_capacity": PULL_POOL_N240_PATH,
        "pull_in_kernel_reduce": PULL_PATH,
        "sentinel_after_plan": SENTINEL_PATH,
        "pull_fc2_early": PULL_PATH,
        "pull_alternate_fit": PULL_N240_PATH,
        "pull_head_one_token_tile": PULL_PATH,
    },
    "test_routing_weights_after_fc2": {
        "sentinel_protocols": SENTINEL_PATH,
        "counter_dispatch": COUNTER_DISPATCH_PATH,
        "counter_protocols": COUNTER_PROTOCOLS_PATH,
        "grouped": GROUPED_PATH,
        "dispatch_rounds": ROUNDS_PATH,
        "pull_1024": PULL_PATH,
        "pull_token_pool": PULL_POOL_PATH,
        "pull_in_kernel_reduce": PULL_PATH,
        "sentinel_shared_bf16_rounding": SENTINEL_PATH,
        "pull_shared": PULL_PATH,
        "pull_shared_in_kernel_reduce": PULL_PATH,
    },
    "test_kimi_k3": {
        f"t{capacity}_{routing}": path
        for capacity, path in KIMI_K3_PATHS.items()
        for routing in KIMI_K3_ROUTINGS
    },
}


def _path(config) -> str:
    """The engine path of a resolved engine configuration, as ``PATHS`` spells it."""
    mode, hint = config.schedule_policy
    alternate = config.resolved_alternate_token_tile_n
    parts = [
        f"{config.dispatch_protocol}/{config.combine_protocol}",
        f"N{config.resolved_mma_tiler_mnk[1]}"
        + (f"|N{alternate}" if alternate else ""),
        f"rounds{config.dispatch_rounds}",
        f"{mode}{hint}" + ("+pipelined" if config.pipelined_groups else ""),
    ]
    if config.resolved_just_in_time_claims:
        parts.append("jit")
    if config.dispatch_protocol == "pull":
        parts.append(
            f"pull_warps{config.pull_warps}" if config.pull_warps else "walkers"
        )
    if config.resolved_fc1_token_pool:
        parts.append("pool")
    return " ".join(parts)


def _assert_path(request, config, world: int) -> None:
    """Assert the path ``PATHS`` records for this case at this EP size."""
    test = request.node.originalname
    case = request.node.callspec.id if hasattr(request.node, "callspec") else ""
    if world not in EP_SIZES:
        return
    expected = PATHS[test][case][EP_SIZES.index(world)]
    assert _path(config) == expected, (
        f"{test}[{case}] at EP{world} resolves to {_path(config)!r}, not {expected!r}"
    )


# ----------------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------------


@pytest.mark.arch_blackwell
def test_init_builds_engine(dist_not_initialized):
    """On SM100 / SM103 ``validate_init`` builds the engine, so its limits fail at init."""
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpConfigError,
        Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    blackwell = (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability() in BLACKWELL_CAPABILITIES
    )
    if not blackwell:
        pytest.skip("the NVLinkOneSided MegaMoE requires SM100 or SM103")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    bootstrap = BootstrapConfig(world_size=4, rank=1, auto_bootstrap=False)
    kernel = create_mega_kernel(
        Sm103_Mxfp8_Mxfp4_Bf16_NVLinkOneSidedCutedsl_MegaMoeConfig(
            intermediate_size=INTERMEDIATE, top_k=TOPK, activation_clamp=CLAMP
        )
    )
    kernel.bind_ep_bootstrap(bootstrap)
    for capacity in (SENTINEL_CAPACITY, COUNTER_DISPATCH_CAPACITY):
        kernel.validate_init(
            bootstrap,
            FleetParams(
                num_experts=EXPERTS,
                max_tokens_per_rank=capacity,
                token_hidden_size=HIDDEN,
            ),
        )
        # A multiple of 512 whose staging rows do not fit in shared memory.
        with pytest.raises(MoEEpConfigError, match="cannot build its engine"):
            kernel.validate_init(
                bootstrap,
                FleetParams(
                    num_experts=EXPERTS,
                    max_tokens_per_rank=capacity,
                    token_hidden_size=32768,
                ),
            )


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity, knobs",
    [
        (SENTINEL_CAPACITY, None),
        (JUST_IN_TIME_CAPACITY, None),
        (COUNTER_DISPATCH_CAPACITY, None),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_DISPATCH_KNOBS),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_PROTOCOLS_KNOBS),
        (GROUPED_CAPACITY, None),
        (COUNTER_DISPATCH_CAPACITY, N240_KNOBS),
        (GROUPED_CAPACITY, ROUNDS_KNOBS),
        (SENTINEL_FORCED_CAPACITY, SENTINEL_KNOBS),
        (GROUPED_CAPACITY, SENTINEL_KNOBS),
        (GROUPED_CAPACITY, PULL_KNOBS),
        (GROUPED_CAPACITY, PULL_WARP_KNOBS),
        (GROUPED_CAPACITY, PULL_POOL_KNOBS),
        (GROUPED_CAPACITY, PULL_DEDUP_KNOBS),
        (GROUPED_CAPACITY, PULL_WALKER_ACQUIRE_KNOBS),
        (GROUPED_CAPACITY, PULL_EXPERT_GATE_KNOBS),
    ],
    ids=[
        "sentinel_protocols",
        "just_in_time_claims",
        "capacity_256",
        "counter_dispatch",
        "counter_protocols",
        "grouped_schedule",
        "n240_tile",
        "dispatch_rounds",
        "sentinel_512",
        "sentinel_1024",
        "pull_1024",
        "pull_warps_1024",
        "pull_token_pool_1024",
        "pull_dedup_1024",
        "pull_walker_acquire_1024",
        "pull_expert_gate_1024",
    ],
)
def test_layer_vs_oracle(request, capacity, knobs):
    """Weight/input formats, token counts, idle launches, skew, round reuse, views and CUDA graphs."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        MoEEpConfigError,
        MoEEpTensors,
        MoEWeightPack,
        preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend import (
        nvlink_one_sided_tiling,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.block_scaled import (
        JUST_IN_TIME_CLAIMS_MAX_TOKENS,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.genphase import (
        FENCE_JUST_IN_TIME_MIN_TOKENS,
    )

    rank, world = _init_distributed()
    if knobs is ROUNDS_KNOBS and EXPERTS // world > 32:
        pytest.skip("dispatch rounds need at most 32 local experts")
    group = dist.group.WORLD
    weights = _make_weights(EXPERTS // world, seed=1009 * rank + capacity)
    seed = 104729 * (rank + 1) + capacity
    layers = []
    try:
        # Packed MXFP4 weights + MXFP8 activations; BF16 weights + BF16
        # activations quantized while staging; kernel-layout weights from the
        # public preprocessing helper.
        packed = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            knobs=knobs,
        )
        layers.append(packed)
        bf16 = _layer(
            capacity,
            MoEWeightPack(weights.w13_bf16, weights.w2_bf16),
            rank=rank,
            world=world,
            quantize_input=True,
            knobs=knobs,
        )
        layers.append(bf16)
        weights.w13_bf16 = weights.w2_bf16 = None
        workspace = packed._ensure_workspace()
        _assert_path(request, workspace.config, world)
        die_local = workspace.config.resolved_die_local_weights
        kernel_layout = preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights(
            _prequantized_pack(weights),
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            die_local_weights=die_local,
        )
        transformed = _layer(
            capacity,
            None,
            rank=rank,
            world=world,
            quantize_input=False,
            transformed=kernel_layout,
            knobs=knobs,
        )
        layers.append(transformed)
        same = all(
            torch.equal(a.view(torch.uint8), b.view(torch.uint8))
            and torch.equal(a.view(torch.uint8), c.view(torch.uint8))
            for legs in zip(
                packed._transformed, bf16._transformed, kernel_layout, strict=True
            )
            for a, b, c in zip(*legs, strict=True)
        )
        _assert_all_ranks(same, group, "BF16, packed and kernel-layout weights differ")
        assert all(
            leg[0]._mega_weight_layout == "blockmajor"
            and getattr(leg[0], "_mega_die_local", False) == die_local
            for leg in packed._transformed
        )
        assert bf16._ensure_workspace() is workspace
        assert transformed._ensure_workspace() is workspace
        expected = nvlink_one_sided_tiling(capacity, world, EXPERTS, TOPK) | (
            knobs or {}
        )
        assert all(getattr(workspace.config, k) == v for k, v in expected.items())
        # The capacity-16 case compiles just-in-time claims for the Sentinel
        # dispatch, the capacities from FENCE_JUST_IN_TIME_MIN_TOKENS, whose tiling
        # picks a Fence dispatch, for those dispatches (kept when a knob switches
        # the dispatch to another protocol).
        assert (
            JUST_IN_TIME_CAPACITY <= JUST_IN_TIME_CLAIMS_MAX_TOKENS < SENTINEL_CAPACITY
        )
        assert (
            COUNTER_DISPATCH_CAPACITY
            < FENCE_JUST_IN_TIME_MIN_TOKENS
            <= GROUPED_CAPACITY
        )
        just_in_time = (
            capacity == JUST_IN_TIME_CAPACITY
            or capacity >= FENCE_JUST_IN_TIME_MIN_TOKENS
        )
        assert workspace.config.resolved_just_in_time_claims == just_in_time
        assert workspace.kernel.just_in_time_claims == just_in_time
        assert ("_claimlead" in workspace.kernel.name()) == just_in_time

        # Every rank sees every token count once, at a different step than its
        # peers, so empty and full ranks meet in the same launch.
        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts)):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(count, seed + step, holes=step % 2 == 1)
            y = packed.forward(_tensors(batch, quantize_input=False))
            assert y.shape == (count, HIDDEN) and y.dtype == torch.bfloat16
            _check(y, batch, weights, capacity, group, label=f"count {count}")
            y_bf16 = bf16.forward(_tensors(batch, quantize_input=True))
            _assert_all_ranks(
                torch.equal(y_bf16, y), group, f"count {count}: BF16 staging differs"
            )
            if step == 0:
                y_transformed = transformed.forward(
                    _tensors(batch, quantize_input=False)
                )
                _assert_all_ranks(
                    torch.equal(y_transformed, y), group, "kernel-layout weights differ"
                )

        # Idle launches: no rank stages a token, then every rank stages tokens
        # whose routes are all masked, so every output row is zero.
        for i, count in enumerate((0, capacity)):
            batch = _make_batch(count, seed + 40 + i, unrouted=True)
            y = packed.forward(_tensors(batch, quantize_input=False))
            assert y.shape == (count, HIDDEN)
            _assert_all_ranks(
                not bool(y.any()), group, f"unrouted count {count}: nonzero output"
            )

        # Skewed routing: every route of every rank lands on EP rank 0's first
        # TOPK experts, so each of them receives every token of every rank and
        # the other ranks' experts receive none.
        for i, count in enumerate((capacity, counts[rank % len(counts)])):
            batch = _make_batch(count, seed + 50 + i, skewed=True)
            y = packed.forward(_tensors(batch, quantize_input=False))
            _check(y, batch, weights, capacity, group, label=f"skewed count {count}")

        # Zero-copy output: the view aliases the workspace's output plane and
        # equals the owned output of the same batch.
        batch = _make_batch(min(64, capacity), seed + 60)
        owned = packed.forward(_tensors(batch, quantize_input=False))
        view = packed.forward(
            _tensors(batch, quantize_input=False), return_workspace_view=True
        )
        _assert_all_ranks(
            view.data_ptr() == packed.output_buffer.data_ptr()
            and view.shape == owned.shape
            and torch.equal(view, owned),
            group,
            "workspace view differs from the owned output",
        )

        # Six launches without host synchronization, each with a distinct
        # batch. Landing sets alternate by round, so each set is reused with
        # different data; a landing row or combine slot that kept a previous
        # round's bytes would put that round's token into the output and fail
        # the oracle bound.
        batches = [
            _make_batch(counts[(i + rank) % len(counts)], seed + 100 + i, holes=i == 2)
            for i in range(6)
        ]
        outputs = [
            packed.forward(_tensors(batch, quantize_input=False)) for batch in batches
        ]
        for i, (y, batch) in enumerate(zip(outputs, batches, strict=True)):
            _check(y, batch, weights, capacity, group, label=f"round {i}")

        # 0xFF bytes in the input: a NaN activation in row 0 and a NaN scale in
        # row 1 of every rank. Staging rewrites them, so the Sentinel dispatch
        # still sees the rows arrive; the other rows stay exact.
        batch = _make_batch(capacity, seed + 200)
        batch.x_fp8.view(torch.uint8)[0, 5] = 0xFF
        batch.x_sf_codes[1, 3] = 0xFF
        y = packed.forward(_tensors(batch, quantize_input=False))
        staged = workspace.x[:capacity].view(torch.uint8)
        staged_sf = workspace.x_sf[:capacity, : HIDDEN // SF_VEC].view(torch.uint8)
        _assert_all_ranks(
            not bool((staged == 0xFF).any())
            and not bool((staged_sf == 0xFF).any())
            and int(staged[0, 5]) == 0x7F
            and int(staged_sf[1, 3]) == 0xFE,
            group,
            "staging left a 0xFF byte",
        )
        _check(y, batch, weights, capacity, group, skip_rows=2, label="0xFF inputs")

        # A workspace that never launched has no compiled kernel: capturing it
        # fails before staging writes anything.
        unwarmed = packed.create_workspace(capacity // 2)
        try:
            probe = _tensors(_make_batch(1, seed + 250), quantize_input=False)
            graph = torch.cuda.CUDAGraph()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with (
                pytest.raises(MoEEpConfigError, match="warmup"),
                torch.cuda.graph(graph, stream=stream),
            ):
                packed.forward(probe, workspace=unwarmed)
            torch.cuda.current_stream().wait_stream(stream)
            del graph
            assert unwarmed._backend_workspace.staged_tokens() is None
        finally:
            unwarmed.destroy()

        # CUDA graphs: one captured with a caller-owned output and one through
        # the workspace view, replayed in turn with new inputs and interleaved
        # with eager launches of another layer on the same workspace.
        count = min(64, capacity)
        static_batch = _make_batch(count, seed + 300)
        static = _tensors(static_batch, quantize_input=False)
        static_view = MoEEpTensors(
            hidden_states=static.hidden_states,
            topk_ids=static.topk_ids,
            topk_weights=static.topk_weights,
            scales=static.scales,
        )
        static.output = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device="cuda")
        packed.warmup(static)
        packed.warmup(static_view)
        owned_graph, view_graph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(owned_graph, stream=stream):
            packed.forward(static)
        with torch.cuda.graph(view_graph, stream=stream):
            captured_view = packed.forward(static_view, return_workspace_view=True)
        torch.cuda.current_stream().wait_stream(stream)
        replays, eager = [], []
        for i in range(6):
            batch = _make_batch(count, seed + 400 + i, holes=i == 3)
            static.hidden_states.copy_(batch.x_fp8)
            static.scales.copy_(batch.x_sf_codes)
            static.topk_ids.copy_(batch.topk_ids)
            static.topk_weights.copy_(batch.topk_weights)
            if i % 2:
                view_graph.replay()
                replays.append((batch, captured_view.clone()))
                other = _make_batch(counts[(i + rank) % len(counts)], seed + 500 + i)
                eager.append(
                    (other, bf16.forward(_tensors(other, quantize_input=True)))
                )
            else:
                owned_graph.replay()
                replays.append((batch, static.output.clone()))
        for i, (batch, y) in enumerate(replays):
            kind = "view" if i % 2 else "owned"
            _check(y, batch, weights, capacity, group, label=f"{kind} graph replay {i}")
        for i, (batch, y) in enumerate(eager):
            _check(
                y, batch, weights, capacity, group, label=f"eager between replays {i}"
            )
        del owned_graph, view_graph
    finally:
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
        dist.barrier()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity", [SENTINEL_CAPACITY, GROUPED_CAPACITY], ids=["sentinel", "grouped"]
)
def test_die_local_weights(request, capacity):
    """Die-local weights set through the backend config give the plain layer's outputs bit for bit."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided.frontend.die_local import (
        locality_domain_count,
    )

    rank, world = _init_distributed()
    if locality_domain_count() < 2:
        pytest.skip("die-local weights need a device with two locality domains")
    group = dist.group.WORLD
    local = EXPERTS // world
    weights = _make_weights(local, seed=1031 * rank + capacity)
    seed = 15485863 * (rank + 1) + capacity
    layers = []
    try:
        plain = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            die_local_weights=False,
        )
        layers.append(plain)
        die_local = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            die_local_weights=True,
        )
        layers.append(die_local)
        # Kernel-layout weights from the public helper: die-local, and plain ones
        # the launch converts on the first launch.
        helper = _layer(
            capacity,
            None,
            rank=rank,
            world=world,
            quantize_input=False,
            transformed=preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights(
                _prequantized_pack(weights),
                intermediate_size=INTERMEDIATE,
                hidden_size=HIDDEN,
                die_local_weights=True,
            ),
            die_local_weights=True,
        )
        layers.append(helper)
        converted = _layer(
            capacity,
            None,
            rank=rank,
            world=world,
            quantize_input=False,
            transformed=preprocess_sm103_mxfp4_nvlink_one_sided_mega_weights(
                _prequantized_pack(weights),
                intermediate_size=INTERMEDIATE,
                hidden_size=HIDDEN,
            ),
            die_local_weights=True,
        )
        layers.append(converted)

        workspace = die_local._ensure_workspace()
        _assert_path(request, workspace.config, world)
        assert workspace.config.resolved_die_local_weights
        assert "_dielocal" in workspace.kernel.name()
        assert not plain._ensure_workspace().config.resolved_die_local_weights
        assert "_dielocal" not in plain._ensure_workspace().kernel.name()
        assert helper._ensure_workspace() is workspace
        assert converted._ensure_workspace() is workspace
        for leg, plain_leg, helper_leg in zip(
            die_local._transformed, plain._transformed, helper._transformed, strict=True
        ):
            experts, k_elements, rows = plain_leg[0].shape
            assert leg[0]._mega_die_local and not hasattr(
                plain_leg[0], "_mega_die_local"
            )
            assert tuple(leg[0].shape) == (2 * experts, k_elements, rows // 2)
            assert all(
                torch.equal(a.view(torch.uint8), b.view(torch.uint8))
                for a, b in zip(leg, helper_leg, strict=True)
            )

        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts) + 2):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(
                count, seed + step, holes=step % 2 == 1, skewed=step == len(counts)
            )
            y = plain.forward(_tensors(batch, quantize_input=False))
            if step == 0:
                _check(y, batch, weights, capacity, group, label=f"count {count}")
            for name, layer in (
                ("die-local", die_local),
                ("helper die-local", helper),
                ("converted", converted),
            ):
                _assert_all_ranks(
                    torch.equal(
                        layer.forward(_tensors(batch, quantize_input=False)), y
                    ),
                    group,
                    f"count {count}: {name} weights differ from plain weights",
                )

        # A CUDA graph of the die-local layer, replayed with new inputs.
        count = min(64, capacity)
        static = _tensors(_make_batch(count, seed + 300), quantize_input=False)
        static.output = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device="cuda")
        die_local.warmup(static)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            die_local.forward(static)
        torch.cuda.current_stream().wait_stream(stream)
        for i in range(3):
            batch = _make_batch(count, seed + 400 + i, holes=i == 1)
            static.hidden_states.copy_(batch.x_fp8)
            static.scales.copy_(batch.x_sf_codes)
            static.topk_ids.copy_(batch.topk_ids)
            static.topk_weights.copy_(batch.topk_weights)
            graph.replay()
            y = plain.forward(_tensors(batch, quantize_input=False))
            _assert_all_ranks(
                torch.equal(static.output, y),
                group,
                f"graph replay {i}: die-local weights differ from plain weights",
            )
        del graph
    finally:
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
        dist.barrier()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity, knobs",
    [
        (SENTINEL_CAPACITY, COUNTER_COMBINE_KNOBS),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_PROTOCOLS_KNOBS),
        (GROUPED_CAPACITY, None),
        (GROUPED_CAPACITY, PULL_KNOBS),
        (GROUPED_CAPACITY, PULL_POOL_KNOBS),
    ],
    ids=[
        "sentinel_dispatch",
        "counter_protocols",
        "grouped",
        "pull",
        "pull_token_pool",
    ],
)
def test_reduce_topk_in_kernel(request, capacity, knobs):
    """The in-kernel top-k reduce against both oracles and the separate reduce."""
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    group = dist.group.WORLD
    weights = _make_weights(EXPERTS // world, seed=1061 * rank + capacity)
    seed = 15485863 * (rank + 1) + capacity
    layers = []
    try:
        separate = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            knobs=knobs,
        )
        layers.append(separate)
        inplace = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            knobs=knobs,
            reduce_topk_in_kernel=True,
        )
        layers.append(inplace)
        workspace = inplace._ensure_workspace()
        _assert_path(request, workspace.config, world)
        config = workspace.config
        assert config.combine_protocol == "counter"
        assert config.reduce_topk_in_kernel
        assert not separate._ensure_workspace().config.reduce_topk_in_kernel
        assert "inkernel_reduce" in workspace.kernel.name()
        # The token pool resolves as it does without the in-kernel reduce.
        assert (
            config.resolved_fc1_token_pool
            == separate._ensure_workspace().config.resolved_fc1_token_pool
        )
        # No combine slots: the symmetric workspace loses its (token, top-k) plane.
        assert (
            workspace.kernel.get_workspace_sizes()[1]
            < separate._ensure_workspace().kernel.get_workspace_sizes()[1]
        )

        # Seven launches with distinct token counts per rank, holes and one skewed
        # batch: every launch adds into an output the previous one filled.
        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts) + 2):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(
                count, seed + step, holes=step % 2 == 1, skewed=step == len(counts)
            )
            y = inplace.forward(_tensors(batch, quantize_input=False))
            label = f"in-kernel top-k reduce, count {count}"
            nonfinite, requantized_error, exact_error = _oracle_errors(
                y, batch, weights, capacity, group, label=label
            )
            assert nonfinite == 0, f"{label}: nonfinite output"
            assert requantized_error < REQUANTIZED_TOLERANCE, (
                f"{label}: relL2 {requantized_error:.3e} vs requantized numerics"
            )
            assert exact_error < EXACT_TOLERANCE, (
                f"{label}: relL2 {exact_error:.3e} vs exact FP32"
            )
            y_separate = separate.forward(_tensors(batch, quantize_input=False))
            error = torch.tensor(
                [_relative_l2(y, y_separate) if count else 0.0], device="cuda"
            )
            dist.all_reduce(error, op=dist.ReduceOp.MAX, group=group)
            if rank == 0:
                print(
                    f"[EP{world} capacity {capacity}] {label}: relL2 {float(error):.2e} vs the separate reduce"
                )
            assert float(error) < IN_KERNEL_REDUCE_VS_SEPARATE_TOLERANCE, (
                f"{label}: relL2 {float(error):.3e} vs the separate reduce"
            )
    finally:
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
        dist.barrier()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "pool_rows, fallback",
    [
        (1, "tile"),
        (TOKEN_POOL_PARTIAL_ROWS, "tile"),
        (TOKEN_POOL_PARTIAL_ROWS, "launch"),
    ],
    ids=["every_tile_gathers", "tiles_past_capacity", "launch_past_capacity"],
)
def test_token_pool_capacity(request, pool_rows, fallback):
    """A token pool with a capacity gives the outputs of one sized for any routing bit for bit, whatever falls back."""
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    group = dist.group.WORLD
    capacity = GROUPED_CAPACITY
    weights = _make_weights(EXPERTS // world, seed=1103 * rank + pool_rows)
    seed = 7919 * (rank + 1) + pool_rows
    layers = []
    try:
        unbounded = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            knobs=PULL_POOL_N240_KNOBS,
            fc1_token_pool_capacity=0,
        )
        layers.append(unbounded)
        bounded = _layer(
            capacity,
            _prequantized_pack(weights),
            rank=rank,
            world=world,
            quantize_input=False,
            knobs={**PULL_POOL_N240_KNOBS, "fc1_token_pool_fallback": fallback},
            fc1_token_pool_capacity=pool_rows,
        )
        layers.append(bounded)
        workspace = bounded._ensure_workspace()
        _assert_path(request, workspace.config, world)
        assert workspace.kernel.token_pool_capacity == pool_rows
        assert unbounded._ensure_workspace().kernel.token_pool_capacity == 0
        # The landing slots stay allocated beside a pool of the capacity's rows.
        assert workspace.kernel.token_comm.lands_rows
        assert not unbounded._ensure_workspace().kernel.token_comm.lands_rows
        assert workspace.kernel.token_comm.fc1_token_pool_rows == pool_rows

        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts) + 2):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(
                count, seed + step, holes=step % 2 == 1, skewed=step == len(counts)
            )
            y = bounded.forward(_tensors(batch, quantize_input=False))
            label = f"token pool of {pool_rows} rows ({fallback}), count {count}"
            _check(y, batch, weights, capacity, group, label=label)
            routes, launch_capacity = workspace.recorded_token_pool()
            # The launch's capacity: the pool's, or zero where a launch's routes
            # do not all fit under the per-launch fallback.
            fits = routes <= pool_rows
            assert launch_capacity == (
                pool_rows if fallback == "tile" or fits else 0
            ), f"{label}: recorded {routes} routes, capacity {launch_capacity}"
            y_unbounded = unbounded.forward(_tensors(batch, quantize_input=False))
            _assert_all_ranks(
                torch.equal(y.view(torch.uint8), y_unbounded.view(torch.uint8)),
                group,
                f"{label}: differs from the pool sized for any routing",
            )
    finally:
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
        dist.barrier()


@pytest.mark.gpu_8
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "round0_only", [False, True], ids=["round_gate", "round0_gate_control"]
)
def test_dispatch_round_gate(request, round0_only, monkeypatch):
    """With the later rounds delayed the round gate holds and a round-0 gate fails.

    Every pusher holds the rounds after the first back by 50 us. Each refine
    block waits for its own expert's round, so the output still matches the
    oracle. The negative control gates every block on round 0 only: the later
    rounds' experts then read rows before they land, and at least one launch
    has to miss the oracle bound, or this test could not see a broken gate.
    """
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4_nvlink_one_sided import frontend

    rank, world = _init_distributed()
    if EXPERTS // world > 32:
        pytest.skip("dispatch rounds need at most 32 local experts")
    knobs = dict(ROUND_DELAY_KNOBS)
    if round0_only:
        knobs["refine_gate_round0_only"] = True
    # The fault-injection fields are engine diagnostics, not backend knobs; the
    # backend accepts them only while this test admits them.
    monkeypatch.setattr(
        frontend, "TUNING_FIELDS", frontend.TUNING_FIELDS | frozenset(knobs)
    )
    group = dist.group.WORLD
    capacity = GROUPED_CAPACITY
    weights = _make_weights(EXPERTS // world, seed=1013 * rank + 7)
    seed = 7841 * (rank + 1)
    layer = _layer(
        capacity,
        _prequantized_pack(weights),
        rank=rank,
        world=world,
        quantize_input=False,
        knobs=knobs,
    )
    try:
        config = layer._ensure_workspace().config
        _assert_path(request, config, world)
        assert config.dispatch_rounds == ROUNDS_KNOBS["dispatch_rounds"]
        assert config.round_publish_delay_ns == knobs["round_publish_delay_ns"]
        assert config.refine_gate_round0_only == round0_only
        # Full batches launched back to back, each with its own tokens, so a row
        # read before it lands holds another launch's token or none.
        batches = [_make_batch(capacity, seed + i) for i in range(3)]
        outputs = [
            layer.forward(_tensors(batch, quantize_input=False)) for batch in batches
        ]
        gate = "round-0 gate" if round0_only else "round gate"
        within = [
            _within_bounds(
                _oracle_errors(
                    y, batch, weights, capacity, group, label=f"{gate}, launch {i}"
                )
            )
            for i, (y, batch) in enumerate(zip(outputs, batches, strict=True))
        ]
        if round0_only:
            assert not all(within), (
                "every launch gated on round 0 only matched the oracle, so the "
                "delay does not expose a broken round gate"
            )
        else:
            assert all(within), f"round gate under the delay: within bounds {within}"
    finally:
        torch.cuda.synchronize()
        layer.destroy()
        dist.barrier()


@pytest.mark.gpu_4
@pytest.mark.arch_blackwell
def test_layer_on_ep_subgroup(request):
    """EP2 on each half of a 4-rank world: symmetric memory follows the EP group."""
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    if world != 4:
        pytest.skip("splits a 4-rank world into two EP2 groups")
    groups = [dist.new_group([0, 1]), dist.new_group([2, 3])]
    group = groups[rank // 2]
    ep_rank, ep_world = dist.get_rank(group), dist.get_world_size(group)
    capacity = 64
    weights = _make_weights(EXPERTS // ep_world, seed=7919 * rank)
    layer = _layer(
        capacity,
        _prequantized_pack(weights),
        rank=ep_rank,
        world=ep_world,
        quantize_input=False,
        process_group=group,
    )
    try:
        assert layer._ensure_workspace().process_group is group
        _assert_path(request, layer._ensure_workspace().config, ep_world)
        counts = (0, 7, capacity, 1)
        for step in range(len(counts)):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(count, 31 * rank + step)
            y = layer.forward(_tensors(batch, quantize_input=False))
            _check(y, batch, weights, capacity, group, label=f"subgroup count {count}")
    finally:
        torch.cuda.synchronize()
        layer.destroy()
        dist.barrier()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity, knobs, fields",
    [
        (JUST_IN_TIME_CAPACITY, None, {}),
        (SENTINEL_CAPACITY, None, {}),
        (SENTINEL_CAPACITY, None, {"shared_bf16_rounding": True}),
        (COUNTER_DISPATCH_CAPACITY, None, {}),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_DISPATCH_KNOBS, {}),
        (COUNTER_DISPATCH_CAPACITY, N240_KNOBS, {}),
        (GROUPED_CAPACITY, None, {"shared_gate_up_clamp": CLAMP}),
        (GROUPED_CAPACITY, ROUNDS_KNOBS, {}),
        (GROUPED_CAPACITY, PULL_KNOBS, {}),
        (
            GROUPED_CAPACITY,
            PULL_POOL_N240_KNOBS,
            {"fc1_token_pool_capacity": TOKEN_POOL_PARTIAL_ROWS},
        ),
        (GROUPED_CAPACITY, PULL_KNOBS, {"reduce_topk_in_kernel": True}),
        (
            SENTINEL_CAPACITY,
            {"shared_expert_entry": False, "shared_fc2_placement": "auto"},
            {},
        ),
        (GROUPED_CAPACITY, {**PULL_KNOBS, "shared_fc2_placement": "early"}, {}),
        (
            GROUPED_CAPACITY,
            {**PULL_KNOBS, **N240_KNOBS, "shared_alternate_token_tile": "always"},
            {},
        ),
        (GROUPED_CAPACITY, {**PULL_KNOBS, "shared_head_token_tiles": 1}, {}),
    ],
    ids=[
        "just_in_time_claims",
        "sentinel_protocols",
        "sentinel_bf16_rounding",
        "capacity_256",
        "counter_dispatch",
        "n240_tile",
        "grouped_shared_clamp",
        "dispatch_rounds",
        "pull_1024",
        "pull_token_pool_capacity",
        "pull_in_kernel_reduce",
        "sentinel_after_plan",
        "pull_fc2_early",
        "pull_alternate_fit",
        "pull_head_one_token_tile",
    ],
)
def test_shared_expert(request, capacity, knobs, fields):
    """The fused shared expert against both oracles, alone, across reruns and CUDA graph replays."""
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    if knobs is ROUNDS_KNOBS and EXPERTS // world > 32:
        pytest.skip("dispatch rounds need at most 32 local experts")
    group = dist.group.WORLD
    weights = _make_weights(EXPERTS // world, seed=1117 * rank + capacity)
    shared = _make_shared_expert(seed=4049 + capacity)
    shared.clamp = fields.get("shared_gate_up_clamp")
    shared.bf16_rounding = bool(fields.get("shared_bf16_rounding", False))
    weights.w13_bf16 = weights.w2_bf16 = None
    seed = 2017 * (rank + 1) + capacity
    layer = _layer(
        capacity,
        _shared_pack(weights, shared),
        rank=rank,
        world=world,
        quantize_input=False,
        knobs=knobs,
        shared_expert="fp8_in_kernel",
        **fields,
    )
    try:
        workspace = layer._ensure_workspace()
        _assert_path(request, workspace.config, world)
        assert workspace.config.shared_expert == "fp8_in_kernel"
        assert "_sharedfp8_in_kernel" in workspace.kernel.name()
        # The shared expert's slot: one more combine slot per token unless the
        # in-kernel reduce adds its partials into the output.
        assert workspace.kernel.combine_slots_per_token == TOPK + 1
        # Its FC1 tiles are claimed from kernel entry unless a knob keeps them behind the plan.
        tiles = workspace.kernel.shared_expert_tiles
        assert tiles.entry_head == (knobs or {}).get("shared_expert_entry", True)
        if (knobs or {}).get("shared_fc2_placement", "tail") != "auto":
            assert tiles.fc2_early == (
                (knobs or {}).get("shared_fc2_placement") == "early"
            )
        if fields.get("reduce_topk_in_kernel"):
            assert workspace.config.reduce_topk_in_kernel
        # Shared token tiles that fit the N128 alternate keep it (the launch takes
        # it for the small batches below and the plan's tile for the full one); a
        # capped head leaves the other token tiles' FC1 tiles to the claim streams.
        if (knobs or {}).get("shared_alternate_token_tile") == "always":
            assert workspace.kernel.alternate_tile_n == 128 and tiles.tile_rows <= 128
        if (knobs or {}).get("shared_head_token_tiles") == 1:
            assert (
                tiles.head_token_tiles == 1
                and tiles.streamed_token_tiles == tiles.token_tiles - 1
            )

        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts) + 1):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(
                count, seed + step, holes=step % 2 == 1, skewed=step == len(counts)
            )
            y = layer.forward(_tensors(batch, quantize_input=False))
            assert y.shape == (count, HIDDEN) and y.dtype == torch.bfloat16
            _check(
                y,
                batch,
                weights,
                capacity,
                group,
                label=f"count {count}",
                shared=shared,
            )
            again = layer.forward(_tensors(batch, quantize_input=False))
            if not fields.get("reduce_topk_in_kernel"):
                _assert_all_ranks(
                    torch.equal(again.view(torch.int16), y.view(torch.int16)),
                    group,
                    f"count {count}: a rerun differs",
                )

        # Every route masked: the output is the shared expert's alone.
        batch = _make_batch(capacity, seed + 40, unrouted=True)
        y = layer.forward(_tensors(batch, quantize_input=False))
        _check(
            y,
            batch,
            weights,
            capacity,
            group,
            label="shared expert alone",
            shared=shared,
        )

        # A CUDA graph replayed with new inputs, interleaved with eager launches.
        count = min(64, capacity)
        static = _tensors(_make_batch(count, seed + 300), quantize_input=False)
        static.output = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.warmup(static)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            layer.forward(static)
        torch.cuda.current_stream().wait_stream(stream)
        for i in range(3):
            batch = _make_batch(count, seed + 400 + i, holes=i == 1)
            static.hidden_states.copy_(batch.x_fp8)
            static.scales.copy_(batch.x_sf_codes)
            static.topk_ids.copy_(batch.topk_ids)
            static.topk_weights.copy_(batch.topk_weights)
            graph.replay()
            replayed = static.output.clone()
            _check(
                replayed,
                batch,
                weights,
                capacity,
                group,
                label=f"graph replay {i}",
                shared=shared,
            )
            eager = layer.forward(_tensors(batch, quantize_input=False))
            if not fields.get("reduce_topk_in_kernel"):
                _assert_all_ranks(
                    torch.equal(eager.view(torch.int16), replayed.view(torch.int16)),
                    group,
                    f"graph replay {i} differs from the eager launch",
                )
        del graph
    finally:
        torch.cuda.synchronize()
        layer.destroy()
        dist.barrier()


@pytest.mark.gpu_2
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity, knobs, fields",
    [
        (SENTINEL_CAPACITY, None, {}),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_DISPATCH_KNOBS, {}),
        (COUNTER_DISPATCH_CAPACITY, COUNTER_PROTOCOLS_KNOBS, {}),
        (GROUPED_CAPACITY, None, {}),
        (GROUPED_CAPACITY, ROUNDS_KNOBS, {}),
        (GROUPED_CAPACITY, PULL_KNOBS, {}),
        (GROUPED_CAPACITY, PULL_POOL_KNOBS, {}),
        (GROUPED_CAPACITY, PULL_KNOBS, {"reduce_topk_in_kernel": True}),
        (
            SENTINEL_CAPACITY,
            None,
            {"shared_expert": "fp8_in_kernel", "shared_bf16_rounding": True},
        ),
        (GROUPED_CAPACITY, PULL_KNOBS, {"shared_expert": "fp8_in_kernel"}),
        (
            GROUPED_CAPACITY,
            PULL_KNOBS,
            {"shared_expert": "fp8_in_kernel", "reduce_topk_in_kernel": True},
        ),
    ],
    ids=[
        "sentinel_protocols",
        "counter_dispatch",
        "counter_protocols",
        "grouped",
        "dispatch_rounds",
        "pull_1024",
        "pull_token_pool",
        "pull_in_kernel_reduce",
        "sentinel_shared_bf16_rounding",
        "pull_shared",
        "pull_shared_in_kernel_reduce",
    ],
)
def test_routing_weights_after_fc2(request, capacity, knobs, fields):
    """Routing weights after FC2 against both oracles in that order, across reruns and CUDA graph replays."""
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    if knobs is ROUNDS_KNOBS and EXPERTS // world > 32:
        pytest.skip("dispatch rounds need at most 32 local experts")
    group = dist.group.WORLD
    inplace = bool(fields.get("reduce_topk_in_kernel"))
    weights = _make_weights(EXPERTS // world, seed=1153 * rank + capacity)
    shared = None
    if fields.get("shared_expert"):
        shared = _make_shared_expert(seed=4073 + capacity)
        shared.clamp = None
        shared.bf16_rounding = bool(fields.get("shared_bf16_rounding", False))
        weights.w13_bf16 = weights.w2_bf16 = None
        pack = _shared_pack(weights, shared)
    else:
        pack = _prequantized_pack(weights)
    seed = 3001 * (rank + 1) + capacity
    layer = _layer(
        capacity,
        pack,
        rank=rank,
        world=world,
        quantize_input=False,
        knobs=knobs,
        routing_weights_after_fc2=True,
        **fields,
    )
    try:
        workspace = layer._ensure_workspace()
        _assert_path(request, workspace.config, world)
        assert not workspace.config.apply_routing_weights_before_fc2
        assert "apply_topk_fc2" in workspace.kernel.name()
        assert workspace.kernel._topk_reduce.weight_routes == (not inplace)

        def check(y, batch, label):
            _check(
                y,
                batch,
                weights,
                capacity,
                group,
                label=label,
                shared=shared,
                weight_after_fc2=True,
            )
            # The default order's oracle, which weights the SwiGLU output before
            # its requantization, is missed by more.
            _, after, _ = _oracle_errors(
                y,
                batch,
                weights,
                capacity,
                group,
                label=f"{label} (weights after FC2)",
                shared=shared,
                weight_after_fc2=True,
            )
            _, before, _ = _oracle_errors(
                y,
                batch,
                weights,
                capacity,
                group,
                label=f"{label} (weights before the requantization)",
                shared=shared,
            )
            if before > 0:
                assert after < before, f"{label}: closer to the default order's oracle"

        counts = (0, 1, 7, min(64, capacity), capacity)
        for step in range(len(counts) + 1):
            count = counts[(step + rank) % len(counts)]
            batch = _make_batch(
                count, seed + step, holes=step % 2 == 1, skewed=step == len(counts)
            )
            y = layer.forward(_tensors(batch, quantize_input=False))
            assert y.shape == (count, HIDDEN) and y.dtype == torch.bfloat16
            check(y, batch, f"count {count}")
            again = layer.forward(_tensors(batch, quantize_input=False))
            if not inplace:
                _assert_all_ranks(
                    torch.equal(again.view(torch.int16), y.view(torch.int16)),
                    group,
                    f"count {count}: a rerun differs",
                )

        # A CUDA graph replayed with new inputs, interleaved with eager launches.
        count = min(64, capacity)
        static = _tensors(_make_batch(count, seed + 300), quantize_input=False)
        static.output = torch.empty(count, HIDDEN, dtype=torch.bfloat16, device="cuda")
        layer.warmup(static)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            layer.forward(static)
        torch.cuda.current_stream().wait_stream(stream)
        for i in range(3):
            batch = _make_batch(count, seed + 400 + i, holes=i == 1)
            static.hidden_states.copy_(batch.x_fp8)
            static.scales.copy_(batch.x_sf_codes)
            static.topk_ids.copy_(batch.topk_ids)
            static.topk_weights.copy_(batch.topk_weights)
            graph.replay()
            replayed = static.output.clone()
            check(replayed, batch, f"graph replay {i}")
            eager = layer.forward(_tensors(batch, quantize_input=False))
            if not inplace:
                _assert_all_ranks(
                    torch.equal(eager.view(torch.int16), replayed.view(torch.int16)),
                    group,
                    f"graph replay {i} differs from the eager launch",
                )
        del graph
    finally:
        torch.cuda.synchronize()
        layer.destroy()
        dist.barrier()


@pytest.mark.gpu_8
@pytest.mark.arch_blackwell
@pytest.mark.parametrize(
    "capacity, routing",
    [(capacity, routing) for capacity in KIMI_K3_PATHS for routing in KIMI_K3_ROUTINGS],
    ids=[
        f"t{capacity}_{routing}"
        for capacity in KIMI_K3_PATHS
        for routing in KIMI_K3_ROUTINGS
    ],
)
def test_kimi_k3(request, capacity, routing):
    """Kimi-K3 at EP8 matches both oracles and reproduces its output bit for bit.

    Each capacity runs its own engine path (``KIMI_K3_PATHS``) on a full batch,
    then on counts that differ by rank (rank 0 empty), under perfect or random
    routing. The full batch then runs again on both landing sets and from a
    CUDA graph, each giving the first output's bits.
    """
    import torch
    import torch.distributed as dist

    rank, world = _init_distributed()
    if world != 8:
        pytest.skip("Kimi-K3 runs at EP8")
    problem = KIMI_K3
    group = dist.group.WORLD
    weights = _make_weights(
        problem.experts // world, seed=1171 * rank + capacity, problem=problem
    )
    seed = 7877 * (rank + 1) + capacity
    perfect = (rank, world) if routing == "perfect" else None
    layer = _layer(
        capacity,
        _prequantized_pack(weights),
        rank=rank,
        world=world,
        quantize_input=False,
        problem=problem,
    )
    try:
        _assert_path(request, layer._ensure_workspace().config, world)

        def check(y, batch, label):
            assert y.shape == (batch.count, problem.hidden)
            _check(y, batch, weights, capacity, group, label=label, problem=problem)

        batch = _make_batch(capacity, seed, problem=problem, perfect=perfect)
        y = layer.forward(_tensors(batch, quantize_input=False))
        check(y, batch, f"{routing} routing, full batch")
        partial = _make_batch(
            capacity * rank // world, seed + 1, problem=problem, perfect=perfect
        )
        check(
            layer.forward(_tensors(partial, quantize_input=False)),
            partial,
            f"{routing} routing, counts by rank",
        )
        reruns = [
            layer.forward(_tensors(batch, quantize_input=False)) for _ in range(2)
        ]

        static = _tensors(batch, quantize_input=False)
        static.output = torch.empty_like(y)
        layer.warmup(static)
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.graph(graph, stream=stream):
            layer.forward(static)
        torch.cuda.current_stream().wait_stream(stream)
        static.output.zero_()
        graph.replay()
        reruns.append(static.output.clone())
        del graph
        _assert_all_ranks(
            all(
                torch.equal(again.view(torch.int16), y.view(torch.int16))
                for again in reruns
            ),
            group,
            "a rerun of the full batch differs from its first output",
        )
    finally:
        torch.cuda.synchronize()
        layer.destroy()
        dist.barrier()
