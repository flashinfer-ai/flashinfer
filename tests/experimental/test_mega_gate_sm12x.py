# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0.
# https://www.apache.org/licenses/LICENSE-2.0

"""SM120 CuTe DSL Mega Gate against an FP64 reference.

Covers both DeepSeek-V4-Flash and V4.1-Flash router shapes over many token
counts, other K/E/top-k, shared slots, physical mapping, masks, image bias, the
no-bias ranking, NaN scores, fixed routing, the SM cap, deterministic mode
(against the reference and batch invariant), bitwise repeatability (the arrival
counters rearm themselves), CUDA graph replay with changed inputs on a side
stream, PDL behind a producer that triggers before it writes the tokens, input
validation, and the schedule invariants (CPU-only).
"""

import functools
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.nn.functional as F

from flashinfer.experimental.sm12x_mega_gate import heuristics
from flashinfer.mega_gate import prepare_mega_gate

DUPLICATE_HASH = 23333
SCALE = 1.5
PRODUCTION = dict(hidden=5120, num_experts=384, num_topk=6)
V4_FLASH = dict(hidden=4096, num_experts=256, num_topk=6)
"""DeepSeek-V4-Flash's router: two 128-expert groups per token block."""
TOKEN_COUNTS = (
    1,
    2,
    3,
    7,
    16,
    17,
    33,
    64,
    65,
    100,
    128,
    200,
    256,
    400,
    512,
    777,
    1024,
    1025,
    3000,
    4096,
    8192,
)


def _skip_unless_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device_index = torch.cuda.current_device()
    capability = torch.cuda.get_device_capability(device_index)
    if capability != (12, 0):
        pytest.skip("SM120 device required")
    from flashinfer.cute_dsl.availability import is_cute_dsl_arch_supported

    if not is_cute_dsl_arch_supported(*capability, native_only=True):
        pytest.skip("CuTe DSL native SM120 support required")
    from flashinfer.experimental.sm12x_mega_gate import mega_gate as runtime

    runtime.device_facts(device_index)


# ----------------------------------------------------------------- CPU only


@pytest.mark.cpu_only
@pytest.mark.parametrize("num_sms", [48, 110, 170, 188])
@pytest.mark.parametrize("deterministic", [False, True])
def test_schedule_invariants(num_sms, deterministic):
    for num_experts in (96, 128, 256, 260, 384, 512, 1024):
        for hidden in (2048, 4096, 5120, 7168):
            for num_tokens in (
                *range(1, 130),
                255,
                256,
                257,
                1000,
                1024,
                1025,
                4096,
                8192,
                1 << 16,
            ):
                config = heuristics.select_config(
                    num_tokens, hidden, num_experts, num_sms, deterministic
                )
                k_blocks = hidden // heuristics.BLOCK_K
                assert config.block_tokens in heuristics.BLOCK_TOKENS_CHOICES
                assert k_blocks % config.num_split_k == 0
                assert config.num_split_k <= heuristics.MAX_SPLIT_K
                assert config.num_stages >= 2
                assert config.num_gate_warps in heuristics.GATE_WARP_CHOICES
                assert config.mma_warps_n == heuristics.select_mma_warps_n(
                    config.block_tokens
                )
                if config.mma_warps_n > 1:  # 17 warps would spill the mainloop
                    assert config.num_gate_warps == 4
                smem = (
                    config.num_stages * heuristics.stage_bytes(config.block_tokens)
                    + heuristics.SMEM_RESERVED_BYTES
                )
                if config.cooperative_gate:
                    slots = heuristics.cooperative_slots(
                        config.block_tokens, config.num_logical_ctas
                    )
                    assert slots <= heuristics.MAX_COOPERATIVE_TOKENS
                    assert config.num_stages >= heuristics.MIN_COOPERATIVE_STAGES
                    smem += slots * heuristics.exchange_bytes(num_experts)
                assert smem <= heuristics.SMEM_CAPACITY_BYTES
                assert (
                    config.num_expert_groups * heuristics.EXPERTS_PER_CTA >= num_experts
                )
                if deterministic:
                    assert config.num_split_k == 1
                # Persistent and co-resident: every worker group fits, one CTA per SM.
                assert config.num_logical_ctas <= num_sms
                assert config.num_ctas(num_tokens, num_sms) <= num_sms
                workers = config.num_worker_groups(num_tokens, num_sms)
                assert 1 <= workers <= config.num_token_blocks(num_tokens)


@pytest.mark.cpu_only
def test_measured_splits_take_the_smallest_one_wave_block():
    for (num_sms, hidden, num_experts), ranges in heuristics.MEASURED_SPLITS.items():
        num_groups = -(-num_experts // heuristics.EXPERTS_PER_CTA)
        bound = 0
        for max_tokens, split in ranges:
            assert max_tokens > bound
            for num_tokens in range(bound + 1, max_tokens + 1):
                config = heuristics.select_config(
                    num_tokens, hidden, num_experts, num_sms
                )
                assert config.num_split_k == split
                worker_groups = num_sms // (num_groups * split)
                blocks = config.num_token_blocks(num_tokens)
                assert blocks <= worker_groups  # one wave
                smaller = [
                    b
                    for b in heuristics.BLOCK_TOKENS_CHOICES
                    if b < config.block_tokens
                ]
                assert all(-(-num_tokens // b) > worker_groups for b in smaller)
                deterministic = heuristics.select_config(
                    num_tokens, hidden, num_experts, num_sms, deterministic=True
                )
                assert deterministic.num_split_k == 1
            bound = max_tokens


@pytest.mark.cpu_only
def test_num_sms_is_sm12x_only():
    x = torch.zeros((1, 64), dtype=torch.bfloat16)
    weight = torch.zeros((8, 64), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="SM12x"):
        prepare_mega_gate(x, weight, 2, num_sms=16)


# --------------------------------------------------------------- reference


def _physical_map(num_logical, shared_offset, replicas=2):
    """``replicas`` distinct replica ids per logical expert. Two replicas: every third
    expert uses one; more: the counts cycle through 1..replicas."""
    mapping = torch.stack(
        [
            torch.arange(num_logical, dtype=torch.int32, device="cuda")
            + replica * shared_offset
            for replica in range(replicas)
        ],
        1,
    ).contiguous()
    if replicas == 2:
        counts = torch.full((num_logical,), 2, dtype=torch.int32, device="cuda")
        counts[::3] = 1  # some experts have a single replica
    else:
        counts = (
            torch.arange(num_logical, dtype=torch.int32, device="cuda") % replicas + 1
        )
    return mapping, counts


def _make_inputs(num_tokens, hidden, num_experts, *, seed=0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(
        (num_tokens, hidden), dtype=torch.bfloat16, device="cuda", generator=generator
    )
    weight = (
        torch.randn(
            (num_experts, hidden),
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * hidden**-0.5
    ).to(torch.bfloat16)
    bias = (
        torch.randn(
            (num_experts,), dtype=torch.float32, device="cuda", generator=generator
        )
        * 0.1
    )
    return x, weight, bias


def _reference_scores(x, weight, bias, image_bias=None, image_token_mask=None):
    logits = x.double() @ weight.double().T
    scores = F.softplus(logits).sqrt()
    if bias is None and image_bias is None:
        return scores, logits
    ranking_bias = (
        bias.double()
        if bias is not None
        else torch.zeros(weight.shape[0], dtype=torch.float64, device="cuda")
    )
    if image_bias is not None:
        ranking_bias = torch.where(
            image_token_mask[:, None], image_bias.double(), ranking_bias
        )
    return scores, scores + ranking_bias


def _check_routing(
    indices,
    weights,
    logical,
    scores,
    ranking,
    *,
    num_topk,
    num_shared,
    num_experts,
    mask=None,
    to_physical_map=None,
    logical_count=None,
    ep_rank=0,
    fixed=None,
    given=None,
):
    num_tokens = indices.shape[0]
    active = (
        mask
        if mask is not None
        else torch.ones(num_tokens, dtype=torch.bool, device="cuda")
    )
    if (~active).any():
        assert torch.all(indices[~active] == -1)
        assert torch.all(weights[~active] == 0)
        assert torch.all(logical[~active] == -1)
    if not active.any():
        return
    selected = logical[active]
    assert torch.all((selected >= 0) & (selected < num_experts))
    # Fixed-routing rows keep their given experts (duplicates allowed); the rest are ranked.
    ranked = active if fixed is None else active & ~fixed
    if fixed is not None:
        assert torch.equal(logical[active & fixed], given[active & fixed])
    ordered = logical[ranked].sort(dim=1).values
    assert torch.all(ordered[:, 1:] != ordered[:, :-1]), "duplicate experts in a row"

    # Top-k up to numerical ties of the FP32 kernel against the FP64 reference.
    rows = ranking[ranked]
    chosen = rows.gather(1, logical[ranked])
    rest = rows.scatter(1, logical[ranked], float("-inf")).max(dim=1).values
    tolerance = 1e-5 + 1e-5 * rows.abs().amax(dim=1)
    crossing = rest - chosen.min(dim=1).values
    assert torch.all(crossing <= tolerance), (
        f"top-k beyond ties: {crossing.max().item()}"
    )

    assert torch.all(chosen[:, 1:] - chosen[:, :-1] <= tolerance[:, None]), (
        "selected experts must be in descending routing-rank order"
    )

    # Weights of the kernel's own selection, from the reference scores.
    picked = scores[active].gather(1, selected)
    expected = picked / (picked.sum(dim=1, keepdim=True) + 1e-20) * SCALE
    torch.testing.assert_close(
        weights[active, :num_topk].double(), expected, rtol=1e-4, atol=1e-6
    )
    if num_shared:
        assert torch.all(weights[active, num_topk:] == 1.0)

    # Logical slots (routed then shared) and their physical replicas.
    shared_slots = (
        torch.arange(num_shared, device="cuda", dtype=torch.int64) + num_experts
    ).expand(num_tokens, num_shared)
    logical_slots = torch.cat((logical, shared_slots), dim=1)[active]
    if to_physical_map is None:
        assert torch.equal(indices[active], logical_slots)
    else:
        tokens = torch.arange(num_tokens, device="cuda", dtype=torch.int64)[active][
            :, None
        ]
        hashed = (ep_rank + tokens * DUPLICATE_HASH) % (1 << 32)
        replica = hashed % logical_count.long()[logical_slots]
        expected_physical = to_physical_map.long()[logical_slots, replica]
        assert torch.equal(indices[active], expected_physical)


def _run_case(
    num_tokens,
    hidden,
    num_experts,
    num_topk,
    *,
    num_shared=0,
    physical=True,
    with_bias=True,
    with_mask=False,
    with_image_bias=False,
    ep_rank=3,
    replicas=2,
    seed=0,
    config=None,
    num_sms=None,
    fixed_routing=None,
    deterministic=False,
):
    x, weight, bias = _make_inputs(num_tokens, hidden, num_experts, seed=seed)
    if not with_bias:
        bias = None
    num_logical = num_experts + num_shared
    mapping, counts = (
        _physical_map(num_logical, num_logical, replicas) if physical else (None, None)
    )
    masks = torch.Generator(device="cuda").manual_seed(seed + 2)
    mask = None
    if with_mask:
        mask = torch.rand(num_tokens, device="cuda", generator=masks) >= 0.25
        mask[0] = True
    image_bias = image_token_mask = None
    if with_image_bias:
        image_bias = torch.randn(
            num_experts, dtype=torch.float32, device="cuda", generator=masks
        )
        image_token_mask = torch.rand(num_tokens, device="cuda", generator=masks) < 0.5
    logical = torch.empty((num_tokens, num_topk), dtype=torch.int64, device="cuda")
    fixed = given = None
    if fixed_routing is not None:  # "some" (every third token) or "all"
        token_idx = torch.arange(num_tokens, device="cuda")
        fixed = token_idx % 3 == 0 if fixed_routing == "some" else token_idx >= 0
        given = torch.randint(
            0,
            num_experts,
            (num_tokens, num_topk),
            device="cuda",
            generator=torch.Generator(device="cuda").manual_seed(seed + 1),
        )
        logical.copy_(given)
    prepare = prepare_mega_gate
    if config is not None:  # a forced schedule, for paths the heuristic picks elsewhere
        from flashinfer.experimental.sm12x_mega_gate.mega_gate import MegaGateSm12xPlan

        prepare = functools.partial(MegaGateSm12xPlan, _config=config)
    plan = prepare(
        x,
        weight,
        num_topk,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        mask=mask,
        to_physical_map=mapping,
        logical_count=counts,
        unmapped_topk_idx=logical,
        use_shared_as_routed=num_shared > 0,
        num_shared_experts=max(num_shared, 1),
        routed_scaling_factor=SCALE,
        ep_rank=ep_rank,
        **({} if num_sms is None else {"num_sms": num_sms}),
        **({} if fixed is None else {"fix_routing_mask": fixed}),
        deterministic=deterministic,
    )
    indices, weights = plan.run()
    torch.cuda.synchronize()
    scores, ranking = _reference_scores(x, weight, bias, image_bias, image_token_mask)
    _check_routing(
        indices,
        weights,
        logical,
        scores,
        ranking,
        num_topk=num_topk,
        num_shared=num_shared,
        num_experts=num_experts,
        mask=mask,
        to_physical_map=mapping,
        logical_count=counts,
        ep_rank=ep_rank,
        fixed=fixed,
        given=given,
    )
    return plan


# -------------------------------------------------------------------- GPU


@pytest.mark.parametrize("num_tokens", [1, 37, 1000])
@pytest.mark.parametrize("ep_rank", [11, 2**31 - 7])
def test_replica_counts(num_tokens, ep_rank):
    """Replica counts 1..7, not only powers of two, and token hashes near 2^32: the
    reciprocal multiply of the replica selection and its correction step."""
    _skip_unless_sm120()
    _run_case(
        num_tokens, 5120, 384, 6, num_shared=1, ep_rank=ep_rank, replicas=7, seed=2
    )


@pytest.mark.parametrize(
    "block_tokens,num_split_k,num_tokens",
    [(16, 8, 300), (32, 8, 600), (64, 8, 900), (64, 4, 1000)],
)
@pytest.mark.parametrize(
    "shape,num_shared", [(PRODUCTION, 1), (V4_FLASH, 0)], ids=["E384", "E256"]
)
def test_cooperative_gate_over_several_blocks(
    block_tokens, num_split_k, num_tokens, shape, num_shared
):
    """Cooperative gates of 1-3 tokens per CTA (and a per-token gate) whose CTAs rank
    several blocks: the exchange is reused block after block."""
    _skip_unless_sm120()
    config = heuristics.make_config(block_tokens, num_split_k, shape["num_experts"])
    assert config.cooperative_gate == (block_tokens <= 3 * config.num_logical_ctas)
    # More blocks than worker groups on any SM count (a 188-SM board runs 23 groups of
    # 8 CTAs), and a partial last block.
    sms = torch.cuda.get_device_properties(torch.cuda.current_device())
    groups = sms.multi_processor_count // config.num_logical_ctas
    num_tokens = max(num_tokens, block_tokens * (groups + 2) + 5)
    plan = _run_case(
        num_tokens, **shape, num_shared=num_shared, with_mask=True, config=config
    )
    assert plan.route["num_worker_groups"] < config.num_token_blocks(num_tokens)


@pytest.mark.parametrize("num_tokens", TOKEN_COUNTS)
@pytest.mark.parametrize("physical", [True, False])
def test_production_shape(num_tokens, physical):
    _skip_unless_sm120()
    _run_case(num_tokens, **PRODUCTION, physical=physical)


@pytest.mark.parametrize(
    "num_tokens", [1, 6, 40, 120, 240, 280, 448, 576, 800, 960, 1120, 4096, 8192]
)
@pytest.mark.parametrize(
    "physical,with_mask", [(False, False), (True, True)], ids=["plain", "mapped-masked"]
)
def test_v4_flash_shape(num_tokens, physical, with_mask):
    """Decode and prefill token counts of DeepSeek-V4-Flash deployments (256 experts
    and top-6 admit no shared slot), with and without a physical map and a mask."""
    _skip_unless_sm120()
    _run_case(num_tokens, **V4_FLASH, physical=physical, with_mask=with_mask)


@pytest.mark.parametrize(
    "hidden,num_experts,num_topk,num_shared",
    [
        (4096, 256, 8, 0),
        (7168, 256, 8, 1),
        (2048, 128, 4, 0),
        (4096, 260, 6, 0),
        (4096, 96, 6, 2),
        (5120, 384, 6, 1),
        (5120, 512, 8, 2),
        (4096, 1024, 16, 0),
        (320, 128, 17, 0),
        (320, 124, 31, 1),
        (320, 128, 32, 0),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 5, 129, 2048])
def test_other_shapes(hidden, num_experts, num_topk, num_shared, num_tokens):
    _skip_unless_sm120()
    _run_case(num_tokens, hidden, num_experts, num_topk, num_shared=num_shared)


@pytest.mark.parametrize("num_tokens", [1, 128, 1000])
@pytest.mark.parametrize("num_sms", [12, 64])
@pytest.mark.parametrize("shape", [PRODUCTION, V4_FLASH], ids=["E384", "E256"])
def test_num_sms_caps_the_grid(num_tokens, num_sms, shape):
    """A capped grid (SMs left to concurrent kernels) stays correct and within the cap."""
    _skip_unless_sm120()
    plan = _run_case(num_tokens, **shape, num_sms=num_sms)
    assert plan.route["num_sms"] == num_sms
    assert plan.route["launch_ctas"] <= num_sms


@pytest.mark.parametrize("num_tokens", [1, 7, 129, 1000])
@pytest.mark.parametrize("fixed_routing", ["some", "all"])
@pytest.mark.parametrize(
    "shape,num_shared", [(PRODUCTION, 1), (V4_FLASH, 0)], ids=["E384", "E256"]
)
def test_fixed_routing(num_tokens, fixed_routing, shape, num_shared):
    """Hash-routed tokens (fix_routing_mask) keep the logical experts given in
    unmapped_topk_idx, duplicates included, mapped and weighted from their unbiased
    scores; unflagged tokens are ranked; masked tokens route nowhere."""
    _skip_unless_sm120()
    _run_case(
        num_tokens,
        **shape,
        num_shared=num_shared,
        with_mask=True,
        fixed_routing=fixed_routing,
    )


def test_fixed_routing_needs_given_experts():
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4, 5120, 384)
    fixed = torch.ones(4, dtype=torch.bool, device="cuda")
    with pytest.raises(ValueError, match="unmapped_topk_idx"):
        prepare_mega_gate(x, weight, 6, bias=bias, fix_routing_mask=fixed)


def test_num_sms_out_of_range():
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4, 5120, 384)
    for num_sms in (0, 1 << 10):
        with pytest.raises(ValueError, match="num_sms"):
            prepare_mega_gate(x, weight, 6, bias=bias, num_sms=num_sms)


@pytest.mark.parametrize("num_tokens", [1, 33, 1000])
def test_without_bias(num_tokens):
    _skip_unless_sm120()
    _run_case(num_tokens, **PRODUCTION, with_bias=False)


@pytest.mark.parametrize("num_tokens", [1, 7, 129, 3000])
def test_mask_and_image_bias(num_tokens):
    _skip_unless_sm120()
    _run_case(num_tokens, **PRODUCTION, with_mask=True, with_image_bias=True)
    _run_case(num_tokens, **PRODUCTION, with_mask=True, with_bias=False, physical=False)


def test_strided_logical_output():
    _skip_unless_sm120()
    num_tokens, num_topk = 40, 6
    x, weight, bias = _make_inputs(num_tokens, 5120, 384)
    storage = torch.full(
        (num_tokens, num_topk + 3), -7, dtype=torch.int64, device="cuda"
    )
    logical = storage[:, :num_topk]
    indices, weights = prepare_mega_gate(
        x,
        weight,
        num_topk,
        bias=bias,
        unmapped_topk_idx=logical,
        routed_scaling_factor=SCALE,
    ).run()
    torch.cuda.synchronize()
    assert torch.all(storage[:, num_topk:] == -7)
    assert torch.equal(logical, indices)


def test_zero_inputs_resolve_ties_to_the_smallest_experts():
    _skip_unless_sm120()
    num_topk = 8
    x = torch.zeros((8, 4096), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((128, 4096), dtype=torch.bfloat16, device="cuda")
    indices, weights = prepare_mega_gate(x, weight, num_topk).run()
    torch.cuda.synchronize()
    assert torch.equal(
        indices, torch.arange(num_topk, dtype=torch.int64, device="cuda").expand(8, -1)
    )
    torch.testing.assert_close(weights, torch.full_like(weights, 1.5 / num_topk))


def test_zero_tokens():
    _skip_unless_sm120()
    x = torch.empty((0, 5120), dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros((384, 5120), dtype=torch.bfloat16, device="cuda")
    indices, weights = prepare_mega_gate(x, weight, 6).run()
    assert indices.shape == (0, 6) and weights.shape == (0, 6)


@pytest.mark.parametrize("num_tokens", [1, 64, 2048])
def test_repeated_runs_are_bitwise_identical(num_tokens):
    """The split partials are combined in a fixed order and the counters rearm."""
    _skip_unless_sm120()
    plan = _run_case(num_tokens, **PRODUCTION)
    expected = [t.clone() for t in plan.outputs]
    for _ in range(5):
        for result, reference in zip(plan.run(), expected, strict=True):
            assert torch.equal(result, reference)
    assert torch.all(plan.score_barriers.view(torch.int64) == 0)


@pytest.mark.parametrize("num_tokens", [1, 512])
def test_pdl_waits_for_a_producer_that_triggers_early(num_tokens):
    """Under PDL the gate launches once the kernel before it triggers. This producer
    triggers first and writes the tokens only after a delay, so a gate that read them
    before ``griddepcontrol.wait`` would route the stale tokens."""
    _skip_unless_sm120()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    from cutlass import Float32
    from cutlass.cute.runtime import from_dlpack

    @cute.kernel
    def write_late(src: cute.Tensor, dst: cute.Tensor, sink: cute.Tensor):
        cute.arch.griddepcontrol_launch_dependents()
        thread = cute.arch.block_idx()[0] * 256 + cute.arch.thread_idx()[0]
        # ~0.2 ms of dependent FMAs: the gate is resident long before the tokens land.
        delay = Float32(thread)
        for _ in cutlass.range(100000, unroll=1):
            delay = delay * Float32(1.0000001) + Float32(1.0)
        if delay == Float32(-1.0):
            sink[0] = delay
        for i in cutlass.range(thread, src.shape[0], 8 * 256):
            dst[i] = src[i]

    @cute.jit
    def launch(src, dst, sink, stream: cuda.CUstream):
        write_late(src, dst, sink).launch(
            grid=(8, 1, 1), block=(256, 1, 1), stream=stream
        )

    x, weight, bias = _make_inputs(num_tokens, 5120, 384, seed=3)
    tokens = x.clone()
    expected = [
        t.clone()
        for t in prepare_mega_gate(tokens, weight, 6, bias=bias, enable_pdl=False).run()
    ]
    plan = prepare_mega_gate(x, weight, 6, bias=bias, enable_pdl=True)
    assert plan.route["enable_pdl"]
    sink = torch.zeros(1, device="cuda")
    args = [from_dlpack(t) for t in (tokens.view(-1), x.view(-1), sink)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    write_tokens = cute.compile(launch, *args, stream)
    stale = torch.Generator(device="cuda").manual_seed(5)
    for _ in range(3):
        x.copy_(torch.randn(x.shape, dtype=x.dtype, device="cuda", generator=stale))
        write_tokens(*args, stream)
        indices, weights = plan.run()
        torch.cuda.synchronize()
        assert torch.equal(indices, expected[0])
        assert torch.equal(weights, expected[1])


@pytest.mark.parametrize("with_bias", [True, False])
def test_nan_scores_rank_last(with_bias):
    """An expert whose logit is NaN is never chosen; a token whose logits are all NaN has
    no finite candidate and gets the clamped expert E-1 with weight 0 in every slot."""
    _skip_unless_sm120()
    num_tokens, num_experts = 64, 384
    x, weight, bias = _make_inputs(num_tokens, 5120, num_experts, seed=5)
    if not with_bias:
        bias = None
    weight[7, 0] = float("nan")  # expert 7: a NaN logit for every token
    x[3, 0] = float("nan")  # token 3: every logit NaN
    logical = torch.empty((num_tokens, 6), dtype=torch.int64, device="cuda")
    plan = prepare_mega_gate(
        x,
        weight,
        6,
        bias=bias,
        unmapped_topk_idx=logical,
        routed_scaling_factor=SCALE,
    )
    indices, weights = plan.run()
    torch.cuda.synchronize()
    assert torch.all(logical[3] == num_experts - 1)
    assert torch.all(indices[3] == num_experts - 1)
    assert torch.all(weights[3] == 0)
    rows = torch.arange(num_tokens, device="cuda") != 3
    assert not torch.any(logical[rows] == 7)
    scores, ranking = _reference_scores(x[rows], weight, bias)
    ranking[:, 7] = float("-inf")
    _check_routing(
        indices[rows],
        weights[rows],
        logical[rows],
        scores,
        ranking,
        num_topk=6,
        num_shared=0,
        num_experts=num_experts,
    )


@pytest.mark.parametrize("shape", [PRODUCTION, V4_FLASH], ids=["E384", "E256"])
def test_deterministic_is_batch_invariant(shape):
    """Every token routes bitwise identically whatever the call's token count."""
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4096, shape["hidden"], shape["num_experts"], seed=1)
    full = prepare_mega_gate(x, weight, 6, bias=bias, deterministic=True)
    reference = [t.clone() for t in full.run()]
    for num_tokens in (1, 16, 17, 300, 512, 1024):
        plan = prepare_mega_gate(
            x[:num_tokens], weight, 6, bias=bias, deterministic=True
        )
        assert plan.route["num_split_k"] == 1
        indices, weights = plan.run()
        torch.cuda.synchronize()
        assert torch.equal(indices, reference[0][:num_tokens])
        assert torch.equal(weights, reference[1][:num_tokens])


@pytest.mark.parametrize("num_tokens", [1, 17, 300, 512, 1024])
@pytest.mark.parametrize(
    "shape,num_shared", [(PRODUCTION, 1), (V4_FLASH, 0)], ids=["E384", "E256"]
)
def test_deterministic_matches_reference(num_tokens, shape, num_shared):
    """Deterministic schedules (no split-K): 64-token blocks of 193-512 tokens rank the
    last block with the MMA warps too."""
    _skip_unless_sm120()
    _run_case(
        num_tokens, **shape, num_shared=num_shared, with_mask=True, deterministic=True
    )


def test_graph_replay_with_changed_inputs_on_a_side_stream():
    _skip_unless_sm120()
    num_tokens = 48
    x, weight, bias = _make_inputs(num_tokens, 5120, 384, seed=2)
    mapping, counts = _physical_map(384, 384)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        plan = prepare_mega_gate(
            x, weight, 6, bias=bias, to_physical_map=mapping, logical_count=counts
        )
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        for seed in (3, 4):
            new_x, _, new_bias = _make_inputs(num_tokens, 5120, 384, seed=seed)
            x.copy_(new_x)
            bias.copy_(new_bias)
            graph.replay()
            stream.synchronize()
            eager = prepare_mega_gate(
                new_x,
                weight,
                6,
                bias=new_bias,
                to_physical_map=mapping,
                logical_count=counts,
            ).run()
            stream.synchronize()
            for replayed, expected in zip(plan.outputs, eager, strict=True):
                assert torch.equal(replayed, expected)


def test_rejects_unsupported_requests():
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4, 5120, 384)
    with pytest.raises(NotImplementedError):
        prepare_mega_gate(x, weight, 6, bias=bias, scoring_func="softmax")
    with pytest.raises(NotImplementedError):
        prepare_mega_gate(
            x,
            weight,
            6,
            bias=bias,
            force_random=torch.ones(4, dtype=torch.bool, device="cuda"),
        )
    with pytest.raises(ValueError):
        prepare_mega_gate(x[:, :100].contiguous(), weight[:, :100].contiguous(), 6)


def test_rejects_misplaced_or_misaligned_buffers():
    """Checked when the plan is prepared, not at its first run (possibly in a capture)."""
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4, 5120, 384)
    with pytest.raises(ValueError, match="same device"):
        prepare_mega_gate(x, weight.cpu(), 6, bias=bias)
    shifted_x = torch.empty(x.numel() + 1, dtype=x.dtype, device="cuda")[1:]
    with pytest.raises(ValueError, match="16-byte aligned"):
        prepare_mega_gate(shifted_x.view(x.shape), weight, 6, bias=bias)
    with pytest.raises(ValueError, match="16-byte aligned"):
        prepare_mega_gate(x, weight, 6, bias=torch.empty(385, device="cuda")[1:])
    shared_row = torch.empty((1, 6), dtype=torch.int64, device="cuda").expand(4, 6)
    with pytest.raises(ValueError, match="overlap"):
        prepare_mega_gate(x, weight, 6, bias=bias, unmapped_topk_idx=shared_row)
    scratch = prepare_mega_gate(x, weight, 6, bias=bias).scratch
    shifted = torch.empty(scratch.numel() + 1, device="cuda")[1:].view(scratch.shape)
    with pytest.raises(ValueError, match="16-byte aligned"):
        prepare_mega_gate(x, weight, 6, bias=bias, scratch=shifted)


@pytest.mark.parametrize("split", [0, -1, 2, 4])
def test_rejects_invalid_split_k(split):
    """A private schedule override must cover every 64-element K block."""
    _skip_unless_sm120()
    from dataclasses import replace
    from flashinfer.experimental.sm12x_mega_gate.mega_gate import MegaGateSm12xPlan

    x, weight, _ = _make_inputs(4, 192, 128)
    config = replace(heuristics.make_config(16, 1, 128), num_split_k=split)
    with pytest.raises(ValueError, match="num_split_k"):
        MegaGateSm12xPlan(x, weight, 2, _config=config)


@pytest.mark.parametrize(
    "pair", ["indices-logical", "weights-logical", "indices-weights"]
)
@pytest.mark.parametrize("offset", [0, 8])
def test_rejects_overlapping_outputs(pair, offset):
    """Reject exact aliases and partially overlapping views, including mixed dtypes."""
    _skip_unless_sm120()
    x, weight, _ = _make_inputs(4, 320, 12)
    storage = torch.empty(256, dtype=torch.uint8, device="cuda")
    indices = torch.empty((4, 2), dtype=torch.int64, device="cuda")
    weights = torch.empty((4, 2), device="cuda")
    logical = torch.empty_like(indices)
    if pair == "indices-logical":
        indices = storage[:64].view(torch.int64).view(4, 2)
        logical = storage[offset : offset + 64].view(torch.int64).view(4, 2)
    elif pair == "weights-logical":
        weights = storage[:32].view(torch.float32).view(4, 2)
        logical = storage[offset : offset + 64].view(torch.int64).view(4, 2)
    else:
        indices = storage[:64].view(torch.int64).view(4, 2)
        weights = storage[offset : offset + 32].view(torch.float32).view(4, 2)
    with pytest.raises(ValueError, match="overlap"):
        prepare_mega_gate(
            x, weight, 2, out=(indices, weights), unmapped_topk_idx=logical
        )


@pytest.mark.parametrize("offset", [2, 120])
def test_disjoint_outputs_can_share_storage(offset):
    """Disjoint output views are safe even when they share one allocation."""
    _skip_unless_sm120()
    x, weight, bias = _make_inputs(4, 320, 12)
    storage = torch.empty(128, dtype=torch.int64, device="cuda")
    logical = storage.as_strided((4, 2), (32, 1))
    # The first row gap, or the tail after all logical rows.
    indices = storage[offset : offset + 8].view(4, 2)
    plan = prepare_mega_gate(
        x,
        weight,
        2,
        bias=bias,
        out=(indices, torch.empty((4, 2), device="cuda")),
        unmapped_topk_idx=logical,
    )
    plan.run()
    torch.cuda.synchronize()
    assert torch.equal(indices, logical)


def test_first_invocation_capture_and_disk_reload(tmp_path):
    """Fresh processes capture their first call, on both a cache miss and a disk hit."""
    _skip_unless_sm120()
    script = textwrap.dedent("""\
        import runpy
        import sys
        from unittest.mock import patch
        import torch
        from flashinfer.jit import cute_dsl_core as cache
        from flashinfer.mega_gate import prepare_mega_gate

        helpers = runpy.run_path(sys.argv[1])
        x, weight, bias = helpers['_make_inputs'](7, 320, 12)
        logical = torch.empty((7, 2), dtype=torch.int64, device='cuda')
        builds = []
        loads = []
        original = cache.build_and_load_cute_dsl_kernel
        original_load = cache.JitSpecCuteDsl._load_from_disk

        def load(self):
            loads.append(str(self.object_path))
            return original_load(self)

        def build(module_name, kernel_name, compile_fn, **kwargs):
            def compile_once():
                builds.append(kernel_name)
                assert sys.argv[2] == 'cold', 'disk-cache reload unexpectedly compiled'
                return compile_fn()
            return original(module_name, kernel_name, compile_once, **kwargs)

        with patch.object(cache, 'build_and_load_cute_dsl_kernel', build), \
             patch.object(cache.JitSpecCuteDsl, '_load_from_disk', load):
            plan = prepare_mega_gate(x, weight, 2, bias=bias, unmapped_topk_idx=logical,
                                     enable_pdl=False)
        assert len(builds) == (1 if sys.argv[2] == 'cold' else 0), builds
        assert loads and all(__import__('pathlib').Path(p).is_file() for p in loads)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            plan.run()
        graph.replay()
        torch.cuda.synchronize()
        scores, ranking = helpers['_reference_scores'](x, weight, bias)
        helpers['_check_routing'](*plan.outputs, logical, scores, ranking,
                                  num_topk=2, num_shared=0, num_experts=12)
        assert torch.all(plan.score_barriers.view(torch.int64) == 0)
        """)
    env = dict(
        os.environ,
        FLASHINFER_WORKSPACE_BASE=str(tmp_path),
        FLASHINFER_CUTE_DSL_DISABLE_CACHE="0",
    )
    env.pop("FLASHINFER_DISABLE_JIT", None)
    for mode in ("cold", "reload"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(Path(__file__).resolve()), mode],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_replica_selection_wraparound():
    """Exercise the production hash and physical lookup after uint32 overflow."""
    _skip_unless_sm120()
    import cuda.bindings.driver as cuda
    import cutlass.cute as cute
    from cutlass import Int32
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.experimental.sm12x_mega_gate.kernel import (
        Sm12xMegaGateKernel,
        replica_hash,
    )

    gate = Sm12xMegaGateKernel(
        heuristics.make_config(16, 1, 4),
        hidden=320,
        num_experts=4,
        num_topk=2,
        num_shared=1,
        has_bias=False,
        has_image_bias=False,
        has_mask=False,
        has_physical_map=True,
        has_unmapped=False,
    )

    @cute.kernel
    def lookup(mapping: cute.Tensor, output: cute.Tensor, token: Int32, ep_rank: Int32):
        hashed = replica_hash(token, ep_rank)
        count = Int32(7)
        output[0] = gate.lookup_shared_physical(
            mapping, hashed, Int32(2), count, gate.count_magic(count), Int32(7)
        )

    @cute.jit
    def launch(mapping, output, token: Int32, ep_rank: Int32, stream: cuda.CUstream):
        lookup(mapping, output, token, ep_rank).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    mapping = torch.arange(35, dtype=torch.int32, device="cuda")
    output = torch.empty(1, dtype=torch.int32, device="cuda")
    tensors = [from_dlpack(t) for t in (mapping, output)]
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    token, ep_rank = 92037, 2**31 - 7
    compiled = cute.compile(launch, *tensors, Int32(token), Int32(ep_rank), stream)
    compiled(*tensors, Int32(token), Int32(ep_rank), stream)
    torch.cuda.synchronize()
    hashed = ep_rank + token * DUPLICATE_HASH
    assert hashed >= 2**32
    assert hashed % 7 != (hashed % 2**32) % 7
    assert output.item() == 28 + (hashed % 2**32) % 7


def test_rejects_single_row_alias_with_zero_stride():
    """A contiguous single row may legally have a zero row stride."""
    _skip_unless_sm120()
    x, weight, _ = _make_inputs(1, 320, 12)
    indices = torch.empty((1, 2), dtype=torch.int64, device="cuda")
    logical = indices.as_strided((1, 2), (0, 1))
    weights = torch.empty((1, 2), device="cuda")
    with pytest.raises(ValueError, match="overlap"):
        prepare_mega_gate(
            x, weight, 2, out=(indices, weights), unmapped_topk_idx=logical
        )
