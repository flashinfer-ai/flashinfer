"""1/2/4/8-rank correctness for production SM90 Humming MXFP4 MegaMoE.

Launch this file in its own process because the SM90 and SM100 CuTeDSL drops
use colliding top-level module names::

    torchrun --nproc_per_node=2 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_mega_multirank.py -v \
        -m "gpu_2 and arch_hopper"
    torchrun --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_mega_multirank.py -v \
        -m "gpu_2 and arch_hopper"
    torchrun --standalone --nproc_per_node=8 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_mxfp4_mega_multirank.py -v \
        -m "gpu_2 and arch_hopper"

The layer receives the production raw ABI: packed E2M1 payload bytes and K32
E8M0 exponent planes in ``PrequantizedMoEWeights``.  The oracle is deliberately
test-owned.  It uses the independent Humming preprocessing reference, forms
the transient E4M3 operands mathematically, and implements global expert
routing, SwiGLU, per-token/K64 FC2-input quantization, and top-k reduction in
this file.  It never imports the donor or the vendored raw kernel packages.

Every source token selects one expert on every EP rank.  Consequently the
2-rank, 4-rank, and 8-rank launches exercise local and peer pulls, every
destination rank receives remote tokens, and every process validates every
rank's output.
Three forwards on one layer additionally guard counter cleanup, launch-cache
reuse, and reuse of the same symmetric workspace.  A separate test captures
one fully specified fused tactic in an outer ``torch.cuda.CUDAGraph`` and
replays it in rank lockstep to guard graph liveness and pointer stability.
"""

from __future__ import annotations

import os

import pytest


# Import only the public shim boundary.  The package keeps CUDA/CuTe imports
# lazy, while also preserving the SM90/SM100 process-isolation guard.
pytest.importorskip("flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel")


HIDDEN = 128
INTERMEDIATE = 128
LOCAL_EXPERTS = 4
TOKENS_PER_RANK = 8
K64 = 64
E4M3_MAX = 448.0
GATE_UP_CLAMP = 10.0
GRAPH_WARMUPS = 3
GRAPH_REPLAYS = 16


def _launcher_ranks() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    return rank, world_size, local_rank


def _pack_e2m1_codes(codes):
    import torch

    assert codes.dtype == torch.uint8
    assert codes.shape[-1] % 2 == 0
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _make_scale_plane(
    *,
    global_expert_begin: int,
    rows: int,
    k32_groups: int,
):
    """Make bounded O(1) weights with one >11-span clamp case per expert."""
    import torch

    expert = torch.arange(LOCAL_EXPERTS, dtype=torch.int64).view(-1, 1, 1)
    row = torch.arange(rows, dtype=torch.int64).view(1, -1, 1)
    group = torch.arange(k32_groups, dtype=torch.int64).view(1, 1, -1)
    global_expert = expert + global_expert_begin

    # Typical exponents 121..125 yield useful, well-conditioned outputs.  A
    # single scale is 15 below the expert maximum, so Humming must clamp it to
    # the retained 11-wide window and rewrite the corresponding E2M1 payload.
    scale = 121 + global_expert.remainder(3) + (row + 2 * group).remainder(3)
    scale = scale.expand(LOCAL_EXPERTS, rows, k32_groups).clone()
    expert_max = scale.reshape(LOCAL_EXPERTS, -1).amax(dim=1)
    scale[:, 0, 0] = expert_max - 15
    return scale.to(torch.uint8).contiguous()


def _make_raw_weights(rank: int):
    """Return this rank's canonical raw packed MXFP4/E8M0 production pack."""
    import torch

    from flashinfer.moe_ep import PrequantizedMoEWeights

    generator = torch.Generator(device="cpu").manual_seed(1701 + rank)

    def payload(rows: int, logical_k: int):
        codes = torch.randint(
            0,
            16,
            (LOCAL_EXPERTS, rows, logical_k),
            dtype=torch.uint8,
            generator=generator,
        )
        # Humming canonicalizes negative zero too; doing so here avoids
        # spending random mass on a second representation of zero.
        codes[codes == 8] = 0
        return _pack_e2m1_codes(codes).cuda()

    global_expert_begin = rank * LOCAL_EXPERTS
    w13 = payload(2 * INTERMEDIATE, HIDDEN)
    w2 = payload(HIDDEN, INTERMEDIATE)
    w13_scale = _make_scale_plane(
        global_expert_begin=global_expert_begin,
        rows=2 * INTERMEDIATE,
        k32_groups=HIDDEN // 32,
    ).cuda()
    w2_scale = _make_scale_plane(
        global_expert_begin=global_expert_begin,
        rows=HIDDEN,
        k32_groups=INTERMEDIATE // 32,
    ).cuda()
    return PrequantizedMoEWeights(
        w13=w13,
        w2=w2,
        w13_scale=w13_scale,
        w2_scale=w2_scale,
    )


def _make_tokens_and_routes(
    rank: int, world_size: int, *, launch: int, routing_pattern: str = "cross_rank"
):
    import torch

    generator = torch.Generator(device="cpu").manual_seed(2909 + 97 * launch + rank)
    hidden = (
        0.75
        * torch.randn(
            TOKENS_PER_RANK,
            HIDDEN,
            dtype=torch.float32,
            generator=generator,
        )
        + 0.03125 * (rank + 1)
    ).to(torch.bfloat16)

    token = torch.arange(TOKENS_PER_RANK, dtype=torch.int64).view(-1, 1)
    slot = torch.arange(world_size, dtype=torch.int64).view(1, -1)
    owner = (rank + slot + launch) % world_size
    local_expert = (token + slot + launch) % LOCAL_EXPERTS
    if routing_pattern == "sparse_owner":
        # Keep top-k unique but concentrate all ranks' tokens onto the fewest
        # possible owners. Rotate those owners on the next launch, so a rank
        # transitions between empty and active on the SAME workspace.
        owner = (slot // LOCAL_EXPERTS + launch) % world_size
        local_expert = (token * 0 + slot) % LOCAL_EXPERTS
    elif routing_pattern not in ("cross_rank", "zero_source", "all_masked"):
        raise ValueError(f"unknown routing pattern: {routing_pattern}")
    topk_ids = owner * LOCAL_EXPERTS + local_expert
    if (routing_pattern == "zero_source" and rank == launch % world_size) or (
        routing_pattern == "all_masked" and launch == 0
    ):
        topk_ids = torch.full_like(topk_ids, -1)

    # Exact binary fractions prevent routing-weight representation noise while
    # still making every slot numerically distinct.
    if world_size == 1:
        route_weights = torch.tensor([1.0], dtype=torch.float32)
    elif world_size == 2:
        route_weights = torch.tensor([0.25, 0.75], dtype=torch.float32)
    elif world_size == 4:
        route_weights = torch.tensor([0.125, 0.25, 0.25, 0.375], dtype=torch.float32)
    elif world_size == 8:
        # Eight distinct dyadic weights that sum exactly to one.
        route_weights = torch.arange(121, 136, 2, dtype=torch.float32) / 1024.0
    else:
        raise AssertionError(f"unsupported WORLD_SIZE={world_size}")
    topk_weights = route_weights.expand(TOKENS_PER_RANK, world_size).contiguous()
    return hidden.cuda(), topk_ids.cuda(), topk_weights.cuda()


def _all_gather_stack(tensor):
    import torch
    import torch.distributed as dist

    tensor = tensor.contiguous()
    gathered = [torch.empty_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, tensor)
    return torch.stack(gathered)


def _interleave_gate_up_8(tensor):
    """Canonical ``gate || up`` rows to the kernel's ``gate8, up8`` order."""
    import torch

    experts, rows, cols = tensor.shape
    assert rows == 2 * INTERMEDIATE
    pairs = INTERMEDIATE // 8
    gate = tensor[:, :INTERMEDIATE].reshape(experts, pairs, 8, cols)
    up = tensor[:, INTERMEDIATE:].reshape(experts, pairs, 8, cols)
    return torch.stack((gate, up), dim=2).reshape(experts, rows, cols).contiguous()


def _unpack_e2m1(processed):
    import torch

    low = processed & 0x0F
    high = (processed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(
        *processed.shape[:-1], processed.shape[-1] * 2
    )
    magnitude_lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=processed.device,
    )
    value = magnitude_lut[(codes & 0x07).long()]
    return torch.where((codes & 0x08) != 0, -value, value)


def _humming_fp8_operand(payload, raw_scale, *, gate_up: bool):
    """Form the exact transient E4M3 operand and per-expert common scale."""
    import torch

    from tests.moe_ep._sm90_mxfp4_humming_reference import reference_preprocess

    if gate_up:
        payload = _interleave_gate_up_8(payload)
        raw_scale = _interleave_gate_up_8(raw_scale)

    processed, offset, residual = reference_preprocess(
        payload,
        raw_scale,
    )
    value = _unpack_e2m1(processed)
    relative_exponent = offset.repeat_interleave(32, dim=-1).to(torch.float32) - 6.0
    # Every value is exactly representable in E4M3; the cast documents and
    # checks the operand format used by Hopper WGMMA.
    operand = (value * torch.exp2(relative_exponent)).to(torch.float8_e4m3fn)
    return operand, residual.to(torch.float32) * 64.0


def _prepare_global_humming_operands(raw_global):
    ranks, local_experts = raw_global.w13.shape[:2]

    def flatten(tensor):
        return tensor.reshape(ranks * local_experts, *tensor.shape[2:])

    fc1, fc1_common = _humming_fp8_operand(
        flatten(raw_global.w13),
        flatten(raw_global.w13_scale),
        gate_up=True,
    )
    fc2, fc2_common = _humming_fp8_operand(
        flatten(raw_global.w2),
        flatten(raw_global.w2_scale),
        gate_up=False,
    )
    return (
        fc1.reshape(ranks, local_experts, 2 * INTERMEDIATE, HIDDEN),
        fc1_common.reshape(ranks, local_experts),
        fc2.reshape(ranks, local_experts, HIDDEN, INTERMEDIATE),
        fc2_common.reshape(ranks, local_experts),
    )


def _quantize_input_per_token(hidden):
    import torch

    fp32 = hidden.to(torch.float32)
    scale = (fp32.abs().amax(dim=-1, keepdim=True) / E4M3_MAX).clamp_min(1.0e-30)
    return (fp32 / scale).to(torch.float8_e4m3fn), scale


def _fast_fp8_mm(a, b):
    """Independent native K32 sequence, not a differently tiled library GEMM."""
    import torch
    from tests.moe_ep._sm90_fp8_wgmma_reference import rs_k32_mm

    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    rows = a.shape[0]
    if rows == 0:
        return torch.empty((0, b.shape[1]), device=a.device, dtype=torch.float32)
    return torch.cat(
        [rs_k32_mm(a[begin : begin + 64], b) for begin in range(0, rows, 64)], dim=0
    )


def _swiglu_sm90_formula(gate, up):
    """Independent bit-match of the SM90 exp2/reciprocal SwiGLU formula."""
    from tests.moe_ep._sm90_swiglu_reference import swiglu_sm90_reference

    return swiglu_sm90_reference(gate, up)


def _global_route_reference(
    hidden, topk_ids, topk_weights, raw_global, *, require_all_experts=True
):
    """Compute all ranks' fused output from raw operands and global routing."""
    import torch
    from tests.moe_ep._sm90_fp8_wgmma_reference import fma_add

    world_size, num_tokens, topk = topk_ids.shape
    assert topk == world_size
    fc1, fc1_common, fc2, fc2_common = _prepare_global_humming_operands(raw_global)
    input_fp8, input_scale = _quantize_input_per_token(hidden)
    terms = torch.zeros(
        world_size,
        num_tokens,
        topk,
        HIDDEN,
        dtype=torch.bfloat16,
        device=hidden.device,
    )

    for global_expert in range(world_size * LOCAL_EXPERTS):
        routed = (topk_ids == global_expert).nonzero(as_tuple=False)
        if require_all_experts:
            assert routed.numel() > 0, (
                f"global expert {global_expert} was not exercised"
            )
        elif routed.numel() == 0:
            # Explicit sparse-route test: an unselected expert contributes
            # nothing. Dense coverage remains mandatory for existing callers.
            continue
        source_rank, source_token, source_slot = routed.unbind(dim=1)
        target_rank = global_expert // LOCAL_EXPERTS
        local_expert = global_expert % LOCAL_EXPERTS

        fc1_raw = _fast_fp8_mm(
            input_fp8[source_rank, source_token],
            fc1[target_rank, local_expert].transpose(0, 1),
        )
        fc1_output = (
            fc1_raw
            * input_scale[source_rank, source_token]
            * fc1_common[target_rank, local_expert]
        )
        paired = fc1_output.reshape(-1, INTERMEDIATE // 8, 2, 8)
        gate = paired[:, :, 0].clamp(max=GATE_UP_CLAMP)
        up = paired[:, :, 1].clamp(min=-GATE_UP_CLAMP, max=GATE_UP_CLAMP)
        swiglu = _swiglu_sm90_formula(gate, up).reshape(-1, INTERMEDIATE)
        swiglu.mul_(topk_weights[source_rank, source_token, source_slot].unsqueeze(1))

        grouped = swiglu.reshape(-1, INTERMEDIATE // K64, K64)
        fc2_scale = (grouped.abs().amax(dim=2, keepdim=True) / E4M3_MAX).clamp_min(
            1.0e-30
        )
        fc2_input = (grouped * torch.reciprocal(fc2_scale)).to(torch.float8_e4m3fn)

        fc2_accum = torch.zeros(
            (routed.shape[0], HIDDEN), dtype=torch.float32, device=hidden.device
        )
        for group in range(INTERMEDIATE // K64):
            begin = group * K64
            end = begin + K64
            partial = _fast_fp8_mm(
                fc2_input[:, group].contiguous(),
                fc2[target_rank, local_expert, :, begin:end].transpose(0, 1),
            )
            fc2_accum = fma_add(fc2_accum, partial, fc2_scale[:, group])
        fc2_output = fc2_accum * fc2_common[target_rank, local_expert]
        terms[source_rank, source_token, source_slot] = fc2_output.to(torch.bfloat16)

    # The fused kernel emits one BF16 term per top-k slot; its standalone
    # reducer accumulates those terms into the final BF16 token output.
    return terms.to(torch.float32).sum(dim=2)


class _RawGlobalWeights:
    def __init__(self, *, w13, w2, w13_scale, w2_scale):
        self.w13 = w13
        self.w2 = w2
        self.w13_scale = w13_scale
        self.w2_scale = w2_scale


def _gather_raw_weights(raw):
    return _RawGlobalWeights(
        w13=_all_gather_stack(raw.w13),
        w2=_all_gather_stack(raw.w2),
        w13_scale=_all_gather_stack(raw.w13_scale),
        w2_scale=_all_gather_stack(raw.w2_scale),
    )


def _assert_cross_rank_coverage(topk_ids, world_size: int) -> None:
    import torch

    owners = topk_ids // LOCAL_EXPERTS
    expected = torch.arange(world_size, device=topk_ids.device)
    for source_rank in range(world_size):
        assert torch.equal(torch.unique(owners[source_rank]).sort().values, expected)
        if world_size > 1:
            assert (owners[source_rank] != source_rank).any()
    counts = torch.bincount(topk_ids.flatten(), minlength=world_size * LOCAL_EXPERTS)
    assert (counts > 0).all()


def _assert_matches_reference(actual, expected, *, launch: int) -> None:
    import torch

    assert actual.shape == expected.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    actual_fp32 = actual.to(torch.float32)
    diff = actual_fp32 - expected
    rel_l2 = diff.norm() / expected.norm().clamp_min(1.0e-6)
    print(
        f"[sm90 mxfp4 multirank launch={launch}] "
        f"rel_l2={rel_l2.item():.5g} max|d|={diff.abs().max().item():.5g} "
        f"amax(ref)={expected.abs().max().item():.5g}"
    )
    torch.testing.assert_close(actual_fp32, expected, atol=2.0e-2, rtol=2.0e-2)
    assert rel_l2.item() < 2.5e-2


def _complete_fused_graph_tactic() -> dict:
    """Small-shape fused tactic with every runtime identity field explicit."""

    return {
        "active_dispatch_warps": 1,
        "cluster_shape_mnk": (1, 1, 1),
        "combine_format": "bf16",
        "dedup_dispatch": False,
        "fc1_early_done_publish": False,
        "fc1_store_offload": True,
        "fold_producer_warps": True,
        "fp8_accum_mode": "1xacc",
        "group_hint": 132,
        "grouped_token_back": False,
        "in_kernel_fc2_reduce": False,
        "load_balance_mode": "static",
        "mma_tiler_mnk": (128, 32, 128),
        "num_sched_stages": 2,
        "pingpong": False,
        "swap_ab": True,
        "token_back_mode": "epi_warps",
    }


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "routing_pattern", ("cross_rank", "sparse_owner", "zero_source", "all_masked")
)
def test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
    routing_pattern,
    *,
    tactic=None,
    expected_policy=None,
    expected_kernel=None,
):
    """Production raw ABI vs independent global math on 1, 2, 4, or 8 ranks."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    assert torch.cuda.is_available(), "gpu_2 test collected without CUDA"
    rank, world_size, local_rank = _launcher_ranks()
    assert world_size in (1, 2, 4, 8), (
        "launch this test with torchrun --nproc_per_node=1, 2, 4, or 8; "
        f"got WORLD_SIZE={world_size}"
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        device=local_rank,
    )
    ensure_moe_ep_cuda_device(bootstrap)
    geometry = dict(
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(1, 1, 1),
        load_balance_mode="static",
        token_back_mode="epi_warps",
    )
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=world_size,
        gate_up_clamp=GATE_UP_CLAMP,
        **(geometry if tactic is None else {"knobs": tactic}),
    )
    registry_kernel = create_mega_kernel(config)
    assert registry_kernel.kernel_name() == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        registry_kernel.runtime_requirements(bootstrap),
    )

    layer = None
    try:
        raw = _make_raw_weights(rank)
        raw_global = _gather_raw_weights(raw)
        launches = [
            _make_tokens_and_routes(
                rank, world_size, launch=launch, routing_pattern=routing_pattern
            )
            for launch in range(2)
        ]
        hidden_global = [_all_gather_stack(item[0]) for item in launches]
        ids_global = [_all_gather_stack(item[1]) for item in launches]
        weights_global = [_all_gather_stack(item[2]) for item in launches]
        for launch, ids in enumerate(ids_global):
            if routing_pattern == "cross_rank":
                _assert_cross_rank_coverage(ids, world_size)
            elif routing_pattern == "sparse_owner":
                active_owners = torch.unique(ids // LOCAL_EXPERTS).numel()
                assert (
                    active_owners == (world_size + LOCAL_EXPERTS - 1) // LOCAL_EXPERTS
                )
                if world_size > 1:
                    assert active_owners < world_size
                assert torch.unique(ids).numel() < world_size * LOCAL_EXPERTS
            elif routing_pattern == "zero_source":
                assert (ids[launch % world_size] == -1).all()
            elif routing_pattern == "all_masked":
                if launch == 0:
                    assert (ids == -1).all()
                else:
                    _assert_cross_rank_coverage(ids, world_size)
        assert not torch.equal(ids_global[0], ids_global[1])

        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            fleet_params=FleetParams(
                num_experts=world_size * LOCAL_EXPERTS,
                max_tokens_per_rank=TOKENS_PER_RANK,
                token_hidden_size=HIDDEN,
            ),
            weights=raw,
            backend=MegaConfig(
                megakernel=config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        assert isinstance(layer, MoEEpMegaLayer)

        def tensors(item):
            hidden, ids, weights = item
            return MoEEpTensors(
                hidden_states=hidden,
                topk_ids=ids,
                topk_weights=weights,
            )

        first = layer.forward(tensors(launches[0])).clone()
        workspace = layer._workspace
        assert workspace is not None
        if expected_kernel is not None:
            kernel = workspace._frontend._mega.kernel
            for field, value in expected_kernel.items():
                assert getattr(kernel, field) == value, (field, value)
            if "tail_split_pairs" in expected_kernel:
                assert (
                    workspace._frontend.effective_tactic()["tail_split_pairs"]
                    is expected_kernel["tail_split_pairs"]
                )
        if expected_policy is not None:
            policy = workspace._frontend._mega.kernel.mxfp4_optimizations
            for field, value in expected_policy.items():
                assert getattr(policy, field) == value, (field, policy)
        second = layer.forward(tensors(launches[1])).clone()
        assert layer._workspace is workspace
        second_repeat = layer.forward(tensors(launches[1])).clone()
        assert layer._workspace is workspace
        torch.cuda.synchronize()
        torch.testing.assert_close(second_repeat, second, atol=0.0, rtol=0.0)
        if expected_kernel is not None:
            from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.comm import (
                reset_compiled_mega_workspaces,
            )

            dist.barrier()
            reset_compiled_mega_workspaces(workspace._frontend._mega)
            torch.cuda.synchronize()
            dist.barrier()
            after_reset = layer.forward(tensors(launches[1])).clone()
            torch.cuda.synchronize()
            torch.testing.assert_close(after_reset, second, atol=0.0, rtol=0.0)

        actual_global = [
            _all_gather_stack(first),
            _all_gather_stack(second),
        ]
        expected_global = [
            _global_route_reference(
                hidden_global[launch],
                ids_global[launch],
                weights_global[launch],
                raw_global,
                require_all_experts=routing_pattern == "cross_rank",
            )
            for launch in range(2)
        ]
        for launch, (actual, expected) in enumerate(
            zip(actual_global, expected_global, strict=True)
        ):
            _assert_matches_reference(actual, expected, launch=launch)
        # Expert-specific weights and rank-specific activations must prevent a
        # rank-0-only or wrong-owner implementation from passing accidentally.
        if world_size > 1 and routing_pattern != "all_masked":
            assert not torch.equal(expected_global[0][0], expected_global[0][1])
        print(
            f"rank {rank}: production SM90 MXFP4 fused MegaMoE matched the "
            f"independent global oracle for all {world_size} ranks"
        )
        dist.barrier()
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "tile,hidden,intermediate,tokens,return_mode,pingpong,routing_pattern",
    [
        pytest.param(
            (256, 16, 128),
            768,
            384,
            12,
            "epi_warps",
            False,
            "cross_rank",
            id="k128-odd-weights-tail",
        ),
        pytest.param(
            (128, 16, 128),
            256,
            256,
            16,
            "epi_warps",
            True,
            "cross_rank",
            id="k128-pingpong-tail",
        ),
        pytest.param(
            (128, 16, 128),
            384,
            256,
            17,
            "reuse_dispatch_warps",
            False,
            "cross_rank",
            id="k128-even-token-tiles",
        ),
        pytest.param(
            (256, 16, 256),
            768,
            256,
            12,
            "epi_warps",
            False,
            "cross_rank",
            id="k256-odd-weight-tail",
        ),
        pytest.param(
            (128, 16, 256),
            256,
            256,
            48,
            "reuse_dispatch_warps",
            False,
            "cross_rank",
            id="k256-tail-after-full-cluster",
        ),
        pytest.param(
            (128, 16, 256),
            256,
            256,
            32,
            "standalone_warps",
            False,
            "cross_rank",
            id="k256-even-token-tiles",
        ),
        pytest.param(
            (256, 16, 128),
            768,
            384,
            3,
            "epi_warps",
            False,
            "sparse_owner",
            id="k128-empty-rank-tail",
        ),
        pytest.param(
            (128, 16, 256),
            256,
            256,
            3,
            "reuse_dispatch_warps",
            False,
            "sparse_owner",
            id="k256-empty-rank-tail",
        ),
        pytest.param(
            (128, 16, 256),
            256,
            256,
            12,
            "standalone_warps",
            False,
            "all_masked",
            id="k256-all-empty-reset",
        ),
        pytest.param(
            (256, 16, 128),
            768,
            384,
            12,
            "reuse_dispatch_warps",
            False,
            "zero_source",
            id="k128-empty-source-tail",
        ),
    ],
)
def test_moe_ep_sm90_pull_mxfp4_tail_pairs_independent_oracle(
    monkeypatch,
    tile,
    hidden,
    intermediate,
    tokens,
    return_mode,
    pingpong,
    routing_pattern,
):
    """Packed tail pairs vs full raw oracle, with repeat and empty-rank reset."""
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "HIDDEN", hidden)
    monkeypatch.setattr(module, "INTERMEDIATE", intermediate)
    monkeypatch.setattr(module, "TOKENS_PER_RANK", tokens)
    tactic = dict(
        _complete_fused_graph_tactic(),
        mma_tiler_mnk=tile,
        cluster_shape_mnk=(1, 2, 1),
        tail_split_pairs=True,
        pingpong=pingpong,
        token_back_mode=return_mode,
    )
    test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
        routing_pattern,
        tactic=tactic,
        expected_kernel={"tail_split_pairs": True},
    )


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "hidden,intermediate,local_experts,tokens,tile,tail_n8",
    [
        pytest.param(384, 384, 3, 12, (256, 32, 128), False, id="k128-only"),
        pytest.param(
            768, 512, 3, 12, (256, 16, 256), False, id="k256-odd-weight-tiles"
        ),
        pytest.param(4096, 768, 4, 16, (256, 64, 256), False, id="wide-hidden"),
        pytest.param(7168, 512, 4, 4, (256, 64, 256), True, id="n8-tail"),
    ],
)
def test_moe_ep_sm90_pull_mxfp4_tail_candidates_other_shapes(
    monkeypatch, hidden, intermediate, local_experts, tokens, tile, tail_n8
):
    """Execute normal tuning candidates beyond the original model/EP shape."""
    import sys

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_mxfp4_candidates,
    )

    module = sys.modules[__name__]
    for field, value in (
        ("HIDDEN", hidden),
        ("INTERMEDIATE", intermediate),
        ("LOCAL_EXPERTS", local_experts),
        ("TOKENS_PER_RANK", tokens),
    ):
        monkeypatch.setattr(module, field, value)
    _, world_size, _ = _launcher_ranks()
    candidates = hopper_mxfp4_candidates(
        tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=local_experts * world_size,
        world_size=world_size,
    )
    tactic = next(
        candidate
        for candidate in candidates
        if candidate["tail_split_pairs"]
        and candidate["mma_tiler_mnk"] == tile
        and candidate["fc2_tail_n8"] == tail_n8
    )
    test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
        "cross_rank",
        tactic=tactic,
        expected_kernel={"tail_split_pairs": True},
        expected_policy={"fc2_tail_n8": tail_n8},
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "tile,hidden,cluster,tail_pairs",
    [
        pytest.param(
            (256, 16, 128), 384, (1, 2, 1), True, id="k128-partial-m-tail-pair"
        ),
        pytest.param(
            (256, 16, 128), 384, (2, 1, 1), False, id="k128-partial-m-cluster"
        ),
        pytest.param(
            (256, 16, 256), 768, (2, 1, 1), False, id="k256-padded-cta-bulk-and-cpasync"
        ),
    ],
)
def test_moe_ep_sm90_pull_mxfp4_offset_m_boundary(
    monkeypatch, tile, hidden, cluster, tail_pairs
):
    """Partial/padded weight tiles keep offsets in bounds and reset cleanly."""
    import dataclasses
    import sys

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "HIDDEN", hidden)
    monkeypatch.setattr(module, "INTERMEDIATE", 256)
    monkeypatch.setattr(module, "TOKENS_PER_RANK", 12)
    tactic = dict(
        _complete_fused_graph_tactic(),
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cluster,
        tail_split_pairs=tail_pairs,
    )
    for bulk in (True, False) if tile[2] == 256 else (False,):
        if tile[2] == 256 and not bulk:
            # The preceding forward imported the raw kernel. Match the host
            # compile identity and device policy, as in the bulk/reset test.
            kernel_module = sys.modules[
                "moe_hopper_fp8.kernel_mxfp4_fp8_glu_fc12_swapab"
            ]
            for target in (hopper_mxfp4, kernel_module):
                resolve = target.resolve_mxfp4_optimizations

                def cpasync_offsets(*args, _resolve=resolve, **kwargs):
                    return dataclasses.replace(
                        _resolve(*args, **kwargs), offset_bulk=False
                    )

                monkeypatch.setattr(
                    target, "resolve_mxfp4_optimizations", cpasync_offsets
                )
        test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
            "cross_rank",
            tactic=tactic,
            expected_kernel={"tail_split_pairs": tail_pairs},
            expected_policy={"offset_bulk": bulk},
        )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "hidden,tile,dedup,tail,routing_pattern",
    [
        (256, (128, 32, 256), False, False, "cross_rank"),
        (256, (256, 64, 256), True, False, "sparse_owner"),
        (7168, (256, 64, 256), False, False, "cross_rank"),
        (7168, (256, 64, 256), False, True, "cross_rank"),
    ],
)
def test_moe_ep_sm90_pull_mxfp4_local_optimizations_independent_oracle(
    monkeypatch, hidden, tile, dedup, tail, routing_pattern
):
    """Small expert/I fixture, independent raw math; never a perf substitute."""
    import sys

    module = sys.modules[__name__]
    monkeypatch.setattr(module, "HIDDEN", hidden)
    monkeypatch.setattr(module, "INTERMEDIATE", 256)
    tactic = dict(
        _complete_fused_graph_tactic(),
        mma_tiler_mnk=tile,
        dedup_dispatch=dedup,
        fc2_tail_n8=tail,
        fc1_ready_mode="tile",
    )
    test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
        routing_pattern,
        tactic=tactic,
        expected_policy=dict(
            peer32=hidden == 7168,
            offset_bulk=True,
            skip_zero_counts=not dedup,
            fc2_tail_n8=tail,
            fc1_ready_mode="tile",
        ),
    )


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize("cluster_m,group_hint", [(1, 528), (2, 330)])
def test_mxfp4_bulk_offsets_match_cpasync_after_repeat_and_reset(
    monkeypatch, cluster_m, group_hint
):
    """Real failing geometry: full outputs, 20 repeats, and workspace reset."""
    import dataclasses
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace

    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
        preprocess_sm90_pull_mxfp4_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_mxfp4,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.comm import (
        reset_compiled_mega_workspaces,
    )

    rank, world_size, local_rank = _launcher_ranks()
    assert world_size == 4, "launch this regression with torchrun --nproc_per_node=4"
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    bootstrap = BootstrapConfig(world_size=world_size, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    device = torch.device("cuda", local_rank)
    tactic = dict(
        _complete_fused_graph_tactic(),
        mma_tiler_mnk=(128, 64, 256),
        cluster_shape_mnk=(cluster_m, 1, 1),
        active_dispatch_warps=4,
        fold_producer_warps=False,
        fc1_store_offload=False,
        fc1_early_done_publish=False,
        group_hint=group_hint,
        load_balance_mode="atomic_counter",
        token_back_mode="reuse_dispatch_warps",
    )
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=3072, top_k=6, gate_up_clamp=10.0, knobs=tactic
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    layer = None
    try:
        # Reuse the input recipe which exposed this race; the small oracle
        # fixture does not reproduce it. Keep the mathematical oracle separate.
        path = (
            Path(__file__).resolve().parents[2] / "benchmarks/bench_moe_ep_sm90_mega.py"
        )
        spec = importlib.util.spec_from_file_location("_mxfp4_repeat_benchmark", path)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, bench)
        spec.loader.exec_module(bench)
        args = SimpleNamespace(
            hidden=7168,
            intermediate=3072,
            num_experts=384,
            top_k=6,
            routing_mode="block_permutation",
        )
        raw = bench._make_raw_mxfp4_weights(args, 96, rank, device)
        weights = preprocess_sm90_pull_mxfp4_mega_weights(
            raw, intermediate_size=3072, hidden_size=7168
        )
        del raw
        hidden, ids, scores = bench._make_point_inputs(
            args, 4096, rank, world_size, device
        )
        tensors = MoEEpTensors(hidden_states=hidden, topk_ids=ids, topk_weights=scores)

        def assert_equal(actual, expected):
            invalid = not torch.isfinite(actual).all() or not torch.equal(
                actual.view(torch.uint8), expected.view(torch.uint8)
            )
            status = torch.tensor(int(invalid), dtype=torch.int32, device=device)
            dist.all_reduce(status, op=dist.ReduceOp.MAX)
            assert status.item() == 0, "MXFP4 bulk offset output byte mismatch"

        bulk_output = None
        for bulk in (True, False):
            if not bulk:
                # The host compile identity and device specialization must both
                # select the existing non-bulk copy implementation for control.
                kernel_module = sys.modules[
                    "moe_hopper_fp8.kernel_mxfp4_fp8_glu_fc12_swapab"
                ]
                for module in (hopper_mxfp4, kernel_module):
                    resolve = module.resolve_mxfp4_optimizations

                    def cpasync_offsets(*args, _resolve=resolve, **kwargs):
                        return dataclasses.replace(
                            _resolve(*args, **kwargs), offset_bulk=False
                        )

                    monkeypatch.setattr(
                        module, "resolve_mxfp4_optimizations", cpasync_offsets
                    )
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    world_size=world_size,
                    rank=rank,
                    device=local_rank,
                    auto_bootstrap=False,
                ),
                fleet_params=FleetParams(
                    num_experts=384,
                    max_tokens_per_rank=4096,
                    token_hidden_size=7168,
                ),
                weights=None,
                backend=MegaConfig(
                    megakernel=config,
                    quantize_input=True,
                    preprocess_weights=False,
                    transformed_weights=weights,
                ),
            )
            first = layer.forward(tensors).detach().cpu().contiguous()
            workspace = layer._workspace
            compiled = workspace._frontend._mega
            assert compiled.kernel.mxfp4_optimizations.offset_bulk is bulk
            for _ in range(20):
                actual = layer.forward(tensors).detach().cpu().contiguous()
                assert layer._workspace is workspace
                assert_equal(actual, first)
            torch.cuda.synchronize()
            dist.barrier()
            reset_compiled_mega_workspaces(compiled)
            torch.cuda.synchronize()
            dist.barrier()
            assert_equal(layer.forward(tensors).detach().cpu().contiguous(), first)
            if bulk:
                bulk_output = first
            else:
                assert_equal(first, bulk_output)
            layer.destroy()
            layer = None
    finally:
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
@pytest.mark.parametrize("tail_split_pairs", (False, True), ids=("tail-off", "tail-on"))
def test_moe_ep_sm90_pull_mxfp4_fused_outer_graph_replay_matches_oracle(
    tail_split_pairs,
):
    """Capture once and replay a fixed fused call collectively 16 times."""
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpMegaLayer,
        MoEEpTensors,
        Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    assert torch.cuda.is_available(), "gpu_2 test collected without CUDA"
    rank, world_size, local_rank = _launcher_ranks()
    assert world_size in (1, 2, 4, 8), (
        "launch this graph test with torchrun --nproc_per_node=1, 2, 4, or 8; "
        f"got WORLD_SIZE={world_size}"
    )

    bootstrap = BootstrapConfig(
        world_size=world_size,
        rank=rank,
        device=local_rank,
    )
    ensure_moe_ep_cuda_device(bootstrap)
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=world_size,
        knobs=dict(
            _complete_fused_graph_tactic(),
            cluster_shape_mnk=(1, 2, 1) if tail_split_pairs else (1, 1, 1),
            tail_split_pairs=tail_split_pairs,
        ),
        gate_up_clamp=GATE_UP_CLAMP,
    )
    registry_kernel = create_mega_kernel(config)
    assert registry_kernel.kernel_name() == "sm90_fp8_mxfp4_bf16_pull_cutedsl"
    runtime = bootstrap_moe_ep_runtime(
        bootstrap,
        registry_kernel.runtime_requirements(bootstrap),
    )

    layer = None
    graph = None
    captured_output = None
    try:
        raw = _make_raw_weights(rank)
        raw_global = _gather_raw_weights(raw)
        launch = _make_tokens_and_routes(rank, world_size, launch=0)
        hidden_global = _all_gather_stack(launch[0])
        ids_global = _all_gather_stack(launch[1])
        weights_global = _all_gather_stack(launch[2])
        _assert_cross_rank_coverage(ids_global, world_size)

        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=world_size,
                rank=rank,
                auto_bootstrap=False,
                device=local_rank,
            ),
            fleet_params=FleetParams(
                num_experts=world_size * LOCAL_EXPERTS,
                max_tokens_per_rank=TOKENS_PER_RANK,
                token_hidden_size=HIDDEN,
            ),
            weights=raw,
            backend=MegaConfig(
                megakernel=config,
                quantize_input=True,
                preprocess_weights=True,
            ),
        )
        assert isinstance(layer, MoEEpMegaLayer)
        tensors = MoEEpTensors(
            hidden_states=launch[0],
            topk_ids=launch[1],
            topk_weights=launch[2],
        )
        input_ptrs = (
            tensors.hidden_states.data_ptr(),
            tensors.topk_ids.data_ptr(),
            tensors.topk_weights.data_ptr(),
        )

        # The first call resolves every lazy allocation/JIT path.  Two more
        # collective calls exercise the exact fixed inputs before capture.
        layer.warmup(tensors)
        for _ in range(GRAPH_WARMUPS - 1):
            layer.forward(tensors)
        torch.cuda.synchronize()
        dist.barrier()

        eager = layer.forward(tensors).clone()
        torch.cuda.synchronize()
        dist.barrier()

        # Capture records each rank independently.  The barriers ensure no
        # rank starts a cross-rank replay while a peer is still capturing.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = layer.forward(tensors)
        captured_output_ptr = captured_output.data_ptr()
        dist.barrier()

        replayed = []
        for _ in range(GRAPH_REPLAYS):
            graph.replay()
            torch.cuda.synchronize()
            replayed.append(captured_output.clone())
            torch.cuda.synchronize()
            dist.barrier()

        input_ptrs_after = (
            tensors.hidden_states.data_ptr(),
            tensors.topk_ids.data_ptr(),
            tensors.topk_weights.data_ptr(),
        )
        captured_output_ptr_after = captured_output.data_ptr()

        actual_global = _all_gather_stack(replayed[-1])
        expected_global = _global_route_reference(
            hidden_global,
            ids_global,
            weights_global,
            raw_global,
        )
        dist.barrier()
    finally:
        # Drop graph-owned output references before releasing the symmetric
        # workspace they read.  Every successful iteration synchronized all
        # ranks, so teardown cannot race an outstanding peer launch.
        torch.cuda.synchronize()
        captured_output = None
        graph = None
        if layer is not None:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)

    assert input_ptrs_after == input_ptrs
    assert captured_output_ptr_after == captured_output_ptr
    for replay in replayed:
        assert torch.equal(replay, eager), (
            f"rank {rank}: fused graph replay diverged from eager output"
        )
    _assert_matches_reference(actual_global, expected_global, launch=0)
    print(
        f"rank {rank}: production SM90 MXFP4 fused outer graph completed "
        f"{GRAPH_REPLAYS} stable lockstep replays across {world_size} ranks"
    )


def _tiny_quantize_reference(x, *, fc2=False):
    """Independent FP64 scale calculation, rounded only at FP32 boundaries."""
    import torch

    x = x.float()
    amax = x.abs().amax(-1, keepdim=True)
    old_scale = (amax / 448).clamp_min(1e-30)
    tiny = (amax > 0) & (amax < 448e-30)
    multiplier = (
        448 / amax.double().clamp_min(448 / torch.finfo(torch.float32).max)
    ).float()
    scale = (1 / multiplier.double()).float()
    normal = x * old_scale.reciprocal() if fc2 else x / old_scale
    payload = torch.where(tiny, x * multiplier, normal).to(torch.float8_e4m3fn)
    return payload, torch.where(tiny, scale, old_scale)


def _tiny_full_output_reference(hidden, ids, scores, raw, *, require_all_experts=True):
    import torch

    from tests.moe_ep._sm90_fp8_wgmma_reference import fma_add

    world, tokens, topk = ids.shape
    w1, common1, w2, common2 = _prepare_global_humming_operands(raw)
    x, input_scale = _tiny_quantize_reference(hidden)
    terms = torch.zeros(
        (world, tokens, topk, HIDDEN), dtype=torch.bfloat16, device=hidden.device
    )
    for expert in range(world * LOCAL_EXPERTS):
        routes = (ids == expert).nonzero(as_tuple=False)
        if require_all_experts:
            assert routes.numel() > 0
        rank, token, slot = routes.unbind(1)
        owner, local = divmod(expert, LOCAL_EXPERTS)
        accum = _fast_fp8_mm(x[rank, token], w1[owner, local].T)
        fc1 = accum * input_scale[rank, token] * common1[owner, local]
        paired = fc1.reshape(-1, INTERMEDIATE // 8, 2, 8)
        gate = paired[:, :, 0].clamp(max=GATE_UP_CLAMP)
        up = paired[:, :, 1].clamp(min=-GATE_UP_CLAMP, max=GATE_UP_CLAMP)
        activation = _swiglu_sm90_formula(gate, up).reshape(-1, INTERMEDIATE)
        activation = activation * scores[rank, token, slot, None]
        activation, scales = _tiny_quantize_reference(
            activation.reshape(-1, INTERMEDIATE // 64, 64), fc2=True
        )
        output = torch.zeros((routes.shape[0], HIDDEN), device=hidden.device)
        for group in range(INTERMEDIATE // 64):
            partial = _fast_fp8_mm(
                activation[:, group].contiguous(),
                w2[owner, local, :, group * 64 : (group + 1) * 64].T,
            )
            output = fma_add(output, partial, scales[:, group])
        terms[rank, token, slot] = (output * common2[owner, local]).bfloat16()
    return terms.float().sum(2)


@pytest.mark.gpu_4
@pytest.mark.arch_hopper
@pytest.mark.parametrize(
    "amplitude,fc1_exponent_shift",
    [(1e-15, 0), (1e-18, 0), (1e-19, 0), (1e-35, 55)],
)
def test_fused_mxfp4_tiny_full_output(monkeypatch, amplitude, fc1_exponent_shift):
    """No absolute tolerance that could accidentally accept all-zero output."""
    import sys

    import torch

    from flashinfer.moe_ep import PrequantizedMoEWeights

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 fused MXFP4 safe quantization requires Hopper")
    test_module = sys.modules[__name__]

    monkeypatch.setattr(test_module, "HIDDEN", 256)
    monkeypatch.setattr(test_module, "INTERMEDIATE", 256)
    original_inputs = _make_tokens_and_routes
    original_weights = _make_raw_weights

    def inputs(*args, **kwargs):
        hidden, ids, scores = original_inputs(*args, **kwargs)
        hidden = (hidden.float() * amplitude).bfloat16()
        if fc1_exponent_shift:
            assert hidden.float().abs().max() < 448e-30
        return hidden, ids, scores

    def weights(*args, **kwargs):
        raw = original_weights(*args, **kwargs)
        exponent = raw.w13_scale.int() + fc1_exponent_shift
        assert (exponent < 255).all()
        return PrequantizedMoEWeights(
            w13=raw.w13,
            w2=raw.w2,
            w13_scale=exponent.to(raw.w13_scale.dtype),
            w2_scale=raw.w2_scale,
        )

    def check(actual, expected, *, launch):
        assert torch.isfinite(actual).all() and torch.isfinite(expected).all()
        assert expected.count_nonzero() > 0 and actual.count_nonzero() > 0
        relative_l2 = (
            actual.double() - expected.double()
        ).norm() / expected.double().norm()
        assert relative_l2 < 0.02, (amplitude, launch, relative_l2.item())
        print(
            "MXFP4_TINY_FULL_OUTPUT_PASS",
            amplitude,
            launch,
            relative_l2.item(),
            flush=True,
        )

    monkeypatch.setattr(test_module, "_make_tokens_and_routes", inputs)
    monkeypatch.setattr(test_module, "_make_raw_weights", weights)
    monkeypatch.setattr(
        test_module, "_global_route_reference", _tiny_full_output_reference
    )
    monkeypatch.setattr(test_module, "_assert_matches_reference", check)
    tactic = dict(_complete_fused_graph_tactic(), mma_tiler_mnk=(256, 64, 256))
    test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse(
        "cross_rank", tactic=tactic
    )
