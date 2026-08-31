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
reuse, and reuse of the same symmetric workspace.
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


def _make_tokens_and_routes(rank: int, world_size: int, *, launch: int):
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
    topk_ids = owner * LOCAL_EXPERTS + local_expert

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
        interleave=False,
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
    """Hopper 1xacc FP8 matmul, padding only the routed-token dimension."""
    import torch

    assert a.dtype == b.dtype == torch.float8_e4m3fn
    assert a.ndim == b.ndim == 2 and a.shape[1] == b.shape[0]
    rows = a.shape[0]
    padded_rows = (rows + 15) // 16 * 16
    if padded_rows != rows:
        padded = torch.zeros((padded_rows, a.shape[1]), dtype=a.dtype, device=a.device)
        padded[:rows].view(torch.uint8).copy_(a.view(torch.uint8))
        a = padded
    one = torch.ones((), dtype=torch.float32, device=a.device)
    result = torch._scaled_mm(
        a.contiguous(),
        b,
        one,
        one,
        out_dtype=torch.float32,
        use_fast_accum=True,
    )
    return result[:rows]


def _swiglu_sm90_formula(gate, up):
    """Independent bit-match of the SM90 exp2/reciprocal SwiGLU formula."""
    from tests.moe_ep._sm90_swiglu_reference import swiglu_sm90_reference

    return swiglu_sm90_reference(gate, up)


def _global_route_reference(hidden, topk_ids, topk_weights, raw_global):
    """Compute all ranks' fused output from raw operands and global routing."""
    import torch

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
        assert routed.numel() > 0, f"global expert {global_expert} was not exercised"
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
        swiglu = _swiglu_sm90_formula(paired[:, :, 0], paired[:, :, 1]).reshape(
            -1, INTERMEDIATE
        )
        swiglu.mul_(topk_weights[source_rank, source_token, source_slot].unsqueeze(1))

        grouped = swiglu.reshape(-1, INTERMEDIATE // K64, K64)
        fc2_scale = (grouped.abs().amax(dim=2, keepdim=True) / E4M3_MAX).clamp_min(
            1.0e-30
        )
        fc2_input = (grouped / fc2_scale).to(torch.float8_e4m3fn)

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
            fc2_accum.add_(partial * fc2_scale[:, group])
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


@pytest.mark.gpu_2
@pytest.mark.arch_hopper
def test_moe_ep_sm90_pull_mxfp4_mega_multirank_raw_oracle_and_workspace_reuse():
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
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=world_size,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(1, 1, 1),
        load_balance_mode="static",
        token_back_mode="epi_warps",
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
            _make_tokens_and_routes(rank, world_size, launch=launch)
            for launch in range(2)
        ]
        hidden_global = [_all_gather_stack(item[0]) for item in launches]
        ids_global = [_all_gather_stack(item[1]) for item in launches]
        weights_global = [_all_gather_stack(item[2]) for item in launches]
        for ids in ids_global:
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
        second = layer.forward(tensors(launches[1])).clone()
        assert layer._workspace is workspace
        second_repeat = layer.forward(tensors(launches[1])).clone()
        assert layer._workspace is workspace
        torch.cuda.synchronize()
        torch.testing.assert_close(second_repeat, second, atol=0.0, rtol=0.0)

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
            )
            for launch in range(2)
        ]
        for launch, (actual, expected) in enumerate(
            zip(actual_global, expected_global, strict=True)
        ):
            _assert_matches_reference(actual, expected, launch=launch)
        # Expert-specific weights and rank-specific activations must prevent a
        # rank-0-only or wrong-owner implementation from passing accidentally.
        if world_size > 1:
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
