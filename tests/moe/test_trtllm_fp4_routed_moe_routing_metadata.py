"""trtllm_fp4_block_scale_routed_moe with caller-populated routing metadata.

The metadata is populated on a second stream, as a caller overlapping it with other work would,
and the routed call then runs only the expert body; it must match the ordinary routed call.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
    TrtllmFp4Config,
    populate_trtllm_moe_routing_metadata_,
    trtllm_fp4_block_scale_routed_moe,
    trtllm_moe_allocate_routing_metadata_multi_tile,
)
from flashinfer.tllm_enums import RoutingInputMode, RoutingMethodType
from flashinfer.utils import get_compute_capability

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 1024, 1024, 256, 8
# Local experts [LOCAL, 2 * LOCAL) of NUM_EXPERTS.
LOCAL = 32
TILES = (8, 16, 32, 64, 128, 192, 256)


@pytest.fixture(scope="module")
def experts():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    major, minor = get_compute_capability(torch.device("cuda"))
    if not TrtllmFp4Config.supported(major * 10 + minor):
        pytest.skip("TRT-LLM FP4 routed MoE is not supported on this GPU")
    torch.manual_seed(0)
    quant = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
    w1 = torch.randn(LOCAL, 2 * INTERMEDIATE, HIDDEN, device="cuda") * 0.02
    w2 = torch.randn(LOCAL, HIDDEN, INTERMEDIATE, device="cuda") * 0.02
    view = TrtllmFp4Config.prepare_weights(
        w1.bfloat16(),
        w2.bfloat16(),
        quant=quant,
        num_local_experts=LOCAL,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        device=torch.device("cuda"),
    )
    return quant, view


def _activations(quant, num_tokens: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, device="cuda", generator=g) * 2
    return TrtllmFp4Config.prepare_activations(x.bfloat16(), quant=quant)


def _routing(num_tokens: int, seed: int, packed: bool, no_local: bool = False):
    """Distinct experts per token; packed int32 or an (ids, fp32 weights) pair."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    pool = torch.arange(NUM_EXPERTS, device="cuda")
    if no_local:
        pool = pool[(pool < LOCAL) | (pool >= 2 * LOCAL)]
    ids = torch.stack(
        [
            pool[torch.randperm(pool.numel(), device="cuda", generator=g)[:TOP_K]]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(num_tokens, TOP_K, device="cuda", generator=g)
    if not packed:
        return ids, weights
    w = weights.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    return (ids << 16) | w


def _split_routing(routing):
    """(topk_ids, topk_weights, routing_input_mode) for the metadata API."""
    if isinstance(routing, tuple):
        return routing[0], routing[1], RoutingInputMode.UnpackedPrecomputed
    return routing, None, RoutingInputMode.PackedPrecomputed


def _allocate(routing, tile_n: int):
    ids, weights, mode = _split_routing(routing)
    return trtllm_moe_allocate_routing_metadata_multi_tile(
        ids,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        local_expert_offset=LOCAL,
        num_local_experts=LOCAL,
        tile_ns=(tile_n,),
        routing_input_mode=mode,
        topk_weights=weights,
    )


def _populate_on(stream, metadata, routing):
    """Populate on ``stream`` and make the current stream wait for it."""
    ids, weights, _ = _split_routing(routing)
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        populate_trtllm_moe_routing_metadata_(metadata, ids, weights)
    current.wait_stream(stream)


def _call(view, hidden_states, scale, routing, do_finalize, **overrides):
    kwargs = dict(
        topk_ids=routing,
        routing_bias=None,
        hidden_states=hidden_states,
        hidden_states_scale=scale,
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=view.get("gemm1_alpha"),
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=view.get("output1_scale_scalar"),
        output1_scale_gate_scalar=view.get("output1_scale_gate_scalar"),
        output2_scale_scalar=view.get("output2_scale_scalar"),
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        n_group=0,
        topk_group=0,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=LOCAL,
        local_num_experts=LOCAL,
        routed_scaling_factor=1.0,
        routing_method_type=RoutingMethodType.Renormalize.value,
        do_finalize=do_finalize,
    )
    kwargs.update(overrides)
    return trtllm_fp4_block_scale_routed_moe(**kwargs)


def _result(num_tokens, out, do_finalize):
    """The finalized output, or the weighted per-token sum of the unfinalized one."""
    if do_finalize:
        return out[0]
    gemm2_out, expert_weights, expanded_idx_to_permuted_idx = out[:3]
    idx = expanded_idx_to_permuted_idx.long().view(num_tokens, TOP_K)
    # Invalid slots (-1) gather row 0, which may be uninitialized; zero them explicitly.
    rows = torch.where(
        (idx >= 0).view(num_tokens, TOP_K, 1), gemm2_out.float()[idx.clamp(min=0)], 0.0
    )
    return (rows * expert_weights.float().view(num_tokens, TOP_K, 1)).sum(dim=1)


@pytest.mark.parametrize("packed", [True, False], ids=["packed", "unpacked_fp32"])
@pytest.mark.parametrize("tile_n", TILES)
@pytest.mark.parametrize("num_tokens", [1, 64, 960])
@pytest.mark.parametrize("do_finalize", [False, True])
def test_routing_metadata_matches_ordinary_call(
    experts, num_tokens, tile_n, do_finalize, packed
):
    quant, view = experts
    hidden_states, scale = _activations(quant, num_tokens, seed=num_tokens)
    routing = _routing(num_tokens, seed=num_tokens, packed=packed)
    want = _result(
        num_tokens, _call(view, hidden_states, scale, routing, do_finalize), do_finalize
    )

    metadata = _allocate(routing, tile_n)
    _populate_on(torch.cuda.Stream(), metadata, routing)
    out = _call(
        view,
        hidden_states,
        scale,
        routing,
        do_finalize,
        routing_metadata=metadata.slots[0],
    )
    torch.testing.assert_close(
        _result(num_tokens, out, do_finalize), want, rtol=1e-2, atol=1e-2
    )


@pytest.mark.parametrize("tile_n", [8, 64, 192])
@pytest.mark.parametrize("do_finalize", [False, True])
def test_graph_replay_follows_new_routing_and_activations(experts, tile_n, do_finalize):
    """Each replay repopulates the metadata from the live routing; stale metadata fails."""
    quant, view = experts
    num_tokens = 64
    hidden_states, scale = _activations(quant, num_tokens, seed=0)
    routing = _routing(num_tokens, seed=0, packed=True)
    metadata = _allocate(routing, tile_n)
    side = torch.cuda.Stream()

    def run():
        _populate_on(side, metadata, routing)
        return _call(
            view,
            hidden_states,
            scale,
            routing,
            do_finalize,
            routing_metadata=metadata.slots[0],
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    for seed in (1, 2, 3):
        new_hidden_states, new_scale = _activations(quant, num_tokens, seed=seed)
        hidden_states.copy_(new_hidden_states)
        scale.copy_(new_scale)
        routing.copy_(_routing(num_tokens, seed=seed, packed=True))
        graph.replay()
        torch.cuda.synchronize()
        want = _result(
            num_tokens,
            _call(view, new_hidden_states, new_scale, routing, do_finalize),
            do_finalize,
        )
        torch.testing.assert_close(
            _result(num_tokens, captured, do_finalize), want, rtol=1e-2, atol=1e-2
        )


@pytest.mark.parametrize("do_finalize", [False, True])
def test_no_token_routed_to_local_experts(experts, do_finalize):
    quant, view = experts
    num_tokens = 16
    hidden_states, scale = _activations(quant, num_tokens, seed=7)
    routing = _routing(num_tokens, seed=7, packed=True, no_local=True)
    want = _result(
        num_tokens, _call(view, hidden_states, scale, routing, do_finalize), do_finalize
    )

    metadata = _allocate(routing, 8)
    _populate_on(torch.cuda.Stream(), metadata, routing)
    out = _call(
        view,
        hidden_states,
        scale,
        routing,
        do_finalize,
        routing_metadata=metadata.slots[0],
    )
    got = _result(num_tokens, out, do_finalize)
    assert torch.equal(got, want)
    assert not got.any()


@pytest.mark.parametrize(
    "case", ["gemm1_lora_delta", "fused_shared", "workspace_buffer"]
)
def test_routing_metadata_rejects_unsupported_options(experts, case):
    quant, view = experts
    num_tokens = 4
    hidden_states, scale = _activations(quant, num_tokens, seed=4)
    routing = _routing(num_tokens, seed=4, packed=True)
    metadata = _allocate(routing, 8)
    overrides = {
        "gemm1_lora_delta": dict(
            gemm1_lora_delta=torch.zeros(
                num_tokens, TOP_K, 2 * INTERMEDIATE, device="cuda", dtype=torch.bfloat16
            )
        ),
        # Fused shared experts require DeepSeekV3 routing over experts that are all local.
        "fused_shared": dict(
            num_fused_shared_experts=1,
            num_experts=LOCAL,
            local_expert_offset=0,
            routing_method_type=RoutingMethodType.DeepSeekV3.value,
        ),
        "workspace_buffer": dict(
            workspace_buffer=torch.empty(1 << 20, device="cuda", dtype=torch.uint8)
        ),
    }[case]
    with pytest.raises(ValueError, match="routing_metadata supports neither"):
        _call(
            view,
            hidden_states,
            scale,
            routing,
            False,
            routing_metadata=metadata.slots[0],
            **overrides,
        )
