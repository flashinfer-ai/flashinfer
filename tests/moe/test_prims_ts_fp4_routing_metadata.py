"""Prims-TS NVFP4 MoE with caller-populated routing metadata.

The metadata is populated on a second stream, as a caller overlapping it with other work would;
the call then runs only the expert GEMMs and must match the ordinary call bitwise.
"""

import pytest
import torch

from flashinfer.fused_moe import (
    QuantConfig,
    QuantFormat,
    TrtllmFp4Config,
    populate_trtllm_moe_routing_metadata_,
    prims_ts_fp4_block_scale_moe_partitioned,
    prims_ts_fp4_block_scale_routed_moe,
    trtllm_moe_allocate_routing_metadata_multi_tile,
)
from flashinfer.prims_ts.utils import is_prims_ts_available
from flashinfer.tllm_enums import RoutingInputMode, RoutingMethodType
from flashinfer.utils import get_compute_capability

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K = 1024, 1024, 256, 8
# Local experts [LOCAL, 2 * LOCAL) of NUM_EXPERTS.
LOCAL = 32
TILES = (8, 16, 32, 64, 128, 256)
NUM_TOKENS = (8, 64, 512, 2048)


@pytest.fixture(scope="module")
def experts():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    if get_compute_capability(torch.device("cuda"))[0] != 10:
        pytest.skip("Prims-TS MoE requires SM10x")
    if not is_prims_ts_available():
        pytest.skip("Prims-TS dependencies are unavailable")
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


def _routing(
    num_tokens: int, seed: int, num_experts: int = NUM_EXPERTS, top_k: int = TOP_K
) -> torch.Tensor:
    """Distinct experts per token, packed as (id << 16) | bf16 weight."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    ids = torch.stack(
        [
            torch.randperm(num_experts, device="cuda", generator=g)[:top_k]
            for _ in range(num_tokens)
        ]
    ).to(torch.int32)
    weights = torch.rand(num_tokens, top_k, device="cuda", generator=g)
    w = weights.bfloat16().view(torch.int16).to(torch.int32) & 0xFFFF
    return (ids << 16) | w


def _metadata(
    routing: torch.Tensor,
    tile_ns=TILES,
    num_experts: int = NUM_EXPERTS,
    local_expert_offset: int = LOCAL,
    num_local_experts: int = LOCAL,
):
    """Allocate and populate on a side stream; the current stream waits for it."""
    metadata = trtllm_moe_allocate_routing_metadata_multi_tile(
        routing,
        num_experts=num_experts,
        top_k=routing.shape[1],
        local_expert_offset=local_expert_offset,
        num_local_experts=num_local_experts,
        tile_ns=tile_ns,
        routing_input_mode=RoutingInputMode.PackedPrecomputed,
    )
    _populate(metadata, routing)
    return metadata


def _populate(metadata, routing, stream=None):
    stream = stream or torch.cuda.Stream()
    current = torch.cuda.current_stream()
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        populate_trtllm_moe_routing_metadata_(metadata, routing)
    current.wait_stream(stream)


def _common(view, hidden_states, scale, do_finalize):
    return dict(
        hidden_states=hidden_states,
        hidden_states_scale=scale,
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


def _call(view, hidden_states, scale, routing, do_finalize, **overrides):
    return prims_ts_fp4_block_scale_routed_moe(
        topk_ids=routing,
        routing_bias=None,
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=None,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        **_common(view, hidden_states, scale, do_finalize),
        **overrides,
    )


def _result(num_tokens, out, do_finalize, top_k=TOP_K):
    """The finalized output, or the weighted per-token sum of the unfinalized one."""
    if do_finalize:
        return out[0]
    gemm2_out, expert_weights, expanded_idx_to_permuted_idx = out[:3]
    idx = expanded_idx_to_permuted_idx.long().view(num_tokens, top_k)
    # Padded slots (-1) may hold uninitialized rows (NaN * 0 is NaN); zero them explicitly.
    rows = gemm2_out.float()[idx.clamp(min=0)] * expert_weights.float().view(
        num_tokens, top_k, 1
    )
    return torch.where((idx >= 0).view(num_tokens, top_k, 1), rows, 0.0).sum(dim=1)


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("do_finalize", [False, True])
def test_routed_matches_ordinary_call(experts, num_tokens, do_finalize):
    quant, view = experts
    hidden_states, scale = _activations(quant, num_tokens, seed=num_tokens)
    routing = _routing(num_tokens, seed=num_tokens)
    want = _result(
        num_tokens, _call(view, hidden_states, scale, routing, do_finalize), do_finalize
    )
    out = _call(
        view,
        hidden_states,
        scale,
        routing,
        do_finalize,
        routing_metadata=_metadata(routing),
    )
    assert torch.equal(_result(num_tokens, out, do_finalize), want)


@pytest.mark.parametrize("tile_n", [8, 16])
def test_slot_runs_its_tile(experts, tile_n):
    """A single slot runs its own tile, whatever the call's default tile is."""
    quant, view = experts
    num_tokens = 64
    hidden_states, scale = _activations(quant, num_tokens, seed=1)
    routing = _routing(num_tokens, seed=1)
    want = _call(view, hidden_states, scale, routing, True)[0]
    slot = _metadata(routing, tile_ns=(tile_n,)).slots[0]
    out = _call(view, hidden_states, scale, routing, True, routing_metadata=slot)
    torch.testing.assert_close(out[0], want, rtol=1e-2, atol=1e-2)


def test_graph_replay_follows_new_routing(experts):
    """Each replay repopulates the metadata from the live routing."""
    quant, view = experts
    num_tokens = 64
    hidden_states, scale = _activations(quant, num_tokens, seed=0)
    routing = _routing(num_tokens, seed=0)
    metadata = _metadata(routing)
    side = torch.cuda.Stream()

    def run():
        _populate(metadata, routing, side)
        return _call(
            view, hidden_states, scale, routing, True, routing_metadata=metadata
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()

    for seed in (1, 2):
        new_hidden_states, new_scale = _activations(quant, num_tokens, seed=seed)
        hidden_states.copy_(new_hidden_states)
        scale.copy_(new_scale)
        routing.copy_(_routing(num_tokens, seed=seed))
        graph.replay()
        torch.cuda.synchronize()
        want = _call(view, new_hidden_states, new_scale, routing, True)[0]
        assert torch.equal(captured[0], want)


def _partitioned_call(num_tokens: int):
    """The partitioned op on the partition tests' NVFP4 workload, with packed routing."""
    from tests.prims_ts.test_moe_partition_execution import (
        _NUM_EXPERTS,
        _TOP_K,
        _green_partition_streams,
        _prepare_workload,
    )

    workload = _prepare_workload("nvfp4", num_tokens)
    streams, green_resources = _green_partition_streams()
    resources, strategy, shards = workload.partition("tp_n_tp_n", streams)
    kwargs = {
        **workload.kwargs,
        "routing_logits": None,
        "topk_ids": _routing(num_tokens, num_tokens, _NUM_EXPERTS, _TOP_K),
        "topk_weights": None,
        "gemm1_weights": shards.fc1,
        "gemm1_weights_scale": shards.fc1_scale,
        "gemm2_weights": shards.fc2,
        "gemm2_weights_scale": shards.fc2_scale,
        "partition_resources": resources,
        "partition_strategy": strategy,
    }
    for name in ("gemm1_bias", "gemm2_bias", "num_fused_shared_experts"):
        kwargs.pop(name, None)

    def call(do_finalize, **overrides):
        return prims_ts_fp4_block_scale_moe_partitioned(
            **{**kwargs, **overrides}, do_finalize=do_finalize
        )

    # Keep the green-context resources alive with the call.
    call.green_resources = green_resources
    call.num_experts, call.top_k = _NUM_EXPERTS, _TOP_K
    return call, kwargs["topk_ids"]


# The fused populate takes up to 1024 tokens with the workload's 8 experts.
@pytest.mark.parametrize("num_tokens", [8, 64, 512, 1024])
@pytest.mark.parametrize("do_finalize", [False, True])
def test_partitioned_matches_ordinary_call(num_tokens, do_finalize):
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    call, routing = _partitioned_call(num_tokens)
    top_k = call.top_k
    want = _result(num_tokens, call(do_finalize), do_finalize, top_k).clone()
    metadata = _metadata(
        routing,
        num_experts=call.num_experts,
        local_expert_offset=0,
        num_local_experts=call.num_experts,
    )
    out = call(do_finalize, routing_metadata=metadata)
    assert torch.equal(_result(num_tokens, out, do_finalize, top_k), want)


def test_partitioned_rejects_logits():
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    call, routing = _partitioned_call(8)
    metadata = _metadata(
        routing,
        num_experts=call.num_experts,
        local_expert_offset=0,
        num_local_experts=call.num_experts,
    )
    logits = torch.randn(8, call.num_experts, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="requires precomputed topk_ids"):
        call(True, routing_logits=logits, topk_ids=None, routing_metadata=metadata)
