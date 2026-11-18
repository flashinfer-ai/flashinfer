"""Exercise opt-in NCCL scale forwarding and NVFP4 combine."""

import pytest
import torch
import torch.distributed as dist

from flashinfer.moe_ep import (
    BootstrapConfig,
    CombineInputParams,
    DispatchInputParams,
    EpAlgorithm,
    EpLayout,
    FleetAlgoKnobNcclEpConfig,
    FleetParams,
    HandleAlgoKnobTopKWeights,
    HandleAlgoKnobUserStream,
    HandleParams,
    create_fleet,
)
from tests.moe_ep.test_nccl_ep_release_multirank import (
    _assert_collective,
    _assert_collective_close,
    _assert_receive_ownership,
    _require_single_lsa_team,
    nccl_lsa_teams as nccl_lsa_teams,
    released_runtime as released_runtime,
)

pytestmark = [pytest.mark.nvep, pytest.mark.gpu_4]


@pytest.mark.parametrize(
    "algorithm,layout,flow,windows,eager",
    [
        (EpAlgorithm.LOW_LATENCY, EpLayout.EXPERT_MAJOR, None, False, False),
        (EpAlgorithm.LOW_LATENCY, EpLayout.RANK_MAJOR, None, False, False),
        (EpAlgorithm.LOW_LATENCY, EpLayout.RANK_MAJOR, None, True, False),
        (EpAlgorithm.HIGH_THROUGHPUT, EpLayout.EXPERT_MAJOR, "push_pull", False, False),
        (EpAlgorithm.HIGH_THROUGHPUT, EpLayout.EXPERT_MAJOR, "push_pull", False, True),
        (EpAlgorithm.HIGH_THROUGHPUT, EpLayout.EXPERT_MAJOR, "pull_push", False, False),
        (EpAlgorithm.HIGH_THROUGHPUT, EpLayout.EXPERT_MAJOR, "pull_push", False, True),
    ],
)
def test_forwarded_scales_graph_and_empty_receivers(
    released_runtime,  # noqa: F811 - shared pytest fixture
    algorithm,
    layout,
    flow,
    windows,
    eager,
    nccl_lsa_teams,  # noqa: F811 - shared pytest fixture
):
    rank, world = released_runtime
    if flow == "pull_push":
        _require_single_lsa_team(nccl_lsa_teams)
    tokens, hidden, capacity = 4, 2048, 8
    fleet = create_fleet(
        BootstrapConfig(
            rank=rank, world_size=world, stream=torch.cuda.current_stream().cuda_stream
        ),
        FleetParams(
            num_experts=2 * world,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
            algorithm=algorithm,
            layout=layout,
        ),
        [
            FleetAlgoKnobNcclEpConfig(
                num_topk=2,
                ht_flow=flow,
                dispatch_quantization="forward",
                dispatch_zero_copy=windows,
                eager_recv=eager,
            )
        ],
        backend="nccl_ep",
    )
    row = torch.arange(tokens, device="cuda", dtype=torch.int64)
    own = row % 2 + rank * 2
    peer = row % 2 + ((rank + 1) % world) * 2
    ids = torch.stack((own, peer), dim=1)
    weights = torch.tensor([0.25, 0.75], device="cuda").expand(tokens, 2).contiguous()
    values = torch.full(
        (tokens, hidden), float(rank + 1), device="cuda", dtype=torch.bfloat16
    ).to(torch.float8_e4m3fn)
    scales = (
        ((torch.arange(16, device="cuda", dtype=torch.float32) + 1) * (rank + 1))
        .expand(tokens, -1)
        .contiguous()
    )
    out = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    handle = fleet.create_handle(
        HandleParams(ids),
        [
            HandleAlgoKnobTopKWeights(weights),
            HandleAlgoKnobUserStream(torch.cuda.current_stream().cuda_stream),
        ],
    )
    ht_indices = None
    if algorithm is EpAlgorithm.HIGH_THROUGHPUT and not eager:
        ht_indices, _ = handle._native_receive(
            "indices", (world * capacity, ids.shape[1]), torch.int64, values.device
        )

    received = None

    def step():
        nonlocal received
        if ht_indices is not None:
            # HT leaves rows past its live prefix untouched. Count fresh writes.
            ht_indices.fill_(-1)
        handle.update(HandleParams(ids))
        received = handle.dispatch(DispatchInputParams(x=[values, scales]))
        local = (
            received.expert_tensors.float()
            * received.expert_scales.repeat_interleave(
                hidden // scales.shape[1], dim=-1
            )
        )
        if received.recv_topk_idx is not None:
            is_local = (received.recv_topk_idx >= 0) & (received.recv_topk_idx < 2)
            gain = received.recv_topk_weights.masked_fill(~is_local, 0).sum(-1)
            local *= gain.reshape(*local.shape[:-1], 1)
        return handle.combine(
            CombineInputParams(
                x=[local.to(torch.bfloat16)], out=out, out_is_stable=True
            )
        ).x

    graph = None
    try:
        step()
        step()
        torch.cuda.synchronize()
        dist.barrier()
        if not eager:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
                result = step()
            with pytest.raises(ValueError, match="geometry changed"):
                handle.dispatch(DispatchInputParams(x=[values, scales.repeat(1, 2)]))
        for phase in range(3):
            if phase == 1:
                # Only rank zero owns these experts; every other receiver empties.
                ids[:, 0].fill_(0)
                ids[:, 1].fill_(1)
                scales.mul_(0.5)
            elif phase == 2:
                ids[:, 0].copy_(own)
                ids[:, 1].copy_(peer)
                values.copy_(torch.full_like(out, float(rank + 2)))
                scales.mul_(3)
            if eager:
                result = step()
            else:
                graph.replay()
            torch.cuda.synchronize()
            expected = (
                values.float()
                * scales.repeat_interleave(hidden // scales.shape[1], dim=-1)
            ).to(torch.bfloat16)
            _assert_collective_close(result, expected)
            _assert_receive_ownership(
                received,
                rank,
                world,
                tokens,
                concentrated=phase == 1,
                expert_major=algorithm is EpAlgorithm.LOW_LATENCY
                and layout is EpLayout.EXPERT_MAJOR,
            )
            if eager and phase == 1:
                _assert_collective(
                    rank == 0 or received.expert_tensors.numel() == hidden,
                    "empty eager receiver needs one inactive row",
                )
        _assert_collective(
            len(fleet._dispatch_windows) == (2 if windows else 0),
            "unexpected forwarded dispatch-window count",
        )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        handle.destroy()
        fleet.destroy()
        _assert_collective(
            not fleet._dispatch_windows, "dispatch windows remain after teardown"
        )
        dist.barrier()


def _nvfp4_reference(handle, rank):
    """Quantize each expert partial independently, then restore token order."""
    from flashinfer.quantization.fp4_quantization import fp4_quantize

    expert = handle._combine_inputs.tokens._buffer
    world, capacity, hidden = expert.shape
    _assert_collective(
        handle._recv_count_t.eq(capacity).all().item(),
        "reference requires full receive rows",
    )
    rows = expert.reshape(-1, hidden)
    amax = rows.abs().amax(-1, keepdim=True).float()
    encode_scale = torch.where(amax > 0, 2688.0 / amax, 0.0)
    _assert_collective(
        torch.equal(handle._combine_scales.reshape(-1, 1), encode_scale),
        "NVFP4 encode scales differ from the reference",
    )
    # Supply the same encode scale to the independent quantizer. Deriving it
    # through a decode-scale reciprocal can change FP4 rounding at ties.
    encoded = [
        fp4_quantize(
            rows[i : i + 1],
            encode_scale[i],
            is_sf_swizzled_layout=False,
            backend="cuda",
            nvfp4_4over6=None,
        )
        for i in range(rows.shape[0])
    ]
    packed = torch.cat([item[0] for item in encoded])
    scales = torch.cat([item[1] for item in encoded])
    row_scale = torch.where(encode_scale > 0, 1.0 / encode_scale, 0.0)
    packed = packed.view(torch.uint8).reshape(-1, hidden // 2)
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, hidden)
    levels = torch.tensor(
        [
            0.0,
            0.5,
            1.0,
            1.5,
            2.0,
            3.0,
            4.0,
            6.0,
            -0.0,
            -0.5,
            -1.0,
            -1.5,
            -2.0,
            -3.0,
            -4.0,
            -6.0,
        ],
        device=expert.device,
    )
    block_scale = scales.view(torch.float8_e4m3fn).float().reshape(-1, hidden // 16)
    decoded = (
        levels[codes.long()]
        * block_scale.repeat_interleave(16, dim=-1)
        * row_scale.reshape(-1, 1)
    ).reshape(world, capacity, hidden)
    original = handle._dispatch_inputs.tokens._buffer.view(torch.uint8)
    sent = [torch.empty_like(original) for _ in range(world)]
    sent_ids = [torch.empty_like(handle._topk_idx) for _ in range(world)]
    dist.all_gather(sent, original)
    dist.all_gather(sent_ids, handle._topk_idx)
    received = handle._dispatch_output_t.view(torch.uint8)
    ordered = torch.empty_like(decoded)
    # This controlled input has unique rows. Validate the full row after matching
    # its prefix; LL atomic receive-slot assignment need not preserve row order.
    for source, rows in enumerate(sent):
        matches = (received[source, :, None, :32] == rows[None, :, :32]).all(-1)
        _assert_collective(
            matches.sum(1).eq(1).all().item(), "reference rows must be unique"
        )
        mapping = matches.to(torch.int32).argmax(1)
        _assert_collective(
            mapping.unique().numel() == capacity,
            "receive-row mapping is not a permutation",
        )
        _assert_collective(
            torch.equal(received[source], rows[mapping]), "received token bytes differ"
        )
        expected_ids = sent_ids[source][mapping]
        expected_ids = torch.where(
            expected_ids // handle._num_local_experts == rank,
            expected_ids % handle._num_local_experts,
            -1,
        )
        actual_ids = handle._dispatch_out_idx.reshape(world, capacity, -1)[source]
        _assert_collective(
            torch.equal(actual_ids, expected_ids),
            "received expert IDs do not match current routes",
        )
        ordered[source].index_copy_(0, mapping, decoded[source])
    dist.all_reduce(ordered)
    return ordered[rank].to(torch.bfloat16)


@pytest.mark.parametrize("hidden", [2048, 5120])
@pytest.mark.parametrize("forward", [False, True])
@pytest.mark.parametrize("windows", [False, True])
def test_nvfp4_combine_graph(
    released_runtime,  # noqa: F811 - shared pytest fixture
    hidden,
    forward,
    windows,
):
    rank, world = released_runtime
    supported = torch.tensor(
        int(torch.cuda.get_device_capability()[0] >= 10), device="cuda"
    )
    dist.all_reduce(supported, op=dist.ReduceOp.MIN)
    if not supported.item():
        pytest.skip("NVFP4 reference quantizer requires SM100+ on every rank")
    tokens = 8
    fleet = create_fleet(
        BootstrapConfig(
            rank=rank, world_size=world, stream=torch.cuda.current_stream().cuda_stream
        ),
        FleetParams(
            num_experts=2 * world,
            max_tokens_per_rank=tokens,
            token_hidden_size=hidden,
            algorithm=EpAlgorithm.LOW_LATENCY,
            layout=EpLayout.RANK_MAJOR,
        ),
        [
            FleetAlgoKnobNcclEpConfig(
                num_topk=world,
                dispatch_zero_copy=windows,
                dispatch_quantization="forward" if forward else "none",
                combine_quantization="nvfp4",
            )
        ],
        backend="nccl_ep",
    )
    generator = torch.Generator(device="cuda").manual_seed(4100 + rank)
    values = torch.randn(
        tokens, hidden, device="cuda", dtype=torch.bfloat16, generator=generator
    )
    values[0].zero_()
    if forward:
        values = values.to(torch.float8_e4m3fn)
    scales = torch.full((tokens, 16), 1.0 + rank / 4, device="cuda")
    row = torch.arange(tokens, device="cuda", dtype=torch.int64)
    ids = (
        row[:, None] % 2 + 2 * torch.arange(world, device="cuda")[None, :]
    ).contiguous()
    weights = torch.full((tokens, world), 1.0 / world, device="cuda")
    out = torch.empty(tokens, hidden, dtype=torch.bfloat16, device="cuda")
    handle = fleet.create_handle(
        HandleParams(ids),
        [
            HandleAlgoKnobTopKWeights(weights),
            HandleAlgoKnobUserStream(torch.cuda.current_stream().cuda_stream),
        ],
    )

    def step():
        handle.update(HandleParams(ids))
        payload = [values, scales] if forward else [values]
        received = handle.dispatch(DispatchInputParams(x=payload))
        local = received.expert_tensors.float()
        if forward:
            local *= received.expert_scales[..., :1]
        active = (received.recv_topk_idx >= 0) & (received.recv_topk_idx < 2)
        gain = received.recv_topk_weights.masked_fill(~active, 0).sum(-1)
        local *= gain.reshape(*local.shape[:-1], 1)
        return handle.combine(
            CombineInputParams(
                x=[local.to(torch.bfloat16)], out=out, out_is_stable=True
            )
        ).x

    graph = None
    try:
        step()
        step()
        torch.cuda.synchronize()
        dist.barrier()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=torch.cuda.Stream()):
            result = step()
        for phase in range(2):
            if phase:
                values.copy_((values.float() * 0.5).to(values.dtype))
                weights.mul_(0.75)
                ids.copy_((ids + 2) % (2 * world))
            graph.replay()
            torch.cuda.synchronize()
            expected = _nvfp4_reference(handle, rank)
            relative = torch.linalg.vector_norm(
                result.float() - expected.float()
            ) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-6)
            dist.all_reduce(relative, op=dist.ReduceOp.MAX)
            _assert_collective(
                torch.isfinite(result).all().item(), "NVFP4 output is not finite"
            )
            _assert_collective(
                relative.item() < 0.02, f"NVFP4 format error: {relative.item()}"
            )
            _assert_collective(result[0].eq(0).all().item(), "NVFP4 zero row changed")
        before = torch.cuda.memory_allocated()
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()
        _assert_collective(
            torch.cuda.memory_allocated() == before, "graph replay retained allocations"
        )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        handle.destroy()
        fleet.destroy()
        _assert_collective(
            not fleet._dispatch_windows, "dispatch windows remain after teardown"
        )
        dist.barrier()
