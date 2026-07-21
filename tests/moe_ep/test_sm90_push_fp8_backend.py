"""Public MoEEpLayer correctness gates for the SM90 push FP8 backend."""

from __future__ import annotations

import os

import pytest
import torch

from ._sm90_push_fp8_reference import dequant_weight_128x128, reference_moe

pytestmark = pytest.mark.usefixtures("isolated_deep_gemm_cache")


def _sm90_cuda_12_8_available() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        from flashinfer.jit.cpp_ext import is_cuda_version_at_least
        from flashinfer.utils import is_sm90a_supported

        return is_cuda_version_at_least("12.8") and is_sm90a_supported(
            torch.device("cuda")
        )
    except Exception:
        return False


_WORLD = int(os.environ.get("WORLD_SIZE", "1"))
requires_sm90 = pytest.mark.skipif(
    not _sm90_cuda_12_8_available() or _WORLD > 1,
    reason="requires one SM90 GPU and CUDA Toolkit 12.8+ outside torchrun",
)
requires_dist = pytest.mark.skipif(
    _WORLD < 2 or not _sm90_cuda_12_8_available(),
    reason="requires torchrun with at least two SM90 GPUs and CUDA Toolkit 12.8+",
)

HIDDEN = 512
INTERMEDIATE = 768
LOCAL_EXPERTS = 4
TOP_K = 2
TOKEN_CAPACITY = 64

_KEEP_ALIVE: list[object] = []


def _make_weights(
    num_experts: int, seed: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(
            num_experts,
            2 * INTERMEDIATE,
            HIDDEN,
            generator=generator,
        )
        * HIDDEN**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(
            num_experts,
            HIDDEN,
            INTERMEDIATE,
            generator=generator,
        )
        * INTERMEDIATE**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def _make_inputs(
    num_tokens: int,
    num_experts: int,
    seed: int,
    device: torch.device,
    *,
    mode: str = "random",
    rank: int = 0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(num_tokens, HIDDEN, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )
    if mode == "hot":
        ids = torch.zeros(num_tokens, TOP_K, dtype=torch.int32)
    else:
        logits = torch.randn(num_tokens, num_experts, generator=generator)
        if mode == "all_remote" and num_experts > LOCAL_EXPERTS:
            local_start = rank * LOCAL_EXPERTS
            logits[:, local_start : local_start + LOCAL_EXPERTS] = float("-inf")
        ids = logits.topk(TOP_K, dim=1).indices.to(torch.int32)
    weights = torch.rand(num_tokens, TOP_K, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return (
        x,
        ids.to(device),
        weights.to(device=device, dtype=torch.float32),
    )


def _build_layer(
    world_size: int,
    rank: int,
    device: torch.device,
    *,
    payload_dtype: str = "fp8",
    combine_dtype: str = "fp8",
    dedup_dispatch: bool = True,
    grouped_combine: bool = True,
    fuse_fc1_epilogue: bool = True,
    capacity_factor: float = 1.0,
    seed: int = 7,
):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEWeightPack,
        Sm90PushFp8MegaMoeConfig,
    )

    total_experts = LOCAL_EXPERTS * world_size
    w13, w2 = _make_weights(total_experts, seed, device)
    local_start = rank * LOCAL_EXPERTS
    local_end = local_start + LOCAL_EXPERTS
    process_group = None
    if world_size > 1:
        import torch.distributed as dist

        process_group = dist.group.WORLD
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world_size,
            rank=rank,
            process_group=process_group,
        ),
        fleet_params=FleetParams(
            num_experts=total_experts,
            max_tokens_per_rank=TOKEN_CAPACITY,
            token_hidden_size=HIDDEN,
        ),
        weights=MoEWeightPack(
            w13=w13[local_start:local_end].contiguous(),
            w2=w2[local_start:local_end].contiguous(),
        ),
        backend=MegaConfig(
            megakernel=Sm90PushFp8MegaMoeConfig(
                intermediate_size=INTERMEDIATE,
                top_k=TOP_K,
                capacity_factor=capacity_factor,
                payload_dtype=payload_dtype,
                combine_dtype=combine_dtype,
                dedup_dispatch=dedup_dispatch,
                grouped_combine=grouped_combine,
                fuse_fc1_epilogue=fuse_fc1_epilogue,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )
    _KEEP_ALIVE.append(layer)
    return layer, w13, w2


def _forward(
    layer,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.moe_ep import MoEEpTensors

    return layer(
        MoEEpTensors(
            hidden_states=x,
            topk_ids=topk_ids,
            topk_weights=topk_weights,
        )
    )


def _dequant_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
) -> torch.Tensor:
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        transform_weights_for_sm90_push,
    )

    w13_fp8, w13_scales, w2_fp8, w2_scales = transform_weights_for_sm90_push(w13, w2)
    w13_dequant = torch.stack(
        [dequant_weight_128x128(w13_fp8[e], w13_scales[e]) for e in range(w13.shape[0])]
    )
    w2_dequant = torch.stack(
        [dequant_weight_128x128(w2_fp8[e], w2_scales[e]) for e in range(w2.shape[0])]
    )
    return reference_moe(
        x,
        w13_dequant,
        w2_dequant,
        topk_ids,
        topk_weights,
    )


def _relative_error(output: torch.Tensor, reference: torch.Tensor) -> float:
    norm = reference.float().square().mean().sqrt().clamp_min(1e-6)
    error = (output.float() - reference.float()).square().mean().sqrt()
    return float(error / norm)


def _cosine(output: torch.Tensor, reference: torch.Tensor) -> float:
    return float(
        torch.nn.functional.cosine_similarity(
            output.float().flatten(), reference.float().flatten(), dim=0
        )
    )


@requires_sm90
@pytest.mark.parametrize(
    "payload_dtype,combine_dtype,dedup_dispatch,grouped_combine,fuse_fc1_epilogue",
    [
        ("bf16", "bf16", False, False, False),
        ("fp8", "bf16", True, False, True),
        ("bf16", "fp8", False, True, True),
        ("fp8", "fp8", True, True, True),
    ],
)
def test_public_ep1_forward_configs(
    payload_dtype: str,
    combine_dtype: str,
    dedup_dispatch: bool,
    grouped_combine: bool,
    fuse_fc1_epilogue: bool,
) -> None:
    from flashinfer.moe_ep import MoEEpMegaLayer

    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(
        1,
        0,
        device,
        payload_dtype=payload_dtype,
        combine_dtype=combine_dtype,
        dedup_dispatch=dedup_dispatch,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    assert isinstance(layer, MoEEpMegaLayer)
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 11, device)
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, w13, w2)
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert _cosine(output, reference) > 0.997
    assert _relative_error(output, reference) < 0.10


@requires_sm90
@pytest.mark.parametrize("case", ["short", "masked", "hot", "empty"])
def test_public_ep1_edge_routes_and_recovery(case: str) -> None:
    device = torch.device("cuda", 0)
    layer, w13, w2 = _build_layer(1, 0, device)
    if case == "empty":
        num_tokens = 0
    elif case == "short":
        num_tokens = 7
    else:
        num_tokens = TOKEN_CAPACITY
    x, ids, weights = _make_inputs(
        num_tokens,
        LOCAL_EXPERTS,
        21,
        device,
        mode="hot" if case == "hot" else "random",
    )
    if case == "masked":
        ids = ids.clone()
        ids[::3, 0] = -1
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    if num_tokens:
        reference = _dequant_reference(x, ids, weights, w13, w2)
        assert _cosine(output, reference) > 0.997

    recovery_x, recovery_ids, recovery_weights = _make_inputs(
        TOKEN_CAPACITY, LOCAL_EXPERTS, 22, device
    )
    recovered = _forward(layer, recovery_x, recovery_ids, recovery_weights)
    torch.cuda.synchronize()
    recovery_reference = _dequant_reference(
        recovery_x, recovery_ids, recovery_weights, w13, w2
    )
    assert _cosine(recovered, recovery_reference) > 0.997


@requires_sm90
def test_public_ep1_forward_validation_and_capacity() -> None:
    from flashinfer.moe_ep import MoEEpConfigError

    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, device)
    x, ids, weights = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 31, device)
    with pytest.raises(MoEEpConfigError, match="bf16"):
        _forward(layer, x.half(), ids, weights)
    with pytest.raises(MoEEpConfigError, match="int32"):
        _forward(layer, x, ids.long(), weights)
    with pytest.raises(MoEEpConfigError, match="float32"):
        _forward(layer, x, ids, weights.half())

    large_x, large_ids, large_weights = _make_inputs(
        TOKEN_CAPACITY + 1, LOCAL_EXPERTS, 32, device
    )
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        _forward(layer, large_x, large_ids, large_weights)


@requires_sm90
def test_public_ep1_outputs_do_not_alias() -> None:
    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, device)
    x1, ids1, weights1 = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 41, device)
    output1 = _forward(layer, x1, ids1, weights1)
    torch.cuda.synchronize()
    snapshot = output1.clone()
    x2, ids2, weights2 = _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 42, device)
    output2 = _forward(layer, x2, ids2, weights2)
    torch.cuda.synchronize()
    assert output1.data_ptr() != output2.data_ptr()
    assert torch.equal(output1, snapshot)
    assert not torch.equal(output1, output2)


@requires_sm90
def test_public_ep1_graph_replay() -> None:
    device = torch.device("cuda", 0)
    layer, _, _ = _build_layer(1, 0, device)
    inputs = [
        _make_inputs(TOKEN_CAPACITY, LOCAL_EXPERTS, 51 + index, device)
        for index in range(2)
    ]
    eager = []
    for x, ids, weights in inputs:
        eager.append(_forward(layer, x, ids, weights).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(inputs[0][0]).copy_(inputs[0][0])
    static_ids = torch.empty_like(inputs[0][1]).copy_(inputs[0][1])
    static_weights = torch.empty_like(inputs[0][2]).copy_(inputs[0][2])
    for _ in range(2):
        side_stream = torch.cuda.Stream()
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _forward(layer, static_x, static_ids, static_weights)
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        static_output = _forward(layer, static_x, static_ids, static_weights)
    replayed = []
    for x, ids, weights in inputs:
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
    assert not torch.equal(replayed[0], replayed[1])
    for index in range(2):
        assert torch.equal(replayed[index], eager[index])


def _dist_setup() -> tuple[int, int]:
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world_size = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


@requires_dist
@pytest.mark.parametrize(
    "payload_dtype,combine_dtype,dedup_dispatch,grouped_combine,fuse_fc1_epilogue,mode",
    [
        ("bf16", "bf16", False, False, False, "random"),
        ("fp8", "fp8", True, True, True, "all_remote"),
    ],
)
def test_public_ep2_forward_configs(
    payload_dtype: str,
    combine_dtype: str,
    dedup_dispatch: bool,
    grouped_combine: bool,
    fuse_fc1_epilogue: bool,
    mode: str,
) -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    layer, w13, w2 = _build_layer(
        world_size,
        rank,
        device,
        payload_dtype=payload_dtype,
        combine_dtype=combine_dtype,
        dedup_dispatch=dedup_dispatch,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    x, ids, weights = _make_inputs(
        TOKEN_CAPACITY,
        total_experts,
        61 + rank,
        device,
        mode=mode,
        rank=rank,
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, w13, w2)
    assert output.shape == x.shape
    assert output.dtype == torch.bfloat16
    assert torch.isfinite(output.float()).all()
    assert _cosine(output, reference) > 0.997
    dist.barrier()


@requires_dist
def test_public_ep2_uneven_empty_and_recovery() -> None:
    import torch.distributed as dist

    rank, world_size = _dist_setup()
    device = torch.device("cuda", rank)
    total_experts = LOCAL_EXPERTS * world_size
    layer, w13, w2 = _build_layer(world_size, rank, device)
    num_tokens = 0 if rank == 1 else max(TOKEN_CAPACITY - 13 * rank, 1)
    x, ids, weights = _make_inputs(
        num_tokens, total_experts, 71 + rank, device, rank=rank
    )
    output = _forward(layer, x, ids, weights)
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    if num_tokens:
        reference = _dequant_reference(x, ids, weights, w13, w2)
        assert _cosine(output, reference) > 0.997

    recovery_x, recovery_ids, recovery_weights = _make_inputs(
        TOKEN_CAPACITY, total_experts, 81 + rank, device, rank=rank
    )
    recovered = _forward(layer, recovery_x, recovery_ids, recovery_weights)
    torch.cuda.synchronize()
    recovery_reference = _dequant_reference(
        recovery_x, recovery_ids, recovery_weights, w13, w2
    )
    assert _cosine(recovered, recovery_reference) > 0.997
    dist.barrier()
