"""Correctness and protocol gates for the SM90 push FP8 kernel package."""

from __future__ import annotations

import os
import random
import subprocess
import sys

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
    num_experts: int,
    seed: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    w13 = (
        torch.randn(num_experts, 2 * intermediate, hidden, generator=generator)
        * hidden**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    w2 = (
        torch.randn(num_experts, hidden, intermediate, generator=generator)
        * intermediate**-0.5
    ).to(device=device, dtype=torch.bfloat16)
    return w13, w2


def _make_x(
    num_tokens: int,
    seed: int,
    device: torch.device,
    *,
    hidden: int = HIDDEN,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return torch.randn(num_tokens, hidden, generator=generator).to(
        device=device, dtype=torch.bfloat16
    )


def _make_routing(
    num_tokens: int,
    num_experts: int,
    top_k: int,
    seed: int,
    device: torch.device,
    *,
    mode: str = "random",
    rank: int = 0,
    num_local_experts: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    if mode == "hot":
        ids = torch.zeros(num_tokens, top_k, dtype=torch.int32)
    else:
        logits = torch.randn(num_tokens, num_experts, generator=generator)
        if mode == "all_remote" and 0 < num_local_experts < num_experts:
            local_start = rank * num_local_experts
            logits[:, local_start : local_start + num_local_experts] = float("-inf")
        ids = logits.topk(top_k, dim=1).indices.to(torch.int32)
    weights = torch.rand(num_tokens, top_k, generator=generator) + 0.1
    weights = weights / weights.sum(dim=1, keepdim=True)
    return ids.to(device), weights.to(device=device, dtype=torch.float32)


def _build(
    *,
    payload_dtype: str = "fp8",
    combine_dtype: str = "fp8",
    fuse_act: bool = True,
    capacity_factor: float = 1.0,
    device_index: int = 0,
    ep_size: int = 1,
    rank: int = 0,
    comm_backend=None,
    total_experts: int = LOCAL_EXPERTS,
    token_capacity: int = TOKEN_CAPACITY,
    dedup_dispatch: bool = False,
    grouped_combine: bool = False,
    fuse_fc1_epilogue: bool = False,
    top_k: int = TOP_K,
    hidden: int = HIDDEN,
    intermediate: int = INTERMEDIATE,
):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushMoERunner,
        Sm90PushPayload,
        Sm90PushPipe,
        make_sm90_push_weights,
    )

    assert total_experts % ep_size == 0
    num_local_experts = total_experts // ep_size
    pipe = Sm90PushPipe(
        ep_size=ep_size,
        rank=rank,
        num_local_experts=num_local_experts,
        hidden_size=hidden,
        top_k=top_k,
        token_capacity=token_capacity,
        device_index=device_index,
        config=Sm90PushConfig(
            payload_dtype=Sm90PushPayload(payload_dtype),
            combine_dtype=Sm90PushCombine(combine_dtype),
            fuse_act=fuse_act,
            capacity_factor=capacity_factor,
            dedup_dispatch=dedup_dispatch,
            grouped_combine=grouped_combine,
            fuse_fc1_epilogue=fuse_fc1_epilogue,
        ),
        comm_backend=comm_backend,
    )
    device = torch.device("cuda", device_index)
    w13, w2 = _make_weights(
        total_experts,
        seed=7,
        device=device,
        hidden=hidden,
        intermediate=intermediate,
    )
    local_start = rank * num_local_experts
    local_end = local_start + num_local_experts
    transformed = make_sm90_push_weights(
        w13[local_start:local_end],
        w2[local_start:local_end],
        interleave_gate_up=fuse_fc1_epilogue,
    )
    runner = Sm90PushMoERunner(pipe, transformed)
    _KEEP_ALIVE.extend((pipe, runner))
    return pipe, runner, (w13, w2), transformed


def _run(
    runner,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run the two-phase API and require direct writes to the supplied output."""
    if output is None:
        output = torch.empty(
            x.shape[0],
            runner.pipe.H,
            dtype=runner.pipe.out_dtype,
            device=x.device,
        )
    runner.stage_inputs(x, topk_ids, topk_weights, output=output)
    result = runner.compute(output=output)
    assert result is output
    return output


def _weight_tuple(weights) -> tuple[torch.Tensor, ...]:
    return weights.w13_fp8, weights.w13_sf, weights.w2_fp8, weights.w2_sf


def _dequant_reference(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    transformed,
) -> torch.Tensor:
    w13_fp8, w13_scales, w2_fp8, w2_scales = _weight_tuple(transformed)
    if transformed.w13_interleaved:
        intermediate_blocks = w13_fp8.shape[1] // 256
        w13_fp8 = (
            w13_fp8.reshape(
                w13_fp8.shape[0],
                intermediate_blocks,
                2,
                128,
                w13_fp8.shape[2],
            )
            .transpose(1, 2)
            .reshape_as(w13_fp8)
            .contiguous()
        )
        w13_scales = (
            w13_scales.reshape(
                w13_scales.shape[0],
                intermediate_blocks,
                2,
                w13_scales.shape[2],
            )
            .transpose(1, 2)
            .reshape_as(w13_scales)
            .contiguous()
        )
    w13 = torch.stack(
        [
            dequant_weight_128x128(w13_fp8[e], w13_scales[e])
            for e in range(w13_fp8.shape[0])
        ]
    )
    w2 = torch.stack(
        [
            dequant_weight_128x128(w2_fp8[e], w2_scales[e])
            for e in range(w2_fp8.shape[0])
        ]
    )
    return reference_moe(x, w13, w2, topk_ids, topk_weights)


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


def _assert_device_trap(result: subprocess.CompletedProcess[str], marker: str) -> None:
    combined = result.stdout + result.stderr
    assert result.returncode != 0, (
        f"expected a device trap, process passed: {combined[-1500:]!r}"
    )
    assert "UNEXPECTED-SURVIVAL" not in result.stdout, combined[-1500:]
    assert marker in combined, f"trap marker {marker!r} missing: {combined[-1500:]!r}"
    for bad_marker in ("ImportError", "ModuleNotFoundError"):
        assert bad_marker not in combined, combined[-1500:]


@requires_sm90
def test_constructor_validation() -> None:
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        Sm90PushCombine,
        Sm90PushConfig,
        Sm90PushPipe,
    )

    def build(**overrides):
        arguments = {
            "ep_size": 1,
            "rank": 0,
            "num_local_experts": LOCAL_EXPERTS,
            "hidden_size": HIDDEN,
            "top_k": TOP_K,
            "token_capacity": TOKEN_CAPACITY,
            "device_index": 0,
        }
        arguments.update(overrides)
        return Sm90PushPipe(**arguments)

    with pytest.raises(ValueError, match="ep_size"):
        build(ep_size=0)
    with pytest.raises(RuntimeError, match="hidden_size"):
        build(hidden_size=500)
    with pytest.raises(RuntimeError, match="top_k"):
        build(top_k=3)
    with pytest.raises(RuntimeError, match="capacity_factor"):
        build(config=Sm90PushConfig(capacity_factor=0.0))
    with pytest.raises(RuntimeError, match="grouped_combine"):
        build(
            config=Sm90PushConfig(
                grouped_combine=True,
                combine_dtype=Sm90PushCombine.BF16,
            )
        )


def test_weight_transform_validation() -> None:
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        transform_weights_for_sm90_push,
    )

    w13 = torch.zeros(2, 2 * 256, 256, dtype=torch.bfloat16)
    w2 = torch.zeros(2, 256, 256, dtype=torch.bfloat16)
    for weight_format in ("int4", "mxfp8", "nvfp4"):
        with pytest.raises(ValueError, match="weight_format"):
            transform_weights_for_sm90_push(
                w13,
                w2,
                weight_format=weight_format,
            )
    with pytest.raises(ValueError, match="BF16"):
        transform_weights_for_sm90_push(w13.float(), w2)
    with pytest.raises(ValueError, match="inconsistent"):
        transform_weights_for_sm90_push(
            w13,
            torch.zeros(2, 128, 256, dtype=torch.bfloat16),
        )


def test_weight_transform_interleave() -> None:
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        transform_weights_for_sm90_push,
    )

    torch.manual_seed(3)
    num_experts, intermediate, hidden = 2, 384, 256
    w13 = (torch.randn(num_experts, 2 * intermediate, hidden) * hidden**-0.5).to(
        torch.bfloat16
    )
    w2 = (torch.randn(num_experts, hidden, intermediate) * intermediate**-0.5).to(
        torch.bfloat16
    )
    plain_fp8, plain_scales, w2_fp8_a, w2_scales_a = transform_weights_for_sm90_push(
        w13, w2
    )
    interleaved_fp8, interleaved_scales, w2_fp8_b, w2_scales_b = (
        transform_weights_for_sm90_push(w13, w2, interleave_gate_up=True)
    )
    blocks = intermediate // 128
    permutation = torch.tensor(
        [index for block in range(blocks) for index in (block, blocks + block)]
    )
    expected_fp8 = (
        plain_fp8.view(torch.uint8)
        .reshape(num_experts, 2 * blocks, 128, hidden)[:, permutation]
        .reshape(num_experts, 2 * intermediate, hidden)
    )
    expected_scales = plain_scales[:, permutation]
    assert torch.equal(interleaved_fp8.view(torch.uint8), expected_fp8)
    assert torch.equal(interleaved_scales, expected_scales)
    assert torch.equal(w2_fp8_a.view(torch.uint8), w2_fp8_b.view(torch.uint8))
    assert torch.equal(w2_scales_a, w2_scales_b)


@requires_sm90
@pytest.mark.parametrize(
    "baseline_kwargs,candidate_kwargs",
    [
        (
            {"payload_dtype": "bf16", "combine_dtype": "bf16"},
            {"payload_dtype": "fp8", "combine_dtype": "bf16"},
        ),
        (
            {"fuse_act": True, "combine_dtype": "bf16"},
            {"fuse_act": False, "combine_dtype": "bf16"},
        ),
        (
            {"fuse_fc1_epilogue": False, "combine_dtype": "bf16"},
            {"fuse_fc1_epilogue": True, "combine_dtype": "bf16"},
        ),
    ],
    ids=["fp8-payload", "fused-activation", "fused-fc1-epilogue"],
)
def test_ep1_optimized_paths_are_bitwise_transparent(
    baseline_kwargs: dict[str, object],
    candidate_kwargs: dict[str, object],
) -> None:
    device = torch.device("cuda", 0)
    x = _make_x(TOKEN_CAPACITY, 5, device)
    ids, weights = _make_routing(
        TOKEN_CAPACITY,
        LOCAL_EXPERTS,
        TOP_K,
        6,
        device,
    )
    _, baseline_runner, _, _ = _build(**baseline_kwargs)
    baseline = _run(baseline_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    _, candidate_runner, _, _ = _build(**candidate_kwargs)
    candidate = _run(candidate_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    assert torch.equal(candidate, baseline)


@requires_sm90
@pytest.mark.parametrize(
    "payload_dtype,combine_dtype",
    [("bf16", "bf16"), ("fp8", "bf16"), ("bf16", "fp8"), ("fp8", "fp8")],
)
def test_ep1_payload_and_combine_configs(
    payload_dtype: str, combine_dtype: str
) -> None:
    pipe, runner, _, transformed = _build(
        payload_dtype=payload_dtype, combine_dtype=combine_dtype
    )
    x = _make_x(TOKEN_CAPACITY, 1, pipe.device)
    ids, weights = _make_routing(TOKEN_CAPACITY, LOCAL_EXPERTS, TOP_K, 2, pipe.device)
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, transformed)
    assert output.shape == (TOKEN_CAPACITY, HIDDEN)
    assert torch.isfinite(output).all()
    assert _cosine(output, reference) > 0.997
    assert _relative_error(output, reference) < 0.10


@requires_sm90
def test_ep1_capacity_factor_happy_path() -> None:
    token_capacity, num_tokens = 256, 64
    pipe, runner, _, transformed = _build(
        capacity_factor=0.25,
        token_capacity=token_capacity,
        combine_dtype="bf16",
        fuse_fc1_epilogue=False,
    )
    assert pipe.m_ws == token_capacity * TOP_K
    assert pipe.m_cap == num_tokens * TOP_K
    assert runner.a1.shape[0] == pipe.m_cap < pipe.m_ws
    assert runner.h is not None and runner.h.shape[0] == pipe.m_cap
    assert runner.a2.shape[0] == runner.y.shape[0] == pipe.m_cap

    x = _make_x(num_tokens, 13, pipe.device)
    ids, weights = _make_routing(num_tokens, LOCAL_EXPERTS, TOP_K, 14, pipe.device)
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, transformed)
    assert output.shape == (num_tokens, HIDDEN)
    assert torch.isfinite(output).all()
    assert _cosine(output, reference) > 0.997
    assert _relative_error(output, reference) < 0.10


@requires_sm90
@pytest.mark.parametrize(
    "dedup_dispatch,grouped_combine,fuse_fc1_epilogue",
    [
        (False, False, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ],
)
def test_ep1_protocol_config_matrix(
    dedup_dispatch: bool,
    grouped_combine: bool,
    fuse_fc1_epilogue: bool,
) -> None:
    device = torch.device("cuda", 0)
    x = _make_x(TOKEN_CAPACITY, 11, device)
    ids, weights = _make_routing(TOKEN_CAPACITY, LOCAL_EXPERTS, TOP_K, 12, device)
    _, anchor_runner, _, transformed = _build(combine_dtype="bf16")
    anchor = _run(anchor_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    _, runner, _, _ = _build(
        combine_dtype="bf16" if not grouped_combine else "fp8",
        dedup_dispatch=dedup_dispatch,
        grouped_combine=grouped_combine,
        fuse_fc1_epilogue=fuse_fc1_epilogue,
    )
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, transformed)
    assert _cosine(output, reference) > 0.997
    if not grouped_combine:
        assert torch.equal(output, anchor)
    else:
        # fp8 combine-wire quantization noise is additive, not proportional to
        # the bf16 anchor's error, and the anchor's unfused tactic varies with
        # the GPU's SM count (swapAB threshold), so the ratio needs headroom.
        assert _relative_error(output, reference) <= (
            _relative_error(anchor, reference) * 1.25 + 1e-7
        )


@requires_sm90
@pytest.mark.parametrize("case", ["short", "masked", "hot", "empty"])
def test_ep1_edge_routes_and_recovery(case: str) -> None:
    pipe, runner, _, transformed = _build(
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    if case == "empty":
        num_tokens = 0
    elif case == "short":
        num_tokens = 7
    else:
        num_tokens = TOKEN_CAPACITY
    x = _make_x(num_tokens, 21, pipe.device)
    ids, weights = _make_routing(
        num_tokens,
        LOCAL_EXPERTS,
        TOP_K,
        22,
        pipe.device,
        mode="hot" if case == "hot" else "random",
    )
    if case == "masked":
        ids = ids.clone()
        ids[::3, 0] = -1
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    if num_tokens:
        reference = _dequant_reference(x, ids, weights, transformed)
        assert torch.isfinite(output).all()
        assert _cosine(output, reference) > 0.997

    recovery_x = _make_x(TOKEN_CAPACITY, 23, pipe.device)
    recovery_ids, recovery_weights = _make_routing(
        TOKEN_CAPACITY, LOCAL_EXPERTS, TOP_K, 24, pipe.device
    )
    recovered = _run(runner, recovery_x, recovery_ids, recovery_weights).clone()
    torch.cuda.synchronize()
    recovery_reference = _dequant_reference(
        recovery_x, recovery_ids, recovery_weights, transformed
    )
    assert _cosine(recovered, recovery_reference) > 0.997


@requires_sm90
def test_runner_output_identity_and_distinct_round_storage() -> None:
    pipe, runner, _, _ = _build()
    x1 = _make_x(TOKEN_CAPACITY, 31, pipe.device)
    ids1, weights1 = _make_routing(
        TOKEN_CAPACITY, LOCAL_EXPERTS, TOP_K, 32, pipe.device
    )
    output1 = torch.empty(
        TOKEN_CAPACITY, HIDDEN, dtype=torch.float32, device=pipe.device
    )
    assert _run(runner, x1, ids1, weights1, output=output1) is output1
    torch.cuda.synchronize()
    snapshot = output1.clone()

    x2 = _make_x(TOKEN_CAPACITY, 33, pipe.device)
    ids2, weights2 = _make_routing(
        TOKEN_CAPACITY, LOCAL_EXPERTS, TOP_K, 34, pipe.device
    )
    output2 = torch.empty_like(output1)
    assert _run(runner, x2, ids2, weights2, output=output2) is output2
    torch.cuda.synchronize()
    assert output1.data_ptr() != output2.data_ptr()
    assert torch.equal(output1, snapshot)
    assert not torch.equal(output1, output2)


@requires_sm90
def test_runner_state_machine_and_poison() -> None:
    pipe, runner, _, _ = _build()
    x = _make_x(7, 41, pipe.device)
    ids, weights = _make_routing(7, LOCAL_EXPERTS, TOP_K, 42, pipe.device)
    output = torch.empty(7, HIDDEN, dtype=torch.float32, device=pipe.device)
    assert runner.state == "idle"
    with pytest.raises(RuntimeError, match="preceding stage_inputs"):
        runner.compute(output=output)
    assert runner.state == "idle"

    runner.stage_inputs(x, ids, weights, output=output)
    assert runner.state == "staged"
    with pytest.raises(RuntimeError, match="called twice"):
        runner.stage_inputs(x, ids, weights, output=output)
    assert runner.state == "staged"
    assert runner.compute(output=output) is output
    assert runner.state == "idle"

    invalid_output = torch.empty(6, HIDDEN, dtype=torch.float32, device=pipe.device)
    with pytest.raises(ValueError, match="output must be"):
        runner.stage_inputs(x, ids, weights, output=invalid_output)
    assert runner.state == "idle"

    runner.stage_inputs(x, ids, weights, output=output)
    with pytest.raises(RuntimeError, match="same output tensor"):
        runner.compute(output=torch.empty_like(output))
    assert runner.state == "idle"

    runner.abort()
    assert runner.state == "poisoned"
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.stage_inputs(x, ids, weights, output=output)
    runner.destroy()
    assert runner.state == "destroyed"
    with pytest.raises(RuntimeError, match="destroyed"):
        runner.compute(output=output)


@requires_sm90
def test_runner_abort_poison_is_idempotent() -> None:
    pipe, runner, _, _ = _build()
    runner.abort()
    runner.abort()
    assert runner.state == "poisoned"
    x = _make_x(1, 43, pipe.device)
    ids, weights = _make_routing(1, LOCAL_EXPERTS, TOP_K, 44, pipe.device)
    output = torch.empty(1, HIDDEN, dtype=torch.float32, device=pipe.device)
    with pytest.raises(RuntimeError, match="poisoned"):
        runner.stage_inputs(x, ids, weights, output=output)


@requires_sm90
def test_ep1_graph_replay() -> None:
    pipe, runner, _, _ = _build(
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    inputs = [_make_x(TOKEN_CAPACITY, 51 + i, pipe.device) for i in range(2)]
    routes = [
        _make_routing(
            TOKEN_CAPACITY,
            LOCAL_EXPERTS,
            TOP_K,
            61 + i,
            pipe.device,
        )
        for i in range(2)
    ]
    eager = []
    for x, (ids, weights) in zip(inputs, routes, strict=True):
        eager.append(_run(runner, x, ids, weights).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(inputs[0]).copy_(inputs[0])
    static_ids = torch.empty_like(routes[0][0]).copy_(routes[0][0])
    static_weights = torch.empty_like(routes[0][1]).copy_(routes[0][1])
    static_output = torch.empty(
        TOKEN_CAPACITY, HIDDEN, dtype=torch.float32, device=pipe.device
    )
    # One persistent side stream: the runner's cross-stream guard requires
    # host-visible completion of the previous round before a stream switch,
    # which a fresh stream per warm-up iteration cannot guarantee.
    side_stream = torch.cuda.Stream()
    for _ in range(2):
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _run(
                runner,
                static_x,
                static_ids,
                static_weights,
                output=static_output,
            )
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = _run(
            runner,
            static_x,
            static_ids,
            static_weights,
            output=static_output,
        )
    assert captured_output is static_output

    replayed = []
    for x, (ids, weights) in zip(inputs, routes, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
    assert not torch.equal(replayed[0], replayed[1])
    for index in range(2):
        assert torch.equal(replayed[index], eager[index])


@requires_sm90
def test_ep1_soak() -> None:
    rounds = int(os.environ.get("SM90_PUSH_SOAK_ROUNDS", "60"))
    assert rounds >= 1
    pipe, runner, _, transformed = _build(
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    token_choices = [0, 1, 7, TOKEN_CAPACITY // 2, TOKEN_CAPACITY]
    pending = []
    for round_index in range(rounds):
        num_tokens = token_choices[(round_index * 7 + 3) % len(token_choices)]
        mode = "hot" if round_index % 3 == 2 else "random"
        x = _make_x(num_tokens, 1000 + round_index, pipe.device)
        ids, weights = _make_routing(
            num_tokens,
            LOCAL_EXPERTS,
            TOP_K,
            2000 + round_index,
            pipe.device,
            mode=mode,
        )
        output = _run(runner, x, ids, weights)
        pending.append((round_index, x, ids, weights, output.clone()))
        if round_index % 10 == 9 or round_index == rounds - 1:
            torch.cuda.synchronize()
            for saved_round, saved_x, saved_ids, saved_weights, saved_output in pending:
                if saved_x.shape[0] == 0:
                    continue
                assert torch.isfinite(saved_output).all()
                reference = _dequant_reference(
                    saved_x, saved_ids, saved_weights, transformed
                )
                assert _cosine(saved_output, reference) > 0.997, saved_round
            pending.clear()


@requires_sm90
def test_ep1_pool_overflow_traps() -> None:
    code = r"""
import torch
from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
    Sm90PushConfig, Sm90PushMoERunner, Sm90PushPipe, make_sm90_push_weights,
)
H, I, E, K, T = 512, 768, 4, 2, 64
pipe = Sm90PushPipe(
    ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
    token_capacity=T, device_index=0, config=Sm90PushConfig(capacity_factor=0.25),
)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device="cuda") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device="cuda") * I ** -0.5
runner = Sm90PushMoERunner(pipe, make_sm90_push_weights(w13, w2))
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.randint(0, E, (T, K), dtype=torch.int32, device="cuda")
weights = torch.rand(T, K, dtype=torch.float32, device="cuda")
output = torch.empty(T, H, dtype=torch.float32, device="cuda")
runner.stage_inputs(x, ids, weights, output=output)
runner.compute(output=output)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    _assert_device_trap(result, "sm90_push: pool overflow")


@requires_sm90
def test_ep1_wait_prefix_capacity_corruption_traps() -> None:
    code = r"""
import torch
from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import Sm90PushConfig, Sm90PushPipe

H, E, K, T = 512, 4, 2, 4
pipe = Sm90PushPipe(
    ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
    token_capacity=T, device_index=0, config=Sm90PushConfig(dedup_dispatch=False),
)
pipe.proto_begin_round()
x = torch.randn(T, H, dtype=torch.bfloat16, device="cuda")
ids = torch.zeros(T, K, dtype=torch.int32, device="cuda")
weights = torch.rand(T, K, dtype=torch.float32, device="cuda")
pipe.proto_dispatch(x, ids, weights)
pipe.module.sm90_push_wait_prefix(
    *pipe._layout_args(), pipe._round, pipe._rows_per_src, pipe._offsets,
    pipe._seg_src_base, pipe._seg_out_base, pipe._pad_base, pipe._m_dev,
    pipe._p_dev, pipe._next_row, 1,
)
torch.cuda.synchronize()
print("UNEXPECTED-SURVIVAL")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    _assert_device_trap(result, "sm90_push: accumulated row count")


@requires_sm90
@pytest.mark.parametrize(
    "case,marker",
    [
        ("invalid_expert", "sm90_push: invalid expert id"),
        ("dedup_payload_overflow", "sm90_push: dedup pool overflow"),
        ("dedup_meta_overflow", "sm90_push: dedup pool overflow"),
    ],
)
def test_ep1_dispatch_contract_traps(case: str, marker: str) -> None:
    if case == "invalid_expert":
        top_k = 2
        config = "Sm90PushConfig()"
        route_setup = "ids[3, 1] = E + 7"
    elif case == "dedup_payload_overflow":
        top_k = 4
        config = "Sm90PushConfig(capacity_factor=0.25, dedup_dispatch=True)"
        route_setup = """
ids.fill_(-1)
ids[:, 0] = torch.arange(T, dtype=torch.int32, device=\"cuda\") % E
"""
    else:
        top_k = 4
        config = "Sm90PushConfig(capacity_factor=0.25, dedup_dispatch=True)"
        route_setup = "ids.zero_()"

    code = f"""
import torch
from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
    Sm90PushConfig, Sm90PushMoERunner, Sm90PushPipe, make_sm90_push_weights,
)
H, I, E, K, T = 512, 768, 4, {top_k}, 32
pipe = Sm90PushPipe(
    ep_size=1, rank=0, num_local_experts=E, hidden_size=H, top_k=K,
    token_capacity=64, device_index=0, config={config},
)
w13 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=\"cuda\") * H ** -0.5
w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=\"cuda\") * I ** -0.5
runner = Sm90PushMoERunner(pipe, make_sm90_push_weights(w13, w2))
x = torch.randn(T, H, dtype=torch.bfloat16, device=\"cuda\")
ids = torch.randint(0, E, (T, K), dtype=torch.int32, device=\"cuda\")
{route_setup}
weights = torch.rand(T, K, dtype=torch.float32, device=\"cuda\")
output = torch.empty(T, H, dtype=torch.float32, device=\"cuda\")
runner.stage_inputs(x, ids, weights, output=output)
runner.compute(output=output)
torch.cuda.synchronize()
print(\"UNEXPECTED-SURVIVAL\")
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=600,
    )
    _assert_device_trap(result, marker)


def _dist_setup():
    import torch.distributed as dist

    if not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    from flashinfer.comm.mnnvl import TorchDistBackend

    return rank, world, TorchDistBackend(group=dist.group.WORLD)


def _full_transformed_weights(total_experts: int, device: torch.device):
    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        make_sm90_push_weights,
    )

    w13, w2 = _make_weights(total_experts, 7, device)
    return make_sm90_push_weights(w13, w2)


@requires_dist
@pytest.mark.parametrize("field", ["grouped_combine", "dedup_dispatch"])
def test_ep2_layout_fingerprint_mismatch_fails_fast(field: str) -> None:
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm90_push_megamoe import (
        Sm90PushConfig,
        Sm90PushPipe,
    )

    rank, world, comm = _dist_setup()
    config = Sm90PushConfig(**{field: rank == 0})
    with pytest.raises(RuntimeError, match="fingerprint mismatch"):
        Sm90PushPipe(
            ep_size=world,
            rank=rank,
            num_local_experts=LOCAL_EXPERTS,
            hidden_size=HIDDEN,
            top_k=TOP_K,
            token_capacity=TOKEN_CAPACITY,
            device_index=rank,
            config=config,
            comm_backend=comm,
        )
    dist.barrier()


@requires_dist
def test_ep2_protocol_optimizations_are_bitwise_transparent() -> None:
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    total_experts = LOCAL_EXPERTS * world
    common = {
        "device_index": rank,
        "ep_size": world,
        "rank": rank,
        "comm_backend": comm,
        "total_experts": total_experts,
        "grouped_combine": True,
    }
    _, grouped_runner, _, _ = _build(**common)
    _, dedup_grouped_runner, _, _ = _build(**common, dedup_dispatch=True)
    _, fully_fused_runner, _, _ = _build(
        **common,
        dedup_dispatch=True,
        fuse_fc1_epilogue=True,
    )
    device = torch.device("cuda", rank)
    x = _make_x(TOKEN_CAPACITY, 700 + rank, device)
    ids, weights = _make_routing(
        TOKEN_CAPACITY,
        total_experts,
        TOP_K,
        800 + rank,
        device,
    )
    grouped = _run(grouped_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    dedup_grouped = _run(dedup_grouped_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    fully_fused = _run(fully_fused_runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    assert torch.equal(dedup_grouped, grouped)
    assert torch.equal(fully_fused, dedup_grouped)
    transformed = _full_transformed_weights(total_experts, device)
    reference = _dequant_reference(x, ids, weights, transformed)
    assert _cosine(fully_fused, reference) > 0.997
    dist.barrier()


@requires_dist
def test_ep2_deterministic_round_skew() -> None:
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    total_experts = LOCAL_EXPERTS * world
    pipe, runner, _, _ = _build(
        device_index=rank,
        ep_size=world,
        rank=rank,
        comm_backend=comm,
        total_experts=total_experts,
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    transformed = _full_transformed_weights(total_experts, pipe.device)
    x = _make_x(TOKEN_CAPACITY, 900 + rank, pipe.device)
    ids, weights = _make_routing(
        TOKEN_CAPACITY,
        total_experts,
        TOP_K,
        1000 + rank,
        pipe.device,
    )
    if rank == 0:
        torch.cuda._sleep(int(2e8))
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    reference = _dequant_reference(x, ids, weights, transformed)
    assert torch.isfinite(output).all()
    assert _cosine(output, reference) > 0.997
    dist.barrier()


@requires_dist
@pytest.mark.parametrize("mode", ["random", "hot", "all_remote", "uneven"])
def test_ep2_forward_edge_routes(mode: str) -> None:
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    total_experts = LOCAL_EXPERTS * world
    pipe, runner, _, _ = _build(
        device_index=rank,
        ep_size=world,
        rank=rank,
        comm_backend=comm,
        total_experts=total_experts,
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    transformed = _full_transformed_weights(total_experts, pipe.device)
    num_tokens = 0 if mode == "uneven" and rank == 1 else TOKEN_CAPACITY
    x = _make_x(num_tokens, 300 + rank, pipe.device)
    ids, weights = _make_routing(
        num_tokens,
        total_experts,
        TOP_K,
        400 + rank,
        pipe.device,
        mode="random" if mode == "uneven" else mode,
        rank=rank,
        num_local_experts=LOCAL_EXPERTS,
    )
    output = _run(runner, x, ids, weights).clone()
    torch.cuda.synchronize()
    assert output.shape == (num_tokens, HIDDEN)
    assert torch.isfinite(output).all()
    if num_tokens:
        reference = _dequant_reference(x, ids, weights, transformed)
        assert _cosine(output, reference) > 0.997

    recovery_x = _make_x(TOKEN_CAPACITY, 500 + rank, pipe.device)
    recovery_ids, recovery_weights = _make_routing(
        TOKEN_CAPACITY,
        total_experts,
        TOP_K,
        600 + rank,
        pipe.device,
    )
    recovered = _run(runner, recovery_x, recovery_ids, recovery_weights).clone()
    torch.cuda.synchronize()
    recovery_reference = _dequant_reference(
        recovery_x, recovery_ids, recovery_weights, transformed
    )
    assert _cosine(recovered, recovery_reference) > 0.997
    dist.barrier()


@requires_dist
def test_ep2_graph_replay() -> None:
    import torch.distributed as dist

    rank, world, comm = _dist_setup()
    total_experts = LOCAL_EXPERTS * world
    pipe, runner, _, _ = _build(
        device_index=rank,
        ep_size=world,
        rank=rank,
        comm_backend=comm,
        total_experts=total_experts,
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    inputs = [
        _make_x(TOKEN_CAPACITY, 700 + 10 * i + rank, pipe.device) for i in range(2)
    ]
    routes = [
        _make_routing(
            TOKEN_CAPACITY,
            total_experts,
            TOP_K,
            800 + 10 * i + rank,
            pipe.device,
        )
        for i in range(2)
    ]
    eager = []
    for x, (ids, weights) in zip(inputs, routes, strict=True):
        eager.append(_run(runner, x, ids, weights).clone())
        torch.cuda.synchronize()

    static_x = torch.empty_like(inputs[0]).copy_(inputs[0])
    static_ids = torch.empty_like(routes[0][0]).copy_(routes[0][0])
    static_weights = torch.empty_like(routes[0][1]).copy_(routes[0][1])
    static_output = torch.empty(
        TOKEN_CAPACITY, HIDDEN, dtype=torch.float32, device=pipe.device
    )
    # One persistent side stream: the runner's cross-stream guard requires
    # host-visible completion of the previous round before a stream switch,
    # which a fresh stream per warm-up iteration cannot guarantee.
    side_stream = torch.cuda.Stream()
    for _ in range(2):
        side_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side_stream):
            _run(
                runner,
                static_x,
                static_ids,
                static_weights,
                output=static_output,
            )
        torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(
            runner,
            static_x,
            static_ids,
            static_weights,
            output=static_output,
        )
    replayed = []
    for x, (ids, weights) in zip(inputs, routes, strict=True):
        static_x.copy_(x)
        static_ids.copy_(ids)
        static_weights.copy_(weights)
        graph.replay()
        torch.cuda.synchronize()
        replayed.append(static_output.clone())
    assert not torch.equal(replayed[0], replayed[1])
    for index in range(2):
        assert torch.equal(replayed[index], eager[index])
    dist.barrier()


@requires_dist
def test_ep2_soak() -> None:
    import torch.distributed as dist

    rounds = int(os.environ.get("SM90_PUSH_SOAK_ROUNDS", "100"))
    assert rounds >= 1
    rank, world, comm = _dist_setup()
    total_experts = LOCAL_EXPERTS * world
    pipe, runner, _, _ = _build(
        device_index=rank,
        ep_size=world,
        rank=rank,
        comm_backend=comm,
        total_experts=total_experts,
        dedup_dispatch=True,
        grouped_combine=True,
        fuse_fc1_epilogue=True,
    )
    transformed = _full_transformed_weights(total_experts, pipe.device)
    generator = random.Random(4242 + rank)
    token_choices = [0, 1, 13, TOKEN_CAPACITY // 2, TOKEN_CAPACITY]
    modes = ["random", "hot", "all_remote"]
    pending = []
    for round_index in range(rounds):
        num_tokens = generator.choice(token_choices)
        mode = modes[round_index % len(modes)]
        if generator.random() < 0.2:
            torch.cuda._sleep(int(generator.random() * 1e8))
        x = _make_x(num_tokens, 3000 + 17 * round_index + rank, pipe.device)
        ids, weights = _make_routing(
            num_tokens,
            total_experts,
            TOP_K,
            5000 + 31 * round_index + rank,
            pipe.device,
            mode=mode,
            rank=rank,
            num_local_experts=LOCAL_EXPERTS,
        )
        output = _run(runner, x, ids, weights)
        pending.append((round_index, mode, x, ids, weights, output.clone()))
        if round_index % 10 == 9 or round_index == rounds - 1:
            torch.cuda.synchronize()
            for (
                saved_round,
                saved_mode,
                saved_x,
                saved_ids,
                saved_weights,
                saved_output,
            ) in pending:
                if saved_x.shape[0] == 0:
                    continue
                assert torch.isfinite(saved_output).all()
                reference = _dequant_reference(
                    saved_x, saved_ids, saved_weights, transformed
                )
                cosine = _cosine(saved_output, reference)
                assert cosine > 0.997, (
                    saved_round,
                    saved_mode,
                    saved_x.shape[0],
                    cosine,
                )
            pending.clear()
            dist.barrier()
    dist.barrier()
