"""Shared GPU harness for the exact-SM100/SM103 Cake warp-decode MoE tests.

One fixture builder, one public TRT-LLM NVFP4 reference and one warp-decode
launch path serve the parity matrix in ``test_cake_warp_decode.py``, the
clamped SM100 integration test in ``test_cake_warp_decode_clamped_gpu.py`` and
the workspace-receipt tests in ``test_cake_warp_decode_workspace_gpu.py``. Both
arms of every comparison consume the same prepared NVFP4 activation, routing,
packed-weight and block-scale tensors.

The ``check_*`` functions are the test bodies shared by the clamped fixture and
the parity-geometry fixture: cross-stream parity, MoELayer graph replay,
same-address receipt generations, re-preparation and release without caller
stream synchronisation, and graph replay after live input mutation with the
receipt released while a replay is still in flight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, MutableMapping

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CakeWarpDecodeConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    RoutingMethodType,
    SiTU,
    SwiGLU,
    TrtllmFp4Config,
    trtllm_fp4_block_scale_routed_moe,
)
from flashinfer.jit.cake_fused_moe_warp_decode import (
    get_cake_fused_moe_warp_decode_module,
)

MAX_TOKENS = 32
ATOL = 1e-2
RTOL = 1e-2
QUANT = QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.NVFP4)
TARGETS = {(10, 0): "sm100a", (10, 3): "sm103a"}
CLAMPED_SWIGLU = SwiGLU(limit=10.0)
# Deterministic replay delay that keeps a replay in flight across a release.
RETIREMENT_DELAY_CYCLES = 50_000_000
_EXPERT_TENSORS = (
    "gemm1_weights",
    "gemm1_weights_scale",
    "gemm2_weights",
    "gemm2_weights_scale",
)


@dataclass(frozen=True)
class Geometry:
    """One exact warp-decode model geometry and the token counts it is tested at."""

    name: str
    hidden_size: int
    intermediate_size: int
    num_experts: int
    top_k: int
    token_counts: tuple[int, ...]
    activation: object = SwiGLU()

    @property
    def has_public_reference(self) -> bool:
        return isinstance(self.activation, (SwiGLU, SiTU))

    @property
    def uses_activation_params(self) -> bool:
        return self.activation.is_gated and self.activation != SwiGLU()


# The clamped E256 route is served on exact SM100 only; test every token count.
CLAMPED_GEOMETRY = Geometry(
    "e256_i2048_k6_clamp10", 4096, 2048, 256, 6, tuple(range(1, 33)), CLAMPED_SWIGLU
)


@dataclass
class Fixture:
    """Prepared TRT-LLM NVFP4 tensors for one geometry at ``MAX_TOKENS`` rows."""

    geometry: Geometry
    hidden_states_q: torch.Tensor
    hidden_states_scale: torch.Tensor
    topk_ids: torch.Tensor
    topk_weights: torch.Tensor
    weight_view: MutableMapping[str, torch.Tensor]

    @property
    def device(self) -> torch.device:
        return self.hidden_states_q.device

    def shape(self, num_tokens: int) -> tuple[int, int, int, int, int]:
        geometry = self.geometry
        return (
            num_tokens,
            geometry.hidden_size,
            geometry.intermediate_size,
            geometry.num_experts,
            geometry.top_k,
        )


def routing(
    geometry: Geometry, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic unique top-k ids (expert 0 first) and normalized BF16 weights."""
    tokens = torch.arange(MAX_TOKENS, device=device, dtype=torch.int64)[:, None]
    ranks = torch.arange(geometry.top_k, device=device, dtype=torch.int64)[None, :]
    ids = ((tokens * 17 + ranks * 29) % (geometry.num_experts - 1) + 1).to(torch.int32)
    ids[:, 0] = 0
    raw_weights = (geometry.top_k + 1 - ranks).expand(MAX_TOKENS, -1).float()
    raw_weights = raw_weights + (tokens % 5).float() * 0.03125
    weights = (raw_weights / raw_weights.sum(dim=1, keepdim=True)).to(torch.bfloat16)
    return ids.contiguous(), weights.contiguous()


def mutated_topk_ids(geometry: Geometry, device: torch.device) -> torch.Tensor:
    """A second unique top-k assignment that still routes expert 0 first."""
    tokens = torch.arange(MAX_TOKENS, device=device, dtype=torch.int64)[:, None]
    ranks = torch.arange(geometry.top_k, device=device, dtype=torch.int64)[None, :]
    ids = ((tokens * 31 + ranks * 37 + 11) % (geometry.num_experts - 1) + 1).to(
        torch.int32
    )
    ids[:, 0] = 0
    return ids.contiguous()


def prepare_fixture(geometry: Geometry, seed: int) -> Fixture:
    """Quantize one geometry's activations and expert weights with the public API."""
    device = torch.device("cuda")
    torch.manual_seed(seed)
    hidden = torch.randn(
        MAX_TOKENS, geometry.hidden_size, device=device, dtype=torch.bfloat16
    )
    gate_rows = 2 if geometry.activation.is_gated else 1
    w1_shape = (
        geometry.num_experts,
        geometry.intermediate_size * gate_rows,
        geometry.hidden_size,
    )
    w2_shape = (geometry.num_experts, geometry.hidden_size, geometry.intermediate_size)
    if geometry.activation == CLAMPED_SWIGLU:
        # The clamped fixture keeps the FP32 draw of the benchmark it came from,
        # so its NVFP4 codes match that baseline bit for bit. The parity
        # geometries draw in BF16: the largest one would need ~160 GB of FP32
        # temporaries.
        w1 = (torch.randn(*w1_shape, device=device) * 0.02).to(torch.bfloat16)
        w2 = (torch.randn(*w2_shape, device=device) * 0.02).to(torch.bfloat16)
        hidden.mul_(16.0)  # Exercise both upper-gate and symmetric-linear clamp.
    else:
        w1 = torch.randn(*w1_shape, device=device, dtype=torch.bfloat16)
        w1 *= 0.02
        w2 = torch.randn(*w2_shape, device=device, dtype=torch.bfloat16)
        w2 *= 0.02
    hidden_q, hidden_scale = TrtllmFp4Config.prepare_activations(hidden, quant=QUANT)
    weight_view = CakeWarpDecodeConfig.prepare_weights(
        w1,
        w2,
        quant=QUANT,
        num_local_experts=geometry.num_experts,
        hidden_size=geometry.hidden_size,
        intermediate_size=geometry.intermediate_size,
        activation=geometry.activation,
        device=device,
    )
    del w1, w2
    if geometry.activation == CLAMPED_SWIGLU:
        # Both arms consume these same physical tensors. Non-unit gate scales
        # exercise the raw-accumulator clamp conversion used by calibrated views.
        gate_scale = torch.linspace(
            0.5, 1.5, geometry.num_experts, dtype=torch.float32, device=device
        )
        weight_view["output1_scale_gate_scalar"] = gate_scale
        weight_view["gemm1_clamp_limit"] = (
            torch.full_like(gate_scale, 10.0) / gate_scale
        )
    topk_ids, topk_weights = routing(geometry, device)
    return Fixture(
        geometry=geometry,
        hidden_states_q=hidden_q,
        hidden_states_scale=hidden_scale,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        weight_view=weight_view,
    )


def cake_module(device: torch.device) -> Any:
    """Load the generated module for the exact SM100 / SM103 target of ``device``."""
    target = TARGETS[torch.cuda.get_device_capability(device)]
    return get_cake_fused_moe_warp_decode_module(target, device=device)


def reference(fixture: Fixture, num_tokens: int) -> torch.Tensor:
    """Run the public TRT-LLM NVFP4 routed MoE on the same physical tensors."""
    geometry = fixture.geometry
    view = fixture.weight_view
    official_beta = view.get("gemm1_beta")
    official_clamp_limit = view.get("gemm1_clamp_limit")
    if geometry.name == "e128_i3072_k4_swiglu_oa":
        # The public baseline consumes beta and clamp in raw-accumulator units.
        official_beta = view["gemm1_beta"] / view["output1_scale_gate_scalar"]
        official_clamp_limit = (
            view["gemm1_clamp_limit"] / view["output1_scale_gate_scalar"]
        )
    output = torch.empty(
        num_tokens, geometry.hidden_size, dtype=torch.bfloat16, device=fixture.device
    )
    result = trtllm_fp4_block_scale_routed_moe(
        topk_ids=(fixture.topk_ids[:num_tokens], fixture.topk_weights[:num_tokens]),
        routing_bias=None,
        hidden_states=fixture.hidden_states_q[:num_tokens],
        hidden_states_scale=fixture.hidden_states_scale[:num_tokens],
        gemm1_weights=view["gemm1_weights"],
        gemm1_weights_scale=view["gemm1_weights_scale"],
        gemm1_bias=None,
        gemm1_alpha=view.get("gemm1_alpha"),
        gemm1_beta=official_beta,
        gemm1_clamp_limit=official_clamp_limit,
        gemm2_weights=view["gemm2_weights"],
        gemm2_weights_scale=view["gemm2_weights_scale"],
        gemm2_bias=None,
        output1_scale_scalar=view["output1_scale_scalar"],
        output1_scale_gate_scalar=view["output1_scale_gate_scalar"],
        output2_scale_scalar=view["output2_scale_scalar"],
        num_experts=geometry.num_experts,
        top_k=geometry.top_k,
        n_group=None,
        topk_group=None,
        intermediate_size=geometry.intermediate_size,
        local_expert_offset=0,
        local_num_experts=geometry.num_experts,
        routed_scaling_factor=None,
        routing_method_type=RoutingMethodType.TopK.value,
        do_finalize=True,
        enable_pdl=True,
        activation_type=geometry.activation.type.value,
        per_token_scale=None,
        output=output,
        tune_max_num_tokens=MAX_TOKENS,
    )
    if isinstance(result, (list, tuple)):
        result = result[0]
    assert result.data_ptr() == output.data_ptr()
    return output


def prepare_launch(
    module: Any, fixture: Fixture, num_tokens: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Allocate the output and workspace for one token count and prepare a receipt.

    The caller releases the receipt with
    ``module.cake_fused_moe_warp_decode_release_workspace``.
    """
    shape = fixture.shape(num_tokens)
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    assert workspace_size > 0
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=fixture.device)
    output = torch.empty(
        num_tokens,
        fixture.geometry.hidden_size,
        dtype=torch.bfloat16,
        device=fixture.device,
    )
    receipt = int(
        module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
    )
    assert receipt > 0
    return output, workspace, receipt


def run_cake(
    module: Any,
    fixture: Fixture,
    num_tokens: int,
    output: torch.Tensor,
    workspace: torch.Tensor,
    receipt: int,
) -> torch.Tensor:
    """Launch the generated warp-decode kernels into ``output`` and return it."""
    geometry = fixture.geometry
    view = fixture.weight_view
    inputs = (
        output,
        workspace,
        fixture.hidden_states_q[:num_tokens],
        fixture.hidden_states_scale[:num_tokens],
        fixture.topk_ids[:num_tokens],
        fixture.topk_weights[:num_tokens],
        view["gemm1_weights"],
        view["gemm1_weights_scale"],
        view["gemm2_weights"],
        view["gemm2_weights_scale"],
        view["output1_scale_scalar"],
        view["output1_scale_gate_scalar"],
        view["output2_scale_scalar"],
    )
    if geometry.activation == CLAMPED_SWIGLU:
        module.cake_fused_moe_warp_decode_clamped_swiglu(
            *inputs,
            view["gemm1_alpha"],
            view["gemm1_beta"],
            view["gemm1_clamp_limit"],
            receipt,
            True,
        )
    elif geometry.uses_activation_params:
        module.cake_fused_moe_warp_decode_with_activation_params(
            *inputs,
            view.get("gemm1_alpha"),
            view.get("gemm1_beta"),
            view.get("gemm1_clamp_limit"),
            receipt,
            True,
        )
    else:
        module.cake_fused_moe_warp_decode(*inputs, receipt, True)
    return output


def nondefault_streams(
    device: torch.device, count: int
) -> tuple[torch.cuda.Stream, ...]:
    """Create ``count`` distinct non-default streams ordered after the current one."""
    current = torch.cuda.current_stream(device)
    streams = tuple(torch.cuda.Stream(device=device) for _ in range(count))
    handles = {int(stream.cuda_stream) for stream in (current, *streams)}
    assert len(handles) == count + 1, "failed to create distinct CUDA streams"
    for stream in streams:
        stream.wait_stream(current)
    return streams


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Compare two BF16 outputs at the shared warp-decode tolerance."""
    assert actual.dtype == expected.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=ATOL, rtol=RTOL)


def check_cross_stream_parity(fixture: Fixture, num_tokens: int) -> None:
    """Prepare on one non-default stream, launch twice on another, compare both.

    The output is poisoned between the launches so the second one proves that
    the prepared workspace and the caller-owned output are reusable.
    """
    prepare_stream, run_stream = nondefault_streams(fixture.device, 2)
    with torch.cuda.stream(prepare_stream):
        module = cake_module(fixture.device)
        output, workspace, receipt = prepare_launch(module, fixture, num_tokens)
    try:
        with torch.cuda.stream(run_stream):
            first = run_cake(
                module, fixture, num_tokens, output, workspace, receipt
            ).clone()
            expected = reference(fixture, num_tokens).clone()
            output.fill_(float("nan"))
            second = run_cake(
                module, fixture, num_tokens, output, workspace, receipt
            ).clone()
        run_stream.synchronize()
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
    assert_close(first, expected)
    assert_close(second, expected)
    assert_close(second, first)


def check_layer_graph_replay(fixture: Fixture, num_tokens: int) -> None:
    """Exercise the public MoELayer winner path and graph capture end to end.

    The first public call runs the real one-backend winner-selection path on a
    non-default warm-up stream; the layer is then captured on a second stream
    and replayed on a third, and both outputs are compared with the reference.
    """
    geometry = fixture.geometry
    device = fixture.device
    activations = MoEActivationPack(
        hidden_states_q=fixture.hidden_states_q[:num_tokens],
        hidden_states_scale=fixture.hidden_states_scale[:num_tokens],
        topk_ids=fixture.topk_ids[:num_tokens],
        topk_weights=fixture.topk_weights[:num_tokens],
        routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
    )
    weights = MoEWeightPack()
    weights.prepare_for("cake", dict(fixture.weight_view))
    config = MoEConfig(
        routing=RoutingConfig(
            num_experts=geometry.num_experts,
            top_k=geometry.top_k,
            method=RoutingMethodType.TopK,
        ),
        quant=QUANT,
        experts=ExpertConfig(intermediate_size=geometry.intermediate_size),
        activation=geometry.activation,
        backend=BackendOptions((CakeWarpDecodeConfig(),)),
        execution=ExecutionConfig(tune_max_num_tokens=MAX_TOKENS, enable_pdl=True),
    )
    layer = MoELayer(config, device=device)
    layer.tuner.clear_cache()
    layer.tuner.reset_statistics()
    warmup_stream, capture_stream, replay_stream = nondefault_streams(device, 3)

    with autotune(True, tuning_buckets=(num_tokens,)), torch.cuda.stream(warmup_stream):
        eager = layer(activations, weights).clone()
        expected_eager = reference(fixture, num_tokens).clone()
    stats = layer.tuner.stats
    assert stats.tuned_op_total_configs.get("moe_cake", 0) >= 1
    assert stats.tuned_op_successful_configs.get("moe_cake", 0) >= 1
    warmup_stream.synchronize()
    assert_close(eager, expected_eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=capture_stream):
        captured_output = layer(activations, weights)
    with torch.cuda.stream(replay_stream):
        graph.replay()
        replayed = captured_output.clone()
        expected_replay = reference(fixture, num_tokens).clone()
    replay_stream.synchronize()
    assert_close(replayed, expected_replay)


def _expect_launch_rejected(
    module: Any, fixture: Fixture, num_tokens: int, output, workspace, receipt: int
) -> None:
    with pytest.raises(Exception, match="current preparation receipt"):
        run_cake(module, fixture, num_tokens, output, workspace, receipt)


def _assert_changed(mutated: torch.Tensor, baseline: torch.Tensor, what: str) -> None:
    assert not torch.equal(mutated, baseline), f"{what} left the reference unchanged"


def check_same_address_receipt_generations(fixture: Fixture) -> None:
    """A new tensor generation at one device address advances the receipt.

    Releasing receipt ``0``, the stale receipt or an already released receipt is
    rejected, and so is a launch with a stale or released receipt; the launch
    with the live receipt matches the reference.
    """
    geometry = fixture.geometry
    num_tokens = geometry.token_counts[0]
    shape = fixture.shape(num_tokens)
    device = fixture.device
    prepare_stream, run_stream = nondefault_streams(device, 2)
    with torch.cuda.stream(prepare_stream):
        module = cake_module(device)
        workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
        workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
        # Model allocator reuse: a new tensor object over the same storage.
        replacement = workspace.view(-1)
        assert replacement is not workspace
        assert replacement.data_ptr() == workspace.data_ptr()
        output = torch.empty(
            num_tokens, geometry.hidden_size, dtype=torch.bfloat16, device=device
        )
        first_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
        )
        # A same-address re-prepare retires the first generation synchronously.
        second_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(replacement, *shape)
        )
    assert first_receipt > 0 and second_receipt > 0
    assert second_receipt != first_receipt
    try:
        with pytest.raises(Exception, match="positive preparation receipt"):
            module.cake_fused_moe_warp_decode_release_workspace(0)
        with pytest.raises(Exception, match="unknown or already released receipt"):
            module.cake_fused_moe_warp_decode_release_workspace(first_receipt)
        with torch.cuda.stream(run_stream):
            _expect_launch_rejected(
                module, fixture, num_tokens, output, replacement, first_receipt
            )
            actual = run_cake(
                module, fixture, num_tokens, output, replacement, second_receipt
            ).clone()
            expected = reference(fixture, num_tokens).clone()
        run_stream.synchronize()
    finally:
        module.cake_fused_moe_warp_decode_release_workspace(second_receipt)
    assert_close(actual, expected)
    with pytest.raises(Exception, match="unknown or already released receipt"):
        module.cake_fused_moe_warp_decode_release_workspace(second_receipt)
    with torch.cuda.stream(run_stream):
        _expect_launch_rejected(
            module, fixture, num_tokens, output, replacement, second_receipt
        )


def check_reprepare_and_release_without_stream_sync(fixture: Fixture) -> None:
    """Re-preparation and release retire recorded submissions on their own."""
    geometry = fixture.geometry
    num_tokens = geometry.token_counts[-1]
    shape = fixture.shape(num_tokens)
    device = fixture.device
    prepare_stream, first_stream, second_stream = nondefault_streams(device, 3)
    module = cake_module(device)
    workspace_size = int(module.cake_fused_moe_warp_decode_workspace_size(*shape))
    workspace = torch.empty(workspace_size, dtype=torch.uint8, device=device)
    output = torch.empty(
        num_tokens, geometry.hidden_size, dtype=torch.bfloat16, device=device
    )
    with torch.cuda.stream(prepare_stream):
        first_receipt = int(
            module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
        )
    live_receipt = first_receipt
    try:
        with torch.cuda.stream(first_stream):
            run_cake(module, fixture, num_tokens, output, workspace, first_receipt)
        # No first_stream synchronisation: re-preparation must retire the
        # recorded submission before it clears and reinitialises the workspace.
        # The re-preparation revokes first_receipt even when it fails, so stop
        # tracking it before the transition.
        live_receipt = None
        with torch.cuda.stream(second_stream):
            second_receipt = int(
                module.cake_fused_moe_warp_decode_prepare_workspace(workspace, *shape)
            )
            live_receipt = second_receipt
            assert second_receipt > 0 and second_receipt != first_receipt
            replacement = run_cake(
                module, fixture, num_tokens, output, workspace, second_receipt
            ).clone()
        # No second_stream synchronisation either: the release must wait for
        # its completion event before retiring the state.
        module.cake_fused_moe_warp_decode_release_workspace(second_receipt)
        live_receipt = None
    finally:
        if live_receipt is not None:
            module.cake_fused_moe_warp_decode_release_workspace(live_receipt)
    with torch.cuda.stream(second_stream):
        expected = reference(fixture, num_tokens).clone()
    second_stream.synchronize()
    assert_close(replacement, expected)


def _replay_parameter_mutations(
    fixture: Fixture,
    num_tokens: int,
    graph: torch.cuda.CUDAGraph,
    output: torch.Tensor,
    before: torch.Tensor,
    stream: torch.cuda.Stream,
) -> None:
    """Replay after perturbing one expert's activation parameters in place."""
    view = fixture.weight_view
    expert = int(fixture.topk_ids[0, 0].item())
    perturbations = (
        ("gemm1_alpha", 0.0),
        ("gemm1_beta", 4.0),
        ("gemm1_clamp_limit", 0.0625),
    )
    for name, value in perturbations:
        parameter = view.get(name)
        if parameter is None:
            continue
        with torch.cuda.stream(stream):
            saved = parameter[expert].clone()
            parameter[expert].fill_(value)
            try:
                graph.replay()
                actual = output.clone()
                expected = reference(fixture, num_tokens).clone()
            finally:
                parameter[expert].copy_(saved)
        stream.synchronize()
        _assert_changed(expected, before, f"the {name} mutation")
        assert_close(actual, expected)


def check_graph_replay_after_mutation(fixture: Fixture, num_tokens: int) -> None:
    """Capture once, mutate the live inputs, replay, and release mid-replay.

    After the warm-up and an initial replay, parameterised activations are
    perturbed one expert at a time. Then the captured routing slots receive a
    second assignment with all weight on expert 0, whose packed weights and
    scales are zeroed; a delayed replay is released while still in flight, the
    workspace is overwritten, re-prepared at the same address and launched again.
    """
    geometry = fixture.geometry
    device = fixture.device
    view = fixture.weight_view
    prepare_stream, capture_stream, replay_stream = nondefault_streams(device, 3)
    with torch.cuda.stream(prepare_stream):
        module = cake_module(device)
        output, workspace, receipt = prepare_launch(module, fixture, num_tokens)
    live_receipt = receipt
    saved_topk_ids = fixture.topk_ids.clone()
    saved_topk_weights = fixture.topk_weights.clone()
    saved_experts = [(view[name][0], view[name][0].clone()) for name in _EXPERT_TENSORS]
    try:
        with torch.cuda.stream(capture_stream):
            warmup = run_cake(
                module, fixture, num_tokens, output, workspace, receipt
            ).clone()
            before = reference(fixture, num_tokens).clone()
        capture_stream.synchronize()
        assert_close(warmup, before)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            run_cake(module, fixture, num_tokens, output, workspace, receipt)
        with torch.cuda.stream(capture_stream):
            output.fill_(float("nan"))
            graph.replay()
            initial_replay = output.clone()
        capture_stream.synchronize()
        assert_close(initial_replay, before)

        if geometry.uses_activation_params:
            _replay_parameter_mutations(
                fixture, num_tokens, graph, output, before, capture_stream
            )

        # Route every token's whole weight through expert 0, whose packed
        # weights and scales are zeroed, so the replay perturbation stays
        # observable even where NVFP4 quantisation would collapse a small delta
        # at BF16 output precision.
        replay_completion = torch.cuda.Event()
        with torch.cuda.stream(replay_stream):
            mutated_ids = mutated_topk_ids(geometry, device)[:num_tokens]
            fixture.topk_ids[:num_tokens].copy_(mutated_ids)
            fixture.topk_weights[:num_tokens].zero_()
            fixture.topk_weights[:num_tokens, 0].fill_(1.0)
            for tensor, _saved in saved_experts:
                tensor.zero_()
            output.fill_(float("nan"))
            torch.cuda._sleep(RETIREMENT_DELAY_CYCLES)
            graph.replay()
            replay_completion.record()
            replayed = output.clone()
            expected = reference(fixture, num_tokens).clone()
        # Release while the delayed replay is still in flight, without any
        # replay-stream synchronisation: the binding must wait for the
        # graph-owned completion event before the storage is overwritten below.
        assert not replay_completion.query(), "the replay delay already completed"
        module.cake_fused_moe_warp_decode_release_workspace(receipt)
        live_receipt = None
        replacement_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(replacement_stream):
            workspace.fill_(0xA5)
        replacement_stream.synchronize()
        replay_stream.synchronize()

        # Reinitialise the overwritten storage and launch again at that address.
        with torch.cuda.stream(replacement_stream):
            replacement_receipt = int(
                module.cake_fused_moe_warp_decode_prepare_workspace(
                    workspace, *fixture.shape(num_tokens)
                )
            )
            live_receipt = replacement_receipt
            assert replacement_receipt > 0
            replacement = run_cake(
                module,
                fixture,
                num_tokens,
                torch.empty_like(output),
                workspace,
                replacement_receipt,
            ).clone()
        module.cake_fused_moe_warp_decode_release_workspace(replacement_receipt)
        live_receipt = None
        replacement_stream.synchronize()
    finally:
        if live_receipt is not None:
            module.cake_fused_moe_warp_decode_release_workspace(live_receipt)
        with torch.cuda.stream(replay_stream):
            for tensor, saved in saved_experts:
                tensor.copy_(saved)
            fixture.topk_ids.copy_(saved_topk_ids)
            fixture.topk_weights.copy_(saved_topk_weights)
        replay_stream.synchronize()

    _assert_changed(expected, before, "the routing and expert mutation")
    assert_close(replayed, expected)
    assert_close(replacement, expected)
