from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from b12x.norm import hyperconnection as hc

from ..conftest import require_b12x as require_sm120


def _allocate_binding(
    *,
    device: torch.device | str,
    tokens: int,
    max_tokens: int | None = None,
    hidden_size: int = 2560,
    streams: int = 4,
    lowrank: int = 320,
) -> hc.Binding:
    device = torch.device(device)
    capacity = tokens if max_tokens is None else max_tokens
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )
    width = streams * hidden_size
    return hc.bind(
        plan,
        tokens=tokens,
        normalized=torch.empty((capacity, width), dtype=torch.bfloat16, device=device),
        bottleneck=torch.empty(
            (capacity, lowrank), dtype=torch.bfloat16, device=device
        ),
        block_input=torch.empty(
            (capacity, hidden_size), dtype=torch.bfloat16, device=device
        ),
    )


def _direct_grouped_rmsnorm(
    state: torch.Tensor,
    weight: torch.Tensor,
    *,
    streams: int,
    eps: float,
) -> torch.Tensor:
    tokens, width = state.shape
    hidden_size = width // streams
    grouped = state.float().view(tokens, streams, hidden_size)
    variance = grouped.square().mean(dim=-1, keepdim=True)
    normalized = grouped * torch.rsqrt(variance + eps)
    return (normalized.flatten(1) * (1.0 + weight.float())).to(state.dtype)


def test_reference_matches_explicit_target_expressions() -> None:
    generator = torch.Generator().manual_seed(20260825)
    tokens, streams, hidden_size, lowrank = 2, 4, 2560, 320
    width = streams * hidden_size
    state = torch.randn((tokens, width), generator=generator, dtype=torch.bfloat16)
    norm_weight = torch.randn((width,), generator=generator, dtype=torch.bfloat16) / 16
    projected_down = torch.randn(
        (tokens, lowrank), generator=generator, dtype=torch.bfloat16
    )
    gate_logits = torch.randn(
        (tokens, width), generator=generator, dtype=torch.bfloat16
    )
    block_output = torch.randn(
        (tokens, hidden_size), generator=generator, dtype=torch.bfloat16
    )
    injection_logits = torch.randn(
        (tokens, streams), generator=generator, dtype=torch.bfloat16
    )
    eps = 1e-6

    normalized = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    expected_normalized = _direct_grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=0)

    bottleneck = hc.reference.scaled_silu(projected_down, streams=streams)
    torch.testing.assert_close(
        bottleneck,
        F.silu(projected_down / streams),
        rtol=0,
        atol=0,
    )

    block_input = hc.reference.gate_mean(normalized, gate_logits, streams=streams)
    expected_input = (
        torch.sigmoid(gate_logits).view(tokens, streams, hidden_size)
        * normalized.view(tokens, streams, hidden_size)
    ).mean(dim=1)
    torch.testing.assert_close(block_input, expected_input, rtol=0, atol=0)

    combined, next_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    scale = 2.0 * torch.sigmoid(injection_logits.float() / streams)
    expected_combined = (
        (
            state.float().view(tokens, streams, hidden_size)
            + block_output.float().unsqueeze(1) * scale.unsqueeze(-1)
        )
        .to(torch.bfloat16)
        .flatten(1)
    )
    expected_next_normalized = _direct_grouped_rmsnorm(
        expected_combined,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=0)
    torch.testing.assert_close(
        next_normalized, expected_next_normalized, rtol=0, atol=0
    )


def test_plan_bind_uses_live_views_and_no_scratch() -> None:
    binding = _allocate_binding(
        device="cpu",
        tokens=3,
        max_tokens=8,
        hidden_size=64,
        lowrank=16,
    )
    assert binding.plan.scratch_specs() == ()
    assert binding.plan.output_shapes(tokens=3) == {
        "normalized": (3, 256),
        "bottleneck": (3, 16),
        "block_input": (3, 64),
    }
    assert binding.normalized.shape == (3, 256)
    assert binding.bottleneck.shape == (3, 16)
    assert binding.block_input.shape == (3, 64)
    assert binding.normalized_capacity.shape == (8, 256)
    assert binding.bottleneck_capacity.shape == (8, 16)
    assert binding.block_input_capacity.shape == (8, 64)
    assert binding.normalized.data_ptr() == binding.normalized_capacity.data_ptr()


def test_projection_slices_match_packed_inputs_under_frozen_replay() -> None:
    """Merged projection views retain values, padding and live row strides."""
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.norm.hyperconnection import _cute, _kernels

    device = require_sm120()
    capacity, streams, lowrank, hidden = 16, 4, 320, 2560
    binding = _allocate_binding(device=device, tokens=capacity)
    weights = torch.randn(streams * hidden, device=device, dtype=torch.bfloat16)
    _cute.clear_caches()

    def check(rows: int, padding: int) -> None:
        merged = torch.randn(
            rows, lowrank + streams + padding, device=device, dtype=torch.bfloat16
        )
        down = merged[:, :lowrank]
        injection = merged[:, lowrank : lowrank + streams]
        state = torch.randn(rows, streams * hidden, device=device, dtype=torch.bfloat16)
        block = torch.randn(rows, hidden, device=device, dtype=torch.bfloat16)
        live = binding.plan.bind(
            tokens=rows,
            normalized=binding.normalized_capacity,
            bottleneck=binding.bottleneck_capacity,
            block_input=binding.block_input_capacity,
        )

        def launch():
            silu = hc.run_scaled_silu(down, binding=live)
            combined = hc.run_combine(state, block, injection, plan=live.plan)
            fused, normalized = hc.run_combine_norm(
                state, block, injection, weights, eps=1e-6, plan=live.plan
            )
            return silu, combined, fused, normalized

        launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs = launch()
        for _ in range(2):
            merged.normal_()
            before = merged.clone()
            expected_silu = torch.empty_like(down)
            _kernels._scaled_silu_launch(down.contiguous(), expected_silu, streams, 256)
            expected_combine = torch.empty_like(state)
            _kernels._combine_launch(
                state,
                block,
                injection.contiguous(),
                expected_combine,
                streams,
                hidden,
                4096,
                4,
            )
            expected_pair = hc.run_combine_norm(
                state, block, injection.contiguous(), weights, eps=1e-6, plan=live.plan
            )
            allocated = torch.cuda.memory_allocated(device)
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_allocated(device) == allocated
            for actual, expected in zip(
                outputs, (expected_silu, expected_combine, *expected_pair), strict=True
            ):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(merged, before, rtol=0, atol=0)

    check(1, 12)
    with kernel_resolution_guard():
        for rows in (4, 16):
            for padding in (3, 12, 28):
                check(rows, padding)


def test_projection_stride_alias_checks_cover_last_live_row() -> None:
    """An output alias in a strided source's final row must fail closed."""
    device = require_sm120()
    binding = _allocate_binding(device=device, tokens=2, lowrank=8)
    storage = torch.empty(80, device=device, dtype=torch.bfloat16)
    projected = storage.as_strided((2, 8), (64, 1))
    live = binding.plan.bind(
        normalized=binding.normalized_capacity,
        bottleneck=storage[64:80].view(2, 8),
        block_input=binding.block_input_capacity,
    )
    with pytest.raises(ValueError, match="must not overlap"):
        hc.run_scaled_silu(projected, binding=live)
    with pytest.raises(ValueError, match="unit column stride"):
        hc.run_scaled_silu(
            torch.empty(2, 16, device=device, dtype=torch.bfloat16)[:, ::2],
            binding=binding,
        )


@pytest.mark.parametrize("streams", [1, 3, 4, 8])
def test_scaled_silu_preserves_triton_finite_bf16_bit_patterns(streams: int) -> None:
    from b12x.norm.hyperconnection import _cute, _kernels

    device = require_sm120()
    patterns = torch.arange(65536, device=device, dtype=torch.int32).to(torch.int16)
    values = patterns.view(torch.bfloat16)
    values = values[torch.isfinite(values)].view(-1, 256)
    oracle = torch.empty_like(values)
    actual = torch.empty_like(values)
    _kernels._scaled_silu_launch(values, oracle, streams, 256)
    _cute.scaled_silu(values, actual, streams=streams)
    torch.testing.assert_close(actual, oracle, rtol=0, atol=0)


def test_projection_readers_address_rows_beyond_int32_elements() -> None:
    """Only the two live rows are initialized in the wide projection owner."""
    device = require_sm120()
    stride, lowrank, streams = (1 << 31) + 32, 8, 4
    storage = torch.empty(stride + 16, device=device, dtype=torch.bfloat16)
    projected = storage.as_strided((2, lowrank), (stride, 1))
    injection = storage.as_strided((2, streams), (stride, 1), storage_offset=8)
    projected.normal_()
    injection.normal_()
    binding = _allocate_binding(device=device, tokens=2, lowrank=lowrank)
    state = torch.randn(2, streams * 2560, device=device, dtype=torch.bfloat16)
    block = torch.randn(2, 2560, device=device, dtype=torch.bfloat16)
    weight = torch.randn(streams * 2560, device=device, dtype=torch.bfloat16)
    expected_silu = hc.run_scaled_silu(projected.contiguous(), binding=binding).clone()
    expected = hc.run_combine_norm(
        state, block, injection.contiguous(), weight, eps=1e-6, plan=binding.plan
    )
    actual_silu = hc.run_scaled_silu(projected, binding=binding)
    actual = hc.run_combine_norm(
        state, block, injection, weight, eps=1e-6, plan=binding.plan
    )
    torch.testing.assert_close(actual_silu, expected_silu, rtol=0, atol=0)
    for value, oracle in zip(actual, expected, strict=True):
        torch.testing.assert_close(value, oracle, rtol=0, atol=0)


def test_production_entry_point_never_falls_back_to_torch_on_cpu() -> None:
    binding = _allocate_binding(device="cpu", tokens=1, hidden_size=64, lowrank=16)
    state = torch.zeros((1, 256), dtype=torch.bfloat16)
    weight = torch.zeros((256,), dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="require CUDA"):
        hc.run_grouped_rmsnorm(state, weight, eps=1e-6, binding=binding)


def test_cute_combine_norm_dispatch_contract_is_explicit() -> None:
    from b12x.norm.hyperconnection import _cute_config

    assert _cute_config.is_qwen_combine_norm_contract(
        streams=4,
        hidden_size=2560,
    )
    assert _cute_config.supports_combine_norm(
        streams=4,
        hidden_size=2560,
    )
    assert not _cute_config.supports_combine_norm(
        streams=3,
        hidden_size=2560,
    )


def test_bind_rejects_overlapping_outputs() -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    storage = torch.empty((64,), dtype=torch.bfloat16)
    normalized = storage.view(2, 32)
    with pytest.raises(ValueError, match="must not overlap"):
        hc.bind(
            plan,
            normalized=normalized,
            bottleneck=storage[:16].view(2, 8),
            block_input=torch.empty((2, 8), dtype=torch.bfloat16),
        )


@pytest.mark.parametrize("overlap_name", ["bottleneck", "block_input"])
def test_bind_rejects_normalized_live_range_alias(overlap_name: str) -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    storage = torch.empty((64,), dtype=torch.bfloat16)
    normalized = storage.view(2, 32)
    outputs = {
        "normalized": normalized,
        "bottleneck": torch.empty((2, 8), dtype=torch.bfloat16),
        "block_input": torch.empty((2, 8), dtype=torch.bfloat16),
    }
    outputs[overlap_name] = storage[:16].view(2, 8)
    with pytest.raises(ValueError, match=f"normalized and {overlap_name}"):
        hc.bind(plan, **outputs)


def test_bind_allows_outputs_with_disjoint_live_ranges_to_share_storage() -> None:
    caps = hc.Caps(device="cpu", max_tokens=2, hidden_size=8, lowrank=8)
    plan = hc.plan(caps)
    shared = torch.empty((2, 8), dtype=torch.bfloat16)
    binding = hc.bind(
        plan,
        normalized=torch.empty((2, 32), dtype=torch.bfloat16),
        bottleneck=shared,
        block_input=shared,
    )
    assert binding.bottleneck.data_ptr() == binding.block_input.data_ptr()


def test_live_launches_preserve_capacity_tails_and_read_only_inputs() -> None:
    device = require_sm120()
    tokens, capacity, streams, hidden_size, lowrank = 2, 5, 4, 2560, 320
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.bfloat16, device=device).contiguous()

    outputs = {
        "normalized": torch.full(
            (capacity, width), 7.0, dtype=torch.bfloat16, device=device
        ),
        "bottleneck": torch.full(
            (capacity, lowrank), 7.0, dtype=torch.bfloat16, device=device
        ),
        "block_input": torch.full(
            (capacity, hidden_size), 7.0, dtype=torch.bfloat16, device=device
        ),
    }
    binding = hc.bind(plan, tokens=tokens, **outputs)
    state = randn((tokens, width))
    norm_weight = randn((width,))
    projected_down = randn((tokens, lowrank))
    gate_logits = randn((tokens, width))
    block_output = randn((tokens, hidden_size))
    injection_logits = randn((tokens, streams))
    read_only = {
        "state": state,
        "norm_weight": norm_weight,
        "projected_down": projected_down,
        "gate_logits": gate_logits,
        "block_output": block_output,
        "injection_logits": injection_logits,
    }
    read_only_before = {name: tensor.clone() for name, tensor in read_only.items()}
    tails_before = {name: tensor[tokens:].clone() for name, tensor in outputs.items()}

    normalized = hc.run_grouped_rmsnorm(state, norm_weight, eps=1e-6, binding=binding)
    hc.run_scaled_silu(projected_down, binding=binding)
    hc.run_gate_mean(normalized, gate_logits, binding=binding)
    hc.run_combine(state, block_output, injection_logits, plan=plan)
    hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        eps=1e-6,
        plan=plan,
    )
    torch.cuda.synchronize(device)

    for name, before in read_only_before.items():
        torch.testing.assert_close(read_only[name], before, rtol=0, atol=0)
    for name, before in tails_before.items():
        torch.testing.assert_close(outputs[name][tokens:], before, rtol=0, atol=0)


def test_grouped_norm_reuses_distinct_stream_weights_for_every_token() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 3, 4, 2560
    width = streams * hidden_size
    state = (
        torch.linspace(
            -1.0,
            1.0,
            tokens * width,
            dtype=torch.float32,
            device=device,
        )
        .view(tokens, width)
        .to(torch.bfloat16)
    )
    stream_weights = torch.tensor(
        [0.0, 0.125, -0.25, 0.5],
        dtype=torch.bfloat16,
        device=device,
    )
    weight = stream_weights.repeat_interleave(hidden_size).contiguous()
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
    )
    actual = hc.run_grouped_rmsnorm(state, weight, eps=1e-6, binding=binding)
    expected = hc.reference.grouped_rmsnorm(state, weight, streams=streams, eps=1e-6)
    torch.testing.assert_close(actual, expected, rtol=0, atol=2e-2)


@pytest.mark.parametrize("tokens", [1, 3])
def test_target_kernels_match_reference(tokens: int) -> None:
    device = require_sm120()
    generator = torch.Generator(device="cpu").manual_seed(83400 + tokens)
    streams, hidden_size, lowrank = 4, 2560, 320
    width = streams * hidden_size

    def randn(shape: tuple[int, ...], divisor: float = 1.0) -> torch.Tensor:
        return (
            torch.randn(shape, generator=generator, dtype=torch.float32)
            .div(divisor)
            .to(device=device, dtype=torch.bfloat16)
            .contiguous()
        )

    state = randn((tokens, width), 3.0)
    norm_weight = randn((width,), 32.0)
    projected_down = randn((tokens, lowrank), 2.0)
    gate_logits = randn((tokens, width), 2.0)
    block_output = randn((tokens, hidden_size), 4.0)
    injection_logits = randn((tokens, streams), 2.0)
    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
        lowrank=lowrank,
    )
    eps = 1e-6

    normalized = hc.run_grouped_rmsnorm(state, norm_weight, eps=eps, binding=binding)
    normalized_ref = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=eps
    )
    torch.testing.assert_close(normalized, normalized_ref, rtol=0, atol=2e-2)

    bottleneck = hc.run_scaled_silu(projected_down, binding=binding)
    bottleneck_ref = hc.reference.scaled_silu(projected_down, streams=streams)
    torch.testing.assert_close(bottleneck, bottleneck_ref, rtol=0, atol=8e-3)

    block_input = hc.run_gate_mean(normalized, gate_logits, binding=binding)
    block_input_ref = hc.reference.gate_mean(normalized, gate_logits, streams=streams)
    torch.testing.assert_close(block_input, block_input_ref, rtol=0, atol=8e-3)

    combined = hc.run_combine(state, block_output, injection_logits, plan=binding.plan)
    combined_ref = hc.reference.combine(
        state, block_output, injection_logits, streams=streams
    )
    torch.testing.assert_close(combined, combined_ref, rtol=0, atol=8e-3)
    combined, next_normalized = hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        eps=eps,
        plan=binding.plan,
    )
    combined_ref, next_normalized_ref = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    torch.testing.assert_close(combined, combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(next_normalized, next_normalized_ref, rtol=0, atol=2e-2)


def test_cute_backend_is_callable_alongside_triton_and_graph_stable() -> None:
    from b12x.norm.hyperconnection import _cute, _kernels

    device = require_sm120()
    tokens, streams, hidden_size, lowrank = 1, 4, 2560, 320
    width = streams * hidden_size

    def randn(shape: tuple[int, ...], divisor: float = 1.0) -> torch.Tensor:
        return (
            torch.randn(shape, dtype=torch.float32, device=device)
            .div_(divisor)
            .to(torch.bfloat16)
            .contiguous()
        )

    state = randn((tokens, width), 3.0)
    norm_weight = randn((width,), 32.0)
    projected_down = randn((tokens, lowrank), 2.0)
    gate_logits = randn((tokens, width), 2.0)
    block_output = randn((tokens, hidden_size), 4.0)
    injection_logits = randn((tokens, streams), 2.0)
    outputs = {
        "normalized": torch.empty_like(state),
        "bottleneck": torch.empty_like(projected_down),
        "block_input": torch.empty_like(block_output),
        "combined": torch.empty_like(state),
        "fused_combined": torch.empty_like(state),
        "next_normalized": torch.empty_like(state),
    }
    eps = 1.0e-6

    def launch_cute() -> None:
        _cute.grouped_rmsnorm(
            state,
            norm_weight,
            outputs["normalized"],
            eps=eps,
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.scaled_silu(
            projected_down,
            outputs["bottleneck"],
            streams=streams,
        )
        _cute.gate_mean(
            outputs["normalized"],
            gate_logits,
            outputs["block_input"],
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.combine(
            state,
            block_output,
            injection_logits,
            outputs["combined"],
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.combine_norm(
            state,
            block_output,
            injection_logits,
            norm_weight,
            outputs["fused_combined"],
            outputs["next_normalized"],
            eps=eps,
            streams=streams,
            hidden_size=hidden_size,
        )

    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
        lowrank=lowrank,
    )
    triton_normalized = hc.run_grouped_rmsnorm(
        state,
        norm_weight,
        eps=eps,
        binding=binding,
    )
    assert _kernels.__name__.endswith("._kernels")

    launch_cute()
    torch.cuda.synchronize(device)
    normalized_ref = hc.reference.grouped_rmsnorm(
        state,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    bottleneck_ref = hc.reference.scaled_silu(projected_down, streams=streams)
    block_input_ref = hc.reference.gate_mean(
        normalized_ref,
        gate_logits,
        streams=streams,
    )
    combined_ref, next_normalized_ref = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=eps,
    )
    torch.testing.assert_close(triton_normalized, normalized_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(outputs["normalized"], normalized_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(outputs["bottleneck"], bottleneck_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(
        outputs["block_input"], block_input_ref, rtol=0, atol=8e-3
    )
    torch.testing.assert_close(outputs["combined"], combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(
        outputs["fused_combined"], combined_ref, rtol=0, atol=8e-3
    )
    torch.testing.assert_close(
        outputs["next_normalized"], next_normalized_ref, rtol=0, atol=2e-2
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch_cute()
    addresses = tuple(tensor.data_ptr() for tensor in outputs.values())
    for tensor in outputs.values():
        tensor.fill_(float("nan"))
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert tuple(tensor.data_ptr() for tensor in outputs.values()) == addresses
    assert allocated_after == allocated_before
    torch.testing.assert_close(outputs["normalized"], normalized_ref, rtol=0, atol=2e-2)
    torch.testing.assert_close(outputs["bottleneck"], bottleneck_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(
        outputs["block_input"], block_input_ref, rtol=0, atol=8e-3
    )
    torch.testing.assert_close(outputs["combined"], combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(
        outputs["fused_combined"], combined_ref, rtol=0, atol=8e-3
    )
    torch.testing.assert_close(
        outputs["next_normalized"], next_normalized_ref, rtol=0, atol=2e-2
    )


def test_cute_reference_helpers_reuse_binaries_across_live_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.norm.hyperconnection import _cute

    device = require_sm120()
    streams, hidden_size, lowrank = 4, 63, 17
    width = streams * hidden_size
    eps = 1.0e-6
    compile_keys: list[tuple[object, ...]] = []
    original_compile = _cute._compile

    def traced_compile(
        key: tuple[object, ...], *args: object, **kwargs: object
    ) -> object:
        compile_keys.append(key)
        return original_compile(key, *args, **kwargs)

    def launch(tokens: int) -> None:
        state = (
            torch.randn((tokens, width), dtype=torch.float32, device=device)
            .div_(4)
            .to(torch.bfloat16)
            .contiguous()
        )
        weight = (
            torch.randn((width,), dtype=torch.float32, device=device)
            .div_(32)
            .to(torch.bfloat16)
            .contiguous()
        )
        projected = torch.randn(
            (tokens, lowrank), dtype=torch.bfloat16, device=device
        ).contiguous()
        gate_logits = torch.randn_like(state)
        block_output = torch.randn(
            (tokens, hidden_size), dtype=torch.bfloat16, device=device
        ).contiguous()
        injection_logits = torch.randn(
            (tokens, streams), dtype=torch.bfloat16, device=device
        ).contiguous()
        normalized = torch.empty_like(state)
        bottleneck = torch.empty_like(projected)
        block_input = torch.empty_like(block_output)
        combined = torch.empty_like(state)
        fused_combined = torch.empty_like(state)
        next_normalized = torch.empty_like(state)

        _cute.grouped_rmsnorm(
            state,
            weight,
            normalized,
            eps=eps,
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.scaled_silu(projected, bottleneck, streams=streams)
        _cute.gate_mean(
            normalized,
            gate_logits,
            block_input,
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.combine(
            state,
            block_output,
            injection_logits,
            combined,
            streams=streams,
            hidden_size=hidden_size,
        )
        _cute.combine_norm(
            state,
            block_output,
            injection_logits,
            weight,
            fused_combined,
            next_normalized,
            eps=eps,
            streams=streams,
            hidden_size=hidden_size,
        )
        torch.cuda.synchronize(device)

        normalized_ref = hc.reference.grouped_rmsnorm(
            state,
            weight,
            streams=streams,
            eps=eps,
        )
        combined_ref, next_normalized_ref = hc.reference.combine_norm(
            state,
            block_output,
            injection_logits,
            weight,
            streams=streams,
            eps=eps,
        )
        torch.testing.assert_close(normalized, normalized_ref, rtol=0, atol=2e-2)
        torch.testing.assert_close(
            bottleneck,
            hc.reference.scaled_silu(projected, streams=streams),
            rtol=0,
            atol=8e-3,
        )
        torch.testing.assert_close(
            block_input,
            hc.reference.gate_mean(
                normalized_ref,
                gate_logits,
                streams=streams,
            ),
            rtol=0,
            atol=8e-3,
        )
        torch.testing.assert_close(combined, combined_ref, rtol=0, atol=8e-3)
        torch.testing.assert_close(fused_combined, combined_ref, rtol=0, atol=8e-3)
        torch.testing.assert_close(
            next_normalized,
            next_normalized_ref,
            rtol=0,
            atol=2e-2,
        )

    _cute.clear_caches()
    monkeypatch.setattr(_cute, "_compile", traced_compile)
    try:
        launch(1)
        with kernel_resolution_guard('HyperConnection live counts must reuse warmed CuTe binaries'):
            launch(17)
    finally:
        _cute.clear_caches()

    keys_by_name = {str(key[0]): key for key in compile_keys}
    assert len(compile_keys) == 5
    assert keys_by_name["norm"][2:] == (streams, hidden_size)
    assert keys_by_name["silu"][2:] == (streams, lowrank)
    assert keys_by_name["gate"][2:] == (streams, hidden_size)
    assert keys_by_name["combine"][2:] == (streams, hidden_size)
    assert keys_by_name["combine_norm"][2:] == (streams, hidden_size)


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty.*:UserWarning")
def test_cute_backend_rejects_cold_cuda_graph_capture() -> None:
    from b12x.norm.hyperconnection import _cute

    device = require_sm120()
    projected = torch.randn((1, 320), dtype=torch.bfloat16, device=device).contiguous()
    output = torch.empty_like(projected)

    _cute.clear_caches()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(
            RuntimeError,
            match="compiled and warm-run before CUDA graph capture",
        ),
        torch.cuda.graph(graph),
    ):
        _cute.scaled_silu(projected, output, streams=4)


def test_cute_backend_uses_tensor_device_for_resolution_and_launch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.norm.hyperconnection import _cute

    require_sm120()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two visible CUDA devices")

    current_index = torch.cuda.current_device()
    target_index = next(
        index for index in range(torch.cuda.device_count()) if index != current_index
    )
    target = torch.device("cuda", target_index)
    projected = torch.randn((1, 320), dtype=torch.bfloat16, device=target).contiguous()
    output = torch.empty_like(projected)
    compile_devices: list[int] = []
    launch_devices: list[int] = []
    original_compile = _cute._compile
    original_run_compiled = _cute.run_compiled

    def traced_compile(*args: object, **kwargs: object) -> object:
        compile_devices.append(torch.cuda.current_device())
        return original_compile(*args, **kwargs)

    def traced_run_compiled(compiled: object, args: tuple[object, ...]) -> None:
        launch_devices.append(torch.cuda.current_device())
        original_run_compiled(compiled, args)

    _cute.clear_caches()
    monkeypatch.setattr(_cute, "_compile", traced_compile)
    monkeypatch.setattr(_cute, "run_compiled", traced_run_compiled)
    try:
        _cute.scaled_silu(projected, output, streams=4)
        _cute.scaled_silu(projected, output, streams=4)
        torch.cuda.synchronize(target)
    finally:
        _cute.clear_caches()

    assert torch.cuda.current_device() == current_index
    assert compile_devices == [target_index]
    assert launch_devices == [target_index, target_index]
    torch.testing.assert_close(output, F.silu(projected / 4), rtol=0, atol=8e-3)


def test_combine_norm_cuda_graph_replay_uses_stable_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 2, 4, 2560
    width = streams * hidden_size
    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = torch.nn.Parameter(
        torch.randn((width,), dtype=torch.bfloat16, device=device)
        .div_(32)
        .contiguous(),
        requires_grad=False,
    )
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=hidden_size,
            streams=streams,
        )
    )

    hc.run_combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        eps=1e-6,
        plan=plan,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        combined, normalized = hc.run_combine_norm(
            state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            plan=plan,
        )
    output_addresses = combined.data_ptr(), normalized.data_ptr()

    state.copy_(torch.randn_like(state))
    block_output.copy_(torch.randn_like(block_output))
    injection_logits.copy_(torch.randn_like(injection_logits))
    expected_combined, expected_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        streams=streams,
        eps=1e-6,
    )
    allocated_before = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after = torch.cuda.memory_allocated(device)

    assert (combined.data_ptr(), normalized.data_ptr()) == output_addresses
    assert allocated_after == allocated_before
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=8e-3)
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=2e-2)


def test_combine_norm_reuses_one_cute_binary_across_token_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.norm.hyperconnection import _cute

    device = require_sm120()
    streams, hidden_size = 4, 2560
    width = streams * hidden_size
    compile_keys: list[tuple[object, ...]] = []
    original_compile = _cute._compile

    def traced_compile(
        key: tuple[object, ...], *args: object, **kwargs: object
    ) -> object:
        compile_keys.append(key)
        return original_compile(key, *args, **kwargs)

    _cute.clear_caches()
    monkeypatch.setattr(_cute, "_compile", traced_compile)
    try:
        for tokens in (1, 17):
            state = torch.randn(
                (tokens, width), dtype=torch.bfloat16, device=device
            ).contiguous()
            block_output = torch.randn(
                (tokens, hidden_size), dtype=torch.bfloat16, device=device
            ).contiguous()
            injection_logits = torch.randn(
                (tokens, streams), dtype=torch.bfloat16, device=device
            ).contiguous()
            weight = torch.zeros(
                (width,), dtype=torch.bfloat16, device=device
            ).contiguous()
            plan = hc.plan(
                hc.Caps(
                    device=device,
                    max_tokens=tokens,
                    hidden_size=hidden_size,
                    streams=streams,
                )
            )
            combined, normalized = hc.run_combine_norm(
                state,
                block_output,
                injection_logits,
                weight,
                eps=1.0e-6,
                plan=plan,
            )
            expected = hc.reference.combine_norm(
                state,
                block_output,
                injection_logits,
                weight,
                streams=streams,
                eps=1.0e-6,
            )
            torch.testing.assert_close(combined, expected[0], rtol=0, atol=8e-3)
            torch.testing.assert_close(normalized, expected[1], rtol=0, atol=2e-2)
    finally:
        _cute.clear_caches()

    assert len(compile_keys) == 1
    assert compile_keys[0][0] == "combine_norm_packed"
    assert compile_keys[0][2:] == (streams, hidden_size)


def test_combine_norm_uses_cute_and_rejects_non_qwen_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x.norm.hyperconnection import _cute, _kernels

    device = require_sm120()
    cute_shapes: list[tuple[int, int]] = []

    def fake_cute(
        state: torch.Tensor,
        block_output: torch.Tensor,
        injection_logits: torch.Tensor,
        next_norm_weight: torch.Tensor,
        combined: torch.Tensor,
        normalized: torch.Tensor,
        **kwargs: object,
    ) -> None:
        del block_output, injection_logits, next_norm_weight, kwargs
        cute_shapes.append((int(state.shape[0]), int(state.shape[1])))
        combined.copy_(state)
        normalized.copy_(state)

    monkeypatch.setattr(_cute, "combine_norm", fake_cute)
    for tokens in (4, 64):
        streams, hidden_size = 4, 2560
        width = streams * hidden_size
        plan = hc.plan(
            hc.Caps(
                device=device,
                max_tokens=tokens,
                hidden_size=hidden_size,
                streams=streams,
            )
        )
        hc.run_combine_norm(
            torch.zeros((tokens, width), dtype=torch.bfloat16, device=device),
            torch.zeros((tokens, hidden_size), dtype=torch.bfloat16, device=device),
            torch.zeros((tokens, streams), dtype=torch.bfloat16, device=device),
            torch.nn.Parameter(
                torch.zeros((width,), dtype=torch.bfloat16, device=device),
                requires_grad=False,
            ),
            eps=1.0e-6,
            plan=plan,
        )
    tokens, streams, hidden_size = 3, 4, 64
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=hidden_size,
            streams=streams,
        )
    )
    with pytest.raises(ValueError, match="requires streams=4 and hidden_size=2560"):
        hc.run_combine_norm(
            torch.zeros((tokens, width), dtype=torch.bfloat16, device=device),
            torch.zeros((tokens, hidden_size), dtype=torch.bfloat16, device=device),
            torch.zeros((tokens, streams), dtype=torch.bfloat16, device=device),
            torch.zeros((width,), dtype=torch.bfloat16, device=device),
            eps=1.0e-6,
            plan=plan,
        )
    torch.cuda.synchronize(device)

    assert cute_shapes == [(4, 10240), (64, 10240)]
    assert not hasattr(_kernels, "_combine_norm_kernel")


@pytest.mark.filterwarnings("ignore:The CUDA Graph is empty.*:UserWarning")
def test_public_cute_combine_norm_rejects_cold_cuda_graph_capture() -> None:
    from b12x.norm.hyperconnection import _cute

    device = require_sm120()
    tokens, streams, hidden_size = 1, 4, 2560
    width = streams * hidden_size
    state = torch.zeros((tokens, width), dtype=torch.bfloat16, device=device)
    block_output = torch.zeros(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    )
    injection_logits = torch.zeros(
        (tokens, streams), dtype=torch.bfloat16, device=device
    )
    weight = torch.zeros((width,), dtype=torch.bfloat16, device=device)
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=hidden_size,
            streams=streams,
        )
    )

    _cute.clear_caches()
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(
            RuntimeError,
            match="compiled and warm-run before CUDA graph capture",
        ),
        torch.cuda.graph(graph),
    ):
        hc.run_combine_norm(
            state,
            block_output,
            injection_logits,
            weight,
            eps=1.0e-6,
            plan=plan,
        )


def test_target_full_chain_cuda_graph_replay_uses_stable_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size, lowrank = 3, 4, 2560, 320
    width = streams * hidden_size

    def randn(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, dtype=torch.bfloat16, device=device).contiguous()

    binding = _allocate_binding(
        device=device,
        tokens=tokens,
        hidden_size=hidden_size,
        streams=streams,
        lowrank=lowrank,
    )
    state = randn((tokens, width))
    norm_weight = randn((width,)).div_(32)
    projected_down = randn((tokens, lowrank))
    gate_logits = randn((tokens, width))
    block_output = randn((tokens, hidden_size))
    injection_logits = randn((tokens, streams))

    def launch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        normalized = hc.run_grouped_rmsnorm(
            state, norm_weight, eps=1e-6, binding=binding
        )
        bottleneck = hc.run_scaled_silu(projected_down, binding=binding)
        block_input = hc.run_gate_mean(normalized, gate_logits, binding=binding)
        combined, next_normalized = hc.run_combine_norm(
            state,
            block_output,
            injection_logits,
            norm_weight,
            eps=1e-6,
            plan=binding.plan,
        )
        return bottleneck, block_input, combined, next_normalized

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = launch()
    output_addresses = tuple(tensor.data_ptr() for tensor in captured)

    state.copy_(randn(state.shape))
    projected_down.copy_(randn(projected_down.shape))
    gate_logits.copy_(randn(gate_logits.shape))
    block_output.copy_(randn(block_output.shape))
    injection_logits.copy_(randn(injection_logits.shape))
    normalized_ref = hc.reference.grouped_rmsnorm(
        state, norm_weight, streams=streams, eps=1e-6
    )
    bottleneck_ref = hc.reference.scaled_silu(projected_down, streams=streams)
    block_input_ref = hc.reference.gate_mean(
        normalized_ref, gate_logits, streams=streams
    )
    combined_ref, next_normalized_ref = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        norm_weight,
        streams=streams,
        eps=1e-6,
    )
    allocated_before_replay = torch.cuda.memory_allocated(device)
    graph.replay()
    torch.cuda.synchronize(device)
    allocated_after_replay = torch.cuda.memory_allocated(device)

    assert tuple(tensor.data_ptr() for tensor in captured) == output_addresses
    assert output_addresses[:2] == (
        binding.bottleneck.data_ptr(),
        binding.block_input.data_ptr(),
    )
    assert allocated_after_replay == allocated_before_replay
    torch.testing.assert_close(captured[0], bottleneck_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[1], block_input_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[2], combined_ref, rtol=0, atol=8e-3)
    torch.testing.assert_close(captured[3], next_normalized_ref, rtol=0, atol=2e-2)


def test_combine_norm_torch_compile_fullgraph_returns_live_outputs() -> None:
    device = require_sm120()
    tokens, streams, hidden_size = 2, 4, 2560
    width = streams * hidden_size
    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = (
        torch.randn((width,), dtype=torch.bfloat16, device=device).div_(32).contiguous()
    )
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=tokens,
            hidden_size=hidden_size,
            streams=streams,
        )
    )

    def run(
        live_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hc.run_combine_norm(
            live_state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            plan=plan,
        )

    # Warm the selected kernel specialization before compiling.
    run(state)
    compiled = torch.compile(run, fullgraph=True)

    state.copy_(torch.randn_like(state))
    expected_combined, expected_normalized = hc.reference.combine_norm(
        state,
        block_output,
        injection_logits,
        weight,
        streams=streams,
        eps=1e-6,
    )
    combined, normalized = compiled(state)
    torch.cuda.synchronize(device)

    assert combined.shape == state.shape
    assert normalized.shape == state.shape
    torch.testing.assert_close(combined, expected_combined, rtol=0, atol=8e-3)
    torch.testing.assert_close(normalized, expected_normalized, rtol=0, atol=2e-2)


def test_combine_norm_compile_avoids_recurrent_capacity_writebacks() -> None:
    device = require_sm120()
    tokens, capacity, streams, hidden_size = 2, 4096, 4, 2560
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
        )
    )
    state_capacity = torch.randn(
        (capacity, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    state = state_capacity[:tokens]
    state_before = state.clone()
    block_output = torch.randn(
        (tokens, hidden_size), dtype=torch.bfloat16, device=device
    ).contiguous()
    injection_logits = torch.randn(
        (tokens, streams), dtype=torch.bfloat16, device=device
    ).contiguous()
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device).contiguous()

    def run(live_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined, _ = hc.run_combine_norm(
            live_state,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            plan=plan,
        )
        return hc.run_combine_norm(
            combined,
            block_output,
            injection_logits,
            weight,
            eps=1e-6,
            plan=plan,
        )

    run(state)
    compiled = torch.compile(run, fullgraph=True, dynamic=True)
    compiled(state)
    torch.cuda.synchronize(device)
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
    ) as profiler:
        compiled(state)
        torch.cuda.synchronize(device)

    event_names = {event.name for event in profiler.events()}
    assert any("PackedCombineNorm" in name for name in event_names)
    assert "_combine_norm_kernel" not in event_names
    assert not {name for name in event_names if name.startswith("triton_poi_fused")}
    torch.testing.assert_close(state, state_before, rtol=0, atol=0)


def test_bind_inside_torch_compile_fullgraph_uses_live_views() -> None:
    device = require_sm120()
    tokens, capacity, streams, hidden_size, lowrank = 2, 8, 4, 64, 16
    width = streams * hidden_size
    plan = hc.plan(
        hc.Caps(
            device=device,
            max_tokens=capacity,
            hidden_size=hidden_size,
            streams=streams,
            lowrank=lowrank,
        )
    )
    outputs = {
        "normalized": torch.empty(
            (capacity, width), dtype=torch.bfloat16, device=device
        ),
        "bottleneck": torch.empty(
            (capacity, lowrank), dtype=torch.bfloat16, device=device
        ),
        "block_input": torch.empty(
            (capacity, hidden_size), dtype=torch.bfloat16, device=device
        ),
    }
    weight = torch.randn((width,), dtype=torch.bfloat16, device=device).contiguous()

    # Validate the fixed workspace before compiling its live-prefix binding.
    hc.bind(plan, tokens=capacity, **outputs)

    def run(live_state: torch.Tensor) -> torch.Tensor:
        binding = hc.bind(plan, tokens=live_state.shape[0], **outputs)
        return hc.run_grouped_rmsnorm(
            live_state,
            weight,
            eps=1e-6,
            binding=binding,
        )

    state = torch.randn(
        (tokens, width), dtype=torch.bfloat16, device=device
    ).contiguous()
    run(state)
    compiled = torch.compile(run, fullgraph=True)
    normalized = compiled(state)
    torch.cuda.synchronize(device)

    assert normalized.data_ptr() == outputs["normalized"].data_ptr()


def test_torch_compile_rejects_dynamic_input_aliasing_bound_output() -> None:
    device = require_sm120()
    binding = _allocate_binding(
        device=device,
        tokens=1,
        hidden_size=64,
        lowrank=16,
    )
    projected_down = torch.randn(
        (1, 16), dtype=torch.bfloat16, device=device
    ).contiguous()

    def launch(value: torch.Tensor) -> torch.Tensor:
        return hc.run_scaled_silu(value, binding=binding)

    launch(projected_down)
    compiled = torch.compile(launch, fullgraph=True)
    compiled(projected_down)
    with pytest.raises(ValueError, match="bottleneck must not overlap projected_down"):
        compiled(binding.bottleneck)


def test_stateless_swiglu_asymmetric_clamp_and_vision_rounding():
    device = require_sm120()
    gate = torch.tensor(
        [-20.0, -2.0, -0.5, 0.5, 1.5, 20.0], device=device, dtype=torch.bfloat16,
    )
    up = torch.tensor(
        [20.0, -20.0, 1.75, 1.75, -1.75, 20.0], device=device, dtype=torch.bfloat16,
    )
    gate_up = torch.cat((gate, up)).reshape(1, -1)
    out = torch.empty(1, gate.numel(), device=device, dtype=torch.bfloat16)
    hc.run_swiglu(gate_up, limit=2.0, out=out)
    expected = (
        F.silu(gate.float().clamp(max=2.0)) * up.float().clamp(-2.0, 2.0)
    ).bfloat16().reshape_as(out)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    # The negative gate remains -20, rather than being symmetrically clamped.
    assert out[0, 0].abs() < 1e-6

    # Sweep exactly representable BF16 values, including rounding boundaries.
    gates = torch.linspace(-4, 4, 1024, device=device).bfloat16().reshape(1, -1)
    ups = torch.full_like(gates, 1.75)
    merged = torch.cat((gates, ups), dim=1)
    shared = torch.empty_like(gates)
    vision = torch.empty_like(gates)
    hc.run_swiglu(merged, limit=float("inf"), out=shared)
    hc.run_swiglu(merged, limit=float("inf"), out=vision, round_silu=True)
    expected_shared = (F.silu(gates.float()) * ups.float()).bfloat16()
    expected_vision = F.silu(gates) * ups
    torch.testing.assert_close(shared, expected_shared, rtol=0, atol=0)
    torch.testing.assert_close(vision, expected_vision, rtol=0, atol=0)
    assert bool((expected_shared != expected_vision).any())


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float32])
def test_stateless_add_retains_fp32_before_cancellation(output_dtype):
    device = require_sm120()
    left = torch.tensor([[256.5, -256.5, 0.00390625]], device=device)
    right = torch.tensor([[-256.0, 256.0, 1.0]], device=device, dtype=torch.bfloat16)
    out = torch.empty_like(left, dtype=output_dtype)
    result = hc.run_add(left, right, out=out)
    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(out, (left + right.float()).to(output_dtype), rtol=0, atol=0)
    assert out[0, 0] == 0.5 and out[0, 1] == -0.5


def test_stateless_tails_frozen_live_counts_and_graph_mutation(request):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    device = require_sm120()
    capacity, width = 9, 257
    gate_up = torch.randn(capacity, 2 * width, device=device, dtype=torch.bfloat16)
    residual = torch.randn(capacity, width, device=device, dtype=torch.bfloat16)
    routed = torch.randn(capacity, width, device=device)
    shared = torch.empty_like(residual)
    vision = torch.empty_like(residual)
    added = torch.empty_like(residual)
    vision_added = torch.empty_like(residual)
    confidence = torch.empty_like(routed)

    def launch(rows):
        hc.run_swiglu(gate_up[:rows], limit=2.0, out=shared[:rows])
        hc.run_swiglu(
            gate_up[:rows], limit=float("inf"), out=vision[:rows], round_silu=True,
        )
        hc.run_add(routed[:rows], shared[:rows], out=added[:rows])
        hc.run_add(residual[:rows], vision[:rows], out=vision_added[:rows])
        hc.run_sigmoid(routed[:rows], out=confidence[:rows])

    launch(capacity)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("stateless activation/add live row reuse")
    outputs = (shared, vision, added, vision_added, confidence)
    pointers = tuple(t.data_ptr() for t in outputs)
    for rows in (0, 1, 7, capacity):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch(rows)
        gate_up.add_(0.125)
        routed.add_(0.25)
        residual.add_(0.5)
        for tensor in outputs:
            tensor.fill_(float("nan"))
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        graph.replay()
        torch.cuda.synchronize(device)
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations
        assert tuple(t.data_ptr() for t in outputs) == pointers
        gate, up = gate_up[:rows].chunk(2, dim=1)
        shared_ref = (
            F.silu(gate.float().clamp(max=2.0)) * up.float().clamp(-2.0, 2.0)
        ).bfloat16()
        vision_ref = F.silu(gate) * up
        references = (
            shared_ref,
            vision_ref,
            (routed[:rows] + shared_ref.float()).bfloat16(),
            (residual[:rows].float() + vision_ref.float()).bfloat16(),
            routed[:rows].sigmoid(),
        )
        for actual, expected in zip(outputs, references, strict=True):
            torch.testing.assert_close(actual[:rows], expected, rtol=1e-5, atol=0.008)
            assert bool(torch.isnan(actual[rows:]).all())


def test_ordinary_rmsnorm_preserves_fp32_weights_and_pointer_abi(request):
    from b12x import freeze_kernel_resolution, unfreeze_kernel_resolution

    device = require_sm120()
    capacity, streams, hidden = 7, 4, 256
    width = streams * hidden
    binding = _allocate_binding(
        device=device, tokens=capacity, hidden_size=hidden, streams=streams, lowrank=16,
    )
    state = torch.linspace(-2.0, 3.0, capacity * width, device=device).reshape(
        capacity, width,
    ).bfloat16()
    weights = torch.linspace(0.9931, 1.0137, width, device=device)
    rounded = weights.bfloat16()
    assert bool((weights != rounded.float()).any())

    def launch(rows, weight):
        live = binding.plan.bind(
            tokens=rows, normalized=binding.normalized_capacity,
            bottleneck=binding.bottleneck_capacity, block_input=binding.block_input_capacity,
        )
        return hc.run_grouped_rmsnorm(
            state[:rows], weight, eps=1e-6, binding=live, zero_centered=False,
        )

    # Warming both dtypes must produce distinct pointer-ABI specializations.
    launch(capacity, rounded)
    launch(capacity, weights)
    torch.cuda.synchronize(device)
    request.addfinalizer(unfreeze_kernel_resolution)
    freeze_kernel_resolution("ordinary RMSNorm BF16/FP32 weight ABI")
    for rows in (1, 3, capacity):
        for weight in (rounded, weights):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = launch(rows, weight)
            state.add_(0.03125)
            weight.add_(0.0003)
            graph.replay()
            grouped = state[:rows].float().reshape(rows, streams, hidden)
            normalized = (
                grouped * torch.rsqrt(grouped.square().mean(-1, keepdim=True) + 1e-6)
            ).reshape(rows, width)
            expected = (normalized * weight.float()).bfloat16()
            torch.testing.assert_close(output, expected, rtol=0, atol=0.0001)
            if weight.dtype == torch.float32:
                rounded_expected = (normalized * weight.bfloat16().float()).bfloat16()
                assert bool((output != rounded_expected).any())
    with pytest.raises(ValueError, match="dtype"):
        hc.run_grouped_rmsnorm(
            state, weights, eps=1e-6, binding=binding, zero_centered=True,
        )
