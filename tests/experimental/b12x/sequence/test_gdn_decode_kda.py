from __future__ import annotations

import gc
from dataclasses import replace

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall, require_prepared
from b12x._lib.runtime_control import kernel_resolution_guard
from b12x.sequence import gdn_decode as gdn

from ..conftest import require_b12x as require_sm120


def _randn(
    shape: tuple[int, ...],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 0.25,
) -> torch.Tensor:
    return (
        torch.randn(shape, dtype=torch.float32, device=device)
        .mul_(scale)
        .to(dtype)
        .contiguous()
    )


_PREPARATIONS: list[tuple[object, PreparationSession]] = []


@pytest.fixture(autouse=True)
def _prepared_scopes():
    try:
        yield
    finally:
        gc.collect()
        while _PREPARATIONS:
            result, session = _PREPARATIONS.pop()
            result.close()
            session.close()


def _make_case(
    *,
    device: torch.device,
    query_lengths: tuple[int, ...] = (3, 1),
    heads: int = 2,
    columns: int = 3,
    max_tokens: int = 5,
    tensor_tokens: int | None = None,
    tensor_columns: int | None = None,
    state_dtype: torch.dtype = torch.float32,
    state_indices_dtype: torch.dtype = torch.int64,
    null_state_index: int | None = None,
    noncontiguous_beta: bool = False,
    recurrent_block_v: int | None = None,
) -> gdn.KdaBinding:
    max_seqs, live_tokens = len(query_lengths), sum(query_lengths)
    token_capacity = max_tokens if tensor_tokens is None else tensor_tokens
    column_capacity = columns if tensor_columns is None else tensor_columns
    caps = gdn.Caps(
        device=device, max_tokens=max_tokens, max_seqs=max_seqs,
        max_state_slots=max_seqs * columns + 1, key_heads=heads, value_heads=heads,
        state_index_columns=columns, state_dtype=state_dtype, gate_activation="sigmoid",
        null_state_index=null_state_index,
    )
    query_start_loc = torch.tensor(
        [0, *torch.tensor(query_lengths).cumsum(0).tolist()], dtype=torch.int32, device=device,
    )
    raw_beta = _randn((token_capacity, heads), device=device)
    if noncontiguous_beta:
        storage = torch.empty((token_capacity, heads + 3), dtype=raw_beta.dtype, device=device)
        storage[:, :heads].copy_(raw_beta)
        raw_beta = storage[:, :heads]
    args = {
        "mixed_qkv": _randn((token_capacity, caps.packed_qkv_width), device=device),
        "raw_g": _randn((token_capacity, heads, 128), device=device),
        "raw_beta": raw_beta, "z": _randn((token_capacity, heads, 128), device=device),
        "A_log": _randn((heads,), device=device, dtype=torch.float32, scale=0.1),
        "dt_bias": _randn((heads, 128), device=device, dtype=torch.float32, scale=0.1),
        "norm_weight": (1.0 + _randn((128,), device=device, dtype=torch.float32, scale=0.05)).contiguous(),
        "recurrent_state": _randn(
            (max_seqs * columns + 1, heads, 128, 128), device=device, dtype=state_dtype, scale=0.1,
        ),
        "query_start_loc": query_start_loc,
        "num_accepted_tokens": torch.tensor(
            [min(2, length, column_capacity) for length in query_lengths], dtype=torch.int32, device=device,
        ),
        "state_indices": torch.arange(
            max_seqs * column_capacity, dtype=state_indices_dtype, device=device
        ).view(max_seqs, column_capacity),
        "num_seqs": torch.tensor([max_seqs], dtype=torch.int32, device=device),
        "num_tokens": torch.tensor([live_tokens], dtype=torch.int32, device=device),
        "output": torch.empty((token_capacity, heads, 128), dtype=torch.bfloat16, device=device),
    }
    override = gdn.GdnConfig(backend="triton", recurrent_block_v=recurrent_block_v) if recurrent_block_v else None
    plan = gdn.plan(caps, invocation=gdn.invocation_from_tensors(caps, **args), override=override)

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        binding = state.bind_kda(
            scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **args
        )
        original_state, original_output = binding.recurrent_state.clone(), binding.output.clone()

        def reset():
            binding.recurrent_state.copy_(original_state)
            binding.output.copy_(original_output)

        return PreparedCall(
            run=lambda: state.run(binding, lower_bound=-5.0),
            output=binding.output, reset=reset, restore=reset,
        )

    request = plan.request(
        name=f"kda-{id(args)}", prepare_call=prepare_call,
    )
    session = PreparationSession(device=device, autotune=False)
    result = session.prepare((request,))
    state = require_prepared(plan, "attention.gdn")
    (spec,) = state.layout.scratch_specs()
    _PREPARATIONS.append((result, session))
    return gdn.bind_kda(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **args
    )


def _prepare_binding(
    caps: gdn.Caps, args: dict[str, torch.Tensor], *, override: gdn.GdnConfig | None = None
) -> gdn.KdaBinding:
    device = caps.device
    plan = gdn.plan(caps, invocation=gdn.invocation_from_tensors(caps, **args), override=override)

    def prepare_call(state):
        (spec,) = state.layout.scratch_specs()
        binding = state.bind_kda(
            scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **args
        )
        original_state, original_output = binding.recurrent_state.clone(), binding.output.clone()

        def reset():
            binding.recurrent_state.copy_(original_state)
            binding.output.copy_(original_output)

        return PreparedCall(
            run=lambda: state.run(binding, lower_bound=-5.0),
            output=binding.output, reset=reset, restore=reset,
        )

    request = plan.request(
        name=f"kda-{id(args)}", prepare_call=prepare_call,
    )
    session = PreparationSession(device=device, autotune=False)
    result = session.prepare((request,))
    state = require_prepared(plan, "attention.gdn")
    (spec,) = state.layout.scratch_specs()
    _PREPARATIONS.append((result, session))
    return gdn.bind_kda(
        plan, scratch=torch.empty(spec.shape, dtype=spec.dtype, device=device), **args
    )


def _rebind(binding: gdn.KdaBinding, **overrides: torch.Tensor) -> gdn.KdaBinding:
    arguments = {
        "scratch": binding.scratch, "mixed_qkv": binding.mixed_qkv, "raw_g": binding.raw_g,
        "raw_beta": binding.raw_beta, "z": binding.z, "A_log": binding.A_log,
        "dt_bias": binding.dt_bias, "norm_weight": binding.norm_weight,
        "recurrent_state": binding.recurrent_state, "query_start_loc": binding.query_start_loc,
        "num_accepted_tokens": binding.num_accepted_tokens, "state_indices": binding.state_indices,
        "num_seqs": binding.num_seqs, "num_tokens": binding.num_tokens, "output": binding.output,
    }
    arguments.update(overrides)
    return gdn.bind_kda(binding.plan, **arguments)


def _row_padded(tensor: torch.Tensor, padding: int = 3) -> torch.Tensor:
    rows, row_elements = tensor.shape[0], tensor[0].numel()
    storage = torch.empty((rows, row_elements + padding), dtype=tensor.dtype, device=tensor.device)
    result = storage[:, :row_elements].view(tensor.shape)
    result.copy_(tensor)
    return result


def _reference(binding: gdn.KdaBinding, state: torch.Tensor) -> torch.Tensor:
    caps = binding._state.caps
    return gdn.reference.decode_kda(
        binding.mixed_qkv, binding.raw_g, binding.raw_beta, binding.z, binding.A_log,
        binding.dt_bias, binding.norm_weight, state, binding.query_start_loc,
        binding.num_accepted_tokens, binding.state_indices, binding.num_seqs, binding.num_tokens,
        heads=caps.value_heads, qk_l2norm=caps.qk_l2norm, null_state_index=caps.null_state_index,
    )


@pytest.mark.parametrize("columns", [1, 4, 8])
@pytest.mark.parametrize("block_v", [16, 32])
def test_recovery_preserves_outputs_and_accepted_fp32_checkpoints(columns, block_v, record_property, monkeypatch):
    """Record production must not mutate checkpoints or commit rejected tokens."""
    device = require_sm120()
    torch.manual_seed(591)
    baseline = _make_case(
        device=device, heads=32, columns=columns, query_lengths=(columns, 1),
        max_tokens=2 * columns, noncontiguous_beta=True, null_state_index=0,
    )
    baseline.state_indices.copy_(torch.arange(1, 2 * columns + 1, device=device).view(2, columns))
    baseline.num_accepted_tokens.fill_(1)
    before = baseline.recurrent_state.clone()
    expected_state = before.clone()
    expected = _reference(baseline, expected_state)
    caps = replace(baseline._state.caps, recover_speculative_state=True)
    args = {name: getattr(baseline, name) for name in (
        "mixed_qkv", "raw_g", "raw_beta", "z", "A_log", "dt_bias", "norm_weight",
        "recurrent_state", "query_start_loc", "num_accepted_tokens", "num_seqs", "num_tokens", "output",
    )}
    args.update(
        state_indices=baseline.state_indices[:, :1],
        correction_cache=torch.empty((caps.max_state_slots, 32, columns, 128), device=device),
        kg_cache=torch.empty((caps.max_state_slots, 32, columns, 256), dtype=torch.bfloat16, device=device),
    )
    binding = _prepare_binding(caps, args, override=gdn.GdnConfig(backend="cutedsl", recurrent_block_v=block_v))
    with pytest.raises(ValueError, match=rf"{2 * columns} > 1 \* {columns}"):
        gdn.bind_kda(
            binding.plan, scratch=binding.scratch,
            **{**args, "state_indices": binding.state_indices[:1]},
        )
    def unexpected_revalidation(*_args):
        raise AssertionError("Recovery run must use the already validated binding")

    monkeypatch.setattr(type(require_prepared(binding.plan, "attention.gdn")), "_check", unexpected_revalidation)
    gdn.run_kda(binding)
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)
    actual = binding.output[:columns + 1].float()
    torch.testing.assert_close(actual, expected[:columns + 1].float(), atol=0.016, rtol=0.012)
    relative_output_error = (actual - expected[:columns + 1].float()).norm() / expected[:columns + 1].float().norm()
    assert relative_output_error < 0.003
    record_property("output_relative_l2", relative_output_error.item())

    def table(value):
        return torch.tensor([value], dtype=torch.int64, device=device)

    lengths = torch.tensor([columns, 1], dtype=torch.int32, device=device)
    final = binding.state_indices[:, 0].to(torch.int32).contiguous()
    boundary = torch.zeros(2, dtype=torch.int32, device=device)
    boundary_lens = torch.zeros_like(boundary)
    if columns > 1:
        boundary[0] = 2
        boundary_lens[0] = 2
    commit = gdn.bind_kda_commit(
        binding.plan, scratch=binding.scratch,
        state_base_addrs=table(binding.recurrent_state.data_ptr()),
        state_block_strides=table(binding.recurrent_state.stride(0)),
        correction_cache_base_addrs=table(binding.correction_cache.data_ptr()),
        correction_cache_block_strides=table(binding.correction_cache.stride(0)),
        kg_cache_base_addrs=table(binding.kg_cache.data_ptr()),
        kg_cache_block_strides=table(binding.kg_cache.stride(0)),
        A_log=binding.A_log.unsqueeze(0), dt_bias=binding.dt_bias.unsqueeze(0),
        state_indices=binding.state_indices[:, 0], commit_lens=lengths,
        final_state_indices=final, boundary_state_indices=boundary,
        boundary_recovery_lens=boundary_lens,
    )
    gdn.run_kda_commit(commit)
    record_property("accepted_state_max_abs", (binding.recurrent_state[1] - expected_state[columns]).abs().max().item())
    torch.testing.assert_close(binding.recurrent_state[1], expected_state[columns], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(binding.recurrent_state[columns + 1], expected_state[columns + 1], rtol=1e-5, atol=1e-6)
    if columns > 1:
        torch.testing.assert_close(binding.recurrent_state[2], expected_state[2], rtol=1e-5, atol=1e-6)
        binding.recurrent_state.copy_(before)
        boundary[0] = final[0]
        boundary_lens[0] = 1
        gdn.run_kda_commit(commit)
        torch.testing.assert_close(binding.recurrent_state[1], expected_state[columns], rtol=1e-5, atol=1e-6)

    binding.recurrent_state.copy_(before)
    graph = torch.cuda.CUDAGraph()
    with kernel_resolution_guard("KDA recovery graph"), torch.cuda.graph(graph):
        gdn.run_kda(binding)
        gdn.run_kda_commit(commit)
    for accepted in (0, 1, columns):
        lengths[0] = accepted
        boundary.zero_()
        binding.recurrent_state.copy_(before)
        with kernel_resolution_guard("KDA recovery replay"):
            allocated = torch.cuda.memory_allocated()
            graph.replay()
            assert torch.cuda.memory_allocated() == allocated
        reference = before[1] if accepted == 0 else expected_state[accepted]
        torch.testing.assert_close(binding.recurrent_state[1], reference, rtol=1e-5, atol=1e-6)

    # Weak forgetting retains information across windows and exposes drift that
    # a single, strongly decayed random state can hide.
    binding.raw_g.fill_(-8)
    baseline.recurrent_state.copy_(before)
    recurrent_reference = before.clone()
    lengths.copy_(torch.tensor([columns, 1], device=device, dtype=torch.int32))
    for _ in range(16):
        _reference(baseline, recurrent_reference)
        recurrent_reference[1].copy_(recurrent_reference[columns])
        gdn.run_kda(binding)
        gdn.run_kda_commit(commit)
    record_property("persistent_state_max_abs", (binding.recurrent_state[1] - recurrent_reference[1]).abs().max().item())
    torch.testing.assert_close(binding.recurrent_state[1], recurrent_reference[1], rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("large_stride", [False, True], ids=["interleaved-pages", "int64-pool-offset"])
def test_recovery_reads_live_paged_fields_without_aliasing(large_stride):
    """Both kernels must handle interleaved fields and offsets exceeding 2^31."""
    device = require_sm120()
    baseline = _make_case(device=device, heads=1, columns=1, query_lengths=(1,), max_tokens=1, null_state_index=0)
    baseline.state_indices.fill_(1)
    before = baseline.recurrent_state.clone()
    expected_state = before.clone()
    expected_output = _reference(baseline, expected_state)
    page_bytes = 4 * ((1 << 31) + 32768) if large_stride else 67584
    # One allocation shares the untouched gap and fits a 16 GiB GPU. Each
    # field's second page still starts beyond the Int32 element-offset limit.
    storage = torch.empty((page_bytes + 67584,), dtype=torch.uint8, device=device)
    if not large_stride:
        storage.fill_(123)
    checkpoint = storage.view(torch.float32).as_strided((2, 1, 128, 128), (page_bytes // 4, 16384, 128, 1))
    correction = storage[65536:].view(torch.float32).as_strided((2, 1, 1, 128), (page_bytes // 4, 128, 128, 1))
    kg = storage[66048:].view(torch.bfloat16).as_strided((2, 1, 1, 256), (page_bytes // 2, 256, 256, 1))
    if large_stride:
        assert all(tensor.stride(0) > 1 << 31 for tensor in (checkpoint, correction, kg))
    checkpoint.copy_(before)
    args = {name: getattr(baseline, name) for name in (
        "mixed_qkv", "raw_g", "raw_beta", "z", "A_log", "dt_bias", "norm_weight",
        "query_start_loc", "num_accepted_tokens", "state_indices", "num_seqs", "num_tokens", "output",
    )}
    args.update(recurrent_state=checkpoint, correction_cache=correction, kg_cache=kg)
    binding = _prepare_binding(replace(baseline._state.caps, recover_speculative_state=True), args)
    gdn.run_kda(binding)
    torch.testing.assert_close(checkpoint, before, rtol=0, atol=0)
    torch.testing.assert_close(binding.output.float(), expected_output.float(), rtol=0.012, atol=0.016)
    def table(value):
        return torch.tensor([value], dtype=torch.int64, device=device)
    one = torch.ones(1, dtype=torch.int32, device=device)
    zero = torch.zeros_like(one)
    commit = gdn.bind_kda_commit(
        binding.plan, scratch=binding.scratch,
        state_base_addrs=table(checkpoint.data_ptr()), state_block_strides=table(checkpoint.stride(0)),
        correction_cache_base_addrs=table(correction.data_ptr()), correction_cache_block_strides=table(correction.stride(0)),
        kg_cache_base_addrs=table(kg.data_ptr()), kg_cache_block_strides=table(kg.stride(0)),
        A_log=binding.A_log.unsqueeze(0), dt_bias=binding.dt_bias.unsqueeze(0),
        state_indices=binding.state_indices[:, 0], commit_lens=one,
        final_state_indices=one, boundary_state_indices=zero, boundary_recovery_lens=zero,
    )
    gdn.run_kda_commit(commit)
    torch.testing.assert_close(checkpoint, expected_state, atol=1e-6, rtol=1e-5)
    if not large_stride:
        assert (storage.view(2, 67584)[:, 66560:] == 123).all()
    saved = checkpoint.clone()
    binding.state_indices.zero_()
    binding.output.fill_(float("nan"))
    gdn.run_kda(binding)
    gdn.run_kda_commit(commit)
    torch.testing.assert_close(checkpoint, saved, rtol=0, atol=0)
    assert (binding.output == 0).all()


def test_reference_applies_per_key_lower_bounded_decay() -> None:
    torch.manual_seed(19)
    device = torch.device("cpu")
    mixed_qkv = _randn((1, 384), device=device)
    raw_g = _randn((1, 1, 128), device=device)
    raw_beta = _randn((1, 1), device=device)
    z = _randn((1, 1, 128), device=device)
    A_log = torch.tensor([0.2], dtype=torch.float32)
    dt_bias = torch.linspace(-0.4, 0.3, 128, dtype=torch.float32).view(1, 128)
    norm_weight = torch.linspace(0.8, 1.2, 128, dtype=torch.float32)
    state = _randn((1, 1, 128, 128), device=device, dtype=torch.float32)
    initial_state = state.clone()
    state_indices = torch.zeros((1, 1), dtype=torch.int64)

    actual = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        state_indices,
        1,
        1,
        heads=1,
    )

    q, k, value = mixed_qkv.float().view(3, 128).unbind(dim=0)
    q *= torch.rsqrt(q.square().sum() + 1e-6)
    k *= torch.rsqrt(k.square().sum() + 1e-6)
    q *= 128**-0.5
    log_decay = -5.0 * torch.sigmoid(
        torch.exp(A_log[0]) * (raw_g[0, 0].float() + dt_bias[0])
    )
    expected_state = initial_state[0, 0] * torch.exp(log_decay).unsqueeze(0)
    delta = value - expected_state.mv(k)
    expected_state += (delta * torch.sigmoid(raw_beta[0, 0].float())).unsqueeze(
        1
    ) * k.unsqueeze(0)
    core = expected_state.mv(q).to(torch.bfloat16)
    normalized = core.float() * torch.rsqrt(core.float().square().mean() + 1e-6)
    expected = (normalized * norm_weight * torch.sigmoid(z[0, 0].float())).to(
        torch.bfloat16
    )

    torch.testing.assert_close(actual[0, 0], expected, rtol=0, atol=0)
    torch.testing.assert_close(state[0, 0], expected_state, rtol=0, atol=0)


def test_reference_keeps_kda_rmsnorm_in_fp32_until_final_store() -> None:
    core = torch.linspace(-1.3, 1.7, 128, dtype=torch.float32).to(torch.bfloat16)
    q = torch.zeros(128, dtype=torch.bfloat16)
    q[0] = 1
    k = q.clone()
    value = core.clone()
    mixed_qkv = torch.cat((q, k, value)).view(1, -1)
    raw_g = torch.full((1, 1, 128), -100, dtype=torch.bfloat16)
    raw_beta = torch.full((1, 1), 100, dtype=torch.bfloat16)
    z = (
        torch.linspace(-2, 2, 128, dtype=torch.float32)
        .to(torch.bfloat16)
        .view(1, 1, 128)
    )
    norm_weight = torch.linspace(0.7, 1.4, 128, dtype=torch.float32).to(torch.bfloat16)
    state = torch.zeros((1, 1, 128, 128), dtype=torch.float32)

    actual = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        torch.zeros(1, dtype=torch.float32),
        torch.zeros((1, 128), dtype=torch.float32),
        norm_weight,
        state,
        torch.tensor([0, 1], dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.zeros((1, 1), dtype=torch.int64),
        1,
        1,
        heads=1,
    )[0, 0]

    decoded = (core.float() * (128**-0.5)).to(torch.bfloat16)
    normalized = decoded.float() * torch.rsqrt(decoded.float().square().mean() + 1e-6)
    gate = torch.sigmoid(z[0, 0].float())
    expected = (normalized * norm_weight.float() * gate).to(torch.bfloat16)
    old_intermediate_bf16 = (
        normalized.to(torch.bfloat16).float() * norm_weight.float() * gate
    ).to(torch.bfloat16)

    assert torch.count_nonzero(expected != old_intermediate_bf16) > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_reference_null_state_sentinel_zeroes_requests_and_skips_writes() -> None:
    torch.manual_seed(23)
    device = torch.device("cpu")
    heads = 1
    max_tokens = 2
    mixed_qkv = _randn((max_tokens, heads * 384), device=device)
    raw_g = _randn((max_tokens, heads, 128), device=device)
    raw_beta = _randn((max_tokens, heads), device=device)
    z = _randn((max_tokens, heads, 128), device=device)
    A_log = _randn((heads,), device=device, dtype=torch.float32)
    dt_bias = _randn((heads, 128), device=device, dtype=torch.float32)
    norm_weight = torch.ones(128, dtype=torch.float32)
    query_start_loc = torch.tensor([0, 1, 2], dtype=torch.int32)
    accepted = torch.full((2,), 2, dtype=torch.int32)
    state = _randn((3, heads, 128, 128), device=device, dtype=torch.float32)
    before = state.clone()

    output = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        query_start_loc,
        accepted,
        torch.tensor([[1, 0], [2, 0]], dtype=torch.int64),
        2,
        2,
        heads=heads,
        null_state_index=0,
    )

    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(state, before, rtol=0, atol=0)

    output = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        state,
        query_start_loc,
        torch.full((2,), 2, dtype=torch.int32),
        torch.tensor([[0, 1], [0, 2]], dtype=torch.int64),
        2,
        2,
        heads=heads,
        null_state_index=0,
    )

    assert torch.count_nonzero(output) > 0
    torch.testing.assert_close(state, before, rtol=0, atol=0)


def test_kda_rmsnorm_kernel_avoids_intermediate_bf16_rounding() -> None:
    device = require_sm120()
    from b12x.sequence.gdn_decode._kernels import _gated_rmsnorm_kernel

    eps = 1e-5
    output = (
        torch.linspace(-1.3, 1.7, 128, dtype=torch.float32, device=device)
        .to(torch.bfloat16)
        .view(1, 1, 128)
    )
    core = output.clone()
    z = torch.full_like(output, 100)
    norm_weight = torch.linspace(0.7, 1.4, 128, dtype=torch.float32, device=device).to(
        torch.bfloat16
    )
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)

    _gated_rmsnorm_kernel[(1,)](
        output,
        z,
        norm_weight,
        num_tokens,
        eps,
        1,
        stride_output_token=output.stride(0),
        stride_output_head=output.stride(1),
        stride_z_token=z.stride(0),
        stride_z_head=z.stride(1),
        VALUE_HEADS=1,
        VALUE_HEAD_DIM=128,
        SIGMOID_GATE=True,
        NORM_WEIGHT_FP32=False,
        KDA_NORM_FP32=True,
        num_warps=4,
        num_stages=1,
    )
    torch.cuda.synchronize(device)

    values = core[0, 0].float()
    normalized = values * torch.rsqrt(values.square().mean() + eps)
    expected = (normalized * norm_weight.float()).to(torch.bfloat16)
    old_intermediate_bf16 = (normalized.to(torch.bfloat16) * norm_weight).to(
        torch.bfloat16
    )
    expected_delta = torch.count_nonzero(expected != old_intermediate_bf16)
    assert expected_delta >= 16
    new_error = (output[0, 0].float() - expected.float()).abs().sum()
    old_error = (output[0, 0].float() - old_intermediate_bf16.float()).abs().sum()
    assert new_error < old_error


@pytest.mark.parametrize("recurrent_block_v", [16, 32])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_packed_kda_matches_reference(
    state_dtype: torch.dtype,
    recurrent_block_v: int,
) -> None:
    device = require_sm120()
    binding = _make_case(
        device=device,
        state_dtype=state_dtype,
        recurrent_block_v=recurrent_block_v,
    )
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    assert binding._state.recurrent_block_v == recurrent_block_v
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state,
        state_reference,
        rtol=1e-2 if state_dtype == torch.bfloat16 else 1e-5,
        atol=8e-3 if state_dtype == torch.bfloat16 else 2e-5,
    )
    assert torch.count_nonzero(actual[4:]) == 0


def test_kda_binds_live_tensors_within_planned_capacity() -> None:
    device = require_sm120()
    binding = _make_case(
        device=device,
        query_lengths=(1, 1),
        columns=3,
        max_tokens=6,
        tensor_tokens=2,
        tensor_columns=1,
        noncontiguous_beta=True,
    )
    padded = {
        name: _row_padded(getattr(binding, name))
        for name in ("mixed_qkv", "raw_g", "z", "output")
    }
    # Projection strides specialize the executable, unlike live row counts.
    with pytest.raises(ValueError, match="compile-time strides"):
        _rebind(binding, **padded)
    args = {
        name: getattr(binding, name)
        for name in (
            "mixed_qkv", "raw_g", "raw_beta", "z", "A_log", "dt_bias",
            "norm_weight", "recurrent_state", "query_start_loc",
            "num_accepted_tokens", "state_indices", "num_seqs", "num_tokens",
            "output",
        )
    }
    args.update(padded)
    binding = _prepare_binding(binding._state.caps, args)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert not binding.raw_beta.is_contiguous()
    assert not binding.mixed_qkv.is_contiguous()
    assert not binding.raw_g.is_contiguous()
    assert not binding.z.is_contiguous()
    assert not binding.output.is_contiguous()
    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_binding_rejects_invalid_live_contract() -> None:
    device = require_sm120()
    binding = _make_case(device=device)

    with pytest.raises(ValueError, match="exceeds planned capacity"):
        _rebind(
            binding,
            mixed_qkv=torch.empty(
                binding._state.caps.max_tokens + 1,
                binding._state.caps.packed_qkv_width,
                dtype=torch.bfloat16,
                device=device,
            ),
        )

    with pytest.raises(TypeError, match="num_accepted_tokens must have dtype"):
        _rebind(
            binding,
            num_accepted_tokens=binding.num_accepted_tokens.to(torch.int64),
        )


@pytest.mark.parametrize("state_indices_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("index_offset", [0, 1, 2])
def test_live_kda_cuda_graph_replays_bound_capacity(
    state_indices_dtype: torch.dtype, index_offset: int,
) -> None:
    """Sliced metadata must reuse the prepared kernel and remain graph-safe."""
    device = require_sm120()
    binding = _make_case(
        device=device,
        query_lengths=(1, 1),
        columns=3,
        max_tokens=6,
        tensor_tokens=2,
        tensor_columns=1,
        state_indices_dtype=state_indices_dtype,
    )
    storage = torch.empty(
        binding.state_indices.numel() + index_offset,
        dtype=state_indices_dtype, device=device,
    )
    indices = storage[index_offset:].view_as(binding.state_indices)
    indices.copy_(binding.state_indices)
    with kernel_resolution_guard("KDA sliced state indices"):
        binding = _rebind(binding, state_indices=indices)

    def launch() -> torch.Tensor:
        return gdn.run_kda(_rebind(binding))

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = launch()

    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    output_ptr = captured_output.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)

    assert captured_output.data_ptr() == output_ptr
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )
    graph.reset()


def test_glm53_tp8_head_geometry_matches_reference() -> None:
    device = require_sm120()
    binding = _make_case(device=device, heads=8)
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    assert binding.mixed_qkv.shape == (5, 3072)
    assert binding.raw_g.shape == (5, 8, 128)
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_rejected_draft_restarts_from_accepted_checkpoint() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    binding.num_accepted_tokens[0] = 1
    gdn.run_kda(binding)
    torch.cuda.synchronize(device)
    accepted_checkpoint = binding.recurrent_state[1].clone()

    binding.num_accepted_tokens[0] = 2
    binding.recurrent_state[2].fill_(73.0)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    binding.z.copy_(torch.randn_like(binding.z).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    torch.testing.assert_close(state_reference[1], accepted_checkpoint, rtol=0, atol=0)
    expected = _reference(binding, state_reference)

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_null_state_sentinel_is_graph_safe_and_immutable() -> None:
    device = require_sm120()
    binding = _make_case(device=device, null_state_index=0)
    binding.state_indices.zero_()
    binding.state_indices[:, 0].copy_(
        torch.arange(1, 3, dtype=binding.state_indices.dtype, device=device)
    )
    binding.num_accepted_tokens.fill_(2)
    before = binding.recurrent_state.clone()

    gdn.run_kda(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(binding)

    output_ptr = captured_output.data_ptr()
    state_ptr = binding.recurrent_state.data_ptr()
    captured_output.fill_(float("nan"))
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    graph.replay()
    torch.cuda.synchronize(device)

    assert captured_output.data_ptr() == output_ptr
    assert binding.recurrent_state.data_ptr() == state_ptr
    assert torch.count_nonzero(captured_output) == 0
    torch.testing.assert_close(binding.recurrent_state, before, rtol=0, atol=0)
    graph.reset()


def test_kda_cuda_graph_replay_preserves_addresses() -> None:
    device = require_sm120()
    binding = _make_case(device=device)
    gdn.run_kda(binding)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_output = gdn.run_kda(binding)

    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)
    output_ptr = captured_output.data_ptr()
    state_ptr = binding.recurrent_state.data_ptr()

    graph.replay()
    torch.cuda.synchronize(device)

    assert captured_output.data_ptr() == output_ptr
    assert binding.recurrent_state.data_ptr() == state_ptr
    torch.testing.assert_close(captured_output, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )
    graph.reset()


@pytest.mark.parametrize("recurrent_block_v", [16, 32])
@pytest.mark.parametrize("state_dtype", [torch.bfloat16, torch.float32])
def test_kda_tiles_reuse_compiled_kernels_with_mutable_live_counts(
    recurrent_block_v: int,
    state_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from b12x._lib.runtime_control import kernel_resolution_guard
    from b12x.sequence.gdn_decode import _kernels

    device = require_sm120()
    binding = _make_case(
        device=device,
        state_dtype=state_dtype,
        recurrent_block_v=recurrent_block_v,
    )
    initial_state = binding.recurrent_state.clone()
    gdn.run_kda(binding)
    kernel = _kernels._packed_sequential_kda_decode_kernel
    cache = kernel.device_caches[torch.cuda.current_device()][0]
    compiled = {key: id(value) for key, value in cache.items()}
    assert compiled

    def refuse_compile(*args, **kwargs):
        pytest.fail("KDA live counts triggered compilation after warmup")

    monkeypatch.setattr(kernel, "compile", refuse_compile)
    addresses = tuple(
        tensor.data_ptr()
        for tensor in (
            binding.scratch,
            binding.mixed_qkv,
            binding.raw_g,
            binding.raw_beta,
            binding.recurrent_state,
            binding.output,
        )
    )
    with kernel_resolution_guard("KDA tile live-count qualification"):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            gdn.run_kda(binding)
        for starts, sequences, tokens in (([0, 3, 4], 2, 4), ([0, 2, 3], 2, 3), ([0, 1, 1], 1, 1)):
            binding.query_start_loc.copy_(
                torch.tensor(starts, dtype=torch.int32, device=device)
            )
            binding.num_seqs.fill_(sequences)
            binding.num_tokens.fill_(tokens)
            binding.num_accepted_tokens.fill_(1)
            binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
            binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
            binding.raw_beta.copy_(torch.randn_like(binding.raw_beta).mul_(0.2))
            state_reference = initial_state.clone()
            expected = _reference(binding, state_reference)
            binding.recurrent_state.copy_(initial_state)
            gdn.run_kda(binding)
            assert {key: id(value) for key, value in cache.items()} == compiled
            binding.recurrent_state.copy_(initial_state)
            binding.output.fill_(float("nan"))
            torch.cuda.synchronize(device)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for key in ("allocation.all.allocated", "allocated_bytes.all.allocated"):
                assert before[key] == after[key]
            assert bool(torch.isfinite(binding.output).all())
            assert int(torch.count_nonzero(binding.output[:tokens])) > 0
            torch.testing.assert_close(binding.output, expected, rtol=1e-2, atol=2e-2)
            torch.testing.assert_close(
                binding.recurrent_state,
                state_reference,
                rtol=1e-2 if state_dtype == torch.bfloat16 else 1e-5,
                atol=8e-3 if state_dtype == torch.bfloat16 else 2e-5,
            )
        assert tuple(
            tensor.data_ptr()
            for tensor in (
                binding.scratch,
                binding.mixed_qkv,
                binding.raw_g,
                binding.raw_beta,
                binding.recurrent_state,
                binding.output,
            )
        ) == addresses
    graph.reset()


def test_kda_torch_compile_fullgraph_keeps_outer_op_opaque() -> None:
    device = require_sm120()
    binding = _make_case(device=device)

    def launch() -> torch.Tensor:
        return gdn.run_kda(binding)

    launch()
    compiled = torch.compile(launch, fullgraph=True)
    binding.mixed_qkv.copy_(torch.randn_like(binding.mixed_qkv).mul_(0.2))
    binding.raw_g.copy_(torch.randn_like(binding.raw_g).mul_(0.2))
    binding.recurrent_state.copy_(torch.randn_like(binding.recurrent_state).mul_(0.1))
    state_reference = binding.recurrent_state.clone()
    expected = _reference(binding, state_reference)

    actual = compiled()
    torch.cuda.synchronize(device)

    assert actual.data_ptr() == binding.output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        binding.recurrent_state, state_reference, rtol=1e-5, atol=2e-5
    )


def test_kda_padded_state_slot_past_int32_element_boundary() -> None:
    device = require_sm120()
    heads = 1
    slot_elements = heads * 128 * 128
    slot_stride = slot_elements + 2_048
    tail_slot = (1 << 31) // slot_stride + 1
    assert tail_slot * slot_stride > 1 << 31
    caps = gdn.Caps(
        device=device,
        max_tokens=1,
        max_seqs=1,
        max_state_slots=tail_slot + 1,
        key_heads=heads,
        value_heads=heads,
        state_dtype=torch.bfloat16,
        gate_activation="sigmoid",
    )
    state_storage = torch.empty(
        tail_slot * slot_stride + slot_elements,
        dtype=torch.bfloat16,
        device=device,
    )
    recurrent_state = torch.as_strided(
        state_storage,
        size=(tail_slot + 1, heads, 128, 128),
        stride=(slot_stride, 128 * 128, 128, 1),
    )
    recurrent_state[tail_slot].copy_(
        _randn((heads, 128, 128), device=device, scale=0.1)
    )
    compact_state = recurrent_state[tail_slot : tail_slot + 1].clone()
    mixed_qkv = _randn((1, caps.packed_qkv_width), device=device)
    raw_g = _randn((1, heads, 128), device=device)
    raw_beta = _randn((1, heads), device=device)
    z = _randn((1, heads, 128), device=device)
    A_log = _randn((heads,), device=device, dtype=torch.float32, scale=0.1)
    dt_bias = _randn((heads, 128), device=device, dtype=torch.float32, scale=0.1)
    norm_weight = (
        1.0 + _randn((128,), device=device, dtype=torch.float32, scale=0.05)
    ).contiguous()
    query_start_loc = torch.tensor([0, 1], dtype=torch.int32, device=device)
    accepted = torch.ones(1, dtype=torch.int32, device=device)
    state_indices = torch.tensor([[tail_slot]], dtype=torch.int64, device=device)
    compact_indices = torch.zeros((1, 1), dtype=torch.int64, device=device)
    num_seqs = torch.ones(1, dtype=torch.int32, device=device)
    num_tokens = torch.ones(1, dtype=torch.int32, device=device)
    output = torch.empty((1, heads, 128), dtype=torch.bfloat16, device=device)
    binding = _prepare_binding(caps, {
        "mixed_qkv": mixed_qkv, "raw_g": raw_g, "raw_beta": raw_beta, "z": z,
        "A_log": A_log, "dt_bias": dt_bias, "norm_weight": norm_weight,
        "recurrent_state": recurrent_state, "query_start_loc": query_start_loc,
        "num_accepted_tokens": accepted, "state_indices": state_indices,
        "num_seqs": num_seqs, "num_tokens": num_tokens, "output": output,
    })
    expected = gdn.reference.decode_kda(
        mixed_qkv,
        raw_g,
        raw_beta,
        z,
        A_log,
        dt_bias,
        norm_weight,
        compact_state,
        query_start_loc,
        accepted,
        compact_indices,
        num_seqs,
        num_tokens,
        heads=heads,
    )

    actual = gdn.run_kda(binding)
    torch.cuda.synchronize(device)

    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=2e-2)
    torch.testing.assert_close(
        recurrent_state[tail_slot], compact_state[0], rtol=1e-2, atol=8e-3
    )

    del binding, recurrent_state, state_storage
    gc.collect()
    torch.cuda.empty_cache()
