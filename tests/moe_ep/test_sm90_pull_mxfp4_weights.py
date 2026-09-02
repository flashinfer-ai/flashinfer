"""CPU-only raw ABI, gate/up, chunking, and transformed-weight tests."""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
from flashinfer.moe_ep.weights import MoEWeightPack, PrequantizedMoEWeights
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    weights as mxfp4_weights,
)


E, HIDDEN, INTERMEDIATE = 5, 128, 128


def _raw_pack(*, experts=E, hidden=HIDDEN, intermediate=INTERMEDIATE):
    w13 = torch.empty(experts, 2 * intermediate, hidden // 2, dtype=torch.uint8)
    w2 = torch.empty(experts, hidden, intermediate // 2, dtype=torch.uint8)
    s13 = torch.empty(experts, 2 * intermediate, hidden // 32, dtype=torch.uint8)
    s2 = torch.empty(experts, hidden, intermediate // 32, dtype=torch.uint8)
    for expert in range(experts):
        w13[expert].fill_(expert + 1)
        w2[expert].fill_(expert + 17)
        s13[expert].fill_(expert + 33)
        s2[expert].fill_(expert + 49)
    return PrequantizedMoEWeights(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2)


def _fake_humming(calls):
    def run(weight, raw_scale, *, max_range):
        calls.append((weight.clone(), raw_scale.clone(), max_range))
        experts, rows, packed_k = weight.shape
        logical_k = packed_k * 2
        folded = torch.empty(
            experts,
            rows // 64,
            logical_k // 128,
            16,
            16,
            dtype=torch.uint8,
        )
        folded.zero_()
        residual = weight[:, 0, 0].to(torch.float32)
        return weight.clone(), folded, residual

    return run


def _allow_cpu_host_preprocessing(monkeypatch):
    monkeypatch.setattr(
        mxfp4_weights, "_require_cuda_tensor", lambda *_args, **_kwargs: None
    )


def test_gate_up_8_row_permutation_is_exact_and_dtype_agnostic():
    intermediate = 16
    rows = 2 * intermediate
    payload = torch.arange(rows, dtype=torch.uint8).view(1, rows, 1)
    scale = (payload + 100).clone()
    payload_out = mxfp4_weights._interleave_gate_up_8(
        payload, intermediate_size=intermediate
    )
    scale_out = mxfp4_weights._interleave_gate_up_8(
        scale, intermediate_size=intermediate
    )
    expected = torch.tensor(
        list(range(0, 8))
        + list(range(16, 24))
        + list(range(8, 16))
        + list(range(24, 32)),
        dtype=torch.uint8,
    )
    assert torch.equal(payload_out.flatten(), expected)
    assert torch.equal(scale_out.flatten(), expected + 100)


def test_chunked_humming_uses_4_experts_and_synchronously_permutes_fc1(monkeypatch):
    calls = []
    monkeypatch.setattr(mxfp4_weights, "_humming_preprocess", _fake_humming(calls))
    pack = _raw_pack(experts=10)
    processed, folded, residual = mxfp4_weights._preprocess_humming_leg_chunked(
        pack.w13,
        pack.w13_scale,
        gate_up_intermediate_size=INTERMEDIATE,
    )
    assert [call[0].shape[0] for call in calls] == [4, 4, 2]
    assert all(call[2] == 11 for call in calls)
    assert processed.shape == pack.w13.shape
    assert folded.shape == (10, 4, 1, 16, 16)
    assert torch.equal(residual, torch.arange(1, 11, dtype=torch.float32))

    expected_payload = mxfp4_weights._interleave_gate_up_8(
        pack.w13[:4], intermediate_size=INTERMEDIATE
    )
    expected_scale = mxfp4_weights._interleave_gate_up_8(
        pack.w13_scale[:4], intermediate_size=INTERMEDIATE
    )
    assert torch.equal(calls[0][0], expected_payload)
    assert torch.equal(calls[0][1], expected_scale)


def test_preprocess_builds_four_slot_packed_k_abi(monkeypatch):
    _allow_cpu_host_preprocessing(monkeypatch)
    calls = []
    monkeypatch.setattr(mxfp4_weights, "_humming_preprocess", _fake_humming(calls))
    transformed = mxfp4_weights.preprocess_mega_weights(
        _raw_pack(), intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    # FC1 chunks, then FC2 chunks.
    assert [call[0].shape[0] for call in calls] == [4, 1, 4, 1]
    fc1, fc2 = transformed
    assert fc1[0].shape == (E, HIDDEN // 2, 2 * INTERMEDIATE)
    assert fc2[0].shape == (E, INTERMEDIATE // 2, HIDDEN)
    assert fc1[0].dtype == fc2[0].dtype == torch.uint8
    assert fc1[0].stride(1) == fc2[0].stride(1) == 1
    assert fc1[1].shape == (E, (2 * INTERMEDIATE) // 64, HIDDEN // 128, 16, 16)
    assert fc2[1].shape == (E, HIDDEN // 64, INTERMEDIATE // 128, 16, 16)
    assert torch.equal(fc1[2], torch.ones(1, dtype=torch.float32))
    assert torch.equal(fc2[2], torch.ones(1, dtype=torch.float32))
    assert torch.equal(fc1[3], torch.arange(1, E + 1, dtype=torch.float32) * 64)
    assert torch.equal(fc2[3], torch.arange(17, 17 + E, dtype=torch.float32) * 64)

    mxfp4_weights.validate_transformed_mega_weights(
        transformed,
        intermediate_size=INTERMEDIATE,
        hidden_size=HIDDEN,
        world_size=1,
        num_experts=E,
    )


def test_raw_abi_rejects_unquantized_wrong_dtype_and_wrong_scale_shape(monkeypatch):
    monkeypatch.setattr(mxfp4_weights, "_humming_preprocess", _fake_humming([]))
    bf16 = MoEWeightPack(
        torch.zeros(E, 2 * INTERMEDIATE, HIDDEN, dtype=torch.bfloat16),
        torch.zeros(E, HIDDEN, INTERMEDIATE, dtype=torch.bfloat16),
    )
    with pytest.raises(MoEEpConfigError, match="PrequantizedMoEWeights"):
        mxfp4_weights.preprocess_mega_weights(
            bf16, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )

    pack = _raw_pack()
    bad_dtype = PrequantizedMoEWeights(
        w13=pack.w13.to(torch.int8),
        w2=pack.w2,
        w13_scale=pack.w13_scale,
        w2_scale=pack.w2_scale,
    )
    with pytest.raises(MoEEpConfigError, match="w13.*torch.uint8"):
        mxfp4_weights.preprocess_mega_weights(
            bad_dtype, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )

    bad_scale = PrequantizedMoEWeights(
        w13=pack.w13,
        w2=pack.w2,
        w13_scale=pack.w13_scale[..., :-1].contiguous(),
        w2_scale=pack.w2_scale,
    )
    with pytest.raises(MoEEpConfigError, match="w13_scale must have shape"):
        mxfp4_weights.preprocess_mega_weights(
            bad_scale, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )


def test_raw_abi_rejects_non_tensor_before_inspecting_shape():
    bad = PrequantizedMoEWeights(
        w13=object(),
        w2=torch.empty(E, HIDDEN, INTERMEDIATE // 2, dtype=torch.uint8),
        w13_scale=torch.empty(E, 2 * INTERMEDIATE, HIDDEN // 32, dtype=torch.uint8),
        w2_scale=torch.empty(E, HIDDEN, INTERMEDIATE // 32, dtype=torch.uint8),
    )
    with pytest.raises(MoEEpConfigError, match="w13 must be a torch.Tensor"):
        mxfp4_weights.preprocess_mega_weights(
            bad, intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )


def test_raw_abi_requires_cuda():
    with pytest.raises(MoEEpConfigError, match="CUDA tensor"):
        mxfp4_weights.preprocess_mega_weights(
            _raw_pack(), intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
        )


def test_transformed_validator_requires_cuda(monkeypatch):
    production_guard = mxfp4_weights._require_cuda_tensor
    _allow_cpu_host_preprocessing(monkeypatch)
    monkeypatch.setattr(mxfp4_weights, "_humming_preprocess", _fake_humming([]))
    transformed = mxfp4_weights.preprocess_mega_weights(
        _raw_pack(), intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    monkeypatch.setattr(mxfp4_weights, "_require_cuda_tensor", production_guard)
    with pytest.raises(MoEEpConfigError, match="CUDA tensor"):
        mxfp4_weights.validate_transformed_mega_weights(
            transformed,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            world_size=1,
            num_experts=E,
        )


def test_transformed_validator_rejects_row_major_weight(monkeypatch):
    _allow_cpu_host_preprocessing(monkeypatch)
    monkeypatch.setattr(mxfp4_weights, "_humming_preprocess", _fake_humming([]))
    transformed = mxfp4_weights.preprocess_mega_weights(
        _raw_pack(), intermediate_size=INTERMEDIATE, hidden_size=HIDDEN
    )
    fc1, fc2 = transformed
    bad = ((fc1[0].contiguous(), *fc1[1:]), fc2)
    with pytest.raises(MoEEpConfigError, match="storage-K stride 1"):
        mxfp4_weights.validate_transformed_mega_weights(
            bad,
            intermediate_size=INTERMEDIATE,
            hidden_size=HIDDEN,
            world_size=1,
            num_experts=E,
        )
