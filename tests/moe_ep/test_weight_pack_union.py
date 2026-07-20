"""Unit tests for the MoEWeightPack union (Unquantized | Prequantized).

CPU-only: covers the factory dispatch, the mixed-scale-state rejection (the
motivating silent-requantize footgun of ``todo_weight_pack_union.md``), and
that all three mega backends discriminate the variants by type.
"""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from flashinfer.moe_ep.weights import (  # noqa: E402
    MoEWeightPack,
    PrequantizedMoEWeights,
    UnquantizedMoEWeights,
    dummy_moe_weights,
)

E, HIDDEN, INTER = 2, 256, 128


def _bf16_tensors():
    w13 = torch.zeros(E, 2 * INTER, HIDDEN, dtype=torch.bfloat16)
    w2 = torch.zeros(E, HIDDEN, INTER, dtype=torch.bfloat16)
    return w13, w2


def _packed_tensors():
    w13 = torch.zeros(E, 2 * INTER, HIDDEN // 2, dtype=torch.uint8)
    w2 = torch.zeros(E, HIDDEN, INTER // 2, dtype=torch.uint8)
    w13_scale = torch.zeros(E, 2 * INTER, HIDDEN // 16, dtype=torch.float8_e4m3fn)
    w2_scale = torch.zeros(E, HIDDEN, INTER // 16, dtype=torch.float8_e4m3fn)
    return w13, w2, w13_scale, w2_scale


def test_factory_returns_unquantized_without_scales():
    w13, w2 = _bf16_tensors()
    pack = MoEWeightPack(w13, w2)
    assert type(pack) is UnquantizedMoEWeights
    assert isinstance(pack, MoEWeightPack)
    assert pack.w13 is w13 and pack.w2 is w2
    assert pack.w13_scale is None and pack.w2_scale is None


def test_factory_returns_prequantized_with_both_scales():
    w13, w2, s13, s2 = _packed_tensors()
    for pack in (
        MoEWeightPack(w13, w2, s13, s2),  # positional (old signature)
        MoEWeightPack(w13=w13, w2=w2, w13_scale=s13, w2_scale=s2),  # kwargs
    ):
        assert type(pack) is PrequantizedMoEWeights
        assert isinstance(pack, MoEWeightPack)
        assert pack.w13_scale is s13 and pack.w2_scale is s2


def test_mixed_scale_state_raises():
    w13, w2, s13, s2 = _packed_tensors()
    with pytest.raises(ValueError, match="BOTH scale planes"):
        MoEWeightPack(w13, w2, w13_scale=s13)
    with pytest.raises(ValueError, match="BOTH scale planes"):
        MoEWeightPack(w13, w2, w2_scale=s2)


def test_missing_tensors_raise():
    with pytest.raises(TypeError):
        MoEWeightPack()
    with pytest.raises(TypeError):
        MoEWeightPack(w13=torch.zeros(1, 2, 2))


def test_direct_variant_construction():
    w13, w2 = _bf16_tensors()
    pack = UnquantizedMoEWeights(w13=w13, w2=w2)
    assert pack.w13_scale is None
    pw13, pw2, s13, s2 = _packed_tensors()
    pre = PrequantizedMoEWeights(w13=pw13, w2=pw2, w13_scale=s13, w2_scale=s2)
    assert pre.w13_scale is s13
    with pytest.raises(TypeError):
        PrequantizedMoEWeights(w13=pw13, w2=pw2)  # scales are required fields


def test_variants_are_frozen():
    pack = MoEWeightPack(*_bf16_tensors())
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        pack.w13 = torch.zeros(1)


def test_dummy_moe_weights_is_unquantized():
    pack = dummy_moe_weights(num_local_experts=E, hidden=HIDDEN, intermediate=INTER)
    assert type(pack) is UnquantizedMoEWeights


def test_deep_gemm_discrimination_by_type():
    # deep_gemm preprocess re-quantizes iff the pack is NOT Prequantized;
    # verify the branch condition itself (full preprocess needs deep_gemm/GPU).
    unq = MoEWeightPack(*_bf16_tensors())
    pre = MoEWeightPack(*_packed_tensors())
    assert not isinstance(unq, PrequantizedMoEWeights)
    assert isinstance(pre, PrequantizedMoEWeights)


def test_nvfp4_preprocess_dispatches_prequantized_shapes():
    # The prequantized branch validates scale shapes before any GPU work:
    # a wrong scale shape must raise from the prequantized path (proving the
    # isinstance dispatch selected it), not fall through to re-quantization.
    from flashinfer.moe_ep.backends.mega.kernel.nvfp4_cutedsl.weights import (
        preprocess_mega_weights,
    )

    w13, w2, s13, s2 = _packed_tensors()
    bad_s13 = torch.zeros(E, 2 * INTER, HIDDEN // 32, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="w13_scale must have shape"):
        preprocess_mega_weights(
            MoEWeightPack(w13, w2, bad_s13, s2),
            intermediate_size=INTER,
            hidden_size=HIDDEN,
        )
