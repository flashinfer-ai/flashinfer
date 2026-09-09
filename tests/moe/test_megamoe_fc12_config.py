"""Host-side contracts for the unified MegaMOE FC12 backend."""

from __future__ import annotations

import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MegaMoeFc12Config,
    MoEConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.megamoe_fc12 import prepare_megamoe_fc12_weights


def _config() -> MoEConfig:
    return MoEConfig(
        routing=RoutingConfig(num_experts=2, top_k=1),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=64, local_num_experts=2),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )


def test_config_is_registered_and_exported():
    cfg = _config()
    assert cfg.backend.valid_for(100) == [MegaMoeFc12Config()]
    assert cfg.backend.valid_for(89) == []


def test_bf16_weight_view_uses_fc12_layout():
    w13 = torch.arange(2 * 128 * 32, dtype=torch.float32).reshape(2, 128, 32).bfloat16()
    w2 = torch.arange(2 * 32 * 64, dtype=torch.float32).reshape(2, 32, 64).bfloat16()
    view = prepare_megamoe_fc12_weights(
        w13,
        w2,
        quant=QuantConfig(),
        num_local_experts=2,
        hidden_size=32,
        intermediate_size=64,
    )
    assert view["fc1_weight"].shape == (2, 32, 128)
    assert view["fc2_weight"].shape == (2, 64, 32)
    # FC12 uses gate/up groups of 32: gate group 0, then up group 0.
    assert torch.equal(view["fc1_weight"][0, :, :32], w13[0, :32].transpose(0, 1))
    assert torch.equal(view["fc1_weight"][0, :, 32:64], w13[0, 64:96].transpose(0, 1))


def test_non_bf16_family_is_not_silently_prepared():
    w13 = torch.empty(1, 128, 32, dtype=torch.bfloat16)
    w2 = torch.empty(1, 32, 64, dtype=torch.bfloat16)
    try:
        prepare_megamoe_fc12_weights(
            w13,
            w2,
            quant=QuantConfig(
                weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8
            ),
            num_local_experts=1,
            hidden_size=32,
            intermediate_size=64,
        )
    except NotImplementedError as error:
        assert "MXFP8" in str(error)
    else:
        raise AssertionError("unsupported FC12 family must not get a BF16 view")


def test_ep_materializes_megamoe_view_alongside_other_candidates():
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.weights import (
        materialize_fused_moe_weights,
    )

    w13 = torch.zeros(1, 128, 32, dtype=torch.bfloat16)
    w2 = torch.zeros(1, 32, 64, dtype=torch.bfloat16)
    cfg = MoEConfig(
        routing=RoutingConfig(num_experts=1, top_k=1),
        quant=QuantConfig(),
        experts=ExpertConfig(intermediate_size=64, local_num_experts=1),
        backend=BackendOptions(candidates=(MegaMoeFc12Config(),)),
    )
    native = materialize_fused_moe_weights(MoEWeightPack(w13=w13, w2=w2), cfg)
    assert set(native.get_view("megamoe_fc12")) == {"fc1_weight", "fc2_weight"}


def test_rank_major_bridge_preserves_int32_ids_and_bf16_weights():
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.bridge import (
        build_activation_pack_rank_major,
    )

    received = torch.zeros(1, 2, 32, dtype=torch.bfloat16)
    ids = torch.tensor([[0, -1], [0, 0]], dtype=torch.int32)
    weights = torch.tensor([[0.5, 1.0], [0.25, 0.75]], dtype=torch.bfloat16)
    pack = build_activation_pack_rank_major(
        received,
        ids,
        weights,
        num_local_experts=1,
        quant=QuantConfig(),
    )
    assert pack.topk_ids.dtype is torch.int32
    assert pack.topk_weights.dtype is torch.bfloat16
    assert pack.topk_ids.tolist() == [[0, 0], [0, 0]]
    assert pack.topk_weights.tolist() == [[0.5, 0.0], [0.25, 0.75]]
