"""MoEEpMegaLayer validation error paths (no deep_gemm kernel launch)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest


def _mega_layer(*, stage_inputs: bool = True, preprocess_weights: bool = False):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    with mock.patch("flashinfer.moe_ep.core.validation.common.validate_mega_arch"):
        return MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=8,
                max_tokens_per_rank=64,
                token_hidden_size=128,
                weights=MoEWeightPack(
                    w13=torch.zeros(1, 256, 128),
                    w2=torch.zeros(1, 128, 128),
                ),
            ),
            backend=MegaConfig(
                megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                stage_inputs=stage_inputs,
                preprocess_weights=preprocess_weights,
            ),
        )


def _fake_symm_buffer(*, max_tokens: int = 64, hidden: int = 128, top_k: int = 2):
    import torch

    return SimpleNamespace(
        x=torch.zeros(max_tokens, hidden),
        x_sf=torch.zeros(max_tokens, hidden // 32),
        topk_idx=torch.zeros(max_tokens, top_k, dtype=torch.int64),
        topk_weights=torch.zeros(max_tokens, top_k),
    )


def test_mega_layer_init_requires_weights():
    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    with mock.patch("flashinfer.moe_ep.core.validation.common.validate_mega_arch"):
        with pytest.raises(ValueError, match="FleetParams.weights"):
            MoEEpMegaLayer(
                bootstrap=BootstrapConfig(world_size=1, rank=0),
                fleet_params=FleetParams(
                    num_experts=8,
                    max_tokens_per_rank=64,
                    token_hidden_size=128,
                ),
                backend=MegaConfig(
                    megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                    preprocess_weights=False,
                ),
            )


def test_mega_layer_forward_rejects_token_overflow():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    layer._transformed = ((None, None), (None, None))

    t = MoEEpTensors(
        hidden_states=torch.zeros(65, 128),
        topk_ids=torch.zeros(65, 2, dtype=torch.int64),
        topk_weights=torch.zeros(65, 2),
    )
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        layer.forward(t)


def test_mega_layer_forward_rejects_topk_mismatch():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    layer._transformed = ((None, None), (None, None))

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128),
        topk_ids=torch.zeros(4, 3, dtype=torch.int64),
        topk_weights=torch.zeros(4, 3),
    )
    with pytest.raises(MoEEpConfigError, match="topk_ids.shape"):
        layer.forward(t)


def test_mega_layer_forward_rejects_topk_weights_shape_mismatch():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    layer._transformed = ((None, None), (None, None))

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 3),
    )
    with pytest.raises(MoEEpConfigError, match="same shape"):
        layer.forward(t)


def test_mega_layer_forward_requires_scales_when_not_staging():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer(stage_inputs=False)
    layer._transformed = ((None, None), (None, None))
    layer._workspace = _fake_symm_buffer()  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 2),
        scales=None,
    )
    with pytest.raises(MoEEpConfigError, match="scales is required"):
        layer.forward(t)


def test_mega_layer_forward_rejects_hidden_mismatch():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    layer._transformed = ((None, None), (None, None))

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 64),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 2),
    )
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        layer.forward(t)


@mock.patch("torch.distributed.is_initialized", return_value=False)
def test_mega_layer_prepare_workspace_requires_dist(mock_dist_init):
    layer = _mega_layer()
    layer._transformed = ((None, None), (None, None))

    with pytest.raises(RuntimeError, match="torch.distributed"):
        layer._ensure_workspace()


def test_mega_layer_init_rejects_bootstrap_world_size_mismatch():
    from unittest import mock

    from flashinfer.moe_ep import MoEEpConfigError

    with mock.patch("torch.distributed.is_initialized", return_value=True):
        with mock.patch("torch.distributed.get_world_size", return_value=8):
            with pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"):
                _mega_layer()
