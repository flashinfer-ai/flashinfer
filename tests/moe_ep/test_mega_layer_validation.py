"""MoEEpMegaLayer validation error paths (no deep_gemm kernel launch)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

_KERNEL_READY = ((None, None), (None, None))


def _fake_deep_gemm_transformed(
    *,
    num_experts: int = 1,
    intermediate: int = 128,
    hidden: int = 128,
):
    import torch

    fc1_out = 2 * intermediate
    w1 = torch.zeros(num_experts, fc1_out, hidden // 2, dtype=torch.int8)
    sf1 = torch.zeros(num_experts, fc1_out, hidden // 32)
    w2 = torch.zeros(num_experts, hidden, intermediate // 2, dtype=torch.int8)
    sf2 = torch.zeros(num_experts, hidden, intermediate // 32)
    return ((w1, sf1), (w2, sf2))


def _mega_layer(
    *,
    quantize_input: bool = True,
    preprocess_weights: bool = False,
    transformed_weights=None,
):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    with mock.patch(
        "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend.validate_mega_arch"
    ):
        if transformed_weights is None:
            transformed_weights = _fake_deep_gemm_transformed()
        return MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=1,
                max_tokens_per_rank=64,
                token_hidden_size=128,
            ),
            weights=MoEWeightPack(
                w13=torch.zeros(1, 256, 128),
                w2=torch.zeros(1, 128, 128),
            ),
            backend=MegaConfig(
                megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                quantize_input=quantize_input,
                preprocess_weights=preprocess_weights,
                transformed_weights=transformed_weights,
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


def test_mega_layer_requires_weights():
    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
    )

    with pytest.raises(TypeError):
        MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=8,
                max_tokens_per_rank=64,
                token_hidden_size=128,
            ),
            backend=MegaConfig(
                megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                transformed_weights=_fake_deep_gemm_transformed(),
            ),
        )


def test_mega_layer_forward_rejects_token_overflow():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    t = MoEEpTensors(
        hidden_states=torch.zeros(65, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(65, 2, dtype=torch.int64),
        topk_weights=torch.zeros(65, 2),
    )
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        layer.forward(t)


def test_mega_layer_forward_accepts_partial_batch():
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    layer = _mega_layer()
    layer._workspace = _fake_symm_buffer(max_tokens=64)  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(16, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(16, 2, dtype=torch.int64),
        topk_weights=torch.zeros(16, 2),
    )
    with (
        mock.patch.object(layer._kernel, "compute", return_value=t.hidden_states),
        mock.patch.object(layer._kernel, "stage_inputs"),
    ):
        out = layer.forward(t)
    assert out.shape == (16, 128)


def test_mega_layer_forward_rejects_topk_mismatch():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 3, dtype=torch.int64),
        topk_weights=torch.zeros(4, 3),
    )
    with pytest.raises(MoEEpConfigError, match="topk_ids.shape"):
        layer.forward(t)


def test_mega_layer_forward_rejects_topk_weights_shape_mismatch():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 3),
    )
    with pytest.raises(MoEEpConfigError, match="same shape"):
        layer.forward(t)


def test_mega_layer_forward_requires_scales_when_copy_mode():
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs torch.float8_e4m3fn")

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer(quantize_input=False)
    layer._workspace = _fake_symm_buffer()  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128, dtype=torch.float8_e4m3fn),
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
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 64, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 2),
    )
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        layer.forward(t)


@mock.patch("torch.distributed.is_initialized", return_value=False)
def test_mega_layer_prepare_workspace_requires_dist(mock_dist_init):
    import sys

    layer = _mega_layer()
    with (
        mock.patch.dict(sys.modules, {"deep_gemm": mock.MagicMock()}),
        pytest.raises(RuntimeError, match="torch.distributed"),
    ):
        layer._ensure_workspace()


def test_mega_layer_init_rejects_bootstrap_world_size_mismatch():
    from flashinfer.moe_ep import MoEEpConfigError

    mock_pg = mock.MagicMock()
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_comm_group",
            return_value=mock_pg,
        ),
        mock.patch("torch.distributed.get_world_size", return_value=8),
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_ep_rank_world",
            return_value=(0, 8),
        ),
        pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"),
    ):
        _mega_layer()


def test_mega_layer_forward_passes_quantize_input_to_kernel():
    import torch

    from flashinfer.moe_ep import MoEEpTensors

    layer = _mega_layer(quantize_input=True)
    layer._workspace = _fake_symm_buffer(max_tokens=64)  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(8, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(8, 2, dtype=torch.int64),
        topk_weights=torch.zeros(8, 2),
    )
    with (
        mock.patch.object(layer._kernel, "compute", return_value=t.hidden_states),
        mock.patch.object(layer._kernel, "stage_inputs") as stage_mock,
    ):
        layer.forward(t)
        stage_mock.assert_called_once()
        assert stage_mock.call_args.kwargs["quantize_input"] is True


def test_mega_layer_forward_skips_quantize_when_config_disabled():
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs torch.float8_e4m3fn")

    from flashinfer.moe_ep import MoEEpTensors

    layer = _mega_layer(quantize_input=False)
    layer._workspace = _fake_symm_buffer(max_tokens=64)  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(8, 128, dtype=torch.float8_e4m3fn),
        topk_ids=torch.zeros(8, 2, dtype=torch.int64),
        topk_weights=torch.zeros(8, 2),
        scales=torch.zeros(8, 4),
    )
    with (
        mock.patch.object(layer._kernel, "compute", return_value=t.hidden_states),
        mock.patch.object(layer._kernel, "stage_inputs") as stage_mock,
    ):
        layer.forward(t)
        stage_mock.assert_called_once()
        assert stage_mock.call_args.kwargs["quantize_input"] is False


def test_mega_layer_forward_rejects_non_bf16_with_quantize_input():
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs torch.float8_e4m3fn")

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer(quantize_input=True)
    t = MoEEpTensors(
        hidden_states=torch.zeros(8, 128, dtype=torch.float8_e4m3fn),
        topk_ids=torch.zeros(8, 2, dtype=torch.int64),
        topk_weights=torch.zeros(8, 2),
        scales=torch.zeros(8, 4),
    )
    with pytest.raises(MoEEpConfigError, match="quantize_input=True expects bf16"):
        layer.forward(t)


def test_mega_layer_init_rejects_bad_fleet_weights(dist_not_initialized):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpConfigError,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend.validate_mega_arch"
        ),
        pytest.raises(MoEEpConfigError, match="num_experts // world_size"),
    ):
        MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=4, rank=0, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=8,
                max_tokens_per_rank=64,
                token_hidden_size=128,
            ),
            weights=MoEWeightPack(
                w13=torch.zeros(4, 256, 128),
                w2=torch.zeros(4, 128, 128),
            ),
            backend=MegaConfig(
                megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                preprocess_weights=True,
            ),
        )


def test_mega_layer_init_skips_fleet_weights_when_transformed_supplied(
    dist_not_initialized,
):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        DeepGemmMegaMoeConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        MoEWeightPack,
    )

    with mock.patch(
        "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend.validate_mega_arch"
    ):
        layer = MoEEpMegaLayer(
            bootstrap=BootstrapConfig(world_size=4, rank=0, auto_bootstrap=False),
            fleet_params=FleetParams(
                num_experts=8,
                max_tokens_per_rank=64,
                token_hidden_size=128,
            ),
            weights=MoEWeightPack(
                w13=torch.zeros(4, 256, 128),
                w2=torch.zeros(4, 128, 128),
            ),
            backend=MegaConfig(
                megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
                preprocess_weights=False,
                transformed_weights=_fake_deep_gemm_transformed(num_experts=2),
            ),
        )
    assert layer._transformed is not None


def test_mega_layer_forward_deferred_bootstrap_validation():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer = _mega_layer()
    layer._workspace = _fake_symm_buffer(max_tokens=64)  # type: ignore[attr-defined]

    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 128, dtype=torch.bfloat16),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64),
        topk_weights=torch.zeros(4, 2),
    )
    mock_pg = mock.MagicMock()
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_comm_group",
            return_value=mock_pg,
        ),
        mock.patch("torch.distributed.get_world_size", return_value=8),
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_ep_rank_world",
            return_value=(0, 8),
        ),
        pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"),
    ):
        layer.forward(t)


def test_deep_gemm_stage_inputs_copy_path_stages_prequantized():
    import torch

    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("needs torch.float8_e4m3fn")

    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend import (
        DeepGemmMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.config import (
        DeepGemmMegaMoeConfig,
    )

    backend = DeepGemmMegaKernelBackend(
        DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2)
    )
    num_tokens = 4
    hidden = 128
    top_k = 2
    workspace = _fake_symm_buffer(max_tokens=64, hidden=hidden, top_k=top_k)

    hidden_fp8 = torch.randn(num_tokens, hidden).to(torch.float8_e4m3fn)
    scales = torch.randn(num_tokens, hidden // 32)
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 0], [1, 1]], dtype=torch.int64)
    topk_weights = torch.randn(num_tokens, top_k)

    t = MoEEpTensors(
        hidden_states=hidden_fp8,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        scales=scales,
    )
    backend.stage_inputs(t, workspace, quantize_input=False)

    assert torch.equal(workspace.x[:num_tokens], hidden_fp8.to(torch.float32))
    assert torch.equal(workspace.x_sf[:num_tokens], scales)
    assert torch.equal(workspace.topk_idx[:num_tokens], topk_ids)
    assert torch.equal(workspace.topk_weights[:num_tokens], topk_weights)


def test_mega_layer_init_accepts_valid_transformed_weights():
    layer = _mega_layer()
    assert layer._transformed is not None


def test_mega_layer_init_rejects_invalid_transformed_weight_dtype():
    import torch

    from flashinfer.moe_ep import MoEEpConfigError

    bad = _fake_deep_gemm_transformed()
    bad = ((bad[0][0].to(torch.float32), bad[0][1]), bad[1])
    with pytest.raises(MoEEpConfigError, match="torch.int8"):
        _mega_layer(transformed_weights=bad)


def test_mega_layer_init_rejects_invalid_transformed_structure():
    from flashinfer.moe_ep import MoEEpConfigError

    with pytest.raises(MoEEpConfigError, match="2-tuple"):
        _mega_layer(
            transformed_weights=(_fake_deep_gemm_transformed()[0],),
        )


def test_deep_gemm_validate_transformed_weights_accepts_preprocess_output():
    pytest.importorskip("deep_gemm")
    import torch

    cap = torch.cuda.get_device_capability()
    if cap[0] != 10:
        pytest.skip(
            f"deep_gemm transform requires sm_100a or sm_103a; got sm_{cap[0]}{cap[1]}"
        )

    from flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.weights import (
        preprocess_mega_weights,
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep.weights import MoEWeightPack

    num_experts = 1
    intermediate = 128
    hidden = 128
    weights = MoEWeightPack(
        w13=torch.randn(num_experts, 2 * intermediate, hidden),
        w2=torch.randn(num_experts, hidden, intermediate),
    )
    transformed = preprocess_mega_weights(
        weights,
        intermediate_size=intermediate,
        hidden_size=hidden,
    )
    validate_transformed_mega_weights(
        transformed,
        intermediate_size=intermediate,
        hidden_size=hidden,
        world_size=1,
        num_experts=num_experts,
    )
