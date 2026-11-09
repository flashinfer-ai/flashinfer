"""Weight snapshot ownership and refresh validation without kernel launches."""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import MoEWeightPack, preprocess_bf16_cutedsl_mega_weights


def _source():
    return MoEWeightPack(
        torch.nn.Parameter(torch.arange(2 * 128 * 128).reshape(2, 128, 128).bfloat16()),
        torch.nn.Parameter(torch.full((2, 128, 64), 0.5, dtype=torch.bfloat16)),
    )


def _prepare(source, *, out=None):
    return preprocess_bf16_cutedsl_mega_weights(
        source, hidden_size=128, intermediate_size=64, out=out
    )


def _tensors(prepared):
    return [tensor for pair in prepared for tensor in pair if tensor is not None]


def _bits(tensor):
    return tensor.detach().contiguous().view(torch.uint8)


def _assert_same(actual, expected):
    for left, right in zip(_tensors(actual), _tensors(expected), strict=True):
        assert torch.equal(_bits(left), _bits(right))


def test_bf16_snapshot_is_detached_and_refresh_preserves_addresses():
    source = _source()
    prepared = _prepare(source)
    saved = [
        (tensor.clone(), tensor.data_ptr(), tensor.stride())
        for tensor in _tensors(prepared)
    ]
    for tensor in _tensors(prepared):
        assert not tensor.requires_grad
        assert tensor.grad_fn is None

    # FC1 used to be a copy while FC2 was a view of this updated parameter.
    with torch.no_grad():
        source.w13.zero_()
        source.w2.add_(2)
    for tensor, (old, _, _) in zip(_tensors(prepared), saved, strict=True):
        assert torch.equal(_bits(tensor), _bits(old))

    assert _prepare(source, out=prepared) is prepared
    _assert_same(prepared, _prepare(source))
    for tensor, (_, pointer, strides) in zip(_tensors(prepared), saved, strict=True):
        assert tensor.data_ptr() == pointer
        assert tensor.stride() == strides
    assert torch.count_nonzero(prepared[0][0]) == 0
    assert torch.equal(prepared[1][0], torch.full_like(prepared[1][0], 2.5))


@pytest.mark.parametrize(
    "failure", ["dtype", "stride", "scale", "grad", "alias", "alignment"]
)
def test_refresh_validates_both_projections_before_writing(failure):
    source = _source()
    prepared = _prepare(source)
    before = prepared[0][0].clone()
    target = prepared[1][0]
    scale = None
    if failure == "dtype":
        target = target.float()
    elif failure == "stride":
        target = target.contiguous()
    elif failure == "scale":
        scale = torch.ones(1)
    elif failure == "grad":
        target.requires_grad_()
    elif failure == "alias":
        target = source.w2.detach().transpose(1, 2)
    else:
        target = torch.empty(target.numel() + 1, dtype=target.dtype)[1:].as_strided(
            target.shape, target.stride()
        )
    with torch.no_grad():
        source.w13.zero_()
    with pytest.raises(ValueError, match="out"):
        _prepare(source, out=(prepared[0], (target, scale)))
    assert torch.equal(_bits(prepared[0][0]), _bits(before))


def test_refresh_rejects_overlapping_destinations():
    source = _source()
    prepared = _prepare(source)
    fc2 = prepared[1][0]
    overlapping = prepared[0][0].as_strided(fc2.shape, fc2.stride())
    before = prepared[0][0].clone()
    with pytest.raises(ValueError, match="overlap"):
        _prepare(source, out=(prepared[0], (overlapping, None)))
    assert torch.equal(_bits(prepared[0][0]), _bits(before))


def test_refresh_handles_strided_sources_and_inference_buffers():
    source = _source()
    source = MoEWeightPack(source.w13[..., ::2], source.w2[:, ::2, :])
    with torch.inference_mode():
        prepared = preprocess_bf16_cutedsl_mega_weights(
            source, hidden_size=64, intermediate_size=64
        )
    assert prepared[0][0].is_inference()
    assert prepared[1][0].is_inference()
    with torch.no_grad():
        source.w13.zero_()
        source.w2.add_(1)
    assert (
        preprocess_bf16_cutedsl_mega_weights(
            source, hidden_size=64, intermediate_size=64, out=prepared
        )
        is prepared
    )
    assert torch.count_nonzero(prepared[0][0]) == 0
    assert torch.equal(prepared[1][0], torch.full_like(prepared[1][0], 1.5))


def test_public_refresh_rejects_capture_before_preparing():
    source = _source()
    prepared = _prepare(source)
    with (
        mock.patch("torch.cuda.is_available", return_value=True),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_cutedsl.weights._interleave_gate_up_32"
        ) as prepare,
        pytest.raises(ValueError, match="capture"),
    ):
        _prepare(source, out=prepared)
    prepare.assert_not_called()


def test_bf16_snapshot_rejects_ignored_scales():
    source = _source()
    with pytest.raises(ValueError, match="quantization scales"):
        _prepare(MoEWeightPack(source.w13, source.w2, torch.ones(1), torch.ones(1)))


@pytest.fixture(params=[False, True], ids=["layer-prepared", "caller-prepared"])
def layer_and_source(dist_not_initialized, request):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpMegaLayer,
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
    )

    source = _source()
    with (
        mock.patch("flashinfer.moe_ep.modes.mega_layer.ensure_moe_ep_cuda_device"),
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100."
            "bf16_bf16_bf16_cutedsl.backend.validate_mega_arch"
        ),
    ):
        layer = MoEEpMegaLayer(
            BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=128),
            source,
            MegaConfig(
                megakernel=Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=64, top_k=2
                ),
                preprocess_weights=not request.param,
                transformed_weights=_prepare(source) if request.param else None,
            ),
        )
    try:
        yield layer, source
    finally:
        layer.destroy()


def test_layer_refresh_uses_live_parameters_without_retaining_them(layer_and_source):
    layer, source = layer_and_source
    prepared = layer._transformed
    pointers = [tensor.data_ptr() for tensor in _tensors(prepared)]
    count = layer._preprocessing_count
    with torch.no_grad():
        source.w13.zero_()
        source.w2.mul_(4)
    assert layer.supports_weight_refresh
    layer.refresh_weights(source)
    assert layer._transformed is prepared
    assert layer._weights is None
    assert layer._preprocessing_count == count + 1
    assert [tensor.data_ptr() for tensor in _tensors(prepared)] == pointers
    _assert_same(prepared, _prepare(source))


@pytest.mark.parametrize("state", ["capture", "unsupported", "destroyed"])
def test_layer_rejects_refresh_before_preprocessing(layer_and_source, state):
    from flashinfer.moe_ep import MoEEpConfigError

    layer, source = layer_and_source
    if state == "unsupported":
        layer._kernel.supports_weight_refresh = False
    elif state == "destroyed":
        layer.destroy()
    with (
        mock.patch("torch.cuda.is_available", return_value=state == "capture"),
        mock.patch("torch.cuda.is_current_stream_capturing", return_value=True),
        mock.patch.object(layer._kernel, "preprocess_weights") as preprocess,
        pytest.raises(MoEEpConfigError),
    ):
        layer.refresh_weights(source)
    preprocess.assert_not_called()


def test_layer_refresh_cannot_split_stage_and_compute(layer_and_source):
    from flashinfer.moe_ep import MoEEpConfigError, MoEEpTensors

    layer, source = layer_and_source
    t = MoEEpTensors(
        torch.ones(1, 128, dtype=torch.bfloat16),
        torch.tensor([[0, 1]]),
        torch.tensor([[0.25, 0.75]]),
    )
    workspace = object()
    with (
        mock.patch.object(
            layer,
            "_prepare_inputs",
            return_value=(
                layer._fleet_params,
                workspace,
                layer._kernel,
                False,
                layer._transformed,
            ),
        ),
        mock.patch.object(layer._kernel, "stage_inputs"),
        mock.patch.object(layer._kernel, "compute", return_value=t.hidden_states),
    ):
        layer.stage_inputs(t)
        with pytest.raises(MoEEpConfigError, match="between stage and compute"):
            layer.refresh_weights(source)
        layer.compute_staged(output=t.hidden_states)
    layer.refresh_weights(source)
