"""Host-side W4A16 MegaMoE weight and activation contracts."""

from __future__ import annotations

import dataclasses
from unittest import mock

import pytest
import torch

import flashinfer.moe_ep as moe_ep
from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpConfigError,
    MoEEpLayer,
    MoEEpTensors,
    MoEWeightPack,
    PrequantizedMoEWeights,
    Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
    SplitConfig,
    UnquantizedMoEWeights,
    preprocess_w4a16_cutedsl_mega_weights,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.backend import (
    W4A16CutedslMegaKernelBackend,
)

_E, _H, _I = 2, 64, 64


def _native_sf_bytes(scales):
    """Independent CPU byte address oracle for native128x4 scale blocks."""
    data = scales.view(torch.uint8)
    experts, rows, columns = data.shape
    padded_rows = (rows + 127) // 128 * 128
    padded_columns = (columns + 3) // 4 * 4
    out = torch.zeros(experts * padded_rows * padded_columns, dtype=torch.uint8)
    for expert in range(experts):
        for row in range(rows):
            for column in range(columns):
                offset = (
                    expert * padded_rows * padded_columns
                    + (row // 128) * 128 * padded_columns
                    + (column // 4) * 512
                    + (row % 32) * 16
                    + (row // 32 % 4) * 4
                    + column % 4
                )
                out[offset] = data[expert, row, column]
    return out


@pytest.fixture(autouse=True)
def native_scale_interleave():
    # Production preprocessing remains GPU-only. Host contract tests replace
    # only the CUDA swizzler, preserving the actual backend row permutation,
    # dtype reinterpretation and prepared-layout validation around it.
    with mock.patch(
        "flashinfer.quantization.block_scale_interleave", side_effect=_native_sf_bytes
    ) as interleave:
        yield interleave


def _pack():
    def packed(shape):
        rows = torch.arange(shape[0] * shape[1]).reshape(*shape[:2], 1)
        columns = torch.arange(shape[2]).reshape(1, 1, -1)
        return (rows + 3 * columns).to(torch.uint8)

    def scales(shape):
        values = torch.arange(torch.tensor(shape).prod().item()).reshape(shape)
        return ((values % 7 + 1).float() / 8).to(torch.float8_e4m3fn)

    return PrequantizedMoEWeights(
        w13=packed((_E, 2 * _I, _H // 2)),
        w2=packed((_E, _H, _I // 2)),
        w13_scale=scales((_E, 2 * _I, _H // 16)),
        w2_scale=scales((_E, _H, _I // 16)),
    )


def _prepare(pack):
    return preprocess_w4a16_cutedsl_mega_weights(
        pack, intermediate_size=_I, hidden_size=_H
    )


def _fleet():
    return FleetParams(num_experts=_E, max_tokens_per_rank=4, token_hidden_size=_H)


def _config():
    return Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=_I, top_k=2)


def _construct(pack, backend):
    # This constructor test exercises validation/preprocessing without
    # creating a distributed runtime, even when run on a GPU test host.
    with (
        mock.patch("torch.cuda.is_available", return_value=False),
        mock.patch("torch.distributed.is_initialized", return_value=False),
    ):
        return MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=_fleet(),
            weights=pack,
            backend=backend,
        )


def test_global_scale_defaults_preserve_existing_pack_constructors():
    pack = _pack()
    for constructor in (MoEWeightPack, PrequantizedMoEWeights):
        copied = constructor(pack.w13, pack.w2, pack.w13_scale, pack.w2_scale)
        assert copied.w13_global_scale is None
        assert copied.w2_global_scale is None
    for constructor in (MoEWeightPack, UnquantizedMoEWeights):
        unquantized = constructor(
            torch.zeros(_E, 2 * _I, _H, dtype=torch.bfloat16),
            torch.zeros(_E, _H, _I, dtype=torch.bfloat16),
            None,
            None,
            w13_global_scale=None,
            w2_global_scale=None,
        )
        assert isinstance(unquantized, UnquantizedMoEWeights)
        assert unquantized.w13_global_scale is None
        assert unquantized.w2_global_scale is None


@pytest.mark.parametrize("constructor", (MoEWeightPack, PrequantizedMoEWeights))
def test_global_scale_factory_and_dataclass_replace(constructor):
    source = _pack()
    alpha = torch.tensor([1.00390625, 0.71013], dtype=torch.float32)
    pack = constructor(
        source.w13,
        source.w2,
        source.w13_scale,
        source.w2_scale,
        w13_global_scale=alpha,
    )
    assert pack.w13_global_scale is alpha
    assert pack.w2_global_scale is None
    updated = dataclasses.replace(pack, w2_global_scale=alpha.flip(0))
    assert updated.w13_global_scale is alpha
    torch.testing.assert_close(updated.w2_global_scale, alpha.flip(0), rtol=0, atol=0)
    assert all(
        field.kw_only
        for field in dataclasses.fields(pack)
        if field.name.endswith("global_scale")
    )


def test_global_scales_require_prequantized_weights():
    pack = _pack()
    with pytest.raises(ValueError, match="pre-quantized"):
        MoEWeightPack(pack.w13, pack.w2, w13_global_scale=torch.ones(_E))
    with pytest.raises(TypeError, match="no scale planes"):
        UnquantizedMoEWeights(pack.w13, pack.w2, w2_global_scale=torch.ones(_E))


@pytest.mark.parametrize("mode", ("mega", "split"))
def test_existing_backends_reject_global_scales(mode):
    pack = dataclasses.replace(_pack(), w13_global_scale=torch.ones(_E))
    backend = (
        MegaConfig(
            megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
                intermediate_size=_I, top_k=2
            )
        )
        if mode == "mega"
        else SplitConfig()
    )
    with pytest.raises(ValueError, match="does not support global weight scales"):
        _construct(pack, backend)


@pytest.mark.parametrize(
    "name",
    (
        "preprocess_mega_weights",
        "preprocess_bf16_cutedsl_mega_weights",
        "preprocess_nvfp4_cutedsl_mega_weights",
        "preprocess_mxfp8_cutedsl_mega_weights",
        "preprocess_sm120_mxfp8_cutedsl_mega_weights",
    ),
)
@pytest.mark.parametrize("field", ("w13_global_scale", "w2_global_scale"))
def test_public_preprocessing_rejects_unsupported_global_scales(name, field):
    # A caller can prepare weights before constructing a layer, then pass only
    # transformed_weights. Reject here, before imports or transforms can lose
    # the original pack's global scales and bypass the layer's validation.
    pack = dataclasses.replace(
        _pack(), **{field: torch.tensor([1.00390625, 0.71013], dtype=torch.float32)}
    )
    with pytest.raises(
        ValueError, match=f"{name} does not support global weight scales"
    ):
        getattr(moe_ep, name)(pack, intermediate_size=_I, hidden_size=_H)


def test_workspace_pool_accepts_list_tuning_values():
    group = object()
    with (
        mock.patch("torch.cuda.current_device", return_value=0),
        mock.patch.object(
            W4A16CutedslMegaKernelBackend,
            "ep_rank",
            new_callable=mock.PropertyMock,
            return_value=0,
        ),
        mock.patch.object(
            W4A16CutedslMegaKernelBackend,
            "ep_world_size",
            new_callable=mock.PropertyMock,
            return_value=1,
        ),
        mock.patch.object(
            W4A16CutedslMegaKernelBackend,
            "ep_comm_group",
            new_callable=mock.PropertyMock,
            return_value=group,
        ),
    ):
        keys = [
            W4A16CutedslMegaKernelBackend(
                dataclasses.replace(_config(), knobs={"epi_flag_batch": batch})
            )._workspace_pool_key(_fleet())
            for batch in ([1, 1], (1, 1))
        ]
    # JSON-loaded tuning values must share a hashable key with tuple values.
    pool = {keys[0]: group}
    assert pool[keys[1]] is group


def test_preparation_preserves_packed_weights_and_separate_fp32_globals(
    native_scale_interleave,
):
    source = _pack()
    alpha13 = torch.tensor([1.00390625, 0.71013], dtype=torch.float32)
    alpha2 = torch.tensor([0.83023, 1.17019], dtype=torch.float32)
    scaled = dataclasses.replace(
        source, w13_global_scale=alpha13, w2_global_scale=alpha2
    )
    plain, prepared = _prepare(source), _prepare(scaled)
    # Build the canonical row permutation independently of the backend helper.
    rows = torch.cat(
        [
            part
            for start in range(0, _I, 32)
            for part in (
                torch.arange(start, start + 32),
                torch.arange(_I + start, _I + start + 32),
            )
        ]
    )
    torch.testing.assert_close(prepared[0][0], source.w13[:, rows], rtol=0, atol=0)
    assert torch.equal(
        prepared[0][1].view(torch.uint8),
        _native_sf_bytes(source.w13_scale.view(torch.uint8)[:, rows]),
    )
    assert torch.equal(prepared[1][0], source.w2)
    assert torch.equal(
        prepared[1][1].view(torch.uint8), _native_sf_bytes(source.w2_scale)
    )
    assert native_scale_interleave.call_count == 4  # Two legs, two preparations.
    for default_leg, scaled_leg, alpha in zip(
        plain, prepared, (alpha13, alpha2), strict=True
    ):
        assert torch.equal(default_leg[0], scaled_leg[0])
        assert torch.equal(
            default_leg[1].view(torch.uint8), scaled_leg[1].view(torch.uint8)
        )
        torch.testing.assert_close(default_leg[2], torch.ones(_E), rtol=0, atol=0)
        torch.testing.assert_close(scaled_leg[2], alpha, rtol=0, atol=0)
        assert scaled_leg[2].dtype == torch.float32
        assert all(tensor.is_contiguous() for tensor in scaled_leg)


def test_pretransformed_weights_do_not_require_a_source_pack():
    transformed = _prepare(_pack())
    layer = _construct(
        None,
        MegaConfig(
            megakernel=_config(),
            preprocess_weights=False,
            transformed_weights=transformed,
        ),
    )
    layer.destroy()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("w13_global_scale", torch.ones(_E, dtype=torch.bfloat16), "FP32"),
        ("w2_global_scale", torch.ones(_E, 1), "local_experts"),
        ("w13_global_scale", torch.ones(_E, device="meta"), "weight device"),
        ("w13_scale", torch.ones(_E, 2 * _I, _H // 16), "linear E4M3"),
        ("w2", torch.zeros(_E, _H, _I // 2, dtype=torch.int8), "packed weights"),
    ),
)
def test_preparation_rejects_incompatible_weight_encodings(field, value, message):
    with pytest.raises(ValueError, match=message):
        _prepare(dataclasses.replace(_pack(), **{field: value}))


def _inputs(*, id_dtype=torch.int32, num_tokens=3):
    return MoEEpTensors(
        hidden_states=torch.zeros(num_tokens, _H, dtype=torch.bfloat16),
        topk_ids=torch.zeros(num_tokens, 2, dtype=id_dtype),
        topk_weights=torch.full((num_tokens, 2), 1.00390625, dtype=torch.float32),
    )


@pytest.mark.parametrize("id_dtype", (torch.int32, torch.int64))
@pytest.mark.parametrize("num_tokens", (0, 3))
def test_forward_accepts_scale_free_bf16_and_fp32_routing(id_dtype, num_tokens):
    backend = W4A16CutedslMegaKernelBackend(_config())
    backend.validate_forward(
        _inputs(id_dtype=id_dtype, num_tokens=num_tokens), _fleet(), quantize_input=True
    )


@pytest.mark.parametrize(
    "field", ("scales", "fc1_alpha", "fc2_alpha", "fc1_norm_const")
)
def test_forward_rejects_activation_quantization_fields(field):
    backend = W4A16CutedslMegaKernelBackend(_config())
    tensors = dataclasses.replace(_inputs(), **{field: torch.ones(1)})
    with pytest.raises(
        MoEEpConfigError, match="does not accept activation quantization"
    ):
        backend.validate_forward(tensors, _fleet(), quantize_input=True)


@pytest.mark.parametrize(
    ("field", "dtype", "message"),
    (
        ("hidden_states", torch.float16, "must be BF16"),
        ("hidden_states", torch.uint8, "must be BF16"),
        ("topk_ids", torch.float32, "int32 or int64"),
        ("topk_weights", torch.bfloat16, "must be FP32"),
    ),
)
def test_forward_rejects_changed_numeric_contract(field, dtype, message):
    backend = W4A16CutedslMegaKernelBackend(_config())
    tensors = _inputs()
    tensors = dataclasses.replace(tensors, **{field: getattr(tensors, field).to(dtype)})
    with pytest.raises(MoEEpConfigError, match=message):
        backend.validate_forward(tensors, _fleet(), quantize_input=True)


def test_forward_rejects_prequantized_activation_mode_and_capacity_overflow():
    backend = W4A16CutedslMegaKernelBackend(_config())
    with pytest.raises(MoEEpConfigError, match="quantize_input=True"):
        backend.validate_forward(_inputs(), _fleet(), quantize_input=False)
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        backend.validate_forward(_inputs(num_tokens=5), _fleet(), quantize_input=True)


@pytest.mark.parametrize("hidden,intermediate", [(64, 64), (192, 320), (256, 256)])
def test_native_sf_preparation_and_pretransformed_validation_with_tails(
    hidden, intermediate
):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.weights import (
        validate_transformed_mega_weights,
    )

    pack = PrequantizedMoEWeights(
        torch.zeros(_E, 2 * intermediate, hidden // 2, dtype=torch.uint8),
        torch.zeros(_E, hidden, intermediate // 2, dtype=torch.uint8),
        torch.zeros(_E, 2 * intermediate, hidden // 16, dtype=torch.float8_e4m3fn),
        torch.zeros(_E, hidden, intermediate // 16, dtype=torch.float8_e4m3fn),
    )
    prepared = preprocess_w4a16_cutedsl_mega_weights(
        pack, intermediate_size=intermediate, hidden_size=hidden
    )
    kwargs = dict(
        intermediate_size=intermediate, hidden_size=hidden, world_size=1, num_experts=_E
    )
    validate_transformed_mega_weights(prepared, **kwargs)
    # A caller supplying pretransformed weights must pass native flat scales,
    # including padded N rows and K/16 columns, rather than the canonical plane.
    bad_fc2 = (prepared[1][0], pack.w2_scale, prepared[1][2])
    with pytest.raises(ValueError, match="native flat E4M3"):
        validate_transformed_mega_weights((prepared[0], bad_fc2), **kwargs)
    truncated = (prepared[1][0], prepared[1][1][:-1], prepared[1][2])
    with pytest.raises(ValueError, match="native flat E4M3"):
        validate_transformed_mega_weights((prepared[0], truncated), **kwargs)
