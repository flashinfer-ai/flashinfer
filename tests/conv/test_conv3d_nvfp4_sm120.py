# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch.nn.functional as F

from flashinfer import conv3d_nvfp4, prepare_nvfp4_conv3d_weight
from flashinfer.conv.nvfp4 import _quantize_nvfp4_conv3d_activation


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0:2] != (12, 0)
    or not torch.version.cuda
    or int(torch.version.cuda.split(".")[0]) < 13,
    reason="SM120 NVFP4 Conv3d requires SM120 and CUDA 13+",
)

_E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _decode_e2m1(packed: torch.Tensor, rows: int, columns: int) -> torch.Tensor:
    lut = torch.tensor(_E2M1_VALUES, device=packed.device, dtype=torch.float32)
    packed_int = packed.reshape(rows, columns // 2).to(torch.int64)
    codes = torch.stack(
        (packed_int & 0xF, (packed_int >> 4) & 0xF),
        dim=-1,
    ).reshape(rows, columns)
    return lut[codes]


def _unswizzle_128x4(
    scale: torch.Tensor,
    rows: int,
    scale_columns: int,
) -> torch.Tensor:
    scale_flat = scale.contiguous().view(torch.uint8).reshape(-1)
    scale_column_blocks = (scale_columns + 3) // 4
    row_index = torch.arange(rows, device=scale.device, dtype=torch.int64)
    column_index = torch.arange(
        scale_columns,
        device=scale.device,
        dtype=torch.int64,
    )
    row_grid, column_grid = torch.meshgrid(
        row_index,
        column_index,
        indexing="ij",
    )
    offsets = (
        ((row_grid // 128) * scale_column_blocks + column_grid // 4) * 512
        + (row_grid % 32) * 16
        + ((row_grid % 128) // 32) * 4
        + column_grid % 4
    )
    return scale_flat[offsets]


def _dequantize_activation(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    batch, depth, height, width, packed_channels = map(int, packed.shape)
    channels = packed_channels * 2
    rows = batch * depth * height * width
    values = _decode_e2m1(packed, rows, channels)
    local_scale = scale.reshape(rows, channels // 16).view(torch.float8_e4m3fn)
    values = values * local_scale.float().repeat_interleave(16, dim=1)
    values = values * torch.reciprocal(global_scale)
    return (
        values.reshape(batch, depth, height, width, channels)
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def _dequantize_weight(
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    output_channels, filter_t, filter_r, filter_s, packed_channels = map(
        int, packed.shape
    )
    input_channels = packed_channels * 2
    flattened_columns = input_channels * filter_t * filter_r * filter_s
    values = _decode_e2m1(
        packed,
        output_channels,
        flattened_columns,
    )
    local_scale = _unswizzle_128x4(
        scale,
        output_channels,
        flattened_columns // 16,
    ).view(torch.float8_e4m3fn)
    values = values * local_scale.float().repeat_interleave(16, dim=1)
    values = values * torch.reciprocal(global_scale)
    return (
        values.reshape(
            output_channels,
            filter_t,
            filter_r,
            filter_s,
            input_channels,
        )
        .permute(0, 4, 1, 2, 3)
        .contiguous()
    )


def _make_problem(
    *,
    input_channels: int = 128,
    output_channels: int = 128,
    spatial_shape: tuple[int, int, int] = (4, 5, 7),
):
    torch.manual_seed(7)
    input = torch.randn(
        (1, input_channels, *spatial_shape),
        device="cuda",
        dtype=torch.bfloat16,
    )
    weight = (
        torch.randn(
            (output_channels, input_channels, 3, 3, 3),
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.02
    )
    input_global_scale = (
        448.0 * 6.0 / input.abs().amax().float().clamp(min=1e-8)
    ).reshape(1)
    packed_weight, weight_scale, weight_global_scale = prepare_nvfp4_conv3d_weight(
        weight
    )
    return (
        input,
        weight,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    )


def _exact_quantized_reference(
    input: torch.Tensor,
    packed_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    input_global_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: torch.Tensor | None,
    padding: tuple[int, int, int],
) -> torch.Tensor:
    packed_input, input_scale = _quantize_nvfp4_conv3d_activation(
        input,
        input_global_scale,
        padding,
    )
    dequantized_input = _dequantize_activation(
        packed_input,
        input_scale,
        input_global_scale,
    )
    dequantized_weight = _dequantize_weight(
        packed_weight,
        weight_scale,
        weight_global_scale,
    )
    return F.conv3d(
        dequantized_input,
        dequantized_weight,
        bias.float() if bias is not None else None,
    ).to(torch.bfloat16)


@pytest.mark.parametrize("padding", [(0, 0, 0), (0, 1, 1)])
@pytest.mark.parametrize("bias_dtype", [None, torch.bfloat16, torch.float32])
def test_conv3d_nvfp4_matches_exact_quantized_reference(padding, bias_dtype):
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    bias = None
    if bias_dtype is not None:
        bias = torch.linspace(
            -0.25,
            0.25,
            128,
            device="cuda",
            dtype=bias_dtype,
        )

    actual = conv3d_nvfp4(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        padding=padding,
    )
    expected = _exact_quantized_reference(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        padding,
    )

    assert actual.is_contiguous(memory_format=torch.channels_last_3d)
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.01,
        atol=0.03125,
    )
    cosine = F.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert cosine > 0.99998


@pytest.mark.parametrize(
    (
        "input_channels",
        "output_channels",
        "spatial_shape",
        "padding",
    ),
    [
        pytest.param(
            128,
            128,
            (4, 3, 3),
            (0, 0, 0),
            id="minimal-supported-output-m2-k128-tail",
        ),
        pytest.param(
            128,
            256,
            (4, 5, 7),
            (0, 1, 1),
            id="small-output-m70-k256-n-pair",
        ),
        pytest.param(
            256,
            384,
            (5, 6, 9),
            (0, 0, 0),
            id="odd-n-tile-count-k384",
        ),
        pytest.param(
            384,
            512,
            (4, 8, 10),
            (0, 1, 1),
            id="non-power-of-two-c384-k512",
        ),
        pytest.param(
            512,
            256,
            (6, 7, 11),
            (0, 0, 0),
            id="deep-reduction-c512-k256",
        ),
    ],
)
def test_conv3d_nvfp4_shape_matrix(
    input_channels,
    output_channels,
    spatial_shape,
    padding,
):
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem(
        input_channels=input_channels,
        output_channels=output_channels,
        spatial_shape=spatial_shape,
    )
    bias = torch.linspace(
        -0.25,
        0.25,
        output_channels,
        device="cuda",
        dtype=torch.bfloat16,
    )

    actual = conv3d_nvfp4(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        padding=padding,
    )
    expected = _exact_quantized_reference(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        bias,
        padding,
    )

    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.01,
        atol=0.03125,
    )
    cosine = F.cosine_similarity(
        actual.float().flatten(),
        expected.float().flatten(),
        dim=0,
    )
    assert cosine > 0.99998


def test_conv3d_nvfp4_out_buffer_and_current_stream():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    expected_shape = (1, 128, 2, 5, 7)
    out = torch.empty(
        expected_shape,
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )

    conv3d_nvfp4(
        input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        out=out,
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        result = conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            out=out,
        )
        event = torch.cuda.Event()
        event.record()
    event.synchronize()

    assert result.data_ptr() == out.data_ptr()
    assert torch.isfinite(result).all()
    assert result.count_nonzero() > 0


def test_conv3d_nvfp4_cuda_graph_replay():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    static_input = input.clone()
    static_out = torch.empty(
        (1, 128, 2, 5, 7),
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    conv3d_nvfp4(
        static_input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        out=static_out,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        conv3d_nvfp4(
            static_input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            out=static_out,
        )

    static_input.copy_(input * 0.5)
    graph.replay()
    graph_result = static_out.clone()
    eager_out = torch.empty_like(static_out)
    eager_result = conv3d_nvfp4(
        static_input,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
        out=eager_out,
    )
    torch.cuda.synchronize()
    assert torch.equal(graph_result, eager_result)


def test_conv3d_nvfp4_torch_compile_fullgraph():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    eager_out = torch.empty(
        (1, 128, 2, 5, 7),
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    compiled_out = torch.empty_like(eager_out)

    def operation(input_arg, out_arg):
        return conv3d_nvfp4(
            input_arg,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            out=out_arg,
        )

    expected = operation(input, eager_out).clone()
    actual = torch.compile(operation, fullgraph=True)(input, compiled_out)
    torch.cuda.synchronize()
    assert actual.data_ptr() == compiled_out.data_ptr()
    assert torch.equal(actual, expected)


def test_conv3d_nvfp4_trace_schema():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    out = torch.empty(
        (1, 128, 2, 5, 7),
        device="cuda",
        dtype=torch.bfloat16,
        memory_format=torch.channels_last_3d,
    )
    trace = conv3d_nvfp4.fi_trace(
        input=input,
        packed_weight=packed_weight,
        weight_scale=weight_scale,
        input_global_scale=input_global_scale,
        weight_global_scale=weight_global_scale,
        bias=None,
        stride=(1, 1, 1),
        padding=(0, 1, 1),
        dilation=(1, 1, 1),
        groups=1,
        out=out,
    )
    assert trace["name"] == "conv3d_nvfp4_sm120_n1_c128_k128_t3_r3_s3"
    assert trace["axes"]["output_depth"]["type"] == "var"
    assert trace["inputs"]["packed_weight"]["dtype"] == "uint8"
    assert trace["outputs"]["output"]["dtype"] == "bfloat16"


def test_conv3d_nvfp4_fake_tensor_shape_and_strides():
    from torch._subclasses.fake_tensor import FakeTensorMode

    with FakeTensorMode():
        input = torch.empty(
            (1, 128, 4, 5, 7),
            device="cuda",
            dtype=torch.bfloat16,
        )
        packed_weight = torch.empty(
            (128, 3, 3, 3, 64),
            device="cuda",
            dtype=torch.uint8,
        )
        weight_scale = torch.empty(
            (27648,),
            device="cuda",
            dtype=torch.uint8,
        )
        input_global_scale = torch.empty(
            (1,),
            device="cuda",
            dtype=torch.float32,
        )
        weight_global_scale = torch.empty_like(input_global_scale)
        output = conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
        )

    assert tuple(output.shape) == (1, 128, 2, 5, 7)
    assert output.stride() == (8960, 1, 4480, 896, 128)
    assert output.is_contiguous(memory_format=torch.channels_last_3d)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"stride": (1, 2, 1)}, "stride"),
        ({"padding": (1, 1, 1)}, "padding"),
        ({"dilation": (1, 2, 1)}, "dilation"),
        ({"groups": 2}, "groups"),
    ],
)
def test_conv3d_nvfp4_rejects_unsupported_convolution_parameters(kwargs, message):
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    with pytest.raises(ValueError, match=message):
        conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            **kwargs,
        )


def test_conv3d_nvfp4_rejects_wrong_output_layout():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem()
    out = torch.empty(
        (1, 128, 2, 5, 7),
        device="cuda",
        dtype=torch.bfloat16,
    )
    with pytest.raises(ValueError, match="channels_last_3d"):
        conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            out=out,
        )


def test_conv3d_nvfp4_rejects_single_output_spatial_position():
    (
        input,
        _,
        packed_weight,
        weight_scale,
        input_global_scale,
        weight_global_scale,
    ) = _make_problem(spatial_shape=(3, 3, 3))
    with pytest.raises(ValueError, match="at least two output spatial positions"):
        conv3d_nvfp4(
            input,
            packed_weight,
            weight_scale,
            input_global_scale,
            weight_global_scale,
            padding=(0, 0, 0),
        )
