# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""MiniMax-H3 one-pass QKV quantize-and-pack (issue #4532 candidate 7).

The GPU tests compare the fused output byte for byte against the segmented
FlashInfer reference: a torch destination-major copy followed by
``flashinfer.fp4_quantize`` (static global scale, swizzled 128x4 E4M3 block-16
scales) or ``flashinfer.mxfp8_quantize`` (swizzled 128x4 UE8M0 block-32
scales) applied independently per destination, including the zero padding rows
of every scale tile.
"""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

import flashinfer
from flashinfer.cake_minimax_h3 import MiniMaxH3QkvQuantizePack

NUM_HEADS = 56
HEAD_DIM = 128
QKV_KINDS = 3
FP4_BLOCK = 16
MXFP8_BLOCK = 32
PARTITIONS = (1, 2, 4, 8)
FORMATS = ("nvfp4", "mxfp8")
TOKEN_COUNTS = (1, 127, 129, 4097)

_RUN_TENSOR_NAMES = ("q", "k", "v", "out_global_scale", "out_q", "out_sf")


def _round_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _scale_cols(fmt: str) -> int:
    return HEAD_DIM // (FP4_BLOCK if fmt == "nvfp4" else MXFP8_BLOCK)


def _output_shapes(M: int, P: int, fmt: str):
    hpd = NUM_HEADS // P
    rows = M * hpd * QKV_KINDS
    scale_stride = _round_up(rows, 128) * _scale_cols(fmt)
    if fmt == "nvfp4":
        return (P, M, hpd, QKV_KINDS, HEAD_DIM // 2), torch.uint8, (P, scale_stride)
    return (P, M, hpd, QKV_KINDS, HEAD_DIM), torch.float8_e4m3fn, (P, scale_stride)


# ---------------------------------------------------------------------------
# Host-only tests (no GPU)
# ---------------------------------------------------------------------------


def test_prepared_api_preserves_caller_owned_outputs(monkeypatch) -> None:
    values = {name: object() for name in _RUN_TENSOR_NAMES}
    output = (values["out_q"], values["out_sf"])
    calls = []

    class _Prepared:
        format = "nvfp4"

        def __call__(self):
            calls.append("run")
            return output

    generated = ModuleType("flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack")

    def _prepare(**kwargs):
        calls.append(kwargs)
        return _Prepared()

    generated.prepare_minimax_h3_qkv_quantize_pack = _prepare
    monkeypatch.setitem(sys.modules, generated.__name__, generated)

    operation = MiniMaxH3QkvQuantizePack(**values, P=8, format="nvfp4")
    actual = operation.run(**values)

    assert calls[0]["P"] == 8 and calls[0]["format"] == "nvfp4"
    assert calls[-1] == "run"
    assert actual[0] is values["out_q"]
    assert actual[1] is values["out_sf"]
    assert operation.format == "nvfp4"

    with pytest.raises(ValueError, match="q"):
        operation.run(**{**values, "q": object()})


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm100a"), ((10, 3), "sm103a")],
)
def test_exact_architecture_router(monkeypatch, capability, expected) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_qkv_quantize_pack")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert router.minimax_h3_qkv_pack_target(torch.device("cuda")) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_qkv_quantize_pack")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        router.minimax_h3_qkv_pack_target(torch.device("cuda"))


def test_aot_inventory_covers_every_exact_route(monkeypatch) -> None:
    from flashinfer.jit import cake_minimax_h3_qkv_pack as jit

    calls = []

    class _PhysicalModule:
        @staticmethod
        def minimax_h3_qkv_pack_route_record(P, fmt):
            return {"target": "sm103a", "P": P, "format": fmt}

        @staticmethod
        def gen_minimax_h3_qkv_pack_module(P, fmt):
            calls.append((P, fmt))
            return SimpleNamespace(name=f"{fmt}_{P}")

    monkeypatch.setattr(jit.importlib, "import_module", lambda *_args: _PhysicalModule)
    specs = jit.gen_minimax_h3_qkv_pack_aot_modules("sm103a")

    assert jit.MINIMAX_H3_QKV_PACK_PARTITIONS == PARTITIONS
    assert jit.MINIMAX_H3_QKV_PACK_FORMATS == FORMATS
    assert len(calls) == 8
    assert len(specs) == 8
    assert {fmt for _, fmt in calls} == set(FORMATS)


@pytest.mark.parametrize(
    ("M", "P", "expected"),
    [
        (1, 8, (4, 8, 1)),
        (16, 8, (4, 8, 1)),
        (17, 8, (7, 8, 1)),
        (127, 8, (22, 8, 1)),
        (129, 8, (25, 8, 1)),
        (4824, 8, (794, 8, 1)),
        (9648, 4, (3167, 4, 1)),
        (19296, 2, (12664, 2, 1)),
        (38592, 1, (50653, 1, 1)),
        (38591, 1, (50653, 1, 1)),
    ],
)
def test_launch_grid_follows_the_runtime_token_count(M, P, expected) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_qkv_pack as ops

    record = {
        "launch_grid_rule": {
            "kind": "pack_warps_2d_plus_padding",
            "tokens_per_warp": 16,
            "warps_per_cta": 8,
        }
    }
    assert ops._stage_launch_grid(record, M=M, P=P) == expected
    with pytest.raises(RuntimeError, match="launch grid rule"):
        ops._stage_launch_grid({"launch_grid_rule": {"kind": "other"}}, M=M, P=P)


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize(("M", "P"), [(1, 8), (129, 8), (4097, 2), (38591, 1)])
def test_output_shapes_follow_the_send_buffer_abi(M, P, fmt) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_qkv_pack as ops

    q_shape, q_dtype, sf_shape = _output_shapes(M, P, fmt)
    shapes = ops.minimax_h3_qkv_pack_output_shapes(M, P, fmt)
    assert shapes["out_q"] == (q_shape, q_dtype)
    assert shapes["out_sf"] == (sf_shape, torch.uint8)
    assert ops.minimax_h3_qkv_pack_scale_stride(M, P, fmt) == sf_shape[1]


# ---------------------------------------------------------------------------
# GPU tests (exact SM100a / SM103a)
# ---------------------------------------------------------------------------


def _require_exact_blackwell() -> torch.device:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")
    return torch.device("cuda")


def _make_sources(M: int, device: torch.device, *, fused: bool, seed: int):
    generator = torch.Generator(device=device).manual_seed(seed)
    if fused:
        qkv = torch.empty(
            (M, NUM_HEADS, QKV_KINDS, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        qkv.normal_(0.0, 1.0, generator=generator)
        sources = [qkv[:, :, kind, :] for kind in range(QKV_KINDS)]
    else:
        sources = []
        for _ in range(QKV_KINDS):
            t = torch.empty(
                (M, NUM_HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
            )
            t.normal_(0.0, 1.0, generator=generator)
            sources.append(t)
    # Exercise exact-zero blocks (zero scale byte, zero codes) as well.
    sources[0][::7] = 0
    return sources


def _global_encode_scale(sources) -> torch.Tensor:
    amax = torch.stack([s.float().abs().amax() for s in sources]).amax()
    scale = (448.0 * 6.0) / amax
    return (
        torch.where(torch.isfinite(scale), scale, torch.ones_like(scale))
        .reshape(1)
        .float()
    )


def _destination_major(sources, P: int) -> torch.Tensor:
    M = sources[0].shape[0]
    stacked = torch.stack(tuple(sources), dim=2)  # [M, 56, 3, 128]
    return (
        stacked.view(M, P, NUM_HEADS // P, QKV_KINDS, HEAD_DIM)
        .permute(1, 0, 2, 3, 4)
        .contiguous()
    )


def _swizzled_padding_mask(rows: int, cols: int, device: torch.device) -> torch.Tensor:
    """Boolean mask of the padding-row bytes inside one swizzled 128x4 scale tile."""
    padded_rows = _round_up(rows, 128)
    padded_cols = _round_up(cols, 4)
    index = torch.arange(padded_rows * padded_cols, dtype=torch.int64, device=device)
    tile = index // (128 * padded_cols)
    within = index % (128 * padded_cols)
    col_group = within // 512
    rem = within % 512
    row_in_32 = rem // 16
    rem = rem % 16
    row_group = rem // 4
    col = col_group * 4 + rem % 4
    row = tile * 128 + row_group * 32 + row_in_32
    return (row >= rows) | (col >= cols)


def _segmented_reference(sources, P: int, fmt: str, global_scale):
    """Torch destination copy + FlashInfer quantizer per destination (the fallback chain)."""
    destination = _destination_major(sources, P)
    rows = destination.shape[1] * destination.shape[2] * destination.shape[3]
    values, scales = [], []
    padding = _swizzled_padding_mask(rows, _scale_cols(fmt), destination.device)
    for shard in destination:
        flat = shard.reshape(rows, HEAD_DIM)
        if fmt == "nvfp4":
            x_q, sf = flashinfer.fp4_quantize(
                flat,
                global_scale,
                sf_vec_size=FP4_BLOCK,
                sf_use_ue8m0=False,
                is_sf_swizzled_layout=True,
            )
            values.append(
                x_q.view(torch.uint8).reshape(*shard.shape[:-1], HEAD_DIM // 2)
            )
        else:
            x_q, sf = flashinfer.mxfp8_quantize(flat, is_sf_swizzled_layout=True)
            values.append(
                x_q.view(torch.float8_e4m3fn).reshape(*shard.shape[:-1], HEAD_DIM)
            )
        sf = sf.reshape(-1).view(torch.uint8).clone()
        # The send-buffer ABI defines the scale-tile padding rows as zero bytes
        # (the FlashInfer quantizers write them as such); pin that explicitly so
        # the comparison does not depend on uninitialised storage.
        sf[padding] = 0
        scales.append(sf)
    return torch.stack(values), torch.stack(scales)


def _prepare(sources, P: int, fmt: str, device: torch.device):
    M = sources[0].shape[0]
    q_shape, q_dtype, sf_shape = _output_shapes(M, P, fmt)
    out_q = torch.empty(q_shape, dtype=q_dtype, device=device)
    out_sf = torch.empty(sf_shape, dtype=torch.uint8, device=device)
    global_scale = _global_encode_scale(sources) if fmt == "nvfp4" else None
    values = {
        "q": sources[0],
        "k": sources[1],
        "v": sources[2],
        "out_global_scale": global_scale,
        "out_q": out_q,
        "out_sf": out_sf,
    }
    operation = MiniMaxH3QkvQuantizePack(**values, P=P, format=fmt)
    return operation, values


def _assert_exact(values, expected_q, expected_sf) -> None:
    actual_q = values["out_q"].view(torch.uint8)
    assert torch.equal(actual_q, expected_q.view(torch.uint8)), (
        f"packed value bytes differ: {int((actual_q != expected_q.view(torch.uint8)).sum())} mismatches"
    )
    assert torch.equal(values["out_sf"], expected_sf), (
        f"scale bytes differ: {int((values['out_sf'] != expected_sf).sum())} mismatches"
    )


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("P", PARTITIONS)
@pytest.mark.parametrize("M", TOKEN_COUNTS)
def test_pack_matches_segmented_flashinfer_chain_bitwise(M, P, fmt) -> None:
    device = _require_exact_blackwell()
    sources = _make_sources(M, device, fused=False, seed=613_000 + 10 * M + P)
    operation, values = _prepare(sources, P, fmt, device)
    expected_q, expected_sf = _segmented_reference(
        sources, P, fmt, values["out_global_scale"]
    )

    # 0xFF sentinels: any byte the kernel fails to write (including the scale
    # tile's padding rows, which must come back as zero) fails the exact check.
    values["out_q"].view(torch.uint8).fill_(0xFF)
    values["out_sf"].fill_(0xFF)
    actual_q, actual_sf = operation.run(**values)
    torch.cuda.synchronize()

    assert actual_q is values["out_q"]
    assert actual_sf is values["out_sf"]
    _assert_exact(values, expected_q, expected_sf)


@pytest.mark.parametrize("fmt", FORMATS)
def test_fused_projection_slices_are_read_in_place(fmt) -> None:
    device = _require_exact_blackwell()
    M, P = 4097, 2
    sources = _make_sources(M, device, fused=True, seed=613_777)
    assert (
        sources[0].stride(1) == QKV_KINDS * HEAD_DIM
    )  # kind slices of [M, 56, 3, 128]
    operation, values = _prepare(sources, P, fmt, device)
    expected_q, expected_sf = _segmented_reference(
        sources, P, fmt, values["out_global_scale"]
    )
    values["out_q"].view(torch.uint8).fill_(0xFF)
    values["out_sf"].fill_(0xFF)
    operation.run(**values)
    torch.cuda.synchronize()
    _assert_exact(values, expected_q, expected_sf)


@pytest.mark.parametrize("fmt", FORMATS)
def test_prepared_api_cuda_graph_replay(fmt) -> None:
    device = _require_exact_blackwell()
    M, P = 129, 8
    sources = _make_sources(M, device, fused=False, seed=613_101)
    operation, values = _prepare(sources, P, fmt, device)
    expected_q, expected_sf = _segmented_reference(
        sources, P, fmt, values["out_global_scale"]
    )

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        operation.run(**values)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_sf = operation.run(**values)
    values["out_q"].view(torch.uint8).fill_(0xFF)
    values["out_sf"].fill_(0xFF)
    graph.replay()
    torch.cuda.synchronize()

    assert captured_q is values["out_q"]
    assert captured_sf is values["out_sf"]
    _assert_exact(values, expected_q, expected_sf)

    # Same graph, new source contents: the replay must follow the data.
    for source in sources:
        source.mul_(-0.5)
    expected_q, expected_sf = _segmented_reference(
        sources, P, fmt, values["out_global_scale"]
    )
    graph.replay()
    torch.cuda.synchronize()
    _assert_exact(values, expected_q, expected_sf)


@pytest.mark.parametrize("fmt", FORMATS)
def test_prepared_instances_run_on_independent_streams(fmt) -> None:
    device = _require_exact_blackwell()
    first_sources = _make_sources(127, device, fused=False, seed=613_201)
    second_sources = _make_sources(4097, device, fused=False, seed=613_202)
    first, first_values = _prepare(first_sources, 8, fmt, device)
    second, second_values = _prepare(second_sources, 4, fmt, device)
    first_expected = _segmented_reference(
        first_sources, 8, fmt, first_values["out_global_scale"]
    )
    second_expected = _segmented_reference(
        second_sources, 4, fmt, second_values["out_global_scale"]
    )

    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    first_stream.wait_stream(current_stream)
    second_stream.wait_stream(current_stream)
    with torch.cuda.stream(first_stream):
        first_values["out_sf"].fill_(0xFF)
        first_q, first_sf = first.run(**first_values)
    with torch.cuda.stream(second_stream):
        second_values["out_sf"].fill_(0xFF)
        second_q, second_sf = second.run(**second_values)
    first_stream.synchronize()
    second_stream.synchronize()

    assert first_q is first_values["out_q"] and first_sf is first_values["out_sf"]
    assert second_q is second_values["out_q"] and second_sf is second_values["out_sf"]
    _assert_exact(first_values, *first_expected)
    _assert_exact(second_values, *second_expected)


def test_functional_wrapper_allocates_outputs() -> None:
    device = _require_exact_blackwell()
    from flashinfer.diffusion_ops import minimax_h3_qkv_quantize_pack

    sources = _make_sources(129, device, fused=False, seed=613_301)
    for fmt in FORMATS:
        global_scale = _global_encode_scale(sources) if fmt == "nvfp4" else None
        out_q, out_sf = minimax_h3_qkv_quantize_pack(
            *sources, P=8, format=fmt, out_global_scale=global_scale
        )
        torch.cuda.synchronize()
        expected_q, expected_sf = _segmented_reference(sources, 8, fmt, global_scale)
        q_shape, q_dtype, sf_shape = _output_shapes(129, 8, fmt)
        assert tuple(out_q.shape) == q_shape and out_q.dtype == q_dtype
        assert tuple(out_sf.shape) == sf_shape
        _assert_exact({"out_q": out_q, "out_sf": out_sf}, expected_q, expected_sf)


def test_prepare_rejects_mismatched_arguments() -> None:
    device = _require_exact_blackwell()
    from flashinfer.diffusion_ops.cake_minimax_h3_qkv_pack import (
        prepare_minimax_h3_qkv_quantize_pack,
    )

    sources = _make_sources(8, device, fused=False, seed=613_401)
    q_shape, q_dtype, sf_shape = _output_shapes(8, 8, "nvfp4")
    out_q = torch.empty(q_shape, dtype=q_dtype, device=device)
    out_sf = torch.empty(sf_shape, dtype=torch.uint8, device=device)
    with pytest.raises(ValueError, match="out_global_scale is required"):
        prepare_minimax_h3_qkv_quantize_pack(
            q=sources[0],
            k=sources[1],
            v=sources[2],
            out_q=out_q,
            out_sf=out_sf,
            P=8,
            format="nvfp4",
        )
    with pytest.raises(ValueError, match="P must be one of"):
        prepare_minimax_h3_qkv_quantize_pack(
            q=sources[0],
            k=sources[1],
            v=sources[2],
            out_q=out_q,
            out_sf=out_sf,
            P=3,
            format="mxfp8",
        )
    with pytest.raises(ValueError, match="same token and head strides"):
        prepare_minimax_h3_qkv_quantize_pack(
            q=sources[0],
            k=sources[1],
            v=sources[2].transpose(0, 1).contiguous().transpose(0, 1),
            out_q=out_q,
            out_sf=out_sf,
            P=8,
            format="mxfp8",
        )


@pytest.mark.parametrize("fused", [False, True])
def test_flat_source_views_cover_the_addressed_span(fused: bool) -> None:
    from flashinfer.diffusion_ops import cake_minimax_h3_qkv_pack as ops

    M = 5
    if fused:
        parent = torch.arange(M * 56 * 3 * 128, dtype=torch.float32).to(torch.bfloat16)
        parent = parent.view(M, 56, 3, 128)
        sources = [parent[:, :, kind] for kind in range(3)]
    else:
        sources = [torch.randn(M, 56, 128, dtype=torch.bfloat16) for _ in range(3)]
    token_stride, head_stride = sources[0].stride(0), sources[0].stride(1)
    for source in sources:
        flat = ops._flat_source_view(source, M, token_stride, head_stride)
        assert flat.dim() == 1 and flat.is_contiguous()
        assert flat.data_ptr() == source.data_ptr()
        assert flat.numel() == (M - 1) * token_stride + 55 * head_stride + 128
        assert flat[3 * token_stride + 7 * head_stride + 9] == source[3, 7, 9]
