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

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from flashinfer.cake_minimax_h3 import MiniMaxH3Mxfp8PreAttention


_RUN_TENSOR_NAMES = (
    "x",
    "x_norm_weight",
    "adaln_scale",
    "adaln_shift",
    "adaln_index",
    "qkv_weight_q",
    "qkv_weight_sf",
    "q_norm_weight",
    "k_norm_weight",
    "rope_cos_sin",
    "out_q",
    "out_sf",
)


def test_prepared_api_preserves_caller_owned_outputs(monkeypatch) -> None:
    values = {name: object() for name in _RUN_TENSOR_NAMES}
    output = (values["out_q"], values["out_sf"])
    calls = []

    class _Prepared:
        def __call__(self):
            calls.append("run")
            return output

    generated = ModuleType("flashinfer.diffusion_ops.cake_minimax_h3_mxfp8")

    def _prepare(**kwargs):
        calls.append(kwargs)
        return _Prepared()

    generated.prepare_minimax_h3_mxfp8_pre_attention = _prepare
    monkeypatch.setitem(sys.modules, generated.__name__, generated)

    operation = MiniMaxH3Mxfp8PreAttention(
        **values,
        activation_q=object(),
        activation_sf=object(),
        qkv_bf16=object(),
        gemm_workspace=object(),
        P=8,
    )
    actual = operation.run(**values)

    assert calls[-1] == "run"
    assert actual[0] is values["out_q"]
    assert actual[1] is values["out_sf"]

    with pytest.raises(ValueError, match="x"):
        operation.run(**{**values, "x": object()})


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm100a"), ((10, 3), "sm103a")],
)
def test_exact_architecture_router(monkeypatch, capability, expected) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_mxfp8_pre_attention")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: capability)
    assert router.minimax_h3_mxfp8_target(torch.device("cuda")) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch) -> None:
    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_mxfp8_pre_attention")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        router.minimax_h3_mxfp8_target(torch.device("cuda"))


def _aligned_workspace(size: int, device: torch.device):
    if size == 0:
        return None, None
    backing = torch.empty((size + 127,), dtype=torch.uint8, device=device)
    offset = (-int(backing.data_ptr())) % 128
    return backing, backing[offset : offset + size]


def _prepare_zero_smoke(adaln_index: int, M: int = 1, P: int = 8):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100a or SM103a")

    router = pytest.importorskip("flashinfer.jit.cake_minimax_h3_mxfp8_pre_attention")
    from flashinfer.gemm import gemm_base

    hidden, qkv_width, head_dim = 5376, 21504, 128
    device = torch.device("cuda")
    rows_per_destination = M * (56 // P) * 3
    out_sf_stride = ((rows_per_destination + 127) // 128 * 128) * (head_dim // 32)
    activation_sf_len = ((M + 127) // 128 * 128) * (hidden // 32)
    values = {
        "x": torch.zeros((M, hidden), dtype=torch.bfloat16, device=device),
        "x_norm_weight": torch.ones((hidden,), dtype=torch.bfloat16, device=device),
        "adaln_scale": torch.zeros((9, hidden), dtype=torch.bfloat16, device=device),
        "adaln_shift": torch.zeros((9, hidden), dtype=torch.bfloat16, device=device),
        "adaln_index": torch.full((M,), adaln_index, dtype=torch.int32, device=device),
        "qkv_weight_q": torch.zeros(
            (qkv_width, hidden), dtype=torch.float8_e4m3fn, device=device
        ),
        "qkv_weight_sf": torch.zeros(
            (qkv_width * (hidden // 32),), dtype=torch.uint8, device=device
        ),
        "q_norm_weight": torch.ones((head_dim,), dtype=torch.bfloat16, device=device),
        "k_norm_weight": torch.ones((head_dim,), dtype=torch.bfloat16, device=device),
        "rope_cos_sin": torch.zeros((M, 96), dtype=torch.bfloat16, device=device),
        "out_q": torch.ones(
            (P, M, 56 // P, 3, head_dim),
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        "out_sf": torch.full((P, out_sf_stride), 255, dtype=torch.uint8, device=device),
    }
    route = router.minimax_h3_mxfp8_route_record(device, M, P)
    norm_backing, norm_workspace = _aligned_workspace(
        int(route["stages"]["norm_adaln_mxfp8_quantize"]["tma_workspace_bytes"]),
        device,
    )
    post_backing, post_workspace = _aligned_workspace(
        int(route["stages"]["qk_rope_destination_mxfp8_pack"]["tma_workspace_bytes"]),
        device,
    )
    operation = MiniMaxH3Mxfp8PreAttention(
        **values,
        activation_q=torch.empty((M, hidden), dtype=torch.float8_e4m3fn, device=device),
        activation_sf=torch.empty(
            (activation_sf_len,), dtype=torch.uint8, device=device
        ),
        qkv_bf16=torch.empty((M, qkv_width), dtype=torch.bfloat16, device=device),
        gemm_workspace=torch.empty(
            (int(gemm_base.DEFAULT_WORKSPACE_SIZE),),
            dtype=torch.uint8,
            device=device,
        ),
        P=P,
        norm_descriptor_workspace=norm_workspace,
        post_descriptor_workspace=post_workspace,
    )
    return (
        operation,
        values,
        norm_backing,
        norm_workspace,
        post_backing,
        post_workspace,
    )


@pytest.mark.parametrize("invalid_index", [-1, 9, -(2**31), 2**31 - 1])
def test_invalid_adaln_row_writes_zero_to_caller_outputs(invalid_index) -> None:
    operation, values, norm_backing, norm_workspace, post_backing, post_workspace = (
        _prepare_zero_smoke(invalid_index)
    )
    actual_q, actual_sf = operation.run(**values)
    torch.cuda.synchronize()

    assert actual_q is values["out_q"]
    assert actual_sf is values["out_sf"]
    assert torch.count_nonzero(actual_q.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(actual_sf).item() == 0
    assert norm_backing is not None or norm_workspace is None
    assert post_backing is not None or post_workspace is None


def test_prepared_api_cuda_graph_replay() -> None:
    operation, values, *_workspaces = _prepare_zero_smoke(-1)
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        operation.run(**values)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_q, captured_sf = operation.run(**values)
    values["out_q"].fill_(1)
    values["out_sf"].fill_(255)
    graph.replay()
    torch.cuda.synchronize()

    assert captured_q is values["out_q"]
    assert captured_sf is values["out_sf"]
    assert torch.count_nonzero(captured_q.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(captured_sf).item() == 0


def test_prepared_instances_run_on_independent_streams() -> None:
    first, first_values, first_norm, _first_norm_view, first_post, _first_post_view = (
        _prepare_zero_smoke(-1)
    )
    (
        second,
        second_values,
        second_norm,
        _second_norm_view,
        second_post,
        _second_post_view,
    ) = _prepare_zero_smoke(0)
    second_values["adaln_shift"][0].fill_(1)
    second_values["qkv_weight_q"].fill_(1)
    second_values["qkv_weight_sf"].fill_(127)
    first_stream = torch.cuda.Stream()
    second_stream = torch.cuda.Stream()
    current_stream = torch.cuda.current_stream()
    first_stream.wait_stream(current_stream)
    second_stream.wait_stream(current_stream)

    with torch.cuda.stream(first_stream):
        first_q, first_sf = first.run(**first_values)
    with torch.cuda.stream(second_stream):
        second_q, second_sf = second.run(**second_values)
    first_stream.synchronize()
    second_stream.synchronize()

    assert first_q is first_values["out_q"]
    assert first_sf is first_values["out_sf"]
    assert second_q is second_values["out_q"]
    assert second_sf is second_values["out_sf"]
    assert torch.count_nonzero(first_q.view(torch.uint8)).item() == 0
    assert torch.count_nonzero(first_sf).item() == 0
    assert torch.count_nonzero(second_q.view(torch.uint8)).item() > 0
    assert torch.count_nonzero(second_sf).item() > 0
    if first_norm is not None:
        assert second_norm is not None
        assert first_norm is not second_norm
    if first_post is not None:
        assert second_post is not None
        assert first_post is not second_post


def test_aot_inventory_covers_every_exact_route(monkeypatch) -> None:
    from flashinfer.jit import cake_minimax_h3_mxfp8 as jit

    calls = []

    class _PhysicalModule:
        @staticmethod
        def minimax_h3_mxfp8_route_record(M, P):
            return {"target": "sm103a", "M": M, "P": P}

        @staticmethod
        def gen_minimax_h3_mxfp8_stage_module(M, P, stage):
            calls.append((M, P, stage))
            return SimpleNamespace(name=f"{M}_{P}_{stage}")

    monkeypatch.setattr(jit.importlib, "import_module", lambda *_args: _PhysicalModule)
    specs = jit.gen_minimax_h3_mxfp8_aot_modules("sm103a")

    assert len(jit.MINIMAX_H3_MXFP8_SHAPES) == 44
    assert len(calls) == 88
    assert len(specs) == 88
    assert {stage for _, _, stage in calls} == {
        "norm_adaln_mxfp8_quantize",
        "qk_rope_destination_mxfp8_pack",
    }
