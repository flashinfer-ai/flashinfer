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
"""The MiniMax-H3 SM120 quantized operators ship sm_120a cubins only; on any other compute
capability (notably SM121 DGX Spark, #5723) the public entry points must raise
``NotImplementedError`` before reaching the kernel instead of a CUDA ``no kernel image`` error."""

import importlib

import pytest
import torch

import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_fc1_swiglu as fc1_sm120
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_gate as gate_sm120
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_out_proj as out_proj_sm120
import flashinfer.diffusion_ops.cake_minimax_h3_sm120_quant_pre_attention as pre_sm120

# flashinfer.diffusion_ops re-exports a function named minimax_h3_fc1_swiglu, so import the module by name.
fc1_public = importlib.import_module("flashinfer.diffusion_ops.minimax_h3_fc1_swiglu")

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)

HIDDEN = 5376
ATTN_DIM = 7168
QKV_WIDTH = 21504
FC1_ROWS = 28672
ROWS = 4


@pytest.fixture
def mock_cc(monkeypatch, request):
    cc = request.param
    for module in (gate_sm120, fc1_public):
        monkeypatch.setattr(
            module, "get_compute_capability", lambda device, _cc=cc: _cc
        )
    return cc


def _bf16(shape, device):
    return torch.ones(shape, dtype=torch.bfloat16, device=device)


def _u8(shape, device):
    return torch.zeros(shape, dtype=torch.uint8, device=device)


def _f32(shape, device):
    return torch.ones(shape, dtype=torch.float32, device=device)


def _fp8(shape, device):
    return torch.empty(shape, dtype=torch.float8_e4m3fn, device=device)


def _calls(device):
    x = _bf16((ROWS, HIDDEN), device)
    norm = _bf16((HIDDEN,), device)
    adaln = _bf16((9, HIDDEN), device)
    index = torch.zeros((ROWS,), dtype=torch.int32, device=device)
    head = _bf16((128,), device)
    rope = _bf16((ROWS, 96), device)
    index64 = torch.zeros((ROWS,), dtype=torch.int64, device=device)
    return {
        "fp8_pre_attention": lambda: pre_sm120.minimax_h3_fp8_pre_attention(
            x,
            norm,
            adaln,
            adaln,
            index,
            _fp8((QKV_WIDTH, HIDDEN), device),
            _f32((QKV_WIDTH,), device),
            head,
            head,
            rope,
        ),
        "nvfp4_pre_attention": lambda: pre_sm120.minimax_h3_nvfp4_pre_attention(
            x,
            norm,
            adaln,
            adaln,
            index,
            _u8((QKV_WIDTH, HIDDEN // 2), device),
            _u8((QKV_WIDTH * 336,), device),
            1.0,
            _f32((1,), device),
            head,
            head,
            rope,
        ),
        "fc1_swiglu_fp8": lambda: fc1_sm120.minimax_h3_fc1_swiglu_fp8(
            x,
            norm,
            adaln,
            adaln,
            index,
            _fp8((FC1_ROWS, HIDDEN), device),
            _f32((FC1_ROWS,), device),
        ),
        "fc1_swiglu_nvfp4_public": lambda: fc1_public.minimax_h3_fc1_swiglu_nvfp4(
            x,
            norm,
            adaln,
            adaln,
            index,
            _f32((1,), device),
            _u8((FC1_ROWS, HIDDEN // 2), device),
            _u8((FC1_ROWS * 336,), device),
            _f32((1,), device),
        ),
        "fp8_out_proj": lambda: out_proj_sm120.minimax_h3_fp8_out_proj(
            _bf16((ROWS, ATTN_DIM), device),
            _fp8((HIDDEN, ATTN_DIM), device),
            _f32((HIDDEN,), device),
            adaln,
            index,
            x,
        ),
        "nvfp4_out_proj": lambda: out_proj_sm120.minimax_h3_nvfp4_out_proj(
            _bf16((ROWS, ATTN_DIM), device),
            _u8((HIDDEN, ATTN_DIM // 2), device),
            _u8((HIDDEN * 448,), device),
            1.0,
            _f32((1,), device),
            adaln,
            index,
            x,
        ),
    }


SM120_ONLY_ENTRIES = [
    "fp8_pre_attention",
    "nvfp4_pre_attention",
    "fc1_swiglu_fp8",
    "fp8_out_proj",
    "nvfp4_out_proj",
]
# The public NVFP4 FC1 routes to the tcgen05 kernels below major 12, so only 12.x reaches this gate.
CASES = [(e, cc) for e in SM120_ONLY_ENTRIES for cc in [(12, 1), (10, 0), (9, 0)]] + [
    ("fc1_swiglu_nvfp4_public", (12, 1))
]


@requires_cuda
@pytest.mark.parametrize(
    "entry, mock_cc",
    CASES,
    ids=[f"{e}-{cc[0]}.{cc[1]}" for e, cc in CASES],
    indirect=["mock_cc"],
)
def test_sm120_quant_ops_reject_other_compute_capabilities(mock_cc, entry):
    call = _calls(torch.device("cuda:0"))[entry]
    with pytest.raises(
        NotImplementedError, match=rf"12\.0.*got {mock_cc[0]}\.{mock_cc[1]}"
    ):
        call()
