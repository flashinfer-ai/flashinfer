# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Independent global scale operands for dense and block-sparse attention."""

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts import (
    BlockSparsePagedTSWrapper,
    BlockSparseTSWrapper,
    SageAttentionConfig,
    block_sparse_attention,
    block_sparse_attention_with_paged_kv_cache,
)
from tests.attention.test_attention_ts_block_sparse import _REQUIRES_PRIMTS_GPU
from tests.attention.test_attention_ts_sage import _SageCase, _random_inputs

pytestmark = [_REQUIRES_PRIMTS_GPU, pytest.mark.arch_blackwell]


def _operand(value, tensor):
    return (
        torch.tensor([value], device="cuda", dtype=torch.float32) if tensor else value
    )


def _case(kind, sm_scale, output_scale):
    torch.manual_seed(911)
    q = torch.randn((1, 16, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 128, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    indptr = torch.tensor([[[0, 1, 2]]], device="cuda", dtype=torch.int32)
    indices = torch.tensor([0, 1], device="cuda", dtype=torch.int32)
    plan_args = dict(
        device=q.device,
        q_data_type=q.dtype,
        sm_scale=sm_scale,
        output_scale=output_scale,
    )
    if kind == "paged":
        wrapper = BlockSparsePagedTSWrapper()
        wrapper.plan(
            1,
            16,
            128,
            1,
            1,
            128,
            8,
            64,
            64,
            max_blocks_per_row=1,
            use_kv_valid_bits=False,
            **plan_args,
        )
        # Allocate canonical HND strides, including the singleton head axis.
        k_cache = torch.empty((2, 1, 64, 128), device=k.device, dtype=k.dtype)
        v_cache = torch.empty_like(k_cache)
        k_cache.copy_(k.view(2, 64, 1, 128).transpose(1, 2))
        v_cache.copy_(v.view(2, 64, 1, 128).transpose(1, 2))
        tables = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
        lengths = torch.tensor([128], device="cuda", dtype=torch.int32)
        run_args = (q, (k_cache, v_cache), tables, lengths, indptr, indices)
    else:
        wrapper = BlockSparseTSWrapper()
        route_args = {"max_blocks_per_row": 1} if kind == "sparse" else {}
        wrapper.plan(
            1,
            16,
            128,
            1,
            1,
            128,
            8,
            64,
            use_block_sparse=kind == "sparse",
            **route_args,
            **plan_args,
        )
        run_args = (q, k, v, indptr, indices) if kind == "sparse" else (q, k, v)
    return wrapper, run_args, (q, k, v)


def _reference(kind, inputs, sm_scale, output_scale):
    q, k, v = (x[0, :, 0].float() for x in inputs)
    logits = q @ k.T * sm_scale
    if kind != "dense":
        allowed = torch.arange(16, device=q.device)[:, None] // 8 == (
            torch.arange(128, device=q.device)[None, :] // 64
        )
        logits.masked_fill_(~allowed, -torch.inf)
    return (logits.softmax(-1) @ v * output_scale)[None, :, None]


@pytest.mark.parametrize("kind", ("dense", "sparse", "paged"))
@pytest.mark.parametrize("sm_tensor", (False, True))
@pytest.mark.parametrize("output_tensor", (False, True))
@torch.no_grad()
def test_block_sparse_independent_global_scales(kind, sm_tensor, output_tensor):
    sm, output = _operand(0.0625, sm_tensor), _operand(1.75, output_tensor)
    wrapper, args, inputs = _case(kind, sm, output)
    expected = _reference(kind, inputs, 0.0625, 1.75)
    actual = wrapper.run(*args)
    torch.testing.assert_close(actual.float(), expected, atol=5e-3, rtol=5e-3)
    assert torch.equal(actual, wrapper.run(*args, validate=False))

    # Both one-shot APIs must specialize before their first run, too.
    if kind == "paged":
        one_shot = block_sparse_attention_with_paged_kv_cache(
            *args,
            q_block_size=8,
            kv_block_size=64,
            max_seq_len_kv=128,
            sm_scale=sm,
            output_scale=output,
        )
    else:
        one_shot_args = (*args, None, None) if kind == "dense" else args
        one_shot = block_sparse_attention(
            *one_shot_args,
            q_block_size=8,
            kv_block_size=64,
            use_block_sparse=kind == "sparse",
            sm_scale=sm,
            output_scale=output,
        )
    torch.testing.assert_close(one_shot.float(), expected, atol=5e-3, rtol=5e-3)

    # A trusted run must never feed a tensor to a compiled scalar signature.
    opposite_sm = 0.0625 if sm_tensor else _operand(0.0625, True)
    with pytest.raises(TypeError, match="same scalar/tensor type"):
        wrapper.run(*args, sm_scale=opposite_sm, validate=False)
    opposite_output = 1.75 if output_tensor else _operand(1.75, True)
    with pytest.raises(TypeError, match="same scalar/tensor type"):
        wrapper.run(*args, output_scale=opposite_output, validate=False)


@pytest.mark.parametrize("kind", ("dense", "sparse", "paged"))
@torch.no_grad()
def test_block_sparse_global_scales_cuda_graph(kind):
    sm, output = _operand(0.0625, True), _operand(1.75, True)
    wrapper, args, inputs = _case(kind, sm, output)
    out = wrapper.run(*args)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(*args, out=out)
    for sm_value, output_value in ((0.125, 0.75), (0.03125, 2.0)):
        sm.fill_(sm_value)
        output.fill_(output_value)
        graph.replay()
        torch.testing.assert_close(
            out.float(),
            _reference(kind, inputs, sm_value, output_value),
            atol=5e-3,
            rtol=5e-3,
        )


@pytest.mark.parametrize("sm_tensor", (False, True))
@pytest.mark.parametrize("output_tensor", (False, True))
@torch.no_grad()
def test_sage_global_output_scale_includes_restored_mean(sm_tensor, output_tensor):
    case = _SageCase(
        "global_mean",
        batch_size=1,
        seq_len_q=64,
        seq_len_kv=256,
        num_qo_heads=1,
        num_kv_heads=1,
        q_block_size=64,
        kv_block_size=64,
        kv_tile=256,
        sage=SageAttentionConfig(v_mean=True),
    )
    q, k, v, params, _ = _random_inputs(case)
    # All dequantized V values are the channel mean, making the exact result
    # independent of FP8 probability rounding and Q/K quantization.
    v = torch.zeros(v.shape, device=v.device).to(v.dtype)
    output_scale = _operand(1.75, output_tensor)
    result = block_sparse_attention(
        q,
        k,
        v,
        None,
        None,
        use_block_sparse=False,
        q_block_size=64,
        kv_block_size=64,
        sm_scale=_operand(0.0625, sm_tensor),
        output_scale=output_scale,
        sage=params,
        sage_config=case.sage,
    )
    expected = (params.v_mean[:, None] * 1.75).expand_as(result).to(result.dtype)
    torch.testing.assert_close(result, expected, atol=2e-3, rtol=8e-3)
