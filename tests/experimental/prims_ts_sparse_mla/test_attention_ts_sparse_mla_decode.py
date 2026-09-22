# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sparse task-graph integration, forced profiles rather than auto-only tests."""

import math

import pytest
import torch

pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts.mla_decode import (
    _MLADecodeLaunchSpec,
    _MLARuntime,
    _MLAWorkspaceViews,
    _get_compiled_mla_decode,
    _launch_mla_decode,
    _make_mla_decode_compile_spec,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    MlaProfile,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
    ThroughputLatencyMlaDecodeTs,
)
from flashinfer.testing.sparse_mla import sparse_mla_reference
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
    MlaDecodeTs,
)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("family,tile", [("swap", 16), ("keep", 64), ("2cta", 128)])
def test_two_source_native(dtype, family, tile):
    torch.manual_seed(41)
    batch, heads = 2, 64
    q = (torch.randn(batch, 1, heads, 512, device="cuda") * 0.2).to(dtype)
    swa = (torch.randn(193, 1, 512, device="cuda") * 0.2 - 0.1).to(dtype)
    compressed = (torch.randn(257, 1, 512, device="cuda") * 0.2 + 0.1).to(dtype)
    si = torch.randint(0, 193, (batch, 1, 128), device="cuda", dtype=torch.int32)
    ci = torch.randint(0, 257, (batch, 1, 257), device="cuda", dtype=torch.int32)
    si[..., ::11] = -1
    ci[..., ::7] = -1
    sl = torch.tensor([[73], [128]], device="cuda", dtype=torch.int32)
    cl = torch.tensor([[131], [257]], device="cuda", dtype=torch.int32)
    table = torch.full((batch, 512), 0x7FFFFFFF, device="cuda", dtype=torch.int32)
    for b in range(batch):
        table[b, : int(sl[b])] = torch.where(
            si[b, 0, : int(sl[b])] >= 0, si[b, 0, : int(sl[b])], 0x7FFFFFFF
        )
        table[b, 128:] = -1
        table[b, 128 : 128 + int(cl[b])] = ci[b, 0, : int(cl[b])] | -2147483648
    lengths = torch.tensor([384, 512], device="cuda", dtype=torch.int32)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(batch, 1, heads, device="cuda", dtype=torch.float32)
    profile = MlaProfile(
        name="sparse_test",
        kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
        tile_size_q=tile,
        use_persistent_scheduler=0,
    )
    if family == "2cta":
        kernel = MlaDecodeTs(
            page_size=1,
            max_active_clusters=torch.cuda.get_device_properties(
                0
            ).multi_processor_count
            // 2,
            is_persistent=False,
            is_var_seq=False,
            is_var_split_kv=False,
            static_split_kv=1,
            qkv_dtype="bf16" if dtype == torch.bfloat16 else "e4m3",
            out_dtype="bf16",
            rope_dim=0,
            num_heads=heads,
            seq_len_q=1,
            batch_size=batch,
            mask_type="dense",
        )
    else:
        kernel = ThroughputLatencyMlaDecodeTs(
            batch_size=batch,
            num_heads=tile,
            seq_len_q=heads // tile,
            seq_len_k=512,
            rope_dim=0,
            page_size=1,
            max_active_clusters=torch.cuda.get_device_properties(
                0
            ).multi_processor_count,
            qkv_dtype="bf16" if dtype == torch.bfloat16 else "e4m3",
            out_dtype="bf16",
            profile=profile,
            logical_num_heads=heads,
            logical_seq_len_q=1,
            tile_size_q=tile,
            mask_type="dense",
        )
    spec = _make_mla_decode_compile_spec(
        _MLADecodeLaunchSpec(kernel, (), 0, 1),
        device_index=0,
        num_heads=heads,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
        page_size=1,
        q_dtype_key=str(dtype).removeprefix("torch."),
        output_dtype_key="bfloat16",
        max_seq_len_q=1,
        packed_query=False,
    )
    compiled = _get_compiled_mla_decode(spec)
    _launch_mla_decode(
        _MLARuntime(q, swa, out, swa.shape[0], 512**-0.5, 1.0, extra_cache=compressed),
        block_tables=table,
        seq_lens=lengths,
        qo_indptr=None,
        packed_query=False,
        kv_lora_rank=512,
        split_kv=1,
        workspace=_MLAWorkspaceViews(None, lse),
        compiled=compiled,
    )
    torch.cuda.synchronize()
    first = out.clone()
    first_lse = lse.clone()
    for _repeat in range(10):
        _launch_mla_decode(
            _MLARuntime(
                q, swa, out, swa.shape[0], 512**-0.5, 1.0, extra_cache=compressed
            ),
            block_tables=table,
            seq_lens=lengths,
            qo_indptr=None,
            packed_query=False,
            kv_lora_rank=512,
            split_kv=1,
            workspace=_MLAWorkspaceViews(None, lse),
            compiled=compiled,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(out, first, rtol=0, atol=0)
        torch.testing.assert_close(lse, first_lse, rtol=0, atol=0)
    torch.cuda.synchronize()
    expected, expected_lse = sparse_mla_reference(
        q,
        swa,
        compressed,
        si,
        ci,
        swa_topk_lens=sl,
        compressed_topk_lens=cl,
    )
    rtol, atol = (0.01, 5e-4) if dtype == torch.bfloat16 else (0.05, 1.5e-3)
    torch.testing.assert_close(out.double(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        lse.double() * math.log(2), expected_lse, rtol=1e-4, atol=1e-4
    )
