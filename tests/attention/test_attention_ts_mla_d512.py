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

"""D512/D512 native geometry before attaching sparse routes to the kernels."""

import pytest
import torch
import math

pytest.importorskip("cutlass", minversion="4.7.0")

from flashinfer.attention.prims_ts.mla_decode import (
    _MLADecodeLaunchSpec,
    _MLARuntime,
    _MLAWorkspaceViews,
    _get_compiled_mla_decode,
    _launch_mla_decode,
    _make_mla_decode_compile_spec,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_2cta.kernel import (
    MlaDecodeTs,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.config import (
    MlaProfile,
)
from flashinfer.attention.prims_ts.kernels.mla_decode.throughput_latency_1cta.kernel import (
    ThroughputLatencyMlaDecodeTs,
)


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() not in ((10, 0), (10, 3)),
    reason="Requires SM100/SM103",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn])
@pytest.mark.parametrize("family,tile", [("swap", 16), ("keep", 64), ("2cta", 128)])
def test_native_d512(dtype, family, tile):
    torch.manual_seed(17)
    batch, heads, length, page = 2, 64, 256, 32
    name = "bf16" if dtype == torch.bfloat16 else "e4m3"
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    q = (torch.randn(batch, 1, heads, 512, device="cuda") * 0.2).to(dtype)
    kv = (torch.randn(batch * length // page, page, 512, device="cuda") * 0.2).to(dtype)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    lse = torch.empty(batch, 1, heads, device="cuda", dtype=torch.float32)
    table = torch.arange(batch * length // page, device="cuda", dtype=torch.int32).view(
        batch, -1
    )
    lengths = torch.full((batch,), length, device="cuda", dtype=torch.int32)
    if family == "2cta":
        kernel = MlaDecodeTs(
            page_size=page,
            max_active_clusters=sm_count // 2,
            is_persistent=False,
            is_var_seq=False,
            is_var_split_kv=False,
            static_split_kv=1,
            qkv_dtype=name,
            out_dtype="bf16",
            rope_dim=0,
            num_heads=heads,
            seq_len_q=1,
            batch_size=batch,
            mask_type="dense",
        )
    else:
        profile = MlaProfile(
            name="d512_test",
            kernel_variant="swaps_mma_ab" if family == "swap" else "keeps_mma_ab",
            tile_size_q=tile,
            use_persistent_scheduler=0,
        )
        kernel = ThroughputLatencyMlaDecodeTs(
            batch_size=batch,
            num_heads=tile,
            seq_len_q=(heads + tile - 1) // tile,
            seq_len_k=length,
            rope_dim=0,
            page_size=page,
            max_active_clusters=sm_count,
            qkv_dtype=name,
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
        page_size=page,
        q_dtype_key=str(dtype).removeprefix("torch."),
        output_dtype_key="bfloat16",
        max_seq_len_q=1,
        packed_query=False,
    )
    compiled = _get_compiled_mla_decode(spec)
    _launch_mla_decode(
        _MLARuntime(q, kv, out, kv.shape[0], 512**-0.5, 1.0),
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
    q_ref = q.double().squeeze(1)
    kv_ref = kv.double().view(batch, length, 512)
    scores = (q_ref @ kv_ref.transpose(-1, -2)) * 512**-0.5
    expected = (scores.softmax(-1) @ kv_ref).unsqueeze(1)
    rtol, atol = (0.01, 5e-4) if dtype == torch.bfloat16 else (0.05, 1.5e-3)
    torch.testing.assert_close(out.double(), expected, rtol=rtol, atol=atol)
    torch.testing.assert_close(
        lse.double() * math.log(2),
        scores.logsumexp(-1).unsqueeze(1),
        rtol=1e-4,
        atol=1e-4,
    )
