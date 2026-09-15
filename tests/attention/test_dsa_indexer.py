"""
Copyright (c) 2024 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import pytest
import torch

deep_gemm = pytest.importorskip("deep_gemm")

import flashinfer


def _sm100_only():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    major, _ = torch.cuda.get_device_capability()
    if major != 10:
        pytest.skip("DSA indexer kernel is SM100 (Blackwell) only")


def _rand_fp8(shape, device):
    return (torch.randn(shape, device=device) * 0.25).to(torch.float8_e4m3fn)


@pytest.mark.parametrize("num_q", [64, 512])
@pytest.mark.parametrize("seq_len_kv", [16384, 65536])
@pytest.mark.parametrize("top_k", [512, 2048])
def test_dsa_indexer_topk_recall(num_q, seq_len_kv, top_k):
    _sm100_only()
    torch.manual_seed(0)
    dev = "cuda"
    H, D = 32, 128
    sample_len = 8192

    q = _rand_fp8((num_q, H, D), dev)
    kv = _rand_fp8((seq_len_kv, D), dev)
    kv_scales = torch.rand(seq_len_kv, device=dev, dtype=torch.float32) + 0.5
    weights = (torch.rand(num_q, H, device=dev, dtype=torch.float32)) * (D ** -0.5)

    cu_start = torch.zeros(num_q, dtype=torch.int32, device=dev)
    cu_end = torch.full((num_q,), seq_len_kv, dtype=torch.int32, device=dev)
    cu_start_s = torch.zeros(num_q, dtype=torch.int32, device=dev)
    cu_end_s = torch.full((num_q,), sample_len, dtype=torch.int32, device=dev)

    ref_logits = deep_gemm.fp8_fp4_mqa_logits(
        (q, None), (kv, kv_scales), weights, cu_start, cu_end, clean_logits=True
    )
    ref = ref_logits.topk(top_k, dim=1).indices.long().sort(dim=1).values

    sample_logits = deep_gemm.fp8_fp4_mqa_logits(
        (q, None), (kv[:sample_len], kv_scales[:sample_len]), weights,
        cu_start_s, cu_end_s, clean_logits=False,
    ).contiguous()

    idx = flashinfer.dsa_indexer_topk(
        sample_logits, q, kv, kv_scales, weights, cu_end, top_k,
        cand_cap=max(4 * top_k, 16384),
    )
    got = idx.long().sort(dim=1).values

    p = torch.searchsorted(ref, got).clamp(max=top_k - 1)
    recall = (torch.gather(ref, 1, p) == got).float().mean().item()
    assert recall > 0.99, f"recall {recall:.4f} too low (num_q={num_q}, S={seq_len_kv}, k={top_k})"
