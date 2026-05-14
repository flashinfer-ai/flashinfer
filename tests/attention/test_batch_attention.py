"""
Copyright (c) 2025 by FlashInfer team.

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

import numpy as np
import pytest
import torch

import flashinfer
from tests.test_helpers.jit_utils import (
    gen_persistent_batch_attention_modules,
    gen_prefill_attention_modules,
)
from tests.test_helpers.utils_fp4 import create_nvfp4_kv, nvfp4_to_float
from flashinfer.utils import get_compute_capability, has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_persistent_batch_attention_modules(
            [torch.float16, torch.bfloat16],  # q_dtypes
            [torch.float16, torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [False, True],  # use_logits_soft_cap
        )
        + gen_prefill_attention_modules(
            [torch.float16, torch.bfloat16],  # q_dtypes
            [torch.float16, torch.bfloat16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False, True],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )


# -------------------------  Configuration generation function  ----------------------------- #
def _build_seq_len_configs():
    """
    Reproduce the sequence length configurations from the original benchmark (including random cases).
    Returns: List[List[Tuple[int,int]]]  -> Each element is a list of (kv_len, qo_len) pairs.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    seq_len_configs = [
        [(146, 146)],
        [(67, 67)],
        [(8190, 7939)],
        [(2048, 1)] * 77,  # decode-only
        [(4099, 129)] * 2,  # prefill-only
        [(600, 1)] * 132 * 2 + [(5000, 3)] * 128,
        [(1024, 1)] * 100 + [(8192, 17)] * 8,  # speculative decode
        [(766, 2)] * 99 + [(1024, 512)] * 1,  # chunked prefill
        [(2, 235)] + [(1, 13353)],  # real workload
    ]

    # Construct random seqlen tests
    bsz, stride, sparsity = 256, 16, 0.05
    full_kv_len = np.random.randint(1000, 11000, size=bsz)
    seq_len = []
    for i in range(bsz):
        if i % stride == 0:
            kv_len, qo_len = full_kv_len[i], stride + 1
        else:
            kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
        seq_len.append((kv_len, qo_len))
    seq_len_configs.append(seq_len)

    return seq_len_configs


def _run_attention(
    kv_lens,
    qo_lens,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    v_scale=None,
    layout="NHD",
    test_dtype=torch.bfloat16,
    logits_soft_cap=0.0,
    device="cuda",
    causal=True,
    is_chunked_q=False,
):
    """
    Run both implementations and return (output_old, lse_old, output_new, lse_new)
    """
    dev = torch.device(device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32, device=dev)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32, device=dev)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat(
        [torch.tensor([0], device=dev), torch.cumsum(q_lens, 0)], dim=0
    ).int()
    kv_indptr = torch.cat(
        [torch.tensor([0], device=dev), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

    if is_chunked_q:
        q_base = torch.rand(
            q_indptr[-1].item(),
            num_qo_heads,
            head_dim * 2,
            dtype=test_dtype,
            device=dev,
        )
        q = torch.chunk(q_base, 2, dim=-1)[0]
    else:
        q = torch.rand(
            q_indptr[-1].item(), num_qo_heads, head_dim, dtype=test_dtype, device=dev
        )
    if layout == "NHD":
        kv_data = torch.randn(
            num_blocks,
            2,
            page_block_size,
            num_kv_heads,
            head_dim,
            dtype=test_dtype,
            device=dev,
        )
    elif layout == "HND":
        kv_data = torch.randn(
            num_blocks,
            2,
            num_kv_heads,
            page_block_size,
            head_dim,
            dtype=test_dtype,
            device=dev,
        )

    # --------- old scheduler --------- #
    wrapper_old = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=dev),
        kv_layout=layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr,
        kv_indptr,
        torch.arange(num_blocks, device=dev).int(),
        last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_old, lse_old = wrapper_old.run(q, kv_data, return_lse=True, v_scale=v_scale)

    # --------- new / mixed scheduler --------- #
    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        torch.arange(num_blocks, device=dev).int(),
        seq_lens,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_new, lse_new = wrapper.run(
        q, kv_data, v_scale=v_scale, logits_soft_cap=logits_soft_cap
    )

    torch.cuda.synchronize()
    torch.testing.assert_close(out_old, out_new, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse_old, lse_new, rtol=1e-2, atol=1e-2)


# -------------------------  PyTest test case  ----------------------------- #
@pytest.mark.xfail(
    get_compute_capability(torch.device(device="cuda"))[0] == 12,
    reason="Expected failure for SM120/121 for now since the tile size/number of stages is too large.",
)
def test_batch_attention_with_noncontiguous_q():
    # Pick the first sequence length config's first pair
    seq_len_pairs = _build_seq_len_configs()[0]
    kv_lens = [p[0] for p in seq_len_pairs]
    qo_lens = [p[1] for p in seq_len_pairs]

    # Fixed single-case parameters
    page_block_size = 1
    num_kv_heads = 1
    gqa_group_size = 1
    num_qo_heads = num_kv_heads * gqa_group_size
    head_dim = 64
    test_dtype = torch.bfloat16
    layout = "NHD"
    logits_soft_cap = 0.0
    v_scale = None
    causal = True

    _run_attention(
        kv_lens=kv_lens,
        qo_lens=qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        v_scale=v_scale,
        causal=causal,
        layout=layout,
        test_dtype=test_dtype,
        logits_soft_cap=logits_soft_cap,
        device="cuda",
        is_chunked_q=True,
    )


@pytest.mark.xfail(
    get_compute_capability(torch.device(device="cuda"))[0] == 12,
    reason="Expected failure for SM120/121 for now since the tile size/number of stages is too large.",
)
@pytest.mark.parametrize("seq_len_pairs", _build_seq_len_configs())
@pytest.mark.parametrize("page_block_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [1, 4])
@pytest.mark.parametrize("gqa_group_size", [1, 4, 7, 8])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("v_scale", [2.0, None])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("logits_soft_cap", [0.0, 50.0])
def test_batch_attention_correctness(
    seq_len_pairs,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
    v_scale,
    causal,
    layout,
    test_dtype,
    logits_soft_cap,
):
    num_qo_heads = num_kv_heads * gqa_group_size
    kv_lens = [p[0] for p in seq_len_pairs]
    qo_lens = [p[1] for p in seq_len_pairs]

    _run_attention(
        kv_lens=kv_lens,
        qo_lens=qo_lens,
        page_block_size=page_block_size,
        num_kv_heads=num_kv_heads,
        num_qo_heads=num_qo_heads,
        head_dim=head_dim,
        v_scale=v_scale,
        causal=causal,
        layout=layout,
        test_dtype=test_dtype,
        logits_soft_cap=logits_soft_cap,
        device="cuda",
    )


@pytest.mark.xfail(
    get_compute_capability(torch.device(device="cuda"))[0] == 12,
    reason="Expected failure for SM120/121 for now since the tile size/number of stages is too large.",
)
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("kv_len", [128, 256])
@pytest.mark.parametrize("qo_len", [64, 128])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_kv_heads", [1])
@pytest.mark.parametrize("num_qo_heads", [1])
@pytest.mark.parametrize("head_dim", [128])
@pytest.mark.parametrize("causal", [False])
@pytest.mark.parametrize("q_dtype", [torch.float16, torch.bfloat16])
def test_batch_attention_nvfp4(
    batch_size,
    kv_len,
    qo_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    causal,
    q_dtype,
):
    """Test BatchAttention with NVFP4 KV cache.

    KV cache layout (NHD):
      kv_cache:    [num_pages, 2, page_size, num_kv_heads, head_dim//2]   uint8 (packed FP4x2)
      kv_cache_sf: [num_pages, 2, page_size, num_kv_heads, head_dim//16]  uint8 (FP8 SFs)

    Reference is computed by dequantizing the packed KV back to q_dtype and running
    single_prefill_with_kv_cache per batch item.
    """
    if qo_len > kv_len and causal:
        pytest.skip("qo_len > kv_len and causal is not supported")

    kv_layout = "NHD"
    torch.manual_seed(42)

    # --- query ---
    q = torch.randn(
        batch_size * qo_len, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype
    )
    q_indptr_cpu = torch.arange(0, batch_size + 1, dtype=torch.int32) * qo_len

    # --- paged KV metadata ---
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_indptr_cpu = (
        torch.arange(0, batch_size + 1, dtype=torch.int32) * num_pages_per_seq
    )
    kv_indices_cpu = torch.arange(0, total_num_pages, dtype=torch.int32)
    kv_last_page_len_cpu = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32
    )
    kv_len_arr_cpu = torch.full((batch_size,), kv_len, dtype=torch.int32)

    # --- create NVFP4 KV pages directly (NHD: [num_pages, page_size, num_kv_heads, head_dim//2]) ---
    kv_shape = (total_num_pages, page_size, num_kv_heads, head_dim // 2)
    k_packed, k_sf, k_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")
    v_packed, v_sf, v_global_scale = create_nvfp4_kv(kv_shape, "cuda:0")

    # Dequantize for reference attention
    k_dq = nvfp4_to_float(k_packed, k_sf, k_global_scale).to(q_dtype)
    v_dq = nvfp4_to_float(v_packed, v_sf, v_global_scale).to(q_dtype)

    # Pack into combined tensors:
    #   kv_cache:    [num_pages, 2, page_size, num_kv_heads, head_dim//2]
    #   kv_cache_sf: [num_pages, 2, page_size, num_kv_heads, head_dim//16]
    kv_cache = torch.stack([k_packed, v_packed], dim=1)
    kv_cache_sf = torch.stack([k_sf, v_sf], dim=1)

    # --- run BatchAttention ---
    q_indptr_gpu = q_indptr_cpu.to("cuda:0")
    kv_indptr_gpu = kv_indptr_cpu.to("cuda:0")
    kv_indices_gpu = kv_indices_cpu.to("cuda:0")
    kv_len_arr_gpu = kv_len_arr_cpu.to("cuda:0")

    wrapper = flashinfer.BatchAttention(kv_layout=kv_layout)
    wrapper.plan(
        q_indptr_gpu,
        kv_indptr_gpu,
        kv_indices_gpu,
        kv_len_arr_gpu,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_size,
        causal=causal,
        q_data_type=q_dtype,
        kv_data_type=torch.uint8,
    )
    o, _ = wrapper.run(
        q,
        kv_cache,
        k_scale=k_global_scale.item(),
        v_scale=v_global_scale.item(),
        kv_cache_sf=kv_cache_sf,
    )

    # --- reference: single_prefill_with_kv_cache per batch item using dequantized KV ---
    for i in range(batch_size):
        qi = q[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        full_pages_k = k_dq[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
        last_page_k = k_dq[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i]]
        ki = torch.cat(
            [
                full_pages_k.reshape(-1, num_kv_heads, head_dim),
                last_page_k.reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )

        full_pages_v = v_dq[kv_indptr_cpu[i] : kv_indptr_cpu[i + 1] - 1]
        last_page_v = v_dq[kv_indptr_cpu[i + 1] - 1, : kv_last_page_len_cpu[i]]
        vi = torch.cat(
            [
                full_pages_v.reshape(-1, num_kv_heads, head_dim),
                last_page_v.reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        )

        o_ref_i = flashinfer.prefill.single_prefill_with_kv_cache(
            qi, ki, vi, causal=causal, pos_encoding_mode="NONE", logits_soft_cap=0.0
        )
        o_i = o[q_indptr_cpu[i] : q_indptr_cpu[i + 1]]

        torch.testing.assert_close(o_i, o_ref_i, rtol=1e-1, atol=1e-1)


if __name__ == "__main__":
    test_batch_attention_correctness(
        seq_len_pairs=[(1000, 1000)],
        page_block_size=1,
        num_kv_heads=4,
        gqa_group_size=7,
        head_dim=128,
        v_scale=2.0,
        causal=True,
        layout="NHD",
        test_dtype=torch.bfloat16,
        logits_soft_cap=0.0,
    )
    test_batch_attention_nvfp4(
        batch_size=4,
        kv_len=128,
        qo_len=64,
        page_size=16,
        num_kv_heads=1,
        num_qo_heads=1,
        head_dim=128,
        causal=False,
        q_dtype=torch.float16,
    )
