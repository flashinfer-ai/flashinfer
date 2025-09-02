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


# -------------------------  Configuration generation function  ----------------------------- #
def _build_seq_len_configs():
    """
    Reproduce the sequence length configurations from the original benchmark (including random cases).
    Returns: List[List[Tuple[int,int]]]  -> Each element is a list of (kv_len, qo_len) pairs.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    seq_len_configs = [
        [(8190, 7939)]
    ]

    # Construct random seqlen tests
    # bsz, stride, sparsity = 256, 16, 0.05
    # full_kv_len = np.random.randint(1000, 11000, size=bsz)
    # seq_len = []
    # for i in range(bsz):
    #     if i % stride == 0:
    #         kv_len, qo_len = full_kv_len[i], stride + 1
    #     else:
    #         kv_len, qo_len = int(full_kv_len[i] * sparsity), 1
    #     seq_len.append((kv_len, qo_len))
    # seq_len_configs.append(seq_len)

    return seq_len_configs


def _run_attention(
    kv_lens,
    qo_lens,
    page_block_size=1,
    num_kv_heads=1,
    num_qo_heads=1,
    head_dim=128,
    layout="NHD",
    test_dtype=torch.bfloat16,
    logits_soft_cap=0.0,
    device="cuda",
    causal=True,
):
    """
    Run both implementations and return (output_old, lse_old, output_new, lse_new)
    """
    dev = torch.device(device)
    seq_lens = torch.tensor(kv_lens, dtype=torch.int32)
    q_lens = torch.tensor(qo_lens, dtype=torch.int32)

    seq_lens_blocks = torch.ceil(seq_lens / page_block_size).int()

    q_indptr = torch.cat([torch.tensor([0]), torch.cumsum(q_lens, 0)], dim=0).int()
    kv_indptr = torch.cat(
        [torch.tensor([0]), torch.cumsum(seq_lens_blocks, 0)], dim=0
    ).int()

    num_blocks = kv_indptr[-1].item()

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
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=dev),
        kv_layout=layout,
        backend="fa2",
    )
    last_page_len = (seq_lens - 1) % page_block_size + 1
    wrapper_old.plan(
        q_indptr.to(dev),
        kv_indptr.to(dev),
        torch.arange(num_blocks, device=dev).int(),
        last_page_len.to(dev),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        logits_soft_cap=logits_soft_cap,
    )
    out_old, lse_old = wrapper_old.run(q, kv_data, return_lse=True)

    # Debug
    num_blocks_debug = kv_indptr[1].item()
    # wrapper_old.plan(q_indptr[:2],
    #                  kv_indptr[:2],
    #                  torch.arange(num_blocks_debug, device=dev).int(),
    #                  last_page_len[:1],
    #                  num_qo_heads,
    #                  num_kv_heads,
    #                  head_dim,
    #                  page_block_size,
    #                  causal=causal,
    #                  q_data_type=test_dtype,
    #                  kv_data_type=test_dtype,
    #                  logits_soft_cap=logits_soft_cap,
                    #  )
    # out_old_debug, lse_old_debug = wrapper_old.run(q[:q_indptr[1]], kv_data[:num_blocks_debug], return_lse=True)
    # k = kv_data[:num_blocks_debug][:, 0, ...].squeeze(2).transpose(0, 1)
    # v = kv_data[:num_blocks_debug][:, 1, ...].squeeze(2).transpose(0, 1)
    # out_single = flashinfer.single_prefill_with_kv_cache(q[:q_indptr[1]], kv_data[:num_blocks_debug][:, 0, ...].squeeze(2),  kv_data[:num_blocks_debug][:, 1, ...].squeeze(2), causal=causal)
    # out_torch = torch.nn.functional.scaled_dot_product_attention(q[:q_indptr[-1]].transpose(0, 1), k, v, is_causal=causal)
    # --------- new / mixed scheduler --------- #
    wrapper = flashinfer.BatchAttention(kv_layout=layout)
    wrapper.plan(
        q_indptr.to(dev),
        kv_indptr.to(dev),
        torch.arange(num_blocks, device=dev).int(),
        seq_lens.to(dev),
        num_qo_heads,
        num_kv_heads,
        head_dim,
        head_dim,
        page_block_size,
        causal=causal,
        q_data_type=test_dtype,
        kv_data_type=test_dtype,
        # logits_soft_cap=logits_soft_cap,
    )
    out_new, lse_new = wrapper.run(q, kv_data)
    # out_single = flashinfer.single_prefill_with_kv_cache(q[:q_indptr[1]], kv_data[:num_blocks_debug][:, 0, ...].squeeze(2),  kv_data[:num_blocks_debug][:, 1, ...].squeeze(2), causal=causal)
    torch.cuda.synchronize()
    torch.cuda.current_stream().synchronize()
    # NOTE(add this and the test passes...)
    torch.testing.assert_close(out_old, out_new, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse_old, lse_new, rtol=1e-2, atol=1e-2)
    


# -------------------------  PyTest test case  ----------------------------- #
@pytest.mark.parametrize("seq_len_pairs", _build_seq_len_configs())
@pytest.mark.parametrize("page_block_size", [1])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("gqa_group_size", [4])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("causal", [True])
@pytest.mark.parametrize("layout", ["HND"])
@pytest.mark.parametrize("test_dtype", [torch.bfloat16])
@pytest.mark.parametrize("logits_soft_cap", [0.0])
def test_batch_attention_correctness(
    seq_len_pairs,
    page_block_size,
    num_kv_heads,
    gqa_group_size,
    head_dim,
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
        causal=causal,
        layout=layout,
        test_dtype=test_dtype,
        logits_soft_cap=logits_soft_cap,
        device="cuda",
    )

test_batch_attention_correctness(_build_seq_len_configs()[0], page_block_size=1, num_kv_heads=1, gqa_group_size=1, head_dim=64, causal=True, layout="HND", test_dtype=torch.bfloat16, logits_soft_cap=0.0)