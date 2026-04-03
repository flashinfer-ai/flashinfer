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

"""Tests for the modular MLA decode kernel (flashinfer.cute_dsl.attention).

Mirrors the monolithic test_cute_dsl_mla_decode.py but imports from the modular
wrappers.batch_mla path to validate the refactored kernel.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer.utils import is_sm100a_supported, is_sm110a_supported
from flashinfer.cute_dsl import is_cute_dsl_available


def skip_if_unsupported():
    device = torch.device("cuda")
    if not (is_sm100a_supported(device) or is_sm110a_supported(device)):
        pytest.skip("Requires SM100-SM110 (tcgen05)")
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")


def torch_reference_mla(
    q_nope, q_rope, c_latent, c_rope, page_table, cache_seqs,
    softmax_scale, output_scale, page_size,
):
    B, q_len, H, latent_dim = q_nope.shape
    outputs = []
    for b in range(B):
        seq_len = cache_seqs[b].item()
        num_pages_needed = (seq_len + page_size - 1) // page_size
        page_indices = page_table[b, :num_pages_needed]
        kv_indices = []
        for p in page_indices:
            start = p.item() * page_size
            kv_indices.extend(range(start, start + page_size))
        kv_indices = kv_indices[:seq_len]
        kv_indices_t = torch.tensor(kv_indices, device=q_nope.device)

        k_latent = c_latent[kv_indices_t]
        k_rope = c_rope[kv_indices_t]
        q_lat_b = q_nope[b]
        q_rope_b = q_rope[b]

        attn_latent = torch.einsum("qhd,kd->qhk", q_lat_b.float(), k_latent.float())
        attn_rope = torch.einsum("qhd,kd->qhk", q_rope_b.float(), k_rope.float())
        attn = (attn_latent + attn_rope) * softmax_scale
        attn = F.softmax(attn, dim=-1)
        out_b = torch.einsum("qhk,kd->qhd", attn, k_latent.float())
        out_b = out_b * output_scale
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048, 8192])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("enable_pdl", [False, True])
def test_modular_mla_decode_fp16(
    batch_size, seq_len_k, page_size, dtype, q_len, enable_pdl
):
    """Test modular MLA decode kernel against PyTorch reference."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    D_qk = latent_dim + rope_dim
    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device)

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(
        total_pages, page_size, latent_dim + rope_dim, dtype=dtype, device=device,
    )

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out = cute_dsl_mla_decode(
        query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seq_len_k,
        softmax_scale=softmax_scale, output_scale=output_scale,
        is_var_seq=False, enable_pdl=enable_pdl,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope_q = query[..., latent_dim:]

    ref_out = torch_reference_mla(
        q_nope, q_rope_q, c_latent_ref, c_rope_ref,
        block_tables, seq_lens, softmax_scale, output_scale, page_size,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_modular_mla_decode_variable_seq_len(batch_size, seq_len_k, page_size, dtype):
    """Test modular MLA decode with variable sequence lengths."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    D_qk = latent_dim + rope_dim

    query = torch.randn(
        batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device
    )

    seq_lens = torch.randint(
        page_size, seq_len_k + 1, (batch_size,), dtype=torch.int32, device=device
    )
    max_seq_len = seq_lens.max().item()

    num_pages_per_batch = (max_seq_len + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(
        total_pages, page_size, D_qk, dtype=dtype, device=device
    )

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out = cute_dsl_mla_decode(
        query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=max_seq_len,
        softmax_scale=softmax_scale, output_scale=output_scale,
        is_var_seq=True,
    )

    kv_flat = kv_cache.reshape(-1, D_qk)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope_q = query[..., latent_dim:]

    ref_out = torch_reference_mla(
        q_nope, q_rope_q, c_latent_ref, c_rope_ref,
        block_tables, seq_lens, softmax_scale, output_scale, page_size,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048, 8192])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_modular_vs_monolithic(batch_size, seq_len_k, page_size, dtype):
    """Cross-validate modular MLA decode against the monolithic kernel."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        cute_dsl_mla_decode as modular,
    )
    from flashinfer.mla.cute_dsl import cute_dsl_mla_decode as monolithic

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    D_qk = latent_dim + rope_dim

    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device)

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(total_pages, page_size, D_qk, dtype=dtype, device=device)

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out_mono = monolithic(
        query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seq_len_k,
        softmax_scale=softmax_scale, is_var_seq=False,
    )
    out_mod = modular(
        query=query, kv_cache=kv_cache, workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim, qk_rope_head_dim=rope_dim,
        block_tables=block_tables, seq_lens=seq_lens, max_seq_len=seq_len_k,
        softmax_scale=softmax_scale, is_var_seq=False,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(out_mod, out_mono, atol=0, rtol=0)
