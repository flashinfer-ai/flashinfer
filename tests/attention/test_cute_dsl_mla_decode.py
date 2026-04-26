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

"""Tests for CuTe DSL MLA decode kernel."""

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
    q_nope,
    q_rope,
    c_latent,
    c_rope,
    page_table,
    cache_seqs,
    softmax_scale,
    output_scale,
    page_size,
):
    """PyTorch reference implementation for MLA decode.

    Args:
        q_nope: [B, q_len, H, latent_dim]
        q_rope: [B, q_len, H, rope_dim]
        c_latent: [num_pages * page_size, latent_dim]
        c_rope: [num_pages * page_size, rope_dim]
        page_table: [B, max_pages]
        cache_seqs: [B] — actual sequence lengths
        softmax_scale: float
        output_scale: float
        page_size: int
    """
    B, q_len, H, latent_dim = q_nope.shape

    outputs = []
    for b in range(B):
        seq_len = cache_seqs[b].item()
        num_pages_needed = (seq_len + page_size - 1) // page_size

        # Gather KV for this batch via page table
        page_indices = page_table[b, :num_pages_needed]
        kv_indices = []
        for p in page_indices:
            start = p.item() * page_size
            kv_indices.extend(range(start, start + page_size))
        kv_indices = kv_indices[:seq_len]
        kv_indices_t = torch.tensor(kv_indices, device=q_nope.device)

        k_latent = c_latent[kv_indices_t]  # [seq_len, latent_dim]
        k_rope = c_rope[kv_indices_t]  # [seq_len, rope_dim]

        # q: [q_len, H, D], k: [seq_len, D]
        q_lat_b = q_nope[b]  # [q_len, H, latent_dim]
        q_rope_b = q_rope[b]  # [q_len, H, rope_dim]

        # Compute attention scores
        # QK^T = q_latent @ k_latent^T + q_rope @ k_rope^T
        # [q_len, H, latent_dim] @ [latent_dim, seq_len] -> [q_len, H, seq_len]
        attn_latent = torch.einsum("qhd,kd->qhk", q_lat_b.float(), k_latent.float())
        attn_rope = torch.einsum("qhd,kd->qhk", q_rope_b.float(), k_rope.float())
        attn = (attn_latent + attn_rope) * softmax_scale

        # Softmax
        attn = F.softmax(attn, dim=-1)

        # Output: attn @ V (V = k_latent for MLA)
        # [q_len, H, seq_len] @ [seq_len, latent_dim] -> [q_len, H, latent_dim]
        out_b = torch.einsum("qhk,kd->qhd", attn, k_latent.float())
        out_b = out_b * output_scale
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)  # [B, q_len, H, latent_dim]


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048, 8192])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_cute_dsl_mla_decode_fp16(
    batch_size, seq_len_k, page_size, dtype, q_len, enable_pdl
):
    """Test FP16/BF16 MLA decode kernel."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    # Allocate query: [B, q_len, H, D_qk]
    D_qk = latent_dim + rope_dim
    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device)

    # Allocate paged KV cache
    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10  # extra pages
    kv_cache = torch.randn(
        total_pages,
        page_size,
        latent_dim + rope_dim,
        dtype=dtype,
        device=device,
    )

    # Page table: [B, max_pages] — sequential assignment
    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    # Sequence lengths
    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)

    # Workspace
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    # Run kernel
    out = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        is_var_seq=False,
        enable_pdl=enable_pdl,
    )

    # Reference
    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    ref_out = torch_reference_mla(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
    )

    ref_out_cast = ref_out.to(dtype)

    # Check with tolerance appropriate for FP16/BF16
    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_dsl_mla_decode_variable_seq_len(batch_size, seq_len_k, page_size, dtype):
    """Test MLA decode with variable sequence lengths across the batch."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    D_qk = latent_dim + rope_dim

    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device)

    max_seq_len = seq_len_k
    seq_lens = torch.randint(
        page_size, max_seq_len + 1, (batch_size,), dtype=torch.int32, device=device
    )

    max_pages_per_batch = (max_seq_len + page_size - 1) // page_size
    total_pages = max_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(total_pages, page_size, D_qk, dtype=dtype, device=device)

    block_tables = torch.zeros(
        batch_size, max_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(max_pages_per_batch):
            block_tables[b, p] = b * max_pages_per_batch + p

    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=max_seq_len,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        is_var_seq=True,
    )

    kv_flat = kv_cache.reshape(-1, D_qk)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    ref_out = torch_reference_mla(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
def test_cute_dsl_mla_decode_via_api(
    batch_size, seq_len_k, page_size=128, enable_pdl=False
):
    """Test MLA decode via the trtllm_batch_decode_with_kv_cache_mla API with cute-dsl backend."""
    skip_if_unsupported()

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    D_qk = latent_dim + rope_dim

    query = torch.randn(
        batch_size, q_len, num_heads, D_qk, dtype=torch.float16, device=device
    )

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(
        total_pages, page_size, D_qk, dtype=torch.float16, device=device
    )

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=latent_dim,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
        backend="cute-dsl",
        is_var_seq=False,
        enable_pdl=enable_pdl,
    )

    assert out.shape == (batch_size, q_len, num_heads, latent_dim)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_cute_dsl_vs_trtllm_gen(batch_size, seq_len_k, enable_pdl, page_size=64):
    """Test cute-dsl backend output matches trtllm-gen backend output."""
    skip_if_unsupported()

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    D_qk = latent_dim + rope_dim

    query = torch.randn(
        batch_size, q_len, num_heads, D_qk, dtype=torch.bfloat16, device=device
    )

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    # trtllm-gen expects 4D kv_cache: [num_pages, 1, page_size, D]
    kv_cache = torch.randn(
        total_pages, 1, page_size, D_qk, dtype=torch.bfloat16, device=device
    )

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.zeros(256 * 1024 * 1024, dtype=torch.int8, device=device)

    common_args = dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=latent_dim,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        bmm1_scale=softmax_scale,
        bmm2_scale=1.0,
    )

    out_trtllm = trtllm_batch_decode_with_kv_cache_mla(
        **common_args, backend="trtllm-gen", is_var_seq=False
    )
    out_cute_dsl = trtllm_batch_decode_with_kv_cache_mla(
        **common_args, backend="cute-dsl", is_var_seq=False
    )

    torch.testing.assert_close(
        out_cute_dsl.to(torch.float32),
        out_trtllm.to(torch.float32),
        atol=1e-2,
        rtol=1e-2,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048])
@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("enable_pdl", [False])
def test_cute_dsl_mla_decode_fp8(batch_size, seq_len_k, page_size, enable_pdl):
    """Test FP8 MLA decode kernel against FP32 reference."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    q_len = 1
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    D_qk = latent_dim + rope_dim

    # Create FP8 query and KV cache (cast from small-valued FP16 to stay in FP8 range)
    query = (
        torch.randn(
            batch_size, q_len, num_heads, D_qk, dtype=torch.float16, device=device
        )
        * 0.1
    ).to(torch.float8_e4m3fn)

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = (
        torch.randn(total_pages, page_size, D_qk, dtype=torch.float16, device=device)
        * 0.1
    ).to(torch.float8_e4m3fn)

    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    out = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
        enable_pdl=enable_pdl,
    )

    assert out.dtype == torch.bfloat16
    assert out.shape == (batch_size, q_len, num_heads, latent_dim)

    # Reference: compute in FP32 using FP8 values dequantized to FP32
    kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim].to(torch.float32)
    q_rope_tensor = query[..., latent_dim:].to(torch.float32)

    ref_out = torch_reference_mla(
        q_nope,
        q_rope_tensor,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
    )
    # Compare outputs in FP32; FP8 has limited precision so use wider tolerance
    torch.testing.assert_close(
        out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
    )


# ---------------------------------------------------------------------------
#  Variant tests: score_mod, update_statistics, transform_output
# ---------------------------------------------------------------------------


def torch_reference_mla_with_variant(
    q_nope,
    q_rope,
    c_latent,
    c_rope,
    page_table,
    cache_seqs,
    softmax_scale,
    output_scale,
    page_size,
    score_mod_fn=None,
    sink=None,
):
    """PyTorch reference for MLA decode with variant hooks.

    Args:
        score_mod_fn: callable(score, batch_idx, qo_idx, kv_idx, head_idx) -> score
        sink: (num_heads,) tensor for attention sink
    """
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
        attn = attn_latent + attn_rope

        if score_mod_fn is not None:
            for qi in range(q_len):
                for hi in range(H):
                    for ki in range(seq_len):
                        attn[qi, hi, ki] = score_mod_fn(attn[qi, hi, ki], b, qi, ki, hi)

        attn = attn * softmax_scale

        if sink is not None:
            sink_dev = sink.to(q_nope.device).float()
            for qi in range(q_len):
                for hi in range(H):
                    scores = attn[qi, hi, :]
                    # sink[hi] is in natural-log domain: effective weight = exp(sink[hi]).
                    # scores are already multiplied by softmax_scale, so place
                    # sink[hi] directly as the virtual score (torch.softmax
                    # computes exp(x_i) / sum(exp(x_j))).
                    virtual_scores = torch.cat([sink_dev[hi].unsqueeze(0), scores])
                    weights = torch.softmax(virtual_scores, dim=-1)
                    real_weights = weights[1:]
                    out_qh = torch.einsum("k,kd->d", real_weights, k_latent.float())
                    out_qh = out_qh * output_scale
                    if qi == 0 and hi == 0:
                        out_b = torch.zeros(q_len, H, latent_dim, device=q_nope.device)
                    out_b[qi, hi] = out_qh
            outputs.append(out_b)
            continue

        attn = F.softmax(attn, dim=-1)
        out_b = torch.einsum("qhk,kd->qhd", attn, k_latent.float())
        out_b = out_b * output_scale
        outputs.append(out_b)

    return torch.stack(outputs, dim=0)


def _make_mla_test_data(batch_size, seq_len_k, page_size, dtype, q_len=1):
    """Create standard MLA test data (query, kv_cache, block_tables, seq_lens)."""
    device = torch.device("cuda")
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    D_qk = latent_dim + rope_dim

    query = torch.randn(batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device)

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(
        total_pages,
        page_size,
        D_qk,
        dtype=dtype,
        device=device,
    )

    block_tables = torch.zeros(
        batch_size,
        num_pages_per_batch,
        dtype=torch.int32,
        device=device,
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p

    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)

    return (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [256, 2048])
@pytest.mark.parametrize("page_size", [64, 128])
def test_cute_dsl_mla_decode_alibi(batch_size, seq_len_k, page_size):
    """Test MLA decode with ALiBi variant (score_mod with per-head slopes)."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import ALiBiAttention

    torch.manual_seed(42)
    dtype = torch.bfloat16

    (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    ) = _make_mla_test_data(batch_size, seq_len_k, page_size, dtype)

    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    alibi_slopes = ALiBiAttention.get_slopes(num_heads).cuda()
    variant = ALiBiAttention(alibi_slopes)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    slopes_cpu = alibi_slopes.float()

    def alibi_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        return score + slopes_cpu[head_idx].item() * (kv_idx - qo_idx)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=alibi_score_mod,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [256, 2048])
@pytest.mark.parametrize("page_size", [64, 128])
def test_cute_dsl_mla_decode_soft_capping(batch_size, seq_len_k, page_size):
    """Test MLA decode with SoftCapping variant (score_mod, no extra_params)."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import SoftCappingAttention

    torch.manual_seed(42)
    dtype = torch.bfloat16

    (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    ) = _make_mla_test_data(batch_size, seq_len_k, page_size, dtype)

    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    cap = 50.0

    variant = SoftCappingAttention(cap=cap)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    import math

    def soft_capping_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        return cap * math.tanh(score.item() / cap)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=soft_capping_score_mod,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [256, 2048])
@pytest.mark.parametrize("page_size", [64, 128])
def test_cute_dsl_mla_decode_attention_sink(batch_size, seq_len_k, page_size):
    """Test MLA decode with AttentionWithSink (update_statistics + transform_output)."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import AttentionWithSink

    torch.manual_seed(42)
    dtype = torch.bfloat16
    num_heads = 128

    (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    ) = _make_mla_test_data(batch_size, seq_len_k, page_size, dtype)

    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    sink = torch.randn((num_heads,), dtype=dtype, device="cuda")
    variant = AttentionWithSink(sink)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        sink=sink.cpu(),
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [256, 2048])
@pytest.mark.parametrize("page_size", [64, 128])
def test_cute_dsl_mla_decode_rpe(batch_size, seq_len_k, page_size):
    """Test MLA decode with RPEAttention (score_mod with 2-D per-head bias table)."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import RPEAttention

    torch.manual_seed(42)
    dtype = torch.bfloat16

    (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    ) = _make_mla_test_data(batch_size, seq_len_k, page_size, dtype)

    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    max_rel_dist = 64
    table_size = 2 * max_rel_dist + 1
    rpe_table = (
        torch.randn((num_heads, table_size), dtype=torch.float32, device="cuda") * 0.1
    )
    variant = RPEAttention(rpe_table, max_rel_dist=max_rel_dist)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    rpe_cpu = rpe_table.float().cpu()

    def rpe_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        rel_pos = kv_idx - qo_idx + max_rel_dist
        rel_pos = max(0, min(rel_pos, table_size - 1))
        return score + rpe_cpu[head_idx, rel_pos].item()

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=rpe_score_mod,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


# ---------------------------------------------------------------------------
# FP8 variant tests
# ---------------------------------------------------------------------------


def _make_fp8_mla_inputs(
    batch_size, seq_len_k, page_size, num_heads=128, latent_dim=512, rope_dim=64
):
    """Helper to create FP8 query/kv/block_tables for variant tests."""
    device = torch.device("cuda")
    D_qk = latent_dim + rope_dim
    query = (
        torch.randn(batch_size, 1, num_heads, D_qk, dtype=torch.float16, device=device)
        * 0.1
    ).to(torch.float8_e4m3fn)
    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = (
        torch.randn(total_pages, page_size, D_qk, dtype=torch.float16, device=device)
        * 0.1
    ).to(torch.float8_e4m3fn)
    block_tables = torch.zeros(
        batch_size, num_pages_per_batch, dtype=torch.int32, device=device
    )
    for b in range(batch_size):
        for p in range(num_pages_per_batch):
            block_tables[b, p] = b * num_pages_per_batch + p
    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    return query, kv_cache, block_tables, seq_lens, workspace_buffer


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("page_size", [64])
def test_cute_dsl_mla_decode_fp8_alibi(batch_size, seq_len_k, page_size):
    """Test FP8 MLA decode with ALiBi variant."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import ALiBiAttention

    torch.manual_seed(42)
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_fp8_mla_inputs(
        batch_size, seq_len_k, page_size
    )
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    alibi_slopes = ALiBiAttention.get_slopes(num_heads).cuda()
    variant = ALiBiAttention(alibi_slopes)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=query.dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    D_qk = latent_dim + rope_dim
    kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim].to(torch.float32)
    q_rope = query[..., latent_dim:].to(torch.float32)

    slopes_cpu = alibi_slopes.cpu().float()

    def alibi_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        return score + slopes_cpu[head_idx].item() * (kv_idx - qo_idx)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=alibi_score_mod,
    )
    torch.testing.assert_close(
        out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("page_size", [64])
def test_cute_dsl_mla_decode_fp8_soft_capping(batch_size, seq_len_k, page_size):
    """Test FP8 MLA decode with SoftCapping variant."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import SoftCappingAttention

    torch.manual_seed(42)
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_fp8_mla_inputs(
        batch_size, seq_len_k, page_size
    )
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    cap = 50.0
    variant = SoftCappingAttention(cap=cap)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=query.dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    import math

    D_qk = latent_dim + rope_dim
    kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim].to(torch.float32)
    q_rope = query[..., latent_dim:].to(torch.float32)

    def capping_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        return cap * math.tanh(score.item() / cap)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=capping_score_mod,
    )
    torch.testing.assert_close(
        out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
    )


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("page_size", [64])
def test_cute_dsl_mla_decode_fp8_attention_sink(batch_size, seq_len_k, page_size):
    """Test FP8 MLA decode with AttentionWithSink variant."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import AttentionWithSink

    torch.manual_seed(42)
    num_heads = 128
    latent_dim = 512
    rope_dim = 64
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_fp8_mla_inputs(
        batch_size, seq_len_k, page_size
    )
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0

    sink = torch.randn((num_heads,), dtype=torch.bfloat16, device="cuda")
    variant = AttentionWithSink(sink)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=query.dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    D_qk = latent_dim + rope_dim
    kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim].to(torch.float32)
    q_rope = query[..., latent_dim:].to(torch.float32)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        sink=sink.cpu().to(torch.float32),
    )
    torch.testing.assert_close(
        out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
    )


# ---------------------------------------------------------------------------
# Regression: SoftCapping with non-tile-aligned seq_len
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("seq_len_k", [200])
@pytest.mark.parametrize("page_size", [64])
def test_cute_dsl_mla_decode_soft_capping_small_cap(batch_size, seq_len_k, page_size):
    """Expose SoftCapping + last-tile masking interaction.

    With a small cap and seq_len_k not aligned to the 128-element MMA tile,
    masked-out positions (beyond seq_len_k) are transformed from -inf to -cap
    by score_mod.  When cap is small (e.g. 1.0), -cap sits within the range of
    valid scores, giving masked positions non-negligible softmax probability.
    Those positions carry garbage KV data, corrupting the output.

    This test uses cap=1.0 and seq_len_k=200 (last tile has 72 valid + 56
    masked elements).  The reference only sums over valid positions, so any
    leakage from masked positions shows up as a numerical mismatch.
    """
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import SoftCappingAttention

    torch.manual_seed(42)
    dtype = torch.bfloat16

    (
        query,
        kv_cache,
        block_tables,
        seq_lens,
        workspace_buffer,
        num_heads,
        latent_dim,
        rope_dim,
    ) = _make_mla_test_data(batch_size, seq_len_k, page_size, dtype)

    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    cap = 1.0

    variant = SoftCappingAttention(cap=cap)

    wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
    wrapper.plan(
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        num_heads=num_heads,
        page_size=page_size,
        q_dtype=dtype,
        is_var_seq=False,
        variant=variant,
    )
    out = wrapper.run(
        q=query,
        kv_cache=kv_cache,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=softmax_scale,
        output_scale=output_scale,
    )

    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    import math

    def soft_capping_score_mod(score, batch_idx, qo_idx, kv_idx, head_idx):
        return cap * math.tanh(score.item() / cap)

    ref_out = torch_reference_mla_with_variant(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        score_mod_fn=soft_capping_score_mod,
    )
    ref_out_cast = ref_out.to(dtype)

    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)
