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


# Tests that exercise the standalone cute_dsl_mla_decode function or the
# public trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl") path
# pass this fixture's value as the cute_dsl_impl= kwarg, exercising both
# implementations explicitly.  Variant tests use BatchMLADecodeCuteDSLWrapper
# directly (which is modular-only) and are not parametrized here.
@pytest.fixture(params=["modular", "monolithic"], ids=["modular", "monolithic"])
def cute_dsl_impl(request):
    return request.param


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
    apply_mtp_mask=False,
    return_lse=False,
):
    """PyTorch reference implementation for MLA decode.

    When ``apply_mtp_mask`` is True, applies the spec-decoding (MTP) causal
    mask the monolithic kernel uses: for q_token qi ∈ [0, q_len), valid KV
    positions are [0, seq_len - q_len + 1 + qi). For q_len=1 this reduces
    to the plain K-bound check (no-op).  The modular implementation does
    not apply this mask, so callers exercising the modular path should
    leave ``apply_mtp_mask=False``.

    When ``return_lse=True``, also returns the Log-Sum-Exp of the
    pre-softmax scores: ``LSE = log(sum(exp(QK^T * softmax_scale)))``
    in natural log, matching the cute_dsl kernel's LSE convention.

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
        apply_mtp_mask: bool — whether to apply the MTP causal mask.
        return_lse: bool — also return LSE [B, q_len, H] (float32).
    """
    B, q_len, H, latent_dim = q_nope.shape

    outputs = []
    lses = []
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

        # Spec-decoding (MTP) causal mask: row qi's k_bound is seq_len-(q_len-1)+qi.
        if apply_mtp_mask and q_len > 1:
            mask = torch.zeros(q_len, seq_len, dtype=torch.bool, device=attn.device)
            for qi in range(q_len):
                upper = max(0, seq_len - q_len + 1 + qi)
                mask[qi, :upper] = True
            attn = attn.masked_fill(~mask.unsqueeze(1), float("-inf"))

        if return_lse:
            # LSE = logsumexp over the KV dimension (natural log).
            lses.append(torch.logsumexp(attn, dim=-1))  # [q_len, H]

        # Softmax
        attn = F.softmax(attn, dim=-1)

        # Output: attn @ V (V = k_latent for MLA)
        # [q_len, H, seq_len] @ [seq_len, latent_dim] -> [q_len, H, latent_dim]
        out_b = torch.einsum("qhk,kd->qhd", attn, k_latent.float())
        out_b = out_b * output_scale
        outputs.append(out_b)

    out_stack = torch.stack(outputs, dim=0)  # [B, q_len, H, latent_dim]
    if return_lse:
        return out_stack, torch.stack(lses, dim=0)  # ([B,q_len,H,D], [B,q_len,H])
    return out_stack


@pytest.mark.parametrize("batch_size", [1, 4, 32])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048, 8192])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("q_len", [1, 2])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_cute_dsl_mla_decode_fp16(
    batch_size, seq_len_k, page_size, dtype, q_len, enable_pdl, cute_dsl_impl
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

    # Run kernel.  Request LSE in the native 3D [B, q_len, H] shape; the
    # wrapper also accepts [B*q_len, H] (trtllm-gen shape) which gets
    # reshaped internally.
    # LSE output is currently monolithic-only; the modular path raises
    # NotImplementedError, so only request it on the monolithic path.
    result = cute_dsl_mla_decode(
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
        cute_dsl_impl=cute_dsl_impl,
        return_lse=(cute_dsl_impl == "monolithic"),
    )
    if cute_dsl_impl == "monolithic":
        out, lse = result
        assert lse.dtype == torch.float32
        assert lse.shape == (batch_size, q_len, num_heads)
    else:
        out = result
        lse = None

    # Reference
    kv_flat = kv_cache.reshape(-1, latent_dim + rope_dim)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim]
    q_rope = query[..., latent_dim:]

    # Monolithic applies the MTP causal mask for q_len > 1; modular does not.
    ref = torch_reference_mla(
        q_nope,
        q_rope,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        apply_mtp_mask=(cute_dsl_impl == "monolithic"),
        return_lse=(cute_dsl_impl == "monolithic"),
    )
    if cute_dsl_impl == "monolithic":
        ref_out, ref_lse = ref
    else:
        ref_out = ref
        ref_lse = None

    ref_out_cast = ref_out.to(dtype)

    # Check with tolerance appropriate for FP16/BF16
    torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)
    if cute_dsl_impl == "monolithic":
        # LSE is float32 — tighter tolerance.
        torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


# Exercises the spec-decoding (MTP) causal mask + fold_sq path: num_heads < 128
# forces the kernel to pack F = compute_fold_sq_ratio(H, q_len, 128) tokens of
# q_len into the head dim so the 128-wide MMA-M tile is fully populated.
# (H=128, q_len=any) → F=1 (no fold), (H=64, q_len=2) → F=2, (H=64, q_len=4) → F=2,
# (H=32, q_len=4) → F=4, (H=32, q_len=2) → F=2.  All paths share the same
# kernel; the MTP causal mask is applied uniformly for q_len > 1.
# Monolithic-only: the modular path doesn't implement fold_sq or the MTP mask.
@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 1024])
@pytest.mark.parametrize("num_heads", [16, 32, 64])
@pytest.mark.parametrize("q_len", [2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float8_e4m3fn])
def test_cute_dsl_mla_decode_fold_sq(
    batch_size, seq_len_k, num_heads, q_len, dtype, cute_dsl_impl
):
    """Verify the MTP causal mask + fold_sq packing for H ≤ 128 and q_len > 1."""
    if cute_dsl_impl != "monolithic":
        pytest.skip("fold_sq / MTP causal mask are monolithic-only features")
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

    page_size = 64
    latent_dim = 512
    rope_dim = 64
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    D_qk = latent_dim + rope_dim

    # torch.randn doesn't support fp8; for FP8 inputs create as fp16 then convert.
    is_fp8 = dtype == torch.float8_e4m3fn
    if is_fp8:
        query = (
            torch.randn(
                batch_size, q_len, num_heads, D_qk, dtype=torch.float16, device=device
            )
            * 0.1
        ).to(torch.float8_e4m3fn)
    else:
        query = torch.randn(
            batch_size, q_len, num_heads, D_qk, dtype=dtype, device=device
        )

    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    if is_fp8:
        kv_cache = (
            torch.randn(
                total_pages, page_size, D_qk, dtype=torch.float16, device=device
            )
            * 0.1
        ).to(torch.float8_e4m3fn)
    else:
        kv_cache = torch.randn(total_pages, page_size, D_qk, dtype=dtype, device=device)

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
        is_var_seq=False,
        cute_dsl_impl=cute_dsl_impl,
    )

    # FP8 input → BF16 output (default), so do the reference in FP32 with wider tolerance.
    if is_fp8:
        kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
        q_nope = query[..., :latent_dim].to(torch.float32)
        q_rope = query[..., latent_dim:].to(torch.float32)
    else:
        kv_flat = kv_cache.reshape(-1, D_qk)
        q_nope = query[..., :latent_dim]
        q_rope = query[..., latent_dim:]
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]

    # Monolithic-only test — kernel always applies the MTP causal mask here.
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
        apply_mtp_mask=True,
    )

    if is_fp8:
        # FP8 has limited precision; compare in FP32 with wider tolerance.
        torch.testing.assert_close(
            out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
        )
    else:
        ref_out_cast = ref_out.to(dtype)
        torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)


def test_compute_fold_sq_ratio():
    """Unit test the static helper used by both run() and the wrapper."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl.attention.monolithic.mla_decode_fp16 import (
        BlackwellMultiHeadLatentAttentionForwardFP16 as FP16,
    )
    from flashinfer.cute_dsl.attention.monolithic.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8 as FP8,
    )

    cases = [
        # (num_heads, seq_len_q, m_tile, expected)
        (128, 1, 128, 1),  # H == M_tile → no fold
        (128, 4, 128, 1),  # H == M_tile → no fold
        (64, 1, 128, 1),  # seq_len_q=1 → F=1
        (64, 2, 128, 2),  # exact divisor, H*F=128 ≤ M_tile
        (64, 4, 128, 2),  # H*F ≤ 128 caps F at 2; 4 % 2 == 0
        (64, 3, 128, 1),  # 3's only divisors are 1 and 3; H*3=192 > M_tile → F=1
        (32, 4, 128, 4),  # tighter pack: F=4, H*F=128
        (32, 8, 128, 4),  # capped by M_tile/H = 4
        (32, 3, 128, 3),  # max_fold=min(3, 4)=3; 3 % 3 == 0 → F=3
        (16, 8, 128, 8),  # max_fold=min(8, 8)=8; 8 % 8 == 0 → F=8
        (16, 6, 128, 6),  # max_fold=min(6, 8)=6; 6 % 6 == 0 → F=6
    ]
    for H, S_q, m_tile, expected in cases:
        assert FP16.compute_fold_sq_ratio(H, S_q, m_tile) == expected, (
            f"FP16.compute_fold_sq_ratio({H}, {S_q}, {m_tile}) "
            f"= {FP16.compute_fold_sq_ratio(H, S_q, m_tile)}, expected {expected}"
        )
        assert FP8.compute_fold_sq_ratio(H, S_q, m_tile) == expected, (
            f"FP8.compute_fold_sq_ratio({H}, {S_q}, {m_tile}) "
            f"= {FP8.compute_fold_sq_ratio(H, S_q, m_tile)}, expected {expected}"
        )


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("seq_len_k", [128, 512, 2048])
@pytest.mark.parametrize("page_size", [32, 128])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cute_dsl_mla_decode_variable_seq_len(
    batch_size, seq_len_k, page_size, dtype, cute_dsl_impl
):
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
        cute_dsl_impl=cute_dsl_impl,
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


@pytest.mark.parametrize("batch_size", [1, 4, 128])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("num_heads", [128, 64])
def test_cute_dsl_mla_decode_via_api(
    batch_size, seq_len_k, num_heads, cute_dsl_impl, page_size=128, enable_pdl=False
):
    """Test MLA decode via the trtllm_batch_decode_with_kv_cache_mla API with cute-dsl backend."""
    skip_if_unsupported()

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(42)
    device = torch.device("cuda")

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
        cute_dsl_impl=cute_dsl_impl,
    )

    assert out.shape == (batch_size, q_len, num_heads, latent_dim)
    assert torch.isfinite(out).all(), "cute-dsl MLA decode produced non-finite values"


@pytest.mark.parametrize("batch_size", [1, 4])
@pytest.mark.parametrize("seq_len_k", [128, 512])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_cute_dsl_vs_trtllm_gen(
    batch_size, seq_len_k, enable_pdl, cute_dsl_impl, page_size=64
):
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
        **common_args,
        backend="cute-dsl",
        is_var_seq=False,
        cute_dsl_impl=cute_dsl_impl,
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
@pytest.mark.parametrize("num_heads", [128, 64])
@pytest.mark.parametrize("enable_pdl", [False])
def test_cute_dsl_mla_decode_fp8(
    batch_size, seq_len_k, page_size, num_heads, enable_pdl, cute_dsl_impl
):
    """Test FP8 MLA decode kernel against FP32 reference."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")

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

    # Exercise the 2D trtllm-gen-style lse buffer here for coverage when
    # available (monolithic only — the modular path raises NotImplementedError
    # for LSE output).  The wrapper reshapes via .view to the kernel's native
    # [B, q_len, H] layout.
    lse_buf = (
        torch.empty((batch_size * q_len, num_heads), dtype=torch.float32, device=device)
        if cute_dsl_impl == "monolithic"
        else None
    )
    result = cute_dsl_mla_decode(
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
        cute_dsl_impl=cute_dsl_impl,
        lse=lse_buf,
        return_lse=(cute_dsl_impl == "monolithic"),
    )
    if cute_dsl_impl == "monolithic":
        out, lse = result
        # Caller-supplied buffer must be returned (identity), not a copy.
        assert lse.data_ptr() == lse_buf.data_ptr()
        assert lse.shape == (batch_size * q_len, num_heads)
        assert lse.dtype == torch.float32
        assert torch.isfinite(lse).all(), "FP8 cute-dsl MLA LSE produced non-finite"
    else:
        out = result
        lse = None

    assert out.dtype == torch.bfloat16
    assert out.shape == (batch_size, q_len, num_heads, latent_dim)
    assert torch.isfinite(out).all(), "FP8 cute-dsl MLA decode produced non-finite"

    # Reference: compute in FP32 using FP8 values dequantized to FP32
    kv_flat = kv_cache.reshape(-1, D_qk).to(torch.float32)
    c_latent_ref = kv_flat[:, :latent_dim]
    c_rope_ref = kv_flat[:, latent_dim:]
    q_nope = query[..., :latent_dim].to(torch.float32)
    q_rope_tensor = query[..., latent_dim:].to(torch.float32)

    ref = torch_reference_mla(
        q_nope,
        q_rope_tensor,
        c_latent_ref,
        c_rope_ref,
        block_tables,
        seq_lens,
        softmax_scale,
        output_scale,
        page_size,
        return_lse=(cute_dsl_impl == "monolithic"),
    )
    if cute_dsl_impl == "monolithic":
        ref_out, ref_lse = ref
    else:
        ref_out = ref
        ref_lse = None
    # Compare outputs in FP32; FP8 has limited precision so use wider tolerance
    torch.testing.assert_close(
        out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
    )
    if cute_dsl_impl == "monolithic":
        # LSE reshaped back to native shape for comparison.  FP8 quantization
        # noise propagates into LSE so use the same wide tolerance as `out`.
        torch.testing.assert_close(
            lse.view(batch_size, q_len, num_heads), ref_lse, atol=0.1, rtol=0.1
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


@pytest.mark.parametrize("cute_dsl_impl_arg", ["auto", "modular"])
def test_cute_dsl_mla_decode_via_api_with_sinks(cute_dsl_impl_arg):
    """Public trtllm_batch_decode_with_kv_cache_mla(backend='cute-dsl', sinks=...)
    works end-to-end on both ``cute_dsl_impl="auto"`` (which auto-promotes
    to modular due to sinks) and ``cute_dsl_impl="modular"`` (explicit).
    The ``cute_dsl_impl="monolithic"`` case is the strict-error contract
    covered separately by test_via_api_monolithic_with_sinks_raises below.

    Single shape is sufficient — sinks correctness across shapes is
    already covered by test_cute_dsl_mla_decode_attention_sink; this
    test pins the dispatcher's auto/modular branches at the public API.
    """
    skip_if_unsupported()
    batch_size, seq_len_k, page_size = 4, 2048, 64

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

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
    sink = torch.randn((num_heads,), dtype=dtype, device="cuda")

    # The public API takes a 4D KV cache: [num_pages, 1, page_size, D]
    out = trtllm_batch_decode_with_kv_cache_mla(
        query=query,
        kv_cache=kv_cache.unsqueeze(1),
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=latent_dim,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        bmm1_scale=1.0 / (latent_dim**0.5),
        bmm2_scale=1.0,
        sinks=sink,
        backend="cute-dsl",
        is_var_seq=False,
        cute_dsl_impl=cute_dsl_impl_arg,
    )
    assert out.shape == (batch_size, query.shape[1], num_heads, latent_dim)
    assert torch.isfinite(out).all(), (
        "public-API cute-dsl with sinks produced non-finite values"
    )


def test_via_api_monolithic_with_sinks_raises():
    """Strict-mode contract: cute_dsl_impl='monolithic' + sinks must raise
    ValueError, never silently substitute modular.  Inputs are minimal —
    we just need to reach the dispatcher's resolver, not actually run the
    kernel."""
    skip_if_unsupported()

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

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
    ) = _make_mla_test_data(batch_size=1, seq_len_k=128, page_size=64, dtype=dtype)
    sink = torch.randn((num_heads,), dtype=dtype, device="cuda")

    with pytest.raises(ValueError, match=r"monolithic.*sinks.*modular"):
        trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=latent_dim,
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=rope_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=128,
            bmm1_scale=1.0 / (latent_dim**0.5),
            bmm2_scale=1.0,
            sinks=sink,
            backend="cute-dsl",
            is_var_seq=False,
            cute_dsl_impl="monolithic",
        )


def test_via_api_cute_dsl_sinks_wrong_shape_raises():
    """The cute-dsl standalone validates the sinks shape at the API boundary
    instead of letting a wrong-shape tensor surface as a confusing kernel
    failure.  ``AttentionWithSink.update_statistics`` indexes
    ``self.params[qo_head_idx]``, so the tensor must be 1-D of length
    num_qo_heads."""
    skip_if_unsupported()

    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

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
    ) = _make_mla_test_data(batch_size=1, seq_len_k=128, page_size=64, dtype=dtype)

    # Off-by-one length triggers the shape check.
    wrong_sink = torch.randn((num_heads + 1,), dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match=r"shape \(num_qo_heads,\)"):
        trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=latent_dim,
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=rope_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=128,
            bmm1_scale=1.0 / (latent_dim**0.5),
            bmm2_scale=1.0,
            sinks=wrong_sink,
            backend="cute-dsl",
            is_var_seq=False,
        )

    # 2-D shape also rejected, even if total numel matches.
    wrong_sink_2d = torch.randn((1, num_heads), dtype=dtype, device="cuda")
    with pytest.raises(ValueError, match=r"shape \(num_qo_heads,\)"):
        trtllm_batch_decode_with_kv_cache_mla(
            query=query,
            kv_cache=kv_cache.unsqueeze(1),
            workspace_buffer=workspace_buffer,
            qk_nope_head_dim=latent_dim,
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=rope_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=128,
            bmm1_scale=1.0 / (latent_dim**0.5),
            bmm2_scale=1.0,
            sinks=wrong_sink_2d,
            backend="cute-dsl",
            is_var_seq=False,
        )


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
@pytest.mark.parametrize("num_heads", [128, 64])
def test_cute_dsl_mla_decode_fp8_alibi(batch_size, seq_len_k, page_size, num_heads):
    """Test FP8 MLA decode with ALiBi variant."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        BatchMLADecodeCuteDSLWrapper,
    )
    from flashinfer.cute_dsl.attention.fusion.variant import ALiBiAttention

    torch.manual_seed(42)
    latent_dim = 512
    rope_dim = 64
    query, kv_cache, block_tables, seq_lens, workspace_buffer = _make_fp8_mla_inputs(
        batch_size, seq_len_k, page_size, num_heads=num_heads
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
    assert torch.isfinite(out).all(), "FP8 ALiBi MLA decode produced non-finite"
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


def _mla_decode_block_size_128_inputs(batch_size=4, seq_len_k=512, num_heads=128):
    device = torch.device("cuda")
    latent_dim = 512
    rope_dim = 64
    page_size = 128
    D_qk = latent_dim + rope_dim
    query = torch.randn(
        batch_size, 1, num_heads, D_qk, dtype=torch.float16, device=device
    )
    num_pages_per_batch = (seq_len_k + page_size - 1) // page_size
    total_pages = num_pages_per_batch * batch_size + 10
    kv_cache = torch.randn(
        total_pages, page_size, D_qk, dtype=torch.float16, device=device
    )
    block_tables = torch.arange(
        batch_size * num_pages_per_batch, dtype=torch.int32, device=device
    ).view(batch_size, num_pages_per_batch)
    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    return dict(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace_buffer,
        qk_nope_head_dim=latent_dim,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        bmm1_scale=1.0 / (latent_dim**0.5),
        bmm2_scale=1.0,
        is_var_seq=False,
    )


def test_mla_decode_auto_dispatches_to_cute_dsl_for_block_size_128():
    skip_if_unsupported()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(42)
    args = _mla_decode_block_size_128_inputs()
    out = trtllm_batch_decode_with_kv_cache_mla(**args, backend="auto")
    assert out.shape == (args["query"].size(0), 1, args["query"].size(2), 512)
    assert torch.isfinite(out).all()


def test_mla_decode_trtllm_gen_rejects_block_size_128():
    skip_if_unsupported()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    args = _mla_decode_block_size_128_inputs()
    with pytest.raises(ValueError, match=r"trtllm-gen requires block_size"):
        trtllm_batch_decode_with_kv_cache_mla(**args, backend="trtllm-gen")
