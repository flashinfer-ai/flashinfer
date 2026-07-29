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


def skip_if_sm100a_unsupported():
    """Guard public auto-routing assertions that are specific to SM100/SM103."""
    device = torch.device("cuda")
    if not is_sm100a_supported(device):
        pytest.skip("Requires SM100/SM103")
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
    # Copy the small metadata arrays once.  Calling CUDA ``.item()`` for every
    # request and page serializes hundreds of device synchronizations at B128.
    cache_seqs_host = cache_seqs.tolist()
    page_table_host = page_table.tolist()
    for b in range(B):
        seq_len = cache_seqs_host[b]
        num_pages_needed = (seq_len + page_size - 1) // page_size

        # Gather KV for this batch via page table
        page_indices = page_table_host[b][:num_pages_needed]
        kv_indices = []
        for p in page_indices:
            start = p * page_size
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


def torch_reference_variable_q_mla(
    q_nope,
    q_rope,
    c_latent,
    c_rope,
    page_table,
    cache_seqs,
    q_lens,
    softmax_scale,
    output_scale,
    page_size,
):
    """Reference compact variable-Q by evaluating each request independently."""
    outputs = []
    lses = []
    q_begin = 0
    for batch_idx, request_q_len in enumerate(q_lens):
        if request_q_len == 0:
            continue
        q_end = q_begin + request_q_len
        request_out, request_lse = torch_reference_mla(
            q_nope[q_begin:q_end].unsqueeze(0),
            q_rope[q_begin:q_end].unsqueeze(0),
            c_latent,
            c_rope,
            page_table[batch_idx : batch_idx + 1],
            cache_seqs[batch_idx : batch_idx + 1],
            softmax_scale,
            output_scale,
            page_size,
            apply_mtp_mask=True,
            return_lse=True,
        )
        outputs.append(request_out.squeeze(0))
        lses.append(request_lse.squeeze(0))
        q_begin = q_end
    if not outputs:
        return (
            q_nope.new_empty((0, q_nope.shape[1], q_nope.shape[2])),
            torch.empty(
                0,
                q_nope.shape[1],
                dtype=torch.float32,
                device=q_nope.device,
            ),
        )
    return torch.cat(outputs), torch.cat(lses)


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


def _run_padded_q_tile_case(
    batch_size,
    seq_len_k,
    num_heads,
    q_len,
    dtype,
    is_var_seq=False,
    enable_pdl=None,
    out_dtype=None,
    query_token_stride=1,
    via_public_api=False,
    q_lens=None,
    max_q_len=None,
    public_backend="cute-dsl",
    page_size=64,
):
    """Run and reference-check one monolithic packed-query configuration.

    ``q_lens=None`` uses the fixed rectangular query layout. Otherwise,
    ``q_lens`` describes compact per-request query segments and ``q_len`` is
    the static launch capacity.
    """
    skip_if_unsupported()

    torch.manual_seed(42)
    device = torch.device("cuda")

    latent_dim = 512
    rope_dim = 64
    softmax_scale = 1.0 / (latent_dim**0.5)
    output_scale = 1.0
    D_qk = latent_dim + rope_dim
    is_var_q = q_lens is not None
    if is_var_q:
        q_lens = list(q_lens)
        batch_size = len(q_lens)
        total_q = sum(q_lens)
        cum_seq_lens_q = torch.tensor(
            [0, *torch.tensor(q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        query_storage_shape = (
            total_q * query_token_stride,
            num_heads,
            D_qk,
        )
    else:
        total_q = batch_size * q_len
        cum_seq_lens_q = None
        query_storage_shape = (
            batch_size,
            q_len * query_token_stride,
            num_heads,
            D_qk,
        )

    # torch.randn doesn't support fp8; for FP8 inputs create as fp16 then convert.
    is_fp8 = dtype == torch.float8_e4m3fn
    query_storage = torch.randn(
        *query_storage_shape,
        dtype=torch.float16 if is_fp8 else dtype,
        device=device,
    )
    if is_fp8:
        query_storage = (query_storage * 0.1).to(torch.float8_e4m3fn)
    query = (
        query_storage[::query_token_stride]
        if is_var_q
        else query_storage[:, ::query_token_stride]
    )
    if query_token_stride != 1:
        assert query.stride(-3) != num_heads * query.stride(-2)

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

    if is_var_seq:
        seq_lens = torch.tensor(
            [max(page_size, seq_len_k - b * 37) for b in range(batch_size)],
            dtype=torch.int32,
            device=device,
        )
    else:
        seq_lens = torch.full(
            (batch_size,), seq_len_k, dtype=torch.int32, device=device
        )

    workspace_factory = torch.zeros if via_public_api else torch.empty
    workspace_buffer = workspace_factory(
        256 * 1024 * 1024, dtype=torch.int8, device=device
    )

    if via_public_api:
        assert out_dtype is None
        from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

        lse_shape = (total_q, num_heads) if is_var_q else (batch_size, q_len, num_heads)
        lse_out = torch.empty(lse_shape, dtype=torch.float32, device=device)
        out, lse = trtllm_batch_decode_with_kv_cache_mla(
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
            bmm2_scale=output_scale,
            backend=public_backend,
            is_var_seq=is_var_seq,
            enable_pdl=enable_pdl,
            lse=lse_out,
            return_lse=True,
            cute_dsl_impl="monolithic",
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )
    else:
        from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

        out, lse = cute_dsl_mla_decode(
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
            is_var_seq=is_var_seq,
            enable_pdl=enable_pdl,
            cute_dsl_impl="auto" if is_var_q else "monolithic",
            return_lse=True,
            out_dtype=out_dtype,
            cum_seq_lens_q=cum_seq_lens_q,
            max_q_len=max_q_len,
        )

    if via_public_api:
        expected_out_dtype = torch.bfloat16
    elif out_dtype is not None:
        expected_out_dtype = out_dtype
    elif is_fp8:
        expected_out_dtype = torch.bfloat16
    else:
        expected_out_dtype = dtype
    assert out.dtype == expected_out_dtype

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

    # Monolithic always applies the request-local MTP causal mask.  Compact
    # variable-Q reference segments independently so each request uses its
    # actual q_len rather than the launch capacity.
    if is_var_q:
        ref_out, ref_lse = torch_reference_variable_q_mla(
            q_nope,
            q_rope,
            c_latent_ref,
            c_rope_ref,
            block_tables,
            seq_lens,
            q_lens,
            softmax_scale,
            output_scale,
            page_size,
        )
    else:
        ref_out, ref_lse = torch_reference_mla(
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
            return_lse=True,
        )

    if is_fp8:
        # FP8 has limited precision; compare in FP32 with wider tolerance.
        torch.testing.assert_close(
            out.to(torch.float32), ref_out.to(torch.float32), atol=0.1, rtol=0.1
        )
        torch.testing.assert_close(lse, ref_lse, atol=0.2, rtol=0.1)
    else:
        ref_out_cast = ref_out.to(out.dtype)
        torch.testing.assert_close(out, ref_out_cast, atol=1e-2, rtol=1e-2)
        torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


# Exercises the spec-decoding (MTP) causal mask + flat query-row packing.
# A cooperative 2-CTA tile owns 128 consecutive (token, head) rows and may
# cross token boundaries; only the final flattened tile is safely padded.
# All paths share the same kernel, and the MTP mask applies for q_len > 1.
# Monolithic-only: the modular path doesn't implement packed query tiles or MTP.
@pytest.mark.parametrize(
    "num_heads,q_len,dtype",
    [
        pytest.param(64, 3, torch.bfloat16, id="h64-sq3-bf16"),
        pytest.param(24, 13, torch.float8_e4m3fn, id="h24-sq13-fp8"),
        pytest.param(12, 11, torch.bfloat16, id="h12-sq11-bf16-tail4"),
    ],
)
def test_cute_dsl_mla_decode_padded_q_tiles(num_heads, q_len, dtype):
    """Cover cross-token rows and final tails in both kernel families."""
    _run_padded_q_tile_case(1, 1024, num_heads, q_len, dtype)


@pytest.mark.parametrize(
    "num_heads,dtype",
    [
        pytest.param(12, torch.bfloat16, id="h12-bf16"),
        pytest.param(24, torch.bfloat16, id="h24-bf16"),
        pytest.param(48, torch.bfloat16, id="h48-bf16"),
        pytest.param(96, torch.bfloat16, id="h96-bf16"),
        pytest.param(12, torch.float8_e4m3fn, id="h12-fp8"),
        pytest.param(24, torch.float8_e4m3fn, id="h24-fp8"),
        pytest.param(48, torch.float8_e4m3fn, id="h48-fp8"),
        pytest.param(96, torch.float8_e4m3fn, id="h96-fp8"),
    ],
)
def test_cute_dsl_mla_decode_variable_q_packing(num_heads, dtype):
    """Compact ragged Q preserves per-request token/head capacity packing."""
    _run_padded_q_tile_case(
        4,
        16 * 1024,
        num_heads,
        8,
        dtype,
        q_lens=[8, 1, 0, 3],
        max_q_len=8,
    )


def test_cute_dsl_mla_decode_variable_q_infers_max_q_len():
    """Inference supports a strided compact query and a wider page size."""
    _run_padded_q_tile_case(
        2,
        128,
        24,
        8,
        torch.bfloat16,
        q_lens=[3, 1],
        query_token_stride=2,
        page_size=128,
    )


def test_cute_dsl_mla_decode_variable_q_fp8_persistent_inactive_gap():
    """Persistent FP8 clusters can skip ragged slots and resume later work."""
    _run_padded_q_tile_case(
        148,
        128,
        96,
        8,
        torch.float8_e4m3fn,
        q_lens=[8, *([0] * 146), 1],
        max_q_len=8,
    )


def test_cute_dsl_mla_decode_variable_q_all_empty():
    """A positive launch capacity permits an all-empty compact FP8 batch."""
    _run_padded_q_tile_case(
        2,
        128,
        24,
        8,
        torch.float8_e4m3fn,
        q_lens=[0, 0],
        max_q_len=8,
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_cute_dsl_mla_decode_variable_q_and_k_lengths(dtype):
    """Ragged Q composes with per-request KV lengths and split reduction."""
    _run_padded_q_tile_case(
        4,
        8 * 1024,
        24,
        8,
        dtype,
        is_var_seq=True,
        q_lens=[8, 1, 0, 3],
        max_q_len=8,
    )


def test_cute_dsl_mla_decode_variable_q_cuda_graph_replay():
    """A captured rectangular launch can replay with new ragged metadata."""
    skip_if_unsupported()
    from flashinfer.cute_dsl.attention import cute_dsl_mla_decode

    torch.manual_seed(42)
    device = torch.device("cuda")
    batch_size, num_heads, total_q, max_q_len = 4, 96, 12, 8
    seq_len_k, page_size = 128, 64
    latent_dim, rope_dim = 512, 64
    d_qk = latent_dim + rope_dim
    scale = 1.0 / (latent_dim**0.5)

    query = torch.randn(total_q, num_heads, d_qk, dtype=torch.bfloat16, device=device)
    pages_per_request = seq_len_k // page_size
    kv_cache = torch.randn(
        batch_size * pages_per_request,
        page_size,
        d_qk,
        dtype=torch.bfloat16,
        device=device,
    )
    block_tables = torch.arange(
        batch_size * pages_per_request,
        dtype=torch.int32,
        device=device,
    ).view(batch_size, pages_per_request)
    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    out = torch.empty(
        total_q, num_heads, latent_dim, dtype=torch.bfloat16, device=device
    )
    lse = torch.empty(total_q, num_heads, dtype=torch.float32, device=device)
    cum_q = torch.tensor([0, 4, 8, 12, 12], dtype=torch.int32, device=device)

    def launch():
        return cute_dsl_mla_decode(
            query=query,
            kv_cache=kv_cache,
            workspace_buffer=workspace,
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=rope_dim,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_seq_len=seq_len_k,
            softmax_scale=scale,
            out=out,
            lse=lse,
            return_lse=True,
            is_var_seq=False,
            cute_dsl_impl="monolithic",
            cum_seq_lens_q=cum_q,
            max_q_len=max_q_len,
        )

    def reference(q_lens):
        kv_flat = kv_cache.reshape(-1, d_qk)
        return torch_reference_variable_q_mla(
            query[..., :latent_dim],
            query[..., latent_dim:],
            kv_flat[:, :latent_dim],
            kv_flat[:, latent_dim:],
            block_tables,
            seq_lens,
            q_lens,
            scale,
            1.0,
            page_size,
        )

    # Compile and allocate all internal state before capture.
    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = reference([4, 4, 4, 0])
    torch.testing.assert_close(out, ref_out.to(out.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)

    cum_q.copy_(torch.tensor([0, 0, 1, 4, 12], dtype=torch.int32, device=device))
    graph.replay()
    torch.cuda.synchronize()
    ref_out, ref_lse = reference([0, 1, 3, 8])
    torch.testing.assert_close(out, ref_out.to(out.dtype), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(lse, ref_lse, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
@pytest.mark.parametrize(
    "q_lens",
    [None, [32, 1]],
    ids=["fixed-q", "variable-q"],
)
def test_cute_dsl_mla_decode_per_q_tile_mask_boundary(dtype, q_lens):
    """Use each request-local M128 tile's first token for dense-mask selection."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
        cute_dsl_mla_decode,
    )
    from flashinfer.cute_dsl.utils import get_num_sm

    q_len, num_heads = 32, 6
    is_var_q = q_lens is not None
    request_q_lens = [q_len] if q_lens is None else q_lens
    batch_size = len(request_q_lens)
    total_q = sum(request_q_lens)
    seq_len_k, page_size = 144, 64
    latent_dim, rope_dim = 512, 64
    d_qk = latent_dim + rope_dim
    device = torch.device("cuda")

    # H6/Sq32 has two M128 tiles. Token 21 straddles their boundary: heads
    # 0-1 are in tile 0 and heads 2-5 are in tile 1. Tile 1 can treat keys
    # 0:128 as dense from its first token (21), while tile 0 must still mask
    # keys 113:128 for its earlier tokens. Using tile 0's last token for the
    # coarse decision would expose those marked values incorrectly.
    query_shape = (
        (total_q, num_heads, d_qk) if is_var_q else (batch_size, q_len, num_heads, d_qk)
    )
    query = torch.zeros(query_shape, dtype=dtype, device=device)
    num_pages = (seq_len_k + page_size - 1) // page_size
    kv_cache = torch.zeros(
        batch_size * num_pages,
        page_size,
        d_qk,
        dtype=dtype,
        device=device,
    )
    kv_flat = kv_cache.view(-1, d_qk)
    for batch_idx in range(batch_size):
        request_begin = batch_idx * num_pages * page_size
        kv_flat[request_begin + 113 : request_begin + 128, 0] = 4.0
    block_tables = torch.arange(
        batch_size * num_pages, dtype=torch.int32, device=device
    ).view(batch_size, num_pages)
    seq_lens = torch.full((batch_size,), seq_len_k, dtype=torch.int32, device=device)
    cum_seq_lens_q = (
        torch.tensor(
            [0, *torch.tensor(request_q_lens).cumsum(0).tolist()],
            dtype=torch.int32,
            device=device,
        )
        if is_var_q
        else None
    )

    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        batch_size,
        q_len,
        num_heads,
        latent_dim,
        get_num_sm(device),
        seq_len_k,
    )
    assert split_kv > 1
    workspace = torch.empty(workspace_size, dtype=torch.int8, device=device)

    out, lse = cute_dsl_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        kv_lora_rank=latent_dim,
        qk_rope_head_dim=rope_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=seq_len_k,
        softmax_scale=1.0 / (latent_dim**0.5),
        return_lse=True,
        enable_pdl=False,
        cum_seq_lens_q=cum_seq_lens_q,
        max_q_len=q_len if is_var_q else None,
    )

    bounds = torch.cat(
        [
            torch.arange(request_q_len, dtype=torch.float32, device=device)
            + (seq_len_k - request_q_len + 1)
            for request_q_len in request_q_lens
        ]
    )
    marked_visible = (bounds - 113).clamp(min=0, max=15)
    expected_out_flat = torch.zeros(
        total_q,
        num_heads,
        latent_dim,
        dtype=torch.float32,
        device=device,
    )
    expected_out_flat[..., 0] = (4.0 * marked_visible / bounds).view(total_q, 1)
    expected_lse_flat = torch.log(bounds).view(total_q, 1).expand(total_q, num_heads)
    expected_out = expected_out_flat if is_var_q else expected_out_flat.unsqueeze(0)
    expected_lse = expected_lse_flat if is_var_q else expected_lse_flat.unsqueeze(0)

    torch.testing.assert_close(out.float(), expected_out, atol=5e-3, rtol=5e-3)
    torch.testing.assert_close(lse, expected_lse, atol=2e-3, rtol=2e-3)


@pytest.mark.parametrize("num_heads", [12, 24, 48, 96])
@pytest.mark.parametrize("q_len", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_cute_dsl_mla_decode_packed_q_accuracy_matrix(num_heads, q_len, dtype):
    """Qualify the supported packed-query, causal-mask, and auto-split matrix."""
    # Keep the batch sweep within one pytest item while all 128 requested
    # Cartesian cells are reference checked. Split-KV and query packing are
    # selected automatically by the public launcher.
    for batch_size in (1, 4, 16, 128):
        _run_padded_q_tile_case(
            batch_size=batch_size,
            seq_len_k=1024,
            num_heads=num_heads,
            q_len=q_len,
            dtype=dtype,
        )


def test_cute_dsl_mla_decode_fp8_persistent_multi_work_boundaries():
    """Reuse the FP8 two-softmax pipelines across persistent work items."""
    # B128 exceeds one resident wave of 2CTA clusters on supported Blackwell
    # devices.  These cases exercise one-, even-, and odd-K-tile work items;
    # the packed-Q matrix above additionally covers the eight-tile boundary.
    for seq_len_k, enable_pdl in ((128, True), (256, False), (384, True)):
        _run_padded_q_tile_case(
            batch_size=128,
            seq_len_k=seq_len_k,
            num_heads=96,
            q_len=1,
            dtype=torch.float8_e4m3fn,
            enable_pdl=enable_pdl,
        )


def test_cute_dsl_mla_decode_fp8_variable_seq_order_boundaries():
    """Balance both FP8 softmax groups for mixed nonpersistent K parity."""
    _run_padded_q_tile_case(
        batch_size=10,
        seq_len_k=385,
        num_heads=96,
        q_len=8,
        dtype=torch.float8_e4m3fn,
        is_var_seq=True,
        enable_pdl=False,
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_cute_dsl_mla_decode_h96_max_split_reducer_capacity(dtype):
    """Reference-check output and LSE at the static reducer's 32-split cap."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=8192,
        num_heads=96,
        q_len=1,
        dtype=dtype,
    )


def test_cute_dsl_mla_decode_h96_sq8_nonempty_split_reducer():
    """Reference-check the normalized H96/Sq8 long-K split-reducer path."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=8192,
        num_heads=96,
        q_len=8,
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8"]
)
def test_cute_dsl_mla_decode_h96_odd_split_reducer_pdl_off(dtype):
    """Cover adaptive D4 with an odd 17-split prefix and PDL disabled."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=4097,
        num_heads=96,
        q_len=1,
        dtype=dtype,
        enable_pdl=False,
    )


def test_cute_dsl_mla_decode_variable_seq_d2_reducer():
    """Cover adaptive D2 when batches have distinct non-power split prefixes."""
    _run_padded_q_tile_case(
        batch_size=2,
        seq_len_k=4097,
        num_heads=24,
        q_len=1,
        dtype=torch.bfloat16,
        is_var_seq=True,
        enable_pdl=False,
    )


def test_cute_dsl_mla_decode_padded_q_tile_direct_output():
    """Exercise H48/Sq3 full/tail padding with split_kv=1 (no reducer)."""
    skip_if_unsupported()

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
    )
    from flashinfer.cute_dsl.utils import get_num_sm

    num_q_tiles = 2
    num_sm = get_num_sm(torch.device("cuda"))
    batch_size = (num_sm + 2 * num_q_tiles - 1) // (2 * num_q_tiles)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        batch_size, 3, 48, 512, num_sm
    )
    assert split_kv == 1
    assert workspace_size == 0

    _run_padded_q_tile_case(
        batch_size=batch_size,
        seq_len_k=128,
        num_heads=48,
        q_len=3,
        dtype=torch.float16,
    )


def test_cute_dsl_mla_decode_padded_q_tile_variable_seq():
    """Cover nonpersistent variable-K scheduling with a padded H24/Sq6 tail."""
    _run_padded_q_tile_case(
        batch_size=3,
        seq_len_k=385,
        num_heads=24,
        q_len=6,
        dtype=torch.float16,
        is_var_seq=True,
        enable_pdl=False,
    )


def test_cute_dsl_mla_decode_padded_q_tile_fp8_output():
    """Qualify packed-query FP8 input and FP8 output together."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=128,
        num_heads=64,
        q_len=3,
        dtype=torch.float8_e4m3fn,
        out_dtype=torch.float8_e4m3fn,
    )


def test_cute_dsl_mla_decode_padded_q_tile_strided_query():
    """Cover the contiguous fallback for a token-strided query view."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=128,
        num_heads=48,
        q_len=3,
        dtype=torch.float16,
        query_token_stride=2,
    )


@pytest.mark.parametrize("buffer_name", ["out", "lse"])
def test_cute_dsl_mla_decode_rejects_token_gapped_output(buffer_name):
    """Reject output views that cannot represent flat rows across tokens."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        cute_dsl_mla_decode,
    )

    batch_size, q_len, num_heads = 1, 2, 96
    latent_dim, rope_dim = 512, 64
    query = torch.empty(
        batch_size,
        q_len,
        num_heads,
        latent_dim + rope_dim,
        dtype=torch.bfloat16,
    )
    out = torch.empty(batch_size, q_len * 2, num_heads, latent_dim)[:, ::2]
    lse = torch.empty(batch_size, q_len * 2, num_heads, dtype=torch.float32)[:, ::2]
    kwargs = {buffer_name: out if buffer_name == "out" else lse}

    with pytest.raises(ValueError, match=rf"{buffer_name} must be contiguous"):
        cute_dsl_mla_decode(
            query=query,
            kv_cache=torch.empty(1, 64, latent_dim + rope_dim, dtype=torch.bfloat16),
            workspace_buffer=torch.empty(0, dtype=torch.int8),
            kv_lora_rank=latent_dim,
            qk_rope_head_dim=rope_dim,
            block_tables=torch.zeros(1, 1, dtype=torch.int32),
            seq_lens=torch.tensor([64], dtype=torch.int32),
            max_seq_len=64,
            softmax_scale=1.0,
            **kwargs,
        )


def test_cute_dsl_mla_decode_padded_q_tile_via_public_api():
    """Exercise the reported H96/Sq8 shape through the public dispatcher."""
    _run_padded_q_tile_case(
        batch_size=1,
        seq_len_k=128,
        num_heads=96,
        q_len=8,
        dtype=torch.bfloat16,
        enable_pdl=False,
        via_public_api=True,
    )


def test_compute_q_tile_layout():
    """Unit test the shared host/kernel query-tile geometry."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl.attention.monolithic.mla_helpers import (
        compute_q_tile_layout,
    )

    cases = [
        # (H, Sq, M, (total_rows, num_tiles, tail_rows))
        (128, 3, 128, (384, 3, 128)),
        (96, 1, 128, (96, 1, 96)),
        (96, 3, 128, (288, 3, 32)),
        (48, 8, 128, (384, 3, 128)),
        (12, 11, 128, (132, 2, 4)),
    ]
    for H, S_q, m_tile, expected in cases:
        assert compute_q_tile_layout(H, S_q, m_tile) == expected, (
            f"compute_q_tile_layout({H}, {S_q}, {m_tile}) "
            f"= {compute_q_tile_layout(H, S_q, m_tile)}, expected {expected}"
        )

    for invalid in [(0, 1, 128), (129, 1, 128), (64, 0, 128), (64, 1, 0)]:
        with pytest.raises(ValueError):
            compute_q_tile_layout(*invalid)


def test_nonpersistent_grid_y_limit():
    """Reject only nonpersistent grids beyond CUDA's Y-dimension limit."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _validate_nonpersistent_grid_y,
    )

    _validate_nonpersistent_grid_y(8_191, 8, is_persistent=False)
    _validate_nonpersistent_grid_y(65_535, 1, is_persistent=False)
    _validate_nonpersistent_grid_y(8_192, 8, is_persistent=True)

    with pytest.raises(ValueError, match=r"grid\.y would be 65536"):
        _validate_nonpersistent_grid_y(8_192, 8, is_persistent=False)


def test_mla_reducer_d_tile_selection():
    """Use output bands only when they shorten an underfilled reducer wave."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_reducer_d_tiles,
    )

    # B1/H32 and B1/H96 use D4; B1/H64 uses D2. H128 already fills a wave.
    assert _get_reducer_d_tiles(1, 1, 32, 148, 32) == 4
    assert _get_reducer_d_tiles(1, 1, 64, 148, 32) == 2
    assert _get_reducer_d_tiles(1, 1, 96, 148, 32) == 4
    # Prefer the smaller tied topology and avoid duplication once rows fill a wave.
    assert _get_reducer_d_tiles(1, 1, 48, 148, 32) == 2
    assert _get_reducer_d_tiles(1, 1, 128, 148, 32) == 1
    assert _get_reducer_d_tiles(4, 1, 96, 148, 32) == 1
    assert _get_reducer_d_tiles(1, 1, 96, 0, 32) == 1
    # Do not duplicate LSE work when a short sequence exposes too few splits.
    assert _get_reducer_d_tiles(1, 1, 96, 148, 1) == 1
    assert _get_reducer_d_tiles(1, 1, 24, 148, 2) == 2
    # Variable-Q reducers still launch their rectangular B x max_q_len grid.
    assert _get_reducer_d_tiles(148, 8, 96, 148, 32) == 1


def test_mla_reducer_direct_class_capacity_defaults():
    """Direct class users keep the generic capacity unless opting into a cap."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    import cutlass

    from flashinfer.cute_dsl.attention.monolithic.mla_decode_fp16 import (
        BlackwellMultiHeadLatentAttentionForwardFP16,
    )
    from flashinfer.cute_dsl.attention.monolithic.mla_decode_fp8 import (
        BlackwellMultiHeadLatentAttentionForwardFP8,
    )
    from flashinfer.cute_dsl.attention.monolithic.mla_helpers import MAX_SPLITS

    kwargs = dict(
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=1,
        page_size=64,
        skip_correction_threshold=0.0,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        enable_pdl=False,
    )
    for kernel_cls in (
        BlackwellMultiHeadLatentAttentionForwardFP16,
        BlackwellMultiHeadLatentAttentionForwardFP8,
    ):
        assert kernel_cls(**kwargs).reducer_max_splits == MAX_SPLITS
        assert kernel_cls(**kwargs, reducer_max_splits=32).reducer_max_splits == 32
        with pytest.raises(ValueError, match="variable split-KV"):
            kernel_cls(**{**kwargs, "is_var_split_kv": True}, reducer_max_splits=32)


def test_flat_q_tile_split_workspace_geometry():
    """Workspace sizing uses the flattened M128 query-tile count."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
    )

    batch_size, num_heads, q_len, num_q_tiles = 1, 48, 8, 3
    latent_dim = 512
    max_active_blocks = 148
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        batch_size,
        q_len,
        num_heads,
        latent_dim,
        max_active_blocks,
    )
    expected_split = min(max_active_blocks // (batch_size * num_q_tiles * 2), 32)
    expected_workspace = (
        batch_size * 128 * num_q_tiles * expected_split * (latent_dim + 1) * 4
    )
    assert split_kv == expected_split
    assert workspace_size == expected_workspace

    # Once one cluster per (batch, query tile) fills the machine, split-KV is
    # disabled and no workspace is required.
    direct_batch = max_active_blocks // (num_q_tiles * 2)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        direct_batch,
        q_len,
        num_heads,
        latent_dim,
        max_active_blocks,
    )
    assert split_kv == 1
    assert workspace_size == 0


def test_h96_sq8_split_workspace_drops_empty_partition():
    """Size only the eight nonempty K partitions at B1/H96/Sq8."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")
    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size,
    )

    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        1, 8, 96, 512, 148, 1024
    )
    assert split_kv == 8
    assert workspace_size == 1 * 128 * 6 * 8 * (512 + 1) * 4


def test_cute_dsl_workspace_sizer_follows_selected_impl():
    """Autotuning must size workspace with the implementation it will launch."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
        _get_split_kv_and_workspace_size as monolithic_sizer,
    )
    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        _get_split_kv_and_workspace_size as modular_sizer,
    )
    from flashinfer.mla._core import _get_cute_dsl_workspace_sizer

    assert _get_cute_dsl_workspace_sizer("monolithic", None) is monolithic_sizer
    assert _get_cute_dsl_workspace_sizer("modular", None) is modular_sizer
    assert _get_cute_dsl_workspace_sizer("auto", None) is monolithic_sizer
    assert _get_cute_dsl_workspace_sizer("auto", torch.empty(0)) is modular_sizer


def test_monolithic_workspace_cap_drops_empty_partitions():
    """Autotuning must use the launcher's max-sequence split normalization."""
    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL not available")

    from flashinfer.mla._core import _cute_dsl_max_supported_batch

    # One K tile needs no split workspace for any candidate batch. Without
    # max_seq_len propagation this is conservatively sized as 32 splits and
    # the zero-byte workspace incorrectly caps the autotune sweep at B=1.
    assert (
        _cute_dsl_max_supported_batch(
            workspace_bytes=0,
            q_len=1,
            num_heads=128,
            kv_lora_rank=512,
            max_active_blocks=148,
            max_seq_len=128,
            candidate_max=8,
            cute_dsl_impl="monolithic",
            sinks=None,
        )
        == 8
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


def _mla_decode_inputs(batch_size=4, seq_len_k=512, num_heads=128, page_size=128):
    device = torch.device("cuda")
    latent_dim = 512
    rope_dim = 64
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
    args = _mla_decode_inputs()
    out = trtllm_batch_decode_with_kv_cache_mla(**args, backend="auto")
    assert out.shape == (args["query"].size(0), 1, args["query"].size(2), 512)
    assert torch.isfinite(out).all()


def test_mla_decode_trtllm_gen_rejects_block_size_128():
    skip_if_unsupported()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    args = _mla_decode_inputs()
    with pytest.raises(ValueError, match=r"trtllm-gen requires block_size"):
        trtllm_batch_decode_with_kv_cache_mla(**args, backend="trtllm-gen")


def test_mla_decode_auto_dispatches_to_cute_dsl_for_trtllm_head_gap():
    skip_if_unsupported()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    torch.manual_seed(42)
    args = _mla_decode_inputs(num_heads=96, page_size=64)
    out = trtllm_batch_decode_with_kv_cache_mla(**args, backend="auto")
    assert out.shape == (args["query"].size(0), 1, 96, 512)
    assert torch.isfinite(out).all()


def test_mla_decode_trtllm_gen_rejects_head_gap():
    skip_if_unsupported()
    from flashinfer.mla import trtllm_batch_decode_with_kv_cache_mla

    args = _mla_decode_inputs(num_heads=96, page_size=64)
    with pytest.raises(
        ValueError,
        match=r"64 < num_heads_q < 128.*backend='cute-dsl'",
    ):
        trtllm_batch_decode_with_kv_cache_mla(**args, backend="trtllm-gen")


def test_mla_decode_variable_q_auto_uses_cute_dsl_for_head_gap():
    """Auto-routing returns accurate compact H96 output and LSE."""
    skip_if_sm100a_unsupported()
    _run_padded_q_tile_case(
        1,
        16 * 1024,
        96,
        8,
        torch.bfloat16,
        via_public_api=True,
        q_lens=[1],
        max_q_len=8,
        public_backend="auto",
    )
