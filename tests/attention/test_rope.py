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
from tests.test_helpers.rope_reference import *

import flashinfer


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
def test_rope(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=torch.int32, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=torch.int32, device="cuda:0")

    # reference implementation
    if llama_version == "llama":
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 10000.0, use_scaled=False, device="cuda:0"
        ).to("cuda:0")
    else:
        freqs_cis = precompute_freqs_cis(
            rotary_dim, qkv_len + offset, 5e5, use_scaled=True, device="cuda:0"
        ).to("cuda:0")
    q_rot_ref, k_rot_ref = apply_rotary_emb(
        q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[..., :rotary_dim],
        k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[..., :rotary_dim],
        freqs_cis[offset : offset + qkv_len],
    )
    q_pass_ref = q.reshape(batch_size, qkv_len, num_qo_heads, head_dim)[
        ..., rotary_dim:
    ]
    k_pass_ref = k.reshape(batch_size, qkv_len, num_kv_heads, head_dim)[
        ..., rotary_dim:
    ]
    q_rope_ref = torch.cat([q_rot_ref, q_pass_ref], dim=-1).reshape(
        nnz, num_qo_heads, head_dim
    )
    k_rope_ref = torch.cat([k_rot_ref, k_pass_ref], dim=-1).reshape(
        nnz, num_kv_heads, head_dim
    )

    # flashinfer implementation
    if llama_version == "llama":
        if inplace:
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=1e4,
            )
            q_rope, k_rope = q, k
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=1e4,
            )
    else:
        if inplace:
            flashinfer.apply_llama31_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=5e5,
            )
            q_rope, k_rope = q, k
        else:
            q_rope, k_rope = flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=True,
                rope_theta=5e5,
            )

    # compare
    torch.testing.assert_close(q_rope_ref, q_rope, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_rope_ref, k_rope, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 19, 99, 989])
@pytest.mark.parametrize("qkv_len", [1, 4, 19, 204])
@pytest.mark.parametrize("num_qo_heads", [8, 16])
@pytest.mark.parametrize("num_kv_heads", [8])
@pytest.mark.parametrize("offset", [0, 15, 99])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize("llama_version", ["llama", "llama31"])
@pytest.mark.parametrize("partial_rotary_factor", [0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("idtype", [torch.int32, torch.int64])
def test_rope_pos_ids(
    batch_size,
    qkv_len,
    num_qo_heads,
    num_kv_heads,
    offset,
    head_dim,
    llama_version,
    partial_rotary_factor,
    inplace,
    interleave,
    idtype,
):
    rotary_dim = int(head_dim * partial_rotary_factor)
    nnz = batch_size * qkv_len
    qkv_packed = torch.randn(
        nnz,
        (num_qo_heads + 2 * num_kv_heads) * head_dim,
        dtype=torch.float16,
        device="cuda:0",
    )
    q = qkv_packed[:, : num_qo_heads * head_dim].reshape(nnz, num_qo_heads, head_dim)
    k = qkv_packed[
        :, num_qo_heads * head_dim : (num_qo_heads + num_kv_heads) * head_dim
    ].reshape(nnz, num_kv_heads, head_dim)
    indptr = torch.tensor(
        [i * qkv_len for i in range(batch_size + 1)], dtype=idtype, device="cuda:0"
    )
    offsets = torch.full((batch_size,), offset, dtype=idtype, device="cuda:0")

    pos_ids = torch.cat(
        [
            torch.arange(offset, qkv_len + offset, dtype=idtype)
            for _ in range(batch_size)
        ]
    ).to("cuda:0")

    if llama_version == "llama":
        if inplace:
            q_clone, k_clone = q.clone(), k.clone()
            flashinfer.apply_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
            q_rope, k_rope = q, k
            flashinfer.apply_rope_pos_ids_inplace(
                q_clone,
                k_clone,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
            q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
        else:
            q_rope, k_rope = flashinfer.apply_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )

            q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=1e4,
            )
    else:
        if inplace:
            q_clone, k_clone = q.clone(), k.clone()
            flashinfer.apply_llama31_rope_inplace(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )
            q_rope, k_rope = q, k
            flashinfer.apply_llama31_rope_pos_ids_inplace(
                q_clone,
                k_clone,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )
            q_rope_pos_ids, k_rope_pos_ids = q_clone, k_clone
        else:
            q_rope, k_rope = flashinfer.apply_llama31_rope(
                q,
                k,
                indptr,
                offsets,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )

            q_rope_pos_ids, k_rope_pos_ids = flashinfer.apply_llama31_rope_pos_ids(
                q,
                k,
                pos_ids,
                rotary_dim=rotary_dim,
                interleave=interleave,
                rope_theta=5e5,
            )

    # compare
    torch.testing.assert_close(q_rope_pos_ids, q_rope, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(k_rope_pos_ids, k_rope, rtol=1e-3, atol=1e-3)


class FlashInferRotaryEmbedding(RotaryEmbedding):
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        flashinfer.apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )
        return query, key


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
        (64, 32, 2048, 8432, True, torch.bfloat16, "cuda", 2, 199, 4, 1),
        (64, 64, 32, 8000, False, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (64, 64, 32, 8000, False, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 9231, False, torch.bfloat16, "cuda", 3, 231, 4, 2),
    ],
)
def test_rope_cos_sin_cache(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
):
    rope_ref = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )
    rope_flashinfer = FlashInferRotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
    )

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )

    query_ref, key_ref = query.clone(), key.clone()
    query_flashinfer, key_flashinfer = query.clone(), key.clone()

    query_ref_out, key_ref_out = rope_ref.forward_native(pos_ids, query_ref, key_ref)
    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(
        pos_ids, query_flashinfer, key_flashinfer
    )

    torch.testing.assert_close(
        query_ref_out, query_flashinfer_out, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(key_ref_out, key_flashinfer_out, atol=1e-2, rtol=1e-2)


@pytest.mark.parametrize(
    "attention_type,num_qo_heads,num_kv_heads,rope_dim,no_rope_dim",
    [
        # MLA: Multiple Q heads, single shared K/V head
        ("mla", 128, 1, 64, 512),
        ("mla", 64, 1, 128, 256),
        ("mla", 128, 1, 64, 128),  # Explicit DeepSeek R1 MLA config case
        ("mla", 32, 1, 32, 96),
        # GQA: Multiple Q heads, fewer K/V heads (grouped)
        ("gqa", 32, 8, 64, 64),
        ("gqa", 64, 16, 128, 128),
        ("gqa", 24, 6, 32, 96),
        ("gqa", 32, 8, 128, 0),  # Llama3 8B standard config
        ("gqa", 64, 8, 128, 0),  # Llama3 70B standard config
        ("gqa", 64, 8, 64, 0),  # (plausible) GPT-OSS config
        # MHA: Equal Q and K/V heads
        ("mha", 32, 32, 64, 64),
        ("mha", 16, 16, 128, 128),
        ("mha", 8, 8, 32, 96),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 19, 128, 199, 899, 2047])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_generalized_rope_quantize(
    attention_type,
    num_qo_heads,
    num_kv_heads,
    rope_dim,
    no_rope_dim,
    num_tokens,
    input_dtype,
    quant_dtype,
    enable_pdl,
):
    """Test generalized rope + quantization for MLA, GQA, and MHA architectures."""
    device = "cuda:0"
    # Fixed seed for reproducibility across tests
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    total_dim = rope_dim + no_rope_dim

    # Create input tensors based on attention type
    q_in = torch.randn(
        num_tokens, num_qo_heads, total_dim, dtype=input_dtype, device=device
    )

    if attention_type == "mla":
        # MLA: K tensor is 2D (shared across all Q heads)
        k_in = torch.randn(num_tokens, total_dim, dtype=input_dtype, device=device)
    else:
        # GQA/MHA: K tensor is 3D (multiple K heads)
        k_in = torch.randn(
            num_tokens, num_kv_heads, total_dim, dtype=input_dtype, device=device
        )

    pos_ids = torch.arange(num_tokens, device=device)

    # Create reference implementation using FlashInferRotaryEmbedding
    rope_flashinfer = FlashInferRotaryEmbedding(
        total_dim,
        rope_dim,
        4096,  # max_position_embeddings
        10000,  # base
        False,  # is_neox_style
        input_dtype,
        device,
    )

    # Compute reference output
    q_out_f16_ref, k_out_f16_ref = rope_flashinfer.forward_native(pos_ids, q_in, k_in)
    q_out_f8_ref, k_out_f8_ref = map(
        lambda x: x.to(quant_dtype),
        (q_out_f16_ref, k_out_f16_ref),
    )

    # Prepare output tensors
    q_out = torch.empty_like(q_in, dtype=quant_dtype)
    k_out = torch.empty_like(k_in, dtype=quant_dtype)

    # Split input tensors into rope and nope parts
    q_rope_in = q_in[..., :rope_dim]
    q_nope_in = q_in[..., rope_dim:]
    k_rope_in = k_in[..., :rope_dim]
    k_nope_in = k_in[..., rope_dim:]

    # Prepare output tensor slices
    q_rope_out = q_out[..., :rope_dim]
    q_nope_out = q_out[..., rope_dim:]
    k_rope_out = k_out[..., :rope_dim]
    k_nope_out = k_out[..., rope_dim:]

    # Call the generalized function
    flashinfer.rope.rope_quantize_fp8(
        q_rope_in,
        k_rope_in,
        q_nope_in,
        k_nope_in,
        rope_flashinfer.cos_sin_cache,
        pos_ids,
        is_neox=False,
        q_rope_out=q_rope_out,
        k_rope_out=k_rope_out,
        q_nope_out=q_nope_out,
        k_nope_out=k_nope_out,
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        enable_pdl=enable_pdl,
    )

    # Verify results
    torch.testing.assert_close(
        q_out_f8_ref.float(),
        q_out.float(),
        atol=1e-2,
        rtol=2e-1,
        msg=f"Q output mismatch for {attention_type} with {num_qo_heads}/{num_kv_heads} heads, {rope_dim}/{no_rope_dim} dims",
    )
    torch.testing.assert_close(
        k_out_f8_ref.float(),
        k_out.float(),
        atol=1e-2,
        rtol=2e-1,
        msg=f"K output mismatch for {attention_type} with {num_qo_heads}/{num_kv_heads} heads, {rope_dim}/{no_rope_dim} dims",
    )


@pytest.mark.parametrize(
    "attention_type,num_qo_heads,num_kv_heads,rope_dim,no_rope_dim",
    [
        # MLA: Multiple Q heads, single shared K/V head
        ("mla", 128, 1, 64, 512),
        ("mla", 64, 1, 128, 256),
        ("mla", 128, 1, 64, 128),  # Explicit DeepSeek R1 MLA config case
        ("mla", 32, 1, 32, 96),
        # GQA: Multiple Q heads, fewer K/V heads (grouped)
        ("gqa", 32, 8, 64, 64),
        ("gqa", 64, 16, 128, 128),
        ("gqa", 24, 6, 32, 96),
        ("gqa", 32, 8, 128, 0),  # Llama3 8B standard config
        ("gqa", 64, 8, 128, 0),  # Llama3 70B standard config
        ("gqa", 64, 8, 64, 0),  # (plausible) GPT-OSS config
        # MHA: Equal Q and K/V heads
        ("mha", 32, 32, 64, 64),
        ("mha", 16, 16, 128, 128),
        ("mha", 8, 8, 32, 96),
    ],
)
@pytest.mark.parametrize("num_tokens", [1, 19, 128, 199, 899, 2047])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 32])
def test_generalized_rope_quantize_append_kv_cache(
    attention_type,
    num_qo_heads,
    num_kv_heads,
    rope_dim,
    no_rope_dim,
    num_tokens,
    input_dtype,
    quant_dtype,
    enable_pdl,
    kv_layout,
    page_size,
):
    device = "cuda:0"
    # Fixed seed for reproducibility
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    head_dim = rope_dim + no_rope_dim
    batch_size = 4

    # Build inputs following the same pattern used elsewhere
    if attention_type == "mla":
        # Q: (N, Hq, *), K: 2D (N, *)
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
            )
        )
        k_rope = torch.randn(num_tokens, rope_dim, dtype=input_dtype, device=device)
        k_nope = (
            None
            if no_rope_dim == 0
            else torch.randn(num_tokens, no_rope_dim, dtype=input_dtype, device=device)
        )
        v = None
    else:
        # GQA/MHA: K/V are 3D
        q_rope = torch.randn(
            num_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_tokens, num_qo_heads, no_rope_dim, dtype=input_dtype, device=device
            )
        )
        k_rope = torch.randn(
            num_tokens, num_kv_heads, rope_dim, dtype=input_dtype, device=device
        )
        k_nope = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_tokens, num_kv_heads, no_rope_dim, dtype=input_dtype, device=device
            )
        )
        v = torch.randn(
            num_tokens, num_kv_heads, head_dim, dtype=input_dtype, device=device
        )

    # Cos/sin and positions
    max_seq_len = 4096
    rope_ref = FlashInferRotaryEmbedding(
        head_dim, rope_dim, max_seq_len, 10000, False, input_dtype, device
    )
    pos_ids = torch.arange(num_tokens, device=device, dtype=torch.int32)

    # Build paged metadata
    kv_append_length = torch.tensor(
        [num_tokens] + [0] * (batch_size - 1), dtype=torch.int32, device=device
    )
    kv_append_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length, dim=0),
        ]
    )
    num_pages_per_req = torch.tensor(
        [(num_tokens + page_size - 1) // page_size] + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indptr = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(num_pages_per_req, dim=0),
        ]
    )
    kv_page_indices = torch.arange(
        kv_page_indptr[-1].item(), dtype=torch.int32, device=device
    )
    kv_last_page_len = torch.tensor(
        [num_tokens % page_size if num_tokens % page_size != 0 else page_size]
        + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    # Allocate caches sized by required pages
    max_pages = kv_page_indptr[-1].item()

    # Get batch_indices and positions
    seq_lens = flashinfer.get_seq_lens(kv_page_indptr, kv_last_page_len, page_size)
    batch_indices, positions = flashinfer.get_batch_indices_positions(
        kv_append_indptr, seq_lens, num_tokens
    )

    # Fused call + cache allocation
    if attention_type == "mla":
        ckv_cache = torch.zeros(
            max_pages, page_size, no_rope_dim, dtype=quant_dtype, device=device
        )
        kpe_cache = torch.zeros(
            max_pages, page_size, rope_dim, dtype=quant_dtype, device=device
        )
        q_rope_out_fused, q_nope_out_fused = (
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                None,
                rope_ref.cos_sin_cache,
                pos_ids,
                (ckv_cache, kpe_cache),
                kv_page_indices,
                kv_page_indptr,
                batch_indices,
                positions,
                page_size=page_size,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                is_neox=False,
                enable_pdl=enable_pdl,
            )
        )
    else:
        # Allocate cache based on layout
        if kv_layout == "NHD":
            k_cache = torch.zeros(
                max_pages,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
            v_cache = torch.zeros(
                max_pages,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
        else:  # HND
            k_cache = torch.zeros(
                max_pages,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
            v_cache = torch.zeros(
                max_pages,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
        q_rope_out_fused, q_nope_out_fused = (
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope,
                k_rope,
                q_nope,
                k_nope,
                v,
                rope_ref.cos_sin_cache,
                pos_ids,
                (k_cache, v_cache),
                kv_page_indices,
                kv_page_indptr,
                batch_indices,
                positions,
                page_size=page_size,
                kv_layout=kv_layout,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                is_neox=False,
                enable_pdl=enable_pdl,
            )
        )
    # Compute reference output (handle None for no_rope_dim == 0)
    q_in = q_rope if q_nope is None else torch.cat([q_rope, q_nope], dim=-1)
    k_in = k_rope if k_nope is None else torch.cat([k_rope, k_nope], dim=-1)
    q_out_f16_ref, k_out_f16_ref = rope_ref.forward_native(pos_ids, q_in, k_in)
    q_out_f8_ref, k_out_f8_ref = map(
        lambda x: x.to(quant_dtype),
        (q_out_f16_ref, k_out_f16_ref),
    )

    # Fused vs Pytorch reference Q checks
    torch.testing.assert_close(
        q_out_f8_ref[..., :rope_dim].float(),
        q_rope_out_fused.float(),
        rtol=2e-1,
        atol=1e-2,
    )
    torch.testing.assert_close(
        q_out_f8_ref[..., rope_dim:].float(),
        q_nope_out_fused.float(),
        rtol=2e-1,
        atol=1e-2,
    )

    # expect 1-ULP differences between FP8 device rounding and PyTorch .to(fp8)
    if quant_dtype == torch.float8_e4m3fn:
        rtol_val, atol_val = 0.25, 0.5
    else:  # quant_dtype == torch.float8_e5m2:
        rtol_val, atol_val = 0.25, 1.0

    # if MLA: check ckv_cache, kpe_cache
    if attention_type == "mla":
        # Split K reference
        k_rope_ref = k_out_f8_ref[..., :rope_dim]
        k_nope_ref = k_out_f8_ref[..., rope_dim:]

        ckv_ref = torch.zeros_like(ckv_cache)
        kpe_ref = torch.zeros_like(kpe_cache)

        for i in range(num_tokens):
            b = batch_indices[i].item()
            pos = positions[i].item()
            page_iter = (kv_page_indptr[b].item() * page_size + pos) // page_size
            entry_idx = (kv_page_indptr[b].item() * page_size + pos) % page_size
            page_idx = kv_page_indices[page_iter].item()
            ckv_ref[page_idx, entry_idx, :] = k_nope_ref[i]
            kpe_ref[page_idx, entry_idx, :] = k_rope_ref[i]

        torch.testing.assert_close(
            ckv_cache.float(), ckv_ref.float(), rtol=rtol_val, atol=atol_val
        )
        torch.testing.assert_close(
            kpe_cache.float(), kpe_ref.float(), rtol=rtol_val, atol=atol_val
        )

    # if GQA/MHA: check k_cache, v_cache
    if attention_type == "gqa" or attention_type == "mha":
        # K reference
        k_ref = torch.zeros_like(k_cache)
        for i in range(num_tokens):
            b = batch_indices[i].item()
            pos = positions[i].item()
            page_iter = (kv_page_indptr[b].item() * page_size + pos) // page_size
            entry_idx = (kv_page_indptr[b].item() * page_size + pos) % page_size
            page_idx = kv_page_indices[page_iter].item()
            if kv_layout == "NHD":
                k_ref[page_idx, entry_idx, :, :] = k_out_f8_ref[i]  # [Hkv, head_dim]
            else:  # HND
                k_ref[page_idx, :, entry_idx, :] = k_out_f8_ref[i]  # [Hkv, head_dim]

        torch.testing.assert_close(
            k_cache.float(), k_ref.float(), rtol=rtol_val, atol=atol_val
        )

        # V reference (no RoPE on V; same quant scale as KV)
        quant_scale_kv = 1.0  # match fused call
        v_ref_tokens = (v * quant_scale_kv).to(quant_dtype)
        v_ref = torch.zeros_like(v_cache)
        for i in range(num_tokens):
            b = batch_indices[i].item()
            pos = positions[i].item()
            page_iter = (kv_page_indptr[b].item() * page_size + pos) // page_size
            entry_idx = (kv_page_indptr[b].item() * page_size + pos) % page_size
            page_idx = kv_page_indices[page_iter].item()
            if kv_layout == "NHD":
                v_ref[page_idx, entry_idx, :, :] = v_ref_tokens[i]
            else:  # HND
                v_ref[page_idx, :, entry_idx, :] = v_ref_tokens[i]

        torch.testing.assert_close(
            v_cache.float(), v_ref.float(), rtol=rtol_val, atol=atol_val
        )


@pytest.mark.parametrize(
    "attention_type,num_qo_heads,num_kv_heads,rope_dim,no_rope_dim",
    [
        # MLA: Multiple Q heads, single shared K/V head
        ("mla", 128, 1, 64, 512),
        ("mla", 32, 1, 32, 96),
        # GQA: Multiple Q heads, fewer K/V heads (grouped)
        ("gqa", 32, 8, 64, 64),
        ("gqa", 32, 8, 128, 0),  # Llama3 8B standard config
        # MHA: Equal Q and K/V heads
        ("mha", 32, 32, 64, 64),
        ("mha", 16, 16, 128, 128),
    ],
)
@pytest.mark.parametrize("num_existing_tokens", [10, 50])
@pytest.mark.parametrize("num_new_tokens", [1, 8])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("enable_pdl", [True, False])
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("page_size", [16, 32])
def test_rope_quantize_fp8_append_paged_kv_cache_decode(
    attention_type,
    num_qo_heads,
    num_kv_heads,
    rope_dim,
    no_rope_dim,
    num_existing_tokens,
    num_new_tokens,
    input_dtype,
    quant_dtype,
    enable_pdl,
    kv_layout,
    page_size,
):
    """Test append to non-empty cache (decode/continuation scenario)."""
    device = "cuda:0"
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    head_dim = rope_dim + no_rope_dim
    batch_size = 2

    # Step 1: Pre-populate cache with existing tokens
    if attention_type == "mla":
        q_rope_existing = torch.randn(
            num_existing_tokens,
            num_qo_heads,
            rope_dim,
            dtype=input_dtype,
            device=device,
        )
        q_nope_existing = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_existing_tokens,
                num_qo_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        k_rope_existing = torch.randn(
            num_existing_tokens, rope_dim, dtype=input_dtype, device=device
        )
        k_nope_existing = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_existing_tokens, no_rope_dim, dtype=input_dtype, device=device
            )
        )
        v_existing = None
    else:
        q_rope_existing = torch.randn(
            num_existing_tokens,
            num_qo_heads,
            rope_dim,
            dtype=input_dtype,
            device=device,
        )
        q_nope_existing = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_existing_tokens,
                num_qo_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        k_rope_existing = torch.randn(
            num_existing_tokens,
            num_kv_heads,
            rope_dim,
            dtype=input_dtype,
            device=device,
        )
        k_nope_existing = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_existing_tokens,
                num_kv_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        v_existing = torch.randn(
            num_existing_tokens,
            num_kv_heads,
            head_dim,
            dtype=input_dtype,
            device=device,
        )

    # Create RoPE reference
    max_seq_len = 4096
    rope_ref = FlashInferRotaryEmbedding(
        head_dim, rope_dim, max_seq_len, 10000, False, input_dtype, device
    )
    pos_ids_existing = torch.arange(
        num_existing_tokens, device=device, dtype=torch.int32
    )

    # Build metadata for existing tokens (single request for simplicity)
    kv_append_length_existing = torch.tensor(
        [num_existing_tokens] + [0] * (batch_size - 1), dtype=torch.int32, device=device
    )
    kv_append_indptr_existing = torch.cat(
        [
            torch.zeros(1, dtype=torch.int32, device=device),
            torch.cumsum(kv_append_length_existing, dim=0),
        ]
    )
    num_pages_existing = (num_existing_tokens + page_size - 1) // page_size
    kv_page_indptr_existing = torch.tensor(
        [0, num_pages_existing] + [num_pages_existing] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indices_existing = torch.arange(
        num_pages_existing, dtype=torch.int32, device=device
    )
    kv_last_page_len_existing = torch.tensor(
        [
            num_existing_tokens % page_size
            if num_existing_tokens % page_size != 0
            else page_size
        ]
        + [0] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    seq_lens_existing = flashinfer.get_seq_lens(
        kv_page_indptr_existing, kv_last_page_len_existing, page_size
    )
    batch_indices_existing, positions_existing = flashinfer.get_batch_indices_positions(
        kv_append_indptr_existing, seq_lens_existing, num_existing_tokens
    )

    # Allocate cache sized for existing + new tokens
    total_tokens = num_existing_tokens + num_new_tokens
    max_pages = (total_tokens + page_size - 1) // page_size

    if attention_type == "mla":
        ckv_cache = torch.zeros(
            max_pages, page_size, no_rope_dim, dtype=quant_dtype, device=device
        )
        kpe_cache = torch.zeros(
            max_pages, page_size, rope_dim, dtype=quant_dtype, device=device
        )
        # Pre-populate with existing tokens
        _, _ = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope_existing,
            k_rope_existing,
            q_nope_existing,
            k_nope_existing,
            None,
            rope_ref.cos_sin_cache,
            pos_ids_existing,
            (ckv_cache, kpe_cache),
            kv_page_indices_existing,
            kv_page_indptr_existing,
            batch_indices_existing,
            positions_existing,
            page_size=page_size,
            quantize_dtype=quant_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            is_neox=False,
            enable_pdl=enable_pdl,
        )
    else:
        if kv_layout == "NHD":
            k_cache = torch.zeros(
                max_pages,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
            v_cache = torch.zeros(
                max_pages,
                page_size,
                num_kv_heads,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
        else:  # HND
            k_cache = torch.zeros(
                max_pages,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
            v_cache = torch.zeros(
                max_pages,
                num_kv_heads,
                page_size,
                head_dim,
                dtype=quant_dtype,
                device=device,
            )
        # Pre-populate with existing tokens
        _, _ = flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
            q_rope_existing,
            k_rope_existing,
            q_nope_existing,
            k_nope_existing,
            v_existing,
            rope_ref.cos_sin_cache,
            pos_ids_existing,
            (k_cache, v_cache),
            kv_page_indices_existing,
            kv_page_indptr_existing,
            batch_indices_existing,
            positions_existing,
            page_size=page_size,
            kv_layout=kv_layout,
            quantize_dtype=quant_dtype,
            quant_scale_q=1.0,
            quant_scale_kv=1.0,
            is_neox=False,
            enable_pdl=enable_pdl,
        )

    # Step 2: Append new tokens to the pre-populated cache
    if attention_type == "mla":
        q_rope_new = torch.randn(
            num_new_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope_new = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_new_tokens,
                num_qo_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        k_rope_new = torch.randn(
            num_new_tokens, rope_dim, dtype=input_dtype, device=device
        )
        k_nope_new = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_new_tokens, no_rope_dim, dtype=input_dtype, device=device
            )
        )
        v_new = None
    else:
        q_rope_new = torch.randn(
            num_new_tokens, num_qo_heads, rope_dim, dtype=input_dtype, device=device
        )
        q_nope_new = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_new_tokens,
                num_qo_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        k_rope_new = torch.randn(
            num_new_tokens, num_kv_heads, rope_dim, dtype=input_dtype, device=device
        )
        k_nope_new = (
            None
            if no_rope_dim == 0
            else torch.randn(
                num_new_tokens,
                num_kv_heads,
                no_rope_dim,
                dtype=input_dtype,
                device=device,
            )
        )
        v_new = torch.randn(
            num_new_tokens, num_kv_heads, head_dim, dtype=input_dtype, device=device
        )

    pos_ids_new = torch.arange(
        num_existing_tokens,
        num_existing_tokens + num_new_tokens,
        device=device,
        dtype=torch.int32,
    )

    # Build metadata for new tokens (continue appending to first request)
    num_pages_new_needed = (total_tokens + page_size - 1) // page_size
    kv_page_indptr_new = torch.tensor(
        [0, num_pages_new_needed] + [num_pages_new_needed] * (batch_size - 1),
        dtype=torch.int32,
        device=device,
    )
    kv_page_indices_new = torch.arange(
        num_pages_new_needed, dtype=torch.int32, device=device
    )
    # For continuation, positions start at num_existing_tokens
    batch_indices_new = torch.zeros(num_new_tokens, device=device, dtype=torch.int32)
    positions_new = torch.arange(
        num_existing_tokens,
        num_existing_tokens + num_new_tokens,
        device=device,
        dtype=torch.int32,
    )

    # Snapshot existing cache for later comparison
    if attention_type == "mla":
        ckv_cache_before = ckv_cache.clone()
        kpe_cache_before = kpe_cache.clone()
    else:
        k_cache_before = k_cache.clone()
        v_cache_before = v_cache.clone()

    # Append new tokens
    if attention_type == "mla":
        q_rope_out_new, q_nope_out_new = (
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope_new,
                k_rope_new,
                q_nope_new,
                k_nope_new,
                None,
                rope_ref.cos_sin_cache,
                pos_ids_new,
                (ckv_cache, kpe_cache),
                kv_page_indices_new,
                kv_page_indptr_new,
                batch_indices_new,
                positions_new,
                page_size=page_size,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                is_neox=False,
                enable_pdl=enable_pdl,
            )
        )
    else:
        q_rope_out_new, q_nope_out_new = (
            flashinfer.rope.rope_quantize_fp8_append_paged_kv_cache(
                q_rope_new,
                k_rope_new,
                q_nope_new,
                k_nope_new,
                v_new,
                rope_ref.cos_sin_cache,
                pos_ids_new,
                (k_cache, v_cache),
                kv_page_indices_new,
                kv_page_indptr_new,
                batch_indices_new,
                positions_new,
                page_size=page_size,
                kv_layout=kv_layout,
                quantize_dtype=quant_dtype,
                quant_scale_q=1.0,
                quant_scale_kv=1.0,
                is_neox=False,
                enable_pdl=enable_pdl,
            )
        )

    # Verify Q outputs for new tokens (handle None for no_rope_dim == 0)
    q_in_new = (
        q_rope_new
        if q_nope_new is None
        else torch.cat([q_rope_new, q_nope_new], dim=-1)
    )
    k_in_new = (
        k_rope_new
        if k_nope_new is None
        else torch.cat([k_rope_new, k_nope_new], dim=-1)
    )
    q_out_f16_ref_new, k_out_f16_ref_new = rope_ref.forward_native(
        pos_ids_new, q_in_new, k_in_new
    )
    q_out_f8_ref_new = q_out_f16_ref_new.to(quant_dtype)
    k_out_f8_ref_new = k_out_f16_ref_new.to(quant_dtype)

    torch.testing.assert_close(
        q_out_f8_ref_new[..., :rope_dim].float(),
        q_rope_out_new.float(),
        rtol=2e-1,
        atol=1e-2,
    )
    torch.testing.assert_close(
        q_out_f8_ref_new[..., rope_dim:].float(),
        q_nope_out_new.float(),
        rtol=2e-1,
        atol=1e-2,
    )

    # FP8 tolerances
    if quant_dtype == torch.float8_e4m3fn:
        rtol_val, atol_val = 0.25, 0.5
    else:
        rtol_val, atol_val = 0.25, 1.0

    # Verify existing cache entries remain unchanged
    if attention_type == "mla":
        # Check that entries before num_existing_tokens are unchanged
        for i in range(num_existing_tokens):
            b = batch_indices_existing[i].item()
            pos = positions_existing[i].item()
            page_iter = (
                kv_page_indptr_existing[b].item() * page_size + pos
            ) // page_size
            entry_idx = (
                kv_page_indptr_existing[b].item() * page_size + pos
            ) % page_size
            page_idx = kv_page_indices_existing[page_iter].item()
            torch.testing.assert_close(
                ckv_cache[page_idx, entry_idx, :].float(),
                ckv_cache_before[page_idx, entry_idx, :].float(),
                rtol=0,
                atol=0,
                msg=f"Existing CKV cache entry {i} was modified",
            )
            torch.testing.assert_close(
                kpe_cache[page_idx, entry_idx, :].float(),
                kpe_cache_before[page_idx, entry_idx, :].float(),
                rtol=0,
                atol=0,
                msg=f"Existing KPE cache entry {i} was modified",
            )
    else:
        for i in range(num_existing_tokens):
            b = batch_indices_existing[i].item()
            pos = positions_existing[i].item()
            page_iter = (
                kv_page_indptr_existing[b].item() * page_size + pos
            ) // page_size
            entry_idx = (
                kv_page_indptr_existing[b].item() * page_size + pos
            ) % page_size
            page_idx = kv_page_indices_existing[page_iter].item()
            if kv_layout == "NHD":
                torch.testing.assert_close(
                    k_cache[page_idx, entry_idx, :, :].float(),
                    k_cache_before[page_idx, entry_idx, :, :].float(),
                    rtol=0,
                    atol=0,
                    msg=f"Existing K cache entry {i} was modified",
                )
                torch.testing.assert_close(
                    v_cache[page_idx, entry_idx, :, :].float(),
                    v_cache_before[page_idx, entry_idx, :, :].float(),
                    rtol=0,
                    atol=0,
                    msg=f"Existing V cache entry {i} was modified",
                )
            else:  # HND
                torch.testing.assert_close(
                    k_cache[page_idx, :, entry_idx, :].float(),
                    k_cache_before[page_idx, :, entry_idx, :].float(),
                    rtol=0,
                    atol=0,
                    msg=f"Existing K cache entry {i} was modified",
                )
                torch.testing.assert_close(
                    v_cache[page_idx, :, entry_idx, :].float(),
                    v_cache_before[page_idx, :, entry_idx, :].float(),
                    rtol=0,
                    atol=0,
                    msg=f"Existing V cache entry {i} was modified",
                )

    # Verify new cache entries are correct
    if attention_type == "mla":
        k_rope_ref_new = k_out_f8_ref_new[..., :rope_dim]
        k_nope_ref_new = k_out_f8_ref_new[..., rope_dim:]

        for i in range(num_new_tokens):
            b = batch_indices_new[i].item()
            pos = positions_new[i].item()
            page_iter = (kv_page_indptr_new[b].item() * page_size + pos) // page_size
            entry_idx = (kv_page_indptr_new[b].item() * page_size + pos) % page_size
            page_idx = kv_page_indices_new[page_iter].item()
            torch.testing.assert_close(
                ckv_cache[page_idx, entry_idx, :].float(),
                k_nope_ref_new[i].float(),
                rtol=rtol_val,
                atol=atol_val,
            )
            torch.testing.assert_close(
                kpe_cache[page_idx, entry_idx, :].float(),
                k_rope_ref_new[i].float(),
                rtol=rtol_val,
                atol=atol_val,
            )
    else:
        quant_scale_kv = 1.0
        v_ref_tokens_new = (v_new * quant_scale_kv).to(quant_dtype)

        for i in range(num_new_tokens):
            b = batch_indices_new[i].item()
            pos = positions_new[i].item()
            page_iter = (kv_page_indptr_new[b].item() * page_size + pos) // page_size
            entry_idx = (kv_page_indptr_new[b].item() * page_size + pos) % page_size
            page_idx = kv_page_indices_new[page_iter].item()
            if kv_layout == "NHD":
                torch.testing.assert_close(
                    k_cache[page_idx, entry_idx, :, :].float(),
                    k_out_f8_ref_new[i].float(),
                    rtol=rtol_val,
                    atol=atol_val,
                )
                torch.testing.assert_close(
                    v_cache[page_idx, entry_idx, :, :].float(),
                    v_ref_tokens_new[i].float(),
                    rtol=rtol_val,
                    atol=atol_val,
                )
            else:  # HND
                torch.testing.assert_close(
                    k_cache[page_idx, :, entry_idx, :].float(),
                    k_out_f8_ref_new[i].float(),
                    rtol=rtol_val,
                    atol=atol_val,
                )
                torch.testing.assert_close(
                    v_cache[page_idx, :, entry_idx, :].float(),
                    v_ref_tokens_new[i].float(),
                    rtol=rtol_val,
                    atol=atol_val,
                )


@pytest.mark.parametrize("num_tokens", [1, 19, 128, 199, 899, 2047])
@pytest.mark.parametrize("input_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("quant_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("enable_pdl", [True, False])
def test_mla_rope_quantize(
    num_tokens,
    input_dtype,
    quant_dtype,
    enable_pdl,
):
    device = "cuda:0"
    # Fixed seed for reproducibility across tests
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    num_qo_heads = 128
    q_in = torch.randn(num_tokens, num_qo_heads, 576, dtype=input_dtype, device=device)
    k_in = torch.randn(num_tokens, 576, dtype=input_dtype, device=device)
    pos_ids = torch.arange(num_tokens, device=device)

    # baseline
    rope_flashinfer = FlashInferRotaryEmbedding(
        576,
        64,
        4096,
        10000,
        False,  # is_neox_style
        input_dtype,
        device,
    )

    q_out_f16_ref, k_out_f16_ref = rope_flashinfer.forward_native(pos_ids, q_in, k_in)
    q_out_f8_ref, k_out_f8_ref = map(
        lambda x: x.to(quant_dtype),
        (q_out_f16_ref, k_out_f16_ref),
    )

    q_out = torch.empty_like(q_in, dtype=quant_dtype)
    k_out = torch.empty_like(k_in, dtype=quant_dtype)
    flashinfer.rope.mla_rope_quantize_fp8(
        q_in[..., :64],
        k_in[..., :64],
        q_in[..., 64:],
        k_in[..., 64:],
        rope_flashinfer.cos_sin_cache,
        pos_ids,
        is_neox=False,
        q_rope_out=q_out[..., :64],
        k_rope_out=k_out[..., :64],
        q_nope_out=q_out[..., 64:],
        k_nope_out=k_out[..., 64:],
        quant_scale_q=1.0,
        quant_scale_kv=1.0,
        enable_pdl=enable_pdl,
    )

    torch.testing.assert_close(
        q_out_f8_ref.float(), q_out.float(), atol=1e-2, rtol=2e-1
    )
    torch.testing.assert_close(
        k_out_f8_ref.float(), k_out.float(), atol=1e-2, rtol=2e-1
    )


if __name__ == "__main__":
    # test_rope(2, 1, 8, 8, 1, 128, "llama", 1.0, False)
    # test_rope_pos_ids(2, 1, 8, 8, 1, 128, "llama31", 1.0, False)
    # test_rope_cos_sin_cache(
    #     64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1
    # )
    test_mla_rope_quantize(1, torch.float16, torch.float8_e4m3fn)
