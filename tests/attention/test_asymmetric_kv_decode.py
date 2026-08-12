"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for asymmetric K/V cache dtypes in batch decode: the K cache stays
at native precision (fp16/bf16) while the V cache is stored in FP8. Keys
feed the softmax score path, so K precision dominates output quality;
values sit behind the associative reduction, so FP8 V costs only FP8-level
noise. These tests validate the plumbing (JIT URI keying, plan/run dtype
validation, backend guards) and the kernel numerics against a float32
torch reference that reads K natively and dequantizes V.
"""

import itertools

import pytest
import torch

import flashinfer
from flashinfer.jit.attention.modules import (
    gen_batch_decode_module,
    get_batch_decode_uri,
)
from flashinfer.utils import has_flashinfer_jit_cache

K_DTYPES = [torch.float16, torch.bfloat16]
V_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2]
HEAD_DIMS = [128, 256]

# fp8-e4m3 quantization of V bounds the relative error of the attention
# output; empirically the median sits near 0.02 for random inputs. 0.10
# gives comfortable headroom without letting a K-side bug (whose error
# would be O(1)) pass.
MEDIAN_REL_ERR_BOUND = 0.10


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    specs = []
    for k_dtype, v_dtype, head_dim in itertools.product(K_DTYPES, V_DTYPES, HEAD_DIMS):
        specs.append(
            gen_batch_decode_module(
                k_dtype,  # dtype_q (q matches k precision)
                k_dtype,  # dtype_kv (anchor; dtype_k defaults to this)
                k_dtype,  # dtype_o
                torch.int32,
                head_dim,
                head_dim,
                0,  # pos_encoding_mode
                False,  # use_sliding_window
                False,  # use_logits_soft_cap
                dtype_k=k_dtype,
                dtype_v=v_dtype,
            )
        )
    flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


def _build_paged_cache(
    batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, dtype
):
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    if kv_layout == "HND":
        shape = [total_num_pages, num_kv_heads, page_size, head_dim]
    else:
        shape = [total_num_pages, page_size, num_kv_heads, head_dim]
    data_fp32 = torch.randn(*shape, dtype=torch.float32, device="cuda:0") * 0.5
    data = data_fp32.to(dtype)
    return data, data_fp32, num_pages_per_seq, total_num_pages


def _gather_seq(cache_fp32, indptr, last_page_len, i, kv_layout):
    """Gather sequence i's entries from a paged cache into [L, H, D]."""
    if kv_layout == "HND":
        full = cache_fp32[indptr[i] : indptr[i + 1] - 1].permute(0, 2, 1, 3)
        last = cache_fp32[indptr[i + 1] - 1, :, : last_page_len[i]].permute(1, 0, 2)
    else:
        full = cache_fp32[indptr[i] : indptr[i + 1] - 1]
        last = cache_fp32[indptr[i + 1] - 1, : last_page_len[i]]
    num_kv_heads, head_dim = full.shape[-2], full.shape[-1]
    return torch.cat([full.reshape(-1, num_kv_heads, head_dim), last], dim=0)


def _ref_decode(q, k, v, sm_scale):
    """float32 reference decode. q: [H_qo, D]; k, v: [L, H_kv, D]."""
    num_qo_heads = q.shape[0]
    num_kv_heads = k.shape[1]
    group_size = num_qo_heads // num_kv_heads
    k = k.repeat_interleave(group_size, dim=1)  # [L, H_qo, D]
    v = v.repeat_interleave(group_size, dim=1)
    logits = torch.einsum("hd,lhd->hl", q.float(), k.float()) * sm_scale
    p = torch.softmax(logits, dim=-1)
    return torch.einsum("hl,lhd->hd", p, v.float())


@pytest.mark.parametrize("batch_size", [7, 61])
@pytest.mark.parametrize("kv_len", [54, 517])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [4, 32])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
@pytest.mark.parametrize("kv_layout", ["NHD", "HND"])
@pytest.mark.parametrize("k_dtype", K_DTYPES)
@pytest.mark.parametrize("v_dtype", V_DTYPES)
def test_batch_decode_asymmetric_kv(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    k_dtype,
    v_dtype,
):
    torch.manual_seed(42)
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=k_dtype)
    k_cache, _, num_pages_per_seq, total_num_pages = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, k_dtype
    )
    v_cache, _, _, _ = _build_paged_cache(
        batch_size, kv_len, page_size, num_kv_heads, head_dim, kv_layout, v_dtype
    )

    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,),
        (kv_len - 1) % page_size + 1,
        dtype=torch.int32,
        device="cuda:0",
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    o = wrapper.run(q, (k_cache, v_cache))

    # Reference: K read at its stored precision (native fp16/bf16), V
    # dequantized from fp8 -- exactly what the kernel is expected to do.
    sm_scale = 1.0 / (head_dim**0.5)
    k_stored_fp32 = k_cache.float()
    v_dequant_fp32 = v_cache.float()
    rel_errs = []
    for i in range(batch_size):
        ki = _gather_seq(k_stored_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        vi = _gather_seq(v_dequant_fp32, kv_indptr, kv_last_page_len, i, kv_layout)
        o_ref = _ref_decode(q[i], ki, vi, sm_scale)
        rel = (o[i].float() - o_ref).norm() / o_ref.norm().clamp_min(1e-6)
        rel_errs.append(rel.item())
    median_rel_err = sorted(rel_errs)[len(rel_errs) // 2]
    assert median_rel_err < MEDIAN_REL_ERR_BOUND, (
        f"median rel err {median_rel_err} exceeds {MEDIAN_REL_ERR_BOUND} "
        f"(k={k_dtype}, v={v_dtype})"
    )


def test_asymmetric_one_hot_v_selection_and_lse():
    """Structured oracle: K concentrates attention on one known token per
    sequence and V carries a distinctive fp8-representable constant there,
    so the output must reproduce the selected V row exactly. The LSE and
    output are also compared against the symmetric kernel run on the
    dequantized V (a convention-free oracle: LSE depends only on Q and K,
    and K is stored bit-exact)."""
    batch, page_size, pages_per_seq = 4, 16, 4
    num_kv = num_qo = 4
    head_dim = 128
    k_dtype, v_dtype = torch.bfloat16, torch.float8_e4m3fn

    q = torch.ones(batch, num_qo, head_dim, device="cuda:0", dtype=k_dtype)
    k = torch.zeros(
        batch * pages_per_seq,
        page_size,
        num_kv,
        head_dim,
        device="cuda:0",
        dtype=k_dtype,
    )
    v = torch.zeros(
        batch * pages_per_seq,
        page_size,
        num_kv,
        head_dim,
        device="cuda:0",
        dtype=torch.float32,
    )
    expected = torch.zeros(batch, num_qo, head_dim, device="cuda:0")
    targets = [3, 17, 40, 62]  # one token per sequence, distinct pages
    for i, t in enumerate(targets):
        page, slot = divmod(t, page_size)
        k[i * pages_per_seq + page, slot, :, :] = 10.0
        val = 0.5 + 0.25 * i  # exactly representable in fp8 e4m3
        v[i * pages_per_seq + page, slot, :, :] = val
        expected[i, :, :] = val
    v_fp8 = v.to(v_dtype)

    indptr = (
        torch.arange(0, batch + 1, device="cuda:0", dtype=torch.int32) * pages_per_seq
    )
    indices = torch.arange(0, batch * pages_per_seq, device="cuda:0", dtype=torch.int32)
    last = torch.full((batch,), page_size, dtype=torch.int32, device="cuda:0")
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    w = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, "NHD")
    w.plan(
        indptr,
        indices,
        last,
        num_qo,
        num_kv,
        head_dim,
        page_size,
        q_data_type=k_dtype,
        k_data_type=k_dtype,
        v_data_type=v_dtype,
    )
    out, lse = w.run(q, (k, v_fp8), return_lse=True)
    assert (out.float() - expected).abs().max().item() < 1e-2
    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(lse).all()

    w_sym = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0"),
        "NHD",
    )
    w_sym.plan(
        indptr,
        indices,
        last,
        num_qo,
        num_kv,
        head_dim,
        page_size,
        q_data_type=k_dtype,
        kv_data_type=k_dtype,
    )
    out_sym, lse_sym = w_sym.run(q, (k, v_fp8.to(k_dtype)), return_lse=True)
    assert (lse.float() - lse_sym.float()).abs().max().item() < 1e-2
    assert (out.float() - out_sym.float()).abs().max().item() < 1e-2


def test_asymmetric_uri_backward_compat():
    """Symmetric callers must keep their exact JIT cache keys."""
    common = (
        torch.float16,  # dtype_q
        torch.float16,  # dtype_kv
        torch.float16,  # dtype_o
        torch.int32,
        128,
        128,
        0,
        False,
        False,
    )
    legacy = get_batch_decode_uri(*common)
    explicit_symmetric = get_batch_decode_uri(
        *common, dtype_k=torch.float16, dtype_v=torch.float16
    )
    assert legacy == explicit_symmetric
    assert "dtype_kv_f16" in legacy

    asym = get_batch_decode_uri(
        *common, dtype_k=torch.float16, dtype_v=torch.float8_e4m3fn
    )
    assert asym != legacy
    assert "dtype_k_f16" in asym and "dtype_v_e4m3" in asym


def _plan_asym(wrapper, head_dim=128, page_size=16):
    batch_size, kv_len = 4, 64
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(
        0, num_pages_per_seq * batch_size, device="cuda:0", dtype=torch.int32
    )
    kv_last_page_len = torch.full(
        (batch_size,), page_size, dtype=torch.int32, device="cuda:0"
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        4,
        4,
        head_dim,
        page_size,
        q_data_type=torch.float16,
        k_data_type=torch.float16,
        v_data_type=torch.float8_e4m3fn,
    )
    return batch_size, kv_len


def test_asymmetric_run_v_dtype_mismatch_raises():
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    batch_size, kv_len = _plan_asym(wrapper)
    page_size, num_kv_heads, head_dim = 16, 4, 128
    total_pages = ((kv_len + page_size - 1) // page_size) * batch_size
    q = torch.randn(batch_size, 4, head_dim, device="cuda:0", dtype=torch.float16)
    k_cache = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    # V handed over in fp16 against a plan that promised fp8 must raise,
    # not silently reinterpret bytes.
    v_cache_wrong = torch.randn(
        total_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    with pytest.raises(ValueError, match="v_data_type"):
        wrapper.run(q, (k_cache, v_cache_wrong))


def test_asymmetric_tensor_cores_raises():
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True
    )
    with pytest.raises(NotImplementedError, match=r"[Aa]symmetric"):
        _plan_asym(wrapper)


def test_single_decode_asymmetric_raises():
    # The public single-decode API keys its JIT module on k.dtype only;
    # mixed-dtype V must be rejected, not reinterpreted.
    q = torch.randn(4, 128, device="cuda:0", dtype=torch.float16)
    k = torch.randn(64, 4, 128, device="cuda:0", dtype=torch.float16)
    v = torch.randn(64, 4, 128, device="cuda:0", dtype=torch.float16).to(
        torch.float8_e4m3fn
    )
    with pytest.raises(NotImplementedError, match=r"[Aa]symmetric"):
        flashinfer.single_decode_with_kv_cache(q, k, v)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
