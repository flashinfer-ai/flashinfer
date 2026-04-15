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

import flashinfer
from flashinfer.utils import get_compute_capability
from flashinfer.quantization.fp4_quantization import nvfp4_quantize_paged_kv_cache
from tests.utils_fp8 import to_float8


def test_mlc_failed_case():
    kv_layout = "HND"
    kv_indptr_1 = torch.tensor([0, 0, 9]).int().to(0)
    kv_indices_1 = torch.tensor([3, 4, 5, 6, 7, 8, 9, 10, 11]).int().to(0)
    kv_last_page_len_1 = torch.tensor([0, 1]).int().to(0)
    num_qo_heads = 32
    num_kv_heads = 32
    page_size = 16
    head_dim = 128
    q = torch.randn(2, num_qo_heads, head_dim).to(0).half()
    kv_data = torch.randn(12, 2, num_kv_heads, page_size, head_dim).to(0).half()

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0)
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_buffer, kv_layout)
    wrapper.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_1, lse_1 = wrapper.run_return_lse(q, kv_data)

    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr_1,
        kv_indices_1,
        kv_last_page_len_1,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode="NONE",
        data_type=torch.float16,
        q_data_type=torch.float16,
    )
    o_1_tc, lse_1_tc = wrapper_tensor_cores.run_return_lse(q, kv_data)

    print(lse_1, lse_1_tc)
    print(o_1, o_1_tc)

    torch.testing.assert_close(lse_1, lse_1_tc, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(o_1, o_1_tc, rtol=1e-3, atol=1e-3)


# ── trtllm-gen: return_lse for paged decode + prefill ───────────────
# Reference: BF16 FA2 backend (return_lse always worked).
# Covers all Q/KV/O dtype combos supported by trtllm-gen cubins.

FP8 = torch.float8_e4m3fn
_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp8": FP8}

# (q_dtype, kv_dtype, o_dtype) from trtllm-gen cubins
_DECODE_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("bf16", "fp8", "bf16"),
    ("fp16", "fp8", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "nvfp4", "fp8"),
]
_PREFILL_DTYPES = [
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "nvfp4", "fp8"),
]


def _quantize(q_bf16, k_bf16, v_bf16, q_dtype, kv_dtype):
    """Quantize Q/K/V and return (q, kv, run_kwargs)."""
    if q_dtype == "fp8":
        q, q_s = to_float8(q_bf16)
        q_s = q_s.item()
    else:
        q, q_s = q_bf16.to(_DTYPES[q_dtype]), 1.0

    if kv_dtype == "nvfp4":
        kv, sf, k_s, v_s = nvfp4_quantize_paged_kv_cache(
            k_bf16, v_bf16, kv_layout="HND"
        )
    elif kv_dtype == "fp8":
        k, k_s = to_float8(k_bf16)
        v, v_s = to_float8(v_bf16)
        kv, sf = (k, v), None
        k_s, v_s = k_s.item(), v_s.item()
    else:
        dt = _DTYPES[kv_dtype]
        kv, k_s, v_s, sf = (k_bf16.to(dt), v_bf16.to(dt)), 1.0, 1.0, None

    kw = {"q_scale": q_s, "k_scale": k_s, "v_scale": v_s}
    if sf is not None:
        kw["kv_cache_sf"] = sf
    return q, kv, kw


def _test_trtllm_return_lse(mode, q_dtype, kv_dtype, o_dtype):
    cc = get_compute_capability(torch.device("cuda"))
    if cc[0] < 10:
        pytest.skip("trtllm-gen requires SM100+")

    B, Hq, Hkv, D = 4, 32, 8, 128
    page_size = 64 if kv_dtype == "nvfp4" else 16
    kv_len, qo_len = 256, (1 if mode == "decode" else 16)
    device = "cuda"
    npps = (kv_len + page_size - 1) // page_size
    total_pages = npps * B
    # decode q: [B, Hq, D], prefill q: [B * qo_len, Hq, D]
    q_shape = (B, Hq, D) if mode == "decode" else (B * qo_len, Hq, D)
    q_bf16 = torch.randn(*q_shape, dtype=torch.bfloat16, device=device)
    k_bf16 = torch.randn(
        total_pages, Hkv, page_size, D, dtype=torch.bfloat16, device=device
    )
    v_bf16 = torch.randn(
        total_pages, Hkv, page_size, D, dtype=torch.bfloat16, device=device
    )

    kv_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * npps
    kv_indices = torch.arange(total_pages, dtype=torch.int32, device=device)
    kv_last_page_len = torch.full((B,), page_size, dtype=torch.int32, device=device)
    qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=device) * qo_len

    ws = lambda: torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    kv_plan_dtype = torch.uint8 if kv_dtype == "nvfp4" else _DTYPES[kv_dtype]

    if mode == "decode":
        # Reference: BF16 FA2
        ref = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws(), "HND", use_tensor_cores=True
        )
        ref.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            Hq,
            Hkv,
            D,
            page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        ref_o, ref_lse = ref.run(q_bf16, (k_bf16, v_bf16), return_lse=True)

        # Test: trtllm-gen
        q, kv, run_kw = _quantize(q_bf16, k_bf16, v_bf16, q_dtype, kv_dtype)
        test = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            ws(), "HND", backend="trtllm-gen"
        )
        test.plan(
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            Hq,
            Hkv,
            D,
            page_size,
            q_data_type=q.dtype,
            kv_data_type=kv_plan_dtype,
            o_data_type=_DTYPES[o_dtype],
        )
        test_o, test_lse = test.run(q, kv, return_lse=True, **run_kw)
    else:
        # Reference: BF16 FA2 causal
        ref = flashinfer.BatchPrefillWithPagedKVCacheWrapper(ws(), "HND")
        ref.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            Hq,
            Hkv,
            D,
            page_size,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
            causal=True,
        )
        ref_o, ref_lse = ref.run(q_bf16, (k_bf16, v_bf16), return_lse=True)

        # Test: trtllm-gen
        q, kv, run_kw = _quantize(q_bf16, k_bf16, v_bf16, q_dtype, kv_dtype)
        test = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            ws(), "HND", backend="trtllm-gen"
        )
        test.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            Hq,
            Hkv,
            D,
            page_size,
            q_data_type=q.dtype,
            kv_data_type=kv_plan_dtype,
            o_data_type=_DTYPES[o_dtype],
            causal=True,
        )
        test_o, test_lse = test.run(q, kv, return_lse=True, **run_kw)

    # Assert LSE relative error < 5%
    lse_rel = (ref_lse.float() - test_lse.float()).abs() / ref_lse.float().abs().clamp(
        min=1e-6
    )
    assert lse_rel.mean().item() < 0.05, (
        f"LSE mean_rel_err {lse_rel.mean().item():.4f} >= 0.05"
    )

    # Assert output cosine similarity > 0.95
    cos = torch.nn.functional.cosine_similarity(
        ref_o.float().reshape(1, -1),
        test_o.float().reshape(1, -1),
    ).item()
    assert cos > 0.95, f"Output cos_sim {cos:.4f} < 0.95"


@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    _DECODE_DTYPES,
    ids=[f"{q}_{kv}_{o}" for q, kv, o in _DECODE_DTYPES],
)
def test_trtllm_decode_return_lse(q_dtype, kv_dtype, o_dtype):
    _test_trtllm_return_lse("decode", q_dtype, kv_dtype, o_dtype)


@pytest.mark.parametrize(
    "q_dtype,kv_dtype,o_dtype",
    _PREFILL_DTYPES,
    ids=[f"{q}_{kv}_{o}" for q, kv, o in _PREFILL_DTYPES],
)
def test_trtllm_prefill_return_lse(q_dtype, kv_dtype, o_dtype):
    _test_trtllm_return_lse("prefill", q_dtype, kv_dtype, o_dtype)


if __name__ == "__main__":
    test_mlc_failed_case()
