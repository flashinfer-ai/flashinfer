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

Single prefill/decode FP8 per-token-head KV cache tests.
"""

import pytest
import torch
import flashinfer
from tests.test_helpers.rope_reference import (
    apply_rotary_pos_emb,
    generate_cos_sin_f32_cache,
)
from tests.utils_fp8 import get_cos_sim_threshold, to_float8, to_float8_per_token_head


# ============================================================
# Helpers
# ============================================================


def _cc():
    return torch.cuda.get_device_capability(0)


def _skip_if_sm_below_75():
    if _cc()[0] < 7 or (_cc()[0] == 7 and _cc()[1] < 5):
        pytest.skip("Requires SM75+")


def _skip_if_not_fp16_sm75(dtype: torch.dtype):
    if dtype != torch.float16 and _cc()[0] <= 7:
        pytest.skip(f"{dtype} skipped on SM75")


def _skip_if_sm75_head_dim_256(head_dim):
    if _cc()[0] <= 7 and head_dim > 256:
        pytest.skip("head_dim>256 exceeds SM75 smem limit")


def check_accuracy(
    o_ref: torch.Tensor, o: torch.Tensor, kv_dtype: torch.dtype, mode, label=""
):
    cos_sim = torch.nn.functional.cosine_similarity(
        o_ref.reshape(-1).float(), o.reshape(-1).float(), dim=0
    ).item()
    max_diff = (o_ref - o).abs().max().item()
    prefix = f"[{label}] " if label else ""
    threshold = get_cos_sim_threshold(kv_dtype, mode)
    print(f"{prefix}{kv_dtype} cos_sim={cos_sim:.8f} max_diff={max_diff:.8f}")
    assert cos_sim >= threshold, (
        f"{prefix}cos_sim={cos_sim:.8f} < {threshold} ({kv_dtype})"
    )
    return cos_sim, max_diff


def build_strided_cache(x, scales, head_dim):
    kv_len, num_kv_heads = x.shape[0], x.shape[1]
    stride = head_dim + 16
    buf_size = kv_len * num_kv_heads * stride
    buf = torch.zeros(buf_size, dtype=torch.uint8, device=x.device)
    rows = buf.reshape(-1, stride)
    fp8_flat = x.reshape(-1, head_dim).view(torch.uint8)
    rows[:, :head_dim].copy_(fp8_flat)
    scales_f32 = scales.reshape(-1).to(torch.float32)
    scales_bytes = scales_f32.view(torch.uint8)
    rows[:, head_dim : head_dim + 4].copy_(scales_bytes.reshape(-1, 4))
    cache = torch.as_strided(
        buf.view(x.dtype),
        (kv_len, num_kv_heads, head_dim),
        (num_kv_heads * stride, stride, 1),
        storage_offset=0,
    )
    return cache


def _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype):
    k = 0.3 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    v = 0.3 * torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
    cos, sin = generate_cos_sin_f32_cache(kv_len, head_dim, device=device)
    k = apply_rotary_pos_emb(k, k, cos, sin)[1]
    return k, v


def _build_pth_caches(k, v, kv_dtype):
    head_dim = k.shape[-1]
    k_fp8, k_scales = to_float8_per_token_head(k, kv_dtype)
    v_fp8, v_scales = to_float8_per_token_head(v, kv_dtype)
    k_cache = build_strided_cache(k_fp8, k_scales, head_dim)
    v_cache = build_strided_cache(v_fp8, v_scales, head_dim)
    return k_cache, v_cache


def _run_single_prefill_pth(q, k, v, kv_dtype, label="", **kwargs):
    k_cache, v_cache = _build_pth_caches(k, v, kv_dtype)
    o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, **kwargs)
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q, k_cache, v_cache, use_per_token_head=True, **kwargs
    )
    return *check_accuracy(o_ref, o_pth, kv_dtype, "prefill", label=label), o_ref


def _run_single_decode_pth(
    q, k, v, kv_dtype, pos_encoding_mode="NONE", label="", **kwargs
):
    k_cache, v_cache = _build_pth_caches(k, v, kv_dtype)

    if pos_encoding_mode != "NONE":
        kwargs["pos_encoding_mode"] = pos_encoding_mode
    o_ref = flashinfer.single_decode_with_kv_cache(q, k, v, **kwargs)
    o_pth = flashinfer.single_decode_with_kv_cache(
        q, k_cache, v_cache, use_per_token_head=True, **kwargs
    )
    return *check_accuracy(o_ref, o_pth, kv_dtype, "decode", label=label), o_ref


def _run_single_prefill_pth_vs_pt(q, k, v, kv_dtype, dtype, backend="fa2", **kwargs):
    label = "single prefill"
    kwargs = {"causal": True, "backend": backend, "o_dtype": dtype, **kwargs}
    cos_sim_pth, _, o_ref = _run_single_prefill_pth(
        q, k, v, kv_dtype, label=f"{label} pth", **kwargs
    )

    k_fp8_pt, k_scale_pt = to_float8(k, kv_dtype)
    v_fp8_pt, v_scale_pt = to_float8(v, kv_dtype)
    o_pt = flashinfer.single_prefill_with_kv_cache(
        q,
        k_fp8_pt,
        v_fp8_pt,
        k_scale=k_scale_pt.item(),
        v_scale=v_scale_pt.item(),
        **kwargs,
    )
    cos_sim_pt, _ = check_accuracy(
        o_ref, o_pt, kv_dtype, "prefill", label=f"{label} pt"
    )

    if cos_sim_pth < cos_sim_pt:
        pytest.xfail(
            f"[{label}] cos_sim_pth={cos_sim_pth:.8f} < cos_sim_pt={cos_sim_pt} ({kv_dtype})"
        )


def _run_single_decode_pth_vs_pt(q, k, v, kv_dtype, **kwargs):
    label = "single decode"
    cos_sim_pth, _, o_ref = _run_single_decode_pth(
        q, k, v, kv_dtype, label=f"{label} pth", **kwargs
    )

    k_fp8_pt, k_scale_pt = to_float8(k, kv_dtype)
    v_fp8_pt, v_scale_pt = to_float8(v, kv_dtype)
    o_pt = flashinfer.single_decode_with_kv_cache(
        q,
        k_fp8_pt,
        v_fp8_pt,
        k_scale=k_scale_pt.item(),
        v_scale=v_scale_pt.item(),
        **kwargs,
    )
    cos_sim_pt, _ = check_accuracy(o_ref, o_pt, kv_dtype, "decode", label=f"{label} pt")

    if cos_sim_pth < cos_sim_pt:
        pytest.xfail(
            f"[{label}] cos_sim_pth={cos_sim_pth:.8f} < cos_sim_pt={cos_sim_pt} ({kv_dtype})"
        )


# ============================================================
# Single Prefill
# ============================================================


def run_single_prefill_pth(
    qo_len,
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    kv_dtype,
    causal,
    backend,
):
    device = "cuda:0"
    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(qo_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q, q, cos, sin)[0]
    kwargs = {"causal": causal, "backend": backend, "o_dtype": dtype}
    return _run_single_prefill_pth(
        q, k, v, kv_dtype, label=f"single prefill {backend}", **kwargs
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("causal", [True, False], ids=["causal", "no-causal"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("kv_len", [16, 1024], ids=["kv16", "long_seq"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_single_prefill_pth(is_gqa, dtype, kv_dtype, causal, head_dim, kv_len, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    _skip_if_sm75_head_dim_256(head_dim)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_single_prefill_pth(
        8,
        kv_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        kv_dtype,
        causal,
        backend,
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("backend", ["fa2"], ids=["fa2"])
def test_single_prefill_pth_vs_pt(is_gqa, dtype, kv_dtype, backend):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    device = "cuda:0"
    qo_len, kv_len, head_dim = 16, 64, 128
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(qo_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q, q, cos, sin)[0]
    _run_single_prefill_pth_vs_pt(q, k, v, kv_dtype, dtype, backend=backend)


# ============================================================
# Single Decode
# ============================================================


def run_single_decode_pth(
    kv_len,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    dtype,
    kv_dtype,
    pos_encoding_mode="NONE",
):
    device = "cuda:0"
    q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(kv_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q[None], q[None], cos[:1], sin[:1])[0].squeeze(0)
    return _run_single_decode_pth(
        q,
        k,
        v,
        kv_dtype,
        pos_encoding_mode=pos_encoding_mode,
        label="single decode",
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
@pytest.mark.parametrize("head_dim", [64, 128, 256], ids=["hd64", "hd128", "hd256"])
@pytest.mark.parametrize("kv_len", [64, 1024], ids=["kv64", "long_seq"])
def test_single_decode_pth(is_gqa, dtype, kv_dtype, head_dim, kv_len):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    _skip_if_sm75_head_dim_256(head_dim)
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)
    run_single_decode_pth(kv_len, num_qo_heads, num_kv_heads, head_dim, dtype, kv_dtype)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
def test_single_decode_pth_rope_llama(dtype, kv_dtype):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    kv_len, num_qo_heads, num_kv_heads, head_dim = 64, 4, 2, 128
    run_single_decode_pth(
        kv_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        dtype,
        kv_dtype,
        pos_encoding_mode="ROPE_LLAMA",
    )


@pytest.mark.parametrize("is_gqa", [False, True], ids=["mha", "gqa"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("kv_dtype", [torch.float8_e4m3fn], ids=["e4m3"])
def test_single_decode_pth_vs_per_tensor(is_gqa, dtype, kv_dtype):
    _skip_if_sm_below_75()
    _skip_if_not_fp16_sm75(dtype)
    device = "cuda:0"
    head_dim, kv_len = 128, 128
    num_qo_heads, num_kv_heads = (4, 2) if is_gqa else (4, 4)

    q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(kv_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q[None], q[None], cos[:1], sin[:1])[0].squeeze(0)
    _run_single_decode_pth_vs_pt(q, k, v, kv_dtype)


# ============================================================
# Single Prefill/Decode with explicit non-contiguous stride
# ============================================================


def test_single_prefill_pth_non_contiguous_stride():
    """Single prefill with PTH using explicitly non-contiguous stride.

    Verifies that the kernel correctly reads inline scales from the
    head_dim+16 stride offset when the tensor is not contiguous.
    """
    _skip_if_sm_below_75()
    device = "cuda:0"
    qo_len, kv_len, head_dim = 32, 64, 64
    num_qo_heads, num_kv_heads = 4, 4
    dtype = torch.float16
    kv_dtype = torch.float8_e4m3fn

    q = torch.randn(qo_len, num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(qo_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q, q, cos, sin)[0]

    k_fp8, k_s = to_float8_per_token_head(k, kv_dtype)
    v_fp8, v_s = to_float8_per_token_head(v, kv_dtype)

    k_cache = build_strided_cache(k_fp8, k_s, head_dim)
    v_cache = build_strided_cache(v_fp8, v_s, head_dim)

    assert k_cache.stride(1) == head_dim + 16, (
        f"Expected stride {head_dim + 16}, got {k_cache.stride(1)}"
    )
    assert not k_cache.is_contiguous(), "Cache should not be contiguous"

    o_ref = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=True, backend="fa2", o_dtype=dtype
    )
    o_pth = flashinfer.single_prefill_with_kv_cache(
        q,
        k_cache,
        v_cache,
        causal=True,
        backend="fa2",
        o_dtype=dtype,
        use_per_token_head=True,
    )

    assert not torch.isnan(o_pth).any(), "PTH output contains NaN"
    check_accuracy(o_ref, o_pth, kv_dtype, "prefill", label="single non-contig stride")


def test_single_decode_pth_non_contiguous_stride():
    """Single decode with PTH using explicitly non-contiguous stride."""
    _skip_if_sm_below_75()
    device = "cuda:0"
    kv_len, head_dim = 64, 128
    num_qo_heads, num_kv_heads = 4, 4
    dtype = torch.float16
    kv_dtype = torch.float8_e4m3fn

    q = torch.randn(num_qo_heads, head_dim, dtype=dtype, device=device)
    k, v = _make_rand_kv(kv_len, num_kv_heads, head_dim, device, dtype)
    cos, sin = generate_cos_sin_f32_cache(kv_len, head_dim, device=device)
    q = apply_rotary_pos_emb(q[None], q[None], cos[:1], sin[:1])[0].squeeze(0)

    k_fp8, k_s = to_float8_per_token_head(k, kv_dtype)
    v_fp8, v_s = to_float8_per_token_head(v, kv_dtype)

    k_cache = build_strided_cache(k_fp8, k_s, head_dim)
    v_cache = build_strided_cache(v_fp8, v_s, head_dim)

    assert k_cache.stride(1) == head_dim + 16
    assert not k_cache.is_contiguous()

    o_ref = flashinfer.single_decode_with_kv_cache(q, k, v)
    o_pth = flashinfer.single_decode_with_kv_cache(
        q,
        k_cache,
        v_cache,
        use_per_token_head=True,
    )

    assert not torch.isnan(o_pth).any(), "PTH output contains NaN"
    check_accuracy(o_ref, o_pth, kv_dtype, "decode", label="single decode non-contig")


# ============================================================
# Smoke tests
# ============================================================

if __name__ == "__main__":
    dtypes = [torch.float16]
    if _cc()[0] > 7:
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        kv_dtype = torch.float8_e4m3fn

        test_single_prefill_pth(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            causal=True,
            head_dim=128,
            kv_len=16,
            backend="fa2",
        )
        print(f"single prefill MHA {dtype}/{kv_dtype} smoke passed")

        test_single_prefill_pth(
            is_gqa=True,
            dtype=dtype,
            kv_dtype=kv_dtype,
            causal=False,
            head_dim=64,
            kv_len=16,
            backend="fa2",
        )
        print(f"single prefill GQA {dtype}/{kv_dtype} smoke passed")

        test_single_decode_pth(
            is_gqa=False,
            dtype=dtype,
            kv_dtype=kv_dtype,
            head_dim=128,
            kv_len=64,
        )
        print(f"single decode MHA {dtype}/{kv_dtype} smoke passed")

        test_single_decode_pth(
            is_gqa=True,
            dtype=dtype,
            kv_dtype=kv_dtype,
            head_dim=64,
            kv_len=64,
        )
        print(f"single decode GQA {dtype}/{kv_dtype} smoke passed")

        test_single_prefill_pth_vs_pt(
            is_gqa=False, dtype=dtype, kv_dtype=kv_dtype, backend="fa2"
        )
        print(f"single prefill per-token-head vs per-tensor {dtype} smoke passed")

        test_single_decode_pth_vs_per_tensor(
            is_gqa=False, dtype=dtype, kv_dtype=kv_dtype
        )
        print(f"single decode per-token-head vs per-tensor {dtype} smoke passed")

    test_single_prefill_pth_non_contiguous_stride()
    print("single prefill non-contiguous stride OK")

    test_single_decode_pth_non_contiguous_stride()
    print("single decode non-contiguous stride OK")

    print("\nAll single per-token-head smoke tests passed")
