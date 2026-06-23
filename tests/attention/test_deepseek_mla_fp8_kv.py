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

Tests for the FP8 KV cache path on the Hopper SM90 MLA (FA3) backend
(issue flashinfer-ai/flashinfer#2144). The path stores KV as FP8 e4m3 in
shared memory, dequantizes one tile at a time to BF16 inside the kernel
right before the WGMMA, and runs the existing BF16xBF16 GMMA unchanged.

Numerical reference: dequant the same FP8 KV tensors in Python (to the
same BF16 layout the kernel uses), run the standard BF16 MLA kernel on
that, and compare to the FP8-KV kernel output. The remaining differences
are limited to BF16 accumulation noise.
"""

import math

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported


HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


def _per_tensor_symmetric_quant_fp8(
    x: torch.Tensor, fp8_max: float = 448.0
) -> tuple[torch.Tensor, float]:
    """Per-tensor symmetric quantize FP32 -> FP8 E4M3, returning the FP8 tensor
    and the scale (real = quantized * scale)."""
    amax = x.abs().max().item()
    scale = amax / fp8_max if amax > 0 else 1.0
    q = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale


def _ref_dequant_to_bf16(fp8: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference dequant matching the in-kernel numerics: cast FP8 -> BF16
    directly, then multiply by a BF16-precision scale via __hmul2 semantics.
    """
    scale_bf16 = torch.tensor(scale, dtype=torch.bfloat16).to(fp8.device)
    return (fp8.to(torch.bfloat16) * scale_bf16).to(torch.bfloat16)


def _run_mla(
    backend: str,
    q_nope: torch.Tensor,
    q_pe: torch.Tensor,
    ckv: torch.Tensor,
    kpe: torch.Tensor,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    num_heads: int,
    page_size: int,
    causal: bool,
    sm_scale: float,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    ckv_scale: float | None = None,
    kpe_scale: float | None = None,
) -> torch.Tensor:
    device = q_nope.device
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=backend)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_data_type=q_dtype,
        kv_data_type=kv_dtype,
    )
    kwargs = {}
    if ckv_scale is not None:
        kwargs["ckv_scale"] = ckv_scale
    if kpe_scale is not None:
        kwargs["kpe_scale"] = kpe_scale
    return wrapper.run(q_nope, q_pe, ckv, kpe, **kwargs)


@pytest.fixture(autouse=True)
def _skip_if_no_sm90a():
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")


@pytest.mark.parametrize("batch_size", [1, 4, 16])
@pytest.mark.parametrize("kv_len", [256, 1024, 4096])
@pytest.mark.parametrize("qo_len", [1, 16])
@pytest.mark.parametrize("page_size", [16, 64])
@pytest.mark.parametrize("num_heads", [16, 128])
@pytest.mark.parametrize("causal", [False, True])
def test_batch_mla_fp8_kv_matches_bf16_reference(
    batch_size, kv_len, qo_len, page_size, num_heads, causal
):
    """For random FP8 KV with per-tensor scales, the FP8-KV kernel output
    must match the BF16 reference (run on the BF16-dequant of the same FP8
    data) within BF16 precision."""
    if causal and qo_len > kv_len:
        pytest.skip("invalid causal config (qo_len > kv_len)")
    if kv_len % page_size != 0:
        pytest.skip("kv_len must be divisible by page_size")

    torch.manual_seed(0xCAFE)
    device = torch.device("cuda:0")

    q_nope = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_CKV,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    q_pe = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_KPE,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )

    num_pages = (batch_size * kv_len + page_size - 1) // page_size
    ckv_fp32 = torch.randn(num_pages, page_size, HEAD_DIM_CKV, device=device) * 0.1
    kpe_fp32 = torch.randn(num_pages, page_size, HEAD_DIM_KPE, device=device) * 0.1
    ckv_fp8, ckv_scale = _per_tensor_symmetric_quant_fp8(ckv_fp32)
    kpe_fp8, kpe_scale = _per_tensor_symmetric_quant_fp8(kpe_fp32)

    ckv_ref = _ref_dequant_to_bf16(ckv_fp8, ckv_scale)
    kpe_ref = _ref_dequant_to_bf16(kpe_fp8, kpe_scale)

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * (
        kv_len // page_size
    )
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    o_bf16 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_ref,
        kpe_ref,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        page_size=page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )

    o_fp8 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_fp8,
        kpe_fp8,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        page_size=page_size,
        causal=causal,
        sm_scale=sm_scale,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        ckv_scale=ckv_scale,
        kpe_scale=kpe_scale,
    )

    torch.cuda.synchronize()
    assert o_bf16.shape == o_fp8.shape
    diff = (o_fp8.float() - o_bf16.float()).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    o_scale = o_bf16.abs().max().item() + 1e-6
    rel = max_diff / o_scale

    # BF16 accumulation noise + per-tensor scale * exact fp8 -> bf16 cast
    # is well under 1% relative; 1.5e-2 absolute is a comfortable bound for
    # the random-data configs in this matrix.
    assert max_diff < 1.5e-2, (
        f"max_abs_diff={max_diff:.4e} mean={mean_diff:.4e} rel={rel:.4e} "
        f"o_bf16.norm={o_bf16.norm().item():.4f} o_fp8.norm={o_fp8.norm().item():.4f}"
    )


@pytest.mark.parametrize("ckv_magnitude", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("kpe_magnitude", [0.01, 0.1, 1.0])
def test_batch_mla_fp8_kv_scale_sensitivity(ckv_magnitude, kpe_magnitude):
    """The kernel must correctly apply per-tensor scales across orders of
    magnitude. We control the underlying data range, derive a realistic
    scale (max_abs / 448) per tensor, and verify both paths match.

    Q is normalized to keep softmax inputs bounded across the data range
    sweep; the kernel correctness (BF16 == FP8) is independent of softmax
    magnitudes."""
    torch.manual_seed(7)
    device = torch.device("cuda:0")
    batch_size, qo_len, kv_len = 2, 1, 256
    num_heads = 64
    page_size = 64

    # Bound QK so softmax stays in a numerically stable range regardless of
    # the KV magnitude sweep. Attention dot product magnitude scales as
    # ||q||_inf * ||k||_inf * HEAD_DIM_QK, so we scale Q down with the KV.
    max_kv_mag = max(ckv_magnitude, kpe_magnitude)
    q_scale = 0.1 / max_kv_mag
    q_nope = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_CKV,
            device=device,
            dtype=torch.bfloat16,
        )
        * q_scale
    )
    q_pe = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_KPE,
            device=device,
            dtype=torch.bfloat16,
        )
        * q_scale
    )

    num_pages = (batch_size * kv_len + page_size - 1) // page_size
    ckv_fp32 = (
        torch.randn(num_pages, page_size, HEAD_DIM_CKV, device=device) * ckv_magnitude
    )
    kpe_fp32 = (
        torch.randn(num_pages, page_size, HEAD_DIM_KPE, device=device) * kpe_magnitude
    )
    ckv_fp8, ckv_scale = _per_tensor_symmetric_quant_fp8(ckv_fp32)
    kpe_fp8, kpe_scale = _per_tensor_symmetric_quant_fp8(kpe_fp32)

    ckv_ref = _ref_dequant_to_bf16(ckv_fp8, ckv_scale)
    kpe_ref = _ref_dequant_to_bf16(kpe_fp8, kpe_scale)

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * (
        kv_len // page_size
    )
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    o_bf16 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_ref,
        kpe_ref,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    o_fp8 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_fp8,
        kpe_fp8,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        ckv_scale=ckv_scale,
        kpe_scale=kpe_scale,
    )

    torch.cuda.synchronize()
    diff = (o_fp8.float() - o_bf16.float()).abs()
    o_scale_val = o_bf16.abs().max().item() + 1e-6
    rel = diff.max().item() / o_scale_val
    # Bigger absolute tolerance than the realistic-magnitude matrix because
    # the SW FP8->BF16 dequant (fast_dequant_f8f16x4: bit-manip + bias
    # multiply) has slightly more drift than Python's hardware-backed
    # tensor.to(bf16) at large FP8 magnitudes, and that drift propagates
    # through the K=576 WGMMA accumulation. The bound here still proves
    # the scale is applied correctly.
    assert diff.max().item() < 5e-2, (
        f"ckv_mag={ckv_magnitude} kpe_mag={kpe_magnitude} "
        f"(ckv_scale={ckv_scale:.6f} kpe_scale={kpe_scale:.6f}): "
        f"max={diff.max().item():.4e} rel={rel:.4e} "
        f"o_bf16.norm={o_bf16.norm().item():.4f}"
    )


def test_batch_mla_fp8_kv_zero_kv_gives_zero_output():
    """All-zero FP8 KV must produce all-zero attention output. Catches any
    BF16-staging buffer overflow from load_kv writing past its intended
    region (an earlier bug fixed by the dtype-aware inner-loop bounds)."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    batch_size, qo_len, kv_len = 2, 1, 256
    num_heads = 128
    page_size = 64

    q_nope = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_CKV,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )
    q_pe = (
        torch.randn(
            batch_size * qo_len,
            num_heads,
            HEAD_DIM_KPE,
            device=device,
            dtype=torch.bfloat16,
        )
        * 0.1
    )

    num_pages = (batch_size * kv_len + page_size - 1) // page_size
    ckv_fp8 = torch.zeros(
        num_pages, page_size, HEAD_DIM_CKV, device=device, dtype=torch.float8_e4m3fn
    )
    kpe_fp8 = torch.zeros(
        num_pages, page_size, HEAD_DIM_KPE, device=device, dtype=torch.float8_e4m3fn
    )

    qo_indptr = (
        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * qo_len
    )
    kv_indptr = torch.arange(0, batch_size + 1, dtype=torch.int32, device=device) * (
        kv_len // page_size
    )
    kv_len_arr = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, num_pages, dtype=torch.int32, device=device)

    sm_scale = 1.0 / math.sqrt(HEAD_DIM_CKV + HEAD_DIM_KPE)

    o = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_fp8,
        kpe_fp8,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        page_size=page_size,
        causal=False,
        sm_scale=sm_scale,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        ckv_scale=1.0,
        kpe_scale=1.0,
    )
    torch.cuda.synchronize()
    assert o.abs().max().item() == 0.0, f"non-zero output: {o.abs().max().item()}"


def test_fp8_kv_kpe_dominant_no_row_aliasing():
    """Deterministic regression for the FP8 KPE shmem swizzle aliasing bug.

    With HEAD_DIM_KPE=64 on the FP8 path, the raw KPE buffer has 4 b128
    cols per row. The k128B swizzle (N=8) used elsewhere makes rows K and
    K+4 collide at the same shared-memory offset, so random-data tests
    can pass while specific KPE-dominant attention masks corrupt silently.

    Here token 4 of KPE is all +1 and token 8 is all -1, with the
    corresponding rows of CKV (== V on the MLA path) set to a large
    distinctive value. The softmax with q_pe = ones picks the token whose
    KPE matches Q sign-wise; the resulting output's first dim must be
    +100 for the BF16 baseline. If the FP8 path silently aliases KPE
    row 4 with row 8, the output flips toward 0 or -100.
    """
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    B, ql, kl, H = 1, 1, 64, 16
    P = 64

    q_nope = torch.zeros(B * ql, H, HEAD_DIM_CKV, device=device, dtype=torch.bfloat16)
    q_pe = torch.ones(B * ql, H, HEAD_DIM_KPE, device=device, dtype=torch.bfloat16)

    nps = (B * kl + P - 1) // P
    ckv_fp32 = torch.zeros(nps, P, HEAD_DIM_CKV, device=device)
    kpe_fp32 = torch.zeros(nps, P, HEAD_DIM_KPE, device=device)
    kpe_fp32[0, 4, :] = 1.0
    kpe_fp32[0, 8, :] = -1.0
    ckv_fp32[0, 4, 0] = 100.0
    ckv_fp32[0, 8, 0] = -100.0

    ckv_fp8, ckv_scale = _per_tensor_symmetric_quant_fp8(ckv_fp32)
    kpe_fp8, kpe_scale = _per_tensor_symmetric_quant_fp8(kpe_fp32)
    ckv_ref = _ref_dequant_to_bf16(ckv_fp8, ckv_scale)
    kpe_ref = _ref_dequant_to_bf16(kpe_fp8, kpe_scale)

    qo_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * ql
    kv_indptr = torch.arange(0, B + 1, dtype=torch.int32, device=device) * (kl // P)
    kv_len_arr = torch.full((B,), kl, dtype=torch.int32, device=device)
    kv_indices = torch.arange(0, nps, dtype=torch.int32, device=device)

    o_bf16 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_ref,
        kpe_ref,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=H,
        page_size=P,
        causal=False,
        sm_scale=1.0,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
    )
    o_fp8 = _run_mla(
        "fa3",
        q_nope,
        q_pe,
        ckv_fp8,
        kpe_fp8,
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=H,
        page_size=P,
        causal=False,
        sm_scale=1.0,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.float8_e4m3fn,
        ckv_scale=ckv_scale,
        kpe_scale=kpe_scale,
    )
    torch.cuda.synchronize()

    # If FP8 KPE rows 4 and 8 alias, token 4's data is corrupted and the
    # softmax-weighted V at dim 0 drops to ~0 or flips sign.
    assert o_bf16[0, 0, 0].item() > 50.0, (
        f"sanity check failed on BF16 baseline: o_bf16[0,0,0]={o_bf16[0, 0, 0].item()}"
    )
    diff = (o_fp8.float() - o_bf16.float()).abs()
    assert diff.max().item() < 1e-3, (
        f"FP8 KPE row aliasing regression: "
        f"o_bf16[0,0,:4]={o_bf16[0, 0, :4].tolist()} "
        f"o_fp8[0,0,:4]={o_fp8[0, 0, :4].tolist()} "
        f"max_diff={diff.max().item()}"
    )


@pytest.mark.parametrize(
    "backend,kv_dtype,exc_match",
    [
        # FP8 e4m3 with the wrong backend must fail with a backend-specific
        # message before generic dtype-allowlist rejection (e4m3 IS in the
        # supported set, so the backend check fires).
        ("fa2", torch.float8_e4m3fn, "only supported with the fa3 backend"),
        # Other 1-byte / non-allowlisted KV dtypes hit the allowlist guard.
        ("fa3", torch.float8_e5m2, "is not supported"),
        ("fa3", torch.uint8, "is not supported"),
        ("fa3", torch.int8, "is not supported"),
        ("fa3", torch.float32, "is not supported"),
    ],
)
def test_fp8_kv_plan_rejects_unsupported_kv_dtype(backend, kv_dtype, exc_match):
    """Unsupported backend / KV dtype combinations must fail at plan() time
    with a clear ValueError, before any JIT compilation kicks in.

    Defense in depth: 1-byte dtypes (uint8, fp4, e5m2) would otherwise be
    JIT-mapped to non-FP8 element types and silently take an unsupported
    code path inside the kernel."""
    device = torch.device("cuda:0")
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=backend)
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([64], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match=exc_match):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads=16,
            head_dim_ckv=HEAD_DIM_CKV,
            head_dim_kpe=HEAD_DIM_KPE,
            page_size=64,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.bfloat16,
            kv_data_type=kv_dtype,
        )


@pytest.mark.parametrize(
    "head_dim_ckv,head_dim_kpe",
    [
        (256, 64),  # non-DeepSeek CKV
        (512, 32),  # non-DeepSeek KPE -- would also trigger KPE row aliasing
        (1024, 128),  # both non-DeepSeek
    ],
)
def test_fp8_kv_plan_rejects_unsupported_head_dim(head_dim_ckv, head_dim_kpe):
    """The FP8 KV layout / dequant helpers are only verified for the DeepSeek
    MLA dimensions (HEAD_DIM_CKV=512, HEAD_DIM_KPE=64). Other dimensions
    must be rejected at plan() time; otherwise the smem swizzle / dequant
    routine can silently produce wrong output."""
    device = torch.device("cuda:0")
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([64], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="head_dim_ckv=512 and head_dim_kpe=64"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads=16,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=64,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.float8_e4m3fn,
        )


def test_fp8_kv_plan_rejects_fp16_q():
    """FP8 KV path is BF16-Q only in this PR; FP16 Q must be rejected at
    plan() time so users get a clear error instead of an untested path."""
    device = torch.device("cuda:0")
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([64], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="q_data_type=torch.bfloat16"):
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads=16,
            head_dim_ckv=HEAD_DIM_CKV,
            head_dim_kpe=HEAD_DIM_KPE,
            page_size=64,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.float16,
            kv_data_type=torch.float8_e4m3fn,
        )


def test_fp8_kv_scales_are_keyword_only():
    """ckv_scale / kpe_scale must be passed by keyword. Static positional
    passing is rejected by the signature (`*` marker before the scales).
    This pins down the documented API contract from the PR summary."""
    import inspect

    sig = inspect.signature(flashinfer.mla.BatchMLAPagedAttentionWrapper.run)
    params = sig.parameters
    assert params["ckv_scale"].kind == inspect.Parameter.KEYWORD_ONLY, (
        f"ckv_scale kind={params['ckv_scale'].kind}, expected KEYWORD_ONLY"
    )
    assert params["kpe_scale"].kind == inspect.Parameter.KEYWORD_ONLY, (
        f"kpe_scale kind={params['kpe_scale'].kind}, expected KEYWORD_ONLY"
    )


@pytest.mark.parametrize(
    "wrong_tensor,wrong_dtype,exc_match",
    [
        # Plan BF16-Q + FP8-KV, then pass FP16 Q tensors at run() -- the C++
        # launcher would otherwise reinterpret the FP16 storage as BF16.
        ("q_nope", torch.float16, "q_nope.dtype"),
        ("q_pe", torch.float16, "q_pe.dtype"),
        # Plan FP8 KV, then pass BF16 KV tensors -- C++ launcher would read
        # BF16 storage as FP8.
        ("ckv_cache", torch.bfloat16, "ckv_cache.dtype"),
        ("kpe_cache", torch.bfloat16, "kpe_cache.dtype"),
    ],
)
def test_fp8_kv_run_rejects_dtype_mismatch(wrong_tensor, wrong_dtype, exc_match):
    """The C++ launcher casts tensor pointers directly to the JIT-compiled
    template type. If the caller plans BF16-Q + FP8-KV but then passes a
    tensor with a different dtype to run(), the kernel reads the storage
    as if it had the planned dtype and produces silent wrong output.

    run() must reject any tensor whose dtype doesn't match the planned dtype.
    """
    device = torch.device("cuda:0")
    batch_size, qo_len, kv_len = 1, 1, 64
    page_size = 64
    num_heads = 16

    q_nope = torch.zeros(
        batch_size * qo_len,
        num_heads,
        HEAD_DIM_CKV,
        device=device,
        dtype=torch.bfloat16,
    )
    q_pe = torch.zeros(
        batch_size * qo_len,
        num_heads,
        HEAD_DIM_KPE,
        device=device,
        dtype=torch.bfloat16,
    )
    ckv_cache = torch.zeros(
        1,
        page_size,
        HEAD_DIM_CKV,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    kpe_cache = torch.zeros(
        1,
        page_size,
        HEAD_DIM_KPE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )

    # Replace the chosen tensor with a wrong-dtype variant.
    tensors = {
        "q_nope": q_nope,
        "q_pe": q_pe,
        "ckv_cache": ckv_cache,
        "kpe_cache": kpe_cache,
    }
    orig = tensors[wrong_tensor]
    tensors[wrong_tensor] = torch.zeros(orig.shape, dtype=wrong_dtype, device=device)

    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([kv_len], dtype=torch.int32, device=device)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=page_size,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
    )
    with pytest.raises(ValueError, match=exc_match):
        wrapper.run(
            tensors["q_nope"],
            tensors["q_pe"],
            tensors["ckv_cache"],
            tensors["kpe_cache"],
            ckv_scale=1.0,
            kpe_scale=1.0,
        )


def test_fp8_kv_requires_scales():
    """Forgetting to pass ckv_scale / kpe_scale on the FP8 path should raise
    a clear error rather than silently producing wrong output."""
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    batch_size, kv_len = 1, 64
    page_size = 64
    num_heads = 16

    q_nope = torch.zeros(
        batch_size,
        num_heads,
        HEAD_DIM_CKV,
        device=device,
        dtype=torch.bfloat16,
    )
    q_pe = torch.zeros(
        batch_size,
        num_heads,
        HEAD_DIM_KPE,
        device=device,
        dtype=torch.bfloat16,
    )
    ckv_fp8 = torch.zeros(
        1,
        page_size,
        HEAD_DIM_CKV,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    kpe_fp8 = torch.zeros(
        1,
        page_size,
        HEAD_DIM_KPE,
        device=device,
        dtype=torch.float8_e4m3fn,
    )
    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=device)
    kv_len_arr = torch.tensor([kv_len], dtype=torch.int32, device=device)
    workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend="fa3")
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_len_arr,
        num_heads=num_heads,
        head_dim_ckv=HEAD_DIM_CKV,
        head_dim_kpe=HEAD_DIM_KPE,
        page_size=page_size,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float8_e4m3fn,
    )
    with pytest.raises(ValueError, match="ckv_scale and kpe_scale are required"):
        wrapper.run(q_nope, q_pe, ckv_fp8, kpe_fp8)
