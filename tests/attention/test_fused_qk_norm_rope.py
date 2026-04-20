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

Tests for :func:`flashinfer.fused_qk_norm_rope`, the flashinfer port of
TensorRT-LLM's ``trtllm::fused_qk_norm_rope`` torch op.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import get_compute_capability


# ---------------------------------------------------------------------------
# Reference implementation
# ---------------------------------------------------------------------------


def _yarn_inv_freq(
    rotary_dim: int,
    rope_theta: float,
    yarn_factor: float,
    yarn_low: float,
    yarn_high: float,
    device,
) -> torch.Tensor:
    """Torch reference for the YaRN ramp used inside the kernel.

    Mirrors ``compute_freq_yarn`` in ``include/flashinfer/fused_qk_norm_rope.cuh``
    but vectorized over ``half_dim``.
    """
    half_dim = torch.arange(rotary_dim // 2, dtype=torch.float32, device=device)
    freq = rope_theta ** (-2.0 * half_dim / rotary_dim)
    if yarn_factor != 1.0:
        inv_freq_extrapolation = freq
        inv_freq_interpolation = freq / yarn_factor
        # Match the kernel's guard against low == high.
        if abs(yarn_low - yarn_high) <= 1e-6:
            high_adj = yarn_high + 0.001
        else:
            high_adj = yarn_high
        linear = (half_dim - yarn_low) / (high_adj - yarn_low)
        ramp = torch.clamp(linear, 0.0, 1.0)
        inv_freq_extrapolation_factor = 1.0 - ramp
        freq = (
            inv_freq_interpolation * (1.0 - inv_freq_extrapolation_factor)
            + inv_freq_extrapolation * inv_freq_extrapolation_factor
        )
    return freq


def ref_fused_qk_norm_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    pos_ids: torch.Tensor,
    rotary_dim: int,
    eps: float,
    rope_theta: float,
    interleave: bool,
    is_qk_norm: bool,
    yarn_factor: float,
    yarn_low: float,
    yarn_high: float,
    yarn_attention_factor: float,
):
    """Reference implementation for the fused RMSNorm + RoPE op.

    All math runs in float32 to keep the reference tight; we downcast the final
    result back to ``q.dtype``. Only the RoPE region is multiplied by
    ``yarn_attention_factor`` — the kernel leaves the pass-through tail
    unchanged, and so do we.
    """
    orig_dtype = q.dtype
    q = q.to(torch.float32)
    k = k.to(torch.float32)

    if is_qk_norm:
        rms_q = torch.rsqrt(q.pow(2).mean(dim=-1, keepdim=True) + eps)
        q = q * rms_q * q_weight.to(torch.float32)
        rms_k = torch.rsqrt(k.pow(2).mean(dim=-1, keepdim=True) + eps)
        k = k * rms_k * k_weight.to(torch.float32)

    inv_freq = _yarn_inv_freq(
        rotary_dim, rope_theta, yarn_factor, yarn_low, yarn_high, pos_ids.device
    )
    theta = pos_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = theta.cos()  # [nnz, rotary_dim/2]
    sin = theta.sin()

    def _rotate(x: torch.Tensor) -> torch.Tensor:
        x_rot = x[..., :rotary_dim]
        x_pass = x[..., rotary_dim:]
        cos_b = cos.unsqueeze(1)  # [nnz, 1, rotary_dim/2]
        sin_b = sin.unsqueeze(1)
        if interleave:
            x_even = x_rot[..., 0::2]
            x_odd = x_rot[..., 1::2]
            new_even = x_even * cos_b - x_odd * sin_b
            new_odd = x_even * sin_b + x_odd * cos_b
            x_rot_new = torch.stack([new_even, new_odd], dim=-1).flatten(-2)
        else:
            x1 = x_rot[..., : rotary_dim // 2]
            x2 = x_rot[..., rotary_dim // 2 :]
            new1 = x1 * cos_b - x2 * sin_b
            new2 = x1 * sin_b + x2 * cos_b
            x_rot_new = torch.cat([new1, new2], dim=-1)
        x_rot_new = x_rot_new * yarn_attention_factor
        return torch.cat([x_rot_new, x_pass], dim=-1)

    q_out = _rotate(q).to(orig_dtype)
    k_out = _rotate(k).to(orig_dtype)
    return q_out, k_out


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------


def _skip_if_unsupported():
    """BF16 native arithmetic, warp shuffles, and tensor-core-sized loads are
    all available on SM80+, which is what the kernel targets."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    device = torch.device("cuda:0")
    cc = get_compute_capability(device)
    if cc[0] < 8:
        pytest.skip(f"fused_qk_norm_rope requires SM80+ (got SM{cc[0]}{cc[1]})")


def _tolerances(dtype: torch.dtype):
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 5e-3, 5e-3
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128, 256])
@pytest.mark.parametrize(
    "num_q_heads, num_kv_heads", [(8, 8), (32, 8), (16, 4), (4, 4)]
)
@pytest.mark.parametrize("nnz", [1, 17, 1024])
@pytest.mark.parametrize("partial_rotary_factor", [1.0, 0.5])
@pytest.mark.parametrize("interleave", [True, False])
@pytest.mark.parametrize("is_qk_norm", [True, False])
def test_fused_qk_norm_rope_correctness(
    dtype,
    head_dim,
    num_q_heads,
    num_kv_heads,
    nnz,
    partial_rotary_factor,
    interleave,
    is_qk_norm,
):
    _skip_if_unsupported()

    torch.manual_seed(0)
    device = "cuda:0"
    rotary_dim = int(head_dim * partial_rotary_factor)
    if rotary_dim % 2 != 0:
        pytest.skip("rotary_dim must be even")

    # Neox partial-rope requires `(rotary_dim / 2) / num_elems_per_thread` to be a
    # power of 2. For the head_dims we dispatch over, partial_rotary_factor=0.5
    # always yields pair_offset = 8, 16, 32 which is fine.

    q = torch.randn(nnz, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    pos_ids = torch.randint(0, 4096, (nnz,), dtype=torch.int32, device=device)
    eps = 1e-6
    rope_theta = 1e4

    q_ref_out, k_ref_out = ref_fused_qk_norm_rope(
        q.clone(),
        k.clone(),
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim,
        eps,
        rope_theta,
        interleave,
        is_qk_norm,
        1.0,
        0.0,
        0.0,
        1.0,
    )

    flashinfer.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim=rotary_dim,
        eps=eps,
        rope_theta=rope_theta,
        interleave=interleave,
        is_qk_norm=is_qk_norm,
    )

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(q, q_ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(k, k_ref_out, rtol=rtol, atol=atol)


def _yarn_tolerances(dtype: torch.dtype):
    # YaRN uses rope_theta=5e5 with pos_ids up to 32768, so the kernel's
    # __sincosf drifts a bit further from torch's IEEE sin/cos than it does
    # at rope_theta=1e4. Combined with fp16 rounding plus the
    # yarn_attention_factor scaling, a handful of elements land just past
    # the tight per-dtype tolerance used elsewhere.
    if dtype == torch.bfloat16:
        return 2e-2, 2e-2
    if dtype == torch.float16:
        return 1e-2, 1e-2
    raise ValueError(f"Unsupported dtype {dtype}")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("interleave", [True, False])
def test_fused_qk_norm_rope_yarn(dtype, head_dim, interleave):
    """Correctness with YaRN scaling enabled (factor != 1.0)."""
    _skip_if_unsupported()

    torch.manual_seed(1)
    device = "cuda:0"
    nnz = 37
    num_q_heads, num_kv_heads = 8, 4
    rotary_dim = head_dim  # full rope

    q = torch.randn(nnz, num_q_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    pos_ids = torch.randint(0, 32768, (nnz,), dtype=torch.int32, device=device)

    yarn_factor = 4.0
    yarn_low = float(head_dim // 8)
    yarn_high = float(head_dim // 4)
    yarn_attention_factor = 0.9

    q_ref_out, k_ref_out = ref_fused_qk_norm_rope(
        q.clone(),
        k.clone(),
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim,
        1e-6,
        5e5,  # larger base for long-context models
        interleave,
        True,
        yarn_factor,
        yarn_low,
        yarn_high,
        yarn_attention_factor,
    )

    flashinfer.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim=rotary_dim,
        eps=1e-6,
        rope_theta=5e5,
        interleave=interleave,
        is_qk_norm=True,
        yarn_factor=yarn_factor,
        yarn_low=yarn_low,
        yarn_high=yarn_high,
        yarn_attention_factor=yarn_attention_factor,
    )

    rtol, atol = _yarn_tolerances(dtype)
    torch.testing.assert_close(q, q_ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(k, k_ref_out, rtol=rtol, atol=atol)


def test_fused_qk_norm_rope_noncontiguous():
    """Last dim contiguous but outer strides non-standard (e.g., q and k
    slices from a packed qkv buffer)."""
    _skip_if_unsupported()

    torch.manual_seed(2)
    device = "cuda:0"
    dtype = torch.bfloat16
    nnz, head_dim = 23, 128
    num_q_heads, num_kv_heads, num_v_heads = 8, 2, 2

    total_heads = num_q_heads + num_kv_heads + num_v_heads
    qkv = torch.randn(nnz, total_heads, head_dim, dtype=dtype, device=device)
    q = qkv[:, :num_q_heads, :]
    k = qkv[:, num_q_heads : num_q_heads + num_kv_heads, :]

    # These are views, not contiguous, but have contiguous last dim.
    assert not q.is_contiguous()
    assert not k.is_contiguous()
    assert q.stride(-1) == 1 and k.stride(-1) == 1

    q_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    k_weight = torch.randn(head_dim, dtype=dtype, device=device) * 0.1 + 1.0
    pos_ids = torch.arange(nnz, dtype=torch.int32, device=device)

    # Reference on a contiguous clone.
    q_ref, k_ref = q.contiguous(), k.contiguous()
    q_ref_out, k_ref_out = ref_fused_qk_norm_rope(
        q_ref,
        k_ref,
        q_weight,
        k_weight,
        pos_ids,
        head_dim,
        1e-6,
        1e4,
        False,
        True,
        1.0,
        0.0,
        0.0,
        1.0,
    )

    flashinfer.fused_qk_norm_rope(q, k, q_weight, k_weight, pos_ids)

    rtol, atol = _tolerances(dtype)
    torch.testing.assert_close(q.contiguous(), q_ref_out, rtol=rtol, atol=atol)
    torch.testing.assert_close(k.contiguous(), k_ref_out, rtol=rtol, atol=atol)


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def _make_inputs(head_dim: int = 128, dtype: torch.dtype = torch.bfloat16):
    device = "cuda:0"
    nnz = 4
    q = torch.randn(nnz, 4, head_dim, dtype=dtype, device=device)
    k = torch.randn(nnz, 2, head_dim, dtype=dtype, device=device)
    q_weight = torch.ones(head_dim, dtype=dtype, device=device)
    k_weight = torch.ones(head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(nnz, dtype=torch.int32, device=device)
    return q, k, q_weight, k_weight, pos_ids


# ``TVM_FFI_ICHECK`` raises ``InternalError`` which surfaces as Python
# ``RuntimeError`` via ``tvm_ffi.core.ERROR_NAME_TO_TYPE`` (the fallback for
# unregistered kinds).
_KERNEL_VALIDATION_EXC = RuntimeError


def test_fused_qk_norm_rope_invalid_head_dim():
    _skip_if_unsupported()
    q, k, q_weight, k_weight, pos_ids = _make_inputs(head_dim=96)
    with pytest.raises(_KERNEL_VALIDATION_EXC):
        flashinfer.fused_qk_norm_rope(q, k, q_weight, k_weight, pos_ids)


def test_fused_qk_norm_rope_pos_ids_wrong_dtype():
    _skip_if_unsupported()
    q, k, q_weight, k_weight, pos_ids = _make_inputs()
    pos_ids = pos_ids.to(torch.int64)
    with pytest.raises(_KERNEL_VALIDATION_EXC):
        flashinfer.fused_qk_norm_rope(q, k, q_weight, k_weight, pos_ids)


def test_fused_qk_norm_rope_dtype_mismatch():
    _skip_if_unsupported()
    q, k, q_weight, k_weight, pos_ids = _make_inputs(dtype=torch.bfloat16)
    k_weight = k_weight.to(torch.float16)  # intentionally mismatched
    with pytest.raises(_KERNEL_VALIDATION_EXC):
        flashinfer.fused_qk_norm_rope(q, k, q_weight, k_weight, pos_ids)


def test_fused_qk_norm_rope_neox_non_power_of_2_partial():
    """Neox (interleave=False) with partial rope where pair_offset is not a
    power of 2 should be rejected. For head_dim=128 / numElemsPerThread=4,
    rotary_dim=24 gives pair_offset=3 which is not a power of 2."""
    _skip_if_unsupported()
    q, k, q_weight, k_weight, pos_ids = _make_inputs(head_dim=128)
    with pytest.raises(_KERNEL_VALIDATION_EXC):
        flashinfer.fused_qk_norm_rope(
            q,
            k,
            q_weight,
            k_weight,
            pos_ids,
            rotary_dim=24,
            interleave=False,
        )


def test_fused_qk_norm_rope_interleave_accepts_partial():
    """Interleave=True has no power-of-2 constraint on partial rope; any even
    rotary_dim <= head_dim should work."""
    _skip_if_unsupported()
    q, k, q_weight, k_weight, pos_ids = _make_inputs(head_dim=128)
    # rotary_dim = 24 would fail in neox mode, but is allowed in interleave mode.
    flashinfer.fused_qk_norm_rope(
        q,
        k,
        q_weight,
        k_weight,
        pos_ids,
        rotary_dim=24,
        interleave=True,
    )


def test_fused_qk_norm_rope_misaligned_stride():
    """Stride along a non-trivial dimension must be a multiple of
    ``num_elems_per_thread = head_dim / 32`` so per-lane vectorized loads stay
    aligned. Construct an ``as_strided`` view with an odd outer stride and
    check we get a clear validation error rather than a misaligned-address
    crash inside the kernel."""
    _skip_if_unsupported()
    device = "cuda:0"
    dtype = torch.bfloat16
    nnz, head_dim = 2, 128  # num_elems_per_thread = 4
    num_q_heads, num_kv_heads = 4, 2

    bad_stride_n = num_q_heads * head_dim + 1  # not a multiple of 4
    storage_q = torch.randn(nnz * bad_stride_n, dtype=dtype, device=device)
    q = torch.as_strided(
        storage_q,
        size=(nnz, num_q_heads, head_dim),
        stride=(bad_stride_n, head_dim, 1),
    )
    assert q.stride(-1) == 1 and q.stride(0) % 4 != 0

    k = torch.randn(nnz, num_kv_heads, head_dim, dtype=dtype, device=device)
    q_weight = torch.ones(head_dim, dtype=dtype, device=device)
    k_weight = torch.ones(head_dim, dtype=dtype, device=device)
    pos_ids = torch.arange(nnz, dtype=torch.int32, device=device)

    with pytest.raises(_KERNEL_VALIDATION_EXC):
        flashinfer.fused_qk_norm_rope(q, k, q_weight, k_weight, pos_ids)
