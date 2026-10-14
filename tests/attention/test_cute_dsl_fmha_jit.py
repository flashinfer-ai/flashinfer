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
"""Correctness tests for the JIT-compiled TRTLLM CuTe DSL FMHA kernels.

The kernels are compiled in-process via cute.compile (CuTe native ABI) and checked
against a torch reference. Covers non-block-scaled (bf16, fp8, split-BMM bf16-QK/E4M3-V,
varlen) and block-scaled MXFP8/NVFP4 (non-varlen).
"""

import math

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.typing import Float32, Int32

from flashinfer.cute_dsl.attention.fmha.fmha import (
    BlackwellFusedMultiHeadAttentionForward,
)
from flashinfer.cute_dsl.attention.fmha.fmha_blockscaled import (
    BlackwellFusedMultiHeadBlockScaledAttentionForward,
    compact_fp4_data,
    create_scale_factor_tensor,
)
from flashinfer.cute_dsl.attention.fmha.helpers import fmha_helpers as fmha_utils
from flashinfer.cute_dsl.attention.fmha.compile import _ex2_emulation_enabled

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="CuTe DSL FMHA requires Blackwell (SM100a+)",
)

DEVICE = "cuda"
MMA_TILER = (128, 128)
LOG2_E = math.log2(math.e)
ENABLE_EX2 = _ex2_emulation_enabled(torch.device(DEVICE))


def _make_cute(shape, cutlass_dtype, *, zero_out=False):
    """Build a device cute tensor of the given dtype and return (f32_ref, cute_tensor).

    f32_ref holds the exact (dequantized) stored values, extracted from the cute
    tensor via cute.testing.convert, so it matches what the kernel reads. Integer
    inputs in [-2, 2] are exactly representable in bf16/E4M3/E2M1, so the round-trip
    is lossless.
    """
    if zero_out:
        f32 = torch.zeros(*shape, dtype=torch.float32)
    else:
        f32 = torch.randint(-2, 3, shape, dtype=torch.float32)
    cute_t, torch_t = cutlass_torch.cute_tensor_like(
        f32, cutlass_dtype, is_dynamic_layout=True, assumed_align=32
    )
    compact_fp4_data(torch_t, cutlass_dtype)
    ref = f32.cuda()
    if not zero_out:
        cute.testing.convert(cute_t, from_dlpack(ref, assumed_align=32))
    return ref, cute_t


def _readback(cute_t, shape):
    """Convert a device cute tensor back to an f32 CPU tensor for comparison."""
    f32_cute, f32_torch = cutlass_torch.cute_tensor_like(
        torch.zeros(*shape, dtype=torch.float32),
        cutlass.Float32,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    cute.testing.convert(cute_t, f32_cute)
    return f32_torch.cpu()


def _ref_attention(
    q5,
    k5,
    v5,
    *,
    batch,
    s_q_max,
    s_k_max,
    scale_softmax,
    scale_output,
    causal,
    cum_q=None,
    cum_k=None,
    q_sf5=None,
    k_sf5=None,
):
    """float32 attention reference over 5D tensors (b, s, h_k, {h_r|1}, d).

    Mirrors the kernel's run_torch_fmha: per (batch, head), optionally scale q/k by
    the per-block SF, S = qk*scale_softmax, bottom-right causal mask, softmax over
    s_k, O = P@v*scale_output. Returns (o_ref_f32, lse_ref_f32) with o_ref shaped
    like q5 (dv last) and lse_ref shaped (b_out, s, h_k, h_r).
    """
    h_k, h_r, d = q5.shape[2], q5.shape[3], q5.shape[4]
    dv = v5.shape[4]
    o_ref = torch.zeros(*q5.shape[:-1], dv, dtype=torch.float32, device=q5.device)
    lse_ref = torch.zeros(*q5.shape[:-1], dtype=torch.float32, device=q5.device)
    for bi in range(batch):
        bq = 0 if cum_q is not None else bi
        bk = 0 if cum_k is not None else bi
        qo = cum_q[bi] if cum_q is not None else 0
        ko = cum_k[bi] if cum_k is not None else 0
        sq = (cum_q[bi + 1] - cum_q[bi]) if cum_q is not None else s_q_max
        sk = (cum_k[bi + 1] - cum_k[bi]) if cum_k is not None else s_k_max
        cq = q5[bq, qo : qo + sq].float()  # (sq, h_k, h_r, d)
        ck = k5[bk, ko : ko + sk].float()  # (sk, h_k, 1, d)
        cv = v5[bk, ko : ko + sk].float()  # (sk, h_k, 1, dv)
        if q_sf5 is not None:
            cq = cq * q_sf5[bi, qo : qo + sq].float()
            ck = ck * k_sf5[bk, ko : ko + sk].float()
        ck = ck.expand(sk, h_k, h_r, d)
        cv = cv.expand(sk, h_k, h_r, dv)
        s = (
            torch.einsum("qhrd,khrd->qkhr", cq, ck) * scale_softmax
        )  # (sq, sk, h_k, h_r)
        if causal:
            qc = torch.arange(sq, device=q5.device).view(-1, 1)
            kc = torch.arange(sk, device=q5.device).view(1, -1)
            masked = kc > (qc + (sk - sq))  # bottom-right aligned
            s = s.masked_fill(masked.view(sq, sk, 1, 1), float("-inf"))
        p = torch.softmax(s, dim=1)
        o = torch.einsum("qkhr,khrd->qhrd", p, cv) * scale_output
        o_ref[bq, qo : qo + sq] = o
        lse_ref[bq, qo : qo + sq] = torch.logsumexp(s, dim=1) * LOG2_E
    return o_ref, lse_ref


def _default_stream():
    return cutlass_torch.default_stream()


# =============================================================================
# Non-block-scaled: bf16, fp8 (e4m3->bf16), split-BMM (bf16 QK / e4m3 V -> bf16)
# =============================================================================

# mode -> (qk cutlass dtype, v cutlass dtype, out cutlass dtype, out torch dtype, atol)
_NONBS_MODES = {
    "bf16": (
        cutlass.BFloat16,
        cutlass.BFloat16,
        cutlass.BFloat16,
        torch.bfloat16,
        3e-2,
    ),
    # FP8 inputs + bf16 output: a few elements land just past 3e-2 (near-one-hot
    # softmax on coarse E4M3 logits), so allow a little more headroom.
    "fp8": (
        cutlass.Float8E4M3FN,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        torch.bfloat16,
        5e-2,
    ),
    "split": (
        cutlass.BFloat16,
        cutlass.Float8E4M3FN,
        cutlass.BFloat16,
        torch.bfloat16,
        5e-2,
    ),
}


@pytest.mark.parametrize("mode", list(_NONBS_MODES))
@pytest.mark.parametrize("causal", [False, True])
def test_cute_dsl_fmha_jit(mode, causal):
    """Non-block-scaled varlen prefill, JIT-compiled."""
    torch.manual_seed(0)
    qk_dt, v_dt, out_dt, out_torch_dt, atol = _NONBS_MODES[mode]
    d = dv = 128
    h_k, h_r = 4, 1
    h_q = h_k * h_r
    seq_lens_q = (64, 128, 32)
    seq_lens_k = (64, 128, 32)
    batch = len(seq_lens_q)
    s_q, s_k = sum(seq_lens_q), sum(seq_lens_k)
    max_s_q, max_s_k = max(seq_lens_q), max(seq_lens_k)

    q_ref, q_cute = _make_cute((1, s_q, h_k, h_r, d), qk_dt)
    k_ref, k_cute = _make_cute((1, s_k, h_k, 1, d), qk_dt)
    v_ref, v_cute = _make_cute((1, s_k, h_k, 1, dv), v_dt)
    _, o_cute = _make_cute((1, s_q, h_k, h_r, dv), out_dt, zero_out=True)

    def _cum(lens):
        c = [0]
        for x in lens:
            c.append(c[-1] + x)
        return c

    cum_q, cum_k = _cum(seq_lens_q), _cum(seq_lens_k)
    cum_q_cute, _ = cutlass_torch.cute_tensor_like(
        torch.tensor(cum_q, dtype=torch.int32),
        Int32,
        is_dynamic_layout=True,
        assumed_align=16,
    )
    cum_k_cute, _ = cutlass_torch.cute_tensor_like(
        torch.tensor(cum_k, dtype=torch.int32),
        Int32,
        is_dynamic_layout=True,
        assumed_align=16,
    )

    problem_size = (batch, max_s_q, s_q, max_s_k, h_q, h_k, d, dv)
    scale_softmax = 1.0 / math.sqrt(d)
    mask_type = (
        fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        if causal
        else fmha_utils.MaskEnum.RESIDUAL_MASK
    )
    ws_right = Int32(0) if causal else None

    fmha = BlackwellFusedMultiHeadAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=MMA_TILER,
        head_dim=d,
        is_persistent=False,
        mask_type=mask_type,
        enable_ex2_emulation=ENABLE_EX2,
        enable_skip_correction=True,
        use_tma_store=False,  # varlen
    )
    stream = _default_stream()
    compiled = cute.compile(
        fmha,
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        problem_size,
        cum_q_cute,
        cum_k_cute,
        None,
        None,
        scale_softmax * LOG2_E,
        scale_softmax,
        1.0,
        None,
        None,
        ws_right,
        None,
        None,
        stream,
        False,
    )
    compiled(
        q_cute,
        k_cute,
        v_cute,
        o_cute,
        problem_size,
        cum_q_cute,
        cum_k_cute,
        None,
        None,
        Float32(scale_softmax * LOG2_E),
        Float32(scale_softmax),
        Float32(1.0),
        None,
        None,
        ws_right,
        None,
        None,
        stream,
        False,
    )
    torch.cuda.synchronize()

    o = _readback(o_cute, (1, s_q, h_k, h_r, dv))
    o_ref, _ = _ref_attention(
        q_ref,
        k_ref,
        v_ref,
        batch=batch,
        s_q_max=max_s_q,
        s_k_max=max_s_k,
        scale_softmax=scale_softmax,
        scale_output=1.0,
        causal=causal,
        cum_q=cum_q,
        cum_k=cum_k,
    )
    torch.testing.assert_close(o, o_ref.cpu(), atol=atol, rtol=atol)


# =============================================================================
# Block-scaled: MXFP8 / NVFP4 (non-varlen, h_r = 1)
# =============================================================================

# qk_mode -> (qk dtype, sf dtype, sf_vec_size)
_BS_MODES = {
    "mxfp8": (cutlass.Float8E4M3FN, cutlass.Float8E8M0FNU, 32),
    "nvfp4": (cutlass.Float4E2M1FN, cutlass.Float8E4M3FN, 16),
}


def _sf_ref_5d(ref_flat, s, b, h_k, h_r, d):
    # create_scale_factor_tensor ref: (mn=s, k=d, l=b*h) -> (b, s, h_k, h_r, d)
    return (
        ref_flat.reshape(s, d, b, h_k, h_r).permute(2, 0, 3, 4, 1).contiguous().cuda()
    )


@pytest.mark.parametrize("qk_mode", list(_BS_MODES))
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("with_lse", [False, True])
def test_cute_dsl_fmha_blockscaled_jit(qk_mode, causal, with_lse):
    """Block-scaled (MXFP8 / NVFP4) non-varlen prefill, JIT-compiled."""
    torch.manual_seed(0)
    qk_dt, sf_dt, sf_vec = _BS_MODES[qk_mode]
    d = dv = 128
    b, s_q, s_k = 2, 128, 128
    h_k, h_r = 4, 1
    h_q = h_k * h_r

    q_ref, q_cute = _make_cute((b, s_q, h_k, h_r, d), qk_dt)
    k_ref, k_cute = _make_cute((b, s_k, h_k, 1, d), qk_dt)
    v_ref, v_cute = _make_cute((b, s_k, h_k, 1, dv), cutlass.Float8E4M3FN)
    _, o_cute = _make_cute((b, s_q, h_k, h_r, dv), cutlass.Float8E4M3FN, zero_out=True)

    q_sf_flat, q_sf_cute = create_scale_factor_tensor(s_q, d, b * h_q, sf_vec, sf_dt)
    k_sf_flat, k_sf_cute = create_scale_factor_tensor(s_k, d, b * h_k, sf_vec, sf_dt)
    q_sf5 = _sf_ref_5d(q_sf_flat, s_q, b, h_k, h_r, d)
    k_sf5 = _sf_ref_5d(k_sf_flat, s_k, b, h_k, 1, d)

    lse_cute = None
    if with_lse:
        _, lse_cute = _make_cute((b, s_q, h_k, h_r), cutlass.Float32, zero_out=True)

    problem_size = (b, s_q, s_q, s_k, h_q, h_k, d, dv)
    scale_softmax = 1.0 / math.sqrt(d)
    mask_type = (
        fmha_utils.MaskEnum.WINDOW_MASK_INFERENCE
        if causal
        else fmha_utils.MaskEnum.RESIDUAL_MASK
    )
    ws_right = Int32(0) if causal else None

    fmha = BlackwellFusedMultiHeadBlockScaledAttentionForward(
        qk_acc_dtype=cutlass.Float32,
        pv_acc_dtype=cutlass.Float32,
        mma_tiler=MMA_TILER,
        head_dim=d,
        is_persistent=False,
        mask_type=mask_type,
        enable_ex2_emulation=ENABLE_EX2,
        enable_skip_correction=True,
        qk_sf_vec_size=sf_vec,
        use_tma_store=True,  # non-varlen
    )
    stream = _default_stream()
    args = (
        q_cute,
        k_cute,
        q_sf_cute,
        k_sf_cute,
        v_cute,
        o_cute,
        problem_size,
        None,
        None,
        lse_cute,
        None,
    )
    compiled = cute.compile(
        fmha,
        *args,
        scale_softmax * LOG2_E,
        scale_softmax,
        1.0,
        None,
        None,
        None,
        ws_right,
        None,
        None,
        stream,
        False,
    )
    compiled(
        *args,
        Float32(scale_softmax * LOG2_E),
        Float32(scale_softmax),
        Float32(1.0),
        None,
        None,
        None,
        ws_right,
        None,
        None,
        stream,
        False,
    )
    torch.cuda.synchronize()

    o = _readback(o_cute, (b, s_q, h_k, h_r, dv))
    o_ref, lse_ref = _ref_attention(
        q_ref,
        k_ref,
        v_ref,
        batch=b,
        s_q_max=s_q,
        s_k_max=s_k,
        scale_softmax=scale_softmax,
        scale_output=1.0,
        causal=causal,
        q_sf5=q_sf5,
        k_sf5=k_sf5,
    )
    # Compare like the kernel refcheck: round the reference through E4M3.
    o_ref_q = o_ref.to(torch.float8_e4m3fn).float().cpu()
    torch.testing.assert_close(o.float(), o_ref_q, atol=0.2, rtol=1e-2)

    if with_lse:
        lse = _readback(lse_cute, (b, s_q, h_k, h_r))
        torch.testing.assert_close(lse, lse_ref.cpu(), atol=0.2, rtol=1e-2)
