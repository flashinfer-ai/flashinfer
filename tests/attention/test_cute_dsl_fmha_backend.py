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
"""Frontend tests for the `cute-dsl` backend routed to the trtllm JIT FMHA kernel.

Covers entry points:
* `trtllm_ragged_attention_deepseek(backend="cute-dsl")` — the shared low-level API.
* `BatchPrefillWithRaggedKVCacheWrapper(backend="cute-dsl")` — delegates to the former
  for standard attention, falls back to prefill.py for ALiBi / soft-cap.
"""

import math

import pytest
import torch
import torch.nn.functional as F

import flashinfer
from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.utils import is_sm100a_supported

if not is_cute_dsl_available():
    pytest.skip("CuTe DSL not available", allow_module_level=True)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not is_sm100a_supported(torch.device("cuda")),
    reason="CuTe DSL FMHA requires Blackwell (SM100a+)",
)

DEVICE = "cuda"


def _indptr(lengths):
    t = torch.tensor(lengths, device=DEVICE, dtype=torch.int32)
    return torch.cat([torch.zeros(1, device=DEVICE, dtype=torch.int32), t.cumsum(0)])


def _ragged_ref(q, k, v, qo_indptr, kv_indptr, sm_scale, causal):
    """Per-batch float32 ragged attention (top-left causal; equal q/k lengths here)."""
    pieces = []
    for i in range(qo_indptr.numel() - 1):
        qb, qe = int(qo_indptr[i]), int(qo_indptr[i + 1])
        kb, ke = int(kv_indptr[i]), int(kv_indptr[i + 1])
        qi = q[qb:qe].transpose(0, 1).unsqueeze(0).float()
        ki = k[kb:ke].transpose(0, 1).unsqueeze(0).float()
        vi = v[kb:ke].transpose(0, 1).unsqueeze(0).float()
        oi = F.scaled_dot_product_attention(
            qi, ki, vi, scale=sm_scale, is_causal=causal
        )
        pieces.append(oi.squeeze(0).transpose(0, 1))
    return torch.cat(pieces, dim=0)


def _make_qkv(total_q, total_kv, Hq, Hk, dtype_qk, dtype_vo, Dqk, Dvo):
    q = torch.randn(total_q, Hq, Dqk, device=DEVICE, dtype=torch.bfloat16).to(dtype_qk)
    k = torch.randn(total_kv, Hk, Dqk, device=DEVICE, dtype=torch.bfloat16).to(dtype_qk)
    v = torch.randn(total_kv, Hk, Dvo, device=DEVICE, dtype=torch.bfloat16).to(dtype_vo)
    # reference operands (exact stored values)
    qr = q.float() if dtype_qk.itemsize == 1 else q
    kr = k.float() if dtype_qk.itemsize == 1 else k
    vr = v.float() if dtype_vo.itemsize == 1 else v
    return q, k, v, qr, kr, vr


# =============================================================================
# trtllm_ragged_attention_deepseek(backend="cute-dsl")
# =============================================================================


@pytest.mark.parametrize(
    "dtype_qk,dtype_vo,Dqk,Dvo",
    [
        (torch.bfloat16, torch.bfloat16, 128, 128),
        (torch.bfloat16, torch.float8_e4m3fn, 128, 128),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, 128, 128),
        (torch.float8_e4m3fn, torch.float8_e4m3fn, 192, 128),  # MLA
    ],
)
def test_deepseek_cute_dsl(dtype_qk, dtype_vo, Dqk, Dvo):
    torch.manual_seed(0)
    b, s, H = 2, 256, 8
    qo = _indptr([s] * b)
    kv = qo.clone()
    total = b * s
    sm = 1.0 / math.sqrt(Dqk)
    q, k, v, qr, kr, vr = _make_qkv(total, total, H, H, dtype_qk, dtype_vo, Dqk, Dvo)
    ws = torch.empty(128 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)
    seq_lens = torch.tensor([s] * b, device=DEVICE, dtype=torch.int32)

    o = flashinfer.prefill.trtllm_ragged_attention_deepseek(
        query=q,
        key=k,
        value=v,
        workspace_buffer=ws,
        seq_lens=seq_lens,
        max_q_len=s,
        max_kv_len=s,
        bmm1_scale=sm,
        bmm2_scale=1.0,
        o_sf_scale=1.0,
        batch_size=b,
        window_left=-1,
        cum_seq_lens_q=qo,
        cum_seq_lens_kv=kv,
        enable_pdl=False,
        is_causal=False,
        return_lse=False,
        backend="cute-dsl",
    )
    ref = _ragged_ref(qr, kr, vr, qo, kv, sm, causal=False)
    atol = 4e-2 if min(dtype_qk.itemsize, dtype_vo.itemsize) == 1 else 6e-3
    torch.testing.assert_close(o.float(), ref.float(), atol=atol, rtol=atol)


# =============================================================================
# BatchPrefillWithRaggedKVCacheWrapper(backend="cute-dsl")
# =============================================================================


@pytest.mark.parametrize(
    "dtype_qk,dtype_vo",
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.bfloat16, torch.float8_e4m3fn),
        (torch.float8_e4m3fn, torch.float8_e4m3fn),
    ],
)
@pytest.mark.parametrize("causal", [False, True])
def test_batch_prefill_cute_dsl(dtype_qk, dtype_vo, causal):
    torch.manual_seed(0)
    b, s, Hq, Hk, D = 2, 1024, 8, 8, 128
    qo = _indptr([s] * b)
    kv = qo.clone()
    total = b * s
    sm = 1.0 / math.sqrt(D)
    q, k, v, qr, kr, vr = _make_qkv(total, total, Hq, Hk, dtype_qk, dtype_vo, D, D)
    ws = torch.empty(128 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)

    w = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", backend="cute-dsl"
    )
    w.plan(qo, kv, Hq, Hk, D, causal=causal, q_data_type=dtype_qk)
    o = w.run(q, k, v)
    ref = _ragged_ref(qr, kr, vr, qo, kv, sm, causal=causal)
    atol = 4e-2 if min(dtype_qk.itemsize, dtype_vo.itemsize) == 1 else 6e-3
    torch.testing.assert_close(o.float(), ref.float(), atol=atol, rtol=atol)

    # LSE is now supported for standard attention.
    o2, lse = w.run(q, k, v, return_lse=True)
    torch.testing.assert_close(o2.float(), o.float(), atol=1e-2, rtol=1e-2)
    assert lse.shape == (total, Hq)


def test_cute_dsl_jit_fallback(monkeypatch):
    """When the cubin variant is unavailable, cute_dsl_fmha_ragged_prefill must
    fall back to the JIT runner and still produce correct results."""
    import flashinfer.attention.cute_dsl.fmha as cubin_mod

    def _raise(*args, **kwargs):
        raise RuntimeError("forced: cubin unavailable")

    monkeypatch.setattr(cubin_mod, "get_cute_dsl_fmha_kernel", _raise)

    torch.manual_seed(0)
    b, s, H, D = 2, 256, 4, 128
    qo = _indptr([s] * b)
    kv = qo.clone()
    total = b * s
    sm = 1.0 / math.sqrt(D)
    q, k, v, qr, kr, vr = _make_qkv(
        total, total, H, H, torch.bfloat16, torch.bfloat16, D, D
    )
    o = torch.empty(total, H, D, device=DEVICE, dtype=torch.bfloat16)

    cubin_mod.cute_dsl_fmha_ragged_prefill(
        q=q,
        k=k,
        v=v,
        o=o,
        qo_indptr=qo,
        kv_indptr=kv,
        is_causal=False,
        sm_scale=sm,
        max_qo_len=s,
        max_kv_len=s,
    )
    ref = _ragged_ref(qr, kr, vr, qo, kv, sm, causal=False)
    torch.testing.assert_close(o.float(), ref.float(), atol=3e-2, rtol=3e-2)


def test_batch_prefill_cute_dsl_alibi_fallback():
    """ALiBi is unsupported by the vendored kernel; must fall back to prefill.py."""
    torch.manual_seed(0)
    b, s, H, D = 2, 256, 8, 128
    qo = _indptr([s] * b)
    kv = qo.clone()
    total = b * s
    q = torch.randn(total, H, D, device=DEVICE, dtype=torch.bfloat16)
    k = torch.randn(total, H, D, device=DEVICE, dtype=torch.bfloat16)
    v = torch.randn(total, H, D, device=DEVICE, dtype=torch.bfloat16)
    ws = torch.empty(128 * 1024 * 1024, device=DEVICE, dtype=torch.uint8)

    w = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        ws, kv_layout="NHD", backend="cute-dsl"
    )
    w.plan(
        qo,
        kv,
        H,
        H,
        D,
        causal=False,
        pos_encoding_mode="ALIBI",
        q_data_type=torch.bfloat16,
    )
    o = w.run(q, k, v)
    assert o.shape == (total, H, D)
    assert torch.isfinite(o.float()).all()
