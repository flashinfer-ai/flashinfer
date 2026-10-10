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
"""

import math

import pytest
import torch
from tests.test_helpers.parametrize import pairwise_product_cases, parametrize_product

import flashinfer
from flashinfer import prefill
from flashinfer.jit.attention.modules import (
    _gen_batch_prefill_primary_module,
    gen_batch_prefill_module,
    gen_single_prefill_module,
)
from flashinfer.utils import has_flashinfer_jit_cache, is_sm12x_supported

PAGE_SIZE = 16
_WORKSPACE_BUFFER_SIZE = 128 * 1024 * 1024
_workspace_buffer = None

# (qo_len, kv_len) per request, mixing multiples and non-multiples of 16 and 64.
SEQ_LENS = {
    "self": [(37, 37), (130, 130), (257, 257), (64, 64), (1, 1)],
    "append": [(5, 70), (33, 200), (1, 129), (100, 100)],
}


def _requires_sm12x():
    # The FP16-accumulate MMA kernels are only selected on SM12x; elsewhere the flag keeps
    # its previous behavior.
    if not is_sm12x_supported(torch.device("cuda:0")):
        pytest.skip("FA2 FP16-accumulate MMA kernels are SM12x only")


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    device = torch.device("cuda:0")
    if not is_sm12x_supported(device):
        yield
        return
    specs = []
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (64, 128, 256):
            specs.append(
                _gen_batch_prefill_primary_module(
                    "fa2",
                    dtype,
                    dtype,
                    dtype,
                    torch.int32,
                    head_dim,
                    head_dim,
                    0,
                    False,
                    False,
                    True,
                    **prefill._fp16_accum_mma_kwargs(True, dtype, dtype, "fa2", device),
                )
            )
    flashinfer.jit.build_jit_specs(specs, verbose=False)
    yield


def _get_workspace_buffer():
    global _workspace_buffer
    if _workspace_buffer is None:
        _workspace_buffer = torch.zeros(
            _WORKSPACE_BUFFER_SIZE, dtype=torch.uint8, device="cuda:0"
        )
    else:
        _workspace_buffer.zero_()
    return _workspace_buffer


def _ref_attention(q, k, v, causal, sm_scale):
    """FP32 attention of one request. q: [qo_len, Hq, D]; k, v: [kv_len, Hkv, D]."""
    qo_len, kv_len = q.shape[0], k.shape[0]
    group = q.shape[1] // k.shape[1]
    k = k.float().repeat_interleave(group, dim=1)
    v = v.float().repeat_interleave(group, dim=1)
    logits = torch.einsum("qhd,khd->hqk", q.float(), k) * sm_scale
    if causal:
        q_pos = torch.arange(qo_len, device=q.device)[:, None] + (kv_len - qo_len)
        mask = torch.arange(kv_len, device=q.device)[None, :] > q_pos
        logits = logits.masked_fill(mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(logits, dim=-1), v)


def _make_inputs(seq_lens, num_qo_heads, num_kv_heads, head_dim, dtype, std):
    torch.manual_seed(42)
    qo_lens = [qo_len for qo_len, _ in seq_lens]
    kv_lens = [kv_len for _, kv_len in seq_lens]
    q = torch.randn(sum(qo_lens), num_qo_heads, head_dim, device="cuda:0")
    k = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda:0")
    v = torch.randn(sum(kv_lens), num_kv_heads, head_dim, device="cuda:0")
    return (q * std).to(dtype), (k * std).to(dtype), v.to(dtype), qo_lens, kv_lens


def _indptr(lens):
    return torch.tensor([0] + lens, device="cuda:0").cumsum(0).to(torch.int32)


def _run_ragged(q, k, v, qo_lens, kv_lens, causal, sm_scale):
    wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
        _get_workspace_buffer(), "NHD", backend="fa2"
    )
    wrapper.plan(
        _indptr(qo_lens),
        _indptr(kv_lens),
        q.shape[1],
        k.shape[1],
        q.shape[2],
        causal=causal,
        sm_scale=sm_scale,
        use_fp16_qk_reduction=True,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    return wrapper.run(q, k, v)


def _run_paged(q, k, v, qo_lens, kv_lens, causal, sm_scale):
    hkv, head_dim = k.shape[1:]
    num_pages = [math.ceil(n / PAGE_SIZE) for n in kv_lens]
    page_ids = torch.randperm(sum(num_pages), device="cuda:0").to(torch.int32)
    kv_cache = torch.zeros(
        sum(num_pages), 2, PAGE_SIZE, hkv, head_dim, dtype=k.dtype, device="cuda:0"
    )
    token, page = 0, 0
    for n, p in zip(kv_lens, num_pages, strict=True):
        pad = (0, 0, 0, 0, 0, p * PAGE_SIZE - n)
        ids = page_ids[page : page + p].long()
        kv_cache[ids, 0] = torch.nn.functional.pad(k[token : token + n], pad).view(
            p, PAGE_SIZE, hkv, head_dim
        )
        kv_cache[ids, 1] = torch.nn.functional.pad(v[token : token + n], pad).view(
            p, PAGE_SIZE, hkv, head_dim
        )
        token += n
        page += p
    last_page_len = torch.tensor(
        [n - (p - 1) * PAGE_SIZE for n, p in zip(kv_lens, num_pages, strict=True)],
        dtype=torch.int32,
        device="cuda:0",
    )
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        _get_workspace_buffer(), "NHD", backend="fa2"
    )
    wrapper.plan(
        _indptr(qo_lens),
        _indptr(num_pages),
        page_ids,
        last_page_len,
        q.shape[1],
        hkv,
        head_dim,
        PAGE_SIZE,
        causal=causal,
        sm_scale=sm_scale,
        use_fp16_qk_reduction=True,
        q_data_type=q.dtype,
        kv_data_type=k.dtype,
    )
    return wrapper.run(q, kv_cache)


_RUN = {"paged": _run_paged, "ragged": _run_ragged}


def _check_against_reference(out, q, k, v, qo_lens, kv_lens, causal, sm_scale, tol):
    assert torch.isfinite(out).all(), "output contains inf/NaN"
    q_off = kv_off = 0
    for qo_len, kv_len in zip(qo_lens, kv_lens, strict=True):
        ref = _ref_attention(
            q[q_off : q_off + qo_len],
            k[kv_off : kv_off + kv_len],
            v[kv_off : kv_off + kv_len],
            causal,
            sm_scale,
        )
        torch.testing.assert_close(
            out[q_off : q_off + qo_len].float(), ref, rtol=tol, atol=tol
        )
        q_off += qo_len
        kv_off += kv_len


@parametrize_product(
    {
        "wrapper": ["paged", "ragged"],
        "dtype": [torch.float16, torch.bfloat16],
        "num_qo_heads,num_kv_heads": [(8, 2), (4, 4)],
        "head_dim": [64, 128, 256],
        "causal": [True, False],
        "seq_lens_name": ["self", "append"],
    },
    regular=pairwise_product_cases,
)
def test_batch_prefill_fp16_qk_reduction(
    wrapper, dtype, num_qo_heads, num_kv_heads, head_dim, causal, seq_lens_name
):
    _requires_sm12x()
    q, k, v, qo_lens, kv_lens = _make_inputs(
        SEQ_LENS[seq_lens_name], num_qo_heads, num_kv_heads, head_dim, dtype, std=1.0
    )
    sm_scale = 1.0 / math.sqrt(head_dim)
    out = _RUN[wrapper](q, k, v, qo_lens, kv_lens, causal, sm_scale)
    _check_against_reference(out, q, k, v, qo_lens, kv_lens, causal, sm_scale, tol=1e-2)


@parametrize_product(
    {
        "wrapper": ["paged", "ragged"],
        "dtype": [torch.float16, torch.bfloat16],
        "std": [8.0, 32.0],
        "causal": [True, False],
    },
    regular=pairwise_product_cases,
)
def test_batch_prefill_fp16_qk_reduction_value_range(wrapper, dtype, std, causal):
    # Raw q.k has std std**2 * sqrt(head_dim) (724 for std 8, 11.6k for std 32), far beyond
    # what a chained FP16 accumulation resolves, while each short FP16 partial stays inside
    # the FP16 range and is carried in FP32. sm_scale keeps the softmax logits at std 4,
    # i.e. well conditioned, so the result must still track the FP32 reference.
    _requires_sm12x()
    head_dim, num_qo_heads, num_kv_heads = 128, 8, 2
    q, k, v, qo_lens, kv_lens = _make_inputs(
        SEQ_LENS["self"], num_qo_heads, num_kv_heads, head_dim, dtype, std=std
    )
    sm_scale = 4.0 / (std**2 * math.sqrt(head_dim))
    out = _RUN[wrapper](q, k, v, qo_lens, kv_lens, causal, sm_scale)
    _check_against_reference(out, q, k, v, qo_lens, kv_lens, causal, sm_scale, tol=3e-2)


@pytest.mark.parametrize(
    "use_fp16_qk_reduction,dtype_q,dtype_kv,backend,sm12x,expected",
    [
        (True, torch.float16, torch.float16, "fa2", True, True),
        (True, torch.bfloat16, torch.bfloat16, "fa2", True, True),
        (False, torch.bfloat16, torch.bfloat16, "fa2", True, False),
        (True, torch.bfloat16, torch.float16, "fa2", True, False),
        (True, torch.float16, torch.float8_e4m3fn, "fa2", True, False),
        (True, torch.bfloat16, torch.bfloat16, "fa3", True, False),
        (True, torch.bfloat16, torch.bfloat16, "fa2", False, False),
    ],
)
def test_fp16_accum_mma_switch(
    monkeypatch, use_fp16_qk_reduction, dtype_q, dtype_kv, backend, sm12x, expected
):
    monkeypatch.setattr(prefill, "is_sm12x_supported", lambda device: sm12x)
    kwargs = prefill._fp16_accum_mma_kwargs(
        use_fp16_qk_reduction, dtype_q, dtype_kv, backend, torch.device("cuda:0")
    )
    assert kwargs == ({"fp16_accum_mma": True} if expected else {})


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "use_fp16_qk_reduction,fp16_accum_mma",
    [(False, False), (True, False), (True, True)],
)
def test_fp16_accum_mma_jit_flags(dtype, use_fp16_qk_reduction, fp16_accum_mma):
    kwargs = {"fp16_accum_mma": True} if fp16_accum_mma else {}
    specs = [
        gen_single_prefill_module(
            "fa2",
            dtype,
            dtype,
            dtype,
            128,
            128,
            0,
            False,
            False,
            use_fp16_qk_reduction,
            **kwargs,
        ),
        gen_batch_prefill_module(
            "fa2",
            dtype,
            dtype,
            dtype,
            torch.int32,
            128,
            128,
            0,
            False,
            False,
            use_fp16_qk_reduction,
            **kwargs,
        ),
    ]
    for spec in specs:
        flags = spec.extra_cuda_cflags or []
        assert ("-DFLASHINFER_FP16_ACCUM_MMA" in flags) == fp16_accum_mma
        assert ("_f16mma" in spec.name) == fp16_accum_mma
        assert ("f16qk_True" in spec.name) == use_fp16_qk_reduction
