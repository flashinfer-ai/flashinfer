"""
Copyright (c) 2023 by FlashInfer team.

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
from tests.test_helpers.test_helpers import clear_cuda_cache

import flashinfer
from flashinfer.jit import build_jit_specs
from flashinfer.jit.attention import (
    gen_batch_mla_module,
    gen_batch_prefill_module,
    gen_single_prefill_module,
)
from flashinfer.utils import (
    has_flashinfer_jit_cache,
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm110a_supported,
)


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    try:
        modules = []
        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
                continue

            modules.append(
                gen_single_prefill_module(
                    backend,
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    192,
                    128,
                    0,
                    False,
                    False,
                    False,
                )
            )

        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
                continue

            modules.append(
                gen_batch_prefill_module(
                    backend,
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    192,
                    128,
                    0,
                    False,
                    False,
                    False,
                )
            )

        for backend in ["fa2", "fa3"]:
            if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
                continue

            modules.append(
                gen_batch_mla_module(
                    backend,
                    torch.float16,
                    torch.float16,
                    torch.float16,
                    torch.int32,
                    512,
                    64,
                    False,
                )
            )

        build_jit_specs(modules, verbose=False)
    except Exception as e:
        # abort the test session if warmup fails
        pytest.exit(str(e))
    finally:
        yield


def attention_ref(
    batch_size,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    sm_scale: float,
) -> torch.Tensor:
    qo_len = q.shape[0] // batch_size
    kv_len = k.shape[0] // batch_size
    num_qo_heads = q.shape[1]
    head_dim_qk = q.shape[2]
    head_dim_vo = v.shape[2]
    logits = (
        torch.einsum(
            "bmhd,bnhd->bhmn",
            q.view(batch_size, qo_len, num_qo_heads, head_dim_qk).float(),
            k.view(batch_size, kv_len, num_qo_heads, head_dim_qk).float(),
        )
        * sm_scale
    )

    if causal:
        mask = torch.arange(kv_len - qo_len, kv_len, device=q.device).unsqueeze(
            1
        ) >= torch.arange(0, kv_len, device=q.device).unsqueeze(0)
    else:
        mask = torch.ones(qo_len, kv_len, device=q.device)

    logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0) == 0, float("-inf"))
    lse_ref = torch.logsumexp(logits, -1).transpose(-1, -2)
    p = torch.softmax(logits, dim=-1)
    o_ref = (
        torch.einsum(
            "bhmn,bnhd->bmhd",
            p,
            v.view(batch_size, kv_len, num_qo_heads, head_dim_vo).float(),
        )
        .contiguous()
        .view(batch_size * qo_len, num_qo_heads, head_dim_vo)
        .to(q)
    )

    return o_ref, lse_ref * math.log2(math.e)


@pytest.mark.parametrize("kv_len", [5532, 7563])
@pytest.mark.parametrize("qo_len", [1832, 3928])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", [torch.half])
def test_single_prefill_with_kv_cache(
    kv_len,
    qo_len,
    num_heads,
    causal,
    backend,
    dtype,
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if is_sm110a_supported(device) and num_heads * kv_len > 700000:
        pytest.skip("skip large tests on Thor due to memory limit")
    torch.manual_seed(42)
    head_dim_qk = 192
    head_dim_vo = 128
    q = torch.randn(qo_len, num_heads, head_dim_qk, dtype=dtype, device=device)
    k = torch.randn(kv_len, num_heads, head_dim_qk, dtype=dtype, device=device)
    v = torch.randn(kv_len, num_heads, head_dim_vo, dtype=dtype, device=device)
    o, lse = flashinfer.single_prefill_with_kv_cache(
        q, k, v, causal=causal, backend=backend, return_lse=True
    )
    sm_scale = 1.0 / (head_dim_qk**0.5)

    o_ref, lse_ref = attention_ref(1, q, k, v, causal, sm_scale)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref.squeeze(0), rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [12, 17])
@pytest.mark.parametrize("kv_len", [544, 977])
@pytest.mark.parametrize("qo_len", [377, 177])
@pytest.mark.parametrize("num_heads", [4, 32, 128])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", [torch.half])
def test_batch_prefill_with_ragged_kv_cache(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    backend,
    dtype,
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    torch.manual_seed(42)
    kv_layout = "NHD"
    head_dim_qk = 192
    head_dim_vo = 128
    q = torch.randn(
        batch_size * qo_len, num_heads, head_dim_qk, dtype=dtype, device=device
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )

    k = torch.zeros(
        batch_size * kv_len, num_heads, head_dim_qk, dtype=dtype, device=device
    )
    v = torch.zeros(
        batch_size * kv_len, num_heads, head_dim_vo, dtype=dtype, device=device
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * kv_len
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, kv_layout, backend=backend
    )
    wrapper.plan(
        q_indptr,
        kv_indptr,
        num_heads,
        num_heads,
        head_dim_qk,
        head_dim_vo=head_dim_vo,
        causal=causal,
    )
    o, lse = wrapper.run_return_lse(q, k, v)

    sm_scale = 1.0 / (head_dim_qk**0.5)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)

    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q, k, v, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)


def generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads):
    bs_page_num, page_size, ckv_dim = ckv.shape
    page_num = bs_page_num // batch_size
    _, _, kpe_dim = kpe.shape
    ckv = ckv.view(batch_size, page_num * page_size, ckv_dim)
    kpe = kpe.view(batch_size, page_num * page_size, kpe_dim)
    ckv = ckv[:, :kv_len, :]
    kpe = kpe[:, :kv_len, :]
    k = (
        torch.cat([ckv, kpe], dim=-1)
        .view(-1, 1, ckv_dim + kpe_dim)
        .repeat_interleave(num_heads, dim=1)
    )
    v = ckv.repeat_interleave(num_heads, dim=1)

    return k, v


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7])
@pytest.mark.parametrize("kv_len_0", [0, 1, 3, 11])
@pytest.mark.parametrize("kv_len_1", [17, 33, 79, 114])
@pytest.mark.parametrize("kv_len_2", [514, 2743, 8736])
@pytest.mark.parametrize("qo_len", [1, 3, 5, 7, 9, 11, 13, 15, 17])
@pytest.mark.parametrize("num_heads", [16, 64])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", [torch.half])
def test_batch_mla_varlen_page_attention(
    batch_size,
    kv_len_0,
    kv_len_1,
    kv_len_2,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
    dtype,
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > min(kv_len_0, kv_len_1, kv_len_2):
        pytest.skip("qo_len > kv_len not supported for causal attention")
    num_different_kv_len = 3
    kv_lens = torch.tensor([kv_len_0, kv_len_1, kv_len_2], dtype=torch.int32)
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        num_different_kv_len * batch_size * qo_len,
        num_heads,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    q_pe = torch.randn(
        num_different_kv_len * batch_size * qo_len,
        num_heads,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    pages_nums = torch.tensor(
        [math.ceil(kv_len / page_size) for kv_len in kv_lens],
        dtype=torch.int32,
    )
    pages_nums_indptr = torch.zeros(num_different_kv_len + 1, dtype=torch.int32)
    pages_nums_indptr[1:] = pages_nums.cumsum(0)
    pages_nums_sum = pages_nums_indptr[-1]
    ckv = torch.randn(
        batch_size * pages_nums_sum,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_nums_sum,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = (
        torch.arange(
            0, num_different_kv_len * batch_size + 1, device=device, dtype=torch.int32
        )
        * qo_len
    )
    kv_indptr = torch.cat(
        [
            torch.arange(0, batch_size + 1).unsqueeze(-1).int() * pages_nums_sum
            + pages_nums_indptr[i]
            for i in range(num_different_kv_len)
        ],
        dim=-1,
    ).flatten()
    kv_indices = torch.arange(
        0, batch_size * pages_nums_sum, device=device, dtype=torch.int32
    )
    kv_lens = torch.tensor(kv_lens, dtype=torch.int32, device=device).repeat(batch_size)
    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    q_rows = (
        torch.arange(0, num_different_kv_len * qo_len)[None, :]
        + torch.arange(0, batch_size)[:, None] * num_different_kv_len * qo_len
    ).int()
    kv_rows = (
        torch.arange(0, pages_nums_sum)[None, :]
        + torch.arange(0, batch_size)[:, None] * pages_nums_sum
    ).int()
    q_rows_arr = [
        q_rows[:, i * qo_len : (i + 1) * qo_len].flatten()
        for i in range(num_different_kv_len)
    ]
    kv_rows_arr = [
        kv_rows[:, pages_nums_indptr[i] : pages_nums_indptr[i + 1]].flatten()
        for i in range(num_different_kv_len)
    ]
    for i in range(num_different_kv_len):
        k, v = generate_kv_from_cache(
            ckv[kv_rows_arr[i]], kpe[kv_rows_arr[i]], kv_lens[i], batch_size, num_heads
        )
        q = torch.cat([q_nope, q_pe], dim=-1)[q_rows_arr[i]]
        o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
        lse_ref = lse_ref.flatten(0, 1)
        o_i = o[q_rows_arr[i]]
        lse_i = lse[q_rows_arr[i]]
        torch.testing.assert_close(o_i, o_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(lse_i, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 3, 4, 5, 6, 7, 157])
@pytest.mark.parametrize("kv_len", [17, 33, 75, 197])
@pytest.mark.parametrize("qo_len", [3, 7, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [16, 32])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("dtype", [torch.half])
def test_batch_mla_oob_kv_nan(
    batch_size, kv_len, qo_len, num_heads, causal, page_size, backend, dtype
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size * qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num, page_size, head_dim_ckv, dtype=dtype, device=device
    )
    kpe = torch.randn(
        batch_size * pages_num, page_size, head_dim_kpe, dtype=dtype, device=device
    )

    # Fill oob positions with nan
    for i in range(batch_size):
        last_page_len = kv_len - (pages_num - 1) * page_size
        ckv[(i + 1) * pages_num - 1, last_page_len:, :] = float("nan")
        kpe[(i + 1) * pages_num - 1, last_page_len:, :] = float("nan")

    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer, backend=backend
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    )
    kv_indices = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    )
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    if kv_len != 0:
        torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 3, 5, 7, 157])
@pytest.mark.parametrize("kv_len", [0, 17, 33, 96, 97, 114, 514, 1024])
@pytest.mark.parametrize("qo_len", [1, 3, 5, 7, 9, 11, 13, 15, 17])
@pytest.mark.parametrize("num_heads", [16])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("page_size", [1, 16])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("use_cuda_graph", [False])
@pytest.mark.parametrize("dtype", [torch.half])
def test_batch_mla_page_attention(
    batch_size,
    kv_len,
    qo_len,
    num_heads,
    causal,
    page_size,
    backend,
    use_cuda_graph,
    dtype,
):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if backend == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 is not supported on this device")
    if causal and qo_len > kv_len:
        pytest.skip("qo_len > kv_len not supported for causal attention")
    torch.manual_seed(42)
    head_dim_ckv = 512
    head_dim_kpe = 64
    q_nope = torch.randn(
        batch_size * qo_len, num_heads, head_dim_ckv, dtype=dtype, device=device
    )
    q_pe = torch.randn(
        batch_size * qo_len, num_heads, head_dim_kpe, dtype=dtype, device=device
    )
    pages_num = math.ceil(kv_len / page_size)
    ckv = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_ckv,
        dtype=dtype,
        device=device,
    )
    kpe = torch.randn(
        batch_size * pages_num,
        page_size,
        head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        workspace_buffer,
        backend=backend,
        use_cuda_graph=True,
        qo_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        kv_indptr=torch.empty(batch_size + 1, dtype=torch.int32, device=device),
        kv_indices=torch.empty(1048576, dtype=torch.int32, device=device),
        kv_len_arr=torch.empty(batch_size, dtype=torch.int32, device=device),
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * qo_len
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32) * pages_num
    )
    kv_indices = torch.arange(
        0, batch_size * pages_num, device=device, dtype=torch.int32
    )
    kv_lens = torch.full((batch_size,), kv_len, dtype=torch.int32, device=device)

    if use_cuda_graph:
        kv_indptr_warmup = torch.zeros(batch_size + 1, device=device, dtype=torch.int32)
        kv_indices_warmup = torch.arange(
            0, batch_size, device=device, dtype=torch.int32
        )
        kv_lens_warmup = torch.full((batch_size,), 0, dtype=torch.int32, device=device)
        wrapper.plan(
            q_indptr,
            kv_indptr_warmup,
            kv_indices_warmup,
            kv_lens_warmup,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            page_size,
            causal,
            sm_scale,
            q_nope.dtype,
            ckv.dtype,
        )

        # warmup
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)
        torch.cuda.current_stream().wait_stream(s)

        # capture
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        causal,
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )
    if use_cuda_graph:
        o.fill_(0)
        lse.fill_(0)
        g.replay()
    else:
        o, lse = wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=True)

    k, v = generate_kv_from_cache(ckv, kpe, kv_len, batch_size, num_heads)

    q = torch.cat([q_nope, q_pe], dim=-1)
    o_ref, lse_ref = attention_ref(batch_size, q, k, v, causal, sm_scale)
    lse_ref = lse_ref.flatten(0, 1)
    torch.testing.assert_close(o, o_ref, rtol=1e-3, atol=1e-3)
    if kv_len != 0:
        torch.testing.assert_close(lse, lse_ref, rtol=1e-3, atol=1e-3)

    # test with pre-allocated output
    o_buffer = torch.empty_like(o)
    lse_buffer = torch.empty_like(lse)
    wrapper.run(q_nope, q_pe, ckv, kpe, out=o_buffer, lse=lse_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)
    torch.testing.assert_close(lse, lse_buffer, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("max_seq_len", [128, 1024, 4096])
@pytest.mark.parametrize("page_size", [1, 16, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.half])
def test_cutlass_mla(batch_size, max_seq_len, page_size, dtype):
    device = torch.device("cuda:0")
    clear_cuda_cache(device)
    if not is_sm100a_supported(device) and not is_sm110a_supported(device):
        pytest.skip("Cutlass MLA is not supported on this device")

    torch.manual_seed(42)

    num_local_heads = 128
    head_dim_ckv = 512
    head_dim_kpe = 64
    total_page_num = 8192

    # NOTE(Zihao): use larger scale to detect bugs such as
    # https://github.com/flashinfer-ai/flashinfer/pull/1055
    q_nope_pe = (
        torch.randn(
            batch_size,
            num_local_heads,
            head_dim_ckv + head_dim_kpe,
            dtype=dtype,
            device=device,
        )
        * 100
    )
    ckv_kpe = torch.randn(
        total_page_num,
        page_size,
        head_dim_ckv + head_dim_kpe,
        dtype=dtype,
        device=device,
    )
    kv_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    page_num_per_batch = (max_seq_len + page_size - 1) // page_size
    # Cutlass MLA requires small pages (< 128) are packed into a 128 page.
    assert page_num_per_batch % (128 // page_size) == 0
    page_table = torch.randint(
        0,
        total_page_num,
        (batch_size, page_num_per_batch),
        dtype=torch.int32,
        device=device,
    )

    mla_ref = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device), backend="fa2"
    )

    # for decode, each query length is 1
    q_indptr = torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
    kv_lens = torch.full((batch_size,), max_seq_len, dtype=torch.int32, device=device)
    kv_indptr = (
        torch.arange(0, batch_size + 1, device=device, dtype=torch.int32)
        * page_num_per_batch
    )
    kv_indices = page_table.flatten()

    q_nope = q_nope_pe[..., :head_dim_ckv]
    q_pe = q_nope_pe[..., head_dim_ckv:]
    ckv = ckv_kpe[..., :head_dim_ckv]
    kpe = ckv_kpe[..., head_dim_ckv:]

    # use head dimension before matrix absorption
    sm_scale = 1.0 / ((128 + 64) ** 0.5)
    mla_ref.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_local_heads,
        head_dim_ckv,
        head_dim_kpe,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        ckv.dtype,
    )

    o_ref = mla_ref.run(q_nope, q_pe, ckv, kpe, return_lse=False)

    mla_ans = flashinfer.mla.BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device),
        backend="cutlass",
    )
    o_ans = mla_ans.run(q_nope, q_pe, ckv, kpe, kv_len=kv_lens, page_table=page_table)
    torch.testing.assert_close(o_ans, o_ref, rtol=1e-2, atol=1e-2)


# ───────────────────────────────────────────────────────────────────────────
# FP8 KV cache path (DeepSeek MLA, fa3 / SM90 only). Stores KV as FP8 e4m3
# in shared memory and dequants one tile at a time to BF16 right before WGMMA.
# Numerical reference: dequant the FP8 KV in Python (matching kernel layout)
# and run the BF16 MLA path on the result; remaining diff is BF16 accumulation.
# ───────────────────────────────────────────────────────────────────────────

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
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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


def test_fp8_kv_plan_rejects_fp16_q():
    """FP8 KV MLA is BF16-Q only; FP16 Q must be rejected at plan() time
    with a clear ValueError."""
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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
    """ckv_scale / kpe_scale must be passed by keyword. The `*` marker
    in run()'s signature pins this; this test guards against accidental
    removal of the marker."""
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
        ("q_nope", torch.float16, "q_nope.dtype"),
        ("q_pe", torch.float16, "q_pe.dtype"),
        ("ckv_cache", torch.bfloat16, "ckv_cache.dtype"),
        ("kpe_cache", torch.bfloat16, "kpe_cache.dtype"),
    ],
)
def test_fp8_kv_run_rejects_dtype_mismatch(wrong_tensor, wrong_dtype, exc_match):
    """The C++ launcher reinterprets tensor storage by the JIT-template type
    chosen at plan(); a run-time dtype mismatch produces silent wrong output.
    Each tensor (q_nope, q_pe, ckv_cache, kpe_cache) has an independent
    check; this test exercises all four."""
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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
    if not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("FP8 KV path on Hopper MLA requires SM90a")
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


if __name__ == "__main__":
    test_batch_mla_varlen_page_attention(
        1, 65, 65, 65, 1, 128, True, 64, "fa2", torch.half
    )
    # test_batch_mla_varlen_page_attention(
    #     155, 1024, 8, 128, 128, 16, False, 1, "fa3", torch.half
    # )
    # test_batch_mla_page_attention(1, 1024, 128, 128, False, 1, "fa2", True, torch.half)
