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
from conftest import clear_cuda_cache

import flashinfer
from flashinfer.jit import build_jit_specs
from flashinfer.jit.attention import (
    gen_batch_mla_module,
    gen_batch_prefill_module,
    gen_single_prefill_module,
)
from flashinfer.utils import is_sm90a_supported, is_sm100a_supported


@pytest.fixture(autouse=True, scope="module")
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
        torch.testing.assert_close(o_i, o_ref, rtol=1e-3, atol=1e-3)
        # if kv_lens[i] != 0:
        #     torch.testing.assert_close(lse_i, lse_ref, rtol=1e-3, atol=1e-3)


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
    if not is_sm100a_supported(device):
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


if __name__ == "__main__":
    test_batch_mla_varlen_page_attention(
        1, 65, 65, 65, 1, 128, True, 64, "fa2", torch.half
    )
    # test_batch_mla_varlen_page_attention(
    #     155, 1024, 8, 128, 128, 16, False, 1, "fa3", torch.half
    # )
    # test_batch_mla_page_attention(1, 1024, 128, 128, False, 1, "fa2", True, torch.half)
