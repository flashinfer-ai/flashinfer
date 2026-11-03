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
from tests.test_helpers.jit_utils import (
    gen_decode_attention_modules,
    gen_prefill_attention_modules,
)

import flashinfer
from flashinfer.utils import has_flashinfer_jit_cache


@pytest.fixture(
    autouse=not has_flashinfer_jit_cache(),
    scope="module",
)
def warmup_jit():
    flashinfer.jit.build_jit_specs(
        gen_decode_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
        )
        + gen_prefill_attention_modules(
            [torch.float16],  # q_dtypes
            [torch.float16],  # kv_dtypes
            [64, 128, 256],  # head_dims
            [0, 1],  # pos_encoding_modes
            [False],  # use_sliding_windows
            [False],  # use_logits_soft_caps
            [False],  # use_fp16_qk_reductions
        ),
        verbose=False,
    )
    yield


@pytest.mark.parametrize("batch_size", [5, 12])
@pytest.mark.parametrize("invariant_bs", [4])
@pytest.mark.parametrize("kv_len", [4096, 8192, 5000])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_decode_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = torch.randn(
        batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=torch.float16
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
        if kv_layout == "HND"
        else torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda:0")

    wrapper_tensor_cores = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout, use_tensor_cores=True
    )
    wrapper_tensor_cores.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=torch.float16,
        q_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
        q, kv_data, return_lse=True
    )

    kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
    kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]
    wrapper_tensor_cores.plan(
        kv_indptr_invariant,
        kv_indices,
        kv_last_page_len_invariant,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        data_type=torch.float16,
        q_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
        q[:invariant_bs], kv_data, return_lse=True
    )
    assert torch.equal(o_tensor_cores[:invariant_bs], o_tensor_cores_invariant)
    assert torch.equal(lse_tensor_cores[:invariant_bs], lse_tensor_cores_invariant)


# test that without fixed split size, precision is different
# TODO: this works for the first 29 cases, but then fails with "illegal memory access"..?

# wrapper_tensor_cores.plan(
#     kv_indptr,
#     kv_indices,
#     kv_last_page_len,
#     num_qo_heads,
#     num_kv_heads,
#     head_dim,
#     page_size,
# )
# o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
#     q[:invariant_bs], kv_data, return_lse=True
# )
# try:
#     torch.testing.assert_close(
#         o_tensor_cores[:invariant_bs], o_tensor_cores_invariant, rtol=1e-7, atol=1e-7
#     )
#     torch.testing.assert_close(
#         lse_tensor_cores[:invariant_bs], lse_tensor_cores_invariant, rtol=1e-7, atol=1e-7
#     )
# except AssertionError:
#     pass
# else:
#     raise AssertionError("Precision is the same without fixed split size")


@pytest.mark.parametrize("batch_size", [3, 4])
@pytest.mark.parametrize("invariant_bs", [2])
@pytest.mark.parametrize("kv_len", [4096, 5000])
@pytest.mark.parametrize("qo_len", [128, 256])
@pytest.mark.parametrize("fixed_split_size", [2048])
@pytest.mark.parametrize("disable_split_kv", [True, False])
@pytest.mark.parametrize("page_size", [1, 8, 16])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("group_size", [1, 4, 8])
@pytest.mark.parametrize("head_dim", [128, 256])
@pytest.mark.parametrize("kv_layout", ["HND", "NHD"])
@pytest.mark.parametrize("pos_encoding_mode", ["NONE", "ROPE_LLAMA"])
def test_batch_prefill_tensor_cores(
    batch_size: int,
    invariant_bs: int,
    kv_len: int,
    qo_len: int,
    fixed_split_size: int,
    disable_split_kv: bool,
    page_size: int,
    num_kv_heads: int,
    group_size: int,
    head_dim: int,
    kv_layout: str,
    pos_encoding_mode: str,
):
    num_qo_heads = num_kv_heads * group_size
    q = torch.randn(
        batch_size * qo_len,
        num_qo_heads,
        head_dim,
        device="cuda:0",
        dtype=torch.float16,
    )
    q_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32) * qo_len
    )
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    kv_data = (
        torch.randn(
            total_num_pages,
            2,
            num_kv_heads,
            page_size,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
        if kv_layout == "HND"
        else torch.randn(
            total_num_pages,
            2,
            page_size,
            num_kv_heads,
            head_dim,
            device="cuda:0",
            dtype=torch.float16,
        )
        / 10
    )
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(
        1024 * 1024 * 1024, dtype=torch.int8, device="cuda:0"
    )

    wrapper_tensor_cores = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper_tensor_cores.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores, lse_tensor_cores = wrapper_tensor_cores.run(
        q, kv_data, return_lse=True
    )

    # Test invariant batch size
    q_indptr_invariant = q_indptr[: invariant_bs + 1]
    kv_indptr_invariant = kv_indptr[: invariant_bs + 1]
    kv_last_page_len_invariant = kv_last_page_len[:invariant_bs]
    wrapper_tensor_cores.plan(
        q_indptr_invariant,
        kv_indptr_invariant,
        kv_indices,
        kv_last_page_len_invariant,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        pos_encoding_mode=pos_encoding_mode,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        fixed_split_size=fixed_split_size if not disable_split_kv else None,
        disable_split_kv=disable_split_kv,
    )
    o_tensor_cores_invariant, lse_tensor_cores_invariant = wrapper_tensor_cores.run(
        q[: invariant_bs * qo_len], kv_data, return_lse=True
    )

    # Compare outputs for the invariant batch size
    assert torch.equal(
        o_tensor_cores[: invariant_bs * qo_len], o_tensor_cores_invariant
    )
    assert torch.equal(
        lse_tensor_cores[: invariant_bs * qo_len], lse_tensor_cores_invariant
    )


_NUM_KV_HEADS = 8
_GROUP_SIZE = 4
_HEAD_DIM = 128
_PAGE_SIZE = 16


def _paged_kv(kv_lens, kv_data):
    pages = [(l + _PAGE_SIZE - 1) // _PAGE_SIZE for l in kv_lens]
    assert sum(pages) <= kv_data.shape[0]
    kv_indptr = torch.tensor([0] + pages, dtype=torch.int32, device="cuda:0").cumsum(
        0, dtype=torch.int32
    )
    kv_indices = torch.arange(kv_indptr[-1].item(), dtype=torch.int32, device="cuda:0")
    kv_last_page_len = torch.tensor(
        [(l - 1) % _PAGE_SIZE + 1 for l in kv_lens], dtype=torch.int32, device="cuda:0"
    )
    return kv_indptr, kv_indices, kv_last_page_len


def _run_decode(kv_lens, q, kv_data, **plan_kwargs):
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=True
    )
    kv_indptr, kv_indices, kv_last_page_len = _paged_kv(kv_lens, kv_data)
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_KV_HEADS * _GROUP_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        **plan_kwargs,
    )
    return wrapper.run(q, kv_data, return_lse=True)


def _run_prefill(qo_lens, kv_lens, q, kv_data, **plan_kwargs):
    workspace_buffer = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    qo_indptr = torch.tensor(
        [0] + list(qo_lens), dtype=torch.int32, device="cuda:0"
    ).cumsum(0, dtype=torch.int32)
    kv_indptr, kv_indices, kv_last_page_len = _paged_kv(kv_lens, kv_data)
    wrapper.plan(
        qo_indptr,
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        _NUM_KV_HEADS * _GROUP_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        _PAGE_SIZE,
        causal=True,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        **plan_kwargs,
    )
    return wrapper.run(q, kv_data, return_lse=True)


def _kv_pool(num_pages):
    return torch.randn(
        num_pages,
        2,
        _PAGE_SIZE,
        _NUM_KV_HEADS,
        _HEAD_DIM,
        device="cuda:0",
        dtype=torch.float16,
    )


@pytest.mark.parametrize("kv_len", [3000])
@pytest.mark.parametrize(
    "mate_kv_lens",
    [
        [17, 40, 900, 120, 64, 2000, 33, 5] * 2,
        [6000, 5000, 7000, 4000] * 4,
    ],
    ids=["short_mates", "long_mates"],
)
@pytest.mark.parametrize(
    "plan_kwargs",
    [dict(disable_split_kv=True), dict(fixed_split_size=2048)],
    ids=["disable_split_kv", "fixed_split_size"],
)
def test_batch_decode_tensor_cores_batch_composition(
    kv_len: int,
    mate_kv_lens: list,
    plan_kwargs: dict,
):
    torch.manual_seed(0)
    kv_lens = [kv_len] + mate_kv_lens
    kv_data = _kv_pool(sum((l + _PAGE_SIZE - 1) // _PAGE_SIZE for l in kv_lens))
    q = torch.randn(
        len(kv_lens),
        _NUM_KV_HEADS * _GROUP_SIZE,
        _HEAD_DIM,
        device="cuda:0",
        dtype=torch.float16,
    )

    o_alone, lse_alone = _run_decode([kv_len], q[:1], kv_data, **plan_kwargs)
    o_batched, lse_batched = _run_decode(kv_lens, q, kv_data, **plan_kwargs)

    assert torch.equal(o_batched[:1], o_alone)
    assert torch.equal(lse_batched[:1], lse_alone)


_TILE_RULE_XFAIL = pytest.mark.xfail(
    reason="FA2 picks cta_tile_q from the batch-average query length (#2424)",
    strict=True,
)


@pytest.mark.parametrize("qo_len", [37])
@pytest.mark.parametrize("kv_len", [3000])
@pytest.mark.parametrize(
    "mate_qo_lens, mate_kv_lens",
    [
        pytest.param(
            [1] * 16,
            [500] * 16,
            id="decode_mates",
            marks=_TILE_RULE_XFAIL,
        ),
        pytest.param([64] * 8, [800] * 8, id="mid_mates"),
        pytest.param([1500] * 3, [1500] * 3, id="long_prefill_mates"),
    ],
)
@pytest.mark.parametrize(
    "plan_kwargs",
    [dict(disable_split_kv=True), dict(fixed_split_size=4096, disable_split_kv=True)],
    ids=["disable_split_kv", "vllm_batch_invariant"],
)
def test_batch_prefill_tensor_cores_batch_composition(
    qo_len: int,
    kv_len: int,
    mate_qo_lens: list,
    mate_kv_lens: list,
    plan_kwargs: dict,
):
    torch.manual_seed(0)
    qo_lens = [qo_len] + mate_qo_lens
    kv_lens = [kv_len] + mate_kv_lens
    kv_data = _kv_pool(sum((l + _PAGE_SIZE - 1) // _PAGE_SIZE for l in kv_lens))
    q = torch.randn(
        sum(qo_lens),
        _NUM_KV_HEADS * _GROUP_SIZE,
        _HEAD_DIM,
        device="cuda:0",
        dtype=torch.float16,
    )

    o_alone, lse_alone = _run_prefill(
        [qo_len], [kv_len], q[:qo_len], kv_data, **plan_kwargs
    )
    o_batched, lse_batched = _run_prefill(qo_lens, kv_lens, q, kv_data, **plan_kwargs)

    assert torch.equal(o_batched[:qo_len], o_alone)
    assert torch.equal(lse_batched[:qo_len], lse_alone)


_UNALIGNED_PREFIX_XFAIL = pytest.mark.xfail(
    reason="isolated rows differ by one ulp on SM100 when the cached prefix length is not a multiple of 128",
    strict=False,
)


@pytest.mark.parametrize(
    "kv_len, tail_len",
    [
        pytest.param(1024, 1, marks=_TILE_RULE_XFAIL),
        pytest.param(3700, 1, marks=_TILE_RULE_XFAIL),
        pytest.param(1024, 4, marks=_TILE_RULE_XFAIL),
        pytest.param(3700, 4, marks=_TILE_RULE_XFAIL),
        pytest.param(1024, 16, marks=_TILE_RULE_XFAIL),
        pytest.param(3700, 16, marks=_TILE_RULE_XFAIL),
        pytest.param(1024, 37, marks=_UNALIGNED_PREFIX_XFAIL),
        pytest.param(3700, 37, marks=_UNALIGNED_PREFIX_XFAIL),
        pytest.param(1024, 200, marks=_UNALIGNED_PREFIX_XFAIL),
        pytest.param(3700, 200, marks=_UNALIGNED_PREFIX_XFAIL),
        pytest.param(1024, 256, id="1024-256-aligned_prefix"),
        pytest.param(3700, 116, id="3700-116-aligned_prefix"),
    ],
)
def test_batch_prefill_tensor_cores_prefix_cache_tail(kv_len: int, tail_len: int):
    torch.manual_seed(0)
    kv_data = _kv_pool((kv_len + _PAGE_SIZE - 1) // _PAGE_SIZE)
    q = torch.randn(
        kv_len,
        _NUM_KV_HEADS * _GROUP_SIZE,
        _HEAD_DIM,
        device="cuda:0",
        dtype=torch.float16,
    )

    o_cold, lse_cold = _run_prefill(
        [kv_len], [kv_len], q, kv_data, disable_split_kv=True
    )
    o_tail, lse_tail = _run_prefill(
        [tail_len], [kv_len], q[kv_len - tail_len :], kv_data, disable_split_kv=True
    )

    assert torch.equal(o_cold[kv_len - tail_len :], o_tail)
    assert torch.equal(lse_cold[kv_len - tail_len :], lse_tail)
