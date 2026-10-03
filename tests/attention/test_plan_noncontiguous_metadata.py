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

Regression tests for https://github.com/flashinfer-ai/flashinfer/issues/5260:
``plan()`` and ``workspace_size()`` must accept non-contiguous (strided) index
tensors, including ``seq_lens``. Before the fix,
strided views were handed to the planner / kernels as raw pointers, causing
illegal memory accesses, planner errors, or silently wrong output.
"""

import pytest
import torch

import flashinfer
from flashinfer.utils import is_sm90a_supported

NUM_QO_HEADS, NUM_KV_HEADS, HEAD_DIM, PAGE_SIZE = 8, 2, 128, 16
DTYPE = torch.float16
QO_LENS = [17, 1, 64, 33]
KV_LENS = [40, 90, 64, 200]


def _strided(t: torch.Tensor) -> torch.Tensor:
    """Same values as ``t`` but as a stride-2 view with garbage in the gaps."""
    buf = torch.full((2 * t.numel(),), 12345, dtype=t.dtype, device=t.device)
    buf[::2] = t
    view = buf[::2]
    assert not view.is_contiguous() and torch.equal(view, t)
    return view


def _indptr(lens) -> torch.Tensor:
    return torch.tensor([0] + lens, dtype=torch.int32).cumsum(0, dtype=torch.int32)


def _workspace() -> torch.Tensor:
    return torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")


def _paged_metadata():
    pages = [(n + PAGE_SIZE - 1) // PAGE_SIZE for n in KV_LENS]
    indptr = _indptr(pages)
    total = int(indptr[-1])
    # Non-trivial page order and a few unused pages.
    indices = torch.randperm(total + 7)[:total].to(torch.int32)
    last_page_len = torch.tensor(
        [(n - 1) % PAGE_SIZE + 1 for n in KV_LENS], dtype=torch.int32
    )
    kv_cache = torch.randn(
        total + 7, 2, PAGE_SIZE, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    return indptr, indices, last_page_len, kv_cache


def _skip_unsupported(backend: str):
    if backend == "fa3" and not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("fa3 requires SM90a")


def _assert_same(ref: torch.Tensor, out: torch.Tensor):
    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("metadata_device", ["cuda", "cpu"])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_ragged_prefill_plan_noncontiguous_indptr(metadata_device, backend):
    _skip_unsupported(backend)
    torch.manual_seed(0)
    qo_indptr, kv_indptr = _indptr(QO_LENS), _indptr(KV_LENS)
    q = torch.randn(
        int(qo_indptr[-1]), NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    k = torch.randn(
        int(kv_indptr[-1]), NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )
    v = torch.randn_like(k)

    outs = []
    for make in (lambda t: t, _strided):
        wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            _workspace(), "NHD", backend=backend
        )
        wrapper.plan(
            make(qo_indptr.to(metadata_device)),
            make(kv_indptr.to(metadata_device)),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            causal=True,
            q_data_type=DTYPE,
        )
        outs.append(wrapper.run(q, k, v))
    _assert_same(*outs)


@pytest.mark.parametrize("pass_seq_lens", [False, True])
@pytest.mark.parametrize("metadata_device", ["cuda", "cpu"])
@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_paged_prefill_plan_noncontiguous_metadata(
    metadata_device, backend, pass_seq_lens
):
    _skip_unsupported(backend)
    torch.manual_seed(0)
    kv_indptr, kv_indices, last_page_len, kv_cache = _paged_metadata()
    qo_indptr = _indptr(QO_LENS)
    kv_lens = torch.tensor(KV_LENS, dtype=torch.int32)
    q = torch.randn(
        int(qo_indptr[-1]), NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda"
    )

    outs = []
    for make in (lambda t: t, _strided):
        extra = {"seq_lens": make(kv_lens.to(metadata_device))} if pass_seq_lens else {}
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            _workspace(), "NHD", backend=backend
        )
        wrapper.plan(
            make(qo_indptr.to(metadata_device)),
            make(kv_indptr.to(metadata_device)),
            make(kv_indices.to(metadata_device)),
            make(last_page_len.to(metadata_device)),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            causal=True,
            q_data_type=DTYPE,
            **extra,
        )
        outs.append(wrapper.run(q, kv_cache))
    _assert_same(*outs)


@pytest.mark.parametrize("pass_seq_lens", [False, True])
@pytest.mark.parametrize("metadata_device", ["cuda", "cpu"])
@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_paged_decode_plan_noncontiguous_metadata(
    metadata_device, use_tensor_cores, pass_seq_lens
):
    torch.manual_seed(0)
    kv_indptr, kv_indices, last_page_len, kv_cache = _paged_metadata()
    kv_lens = torch.tensor(KV_LENS, dtype=torch.int32)
    q = torch.randn(len(KV_LENS), NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")

    outs = []
    for make in (lambda t: t, _strided):
        extra = {"seq_lens": make(kv_lens.to(metadata_device))} if pass_seq_lens else {}
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            _workspace(), "NHD", use_tensor_cores=use_tensor_cores
        )
        wrapper.plan(
            make(kv_indptr.to(metadata_device)),
            make(kv_indices.to(metadata_device)),
            make(last_page_len.to(metadata_device)),
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
            PAGE_SIZE,
            q_data_type=DTYPE,
            **extra,
        )
        outs.append(wrapper.run(q, kv_cache))
    _assert_same(*outs)


@pytest.mark.parametrize("metadata_device", ["cuda", "cpu"])
@pytest.mark.parametrize(
    "wrapper_kind", ["decode", "decode_tensor_cores", "prefill_fa2"]
)
def test_workspace_size_noncontiguous_metadata(metadata_device, wrapper_kind):
    """workspace_size() must size the same problem regardless of strides.

    (The prefill fa3 backend has no workspace_size().)
    """
    kv_indptr, kv_indices, last_page_len, _ = _paged_metadata()
    qo_indptr = _indptr(QO_LENS)
    kv_lens = torch.tensor(KV_LENS, dtype=torch.int32)

    sizes = []
    for make in (lambda t: t, _strided):
        meta = [
            make(t.to(metadata_device)) for t in (kv_indptr, kv_indices, last_page_len)
        ]
        seq_lens = make(kv_lens.to(metadata_device))
        if wrapper_kind.startswith("decode"):
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                _workspace(),
                "NHD",
                use_tensor_cores=wrapper_kind == "decode_tensor_cores",
            )
            args = meta
        else:
            wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
                _workspace(), "NHD", backend=wrapper_kind.split("_")[1]
            )
            args = [make(qo_indptr.to(metadata_device)), *meta]
        sizes.append(
            wrapper.workspace_size(
                *args,
                NUM_QO_HEADS,
                NUM_KV_HEADS,
                HEAD_DIM,
                PAGE_SIZE,
                q_data_type=DTYPE,
                seq_lens=seq_lens,
            )
        )
    assert sizes[0] == sizes[1]


@pytest.mark.parametrize("metadata_device", ["cuda", "cpu"])
def test_block_sparse_plan_noncontiguous_metadata(metadata_device):
    torch.manual_seed(0)
    M, N, R, C = 256, 512, 16, 16
    mask = torch.rand(M // R, N // C) < 0.3
    mask[:, 0] = True  # every block row attends to something
    indptr = _indptr(mask.sum(1).tolist())
    indices = mask.nonzero()[:, 1].to(torch.int32)
    q = torch.randn(M, NUM_QO_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    k = torch.randn(N, NUM_KV_HEADS, HEAD_DIM, dtype=DTYPE, device="cuda")
    v = torch.randn_like(k)

    outs = []
    for make in (lambda t: t, _strided):
        wrapper = flashinfer.sparse.BlockSparseAttentionWrapper(_workspace())
        wrapper.plan(
            make(indptr.to(metadata_device)),
            make(indices.to(metadata_device)),
            M,
            N,
            R,
            C,
            NUM_QO_HEADS,
            NUM_KV_HEADS,
            HEAD_DIM,
        )
        outs.append(wrapper.run(q, k, v))
    _assert_same(*outs)
