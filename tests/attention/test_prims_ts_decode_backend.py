# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Policy tests for PrimTS and the legacy CSR paged-decode wrapper.

PrimTS decode consumes a dense page table directly. The legacy
``BatchDecodeWithPagedKVCacheWrapper`` remains CSR-shaped and therefore does
not expose a PrimTS backend. Numerical and CUDA-graph coverage for the direct
dense interface lives in ``test_attention_ts_decode.py``.
"""

import pytest
import torch

import flashinfer


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires a CUDA device"
)


def _make_page_table(kv_lens, page_size, device):
    pages_per_req = [(kv_len + page_size - 1) // page_size for kv_len in kv_lens]
    offsets = [0]
    for num_pages in pages_per_req:
        offsets.append(offsets[-1] + num_pages)
    indptr = torch.tensor(offsets, dtype=torch.int32, device=device)
    indices = torch.arange(sum(pages_per_req), dtype=torch.int32, device=device)
    last_page_len = torch.tensor(
        [
            kv_len - (num_pages - 1) * page_size
            for kv_len, num_pages in zip(kv_lens, pages_per_req, strict=True)
        ],
        dtype=torch.int32,
        device=device,
    )
    return indptr, indices, last_page_len


def _make_wrapper(backend, kv_layout="HND", device="cuda", **kwargs):
    workspace = torch.zeros(64 * 1024 * 1024, dtype=torch.uint8, device=device)
    return flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, kv_layout, backend=backend, **kwargs
    )


@requires_cuda
def test_legacy_csr_wrapper_rejects_prims_ts_backend():
    with pytest.raises(
        NotImplementedError,
        match=r"dense.*BatchDecodePagedTSWrapper|BatchDecodePagedTSWrapper.*dense",
    ):
        _make_wrapper("prims-ts")


def test_plan_trace_captures_explicit_causal_mode():
    plan_trace = flashinfer.BatchDecodeWithPagedKVCacheWrapper.plan.fi_trace
    indptr, indices, last_page_len = _make_page_table([2, 3], 16, "cpu")
    definition = plan_trace(
        indptr=indptr,
        indices=indices,
        last_page_len=last_page_len,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=16,
        kwargs={"q_len_per_req": 4, "is_causal": False},
    )

    assert definition["op_type"] == "gqa_paged_plan"
    assert definition["inputs"]["q_len_per_req"]["optional"] is True
    assert definition["inputs"]["is_causal"] == {
        "shape": None,
        "dtype": "bool",
        "optional": True,
        "description": "Whether the planned attention mask is causal.",
    }


@requires_cuda
@pytest.mark.parametrize("q_len_per_req,is_causal", [(4, False), (1, True)])
def test_explicit_is_causal_rejected_by_legacy_backends(q_len_per_req, is_causal):
    wrapper = _make_wrapper("fa2", kv_layout="NHD")
    indptr, indices, last_page_len = _make_page_table([64, 96], 16, "cuda")
    with pytest.raises(NotImplementedError, match="is_causal"):
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            8,
            2,
            128,
            16,
            q_data_type=torch.bfloat16,
            q_len_per_req=q_len_per_req,
            is_causal=is_causal,
        )
