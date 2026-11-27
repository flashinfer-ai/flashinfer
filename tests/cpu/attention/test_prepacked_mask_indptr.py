# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Verify byte-unit mask_indptr computation for prepacked custom masks."""

import pytest
import torch

import flashinfer.prefill as fp
import flashinfer.sparse as fs

pytestmark = pytest.mark.cpu_only


class _StopAfterMaskIndptr(Exception):
    pass


def test_compute_page_mask_indptr_packed_vs_unpacked():
    # page_size=4; reqs have (q_len, kv_len):
    #   req 0: (3, 3) -> 9 bits  -> ceil(9/8)  = 2 bytes
    #   req 1: (2, 8) -> 16 bits -> ceil(16/8) = 2 bytes
    #   req 2: (1, 4) -> 4 bits  -> ceil(4/8)  = 1 byte
    qo_indptr = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([3, 4, 4], dtype=torch.int32)

    bit_indptr = fp._compute_page_mask_indptr(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, page_size=4
    )
    byte_indptr = fp._compute_page_mask_indptr(
        qo_indptr, paged_kv_indptr, paged_kv_last_page_len, page_size=4, packed=True
    )

    assert bit_indptr.dtype == torch.int32
    assert byte_indptr.dtype == torch.int32
    assert bit_indptr.tolist() == [0, 9, 25, 29]
    assert byte_indptr.tolist() == [0, 2, 4, 5]


def test_compute_mask_indptr_packed_vs_unpacked():
    # Reqs have (q_len, kv_len):
    #   req 0: (0, 4) -> 0 bits  -> 0 bytes
    #   req 1: (1, 1) -> 1 bit   -> 1 byte
    #   req 2: (2, 4) -> 8 bits  -> 1 byte
    #   req 3: (3, 3) -> 9 bits  -> 2 bytes
    qo_indptr = torch.tensor([0, 0, 1, 3, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 4, 5, 9, 12], dtype=torch.int32)

    bit_indptr = fp._compute_mask_indptr(qo_indptr, kv_indptr)
    byte_indptr = fp._compute_mask_indptr(qo_indptr, kv_indptr, packed=True)

    assert bit_indptr.dtype == torch.int32
    assert byte_indptr.dtype == torch.int32
    assert bit_indptr.tolist() == [0, 0, 1, 9, 18]
    assert byte_indptr.tolist() == [0, 0, 1, 2, 4]


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_paged_wrapper_prepacked_custom_mask_uses_byte_indptr(
    monkeypatch, use_cuda_graph
):
    w = fp.BatchPrefillWithPagedKVCacheWrapper.__new__(
        fp.BatchPrefillWithPagedKVCacheWrapper
    )
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _batch_invariant=None,
        _requested_backend="fa2",
        _backend="fa2",
        _cuda_graph_backend="fa2" if use_cuda_graph else None,
        _jit_module=None,
        _use_cuda_graph=use_cuda_graph,
        _plan_info=None,
        _variant_owns_mask=False,
        _kv_layout="NHD",
        device=torch.device("cpu"),
        _head_dim_qk=64,
        _head_dim_vo=64,
        _cudnn_q_lens_buffer=None,
        _max_total_num_rows=None,
        _fixed_batch_size=3,
        _qo_indptr_buf=torch.zeros(4, dtype=torch.int32),
        _paged_kv_indptr_buf=torch.zeros(4, dtype=torch.int32),
        _paged_kv_indices_buf=torch.zeros(4, dtype=torch.int32),
        _paged_kv_last_page_len_buf=torch.zeros(3, dtype=torch.int32),
        _custom_mask_buf=torch.zeros(16, dtype=torch.uint8) if use_cuda_graph else None,
        _mask_indptr_buf=torch.zeros(4, dtype=torch.int32) if use_cuda_graph else None,
    )

    def _stop(*args, **kwargs):
        raise _StopAfterMaskIndptr

    monkeypatch.setattr(fp, "get_batch_prefill_module", _stop)

    qo_indptr = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    paged_kv_indptr = torch.tensor([0, 1, 3, 4], dtype=torch.int32)
    paged_kv_indices = torch.arange(4, dtype=torch.int32)
    paged_kv_last_page_len = torch.tensor([3, 4, 4], dtype=torch.int32)
    packed_mask = torch.tensor([0xFF, 0x01, 0xAA, 0x55, 0x0F], dtype=torch.uint8)

    with pytest.raises(_StopAfterMaskIndptr):
        w.plan(
            qo_indptr,
            paged_kv_indptr,
            paged_kv_indices,
            paged_kv_last_page_len,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=64,
            page_size=4,
            packed_custom_mask=packed_mask,
            q_data_type=torch.float16,
            non_blocking=False,
        )

    assert w._mask_indptr_buf.tolist() == [0, 2, 4, 5]
    assert w._custom_mask_buf[: len(packed_mask)].tolist() == packed_mask.tolist()


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_ragged_wrapper_prepacked_custom_mask_uses_byte_indptr(
    monkeypatch, use_cuda_graph
):
    w = fp.BatchPrefillWithRaggedKVCacheWrapper.__new__(
        fp.BatchPrefillWithRaggedKVCacheWrapper
    )
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _batch_invariant=None,
        _requested_backend="fa2",
        _backend="fa2",
        _jit_module=None,
        _use_cuda_graph=use_cuda_graph,
        _plan_info=None,
        _variant_owns_mask=False,
        _kv_layout="NHD",
        device=torch.device("cpu"),
        _head_dim_qk=64,
        _head_dim_vo=64,
        _max_total_num_rows=None,
        _fixed_batch_size=3,
        _qo_indptr_buf=torch.zeros(4, dtype=torch.int32),
        _kv_indptr_buf=torch.zeros(4, dtype=torch.int32),
        _custom_mask_buf=torch.zeros(16, dtype=torch.uint8) if use_cuda_graph else None,
        _mask_indptr_buf=torch.zeros(4, dtype=torch.int32) if use_cuda_graph else None,
    )

    def _stop(*args, **kwargs):
        raise _StopAfterMaskIndptr

    monkeypatch.setattr(fp, "get_batch_prefill_module", _stop)

    qo_indptr = torch.tensor([0, 3, 5, 6], dtype=torch.int32)
    kv_indptr = torch.tensor([0, 3, 11, 15], dtype=torch.int32)
    packed_mask = torch.tensor([0xFF, 0x01, 0xAA, 0x55, 0x0F], dtype=torch.uint8)

    with pytest.raises(_StopAfterMaskIndptr):
        w.plan(
            qo_indptr,
            kv_indptr,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim_qk=64,
            packed_custom_mask=packed_mask,
            q_data_type=torch.float16,
            non_blocking=False,
        )

    assert w._mask_indptr_buf.tolist() == [0, 2, 4, 5]
    assert w._custom_mask_buf[: len(packed_mask)].tolist() == packed_mask.tolist()


def test_block_sparse_wrapper_prepacked_mask_uses_byte_indptr(monkeypatch):
    w = fs.BlockSparseAttentionWrapper.__new__(fs.BlockSparseAttentionWrapper)
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _pin_memory_int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _kv_lens_buffer=torch.zeros(16, dtype=torch.int32),
        _backend="fa2",
        _use_cuda_graph=False,
        _kv_layout="NHD",
        device=torch.device("cpu"),
    )

    def _stop(*args, **kwargs):
        raise _StopAfterMaskIndptr

    monkeypatch.setattr(fs, "get_batch_prefill_module", _stop)

    # M=6, N=6, R=3, C=3 -> 2 block rows:
    #   row 0: 1 block -> 3 * 3 = 9 bits  -> ceil(9/8)  = 2 bytes
    #   row 1: 2 blocks -> 3 * 6 = 18 bits -> ceil(18/8) = 3 bytes
    indptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    indices = torch.tensor([0, 0, 1], dtype=torch.int32)
    packed_mask = torch.tensor([0xFF, 0x01, 0xAA, 0x55, 0x03], dtype=torch.uint8)

    with pytest.raises(_StopAfterMaskIndptr):
        w.plan(
            indptr=indptr,
            indices=indices,
            M=6,
            N=6,
            R=3,
            C=3,
            num_qo_heads=4,
            num_kv_heads=4,
            head_dim=64,
            packed_mask=packed_mask,
            q_data_type=torch.float16,
            non_blocking=False,
        )

    assert w._mask_indptr_buf.tolist() == [0, 2, 5]
    assert w._packed_mask_buf.tolist() == packed_mask.tolist()


def test_no_cuda_initialization():
    assert not torch.cuda.is_initialized()
