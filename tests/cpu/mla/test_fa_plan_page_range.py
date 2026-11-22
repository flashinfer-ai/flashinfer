"""Plan-time KV page-range snapshot of the generated FA2/FA3 MLA backends.

The snapshot feeds the run-time "KV cache does not cover the planned CSR page
indices" check. CPU indices, and device indices outside CUDA graph mode, keep
it. In CUDA graph mode with device indices, the caller may refresh the reserved
kv_indices buffer after plan(), so plan() takes no snapshot and no host-device
sync. CPU only: ``meta`` tensors stand in for device indices, and the native
planner and metadata staging are stubbed.
"""

import pytest
import torch

from flashinfer.mla._batch_mla._backends import _fa_common
from flashinfer.mla._batch_mla._backends.fa3_backend import (
    _BatchMLAPagedAttentionFa3Backend,
)

HEADS, CKV = 16, 512
# Two requests over pages [5, 7] and [2]; the trailing -1 is unused capacity.
KV_INDICES = [5, 7, 2, -1]


class _FakeModule:
    def plan_with_staged_workspace_bytes(self, *args):
        return [0] * 4, 0


def _planned_backend(monkeypatch, *, use_cuda_graph, kv_indices):
    monkeypatch.setattr(_fa_common, "_validate_fa_plan_workload", lambda *args: None)
    buffer = torch.empty(16, dtype=torch.uint8)
    backend = _BatchMLAPagedAttentionFa3Backend(
        float_workspace_buffer=buffer,
        use_cuda_graph=use_cuda_graph,
        qo_indptr_buf=None,
        kv_indptr_buf=None,
        kv_indices_buf=None,
        kv_len_arr_buf=None,
        query_split_widths=(CKV, 0),
        kv_split_widths=(CKV, 0),
        int_workspace_buffer=buffer,
        pin_memory_int_workspace_buffer=buffer,
    )
    monkeypatch.setattr(backend, "_stage_metadata", lambda **kwargs: None)
    backend._plan_generated_fa(
        module_loader=_FakeModule,
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 2, 3], dtype=torch.int32),
        kv_indices=kv_indices,
        kv_len_arr=torch.tensor([2, 1], dtype=torch.int32),
        num_heads=HEADS,
        head_dim_ckv=CKV,
        page_size=1,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        use_profiler=False,
    )
    return backend


def _check(backend, num_pages):
    backend._check_planned_geometry(
        torch.empty(2, HEADS, CKV, dtype=torch.bfloat16),
        torch.empty(num_pages, 1, CKV, dtype=torch.bfloat16),
    )


@pytest.mark.cpu_only
@pytest.mark.parametrize("use_cuda_graph", (False, True))
def test_cpu_indices_keep_page_range_check(monkeypatch, use_cuda_graph):
    backend = _planned_backend(
        monkeypatch,
        use_cuda_graph=use_cuda_graph,
        kv_indices=torch.tensor(KV_INDICES, dtype=torch.int32),
    )
    assert backend._planned_kv_page_range == (2, 8)
    _check(backend, 8)
    with pytest.raises(ValueError, match="page indices"):
        _check(backend, 7)


@pytest.mark.cpu_only
def test_eager_device_indices_keep_page_range_snapshot(monkeypatch):
    calls = []

    def aminmax(tensor):
        calls.append(tensor.device.type)
        return torch.tensor(2), torch.tensor(7)

    monkeypatch.setattr(_fa_common.torch, "aminmax", aminmax)
    backend = _planned_backend(
        monkeypatch,
        use_cuda_graph=False,
        kv_indices=torch.empty(len(KV_INDICES), dtype=torch.int32, device="meta"),
    )
    assert calls == ["meta"]
    assert backend._planned_kv_page_range == (2, 8)
    with pytest.raises(ValueError, match="page indices"):
        _check(backend, 7)


@pytest.mark.cpu_only
def test_graph_device_indices_skip_page_range_snapshot(monkeypatch):
    def aminmax(tensor):
        raise AssertionError("graph-mode device indices must not be reduced")

    monkeypatch.setattr(_fa_common.torch, "aminmax", aminmax)
    backend = _planned_backend(
        monkeypatch,
        use_cuda_graph=True,
        kv_indices=torch.empty(len(KV_INDICES), dtype=torch.int32, device="meta"),
    )
    assert backend._planned_kv_page_range is None
    _check(backend, 1)
