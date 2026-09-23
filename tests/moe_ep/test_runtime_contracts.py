"""Host regressions for runtime ownership and non-global EP groups."""

import sys
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch.distributed as dist

from flashinfer.moe_ep import BootstrapConfig
from flashinfer.moe_ep.core import bootstrap_utils
from flashinfer.moe_ep.core.runtime import bootstrap as runtime


@pytest.fixture
def setup_runtime(monkeypatch):
    monkeypatch.setattr(runtime, "_STATE", runtime._RuntimeState())
    monkeypatch.setattr(runtime, "_mega_no_dist", lambda: False)
    nv = ModuleType("nvshmem")
    nv.core = ModuleType("nvshmem.core")
    nv.core.my_pe = lambda: 1
    nv.core.n_pes = lambda: 2
    monkeypatch.setitem(sys.modules, "nvshmem", nv)
    monkeypatch.setitem(sys.modules, "nvshmem.core", nv.core)
    group = object()
    monkeypatch.setattr(bootstrap_utils, "bootstrap_comm_group", lambda b: group)
    monkeypatch.setattr(
        bootstrap_utils, "bootstrap_ep_rank_world", lambda b: (b.rank, b.world_size)
    )
    monkeypatch.setattr(runtime, "_ensure_torch_dist", lambda b: False)
    return nv.core, group, BootstrapConfig(rank=1, world_size=2, device=0)


def test_failed_acquisition_does_not_leak_reference_count(monkeypatch):
    monkeypatch.setattr(runtime, "_STATE", runtime._RuntimeState())
    monkeypatch.setattr(
        runtime, "_ensure_nvshmem", mock.Mock(side_effect=RuntimeError("failed"))
    )
    with pytest.raises(RuntimeError, match="failed"):
        runtime.bootstrap_moe_ep_runtime(
            BootstrapConfig(rank=0, world_size=1), frozenset({runtime.NVSHMEM})
        )
    assert runtime._STATE.ref_count == 0
    assert runtime._STATE.active_requirements == frozenset()


def test_runtime_handles_release_once_and_capture_failure_is_retryable(monkeypatch):
    import torch

    monkeypatch.setattr(runtime, "_STATE", runtime._RuntimeState())
    monkeypatch.setattr(runtime, "_ensure_torch_dist", lambda b: False)
    bootstrap = BootstrapConfig(rank=0, world_size=1)
    first = runtime.bootstrap_moe_ep_runtime(bootstrap, frozenset({runtime.TORCH_DIST}))
    second = runtime.bootstrap_moe_ep_runtime(
        bootstrap, frozenset({runtime.TORCH_DIST})
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(RuntimeError, match="graph capture"):
        runtime.finalize_moe_ep_runtime(first)
    assert not first.closed and runtime._STATE.ref_count == 2
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    runtime.finalize_moe_ep_runtime(first)
    runtime.finalize_moe_ep_runtime(first)
    assert runtime._STATE.ref_count == 1
    runtime.finalize_moe_ep_runtime(second)
    assert runtime._STATE.ref_count == 0


def test_existing_nvshmem_requires_matching_pe_identity(setup_runtime, monkeypatch):
    nv, _, bootstrap = setup_runtime
    monkeypatch.setattr(runtime, "_nvshmem_initialized", lambda: True)
    assert runtime._ensure_nvshmem(bootstrap) == (False, False)
    nv.my_pe = lambda: 0
    with pytest.raises(RuntimeError, match="PE rank/count"):
        runtime._ensure_nvshmem(bootstrap)


def test_existing_nvshmem_rejects_group_and_device_changes(setup_runtime, monkeypatch):
    _, _, bootstrap = setup_runtime
    monkeypatch.setattr(runtime, "_nvshmem_initialized", lambda: True)
    runtime._ensure_nvshmem(bootstrap)
    with pytest.raises(RuntimeError, match="group or CUDA device"):
        runtime._ensure_nvshmem(BootstrapConfig(rank=1, world_size=2, device=1))
    monkeypatch.setattr(bootstrap_utils, "bootstrap_comm_group", lambda b: object())
    with pytest.raises(RuntimeError, match="group or CUDA device"):
        runtime._ensure_nvshmem(bootstrap)


def test_uid_broadcast_uses_global_rank_of_group_local_zero(setup_runtime, monkeypatch):
    import torch

    nv, group, bootstrap = setup_runtime
    nv.get_unique_id = mock.Mock(
        return_value=SimpleNamespace(_data=np.zeros(128, dtype=np.uint8))
    )
    nv.init = mock.Mock()
    cuda_core = ModuleType("cuda.core.experimental")
    cuda_core.Device = lambda device: SimpleNamespace(set_current=lambda: None)
    monkeypatch.setitem(sys.modules, "cuda.core.experimental", cuda_core)
    monkeypatch.setattr(runtime, "_nvshmem_initialized", lambda: False)
    monkeypatch.setattr(runtime, "_single_gpu_gloo", lambda: True)
    monkeypatch.setattr(torch.cuda, "set_device", lambda d: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    source = mock.Mock(return_value=4)
    broadcast = mock.Mock()
    monkeypatch.setattr(dist, "get_global_rank", source)
    monkeypatch.setattr(dist, "broadcast", broadcast)
    monkeypatch.setattr(dist, "barrier", mock.Mock())
    assert runtime._init_nvshmem_after_dist(bootstrap)
    source.assert_called_once_with(group, 0)
    assert broadcast.call_args.kwargs == dict(src=4, group=group)
    assert nv.init.call_args.kwargs["rank"] == 1
    assert nv.init.call_args.kwargs["nranks"] == 2


def test_free_uses_allocation_kind_even_if_environment_changes(
    setup_runtime, monkeypatch
):
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe.shim import comm

    nv, _, _ = setup_runtime
    root = object()
    tensor = SimpleNamespace(_mega_sym_root=root)
    nv.free_tensor = mock.Mock()
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    comm.free_sym_tensor(tensor)
    nv.free_tensor.assert_called_once_with(root)
