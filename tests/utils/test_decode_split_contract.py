"""CPU-only validation of decode split flags at the planner boundary."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flashinfer.decode as decode


def _wrapper(monkeypatch, use_tensor_cores, use_cuda_graph, backend="fa2"):
    # Pinned allocation needs a CUDA driver even with CPU tensors. Keep the
    # real constructor and public methods, but use unpinned CPU buffers here.
    empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    wrapper = decode.BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(1024, dtype=torch.uint8),
        use_tensor_cores=use_tensor_cores,
        use_cuda_graph=use_cuda_graph,
        paged_kv_indptr_buffer=torch.empty(2, dtype=torch.int32),
        paged_kv_indices_buffer=torch.empty(4, dtype=torch.int32),
        paged_kv_last_page_len_buffer=torch.empty(1, dtype=torch.int32),
        backend=backend,
    )
    calls = []

    def plan(*args):
        calls.append(("plan", args))
        return []

    def workspace_size(*args):
        calls.append(("workspace_size", args))
        return 0, 0

    module = SimpleNamespace(plan=plan, workspace_size=workspace_size)
    wrapper._jit_module = module
    wrapper._cached_module = module
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    return wrapper, calls


def _call(wrapper, entrypoint, disable_split_kv, **extra):
    args = (
        torch.tensor([0, 4], dtype=torch.int32),
        torch.arange(4, dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        8,
        2,
        128,
        16,
    )
    kwargs = {"q_data_type": torch.float16, "disable_split_kv": disable_split_kv}
    kwargs.update(extra)
    if entrypoint == "fast_decode_plan":
        return decode.fast_decode_plan(wrapper, *args, **kwargs)
    return getattr(wrapper, entrypoint)(*args, **kwargs)


@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size", "fast_decode_plan"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_cuda_core_decode_rejects_disable_split_kv(
    monkeypatch, entrypoint, use_cuda_graph
):
    wrapper, calls = _wrapper(monkeypatch, False, use_cuda_graph)
    with pytest.raises(ValueError, match="disable_split_kv.*CUDA-core decode"):
        _call(wrapper, entrypoint, True)
    assert not calls, "unsupported option must not reach the compiled planner"


@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size", "fast_decode_plan"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_cuda_core_decode_default_still_reaches_planner(
    monkeypatch, entrypoint, use_cuda_graph
):
    wrapper, calls = _wrapper(monkeypatch, False, use_cuda_graph)
    _call(wrapper, entrypoint, False)
    assert len(calls) == 1


@pytest.mark.parametrize("entrypoint", ["plan", "workspace_size", "fast_decode_plan"])
@pytest.mark.parametrize("disable_split_kv", [False, True])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_tensor_core_decode_preserves_disable_split_kv(
    monkeypatch, entrypoint, disable_split_kv, use_cuda_graph
):
    wrapper, calls = _wrapper(monkeypatch, True, use_cuda_graph)
    _call(wrapper, entrypoint, disable_split_kv)
    assert len(calls) == 1
    # The final FA2 arguments are fixed_split_size, disable_split_kv,
    # num_colocated_ctas, uniform_q_len for both sizing and planning.
    assert calls[0][1][-3] is disable_split_kv


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_rejected_plan_does_not_copy_metadata(monkeypatch, use_cuda_graph):
    wrapper, calls = _wrapper(monkeypatch, False, use_cuda_graph)

    def unexpected_copy(*args, **kwargs):
        pytest.fail("invalid split flag reached metadata transfer")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_copy)
    monkeypatch.setattr(torch.Tensor, "copy_", unexpected_copy)
    with pytest.raises(ValueError, match="disable_split_kv.*CUDA-core decode"):
        _call(wrapper, "plan", True)
    assert not calls


@pytest.mark.parametrize("backend", ["cudnn", "cutile"])
@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_other_backends_are_not_classified_as_cuda_core(
    monkeypatch, backend, use_cuda_graph
):
    wrapper, calls = _wrapper(monkeypatch, False, use_cuda_graph, backend)
    cudnn_calls = []
    monkeypatch.setattr(wrapper, "_plan_cudnn", lambda *args: cudnn_calls.append(args))
    _call(wrapper, "plan", True)
    assert not wrapper.use_tensor_cores
    assert not calls
    assert len(cudnn_calls) == (1 if backend == "cudnn" else 0)


@pytest.mark.parametrize("entrypoint", ["plan", "fast_decode_plan"])
def test_auto_cudnn_resolution_precedes_cuda_core_validation(monkeypatch, entrypoint):
    wrapper, calls = _wrapper(monkeypatch, False, False, "auto")
    monkeypatch.setattr(wrapper, "_auto_prefers_cudnn", lambda *args, **kwargs: True)
    cudnn_calls = []
    monkeypatch.setattr(wrapper, "_plan_cudnn", lambda *args: cudnn_calls.append(args))
    _call(wrapper, entrypoint, True, q_len_per_req=2)
    assert wrapper._cudnn_auto
    assert len(cudnn_calls) == 1
    assert not calls


@pytest.mark.parametrize("entrypoint", ["plan", "fast_decode_plan"])
def test_leaving_auto_cudnn_rechecks_cuda_core_split_flag(monkeypatch, entrypoint):
    wrapper, calls = _wrapper(monkeypatch, False, False, "auto")
    wrapper._cudnn_auto = True
    monkeypatch.setattr(wrapper, "_auto_prefers_cudnn", lambda *args, **kwargs: False)
    with pytest.raises(ValueError, match="disable_split_kv.*CUDA-core decode"):
        _call(wrapper, entrypoint, True)
    assert not wrapper._cudnn_auto
    assert not calls
