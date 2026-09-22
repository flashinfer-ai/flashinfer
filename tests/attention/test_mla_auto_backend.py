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

"""Policy, planning and execution contracts for automatic MLA backend selection.

Adapter expansion tests intentionally use explicit backends: policy fallback
must not conceal an unsupported adapter or a kernel with the wrong mask.
"""

import math
import warnings
from types import SimpleNamespace

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.mla._batch_mla import _auto_policy, _wrapper
from flashinfer.mla._batch_mla._auto_policy import ordered_sm100_backends
from flashinfer.mla._batch_mla._backends._capabilities import (
    _BackendPlanUnsupportedError,
    MLAPlanCapabilities,
)


# Policy: representative routing decisions and metadata contracts.


CANDIDATES = {
    "cute-dsl-monolithic",
    "trtllm-gen",
    "fa2",
    "cute-dsl-modular",
    "cutile",
    "cutlass",
}


def _request(q_lens, kv_lens, *, heads=32, graph=False, lse_mode="none"):
    offsets = [0]
    for length in q_lens:
        offsets.append(offsets[-1] + length)
    csr = SimpleNamespace(
        qo_indptr=torch.tensor(offsets, dtype=torch.int32),
        kv_len_arr=torch.tensor(kv_lens, dtype=torch.int32),
    )
    args = SimpleNamespace(
        csr=lambda: csr, num_heads=heads, _use_cuda_graph=graph, lse_mode=lse_mode
    )
    return args, csr


def _order(args):
    order = ordered_sm100_backends(args)
    assert len(order) == len(set(order))
    assert set(order) == CANDIDATES
    assert ordered_sm100_backends(args) == order
    return order


@pytest.mark.parametrize(
    "q_lens,kv_lens,heads,graph,lse_mode,expected",
    [
        ((1,), (512,), 32, False, "none", "fa2"),
        ((384,), (384,), 32, False, "none", "cute-dsl-monolithic"),
        ((8,), (8192,), 32, False, "none", "fa2"),
        ((1,), (65537,), 32, False, "none", "cute-dsl-monolithic"),
        ((4,), (512,), 32, True, "none", "trtllm-gen"),
        ((4,), (513,), 32, True, "none", "cute-dsl-monolithic"),
        ((3, 4), (512,) * 2, 32, True, "none", "trtllm-gen"),
        ((32,) * 32, (32,) * 32, 32, False, "none", "fa2"),
        ((32,) * 33, (32,) * 33, 32, False, "none", "cute-dsl-monolithic"),
        ((1,) * 256, (1,) * 256, 128, False, "none", "fa2"),
        ((1,) * 257, (1,) * 257, 128, False, "none", "trtllm-gen"),
        ((1,), (516,), 128, False, "none", "fa2"),
        ((1,), (1032,), 16, False, "none", "fa2"),
        ((2,), (1026,), 128, False, "none", "fa2"),
        ((1,) * 4, (2048,) * 4, 32, False, "none", "fa2"),
        ((8,) * 2, (2048,) * 2, 32, False, "none", "fa2"),
        ((8,) * 4, (2048,) * 4, 32, False, "none", "fa2"),
        ((8,) * 5, (1024,) * 5, 32, False, "none", "fa2"),
        ((9,), (2048,), 32, False, "none", "fa2"),
        ((8,), (2049,), 32, False, "none", "fa2"),
        ((8,) * 4, (2048,) * 4, 64, False, "none", "fa2"),
        ((1, 7, 8), (513, 1024, 2048), 32, False, "none", "fa2"),
        ((128,) * 2, (256,) * 2, 32, True, "base2", "fa2"),
        ((128,) * 2, (256,) * 2, 32, True, "none", "cute-dsl-monolithic"),
        ((128,) * 2, (256,) * 2, 32, True, "basee", "cute-dsl-monolithic"),
        ((239,) * 2, (512,) * 2, 16, False, "base2", "fa2"),
        ((239,) * 2, (512,) * 2, 16, True, "base2", "cute-dsl-monolithic"),
        ((240,) * 2, (512,) * 2, 16, False, "base2", "fa2"),
        ((240,) * 2, (512,) * 2, 16, True, "base2", "fa2"),
        ((257,), (1024,), 8, False, "base2", "fa2"),
        ((257,), (1024,), 8, True, "base2", "fa2"),
        ((257,), (1024,), 9, False, "base2", "fa2"),
        ((257,), (1024,), 9, True, "base2", "cute-dsl-monolithic"),
        ((256,) * 2, (512,) * 2, 32, False, "base2", "fa2"),
        ((256,) * 2, (512,) * 2, 32, True, "base2", "fa2"),
        ((257,) * 2, (512,) * 2, 32, False, "base2", "cute-dsl-monolithic"),
        ((257,) * 2, (512,) * 2, 32, True, "base2", "cute-dsl-monolithic"),
        ((247,), (1024,), 8, True, "base2", "cute-dsl-monolithic"),
        ((248,), (1024,), 8, True, "base2", "fa2"),
        ((257,), (1025,), 8, True, "base2", "cute-dsl-monolithic"),
        ((2,) * 128, (2,) * 128, 128, True, "base2", "fa2"),
        ((2,) * 129, (2,) * 129, 128, True, "base2", "trtllm-gen"),
        ((9,), (4095,), 16, False, "none", "fa2"),
        ((9,), (4096,), 16, False, "none", "fa2"),
        ((9,), (4097,), 16, False, "none", "fa2"),
        ((1,), (4096,), 16, True, "none", "trtllm-gen"),
        ((9,), (4096,), 16, True, "none", "cute-dsl-monolithic"),
        ((1,), (2047,), 128, True, "none", "trtllm-gen"),
        ((1,), (2048,), 128, True, "none", "cute-dsl-monolithic"),
        ((17,), (4096,), 128, True, "none", "trtllm-gen"),
        ((8,) * 2, (4096,) * 2, 128, True, "none", "trtllm-gen"),
        ((8,) * 3, (4096,) * 3, 128, True, "none", "trtllm-gen"),
        ((8,) * 2, (2047, 4096), 128, True, "none", "trtllm-gen"),
        ((8,) * 2, (2048, 4096), 128, True, "none", "cute-dsl-monolithic"),
        ((1,), (16384,), 8, False, "none", "fa2"),
        ((1,), (24576,), 128, False, "none", "fa2"),
        ((1,), (32768,), 8, False, "none", "trtllm-gen"),
        ((9,) * 2, (2053,) * 2, 64, False, "none", "fa2"),
        ((9,) * 2, (4096,) * 2, 64, False, "none", "cute-dsl-monolithic"),
        ((17,), (2048,), 128, False, "none", "fa2"),
        ((9,), (4097,), 128, False, "none", "fa2"),
        ((513,), (768,), 8, False, "none", "fa2"),
        ((129,) * 2, (768,) * 2, 16, False, "none", "fa2"),
        ((702,), (702,), 8, False, "none", "fa2"),
        ((1,) * 24, (3072,) * 24, 64, False, "none", "fa2"),
        ((1,) * 32, (2048,) * 32, 32, False, "none", "fa2"),
        ((1,) * 64, (2048,) * 64, 32, False, "none", "cute-dsl-monolithic"),
        ((1,) * 24, (1536,) * 24, 128, False, "none", "fa2"),
        ((1,) * 17, (2048,) * 17, 128, False, "none", "fa2"),
        ((1,) * 32, (2048,) * 32, 64, False, "none", "fa2"),
        ((1,) * 32, (1024,) * 32, 128, False, "none", "fa2"),
        ((1,) * 32, (526,) * 32, 128, False, "none", "fa2"),
        ((1,) * 32, (2048,) * 32, 128, False, "none", "trtllm-gen"),
        ((1,) * 32, (3072,) * 32, 32, False, "none", "cute-dsl-monolithic"),
        ((9,), (512,), 64, True, "none", "trtllm-gen"),
        ((10,), (512,), 64, True, "none", "cute-dsl-monolithic"),
        ((1,), (65536,), 128, True, "none", "trtllm-gen"),
        ((4,), (16384,), 128, True, "none", "cute-dsl-monolithic"),
        ((8,), (8191,), 128, True, "none", "cute-dsl-monolithic"),
        ((8,), (4097,), 128, True, "none", "cute-dsl-monolithic"),
        ((12,), (4096,), 128, True, "none", "cute-dsl-monolithic"),
        ((1,) * 12, (2048,) * 12, 32, True, "none", "trtllm-gen"),
        ((1,) * 13, (2048,) * 13, 32, True, "none", "cute-dsl-monolithic"),
        ((1,) * 19, (4096,) * 19, 64, False, "none", "fa2"),
        ((1,) * 20, (4096,) * 20, 64, False, "none", "cute-dsl-monolithic"),
        ((4,), (22528,), 64, False, "none", "cute-dsl-monolithic"),
        ((4,), (10240,), 128, False, "none", "trtllm-gen"),
        ((4,), (10241,), 128, False, "none", "trtllm-gen"),
        ((10,), (4096,), 128, False, "none", "trtllm-gen"),
        ((1,), (2048,), 16, True, "none", "trtllm-gen"),
        ((1,), (2048,), 17, True, "none", "cute-dsl-monolithic"),
        ((1,), (2048,), 32, True, "none", "trtllm-gen"),
        ((1,), (512,), 24, True, "none", "trtllm-gen"),
        ((1,), (513,), 24, True, "none", "cute-dsl-monolithic"),
        ((1,), (2048,), 24, True, "none", "cute-dsl-monolithic"),
        ((1,), (2049,), 24, True, "none", "trtllm-gen"),
        ((1,) * 2, (512, 2048), 24, True, "none", "trtllm-gen"),
        ((1,) * 2, (513, 2048), 24, True, "none", "cute-dsl-monolithic"),
        ((16,) * 24, (32,) * 24, 64, True, "none", "trtllm-gen"),
        ((24,) * 15, (32,) * 15, 64, True, "none", "cute-dsl-monolithic"),
        ((33,) * 12, (32,) * 12, 64, True, "none", "cute-dsl-monolithic"),
        ((24,) * 32, (32,) * 32, 32, True, "none", "cute-dsl-monolithic"),
        ((15,) * 32, (32,) * 32, 64, True, "none", "cute-dsl-monolithic"),
        ((16,) * 24, (33,) * 24, 64, True, "none", "cute-dsl-monolithic"),
        ((1,) * 95, (512,) * 95, 64, True, "none", "cute-dsl-monolithic"),
        ((1,) * 96, (512,) * 96, 64, True, "none", "trtllm-gen"),
        ((1,) * 96, (513,) * 96, 64, True, "none", "cute-dsl-monolithic"),
        ((1,) * 96, (512,) * 96, 63, True, "none", "cute-dsl-monolithic"),
        ((2,) * 48, (512,) * 48, 64, True, "none", "cute-dsl-monolithic"),
        ((1,) * 95 + (0,) * 33, (512,) * 128, 64, True, "none", "cute-dsl-monolithic"),
        ((1,) * 96 + (0,) * 32, (512,) * 128, 64, True, "none", "trtllm-gen"),
        ((1,) * 96, (1,) * 96, 1024, True, "none", "cute-dsl-monolithic"),
        ((2,) * 48, (32,) * 48, 64, True, "base2", "fa2"),
        ((1,) * 96, (512,) * 96, 64, False, "none", "fa2"),
        ((4,) * 5, (16384, 16384, 16384, 16384, 16383), 8, False, "none", "fa2"),
        ((4,) * 5, (16384,) * 5, 8, False, "none", "cute-dsl-monolithic"),
        ((1,) * 20, (4096,) * 19 + (4095,), 32, False, "none", "fa2"),
        ((1,) * 20, (4096,) * 20, 32, False, "none", "cute-dsl-monolithic"),
        ((1,) * 22, (4096,) * 22, 48, False, "none", "cute-dsl-monolithic"),
        ((1,) * 128, (1024,) * 128, 16, False, "none", "fa2"),
        ((2, 1, 1, 1, 1, 1), (16384,) * 6, 16, False, "none", "trtllm-gen"),
        ((56,) * 320, (256,) * 320, 1, False, "base2", "fa2"),
        (
            (4,) * 5,
            (16384, 16384, 16384, 16384, 16383),
            8,
            True,
            "none",
            "cute-dsl-monolithic",
        ),
        ((2,) * 5, (16384, 16384, 16384, 16384, 16383), 16, False, "none", "fa2"),
        ((2,) * 5, (16384,) * 5, 16, False, "none", "cute-dsl-monolithic"),
        ((1,) * 5, (16384,) * 5, 16, False, "none", "trtllm-gen"),
        ((2,) * 5, (16384,) * 5, 16, True, "none", "cute-dsl-monolithic"),
        ((256,) * 4, (256,) * 4, 64, True, "none", "trtllm-gen"),
        ((256, 256, 256, 255), (256,) * 4, 64, True, "none", "cute-dsl-monolithic"),
        ((128,) * 8, (256,) * 8, 64, True, "none", "trtllm-gen"),
        ((127,) * 9, (256,) * 9, 64, True, "none", "cute-dsl-monolithic"),
        ((256,) * 5, (256,) * 5, 63, True, "none", "cute-dsl-monolithic"),
        ((256,) * 4, (512,) * 4, 64, True, "none", "trtllm-gen"),
        ((256,) * 4, (513,) * 4, 64, True, "none", "cute-dsl-monolithic"),
        ((256,) * 2, (256,) * 2, 64, True, "base2", "fa2"),
        ((256,) * 4, (256,) * 4, 64, False, "none", "cute-dsl-monolithic"),
    ],
)
def test_preferred_backend_for_regression_regions(
    q_lens, kv_lens, heads, graph, lse_mode, expected
):
    args, _ = _request(q_lens, kv_lens, heads=heads, graph=graph, lse_mode=lse_mode)
    assert _order(args)[0] == expected


@pytest.mark.parametrize("failure_site", ["resolution", "transfer"])
def test_metadata_errors_propagate_without_silent_policy_fallback(failure_site):
    args, csr = _request((1,), (512,))
    error = RuntimeError("metadata transfer or resolution failed")

    def fail():
        raise error

    if failure_site == "resolution":
        args.csr = fail
    else:
        csr.kv_len_arr = SimpleNamespace(cpu=fail)
    with pytest.raises(RuntimeError) as caught:
        ordered_sm100_backends(args)
    assert caught.value is error


def test_policy_rereads_mutated_metadata():
    args, csr = _request((8,), (8192,))

    class ObservedMetadata:
        def __init__(self, tensor):
            self.tensor = tensor
            self.reads = []

        def cpu(self):
            return self

        def tolist(self):
            values = self.tensor.tolist()
            self.reads.append(values)
            return values

    offsets = ObservedMetadata(csr.qo_indptr)
    lengths = ObservedMetadata(csr.kv_len_arr)
    csr.qo_indptr, csr.kv_len_arr = offsets, lengths
    ordered_sm100_backends(args)
    assert offsets.tensor.tolist() in offsets.reads
    assert lengths.tensor.tolist() in lengths.reads

    # Mutate the same storage. Require fresh values without pinning a backend
    # preference, threshold, or exact number of metadata reads.
    offsets.tensor[1] = 24
    lengths.tensor[0] = 2048
    offsets.reads.clear()
    lengths.reads.clear()
    ordered_sm100_backends(args)
    assert offsets.tensor.tolist() in offsets.reads
    assert lengths.tensor.tolist() in lengths.reads


# Planning: public-wrapper routing, error handling and transactional replans.


_SM100_BACKENDS = (
    "fa2",
    "cutlass",
    "cutile",
    "trtllm-gen",
    "cute-dsl-monolithic",
    "cute-dsl-modular",
)


def _plan_buffers(args):
    return tuple(
        buffer
        for buffer in (
            args._float_workspace_buffer,
            args._qo_indptr_buf,
            args._kv_indptr_buf,
            args._kv_indices_buf,
            args._kv_len_arr_buf,
            args._graph_plan_int_workspace_buffer,
        )
        if buffer is not None
    )


@pytest.fixture
def _cpu_planners(monkeypatch):
    """Real wrapper/metadata on CPU; only backend implementations are fakes."""
    from flashinfer.mla._batch_mla import _wrapper
    import flashinfer.utils as utils

    state = SimpleNamespace(calls=[], forbidden=False, handler=lambda name, args: None)
    monkeypatch.setattr(utils, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_wrapper, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(_wrapper, "_get_compute_capability", lambda device: (10, 0))

    def legacy_selection(device):
        assert not state.forbidden, "run invoked legacy backend selection"
        return "fa2"

    monkeypatch.setattr(_auto_policy, "determine_mla_backend", legacy_selection)
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda device: (10, 0))
    # Warnings are separately covered by test_mla_auto_backend_warning.py.
    monkeypatch.setattr(
        _auto_policy._BatchMLAPagedAttentionAutoBackend,
        "_blackwell_auto_fallback_warned",
        True,
    )

    def backend_type(name):
        class Backend:
            _backend = name
            _plan_capabilities = MLAPlanCapabilities(
                backend_name=name,
                lse_modes=frozenset({"none"}),
                kv_layouts=frozenset({"combined"}),
                output_scales=frozenset({"none"}),
                scale_modes=frozenset({"default"}),
                supports_cuda_graph_replan=True,
                requires_packed_query=True,
                requires_packed_kv_cache=True,
            )

            @classmethod
            def preflight_plan_from_wrapper(cls, args):
                assert not state.forbidden, "run invoked backend preflight"
                args.csr()  # Validate actual public metadata, rather than a mock sentinel.
                if name not in (*_SM100_BACKENDS, "fa3"):
                    raise _BackendPlanUnsupportedError(
                        f"{name}: hardware or alias exclusion"
                    )

            @classmethod
            def plan_from_wrapper(cls, args):
                assert not state.forbidden, "run invoked backend planning"
                cls.preflight_plan_from_wrapper(args)
                state.calls.append(name)
                state.handler(name, args)
                result = cls()
                result.buffers = _plan_buffers(args)
                result._cached_module = object()
                result._int_workspace_buffer = torch.full((64,), 17, dtype=torch.uint8)
                result._pin_memory_int_workspace_buffer = torch.full(
                    (64,), 19, dtype=torch.uint8
                )
                result._staged_int_workspace_bytes = 7
                return result

            def run_from_wrapper(self, *, out, **kwargs):
                # Output depends on persistent storage: rolling back object references
                # alone cannot hide corruption of the previous executable's buffers.
                checksum = sum(int(buffer.flatten()[0]) for buffer in self.buffers)
                return out.fill_(checksum + int(self._int_workspace_buffer[0]))

        return Backend

    monkeypatch.setattr(
        _wrapper,
        "_BACKEND_TYPES",
        {
            name: planner if name == "auto" else backend_type(name)
            for name, planner in _wrapper._BACKEND_TYPES.items()
        },
    )
    state.module = _wrapper
    return state


def _cpu_request(backend="auto", *, graph=False):
    metadata = MLAPlanMetadata.csr(
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        kv_indices=torch.tensor([0, 1], dtype=torch.int32),
        kv_len_arr=torch.tensor([1, 1], dtype=torch.int32),
    )
    buffers = [torch.full((64,), 3, dtype=torch.uint8)]
    buffers.extend(
        value.clone()
        for value in (
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
        )
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        buffers[0],
        backend=backend,
        use_cuda_graph=graph,
        qo_indptr=buffers[1],
        kv_indptr=buffers[2],
        kv_indices=buffers[3],
        kv_len_arr=buffers[4],
    )
    # Intentionally outside the measured benchmark grid: support comes from
    # the backend's probe, not from membership in a policy training manifest.
    kwargs = dict(
        metadata=metadata,
        num_heads=3,
        head_dim_ckv=4,
        head_dim_kpe=2,
        page_size=1,
        causal=False,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    return wrapper, kwargs, buffers


def _cpu_run(wrapper):
    out = torch.empty((2, 3, 4), dtype=torch.bfloat16)
    result = wrapper.run(
        query=torch.ones((2, 3, 6), dtype=torch.bfloat16),
        kv_cache=torch.ones((2, 1, 6), dtype=torch.bfloat16),
        out=out,
    )
    assert result is out
    return out


def _reject(name, args):
    raise _BackendPlanUnsupportedError(f"{name}: deliberate support rejection")


def test_cpu_auto_defers_backend_resolution_until_plan(_cpu_planners, monkeypatch):
    def premature_selection(device):
        pytest.fail(
            "auto resolved the legacy backend before request facts were available"
        )

    monkeypatch.setattr(_auto_policy, "determine_mla_backend", premature_selection)
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", premature_selection)
    wrapper, _, _ = _cpu_request()
    assert wrapper._backend == "auto"
    assert wrapper._planned_backend is None
    assert wrapper._planned_backend_name is None
    assert not _cpu_planners.calls


@pytest.mark.parametrize("target", _SM100_BACKENDS)
def test_cpu_auto_reaches_every_supported_backend(_cpu_planners, target):
    def only_target(name, args):
        if name != target:
            _reject(name, args)

    _cpu_planners.handler = only_target
    wrapper, kwargs, _ = _cpu_request()
    wrapper.plan(**kwargs)
    assert wrapper._planned_backend_name == target
    assert _cpu_planners.calls[-1] == target
    assert len(_cpu_planners.calls) == len(set(_cpu_planners.calls))
    _cpu_planners.forbidden = True
    _cpu_run(wrapper)


def test_cpu_all_rejected_diagnostics_and_order_are_deterministic(_cpu_planners):
    _cpu_planners.handler = _reject
    orders = []
    for _ in range(2):
        _cpu_planners.calls.clear()
        wrapper, kwargs, _ = _cpu_request()
        with pytest.raises(_BackendPlanUnsupportedError) as error:
            wrapper.plan(**kwargs)
        orders.append(tuple(_cpu_planners.calls))
        assert set(orders[-1]) == set(_SM100_BACKENDS)
        for name in _SM100_BACKENDS:
            assert f"{name}: deliberate support rejection" in str(error.value)
        assert wrapper._planned_backend is None
        with pytest.raises(RuntimeError, match="before plan"):
            _cpu_run(wrapper)
    assert orders[0] == orders[1]


@pytest.mark.parametrize(
    "heads,kv_len,graph,first_three",
    [
        (3, 1, False, ("fa2", "trtllm-gen", "cute-dsl-monolithic")),
        (128, 1, False, ("fa2", "trtllm-gen", "cute-dsl-monolithic")),
        (32, 32768, False, ("trtllm-gen", "cute-dsl-monolithic", "fa2")),
        (32, 32768, True, ("cute-dsl-monolithic", "trtllm-gen", "fa2")),
    ],
)
def test_cpu_public_auto_uses_request_specific_policy_order(
    _cpu_planners, heads, kv_len, graph, first_three
):
    wrapper, kwargs, _ = _cpu_request(graph=graph)
    kwargs["num_heads"] = heads
    # One live page per request; page capacity must not substitute for CSR Q.
    kwargs["page_size"] = kv_len
    kwargs["metadata"].kv_len_arr.fill_(kv_len)
    _cpu_planners.handler = _reject
    with pytest.raises(_BackendPlanUnsupportedError):
        wrapper.plan(**kwargs)
    assert tuple(_cpu_planners.calls) == first_three + (
        "cute-dsl-modular",
        "cutile",
        "cutlass",
    )
    assert wrapper._planned_backend is None


@pytest.mark.parametrize(
    "rejected,expected",
    [
        ((), "fa2"),
        (("fa2",), "trtllm-gen"),
        (("fa2", "trtllm-gen"), "cute-dsl-monolithic"),
    ],
)
def test_cpu_high_head_decode_preserves_geometry_fallback(
    _cpu_planners, rejected, expected
):
    wrapper, kwargs, _ = _cpu_request()
    kwargs["num_heads"] = 128

    def reject_prefix(name, args):
        if name in rejected:
            _reject(name, args)

    _cpu_planners.handler = reject_prefix
    wrapper.plan(**kwargs)
    assert tuple(_cpu_planners.calls) == rejected + (expected,)
    assert wrapper._planned_backend._backend == expected


@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_cpu_auto_propagates_fatal_planner_errors(_cpu_planners, error_type):
    failure = error_type("deliberate compiler or caller error")

    def fail(name, args):
        raise failure

    _cpu_planners.handler = fail
    wrapper, kwargs, _ = _cpu_request()
    with pytest.raises(error_type) as error:
        wrapper.plan(**kwargs)
    assert error.value is failure
    assert len(_cpu_planners.calls) == 1
    assert wrapper._planned_backend is None


def test_cpu_explicit_backend_is_strict(_cpu_planners):
    _cpu_planners.handler = _reject
    wrapper, kwargs, _ = _cpu_request("trtllm-gen")
    with pytest.raises(
        _BackendPlanUnsupportedError, match="deliberate support rejection"
    ):
        wrapper.plan(**kwargs)
    assert _cpu_planners.calls == ["trtllm-gen"]
    assert wrapper._planned_backend is None


@pytest.mark.parametrize(
    "error_type", [_BackendPlanUnsupportedError, RuntimeError, ValueError]
)
def test_cpu_failed_replan_restores_storage_and_executable(_cpu_planners, error_type):
    wrapper, kwargs, buffers = _cpu_request("fa2", graph=True)
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
    old_contract = wrapper._input_contract
    old_output = _cpu_run(wrapper).clone()
    mirrors = (
        wrapper._cached_module,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
    )
    watched = [*buffers, mirrors[1], mirrors[2]]
    snapshots = [(buffer, buffer.data_ptr(), buffer.clone()) for buffer in watched]
    failure = error_type("deliberate failure after writes")

    def mutate_then_fail(name, args):
        assert args._graph_plan_int_workspace_buffer is not None
        assert args._graph_plan_int_workspace_buffer.shape == mirrors[1].shape
        for buffer in _plan_buffers(args):
            buffer.fill_(91)  # Includes the tail beyond _staged_int_workspace_bytes.
        raise failure

    _cpu_planners.handler = mutate_then_fail
    with pytest.raises(error_type) as error:
        wrapper.plan(**kwargs)
    assert error.value is failure
    assert wrapper._planned_backend is previous
    assert wrapper._planned_backend_name == "fa2"
    assert wrapper._input_contract is old_contract
    assert wrapper._cached_module is mirrors[0]
    assert wrapper._int_workspace_buffer is mirrors[1]
    assert wrapper._pin_memory_int_workspace_buffer is mirrors[2]
    assert wrapper._float_workspace_buffer is buffers[0]
    for name, buffer in zip(
        ("_qo_indptr_buf", "_kv_indptr_buf", "_kv_indices_buf", "_kv_len_arr_buf"),
        buffers[1:],
        strict=True,
    ):
        assert getattr(wrapper, name) is buffer
    for buffer, pointer, snapshot in snapshots:
        assert buffer.data_ptr() == pointer
        assert torch.equal(buffer, snapshot)
    _cpu_planners.forbidden = True
    torch.testing.assert_close(_cpu_run(wrapper), old_output)


def test_cpu_typed_fallback_restores_buffers_before_next_candidate(_cpu_planners):
    wrapper, kwargs, buffers = _cpu_request(graph=True)
    originals = [buffer.clone() for buffer in buffers]

    def first_rejects_after_writes(name, args):
        current = _plan_buffers(args)
        assert len(current) == len(originals)
        for actual, expected in zip(current, originals, strict=True):
            assert torch.equal(actual, expected), (
                "candidate saw predecessor's failed writes"
            )
        if len(_cpu_planners.calls) == 1:
            for buffer in current:
                buffer.fill_(91)
            _reject(name, args)

    _cpu_planners.handler = first_rejects_after_writes
    wrapper.plan(**kwargs)
    assert len(_cpu_planners.calls) == 2
    for actual, expected in zip(buffers, originals, strict=True):
        assert torch.equal(actual, expected)
    _cpu_planners.forbidden = True
    _cpu_run(wrapper)


@pytest.mark.parametrize("allowed", [False, True])
def test_cpu_experimental_candidate_requires_upstream_opt_in(
    _cpu_planners, monkeypatch, allowed
):
    from dataclasses import replace

    state = _cpu_planners
    target = state.module._BACKEND_TYPES["trtllm-gen"]
    monkeypatch.setattr(
        target,
        "_plan_capabilities",
        replace(target._plan_capabilities, is_experimental=True),
    )
    monkeypatch.setenv(
        "FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1" if allowed else "0"
    )

    def only_target(name, args):
        if name != "trtllm-gen":
            _reject(name, args)

    state.handler = only_target
    wrapper, kwargs, _ = _cpu_request()
    if allowed:
        wrapper.plan(**kwargs)
        assert wrapper._planned_backend_name == "trtllm-gen"
    else:
        with pytest.raises(
            _BackendPlanUnsupportedError,
            match="FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1",
        ):
            wrapper.plan(**kwargs)
        assert "trtllm-gen" not in state.calls


def test_cpu_eager_auto_reselects_but_failed_replan_preserves_choice(_cpu_planners):
    state = _cpu_planners
    desired = ["trtllm-gen"]

    def only_target(name, args):
        if name != desired[0]:
            _reject(name, args)

    state.handler = only_target
    wrapper, kwargs, _ = _cpu_request()
    wrapper.plan(**kwargs)
    desired[0] = "fa2"
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
    assert wrapper._planned_backend_name == "fa2"
    desired[0] = "none"
    with pytest.raises(_BackendPlanUnsupportedError):
        wrapper.plan(**kwargs)
    assert wrapper._planned_backend is previous
    assert wrapper._backend == "fa2"


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("capability", [(10, 0), (9, 0)])
def test_cpu_auto_replans_reselect_only_outside_graphs(
    _cpu_planners, monkeypatch, graph, capability
):
    state = _cpu_planners
    orders = []
    desired = ["fa3" if capability == (9, 0) else "fa2"]
    initial = desired[0]

    def order(args):
        orders.append(desired[0])
        return (desired[0],)

    def legacy(device):
        orders.append(desired[0])
        return desired[0]

    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: capability)
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", order)
    monkeypatch.setattr(_auto_policy, "determine_mla_backend", legacy)
    wrapper, kwargs, _ = _cpu_request(graph=graph)
    wrapper.plan(**kwargs)
    desired[0] = "fa2" if capability == (9, 0) else "fa3"
    if graph:

        def forbidden_selection(args):
            pytest.fail("Graph replan must not probe hardware or rank candidates")

        monkeypatch.setattr(
            _auto_policy, "_get_compute_capability", forbidden_selection
        )
        monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden_selection)
        monkeypatch.setattr(_auto_policy, "determine_mla_backend", forbidden_selection)
    wrapper.plan(**kwargs)
    expected = initial if graph else desired[0]
    assert orders == ([initial] if graph else [initial, expected])
    assert state.calls == [initial, expected]
    assert wrapper._backend == wrapper._planned_backend_name == expected


def test_cpu_auto_graph_replan_preserves_experimental_eligibility(
    _cpu_planners, monkeypatch
):
    from dataclasses import replace

    state = _cpu_planners
    monkeypatch.setattr(
        _auto_policy, "ordered_sm100_backends", lambda args: ("fa2", "trtllm-gen")
    )
    target = state.module._BACKEND_TYPES["fa2"]
    monkeypatch.setattr(
        target,
        "_plan_capabilities",
        replace(target._plan_capabilities, is_experimental=True),
    )
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    automatic_warnings = []
    monkeypatch.setattr(
        _wrapper,
        "warn_experimental_backend_once",
        lambda api, backend, *, automatic: automatic_warnings.append(automatic),
    )
    wrapper, kwargs, _ = _cpu_request(graph=True)
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
    assert wrapper._backend == "fa2"
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "0")
    with pytest.raises(
        _BackendPlanUnsupportedError, match="experimental backend requires"
    ):
        wrapper.plan(**kwargs)
    assert state.calls == ["fa2"], "Ineligible graph backend cannot run or fall back"
    assert automatic_warnings == [True]
    assert wrapper._planned_backend is previous
    assert wrapper._backend == "fa2"


@pytest.mark.parametrize(
    "capability,cuda_version,expected",
    [
        ((8, 0), "13.0", "fa2"),
        ((9, 0), "12.2", "fa2"),
        ((9, 0), "12.3", "fa3"),
        ((10, 3), "13.0", "fa2"),
        ((12, 0), "13.0", "fa2"),
    ],
)
def test_cpu_off_sm100_keeps_legacy_dispatch(
    _cpu_planners, monkeypatch, capability, cuda_version, expected
):
    import flashinfer.utils as utils

    state = _cpu_planners
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: capability)
    monkeypatch.setattr(utils, "get_compute_capability", lambda _: capability)
    monkeypatch.setattr(torch.version, "cuda", cuda_version)
    monkeypatch.setattr(
        _auto_policy, "determine_mla_backend", utils.determine_mla_backend
    )

    def forbidden_policy(args):
        pytest.fail("Legacy auto must not consult the SM100 ranking")

    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden_policy)
    wrapper, kwargs, _ = _cpu_request()
    assert wrapper._backend == "auto"
    wrapper.plan(**kwargs)
    assert state.calls == [expected]
    assert wrapper._planned_backend_name == wrapper._backend == expected
    state.forbidden = True
    _cpu_run(wrapper)


@pytest.mark.parametrize("graph", [False, True])
@pytest.mark.parametrize("error_type", [_BackendPlanUnsupportedError, RuntimeError])
def test_cpu_legacy_auto_failed_replan_preserves_executable(
    _cpu_planners, monkeypatch, graph, error_type
):
    state = _cpu_planners
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: (9, 0))
    monkeypatch.setattr(_auto_policy, "determine_mla_backend", lambda _: "fa3")
    wrapper, kwargs, buffers = _cpu_request(graph=graph)
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
    contract = wrapper._input_contract
    output = _cpu_run(wrapper).clone()
    watched = [*buffers, *previous.buffers, previous._int_workspace_buffer]
    originals = [(buffer, buffer.clone()) for buffer in watched]
    state.calls.clear()
    failure = error_type("legacy planner failure after writes")

    if graph:

        def forbidden_order(args):
            pytest.fail("Graph replan must retain its prepared backend")

        monkeypatch.setattr(_auto_policy, "_get_compute_capability", forbidden_order)
        monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden_order)
        monkeypatch.setattr(_auto_policy, "determine_mla_backend", forbidden_order)

    def mutate_and_fail(name, args):
        assert name == "fa3"
        if graph:
            assert (
                args._graph_plan_int_workspace_buffer is previous._int_workspace_buffer
            )
        for buffer in _plan_buffers(args):
            buffer.fill_(91)
        raise failure

    state.handler = mutate_and_fail
    with pytest.raises(error_type) as caught:
        wrapper.plan(**kwargs)
    assert caught.value is failure
    assert state.calls == ["fa3"], "Legacy auto must not fall back to FA2"
    assert wrapper._planned_backend is previous
    assert wrapper._input_contract is contract
    assert wrapper._backend == "fa3"
    for buffer, original in originals:
        assert torch.equal(buffer, original)
    state.forbidden = True
    torch.testing.assert_close(_cpu_run(wrapper), output)


def test_cpu_legacy_auto_warns_at_plan_once_at_external_caller(
    _cpu_planners, monkeypatch
):
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: (10, 3))
    monkeypatch.setattr(
        _auto_policy._BatchMLAPagedAttentionAutoBackend,
        "_blackwell_auto_fallback_warned",
        False,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", UserWarning)
        wrapper, kwargs, _ = _cpu_request()
        assert not caught, "Constructing an auto wrapper must not resolve or warn"
        wrapper.plan(**kwargs)
        wrapper.plan(**kwargs)
        other, other_kwargs, _ = _cpu_request()
        other.plan(**other_kwargs)
    fallback = [
        warning for warning in caught if "not Blackwell-native" in str(warning.message)
    ]
    assert len(fallback) == 1
    assert fallback[0].filename == __file__


def test_cpu_legacy_auto_preserves_flat_csr_extra_query_offsets(
    _cpu_planners, monkeypatch
):
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: (8, 0))
    wrapper, kwargs, _ = _cpu_request()
    metadata = kwargs.pop("metadata")
    kwargs.update(
        qo_indptr=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        kv_indptr=metadata.kv_indptr,
        kv_indices=metadata.kv_indices,
        kv_len_arr=metadata.kv_len_arr,
    )
    wrapper.plan(**kwargs)
    assert _cpu_planners.calls == ["fa2"]


def test_cpu_invalid_backend_still_fails_in_constructor(_cpu_planners, monkeypatch):
    def forbidden_probe(device):
        pytest.fail("Invalid selector must be rejected without querying hardware")

    monkeypatch.setattr(_auto_policy, "_get_compute_capability", forbidden_probe)
    with pytest.raises(ValueError, match="backend must be one of"):
        _cpu_request("invalid")


@pytest.mark.parametrize(
    "key,value", [("num_heads", 0), ("page_size", 0), ("sm_scale", float("nan"))]
)
def test_cpu_invalid_request_is_not_candidate_fallback(_cpu_planners, key, value):
    wrapper, kwargs, _ = _cpu_request()
    kwargs[key] = value
    with pytest.raises(ValueError):
        wrapper.plan(**kwargs)
    assert not _cpu_planners.calls


def test_cpu_cute_family_alias_reselects_on_eager_replan(_cpu_planners, monkeypatch):
    state = _cpu_planners
    alias = state.module._BatchMLAPagedAttentionCuteDslBackend
    from dataclasses import replace

    names = ("cute-dsl-monolithic", "cute-dsl-modular")
    modular = state.module._BACKEND_TYPES[names[1]]
    monkeypatch.setattr(
        modular,
        "_plan_capabilities",
        replace(modular._plan_capabilities, supports_sinks=True),
    )
    monkeypatch.setattr(
        alias, "_candidate_types", tuple(state.module._BACKEND_TYPES[n] for n in names)
    )
    monkeypatch.setitem(state.module._BACKEND_TYPES, "cute-dsl", alias)

    def support(name, args):
        if name == "cute-dsl-monolithic" and args.use_sinks:
            _reject(name, args)

    state.handler = support
    wrapper, kwargs, _ = _cpu_request("cute-dsl")
    wrapper.plan(**kwargs)
    assert wrapper._planned_backend_name == wrapper._backend == "cute-dsl-monolithic"
    kwargs["use_sinks"] = True
    wrapper.plan(**kwargs)
    assert wrapper._planned_backend_name == "cute-dsl-modular"
    assert wrapper._backend == "cute-dsl-modular"


@pytest.mark.parametrize("selector", ["auto", "cute-dsl"])
def test_cpu_selector_restores_each_attempt(_cpu_planners, monkeypatch, selector):
    state = _cpu_planners
    names = ("cute-dsl-monolithic", "cute-dsl-modular")
    if selector == "auto":
        monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", lambda args: names)
    else:
        alias = state.module._BatchMLAPagedAttentionCuteDslBackend
        monkeypatch.setattr(
            alias,
            "_candidate_types",
            tuple(state.module._BACKEND_TYPES[n] for n in names),
        )
        monkeypatch.setitem(state.module._BACKEND_TYPES, selector, alias)
    wrapper, kwargs, buffers = _cpu_request(selector)
    metadata = kwargs["metadata"]
    watched = {
        id(buffer): buffer
        for buffer in (
            *buffers,
            metadata.qo_indptr,
            metadata.kv_indptr,
            metadata.kv_indices,
            metadata.kv_len_arr,
        )
    }
    originals = {key: buffer.clone() for key, buffer in watched.items()}

    def first_rejects(name, args):
        for key, buffer in watched.items():
            assert torch.equal(buffer, originals[key]), "A rejected planner left writes"
        if name == names[0]:
            for buffer in watched.values():
                buffer.fill_(91)
            _reject(name, args)

    state.handler = first_rejects
    wrapper.plan(**kwargs)
    assert state.calls == list(names)
    assert wrapper._planned_backend_name == names[1]
    for key, buffer in watched.items():
        assert torch.equal(buffer, originals[key])
    expected = sum(int(buffer.flatten()[0]) for buffer in buffers) + 17
    _cpu_planners.forbidden = True
    torch.testing.assert_close(
        _cpu_run(wrapper), torch.full((2, 3, 4), expected, dtype=torch.bfloat16)
    )


def test_cpu_experimental_warning_failure_restores_unpublished_plan(
    _cpu_planners, monkeypatch
):
    from dataclasses import replace

    state = _cpu_planners
    wrapper, kwargs, buffers = _cpu_request("fa2")
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
    previous_output = _cpu_run(wrapper).clone()
    originals = [buffer.clone() for buffer in buffers]
    target = state.module._BACKEND_TYPES["fa2"]
    monkeypatch.setattr(
        target,
        "_plan_capabilities",
        replace(target._plan_capabilities, is_experimental=True),
    )

    def mutate(name, args):
        for buffer in _plan_buffers(args):
            buffer.fill_(91)

    def warning_error(*args, **kwargs):
        raise UserWarning("experimental warning promoted to error")

    state.handler = mutate
    monkeypatch.setattr(_wrapper, "warn_experimental_backend_once", warning_error)
    with pytest.raises(UserWarning, match="promoted to error"):
        wrapper.plan(**kwargs)
    assert wrapper._planned_backend is previous
    for buffer, original in zip(buffers, originals, strict=True):
        assert torch.equal(buffer, original)
    torch.testing.assert_close(_cpu_run(wrapper), previous_output)


@pytest.mark.parametrize("q_len", [65535, 65536])
def test_cpu_monolithic_grid_boundary_rejects_before_compile(
    _cpu_planners, monkeypatch, q_len
):
    from flashinfer.cute_dsl.attention.monolithic import mla_decode as native
    from flashinfer.mla._batch_mla._backends.cute_dsl_monolithic_backend import (
        _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    )
    from flashinfer.mla._batch_mla._backends._cute_dsl_common import (
        _CuteDslKernelUnsupportedError,
    )

    calls = []
    monkeypatch.setattr(native, "_check_can_implement", lambda **kwargs: None)
    monkeypatch.setattr(
        native, "_get_split_kv_and_workspace_size", lambda *args, **kwargs: (1, 0)
    )
    monkeypatch.setattr(native, "get_num_sm", lambda _: 148)
    monkeypatch.setattr(
        native, "_get_compiled_mla_kernel", lambda *args, **kwargs: calls.append(True)
    )
    workspace = torch.empty(16, dtype=torch.uint8)
    backend = _BatchMLAPagedAttentionCuteDslMonolithicBackend(workspace)

    def prepare():
        return backend._compile_kernel(
            workspace_buffer=workspace,
            device=workspace.device,
            q_data_type=torch.bfloat16,
            out_dtype=torch.bfloat16,
            page_size=64,
            batch_size=1,
            num_heads=128,
            q_len=q_len,
            head_dim_ckv=512,
            head_dim_kpe=64,
            resolved_is_var_seq=True,
            is_var_q=False,
            total_q=q_len,
            max_seq_len=65536,
            use_sinks=False,
            enable_pdl=False,
        )

    if q_len == 65536:
        with pytest.raises(_CuteDslKernelUnsupportedError, match="grid.y.*65536"):
            prepare()
        assert not calls
    else:
        prepare()
        assert calls == [True]


@pytest.mark.parametrize("scale", [1, 1.0])
def test_cpu_numeric_scale_normalized_before_candidate_planning(_cpu_planners, scale):
    def check(name, args):
        assert type(args.sm_scale) is float
        assert args.sm_scale == 1.0

    _cpu_planners.handler = check
    wrapper, kwargs, _ = _cpu_request()
    kwargs["sm_scale"] = scale
    wrapper.plan(**kwargs)
    assert _cpu_planners.calls


@pytest.mark.parametrize("error_type", [RuntimeError, ValueError])
def test_cpu_two_typed_rejections_then_fatal_restores_previous_executable(
    _cpu_planners, monkeypatch, error_type
):
    state = _cpu_planners
    order = (
        "fa2",
        "cutlass",
        "trtllm-gen",
        "cutile",
        "cute-dsl-monolithic",
        "cute-dsl-modular",
    )
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", lambda args: order)
    # Eager auto replanning may explore multiple candidates. A graph replan
    # intentionally retains its previous executable family instead.
    wrapper, kwargs, buffers = _cpu_request()
    wrapper.plan(**kwargs)
    assert wrapper._planned_backend_name == "fa2"
    previous = wrapper._planned_backend
    old_contract = wrapper._input_contract
    old_output = _cpu_run(wrapper).clone()
    mirrors = (
        wrapper._cached_module,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
    )
    metadata = kwargs["metadata"]
    metadata_buffers = [
        metadata.qo_indptr,
        metadata.kv_indptr,
        metadata.kv_indices,
        metadata.kv_len_arr,
    ]
    watched = [*buffers, *metadata_buffers, mirrors[1], mirrors[2]]
    snapshots = [(buffer, buffer.data_ptr(), buffer.clone()) for buffer in watched]
    failure = error_type("fatal third planner after two typed rejections")
    state.calls.clear()

    def mutate_and_fail(name, args):
        # Every candidate must see the original bytes, not a preceding
        # rejected candidate's writes. This also checks metadata-source rollback.
        for buffer, pointer, snapshot in snapshots:
            assert buffer.data_ptr() == pointer
            assert torch.equal(buffer, snapshot), f"{name} observed failed writes"
        attempt = len(state.calls)
        assert name == order[attempt - 1]
        for buffer in (*_plan_buffers(args), *metadata_buffers):
            buffer.fill_(90 + attempt)
        if attempt <= 2:
            raise _BackendPlanUnsupportedError(f"typed rejection {attempt}: {name}")
        raise failure

    state.handler = mutate_and_fail
    with pytest.raises(error_type) as caught:
        wrapper.plan(**kwargs)
    assert caught.value is failure
    assert state.calls == list(order[:3]), "fatal errors must terminate fallback"
    assert wrapper._planned_backend is previous
    assert wrapper._planned_backend_name == wrapper._backend == "fa2"
    assert wrapper._input_contract is old_contract
    assert wrapper._cached_module is mirrors[0]
    assert wrapper._int_workspace_buffer is mirrors[1]
    assert wrapper._pin_memory_int_workspace_buffer is mirrors[2]
    assert wrapper._float_workspace_buffer is buffers[0]
    for name, buffer in zip(
        ("_qo_indptr_buf", "_kv_indptr_buf", "_kv_indices_buf", "_kv_len_arr_buf"),
        buffers[1:],
        strict=True,
    ):
        assert getattr(wrapper, name) is buffer
    for buffer, pointer, snapshot in snapshots:
        assert buffer.data_ptr() == pointer
        assert torch.equal(buffer, snapshot)

    def forbidden(*args, **kwargs):
        pytest.fail("run entered automatic policy or planning after a failed replan")

    state.forbidden = True
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden)
    monkeypatch.setattr(wrapper, "plan", forbidden)
    torch.testing.assert_close(_cpu_run(wrapper), old_output)
    assert state.calls == list(order[:3])


@pytest.mark.parametrize(
    "rejected,expected",
    [
        ((), "cute-dsl-monolithic"),
        (("cute-dsl-monolithic",), "trtllm-gen"),
        (("cute-dsl-monolithic", "trtllm-gen"), "cute-dsl-modular"),
        (("cute-dsl-monolithic", "trtllm-gen", "cute-dsl-modular"), "fa2"),
    ],
)
def test_cpu_modular_volume_preserves_native_prefix_and_typed_fallback(
    _cpu_planners, rejected, expected
):
    wrapper, kwargs, _ = _cpu_request()
    kwargs.update(num_heads=16, page_size=64)
    # Two Q2 requests: total KV=81920 and work=2.5M. Real CSR geometry
    # reaches the volume path without the early short-context guard.
    kwargs["metadata"] = MLAPlanMetadata.csr(
        qo_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 640, 1280], dtype=torch.int32),
        kv_indices=torch.arange(1280, dtype=torch.int32),
        kv_len_arr=torch.tensor([40960, 40960], dtype=torch.int32),
    )

    def reject_prefix(name, args):
        if name in rejected:
            _reject(name, args)

    _cpu_planners.handler = reject_prefix
    wrapper.plan(**kwargs)
    assert tuple(_cpu_planners.calls) == rejected + (expected,)
    assert wrapper._planned_backend_name == expected


@pytest.mark.parametrize(
    "rejected,expected",
    [
        ((), "trtllm-gen"),
        (("trtllm-gen",), "cute-dsl-monolithic"),
        (("trtllm-gen", "cute-dsl-monolithic"), "fa2"),
    ],
)
def test_cpu_large_prefill_graph_preserves_typed_fallback(
    _cpu_planners, rejected, expected
):
    _, kwargs, _ = _cpu_request(graph=True)
    metadata = MLAPlanMetadata.csr(
        qo_indptr=torch.tensor([0, 256, 512, 768, 1024], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32),
        kv_indices=torch.arange(16, dtype=torch.int32),
        kv_len_arr=torch.full((4,), 256, dtype=torch.int32),
    )
    # Graph mirrors must have room for all four requests and sixteen pages.
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.zeros(64, dtype=torch.uint8),
        backend="auto",
        use_cuda_graph=True,
        qo_indptr=torch.empty_like(metadata.qo_indptr),
        kv_indptr=torch.empty_like(metadata.kv_indptr),
        kv_indices=torch.empty_like(metadata.kv_indices),
        kv_len_arr=torch.empty_like(metadata.kv_len_arr),
    )
    kwargs.update(metadata=metadata, num_heads=64, page_size=64, causal=True)

    def reject_prefix(name, args):
        assert args._use_cuda_graph
        if name in rejected:
            _reject(name, args)

    _cpu_planners.handler = reject_prefix
    wrapper.plan(**kwargs)
    assert tuple(_cpu_planners.calls) == rejected + (expected,)
    assert wrapper._planned_backend_name == expected


# GPU execution: real SM100 adapters, numerical references and graph replay.


_HEADS = 16


_CKV = 512


_KPE = 64


_PAGE = 64


_SCALE = 1 / math.sqrt(_CKV + _KPE)


@pytest.fixture
def _sm100_reference_precision():
    if not torch.cuda.is_available():
        pytest.skip("SM100 MLA numerical acceptance requires CUDA")
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("SM100 MLA numerical acceptance requires an SM100 GPU")
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def _inputs(q_lens, kv_lens, dtype, capacity, *, crafted=False):
    torch.manual_seed(4031)
    offsets = [0]
    for length in q_lens:
        offsets.append(offsets[-1] + length)
    pages = [math.ceil(length / _PAGE) for length in kv_lens]
    # Nontrivial page IDs prevent the reference from assuming request-local
    # contiguous storage. Unused table slots point to valid but masked pages.
    ids = list(reversed(range(sum(pages))))
    width = math.ceil(max(pages) / 2) * 2
    table = []
    cursor = 0
    for count in pages:
        table.append(ids[cursor : cursor + count] + [0] * (width - count))
        cursor += count
    query = torch.randn(offsets[-1], _HEADS, _CKV + _KPE, device="cuda").to(dtype)
    cache = torch.randn(sum(pages), _PAGE, _CKV + _KPE, device="cuda").to(dtype)
    if crafted:
        assert q_lens == kv_lens == (4,)
        query.zero_()
        cache.zero_()
        cache[table[0][0], 3, :_CKV] = 8
    table_device = torch.tensor(table, dtype=torch.int32, device="cuda")
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        block_tables=table_device,
        seq_lens=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
        max_q_len=capacity,
    )
    return metadata, query, cache, table_device, offsets


def _reference(query, cache, table, offsets, kv_lens, *, causal):
    """Independent FP32 MLA; causality is bottom-right within each request."""
    outputs, lses = [], []
    for batch, kv_len in enumerate(kv_lens):
        q_begin, q_end = offsets[batch : batch + 2]
        q_len = q_end - q_begin
        live_pages = table[batch, : math.ceil(kv_len / _PAGE)].long()
        kv = cache[live_pages].reshape(-1, _CKV + _KPE)[:kv_len].float()
        # Bounded Q tiles avoid materializing an entire prefill score matrix.
        for start in range(0, q_len, 32):
            stop = min(start + 32, q_len)
            q = query[q_begin + start : q_begin + stop].float()
            logits = torch.einsum("qhd,kd->qhk", q, kv) * _SCALE
            if causal:
                q_positions = torch.arange(start, stop, device=query.device)
                k_positions = torch.arange(kv_len, device=query.device)
                blocked = k_positions[None, :] > (kv_len - q_len + q_positions[:, None])
                logits.masked_fill_(blocked[:, None, :], -torch.inf)
            probabilities = torch.softmax(logits, dim=-1)
            outputs.append(torch.einsum("qhk,kd->qhd", probabilities, kv[:, :_CKV]))
            lses.append(torch.logsumexp(logits, dim=-1))
    return torch.cat(outputs), torch.cat(lses)


def _plan(backend, metadata, dtype, *, causal, lse_mode):
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend=backend)
    wrapper.plan(
        metadata=metadata,
        num_heads=_HEADS,
        head_dim_ckv=_CKV,
        head_dim_kpe=_KPE,
        page_size=_PAGE,
        causal=causal,
        sm_scale=_SCALE,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_dtype=dtype,
        query_layout="packed",
        kv_cache_layout="packed",
        lse_mode=lse_mode,
    )
    return wrapper


def _check_case(backend, q_lens, kv_lens, capacity, dtype, *, causal, crafted=False):
    metadata, query, cache, table, offsets = _inputs(
        q_lens, kv_lens, dtype, capacity, crafted=crafted
    )
    if backend == "cute-dsl-monolithic":
        lse_mode = "basee"
    elif backend == "trtllm-gen" and len(set(q_lens)) == 1:
        lse_mode = "base2"
    else:
        lse_mode = "none"
    expected, expected_lse = _reference(
        query, cache, table, offsets, kv_lens, causal=causal
    )
    if crafted:
        # Independent negative control: omitting the mask changes the first
        # output from exactly zero to two, far beyond BF16 test tolerance.
        masked, _ = _reference(query, cache, table, offsets, kv_lens, causal=True)
        unmasked, _ = _reference(query, cache, table, offsets, kv_lens, causal=False)
        assert torch.equal(masked[0], torch.zeros_like(masked[0]))
        torch.testing.assert_close(unmasked[0], torch.full_like(unmasked[0], 2))
        assert not torch.allclose(masked, unmasked, rtol=1e-2, atol=1e-2)
    wrapper = _plan(backend, metadata, dtype, causal=causal, lse_mode=lse_mode)
    out = torch.empty_like(expected, dtype=dtype)
    lse = torch.empty_like(expected_lse) if lse_mode != "none" else None
    result = wrapper.run(
        query=query,
        kv_cache=cache,
        out=out,
        lse=lse,
        return_lse=lse is not None,
        return_lse_base_on_e=lse_mode == "basee",
    )
    if lse is None:
        assert result is out
    else:
        assert result[0] is out
        assert result[1] is lse
        if lse_mode == "base2":
            expected_lse /= math.log(2)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-2, atol=1e-2)
    # Match existing native CuTe FP16/BF16 tests; inputs have no FP8
    # quantization. The crafted test rules out masking errors at this tolerance.
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("backend", ["trtllm-gen", "cute-dsl-monolithic"])
@pytest.mark.parametrize(
    "q_lens,kv_lens,capacity",
    [
        pytest.param((2, 2), (130, 129), 8, id="prefix-q2-capacity8"),
        pytest.param((128,), (128,), 128, id="no-prefix-q128"),
        pytest.param((1, 3, 7), (129, 131, 137), 16, id="compact-ragged-capacity16"),
    ],
)
def test_explicit_causal_adapter_matches_reference(backend, q_lens, kv_lens, capacity):
    _check_case(backend, q_lens, kv_lens, capacity, torch.bfloat16, causal=True)


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize(
    "q_lens,kv_lens,capacity",
    [
        pytest.param((2, 2), (130, 129), 8, id="uniform"),
        pytest.param((1, 3, 7), (129, 131, 137), 16, id="compact-ragged"),
    ],
)
def test_monolithic_fp16_adapter_matches_reference(q_lens, kv_lens, capacity):
    _check_case(
        "cute-dsl-monolithic", q_lens, kv_lens, capacity, torch.float16, causal=True
    )


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("crafted", [False, True], ids=["prefix-q2", "strong-mask-q4"])
def test_modular_fp16_noncausal_adapter_matches_reference(crafted):
    _check_case(
        "cute-dsl-modular",
        (4,) if crafted else (2, 2),
        (4,) if crafted else (130, 129),
        4 if crafted else 8,
        torch.float16,
        causal=False,
        crafted=crafted,
    )


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("backend", ["trtllm-gen", "cute-dsl-monolithic"])
def test_causal_adapter_obeys_strong_mask_case(backend):
    _check_case(backend, (4,), (4,), 4, torch.bfloat16, causal=True, crafted=True)


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("backend", ["trtllm-gen", "cute-dsl-monolithic"])
def test_causal_only_adapter_rejects_noncausal_multi_query(backend):
    metadata, *_ = _inputs((4,), (4,), torch.bfloat16, 4)
    with pytest.raises(_BackendPlanUnsupportedError, match="[Cc]ausal"):
        _plan(backend, metadata, torch.bfloat16, causal=False, lse_mode="none")


@pytest.mark.usefixtures("_sm100_reference_precision")
def test_modular_rejects_causal_multi_query():
    metadata, *_ = _inputs((4,), (4,), torch.bfloat16, 4)
    with pytest.raises(_BackendPlanUnsupportedError, match="[Cc]ausal"):
        _plan(
            "cute-dsl-modular", metadata, torch.bfloat16, causal=True, lse_mode="none"
        )


@pytest.mark.usefixtures("_sm100_reference_precision")
def test_auto_fp8_default_scale_matches_reference():
    metadata, query, cache, table, offsets = _inputs((2,), (9,), torch.float8_e4m3fn, 4)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=_HEADS,
        head_dim_ckv=_CKV,
        head_dim_kpe=_KPE,
        page_size=_PAGE,
        causal=True,
        sm_scale=_SCALE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
    )
    assert wrapper._input_contract.scale_mode == "default"
    out = torch.empty((2, _HEADS, _CKV), dtype=torch.bfloat16, device="cuda")
    actual = wrapper.run(query=query, kv_cache=cache, out=out)
    expected, _ = _reference(query, cache, table, offsets, (9,), causal=True)
    assert actual is out
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=0.05)


@pytest.mark.usefixtures("_sm100_reference_precision")
def test_auto_empty_kv_split_falls_back_and_returns_zero():
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32, device="cuda"),
        block_tables=torch.empty((1, 0), dtype=torch.int32, device="cuda"),
        seq_lens=torch.zeros(1, dtype=torch.int32, device="cuda"),
        max_q_len=1,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=32,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=16,
        causal=False,
        sm_scale=1,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        query_layout="split",
        kv_cache_layout="split",
    )
    assert wrapper._planned_backend_name == "fa2"
    q = (
        torch.ones((1, 32, 512), dtype=torch.bfloat16, device="cuda"),
        torch.ones((1, 32, 64), dtype=torch.bfloat16, device="cuda"),
    )
    kv = (
        torch.ones((1, 16, 512), dtype=torch.bfloat16, device="cuda"),
        torch.ones((1, 16, 64), dtype=torch.bfloat16, device="cuda"),
    )
    out = torch.full((1, 32, 512), float("nan"), dtype=torch.bfloat16, device="cuda")
    assert wrapper.run(query=q, kv_cache=kv, out=out) is out
    assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_auto_graph_typed_fallback_freezes_selection_and_replays_correctly(
    monkeypatch, dtype
):
    from flashinfer.mla._batch_mla import _wrapper
    from flashinfer.mla._batch_mla._backends import fa2_backend

    attempts = []
    policy_calls = []
    order = (
        "cutlass",
        "fa2",
        "trtllm-gen",
        "cute-dsl-monolithic",
        "cute-dsl-modular",
        "cutile",
    )

    def policy(args):
        assert args._use_cuda_graph
        policy_calls.append(order)
        return order

    # Inject only the first candidate's typed refusal. The fallback uses the
    # native FA2 planner, generated kernel, public run path and CUDA graph.
    class RejectingCutlass(_wrapper._BACKEND_TYPES["cutlass"]):
        @classmethod
        def preflight_plan_from_wrapper(cls, args):
            pass

        @classmethod
        def plan_from_wrapper(cls, args):
            attempts.append("cutlass")
            for buffer in _plan_buffers(args):
                buffer.fill_(91)
            raise _BackendPlanUnsupportedError("deliberate first-candidate refusal")

    class ObservedFA2(_wrapper._BACKEND_TYPES["fa2"]):
        @classmethod
        def plan_from_wrapper(cls, args):
            attempts.append("fa2")
            return super().plan_from_wrapper(args)

    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", policy)
    monkeypatch.setitem(_wrapper._BACKEND_TYPES, "cutlass", RejectingCutlass)
    monkeypatch.setitem(_wrapper._BACKEND_TYPES, "fa2", ObservedFA2)
    q_lens, kv_lens = (2, 2), (128, 128)
    metadata, query, cache, table, offsets = _inputs(q_lens, kv_lens, dtype, 2)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.zeros(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="auto",
        use_cuda_graph=True,
        qo_indptr=torch.empty(3, dtype=torch.int32, device="cuda"),
        kv_indptr=torch.empty(3, dtype=torch.int32, device="cuda"),
        kv_indices=torch.empty(4, dtype=torch.int32, device="cuda"),
        kv_len_arr=torch.empty(2, dtype=torch.int32, device="cuda"),
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=_HEADS,
        head_dim_ckv=_CKV,
        head_dim_kpe=_KPE,
        page_size=_PAGE,
        causal=True,
        sm_scale=_SCALE,
        q_data_type=dtype,
        kv_data_type=dtype,
        output_dtype=dtype,
        query_layout="packed",
        kv_cache_layout="packed",
        lse_mode="none",
    )
    assert attempts == ["cutlass", "fa2"]
    assert policy_calls == [order]
    assert wrapper._planned_backend_name == "fa2"
    selected = wrapper._planned_backend

    def forbidden(*args, **kwargs):
        pytest.fail(
            "execution or graph capture entered policy, planning or compilation"
        )

    # Install guards immediately after plan, before even the first warmup run.
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden)
    monkeypatch.setattr(_auto_policy, "determine_mla_backend", forbidden)
    monkeypatch.setattr(wrapper, "plan", forbidden)
    monkeypatch.setattr(selected, "plan", forbidden)
    monkeypatch.setattr(fa2_backend, "get_batch_mla_module", forbidden)
    for backend_type in set(_wrapper._BACKEND_TYPES.values()):
        monkeypatch.setattr(backend_type, "plan_from_wrapper", classmethod(forbidden))
        if hasattr(backend_type, "preflight_plan_from_wrapper"):
            monkeypatch.setattr(
                backend_type, "preflight_plan_from_wrapper", classmethod(forbidden)
            )

    out = torch.empty((sum(q_lens), _HEADS, _CKV), dtype=dtype, device="cuda")
    output_pointer = out.data_ptr()
    expected, _ = _reference(query, cache, table, offsets, kv_lens, causal=True)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            assert wrapper.run(query=query, kv_cache=cache, out=out) is out
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(query=query, kv_cache=cache, out=out)
    assert captured is out
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)

    # New contents at the same addresses distinguish real replay from a stale
    # warmup result, without invoking the out-of-scope graph-update API.
    query.add_(0.125)
    cache.mul_(0.5)
    changed_expected, _ = _reference(query, cache, table, offsets, kv_lens, causal=True)
    assert not torch.allclose(changed_expected, expected, rtol=1e-2, atol=1e-2)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    assert out.data_ptr() == output_pointer
    torch.testing.assert_close(out.float(), changed_expected, rtol=1e-2, atol=1e-2)
    assert wrapper._planned_backend is selected
    assert wrapper._planned_backend_name == "fa2"
    assert attempts == ["cutlass", "fa2"]
    assert policy_calls == [order]


@pytest.mark.usefixtures("_sm100_reference_precision")
@pytest.mark.parametrize("layout", ["packed", "adjacent"])
def test_auto_large_prefill_fp8_graph_matches_reference(layout):
    # Preserve the failing benchmark's exact RNG path, page order and geometry.
    # Creating BF16 random tensors before FP8 conversion is intentional.
    torch.manual_seed(4031)
    heads, batch, length = 64, 4, 256
    query = torch.randn(
        batch * length, heads, _CKV + _KPE, dtype=torch.bfloat16, device="cuda"
    ).to(torch.float8_e4m3fn)
    cache = torch.randn(
        batch * length // _PAGE,
        _PAGE,
        _CKV + _KPE,
        dtype=torch.bfloat16,
        device="cuda",
    ).to(torch.float8_e4m3fn)
    offsets = list(range(0, (batch + 1) * length, length))
    qo = torch.tensor(offsets, dtype=torch.int32, device="cuda")
    kv_indptr = torch.arange(0, 17, 4, dtype=torch.int32, device="cuda")
    kv_indices = torch.arange(16, dtype=torch.int32, device="cuda")
    kv_lens = torch.full((batch,), length, dtype=torch.int32, device="cuda")
    table = kv_indices.reshape(batch, length // _PAGE)
    metadata = MLAPlanMetadata.dual(
        qo_indptr=qo,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_lens,
        cum_seq_lens_q=qo,
        block_tables=table,
        seq_lens=kv_lens,
        max_q_len=length,
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="auto",
        use_cuda_graph=True,
        qo_indptr=torch.empty_like(qo),
        kv_indptr=torch.empty_like(kv_indptr),
        kv_indices=torch.empty_like(kv_indices),
        kv_len_arr=torch.empty_like(kv_lens),
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=heads,
        head_dim_ckv=_CKV,
        head_dim_kpe=_KPE,
        page_size=_PAGE,
        causal=True,
        sm_scale=_SCALE,
        q_data_type=torch.float8_e4m3fn,
        kv_data_type=torch.float8_e4m3fn,
        output_dtype=torch.bfloat16,
        query_layout="packed",
        kv_cache_layout="packed",
        lse_mode="none",
        scale_mode="default",
    )
    runtime_query = (
        query if layout == "packed" else (query[..., :_CKV], query[..., _CKV:])
    )
    runtime_cache = (
        cache if layout == "packed" else (cache[..., :_CKV], cache[..., _CKV:])
    )
    out = torch.empty(
        (batch * length, heads, _CKV), dtype=torch.bfloat16, device="cuda"
    )
    expected, _ = _reference(
        query, cache, table, offsets, (length,) * batch, causal=True
    )
    scales = {}
    if wrapper._input_contract.scale_mode == "kv-per-tensor":
        scales = dict(ckv_scale=1.0, kpe_scale=1.0)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        for _ in range(3):
            assert (
                wrapper.run(
                    query=runtime_query, kv_cache=runtime_cache, out=out, **scales
                )
                is out
            )
    torch.cuda.current_stream().wait_stream(side_stream)
    torch.cuda.synchronize()
    # Numerical assertion comes first so the original CuTe accuracy failure
    # remains a numerical RED, rather than merely a backend-name mismatch.
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=0.05)
    assert wrapper._planned_backend_name == "trtllm-gen"
    assert wrapper._input_contract.scale_mode in ("default", "kv-per-tensor")
    selected = wrapper._planned_backend
    output_pointer = out.data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wrapper.run(
            query=runtime_query, kv_cache=runtime_cache, out=out, **scales
        )
    assert captured is out
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=0.05)

    # Changing values at fixed addresses proves replay consumes live inputs.
    query.zero_()
    changed_expected, _ = _reference(
        query, cache, table, offsets, (length,) * batch, causal=True
    )
    assert not torch.allclose(changed_expected, expected, rtol=0.05, atol=0.05)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), changed_expected, rtol=0.05, atol=0.05)
    assert out.data_ptr() == output_pointer
    assert wrapper._planned_backend is selected
    assert wrapper._planned_backend_name == "trtllm-gen"
