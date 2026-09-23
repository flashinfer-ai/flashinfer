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
                # Distinguish the published executable from a replacement plan.
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


@pytest.fixture
def _cutile_dependency(monkeypatch):
    import importlib.metadata

    from flashinfer.cutile import cutile_common
    from flashinfer.mla._batch_mla._backends import cutile_backend

    state = SimpleNamespace(version="1.4.0", probes=[])
    original_version = importlib.metadata.version

    def installed_version(name):
        if name != "cuda-tile":
            return original_version(name)
        if state.version is None:
            raise importlib.metadata.PackageNotFoundError(name)
        return state.version

    def available():
        state.probes.append("compiler")
        return True

    monkeypatch.setattr(importlib.metadata, "version", installed_version)
    monkeypatch.setattr(cutile_common, "is_cuda_tile_available", available)
    cutile_backend.get_cutile_mla_decode.cache_clear()
    try:
        yield state
    finally:
        cutile_backend.get_cutile_mla_decode.cache_clear()


@pytest.mark.parametrize(
    "version,supported",
    [
        (None, False),
        ("1.3.0", False),
        ("1.4.0rc1", False),
        ("invalid", False),
        ("1.4.0", True),
        ("1.4.1", True),
        ("1.10.0", True),
        ("9.9.99.dev1", True),
    ],
)
def test_cpu_cutile_minimum_version(_cutile_dependency, version, supported):
    from flashinfer.mla._batch_mla._backends import cutile_backend
    from flashinfer.mla._batch_mla._backends._cutile_prepared import (
        prepare_cutile_mla_decode,
    )

    _cutile_dependency.version = version
    if supported:
        assert cutile_backend.get_cutile_mla_decode() is prepare_cutile_mla_decode
        assert _cutile_dependency.probes == ["compiler"]
    else:
        with pytest.raises(_BackendPlanUnsupportedError, match="cuda-tile>=1.4"):
            cutile_backend.get_cutile_mla_decode()
        assert _cutile_dependency.probes == []


@pytest.mark.parametrize(
    "backend,opt_in,version,expected",
    [
        ("auto", None, "1.4.0", "cutlass"),
        ("auto", "0", "1.4.0", "cutlass"),
        ("auto", "1", "1.4.0", "cutile"),
        ("auto", "1", "1.3.0", "cutlass"),
        ("cutile", None, "1.4.0", "cutile"),
        ("cutile", "0", "1.4.0", "cutile"),
        ("cutile", None, "1.3.0", None),
    ],
)
def test_cpu_cutile_version_and_experimental_selection(
    _cpu_planners, _cutile_dependency, monkeypatch, backend, opt_in, version, expected
):
    from flashinfer.mla._batch_mla._backends import cutile_backend

    state = _cpu_planners
    # Use the real cuTile declaration; only native preparation is replaced.
    monkeypatch.setattr(
        state.module._BACKEND_TYPES["cutile"],
        "_plan_capabilities",
        cutile_backend._BatchMLAPagedAttentionCutileBackend._plan_capabilities,
    )
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    if opt_in is not None:
        monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", opt_in)
    _cutile_dependency.version = version
    emitted = []
    monkeypatch.setattr(
        _wrapper,
        "warn_experimental_backend_once",
        lambda api, name, *, automatic: emitted.append((name, automatic)),
    )

    def prepare(name, args):
        if name == "cutile":
            cutile_backend.get_cutile_mla_decode()
        elif name != "cutlass":
            _reject(name, args)

    state.handler = prepare
    wrapper, kwargs, _ = _cpu_request(backend)
    if expected is None:
        with pytest.raises(_BackendPlanUnsupportedError, match="cuda-tile>=1.4"):
            wrapper.plan(**kwargs)
        assert wrapper._planned_backend is None
    else:
        wrapper.plan(**kwargs)
        assert wrapper._planned_backend_name == expected
    assert emitted == ([("cutile", backend == "auto")] if expected == "cutile" else [])
    if backend == "auto" and opt_in != "1":
        assert "cutile" not in state.calls
        assert _cutile_dependency.probes == []


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
    state.calls.clear()
    failure = error_type("legacy planner failure after writes")

    if graph:

        def forbidden_order(args):
            pytest.fail("Graph replan must retain its prepared backend")

        monkeypatch.setattr(_auto_policy, "_get_compute_capability", forbidden_order)
        monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", forbidden_order)
        monkeypatch.setattr(_auto_policy, "determine_mla_backend", forbidden_order)

    def fail(name, args):
        assert name == "fa3"
        if graph:
            assert (
                args._graph_plan_int_workspace_buffer is previous._int_workspace_buffer
            )
        raise failure

    state.handler = fail
    with pytest.raises(error_type) as caught:
        wrapper.plan(**kwargs)
    assert caught.value is failure
    assert state.calls == ["fa3"], "Legacy auto must not fall back to FA2"
    assert wrapper._planned_backend is previous
    assert wrapper._input_contract is contract
    assert wrapper._backend == "fa3"
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
def test_cpu_selector_continues_after_typed_rejection(
    _cpu_planners, monkeypatch, selector
):
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

    def reject_first(name, args):
        if name == names[0]:
            _reject(name, args)

    state.handler = reject_first
    wrapper, kwargs, _ = _cpu_request(selector)
    wrapper.plan(**kwargs)
    assert state.calls == list(names)
    assert wrapper._planned_backend_name == names[1]
    state.forbidden = True
    _cpu_run(wrapper)


def test_cpu_experimental_warning_failure_does_not_publish_or_rollback(
    _cpu_planners, monkeypatch
):
    from dataclasses import replace

    state = _cpu_planners
    wrapper, kwargs, buffers = _cpu_request("fa2")
    wrapper.plan(**kwargs)
    previous = wrapper._planned_backend
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
        assert all(torch.all(buffer == 91) for buffer in buffers)
        raise UserWarning("experimental warning promoted to error")

    state.handler = mutate
    monkeypatch.setattr(_wrapper, "warn_experimental_backend_once", warning_error)
    with pytest.raises(UserWarning, match="promoted to error"):
        wrapper.plan(**kwargs)
    assert wrapper._planned_backend is previous
    # The warning is outside backend rollback, but prevents plan publication.
    assert all(torch.all(buffer == 91) for buffer in buffers)


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
@pytest.mark.parametrize("missing", [(), ("sm",), ("block",), ("sm", "block")])
def test_cpu_fa_shared_memory_compatibility(monkeypatch, backend, missing):
    from flashinfer.mla._batch_mla._backends import _fa_common as fa

    driver = pytest.importorskip("cuda.bindings.driver")
    values = {
        "shared_memory_per_multiprocessor": 232448,
        "shared_memory_per_block_optin": 227328,
    }
    names = {
        "sm": "shared_memory_per_multiprocessor",
        "block": "shared_memory_per_block_optin",
    }
    properties = SimpleNamespace(
        **{k: v for k, v in values.items() if k not in [names[n] for n in missing]}
    )
    monkeypatch.setattr(fa, "get_device_properties", lambda _: properties)
    monkeypatch.setattr(fa, "get_compute_capability", lambda _: (9, 0))
    monkeypatch.setattr(fa, "is_sm90a_supported", lambda _: True)
    queried = []
    attributes = driver.CUdevice_attribute
    limits = {
        attributes.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR: 232448,
        attributes.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: 227328,
    }

    def query(attribute, index):
        assert index == 3
        queried.append(attribute)
        return (0, limits[attribute])

    monkeypatch.setattr(driver, "cuDeviceGetAttribute", query)
    tensors = dict(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_len_arr=torch.tensor([1], dtype=torch.int32),
    )
    kwargs = dict(
        backend=backend,
        device=torch.device("cuda:3"),
        head_dim_ckv=512,
        head_dim_kpe=64,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        output_dtype=torch.bfloat16,
        scale_mode="default",
        **tensors,
    )
    fa._validate_generated_fa_plan(**kwargs)
    # Odd heads exercise the second shared-memory consumer, beyond preflight.
    fa._validate_fa_causal_tile_bound(
        [5], [10], 17, backend, torch.bfloat16, torch.device("cuda:3")
    )
    assert bool(queried) == bool(missing)
    # Preserve the capacity rejection even when torch lacks the properties.
    if "block" in missing:
        limits[attributes.CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN] = 1
    else:
        properties.shared_memory_per_block_optin = 1
    with pytest.raises(_BackendPlanUnsupportedError, match="shared-memory bytes"):
        fa._validate_generated_fa_plan(**kwargs)


@pytest.mark.parametrize("q_len", [65535, 65536])
@pytest.mark.parametrize("grid", ["main", "reducer"])
def test_cpu_monolithic_grid_boundary_rejects_before_compile(
    _cpu_planners, monkeypatch, q_len, grid
):
    from flashinfer.cute_dsl import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("CuTe DSL is unavailable")

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
        native,
        "_get_split_kv_and_workspace_size",
        lambda *args, **kwargs: (2, 16) if grid == "reducer" else (1, 0),
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
            num_heads=8 if grid == "reducer" else 128,
            q_len=q_len,
            head_dim_ckv=512,
            head_dim_kpe=64,
            resolved_is_var_seq=True,
            is_var_q=grid == "reducer",
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


@pytest.mark.parametrize("failure", [_BackendPlanUnsupportedError, RuntimeError])
def test_cpu_failed_replan_preserves_published_executable(
    _cpu_planners, monkeypatch, failure
):
    state = _cpu_planners
    order = ("fa2", "cutlass", "trtllm-gen")
    monkeypatch.setattr(_auto_policy, "ordered_sm100_backends", lambda args: order)
    wrapper, kwargs, _ = _cpu_request()
    wrapper.plan(**kwargs)
    previous, contract = wrapper._planned_backend, wrapper._input_contract
    output = _cpu_run(wrapper).clone()
    mirrors = (
        wrapper._cached_module,
        wrapper._int_workspace_buffer,
        wrapper._pin_memory_int_workspace_buffer,
    )
    state.calls.clear()

    def reject(name, args):
        if name == "cutlass":
            raise failure("second candidate failed")
        _reject(name, args)

    state.handler = reject
    with pytest.raises(failure):
        wrapper.plan(**kwargs)
    assert state.calls == list(
        order if failure is _BackendPlanUnsupportedError else order[:2]
    )
    assert wrapper._planned_backend is previous
    assert wrapper._input_contract is contract
    assert wrapper._planned_backend_name == wrapper._backend == "fa2"
    assert all(
        actual is expected
        for actual, expected in zip(
            (
                wrapper._cached_module,
                wrapper._int_workspace_buffer,
                wrapper._pin_memory_int_workspace_buffer,
            ),
            mirrors,
            strict=True,
        )
    )
    state.forbidden = True
    torch.testing.assert_close(_cpu_run(wrapper), output)


def test_modular_volume_preserves_native_prefix():
    args, _ = _request((2, 2), (40960, 40960), heads=16)
    assert _order(args)[:4] == (
        "cute-dsl-monolithic",
        "trtllm-gen",
        "cute-dsl-modular",
        "fa2",
    )


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
@pytest.mark.parametrize("backend", ["fa2", "auto"])
def test_fa_plan_without_torch_shared_memory_properties(monkeypatch, backend):
    from flashinfer.mla._batch_mla._backends import _fa_common as fa

    monkeypatch.setattr(fa, "get_device_properties", lambda _: SimpleNamespace())
    # Exercise the legacy auto branch on this GPU, with real FA planning/run.
    monkeypatch.setattr(_auto_policy, "_get_compute_capability", lambda _: (8, 0))
    _check_case(backend, (2,), (9,), 4, torch.bfloat16, causal=False)


@pytest.mark.usefixtures("_sm100_reference_precision")
def test_auto_falls_back_when_cutile_library_budget_is_full(monkeypatch):
    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    pytest.importorskip("cuda.tile.compilation")
    from flashinfer.cutile.cutile_common import is_cuda_tile_available

    if not is_cuda_tile_available():
        pytest.skip("cuTile compiler toolchain is unavailable")

    from flashinfer.mla._batch_mla._backends import _cutile_prepared as prepared

    monkeypatch.setattr(prepared, "_LOADED_LIBRARIES", {})
    monkeypatch.setattr(prepared, "_MAX_LOADED_LIBRARIES", 0)
    load_kernel = prepared._load_kernel
    attempted = []

    def load(*args):
        attempted.append(True)
        return load_kernel(*args)

    monkeypatch.setattr(prepared, "_load_kernel", load)
    monkeypatch.setattr(
        _auto_policy, "ordered_sm100_backends", lambda _: ("cutile", "fa2")
    )
    _check_case("auto", (1,), (9,), 4, torch.bfloat16, causal=False)
    assert attempted == [True]


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
@pytest.mark.parametrize(
    "dtype,kv_len,initial_lengths,replay_lengths",
    [
        pytest.param(
            torch.bfloat16,
            128,
            (1, 3, 7),
            [(1, 1, 9), (3, 4, 4)],
            id="reported-unsplit",
        ),
        pytest.param(
            torch.bfloat16, 4096, (1, 3, 7), [(1, 1, 9), (7, 3, 1)], id="reported-split"
        ),
        pytest.param(
            torch.float16,
            128,
            (4, 6, 8),
            [(1, 1, 16), (6, 6, 6)],
            id="capacity-unsplit-fp16",
        ),
        pytest.param(
            torch.bfloat16,
            4096,
            (4, 6, 8),
            [(1, 1, 16), (8, 6, 4)],
            id="capacity-split",
        ),
        pytest.param(torch.bfloat16, 4096, (2, 2, 2), [], id="uniform-fixed-query"),
    ],
)
def test_monolithic_graph_query_capacity(
    dtype, kv_len, initial_lengths, replay_lengths
):
    capacity = 16
    kv_lens = (kv_len,) * len(initial_lengths)
    metadata, query, cache, table, offsets = _inputs(
        initial_lengths, kv_lens, dtype, capacity
    )
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="cute-dsl-monolithic",
        use_cuda_graph=True,
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
        lse_mode="basee",
    )
    selected = wrapper._planned_backend
    state = selected._execution_state
    assert (state.split_kv > 1) == (kv_len > 128)
    out = torch.empty((sum(initial_lengths), _HEADS, _CKV), dtype=dtype, device="cuda")
    lse = torch.empty(
        (sum(initial_lengths), _HEADS), dtype=torch.float32, device="cuda"
    )
    out_pointer, lse_pointer = out.data_ptr(), lse.data_ptr()

    def run():
        return wrapper.run(
            query=query,
            kv_cache=cache,
            out=out,
            lse=lse,
            return_lse=True,
            return_lse_base_on_e=True,
        )

    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        run()
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run()
    assert captured[0] is out and captured[1] is lse

    for lengths in [initial_lengths, *replay_lengths]:
        offsets = [0]
        for length in lengths:
            offsets.append(offsets[-1] + length)
        metadata.cum_seq_lens_q.copy_(
            torch.tensor(offsets, dtype=torch.int32, device="cuda")
        )
        expected, expected_lse = _reference(
            query, cache, table, offsets, kv_lens, causal=True
        )
        # Missing tiles/reducer rows must fail instead of retaining old output.
        out.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(lse, expected_lse, rtol=1e-2, atol=1e-2)
        assert out.data_ptr() == out_pointer and lse.data_ptr() == lse_pointer
        assert wrapper._planned_backend is selected

    # An initially uniform plan must retain its fixed-query tensor views.
    if len(set(initial_lengths)) == 1:
        assert state.cum_seq_lens_q is None
        assert state.q_len == initial_lengths[0]
    else:
        assert state.cum_seq_lens_q.data_ptr() == metadata.cum_seq_lens_q.data_ptr()
        assert state.q_len == capacity


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
def test_auto_graph_falls_back_without_fa_reserved_buffers(monkeypatch):
    attempts = []
    for name in ("fa2", "trtllm-gen"):
        backend_type = _wrapper._BACKEND_TYPES[name]
        original_plan = backend_type.plan_from_wrapper

        def record_attempt(cls, args, original_plan=original_plan, name=name):
            attempts.append(name)
            return original_plan(args)

        monkeypatch.setattr(
            backend_type, "plan_from_wrapper", classmethod(record_attempt)
        )
    torch.manual_seed(5463)
    query = torch.randn(2, 64, _CKV + _KPE, device="cuda", dtype=torch.bfloat16)
    cache = torch.randn(1, 32, _CKV + _KPE, device="cuda", dtype=torch.bfloat16)
    table = torch.zeros((1, 2), dtype=torch.int32, device="cuda")
    metadata = MLAPlanMetadata.dense(
        cum_seq_lens_q=torch.tensor([0, 2], dtype=torch.int32, device="cuda"),
        block_tables=table,
        seq_lens=torch.tensor([32], dtype=torch.int32, device="cuda"),
        max_q_len=2,
    )
    # Exercise the real base-2 policy branch that ranks FA2 before TRT.
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda"),
        backend="auto",
        use_cuda_graph=True,
    )
    wrapper.plan(
        metadata=metadata,
        num_heads=64,
        head_dim_ckv=_CKV,
        head_dim_kpe=_KPE,
        page_size=32,
        causal=True,
        sm_scale=_SCALE,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        output_dtype=torch.bfloat16,
        lse_mode="base2",
    )
    assert attempts == ["fa2", "trtllm-gen"]
    assert wrapper._planned_backend_name == "trtllm-gen"
    expected, expected_lse = _reference(query, cache, table, [0, 2], (32,), causal=True)
    out = torch.empty_like(expected, dtype=torch.bfloat16)
    lse = torch.empty_like(expected_lse)
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side_stream):
        wrapper.run(query=query, kv_cache=cache, out=out, lse=lse, return_lse=True)
    torch.cuda.current_stream().wait_stream(side_stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(query=query, kv_cache=cache, out=out, lse=lse, return_lse=True)
    out.fill_(float("nan"))
    lse.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(lse, expected_lse / math.log(2), rtol=1e-2, atol=1e-2)


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
    # Native TRT FP8 attention-weight rounding needs a slightly larger absolute
    # tolerance; preserve the stricter check for every other backend.
    atol = 0.06 if wrapper._planned_backend_name == "trtllm-gen" else 0.05
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=atol)
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
    torch.testing.assert_close(out.float(), expected, rtol=0.05, atol=atol)

    # Changing values at fixed addresses proves replay consumes live inputs.
    query.zero_()
    changed_expected, _ = _reference(
        query, cache, table, offsets, (length,) * batch, causal=True
    )
    assert not torch.allclose(changed_expected, expected, rtol=0.05, atol=atol)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), changed_expected, rtol=0.05, atol=atol)
    assert out.data_ptr() == output_pointer
    assert wrapper._planned_backend is selected
    assert wrapper._planned_backend_name == "trtllm-gen"
