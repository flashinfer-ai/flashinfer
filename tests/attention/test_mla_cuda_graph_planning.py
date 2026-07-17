"""CUDA-graph planning contracts for the public batch MLA wrapper."""

import gc
import weakref

import pytest
import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.mla import _core as mla_core
from flashinfer.mla._batch_mla import _core as batch_mla_core
from flashinfer.mla._batch_mla._backends import _fa_common


class _RecordingBackend:
    def __init__(self):
        self.plan_calls = 0

    def plan(self, **kwargs):
        self.plan_calls += 1


class _MetadataBuffer:
    def __init__(self, name, start, size, events, *, fail=False):
        self.name = name
        self._start = start
        self._size = size
        self._events = events
        self._fail = fail

    def data_ptr(self):
        return self._start

    def numel(self):
        return self._size

    def element_size(self):
        return 1

    def copy_(self, source, **kwargs):
        self._events.append(f"copy:{self.name}")
        if self._fail:
            raise RuntimeError(f"copy failed: {self.name}")
        return self


def _generated_fa_mechanics(workspace, *, use_cuda_graph=True):
    mechanics = object.__new__(_fa_common._BatchMLAGeneratedFaMechanics)
    mechanics._generated_fa_workspace = workspace
    mechanics._use_cuda_graph = use_cuda_graph
    return mechanics


def _dense_plan_args(backend_name):
    return dict(
        cum_seq_lens_q=torch.tensor([0, 1, 2], dtype=torch.int32),
        block_tables=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
        seq_lens=torch.tensor([64, 96], dtype=torch.int32),
        max_q_len=1,
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=64,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        qk_nope_head_dim=128 if backend_name == "trtllm-gen" else None,
    )


def _csr_plan_args():
    return dict(
        qo_indptr=torch.tensor([0, 1, 2], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_indices=torch.tensor([0, 1, 2, 3], dtype=torch.int32),
        kv_len_arr=torch.tensor([64, 96], dtype=torch.int32),
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=64,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )


@pytest.mark.parametrize(
    "backend_name", ("cutlass", "trtllm-gen", "cute-dsl", "xqa")
)
def test_cuda_graph_dense_backend_supports_first_plan_but_rejects_replan(
    monkeypatch, backend_name
):
    backend = _RecordingBackend()
    class_name = {
        "cutlass": "_BatchMLAPagedAttentionCutlassBackend",
        "trtllm-gen": "_BatchMLAPagedAttentionTrtllmGenBackend",
        "cute-dsl": "_BatchMLAPagedAttentionCuteDslBackend",
        "xqa": "_BatchMLAPagedAttentionXqaBackend",
    }[backend_name]
    monkeypatch.setattr(batch_mla_core, class_name, lambda *args: backend)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), use_cuda_graph=True, backend=backend_name
    )
    args = _dense_plan_args(backend_name)

    wrapper.plan(**args)
    assert wrapper._backend_impl is backend

    with pytest.raises(ValueError, match="CUDA graph.*replan"):
        # The guard must run before argument validation or metadata conversion.
        wrapper.plan(cum_seq_lens_q=object())

    assert wrapper._backend_impl is backend
    assert backend.plan_calls == 1
    assert not hasattr(wrapper, "_plan_in_progress")


def test_generated_fa_workspace_is_lazy_and_allocated_once(monkeypatch):
    allocations = []

    class _Buffer:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype

    def fake_empty(shape, *, dtype, device, pin_memory=False):
        buffer = _Buffer(shape, dtype)
        allocations.append((buffer, device, pin_memory))
        return buffer

    monkeypatch.setattr(_fa_common.torch, "empty", fake_empty)
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))

    first = workspace.get_buffers()
    second = workspace.get_buffers()

    assert first == second
    assert len(allocations) == 2
    assert allocations[0][0].shape == (8 * 1024 * 1024,)
    assert allocations[0][1] == torch.device("cpu")
    assert allocations[0][2] is False
    assert allocations[1][0].shape == allocations[0][0].shape
    assert allocations[1][0].dtype == allocations[0][0].dtype
    assert allocations[1][1] == "cpu"
    assert allocations[1][2] is True


def test_failed_generated_fa_replan_does_not_mutate_live_workspace(monkeypatch):
    allocations = []

    class _Buffer:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.contents = []

        def copy_(self, other, **kwargs):
            self.contents = list(other.contents)

    def fake_empty(shape, *, dtype, device, pin_memory=False):
        buffer = _Buffer(shape, dtype)
        allocations.append((buffer, device, pin_memory))
        return buffer

    monkeypatch.setattr(_fa_common.torch, "empty", fake_empty)
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))

    initial_plan_buffer, _ = workspace.get_buffers()
    initial_plan_buffer.contents = [1, 2, 3]
    live_buffer = workspace.commit_buffers(
        initial_plan_buffer, use_cuda_graph=True
    )
    previous_contents = list(live_buffer.contents)

    replan_buffer, _ = workspace.get_buffers()
    assert replan_buffer is not live_buffer
    replan_buffer.contents = [7, 8, 9]  # Native planning writes before failing.

    # Simulate a native-plan or pre-commit-callback failure by not committing.
    assert workspace.live_buffer is live_buffer
    assert live_buffer.contents == previous_contents
    assert workspace.get_buffers()[0] is replan_buffer
    assert len(allocations) == 3  # live device, pinned host, staging device


def test_generated_fa_workspace_commits_replans_by_mode(monkeypatch):
    class _Buffer:
        def __init__(self, shape, dtype):
            self.shape = shape
            self.dtype = dtype
            self.contents = []

        def copy_(self, other, **kwargs):
            self.contents = list(other.contents)

    monkeypatch.setattr(
        _fa_common.torch,
        "empty",
        lambda shape, *, dtype, device, pin_memory=False: _Buffer(shape, dtype),
    )
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))

    initial_buffer, _ = workspace.get_buffers()
    initial_buffer.contents = [1]
    assert (
        workspace.commit_buffers(initial_buffer, use_cuda_graph=False)
        is initial_buffer
    )

    non_graph_staging, _ = workspace.get_buffers()
    non_graph_staging.contents = [2]
    assert (
        workspace.commit_buffers(non_graph_staging, use_cuda_graph=False)
        is non_graph_staging
    )
    assert workspace.live_buffer is non_graph_staging
    assert workspace.get_buffers()[0] is initial_buffer

    graph_staging = workspace.get_buffers()[0]
    graph_staging.contents = [3]
    assert (
        workspace.commit_buffers(graph_staging, use_cuda_graph=True)
        is non_graph_staging
    )
    assert workspace.live_buffer is non_graph_staging
    assert non_graph_staging.contents == [3]


def test_generated_fa_metadata_preflight_rejects_overlap_but_allows_alias():
    events = []
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))
    mechanics = _generated_fa_mechanics(workspace)
    source = _MetadataBuffer("source", 100, 8, events)
    partial_overlap = _MetadataBuffer("reserved", 104, 8, events)

    with pytest.raises(ValueError, match="overlap"):
        mechanics._preflight_cuda_graph_metadata_copies(
            (("qo_indptr", partial_overlap, source),)
        )

    mechanics._preflight_cuda_graph_metadata_copies(
        (("qo_indptr", source, source),)
    )
    mechanics._commit_cuda_graph_metadata((("qo_indptr", source, source),))
    assert events == []


def test_generated_fa_metadata_preflight_runs_before_native_plan(monkeypatch):
    mechanics = object.__new__(_fa_common._BatchMLAGeneratedFaMechanics)
    mechanics._use_cuda_graph = True
    mechanics._validate_plan_metadata = lambda **kwargs: None
    mechanics._metadata_copy_pairs = lambda **kwargs: (("qo_indptr", object(), object()),)
    mechanics._preflight_cuda_graph_metadata_copies = lambda pairs: (_ for _ in ()).throw(
        ValueError("metadata overlap")
    )
    module_loads = []

    with pytest.raises(ValueError, match="metadata overlap"):
        mechanics._plan_generated_fa(
            module_loader=lambda: module_loads.append(True),
            qo_indptr=object(),
            kv_indptr=object(),
            kv_indices=object(),
            kv_len_arr=object(),
            num_heads=1,
            head_dim_ckv=512,
            page_size=1,
            causal=False,
            sm_scale=1.0,
            q_data_type=object(),
            kv_data_type=object(),
            use_profiler=False,
        )

    assert module_loads == []


def test_generated_fa_live_workspace_commit_follows_metadata_copies():
    events = []

    class _Workspace:
        def commit_buffers(self, planned_buffer, *, use_cuda_graph):
            events.append("commit:workspace")
            return planned_buffer

        def invalidate_after_partial_metadata_commit(self, name, error):
            raise AssertionError("successful metadata copies must not invalidate")

    mechanics = _generated_fa_mechanics(_Workspace())
    pairs = tuple(
        (
            name,
            _MetadataBuffer(name, start, 4, events),
            _MetadataBuffer(f"{name}-source", start + 100, 4, events),
        )
        for name, start in (("qo", 0), ("kv", 10), ("indices", 20), ("len", 30))
    )

    result = mechanics._commit_generated_fa_plan(object(), pairs)

    assert result is not None
    assert events == [
        "copy:qo",
        "copy:kv",
        "copy:indices",
        "copy:len",
        "commit:workspace",
    ]


def test_first_metadata_copy_failure_terminally_invalidates_wrapper(monkeypatch):
    events = []
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))
    mechanics = _generated_fa_mechanics(workspace)
    expected_error = RuntimeError("first metadata copy failed")
    destination = _MetadataBuffer("qo", 0, 4, events)

    def fail_first_copy(source, **kwargs):
        events.append("copy:qo")
        raise expected_error

    monkeypatch.setattr(destination, "copy_", fail_first_copy)
    pairs = (
        (
            "qo_indptr",
            destination,
            _MetadataBuffer("qo-source", 100, 4, events),
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        mechanics._commit_generated_fa_plan(object(), pairs)

    assert exc_info.value is expected_error
    wrapper = object.__new__(batch_mla_core.BatchMLAPagedAttentionWrapper)
    wrapper._generated_fa_workspace = workspace
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.plan()
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.run(None, None, None, None)
    assert events == ["copy:qo"]


def test_partial_metadata_commit_terminally_invalidates_wrapper():
    events = []
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))
    mechanics = _generated_fa_mechanics(workspace)
    pairs = (
        (
            "qo_indptr",
            _MetadataBuffer("qo", 0, 4, events),
            _MetadataBuffer("qo-source", 100, 4, events),
        ),
        (
            "kv_indptr",
            _MetadataBuffer("kv", 10, 4, events, fail=True),
            _MetadataBuffer("kv-source", 110, 4, events),
        ),
    )

    with pytest.raises(RuntimeError, match="copy failed: kv"):
        mechanics._commit_generated_fa_plan(object(), pairs)

    wrapper = object.__new__(batch_mla_core.BatchMLAPagedAttentionWrapper)
    wrapper._generated_fa_workspace = workspace
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.plan()
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.run(None, None, None, None)
    assert events == ["copy:qo", "copy:kv"]


def test_scheduler_commit_failure_terminally_invalidates_wrapper(monkeypatch):
    events = []
    workspace = _fa_common._BatchMLAGeneratedFaWorkspace(torch.device("cpu"))
    mechanics = _generated_fa_mechanics(workspace)
    expected_error = RuntimeError("scheduler commit failed")

    def fail_scheduler_commit(planned_buffer, *, use_cuda_graph):
        events.append("commit:workspace")
        raise expected_error

    monkeypatch.setattr(workspace, "commit_buffers", fail_scheduler_commit)
    pairs = (
        (
            "qo_indptr",
            _MetadataBuffer("qo", 0, 4, events),
            _MetadataBuffer("qo-source", 100, 4, events),
        ),
    )

    with pytest.raises(RuntimeError) as exc_info:
        mechanics._commit_generated_fa_plan(object(), pairs)

    assert exc_info.value is expected_error
    wrapper = object.__new__(batch_mla_core.BatchMLAPagedAttentionWrapper)
    wrapper._generated_fa_workspace = workspace
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.plan()
    with pytest.raises(RuntimeError, match="terminally invalidated"):
        wrapper.run(None, None, None, None)
    assert events == ["copy:qo", "commit:workspace"]


@pytest.mark.parametrize(
    ("backend_name", "class_name"),
    (
        ("fa2", "_BatchMLAPagedAttentionFa2Backend"),
        ("fa3", "_BatchMLAPagedAttentionFa3Backend"),
    ),
)
def test_generated_fa_replans_share_wrapper_workspace_and_replace_backend(
    monkeypatch, backend_name, class_name
):
    instance_refs = []
    workspaces = []

    class _FaBackend(_RecordingBackend):
        def __init__(self, float_workspace, generated_fa_workspace, *args):
            super().__init__()
            workspaces.append(generated_fa_workspace)
            instance_refs.append(weakref.ref(self))

    monkeypatch.setattr(batch_mla_core, class_name, _FaBackend)
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), use_cuda_graph=True, backend=backend_name
    )
    args = _csr_plan_args()

    wrapper.plan(**args)
    first_backend_ref = instance_refs[0]
    wrapper.plan(**args)
    gc.collect()

    assert len(instance_refs) == 2
    assert workspaces[0] is wrapper._generated_fa_workspace
    assert workspaces[1] is wrapper._generated_fa_workspace
    assert wrapper._backend_impl is instance_refs[1]()
    assert first_backend_ref() is None


def test_auto_cuda_graph_replan_stays_on_selected_fa_backend(monkeypatch):
    instances = []
    cutlass_plan_calls = 0

    class _FaBackend(_RecordingBackend):
        def __init__(self, float_workspace, generated_fa_workspace, *args):
            super().__init__()
            instances.append(self)

        def plan(self, **kwargs):
            super().plan(**kwargs)
            if len(instances) == 2:
                raise _BackendPlanUnsupportedError("second FA2 plan rejected")

    def reject_cutlass_fallback(self, plan_args):
        nonlocal cutlass_plan_calls
        cutlass_plan_calls += 1
        raise AssertionError("CUDA-graph FA replan must not fall back to CUTLASS")

    monkeypatch.setattr(batch_mla_core, "determine_mla_backend", lambda device: "fa2")
    monkeypatch.setattr(batch_mla_core, "_BatchMLAPagedAttentionFa2Backend", _FaBackend)
    monkeypatch.setattr(
        batch_mla_core.BatchMLAPagedAttentionWrapper,
        "_plan_cutlass",
        reject_cutlass_fallback,
    )
    wrapper = mla_core.BatchMLAPagedAttentionWrapper(
        torch.empty(1), use_cuda_graph=True, backend="auto"
    )
    args = _csr_plan_args()

    wrapper.plan(**args)
    successful_backend = wrapper._backend_impl
    successful_metadata = wrapper._csr_plan_metadata

    with pytest.raises(_BackendPlanUnsupportedError, match="second FA2 plan rejected"):
        wrapper.plan(**args)

    assert cutlass_plan_calls == 0
    assert wrapper._selected_backend == "fa2"
    assert wrapper._backend_impl is successful_backend
    assert wrapper._csr_plan_metadata is successful_metadata
