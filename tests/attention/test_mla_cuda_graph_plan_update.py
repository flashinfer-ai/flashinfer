"""Focused lifecycle and transaction coverage for MLA CUDA-graph plan updates."""

import ctypes
import importlib
import inspect
import re
import time
from contextlib import nullcontext
from types import ModuleType
from threading import Event
import pytest
import torch

from flashinfer.mla import MLAPlanMetadata
from flashinfer.mla._batch_mla._backends._capabilities import MLAPlanCapabilities
from flashinfer.mla._batch_mla._backends._fa_common import (
    _BatchMLAPagedAttentionFaBackendBase,
)
from flashinfer.mla._batch_mla._wrapper import BatchMLAPagedAttentionWrapper
from flashinfer.utils import is_sm90a_supported


class _RecordingPlannedBackend:
    def __init__(self, *, backend_name: str = "recording", fail: bool = False) -> None:
        self._backend = backend_name
        self._plan_capabilities = type(
            "Capabilities", (), {"supports_cuda_graph_plan_update": True}
        )()
        self._fail = fail
        self.metadata_calls: list[MLAPlanMetadata] = []

    def update_cuda_graph_plan_from_wrapper(self, *, metadata: MLAPlanMetadata) -> None:
        self.metadata_calls.append(metadata)
        if self._fail:
            raise ValueError("backend update failed")


class _DefaultPlannedBackend:
    _backend = "recording"
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="recording",
        lse_modes=frozenset(),
        kv_layouts=frozenset(),
        output_scales=frozenset(),
        scale_modes=frozenset(),
    )


class _FrozenBackend:
    pass


def _update_module() -> ModuleType:
    return importlib.import_module(
        "flashinfer.mla._batch_mla._backends._fa_cuda_graph_plan_update"
    )


def _metadata(*, kv_indices: torch.Tensor | None = None) -> MLAPlanMetadata:
    return MLAPlanMetadata.csr(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=(
            torch.tensor([3], dtype=torch.int32) if kv_indices is None else kv_indices
        ),
        kv_len_arr=torch.tensor([1], dtype=torch.int32),
    )


def _wrapper(
    backend: object | None,
    *,
    use_cuda_graph: bool = True,
    enable_update: bool = True,
) -> BatchMLAPagedAttentionWrapper:
    wrapper = BatchMLAPagedAttentionWrapper.__new__(BatchMLAPagedAttentionWrapper)
    wrapper._use_cuda_graph = use_cuda_graph
    wrapper._enable_cuda_graph_plan_update = enable_update
    wrapper._planned_backend = backend
    wrapper._backend = "fa2"
    wrapper.device = torch.device("cuda:0")
    wrapper._cuda_graph_plan_update_in_progress = False
    wrapper._cuda_graph_plan_update_stream = None
    return wrapper


def _patch_cuda_state(
    monkeypatch: pytest.MonkeyPatch, *, stream: int = 17, capturing: bool = False
) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device=None: type(
            "Stream",
            (),
            {"cuda_stream": stream, "device": torch.device("cuda:0")},
        )(),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)


def test_cuda_graph_plan_update_has_keyword_only_metadata_signature() -> None:
    assert str(
        inspect.signature(BatchMLAPagedAttentionWrapper.update_cuda_graph_plan)
    ) == (
        "(self, *, metadata: flashinfer.mla._batch_mla._contracts.MLAPlanMetadata) -> None"
    )


def test_cuda_graph_plan_update_constructor_requires_explicit_graph_opt_in() -> None:
    workspace = torch.empty(1, dtype=torch.uint8)
    wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa2")
    assert wrapper._enable_cuda_graph_plan_update is False

    with pytest.raises(
        ValueError,
        match=("^enable_cuda_graph_plan_update=True requires use_cuda_graph=True\\.$"),
    ):
        BatchMLAPagedAttentionWrapper(
            workspace,
            backend="fa2",
            enable_cuda_graph_plan_update=True,
        )


def test_cuda_graph_plan_update_delegates_once_and_binds_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_cuda_state(monkeypatch, stream=17)
    backend = _RecordingPlannedBackend()
    wrapper = _wrapper(backend)
    metadata = _metadata()

    wrapper.update_cuda_graph_plan(metadata=metadata)

    assert backend.metadata_calls == [metadata]
    assert wrapper._cuda_graph_plan_update_stream == 17
    assert wrapper._cuda_graph_plan_update_in_progress is False


def test_cuda_graph_plan_update_rejects_non_cuda_before_stream_access(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingPlannedBackend()
    wrapper = _wrapper(backend)
    wrapper.device = torch.device("cpu")

    def unexpected_stream_access(*args, **kwargs):
        pytest.fail("non-CUDA update reached CUDA stream access")

    monkeypatch.setattr(torch.cuda, "current_stream", unexpected_stream_access)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        wrapper.update_cuda_graph_plan(metadata=_metadata())
    assert not backend.metadata_calls


def test_cuda_graph_replan_rejects_detectable_in_progress_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(16, dtype=torch.uint8),
        backend="fa2",
        use_cuda_graph=True,
        enable_cuda_graph_plan_update=True,
    )
    wrapper_module = importlib.import_module("flashinfer.mla._batch_mla._wrapper")
    monkeypatch.setattr(
        wrapper_module._BACKEND_TYPES["fa2"],
        "plan_from_wrapper",
        lambda args: _RecordingPlannedBackend(),
    )
    wrapper._cuda_graph_plan_update_in_progress = True
    with pytest.raises(RuntimeError, match="plan update is in progress"):
        wrapper.plan(
            metadata=_metadata(),
            num_heads=16,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=False,
            sm_scale=0.125,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )


@pytest.mark.parametrize(
    ("wrapper", "message"),
    [
        (
            _wrapper(_RecordingPlannedBackend(), use_cuda_graph=False),
            "update_cuda_graph_plan() requires use_cuda_graph=True.",
        ),
        (
            _wrapper(_RecordingPlannedBackend(), enable_update=False),
            "update_cuda_graph_plan() requires enable_cuda_graph_plan_update=True.",
        ),
        (
            _wrapper(None),
            "update_cuda_graph_plan() called before plan().",
        ),
        (
            _wrapper(_DefaultPlannedBackend()),
            "the planned backend does not support CUDA graph plan updates.",
        ),
    ],
    ids=("graph-mode", "explicit-opt-in", "initial-plan", "capability"),
)
def test_cuda_graph_plan_update_rejects_invalid_lifecycle(
    monkeypatch: pytest.MonkeyPatch,
    wrapper: BatchMLAPagedAttentionWrapper,
    message: str,
) -> None:
    _patch_cuda_state(monkeypatch)
    with pytest.raises(RuntimeError, match=f"^{re.escape(message)}$"):
        wrapper.update_cuda_graph_plan(metadata=_metadata())


def test_cuda_graph_plan_update_rejects_capture_concurrency_and_stream_change(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = _RecordingPlannedBackend()
    wrapper = _wrapper(backend)

    _patch_cuda_state(monkeypatch, capturing=True)
    with pytest.raises(RuntimeError, match="cannot run during CUDA graph capture"):
        wrapper.update_cuda_graph_plan(metadata=_metadata())

    _patch_cuda_state(monkeypatch, stream=17)
    wrapper.update_cuda_graph_plan(metadata=_metadata())
    _patch_cuda_state(monkeypatch, stream=29)
    with pytest.raises(RuntimeError, match="initially bound CUDA stream"):
        wrapper.update_cuda_graph_plan(metadata=_metadata())

    assert len(backend.metadata_calls) == 1


def _frozen_contract_fixture(
    *, device: torch.device | str = "cpu"
) -> tuple[object, object, dict[str, torch.Tensor]]:
    update = _update_module()
    device = torch.device(device)
    tensors = {
        "float_workspace": torch.empty((32,), dtype=torch.uint8, device=device),
        "int_workspace": torch.empty((64,), dtype=torch.uint8, device=device),
        "qo_indptr": torch.tensor([0, 2, 3], dtype=torch.int32, device=device),
        "kv_indptr": torch.tensor([0, 2, 4], dtype=torch.int32, device=device),
        "kv_indices": torch.tensor(
            [90, 91, 92, 93, 94, 95, 96, 97],
            dtype=torch.int32,
            device=device,
        ),
        "kv_len_arr": torch.tensor([3, 4], dtype=torch.int32, device=device),
    }
    module = object()
    contract = update._make_mla_cuda_graph_frozen_contract(
        backend_type=_FrozenBackend,
        module=module,
        device=device,
        float_workspace=tensors["float_workspace"],
        int_workspace=tensors["int_workspace"],
        qo_indptr=tensors["qo_indptr"],
        kv_indptr=tensors["kv_indptr"],
        kv_indices=tensors["kv_indices"],
        kv_len_arr=tensors["kv_len_arr"],
        batch_size=2,
        total_qo_rows=3,
        num_heads=8,
        head_dim_ckv=512,
        page_size=2,
        causal=True,
        sm_scale=0.125,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.float16,
        use_profiler=False,
        plan_info=[11, 22, 33, 44],
        staged_int_workspace_bytes=24,
    )
    return module, contract, tensors


def _valid_update_metadata(
    *,
    kv_indices: torch.Tensor,
    qo_indptr: torch.Tensor | None = None,
    kv_indptr: torch.Tensor | None = None,
    kv_len_arr: torch.Tensor | None = None,
) -> MLAPlanMetadata:
    return MLAPlanMetadata.csr(
        qo_indptr=(
            torch.tensor([0, 2, 3], dtype=torch.int32)
            if qo_indptr is None
            else qo_indptr
        ),
        kv_indptr=(
            torch.tensor([0, 2, 4], dtype=torch.int32)
            if kv_indptr is None
            else kv_indptr
        ),
        kv_indices=kv_indices,
        kv_len_arr=(
            torch.tensor([3, 4], dtype=torch.int32)
            if kv_len_arr is None
            else kv_len_arr
        ),
    )


def test_frozen_contract_snapshots_tensor_identity_and_static_plan() -> None:
    _, contract, tensors = _frozen_contract_fixture()

    for name, tensor in tensors.items():
        identity = getattr(contract, name)
        assert identity.object_id == id(tensor)
        assert identity.data_ptr == tensor.data_ptr()
        assert identity.shape == tuple(tensor.shape)
        assert identity.dtype == tensor.dtype
        assert identity.device == tensor.device
    assert contract.plan_info == (11, 22, 33, 44)
    assert contract.staged_int_workspace_bytes == 24


def test_csr_update_resolver_accepts_complete_host_controls_and_wrapper_indices() -> (
    None
):
    update = _update_module()
    _, contract, _ = _frozen_contract_fixture()
    indices = torch.tensor([-17, 2**30, 7, -1, 1234], dtype=torch.int32)

    resolved = update._resolve_mla_cuda_graph_plan_update(
        metadata=_valid_update_metadata(kv_indices=indices),
        frozen=contract,
    )

    assert resolved.kv_indices is indices
    assert resolved.live_kv_indices == 4


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        (
            MLAPlanMetadata(),
            "CUDA graph plan updates require complete CSR metadata only",
        ),
        (
            _valid_update_metadata(
                kv_indices=torch.arange(4, dtype=torch.int32),
                qo_indptr=torch.tensor([1, 2, 3], dtype=torch.int32),
            ),
            "qo_indptr must start at zero",
        ),
        (
            _valid_update_metadata(
                kv_indices=torch.arange(3, dtype=torch.int32),
            ),
            "kv_indices source has insufficient capacity",
        ),
        (
            _valid_update_metadata(
                kv_indices=torch.arange(4, dtype=torch.int32),
                kv_len_arr=torch.tensor([1, 4], dtype=torch.int32),
            ),
            "kv_indptr page counts must equal",
        ),
    ],
)
def test_csr_update_resolver_rejects_invalid_metadata(
    metadata: MLAPlanMetadata, message: str
) -> None:
    update = _update_module()
    _, contract, _ = _frozen_contract_fixture()
    with pytest.raises(ValueError, match=message):
        update._resolve_mla_cuda_graph_plan_update(
            metadata=metadata,
            frozen=contract,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_csr_update_requires_device_non_alias_indices() -> None:
    update = _update_module()
    _, contract, tensors = _frozen_contract_fixture(device="cuda")

    with pytest.raises(ValueError, match="kv_indices must be on wrapper device"):
        update._resolve_mla_cuda_graph_plan_update(
            metadata=_valid_update_metadata(
                kv_indices=torch.arange(4, dtype=torch.int32)
            ),
            frozen=contract,
        )

    with pytest.raises(
        ValueError,
        match="kv_indices source overlaps reserved kv_indices",
    ):
        update._resolve_mla_cuda_graph_plan_update(
            metadata=_valid_update_metadata(kv_indices=tensors["kv_indices"]),
            frozen=contract,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_update_state_retains_one_candidate_and_no_index_images() -> None:
    update = _update_module()
    _, frozen, tensors = _frozen_contract_fixture(device="cuda")
    reusable = torch.empty(64, dtype=torch.uint8, device="cpu", pin_memory=True)

    state = update._make_mla_cuda_graph_plan_update_state(
        frozen=frozen,
        reusable_device_workspace=tensors["int_workspace"],
        reusable_planner_workspace=reusable,
        qo_indptr=tensors["qo_indptr"],
        kv_indptr=tensors["kv_indptr"],
        kv_len_arr=tensors["kv_len_arr"],
    )

    assert not hasattr(state, "committed")
    assert not hasattr(state.candidate, "kv_indices")
    assert state.candidate.int_workspace.numel() == 24
    assert (
        state.candidate.int_workspace.data_ptr()
        == tensors["int_workspace"].data_ptr() + 24
    )
    assert state.slots[0].planner_workspace.data_ptr() == reusable.data_ptr()
    assert state.slots[1].planner_workspace.data_ptr() == reusable.data_ptr() + 24
    assert all(not hasattr(slot, "kv_indices") for slot in state.slots)

    small_device_workspace = torch.empty(32, dtype=torch.uint8, device="cuda")
    small_planner_workspace = torch.empty(
        32, dtype=torch.uint8, device="cpu", pin_memory=True
    )
    fallback_state = update._make_mla_cuda_graph_plan_update_state(
        frozen=frozen,
        reusable_device_workspace=small_device_workspace,
        reusable_planner_workspace=small_planner_workspace,
        qo_indptr=tensors["qo_indptr"],
        kv_indptr=tensors["kv_indptr"],
        kv_len_arr=tensors["kv_len_arr"],
    )
    assert (
        fallback_state.candidate.int_workspace.untyped_storage().data_ptr()
        != small_device_workspace.untyped_storage().data_ptr()
    )
    assert (
        fallback_state.slots[0].planner_workspace.untyped_storage().data_ptr()
        == small_planner_workspace.untyped_storage().data_ptr()
    )
    assert (
        fallback_state.slots[1].planner_workspace.untyped_storage().data_ptr()
        != small_planner_workspace.untyped_storage().data_ptr()
    )


class _ReadyEvent:
    def __init__(self, order: list[str] | None = None) -> None:
        self.order = order
        self.query_calls = 0
        self.record_calls = 0

    def query(self) -> bool:
        self.query_calls += 1
        return True

    def record(self) -> None:
        self.record_calls += 1
        if self.order is not None:
            self.order.append("record")


class _FakeTransactionModule:
    def __init__(self) -> None:
        self.order: list[str] = []
        self.plan_info: tuple[int, ...] = (11, 22, 33, 44)
        self.staged_int_workspace_bytes = 24
        self.schedule_bytes = torch.full((24,), 211, dtype=torch.uint8)
        self.planner_error: Exception | None = None
        self.commit_error: Exception | None = None

    def plan_with_preallocated_staging(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
        page_locked_int_workspace_buffer: torch.Tensor,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_o: int,
        causal: bool,
    ) -> tuple[list[int], int]:
        del (
            float_workspace_buffer,
            int_workspace_buffer,
            qo_indptr,
            kv_indptr,
            kv_len_arr,
            num_heads,
            head_dim_o,
            causal,
        )
        self.order.append("plan")
        if self.planner_error is not None:
            raise self.planner_error
        page_locked_int_workspace_buffer.copy_(self.schedule_bytes)
        return list(self.plan_info), self.staged_int_workspace_bytes

    def commit_cuda_graph_plan_update(
        self,
        live_int_workspace: torch.Tensor,
        live_qo_indptr: torch.Tensor,
        live_kv_indptr: torch.Tensor,
        live_kv_indices: torch.Tensor,
        live_kv_len_arr: torch.Tensor,
        candidate_int_workspace: torch.Tensor,
        candidate_qo_indptr: torch.Tensor,
        candidate_kv_indptr: torch.Tensor,
        source_kv_indices: torch.Tensor,
        candidate_kv_len_arr: torch.Tensor,
        staged_int_workspace_bytes: int,
        live_kv_indices_length: int,
    ) -> None:
        self.order.append("commit")
        if self.commit_error is not None:
            raise self.commit_error
        live_int_workspace[:staged_int_workspace_bytes].copy_(
            candidate_int_workspace[:staged_int_workspace_bytes]
        )
        live_qo_indptr.copy_(candidate_qo_indptr)
        live_kv_indptr.copy_(candidate_kv_indptr)
        live_kv_indices[:live_kv_indices_length].copy_(
            source_kv_indices[:live_kv_indices_length]
        )
        live_kv_len_arr.copy_(candidate_kv_len_arr)


class _TransactionBackend(_BatchMLAPagedAttentionFaBackendBase):
    _plan_capabilities = MLAPlanCapabilities(
        backend_name="transaction",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"independent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default"}),
        supports_cuda_graph_plan_update=True,
    )


def _transaction_fixture() -> tuple[
    _TransactionBackend,
    BatchMLAPagedAttentionWrapper,
    _FakeTransactionModule,
]:
    update = _update_module()
    device = torch.device("cuda", torch.cuda.current_device())
    module = _FakeTransactionModule()
    float_workspace = torch.zeros((32,), dtype=torch.uint8, device=device)
    int_workspace = torch.arange(64, dtype=torch.uint8, device=device)
    qo_indptr = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    kv_indices = torch.tensor(
        [90, 91, 92, 93, 94, 95, 96, 97], dtype=torch.int32, device=device
    )
    kv_len_arr = torch.tensor([3, 4], dtype=torch.int32, device=device)
    frozen = update._make_mla_cuda_graph_frozen_contract(
        backend_type=_TransactionBackend,
        module=module,
        device=device,
        float_workspace=float_workspace,
        int_workspace=int_workspace,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=kv_len_arr,
        batch_size=2,
        total_qo_rows=3,
        num_heads=8,
        head_dim_ckv=512,
        page_size=2,
        causal=True,
        sm_scale=0.125,
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        use_profiler=False,
        plan_info=module.plan_info,
        staged_int_workspace_bytes=24,
    )
    state = update._make_mla_cuda_graph_plan_update_state(
        frozen=frozen,
        qo_indptr=qo_indptr,
        kv_indptr=kv_indptr,
        kv_len_arr=kv_len_arr,
    )
    state.slots[0].completion_event = _ReadyEvent(module.order)
    backend = _TransactionBackend.__new__(_TransactionBackend)
    backend._backend = "transaction"
    backend.device = device
    backend._float_workspace_buffer = float_workspace
    backend._int_workspace_buffer = int_workspace
    backend._qo_indptr_buf = qo_indptr
    backend._kv_indptr_buf = kv_indptr
    backend._kv_indices_buf = kv_indices
    backend._kv_len_arr_buf = kv_len_arr
    backend._cached_module = module
    backend._causal = True
    backend._page_size = 2
    backend._sm_scale = 0.125
    backend._head_dim_ckv = 512
    backend._q_data_type = torch.float16
    backend._kv_data_type = torch.float16
    backend._use_profiler = False
    backend._plan_info = list(module.plan_info)
    backend._staged_int_workspace_bytes = 24
    backend._cuda_graph_plan_update_state = state
    return backend, _wrapper(backend), module


def _transaction_metadata(
    *,
    kv_indices: torch.Tensor,
    qo_indptr: torch.Tensor | None = None,
) -> MLAPlanMetadata:
    return MLAPlanMetadata.csr(
        qo_indptr=(
            torch.tensor([0, 2, 3], dtype=torch.int32)
            if qo_indptr is None
            else qo_indptr
        ),
        kv_indptr=torch.tensor([0, 2, 4], dtype=torch.int32),
        kv_indices=kv_indices,
        kv_len_arr=torch.tensor([3, 4], dtype=torch.int32),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_transaction_publishes_device_source_and_preserves_tail() -> None:
    backend, wrapper, module = _transaction_fixture()
    state = backend._cuda_graph_plan_update_state
    candidate_identity = id(state.candidate)
    tail = backend._kv_indices_buf[4:].clone()
    source = torch.tensor([7, 8, 9, 10], dtype=torch.int32, device="cuda")

    wrapper.update_cuda_graph_plan(metadata=_transaction_metadata(kv_indices=source))
    torch.cuda.synchronize(backend.device)

    assert module.order == ["plan", "record", "commit"]
    assert torch.equal(backend._kv_indices_buf[:4], source)
    assert torch.equal(backend._kv_indices_buf[4:], tail)
    assert id(state.candidate) == candidate_identity
    assert not hasattr(state, "committed")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "failure", ["invalid-csr", "planner", "commit", "plan-info", "staged-bytes"]
)
def test_prepublication_failures_leave_live_plan_untouched(failure: str) -> None:
    backend, wrapper, module = _transaction_fixture()
    live_before = {
        "workspace": backend._int_workspace_buffer.clone(),
        "qo": backend._qo_indptr_buf.clone(),
        "kv": backend._kv_indptr_buf.clone(),
        "indices": backend._kv_indices_buf.clone(),
        "lengths": backend._kv_len_arr_buf.clone(),
    }
    if failure == "planner":
        module.planner_error = RuntimeError("synthetic planner failure")
    elif failure == "commit":
        module.commit_error = RuntimeError("synthetic commit submission failure")
    elif failure == "plan-info":
        module.plan_info = (11, 22, 33, 45)
    elif failure == "staged-bytes":
        module.staged_int_workspace_bytes = 23
    qo_indptr = (
        torch.tensor([1, 2, 3], dtype=torch.int32) if failure == "invalid-csr" else None
    )
    source = torch.tensor([7, 8, 9, 10], dtype=torch.int32, device="cuda")

    message = {
        "invalid-csr": "qo_indptr must start at zero",
        "planner": "synthetic planner failure",
        "commit": "synthetic commit submission failure",
        "plan-info": "candidate plan_info changed",
        "staged-bytes": "candidate staged_int_workspace_bytes changed",
    }[failure]
    with pytest.raises((ValueError, RuntimeError), match=message):
        wrapper.update_cuda_graph_plan(
            metadata=_transaction_metadata(
                kv_indices=source,
                qo_indptr=qo_indptr,
            )
        )
    torch.cuda.synchronize(backend.device)

    assert torch.equal(backend._int_workspace_buffer, live_before["workspace"])
    assert torch.equal(backend._qo_indptr_buf, live_before["qo"])
    assert torch.equal(backend._kv_indptr_buf, live_before["kv"])
    assert torch.equal(backend._kv_indices_buf, live_before["indices"])
    assert torch.equal(backend._kv_len_arr_buf, live_before["lengths"])
    assert "commit" not in module.order or failure == "commit"
    if failure in ("plan-info", "staged-bytes"):
        assert module.order == ["plan"]
        assert backend._cuda_graph_plan_update_state.slots[0].status == "idle"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_real_events_keep_both_slots_busy_until_copies_complete() -> None:
    from cuda.bindings import runtime

    backend, wrapper, module = _transaction_fixture()
    state = backend._cuda_graph_plan_update_state
    # Replace the fixture's always-ready event with a real, warmed CUDA event.
    state.slots[0].completion_event = torch.cuda.Event()
    state.slots[0].completion_event.record()
    state.slots[0].completion_event.synchronize()
    source = torch.tensor([7, 8, 9, 10], dtype=torch.int32, device=backend.device)
    metadata = _transaction_metadata(kv_indices=source)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream(backend.device))
    stream.synchronize()
    release = Event()
    expired = Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def hold_stream(_):
        # No CUDA calls in a CUDA host callback. The timeout also bounds cleanup
        # if an unexpected driver/API path blocks the submitting Python thread.
        if not release.wait(timeout=10):
            expired.set()

    try:
        (error,) = runtime.cudaLaunchHostFunc(
            stream.cuda_stream, ctypes.cast(hold_stream, ctypes.c_void_p).value, 0
        )
        assert error == runtime.cudaError_t.cudaSuccess
        with torch.cuda.stream(stream):
            wrapper.update_cuda_graph_plan(metadata=metadata)
            assert state.slots[0].status == "pending"
            assert not state.slots[0].completion_event.query()
            wrapper.update_cuda_graph_plan(metadata=metadata)
            assert all(slot.status == "pending" for slot in state.slots)
            assert all(not slot.completion_event.query() for slot in state.slots)
            with pytest.raises(RuntimeError, match="staging slots are busy"):
                wrapper.update_cuda_graph_plan(metadata=metadata)
            assert module.order == ["plan", "commit", "plan", "commit"]
    finally:
        release.set()
        stream.synchronize()
    assert not expired.is_set(), "stream gate timed out before the assertions finished"
    assert all(slot.completion_event.query() for slot in state.slots)
    with torch.cuda.stream(stream):
        wrapper.update_cuda_graph_plan(metadata=metadata)
    stream.synchronize()
    assert module.order == ["plan", "commit"] * 3
    assert torch.equal(backend._kv_indices_buf[:4], source)


class _RecordFailureEvent(_ReadyEvent):
    def record(self) -> None:
        raise RuntimeError("synthetic event record failure")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_event_record_failure_poisons_slots_without_publishing() -> None:
    backend, wrapper, module = _transaction_fixture()
    state = backend._cuda_graph_plan_update_state
    live = (
        backend._int_workspace_buffer,
        backend._qo_indptr_buf,
        backend._kv_indptr_buf,
        backend._kv_indices_buf,
        backend._kv_len_arr_buf,
    )
    snapshots = tuple(tensor.clone() for tensor in live)
    source = torch.tensor([7, 8, 9, 10], dtype=torch.int32, device=backend.device)
    metadata = _transaction_metadata(kv_indices=source)
    for slot in state.slots:
        slot.completion_event = _RecordFailureEvent()
    for index in range(2):
        with pytest.raises(RuntimeError, match="synthetic event record failure"):
            wrapper.update_cuda_graph_plan(metadata=metadata)
        assert state.slots[index].status == "poisoned"
    with pytest.raises(RuntimeError, match="slots are poisoned; call plan"):
        wrapper.update_cuda_graph_plan(metadata=metadata)
    torch.cuda.synchronize(backend.device)
    assert module.order == ["plan", "plan"]
    for tensor, snapshot in zip(live, snapshots, strict=True):
        assert torch.equal(tensor, snapshot)


def test_transaction_capability_is_enabled_only_by_generated_fa_backends() -> None:
    from flashinfer.mla._batch_mla._backends.cutlass_backend import (
        _BatchMLAPagedAttentionCutlassBackend,
    )
    from flashinfer.mla._batch_mla._backends.cutile_backend import (
        _BatchMLAPagedAttentionCutileBackend,
    )
    from flashinfer.mla._batch_mla._backends.fa2_backend import (
        _BatchMLAPagedAttentionFa2Backend,
    )
    from flashinfer.mla._batch_mla._backends.fa3_backend import (
        _BatchMLAPagedAttentionFa3Backend,
    )

    assert (
        _BatchMLAPagedAttentionFa2Backend._plan_capabilities.supports_cuda_graph_plan_update
        is True
    )
    assert (
        _BatchMLAPagedAttentionFa3Backend._plan_capabilities.supports_cuda_graph_plan_update
        is True
    )
    assert (
        _BatchMLAPagedAttentionCutlassBackend._plan_capabilities.supports_cuda_graph_plan_update
        is False
    )
    assert (
        _BatchMLAPagedAttentionCutileBackend._plan_capabilities.supports_cuda_graph_plan_update
        is False
    )
    assert (
        _DefaultPlannedBackend._plan_capabilities.supports_cuda_graph_plan_update
        is False
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("backend_name", ["fa2", "fa3"])
def test_real_generated_capture_update_replay_uses_lean_state(
    backend_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    if backend_name == "fa3" and not is_sm90a_supported(device):
        pytest.skip("FA3 requires SM90a")

    torch.manual_seed(7)
    batch_size = 2
    num_heads = 16
    head_dim_ckv = 512
    head_dim_kpe = 64
    qo_indptr_buf = torch.empty((batch_size + 1,), dtype=torch.int32, device=device)
    kv_indptr_buf = torch.empty_like(qo_indptr_buf)
    kv_indices_buf = torch.full((16,), 63, dtype=torch.int32, device=device)
    kv_len_arr_buf = torch.empty((batch_size,), dtype=torch.int32, device=device)
    wrapper = BatchMLAPagedAttentionWrapper(
        torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
        backend=backend_name,
        use_cuda_graph=True,
        enable_cuda_graph_plan_update=True,
        qo_indptr=qo_indptr_buf,
        kv_indptr=kv_indptr_buf,
        kv_indices=kv_indices_buf,
        kv_len_arr=kv_len_arr_buf,
    )
    q_nope = torch.randn(
        (batch_size, num_heads, head_dim_ckv),
        dtype=torch.float16,
        device=device,
    )
    q_pe = torch.randn(
        (batch_size, num_heads, head_dim_kpe),
        dtype=torch.float16,
        device=device,
    )
    ckv = torch.randn((8, 1, head_dim_ckv), dtype=torch.float16, device=device)
    kpe = torch.randn((8, 1, head_dim_kpe), dtype=torch.float16, device=device)
    initial_qo = torch.tensor([0, 1, 2], dtype=torch.int32)
    initial_kv = torch.tensor([0, 2, 4], dtype=torch.int32)
    initial_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    initial_lengths = torch.tensor([2, 2], dtype=torch.int32)
    plan_kwargs = dict(
        num_heads=num_heads,
        head_dim_ckv=head_dim_ckv,
        head_dim_kpe=head_dim_kpe,
        page_size=1,
        causal=False,
        sm_scale=1.0 / (192.0**0.5),
        q_data_type=torch.float16,
        kv_data_type=torch.float16,
        query_layout="split",
        kv_cache_layout="split",
        lse_mode="none",
    )
    stream = torch.cuda.Stream()

    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream):
        wrapper.plan(
            metadata=MLAPlanMetadata.csr(
                initial_qo,
                initial_kv,
                initial_indices,
                initial_lengths,
            ),
            **plan_kwargs,
        )
        out = torch.empty_like(q_nope)
        expected = torch.empty_like(q_nope)
        for _ in range(3):
            wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe), out=out)
    stream.synchronize()

    state = wrapper._planned_backend._cuda_graph_plan_update_state
    assert not hasattr(state, "committed")
    assert not hasattr(state.candidate, "kv_indices")
    assert all(not hasattr(slot, "kv_indices") for slot in state.slots)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe), out=out)
    original_tail = kv_indices_buf[4:].clone()

    updated_qo = torch.tensor([0, 0, 2], dtype=torch.int32)
    updated_kv = torch.tensor([0, 1, 4], dtype=torch.int32)
    updated_lengths = torch.tensor([1, 3], dtype=torch.int32)
    source = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=device)
    with torch.cuda.stream(stream):
        wrapper.update_cuda_graph_plan(
            metadata=MLAPlanMetadata.csr(
                updated_qo,
                updated_kv,
                source,
                updated_lengths,
            )
        )
        graph.replay()
        wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe), out=expected)
    stream.synchronize()
    torch.testing.assert_close(out, expected, rtol=1e-3, atol=1e-3)
    assert torch.equal(kv_indices_buf[:4], source)
    assert torch.equal(kv_indices_buf[4:], original_tail)

    committed_output = out.clone()
    live_before_failure = kv_indices_buf.clone()
    with torch.cuda.stream(stream):
        with pytest.raises(ValueError, match="qo_indptr must start at zero"):
            wrapper.update_cuda_graph_plan(
                metadata=MLAPlanMetadata.csr(
                    torch.tensor([1, 1, 3], dtype=torch.int32),
                    updated_kv,
                    torch.tensor([3, 2, 1, 0], dtype=torch.int32, device=device),
                    updated_lengths,
                )
            )
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out, committed_output, rtol=1e-3, atol=1e-3)
    assert torch.equal(kv_indices_buf, live_before_failure)
    assert wrapper._cuda_graph_plan_update_stream == stream.cuda_stream

    # A full replan writes the reserved CSR before initializing update state.
    # Failure at that boundary must preserve both bytes and the runnable graph.
    backend_before = wrapper._planned_backend
    live_tensors = (
        backend_before._int_workspace_buffer[
            : backend_before._staged_int_workspace_bytes
        ],
        qo_indptr_buf,
        kv_indptr_buf,
        kv_indices_buf,
        kv_len_arr_buf,
    )
    live_snapshots = tuple(tensor.clone() for tensor in live_tensors)
    fa_common = importlib.import_module(
        "flashinfer.mla._batch_mla._backends._fa_common"
    )

    replacement_kv = torch.tensor([0, 1, 2], dtype=torch.int32)
    replacement_indices = torch.tensor([0, 1], dtype=torch.int32)
    replacement_lengths = torch.tensor([1, 1], dtype=torch.int32)

    from cuda.bindings import runtime

    rollback_release = Event()
    rollback_expired = Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def hold_rollback(_):
        if not rollback_release.wait(timeout=10):
            rollback_expired.set()

    def fail_update_state_init(**kwargs):
        # Prove this is the late failure path, after backend.plan staged CSR.
        assert torch.equal(qo_indptr_buf.cpu(), initial_qo)
        assert torch.equal(kv_indptr_buf.cpu(), replacement_kv)
        assert torch.equal(kv_indices_buf[:2].cpu(), replacement_indices)
        if failed_plan_stream != stream:
            # Hold restoration after live CSR changed, without making the
            # original stream busy when plan() checks it at entry.
            (error,) = runtime.cudaLaunchHostFunc(
                failed_plan_stream.cuda_stream,
                ctypes.cast(hold_rollback, ctypes.c_void_p).value,
                0,
            )
            assert error == runtime.cudaError_t.cudaSuccess
        raise RuntimeError("synthetic update state initialization failure")

    for failed_plan_stream in (stream, torch.cuda.Stream()):
        failed_plan_stream.wait_stream(torch.cuda.current_stream(device))
        replay_done = torch.cuda.Event()
        try:
            with monkeypatch.context() as patch, torch.cuda.stream(failed_plan_stream):
                patch.setattr(
                    fa_common,
                    "_make_mla_cuda_graph_plan_update_state",
                    fail_update_state_init,
                )
                with pytest.raises(
                    RuntimeError, match="update state initialization failure"
                ):
                    wrapper.plan(
                        metadata=MLAPlanMetadata.csr(
                            initial_qo,
                            replacement_kv,
                            replacement_indices,
                            replacement_lengths,
                        ),
                        **plan_kwargs,
                    )
            with torch.cuda.stream(stream):
                graph.replay()
                replay_done.record()
            if failed_plan_stream != stream:
                # Observe a real replay completion event, not a host callback
                # that CUDA may serialize with the gate on the other stream.
                deadline = time.monotonic() + 1.0
                while not replay_done.query() and time.monotonic() < deadline:
                    rollback_release.wait(timeout=0.001)
                assert not replay_done.query(), (
                    "old-stream replay completed before cross-stream rollback"
                )
        finally:
            rollback_release.set()
            failed_plan_stream.synchronize()
            stream.synchronize()
        assert not rollback_expired.is_set(), "rollback gate timed out"
        rollback_release.clear()
        assert wrapper._planned_backend is backend_before
        assert wrapper._cuda_graph_plan_update_stream == stream.cuda_stream
        for tensor, snapshot in zip(live_tensors, live_snapshots, strict=True):
            assert torch.equal(tensor, snapshot)
        torch.testing.assert_close(out, committed_output, rtol=1e-3, atol=1e-3)

    for slot in backend_before._cuda_graph_plan_update_state.slots:
        slot.completion_event = _RecordFailureEvent()
    with torch.cuda.stream(stream):
        for _ in range(2):
            with pytest.raises(RuntimeError, match="synthetic event record failure"):
                wrapper.update_cuda_graph_plan(
                    metadata=MLAPlanMetadata.csr(
                        updated_qo, updated_kv, source, updated_lengths
                    )
                )
        with pytest.raises(RuntimeError, match="slots are poisoned; call plan"):
            wrapper.update_cuda_graph_plan(
                metadata=MLAPlanMetadata.csr(
                    updated_qo, updated_kv, source, updated_lengths
                )
            )
        graph.replay()
    stream.synchronize()
    torch.testing.assert_close(out, committed_output, rtol=1e-3, atol=1e-3)

    # A successful full plan starts a fresh stream/capture lifecycle.
    replacement_stream = torch.cuda.Stream()
    replacement_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(replacement_stream):
        wrapper.plan(
            metadata=MLAPlanMetadata.csr(
                initial_qo, initial_kv, initial_indices, initial_lengths
            ),
            **plan_kwargs,
        )
    replacement_stream.synchronize()
    replacement_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(replacement_graph, stream=replacement_stream):
        wrapper.run(query=(q_nope, q_pe), kv_cache=(ckv, kpe), out=out)
    with torch.cuda.stream(replacement_stream):
        wrapper.update_cuda_graph_plan(
            metadata=MLAPlanMetadata.csr(
                updated_qo, updated_kv, source, updated_lengths
            )
        )
        replacement_graph.replay()
    replacement_stream.synchronize()
    assert wrapper._cuda_graph_plan_update_stream == replacement_stream.cuda_stream
    torch.testing.assert_close(out, committed_output, rtol=1e-3, atol=1e-3)

    # A completed staging event does not cover later publication or replay.
    # Block a real replay after all slot events completed and require a replan
    # on another stream to reject before even snapshotting the shared buffers.
    current_backend = wrapper._planned_backend
    current_state = current_backend._cuda_graph_plan_update_state
    assert all(slot.completion_event.query() for slot in current_state.slots)
    release = Event()
    expired = Event()

    @ctypes.CFUNCTYPE(None, ctypes.c_void_p)
    def hold_replay(_):
        if not release.wait(timeout=10):
            expired.set()

    def unexpected_snapshot(*args, **kwargs):
        raise RuntimeError("replan reached buffer snapshotting")

    try:
        (error,) = runtime.cudaLaunchHostFunc(
            replacement_stream.cuda_stream,
            ctypes.cast(hold_replay, ctypes.c_void_p).value,
            0,
        )
        assert error == runtime.cudaError_t.cudaSuccess
        with torch.cuda.stream(replacement_stream):
            replacement_graph.replay()
        assert all(slot.completion_event.query() for slot in current_state.slots)
        assert not replacement_stream.query()
        with monkeypatch.context() as patch, torch.cuda.stream(stream):
            patch.setattr(torch.Tensor, "clone", unexpected_snapshot)
            with pytest.raises(RuntimeError, match="bound CUDA stream.*pending work"):
                wrapper.plan(
                    metadata=MLAPlanMetadata.csr(
                        initial_qo, initial_kv, initial_indices, initial_lengths
                    ),
                    **plan_kwargs,
                )
        assert wrapper._planned_backend is current_backend
        assert wrapper._cuda_graph_plan_update_stream == replacement_stream.cuda_stream
        # Same-stream work is already ordered. Stop at the first snapshot so
        # this check itself cannot wait behind the gated replay or mutate it.
        with monkeypatch.context() as patch, torch.cuda.stream(replacement_stream):
            patch.setattr(torch.Tensor, "clone", unexpected_snapshot)
            with pytest.raises(
                RuntimeError, match="replan reached buffer snapshotting"
            ):
                wrapper.plan(
                    metadata=MLAPlanMetadata.csr(
                        initial_qo, initial_kv, initial_indices, initial_lengths
                    ),
                    **plan_kwargs,
                )
    finally:
        release.set()
        replacement_stream.synchronize()
    assert not expired.is_set(), "replay gate timed out before the assertions finished"
    torch.testing.assert_close(out, committed_output, rtol=1e-3, atol=1e-3)

    # Once the old stream is complete, switching streams can publish a new plan.
    with torch.cuda.stream(stream):
        wrapper.plan(
            metadata=MLAPlanMetadata.csr(
                initial_qo, initial_kv, initial_indices, initial_lengths
            ),
            **plan_kwargs,
        )
    stream.synchronize()
    assert wrapper._planned_backend is not current_backend
    assert wrapper._cuda_graph_plan_update_stream is None
