"""
Support-check behavior tests for the communication AllReduce fusion API.

The checks live on the unified AllReduce fusion family in
``flashinfer/comm/allreduce.py``:

- ``create_allreduce_fusion_workspace`` selects a backend (``trtllm``/``mnnvl``,
  or ``auto`` over those two), so it answers queries per backend.
- ``allreduce_fusion`` takes no ``backend`` argument at all -- the workspace
  carries the decision -- so it has no backend choices to answer for.

Host-only tests (no GPU, no JIT, no process group):

    pytest -s tests/comm/test_comm_support_checks.py

The collective test needs two ranks on an architecture the kernels cover:

    mpirun -np 2 pytest -s tests/comm/test_comm_support_checks.py
"""

import inspect
import multiprocessing as mp
import queue
import time

import pytest
import torch
import torch.distributed as dist

import flashinfer.comm as comm
import flashinfer.comm.allreduce as allreduce
import flashinfer.comm.trtllm_ar as trtllm_ar
from flashinfer.comm import (
    AllReduceFusionPattern,
    allreduce_fusion,
    create_allreduce_fusion_workspace,
)
from flashinfer.comm.allreduce import TRTLLMAllReduceFusionWorkspace
from flashinfer.utils import BackendSupportedError, is_sm90a_supported

# The architecture of the L20 (Ada) parts this work targets. Both backends of
# the family are built for newer families, so 89 is the interesting rejection.
L20_COMPUTE_CAPABILITY = 89
HOPPER_COMPUTE_CAPABILITY = 90


def _refuse(*args, **kwargs):
    raise AssertionError("a support query reached a device, loader or allocator")


def _record(calls, result=None):
    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    return record


def _trtllm_workspace_stub():
    """A TRT-LLM workspace with its allocating constructor bypassed."""
    workspace = TRTLLMAllReduceFusionWorkspace.__new__(TRTLLMAllReduceFusionWorkspace)
    workspace.world_size = 2
    workspace.rank = 0
    workspace._destroyed = False
    workspace.ipc_handles = [[1, 2]]
    workspace.workspace_tensor = torch.empty(0, dtype=torch.int64)
    workspace.mem_handles = []
    workspace.metadata = {}
    return workspace


# ---------------------------------------------------------------------------
# Static queries
# ---------------------------------------------------------------------------


def test_query_surface_matches_the_support_check_convention():
    """B1-01: same query pair and signatures as ``backend_requirement``."""
    for api in (create_allreduce_fusion_workspace, allreduce_fusion):
        assert callable(api.is_backend_supported)
        assert callable(api.is_compute_capability_supported)
        assert callable(api.has_backend)
        assert callable(api.has_backend_choices)

        backend_signature = inspect.signature(api.is_backend_supported)
        assert list(backend_signature.parameters) == ["backend", "cc"]
        assert backend_signature.parameters["cc"].default is None
        cc_signature = inspect.signature(api.is_compute_capability_supported)
        assert list(cc_signature.parameters) == ["cc"]

    assert create_allreduce_fusion_workspace.has_backend_choices() is True
    assert create_allreduce_fusion_workspace.has_backend("trtllm") is True
    assert create_allreduce_fusion_workspace.has_backend("mnnvl") is True
    assert create_allreduce_fusion_workspace.has_backend("nccl") is False


def test_capability_answers_follow_the_kernel_architecture_families():
    """B1-02: answers are the architecture families the kernels are built for.

    ``flashinfer/jit/comm.py`` builds the TRT-LLM comm module for major versions
    9/10/12 and the MNNVL comm module for 9/10, and 89 (L20) is outside both.
    """
    is_backend_supported = create_allreduce_fusion_workspace.is_backend_supported
    is_cc_supported = create_allreduce_fusion_workspace.is_compute_capability_supported

    for cc in (90, 100, 103):
        assert is_backend_supported("trtllm", cc) is True
        assert is_backend_supported("mnnvl", cc) is True
        assert is_cc_supported(cc) is True
    for cc in (120, 121):
        assert is_backend_supported("trtllm", cc) is True
        assert is_backend_supported("mnnvl", cc) is False
    for cc in (70, 75, 80, 86, 89):
        assert is_backend_supported("trtllm", cc) is False
        assert is_backend_supported("mnnvl", cc) is False
        assert is_cc_supported(cc) is False

    # Without a capability the query answers the name question only.
    assert is_backend_supported("trtllm") is True
    assert is_backend_supported("mnnvl") is True
    assert is_backend_supported("nccl") is False
    assert is_backend_supported("nccl", 90) is False

    # The run API answers the same architecture question from the same registry.
    assert allreduce_fusion.is_compute_capability_supported(90) is True
    assert allreduce_fusion.is_compute_capability_supported(89) is False
    assert allreduce_fusion.has_backend("trtllm") is True
    assert allreduce_fusion.has_backend("nccl") is False


def test_run_api_without_backend_choices_keeps_the_convention():
    """B1-04: no backend choices means no backend argument to answer for."""
    assert allreduce_fusion.has_backend_choices() is False
    with pytest.raises(ValueError, match="no backend choices for allreduce_fusion"):
        allreduce_fusion.is_backend_supported("trtllm")
    for backend, cc in (
        ("trtllm", 90),
        ("mnnvl", 90),
        (None, 90),
        ("nccl", 90),
        ("trtllm", None),
    ):
        with pytest.raises(ValueError, match="no backend choices for allreduce_fusion"):
            allreduce_fusion.is_backend_supported(backend, cc)


def test_static_queries_touch_no_device_loader_allocator_or_group(monkeypatch):
    """B1-05: every query is answerable without a device or a process group."""
    monkeypatch.setattr(trtllm_ar, "get_trtllm_comm_module", _refuse)
    monkeypatch.setattr(allreduce, "TRTLLMAllReduceFusionWorkspace", _refuse)
    monkeypatch.setattr(allreduce, "MNNVLAllReduceFusionWorkspace", _refuse)
    monkeypatch.setattr(allreduce, "is_multicast_supported", _refuse)
    monkeypatch.setattr(allreduce, "all_ranks_support_mnnvl", _refuse)
    monkeypatch.setattr(allreduce, "_device_capability", _refuse)
    monkeypatch.setattr(torch.cuda, "current_device", _refuse)
    monkeypatch.setattr(torch.cuda, "device_count", _refuse)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _refuse)
    monkeypatch.setattr(dist, "init_process_group", _refuse)

    assert create_allreduce_fusion_workspace.has_backend_choices() is True
    assert create_allreduce_fusion_workspace.has_backend("trtllm") is True
    assert create_allreduce_fusion_workspace.has_backend("nccl") is False
    assert create_allreduce_fusion_workspace.is_backend_supported("trtllm", 90) is True
    assert create_allreduce_fusion_workspace.is_backend_supported("trtllm", 89) is False
    assert create_allreduce_fusion_workspace.is_compute_capability_supported(90)
    assert not create_allreduce_fusion_workspace.is_compute_capability_supported(89)
    assert not allreduce_fusion.is_compute_capability_supported(89)
    assert allreduce_fusion.has_backend_choices() is False


# ---------------------------------------------------------------------------
# Creation-time rejection
# ---------------------------------------------------------------------------


def test_unknown_backend_is_refused_without_an_auto_fallback(monkeypatch):
    """B1-03: an unknown explicit backend is refused, not silently auto-selected."""
    heuristic_calls = []
    monkeypatch.setattr(
        allreduce, "_workspace_creation_heuristic", _record(heuristic_calls)
    )
    workspace_calls = []
    monkeypatch.setattr(
        allreduce, "TRTLLMAllReduceFusionWorkspace", _record(workspace_calls)
    )
    monkeypatch.setattr(
        allreduce, "MNNVLAllReduceFusionWorkspace", _record(workspace_calls)
    )

    with pytest.raises(BackendSupportedError, match="Unknown backend 'nccl'"):
        create_allreduce_fusion_workspace(
            backend="nccl",
            world_size=2,
            rank=0,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
        )

    assert heuristic_calls == []
    assert workspace_calls == []


def test_unsupported_capability_is_refused_before_loader_and_allocator(monkeypatch):
    """B1-06: an unsupported configuration never reaches the workspace path.

    The capability is forced to the L20 value, so this is about the gate rather
    than about the host it runs on. Every step that would load a module,
    allocate workspace memory or rendezvous is a spy, and the spies stay empty.
    """
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9)
    )
    reached = []
    monkeypatch.setattr(trtllm_ar, "get_trtllm_comm_module", _record(reached))
    monkeypatch.setattr(
        trtllm_ar,
        "trtllm_create_ipc_workspace_for_all_reduce_fusion",
        _record(reached),
    )
    monkeypatch.setattr(allreduce, "TRTLLMAllReduceFusionWorkspace", _record(reached))
    monkeypatch.setattr(allreduce, "MNNVLAllReduceFusionWorkspace", _record(reached))
    monkeypatch.setattr(allreduce, "is_multicast_supported", _record(reached))
    monkeypatch.setattr(allreduce, "all_ranks_support_mnnvl", _record(reached))

    for backend in ("trtllm", "mnnvl"):
        with pytest.raises(
            BackendSupportedError, match=f"capability {L20_COMPUTE_CAPABILITY}"
        ):
            create_allreduce_fusion_workspace(
                backend=backend,
                world_size=2,
                rank=0,
                max_token_num=128,
                hidden_dim=2880,
                dtype=torch.float16,
            )

    assert reached == []


def test_supported_capability_reaches_the_workspace_constructor(monkeypatch):
    """B1-11 (dispatch half): a capability the kernels cover is not refused.

    The constructor is a recorder here, so this asserts the gate and the exact
    dispatch, not GPU support.
    """
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0)
    )
    created = []
    sentinel = object()
    monkeypatch.setattr(
        allreduce, "TRTLLMAllReduceFusionWorkspace", _record(created, sentinel)
    )

    workspace = create_allreduce_fusion_workspace(
        backend="trtllm",
        world_size=2,
        rank=1,
        max_token_num=128,
        hidden_dim=2880,
        dtype=torch.bfloat16,
    )

    assert workspace is sentinel
    assert len(created) == 1
    _, kwargs = created[0]
    assert kwargs == {
        "tp_size": 2,
        "tp_rank": 1,
        "max_token_num": 128,
        "hidden_dim": 2880,
        "dtype": torch.bfloat16,
        "comm_backend": None,
        "group": None,
    }


def test_auto_selection_skips_architecturally_unsupported_backends(monkeypatch):
    """B1-06 (auto half): the default policy is capability-aware, and still votes.

    MNNVL reports itself available here, so the capability filter is what rules
    it out. The topology vote is a collective and has to be cast by every rank
    regardless of what that rank's own capability says.
    """
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9)
    )
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(allreduce, "is_confidential_compute", lambda: False)
    votes = []
    monkeypatch.setattr(allreduce, "is_multicast_supported", lambda device: True)
    monkeypatch.setattr(allreduce, "all_ranks_support_mnnvl", _record(votes, True))
    trtllm_created = []
    mnnvl_created = []
    monkeypatch.setattr(
        allreduce, "TRTLLMAllReduceFusionWorkspace", _record(trtllm_created)
    )
    monkeypatch.setattr(
        allreduce, "MNNVLAllReduceFusionWorkspace", _record(mnnvl_created)
    )

    with pytest.raises(
        ValueError, match="No suitable backend found for compute capability 89"
    ):
        create_allreduce_fusion_workspace(
            backend="auto",
            world_size=2,
            rank=0,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
        )

    assert len(votes) == 1
    assert trtllm_created == []
    assert mnnvl_created == []

    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (9, 0)
    )
    sentinel = object()
    monkeypatch.setattr(
        allreduce, "MNNVLAllReduceFusionWorkspace", _record(mnnvl_created, sentinel)
    )
    workspace = create_allreduce_fusion_workspace(
        backend="auto",
        world_size=2,
        rank=0,
        max_token_num=128,
        hidden_dim=2880,
        dtype=torch.float16,
    )

    assert workspace is sentinel
    assert trtllm_created == []
    assert len(mnnvl_created) == 1


# ---------------------------------------------------------------------------
# Run and destroy
# ---------------------------------------------------------------------------


def test_run_api_does_not_probe_topology_per_call(monkeypatch):
    """B1-12: the collective reuses the decision the workspace was built with.

    Nothing on this path may query a capability, probe multicast, vote or enter
    a process-group collective: that belongs to creation, and repeating it per
    forward would both cost and risk diverging between ranks.
    """
    workspace = _trtllm_workspace_stub()
    launches = []
    monkeypatch.setattr(allreduce, "trtllm_allreduce_fusion", _record(launches))
    monkeypatch.setattr(allreduce, "is_multicast_supported", _refuse)
    monkeypatch.setattr(allreduce, "all_ranks_support_mnnvl", _refuse)
    monkeypatch.setattr(allreduce, "_device_capability", _refuse)
    monkeypatch.setattr(torch.cuda, "get_device_capability", _refuse)
    monkeypatch.setattr(dist, "barrier", _refuse)
    monkeypatch.setattr(dist, "all_gather", _refuse)
    monkeypatch.setattr(dist, "all_gather_object", _refuse)

    x = torch.zeros(4, 8, dtype=torch.float16)
    for _ in range(3):
        output = allreduce_fusion(
            input=x,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
        )
        assert output.shape == x.shape

    assert len(launches) == 3
    assert all(call[1]["world_size"] == 2 for call in launches)


def test_destroy_is_not_gated_by_the_creation_policy(monkeypatch):
    """B1-08: a workspace created while the policy allowed it stays disposable."""
    monkeypatch.setattr(
        torch.cuda, "get_device_capability", lambda *args, **kwargs: (8, 9)
    )
    with pytest.raises(BackendSupportedError):
        create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=2,
            rank=0,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
        )

    workspace = _trtllm_workspace_stub()
    workspace.destroy()
    assert workspace._destroyed is True

    # A repeated destroy stays a no-op rather than a double free.
    workspace.destroy()
    assert workspace._destroyed is True


# ---------------------------------------------------------------------------
# Multi-rank failure consistency
# ---------------------------------------------------------------------------

_PROCESS_DEADLINE_S = 120.0


def _attempt_creation(rank, backend, outcome):
    """Child process: one creation attempt, reported as ``(rank, type, message)``."""
    torch.cuda.get_device_capability = lambda *args, **kwargs: (8, 9)

    def refuse_collective(*args, **kwargs):
        raise AssertionError("a rejected creation attempt entered a collective")

    dist.init_process_group = refuse_collective
    dist.barrier = refuse_collective
    dist.all_gather = refuse_collective
    dist.all_gather_object = refuse_collective

    try:
        create_allreduce_fusion_workspace(
            backend=backend,
            world_size=2,
            rank=rank,
            max_token_num=128,
            hidden_dim=2880,
            dtype=torch.float16,
        )
    except Exception as error:  # recorded, not handled: this is the outcome
        outcome.put((rank, type(error).__name__, str(error)))
    else:
        outcome.put((rank, "NO_ERROR", ""))


def _run_creation_attempts(backends):
    """Run one creation attempt per rank in its own process, bounded in time."""
    context = mp.get_context("spawn")
    outcome = context.Queue()
    processes = [
        context.Process(target=_attempt_creation, args=(rank, backend, outcome))
        for rank, backend in enumerate(backends)
    ]
    for process in processes:
        process.start()

    results = {}
    deadline = time.monotonic() + _PROCESS_DEADLINE_S
    while len(results) < len(processes) and time.monotonic() < deadline:
        try:
            rank, error_type, message = outcome.get(timeout=1.0)
        except queue.Empty:
            if all(process.exitcode is not None for process in processes):
                break
            continue
        results[rank] = (error_type, message)
    for process in processes:
        process.join(timeout=5.0)
    hung = [rank for rank, process in enumerate(processes) if process.is_alive()]
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5.0)
    assert hung == [], f"ranks {hung} did not exit within {_PROCESS_DEADLINE_S}s"
    return results


def test_all_ranks_unsupported_exit_together():
    """B1-09: one unsupported condition, one reason, no rank left waiting."""
    results = _run_creation_attempts(["trtllm", "trtllm"])
    assert sorted(results) == [0, 1]
    assert results[0] == results[1]
    assert results[0][0] == "BackendSupportedError"
    assert f"capability {L20_COMPUTE_CAPABILITY}" in results[0][1]


def test_each_rank_reports_its_own_reason_when_arguments_differ():
    """B1-10: separate processes, bounded exit, per-rank reason visible."""
    results = _run_creation_attempts(["trtllm", "nccl"])
    assert sorted(results) == [0, 1]
    assert results[0][0] == "BackendSupportedError"
    assert f"capability {L20_COMPUTE_CAPABILITY}" in results[0][1]
    assert results[1][0] == "BackendSupportedError"
    assert "Unknown backend 'nccl'" in results[1][1]


# ---------------------------------------------------------------------------
# Group agreement before any workspace constructor
# ---------------------------------------------------------------------------


class _FakeComm:
    """Collective that reports the verdicts of the whole (fake) group."""

    def __init__(self, size, all_verdicts):
        self._size = size
        self._all = all_verdicts
        self.gathers = 0

    def Get_size(self):
        return self._size

    def allgather(self, local):
        self.gathers += 1
        return tuple(self._all)


_SUPPORTED = (True, None, None, "trtllm")
_REFUSED = (
    False,
    "BackendSupportedError",
    f"Backend 'trtllm' does not support compute capability {L20_COMPUTE_CAPABILITY}",
    None,
)


def test_unanimous_support_returns_the_agreed_verdict():
    """B1-13: an agreed verdict is returned unchanged, from one collective."""
    comm = _FakeComm(2, [_SUPPORTED, _SUPPORTED])
    assert allreduce._agree_on_support(_SUPPORTED, 2, comm, None) == _SUPPORTED
    assert comm.gathers == 1


def test_a_rank_that_supports_the_backend_still_refuses_a_mixed_group():
    """B1-13: the supported rank must not reach a constructor alone.

    Rank 0 can run trtllm, rank 1 cannot. Without an agreement rank 0 enters the
    workspace constructor and waits in a rendezvous rank 1 will never join.
    """
    for local in (_SUPPORTED, _REFUSED):
        with pytest.raises(BackendSupportedError) as excinfo:
            allreduce._agree_on_support(
                local, 2, _FakeComm(2, [_SUPPORTED, _REFUSED]), None
            )
        message = str(excinfo.value)
        assert "differs across ranks" in message
        assert f"compute capability {L20_COMPUTE_CAPABILITY}" in message


def test_divergent_backend_selections_are_refused_as_a_group():
    """B1-13: two ranks supporting different backends cannot share a workspace."""
    other = (True, None, None, "mnnvl")
    with pytest.raises(BackendSupportedError, match="differs across ranks"):
        allreduce._agree_on_support(
            _SUPPORTED, 2, _FakeComm(2, [_SUPPORTED, other]), None
        )


def test_a_unanimous_refusal_keeps_one_reason_for_every_rank():
    """B1-13: the verdict survives the agreement, so all ranks report one reason."""
    assert allreduce._agree_on_support(
        _REFUSED, 2, _FakeComm(2, [_REFUSED, _REFUSED]), None
    ) == (_REFUSED)


def test_comm_scoped_to_other_ranks_warns_and_decides_locally(monkeypatch):
    """B1-13: a mismatched comm is not used; the local verdict stands."""
    warnings = []
    monkeypatch.setattr(
        allreduce.logger,
        "warning",
        lambda message, *args: warnings.append(message % args),
    )

    class _NeverGathers(_FakeComm):
        def allgather(self, local):
            raise AssertionError("a mismatched comm must not be used")

    assert (
        allreduce._agree_on_support(_SUPPORTED, 8, _NeverGathers(4, None), None)
        == _SUPPORTED
    )
    assert len(warnings) == 1
    assert "support-agreement comm size 4 != world_size 8" in warnings[0]


def test_absent_collective_leaves_the_rank_local_decision_alone(monkeypatch):
    """B1-13: mpi4py is optional, so no collective means no new failure.

    ``MPIBackend`` is a *lazy* adapter -- with mpi4py absent the constructor
    succeeds and the first attribute access raises -- so the probe has to reach
    ``Get_size()``; a stub that raises in ``__init__`` would pass while the real
    adapter still turned a rejected creation into an ImportError.
    """

    class _LazyNoMpi:
        def __init__(self):
            pass

        def Get_size(self):
            raise ImportError("mpi4py is not installed")

    class _EagerNoMpi:
        def __init__(self):
            raise ImportError("mpi4py is not installed")

    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    for adapter in (_LazyNoMpi, _EagerNoMpi):
        monkeypatch.setattr(allreduce, "MPIBackend", adapter)
        assert allreduce._agree_on_support(_SUPPORTED, 2, None, None) == _SUPPORTED
        assert allreduce._agree_on_support(_REFUSED, 2, None, None) == _REFUSED


# ---------------------------------------------------------------------------
# Positive path on an architecture the kernels cover
# ---------------------------------------------------------------------------


def test_hopper_host_answers_supported_for_the_fast_path():
    """B1-14: where the architecture exists, the query says so.

    Skips unless the host is SM90a, the way the attention suites gate on Hopper:
    this asserts the *positive* answer, and the negative direction is covered on
    every host by the capability-refusal cases above.
    """
    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("requires SM90a")
    major, minor = torch.cuda.get_device_capability()
    cc = major * 10 + minor
    assert cc == HOPPER_COMPUTE_CAPABILITY
    assert create_allreduce_fusion_workspace.is_compute_capability_supported(cc) is True
    assert create_allreduce_fusion_workspace.is_backend_supported("trtllm", cc) is True


def test_two_rank_allreduce_matches_the_reference():
    """B1-11: the supported collective against ``torch.distributed``.

    Run with: mpirun -np 2 pytest -s tests/comm/test_comm_support_checks.py
    Skips unless the host is SM90a and at least two ranks are launched; the
    TRT-LLM allreduce kernels this builds a workspace for are Hopper-or-newer.
    """
    if not torch.cuda.is_available() or not is_sm90a_supported(torch.device("cuda")):
        pytest.skip("requires SM90a")
    from tests.test_helpers.comm import (
        cleanup_torch_distributed,
        init_torch_distributed_from_mpi,
        setup_mpi_and_cuda,
    )

    rank, world_size, _ = setup_mpi_and_cuda()
    if world_size != 2:
        pytest.skip(f"requires exactly 2 ranks, got {world_size}")

    init_torch_distributed_from_mpi()
    workspace = None
    try:
        workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=2,
            rank=rank,
            max_token_num=64,
            hidden_dim=256,
            dtype=torch.float16,
        )
        assert workspace.backend == "trtllm"

        token_num, hidden_dim = 64, 256
        x = torch.arange(token_num * hidden_dim, dtype=torch.float16, device="cuda")
        x = x.view(token_num, hidden_dim)
        reference = x.clone()
        dist.all_reduce(reference)

        output = allreduce_fusion(
            input=x,
            workspace=workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
        )
        torch.testing.assert_close(output, reference)
    finally:
        if workspace is not None:
            workspace.destroy()
        cleanup_torch_distributed()


# ---------------------------------------------------------------------------
# Tracker hygiene
# ---------------------------------------------------------------------------


def test_optional_dependency_comm_apis_skip_instead_of_xfailing():
    """B1-13: an absent optional dependency is a skip, never a quiet xfail."""
    # Plain importorskip form: the reason kwarg needs pytest >= 8.2.
    nvshmem_allreduce = pytest.importorskip("flashinfer.comm.nvshmem_allreduce")
    assert not hasattr(
        nvshmem_allreduce.NVSHMEMAllReduce, "is_compute_capability_supported"
    )


@pytest.mark.xfail(
    strict=True,
    reason="support checks for the remaining comm APIs are tracked in #2224",
)
@pytest.mark.parametrize(
    "api",
    [
        comm.vllm_all_reduce,
        comm.vllm_init_custom_ar,
        comm.vllm_dispose,
        comm.vllm_register_buffer,
        comm.vllm_register_graph_buffers,
        comm.MoeAlltoAll.dispatch,
    ],
    ids=lambda api: api.__qualname__,
)
def test_tracker_entry_not_converted_yet(api):
    """The #2224 entries that this slice did not convert, kept as strict xfails.

    Only converted entries may answer the queries: a non-strict xfail would keep
    reporting the same state after a check appeared, so ``strict=True`` turns a
    stray attribute into a failure that forces the entry to be converted into a
    behavior test such as the ones above.
    """
    assert hasattr(api, "is_compute_capability_supported")
