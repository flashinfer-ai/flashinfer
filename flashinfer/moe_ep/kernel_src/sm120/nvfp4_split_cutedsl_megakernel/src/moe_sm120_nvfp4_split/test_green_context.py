"""CPU-only regression tests for context binding and K3 completion scope."""

from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest
from cuda import bindings

from .runtime import green_context as green


def test_library_lookup_deduplicates_libraries(monkeypatch):
    calls = []
    driver = NS(
        CUlibrary=int,
        cuLibraryGetKernelCount=lambda lib: (calls.append(lib) or 0, 2),
        cuLibraryEnumerateKernels=lambda n, lib: (0, [lib * 10, lib * 10 + 1]),
        cuKernelGetFunction=lambda kernel: (0, kernel + 100),
    )
    monkeypatch.setattr(bindings, "driver", driver)
    result = green.NativeGreenContextGraph._library_kernels(
        (NS(library=1), NS(library=1), NS(library=2))
    )
    assert calls == [1, 2]
    assert result == {110: 10, 111: 11, 120: 20, 121: 21}


def test_binding_uses_cukernel_and_preserves_launch_attributes(monkeypatch):
    writes = []
    driver = NS(
        CUfunction=int,
        CUfunction_attribute=NS(
            CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES=1,
            CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT=2,
        ),
        cuFuncGetAttribute=lambda attribute, function: (
            0,
            {1: 98304, 2: 100}[attribute],
        ),
        cuKernelSetAttribute=lambda *args: (writes.append(args) or 0,),
    )
    monkeypatch.setattr(bindings, "driver", driver)
    params = NS(func=123, kern=0, ctx=0)
    green.NativeGreenContextGraph._bind_kernel(params, 123, 456, 789, 3)
    assert (params.func, params.kern, params.ctx) == (0, 456, 789)
    assert writes == [(1, 98304, 456, 3), (2, 100, 456, 3)]


def test_cuda_error_is_not_silently_ignored():
    with pytest.raises(RuntimeError, match="test operation failed"):
        green._check_cuda((1,), "test operation")


def _graph(monkeypatch, post):
    log = []
    root = NS(
        cuda_stream=11,
        wait_event=lambda event: log.append("root waits caller"),
        synchronize=lambda: log.append("root synchronize"),
    )
    caller = NS(cuda_stream=22, wait_event=lambda event: log.append("caller waits K3"))
    driver = NS(
        CUstream=int,
        cuGraphLaunch=lambda *args: (log.append("graph") or 0,),
        cuGraphExecDestroy=lambda *args: (log.append("destroy exec") or 0,),
        cuGraphDestroy=lambda *args: (log.append("destroy graph") or 0,),
        cuGreenCtxDestroy=lambda *args: (log.append("destroy context") or 0,),
    )
    monkeypatch.setattr(bindings, "driver", driver)
    monkeypatch.setattr(green.torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(green.torch.cuda, "is_current_stream_capturing", lambda: False)
    graph = green.NativeGreenContextGraph(
        root_stream=root,
        k1_sm_count=72,
        k2_sm_count=38,
        _graph=1,
        _graph_exec=2,
        _green_contexts=(3, 4),
        _owns_green_contexts=True,
        _launch_ready=NS(record=lambda stream: log.append("caller ready")),
        _launch_done=NS(record=lambda stream: log.append("done after K3")),
        _compiled_kernels=(NS(library=1),),
        _post_graph_k3=(lambda: post(log)) if post else None,
    )
    return graph, caller, log


def test_completion_includes_post_graph_k3_and_every_replay(monkeypatch):
    graph, caller, log = _graph(monkeypatch, lambda log: log.append("K3"))
    for _ in range(2):
        graph.launch(caller)
    assert (
        log
        == [
            "caller ready",
            "root waits caller",
            "graph",
            "K3",
            "done after K3",
            "caller waits K3",
        ]
        * 2
    )


def test_no_k3_callback_does_not_add_a_launch(monkeypatch):
    graph, caller, log = _graph(monkeypatch, None)
    graph.launch(caller)
    assert log == [
        "caller ready",
        "root waits caller",
        "graph",
        "done after K3",
        "caller waits K3",
    ]


def test_failed_k3_does_not_publish_completion(monkeypatch):
    def fail(log):
        raise RuntimeError("K3 launch failed")

    graph, caller, log = _graph(monkeypatch, fail)
    with pytest.raises(RuntimeError, match="K3 launch failed"):
        graph.launch(caller)
    assert "done after K3" not in log
    assert "caller waits K3" not in log


def test_close_releases_kernel_owners_after_graph_and_contexts(monkeypatch):
    graph, _, log = _graph(monkeypatch, lambda log: None)
    assert graph._compiled_kernels
    graph.close()
    assert log == [
        "root synchronize",
        "destroy exec",
        "destroy graph",
        "destroy context",
        "destroy context",
    ]
    assert graph._compiled_kernels == () and graph._post_graph_k3 is None
    graph.close()
    assert len(log) == 5


def test_finalizer_precedes_k3_and_completion(monkeypatch):
    graph, caller, log = _graph(monkeypatch, lambda log: log.append("K3"))
    graph._post_graph_finalizer = lambda: log.append("finalizer")
    graph.launch(caller)
    assert log == [
        "caller ready",
        "root waits caller",
        "graph",
        "finalizer",
        "K3",
        "done after K3",
        "caller waits K3",
    ]


def test_failed_finalizer_does_not_publish_completion(monkeypatch):
    graph, caller, log = _graph(monkeypatch, lambda log: log.append("K3"))

    def fail():
        raise RuntimeError("finalizer failed")

    graph._post_graph_finalizer = fail
    with pytest.raises(RuntimeError, match="finalizer failed"):
        graph.launch(caller)
    assert "K3" not in log and "done after K3" not in log


def test_outer_capture_child_then_finalizer_then_k3(monkeypatch):
    graph, caller, log = _graph(monkeypatch, lambda log: log.append("K3"))
    graph._post_graph_finalizer = lambda: log.append("finalizer")
    driver = bindings.driver
    driver.CUstreamCaptureStatus = NS(CU_STREAM_CAPTURE_STATUS_ACTIVE=1)
    driver.CUstreamUpdateCaptureDependencies_flags = NS(
        CU_STREAM_SET_CAPTURE_DEPENDENCIES=0
    )
    driver.cuStreamGetCaptureInfo = lambda stream: (0, 1, 9, 10, [11], None, 1)
    driver.cuGraphAddChildGraphNode = lambda *args: (log.append("child") or 0, 12)
    driver.cuStreamUpdateCaptureDependencies = lambda *args: (
        log.append("dependency") or 0,
    )
    monkeypatch.setattr(green.torch.cuda, "is_current_stream_capturing", lambda: True)
    graph.launch(caller)
    assert log == ["child", "dependency", "finalizer", "K3"]
