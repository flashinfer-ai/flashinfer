# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU-only graph-topology and lifecycle contracts for SM90 split MegaMoE."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
    GreenGraph,
    GreenGraphCaptureError,
    GreenGraphCleanupError,
    GreenGraphError,
)


class _FakeStatus(int):
    @property
    def name(self) -> str:
        return "CUDA_SUCCESS" if int(self) == 0 else "CUDA_ERROR_FAKE"


class _FakeDriver:
    CUstreamCaptureMode = SimpleNamespace(CU_STREAM_CAPTURE_MODE_THREAD_LOCAL=1)

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.active_captures: set[str] = set()
        self._child_index = 0
        self._node_index = 0
        self.fail_api: str | None = None
        self.fail_destroy_handles: set[str] = set()
        self.null_end_stream: str | None = None

    def _status(self, api: str) -> _FakeStatus:
        return _FakeStatus(7 if self.fail_api == api else 0)

    def cuStreamSynchronize(self, stream):
        self.calls.append(("sync", stream))
        return (self._status("cuStreamSynchronize"),)

    def cuStreamBeginCapture(self, stream, mode):
        self.calls.append(("begin", stream, mode))
        status = self._status("cuStreamBeginCapture")
        if int(status) == 0:
            self.active_captures.add(stream)
        return (status,)

    def cuStreamEndCapture(self, stream):
        self.calls.append(("end", stream))
        self.active_captures.discard(stream)
        status = self._status("cuStreamEndCapture")
        if int(status) != 0:
            return (status, None)
        if stream == self.null_end_stream:
            return (status, 0)
        self._child_index += 1
        return (status, f"child{self._child_index}")

    def cuGraphCreate(self, flags):
        self.calls.append(("graph_create", flags))
        return (self._status("cuGraphCreate"), "parent")

    def cuGraphAddChildGraphNode(self, parent, dependencies, num_dependencies, child):
        deps = None if dependencies is None else tuple(dependencies)
        self.calls.append(("add_child", parent, deps, num_dependencies, child))
        self._node_index += 1
        return (
            self._status("cuGraphAddChildGraphNode"),
            f"node{self._node_index}",
        )

    def cuGraphInstantiate(self, parent, flags):
        self.calls.append(("instantiate", parent, flags))
        return (self._status("cuGraphInstantiate"), "exec")

    def cuGraphLaunch(self, executable, stream):
        self.calls.append(("graph_launch", executable, stream))
        return (self._status("cuGraphLaunch"),)

    def cuGraphExecDestroy(self, executable):
        self.calls.append(("destroy_exec", executable))
        status = 9 if executable in self.fail_destroy_handles else 0
        return (_FakeStatus(status),)

    def cuGraphDestroy(self, graph):
        self.calls.append(("destroy_graph", graph))
        status = 9 if graph in self.fail_destroy_handles else 0
        return (_FakeStatus(status),)


def _launch(role: str, driver: _FakeDriver, launches: list[tuple]):
    def inner(stream):
        launches.append((role, stream))
        driver.calls.append(("enqueue", role, stream))

    return inner


def _capture_cold(driver: _FakeDriver, launches: list[tuple]) -> GreenGraph:
    return GreenGraph.capture(
        k0_stream="s0",
        k0_launch=_launch("k0", driver, launches),
        k1_stream="s1",
        k1_launch=_launch("k1", driver, launches),
        k2_stream="s2",
        k2_launch=_launch("k2", driver, launches),
        k3_stream="s3",
        k3_launch=_launch("k3", driver, launches),
        driver=driver,
    )


def _capture_steady(driver: _FakeDriver, launches: list[tuple]) -> GreenGraph:
    return GreenGraph.capture_steady(
        k1_stream="s1",
        k1_launch=_launch("k1", driver, launches),
        k2_stream="s2",
        k2_launch=_launch("k2", driver, launches),
        k3_stream="s3",
        k3_launch=_launch("k3_tail", driver, launches),
        driver=driver,
    )


def test_cold_graph_is_k0_fork_k1_k2_join_k3() -> None:
    driver = _FakeDriver()
    launches: list[tuple] = []
    graph = _capture_cold(driver, launches)

    assert launches == [
        ("k0", "s0"),
        ("k1", "s1"),
        ("k2", "s2"),
        ("k3", "s3"),
    ]
    assert graph.topology.k0_child == "child1"
    assert graph.topology.k1_child == "child2"
    assert graph.topology.k2_child == "child3"
    assert graph.topology.k3_child == "child4"
    assert [call for call in driver.calls if call[0] == "add_child"] == [
        ("add_child", "parent", None, 0, "child1"),
        ("add_child", "parent", ("node1",), 1, "child2"),
        ("add_child", "parent", ("node1",), 1, "child3"),
        ("add_child", "parent", ("node2", "node3"), 2, "child4"),
    ]

    # The defining split invariant: K1 and K2 have the same sole parent and
    # neither child depends on the other child.
    k1_add, k2_add = [call for call in driver.calls if call[0] == "add_child"][1:3]
    assert k1_add[2] == k2_add[2] == ("node1",)

    graph.launch("launch_stream")
    graph.close()
    assert driver.calls[-7:] == [
        ("sync", "launch_stream"),
        ("destroy_exec", "exec"),
        ("destroy_graph", "parent"),
        ("destroy_graph", "child4"),
        ("destroy_graph", "child3"),
        ("destroy_graph", "child2"),
        ("destroy_graph", "child1"),
    ]


def test_steady_graph_is_independent_k1_k2_fork_join_k3_tail() -> None:
    driver = _FakeDriver()
    graph = _capture_steady(driver, [])

    assert graph.topology.k0_child is None
    assert graph.topology.k0_node is None
    assert [call for call in driver.calls if call[0] == "add_child"] == [
        ("add_child", "parent", None, 0, "child1"),
        ("add_child", "parent", None, 0, "child2"),
        ("add_child", "parent", ("node1", "node2"), 2, "child3"),
    ]
    assert ("sync", "s0") not in driver.calls

    graph.launch("launch_stream")
    graph.close()
    assert driver.calls[-6:] == [
        ("sync", "launch_stream"),
        ("destroy_exec", "exec"),
        ("destroy_graph", "parent"),
        ("destroy_graph", "child3"),
        ("destroy_graph", "child2"),
        ("destroy_graph", "child1"),
    ]


def test_launch_sync_and_close_obey_fixed_graph_lifetime() -> None:
    driver = _FakeDriver()
    graph = _capture_cold(driver, [])
    graph.launch("torch_current")
    graph.synchronize()

    assert graph.last_launch_stream == "torch_current"
    assert driver.calls[-2:] == [
        ("graph_launch", "exec", "torch_current"),
        ("sync", "torch_current"),
    ]
    graph.close(synchronize=False)
    count = len(driver.calls)
    graph.close()
    assert len(driver.calls) == count
    with pytest.raises(GreenGraphError, match="already closed"):
        graph.launch("late")


def test_launch_failure_does_not_publish_a_last_stream() -> None:
    driver = _FakeDriver()
    graph = _capture_cold(driver, [])
    driver.fail_api = "cuGraphLaunch"

    with pytest.raises(GreenGraphError, match="cuGraphLaunch failed"):
        graph.launch("bad_stream")
    assert graph.last_launch_stream is None
    graph.close()


@pytest.mark.parametrize("failed_role", ["k1", "k2"])
def test_launch_callback_failure_ends_capture_and_cleans_children(
    failed_role: str,
) -> None:
    driver = _FakeDriver()

    def good(_stream):
        return None

    def fail(_stream):
        raise ValueError(f"bad {failed_role} launch")

    launches = {"k0": good, "k1": good, "k2": good, "k3": good}
    launches[failed_role] = fail
    with pytest.raises(GreenGraphCaptureError, match=f"bad {failed_role} launch"):
        GreenGraph.capture(
            k0_stream="s0",
            k0_launch=launches["k0"],
            k1_stream="s1",
            k1_launch=launches["k1"],
            k2_stream="s2",
            k2_launch=launches["k2"],
            k3_stream="s3",
            k3_launch=launches["k3"],
            driver=driver,
        )

    assert driver.active_captures == set()
    assert ("end", "s1" if failed_role == "k1" else "s2") in driver.calls
    destroys = [call for call in driver.calls if call[0] == "destroy_graph"]
    assert len(destroys) >= 1


def test_end_capture_failure_leaves_no_active_capture() -> None:
    driver = _FakeDriver()
    original_end = driver.cuStreamEndCapture

    def fail_second_end(stream):
        if stream == "s2":
            driver.calls.append(("end", stream))
            driver.active_captures.discard(stream)
            return (_FakeStatus(11), None)
        return original_end(stream)

    driver.cuStreamEndCapture = fail_second_end
    with pytest.raises(GreenGraphCaptureError, match="cuStreamEndCapture"):
        _capture_cold(driver, [])

    assert driver.active_captures == set()
    assert ("destroy_graph", "child1") in driver.calls


def test_null_child_graph_is_rejected() -> None:
    driver = _FakeDriver()
    driver.null_end_stream = "s1"
    with pytest.raises(GreenGraphCaptureError, match="null graph"):
        _capture_cold(driver, [])
    assert driver.active_captures == set()


@pytest.mark.parametrize("steady", [False, True])
def test_parent_instantiation_failure_destroys_every_child(steady: bool) -> None:
    driver = _FakeDriver()
    driver.fail_api = "cuGraphInstantiate"
    with pytest.raises(GreenGraphCaptureError, match="cuGraphInstantiate"):
        (_capture_steady if steady else _capture_cold)(driver, [])

    destroys = [call for call in driver.calls if call[0].startswith("destroy")]
    assert destroys[0] == ("destroy_graph", "parent")
    assert len(destroys) == (4 if steady else 5)


@pytest.mark.parametrize("steady", [False, True])
def test_cleanup_attempts_every_handle_before_raising(steady: bool) -> None:
    driver = _FakeDriver()
    graph = (_capture_steady if steady else _capture_cold)(driver, [])
    driver.fail_destroy_handles = {"exec", "child2"}

    with pytest.raises(GreenGraphCleanupError) as error:
        graph.close()
    assert "cuGraphExecDestroy" in str(error.value)
    destroy_calls = [call for call in driver.calls if call[0].startswith("destroy")]
    assert destroy_calls[0] == ("destroy_exec", "exec")
    assert len(destroy_calls) == (5 if steady else 6)


def test_non_callable_launch_is_rejected_before_cuda_calls() -> None:
    driver = _FakeDriver()
    with pytest.raises(TypeError, match="k2_launch"):
        GreenGraph.capture(
            k0_stream="s0",
            k0_launch=lambda _stream: None,
            k1_stream="s1",
            k1_launch=lambda _stream: None,
            k2_stream="s2",
            k2_launch=None,
            k3_stream="s3",
            k3_launch=lambda _stream: None,
            driver=driver,
        )
    assert driver.calls == []
