"""SM12x MoE workspace cache must never free graph-referenced storage.

A captured CUDA graph replays through the raw device pointers of whatever
workspace was live during capture. If the cache replaces that workspace (a
larger routed_rows call arrives) or is cleared, dropping the old workspace
returns its pages to the allocator and later replay overwrites an unrelated
allocation. These tests pin the marking and retirement semantics and drive
the real cache functions through hit, replacement, clear, and release.
"""

from types import SimpleNamespace
from unittest import mock

import pytest

from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch


_MODULE_CACHES = (
    "_WORKSPACE_CACHE",
    "_WEIGHT_CACHE",
    "_W4A16_WEIGHT_CACHE",
    "_PADDED_WEIGHT_CACHE",
    "_STATIC_KERNEL_CACHE",
    "_MICRO_KERNEL_CACHE",
    "_DIRECT_MICRO_LAUNCH_CACHE",
    "_DIRECT_MICRO_KERNEL_CACHE",
    "_DYNAMIC_KERNEL_CACHE",
)


@pytest.fixture(autouse=True)
def _isolated_workspace_state():
    saved = {name: dict(getattr(moe_dispatch, name)) for name in _MODULE_CACHES}
    saved_parked = list(moe_dispatch._GRAPH_REFERENCED_WORKSPACES)
    moe_dispatch._WORKSPACE_CACHE.clear()
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.clear()
    yield
    for name in _MODULE_CACHES:
        cache = getattr(moe_dispatch, name)
        cache.clear()
        cache.update(saved[name])
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.clear()
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.extend(saved_parked)


def _get(routed_rows, capturing):
    with mock.patch.object(
        moe_dispatch, "_is_cuda_graph_capturing", return_value=capturing
    ):
        return moe_dispatch._get_cached_workspace(
            backend="static",
            state_E=2,
            weight_E=2,
            routed_rows=routed_rows,
            k=64,
            n=64,
            num_topk=1,
            device="cpu",
            quant_mode="nvfp4",
        )


def _fake_allocate(*, routed_rows, **kwargs):
    return SimpleNamespace(max_rows=routed_rows)


def test_mark_is_noop_outside_capture():
    workspace = SimpleNamespace()
    with mock.patch.object(
        moe_dispatch, "_is_cuda_graph_capturing", return_value=False
    ):
        out = moe_dispatch._mark_graph_referenced(workspace)
    assert out is workspace
    assert not getattr(workspace, "_sm12x_graph_referenced", False)
    moe_dispatch._retire_workspace(workspace)
    assert moe_dispatch._GRAPH_REFERENCED_WORKSPACES == []


def test_capture_marked_workspace_is_parked_on_retire():
    workspace = SimpleNamespace()
    with mock.patch.object(moe_dispatch, "_is_cuda_graph_capturing", return_value=True):
        out = moe_dispatch._mark_graph_referenced(workspace)
    assert out is workspace
    assert workspace._sm12x_graph_referenced
    moe_dispatch._retire_workspace(None)
    moe_dispatch._retire_workspace(workspace)
    assert [workspace] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


def test_cache_hit_during_capture_marks_the_cached_workspace():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        first = _get(routed_rows=8, capturing=False)
        assert not getattr(first, "_sm12x_graph_referenced", False)
        hit = _get(routed_rows=8, capturing=True)
    assert hit is first
    assert hit._sm12x_graph_referenced


def test_replacement_parks_graph_referenced_workspace_and_drops_plain():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        captured = _get(routed_rows=8, capturing=True)
        replaced = _get(routed_rows=16, capturing=False)
        assert replaced is not captured
        assert [captured] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES

        plain = _get(routed_rows=16, capturing=False)
        assert plain is replaced
        _get(routed_rows=32, capturing=False)
    assert [captured] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


def test_workspace_allocated_during_capture_is_marked():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        workspace = _get(routed_rows=8, capturing=True)
    assert workspace._sm12x_graph_referenced


def test_clear_caches_parks_graph_referenced_entries_and_release_frees_them():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        captured = _get(routed_rows=8, capturing=True)
    plain = SimpleNamespace()
    moe_dispatch._WORKSPACE_CACHE[("plain",)] = plain
    moe_dispatch.clear_sm120_moe_caches()
    assert moe_dispatch._WORKSPACE_CACHE == {}
    assert [captured] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES

    moe_dispatch.release_graph_referenced_workspaces()
    assert moe_dispatch._GRAPH_REFERENCED_WORKSPACES == []


def test_release_unmarks_workspaces_still_in_the_cache():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        captured = _get(routed_rows=8, capturing=True)
        moe_dispatch.release_graph_referenced_workspaces()
        assert not captured._sm12x_graph_referenced
        # After release the entry behaves as plain: growth drops it.
        _get(routed_rows=16, capturing=False)
    assert moe_dispatch._GRAPH_REFERENCED_WORKSPACES == []


def test_concurrent_replacement_cannot_drop_a_capture_marked_workspace():
    import threading

    entered = threading.Event()
    proceed = threading.Event()

    def capturing_slowly():
        entered.set()
        proceed.wait(timeout=5)
        return True

    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        first = _get(routed_rows=8, capturing=False)

        def capture_lookup():
            with mock.patch.object(
                moe_dispatch, "_is_cuda_graph_capturing", capturing_slowly
            ):
                moe_dispatch._get_cached_workspace(
                    backend="static",
                    state_E=2,
                    weight_E=2,
                    routed_rows=8,
                    k=64,
                    n=64,
                    num_topk=1,
                    device="cpu",
                    quant_mode="nvfp4",
                )

        capture_thread = threading.Thread(target=capture_lookup)
        capture_thread.start()
        assert entered.wait(timeout=5)

        replace_thread = threading.Thread(
            target=lambda: _get(routed_rows=16, capturing=False)
        )
        replace_thread.start()
        # The replacement must block on the cache lock until marking finishes.
        replace_thread.join(timeout=0.2)
        assert replace_thread.is_alive()
        proceed.set()
        capture_thread.join(timeout=5)
        replace_thread.join(timeout=5)

    assert first._sm12x_graph_referenced
    assert [first] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
