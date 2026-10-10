"""SM12x MoE workspace cache must never free graph-referenced storage.

A captured CUDA graph replays through the raw device pointers of whatever
workspace was live during capture. If the cache replaces that workspace (a
larger routed_rows call arrives) or is cleared, dropping the old workspace
returns its pages to the allocator and later replay overwrites an unrelated
allocation. The same holds for prepared weights (converted scale factors,
padded and W4A16-packed weights) that the weight caches own. These tests pin
the marking and retirement semantics and drive the real cache functions
through hit, replacement, clear, and release.
"""

import gc
import threading
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

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
_WEIGHT_CACHES = ("_WEIGHT_CACHE", "_W4A16_WEIGHT_CACHE", "_PADDED_WEIGHT_CACHE")


@pytest.fixture(autouse=True)
def _isolated_workspace_state():
    saved = {name: dict(getattr(moe_dispatch, name)) for name in _MODULE_CACHES}
    saved_parked = list(moe_dispatch._GRAPH_REFERENCED_WORKSPACES)
    saved_weight_keys = set(moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS)
    saved_generations = dict(moe_dispatch._WEIGHT_ENTRY_GENERATIONS)
    capture_probe = moe_dispatch._is_cuda_graph_capturing
    for name in ("_WORKSPACE_CACHE", *_WEIGHT_CACHES):
        getattr(moe_dispatch, name).clear()
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.clear()
    moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS.clear()
    moe_dispatch._WEIGHT_ENTRY_GENERATIONS.clear()
    yield
    assert moe_dispatch._is_cuda_graph_capturing is capture_probe
    for name in _MODULE_CACHES:
        cache = getattr(moe_dispatch, name)
        cache.clear()
        cache.update(saved[name])
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.clear()
    moe_dispatch._GRAPH_REFERENCED_WORKSPACES.extend(saved_parked)
    moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS.clear()
    moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS.update(saved_weight_keys)
    moe_dispatch._WEIGHT_ENTRY_GENERATIONS.clear()
    moe_dispatch._WEIGHT_ENTRY_GENERATIONS.update(saved_generations)


def _lookup(routed_rows):
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


def _get(routed_rows, capturing):
    with mock.patch.object(
        moe_dispatch, "_is_cuda_graph_capturing", return_value=capturing
    ):
        return _lookup(routed_rows)


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
    entered = threading.Event()
    proceed = threading.Event()
    thread_state = threading.local()

    def capturing_only_in_capture_thread():
        # One mock serves every thread, so no thread restores another's patch.
        if not getattr(thread_state, "capturing", False):
            return False
        entered.set()
        proceed.wait(timeout=5)
        return True

    def capture_lookup():
        thread_state.capturing = True
        _lookup(routed_rows=8)

    with (
        mock.patch.object(
            moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
        ),
        mock.patch.object(
            moe_dispatch, "_is_cuda_graph_capturing", capturing_only_in_capture_thread
        ),
    ):
        first = _lookup(routed_rows=8)

        capture_thread = threading.Thread(target=capture_lookup)
        capture_thread.start()
        assert entered.wait(timeout=5)

        replace_thread = threading.Thread(target=lambda: _lookup(routed_rows=16))
        replace_thread.start()
        # The replacement must block on the cache lock until marking finishes.
        replace_thread.join(timeout=0.2)
        assert replace_thread.is_alive()
        proceed.set()
        capture_thread.join(timeout=5)
        replace_thread.join(timeout=5)
        assert not capture_thread.is_alive()
        assert not replace_thread.is_alive()

    assert first._sm12x_graph_referenced
    assert [first] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


def _capture_weight_entries(capturing):
    """Insert one entry in each prepared-weight cache and read it back."""
    entries = {}
    for name in _WEIGHT_CACHES:
        cache = getattr(moe_dispatch, name)
        key = (name, capturing)
        value = (torch.empty(4), torch.empty(2))
        with mock.patch.object(
            moe_dispatch, "_is_cuda_graph_capturing", return_value=False
        ):
            moe_dispatch._put_weight_cache_entry(cache, key, value)
        with mock.patch.object(
            moe_dispatch, "_is_cuda_graph_capturing", return_value=capturing
        ):
            assert moe_dispatch._get_weight_cache_entry(cache, key) is value
        entries[name] = value
    return entries


def test_clear_caches_parks_prepared_weights_read_during_capture():
    captured = _capture_weight_entries(capturing=True)
    plain = _capture_weight_entries(capturing=False)

    moe_dispatch.clear_sm120_moe_caches()

    for name in _WEIGHT_CACHES:
        assert getattr(moe_dispatch, name) == {}
    parked = moe_dispatch._GRAPH_REFERENCED_WORKSPACES
    assert all(any(value is entry for entry in parked) for value in captured.values())
    assert not any(any(value is entry for entry in parked) for value in plain.values())
    assert not moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS

    moe_dispatch.release_graph_referenced_workspaces()
    assert moe_dispatch._GRAPH_REFERENCED_WORKSPACES == []


def test_prepared_weights_inserted_during_capture_are_parked_on_clear():
    value = (torch.empty(4),)
    with mock.patch.object(moe_dispatch, "_is_cuda_graph_capturing", return_value=True):
        moe_dispatch._put_weight_cache_entry(
            moe_dispatch._PADDED_WEIGHT_CACHE, ("k",), value
        )
    moe_dispatch.clear_sm120_moe_caches()
    assert [value] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


@pytest.mark.parametrize("capturing", [True, False])
def test_source_collection_parks_graph_referenced_prepared_weights(capturing):
    cache = moe_dispatch._WEIGHT_CACHE
    source = torch.empty(4)
    value = (torch.empty(4), torch.empty(2))
    with mock.patch.object(
        moe_dispatch, "_is_cuda_graph_capturing", return_value=capturing
    ):
        moe_dispatch._put_weight_cache_entry(cache, ("evict",), value, source)

    del source
    gc.collect()

    assert cache == {}
    parked = moe_dispatch._GRAPH_REFERENCED_WORKSPACES
    assert parked == ([value] if capturing else [])
    assert not moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS


def test_source_collection_while_holding_the_cache_lock_does_not_deadlock():
    cache = moe_dispatch._WEIGHT_CACHE
    source = torch.empty(4)
    moe_dispatch._put_weight_cache_entry(cache, ("lock",), (torch.empty(1),), source)
    sources = [source]
    del source

    def collect_under_lock():
        # Garbage collection can run eviction finalizers on a thread that
        # already holds the lock, e.g. while it allocates a workspace.
        with moe_dispatch._WORKSPACE_CACHE_LOCK:
            sources.clear()
            gc.collect()

    thread = threading.Thread(target=collect_under_lock, daemon=True)
    thread.start()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert cache == {}


def test_source_collection_does_not_evict_a_later_entry_for_the_key():
    cache = moe_dispatch._PADDED_WEIGHT_CACHE
    for _ in range(50):
        old_source = torch.empty(4)
        moe_dispatch._put_weight_cache_entry(
            cache, ("key",), (torch.empty(1),), old_source
        )
        # Free the old entry so a later entry can reuse its address.
        moe_dispatch.clear_sm120_moe_caches()
        gc.collect()

        new_source = torch.empty(4)
        new_value = (torch.empty(1),)
        moe_dispatch._put_weight_cache_entry(cache, ("key",), new_value, new_source)
        del old_source
        gc.collect()

        assert cache == {("key",): new_value}
        moe_dispatch.clear_sm120_moe_caches()


def test_source_alias_kept_across_clear_evicts_only_the_current_entry():
    cache = moe_dispatch._PADDED_WEIGHT_CACHE
    source = torch.empty(4)
    moe_dispatch._put_weight_cache_entry(cache, ("key",), (torch.empty(1),), source)
    moe_dispatch.clear_sm120_moe_caches()
    moe_dispatch._put_weight_cache_entry(cache, ("key",), (torch.empty(1),), source)

    del source
    gc.collect()

    assert cache == {}
    assert not moe_dispatch._WEIGHT_ENTRY_GENERATIONS


def _w4a16_sources():
    return {
        name: torch.empty(4)
        for name in (
            "w1_weight",
            "w1_weight_sf",
            "w1_alpha",
            "w2_weight",
            "w2_weight_sf",
            "w2_alpha",
        )
    }


def test_delayed_preparation_keeps_the_entry_captured_in_between():
    sources = _w4a16_sources()

    def lookup():
        return moe_dispatch._get_w4a16_packed_weights(
            **sources, activation="silu", params_dtype=torch.bfloat16
        )

    captured = SimpleNamespace(name="captured")
    delayed = SimpleNamespace(name="delayed")

    def prepare_slowly(*args, **kwargs):
        # While this caller prepares, another caller misses, inserts its own
        # entry, and a graph captures that entry.
        with mock.patch.object(
            moe_dispatch, "prepare_w4a16_packed_weights", return_value=captured
        ):
            assert lookup() is captured
        with mock.patch.object(
            moe_dispatch, "_is_cuda_graph_capturing", return_value=True
        ):
            assert lookup() is captured
        return delayed

    with (
        mock.patch.object(moe_dispatch, "_is_cuda_graph_capturing", return_value=False),
        mock.patch.object(
            moe_dispatch, "prepare_w4a16_packed_weights", side_effect=prepare_slowly
        ),
    ):
        # The delayed caller keeps the buffers it prepared on its own stream
        # rather than the cached ones, which may still be pending elsewhere.
        assert lookup() is delayed

    assert list(moe_dispatch._W4A16_WEIGHT_CACHE.values()) == [captured]
    moe_dispatch.clear_sm120_moe_caches()
    assert [captured] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES


def test_losing_insert_during_capture_parks_its_own_buffers():
    cache = moe_dispatch._PADDED_WEIGHT_CACHE
    cached = (torch.empty(1),)
    own = (torch.empty(1),)
    moe_dispatch._put_weight_cache_entry(cache, ("key",), cached)
    with mock.patch.object(moe_dispatch, "_is_cuda_graph_capturing", return_value=True):
        assert moe_dispatch._put_weight_cache_entry(cache, ("key",), own) is own

    assert cache == {("key",): cached}
    assert [own] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES
    assert not moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS


def test_release_synchronizes_before_dropping_graph_referenced_storage():
    with mock.patch.object(
        moe_dispatch, "allocate_sm120_moe_workspace", side_effect=_fake_allocate
    ):
        parked = _get(routed_rows=8, capturing=True)
        cached = _get(routed_rows=16, capturing=True)
    weights = _capture_weight_entries(capturing=True)
    assert [parked] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES
    devices = {torch.device("cuda", 0), torch.device("cuda", 1)}
    synchronized = []

    def synchronize(device):
        # Replays may still be in flight: storage must stay referenced and
        # marked until every device has drained.
        assert [parked] == moe_dispatch._GRAPH_REFERENCED_WORKSPACES
        assert cached._sm12x_graph_referenced
        assert moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS
        synchronized.append(device)

    def devices_of(entries):
        entries = list(entries)
        assert any(entry is parked for entry in entries)
        assert any(entry is cached for entry in entries)
        for value in weights.values():
            assert any(entry is value for entry in entries)
        return devices

    with (
        mock.patch.object(moe_dispatch, "_cuda_devices_of", side_effect=devices_of),
        mock.patch.object(torch.cuda, "synchronize", side_effect=synchronize),
    ):
        moe_dispatch.release_graph_referenced_workspaces()

    assert set(synchronized) == devices
    assert moe_dispatch._GRAPH_REFERENCED_WORKSPACES == []
    assert not moe_dispatch._GRAPH_REFERENCED_WEIGHT_KEYS
    assert not cached._sm12x_graph_referenced


def test_release_without_graph_referenced_storage_does_not_synchronize():
    with mock.patch.object(torch.cuda, "synchronize") as synchronize:
        moe_dispatch.release_graph_referenced_workspaces()
    synchronize.assert_not_called()


def test_cuda_devices_of_reads_tuple_and_attribute_tensors():
    entries = [
        (torch.empty(1), None, 3),
        SimpleNamespace(storage=torch.empty(1), rows=4),
    ]
    assert moe_dispatch._cuda_devices_of(entries) == set()
    if torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
        entries = [
            (torch.empty(1, device=device),),
            SimpleNamespace(storage=torch.empty(1, device=device)),
        ]
        assert moe_dispatch._cuda_devices_of(entries) == {device}


def _sm12x_functional_available():
    try:
        from flashinfer.cute_dsl import is_cute_dsl_available
        from flashinfer.jit.cpp_ext import get_cuda_version
        from flashinfer.utils import is_sm120a_supported, is_sm121a_supported
    except Exception:
        return False
    if not torch.cuda.is_available() or not is_cute_dsl_available():
        return False
    try:
        if get_cuda_version().major < 13:
            return False
    except Exception:
        return False
    device = torch.device("cuda")
    return is_sm120a_supported(device) or is_sm121a_supported(device)


def _cuda_storages(*caches):
    """(data_ptr, nbytes) of the CUDA storages held by cache entries.

    Returns plain integers so that no tensor reference outlives the call.
    """
    storages = set()
    for cache in caches:
        for entry in cache.values():
            values = entry if isinstance(entry, tuple) else tuple(vars(entry).values())
            for value in values:
                if isinstance(value, torch.Tensor) and value.is_cuda:
                    storage = value.untyped_storage()
                    storages.add((storage.data_ptr(), storage.nbytes()))
    return storages


def _padded_weight_storages():
    """Storages of the padded weights and scale factors, not their inputs."""
    storages = set()
    for entry in moe_dispatch._PADDED_WEIGHT_CACHE.values():
        for value in entry[:4]:
            storage = value.untyped_storage()
            storages.add((storage.data_ptr(), storage.nbytes()))
    return storages


@pytest.mark.skipif(
    not _sm12x_functional_available(),
    reason="Requires an SM120/SM121 GPU, CuTe DSL and CUDA 13",
)
def test_functional_capture_replays_correctly_after_cache_clear(monkeypatch):
    from flashinfer import b12x_fused_moe

    from .utils import create_b12x_moe_tensors

    # The static backend pads an intermediate size that is not a multiple of
    # its retained N group into cache-owned weights and scale factors. The
    # captured graph reads only those copies, so after clearing only the
    # cache can keep them alive.
    monkeypatch.setattr(moe_dispatch, "_FORCED_BACKEND", "static")
    num_tokens, hidden_size, intermediate_size = 128, 512, 704
    num_experts, top_k = 8, 2
    assert intermediate_size % moe_dispatch._STATIC_RETAINED_GROUP_N != 0
    tensors = create_b12x_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=num_experts,
        top_k=top_k,
    )
    output = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device="cuda")

    def run():
        b12x_fused_moe(
            x=tensors["x_bf16"],
            w1_weight=tensors["w1_weight"],
            w1_weight_sf=tensors["w1_weight_sf"],
            w1_alpha=tensors["w1_alpha"],
            fc2_input_scale=tensors["fc2_input_scale"],
            w2_weight=tensors["w2_weight"],
            w2_weight_sf=tensors["w2_weight_sf"],
            w2_alpha=tensors["w2_alpha"],
            token_selected_experts=tensors["token_selected_experts"],
            token_final_scales=tensors["token_final_scales"],
            num_experts=num_experts,
            top_k=top_k,
            num_local_experts=num_experts,
            output=output,
        )

    run()
    torch.cuda.synchronize()
    input_ptrs = {
        value.untyped_storage().data_ptr()
        for value in tensors.values()
        if isinstance(value, torch.Tensor) and value.is_cuda
    }
    padded = _padded_weight_storages()
    assert padded
    assert not {ptr for ptr, _ in padded} & input_ptrs

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    expected = output.clone()
    assert torch.isfinite(expected.float()).all()

    sizes = [
        nbytes
        for _, nbytes in _cuda_storages(
            moe_dispatch._PADDED_WEIGHT_CACHE, moe_dispatch._WEIGHT_CACHE
        )
    ]
    try:
        moe_dispatch.clear_sm120_moe_caches()
        # Without retention the allocator hands the freed blocks back here,
        # and replay reads NaN scale factors (0xFF) and garbage weights.
        scribbles = [
            torch.full((nbytes,), 0xFF, dtype=torch.uint8, device="cuda")
            for nbytes in sizes
        ]
        output.zero_()
        graph.replay()
        torch.cuda.synchronize()
        del scribbles
        torch.testing.assert_close(output, expected, rtol=2e-2, atol=2e-2)
    finally:
        del graph
        moe_dispatch.release_graph_referenced_workspaces()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
