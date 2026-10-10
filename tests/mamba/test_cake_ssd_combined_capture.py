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

# The Cake SSDCombined backend as a graph-capturable prepared submission:
# ``prepare`` materialises a call shape eagerly (pinned workspace, frozen
# family choice, loaded program) so that ``run`` on it with a static ``out``
# is a pure launch; a warm call captured without ``prepare`` allocates its
# outputs from the graph's private pool; a captured call refuses with
# ``CakeSSDCombinedCaptureError`` only for a shape this runner never ran or
# prepared and for a program never launched in this process; the workspace a
# graph recorded is pinned.  GPU tests need SM100/SM103; the registry and the
# capture logic are also exercised on CPU through a runner whose launcher is
# captured and whose capture status is faked.

import importlib
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace
from typing import Optional

import pytest
import torch

from flashinfer.mamba import SSDCombined
from flashinfer.mamba.cake_ssd_combined import (
    CakeSSDCombinedCaptureError,
    PreparedSSDCombined,
)

_CHUNK_PARALLEL_ENV = "FLASHINFER_CAKE_SSD_CHUNK_PARALLEL"


def _supported_device() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


requires_blackwell = pytest.mark.skipif(
    not _supported_device(), reason="Cake SSDCombined requires SM100 or SM103"
)


@pytest.fixture(autouse=True)
def _without_chunk_parallel_override(monkeypatch):
    """Every test starts with ``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` unset
    (``auto``): an unprepared call reads it, so a value inherited from the
    invoking shell would route every call through the other family or make
    every call raise.  Tests of the override set it through ``monkeypatch``
    afterwards."""

    monkeypatch.delenv(_CHUNK_PARALLEL_ENV, raising=False)


@pytest.fixture(autouse=True)
def _fresh_launched_programs(monkeypatch):
    """Every test starts with no program launched in this process, so the
    never-launched refusal is deterministic and GPU tests cannot leak launch
    state into the CPU tests that fake an sm_103a device."""

    monkeypatch.setattr(_module(), "_LOADED_PROGRAMS", set())


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


# ---------------------------------------------------------------------------
# GPU: prepared capture, bitwise replays and the refusals.


def _inputs(
    *,
    varlen=False,
    lengths=(96, 160),
    batch=2,
    seqlen=128,
    nheads=8,
    ngroups=8,
    seed=7,
):
    """One Cake SSDCombined call on the current device -- bf16 x/B/C/z/D,
    f32 dt/A/dt_bias, softplus with a finite clamp, bf16 initial states --
    batched or packed varlen on the ``cu_seqlens`` form.  Returns the
    constructor kwargs, the positional tensors and the keyword arguments."""

    torch.manual_seed(seed)
    if varlen:
        batch, seqlen = 1, sum(lengths)
    x = torch.randn(batch, seqlen, nheads, 64, device="cuda").to(torch.bfloat16)
    dt = torch.randn(batch, seqlen, nheads, device="cuda", dtype=torch.float32)
    A = -torch.rand(nheads, device="cuda", dtype=torch.float32) - 1.0
    B = torch.randn(batch, seqlen, ngroups, 128, device="cuda").to(torch.bfloat16)
    C = torch.randn_like(B)
    D = torch.randn(nheads, 64, device="cuda").to(torch.bfloat16)
    z = torch.randn_like(x)
    dt_bias = torch.rand(nheads, device="cuda", dtype=torch.float32) - 4.0
    num_seqs = len(lengths) if varlen else batch
    initial_states = torch.randn(num_seqs, nheads, 64, 128, device="cuda").to(
        torch.bfloat16
    )
    constructor = dict(
        chunk_size=128,
        nheads=nheads,
        headdim=64,
        dstate=128,
        ngroups=ngroups,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=True,
        d_has_hdim=True,
        has_initial_states=True,
        has_varlen=varlen,
        has_z=True,
        seq_idx_dtype=torch.int32,
    )
    arguments = dict(
        D=D,
        z=z,
        dt_bias=dt_bias,
        dt_softplus=True,
        dt_limit=(0.001, 0.1),
        initial_states=initial_states,
    )
    if varlen:
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + int(length))
        arguments["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32, device="cuda")
    return constructor, (x, dt, A, B, C), arguments


def _prepare(runner, tensors, arguments, **options):
    x, dt, A, B, C = tensors
    return runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **arguments, **options)


def _allocation_requests(device) -> Optional[int]:
    """The caching allocator's allocation-request count on ``device``
    (``None`` under a non-native allocator, whose statistics this reads)."""

    if torch.cuda.get_allocator_backend() != "native":
        return None
    return int(torch.cuda.memory_stats(device)["allocation.all.allocated"])


@contextmanager
def _no_allocation_requests(device):
    """Assert that the block issues no allocation request to the caching
    allocator: the loader's own counter cannot see an allocation it does not
    route through ``_allocating``."""

    before = _allocation_requests(device)
    yield
    after = _allocation_requests(device)
    if before is not None:
        assert after == before, f"{after - before} allocator request(s)"


def _replay_matches(graph, out, final, expected_out, expected_final, repeats=3):
    """Replay ``graph`` ``repeats`` times from NaN-filled outputs; every
    replay must reproduce the eager results bit for bit and request no
    allocation."""

    for _ in range(repeats):
        out.fill_(float("nan"))
        final.fill_(float("nan"))
        with _no_allocation_requests(out.device):
            graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected_out), "replayed output differs from eager"
        assert torch.equal(final, expected_final), "replayed final states differ"


@requires_blackwell
@pytest.mark.parametrize("form", ("batched", "cu_seqlens", "chunk_parallel"))
def test_prepared_run_replays_bitwise_from_a_cuda_graph(monkeypatch, form):
    """``prepare`` then capture ``run`` with static inputs and ``out``: the
    captured call allocates nothing, never reads the environment, returns
    the prepared static final-states buffer, pins its workspace, and three
    replays reproduce the eager run bit for bit -- for the exact family in
    batched bf16-state form, for a packed-varlen ``cu_seqlens`` shape and
    for the chunk-parallel program forced through the override.

    Chunk-parallel caveat: that program needs every CTA of its grid (one
    per SM) co-resident for its grid barrier, and the barrier words are
    shared per device.  A replay is stream-ordered like an eager launch, so
    it must not run concurrently with another chunk-parallel launch on the
    device (two streams); the same holds for two eager calls today."""

    if form == "chunk_parallel":
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
        constructor, tensors, arguments = _inputs(batch=2, seqlen=1024)
    else:
        constructor, tensors, arguments = _inputs(varlen=form == "cu_seqlens")
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    out = torch.empty(tensors[0].shape, dtype=torch.bfloat16, device="cuda")

    prepared = _prepare(runner, tensors, arguments, return_final_states=True, out=out)

    assert isinstance(prepared, PreparedSSDCombined)
    assert not cake.is_captured(prepared)
    family = "chunkpar" if form == "chunk_parallel" else "exact"
    mode = "varlen" if form == "cu_seqlens" else "batched"
    assert prepared.program_name == f"{family}_bf16_{mode}"
    assert cake.last_program_name is None  # prepare launched nothing
    # The override was read once by prepare; the prepared shape must never
    # consult it again (an invalid value would make a read raise).
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "invalid-after-prepare")
    # Eager reference on the prepared shape: a pure launch, and the first
    # launch of the program, which resolves the host shim's kernel handles.
    with _no_allocation_requests(out.device):
        out_eager, final = runner.run(
            *tensors, **arguments, out=out, return_final_states=True
        )
    assert out_eager is out
    assert cake._allocations_in_last_run == 0
    assert cake.last_program_name == prepared.program_name
    expected_out, expected_final = out.clone(), final.clone()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), _no_allocation_requests(out.device):
        out_captured, final_captured = runner.run(
            *tensors, **arguments, out=out, return_final_states=True
        )

    assert out_captured is out
    assert final_captured is final, "the prepared static final-states buffer"
    assert cake._allocations_in_last_run == 0
    assert cake._workspaces.captured(prepared.workspace_key)
    _replay_matches(graph, out, final, expected_out, expected_final)
    # The eager path on the pinned shape is unchanged after the capture.
    out_again, final_again = runner.run(
        *tensors, **arguments, out=out, return_final_states=True
    )
    assert out_again is out and final_again is final
    assert torch.equal(out, expected_out) and torch.equal(final, expected_final)


@requires_blackwell
def test_capture_of_an_unseen_shape_or_unlaunched_program_is_refused():
    """Two things a graph cannot record are refused by name: materialising
    the workspace of a call shape this runner never ran or prepared, and the
    first launch of a program (load, build, kernel-handle resolution).  The
    runner stays usable after each refusal."""

    constructor, tensors, arguments = _inputs()
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    out = torch.empty(tensors[0].shape, dtype=torch.bfloat16, device="cuda")

    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(CakeSSDCombinedCaptureError, match="has not run on this"),
        torch.cuda.graph(graph),
    ):
        runner.run(*tensors, **arguments, out=out)
    assert len(cake._workspaces) == 0

    # Prepared (workspace registered, program loaded) but never launched.
    prepared = _prepare(runner, tensors, arguments, out=out)
    graph = torch.cuda.CUDAGraph()
    with (
        pytest.raises(CakeSSDCombinedCaptureError, match="not been launched"),
        torch.cuda.graph(graph),
    ):
        runner.run(*tensors, **arguments, out=out)
    assert prepared.workspace_key in cake._workspaces
    assert not cake.is_captured(prepared)

    runner.run(*tensors, **arguments, out=out)
    torch.cuda.synchronize()
    assert cake.last_program_name == "exact_bf16_batched"


@requires_blackwell
def test_unprepared_warm_capture_allocates_from_the_graph_pool_and_replays_bitwise():
    """The pattern callers already use: one eager (warm) call, then capture
    the same call without ``prepare`` and without a static ``out``.  The
    captured call allocates its output and final states from the graph's
    private pool (counted, not refused), pins its workspace, and three
    replays reproduce the eager results bit for bit.  ``prepare`` itself is
    refused under capture."""

    constructor, tensors, arguments = _inputs()
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    eager_out, eager_final = runner.run(*tensors, **arguments)
    torch.cuda.synchronize()
    expected_out, expected_final = eager_out.clone(), eager_final.clone()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out, final = runner.run(*tensors, **arguments)

    assert out is not eager_out and final is not eager_final
    assert cake._allocations_in_last_run >= 2, "output + final states"
    (key,) = cake._workspaces.keys()
    assert cake._workspaces.captured(key)
    _replay_matches(graph, out, final, expected_out, expected_final)

    graph2 = torch.cuda.CUDAGraph()
    with (
        pytest.raises(CakeSSDCombinedCaptureError, match="outside CUDA-graph capture"),
        torch.cuda.graph(graph2),
    ):
        _prepare(runner, tensors, arguments, return_final_states=True)


@requires_blackwell
def test_replay_reads_mutated_static_inputs_through_the_recorded_copies():
    """A captured call with a strided ``dt`` (packed into the workspace's
    contiguous buffer by a recorded copy) and a bf16 ``dt_bias`` (widened by
    a recorded copy): after every input is overwritten in place, a replay
    must reproduce a fresh eager run on the new values bit for bit -- the
    recorded copies read the caller's storage at replay time."""

    constructor, (x, _dense_dt, A, B, C), arguments = _inputs()
    dt = torch.randn(x.shape[0], x.shape[1], 2 * x.shape[2], device="cuda")[:, :, ::2]
    assert not dt.is_contiguous()
    arguments = dict(arguments, dt_bias=arguments["dt_bias"].to(torch.bfloat16))
    runner = SSDCombined(**constructor, backend="cake")
    out = torch.empty(x.shape, dtype=torch.bfloat16, device="cuda")
    _prepare(runner, (x, dt, A, B, C), arguments, return_final_states=True, out=out)
    _, final = runner.run(
        x, dt, A, B, C, **arguments, out=out, return_final_states=True
    )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run(x, dt, A, B, C, **arguments, out=out, return_final_states=True)

    # Overwrite every static input in place, then replay against a fresh
    # eager run on the new values.
    torch.manual_seed(11)
    for tensor in (x, B, C, arguments["z"], arguments["initial_states"]):
        tensor.copy_(torch.randn(tensor.shape, device="cuda").to(tensor.dtype))
    dt.copy_(torch.randn(dt.shape, device="cuda"))
    torch.cuda.synchronize()
    expected_out, expected_final = SSDCombined(**constructor, backend="cake").run(
        x, dt, A, B, C, **arguments, return_final_states=True
    )
    torch.cuda.synchronize()
    _replay_matches(graph, out, final, expected_out, expected_final)


@requires_blackwell
def test_captured_workspace_survives_other_shapes(monkeypatch):
    """A workspace a graph recorded is pinned: sweeping more shapes than the
    registry keeps rotates the uncaptured entries out, leaves the captured
    one in place tensor for tensor, and its replays still match eager."""

    module = _module()
    monkeypatch.setattr(module.CakeSSDCombined, "workspace_capacity", 2)
    constructor, tensors, arguments = _inputs()
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    out = torch.empty(tensors[0].shape, dtype=torch.bfloat16, device="cuda")
    prepared = _prepare(runner, tensors, arguments, return_final_states=True, out=out)
    _, final = runner.run(*tensors, **arguments, out=out, return_final_states=True)
    expected_out, expected_final = out.clone(), final.clone()
    workspace = cake._workspaces.get(prepared.workspace_key).workspace
    pinned = {
        name: value
        for name, value in workspace.items()
        if isinstance(value, torch.Tensor)
    }
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.run(*tensors, **arguments, out=out, return_final_states=True)
    assert cake._workspaces.captured(prepared.workspace_key)

    for seqlen in (256, 384, 512):
        _, other_tensors, other_arguments = _inputs(seqlen=seqlen, seed=seqlen)
        runner.run(*other_tensors, **other_arguments)

    # Two uncaptured entries (the capacity) plus the pinned one.
    assert len(cake._workspaces) == 3
    assert cake._workspaces.captured(prepared.workspace_key)
    assert cake._workspaces.get(prepared.workspace_key).workspace is workspace
    for name, value in pinned.items():
        assert workspace[name] is value, name
    _replay_matches(graph, out, final, expected_out, expected_final)


@requires_blackwell
def test_prepared_eager_run_allocates_nothing():
    """After ``prepare``, an eager ``run`` on the shape with static ``out``
    is a pure launch (``_allocations_in_last_run == 0``) and returns the
    prepared static final-states buffer every time; an unprepared first call
    and an eager call without ``out`` do allocate."""

    constructor, tensors, arguments = _inputs()
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    out = torch.empty(tensors[0].shape, dtype=torch.bfloat16, device="cuda")

    runner.run(*tensors, **arguments)
    assert cake._allocations_in_last_run > 0

    _prepare(runner, tensors, arguments, return_final_states=True, out=out)
    out_run, final = runner.run(
        *tensors, **arguments, out=out, return_final_states=True
    )
    assert cake._allocations_in_last_run == 0
    assert out_run is out
    _, final_again = runner.run(
        *tensors, **arguments, out=out, return_final_states=True
    )
    assert final_again is final
    assert cake._allocations_in_last_run == 0
    # Eagerly without ``out`` (and under capture, from the graph pool): the
    # output is allocated.
    allocated, _ = runner.run(*tensors, **arguments, return_final_states=True)
    assert allocated is not out
    assert cake._allocations_in_last_run == 1


# ---------------------------------------------------------------------------
# CPU: the registry and the capture logic on a runner whose launcher is
# captured and whose stream-capture status is faked.


def test_workspace_registry_is_an_lru_over_uncaptured_entries():
    module = _module()
    registry = module._WorkspaceRegistry(2)
    a, b, c = ({"name": name} for name in "abc")

    assert registry.add("a", a).workspace is a
    registry.add("b", b)
    assert registry.keys() == ["a", "b"] and len(registry) == 2
    # A hit makes the entry the most recently used.
    assert registry.get("a").workspace is a
    assert registry.keys() == ["b", "a"]
    registry.add("c", c)
    assert registry.keys() == ["a", "c"]
    assert "b" not in registry and registry.get("b") is None
    assert not registry.captured("a") and not registry.captured("missing")
    with pytest.raises(KeyError):
        registry.add("a", {})
    with pytest.raises(KeyError):
        registry.pin("missing")
    for capacity in (0, -1, True, 2.0):
        with pytest.raises(ValueError, match="positive int"):
            module._WorkspaceRegistry(capacity)


def test_workspace_registry_pins_captured_and_prepared_entries():
    module = _module()
    registry = module._WorkspaceRegistry(1)
    captured = {"name": "captured"}
    registry.add("captured", captured)
    assert registry.pin("captured").captured
    assert registry.captured("captured")

    for key in ("x", "y", "z"):
        registry.add(key, {"name": key})

    # The pinned entry never counts against the capacity and is never
    # evicted; the unpinned entries rotate through the one slot.
    assert registry.keys() == ["captured", "z"]
    # A hit makes any entry, pinned or not, the most recently used.
    assert registry.get("captured").workspace is captured
    assert registry.keys() == ["z", "captured"]
    # A prepared entry is pinned the same way without being captured.
    assert registry.hold("z").prepared
    assert not registry.captured("z")
    registry.add("w", {})
    assert registry.keys() == ["z", "captured", "w"]
    registry.add("v", {})
    assert registry.keys() == ["z", "captured", "v"]


def _cpu_inputs(seqlen, nheads=8, ngroups=8):
    x = torch.empty((1, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((1, seqlen, ngroups, 128), dtype=torch.bfloat16)
    return x, dt, A, B, torch.empty_like(B)


def _cpu_runner(module, monkeypatch, calls, loads=None):
    """A real batched bf16 runner (8 heads, 8 groups) on CPU tensors: the
    device queries answer for a 148-SM sm_103a device, the launcher records
    its calls and the program loader builds nothing."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: "sm_103a")
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(module, "_sm_count", lambda _: 148)

    def launch(name, arch, **kwargs):
        calls.append((name, kwargs))
        module._LOADED_PROGRAMS.add((name, arch))

    def load(name, arch):
        if loads is not None:
            loads.append((name, arch))
        return SimpleNamespace(run=None)

    monkeypatch.setattr(module, "_launch_program", launch)
    monkeypatch.setattr(module, "_load_generated_program", load)
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda *_: SimpleNamespace(cuda_stream=0x1234),
    )
    return module.CakeSSDCombined(
        128,
        8,
        64,
        128,
        8,
        io_dtype=torch.bfloat16,
        state_dtype=torch.bfloat16,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=False,
        has_varlen=False,
        has_z=False,
        seq_idx_dtype=torch.int32,
    )


def test_prepare_resolves_the_shape_without_launching_without_gpu(monkeypatch):
    """``prepare`` materialises the workspace (with the static final-states
    buffer), loads the program exactly once, freezes the family override and
    launches nothing; the following ``run`` on the shape is a pure launch that
    binds the static buffer, loads nothing more and never reads the
    environment, while an unprepared shape still reads it per call."""

    module = _module()
    calls, loads = [], []
    runner = _cpu_runner(module, monkeypatch, calls, loads)
    x, dt, A, B, C = _cpu_inputs(256)  # 2 chunks x 8 heads = 16 tiles
    out = torch.empty_like(x)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")

    prepared = runner.prepare(
        x=x, dt=dt, A=A, B=B, C=C, return_final_states=True, out=out
    )

    assert calls == [] and runner.last_program_name is None
    assert prepared == module.PreparedSSDCombined(
        program_name="chunkpar_bf16_batched",
        workspace_key=module._WorkspaceKey(
            None, 1, 256, 2, 2, 1, torch.bfloat16, False, "always"
        ),
        grid=(16, 1, 1),
        preprocess_grid=(2, 1, 1),
    )
    # prepare loads the program but launches nothing (the first launch must
    # be eager), and pins the workspace.
    assert loads == [("chunkpar_bf16_batched", "sm_103a")]
    assert ("chunkpar_bf16_batched", "sm_103a") not in module._LOADED_PROGRAMS
    assert runner._workspaces.get(prepared.workspace_key).prepared
    workspace = runner._workspaces.get(prepared.workspace_key).workspace
    assert tuple(workspace["final_static"].shape) == (1, 8, 64, 128)
    assert workspace["final_static"].dtype == torch.bfloat16
    assert runner._chunk_parallel_buffers[None]["h_work"].shape[0] == 16

    # Frozen: an invalid override is never read for the prepared shape.
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "later")
    out_run, final = runner.run(x, dt, A, B, C, out=out, return_final_states=True)

    assert runner._allocations_in_last_run == 0
    assert out_run is out and final is workspace["final_static"]
    assert loads == [("chunkpar_bf16_batched", "sm_103a")]  # no new load
    name, launch = calls[-1]
    assert name == runner.last_program_name == "chunkpar_bf16_batched"
    assert launch["main_grid"] == (16, 1, 1)
    assert launch["main"]["final_states"] is final
    assert launch["main"]["out_map"] is out
    # Preparing the shape again re-reads the override and re-uses the buffer.
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    again = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, return_final_states=True)
    assert again.program_name == "exact_bf16_batched"
    assert again.workspace_key.chunk_parallel_mode == "never"
    assert len(runner._workspaces) == 2
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "later")
    runner.run(x, dt, A, B, C, out=out, return_final_states=True)
    assert calls[-1][0] == "exact_bf16_batched"
    assert runner._allocations_in_last_run == 0
    # An unprepared shape reads the environment on every call.
    with pytest.raises(ValueError, match=_CHUNK_PARALLEL_ENV):
        runner.run(*_cpu_inputs(512))


def test_capture_refusals_and_pinning_without_gpu(monkeypatch):
    """With the stream reported as capturing, ``run`` refuses by name the two
    things a graph cannot record -- the workspace of a shape this runner never
    ran or prepared, and a program never launched in this process -- and
    ``prepare`` refuses outright.  Everything else a captured call needs is
    legal: a warm shape allocates its output and final states (graph-pool
    storage, reused by every replay), reads the family override and pins its
    workspace; a fully prepared call with a static ``out`` launches and
    allocates nothing."""

    module = _module()
    calls = []
    runner = _cpu_runner(module, monkeypatch, calls)
    x, dt, A, B, C = _cpu_inputs(256)
    out = torch.empty_like(x)
    capturing = {"value": True}
    monkeypatch.setattr(module, "_stream_capturing", lambda _d: capturing["value"])

    # Never run nor prepared: refused before any workspace is built.
    with pytest.raises(CakeSSDCombinedCaptureError, match="has not run on this"):
        runner.run(x, dt, A, B, C, out=out)
    assert calls == [] and len(runner._workspaces) == 0
    with pytest.raises(CakeSSDCombinedCaptureError, match="outside CUDA-graph"):
        runner.prepare(x=x, dt=dt, A=A, B=B, C=C)

    # Prepared but never launched: the workspace exists, the program does
    # not count as launched until the first eager run.
    capturing["value"] = False
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, out=out)
    capturing["value"] = True
    with pytest.raises(CakeSSDCombinedCaptureError, match="not been launched"):
        runner.run(x, dt, A, B, C, out=out)
    assert calls == [] and len(runner._workspaces) == 1

    # One eager launch makes the program capturable; the prepared call with a
    # static ``out`` is then a pure launch under capture.
    capturing["value"] = False
    runner.run(x, dt, A, B, C, out=out)
    assert len(calls) == 1
    capturing["value"] = True
    out_run, final = runner.run(x, dt, A, B, C, out=out)
    assert out_run is out and final is runner._workspace["final_static"]
    assert len(calls) == 2 and calls[-1][0] == prepared.program_name
    assert runner._allocations_in_last_run == 0
    assert runner.is_captured(prepared)

    # A warm shape captured without ``prepare``: pool allocations are counted,
    # not refused; the workspace is pinned.  The family override is read per
    # call for an unprepared shape and is part of the workspace identity, so
    # a capture under a different override than any eager run is an unseen
    # shape and is refused.
    capturing["value"] = False
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    x2, dt2, A2, B2, C2 = _cpu_inputs(512)
    runner.run(x2, dt2, A2, B2, C2, return_final_states=False)
    (key2,) = [k for k in runner._workspaces.keys() if k != prepared.workspace_key]
    assert key2.chunk_parallel_mode == "never"
    assert not runner._workspaces.captured(key2)
    capturing["value"] = True
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
    with pytest.raises(CakeSSDCombinedCaptureError, match="has not run on this"):
        runner.run(x2, dt2, A2, B2, C2)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    out_c, final_c = runner.run(x2, dt2, A2, B2, C2)
    assert out_c is not out and tuple(out_c.shape) == tuple(x2.shape)
    assert final_c is not None
    assert runner._allocations_in_last_run >= 2, "output + final states"
    assert runner._workspaces.captured(key2)
    strided_dt = torch.empty((1, 512, 16), dtype=torch.float32)[:, :, ::2]
    runner.run(x2, strided_dt, A2, B2, C2, out=torch.empty_like(x2))
    assert calls[-1][1]["main"]["dt"].is_contiguous()
    assert runner._allocations_in_last_run >= 1, "the packed copy of dt"


def test_captured_chunk_parallel_buffers_survive_growth_without_gpu(monkeypatch):
    """The chunk-parallel buffers are per device and grow only.  A captured
    workspace pins the buffers its graph recorded: a later, larger eager
    call replaces the device's buffers without freeing the pinned ones, and
    the captured shape keeps binding them (the recorded addresses) while the
    grid-barrier words stay the device's single pair."""

    module = _module()
    calls = []
    runner = _cpu_runner(module, monkeypatch, calls)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
    x, dt, A, B, C = _cpu_inputs(256)  # 16 tiles
    out = torch.empty_like(x)
    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, out=out)
    # The program's first launch must be eager (kernel-handle resolution).
    runner.run(x, dt, A, B, C, out=out, return_final_states=False)
    capturing = {"value": True}
    monkeypatch.setattr(module, "_stream_capturing", lambda _d: capturing["value"])

    runner.run(x, dt, A, B, C, out=out, return_final_states=False)

    workspace = runner._workspaces.get(prepared.workspace_key).workspace
    pinned = workspace["chunk_parallel"]
    recorded = calls[-1][1]["main"]
    assert recorded["h_map"] is pinned["h_work"]
    assert recorded["s_work"] is pinned["s_work"]
    assert recorded["grid_barrier"] is pinned["grid_barrier"]
    assert pinned["h_work"].shape[0] == 16

    # Growth by an eager, larger call: new device buffers, same barrier.
    capturing["value"] = False
    runner.run(*_cpu_inputs(4096))  # 32 chunks x 8 heads = 256 tiles
    grown = calls[-1][1]["main"]
    assert grown["h_map"].shape[0] == 256
    assert grown["h_map"] is not pinned["h_work"]
    assert grown["grid_barrier"] is pinned["grid_barrier"]
    assert runner._chunk_parallel_buffers[None]["h_work"] is grown["h_map"]

    # The captured shape still binds the pinned buffers, under capture and
    # eagerly, without allocating.
    capturing["value"] = True
    runner.run(x, dt, A, B, C, out=out, return_final_states=False)
    assert calls[-1][1]["main"]["h_map"] is pinned["h_work"]
    assert runner._allocations_in_last_run == 0
    capturing["value"] = False
    runner.run(x, dt, A, B, C, out=out, return_final_states=False)
    assert calls[-1][1]["main"]["h_map"] is pinned["h_work"]
    assert runner._allocations_in_last_run == 0


def test_public_prepare_delegates_to_the_cake_runner_without_gpu():
    """``SSDCombined.prepare`` forwards every argument by keyword to the
    Cake runner and is not implemented for the CuTe backend."""

    runner = object.__new__(SSDCombined)
    runner._backend = "cake"

    class CakeRunner:
        def prepare(self, **kwargs):
            self.kwargs = kwargs
            return "prepared"

    runner._cake_runner = CakeRunner()
    x, dt, A, B, C = _cpu_inputs(128)
    out = torch.empty_like(x)

    assert (
        runner.prepare(x=x, dt=dt, A=A, B=B, C=C, return_final_states=True, out=out)
        == "prepared"
    )

    kwargs = runner._cake_runner.kwargs
    assert kwargs["x"] is x and kwargs["out"] is out
    assert kwargs["return_final_states"] is True
    assert set(kwargs) == {
        "x",
        "dt",
        "A",
        "B",
        "C",
        "D",
        "z",
        "dt_bias",
        "dt_softplus",
        "dt_limit",
        "initial_states",
        "seq_idx",
        "chunk_indices",
        "chunk_offsets",
        "seq_chunk_cumsum",
        "return_final_states",
        "num_seqs",
        "cu_seqlens",
        "out",
    }
    runner._backend = "cute"
    with pytest.raises(NotImplementedError, match="backend='cake'"):
        runner.prepare(x=x, dt=dt, A=A, B=B, C=C)
