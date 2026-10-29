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
# graph recorded is pinned.  GPU tests need SM100, SM103 or SM107; the registry and the
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
        (10, 7),
    )


requires_blackwell = pytest.mark.skipif(
    not _supported_device(), reason="Cake SSDCombined requires SM100, SM103 or SM107"
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


def _triple_arguments(lengths):
    """The packed-varlen ``seq_idx`` / ``chunk_indices`` / ``chunk_offsets``
    triple of ``lengths`` (chunk-128 logical segments) with the caller's
    ``seq_chunk_cumsum`` (exclusive prefix sum of per-sequence logical chunk
    counts) on the current device."""

    total = sum(lengths)
    seq_idx = torch.empty((1, total), dtype=torch.int32, device="cuda")
    chunk_indices, chunk_offsets, cumsum, start = [], [], [0], 0
    for sequence, length in enumerate(lengths):
        seq_idx[0, start : start + length] = sequence
        end = start + length
        cumsum.append(cumsum[-1] + ((-(-end // 128) - start // 128) if length else 0))
        start = end
    for chunk in range(-(-total // 128)):
        values = seq_idx[0, chunk * 128 : (chunk + 1) * 128]
        previous = torch.cat((values[:1] - 1, values[:-1]))
        for offset in (values != previous).nonzero(as_tuple=True)[0].tolist():
            chunk_indices.append(chunk)
            chunk_offsets.append(offset)

    def as_i32(values):
        return torch.tensor(values, dtype=torch.int32, device="cuda")

    return dict(
        seq_idx=seq_idx,
        chunk_indices=as_i32(chunk_indices),
        chunk_offsets=as_i32(chunk_offsets),
        seq_chunk_cumsum=as_i32(cumsum),
    )


def _inputs(
    *,
    varlen=False,
    lengths=(96, 160),
    batch=2,
    seqlen=128,
    nheads=8,
    ngroups=8,
    seed=7,
    form="cu_seqlens",
):
    """One Cake SSDCombined call on the current device -- bf16 x/B/C/z/D,
    f32 dt/A/dt_bias, softplus with a finite clamp, bf16 initial states --
    batched or packed varlen on the ``cu_seqlens`` form (``form``: or the
    seq_idx triple with the caller's trusted ``seq_chunk_cumsum``, or the
    triple deriving its table, ``triple_derived``).  Returns
    the constructor kwargs, the positional tensors and the keyword arguments."""

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
    if varlen and form == "triple_cumsum":
        arguments.update(_triple_arguments(lengths))
    elif varlen and form == "triple_derived":
        # The triple without a caller table: the runner derives it.
        arguments.update(_triple_arguments(lengths))
        del arguments["seq_chunk_cumsum"]
        arguments["num_seqs"] = len(lengths)
    elif varlen:
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
@pytest.mark.parametrize(
    "form",
    ("batched", "cu_seqlens", "triple_cumsum", "triple_derived", "chunk_parallel"),
)
def test_prepared_run_replays_bitwise_from_a_cuda_graph(monkeypatch, form):
    """``prepare`` then capture ``run`` with static inputs and ``out``: the
    captured call allocates nothing, never reads the environment, returns
    the prepared static final-states buffer, pins its workspace, and three
    replays reproduce the eager run bit for bit -- for the exact family in
    batched bf16-state form, for a packed-varlen ``cu_seqlens`` shape, for
    the seq_idx triple with the caller's trusted ``seq_chunk_cumsum`` (the
    fused single-launch program of the trusted-table route: ``prepare`` and the
    captured ``run`` resolve the same program from the same arguments), for
    the seq_idx triple deriving a runner-owned table (the capability's
    program: ``exactmeta`` on SM100, two launches; ``exactone`` on SM103 and
    SM107, one launch whose CTA prologue derives the table into the workspace scratch
    the graph replays against) and for the chunk-parallel program forced
    through the override.

    Chunk-parallel caveat: that program needs every CTA of its grid (one
    per SM) co-resident for its grid barrier, and the barrier words are
    shared per device.  A replay is stream-ordered like an eager launch, so
    it must not run concurrently with another chunk-parallel launch on the
    device (two streams); the same holds for two eager calls today."""

    if form == "chunk_parallel":
        monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "always")
        constructor, tensors, arguments = _inputs(batch=2, seqlen=1024)
    else:
        constructor, tensors, arguments = _inputs(
            varlen=form in ("cu_seqlens", "triple_cumsum", "triple_derived"),
            form=form,
        )
    runner = SSDCombined(**constructor, backend="cake")
    cake = runner._cake_runner
    out = torch.empty(tensors[0].shape, dtype=torch.bfloat16, device="cuda")

    prepared = _prepare(runner, tensors, arguments, return_final_states=True, out=out)

    assert isinstance(prepared, PreparedSSDCombined)
    assert not cake.is_captured(prepared)
    if form == "chunk_parallel":
        family = "chunkpar"
    elif form == "triple_derived":
        family = (
            "exactmeta" if torch.cuda.get_device_capability() == (10, 0) else "exactone"
        )
    else:
        family = "exactfused"
    mode = "batched" if form in ("batched", "chunk_parallel") else "varlen"
    table_form = ""
    if family == "exactone":
        # SM107 picks the one-launch prologue form by the launch grid.
        table_form = importlib.import_module(
            "flashinfer.mamba.cake_ssd_combined"
        ).exactone_table_variant(
            capability=torch.cuda.get_device_capability(),
            variant="",
            state_dtype=torch.bfloat16,
            grid=prepared.grid[0],
        )
    assert prepared.program_name == f"{family}_bf16_{mode}" + (
        f"_{table_form}" if table_form else ""
    )
    if family in ("exactfused", "exactone"):
        # The fused program is prepared only as a single launch (the
        # one-launch program always is).
        assert prepared.preprocess_grid == (0, 1, 1)
    elif family == "exactmeta":
        assert prepared.preprocess_grid[0] >= 1
    assert prepared.workspace_key.caller_cumsum == (form == "triple_cumsum")
    assert prepared.workspace_key.from_cu_seqlens == (form == "cu_seqlens")
    assert prepared.workspace_key.publish_table is False
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
    assert cake.last_program_name == "exactfused_bf16_batched"


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


def _cpu_runner(
    module,
    monkeypatch,
    calls,
    loads=None,
    *,
    has_varlen=False,
    arch="sm_103a",
    seq_idx_dtype=torch.int32,
):
    """A real bf16 runner (8 heads, 8 groups; batched unless ``has_varlen``)
    on CPU tensors: the device queries answer for a 148-SM device of ``arch``
    (default sm_103a), the launcher records its calls and the program loader
    builds nothing."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: arch)
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
        has_varlen=has_varlen,
        has_z=False,
        seq_idx_dtype=seq_idx_dtype,
    )


_ROUTE_ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# The architectures routing the seq_idx triple form deriving its table
# through the light metadata stage (``exactmeta``): SM100 alone (SM103 and
# SM107 run the one-launch ``exactone`` render).
_LIGHT_STAGE_ARCHES = ("sm_100a",)


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
def test_prepare_and_run_resolve_the_same_program_for_a_trusted_cumsum_without_gpu(
    monkeypatch, arch
):
    """The seq_idx triple form with a caller ``seq_chunk_cumsum`` the
    preprocess does not rewrite is one workspace key (``caller_cumsum``) for
    ``prepare`` and ``run``: ``prepare`` reads the table as trusted (the
    ``run`` default), freezes the fused single-launch program for the shape
    and the following ``run`` is a pure launch of that program; a ``run``
    that asks the preprocess to rewrite the table is another shape with the
    publishing program of the capability (``exactmeta`` on SM100, the
    two-kernel program on SM103); the ``cu_seqlens`` form with a caller
    buffer is still prepared as the publishing call (its only reading)."""

    module = _module()
    calls, loads = [], []
    runner = _cpu_runner(module, monkeypatch, calls, loads, has_varlen=True, arch=arch)
    seqlen, num_seqs = 300, 2  # 2 sequences x 8 heads, 3 packed chunks
    x, dt, A, B, C = _cpu_inputs(seqlen)
    out = torch.empty_like(x)
    triple = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
        "num_seqs": num_seqs,
    }
    table = torch.tensor([0, 1, 3], dtype=torch.int32)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")

    prepared = runner.prepare(
        x=x, dt=dt, A=A, B=B, C=C, **triple, seq_chunk_cumsum=table, out=out
    )

    assert calls == [] and runner.last_program_name is None
    assert prepared.program_name == "exactfused_bf16_varlen"
    assert prepared.preprocess_grid == (0, 1, 1)
    assert prepared.workspace_key == module._WorkspaceKey(
        None, 1, seqlen, 3, 3, num_seqs, torch.bfloat16, False, True, False, "never"
    )
    assert loads == [("exactfused_bf16_varlen", arch)]
    assert runner._workspaces.get(prepared.workspace_key).prepared

    # The run of the same call: the frozen program, a pure launch, the
    # caller's tables bound to the single main kernel.
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "invalid-after-prepare")
    out_run, _final = runner.run(
        x, dt, A, B, C, **triple, seq_chunk_cumsum=table, out=out
    )
    name, launch = calls[-1]
    assert out_run is out
    assert name == runner.last_program_name == prepared.program_name
    assert runner._allocations_in_last_run == 0
    assert loads == [("exactfused_bf16_varlen", arch)]  # no new load
    assert launch["preprocess_grid"] == (0, 1, 1)
    assert launch["preprocess"]["write_seq_chunk_cumsum"] == 0
    assert launch["main"]["metadata_from_cu_seqlens"] == 0
    assert launch["main"]["seq_chunk_cumsum"] is table
    assert launch["main"]["chunk_indices"] is triple["chunk_indices"]
    assert len(runner._workspaces) == 1

    # Asking the publishing stage to rewrite the table is another call shape
    # under its own key (unprepared, so it reads the environment): on SM100
    # the exactmeta program (the light metadata kernel publishing into the
    # caller's buffer, then the same fused main kernel), on SM103 and SM107
    # the two-kernel program (the segment preprocess publishes).
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")
    runner.run(
        x, dt, A, B, C, **triple, seq_chunk_cumsum=table, update_seq_chunk_cumsum=True
    )
    name, launch = calls[-1]
    assert name == runner.last_program_name
    if arch in _LIGHT_STAGE_ARCHES:
        assert name == "exactmeta_bf16_varlen"
        assert launch["preprocess"]["sequence_offsets"] is table
        assert launch["preprocess_grid"] == (1, 1, 1)
    else:
        family = module.two_kernel_family(
            mode_varlen=True, num_sequences=num_seqs, seqlen=seqlen
        )
        assert module._PROGRAMS[name].family == family
        assert launch["preprocess"]["seq_chunk_cumsum"] is table
        assert launch["preprocess"]["write_seq_chunk_cumsum"] == 1
        assert launch["preprocess_grid"][0] >= 1
    assert launch["main"]["seq_chunk_cumsum"] is table
    assert (
        module._WorkspaceKey(
            None, 1, seqlen, 3, 3, num_seqs, torch.bfloat16, False, False, True, "never"
        )
        in runner._workspaces
    )
    assert len(runner._workspaces) == 2

    # cu_seqlens form with a caller buffer: prepared as the publishing call
    # (the fused program with its preprocess, two launches), never as a
    # trusted table.
    cu_seqlens = torch.tensor([0, 150, seqlen], dtype=torch.int32)
    buffer = torch.empty(num_seqs + 1, dtype=torch.int32)
    prepared_cu = runner.prepare(
        x=x, dt=dt, A=A, B=B, C=C, cu_seqlens=cu_seqlens, seq_chunk_cumsum=buffer
    )
    assert prepared_cu.program_name == "exactfused_bf16_varlen"
    assert prepared_cu.preprocess_grid[0] >= 1
    assert prepared_cu.workspace_key.from_cu_seqlens
    assert not prepared_cu.workspace_key.caller_cumsum
    runner.run(
        x,
        dt,
        A,
        B,
        C,
        cu_seqlens=cu_seqlens,
        seq_chunk_cumsum=buffer,
        update_seq_chunk_cumsum=True,
        out=out,
    )
    name, launch = calls[-1]
    assert name == prepared_cu.program_name
    assert launch["preprocess_grid"] == prepared_cu.preprocess_grid
    assert launch["preprocess"]["seq_chunk_cumsum"] is buffer
    assert runner._allocations_in_last_run == 0


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
@pytest.mark.parametrize("form", ("cu_seqlens", "triple_i32", "triple_i64", "trusted"))
def test_prepare_and_run_resolve_the_same_program_by_capability_without_gpu(
    monkeypatch, arch, form
):
    """``prepare`` and ``run`` of one call resolve the same program, workspace
    key and launch grids for every packed-varlen form on both capabilities:
    the ``cu_seqlens`` form and the trusted-table triple form (the fused
    program, one launch), and the triple form deriving its table with int32
    or int64 ``seq_idx`` (``exactmeta`` on SM100, the table published ahead
    of the main kernel; ``exactone`` on SM103 and SM107, one launch with the
    table derived in the main kernel's prologue)."""

    module = _module()
    seq_idx_dtype = torch.int64 if form == "triple_i64" else torch.int32
    calls, loads = [], []
    runner = _cpu_runner(
        module,
        monkeypatch,
        calls,
        loads,
        has_varlen=True,
        arch=arch,
        seq_idx_dtype=seq_idx_dtype,
    )
    seqlen, num_seqs = 300, 2
    x, dt, A, B, C = _cpu_inputs(seqlen)
    out = torch.empty_like(x)
    if form == "cu_seqlens":
        kwargs = {"cu_seqlens": torch.tensor([0, 150, seqlen], dtype=torch.int32)}
    else:
        kwargs = {
            "seq_idx": torch.zeros((1, seqlen), dtype=seq_idx_dtype),
            "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
            "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
            "num_seqs": num_seqs,
        }
        if form == "trusted":
            kwargs["seq_chunk_cumsum"] = torch.tensor([0, 1, 3], dtype=torch.int32)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "never")

    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs, out=out)
    assert calls == [] and runner.last_program_name is None
    family = module._PROGRAMS[prepared.program_name].family
    if form in ("cu_seqlens", "trusted"):
        assert family == "exactfused" and prepared.preprocess_grid == (0, 1, 1)
    elif arch in _LIGHT_STAGE_ARCHES:
        assert family == "exactmeta", (arch, form, prepared.program_name)
        assert prepared.preprocess_grid[0] >= 1
    else:
        assert family == "exactone", (arch, form, prepared.program_name)
        assert prepared.preprocess_grid == (0, 1, 1)
    assert prepared.workspace_key.publish_table is False
    assert loads == [(prepared.program_name, arch)]

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "invalid-after-prepare")
    out_run, _final = runner.run(x, dt, A, B, C, **kwargs, out=out)
    name, launch = calls[-1]
    assert out_run is out
    assert name == runner.last_program_name == prepared.program_name
    assert launch["preprocess_grid"] == prepared.preprocess_grid
    assert launch["main_grid"] == prepared.grid
    if family == "exactone":
        # One table row per CTA of the prepared grid, int32.
        scratch = launch["main"]["seq_chunk_cumsum"]
        assert scratch.numel() == prepared.grid[0] * (num_seqs + 1)
        assert launch["main"]["metadata_from_cu_seqlens"] == (
            module._TABLE_IN_KERNEL_I64
            if form == "triple_i64"
            else module._TABLE_IN_KERNEL_I32
        )
    assert loads == [(prepared.program_name, arch)]  # no new load
    assert runner._allocations_in_last_run == 0
    assert len(runner._workspaces) == 1
    assert prepared.workspace_key in runner._workspaces


def test_prepare_and_run_refuse_other_capabilities_alike_without_gpu(monkeypatch):
    """Neither ``prepare`` nor ``run`` guesses a route for a device that is
    not SM100, SM103 or SM107: both raise ``_target_arch``'s ``ValueError``."""

    module = _module()
    target_arch = module._target_arch
    calls, loads = [], []
    runner = _cpu_runner(module, monkeypatch, calls, loads, has_varlen=True)
    monkeypatch.setattr(module, "_target_arch", target_arch.__wrapped__)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *_: (9, 0))
    seqlen = 300
    x, dt, A, B, C = _cpu_inputs(seqlen)
    triple = {
        "seq_idx": torch.zeros((1, seqlen), dtype=torch.int32),
        "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
        "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
        "num_seqs": 2,
    }
    with pytest.raises(ValueError, match="requires SM100, SM103 or SM107"):
        runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **triple)
    with pytest.raises(ValueError, match="requires SM100, SM103 or SM107"):
        runner.run(x, dt, A, B, C, **triple)
    assert calls == [] and loads == []


# ---------------------------------------------------------------------------
# CPU: the two-kernel family's tail-wave pair program against the other route
# rules -- the chunk-parallel override, the fused window with its capability
# split of the derived triple form, and the ring / noring choice.

_TAIL_SM_COUNTS = (148, 152)
# 64 sequences x 8 heads = 512 (sequence, head) items of 16 chunks each: the
# pair program's measured row (4 sequences x 128 heads on the production head
# count) at this runner's head count.  Above one item per SM on both devices,
# the balanced persistent grid is 128 CTAs (makespan 4 items) and the pair
# makespan ceil(512 / 74) / 2 = 3.5 beats it.
_TAIL_NUM_SEQS, _TAIL_SEQLEN = 64, 2048
_TAIL_FORMS = ("batched", "cu_seqlens", "trusted", "triple_i32", "triple_i64")


def _tail_call(form, *, num_seqs=_TAIL_NUM_SEQS, seqlen=_TAIL_SEQLEN):
    """Inputs and keyword arguments of a 512-item x 16-chunk call on CPU
    tensors: batched, or packed varlen (``num_seqs`` sequences of ``seqlen``
    tokens) on the ``cu_seqlens`` form, the seq_idx triple with the caller's
    trusted table (``trusted``) or the triple deriving its table
    (``triple_i32`` / ``triple_i64``)."""

    if form == "batched":
        x = torch.empty((num_seqs, seqlen, 8, 64), dtype=torch.bfloat16)
        dt = torch.empty((num_seqs, seqlen, 8), dtype=torch.float32)
        B = torch.empty((num_seqs, seqlen, 8, 128), dtype=torch.bfloat16)
        A = torch.empty((8,), dtype=torch.float32)
        return (x, dt, A, B, torch.empty_like(B)), {}
    total = num_seqs * seqlen
    tensors = _cpu_inputs(total)
    if form == "cu_seqlens":
        cu_seqlens = torch.arange(0, total + 1, seqlen, dtype=torch.int32)
        return tensors, {"cu_seqlens": cu_seqlens}
    dtype = torch.int64 if form == "triple_i64" else torch.int32
    chunks = total // 128  # sequences aligned to chunk boundaries
    ids = torch.arange(num_seqs, dtype=dtype)
    kwargs = {
        "seq_idx": torch.repeat_interleave(ids, seqlen).reshape(1, total),
        "chunk_indices": torch.arange(chunks, dtype=torch.int32),
        "chunk_offsets": torch.zeros(chunks, dtype=torch.int32),
        "num_seqs": num_seqs,
    }
    if form == "trusted":
        kwargs["seq_chunk_cumsum"] = torch.arange(
            0, num_seqs + 1, dtype=torch.int32
        ) * (seqlen // 128)
    return tensors, kwargs


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
@pytest.mark.parametrize("sm_count", _TAIL_SM_COUNTS)
@pytest.mark.parametrize("form", _TAIL_FORMS)
def test_prepare_and_run_resolve_the_tail_pair_program_without_gpu(
    monkeypatch, arch, sm_count, form
):
    """``prepare`` and ``run`` of a two-kernel exact-scan call above one item
    per SM with 16 chunks per item resolve the ``exactpre_*_tail`` pair
    program -- the even grid ``2 * (sm_count // 2)``, the 2-CTA cluster shape,
    the two-kernel preprocess ahead of it -- on 148 and 152 SMs, both
    capabilities, batched and every packed-varlen form, from the one route
    path.  The capability split of the derived triple form (``exactmeta`` /
    ``exactone``) acts inside the fused window only, so on this shape both
    capabilities reach the pair program."""

    module = _module()
    seq_idx_dtype = torch.int64 if form == "triple_i64" else torch.int32
    varlen = form != "batched"
    calls, loads = [], []
    runner = _cpu_runner(
        module,
        monkeypatch,
        calls,
        loads,
        has_varlen=varlen,
        arch=arch,
        seq_idx_dtype=seq_idx_dtype,
    )
    monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
    (x, dt, A, B, C), kwargs = _tail_call(form)
    out = torch.empty_like(x)
    items, chunks = _TAIL_NUM_SEQS * 8, _TAIL_SEQLEN // 128
    total_tokens = _TAIL_NUM_SEQS * _TAIL_SEQLEN
    # The route's quantities for this shape: outside the fused window, the
    # pair rule met, no noring condition (that is one-chunk geometry).
    assert not module.fused_selected(
        nheads=8,
        num_sequences=_TAIL_NUM_SEQS,
        nchunks=chunks,
        mode_varlen=varlen,
        total_tokens=total_tokens,
    )
    assert (
        module.host_chunks_per_item(
            mode_varlen=varlen,
            num_sequences=_TAIL_NUM_SEQS,
            seqlen=total_tokens if varlen else _TAIL_SEQLEN,
        )
        == chunks
    )
    assert module.tail_selected(
        total_work=items, chunks_per_item=chunks, sm_count=sm_count
    )
    assert (
        module._exact_scan_variant(
            mode_varlen=varlen,
            has_z=False,
            nheads=8,
            num_sequences=_TAIL_NUM_SEQS,
            total_tokens=total_tokens,
            batch=1 if varlen else _TAIL_NUM_SEQS,
            seqlen=total_tokens if varlen else _TAIL_SEQLEN,
            state_dtype=torch.bfloat16,
        )
        == ""
    )

    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs, out=out)

    assert calls == [] and runner.last_program_name is None
    mode = "varlen" if varlen else "batched"
    assert prepared.program_name == f"exactpre_bf16_{mode}_tail"
    program = module._PROGRAMS[prepared.program_name]
    assert program.family == "exactpre" and program.variant == "tail"
    assert program.cluster_dims == (2, 1, 1)
    assert prepared.grid == (module._tail_grid_size(sm_count=sm_count), 1, 1)
    assert prepared.grid == (sm_count, 1, 1) and prepared.grid[0] % 2 == 0
    # the two-kernel preprocess publishes delta / cumsum ahead of the scan
    assert prepared.preprocess_grid[0] >= 1
    assert prepared.workspace_key.from_cu_seqlens == (form == "cu_seqlens")
    assert prepared.workspace_key.caller_cumsum == (form == "trusted")
    assert prepared.workspace_key.publish_table is False
    assert loads == [(prepared.program_name, arch)]

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "invalid-after-prepare")
    out_run, _final = runner.run(x, dt, A, B, C, **kwargs, out=out)
    name, launch = calls[-1]
    assert out_run is out
    assert name == runner.last_program_name == prepared.program_name
    assert launch["main_grid"] == prepared.grid
    assert launch["preprocess_grid"] == prepared.preprocess_grid
    # the two-kernel main kernel reads the published table
    assert launch["main"]["metadata_from_cu_seqlens"] == module._TABLE_PUBLISHED
    assert loads == [(prepared.program_name, arch)]  # no new load
    assert runner._allocations_in_last_run == 0
    assert len(runner._workspaces) == 1
    assert prepared.workspace_key in runner._workspaces


@pytest.mark.parametrize("sm_count", _TAIL_SM_COUNTS)
def test_tail_route_precedes_the_noring_variant_without_gpu(monkeypatch, sm_count):
    """The variant is decided in order -- chunk-parallel, fused, the pair
    rule, then the ring / noring choice -- so where the pair rule fires the
    ``_tail`` program wins over a ``noring`` reading of the same shape; with
    the pair rule off, the same shape takes the variant function's answer on
    the balanced persistent grid, from ``prepare`` and ``run`` alike.  The
    two readings never coincide on a real shape (``noring`` is one-chunk
    geometry, the pair rule needs 16 chunks), so each is forced here.  The
    ``_noring`` programs ship only where the variant rule can select them
    (one-chunk items of the retained ``exact`` family, never the two-kernel
    family), so the second arm takes the same 512-item x 16-chunk call in its
    packed-varlen form and also forces the retained family."""

    module = _module()
    rule = module.tail_selected
    for pair_rule, form, family, expected in (
        (True, "batched", None, "exactpre_bf16_batched_tail"),
        (False, "cu_seqlens", "exact", "exact_bf16_varlen_noring"),
    ):
        calls, loads = [], []
        varlen = form != "batched"
        runner = _cpu_runner(module, monkeypatch, calls, loads, has_varlen=varlen)
        monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
        monkeypatch.setattr(module, "_exact_scan_variant", lambda **_: "noring")
        if family is not None:
            monkeypatch.setattr(
                module, "two_kernel_family", lambda _family=family, **_: _family
            )
        monkeypatch.setattr(
            module,
            "tail_selected",
            lambda _on=pair_rule, **kwargs: _on and rule(**kwargs),
        )
        (x, dt, A, B, C), kwargs = _tail_call(form)
        out = torch.empty_like(x)
        grid = (
            module._tail_grid_size(sm_count=sm_count)
            if pair_rule
            else module._persistent_grid_size(total_work=512, sm_count=sm_count)
        )

        prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, out=out, **kwargs)

        assert prepared.program_name == expected, (pair_rule, prepared.program_name)
        assert prepared.grid == (grid, 1, 1)
        runner.run(x, dt, A, B, C, out=out, **kwargs)
        name, launch = calls[-1]
        assert name == runner.last_program_name == prepared.program_name
        assert launch["main_grid"] == prepared.grid
        assert runner._allocations_in_last_run == 0
    # Disjoint on real shapes: every noring window lies below the pair rule's
    # chunk floor.
    assert module._TAIL_MIN_CHUNKS == 16
    _low, high = module._EXACT_NORING_BATCHED_SEQLEN
    assert -(-(high - 1) // 128) < module._TAIL_MIN_CHUNKS
    _low, high = module._EXACT_NORING_VARLEN_TOKENS
    assert -(-(high - 1) // 128) < module._TAIL_MIN_CHUNKS


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
@pytest.mark.parametrize("sm_count", _TAIL_SM_COUNTS)
def test_fused_window_programs_never_take_the_tail_variant_without_gpu(
    monkeypatch, arch, sm_count
):
    """The fused window and the pair rule are disjoint on both capabilities
    and both devices: no shape is both ``fused_selected`` and
    ``tail_selected`` (the window caps batched calls at 16 chunks and 2048
    chunk-items and varlen calls at 256 items and 512 chunk-items; the pair
    rule needs more items than SMs and 16 chunks per item), so the fused
    programs -- ``exactfused``, and the derived triple form's ``exactmeta``
    on SM100 / ``exactone`` on SM103 and SM107 -- ship no pair program and never launch
    on the even grid.  Asserted over a grid of shapes on the route functions,
    on the program table, and through ``prepare`` / ``run`` for every
    packed-varlen form inside the window, including the publishing form that
    keeps the two-kernel program on SM103."""

    module = _module()
    rule, fused = module.tail_selected, module.fused_selected
    for nheads in (8, 128):
        for num_sequences in (1, 2, 4, 8, 16, 19, 32, 37, 64, 128, 256):
            items = num_sequences * nheads
            for chunks in (1, 2, 4, 8, 15, 16, 17, 32, 64):
                tail = rule(total_work=items, chunks_per_item=chunks, sm_count=sm_count)
                in_window = any(
                    fused(
                        nheads=nheads,
                        num_sequences=num_sequences,
                        nchunks=chunks,
                        mode_varlen=mode_varlen,
                        total_tokens=num_sequences * chunks * 128,
                    )
                    for mode_varlen in (False, True)
                )
                assert not (tail and in_window), (nheads, num_sequences, chunks)
    for name, program in module._PROGRAMS.items():
        if program.family in ("exactfused", "exactmeta", "exactone"):
            assert program.variant != "tail" and not name.endswith("_tail"), name
    assert set(module._VARIANT_CLUSTER_DIMS) == {("exactpre", "tail")}

    # Inside the window (2 sequences x 8 heads, 3 packed chunks): the program
    # of every form by capability, none on the pair program's grid.
    seqlen, num_seqs = 300, 2
    derived = "exactmeta" if arch in _LIGHT_STAGE_ARCHES else "exactone"
    two_kernel = module.two_kernel_family(
        mode_varlen=True, num_sequences=num_seqs, seqlen=seqlen
    )
    families = {
        "cu_seqlens": "exactfused",
        "trusted": "exactfused",
        "triple_i32": derived,
        "triple_i64": derived,
        "publish": "exactmeta" if arch in _LIGHT_STAGE_ARCHES else two_kernel,
    }
    assert fused(
        nheads=8,
        num_sequences=num_seqs,
        nchunks=3,
        mode_varlen=True,
        total_tokens=seqlen,
    )
    assert not rule(
        total_work=num_seqs * 8,
        chunks_per_item=module.host_chunks_per_item(
            mode_varlen=True, num_sequences=num_seqs, seqlen=seqlen
        ),
        sm_count=sm_count,
    )
    grid = module._persistent_grid_size(total_work=num_seqs * 8, sm_count=sm_count)
    for form, family in families.items():
        seq_idx_dtype = torch.int64 if form == "triple_i64" else torch.int32
        calls, loads = [], []
        runner = _cpu_runner(
            module,
            monkeypatch,
            calls,
            loads,
            has_varlen=True,
            arch=arch,
            seq_idx_dtype=seq_idx_dtype,
        )
        monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
        x, dt, A, B, C = _cpu_inputs(seqlen)
        out = torch.empty_like(x)
        if form == "cu_seqlens":
            kwargs = {"cu_seqlens": torch.tensor([0, 150, seqlen], dtype=torch.int32)}
        else:
            kwargs = {
                "seq_idx": torch.zeros((1, seqlen), dtype=seq_idx_dtype),
                "chunk_indices": torch.tensor([0, 1, 2], dtype=torch.int32),
                "chunk_offsets": torch.tensor([0, 0, 0], dtype=torch.int32),
                "num_seqs": num_seqs,
            }
            if form == "trusted":
                kwargs["seq_chunk_cumsum"] = torch.tensor([0, 1, 3], dtype=torch.int32)
            elif form == "publish":
                kwargs["seq_chunk_cumsum"] = torch.full(
                    (num_seqs + 1,), -1, dtype=torch.int32
                )
                kwargs["update_seq_chunk_cumsum"] = True
        if form == "publish":
            # ``prepare`` reads a caller table as trusted; the publishing
            # form is a ``run`` shape of its own (``publish_table``): its
            # first call materialises the workspace, the second resolves the
            # same program from the same key and allocates nothing.
            runner.run(x, dt, A, B, C, **kwargs, out=out)
            first, _launch = calls[-1]
            runner.run(x, dt, A, B, C, **kwargs, out=out, return_final_states=False)
            name, launch = calls[-1]
            assert name == first
            (key,) = runner._workspaces.keys()
            assert runner._workspaces.get(key).workspace["program"] == name
            assert len(runner._workspaces) == 1
        else:
            prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, **kwargs, out=out)
            runner.run(x, dt, A, B, C, **kwargs, out=out)
            name, launch = calls[-1]
            assert name == prepared.program_name
            assert launch["main_grid"] == prepared.grid
            assert launch["preprocess_grid"] == prepared.preprocess_grid
            assert prepared.workspace_key.publish_table is False
            assert runner._allocations_in_last_run == 0
        program = module._PROGRAMS[name]
        assert program.family == family, (arch, form, name)
        # SM107's one-launch program takes the prologue form of its grid.
        table_form = (
            module.exactone_table_variant(
                capability=module._ARCH_CAPABILITIES[arch],
                variant="",
                state_dtype=torch.bfloat16,
                grid=grid,
            )
            if family == "exactone"
            else ""
        )
        assert program.variant == table_form and program.cluster_dims == (1, 1, 1)
        assert not name.endswith("_tail")
        assert launch["main_grid"] == (grid, 1, 1) == (16, 1, 1)
        if family == "exactone":
            # one launch; its private table has one row per CTA of the
            # persistent grid, never of the pair program's grid
            assert launch["preprocess_grid"] == (0, 1, 1)
            scratch = launch["main"]["seq_chunk_cumsum"]
            assert scratch.numel() == grid * (num_seqs + 1)
        else:
            assert launch["preprocess_grid"][0] >= (family != "exactfused")


@pytest.mark.parametrize("arch", _ROUTE_ARCHES)
@pytest.mark.parametrize("sm_count", _TAIL_SM_COUNTS)
@pytest.mark.parametrize("mode", ("always", "three_phase", "never"))
def test_chunk_parallel_override_decides_before_the_tail_route_without_gpu(
    monkeypatch, arch, sm_count, mode
):
    """``FLASHINFER_CAKE_SSD_CHUNK_PARALLEL`` is read first: on the pair
    program's shape ``always`` / ``three_phase`` run the forced chunk-parallel
    program (one CTA per SM, no variant, no cluster shape) and ``never`` only
    disables the chunk-parallel rule -- the exact-scan route still reaches
    the pair program.  ``prepare`` freezes the reading and ``run`` launches
    it, allocating nothing."""

    module = _module()
    calls, loads = [], []
    runner = _cpu_runner(module, monkeypatch, calls, loads, arch=arch)
    monkeypatch.setattr(module, "_sm_count", lambda _: sm_count)
    (x, dt, A, B, C), _ = _tail_call("batched")
    out = torch.empty_like(x)
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, mode)
    expected = {
        "always": "chunkpar_bf16_batched",
        "three_phase": "chunkpar3_bf16_batched",
        "never": "exactpre_bf16_batched_tail",
    }[mode]

    prepared = runner.prepare(x=x, dt=dt, A=A, B=B, C=C, out=out)

    assert calls == [] and runner.last_program_name is None
    assert prepared.program_name == expected
    assert prepared.workspace_key.chunk_parallel_mode == mode
    program = module._PROGRAMS[expected]
    if mode == "never":
        assert program.variant == "tail" and program.cluster_dims == (2, 1, 1)
    else:
        assert program.variant == "" and program.cluster_dims == (1, 1, 1)
    # the pair program's even grid and the chunk-parallel one-CTA-per-SM grid
    # (8192 tiles here) both fill the device
    assert prepared.grid == (sm_count, 1, 1)
    assert loads == [(expected, arch)]

    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "invalid-after-prepare")
    runner.run(x, dt, A, B, C, out=out)
    name, launch = calls[-1]
    assert name == runner.last_program_name == prepared.program_name
    assert launch["main_grid"] == prepared.grid
    assert launch["preprocess_grid"] == prepared.preprocess_grid
    assert loads == [(expected, arch)]  # no new load
    assert runner._allocations_in_last_run == 0
    assert len(runner._workspaces) == 1


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
            None, 1, 256, 2, 2, 1, torch.bfloat16, False, False, False, "always"
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
    assert again.program_name == "exactfused_bf16_batched"
    assert again.workspace_key.chunk_parallel_mode == "never"
    assert len(runner._workspaces) == 2
    monkeypatch.setenv(_CHUNK_PARALLEL_ENV, "later")
    runner.run(x, dt, A, B, C, out=out, return_final_states=True)
    assert calls[-1][0] == "exactfused_bf16_batched"
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
