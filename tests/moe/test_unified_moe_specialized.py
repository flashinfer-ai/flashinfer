"""The ``specialized`` backend of the unified MoE API, for every registered kernel.

``SpecializedMoEConfig`` holds the MoE kernels generated for one served
configuration each (one subpackage of
``flashinfer/fused_moe/backends/specialized/kernels/`` per kernel). This file
runs the backend's contract on every kernel that has a spec in
``tests/moe/specialized_kernels/`` (``SPEC``: the served configuration, its
weights and inputs, guard cases, tolerances and the kernel's own checks):

* registration, both guards, the kill switch (direct and through ``MoELayer``),
  stats and AOT on any host (meta tensors, pinned devices);
* on the kernel's device: ``MoELayer`` candidates, the lazy build, numerics
  against the kernel's fp32 reference whichever backend the layer runs, CUDA
  graphs (selected eagerly, unbuilt, unselected, after ``reset_winner()`` and
  after a failed selection), ``auto`` (the default list) taking the kernel first
  inside its envelope and autotune timing every backend, the kernel's own
  checks, and a separate performance test.

The framework's kernel-independent CPU checks are in
``tests/cpu/moe/test_unified_moe_specialized.py``. A new kernel adds its spec,
not lines here.
"""

from __future__ import annotations

import json
import sys
import warnings
from importlib import resources
from statistics import median
from types import SimpleNamespace

import pytest
import torch

from flashinfer.autotuner import AutoTuner, autotune
from flashinfer.fused_moe import (
    BackendOptions,
    MoEActivationPack,
    MoELayer,
    SpecializedMoEConfig,
)
from flashinfer.fused_moe.backends import specialized
from flashinfer.fused_moe.backends.specialized.kernel import KILL_SWITCH_ENV
from flashinfer.fused_moe.backends.specialized.runner import SpecializedMoERunner
from flashinfer.utils import get_compute_capability

from tests.moe.specialized_kernels import load_specs

_KEY = "specialized"
SPECS = load_specs()
NAMES = list(SPECS)
_SPECIALIZED_ONLY = BackendOptions(candidates=(SpecializedMoEConfig(),))


def _cases(attr):
    """(kernel, case id) pairs of a spec dict / tuple attribute, as pytest params."""
    out = []
    for name, spec in SPECS.items():
        cases = getattr(spec, attr)
        keys = list(cases) if isinstance(cases, dict) else [c[0] for c in cases]
        out += [pytest.param(name, key, id=f"{name}-{key}") for key in keys]
    return out


@pytest.fixture
def spec(name, monkeypatch):
    """The kernel's spec, with its process-wide state reset around the test."""
    s = SPECS[name]
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    s.kernel_module._reset_for_tests()
    yield s
    s.kernel_module._reset_for_tests()


def _needs_kernel(spec):
    if not spec.available():
        pytest.skip(spec.skip_reason)


def _meta(shape, dtype):
    return torch.empty(shape, dtype=dtype, device="meta")


def _runner(layer):
    (runner,) = [r for r in layer.runners if r.backend_key == _KEY]
    return runner


def _sibling_only(spec):
    return BackendOptions(candidates=(spec.sibling_config,))


def _payload(spec):
    return json.loads(
        resources.files(spec.package.__name__).joinpath("workloads.json").read_text()
    )


def _envelope(spec, ceiling):
    """Token counts of the packaged allowlist to run: its listed counts, or the
    ends of its range."""
    counts = _payload(spec)["num_tokens"]
    if isinstance(counts, dict):
        excluded = set(counts.get("exclude", ()))
        admitted = [t for t in (counts["min"], counts["max"]) if t not in excluded]
    else:
        admitted = sorted(set(counts))
    return [t for t in admitted if t <= ceiling]


def _counting_bench_gpu_time(monkeypatch):
    """Record every cross-backend timing ``MoELayer`` runs."""
    from flashinfer.testing import utils as testing_utils

    timed = []
    bench = testing_utils.bench_gpu_time

    def counting(fn, **kwargs):
        timed.append(fn)
        return bench(fn, **kwargs)

    monkeypatch.setattr(testing_utils, "bench_gpu_time", counting)
    return timed


def _layer_takes_preferred_backends_untimed():
    """Whether this build's MoELayer takes a backend that prefers itself
    (``prefer_when_accepted``) without timing it, outside autotune; without that,
    the default layer times the specialized backend like any candidate."""
    from flashinfer.fused_moe.runners import MoERunner

    return "prefer_when_accepted" in vars(MoERunner)


def _eligible_keys(layer, act, weights):
    return [
        layer.runners[index].backend_key
        for index in layer._eligible_runners(act, weights)
    ]


# ---------------------------------------------------------------------------
# Registration, guards, kill switch, stats, AOT (any host)
# ---------------------------------------------------------------------------


def test_every_registered_kernel_has_a_spec():
    assert set(specialized.registered_kernels()) == set(SPECS)


@pytest.mark.parametrize("name", NAMES)
def test_registration(spec):
    kernel = specialized.registered_kernels()[spec.name]
    assert kernel is spec.package.KERNEL is spec.kernel_class
    assert kernel.weight_view == spec.sibling
    assert kernel.quant_variants == spec.quant_variants
    assert kernel.routing_modes == spec.routing_modes
    assert kernel.activation_classes == spec.activation_classes
    assert kernel.compute_capabilities == spec.compute_capabilities
    for cc in spec.compute_capabilities:
        assert SpecializedMoEConfig.supported(cc)
    for pair in spec.quant_variants:
        assert pair in SpecializedMoERunner.supported_quant_variants
    assert spec.sibling in SpecializedMoERunner.weight_view_aliases
    # the token-count allowlist ships as package data
    spec.kernel_module._load_allowlist.cache_clear()
    allowlist = spec.kernel_module._load_allowlist()
    assert allowlist is not None
    spec.check_allowlist(allowlist, _payload(spec))
    path = getattr(spec.kernel_module, "_KERNEL_MODULE", None)
    if path is not None:  # programs imported lazily from this module
        assert path == spec.kernel_module_path


@pytest.mark.parametrize("name,case", _cases("config_cases"))
def test_config_guard(spec, monkeypatch, case):
    """``check_support`` binds the kernel for exactly the served configuration
    (so MoELayer skips the backend for every other one). The device is pinned to
    one the kernel serves unless the case names another, so the rules run on
    any host."""
    overrides, message = spec.config_cases[case]
    overrides = spec.pin_device(monkeypatch, dict(overrides))
    runner = SpecializedMoERunner(spec.config(**overrides), torch.device("cuda", 0))
    if message is None:
        runner.check_support()
        assert [k.name for k in runner.kernels] == [spec.name]
        return
    with pytest.raises(NotImplementedError, match=message):
        runner.check_support()


@pytest.mark.parametrize("name,case", _cases("call_cases"))
def test_call_guard(spec, case):
    """Meta tensors pass every check but the device check, the last one."""
    pack_overrides, view_overrides, reason = spec.call_cases[case]
    pack_overrides = dict(pack_overrides)
    pack, view = spec.meta_call(pack_overrides.pop("num_tokens", None))
    pack.update(pack_overrides)
    view.update(view_overrides)
    seen: set = set()
    device = torch.device("cuda", 0)
    assert not spec.kernel_module.admit(MoEActivationPack(**pack), view, device, seen)
    assert spec.kernel_module.stats()["rejects"] == {reason: 1}
    assert not seen


@pytest.mark.parametrize("name", NAMES)
def test_kill_switch_and_errors(spec, monkeypatch):
    k = spec.kernel_module
    pack, view = spec.meta_call()
    pack = MoEActivationPack(**pack)
    device = torch.device("cuda", 0)
    # read at call time
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    assert not k.admit(pack, view, device, set())
    monkeypatch.setenv(KILL_SWITCH_ENV, "0")
    assert not k.admit(pack, view, device, set())
    st = k.stats()
    assert st["rejects"] == {"kill_switch": 1, "device": 1}
    assert st["calls"] == 2 and st["dispatch_count"] == 0
    assert specialized.stats()[spec.name]["rejects"] == st["rejects"]
    json.dumps(st)

    # a guard failure rejects the call instead of raising
    def broken():
        raise RuntimeError("allowlist read failed")

    monkeypatch.setattr(k, "_load_allowlist", broken)
    assert not k.admit(pack, view, device, set())
    assert k.stats()["rejects"]["guard_error"] == 1

    # direct use outside the contract raises instead of launching
    bad = MoEActivationPack(**{**spec.meta_call()[0], **spec.contract_violation()})
    with pytest.raises(ValueError, match="input_layout"):
        k.check_contract(bad, view)


@pytest.mark.parametrize("name", NAMES)
def test_stats(spec, monkeypatch):
    k = spec.kernel_module
    device = torch.device("cuda", 0)
    for num_tokens in (spec.meta_rejected_tokens, spec.meta_tokens):
        pack, view = spec.meta_call(num_tokens)
        k.admit(MoEActivationPack(**pack), view, device, set())
    st = spec.package.stats()
    assert st == specialized.stats()[spec.name]
    assert st["calls"] == 2 and st["dispatch_count"] == 0
    assert st["rejects"] == {"num_tokens": 1, "device": 1}
    json.dumps(st)

    # side-effect free: no kernel-module import, build, CUDA call or file read
    path = getattr(k, "_KERNEL_MODULE", None)
    if path is not None:
        monkeypatch.delitem(sys.modules, path, raising=False)

    def no_cuda(*args, **kwargs):
        raise AssertionError("stats() touched CUDA")

    monkeypatch.setattr(torch.cuda, "get_device_capability", no_cuda)
    monkeypatch.setattr(torch.cuda, "current_device", no_cuda)
    k._load_allowlist.cache_clear()
    st = spec.package.stats()
    assert k._load_allowlist.cache_info().currsize == 0
    assert st["allowlist"] is None and st["precompiled"] == []
    if path is not None:
        assert path not in sys.modules
        assert st["kernel_module_loaded"] is False


@pytest.mark.parametrize("name", NAMES)
def test_aot(spec, monkeypatch):
    spec.check_aot(monkeypatch)


@pytest.mark.parametrize("name", NAMES)
def test_auto_lists_the_specialized_backend_first(spec):
    """``auto`` (the default backend list) lists the specialized backend first,
    and it prefers itself where it admits a call (its kernels admit only the
    calls where they were measured faster than the other backends)."""
    default = spec.config().backend
    assert list(default)[0] == SpecializedMoEConfig()
    assert SpecializedMoERunner.prefer_when_accepted is True


# ---------------------------------------------------------------------------
# MoELayer on the kernel's device
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", NAMES)
def test_layer_candidates(spec):
    """The default layer of the served configuration builds the specialized
    backend, first, with this kernel only where the kernel runs; elsewhere it is
    inert. An explicit list without it never builds it."""
    if not torch.cuda.is_available():
        pytest.skip("MoELayer needs CUDA")
    # Both layers need the reference backend; elsewhere (e.g. SM90 for
    # trtllm_fp8_block) no configured backend serves the configuration.
    major, minor = get_compute_capability(torch.device("cuda"))
    if not type(spec.sibling_config).supported(major * 10 + minor):
        pytest.skip(f"{spec.sibling} does not run on SM{major}{minor}")
    keys = [r.backend_key for r in MoELayer(spec.config()).runners]
    assert spec.sibling in keys
    if spec.available():
        assert keys[0] == _KEY
        assert [k.name for k in _runner(MoELayer(spec.config())).kernels] == [spec.name]
    else:
        assert _KEY not in keys
    explicit = MoELayer(spec.config(backend=_sibling_only(spec)))
    assert [r.backend_key for r in explicit.runners] == [spec.sibling]


@pytest.mark.parametrize("name", NAMES)
def test_kill_switch_through_layer(spec, monkeypatch):
    """``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` keeps every ``MoELayer`` call
    off the kernel (no build, no launch), whatever the layer's candidates, and
    is read per call."""
    _needs_kernel(spec)
    k = spec.kernel_module
    _, weights = spec.weights()
    pack = spec.pack(*spec.inputs(spec.lazy_admitted_tokens))
    layer = MoELayer(spec.config())
    only = MoELayer(spec.config(backend=_SPECIALIZED_ONLY))
    assert [r.backend_key for r in only.runners] == [_KEY]

    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    assert _KEY not in _eligible_keys(layer, pack, weights)
    layer(pack, weights)
    assert layer.winner_backend == spec.sibling
    # the specialized backend alone has no eligible runner left
    with pytest.raises(NotImplementedError, match="none of the usable backends"):
        only(pack, weights)
    st = k.stats()
    assert st["dispatch_count"] == 0 and st["precompiled"] == [], st

    # read at call time: the same layers reach the kernel once it is unset
    monkeypatch.delenv(KILL_SWITCH_ENV)
    only(pack, weights)
    assert only.winner_backend == _KEY
    st = k.stats()
    assert torch.cuda.current_device() in st["precompiled"]
    dispatched = st["dispatch_count"]
    assert dispatched >= 1
    layer(pack, weights)  # a new eligible set: untimed or timed, it runs the kernel
    assert _KEY in _eligible_keys(layer, pack, weights)
    assert k.stats()["dispatch_count"] > dispatched


@pytest.mark.parametrize("name", NAMES)
def test_lazy_build_and_range(spec):
    _needs_kernel(spec)
    k = spec.kernel_module
    _, weights = spec.weights()
    layer = MoELayer(spec.config())
    # outside the allowlist: the reference backend only, nothing built
    layer(spec.pack(*spec.inputs(spec.lazy_rejected_tokens)), weights)
    assert layer.winner_backend == spec.sibling
    st = k.stats()
    assert st["rejects"] == {"num_tokens": 1} and st["precompiled"] == []
    # the first admitted eager call builds the kernel for the device
    num_tokens = spec.lazy_admitted_tokens
    pack = spec.pack(*spec.inputs(num_tokens))
    layer(pack, weights)
    st = k.stats()
    assert torch.cuda.current_device() in st["precompiled"]
    assert num_tokens in st["dispatched_num_tokens"]
    spec.check_build_stats(st)
    # a cached winner: one launch per later call (either backend may have won;
    # which one the default layer takes is test_auto_first_inside_the_envelope)
    if layer.winner_backend == _KEY:
        dispatched = st["dispatch_count"]
        layer(pack, weights)
        assert k.stats()["dispatch_count"] == dispatched + 1


@pytest.mark.parametrize("name,case", _cases("numerics_cases"))
def test_matches_reference(spec, case):
    """The kernel, the reference backend and the default layer against the
    kernel's fp32 reference. The default layer may run either backend here (it
    takes the kernel untimed where MoELayer honours ``prefer_when_accepted``,
    else it times both); speed is ``test_performance_layer_selects_specialized``."""
    _needs_kernel(spec)
    k = spec.kernel_module
    params = dict(spec.numerics_cases)[case]
    num_tokens = params["num_tokens"]
    _, weights = spec.weights()
    inputs = spec.inputs(num_tokens, params.get("kind", "serving"))
    pack = spec.pack(*inputs)
    layer = MoELayer(spec.config())
    assert [r.backend_key for r in layer.runners][0] == _KEY

    # The specialized runner on its own (the numerics under test) ...
    runner = _runner(layer)
    assert runner.accepts(pack, weights)
    out = runner.forward(runner.pack_inputs(pack, weights)).clone()
    assert k.stats()["dispatched_num_tokens"] == {num_tokens: 1}
    # ... the reference backend on its own ...
    stock = MoELayer(spec.config(backend=_sibling_only(spec)))(pack, weights).clone()
    assert k.stats()["dispatch_count"] == 1
    # ... and the default layer.
    layered = layer(pack, weights).clone()
    assert k.stats()["dispatched_num_tokens"][num_tokens] > 1
    assert layer.winner_backend in (_KEY, spec.sibling)
    torch.cuda.synchronize()
    spec.check_numerics(inputs, out, stock, layered, layer.winner_backend == _KEY)


@pytest.mark.parametrize(
    "name,num_tokens",
    [
        pytest.param(n, t, id=f"{n}-{t}")
        for n, s in SPECS.items()
        for t in s.compact_padded_tokens
    ],
)
def test_compact_and_padded_agree(spec, num_tokens):
    """Kernels with more than one program set for a call agree across them."""
    _needs_kernel(spec)
    spec.check_compact_padded(num_tokens)


def _copy_inputs(dst, src):
    for d, s in zip(dst, src, strict=True):
        d.copy_(s)


def _capture(layer, pack, weights):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        out = layer(pack, weights)
    return graph, out


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("case", ["selected_eagerly", "unbuilt", "unselected"])
def test_cuda_graph(spec, monkeypatch, case):
    """A captured call runs the kernel only when it is built and the layer
    already selected its winner for that call eagerly; otherwise it is captured
    on the reference backend (never a build or a timing inside capture)."""
    _needs_kernel(spec)
    k = spec.kernel_module
    _, weights = spec.weights()
    num_tokens = spec.graph_tokens
    inputs = spec.inputs(num_tokens, seed=2)
    pack = spec.pack(*inputs)
    if case == "selected_eagerly":
        # The specialized backend alone, so the eager selection is the kernel
        # whatever the timing (the speed comparison is a separate test).
        layer = MoELayer(spec.config(backend=_SPECIALIZED_ONLY))
        layer(pack, weights)
        assert layer.winner_backend == _KEY
    else:
        layer = MoELayer(spec.config())
        # warm the reference backend for this call eagerly (its first launch
        # must not happen in capture) with the specialized backend switched off
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
        layer(pack, weights)
        monkeypatch.delenv(KILL_SWITCH_ENV)
        if case == "unselected":  # built by another call size, then capture
            layer(spec.pack(*spec.inputs(spec.lazy_admitted_tokens)), weights)
            assert k.is_ready(pack.hidden_states_q.device)
    torch.cuda.synchronize()

    captured_before = k.stats()["capture_dispatch_count"]
    graph, out = _capture(layer, pack, weights)
    st = k.stats()
    on_kernel = case == "selected_eagerly"
    assert st["capture_dispatch_count"] - captured_before == int(on_kernel), st
    if not on_kernel:
        reason = "capture_unready" if case == "unbuilt" else "capture_unselected"
        assert st["rejects"].get(reason, 0) == 1, st

    new = spec.inputs(num_tokens, seed=3)
    _copy_inputs(inputs, new)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    if not on_kernel:
        monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    eager = layer(spec.pack(*new), weights)
    torch.cuda.synchronize()
    assert layer.winner_backend == (_KEY if on_kernel else spec.sibling)
    spec.check_replay(new, replayed, eager, on_kernel)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("case", ["reset_winner", "failed_selection"])
def test_cuda_graph_without_a_cached_winner(spec, monkeypatch, case):
    """An eager admission whose winner is gone (``reset_winner``) or was never
    cached (the selection raised) must not reach candidate timing inside a
    capture: the reset forgets the admission, and a capture-time miss leaves the
    specialized backend out, so the call is captured on the reference backend."""
    _needs_kernel(spec)
    k = spec.kernel_module
    _, weights = spec.weights()
    num_tokens = spec.graph_tokens
    inputs = spec.inputs(num_tokens, seed=4)
    pack = spec.pack(*inputs)
    layer = MoELayer(spec.config())
    if case == "reset_winner":
        layer(pack, weights)  # admits the call eagerly, whichever backend wins
        assert _KEY in _eligible_keys(layer, pack, weights)
        layer.reset_winner()
    else:
        select = MoELayer._select_winner

        def failing(self, *args, **kwargs):
            select(self, *args, **kwargs)  # selects (timing when needed), then fails
            raise RuntimeError("selection failed")

        monkeypatch.setattr(MoELayer, "_select_winner", failing)
        with pytest.raises(RuntimeError, match="selection failed"):
            layer(pack, weights)
        monkeypatch.setattr(MoELayer, "_select_winner", select)
        assert not layer._winners
    torch.cuda.synchronize()

    captured_before = k.stats()["capture_dispatch_count"]
    graph, out = _capture(layer, pack, weights)
    assert layer.winner_backend == spec.sibling
    st = k.stats()
    assert st["capture_dispatch_count"] == captured_before, st
    if case == "reset_winner":
        assert st["rejects"].get("capture_unselected", 0) == 1, st

    new = spec.inputs(num_tokens, seed=5)
    _copy_inputs(inputs, new)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    replayed = out.clone()
    eager = MoELayer(spec.config(backend=_sibling_only(spec)))(spec.pack(*new), weights)
    torch.cuda.synchronize()
    spec.check_replay(new, replayed, eager, False)


@pytest.mark.parametrize("name", NAMES)
def test_auto_first_inside_the_envelope(spec, monkeypatch):
    """``auto`` dispatches the kernel for every token count of its envelope
    without timing it against the other backends (where MoELayer honours
    ``prefer_when_accepted``; otherwise it is timed like any candidate);
    outside the envelope the backend is not a candidate."""
    _needs_kernel(spec)
    k = spec.kernel_module
    _, weights = spec.weights()
    config = spec.config()
    layer = MoELayer(config)
    timed = _counting_bench_gpu_time(monkeypatch)
    untimed = _layer_takes_preferred_backends_untimed()
    for num_tokens in _envelope(spec, config.execution.tune_max_num_tokens):
        pack = spec.pack(*spec.inputs(num_tokens))
        keys = _eligible_keys(layer, pack, weights)
        assert keys[0] == _KEY, (num_tokens, keys)
        before, dispatched = len(timed), k.stats()["dispatch_count"]
        out = layer(pack, weights)
        torch.cuda.synchronize()
        assert bool(torch.isfinite(out.float()).all()), num_tokens
        if untimed:
            assert layer.winner_backend == _KEY, num_tokens
            assert len(timed) == before, num_tokens  # no cross-backend timing
            assert k.stats()["dispatch_count"] == dispatched + 1, num_tokens
        else:
            assert layer.winner_backend in keys, num_tokens
    pack = spec.pack(*spec.inputs(spec.lazy_rejected_tokens))
    dispatched = k.stats()["dispatch_count"]
    assert _KEY not in _eligible_keys(layer, pack, weights)
    layer(pack, weights)
    assert layer.winner_backend != _KEY
    assert k.stats()["dispatch_count"] == dispatched


@pytest.mark.parametrize("name", NAMES)
def test_autotune_times_every_backend(spec, monkeypatch):
    """Under ``autotune(True)`` the default layer times every eligible backend,
    the specialized one included, and keeps the measured winner afterwards; a
    bucket selected untimed before autotuning is selected again by timing."""
    _needs_kernel(spec)
    _, weights = spec.weights()
    config = spec.config()
    pack = spec.pack(*spec.inputs(spec.lazy_admitted_tokens))
    timed = _counting_bench_gpu_time(monkeypatch)
    try:
        layer = MoELayer(config)
        eligible = _eligible_keys(layer, pack, weights)
        assert _KEY in eligible and len(eligible) >= 2, eligible
        with autotune(True):
            layer(pack, weights)
        assert len(timed) == len(eligible)
        winner = layer.winner_backend
        assert winner in eligible
        layer(pack, weights)
        assert layer.winner_backend == winner and len(timed) == len(eligible)

        if _layer_takes_preferred_backends_untimed():
            retuned = MoELayer(config)
            retuned(pack, weights)
            assert retuned.winner_backend == _KEY
            assert len(timed) == len(eligible)
            with autotune(True):
                retuned(pack, weights)
            assert len(timed) == 2 * len(eligible)
            assert retuned.winner_backend in eligible
    finally:
        # leave no tuned tactic behind for the tests that time untuned paths
        AutoTuner.get().clear_cache()


def _check_cases():
    out = []
    for name, spec in SPECS.items():
        for check in spec.checks:
            for param_id, _ in check.params:
                case = f"{check.name}/{param_id}"
                out.append(pytest.param(name, case, id=f"{name}-{case}".rstrip("/")))
    return out


@pytest.mark.parametrize("name,case", _check_cases())
def test_kernel_check(spec, monkeypatch, case):
    """The kernel's own checks (``SPEC.checks``): layout, permutation,
    tie-breaking, streams, graph pools and the like."""
    check_name, _, param_id = case.partition("/")
    (check,) = [c for c in spec.checks if c.name == check_name]
    if check.needs_kernel:
        _needs_kernel(spec)
    reason = check.skip() if check.skip is not None else None
    if reason:
        pytest.skip(reason)
    params = dict(check.params)[param_id]
    check.fn(spec, SimpleNamespace(monkeypatch=monkeypatch, **params))


# ---------------------------------------------------------------------------
# Performance on the kernel's device (the functional tests accept either winner)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,case", _cases("perf_cases"))
def test_performance_layer_selects_specialized(spec, case):
    """A performance test, not a functional one: the kernel is at least
    ``SPEC.min_speedup`` faster than the reference backend at these sizes
    (``MoELayer`` with only one or the other, CUPTI on CUDA-graph replay, so
    host launch gaps between the layer's kernels do not count; the reference
    backend's default tactic; the speedup is asserted only when CUPTI timing is
    available), and the default layer runs the kernel. A failure here is a
    speed regression (or a contended GPU), not a numerical one."""
    from flashinfer.testing import bench_gpu_time

    _needs_kernel(spec)
    AutoTuner.get().clear_cache()
    _, weights = spec.weights()
    pack = spec.pack(*spec.inputs(dict(spec.perf_cases)[case]))
    times_ms = {}
    cupti = True
    for key, backend in (
        (spec.sibling, _sibling_only(spec)),
        (_KEY, _SPECIALIZED_ONLY),
    ):
        single = MoELayer(spec.config(backend=backend))
        single(pack, weights)  # build, tune and select outside the timing
        assert single.winner_backend == key
        # bench_gpu_time reports its CUDA-event fallback only by this warning.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            times = bench_gpu_time(
                lambda single=single: single(pack, weights),
                enable_cupti=True,
                use_cuda_graph=True,
            )
        if any("Falling back to CUDA events" in str(w.message) for w in caught):
            cupti = False
        times_ms[key] = median(times)
    speedup = times_ms[spec.sibling] / times_ms[_KEY]
    # The thresholds come from CUPTI kernel times; without cupti-python >= 13
    # bench_gpu_time falls back to CUDA events, which include launch overhead.
    if cupti:
        assert speedup >= spec.min_speedup, (times_ms, speedup)

    layer = MoELayer(spec.config())
    layer(pack, weights)
    assert layer.winner_backend == _KEY, (times_ms, speedup)
