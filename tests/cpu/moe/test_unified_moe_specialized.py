"""CPU-only checks of the unified MoE API's ``specialized`` backend framework.

Kernel-independent: kernel discovery, every registered kernel's declarations, the
runner's declarations and kernel binding, the backend's registration, the runner's
kill switch, how ``MoELayer`` keeps the backend's capture-time admissions tied
to its cached winners, and how it selects without timing: a backend that prefers
itself outside autotune, any call inside a CUDA-graph capture. Every kernel's guards, numerics, kill switch, CUDA-graph
behavior and own checks are tested in ``tests/moe/test_unified_moe_specialized.py``
from its spec in ``tests/moe/specialized_kernels/``.
"""

from __future__ import annotations

import json
import pkgutil
import sys
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import torch

import flashinfer.utils as fi_utils
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MoEConfig,
    MoELayer,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SpecializedMoEConfig,
    SwiGLU,
)
from flashinfer.fused_moe import api as moe_api
from flashinfer.fused_moe import layer as layer_module
from flashinfer.fused_moe.backends import specialized
from flashinfer.fused_moe.backends.specialized import kernels as kernels_package
from flashinfer.fused_moe.backends.specialized import registry
from flashinfer.fused_moe.backends.specialized.kernel import (
    KILL_SWITCH_ENV,
    SpecializedMoEKernel,
    kill_switch_engaged,
)
from flashinfer.fused_moe.backends.specialized.runner import SpecializedMoERunner
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import MoERunner

_ARCHS = (75, 80, 89, 90, 100, 103, 107, 110, 120, 121)
# Every kernel's stats() reports at least these fields.
_STATS_KEYS = {
    "calls",
    "dispatch_count",
    "capture_dispatch_count",
    "dispatched_num_tokens",
    "rejects",
    "kernel_module_loaded",
    "compiled_variants",
    "precompiled",
    "failed_devices",
    "allowlist",
    "precompile_seconds",
    "precompile_errors",
    "run_errors",
}


def _kernel_subpackages():
    return sorted(
        info.name
        for info in pkgutil.iter_modules(kernels_package.__path__)
        if info.ispkg and not info.name.startswith("_")
    )


@pytest.mark.cpu_only
def test_every_kernel_subpackage_registers():
    names = _kernel_subpackages()
    assert names, "the specialized backend has no kernel subpackage"
    assert specialized.discovery_errors() == {}
    assert list(specialized.registered_kernels()) == names


@pytest.mark.cpu_only
@pytest.mark.parametrize("name", _kernel_subpackages())
def test_kernel_declarations(name):
    kernel = specialized.registered_kernels()[name]
    assert issubclass(kernel, SpecializedMoEKernel) and kernel.name == name
    assert kernel.quant_variants and kernel.routing_modes
    assert kernel.activation_classes and kernel.compute_capabilities
    assert kernel.output_formats
    # The kernel reads the weight view of another registered backend that serves
    # its quantization pairs, so a default MoELayer needs no extra preparation.
    siblings = [
        runner
        for runner in _BACKEND_RUNNERS.values()
        if runner.backend_key == kernel.weight_view
    ]
    assert siblings, f"{name} reads view {kernel.weight_view!r} of no backend"
    assert any(
        set(kernel.quant_variants) <= set(runner.supported_quant_variants)
        for runner in siblings
    )


@pytest.mark.cpu_only
def test_runner_declarations_are_the_union_of_the_kernels():
    kernels = list(specialized.registered_kernels().values())
    runner = SpecializedMoERunner
    assert runner.backend_key == "specialized"
    assert runner.capabilities_from_kernel_registry is True
    assert set(runner.supported_quant_variants) == {
        pair for kernel in kernels for pair in kernel.quant_variants
    }
    assert set(runner.supported_routing_modes) == {
        mode for kernel in kernels for mode in kernel.routing_modes
    }
    assert set(runner.weight_view_aliases) == {kernel.weight_view for kernel in kernels}
    for pair, activations in runner.supported_activation_classes_by_quant.items():
        assert set(activations) == {
            cls
            for kernel in kernels
            if pair in kernel.quant_variants
            for cls in kernel.activation_classes
        }
    for flag in (
        "supports_expert_parallelism",
        "supports_fused_shared_experts",
        "supports_per_token_scale",
    ):
        assert getattr(runner, flag) == any(getattr(k, flag) for k in kernels)


@pytest.mark.cpu_only
def test_backend_registration():
    config = SpecializedMoEConfig()
    assert SpecializedMoEConfig in moe_api.ALL_BACKEND_CONFIGS
    assert eval(repr(config), vars(moe_api)) == config
    assert _BACKEND_RUNNERS[SpecializedMoEConfig] is SpecializedMoERunner
    assert config in moe_api._DEFAULT_BACKEND.candidates
    kernels = specialized.registered_kernels().values()
    assert [a for a in _ARCHS if SpecializedMoEConfig.supported(a)] == [
        a for a in _ARCHS if any(a in k.compute_capabilities for k in kernels)
    ]


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", sorted({103, 100, 90}))
def test_runner_rejects_a_configuration_no_kernel_serves(monkeypatch, arch):
    """check_support names every registered kernel's reason, so MoELayer skips
    the backend and reports why."""
    monkeypatch.setattr(
        fi_utils, "get_compute_capability", lambda device: divmod(arch, 10)
    )
    pair = SpecializedMoERunner.supported_quant_variants[0]
    activation = SpecializedMoERunner.supported_activation_classes_by_quant[pair][0]
    config = MoEConfig(
        routing=RoutingConfig(num_experts=24, top_k=3),
        quant=QuantConfig(weight=pair[0], activation=pair[1]),
        experts=ExpertConfig(intermediate_size=384),
        activation=activation(),
        backend=BackendOptions(candidates=(SpecializedMoEConfig(),)),
    )
    runner = SpecializedMoERunner(config, torch.device("cuda", 0))
    with pytest.raises(NotImplementedError, match="no specialized kernel serves") as e:
        runner.check_support()
    for name in specialized.registered_kernels():
        assert name in str(e.value)
    assert runner.kernels == []


@pytest.mark.cpu_only
def test_declared_rejection_reasons():
    kernel = next(iter(specialized.registered_kernels().values()))
    pair = kernel.quant_variants[0]

    def config(**kw):
        experts = dict(intermediate_size=384)
        experts.update(kw.pop("experts", {}))
        return MoEConfig(
            routing=RoutingConfig(num_experts=24, top_k=3),
            quant=kw.pop("quant", QuantConfig(weight=pair[0], activation=pair[1])),
            experts=ExpertConfig(**experts),
            activation=kernel.activation_classes[0](),
        )

    reject = SpecializedMoERunner._declared_rejection
    assert reject(kernel, config()) is None
    other = next(
        p
        for p in (
            (QuantFormat.BF16, QuantFormat.BF16),
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
        )
        if p not in kernel.quant_variants
    )
    quant = QuantConfig(weight=other[0], activation=other[1])
    assert "does not serve" in reject(kernel, config(quant=quant))
    if not kernel.supports_expert_parallelism:
        assert "expert parallelism" in reject(
            kernel, config(experts={"local_num_experts": 12})
        )


@pytest.mark.cpu_only
def test_activation_matrix_leaves_the_specialized_backend_out():
    from scripts.generate_moe_activation_matrix import get_activation_matrix_rows

    assert "specialized" not in {row[0] for row in get_activation_matrix_rows()}


@pytest.mark.cpu_only
def test_kill_switch_is_read_at_call_time(monkeypatch):
    assert KILL_SWITCH_ENV == "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"
    assert specialized.KILL_SWITCH_ENV == KILL_SWITCH_ENV
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert not kill_switch_engaged()
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    assert kill_switch_engaged()
    monkeypatch.setenv(KILL_SWITCH_ENV, "0")
    assert not kill_switch_engaged()


@pytest.mark.cpu_only
def test_stats_are_side_effect_free(monkeypatch):
    def no_cuda(*args, **kwargs):
        raise AssertionError("stats() touched CUDA")

    for name in ("get_device_capability", "current_device", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, no_cuda)
    stats = specialized.stats()
    assert list(stats) == list(specialized.registered_kernels())
    for name, kernel_stats in stats.items():
        assert set(kernel_stats) >= _STATS_KEYS, name
    json.dumps(stats)


@pytest.mark.cpu_only
def test_aot_modules_follow_the_capabilities():
    assert registry.gen_specialized_moe_aot_modules({}) == []


@pytest.mark.cpu_only
def test_discovery_skips_and_records_broken_kernels(monkeypatch, tmp_path):
    """A subpackage that fails to import or to declare its kernel is skipped and
    recorded; the others still register, and private subpackages are ignored."""
    template = (
        "from flashinfer.fused_moe.api import QuantFormat, RoutingInputMode, SwiGLU\n"
        "from flashinfer.fused_moe.backends.specialized.kernel import (\n"
        "    SpecializedMoEKernel,\n"
        ")\n\n\n"
        "class Kernel(SpecializedMoEKernel):\n"
        "    name = {name!r}\n"
        "    weight_view = 'trtllm_bf16_routed'\n"
        "    quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)\n"
        "    routing_modes = (RoutingInputMode.FromLogits,)\n"
        "    activation_classes = (SwiGLU,)\n"
        "    compute_capabilities = (103,)\n\n"
        "    @classmethod\n"
        "    def config_rejection(cls, config, arch, device):\n"
        "        return 'test kernel'\n\n"
        "    def accepts(self, act, view):\n"
        "        return False\n\n"
        "    def pack_inputs(self, act, view):\n"
        "        raise NotImplementedError\n\n"
        "    def forward(self, inputs):\n"
        "        raise NotImplementedError\n\n"
        "    @classmethod\n"
        "    def stats(cls):\n"
        "        return {{}}\n\n\n"
        "KERNEL = Kernel\n"
    )
    packages = {
        "good_kernel": template.format(name="good_kernel"),
        "misnamed_kernel": template.format(name="another_name"),
        "broken_kernel": "raise ImportError('broken on purpose')\n",
        "no_kernel": "VALUE = 1\n",
        "_private": "raise AssertionError('private subpackages are not imported')\n",
    }
    for name, source in packages.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "__init__.py").write_text(source)
    monkeypatch.setattr(kernels_package, "__path__", [str(tmp_path)])
    monkeypatch.setattr(registry, "_discovery_errors", {})
    for name in packages:
        monkeypatch.delitem(
            sys.modules, f"{kernels_package.__name__}.{name}", raising=False
        )
    try:
        found = registry.registered_kernels.__wrapped__()
        assert list(found) == ["good_kernel"]
        errors = registry._discovery_errors
        assert set(errors) == {"broken_kernel", "misnamed_kernel", "no_kernel"}
        assert "broken on purpose" in errors["broken_kernel"]
        assert "subpackage name" in errors["misnamed_kernel"]
    finally:
        for name in packages:
            sys.modules.pop(f"{kernels_package.__name__}.{name}", None)


class _StubKernel(SpecializedMoEKernel):
    """A kernel that admits every call (and records it like a real kernel's
    eager admission), so the runner's own checks are what is under test."""

    name = "stub_kernel"
    weight_view = "trtllm_bf16_routed"
    quant_variants = ((QuantFormat.BF16, QuantFormat.BF16),)
    routing_modes = (RoutingInputMode.FromLogits,)
    activation_classes = (SwiGLU,)
    compute_capabilities = (103,)

    def __init__(self, config, device):
        super().__init__(config, device)
        self.asked = 0

    @classmethod
    def config_rejection(cls, config, arch, device):
        return None

    def accepts(self, act, view):
        self.asked += 1
        self.eager_num_tokens.add(act.num_tokens)
        return True

    def pack_inputs(self, act, view):
        raise AssertionError("the stub kernel is never packed")

    def forward(self, inputs):
        raise AssertionError("the stub kernel never runs")

    @classmethod
    def stats(cls):
        return {}


def _stub_binding(monkeypatch):
    """A specialized runner bound to one stub kernel, and a call it admits (the
    kill switch cleared, whatever the caller's environment)."""
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setattr(fi_utils, "get_compute_capability", lambda device: (10, 3))
    config = MoEConfig(
        routing=RoutingConfig(num_experts=24, top_k=3),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=384),
        backend=BackendOptions(candidates=(SpecializedMoEConfig(),)),
    )
    runner = SpecializedMoERunner(config, torch.device("cuda", 0))
    kernel = _StubKernel(config, runner.device)
    runner.kernels = [kernel]
    act = SimpleNamespace(routing_input_mode=RoutingInputMode.FromLogits, num_tokens=6)
    weights = SimpleNamespace(native_views={kernel.weight_view: {}})
    return runner, kernel, act, weights


@pytest.mark.cpu_only
def test_runner_kill_switch_rejects_before_any_kernel(monkeypatch):
    """The runner enforces the kill switch itself: a kernel that admits every
    call is not even asked while it is set."""
    runner, kernel, act, weights = _stub_binding(monkeypatch)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    assert runner.accepts(act, weights)
    assert kernel.asked == 1
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    assert not runner.accepts(act, weights)
    assert kernel.asked == 1
    monkeypatch.setenv(KILL_SWITCH_ENV, "0")
    assert runner.accepts(act, weights)
    assert kernel.asked == 2


@pytest.mark.cpu_only
def test_reset_winner_forgets_eager_admissions(monkeypatch):
    """The eager admissions stand for the layer's cached winners, so
    ``MoELayer.reset_winner()`` clears them with the winners: a captured call
    then needs a new eager selection rather than missing the cache."""
    runner, kernel, act, weights = _stub_binding(monkeypatch)
    assert runner.accepts(act, weights)
    assert kernel.eager_num_tokens == {6} and runner._admitted_by == {6: kernel}
    layer = SimpleNamespace(
        runners=[runner],
        _winners=OrderedDict({(8, act.routing_input_mode, (0,)): (runner, -1)}),
        _last_winner_backend=runner.backend_key,
    )
    MoELayer.reset_winner(layer)
    assert not layer._winners and layer._last_winner_backend is None
    assert kernel.eager_num_tokens == set() and runner._admitted_by == {}


def _plain_runner(name):
    return SimpleNamespace(
        backend_key=name,
        pack_inputs=lambda act, weights: [name],
        launch_kwargs_for=lambda inputs: {},
        tuning_config_for=lambda inputs: None,
    )


def _bench_outside_capture(monkeypatch, capturing, cost=None):
    """Patch ``bench_gpu_time`` to record the timed backends; timing inside a
    CUDA-graph capture fails, as the real one does (it synchronizes)."""
    timed = []

    def bench(fn, **kwargs):
        assert not capturing[0], "a candidate was timed inside a CUDA-graph capture"
        runner = fn.__defaults__[0]
        timed.append(runner.backend_key)
        return [cost(runner) if cost else float(len(timed))]

    monkeypatch.setattr("flashinfer.testing.utils.bench_gpu_time", bench)
    monkeypatch.setattr(layer_module, "_stream_capturing", lambda: capturing[0])
    return timed


@pytest.mark.cpu_only
@pytest.mark.parametrize("others", [1, 2])
def test_capture_miss_leaves_the_specialized_backend_out(monkeypatch, others):
    """A winner-cache miss inside a CUDA-graph capture (e.g. an admission whose
    winner selection failed or was evicted) times nothing: the specialized
    backend steps aside and the first other candidate is taken untimed, marked
    so that the next call outside a capture selects again."""
    runner, _, act, weights = _stub_binding(monkeypatch)
    plain = [_plain_runner(f"plain{i}") for i in range(others)]
    timed = _bench_outside_capture(monkeypatch, [True])

    def choose_one(custom_op, runners, tuning_config, inputs, **kwargs):
        assert runners[0] is not runner, "the specialized backend was tuned"
        return runners[0], 3

    layer = SimpleNamespace(tuner=SimpleNamespace(choose_one=choose_one))
    candidates = [runner, *plain]

    winner = MoELayer._select_winner(layer, act, weights, candidates)
    assert winner == (plain[0], 3) and timed == []
    # one other candidate is the selection of the reduced pool; of several,
    # the first was taken without timing
    assert isinstance(winner, layer_module._CaptureWinner) == (others > 1)
    assert SpecializedMoERunner.capture_requires_eager_selection is True


def _layer_runner(name, cost, *, eager_only=False):
    """A runner that ``MoELayer.__call__`` can select and run: it accepts every
    call, times as ``cost`` and returns its name."""
    runner = SimpleNamespace(
        backend_key=name,
        cost=cost,
        supported_routing_modes=(RoutingInputMode.FromLogits,),
        accepts=lambda act, weights: True,
        pack_inputs=lambda act, weights: [name],
        launch_kwargs_for=lambda inputs: {},
        tuning_config_for=lambda inputs: None,
        forward=lambda inputs, tactic=-1, **kwargs: name,
    )
    if eager_only:
        runner.capture_requires_eager_selection = True
    return runner


def _cpu_layer(*runners):
    """A ``MoELayer`` over stub runners (no device), outside autotune."""
    layer = MoELayer.__new__(MoELayer)
    layer.config = MoEConfig(
        routing=RoutingConfig(num_experts=24, top_k=3),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=384),
    )
    layer.runners = list(runners)
    layer._tuning_runner_ids = list(range(len(runners)))
    layer._winners = OrderedDict()
    layer._last_winner_backend = None
    layer.tuner = SimpleNamespace(
        is_tuning_mode=False,
        choose_one=lambda custom_op, runners, tuning_config, inputs, **kw: (
            runners[0],
            -1,
        ),
    )
    return layer


def _cost(runner):
    return runner.cost


@pytest.mark.cpu_only
def test_capture_selection_never_takes_the_full_pool_entry(monkeypatch):
    """A first call of a bucket inside a CUDA-graph capture selects without the
    specialized backend and caches that under its reduced pool's key, so a later
    eager call of the same bucket still selects among every eligible backend and
    can take the specialized one (which is faster here)."""
    plain = _layer_runner("plain", cost=2.0)
    special = _layer_runner("specialized", cost=1.0, eager_only=True)
    layer = _cpu_layer(plain, special)
    capturing = [True]
    timed = _bench_outside_capture(monkeypatch, capturing, _cost)
    act = SimpleNamespace(routing_input_mode=RoutingInputMode.FromLogits, num_tokens=6)

    assert layer(act, None) == "plain"
    assert timed == []  # one candidate left in capture: nothing is timed
    assert [key[2] for key in layer._winners] == [(0,)]

    capturing[0] = False
    assert layer(act, None) == "specialized"
    assert timed == ["plain", "specialized"]
    assert [key[2] for key in layer._winners] == [(0,), (0, 1)]

    # a later capture of the bucket replays the eager selection
    capturing[0] = True
    assert layer(act, None) == "specialized"
    assert layer.winner_backend == "specialized" and len(timed) == 2


@pytest.mark.cpu_only
def test_capture_pick_is_selected_again_outside_capture(monkeypatch):
    """A first call of a bucket inside a capture with several ordinary
    candidates times nothing: it takes the first in configured order. The next
    call outside a capture selects the bucket again, timing every candidate,
    and a later capture replays that selection."""
    slow = _layer_runner("slow", cost=2.0)
    fast = _layer_runner("fast", cost=1.0)
    layer = _cpu_layer(slow, fast)
    capturing = [True]
    timed = _bench_outside_capture(monkeypatch, capturing, _cost)
    act = SimpleNamespace(routing_input_mode=RoutingInputMode.FromLogits, num_tokens=6)

    assert layer(act, None) == "slow" and timed == []
    assert layer(act, None) == "slow" and timed == []  # cached for captures

    capturing[0] = False
    assert layer(act, None) == "fast"
    assert timed == ["slow", "fast"]
    assert [key[2] for key in layer._winners] == [(0, 1)]
    assert type(layer._winners[(8, act.routing_input_mode, (0, 1))]) is tuple

    capturing[0] = True
    assert layer(act, None) == "fast" and len(timed) == 2


def _preferred_runner(name, cost):
    runner = _layer_runner(name, cost=cost, eager_only=True)
    runner.prefer_when_accepted = True
    return runner


@pytest.mark.cpu_only
def test_preferred_backend_wins_untimed_outside_autotune(monkeypatch):
    """A backend that prefers itself (``prefer_when_accepted``, as the
    specialized one does) wins every call it accepts without cross-backend
    timing, even where it would time slower. Under ``autotune(True)`` every
    eligible backend is timed, a bucket whose winner was taken untimed is
    selected again, and the measured winner stays afterwards."""
    assert SpecializedMoERunner.prefer_when_accepted is True
    assert MoERunner.prefer_when_accepted is False  # stock default
    plain = _layer_runner("plain", cost=1.0)
    special = _preferred_runner("specialized", cost=2.0)
    layer = _cpu_layer(special, plain)
    timed = _bench_outside_capture(monkeypatch, [False], _cost)
    act = SimpleNamespace(routing_input_mode=RoutingInputMode.FromLogits, num_tokens=6)

    assert layer(act, None) == "specialized" and timed == []
    assert layer(act, None) == "specialized" and timed == []

    layer.tuner.is_tuning_mode = True  # flashinfer.autotune(True)
    assert layer(act, None) == "plain"
    assert timed == ["specialized", "plain"]
    assert layer(act, None) == "plain" and len(timed) == 2  # measured: kept

    layer.tuner.is_tuning_mode = False
    assert layer(act, None) == "plain" and len(timed) == 2

    # a call the preferred backend does not accept selects as without it
    other = _cpu_layer(_preferred_runner("specialized", cost=2.0), plain)
    other.runners[0].accepts = lambda act, weights: False
    assert other(act, None) == "plain" and len(timed) == 2  # one candidate


@pytest.mark.cpu_only
def test_preferred_backend_is_not_taken_inside_a_capture(monkeypatch):
    """Inside a CUDA-graph capture the preference does not apply: a cached
    winner is replayed, and a first call of a bucket is selected untimed among
    the candidates that do not require an eager selection."""
    plain = _layer_runner("plain", cost=1.0)
    special = _preferred_runner("specialized", cost=2.0)
    layer = _cpu_layer(special, plain)
    capturing = [True]
    timed = _bench_outside_capture(monkeypatch, capturing, _cost)
    act = SimpleNamespace(routing_input_mode=RoutingInputMode.FromLogits, num_tokens=6)

    assert layer(act, None) == "plain" and timed == []
    capturing[0] = False
    assert layer(act, None) == "specialized" and timed == []
    capturing[0] = True
    assert layer(act, None) == "specialized" and timed == []


@pytest.mark.cpu_only
def test_stored_selection_respects_the_kill_switch(monkeypatch):
    """A selection stored outside ``__call__`` (``_prepare_tactic`` /
    ``_run_tactic``, e.g. a tuned expert-parallel tactic) that names the
    specialized backend runs only while the backend accepts the call: with
    ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` nothing is packed or run on it
    and the call is selected among the other backends, as ``__call__`` does."""
    runner, kernel, act, weights = _stub_binding(monkeypatch)
    plain = _layer_runner("plain", cost=1.0)
    forwarded = []
    runner.forward = lambda inputs, **kwargs: forwarded.append(inputs) or "specialized"
    runner.pack_inputs = lambda act, weights: ["packed"]
    layer = _cpu_layer(runner, plain)
    _bench_outside_capture(monkeypatch, [False], lambda r: 1.0)
    act.num_tokens = 6
    assert runner.stored_selection_requires_accepts is True

    assert layer._run_tactic(act, weights, (0, -1)) == "specialized"
    assert forwarded == [["packed"]] and kernel.asked == 1

    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    runner.pack_inputs = lambda act, weights: pytest.fail("packed while disabled")
    layer._prepare_tactic(act, weights, (0, -1))  # prepares nothing
    assert layer._run_tactic(act, weights, (0, -1)) == "plain"
    assert layer.winner_backend == "plain"
    assert len(forwarded) == 1 and kernel.asked == 1  # the kernel was not asked

    # other runners' stored selections are not re-checked
    plain.accepts = lambda act, weights: pytest.fail("a stock runner was asked")
    assert layer._run_tactic(act, weights, (1, -1)) == "plain"
