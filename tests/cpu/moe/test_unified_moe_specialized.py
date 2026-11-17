"""CPU-only checks of the unified MoE API's ``specialized`` backend framework.

Kernel-independent: kernel discovery, every registered kernel's declarations, the
runner's declarations and kernel binding, and the backend's registration. Each
kernel's own guards, numerics, kill switch and CUDA-graph behavior are tested in
``tests/moe/test_unified_moe_specialized_<kernel>.py``.
"""

from __future__ import annotations

import json
import pkgutil
import sys

import pytest
import torch

import flashinfer.utils as fi_utils
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MoEConfig,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SpecializedMoEConfig,
)
from flashinfer.fused_moe import api as moe_api
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
