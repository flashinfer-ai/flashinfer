"""Runner-level pack contract: routing mode and per-token scale rejection.

CPU-only. Every runner shares ``MoERunner._validate_pack_contract`` and the
``supports_per_token_scale`` gate, so a backend can neither silently drop
``per_token_scale`` nor forward a routing mode its kernel cannot execute.
"""

from __future__ import annotations

import inspect

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    QuantConfig,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS
from flashinfer.fused_moe.runners import (
    CuteDslRunner,
    MoERunner,
    PrimsTsRunner,
    TrtllmFp4RoutedRunner,
)

_PER_TOKEN_RUNNERS = {CuteDslRunner, PrimsTsRunner, TrtllmFp4RoutedRunner}
_RUNNERS = sorted(set(_BACKEND_RUNNERS.values()), key=lambda cls: cls.__name__)
_CONFIG_FOR_RUNNER = {runner: cfg for cfg, runner in _BACKEND_RUNNERS.items()}


def _bare_runner(runner_cls, config):
    runner = runner_cls.__new__(runner_cls)
    runner.config = config
    runner.device = torch.device("cpu")
    return runner


def _config_for(runner_cls, *, per_token_scale=None):
    pair = runner_cls.supported_quant_variants[0]
    by_quant = runner_cls.supported_activation_classes_by_quant
    activations = (
        by_quant[pair] if by_quant else runner_cls.supported_activation_classes
    )
    return MoEConfig(
        routing=RoutingConfig(num_experts=32, top_k=2),
        quant=QuantConfig(
            weight=pair[0], activation=pair[1], per_token_scale=per_token_scale
        ),
        experts=ExpertConfig(intermediate_size=512),
        activation=activations[0](),
        backend=BackendOptions((_CONFIG_FOR_RUNNER[runner_cls](),)),
    )


def _pack(mode, *, per_token_scale=None):
    hidden = torch.zeros(4, 64, dtype=torch.bfloat16)
    if mode is RoutingInputMode.FromLogits:
        return MoEActivationPack(
            hidden,
            None,
            routing_input_mode=mode,
            routing_logits=torch.zeros(4, 32, dtype=torch.float32),
            per_token_scale=per_token_scale,
        )
    return MoEActivationPack(
        hidden,
        None,
        torch.zeros(4, 2, dtype=torch.int32),
        torch.ones(4, 2, dtype=torch.float32),
        routing_input_mode=mode,
        per_token_scale=per_token_scale,
    )


def test_per_token_scale_opt_in_set():
    """Only the runners whose kernels read the per-token scale opt in."""
    assert {r for r in _RUNNERS if r.supports_per_token_scale} == _PER_TOKEN_RUNNERS


@pytest.mark.parametrize("runner_cls", _RUNNERS, ids=lambda c: c.__name__)
def test_every_pack_inputs_checks_the_contract(runner_cls):
    """The shared check must be the first thing ``pack_inputs`` does."""
    source = inspect.getsource(runner_cls.pack_inputs)
    if "_validate_pack_contract" not in source:
        # cuTile runners validate through a shared helper.
        source = inspect.getsource(runner_cls._validate_inputs)
    assert "self._validate_pack_contract(act)" in source


@pytest.mark.parametrize(
    "runner_cls",
    [r for r in _RUNNERS if not r.supports_per_token_scale],
    ids=lambda c: c.__name__,
)
def test_declared_per_token_scale_is_rejected(runner_cls):
    runner = _bare_runner(runner_cls, _config_for(runner_cls, per_token_scale=True))
    with pytest.raises(NotImplementedError, match="per_token_scale=True"):
        MoERunner._check_support(runner)


@pytest.mark.parametrize(
    "runner_cls",
    [r for r in _RUNNERS if not r.supports_per_token_scale],
    ids=lambda c: c.__name__,
)
def test_pack_per_token_scale_is_rejected(runner_cls):
    runner = _bare_runner(runner_cls, _config_for(runner_cls))
    mode = runner_cls.supported_routing_modes[0]
    with pytest.raises(ValueError, match="per_token_scale"):
        runner._validate_pack_contract(
            _pack(mode, per_token_scale=torch.ones(4, dtype=torch.float32))
        )
    runner._validate_pack_contract(_pack(mode))


@pytest.mark.parametrize("runner_cls", _RUNNERS, ids=lambda c: c.__name__)
def test_unsupported_routing_mode_is_rejected(runner_cls):
    unsupported = [
        m for m in RoutingInputMode if m not in runner_cls.supported_routing_modes
    ]
    if not unsupported:
        pytest.skip(f"{runner_cls.__name__} supports every routing mode")
    runner = _bare_runner(runner_cls, _config_for(runner_cls))
    with pytest.raises(NotImplementedError, match="routing_input_mode"):
        runner._validate_pack_contract(_pack(unsupported[0]))
