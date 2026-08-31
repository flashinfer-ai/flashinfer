"""Exact-device and explicit-tactic bypass contracts for H200 MXFP4 tuning."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import FleetParams
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90PullMxfp4MegaKernelBackend,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    tuner as backend_tuner,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    hopper_mxfp4,
    hopper_mxfp4_split,
    mxfp4_tuner,
)


def _mock_cuda_device(
    monkeypatch,
    *,
    name: str,
    capability: tuple[int, int],
    sm_count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sm_count),
    )


def test_exact_standard_h200_is_accepted(monkeypatch) -> None:
    _mock_cuda_device(
        monkeypatch,
        name="NVIDIA H200",
        capability=(9, 0),
        sm_count=132,
    )
    mxfp4_tuner.require_hopper_mxfp4_tuning_device()


@pytest.mark.parametrize(
    ("name", "capability", "sm_count"),
    [
        ("NVIDIA H200 NVL", (9, 0), 132),
        ("NVIDIA H100 80GB HBM3", (9, 0), 132),
        ("NVIDIA H200", (9, 1), 132),
        ("NVIDIA H200", (9, 0), 114),
    ],
)
def test_non_manifest_device_is_rejected(
    monkeypatch,
    name: str,
    capability: tuple[int, int],
    sm_count: int,
) -> None:
    _mock_cuda_device(
        monkeypatch,
        name=name,
        capability=capability,
        sm_count=sm_count,
    )
    with pytest.raises(RuntimeError, match="only for standard NVIDIA H200"):
        mxfp4_tuner.require_hopper_mxfp4_tuning_device()


def test_no_cuda_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        mxfp4_tuner.require_hopper_mxfp4_tuning_device()


def _resolver_kwargs() -> dict:
    return {
        "world_size": 4,
        "hidden": 7168,
        "intermediate": 3072,
        "num_total_experts": 384,
        "num_topk": 6,
        "num_max_tokens": 512,
        "gate_up_clamp": 10.0,
    }


def test_none_resolvers_require_certified_device(monkeypatch) -> None:
    guard = mock.Mock(side_effect=RuntimeError("device guard"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        guard,
    )
    with pytest.raises(RuntimeError, match="device guard"):
        hopper_mxfp4._resolve_mxfp4_knobs(None, **_resolver_kwargs())
    with pytest.raises(RuntimeError, match="device guard"):
        hopper_mxfp4_split._resolve_mxfp4_split_tactic(
            None,
            **_resolver_kwargs(),
        )
    assert guard.call_count == 2


@pytest.mark.parametrize("execution_mode", ["fused", "split"])
def test_backend_auto_allocation_requires_certified_device(
    monkeypatch,
    execution_mode: str,
) -> None:
    guard = mock.Mock(side_effect=RuntimeError("device guard"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        guard,
    )
    config = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode=execution_mode,
        knobs="auto",
    )
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = Sm90PullMxfp4MegaKernelBackend(config)
    backend._ep_bootstrap = object()
    backend._ep_rank = 0
    backend._ep_world_size = 1
    backend._ep_comm_group = None
    fleet = FleetParams(
        num_experts=8,
        max_tokens_per_rank=8,
        token_hidden_size=128,
    )
    with pytest.raises(RuntimeError, match="device guard"):
        backend._allocate_workspace(fleet)
    guard.assert_called_once_with()


def test_complete_explicit_tactics_bypass_device_guard(monkeypatch) -> None:
    guard = mock.Mock(side_effect=AssertionError("guard must not run"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        guard,
    )
    fused = mxfp4_tuner.hopper_mxfp4_candidates(execution_mode="fused")[0]
    split = mxfp4_tuner.hopper_mxfp4_candidates(execution_mode="split")[0]
    assert (
        hopper_mxfp4._resolve_mxfp4_knobs(
            fused,
            **_resolver_kwargs(),
        )
        == fused
    )
    assert (
        hopper_mxfp4_split._resolve_mxfp4_split_tactic(
            split,
            **_resolver_kwargs(),
        )
        == split
    )
    guard.assert_not_called()


def test_offline_cli_checks_device_before_input_creation(monkeypatch) -> None:
    guard = mock.Mock(side_effect=RuntimeError("device guard"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_tuning_device",
        guard,
    )
    with pytest.raises(RuntimeError, match="device guard"):
        backend_tuner.tune_one(
            SimpleNamespace(),
            rank=0,
            world_size=4,
            max_tokens=8,
        )
    guard.assert_called_once_with()
