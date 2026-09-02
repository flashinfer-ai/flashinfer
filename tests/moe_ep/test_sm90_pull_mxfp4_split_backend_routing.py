"""Host contracts for routing the public MXFP4 backend into split runtime."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm90 import (
    fp8_mxfp4_bf16_pull_cutedsl as mxfp4_backend,
)
import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)


def _config() -> Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig:
    return Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        execution_mode="split",
        load_balance_mode="static",
        token_back_mode="epi_warps",
        gate_up_clamp=9.5,
        split_k1_mma_tiler_mnk=(256, 128, 128),
        split_k2_mma_tiler_mnk=(128, 64, 128),
        split_k1_cluster_shape_mnk=(1, 1, 1),
        split_k2_cluster_shape_mnk=(1, 1, 1),
        split_k1_group_hint=240,
        split_k2_group_hint=156,
        split_k1_num_sched_stages=1,
        split_k2_num_sched_stages=3,
        split_k1_sm_count=80,
        split_k2_sm_count=52,
        split_counter_epoch_banks=2,
        split_graph_variant="steady_k3_reset",
        split_enable_iket=True,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )


def _backend():
    backend = mxfp4_backend.Sm90PullMxfp4MegaKernelBackend(_config())
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=2, rank=0, device=0))
    backend._ep_comm_group = object()
    return backend


def _fleet() -> FleetParams:
    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=32,
        token_hidden_size=128,
    )


def test_allocate_forwards_every_split_identity_axis_without_fused_fallback(
    monkeypatch,
) -> None:
    calls = []
    sentinel = object()

    def split_allocate(*args, **kwargs):
        calls.append((args, kwargs))
        return sentinel

    monkeypatch.setattr(
        sm90_mega,
        "get_symm_buffer_for_hopper_mxfp4_split_mega_moe",
        split_allocate,
    )
    monkeypatch.setattr(
        sm90_mega,
        "get_symm_buffer_for_hopper_mxfp4_mega_moe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fused allocator fallback")
        ),
    )
    backend = _backend()
    assert backend._allocate_workspace(_fleet()) is sentinel
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (8, 32, 2, 128, 128, 0, 2)
    assert kwargs == {
        "split_k1_mma_tiler_mnk": (256, 128, 128),
        "split_k2_mma_tiler_mnk": (128, 64, 128),
        "split_k1_cluster_shape_mnk": (1, 1, 1),
        "split_k2_cluster_shape_mnk": (1, 1, 1),
        "split_k1_group_hint": 240,
        "split_k2_group_hint": 156,
        "split_k1_num_sched_stages": 1,
        "split_k2_num_sched_stages": 3,
        "split_k1_sm_count": 80,
        "split_k2_sm_count": 52,
        "split_counter_epoch_banks": 2,
        "split_graph_variant": "steady_k3_reset",
        "gate_up_clamp": 9.5,
        "split_enable_iket": True,
        "routing_profile": SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        "process_group": backend.ep_comm_group,
    }


def test_compute_calls_only_split_launcher_and_preserves_owned_output(
    monkeypatch,
) -> None:
    calls = []

    def split_launch(*args, **kwargs):
        calls.append((args, kwargs))
        return None

    monkeypatch.setattr(sm90_mega, "hopper_mxfp4_split_mega_moe", split_launch)
    monkeypatch.setattr(
        sm90_mega,
        "hopper_mxfp4_mega_moe",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("fused launch fallback")
        ),
    )
    backend = _backend()
    output = torch.empty((3, 128), dtype=torch.bfloat16)
    workspace = SimpleNamespace()
    transformed = ("transformed_l1", "transformed_l2")

    result = backend.compute(workspace, transformed, output=output)
    assert result is output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (output, "transformed_l1", "transformed_l2", workspace)
    assert kwargs == {
        "num_tokens": 3,
        "gate_up_clamp": 9.5,
        "activation_clamp": None,
        "fast_math": True,
    }
