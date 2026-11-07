"""Rows-per-expert bands and explicit baseline selection for MXFP4/MXFP8."""

from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
    StandardMxfp8Mxfp4Backend,
)
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import default_knobs

# Explicit phase-interleave profile in the legacy knob dialect.
PHASE_INTERLEAVE = dict(
    mma_tiler_mnk=(256, 128, 128),
    schedule_mode="phase_interleave",
    group_hint=None,
    flag_batch=1,
    load_balance_mode="atomic_counter",
)


NUM_EXPERTS = 64
TOP_K = 2


def _rows_per_expert(world_size, capacity):
    """Compute expected expert fill for the synthetic routing geometry."""
    return world_size * capacity * TOP_K / NUM_EXPERTS


def _config(
    capacity,
    knobs=None,
    world_size=4,
    *,
    num_experts=NUM_EXPERTS,
    top_k=TOP_K,
    intermediate_size=8192,
    active_clusters=76,
):
    """Resolve native knobs with a controlled occupancy result."""
    backend = StandardMxfp8Mxfp4Backend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=intermediate_size,
            top_k=top_k,
            apply_routing_weights_before_fc2=False,
            knobs=knobs,
        )
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(world_size=world_size, rank=0, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=capacity,
        token_hidden_size=256,
    )
    with mock.patch.object(backend, "_active_clusters", return_value=active_clusters):
        return backend._workspace_config(fleet)


@pytest.mark.parametrize(
    "world_size,capacity",
    [
        # Synthetic 64-expert/top-2 geometry. Edges scale with EP size.
        (1, 1),
        (1, 2047),
        (1, 2048),
        (1, 4095),
        (1, 4096),
        (1, 8192),
        (2, 32),
        (2, 1023),
        (2, 1024),
        (2, 2047),
        (2, 2048),
        (2, 4096),
        (4, 1),
        (4, 64),
        (4, 511),
        (4, 512),
        (4, 1023),
        (4, 1024),
        (4, 2047),
        (4, 2048),
        (4, 4096),
    ],
)
def test_mxfp4_rows_per_expert_bands(world_size, capacity):
    """Exercise both fill boundaries across EP sizes and token-back modes."""
    rows = _rows_per_expert(world_size, capacity)
    config = _config(capacity, world_size=world_size)
    assert config.mma_tiler_mnk == (256, 64 if rows < 64 else 128, 128)
    assert config.token_in_flag_batch == 1
    assert config.schedule_policy == (
        ("grouped", 4096) if rows < 128 else ("phase_interleave", None)
    )
    assert config.token_back_mode == (
        "epi_warps" if capacity < 2048 else "reuse_dispatch_warps"
    )
    assert config.work_id_mode == (
        "grid_stride" if 64 <= rows < 128 else "atomic_counter"
    )
    assert not config.apply_routing_weights_before_fc2


@pytest.mark.parametrize(
    "capacity,tile_n,schedule,claims",
    [
        (32, 64, ("grouped", 4096), "atomic_counter"),
        (127, 64, ("grouped", 4096), "atomic_counter"),
        (128, 128, ("grouped", 4096), "grid_stride"),
        (256, 128, ("grouped", 4096), "grid_stride"),
        (512, 128, ("grouped", 4096), "grid_stride"),
        (513, 128, ("phase_interleave", None), "atomic_counter"),
        (768, 128, ("phase_interleave", None), "atomic_counter"),
        (1023, 128, ("phase_interleave", None), "atomic_counter"),
        (1024, 128, ("phase_interleave", None), "atomic_counter"),
    ],
)
def test_mxfp4_pr7149_defaults(capacity, tile_n, schedule, claims):
    """Keep shallow synthetic workloads grouped until sufficient FC1 depth."""
    config = _config(
        capacity, world_size=2, num_experts=8, top_k=2, intermediate_size=6400
    )
    assert config.mma_tiler_mnk == (256, tile_n, 128)
    assert config.schedule_policy == schedule
    assert config.work_id_mode == claims
    assert config.token_in_flag_batch == 1


def test_mxfp4_phase_override_bypasses_depth_threshold():
    """Honor a full phase profile even for a shallow workload."""
    config = _config(
        512,
        PHASE_INTERLEAVE,
        world_size=2,
        num_experts=8,
        top_k=2,
        intermediate_size=6400,
    )
    assert config.schedule_policy == ("phase_interleave", None)
    assert config.work_id_mode == "atomic_counter"


@pytest.mark.parametrize("capacity", [1024, 2048, 4096])
def test_mxfp4_large_band_is_inherited_plus_phase_interleave(capacity):
    """Retain inherited settings outside the explicit phase profile."""
    # A full legacy dict replaces the band default, so this builds the
    # expected configuration independently of the band under test.
    expected = default_knobs(capacity, dtype="mxfp8")
    expected.update(PHASE_INTERLEAVE)
    assert _config(capacity) == _config(capacity, expected)


@pytest.mark.parametrize("capacity", [64, 256, 512, 2048])
def test_mxfp4_explicit_inherited_profile(capacity):
    """Allow callers to restore the inherited MXFP8 profile."""
    inherited = default_knobs(capacity, dtype="mxfp8")
    inherited.update(mma_tiler_mnk=(256, 128, 128), schedule_mode="grouped")
    config = _config(capacity, inherited)
    assert config.mma_tiler_mnk == (256, 128, 128)
    assert config.schedule_policy == ("grouped", 512)
    assert config.work_id_mode == "atomic_counter"
    assert config.token_in_flag_batch == 4


def test_mxfp4_native_profile_overrides_bands():
    """Honor native tile, scheduler, and flag overrides."""
    config = _config(
        64,
        dict(
            mma_tiler_mnk=(256, 128, 128),
            schedule_policy=("grouped", 512),
            token_in_flag_batch=4,
        ),
    )
    assert config.mma_tiler_mnk == (256, 128, 128)
    assert config.schedule_policy == ("grouped", 512)
    assert config.token_in_flag_batch == 4


@pytest.mark.parametrize("capacity", [32, 128, 768])
@pytest.mark.parametrize("native", [False, True])
def test_mxfp4_phase_only_override_defaults_to_atomic(capacity, native):
    """Resolve either phase-only dialect without static claims or grouped hints."""
    knobs = (
        {"schedule_policy": ("phase_interleave", None)}
        if native
        else {"schedule_mode": "phase_interleave"}
    )
    config = _config(
        capacity, knobs, world_size=2, num_experts=8, top_k=2, intermediate_size=6400
    )
    assert config.schedule_policy == ("phase_interleave", None)
    assert config.work_id_mode == "atomic_counter"


@pytest.mark.parametrize("capacity", [64, 2048])
def test_mxfp8_keeps_inherited_profile(capacity):
    """Keep the MXFP8 backend independent of MXFP4 tuning."""
    from flashinfer.moe_ep import Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
        StandardMxfp8Backend,
    )

    backend = StandardMxfp8Backend(
        Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=8192, top_k=TOP_K
        )
    )
    fleet = FleetParams(
        num_experts=NUM_EXPERTS, max_tokens_per_rank=capacity, token_hidden_size=256
    )
    assert backend._default_knobs(fleet) == default_knobs(capacity, dtype="mxfp8")


@pytest.mark.parametrize("capacity", [64, 256, 512, 2048])
def test_mxfp4_legacy_phase_interleave_is_explicit(capacity):
    """Honor explicit phase scheduling at every tested capacity."""
    config = _config(
        capacity,
        dict(
            schedule_mode="phase_interleave",
            group_hint=None,
            load_balance_mode="atomic_counter",
        ),
    )
    assert config.schedule_policy == ("phase_interleave", None)
    assert config.work_id_mode == "atomic_counter"


@pytest.mark.parametrize(
    "world_size,capacity,active_clusters,intermediate,phase",
    [
        (2, 512, 74, 6400, False),
        (2, 512, 76, 6400, False),
        (2, 513, 76, 6400, True),
        (2, 768, 76, 6400, True),
        (2, 512, 37, 6400, True),
        (2, 1024, 74, 6400, True),
        (2, 1024, 148, 6400, False),
        (2, 512, 74, 12800, True),
        (4, 512, 76, 6400, False),
        (4, 640, 76, 6400, False),
        (4, 641, 76, 6400, True),
        (4, 768, 76, 6400, True),
    ],
)
def test_mxfp4_phase_depends_on_cluster_depth(
    world_size, capacity, active_clusters, intermediate, phase
):
    """Scale the phase threshold with FC1 work and active cluster count."""
    config = _config(
        capacity,
        world_size=world_size,
        num_experts=8,
        top_k=2,
        intermediate_size=intermediate,
        active_clusters=active_clusters,
    )
    assert config.schedule_policy == (
        ("phase_interleave", None) if phase else ("grouped", 4096)
    )


@pytest.mark.parametrize(
    "knobs",
    [
        {"load_balance_mode": "static"},
        {"work_id_mode": "grid_stride"},
        {"mma_tiler_mnk": (256, 256, 128)},
        {"cluster_shape_mnk": (4, 1, 1)},
        {"cluster_shape_mn": (4, 1)},
        {"group_hint": 512},
    ],
)
def test_mxfp4_partial_overrides_keep_grouped(knobs):
    """Preserve compatible grouped defaults for partial geometry or claim overrides."""
    config = _config(
        768, knobs, world_size=2, num_experts=8, top_k=2, intermediate_size=6400
    )
    assert config.schedule_policy == ("grouped", knobs.get("group_hint", 4096))
    if "load_balance_mode" in knobs or "work_id_mode" in knobs:
        assert config.work_id_mode == "grid_stride"
    if "mma_tiler_mnk" in knobs:
        assert config.mma_tiler_mnk == knobs["mma_tiler_mnk"]
    if "cluster_shape_mnk" in knobs or "cluster_shape_mn" in knobs:
        assert config.cluster_shape_mn == (4, 1)


@pytest.mark.parametrize(
    "knobs,message",
    [
        ({"schedule_mode": "phase-interleave"}, "Unsupported schedule_mode"),
        (
            {"schedule_mode": "phase_interleave", "load_balance_mode": "static"},
            "requires load_balance_mode='atomic_counter'",
        ),
        (
            {
                "schedule_policy": ("phase_interleave", None),
                "work_id_mode": "grid_stride",
            },
            "Invalid native standard tuning knobs",
        ),
    ],
)
def test_mxfp4_invalid_schedule_is_config_error(knobs, message):
    """Reject unknown schedules and incompatible explicit claims."""
    with pytest.raises(MoEEpConfigError, match=message):
        _config(
            768, knobs, world_size=2, num_experts=8, top_k=2, intermediate_size=6400
        )


def test_mxfp4_automatic_schedule_requires_device():
    """Require a device only when configuration needs automatic occupancy."""
    backend = StandardMxfp8Mxfp4Backend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=8192, top_k=TOP_K
        )
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(world_size=2, rank=0, auto_bootstrap=False)
    )
    fleet = FleetParams(
        num_experts=NUM_EXPERTS, max_tokens_per_rank=8192, token_hidden_size=256
    )
    with mock.patch.object(torch.cuda, "is_available", return_value=False):
        with pytest.raises(MoEEpConfigError, match="requires CUDA"):
            backend._workspace_config(fleet)
        backend._config.knobs = {"schedule_mode": "grouped"}
        assert backend._workspace_config(fleet).schedule_policy == ("grouped", 4096)


@pytest.mark.parametrize("mismatch", ["config", "clusters"])
def test_mxfp4_rejects_rank_divergence_before_pool_lookup(mismatch):
    """Reject differing configurations or occupancy before allocating or reusing storage."""
    import dataclasses
    import torch.distributed as dist
    from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
        _StandardBlockScaledBackend,
    )

    backend = StandardMxfp8Mxfp4Backend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=8192, top_k=TOP_K
        )
    )
    backend.bind_ep_bootstrap(
        BootstrapConfig(world_size=2, rank=0, auto_bootstrap=False)
    )
    backend._ep_comm_group = object()
    fleet = FleetParams(
        num_experts=NUM_EXPERTS, max_tokens_per_rank=4096, token_hidden_size=256
    )

    def gather(reports, report, **kwargs):
        """Simulate a peer with different valid launch settings."""
        assert kwargs["group"] is backend._ep_comm_group
        error, proposal = report
        assert error is None
        config, clusters = proposal
        peer = (
            (dataclasses.replace(config, token_in_flag_batch=4), clusters)
            if mismatch == "config"
            else (config, clusters - 1)
        )
        reports[:] = [report, (None, peer)]

    with (
        mock.patch.object(backend, "_active_clusters", return_value=76),
        mock.patch.object(dist, "all_gather_object", side_effect=gather),
        mock.patch.object(_StandardBlockScaledBackend, "prepare_workspace") as prepare,
    ):
        with pytest.raises(MoEEpConfigError, match="differs across EP ranks"):
            backend.prepare_workspace(backend._ep_bootstrap, fleet)
        prepare.assert_not_called()


@pytest.mark.parametrize("hint", [None, 8192])
def test_mxfp4_legacy_phase_preserves_explicit_hint(hint):
    """Explicit phase hints take precedence over the inferred default."""
    config = _config(768, {"schedule_mode": "phase_interleave", "group_hint": hint})
    assert config.schedule_policy == ("phase_interleave", hint)


@pytest.mark.parametrize("failing_rank", [0, 1])
@pytest.mark.parametrize("method", ["_workspace_config", "_active_clusters"])
@pytest.mark.parametrize("operation", ["validate_init", "prepare_workspace"])
def test_mxfp4_workspace_failure_is_collective(failing_rank, method, operation):
    """Validation and preparation propagate both local and peer failures."""
    from contextlib import nullcontext

    import torch.distributed as dist
    from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
        _StandardBlockScaledBackend,
    )

    backend = StandardMxfp8Mxfp4Backend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=6400, top_k=2)
    )
    bootstrap = BootstrapConfig(world_size=2, rank=0, auto_bootstrap=False)
    backend.bind_ep_bootstrap(bootstrap)
    backend._ep_comm_group = object()
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=768, token_hidden_size=256)
    failure = f"RuntimeError: {method} failed"

    def gather(reports, report, **kwargs):
        """Return the same failure report to both simulated ranks."""
        assert kwargs["group"] is backend._ep_comm_group
        error, proposal = report
        assert error == (failure if failing_rank == 0 else None)
        assert (proposal is None) == (operation == "validate_init" or failing_rank == 0)
        reports[:] = [report, (failure, None) if failing_rank == 1 else (None, None)]

    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl.validate_mega_arch"
        ),
        mock.patch.object(backend, "_active_clusters", return_value=76),
        mock.patch.object(dist, "all_gather_object", side_effect=gather) as collective,
        mock.patch.object(_StandardBlockScaledBackend, "prepare_workspace") as prepare,
        mock.patch.object(backend, method, side_effect=RuntimeError(f"{method} failed"))
        if failing_rank == 0
        else nullcontext(),
    ):
        with pytest.raises(MoEEpConfigError, match=f"rank {failing_rank}: {failure}"):
            getattr(backend, operation)(bootstrap, fleet)
        collective.assert_called_once()
        prepare.assert_not_called()


@pytest.mark.parametrize("world_size,has_group", [(1, True), (2, False)])
def test_mxfp4_validation_without_collective(world_size, has_group):
    """Keep single-rank and pre-bootstrap validation local."""
    import torch.distributed as dist
    from flashinfer.moe_ep.backends.mega.kernel.sm100.standard_cutedsl import (
        _StandardBlockScaledBackend,
    )

    backend = StandardMxfp8Mxfp4Backend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=256, top_k=2)
    )
    bootstrap = BootstrapConfig(world_size=world_size, rank=0, auto_bootstrap=False)
    backend.bind_ep_bootstrap(bootstrap)
    backend._ep_comm_group = object() if has_group else None
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=32, token_hidden_size=256)
    with (
        mock.patch.object(dist, "is_initialized", return_value=False),
        mock.patch.object(dist, "all_gather_object") as collective,
        mock.patch.object(
            _StandardBlockScaledBackend,
            "validate_init",
            side_effect=RuntimeError("local validation failed"),
        ) as validate,
    ):
        with pytest.raises(RuntimeError, match="local validation failed"):
            backend.validate_init(bootstrap, fleet)
        validate.assert_called_once_with(bootstrap, fleet)
        collective.assert_not_called()
