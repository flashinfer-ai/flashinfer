"""W4A16's existing Mega autotune/cache lifecycle, without CUDA launches."""

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_nvfp4_bf16_cutedsl.backend import (
    W4A16CutedslMegaKernelBackend,
)


@pytest.fixture
def public():
    pytest.importorskip("cutlass")
    from flashinfer.moe_ep.cute_dsl.megamoe import nvfp4_w4a16 as cutedsl_megamoe

    return cutedsl_megamoe


@pytest.fixture
def factory(public, monkeypatch, tmp_path):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import knob_cache

    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    monkeypatch.setattr(knob_cache, "_current_device_name", lambda: "test-device")
    result = public.get_symm_buffer_for_w4a16_mega_moe
    with (
        mock.patch(
            f"{result.__module__}.sym_zeros",
            side_effect=lambda shape, dtype: torch.zeros(shape, dtype=dtype),
        ),
        mock.patch(f"{result.__module__}.free_sym_tensor"),
    ):
        yield result


def _backend(knobs):
    backend = W4A16CutedslMegaKernelBackend(
        Sm100_Bf16_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            64, 2, gate_up_clamp=1.5, knobs=knobs
        )
    )
    backend.bind_ep_bootstrap(BootstrapConfig(world_size=1, rank=0))
    return backend


def _record(knobs, *, max_tokens=512, dtype="w4a16"):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.knob_cache import (
        record_knobs,
    )

    record_knobs(
        knobs,
        dtype=dtype,
        world_size=1,
        hidden=64,
        intermediate=64,
        num_experts=4,
        topk=2,
        max_tokens=max_tokens,
    )


@pytest.mark.parametrize("knobs", (None, {}, {"flag_batch": 8}, "auto"))
def test_backend_allocation_resolves_only_dict_knobs(public, knobs):
    backend = _backend(knobs)
    fleet = SimpleNamespace(
        num_experts=4, max_tokens_per_rank=257, token_hidden_size=64
    )
    with mock.patch.object(public, "get_symm_buffer_for_w4a16_mega_moe") as allocate:
        backend._allocate_workspace(fleet)
    assert allocate.call_args.kwargs["knobs"] == (
        knobs if isinstance(knobs, dict) else None
    )
    assert allocate.call_args.kwargs["gate_up_clamp"] == 1.5
    assert backend._autotune_pending == (knobs == "auto")
    if knobs == "auto":
        # No current-device query or shared workspace pool for mutable tuning.
        assert backend._workspace_pool_key(fleet) is None


def test_config_rejects_unknown_tuning_mode():
    with pytest.raises(ValueError, match="dict, 'auto', or None"):
        _backend("everything")


def test_auto_tunes_once_then_uses_the_normal_full_forward(public):
    backend = _backend("auto")
    output = torch.empty(3, 64, dtype=torch.bfloat16)
    weights = (object(), object())
    workspace = object()
    winner = {"mma_tiler_mnk": (256, 64, 256), "flag_batch": 8}
    events = []
    with (
        mock.patch.object(
            public,
            "autotune_w4a16_mega_moe",
            side_effect=lambda *a, **k: (events.append("tune") or winner),
        ) as tune,
        mock.patch.object(
            public,
            "w4a16_mega_moe",
            side_effect=lambda *a, **k: events.append("forward"),
        ),
    ):
        assert backend.compute(workspace, weights, output=output) is output
        backend.compute(workspace, weights, output=output)
    assert events == ["tune", "forward", "forward"]
    tune.assert_called_once_with(
        output, *weights, workspace, num_tokens=3, gate_up_clamp=1.5
    )
    assert backend._autotune_winner == winner and not backend._autotune_pending


def test_failed_auto_remains_pending_and_does_not_launch_normal_forward(public):
    backend = _backend("auto")
    with (
        mock.patch.object(
            public,
            "autotune_w4a16_mega_moe",
            side_effect=RuntimeError("capture or tune rejected"),
        ),
        mock.patch.object(public, "w4a16_mega_moe") as launch,
        pytest.raises(RuntimeError, match="capture or tune rejected"),
    ):
        backend.compute(object(), (object(), object()), output=torch.empty(0, 64))
    assert backend._autotune_pending and backend._autotune_winner is None
    launch.assert_not_called()


@pytest.mark.parametrize("capacity,expected_flag", ((256, 4), (257, 8)))
def test_cache_resolution_uses_buffer_capacity_and_preserves_return_mode(
    factory, capacity, expected_flag
):
    _record(
        {"flag_batch": 4, "token_back_mode": "reuse_dispatch_warps"}, max_tokens=256
    )
    _record(
        {
            "flag_batch": 8,
            "token_back_mode": "reuse_dispatch_warps",
            "gate_up_clamp": 99.0,
        },
        max_tokens=512,
    )
    workspace = factory(4, capacity, 2, 64, 64, 0, 1, gate_up_clamp=1.5)
    try:
        config = workspace._frontend.config
        assert config.flag_batch == expected_flag
        assert config.token_back_mode == "reuse_dispatch_warps"
        assert config.gate_up_clamp == 1.5
        assert not config.in_kernel_fc2_reduce and not config.apply_topk_in_fc1
    finally:
        workspace.destroy()


@pytest.mark.parametrize(
    "knobs",
    (
        {},
        {
            "flag_batch": 8,
            "token_back_mode": "reuse_dispatch_warps",
            "gate_up_clamp": 2.0,
        },
    ),
)
def test_explicit_knobs_bypass_cache_and_preserve_final_override(factory, knobs):
    from flashinfer.moe_ep.kernel_src import cutedsl_megamoe as knob_cache

    with mock.patch.object(knob_cache, "resolve_knobs") as lookup:
        workspace = factory(
            4,
            257,
            2,
            64,
            64,
            0,
            1,
            gate_up_clamp=1.5,
            token_back_mode="epi_warps",
            knobs=knobs,
        )
    try:
        lookup.assert_not_called()
        config = workspace._frontend.config
        assert config.flag_batch == knobs.get("flag_batch", 1)
        assert config.token_back_mode == knobs.get("token_back_mode", "epi_warps")
        assert config.gate_up_clamp == knobs.get("gate_up_clamp", 1.5)
    finally:
        workspace.destroy()


@pytest.mark.parametrize("mode", ("epi_warps", "reuse_dispatch_warps"))
def test_named_token_return_overrides_cached_choice(factory, mode):
    _record(
        {
            "token_back_mode": "reuse_dispatch_warps"
            if mode == "epi_warps"
            else "epi_warps"
        }
    )
    workspace = factory(4, 257, 2, 64, 64, 0, 1, token_back_mode=mode)
    try:
        assert workspace._frontend.config.token_back_mode == mode
    finally:
        workspace.destroy()


def test_other_dtype_cache_does_not_replace_w4a16_default(factory):
    _record({"flag_batch": 16}, dtype="nvfp4")
    workspace = factory(4, 257, 2, 64, 64, 0, 1)
    try:
        config = workspace._frontend.config
        assert (config.flag_batch, config.group_hint, config.epi_flag_batch) == (
            4,
            512,
            (2, 4),
        )
    finally:
        workspace.destroy()


def test_apply_knobs_capture_guard_preserves_existing_config(factory):
    workspace = factory(4, 257, 2, 64, 64, 0, 1, knobs={})
    frontend = workspace._frontend
    original = frontend.config
    try:
        with (
            mock.patch(
                "flashinfer.moe_ep.kernel_src.cutedsl_megamoe.ensure_not_capturing",
                side_effect=RuntimeError("capture"),
            ),
            mock.patch.object(frontend, "_release_workspace") as release,
            pytest.raises(RuntimeError, match="capture"),
        ):
            frontend.apply_knobs({"flag_batch": 8})
        assert frontend.config is original
        release.assert_not_called()
    finally:
        workspace.destroy()


def test_cached_c1_geometry_and_cluster_change_preserve_lifecycle(factory):
    knobs = {
        "mma_tiler_mnk": [128, 64, 256],
        "cluster_shape_mnk": [1, 1, 1],
        "use_2cta_instrs": False,
        "flag_batch": 8,
        "token_back_mode": "reuse_dispatch_warps",
    }
    _record(knobs)
    workspace = factory(4, 257, 2, 64, 64, 0, 1, gate_up_clamp=1.5)
    frontend = workspace._frontend
    try:
        config = frontend.config
        assert config.mma_tiler_mnk == (128, 64, 256)
        assert config.cluster_shape_mnk == (1, 1, 1)
        assert not config.use_2cta_instrs
        assert config.flag_batch == 8
        assert config.token_back_mode == "reuse_dispatch_warps"
        first_key = frontend._compile_key()
        hash(first_key)
        with mock.patch.object(frontend, "_release_workspace") as release:
            frontend.apply_knobs({"cluster_shape_mnk": [2, 1, 1]})
        release.assert_called_once_with()
        assert frontend._compile_key() != first_key
        assert frontend._mega is None and frontend._mega_key is None
        assert frontend.config.mma_tiler_mnk == (128, 64, 256)
        assert frontend.config.cluster_shape_mnk == (2, 1, 1)
        assert frontend.config.gate_up_clamp == 1.5
        assert frontend.config.token_back_mode == "reuse_dispatch_warps"
    finally:
        workspace.destroy()


def test_catalog_winner_restores_scheduler_depth_and_invalidates_compile(factory):
    from flashinfer.moe_ep.cute_dsl.megamoe.nvfp4_w4a16 import (
        w4a16_candidates,
    )

    candidates = w4a16_candidates()
    assert candidates
    workspace = factory(4, 257, 2, 64, 64, 0, 1, gate_up_clamp=1.5, knobs={})
    frontend = workspace._frontend
    try:
        for winner in candidates:
            frontend.apply_knobs(dict(winner, num_sched_stages=3))
            assert frontend.config.num_sched_stages == 3
            stage3_key = frontend._compile_key()
            frontend._mega, frontend._mega_key = object(), stage3_key
            with mock.patch.object(frontend, "_release_workspace") as release:
                frontend.apply_knobs(winner)
            release.assert_called_once_with()
            assert frontend.config.num_sched_stages == 2
            assert frontend._compile_key() != stage3_key
            assert frontend._mega is None and frontend._mega_key is None
            assert frontend.config.gate_up_clamp == 1.5
    finally:
        workspace.destroy()
