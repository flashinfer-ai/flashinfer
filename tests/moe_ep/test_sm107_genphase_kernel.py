"""Native Rubin correctness for repeated launches, masked routes, and zero input."""

import pytest
import torch


@pytest.mark.arch_rubin
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("early,ikr", [(True, False), (False, False), (True, True)])
def test_genphase_repeated_launches(monkeypatch, kind, early, ikr):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from tests.moe_ep.test_sm107_kernel_boundaries import _run_case

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    _run_case(
        kind,
        5,
        3,
        dict(
            kernel_variant="genphase",
            cluster_shape_mn=(4, 1),
            fc2_use_bulk=True,
            apply_routing_weights_before_fc2=early,
            reduce_topk_in_kernel=ikr,
        ),
    )


@pytest.mark.arch_rubin
@pytest.mark.parametrize("kind", ["mxfp8_e4m3", "mxfp4_mxfp8"])
@pytest.mark.parametrize("capacity", [1, 17, 65, 129])
@pytest.mark.parametrize("tile_n", [64, 128])
@pytest.mark.parametrize(
    "schedule, work_id",
    [
        (("grouped", None), "grid_stride"),
        (("phase_interleave", None), "atomic_counter"),
    ],
)
def test_genphase_live_rows_and_sf_tile_boundaries(
    monkeypatch, kind, capacity, tile_n, schedule, work_id
):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("Requires SM107")
    from tests.moe_ep.test_sm107_kernel_boundaries import _run_case

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    # Capacity one also verifies that Gather4 retains a 2D TMA descriptor.
    # Route every token to all eight experts, crossing both the N64 tile edge
    # and the 128-row scale-factor block edge. The shared runner also changes
    # inputs, masks routes and sends zeros on consecutive workspace replays.
    _run_case(
        kind,
        capacity,
        8,
        dict(
            kernel_variant="genphase",
            mma_tiler_mnk=(256, tile_n, 128),
            cluster_shape_mn=(4, 1),
            fc2_use_bulk=True,
            fc2_tma_stages=1,
            epi_flag_batches=(1, 4),
            schedule_policy=schedule,
            work_id_mode=work_id,
        ),
    )
