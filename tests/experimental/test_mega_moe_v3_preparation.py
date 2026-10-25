"""Host-side contract of the generated MegaMoE v3 pipeline preparation (no GPU required)."""

import pytest

from flashinfer.experimental.mega_moe_v3 import preparation

GEOMETRY = dict(top_k=6, hidden=5120, routed_weight_dtype="fp4", activation_clamp=10.0)


@pytest.mark.parametrize(
    ("logical", "physical"),
    [(2304, 2304), (576, 640), (288, 384), (640, 640), (384, 384), (32, 128)],
)
def test_logical_intermediate_pads_to_the_next_multiple_of_128(logical, physical):
    assert preparation.physical_intermediate(logical) == physical


@pytest.mark.parametrize("logical", [0, 16, 100, 577, -32])
def test_intermediate_must_be_a_positive_multiple_of_the_scale_group(logical):
    with pytest.raises(ValueError, match="multiple of 32"):
        preparation.physical_intermediate(logical)


def test_route_args_name_the_physical_width_and_normalise_the_key_types():
    args = preparation.pipeline_route_args(
        dict(
            GEOMETRY,
            num_tokens=4,
            num_experts=384,
            intermediate=576,
            activation_clamp=10,
        )
    )
    assert args == dict(
        num_tokens=4,
        num_experts=384,
        top_k=6,
        hidden=5120,
        intermediate=640,
        routed_weight_dtype="fp4",
        activation_clamp=10.0,
    )
    assert isinstance(args["activation_clamp"], float)
    # The routed format defaults to FP4 and a missing clamp stays None.
    bare = preparation.pipeline_route_args(
        dict(num_tokens=1, num_experts=96, top_k=6, hidden=5120, intermediate=2304)
    )
    assert bare["routed_weight_dtype"] == "fp4" and bare["activation_clamp"] is None


def test_expert_window_defaults_and_bounds():
    assert preparation.expert_window(dict(num_experts=384)) == (0, 384, 384)
    assert preparation.expert_window(
        dict(num_experts=96, local_expert_offset=288, num_global_experts=384)
    ) == (288, 96, 384)
    for bad in (
        dict(num_experts=96, local_expert_offset=-1, num_global_experts=384),
        dict(num_experts=96, local_expert_offset=289, num_global_experts=384),
        dict(num_experts=0, local_expert_offset=0, num_global_experts=384),
        dict(num_experts=48, local_expert_offset=0, num_global_experts=47),
    ):
        with pytest.raises(ValueError, match="expert window"):
            preparation.expert_window(bad)


@pytest.mark.parametrize(
    ("metadata", "num_tokens", "expected"),
    [
        # Forms a route names explicitly. The rendered TP/EP rows use per-block
        # readiness with the slot list, and without it at one token; a row may
        # also name grid readiness (no readiness words, no slot list).
        (
            {"block_m": 16, "readiness": "block", "slot_list": False},
            1,
            (16, True, False),
        ),
        (
            {"block_m": 16, "readiness": "block", "slot_list": True},
            4,
            (16, True, True),
        ),
        (
            {"block_m": 16, "readiness": "grid", "slot_list": False},
            4,
            (16, False, False),
        ),
        (
            {"block_m": 16, "readiness": "grid", "slot_list": False},
            512,
            (16, False, False),
        ),
        (
            {"block_m": 32, "readiness": "grid", "slot_list": False},
            1024,
            (32, False, False),
        ),
        (
            {"block_m": 128, "readiness": "grid", "slot_list": False},
            4096,
            (128, False, False),
        ),
        # Without an explicit slot_list, "block" implies the slot list except at one token.
        ({"block_m": 16, "readiness": "block"}, 1, (16, True, False)),
        ({"block_m": 16, "readiness": "block"}, 16, (16, True, True)),
        # The shipped routes carry no metadata: the original heuristics.
        ({}, 1, (16, True, False)),
        ({}, 4, (16, False, False)),
        ({}, 16, (16, True, True)),
        ({}, 1024, (32, True, True)),
        ({}, 4096, (128, True, True)),
    ],
)
def test_source_schedule_form(metadata, num_tokens, expected):
    assert (
        preparation.source_schedule_form(metadata, num_tokens, 384, 6, "fp4")
        == expected
    )


def test_source_schedule_form_rejects_unknown_forms():
    with pytest.raises(ValueError, match="readiness"):
        preparation.source_schedule_form({"readiness": "eager"}, 16, 384, 6, "fp4")
    # The local-expert heuristic would pick 192 rows at 4096 tokens for 96 experts.
    with pytest.raises(ValueError, match="tile height 192"):
        preparation.source_schedule_form({}, 4096, 96, 6, "fp4")
    with pytest.raises(ValueError, match="tile height 64"):
        preparation.source_schedule_form({"block_m": 64}, 512, 96, 6, "fp4")
    assert (
        preparation.source_schedule_form({"block_m": 128}, 4096, 96, 6, "fp4")[0] == 128
    )


def test_workspace_words_for_block_readiness_without_a_slot_list_at_one_token():
    """The one-token expert-parallel rows keep per-block readiness and the
    per-expert claim scratch: one readiness word per source block on top of
    the two counters and two mask words, and the scratch lives in the counter
    storage (``2 * E`` words before the gates), not in a slot list."""
    for experts in (384, 96, 48):
        blocks, words = preparation.source_workspace_words(
            experts, 1, 6, 16, True, total_m_tiles=0, k2_tiles=18
        )
        assert blocks == experts + 1
        assert words == 2 + 3 * blocks
        grid_blocks, grid_words = preparation.source_workspace_words(
            experts, 4, 6, 16, False, total_m_tiles=0, k2_tiles=18
        )
        assert grid_blocks == experts + 2 and grid_words == 2 + 2 * grid_blocks
    # The original per-tile bound still wins when it is larger.
    assert (
        preparation.source_workspace_words(384, 4096, 6, 128, True, 4000, 9)[1] == 36000
    )


@pytest.mark.parametrize(("logical", "physical"), [(576, 640), (288, 384)])
def test_w1_launch_view_hands_the_expert_map_the_logical_rows_under_the_padded_pitch(
    logical, physical
):
    """A route with ``w1_expert_map`` takes B1 as a strided per-expert view of
    the logical gate / up rows (no copy); every other route takes the dense
    two-dimensional map."""
    torch = pytest.importorskip("torch")
    experts, k_bytes = 3, 64
    b1 = torch.arange(experts * 2 * physical * k_bytes, dtype=torch.int32).view(
        experts, 2 * physical, k_bytes
    )
    view = preparation.w1_launch_view(b1, logical, True)
    assert view.shape == (experts, 2 * logical, k_bytes)
    assert view.stride() == (2 * physical * k_bytes, k_bytes, 1)
    assert view.data_ptr() == b1.data_ptr() and not view.is_contiguous()
    assert torch.equal(view[1, 2 * logical - 1], b1[1, 2 * logical - 1])
    dense = preparation.w1_launch_view(b1, logical, False)
    assert dense.shape == (experts * 2 * physical, k_bytes) and dense.is_contiguous()
    # Equal widths: the expert map is the whole tensor.
    same = preparation.w1_launch_view(b1, physical, True)
    assert same.shape == b1.shape and same.is_contiguous()


def test_dispatch_reference_filters_the_window_and_keeps_the_tile_lists():
    torch = pytest.importorskip("torch")
    ids = torch.tensor([[0, 95, 96, 191, 383, -1], [96, 96, 97, -1, 300, 5]])
    ref = preparation._mega_dispatch_reference(ids, 96, local_expert_offset=96)
    counts = torch.zeros(96, dtype=torch.int32)
    counts[0], counts[1], counts[95] = 3, 1, 1
    assert torch.equal(ref["counts"], counts)
    assert ref["M_total"] == 3 * 256 and ref["total_m_tiles"] == 6
    assert (
        ref["offsets"][0] == 0
        and ref["offsets"][1] == 256
        and ref["offsets"][95] == 512
    )
    assert ref["tile_expert"].tolist() == [0, 0, 1, 1, 95, 95]
    assert ref["tile_m_local"].tolist() == [0, 128, 0, 128, 0, 128]
    assert (
        ref["tile_expert"].dtype == torch.int32 and ref["offsets"].dtype == torch.int32
    )


def test_dispatch_reference_all_remote_routes_no_row():
    """Every route held by another rank: no counts, no tiles, zero rows; the
    preparation then keeps one 256-row aligned block so the pools' tensor
    maps stay valid (``max(M_total, ALIGN_M)``)."""
    torch = pytest.importorskip("torch")
    ids = torch.randint(0, 96, (16, 6), dtype=torch.int64)
    ref = preparation._mega_dispatch_reference(ids, 96, local_expert_offset=96)
    assert ref["M_total"] == 0 and ref["total_m_tiles"] == 0
    assert ref["counts"].count_nonzero().item() == 0
    assert ref["tile_expert"].shape == (0,) and ref["tile_m_local"].shape == (0,)
    assert max(ref["M_total"], preparation.ALIGN_M) == 256
    masked = preparation._mega_dispatch_reference(torch.full((4, 6), -1), 48)
    assert masked["M_total"] == 0 and masked["total_m_tiles"] == 0
