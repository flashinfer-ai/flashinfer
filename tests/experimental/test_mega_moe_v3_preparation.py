"""Host-side contract of the generated MegaMoE v3 pipeline preparation.

No GPU is required except by the binding tests, which allocate a small
route's workspace on the device and skip without one.
"""

import pytest

from flashinfer.experimental.mega_moe_v3 import preparation
from flashinfer.experimental.mega_moe_v3.preparation import SF_GRAN

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


# The source form the short TP/EP rows state (one and four tokens: per-block
# readiness, no slot list, 16-row tiles).
SHORT_ROW_METADATA = {"block_m": 16, "readiness": "block", "slot_list": False}


def cuda_torch():
    """``torch`` with a CUDA device, else skip: the binding tests allocate the
    route's pools and counters on the device the way a real call does."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    return torch


def small_pipeline_inputs(torch, device="cuda"):
    """A packed FP4 call at the smallest geometry the preparation accepts:
    four tokens, two experts, top-1, hidden 256 (one 256-wide K tile),
    intermediate 128 (one 128-wide step). Random weight and activation bytes
    with unit scales; nothing is launched on them."""
    T, E, H, I = 4, 2, 256, 128
    generator = torch.Generator(device=device).manual_seed(0)

    def packed(*shape):
        return torch.randint(
            0, 256, shape, dtype=torch.uint8, device=device, generator=generator
        )

    def scales(*shape):
        return torch.ones(shape, dtype=torch.float32, device=device)

    return dict(
        num_tokens=T,
        num_experts=E,
        top_k=1,
        hidden=H,
        intermediate=I,
        routed_weight_dtype="fp4",
        activation_clamp=10.0,
        x_fp8_packed=packed(T, H),
        x_sf_packed=packed(T, H // SF_GRAN).view(torch.int32),
        topk_idx=torch.arange(T, device=device).remainder(E).reshape(T, 1),
        topk_weights=torch.ones(T, 1, dtype=torch.float32, device=device),
        w1_fp4=packed(E, 2 * I, H // 2),
        w1_sf=scales(E, 2 * I, H // SF_GRAN),
        w2_fp4=packed(E, H, I // 2),
        w2_sf=scales(E, H, I // SF_GRAN),
    )


@pytest.mark.parametrize("a_source", ["x", "pool", None])
def test_pipeline_bindings_take_the_a_map_from_the_stated_source(a_source):
    """A route whose metadata states ``a_source`` ``"x"`` binds the FC1 ``A``
    map to the activation storage itself: the contiguous ``[T, H]`` byte view
    of ``x_fp8_packed`` (one byte per element, row pitch ``H``), the bytes the
    ``x_fp8`` word view also hands the dispatch. The pool stays allocated and
    bound. ``"pool"`` and an absent key (every route rendered before the key
    existed) bind the dispatch pool rows, as before."""
    torch = cuda_torch()
    inputs = small_pipeline_inputs(torch)
    metadata = dict(SHORT_ROW_METADATA)
    if a_source is not None:
        metadata["a_source"] = a_source
    data = preparation.prepare_pipeline_bindings(inputs, 4, route_metadata=metadata)
    bindings = data["bindings"]
    A, pool, x = bindings["A"], bindings["pool_fp8"], bindings["x_fp8"]
    T, H = inputs["num_tokens"], inputs["hidden"]
    assert A.dtype == torch.uint8 and A.is_contiguous()
    assert x.data_ptr() == inputs["x_fp8_packed"].data_ptr()
    assert pool.shape == (data["M_total"], H // 4) and pool.dtype == torch.int32
    if a_source == "x":
        assert A.shape == (T, H) and A.stride() == (H, 1)
        assert A.data_ptr() == inputs["x_fp8_packed"].data_ptr() == x.data_ptr()
        assert A.data_ptr() != pool.data_ptr()
        assert torch.equal(A, inputs["x_fp8_packed"])
    else:
        assert A.shape == (data["M_total"], H)
        assert A.data_ptr() == pool.data_ptr()


def test_pipeline_bindings_refuse_an_unknown_a_source():
    """The metadata is refused before any input is touched, so the inputs
    may live on the host here."""
    torch = pytest.importorskip("torch")
    inputs = small_pipeline_inputs(torch, device="cpu")
    for bad in ("pool_fp8", "", None, 1):
        metadata = dict(SHORT_ROW_METADATA, a_source=bad)
        with pytest.raises(ValueError, match="a_source"):
            preparation.prepare_pipeline_bindings(inputs, 4, route_metadata=metadata)
