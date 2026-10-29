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


def test_pipeline_bindings_align_the_scale_words_for_an_activation_route():
    """A route that reads A from the activations stages the scale words with
    16-byte loads: an ``x_sf_packed`` view that starts four bytes into a larger
    buffer is copied to a view that starts on a 16-byte boundary (the same
    words); a pool route binds the caller's view as before."""
    torch = cuda_torch()
    inputs = small_pipeline_inputs(torch)
    T, H = inputs["num_tokens"], inputs["hidden"]
    words = T * H // SF_GRAN // 4
    buffer = torch.randint(0, 2**31 - 1, (words + 1,), dtype=torch.int32, device="cuda")
    inputs["x_sf_packed"] = buffer[1:]
    assert inputs["x_sf_packed"].data_ptr() % 16 == 4
    for a_source in ("x", "pool"):
        metadata = dict(SHORT_ROW_METADATA, a_source=a_source)
        data = preparation.prepare_pipeline_bindings(inputs, 4, route_metadata=metadata)
        x_sf = data["bindings"]["x_sf"]
        assert torch.equal(x_sf.view(torch.int32).reshape(-1), buffer[1:])
        if a_source == "x":
            assert x_sf.data_ptr() % 16 == 0
            assert x_sf.data_ptr() != inputs["x_sf_packed"].data_ptr()
        else:
            assert x_sf.data_ptr() == inputs["x_sf_packed"].data_ptr()


def test_pipeline_bindings_refuse_a_token_routed_twice_to_one_expert_on_an_activation_route():
    """The in-place A read ranks tokens within an expert, not slots: a token
    that routes to one local expert through two slots is refused on an ``"x"``
    route (a pool route places both slots, as before). Masked slots (``-1``)
    take no row and may repeat."""
    torch = cuda_torch()
    inputs = small_pipeline_inputs(torch)
    T = inputs["num_tokens"]
    inputs["top_k"] = 2
    inputs["topk_weights"] = torch.ones(T, 2, dtype=torch.float32, device="cuda")

    def prepare(topk_idx, a_source):
        routed = dict(inputs, topk_idx=torch.tensor(topk_idx, device="cuda"))
        metadata = dict(SHORT_ROW_METADATA, a_source=a_source)
        return preparation.prepare_pipeline_bindings(routed, 4, route_metadata=metadata)

    twice = [[0, 0], [0, 1], [1, 0], [1, 1]]
    with pytest.raises(ValueError, match="distinct experts per token"):
        prepare(twice, "x")
    prepare(twice, "pool")
    prepare([[0, -1], [-1, -1], [1, 0], [-1, 1]], "x")


# The control buffers the arena carves after ``reset_storage``, in carve
# order (``_carve_control_arena``); a slot-list route adds its
# ``expert_scatter_offsets`` after ``PrivateSFBlockOffsets``. The tile lists
# share one page, so only the first starts on a page boundary.
ARENA_STORES = (
    "expert_row_offsets",
    "PrivateSFBlockOffsets",
    "tile_expert",
    "tile_m_local",
    "total_m_tiles_out",
    "token_to_permuted",
    "meta_token",
    "meta_slot",
    "routing_weight_pool",
)
SHARED_PAGE_MEMBERS = ("tile_m_local", "total_m_tiles_out")
# The bindings that are views of ``reset_storage`` on every route; a route
# without the slot list adds ``expert_scatter_offsets``.
RESET_VIEWS = (
    "expert_counts",
    "histogram_done",
    "prefix_done",
    "dispatch_done",
    "l2_done",
    "l1_arrival",
    "PrivateCounters",
    "PrivateMasks",
)
MINUS_ONE_FILLED = (
    "tile_expert",
    "tile_m_local",
    "token_to_permuted",
    "meta_token",
    "meta_slot",
)


def byte_range(tensor):
    """``(first, end)`` device byte addresses of a contiguous tensor."""
    first = tensor.data_ptr()
    return first, first + tensor.numel() * tensor.element_size()


def owns_its_storage(tensor):
    """Whether the tensor is a separate allocation (not a view of a slab)."""
    return tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()


def prepare_small_route(torch, monkeypatch, mode, slot_list=False):
    """The small route prepared under ``FLASHINFER_MEGA_MOE_CONTROL_ARENA``
    ``mode`` (``None``: the knob unset, the default)."""
    if mode is None:
        monkeypatch.delenv(preparation.CONTROL_ARENA_KNOB, raising=False)
    else:
        monkeypatch.setenv(preparation.CONTROL_ARENA_KNOB, mode)
    metadata = dict(SHORT_ROW_METADATA, slot_list=slot_list)
    inputs = small_pipeline_inputs(torch)
    return preparation.prepare_pipeline_bindings(inputs, 4, route_metadata=metadata)


@pytest.mark.parametrize("slot_list", [False, True])
def test_pipeline_bindings_carve_the_control_words_into_a_page_aligned_arena(
    slot_list, monkeypatch
):
    """By default every kernel-written control buffer is carved from the
    plan's ``control_slab``: ``reset_storage`` starts 64 KiB (the leading
    guard) into a 2 MiB page of the slab, a second guard follows it, every
    other control buffer takes its own 4 KiB page after that guard in carve
    order (the tile lists share one page at 128-byte steps), the trailing
    guard fits in the slab and no other binding touches the slab. The
    counters, gates and masks stay views of ``reset_storage``, and the fills
    are those of the separate allocations: -1 for the tile lists and the
    dispatch scratch, 0 elsewhere (``PrivateSFBlockOffsets`` and the slot
    list were uninitialised; every reader writes them first)."""
    torch = cuda_torch()
    PAGE, GUARD, SLOT = (
        preparation.ARENA_PAGE,
        preparation.ARENA_GUARD,
        preparation.ARENA_SLOT,
    )
    data = prepare_small_route(torch, monkeypatch, None, slot_list)
    bindings, slab = data["bindings"], data["control_slab"]
    reset_storage = data["reset_storage"]
    assert slab.dtype == torch.uint8 and slab.numel() == 2 * PAGE
    slab_first, slab_end = byte_range(slab)
    reset_first, reset_end = byte_range(reset_storage)
    assert reset_first % PAGE == GUARD and reset_first - GUARD >= slab_first
    stores = list(ARENA_STORES)
    if slot_list:
        stores.insert(2, "expert_scatter_offsets")
    # Class A alone between two guards, then the stores in carve order, each
    # on its own page except the shared tile-list page.
    reset_pages = -(-(reset_end - reset_first) // SLOT) * SLOT
    stores_first = reset_first + reset_pages + GUARD
    firsts = [bindings[name].data_ptr() for name in stores]
    assert firsts == sorted(firsts) and firsts[0] >= stores_first
    pages = [first // SLOT for first in firsts]
    for name, first in zip(stores, firsts, strict=True):
        step = preparation.ARENA_SUB if name in SHARED_PAGE_MEMBERS else SLOT
        assert first % step == 0, name
    heads = [
        p
        for name, p in zip(stores, pages, strict=True)
        if name not in SHARED_PAGE_MEMBERS
    ]
    assert len(set(heads)) == len(heads)
    arena_end = max(byte_range(bindings[name])[1] for name in stores)
    assert arena_end + GUARD <= slab_end
    # The reset views lie inside ``reset_storage``; nothing else is in the slab.
    reset_views = list(RESET_VIEWS)
    if not slot_list:
        reset_views.append("expert_scatter_offsets")
    for name in reset_views:
        first, end = byte_range(bindings[name])
        assert reset_first <= first and end <= reset_end, name
        assert bindings[name].untyped_storage().data_ptr() == slab.data_ptr(), name
    assert bindings["PrivateMasks"].data_ptr() % 8 == 0
    for name, value in bindings.items():
        if isinstance(value, torch.Tensor) and name not in stores + reset_views:
            first, end = byte_range(value)
            assert end <= slab_first or first >= slab_end, name
    # Fills: every control buffer is four bytes wide, so compare the words.
    for name in stores:
        words = bindings[name].view(torch.int32)
        fill = -1 if name in MINUS_ONE_FILLED else 0
        assert bool((words == fill).all()), name
    assert bool((reset_storage.view(torch.int32) == 0).all())


@pytest.mark.parametrize("mode", ["off", "reset", "all"])
def test_control_arena_knob_moves_pointers_only(mode, monkeypatch):
    """``FLASHINFER_MEGA_MOE_CONTROL_ARENA`` changes where the control
    buffers live and nothing else: the binding names, dtypes and shapes,
    ``reset_storage``'s words and the ``reset_buffers`` tuple are the same
    under every value. ``off`` allocates every buffer separately (no slab,
    as before the arena); ``reset`` carves ``reset_storage`` alone and keeps
    the other control buffers separate allocations."""
    torch = cuda_torch()

    def surface(data):
        bindings = data["bindings"]
        tensors = {
            name: (value.dtype, tuple(value.shape))
            for name, value in bindings.items()
            if isinstance(value, torch.Tensor)
        }
        reset_storage = data["reset_storage"]
        return (
            set(bindings),
            tensors,
            (reset_storage.dtype, tuple(reset_storage.shape)),
            len(data["reset_buffers"]),
        )

    data = prepare_small_route(torch, monkeypatch, mode)
    assert surface(data) == surface(prepare_small_route(torch, monkeypatch, "off"))
    assert len(data["reset_buffers"]) == 7
    slab, reset_storage = data["control_slab"], data["reset_storage"]
    stores = [data["bindings"][name] for name in ARENA_STORES]
    if mode == "off":
        assert slab is None and owns_its_storage(reset_storage)
        assert all(owns_its_storage(tensor) for tensor in stores)
        return
    assert reset_storage.data_ptr() % preparation.ARENA_PAGE == preparation.ARENA_GUARD
    assert reset_storage.untyped_storage().data_ptr() == slab.data_ptr()
    in_slab = [
        tensor.untyped_storage().data_ptr() == slab.data_ptr() for tensor in stores
    ]
    assert in_slab == [mode == "all"] * len(stores)
    if mode == "reset":
        assert all(owns_its_storage(tensor) for tensor in stores)


def test_control_arena_knob_refuses_an_unknown_value(monkeypatch):
    """The knob is checked before any input is touched, so the inputs may
    live on the host here."""
    torch = pytest.importorskip("torch")
    inputs = small_pipeline_inputs(torch, device="cpu")
    monkeypatch.setenv(preparation.CONTROL_ARENA_KNOB, "on")
    with pytest.raises(ValueError, match=preparation.CONTROL_ARENA_KNOB):
        preparation.prepare_pipeline_bindings(
            inputs, 4, route_metadata=SHORT_ROW_METADATA
        )
