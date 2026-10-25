"""Selected packed-input and workspace preparation; no compiler or oracle dependency.

``prepare_pipeline_bindings`` packs one pipeline call. The caller's
``intermediate`` is the model's logical width (a multiple of the 32-value scale
group); the generated programs compute the intermediate in 128-wide steps, so
the host pads the routed weights and their scales to the next multiple of 128
with zero weights when the logical width is not one already. The padded
columns add exact zeros, so the result is bit-identical to the unpadded
arithmetic, and a width that already is a multiple of 128 is packed exactly as
before (no copy).

Expert-parallel calls describe the rank's window of the global expert space:
``num_experts`` is the number of experts this rank holds (the leading dimension
of its weights), ``local_expert_offset`` the global id of its first expert and
``num_global_experts`` the size of the global space. ``topk_idx`` carries
global ids; a slot whose id falls outside the window (or is -1) routes to no
expert on this rank, takes no row and adds nothing, so the output is the rank's
partial sum over its local routes (zeros for a token without one).
"""

import math

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 256
CTA_GROUP = 2
GRAN_OUT = 32
ALIGN_M = 256
DISP_NUM_EXPERTS_MAX = 384
# The pipeline programs tile the intermediate width in 128-wide steps (two
# 64-column store tiles per step); every scale covers 32 values.
INTERMEDIATE_ALIGN = 128
SF_GRAN = 32
# Scale of the padded weight groups: float 1.0 packs to the UE8M0 byte 127.
# The padded weights are zero, so any finite scale gives exact zero products;
# 1.0 keeps the padded scale words away from the denormal exponents.
PAD_SCALE = 1.0
PIPELINE_KEY_FIELDS = (
    "num_tokens",
    "num_experts",
    "top_k",
    "hidden",
    "intermediate",
    "routed_weight_dtype",
    "activation_clamp",
)


def physical_intermediate(intermediate) -> int:
    """Physical intermediate width of the generated programs for a logical width.

    The logical width must be a positive multiple of the scale group (32). The
    physical width is the next multiple of 128 (the width the programs were
    generated for and the route key names); a logical width that already is a
    multiple of 128 is its own physical width.
    """
    intermediate = int(intermediate)
    if intermediate < SF_GRAN or intermediate % SF_GRAN:
        raise ValueError(
            "MegaMoE requires a positive intermediate width that is a multiple of "
            f"{SF_GRAN} (the scale group); got {intermediate}"
        )
    return -(-intermediate // INTERMEDIATE_ALIGN) * INTERMEDIATE_ALIGN


def pipeline_route_args(inputs):
    """The route-key fields of a pipeline call.

    ``intermediate`` is the physical width the programs were generated for
    (see ``physical_intermediate``), ``routed_weight_dtype`` defaults to
    ``"fp4"`` and a numeric ``activation_clamp`` is a float: the route key is
    JSON text, so ``10`` and ``10.0`` would otherwise name different routes.
    """
    clamp = inputs.get("activation_clamp")
    return dict(
        num_tokens=int(inputs["num_tokens"]),
        num_experts=int(inputs["num_experts"]),
        top_k=int(inputs["top_k"]),
        hidden=int(inputs["hidden"]),
        intermediate=physical_intermediate(inputs["intermediate"]),
        routed_weight_dtype=inputs.get("routed_weight_dtype", "fp4"),
        activation_clamp=None if clamp is None else float(clamp),
    )


def expert_window(inputs):
    """``(local_expert_offset, num_experts, num_global_experts)`` of a call.

    The rank holds the global experts ``[offset, offset + num_experts)``;
    ``local_expert_offset`` defaults to 0 and ``num_global_experts`` to
    ``num_experts`` (a single-rank call).
    """
    local = int(inputs["num_experts"])
    offset = int(inputs.get("local_expert_offset", 0))
    global_experts = inputs.get("num_global_experts")
    global_experts = local if global_experts is None else int(global_experts)
    if offset < 0 or local < 1 or offset + local > global_experts:
        raise ValueError(
            "MegaMoE expert window: local_expert_offset >= 0 and "
            "local_expert_offset + num_experts <= num_global_experts are required; "
            f"got offset {offset}, {local} local experts, {global_experts} global"
        )
    return offset, local, global_experts


def _check_tensor(name, tensor, *, shape=None, numel=None, dtype=None, byte_wide=False):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"MegaMoE input {name!r} must be a tensor")
    if shape is not None and tuple(tensor.shape) != tuple(shape):
        raise ValueError(
            f"MegaMoE input {name!r} must have shape {tuple(shape)}; "
            f"got {tuple(tensor.shape)}"
        )
    if numel is not None and tensor.numel() != numel:
        raise ValueError(
            f"MegaMoE input {name!r} must hold {numel} elements; got {tensor.numel()}"
        )
    if dtype is not None and tensor.dtype != dtype:
        raise ValueError(
            f"MegaMoE input {name!r} must have dtype {dtype}; got {tensor.dtype}"
        )
    if byte_wide and tensor.element_size() != 1:
        raise ValueError(
            f"MegaMoE input {name!r} must be a byte tensor; got {tensor.dtype}"
        )


def _check_bytes(name, tensor, nbytes):
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"MegaMoE input {name!r} must be a tensor")
    if tensor.numel() * tensor.element_size() != nbytes:
        raise ValueError(
            f"MegaMoE input {name!r} must hold {nbytes} bytes; "
            f"got {tensor.numel() * tensor.element_size()}"
        )


def _source_block_m(num_tokens: int, num_experts: int, top_k: int) -> int:
    """Expert tile height of the source schedule for a single-rank model row.

    Mirrors the generator's heuristic: 16 rows up to ten expected tokens per
    expert, 32 rows up to 24, then the smallest wide candidate whose tile
    count covers the expected load plus one standard deviation.
    """
    expected = num_tokens * top_k / num_experts
    if expected <= 10:
        return 16
    if expected <= 24:
        return 32
    covered = expected + math.sqrt(expected)
    num_blocks = math.ceil(covered / 240)
    if num_blocks == 1 and covered > 192 and num_experts < 14:
        num_blocks = 2
    return next(
        candidate
        for candidate in (64, 128, 192, 240)
        if num_blocks * candidate >= covered
    )


def _mega_dispatch_reference(
    topk_idx,
    num_experts: int,
    align_m: int = ALIGN_M,
    block_m: int = BLOCK_M,
    local_expert_offset: int = 0,
):
    """CPU reference for the on-device dispatch: per-expert counts, 256-aligned
    row offsets, tile list, and total M. Row assignment within an expert is
    runtime-nondeterministic (atomic claim order), so pool/metadata are checked
    via token_to_permuted, not by fixed row — this returns only the
    order-independent quantities. ``topk_idx`` holds global ids: a slot counts
    for local expert ``id - local_expert_offset`` only when that falls in
    ``[0, num_experts)``; a negative id (a masked token) or an id another rank
    holds routes its slot to no expert and takes no row."""
    import torch

    ids = topk_idx.detach().reshape(-1).cpu().to(torch.int64) - int(local_expert_offset)
    local = ids[(ids >= 0) & (ids < num_experts)]
    counts = torch.bincount(local, minlength=num_experts)
    pad = ((counts + align_m - 1) // align_m) * align_m
    offsets = torch.cumsum(pad, 0) - pad
    tiles = pad // block_m
    total_m_tiles = int(tiles.sum())
    tile_expert = torch.repeat_interleave(torch.arange(num_experts), tiles)
    first_tile = torch.cumsum(tiles, 0) - tiles
    tile_m_local = (
        torch.arange(total_m_tiles) - torch.repeat_interleave(first_tile, tiles)
    ) * block_m
    return {
        "counts": counts.to(torch.int32),
        "offsets": offsets.to(torch.int32),
        "M_total": int(pad.sum()),
        "tile_expert": tile_expert.to(torch.int32),
        "tile_m_local": tile_m_local.to(torch.int32),
        "total_m_tiles": total_m_tiles,
    }


def _interleave_gate_up(t, gran: int = 8):
    """Mirror deep_gemm.mega._interleave_weights along the N (gate/up) axis:
    [gate0..7, up0..7, gate8..15, up8..15, ...] from [gate(half) | up(half)]."""
    import torch  # noqa: F401

    g, n = t.shape[0], t.shape[1]
    rest = t.shape[2:]
    half = n // 2
    gate = t[:, :half].reshape(g, half // gran, gran, *rest)
    up = t[:, half:].reshape(g, half // gran, gran, *rest)
    return torch.stack([gate, up], dim=2).reshape(g, n, *rest).contiguous()


def _pad_intermediate(w1, w1_sf, w2, w2_sf, logical: int, physical: int, pack: int):
    """Zero-pad the routed weights from the logical to the physical intermediate width.

    W1 rows of both halves move apart (gate ``[0, I)`` to ``[0, I_phys)``, up
    ``[I, 2I)`` to ``[I_phys, 2 I_phys)``) before the gate/up interleave; W2
    gains zero columns ``[I, I_phys)`` (``pack`` values per byte); both scale
    tensors grow alike with ``PAD_SCALE`` in the padded groups. Every padded
    intermediate column activates to exactly 0 (gate and up are 0) and every
    padded down-projection product is exactly 0. Equal widths return the
    caller's tensors untouched.
    """
    import torch

    if physical == logical:
        return w1, w1_sf, w2, w2_sf
    experts = w1.shape[0]

    def rows(t, fill):
        out = torch.full(
            (experts, 2 * physical, *t.shape[2:]), fill, dtype=t.dtype, device=t.device
        )
        out[:, :logical] = t[:, :logical]
        out[:, physical : physical + logical] = t[:, logical:]
        return out

    def cols(t, width, fill):
        out = torch.full((*t.shape[:-1], width), fill, dtype=t.dtype, device=t.device)
        out[..., : t.shape[-1]] = t
        return out

    return (
        rows(w1, 0),
        rows(w1_sf, PAD_SCALE),
        cols(w2, physical // pack, 0),
        cols(w2_sf, physical // SF_GRAN, PAD_SCALE),
    )


def _sf_float_to_u32_words(sf_float):
    """UE8M0 float SF [.., K/32] → packed u32 words [.., K/128] (4 exponent
    bytes per word, natural order) — the gran-32 SF ABI the GEMM load consumes."""
    import torch

    sf_bytes = (
        ((sf_float.contiguous().view(torch.int32) >> 23) & 0xFF)
        .to(torch.uint8)
        .contiguous()
    )
    return sf_bytes.view(torch.int32)


def build_tile_lists(per_expert_M, num_experts: int):
    """Build (tile_expert, tile_m_local, expert_row_offsets, M_total) for the
    contiguous-grouped layout. ``per_expert_M`` is the **aligned** count for
    each expert (rounded up to BLOCK_M)."""
    import torch

    tile_expert = []
    tile_m_local = []
    expert_row_offsets = []
    cumul = 0
    for e in range(num_experts):
        expert_row_offsets.append(cumul)
        m_e = per_expert_M[e]
        assert m_e % BLOCK_M == 0, (
            f"expert {e} M={m_e} not aligned to BLOCK_M={BLOCK_M}"
        )
        for t in range(m_e // BLOCK_M):
            tile_expert.append(e)
            tile_m_local.append(t * BLOCK_M)
        cumul += m_e
    M_total = cumul
    return (
        torch.tensor(tile_expert, dtype=torch.int32, device="cuda"),
        torch.tensor(tile_m_local, dtype=torch.int32, device="cuda"),
        torch.tensor(expert_row_offsets, dtype=torch.int32, device="cuda"),
        M_total,
    )


def source_schedule_form(metadata, num_tokens, num_experts, top_k, weight_fmt):
    """``(block_m, block_readiness, slot_list)`` of a route's generated program.

    A route's ``metadata`` names the source tile height (``block_m``) and the
    readiness form (``readiness``: ``"block"`` when the program starts an
    expert block as soon as its rows arrive, ``"grid"`` when it waits for the
    whole dispatch). ``slot_list`` (per-expert slot lists of ``num_tokens``
    entries, rewritten by every launch) defaults to the block-readiness form
    except at one token, where the program folds the metadata instead. The
    original routes carry none of these: their programs were generated with
    the heuristic tile height and use the slot-list form at the token counts
    listed here.
    """
    block_m = metadata.get("block_m")
    block_m = (
        _source_block_m(num_tokens, num_experts, top_k)
        if block_m is None
        else int(block_m)
    )
    if block_m < 1 or block_m > 128 or 128 % block_m:
        raise ValueError(
            f"MegaMoE source tile height {block_m} is outside the generated "
            "programs' 16-, 32- and 128-row forms"
        )
    readiness = metadata.get("readiness")
    if readiness is None:
        slot_list = (
            weight_fmt == "fp4" and num_tokens in (16, 128, 512, 1024, 4096)
        ) or (weight_fmt == "fp8" and num_tokens == 16)
        block_readiness = slot_list or num_tokens == 1
    elif readiness in ("block", "grid"):
        block_readiness = readiness == "block"
        slot_list = bool(metadata.get("slot_list", block_readiness and num_tokens != 1))
    else:
        raise ValueError(
            f"route metadata 'readiness' must be 'block' or 'grid'; got {readiness!r}"
        )
    return block_m, block_readiness, slot_list


def prepare_pipeline_bindings(inputs, num_sms, *, route_metadata=None):
    """Pack ``inputs`` for the persistent grid of a ``num_sms``-SM catalogued route.

    ``inputs["intermediate"]`` is the logical width; the weights and scales are
    padded to the physical width of the route (``physical_intermediate``) when
    the two differ, and the kernel scalars ``N1``/``K2`` name the physical width.
    ``route_metadata`` is the catalogued route's ``metadata`` mapping: its
    ``block_m`` and ``readiness`` size the source-schedule workspace
    (``source_schedule_form``) and ``local_expert_filter`` states that the
    program filters global ids through the ``local_expert_offset`` kernel
    scalar; without the filter the call must describe a single-rank window
    (offset 0, every id below ``num_experts``). Without metadata (the original
    routes) the host uses the original heuristics.
    """
    import torch

    metadata = {} if route_metadata is None else route_metadata
    local_expert_filter = bool(metadata.get("local_expert_filter", False))
    args = pipeline_route_args(inputs)
    E, TK, T, H, I = (
        args[k]
        for k in ("num_experts", "top_k", "num_tokens", "hidden", "intermediate")
    )
    logical_I = int(inputs["intermediate"])
    weight_fmt = args["routed_weight_dtype"]
    if weight_fmt not in ("fp4", "fp8"):
        raise ValueError("routed_weight_dtype must be 'fp4' or 'fp8'")
    pack = 2 if weight_fmt == "fp4" else 1
    if not 1 <= E <= DISP_NUM_EXPERTS_MAX or not 1 <= TK <= min(E, 32):
        raise ValueError(
            "MegaMoE requires 1..384 experts and 1..min(experts,32) routes"
        )
    if T < 1 or H % BLOCK_K:
        raise ValueError("MegaMoE requires positive T and H divisible by 256")
    assert I % INTERMEDIATE_ALIGN == 0, I
    offset, _, global_experts = expert_window(inputs)
    if (offset or global_experts != E) and not local_expert_filter:
        raise ValueError(
            "This route's program has no local expert filter: it requires "
            "local_expert_offset 0 and num_global_experts == num_experts "
            f"({E}); got offset {offset} and {global_experts} global experts"
        )
    w1, w1_sf, w2, w2_sf = (
        inputs[k] for k in (f"w1_{weight_fmt}", "w1_sf", f"w2_{weight_fmt}", "w2_sf")
    )
    _check_tensor(
        f"w1_{weight_fmt}", w1, shape=(E, 2 * logical_I, H // pack), byte_wide=True
    )
    _check_tensor(
        "w1_sf", w1_sf, shape=(E, 2 * logical_I, H // SF_GRAN), dtype=torch.float32
    )
    _check_tensor(
        f"w2_{weight_fmt}", w2, shape=(E, H, logical_I // pack), byte_wide=True
    )
    _check_tensor(
        "w2_sf", w2_sf, shape=(E, H, logical_I // SF_GRAN), dtype=torch.float32
    )
    _check_tensor("topk_idx", inputs["topk_idx"], numel=T * TK, dtype=torch.int64)
    _check_tensor(
        "topk_weights", inputs["topk_weights"], numel=T * TK, dtype=torch.float32
    )
    _check_bytes("x_fp8_packed", inputs["x_fp8_packed"], T * H)
    _check_bytes("x_sf_packed", inputs["x_sf_packed"], T * H // SF_GRAN)
    ids = inputs["topk_idx"].detach().reshape(-1).cpu()
    if int(ids.min()) < -1 or int(ids.max()) >= global_experts:
        raise ValueError(
            f"topk_idx must hold global expert ids in [0, {global_experts}) or -1; "
            f"got [{int(ids.min())}, {int(ids.max())}]"
        )
    source_block_m, block_readiness, slot_list = source_schedule_form(
        metadata, T, E, TK, weight_fmt
    )
    source_blocks = E + (T * TK + source_block_m - 1) // source_block_m
    N1, K1, N2, K2 = (2 * I, H, H, I)
    grid_n1, K1_tiles = (N1 // BLOCK_N, K1 // BLOCK_K)
    # K2_tiles is a legacy term of the arrival-word bound and an unused kernel
    # scalar; the physical width needs ceil(I / 256) of the 256-wide tiles.
    grid_n2, K2_tiles = (N2 // BLOCK_N, -(-K2 // BLOCK_K))
    ref = _mega_dispatch_reference(ids, E, local_expert_offset=offset)
    # A batch whose every slot is -1 routes no row; its pools keep one aligned
    # block so that their tensor maps stay valid.
    M_total = max(ref["M_total"], ALIGN_M)
    total_m_tiles = ref["total_m_tiles"]
    topk_idx_i32 = inputs["topk_idx"].contiguous().view(torch.int32)
    topk_weights_flat = inputs["topk_weights"].contiguous().view(-1)
    x_fp8 = (
        inputs["x_fp8_packed"]
        .view(torch.uint8)
        .reshape(T, H)
        .view(torch.int32)
        .contiguous()
    )
    x_sf = inputs["x_sf_packed"].contiguous().view(torch.int32).view(torch.uint32)
    w1, w1_sf, w2, w2_sf = _pad_intermediate(w1, w1_sf, w2, w2_sf, logical_I, I, pack)
    B1 = _interleave_gate_up(w1).contiguous().view(torch.uint8)
    SFB1 = _sf_float_to_u32_words(_interleave_gate_up(w1_sf)).view(torch.uint32)
    B2 = w2.contiguous().view(torch.uint8)
    SFB2 = _sf_float_to_u32_words(w2_sf).view(torch.uint32)
    pool_fp8 = torch.zeros(M_total, H // 4, dtype=torch.int32, device="cuda")
    pool_sf = torch.zeros(M_total, H // 128, dtype=torch.int32, device="cuda")
    routing_weight_pool = torch.zeros(M_total, dtype=torch.float32, device="cuda")
    token_to_permuted = torch.full((T * TK,), -1, dtype=torch.int32, device="cuda")
    meta_token = torch.full((M_total,), -1, dtype=torch.int32, device="cuda")
    meta_slot = torch.full((M_total,), -1, dtype=torch.int32, device="cuda")
    num_sms = int(num_sms)
    # The source schedule runs one CTA per SM (NUM_CTAS) however few tiles
    # the routes fill, e.g. when -1 slots leave fewer routed tokens.
    persistent_grid = num_sms
    grid = (persistent_grid, 1, 1)
    # Two task counters, two mask words per source block and, with per-block
    # readiness, one arrival word per source block.
    arrival_words = total_m_tiles * K2_tiles
    arrival_words = max(
        arrival_words,
        2 + (3 if block_readiness else 2) * source_blocks,
    )
    reset_storage = torch.zeros(
        2 * E + 4 + arrival_words, dtype=torch.uint32, device="cuda"
    )
    expert_counts = reset_storage[:E].view(torch.int32)
    expert_row_offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    if slot_list:
        # Slot-list dispatch: per-expert slot list [E, T] (native
        # src_token_topk_idx), fully rewritten by every launch, never reset.
        expert_scatter_offsets = torch.empty(E * T, dtype=torch.int32, device="cuda")
    else:
        expert_scatter_offsets = reset_storage[E : 2 * E].view(torch.int32)
    tile_expert = torch.full((total_m_tiles,), -1, dtype=torch.int32, device="cuda")
    tile_m_local = torch.full((total_m_tiles,), -1, dtype=torch.int32, device="cuda")
    total_m_tiles_out = torch.zeros(1, dtype=torch.int32, device="cuda")
    I_fp8 = torch.zeros(M_total, I, dtype=torch.uint8, device="cuda")
    SF_I = torch.zeros(M_total, I // GRAN_OUT, dtype=torch.uint8, device="cuda")
    expert_output = torch.zeros(M_total, H, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    histogram_done = reset_storage[2 * E : 2 * E + 1]
    prefix_done = reset_storage[2 * E + 1 : 2 * E + 2]
    dispatch_done = reset_storage[2 * E + 2 : 2 * E + 3]
    l2_done = reset_storage[2 * E + 3 : 2 * E + 4]
    l1_arrival = reset_storage[2 * E + 4 :]
    launch_kwargs = dict(
        grid=grid,
        topk_idx_i32=topk_idx_i32,
        topk_weights=topk_weights_flat,
        x_fp8=x_fp8,
        x_sf=x_sf,
        A=pool_fp8.view(torch.uint8),
        pool_fp8=pool_fp8,
        pool_sf=pool_sf.view(torch.uint32),
        routing_weight_pool=routing_weight_pool,
        token_to_permuted=token_to_permuted,
        meta_token=meta_token,
        meta_slot=meta_slot,
        expert_counts=expert_counts,
        expert_row_offsets=expert_row_offsets,
        expert_scatter_offsets=expert_scatter_offsets,
        tile_expert=tile_expert,
        tile_m_local=tile_m_local,
        total_m_tiles_out=total_m_tiles_out,
        B1=B1,
        SFB1=SFB1,
        I_fp8_w=I_fp8,
        SF_I_w=SF_I,
        l1_bf16_capture=expert_output,
        A2=I_fp8,
        SFA2=SF_I.view(torch.uint32),
        B2=B2,
        SFB2=SFB2,
        expert_output=expert_output,
        y=y,
        histogram_done=histogram_done,
        prefix_done=prefix_done,
        dispatch_done=dispatch_done,
        l1_arrival=l1_arrival,
        l2_done=l2_done,
        num_tokens=T,
        top_k=TK,
        num_experts=E,
        N1=N1,
        K1=K1,
        grid_n1=grid_n1,
        K1_tiles=K1_tiles,
        N2=N2,
        K2=K2,
        grid_n2=grid_n2,
        K2_tiles=K2_tiles,
        total_m_tiles=total_m_tiles,
        M_total=M_total,
        activation_clamp=float("inf")
        if inputs.get("activation_clamp") is None
        else float(inputs["activation_clamp"]),
        # Global id of this rank's first expert: a program with the local
        # expert filter routes ids outside [offset, offset + E) to no expert.
        # A program whose argument plan lacks the scalar never reads it.
        local_expert_offset=offset,
    )
    for name in ("SFB1", "SFB2"):
        original = launch_kwargs[name]
        groups, rows, words_k = original.shape
        packed = original.reshape(groups, rows // 128, 4, 32, words_k).transpose(2, 3)
        launch_kwargs[name] = (
            packed.reshape(groups, rows, words_k)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(-1, rows)
        )
    for name in ("B1", "B2"):
        original = launch_kwargs[name]
        launch_kwargs[name] = original.reshape(-1, original.shape[-1])
    source_rows = source_blocks * 128
    source_sf1 = torch.zeros((H // 128, source_rows), dtype=torch.uint32, device="cuda")
    source_sf2 = torch.zeros((I // 128, source_rows), dtype=torch.uint32, device="cuda")
    launch_kwargs.update(
        PrivateCounters=l1_arrival[:2],
        PrivateMasks=l1_arrival[2 : 2 + 2 * source_blocks].view(torch.uint64),
        PublicExpertOutput=expert_output.view(torch.uint32),
        PrivateSFBlockOffsets=torch.empty((E,), dtype=torch.uint32, device="cuda"),
        PrivateSF1=source_sf1,
        PrivateSF2=source_sf2,
        PrivateSF1Words=source_sf1.reshape(-1),
        PrivateSF2Bytes=source_sf2.view(torch.uint8).reshape(-1),
    )
    reset_buffers = (
        expert_counts,
        expert_scatter_offsets,
        histogram_done,
        prefix_done,
        dispatch_done,
        l1_arrival,
        l2_done,
    )
    return dict(
        bindings=launch_kwargs,
        reset_storage=reset_storage,
        reset_buffers=reset_buffers,
        output=y,
        expert_output=expert_output,
        source_schedule=True,
        grid=grid,
        M_total=M_total,
        total_m_tiles=total_m_tiles,
        intermediate=I,
        local_expert_offset=offset,
        num_global_experts=global_experts,
        block_m=source_block_m,
    )
