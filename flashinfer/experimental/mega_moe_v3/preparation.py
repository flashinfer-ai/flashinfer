"""Selected packed-input and workspace preparation; no compiler or oracle dependency."""

BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 256
CTA_GROUP = 2
GRAN_OUT = 32
ALIGN_M = 256
DISP_NUM_EXPERTS_MAX = 384


def _mega_dispatch_reference(
    topk_idx, num_experts: int, align_m: int = ALIGN_M, block_m: int = BLOCK_M
):
    """CPU reference for the on-device dispatch: per-expert counts, 256-aligned
    row offsets, tile list, and total M. Row assignment within an expert is
    runtime-nondeterministic (atomic claim order), so pool/metadata are checked
    via token_to_permuted, not by fixed row — this returns only the
    order-independent quantities."""
    import torch

    T, TK = topk_idx.shape
    counts = torch.zeros(num_experts, dtype=torch.int64)
    ids = topk_idx.cpu()
    for t in range(T):
        for k in range(TK):
            counts[int(ids[t, k].item())] += 1
    pad = ((counts + align_m - 1) // align_m) * align_m
    offsets = torch.zeros(num_experts, dtype=torch.int64)
    running = 0
    tile_expert = []
    tile_m_local = []
    for e in range(num_experts):
        offsets[e] = running
        for tt in range(int(pad[e].item()) // block_m):
            tile_expert.append(e)
            tile_m_local.append(tt * block_m)
        running += int(pad[e].item())
    return {
        "counts": counts.to(torch.int32),
        "offsets": offsets.to(torch.int32),
        "M_total": int(running),
        "tile_expert": torch.tensor(tile_expert, dtype=torch.int32),
        "tile_m_local": torch.tensor(tile_m_local, dtype=torch.int32),
        "total_m_tiles": len(tile_expert),
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


def prepare_pipeline_bindings(inputs, num_sms):
    """Pack ``inputs`` for the persistent grid of a ``num_sms``-SM catalogued route."""
    pr432_optimizations = True
    capture_l1 = False
    import torch

    E, TK, T, H, I = (
        inputs[k]
        for k in ("num_experts", "top_k", "num_tokens", "hidden", "intermediate")
    )
    weight_fmt = inputs.get("routed_weight_dtype", "fp4")
    if weight_fmt not in ("fp4", "fp8"):
        raise ValueError("routed_weight_dtype must be 'fp4' or 'fp8'")
    if not 1 <= E <= DISP_NUM_EXPERTS_MAX or not 1 <= TK <= min(E, 32):
        raise ValueError(
            "MegaMoE requires 1..384 experts and 1..min(experts,32) routes"
        )
    if T < 1 or H % BLOCK_K or I % BLOCK_K:
        raise ValueError("MegaMoE requires positive T and H/I divisible by 256")
    source_schedule = (
        bool(pr432_optimizations)
        and (not capture_l1)
        and ((E, TK, H, I) == (384, 6, 5120, 2304))
    )
    source_blocks = E + (T * TK + 15) // 16
    N1, K1, N2, K2 = (2 * I, H, H, I)
    grid_n1, K1_tiles = (N1 // BLOCK_N, K1 // BLOCK_K)
    grid_n2, K2_tiles = (N2 // BLOCK_N, K2 // BLOCK_K)
    ref = _mega_dispatch_reference(inputs["topk_idx"], E)
    M_total = ref["M_total"]
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
    B1 = _interleave_gate_up(inputs[f"w1_{weight_fmt}"]).contiguous().view(torch.uint8)
    SFB1 = _sf_float_to_u32_words(_interleave_gate_up(inputs["w1_sf"])).view(
        torch.uint32
    )
    B2 = inputs[f"w2_{weight_fmt}"].contiguous().view(torch.uint8)
    SFB2 = _sf_float_to_u32_words(inputs["w2_sf"]).view(torch.uint32)
    pool_fp8 = torch.zeros(M_total, H // 4, dtype=torch.int32, device="cuda")
    pool_sf = torch.zeros(M_total, H // 128, dtype=torch.int32, device="cuda")
    routing_weight_pool = torch.zeros(M_total, dtype=torch.float32, device="cuda")
    token_to_permuted = torch.full((T * TK,), -1, dtype=torch.int32, device="cuda")
    meta_token = torch.full((M_total,), -1, dtype=torch.int32, device="cuda")
    meta_slot = torch.full((M_total,), -1, dtype=torch.int32, device="cuda")
    num_sms = int(num_sms)
    num_tiles = max(total_m_tiles * grid_n1, total_m_tiles * grid_n2)
    persistent_grid = max(CTA_GROUP, min(num_sms - num_sms % CTA_GROUP, num_tiles))
    if persistent_grid % CTA_GROUP != 0:
        persistent_grid += CTA_GROUP - persistent_grid % CTA_GROUP
    grid = (persistent_grid, 1, 1)
    residual_readiness = (
        source_schedule
        and persistent_grid in (148, 152)
        and (
            (
                float("inf")
                if inputs.get("activation_clamp") is None
                else float(inputs["activation_clamp"])
            )
            == 10.0
        )
        and (
            weight_fmt == "fp4"
            and T in (16, 128, 512)
            or (weight_fmt == "fp8" and T == 16)
        )
    )
    arrival_words = total_m_tiles * K2_tiles
    if source_schedule:
        arrival_words = max(
            arrival_words,
            2 + (3 if T == 1 or residual_readiness else 2) * source_blocks,
        )
    reset_storage = (
        torch.zeros(2 * E + 4 + arrival_words, dtype=torch.uint32, device="cuda")
        if source_schedule and (T == 1 or residual_readiness)
        else None
    )
    expert_counts = (
        reset_storage[:E].view(torch.int32)
        if reset_storage is not None
        else torch.zeros(E, dtype=torch.int32, device="cuda")
    )
    expert_row_offsets = torch.zeros(E, dtype=torch.int32, device="cuda")
    if source_schedule and residual_readiness:
        # Round-7 slot-list dispatch: per-expert slot list [E, T] (native
        # src_token_topk_idx), fully rewritten by every launch, never reset.
        expert_scatter_offsets = torch.empty(E * T, dtype=torch.int32, device="cuda")
    else:
        expert_scatter_offsets = (
            reset_storage[E : 2 * E].view(torch.int32)
            if reset_storage is not None
            else torch.zeros(E, dtype=torch.int32, device="cuda")
        )
    tile_expert = torch.full((total_m_tiles,), -1, dtype=torch.int32, device="cuda")
    tile_m_local = torch.full((total_m_tiles,), -1, dtype=torch.int32, device="cuda")
    total_m_tiles_out = torch.zeros(1, dtype=torch.int32, device="cuda")
    I_fp8 = torch.zeros(M_total, I, dtype=torch.uint8, device="cuda")
    SF_I = torch.zeros(M_total, I // GRAN_OUT, dtype=torch.uint8, device="cuda")
    expert_output = torch.zeros(M_total, H, dtype=torch.bfloat16, device="cuda")
    y = torch.zeros(T, H, dtype=torch.bfloat16, device="cuda")
    l1_bf16 = (
        torch.full((M_total, N1), float("nan"), dtype=torch.bfloat16, device="cuda")
        if capture_l1
        else expert_output
    )
    if reset_storage is not None:
        histogram_done = reset_storage[2 * E : 2 * E + 1]
        prefix_done = reset_storage[2 * E + 1 : 2 * E + 2]
        dispatch_done = reset_storage[2 * E + 2 : 2 * E + 3]
        l2_done = reset_storage[2 * E + 3 : 2 * E + 4]
        l1_arrival = reset_storage[2 * E + 4 :]
    else:
        histogram_done = torch.zeros(1, dtype=torch.uint32, device="cuda")
        prefix_done = torch.zeros(1, dtype=torch.uint32, device="cuda")
        dispatch_done = torch.zeros(1, dtype=torch.uint32, device="cuda")
        l1_arrival = torch.zeros(arrival_words, dtype=torch.uint32, device="cuda")
        l2_done = torch.zeros(1, dtype=torch.uint32, device="cuda")
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
        l1_bf16_capture=l1_bf16,
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
    )
    if source_schedule:
        for name in ("SFB1", "SFB2"):
            original = launch_kwargs[name]
            groups, rows, words_k = original.shape
            packed = original.reshape(groups, rows // 128, 4, 32, words_k).transpose(
                2, 3
            )
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
        source_sf1 = torch.zeros(
            (H // 128, source_rows), dtype=torch.uint32, device="cuda"
        )
        source_sf2 = torch.zeros(
            (I // 128, source_rows), dtype=torch.uint32, device="cuda"
        )
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
        source_schedule=source_schedule,
        grid=grid,
        M_total=M_total,
        total_m_tiles=total_m_tiles,
    )
