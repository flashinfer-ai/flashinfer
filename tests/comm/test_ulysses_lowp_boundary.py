"""Independent global-grid references; CPU boundary proofs and per-layout GPU gates."""

import pytest
import torch

import flashinfer.comm.ulysses_lowp as lowp

LAYOUTS = [lowp.UlyssesLowpSageLayout, lowp.UlyssesLowpSageLayoutSM90]
LENGTHS = [1, 15, 16, 17, 63, 64, 65, 127, 128, 129, 160, 192, 256]


def global_amax(x, group):
    """Independent reference on live NHD rows, without payload geometry helpers."""
    return torch.stack(
        [
            x[:, start : start + group].abs().amax(dim=(1, 3)).clamp_min(1e-7)
            for start in range(0, x.shape[1], group)
        ],
        dim=-1,
    )


@pytest.mark.parametrize("layout_cls", LAYOUTS)
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("head_dim", [64, 128])
def test_boundary_padding_cpu(layout_cls, world, length, monkeypatch, head_dim):
    """Exercise actual min/max and merge helpers against a live global oracle.

    Only the CUDA admission check is bypassed; no compiled math is claimed tested.
    Includes empty ranks and a live group spanning more than two ranks.
    """
    monkeypatch.setattr(lowp, "_validate_nhd_input", lambda name, x: tuple(x.shape))
    layout = layout_cls(head_dim=head_dim)
    group = layout.K_GROUP
    total = world * length
    torch.manual_seed(79)
    for used in sorted({1, total, max(1, total - 1), min(total, group + 1)}):
        k = (torch.randn(2, total, world, head_dim) + 16).to(torch.bfloat16)
        k[:, used:] = 0
        mean = k[:, :used].float().mean(1).to(k.dtype)
        ref = global_amax(k[:, :used].float() - mean.float().unsqueeze(1), group)
        desc = []
        for rank in range(world):
            shard = k[:, rank * length : (rank + 1) * length]
            if group == 64:
                desc.append(
                    lowp.k_boundary_minmax(
                        shard, rank=rank, world_size=world, used_sequence=used
                    )
                )
            else:
                desc.append(layout._k_boundary_minmax(shard, rank, world, used))
        gathered = torch.stack(desc)
        for rank in range(world):
            first = rank * length // group
            last = ((rank + 1) * length - 1) // group
            local = torch.zeros(2, world, (length + 2 * group - 2) // group)
            if group == 64:
                lowp.derive_k_boundary_amax(
                    local,
                    gathered,
                    mean,
                    rank=rank,
                    local_sequence=length,
                    world_size=world,
                )
            else:
                layout._derive_k_boundary_amax(
                    local, gathered, mean, rank, length, world
                )
            for g in {first, last}:
                expected = (
                    ref[..., g] if g < ref.shape[-1] else torch.full((2, world), 1e-7)
                )
                assert torch.equal(local[..., g - first], expected)
        assert layout.stats_protocol_for(length, world) == lowp.BOUNDARY_MERGE


@pytest.mark.parametrize("group", [64, 128])
def test_tail_repair_preserves_other_groups_cpu(group):
    """Repair only the live/padded group, including padding-only contributors."""
    world, length = 4, group // 2 + 1
    total = world * length
    mean = torch.full((1, 2, 128), 257, dtype=torch.bfloat16)  # rounds to 256
    for used in [1, group, group + 3, total - 1, total]:
        source = torch.full((1, total, 2, 128), 258, dtype=torch.bfloat16)
        source[:, used:] = 0
        for rank in range(world):
            token_ids = torch.arange(rank * length, (rank + 1) * length)
            shard = source[:, token_ids]
            first, last = int(token_ids[0]) // group, int(token_ids[-1]) // group
            amax = torch.full((1, 2, lowp.slots(length, group)), -7.0)
            expected = amax.clone()
            if used < total and used % group:
                tail = (used - 1) // group
                if first <= tail <= last:
                    live = (token_ids // group == tail) & (token_ids < used)
                    expected[..., tail - first] = (
                        (shard[:, live].float() - mean.float().unsqueeze(1))
                        .abs()
                        .amax(dim=(1, 3))
                        .clamp_min(1e-7)
                        if live.any()
                        else 1e-7
                    )
            lowp._repair_k_tail_amax(amax, shard, mean, rank, world, used, group)
            assert torch.equal(amax, expected)


@pytest.fixture(
    params=[(cls, d) for cls in LAYOUTS for d in (64, 128)],
    ids=["sm120-d64", "sm120-d128", "sm90-d64", "sm90-d128"],
)
def gpu_layout(request):
    layout_cls, dim = request.param
    expected = (9, 0) if layout_cls is lowp.UlyssesLowpSageLayoutSM90 else (12, 0)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != expected:
        pytest.skip(f"requires {expected}; run separately on each target architecture")
    return layout_cls(head_dim=dim)


def inputs(batch, total, heads, dtype, used, device="cuda", head_dim=128):
    torch.manual_seed(79)
    projection = torch.randn(
        batch, total, heads, 3, head_dim, device=device, dtype=dtype
    )
    projection[:, used:] = 0
    q, k, v = projection.unbind(3)  # legal outer projection strides
    k[:, :used] += 16
    v[..., 0] = 0
    v[..., 1] = -0.0
    return q, k, v


def simulate(layout, q, k, v, world, used):
    length = q.shape[1] // world
    local = [
        tuple(x[:, r * length : (r + 1) * length] for x in (q, k, v))
        for r in range(world)
    ]
    sends, ctxs = zip(
        *[
            layout.local_stats(
                *xs, rank=r, world_size=world, used_sequence=used, enable_pdl=False
            )
            for r, xs in enumerate(local)
        ],
        strict=True,
    )
    gathered = torch.stack(sends)
    stats = [
        layout.finalize_stats(gathered, ctx, xs[1], enable_pdl=False)
        for ctx, xs in zip(ctxs, local, strict=True)
    ]
    payloads = [
        layout.quant_and_pack(*xs, st, enable_pdl=False)
        for xs, st in zip(local, stats, strict=True)
    ]
    return local, stats, payloads


def check_receiver(layout, q, k, v, world, used, rank, stats, recv, report=None):
    batch, total, heads, dim = q.shape
    length, local_heads = total // world, heads // world
    kwargs = dict(
        batch_size=batch,
        local_sequence=length,
        local_heads=local_heads,
        world_size=world,
        scale_sequence=used,
        enable_pdl=False,
    )
    out = layout.unpack_for_sage(recv, **kwargs)
    reused = tuple(torch.empty_like(t) for t in out)
    layout.unpack_for_sage(recv, out=reused, **kwargs)
    for a, b in zip(out, reused, strict=True):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    qi, ki, vp, qs, ks = out
    sl = slice(rank * local_heads, (rank + 1) * local_heads)
    mean = stats.k_mean_global[:, sl].float()
    for name, source, codes, scales, group in (
        ("q", q[:, :used, sl].float(), qi[:, :used], qs, layout.Q_GROUP),
        (
            "k",
            k[:, :used, sl].float() - mean.unsqueeze(1),
            ki[:, :used],
            ks,
            layout.K_GROUP,
        ),
    ):
        amax = global_amax(source, group)
        torch.testing.assert_close(
            scales[..., : amax.shape[-1]], amax / 127, rtol=2e-6, atol=0
        )
        reciprocal = (
            (127 / amax)
            .repeat_interleave(group, -1)[..., :used]
            .permute(0, 2, 1)
            .unsqueeze(-1)
        )
        expected = (source * reciprocal).round().clamp(-128, 127)
        delta = (codes.float() - expected).abs()
        if report is not None:
            scale_error = (scales[..., : amax.shape[-1]] - amax / 127).abs()
            report[name] = dict(
                scale_max_abs=scale_error.max().item(),
                scale_max_relative=(scale_error / (amax / 127)).max().item(),
                int8_mismatch_fraction=(delta != 0).float().mean().item(),
                int8_max_delta=delta.max().item(),
            )
        # Initial locked-toolchain diagnostic gate; full reports retain mismatch rates.
        assert delta.max() <= 1
    t = torch.arange(used, device=q.device)
    m = t % 16
    perm = t - m + (m // 8) * 2 + ((m // 2) % 4) * 4 + m % 2
    logical_v = vp[..., perm].permute(0, 3, 2, 1).contiguous()
    assert torch.count_nonzero(logical_v.view(torch.uint8)[..., :2]) == 0
    amax_v = v[:, :used].float().abs().amax(1)
    assert torch.equal(stats.v_scale_global, amax_v / 2.25)
    scale = stats.v_scale_global[:, sl].unsqueeze(1)
    reconstructed = logical_v.float() * scale
    source_v = v[:, :used, sl].float()
    # E4M3 has three mantissa bits, and subnormal spacing 2^-9.
    # Bound rounding by half an ULP, plus FP32 arithmetic slack.
    error_bound = source_v.abs() * 0.062501 + scale * (2**-10 + 1e-6)
    assert torch.all((reconstructed - source_v).abs() <= error_bound)
    if report is not None:
        reference_v = torch.where(scale == 0, 0.0, source_v / scale).to(
            torch.float8_e4m3fn
        )
        report["v"] = dict(
            fp8_byte_mismatch_fraction=(
                reference_v.contiguous().view(torch.uint8)
                != logical_v.view(torch.uint8)
            )
            .float()
            .mean()
            .item(),
            reconstruction_max_abs=(reconstructed - source_v).abs().max().item(),
            underflow_fraction=((source_v != 0) & (logical_v.float() == 0))
            .float()
            .mean()
            .item(),
        )
    assert torch.isfinite(qs).all() and torch.isfinite(ks).all()
    return out


@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("length", LENGTHS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_general_gpu(gpu_layout, world, length, dtype):
    layout = gpu_layout
    total = length * world
    for used in sorted({1, total, max(1, total - 1), min(total, layout.K_GROUP + 1)}):
        q, k, v = inputs(2, total, world, dtype, used, head_dim=layout.HEAD_DIM)
        local, stats, payloads = simulate(layout, q, k, v, world, used)
        for r, (xs, st, payload) in enumerate(zip(local, stats, payloads, strict=True)):
            assert st.stats_protocol == lowp.BOUNDARY_MERGE
            contiguous = layout.quant_and_pack(
                *(x.contiguous() for x in xs), st, enable_pdl=False
            )
            assert torch.equal(payload, contiguous)
            reused = torch.full_like(payload, 255)
            layout.quant_and_pack(*xs, st, out=reused, enable_pdl=False)
            assert torch.equal(payload, reused)
            spec = layout.payload_spec(
                batch_size=2, local_sequence=length, num_heads=world, world_size=world
            )
            assert torch.count_nonzero(payload[:, spec["raw_chunk_bytes"] :]) == 0
            for group, offset, count in (
                (layout.Q_GROUP, spec["q_scale_offset"], spec["q_slots_per_source"]),
                (layout.K_GROUP, spec["k_scale_offset"], spec["k_slots_per_source"]),
            ):
                touched = ((r + 1) * length - 1) // group - r * length // group + 1
                scales = (
                    payload[:, offset : offset + 2 * count * 4]
                    .contiguous()
                    .view(torch.float32)
                    .view(world, 2, 1, count)
                )
                assert torch.count_nonzero(scales[..., touched:]) == 0
            recv = torch.stack([p[r] for p in payloads])
            check_receiver(layout, q, k, v, world, used, r, st, recv)


@pytest.mark.parametrize("world", [2, 4])
def test_production_heads_gpu(gpu_layout, world):
    q, k, v = inputs(
        1,
        world * 129,
        28,
        torch.bfloat16,
        world * 129 - 1,
        head_dim=gpu_layout.HEAD_DIM,
    )
    _, stats, payloads = simulate(gpu_layout, q, k, v, world, world * 129 - 1)
    for r in range(world):
        check_receiver(
            gpu_layout,
            q,
            k,
            v,
            world,
            world * 129 - 1,
            r,
            stats[r],
            torch.stack([p[r] for p in payloads]),
        )


@pytest.mark.parametrize("layout_cls", LAYOUTS)
def test_geometry_rejects_invalid_inputs(layout_cls):
    layout = layout_cls()
    for name, value in (
        ("batch_size", 0),
        ("local_sequence", -1),
        ("num_heads", 3),
        ("world_size", 3),
    ):
        args = dict(batch_size=1, local_sequence=128, num_heads=8, world_size=4)
        args[name] = value
        with pytest.raises(ValueError):
            layout.payload_spec(**args)


def test_mean_rounding_contract():
    k = torch.tensor([256, 258], dtype=torch.bfloat16)
    mean32 = k.float().mean()
    assert mean32.item() == 257
    assert mean32.to(k.dtype).float().item() == 256


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fixed_stats_partition_invariance_gpu(gpu_layout, dtype):
    """Same arithmetic inputs must give identical live bytes across S/P."""
    layout, used, heads = gpu_layout, 129, 8
    q, k, v = inputs(1, used, heads, dtype, used, head_dim=layout.HEAD_DIM)
    mean = k.float().mean(1).to(dtype).contiguous()
    qmax = global_amax(q.float(), layout.Q_GROUP)
    kmax = global_amax(k.float() - mean.float().unsqueeze(1), layout.K_GROUP)
    vs = (v.float().abs().amax(1) / 2.25).contiguous()
    baseline = None
    for world, length in ((2, 65), (4, 33), (8, 17), (4, 128)):
        padded = [
            torch.zeros(
                1, world * length, heads, layout.HEAD_DIM, dtype=dtype, device=q.device
            )
            for _ in range(3)
        ]
        for dst, src in zip(padded, (q, k, v), strict=True):
            dst[:, :used] = src
        payloads = []
        for rank in range(world):
            final = []
            for group, ref in ((layout.Q_GROUP, qmax), (layout.K_GROUP, kmax)):
                am = torch.full(
                    (1, heads, (length + 2 * group - 2) // group), 1e-7, device=q.device
                )
                first = rank * length // group
                count = max(0, min(am.shape[-1], ref.shape[-1] - first))
                if count:
                    am[..., :count] = ref[..., first : first + count]
                final.append(am)
            st = lowp.V2GStats(
                lowp.BOUNDARY_MERGE,
                rank,
                world,
                used,
                mean,
                vs,
                *final,
                q_group=layout.Q_GROUP,
                k_group=layout.K_GROUP,
            )
            shard = [x[:, rank * length : (rank + 1) * length] for x in padded]
            payloads.append(layout.quant_and_pack(*shard, st, enable_pdl=False))
        outputs = []
        for rank in range(world):
            qi, ki, vp, qs, ks = layout.unpack_for_sage(
                torch.stack([p[rank] for p in payloads]),
                batch_size=1,
                local_sequence=length,
                local_heads=heads // world,
                world_size=world,
                scale_sequence=used,
                enable_pdl=False,
            )
            t = torch.arange(used, device=q.device)
            m = t % 16
            perm = t - m + (m // 8) * 2 + ((m // 2) % 4) * 4 + m % 2
            outputs.append(
                (qi[:, :used], ki[:, :used], vp[..., perm].permute(0, 3, 2, 1))
            )
        combined = tuple(
            torch.cat([o[i] for o in outputs], dim=2).contiguous().view(torch.uint8)
            for i in range(3)
        )
        if baseline is None:
            baseline = combined
        else:
            for a, b in zip(combined, baseline, strict=True):
                assert torch.equal(a, b)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("all_zero", [False, True])
def test_zero_v_canonical_and_pack_gpu(gpu_layout, dtype, all_zero):
    layout = gpu_layout
    q, k, v = inputs(1, 130, 2, dtype, 129, head_dim=layout.HEAD_DIM)
    if all_zero:
        v.zero_()
    local, stats, payloads = simulate(layout, q, k, v, 2, 129)
    mod = (
        lowp.get_ulysses_lowp_sm90_module()
        if layout.Q_GROUP == 16
        else lowp.get_ulysses_lowp_module()
    )
    spec = layout.payload_spec(
        batch_size=1, local_sequence=65, num_heads=2, world_size=2
    )
    for xs, st, payload in zip(local, stats, payloads, strict=True):
        canonical = torch.empty_like(
            xs[2], dtype=torch.uint8, memory_format=torch.contiguous_format
        )
        mod.ulysses_lowp_quant_v_fp8_with_scale(
            xs[2].contiguous(), st.v_scale_global, canonical, False
        )
        for dest in range(2):
            packed = (
                payload[dest, spec["v_offset"] : spec["q_scale_offset"]]
                .view(65, 1, 1, layout.HEAD_DIM)
                .permute(1, 0, 2, 3)
            )
            assert torch.equal(packed, canonical[:, :, dest : dest + 1])
        assert torch.count_nonzero(canonical[..., :2]) == 0
        if all_zero:
            assert torch.count_nonzero(canonical) == 0
        pdl_payload = layout.quant_and_pack(*xs, st, enable_pdl=True)
        assert torch.equal(payload, pdl_payload)


@pytest.mark.parametrize("layout_cls", LAYOUTS)
def test_context_validation_cpu(layout_cls):
    layout = layout_cls()
    k = torch.empty(1, 17, 4, 128, dtype=torch.bfloat16)
    ctx = lowp.StatsContext(
        lowp.BOUNDARY_MERGE,
        0,
        4,
        1,
        1,
        17,
        4,
        128,
        k.dtype,
        512,
        torch.zeros(1, 4, lowp.slots(17, layout.Q_GROUP)),
        (1, 4, 2),
        (1, 4, 2, 2, 128),
        layout.Q_GROUP,
        layout.K_GROUP,
    )
    gathered = torch.zeros(4, 4 * (6 * 128 + 2))
    lowp._validate_stats_context(gathered, ctx, k, layout.Q_GROUP, layout.K_GROUP)
    for bad in (0, -1, 69, 1.5, True):
        ctx.used_sequence = bad
        with pytest.raises(ValueError):
            lowp._validate_stats_context(
                gathered, ctx, k, layout.Q_GROUP, layout.K_GROUP
            )
    ctx.used_sequence = 1
    with pytest.raises(ValueError):
        lowp._validate_stats_context(
            gathered[:, :-1], ctx, k, layout.Q_GROUP, layout.K_GROUP
        )
    with pytest.raises(ValueError, match="layout"):
        lowp._validate_stats_context(
            gathered, ctx, k, 16 if layout.Q_GROUP == 32 else 32, layout.K_GROUP
        )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("case", ["smooth_k", "outlier", "small_v"])
def test_stress_distribution_gpu(gpu_layout, dtype, case):
    layout, world, used = gpu_layout, 8, 129
    q, k, v = inputs(1, 136, world, dtype, used, head_dim=layout.HEAD_DIM)
    if case == "smooth_k":
        k[:, :used] = 16 + (k[:, :used] - 16) * 1e-3
    elif case == "outlier":
        q[:, used - 1] *= 16
        k[:, used - 1] += 64
        v[:, used - 1] *= 32
    else:
        v[..., 2:] *= 1e-4
    _, stats, payloads = simulate(layout, q, k, v, world, used)
    for rank in range(world):
        check_receiver(
            layout,
            q,
            k,
            v,
            world,
            used,
            rank,
            stats[rank],
            torch.stack([p[rank] for p in payloads]),
        )


@pytest.mark.parametrize("layout_cls", LAYOUTS)
def test_head_dimension_geometry(layout_cls):
    args = dict(batch_size=2, local_sequence=65, num_heads=8, world_size=4)
    small = layout_cls(head_dim=64).payload_spec(**args)
    large = layout_cls().payload_spec(**args)
    assert small["main_bytes"] == 2 * 65 * 2 * 64
    assert large["main_bytes"] == 2 * small["main_bytes"]
    for name in (
        "q_scale_alloc",
        "k_scale_alloc",
        "q_slots_per_source",
        "k_slots_per_source",
    ):
        assert small[name] == large[name]
    assert (
        small["raw_chunk_bytes"] - 3 * small["main_bytes"]
        == large["raw_chunk_bytes"] - 3 * large["main_bytes"]
    )
    for dim in (0, 32, 96, 256, True, 64.0):
        with pytest.raises(ValueError):
            layout_cls(head_dim=dim)


def test_layout_dimension_mismatch_gpu(gpu_layout):
    layout = gpu_layout
    other = 128 if layout.HEAD_DIM == 64 else 64
    q, k, v = inputs(1, 130, 2, torch.bfloat16, 129, head_dim=other)
    with pytest.raises(ValueError, match="for this layout"):
        layout.local_stats(
            q[:, :65], k[:, :65], v[:, :65], rank=0, world_size=2, used_sequence=129
        )
    _, stats, payload = simulate(type(layout)(head_dim=other), q, k, v, 2, 129)
    out = torch.full_like(payload[0], 255)
    with pytest.raises(ValueError, match="for this layout"):
        layout.quant_and_pack(q[:, :65], k[:, :65], v[:, :65], stats[0], out=out)
    assert torch.all(out == 255)


def test_compiled_dimensions_gpu(gpu_layout):
    assert gpu_layout.is_supported()
    assert lowp.capability()["supported_head_dims"] == (64, 128)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_long_scale_rows_gpu(gpu_layout, dtype):
    """Exercise multiple statistics chunks and more scale slots than CTA threads."""
    layout, world, length = gpu_layout, 8, 1025
    used = world * length - 1
    q, k, v = inputs(1, world * length, world, dtype, used, head_dim=layout.HEAD_DIM)
    _, stats, payloads = simulate(layout, q, k, v, world, used)
    for rank in (0, world - 1):
        check_receiver(
            layout,
            q,
            k,
            v,
            world,
            used,
            rank,
            stats[rank],
            torch.stack([p[rank] for p in payloads]),
        )
