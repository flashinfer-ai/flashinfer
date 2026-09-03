"""Tests for flashinfer.comm.ulysses_lowp (V2-G payload ABI v3, stats
protocol 3 / ALIGN-128).

Four layers:
1. Structural: grid arithmetic, payload spec, preconditions (no GPU math).
2. Pipeline self-consistency (bit-exact): the packed V section must be
   byte-identical to the standalone V quantizer's output rearranged into the
   destination-major layout, and unpack must reproduce the packed bytes.
3. Numerical: quantized values match a pure-torch reference within the
   quantization step (the kernels compile under --use_fast_math, whose
   division may differ from IEEE torch division by ULPs, so the torch
   reference is a closeness oracle, not a bit oracle).
4. Optional bit-parity against the SageAttention fork reference when the
   package is importable (the authoritative byte gate lives out of tree).
"""

import pytest
import torch

import flashinfer.comm.ulysses_lowp as lowp

requires_sm120 = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (12, 0),
    reason="V2-G requires an SM120 CUDA device",
)

_HEADS = 56
_HEAD_DIM = 128
_SCALE_MAX = 2.25


# ---------------------------------------------------------------------------
# 1. Structural
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("local_sequence", "group", "expected_slots"),
    [
        (1, 32, 1),
        (32, 32, 2),
        (64, 32, 3),
        (64, 64, 2),
        (9440, 32, 296),
        (9440, 64, 149),
    ],
)
def test_slot_upper_bound_formula(local_sequence, group, expected_slots):
    assert lowp.slots(local_sequence, group) == expected_slots
    for world in (2, 4, 8):
        for rank in range(world):
            assert lowp.touched(rank, local_sequence, group) <= lowp.slots(
                local_sequence, group
            )


def test_owner_rule_holds_for_aligned_shapes():
    for world in (2, 4, 8):
        for L in (128, 256, 384, 1152):
            S = world * L
            for group in (32, 64):
                for g in range((S + group - 1) // group):
                    owner = lowp.owner(g, L, group)
                    assert 0 <= owner < world
                    assert (
                        lowp.group_first(owner, L, group)
                        <= g
                        <= lowp.group_last(owner, L, group)
                    )


def test_payload_spec_reference_shape():
    spec = lowp.payload_spec(
        batch_size=1, local_sequence=9440, num_heads=56, head_dim=128, world_size=4
    )
    assert spec["chunk_bytes"] == 50_774_400
    assert spec["payload_bytes"] == 203_097_600
    assert spec["chunk_bytes"] % 128 == 0
    assert spec["payload_reduction_pct"] == pytest.approx(49.9754, abs=1e-3)


def test_payload_spec_rejects_bad_world_size_and_heads():
    with pytest.raises(ValueError, match="world_size"):
        lowp.payload_spec(
            batch_size=1, local_sequence=128, num_heads=56, head_dim=128, world_size=3
        )
    with pytest.raises(ValueError, match="divisible"):
        lowp.payload_spec(
            batch_size=1, local_sequence=128, num_heads=54, head_dim=128, world_size=4
        )


@requires_sm120
def test_unpack_rejects_unaligned_local_sequence():
    recv = torch.zeros((4, 128), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="ALIGN-128"):
        lowp.unpack_for_sage(
            recv,
            batch_size=1,
            local_sequence=65,
            local_heads=14,
            head_dim=_HEAD_DIM,
            world_size=4,
        )


@requires_sm120
def test_capability_and_abi():
    assert lowp.abi_version() == 3
    cap = lowp.capability("cuda")
    assert cap["compiled_abi_version"] == 3
    assert cap["supported"] is True
    assert lowp.STATS_PROTOCOL == 3


# ---------------------------------------------------------------------------
# Shared pipeline driver (single-GPU simulation of all P ranks)
# ---------------------------------------------------------------------------


def _global_inputs(dtype, world, L, seed=20260902):
    torch.manual_seed(seed)
    q = torch.randn((1, world * L, _HEADS, _HEAD_DIM), device="cuda", dtype=dtype)
    return q, torch.randn_like(q), torch.randn_like(q)


def _stats(k, v, world, L):
    S = world * L
    k_sum = torch.zeros((1, _HEADS, _HEAD_DIM), dtype=torch.float32, device="cuda")
    v_amax = torch.zeros_like(k_sum)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        ks, va = lowp.k_sum_v_amax(k[:, s].contiguous(), v[:, s].contiguous())
        k_sum += ks
        v_amax = torch.maximum(v_amax, va)
    return (k_sum / S).to(k.dtype).contiguous(), (v_amax / _SCALE_MAX).contiguous()


def _run_pipeline(q, k, v, k_mean, v_scale, world, L):
    sends = []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
        k_amax = lowp.k_grouped_amax(k_r, k_mean, rank=r, world_size=world)
        sends.append(
            lowp.quant_qkv_pack(
                q_r,
                k_r,
                v_r,
                k_mean,
                q_amax,
                k_amax,
                v_scale,
                rank=r,
                world_size=world,
            )
        )
    return sends


# ---------------------------------------------------------------------------
# 2. Pipeline self-consistency (bit-exact within one compiled module)
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("local_sequence", [128, 384])
def test_packed_v_section_matches_standalone_quantizer(world, local_sequence):
    """The payload V section and quant_v_fp8_with_scale share one math path;
    their bytes must agree exactly after the destination-major rearrange."""
    L = local_sequence
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    local_heads = _HEADS // world
    sends = _run_pipeline(q, k, v, k_mean, v_scale, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        v_r = v[:, s].contiguous()
        canonical = lowp.quant_v_fp8_with_scale(v_r, v_scale)  # [1, L, H, D] uint8
        for d in range(world):
            section = sends[r][d][
                int(spec["v_offset"]) : int(spec["v_offset"]) + int(spec["main_bytes"])
            ].view(L, 1, local_heads, _HEAD_DIM)
            heads = slice(d * local_heads, (d + 1) * local_heads)
            expected = canonical[:, :, heads].permute(1, 0, 2, 3)  # [L, 1, h, D]
            assert torch.equal(section, expected.contiguous().view_as(section))


@requires_sm120
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_unpack_reproduces_packed_bytes(world, dtype):
    """unpack must be a pure rearrangement of the payload bytes: Q/K logical
    tensors and scale tensors must byte-match the corresponding payload
    sections gathered per source."""
    L = 256
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    local_heads = _HEADS // world
    sends = _run_pipeline(q, k, v, k_mean, v_scale, world, L)
    for d in range(world):
        recv = torch.stack([sends[r][d] for r in range(world)]).contiguous()
        q_u, k_u, v_u, q_scale_u, k_scale_u = lowp.unpack_for_sage(
            recv,
            batch_size=1,
            local_sequence=L,
            local_heads=local_heads,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        for name, out, offset in (
            ("q", q_u, 0),
            ("k", k_u, int(spec["k_offset"])),
        ):
            rebuilt = torch.cat(
                [
                    recv[src, offset : offset + int(spec["main_bytes"])]
                    .view(L, 1, local_heads, _HEAD_DIM)
                    .permute(1, 0, 2, 3)
                    for src in range(world)
                ],
                dim=1,
            )
            assert torch.equal(
                out.view(torch.uint8), rebuilt.contiguous().view(torch.uint8)
            ), name
        q_groups = L // 32
        k_groups = L // 64
        for src in range(world):
            q_section = (
                recv[
                    src,
                    int(spec["q_scale_offset"]) : int(spec["q_scale_offset"])
                    + local_heads * int(spec["q_slots_per_source"]) * 4,
                ]
                .view(torch.float32)
                .view(1, local_heads, -1)
            )
            assert torch.equal(
                q_scale_u[..., src * q_groups : (src + 1) * q_groups],
                q_section[..., :q_groups],
            )
            k_section = (
                recv[
                    src,
                    int(spec["k_scale_offset"]) : int(spec["k_scale_offset"])
                    + local_heads * int(spec["k_slots_per_source"]) * 4,
                ]
                .view(torch.float32)
                .view(1, local_heads, -1)
            )
            assert torch.equal(
                k_scale_u[..., src * k_groups : (src + 1) * k_groups],
                k_section[..., :k_groups],
            )


# ---------------------------------------------------------------------------
# 3. Numerical closeness vs pure-torch reference
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_quantized_values_close_to_torch_reference(dtype):
    """int8 codes within +-1 of the IEEE-division torch reference (the kernel
    compiles under --use_fast_math), scales exact."""
    world, L = 4, 256
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    local_heads = _HEADS // world
    r = 1
    s = slice(r * L, (r + 1) * L)
    q_r = q[:, s].contiguous()
    q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)

    # torch reference for this rank's touched groups
    g_first = lowp.group_first(r, L, 32)
    touched = lowp.touched(r, L, 32)
    qf = q_r.float()
    for slot in range(touched):
        g = g_first + slot
        lo = max(g * 32 - r * L, 0)
        hi = min((g + 1) * 32 - r * L, L)
        ref = qf[:, lo:hi].abs().amax(dim=(1, 3)).clamp_(min=1e-7)
        assert torch.equal(q_amax[..., slot], ref), f"q amax slot {slot}"

    k_amax = lowp.k_grouped_amax(k[:, s].contiguous(), k_mean, rank=r, world_size=world)
    send = lowp.quant_qkv_pack(
        q_r,
        k[:, s].contiguous(),
        v[:, s].contiguous(),
        k_mean,
        q_amax,
        k_amax,
        v_scale,
        rank=r,
        world_size=world,
    )
    dest = 0
    codes = (
        send[dest, : int(spec["main_bytes"])]
        .view(torch.int8)
        .view(L, 1, local_heads, _HEAD_DIM)
        .permute(1, 0, 2, 3)
        .float()
    )
    heads = slice(dest * local_heads, (dest + 1) * local_heads)
    scale = (q_amax[:, heads] / 127.0).float()
    token_groups = (torch.arange(L, device="cuda") + r * L) // 32 - g_first
    per_token_scale = scale[:, :, token_groups].permute(0, 2, 1).unsqueeze(-1)
    ref_codes = torch.clamp(
        torch.round(qf[:, :, heads] / per_token_scale.squeeze(-1).unsqueeze(-1)),
        -128,
        127,
    )
    assert (codes - ref_codes).abs().max() <= 1


# ---------------------------------------------------------------------------
# 4. Optional bit-parity vs the SageAttention fork
# ---------------------------------------------------------------------------

try:
    import sageattention.ulysses_v2g_ops as sage_v2g
except ImportError:  # only the fork-parity test below needs the reference
    sage_v2g = None


@requires_sm120
@pytest.mark.skipif(sage_v2g is None, reason="fork reference not installed")
@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("local_sequence", [128, 384])
def test_bit_parity_vs_fork(world, local_sequence):
    from sageattention.ulysses_ops import k_sum_v_amax as ref_ksva

    L = local_sequence
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    local_heads = _HEADS // world
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        ks_fi, va_fi = lowp.k_sum_v_amax(k_r, v_r)
        ks_rf, va_rf = ref_ksva(k_r, v_r)
        assert torch.equal(ks_fi, ks_rf) and torch.equal(va_fi, va_rf)
        qa_fi = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
        qa_rf = sage_v2g.q_grouped_amax_v2g(q_r, rank=r, world_size=world)
        assert torch.equal(qa_fi, qa_rf)
        ka_fi = lowp.k_grouped_amax(k_r, k_mean, rank=r, world_size=world)
        ka_rf = sage_v2g.k_grouped_amax_v2g(k_r, k_mean, rank=r, world_size=world)
        assert torch.equal(ka_fi, ka_rf)
        send_fi = lowp.quant_qkv_pack(
            q_r, k_r, v_r, k_mean, qa_fi, ka_fi, v_scale, rank=r, world_size=world
        )
        send_rf = sage_v2g.quant_qkv_pack_lowp_a2a_v2g(
            q_r, k_r, v_r, k_mean, qa_rf, ka_rf, v_scale, rank=r, world_size=world
        )
        assert torch.equal(send_fi, send_rf)
    # full-destination unpack parity
    sends_fi = _run_pipeline(q, k, v, k_mean, v_scale, world, L)
    for d in range(world):
        recv = torch.stack([sends_fi[r][d] for r in range(world)]).contiguous()
        o_fi = lowp.unpack_for_sage(
            recv,
            batch_size=1,
            local_sequence=L,
            local_heads=local_heads,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        o_rf = sage_v2g.unpack_lowp_a2a_v2g_for_sage(
            recv,
            batch_size=1,
            local_sequence=L,
            local_heads=local_heads,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        for x, y in zip(o_fi, o_rf, strict=True):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))


# ---------------------------------------------------------------------------
# 5. Protocol-2 (64-aligned global packing) additions
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("local_sequence", [65, 193, 1180])
def test_unaligned_unpack_reproduces_packed_bytes(world, local_sequence):
    """Protocol-2 receiver: Q/K logical tensors and owner-rule scale rebuild
    must byte-match the payload sections for arbitrary (unaligned) L."""
    L = local_sequence
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    local_heads = _HEADS // world
    S = world * L
    sends = _run_pipeline(q, k, v, k_mean, v_scale, world, L)
    d = world - 1
    recv = torch.stack([sends[r][d] for r in range(world)]).contiguous()
    q_u, k_u, v_u, q_scale_u, k_scale_u = lowp.unpack_for_sage(
        recv,
        batch_size=1,
        local_sequence=L,
        local_heads=local_heads,
        head_dim=_HEAD_DIM,
        world_size=world,
        aligned=False,
    )
    rebuilt_q = torch.cat(
        [
            recv[src, : int(spec["main_bytes"])]
            .view(L, 1, local_heads, _HEAD_DIM)
            .permute(1, 0, 2, 3)
            for src in range(world)
        ],
        dim=1,
    )
    assert torch.equal(q_u.view(torch.uint8), rebuilt_q.contiguous().view(torch.uint8))
    # owner-rule scale rebuild incl. deterministic zero tail
    q_groups_total = (S + 31) // 32
    for g in range(int(spec["q_scale_alloc"])):
        col = q_scale_u[..., g]
        if g >= q_groups_total:
            assert torch.all(col == 0.0), f"tail slot {g} not zero"
            continue
        owner = (g * 32) // L
        owner_slot = g - (owner * L) // 32
        section = (
            recv[
                owner,
                int(spec["q_scale_offset"]) : int(spec["q_scale_offset"])
                + local_heads * int(spec["q_slots_per_source"]) * 4,
            ]
            .view(torch.float32)
            .view(1, local_heads, -1)
        )
        assert torch.equal(col, section[..., owner_slot]), f"slot {g}"


@requires_sm120
@pytest.mark.parametrize("world", [4, 8])
def test_boundary_derive_matches_direct_computation(world):
    """derive_k_boundary_amax must reproduce, bit-exactly, the max over every
    touching rank's directly-computed |K - mean| partial amax."""
    L = 193
    S = world * L
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, _ = _stats(k, v, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        k_r = k[:, s].contiguous()
        ka = lowp.k_grouped_amax(k_r, k_mean, rank=r, world_size=world)
        gathered = torch.stack(
            [
                lowp.k_boundary_minmax(
                    k[:, o * L : (o + 1) * L].contiguous(), rank=o, world_size=world
                )
                for o in range(world)
            ]
        )
        lowp.derive_k_boundary_amax(
            ka, gathered, k_mean, rank=r, local_sequence=L, world_size=world
        )
        g_first = lowp.group_first(r, L, 64)
        touched_count = lowp.touched(r, L, 64)
        for g in {g_first, g_first + touched_count - 1}:
            direct = None
            for o in range(world):
                lo = max(g * 64, o * L)
                hi = min((g + 1) * 64, (o + 1) * L, S)
                if hi <= lo:
                    continue
                kc = k[:, lo:hi].float() - k_mean.float().unsqueeze(1)
                part = kc.abs().amax(dim=(1, 3)).clamp_(min=1e-7)
                direct = part if direct is None else torch.maximum(direct, part)
            assert torch.equal(ka[..., g - g_first], direct), f"group {g}"


@requires_sm120
def test_boundary_merge_is_idempotent_and_max():
    world, L = 4, 65
    q, _, _ = _global_inputs(torch.bfloat16, world, L)
    amaxes = [
        lowp.q_grouped_amax(
            q[:, r * L : (r + 1) * L].contiguous(), rank=r, world_size=world
        )
        for r in range(world)
    ]
    descs = torch.stack(
        [
            lowp.boundary_descriptors(
                amaxes[r], rank=r, local_sequence=L, group=32, world_size=world
            )
            for r in range(world)
        ]
    )
    once = [a.clone() for a in amaxes]
    for r in range(world):
        lowp.merge_boundary_amax(
            once[r], descs, rank=r, local_sequence=L, group=32, world_size=world
        )
    twice = [a.clone() for a in once]
    for r in range(world):
        lowp.merge_boundary_amax(
            twice[r], descs, rank=r, local_sequence=L, group=32, world_size=world
        )
    for a, b in zip(once, twice, strict=True):
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# 6. Fused amax+quant fast path (ALIGN-128)
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("world", [2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fused_pack_matches_split_path(world, dtype):
    L = 256
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
        k_amax = lowp.k_grouped_amax(k_r, k_mean, rank=r, world_size=world)
        split = lowp.quant_qkv_pack(
            q_r, k_r, v_r, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world
        )
        fused = lowp.quant_qkv_pack_fused(
            q_r, k_r, v_r, k_mean, v_scale, rank=r, world_size=world
        )
        assert torch.equal(split, fused)


@requires_sm120
def test_fused_pack_used_sequence_tail_repair_matches_split():
    world, L = 4, 128
    S = world * L
    for used in (S - 35, S - 64, S - 130):
        q, k, v = _global_inputs(torch.bfloat16, world, L)
        q[:, used:] = 0
        k[:, used:] = 0
        v[:, used:] = 0
        k_mean, v_scale = _stats(k, v, world, L)
        for r in range(world):
            s = slice(r * L, (r + 1) * L)
            q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
            q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
            k_amax = lowp.k_grouped_amax(
                k_r, k_mean, rank=r, world_size=world, used_sequence=used
            )
            split = lowp.quant_qkv_pack(
                q_r, k_r, v_r, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world
            )
            fused = lowp.quant_qkv_pack_fused(
                q_r,
                k_r,
                v_r,
                k_mean,
                v_scale,
                rank=r,
                world_size=world,
                used_sequence=used,
            )
            assert torch.equal(split, fused), f"used={used} r={r}"


@requires_sm120
def test_fused_pack_rejects_unaligned():
    q = torch.randn((1, 65, 56, 128), device="cuda", dtype=torch.bfloat16)
    k, v = torch.randn_like(q), torch.randn_like(q)
    km = torch.randn((1, 56, 128), device="cuda", dtype=torch.bfloat16).contiguous()
    vs = torch.rand((1, 56, 128), device="cuda", dtype=torch.float32).contiguous()
    with pytest.raises(ValueError, match="ALIGN-128"):
        lowp.quant_qkv_pack_fused(q, k, v, km, vs, rank=0, world_size=4)


# ---------------------------------------------------------------------------
# 7. Fused-projection interleaved views (head stride 3*D) are admitted and
#    produce the same bytes as contiguous copies
# ---------------------------------------------------------------------------


def _interleaved_inputs(dtype, world, L, seed=20260903):
    """q/k/v as views of one [B, S, H, 3, D] projection output (head stride
    3*D, token stride 3*H*D), exactly what a fused QKV GEMM emits."""
    torch.manual_seed(seed)
    qkv = torch.randn((1, world * L, _HEADS, 3, _HEAD_DIM), device="cuda", dtype=dtype)
    q, k, v = (qkv[..., i, :] for i in range(3))
    assert q.stride(-2) == 3 * _HEAD_DIM and not q.is_contiguous()
    return q, k, v


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_views_match_contiguous_stats(dtype):
    world, L = 4, 256
    _, k, v = _interleaved_inputs(dtype, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        ks_view, va_view = lowp.k_sum_v_amax(k[:, s], v[:, s])
        ks_ref, va_ref = lowp.k_sum_v_amax(k[:, s].contiguous(), v[:, s].contiguous())
        assert torch.equal(ks_view, ks_ref) and torch.equal(va_view, va_ref)


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_interleaved_views_match_contiguous_pack(dtype):
    world, L = 4, 256
    S = world * L
    used = S - 100
    q, k, v = _interleaved_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = q[:, s], k[:, s], v[:, s]
        q_c, k_c, v_c = (x.contiguous() for x in (q_r, k_r, v_r))
        q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
        k_amax = lowp.k_grouped_amax(
            k_r, k_mean, rank=r, world_size=world, used_sequence=used
        )
        assert torch.equal(q_amax, lowp.q_grouped_amax(q_c, rank=r, world_size=world))
        assert torch.equal(
            k_amax,
            lowp.k_grouped_amax(
                k_c, k_mean, rank=r, world_size=world, used_sequence=used
            ),
        )
        split_view = lowp.quant_qkv_pack(
            q_r, k_r, v_r, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world
        )
        split_ref = lowp.quant_qkv_pack(
            q_c, k_c, v_c, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world
        )
        assert torch.equal(split_view, split_ref)
        fused_view = lowp.quant_qkv_pack_fused(
            q_r, k_r, v_r, k_mean, v_scale, rank=r, world_size=world, used_sequence=used
        )
        assert torch.equal(fused_view, split_ref)


@requires_sm120
def test_interleaved_views_match_contiguous_boundary_minmax():
    world, L = 4, 1056  # protocol-2 shape: 64-groups straddle ranks
    _, k, _ = _interleaved_inputs(torch.bfloat16, world, L)
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        assert torch.equal(
            lowp.k_boundary_minmax(k[:, s], rank=r, world_size=world),
            lowp.k_boundary_minmax(k[:, s].contiguous(), rank=r, world_size=world),
        )


@requires_sm120
def test_rejects_views_the_vector_loads_cannot_address():
    base = torch.randn(
        (1, 128, _HEADS, 3 * _HEAD_DIM + 8), device="cuda", dtype=torch.bfloat16
    )
    misaligned = base[..., 1 : 1 + _HEAD_DIM]  # row starts 2 bytes past alignment
    with pytest.raises(ValueError, match="16-byte alignment"):
        lowp.q_grouped_amax(misaligned, rank=0, world_size=4)
    transposed = torch.randn(
        (1, 128, _HEAD_DIM, _HEADS), device="cuda", dtype=torch.bfloat16
    ).transpose(-1, -2)
    with pytest.raises(ValueError, match="contiguous along head_dim"):
        lowp.q_grouped_amax(transposed, rank=0, world_size=4)


# ---------------------------------------------------------------------------
# 8. Split fused entries: the Q half may overlap the statistics AllGather
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_split_fused_halves_compose_to_the_combined_bytes(dtype):
    world, L = 4, 256
    S = world * L
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    for used in (None, S - 130):
        for r in range(world):
            s = slice(r * L, (r + 1) * L)
            q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
            combined = lowp.quant_qkv_pack_fused(
                q_r,
                k_r,
                v_r,
                k_mean,
                v_scale,
                rank=r,
                world_size=world,
                used_sequence=used,
            )
            send = torch.full(
                (world, spec["chunk_bytes"]), 0xAB, dtype=torch.uint8, device="cuda"
            )
            lowp.zero_scale_and_padding(send, spec)
            # KV first, Q second: the halves touch disjoint bytes so the
            # order must not matter.
            lowp.quant_kv_into_payload_fused(
                k_r,
                v_r,
                k_mean,
                v_scale,
                send,
                rank=r,
                world_size=world,
                used_sequence=used,
            )
            lowp.quant_q_into_payload_fused(q_r, send, rank=r, world_size=world)
            assert torch.equal(send, combined), f"used={used} r={r}"


@requires_sm120
def test_split_fused_q_half_on_a_side_stream():
    """The intended schedule: Q packs on a side stream while the main stream
    would be waiting on the stats AllGather, KV packs afterwards."""
    world, L = 8, 512
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    side = torch.cuda.Stream()
    main = torch.cuda.current_stream()
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        combined = lowp.quant_qkv_pack_fused(
            q_r, k_r, v_r, k_mean, v_scale, rank=r, world_size=world
        )
        send = torch.empty(
            (world, spec["chunk_bytes"]), dtype=torch.uint8, device="cuda"
        )
        lowp.zero_scale_and_padding(send, spec)
        side.wait_stream(main)
        with torch.cuda.stream(side):
            lowp.quant_q_into_payload_fused(q_r, send, rank=r, world_size=world)
        lowp.quant_kv_into_payload_fused(
            k_r, v_r, k_mean, v_scale, send, rank=r, world_size=world
        )
        main.wait_stream(side)
        assert torch.equal(send, combined), f"r={r}"


@requires_sm120
def test_split_fused_halves_reject_unaligned():
    q = torch.randn((1, 65, 56, 128), device="cuda", dtype=torch.bfloat16)
    send = torch.zeros((4, 128), dtype=torch.uint8, device="cuda")
    with pytest.raises(ValueError, match="ALIGN-128"):
        lowp.quant_q_into_payload_fused(q, send, rank=0, world_size=4)
    km = torch.randn((1, 56, 128), device="cuda", dtype=torch.bfloat16)
    vs = torch.rand((1, 56, 128), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError, match="ALIGN-128"):
        lowp.quant_kv_into_payload_fused(
            q, q.clone(), km, vs, send, rank=0, world_size=4
        )


# ---------------------------------------------------------------------------
# 9. unpack_for_sage(scale_sequence=) emits consumer-width scale tensors
# ---------------------------------------------------------------------------


def test_scale_widths_match_sage_per_warp_contract():
    assert lowp.scale_widths(1024) == (32, 16)
    assert lowp.scale_widths(1023) == (32, 16)
    assert lowp.scale_widths(1025) == (36, 17)
    assert lowp.scale_widths(1) == (4, 1)
    with pytest.raises(ValueError):
        lowp.scale_widths(0)


@requires_sm120
@pytest.mark.parametrize(
    ("world", "local_sequence", "aligned"),
    [(4, 256, True), (8, 128, True), (4, 1056, False), (2, 96, False)],
)
def test_scale_sequence_is_the_prefix_of_the_full_width(world, local_sequence, aligned):
    """Unpack moves bytes, so any payload exercises the width contract; the
    narrow scale tensors must equal the leading slots of the full-width ones
    and Q/K/V must be untouched by the option."""
    local_heads = _HEADS // world
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=local_sequence,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    S = spec["logical_sequence"]
    torch.manual_seed(1)
    recv = torch.randint(
        0, 256, (world, spec["chunk_bytes"]), dtype=torch.uint8, device="cuda"
    )
    kwargs = dict(
        batch_size=1,
        local_sequence=local_sequence,
        local_heads=local_heads,
        head_dim=_HEAD_DIM,
        world_size=world,
        aligned=aligned,
    )
    q_full, k_full, v_full, qs_full, ks_full = lowp.unpack_for_sage(recv, **kwargs)
    assert qs_full.shape[-1] == spec["q_scale_alloc"]
    assert ks_full.shape[-1] == spec["k_scale_alloc"]
    for used in (S, S - 1, S - 64, S - 129, 1):
        if used <= 0:
            continue
        q_w, k_w = lowp.scale_widths(used)
        q, k, v, qs, ks = lowp.unpack_for_sage(recv, scale_sequence=used, **kwargs)
        assert qs.shape == (1, local_heads, q_w) and ks.shape == (1, local_heads, k_w)
        assert qs.is_contiguous() and ks.is_contiguous()
        # Random bytes decode to NaN in fp32/fp8 slots, so compare bit patterns.
        assert _same_bits(qs, qs_full[..., :q_w]), f"used={used}"
        assert _same_bits(ks, ks_full[..., :k_w]), f"used={used}"
        assert torch.equal(q, q_full) and torch.equal(k, k_full)
        assert _same_bits(v, v_full)
        # out= must be sized with scale_widths(used)
        outs = (
            torch.empty_like(q),
            torch.empty_like(k),
            torch.empty_like(v),
            torch.empty_like(qs),
            torch.empty_like(ks),
        )
        lowp.unpack_for_sage(recv, scale_sequence=used, out=outs, **kwargs)
        assert _same_bits(outs[3], qs) and _same_bits(outs[4], ks)


def _same_bits(a, b):
    return torch.equal(
        a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8)
    )


@requires_sm120
def test_scale_sequence_rejects_out_of_range_and_mismatched_out():
    world, local_sequence = 4, 256
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=local_sequence,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    S = spec["logical_sequence"]
    recv = torch.zeros((world, spec["chunk_bytes"]), dtype=torch.uint8, device="cuda")
    kwargs = dict(
        batch_size=1,
        local_sequence=local_sequence,
        local_heads=_HEADS // world,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    for bad in (0, S + 1):
        with pytest.raises(ValueError, match="scale_sequence"):
            lowp.unpack_for_sage(recv, scale_sequence=bad, **kwargs)
    full = lowp.unpack_for_sage(recv, **kwargs)
    # The FFI shape check rejects full-width scale buffers for a narrow request.
    with pytest.raises(Exception, match="q_scale has shape"):
        lowp.unpack_for_sage(recv, scale_sequence=S - 130, out=full, **kwargs)


# ---------------------------------------------------------------------------
# 10. Programmatic Dependent Launch: identical bytes with the attribute on/off
# ---------------------------------------------------------------------------


def _full_chain(q, k, v, k_mean, v_scale, world, L, used, enable_pdl):
    """Every kernel of the module, launched back-to-back on one stream with no
    host synchronization in between, so the PDL edges really overlap.  A
    misplaced griddepcontrol.wait shows up as a byte difference."""
    outs = []
    sends = []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        ks, va = lowp.k_sum_v_amax(k_r, v_r, enable_pdl=enable_pdl)
        qa = lowp.q_grouped_amax(q_r, rank=r, world_size=world, enable_pdl=enable_pdl)
        ka = lowp.k_grouped_amax(
            k_r,
            k_mean,
            rank=r,
            world_size=world,
            used_sequence=used,
            enable_pdl=enable_pdl,
        )
        split = lowp.quant_qkv_pack(
            q_r,
            k_r,
            v_r,
            k_mean,
            qa,
            ka,
            v_scale,
            rank=r,
            world_size=world,
            enable_pdl=enable_pdl,
        )
        fused = lowp.quant_qkv_pack_fused(
            q_r,
            k_r,
            v_r,
            k_mean,
            v_scale,
            rank=r,
            world_size=world,
            used_sequence=used,
            enable_pdl=enable_pdl,
        )
        v8 = lowp.quant_v_fp8_with_scale(v_r, v_scale, enable_pdl=enable_pdl)
        outs.extend([ks, va, qa, ka, split, fused, v8])
        sends.append(split)
    recv = torch.stack([sends[src][0] for src in range(world)])
    outs.extend(
        lowp.unpack_for_sage(
            recv,
            batch_size=1,
            local_sequence=L,
            local_heads=_HEADS // world,
            head_dim=_HEAD_DIM,
            world_size=world,
            scale_sequence=used,
            enable_pdl=enable_pdl,
        )
    )
    return outs


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_pdl_on_and_off_produce_identical_bytes(dtype):
    world, L = 4, 256
    S = world * L
    for seed in range(6):
        q, k, v = _global_inputs(dtype, world, L, seed=seed)
        k_mean, v_scale = _stats(k, v, world, L)
        on = _full_chain(q, k, v, k_mean, v_scale, world, L, S - 130, True)
        off = _full_chain(q, k, v, k_mean, v_scale, world, L, S - 130, False)
        assert len(on) == len(off)
        for i, (a, b) in enumerate(zip(on, off, strict=True)):
            assert _same_bits(a, b), f"seed={seed} output #{i} differs with PDL"


@requires_sm120
def test_pdl_launches_capture_into_a_cuda_graph():
    world, L = 4, 256
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    q_r, k_r, v_r = (x[:, :L].contiguous() for x in (q, k, v))
    eager = lowp.quant_qkv_pack_fused(
        q_r, k_r, v_r, k_mean, v_scale, rank=0, world_size=world, enable_pdl=True
    )
    send = torch.empty_like(eager)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):  # warm-up on the capture stream
        lowp.quant_qkv_pack_fused(
            q_r,
            k_r,
            v_r,
            k_mean,
            v_scale,
            rank=0,
            world_size=world,
            out=send,
            enable_pdl=True,
        )
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        lowp.quant_qkv_pack_fused(
            q_r,
            k_r,
            v_r,
            k_mean,
            v_scale,
            rank=0,
            world_size=world,
            out=send,
            enable_pdl=True,
        )
    send.fill_(0xCD)
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(send, eager)


@requires_sm120
@pytest.mark.parametrize("aligned_shape", [True, False])
def test_pdl_raw_edges_between_module_kernels(aligned_shape):
    """Back-to-back MODULE kernels with a real read-after-write dependency and
    no torch kernel in between: GroupedAmax -> pack reads the amax the
    previous grid wrote, Partial -> Combine reads the partials, fused Q ->
    fused K/V -> V share the payload.  Inputs are pre-contiguous and the
    payload pre-zeroed so the only launches on the stream are ours; a
    pdl_wait placed after a dependent read would surface here as a byte
    difference against the fully serialized (PDL off) run."""
    world, L = 4, (256 if aligned_shape else 1056)
    dtype = torch.bfloat16
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    spec = lowp.payload_spec(
        batch_size=1,
        local_sequence=L,
        num_heads=_HEADS,
        head_dim=_HEAD_DIM,
        world_size=world,
    )
    shards = [
        tuple(x[:, r * L : (r + 1) * L].contiguous() for x in (q, k, v))
        for r in range(world)
    ]
    sends = {
        pdl: [
            torch.empty((world, spec["chunk_bytes"]), dtype=torch.uint8, device="cuda")
            for _ in range(world)
        ]
        for pdl in (True, False)
    }
    fused = {
        pdl: [torch.empty_like(sends[pdl][0]) for _ in range(world)]
        for pdl in (True, False)
    }
    for pdl in (True, False):
        for s in sends[pdl] + fused[pdl]:
            lowp.zero_scale_and_padding(s, spec)
    torch.cuda.synchronize()
    stats = {}
    for pdl in (True, False):
        chain_stats = []
        for r, (q_r, k_r, v_r) in enumerate(shards):
            # Partial -> Combine (RAW on the partial workspaces)
            chain_stats.append(lowp.k_sum_v_amax(k_r, v_r, enable_pdl=pdl))
            # GroupedAmax(Q) -> Q pack (RAW on q_amax); GroupedAmax(K) -> K pack
            qa = lowp.q_grouped_amax(q_r, rank=r, world_size=world, enable_pdl=pdl)
            lowp.quant_q_into_payload(
                q_r, qa, sends[pdl][r], rank=r, world_size=world, enable_pdl=pdl
            )
            ka = lowp.k_grouped_amax(
                k_r, k_mean, rank=r, world_size=world, enable_pdl=pdl
            )
            lowp.quant_kv_into_payload(
                k_r,
                v_r,
                k_mean,
                ka,
                v_scale,
                sends[pdl][r],
                rank=r,
                world_size=world,
                enable_pdl=pdl,
            )
            if aligned_shape:
                # fused Q -> fused K -> V pack on one payload
                lowp.quant_q_into_payload_fused(
                    q_r, fused[pdl][r], rank=r, world_size=world, enable_pdl=pdl
                )
                lowp.quant_kv_into_payload_fused(
                    k_r,
                    v_r,
                    k_mean,
                    v_scale,
                    fused[pdl][r],
                    rank=r,
                    world_size=world,
                    enable_pdl=pdl,
                )
        stats[pdl] = chain_stats
        if not pdl:
            torch.cuda.synchronize()
    torch.cuda.synchronize()
    for r in range(world):
        assert torch.equal(sends[True][r], sends[False][r]), f"split payload r={r}"
        if aligned_shape:
            assert torch.equal(fused[True][r], fused[False][r]), f"fused payload r={r}"
            assert torch.equal(fused[True][r], sends[True][r]), f"fused!=split r={r}"
        for a, b in zip(stats[True][r], stats[False][r], strict=True):
            assert torch.equal(a, b), f"stats r={r}"


# ---------------------------------------------------------------------------
# 11. Packed (HMNMX2) Q amax must equal the fp32 fabsf/fmaxf loop on the
#     special values where the two could conceivably diverge
# ---------------------------------------------------------------------------


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_packed_q_amax_matches_split_path_on_special_values(dtype):
    world, L = 4, 256
    q, k, v = _global_inputs(dtype, world, L)
    k_mean, v_scale = _stats(k, v, world, L)
    finfo = torch.finfo(dtype)
    specials = {
        "all_subnormal": finfo.smallest_normal / 4,
        "signed_zero": -0.0,
        "inf": float("inf"),
        "nan": float("nan"),
        "max": finfo.max,
        "below_floor": 1e-8,
    }
    # One 32-token Q group per case, each on a different rank/head, plus a
    # NaN-only group and a group mixing NaN with finite values.
    for gi, value in enumerate(specials.values()):
        r, h = gi % world, gi
        rows = slice(r * L + gi * 32, r * L + (gi + 1) * 32)
        q[:, rows, h] = value
    q[:, 2 * L : 2 * L + 32, 20] = float("nan")
    q[:, 2 * L + 32 : 2 * L + 64, 21, ::2] = float("nan")
    q[:, 3 * L : 3 * L + 32, 22] = finfo.smallest_normal / 8  # subnormal group
    q[:, 3 * L + 5 : 3 * L + 6, 22] = 3.0  # ... with one normal winner
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        q_r, k_r, v_r = (x[:, s].contiguous() for x in (q, k, v))
        q_amax = lowp.q_grouped_amax(q_r, rank=r, world_size=world)
        k_amax = lowp.k_grouped_amax(k_r, k_mean, rank=r, world_size=world)
        split = lowp.quant_qkv_pack(
            q_r, k_r, v_r, k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world
        )
        fused = lowp.quant_qkv_pack_fused(
            q_r, k_r, v_r, k_mean, v_scale, rank=r, world_size=world
        )
        assert torch.equal(split, fused), f"rank {r}"


# ---------------------------------------------------------------------------
# 12. Automatic protocol routing (local_stats / finalize_stats /
#     quant_and_pack / unpack aligned=None)
# ---------------------------------------------------------------------------


def test_protocol_routing_structural():
    assert lowp.stats_protocol_for(256, 4) == 3
    assert lowp.stats_protocol_for(4736, 8) == 3
    assert lowp.stats_protocol_for(1180, 4) == 2
    assert lowp.stats_protocol_for(65, 8) == 2
    assert lowp.required_alignment(8, 3) == 1024
    assert lowp.required_alignment(4, 2) == 64
    with pytest.raises(ValueError, match="stats_protocol"):
        lowp.required_alignment(4, 1)
    assert lowp.aligned_length(37730, 8, 3) == 37888
    assert lowp.aligned_length(4685, 4, 2) == 4736
    assert lowp.aligned_length(1024, 4, 3) == 1024


def _reference_global_stats(k, v, world, L, used, dtype):
    """The production backends' exact reduction: rank-major stack,
    sum(dim=0) / live rows for the mean, amax(dim=0) / 2.25 for V."""
    ks, va = [], []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        a, b = lowp.k_sum_v_amax(k[:, s].contiguous(), v[:, s].contiguous())
        ks.append(a)
        va.append(b)
    denom = used if used is not None else world * L
    k_mean = (torch.stack(ks).sum(dim=0) / denom).to(dtype).contiguous()
    v_scale = (torch.stack(va).amax(dim=0) / _SCALE_MAX).contiguous()
    return k_mean, v_scale


def _auto_chain(q, k, v, world, L, used, flatten_gathered=False):
    """local_stats -> simulated AllGather -> finalize_stats -> quant_and_pack
    on every rank; returns (payloads, stats_list)."""
    sends, ctxs = [], []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        send, ctx = lowp.local_stats(
            q[:, s].contiguous(),
            k[:, s].contiguous(),
            v[:, s].contiguous(),
            rank=r,
            world_size=world,
            used_sequence=used,
        )
        sends.append(send)
        ctxs.append(ctx)
    gathered = torch.stack(sends).contiguous()
    if flatten_gathered:
        gathered = gathered.view(-1)
    payloads, stats_list = [], []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        st = lowp.finalize_stats(gathered, ctxs[r], k[:, s].contiguous())
        payloads.append(
            lowp.quant_and_pack(
                q[:, s].contiguous(), k[:, s].contiguous(), v[:, s].contiguous(), st
            )
        )
        stats_list.append(st)
    return payloads, stats_list


@requires_sm120
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("flatten_gathered", [False, True])
def test_auto_routing_matches_explicit_protocol3(dtype, flatten_gathered):
    world, L = 4, 256
    S = world * L
    for used in (None, S - 130):
        q, k, v = _global_inputs(dtype, world, L)
        if used is not None:
            q[:, used:] = 0
            k[:, used:] = 0
            v[:, used:] = 0
        k_mean, v_scale = _reference_global_stats(k, v, world, L, used, dtype)
        payloads, stats_list = _auto_chain(
            q, k, v, world, L, used, flatten_gathered=flatten_gathered
        )
        for r in range(world):
            st = stats_list[r]
            assert st.stats_protocol == 3
            assert st.q_amax_final is None and st.k_amax_final is None
            assert _same_bits(st.k_mean_global, k_mean)
            assert _same_bits(st.v_scale_global, v_scale)
            s = slice(r * L, (r + 1) * L)
            explicit = lowp.quant_qkv_pack_fused(
                q[:, s].contiguous(),
                k[:, s].contiguous(),
                v[:, s].contiguous(),
                k_mean,
                v_scale,
                rank=r,
                world_size=world,
                used_sequence=used,
            )
            assert torch.equal(payloads[r], explicit), f"used={used} r={r}"
        recv = torch.stack([payloads[src][0] for src in range(world)]).contiguous()
        common = dict(
            batch_size=1,
            local_sequence=L,
            local_heads=_HEADS // world,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        auto = lowp.unpack_for_sage(recv, aligned=None, **common)
        explicit = lowp.unpack_for_sage(recv, aligned=True, **common)
        for a, b in zip(auto, explicit, strict=True):
            assert _same_bits(a, b)


def _reference_protocol2_payloads(q, k, v, world, L, used, k_mean, v_scale):
    """The protocol-2 pipeline composed exactly as the validated sglang
    wiring does it: descriptors and raw-K min/max ride the (simulated)
    AllGather, merge/derive finalize the boundary slots, split pack."""
    q_amaxes, descs, minmaxes = [], [], []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        qa = lowp.q_grouped_amax(q[:, s].contiguous(), rank=r, world_size=world)
        descs.append(
            lowp.boundary_descriptors(
                qa, rank=r, local_sequence=L, group=32, world_size=world
            )
        )
        minmaxes.append(
            lowp.k_boundary_minmax(
                k[:, s].contiguous(), rank=r, world_size=world, used_sequence=used
            )
        )
        q_amaxes.append(qa)
    qg = torch.stack(descs).contiguous()
    mmg = torch.stack(minmaxes).contiguous()
    payloads = []
    for r in range(world):
        s = slice(r * L, (r + 1) * L)
        lowp.merge_boundary_amax(
            q_amaxes[r], qg, rank=r, local_sequence=L, group=32, world_size=world
        )
        ka = lowp.k_grouped_amax(
            k[:, s].contiguous(),
            k_mean,
            rank=r,
            world_size=world,
            used_sequence=used,
        )
        lowp.derive_k_boundary_amax(
            ka, mmg, k_mean, rank=r, local_sequence=L, world_size=world
        )
        payloads.append(
            lowp.quant_qkv_pack(
                q[:, s].contiguous(),
                k[:, s].contiguous(),
                v[:, s].contiguous(),
                k_mean,
                q_amaxes[r],
                ka,
                v_scale,
                rank=r,
                world_size=world,
            )
        )
    return payloads


@requires_sm120
@pytest.mark.parametrize(("world", "local_sequence"), [(4, 1180), (8, 65), (4, 4720)])
def test_auto_routing_matches_explicit_protocol2(world, local_sequence):
    L = local_sequence
    S = world * L
    dtype = torch.bfloat16
    # Protocol 2's legal domain keeps the tail padding inside the last
    # global K group (ceil(used/64) == ceil(S/64)).
    padded_used = max(S - 35, 64 * ((S + 63) // 64 - 1) + 1)
    for used in (None, padded_used):
        q, k, v = _global_inputs(dtype, world, L)
        if used is not None:
            q[:, used:] = 0
            k[:, used:] = 0
            v[:, used:] = 0
        k_mean, v_scale = _reference_global_stats(k, v, world, L, used, dtype)
        reference = _reference_protocol2_payloads(
            q, k, v, world, L, used, k_mean, v_scale
        )
        payloads, stats_list = _auto_chain(q, k, v, world, L, used)
        for r in range(world):
            assert stats_list[r].stats_protocol == 2
            assert _same_bits(stats_list[r].k_mean_global, k_mean)
            assert torch.equal(payloads[r], reference[r]), f"used={used} r={r}"
        recv = torch.stack([payloads[src][0] for src in range(world)]).contiguous()
        common = dict(
            batch_size=1,
            local_sequence=L,
            local_heads=_HEADS // world,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        auto = lowp.unpack_for_sage(recv, aligned=None, **common)
        explicit = lowp.unpack_for_sage(recv, aligned=False, **common)
        for a, b in zip(auto, explicit, strict=True):
            assert _same_bits(a, b)


@requires_sm120
@pytest.mark.parametrize("world", [4, 8])
@pytest.mark.parametrize("local_sequence", [128, 256])
def test_protocol2_machinery_equals_protocol3_on_aligned_shards(world, local_sequence):
    """The theorem behind automatic routing: on 128-aligned shards the FULL
    protocol-2 pipeline (descriptors, min/max, merge, derive, split pack,
    per-token unpack) produces byte-identical payloads and unpacked outputs
    to the protocol-3 fast path (fused pack, tile-hoisted unpack), because
    no group crosses a rank boundary: the merge degenerates to the local
    amax and the derive to the kernel's own partials."""
    L = local_sequence
    S = world * L
    dtype = torch.bfloat16
    for used in (None, S - 35):
        q, k, v = _global_inputs(dtype, world, L)
        if used is not None:
            q[:, used:] = 0
            k[:, used:] = 0
            v[:, used:] = 0
        k_mean, v_scale = _reference_global_stats(k, v, world, L, used, dtype)
        p2 = _reference_protocol2_payloads(q, k, v, world, L, used, k_mean, v_scale)
        for r in range(world):
            s = slice(r * L, (r + 1) * L)
            p3 = lowp.quant_qkv_pack_fused(
                q[:, s].contiguous(),
                k[:, s].contiguous(),
                v[:, s].contiguous(),
                k_mean,
                v_scale,
                rank=r,
                world_size=world,
                used_sequence=used,
            )
            assert torch.equal(p2[r], p3), f"used={used} r={r}"
        recv = torch.stack([p2[src][0] for src in range(world)]).contiguous()
        common = dict(
            batch_size=1,
            local_sequence=L,
            local_heads=_HEADS // world,
            head_dim=_HEAD_DIM,
            world_size=world,
        )
        via_p2 = lowp.unpack_for_sage(recv, aligned=False, **common)
        via_p3 = lowp.unpack_for_sage(recv, aligned=True, **common)
        for name, a, b in zip(
            ("q", "k", "v", "q_scale", "k_scale"), via_p2, via_p3, strict=True
        ):
            assert _same_bits(a, b), f"used={used} unpack {name}"


@requires_sm120
def test_auto_routing_rejects_bad_inputs():
    world, L = 4, 256
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    q_r, k_r, v_r = (x[:, :L].contiguous() for x in (q, k, v))
    send, ctx = lowp.local_stats(q_r, k_r, v_r, rank=0, world_size=world)
    with pytest.raises(ValueError, match="gathered has"):
        lowp.finalize_stats(torch.stack([send] * 3), ctx, k_r)
    with pytest.raises(TypeError, match="fp32"):
        lowp.finalize_stats(torch.stack([send] * world).to(torch.float16), ctx, k_r)
    with pytest.raises(ValueError, match="same shard"):
        lowp.finalize_stats(torch.stack([send] * world), ctx, k_r.to(torch.float16))
    with pytest.raises(TypeError, match="V2GStats"):
        lowp.quant_and_pack(q_r, k_r, v_r, {"stats_protocol": 3})
    with pytest.raises(TypeError, match="StatsContext"):
        lowp.finalize_stats(torch.stack([send] * world), {"rank": 0}, k_r)


@requires_sm120
def test_routing_dataclasses_repr_without_touching_tensor_contents():
    """StatsContext / V2GStats flow through @flashinfer_api-logged calls; their
    repr must describe tensors by shape/dtype/device only (no D2H copy)."""
    world, L = 4, 1180  # protocol 2: q_amax rides in the context
    q, k, v = _global_inputs(torch.bfloat16, world, L)
    q_r, k_r, v_r = (x[:, :L].contiguous() for x in (q, k, v))
    send, ctx = lowp.local_stats(q_r, k_r, v_r, rank=0, world_size=world)
    st = lowp.finalize_stats(torch.stack([send] * world), ctx, k_r)
    for text in (repr(ctx), repr(st)):
        assert "Tensor(" in text and "tensor(" not in text and "[" in text
    assert "stats_protocol=2" in repr(st)
