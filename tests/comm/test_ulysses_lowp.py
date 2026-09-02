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
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability(0) != (12, 0),
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
    [(1, 32, 1), (32, 32, 2), (64, 32, 3), (64, 64, 2), (9440, 32, 296), (9440, 64, 149)],
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
                q_r, k_r, v_r, k_mean, q_amax, k_amax, v_scale,
                rank=r, world_size=world,
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
        batch_size=1, local_sequence=L, num_heads=_HEADS,
        head_dim=_HEAD_DIM, world_size=world,
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
        batch_size=1, local_sequence=L, num_heads=_HEADS,
        head_dim=_HEAD_DIM, world_size=world,
    )
    local_heads = _HEADS // world
    sends = _run_pipeline(q, k, v, k_mean, v_scale, world, L)
    for d in range(world):
        recv = torch.stack([sends[r][d] for r in range(world)]).contiguous()
        q_u, k_u, v_u, q_scale_u, k_scale_u = lowp.unpack_for_sage(
            recv, batch_size=1, local_sequence=L,
            local_heads=local_heads, head_dim=_HEAD_DIM, world_size=world,
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
            q_section = recv[
                src,
                int(spec["q_scale_offset"]) : int(spec["q_scale_offset"])
                + local_heads * int(spec["q_slots_per_source"]) * 4,
            ].view(torch.float32).view(1, local_heads, -1)
            assert torch.equal(
                q_scale_u[..., src * q_groups : (src + 1) * q_groups],
                q_section[..., :q_groups],
            )
            k_section = recv[
                src,
                int(spec["k_scale_offset"]) : int(spec["k_scale_offset"])
                + local_heads * int(spec["k_slots_per_source"]) * 4,
            ].view(torch.float32).view(1, local_heads, -1)
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
        batch_size=1, local_sequence=L, num_heads=_HEADS,
        head_dim=_HEAD_DIM, world_size=world,
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
        q_r, k[:, s].contiguous(), v[:, s].contiguous(),
        k_mean, q_amax, k_amax, v_scale, rank=r, world_size=world,
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
    token_groups = (
        torch.arange(L, device="cuda") + r * L
    ) // 32 - g_first
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

sage_v2g = pytest.importorskip(
    "sageattention.ulysses_v2g_ops", reason="fork reference not installed"
)


@requires_sm120
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
            recv, batch_size=1, local_sequence=L,
            local_heads=local_heads, head_dim=_HEAD_DIM, world_size=world,
        )
        o_rf = sage_v2g.unpack_lowp_a2a_v2g_for_sage(
            recv, batch_size=1, local_sequence=L,
            local_heads=local_heads, head_dim=_HEAD_DIM, world_size=world,
        )
        for x, y in zip(o_fi, o_rf):
            assert torch.equal(x.view(torch.uint8), y.view(torch.uint8))
