# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Host-side (no GPU) contracts of the Cake SM120 MXFP4 x MXFP8 MegaMoE backend."""

from __future__ import annotations

import pytest
import torch
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_config import (
    Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
)
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_staging import (
    dequantize_activations_k32,
    quantize_activations_k32,
)
from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_weights import (
    block128_scales_to_k32,
    gate_up_interleave_index,
    interleave_gate_up_rows,
    pack_scales_kblock_major,
    scale_vector_layout,
    scale_vector_layout_inverse,
    transform_prequantized_weights,
    unpack_scales_kblock_major,
)


def test_config_defaults_and_variant() -> None:
    config = Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig()
    assert config.intermediate_size == 2048
    assert config.top_k == 6
    assert config.swiglu_limit == 10
    assert config.codec_overflow_blocks == 16
    assert config.transport_table == "all_gin"
    assert config.kernel_name == "sm120_mxfp4_mxfp8_bf16_cake"
    assert config.variant == "noshared"
    assert (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(shared_expert="fp8_in_kernel").variant
        == "fp8shared"
    )


def test_config_rejects_unknown_shared_expert() -> None:
    with pytest.raises(ValueError):
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(shared_expert="bf16")


def test_config_swiglu_limit_validation() -> None:
    assert Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(swiglu_limit=0).swiglu_limit == 0
    for bad in (-1, 7.5, True, "10"):
        with pytest.raises(ValueError):
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(swiglu_limit=bad)


def test_config_codec_overflow_blocks_validation() -> None:
    for good in (16, 28, 256):
        assert (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                codec_overflow_blocks=good
            ).codec_overflow_blocks
            == good
        )
    for bad in (0, 15, 257, 16.0, True, "16"):
        with pytest.raises(ValueError):
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(codec_overflow_blocks=bad)


def test_gate_up_interleave_is_grouped_in_eight_rows() -> None:
    intermediate = 16
    index = gate_up_interleave_index(intermediate).tolist()
    # 16-row groups: 8 gate rows (logical 0..7) then the matching 8 up rows (I + 0..7).
    assert index[:16] == list(range(0, 8)) + [intermediate + r for r in range(0, 8)]
    assert index[16:32] == list(range(8, 16)) + [intermediate + r for r in range(8, 16)]
    assert sorted(index) == list(range(2 * intermediate))


def test_interleave_gate_up_rows_is_a_permutation_with_inverse() -> None:
    experts, intermediate, cols = 2, 32, 8
    stacked = torch.arange(experts * 2 * intermediate * cols, dtype=torch.int32).view(
        experts, 2 * intermediate, cols
    )
    kernel_layout = interleave_gate_up_rows(stacked)
    assert kernel_layout.shape == stacked.shape
    inverse = torch.argsort(gate_up_interleave_index(intermediate))
    assert torch.equal(kernel_layout.index_select(1, inverse), stacked)
    # Row 0 of every 16-row group is gate row 8*g; row 8 is up row 8*g.
    for g in range(2 * intermediate // 16):
        assert torch.equal(kernel_layout[0, 16 * g], stacked[0, 8 * g])
        assert torch.equal(
            kernel_layout[0, 16 * g + 8], stacked[0, intermediate + 8 * g]
        )


def test_scale_packing_round_trips_and_is_kblock_major() -> None:
    experts, n, hidden = 3, 64, 512
    k_blocks = hidden // 32
    scales = torch.randint(0, 255, (experts, n, k_blocks), dtype=torch.uint8)
    packed = pack_scales_kblock_major(scales)
    assert packed.dtype == torch.uint32
    assert tuple(packed.shape) == (experts * (hidden // 128), n)
    assert torch.equal(unpack_scales_kblock_major(packed, experts=experts), scales)
    # Word (e, kb, n) holds blocks 4kb..4kb+3 of column n, block 4kb in the low byte.
    e, kb, col = 1, 2, 17
    word = int(packed[e * (hidden // 128) + kb, col].item())
    for j in range(4):
        assert (word >> (8 * j)) & 0xFF == int(scales[e, col, 4 * kb + j])


def test_activation_quantization_matches_reference_rule() -> None:
    torch.manual_seed(0)
    x = torch.randn(64, 256, dtype=torch.bfloat16) * 3.0
    codes, exponents = quantize_activations_k32(x)
    assert codes.dtype == torch.uint8 and tuple(codes.shape) == (64, 256)
    assert exponents.dtype == torch.uint8 and tuple(exponents.shape) == (64, 8)
    # Scale is the smallest power of two >= amax / 448, so |code| <= 448 always.
    dequant = dequantize_activations_k32(codes, exponents)
    scaled = codes.view(torch.float8_e4m3fn).float().abs()
    assert float(scaled.max()) <= 448.0
    grouped_amax = x.float().reshape(64, 8, 32).abs().amax(dim=2)
    scale = (exponents.to(torch.int32) << 23).view(torch.float32)
    assert torch.all(scale * 448.0 >= grouped_amax)
    assert torch.all(scale * 448.0 < grouped_amax * 2.0 + 1e-30)
    rel_l2 = torch.linalg.norm(dequant - x.float()) / torch.linalg.norm(x.float())
    assert float(rel_l2) < 0.08  # e4m3 mantissa (3 bits) + block scale


def test_zero_block_quantizes_to_zero() -> None:
    x = torch.zeros(2, 64, dtype=torch.bfloat16)
    codes, exponents = quantize_activations_k32(x)
    assert int(codes.max()) == 0
    assert torch.equal(dequantize_activations_k32(codes, exponents), torch.zeros(2, 64))


# --------------------------------------------------------------------------- hybrid closure


def _reference_scale_vectors(source: torch.Tensor) -> torch.Tensor:
    """The generator harness's bit-preserving N-tile/group transpose, verbatim."""

    columns = source.shape[-1]
    tiles = (
        source.view(torch.int32).reshape(-1, columns // 128, 16, 8).transpose(-1, -2)
    )
    group = torch.arange(8).reshape(8, 1)
    subtile = torch.arange(16).reshape(1, 16)
    permutation = subtile ^ ((group >> 1) * 4)
    index = permutation.reshape(1, 1, 8, 16).expand(tiles.shape)
    return (
        torch.gather(tiles, -1, index).contiguous().reshape(-1, 16).view(torch.uint32)
    )


def test_scale_vector_layout_matches_generator_and_inverts() -> None:
    torch.manual_seed(1)
    packed = torch.randint(0, 2**31 - 1, (6, 512), dtype=torch.int32).view(torch.uint32)
    vectors = scale_vector_layout(packed)
    assert tuple(vectors.shape) == (6 * 512 // 16, 16)
    assert torch.equal(vectors, _reference_scale_vectors(packed))
    assert torch.equal(scale_vector_layout_inverse(vectors, columns=512), packed)


def test_block128_scales_expand_exactly() -> None:
    blocks = torch.tensor([[120, 121], [122, 123]], dtype=torch.uint8)
    sf = block128_scales_to_k32(blocks, 256, 256)
    assert tuple(sf.shape) == (256, 8)
    assert int(sf[0, 0]) == 120 and int(sf[0, 4]) == 121
    assert int(sf[128, 3]) == 122 and int(sf[255, 7]) == 123


def _small_routed(experts: int, hidden: int, inter: int):
    g = torch.Generator().manual_seed(3)
    w13 = torch.randint(
        0, 255, (experts, 2 * inter, hidden // 2), dtype=torch.uint8, generator=g
    )
    w2 = torch.randint(
        0, 255, (experts, hidden, inter // 2), dtype=torch.uint8, generator=g
    )
    s13 = torch.randint(
        100, 130, (experts, 2 * inter, hidden // 32), dtype=torch.uint8, generator=g
    )
    s2 = torch.randint(
        100, 130, (experts, hidden, inter // 32), dtype=torch.uint8, generator=g
    )
    return w13, w2, s13, s2


def test_transform_routed_only_appends_constant_shared_segment() -> None:
    experts, hidden, inter = 2, 256, 128
    w13, w2, s13, s2 = _small_routed(experts, hidden, inter)
    t = transform_prequantized_weights(
        w13, w2, s13, s2, hidden_size=hidden, intermediate_size=inter
    )
    assert not t.has_shared_expert
    w1_rows = scale_vector_layout_inverse(t.w1_sfb, columns=2 * inter)
    assert tuple(w1_rows.shape) == ((experts + 1) * hidden // 128, 2 * inter)
    assert torch.equal(
        w1_rows[: experts * hidden // 128],
        pack_scales_kblock_major(interleave_gate_up_rows(s13)),
    )
    assert bool(
        (w1_rows[experts * hidden // 128 :].view(torch.int32) == 0x7F7F7F7F).all()
    )
    assert bool((t.w1_b8 == 0x38).all()) and tuple(t.w1_b8.shape) == (2 * inter, hidden)
    assert torch.equal(t.w1_fp4, interleave_gate_up_rows(w13))


def test_transform_shared_expert_layout() -> None:
    experts, hidden, inter = 1, 256, 128
    w13, w2, s13, s2 = _small_routed(experts, hidden, inter)
    g = torch.Generator().manual_seed(4)
    sw13 = torch.randint(0, 120, (2 * inter, hidden), dtype=torch.uint8, generator=g)
    sw2 = torch.randint(0, 120, (hidden, inter), dtype=torch.uint8, generator=g)
    ss13 = torch.randint(
        110, 125, (2 * inter, hidden // 32), dtype=torch.uint8, generator=g
    )
    ss2 = torch.randint(110, 125, (hidden, inter // 32), dtype=torch.uint8, generator=g)
    t = transform_prequantized_weights(
        w13,
        w2,
        s13,
        s2,
        hidden_size=hidden,
        intermediate_size=inter,
        shared_w13=sw13,
        shared_w2=sw2,
        shared_w13_scale=ss13,
        shared_w2_scale=ss2,
    )
    assert t.has_shared_expert
    assert torch.equal(t.w1_b8, sw13.index_select(0, gate_up_interleave_index(inter)))
    assert torch.equal(t.w2_b8, sw2)
    w2_rows = scale_vector_layout_inverse(t.w2_sfb, columns=hidden)
    assert torch.equal(
        w2_rows[experts * inter // 128 :], pack_scales_kblock_major(ss2.unsqueeze(0))
    )


def test_selection_rule_prefers_uniform_unless_mixed_clearly_wins() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_transport import (
        TRANSPORT_GIN,
        TRANSPORT_LSA,
        evaluate_selection,
        lsa_where_reachable,
        uniform_table,
    )

    world = 4
    pairs = [[a // 2 == b // 2 for b in range(world)] for a in range(world)]
    gin, lsa = uniform_table(world, TRANSPORT_GIN), uniform_table(world, TRANSPORT_LSA)
    mixed = lsa_where_reachable(world, pairs)
    assert mixed.get(0, 1) == TRANSPORT_LSA and mixed.get(0, 2) == TRANSPORT_GIN
    # all_lsa within the margin of all_gin: all_gin (keeps the copier-free instance).
    chosen = evaluate_selection(
        [(gin, 1.00, [1.00] * 3), (lsa, 0.98, [0.98] * 3)]
    ).chosen
    assert chosen.name == "all_gin"
    # all_lsa beats all_gin by more than 3 %.
    chosen = evaluate_selection(
        [(gin, 1.00, [1.00] * 3), (lsa, 0.90, [0.90] * 3)]
    ).chosen
    assert chosen.name == "all_lsa"
    # A mixed table needs a 3 % win over the best uniform score and stable rounds.
    chosen = evaluate_selection(
        [(gin, 1.00, [1.0] * 3), (mixed, 0.98, [0.98] * 3)]
    ).chosen
    assert chosen.name == "all_gin"
    chosen = evaluate_selection(
        [(gin, 1.00, [1.0] * 3), (mixed, 0.90, [0.90, 0.95, 2.0])]
    ).chosen
    assert chosen.name == "all_gin"
    chosen = evaluate_selection(
        [(gin, 1.00, [1.0] * 3), (mixed, 0.90, [0.90, 0.91, 0.92])]
    ).chosen
    assert chosen.name == "lsa_where_reachable"


def test_validate_table_rejects_unreachable_lsa_and_missing_gin() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_transport import (
        TRANSPORT_GIN,
        TRANSPORT_LSA,
        uniform_table,
        validate_table,
    )

    two_hosts = [[a // 2 == b // 2 for b in range(4)] for a in range(4)]
    validate_table(uniform_table(4, TRANSPORT_GIN), two_hosts, True)
    with pytest.raises(ValueError):
        validate_table(uniform_table(4, TRANSPORT_LSA), two_hosts, True)
    with pytest.raises(ValueError):
        validate_table(uniform_table(4, TRANSPORT_GIN), two_hosts, False)


def test_residency_grid_rule() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        residency_grid,
    )

    C = {
        "W1_N_BLOCKS": 32,
        "W2_N_BLOCKS": 32,
        "PR_COPIER_CTAS_EFFECTIVE": 4,
        "C56_MB_ROWS": 256,
        "FRONT_EXCLUDED_CTAS": 4,
    }
    common = dict(
        local_experts=2, task_m=64, capacity=188, C=C, resident_peer_fold=False
    )
    # Rank 1 receives one route: one task -> 32 tiles bound the workers.
    grid = residency_grid(
        rows_by_rank=[4, 0],
        expert_counts=[3, 0, 1, 0],
        with_shared_expert=False,
        **common,
    )
    assert grid == 33
    # The shared expert adds ceil(rows / task_m) tasks on every rank.
    grid = residency_grid(
        rows_by_rank=[4, 70],
        expert_counts=[3, 0, 1, 0],
        with_shared_expert=True,
        **common,
    )
    assert grid == 2 * 32 + 1
    # Sixteen ranks, a rank without tasks, 12 tiles per task: the tile bound (12 workers)
    # is below the structural minimum FRONT_EXCLUDED_CTAS + world + 1 = 21; the grid is raised to it.
    C16 = dict(C, W1_N_BLOCKS=12, W2_N_BLOCKS=16)
    grid = residency_grid(
        rows_by_rank=[6] + [0] * 15,
        expert_counts=[6] + [0] * 31,
        with_shared_expert=False,
        **dict(common, C=C16),
    )
    assert grid == 4 + 16 + 1
    # Plenty of work: capacity bounds the grid.
    grid = residency_grid(
        rows_by_rank=[512, 512],
        expert_counts=[400, 400, 400, 400],
        with_shared_expert=False,
        **common,
    )
    assert grid == 188


def _geometry():
    try:
        from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
            cake_geometry as geometry,
        )

        geometry.describe_kernel(
            geometry.KernelParams(4, 256, 6, 8192, 4096, 2048, True)
        )
    except (ImportError, FileNotFoundError) as error:  # pragma: no cover
        pytest.skip(f"Cake SM120 MegaMoE closure unavailable: {error}")
    return geometry


# (variant, EP, instance) -> (TASK_M, POOL_ROWS, MAX_TASKS, SMEM bytes) of the qualified
# DeepSeek-V4-Flash builds (E256 / H4096 / I2048 / top-k 6 / 8192 rows per rank).
_QUALIFIED = {
    ("fp8shared", 4, "m128_copier"): (128, 212928, 1664, 101376),
    ("fp8shared", 16, "m128_nocopier"): (128, 796656, 6224, 101376),
    ("fp8shared", 16, "m64_copier"): (64, 795632, 12432, 99328),
}


def test_geometry_reproduces_qualified_points() -> None:
    G = _geometry()
    for (variant, ep, name), expected in _QUALIFIED.items():
        params = G.KernelParams(ep, 256, 6, 8192, 4096, 2048, variant == "fp8shared")
        inst = G.describe_kernel(params)["instances"][name]
        c = inst["constants"]
        assert (c["TASK_M"], c["POOL_ROWS"], c["MAX_TASKS"], inst["smem_bytes"]) == (
            expected
        )
    kernel = G.describe_kernel(G.KernelParams(4, 256, 6, 8192, 4096, 2048, True))
    assert kernel["windows"][0] == ("dispatch_header_out", 4352)
    assert dict(kernel["windows"])["result_out"] == 3273129984
    assert kernel["gin"]["signal_count"] == 48
    assert kernel["transport"]["m64_max_rows"] == 512


def test_geometry_guards_reject_unsupported_shapes() -> None:
    G = _geometry()
    ok = G.KernelParams(8, 256, 8, 2048, 7168, 2048, True)
    G.describe_kernel(ok)  # DeepSeek-V3 routed geometry
    for bad in (
        G.KernelParams(8, 256, 6, 2048, 4000, 2048, False),  # hidden % 512
        G.KernelParams(8, 256, 6, 2048, 8192, 2048, False),  # staging > 99 KB
        G.KernelParams(8, 256, 6, 2048, 4096, 2000, False),  # intermediate % 128
        G.KernelParams(8, 256, 10, 2048, 4096, 2048, False),  # top-k > 9
        G.KernelParams(8, 256, 6, 2000, 4096, 2048, False),  # rows % 128
    ):
        with pytest.raises(G.GeometryError):
            G.describe_kernel(bad)
    assert G.kernel_max_rows(1) == 128 and G.kernel_max_rows(8192) == 8192


def test_geometry_swiglu_limit_is_a_device_parameter_only() -> None:
    """The clamp changes no structural switch, buffer or ABI; negative limits are rejected."""

    G = _geometry()
    base = G.KernelParams(4, 128, 8, 4096, 2048, 768, True)
    assert base.swiglu_limit == 10 and base.env()["MOE_SWIGLU_LIMIT"] == 10
    reference = G.describe_kernel(base)
    for limit in (0, 7):
        params = G.KernelParams(4, 128, 8, 4096, 2048, 768, True, limit)
        assert params.env()["MOE_SWIGLU_LIMIT"] == limit
        kernel = G.describe_kernel(params)
        for name, inst in kernel["instances"].items():
            ref = reference["instances"][name]
            assert inst["smem_bytes"] == ref["smem_bytes"]
            assert {
                k: v for k, v in inst["constants"].items() if k != "SWIGLU_LIMIT"
            } == {k: v for k, v in ref["constants"].items() if k != "SWIGLU_LIMIT"}
        assert kernel["windows"] == reference["windows"]
    with pytest.raises(G.GeometryError):
        G.describe_kernel(G.KernelParams(4, 128, 8, 4096, 2048, 768, True, -1))


def test_geometry_codec_overflow_capacity_is_a_window_parameter() -> None:
    """MOE_CODEC_OVF_BLOCKS sizes the return chunks' raw-overflow area: it grows the result
    windows linearly and changes no SMEM, ABI or schedule constant; the default is 16 at
    every hidden size (DeepSeek-V3 included)."""

    G = _geometry()
    for ep, hidden, top_k in ((4, 4096, 6), (8, 7168, 8), (16, 4096, 6)):
        base = G.KernelParams(ep, 256, top_k, 4096, hidden, 2048, True)
        ref = G.describe_kernel(base)
        for inst in ref["instances"].values():
            assert inst["constants"]["CODEC_OVF_BLOCKS_PER_ROW"] == 16
        for blocks in (17, 32, 256):
            params = G.KernelParams(
                ep, 256, top_k, 4096, hidden, 2048, True, codec_ovf_blocks=blocks
            )
            kernel = G.describe_kernel(params)
            grown = dict(kernel["windows"])["result_out"]
            expected = (blocks - 16) * 128 * 4096 * top_k * ep * 2
            assert grown - dict(ref["windows"])["result_out"] == expected
            for name, inst in kernel["instances"].items():
                assert inst["smem_bytes"] == ref["instances"][name]["smem_bytes"]
                assert inst["task_m"] == ref["instances"][name]["task_m"]
            assert kernel["abi"] == ref["abi"]
    for blocks in (15, 257):
        with pytest.raises(G.GeometryError):
            G.describe_kernel(
                G.KernelParams(
                    4, 256, 6, 4096, 4096, 2048, True, codec_ovf_blocks=blocks
                )
            )


def test_geometry_pool_rows_keep_sfa_tensor_maps_aligned() -> None:
    """POOL_ROWS is the row stride of the u32 W1/W2 SFA tensor maps (16-byte multiple for
    cuTensorMapEncodeTiled) for any local expert count, including counts not divisible by 4
    (one local expert, EPLB 288 physical experts at EP 16 = 18 per rank)."""

    G = _geometry()
    for ep, experts, variant in (
        (4, 4, True),
        (8, 8, False),
        (8, 24, True),
        (16, 288, True),
        (16, 288, False),
        (4, 20, False),
    ):
        kernel = G.describe_kernel(
            G.KernelParams(ep, experts, 6, 4096, 4096, 2048, variant)
        )
        assert kernel["local_experts"] % 4 != 0
        for inst in kernel["instances"].values():
            c = inst["constants"]
            assert (c["POOL_ROWS"] * 4) % 16 == 0
            assert c["MAX_PADDED_ROWS"] % 4 == 0
            routes = ep * 4096 * 6
            exact = routes + kernel["local_experts"] * (c["TASK_M"] - 1)
            assert exact <= c["MAX_PADDED_ROWS"] < exact + 4
    # Local counts divisible by 4 keep the exact bound (qualified builds unchanged).
    c = G.describe_kernel(G.KernelParams(4, 256, 6, 8192, 4096, 2048, True))[
        "instances"
    ]["m128_copier"]["constants"]
    assert c["MAX_PADDED_ROWS"] == 4 * 8192 * 6 + 64 * 127


def test_lsa_copier_pool_follows_the_transport_table() -> None:
    """The copier instances reserve one copier CTA per LSA destination of a rank
    (MOE_LSA_COPIERS = largest LSA count of any table row); the copier-free instances do
    not depend on it."""

    G = _geometry()
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        cake_transport as T,
    )

    def pairs(world: int, group: int) -> T.TransportTable:
        return T.table_from_pairs(
            world,
            lambda a, b: T.TRANSPORT_LSA
            if a // group == b // group
            else T.TRANSPORT_GIN,
            f"lsa_groups_of_{group}",
        )

    assert T.lsa_copier_count(T.uniform_table(8, T.TRANSPORT_GIN)) == 1
    assert T.lsa_copier_count(T.uniform_table(8, T.TRANSPORT_LSA)) == 7
    assert T.lsa_copier_count(pairs(4, 2)) == 1  # in-pair LSA (one PCIe switch)
    assert T.lsa_copier_count(pairs(8, 4)) == 3  # socket-local LSA
    assert T.lsa_copier_count(pairs(16, 8)) == 7  # two hosts, LSA inside each
    ref = G.describe_kernel(G.KernelParams(8, 256, 6, 8192, 4096, 2048, True))
    for slots in (1, 3, 7):
        kernel = G.describe_kernel(
            G.KernelParams(8, 256, 6, 8192, 4096, 2048, True, lsa_copiers=slots)
        )
        for name, inst in kernel["instances"].items():
            c = inst["constants"]
            if inst["copiers_active"]:
                assert c["PR_COPIER_CTAS_EFFECTIVE"] == slots
            else:
                assert c["PR_COPIER_CTAS_EFFECTIVE"] == 0
                same = ("COPIER_LSA_SLOTS", "PR_COPIER_CTAS")
                assert {k: v for k, v in c.items() if k not in same} == {
                    k: v
                    for k, v in ref["instances"][name]["constants"].items()
                    if k not in same
                }
        assert kernel["windows"] == ref["windows"]
    for slots in (0, 8, 17):
        with pytest.raises(G.GeometryError):
            G.describe_kernel(
                G.KernelParams(8, 256, 6, 8192, 4096, 2048, True, lsa_copiers=slots)
            )


def test_geometry_eplb_physical_expert_counts() -> None:
    G = _geometry()
    for ep, physical in ((4, 288), (8, 288), (16, 288), (8, 320)):
        kernel = G.describe_kernel(
            G.KernelParams(ep, physical, 6, 4096, 4096, 2048, False)
        )
        assert kernel["local_experts"] == physical // ep
        assert kernel["expert_slots"] >= physical // ep
        assert kernel["expert_slots"] % 32 == 0


def test_geometry_expressions_are_whitelisted() -> None:
    G = _geometry()
    assert (
        G.compile_expression("_cdiv(MOE_HIDDEN, 128) * 2")({"MOE_HIDDEN": 4096}) == 64
    )
    assert G.compile_expression("_cdiv(0 - 7, 2)")({}) == -3  # C truncation
    for text in (
        "__import__('os')",
        "MOE_HIDDEN.bit_length()",
        "MOE_HIDDEN ** 2",
        "[1]",
    ):
        with pytest.raises(ValueError):
            G.compile_expression(text)


def test_kernel_abi_is_fully_bound_and_scratch_matches_geometry() -> None:
    G = _geometry()
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        _KERNEL_WINDOWS,
        _LAYER_DESCRIPTORS,
        _PER_CALL_NAMES,
        _STATIC_DESCRIPTORS,
        _check_scratch_extent,
        scratch_specs,
    )

    transport = {
        "lsa_peer_ptrs",
        "lsa_send_state",
        "lsa_signals",
        "peer_transport",
        "gin_dev_comm",
    }
    for w in _KERNEL_WINDOWS:
        transport |= {w, f"{w}_gwin", f"{w}_gwin_window"}
    shapes = (
        (4096, 2048, 256, 6),
        (7168, 2048, 256, 8),
        (4096, 1536, 128, 8),
        (2048, 768, 128, 8),
        (4096, 2048, 288, 6),
    )
    for (hidden, inter, experts, top_k), ep, shared in (
        (s, ep, shared) for s in shapes for ep in (4, 8, 16) for shared in (False, True)
    ):
        kernel = G.describe_kernel(
            G.KernelParams(ep, experts, top_k, 1024, hidden, inter, shared)
        )
        names = set(kernel["abi"]["binding_names"])
        assert len(kernel["abi"]["args"]) == 130
        assert [w for w, _ in kernel["windows"]] == [*_KERNEL_WINDOWS, "lsa_signals"]
        for instance in kernel["instances"].values():
            specs = scratch_specs(
                instance["constants"], task_gate_u64=instance["flags"]["_TASK_GATE_U64"]
            )
            for name, dtype, shape, _ in specs:
                _check_scratch_extent(
                    instance, name, torch.empty((), dtype=dtype).element_size(), shape
                )
            bound = (
                {s[0] for s in specs}
                | transport
                | set(_STATIC_DESCRIPTORS)
                | set(_LAYER_DESCRIPTORS)
                | set(_PER_CALL_NAMES)
                | {"rank", "world_size"}
            )
            assert names <= bound, sorted(names - bound)
            assert set(instance["descriptors"]) == set(_STATIC_DESCRIPTORS) | set(
                _LAYER_DESCRIPTORS
            )


def test_epoch_is_device_resident() -> None:
    """The kernel takes its epoch from a zero-initialised u32[1] scratch word, never a host scalar
    (CUDA-graph capture: a captured launch must not freeze the epoch)."""

    G = _geometry()
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        _FILL_ZERO,
        _PER_CALL_NAMES,
        scratch_specs,
    )

    assert "epoch" not in _PER_CALL_NAMES
    for ep in (4, 8, 16):
        kernel = G.describe_kernel(G.KernelParams(ep, 256, 6, 1024, 4096, 2048, True))
        args = {a["name"]: a for a in kernel["abi"]["args"]}
        assert "epoch" not in args
        assert (args["epoch_state"]["kind"], args["epoch_state"]["abi"]) == (
            "buffer",
            "ptr",
        )
        for instance in kernel["instances"].values():
            assert instance["buffers"]["epoch_state"] == ("torch.uint32", (1,))
            specs = scratch_specs(
                instance["constants"], task_gate_u64=instance["flags"]["_TASK_GATE_U64"]
            )
            assert ("epoch_state", torch.uint32, (1,), _FILL_ZERO) in specs


def test_manifest_seals_the_parametric_closure() -> None:
    from flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe import (
        DEVICE_PARAMETERS,
        INSTANCE_NAMES,
        device_defines,
        read_manifest,
    )

    device = read_manifest()["device"]
    assert [w["name"] for w in device["instances"]] == list(INSTANCE_NAMES)
    assert set(device["parameters"]) == {*DEVICE_PARAMETERS, "MOE_INSTANCE"}
    defines = device_defines((4, 256, 6, 8192, 4096, 2048, 1, 0, 32, 3))
    assert defines[0] == "-DMOE_RANKS=4" and "-DMOE_SWIGLU_LIMIT=0" in defines
    assert defines[-2:] == ["-DMOE_CODEC_OVF_BLOCKS=32", "-DMOE_LSA_COPIERS=3"]
    with pytest.raises(ValueError):
        device_defines((4, 256, 6, 8192, 4096, 2048, 1, 0))


def test_aot_device_parameters_match_the_default_deployment() -> None:
    """The AOT-packaged device modules are the ones a DeepSeek-V4-Flash deployment with the
    in-kernel shared expert, 4096 tokens per rank and the default all-GIN table launches
    (same parameter tuple, hence the same JIT module name)."""

    from flashinfer.moe_ep import FleetParams
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        Sm120Mxfp4Mxfp8CakeMegaKernelBackend,
        candidate_instances,
    )
    from flashinfer.moe_ep.kernel_src.sm120.cake_mxfp4_megamoe import (
        AOT_DEVICE_PARAMETERS,
    )

    config = Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(shared_expert="fp8_in_kernel")
    backend = Sm120Mxfp4Mxfp8CakeMegaKernelBackend(config)
    assert candidate_instances(config.transport_table) == (2, 3)  # copier-free flavor
    expected = tuple(
        backend.kernel_params(
            world,
            FleetParams(
                num_experts=256, token_hidden_size=4096, max_tokens_per_rank=4096
            ),
        ).device_parameters()
        for world in (4, 8)
    )
    assert expected == AOT_DEVICE_PARAMETERS


def test_config_transport_table_validation() -> None:
    assert (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
            transport_table="all_gin"
        ).transport_table
        == "all_gin"
    )
    assert Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
        transport_table=[0, 2, 2, 0]
    ).transport_table == (0, 2, 2, 0)
    assert (
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
            transport_table="auto"
        ).transport_table
        == "auto"
    )
    with pytest.raises(ValueError):
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(transport_table=None)
    with pytest.raises(ValueError):
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(transport_table="nvlink")
    with pytest.raises(ValueError):
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(transport_table=(0, 3, 3, 0))


# (name, hidden, intermediate, experts, top_k) of the shape-generic probe / region tests.
_PROBE_SHAPES = (
    ("v4flash", 4096, 2048, 256, 6),
    ("dsv3", 7168, 2048, 256, 8),
    ("qwen3_235b", 4096, 1536, 128, 8),
    ("qwen3_30b", 2048, 768, 128, 8),
    ("h512", 512, 2048, 256, 6),
    ("h7680", 7680, 2048, 256, 6),
    ("k1", 4096, 2048, 256, 1),
    ("k9", 4096, 2048, 256, 9),
)


def test_probe_traffic_is_sized_from_the_geometry() -> None:
    G = _geometry()
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        cake_transport as T,
    )

    # DeepSeek-V4-Flash at 8192 rows reproduces the original fixed probe sizing.
    for ep, expected in ((4, 33554432), (8, 33554432), (16, 32636928)):
        kernel = G.describe_kernel(G.KernelParams(ep, 256, 6, 8192, 4096, 2048, True))
        assert kernel["transport"]["record_bytes"] == 4352
        assert T.probe_traffic(kernel) == {
            "bytes_per_dest": expected,
            "gin_chunk_bytes": 704 * 1024,
        }
    seen_records = set()
    for name, hidden, inter, experts, top_k in _PROBE_SHAPES:
        for ep, rows in ((4, 1024), (8, 512), (16, 128)):
            kernel = G.describe_kernel(
                G.KernelParams(ep, experts, top_k, rows, hidden, inter, False)
            )
            c = kernel["instances"]["m128_copier"]["constants"]
            t = kernel["transport"]
            record = t["record_bytes"]
            seen_records.add(record)
            # The probe's record and result-row sizes are the build's own.
            assert record == c["RECORD_BYTES"], name
            assert record * 2 * ep * rows == c["PAYLOAD_WINDOW_BYTES"], name
            assert t["codec_row_bytes"] == c["CODEC_ROW_BYTES"], name
            traffic = T.probe_traffic(kernel)
            full = rows * top_k // ep * (record + c["CODEC_ROW_BYTES"])
            assert traffic["bytes_per_dest"] == min(full, 32 << 20) // 16 * 16, name
            chunk = traffic["gin_chunk_bytes"]
            assert chunk % (64 << 10) == 0, name
            assert 0 <= chunk - c["DISPATCH_CHUNK_MIN_RECORDS"] * record < 64 << 10, (
                name
            )
    assert len(seen_records) >= 5  # the record size follows the hidden size


def test_geometry_rejects_outside_the_supported_region() -> None:
    G = _geometry()
    for params in (
        G.KernelParams(32, 1056, 6, 1024, 4096, 2048, False),  # 1056 experts > 1024
        G.KernelParams(2, 258, 6, 1024, 4096, 2048, False),  # 129 experts per rank
        G.KernelParams(33, 264, 6, 1024, 4096, 2048, False),  # EP 33
        G.KernelParams(1, 256, 6, 1024, 4096, 2048, False),  # EP 1
        G.KernelParams(2, 256, 1, 65536, 512, 128, False),  # 16-bit active_rows field
    ):
        with pytest.raises(G.GeometryError):
            G.describe_kernel(params)
    G.describe_kernel(G.KernelParams(2, 256, 1, 65408, 512, 128, False))
    G.describe_kernel(
        G.KernelParams(8, 1024, 8, 512, 4096, 2048, False)
    )  # 128 per rank
    G.describe_kernel(G.KernelParams(32, 256, 8, 1024, 7168, 2048, True))


def test_device_resource_check_rejects_spills_above_the_verified_ranks() -> None:
    G = _geometry()
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        LOCAL_BYTES_LIMIT,
        REGISTER_BUDGET_VERIFIED_MAX_RANKS,
        candidate_instances,
        check_device_resources,
    )

    assert candidate_instances("all_gin") == (2, 3)
    assert candidate_instances(None) == candidate_instances("all_lsa") == (0, 1, 2, 3)
    calls: list[tuple[tuple[int, ...], int]] = []

    def clean(parameters, instance):
        calls.append((parameters, instance))
        return {"reg": 168, "stack": 8, "local": 0}

    inside = G.KernelParams(
        REGISTER_BUDGET_VERIFIED_MAX_RANKS, 256, 8, 1024, 7168, 2048, False
    )
    assert check_device_resources(inside, (0, 1, 2, 3), resource_usage=clean) == {}
    assert calls == []  # verified region: nothing is compiled
    wide = G.KernelParams(32, 256, 8, 1024, 7168, 2048, False)
    report = check_device_resources(wide, (2, 3), resource_usage=clean)
    assert sorted(report) == [2, 3]
    assert [i for _, i in calls] == [2, 3]
    assert all(p == wide.device_parameters() for p, _ in calls)

    def spills(parameters, instance):
        del parameters
        stack = LOCAL_BYTES_LIMIT + 8 if instance == 2 else 8
        return {"reg": 168, "stack": stack, "local": 0}

    with pytest.raises(G.GeometryError, match="m128_nocopier"):
        check_device_resources(wide, (0, 1, 2, 3), resource_usage=spills)
    # The copier-only flavors of the same build are fine.
    check_device_resources(wide, (0, 1), resource_usage=spills)


def test_static_residency_grid_is_a_routing_independent_lower_bound() -> None:
    import random

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        residency_grid,
        static_residency_grid,
    )

    rng = random.Random(7)
    for world, local, blocks, fold in (
        (2, 2, 32, False),
        (4, 64, 12, False),
        (8, 32, 16, True),
    ):
        C = {
            "W1_N_BLOCKS": blocks,
            "W2_N_BLOCKS": 32,
            "PR_COPIER_CTAS_EFFECTIVE": 4,
            "C56_MB_ROWS": 256,
            "FRONT_EXCLUDED_CTAS": 4,
        }
        for shared in (False, True):
            common = dict(
                local_experts=local,
                task_m=64,
                with_shared_expert=shared,
                capacity=188,
                C=C,
                resident_peer_fold=fold,
            )
            static = static_residency_grid(world=world, **common)
            for _ in range(50):
                rows = [rng.randrange(0, 1024) for _ in range(world)]
                counts = [rng.randrange(0, 300) for _ in range(world * local)]
                dynamic = residency_grid(
                    rows_by_rank=rows, expert_counts=counts, **common
                )
                assert static <= dynamic


def test_rows_residency_grid_needs_no_routing_and_stays_in_the_valid_range() -> None:
    """The rows-only grid (caller-supplied per-rank rows) lies between the static grid and
    the capacity for any rows, and equals the routing grid of the most even routing."""

    import random

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_backend import (
        even_expert_counts,
        residency_grid,
        rows_residency_grid,
        static_residency_grid,
    )

    assert even_expert_counts(rows_by_rank=[1, 0], top_k=3, num_experts=4) == [
        1,
        1,
        1,
        0,
    ]
    assert (
        even_expert_counts(rows_by_rank=[100, 28], top_k=6, num_experts=256)
        == [3] * 256
    )
    rng = random.Random(11)
    for world, local, blocks, fold in (
        (2, 2, 32, False),
        (4, 64, 12, False),
        (8, 36, 16, True),
    ):
        C = {
            "W1_N_BLOCKS": blocks,
            "W2_N_BLOCKS": 32,
            "PR_COPIER_CTAS_EFFECTIVE": 4,
            "C56_MB_ROWS": 256,
            "FRONT_EXCLUDED_CTAS": 4,
        }
        for shared in (False, True):
            common = dict(
                task_m=64,
                with_shared_expert=shared,
                capacity=188,
                C=C,
                resident_peer_fold=fold,
            )
            static = static_residency_grid(world=world, local_experts=local, **common)
            for _ in range(50):
                rows = [rng.choice((0, 1, 7, 64, 511, 4096)) for _ in range(world)]
                grid = rows_residency_grid(
                    rows_by_rank=rows, top_k=6, num_experts=world * local, **common
                )
                assert static <= grid <= 188
                counts = even_expert_counts(
                    rows_by_rank=rows, top_k=6, num_experts=world * local
                )
                assert sum(counts) == 6 * sum(rows)
                assert grid == residency_grid(
                    rows_by_rank=rows,
                    expert_counts=counts,
                    local_experts=local,
                    **common,
                )


def test_knob_cache_round_trip_and_bucket_choice(tmp_path, monkeypatch) -> None:
    """Tuned transport tables: exact key match (topology included), nearest capacity bucket,
    entries of other backends preserved."""

    import json

    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        cake_knob_cache as K,
    )

    path = tmp_path / "knobs.json"
    other = {"dtype": "fp8_e4m3", "world_size": 4, "max_tokens": 512, "knobs": {"x": 1}}
    path.write_text(json.dumps({"version": 1, "entries": [other]}))
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
    reach = [[i != j for j in range(4)] for i in range(4)]
    key = dict(
        device="RTX",
        variant="fp8shared",
        world_size=4,
        hidden=4096,
        intermediate=2048,
        num_experts=256,
        topk=6,
        topology=K.topology_digest(reach, True),
    )
    pairs = [0, 1, 2, 2, 1, 0, 2, 2, 2, 2, 0, 1, 2, 2, 1, 0]
    assert K.lookup_knobs(max_tokens=512, **key) is None
    assert K.record_knobs(
        {"transport_table": [0] * 16, "table_name": "a"}, max_tokens=512, **key
    )
    assert K.record_knobs(
        {"transport_table": pairs, "table_name": "lsa_pairs"}, max_tokens=4096, **key
    )
    assert K.lookup_knobs(max_tokens=512, **key)["table_name"] == "a"
    assert K.lookup_knobs(max_tokens=1024, **key)["table_name"] == "lsa_pairs"
    assert K.lookup_knobs(max_tokens=8192, **key)["table_name"] == "lsa_pairs"
    other_topology = dict(key, topology=K.topology_digest(reach, False))
    assert K.lookup_knobs(max_tokens=512, **other_topology) is None
    entries = json.loads(path.read_text())["entries"]
    assert other in entries and len(entries) == 3
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    assert K.lookup_knobs(max_tokens=512, **key) is None


def test_tuner_candidate_tables_are_feasible_and_distinct() -> None:
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_tuner import (
        candidate_tables,
    )

    # Two LSA domains of 4 ranks (e.g. two sockets / hosts): no all-LSA candidate.
    reach = [[i != j and i // 4 == j // 4 for j in range(8)] for i in range(8)]
    names = [name for name, _ in candidate_tables(8, reach, True)]
    assert names == ["all_gin", "lsa_where_reachable", "lsa_pairs"]
    # One LSA domain: lsa_where_reachable == all_lsa, kept once.
    full = [[i != j for j in range(4)] for i in range(4)]
    assert [name for name, _ in candidate_tables(4, full, True)] == [
        "all_gin",
        "lsa_where_reachable",
        "lsa_pairs",
    ]
    # No GIN: only the all-LSA table is feasible.
    assert [name for name, _ in candidate_tables(4, full, False)] == [
        "lsa_where_reachable"
    ]
