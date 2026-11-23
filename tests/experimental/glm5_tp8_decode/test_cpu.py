# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only numerical-contract and packed-storage checks."""

import pytest
import torch

from flashinfer.experimental.glm5_tp8_decode import _moe_packing
from flashinfer.experimental.glm5_tp8_decode import _attention_packing
from flashinfer.experimental.glm5_tp8_decode import geometry

from . import attention_reference as attention_ref
from . import moe_reference as moe_ref
from ._selector_reference import check_selection


def test_full_key_selector_oracle_rejects_zero_sign_and_lower_id_regressions():
    logits = torch.zeros((4, 2053), dtype=torch.float32)
    logits[:, :2] = -0.0
    ids = torch.arange(2, 2050, dtype=torch.int32).expand(4, -1).clone()
    check_selection(logits, ids, 2049)
    wrong = ids.clone()
    wrong[0, -1] = 0
    with pytest.raises(AssertionError):
        check_selection(logits, wrong, 2049)
    logits.fill_(1)
    lowest = torch.arange(2048, dtype=torch.int32).expand(4, -1).clone()
    check_selection(logits, lowest, 2049)
    with pytest.raises(AssertionError):
        check_selection(logits, lowest + 1, 2049)


def test_full_key_selector_oracle_distinguishes_fp32_values_in_one_fp16_bin():
    row = 1 + torch.arange(2053, dtype=torch.float32) * 2**-23
    assert torch.unique(row.half()).numel() == 1
    logits = row.expand(4, -1).clone()
    ids = torch.stack(
        [torch.arange(q + 2, q + 2050, dtype=torch.int32) for q in range(4)]
    )
    check_selection(logits, ids, 2049)
    ids[0, 0] = 0
    with pytest.raises(AssertionError):
        check_selection(logits, ids, 2049)


@pytest.mark.parametrize(
    "capacity", [1, 202752, 202753, 202760, 524288, 1048576, 1052672]
)
def test_long_context_capacity_accepts_the_extended_logical_range(capacity):
    assert geometry.check_capacity(capacity) == capacity


@pytest.mark.parametrize("capacity", [0, -1, 1052673, True, False, 1.0, None])
def test_long_context_capacity_rejects_invalid_values(capacity):
    with pytest.raises(ValueError):
        geometry.check_capacity(capacity)


def test_long_context_extension_preserves_the_default_allocation_capacity():
    import inspect

    from flashinfer.glm5_tp8_decode import create_glm5_tp8_decode_workspace
    from flashinfer.experimental.glm5_tp8_decode.workspace import DecodeWorkspace

    assert geometry.DEFAULT_MAX_SEQ_LEN == 202752
    assert geometry.MAX_SEQ_LEN == 1052672
    for factory in (create_glm5_tp8_decode_workspace, DecodeWorkspace):
        assert inspect.signature(factory).parameters["max_seq_len"].default == 202752


def test_long_context_reference_uses_unsigned_key32_order():
    # Do not pack key32<<32 into signed torch.int64: positive-score composites
    # would acquire bit63. The independent reference ranks key32 and ID separately.
    scores = torch.tensor([[-1.0, -0.0, 0.0, 1.0]], dtype=torch.float32)
    expected_keys = torch.tensor(
        [[0x407FFFFF, 0x7FFFFFFF, 0x80000000, 0xBF800000]], dtype=torch.int64
    )
    assert torch.equal(attention_ref.topk_key(scores), expected_keys)
    selected, lengths = attention_ref.ref_topk_select(scores, 3, topk=2)
    assert torch.equal(selected, torch.tensor([[2, 3]], dtype=torch.int32))
    assert lengths.tolist() == [2]


def test_bitwise_singleton_strided_slice_preserves_exact_bits():
    storage = torch.zeros((1, 6144), dtype=torch.bfloat16)
    value = storage[:, 0]
    assert value.stride() == (6144,) and value.is_contiguous()
    expected = torch.tensor([1 / 256], dtype=torch.bfloat16)
    value.copy_(expected)
    assert moe_ref.bitwise(value, expected)
    changed = torch.tensor([1 / 256 + 3 * 2**-15], dtype=torch.bfloat16)
    assert not moe_ref.bitwise(value, changed)
    value.fill_(-0.0)
    expected.zero_()
    assert torch.equal(value, expected)
    assert not moe_ref.bitwise(value, expected), "signed zero must compare by bits"


def test_tp_exit_has_one_final_bf16_rounding():
    residual = torch.tensor([[2.0]], dtype=torch.bfloat16)
    partials = [
        torch.tensor([[-2.0]], dtype=torch.bfloat16),
        torch.tensor([[1 / 256]], dtype=torch.bfloat16),
    ]
    partials += [torch.zeros_like(residual) for _ in range(6)]
    expected = torch.tensor([[1 / 256]], dtype=torch.bfloat16)
    assert torch.equal(moe_ref.reduce_ranks(partials, residual), expected)
    assert torch.equal(attention_ref.ref_tp_exit_sum(partials, residual), expected)
    # Negative control: rounding the partial sum before adding the residual is wrong.
    rounded_sum = sum(p.float() for p in partials).bfloat16()
    assert not torch.equal(
        (rounded_sum.float() + residual.float()).bfloat16(), expected
    )


def test_q128_zero_floor_and_batch_independence():
    x = torch.zeros(1, 256, dtype=torch.bfloat16)
    codes, scale = moe_ref.q128(x)
    assert torch.equal(codes.float(), torch.zeros_like(x, dtype=torch.float32))
    assert bool((scale > 0).all())
    expected_scale = torch.tensor(1e-4, dtype=torch.float32) * torch.tensor(
        1 / 448, dtype=torch.float32
    )
    assert torch.equal(scale, expected_scale.expand_as(scale))
    x[0, 0] = 3.5
    one, one_scale = moe_ref.q128(x)
    four, four_scale = moe_ref.q128(x.repeat(4, 1))
    assert torch.equal(four.view(torch.uint8), one.view(torch.uint8).repeat(4, 1))
    assert torch.equal(four_scale, one_scale.repeat(4, 1))


def test_exact_topk_accepts_unordered_set_but_rejects_wrong_membership():
    logits = torch.arange(2050, dtype=torch.float32).reshape(1, -1)
    ids = torch.arange(2, 2050, dtype=torch.int32).flip(0).reshape(1, -1)
    assert torch.equal(attention_ref.check_exact_topk(logits, ids, 2049, 1), ids)
    below = ids.clone()
    below[0, -1] = 0
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, below, 2049, 1)
    duplicate = ids.clone()
    duplicate[0, -1] = duplicate[0, 0]
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, duplicate, 2049, 1)


def test_dense_selection_is_a_causal_prefix_not_stale_storage():
    old_ids = torch.full((1, 2048), 123456, dtype=torch.int32)
    selected = attention_ref.check_exact_topk(torch.empty(1, 32), old_ids, 31, 1)
    assert torch.equal(selected[0, :32], torch.arange(32, dtype=torch.int32))
    assert bool((selected[0, 32:] == -1).all())
    assert bool((old_ids == 123456).all())


def test_numerical_gate_rejects_three_bf16_ulps():
    expected = torch.ones(1, 4, dtype=torch.bfloat16)
    wrong = torch.full_like(expected, 1 + 3 / 128)
    with pytest.raises(AssertionError):
        moe_ref.floored_ulp(wrong, expected)
    with pytest.raises(AssertionError):
        attention_ref.assert_close(wrong, expected, 2, 1 / 16)


@pytest.mark.parametrize("argument,center", [(1.0, 1.0), (4.0, 0.5)])
def test_rsqrt_candidates_cover_both_spacings_at_exponent_boundary(argument, center):
    candidates = moe_ref._rsqrt_candidates(argument)
    ulp = center * 2.0**-23
    assert candidates == [
        center - 2 * ulp,
        center - 1.5 * ulp,
        center - ulp,
        center - 0.5 * ulp,
        center,
        center + ulp,
        center + 2 * ulp,
    ]


def test_router_gate_rejects_bf16_rounded_logits_and_wrong_normalization():
    x = torch.ones(1, 6144, dtype=torch.bfloat16)
    gamma = torch.ones(6144, dtype=torch.float32)
    router = torch.zeros(256, 6144, dtype=torch.bfloat16)
    router[:, 0] = 1
    router[:, 1] = 1 / 512
    # Every permitted rstd rounds this row to BF16 ones. Its dot is exactly
    # 1+1/512 in FP32, but rounds to 1 in BF16: the gamma64 gate must catch it.
    logits = torch.full((1, 256), 1 + 1 / 512, dtype=torch.float32)
    moe_ref.check_router_logits(logits, x, gamma, router)
    with pytest.raises(AssertionError):
        moe_ref.check_router_logits(logits.bfloat16().float(), x, gamma, router)
    wrong_norm = torch.full_like(x, 1 + 3 / 128)
    with pytest.raises(AssertionError):
        moe_ref.floored_ulp(wrong_norm, moe_ref.rmsnorm(x, gamma))
    with pytest.raises(AssertionError):
        moe_ref.check_router_logits(
            wrong_norm.double() @ router.double().T, x, gamma, router
        )


def test_router_gate_requires_one_candidate_for_the_whole_expert_row():
    # Test candidate selection separately from the norm generator: accepting
    # each expert from a different allowed candidate would incorrectly pass all
    # ones here, even though neither complete candidate produces that row.
    norms = torch.zeros(2, 6144, dtype=torch.bfloat16)
    norms[0, 0] = 1
    norms[1, 1] = 1
    router = torch.zeros(256, 6144, dtype=torch.bfloat16)
    router[::2, 0] = 1
    router[1::2, 1] = 1
    for candidate in norms:
        logits = candidate.double()[None, :] @ router.double().T
        moe_ref.check_router_candidates(logits, [norms], router)
    with pytest.raises(AssertionError):
        moe_ref.check_router_candidates(torch.ones(1, 256), [norms], router)


def test_approximate_sigmoid_accepts_only_analytically_ambiguous_routes():
    sigma = torch.full((1, 256), 0.5, dtype=torch.float64)
    choice = torch.full_like(sigma, 0.25)
    choice[0, :7] = 0.75
    choice[0, 7] = 0.5 + moe_ref.SIGMOID_ABS_ERR / 2
    choice[0, 8] = 0.5
    swapped = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 8]], dtype=torch.int32)
    probabilities = torch.full((1, 8), 2.5 / 8)
    assert (
        moe_ref.check_routing(swapped, probabilities, sigma, choice)[
            "swapped_at_boundary"
        ]
        == 1
    )
    legal = moe_ref.candidate_selections(choice[0], moe_ref.SIGMOID_ABS_ERR + 1e-6)
    assert set(swapped[0].tolist()) in [set(candidate) for candidate in legal]
    wrong_ids = swapped.clone()
    wrong_ids[0, -1] = 9
    with pytest.raises(AssertionError):
        moe_ref.check_routing(wrong_ids, probabilities, sigma, choice)
    wrong_probs = probabilities.clone()
    wrong_probs[0, 0] += 0.01
    wrong_probs[0, 1] -= 0.01
    with pytest.raises(AssertionError):
        moe_ref.check_routing(swapped, wrong_probs, sigma, choice)


def test_numerical_gates_reject_nonfinite_values():
    expected = torch.ones(1, 4, dtype=torch.bfloat16)
    for value in (float("nan"), float("inf")):
        wrong = torch.full_like(expected, value)
        with pytest.raises(AssertionError):
            moe_ref.floored_ulp(wrong, expected)
        with pytest.raises(AssertionError):
            attention_ref.assert_close(wrong, expected, 2, 1 / 16)


def test_exact_topk_allows_equal_threshold_membership():
    logits = torch.zeros(1, 2050, dtype=torch.float32)
    logits[0, 0] = 1
    ids = torch.arange(2, 2050, dtype=torch.int32).reshape(1, -1)
    ids[0, -1] = 0
    assert torch.equal(attention_ref.check_exact_topk(logits, ids, 2049, 1), ids)
    ids[0, -1] = 1
    with pytest.raises(AssertionError):
        attention_ref.check_exact_topk(logits, ids, 2049, 1)


def test_moe_gate_up_packing_preserves_codes_and_scale_headers():
    # One bank exercises the same private layout without allocating the public
    # 257-bank fixture. Selected coordinates cross row-group, K-page and MMA tiles.
    gate = torch.zeros(1, 256, 6144, dtype=torch.uint8)
    up = torch.zeros_like(gate)
    coordinates = [
        (0, 0),
        (7, 7),
        (8, 8),
        (15, 15),
        (16, 1023),
        (127, 1024),
        (128, 2048),
        (255, 6143),
    ]
    for i, (row, col) in enumerate(coordinates):
        gate[0, row, col] = 0x30 + i
        up[0, row, col] = 0xB0 + i
    gs = torch.arange(96, dtype=torch.float32).reshape(1, 2, 48) / 128 + 0.25
    us = gs + 1
    packed = _moe_packing.pack_gate_up_pages(
        gate.view(torch.float8_e4m3fn), gs, up.view(torch.float8_e4m3fn), us
    )
    assert packed.dtype == torch.uint8
    assert tuple(packed.shape) == (1, 16, 6, 32896)
    for i, (row, col) in enumerate(coordinates):
        group, within_row = divmod(row, 16)
        page, within_page = divmod(col, 1024)
        tile, within_col = divmod(within_page, 16)
        lane = (within_row % 8) * 4 + (within_col % 8) // 2
        fragment = (within_col // 8) * 2 + within_row // 8
        offset = tile * 256 + lane * 8 + fragment * 2 + within_col % 2
        assert packed[0, group, page, offset].item() == 0x30 + i
        assert packed[0, group, page, 16384 + offset].item() == 0xB0 + i
    assert int(torch.count_nonzero(packed[..., :32768])) == 2 * len(coordinates)
    for group in (0, 7, 8, 15):
        for page in (0, 2, 5):
            header = (
                packed[0, group, page, 32768:32832].contiguous().view(torch.float32)
            )
            assert torch.equal(header[:8], gs[0, group // 8, page * 8 : (page + 1) * 8])
            assert torch.equal(header[8:], us[0, group // 8, page * 8 : (page + 1) * 8])
    assert bool((packed[..., 32832:] == 0).all())


def test_moe_down_slices_place_every_code_and_replicate_scales():
    # Two banks of random codes: every byte must land at the address derived
    # from the documented slice/tile/chunk/lane layout, so any permutation
    # error inside a bank or across banks fails.
    banks = 2
    generator = torch.Generator().manual_seed(0)
    data = torch.randint(
        0, 256, (banks, 6144, 256), generator=generator, dtype=torch.uint8
    )
    scales = torch.arange(banks * 96, dtype=torch.float32).reshape(banks, 48, 2) + 1
    packed, packed_scales = _moe_packing.pack_down_slices(
        data.view(torch.float8_e4m3fn), scales
    )
    assert packed.dtype == torch.uint8 and packed.shape == data.shape

    bank = torch.arange(banks).view(-1, 1, 1)
    row = torch.arange(6144).view(1, -1, 1)
    col = torch.arange(256).view(1, 1, -1)
    slice_, slice_row = row // 48, row % 48
    tile, r = slice_row // 16, slice_row % 16
    chunk, half, k = col // 32, col % 32 // 16, col % 16
    lane = (r % 8) * 4 + (k % 8) // 2
    register = (k // 8) * 2 + r // 8
    address = (
        bank * (6144 * 256)
        + slice_ * 0x3000
        + tile * 0x1000
        + chunk * 0x200
        + lane * 16
        + half * 8
        + register * 2
        + k % 2
    )
    address = address.reshape(-1)
    assert torch.equal(address.sort().values, torch.arange(address.numel()))
    assert torch.equal(packed.reshape(-1)[address], data.reshape(-1))

    assert packed_scales.dtype == torch.float32
    scale_rows = packed_scales.reshape(banks, 128, 8, 2)
    for b in range(banks):
        for s in range(128):
            for z in range(6):
                block_row = (48 * s + 8 * z) // 128
                assert torch.equal(scale_rows[b, s, z], scales[b, block_row])
    assert bool((scale_rows[:, :, 6:] == 0).all())


@pytest.mark.parametrize("value", [0, 2, 3, 8, True, 1.0])
def test_supported_query_lengths_reject_other_shapes(value):
    with pytest.raises(ValueError):
        geometry.check_q_len(value)


def test_workspace_regions_are_disjoint_for_all_operator_sets():
    sizes = {
        "attention": 16 << 20,
        "attention_q1": 0x38000,
        "ll": 128 << 10,
        "moe_q4": 0xE0000,
        "moe_q1": 0x38000,
    }
    for operators in (("moe",), ("attention",), ("attention", "moe")):
        total, offsets = geometry.communication_layout(operators)
        spans = sorted(
            (offset, offset + sizes[name]) for name, offset in offsets.items()
        )
        assert spans[0][0] == 0 and spans[-1][1] == total
        assert all(
            left[1] == right[0] for left, right in zip(spans, spans[1:], strict=False)
        )
        assert all(begin % 16 == 0 for begin, _ in spans)


def _mma_element_offset(row, column):
    # Scalar inverse of the documented m16n8 register convention.
    lane = (row % 8) * 4 + (column % 8) // 2
    fragment = (column // 8) * 2 + row // 8
    return lane * 8 + fragment * 2 + column % 2


def test_attention_fragment_pages_codes_headers_and_padding():
    weights = torch.zeros((256, 6144), dtype=torch.uint8)
    points = [
        (0, 0),
        (15, 15),
        (16, 16),
        (31, 127),
        (32, 128),
        (127, 1023),
        (128, 1024),
        (255, 6143),
    ]
    for index, (row, col) in enumerate(points):
        weights[row, col] = 0x31 + index
    scales = torch.arange(96, dtype=torch.float32).reshape(2, 48) + 0.25
    packed = _attention_packing.pack_fragment_pages(weights, scales).reshape(
        8, 6, 32896
    )
    for index, (row, col) in enumerate(points):
        cta, row32 = divmod(row, 32)
        page, column1024 = divmod(col, 1024)
        row_group, row16 = divmod(row32, 16)
        tile, col16 = divmod(column1024, 16)
        # Warp-interleaved order: k16 step s of warp w is stored as tile 8 * s + w.
        tile = (tile % 8) * 8 + tile // 8
        offset = row_group * 16384 + tile * 256 + _mma_element_offset(row16, col16)
        assert packed[cta, page, offset].item() == 0x31 + index
    assert int(torch.count_nonzero(packed[..., :32768])) == len(points)
    for cta in range(8):
        header = packed[cta, :, 32768:32800].contiguous().view(torch.float32)
        assert torch.equal(header, scales[cta // 4].reshape(6, 8))
    assert bool((packed[..., 32800:] == 0).all())


@pytest.mark.parametrize("kind", ["w_uk", "w_uv"])
def test_attention_per_head_projection_codes_and_scale_bytes(kind):
    rows, columns, row_group, code_bytes, page_bytes = (
        (512, 192, 64, 12288, 12416) if kind == "w_uk" else (256, 512, 32, 16384, 16512)
    )
    weights = torch.zeros((10, rows, columns), dtype=torch.uint8)
    points = [
        (0, 0, 0),
        (0, row_group - 1, columns - 1),
        (3, row_group, 16),
        (9, rows - 1, columns - 1),
    ]
    for index, (head, row, col) in enumerate(points):
        weights[head, row, col] = 0x39 + index
    scale_columns = columns // (64 if kind == "w_uk" else 128)
    scales = (
        torch.arange(10 * 4 * scale_columns, dtype=torch.float32).reshape(
            10, 4, scale_columns
        )
        / 128
        + 0.5
    )
    packed = getattr(_attention_packing, "pack_" + kind)(
        weights.view(torch.float8_e4m3fn), scales
    )
    assert packed.shape == (80, page_bytes)
    for index, (head, row, col) in enumerate(points):
        cta = head * 8 + row // row_group
        r = row % row_group
        offset = (
            (r // 16) * (16 * columns)
            + (col // 16) * 256
            + _mma_element_offset(r % 16, col % 16)
        )
        assert packed[cta, offset].item() == 0x39 + index
    assert int(torch.count_nonzero(packed[:, :code_bytes])) == len(points)
    header = (
        packed[:, code_bytes : code_bytes + 4 * scale_columns]
        .contiguous()
        .view(torch.float32)
    )
    assert torch.equal(
        header, scales.repeat_interleave(2, 1).reshape(80, scale_columns)
    )
    assert bool((packed[:, code_bytes + 4 * scale_columns :] == 0).all())


def test_attention_head_weight_packing_preserves_bf16_bits():
    bits = torch.zeros((32, 6144), dtype=torch.int16)
    points = [(0, 0), (7, 7), (8, 8), (15, 1023), (16, 1024), (31, 6143)]
    for index, (row, col) in enumerate(points):
        bits[row, col] = 0x3F80 + index
    packed = _attention_packing.pack_indexer_weights_proj(
        bits.view(torch.bfloat16)
    ).view(torch.int16)
    assert packed.shape == (2, 98304)
    for index, (row, col) in enumerate(points):
        group, row16 = divmod(row, 16)
        page, col1024 = divmod(col, 1024)
        tile, col16 = divmod(col1024, 16)
        offset = page * 16384 + tile * 256 + _mma_element_offset(row16, col16)
        assert packed[group, offset].item() == 0x3F80 + index
    assert int(torch.count_nonzero(packed)) == len(points)


_FP8 = torch.float8_e4m3fn


def _codes(shape, generator):
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator).view(
        _FP8
    )


def _attention_layer(seed=0, norm_dtype=torch.bfloat16):
    """A full-shape attention checkpoint layer of random codes and scales."""
    g = torch.Generator().manual_seed(seed)

    def scales(*shape):
        return torch.rand(shape, generator=g) + 0.5

    def norm(n):
        return torch.randn(n, generator=g).to(norm_dtype)

    return {
        "input_layernorm.weight": norm(6144),
        "self_attn.q_a_proj.weight": _codes((2048, 6144), g),
        "self_attn.q_a_proj.weight_scale_inv": scales(16, 48),
        "self_attn.q_a_layernorm.weight": norm(2048),
        "self_attn.kv_a_proj_with_mqa.weight": _codes((576, 6144), g),
        "self_attn.kv_a_proj_with_mqa.weight_scale_inv": scales(5, 48),
        "self_attn.kv_a_layernorm.weight": norm(512),
        "self_attn.q_b_proj.weight": _codes((16384, 2048), g),
        "self_attn.q_b_proj.weight_scale_inv": scales(128, 16),
        "self_attn.kv_b_proj.weight": _codes((28672, 512), g),
        "self_attn.kv_b_proj.weight_scale_inv": scales(224, 4),
        "self_attn.o_proj.weight": _codes((6144, 16384), g),
        "self_attn.o_proj.weight_scale_inv": scales(48, 128),
        "self_attn.indexer.wk.weight": _codes((128, 6144), g),
        "self_attn.indexer.wk.weight_scale_inv": scales(1, 48),
        "self_attn.indexer.wq_b.weight": _codes((4096, 2048), g),
        "self_attn.indexer.wq_b.weight_scale_inv": scales(32, 16),
        "self_attn.indexer.weights_proj.weight": torch.randn(
            32, 6144, generator=g
        ).bfloat16(),
        "self_attn.indexer.k_norm.weight": norm(128),
        "self_attn.indexer.k_norm.bias": norm(128),
    }


def _same_bits(actual, expected):
    return actual.dtype == expected.dtype and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


def test_attention_shard_selects_rank_heads_and_zero_pads_rank_seven():
    from flashinfer.experimental.glm5_tp8_decode.attention_weights import (
        shard_attention_weights,
    )

    # Label every row/column by its global head (plus one) and every scale by
    # its block, using broadcast views: inputs may be noncontiguous.
    def by_row(rows, per_head, columns):
        label = (torch.arange(rows) // per_head + 1).to(torch.uint8)
        return label.view(-1, 1).expand(rows, columns).view(_FP8)

    layer = _attention_layer()
    layer["self_attn.q_b_proj.weight"] = by_row(16384, 256, 2048)
    layer["self_attn.q_b_proj.weight_scale_inv"] = (
        (torch.arange(128) // 2 + 1).float().view(-1, 1).expand(128, 16)
    )
    layer["self_attn.kv_b_proj.weight"] = by_row(28672, 448, 512)
    layer["self_attn.kv_b_proj.weight_scale_inv"] = (
        (torch.arange(224) + 1).float().view(-1, 1).expand(224, 4)
    )
    layer["self_attn.o_proj.weight"] = by_row(16384, 256, 6144).t()
    layer["self_attn.o_proj.weight_scale_inv"] = (
        (torch.arange(128) // 2 + 1).float().view(1, -1).expand(48, 128)
    )
    assert all(
        shard_attention_weights(layer, rank=0)[name] is layer[name]
        for name in shard_attention_weights(layer, rank=0)
    )
    for rank in (1, 6, 7):
        h0 = 10 * (rank - 1)
        hv = min(10, 64 - h0)
        shard = shard_attention_weights(layer, rank=rank)
        assert len(shard) == 13
        heads = torch.cat([torch.arange(h0, h0 + hv) + 1.0, torch.zeros(10 - hv)])
        q_b = shard["self_attn.q_b_proj.weight"].view(torch.uint8)
        assert q_b.shape == (2560, 2048)
        assert torch.equal(q_b[::256, 0].float(), heads)
        assert torch.equal(q_b[255::256, -1].float(), heads)
        q_b_scale = shard["self_attn.q_b_proj.weight_scale_inv"]
        assert torch.equal(q_b_scale[::2, 0], heads)
        kv_b = shard["self_attn.kv_b_proj.weight"].view(torch.uint8)
        assert kv_b.shape == (4480, 512)
        assert torch.equal(kv_b[447::448, 7].float(), heads)
        blocks = torch.zeros(35)
        blocks[: 7 * hv // 2] = (
            torch.arange(35 * (rank - 1), 35 * (rank - 1) + 7 * hv // 2) + 1
        )
        assert torch.equal(shard["self_attn.kv_b_proj.weight_scale_inv"][:, 3], blocks)
        o_proj = shard["self_attn.o_proj.weight"].view(torch.uint8)
        assert o_proj.shape == (6144, 2560)
        assert torch.equal(o_proj[11, ::256].float(), heads)
        assert torch.equal(shard["self_attn.o_proj.weight_scale_inv"][5, ::2], heads)
        assert shard["input_layernorm.weight"] is layer["input_layernorm.weight"]


def test_split_kv_b_proj_maps_head_rows_and_64_row_scale_chunks():
    g = torch.Generator().manual_seed(3)
    weight = _codes((4480, 512), g)
    scale = torch.rand(35, 4, generator=g)
    w_uk, w_uk_scale, w_uv, w_uv_scale = _attention_packing.split_kv_b_proj(
        weight, scale
    )
    assert w_uk.shape == (10, 512, 192) and w_uk.dtype == _FP8
    assert w_uv.shape == (10, 256, 512) and w_uv.dtype == _FP8
    assert w_uk_scale.shape == (10, 4, 3) and w_uv_scale.shape == (10, 4, 4)
    codes = weight.view(torch.uint8)
    for head in (0, 3, 9):
        rows = codes[448 * head : 448 * head + 448]
        assert torch.equal(w_uk[head].view(torch.uint8), rows[:192].t())
        assert torch.equal(w_uv[head].view(torch.uint8), rows[192:])
        for j in range(3):
            assert torch.equal(
                w_uk_scale[head, :, j], scale[(448 * head + 64 * j) // 128]
            )
        for j in range(4):
            assert torch.equal(
                w_uv_scale[head, j], scale[(448 * head + 192 + 64 * j) // 128]
            )


@pytest.mark.parametrize("rank", [0, 7])
def test_public_attention_packing_wires_checkpoint_tensors_to_layouts(rank):
    from flashinfer.experimental.glm5_tp8_decode.attention_weights import (
        shard_attention_weights,
    )
    from flashinfer.glm5_tp8_decode import pack_glm5_tp8_attention_weights

    layer = _attention_layer(seed=rank)
    layer["mlp.gate.weight"] = torch.empty(0)  # other entries are ignored
    snapshot = {name: tensor.clone() for name, tensor in layer.items()}
    packed = pack_glm5_tp8_attention_weights(layer, rank=rank, device="cpu")
    shard = shard_attention_weights(layer, rank=rank)
    pack = _attention_packing

    def proj(name):
        return shard[f"self_attn.{name}.weight"], shard[
            f"self_attn.{name}.weight_scale_inv"
        ]

    expected = {
        "x_gamma": shard["input_layernorm.weight"].float(),
        "q_gamma": shard["self_attn.q_a_layernorm.weight"].float(),
    }
    if rank == 0:
        expected.update(
            q_a_indexer_wk=pack.pack_q_a_indexer_wk(
                *proj("q_a_proj"), *proj("indexer.wk")
            ),
            indexer_weights_proj=pack.pack_indexer_weights_proj(
                shard["self_attn.indexer.weights_proj.weight"]
            ),
            indexer_wq_b=pack.pack_indexer_wq_b(*proj("indexer.wq_b")),
            ln_gamma=shard["self_attn.indexer.k_norm.weight"].float(),
            ln_beta=shard["self_attn.indexer.k_norm.bias"].float(),
        )
    else:
        w_uk, w_uk_scale, w_uv, w_uv_scale = pack.split_kv_b_proj(*proj("kv_b_proj"))
        expected.update(
            kv_gamma=shard["self_attn.kv_a_layernorm.weight"].float(),
            qkv_a_proj=pack.pack_qkv_a_proj(
                *proj("q_a_proj"), *proj("kv_a_proj_with_mqa")
            ),
            q_b_proj=pack.pack_q_b_proj(*proj("q_b_proj")),
            w_uk=pack.pack_w_uk(w_uk, w_uk_scale),
            w_uv=pack.pack_w_uv(w_uv, w_uv_scale),
            o_proj=pack.pack_o_proj(*proj("o_proj")),
        )
    assert packed.rank == rank
    assert list(packed.tensors) == list(expected)
    inputs = [t.untyped_storage().data_ptr() for t in layer.values() if t.numel()]
    for name, tensor in packed.tensors.items():
        assert tensor.is_contiguous(), name
        assert tensor.untyped_storage().data_ptr() not in inputs, name
        assert _same_bits(tensor, expected[name]), name
    assert all(_same_bits(layer[name], snapshot[name]) for name in layer), (
        "packing modified its input"
    )
    if rank == 7:
        # Six absent head slots: zero codes and zero scales in every page.
        tensors = packed.tensors
        assert not tensors["q_b_proj"][24:60].any()  # NOPE blocks of heads 4..9
        assert not tensors["q_b_proj"][68:80].any()  # RoPE blocks of heads 4..9
        assert not tensors["w_uk"][32:].any() and not tensors["w_uv"][32:].any()
        assert not tensors["o_proj"][:, 2:].any()  # columns of heads 4..9
        assert tensors["o_proj"][:, :2].any()


def test_public_attention_packing_validates_checkpoint_entries():
    from flashinfer.glm5_tp8_decode import pack_glm5_tp8_attention_weights

    with pytest.raises(ValueError, match="missing"):
        pack_glm5_tp8_attention_weights({}, rank=0)
    with pytest.raises(ValueError, match="rank"):
        pack_glm5_tp8_attention_weights({}, rank=8)
    with pytest.raises(ValueError, match="rank"):
        pack_glm5_tp8_attention_weights({}, rank=True)
    with pytest.raises(TypeError, match="mapping"):
        pack_glm5_tp8_attention_weights([], rank=0)
    layer = _attention_layer(norm_dtype=torch.float32)
    del layer["self_attn.o_proj.weight_scale_inv"]
    with pytest.raises(ValueError, match="self_attn.o_proj.weight_scale_inv"):
        pack_glm5_tp8_attention_weights(layer, rank=3)
    bad = dict(layer, **{"self_attn.q_a_proj.weight": torch.empty(2048, 6143)})
    with pytest.raises(ValueError, match="q_a_proj.weight.*shape"):
        pack_glm5_tp8_attention_weights(bad, rank=0)
    weights_proj = layer["self_attn.indexer.weights_proj.weight"].float()
    bad = dict(layer, **{"self_attn.indexer.weights_proj.weight": weights_proj})
    with pytest.raises(TypeError, match="weights_proj.weight.*dtype"):
        pack_glm5_tp8_attention_weights(bad, rank=0)
    with pytest.raises(ValueError, match="CPU or CUDA"):
        pack_glm5_tp8_attention_weights(layer, rank=0, device="meta")
    # FP32 norms are accepted; the MLA-only entry removed above is not read.
    packed = pack_glm5_tp8_attention_weights(layer, rank=0)
    assert torch.equal(
        packed.tensors["ln_beta"], layer["self_attn.indexer.k_norm.bias"]
    )


def _moe_layer(gamma_shape=(6144,), router_dtype=torch.bfloat16):
    """A full-shape MoE layer whose experts alias one set of zero tensors."""
    gate_up = torch.zeros((2048, 6144), dtype=torch.uint8).view(_FP8)
    down = torch.zeros((6144, 2048), dtype=torch.uint8).view(_FP8)
    layer = {
        "post_attention_layernorm.weight": torch.ones(gamma_shape),
        "mlp.gate.weight": torch.zeros((256, 6144), dtype=router_dtype),
        "mlp.gate.e_score_correction_bias": torch.zeros(256),
    }
    for prefix in ["mlp.shared_experts"] + [f"mlp.experts.{e}" for e in range(256)]:
        for name, weight, scale in (
            ("gate_proj", gate_up, torch.ones(16, 48)),
            ("up_proj", gate_up, torch.ones(16, 48)),
            ("down_proj", down, torch.ones(48, 16)),
        ):
            layer[f"{prefix}.{name}.weight"] = weight
            layer[f"{prefix}.{name}.weight_scale_inv"] = scale
    return layer


def test_public_moe_packing_rejects_bad_entries_without_large_allocation():
    from flashinfer.glm5_tp8_decode import pack_glm5_tp8_moe_weights

    with pytest.raises(ValueError, match="rank"):
        pack_glm5_tp8_moe_weights(_moe_layer(), rank=-1)
    with pytest.raises(ValueError, match="post_attention_layernorm.weight.*6144"):
        pack_glm5_tp8_moe_weights(_moe_layer(gamma_shape=(6143,)), rank=0)
    with pytest.raises(TypeError, match="mlp.gate.weight.*dtype"):
        pack_glm5_tp8_moe_weights(_moe_layer(router_dtype=torch.float32), rank=0)
    layer = _moe_layer()
    del layer["mlp.experts.255.down_proj.weight_scale_inv"]
    with pytest.raises(ValueError, match="mlp.experts.255.down_proj.weight_scale_inv"):
        pack_glm5_tp8_moe_weights(layer, rank=0)


@pytest.mark.parametrize("rank", [0, 5, 7])
def test_moe_shard_keeps_rank_intermediate_channels_as_views(rank):
    from flashinfer.experimental.glm5_tp8_decode.weights import shard_moe_weights

    layer = _moe_layer()
    # Distinct expert: label rows (gate/up) or columns (down) by channel.
    channel = (torch.arange(2048) % 251).to(torch.uint8)
    gate = channel.view(-1, 1).expand(2048, 6144).view(_FP8)
    down = channel.view(1, -1).expand(6144, 2048).view(_FP8)
    gate_scale = torch.arange(16).float().view(-1, 1).expand(16, 48)
    down_scale = torch.arange(16).float().view(1, -1).expand(48, 16)
    expert = "mlp.experts.200"
    layer[f"{expert}.gate_proj.weight"] = gate
    layer[f"{expert}.gate_proj.weight_scale_inv"] = gate_scale
    layer[f"{expert}.down_proj.weight"] = down
    layer[f"{expert}.down_proj.weight_scale_inv"] = down_scale
    shard = shard_moe_weights(layer, rank=rank)
    assert len(shard) == 1545
    rows = torch.arange(256 * rank, 256 * rank + 256) % 251
    local_gate = shard[f"{expert}.gate_proj.weight"]
    assert local_gate.shape == (256, 6144)
    assert torch.equal(local_gate.view(torch.uint8)[:, 9].long(), rows)
    assert torch.equal(
        shard[f"{expert}.gate_proj.weight_scale_inv"][:, 0],
        torch.tensor([2.0 * rank, 2.0 * rank + 1]),
    )
    local_down = shard[f"{expert}.down_proj.weight"]
    assert local_down.shape == (6144, 256)
    assert torch.equal(local_down.view(torch.uint8)[17].long(), rows)
    assert shard[f"{expert}.down_proj.weight_scale_inv"].shape == (48, 2)
    assert torch.equal(
        shard[f"{expert}.down_proj.weight_scale_inv"][0],
        torch.tensor([2.0 * rank, 2.0 * rank + 1]),
    )
    for name, tensor in shard.items():
        source = layer[name]
        assert (
            tensor.untyped_storage().data_ptr() == source.untyped_storage().data_ptr()
        )
    assert shard["mlp.gate.weight"] is layer["mlp.gate.weight"]


def test_packed_attention_validation_rejects_short_gamma_before_large_weights():
    from flashinfer.experimental.glm5_tp8_decode.attention_weights import (
        AttentionWeights,
        validate_packed,
    )

    # Supply the complete key set but only the two small vectors preceding
    # projection blobs. The short gamma must fail before blobs are inspected.
    weights = AttentionWeights(
        rank=1,
        tensors={
            "x_gamma": torch.empty(6144, dtype=torch.float32),
            "q_gamma": torch.empty(2047, dtype=torch.float32),
            "kv_gamma": None,
            "qkv_a_proj": None,
            "q_b_proj": None,
            "w_uk": None,
            "w_uv": None,
            "o_proj": None,
        },
    )
    with pytest.raises(ValueError, match="q_gamma.*shape"):
        validate_packed(weights, rank=1, device=torch.device("cpu"))
