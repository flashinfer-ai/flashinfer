# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent-oracle regressions for communication-free MXFP8 x MXFP4 FC12.

Valid output rows are unweighted BF16 route terms. Padding is deliberately not
checked: the lean ABI leaves it undefined. No communication backend is used.
"""

import pytest
import torch


def _require_blackwell():
    if not torch.cuda.is_available():
        pytest.skip("Lean MXFP8 x MXFP4 FC12 requires CUDA")
    pytest.importorskip("cutlass")
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("requires SM100/SM103 and CUDA 12.8+")


def _case(counts, scale_alignment, *, hidden=256, intermediate=256):
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_abi import (
        LeanFc12Config,
        LeanFc12Inputs,
        build_lean_row_plan,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.lean_kernel import (
        LeanFc12Launcher,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
        compute_reference,
        interleave_gate_up_8,
        quantize_mxfp4,
        quantize_mxfp8,
    )
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        to_atom_swizzled,
    )

    experts = len(counts)
    data_capacity = sum((count + 63) // 64 * 64 for count in counts)
    scale_capacity = sum(
        (count + scale_alignment - 1) // scale_alignment * scale_alignment
        for count in counts
    )
    config = LeanFc12Config(
        experts,
        hidden,
        intermediate,
        data_capacity,
        scale_capacity,
        data_row_alignment=64,
        scale_row_alignment=scale_alignment,
    )
    plan = build_lean_row_plan(counts, config)
    if scale_alignment == 128:
        assert plan.expert_data_row_offsets != plan.expert_scale_row_offsets

    generator = torch.Generator(device="cuda").manual_seed(20260922)

    def random(shape, scale):
        return torch.randn(shape, generator=generator, device="cuda") * scale

    # Vary K32 magnitudes so a misplaced scale row/block is observable.
    magnitudes = (
        torch.tensor([0.125, 0.25, 0.5, 1.0, 2.0, 0.5, 0.25, 1.0], device="cuda")
        .repeat_interleave(32)
        .repeat((hidden + 255) // 256)[:hidden]
    )
    xq, xs = quantize_mxfp8(random((sum(counts), hidden), 0.5) * magnitudes)
    w13q, w13s = quantize_mxfp4(
        interleave_gate_up_8(random((experts, 2 * intermediate, hidden), 0.125))
    )
    w2q, w2s = quantize_mxfp4(random((experts, hidden, intermediate), 0.125))
    ids = torch.repeat_interleave(
        torch.arange(experts, device="cuda"),
        torch.tensor(counts, device="cuda"),
    ).reshape(1, -1, 1)
    expected = compute_reference(
        input_q=xq.unsqueeze(0),
        input_scale=xs.unsqueeze(0),
        topk_ids=ids,
        topk_weights=torch.ones(ids.shape, device="cuda", dtype=torch.float32),
        w13_packed=w13q,
        w13_scale=w13s,
        w2_packed=w2q,
        w2_scale=w2s,
    ).route_terms[0, :, 0]

    activation = torch.zeros(
        (data_capacity, hidden), device="cuda", dtype=torch.float8_e4m3fn
    )
    scales = torch.full(
        (scale_capacity, hidden // 32), 127, device="cuda", dtype=torch.uint8
    )
    valid_rows = []
    offset = 0
    for expert, count in enumerate(counts):
        data_start = plan.expert_data_row_offsets[expert]
        scale_start = plan.expert_scale_row_offsets[expert]
        activation[data_start : data_start + count].copy_(xq[offset : offset + count])
        scales[scale_start : scale_start + count].copy_(xs[offset : offset + count])
        valid_rows.extend(range(data_start, data_start + count))
        offset += count

    def metadata(values):
        return torch.tensor(values, device="cuda", dtype=torch.int32)

    inputs = LeanFc12Inputs(
        activation=activation,
        activation_scales=scales,
        fc1_weight=w13q.transpose(1, 2),
        fc1_weight_scales=torch.stack([to_atom_swizzled(s) for s in w13s]),
        fc2_weight=w2q.transpose(1, 2),
        fc2_weight_scales=torch.stack([to_atom_swizzled(s) for s in w2s]),
        output=torch.empty(
            (data_capacity, hidden), device="cuda", dtype=torch.bfloat16
        ),
        expert_token_sizes=metadata(counts),
        expert_data_row_offsets=metadata(plan.expert_data_row_offsets),
        expert_scale_row_offsets=metadata(plan.expert_scale_row_offsets),
    )
    rows = torch.tensor(valid_rows, device="cuda", dtype=torch.int64)
    return LeanFc12Launcher(config), inputs, rows, expected


@pytest.mark.arch_blackwell
# Per-expert valid-token counts. Besides empty and single-token experts, the
# cases cover 32-token epilogue-group boundaries (31/32/33), the N128 tile
# boundary (127/128), and multi-tile experts (129).
@pytest.mark.parametrize(
    "counts",
    [(0, 3, 65, 1), (129, 0, 1, 7), (31, 32, 33, 127), (128, 1, 0, 33)],
)
@pytest.mark.parametrize("scale_alignment", [64, 128])
def test_lean_fc12_asymmetric_experts_repeated(counts, scale_alignment):
    _require_blackwell()
    launcher, inputs, rows, expected = _case(counts, scale_alignment)
    original_data = inputs.activation.view(torch.uint8).clone()
    original_scales = inputs.activation_scales.clone()
    baseline = None
    for _ in range(3):
        inputs.output.fill_(float("nan"))
        actual = launcher.run(inputs)
        torch.cuda.synchronize()
        valid = actual[rows]
        torch.testing.assert_close(valid, expected, atol=0.015, rtol=0.05)
        if baseline is None:
            baseline = valid.clone()
        else:
            torch.testing.assert_close(valid, baseline, atol=0, rtol=0)
    assert torch.equal(inputs.activation.view(torch.uint8), original_data)
    assert torch.equal(inputs.activation_scales, original_scales)


@pytest.mark.arch_blackwell
# N64 token tiles share one 128-row SFB atom: odd tiles (row 64 of 65/129/257)
# read the atom's upper half, and the trailing single-token expert after an
# odd tile count exercises FC1-done counter sizing by N64 tiles.
@pytest.mark.parametrize("counts", [(33, 65, 7, 129), (257, 1), (200, 31, 257, 5)])
@pytest.mark.parametrize("scale_alignment", [64, 128])
def test_lean_fc12_n64_multi_tile(counts, scale_alignment):
    _require_blackwell()
    launcher, inputs, rows, expected = _case(counts, scale_alignment)
    launcher.tactic["mma_tiler_mnk"] = (128, 64, 128)
    for _ in range(3):
        inputs.output.fill_(float("nan"))
        actual = launcher.run(inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[rows], expected, atol=0.015, rtol=0.05)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tile_tokens", [64, 128, 192])
@pytest.mark.parametrize("warps", [4, 8])
@pytest.mark.parametrize("fast_math,clamp", [(False, None), (True, 0.5)])
def test_native_fc1_keeps_padding_and_is_deterministic(
    tile_tokens, warps, fast_math, clamp
):
    _require_blackwell()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        from_atom_swizzled,
    )

    # Cross every SF atom sub-row and K-block boundary, full/partial groups,
    # and N64/N128/N192 tile boundaries with unequal data and scale row bases.
    counts = (0, 1, 31, 32, 33, 65, 129, 193)
    saved = []
    for _ in range(2):
        launcher, inputs, rows, _ = _case(counts, 128, intermediate=640)
        launcher.tactic.update(
            mma_tiler_mnk=(128, tile_tokens, 128), epilogue_warps=warps
        )
        launcher.gate_up_clamp = clamp
        launcher.fast_math = fast_math
        launcher.run(inputs)  # Allocate and compile before seeding guard bytes.
        launcher._fc1.view(torch.uint8).fill_(0xA5)
        launcher._fc1_sf.view(torch.uint8).fill_(0xA5)
        output = launcher.run(inputs)
        scales = from_atom_swizzled(
            launcher._fc1_sf.view(torch.uint8).flatten(),
            rows=launcher._sf_capacity,
            k_elements=launcher.config.intermediate,
        )
        data_padding = torch.ones(
            launcher._fc1.shape[0], dtype=torch.bool, device="cuda"
        )
        data_padding[rows] = False
        scale_padding = torch.ones(
            launcher._sf_capacity, dtype=torch.bool, device="cuda"
        )
        for start, count in zip(launcher._sf_offsets.tolist(), counts, strict=False):
            scale_padding[start : start + count] = False
        assert torch.all(launcher._fc1.view(torch.uint8)[data_padding] == 0xA5)
        assert torch.all(scales[scale_padding] == 0xA5)
        saved.append(
            (
                output[rows].view(torch.uint8).clone(),
                launcher._fc1.view(torch.uint8).clone(),
                launcher._fc1_sf.view(torch.uint8).clone(),
            )
        )
    for before, after in zip(saved[0], saved[1], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)


@pytest.mark.arch_blackwell
def test_native_fc1_preserves_amax_zero_seed():
    _require_blackwell()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.weight_scale_views import (
        from_atom_swizzled,
    )

    saved = []
    for _ in range(2):
        launcher, inputs, rows, _ = _case((128,), 64)
        inputs.activation.view(torch.uint8).fill_(0x7F)  # E4M3 NaN
        launcher.run(inputs)
        scales = from_atom_swizzled(
            launcher._fc1_sf.view(torch.uint8).flatten(),
            rows=launcher._sf_capacity,
            k_elements=launcher.config.intermediate,
        )[:128].clone()
        saved.append((launcher._fc1.view(torch.uint8)[rows].clone(), scales))
    for before, after in zip(saved[0], saved[1], strict=True):
        torch.testing.assert_close(before, after, rtol=0, atol=0)
    assert torch.count_nonzero(saved[1][1]) == 0


@pytest.mark.arch_blackwell
def test_lean_fc12_wait_stats(monkeypatch):
    # Debug-only wait accounting must not change results, and every FC1/FC2
    # tile must be counted exactly once: 1+2+1+3 N64 token tiles times 4 FC1
    # (2 * 256 / 128) and 2 FC2 (256 / 128) feature tiles.
    _require_blackwell()
    from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.wait_stats import (
        MAX_CTAS,
        SLOTS,
        WAIT_STATS_ENV,
        Slot,
    )

    monkeypatch.setenv(WAIT_STATS_ENV, "1")
    launcher, inputs, rows, expected = _case((33, 65, 7, 129), 128)
    launcher.tactic["mma_tiler_mnk"] = (128, 64, 128)
    actual = launcher.run(inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[rows], expected, atol=0.015, rtol=0.05)
    stats = launcher.wait_stats().reshape(MAX_CTAS, SLOTS)
    assert int(stats[:, Slot.EPI_FC1_TILE_COUNT].sum()) == 28
    assert int(stats[:, Slot.EPI_FC2_TILE_COUNT].sum()) == 14
    assert int((stats[:, Slot.MMA_LOOP_NS] > 0).sum()) > 0


@pytest.mark.arch_blackwell
def test_lean_fc12_large_hidden():
    _require_blackwell()
    launcher, inputs, rows, expected = _case(
        (0, 3, 5, 1), 64, hidden=2048, intermediate=128
    )
    actual = launcher.run(inputs)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual[rows], expected, atol=0.015, rtol=0.05)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("scale_alignment", [64, 128])
def test_lean_fc12_cuda_graph_replay(scale_alignment):
    _require_blackwell()
    launcher, inputs, rows, expected = _case((0, 3, 65, 1), scale_alignment)
    baseline = launcher.run(inputs)[rows].clone()  # JIT/scratch before capture.
    torch.cuda.synchronize()
    torch.testing.assert_close(baseline, expected, atol=0.015, rtol=0.05)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = launcher.run(inputs)
    for _ in range(3):
        inputs.output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual[rows], baseline, atol=0, rtol=0)
        torch.testing.assert_close(actual[rows], expected, atol=0.015, rtol=0.05)


@pytest.mark.arch_blackwell
@pytest.mark.parametrize("tile_tokens", [64, 128, 192])
@pytest.mark.parametrize("mma_m", [128, 256])
def test_lean_eight_warp_epilogue_is_bit_exact(tile_tokens, mma_m):
    """Both warpgroups must drain even when E1 gives one no token groups."""
    _require_blackwell()
    baseline = None
    for warps in (4, 8):
        launcher, inputs, rows, expected = _case((1, 33, 65, 193), 64)
        launcher.tactic["epilogue_warps"] = warps
        launcher.tactic["mma_tiler_mnk"] = (mma_m, tile_tokens, 128)
        launcher.tactic["cluster_shape_mnk"] = (2 if mma_m == 256 else 1, 1, 1)
        actual = launcher.run(inputs)[rows].clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, atol=0.015, rtol=0.05)
        if baseline is None:
            baseline = actual
        else:
            assert torch.equal(actual, baseline)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = launcher.run(inputs)
        for _ in range(3):
            inputs.output.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(output[rows], baseline)
