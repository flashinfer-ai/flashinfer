# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Single-GPU checks of the native MXFP8 MoK training kernels.

The reference is MoK's MXFP8 recipe in FP32 PyTorch arithmetic: E4M3 data, one E8M0
scale per 32-element block (``amax / 448`` rounded up to a power of two, floor
``1e-12``) in MoK's ``[rows / 128, cols / 128, 32, 16]`` scale-tile layout. Per
geometry the test checks the fused dispatch quantization bitwise in both layouts,
every BF16 GEMM output against the FP32 product of the kernel's own dequantized
E4M3 operands (atol = rtol = 1e-2 with at most ``max(4, 2e-7 * numel)`` exceptions),
the saved E4M3 context bitwise against the recipe applied to the kernel's BF16
activations, the re-quantized gradients, the exact score derivative, the routed
weight gradients (BF16 or FP32 accumulation), the BF16 shared-expert path against
an independent BF16-rounding-point reference, bitwise repeatability and the bitwise
identity of the recomputed context. The relative error against a plain FP32 oracle
is printed for the documentation.
"""

import pytest
import torch

from flashinfer.experimental.cake_mok_bf16._kernels import (
    MoKBackwardMxfp8,
    MoKForwardMxfp8,
    MoKMxfp8Quantize,
)

from flashinfer.experimental.cake_mok_bf16.mxfp8_reference import (
    dequantize_mxfp8,
    mxfp8_quantize_reference,
)


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("Requires an SM100a-compatible CUDA device")


def _swiglu(gate, up, limit):
    gate_f, up_f = gate.float(), up.float()
    if limit is not None:
        gate_f = torch.clamp(gate_f, max=limit)
        up_f = torch.clamp(up_f, min=-limit, max=limit)
    return (gate_f * torch.sigmoid(gate_f) * up_f).bfloat16()


# --- check rules --------------------------------------------------------------------------


def _bf16_rule(name, actual, expected):
    """Elementwise atol = rtol = 1e-2 with at most ``max(4, 2e-7 * numel)`` exceptions."""
    delta = (actual.float() - expected.float()).abs()
    tolerance = 1e-2 + 1e-2 * expected.float().abs()
    exceptions = int((delta > tolerance).sum())
    allowed = int(max(4, 2e-7 * expected.numel()))
    assert exceptions <= allowed, (
        f"{name}: {exceptions} elements beyond atol = rtol = 1e-2 (allowed {allowed}), "
        f"max |delta| {float(delta.max()) if delta.numel() else 0.0:.3e}"
    )


def _tiles_equal(name, actual, expected):
    """Bitwise compare MoK scale tiles given in any ``[..., 32, 16]`` shape."""
    a = actual.reshape(-1, 32, 16)
    e = expected.reshape(-1, 32, 16)
    assert a.shape == e.shape, (
        f"{name}: {tuple(a.shape)} vs {tuple(e.shape)} scale tiles"
    )
    mismatch = a != e
    assert not mismatch.any(), (
        f"{name}: {int(mismatch.sum())} scale bytes differ in "
        f"{int(mismatch.flatten(1).any(dim=1).sum())} tiles"
    )


def _codes_close(name, actual_fp8, actual_sc, expected_fp8, expected_sc, rows):
    """Two MXFP8 encodings of (nearly) the same BF16 values: FP32 rounding of ``exp``
    may flip rare codes, so a mismatch fraction below 2e-3 is required in both parts."""
    a = actual_fp8[:rows].view(torch.uint8)
    e = expected_fp8[:rows].view(torch.uint8)
    codes = (a != e).float().mean().item()
    cols = actual_fp8.shape[1] // 128
    sa = actual_sc.reshape(-1, cols, 32, 16)[: rows // 128]
    se = expected_sc.reshape(-1, cols, 32, 16)[: rows // 128]
    scales = (sa != se).float().mean().item()
    assert codes < 2e-3 and scales < 2e-3, (
        f"{name}: code mismatch fraction {codes:.2e}, scale mismatch fraction {scales:.2e}"
    )


def _forward_defined(index, tensor, rows, hidden, intermediate, macro):
    """Defined slice of forward output ``index`` (``MoKForwardMxfp8`` order): ring-shaped
    outputs are written for the routed rows only, the rest is allocator content."""
    blocks = rows // 128
    if index in (0, 9):  # transposed E4M3 rings [cols, macro]
        return tensor[:, :rows]
    if index in (
        1,
        10,
    ):  # transposed scale tiles [(cols / 128) * (macro / 128), 32, 16]
        cols = hidden if index == 1 else intermediate
        return tensor.view(cols // 128, macro // 128, 32, 16)[:, :blocks]
    if index in (3, 6, 12):  # routed rings [macro, cols]
        return tensor[:rows]
    if index in (4, 7):  # normal scale tiles, row-block major
        return tensor[: blocks * (intermediate // 128)]
    return tensor


def _backward_defined(index, tensor, rows, hidden, intermediate):
    """Defined slice of backward output ``index`` (``MoKBackwardMxfp8`` order)."""
    blocks = rows // 128
    if index in (1, 3, 6, 9, 10):  # routed rings [macro, cols]
        return tensor[:rows]
    if index in (4, 7):  # d_gate / d_up scale tiles, row-block major
        return tensor[: blocks * (intermediate // 128)]
    if index == 11:  # dispatched dy scale tiles
        return tensor[: blocks * (hidden // 128)]
    return tensor


# (hidden, intermediate, macrobatch, source tokens, swiglu limit, FP32 weight gradients)
GEOMETRIES = {
    "base": ((256, 256, 1536, 512, None, False), (512, 512, 1536, 512, None, False)),
    "clamped": ((256, 256, 1536, 512, 0.08, False), (512, 512, 1536, 501, 0.1, False)),
    "ragged": ((256, 256, 1536, 501, None, False), (512, 512, 1536, 129, None, False)),
    "wgrad_f32": ((256, 256, 1536, 512, None, True), (512, 512, 1536, 501, 0.1, True)),
}


def test_mxfp8_quantize_matches_recipe():
    """The exported quantizer reproduces the FP32 recipe bitwise in all three layout variants."""
    _require_gpu()
    quantize = MoKMxfp8Quantize()
    torch.manual_seed(1105)
    x = torch.randn(3, 256, 384, device="cuda", dtype=torch.bfloat16)
    x[0, :32, :64] = 0.0  # all-zero blocks take the scale floor
    x[1, 40, 100:132] = 448.0 * 4  # saturating magnitudes
    x[2, 200:232, 32:64] = 1e-20  # tiny values (scale floor, codes saturate to zero)
    for return_normal, return_transposed in (
        (True, True),
        (True, False),
        (False, True),
    ):
        expected = mxfp8_quantize_reference(x, return_normal, return_transposed)
        actual = quantize(x, return_normal, return_transposed)
        for index, (a, e) in enumerate(zip(actual, expected, strict=True)):
            assert (a is None) == (e is None), index
            if a is not None:
                if a.dtype == torch.float8_e4m3fn:
                    assert torch.equal(a.view(torch.uint8), e.view(torch.uint8)), index
                else:
                    _tiles_equal(f"quantize output {index}", a, e)
    two_d = quantize(x[0], True, True)
    assert two_d[0].shape == (256, 384) and two_d[2].shape == (384, 256)


@pytest.mark.parametrize("variant", sorted(GEOMETRIES))
def test_fused_training_mxfp8(monkeypatch, variant):
    _require_gpu()
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )
    forward = MoKForwardMxfp8()
    backward = MoKBackwardMxfp8(wgrad_f32=GEOMETRIES[variant][0][5])
    # The hosts drop their ring references after each launch by default; this
    # test inspects the dispatched and saved tiles, so it keeps them.
    forward.keep_rings = True
    backward.keep_rings = True
    for geometry in GEOMETRIES[variant]:
        _check_geometry(forward, backward, *geometry)


def _check_geometry(
    forward, backward, hidden, intermediate, macro, tokens, limit, wgrad_f32
):
    experts, topk, mini, comm = 4, 2, 256, 4
    torch.manual_seed(9173 + hidden)
    options = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(tokens, hidden, **options) * 0.125
    dy = torch.randn_like(x) * 0.125
    shared = [
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(intermediate, hidden, **options) / hidden**0.5,
        torch.randn(hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed = [
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, intermediate, hidden, **options) / hidden**0.5,
        torch.randn(experts, hidden, intermediate, **options) / intermediate**0.5,
    ]
    routed_q = [mxfp8_quantize_reference(w, True, True) for w in routed]
    scores = torch.rand(tokens, topk, device="cuda") + 0.1
    scores.div_(scores.sum(-1, keepdim=True)).mul_(2.5)
    row = torch.arange(tokens, device="cuda")
    routes = torch.stack([(row + slot) % experts for slot in range(topk)], -1)
    slots = [torch.where(routes.flatten() == expert)[0] for expert in range(experts)]
    count_values = [((value.numel() + 255) // 256) * 256 for value in slots]
    schedule = torch.cat(
        [
            torch.cat(
                (
                    value.int(),
                    torch.full(
                        (count - value.numel(),), -1, dtype=torch.int32, device="cuda"
                    ),
                )
            )
            for value, count in zip(slots, count_values, strict=True)
        ]
    )
    T = schedule.numel()
    assert macro >= T, "single-macrobatch test geometry"
    schedule = torch.cat(
        (schedule, torch.full((512,), -1, dtype=torch.int32, device="cuda"))
    )
    peers = torch.where(schedule >= 0, 0, -1).int()
    num_tokens = torch.tensor([T], device="cuda", dtype=torch.int32)
    counts = torch.tensor(count_values, device="cuda", dtype=torch.int32)
    combine = torch.zeros(tokens * topk, hidden, **options)
    dx_peer = torch.zeros(tokens * topk, hidden, **options)
    ds_peer = torch.zeros_like(scores)
    valid = schedule[:T] >= 0
    source = schedule[:T][valid].long()
    t_blocks = T // 128

    # MoK's forward signature passes (w_fp8, w_sc) pairs; the backward the gate/up 4-tuples
    # and the down (w_t_fp8, w_t_sc) pair.
    forward_weights = [(q[0], q[1]) for q in routed_q]
    backward_weights = [routed_q[0], routed_q[1], (routed_q[2][2], routed_q[2][3])]

    def run_forward(recompute_only=False):
        # MoK's recompute_forward_context takes no down weights.
        return forward(
            x,
            [x.data_ptr()],
            combine,
            [combine.data_ptr()],
            shared[0],
            forward_weights[0],
            shared[1],
            forward_weights[1],
            shared[2],
            None if recompute_only else forward_weights[2],
            peers,
            schedule,
            num_tokens,
            counts,
            topk,
            limit,
            comm,
            macro,
            mini,
            recompute_only=recompute_only,
        )

    def run_backward(context):
        dx_peer.zero_()
        ds_peer.zero_()
        return backward(
            dy,
            [dy.data_ptr()],
            dx_peer,
            [dx_peer.data_ptr()],
            scores,
            [scores.data_ptr()],
            ds_peer,
            [ds_peer.data_ptr()],
            shared[0],
            backward_weights[0],
            shared[1],
            backward_weights[1],
            shared[2],
            backward_weights[2],
            *context[:11],
            x,
            [x.data_ptr()],
            peers,
            schedule,
            num_tokens,
            counts,
            topk,
            limit,
            comm,
            macro,
            mini,
        )

    # ---------------------------------------------------------------- forward
    result = run_forward()
    rings = forward.last_rings
    xr = torch.zeros(T, hidden, **options)
    xr[valid] = x[source // topk]
    xq, xsc, xqt, xsct = mxfp8_quantize_reference(xr, True, True)
    tiles_x = t_blocks * (hidden // 128)
    assert torch.equal(
        rings["x_fp8_routed"][:T].view(torch.uint8), xq.view(torch.uint8)
    )
    _tiles_equal("dispatch scales", rings["x_sc_routed"][:tiles_x], xsc)
    assert torch.equal(result[0][:, :T].view(torch.uint8), xqt.view(torch.uint8))
    _tiles_equal(
        "dispatch transposed scales",
        result[1].view(hidden // 128, macro // 128, 32, 16)[:, :t_blocks],
        xsct.view(hidden // 128, t_blocks, 32, 16),
    )
    x_deq = dequantize_mxfp8(xq, xsc)
    w_deq = [
        [
            dequantize_mxfp8(q[0][e], q[1].view(experts, -1, *q[1].shape[1:])[e])
            for e in range(experts)
        ]
        for q in routed_q
    ]
    gate_ref = torch.empty(T, intermediate, **options)
    up_ref = torch.empty_like(gate_ref)
    y_fake = torch.empty(T, hidden, **options)
    offset = 0
    expert_rows = []
    for expert, count in enumerate(count_values):
        rows = slice(offset, offset + count)
        expert_rows.append(torch.arange(offset, offset + count, device="cuda"))
        gate_ref[rows] = (x_deq[rows] @ w_deq[0][expert].T).bfloat16()
        up_ref[rows] = (x_deq[rows] @ w_deq[1][expert].T).bfloat16()
        h = _swiglu(gate_ref[rows], up_ref[rows], limit)
        hq, hsc, _, _ = mxfp8_quantize_reference(h, True, False)
        y_fake[rows] = (dequantize_mxfp8(hq, hsc) @ w_deq[2][expert].T).bfloat16()
        offset += count
    _bf16_rule("gate", rings["gate_routed"][:T], gate_ref)
    _bf16_rule("up", rings["up_routed"][:T], up_ref)
    tiles_i = t_blocks * (intermediate // 128)
    for name, ring, fp8_out, sc_out in (
        ("gate", rings["gate_routed"], result[3], result[4]),
        ("up", rings["up_routed"], result[6], result[7]),
    ):
        rq, rsc, _, _ = mxfp8_quantize_reference(ring[:T], True, False)
        assert torch.equal(fp8_out[:T].view(torch.uint8), rq.view(torch.uint8)), name
        _tiles_equal(f"saved {name} scales", sc_out[:tiles_i], rsc)
    h_src = _swiglu(rings["gate_routed"][:T], rings["up_routed"][:T], limit)
    hq, hsc, hqt, hsct = mxfp8_quantize_reference(h_src, True, True)
    _codes_close(
        "hidden", rings["hidden_fp8_routed"], rings["hidden_sc_routed"], hq, hsc, T
    )
    _codes_close(
        "hidden transposed",
        result[9][:, :T].T.contiguous(),
        result[10]
        .view(intermediate // 128, macro // 128, 32, 16)[:, :t_blocks]
        .transpose(0, 1)
        .contiguous(),
        hqt.T.contiguous(),
        hsct.view(intermediate // 128, t_blocks, 32, 16).transpose(0, 1).contiguous(),
        T,
    )
    h_deq = dequantize_mxfp8(
        rings["hidden_fp8_routed"][:T], rings["hidden_sc_routed"][:tiles_i]
    )
    y_kernel = result[12][:T]
    y_tight = torch.empty_like(y_kernel)
    for expert, rows in enumerate(expert_rows):
        y_tight[rows] = (h_deq[rows] @ w_deq[2][expert].T).bfloat16()
    _bf16_rule("y_routed (kernel hidden)", y_kernel, y_tight)
    _bf16_rule("y_routed (fake-quant reference)", y_kernel, y_fake)
    selected = torch.where(valid)[0]
    assert torch.equal(combine[schedule[:T][selected].long()], y_kernel[selected])
    gate_s = (x.float() @ shared[0].float().T).bfloat16()
    up_s = (x.float() @ shared[1].float().T).bfloat16()
    y_s = (_swiglu(gate_s, up_s, limit).float() @ shared[2].float().T).bfloat16()
    _bf16_rule("y_shared", result[11], y_s)

    def fwd_defined(outputs):
        return [
            _forward_defined(i, t, T, hidden, intermediate, macro)
            for i, t in enumerate(outputs)
            if t is not None
        ]

    for other in (run_forward(), run_forward()):
        for i, (a, b) in enumerate(
            zip(fwd_defined(result), fwd_defined(other), strict=True)
        ):
            assert torch.equal(a, b), f"forward output {i} differs between executions"
    recomputed = run_forward(recompute_only=True)
    assert recomputed[11] is None and recomputed[12] is None
    for i, (a, b) in enumerate(
        zip(fwd_defined(result[:11]), fwd_defined(recomputed[:11]), strict=True)
    ):
        assert torch.equal(a, b), f"recomputed context tensor {i} differs"

    # ---------------------------------------------------------------- backward
    context = run_forward()
    values = run_backward(context)
    rings = backward.last_rings
    (
        dx_shared,
        dx_routed,
        dg_shared,
        dg_fp8_routed,
        dg_sc_routed,
        du_shared,
        du_fp8_routed,
        du_sc_routed,
        dh_shared,
        dh_routed,
        dy_fp8_routed,
        dy_sc_routed,
        dwg_shared,
        dwg_routed,
        dwu_shared,
        dwu_routed,
        dwd_shared,
        dwd_routed,
    ) = values
    (
        x_fp8_t,
        x_sc_t,
        _,
        gate_fp8,
        gate_sc,
        _,
        up_fp8,
        up_sc,
        _,
        hidden_fp8_t,
        hidden_sc_t,
    ) = context[:11]

    def bwd_defined(outputs):
        return [
            _backward_defined(i, t, T, hidden, intermediate)
            for i, t in enumerate(outputs)
        ] + [dx_peer, ds_peer]

    assert all(torch.isfinite(t.float()).all() for t in bwd_defined(values))
    # 1. Score-gradient dispatch: unscaled rows normal, score-scaled rows transposed (bitwise).
    gathered = torch.zeros(T, hidden, **options)
    scaled = torch.zeros_like(gathered)
    gathered[valid] = dy[source // topk]
    scaled[valid] = (
        dy[source // topk].float() * scores.flatten()[source][:, None]
    ).bfloat16()
    ref_n = mxfp8_quantize_reference(gathered, True, False)
    ref_t = mxfp8_quantize_reference(scaled, False, True)
    assert torch.equal(dy_fp8_routed[:T].view(torch.uint8), ref_n[0].view(torch.uint8))
    _tiles_equal(
        "dy dispatch scales",
        dy_sc_routed.view(macro // 128, hidden // 128, 32, 16)[:t_blocks],
        ref_n[1].view(t_blocks, hidden // 128, 32, 16),
    )
    assert torch.equal(
        rings["dy_fp8_t_routed"][:, :T].view(torch.uint8), ref_t[2].view(torch.uint8)
    )
    _tiles_equal(
        "dy dispatch transposed scales",
        rings["dy_sc_t_routed"].view(hidden // 128, macro // 128, 32, 16)[:, :t_blocks],
        ref_t[3].view(hidden // 128, t_blocks, 32, 16),
    )
    # 2. Down dgrad from the kernel's own E4M3 inputs.
    dy_deq = dequantize_mxfp8(dy_fp8_routed, dy_sc_routed)[:T]
    dh_ref = torch.zeros(T, intermediate, device="cuda")
    for expert, rows in enumerate(expert_rows):
        wd_t = dequantize_mxfp8(
            routed_q[2][2][expert], routed_q[2][3].view(experts, -1, 32, 16)[expert]
        )
        dh_ref[rows] = dy_deq[rows] @ wd_t.T
    _bf16_rule("dh_routed", dh_routed[:T], dh_ref.bfloat16())
    # 3. SwiGLU backward on the kernel's dh and the saved E4M3 gate/up.
    gate_f = dequantize_mxfp8(gate_fp8, gate_sc)[:T]
    up_f = dequantize_mxfp8(up_fp8, up_sc)[:T]
    if limit is None:
        gate_mask = up_mask = None
        gate_c, up_c = gate_f, up_f
    else:
        gate_mask = gate_f <= limit
        up_mask = (up_f >= -limit) & (up_f <= limit)
        gate_c = torch.clamp(gate_f, max=limit)
        up_c = torch.clamp(up_f, min=-limit, max=limit)
    sigmoid = torch.sigmoid(gate_c)
    silu = gate_c * sigmoid
    hidden_f = silu * up_c
    dh_f = dh_routed[:T].float()
    weight = torch.zeros(T, device="cuda")
    weight[valid] = scores.flatten()[source]
    dhs = dh_f * weight[:, None]
    dg_f = ((1.0 - silu) * sigmoid + silu) * up_c * dhs
    du_f = silu * dhs
    if limit is not None:
        dg_f = torch.where(gate_mask, dg_f, torch.zeros_like(dg_f))
        du_f = torch.where(up_mask, du_f, torch.zeros_like(du_f))
    dg_ref = mxfp8_quantize_reference(dg_f.bfloat16(), True, True)
    du_ref = mxfp8_quantize_reference(du_f.bfloat16(), True, True)
    _codes_close("d_gate", dg_fp8_routed, dg_sc_routed, dg_ref[0], dg_ref[1], T)
    _codes_close("d_up", du_fp8_routed, du_sc_routed, du_ref[0], du_ref[1], T)
    for name, ring_fp8, ring_sc, ref in (
        ("d_gate transposed", "dg_fp8_t_routed", "dg_sc_t_routed", dg_ref),
        ("d_up transposed", "du_fp8_t_routed", "du_sc_t_routed", du_ref),
    ):
        _codes_close(
            name,
            rings[ring_fp8][:, :T].T.contiguous(),
            rings[ring_sc]
            .view(intermediate // 128, macro // 128, 32, 16)[:, :t_blocks]
            .transpose(0, 1)
            .contiguous(),
            ref[2].T.contiguous(),
            ref[3]
            .view(intermediate // 128, t_blocks, 32, 16)
            .transpose(0, 1)
            .contiguous(),
            T,
        )
    dg_deq = dequantize_mxfp8(dg_fp8_routed, dg_sc_routed)[:T]
    du_deq = dequantize_mxfp8(du_fp8_routed, du_sc_routed)[:T]
    # Exact score derivative: sum(dh_unscaled * hidden) over the route's columns.
    ds_ref = torch.zeros(tokens * topk, device="cuda")
    ds_ref[source] = (dh_f * hidden_f).sum(-1)[valid]
    _bf16_rule("d_scores", ds_peer.flatten(), ds_ref)
    # 4. Input gradient: two-pair block-scaled GEMM on the kernel's d_gate/d_up codes.
    dx_ref = torch.zeros(T, hidden, device="cuda")
    for expert, rows in enumerate(expert_rows):
        wg_t = dequantize_mxfp8(
            routed_q[0][2][expert], routed_q[0][3].view(experts, -1, 32, 16)[expert]
        )
        wu_t = dequantize_mxfp8(
            routed_q[1][2][expert], routed_q[1][3].view(experts, -1, 32, 16)[expert]
        )
        dx_ref[rows] = dg_deq[rows] @ wg_t.T + du_deq[rows] @ wu_t.T
    _bf16_rule("dx_routed", dx_routed[:T], dx_ref.bfloat16())
    dx_combined = torch.zeros(tokens * topk, hidden, device="cuda")
    dx_combined[source] = dx_ref[valid]
    _bf16_rule("dx_combine", dx_peer, dx_combined.bfloat16())
    # 5. Routed weight gradients from the kernel's transposed E4M3 operands (K = expert rows).
    dy_t_deq = dequantize_mxfp8(rings["dy_fp8_t_routed"], rings["dy_sc_t_routed"])[
        :, :T
    ]
    h_t_deq = dequantize_mxfp8(hidden_fp8_t, hidden_sc_t)[:, :T]
    dg_t_deq = dequantize_mxfp8(rings["dg_fp8_t_routed"], rings["dg_sc_t_routed"])[
        :, :T
    ]
    du_t_deq = dequantize_mxfp8(rings["du_fp8_t_routed"], rings["du_sc_t_routed"])[
        :, :T
    ]
    x_t_deq = dequantize_mxfp8(x_fp8_t, x_sc_t)[:, :T]
    dwd_ref = torch.zeros(experts, hidden, intermediate, device="cuda")
    dwg_ref = torch.zeros(experts, intermediate, hidden, device="cuda")
    dwu_ref = torch.zeros_like(dwg_ref)
    for expert, rows in enumerate(expert_rows):
        if rows.numel():
            dwd_ref[expert] = dy_t_deq[:, rows] @ h_t_deq[:, rows].T
            dwg_ref[expert] = dg_t_deq[:, rows] @ x_t_deq[:, rows].T
            dwu_ref[expert] = du_t_deq[:, rows] @ x_t_deq[:, rows].T
    for name, actual, expected in (
        ("dwd_routed", dwd_routed, dwd_ref),
        ("dwg_routed", dwg_routed, dwg_ref),
        ("dwu_routed", dwu_routed, dwu_ref),
    ):
        if wgrad_f32:
            assert actual.dtype == torch.float32, name
            _bf16_rule(name, actual, expected)
        else:
            assert actual.dtype == torch.bfloat16, name
            _bf16_rule(name, actual, expected.bfloat16())
    # 6. Shared experts (BF16, independent BF16-rounding-point reference).
    gate_s = gate_s.float()
    up_s = up_s.float()
    if limit is None:
        gmask = umask = None
        gate_sc_, up_sc_ = gate_s, up_s
    else:
        gmask = gate_s <= limit
        umask = (up_s >= -limit) & (up_s <= limit)
        gate_sc_ = torch.clamp(gate_s, max=limit)
        up_sc_ = torch.clamp(up_s, min=-limit, max=limit)
    sig_s = torch.sigmoid(gate_sc_)
    silu_s = gate_sc_ * sig_s
    act_s = (silu_s * up_sc_).bfloat16()
    dh_s = (dy.float() @ shared[2].float()).bfloat16()
    dg_s = ((1.0 - silu_s) * sig_s + silu_s) * up_sc_ * dh_s.float()
    du_s = silu_s * dh_s.float()
    if limit is not None:
        dg_s = torch.where(gmask, dg_s, torch.zeros_like(dg_s))
        du_s = torch.where(umask, du_s, torch.zeros_like(du_s))
    dg_s, du_s = dg_s.bfloat16(), du_s.bfloat16()
    dx_s = (
        dg_s.float() @ shared[0].float() + du_s.float() @ shared[1].float()
    ).bfloat16()
    _bf16_rule("dh_shared", dh_shared, dh_s)
    _bf16_rule("dg_shared", dg_shared, dg_s)
    _bf16_rule("du_shared", du_shared, du_s)
    _bf16_rule("dx_shared", dx_shared, dx_s)
    _bf16_rule("dwg_shared", dwg_shared, (dg_s.float().T @ x.float()).bfloat16())
    _bf16_rule("dwu_shared", dwu_shared, (du_s.float().T @ x.float()).bfloat16())
    _bf16_rule("dwd_shared", dwd_shared, (dy.float().T @ act_s.float()).bfloat16())
    # 7. Relative error against a plain FP32 oracle (no quantization): documentation only.
    x_rows = torch.zeros(T, hidden, device="cuda")
    x_rows[valid] = x[source // topk].float()
    dx_o = torch.zeros(T, hidden, device="cuda")
    dwd_o, dwg_o, dwu_o = (
        torch.zeros_like(dwd_ref),
        torch.zeros_like(dwg_ref),
        torch.zeros_like(dwu_ref),
    )
    ds_o = torch.zeros(tokens * topk, device="cuda")
    for expert, rows in enumerate(expert_rows):
        if not rows.numel():
            continue
        wg, wu, wd = (w[expert].float() for w in routed)
        g, u = x_rows[rows] @ wg.T, x_rows[rows] @ wu.T
        if limit is not None:
            gm, um = g <= limit, (u >= -limit) & (u <= limit)
            g, u = torch.clamp(g, max=limit), torch.clamp(u, min=-limit, max=limit)
        sg = torch.sigmoid(g)
        sl = g * sg
        h = sl * u
        dyr = gathered[rows].float()
        dh = dyr @ wd
        w_rows = weight[rows][:, None]
        dgo = ((1.0 - sl) * sg + sl) * u * dh * w_rows
        duo = sl * dh * w_rows
        if limit is not None:
            dgo, duo = torch.where(gm, dgo, 0.0), torch.where(um, duo, 0.0)
        dx_o[rows] = dgo @ wg + duo @ wu
        dwd_o[expert] = (dyr * w_rows).T @ h
        dwg_o[expert] = dgo.T @ x_rows[rows]
        dwu_o[expert] = duo.T @ x_rows[rows]
        chosen = schedule[rows] >= 0
        ds_o[schedule[rows][chosen].long()] = (dh * h).sum(-1)[chosen]

    def rel(a, b):
        denom = b.float().abs().sum().item()
        return (a.float() - b.float()).abs().sum().item() / denom if denom else 0.0

    print(
        f"MXFP8 vs FP32 oracle (H={hidden}, I={intermediate}, tokens={tokens}, limit={limit}): "
        f"dx {rel(dx_routed[:T], dx_o):.4f} dWd {rel(dwd_routed, dwd_o):.4f} "
        f"dWg {rel(dwg_routed, dwg_o):.4f} dWu {rel(dwu_routed, dwu_o):.4f} "
        f"d_scores {rel(ds_peer.flatten(), ds_o):.4f}",
        flush=True,
    )
    # 8. Three executions bitwise identical.
    snapshot = [t.clone() for t in bwd_defined(values)]
    for _ in range(2):
        again = bwd_defined(run_backward(context))
        for i, (a, b) in enumerate(zip(snapshot, again, strict=True)):
            assert torch.equal(a, b), f"backward output {i} differs between executions"
