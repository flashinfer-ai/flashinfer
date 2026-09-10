"""Single-rank SM90 Humming MXFP4 MegaMoE vs a test-owned reference.

This is the MXFP4 counterpart of
``test_sm90_pull_fp8_kernel_vs_reference.py``.  The production path starts
from the canonical raw ABI -- packed E2M1 payload bytes plus K32 E8M0 scale
bytes -- and runs the real ``preprocess_mega_weights`` before entering the
public Hopper shim.  The oracle does not import the kernel donor or any raw
vendored package.  Instead, it independently reverses the physical Humming
interleave, unfolds offsets, constructs the exact E4M3 operand bytes, and
models the two FP8 WGMMA legs and their hybrid scales.

The SM90 and SM100 CuTeDSL trees use colliding top-level vendor module names,
so this file is intended to run in an isolated Hopper pytest process::

    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 pytest -v -m arch_hopper \
        tests/moe_ep/test_sm90_pull_mxfp4_kernel_vs_reference.py
"""

from __future__ import annotations

import pytest


E4M3_MAX = 448.0
FC2_SCALE_K = 64
HUMMING_GROUP_K = 32
GATE_UP_INTERLEAVE = 8
MMA_TILER_MNK = (128, 32, 128)


def _sm90_tree():
    """Import only FlashInfer's public SM90 package boundary."""
    try:
        import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg
    except RuntimeError as error:
        pytest.skip(f"SM90 kernel tree unavailable in this process: {error}")
    return pkg


def _require_hopper() -> None:
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("an SM90 Hopper GPU is required")


def _pack_e2m1_codes(codes):
    import torch

    if codes.dtype != torch.uint8 or codes.shape[-1] % 2:
        raise ValueError("E2M1 codes must be uint8 with an even logical K")
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()


def _make_raw_leg(*, experts: int, rows: int, logical_k: int, salt: int):
    """Build deterministic canonical E2M1/K32-E8M0 bytes.

    Each expert spans sixteen E8M0 exponents.  This is wider than Humming's
    retained range of eleven and therefore exercises both the payload rewrite
    and the common per-expert residual, rather than degenerating to a cast.
    """
    import torch

    expert = torch.arange(experts, device="cuda", dtype=torch.int64)[:, None, None]
    row = torch.arange(rows, device="cuda", dtype=torch.int64)[None, :, None]
    column = torch.arange(logical_k, device="cuda", dtype=torch.int64)[None, None, :]
    codes = (
        (expert * 11 + row * 5 + column * 3 + column // 17 + salt)
        .remainder(16)
        .to(torch.uint8)
    )

    group = torch.arange(
        logical_k // HUMMING_GROUP_K, device="cuda", dtype=torch.int64
    )[None, None, :]
    # [106, 121] gives a nontrivial clamp/rewrite while keeping the numerical
    # two-layer problem well conditioned.
    exponent = (106 + (expert * 7 + row * 3 + group * 5 + salt).remainder(16)).to(
        torch.uint8
    )
    return _pack_e2m1_codes(codes), exponent.contiguous()


def _single_rank_problem():
    import torch

    hidden = 128
    intermediate = 128
    experts = 2
    topk = 2
    num_tokens = 13
    max_tokens = 32

    generator = torch.Generator(device="cuda").manual_seed(3738)
    hidden_states = (
        torch.randn(
            num_tokens,
            hidden,
            dtype=torch.float32,
            device="cuda",
            generator=generator,
        )
        * 0.5
    ).to(torch.bfloat16)

    token = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
    topk_ids = torch.stack((token.remainder(2), (token + 1).remainder(2)), dim=1)
    first = 0.2 + 0.6 * (token.to(torch.float32) + 1.0) / (num_tokens + 1.0)
    topk_weights = torch.stack((first, 1.0 - first), dim=1).contiguous()

    w13, w13_scale = _make_raw_leg(
        experts=experts,
        rows=2 * intermediate,
        logical_k=hidden,
        salt=1,
    )
    w2, w2_scale = _make_raw_leg(
        experts=experts,
        rows=hidden,
        logical_k=intermediate,
        salt=9,
    )
    return {
        "hidden": hidden,
        "intermediate": intermediate,
        "experts": experts,
        "topk": topk,
        "num_tokens": num_tokens,
        "max_tokens": max_tokens,
        "gate_up_clamp": 4.0,
        "hidden_states": hidden_states.contiguous(),
        "topk_ids": topk_ids.contiguous(),
        "topk_weights": topk_weights,
        "w13": w13,
        "w13_scale": w13_scale,
        "w2": w2,
        "w2_scale": w2_scale,
    }


def _restore_preprocessed_signs(word: int) -> int:
    """Invert Humming's sign-bit permutation for one packed 32-bit word."""
    restored = word & 0x77777777
    physical_sign_bits = (7, 15, 23, 31, 3, 11, 19, 27)
    for source_nibble, physical_bit in enumerate(physical_sign_bits):
        restored |= ((word >> physical_bit) & 1) << (source_nibble * 4 + 3)
    return restored & 0xFFFFFFFF


def _undo_humming_interleave(physical):
    """Physical SM90 FP4-for-FP8 bytes -> logical packed E2M1 bytes."""
    import torch

    if physical.dtype != torch.uint8 or physical.ndim != 3:
        raise ValueError("physical Humming payload must be a 3D uint8 tensor")
    experts, rows, packed_k = physical.shape
    logical_k = packed_k * 2
    if rows % 16 or logical_k % 64 or packed_k % 2:
        raise ValueError("physical payload violates the M16/K64 contract")

    source = physical.detach().cpu().contiguous()
    physical_u16 = source[..., 0::2].to(torch.int64) | (
        source[..., 1::2].to(torch.int64) << 8
    )
    logical_u16 = torch.empty_like(physical_u16)
    for expert in range(experts):
        for block_id in range(rows // 2):
            source_row = (block_id // 8) * 16 + block_id % 8
            for partition in range(logical_k // 64):
                for lane in range(16):
                    physical_row = source_row + ((lane % 8) // 4) * 8
                    source_column = partition * 16 + lane
                    physical_column = partition * 16 + (lane // 8) * 8 + (lane % 4) * 2
                    word = int(physical_u16[expert, physical_row, physical_column]) | (
                        int(physical_u16[expert, physical_row, physical_column + 1])
                        << 16
                    )
                    word = _restore_preprocessed_signs(word)
                    logical_u16[expert, source_row, source_column] = word & 0xFFFF
                    logical_u16[expert, source_row + 8, source_column] = (
                        word >> 16
                    ) & 0xFFFF

    logical = torch.empty_like(source)
    logical[..., 0::2] = (logical_u16 & 0xFF).to(torch.uint8)
    logical[..., 1::2] = ((logical_u16 >> 8) & 0xFF).to(torch.uint8)
    return logical.contiguous()


def _unfold_humming_offsets(folded):
    """Invert ``[E,N/64,K/128,16,16]`` into logical K32 offsets."""
    import torch

    folded = folded.detach().cpu().contiguous()
    if folded.dtype != torch.uint8 or folded.ndim != 5:
        raise ValueError("folded offsets must be a 5D uint8 tensor")
    experts, n64, k128, folded_m, physical_cols = folded.shape
    if (folded_m, physical_cols) != (16, 16):
        raise ValueError("folded offset trailing dimensions must be (16, 16)")

    logical = torch.empty((experts, n64 * 64, k128 * 4), dtype=torch.uint8)
    for n_block in range(n64):
        for k_block in range(k128):
            for row_in_fold in range(16):
                for row_slice in range(4):
                    row = n_block * 64 + row_slice * 16 + row_in_fold
                    for k32 in range(4):
                        logical[:, row, k_block * 4 + k32] = folded[
                            :, n_block, k_block, row_in_fold, row_slice * 4 + k32
                        ]
    return logical.contiguous()


def _e2m1_offsets_to_e4m3_bytes(codes, offsets):
    """Construct the exact E4M3 bytes consumed by WGMMA."""
    import torch

    magnitude = (codes & 0x7).to(torch.int16)
    exponent_field = offsets.repeat_interleave(HUMMING_GROUP_K, dim=-1).to(torch.int16)
    base = exponent_field * 8
    encoded = torch.where(
        magnitude == 0,
        torch.zeros_like(base),
        torch.where(
            magnitude == 1,
            base,
            torch.where(
                magnitude == 2,
                base + 0x08,
                torch.where(
                    magnitude == 3,
                    base + 0x0C,
                    base + 0x10 + (magnitude - 4) * 4,
                ),
            ),
        ),
    )
    sign = (codes & 0x8).to(torch.int16) << 4
    return (encoded | sign).to(torch.uint8).contiguous()


def _decode_processed_leg(transformed):
    """Decode one production Humming leg without using production helpers."""
    import torch

    packed_storage_k, folded_offset, activation_placeholder, residual_x64 = transformed
    assert torch.equal(activation_placeholder, torch.ones_like(activation_placeholder))
    physical = packed_storage_k.transpose(1, 2)
    logical_packed = _undo_humming_interleave(physical)
    low = logical_packed & 0x0F
    high = (logical_packed >> 4) & 0x0F
    codes = torch.stack((low, high), dim=-1).reshape(
        logical_packed.shape[0], logical_packed.shape[1], -1
    )
    offsets = _unfold_humming_offsets(folded_offset)
    encoded = _e2m1_offsets_to_e4m3_bytes(codes, offsets).to(packed_storage_k.device)
    return encoded.view(torch.float8_e4m3fn), residual_x64.to(torch.float32)


def _quantize_full_hidden_reference(hidden_states):
    import torch

    fp32 = hidden_states.to(torch.float32)
    scale = (fp32.abs().amax(dim=1, keepdim=True) / E4M3_MAX).clamp_min(1.0e-30)
    return (fp32 / scale).to(torch.float8_e4m3fn), scale.to(torch.float32)


def _quantize_fc2_k64_reference(value):
    """Match the epilogue's one-reciprocal-then-multiply FP8 handoff."""
    import torch

    rows, columns = value.shape
    assert columns % FC2_SCALE_K == 0
    blocks = value.to(torch.float32).reshape(rows, columns // FC2_SCALE_K, 64)
    scale = (blocks.abs().amax(dim=-1) / E4M3_MAX).clamp_min(1.0e-30)
    reciprocal = torch.reciprocal(scale)
    quantized = (
        (blocks * reciprocal.unsqueeze(-1))
        .reshape(rows, columns)
        .to(torch.float8_e4m3fn)
    )
    return quantized, scale


def _hybrid_wgmma_reference(activation, activation_scale, weight, residual_x64):
    """FP8 WGMMA with independently promoted K64 activation-scale groups."""
    import torch

    rows, logical_k = activation.shape
    assert weight.shape[1] == logical_k
    assert logical_k % FC2_SCALE_K == 0
    assert activation_scale.shape == (rows, logical_k // FC2_SCALE_K)
    accumulator = torch.zeros(
        (rows, weight.shape[0]), dtype=torch.float32, device=activation.device
    )
    for k64 in range(logical_k // FC2_SCALE_K):
        columns = slice(k64 * FC2_SCALE_K, (k64 + 1) * FC2_SCALE_K)
        partial = (
            activation[:, columns].to(torch.float32)
            @ weight[:, columns].to(torch.float32).T
        )
        accumulator.add_(partial * activation_scale[:, k64].unsqueeze(1))
    return accumulator * residual_x64


def _swiglu_sm90_reference(gate, up):
    """Test-owned transcription of the Hopper FP32 operation order."""
    import torch

    neg_gate_log2e = gate * torch.tensor(
        -1.4426950408889634, dtype=torch.float32, device=gate.device
    )
    exp_neg = torch.exp2(neg_gate_log2e)
    sigmoid = torch.reciprocal(exp_neg + 1.0)
    return (up * gate) * sigmoid


def _reference_reduced(*, problem, symm_buffer, transformed_l1, transformed_l2):
    """Independent fused FC1/top-k/FC2/combine reference."""
    import torch

    n = problem["num_tokens"]
    intermediate = problem["intermediate"]
    fc1_weight, fc1_residual = _decode_processed_leg(transformed_l1)
    fc2_weight, fc2_residual = _decode_processed_leg(transformed_l2)
    terms = torch.zeros(
        (n, problem["topk"], problem["hidden"]),
        dtype=torch.bfloat16,
        device="cuda",
    )
    all_fc2_scales = []

    old_allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for expert in range(problem["experts"]):
            routed = (symm_buffer.topk_idx[:n] == expert).nonzero(as_tuple=False)
            assert routed.numel() > 0
            tokens = routed[:, 0]
            slots = routed[:, 1]

            activation = symm_buffer.x[tokens]
            logical_scale = symm_buffer.x_sf[tokens, 0]
            fc1_scale = logical_scale[:, None].expand(
                -1, problem["hidden"] // FC2_SCALE_K
            )
            fc1 = _hybrid_wgmma_reference(
                activation,
                fc1_scale,
                fc1_weight[expert],
                fc1_residual[expert],
            )

            pairs = fc1.shape[1] // (2 * GATE_UP_INTERLEAVE)
            gate_up = fc1.view(-1, pairs, 2, GATE_UP_INTERLEAVE)
            gate = gate_up[:, :, 0, :]
            up = gate_up[:, :, 1, :]
            limit = problem["gate_up_clamp"]
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
            swiglu = _swiglu_sm90_reference(gate, up).reshape(-1, intermediate)

            # The fused deepgemm graph applies top-k after SwiGLU but before
            # the lossy K64 E4M3 handoff.  Combine therefore performs a plain
            # sum and must not multiply these weights a second time.
            swiglu.mul_(
                symm_buffer.topk_weights[tokens, slots].to(torch.float32).unsqueeze(1)
            )
            fc2_activation, fc2_scale = _quantize_fc2_k64_reference(swiglu)
            all_fc2_scales.append(fc2_scale)
            fc2 = _hybrid_wgmma_reference(
                fc2_activation,
                fc2_scale,
                fc2_weight[expert],
                fc2_residual[expert],
            )
            terms[tokens, slots] = fc2.to(torch.bfloat16)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_allow_tf32

    scales = torch.cat(all_fc2_scales, dim=0)
    assert scales.shape[1] == intermediate // FC2_SCALE_K
    return terms.to(torch.float32).sum(dim=1), scales


@pytest.mark.arch_hopper
def test_sm90_mxfp4_kernel_matches_independent_reference(monkeypatch) -> None:
    """Raw MXFP4 ABI -> production preprocess -> fused public launch."""
    _require_hopper()

    import torch

    from flashinfer.moe_ep import PrequantizedMoEWeights
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
        preprocess_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )

    # Restored by pytest: never leak the single-rank no-NVSHMEM mode to other
    # collective tests in the process.
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    pkg = _sm90_tree()
    problem = _single_rank_problem()

    raw = PrequantizedMoEWeights(
        w13=problem["w13"],
        w2=problem["w2"],
        w13_scale=problem["w13_scale"],
        w2_scale=problem["w2_scale"],
    )
    transformed_l1, transformed_l2 = preprocess_mega_weights(
        raw,
        intermediate_size=problem["intermediate"],
        hidden_size=problem["hidden"],
    )

    symm_buffer = pkg.get_symm_buffer_for_hopper_mxfp4_mega_moe(
        problem["experts"],
        problem["max_tokens"],
        problem["topk"],
        problem["hidden"],
        problem["intermediate"],
        0,
        1,
        swap_ab=True,
        pingpong=False,
        mma_tiler_mnk=MMA_TILER_MNK,
        cluster_shape_mnk=(1, 1, 1),
        gate_up_clamp=problem["gate_up_clamp"],
    )
    try:
        stage_mega_moe_inputs(
            problem["hidden_states"],
            problem["topk_weights"],
            problem["topk_ids"],
            symm_buffer.x,
            symm_buffer.x_sf,
            symm_buffer.topk_idx,
            symm_buffer.topk_weights,
            quantize_input=True,
        )

        # Independently pin the production activation contract: one full-H
        # E4M3 scale per token, physically repeated into four FP32 lanes.
        expected_x, expected_scale = _quantize_full_hidden_reference(
            problem["hidden_states"]
        )
        n = problem["num_tokens"]
        assert torch.equal(
            symm_buffer.x[:n].contiguous().view(torch.uint8),
            expected_x.contiguous().view(torch.uint8),
        )
        torch.testing.assert_close(
            symm_buffer.x_sf[:n], expected_scale.expand(-1, 4), rtol=0.0, atol=0.0
        )
        assert torch.equal(
            symm_buffer.topk_idx[n:],
            torch.full_like(symm_buffer.topk_idx[n:], -1),
        )

        y_reference, fc2_scales = _reference_reduced(
            problem=problem,
            symm_buffer=symm_buffer,
            transformed_l1=transformed_l1,
            transformed_l2=transformed_l2,
        )
        assert torch.isfinite(fc2_scales).all()
        assert (fc2_scales > 0).all()

        y_first = torch.empty(
            (n, problem["hidden"]), dtype=torch.bfloat16, device="cuda"
        )
        pkg.hopper_mxfp4_mega_moe(
            y_first,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=n,
            gate_up_clamp=problem["gate_up_clamp"],
            sync=True,
        )
        compiled = symm_buffer._frontend._mega
        assert compiled is not None and compiled.compiled is not None

        # A second launch must reuse the same compiled object and produce the
        # same bytes.  This catches accidental FP8/MXFP4 cache-key crossover
        # or a steady-state workspace lifecycle regression.
        y_second = torch.empty_like(y_first)
        pkg.hopper_mxfp4_mega_moe(
            y_second,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=n,
            sync=True,
        )
        assert symm_buffer._frontend._mega is compiled
        assert torch.equal(y_second, y_first)

        # ``y=None`` is the supported zero-copy output view.  It must alias
        # the symmetric output allocation and retain the live-token shape.
        output_view = pkg.hopper_mxfp4_mega_moe(
            None,
            transformed_l1,
            transformed_l2,
            symm_buffer,
            num_tokens=n,
            sync=True,
        )
        assert output_view is not None
        assert output_view.shape == (n, problem["hidden"])
        assert output_view.data_ptr() == symm_buffer.output_activation.data_ptr()
        assert symm_buffer._frontend._mega is compiled
        assert torch.equal(output_view, y_first)

        actual = y_first.to(torch.float32)
        assert torch.isfinite(actual).all()
        relative_l2 = (actual - y_reference).norm() / y_reference.norm().clamp_min(
            1.0e-6
        )
        print(
            "[sm90 mxfp4 independent oracle] "
            f"rel_l2={relative_l2.item():.4g} "
            f"max|d|={(actual - y_reference).abs().max().item():.4g} "
            f"amax(ref)={y_reference.abs().max().item():.4g}"
        )
        torch.testing.assert_close(actual, y_reference, rtol=3.0e-2, atol=2.0e-2)
        assert relative_l2.item() < 0.035
    finally:
        symm_buffer.destroy()
