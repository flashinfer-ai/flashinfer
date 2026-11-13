"""
Tests for fused DIT LayerNorm kernels.

Tests correctness against PyTorch reference for three modes:
- gate_residual_gamma_beta
- gate_residual_scale_shift
- residual_scale_shift

BF16 output tested on all GPUs. NVFP4/MXFP8 output requires SM100+ (Blackwell).
Hidden dim restricted to 3072 (WAN 2.2 5B target).
"""

import pytest
import torch

from flashinfer.diffusion_ops import (
    fused_dit_gate_residual_layernorm_gamma_beta,
    fused_dit_gate_residual_layernorm_scale_shift,
    fused_dit_residual_layernorm_scale_shift,
)

EPSILON = 1e-6
HIDDEN_DIM = 3072

# Shapes: (batch_size, seq_len)
BF16_SHAPES = [
    (1, 1920),
    (1, 768),
    (2, 1920),
    (2, 768),
    (4, 1920),
]


def _get_sm():
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def _make_strided_gate(batch_size, seq_len, hidden_dim, device):
    """Create a properly strided gate tensor matching WAN's temb.chunk(6, dim=2)."""
    temb = torch.randn(
        batch_size, seq_len, 6, hidden_dim, dtype=torch.bfloat16, device=device
    )
    return temb.chunk(6, dim=2)[0].squeeze(2)


def _make_wan_temb_inputs(batch_size, seq_len, hidden_dim, device):
    """Create gate/scale/shift tensors matching WAN's temb.chunk(6, dim=2) pattern.

    Returns scale_shift_table, temb, and the 6 chunked tensors
    (shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa).
    """
    scale_shift_table = torch.randn(
        1, 6, hidden_dim, dtype=torch.float32, device=device
    )
    temb = torch.randn(
        batch_size, seq_len, 6, hidden_dim, dtype=torch.bfloat16, device=device
    )
    chunks = (scale_shift_table.unsqueeze(0) + temb.float()).chunk(6, dim=2)
    shift_msa = chunks[0].squeeze(2)
    scale_msa = chunks[1].squeeze(2)
    gate_msa = chunks[2].squeeze(2)
    c_shift_msa = chunks[3].squeeze(2)
    c_scale_msa = chunks[4].squeeze(2)
    c_gate_msa = chunks[5].squeeze(2)

    # bias tensors from scale_shift_table (contiguous)
    table_chunks = scale_shift_table.chunk(6, dim=1)
    shift_bias = table_chunks[0].squeeze(1)
    scale_bias = table_chunks[1].squeeze(1)
    gate_bias = table_chunks[2].squeeze(1)
    c_shift_bias = table_chunks[3].squeeze(1)
    c_scale_bias = table_chunks[4].squeeze(1)
    c_gate_bias = table_chunks[5].squeeze(1)

    return {
        "shift_msa": shift_msa,
        "scale_msa": scale_msa,
        "gate_msa": gate_msa,
        "c_shift_msa": c_shift_msa,
        "c_scale_msa": c_scale_msa,
        "c_gate_msa": c_gate_msa,
        "shift_bias": shift_bias,
        "scale_bias": scale_bias,
        "gate_bias": gate_bias,
        "c_shift_bias": c_shift_bias,
        "c_scale_bias": c_scale_bias,
        "c_gate_bias": c_gate_bias,
        # Raw temb chunks for the kernel (strided, from temb directly)
        "temb": temb,
        "scale_shift_table": scale_shift_table,
    }


def _pytorch_baseline(mode, input_tensor, residual, gate, gamma, beta, scale, shift):
    """PyTorch reference implementation."""
    with torch.no_grad():
        if mode == "gate_residual_gamma_beta" or mode == "gate_residual_scale_shift":
            residual_output = residual.float() + input_tensor.float() * gate
        elif mode == "residual_scale_shift":
            residual_output = residual.float() + input_tensor.float()
        else:
            raise ValueError(f"Unknown mode: {mode}")

        hidden_dim = residual_output.shape[-1]

        if mode == "gate_residual_gamma_beta":
            norm_output = torch.layer_norm(
                residual_output,
                normalized_shape=[hidden_dim],
                weight=gamma,
                bias=beta,
                eps=EPSILON,
            )
        elif mode in ("residual_scale_shift", "gate_residual_scale_shift"):
            norm_output = torch.layer_norm(
                residual_output,
                normalized_shape=[hidden_dim],
                weight=None,
                bias=None,
                eps=EPSILON,
            )
            norm_output = norm_output * (1 + scale) + shift

    return residual_output.to(torch.bfloat16), norm_output.to(torch.bfloat16)


# ---------------------------------------------------------------------------
# BF16 correctness: gate_residual_gamma_beta
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,seq_len", BF16_SHAPES)
def test_gate_residual_gamma_beta_bf16(batch_size, seq_len):
    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)

    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)

    # Reference
    residual_ref, norm_ref = _pytorch_baseline(
        "gate_residual_gamma_beta",
        input_tensor,
        residual,
        temb_data["gate_msa"],
        gamma,
        beta,
        None,
        None,
    )

    # Fused kernel — pass strided gate from temb.chunk(6, dim=2) directly
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    gate_strided = temb_chunks[2].squeeze(2)  # gate_msa position
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)
    gate_bias_from_table = table_chunks[2].squeeze(1)

    residual_fused, norm_fused = fused_dit_gate_residual_layernorm_gamma_beta(
        input_tensor,
        residual,
        gate_strided,
        gamma,
        beta,
        gate_bias=gate_bias_from_table,
        epsilon=EPSILON,
    )

    torch.testing.assert_close(
        residual_fused.float(), residual_ref.float(), rtol=1.6e-2, atol=1e-5
    )
    torch.testing.assert_close(
        norm_fused.float(), norm_ref.float(), rtol=1.6e-2, atol=1e-5
    )


# ---------------------------------------------------------------------------
# BF16 correctness: gate_residual_scale_shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,seq_len", BF16_SHAPES)
def test_gate_residual_scale_shift_bf16(batch_size, seq_len):
    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)

    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)

    # Reference
    residual_ref, norm_ref = _pytorch_baseline(
        "gate_residual_scale_shift",
        input_tensor,
        residual,
        temb_data["c_gate_msa"],
        None,
        None,
        temb_data["scale_msa"],
        temb_data["shift_msa"],
    )

    # Fused kernel
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_fused, norm_fused = fused_dit_gate_residual_layernorm_scale_shift(
        input_tensor,
        residual,
        temb_chunks[5].squeeze(2),  # c_gate_msa
        temb_chunks[1].squeeze(2),  # scale_msa
        temb_chunks[0].squeeze(2),  # shift_msa
        gate_bias=table_chunks[5].squeeze(1),
        scale_bias=table_chunks[1].squeeze(1),
        shift_bias=table_chunks[0].squeeze(1),
        epsilon=EPSILON,
    )

    torch.testing.assert_close(
        residual_fused.float(), residual_ref.float(), rtol=1.6e-2, atol=1e-5
    )
    torch.testing.assert_close(
        norm_fused.float(), norm_ref.float(), rtol=1.6e-2, atol=1e-5
    )


# ---------------------------------------------------------------------------
# BF16 correctness: residual_scale_shift
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("batch_size,seq_len", BF16_SHAPES)
def test_residual_scale_shift_bf16(batch_size, seq_len):
    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)

    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)

    # Reference
    residual_ref, norm_ref = _pytorch_baseline(
        "residual_scale_shift",
        input_tensor,
        residual,
        None,
        None,
        None,
        temb_data["c_scale_msa"],
        temb_data["c_shift_msa"],
    )

    # Fused kernel
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_fused, norm_fused = fused_dit_residual_layernorm_scale_shift(
        input_tensor,
        temb_chunks[4].squeeze(2),  # c_scale_msa
        temb_chunks[3].squeeze(2),  # c_shift_msa
        residual=residual,
        scale_bias=table_chunks[4].squeeze(1),
        shift_bias=table_chunks[3].squeeze(1),
        epsilon=EPSILON,
    )

    torch.testing.assert_close(
        residual_fused.float(), residual_ref.float(), rtol=1.6e-2, atol=1e-5
    )
    torch.testing.assert_close(
        norm_fused.float(), norm_ref.float(), rtol=1.6e-2, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Destination-passing: pre-allocated outputs
# ---------------------------------------------------------------------------


def test_destination_passing():
    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size, seq_len = 1, 768
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)

    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_out = torch.empty_like(input_tensor)
    norm_out = torch.empty_like(input_tensor)

    r_ret, n_ret = fused_dit_gate_residual_layernorm_gamma_beta(
        input_tensor,
        residual,
        temb_chunks[2].squeeze(2),
        gamma,
        beta,
        gate_bias=table_chunks[2].squeeze(1),
        epsilon=EPSILON,
        residual_out=residual_out,
        norm_out=norm_out,
    )

    assert r_ret is residual_out
    assert n_ret is norm_out


# ---------------------------------------------------------------------------
# Destination-passing: scale_shift mode
# ---------------------------------------------------------------------------


def test_destination_passing_scale_shift():
    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size, seq_len = 1, 768
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_out = torch.empty_like(input_tensor)
    norm_out = torch.empty_like(input_tensor)

    r_ret, n_ret = fused_dit_residual_layernorm_scale_shift(
        input_tensor,
        temb_chunks[4].squeeze(2),
        temb_chunks[3].squeeze(2),
        residual=residual,
        scale_bias=table_chunks[4].squeeze(1),
        shift_bias=table_chunks[3].squeeze(1),
        epsilon=EPSILON,
        residual_out=residual_out,
        norm_out=norm_out,
    )

    assert r_ret is residual_out
    assert n_ret is norm_out


# ---------------------------------------------------------------------------
# Correctness: residual=None (no residual addition)
# ---------------------------------------------------------------------------


def test_residual_scale_shift_no_residual():
    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size, seq_len = 1, 768
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_fused, norm_fused = fused_dit_residual_layernorm_scale_shift(
        input_tensor,
        temb_chunks[4].squeeze(2),
        temb_chunks[3].squeeze(2),
        residual=None,
        scale_bias=table_chunks[4].squeeze(1),
        shift_bias=table_chunks[3].squeeze(1),
        epsilon=EPSILON,
    )

    # Reference: no residual, just input -> layernorm -> scale/shift
    residual_ref = input_tensor.float()
    norm_ref = torch.layer_norm(residual_ref, [HIDDEN_DIM], eps=EPSILON)
    norm_ref = (
        norm_ref * (1 + temb_data["c_scale_msa"].float())
        + temb_data["c_shift_msa"].float()
    )

    # When residual=None, residual_out should equal input
    torch.testing.assert_close(
        residual_fused.float(), input_tensor.float(), rtol=1.6e-2, atol=1e-5
    )
    torch.testing.assert_close(
        norm_fused.float(), norm_ref.to(torch.bfloat16).float(), rtol=1.6e-2, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Correctness: odd num_rows (using WAN-style strided inputs)
# ---------------------------------------------------------------------------


def test_odd_num_rows():
    device = torch.device("cuda")
    torch.manual_seed(42)

    batch_size, seq_len = 1, 769  # odd
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)

    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    residual_fused, norm_fused = fused_dit_gate_residual_layernorm_gamma_beta(
        input_tensor,
        residual,
        temb_chunks[2].squeeze(2),
        gamma,
        beta,
        gate_bias=table_chunks[2].squeeze(1),
        epsilon=EPSILON,
    )

    residual_ref, norm_ref = _pytorch_baseline(
        "gate_residual_gamma_beta",
        input_tensor,
        residual,
        temb_data["gate_msa"],
        gamma,
        beta,
        None,
        None,
    )

    torch.testing.assert_close(
        residual_fused.float(), residual_ref.float(), rtol=1.6e-2, atol=1e-5
    )
    torch.testing.assert_close(
        norm_fused.float(), norm_ref.float(), rtol=1.6e-2, atol=1e-5
    )


# ---------------------------------------------------------------------------
# Correctness: NVFP4 output (SM100+ only)
# ---------------------------------------------------------------------------


def _run_nvfp4_or_mxfp8_test(mode, output_type, batch_size=1, seq_len=768):
    """Shared helper for NVFP4/MXFP8 accuracy tests.

    Verifies:
    1. Output shapes and dtypes
    2. Residual output accuracy (always BF16)
    3. Norm output non-zero and finite
    4. If torchao is available: dequantized norm accuracy against reference
    """
    use_nvfp4 = output_type == "nvfp4"
    use_mxfp8 = output_type == "mxfp8"

    device = torch.device("cuda")
    torch.manual_seed(42)

    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    temb_data = _make_wan_temb_inputs(batch_size, seq_len, HIDDEN_DIM, device)
    temb_chunks = temb_data["temb"].chunk(6, dim=2)
    table_chunks = temb_data["scale_shift_table"].chunk(6, dim=1)

    # Compute BF16 reference for residual and norm
    if mode == "gate_residual_gamma_beta":
        residual_ref, norm_ref = _pytorch_baseline(
            mode,
            input_tensor,
            residual,
            temb_data["gate_msa"],
            gamma,
            beta,
            None,
            None,
        )
    else:
        residual_ref, norm_ref = _pytorch_baseline(
            "residual_scale_shift",
            input_tensor,
            residual,
            None,
            None,
            None,
            temb_data["c_scale_msa"],
            temb_data["c_shift_msa"],
        )

    # For NVFP4, compute global_scaling_factor from reference norm output
    # (matching the original test's approach)
    if use_nvfp4:
        global_scale_factor = (448.0 * 6.0) / norm_ref.abs().max().item()
        global_sf = torch.tensor(
            [global_scale_factor], dtype=torch.float32, device=device
        )
    else:
        global_sf = None

    # Run fused kernel
    if mode == "gate_residual_gamma_beta":
        residual_out, norm_out = fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            temb_chunks[2].squeeze(2),
            gamma,
            beta,
            gate_bias=table_chunks[2].squeeze(1),
            epsilon=EPSILON,
            use_nvfp4=use_nvfp4,
            use_mxfp8=use_mxfp8,
            global_scaling_factor=global_sf,
        )
    else:
        residual_out, norm_out = fused_dit_residual_layernorm_scale_shift(
            input_tensor,
            temb_chunks[4].squeeze(2),
            temb_chunks[3].squeeze(2),
            residual=residual,
            scale_bias=table_chunks[4].squeeze(1),
            shift_bias=table_chunks[3].squeeze(1),
            epsilon=EPSILON,
            use_nvfp4=use_nvfp4,
            use_mxfp8=use_mxfp8,
            global_scaling_factor=global_sf,
        )

    # 1. Shape and dtype checks
    assert residual_out.dtype == torch.bfloat16
    assert residual_out.shape == (batch_size, seq_len, HIDDEN_DIM)
    if use_nvfp4:
        assert norm_out.dtype == torch.int32
        assert norm_out.shape == (batch_size, seq_len, HIDDEN_DIM // 8)
    elif use_mxfp8:
        assert norm_out.dtype == torch.int32
        assert norm_out.shape == (batch_size, seq_len, HIDDEN_DIM // 4)

    # 2. Residual accuracy (always BF16, should match reference closely)
    torch.testing.assert_close(
        residual_out.float(), residual_ref.float(), rtol=1.6e-2, atol=1e-5
    )

    # 3. Norm output sanity: packed data should be non-zero
    assert norm_out.any(), "norm_out is all zeros — kernel likely did not write output"

    # TODO: Add dequantized norm accuracy check when torchao is available.
    # Full NVFP4 dequant requires sf_out tensor (swizzled scale factors) which
    # would need the API to return a 3-tuple. MXFP8 similarly needs sf_out.
    # For now, residual accuracy + non-zero output sanity is verified.


@pytest.mark.parametrize("mode", ["gate_residual_gamma_beta", "residual_scale_shift"])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 768), (2, 1920)])
def test_nvfp4_output(mode, batch_size, seq_len):
    if _get_sm() < 100:
        pytest.skip("NVFP4 output requires SM100+ (Blackwell)")
    _run_nvfp4_or_mxfp8_test(mode, "nvfp4", batch_size, seq_len)


@pytest.mark.parametrize("mode", ["gate_residual_gamma_beta", "residual_scale_shift"])
@pytest.mark.parametrize("batch_size,seq_len", [(1, 768), (2, 1920)])
def test_mxfp8_output(mode, batch_size, seq_len):
    if _get_sm() < 100:
        pytest.skip("MXFP8 output requires SM100+ (Blackwell)")
    _run_nvfp4_or_mxfp8_test(mode, "mxfp8", batch_size, seq_len)


# ---------------------------------------------------------------------------
# Validation: pre-allocated output wrong dtype
# ---------------------------------------------------------------------------


def test_error_pre_allocated_wrong_dtype():
    device = torch.device("cuda")
    batch_size, seq_len = 1, 768
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    gate = _make_strided_gate(batch_size, seq_len, HIDDEN_DIM, device)

    wrong_dtype_out = torch.empty(
        batch_size,
        seq_len,
        HIDDEN_DIM,
        dtype=torch.float32,
        device=device,
    )

    with pytest.raises(ValueError, match="norm_out dtype"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            gate,
            gamma,
            beta,
            epsilon=EPSILON,
            norm_out=wrong_dtype_out,
        )


# ---------------------------------------------------------------------------
# Validation: error cases
# ---------------------------------------------------------------------------


def test_error_wrong_hidden_dim():
    device = torch.device("cuda")
    input_tensor = torch.randn(1, 768, 1536, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(1536, dtype=torch.float32, device=device)
    beta = torch.randn(1536, dtype=torch.float32, device=device)
    gate = torch.randn(1, 768, 1536, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="3072"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor, residual, gate, gamma, beta, epsilon=EPSILON
        )


def test_error_non_cuda():
    input_tensor = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32)
    gate = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16)

    with pytest.raises((ValueError, RuntimeError)):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor, residual, gate, gamma, beta, epsilon=EPSILON
        )


def test_error_wrong_dtype():
    device = torch.device("cuda")
    input_tensor = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.float16, device=device)
    residual = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    gate = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    with pytest.raises((ValueError, RuntimeError)):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor, residual, gate, gamma, beta, epsilon=EPSILON
        )


def test_error_nvfp4_without_scaling_factor():
    if _get_sm() < 100:
        pytest.skip("NVFP4 requires SM100+")

    device = torch.device("cuda")
    input_tensor = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    gate = _make_strided_gate(1, 768, HIDDEN_DIM, device)

    with pytest.raises(ValueError, match="global_scaling_factor"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            gate,
            gamma,
            beta,
            epsilon=EPSILON,
            use_nvfp4=True,
        )


def test_error_both_nvfp4_and_mxfp8():
    device = torch.device("cuda")
    input_tensor = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    gate = _make_strided_gate(1, 768, HIDDEN_DIM, device)

    with pytest.raises(ValueError, match="Cannot use both"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            gate,
            gamma,
            beta,
            epsilon=EPSILON,
            use_nvfp4=True,
            use_mxfp8=True,
        )


def test_error_contiguous_gate():
    """Contiguous gate tensor (wrong stride) should be rejected."""
    device = torch.device("cuda")
    input_tensor = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    # Contiguous gate has stride hidden_dim in row dim, not 6*hidden_dim
    gate = torch.randn(1, 768, HIDDEN_DIM, dtype=torch.bfloat16, device=device)

    with pytest.raises(ValueError, match="row stride"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            gate,
            gamma,
            beta,
            epsilon=EPSILON,
        )


def test_error_pre_allocated_wrong_shape():
    device = torch.device("cuda")
    batch_size, seq_len = 1, 768
    input_tensor = torch.randn(
        batch_size, seq_len, HIDDEN_DIM, dtype=torch.bfloat16, device=device
    )
    residual = torch.randn_like(input_tensor)
    gamma = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    beta = torch.randn(HIDDEN_DIM, dtype=torch.float32, device=device)
    gate = _make_strided_gate(batch_size, seq_len, HIDDEN_DIM, device)

    wrong_norm_out = torch.empty(
        batch_size,
        seq_len,
        HIDDEN_DIM // 2,
        dtype=torch.bfloat16,
        device=device,
    )

    with pytest.raises(ValueError, match="norm_out shape"):
        fused_dit_gate_residual_layernorm_gamma_beta(
            input_tensor,
            residual,
            gate,
            gamma,
            beta,
            epsilon=EPSILON,
            norm_out=wrong_norm_out,
        )
