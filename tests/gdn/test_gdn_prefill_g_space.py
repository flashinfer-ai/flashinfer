"""
Tests for g_space parameter in chunk_gated_delta_rule.

The kernel always expects g in **linear** space, but callers like vLLM pass
g in ln-space (log(alpha)) for numerical stability.  The g_space parameter
lets callers declare their space and have the API convert transparently.

Tests:
1. Invalid g_space raises ValueError (hardware-independent)
2. g_space="ln" output matches g_space="linear" + manual exp() (requires SM90+)
3. g_space="log2" output matches g_space="linear" + manual exp2() (requires SM90+)
4. g_space="linear" (default) preserves backward-compatible behaviour (requires SM90+)
5. g=None with any g_space is a no-op (hardware-independent)
"""

import pytest
import torch

from flashinfer.gdn_prefill import chunk_gated_delta_rule
from flashinfer.utils import (
    is_sm90a_supported,
    is_sm100a_supported,
    is_sm120a_supported,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _skip_if_unsupported():
    device = torch.device("cuda")
    if is_sm100a_supported(device):
        cuda_major = int(torch.version.cuda.split(".")[0]) if torch.version.cuda else 0
        if cuda_major < 13:
            pytest.skip(
                f"SM100 GDN prefill requires CUDA 13+, got {torch.version.cuda}"
            )
    elif is_sm120a_supported(device) or is_sm90a_supported(device):
        pass
    else:
        pytest.skip("GDN prefill requires SM90, SM100, or SM120")


def _make_inputs(seq_len=128, num_heads=4, head_size=128, device="cuda"):
    """Minimal synthetic inputs for chunk_gated_delta_rule."""
    torch.manual_seed(42)
    q = torch.randn(seq_len, num_heads, head_size, dtype=torch.bfloat16, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(seq_len, num_heads, head_size, dtype=torch.bfloat16, device=device),
        p=2.0,
        dim=-1,
    )
    v = torch.randn(seq_len, num_heads, head_size, dtype=torch.bfloat16, device=device)
    # g in linear space: values in (0, 1] to keep cumulative product stable
    g_linear = torch.rand(seq_len, num_heads, dtype=torch.float32, device=device).clamp(
        0.1, 1.0
    )
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int64, device=device)
    return q, k, v, g_linear, cu_seqlens


# ---------------------------------------------------------------------------
# Hardware-independent tests (validation only)
# ---------------------------------------------------------------------------


def test_invalid_g_space_raises():
    """g_space must be 'linear', 'ln', or 'log2' — anything else raises ValueError."""
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)

    with pytest.raises(ValueError, match="g_space must be one of"):
        chunk_gated_delta_rule(
            q,
            k,
            v,
            g=g_linear,
            cu_seqlens=cu_seqlens,
            g_space="exp",  # invalid
        )


def test_invalid_g_space_typo():
    """Typos like 'Log2' or 'LN' should also raise ValueError."""
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)

    for bad in ("Log2", "LN", "natural", ""):
        with pytest.raises(ValueError, match="g_space must be one of"):
            chunk_gated_delta_rule(
                q,
                k,
                v,
                g=g_linear,
                cu_seqlens=cu_seqlens,
                g_space=bad,
            )


# ---------------------------------------------------------------------------
# Hardware-dependent equivalence tests
# ---------------------------------------------------------------------------


def test_g_none_any_g_space_no_error():
    """When g=None the all-ones gate is used; g_space should be validated but not applied."""
    device = torch.device("cuda")
    _skip_if_unsupported()
    q, k, v, _, cu_seqlens = _make_inputs(device=device)

    # All valid g_space values should work without error when g=None
    for space in ("linear", "ln", "log2"):
        out = chunk_gated_delta_rule(
            q, k, v, g=None, cu_seqlens=cu_seqlens, g_space=space
        )
        assert not out.isnan().any(), f"NaN output with g=None, g_space={space!r}"


def test_g_space_ln_matches_manual_exp():
    """g_space='ln' should give the same output as torch.exp(g) + g_space='linear'."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)
    g_ln = torch.log(g_linear)  # convert to ln-space for the caller

    # Reference: caller converts g manually before passing
    out_ref = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=torch.exp(g_ln),  # manual conversion → linear space
        cu_seqlens=cu_seqlens,
        g_space="linear",
    )

    # With g_space="ln": API should do the exp() internally
    out_api = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g_ln,
        cu_seqlens=cu_seqlens,
        g_space="ln",
    )

    assert not out_ref.isnan().any(), "Reference output contains NaN"
    assert not out_api.isnan().any(), "g_space='ln' output contains NaN"
    torch.testing.assert_close(
        out_api,
        out_ref,
        rtol=0,
        atol=0,
        msg="g_space='ln' did not match manual exp() conversion",
    )


def test_g_space_log2_matches_manual_exp2():
    """g_space='log2' should give the same output as torch.exp2(g) + g_space='linear'."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)
    g_log2 = torch.log2(g_linear)

    out_ref = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=torch.exp2(g_log2),  # manual conversion → linear space
        cu_seqlens=cu_seqlens,
        g_space="linear",
    )

    out_api = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g_log2,
        cu_seqlens=cu_seqlens,
        g_space="log2",
    )

    assert not out_ref.isnan().any(), "Reference output contains NaN"
    assert not out_api.isnan().any(), "g_space='log2' output contains NaN"
    torch.testing.assert_close(
        out_api,
        out_ref,
        rtol=0,
        atol=0,
        msg="g_space='log2' did not match manual exp2() conversion",
    )


def test_g_space_linear_default_unchanged():
    """g_space='linear' (default) must not change existing behaviour."""
    _skip_if_unsupported()
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)

    # Explicit g_space="linear"
    out_explicit = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g_linear,
        cu_seqlens=cu_seqlens,
        g_space="linear",
    )

    # Default (no g_space argument) — must be identical
    out_default = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g_linear,
        cu_seqlens=cu_seqlens,
    )

    assert not out_explicit.isnan().any(), "g_space='linear' output contains NaN"
    torch.testing.assert_close(
        out_explicit,
        out_default,
        rtol=0,
        atol=0,
        msg="Explicit g_space='linear' differs from default",
    )


def test_g_space_ln_fixes_vllm_nan():
    """
    Regression test: vLLM passes g in ln-space (output of torch.nn.functional.logsigmoid or
    similar), which caused all-NaN output when g_space was not declared.
    With g_space='ln', the output must be finite.
    """
    _skip_if_unsupported()
    device = torch.device("cuda")
    q, k, v, g_linear, cu_seqlens = _make_inputs(device=device)

    # Simulate what vLLM does: pass log(sigmoid(raw_gate))
    raw_gate = torch.randn_like(g_linear)
    g_ln = torch.nn.functional.logsigmoid(raw_gate)  # ln-space, values ≤ 0

    # Without g_space fix: passing g_ln as-is (linear) would produce NaN
    # because the kernel sees values like -3.0 as a linear gate and
    # the resulting cumulative product underflows/overflows to 0 or inf.

    # With g_space='ln': should produce finite output
    out = chunk_gated_delta_rule(
        q,
        k,
        v,
        g=g_ln,
        cu_seqlens=cu_seqlens,
        g_space="ln",
    )
    assert out.isfinite().all(), (
        "chunk_gated_delta_rule with g_space='ln' still produces non-finite output"
    )


if __name__ == "__main__":
    test_invalid_g_space_raises()
    test_invalid_g_space_typo()
    print("Validation tests passed.")
    test_g_space_ln_matches_manual_exp()
    test_g_space_log2_matches_manual_exp2()
    test_g_space_linear_default_unchanged()
    test_g_space_ln_fixes_vllm_nan()
    print("All tests passed.")
