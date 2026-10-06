from __future__ import annotations

import pytest
import torch

from b12x.moe._shared.kernels.activations import (
    SITU,
    SITU_DEFAULT_BETA,
    SITU_DEFAULT_LINEAR_BETA,
    is_gated_moe_activation,
    moe_activation_w1_rows,
    normalize_moe_activation,
    normalize_swiglu_alpha_for_activation,
    normalize_swiglu_beta_for_activation,
    normalize_swiglu_limit_for_activation,
)
from b12x.moe._shared.kernels.reference import _apply_gated_activation


def test_situ_activation_contract() -> None:
    assert normalize_moe_activation(SITU) == SITU
    assert is_gated_moe_activation(SITU)
    assert moe_activation_w1_rows(SITU, 128) == 256
    assert normalize_swiglu_limit_for_activation(SITU, None) is None
    assert normalize_swiglu_alpha_for_activation(SITU, None) == 1.0
    assert normalize_swiglu_beta_for_activation(SITU, None) == 0.0


def test_situ_rejects_swiglu_parameters() -> None:
    with pytest.raises(ValueError, match="not part of the situ"):
        normalize_swiglu_limit_for_activation(SITU, 7.0)
    with pytest.raises(ValueError, match="only configurable"):
        normalize_swiglu_alpha_for_activation(SITU, 1.702)
    with pytest.raises(ValueError, match="only configurable"):
        normalize_swiglu_beta_for_activation(SITU, 1.0)


def test_situ_reference_matches_kimi_k3_beta_scaled_contract() -> None:
    gate = torch.tensor(
        [-100.0, -2.0, 0.0, 2.0, 100.0],
        dtype=torch.float32,
    )
    up = torch.tensor(
        [100.0, -2.0, 0.0, 2.0, -100.0],
        dtype=torch.float32,
    )
    actual = _apply_gated_activation(
        gate,
        up,
        activation=SITU,
        swiglu_limit=None,
        swiglu_alpha=1.0,
        swiglu_beta=0.0,
    )
    situ_gate = (
        SITU_DEFAULT_BETA
        * torch.tanh(gate / SITU_DEFAULT_BETA)
        * torch.sigmoid(gate)
    )
    situ_up = SITU_DEFAULT_LINEAR_BETA * torch.tanh(
        up / SITU_DEFAULT_LINEAR_BETA
    )
    expected = situ_gate * situ_up
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    assert torch.isfinite(actual).all()
