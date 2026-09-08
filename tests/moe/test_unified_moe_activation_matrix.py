"""CPU-only consistency checks for the generated Unified MoE activation matrix."""

import pytest

from flashinfer.fused_moe.api import QuantVariant, SwiGLU
from scripts import generate_moe_activation_matrix as matrix
from scripts.generate_moe_activation_matrix import (
    check_activation_matrix,
    get_activation_matrix_rows,
)


def test_activation_matrix_rows_are_unique_and_complete():
    rows = get_activation_matrix_rows()
    keys = [(backend_key, variant) for backend_key, _, variant, _ in rows]

    assert rows
    assert len(keys) == len(set(keys))
    assert all(activations for _, _, _, activations in rows)


def test_documented_activation_matrix_matches_runner_registry():
    check_activation_matrix()


def test_quant_specific_activation_mapping_must_cover_exact_variants(monkeypatch):
    class Config:
        pass

    class Runner:
        backend_key = "incomplete"
        supported_quant_variants = (QuantVariant.BF16, QuantVariant.NVFP4)
        supported_activation_classes_by_quant = {
            QuantVariant.BF16: (SwiGLU,),
        }

    monkeypatch.setattr(matrix, "_BACKEND_RUNNERS", {Config: Runner})
    with pytest.raises(ValueError, match="must cover exactly"):
        get_activation_matrix_rows()
