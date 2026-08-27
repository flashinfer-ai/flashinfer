"""CPU-only consistency checks for Unified MoE capability metadata."""

from pathlib import Path

from flashinfer.fused_moe.capabilities import (
    get_moe_backend_capabilities,
    render_moe_activation_matrix,
)


def test_registered_capability_records_are_unique_and_complete():
    rows = get_moe_backend_capabilities()
    keys = [(row.backend_key, row.quant_variant) for row in rows]

    assert rows
    assert len(keys) == len(set(keys))
    assert all(row.activation_classes for row in rows)
    assert all(row.routing_modes for row in rows)


def test_documented_activation_matrix_matches_runner_registry():
    doc_path = (
        Path(__file__).resolve().parents[2]
        / "docs"
        / "design_docs"
        / "flashinfer_moe_api.md"
    )
    text = doc_path.read_text()
    begin = "<!-- BEGIN GENERATED MOE ACTIVATION MATRIX -->"
    end = "<!-- END GENERATED MOE ACTIVATION MATRIX -->"

    documented = text.split(begin, 1)[1].split(end, 1)[0].strip()
    assert documented == render_moe_activation_matrix()
