#!/usr/bin/env python3
"""Check or update the generated Unified MoE activation matrix."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flashinfer.fused_moe.api import ActivationConfig, QuantFormat
from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

DEFAULT_DOC_PATH = REPO_ROOT / "docs" / "design_docs" / "flashinfer_moe_api.md"
BEGIN_MARKER = "<!-- BEGIN GENERATED MOE ACTIVATION MATRIX -->"
END_MARKER = "<!-- END GENERATED MOE ACTIVATION MATRIX -->"
REGENERATE_COMMAND = "python scripts/generate_moe_activation_matrix.py --write"

ActivationMatrixRow = tuple[
    str,
    str,
    str,
    tuple[type[ActivationConfig], ...],
]


def _quant_label(pair: tuple[QuantFormat, QuantFormat]) -> str:
    weight, activation = pair
    return f"{weight.name}×{activation.name}"


def get_activation_matrix_rows() -> tuple[ActivationMatrixRow, ...]:
    """Collect and validate backend × quantization × activation rows."""
    rows = []
    for config_type, runner_type in _BACKEND_RUNNERS.items():
        pairs = runner_type.supported_quant_variants
        by_quant = runner_type.supported_activation_classes_by_quant
        if by_quant and set(by_quant) != set(pairs):
            expected = tuple(f"{w.name}×{a.name}" for w, a in pairs)
            raise ValueError(
                f"{runner_type.__name__} activation mapping must cover exactly "
                f"{expected}."
            )

        for pair in pairs:
            activations = (
                by_quant[pair] if by_quant else runner_type.supported_activation_classes
            )
            label = _quant_label(pair)
            if not activations:
                raise ValueError(
                    f"{runner_type.__name__} declares no activations for {label}."
                )
            rows.append(
                (
                    runner_type.backend_key,
                    config_type.__name__,
                    label,
                    activations,
                )
            )
    return tuple(sorted(rows, key=lambda row: (row[0], row[2])))


def render_activation_matrix(
    rows: tuple[ActivationMatrixRow, ...] | None = None,
) -> str:
    """Render the backend activation rows as a Markdown table."""
    rows = get_activation_matrix_rows() if rows is None else rows
    lines = [
        "| Backend | Config | Quantization | Typed activations |",
        "| --- | --- | --- | --- |",
    ]
    for backend_key, config_name, label, activation_classes in rows:
        activations = ", ".join(
            f"`{activation.__name__}`" for activation in activation_classes
        )
        lines.append(
            f"| `{backend_key}` | `{config_name}` | `{label}` | {activations} |"
        )
    return "\n".join(lines)


def _replace_documented_matrix(text: str, rendered: str) -> str:
    if text.count(BEGIN_MARKER) != 1 or text.count(END_MARKER) != 1:
        raise ValueError("activation matrix markers must each appear exactly once")
    before, remainder = text.split(BEGIN_MARKER, 1)
    _, after = remainder.split(END_MARKER, 1)
    return f"{before}{BEGIN_MARKER}\n{rendered}\n{END_MARKER}{after}"


def check_activation_matrix(doc_path: Path = DEFAULT_DOC_PATH) -> None:
    """Raise when the documented matrix differs from runner declarations."""
    text = doc_path.read_text()
    if text != _replace_documented_matrix(text, render_activation_matrix()):
        raise AssertionError(
            f"{doc_path.relative_to(REPO_ROOT)} is stale; run: {REGENERATE_COMMAND}"
        )


def write_activation_matrix(doc_path: Path = DEFAULT_DOC_PATH) -> bool:
    """Update the documented matrix and return whether the file changed."""
    text = doc_path.read_text()
    updated = _replace_documented_matrix(text, render_activation_matrix())
    if updated == text:
        return False
    doc_path.write_text(updated)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check", action="store_true", help="fail if the matrix is stale"
    )
    mode.add_argument("--write", action="store_true", help="update the matrix in place")
    args = parser.parse_args()

    if args.check:
        check_activation_matrix()
        print("Unified MoE activation matrix is up to date.")
    else:
        changed = write_activation_matrix()
        print("Updated Unified MoE activation matrix." if changed else "No changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
