"""CPU-only consistency checks for the generated Unified MoE activation matrix."""

import os
import subprocess
import sys

import pytest

from flashinfer.fused_moe.api import QuantFormat, SwiGLU
from scripts import generate_moe_activation_matrix as matrix
from scripts.generate_moe_activation_matrix import (
    check_activation_matrix,
    get_activation_matrix_rows,
)


@pytest.mark.cpu_only
def test_activation_matrix_rows_are_unique_and_complete():
    rows = get_activation_matrix_rows()
    keys = [(backend_key, label) for backend_key, _, label, _ in rows]
    labels = {label for _, _, label, _ in rows}

    assert rows
    assert len(keys) == len(set(keys))
    assert all(activations for _, _, _, activations in rows)
    # Every cell is weight×activation; the two W4A16 encodings stay distinct.
    assert "MXFP4×BF16" in labels
    assert "NVFP4×BF16" in labels
    assert "NVFP4×NVFP4" in labels
    assert "NVFP4" not in labels


@pytest.mark.cpu_only
def test_documented_activation_matrix_matches_runner_registry():
    check_activation_matrix()


@pytest.mark.cpu_only
def test_activation_matrix_round_trip_ignores_locale_encoding(tmp_path):
    """The design doc is UTF-8. A non-UTF-8 locale must still read and rewrite it.

    Native Windows uses cp1252 for ``Path.read_text()`` (NVBug 7058649). An
    ASCII locale rejects the same bytes.
    """
    source = matrix.DEFAULT_DOC_PATH.read_bytes()
    assert any(byte > 127 for byte in source)
    # Corrupt the rendered header, which is inside the generated block. A
    # backend name would make this encoding test fail when that backend goes
    # away.
    header = matrix.render_activation_matrix().splitlines()[0].encode("utf-8")
    assert header in source, "rendered matrix header is not in the design doc"
    stale = source.replace(header, header + b" stale", 1)
    doc = tmp_path / "flashinfer_moe_api.md"
    doc.write_bytes(stale)
    with pytest.raises(AssertionError, match="is stale"):
        check_activation_matrix(doc)
    script = """
import sys
from pathlib import Path
from scripts.generate_moe_activation_matrix import (
    check_activation_matrix,
    write_activation_matrix,
)
doc = Path(sys.argv[1])
assert write_activation_matrix(doc)
check_activation_matrix(doc)
"""
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["LC_CTYPE"] = "C"
    env["PYTHONUTF8"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(doc)],
        cwd=matrix.REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    # The checked-in matrix matches the renderer, so rewriting the one stale
    # header restores the original bytes, including the prose outside the block.
    assert doc.read_bytes() == source


@pytest.mark.cpu_only
def test_quant_specific_activation_mapping_must_cover_exact_variants(monkeypatch):
    class Config:
        pass

    class Runner:
        backend_key = "incomplete"
        supported_quant_variants = (
            (QuantFormat.BF16, QuantFormat.BF16),
            (QuantFormat.NVFP4, QuantFormat.NVFP4),
        )
        supported_activation_classes_by_quant = {
            (QuantFormat.BF16, QuantFormat.BF16): (SwiGLU,),
        }

    monkeypatch.setattr(matrix, "_BACKEND_RUNNERS", {Config: Runner})
    with pytest.raises(ValueError, match="must cover exactly"):
        get_activation_matrix_rows()
