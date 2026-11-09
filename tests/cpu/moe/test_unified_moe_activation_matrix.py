"""CPU-only consistency checks for the generated Unified MoE activation matrix."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from flashinfer.fused_moe.api import QuantFormat, SwiGLU
from scripts import generate_moe_activation_matrix as matrix
from scripts.generate_moe_activation_matrix import (
    check_activation_matrix,
    get_activation_matrix_rows,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DOC_PATH = _REPO_ROOT / "docs" / "design_docs" / "flashinfer_moe_api.md"


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
    source = _DOC_PATH.read_bytes()
    assert any(byte > 127 for byte in source)
    doc = tmp_path / "flashinfer_moe_api.md"
    # A stale cell inside the generated block forces a rewrite of the whole file.
    stale = source.replace(b"`b12x_nvfp4`", b"`b12x_nvfp4_stale`", 1)
    assert stale != source
    doc.write_bytes(stale)
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
data = doc.read_bytes()
data.decode("utf-8")
assert b"\\r\\n" not in data
"""
    env = os.environ.copy()
    env["LC_ALL"] = "C"
    env["LANG"] = "C"
    env["LC_CTYPE"] = "C"
    env["PYTHONUTF8"] = "0"
    completed = subprocess.run(
        [sys.executable, "-c", script, str(doc)],
        cwd=_REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


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
