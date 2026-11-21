"""The cuDNN plan-build core runs in worker processes that must not import torch."""

import pathlib
import subprocess
import sys

import flashinfer

MODULE = (
    pathlib.Path(flashinfer.__file__).parent / "gemm" / "_cudnn_plan_build.py"
).resolve()


def test_plan_build_module_imports_without_torch():
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('m', {str(MODULE)!r})\n"
        "m = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(m)\n"
        "assert 'torch' not in sys.modules, 'torch was imported'\n"
        "assert 'flashinfer' not in sys.modules, 'flashinfer was imported'\n"
        "print('ok')\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ok"
