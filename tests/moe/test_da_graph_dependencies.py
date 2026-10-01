"""Execute the native DA conditional-ancestry regression through pytest."""

import subprocess
from pathlib import Path

import pytest
import torch

from flashinfer.jit.cpp_ext import get_cuda_path
from flashinfer.utils import device_support_pdl


def test_da_graph_dependencies(tmp_path: Path) -> None:
    """PDL descendants of a completed SWITCH safely reuse both workspace roots."""
    if not torch.cuda.is_available() or not device_support_pdl(torch.device("cuda")):
        pytest.skip("conditional ancestry regression requires CUDA with PDL support")
    root = Path(__file__).resolve().parents[2]
    major, minor = torch.cuda.get_device_capability()
    executable = tmp_path / "da_graph_dependencies"
    subprocess.run(
        [
            str(Path(get_cuda_path()) / "bin" / "nvcc"),
            "-std=c++17",
            f"-arch=sm_{major}{minor}",
            f"-I{root / 'include'}",
            f"-I{root / '3rdparty/cccl/cub'}",
            f"-I{root / '3rdparty/cccl/thrust'}",
            f"-I{root / '3rdparty/cccl/libcudacxx/include'}",
            str(Path(__file__).with_name("csrc") / "da_graph_dependencies.cu"),
            "-o",
            str(executable),
        ],
        check=True,
        timeout=180,
    )
    result = subprocess.run(
        [str(executable), str(torch.cuda.current_device())],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert (
        "PASS: metadata query, conditional ancestry replay, and retained frontier roots"
        in result.stdout
    )
