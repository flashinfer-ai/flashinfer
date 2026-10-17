"""Hardware reference checks, isolated from the installed serving package."""

from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize("bm,bn,groups", [(32, 32, 2), (32, 32, 4), (64, 64, 2)])
def test_reference_and_graphs(bm, bn, groups, tmp_path):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 required")
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("check.py")),
            "--bm",
            str(bm),
            "--bn",
            str(bn),
            "--groups",
            str(groups),
            "--output",
            str(tmp_path / "correctness.json"),
        ],
        check=True,
    )
