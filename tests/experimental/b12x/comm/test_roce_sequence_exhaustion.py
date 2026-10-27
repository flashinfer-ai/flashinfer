"""Reject exhausted doorbells before zero ready words can release unstaged data."""

import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def sequence_boundary_driver(tmp_path_factory):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("RoCE proxy boundary checks require a C compiler")
    root = Path(__file__).resolve().parents[4]
    source = root / "flashinfer/experimental/b12x/comm/roce/_roce_proxy.c"
    harness = Path(__file__).with_name("_roce_sequence_exhaustion.c")
    driver = tmp_path_factory.mktemp("roce-sequence") / "boundary"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-pthread",
            f'-DROCE_PROXY_SOURCE="{source}"',
            str(harness),
            "-libverbs",
            "-o",
            str(driver),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return driver


@pytest.mark.parametrize(
    "sequence,exhausted",
    [(1, False), (0x7FFFFFFF, False), (0x80000000, True), (0xFFFFFFFF, True)],
)
def test_proxy_rejects_exhaustion_before_dispatch(
    sequence_boundary_driver, sequence, exhausted
):
    result = subprocess.run(
        [str(sequence_boundary_driver), str(sequence), str(int(exhausted))],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.stdout.startswith(f"{sequence}: ")
