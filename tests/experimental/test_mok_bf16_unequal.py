# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=4 -m pytest <this file>."""

import importlib.util
import os
from pathlib import Path

import pytest

pytest_plugins = ["tests.experimental._mok_bf16_test_utils"]


def test_unequal_source_graphs(monkeypatch, mok_distributed_group):
    if int(os.environ.get("WORLD_SIZE", "1")) != 4:
        pytest.skip("Requires four distributed ranks")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("Requires an SM100a, SM103a or SM107a CUDA device")
    directory = Path(__file__).resolve().parents[2] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "mok_bf16_unequal", directory / "mok_bf16_unequal.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
