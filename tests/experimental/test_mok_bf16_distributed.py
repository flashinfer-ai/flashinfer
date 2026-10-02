# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=4 -m pytest <this file>."""

import importlib.util
import os
from pathlib import Path

import pytest


def test_complete_training_graph(monkeypatch):
    if "RANK" not in os.environ or int(os.environ.get("WORLD_SIZE", "0")) not in (1, 4):
        pytest.skip("Launch with torchrun using one or four ranks")
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
        (10, 7),
    ):
        pytest.skip("Requires an SM100a-compatible CUDA device")

    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(
        torch.backends.cuda.matmul, "allow_bf16_reduced_precision_reduction", False
    )
    path = Path(__file__).resolve().parents[2] / "examples" / "mok_bf16_toy.py"
    spec = importlib.util.spec_from_file_location("mok_bf16_toy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.main()
