# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Run with torchrun --standalone --nproc-per-node=4 -m pytest <this file>.

Clamped SwiGLU, recompute-vs-saved-context, FP32 weight-gradient
accumulation (BF16 and MXFP8) and native MXFP8 over unequal (including empty)
ranks. EP4 uses
256 routed experts; EP8 and EP32 use the 288-expert GLM-5.3-Flash layout.
"""

import importlib.util
import os
from pathlib import Path

import pytest

pytest_plugins = ["tests.experimental._mok_bf16_test_utils"]


def test_training_features(monkeypatch, mok_distributed_group):
    if int(os.environ.get("WORLD_SIZE", "0")) not in (1, 4, 8, 16, 32, 64):
        pytest.skip("Launch with torchrun using 1, 4, 8, 16, 32 or 64 ranks")
    directory = Path(__file__).resolve().parents[2] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location(
        "mok_training_features", directory / "mok_training_features.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records = module.main()
    assert {r["case"] for r in records} == {c["name"] for c in module.CASES}
