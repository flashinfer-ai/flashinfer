# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The cubin downloader must import without the kernel backends' dependencies."""

import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.cpu_only
def test_cubin_downloader_imports_without_cutlass():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(Path(__file__).parents[3]), env.get("PYTHONPATH")))
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import builtins
import importlib.util
import sys

# Block the import even when the PR test image has CUTLASS DSL installed.
original_import = builtins.__import__
original_find_spec = importlib.util.find_spec

def without_cutlass(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "cutlass" or name.startswith("cutlass."):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)
    return original_import(name, globals, locals, fromlist, level)

def find_spec_without_cutlass(name, *args, **kwargs):
    if name == "cutlass" or name.startswith("cutlass."):
        return None
    return original_find_spec(name, *args, **kwargs)

builtins.__import__ = without_cutlass
importlib.util.find_spec = find_spec_without_cutlass
from flashinfer.artifacts import download_artifacts

assert callable(download_artifacts)
assert "flashinfer.fused_moe.batch_invariant.api_fp4" not in sys.modules
assert "flashinfer.fused_moe.batch_invariant.blocks.epilogue" not in sys.modules
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
""",
        ],
        check=True,
        env=env,
        timeout=60,
    )
