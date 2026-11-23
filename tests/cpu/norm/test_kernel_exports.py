# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fresh-process normalization package export contracts; no GPU required."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _fresh_import(code, cuda_norm):
    root = Path(__file__).resolve().parents[3]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["FLASHINFER_USE_CUDA_NORM"] = cuda_norm
    env["PYTHONPATH"] = str(root) + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("cuda_norm", ["0", "1"])
def test_kernel_package_exports_remain_callable_after_parent_import(cuda_norm):
    if importlib.util.find_spec("cutlass") is None:
        pytest.skip("optional CuTe DSL dependency is not installed")
    _fresh_import(
        """
import importlib
import flashinfer
from flashinfer.norm import kernels
from flashinfer.norm.kernels import (
    gated_residual_rmsnorm,
    gated_residual_next_rmsnorm,
    rmsnorm_sigmoid_gate,
    rmsnorm_cute,
)
exports = {
    "gated_residual_rmsnorm": gated_residual_rmsnorm,
    "gated_residual_next_rmsnorm": gated_residual_next_rmsnorm,
    "rmsnorm_sigmoid_gate": rmsnorm_sigmoid_gate,
    "rmsnorm_cute": rmsnorm_cute,
}
for name, value in exports.items():
    assert name in kernels.__all__
    assert callable(value), (name, type(value))
    module_name = "rmsnorm" if name == "rmsnorm_cute" else name
    module = importlib.import_module("flashinfer.norm.kernels." + module_name)
    assert value is getattr(module, name)
    assert getattr(kernels, name) is value
""",
        cuda_norm,
    )


def test_cuda_specialist_stays_importable_without_optional_cute():
    _fresh_import(
        """
import importlib.abc
import importlib.util
import sys
original_find_spec = importlib.util.find_spec
def without_cute(name, *args, **kwargs):
    if name == "cutlass" or name.startswith("cutlass."):
        return None
    return original_find_spec(name, *args, **kwargs)
importlib.util.find_spec = without_cute
class BlockCuTe(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "cutlass" or fullname.startswith("cutlass."):
            raise ModuleNotFoundError("CuTe unavailable for this check", name=fullname)
sys.meta_path.insert(0, BlockCuTe())
import flashinfer.norm as norm
assert norm._USE_CUDA_NORM
assert callable(norm._try_rmsnorm_serve_sm90)
from flashinfer.norm.kernels import rmsnorm_serve_sm90
assert callable(rmsnorm_serve_sm90._get_cuda_module)
assert not any(name == "cutlass" or name.startswith("cutlass.") for name in sys.modules)
""",
        "0",
    )
