# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Validate query tiles without importing FlashInfer's CUDA dependencies."""

import ast
import contextlib
import operator
from pathlib import Path
from typing import Optional

import pytest


@pytest.fixture(scope="module")
def validate():
    # utils.py imports CUDA-dependent JIT helpers. Compile the real, pure
    # validator in isolation so these boundary checks also run on CPU hosts.
    path = Path(__file__).resolve().parents[3] / "flashinfer" / "utils.py"
    function = next(
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_validate_fixed_cta_tile_q"
    )
    namespace = {
        "Optional": Optional,
        "contextlib": contextlib,
        "operator": operator,
    }
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"),
        namespace,
    )
    return namespace[function.name]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "qk,vo,supported",
    [
        (128, 128, (16, 64, 128)),
        (256, 128, (16, 64, 128)),
        (128, 256, (16, 64)),
        (256, 256, (16, 64)),
        (512, 128, (16,)),
        (512, 256, (16,)),
        (576, 256, (16,)),
        (512, 512, (16, 32)),
        (576, 512, (16, 32)),
        (1024, 1024, (16, 32)),
    ],
)
@pytest.mark.parametrize("tile", [16, 32, 64, 128])
def test_fixed_tile_head_dimensions(validate, qk, vo, supported, tile):
    if tile in supported:
        assert validate(tile, vo, qk) == tile
    else:
        with pytest.raises(ValueError, match="is not supported with head_dim"):
            validate(tile, vo, qk)


@pytest.mark.cpu_only
@pytest.mark.parametrize("tile", [-1, 0, 1, 48, 256, True, 16.0, "64"])
def test_invalid_fixed_tile(validate, tile):
    with pytest.raises(ValueError, match="fixed_cta_tile_q should be an integer in"):
        validate(tile, 128)


@pytest.mark.cpu_only
@pytest.mark.parametrize("head_dim", [128, 256, 512, 1024])
def test_fixed_tile_default(validate, head_dim):
    assert validate(None, head_dim) == -1
    assert validate(16, head_dim) == 16


@pytest.mark.cpu_only
def test_fixed_tile_integer_like(validate):
    np = pytest.importorskip("numpy")
    tile = validate(np.int64(64), 128)
    assert tile == 64 and type(tile) is int
