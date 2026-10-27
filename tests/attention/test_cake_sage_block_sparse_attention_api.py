"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests of the flashinfer public surface of the SM120 Cake Sage block-sparse
attention programs: ``bsa_attn_sm120_blk64_sage_fwd(..., backend="cake")``
forwards every argument to the package function and returns its bits.  The
numerical contract is ``test_cake_sage_block_sparse_attention.py``, which
imports only the backend package.
"""

from __future__ import annotations

import pytest
import torch

import flashinfer
import flashinfer.cake_sage_block_sparse_attention_backend as backend
from flashinfer.cake_sage_block_sparse_attention_backend import cake_jit as loader
from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import bsa_attn_sm120_blk64_sage_fwd
from flashinfer.jit.env import has_flashinfer_jit_cache
from tests.attention.test_cake_sage_block_sparse_attention import (
    _CASES,
    _aligned_workspace,
    _make_inputs,
)


def _supported() -> bool:
    return torch.cuda.is_available() and backend.is_cake_sage_sm120_supported()


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="generated Sage block-sparse attention requires compute capability 12.0",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        flashinfer.jit.build_jit_specs(
            loader.specs_for_arch(loader.SM120_ARCH), verbose=False
        )
    yield


_ROUTE_CASES = [case for case in _CASES if case["name"].startswith("ut_")][:4]


@pytest.mark.parametrize(
    "case", _ROUTE_CASES, ids=[case["name"] for case in _ROUTE_CASES]
)
def test_public_entry_routes_backend_cake_to_the_package(case):
    params = dict(case["params"])
    inputs = _make_inputs(params, device=torch.device("cuda"))
    _storage, workspace = _aligned_workspace(device=inputs["Q"].device)
    softmax_scale = (
        None if params["softmax_scale"] == "default" else float(params["softmax_scale"])
    )
    kwargs = dict(
        block_sizes=inputs["block_sizes"],
        q2k_block_nums=inputs["q2k_block_nums"],
        softmax_scale=softmax_scale,
        tma_descriptor_workspace=workspace,
        uniform_block_count=inputs["uniform_block_count"],
        contiguous_block_indices=inputs["contiguous_block_indices"],
    )
    positional = (
        inputs["Q"],
        inputs["K"],
        inputs["V"],
        inputs["Q_scale"],
        inputs["K_scale"],
        inputs["V_scale"],
        inputs["q2k_block_index"],
        inputs["block_sparse_num"],
    )
    out_route = torch.empty_like(inputs["O"])
    out_pkg = torch.empty_like(inputs["O"])
    returned = bsa_attn_sm120_blk64_sage_fwd(
        *positional, out=out_route, backend="cake", **kwargs
    )
    assert returned.data_ptr() == out_route.data_ptr()
    backend.bsa_attn_sm120_blk64_sage_fwd_cake(*positional, out=out_pkg, **kwargs)
    torch.cuda.synchronize()
    assert torch.equal(out_route, out_pkg)


def test_backend_cake_requires_a_caller_owned_output():
    params = dict(_ROUTE_CASES[0]["params"])
    inputs = _make_inputs(params, device=torch.device("cuda"))
    with pytest.raises(ValueError, match="caller-owned out"):
        bsa_attn_sm120_blk64_sage_fwd(
            inputs["Q"],
            inputs["K"],
            inputs["V"],
            inputs["Q_scale"],
            inputs["K_scale"],
            inputs["V_scale"],
            inputs["q2k_block_index"],
            inputs["block_sparse_num"],
            backend="cake",
        )
