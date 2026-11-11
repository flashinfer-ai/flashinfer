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

Tests of the flashinfer public surface of the SM100 / SM103 Cake Sage block-sparse
attention programs: the ``flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake`` facade
and the ``backend="cake"`` route of ``bsa_attn_sm100_blk64_fwd``.  The facade's
entry points are the package's own functions; this module checks that identity,
the dispatcher's route against the package and against the CuTe-DSL backend,
and leaves the numerical contract to
``test_cake_sage_block_sparse_attention_sm100.py``, which imports only the
backend package.
"""

from __future__ import annotations

import math

import pytest
import torch

import flashinfer
import flashinfer.cake_sage_block_sparse_attention_backend as backend
from flashinfer.cake_sage_block_sparse_attention_backend import cake_jit as loader
from flashinfer.cute_dsl.sparse.bsa_sage_sm100_cake import (
    bsa_attn_sm100_sage_fwd_cake,
    is_cake_sage_sm100_supported,
    sage_fp8_quantize_sm100,
)
from flashinfer.jit.env import has_flashinfer_jit_cache

_HEAD_DIM = 128
_BLOCK = 64


def _supported() -> bool:
    return torch.cuda.is_available() and is_cake_sage_sm100_supported()


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="Cake Sage-FP8 block-sparse attention requires SM100 or SM103",
)


@pytest.fixture(autouse=not has_flashinfer_jit_cache(), scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(torch.cuda.get_device_capability())
        flashinfer.jit.build_jit_specs(loader.specs_for_arch(arch), verbose=False)
    yield


def _inputs(
    batch: int, heads: int, kv_heads: int, sq: int, sk: int, sel: int, seed: int
):
    device = torch.device("cuda")
    g = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(
        (batch, sq, heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    k = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    v = torch.randn(
        (batch, sk, kv_heads, _HEAD_DIM), device=device, generator=g
    ).bfloat16()
    q8, k8, v8, q_scale, k_scale, v_scale = sage_fp8_quantize_sm100(q, k, v)
    q_blocks, k_blocks = -(-sq // _BLOCK), -(-sk // _BLOCK)
    scores = torch.rand((batch, heads, q_blocks, k_blocks), device=device, generator=g)
    index = (
        scores.argsort(dim=-1, descending=True)[..., :sel].to(torch.int32).contiguous()
    )
    return q8, k8, v8, q_scale, k_scale, v_scale, index


def test_facade_entry_points_are_the_package_functions() -> None:
    assert bsa_attn_sm100_sage_fwd_cake is backend.bsa_attn_sm100_sage_fwd_cake
    assert sage_fp8_quantize_sm100 is backend.sage_fp8_quantize_sm100
    assert is_cake_sage_sm100_supported is backend.is_cake_sage_sm100_supported


def test_blk64_dispatcher_routes_backend_cake_to_the_package() -> None:
    """``bsa_attn_sm100_blk64_fwd(..., backend="cake")`` returns the package's bits."""
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(2, 4, 2, 257, 1000, 6, 7)
    scale = 1.0 / math.sqrt(_HEAD_DIM)
    out_route, lse_route = bsa_attn_sm100_blk64_fwd(
        q8,
        k8,
        v8,
        index,
        6,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=scale,
        return_lse=True,
        backend="cake",
    )
    out_pkg, lse_pkg = backend.bsa_attn_sm100_sage_fwd_cake(
        q8,
        k8,
        v8,
        index,
        6,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
        softmax_scale=scale,
        return_lse=True,
    )
    torch.cuda.synchronize()
    assert torch.equal(out_route, out_pkg) and torch.equal(lse_route, lse_pkg)


def test_cake_sage_sm100_matches_cute_backend():
    from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import bsa_attn_sm100_blk64_fwd

    q8, k8, v8, q_scale, k_scale, v_scale, index = _inputs(1, 8, 8, 512, 512, 4, 512)
    common = dict(
        q_scale=q_scale,
        k_scale=k_scale,
        softmax_scale=1.0 / math.sqrt(_HEAD_DIM),
        return_lse=True,
    )
    out_cake, lse_cake = bsa_attn_sm100_blk64_fwd(
        q8, k8, v8, index, 4, v_scale=v_scale, backend="cake", **common
    )
    out_cute, lse_cute = bsa_attn_sm100_blk64_fwd(
        q8, k8, v8, index, 4, v_scale=v_scale[0], backend="cute", **common
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(out_cake.float(), out_cute.float(), atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(lse_cake, lse_cute, atol=1e-2, rtol=1e-2)
