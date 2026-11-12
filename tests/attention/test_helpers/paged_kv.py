# Copyright (c) 2026 by FlashInfer team.
#
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

"""Small helpers for constructing deliberate paged-KV stride families."""

from __future__ import annotations

import torch


def make_padded_paged_kv_view(
    source: torch.Tensor,
    kv_layout: str,
    *,
    padding_heads: int = 2,
) -> torch.Tensor:
    """Copy paged KV into an equal-shaped view with different outer strides."""
    head_dimension = {"NHD": 2, "HND": 1}[kv_layout]
    shape = list(source.shape)
    shape[head_dimension] += padding_heads
    view = torch.empty(shape, dtype=source.dtype, device=source.device).narrow(
        head_dimension, 0, source.shape[head_dimension]
    )
    view.copy_(source)
    assert view.stride() != source.stride()
    return view


def _padded_strides(shape, alignment):
    # Leave slack at each level for an isolated inner-stride increment without
    # changing either outer stride. All byte offsets meet vector-load alignment.
    inner = shape[3] + 4 * alignment
    middle = shape[2] * (inner + alignment) + 4 * alignment
    outer = shape[1] * (middle + alignment) + 4 * alignment
    return (outer, middle, inner, 1)


def make_paged_kv_cache_pair(k_dense, v_dense, layout, mode, alignment):
    assert layout in ("NHD", "HND")
    assert mode in ("contiguous", "padded", "page", "token", "head")
    assert k_dense.shape == v_dense.shape
    if mode == "contiguous":
        k, v = k_dense.clone(), v_dense.clone()
        assert k.is_contiguous() and v.is_contiguous()
    else:
        ks = _padded_strides(k_dense.shape, alignment)
        vs = list(ks)
        changed = {
            "page": 0,
            "token": 1 if layout == "NHD" else 2,
            "head": 2 if layout == "NHD" else 1,
        }.get(mode)
        if changed is not None:
            assert k_dense.shape[changed] > 1
            vs[changed] += alignment

        def allocate(dense, strides):
            size = 1 + sum(
                (n - 1) * s for n, s in zip(dense.shape, strides, strict=True)
            )
            # Poison padding so an incorrect offset cannot quietly read zeros.
            storage = torch.full((size,), 31, dtype=dense.dtype, device=dense.device)
            return storage.as_strided(dense.shape, strides)

        k = allocate(k_dense, ks)
        v = allocate(v_dense, tuple(vs))
        k.copy_(k_dense)
        v.copy_(v_dense)
        for tensor in (k, v):
            assert tensor.stride(3) == 1
            assert all(s > 0 and s % alignment == 0 for s in tensor.stride()[:3])
            # Sufficient non-overlap proof for these ordered, positive strides.
            extent = tensor.shape[3]
            for axis in (2, 1, 0):
                assert tensor.stride(axis) >= extent
                extent += (tensor.shape[axis] - 1) * tensor.stride(axis)
    assert k.data_ptr() != v.data_ptr()
    differences = [i for i in range(4) if k.stride(i) != v.stride(i)]
    expected = {
        "page": [0],
        "token": [1 if layout == "NHD" else 2],
        "head": [2 if layout == "NHD" else 1],
    }.get(mode, [])
    assert differences == expected
    # Check copied logical values before the kernel can modify any buffers.
    assert torch.equal(k, k_dense) and torch.equal(v, v_dense)
    return k, v
