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

"""The KDA cuDNN auto region, without a GPU."""

import importlib

import pytest
import torch

kda = importlib.import_module("flashinfer.kda")


@pytest.fixture
def native_auto_policy():
    """This module exercises the auto resolution itself, so the native-only
    pin of the directory's conftest does not apply here."""
    yield


@pytest.mark.parametrize(
    "heads,lengths,state_dtype,expected",
    [
        (4, (2048,), torch.bfloat16, False),
        (4, (4096,), torch.bfloat16, False),
        (4, (8192,), torch.bfloat16, True),
        (8, (32768,), torch.bfloat16, True),
        (16, (2048,), torch.bfloat16, False),
        (16, (4096,), torch.bfloat16, False),
        (16, (8192,), None, True),
        (16, (8192,), torch.float32, True),
        (16, (8192,), torch.float16, True),
        (4, (24576, 8192), torch.bfloat16, True),
        (4, (8192,) * 4, torch.bfloat16, True),
        (8, (4096,) * 3, torch.bfloat16, False),
        (8, (16384, 4096, 4096), torch.bfloat16, True),
        (8, (8192,) * 3, torch.bfloat16, True),
        (8, (8192,) * 4, torch.bfloat16, False),
        (16, (6144, 2048), torch.bfloat16, False),
        (16, (8192,) * 4, torch.bfloat16, False),
        (32, (8192,), torch.bfloat16, False),
        (64, (32768,), torch.bfloat16, False),
        (32, (2048,) * 4, torch.bfloat16, True),
        (64, (2048,) * 2, torch.bfloat16, True),
        (96, (2048,), torch.bfloat16, True),
        (96, (16384,), torch.bfloat16, True),
    ],
)
def test_kda_policy_region(monkeypatch, heads, lengths, state_dtype, expected):
    """Up to 24 (head, sequence) tiles from a mean sequence length of 8k, and
    everything above FlashInfer's small-BH boundary (2 x tiles > SM count, 148
    here); the rest native."""
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(kda, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    total = sum(lengths)
    q = torch.empty(1, total, heads, 128, dtype=torch.bfloat16)
    v = torch.empty(1, total, heads, 128, dtype=torch.bfloat16)
    state = (
        None
        if state_dtype is None
        else torch.empty(len(lengths), heads, 128, 128, dtype=state_dtype)
    )
    cu_seqlens = torch.tensor([0, *lengths]).cumsum(0)
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is expected
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (12, 0))
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is False


@pytest.mark.parametrize(
    "dims,expected",
    [
        ((64, 64), True),
        ((64, 128), True),
        ((128, 64), True),
        ((128, 128), False),
        ((256, 128), False),
        ((128, 96), False),
    ],
)
def test_kda_policy_head_dims(monkeypatch, dims, expected):
    """cuDNN's engine serves K and V head dims of 64 or 128 in any pairing and
    FlashInfer's KDA kernels 128 x 128 only, so a 64-wide K or V head is
    cuDNN's at every shape, here 32 tiles at 8k, a cell the 128 x 128 rule
    leaves native; widths the engine does not serve stay native."""
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(kda, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.empty(1, 8192, 32, dims[0], dtype=torch.bfloat16)
    v = torch.empty(1, 8192, 32, dims[1], dtype=torch.bfloat16)
    state = torch.empty(1, 32, dims[1], dims[0], dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 8192])
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is expected
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (12, 0))
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is False


@pytest.mark.parametrize("problem", ["grad", "fp16_inputs", "batched"])
def test_kda_policy_declines_what_cudnn_cannot_take(monkeypatch, problem):
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(kda, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    q = torch.empty(1, 8192, 8, 128, dtype=torch.bfloat16)
    v = torch.empty(1, 8192, 8, 128, dtype=torch.bfloat16)
    state = torch.empty(1, 8, 128, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, 8192])
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is True
    if problem == "grad":
        state.requires_grad_(True)
    elif problem == "fp16_inputs":
        q = q.half()
    elif problem == "batched":
        q = torch.empty(2, 4096, 8, 128, dtype=torch.bfloat16)
    assert kda._prefer_cudnn_kda_prefill(q, v, state, cu_seqlens, None) is False


@pytest.mark.parametrize(
    "heads,lengths,pool_expected,compact_expected",
    [
        (16, (8192,) * 4, True, False),
        (64, (8192,), True, False),
        (64, (32768,), True, False),
        (8, (8192,) * 4, True, False),
        (96, (8192,), True, True),
        (8, (8192,) * 3, True, True),
        (8, (4096,) * 3, False, False),
        (16, (4096,), False, False),
    ],
)
def test_kda_policy_pool_clause(
    monkeypatch, heads, lengths, pool_expected, compact_expected
):
    """A state pool never reaches FlashInfer's small-BH kernel, so above 24
    tiles the pool form is cuDNN's where the compact form stays native (the
    64-tile pool cells measured 0.94 to 0.96 of the native pool route,
    docs/api/cudnn.rst); up to 24 tiles the mean-length rule applies to both."""
    monkeypatch.setattr(kda, "get_compute_capability", lambda device: (10, 0))
    monkeypatch.setattr(kda, "get_device_sm_count", lambda device: 148)
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    total = sum(lengths)
    q = torch.empty(1, total, heads, 128, dtype=torch.bfloat16)
    v = torch.empty(1, total, heads, 128, dtype=torch.bfloat16)
    cu_seqlens = torch.tensor([0, *lengths]).cumsum(0)
    pool = torch.empty(len(lengths) + 3, heads, 128, 128, dtype=torch.bfloat16)
    slots = torch.arange(len(lengths), dtype=torch.int32)
    assert kda._prefer_cudnn_kda_prefill(q, v, pool, cu_seqlens, slots) is pool_expected
    compact = torch.empty(len(lengths), heads, 128, 128, dtype=torch.bfloat16)
    assert (
        kda._prefer_cudnn_kda_prefill(q, v, compact, cu_seqlens, None)
        is compact_expected
    )
