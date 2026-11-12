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

"""Additive paged auto domains use existing host metadata only."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

import flashinfer.prefill as prefill

pytestmark = pytest.mark.cpu_only


def route(monkeypatch, qs, ks, *, hq=16, hk=4, cc=(10, 0), available=True, **changes):
    monkeypatch.setattr(prefill, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(prefill, "_cudnn_supports_paged_auto", lambda: available)
    kwargs = dict(
        device=torch.device("cuda"),
        layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        head_dim_qk=128,
        head_dim_vo=128,
        batch=len(qs),
        hq=hq,
        hk=hk,
        max_q=max(qs),
        max_kv=max(ks),
        page_size=16,
        total_q=sum(qs),
        qo_indptr_host=torch.tensor([0, *accumulate(qs)], dtype=torch.int32),
        kv_lens=torch.tensor(ks, dtype=torch.int32),
        seq_lens_q=None,
    )
    kwargs.update(changes)
    return prefill._blackwell_paged_auto_cudnn(**kwargs)


@pytest.mark.parametrize(
    "qs,ks,expected",
    [
        ([256] * 6, [4096] * 6, True),
        ([2048], [4096], True),
        ([1024] * 6, [1024] * 6, True),
        ([16384], [16384], True),
        ([16385], [16385], False),
        ([513] * 6, [513] * 6, False),
        ([128] * 6, [6145] * 6, False),
        ([129] * 6, [6145] * 6, True),
        ([8192], [16384], True),
        ([8193], [16384], False),
        ([256] * 6, [2047] * 6, False),
        ([256] * 6, [2048] * 6, True),
        ([256] * 6, [32769] * 6, False),
        ([17, 17, 17, 17, 3072], [32768, 32768, 32768, 32768, 3104], False),
        ([256, 512, 1024, 2048, 3072], [4096, 8192, 6144, 12288, 8192], True),
    ],
)
def test_additional_paged_domains(monkeypatch, qs, ks, expected):
    assert route(monkeypatch, qs, ks) is expected


@pytest.mark.parametrize("q", [64, 65, 94, 127, 128, 1024])
@pytest.mark.parametrize("batch", [1, 2, 4])
def test_existing_short_prefix_admission_is_preserved(monkeypatch, q, batch):
    assert route(monkeypatch, [q] * batch, [8192] * batch)


def test_extended_work_is_paired_and_keeps_original_floor(monkeypatch):
    assert route(monkeypatch, [64], [8192])  # Original 8 Mi floor.
    assert not route(monkeypatch, [129] * 6, [2709] * 6, hq=8)
    assert route(monkeypatch, [129] * 6, [2710] * 6, hq=8)  # 16 Mi extension.
    assert not route(monkeypatch, [1024, 129], [1024, 2048], hq=8)


class DeviceLengths:
    device = SimpleNamespace(type="cuda")

    def flatten(self):
        raise AssertionError("dispatch must not inspect device lengths")

    def tolist(self):
        raise AssertionError("dispatch must not copy device lengths to the host")


@pytest.mark.parametrize("lengths", [None, DeviceLengths()])
def test_extended_domain_does_not_read_back_lengths(monkeypatch, lengths):
    assert not route(monkeypatch, [256] * 6, [4096] * 6, kv_lens=lengths)
    assert route(monkeypatch, [128], [8192], kv_lens=lengths)


@pytest.mark.parametrize("lengths", [torch.tensor([256] * 6), DeviceLengths()])
def test_explicit_query_lengths_do_not_extend_admission(monkeypatch, lengths):
    assert not route(monkeypatch, [256] * 6, [4096] * 6, seq_lens_q=lengths)


@pytest.mark.parametrize(
    "change",
    [
        dict(layout="NHD"),
        dict(page_size=32),
        dict(head_dim_qk=64, head_dim_vo=64),
        dict(q_dtype=torch.float16),
        dict(hq=128),
        dict(hk=3),
    ],
)
def test_extension_keeps_layout_dtype_and_head_contract(monkeypatch, change):
    assert not route(monkeypatch, [256] * 6, [4096] * 6, **change)


@pytest.mark.parametrize("cc", [(9, 0), (10, 3), (10, 7), (12, 0)])
def test_extension_is_sm100_only(monkeypatch, cc):
    assert not route(monkeypatch, [256] * 6, [4096] * 6, cc=cc)


def test_extension_requires_qualified_runtime(monkeypatch):
    assert not route(monkeypatch, [256] * 6, [4096] * 6, available=False)
