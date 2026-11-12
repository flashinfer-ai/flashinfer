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
"""Full-prefill admission must prove every request, without device readback."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch
import flashinfer.prefill as fp
from flashinfer.cudnn import prefill as cp

pytestmark = pytest.mark.cpu_only


def route(monkeypatch, dim=256, qs=(2049, 257, 1), ks=None, **changes):
    cc = changes.pop("cc", (10, 3))
    available = changes.pop("available", True)
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(
        fp, "_cudnn_supports_paged_full_prefill_auto", lambda dim: available
    )
    ks = qs if ks is None else ks
    args = dict(
        device=torch.device("cuda"),
        layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        head_dim_qk=dim,
        head_dim_vo=dim,
        batch=len(qs),
        hq=16,
        hk=16 if dim == 256 else 4,
        max_q=max(qs),
        max_kv=max(ks),
        page_size=64,
        total_q=sum(qs),
        qo_indptr_host=torch.tensor([0, *accumulate(qs)], dtype=torch.int32),
        kv_lens=torch.tensor(ks, dtype=torch.int32),
        seq_lens_q=None,
    )
    args.update(changes)
    return fp._blackwell_paged_auto_cudnn(**args)


@pytest.mark.parametrize("cc", [(10, 0), (10, 3)])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize(
    "dim,hk",
    [(128, 2), (128, 4), (256, 16), (256, 8), (256, 4), (256, 2), (256, 1)],
)
def test_full_prefill_additions(monkeypatch, cc, page, dim, hk):
    assert route(monkeypatch, dim, hk=hk, cc=cc, page_size=page)


@pytest.mark.parametrize(
    "dim,batch,hk",
    [
        (256, 8, 16),
        (256, 16, 8),
        (256, 16, 4),
        (256, 16, 2),
        (256, 16, 1),
        (128, 32, 4),
    ],
)
def test_full_prefill_batch_edge(monkeypatch, dim, batch, hk):
    assert route(monkeypatch, dim, qs=[1027] * batch, hk=hk)
    assert not route(monkeypatch, dim, qs=[1027] * (batch + 1), hk=hk)


@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
def test_full_prefill_work_floor_and_length_edges(monkeypatch, group):
    assert not route(monkeypatch, qs=[1023], hk=16 // group)
    assert route(monkeypatch, qs=[1024], hk=16 // group)
    assert not route(monkeypatch, qs=[511], hq=64, hk=64 // group)
    assert route(monkeypatch, qs=[512], hq=64, hk=64 // group)
    assert route(monkeypatch, qs=[16384], hk=16 // group)
    assert not route(monkeypatch, qs=[16385], hk=16 // group)
    # Equal maxima and a large total cannot hide a partial request.
    assert not route(monkeypatch, qs=[2049, 257], ks=[2049, 2049], hk=16 // group)
    assert not route(
        monkeypatch, qs=[2049, 257], max_q=4096, max_kv=4096, hk=16 // group
    )


class DeviceOnly:
    device = SimpleNamespace(type="cuda")

    def flatten(self):
        raise AssertionError("admission inspected device metadata")

    def tolist(self):
        raise AssertionError("admission copied device metadata")


@pytest.mark.parametrize("field", ["kv_lens", "qo_indptr_host", "seq_lens_q"])
@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
def test_full_prefill_no_device_readback(monkeypatch, field, group):
    assert not route(monkeypatch, hk=16 // group, **{field: DeviceOnly()})


@pytest.mark.parametrize(
    "change",
    [
        dict(layout="NHD"),
        dict(page_size=32),
        dict(q_dtype=torch.float16),
        dict(kv_dtype=torch.float16),
        dict(o_dtype=torch.float16),
        dict(head_dim_vo=128),
        dict(hq=128, hk=128),
        dict(hq=4, hk=4),
        dict(hk=0),
        dict(hq=64, hk=2),
        dict(hq=24, hk=8),
        dict(kv_lens=None),
        dict(seq_lens_q=torch.tensor([2049, 257, 1])),
        dict(total_q=999),
        dict(kv_lens=torch.tensor([2049, 257])),
        dict(qo_indptr_host=torch.tensor([1, 2050, 2307, 2308])),
        dict(qo_indptr_host=torch.tensor([0, 2049, 2049, 2307])),
        dict(cc=(9, 0)),
        dict(cc=(10, 7)),
        dict(cc=(12, 0)),
        dict(available=False),
    ],
)
def test_full_prefill_contract_rejections(monkeypatch, change):
    assert not route(monkeypatch, **change)


@pytest.mark.parametrize("dim", [128, 256])
@pytest.mark.parametrize(
    "version,base,width,expected",
    [
        ("1.32.3", True, True, False),
        ("1.33.0rc1", True, True, False),
        ("1.33.0", True, True, True),
        ("1.33.0+local", True, True, True),
        ("1.33.0", False, True, False),
        ("invalid", True, True, False),
    ],
)
def test_full_prefill_release_gate(monkeypatch, dim, version, base, width, expected):
    monkeypatch.setattr(cp, "cudnn", SimpleNamespace(__version__=version))
    monkeypatch.setattr(cp, "_cudnn_supports_paged_auto", lambda: base)
    monkeypatch.setattr(cp, "_cudnn_supports_paged_gqa_auto", lambda d: width)
    cp._cudnn_supports_paged_full_prefill_auto.cache_clear()
    try:
        assert cp._cudnn_supports_paged_full_prefill_auto(dim) is expected
    finally:
        cp._cudnn_supports_paged_full_prefill_auto.cache_clear()


def test_full_prefill_d256_keeps_native_guard(monkeypatch):
    monkeypatch.setattr(cp, "cudnn", SimpleNamespace(__version__="1.33.0"))
    monkeypatch.setattr(cp, "_cudnn_supports_paged_auto", lambda: True)
    monkeypatch.setattr(cp, "_cudnn_supports_paged_gqa_auto", lambda d: False)
    cp._cudnn_supports_paged_full_prefill_auto.cache_clear()
    try:
        assert cp._cudnn_supports_paged_full_prefill_auto(128)
        assert not cp._cudnn_supports_paged_full_prefill_auto(256)
        assert not cp._cudnn_supports_paged_full_prefill_auto(64)
    finally:
        cp._cudnn_supports_paged_full_prefill_auto.cache_clear()
