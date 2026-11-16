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

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

import flashinfer.prefill as fp
from flashinfer.cudnn import prefill as cp

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize(
    "version,packed,origins,runtime,expected",
    [
        ("1.32.3", True, True, True, False),
        ("1.33.0rc1", True, True, True, False),
        ("1.33.0", True, True, True, True),
        ("1.33.0+local", True, True, True, True),
        ("1.33.0", False, True, True, False),
        ("1.33.0", True, False, True, False),
        ("1.33.0", True, True, False, False),
    ],
)
def test_width_auto_requires_matching_native(
    monkeypatch, dim, version, packed, origins, runtime, expected
):
    binder = SimpleNamespace(
        **{
            f"supports_paged_d{dim}_packed_split": packed,
            f"supports_paged_d{dim}_split_physical_origins": origins,
        }
    )
    monkeypatch.setattr(
        cp,
        "cudnn",
        SimpleNamespace(
            __version__=version, _pybind_module=SimpleNamespace(_SdpaThdBinder=binder)
        ),
    )
    monkeypatch.setattr(cp, "_cudnn_supports_paged_auto", lambda: runtime)
    cp._cudnn_supports_paged_gqa_auto.cache_clear()
    try:
        assert cp._cudnn_supports_paged_gqa_auto(dim) is expected
        assert not cp._cudnn_supports_paged_gqa_auto(128)
        assert not cp._cudnn_supports_paged_gqa_auto(64)
    finally:
        cp._cudnn_supports_paged_gqa_auto.cache_clear()


def route(
    monkeypatch, dim, qs, ks, *, hq=32, hk=8, cc=(10, 0), runtime=True, **changes
):
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_gqa_auto", lambda dim: runtime)
    kwargs = dict(
        device=torch.device("cuda"),
        layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        head_dim_qk=dim,
        head_dim_vo=dim,
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
    return fp._blackwell_paged_auto_cudnn(**kwargs)


@pytest.mark.parametrize("dim", [256])
def test_width_domains_and_actual_work_floor(monkeypatch, dim):
    assert route(monkeypatch, dim, [128], [32768])
    assert route(monkeypatch, dim, [512] * 4, [8192] * 4)
    floor_kv = 8192
    assert not route(monkeypatch, dim, [128], [floor_kv - 1], hq=16, hk=4)
    assert route(monkeypatch, dim, [128], [floor_kv], hq=16, hk=4)
    assert not route(monkeypatch, dim, [128], [32768], hq=16, hk=16)
    assert not route(monkeypatch, dim, [128] * 5, [32768] * 5)
    assert not route(monkeypatch, dim, [63], [32768])
    assert not route(monkeypatch, dim, [1025], [32768])
    assert not route(monkeypatch, dim, [128], [32769])
    assert not route(monkeypatch, dim, [1024], [2048])


@pytest.mark.parametrize("dim", [256])
def test_width_mixed_batches_stay_in_declaration_class(monkeypatch, dim):
    assert route(monkeypatch, dim, [64, 96, 128], [8192, 16384, 32768])
    assert route(monkeypatch, dim, [256, 512, 768], [16384, 8192, 32768])
    assert not route(monkeypatch, dim, [64, 64, 1024], [32768, 32768, 4096])
    assert not route(monkeypatch, dim, [128], [32768], max_q=256)


class DeviceOnly:
    device = SimpleNamespace(type="cuda")

    def flatten(self):
        raise AssertionError("dispatch readback")

    def tolist(self):
        raise AssertionError("dispatch readback")


@pytest.mark.parametrize("dim", [256])
@pytest.mark.parametrize("field", ["kv_lens", "seq_lens_q", "qo_indptr_host"])
def test_width_auto_never_reads_device_metadata(monkeypatch, dim, field):
    assert not route(monkeypatch, dim, [128], [32768], **{field: DeviceOnly()})


@pytest.mark.parametrize("dim", [256])
def test_explicit_query_masks_use_actual_not_physical_work(monkeypatch, dim):
    assert route(monkeypatch, dim, [128], [32768], seq_lens_q=torch.tensor([96]))
    assert not route(monkeypatch, dim, [128], [32768], seq_lens_q=torch.tensor([1]))
    assert not route(monkeypatch, dim, [128], [32768], seq_lens_q=torch.tensor([129]))


@pytest.mark.parametrize(
    "change",
    [
        dict(layout="NHD"),
        dict(page_size=32),
        dict(q_dtype=torch.float16),
        dict(head_dim_vo=128),
        dict(cc=(10, 3)),
        dict(cc=(10, 7)),
        dict(runtime=False),
        dict(kv_lens=None),
    ],
)
@pytest.mark.parametrize("dim", [256])
def test_width_auto_keeps_other_guards(monkeypatch, dim, change):
    assert not route(monkeypatch, dim, [128], [32768], **change)


@pytest.mark.parametrize(
    "qs,ks",
    [([128], [32768]), ([64, 96, 128], [8193, 16385, 32767]), ([512] * 4, [8192] * 4)],
)
def test_d64_remains_native_with_qualified_runtime(monkeypatch, qs, ks):
    assert not route(monkeypatch, 64, qs, ks)
