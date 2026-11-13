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

"""H100 admission follows the native provider boundary without readback."""

from itertools import accumulate
from types import SimpleNamespace

import pytest
import torch

import flashinfer.prefill as fp

pytestmark = pytest.mark.cpu_only


def route(
    monkeypatch,
    qs,
    ks,
    *,
    cc=(9, 0),
    sms=132,
    name="NVIDIA H100 80GB HBM3",
    available=True,
    **changes,
):
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(fp, "get_device_sm_count", lambda device: sms)
    monkeypatch.setattr(fp, "get_device_name", lambda device: name)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_auto", lambda: available)
    args = dict(
        device=torch.device("cpu"),
        layout="HND",
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        o_dtype=torch.bfloat16,
        head_dim_qk=256,
        head_dim_vo=256,
        batch=len(qs),
        hq=32,
        hk=8,
        max_q=max(qs),
        max_kv=max(ks),
        page_size=16,
        total_q=sum(qs),
        qo_indptr_host=torch.tensor([0, *accumulate(qs)], dtype=torch.int32),
        kv_lens=torch.tensor(ks, dtype=torch.int32),
        seq_lens_q=None,
    )
    args.update(changes)
    return fp._hopper_paged_auto_cudnn(**args)


@pytest.mark.parametrize("batch", [1, 4, 32])
@pytest.mark.parametrize("group", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("hq", [16, 32, 48])
@pytest.mark.parametrize(
    "q,expected", [(63, False), (64, False), (65, True), (129, True)]
)
def test_provider_boundary_and_group_transfer(
    monkeypatch, batch, group, hq, q, expected
):
    qs = [q] + [1] * (batch - 1)
    assert route(monkeypatch, qs, [4096] * batch, hq=hq, hk=hq // group) is expected


@pytest.mark.parametrize(
    "qs,ks",
    [
        ([257], [257]),
        ([513] * 4, [513] * 4),
        ([16384], [16384]),
        ([129, 1], [32768, 1]),
    ],
)
def test_full_prefill_and_mixed_lengths_are_included(monkeypatch, qs, ks):
    assert route(monkeypatch, qs, ks)
    assert route(monkeypatch, qs, ks, seq_lens_q=torch.tensor(qs, dtype=torch.int32))


@pytest.mark.parametrize(
    "change",
    [
        dict(layout="NHD"),
        dict(q_dtype=torch.float16),
        dict(kv_dtype=torch.float16),
        dict(o_dtype=torch.float32),
        dict(head_dim_qk=128),
        dict(head_dim_vo=128),
        dict(page_size=64),
        dict(hq=8, hk=2),
        dict(hq=64, hk=8),
        dict(hk=0),
        dict(hq=32, hk=3),
        dict(hq=24, hk=8),
        dict(batch=33),
        dict(max_q=128),
        dict(max_kv=8192),
        dict(total_q=66),
        dict(seq_lens_q=torch.tensor([64], dtype=torch.int32)),
        dict(qo_indptr_host=torch.tensor([1, 66], dtype=torch.int32)),
    ],
)
def test_unqualified_inputs_do_not_change_provider(monkeypatch, change):
    assert not route(monkeypatch, [65], [4096], **change)


@pytest.mark.parametrize(
    "qs,ks",
    [([0, 65], [4096, 4096]), ([65], [64]), ([16385], [32768]), ([65], [32769])],
)
def test_actual_lengths_must_stay_within_the_domain(monkeypatch, qs, ks):
    assert not route(monkeypatch, qs, ks)


class DeviceOnly:
    device = SimpleNamespace(type="cuda")

    def flatten(self):
        raise AssertionError("admission must not read device metadata")

    def tolist(self):
        raise AssertionError("admission must not copy metadata")


@pytest.mark.parametrize("field", ["kv_lens", "seq_lens_q", "qo_indptr_host"])
def test_no_admission_readback(monkeypatch, field):
    assert not route(monkeypatch, [65], [4096], **{field: DeviceOnly()})


@pytest.mark.parametrize(
    "hardware",
    [
        dict(cc=(10, 0)),
        dict(sms=114),
        dict(sms=78),
        dict(name="NVIDIA H20"),
        dict(name="NVIDIA H200"),
        dict(available=False),
    ],
)
def test_h100_evidence_does_not_enable_other_devices(monkeypatch, hardware):
    assert not route(monkeypatch, [65], [4096], **hardware)


class AdmissionObserved(Exception):
    pass


def planned_route(monkeypatch, *, q=65, captured=None, requested="auto", **features):
    route(monkeypatch, [q], [4096])
    w = fp.BatchPrefillWithPagedKVCacheWrapper.__new__(
        fp.BatchPrefillWithPagedKVCacheWrapper
    )
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _batch_invariant=None,
        _requested_backend=requested,
        _backend=captured or requested,
        _cuda_graph_backend=captured,
        _jit_module=None,
        _use_cuda_graph=captured is not None,
        _custom_mask_buf=None,
        _plan_info=object() if captured else None,
        _variant_owns_mask=False,
        _kv_layout="HND",
        device=torch.device("cpu"),
        _qo_indptr_buf=torch.tensor([0, q], dtype=torch.int32),
    )
    monkeypatch.setattr(fp, "determine_attention_backend", lambda *a, **k: "fa3")
    result = []

    def observe(backend, *args):
        result.append(backend)
        raise AdmissionObserved

    monkeypatch.setattr(fp, "_validate_paged_query_lengths", observe)
    args = dict(
        causal=True,
        q_data_type=torch.bfloat16,
        seq_lens=torch.tensor([4096], dtype=torch.int32),
        max_token_per_sequence=q,
        max_sequence_kv=4096,
    )
    args.update(features)
    with pytest.raises(AdmissionObserved):
        w.plan(
            torch.tensor([0, q], dtype=torch.int32),
            torch.tensor([0, 256], dtype=torch.int32),
            torch.arange(256, dtype=torch.int32),
            torch.tensor([16], dtype=torch.int32),
            32,
            8,
            256,
            16,
            **args,
        )
    return result[0]


@pytest.mark.parametrize("captured", [None, "fa2", "fa3"])
@pytest.mark.parametrize("q", [64, 65])
def test_real_planner_preserves_native_capture_schedules(monkeypatch, captured, q):
    expected = captured or ("fa2" if q == 64 else "cudnn")
    assert planned_route(monkeypatch, captured=captured, q=q) == expected


@pytest.mark.parametrize(
    "feature",
    [
        dict(causal=False),
        dict(window_left=127),
        dict(logits_soft_cap=1.0),
        dict(use_fp16_qk_reduction=True),
        dict(fixed_split_size=128),
        dict(disable_split_kv=True),
        dict(custom_mask=torch.ones(65 * 4096, dtype=torch.bool)),
        dict(pos_encoding_mode="ROPE_LLAMA"),
    ],
)
def test_real_planner_keeps_feature_guards(monkeypatch, feature):
    assert planned_route(monkeypatch, **feature) == "fa3"


@pytest.mark.parametrize("requested", ["fa2", "fa3", "cudnn"])
def test_explicit_backends_are_preserved(monkeypatch, requested):
    assert planned_route(monkeypatch, requested=requested) == requested


def test_no_cuda_initialization():
    assert not torch.cuda.is_initialized()
