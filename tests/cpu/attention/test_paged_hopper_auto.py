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


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def dtype(request):
    return request.param


def route(
    monkeypatch,
    qs,
    ks,
    *,
    cc=(9, 0),
    sms=132,
    name="NVIDIA H100 80GB HBM3",
    available=True,
    dtype=torch.bfloat16,
    **changes,
):
    monkeypatch.setattr(fp, "get_compute_capability", lambda device: cc)
    monkeypatch.setattr(fp, "get_device_sm_count", lambda device: sms)
    monkeypatch.setattr(fp, "get_device_name", lambda device: name)
    monkeypatch.setattr(fp, "_cudnn_supports_paged_auto", lambda: available)
    args = dict(
        device=torch.device("cpu"),
        layout="HND",
        q_dtype=dtype,
        kv_dtype=dtype,
        o_dtype=dtype,
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
    dtype, monkeypatch, batch, group, hq, q, expected
):
    qs = [q] + [1] * (batch - 1)
    assert (
        route(monkeypatch, qs, [16384] * batch, hq=hq, hk=hq // group, dtype=dtype)
        is expected
    )


@pytest.mark.parametrize(
    "qs,ks,expected",
    [
        ([257], [257], False),
        ([513] * 4, [513] * 4, True),
        ([16384], [16384], True),
        ([129, 1], [32768, 1], True),
    ],
)
def test_full_prefill_and_mixed_lengths_are_included(
    dtype, monkeypatch, qs, ks, expected
):
    assert route(monkeypatch, qs, ks, dtype=dtype) is expected
    assert (
        route(
            monkeypatch,
            qs,
            ks,
            seq_lens_q=torch.tensor(qs, dtype=torch.int32),
            dtype=dtype,
        )
        is expected
    )


@pytest.mark.parametrize(
    "change",
    [
        dict(layout="NHD"),
        dict(q_dtype=torch.float32),
        dict(kv_dtype=torch.float32),
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
def test_unqualified_inputs_do_not_change_provider(dtype, monkeypatch, change):
    assert not route(monkeypatch, [65], [16384], **change, dtype=dtype)


@pytest.mark.parametrize(
    "qs,ks",
    [([0, 65], [4096, 4096]), ([65], [64]), ([16385], [32768]), ([65], [32769])],
)
def test_actual_lengths_must_stay_within_the_domain(dtype, monkeypatch, qs, ks):
    assert not route(monkeypatch, qs, ks, dtype=dtype)


class DeviceOnly:
    device = SimpleNamespace(type="cuda")

    def flatten(self):
        raise AssertionError("admission must not read device metadata")

    def tolist(self):
        raise AssertionError("admission must not copy metadata")


@pytest.mark.parametrize("field", ["kv_lens", "seq_lens_q", "qo_indptr_host"])
def test_no_admission_readback(dtype, monkeypatch, field):
    assert not route(monkeypatch, [65], [16384], **{field: DeviceOnly()}, dtype=dtype)


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
def test_h100_evidence_does_not_enable_other_devices(dtype, monkeypatch, hardware):
    assert not route(monkeypatch, [65], [16384], **hardware, dtype=dtype)


class AdmissionObserved(Exception):
    pass


def planned_route(
    monkeypatch,
    *,
    q=65,
    captured=None,
    requested="auto",
    dtype=torch.bfloat16,
    native="fa3",
    **features,
):
    route(monkeypatch, [q], [16384], dtype=dtype)
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
    monkeypatch.setattr(fp, "determine_attention_backend", lambda *a, **k: native)
    result = []

    def observe(backend, *args):
        result.append(backend)
        raise AdmissionObserved

    monkeypatch.setattr(fp, "_validate_paged_query_lengths", observe)
    args = dict(
        causal=True,
        q_data_type=dtype,
        seq_lens=torch.tensor([16384], dtype=torch.int32),
        max_token_per_sequence=q,
        max_sequence_kv=16384,
    )
    args.update(features)
    with pytest.raises(AdmissionObserved):
        w.plan(
            torch.tensor([0, q], dtype=torch.int32),
            torch.tensor([0, 1024], dtype=torch.int32),
            torch.arange(1024, dtype=torch.int32),
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
def test_real_planner_preserves_native_capture_schedules(
    dtype, monkeypatch, captured, q
):
    expected = captured or ("fa2" if q == 64 else "cudnn")
    assert planned_route(monkeypatch, captured=captured, q=q, dtype=dtype) == expected


@pytest.mark.parametrize(
    "feature",
    [
        dict(window_left=127),
        dict(logits_soft_cap=1.0),
        dict(use_fp16_qk_reduction=True),
        dict(fixed_split_size=128),
        dict(disable_split_kv=True),
        dict(custom_mask=torch.ones(65 * 16384, dtype=torch.bool)),
        dict(pos_encoding_mode="ROPE_LLAMA"),
    ],
)
def test_real_planner_keeps_feature_guards(dtype, monkeypatch, feature):
    assert planned_route(monkeypatch, **feature, dtype=dtype) == "fa3"


@pytest.mark.parametrize("requested", ["fa2", "fa3", "cudnn"])
def test_explicit_backends_are_preserved(dtype, monkeypatch, requested):
    assert planned_route(monkeypatch, requested=requested, dtype=dtype) == requested


def test_no_cuda_initialization(
    dtype,
):
    assert not torch.cuda.is_initialized()


@pytest.mark.parametrize("kv,expected", [(8097, False), (8098, True)])
def test_actual_work_crossing(dtype, monkeypatch, kv, expected):
    assert route(monkeypatch, [65], [kv], dtype=dtype) is expected


def test_actual_work_uses_paired_lengths_not_declared_maxima(dtype, monkeypatch):
    # Both graphs declare B2/Q129/KV4096, but only one has enough actual work.
    assert route(monkeypatch, [129, 129], [4096, 4096], dtype=dtype)
    assert not route(monkeypatch, [129, 1], [4096, 1], dtype=dtype)


@pytest.mark.parametrize("field", ["q_dtype", "kv_dtype", "o_dtype"])
def test_mixed_precision_is_not_admitted(dtype, monkeypatch, field):
    other = torch.float16 if dtype == torch.bfloat16 else torch.bfloat16
    assert not route(monkeypatch, [65], [16384], dtype=dtype, **{field: other})


@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("captured", [None, "fa2", "fa3"])
def test_real_planner_mask_transfer_preserves_native_capture(
    dtype, monkeypatch, causal, captured
):
    assert planned_route(
        monkeypatch, causal=causal, captured=captured, dtype=dtype
    ) == (captured or "cudnn")


@pytest.mark.parametrize("kv,expected", [(8065, False), (8066, True)])
def test_noncausal_actual_work_crossing(dtype, monkeypatch, kv, expected):
    assert route(monkeypatch, [65], [kv], dtype=dtype, causal=False) is expected


def test_noncausal_actual_work_does_not_use_triangular_estimate(dtype, monkeypatch):
    assert route(monkeypatch, [65], [8066], dtype=dtype, causal=False)
    assert not route(monkeypatch, [65], [8066], dtype=dtype, causal=True)


def test_noncausal_work_uses_paired_lengths(dtype, monkeypatch):
    assert route(monkeypatch, [129, 129], [4096, 4096], dtype=dtype, causal=False)
    assert not route(monkeypatch, [129, 65], [129, 4096], dtype=dtype, causal=False)


def test_noncausal_blackwell_does_not_inherit_hopper_admission(dtype, monkeypatch):
    assert planned_route(monkeypatch, causal=False, native="fa2", dtype=dtype) == "fa2"
    # The helper also independently rejects Blackwell.
    assert not route(monkeypatch, [65], [16384], dtype=dtype, causal=False, cc=(10, 0))
