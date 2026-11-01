# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only scope checks for the bounded SM107 MLA split policy."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass", minversion="4.7.0")

import flashinfer.attention.prims_ts.mla_decode as mla_decode_module
from flashinfer.attention.prims_ts import _hardware
from flashinfer.attention.prims_ts.kernels.mla_decode import kernel_policy


_PLAN = dict(
    device_index=0,
    batch_size=1,
    num_qo_heads=96,
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    page_size=128,
    max_kv_len=1024,
    q_dtype_key="float8_e4m3fn",
    kv_dtype_key="float8_e4m3fn",
    out_dtype_key="bfloat16",
    mask_type="causal",
    seq_len_q=8,
    store_softmax_stats=True,
    use_kv_bounds=True,
)


@pytest.fixture
def resolver(monkeypatch):
    """Use real capability/configuration code without querying or launching a GPU."""

    hardware = SimpleNamespace(capability=(10, 7), multi_processor_count=212)
    monkeypatch.setattr(mla_decode_module.torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(
        mla_decode_module.torch.cuda,
        "get_device_capability",
        lambda _: hardware.capability,
    )
    monkeypatch.setattr(
        mla_decode_module.torch.cuda, "get_device_properties", lambda _: hardware
    )
    monkeypatch.setattr(
        mla_decode_module, "get_max_active_clusters", lambda size: 212 // size
    )
    monkeypatch.setattr(_hardware, "get_max_active_clusters", lambda size: 212 // size)
    monkeypatch.setattr(
        mla_decode_module,
        "get_device_sm_count",
        lambda _: hardware.multi_processor_count,
    )

    def resolve(**changes):
        return mla_decode_module._resolve_mla_decode_launch_spec.__wrapped__(
            **(_PLAN | changes)
        )

    return resolve, hardware


def test_attention_ts_mla_sm107_single_split_policy(resolver):
    resolve, _ = resolver
    spec = resolve()
    policy = dict(spec.policy)
    assert policy["source"] == "auto"
    assert policy["kernel"] == "throughput_2cta"
    assert policy["split_kv"] == spec.split_kv == 1
    assert policy["split_capacity"] == spec.kernel.reduction_split_capacity == 1
    assert policy["separate_reducer_impl"] == "none"
    assert not policy["use_persistent_scheduler"]
    assert not policy["use_clc_dynamic_persistent_scheduler"]
    assert spec.kernel_workspace_bytes == 0
    assert spec.kernel.static_seq_len_k is None
    assert not spec.kernel.is_var_split_kv
    assert spec.kernel.static_split_kv == 1
    layout = mla_decode_module._make_mla_workspace_layout(
        spec.kernel_workspace_bytes, 1, 96, 8, include_lse=False
    )
    assert layout.total_bytes == 0


def test_attention_ts_mla_sm107_single_split_preserves_explicit_policy(
    resolver, monkeypatch
):
    resolve, hardware = resolver
    monkeypatch.setattr(
        kernel_policy,
        "resolve_mla_kernel_policy",
        lambda *args, **kwargs: ("throughput_2cta", "explicit"),
    )
    actual = resolve()
    hardware.capability = (10, 0)
    ordinary = resolve()
    assert dict(actual.policy)["source"] == "explicit"
    assert actual.split_kv > 1
    assert actual.policy == ordinary.policy
    assert actual.kernel_workspace_bytes == ordinary.kernel_workspace_bytes
    assert actual.kernel.compile_signature() == ordinary.kernel.compile_signature()


@pytest.mark.parametrize(
    "changes,hardware_changes",
    [
        pytest.param({}, {"capability": value}, id=f"sm{value[0]}{value[1]}")
        for value in ((10, 0), (10, 3), (11, 0))
    ]
    + [
        pytest.param({}, {"multi_processor_count": value}, id=f"sms{value}")
        for value in (132, 210, 214)
    ]
    + [
        pytest.param({field: value}, {}, id=f"{field}-{value}")
        for field, values in (
            ("batch_size", (2, 8)),
            ("num_qo_heads", (8, 64, 95, 128)),
            ("seq_len_q", (1, 6, 7, 9)),
            ("page_size", (32, 64)),
            ("max_kv_len", (640, 1023, 1025, 5120)),
            ("out_dtype_key", ("float8_e4m3fn",)),
            ("mask_type", ("dense",)),
            ("store_softmax_stats", (False,)),
            ("use_kv_bounds", (False,)),
        )
        for value in values
    ]
    + [
        pytest.param(
            {"q_dtype_key": "bfloat16", "kv_dtype_key": "bfloat16"}, {}, id="bf16"
        ),
        pytest.param(
            dict(
                batch_size=8,
                seq_len_q=6,
                max_kv_len=5120,
                store_softmax_stats=False,
                use_kv_bounds=False,
            ),
            {},
            id="dep-b8-q6-near-capacity",
        ),
    ],
)
def test_attention_ts_mla_sm107_single_split_does_not_change_other_plans(
    resolver, changes, hardware_changes
):
    resolve, hardware = resolver
    for name, value in hardware_changes.items():
        setattr(hardware, name, value)
    actual = resolve(**changes)
    # Holding occupancy fixed isolates the bounded rule from the ordinary
    # architecture-independent family and split heuristics.
    hardware.capability = (10, 0)
    ordinary = resolve(**changes)
    assert actual.policy == ordinary.policy
    assert actual.split_kv == ordinary.split_kv
    assert actual.kernel_workspace_bytes == ordinary.kernel_workspace_bytes
    assert actual.kernel.compile_signature() == ordinary.kernel.compile_signature()


@pytest.mark.parametrize(
    "changes,error,match",
    (
        ({"kv_lora_rank": 256}, NotImplementedError, "kv_lora_rank"),
        ({"qk_rope_head_dim": 32}, NotImplementedError, "qk_rope_head_dim"),
        ({"kv_dtype_key": "bfloat16"}, ValueError, "one QKV dtype"),
        ({"out_dtype_key": "float16"}, NotImplementedError, "dtype key"),
        ({"page_size": 256}, NotImplementedError, "shape/features"),
    ),
)
def test_attention_ts_mla_sm107_single_split_preserves_validation(
    resolver, changes, error, match
):
    resolve, _ = resolver
    with pytest.raises(error, match=match):
        resolve(**changes)
