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

"""Public warm-call contracts, with real plan keys and CPU-only graph submission."""

from types import SimpleNamespace

import pytest
import torch

import flashinfer.prefill as fp
from flashinfer.cudnn import prefill as cp

pytestmark = pytest.mark.cpu_only


@pytest.fixture
def paged_call(monkeypatch):
    builds, executions, legacy = [], [], []
    monkeypatch.setattr(cp, "CUDNN_AVAILABLE", True)
    monkeypatch.setattr(cp, "_cudnn_supports_shape_override", lambda: False)
    monkeypatch.setattr(cp, "_graph_workspace_size", lambda *a: 0)
    monkeypatch.setattr(cp, "supports_native_cudnn_log2", lambda *a: False)
    monkeypatch.setattr(fp, "device_support_pdl", lambda device: False)

    def build(**kwargs):
        builds.append(kwargs)
        return SimpleNamespace(get_workspace_size=lambda: 0), []

    def execute(self, q, out, lse, workspace, bindings, kwargs, base, tensor_uids=None):
        pack = (
            dict(zip(tensor_uids, bindings, strict=True))
            if tensor_uids is not None
            else bindings
        )
        executions.append((pack, workspace, base))
        # This tests binding, not attention arithmetic or device execution.
        out.fill_(17)
        if lse is not None:
            lse.fill_(2)
        return out, lse

    def fallback(*args, **kwargs):
        legacy.append((args, kwargs))
        kwargs["out"].fill_(23)
        return kwargs["out"]

    monkeypatch.setattr(cp, "_build_prefill_graph", build, raising=False)
    monkeypatch.setattr(cp.CudnnPrefillGraph, "_execute", execute)
    monkeypatch.setattr(fp, "cudnn_batch_prefill_with_kv_cache", fallback)

    def make(layout="HND", dtype=torch.bfloat16, ordered=True):
        monkeypatch.setattr(cp, "supports_ordered_cudnn_execution", lambda *a: ordered)
        q = torch.empty(3, 4, 8, dtype=dtype)
        k = torch.empty(4, 2, 4, 8, dtype=dtype)
        v = torch.empty_like(k)
        lens = torch.tensor([1, 2], dtype=torch.int32).reshape(2, 1, 1, 1)
        kv_lens = torch.tensor([4, 3], dtype=torch.int32).reshape(2, 1, 1, 1)
        prefix = torch.tensor([0, 1, 3], dtype=torch.int32)
        table = torch.zeros(2, 1, dtype=torch.int32)
        metadata = cp._PrefillMetadata(
            2,
            4,
            True,
            True,
            actual_seq_lens_q=lens,
            actual_seq_lens_kv=kv_lens,
            batch_offsets_q=prefix,
            batch_offsets_o=prefix,
            batch_offsets_stats=prefix,
            block_tables=table,
        )
        plan = cp._CudnnPrefillPlan.prepare(metadata, dtype, q.device)
        workspace = torch.empty(16, dtype=torch.uint8)
        history = []
        prepared = cp._reuse_cudnn_prefill_preparation(
            history, q, k, v, 0.5, workspace, plan=plan, return_lse=False
        )
        w = fp.BatchPrefillWithPagedKVCacheWrapper.__new__(
            fp.BatchPrefillWithPagedKVCacheWrapper
        )
        w.__dict__.update(
            _backend="cudnn",
            _kv_layout=layout,
            _batch_invariant=None,
            _cached_q_data_type=dtype,
            _cached_kv_data_type=dtype,
            _cached_o_data_type=dtype,
            _cudnn_lse_unservable=False,
            _qo_indptr_last=3,
            _use_token_head_sf=False,
            _window_left=-1,
            _logits_soft_cap=0.0,
            _sm_scale=0.5,
            _rope_scale=None,
            _rope_theta=None,
            _head_dim_qk=8,
            _head_dim_vo=8,
            _causal=True,
            _custom_mask_buf=None,
            _variant_owns_mask=False,
            _prefix_len_ptr=None,
            _eagle_verify_fp8kv_sm90_state=None,
            _seq_lens_q=lens,
            _seq_lens_kv=kv_lens,
            _batch_size=2,
            _max_q_len=2,
            _max_kv_len=4,
            _cudnn_plan=plan,
            _cudnn_prepared=prepared,
            _cudnn_prepared_cache=history,
            _float_workspace_buffer=workspace,
            _qo_indptr_buf=prefix,
            _block_tables=table,
        )
        if layout == "NHD":
            k, v = k.transpose(1, 2), v.transpose(1, 2)
        return w, q, k, v

    return SimpleNamespace(
        make=make, builds=builds, executions=executions, legacy=legacy
    )


@pytest.mark.parametrize("layout", ["HND", "NHD"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("ordered", [False, True])
def test_paged_warm_call_rebinds_every_buffer(paged_call, layout, dtype, ordered):
    w, q, k, v = paged_call.make(layout, dtype, ordered)
    prepared = w._cudnn_prepared
    for _ in range(2):
        q, k, v = q.clone(), k.clone(), v.clone()
        out = torch.empty_like(q)
        w._float_workspace_buffer = torch.empty(32, dtype=torch.uint8)
        assert w.run(q, (k, v), out=out) is out
        pack, workspace, _ = paged_call.executions[-1]
        for uid, tensor in [
            (cp.UIDs.Q_UID, q),
            (cp.UIDs.K_UID, k),
            (cp.UIDs.V_UID, v),
            (cp.UIDs.O_UID, out),
        ]:
            assert pack[uid.value].data_ptr() == tensor.data_ptr()
        assert workspace is w._float_workspace_buffer
        assert torch.all(out == 17)
        assert w._cudnn_prepared is prepared
    assert len(paged_call.builds) == 1 and not paged_call.legacy


@pytest.mark.parametrize(
    "field", ["q_stride", "k_stride", "v_stride", "q_heads", "kv_pages", "scale"]
)
def test_paged_warm_call_checks_current_signature(paged_call, field):
    w, q, k, v = paged_call.make()
    previous = w._cudnn_prepared
    if field == "q_stride":
        # Same object, new metadata: an identity cache would miss this.
        q.as_strided_(q.shape, (32, 1, 4))
    elif field == "k_stride":
        k.as_strided_(k.shape, (64, 1, 16, 2))
    elif field == "v_stride":
        v.as_strided_(v.shape, (64, 1, 16, 2))
    elif field == "q_heads":
        q = torch.empty(3, 8, 8, dtype=q.dtype)
    elif field == "kv_pages":
        k, v = k[:3], v[:3]
    else:
        w._sm_scale = 0.25
    out = torch.empty(q.shape[:-1] + (v.shape[-1],), dtype=q.dtype)
    w.run(q, (k, v), out=out)
    assert w._cudnn_prepared is not previous
    assert len(paged_call.builds) == 2 and len(paged_call.executions) == 1


@pytest.mark.parametrize(
    "fault",
    [
        "out_shape",
        "out_dtype",
        "out_device",
        "out_stride",
        "q_rows",
        "q_dtype",
        "k_dtype",
        "window",
        "lse_base",
        "lse_layout",
    ],
)
def test_paged_warm_call_preserves_validation(paged_call, fault):
    w, q, k, v = paged_call.make()
    out, kwargs = torch.empty_like(q), {}
    if fault == "out_shape":
        out = out[:2]
    elif fault == "out_dtype":
        out = out.float()
    elif fault == "out_device":
        out = torch.empty_like(out, device="meta")
    elif fault == "out_stride":
        out = torch.empty(3, 4, 16, dtype=q.dtype)[..., ::2]
    elif fault == "q_rows":
        q = q[:2]
    elif fault == "q_dtype":
        q = q.float()
    elif fault == "k_dtype":
        k = k.float()
    elif fault == "window":
        kwargs["window_left"] = 7
    else:
        kwargs[fault] = "invalid"
    with pytest.raises((ValueError, AssertionError)):
        w.run(q, (k, v), out=out, **kwargs)
    assert not paged_call.executions and not paged_call.legacy


@pytest.mark.parametrize(
    "option", ["sinks", "kv_cache_sf", "skip_softmax_threshold_scale_factor"]
)
def test_paged_warm_call_keeps_unsupported_options(paged_call, option):
    w, q, k, v = paged_call.make()
    arg = (k, v) if option == "kv_cache_sf" else torch.ones(4)
    with pytest.raises(NotImplementedError):
        w.run(q, (k, v), out=torch.empty_like(q), **{option: arg})
    assert not paged_call.executions and not paged_call.legacy


@pytest.mark.parametrize("field", ["q_scale", "k_scale", "v_scale", "causal", "window"])
def test_paged_warm_call_preserves_runtime_fallback(paged_call, field):
    w, q, k, v = paged_call.make()
    kwargs = {}
    if field == "causal":
        w._causal = False
    elif field == "window":
        w._window_left = 1
    else:
        kwargs[field] = 0.5
    out = torch.empty_like(q)
    assert w.run(q, (k, v), out=out, **kwargs) is out
    assert len(paged_call.legacy) == 1 and not paged_call.executions


@pytest.mark.parametrize("lse_base", ["ln", "log2"])
@pytest.mark.parametrize("lse_layout", ["NH", "HN"])
def test_paged_warm_call_preserves_stats_path(paged_call, lse_base, lse_layout):
    w, q, k, v = paged_call.make()
    out = torch.empty_like(q)
    result, stats = w.run(
        q, (k, v), out=out, return_lse=True, lse_base=lse_base, lse_layout=lse_layout
    )
    assert result is out and tuple(stats.shape) == (
        (3, 4) if lse_layout == "NH" else (4, 3)
    )
    assert len(paged_call.executions) == 1 and not paged_call.legacy
    assert w._cudnn_prepared.return_lse
