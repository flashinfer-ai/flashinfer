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

"""Reject incompatible Q masks before touching an accepted paged plan."""

import pytest
import torch
import flashinfer.prefill as fp


class PlanCommitReached(Exception):
    pass


class ObservedWrapper(fp.BatchPrefillWithPagedKVCacheWrapper):
    def __setattr__(self, name, value):
        if self.__dict__.get("observe_writes", False):
            raise PlanCommitReached(name)
        super().__setattr__(name, value)


def make_wrapper(monkeypatch, backend, selected, captured=False):
    w = ObservedWrapper.__new__(ObservedWrapper)
    w.__dict__.update(
        _float_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _int_workspace_buffer=torch.empty(1024, dtype=torch.uint8),
        _batch_invariant=None,
        _requested_backend=backend,
        _backend=selected,
        _cuda_graph_backend=selected if captured else None,
        _jit_module=None,
        _use_cuda_graph=captured,
        _custom_mask_buf=None,
        _plan_info=object() if captured else None,
        _variant_owns_mask=False,
        _kv_layout="HND",
        device=torch.device("cpu"),
        _qo_indptr_buf=torch.tensor([0, 128], dtype=torch.int32),
        _head_dim_qk=128,
        _head_dim_vo=128,
        observe_writes=True,
    )
    monkeypatch.setattr(fp, "determine_attention_backend", lambda *a, **k: selected)
    monkeypatch.setattr(fp, "_blackwell_paged_auto_cudnn", lambda *a, **k: False)
    return w


def plan(w, qlen, **kwargs):
    w.plan(
        torch.tensor([0, 128], dtype=torch.int32),
        torch.tensor([0, 512], dtype=torch.int32),
        torch.arange(512, dtype=torch.int32),
        torch.tensor([16], dtype=torch.int32),
        16,
        2,
        64,
        16,
        causal=True,
        q_data_type=torch.bfloat16,
        seq_lens_q=qlen,
        max_token_per_sequence=128,
        max_sequence_kv=8192,
        **kwargs,
    )


@pytest.mark.parametrize(
    "backend,selected",
    [("auto", "fa2"), ("auto", "fa3"), ("fa2", "fa2"), ("trtllm-gen", "trtllm-gen")],
)
@pytest.mark.parametrize("captured", [False, True])
@pytest.mark.parametrize("device", ["cpu", "meta"])
def test_independent_query_mask_rejected_before_any_state_write(
    monkeypatch, backend, selected, captured, device
):
    w = make_wrapper(monkeypatch, backend, selected, captured)
    before = w.__dict__.copy()
    qlen = torch.tensor([1], dtype=torch.int32, device=device)
    with pytest.raises(ValueError, match="seq_lens_q"):
        plan(w, qlen)
    assert w.__dict__.keys() == before.keys()
    assert all(w.__dict__[name] is value for name, value in before.items())


@pytest.mark.parametrize("qlen", [None, torch.tensor([128], dtype=torch.int32)])
@pytest.mark.parametrize("captured", [False, True])
def test_physical_query_lengths_still_reach_plan_commit(monkeypatch, qlen, captured):
    w = make_wrapper(monkeypatch, "auto", "fa2", captured)
    with pytest.raises(PlanCommitReached):
        plan(w, qlen)


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("backend", ["cudnn", "auto"])
def test_cudnn_preserves_independent_lengths_without_readback(
    monkeypatch, device, backend
):
    w = make_wrapper(monkeypatch, backend, "fa2" if backend == "auto" else "cudnn")
    monkeypatch.setattr(fp, "_blackwell_paged_auto_cudnn", lambda *a, **k: True)
    qlen = torch.tensor([1], dtype=torch.int32, device=device)
    with pytest.raises(PlanCommitReached):
        plan(w, qlen)


def test_existing_fa2_capture_cannot_escape_mask_validation(monkeypatch):
    w = make_wrapper(monkeypatch, "auto", "fa2", captured=True)
    # Even a newly profitable shape cannot move an existing FA2 capture to cuDNN.
    monkeypatch.setattr(fp, "_blackwell_paged_auto_cudnn", lambda *a, **k: True)
    with pytest.raises(ValueError, match="seq_lens_q"):
        plan(w, torch.tensor([1], dtype=torch.int32))


def test_custom_mask_does_not_accidentally_admit_cudnn(monkeypatch):
    w = make_wrapper(monkeypatch, "auto", "fa2")
    monkeypatch.setattr(fp, "_blackwell_paged_auto_cudnn", lambda *a, **k: True)
    with pytest.raises(ValueError, match="seq_lens_q"):
        plan(
            w,
            torch.tensor([1], dtype=torch.int32),
            custom_mask=torch.ones(128 * 8192, dtype=torch.bool),
        )


def test_no_cuda_initialization():
    assert not torch.cuda.is_initialized()
