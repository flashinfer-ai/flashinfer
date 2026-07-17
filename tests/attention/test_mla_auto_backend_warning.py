"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0
"""

import warnings
from unittest.mock import patch

import pytest
import torch

from flashinfer.mla import BatchMLAPagedAttentionWrapper
from flashinfer.mla._batch_mla import _core as batch_mla_core


WARN_TAG = "not Blackwell-native"


class _NoopBackend:
    def __init__(self, before_metadata_commit):
        self._before_metadata_commit = before_metadata_commit

    def plan(self, **kwargs):
        if self._before_metadata_commit is not None:
            self._before_metadata_commit()


def _fresh_state():
    BatchMLAPagedAttentionWrapper._blackwell_auto_fallback_warned = False


def _make(buf, backend, monkeypatch):
    monkeypatch.setattr(
        batch_mla_core, "determine_mla_backend", lambda device: "fa2"
    )
    monkeypatch.setattr(
        batch_mla_core,
        "_BatchMLAPagedAttentionFa2Backend",
        lambda *args: _NoopBackend(args[-1]),
    )
    monkeypatch.setattr(
        batch_mla_core,
        "_BatchMLAPagedAttentionFa3Backend",
        lambda *args: _NoopBackend(args[-1]),
    )
    monkeypatch.setattr(
        batch_mla_core,
        "_BatchMLAPagedAttentionCutlassBackend",
        lambda *args: _NoopBackend(None),
    )

    qo_indptr = torch.tensor([0, 1], dtype=torch.int32, device=buf.device)
    kv_indptr = torch.tensor([0, 1], dtype=torch.int32, device=buf.device)
    kv_indices = torch.tensor([0], dtype=torch.int32, device=buf.device)
    kv_len_arr = torch.tensor([1], dtype=torch.int32, device=buf.device)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        wrapper = BatchMLAPagedAttentionWrapper(buf, backend=backend)
        wrapper.plan(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            num_heads=1,
            head_dim_ckv=512,
            head_dim_kpe=64,
            page_size=1,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        return [str(x.message) for x in w if WARN_TAG in str(x.message)]


@pytest.fixture
def buf():
    _fresh_state()
    return torch.empty(8, dtype=torch.int8)


@patch("flashinfer.mla._batch_mla._core.get_compute_capability", return_value=(10, 0))
def test_auto_warns_once_on_blackwell(_cc, buf, monkeypatch):
    assert len(_make(buf, "auto", monkeypatch)) == 1
    assert len(_make(buf, "auto", monkeypatch)) == 0  # one-time


@patch("flashinfer.mla._batch_mla._core.get_compute_capability", return_value=(10, 0))
def test_explicit_backend_does_not_warn(_cc, buf, monkeypatch):
    assert _make(buf, "fa2", monkeypatch) == []
    assert _make(buf, "cutlass", monkeypatch) == []


@patch("flashinfer.mla._batch_mla._core.get_compute_capability", return_value=(9, 0))
def test_no_warn_on_hopper(_cc, buf, monkeypatch):
    assert _make(buf, "auto", monkeypatch) == []


def test_warning_as_error_stops_before_fa_commit_and_preserves_plan(
    buf, monkeypatch
):
    events = []
    backends = []

    class CommitRecordingBackend:
        def __init__(self, before_metadata_commit):
            self._before_metadata_commit = before_metadata_commit
            self.committed = False

        def plan(self, **kwargs):
            events.append("planned")
            if self._before_metadata_commit is not None:
                self._before_metadata_commit()
            self.committed = True
            events.append("committed")

    def make_backend(*args):
        backend = CommitRecordingBackend(args[-1])
        backends.append(backend)
        return backend

    monkeypatch.setattr(
        batch_mla_core, "determine_mla_backend", lambda device: "fa2"
    )
    monkeypatch.setattr(
        batch_mla_core, "_BatchMLAPagedAttentionFa2Backend", make_backend
    )
    monkeypatch.setattr(
        batch_mla_core,
        "get_compute_capability",
        lambda device: (9, 0) if len(backends) == 1 else (10, 0),
    )
    plan_args = dict(
        qo_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indptr=torch.tensor([0, 1], dtype=torch.int32),
        kv_indices=torch.tensor([0], dtype=torch.int32),
        kv_len_arr=torch.tensor([1], dtype=torch.int32),
        num_heads=1,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=1,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
    )
    wrapper = BatchMLAPagedAttentionWrapper(buf, backend="auto")
    wrapper.plan(**plan_args)
    original_state = (
        wrapper._backend_impl,
        wrapper._selected_backend,
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        with pytest.raises(UserWarning, match=WARN_TAG):
            wrapper.plan(**plan_args)

    assert events == ["planned", "committed", "planned"]
    assert backends[0].committed
    assert not backends[1].committed
    assert (
        wrapper._backend_impl,
        wrapper._selected_backend,
        wrapper._csr_plan_metadata,
        wrapper._dense_plan_metadata,
    ) == original_state
    assert BatchMLAPagedAttentionWrapper._blackwell_auto_fallback_warned
