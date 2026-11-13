# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""The standard paged adapters must not silently discard attention sinks."""

import inspect
from types import SimpleNamespace

import pytest
import torch

import flashinfer.prefill as fp


def make_adapter(monkeypatch, backend):
    calls = []

    def native(*args, **kwargs):
        calls.append((args, kwargs))

    module = SimpleNamespace(plan=lambda: None, ragged_run=native, paged_run=native)
    spec = SimpleNamespace(build_and_load=lambda: module)
    monkeypatch.setattr(fp, "register_custom_op", lambda *a, **k: lambda f: f)
    monkeypatch.setattr(fp, "register_fake_op", lambda *a, **k: lambda f: f)
    monkeypatch.setattr(fp, "get_batch_prefill_uri", lambda *a: "paged_sink_test")
    for name in (
        "_gen_batch_prefill_primary_module",
        "_gen_batch_prefill_independent_paged_module",
        "gen_batch_prefill_module",
    ):
        monkeypatch.setattr(fp, name, lambda *a: spec)
    monkeypatch.setattr(fp, "get_trtllm_gen_prefill_module", lambda: module)
    return fp.get_batch_prefill_module.__wrapped__(backend).paged_run, calls


def arguments(adapter):
    args = {
        name: None
        for name, parameter in inspect.signature(adapter).parameters.items()
        if parameter.default is inspect.Parameter.empty
    }
    kv = torch.ones((1, 2, 1, 2))
    args.update(
        q=torch.ones((1, 2, 2)),
        paged_k_cache=kv,
        paged_v_cache=kv,
        rope_scale=1.0,
        rope_theta=1e4,
        workspace_size=1024,
        num_qo_heads=2,
        num_kv_heads=2,
        block_tables=torch.zeros((1, 1), dtype=torch.int32),
        kv_lens_buffer=torch.ones(1, dtype=torch.int32),
        page_size=1,
        max_q_len=1,
        max_kv_len=1,
        batch_size=1,
        cum_seq_lens_q=torch.tensor([0, 1], dtype=torch.int32),
        cum_seq_lens_kv=torch.tensor([0, 1], dtype=torch.int32),
        enable_pdl=False,
        multi_ctas_kv_counter_buffer=torch.zeros(1, dtype=torch.int32),
    )
    return args


@pytest.mark.parametrize("backend", ["fa2", "fa3"])
def test_standard_paged_adapter_rejects_sinks_before_launch(monkeypatch, backend):
    adapter, calls = make_adapter(monkeypatch, backend)
    with pytest.raises(NotImplementedError, match="sinks"):
        adapter(**arguments(adapter), sinks=torch.tensor([0.3, -0.7]))
    assert calls == []


@pytest.mark.parametrize("backend", ["fa2", "fa3", "trtllm-gen"])
def test_standard_paged_adapter_without_sinks_still_launches(monkeypatch, backend):
    adapter, calls = make_adapter(monkeypatch, backend)
    adapter(**arguments(adapter))
    assert len(calls) == 1


def test_trtllm_paged_adapter_forwards_sinks(monkeypatch):
    adapter, calls = make_adapter(monkeypatch, "trtllm-gen")
    sinks = torch.tensor([0.3, -0.7])
    adapter(**arguments(adapter), sinks=sinks)
    assert len(calls) == 1
    assert calls[0][1]["sinks"] is sinks


def test_no_cuda_initialization():
    assert not torch.cuda.is_initialized()
