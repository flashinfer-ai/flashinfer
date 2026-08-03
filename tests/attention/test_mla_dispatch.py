"""Minimal safety sentinels for MLA backend dispatch."""

import pytest
import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAKVCache, MLAQuery
from flashinfer.mla._batch_mla import _auto_policy
from flashinfer.mla._batch_mla._backends import cutlass_backend, fa2_backend


def _plan_kwargs():
    return {
        "qo_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "kv_indptr": torch.tensor([0, 1], dtype=torch.int32),
        "kv_indices": torch.tensor([0], dtype=torch.int32),
        "kv_len_arr": torch.tensor([1], dtype=torch.int32),
        "num_heads": 1,
        "head_dim_ckv": 2,
        "head_dim_kpe": 1,
        "page_size": 1,
        "causal": False,
        "sm_scale": 1.0,
        "q_data_type": torch.float32,
        "kv_data_type": torch.float32,
        "kv_layout": "independent-split",
    }


class _SuccessfulBackend:
    def run_from_wrapper(self, **_kwargs):
        return "selected-result"


def _run(wrapper):
    return wrapper.run(
        query=MLAQuery.split(torch.empty(1, 1, 2), torch.empty(1, 1, 1)),
        kv=MLAKVCache.split(torch.empty(1, 1, 2), torch.empty(1, 1, 1)),
    )


def test_invalid_input_does_not_fallback(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _auto_policy, "rank_auto_backend_candidates", lambda _device: ("fa2", "cutlass")
    )
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda _cls, _args: calls.append("fa2")
            or (_ for _ in ()).throw(ValueError("invalid input"))
        ),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda _cls, _args: calls.append("cutlass")),
    )

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    with pytest.raises(ValueError, match="invalid input"):
        wrapper.plan(**_plan_kwargs())

    assert calls == ["fa2"]


def test_typed_unsupportedness_falls_back(monkeypatch):
    calls = []
    monkeypatch.setattr(
        _auto_policy, "rank_auto_backend_candidates", lambda _device: ("fa2", "cutlass")
    )
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda _cls, _args: calls.append("fa2")
            or (_ for _ in ()).throw(_BackendPlanUnsupportedError("unsupported"))
        ),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(
            lambda _cls, _args: calls.append("cutlass") or _SuccessfulBackend()
        ),
    )

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="auto")
    wrapper.plan(**_plan_kwargs())

    assert calls == ["fa2", "cutlass"]
    assert wrapper.resolved_backend == "cutlass"
    assert _run(wrapper) == "selected-result"


def test_explicit_unsupported_backend_does_not_substitute(monkeypatch):
    calls = []
    monkeypatch.setattr(
        fa2_backend._BatchMLAPagedAttentionFa2Backend,
        "plan_from_wrapper",
        classmethod(
            lambda _cls, _args: calls.append("fa2")
            or (_ for _ in ()).throw(_BackendPlanUnsupportedError("unsupported"))
        ),
    )
    monkeypatch.setattr(
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        "plan_from_wrapper",
        classmethod(lambda _cls, _args: calls.append("cutlass")),
    )

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="fa2")
    with pytest.raises(_BackendPlanUnsupportedError, match="unsupported"):
        wrapper.plan(**_plan_kwargs())

    assert calls == ["fa2"]
