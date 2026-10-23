"""CPU routing checks; capability/modules are mocked, not XQA arithmetic."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

import flashinfer.decode as decode


@pytest.fixture
def make_wrapper(monkeypatch):
    original_empty = torch.empty

    def cpu_empty(*args, **kwargs):
        kwargs.pop("pin_memory", None)
        return original_empty(*args, **kwargs)

    monkeypatch.setattr(torch, "empty", cpu_empty)
    monkeypatch.setattr(torch.cuda, "device", lambda *args: nullcontext())
    monkeypatch.setattr(decode, "get_compute_capability", lambda device: (9, 0))
    monkeypatch.setattr(
        decode.BatchDecodeWithPagedKVCacheWrapper,
        "_auto_prefers_cudnn",
        lambda *args, **kwargs: False,
    )
    monkeypatch.setattr(
        decode,
        "get_batch_prefill_module",
        lambda *args, **kwargs: SimpleNamespace(
            plan=lambda *args: [0], workspace_size=lambda *args: (32, 16)
        ),
    )

    def make(batch=16, kvlen=1024, graph=True, backend="auto", layout="NHD"):
        pages = (kvlen + 15) // 16
        indptr = torch.arange(batch + 1, dtype=torch.int32) * pages
        indices = torch.arange(batch * pages, dtype=torch.int32)
        last = torch.full((batch,), (kvlen - 1) % 16 + 1, dtype=torch.int32)
        kwargs = {}
        if graph:
            kwargs = dict(
                use_cuda_graph=True,
                paged_kv_indptr_buffer=indptr.clone(),
                paged_kv_indices_buffer=indices.clone(),
                paged_kv_last_page_len_buffer=last.clone(),
            )
        wrapper = decode.BatchDecodeWithPagedKVCacheWrapper(
            torch.zeros(128, dtype=torch.uint8),
            layout,
            use_tensor_cores=True,
            backend=backend,
            **kwargs,
        )

        def plan(query=False, heads=(32, 8), **overrides):
            options = dict(q_len_per_req=8, q_data_type=torch.float16)
            options.update(overrides)
            method = wrapper.workspace_size if query else wrapper.plan
            return method(indptr, indices, last, *heads, 128, 16, **options)

        return wrapper, plan, indices

    return make


@pytest.mark.parametrize(
    "batch,q,kv,expected",
    [
        (4, 2, 8192, False),
        (4, 8, 1024, False),
        (4, 8, 8192, True),
        (16, 2, 8192, False),
        (16, 4, 1024, True),
        (16, 8, 1024, True),
        (16, 8, 8192, True),
        (64, 2, 8192, False),
        (64, 8, 1024, False),
        (64, 8, 8192, True),
    ],
)
def test_xqa_mtp_policy_matches_selected_reported_shapes(
    make_wrapper, batch, q, kv, expected
):
    wrapper, plan, _ = make_wrapper(batch=batch, kvlen=kv)
    plan(q_len_per_req=q)
    assert wrapper._use_xqa_mtp_decode == expected
    assert wrapper.resolved_backend == ("xqa" if expected else "fa2")


@pytest.mark.parametrize(
    "kwargs", [{"graph": False}, {"backend": "fa2"}, {"layout": "HND"}]
)
def test_xqa_mtp_policy_preserves_other_modes(make_wrapper, kwargs):
    wrapper, plan, _ = make_wrapper(**kwargs)
    plan()
    assert not wrapper._use_xqa_mtp_decode


@pytest.mark.parametrize("cc", [(8, 6), (10, 0), (12, 0)])
def test_xqa_mtp_policy_does_not_extrapolate_devices(make_wrapper, monkeypatch, cc):
    wrapper, plan, _ = make_wrapper()
    monkeypatch.setattr(decode, "get_compute_capability", lambda device: cc)
    plan()
    assert not wrapper._use_xqa_mtp_decode


@pytest.mark.parametrize(
    "override",
    [
        {"q_data_type": torch.bfloat16},
        {"pos_encoding_mode": "ROPE_LLAMA"},
        {"logits_soft_cap": 1.0},
        {"fixed_split_size": 1},
        {"disable_split_kv": True},
        {"o_data_type": torch.bfloat16},
        {"window_left": 128},
        {"heads": (16, 8)},
    ],
)
def test_xqa_mtp_policy_preserves_unsupported_features(make_wrapper, override):
    wrapper, plan, _ = make_wrapper()
    plan(**override)
    assert not wrapper._use_xqa_mtp_decode


@pytest.mark.parametrize(
    "batch,kv,q",
    [
        (8, 8192, 8),
        (32, 8192, 8),
        (16, 32768, 8),
        (16, 1025, 8),
        (16, 1024, 1),
        (16, 1024, 16),
    ],
)
def test_xqa_mtp_policy_does_not_interpolate(make_wrapper, batch, kv, q):
    wrapper, plan, _ = make_wrapper(batch=batch, kvlen=kv)
    plan(q_len_per_req=q)
    assert not wrapper._use_xqa_mtp_decode


def test_xqa_mtp_policy_rejects_ragged_lengths(make_wrapper):
    wrapper, plan, _ = make_wrapper()
    lengths = torch.full((16,), 1024, dtype=torch.int32)
    lengths[0] = 1023
    plan(seq_lens=lengths)
    assert not wrapper._use_xqa_mtp_decode


@pytest.mark.parametrize("first_disabled", [False, True])
def test_xqa_mtp_graph_route_is_frozen(make_wrapper, first_disabled):
    wrapper, plan, _ = make_wrapper()
    plan(disable_split_kv=first_disabled)
    previous = wrapper._use_xqa_mtp_decode
    with pytest.raises(ValueError, match="XQA.*frozen"):
        plan(disable_split_kv=not first_disabled)
    assert wrapper._use_xqa_mtp_decode == previous


def test_xqa_mtp_graph_replan_updates_mapping_in_place(make_wrapper):
    wrapper, plan, indices = make_wrapper()
    plan()
    fields = ("_block_tables", "_kv_lens_buffer", "_xqa_mtp_causal_mask")
    before = {name: getattr(wrapper, name).data_ptr() for name in fields}
    indices.copy_(indices.flip(0))
    plan()
    assert {name: getattr(wrapper, name).data_ptr() for name in fields} == before
    assert wrapper._block_tables[0, 0] == indices[0]


def test_xqa_mtp_fast_plan_rechecks_previous_fallback(make_wrapper):
    wrapper, plan, indices = make_wrapper()
    plan(disable_split_kv=True)
    assert not wrapper._use_xqa_mtp_decode
    with pytest.raises(ValueError, match="XQA.*frozen"):
        decode.fast_decode_plan(
            wrapper,
            torch.arange(17, dtype=torch.int32) * 64,
            indices,
            torch.full((16,), 16, dtype=torch.int32),
            32,
            8,
            128,
            16,
            q_len_per_req=8,
            q_data_type=torch.float16,
        )


@pytest.mark.parametrize("graph", [False, True])
def test_xqa_mtp_workspace_query_matches_policy_before_plan(make_wrapper, graph):
    _, plan, _ = make_wrapper(graph=graph)
    if graph:
        with pytest.raises(NotImplementedError, match="XQA MTP"):
            plan(query=True)
    else:
        assert plan(query=True) == (32, 16)


@pytest.mark.parametrize(
    "options,expected",
    [
        ({}, "xqa"),
        ({"return_lse": True}, "fallback"),
        ({"skip_softmax_threshold_scale_factor": 0.0}, "fallback"),
    ],
)
def test_xqa_mtp_run_fallback_guards(make_wrapper, monkeypatch, options, expected):
    wrapper, plan, _ = make_wrapper()
    plan()
    calls = []
    wrapper._cached_module.paged_run = lambda *args: calls.append("fallback")
    monkeypatch.setattr(
        decode, "xqa_batch_decode_with_kv_cache", lambda **kwargs: calls.append("xqa")
    )
    q = torch.zeros(16 * 8, 32, 128, dtype=torch.float16)
    kv = torch.zeros(16 * 64, 2, 16, 8, 128, dtype=torch.float16)
    wrapper.run(q, kv, enable_pdl=False, **options)
    assert calls == [expected]
