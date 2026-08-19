"""Minimal safety sentinels for MLA backend dispatch."""

from types import SimpleNamespace
import warnings

import pytest
import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.mla import BatchMLAPagedAttentionWrapper, MLAPlanMetadata
from flashinfer.mla._batch_mla import _auto_policy
from flashinfer.mla._batch_mla._backends import (
    cutlass_backend,
    cute_dsl_modular_backend,
    cute_dsl_monolithic_backend,
    fa2_backend,
    fa3_backend,
    trtllm_gen_backend,
    xqa_backend,
)
from flashinfer.mla._batch_mla._backends._capabilities import MLAPlanCapabilities
from flashinfer.mla._batch_mla._contracts import (
    _packed_mla_tensor_reference,
)
from flashinfer.mla._batch_mla._planning import _MLAPlanArguments
from flashinfer.mla import _sparse_mla_sm120


def _plan_kwargs():
    return {
        "metadata": MLAPlanMetadata.csr(
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0, 1], dtype=torch.int32),
            torch.tensor([0], dtype=torch.int32),
            torch.tensor([1], dtype=torch.int32),
        ),
        "num_heads": 1,
        "head_dim_ckv": 2,
        "head_dim_kpe": 1,
        "page_size": 1,
        "causal": False,
        "sm_scale": 1.0,
        "q_data_type": torch.float32,
        "kv_data_type": torch.float32,
        "query_layout": "packed",
        "kv_cache_layout": "packed",
    }


class _SuccessfulBackend:
    def run_from_wrapper(self, **_kwargs):
        return "selected-result"


def _run(wrapper):
    return wrapper.run(
        query=torch.empty(1, 1, 3),
        kv_cache=torch.empty(1, 1, 3),
    )


def _plan_args_for_structural_kinds(query_kind, kv_kind):
    values = _plan_kwargs()
    return _MLAPlanArguments(
        metadata=values["metadata"],
        num_heads=values["num_heads"],
        head_dim_ckv=values["head_dim_ckv"],
        head_dim_kpe=values["head_dim_kpe"],
        page_size=values["page_size"],
        causal=values["causal"],
        sm_scale=values["sm_scale"],
        q_data_type=values["q_data_type"],
        kv_data_type=values["kv_data_type"],
        query_kind=query_kind,
        kv_kind=kv_kind,
        kv_layout="independent-split",
        _float_workspace_buffer=torch.empty(1),
        _generated_fa_workspace=SimpleNamespace(device=torch.device("cpu")),
        _use_cuda_graph=False,
        _qo_indptr_buf=None,
        _kv_indptr_buf=None,
        _kv_indices_buf=None,
        _kv_len_arr_buf=None,
    )


def _backend_run_kwargs(**overrides):
    kwargs = {
        "query": torch.empty(1, 1, 3),
        "kv_cache": torch.empty(1, 1, 3),
        "out": None,
        "lse": None,
        "return_lse": False,
        "profiler_buffer": None,
        "kv_len": None,
        "page_table": None,
        "return_lse_base_on_e": False,
        "o_scale": None,
        "ckv_scale": None,
        "kpe_scale": None,
        "sinks": None,
        "skip_softmax_threshold_scale_factor": None,
        "bmm1_scale": None,
        "bmm2_scale": None,
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "backend_type",
    (
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        cute_dsl_monolithic_backend._BatchMLAPagedAttentionCuteDslMonolithicBackend,
        trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend,
        xqa_backend._BatchMLAPagedAttentionXqaBackend,
    ),
)
def test_backend_run_from_wrapper_requires_plan(backend_type):
    backend = object.__new__(backend_type)

    with pytest.raises(RuntimeError, match=r"run\(\) called before plan\(\)"):
        backend.run_from_wrapper(**_backend_run_kwargs())


def test_fa3_cpu_plan_is_typed_unsupported():
    backend = object.__new__(fa3_backend._BatchMLAPagedAttentionFa3Backend)
    backend.device = torch.device("cpu")
    metadata = _plan_kwargs()["metadata"]

    with pytest.raises(_BackendPlanUnsupportedError, match="cuda device"):
        backend.plan(
            qo_indptr=metadata.qo_indptr,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
            kv_len_arr=metadata.kv_len_arr,
            num_heads=1,
            head_dim_ckv=2,
            head_dim_kpe=1,
            page_size=1,
            causal=False,
            sm_scale=1.0,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
            use_profiler=False,
        )


def test_trtllm_gen_query_offsets_must_start_at_zero():
    with pytest.raises(_BackendPlanUnsupportedError, match="start at zero"):
        trtllm_gen_backend._get_q_layout(torch.tensor([1, 2], dtype=torch.int32))


def test_xqa_cpu_architecture_probe_is_unsupported():
    assert not xqa_backend._is_xqa_wrapper_arch_supported(torch.device("cpu"))


def _plan_xqa_backend(workspace: torch.Tensor, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        xqa_backend, "_is_xqa_wrapper_arch_supported", lambda _device: True
    )
    backend = xqa_backend._BatchMLAPagedAttentionXqaBackend(workspace)
    backend.plan(
        cum_seq_lens_q=None,
        block_tables=torch.zeros((1, 8), dtype=torch.int32),
        seq_lens=torch.ones(1, dtype=torch.int32),
        max_q_len=1,
        num_heads=128,
        head_dim_ckv=512,
        head_dim_kpe=64,
        page_size=16,
        causal=False,
        sm_scale=1.0,
        q_data_type=torch.bfloat16,
        kv_data_type=torch.bfloat16,
        use_profiler=False,
        enable_pdl=False,
        initialize_semaphore=False,
    )


def test_xqa_plan_rejects_noncontiguous_workspace_before_byte_view(monkeypatch):
    workspace = torch.empty((4, 4), dtype=torch.float32).t()
    assert not workspace.is_contiguous()

    with pytest.raises(ValueError, match="workspace buffer must be contiguous"):
        _plan_xqa_backend(workspace, monkeypatch)


def test_xqa_plan_always_validates_workspace_capacity(monkeypatch):
    with pytest.raises(_BackendPlanUnsupportedError, match="at least 128 MiB"):
        _plan_xqa_backend(torch.empty(1, dtype=torch.float32), monkeypatch)


def test_sparse_sm120_rejects_noncontiguous_out_before_launch(monkeypatch):
    monkeypatch.setattr(_sparse_mla_sm120, "is_sm12x_supported", lambda _device: True)
    monkeypatch.setattr(
        _sparse_mla_sm120,
        "_SparseMLAPagedAttentionRunner",
        lambda **_kwargs: (_ for _ in ()).throw(
            AssertionError("sparse runner constructed before out validation")
        ),
    )
    out = torch.empty((1, 1, 1, 1024), dtype=torch.bfloat16)[..., ::2]
    assert tuple(out.shape) == (1, 1, 1, 512)
    assert not out.is_contiguous()

    with pytest.raises(ValueError, match="out must be contiguous"):
        _sparse_mla_sm120._run_mla_decode_sparse_sm120(
            query=torch.empty((1, 1, 1, 576), dtype=torch.bfloat16),
            kv_cache=torch.empty((1, 1, 656), dtype=torch.uint8),
            workspace_buffer=torch.empty(1, dtype=torch.uint8),
            qk_nope_head_dim=512,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
            block_tables=torch.zeros((1, 1, 1), dtype=torch.int32),
            seq_lens=torch.ones(1, dtype=torch.int32),
            sparse_mla_top_k=1,
            out=out,
            bmm1_scale=1.0,
            bmm2_scale=1.0,
            sinks=None,
            skip_softmax_threshold_scale_factor=None,
            uses_shared_paged_kv_idx=True,
            lse=None,
            return_lse=False,
            kv_scale_format="auto",
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
            lambda _cls, _args: (
                calls.append("fa2")
                or (_ for _ in ()).throw(ValueError("invalid input"))
            )
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
            lambda _cls, _args: (
                calls.append("fa2")
                or (_ for _ in ()).throw(_BackendPlanUnsupportedError("unsupported"))
            )
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
            lambda _cls, _args: (
                calls.append("fa2")
                or (_ for _ in ()).throw(_BackendPlanUnsupportedError("unsupported"))
            )
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


@pytest.mark.parametrize(
    "backend_type",
    [
        cutlass_backend._BatchMLAPagedAttentionCutlassBackend,
        trtllm_gen_backend._BatchMLAPagedAttentionTrtllmGenBackend,
        cute_dsl_monolithic_backend._BatchMLAPagedAttentionCuteDslMonolithicBackend,
        cute_dsl_modular_backend._BatchMLAPagedAttentionCuteDslModularBackend,
        xqa_backend._BatchMLAPagedAttentionXqaBackend,
    ],
)
@pytest.mark.parametrize(
    "query_kind,kv_kind",
    [
        ("independent-split", "packed"),
        ("packed", "independent-split"),
        ("independent-split", "independent-split"),
    ],
)
def test_explicit_packed_native_plan_rejects_independent_split_representatives(
    backend_type, query_kind, kv_kind
):
    args = _plan_args_for_structural_kinds(query_kind, kv_kind)

    with pytest.raises(
        _BackendPlanUnsupportedError,
        match="independent split-only",
    ):
        backend_type.plan_from_wrapper(args)


@pytest.mark.parametrize(
    "query_kind",
    ["packed", "adjacent-split", "dual", "independent-split"],
)
@pytest.mark.parametrize(
    "kv_kind",
    ["packed", "adjacent-split", "dual", "independent-split"],
)
def test_auto_plan_filters_packed_native_candidates_by_structural_kind(
    query_kind, kv_kind
):
    calls = []
    args = _plan_args_for_structural_kinds(query_kind, kv_kind)

    class PackedNative:
        _plan_capabilities = MLAPlanCapabilities(
            backend_name="cutlass",
            lse_modes=frozenset(),
            kv_layouts=frozenset(),
            output_scales=frozenset(),
            scale_modes=frozenset(),
            requires_packed_query=True,
            requires_packed_kv_cache=True,
        )

        @classmethod
        def plan_from_wrapper(cls, _args):
            calls.append("cutlass")
            return "packed-native"

    class SplitNative:
        _plan_capabilities = MLAPlanCapabilities(
            backend_name="fa2",
            lse_modes=frozenset(),
            kv_layouts=frozenset(),
            output_scales=frozenset(),
            scale_modes=frozenset(),
        )

        @classmethod
        def plan_from_wrapper(cls, _args):
            calls.append("fa2")
            return "split-native"

    result = _auto_policy.plan_auto_backend(
        args,
        candidates=("cutlass", "fa2"),
        backend_types={"cutlass": PackedNative, "fa2": SplitNative},
        autotune_mode=None,
    )

    if query_kind == "independent-split" or kv_kind == "independent-split":
        assert result.backend_name == "fa2"
        assert calls == ["fa2"]
        assert result.trace.rejections[0][0] == "cutlass"
    else:
        assert result.backend_name == "cutlass"
        assert calls == ["cutlass"]


def test_dense_backends_convert_adjacent_split_inputs_without_copy():
    query_storage = torch.empty(1, 1, 3)
    kv_storage = torch.empty(1, 1, 3)
    q_nope, q_pe = query_storage[..., :2], query_storage[..., 2:]
    ckv_cache, kpe_cache = kv_storage[..., :2], kv_storage[..., 2:]
    recorded = {}
    backend = object.__new__(cutlass_backend._BatchMLAPagedAttentionCutlassBackend)
    backend._cached_module = object()
    backend._head_dim_ckv = 2
    backend._head_dim_kpe = 1
    backend.run = lambda **kwargs: recorded.update(kwargs) or "recorded"

    assert (
        backend.run_from_wrapper(
            **_backend_run_kwargs(
                query=(q_nope, q_pe),
                kv_cache=(ckv_cache, kpe_cache),
            )
        )
        == "recorded"
    )
    assert recorded["query"].data_ptr() == query_storage.data_ptr()
    assert recorded["kv_cache"].data_ptr() == kv_storage.data_ptr()


def test_fa_backends_convert_packed_inputs_to_split_views():
    query = torch.empty(1, 1, 3)
    kv_cache = torch.empty(1, 1, 3)
    recorded = {}
    backend = object.__new__(fa2_backend._BatchMLAPagedAttentionFa2Backend)
    backend._generated_fa_workspace = SimpleNamespace(raise_if_invalid=lambda: None)
    backend._query_split_widths = (2, 1)
    backend._kv_split_widths = (2, 1)
    backend.run = lambda **kwargs: recorded.update(kwargs) or "recorded"

    assert (
        backend.run_from_wrapper(
            **_backend_run_kwargs(
                query=query,
                kv_cache=kv_cache,
            )
        )
        == "recorded"
    )
    assert recorded["q_nope"].data_ptr() == query.data_ptr()
    assert (
        recorded["q_pe"].untyped_storage().data_ptr()
        == query.untyped_storage().data_ptr()
    )
    assert recorded["ckv_cache"].data_ptr() == kv_cache.data_ptr()
    assert (
        recorded["kpe_cache"].untyped_storage().data_ptr()
        == kv_cache.untyped_storage().data_ptr()
    )


def test_unplanned_cutlass_publishes_normal_plan_and_run(monkeypatch):
    captured = {}

    class _Backend:
        def run_from_wrapper(self, **kwargs):
            captured["run_kwargs"] = kwargs
            return "unplanned-result"

    def plan_from_wrapper(_cls, args):
        captured["plan_count"] = captured.get("plan_count", 0) + 1
        captured["plan_args"] = args
        return _Backend()

    backend_type = cutlass_backend._BatchMLAPagedAttentionCutlassBackend
    monkeypatch.setattr(
        backend_type,
        "plan_from_wrapper",
        classmethod(plan_from_wrapper),
    )
    query = torch.empty(1, 1, 576)
    kv_cache = torch.empty(1, 1, 576)
    kv_len = torch.tensor([1], dtype=torch.int32)
    page_table = torch.zeros(1, 1, dtype=torch.int32)

    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")

    with pytest.warns(DeprecationWarning, match="explicitly requested CUTLASS"):
        actual = wrapper.run(
            query=query,
            kv_cache=kv_cache,
            kv_len=kv_len,
            page_table=page_table,
        )

    assert actual == "unplanned-result"
    assert wrapper._selected_backend == "cutlass"
    assert wrapper._backend_impl is not None
    assert captured["plan_args"].metadata.block_tables is page_table
    assert captured["plan_args"].metadata.seq_lens is kv_len
    assert captured["run_kwargs"]["query"] is query
    assert captured["run_kwargs"]["kv_cache"] is kv_cache

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            wrapper.run(
                query=query,
                kv_cache=kv_cache,
                kv_len=kv_len,
                page_table=page_table,
            )
            == "unplanned-result"
        )
    assert captured["plan_args"].metadata.block_tables is page_table
    assert captured["plan_count"] == 1


def test_unplanned_cutlass_requires_runtime_metadata():
    wrapper = BatchMLAPagedAttentionWrapper(torch.empty(1), backend="cutlass")

    with pytest.raises(ValueError, match="requires both kv_len and page_table"):
        wrapper.run(
            query=torch.empty(1, 1, 576),
            kv_cache=torch.empty(1, 1, 576),
        )


def test_independent_split_to_packed_is_rejected_without_copy():
    left, right = torch.empty(1, 1, 2), torch.empty(1, 1, 3)

    with pytest.raises(ValueError, match=r"query.*packed representation zero-copy"):
        _packed_mla_tensor_reference(
            packed=None,
            left=left,
            right=right,
            representation="split",
            widths=(2, 3),
            name="query",
        )


def test_cutlass_wrapper_rejects_independent_split_kv_without_copy():
    backend = object.__new__(cutlass_backend._BatchMLAPagedAttentionCutlassBackend)
    backend._cached_module = object()
    backend._head_dim_ckv = 2
    backend._head_dim_kpe = 1
    captured = {}
    backend.run = lambda **kwargs: captured.update(kwargs) or "launched"
    ckv_cache = torch.empty(1, 1, 2)
    kpe_cache = torch.empty(1, 1, 1)

    with pytest.raises(ValueError, match=r"KV cache.*packed representation zero-copy"):
        backend.run_from_wrapper(
            query=torch.empty(1, 1, 3),
            kv_cache=(ckv_cache, kpe_cache),
            out=None,
            lse=None,
            return_lse=False,
            profiler_buffer=None,
            kv_len=None,
            page_table=None,
            return_lse_base_on_e=False,
            o_scale=None,
            ckv_scale=None,
            kpe_scale=None,
            sinks=None,
            skip_softmax_threshold_scale_factor=None,
            bmm1_scale=None,
            bmm2_scale=None,
        )
    assert captured == {}
