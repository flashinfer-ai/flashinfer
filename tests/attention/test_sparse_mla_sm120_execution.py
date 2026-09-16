import pytest
import torch

from flashinfer.mla._sparse_mla_sm120._execution import (
    resolve_dsv4_nvfp4,
    resolve_attention,
)
from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
from tests.attention.sparse_mla_test_utils import (
    quantize_kv_dsv4,
    quantize_kv_glm53_nope,
    require_sm12x,
)


def test_wrapper_eager_scratch_peak_and_graph_pinning(sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    torch.manual_seed(5197)
    cache = quantize_kv_dsv4_1(
        torch.randn(4, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    q = torch.randn(64, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    indices = torch.randint(256, (64, 512), device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision="fp8"
    )

    def run(tokens):
        return wrapper.run(
            q[:tokens],
            cache,
            indices[:tokens],
            output[:tokens],
            512**-0.5,
            return_lse=True,
        )

    run(1)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    saved_lse = run(1)
    expected_lse = saved_lse.clone()
    for tokens in (4, 8, 16, 24, 32, 48, 64):
        run(tokens)
    torch.cuda.synchronize()
    entries = tuple(wrapper._prepared_calls.values())
    max_scratch = sum(max(entry.workspace[i][2] for entry in entries) for i in (0, 1))
    lse_bytes = sum(entry.workspace[2][2] for entry in entries)
    assert torch.cuda.memory_allocated() - before <= max_scratch + lse_bytes + (1 << 20)
    torch.testing.assert_close(saved_lse, expected_lse, atol=0, rtol=0)
    assert all(entry.mid is None and entry.mlse is None for entry in entries)

    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision="fp8"
    )
    states = []
    for tokens in (4, 8):
        lse = run(tokens)
        states.append(
            (tokens, lse, output[:tokens].clone(), lse.clone(), torch.cuda.CUDAGraph())
        )
    for tokens, _, _, _, graph in states:
        with torch.cuda.graph(graph):
            run(tokens)
    pinned = tuple(wrapper._prepared_calls.values())
    pointers = [(entry.mid.data_ptr(), entry.mlse.data_ptr()) for entry in pinned]
    assert len(set(pointers)) == 2
    run(64)
    run(4)
    run(64)
    for tokens, lse, expected, el, graph in states * 2:
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(output[:tokens], expected, atol=0, rtol=0)
        torch.testing.assert_close(lse, el, atol=0, rtol=0)
    assert pointers == [
        (entry.mid.data_ptr(), entry.mlse.data_ptr()) for entry in pinned
    ]


def test_swapab_preference_allows_pitched_indices_when_decode_selected(sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _api
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv3_2

    q = torch.zeros(4, 64, 576, device="cuda", dtype=torch.bfloat16)
    cache = quantize_kv_dsv3_2(
        torch.ones(2, 64, 1, 576, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.zeros(4, 144, device="cuda", dtype=torch.int32)[:, :128]
    out = torch.empty(4, 64, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(4, 64, device="cuda")
    mid = torch.empty(4, 64, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 64, 2, device="cuda")
    wrapper = SparseMLASm120Wrapper()
    wrapper.run(q, cache, idx, out, 576**-0.5)
    wrapper.run(q, cache, idx, out, 576**-0.5, prefill_impl="swapab")
    _api._sparse_mla_sm120_paged_attention(
        q,
        cache,
        idx,
        out,
        lse,
        576**-0.5,
        mid_out=mid,
        mid_lse=mlse,
        prefill_impl="swapab",
    )
    torch.testing.assert_close(out, torch.ones_like(out))


def test_caller_entry_reuses_prepared_plan_and_buffers(monkeypatch, sm12x):
    from flashinfer.mla._sparse_mla_sm120 import _api, _execution, _calibration
    from tests.attention.sparse_mla_test_utils import (
        dequantize_kv_dsv4,
        _ref_sparse_attn,
    )

    monkeypatch.setattr(_calibration, "get_decode_max_tokens", lambda *args: 0)
    torch.manual_seed(813)
    q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4(
        torch.randn(4, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.randint(256, (4, 144), device="cuda", dtype=torch.int32)[:, :128]
    out = torch.empty_like(q)
    lse = torch.empty(4, 16, device="cuda")
    mid = torch.empty(4, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 16, 2, device="cuda")
    expected, el = _ref_sparse_attn(q, dequantize_kv_dsv4(cache), idx, 512**-0.5, 512)

    def call():
        _api._sparse_mla_sm120_paged_attention(
            q, cache, idx, out, lse, 512**-0.5, mid_out=mid, mid_lse=mlse
        )

    call()
    queries = []
    original = _execution.query

    def query(name, *args, **kwargs):
        queries.append(name)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(_execution, "query", query)
    old_out, old_lse = out, lse
    out, lse, mid, mlse = [torch.empty_like(x) for x in (out, lse, mid, mlse)]
    pointers = [x.data_ptr() for x in (out, lse, mid, mlse)]
    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
    torch.testing.assert_close(lse, el, atol=0.02, rtol=0.02)
    torch.testing.assert_close(old_out, expected, atol=0.005, rtol=0.05)
    torch.testing.assert_close(old_lse, el, atol=0.02, rtol=0.02)
    assert pointers == [x.data_ptr() for x in (out, lse, mid, mlse)]
    assert not {"inspect_metadata", "metadata_candidates"}.intersection(queries)
    with pytest.raises(ValueError, match="caller-supplied"):
        _api._sparse_mla_sm120_paged_attention(q, cache, idx, out, lse, 512**-0.5)


def test_caller_entry_matches_dsv41_profile(monkeypatch, sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _api, _calibration
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    monkeypatch.setattr(
        _calibration,
        "get_ordinary_profile",
        lambda *args: {"buckets": {"4": {"variant": 1, "cpb": 2}}},
    )
    from flashinfer.mla._sparse_mla_sm120 import _prepared

    _prepared._functional_plans.clear()
    q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4_1(
        torch.randn(4, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.randint(256, (4, 128), device="cuda", dtype=torch.int32)
    out, expected = torch.empty_like(q), torch.empty_like(q)
    lse, el = [torch.empty(4, 16, device="cuda") for _ in range(2)]
    wrapper = SparseMLASm120Wrapper(kv_scale_format="ue8m0_g32")
    wrapper.run(q, cache, idx, expected, 512**-0.5, out_lse=el)

    def call():
        _api._sparse_mla_sm120_paged_attention(
            q,
            cache,
            idx,
            out,
            lse,
            512**-0.5,
            kv_scale_format="ue8m0_g32",
            mid_out=torch.empty(0, dtype=torch.int8),
            mid_lse=torch.empty(0, dtype=torch.float64),
        )

    call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected) and torch.equal(lse, el)
    prefill_plans = [
        p
        for p in _prepared._functional_plans.values()
        if p.plan.inspect()["variant"] == 1
    ]
    assert prefill_plans
    mid = torch.empty(4, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 16, 2, device="cuda")
    _api.sparse_mla_sm120_decode_dsv4(
        q,
        cache,
        idx,
        mid,
        mlse,
        out,
        lse,
        512**-0.5,
        model_type=5,
    )

    plans = [
        p
        for key, p in _prepared._functional_plans.items()
        if key[0] == 5 and key[5] and p.plan.inspect()["tokens"] == 4
    ]
    assert any(
        p.plan.inspect()["cpb"] == 2 and p.plan.inspect()["variant"] == 0 for p in plans
    )
    torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)


def test_dsv41_cache_api_names():
    import flashinfer.mla as mla

    for operation in ("pack", "append"):
        assert callable(
            getattr(mla, f"dsv41_fp4_quantize_{operation}_sparse_mla_cache")
        )
        assert not hasattr(mla, f"v41_fp4_quantize_{operation}_sparse_mla_cache")
        assert callable(getattr(mla, f"nvfp4_quantize_{operation}_sparse_mla_cache"))


@pytest.fixture
def sm12x():
    require_sm12x()


def _resolve(**kwargs):
    metadata = dict(
        model=5,
        tokens=4,
        heads=13,
        topk=128,
        extra_topk=0,
        page_size=64,
        extra_page_size=0,
        page_stride_bytes=64 * 528,
        extra_page_stride_bytes=0,
        row_stride_bytes=528,
        indices_stride=128,
        extra_indices_stride=0,
        lse_stride=13,
        has_lengths=False,
        has_extra_lengths=False,
        has_sink=False,
        extra_fp4=False,
        variant=0,
        precision="fp8",
        cpb=1,
        sm_count=148,
        max_shared_bytes=101376,
    )
    metadata.update(kwargs)
    return resolve_attention(**metadata)


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_execute_descriptor_graph_and_mismatch(precision):
    import torch
    from flashinfer.utils import is_sm12x_supported
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    q = torch.zeros(4, 13, 512, device="cuda", dtype=torch.bfloat16)
    cache = quantize_kv_dsv4_1(
        torch.ones(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16)
    )
    idx = torch.zeros(4, 128, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(4, 13, device="cuda")
    mid = torch.empty(4, 16, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 16, 2, device="cuda")
    plan = _resolve(precision=precision)
    module = gen_sparse_mla_sm120_module().build_and_load()

    def call(d=plan, query=q):
        module.execute_attention(
            d,
            query,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
        )

    call()
    expected = out.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    with pytest.raises(RuntimeError, match="mismatch"):
        call(query=q[:2])
    with pytest.raises(RuntimeError, match="module mismatch"):
        call(d=module)
    with pytest.raises(RuntimeError, match="mismatch"):
        call(d=_resolve(tokens=2, precision=precision))
    assert plan.inspect()["tokens"] == 4
    assert plan.inspect()["row_stride_bytes"] == 528


@pytest.mark.parametrize(
    "model,variant,heads", [(3, 0, 24), (3, 4, 64), (1, 2, 64), (5, 1, 16)]
)
def test_resolved_execution_matches_legacy(model, variant, heads):
    import torch
    from flashinfer.utils import is_sm12x_supported
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_glm53_nope,
        quantize_kv_dsv4,
        quantize_kv_dsv4_1,
    )

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    q = torch.randn(4, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    quant = {1: quantize_kv_dsv4, 3: quantize_kv_glm53_nope, 5: quantize_kv_dsv4_1}[
        model
    ]
    bpt = 584 if model == 1 else 528
    cache = quant(latent)[..., :bpt].contiguous()
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(4, heads, device="cuda")
    padded = (heads + 15) // 16 * 16
    mid = torch.empty(4, padded, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, padded, 2, device="cuda")
    plan = _resolve(
        model=model,
        variant=variant,
        heads=heads,
        lse_stride=heads,
        row_stride_bytes=bpt,
        page_stride_bytes=64 * bpt,
        precision="default",
    )
    module = gen_sparse_mla_sm120_module().build_and_load()
    if variant:
        module.sparse_mla_sm120_paged_attention(
            q,
            cache,
            idx,
            out,
            lse,
            512**-0.5,
            model,
            variant,
            None,
            None,
            None,
            None,
            None,
            False,
        )
    else:
        module.sparse_mla_sm120_decode_dsv3_2(
            q, cache, idx, mid, mlse, out, lse, 2, 512**-0.5, None, None, model, 1
        )
    expected = out.clone(), lse.clone()

    def call():
        module.execute_attention(
            plan,
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
        )

    call()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])


@pytest.mark.parametrize(
    "model,variant,extra_fp4", [(5, 0, True), (5, 1, True), (1, 3, False)]
)
def test_dual_resolved_execution_and_optional_mismatch(model, variant, extra_fp4):
    import torch
    from flashinfer.utils import is_sm12x_supported
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_dsv4,
        quantize_kv_dsv4_1,
    )
    from flashinfer.mla import dsv41_fp4_quantize_pack_sparse_mla_cache

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    heads = 16
    q = torch.randn(4, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    quant = quantize_kv_dsv4 if model == 1 else quantize_kv_dsv4_1
    bpt = 584 if model == 1 else 528
    cache = quant(latent)[..., :bpt].contiguous()
    extra = (
        dsv41_fp4_quantize_pack_sparse_mla_cache(latent.squeeze(2))
        if extra_fp4
        else cache
    )
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(4, heads, device="cuda")
    mid = torch.empty(4, heads, 4, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, heads, 4, device="cuda")
    plan = _resolve(
        model=model,
        variant=variant,
        heads=heads,
        lse_stride=heads,
        row_stride_bytes=bpt,
        page_stride_bytes=64 * bpt,
        precision="default",
        extra_topk=128,
        extra_page_size=64,
        extra_page_stride_bytes=64 * (288 if extra_fp4 else bpt),
        extra_indices_stride=128,
        extra_fp4=extra_fp4,
    )
    module = gen_sparse_mla_sm120_module().build_and_load()
    if variant:
        module.sparse_mla_sm120_paged_attention(
            q,
            cache,
            idx,
            out,
            lse,
            512**-0.5,
            model,
            variant,
            None,
            None,
            extra,
            idx,
            None,
            extra_fp4,
        )
    else:
        module.sparse_mla_sm120_decode_dsv4(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            4,
            512**-0.5,
            None,
            None,
            extra,
            idx,
            None,
            model,
            1,
            extra_fp4,
        )
    expected = out.clone(), lse.clone()

    def call(lengths=None):
        module.execute_attention(
            plan,
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            512**-0.5,
            lengths,
            None,
            extra,
            idx,
            None,
        )

    call()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    with pytest.raises(RuntimeError, match="mismatch"):
        call(torch.full((4,), 128, device="cuda", dtype=torch.int32))


@pytest.mark.parametrize("operand", ["indices", "extra_indices"])
def test_nvfp4_legacy_rejects_width_before_narrowing(operand, sm12x):
    from flashinfer.mla._sparse_mla_sm120._execution import (
        get_sparse_mla_dsv4_nvfp4_module,
    )

    module = get_sparse_mla_dsv4_nvfp4_module()
    q = torch.empty(0, 16, 512, device="cuda", dtype=torch.bfloat16)
    cache = torch.empty(1, 64, 384, device="cuda", dtype=torch.uint8)
    idx = torch.empty(
        0,
        2**32 + 128 if operand == "indices" else 128,
        device="cuda",
        dtype=torch.int32,
    )
    extra = (
        torch.empty(0, 2**31 - 1, device="cuda", dtype=torch.int32)
        if operand == "extra_indices"
        else None
    )
    splits = 2 + ((extra.shape[1] + 63) // 64 if extra is not None else 0)
    mid = torch.empty(0, 16, splits, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(0, 16, splits, device="cuda")
    lse = torch.empty(0, 16, device="cuda")
    message = (
        "unsupported NVFP4 heads/topk" if extra is None else "NVFP4 extra_topk exceeds"
    )
    with pytest.raises(RuntimeError, match=message):
        module.sparse_mla_sm120_nvfp4_decode(
            q,
            cache,
            idx,
            mid,
            mlse,
            q,
            lse,
            splits,
            512**-0.5,
            None,
            None,
            cache if extra is not None else None,
            extra,
            None,
            1,
            False,
        )


def test_nvfp4_resolver_extra_width_boundary(sm12x):
    from flashinfer.mla._sparse_mla_sm120._execution import (
        get_sparse_mla_dsv4_nvfp4_module,
    )

    module = get_sparse_mla_dsv4_nvfp4_module()
    limit = 2**31 - 1 - 576

    def resolve(extra_topk):
        return resolve_dsv4_nvfp4(
            tokens=1,
            heads=16,
            topk=512,
            extra_topk=extra_topk,
            page_size=64,
            extra_page_size=64,
            page_stride_bytes=24576,
            extra_page_stride_bytes=24576,
            cpb=1,
            sm_count=148,
            max_shared_bytes=101376,
        )

    plan = resolve(limit)
    assert plan.inspect()["active_splits"] == 8 + (limit + 63) // 64
    assert module.supports_attention(16, 512, 64, limit, 64)
    assert not module.supports_attention(16, 512, 64, limit + 1, 64)
    with pytest.raises(RuntimeError, match="NVFP4 extra_topk exceeds"):
        resolve(limit + 1)


@pytest.mark.parametrize(
    "cpb,prefill,stage1",
    [
        (1, False, False),
        (4, False, False),
        (8, False, False),
        (8, False, True),
        (8, True, False),
    ],
)
def test_dsv4_nvfp4_execute_plan_graph(cpb, prefill, stage1):
    import torch
    from flashinfer.utils import is_sm12x_supported
    from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
    from flashinfer.jit.mla import gen_sparse_mla_sm120_module

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    q = torch.randn(8, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = nvfp4_quantize_pack_sparse_mla_cache(
        torch.randn(8, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.randint(512, (8, 512), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    lse = torch.empty(8, 64, device="cuda")
    mid = torch.empty(8, 64, 8, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(8, 64, 8, device="cuda")
    plan = resolve_dsv4_nvfp4(
        tokens=8,
        heads=64,
        topk=512,
        extra_topk=0,
        page_size=64,
        extra_page_size=0,
        page_stride_bytes=24576,
        extra_page_stride_bytes=0,
        cpb=cpb,
        sm_count=148,
        max_shared_bytes=101376,
        prefill=prefill,
        stage1_only=stage1,
    )
    module = gen_sparse_mla_sm120_module().build_and_load()
    if prefill:
        module.sparse_mla_sm120_nvfp4_prefill(
            q, cache, idx, out, lse, 512**-0.5, None, None, None, None, None
        )
    else:
        module.sparse_mla_sm120_nvfp4_decode(
            q,
            cache,
            idx,
            mid,
            mlse,
            out,
            lse,
            8,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
            cpb,
            stage1,
        )
    partial, split_lse, _ = plan.workspace()
    exact_mid = mid.flatten()[: partial[0][0]] if partial[2] else None
    exact_mlse = mlse.flatten()[: split_lse[0][0]] if split_lse[2] else None
    actual = (exact_mid, exact_mlse) if stage1 else (out, lse)
    expected = tuple(x.clone() for x in actual)

    def call(d=plan):
        module.dsv4_nvfp4_execute_attention(
            d,
            q,
            cache,
            idx,
            exact_mid,
            exact_mlse,
            out,
            lse,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
        )

    call()
    assert all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert all(torch.equal(x, y) for x, y in zip(actual, expected, strict=True))
    with pytest.raises(RuntimeError, match="module mismatch"):
        call(_resolve())
    wrong_optional = resolve_dsv4_nvfp4(
        tokens=8,
        heads=64,
        topk=512,
        extra_topk=0,
        page_size=64,
        extra_page_size=0,
        page_stride_bytes=24576,
        extra_page_stride_bytes=0,
        cpb=cpb,
        sm_count=148,
        max_shared_bytes=101376,
        prefill=prefill,
        stage1_only=stage1,
        has_sink=True,
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        call(wrong_optional)
    assert plan.inspect()["row_stride_bytes"] == 384
    offset_q = torch.empty(q.numel() + 2, device="cuda", dtype=torch.bfloat16)[2:].view(
        q.shape
    )
    offset_q.copy_(q)
    offset_out = torch.empty(out.numel() + 2, device="cuda", dtype=torch.bfloat16)[
        2:
    ].view(out.shape)
    destination = offset_out if plan.inspect()["merge"] == "direct" else out
    module.dsv4_nvfp4_execute_attention(
        plan,
        offset_q,
        cache,
        idx,
        exact_mid,
        exact_mlse,
        destination,
        lse,
        512**-0.5,
        None,
        None,
        None,
        None,
        None,
    )
    torch.cuda.synchronize()
    if not stage1:
        assert torch.equal(destination, expected[0])


@pytest.mark.parametrize(
    "precision,cache_format", [("default", "fp8"), ("bf16", "fp8"), ("nvfp4", "nvfp4")]
)
def test_wrapper_prepared_resolves_once_and_capture(
    monkeypatch, precision, cache_format
):
    import torch
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        nvfp4_quantize_pack_sparse_mla_cache,
    )
    from flashinfer.mla._sparse_mla_sm120 import _prepared as prepared
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = (
        nvfp4_quantize_pack_sparse_mla_cache(latent.squeeze(2))
        if cache_format == "nvfp4"
        else quantize_kv_dsv4_1(latent)
    )
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    wrapper = SparseMLASm120Wrapper(
        kv_cache_format=cache_format,
        compute_precision=precision,
        kv_scale_format="auto" if cache_format == "nvfp4" else "ue8m0_g32",
    )
    calls = []
    original = prepared.resolve_execution

    def counted(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(prepared, "resolve_execution", counted)

    def run():
        return wrapper.run(q, cache, idx, out, 512**-0.5, return_lse=True)

    expected = run().clone(), out.clone()
    monkeypatch.setattr(
        prepared, "validate_metadata", lambda *a, **k: pytest.fail("hot metadata query")
    )
    run()
    run()
    assert len(calls) == 1
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert len(calls) == 1 and torch.equal(out, expected[1])
    assert torch.equal(run(), expected[0])
    replacement_cache = cache.clone()
    replacement_out = torch.empty_like(out)
    wrapper.run(q, replacement_cache, idx, replacement_out, 512**-0.5)
    assert len(calls) == 1 and torch.equal(replacement_out, expected[1])
    unaligned = torch.empty(cache.numel() + 1, device="cuda", dtype=torch.uint8)[
        1:
    ].view(cache.shape)
    unaligned.copy_(cache)
    with pytest.raises(RuntimeError, match="align"):
        wrapper.run(q, unaligned, idx, replacement_out, 512**-0.5)
    assert len(calls) == 1


def test_estimate_never_refines_in_tuning_or_capture(monkeypatch):
    import torch
    from types import SimpleNamespace
    from flashinfer.mla._sparse_mla_sm120 import _policy as policy

    c = object()
    state = {"tuning": False, "capture": False, "override": None, "refines": 0}
    monkeypatch.setattr(
        policy.AutoTuner,
        "get",
        lambda: SimpleNamespace(
            is_tuning_mode=state["tuning"], _get_skip_ops_stack=lambda: []
        ),
    )
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: state["capture"]
    )
    monkeypatch.setattr(policy._cpb, "get_constants", lambda *a: c)
    monkeypatch.setattr(policy._cpb, "_device_key", lambda *a: "refine-test")
    monkeypatch.setattr(policy._cpb, "get_cpb_override", lambda *a: state["override"])
    monkeypatch.setattr(policy._cpb, "select_cpb", lambda *a, **k: 1)
    monkeypatch.setattr(policy._cpb, "_CHUNK_WIDTH", {"dsv4": 64})

    def refine(*a):
        state["refines"] += 1
        return 2

    monkeypatch.setattr(policy._cpb, "refine_cpb", refine)
    policy._cpb_hot_cache.clear()
    args = (torch.device("cuda:0"), "dsv4", 4, 64, 512, 0)
    assert policy._resolve_cpb(*args) == 1
    state["tuning"] = True
    assert policy._resolve_cpb(*args) == 1 and state["refines"] == 0
    monkeypatch.setattr(policy._cpb, "get_constants", lambda *a: None)
    monkeypatch.setattr(
        policy._cpb, "calibrate", lambda *a: pytest.fail("capture calibrated")
    )
    monkeypatch.setattr(
        policy._cpb,
        "mark_calibration_failed",
        lambda *a: pytest.fail("capture marked failed"),
    )
    state["capture"] = True
    assert policy._resolve_cpb(*args) == -1


def test_capture_rejects_insufficient_workspace():
    import torch
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4_1(latent)
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    lengths = torch.full((4,), 128, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    workspace = torch.empty(1 << 20, device="cuda", dtype=torch.uint8)

    def run(w=workspace):
        return trtllm_batch_decode_sparse_mla_dsv4(
            q,
            cache,
            w,
            idx,
            out=out,
            swa_topk_lens=lengths,
            bmm1_scale=512**-0.5,
            backend="sparse",
            kv_layout="NHD",
            kv_cache_format="fp8_dsv41",
        )

    run()
    with (
        pytest.raises(ValueError, match="workspace"),
        torch.cuda.graph(torch.cuda.CUDAGraph()),
    ):
        run(workspace[:0])


def test_prepared_profile_update_and_invalid_extra_order(monkeypatch):
    import torch
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _policy as policy
    from flashinfer.mla._sparse_mla_sm120 import _prepared as prepared
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4

    q = torch.randn(4, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4(
        torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    state = {"tune": False}
    monkeypatch.setattr(
        policy._cpb,
        "get_ordinary_profile",
        lambda *a: {"buckets": {"4": {"variant": 2 if state["tune"] else 0, "cpb": 1}}},
    )
    monkeypatch.setattr(prepared, "tuning_enabled", lambda dsv4_nvfp4: state["tune"])
    wrapper = SparseMLASm120Wrapper()

    def run():
        wrapper.run(q, cache, idx, out, 512**-0.5)

    run()
    first = next(iter(wrapper._prepared_calls.values()))
    assert first.plan.inspect()["variant"] == 0 and first.plan.workspace()[0][2] > 0
    state["tune"] = True
    run()
    current = next(iter(wrapper._prepared_calls.values()))
    assert (
        current.plan.inspect()["implementation"] == "mg"
        and current.plan.inspect()["numeric_route"] == "hybrid"
        and current.plan.workspace()[0][2] == 0
    )
    assert len(wrapper._prepared_calls) == 1
    expected = out.clone()
    original_resolve = prepared.resolve_execution
    monkeypatch.setattr(
        policy._cpb, "_constants_version", policy._cpb._constants_version + 1
    )
    monkeypatch.setattr(
        prepared,
        "resolve_execution",
        lambda *a, **k: pytest.fail("capture or invalid input planned"),
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, expected)
    with pytest.raises(ValueError, match="together"):
        wrapper.run(q, cache, idx, out, 512**-0.5, extra_indices=idx)
    assert next(iter(wrapper._prepared_calls.values())) is current
    monkeypatch.setattr(prepared, "resolve_execution", original_resolve)
    state.update(tune=False, threshold=64)
    run()
    assert len(wrapper._prepared_calls) == 1
    assert next(iter(wrapper._prepared_calls.values())).plan.inspect()["variant"] == 0


@pytest.mark.parametrize("precision", ["fp8", "bf16", "nvfp4"])
@pytest.mark.parametrize("operand", ["q", "output"])
def test_prepared_rejects_offset_vector_operand_before_launch(
    precision, operand, tmp_path
):
    import torch
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        nvfp4_quantize_pack_sparse_mla_cache,
    )
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    dsv4_nvfp4 = precision == "nvfp4"
    q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    latent = torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = (
        nvfp4_quantize_pack_sparse_mla_cache(latent.squeeze(2))
        if dsv4_nvfp4
        else quantize_kv_dsv4_1(latent)
    )
    indices = torch.zeros(4, 128, device="cuda", dtype=torch.int32)
    output = torch.empty_like(q)
    wrapper = SparseMLASm120Wrapper(
        compute_precision=precision,
        kv_cache_format="nvfp4" if dsv4_nvfp4 else "fp8",
        kv_scale_format="auto" if dsv4_nvfp4 else "ue8m0_g32",
    )
    wrapper.run(q, cache, indices, output, 512**-0.5)
    offset = torch.empty(q.numel() + 1, device="cuda", dtype=torch.bfloat16)[1:].view(
        q.shape
    )
    offset.copy_(q)
    bad_q, bad_output = (offset, output) if operand == "q" else (q, offset)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    graph.enable_debug_mode()
    with (
        torch.cuda.graph(graph),
        pytest.raises(RuntimeError, match=f"{operand}.*aligned"),
    ):
        wrapper.run(bad_q, cache, indices, bad_output, 512**-0.5)
    graph.instantiate()
    dot = tmp_path / "rejected.dot"
    graph.debug_dump(str(dot))
    assert "sparse_mla_" not in dot.read_text()
    assert len(wrapper._prepared_calls) == 1


@pytest.mark.parametrize("operand", ["indices", "lengths", "query"])
def test_functional_capture_rejects_copying_normalization(operand):
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    query = torch.zeros(2, 2, 16, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(2, 2, 128, device="cuda", dtype=torch.int32)
    lengths = torch.full((2, 2), 128, device="cuda", dtype=torch.int32)
    if operand == "query":
        query = torch.zeros(2, 3, 16, 512, device="cuda", dtype=torch.bfloat16)[:, :2]
    if operand == "indices":
        indices = torch.zeros(2, 3, 128, device="cuda", dtype=torch.int32)[:, :2]
    if operand == "lengths":
        lengths = torch.full((2, 3), 128, device="cuda", dtype=torch.int32)[:, :2]
    cache = quantize_kv_dsv4_1(
        torch.ones(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16)
    )
    output = torch.empty(2, 2, 16, 512, device="cuda", dtype=torch.bfloat16)
    workspace = torch.empty(1 << 20, device="cuda", dtype=torch.uint8)

    def run():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query,
            cache,
            workspace,
            indices,
            out=output,
            swa_topk_lens=lengths,
            backend="sparse",
            kv_layout="NHD",
            kv_cache_format="fp8_dsv41",
        )

    copies = []

    class WatchCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.clone.default, torch.ops.aten.copy_.default):
                copies.append(str(func))
            return func(*args, **(kwargs or {}))

    with WatchCopies():
        run()
    assert copies
    copies.clear()
    with (
        torch.cuda.graph(torch.cuda.CUDAGraph()),
        WatchCopies(),
        pytest.raises(ValueError, match="capture.*view|view.*capture"),
    ):
        run()
    assert not copies


def test_capture_normalization_preserves_pitched_views():
    import torch
    from torch.utils._python_dispatch import TorchDispatchMode
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("requires SM12x")
    from flashinfer.mla._core import (
        _sparse_mla_reshape,
        _normalize_sm120_sparse_v32_topk_length,
    )
    from flashinfer.mla import trtllm_batch_decode_sparse_mla_dsv4
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4_1

    query = torch.zeros(2, 2, 16, 512, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(2, 2, 144, device="cuda", dtype=torch.int32)[..., :128]
    lengths = torch.full((2, 2), 128, device="cuda", dtype=torch.int32)
    cache = quantize_kv_dsv4_1(
        torch.ones(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16)
    )
    output = torch.empty_like(query)
    workspace = torch.empty(1 << 20, device="cuda", dtype=torch.uint8)

    def run():
        return trtllm_batch_decode_sparse_mla_dsv4(
            query,
            cache,
            workspace,
            indices,
            out=output,
            swa_topk_lens=lengths,
            backend="sparse",
            kv_layout="NHD",
            kv_cache_format="fp8_dsv41",
        )

    expected = run().clone()
    copies = []

    class WatchCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in (torch.ops.aten.clone.default, torch.ops.aten.copy_.default):
                copies.append(str(func))
            return func(*args, **(kwargs or {}))

    strided_lengths = torch.full((8,), 128, device="cuda", dtype=torch.int32)[::2]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), WatchCopies():
        view = _sparse_mla_reshape(indices, (4, 128), "indices")
        assert view.data_ptr() == indices.data_ptr() and view.stride(0) == 144
        with pytest.raises(ValueError, match="contiguous view"):
            _normalize_sm120_sparse_v32_topk_length(
                strided_lengths, batch_size=2, q_len_per_request=2, device=query.device
            )
        run()
    assert not copies
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(output, expected)


def _standalone_case(route="fp8", heads=13):
    q = torch.zeros(2, heads, 512, device="cuda", dtype=torch.bfloat16)
    latent = torch.ones(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16)
    cache = (
        quantize_kv_glm53_nope(latent)[..., :528].contiguous()
        if route == "v32"
        else quantize_kv_dsv4(latent)
    )
    h = heads if heads == 8 else (heads + 15) // 16 * 16
    return dict(
        q=q,
        cache=cache,
        idx=torch.zeros(2, 128, device="cuda", dtype=torch.int32),
        mid=torch.zeros(2, h, 2, 512, device="cuda", dtype=torch.bfloat16),
        mlse=torch.zeros(2, h, 2, device="cuda"),
        out=torch.empty_like(q),
        lse=torch.empty(2, heads, device="cuda"),
    )


def _standalone_call(tensors, route="fp8"):
    t = tensors
    module = _get_sparse_mla_sm120_decode_module()
    if route == "v32":
        module.sparse_mla_sm120_decode_dsv3_2(
            t["q"],
            t["cache"],
            t["idx"],
            t["mid"],
            t["mlse"],
            t["out"],
            t["lse"],
            2,
            512**-0.5,
            None,
            None,
            3,
            1,
        )
    else:
        module.sparse_mla_sm120_decode_dsv4(
            t["q"],
            t["cache"],
            t["idx"],
            t["mid"],
            t["mlse"],
            t["out"],
            t["lse"],
            2,
            512**-0.5,
            None,
            None,
            None,
            None,
            None,
            1,
            1,
            False,
        )
    torch.cuda.synchronize()


@pytest.mark.parametrize("route,heads", [("fp8", 13), ("v32", 24)])
def test_standalone_scratch_capacity_rejected(route, heads, sm12x):
    t = _standalone_case(route, heads)
    backing = t["mid"]
    t["mid"] = backing.reshape(-1)[: 2 * heads * 2 * 512].view(2, heads, 2, 512)
    with pytest.raises(RuntimeError, match="mid_out|scratch"):
        _standalone_call(t, route)


@pytest.mark.parametrize("issue", ["dtype", "noncontiguous"])
def test_standalone_scratch_type_and_layout_rejected(issue, sm12x):
    t = _standalone_case()
    if issue == "dtype":
        t["mid"] = t["mid"].float()
    else:
        t["mid"] = t["mid"].transpose(0, 2)
    with pytest.raises(RuntimeError, match="mid_out|scratch|contiguous"):
        _standalone_call(t)


def test_standalone_v32_runtime_page(sm12x):
    from tests.attention.sparse_mla_test_utils import _ref_sparse_attn

    t = _standalone_case("v32", 16)
    latent = (
        torch.arange(3, device="cuda", dtype=torch.bfloat16)[:, None, None, None]
        .expand(3, 32, 1, 512)
        .contiguous()
    )
    t["cache"] = quantize_kv_glm53_nope(latent)
    t["idx"].random_(1, 96)
    t["idx"][:, 0] = 33
    expected, lse = _ref_sparse_attn(t["q"], latent, t["idx"], 512**-0.5, 512)
    _standalone_call(t, "v32")
    torch.testing.assert_close(t["out"], expected, atol=5e-2, rtol=5e-2)
    torch.testing.assert_close(t["lse"], lse, atol=5e-2, rtol=5e-2)


def test_scratch_rebind_uses_dense_prefix():
    from flashinfer.mla._sparse_mla_sm120 import _decode_scratch_views

    mid = torch.empty(4, 32, 8, 512, dtype=torch.bfloat16)
    mlse = torch.empty(4, 32, 8)
    out, lse = _decode_scratch_views(mid, mlse, 2, 13, 2, 512, scratch_heads=16)
    assert out.is_contiguous() and lse.is_contiguous()
    assert out.data_ptr() == mid.data_ptr() and lse.data_ptr() == mlse.data_ptr()


@pytest.mark.parametrize(
    "tokens,heads,full", [(2, 8, False), (128, 64, False), (128, 64, True)]
)
def test_dsv4_integer_pages_graph_and_alignment(tokens, heads, full, sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from tests.attention.sparse_mla_test_utils import (
        dequantize_kv_dsv4,
        _ref_sparse_attn,
    )

    torch.manual_seed(529)
    q = torch.randn(tokens, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    out = torch.empty_like(q)
    idx = torch.randint(1, 193, (tokens, 128), device="cuda", dtype=torch.int32)
    idx[:, 2] = -1
    lengths = (
        None if full else torch.full((tokens,), 121, device="cuda", dtype=torch.int32)
    )
    if lengths is not None:
        idx[:, 121:] = 2147483647
    wrapper = SparseMLASm120Wrapper()
    captures = []
    for page, extra_page in [(96, 192), (3, 65)]:
        caches, virtual = [], []
        for p in (page, extra_page):
            packed = quantize_kv_dsv4(
                torch.randn(
                    (193 + p - 1) // p, p, 1, 512, device="cuda", dtype=torch.bfloat16
                )
                * 0.1
            )
            virtual.append(dequantize_kv_dsv4(packed).reshape(-1, 512)[:193])
            pitch = (p * 584 + 15) // 16 * 16
            cache = torch.empty_strided(
                packed.shape, (pitch, 584, 584, 1), device="cuda", dtype=torch.uint8
            )
            cache.copy_(packed)
            raw = cache.as_strided((packed.shape[0], p * 584), (pitch, 1))
            raw[0, :576] = 255
            raw[0, p * 576 : p * 576 + 8] = 255
            caches.append(cache)
        ri = (
            idx
            if full
            else torch.where(
                torch.arange(128, device="cuda")[None, :] < lengths[:, None], idx, -1
            )
        )
        ref_indices = torch.cat([ri, torch.where(ri >= 0, ri + 193, -1)], -1)
        expected, _ = _ref_sparse_attn(
            q, torch.cat(virtual).reshape(-1, 1, 1, 512), ref_indices, 512**-0.5, 512
        )

        def call():
            wrapper.run(
                q,
                caches[0],
                idx,
                out,
                512**-0.5,
                topk_length=lengths,
                extra_kv_cache=caches[1],
                extra_indices=idx,
                extra_topk_length=lengths,
            )

        call()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
        saved = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
            call()
        captures.append((graph, caches, saved))
    for graph, _caches, saved in captures:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, saved)
    bad = torch.empty(2, 3, 1, 584, device="cuda", dtype=torch.uint8)
    with pytest.raises((ValueError, RuntimeError), match="16B-aligned"):
        wrapper.run(q, bad, idx, out, 512**-0.5)


@pytest.mark.parametrize("page", [3, 96])
def test_v32_page_gap_overrides_prefill_crossover(page, monkeypatch, sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _calibration, _policy
    from flashinfer.mla._sparse_mla_sm120 import _sparse_mla_sm120_paged_attention
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_dsv3_2,
        dequantize_kv_dsv3_2,
        _ref_sparse_attn,
    )

    torch.manual_seed(418)
    packed = quantize_kv_dsv3_2(
        torch.randn(4, page, 1, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    pitch = page * 672 + 16
    cache = torch.empty_strided(
        (4, page, 1, 672), (pitch, 672, 672, 1), device="cuda", dtype=torch.uint8
    )
    cache.fill_(255)
    cache[..., :656].copy_(packed)
    q = torch.randn(2, 64, 576, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(0, 4 * page, (2, 128), device="cuda", dtype=torch.int32)
    idx[:, 3] = -1
    expected, el = _ref_sparse_attn(
        q, dequantize_kv_dsv3_2(packed), idx, 576**-0.5, 512
    )
    out = torch.empty(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(2, 64, device="cuda")
    mid = torch.empty(2, 64, 2, 512, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(2, 64, 2, device="cuda")
    monkeypatch.setattr(
        _calibration,
        "get_ordinary_profile",
        lambda *args: {
            "buckets": {
                str(t): {"variant": int(_policy.KernelVariant.PREFILL_SWAPAB), "cpb": 1}
                for t in _calibration._PROFILE_T
            }
        },
    )

    wrapper = SparseMLASm120Wrapper()
    for functional in [False, True]:

        def call():
            if functional:
                _sparse_mla_sm120_paged_attention(
                    q, cache, idx, out, lse, 576**-0.5, mid_out=mid, mid_lse=mlse
                )
            else:
                wrapper.run(q, cache, idx, out, 576**-0.5, out_lse=lse)

        call()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
        torch.testing.assert_close(lse, el, atol=0.02, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)


@pytest.mark.parametrize("model", [0, 4])
def test_single_cache_pitched_indices_override_crossover(model, monkeypatch, sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from flashinfer.mla._sparse_mla_sm120 import _api, _calibration
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_dsv3_2,
        dequantize_kv_dsv3_2,
        quantize_kv_dots3_swa,
        dequantize_kv_dots3_swa,
        _ref_sparse_attn,
    )

    monkeypatch.setattr(_calibration, "get_decode_max_tokens", lambda *args: 0)
    torch.manual_seed(745)
    dim, value_dim, topk = (576, 512, 128) if model == 0 else (1088, 1024, 576)
    quantize, dequantize = (
        (quantize_kv_dsv3_2, dequantize_kv_dsv3_2)
        if model == 0
        else (quantize_kv_dots3_swa, dequantize_kv_dots3_swa)
    )
    cache = quantize(
        torch.randn(4, 64, 1, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    q = torch.randn(4, 8, dim, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(256, (4, topk + 16), device="cuda", dtype=torch.int32)[:, :topk]
    ri = idx.clone()
    if model == 4:
        ri[:, 513:] = -1
    expected, el = _ref_sparse_attn(q, dequantize(cache), ri, dim**-0.5, value_dim)
    out = torch.empty(4, 8, value_dim, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(4, 8, device="cuda")
    splits = _api._decode_dsv4_num_splits(topk, model_type=model)
    mid = torch.empty(4, 8, splits, value_dim, device="cuda", dtype=torch.bfloat16)
    mlse = torch.empty(4, 8, splits, device="cuda")
    wrapper = SparseMLASm120Wrapper(d_v=value_dim)
    for legacy in (False, True):

        def call():
            if legacy:
                _api.get_sparse_mla_sm120_module().paged_attention(
                    q,
                    cache,
                    idx,
                    out,
                    lse,
                    dim**-0.5,
                    value_dim,
                    model,
                    0,
                    None,
                    None,
                    None,
                    None,
                    None,
                    mid,
                    mlse,
                    False,
                )
            else:
                wrapper.run(q, cache, idx, out, dim**-0.5, out_lse=lse)

        call()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
        torch.testing.assert_close(lse, el, atol=0.02, rtol=0.02)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)


@pytest.mark.parametrize("api", ["wrapper", "functional", "legacy"])
@pytest.mark.parametrize("pitched", ["main", "extra"])
def test_dsv4_metadata_overrides_prefill_crossover(api, pitched, monkeypatch, sm12x):
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        trtllm_batch_decode_sparse_mla_dsv4,
    )
    from flashinfer.mla._sparse_mla_sm120 import _calibration, _api, _prepared
    from tests.attention.sparse_mla_test_utils import (
        dequantize_kv_dsv4,
        _ref_sparse_attn,
    )

    _prepared._functional_plans.clear()

    monkeypatch.setattr(
        _calibration,
        "get_ordinary_profile",
        lambda *args: {
            "buckets": {
                str(t): {"variant": 3, "cpb": 1} for t in _calibration._PROFILE_T
            }
        },
    )
    torch.manual_seed(743)
    caches = [
        quantize_kv_dsv4(
            torch.randn(4, page, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        )
        for page in (64, 96)
    ]
    virtual = torch.cat(
        [dequantize_kv_dsv4(cache).reshape(-1, 512) for cache in caches]
    ).reshape(-1, 1, 1, 512)
    wrapper = SparseMLASm120Wrapper()
    workspace = torch.empty(8 << 20, device="cuda", dtype=torch.uint8)
    legacy = _api.get_sparse_mla_sm120_module()
    cases = [(4, True), (65, True)]
    if pitched == "main":
        cases.insert(0, (4, False))
    for tokens, strided in cases:
        q = torch.randn(tokens, 16, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        out = torch.empty_like(q)
        lse = torch.empty(tokens, 16, device="cuda")
        indices = []
        for name, slots in [("main", 256), ("extra", 384)]:
            width = 144 if strided and pitched == name else 128
            idx = torch.randint(
                slots, (tokens, width), device="cuda", dtype=torch.int32
            )
            indices.append(idx[:, :128])
        mid = torch.empty(tokens, 16, 4, 512, device="cuda", dtype=torch.bfloat16)
        mlse = torch.empty(tokens, 16, 4, device="cuda")
        lengths = torch.full((tokens,), 128, device="cuda", dtype=torch.int32)

        def call():
            if api == "wrapper":
                wrapper.run(
                    q,
                    caches[0],
                    indices[0],
                    out,
                    512**-0.5,
                    extra_kv_cache=caches[1],
                    extra_indices=indices[1],
                    out_lse=lse,
                )
            elif api == "functional":
                trtllm_batch_decode_sparse_mla_dsv4(
                    q.unsqueeze(1),
                    caches[0],
                    workspace,
                    indices[0],
                    compressed_kv_cache=caches[1],
                    extra_sparse_indices=indices[1],
                    out=out.unsqueeze(1),
                    bmm1_scale=512**-0.5,
                    swa_topk_lens=lengths,
                    extra_sparse_topk_lens=lengths,
                    backend="sparse",
                    kv_layout="NHD",
                )
            else:
                legacy.paged_attention(
                    q,
                    caches[0],
                    indices[0],
                    out,
                    lse,
                    512**-0.5,
                    512,
                    1,
                    0,
                    None,
                    None,
                    caches[1],
                    indices[1],
                    None,
                    mid,
                    mlse,
                    False,
                )

        if tokens > 64:
            with pytest.raises(
                (ValueError, RuntimeError), match="no .*kernel|reject|unsupported"
            ):
                call()
            continue
        expected, el = _ref_sparse_attn(
            q, virtual, torch.cat([indices[0], indices[1] + 256], -1), 512**-0.5, 512
        )
        call()
        torch.cuda.synchronize()
        torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
        if api != "functional":
            torch.testing.assert_close(lse, el, atol=0.02, rtol=0.02)
        saved = out.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, saved)
        plans = (
            list(wrapper._prepared_calls.values())
            if api == "wrapper"
            else list(_prepared._functional_plans.values())
        )
        variants = {p.plan.inspect()["variant"] for p in plans}
        assert (0 if strided else 3) in variants


@pytest.mark.parametrize("tokens,page", [(2, 3), (128, 65)])
def test_dots3_runtime_footer_page(tokens, page, sm12x):
    from flashinfer.mla import SparseMLASm120Wrapper
    from tests.attention.sparse_mla_test_utils import (
        quantize_kv_dots3_swa,
        dequantize_kv_dots3_swa,
        _ref_sparse_attn,
    )

    torch.manual_seed(671)
    packed = quantize_kv_dots3_swa(
        torch.randn(4, page, 1, 1088, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    virtual = dequantize_kv_dots3_swa(packed)
    pitch = (page * 1160 + 15) // 16 * 16
    cache = torch.empty_strided(
        (4, page, 1, 1160), (pitch, 1160, 1160, 1), device="cuda", dtype=torch.uint8
    )
    cache.copy_(packed)
    raw = cache.as_strided((4, page * 1160), (pitch, 1))
    raw[0, :1152] = 255
    raw[0, page * 1152 : page * 1152 + 8] = 255
    q = torch.randn(tokens, 8, 1088, device="cuda", dtype=torch.bfloat16) * 0.1
    idx = torch.randint(1, 4 * page, (tokens, 576), device="cuda", dtype=torch.int32)
    idx[:, 3] = -1
    idx[:, 513:] = 2147483647
    ref_idx = idx.clone()
    ref_idx[:, 513:] = -1
    expected, el = _ref_sparse_attn(q, virtual, ref_idx, 1088**-0.5, 1024)
    out = torch.empty(tokens, 8, 1024, device="cuda", dtype=torch.bfloat16)
    lse = torch.empty(tokens, 8, device="cuda")
    wrapper = SparseMLASm120Wrapper(d_v=1024)

    def call():
        wrapper.run(q, cache, idx, out, 1088**-0.5, out_lse=lse)

    call()
    torch.cuda.synchronize()
    torch.testing.assert_close(out, expected, atol=0.005, rtol=0.05)
    torch.testing.assert_close(lse, el, atol=0.02, rtol=0.02)
    saved = out.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(out, saved)
