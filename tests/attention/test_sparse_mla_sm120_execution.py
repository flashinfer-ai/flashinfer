import pytest
import torch

from flashinfer.mla._sparse_mla_sm120_execution import (
    resolve_dsv4_nvfp4,
    resolve_attention,
)
from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module
from tests.attention.sparse_mla_test_utils import (
    quantize_kv_dsv4,
    quantize_kv_glm53_nope,
    require_sm12x,
)


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
    from flashinfer.mla._sparse_mla_sm120_execution import (
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
    from flashinfer.mla._sparse_mla_sm120_execution import (
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
    from flashinfer.mla import _sparse_mla_sm120_prepared as prepared
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


def test_hot_model_pick_refines_in_tuning_and_capture_does_not_fail(monkeypatch):
    import torch
    from types import SimpleNamespace
    from flashinfer.mla import _sparse_mla_sm120_policy as policy

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

    def refine(*a):
        state["refines"] += 1
        return 2

    monkeypatch.setattr(policy._cpb, "refine_cpb", refine)
    policy._cpb_hot_cache.clear()
    args = (torch.device("cuda:0"), "dsv4", 4, 64, 512, 0)
    assert policy._resolve_cpb(*args) == 1
    state["tuning"] = True
    assert policy._resolve_cpb(*args) == 2 and state["refines"] == 1
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
    from flashinfer.mla import _sparse_mla_sm120_policy as policy
    from flashinfer.mla import _sparse_mla_sm120_prepared as prepared
    from tests.attention.sparse_mla_test_utils import quantize_kv_dsv4

    q = torch.randn(4, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    cache = quantize_kv_dsv4(
        torch.randn(2, 64, 1, 512, device="cuda", dtype=torch.bfloat16) * 0.1
    )
    idx = torch.randint(128, (4, 128), device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    state = {"threshold": None, "tune": False}
    monkeypatch.setattr(
        policy._cpb, "get_decode_max_tokens", lambda *a: state["threshold"]
    )

    def update(*a):
        if state["tune"]:
            state["threshold"] = 0
            monkeypatch.setattr(
                policy._cpb, "_constants_version", policy._cpb._constants_version + 1
            )
        return 1

    monkeypatch.setattr(policy, "_resolve_cpb", update)
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


def test_standalone_v32_page_size_rejected(sm12x):
    t = _standalone_case("v32", 16)
    t["cache"] = t["cache"][:, :32]
    with pytest.raises(RuntimeError, match="page.*64"):
        _standalone_call(t, "v32")


def test_scratch_rebind_uses_dense_prefix():
    from flashinfer.mla._sparse_mla_sm120 import _decode_scratch_views

    mid = torch.empty(4, 32, 8, 512, dtype=torch.bfloat16)
    mlse = torch.empty(4, 32, 8)
    out, lse = _decode_scratch_views(mid, mlse, 2, 13, 2, 512)
    assert out.is_contiguous() and lse.is_contiguous()
    assert out.data_ptr() == mid.data_ptr() and lse.data_ptr() == mlse.data_ptr()
