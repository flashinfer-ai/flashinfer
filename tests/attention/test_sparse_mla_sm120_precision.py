from contextlib import contextmanager

import pytest
import torch

from flashinfer.mla import SparseMLASm120Wrapper
from flashinfer.mla._sparse_mla_sm120 import _policy as planner
from tests.attention.sparse_mla_test_utils import (
    inputs,
    _ref_sparse_attn,
    dequantize_kv_dsv4_1,
)


@contextmanager
def _no_capture_allocations(wrapper, device):
    prepared = tuple(wrapper._prepared_calls.values())
    resources = tuple((entry, entry.mid, entry.mlse, entry.lse) for entry in prepared)
    allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    yield
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocated
    assert tuple(map(id, wrapper._prepared_calls.values())) == tuple(map(id, prepared))
    for entry, mid, mlse, lse in resources:
        assert entry.mid is mid and entry.mlse is mlse and entry.lse is lse


def _kernel_only_graph(topology):
    import re

    nodes = re.findall(r'"graph_\d+_node_\d+"\[.*?\];', topology, re.DOTALL)
    assert nodes and all('label="{KERNEL' in node for node in nodes)


def _require_sm12x():
    from flashinfer.utils import is_sm12x_supported

    if not torch.cuda.is_available() or not is_sm12x_supported(torch.device("cuda")):
        pytest.skip("Sparse-MLA SM120 requires SM12x.")


def test_capture_allocation_guard_tracks_allocations_not_live_bytes():
    _require_sm12x()
    wrapper = SparseMLASm120Wrapper(
        compute_precision="bf16", kv_scale_format="ue8m0_g32"
    )
    device = torch.device("cuda")
    unrelated = torch.empty(1 << 20, device=device)
    with _no_capture_allocations(wrapper, device):
        del unrelated
    with pytest.raises(AssertionError), _no_capture_allocations(wrapper, device):
        temporary = torch.empty(1024, device=device)
        del temporary


@pytest.mark.parametrize("precision", ["default", "fp8", "bf16"])
@pytest.mark.parametrize(
    "heads,mp,ep,mixed", [(13, 61, 53, True), (8, 64, 2, False), (64, 256, 64, True)]
)
def test_precision_eager_and_graph(precision, heads, mp, ep, mixed):
    q, main, idx, kwargs, (ref, rlse) = inputs(heads, mp, ep, mixed)
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", extra_kv_fp4=mixed, compute_precision=precision
    )
    out = torch.empty_like(q)
    lse = wrapper.run(q, main, idx, out, 512**-0.5, return_lse=True, **kwargs)
    expected = out.clone(), lse.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(q, main, idx, out, 512**-0.5, **kwargs)
    for _ in range(3):
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)
    assert (
        (out.float() - ref.float()).square().mean() / ref.float().square().mean()
    ).sqrt() < 0.05


@pytest.mark.parametrize("precision", ["fp8", "bf16"])
def test_precision_single_cache_and_capture_requires_warmup(precision, monkeypatch):
    q, main, idx, _, _ = inputs()
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision=precision
    )
    out = torch.empty_like(q)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(ValueError, match="warm up"):
            wrapper.run(q, main, idx, out, 512**-0.5)
    lse = wrapper.run(q, main, idx, out, 512**-0.5, return_lse=True)
    ref, rlse = _ref_sparse_attn(q, dequantize_kv_dsv4_1(main), idx, 512**-0.5, 512)
    torch.testing.assert_close(out, ref, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)
    assert (
        (out.float() - ref.float()).square().mean() / ref.float().square().mean()
    ).sqrt() < 0.05


def test_precision_planner_isolation(monkeypatch):
    from flashinfer.mla._sparse_mla_sm120 import _execution as execution

    def no_compilation(*args, **kwargs):
        raise AssertionError("pure planner isolation must not load a compiled module")

    monkeypatch.setattr(execution, "get_sparse_mla_sm120_module", no_compilation)

    def candidates(model, heads, topk, page, extra):
        assert (model, heads, topk, page, extra) == (5, 13, 128, 61, True)
        return frozenset({0})

    monkeypatch.setattr(planner, "_candidates", candidates)

    def old_calibration(*args, **kwargs):
        raise AssertionError("explicit precision consulted legacy calibration")

    monkeypatch.setattr(planner, "_resolve_cpb", old_calibration)
    monkeypatch.setattr(planner._cpb, "get_decode_max_tokens", old_calibration)
    for precision in ["fp8", "bf16"]:
        result = planner.plan(
            4,
            13,
            128,
            5,
            61,
            True,
            0,
            torch.device("cuda"),
            extra_topk=77,
            extra_fp4=True,
            compute_precision=precision,
        )
        assert result.variant is planner.KernelVariant.DECODE_SPLITK and result.cpb == 1


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"compute_precision": "nvfp4"}, "compute_precision"),
        ({"compute_precision": "bf16"}, "DSV4.1"),
        ({"compute_precision": "fp8", "kv_cache_format": "nvfp4"}, "DSV4.1"),
    ],
)
def test_precision_unsupported_storage(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SparseMLASm120Wrapper(**kwargs)


@pytest.mark.parametrize("precision", ["fp8"])
def test_precision_no_prefill_fallback(precision):
    q, main, idx, kwargs, _ = inputs()
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision=precision
    )
    q = q[:1].expand(65, -1, -1).contiguous()
    with pytest.raises(ValueError, match="prefill.*heads"):
        wrapper.run(
            q, main, idx[:1].expand(65, -1).contiguous(), torch.empty_like(q), 512**-0.5
        )


@pytest.mark.parametrize("mixed", [None, False, True])
def test_fp8_prefill_cross_bucket_graphs(mixed, tmp_path):
    from flashinfer.mla._sparse_mla_sm120 import _get_sparse_mla_sm120_decode_module

    q, main, idx, kwargs, reference = inputs(16, 61, 53, mixed is True)
    if mixed is None:
        kwargs = {k: v for k, v in kwargs.items() if not k.startswith("extra")}
        masked = idx.masked_fill(
            torch.arange(128, device="cuda")[None] >= kwargs["topk_length"][:, None], -1
        )
        reference = _ref_sparse_attn(
            q,
            dequantize_kv_dsv4_1(main),
            masked,
            512**-0.5,
            512,
            attn_sink=kwargs["attn_sink"],
        )
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", extra_kv_fp4=mixed is True, compute_precision="fp8"
    )
    module = _get_sparse_mla_sm120_decode_module()
    states = []
    for nt in [4, 65, 129]:

        def repeat(x):
            return x.repeat((nt + 3) // 4, *([1] * (x.ndim - 1)))[:nt].contiguous()

        nq, ni = repeat(q), repeat(idx)
        kw = {
            k: repeat(v)
            if k in ("topk_length", "extra_indices", "extra_topk_length")
            else v
            for k, v in kwargs.items()
        }
        output = torch.empty_like(nq)
        lse = wrapper.run(nq, main, ni, output, 512**-0.5, return_lse=True, **kw)
        expected = output.clone(), lse.clone()
        if nt > 64:
            direct, dlse = torch.empty_like(nq), torch.empty_like(lse)
            module.sparse_mla_sm120_paged_attention(
                nq,
                main,
                ni,
                direct,
                dlse,
                512**-0.5,
                5,
                1,
                kw.get("topk_length"),
                kw.get("attn_sink"),
                kw.get("extra_kv_cache"),
                kw.get("extra_indices"),
                kw.get("extra_topk_length"),
                mixed is True,
            )
            assert torch.equal(direct, output) and torch.equal(dlse, lse)
        ref, rlse = (repeat(x) for x in reference)
        torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
        torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)
        assert (
            (output.float() - ref.float()).square().mean() / ref.float().square().mean()
        ).sqrt() < 0.05
        pitched_lse = torch.empty(nt, 21, device="cuda")[:, :16]
        wrapper.run(nq, main, ni, output, 512**-0.5, out_lse=pitched_lse, **kw)
        assert torch.equal(pitched_lse, lse)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph), _no_capture_allocations(wrapper, nq.device):
            wrapper.run(nq, main, ni, output, 512**-0.5, **kw)
        graph.instantiate()
        dot = tmp_path / f"route-{nt}.dot"
        graph.debug_dump(str(dot))
        topology = dot.read_text()
        _kernel_only_graph(topology)
        if nt > 64:
            assert "sparse_mla_prefill_kernel" in topology
            assert "sparse_mla_decode" not in topology
        else:
            assert "sparse_mla_decode_dsv4_kernel" in topology
            assert "sparse_mla_decode_dsv4_merge_kernel" in topology
        states.append((graph, output, lse, expected, nq, ni, kw))
    assert {
        p.workspace[2][0][0]
        for p in wrapper._prepared_calls.values()
        if p.workspace[0][2]
    } == {4}
    for graph, output, lse, expected, *_ in states * 3:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected[0]) and torch.equal(lse, expected[1])
    selected = planner.plan(
        65,
        16,
        128,
        5,
        61,
        mixed is not None,
        2,
        q.device,
        extra_topk=77 if mixed is not None else 0,
        extra_fp4=mixed is True,
        compute_precision="fp8",
    )
    assert selected.variant is planner.KernelVariant.PREFILL_SG
    with pytest.raises(ValueError, match="swapab"):
        planner.plan(65, 16, 128, 5, 61, False, 1, q.device, compute_precision="fp8")


@pytest.mark.parametrize(
    "precision,tokens", [("fp8", 4), ("fp8", 65), ("bf16", 4), ("bf16", 65)]
)
@pytest.mark.parametrize("sink_value", [None, -1000.0, 2.0, 1000.0])
def test_precision_empty_effective_kv(precision, tokens, sink_value):
    q, main, idx, _, _ = inputs(16, 64, 64, False)
    q = q[:1].expand(tokens, -1, -1).contiguous()
    idx = idx[:1].expand(tokens, -1).contiguous()
    lengths = torch.full((tokens,), 128, device="cuda", dtype=torch.int32)
    lengths[0] = 0
    idx[1] = -1
    sink = None if sink_value is None else torch.full((16,), sink_value, device="cuda")
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision=precision
    )
    output = torch.full_like(q, float("nan"))
    lse = torch.full((tokens, 16), float("nan"), device="cuda")
    wrapper.run(
        q,
        main,
        idx,
        output,
        512**-0.5,
        topk_length=lengths,
        attn_sink=sink,
        out_lse=lse,
    )
    ref, rlse = _ref_sparse_attn(
        q,
        dequantize_kv_dsv4_1(main),
        idx,
        512**-0.5,
        512,
        topk_length=lengths,
        attn_sink=sink,
    )
    assert torch.count_nonzero(output[:2]) == 0
    if sink is not None:
        torch.testing.assert_close(lse[:2], (sink * 1.4426950408889634).expand(2, -1))
    torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)

    valid_idx = idx[2].clone()
    expected_output, expected_lse = output[2].clone(), lse[2].clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(
            q,
            main,
            idx,
            output,
            512**-0.5,
            topk_length=lengths,
            attn_sink=sink,
            out_lse=lse,
        )
    for empty in (True, False):
        if empty:
            idx[2].fill_(-1)
            lengths[3] = 0
        else:
            idx[2].copy_(valid_idx)
            lengths[3] = 128
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        if empty:
            assert torch.count_nonzero(output[2:4]) == 0
            if sink is None:
                assert torch.isneginf(lse[2:4]).all()
            else:
                torch.testing.assert_close(
                    lse[2:4], (sink * 1.4426950408889634).expand(2, -1)
                )
        else:
            torch.testing.assert_close(
                output[2:4], expected_output.expand(2, -1, -1), atol=0, rtol=0
            )
            torch.testing.assert_close(
                lse[2:4], expected_lse.expand(2, -1), atol=0, rtol=0
            )


def test_precision_independent_graphs():
    q, main, idx, kwargs, _ = inputs()
    states = []
    for precision in ["fp8", "bf16"]:
        wrapper = SparseMLASm120Wrapper(
            kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision=precision
        )
        out = torch.empty_like(q)
        lse = wrapper.run(q, main, idx, out, 512**-0.5, return_lse=True, **kwargs)
        expected = out.clone(), lse.clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            wrapper.run(q, main, idx, out, 512**-0.5, **kwargs)
        states.append((wrapper, graph, out, lse, expected))
    assert (
        next(iter(states[0][0]._prepared_calls.values())).mid.data_ptr()
        != next(iter(states[1][0]._prepared_calls.values())).mid.data_ptr()
    )
    for wrapper, _, _, _, _ in states:
        nq, nm, ni, nkw, _ = inputs(64, 256, 64)
        wrapper.run(nq, nm, ni, torch.empty_like(nq), 512**-0.5, **nkw)
    for _, graph, out, lse, expected in states * 3:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(out, expected[0]) and torch.equal(lse, expected[1])


@pytest.mark.parametrize("extra_pbs", [None, 2, 64])
def test_nvfp4_precision_cross_bucket_graphs(extra_pbs, tmp_path):
    from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache
    from tests.attention.sparse_mla_test_utils import (
        _dequantize_nvfp4_cache,
        _dequantize_nvfp4_query,
        _reference_sparse_attention,
    )

    _require_sm12x()
    torch.manual_seed(641)
    wrapper = SparseMLASm120Wrapper(kv_cache_format="nvfp4", compute_precision="nvfp4")
    default = SparseMLASm120Wrapper(kv_cache_format="nvfp4")
    main = nvfp4_quantize_pack_sparse_mla_cache(
        torch.randn(8, 64, 512, device="cuda", dtype=torch.bfloat16) * 0.1,
        kv_layout="NHD",
    )
    extra = (
        None
        if extra_pbs is None
        else nvfp4_quantize_pack_sparse_mla_cache(
            torch.randn(8, extra_pbs, 512, device="cuda", dtype=torch.bfloat16) * 0.1,
            kv_layout="HND",
        )
    )
    cached = _dequantize_nvfp4_cache(main).reshape(-1, 512)
    if extra is not None:
        cached = torch.cat([cached, _dequantize_nvfp4_cache(extra).reshape(-1, 512)])

    def pitched(cache):
        storage = torch.empty(
            cache.shape[0], cache[0].numel() + 512, device="cuda", dtype=torch.uint8
        )
        view = storage[:, : cache[0].numel()].view_as(cache)
        view.copy_(cache)
        return view

    main = pitched(main)
    extra = pitched(extra) if extra is not None else None
    states = []
    for nt, heads, topk in [(2, 16, 128), (8, 64, 512), (65, 32, 128), (129, 128, 512)]:
        q = torch.randn(nt, heads, 512, device="cuda", dtype=torch.bfloat16) * 0.1
        idx = torch.randint(512, (nt, topk), device="cuda", dtype=torch.int32)
        idx[:, 8:16] = -1
        idx[:, 32:48] = 3
        lens = torch.full((nt,), topk - 9, device="cuda", dtype=torch.int32)
        lens[0] = 0
        sink = torch.linspace(-1, 1, heads, device="cuda")
        kw = dict(topk_length=lens, attn_sink=sink)
        mi = idx.masked_fill(
            torch.arange(topk, device="cuda")[None] >= lens[:, None], -1
        )
        if extra is not None:
            ei = torch.randint(
                8 * extra_pbs, (nt, 77), device="cuda", dtype=torch.int32
            )
            el = torch.full((nt,), 73, device="cuda", dtype=torch.int32)
            el[0] = 0
            kw.update(extra_kv_cache=extra, extra_indices=ei, extra_topk_length=el)
            masked = ei.masked_fill(
                torch.arange(77, device="cuda")[None] >= el[:, None], -1
            )
            mi = torch.cat([mi, torch.where(masked < 0, masked, masked + 512)], -1)
        output = torch.empty_like(q)
        lse = wrapper.run(q, main, idx, output, 512**-0.5, return_lse=True, **kw)
        expected = output.clone(), lse.clone()
        legacy = torch.empty_like(q)
        legacy_lse = default.run(q, main, idx, legacy, 512**-0.5, return_lse=True, **kw)
        assert torch.equal(legacy, output) and torch.equal(legacy_lse, lse)
        for query in (_dequantize_nvfp4_query(q), q.float()):
            ref, rlse = _reference_sparse_attention(
                query, cached, mi, 512**-0.5, attn_sink=sink
            )
            torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
            torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph), _no_capture_allocations(wrapper, q.device):
            wrapper.run(q, main, idx, output, 512**-0.5, prefill_impl="auto", **kw)
        graph.instantiate()
        dot = tmp_path / f"dsv4-nvfp4-{nt}.dot"
        graph.debug_dump(str(dot))
        topology = dot.read_text()
        _kernel_only_graph(topology)
        selected = list(wrapper._prepared_calls.values())[-1].plan
        kernels = [line for line in topology.splitlines() if "sparse_mla_" in line]
        if selected.inspect()["implementation"] == "dsv4_nvfp4_prefill":
            assert len(kernels) == 1
            assert "sparse_mla_streaming_dsv4_nvfp4_kernel" in kernels[0]
            assert "merge" not in topology
        else:
            assert selected.inspect()["implementation"] in (
                "dsv4_nvfp4_decode",
                "dsv4_nvfp4_grouped_decode",
            )
            assert any(
                "sparse_mla_decode_dsv4_nvfp4_kernel" in line
                or "sparse_mla_streaming_dsv4_nvfp4_kernel" in line
                for line in kernels
            )
            assert 1 <= len(kernels) <= 2
            if len(kernels) == 2:
                assert (
                    sum(
                        "sparse_mla_decode_dsv4_merge_kernel" in line
                        or "sparse_mla_decode_dsv4_nvfp4_merge2_kernel" in line
                        for line in kernels
                    )
                    == 1
                )
        states.append((graph, output, lse, expected, q, idx, kw))
    for graph, output, lse, expected, *_ in states * 3:
        output.fill_(float("nan"))
        lse.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected[0]) and torch.equal(lse, expected[1])


def test_nvfp4_precision_envelope_and_warmup(monkeypatch):
    _require_sm12x()
    from flashinfer.mla import nvfp4_quantize_pack_sparse_mla_cache

    cache = nvfp4_quantize_pack_sparse_mla_cache(
        torch.zeros(2, 64, 512, device="cuda", dtype=torch.bfloat16)
    )
    wrapper = SparseMLASm120Wrapper(kv_cache_format="nvfp4", compute_precision="nvfp4")
    for heads, topk in [(8, 128), (16, 64)]:
        q = torch.zeros(2, heads, 512, device="cuda", dtype=torch.bfloat16)
        idx = torch.zeros(2, topk, device="cuda", dtype=torch.int32)
        with pytest.raises(ValueError, match="no NVFP4"):
            wrapper.run(q, cache, idx, torch.empty_like(q), 512**-0.5)
    q = torch.zeros(2, 16, 512, device="cuda", dtype=torch.bfloat16)
    idx = torch.zeros(2, 128, device="cuda", dtype=torch.int32)
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        with pytest.raises(ValueError, match="warm up"):
            wrapper.run(q, cache, idx, torch.empty_like(q), 512**-0.5)
    with pytest.raises(ValueError, match="no NVFP4"):
        wrapper.run(q, cache[:, :, :2], idx, torch.empty_like(q), 512**-0.5)
    with pytest.raises(ValueError, match="prefill_impl"):
        wrapper.run(
            q, cache, idx, torch.empty_like(q), 512**-0.5, prefill_impl="swapab"
        )


@pytest.mark.parametrize(
    "heads,topk,mixed",
    [
        (1, 65, True),
        (13, 65, True),
        (17, 63, False),
        (128, 129, True),
        (8, 1, None),
        (64, 128, False),
    ],
)
def test_bf16_prefill_runtime_shapes(heads, topk, mixed, tmp_path):
    from tests.attention.sparse_mla_test_utils import dequantize_kv_dsv4_1_fp4

    q, main, _, kwargs, _ = inputs(heads, 61, 53, mixed is True)
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32",
        extra_kv_fp4=mixed is True,
        compute_precision="bf16",
    )
    states = []
    for nt in [4, 65, 129]:
        query = q.repeat((nt + 3) // 4, 1, 1)[:nt].contiguous()
        indices = torch.randint(244, (nt, topk + 3), device="cuda", dtype=torch.int32)[
            :, :topk
        ]
        indices[:, 2::3] = -1
        lengths = torch.full((nt,), topk, device="cuda", dtype=torch.int32)
        lengths[::4] = 0
        lengths[1::4] = max(0, topk - 7)
        kw = dict(topk_length=lengths, attn_sink=kwargs["attn_sink"])
        virtual = dequantize_kv_dsv4_1(main).reshape(-1, 512)
        masked = indices.masked_fill(
            torch.arange(topk, device="cuda")[None] >= lengths[:, None], -1
        )
        if mixed is not None:
            extra = kwargs["extra_kv_cache"]
            exidx = torch.randint(212, (nt, 82), device="cuda", dtype=torch.int32)[
                :, :77
            ]
            exidx[:, 2::3] = -1
            exidx[:, 32:65] = 3
            exlens = torch.full((nt,), 77, device="cuda", dtype=torch.int32)
            exlens[::4] = 0
            exlens[2::4] = 33
            kw.update(
                extra_kv_cache=extra, extra_indices=exidx, extra_topk_length=exlens
            )
            dequant = dequantize_kv_dsv4_1_fp4 if mixed else dequantize_kv_dsv4_1
            virtual = torch.cat([virtual, dequant(extra).reshape(-1, 512)])
            ei = exidx.masked_fill(
                torch.arange(77, device="cuda")[None] >= exlens[:, None], -1
            )
            masked = torch.cat([masked, torch.where(ei < 0, ei, ei + 244)], -1)
        old_tf32 = torch.backends.cuda.matmul.allow_tf32
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            ref, rlse = _ref_sparse_attn(
                query,
                virtual.reshape(-1, 1, 1, 512),
                masked,
                512**-0.5,
                512,
                attn_sink=kw["attn_sink"],
            )
        finally:
            torch.backends.cuda.matmul.allow_tf32 = old_tf32
        output = torch.empty_like(query)
        lse = torch.empty(nt, heads + 3, device="cuda")[:, :heads]
        wrapper.run(query, main, indices, output, 512**-0.5, out_lse=lse, **kw)
        torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
        torch.testing.assert_close(lse, rlse, atol=0.02, rtol=0.02)
        assert (
            (output.float() - ref.float()).square().mean() / ref.float().square().mean()
        ).sqrt() < 0.05
        prepared = list(wrapper._prepared_calls.values())[-1]
        if nt > 64:
            assert prepared.plan.inspect()["merge"] == "direct"
            assert prepared.plan.inspect()["partial_bytes"] == 0
            assert prepared.plan.inspect()["lse_bytes"] == 0
        expected = output.clone(), lse.clone()
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        graph.enable_debug_mode()
        with torch.cuda.graph(graph), _no_capture_allocations(wrapper, query.device):
            wrapper.run(query, main, indices, output, 512**-0.5, out_lse=lse, **kw)
        graph.instantiate()
        dot = tmp_path / f"bf16-{nt}.dot"
        graph.debug_dump(str(dot))
        text = dot.read_text()
        _kernel_only_graph(text)
        if nt > 64:
            assert "sparse_mla_prefill_dsv41_bf16_kernel" in text
            assert "sparse_mla_decode" not in text
        states.append((graph, output, lse, expected, query, indices, kw))
    for graph, output, lse, expected, *_ in states * 3:
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(output, expected[0]) and torch.equal(lse, expected[1])


@pytest.mark.parametrize("tokens", [4, 65])
def test_bf16_fp4_extra_exact_codes_and_scales(tokens):
    from tests.attention.sparse_mla_test_utils import dequantize_kv_dsv4_1_fp4

    _require_sm12x()
    device = torch.device("cuda")
    main = torch.zeros(1, 1, 1, 528, dtype=torch.uint8, device=device)
    extra = torch.empty(1, 1, 1, 288, dtype=torch.uint8, device=device)
    raw = extra.view(-1)
    raw[:256] = torch.arange(256, device=device).to(torch.uint8)
    raw[256:] = torch.tensor(
        [1, 7, 8, 16, 32, 48, 56, 64, 80, 96, 112, 126, 129, 136, 184, 254] * 2,
        device=device,
        dtype=torch.uint8,
    )
    query = torch.zeros(tokens, 13, 512, device=device, dtype=torch.bfloat16)
    indices = torch.full((tokens, 1), -1, device=device, dtype=torch.int32)
    extra_indices = torch.zeros_like(indices)
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", extra_kv_fp4=True, compute_precision="bf16"
    )
    output = torch.empty_like(query)
    kwargs = dict(extra_kv_cache=extra, extra_indices=extra_indices)
    lse = wrapper.run(
        query, main, indices, output, 512**-0.5, return_lse=True, **kwargs
    )
    expected = dequantize_kv_dsv4_1_fp4(extra).reshape(1, 1, 512).expand_as(output)
    torch.testing.assert_close(output, expected, atol=0, rtol=0)
    assert torch.count_nonzero(lse) == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        wrapper.run(query, main, indices, output, 512**-0.5, **kwargs)
    output.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output, expected, atol=0, rtol=0)


def test_bf16_prefill_optional_lengths_and_empty_rows():
    q, main, _, kwargs, _ = inputs(13)
    query = q[:1].expand(65, -1, -1).contiguous()
    indices = torch.full((65, 65), -1, device="cuda", dtype=torch.int32)
    indices[1:, :63] = 3
    wrapper = SparseMLASm120Wrapper(
        kv_scale_format="ue8m0_g32", compute_precision="bf16"
    )
    output = torch.empty_like(query)
    lse = wrapper.run(query, main, indices, output, 512**-0.5, return_lse=True)
    assert torch.count_nonzero(output[0]) == 0
    assert torch.isneginf(lse[0]).all()
    old_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        ref, rlse = _ref_sparse_attn(
            query, dequantize_kv_dsv4_1(main), indices, 512**-0.5, 512
        )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = old_tf32
    torch.testing.assert_close(output, ref, atol=0.05, rtol=0.05)
    torch.testing.assert_close(lse[1:], rlse[1:], atol=0.02, rtol=0.02)
    sink = torch.full((13,), 1000.0, device="cuda")
    lse = wrapper.run(
        query, main, indices, output, 512**-0.5, attn_sink=sink, return_lse=True
    )
    assert torch.isfinite(output).all() and torch.isfinite(lse).all()
    assert torch.count_nonzero(output) == 0
    torch.testing.assert_close(lse, (sink * 1.4426950408889634).expand_as(lse))
