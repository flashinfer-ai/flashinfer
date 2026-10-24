from dataclasses import replace

import pytest
import torch

from ..conftest import require_b12x


def _query():
    from b12x.attention.varlen._tuning import VarlenAttentionQuery

    return VarlenAttentionQuery(
        variant="varlen",
        dtype="bfloat16",
        causal=False,
        batch_size=4,
        q_heads=16,
        kv_heads=16,
        q_head_dim=72,
        v_head_dim=72,
        query_rows=256,
        kv_rows=256,
        max_seqlen_q=256,
        max_seqlen_k=256,
        math_mode="eager_bf16",
    )


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": "float16"},
        {"variant": "batched"},
        {"causal": True},
        {"kv_heads": 8},
        {"block_sparse": True},
        {"per_segment_tiles": True},
        {"math_mode": "unknown"},
    ],
)
def test_eager_bf16_contract_rejects_unsupported_semantics(change):
    from b12x.attention.varlen._tuning import TUNING

    TUNING.validate_query(_query(), None)
    with pytest.raises(ValueError):
        TUNING.validate_query(replace(_query(), **change), None)


def test_eager_bf16_contract_is_semantic_and_scratch_is_linear():
    from b12x.attention.varlen._eager_bf16 import _layout
    from b12x.attention.varlen._tuning import TUNING

    eager = _query()
    flash = replace(eager, math_mode="flash")
    assert TUNING.encode_query(eager) != TUNING.encode_query(flash)
    TUNING.validate_query(
        replace(flash, causal=True, dtype="float16", kv_heads=8), None
    )
    layouts = [
        _layout({"q_shape": (rows, 16, 72), "v_shape": (rows, 16, 72)})
        for rows in (256, 16384)
    ]
    assert layouts[1][1] == layouts[0][1] * 64
    assert layouts[1][1] == 16384 * 16 * (72 * 2 + 3 * 4)


@pytest.mark.parametrize(
    "rows", [(0, 9, 9, 2), (8, 0, 0, 2), (8, 9, 8, 2), (8, 9, 9, 1)]
)
def test_eager_bf16_contract_rejects_unsafe_storage_extents(rows):
    from b12x.attention.varlen._eager_bf16 import validate_extents

    q, k, v, cu = rows
    with pytest.raises(ValueError):
        validate_extents((q, 16, 72), (k, 16, 72), (v, 16, 72), (cu,), (cu,))


def _reference(q, k, v, scale):
    q, k, v = (value.transpose(0, 1) for value in (q, k, v))
    scores = (q @ k.transpose(-2, -1)) * scale
    probabilities = scores.softmax(-1, dtype=torch.float32).bfloat16()
    return (probabilities @ v).transpose(0, 1), scores.float().logsumexp(-1)


@pytest.mark.parametrize("width", [72, 128])
@torch.no_grad()
def test_eager_bf16_ragged_capture_precision_and_reuse(width, monkeypatch):
    from b12x.attention import varlen
    from b12x.attention.varlen import _eager_bf16
    from b12x.preparation import PreparationSession, PreparedCall, require_prepared

    device = require_b12x()
    q, k, v = [
        torch.empty((257, 16, width), dtype=torch.bfloat16, device=device)
        for _ in range(3)
    ]
    cq, ck = [torch.zeros(4, dtype=torch.int32, device=device) for _ in range(2)]
    plan = varlen.plan(
        q, k, v, cq, ck, max_seqlen_q=257, max_seqlen_k=257, math_mode="eager_bf16"
    )
    retained = {}

    def prepare(state):
        (spec,) = state.scratch_plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
        binding = state.bind(
            plan=plan, scratch=scratch, q=q, k=k, v=v, cu_seqlens_q=cq, cu_seqlens_k=ck
        )
        retained.update(scratch=scratch, binding=binding)
        return PreparedCall(run=lambda: state.run(binding), owners=(scratch, binding))

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare((plan.request(name="eager-bf16-ragged", prepare_call=prepare),))
        state = require_prepared(plan, "attention.varlen", q.device)
        binding = retained["binding"]
        second = varlen.bind(
            plan,
            scratch=retained["scratch"],
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cq,
            cu_seqlens_k=ck,
        )
        assert second.binding.maximum.data_ptr() == binding.binding.maximum.data_ptr()
        assert second.binding.inverse.data_ptr() == binding.binding.inverse.data_ptr()
        assert len(state.plan.compiled) == 2

        def forbidden(*args, **kwargs):
            raise AssertionError("capture/replay must retain the two compiled programs")

        monkeypatch.setattr(_eager_bf16, "compile_programs", forbidden)
        session.freeze()
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            output, lse = varlen.run(binding)
        for qlengths, klengths in [
            ([65, 127, 65], [65, 127, 65]),
            ([0, 129, 68], [0, 129, 68]),
            ([23, 0, 63], [0, 41, 80]),
        ]:
            q.normal_()
            k.normal_()
            v.normal_()
            q[sum(qlengths) :].fill_(float("nan"))
            k[sum(klengths) :].fill_(float("nan"))
            v[sum(klengths) :].fill_(float("nan"))
            cq.copy_(
                torch.tensor(
                    [0, *torch.tensor(qlengths).cumsum(0).tolist()], dtype=torch.int32
                )
            )
            ck.copy_(
                torch.tensor(
                    [0, *torch.tensor(klengths).cumsum(0).tolist()], dtype=torch.int32
                )
            )
            for _ in range(2):
                output.fill_(float("nan"))
                lse.fill_(float("nan"))
                torch.cuda.synchronize()
                allocated = torch.cuda.memory_stats(device)["allocation.all.allocated"]
                graph.replay()
                torch.cuda.synchronize()
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocated
                )
                qstart = kstart = 0
                for qlen, klen in zip(qlengths, klengths):
                    if qlen:
                        expected, expected_lse = _reference(
                            q[qstart : qstart + qlen],
                            k[kstart : kstart + klen],
                            v[kstart : kstart + klen],
                            width**-0.5,
                        )
                        actual = output[qstart : qstart + qlen]
                        relative = (
                            actual.float() - expected.float()
                        ).norm() / expected.float().norm().clamp_min(1e-12)
                        assert relative < 5e-4, float(relative)
                        torch.testing.assert_close(
                            actual, expected, atol=0.002, rtol=0.01
                        )
                        torch.testing.assert_close(
                            lse[:, qstart : qstart + qlen],
                            expected_lse,
                            atol=2e-5,
                            rtol=2e-6,
                        )
                    qstart += qlen
                    kstart += klen
                assert torch.isnan(output[qstart:]).all()
                assert torch.isnan(lse[:, qstart:]).all()
        graph.reset()

        small = varlen.bind(
            plan,
            scratch=retained["scratch"],
            q=q[:73],
            k=k[:91],
            v=v[:91],
            cu_seqlens_q=cq[:3],
            cu_seqlens_k=ck[:3],
            max_seqlen_q=73,
            max_seqlen_k=91,
        )
        assert small.binding.lse.stride(0) == 257
        assert small.binding.maximum.data_ptr() == binding.binding.maximum.data_ptr()
        with session.capture(), torch.cuda.graph(graph):
            small_out, small_lse = varlen.run(small)
        for qlengths, klengths in [([37, 36], [44, 47]), ([0, 65], [13, 50])]:
            q.normal_()
            k.normal_()
            v.normal_()
            cq[:3].copy_(
                torch.tensor([0, qlengths[0], sum(qlengths)], dtype=torch.int32)
            )
            ck[:3].copy_(
                torch.tensor([0, klengths[0], sum(klengths)], dtype=torch.int32)
            )
            small_out.fill_(float("nan"))
            small_lse.fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            qs = ks = 0
            for qlen, klen in zip(qlengths, klengths):
                if qlen:
                    truth, truth_lse = _reference(
                        q[qs : qs + qlen],
                        k[ks : ks + klen],
                        v[ks : ks + klen],
                        width**-0.5,
                    )
                    torch.testing.assert_close(
                        small_out[qs : qs + qlen], truth, atol=0.002, rtol=0.01
                    )
                    torch.testing.assert_close(
                        small_lse[:, qs : qs + qlen], truth_lse, atol=2e-5, rtol=2e-6
                    )
                qs += qlen
                ks += klen
            assert torch.isnan(small_out[qs:]).all()
        graph.reset()
        for bad in (
            {"q": q[:0]},
            {"k": k[:0], "v": v[:0]},
            {"v": v[:256]},
            {"cu_seqlens_q": cq[:1], "cu_seqlens_k": ck[:1]},
        ):
            arguments = dict(
                scratch=retained["scratch"],
                q=q,
                k=k,
                v=v,
                cu_seqlens_q=cq,
                cu_seqlens_k=ck,
            )
            with pytest.raises(ValueError):
                varlen.bind(plan, **(arguments | bad))


@torch.no_grad()
def test_eager_bf16_distinct_capacity_plans_share_oversized_arena():
    from b12x.attention import varlen
    from b12x.preparation import PreparationSession, PreparedCall

    device = require_b12x()
    q, k, v = [
        torch.randn((257, 16, 72), dtype=torch.bfloat16, device=device)
        for _ in range(3)
    ]
    cu = torch.zeros(3, dtype=torch.int32, device=device)
    retained = {}
    bindings = []
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        for rows in (257, 73):
            plan = varlen.plan(
                q[:rows],
                k[:rows],
                v[:rows],
                cu,
                max_seqlen_q=rows,
                max_seqlen_k=rows,
                math_mode="eager_bf16",
            )

            def prepare(state, rows=rows, plan=plan):
                (spec,) = state.scratch_plan.scratch_specs()
                if "arena" not in retained:
                    retained["arena"] = torch.empty(
                        spec.shape, dtype=spec.dtype, device=spec.device
                    )
                binding = state.bind(
                    plan=plan,
                    scratch=retained["arena"],
                    q=q[:rows],
                    k=k[:rows],
                    v=v[:rows],
                    cu_seqlens_q=cu,
                )
                bindings.append(binding)
                return PreparedCall(
                    run=lambda: state.run(binding), owners=(retained["arena"], binding)
                )

            session.prepare(
                (
                    plan.request(
                        name=f"eager-bf16-capacity-{rows}", prepare_call=prepare
                    ),
                )
            )
        assert (
            bindings[0].binding.output.data_ptr()
            == bindings[1].binding.output.data_ptr()
        )
        assert bindings[0].binding.lse.stride(0) == 257
        assert bindings[1].binding.lse.stride(0) == 73
        session.freeze()
        graphs = []
        for binding in bindings:
            graph = torch.cuda.CUDAGraph()
            with session.capture(), torch.cuda.graph(graph):
                varlen.run(binding)
            graphs.append(graph)
        for selected, lengths in (
            (0, (129, 100)),
            (1, (33, 40)),
            (0, (0, 257)),
            (1, (0, 61)),
        ):
            q.normal_()
            k.normal_()
            v.normal_()
            cu.copy_(torch.tensor([0, lengths[0], sum(lengths)], dtype=torch.int32))
            graphs[selected].replay()
            torch.cuda.synchronize()
            start = 0
            for count in lengths:
                if count:
                    expected, expected_lse = _reference(
                        q[start : start + count],
                        k[start : start + count],
                        v[start : start + count],
                        72**-0.5,
                    )
                    active = bindings[selected].binding
                    torch.testing.assert_close(
                        active.output[start : start + count],
                        expected,
                        atol=0.002,
                        rtol=0.01,
                    )
                    torch.testing.assert_close(
                        active.lse[:, start : start + count],
                        expected_lse,
                        atol=2e-5,
                        rtol=2e-6,
                    )
                start += count
        for graph in graphs:
            graph.reset()
