from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x.gemm import mla_query_projection
from b12x.gemm.mla_query_projection._tuning import ProjectionQuery
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x
from tests.experimental.b12x.gemm.test_bmm import _make_pack, _rhs_views


def _inputs(*, heads, m, weight_format, seed=31):
    torch.manual_seed(seed)
    q_nope = torch.randn(heads, m, 192, device="cuda", dtype=torch.bfloat16)
    q_full = torch.randn(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    q_pe = q_full[..., 512:]
    q_scale = torch.tensor([.037], device="cuda", dtype=torch.float32)
    if weight_format == "mxfp8":
        values, scales = _make_pack(seed=seed, batch=heads)
        weight = _rhs_views(values, scales, batch=heads)["n"]
    else:
        weight = torch.randn(heads, 192, 512, device="cuda", dtype=torch.bfloat16) * .05
    return q_nope, weight, q_pe, q_scale


@contextmanager
def _prepared(weight, *, heads, m, output_dtype):
    weight_format = "bf16" if isinstance(weight, torch.Tensor) else "mxfp8"
    query = ProjectionQuery(heads=heads, max_rows=m, weight_format=weight_format,
                            output_dtype=str(output_dtype).removeprefix("torch."), b_major="n", sf_axis="n")
    q_nope = torch.zeros(heads, m, 192, device="cuda", dtype=torch.bfloat16)
    q_pe = torch.zeros(m, heads, 64, device="cuda", dtype=torch.bfloat16)
    q_scale = torch.ones(1, device="cuda", dtype=torch.float32) if output_dtype == torch.float8_e4m3fn else None
    out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
    plan = mla_query_projection.plan(query)
    request = plan.request(name="mla",
        prepare_call=lambda state: PreparedCall(run=lambda: state.run(q_nope, weight, q_pe, out, q_scale=q_scale)))
    session = PreparationSession(autotune=False)
    result = session.prepare((request,))
    try: yield plan
    finally: result.close(); session.close()


def _reference(q_nope, weight, q_pe):
    if isinstance(weight, torch.Tensor): projected = torch.bmm(q_nope, weight)
    else:
        values, scales = weight
        physical = values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, dim=-1)
        projected = torch.bmm(q_nope, physical)
    return torch.cat((projected.transpose(0, 1), q_pe), dim=-1)


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("mxfp8", 16), ("bf16", 11)])
@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_prepared_projection_preserves_weight_forms_and_output_modes(weight_format, heads, output_dtype):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=output_dtype)
    with _prepared(weight, heads=heads, m=m, output_dtype=output_dtype) as plan:
        assert mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan,
                                        q_scale=q_scale if output_dtype == torch.float8_e4m3fn else None) is out
    expected = _reference(q_nope, weight, q_pe)
    if output_dtype == torch.bfloat16: torch.testing.assert_close(out, expected, rtol=.03, atol=.03)
    else: assert torch.equal(out.view(torch.uint8), (expected.float() / q_scale).clamp(-448, 448).to(output_dtype).view(torch.uint8))


@pytest.mark.parametrize("weight_format,heads", [("mxfp8", 8), ("bf16", 11)])
def test_prepared_projection_graph_replays_changed_inputs(weight_format, heads):
    require_b12x(); m = 4
    q_nope, weight, q_pe, q_scale = _inputs(heads=heads, m=m, weight_format=weight_format)
    out = torch.empty(m, heads, 576, device="cuda", dtype=torch.bfloat16)
    with _prepared(weight, heads=heads, m=m, output_dtype=torch.bfloat16) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): mla_query_projection.run(q_nope, weight, q_pe, out, plan=plan)
        fresh_nope, fresh_pe = torch.randn_like(q_nope), torch.randn_like(q_pe)
        q_nope.copy_(fresh_nope); q_pe.copy_(fresh_pe); graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, _reference(fresh_nope, weight, fresh_pe), rtol=.03, atol=.03)


def test_projection_rejects_unprepared_execution():
    require_b12x()
    q_nope, weight, q_pe, _ = _inputs(heads=8, m=1, weight_format="mxfp8")
    out = torch.empty(1, 8, 576, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        mla_query_projection.run(q_nope, weight, q_pe, out)
