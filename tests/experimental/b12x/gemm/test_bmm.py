from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch

from b12x import gemm
from b12x.gemm._bmm._tuning import BmmQuery
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x

BATCH, PACK_ROWS, K_N_MAJOR, K_K_MAJOR = 16, 448, 192, 512
N_N_MAJOR, N_K_MAJOR_OUT = 512, 256
BASE_SPEC = dict(a_dtype="bfloat16", b_dtype="float8_e4m3fn", sf_dtype="float8_e8m0fnu", c_dtype="bfloat16", sf_vec_size=32)


def _spec(major): return {**BASE_SPEC, "b_major": major, "sf_axis": major}


def _make_pack(seed=7, batch=BATCH):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = (torch.randn(batch * PACK_ROWS, K_K_MAJOR, device="cuda", generator=generator) * .1).to(torch.float8_e4m3fn)
    scales = torch.randint(118, 132, (batch * PACK_ROWS, K_K_MAJOR // 32), device="cuda", generator=generator, dtype=torch.uint8)
    return values, scales


def _rhs_views(values, scales, batch=BATCH):
    values, scales = values.view(batch, PACK_ROWS, K_K_MAJOR), scales.view(batch, PACK_ROWS, K_K_MAJOR // 32)
    return {"n": (values[:, :K_N_MAJOR, :], scales[:, :K_N_MAJOR, :]), "k": (values[:, K_N_MAJOR:K_N_MAJOR + N_K_MAJOR_OUT, :], scales[:, K_N_MAJOR:K_N_MAJOR + N_K_MAJOR_OUT, :])}


def _logical_b(rhs, major):
    values, scales = rhs
    physical = values.to(torch.bfloat16) * scales.view(torch.float8_e8m0fnu).to(torch.bfloat16).repeat_interleave(32, dim=-1)
    return physical if major == "n" else physical.transpose(1, 2)


@contextmanager
def _prepared(rhs, major, m):
    values, _ = rhs
    batch = values.shape[0]
    k = K_N_MAJOR if major == "n" else K_K_MAJOR
    n = N_N_MAJOR if major == "n" else N_K_MAJOR_OUT
    query = BmmQuery(batch=batch, max_rows=m, in_features=k, out_features=n, b_major=major, sf_axis=major)
    lhs = torch.zeros(batch, m, k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(batch, m, n, device="cuda", dtype=torch.bfloat16)
    plan = gemm.plan_bmm(query)
    request = plan.request(name="bmm", prepare_call=lambda state: PreparedCall(run=lambda: state.run(lhs, rhs, out, b_major=major, sf_axis=major)))
    session = PreparationSession(autotune=False)
    result = session.prepare((request,))
    try:
        yield plan
    finally:
        result.close(); session.close()


@pytest.mark.parametrize("major", ["n", "k"])
@pytest.mark.parametrize("m", [1, 4, 16, 32])
def test_bmm_preserves_qualified_layouts_and_numerics(major, m):
    require_b12x()
    rhs = _rhs_views(*_make_pack())[major]
    logical = _logical_b(rhs, major)
    lhs = torch.randn(BATCH, m, logical.shape[1], device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, m, logical.shape[2], device="cuda", dtype=torch.bfloat16)
    with _prepared(rhs, major, m) as plan:
        assert gemm.bmm(lhs, rhs, out, plan=plan, **_spec(major)) is out
    reference64 = torch.bmm(lhs.double(), logical.double())
    assert (out.double() - reference64).abs().max() <= (torch.bmm(lhs, logical).double() - reference64).abs().max() * 1.05 + 1e-12


@pytest.mark.parametrize("major", ["n", "k"])
def test_bmm_cuda_graph_replays_changed_input(major):
    require_b12x()
    m = 4; rhs = _rhs_views(*_make_pack())[major]; logical = _logical_b(rhs, major)
    lhs = torch.zeros(BATCH, m, logical.shape[1], device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, m, logical.shape[2], device="cuda", dtype=torch.bfloat16)
    with _prepared(rhs, major, m) as plan:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): gemm.bmm(lhs, rhs, out, plan=plan, **_spec(major))
        fresh = torch.randn_like(lhs); lhs.copy_(fresh); graph.replay(); torch.cuda.synchronize()
    torch.testing.assert_close(out, torch.bmm(fresh, logical), rtol=.03, atol=.03)


def test_bmm_rejects_unprepared_and_layout_mismatch():
    require_b12x()
    rhs = _rhs_views(*_make_pack())["n"]
    lhs = torch.zeros(BATCH, 1, K_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(BATCH, 1, N_N_MAJOR, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(TypeError):
        gemm.bmm(lhs, rhs, out, **_spec("n"))
    with _prepared(rhs, "n", 1) as plan:
        with pytest.raises(ValueError, match="layout differs"): gemm.bmm(lhs, rhs, out, plan=plan, **_spec("k"))


@pytest.mark.parametrize("major", ["n", "k"])
def test_bmm_torch_compile_preserves_explicit_output(major):
    require_b12x()
    rhs = _rhs_views(*_make_pack())[major]; logical = _logical_b(rhs, major); m = 2
    with _prepared(rhs, major, m) as plan:
        def run(a, out): return gemm.bmm(a, rhs, out, plan=plan, **_spec(major))
        compiled = torch.compile(run, backend="aot_eager", fullgraph=True)
        lhs = torch.randn(BATCH, m, logical.shape[1], device="cuda", dtype=torch.bfloat16)
        out = torch.empty(BATCH, m, logical.shape[2], device="cuda", dtype=torch.bfloat16)
        assert compiled(lhs, out) is out
    torch.testing.assert_close(out, torch.bmm(lhs, logical), rtol=.03, atol=.03)
