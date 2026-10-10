"""GLM head partitions and chunked prefill use native rowwise-MXFP8 BMM."""

import pytest
import torch

from b12x import gemm
from b12x.gemm._bmm._tuning import BmmQuery
from b12x.preparation import PreparationSession, PreparedCall

from ..conftest import require_b12x


@pytest.mark.parametrize(
    "heads,rows", [(16, 1), (32, 2), (16, 33), (32, 65), (16, 4096), (32, 4096)]
)
@pytest.mark.parametrize("major", ["n", "k"])
@torch.no_grad()
def test_glm_absorbed_projection_compiled_graph(heads, rows, major):
    device = require_b12x()
    torch.compiler.reset()
    generator = torch.Generator(device=device).manual_seed(531512)
    values = (
        torch.randn((heads, 512, 512), device=device, generator=generator) * 0.1
    ).to(torch.float8_e4m3fn)
    scales = torch.randint(
        118,
        132,
        (heads, 512, 16),
        dtype=torch.uint8,
        device=device,
        generator=generator,
    )
    extent = slice(0, 256) if major == "n" else slice(256, 512)
    rhs = (values[:, extent], scales[:, extent])
    k, n = (256, 512) if major == "n" else (512, 256)
    source = torch.zeros((rows + 2, heads, k), dtype=torch.bfloat16, device=device)
    destination = torch.full(
        (rows + 2, heads, n), 42, dtype=torch.bfloat16, device=device
    )
    lhs, output = source[1:-1].transpose(0, 1), destination[1:-1].transpose(0, 1)
    physical = rhs[0].to(torch.bfloat16) * rhs[1].view(torch.float8_e8m0fnu).to(
        torch.bfloat16
    ).repeat_interleave(32, dim=-1)
    logical = physical if major == "n" else physical.transpose(1, 2)
    query = BmmQuery(
        batch=heads,
        max_rows=rows,
        in_features=k,
        out_features=n,
        b_major=major,
        sf_axis=major,
    )
    plan = gemm.plan_bmm(query)
    spec = dict(
        a_dtype="bfloat16",
        b_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        b_major=major,
        sf_axis=major,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare(
            (
                plan.request(
                    name="absorbed-projection",
                    prepare_call=lambda state: PreparedCall(
                        run=lambda: state.run(
                            lhs, rhs, output, b_major=major, sf_axis=major
                        )
                    ),
                ),
            )
        )
        compiled = torch.compile(
            lambda: gemm.bmm(lhs, rhs, output, plan=plan, **spec),
            fullgraph=True,
            dynamic=False,
        )
        compiled()
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with session.capture(), torch.cuda.graph(graph):
            compiled()
        session.freeze()
        for _ in range(2):
            source.normal_(generator=generator)
            destination.fill_(42)
            before = torch.cuda.memory_stats(device)
            graph.replay()
            torch.cuda.synchronize(device)
            after = torch.cuda.memory_stats(device)
            for counter in (
                "allocation.all.allocated",
                "allocation.all.freed",
                "segment.all.allocated",
                "segment.all.freed",
            ):
                assert before[counter] == after[counter]
            reference = torch.bmm(lhs.double(), logical.double())
            torch_error = (torch.bmm(lhs, logical).double() - reference).abs().max()
            assert (
                output.double() - reference
            ).abs().max() <= torch_error * 1.05 + 1e-12
            assert (destination[0] == 42).all() and (destination[-1] == 42).all()
        graph.reset()
