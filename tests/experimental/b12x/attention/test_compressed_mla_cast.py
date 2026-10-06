"""Prepared cache conversion preserves numerical and graph-storage contracts."""
import pytest
import torch

from b12x.attention.compressed_sparse_mla import cast
from b12x.preparation import PreparationSession, PreparedCall
from b12x.testing.reference.helpers import require_b12x


@pytest.mark.parametrize("input_dtype,output_dtype", [
    (torch.bfloat16, torch.float32), (torch.float32, torch.bfloat16),
    (torch.bfloat16, torch.bfloat16), (torch.float32, torch.float32),
])
def test_prepared_cast_live_capacity_and_replay(input_dtype, output_dtype):
    device = require_b12x()
    source = torch.randn(513, device=device, dtype=input_dtype)
    output = torch.empty(513, device=device, dtype=output_dtype)
    plan = cast.plan(cast.Query(max_elements=513,
                               input_dtype=str(input_dtype).removeprefix("torch."),
                               output_dtype=str(output_dtype).removeprefix("torch.")), device=source.device)
    def prepare(state):
        trial = torch.empty_like(output)
        return PreparedCall(run=lambda: state.run(source, out=trial), output=trial)
    with PreparationSession(device=device, autotune=False, compile_workers=2) as session:
        session.prepare((plan.request(name="cast", prepare_call=prepare),))
        session.freeze()
        for count in (0, 1, 255, 256, 257, 513):
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                cast.cast(source[:count], out=output[:count], plan=plan)
            source.neg_()
            output.fill_(float("nan"))
            before = torch.cuda.memory_stats(device)["allocation.all.allocated"]
            graph.replay()
            torch.cuda.synchronize(device)
            assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == before
            torch.testing.assert_close(output[:count], source[:count].to(output_dtype), rtol=0, atol=0)
            assert torch.isnan(output[count:]).all()
            graph.reset()
