"""Prepared prefill mutations survive full-graph compilation and CUDA replay."""

import pytest
import torch

from b12x.preparation import PreparationSession, PreparedCall
from b12x.sequence import gdn_prefill
from b12x.testing.delta_prefill_cases import PrefillCase, make_inputs

from ..conftest import require_b12x


@pytest.mark.parametrize("precision", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("tokens", [17, 65])
@torch.no_grad()
def test_compiled_prefill_replays_changed_inputs_and_pooled_state(precision, tokens):
    device = require_b12x()
    values = make_inputs(
        PrefillCase("gdn", 4, 12, (tokens, 1)),
        device=device,
        max_tokens=tokens + 8,
        max_seqs=2,
    )
    values["a"] = values.pop("raw_g")
    values["b"] = values.pop("raw_beta")
    values["checkpoint_offsets"][0] = 16
    original = values["recurrent_state"].clone()
    caps = gdn_prefill.Caps(
        device=device,
        max_tokens=tokens + 8,
        max_seqs=2,
        max_state_slots=7,
        key_heads=4,
        value_heads=12,
        intermediate_dtype=precision,
        checkpoint_export=True,
    )
    plan = gdn_prefill.plan(
        caps, invocation=gdn_prefill.invocation_from_tensors(**values)
    )

    def prepare(state):
        (spec,) = state.layout.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = state.bind(scratch=scratch, **values)
        return PreparedCall(run=lambda: state.run(binding), owners=(binding,))

    with PreparationSession(
        device=device, autotune=False, compile_workers=0
    ) as session:
        session.prepare((plan.request(name="gdn", prepare_call=prepare),))
        (spec,) = plan.scratch_specs()
        scratch = torch.empty(spec.shape, dtype=spec.dtype, device=device)
        binding = gdn_prefill.bind(plan, scratch=scratch, **values)

        def forward():
            gdn_prefill.run(binding)

        compiled = torch.compile(forward, fullgraph=True, dynamic=False)
        compiled()
        graph = torch.cuda.CUDAGraph()
        try:
            with session.capture(), torch.cuda.graph(graph):
                compiled()
            for empty in (False, True):
                values["q"].mul_(0.75)
                values["v"].add_(0.125)
                if empty:
                    values["num_seqs"].zero_()
                    values["num_tokens"].zero_()
                values["recurrent_state"].copy_(original)
                values["output"].fill_(19)
                plan.prepared.state.run(binding)
                expected_output = values["output"].clone()
                expected_state = values["recurrent_state"].clone()
                values["recurrent_state"].copy_(original)
                values["output"].fill_(19)
                torch.cuda.synchronize(device)
                allocations = torch.cuda.memory_stats(device)[
                    "allocation.all.allocated"
                ]
                graph.replay()
                torch.cuda.synchronize(device)
                assert (
                    torch.cuda.memory_stats(device)["allocation.all.allocated"]
                    == allocations
                )
                torch.testing.assert_close(
                    values["output"], expected_output, rtol=0, atol=0
                )
                torch.testing.assert_close(
                    values["recurrent_state"], expected_state, rtol=0, atol=0
                )
        finally:
            graph.reset()
