# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from flashinfer import kda
from flashinfer.utils import get_compute_capability


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
@pytest.mark.parametrize("capture", [False, True])
@pytest.mark.parametrize(
    ("num_sequences", "extent"),
    [
        pytest.param(3, 4, id="decomp-short"),
        pytest.param(3, 80, id="decomp-drain-only"),
        pytest.param(3, 96, id="decomp-first-handoff"),
        pytest.param(3, 144, id="decomp-ring-wrap"),
        pytest.param(64, 64, id="engine-guard"),
    ],
)
def test_kda_prefill_ignores_nonfinite_padding(poison, capture, num_sequences, extent):
    """Keep poisoned V padding out of output and state across producer schedules."""
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in ((10, 0), (10, 3)):
        pytest.skip("CuTe KDA prefill requires SM100/SM103")
    torch.manual_seed(42)
    device = "cuda"
    heads, dim, slots = 12, 128, 8
    if num_sequences == 3:
        # Ensure these cases exercise the chain kernel, including on 80-SM B300.
        assert (
            num_sequences * heads * 2
            <= torch.cuda.get_device_properties(device).multi_processor_count
        )
    shape = (1, extent + 1, heads, dim)
    q, k, v, g = [
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
    ]
    beta = torch.randn(shape[:-1], device=device, dtype=torch.bfloat16)
    a_log = torch.zeros(heads, device=device)
    dt_bias = torch.zeros(heads * dim, device=device)
    seed_state = torch.randn(slots, heads, dim, dim, device=device) * 0.1
    state = seed_state.clone()
    indices = torch.zeros(num_sequences, device=device, dtype=torch.int32)
    indices[0] = 3
    cu = torch.zeros(num_sequences + 1, device=device, dtype=torch.int64)
    cu[1:] = extent - 1
    wrapper = kda.RecurrentKDAPrefillWrapper(device)
    wrapper.plan(cu)
    output = torch.empty_like(q)
    options = dict(
        A_log=a_log,
        dt_bias=dt_bias,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
    )

    def run():
        return wrapper.run(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            initial_state=state,
            ssm_state_indices=indices,
            output=output,
            **options,
        )[0]

    graph = None
    if capture:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            run()
    else:
        run()

    # Replay the same captured extent with changing logical token counts.
    for n in (extent, extent - 1, extent - 2):
        cu[1:] = n
        wrapper.plan(cu)
        state.copy_(seed_state)
        ref_state = seed_state.clone()
        # V alone isolates the second kernel's raw-V reload from Q/K/gate masking.
        v[:, n:] = poison
        if graph is None:
            run()
        else:
            graph.replay()
        ref, _ = kda.recurrent_kda(
            q=q[:, :n].contiguous(),
            k=k[:, :n].contiguous(),
            v=v[:, :n].contiguous(),
            g=g[:, :n].contiguous(),
            beta=beta[:, :n].contiguous(),
            initial_state=ref_state,
            cu_seqlens=torch.tensor([0, n], device=device, dtype=torch.int64),
            ssm_state_indices=indices[:1],
            backend="cute-dsl",
            **options,
        )
        assert torch.isfinite(output[:, :n]).all()
        assert torch.isfinite(state).all()
        torch.testing.assert_close(output[:, :n], ref, atol=2e-3, rtol=2e-2)
        torch.testing.assert_close(state, ref_state, atol=2e-3, rtol=2e-2)
