# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

import pytest
import torch

from flashinfer.kda import recurrent_kda
from flashinfer.utils import get_compute_capability

HEADS, DIM = 12, 128


def _run(q, k, v, g, beta, state, cu):
    return recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=state,
        cu_seqlens=cu,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=True,
        lower_bound=-5.0,
        beta_is_logit=True,
        A_log=torch.zeros(HEADS, device=q.device),
        dt_bias=torch.zeros(HEADS * DIM, device=q.device),
        backend="small-bh",
    )


@pytest.mark.parametrize("poison", [float("nan"), float("inf")])
@pytest.mark.parametrize(
    "lengths",
    [
        pytest.param((37, 40), id="two-seqs"),
        pytest.param((3, 5, 7, 9), id="short-seqs"),
        pytest.param((64, 64), id="aligned"),
        pytest.param((100, 17, 130), id="multi-tile"),
    ],
)
def test_small_bh_ignores_nonfinite_v_of_next_sequence(lengths, poison):
    """V rows past a sequence's end belong to the next sequence and must not reach it."""
    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) not in ((10, 0), (10, 3)):
        pytest.skip("small-bh KDA prefill requires SM100/SM103")
    torch.manual_seed(0)
    device = "cuda"
    total = sum(lengths)
    shape = (1, total, HEADS, DIM)
    q, k, v, g = [
        torch.randn(shape, device=device, dtype=torch.bfloat16) for _ in range(4)
    ]
    beta = torch.randn(shape[:-1], device=device, dtype=torch.bfloat16)
    seed_state = (
        torch.randn(len(lengths), HEADS, DIM, DIM, device=device) * 0.1
    ).bfloat16()
    cu = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()],
        device=device,
        dtype=torch.int64,
    )

    v_poisoned = v.clone()
    v_poisoned[:, cu[-2] :] = poison
    out, state = _run(q, k, v_poisoned, g, beta, seed_state.clone(), cu)

    clean_end = int(cu[-2])
    ref_out, ref_state = _run(
        q[:, :clean_end].contiguous(),
        k[:, :clean_end].contiguous(),
        v[:, :clean_end].contiguous(),
        g[:, :clean_end].contiguous(),
        beta[:, :clean_end].contiguous(),
        seed_state[:-1].clone(),
        cu[:-1].contiguous(),
    )
    assert torch.isfinite(out[:, :clean_end]).all()
    assert torch.isfinite(state[:-1].float()).all()
    torch.testing.assert_close(out[:, :clean_end], ref_out)
    torch.testing.assert_close(state[:-1], ref_state)
