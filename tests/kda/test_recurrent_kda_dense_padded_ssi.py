# Copyright (c) 2023-2026 FlashInfer contributors
# Regression for flashinfer-ai/flashinfer#5042: dense T=1 ssi==-1 must not wrap.

import torch
import torch.nn.functional as F
import pytest

from flashinfer.kda_decode import _RECURRENT_KDA_AVAILABLE, recurrent_kda

pytestmark = pytest.mark.skipif(
    not _RECURRENT_KDA_AVAILABLE,
    reason="recurrent_kda kernel not available (missing cutlass DSL deps)",
)


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
@pytest.mark.parametrize(
    ("B", "H", "route"),
    [
        # sequence_heads = B * H; one-warp threshold is 128
        pytest.param(2, 32, "grouped", id="grouped-B2-H32"),
        pytest.param(4, 32, "one-warp", id="onewarp-B4-H32"),
    ],
)
def test_dense_padded_ssm_state_indices_no_wrap(
    backend: str, B: int, H: int, route: str
):
    """Dense T=1 with ssi==-1 must not wrap to the last pool slot."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    D = 64
    n_slots = 5
    SENTINEL = 7.0
    state_pool = torch.zeros(n_slots, H, D, D, dtype=dtype, device=device)
    state_pool[-1] = SENTINEL
    if B == 2:
        ssi = torch.tensor([-1, 3], device=device, dtype=torch.int32)
        active_slots = [3]
    else:
        ssi = torch.tensor([-1, 1, -1, 3], device=device, dtype=torch.int32)
        active_slots = [1, 3]

    for slot in active_slots:
        state_pool[slot] = torch.randn(H, D, D, dtype=dtype, device=device) * 0.01
    before = state_pool.clone()

    q = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    g = (F.logsigmoid(torch.randn(B, 1, H, D, device=device)) / 1.0).to(dtype)
    beta = torch.rand(B, 1, H, dtype=dtype, device=device).sigmoid()
    scale = 1.0 / D**0.5

    seq_heads = B * H
    assert (route == "one-warp") == (seq_heads >= 128), (route, seq_heads)

    recurrent_kda(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        scale=scale,
        initial_state=state_pool,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        use_gate_in_kernel=False,
        ssm_state_indices=ssi,
        backend=backend,
    )

    last_delta = (state_pool[-1].float() - before[-1].float()).abs().max().item()
    assert last_delta == 0.0, (
        f"pool[-1] written (wrap-around): delta={last_delta} backend={backend} {route}"
    )
    if 0 not in active_slots:
        slot0_delta = (state_pool[0].float() - before[0].float()).abs().max().item()
        assert slot0_delta == 0.0, f"slot0 corrupted: {slot0_delta}"

    for slot in active_slots:
        changed = (state_pool[slot].float() - before[slot].float()).abs().max().item()
        assert changed > 0.0, f"active slot {slot} not updated"


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
def test_dense_padded_ssi_cuda_graph_safe(backend: str):
    """Dense padded ssi path must remain CUDA-graph capturable (no bool-mask)."""
    torch.manual_seed(1)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, H, D = 4, 32, 64
    n_slots = 8
    SENTINEL = 7.0
    state_pool = torch.zeros(n_slots, H, D, D, dtype=dtype, device=device)
    state_pool[-1] = SENTINEL
    state_pool[1] = torch.randn(H, D, D, dtype=dtype, device=device) * 0.01
    state_pool[3] = torch.randn(H, D, D, dtype=dtype, device=device) * 0.01
    ssi = torch.tensor([-1, 1, -1, 3], device=device, dtype=torch.int32)

    q = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    gate = (F.logsigmoid(torch.randn(B, 1, H, D, device=device)) / 1.0).to(dtype)
    beta = torch.rand(B, 1, H, dtype=dtype, device=device).sigmoid()
    scale = 1.0 / D**0.5
    out = torch.empty(B, 1, H, D, dtype=dtype, device=device)

    def run():
        recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=gate,
            beta=beta,
            scale=scale,
            initial_state=state_pool,
            output=out,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=False,
            ssm_state_indices=ssi,
            backend=backend,
        )

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    before_last = state_pool[-1].clone()
    graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(state_pool[-1], before_last), "graph replay wrote pool[-1]"
