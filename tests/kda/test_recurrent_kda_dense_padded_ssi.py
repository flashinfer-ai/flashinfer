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

# Cake T1 unbounded-softplus is exported only for these compute capabilities.
_CAKE_DECODE_CCS = {(10, 0), (10, 3)}


def _assert_pool_slots(state_pool, before, active_slots, n_slots):
    inactive_slots = set(range(n_slots)) - set(active_slots)
    for slot in inactive_slots:
        assert torch.equal(state_pool[slot], before[slot]), (
            f"inactive slot {slot} was modified"
        )
    for slot in active_slots:
        changed = (state_pool[slot].float() - before[slot].float()).abs().max().item()
        assert changed > 0.0, f"active slot {slot} not updated"


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
    """Dense T=1 with ssi==-1 must not wrap to the last pool slot.

    D=64 with a precomputed gate makes backend=\"auto\" fall through to CuTe
    (Cake-ineligible). That is the #5042 failure mode under auto.
    """
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

    _out, final_state = recurrent_kda(
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
    assert final_state is state_pool
    assert tuple(final_state.shape) == (n_slots, H, D, D)
    _assert_pool_slots(state_pool, before, active_slots, n_slots)


@pytest.mark.parametrize("backend", ["cake", "auto"])
def test_dense_padded_ssi_cake_unbounded_softplus(backend: str):
    """Cake equal-head D128 T1 unbounded-softplus must skip ssi==-1.

    Matches the Cake / auto-Cake-eligible contract. Skips ``backend=\"cake\"``
    on devices outside the exported Cake decode arch map.
    """
    cc = torch.cuda.get_device_capability()
    if backend == "cake" and cc not in _CAKE_DECODE_CCS:
        pytest.skip(f"Cake decode not mapped for compute capability {cc}")

    torch.manual_seed(2)
    device = torch.device("cuda")
    dtype = torch.bfloat16
    B, H, D = 2, 32, 128
    n_slots = 5
    SENTINEL = 7.0
    state_pool = torch.zeros(n_slots, H, D, D, dtype=dtype, device=device)
    state_pool[-1] = SENTINEL
    ssi = torch.tensor([-1, 3], device=device, dtype=torch.int32)
    active_slots = [3]
    state_pool[3] = torch.randn(H, D, D, dtype=dtype, device=device) * 0.01
    before = state_pool.clone()

    q = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    k = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    v = torch.rand(B, 1, H, D, dtype=dtype, device=device)
    g = torch.randn(B, 1, H, D, dtype=dtype, device=device)
    A_log = torch.log(torch.ones(H, dtype=torch.float32, device=device).uniform_(1, 16))
    dt_bias = torch.randn(H * D, dtype=torch.float32, device=device)
    beta = torch.rand(B, 1, H, dtype=dtype, device=device).sigmoid()
    scale = 1.0 / D**0.5

    try:
        _out, final_state = recurrent_kda(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            A_log=A_log,
            dt_bias=dt_bias,
            initial_state=state_pool,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            ssm_state_indices=ssi,
            backend=backend,
        )
    except ValueError as exc:
        if backend == "cake" and "unsupported" in str(exc):
            pytest.skip(str(exc))
        raise

    assert final_state is state_pool
    assert tuple(final_state.shape) == (n_slots, H, D, D)
    _assert_pool_slots(state_pool, before, active_slots, n_slots)


@pytest.mark.parametrize("backend", ["cute-dsl", "auto"])
def test_dense_padded_ssi_cuda_graph_safe(backend: str):
    """Dense padded ssi path must remain CUDA-graph capturable (no bool-mask).

    D=64 / precomputed gate: auto falls through to CuTe (Cake-ineligible).
    """
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
    active_slots = [1, 3]

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
    before = state_pool.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    inactive_slots = set(range(n_slots)) - set(active_slots)
    for slot in inactive_slots:
        assert torch.equal(state_pool[slot], before[slot]), (
            f"graph replay modified inactive slot {slot}"
        )
