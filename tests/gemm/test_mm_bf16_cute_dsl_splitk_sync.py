"""Replay stress test for the synchronization of the cluster split-K BF16 GEMM
(flashinfer/gemm/kernels/dense_bf16_gemm_sm100_splitk.py, a runner of
``mm_bf16(backend="cute-dsl")``).

The kernel hands work between warps and CTAs through mbarriers: TMEM allocation
and release, MMA completion, and the peers' st.async FP32 partials to the owner
CTA. Each case runs one explicit tactic through many CUDA-graph replays and eager
back-to-back PDL launches. Consecutive launches use different activations, so a
stale partial or accumulator is wrong rather than repeated; outputs are poisoned
with NaN before every run; every launch is compared bitwise with the first result,
which is itself checked against an FP64 reference.
"""

import pytest
import torch

from flashinfer.utils import get_compute_capability

# K = 48 tiles of 128: 24 / 16 / 12 per rank at split 2 / 3 / 4, so the 4-stage
# ring wraps several times.
_N, _K, _STAGES = 4096, 6144, 4
_INPUTS = 4  # distinct activations, cycled over consecutive launches
_LAUNCHES = 16  # GEMM launches per graph or eager round
_GRAPH_REPLAYS = 500
_EAGER_ROUNDS = 20
_SLEEP_CYCLES = 4_000_000  # about 2 ms: the eager launches queue up behind it

# Kernel tile (mma_m, mma_n) per public M: single row, full and partial M tiles,
# both mma_m values.
_TILES = {1: (128, 8), 8: (64, 8), 17: (128, 32), 32: (64, 32)}


def _require_splitk():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    if get_compute_capability(torch.device("cuda")) not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("The cluster split-K kernel requires SM100/SM103/SM107.")
    from flashinfer.cute_dsl.utils import is_cute_dsl_available

    if not is_cute_dsl_available():
        pytest.skip("nvidia-cutlass-dsl is not available.")


@pytest.mark.parametrize("m", sorted(_TILES))
@pytest.mark.parametrize("split_k", [2, 3, 4])
def test_splitk_replays_are_bitwise_stable(split_k, m):
    _require_splitk()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm100_splitk import (
        SplitKTactic,
        run_splitk_dense,
    )

    tactic = SplitKTactic(*_TILES[m], split_k, _STAGES)
    case = f"{tactic}, M={m}, N={_N}, K={_K}"

    generator = torch.Generator(device="cuda").manual_seed(100 * split_k + m)
    w = torch.randn((_N, _K), device="cuda", generator=generator) / _K**0.5
    w = w.to(torch.bfloat16)
    acts = torch.randn((_INPUTS, m, _K), device="cuda", generator=generator)
    acts = acts.to(torch.bfloat16)
    reference = acts.double() @ w.double().T
    # The weights are read before the kernel's PDL wait: publish them first.
    torch.cuda.synchronize()

    def launch(a, out):
        run_splitk_dense(a, w.T, None, out, True, tactic)

    first = torch.empty((_INPUTS, m, _N), device="cuda", dtype=torch.bfloat16)
    for i in range(_INPUTS):
        launch(acts[i], first[i])
    torch.cuda.synchronize()
    torch.testing.assert_close(
        first.double(),
        reference,
        rtol=1e-2,
        atol=1e-2,
        msg=lambda msg: f"first result vs FP64 reference ({case}): {msg}",
    )

    expected = first[torch.arange(_LAUNCHES, device="cuda") % _INPUTS]
    expected = expected.view(torch.int16)
    outs = torch.empty((_LAUNCHES, m, _N), device="cuda", dtype=torch.bfloat16)
    bad_values = torch.zeros((), device="cuda", dtype=torch.int64)
    bad_runs = torch.zeros((), device="cuda", dtype=torch.int64)

    def run_and_check():
        outs.fill_(float("nan"))
        for j in range(_LAUNCHES):
            launch(acts[j % _INPUTS], outs[j])
        mismatches = (outs.view(torch.int16) != expected).sum()
        bad_values.add_(mismatches)
        bad_runs.add_((mismatches > 0).long())

    def check(mode, runs):
        torch.cuda.synchronize()
        failed, values = bad_runs.item(), bad_values.item()
        assert failed == 0, (
            f"bitwise mismatch between split-K launches ({mode}, {case}): "
            f"{failed} of {runs} runs of {_LAUNCHES} launches differ from the "
            f"first result, {values} values in total"
        )
        bad_values.zero_()
        bad_runs.zero_()

    run_and_check()  # warm-up outside the graph
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_and_check()
    for _ in range(_GRAPH_REPLAYS):
        graph.replay()
    check("CUDA graph replays", 1 + _GRAPH_REPLAYS)
    del graph

    for _ in range(_EAGER_ROUNDS):
        torch.cuda._sleep(_SLEEP_CYCLES)
        run_and_check()
    check("eager back-to-back launches", _EAGER_ROUNDS)


if __name__ == "__main__":
    pytest.main([__file__])
