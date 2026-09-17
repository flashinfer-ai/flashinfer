"""%globaltimer timing vs cudaEvent timing.

The CC-safe autotuner path (``flashinfer/autotuner.py``) times candidate tactics
with the GPU ``%globaltimer`` register instead of ``cudaEvent``, because under
Confidential Computing ``cudaEventElapsedTime`` is unreliable. This test checks
that, off-CC, the two timers measure the *same* GPU work to within tolerance —
so switching timers doesn't change which tactic the autotuner ranks fastest.

Mirrors TensorRT-LLM PR #11657's ``test_global_timer_vs_cuda_event``
(https://github.com/NVIDIA/TensorRT-LLM/pull/11657), adapted to FlashInfer's
``flashinfer.utils.get_globaltimer_kernel()`` (which stamps the current CUDA
stream into a 1-element int64 tensor).
"""

import statistics

import pytest
import torch

from flashinfer.utils import get_globaltimer_kernel

# GEMM shapes (m, k, n) borrowed from the TRT-LLM reference test.
_SHAPES = [(256, 4096, 11008), (512, 8192, 8192)]
# Matmuls per timed run: keep the measured interval well above %globaltimer
# granularity and cudaEvent overhead so the comparison is meaningful.
_REPEAT = 10
_TRIALS = 6

# Tolerance, matching the TRT-LLM reference: pass if the two means agree to
# within an absolute floor, OR a relative fraction, OR a statistical band.
_ABS_TOL_MS = 0.01
_REL_TOL = 0.05
_STAT_ZSCORE = 3.0


def _require_globaltimer():
    kernel = get_globaltimer_kernel()
    if kernel is None:
        pytest.skip("could not JIT-build the %globaltimer stamp kernel")
    return kernel


def _sem(samples) -> float:
    """Standard error of the mean (0 for a single sample)."""
    if len(samples) < 2:
        return 0.0
    return statistics.stdev(samples) / (len(samples) ** 0.5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_globaltimer_monotonic():
    """Two stamps bracketing real GPU work must be strictly increasing, and a
    longer interval must read larger — a basic sanity check on the kernel."""
    kernel = _require_globaltimer()
    device = torch.device("cuda", torch.cuda.current_device())
    a = torch.randn(1024, 1024, device=device, dtype=torch.bfloat16)

    def stamp_around(n_matmuls: int) -> int:
        start = torch.empty(1, dtype=torch.int64, device=device)
        end = torch.empty(1, dtype=torch.int64, device=device)
        kernel(start)
        for _ in range(n_matmuls):
            torch.mm(a, a)
        kernel(end)
        torch.cuda.synchronize()
        return end.item() - start.item()

    stamp_around(4)  # warmup
    short = stamp_around(2)
    long = stamp_around(20)
    assert short > 0, "%globaltimer delta must be positive"
    assert long > short, "more work must read a larger %globaltimer delta"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_cuda_graph", [False, True], ids=["eager", "cudagraph"])
@pytest.mark.parametrize("shape", _SHAPES, ids=lambda s: "x".join(map(str, s)))
def test_global_timer_vs_cuda_event(use_cuda_graph, shape):
    kernel = _require_globaltimer()
    m, k, n = shape
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16

    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    c = torch.empty(m, n, device=device, dtype=dtype)

    def run():
        for _ in range(_REPEAT):
            torch.mm(a, b, out=c)

    # Warm up, then (optionally) capture the work into a CUDA graph so both
    # timers bracket a graph.replay() — the path the CC autotuner actually uses.
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    if use_cuda_graph:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        work = graph.replay
    else:
        work = run

    start_ts = torch.empty(1, dtype=torch.int64, device=device)
    end_ts = torch.empty(1, dtype=torch.int64, device=device)
    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    def time_both(timed) -> tuple[float, float]:
        """Time one execution of ``timed`` with both timers at once.

        Timing separate executions would measure run-to-run variation in the
        GEMMs themselves, not timer disagreement: on a power-limited part the
        two runs can differ by several percent, which is what the timers are
        being compared to. Bracketing a single execution removes that variance.

        The stamps sit outside the events, so the %globaltimer interval
        strictly contains the cudaEvent interval and %globaltimer reads larger
        by the bracket cost (two stamp launches, order 10us). Passing a no-op
        measures that cost on its own.
        """
        kernel(start_ts)
        start_evt.record()
        timed()
        end_evt.record()
        kernel(end_ts)
        torch.cuda.synchronize()
        return (
            start_evt.elapsed_time(end_evt),  # ms
            (end_ts.item() - start_ts.item()) / 1e6,  # ns -> ms
        )

    def noop():
        pass

    time_both(work)  # warm the timed path
    time_both(noop)

    event_times, gt_times, bracket_costs = [], [], []
    for _ in range(_TRIALS):
        event_time, gt_time = time_both(work)
        event_times.append(event_time)
        gt_times.append(gt_time)
        empty_event, empty_gt = time_both(noop)
        bracket_costs.append(empty_gt - empty_event)

    event_mean = statistics.mean(event_times)
    gt_mean = statistics.mean(gt_times)
    # Median: an occasional stamp launch stalls and would drag a mean up.
    bracket_cost = statistics.median(bracket_costs)

    assert event_mean > 0, f"cudaEvent mean must be positive, got {event_mean}"
    assert gt_mean > 0, f"%globaltimer mean must be positive, got {gt_mean}"

    # Paired samples: the shared workload variance cancels in the difference.
    paired_sem = _sem([gt - ev for ev, gt in zip(event_times, gt_times, strict=True)])
    allowed_diff = max(_ABS_TOL_MS, _REL_TOL * event_mean, _STAT_ZSCORE * paired_sem)
    diff = abs(gt_mean - bracket_cost - event_mean)
    assert diff <= allowed_diff, (
        f"%globaltimer vs cudaEvent mean disagree: "
        f"globaltimer={gt_mean:.4f}ms cudaEvent={event_mean:.4f}ms "
        f"bracket={bracket_cost:.4f}ms diff={diff:.4f}ms > allowed={allowed_diff:.4f}ms"
    )
