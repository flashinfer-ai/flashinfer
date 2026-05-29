"""Randomized stress test for the autotuner <-> cuDNN dynamic-shape interaction.

The cuDNN GEMM runners tune on a set of M (num-token) *buckets* and then must
serve arbitrary runtime M via cuDNN's override-shape graphs, replaying the
tuned tactic.  Historically this seam has been fragile:

  * a tactic stored as a bare plan *index* could silently mis-point after the
    plan list was re-enumerated (the fix under test stores structured
    (engine, knobs) tactics instead, when the frontend supports it);
  * a cache_m bucket mismatch between the runner and the autotuner produced
    sparse NaN/Inf at non-power-of-2 M;
  * out-of-range / between-bucket M must fall back without corruption.

This test hammers that seam: it autotunes over randomized bucket sets, then
runs a *different* random sweep of runtime M (in-range, between buckets, and
beyond the tuned range), asserting every result is finite and numerically
correct.  It also covers the on-disk autotune-cache round-trip, round_up
semantics, and interleaved
shapes within a single tuning context.

Run on a cuDNN override-shape-capable GPU (SM100+, cudnn-frontend >= 1.20 /
backend >= 9.21).  Otherwise the whole module skips.
"""

import os
import random

import pytest
import torch
import torch.nn.functional as F

from flashinfer import SfLayout, autotune, mm_fp4, nvfp4_quantize
from flashinfer.autotuner import AutoTuner
from flashinfer.utils import get_compute_capability, LibraryError
from flashinfer.gemm.gemm_base import (
    is_cudnn_override_shape_available,
    clear_cudnn_graph_cache,
)

# ---------------------------------------------------------------------------
# Module-level skip guards: this test only means something when the cuDNN
# override-shape GEMM path is actually live.
# ---------------------------------------------------------------------------
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

_CC = get_compute_capability(torch.device("cuda"))
_CC_NUM = _CC[0] * 10 + _CC[1]

if not mm_fp4.is_backend_supported("cudnn", _CC_NUM):
    pytest.skip(f"cuDNN mm_fp4 not supported on SM{_CC_NUM}", allow_module_level=True)
if not is_cudnn_override_shape_available():
    pytest.skip(
        "cuDNN override-shape GEMM not available (need cudnn-frontend>=1.20, "
        "backend>=9.21); the autotuner<->dynamic-shape seam under test is inactive",
        allow_module_level=True,
    )

COS_SIM_THRESHOLD = 0.97


def _make_nvfp4_operands(m, n, k, res_dtype, seed):
    """Build randomized nvfp4 operands + a bf16 reference for an m x k @ k x n GEMM."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randn([m, k], device="cuda", dtype=torch.bfloat16, generator=g)
    b = torch.randn([n, k], device="cuda", dtype=torch.bfloat16, generator=g)

    gsf_a = (448 * 6) / a.float().abs().nan_to_num().max()
    gsf_b = (448 * 6) / b.float().abs().nan_to_num().max()

    a_fp4, a_inv_s = nvfp4_quantize(
        a, gsf_a, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    b_fp4, b_inv_s = nvfp4_quantize(
        b, gsf_b, sfLayout=SfLayout.layout_128x4, do_shuffle=False
    )
    alpha = 1.0 / (gsf_a * gsf_b)
    reference = torch.mm(a, b.T)
    return dict(
        a_fp4=a_fp4,
        b_fp4=b_fp4,
        a_inv_s=a_inv_s,
        b_inv_s=b_inv_s,
        alpha=alpha,
        reference=reference,
        m=m,
        n=n,
        k=k,
        res_dtype=res_dtype,
    )


def _run_mm_fp4(op, backend="cudnn"):
    """Run mm_fp4 on prepared operands; return (res, cos_sim)."""
    res = torch.empty([op["m"], op["n"]], device="cuda", dtype=op["res_dtype"])
    mm_fp4(
        op["a_fp4"],
        op["b_fp4"].T,
        op["a_inv_s"],
        op["b_inv_s"].T,
        op["alpha"],
        op["res_dtype"],
        res,
        block_size=16,
        use_8x4_sf_layout=False,
        backend=backend,
        use_nvfp4=True,
        skip_check=False,
    )
    cos = F.cosine_similarity(
        op["reference"].reshape(-1).float(), res.reshape(-1).float(), dim=0
    )
    return res, float(cos)


def _assert_good(res, cos, where):
    assert torch.isfinite(res).all(), f"{where}: non-finite values in output"
    assert cos > COS_SIM_THRESHOLD, f"{where}: cos_sim {cos:.4f} below threshold"


# ---------------------------------------------------------------------------
# 1. Tune over a randomized bucket set, then serve a *different* random M
#    sweep (in-range, between buckets, beyond range).  This is the core
#    dynamic-shape correctness check.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("res_dtype", [torch.bfloat16, torch.float16])
def test_fuzz_dynamic_M_sweep_after_autotune(seed, res_dtype):
    rng = random.Random(seed)
    n = rng.choice([256, 512, 1024])
    k = rng.choice([256, 512, 1024])

    # Random monotonic bucket set covering a range of M.
    max_m = rng.choice([512, 1024, 2048])
    buckets = sorted(
        {1, 8, max_m, *[rng.randint(2, max_m) for _ in range(rng.randint(3, 6))]}
    )

    clear_cudnn_graph_cache()  # fresh tactic <-> graph state

    # Tune at each bucket M.
    try:
        with autotune(True, tuning_buckets=tuple(buckets)):
            for bm in buckets:
                op = _make_nvfp4_operands(bm, n, k, res_dtype, seed)
                _run_mm_fp4(op)
    except LibraryError as e:
        pytest.skip(f"cuDNN backend unavailable for this config: {e}")

    # Now serve a random sweep of runtime M -- deliberately including values
    # between buckets and beyond the tuned range -- using only cached tactics.
    sweep = (
        [rng.randint(1, max_m) for _ in range(8)]
        + [b + 1 for b in buckets if b + 1 <= max_m]  # just-above each bucket
        + [max_m + rng.randint(1, max_m)]  # beyond the tuned range
        + [1, 2, 3]  # tiny
    )
    with autotune(False):
        for m in sweep:
            op = _make_nvfp4_operands(m, n, k, res_dtype, seed + m)
            res, cos = _run_mm_fp4(op)
            _assert_good(res, cos, f"M={m} (n={n},k={k},buckets={buckets})")


# ---------------------------------------------------------------------------
# 2. On-disk autotune cache round-trip: tune+save, wipe in-memory state,
#    load, and serve random M.  Exercises plan-name / structured tactic
#    persistence through JSON.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [3, 4])
def test_fuzz_autotune_cache_roundtrip(seed, tmp_path):
    rng = random.Random(seed)
    n, k = 512, 512
    buckets = (1, 16, 64, 256, 1024)
    res_dtype = torch.bfloat16
    cache_file = os.path.join(tmp_path, "cudnn_fp4_cache.json")

    clear_cudnn_graph_cache()
    try:
        with autotune(True, tuning_buckets=buckets, cache=cache_file):
            for bm in buckets:
                op = _make_nvfp4_operands(bm, n, k, res_dtype, seed)
                _run_mm_fp4(op)
    except LibraryError as e:
        pytest.skip(f"cuDNN backend unavailable: {e}")

    assert os.path.isfile(cache_file), "autotune cache not written"

    # Wipe both the autotuner's in-memory cache and the cuDNN graph LRU, so the
    # only way to get a tuned tactic is to reload it from disk.
    clear_cudnn_graph_cache()
    AutoTuner.get().clear_cache()

    with autotune(False, cache=cache_file):
        for m in [rng.randint(1, 1024) for _ in range(10)]:
            op = _make_nvfp4_operands(m, n, k, res_dtype, seed + m)
            res, cos = _run_mm_fp4(op)
            _assert_good(res, cos, f"reloaded M={m}")


# ---------------------------------------------------------------------------
# 3. round_up True vs False: both must be numerically correct at
#    between-bucket M (they may pick different tactics).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("round_up", [False, True])
def test_fuzz_round_up_between_buckets(round_up):
    n, k = 512, 256
    buckets = (16, 128, 512)
    res_dtype = torch.bfloat16

    clear_cudnn_graph_cache()
    try:
        with autotune(True, tuning_buckets=buckets, round_up=round_up):
            for bm in buckets:
                op = _make_nvfp4_operands(bm, n, k, res_dtype, 7)
                _run_mm_fp4(op)
    except LibraryError as e:
        pytest.skip(f"cuDNN backend unavailable: {e}")

    with autotune(False, round_up=round_up):
        for m in [17, 63, 100, 200, 400, 511, 513]:  # between / above buckets
            op = _make_nvfp4_operands(m, n, k, res_dtype, 7 + m)
            res, cos = _run_mm_fp4(op)
            _assert_good(res, cos, f"round_up={round_up} M={m}")


# ---------------------------------------------------------------------------
# 4. Interleave many random M within a single tuning context -- stresses the
#    graph-cache / tactic-cache lockstep (a mismatch here was the NaN class).
# ---------------------------------------------------------------------------
def test_fuzz_interleaved_shapes_single_context():
    rng = random.Random(99)
    n, k = 256, 512
    res_dtype = torch.bfloat16

    clear_cudnn_graph_cache()
    ms = [rng.randint(1, 1500) for _ in range(25)]
    try:
        with autotune(True):
            for m in ms:
                op = _make_nvfp4_operands(m, n, k, res_dtype, m)
                res, cos = _run_mm_fp4(op)
                _assert_good(res, cos, f"interleaved tune M={m}")
    except LibraryError as e:
        pytest.skip(f"cuDNN backend unavailable: {e}")

    # Replay a reshuffled set under no-tune; must stay correct.
    rng.shuffle(ms)
    with autotune(False):
        for m in ms:
            op = _make_nvfp4_operands(m, n, k, res_dtype, m + 1)
            res, cos = _run_mm_fp4(op)
            _assert_good(res, cos, f"interleaved replay M={m}")


# ---------------------------------------------------------------------------
# 5. Cross-backend agreement: cuDNN and CUTLASS must agree (both vs reference)
#    on the same operands across a random M sweep.
# ---------------------------------------------------------------------------
def test_fuzz_cudnn_vs_cutlass_agreement():
    if not mm_fp4.is_backend_supported("cutlass", _CC_NUM):
        pytest.skip("cutlass mm_fp4 not supported here")
    rng = random.Random(123)
    n, k = 512, 512
    res_dtype = torch.bfloat16

    clear_cudnn_graph_cache()
    for _ in range(6):
        m = rng.randint(1, 1024)
        op = _make_nvfp4_operands(m, n, k, res_dtype, m)
        try:
            with autotune(True):
                res_cudnn, cos_cudnn = _run_mm_fp4(op, backend="cudnn")
                res_cutlass, cos_cutlass = _run_mm_fp4(op, backend="cutlass")
        except LibraryError as e:
            pytest.skip(f"backend unavailable: {e}")
        _assert_good(res_cudnn, cos_cudnn, f"cudnn M={m}")
        _assert_good(res_cutlass, cos_cutlass, f"cutlass M={m}")
        # The two backends should agree closely (same math, different kernels).
        cos_pair = F.cosine_similarity(
            res_cudnn.reshape(-1).float(), res_cutlass.reshape(-1).float(), dim=0
        )
        assert cos_pair > 0.99, f"cudnn vs cutlass disagree at M={m}: {cos_pair:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
