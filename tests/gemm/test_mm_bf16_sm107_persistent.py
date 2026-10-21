"""SM107 persistent BF16 GEMM behind ``mm_bf16(backend="cute-dsl")`` for M > 32.

A few fixed tactics cover both orientations (normal and swap-AB, whose output
is the M-major ``out^T``), 1-CTA and 2-CTA MMA, and direct split-K, including
an uneven K partition and CUDA-graph replay of the in-stream zero fill.

Tolerances are on the relative Frobenius error against an FP32 reference.
Rounding the output to BF16 alone costs about 1.7e-3 (unit roundoff 2^-8,
RMS ~ 2^-8 / sqrt(3) relative to |C|). Split-K adds a BF16 rounding per
partial (S partials of size ~|C| / sqrt(S): another ~1.7e-3 in total) and one
per reduce-add after the first (sizes growing to |C|), about
1.7e-3 * sqrt(1 + (S + 1) / 2 - 1 / S): 2.4e-3 at S=2 (3e-3 measured on VR200
for K=16384) and 3.9e-3 at S=8. The bounds below keep 2x headroom.
"""

import pytest
import torch
import torch.nn.functional as F

from flashinfer import autotune, mm_bf16
from flashinfer.utils import get_compute_capability

_TOL_NO_SPLIT = 4e-3
_TOL_SPLIT_K = 8e-3


def _skip_unless_sm107_cute_dsl():
    if get_compute_capability(torch.device("cuda")) != (10, 7):
        pytest.skip("The persistent BF16 GEMM runner is SM107 only.")
    from flashinfer.cute_dsl.availability import (
        is_cute_dsl_available,
        is_rubin_cute_dsl_available,
    )

    if not (is_cute_dsl_available() and is_rubin_cute_dsl_available()):
        pytest.skip("Requires nvidia-cutlass-dsl with SM107 support (>= 4.8).")


def _make_problem(m, n, k):
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    # Unit-variance outputs, so the error is not dominated by a few large values.
    w = (torch.randn((n, k), device="cuda") / k**0.5).to(torch.bfloat16)
    reference = F.linear(x.float(), w.float())
    return x, w, reference


def _rel_err(out, reference):
    return ((out.float() - reference).norm() / reference.norm()).item()


@pytest.mark.parametrize(
    "m,n,k,tactic",
    [
        # (swap_ab, use_2cta_instrs, mma_m, mma_n, cluster_m, cluster_n, split_k)
        # The VR200 winners for GLM o_proj and a small-N decode GEMM.
        (168, 6144, 16384, (True, True, 256, 128, 2, 1, 2)),
        (128, 6144, 16384, (True, False, 128, 128, 1, 1, 4)),
        (36, 1024, 6144, (True, False, 128, 64, 1, 1, 8)),
        # Normal orientation, with an M tail, without and with split-K.
        (144, 2624, 6144, (False, False, 128, 128, 1, 1, 1)),
        (40, 1024, 6144, (False, False, 128, 64, 1, 1, 2)),
        # 65 K tiles over 4 splits: an uneven partition (17/16/16/16).
        (64, 2048, 4160, (True, False, 128, 128, 1, 1, 4)),
    ],
)
def test_sm107_persistent_tactic(m, n, k, tactic):
    _skip_unless_sm107_cute_dsl()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm107 import (
        Sm107Bf16Tactic,
        run_persistent_dense,
    )

    x, w, reference = _make_problem(m, n, k)
    # Garbage in out: split-K must zero it itself.
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    tactic = Sm107Bf16Tactic(*tactic)
    run_persistent_dense(x, w.T, out, tactic)
    tol = _TOL_SPLIT_K if tactic.split_k > 1 else _TOL_NO_SPLIT
    assert _rel_err(out, reference) < tol


def test_sm107_persistent_split_k_graph_replay():
    """The zero fill is captured with the GEMM, so every replay starts from 0."""
    _skip_unless_sm107_cute_dsl()
    from flashinfer.gemm.kernels.dense_bf16_gemm_sm107 import (
        Sm107Bf16Tactic,
        run_persistent_dense,
    )

    m, n, k = 128, 2048, 8192
    x, w, reference = _make_problem(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    tactic = Sm107Bf16Tactic(True, False, 128, 128, 1, 1, 4)
    run_persistent_dense(x, w.T, out, tactic)  # compile outside the capture

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_persistent_dense(x, w.T, out, tactic)
    for _ in range(3):
        out.fill_(1.0)
        graph.replay()
        torch.cuda.synchronize()
        assert _rel_err(out, reference) < _TOL_SPLIT_K


@pytest.mark.parametrize(
    "m,n,k",
    [
        (128, 6144, 16384),  # split-K 4, 1-CTA 128x128
        (168, 6144, 16384),  # 2-CTA 256x128 with a token tail
        (256, 6016, 7168),
        (36, 1024, 6144),  # 128x64; many splits fit: capped at 8
        (64, 1024, 1536),  # K per slice limits the split to 2
        (300, 7168, 1536),  # no split fits in one wave
        (33, 2624, 6144),
    ],
)
def test_mm_bf16_cute_dsl_sm107_untuned(m, n, k):
    """Explicit cute-dsl serves M > 32 on SM107 without autotuning."""
    _skip_unless_sm107_cute_dsl()
    x, w, reference = _make_problem(m, n, k)
    out = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)
    mm_bf16(x, w.T, out=out, backend="cute-dsl")
    assert _rel_err(out, reference) < _TOL_SPLIT_K


def test_mm_bf16_cute_dsl_sm107_autotune():
    """Autotuning profiles the persistent runner; any winner is correct, also
    for another M of the same bucket."""
    _skip_unless_sm107_cute_dsl()
    n, k = 1024, 6144
    for m in (64, 40):
        x, w, reference = _make_problem(m, n, k)
        out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        with autotune(True):
            mm_bf16(x, w.T, out=out, backend="cute-dsl")
        assert _rel_err(out, reference) < _TOL_SPLIT_K
        mm_bf16(x, w.T, out=out, backend="cute-dsl")
        assert _rel_err(out, reference) < _TOL_SPLIT_K


def test_sm107_persistent_runner_only_under_explicit_cute_dsl():
    _skip_unless_sm107_cute_dsl()
    from flashinfer.gemm.gemm_base import _cute_dsl_bf16_runners

    m, n, k = 64, 1024, 6144
    x, w, _ = _make_problem(m, n, k)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    inputs = [x, w.T, None, False, out, None]

    def names(include):
        runners = _cute_dsl_bf16_runners(inputs, include_sm107_persistent=include)
        return [type(runner).__name__ for runner in runners]

    assert names(True)[0] == "CuteDSLSm107PersistentBf16Runner"
    assert "CuteDSLSm107PersistentBf16Runner" not in names(False)


@pytest.mark.parametrize(
    "n,k,with_bias,pdl,message",
    [
        (1024, 6144, True, False, "bias"),
        (1024, 6144, False, True, "PDL"),
        (1020, 6144, False, False, "multiples of 8"),
        (1024, 6148, False, False, "multiples of 8"),
    ],
)
def test_mm_bf16_cute_dsl_sm107_rejects_unsupported(n, k, with_bias, pdl, message):
    """No fallback stands behind cute-dsl at M > 32, so these raise up front."""
    _skip_unless_sm107_cute_dsl()
    m = 64
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn((n,), device="cuda", dtype=torch.bfloat16) if with_bias else None
    with pytest.raises(ValueError, match=message):
        mm_bf16(x, w.T, bias=bias, pdl=pdl, backend="cute-dsl")
