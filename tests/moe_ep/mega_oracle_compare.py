"""Arch-agnostic compare helpers for mega-kernel torch-oracle tests.

Lives outside the per-arch test files so the sm100 and sm120 MXFP8 suites can
share one band definition without dragging in each other's
``pytest.importorskip`` (the kernel trees are process-exclusive).
"""

from __future__ import annotations


def _assert_mega_oracle_term_band_close(y_kernel, combine_ref, *, ikr, label=""):
    """Kernel-vs-torch-oracle compare with a per-cell TERM-magnitude band.

    The kernel rounds each per-topk fc2 term to bf16 before the combine, so
    where large terms nearly cancel (terms ~±2000 summing to ~100) the
    achievable agreement is bounded by the bf16 round-off of the TERMS
    (2^-8 x sum_k |term_k| per cell), not of the final value — a flat atol
    calibrated on one arch's rounding trips on another's (1-cell |d|=16 vs
    atol=8 seen on B200 with a GB200-calibrated flat band). ``combine_ref``
    is the oracle's pre-reduce (tokens, topk, hidden) term stack, so the band
    is exact per cell. Coefficient: non-ikr terms take one bf16 round-trip
    (<=1 ULP each, safety 2); the ikr REDG reduce additionally accumulates
    the K terms in nondeterministic bf16 order (safety 8, mirroring
    ``_assert_ikr_close``). Sum_k |term_k| already carries the K factor.
    """
    import torch

    yk = y_kernel.to(torch.float32)
    terms = combine_ref.to(torch.float32)
    y_ref = terms.sum(dim=1)
    diff = (yk - y_ref).abs()
    term_band = terms.abs().sum(dim=1)
    coeff = 8.0 if ikr else 2.0
    tol = 1.0 + 0.05 * y_ref.abs() + coeff * 2.0**-8 * term_band
    overshoot = diff - tol
    worst = overshoot.max().item()
    flat = overshoot.argmax().item()
    t, h = divmod(flat, overshoot.shape[1])
    print(
        f"[mega oracle term-band{' ' + label if label else ''} ikr={ikr}] "
        f"worst margin {-worst:.4g} at cell ({t},{h}): "
        f"|d|={diff[t, h].item():.4g} tol={tol[t, h].item():.4g} "
        f"sum|terms|={term_band[t, h].item():.4g} ref={y_ref[t, h].item():.4g}"
    )
    assert worst <= 0.0, (
        f"oracle output outside the bf16 term-magnitude band "
        f"(worst overshoot {worst:.4f} at cell ({t},{h}): |d|={diff[t, h].item():.4f} "
        f"tol={tol[t, h].item():.4f} sum|terms|={term_band[t, h].item():.4f})"
    )
