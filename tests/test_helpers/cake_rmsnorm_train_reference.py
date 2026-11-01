"""FP64 references, accuracy checks and registry readers shared by the coverage
suites of the Cake RMSNorm training package with the non-deterministic weight
gradient (``tests/norm/test_cake_rmsnorm_train_{ckpt,windows,all_sm,dist}.py``).

The budget is the deterministic package's: BF16 ``y`` / ``dx`` at ``atol = rtol =
1e-2`` after normalizing both sides to unit scale, FP32 ``rstd`` at relative L2
``<= 1e-5``, FP32 ``dw`` at relative L2 ``<= 1e-4`` with every element within
``atol = rtol = 1e-4`` of the FP64 reference; a single-kernel window's ``dw`` is
also not worse than the eager FP32 reduction of the same inputs (up to one FP32
ulp).  Nothing here imports a package: the registry readers take the package's
``cake_jit`` module, so a suite run against a vendored copy uses them unchanged.
"""

from __future__ import annotations

from typing import Optional

import torch

ATOL = 1e-2
RTOL = 1e-2
RSTD_REL_L2 = 1e-5
DW_REL_L2 = 1e-4
DW_ATOL = 1e-4
DW_RTOL = 1e-4
DW_EAGER_FLOOR = 2.0**-23
# The contract eps of every width.
EPS = {6144: 1e-5, 5120: 1e-6, 4096: 1e-5, 2048: 1e-6, 1536: 1e-6, 512: 1e-6, 256: 1e-6}
W_DTYPES = {"bf16": torch.bfloat16, "f32": torch.float32}


def reference_forward(x, w, eps, residual=None):
    """FP64 forward over the last axis; the residual boundary rounds to BF16 like the kernel."""
    normalized = (
        (x.double() + residual.double()).to(torch.bfloat16)
        if residual is not None
        else x
    )
    rstd = torch.rsqrt(normalized.double().pow(2).mean(-1) + eps)
    y = (normalized.double() * rstd.unsqueeze(-1)) * w.double()
    return y, rstd, normalized


def reference_backward(g, x, w, rstd, g_h=None):
    """FP64 backward at the kernel's FP32 ``rstd``: ``dx`` per row, ``dw`` over every leading axis."""
    xf, wf, gf = x.double(), w.double(), g.double()
    r = rstd.double().unsqueeze(-1)
    a = gf * wf
    c = (a * xf).sum(-1, keepdim=True) / x.shape[-1]
    dx = r * a - xf * r.pow(3) * c
    del a
    if g_h is not None:
        dx = dx + g_h.double()
    dw = (gf * xf * r).reshape(-1, x.shape[-1]).sum(0)
    return dx, dw


def rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = actual.double().flatten(), reference.double().flatten()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def assert_bf16_close(actual, reference, label: str) -> None:
    """BF16 ``actual`` within ``atol = rtol = 1e-2`` of ``reference`` at unit scale."""
    assert actual.dtype == torch.bfloat16, label
    a, b = actual.double(), reference.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    if a.numel() == 0:
        return
    scale = b.pow(2).mean().sqrt().item()
    scale = scale if scale > 0.0 else 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


def eager_fp32_dw(g, x, rstd) -> torch.Tensor:
    hidden = x.shape[-1]
    return (
        (g.float() * x.float() * rstd.float().unsqueeze(-1)).reshape(-1, hidden).sum(0)
    )


def assert_dw(dw, dw_ref, g, x, rstd, label: str, form: str) -> tuple[float, float]:
    """``dw`` at the deterministic budget against FP64 (rel-L2 and every element);
    a single-kernel window's ``dw`` also not worse than the eager FP32 reduction.
    Returns ``(rel-L2, worst element error / its tolerance)``."""
    assert dw.dtype == torch.float32, label
    assert torch.isfinite(dw).all(), f"{label} has non-finite values"
    error = rel_l2(dw, dw_ref)
    assert error <= DW_REL_L2, f"{label}: dw rel-L2 {error:.3e} > {DW_REL_L2:.0e}"
    diff = (dw.double() - dw_ref).abs()
    tolerance = DW_ATOL + DW_RTOL * dw_ref.abs()
    worst = float((diff / tolerance).max()) if dw.numel() else 0.0
    bad = diff > tolerance
    if bool(bad.any()):
        at = int((diff / tolerance).argmax())
        raise AssertionError(
            f"{label} ({form}): {int(bad.sum())} dw elements outside atol=rtol=1e-4 "
            f"(worst column {at}: {float(dw[at]):.9g} vs {float(dw_ref[at]):.9g}, "
            f"error/tolerance {worst:.3f}); rel-L2 {error:.3e}"
        )
    if form == "single":
        eager = rel_l2(eager_fp32_dw(g, x, rstd), dw_ref)
        assert error <= max(eager, DW_EAGER_FLOOR), (
            f"{label}: single-kernel dw rel-L2 {error:.3e} > eager FP32 {eager:.3e}"
        )
    return error, worst


def bf16_ulp_differences(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """``(differing elements, largest difference in BF16 ulps)`` of two BF16 tensors
    of one shape (the ulp distance of the ordered integer encodings; a sign change
    counts through zero)."""
    assert a.dtype == b.dtype == torch.bfloat16 and a.shape == b.shape
    ia = a.contiguous().view(torch.int16).to(torch.int32)
    ib = b.contiguous().view(torch.int16).to(torch.int32)
    differing = int((ia != ib).sum().item())
    if differing == 0:
        return 0, 0
    # monotone integer order of the encodings: negative values mirror below zero
    oa = torch.where(ia < 0, -(ia & 0x7FFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFF), ib)
    return differing, int((oa - ob).abs().max().item())


def fp32_ulp_differences(a: torch.Tensor, b: torch.Tensor) -> tuple[int, int]:
    """:func:`bf16_ulp_differences` for FP32 tensors."""
    assert a.dtype == b.dtype == torch.float32 and a.shape == b.shape
    ia = a.contiguous().view(torch.int32).to(torch.int64)
    ib = b.contiguous().view(torch.int32).to(torch.int64)
    differing = int((ia != ib).sum().item())
    if differing == 0:
        return 0, 0
    oa = torch.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    ob = torch.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return differing, int((oa - ob).abs().max().item())


# --------------------------------------------------------------------------- registry readers


def admits(rule: Optional[dict], rows: int) -> bool:
    if rule is None:
        return True
    assert rule["kind"] == "rows_between", rule
    return rows > int(rule["above"]) and (
        rule["at_most"] is None or rows <= int(rule["at_most"])
    )


def window_records(
    loader, arch: str, hidden: int, kind: str, residual: bool, w: str, rows: int
) -> list[str]:
    """The records one call of ``rows`` tokens launches, read from ``ROUTES`` and
    the select rules (independent of the host's ``selected_modules``)."""
    key = loader.route_key(hidden, kind, residual, w)
    return [
        name
        for name in loader.ROUTES[arch][key]
        if admits(loader.MODULES[name]["select_rule"], rows)
    ]


def form(loader, names: list[str]) -> str:
    """``single`` (one ``claim_sum_f64`` kernel) or ``pair`` (a ``chunk_partials``
    main kernel and its dw reduce) of one backward window's records."""
    records = [loader.MODULES[name] for name in names]
    owners = [r["workspace_rule"]["kind"] for r in records if r["workspace_rule"]]
    reduces = [r for r in records if r["role"] == "bwd_reduce"]
    if len(records) == 1 and owners == ["claim_sum_f64"]:
        return "single"
    if len(records) == 2 and owners == ["chunk_partials"] and len(reduces) == 1:
        return "pair"
    raise AssertionError(f"a backward window launches {names}: neither form")


def windows(loader, arch: str, hidden: int, kind: str, residual: bool, w: str):
    """``[(above, at_most, library)]`` of the route on ``arch`` (``program_windows``)."""
    return list(loader.program_windows(arch, hidden, kind, residual, w))


def workspace_at_rest(loader, ws: torch.Tensor) -> bool:
    """The documented rest state after every call: every byte before the pair
    partial rows (``PAIR_ROWS_OFFSET``) is zero."""
    torch.cuda.synchronize()
    return not bool(ws[: min(loader.PAIR_ROWS_OFFSET, ws.numel())].any())
