"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests for the Cake fused BF16 RMSNorm training forward / backward
(SM100 / SM103 / SM107).

Every kernel output is compared with an FP64 reference of the same math:
BF16 ``y`` / ``dx`` at ``atol = rtol = 1e-2`` after normalizing both sides to
unit scale (so 1e-3 and 1e3 magnitude inputs are judged like N(0, 1) inputs),
FP32 ``rstd`` at relative L2 <= 1e-5 and FP32 ``dw`` at relative L2 <= 1e-4
and never worse than the eager FP32 reduction of the same formula.  The
matrix covers small and packed token counts (including 1 and 0), tile
boundaries, row-strided inputs, extreme scales, zero input, the residual-add
fusion (against an add-then-norm autograd reference for every input
gradient), bitwise reproducibility of ``dw``, changing token counts in one
process and the autograd wrapper end to end.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_rmsnorm_det_train`` facade and the ``flashinfer`` top-level
exposure are covered by ``test_cake_rmsnorm_det_train_api.py``.
"""

from __future__ import annotations

import pytest
import re
import statistics
from pathlib import Path

import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_det_train")
cake_build = cake_package("cake_rmsnorm_det_train", "cake_build")
loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
windows = cake_package("cake_rmsnorm_det_train", "cake_windows")
cake_rmsnorm_det = pkg.cake_rmsnorm_det
cake_rmsnorm_det_train_backward = pkg.cake_rmsnorm_det_train_backward
cake_rmsnorm_det_train_backward_workspace = (
    pkg.cake_rmsnorm_det_train_backward_workspace
)
cake_rmsnorm_det_train_backward_workspace_bytes = (
    pkg.cake_rmsnorm_det_train_backward_workspace_bytes
)
cake_rmsnorm_det_train_forward = pkg.cake_rmsnorm_det_train_forward
is_cake_rmsnorm_det_train_supported = pkg.is_cake_rmsnorm_det_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
HIDDEN_SIZES = tuple(hidden for hidden, _eps in WIDTHS)
EPS = {hidden: eps for hidden, eps in WIDTHS}
# Widths whose contract names both eps values: every case runs at each.
EPS_VARIANTS = {4096: (1e-5, 1e-6), 1536: (1e-5, 1e-6)}
ROWS = (1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 257, 16172, 16231)
# Further row counts of the H=1536 / H=4096 contract: the program-window edges, the 2^n +- 1 counts and the
# representative training shapes (GLM-5.3-Flash CP16 among them).
CONTRACT_ROWS = {
    4096: (1023, 1025, 4097, 14928, 16263, 16384, 32769, 65536),
    1536: (1023, 1025, 4097, 14928, 15566, 16263, 16384, 32769, 65536),
}
STRIDE_PADS = (0, 64, "2H")
# The row count standing in for the open end of a route's last program window.
OPEN_WINDOW_ROWS = 65536
# Bounds of the differences a window edge of class ``ulp`` / ``dx`` / ``unmeasured`` may show on the common rows
# of T = edge and T = edge + 1 (``cake_windows.EDGES``; the measured figures are in the docstring notes): rstd within
# a few FP32 ulps on at most half of the rows; y and dx within one BF16 ulp of the element -- or within a 2^-14 share
# of the row's RMS, for the elements a cancellation leaves far below it -- on at most a twentieth of the rows across
# an edge of class ``dx`` (whose forward is identical) and, across an edge of class ``ulp`` / ``unmeasured``, only on
# rows whose rstd differs (the residual H = 6144 route moves dx on about a tenth of the rows at its edge 1024).
EDGE_RSTD_ULPS = 4
EDGE_RSTD_ROW_SHARE = 0.5
EDGE_BF16_ROW_SHARE = 0.05
EDGE_BF16_ROW_FLOOR = 2.0**-14
SCALES = (1.0, 1e-3, 1e3)
ATOL = 1e-2
RTOL = 1e-2
RSTD_REL_L2 = 1e-5
DW_REL_L2 = 1e-4
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_unless_hidden(hidden: int) -> None:
    if not is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")


def _skip_unless_route(hidden: int, residual: bool) -> None:
    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def _rows(
    rows: int,
    hidden: int,
    *,
    stride_pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    generator: torch.Generator,
) -> torch.Tensor:
    """BF16 ``[rows, hidden]`` view with ``stride(0) == hidden + pad``."""

    pad = hidden if stride_pad == "2H" else int(stride_pad)
    storage_rows = max(rows, 1)
    if zeros:
        base = torch.zeros(
            storage_rows, hidden + pad, dtype=torch.bfloat16, device=_device()
        )
    else:
        base = (
            scale
            * torch.randn(
                storage_rows,
                hidden + pad,
                dtype=torch.float32,
                device=_device(),
                generator=generator,
            )
        ).to(torch.bfloat16)
    return base[:rows, :hidden]


def _weight(hidden: int, generator: torch.Generator) -> torch.Tensor:
    return (
        1.0
        + 0.1
        * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
    ).to(torch.bfloat16)


def _reference_rstd(x: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.double()
    if xf.shape[0] == 0:
        return torch.empty(0, dtype=torch.float64, device=x.device)
    return torch.rsqrt(xf.pow(2).mean(-1) + eps)


def _reference_forward(x, w, eps, residual=None):
    """FP64 forward; the residual boundary is rounded to BF16 like the kernel."""

    if residual is not None:
        h_new = (x.double() + residual.double()).to(torch.bfloat16)
        normalized = h_new
    else:
        h_new = None
        normalized = x
    rstd = _reference_rstd(normalized, eps)
    y = (normalized.double() * rstd.unsqueeze(-1)) * w.double()
    return y, rstd, h_new


def _reference_backward(g, x, w, eps, g_h=None):
    xf, wf, gf = x.double(), w.double(), g.double()
    hidden = x.shape[-1]
    rstd = _reference_rstd(x, eps).unsqueeze(-1)
    a = gf * wf
    c = (a * xf).sum(-1, keepdim=True) / hidden
    dx = rstd * a - xf * rstd.pow(3) * c
    if g_h is not None:
        dx = dx + g_h.double()
    dw = (gf * xf * rstd).sum(0)
    return dx, dw


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a = actual.double().flatten()
    b = reference.double().flatten()
    den = b.norm().item()
    num = (a - b).norm().item()
    return num / den if den > 0 else num


def _assert_bf16_close(
    actual: torch.Tensor, reference: torch.Tensor, label: str
) -> None:
    assert actual.dtype == torch.bfloat16, label
    a = actual.double()
    b = reference.double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    if a.numel() == 0:
        return
    scale = b.pow(2).mean().sqrt().item()
    if not (scale > 0.0) or scale != scale:
        scale = 1.0
    torch.testing.assert_close(a / scale, b / scale, atol=ATOL, rtol=RTOL, msg=label)


# (label, kernel dw rel-L2, eager-FP32-path dw rel-L2) of every dw case checked in this process; the paired
# "not worse than eager FP32" reading is an aggregate over the matrix (see the last test of this module).
_DW_ERRORS: list[tuple[str, float, float]] = []


def _assert_dw(dw: torch.Tensor, dw_ref: torch.Tensor, g, x, rstd, label: str) -> None:
    assert dw.dtype == torch.float32 and dw.is_contiguous(), label
    assert torch.isfinite(dw).all(), f"{label} has non-finite values"
    error = _rel_l2(dw, dw_ref)
    assert error <= DW_REL_L2, f"{label}: dw rel-L2 {error:.3e} > {DW_REL_L2:.0e}"
    # Paired eager FP32 path ``(g * x * rstd).sum(0)`` on the same inputs and the same FP32 ``rstd`` the kernel
    # consumed.  Both sit at the FP32 rounding floor (one ulp of the per-element products, rounded differently:
    # fused multiply-add versus two multiplies), so a per-case comparison flips at small T; "not worse" is judged
    # on the median ratio over the matrix in ``test_dw_not_worse_than_eager_fp32_over_the_matrix``.
    eager_fp32 = (g.float() * x.float() * rstd.float().unsqueeze(-1)).sum(0)
    _DW_ERRORS.append((label, error, _rel_l2(eager_fp32, dw_ref)))


def _check_case(
    hidden: int,
    rows: int,
    *,
    x_pad=0,
    g_pad=0,
    scale: float = 1.0,
    zeros: bool = False,
    residual: bool = False,
    spike: bool = False,
    eps: float | None = None,
) -> None:
    eps = EPS[hidden] if eps is None else eps
    pad_key = hidden if x_pad == "2H" else int(x_pad)
    generator = _generator(
        hidden, rows, pad_key, int(scale * 1e6) % 9973, zeros, residual, spike
    )
    x = _rows(
        rows, hidden, stride_pad=x_pad, scale=scale, zeros=zeros, generator=generator
    )
    if spike and rows > 0:
        row = min(rows - 1, 3)
        x[row] = (x[row].float() * 1e3).to(torch.bfloat16)
        x[row, min(hidden - 1, 17)] = 1e4
    w = _weight(hidden, generator)
    u = _rows(rows, hidden, scale=scale, generator=generator) if residual else None
    g = _rows(rows, hidden, stride_pad=g_pad, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None

    outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
    y_ref, rstd_ref, h_new_ref = _reference_forward(x, w, eps, residual=u)
    y, rstd = outputs[0], outputs[1]
    assert tuple(y.shape) == (rows, hidden) and y.is_contiguous()
    assert rstd.dtype == torch.float32 and tuple(rstd.shape) == (rows,)
    label = f"hidden={hidden} rows={rows} eps={eps} pad={x_pad} scale={scale} zeros={zeros} residual={residual}"
    _assert_bf16_close(y, y_ref, f"y {label}")
    assert torch.isfinite(rstd).all(), f"rstd {label}"
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    if residual:
        h_new = outputs[2]
        assert torch.equal(h_new, h_new_ref), (
            f"h_new {label} is not the BF16-rounded sum"
        )
        normalized = h_new
    else:
        assert len(outputs) == 2
        normalized = x

    dx, dw = cake_rmsnorm_det_train_backward(g, normalized, w, rstd, g_residual=g_h)
    dx_ref, dw_ref = _reference_backward(g, normalized, w, eps, g_h=g_h)
    assert tuple(dx.shape) == (rows, hidden) and dx.is_contiguous()
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    _assert_dw(dw, dw_ref, g, normalized, rstd, f"dw {label}")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", ROWS)
def test_forward_backward_match_fp64(hidden: int, rows: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows)


def _switch_rows(arch: str, hidden: int, kind: str, residual: bool) -> tuple[int, ...]:
    """Row counts on both sides of every program switch point of one route (and a single row)."""

    switches = set()
    for name in loader.route_modules(arch, hidden, kind, residual):
        rule = loader.MODULES[name]["select_rule"]
        if rule is None:
            continue
        switches.update(
            int(bound) for bound in (rule["above"], rule["at_most"]) if bound
        )
    return tuple(sorted({1, *switches, *(rows + 1 for rows in switches)}))


@pytest.mark.parametrize(
    ("hidden", "rows"),
    [(hidden, rows) for hidden, counts in CONTRACT_ROWS.items() for rows in counts],
)
def test_contract_rows_match_fp64(hidden: int, rows: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows)


@pytest.mark.parametrize(
    ("hidden", "eps"),
    [(hidden, eps) for hidden, variants in EPS_VARIANTS.items() for eps in variants],
)
@pytest.mark.parametrize("rows", (7, 16231))
@pytest.mark.parametrize("scale", SCALES)
def test_eps_variants(hidden: int, eps: float, rows: int, scale: float) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, scale=scale, eps=eps)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_forward_programs_switch_on_row_count(hidden: int, residual: bool) -> None:
    """Exactly one forward program runs per call, every exported program runs for some row count, and each
    matches the FP64 reference on both sides of its switch point."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    names = loader.route_modules(arch, hidden, "fwd", residual)
    eps = EPS[hidden]
    launched = set()
    for rows in _switch_rows(arch, hidden, "fwd", residual):
        selected = loader.selected_modules(arch, hidden, "fwd", residual, rows=rows)
        assert len(selected) == 1, (
            f"hidden={hidden} residual={residual} rows={rows}: {selected}"
        )
        launched.add(selected[0])
        generator = _generator(hidden, rows, 43, residual)
        x = _rows(rows, hidden, generator=generator)
        w = _weight(hidden, generator)
        u = _rows(rows, hidden, generator=generator) if residual else None
        outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
        y_ref, rstd_ref, h_new_ref = _reference_forward(x, w, eps, residual=u)
        label = f"hidden={hidden} residual={residual} rows={rows}"
        _assert_bf16_close(outputs[0], y_ref, f"y {label}")
        assert _rel_l2(outputs[1], rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
        if residual:
            assert torch.equal(outputs[2], h_new_ref), f"h_new {label}"
    assert launched == set(names), (
        f"hidden={hidden} residual={residual}: never launched {set(names) - launched}"
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_backward_programs_switch_on_row_count(hidden: int) -> None:
    """Exactly one pair -- a main program and its dw reduce program -- runs per
    call, every exported backward program runs for some row count, and forward
    + backward match the FP64 reference on both sides of each switch point."""

    _skip_unless_hidden(hidden)
    arch = loader.arch_for_capability(_capability())
    names = loader.route_modules(arch, hidden, "bwd", False)
    launched = set()
    for rows in _switch_rows(arch, hidden, "bwd", False):
        selected = loader.selected_modules(arch, hidden, "bwd", False, rows=rows)
        assert len(selected) == 2, f"hidden={hidden} rows={rows}: {selected}"
        main, reduce = (loader.MODULES[name] for name in selected)
        assert main.get("workspace_rule") is not None and reduce["role"] == "bwd_reduce"
        launched.update(selected)
        _check_case(hidden, rows)
    assert launched == set(names), (
        f"hidden={hidden}: never launched {set(names) - launched}"
    )


# --------------------------------------------------------------------------- the program windows


def _route_edges(
    arch: str, hidden: int, kind: str, residual: bool
) -> list[tuple[int, str]]:
    """``(edge, class)`` of every window edge a call of the route crosses, the edges from the registry's windows
    (``program_windows``) and the class from the exporter's measurements (``cake_windows.EDGES``); a backward route
    crosses the forward's edges inside its windows as well (the backward consumes the forward's rstd) and takes the
    forward's class there."""

    route = loader.route_key(hidden, kind, residual)
    classes = dict(windows.EDGES[arch][route])
    edges = {}
    for _low, high, _library in loader.program_windows(arch, hidden, kind, residual):
        if high is not None:
            edges[high] = classes[high]
    if kind == "bwd":
        forward = dict(windows.EDGES[arch][loader.route_key(hidden, "fwd", residual)])
        for _low, high, _library in loader.program_windows(
            arch, hidden, "fwd", residual
        ):
            if high is not None:
                edges.setdefault(high, forward[high])
    return sorted(edges.items())


def _window_spans(
    arch: str, hidden: int, kind: str, residual: bool
) -> list[tuple[int, int]]:
    """``(first, last)`` row counts of every window of the route (a backward route's windows cut at the forward's
    edges too); the open last window ends at ``OPEN_WINDOW_ROWS``, or ``OPEN_WINDOW_ROWS`` past the last edge when
    that edge is not below it."""

    edges = [edge for edge, _cls in _route_edges(arch, hidden, kind, residual)]
    top = max(edges, default=0)
    end = OPEN_WINDOW_ROWS if top < OPEN_WINDOW_ROWS else top + OPEN_WINDOW_ROWS
    bounds = [0, *edges, end]
    return [(low + 1, high) for low, high in zip(bounds, bounds[1:], strict=False)]


def _base_inputs(
    hidden: int, rows: int, kind: str, residual: bool
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """One random input of ``rows`` rows whose prefixes the window tests slice, and the weight."""

    generator = _generator(hidden, 61, kind == "bwd", residual)
    base = {"x": _rows(rows, hidden, generator=generator)}
    w = _weight(hidden, generator)
    if residual:
        base["u"] = _rows(rows, hidden, generator=generator)
    if kind == "bwd":
        base["g"] = _rows(rows, hidden, generator=generator)
        if residual:
            base["g_h"] = _rows(rows, hidden, generator=generator)
    return base, w


def _route_outputs(
    kind: str,
    residual: bool,
    base: dict[str, torch.Tensor],
    rows: int,
    w: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    """The outputs of one call of the route on the first ``rows`` rows of ``base``: y, rstd (and h_new) of the
    forward, and dx as well for a backward route (dw sums every row and depends on T by definition)."""

    x = base["x"][:rows]
    u = base["u"][:rows] if residual else None
    outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
    result = {"y": outputs[0], "rstd": outputs[1]}
    if residual:
        result["h_new"] = outputs[2]
    if kind == "bwd":
        g_h = base["g_h"][:rows] if residual else None
        dx, _dw = cake_rmsnorm_det_train_backward(
            base["g"][:rows],
            outputs[2] if residual else x,
            w,
            outputs[1],
            g_residual=g_h,
        )
        result["dx"] = dx
    return result


def _ordered_bits(t: torch.Tensor) -> torch.Tensor:
    """The float bits as an integer key monotone in the value: the key distance of two floats is their distance in
    ulps of the format."""

    if t.dtype == torch.bfloat16:
        bits, sign = t.view(torch.int16).to(torch.int64), 1 << 15
    else:
        assert t.dtype == torch.float32
        bits, sign = t.view(torch.int32).to(torch.int64), 1 << 31
    return torch.where(bits >= 0, bits, -(bits + sign))


def _ulps_apart(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (_ordered_bits(a) - _ordered_bits(b)).abs()


def _differing_rows(a: torch.Tensor, b: torch.Tensor) -> int:
    differs = a != b
    if differs.ndim > 1:
        differs = differs.reshape(differs.shape[0], -1).any(dim=1)
    return int(differs.sum().item())


def _assert_within_a_bf16_ulp(
    a: torch.Tensor,
    b: torch.Tensor,
    label: str,
    *,
    row_share: float = EDGE_BF16_ROW_SHARE,
) -> None:
    """``a`` and ``b`` (BF16 ``[rows, H]``) agree to one BF16 ulp per element -- or to ``EDGE_BF16_ROW_FLOOR`` of the
    row's RMS where a cancellation leaves the element far below it -- and differ on at most ``row_share`` of the
    rows."""

    floor = EDGE_BF16_ROW_FLOOR * a.float().pow(2).mean(dim=1, keepdim=True).sqrt()
    within = (_ulps_apart(a, b) <= 1) | ((a.float() - b.float()).abs() <= floor)
    assert within.all(), f"{label}: {int((~within).sum())} elements beyond one BF16 ulp"
    rows = _differing_rows(a, b)
    assert rows <= row_share * a.shape[0], (
        f"{label}: differs on {rows} of {a.shape[0]} rows"
    )


def _assert_edge_class(
    low: dict[str, torch.Tensor], high: dict[str, torch.Tensor], cls: str, label: str
) -> None:
    """The outputs of T = edge (``low``) and T = edge + 1 (``high``) on their common rows behave as the edge's class
    says: ``same`` -- bitwise identical; ``ulp`` -- rstd differs on some rows by a few FP32 ulps, y and dx by at most
    one BF16 ulp and only on rows whose rstd differs; ``dx`` -- y and rstd identical, dx differs on a few rows (the
    base and the bulk-copy backward reduce a row in different orders); ``unmeasured`` -- bounded like ``ulp``."""

    rows = low["rstd"].shape[0]
    common = {key: value[:rows] for key, value in high.items()}
    assert set(low) == set(common), label
    if cls == "same":
        for key, value in low.items():
            assert torch.equal(value, common[key]), (
                f"{label}: {key} differs across an edge of class 'same'"
            )
        return
    assert cls in ("ulp", "dx", "unmeasured"), (label, cls)
    if "h_new" in low:
        assert torch.equal(low["h_new"], common["h_new"]), f"{label}: h_new differs"
    same_rstd = low["rstd"] == common["rstd"]
    rstd_rows = rows - int(same_rstd.sum().item())
    if cls == "dx":
        assert rstd_rows == 0 and torch.equal(low["y"], common["y"]), (
            f"{label}: the forward differs across an edge of class 'dx' ({rstd_rows} rstd rows)"
        )
    else:
        assert (_ulps_apart(low["rstd"], common["rstd"]) <= EDGE_RSTD_ULPS).all(), (
            f"{label}: rstd beyond {EDGE_RSTD_ULPS} FP32 ulps"
        )
        assert rstd_rows <= EDGE_RSTD_ROW_SHARE * rows, (
            f"{label}: rstd differs on {rstd_rows} of {rows} rows"
        )
        # y and dx move only with rstd (asserted exactly for class 'ulp' below): the rstd share bounds their rows.
        _assert_within_a_bf16_ulp(
            low["y"], common["y"], f"{label}: y", row_share=EDGE_RSTD_ROW_SHARE
        )
        if cls == "ulp":
            assert rstd_rows > 0, (
                f"{label}: rstd is identical across an edge of class 'ulp' (re-measure the edge)"
            )
            assert torch.equal(low["y"][same_rstd], common["y"][same_rstd]), (
                f"{label}: y differs on rows whose rstd agrees"
            )
    if "dx" in low:
        share = EDGE_BF16_ROW_SHARE if cls == "dx" else EDGE_RSTD_ROW_SHARE
        _assert_within_a_bf16_ulp(
            low["dx"], common["dx"], f"{label}: dx", row_share=share
        )
        if cls == "ulp":
            assert torch.equal(low["dx"][same_rstd], common["dx"][same_rstd]), (
                f"{label}: dx differs on rows whose rstd agrees"
            )
        elif cls == "dx" and rows >= 4096:
            assert _differing_rows(low["dx"], common["dx"]) > 0, (
                f"{label}: dx is identical across an edge of class 'dx' (re-measure the edge)"
            )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("kind", ("fwd", "bwd"))
@pytest.mark.parametrize("residual", (False, True))
def test_outputs_are_prefix_invariant_within_windows(
    hidden: int, kind: str, residual: bool
) -> None:
    """Within one program window a row's outputs do not depend on T: for every window of the route (a backward
    route's windows cut at the forward's edges too) the calls at the window's first and last row count
    (``_window_spans``' end standing for an open window) return bitwise-identical y, rstd, h_new and dx on their
    common rows."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    label = f"{arch} {loader.route_key(hidden, kind, residual)}"
    spans = _window_spans(arch, hidden, kind, residual)
    base, w = _base_inputs(hidden, max(last for _first, last in spans), kind, residual)
    for first, last in spans:
        low = _route_outputs(kind, residual, base, first, w, EPS[hidden])
        high = _route_outputs(kind, residual, base, last, w, EPS[hidden])
        for key, value in low.items():
            assert torch.equal(value, high[key][:first]), (
                f"{label} window [{first}, {last}]: {key} depends on T"
            )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("kind", ("fwd", "bwd"))
@pytest.mark.parametrize("residual", (False, True))
def test_window_edges_match_the_measured_classes(
    hidden: int, kind: str, residual: bool
) -> None:
    """At every window edge of the route the outputs of T = edge and T = edge + 1 behave on their common rows as the
    exporter measured (``cake_windows.EDGES``, see ``_assert_edge_class``); an unmeasured edge is bounded and named."""

    _skip_unless_route(hidden, residual)
    arch = loader.arch_for_capability(_capability())
    route = loader.route_key(hidden, kind, residual)
    edges = _route_edges(arch, hidden, kind, residual)
    if not edges:
        pytest.skip(f"{arch} {route} has one program window")
    base, w = _base_inputs(hidden, edges[-1][0] + 1, kind, residual)
    unmeasured = []
    for edge, cls in edges:
        low = _route_outputs(kind, residual, base, edge, w, EPS[hidden])
        high = _route_outputs(kind, residual, base, edge + 1, w, EPS[hidden])
        _assert_edge_class(low, high, cls, f"{arch} {route} edge {edge} (class {cls})")
        if cls == "unmeasured":
            unmeasured.append(edge)
    if unmeasured:
        pytest.skip(
            f"{arch} {route}: edges {unmeasured} are within bounds but unmeasured"
        )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("stride_pad", STRIDE_PADS[1:])
@pytest.mark.parametrize("rows", (7, 129, 16231))
def test_row_strided_inputs(hidden: int, stride_pad, rows: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, x_pad=stride_pad, g_pad=stride_pad)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("scale", SCALES[1:])
@pytest.mark.parametrize("rows", (7, 16231))
def test_input_scales(hidden: int, scale: float, rows: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, scale=scale)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (1, 129))
def test_zero_input_has_no_nan(hidden: int, rows: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, rows, zeros=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_extreme_row(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    _check_case(hidden, 257, spike=True)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("stride_pad", STRIDE_PADS)
@pytest.mark.parametrize("rows", (1, 129, 16231))
def test_residual_fusion(hidden: int, stride_pad, rows: int) -> None:
    _skip_unless_hidden(hidden)
    if not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")
    _check_case(hidden, rows, x_pad=stride_pad, g_pad=stride_pad, residual=True)


MISALIGNED_LAYOUTS = ("offset1", "stride_h_plus_1")


def _misaligned_rows(
    rows: int, hidden: int, layout: str, *, generator: torch.Generator
) -> torch.Tensor:
    """A ``[rows, hidden]`` BF16 view the kernels cannot read in place.

    ``offset1``: ``buf[:, 1:1 + H]`` of a ``[T, H + 16]`` buffer -- odd column offset
    (storage base 2 bytes past a 16-byte boundary; aligned row stride).
    ``stride_h_plus_1``: ``buf[:, :H]`` of a ``[T, H + 1]`` buffer -- odd row stride.
    """

    storage_rows = max(rows, 1)
    if layout == "offset1":
        base = torch.randn(
            storage_rows,
            hidden + 16,
            dtype=torch.float32,
            device=_device(),
            generator=generator,
        )
        view = base.to(torch.bfloat16)[:rows, 1 : 1 + hidden]
    else:
        base = torch.randn(
            storage_rows,
            hidden + 1,
            dtype=torch.float32,
            device=_device(),
            generator=generator,
        )
        view = base.to(torch.bfloat16)[:rows, :hidden]
    assert view.data_ptr() % 16 != 0 or (view.stride(0) * view.element_size()) % 16 != 0
    return view


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("layout", MISALIGNED_LAYOUTS)
@pytest.mark.parametrize("residual", (False, True))
def test_misaligned_layouts_are_materialized(
    hidden: int, layout: str, residual: bool
) -> None:
    """Odd column offsets and odd row strides on every ``[T, H]`` input and output (fwd, bwd, residual)."""

    _skip_unless_route(hidden, residual)
    eps = EPS[hidden]
    rows = 129
    generator = _generator(hidden, rows, 41, layout == "offset1", residual)
    x = _misaligned_rows(rows, hidden, layout, generator=generator)
    w = _weight(hidden, generator)
    u = (
        _misaligned_rows(rows, hidden, layout, generator=generator)
        if residual
        else None
    )
    g = _misaligned_rows(rows, hidden, layout, generator=generator)
    g_h = (
        _misaligned_rows(rows, hidden, layout, generator=generator)
        if residual
        else None
    )
    label = f"hidden={hidden} layout={layout} residual={residual}"

    outputs = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
    y_ref, rstd_ref, h_new_ref = _reference_forward(x, w, eps, residual=u)
    y, rstd = outputs[0], outputs[1]
    assert y.is_contiguous() and tuple(y.shape) == (rows, hidden)
    _assert_bf16_close(y, y_ref, f"y {label}")
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    if residual:
        assert torch.equal(outputs[2], h_new_ref), f"h_new {label}"
    # Caller-provided outputs with the same awkward layout are honoured by copy-back.
    y_user = _misaligned_rows(rows, hidden, layout, generator=generator)
    outputs_user = cake_rmsnorm_det_train_forward(x, w, eps, residual=u, out=y_user)
    assert outputs_user[0] is y_user and torch.equal(y_user, y), (
        f"out copy-back {label}"
    )

    # Backward: the normalized tensor, both gradients and the dx buffer are misaligned views.
    normalized = _misaligned_rows(rows, hidden, layout, generator=generator)
    normalized.copy_(outputs[2] if residual else x)
    dx, dw = cake_rmsnorm_det_train_backward(g, normalized, w, rstd, g_residual=g_h)
    dx_ref, dw_ref = _reference_backward(g, normalized, w, eps, g_h=g_h)
    assert dx.is_contiguous() and tuple(dx.shape) == (rows, hidden)
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    _assert_dw(dw, dw_ref, g, normalized, rstd, f"dw {label}")
    dx_user = _misaligned_rows(rows, hidden, layout, generator=generator)
    dx_again, dw_again = cake_rmsnorm_det_train_backward(
        g, normalized, w, rstd, g_residual=g_h, dx=dx_user
    )
    assert dx_again is dx_user and torch.equal(dx_user, dx), f"dx copy-back {label}"
    assert torch.equal(dw_again, dw), f"dw bitwise across layouts {label}"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_half_vector_aligned_weight_is_materialized(hidden: int) -> None:
    """A contiguous ``w`` whose storage base is 16 bytes past a 32-byte boundary.

    The backward kernels load the weight with 16-byte vectors and read this view
    in place; the forward kernels load it with 32-byte vectors, so the forward
    must copy it (an in-place read would be a misaligned vector load).  Both
    match the FP64 reference and the aligned weight's results bitwise.
    """

    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    rows = 129
    generator = _generator(hidden, rows, 47)
    x = _rows(rows, hidden, generator=generator)
    w_aligned = _weight(hidden, generator)
    storage = torch.empty(hidden + 16, dtype=torch.bfloat16, device=_device())
    offset = ((16 - storage.data_ptr() % 32) % 32) // storage.element_size()
    w = storage[offset : offset + hidden]
    w.copy_(w_aligned)
    assert w.is_contiguous() and w.data_ptr() % 32 == 16, "test layout"
    label = f"hidden={hidden} 16-byte aligned w"

    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    y_ref, rstd_ref, _ = _reference_forward(x, w, eps)
    _assert_bf16_close(y, y_ref, f"y {label}")
    assert _rel_l2(rstd, rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    y_aligned, rstd_aligned = cake_rmsnorm_det_train_forward(x, w_aligned, eps)
    assert torch.equal(y, y_aligned) and torch.equal(rstd, rstd_aligned), label

    g = _rows(rows, hidden, generator=generator)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    dx_ref, dw_ref = _reference_backward(g, x, w, eps)
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    _assert_dw(dw, dw_ref, g, x, rstd, f"dw {label}")
    dx_aligned, dw_aligned = cake_rmsnorm_det_train_backward(g, x, w_aligned, rstd)
    assert torch.equal(dx, dx_aligned) and torch.equal(dw, dw_aligned), label


# --------------------------------------------------------------------------- N-D inputs (the row collapse)


def _nd_rows(
    rows: int, heads: int, hidden: int, *, pitch_pad: int, generator: torch.Generator
) -> torch.Tensor:
    """BF16 ``[rows, heads, hidden]`` view of a ``[rows, heads, hidden + pitch_pad]`` buffer: strides
    ``(heads x (hidden + pad), hidden + pad, 1)``, the issue's layout when ``pad > 0``."""

    base = torch.randn(
        max(rows, 1),
        heads,
        hidden + pitch_pad,
        dtype=torch.float32,
        device=_device(),
        generator=generator,
    ).to(torch.bfloat16)
    return base[:rows, :, :hidden]


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("heads", (1, 3, 24))
@pytest.mark.parametrize("rows", (0, 1, 7, 129))
@pytest.mark.parametrize("pitch_pad", (0, 16, 256))
def test_nd_inputs_collapse_to_rows(
    hidden: int, heads: int, rows: int, pitch_pad: int
) -> None:
    """A ``[T, heads, H]`` input whose leading dimensions are contiguous with one another (strides
    ``(heads x pitch, pitch, 1)``) is the ``[T x heads, H]`` call: y, rstd, dx and dw bitwise equal to the 2-D call
    on the same rows, the outputs in the input's leading shape, a flat rstd accepted by the backward too."""

    _skip_unless_hidden(hidden)
    generator = _generator(hidden, heads, rows, pitch_pad, 71)
    x3 = _nd_rows(rows, heads, hidden, pitch_pad=pitch_pad, generator=generator)
    if (hidden, heads, pitch_pad) == (256, 24, 256):
        assert tuple(x3.stride()) == (12288, 512, 1)  # the issue's per-head view
    w = _weight(hidden, generator)
    g3 = _nd_rows(rows, heads, hidden, pitch_pad=pitch_pad, generator=generator)
    eps = EPS[hidden]
    x2 = x3.reshape(rows * heads, hidden)
    g2 = g3.reshape(rows * heads, hidden)
    y3, rstd3 = cake_rmsnorm_det_train_forward(x3, w, eps)
    y2, rstd2 = cake_rmsnorm_det_train_forward(x2, w, eps)
    assert tuple(y3.shape) == (rows, heads, hidden) and y3.is_contiguous()
    assert tuple(rstd3.shape) == (rows, heads) and rstd3.dtype == torch.float32
    assert torch.equal(y3.reshape(-1, hidden), y2) and torch.equal(
        rstd3.reshape(-1), rstd2
    )
    dx3, dw3 = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3)
    dx2, dw2 = cake_rmsnorm_det_train_backward(g2, x2, w, rstd2)
    assert tuple(dx3.shape) == (rows, heads, hidden) and dx3.is_contiguous()
    assert torch.equal(dx3.reshape(-1, hidden), dx2) and torch.equal(dw3, dw2)
    dx3_flat, dw3_flat = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3.reshape(-1))
    assert torch.equal(dx3_flat, dx3) and torch.equal(dw3_flat, dw3)
    if rows == 0:
        return
    # Preallocated N-D outputs: contiguous ones are written in place, a strided one receives a copy.
    out = torch.empty_like(y3)
    rstd_out = torch.empty(rows, heads, dtype=torch.float32, device=_device())
    outputs = cake_rmsnorm_det_train_forward(x3, w, eps, out=out, rstd=rstd_out)
    assert outputs[0] is out and outputs[1] is rstd_out
    assert torch.equal(out, y3) and torch.equal(rstd_out, rstd3)
    strided = torch.empty(
        rows, heads, hidden + 16, dtype=torch.bfloat16, device=_device()
    )[..., :hidden]
    outputs = cake_rmsnorm_det_train_forward(x3, w, eps, out=strided)
    assert outputs[0] is strided and torch.equal(strided, y3)
    dx_out = torch.empty_like(dx3)
    outputs = cake_rmsnorm_det_train_backward(g3, x3, w, rstd3, dx=dx_out)
    assert outputs[0] is dx_out and torch.equal(dx_out, dx3)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_nd_autograd_matches_the_2d_call(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    rows, heads = 7, 3
    generator = _generator(hidden, rows, heads, 73)
    x3 = _nd_rows(rows, heads, hidden, pitch_pad=16, generator=generator)
    g3 = _nd_rows(rows, heads, hidden, pitch_pad=0, generator=generator)
    w = _weight(hidden, generator)
    x3_leaf = x3.detach().clone().requires_grad_(True)
    x2_leaf = x3.reshape(rows * heads, hidden).detach().clone().requires_grad_(True)
    w3 = w.detach().clone().requires_grad_(True)
    w2 = w.detach().clone().requires_grad_(True)
    y3 = cake_rmsnorm_det(x3_leaf, w3, EPS[hidden])
    y2 = cake_rmsnorm_det(x2_leaf, w2, EPS[hidden])
    assert tuple(y3.shape) == (rows, heads, hidden)
    assert torch.equal(y3.reshape(-1, hidden), y2)
    y3.backward(g3)
    y2.backward(g3.reshape(-1, hidden))
    assert tuple(x3_leaf.grad.shape) == (rows, heads, hidden)
    assert torch.equal(x3_leaf.grad.reshape(-1, hidden), x2_leaf.grad)
    assert torch.equal(w3.grad, w2.grad)


def test_the_issues_hidden_256_view_collapses_before_the_route_lookup() -> None:
    """The ``[T, 24, 256]`` view with strides ``(12288, 512, 1)`` of a ``[T, 24, 512]`` buffer collapses to
    ``24 T`` rows of 256: the host reaches the route lookup, which answers for hidden 256 with the width-256
    route (a device without it gets the ValueError naming hidden=256, not the shape)."""

    rows = 16
    buffer = torch.zeros(rows, 24, 512, dtype=torch.bfloat16, device=_device())
    x = buffer[..., :256]
    assert tuple(x.stride()) == (12288, 512, 1)
    w = torch.ones(256, dtype=torch.bfloat16, device=_device())
    if is_cake_rmsnorm_det_train_supported(_device(), 256):
        y, rstd = cake_rmsnorm_det_train_forward(x, w, 1e-6)
        assert tuple(y.shape) == (rows, 24, 256) and tuple(rstd.shape) == (rows, 24)
        dx, dw = cake_rmsnorm_det_train_backward(x, x, w, rstd)
        assert tuple(dx.shape) == (rows, 24, 256) and tuple(dw.shape) == (256,)
        return
    with pytest.raises(
        ValueError, match=r"no exported Cake RMSNorm training forward for hidden=256 "
    ):
        cake_rmsnorm_det_train_forward(x, w, 1e-6)
    rstd = torch.ones(rows, 24, dtype=torch.float32, device=_device())
    with pytest.raises(
        ValueError, match=r"no exported Cake RMSNorm training backward for hidden=256 "
    ):
        cake_rmsnorm_det_train_backward(x, x, w, rstd)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES[:1])
def test_nd_inputs_outside_the_collapse_rule_are_refused(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    rows, heads = 8, 4
    eps = EPS[hidden]
    w = _weight(hidden, _generator(hidden, 5))
    # Heads-major storage viewed as [T, heads, H]: stride(0) != stride(1) * heads.
    x = torch.zeros(
        heads, rows, hidden, dtype=torch.bfloat16, device=_device()
    ).transpose(0, 1)
    assert tuple(x.stride()) == (hidden, rows * hidden, 1)
    with pytest.raises(ValueError, match=r"cannot be viewed as \[32, "):
        cake_rmsnorm_det_train_forward(x, w, eps)
    rstd = torch.ones(rows, heads, dtype=torch.float32, device=_device())
    with pytest.raises(ValueError, match=r"cannot be viewed as \[32, "):
        cake_rmsnorm_det_train_backward(x.contiguous(), x, w, rstd)
    # A feature dimension that is not contiguous.
    x = torch.zeros(rows, heads, hidden, 2, dtype=torch.bfloat16, device=_device())[
        ..., 0
    ]
    with pytest.raises(ValueError, match="cannot be viewed as"):
        cake_rmsnorm_det_train_forward(x, w, eps)
    # One dimension is not enough.
    with pytest.raises(ValueError, match=r"must be a \[T, H\] tensor"):
        cake_rmsnorm_det_train_forward(
            torch.zeros(hidden, dtype=torch.bfloat16, device=_device()), w, eps
        )
    # Companions take x's N-D shape.
    x3 = torch.zeros(rows, heads, hidden, dtype=torch.bfloat16, device=_device())
    with pytest.raises(ValueError, match="residual must have shape"):
        cake_rmsnorm_det_train_forward(
            x3, w, eps, residual=x3.reshape(rows * heads, hidden)
        )
    with pytest.raises(ValueError, match="g must have shape"):
        cake_rmsnorm_det_train_backward(x3.reshape(rows * heads, hidden), x3, w, rstd)
    # Dimensions of size one are free: [T, 1, H] slices of a wider buffer collapse to T rows.
    x = torch.zeros(rows, 3, hidden, dtype=torch.bfloat16, device=_device())[:, 1:2, :]
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    assert tuple(y.shape) == (rows, 1, hidden) and tuple(rstd.shape) == (rows, 1)


# --------------------------------------------------------------------------- FP32 effective weights


def _fp32_route(hidden: int, kind: str, residual: bool) -> bool:
    return loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype="f32",
    )


def _skip_unless_fp32_routes(hidden: int, residual: bool) -> None:
    _skip_unless_route(hidden, residual)
    if not (
        _fp32_route(hidden, "fwd", residual) and _fp32_route(hidden, "bwd", residual)
    ):
        pytest.skip(
            f"hidden={hidden} residual={residual}: no FP32-weight route on this device"
        )


def _fp32_case(
    hidden: int,
    rows: int,
    residual: bool,
    w32: torch.Tensor,
    generator: torch.Generator,
) -> tuple[
    tuple[torch.Tensor, ...], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]
]:
    """One forward + backward with the FP32 weight ``w32``: ``(forward outputs, dx, dw, inputs)``."""

    eps = EPS[hidden]
    x = _rows(rows, hidden, generator=generator)
    u = _rows(rows, hidden, generator=generator) if residual else None
    g = _rows(rows, hidden, generator=generator)
    g_h = _rows(rows, hidden, generator=generator) if residual else None
    outputs = cake_rmsnorm_det_train_forward(x, w32, eps, residual=u)
    normalized = outputs[2] if residual else x
    dx, dw = cake_rmsnorm_det_train_backward(
        g, normalized, w32, outputs[1], g_residual=g_h
    )
    return (
        outputs,
        dx,
        dw,
        {"x": x, "u": u, "g": g, "g_h": g_h, "normalized": normalized},
    )


def _window_rows(hidden: int, residual: bool) -> tuple[int, ...]:
    """The row counts that visit every window of the route's forward and backward programs:
    both sides of every window edge, the smallest counts and the training shape."""

    arch = loader.arch_for_capability(_capability())
    rows = {1, 2, 129, 16231}
    for kind in ("fwd", "bwd"):
        for above, at_most, _library in loader.program_windows(
            arch, hidden, kind, residual
        ):
            rows.add(above + 1)
            if at_most is not None:
                rows.update((at_most, at_most + 1))
    return tuple(sorted(rows))


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_fp32_weight_equal_to_the_bf16_weight_matches_it_bitwise(
    hidden: int, residual: bool
) -> None:
    """``w32 = float(w16)`` gives bitwise the BF16 path's y, rstd, h_new, dx and dw in every
    window of the route: the FP32-weight programs are the BF16 programs with the weight loaded
    at FP32, selected over the same windows."""

    _skip_unless_fp32_routes(hidden, residual)
    for rows in _window_rows(hidden, residual):
        generator = _generator(hidden, rows, residual, 83)
        w16 = _weight(hidden, generator)
        seed = generator.get_state()
        outputs16, dx16, dw16, _ = _fp32_case(hidden, rows, residual, w16, generator)
        generator.set_state(seed)
        outputs32, dx32, dw32, _ = _fp32_case(
            hidden, rows, residual, w16.float(), generator
        )
        assert len(outputs16) == len(outputs32) == (3 if residual else 2)
        for name, a, b in zip(
            ("y", "rstd", "h_new")[: len(outputs16)], outputs16, outputs32, strict=True
        ):
            assert torch.equal(a, b), (
                f"hidden={hidden} residual={residual} rows={rows}: {name} differs"
            )
        label = f"hidden={hidden} residual={residual} rows={rows}"
        assert torch.equal(dx16, dx32), f"{label}: dx differs"
        assert torch.equal(dw16, dw32), f"{label}: dw differs"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
@pytest.mark.parametrize("rows", (7, 16231))
def test_fp32_weight_is_applied_at_fp32(hidden: int, residual: bool, rows: int) -> None:
    """An effective weight ``1 + delta`` that BF16 cannot represent is applied as given: y and dx match the FP64
    reference computed with the FP32 weight, and differ from the BF16 path given ``bf16(w32)``."""

    _skip_unless_fp32_routes(hidden, residual)
    generator = _generator(hidden, rows, residual, 89)
    w32 = 1.0 + 1e-3 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    assert not torch.equal(w32.to(torch.bfloat16).float(), w32)
    seed = generator.get_state()
    outputs, dx, dw, inputs = _fp32_case(hidden, rows, residual, w32, generator)
    eps = EPS[hidden]
    y_ref, rstd_ref, _h = _reference_forward(
        inputs["x"], w32, eps, residual=inputs["u"]
    )
    dx_ref, dw_ref = _reference_backward(
        inputs["g"], inputs["normalized"], w32, eps, g_h=inputs["g_h"]
    )
    label = f"hidden={hidden} residual={residual} rows={rows} fp32 w"
    _assert_bf16_close(outputs[0], y_ref, f"y {label}")
    assert _rel_l2(outputs[1], rstd_ref) <= RSTD_REL_L2, f"rstd {label}"
    _assert_bf16_close(dx, dx_ref, f"dx {label}")
    _assert_dw(dw, dw_ref, inputs["g"], inputs["normalized"], outputs[1], f"dw {label}")
    generator.set_state(seed)
    outputs16, dx16, _dw16, _ = _fp32_case(
        hidden, rows, residual, w32.to(torch.bfloat16), generator
    )
    assert torch.equal(outputs16[1], outputs[1]), f"{label}: rstd does not depend on w"
    assert not torch.equal(outputs16[0], outputs[0]), (
        f"{label}: y ignores the FP32 weight"
    )
    assert not torch.equal(dx16, dx), f"{label}: dx ignores the FP32 weight"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_fp32_weight_layouts_outside_the_rule_are_materialized(hidden: int) -> None:
    """An FP32 weight that is not 32-byte aligned, or not contiguous, is copied once: the same results as the
    aligned contiguous weight (and the autograd path returns ``dw`` in FP32 for an FP32 weight)."""

    _skip_unless_fp32_routes(hidden, False)
    rows = 129
    generator = _generator(hidden, rows, 97)
    w32 = 1.0 + 1e-3 * torch.randn(
        hidden, dtype=torch.float32, device=_device(), generator=generator
    )
    seed = generator.get_state()
    outputs, dx, dw, _ = _fp32_case(hidden, rows, False, w32, generator)
    storage = torch.empty(hidden + 8, dtype=torch.float32, device=_device())
    half_aligned = storage[4 : 4 + hidden]
    half_aligned.copy_(w32)
    assert half_aligned.data_ptr() % loader.FP32_WEIGHT_ALIGNMENT_BYTES == 16
    strided = torch.empty(2 * hidden, dtype=torch.float32, device=_device())[::2]
    strided.copy_(w32)
    for layout in (half_aligned, strided):
        generator.set_state(seed)
        outputs_l, dx_l, dw_l, _ = _fp32_case(hidden, rows, False, layout, generator)
        assert all(torch.equal(a, b) for a, b in zip(outputs, outputs_l, strict=True))
        assert torch.equal(dx, dx_l) and torch.equal(dw, dw_l)
    x = _rows(rows, hidden, generator=generator).requires_grad_(True)
    w_leaf = w32.clone().requires_grad_(True)
    y = cake_rmsnorm_det(x, w_leaf, EPS[hidden])
    y.backward(torch.ones_like(y))
    assert w_leaf.grad.dtype == torch.float32 and x.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_zero_rows(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    generator = _generator(hidden, 0)
    x = torch.empty(0, hidden, dtype=torch.bfloat16, device=_device())
    w = _weight(hidden, generator)
    y, rstd = cake_rmsnorm_det_train_forward(x, w, EPS[hidden])
    assert tuple(y.shape) == (0, hidden) and tuple(rstd.shape) == (0,)
    dx, dw = cake_rmsnorm_det_train_backward(x, x, w, rstd)
    assert tuple(dx.shape) == (0, hidden)
    assert torch.equal(dw, torch.zeros_like(dw))


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("rows", (129, 16231))
def test_deterministic_dw_is_bitwise_reproducible(hidden: int, rows: int) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = _generator(hidden, rows, 11)
    x = _rows(rows, hidden, generator=generator)
    w = _weight(hidden, generator)
    g = _rows(rows, hidden, generator=generator)
    _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    workspace = cake_rmsnorm_det_train_backward_workspace(rows, hidden, _device())
    results = []
    for _repeat in range(3):
        dx = torch.full(
            (rows, hidden), float("nan"), dtype=torch.bfloat16, device=_device()
        )
        dw = torch.full((hidden,), float("nan"), dtype=torch.float32, device=_device())
        # Poison the first partial row (after the counters); the kernels must
        # rewrite every chunk they read.
        begin = loader.COUNTERS_BYTES
        workspace[begin : begin + 4 * hidden].view(torch.float32).fill_(float("nan"))
        cake_rmsnorm_det_train_backward(
            g, x, w, rstd, workspace=workspace, dx=dx, dw=dw
        )
        torch.cuda.synchronize()
        results.append((dx.clone(), dw.clone()))
    fresh = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    for dx, dw in results[1:]:
        assert torch.equal(dx, results[0][0])
        assert torch.equal(dw, results[0][1])
    assert torch.equal(fresh[1], results[0][1]), "a fresh workspace changed dw"
    assert torch.isfinite(results[0][1]).all()


def _taper_rows(record: dict, sms: int, limit: int) -> list[int]:
    """Row counts of the record's window where its taper cuts another chunk
    table than the module constants would: the first, a middle and the last
    within ``limit`` (none for a program at the default taper)."""

    rule = record["grid_rule"]
    taper = loader.chunk_taper(rule)
    if taper == (loader.TAIL_DIV, loader.MAX_HALVING):
        return []
    select = record["select_rule"]
    above = 0 if select is None else int(select["above"])
    at_most = None if select is None else select["at_most"]
    last = limit if at_most is None else min(limit, int(at_most))
    slots, rpi = int(rule["ctas_per_sm"]) * sms, int(rule["rows_per_issue"])
    differing = [
        rows
        for rows in range(above + 1, last + 1)
        if loader.n_chunks(rows, slots, rpi, *taper)
        != loader.n_chunks(rows, slots, rpi)
    ]
    if not differing:
        return []
    return sorted({differing[0], differing[len(differing) // 2], differing[-1]})


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
def test_backward_reuses_a_poisoned_workspace_across_row_counts(
    hidden: int, residual: bool
) -> None:
    """One caller-owned workspace serves every token count of a training step.
    Its partial rows are poisoned with NaN before each call, so a host that cuts
    another chunk table than the kernel -- a chunk count the main kernel wrote
    no partial rows for -- surfaces in ``dw`` instead of hiding behind a fresh
    zeroed buffer.  The row counts are the route's window edges and, for every
    program built with a taper of its own, counts where its table differs from
    the module constants'; they run largest first, then back up from the
    smallest."""

    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(), hidden=hidden, kind="bwd", residual=True
    ):
        pytest.skip(f"hidden={hidden} has no exported residual backward route")
    arch = loader.arch_for_capability(_capability())
    sms = loader.sm_count(0)
    limit = 32768
    rows_set = {
        rows for rows in _switch_rows(arch, hidden, "bwd", residual) if rows <= limit
    }
    for name in loader.route_modules(arch, hidden, "bwd", residual):
        record = loader.MODULES[name]
        if record.get("workspace_rule") is not None:
            rows_set.update(_taper_rows(record, sms, limit))
    descending = sorted(rows_set, reverse=True)
    order = descending + descending[-2::-1]
    eps = EPS[hidden]
    workspace = cake_rmsnorm_det_train_backward_workspace(
        max(rows_set), hidden, _device(), residual=residual
    )
    for rows in order:
        generator = _generator(hidden, rows, 29, residual)
        x = _rows(rows, hidden, generator=generator)
        w = _weight(hidden, generator)
        g = _rows(rows, hidden, generator=generator)
        if residual:
            u = _rows(rows, hidden, generator=generator)
            g_h = _rows(rows, hidden, generator=generator)
            _y, rstd, normalized = cake_rmsnorm_det_train_forward(x, w, eps, residual=u)
        else:
            g_h = None
            _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
            normalized = x
        workspace[loader.COUNTERS_BYTES :].view(torch.float32).fill_(float("nan"))
        dx, dw = cake_rmsnorm_det_train_backward(
            g, normalized, w, rstd, g_residual=g_h, workspace=workspace
        )
        dx_ref, dw_ref = _reference_backward(g, normalized, w, eps, g_h=g_h)
        label = f"hidden={hidden} rows={rows} residual={residual} (reused workspace)"
        _assert_bf16_close(dx, dx_ref, f"dx {label}")
        assert torch.isfinite(dw).all(), f"dw {label} has non-finite values"
        error = _rel_l2(dw, dw_ref)
        assert error <= DW_REL_L2, f"dw {label}: rel-L2 {error:.3e} > {DW_REL_L2:.0e}"


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_changing_rows_in_one_process_does_not_rebuild(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = _generator(hidden, 17)
    w = _weight(hidden, generator)
    _check_case(hidden, 64)
    loaded_before = loader.load.cache_info().misses
    for rows in (1, 65, 16172, 7, 16231, 255):
        x = _rows(rows, hidden, generator=generator)
        g = _rows(rows, hidden, generator=generator)
        y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
        dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
        y_ref, rstd_ref, _ = _reference_forward(x, w, eps)
        dx_ref, dw_ref = _reference_backward(g, x, w, eps)
        _assert_bf16_close(y, y_ref, f"y rows={rows}")
        _assert_bf16_close(dx, dx_ref, f"dx rows={rows}")
        _assert_dw(dw, dw_ref, g, x, rstd, f"dw rows={rows}")
    assert loader.load.cache_info().misses == loaded_before, (
        "a changed token count rebuilt a module"
    )


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("view", ("expand", "slice"))
def test_backward_accepts_gradient_views(hidden: int, view: str) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    rows = 129
    generator = _generator(hidden, rows, 23)
    x = _rows(rows, hidden, generator=generator)
    w = _weight(hidden, generator)
    if view == "expand":
        g = _rows(1, hidden, generator=generator).expand(rows, hidden)
    else:
        g = _rows(rows, hidden, stride_pad="2H", generator=generator)
    _y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    dx_ref, dw_ref = _reference_backward(g, x, w, eps)
    _assert_bf16_close(dx, dx_ref, f"dx view={view}")
    _assert_dw(dw, dw_ref, g, x, rstd, f"dw view={view}")


def _autograd_reference(x, w, eps, residual=None, grad_y=None, grad_h=None):
    """FP32 eager autograd of the same math (BF16 residual boundary kept)."""

    xl = x.detach().float().requires_grad_(True)
    wl = w.detach().float().requires_grad_(True)
    ul = (
        residual.detach().float().requires_grad_(True) if residual is not None else None
    )
    if ul is not None:
        h_new = (xl + ul).to(torch.bfloat16).float()
    else:
        h_new = xl
    rstd = torch.rsqrt(h_new.pow(2).mean(-1, keepdim=True) + eps)
    y = ((h_new * rstd) * wl).to(torch.bfloat16)
    outputs, grads = [y], [grad_y.float().to(torch.bfloat16)]
    if ul is not None and grad_h is not None:
        outputs.append(h_new)
        grads.append(grad_h.float())
    torch.autograd.backward(outputs, grads)
    return xl.grad, wl.grad, (ul.grad if ul is not None else None)


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
@pytest.mark.parametrize("residual", (False, True))
@pytest.mark.parametrize("rows", (7, 16231))
def test_autograd_end_to_end(hidden: int, residual: bool, rows: int) -> None:
    _skip_unless_hidden(hidden)
    if residual and not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="fwd",
        residual=True,
    ):
        pytest.skip(f"hidden={hidden} has no exported residual route on this device")
    eps = EPS[hidden]
    generator = _generator(hidden, rows, 31, residual)
    x0 = _rows(rows, hidden, generator=generator)
    w0 = _weight(hidden, generator)
    u0 = _rows(rows, hidden, generator=generator) if residual else None
    grad_y = _rows(rows, hidden, generator=generator)
    grad_h = _rows(rows, hidden, generator=generator) if residual else None

    x = x0.detach().clone().requires_grad_(True)
    w = w0.detach().clone().requires_grad_(True)
    u = u0.detach().clone().requires_grad_(True) if residual else None
    outputs = cake_rmsnorm_det(x, w, eps, residual=u)
    if residual:
        y, h_new = outputs
        torch.autograd.backward([y, h_new], [grad_y, grad_h])
    else:
        y = outputs
        y.backward(grad_y)

    dx_ref, dw_ref, du_ref = _autograd_reference(x0, w0, eps, u0, grad_y, grad_h)
    y_ref, _rstd, _h = _reference_forward(x0, w0, eps, residual=u0)
    _assert_bf16_close(y.detach(), y_ref, "autograd y")
    assert x.grad.dtype == torch.bfloat16
    _assert_bf16_close(x.grad, dx_ref.double(), "autograd dx")
    assert w.grad.dtype == w0.dtype
    assert _rel_l2(w.grad, dw_ref) <= 1e-2, "autograd dw (rounded to the weight dtype)"
    if residual:
        assert torch.equal(x.grad, u.grad), (
            "both residual-add inputs receive the same gradient"
        )
        _assert_bf16_close(u.grad, du_ref.double(), "autograd du")


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_workspace_bytes_and_layout(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    small = cake_rmsnorm_det_train_backward_workspace_bytes(1, hidden, _device())
    middle = cake_rmsnorm_det_train_backward_workspace_bytes(129, hidden, _device())
    large = cake_rmsnorm_det_train_backward_workspace_bytes(65536, hidden, _device())
    assert small > 0 and small % 256 == 0 and large % 256 == 0
    assert small <= middle <= large
    workspace = cake_rmsnorm_det_train_backward_workspace(65536, hidden, _device())
    assert workspace.dtype == torch.uint8 and workspace.numel() == large
    assert torch.count_nonzero(workspace) == 0
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward_workspace_bytes(-1, hidden, _device())


@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_argument_validation(hidden: int) -> None:
    _skip_unless_hidden(hidden)
    eps = EPS[hidden]
    generator = _generator(hidden, 5)
    x = _rows(8, hidden, generator=generator)
    w = _weight(hidden, generator)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x.float(), w, eps)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x, w.double(), eps)
    if not _fp32_route(hidden, "fwd", False):
        with pytest.raises(ValueError, match="weight dtypes exported for this route"):
            cake_rmsnorm_det_train_forward(x, w.float(), eps)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(x[:, 1:], w[1:], eps)
    # An odd row stride is accepted (materialized once), not rejected.
    odd = _rows(8, hidden, stride_pad=8, generator=generator)
    odd_view = odd.as_strided((8, hidden), (hidden + 4, 1))
    y_odd, _rstd_odd = cake_rmsnorm_det_train_forward(odd_view, w, eps)
    y_odd_ref, _, _ = _reference_forward(odd_view, w, eps)
    _assert_bf16_close(y_odd, y_odd_ref, "y odd row stride")
    y, rstd = cake_rmsnorm_det_train_forward(x, w, eps)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward(y, x, w, rstd, deterministic=False)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward(y, x, w, rstd.double())
    tiny = torch.zeros(256, dtype=torch.uint8, device=_device())
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_backward(y, x, w, rstd, workspace=tiny)
    cpu = torch.zeros(8, hidden, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cake_rmsnorm_det_train_forward(cpu, w.cpu(), eps)


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    # build.json names the package's FlashInfer import path; a vendored copy ships the file unchanged under its own name.
    assert manifest["package"] == "flashinfer.cake_rmsnorm_det_train_backend"
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            assert "--use_fast_math" not in entry["nvcc_flags"]
            assert "--prec-div=true" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


# The per-element dw bound of the H = 5120 contract at the largest row counts of its long-chunk window: every element
# of dw within atol = rtol = 1e-4 of the FP64 sum of the same products (g, x and the FP32 rstd the forward returned),
# where an FP32 chunk sum over many rows accumulates the most rounding.  A few seeds per case; BF16 and FP32 weight.
DW_ELEMENT_TOL = 1e-4
LARGE_ROW_DW_ROWS = (40960, 65536)


def _dw_fp64(g: torch.Tensor, x: torch.Tensor, rstd: torch.Tensor) -> torch.Tensor:
    dw = torch.zeros(x.shape[-1], dtype=torch.float64, device=x.device)
    for start in range(0, x.shape[0], 8192):
        stop = min(start + 8192, x.shape[0])
        dw += (
            g[start:stop].double()
            * x[start:stop].double()
            * rstd[start:stop, None].double()
        ).sum(0)
    return dw


@pytest.mark.parametrize("rows", LARGE_ROW_DW_ROWS)
@pytest.mark.parametrize("fp32_weight", (False, True))
@pytest.mark.parametrize("seed", (0, 1, 2))
def test_large_row_dw_meets_the_per_element_bound(
    rows: int, fp32_weight: bool, seed: int
) -> None:
    hidden = 5120
    if fp32_weight:
        _skip_unless_fp32_routes(hidden, False)
    else:
        _skip_unless_hidden(hidden)
    generator = _generator(hidden, rows, fp32_weight, seed, 97)
    x = _rows(rows, hidden, generator=generator)
    g = _rows(rows, hidden, generator=generator)
    if fp32_weight:
        w = 1.0 + 0.1 * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
    else:
        w = _weight(hidden, generator)
    _y, rstd = cake_rmsnorm_det_train_forward(x, w, EPS[hidden])
    _dx, dw = cake_rmsnorm_det_train_backward(g, x, w, rstd)
    reference = _dw_fp64(g, x, rstd)
    ratio = (dw.double() - reference).abs() / (
        DW_ELEMENT_TOL + DW_ELEMENT_TOL * reference.abs()
    )
    worst = int(ratio.argmax())
    assert float(ratio.max()) <= 1.0, (
        f"rows={rows} fp32_weight={fp32_weight} seed={seed}: dw[{worst}] = "
        f"{float(dw[worst])!r} vs FP64 {float(reference[worst])!r} "
        f"({float(ratio.max()):.3f} x the bound, {int((ratio > 1).sum())} elements over)"
    )


def test_dw_not_worse_than_eager_fp32_over_the_matrix() -> None:
    """Aggregate dw gate: the kernel's dw error is not worse than the eager FP32 path over the checked matrix.

    Per case both errors are FP32 rounding noise; the honest comparison is the median ratio
    kernel / eager over every dw case this process checked (all rows, strides, scales, residual on/off).
    Runs last; skipped when the module's dw cases did not run in this process (e.g. ``-k`` selection).
    """

    if len(_DW_ERRORS) < 8:
        pytest.skip("needs the dw cases of this module in the same process")
    ratios = [
        (k / e) if e > 0 else (0.0 if k == 0 else float("inf"))
        for _label, k, e in _DW_ERRORS
    ]
    median = statistics.median(ratios)
    worst = max(_DW_ERRORS, key=lambda item: item[1])
    assert median <= 1.0, (
        f"median dw error ratio kernel/eager-FP32 {median:.3f} > 1 over {len(ratios)} cases "
        f"(worst kernel rel-L2 {worst[1]:.3e} at {worst[0]})"
    )
