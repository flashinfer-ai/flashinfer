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

Token-count invariance of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (``deterministic=False``; issue #7045 items 6-7).

Within one program window a row's outputs do not depend on the token count:
for every window of every route of the device -- each width, the residual route,
both weight dtypes -- the call on the first ``T`` rows of one input, for ``T`` at
both ends and the middle of the window, gives the same ``dx`` bits on those rows
as the call on the window's largest ``T`` (backward; the single kernel and the
deterministic pair alike: ``dx`` has no cross-row reduction), and the same ``y``
/ ``rstd`` bits (forward).  Where a window claims the deterministic package's
``dx`` -- a deterministic-pair window running the program the deterministic
package selects for that ``T``, or a single kernel listed in the backend suite's
``DX_BITWISE_WITH`` -- and that package serves the call, ``dx`` is bitwise its
``dx`` at every ``T`` measured in the window.

At every window edge ``E`` the calls of ``E`` and ``E + 1`` rows are compared on
the first ``E`` rows and the edge is classified: ``same`` (bitwise), ``ulp``
(backward: every ``dx`` difference within one BF16 ulp; forward: ``y`` within
one BF16 ulp and ``rstd`` within two FP32 ulps, the class the package notes
document) or larger (backward ``dx``; forward ``y<k>/rstd<m>``, the largest
differences in ulps).  The classification of ``cake_windows.EDGES`` must hold
where it is measured (``same`` stays bitwise, ``ulp`` within its ulps); every
edge it lists as ``unmeasured`` is printed with its measured class
(``EDGES_MEASURED``) for the exporter to fill.
"""

from __future__ import annotations

import ast
import functools
from pathlib import Path
from typing import Optional

import pytest
import torch

from tests.test_helpers import cake_rmsnorm_train_reference as ref
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
windows_table = cake_package("cake_rmsnorm_train", "cake_windows")
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (256, 512, 1536, 2048, 4096, 5120, 6144)
W_TOKENS = ("bf16", "f32")
# The largest token count measured in an unbounded window.
MAX_TOKENS = 65536
SEED = 7046


def _det_package():
    """The deterministic package, or ``None`` (a vendored copy without it)."""
    try:
        return cake_package("cake_rmsnorm_det_train")
    except Exception:  # noqa: BLE001 - the det sibling is optional here
        return None


det = _det_package()


def _dx_bitwise_with() -> dict[str, str]:
    """The backend suite's ``DX_BITWISE_WITH`` table (single-kernel library ->
    the det module whose ``dx`` bits it claims), read from its source."""
    source = Path(__file__).with_name("test_cake_rmsnorm_train.py")
    if not source.exists():
        return {}
    for node in ast.parse(source.read_text()).body:
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "DX_BITWISE_WITH"
        ):
            return dict(ast.literal_eval(node.value))
    return {}


DX_BITWISE_WITH = _dx_bitwise_with()


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


def _arch() -> str:
    return loader.arch_for_capability(_capability())


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train(deterministic=False) needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
        if det is not None:
            for spec in det.cake_jit.specs_for_arch(_arch()):
                spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


_LAUNCHED: list[str] = []


@pytest.fixture(autouse=True)
def record_launches(monkeypatch: pytest.MonkeyPatch):
    run_module = loader.run_module

    def recording(name, values, *, rows, device_index):
        _LAUNCHED.append(name)
        return run_module(name, values, rows=rows, device_index=device_index)

    monkeypatch.setattr(loader, "run_module", recording)
    _LAUNCHED.clear()
    yield
    _LAUNCHED.clear()


def _route_params():
    for hidden in HIDDEN_SIZES:
        for w in W_TOKENS:
            yield pytest.param(hidden, False, w, id=f"h{hidden}-{w}")
    for w in W_TOKENS:
        yield pytest.param(6144, True, w, id=f"res-h6144-{w}")


def _skip_unless_route(hidden: int, residual: bool, w: str, kind: str) -> None:
    if not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype=w,
    ):
        pytest.skip(
            f"hidden={hidden} residual={residual} w={w} has no {kind} route here"
        )


def _window_tokens(above: int, at_most: Optional[int]) -> tuple[int, int, int]:
    """``(first, middle, last)`` token counts measured in the window ``(above, at_most]``."""
    first = above + 1
    if at_most is None:
        last = (
            first + 4096
            if first >= MAX_TOKENS
            else min(max(2 * first, first + 4096), MAX_TOKENS)
        )
    else:
        last = at_most
    return first, (first + last) // 2, last


def _inputs(hidden: int, residual: bool, w: str, rows: int, key: int):
    generator = torch.Generator(device=_device()).manual_seed(
        (SEED * 1_000_003 + hidden * 13 + int(residual) * 7 + (w == "f32") + key)
        % (2**31 - 1)
    )

    def randn(*shape):
        return torch.randn(
            *shape, dtype=torch.float32, device=_device(), generator=generator
        ).to(torch.bfloat16)

    x = randn(rows, hidden)
    u = randn(rows, hidden) if residual else None
    g = randn(rows, hidden)
    g_h = randn(rows, hidden) if residual else None
    weight = (
        1.0
        + 0.1
        * torch.randn(
            hidden, dtype=torch.float32, device=_device(), generator=generator
        )
    ).to(ref.W_DTYPES[w])
    return x, u, g, g_h, weight


def _edges(hidden: int, residual: bool, w: str, kind: str) -> dict[int, str]:
    """``cake_windows.EDGES`` of the route on the device's architecture: edge -> class."""
    key = loader.route_key(hidden, kind, residual, w)
    return dict(windows_table.EDGES.get(_arch(), {}).get(key, ()))


# The ``ulp`` edge class of a forward edge (the package notes match it):
# ``rstd`` moves by at most two FP32 ulps (a different reduction order
# on each side; measured on up to ~30 % of the common rows) and ``y`` by
# at most one BF16 ulp.  Each edge reports its largest moves and the fraction of
# the common rows that moved (``EDGE_FWD ... rows_moved=``).
FWD_EDGE_Y_ULPS = 1
FWD_EDGE_RSTD_ULPS = 2


def _classify(differing: int, ulps: int, word: str) -> str:
    """A backward edge: ``same`` (bitwise), ``ulp`` (one BF16 ulp) or ``word``."""
    if differing == 0:
        return "same"
    return "ulp" if ulps <= 1 else word


def _forward_class(y_diff: int, y_ulps: int, r_diff: int, r_ulps: int) -> str:
    """A forward edge: ``same``, ``ulp`` (within the documented class) or
    ``y<k>/rstd<m>`` (the largest differences in BF16 / FP32 ulps, beyond it)."""
    if y_diff == 0 and r_diff == 0:
        return "same"
    if y_ulps <= FWD_EDGE_Y_ULPS and r_ulps <= FWD_EDGE_RSTD_ULPS:
        return "ulp"
    return f"y{y_ulps}/rstd{r_ulps}"


def _consistent(listed: Optional[str], measured: str) -> bool:
    """A listed class holds when the measurement is within it: ``same`` stays
    bitwise, ``ulp`` stays within one ulp; ``dx`` and unmeasured edges admit any."""
    if listed in (None, "unmeasured", "dx"):
        return True
    if listed == "same":
        return measured == "same"
    return measured in ("same", "ulp")


# --------------------------------------------------------------------------- backward


class _Backward:
    """``dx`` of the first ``T`` rows of one route's inputs, cached per ``T``."""

    def __init__(self, hidden: int, residual: bool, w: str, rows: int):
        self.hidden, self.residual, self.w = hidden, residual, w
        x, u, g, g_h, weight = _inputs(hidden, residual, w, rows, key=1)
        outputs = forward(x, weight, ref.EPS[hidden], residual=u)
        self.normalized = outputs[2] if residual else x
        self.rstd, self.g, self.g_h, self.weight = outputs[1], g, g_h, weight
        self.cache: dict[int, torch.Tensor] = {}
        self.records: dict[int, list[str]] = {}

    def dx(self, rows: int) -> torch.Tensor:
        if rows not in self.cache:
            start = len(_LAUNCHED)
            dx, _dw = backward(
                self.g[:rows],
                self.normalized[:rows],
                self.weight,
                self.rstd[:rows],
                g_residual=None if self.g_h is None else self.g_h[:rows],
            )
            self.records[rows] = list(_LAUNCHED[start:])
            self.cache[rows] = dx
        return self.cache[rows]

    def det_dx(self, rows: int) -> torch.Tensor:
        dx, _dw = det.cake_rmsnorm_det_train_backward(
            self.g[:rows],
            self.normalized[:rows],
            self.weight,
            self.rstd[:rows],
            g_residual=None if self.g_h is None else self.g_h[:rows],
        )
        return dx


def _program_id(name: str) -> str:
    """The program part of a module name (the hex digits after the package
    prefix, without a ``__<above>_<at_most>`` window suffix)."""
    return name.split("__", 1)[0].rsplit("_", 1)[1]


def _claims_det_dx(records: list[str], hidden: int, residual: bool, w: str, rows: int):
    """``(claimed, det main module)``: the window runs the det program the det
    package selects for ``rows`` (a det pair of the same program id), or a single
    kernel whose library ``DX_BITWISE_WITH`` maps to that det module; ``(False,
    None)`` when the det package does not serve the call (no det or FP32 weight)."""
    if det is None or w != "bf16":
        return False, None
    arch = det.cake_jit.arch_for_capability(_capability())
    if det.cake_jit.route_key(hidden, "bwd", residual) not in det.cake_jit.ROUTES.get(
        arch, {}
    ):
        return False, None
    det_records = list(
        det.cake_jit.selected_modules(arch, hidden, "bwd", residual, rows=rows)
    )
    det_main = next(
        n for n in det_records if det.cake_jit.MODULES[n]["role"] != "bwd_reduce"
    )
    main = next(n for n in records if loader.MODULES[n]["role"] != "bwd_reduce")
    claim = DX_BITWISE_WITH.get(loader.MODULES[main]["library"])
    if claim is not None:
        return _program_id(claim) == _program_id(det_main), det_main
    return _program_id(main) == _program_id(det_main), det_main


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_route_params()))
def test_backward_dx_is_token_count_invariant_within_each_window(
    hidden: int, residual: bool, w: str
) -> None:
    """Per backward window: ``dx`` of the first ``T`` rows is the same bits for
    ``T`` at both ends and the middle (and the window's records ran each time);
    a window that claims the det ``dx`` matches it bitwise at every measured
    ``T`` where the det package serves the call."""
    _skip_unless_route(hidden, residual, w, "bwd")
    route_windows = ref.windows(loader, _arch(), hidden, "bwd", residual, w)
    largest = max(
        _window_tokens(above, at_most)[2] for above, at_most, _ in route_windows
    )
    calls = _Backward(hidden, residual, w, largest)
    lines, claims = [], 0
    for above, at_most, library in route_windows:
        first, middle, last = _window_tokens(above, at_most)
        expected = ref.window_records(loader, _arch(), hidden, "bwd", residual, w, last)
        form = ref.form(loader, expected)
        reference = calls.dx(last)
        assert calls.records[last] == expected, (calls.records[last], expected)
        for rows in sorted({first, middle}):
            dx = calls.dx(rows)
            assert calls.records[rows] == expected, (
                f"T={rows} in ({above}, {at_most}] launched {calls.records[rows]}, window {expected}"
            )
            differing, ulps = ref.bf16_ulp_differences(dx, reference[:rows])
            assert differing == 0, (
                f"h={hidden} res={residual} w={w} window ({above}, {at_most}] {form}: dx of "
                f"T={rows} differs from T={last} on {differing} elements of the common rows "
                f"(up to {ulps} BF16 ulps)"
            )
        # The claim is judged per T: the det package has its own windows, so the
        # det program it selects may change inside this package's window.
        notes, any_claim = [], False
        for rows in sorted({first, middle, last}):
            claimed, det_main = _claims_det_dx(expected, hidden, residual, w, rows)
            if det_main is None:
                continue
            differing, _ulps = ref.bf16_ulp_differences(
                calls.dx(rows), calls.det_dx(rows)
            )
            if claimed:
                assert differing == 0, (
                    f"h={hidden} res={residual} window ({above}, {at_most}] claims the det dx "
                    f"({det_main}) but T={rows} differs on {differing} elements"
                )
            any_claim = any_claim or claimed
            notes.append(f"{rows}:{'claimed' if claimed else 'unclaimed'}:{differing}")
        claims += int(any_claim)
        det_note = f"det={','.join(notes)}" if notes else "det=n/a"
        lines.append(
            f"XT_DX {_arch()} {loader.route_key(hidden, 'bwd', residual, w)} ({above}, {at_most}] "
            f"{form} {library.rsplit('_', 1)[1][:8]} T={first}/{middle}/{last} bitwise=True {det_note}"
        )
    print("\n" + "\n".join(lines))
    print(
        f"XT_DX_SUMMARY {_arch()} h={hidden} res={residual} w={w} windows={len(route_windows)} "
        f"det_claimed={claims}"
    )


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_route_params()))
def test_backward_window_edges_are_classified(
    hidden: int, residual: bool, w: str
) -> None:
    """Every backward edge ``E``: ``dx`` of ``E`` and ``E + 1`` rows on the first
    ``E`` rows, classified ``same`` / ``ulp`` / ``dx``; the class of
    ``cake_windows.EDGES`` holds where it is measured; the measured class of an
    unmeasured edge is printed (``EDGES_MEASURED``)."""
    _skip_unless_route(hidden, residual, w, "bwd")
    route_windows = ref.windows(loader, _arch(), hidden, "bwd", residual, w)
    edges = [at_most for _above, at_most, _lib in route_windows if at_most is not None]
    if not edges:
        print(f"EDGE_DX {_arch()} h={hidden} res={residual} w={w} edges=0 (one window)")
        return
    calls = _Backward(hidden, residual, w, max(edges) + 1)
    listed = _edges(hidden, residual, w, "bwd")
    key = loader.route_key(hidden, "bwd", residual, w)
    lines, contradictions = [], []
    for edge in edges:
        below, above_ = calls.dx(edge), calls.dx(edge + 1)
        differing, ulps = ref.bf16_ulp_differences(below, above_[:edge])
        measured = _classify(differing, ulps, "dx")
        forms = "/".join(
            ref.form(loader, calls.records[rows]) for rows in (edge, edge + 1)
        )
        entry = listed.get(edge)
        if not _consistent(entry, measured):
            contradictions.append((edge, entry, measured, differing, ulps))
        lines.append(
            f"EDGE_DX {_arch()} {key} edge={edge} forms={forms} listed={entry} "
            f"measured={measured} differing={differing} max_ulps={ulps}"
        )
        if entry in (None, "unmeasured"):
            lines.append(f"EDGES_MEASURED {_arch()} {key} ({edge}, {measured!r})")
    print("\n" + "\n".join(lines))
    assert not contradictions, (
        f"{key}: edges whose measured dx difference exceeds their EDGES class "
        f"(edge, listed, measured, differing, max ulps): {contradictions}"
    )


# --------------------------------------------------------------------------- forward


class _Forward:
    def __init__(self, hidden: int, residual: bool, w: str, rows: int):
        x, u, _g, _g_h, weight = _inputs(hidden, residual, w, rows, key=2)
        self.x, self.u, self.weight, self.eps = x, u, weight, ref.EPS[hidden]
        self.cache: dict[int, tuple[torch.Tensor, ...]] = {}

    def out(self, rows: int) -> tuple[torch.Tensor, ...]:
        if rows not in self.cache:
            self.cache[rows] = tuple(
                forward(
                    self.x[:rows],
                    self.weight,
                    self.eps,
                    residual=None if self.u is None else self.u[:rows],
                )
            )
        return self.cache[rows]


def _forward_difference(a, b, rows: int) -> tuple[int, int, int, int]:
    """``(y differing, y max BF16 ulps, rstd differing, rstd max FP32 ulps)`` on the first ``rows`` rows."""
    y_diff, y_ulps = ref.bf16_ulp_differences(a[0][:rows], b[0][:rows])
    r_diff, r_ulps = ref.fp32_ulp_differences(a[1][:rows], b[1][:rows])
    return y_diff, y_ulps, r_diff, r_ulps


def _rows_moved(a, b, rows: int) -> int:
    """Rows among the first ``rows`` whose ``y`` or ``rstd`` bits differ."""
    y_a, y_b = a[0][:rows].reshape(rows, -1), b[0][:rows].reshape(rows, -1)
    moved = (y_a.view(torch.int16) != y_b.view(torch.int16)).any(dim=1)
    moved |= (
        a[1][:rows]
        .reshape(rows, -1)
        .view(torch.int32)
        .ne(b[1][:rows].reshape(rows, -1).view(torch.int32))
        .any(dim=1)
    )
    return int(moved.sum().item())


@pytest.mark.parametrize(("hidden", "residual", "w"), list(_route_params()))
def test_forward_is_token_count_invariant_within_windows_and_edges_are_classified(
    hidden: int, residual: bool, w: str
) -> None:
    """Per forward window: ``y`` / ``rstd`` (and ``h_new``) of the first ``T``
    rows are the same bits at both ends and the middle; per edge: the class
    (``same`` / ``ulp`` / beyond: ``y<k>/rstd<m>``) holds against
    ``cake_windows.EDGES`` (``ulp``: ``FWD_EDGE_Y_ULPS`` BF16 ulps of ``y``,
    ``FWD_EDGE_RSTD_ULPS`` FP32 ulps of ``rstd``); each
    edge reports its largest moves and the fraction of rows that moved."""
    _skip_unless_route(hidden, residual, w, "fwd")
    route_windows = ref.windows(loader, _arch(), hidden, "fwd", residual, w)
    largest = max(
        _window_tokens(above, at_most)[2] for above, at_most, _ in route_windows
    )
    calls = _Forward(hidden, residual, w, largest + 1)
    key = loader.route_key(hidden, "fwd", residual, w)
    lines = []
    for above, at_most, library in route_windows:
        first, middle, last = _window_tokens(above, at_most)
        reference = calls.out(last)
        for rows in sorted({first, middle}):
            got = calls.out(rows)
            y_diff, y_ulps, r_diff, r_ulps = _forward_difference(got, reference, rows)
            assert y_diff == 0 and r_diff == 0, (
                f"{key} window ({above}, {at_most}]: T={rows} vs T={last}: y differs on "
                f"{y_diff} ({y_ulps} ulps), rstd on {r_diff} ({r_ulps} ulps)"
            )
            if residual:
                assert torch.equal(got[2], reference[2][:rows]), (
                    f"{key}: h_new T={rows}"
                )
        lines.append(
            f"XT_FWD {_arch()} {key} ({above}, {at_most}] {library.rsplit('_', 1)[1][:8]} "
            f"T={first}/{middle}/{last} bitwise=True"
        )
    listed = _edges(hidden, residual, w, "fwd")
    contradictions = []
    for _above, at_most, _library in route_windows:
        if at_most is None:
            continue
        below, above_edge = calls.out(at_most), calls.out(at_most + 1)
        y_diff, y_ulps, r_diff, r_ulps = _forward_difference(below, above_edge, at_most)
        moved = _rows_moved(below, above_edge, at_most)
        measured = _forward_class(y_diff, y_ulps, r_diff, r_ulps)
        entry = listed.get(at_most)
        if not _consistent(entry, measured):
            contradictions.append(
                (at_most, entry, measured, y_diff, y_ulps, r_diff, r_ulps)
            )
        lines.append(
            f"EDGE_FWD {_arch()} {key} edge={at_most} listed={entry} measured={measured} "
            f"y_differing={y_diff} y_max_ulps={y_ulps} rstd_differing={r_diff} rstd_max_ulps={r_ulps} "
            f"rows_moved={moved}/{at_most} ({100.0 * moved / at_most:.1f} %)"
        )
        if entry in (None, "unmeasured"):
            lines.append(f"EDGES_MEASURED {_arch()} {key} ({at_most}, {measured!r})")
    print("\n" + "\n".join(lines))
    assert not contradictions, (
        f"{key}: edges whose measured difference exceeds their EDGES class "
        f"(edge, listed, measured, y differing, y ulps, rstd differing, rstd ulps): {contradictions}"
    )
