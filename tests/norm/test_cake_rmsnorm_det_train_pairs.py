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

Registry and host-rule gates of the Cake BF16 RMSNorm training backward pair.

Every backward route launches two records per call: the main kernel's record
(its persistent CTAs claim row chunks; it owns the workspace) and the dw
reduce record after it (one CTA per column slab, programmatic dependent
launch).  The registry gates and the chunk-formula pins read the registry and
the host functions alone and need no device; the workspace gates need the
exported device.
"""

from __future__ import annotations

import hashlib
import re

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_det_train")
loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
windows = cake_package("cake_rmsnorm_det_train", "cake_windows")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (6144, 5120, 4096, 2048, 1536, 512, 256)
REDUCE_ROLE = "bwd_reduce"
SHIPPED_SLABS = (16, 32, 64)
# Row counts of the contract: the small counts, both sides of every backward window edge of every route
# (``test_rows_cover_every_backward_window_edge``) and the training shapes.
ROWS = (
    1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 256, 257, 1023, 1024, 1025, 2048, 2049, 3072, 3073, 4096, 4097,
    5120, 5121, 12288, 12289, 14928, 14929, 15566, 15567, 16231, 16263, 16384, 16385, 32768, 32769, 40960, 40961,
    65536, 65537, 81920, 81921, 114688, 114689,
)  # fmt: skip
# The row count standing in for an unbounded window's upper edge.
OPEN_WINDOW_ROWS = 65536
ROUTE_KEY = re.compile(r"^(fwd|bwd)(_w32)?(_residual)?_h(\d+)$")


def _route_parts(route: str) -> tuple[int, str, bool, str]:
    """``(hidden, kind, residual, w_dtype)`` of a route key (``cake_jit.route_key``)."""
    match = ROUTE_KEY.fullmatch(route)
    assert match is not None, route
    w_dtype = "f32" if match.group(2) else "bf16"
    return int(match.group(4)), match.group(1), match.group(3) is not None, w_dtype


def _edges(route_windows) -> tuple[int, ...]:
    return tuple(high for _low, high, _library in route_windows if high is not None)


def _owners() -> dict[str, dict]:
    return {
        name: r
        for name, r in loader.MODULES.items()
        if r.get("workspace_rule") is not None
    }


def _companions() -> dict[str, dict]:
    return {
        name: r for name, r in loader.MODULES.items() if r.get("role") == REDUCE_ROLE
    }


# --------------------------------------------------------------------------- registry gates (no device)


def test_no_record_is_a_cooperative_launch() -> None:
    for name, record in loader.MODULES.items():
        assert record["launch"]["cooperative"] is False, name
        assert "cooperative" not in record["grid_rule"], name


def test_owner_records_claim_chunks_and_own_the_workspace() -> None:
    owners = _owners()
    assert owners
    for name, record in owners.items():
        rule = record["grid_rule"]
        assert rule["kind"] == "claimed_chunks", name
        for key in (
            "ctas_per_sm",
            "grid_factor",
            "launch_bound_ctas_per_sm",
            "rows_per_issue",
        ):
            assert isinstance(rule[key], int) and rule[key] >= 1, (name, key)
        # the slots per SM are the production grid factor, capped at the launch bound
        assert rule["ctas_per_sm"] == min(
            rule["grid_factor"], rule["launch_bound_ctas_per_sm"]
        ), name
        # the taper the record's kernel cuts its chunk table with, within the
        # module constants (which bound the workspace)
        tail_div, max_halving = loader.chunk_taper(rule)
        assert (rule["tail_div"], rule["max_halving"]) == (tail_div, max_halving), name
        assert 1 <= tail_div <= loader.TAIL_DIV, name
        assert 0 <= max_halving <= loader.MAX_HALVING, name
        workspace = record["workspace_rule"]
        assert workspace["kind"] == "chunk_partials" and workspace["counters"] >= 2, (
            name
        )
        # the kernel's twelfth parameter keeps its static-schedule name and carries the slot count
        assert record["logical"]["n_chunks"] == "n_chunks", name
        assert record["logical"]["rows_per_chunk"] == "slots", name
        assert ["parameter", "rows_per_chunk"] in [
            list(pair) for pair in record["arg_plan"]
        ], name
        assert "rows_per_chunk" not in record["logical"].values(), name
        assert "slots" not in record["logical"], name
    assert {"n_chunks", "slots"} <= set(loader.LOGICAL_SCALARS)
    assert "rows_per_chunk" not in loader.LOGICAL_SCALARS


def test_reduce_records_launch_one_cta_per_slab_under_pdl() -> None:
    companions = _companions()
    assert companions
    for name, record in companions.items():
        assert record["launch"]["use_pdl"] is True, name
        assert record["grid_rule"]["kind"] == "fixed", name
        assert record["workspace_rule"] is None, name
        hidden, ctas = int(record["hidden"]), int(record["grid_rule"]["ctas"])
        assert hidden % ctas == 0 and hidden // ctas in SHIPPED_SLABS, (
            name,
            hidden,
            ctas,
        )
        assert [list(pair) for pair in record["arg_plan"]][:3] == [
            ["buffer", "partial"],
            ["buffer", "dw"],
            ["parameter", "n_chunks"],
        ], name


@pytest.mark.parametrize("arch", sorted(loader.ROUTES))
def test_every_backward_route_selects_an_owner_then_its_companion(arch: str) -> None:
    owners, companions = _owners(), _companions()
    routes = loader.ROUTES[arch]
    checked = 0
    for hidden in HIDDEN_SIZES:
        for residual in (False, True):
            if loader.route_key(hidden, "bwd", residual) not in routes:
                continue
            names = loader.route_modules(arch, hidden, "bwd", residual)
            launched: set[str] = set()
            for rows in ROWS:
                selected = loader.selected_modules(
                    arch, hidden, "bwd", residual, rows=rows
                )
                assert len(selected) == 2, (arch, hidden, residual, rows, selected)
                owner, companion = selected
                assert owner in owners and companion in companions, (
                    arch,
                    hidden,
                    rows,
                    selected,
                )
                assert (
                    loader.MODULES[owner]["select_rule"]
                    == loader.MODULES[companion]["select_rule"]
                )
                assert (
                    loader.MODULES[owner]["hidden"]
                    == loader.MODULES[companion]["hidden"]
                    == hidden
                )
                launched.update(selected)
            assert launched == set(names), (
                arch,
                hidden,
                residual,
                set(names) - launched,
            )
            # owners precede companions in the route: the host launches admitted records in route order
            roles = [name in companions for name in names]
            assert roles == sorted(roles), (arch, hidden, residual, names)
            checked += 1
    assert checked >= len(HIDDEN_SIZES)


# --------------------------------------------------------------------------- the program windows (no device)


@pytest.mark.parametrize("arch", sorted(loader.ROUTES))
def test_program_windows_tile_every_route_and_name_the_selected_program(
    arch: str,
) -> None:
    """``program_windows`` tiles the positive row counts of every route once, from zero to an open last window,
    and names the program ``selected_modules`` launches at both ends of each window (a backward window's dw reduce
    companion shares the window)."""
    for route in loader.ROUTES[arch]:
        hidden, kind, residual, w_dtype = _route_parts(route)
        assert loader.route_key(hidden, kind, residual, w_dtype) == route
        route_windows = loader.program_windows(arch, hidden, kind, residual, w_dtype)
        assert (
            route_windows and route_windows[0][0] == 0 and route_windows[-1][1] is None
        ), (arch, route)
        for (low, high, _library), (next_low, _next_high, _next) in zip(
            route_windows, route_windows[1:], strict=False
        ):
            assert high is not None and low < high == next_low, (arch, route)
        for low, high, library in route_windows:
            top = max(OPEN_WINDOW_ROWS, low + 1) if high is None else high
            for rows in sorted({low + 1, top}):
                selected = loader.selected_modules(
                    arch, hidden, kind, residual, rows=rows, w_dtype=w_dtype
                )
                assert len(selected) == (2 if kind == "bwd" else 1), (
                    arch,
                    route,
                    rows,
                    selected,
                )
                assert loader.MODULES[selected[0]]["library"] == library, (
                    arch,
                    route,
                    rows,
                )
                if kind == "bwd":
                    assert loader.MODULES[selected[1]]["role"] == REDUCE_ROLE
                    assert (
                        loader.MODULES[selected[1]]["select_rule"]
                        == loader.MODULES[selected[0]]["select_rule"]
                    )


def test_rendered_windows_match_the_registry() -> None:
    """The exporter's ``cake_windows`` module is the registry: its table is ``program_windows`` of every route, its
    edges are the windows' upper bounds with a known class, and its sections (the docstring notes of the forward and
    backward) name every BF16 route of their kind and cover the FP32-weight routes in one sentence (they select among
    the twins over the same windows), the re-cut ones (``RECUT_FP32_ROUTES``) as lines of their own."""
    assert set(windows.PROGRAM_WINDOWS) == set(loader.ROUTES) == set(windows.EDGES)
    assert set(windows.EDGE_CLASSES) == {"same", "ulp", "dx", "unmeasured"}
    for arch, routes in loader.ROUTES.items():
        assert (
            set(windows.PROGRAM_WINDOWS[arch])
            == set(routes)
            == set(windows.EDGES[arch])
        ), arch
        for route in routes:
            hidden, kind, residual, w_dtype = _route_parts(route)
            expected = loader.program_windows(arch, hidden, kind, residual, w_dtype)
            assert windows.PROGRAM_WINDOWS[arch][route] == expected, (arch, route)
            edges = windows.EDGES[arch][route]
            assert tuple(edge for edge, _cls in edges) == _edges(expected), (
                arch,
                route,
            )
            assert all(cls in windows.EDGE_CLASSES for _edge, cls in edges), (
                arch,
                route,
            )
            listed = f"``{arch}`` ``{route}``:" in windows.SECTIONS[kind]
            if w_dtype == "f32":
                # an FP32-weight route is one sentence of the section, a re-cut one a line of its own; its base
                # route is a line either way
                recut = (arch, hidden, kind, residual) in RECUT_FP32_ROUTES
                assert listed == recut, (arch, route)
                base = loader.route_key(hidden, kind, residual)
                assert f"``{arch}`` ``{base}``:" in windows.SECTIONS[kind], (
                    arch,
                    route,
                )
                assert (
                    f"The FP32-weight routes (``{kind}_w32_...``)"
                    in windows.SECTIONS[kind]
                ), (arch, kind)
            else:
                assert listed, (arch, route)
    for kind, function in (
        ("fwd", pkg.cake_rmsnorm_det_train_forward),
        ("bwd", pkg.cake_rmsnorm_det_train_backward),
    ):
        section = windows.SECTIONS[kind]
        assert section.lstrip("\n").startswith("Program selection by token count\n")
        assert windows.NOTES.strip() in section
        doc = function.__doc__ or ""
        assert "    Notes\n    -----\n    Program selection by token count\n" in doc, (
            kind
        )
        for line in section.strip("\n").splitlines():
            assert line.strip() in doc, (kind, line)


def test_rows_cover_every_backward_window_edge() -> None:
    """``ROWS`` holds both sides of every backward window edge of every architecture, so the route test above
    launches each program of the registry (the forward edges are covered by the prefix-invariance test)."""
    for arch, routes in loader.ROUTES.items():
        for route in routes:
            hidden, kind, residual, w_dtype = _route_parts(route)
            if kind != "bwd":
                continue
            for edge in _edges(
                loader.program_windows(arch, hidden, kind, residual, w_dtype)
            ):
                assert edge in ROWS and edge + 1 in ROWS, (arch, route, edge)


# The H = 256 backward route: per sm_100a / sm_103a window the faster of the four-CTA pipeline (``pd4``) and the
# five-CTA ring at issue distance three without the L2 hint (``a3``), both at ``tail_div=4``, in two sessions per
# architecture (BF16 and FP32 weight alike); sm_107a runs the pipeline to 65536 rows and the ring above.
H256_BWD_PROGRAMS = {
    "pd4": "rmsnorm_train_bwd_h256_t128x4_pd4_k_s32_split_pts_cl_td4",
    "a3": "rmsnorm_train_bwd_h256_t128x5_cpa8a3nhkc8_k_s32_split_pts_cl_td4_cs",
}
H256_BWD_SPLIT = {
    **{
        arch: (
            (0, 2048, "a3"),
            (2048, 5120, "pd4"),
            (5120, 12288, "a3"),
            (12288, 40960, "pd4"),
            (40960, None, "a3"),
        )
        for arch in ("sm_100a", "sm_103a")
    },
    "sm_107a": ((0, 65536, "pd4"), (65536, None, "a3")),
}


def _program_name(symbol: str) -> str:
    return "cake_rmsnorm_det_train_" + hashlib.sha256(symbol.encode()).hexdigest()[:20]


@pytest.mark.parametrize("arch", sorted(H256_BWD_SPLIT))
def test_h256_backward_route_runs_the_measured_split(arch: str) -> None:
    """The H = 256 backward windows of ``arch`` are the measured split, each at its
    program's launch bound (four or five CTAs per SM) with the dw reduce of the
    same window, for the BF16 and the FP32-weight route; both rows of every edge
    are in ``ROWS`` (so the route test launches each side)."""
    for w_dtype in ("bf16", "f32"):
        suffix = "_w32" if w_dtype == "f32" else ""
        found = loader.program_windows(arch, 256, "bwd", False, w_dtype)
        want = tuple(
            (low, high, _program_name(H256_BWD_PROGRAMS[key] + suffix))
            for low, high, key in H256_BWD_SPLIT[arch]
        )
        assert tuple(tuple(window) for window in found) == want, (arch, w_dtype)
        for low, high, key in H256_BWD_SPLIT[arch]:
            owner, companion = loader.selected_modules(
                arch, 256, "bwd", False, rows=low + 1, w_dtype=w_dtype
            )
            rule = loader.MODULES[owner]["grid_rule"]
            cps = 4 if key == "pd4" else 5
            assert rule["ctas_per_sm"] == cps, (arch, low)
            assert rule["launch_bound_ctas_per_sm"] == cps, (arch, low)
            assert loader.MODULES[companion]["role"] == REDUCE_ROLE
            if high is not None:
                assert high in ROWS and high + 1 in ROWS, (arch, high)


# The FP32-weight routes the registry re-cuts, (arch, hidden, kind, residual) -> the windows of the FP32-weight route
# that replace the BF16 route's windows they tile (``(above, at_most)`` in rows): between 1024 and 6144 rows the packed
# H = 6144 forward's FP32-weight twin trails its BF16 kernel, a pipelined FP32-weight kernel does not; the packed twin
# keeps the rows above.  H = 5120 runs a parked or a pipelined FP32-weight kernel above 6144 rows at a grid of its
# own, and on SM100 up to 640 rows the FP32-weight twin of the trip kernel.  Such a window's record is the twin route's
# own (this architecture alone, its own grid rule).
RECUT_FP32_ROUTES: dict[
    tuple[str, int, str, bool], tuple[tuple[int, int | None], ...]
] = {
    ("sm_100a", 6144, "fwd", False): ((1024, 6144), (6144, None)),
    ("sm_103a", 6144, "fwd", False): ((1024, 6144), (6144, None)),
    ("sm_100a", 5120, "fwd", False): (
        (0, 640),
        (640, 6144),
        (6144, 16384),
        (16384, None),
    ),
    ("sm_103a", 5120, "fwd", False): ((0, 640), (640, 6144), (6144, None)),
    # sm_107a: the forward routes W32_ROUTE_OVERRIDES re-cuts there (a window of another twin of the route, the
    # windows between kept by the route's own twin under override records of its program)
    ("sm_107a", 2048, "fwd", False): ((12288, 16384),),
    ("sm_107a", 4096, "fwd", False): (
        (1024, 4096),
        (4096, 20480),
        (20480, 49152),
        (49152, None),
    ),
    ("sm_107a", 5120, "fwd", False): ((49152, None),),
}


def _window(record) -> tuple[int | None, int | None]:
    rule = (
        record["select_rule"] or {}
    )  # a record selected without a rule spans every row count
    return (rule.get("above"), rule.get("at_most"))


def _overlaps(first: tuple[int, int | None], second: tuple[int, int | None]) -> bool:
    return (second[1] is None or first[0] < second[1]) and (
        first[1] is None or second[0] < first[1]
    )


def _recut_windows(
    windows: list[tuple[int, int | None]], recut
) -> list[tuple[int, int | None]]:
    """``windows`` of the BF16 route with the ones ``recut`` tiles replaced by ``recut``; ``recut`` tiles whole
    windows."""
    if not recut:
        return list(windows)
    span = (recut[0][0], recut[-1][1])
    replaced = [w for w in windows if _overlaps(w, span)]
    assert replaced and (replaced[0][0], replaced[-1][1]) == span, (windows, recut)
    return sorted([w for w in windows if w not in replaced] + list(recut))


@pytest.mark.parametrize("arch", sorted(loader.ROUTES))
def test_fp32_weight_routes_mirror_their_bf16_routes(arch: str) -> None:
    """Every FP32-weight route (``_w32``) is the twin of the BF16 route of its width, kind and residual form: the
    same windows, companions and host plans, the owners' roles carrying ``_w32`` before ``_residual`` -- except the
    windows ``RECUT_FP32_ROUTES`` names, where the route runs another FP32-weight kernel under its own record; and
    the BF16 routes alone decide ``supported_hidden_sizes``."""
    routes = loader.ROUTES[arch]
    twins = [route for route in routes if _route_parts(route)[3] == "f32"]
    if not twins:
        pytest.skip(f"{arch}: the registry ships no FP32-weight route yet")
    for route in twins:
        hidden, kind, residual, _w_dtype = _route_parts(route)
        base = loader.route_key(hidden, kind, residual)
        assert base in routes, (arch, route)
        assert loader.weight_dtypes(arch, hidden, kind, residual) == ("bf16", "f32")
        mine = loader.program_windows(arch, hidden, kind, residual, "f32")
        theirs = loader.program_windows(arch, hidden, kind, residual)
        recut = RECUT_FP32_ROUTES.get((arch, hidden, kind, residual), ())
        assert [(low, high) for low, high, _l in mine] == _recut_windows(
            [(low, high) for low, high, _l in theirs], recut
        ), (arch, route)
        for low, high, twin_library in mine:
            # the BF16 program of the window's first rows: another program (the twin, or the re-cut kernel)
            base_library = next(
                l for lo, hi, l in theirs if lo <= low and (hi is None or low < hi)
            )
            assert twin_library != base_library, (arch, route, low, high)
        owners = [n for n in routes[route] if loader.MODULES[n]["role"] != REDUCE_ROLE]
        base_owners = [
            n for n in routes[base] if loader.MODULES[n]["role"] != REDUCE_ROLE
        ]
        assert [
            n for n in routes[route] if loader.MODULES[n]["role"] == REDUCE_ROLE
        ] == [n for n in routes[base] if loader.MODULES[n]["role"] == REDUCE_ROLE], (
            arch,
            route,
        )
        base_windows = [_window(loader.MODULES[n]) for n in base_owners]
        assert [_window(loader.MODULES[n]) for n in owners] == _recut_windows(
            base_windows, recut
        ), (arch, route)
        by_window = dict(zip(base_windows, base_owners, strict=True))
        for mine_name in owners:
            twin = loader.MODULES[mine_name]
            window = _window(twin)
            if window in by_window and "__ov" not in mine_name:
                owner = loader.MODULES[by_window[window]]
                # the owner's architectures less the ones RECUT_FP32_ROUTES re-cuts over this window on an
                # architecture whose BF16 route lists this owner record (a window record carries its library's
                # whole architecture list; the override records take the window there); a re-cut owner's twin
                # ships only where a route lists the owner
                recut_arches = {
                    a
                    for (a, h, k, r), ws in RECUT_FP32_ROUTES.items()
                    if (h, k, r) == (hidden, kind, residual)
                    and by_window[window] in loader.ROUTES.get(a, {}).get(base, ())
                    and any(_overlaps(window, w) for w in ws)
                }
                expected_arches = [a for a in owner["arches"] if a not in recut_arches]
                if recut_arches:
                    expected_arches = [
                        a
                        for a in expected_arches
                        if by_window[window] in set().union(*loader.ROUTES[a].values())
                    ]
                for key in (
                    "arches",
                    "grid_rule",
                    "hidden",
                    "select_rule",
                    "workspace_rule",
                ):
                    expected = expected_arches if key == "arches" else owner[key]
                    assert twin[key] == expected, (arch, route, mine_name, key)
                expected_role = (
                    owner["role"][: -len("_residual")] + "_w32_residual"
                    if owner["role"].endswith("_residual")
                    else owner["role"] + "_w32"
                )
                assert twin["role"] == expected_role, (arch, route, mine_name)
                continue
            # a window of a re-cut route: the twin route's own record, for this architecture alone, with the
            # BF16 owner's hidden size and workspace and a rows_per_cta grid rule of its own
            assert window in recut, (arch, route, mine_name)
            owner = loader.MODULES[
                next(
                    n
                    for n in base_owners
                    if _overlaps(_window(loader.MODULES[n]), window)
                )
            ]
            assert twin["arches"] == [arch] and arch in owner["arches"], (
                arch,
                route,
                mine_name,
            )
            assert (twin["hidden"], twin["workspace_rule"]) == (
                owner["hidden"],
                owner["workspace_rule"],
            ), (arch, route, mine_name)
            assert twin["grid_rule"]["kind"] == "rows_per_cta", (arch, route, mine_name)
            assert twin["role"].startswith(kind) and "_w32" in twin["role"], (
                arch,
                route,
                mine_name,
            )
    assert all(
        _route_parts(loader.route_key(h, "fwd", False))[3] == "bf16"
        for h in loader.supported_hidden_sizes(arch)
    )


# --------------------------------------------------------------------------- the chunk formula (no device)


def test_chunk_formula_matches_the_design_examples() -> None:
    """The host's copy of the kernels' chunk table: the halving rounds and the final equalised round, in
    pipeline-stage units (``rows_per_issue`` rows per stage), at the module constants and under the tapers a
    record may spell (its kernel's ``tail_div`` / ``max_halving``)."""
    assert (loader.TAIL_DIV, loader.MAX_HALVING, loader.MAX_ROUNDS) == (12, 5, 4)
    slots = 296  # two CTAs per SM on a 148-SM device
    assert [loader.n_chunks(T, slots, 1) for T in (256, 1024, 4096, 16384, 65536)] == [
        256,
        592,
        888,
        1184,
        1184,
    ]
    assert loader.chunk_rounds(1024, slots, 1) == [(296, 2), (136, 2), (160, 1)]
    assert loader.chunk_rounds(4096, slots, 1) == [
        (296, 7),
        (296, 4),
        (248, 3),
        (48, 2),
    ]
    assert loader.chunk_rounds(16384, slots, 1) == [
        (296, 28),
        (296, 14),
        (296, 7),
        (104, 7),
        (192, 6),
    ]
    assert loader.chunk_rounds(65536, slots, 1) == [
        (296, 111),
        (296, 56),
        (296, 28),
        (120, 27),
        (176, 26),
    ]
    # eight rows per stage (H = 512): a chunk is a whole number of stages, never a fraction of one
    assert loader.chunk_rounds(1024, 148, 8) == [(128, 8)]
    assert loader.chunk_rounds(16384, slots, 8) == [(296, 32), (296, 16), (272, 8)]
    assert loader.n_chunks(16384, slots, 8) == 864
    assert loader.chunk_rounds(65536, slots, 8) == [
        (296, 112),
        (296, 56),
        (296, 32),
        (200, 24),
        (96, 16),
    ]
    assert loader.n_chunks(65536, slots, 8) == 1184
    assert loader.chunk_rounds(16384, 148, 4) == [
        (148, 56),
        (148, 28),
        (148, 16),
        (100, 12),
        (48, 8),
    ]
    assert loader.n_chunks(16384, 148, 4) == 592
    # row counts between the last halving round's reach and the capacity (eight rows per stage, two CTAs per SM)
    assert [
        loader.n_chunks(T, slots, 8) for T in (32768, 33000, 36864, 37888, 38000)
    ] == [888, 888, 888, 888, 888]
    assert loader.chunk_rounds(0, slots, 1) == []
    assert loader.chunk_table(5, slots, 1) == [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    assert loader.chunk_table(5, slots, 4) == [(0, 4), (4, 5)]
    # a kernel built with a taper of its own cuts another table, which the host cuts with the record's taper:
    # tail_div 4 (one halving round at most) on one CTA per SM of a 148-SM device, four rows per stage
    assert [loader.n_chunks(T, 148, 4, 4, 5) for T in (1025, 1536, 2048, 4096)] == [
        148,
        236,
        296,
        296,
    ]
    assert loader.chunk_rounds(4096, 148, 4, 4, 5) == [(148, 16), (136, 12), (12, 8)]
    assert loader.chunk_rounds(4096, 148, 4) == [(148, 16), (148, 8), (136, 4)]
    # max_halving 2 on one CTA per SM of a 212-SM device
    assert [
        loader.n_chunks(T, 212, 4, 12, 2) for T in (14928, 16384, 32768, 65536)
    ] == [636, 636, 636, 636]
    assert loader.chunk_rounds(32768, 212, 4, 12, 2) == [
        (212, 80),
        (212, 40),
        (136, 36),
        (76, 32),
    ]
    assert loader.chunk_rounds(32768, 212, 4) == [
        (212, 80),
        (212, 40),
        (212, 20),
        (136, 16),
        (76, 12),
    ]
    # tail_div 6, and tail_div 4 with max_halving 2, eight rows per stage
    assert [loader.n_chunks(T, 148, 4, 6, 5) for T in (1024, 4096, 8192, 24000)] == [
        148,
        296,
        444,
        444,
    ]
    assert [loader.n_chunks(T, 148, 8, 4, 2) for T in (1024, 4096, 16384, 65536)] == [
        128,
        296,
        296,
        296,
    ]
    # at the constants the taper keywords change nothing; tail_div 1 cuts no halving round: one chunk per slot
    assert loader.n_chunks(16384, slots, 8, 12, 5) == loader.n_chunks(16384, slots, 8)
    assert loader.chunk_rounds(16384, 148, 4, 1, 5) == [(100, 112), (48, 108)]


def test_chunk_taper_reads_the_pair_within_the_constants() -> None:
    """A chunk-claiming record spells the taper its kernel was built with; the
    host reads the pair and refuses a rule that lacks one or leaves the bounds
    (the module constants) instead of cutting the table at the defaults."""
    rule = {
        "kind": "claimed_chunks",
        "ctas_per_sm": 1,
        "grid_factor": 1,
        "launch_bound_ctas_per_sm": 1,
        "rows_per_issue": 4,
        "tail_div": 4,
        "max_halving": 2,
    }
    assert loader.chunk_taper(rule) == (4, 2)
    assert loader.chunk_taper(
        {**rule, "tail_div": loader.TAIL_DIV, "max_halving": loader.MAX_HALVING}
    ) == (loader.TAIL_DIV, loader.MAX_HALVING)
    assert loader.chunk_taper({**rule, "tail_div": 1, "max_halving": 0}) == (1, 0)
    for bad in (
        {key: value for key, value in rule.items() if key != "tail_div"},
        {key: value for key, value in rule.items() if key != "max_halving"},
        {**rule, "max_halving": None},
        {**rule, "tail_div": True},
        {**rule, "tail_div": "4"},
        {**rule, "tail_div": 0},
        {**rule, "tail_div": loader.TAIL_DIV + 1},
        {**rule, "max_halving": -1},
        {**rule, "max_halving": loader.MAX_HALVING + 1},
    ):
        with pytest.raises(ValueError):
            loader.chunk_taper(bad)
    # the pure functions default to the constants and refuse a taper below the floors
    assert loader.n_chunks(16384, 148, 4) == loader.n_chunks(
        16384, 148, 4, loader.TAIL_DIV, loader.MAX_HALVING
    )
    with pytest.raises(ValueError):
        loader.chunk_rounds(16384, 148, 4, 0, 5)
    with pytest.raises(ValueError):
        loader.chunk_rounds(16384, 148, 4, 12, -1)


@pytest.mark.parametrize("taper", ((12, 5), (4, 5), (12, 2), (1, 0)))
@pytest.mark.parametrize("slots", (1, 3, 148, 152, 296, 304, 432))
@pytest.mark.parametrize("rpi", (1, 2, 4, 8))
def test_chunk_table_covers_the_rows_within_the_capacity(
    slots: int, rpi: int, taper: tuple[int, int]
) -> None:
    tail_div, max_halving = taper
    rows_list = [
        *range(1, 600 if taper == (12, 5) else 200),
        1023,
        1024,
        1025,
        4095,
        4096,
        4097,
        16383,
        16384,
        16385,
        65536,
        65537,
        1 << 20,
    ]
    for T in rows_list:
        stages = -(-T // rpi)
        table = loader.chunk_table(T, slots, rpi, tail_div, max_halving)
        assert table[0][0] == 0 and table[-1][1] == T
        assert all(b < e for b, e in table)  # no empty chunk
        assert all(
            table[i][1] == table[i + 1][0] for i in range(len(table) - 1)
        )  # exact cover, no overlap
        assert all(
            (b % rpi == 0) and (e % rpi == 0 or e == T) for b, e in table
        )  # whole stages, ragged last
        n = loader.n_chunks(T, slots, rpi, tail_div, max_halving)
        assert n == len(table)
        assert n <= loader.chunk_capacity(T, slots) == min(T, loader.MAX_ROUNDS * slots)
        if stages <= slots:
            assert n == stages  # one stage per chunk below the slot count
        rounds = loader.chunk_rounds(T, slots, rpi, tail_div, max_halving)
        # a taper within the constants cuts no more rounds, hence no more chunks, than they do
        assert len(rounds) <= len(loader.chunk_rounds(T, slots, rpi))
        assert n <= loader.n_chunks(T, slots, rpi)
        if tail_div == 1 or max_halving == 0:
            # no halving round: one chunk per slot
            assert len(rounds) <= 2 and n == min(stages, slots)
        sizes = [rows for _count, rows in rounds]
        assert sizes == sorted(
            sizes, reverse=True
        )  # chunk sizes never increase in claim order
        assert all(rows % rpi == 0 for rows in sizes)
        assert sum(count * rows for count, rows in rounds) == stages * rpi


# --------------------------------------------------------------------------- the workspace (device)


def _device() -> torch.device:
    return torch.device("cuda:0")


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and torch.cuda.get_device_capability(_device()) in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_det_train_supported(_device())
    )


@pytest.mark.skipif(
    not _supported(),
    reason="needs an SM100 / SM103 / SM107 device with exported routes",
)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_workspace_puts_the_counters_first_and_grows_with_the_rows(hidden: int) -> None:
    if not pkg.is_cake_rmsnorm_det_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")
    device = _device()
    sizes = [
        pkg.cake_rmsnorm_det_train_backward_workspace_bytes(rows, hidden, device)
        for rows in (1, 129, 4096, 16384, 65536)
    ]
    assert sizes == sorted(sizes) and all(size % 256 == 0 for size in sizes)
    assert sizes[0] > loader.COUNTERS_BYTES
    arch = loader.arch_for_capability(torch.cuda.get_device_capability(device))
    sms = loader.sm_count(0)
    for name in loader.route_modules(arch, hidden, "bwd", False):
        record = loader.MODULES[name]
        if record.get("workspace_rule") is None:
            continue
        for rows in (1, 129, 16384, 65536):
            layout = loader.workspace_layout(record, rows, device_index=0)
            slots = int(record["grid_rule"]["ctas_per_sm"]) * sms
            assert (
                layout["counters_offset"] == 0
                and layout["partial_offset"] == loader.COUNTERS_BYTES
            )
            assert layout["slots"] == slots
            assert layout["n_chunks"] == loader.n_chunks(
                rows,
                slots,
                int(record["grid_rule"]["rows_per_issue"]),
                *loader.chunk_taper(record["grid_rule"]),
            )
            assert (
                layout["n_chunks"]
                <= layout["capacity_chunks"]
                == min(rows, loader.MAX_ROUNDS * slots)
            )
            assert layout["partial_bytes"] == 4 * layout["capacity_chunks"] * hidden
            assert (
                layout["total_bytes"]
                == loader.COUNTERS_BYTES + -(-layout["partial_bytes"] // 256) * 256
            )
            assert loader.launch_grid(record, rows, device_index=0) == (
                min(slots, layout["n_chunks"]),
                1,
                1,
            )
        # a record without its taper is refused, not cut at the defaults
        stripped = {
            **record,
            "grid_rule": {
                key: value
                for key, value in record["grid_rule"].items()
                if key != "tail_div"
            },
        }
        with pytest.raises(ValueError, match="tail_div"):
            loader.launch_grid(stripped, 4096, device_index=0)
        with pytest.raises(ValueError, match="tail_div"):
            loader.workspace_layout(stripped, 4096, device_index=0)
    workspace = pkg.cake_rmsnorm_det_train_backward_workspace(65536, hidden, device)
    assert workspace.dtype == torch.uint8 and workspace.numel() == sizes[-1]
    assert torch.count_nonzero(workspace[: loader.COUNTERS_BYTES]) == 0
