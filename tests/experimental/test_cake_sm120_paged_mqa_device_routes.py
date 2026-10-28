"""SM120a paged FP8 MQA logits: catalog schema v2, per-device routes, dyn schedules.

Host-only checks of the catalog's ``policy.device_rule`` and ``device_routes``
(schema, context windows, table well-formedness, the shipped routes left
unchanged), of the route resolver against an independent mirror over a batch,
hint and device grid, of the route-aware schedule atom count the metadata
entry uses, and of the schedule bindings a plan checks before it prepares a
submission.  The committed catalog and a synthetic catalog with populated
tables both run through the same checks.  The synthetic static device routes
reuse shipped sequences, so the GPU checks launch them on an SM120 device (with
the device SM count monkeypatched to the table's key); the synthetic ``dyn``
route checks the plan-private schedule binding without a launch.  Every
committed device route also launches, driven by one of its table entries, and
must score the default route's bits; the first committed static and ``dyn``
routes also replay from a CUDA Graph on changed contents, and the one-shot
entry runs the ``dyn`` one.
"""

import copy
import hashlib
import json
import re

import pytest
import torch
from flashinfer.experimental.deepgemm_sm120_paged_mqa_logits import (
    sm120_paged_mqa as _runtime,
)
from flashinfer.sm120_paged_mqa_logits import (
    fp8_paged_mqa_logits,
    get_paged_mqa_logits_metadata,
    prepare_sm120_paged_mqa_logits,
)

from tests.experimental.test_sm120_paged_mqa_logits import (
    _check,
    _inputs,
    _reference,
    _schedule_reference,
)

# The schema-v1 content (arches, policy without the device rule, the 31
# routes and the programs they launch) that schema v2 must carry unchanged.
SHIPPED_DIGEST = "2e661f3926a3c48f0283416bba44cfa1d0fe6891bf537eed6c22d41fbfca5b45"
SHIPPED_ROUTES = 31
ARG_KINDS = {"buffer", "tma_buffer", "parameter", "grid"}
DEVICE_ROUTE = re.compile(r"^sm120:fp8:h(\d+):p(\d+):n(\d+):a(\d+):(\w+)$")
TABLE_KEY = re.compile(r"^h(\d+):p(\d+):n(\d+):(a\d+):(\w+)$")

# Synthetic device routes.  The static ones reuse a shipped sequence of the
# same geometry, so they can launch: the 32-head page-64 next_n 4 geometry at
# the two-token atom (``t0``) and at its own native atom under another name
# (``t1``), the 64-head next_n 5 geometry at the two-token atom (``t2``).  The
# ``dyn`` route (``t3``) names a program that is never built.
T0 = "sm120:fp8:h32:p64:n4:a2:t0"
T1 = "sm120:fp8:h32:p64:n4:a4:t1"
T2 = "sm120:fp8:h64:p64:n5:a2:t2"
T3 = "sm120:fp8:h64:p64:n6:a6:t3"
DYN_SEQUENCE = "synthetic_dyn_sequence"
DYN = {"kind": "dyn", "dyn_slots": 8, "dyn_pct": 3, "dyn_seg": 1}
SYNTHETIC_TABLES = {
    "170": {
        "h32:p64:n4:a2:c4k": [[8, T0]],
        "h32:p64:n4:a4:c16k": [[9, None], [32, T0], [64, T0], [128, T1], [256, None]],
        "h64:p64:n5:a3:c64k": [[128, T2], [256, T2]],
    },
    "188": {
        "h32:p64:n4:a4:c4k": [[64, T1]],
        "h64:p64:n6:a3:c16k": [[256, T3]],
    },
}
HINT_C16K = 20000


def _synthetic_catalog():
    catalog = copy.deepcopy(_runtime._catalog())
    routes, programs = catalog["routes"], catalog["programs"]
    device = catalog["device_routes"]
    device.clear()  # the synthetic routes replace the committed ones
    device[T0] = {**routes["sm120:fp8:h32:p64:n4:a2"], "schedule": {"kind": "static"}}
    device[T1] = dict(routes["sm120:fp8:h32:p64:n4:a4"])
    device[T2] = dict(routes["sm120:fp8:h64:p64:n5:a2"])
    template = routes["sm120:fp8:h64:p64:n6:a3"]
    device[T3] = {
        **template,
        "atom_variant": "a6",
        "next_n_atom": 6,
        "schedule": dict(DYN),
        "sequence": DYN_SEQUENCE,
    }
    record = copy.deepcopy(programs[template["sequence"]])
    plan = record["arg_plan"]
    at = [key for _kind, key in plan].index("metadata.num_sms") + 1
    plan[at:at] = [
        ["parameter", "metadata.dyn_slots"],
        ["parameter", "metadata.dyn_pct"],
        ["parameter", "metadata.dyn_seg"],
    ]
    record.update(ffi_entry=DYN_SEQUENCE, sources=["csrc/synthetic/dyn_sequence.cu"])
    programs[DYN_SEQUENCE] = record
    catalog["policy"]["threads"]["metadata_dyn"] = 1024
    catalog["policy"]["device_rule"]["tables"] = copy.deepcopy(SYNTHETIC_TABLES)
    return catalog


def _use(monkeypatch, catalog):
    monkeypatch.setattr(_runtime, "_catalog", lambda: catalog)
    return catalog


@pytest.fixture(params=["committed", "synthetic"])
def catalog(request, monkeypatch):
    if request.param == "committed":
        return _runtime._catalog()
    return _use(monkeypatch, _synthetic_catalog())


def _shipped_digest(catalog):
    routes = catalog["routes"]
    names = set()
    for record in routes.values():
        if record.get("sequence"):
            names.add(record["sequence"])
        for _stage, program in record.get("stages") or ():
            names.add(program)
    policy = {k: v for k, v in catalog["policy"].items() if k != "device_rule"}
    policy["threads"] = {
        k: v for k, v in policy["threads"].items() if k != "metadata_dyn"
    }
    shipped = dict(
        arches=catalog["arches"],
        policy=policy,
        routes=routes,
        programs={name: catalog["programs"][name] for name in sorted(names)},
    )
    text = json.dumps(shipped, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Schema and catalog shape (host-only)
# ---------------------------------------------------------------------------


def test_catalog_schema_v2():
    catalog = _runtime._catalog()
    assert _runtime.CATALOG_SCHEMA == "sm120_paged_mqa.v2"
    assert catalog["schema"] == _runtime.CATALOG_SCHEMA
    assert isinstance(catalog["device_routes"], dict)
    assert set(_runtime.device_rule()) == {"ctx_windows", "tables"}
    text = json.dumps(catalog)
    assert _runtime._parse_catalog(text) == catalog
    with pytest.raises(RuntimeError, match="regenerate"):
        _runtime._parse_catalog(
            text.replace('"sm120_paged_mqa.v2"', '"sm120_paged_mqa.v1"')
        )


def test_shipped_routes_and_programs_unchanged():
    """Schema v2 is additive: the shipped routes and their programs keep every value."""
    catalog = _runtime._catalog()
    assert len(catalog["routes"]) == SHIPPED_ROUTES
    assert _shipped_digest(catalog) == SHIPPED_DIGEST
    # The synthetic catalog only adds records, so it keeps the digest too.
    assert _shipped_digest(_synthetic_catalog()) == SHIPPED_DIGEST


def test_ctx_windows_are_ordered_and_contiguous(catalog):
    windows = sorted(
        (int(low), int(high), name)
        for name, (low, high) in _runtime.device_rule()["ctx_windows"].items()
    )
    assert windows and windows[0][0] >= 1
    for (low, high, _name), (following, _h, _n) in zip(
        windows, windows[1:], strict=False
    ):
        assert low < high == following
    assert _runtime.ctx_window(None) is None
    with pytest.raises(ValueError):
        _runtime.ctx_window(0)
    names = [name for _low, _high, name in windows]
    for index, (low, high, name) in enumerate(windows):
        assert _runtime.ctx_window(low) == name
        assert _runtime.ctx_window(high - 1) == name
        before = names[index - 1] if index else None
        after = names[index + 1] if index + 1 < len(names) else None
        if low > 1:
            assert _runtime.ctx_window(low - 1) == before
        assert _runtime.ctx_window(high) == after


def _check_device_routes(catalog, *, check_files):
    from flashinfer.jit import env

    policy = catalog["policy"]
    rule = policy["device_rule"]
    windows = rule["ctx_windows"]
    routes, device, programs = (
        catalog["routes"],
        catalog["device_routes"],
        catalog["programs"],
    )
    heads, pages, next_ns = (
        set(policy["heads"]),
        set(policy["page_kv"]),
        set(policy["next_n"]),
    )
    referenced = set()
    for sms, table in rule["tables"].items():
        assert sms.isdigit() and int(sms) > 0, sms
        for key, grid in table.items():
            match = TABLE_KEY.match(key)
            assert match and grid, key
            h, p, n = (int(v) for v in match.group(1, 2, 3))
            atom, window = match.group(4, 5)
            assert h in heads and p in pages and n in next_ns, key
            assert window in windows, key
            batches = [int(batch) for batch, _route in grid]
            assert batches == sorted(set(batches)), key
            assert batches[0] >= 1 and batches[-1] <= int(policy["max_batch"]), key
            low, high = (int(v) for v in windows[window])
            for batch, route in grid:
                # The key's atom is the default rule's at this batch for some
                # hint of the window (the long-context table flips at one bound).
                reachable = {
                    _runtime.atom_variant(h, n, batch, max_context_hint=hint)
                    for hint in (low, high - 1)
                }
                assert atom in reachable, (key, batch, reachable)
                if route is None:
                    continue
                assert route in device, (key, route)
                record = device[route]
                assert (record["num_heads"], record["page_kv"], record["next_n"]) == (
                    h,
                    p,
                    n,
                ), (key, route)
                referenced.add(route)
    assert referenced == set(device), sorted(set(device) - referenced)
    for name, record in device.items():
        assert name not in routes, name
        match = DEVICE_ROUTE.match(name)
        assert match, name
        h, p, n, a = (int(v) for v in match.group(1, 2, 3, 4))
        assert (record["num_heads"], record["page_kv"], record["next_n"]) == (h, p, n)
        assert record["next_n_atom"] == a and 1 <= a <= n, name
        assert record["atom_variant"] == f"a{a}", name
        assert record["kernel_launches"] == 2 and record["clean_logits"] == "raw"
        program = programs[record["sequence"]]
        assert program["kind"] == "sequence" and program["arches"] == catalog["arches"]
        kinds = {kind for kind, _key in program["arg_plan"]}
        assert kinds <= ARG_KINDS, (name, kinds)
        keys = {key for _kind, key in program["arg_plan"]}
        schedule = _runtime.route_schedule(record)
        if schedule["kind"] == "dyn":
            assert int(schedule["dyn_slots"]) >= 1, name
            assert 1 <= int(schedule["dyn_pct"]) <= 100, name
            assert int(schedule["dyn_seg"]) >= 1, name
            assert {
                "metadata.dyn_slots",
                "metadata.dyn_pct",
                "metadata.dyn_seg",
            } <= keys
            assert int(policy["threads"]["metadata_dyn"]) > 0
        else:
            assert schedule == {"kind": "static"}, name
        if check_files:
            for source in program["sources"]:
                path = env.FLASHINFER_CSRC_DIR / source.removeprefix("csrc/")
                assert path.is_file(), (name, source)


def test_committed_device_routes_are_well_formed():
    _check_device_routes(_runtime._catalog(), check_files=True)


def test_synthetic_device_routes_are_well_formed(monkeypatch):
    _check_device_routes(_use(monkeypatch, _synthetic_catalog()), check_files=False)


def test_route_atoms_and_dyn_rows():
    rule = _runtime.atom_rule()
    for name, record in _runtime._catalog()["routes"].items():
        if not record.get("sequence"):
            continue
        variant = record["atom_variant"]
        assert _runtime.route_next_n_atoms(record) == int(
            rule["num_atoms"][f"n{record['next_n']}:{variant}"]
        ), name
        assert _runtime.route_schedule(record) == {"kind": "static"}
    assert _runtime.dyn_schedule_shape(170, 8) == (1532, 2)
    assert _runtime.dyn_schedule_shape(188, 8) == (1694, 2)
    for num_sms in (1, 84, 170, 188):
        for slots in (0, 1, 8):
            rows = (1 + slots) * num_sms + 2
            assert _runtime.dyn_schedule_shape(num_sms, slots) == (rows, 2)


# ---------------------------------------------------------------------------
# Resolver against an independent mirror (host-only)
# ---------------------------------------------------------------------------


def _mirror_variant(catalog, heads, next_n, batch, hint):
    rule = catalog["policy"]["atom_rule"]
    long_context = hint is not None and hint >= int(rule["long_context_min_kv"])
    table = rule["selection_long" if long_context else "selection"]
    index = len([s for s in rule["batch_class_min"] if int(s) <= batch]) - 1
    return table[f"h{heads}:n{next_n}"][index]


def _mirror_expand(catalog):
    """``{(sms, key): {batch: route or None}}`` over every integer batch a key decides."""
    expanded = {}
    for sms, table in catalog["policy"]["device_rule"]["tables"].items():
        for key, grid in table.items():
            decided = {}
            for (b0, r0), (b1, r1) in zip(grid, grid[1:], strict=False):
                if r0 is not None and r0 == r1:
                    decided.update({b: r0 for b in range(b0 + 1, b1)})
            decided.update({b: r for b, r in grid})
            expanded[(sms, key)] = decided
    return expanded


def _mirror_route(catalog, expanded, geometry, batch, hint, budget, sms):
    heads, page_kv, next_n = geometry
    variant = _mirror_variant(catalog, heads, next_n, batch, hint)
    default = f"sm120:fp8:h{heads}:p{page_kv}:n{next_n}:{variant}"
    if hint is None or budget != sms:
        return default, None
    windows = catalog["policy"]["device_rule"]["ctx_windows"]
    inside = [name for name, (lo, hi) in windows.items() if lo <= hint < hi]
    if not inside:
        return default, None
    key = f"h{heads}:p{page_kv}:n{next_n}:{variant}:{inside[0]}"
    device = expanded.get((str(sms), key), {}).get(batch)
    return (device or default), device


def _grid(catalog):
    policy = catalog["policy"]
    geometries = sorted(
        {
            (r["num_heads"], r["page_kv"], r["next_n"])
            for r in catalog["routes"].values()
            if r.get("sequence")
        }
    )
    batches = set(range(1, 301)) | {int(policy["max_batch"])}
    for start in policy["atom_rule"]["batch_class_min"]:
        batches |= {int(start) - 1, int(start), int(start) + 1}
    for table in policy["device_rule"]["tables"].values():
        for grid in table.values():
            for batch, _route in grid:
                batches |= {batch - 1, batch, batch + 1}
    batches = sorted(b for b in batches if 1 <= b <= int(policy["max_batch"]))
    long_min = int(policy["atom_rule"]["long_context_min_kv"])
    hints = {None, 1, long_min - 1, long_min, 10**7}
    for low, high in policy["device_rule"]["ctx_windows"].values():
        hints |= {low - 1, low, high - 1, high}
    hints.discard(0)
    devices = [(170, 170), (188, 188), (84, 84), (169, 170), (188, 170)]
    return (
        geometries,
        batches,
        sorted(hints, key=lambda v: -1 if v is None else v),
        devices,
    )


def test_resolve_route_matches_the_mirror(catalog):
    """``resolve_route`` against an independent mirror over the batch, hint and device grid.

    Invariant: wherever no table entry applies the resolved route is
    ``route_name()``, the shipped default; with the committed (empty) tables
    that is every point of the grid.
    """
    expanded = _mirror_expand(catalog)
    geometries, batches, hints, devices = _grid(catalog)
    hits = 0
    for geometry in geometries:
        for batch in batches:
            for hint in hints:
                default = _runtime.route_name(*geometry, batch, max_context_hint=hint)
                for budget, sms in devices:
                    want, device = _mirror_route(
                        catalog, expanded, geometry, batch, hint, budget, sms
                    )
                    got = _runtime.resolve_route(
                        *geometry,
                        batch,
                        max_context_hint=hint,
                        cta_budget=budget,
                        device_sms=sms,
                    )
                    assert got == want, (geometry, batch, hint, budget, sms)
                    assert (
                        _runtime.device_route(
                            *geometry,
                            batch,
                            max_context_hint=hint,
                            cta_budget=budget,
                            device_sms=sms,
                        )
                        == device
                    )
                    if device is None:
                        assert got == default
                    else:
                        hits += 1
    if any(catalog["policy"]["device_rule"]["tables"].values()):
        assert hits
    else:
        assert hits == 0
    # Without a device the resolver is the default rule.
    assert _runtime.resolve_route(32, 64, 4, 48, max_context_hint=HINT_C16K) == (
        _runtime.route_name(32, 64, 4, 48, max_context_hint=HINT_C16K)
    )


def test_synthetic_table_spot_checks(monkeypatch):
    _use(monkeypatch, _synthetic_catalog())

    def resolve(heads, page_kv, next_n, batch, hint, budget=170, sms=170):
        return _runtime.resolve_route(
            heads,
            page_kv,
            next_n,
            batch,
            max_context_hint=hint,
            cta_budget=budget,
            device_sms=sms,
        )

    default = _runtime.route_name(32, 64, 4, 48, max_context_hint=HINT_C16K)
    assert default == "sm120:fp8:h32:p64:n4:a4"
    assert resolve(32, 64, 4, 32, HINT_C16K) == T0  # listed batch
    assert resolve(32, 64, 4, 48, HINT_C16K) == T0  # both neighbours agree
    assert resolve(32, 64, 4, 96, HINT_C16K) == default  # neighbours disagree
    assert resolve(32, 64, 4, 128, HINT_C16K) == T1
    assert resolve(32, 64, 4, 9, HINT_C16K) == default  # a null entry
    assert resolve(32, 64, 4, 16, HINT_C16K) == default  # null neighbour
    assert resolve(32, 64, 4, 200, HINT_C16K) == default
    assert resolve(32, 64, 4, 300, HINT_C16K) == default  # above the grid
    assert resolve(32, 64, 4, 48, None) == default  # no hint
    assert resolve(32, 64, 4, 48, 10648) == default  # c4k: no a4 key on 170
    assert resolve(32, 64, 4, 48, 170392) == default  # above every window
    assert resolve(32, 64, 4, 48, HINT_C16K, budget=169) == default
    assert resolve(32, 64, 4, 48, HINT_C16K, sms=84, budget=84) == default
    assert resolve(32, 128, 4, 48, HINT_C16K) == "sm120:fp8:h32:p128:n4:a4"
    # The policy atom keys the table: batch 8 below the long-context bound is
    # the two-token class, at or above it the native atom.
    assert resolve(32, 64, 4, 8, 4096) == T0
    assert resolve(32, 64, 4, 8, 12000) == "sm120:fp8:h32:p64:n4:a2"
    assert resolve(64, 64, 5, 192, 65536) == T2
    assert resolve(64, 64, 6, 256, HINT_C16K, 188, 188) == T3
    assert resolve(64, 64, 6, 256, HINT_C16K) == "sm120:fp8:h64:p64:n6:a3"


def test_schedule_atoms_and_the_metadata_num_heads_rule(catalog):
    """The metadata entry's atom count follows the resolved routes, per head count.

    ``num_heads`` is required exactly where the routes of the exported head
    counts disagree on the atom count; with no table entry this is the
    default rule (:func:`next_n_atoms` without device arguments).
    """
    synthetic = catalog["policy"]["device_rule"]["tables"] == SYNTHETIC_TABLES
    geometries, batches, hints, devices = _grid(catalog)
    pages = sorted(_runtime.exported_page_sizes())
    for next_n in _runtime.exported_next_n():
        for batch in batches[::7] + [8, 9, 48, 64, 128, 256]:
            for hint in hints:
                for page_kv in pages:
                    for budget, sms in devices:
                        device = dict(
                            page_kv=page_kv, cta_budget=budget, device_sms=sms
                        )
                        counts = {}
                        for heads in _runtime.exported_heads():
                            route = _runtime.resolve_route(
                                heads,
                                page_kv,
                                next_n,
                                batch,
                                max_context_hint=hint,
                                cta_budget=budget,
                                device_sms=sms,
                            )
                            if route in catalog["device_routes"]:
                                want = _runtime.route_next_n_atoms(
                                    catalog["device_routes"][route]
                                )
                            else:
                                want = _runtime.next_n_atoms(
                                    next_n, batch, heads, max_context_hint=hint
                                )
                            got = _runtime.next_n_atoms(
                                next_n, batch, heads, max_context_hint=hint, **device
                            )
                            assert got == want
                            counts[heads] = got
                        if len(set(counts.values())) == 1:
                            assert _runtime.next_n_atoms(
                                next_n, batch, max_context_hint=hint, **device
                            ) == next(iter(counts.values()))
                        else:
                            with pytest.raises(ValueError, match="num_heads"):
                                _runtime.next_n_atoms(
                                    next_n, batch, max_context_hint=hint, **device
                                )
    if synthetic:
        on_170 = dict(page_kv=64, cta_budget=170, device_sms=170)
        # A table entry makes the head counts disagree where the default agrees.
        assert _runtime.next_n_atoms(4, 64, max_context_hint=HINT_C16K) == 1
        with pytest.raises(ValueError, match="num_heads"):
            _runtime.next_n_atoms(4, 64, max_context_hint=HINT_C16K, **on_170)
        assert (
            _runtime.next_n_atoms(4, 64, 32, max_context_hint=HINT_C16K, **on_170) == 2
        )
        assert (
            _runtime.next_n_atoms(4, 64, 64, max_context_hint=HINT_C16K, **on_170) == 1
        )
        # ... and agree where the default disagrees (64 heads take one 6-token atom).
        on_188 = dict(page_kv=64, cta_budget=188, device_sms=188)
        with pytest.raises(ValueError, match="num_heads"):
            _runtime.next_n_atoms(6, 256, max_context_hint=HINT_C16K)
        assert _runtime.next_n_atoms(6, 256, max_context_hint=HINT_C16K, **on_188) == 1


# ---------------------------------------------------------------------------
# Bindings and launches (CUDA)
# ---------------------------------------------------------------------------


def _sm120_arch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        return _runtime.device_arch(torch.device("cuda"))
    except RuntimeError as error:
        pytest.skip(str(error))


def test_device_route_bindings_cover_every_program_argument(catalog):
    """Every device-route sequence argument is bound by name (no build, no launch)."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    device = torch.device("cuda")
    num_sms, batch, pages, max_context_len = 4, 2, 4, 128
    for name, record in catalog["device_routes"].items():
        heads, page_kv, next_n = (
            int(record["num_heads"]),
            int(record["page_kv"]),
            int(record["next_n"]),
        )
        context_lens = torch.full(
            (batch, next_n), max_context_len, dtype=torch.int32, device=device
        )
        schedule = _runtime.route_schedule(record)
        rows = (
            _runtime.dyn_schedule_shape(num_sms, schedule["dyn_slots"])
            if schedule["kind"] == "dyn"
            else _runtime.metadata_shape(num_sms)
        )
        walk = torch.zeros(rows, dtype=torch.int32, device=device)
        q = torch.zeros(
            (batch, next_n, heads, 128), dtype=torch.float8_e4m3fn, device=device
        )
        fused = torch.zeros((pages, page_kv * 132), dtype=torch.uint8, device=device)
        weights = torch.zeros(
            (batch * next_n, heads), dtype=torch.float32, device=device
        )
        block_table = torch.zeros((batch, pages), dtype=torch.int32, device=device)
        output = torch.zeros(
            (batch * next_n, _runtime.paged_logits_stride(max_context_len)),
            dtype=torch.float32,
            device=device,
        )
        bindings = {
            "metadata": _runtime.metadata_bindings(
                context_lens,
                walk,
                num_sms=num_sms,
                num_next_n_atoms=_runtime.route_next_n_atoms(record),
                schedule=schedule,
            ),
            "logits": _runtime.logits_bindings(
                q,
                fused,
                weights,
                context_lens,
                block_table,
                walk,
                output,
                num_sms=num_sms,
            ),
        }
        plan = catalog["programs"][record["sequence"]]["arg_plan"]
        missing = [
            key
            for _kind, key in plan
            if key.split(".", 1)[1] not in bindings[key.split(".", 1)[0]]
        ]
        assert not missing, f"{name}: unbound arguments {missing}"
        if schedule["kind"] == "dyn":
            for key in ("dyn_slots", "dyn_pct", "dyn_seg"):
                assert bindings["metadata"][key] == int(schedule[key])


def _host_bindings(schedule, num_sms, rows):
    """Sequence bindings over host tensors (the checks read shapes and pointers)."""
    heads, page_kv, next_n, batch, pages, max_context_len = 64, 64, 6, 2, 4, 128
    walk = torch.zeros(rows, dtype=torch.int32)
    context_lens = torch.full((batch, next_n), max_context_len, dtype=torch.int32)
    q = torch.zeros((batch, next_n, heads, 128), dtype=torch.float8_e4m3fn)
    fused = torch.zeros((pages, page_kv * 132), dtype=torch.uint8)
    weights = torch.zeros((batch * next_n, heads), dtype=torch.float32)
    block_table = torch.zeros((batch, pages), dtype=torch.int32)
    output = torch.zeros(
        (batch * next_n, _runtime.paged_logits_stride(max_context_len)),
        dtype=torch.float32,
    )
    return {
        "metadata": _runtime.metadata_bindings(
            context_lens, walk, num_sms=num_sms, num_next_n_atoms=1, schedule=schedule
        ),
        "logits": _runtime.logits_bindings(
            q,
            fused,
            weights,
            context_lens,
            block_table,
            walk,
            output,
            num_sms=num_sms,
        ),
    }


@pytest.mark.parametrize("schedule", [{"kind": "static"}, DYN], ids=["static", "dyn"])
def test_schedule_bindings_are_checked_before_launch(schedule):
    """The scheduler's grid and CTA count and the shared schedule's size (host-only).

    The plans run this check before every submission is prepared; each broken
    binding below must be refused.
    """
    num_sms = 6
    rows = (
        _runtime.dyn_schedule_shape(num_sms, schedule["dyn_slots"])
        if schedule["kind"] == "dyn"
        else _runtime.metadata_shape(num_sms)
    )
    bindings = _host_bindings(schedule, num_sms, rows)
    _runtime.check_schedule_bindings(bindings, num_sms=num_sms, schedule=schedule)
    _runtime.check_schedule_bindings(
        {"metadata": bindings["metadata"]}, num_sms=num_sms, schedule=schedule
    )

    def broken(stage, **changes):
        return {**bindings, stage: {**bindings[stage], **changes}}

    other = bindings["logits"]["schedule_meta"].clone()
    cases = [
        (broken("metadata", grid_x=2), r"grid \(1, 1, 1\)"),
        (broken("metadata", grid_z=2), r"grid \(1, 1, 1\)"),
        (broken("metadata", num_sms=num_sms + 1), "CTA budget"),
        (broken("logits", grid_x=num_sms + 1), "logits stage must launch"),
        (broken("logits", grid_y=2), "logits stage must launch"),
        (broken("logits", schedule_meta=other), "buffer the scheduler writes"),
        (_host_bindings(schedule, num_sms, (rows[0] - 1, 2)), "int32 words"),
        (_host_bindings(schedule, num_sms, (rows[0] + 1, 2)), "int32 words"),
    ]
    if schedule["kind"] == "dyn":
        cases.append(
            (broken("metadata", dyn_slots=schedule["dyn_slots"] + 1), "dyn_slots")
        )
    else:
        cases.append(
            (
                _host_bindings(
                    schedule, num_sms, _runtime.dyn_schedule_shape(num_sms, 8)
                ),
                "int32 words",
            )
        )
    for bad, match in cases:
        with pytest.raises(ValueError, match=match):
            _runtime.check_schedule_bindings(bad, num_sms=num_sms, schedule=schedule)


def _inside(data):
    position = torch.arange(data["max_context_len"], device="cuda")[None, :]
    return position < data["context_lens"].reshape(-1)[:, None]


def test_static_device_route_runs_and_matches_the_default_route(monkeypatch):
    """A static device route launches its program and scores the default route's bits."""
    arch = _sm120_arch()
    _use(monkeypatch, _synthetic_catalog())
    monkeypatch.setattr(_runtime, "device_facts", lambda index: (arch, 170))
    heads, page_kv, next_n, batch = 32, 64, 4, 48
    data = _inputs(heads, page_kv, next_n, batch, 512, seed=41)
    args = (
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    # The hint is an upper bound of the batch's context lengths; contexts stay
    # small while the hint selects the 16K window.
    plan = prepare_sm120_paged_mqa_logits(*args, max_context_hint=HINT_C16K)
    assert (plan.num_sms, plan.device_sms) == (170, 170)
    assert plan.route_name == T0 and plan.route is _runtime.route_record(T0)
    assert plan.atom_variant == "a2" and plan.num_next_n_atoms == 2
    assert plan.dyn_schedule is None
    shipped = _runtime._catalog()["routes"]["sm120:fp8:h32:p64:n4:a2"]["sequence"]
    assert plan.program_names == [shipped]
    assert plan.launch_count == 1 and plan.kernel_launches == 2
    default = prepare_sm120_paged_mqa_logits(*args)
    assert default.route_name == "sm120:fp8:h32:p64:n4:a4"
    assert default.num_next_n_atoms == 1
    # A CTA budget other than the device's SM count keeps the default route.
    other = prepare_sm120_paged_mqa_logits(
        *args, max_context_hint=HINT_C16K, sm_count=169
    )
    assert other.route_name == "sm120:fp8:h32:p64:n4:a4"
    plan.output.fill_(float("nan"))
    default.output.fill_(float("nan"))
    plan.run()
    default.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    inside = _inside(data)
    assert torch.equal(plan.logical_output[inside], default.logical_output[inside])
    expected = _schedule_reference(data["context_lens"], _runtime.split_kv(), 2, 170)
    assert plan.schedule_meta.cpu().tolist() == expected
    # The metadata entry follows the device route for 32 heads; the head
    # counts disagree there (64 heads keep one native atom), so the
    # head-count-free call names ``num_heads``.
    meta = get_paged_mqa_logits_metadata(
        data["context_lens"],
        page_kv,
        170,
        num_heads=heads,
        max_context_hint=HINT_C16K,
    )
    assert meta.cpu().tolist() == expected
    with pytest.raises(ValueError, match="num_heads"):
        get_paged_mqa_logits_metadata(
            data["context_lens"], page_kv, 170, max_context_hint=HINT_C16K
        )
    one_shot = fp8_paged_mqa_logits(
        *args[:5], meta, data["max_context_len"], max_context_hint=HINT_C16K
    )
    assert torch.equal(one_shot[inside], plan.logical_output[inside])


def test_static_device_route_graph_replay(monkeypatch):
    """CUDA Graph replay of a device route follows changed tensor contents."""
    arch = _sm120_arch()
    _use(monkeypatch, _synthetic_catalog())
    monkeypatch.setattr(_runtime, "device_facts", lambda index: (arch, 170))
    data = _inputs(32, 64, 4, 48, 512, seed=43)
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
        max_context_hint=HINT_C16K,
    )
    assert plan.route_name == T0
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    eager = plan.logical_output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    with torch.cuda.stream(stream):
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    inside = _inside(data)
    assert torch.equal(plan.logical_output[inside], eager[inside])
    with torch.cuda.stream(stream):
        data["weights"].neg_()
        data["context_lens"].copy_((data["context_lens"] - 3).clamp_min(1))
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)


def test_dyn_route_binds_the_private_schedule(monkeypatch):
    """A dyn route binds its plan-private schedule to both stages (no build, no launch).

    The caller's ``schedule_meta`` keeps the public ``[num_sms + 1, 2]`` shape,
    is checked, and is never bound, so a dyn launch cannot write it; the
    private buffer is allocated once at prepare time.
    """
    arch = _sm120_arch()
    catalog = _use(monkeypatch, _synthetic_catalog())
    monkeypatch.setattr(_runtime, "device_facts", lambda index: (arch, 188))
    calls = []

    def load_program(arch_, name):
        record = catalog["programs"][name]
        assert name == DYN_SEQUENCE
        module = {record["ffi_entry"]: lambda *args: calls.append(args)}
        return module, record

    monkeypatch.setattr(_runtime, "load_program", load_program)
    heads, page_kv, next_n, batch = 64, 64, 6, 256
    data = _inputs(heads, page_kv, next_n, batch, 256, seed=47)
    caller = torch.full((189, 2), -7, dtype=torch.int32, device="cuda")
    plan = prepare_sm120_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
        schedule_meta=caller,
        max_context_hint=HINT_C16K,
    )
    assert plan.route_name == T3 and plan.schedule == DYN
    assert plan.atom_variant == "a6" and plan.num_next_n_atoms == 1
    assert plan.schedule_meta is caller
    walk = plan.dyn_schedule
    assert walk.dtype == torch.int32 and walk.device == caller.device
    assert tuple(walk.shape) == _runtime.dyn_schedule_shape(188, 8) == (1694, 2)
    ((_entry, arguments),) = plan._submissions
    keys = [key for _kind, key in catalog["programs"][DYN_SEQUENCE]["arg_plan"]]
    bound = dict(zip(keys, arguments, strict=True))
    for key in ("metadata.schedule_meta", "logits.schedule_meta"):
        assert bound[key].data_ptr() == walk.data_ptr()
        assert bound[key].numel() == walk.numel()
    assert (
        bound["metadata.dyn_slots"],
        bound["metadata.dyn_pct"],
        bound["metadata.dyn_seg"],
    ) == (8, 3, 1)
    assert bound["metadata.num_next_n_atoms"] == 1
    assert bound["metadata.num_sms"] == 188 and bound["logits.grid_x"] == 188
    pointers = {a.data_ptr() for a in arguments if isinstance(a, torch.Tensor)}
    assert caller.data_ptr() not in pointers
    plan.run()
    assert len(calls) == 1
    # The one-shot entry checks the caller's buffer and schedules privately.
    one_shot_calls = len(calls)
    fp8_paged_mqa_logits(
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        caller,
        data["max_context_len"],
        max_context_hint=HINT_C16K,
    )
    assert len(calls) == one_shot_calls + 1
    assert caller.data_ptr() not in {
        a.data_ptr() for a in calls[-1] if isinstance(a, torch.Tensor)
    }
    torch.cuda.synchronize()
    assert bool((caller == -7).all())
    with pytest.raises(ValueError, match="schedule_meta"):
        prepare_sm120_paged_mqa_logits(
            data["q"],
            data["kv_cache"],
            data["weights"],
            data["context_lens"],
            data["block_table"],
            data["max_context_len"],
            schedule_meta=torch.zeros((1694, 2), dtype=torch.int32, device="cuda"),
            max_context_hint=HINT_C16K,
        )


# The grid context of each window: a caller's hint there is 1.3 x that context.
WINDOW_CONTEXTS = (4096, 16384, 65536)


def _route_hint(case, heads, page_kv, next_n, window):
    """A ``max_context_hint`` inside ``window`` at which the case's route applies.

    A table keys its window by the policy atom of the grid's hint (1.3 x the
    window's grid context); inside a window the default atom can change with
    the hint (the 16K window's lower part at batch 2..8), where the default
    route runs.  So the grid's hint comes first, then the window's bounds.
    """
    route, sms, _key, batch = case
    low, high = (int(bound) for bound in _runtime.device_rule()["ctx_windows"][window])
    grid = [int(1.3 * c) for c in WINDOW_CONTEXTS if low <= int(1.3 * c) < high]
    for hint in (*grid, low, high - 1):
        found = _runtime.resolve_route(
            heads, page_kv, next_n, batch, hint, cta_budget=sms, device_sms=sms
        )
        if found == route:
            return hint
    raise AssertionError(f"{route}: no hint in {window} [{low}, {high}) resolves to it")


def _committed_route_cases():
    """One table entry per committed device route: ``(route, sms, key, batch)``."""
    rule = _runtime._catalog()["policy"]["device_rule"]
    cases = {}
    for sms, table in sorted(rule["tables"].items()):
        for key, entries in sorted(table.items()):
            for batch, route in entries:
                if route is not None and route not in cases:
                    cases[route] = (route, int(sms), key, int(batch))
    return [cases[route] for route in sorted(cases)]


@pytest.mark.parametrize(
    "case",
    _committed_route_cases()
    or [pytest.param(None, marks=pytest.mark.skip(reason="no committed device route"))],
    ids=lambda case: case[0] if case else "none",
)
def test_committed_device_route_matches_the_default_route(monkeypatch, case):
    """Every committed device route, driven by its table entry, scores the default route's bits.

    The device SM count is monkeypatched to the table's key and the hint is one
    inside the window at which the route applies (:func:`_route_hint`), so the
    route runs on either SM120 device; contexts stay small.  A ``dyn`` route
    also runs twice to the same bits, leaves its counter word zero after every
    launch and leaves the caller's ``schedule_meta`` unwritten.
    """
    route, sms, key, batch = case
    arch = _sm120_arch()
    monkeypatch.setattr(_runtime, "device_facts", lambda index: (arch, sms))
    heads, page_kv, next_n, _atom, window = TABLE_KEY.match(key).groups()
    heads, page_kv, next_n = int(heads), int(page_kv), int(next_n)
    hint = _route_hint(case, heads, page_kv, next_n, window)
    data = _inputs(heads, page_kv, next_n, batch, 256, seed=53)
    caller = torch.full((sms + 1, 2), -7, dtype=torch.int32, device="cuda")
    args = (
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan = prepare_sm120_paged_mqa_logits(
        *args, schedule_meta=caller, max_context_hint=hint
    )
    record = _runtime.route_record(route)
    assert plan.route_name == route and plan.route is record
    assert plan.program_names == [record["sequence"]]
    assert plan.launch_count == 1 and plan.kernel_launches == 2
    default = prepare_sm120_paged_mqa_logits(*args)
    assert default.route_name == _runtime.route_name(heads, page_kv, next_n, batch)
    plan.output.fill_(float("nan"))
    default.output.fill_(float("nan"))
    plan.run()
    default.run()
    torch.cuda.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    inside = _inside(data)
    assert torch.equal(plan.logical_output[inside], default.logical_output[inside])
    if plan.schedule["kind"] == "dyn":
        assert _counter_word(plan) == 0
        first = plan.logical_output.clone()
        plan.output.fill_(float("nan"))
        plan.run()
        torch.cuda.synchronize()
        assert torch.equal(plan.logical_output[inside], first[inside])
        assert _counter_word(plan) == 0
        assert bool((caller == -7).all())


def _counter_word(plan):
    """Ticket counter word of a ``dyn`` plan's private schedule (zero after a launch)."""
    return int(plan.dyn_schedule[-1, 0])


def _schedule_kind(route):
    return _runtime.route_schedule(_runtime.route_record(route))["kind"]


def _replay_cases():
    """The first committed static and the first committed ``dyn`` device route."""
    picked = {}
    for case in _committed_route_cases():
        picked.setdefault(_schedule_kind(case[0]), case)
    return [picked[kind] for kind in ("static", "dyn") if kind in picked]


def _committed_route_plan(monkeypatch, case, seed):
    """Inputs, the caller's poisoned ``schedule_meta`` and the plan of ``case``."""
    route, sms, key, batch = case
    arch = _sm120_arch()
    monkeypatch.setattr(_runtime, "device_facts", lambda index: (arch, sms))
    heads, page_kv, next_n, _atom, window = TABLE_KEY.match(key).groups()
    heads, page_kv, next_n = int(heads), int(page_kv), int(next_n)
    hint = _route_hint(case, heads, page_kv, next_n, window)
    data = _inputs(heads, page_kv, next_n, batch, 256, seed=seed)
    caller = torch.full((sms + 1, 2), -7, dtype=torch.int32, device="cuda")
    args = (
        data["q"],
        data["kv_cache"],
        data["weights"],
        data["context_lens"],
        data["block_table"],
        data["max_context_len"],
    )
    plan = prepare_sm120_paged_mqa_logits(
        *args, schedule_meta=caller, max_context_hint=hint
    )
    assert plan.route_name == route
    return data, args, caller, hint, plan


@pytest.mark.parametrize(
    "case",
    _replay_cases()
    or [pytest.param(None, marks=pytest.mark.skip(reason="no committed device route"))],
    ids=lambda case: case[0] if case else "none",
)
def test_committed_device_route_graph_replay(monkeypatch, case):
    """CUDA Graph replay of a committed device route follows changed tensor contents.

    One static and one ``dyn`` route: the replay scores the eager bits, then on
    changed weights and context lengths the reference and the default route's
    bits.  A ``dyn`` route's counter word is zero after every replay and the
    caller's ``schedule_meta`` stays unwritten.
    """
    data, args, caller, _hint, plan = _committed_route_plan(monkeypatch, case, 59)
    dyn = plan.schedule["kind"] == "dyn"
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    eager = plan.logical_output.clone()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
    with torch.cuda.stream(stream):
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    inside = _inside(data)
    assert torch.equal(plan.logical_output[inside], eager[inside])
    if dyn:
        assert _counter_word(plan) == 0
    with torch.cuda.stream(stream):
        data["weights"].neg_()
        data["context_lens"].copy_((data["context_lens"] - 3).clamp_min(1))
        plan.output.fill_(float("nan"))
        graph.replay()
    stream.synchronize()
    _check(plan.logical_output, _reference(data), data, plan.output)
    default = prepare_sm120_paged_mqa_logits(*args)
    default.output.fill_(float("nan"))
    default.run()
    torch.cuda.synchronize()
    inside = _inside(data)
    assert torch.equal(plan.logical_output[inside], default.logical_output[inside])
    if dyn:
        assert _counter_word(plan) == 0
        assert bool((caller == -7).all())


@pytest.mark.parametrize(
    "case",
    [case for case in _replay_cases() if _schedule_kind(case[0]) == "dyn"]
    or [pytest.param(None, marks=pytest.mark.skip(reason="no committed dyn route"))],
    ids=lambda case: case[0] if case else "none",
)
def test_committed_dyn_route_one_shot_entry(monkeypatch, case):
    """The one-shot entry on a committed ``dyn`` route scores the plan's bits.

    It schedules privately: the caller's ``schedule_meta`` stays unwritten.
    """
    data, args, caller, hint, plan = _committed_route_plan(monkeypatch, case, 61)
    assert plan.schedule["kind"] == "dyn"
    plan.output.fill_(float("nan"))
    plan.run()
    torch.cuda.synchronize()
    one_shot = fp8_paged_mqa_logits(
        *args[:5], caller, data["max_context_len"], max_context_hint=hint
    )
    torch.cuda.synchronize()
    inside = _inside(data)
    assert torch.equal(one_shot[inside], plan.logical_output[inside])
    assert _counter_word(plan) == 0
    assert bool((caller == -7).all())
