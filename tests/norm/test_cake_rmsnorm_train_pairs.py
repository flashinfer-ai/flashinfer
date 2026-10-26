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

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (6144, 4096, 2048, 1536, 512)
REDUCE_ROLE = "bwd_reduce"
SHIPPED_SLABS = (16, 32, 64)
# Row counts of the contract: the small counts, the window edges of every route and the training shapes.
ROWS = (
    1, 2, 7, 63, 64, 65, 127, 128, 129, 255, 257, 1023, 1024, 1025, 3072, 3073, 4096, 4097,
    12288, 12289, 14928, 16231, 16263, 16384, 16385, 32768, 32769, 65536,
)  # fmt: skip


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
        and pkg.is_cake_rmsnorm_train_supported(_device())
    )


@pytest.mark.skipif(
    not _supported(),
    reason="needs an SM100 / SM103 / SM107 device with exported routes",
)
@pytest.mark.parametrize("hidden", HIDDEN_SIZES)
def test_workspace_puts_the_counters_first_and_grows_with_the_rows(hidden: int) -> None:
    if not pkg.is_cake_rmsnorm_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")
    device = _device()
    sizes = [
        pkg.cake_rmsnorm_train_backward_workspace_bytes(rows, hidden, device)
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
    workspace = pkg.cake_rmsnorm_train_backward_workspace(65536, hidden, device)
    assert workspace.dtype == torch.uint8 and workspace.numel() == sizes[-1]
    assert torch.count_nonzero(workspace[: loader.COUNTERS_BYTES]) == 0
