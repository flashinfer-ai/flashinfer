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

CPU-only tests of the Cake Gated DeltaProduct (GDP) prefill body route of
record (``cake_jit``, the BODY ROUTE OF RECORD block): the per-sequence choice
between the V6 fork-order body and the merged chunked body, the launch plan of
a batch, the sm_90a ``prefill`` registry and the identities of its records.
The route is host arithmetic over Python integers: no GPU, no tensor, no
device fixture.  ``test_cake_gdp_prefill.py`` runs the kernels.
"""

from __future__ import annotations

import inspect
import itertools
import re

import pytest

from tests.test_helpers.cake_package import cake_package

cake_build = cake_package("cake_gdp", "cake_build")
loader = cake_package("cake_gdp", "cake_jit")

P3 = loader.ROUTE_BODY_P3
V6 = loader.ROUTE_BODY_V6
ARCH = "sm_90a"
SM_COUNT = 132  # H100 SXM / H200
N_HH = 3  # num_householder of the performance rows
# The 18 prefill performance rows: (H) x (N, L) at n = 3; every row is a uniform batch.  The merged class takes the
# single long sequences at H = 10 and H = 24 (its split items are the cheaper); every other row is V6-class.
PERF_HEADS = (80, 10, 24)
PERF_NL = ((1, 1024), (1, 8192), (1, 32768), (4, 4096), (16, 1024), (64, 256))
MERGED_ROWS = {(10, 1, 8192), (10, 1, 32768), (24, 1, 8192), (24, 1, 32768)}
HEADS = (1, 2, 3, 6, 10, 12, 20, 24, 40, 80)
SM_COUNTS = (132, 114, 148, 152)
# The sm_90a prefill route: the three V6 programs (plain, initial state, initial state + slot pool), the two unit
# programs and the three split programs of the merged body, in route order.
V6_NAMES = (
    "cake_gdp_0de8fd0ee57885709a1c",
    "cake_gdp_c81addff1e986f49ec99",
    "cake_gdp_55a7b3c2baf155618309",
)
UNIT_NAMES = ("cake_gdp_c8580552e73f2264f8eb", "cake_gdp_f39daf3dd3cf89844fc1")
SPLIT_NAMES = (
    "cake_gdp_3220d37914678385a837",
    "cake_gdp_245414df8449fb1825a1",
    "cake_gdp_b81854850bf4b769295c",
)
# (use_initial_state, use_state_indices, store_final_state, unit_gates, unit_beta) per program group.
V6_FLAGS = (
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
)
UNIT_FLAGS = (
    (False, False, True, True, False),
    (False, False, True, False, True),
)
SPLIT_FLAGS = V6_FLAGS
# The merged body's exact gate-tensor programs the route no longer carries: plain, h0, pool, no-final.
REMOVED_FLAGS = (
    (False, False, True, False, False),
    (True, False, True, False, False),
    (True, True, True, False, False),
    (False, False, False, False, False),
)
MIXED_LENS = (8192, 1000, 0, 8192)


def _offsets(lens: tuple[int, ...]) -> tuple[int, ...]:
    return (0, *itertools.accumulate(lens))


def _classes(
    lens: tuple[int, ...], H: int, n: int = N_HH, sm_count: int = SM_COUNT, **flags
) -> list[str]:
    return loader.prefill_body_classes(_offsets(lens), n, H, sm_count, **flags)


def _select(body: str, flags: tuple[bool, ...], **extra: bool) -> str:
    use_initial, use_indices, store, unit_g, unit_beta = flags
    return loader.selected_prefill_module(
        ARCH,
        body=body,
        use_initial_state=use_initial,
        use_state_indices=use_indices,
        store_final_state=store,
        unit_gates=unit_g,
        unit_beta=unit_beta,
        **extra,
    )


# ---------------------------------------------------------------------------
# The performance rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("H", PERF_HEADS)
@pytest.mark.parametrize("N,L", PERF_NL)
def test_performance_rows_are_uniform_batches(H: int, N: int, L: int) -> None:
    """Every performance row is one launch: the merged class exactly on the single long sequences at H = 10 and
    H = 24, the V6 class everywhere else."""
    classes = _classes((L,) * N, H)
    body = P3 if (H, N, L) in MERGED_ROWS else V6
    assert classes == [body] * N, (H, N, L, classes)
    assert loader.prefill_launch_plan(classes) == [(body, list(range(N)))]


def test_pinned_body_costs() -> None:
    """The costs of record of a 384-chunk sequence (L = 8192 at n = 3) on 132 SMs, in tenths of a V6 chunk-step:
    the merged split items win at H = 10 and lose at H = 80 (38 and 272 item waves at 3.6 V6 steps each); an empty sequence costs nothing under either body."""
    assert loader.prefill_body_costs(384, 10, N_HH, SM_COUNT) == (3840, 1368)
    assert loader.prefill_body_costs(384, 80, N_HH, SM_COUNT) == (3840, 9792)
    assert loader.prefill_body_costs(0, 10, N_HH, SM_COUNT) == (0, 0)
    assert loader.ROUTE_MIN_CHUNKS_P3 == 384
    assert loader.ROUTE_BODIES == (P3, V6)


# ---------------------------------------------------------------------------
# Class properties
# ---------------------------------------------------------------------------

CHUNK_COUNTS = (0, 2, 12, 48, 192, 382, 384, 386, 400, 512, 768, 1024, 1536, 4096)


@pytest.mark.parametrize("unit_gates,unit_beta", ((True, False), (False, True)))
def test_unit_calls_are_merged_class_at_every_length(
    unit_gates: bool, unit_beta: bool
) -> None:
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        assert (
            loader.prefill_body_class(
                chunks, H, n, sm_count, unit_gates=unit_gates, unit_beta=unit_beta
            )
            == P3
        ), (chunks, H, n, sm_count)


def test_calls_without_a_final_state_are_v6_class_at_every_length() -> None:
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        assert (
            loader.prefill_body_class(chunks, H, n, sm_count, store_final_state=False)
            == V6
        ), (chunks, H, n, sm_count)


def test_short_sequences_are_v6_class() -> None:
    """Every chunk count below ``ROUTE_MIN_CHUNKS_P3`` (the empty sequence included) is V6-class, whatever the
    head count, the Householder count and the SM count."""
    for chunks in range(loader.ROUTE_MIN_CHUNKS_P3):
        for H, n, sm_count in itertools.product(
            HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
        ):
            assert loader.prefill_body_class(chunks, H, n, sm_count) == V6, (
                chunks,
                H,
                n,
                sm_count,
            )


def test_merged_class_sequences_are_cut() -> None:
    """A merged-class sequence is cut by the split rule of record (``split_count`` gives ``p >= 2``): the merged
    class always runs a split program.  A merged-class sequence is also at least ``ROUTE_MIN_CHUNKS_P3`` chunks
    long, and some long sequence is merged-class."""
    merged = 0
    for chunks, H, n, sm_count in itertools.product(
        CHUNK_COUNTS, HEADS, loader.NUM_HOUSEHOLDER, SM_COUNTS
    ):
        if loader.prefill_body_class(chunks, H, n, sm_count) != P3:
            continue
        merged += 1
        assert chunks >= loader.ROUTE_MIN_CHUNKS_P3
        p, span = loader.split_count(chunks, H, n, sm_count)
        assert p >= 2 and 0 < span < chunks, (chunks, H, n, sm_count, p, span)
        cost_v6, cost_p3 = loader.prefill_body_costs(chunks, H, n, sm_count)
        assert cost_p3 < cost_v6
    assert merged > 0


@pytest.mark.parametrize("H", PERF_HEADS)
def test_class_of_a_sequence_is_independent_of_the_batch(H: int) -> None:
    """A sequence runs the same body alone and in any batch, in any position."""
    lens = (8192, 1000, 0, 8192, 32768, 4096, 1, 129)
    alone = [_classes((L,), H)[0] for L in lens]
    assert alone == [
        loader.prefill_body_class(loader.split_chunks(L, N_HH), H, N_HH, SM_COUNT)
        for L in lens
    ]
    assert _classes(lens, H) == alone
    for order in (
        tuple(reversed(range(len(lens)))),
        (4, 0, 7, 2, 5, 1, 6, 3),
        (3, 3, 0, 0, 2),
    ):
        assert _classes(tuple(lens[i] for i in order), H) == [alone[i] for i in order]


def test_route_chunk_steps_clamp_like_the_split_rules() -> None:
    """``route_chunk_steps`` reads ``cu_seqlens`` as the device does: the start clamped to ``[0, total_tokens]``,
    the end to ``[start, total_tokens]``, so a decreasing or negative pair is an empty sequence."""
    cu, n, total = [20, 300, -5, 10], 3, 180
    steps = loader.route_chunk_steps(cu, n, total_tokens=total)
    assert steps == [8, 0, 2]
    assert steps == [
        loader.split_chunks(end - start, n)
        for start, end in (
            loader.split_sequence_bounds(cu[i], cu[i + 1], total)
            for i in range(len(cu) - 1)
        )
    ]
    # the item table of the same batch reads the same lengths: one item per (sequence, head) under the exact mode
    table = loader.split_table(
        cu, n, 2, SM_COUNT, split_mode="exact", total_tokens=total
    )
    assert [span for seq, head, cut, span in table if head == 0] == steps
    mixed = loader.route_chunk_steps([0, 8192, 9192, 9192, 17384], n)
    assert mixed == [384, 48, 0, 384]
    assert loader.route_chunk_steps([0], n) == []


# ---------------------------------------------------------------------------
# The launch plan
# ---------------------------------------------------------------------------


def test_launch_plan() -> None:
    """No sequence -> no launch; a uniform batch -> one launch over every sequence; a mixed batch -> one launch per
    body in ``ROUTE_BODIES`` order (the merged body first), each over the ascending indices of its sequences."""
    assert loader.prefill_launch_plan([]) == []
    assert loader.prefill_launch_plan([V6] * 4) == [(V6, [0, 1, 2, 3])]
    assert loader.prefill_launch_plan([P3] * 2) == [(P3, [0, 1])]
    assert loader.prefill_launch_plan([P3, V6, V6, P3]) == [(P3, [0, 3]), (V6, [1, 2])]
    assert loader.prefill_launch_plan([V6, P3]) == [(P3, [1]), (V6, [0])]


def test_mixed_batch_of_record() -> None:
    """The batch (8192, 1000, 0, 8192) at n = 3 on 132 SMs: mixed at H = 10 (the long sequences merged-class, the
    short and the empty one V6-class), uniform V6 at H = 80, uniform merged with unit gates."""
    classes = _classes(MIXED_LENS, 10)
    assert classes == [P3, V6, V6, P3]
    assert loader.prefill_launch_plan(classes) == [(P3, [0, 3]), (V6, [1, 2])]
    assert _classes(MIXED_LENS, 80) == [V6] * 4
    assert _classes(MIXED_LENS, 10, unit_gates=True) == [P3] * 4
    assert _classes(MIXED_LENS, 10, unit_beta=True) == [P3] * 4
    assert _classes(MIXED_LENS, 10, store_final_state=False) == [V6] * 4
    # the compact offsets of the merged class are cut by the split rule: the mixed merged launch has a table
    merged = (0, MIXED_LENS[0], MIXED_LENS[0] + MIXED_LENS[3])
    table = loader.prefill_split_table(
        merged, num_heads=10, num_householder=N_HH, sm_count=SM_COUNT
    )
    assert table is not None and len(table) > 2 * 10
    assert {seq for seq, head, cut, span in table} == {0, 1}


# ---------------------------------------------------------------------------
# The registry
# ---------------------------------------------------------------------------


def test_sm90_prefill_route_lists_the_eight_programs() -> None:
    route = loader.ROUTES[ARCH][loader.PREFILL_ROUTE]
    assert route == [*V6_NAMES, *UNIT_NAMES, *SPLIT_NAMES]
    assert loader.prefill_route_modules(ARCH) == tuple(route)
    for name in V6_NAMES:
        assert loader.MODULES[name]["body"] == V6
        assert not loader.MODULES[name]["build"].get(loader.PREFILL_SPLIT_FLAG)
    for name in UNIT_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == P3
        assert record["build"][loader.PREFILL_SPLIT_FLAG] is False
        assert record["build"]["unit_gates"] != record["build"]["unit_beta"]
    for name in SPLIT_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == P3
        assert record["build"][loader.PREFILL_SPLIT_FLAG] is True
    # no other record of the prefill role is delivered for sm_90a
    assert {
        name
        for name, record in loader.MODULES.items()
        if record["role"] == "prefill" and ARCH in record["arches"]
    } == set(route)


def test_selected_prefill_module_filters_by_body() -> None:
    """The body filter resolves the three V6 flag sets, the two unit sets and the three split sets to their
    records and refuses the removed merged exact programs with ``0 programs``; no body searches every body."""
    for flags, name in zip(V6_FLAGS, V6_NAMES, strict=True):
        assert _select(V6, flags) == name
    for flags, name in zip(UNIT_FLAGS, UNIT_NAMES, strict=True):
        assert _select(P3, flags) == name
    for flags, name in zip(SPLIT_FLAGS, SPLIT_NAMES, strict=True):
        assert _select(P3, flags, **{loader.PREFILL_SPLIT_FLAG: True}) == name
    for flags in REMOVED_FLAGS:
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            _select(P3, flags)
    # no body (None) searches every body of the route: the plain flags resolve to the V6 plain program alone
    assert (
        loader.selected_prefill_module(
            ARCH,
            use_initial_state=False,
            use_state_indices=False,
            store_final_state=True,
        )
        == V6_NAMES[0]
    )
    for flags in (*UNIT_FLAGS, (False, False, False, False, False)):
        with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
            _select(V6, flags)
    with pytest.raises(ValueError, match="0 programs of the 'prefill' route"):
        _select(V6, V6_FLAGS[0], **{loader.PREFILL_SPLIT_FLAG: True})


def test_v6_record_names_are_the_kernel_digests() -> None:
    """A V6 record is named ``cake_gdp_<20 hex>``, the digest its kernel was rendered under; the record's kernel,
    library and symbol carry that name, the three names differ, and the build manifest lists each library for
    sm_90a."""
    libraries = cake_build.libraries_by_name()
    assert len(set(V6_NAMES)) == len(V6_NAMES)
    for name in V6_NAMES:
        record = loader.MODULES[name]
        assert re.fullmatch(r"cake_gdp_[0-9a-f]{20}", name), name
        assert record["kernel"] == record["library"] == name
        assert record["kernel_symbol"] == f"kernel_{name}"
        assert f"{name}_sm_90a" in libraries, name


def test_v6_records_share_the_merged_launch_contract() -> None:
    """A V6 record launches 384 threads per CTA under the ``tiles`` grid rule and takes the merged body's 23
    formals in the merged body's order, so the backend binds one ``values`` dict for every launch."""
    merged = loader.MODULES[UNIT_NAMES[0]]
    for name in V6_NAMES:
        record = loader.MODULES[name]
        assert record["body"] == V6 and record["arches"] == [ARCH]
        assert record["launch"]["block"] == (384, 1, 1)
        assert record["grid_rule"] == {"kind": "tiles"}
        assert record["arg_plan"] == merged["arg_plan"]
        assert record["logical"] == merged["logical"]
        assert set(record["logical"].values()) == set(
            loader.PREFILL_LOGICAL_TENSORS
        ) | set(loader.PREFILL_LOGICAL_SCALARS)
        assert record["select_rule"]["kind"] == "prefill_flags"
        assert set(record["select_rule"]) == {"kind", *loader.PREFILL_FLAGS}
        assert record["select_rule"]["store_final_state"] is True
        assert {flag: record["build"][flag] for flag in loader.PREFILL_FLAGS} == {
            flag: record["select_rule"][flag] for flag in loader.PREFILL_FLAGS
        }
    assert merged["launch"]["block"] == (512, 1, 1)


def test_body_route_block_sits_between_the_split_rules_and_the_table() -> None:
    """The BODY ROUTE OF RECORD block of ``cake_jit.py`` (one text in every copy) follows the END SPLIT RULES marker
    and precedes ``prefill_split_table``."""
    source = inspect.getsource(loader)
    end_split = source.index("# ---- END SPLIT RULES OF RECORD ----")
    start = source.index("# ---- BODY ROUTE OF RECORD")
    end = source.index("# ---- END BODY ROUTE OF RECORD ----")
    table = source.index("def prefill_split_table(")
    assert end_split < start < end < table
    block = source[start:end]
    for symbol in (
        "ROUTE_BODY_P3",
        "ROUTE_BODY_V6",
        "ROUTE_BODIES",
        "ROUTE_MIN_CHUNKS_P3",
        "def route_chunk_steps(",
        "def prefill_body_costs(",
        "def prefill_body_class(",
        "def prefill_body_classes(",
        "def prefill_launch_plan(",
    ):
        assert symbol in block, symbol
    for symbol in (
        "ROUTE_BODIES",
        "ROUTE_BODY_P3",
        "ROUTE_BODY_V6",
        "prefill_body_class",
        "prefill_body_classes",
        "prefill_body_costs",
        "prefill_launch_plan",
        "route_chunk_steps",
    ):
        assert symbol in loader.__all__, symbol
