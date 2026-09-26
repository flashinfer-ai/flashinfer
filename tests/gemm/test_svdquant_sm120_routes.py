"""CPU-only contracts for the SM120 linear route manifest.

Routes only. Tactics are not manifest data: they are measurements, and the
only thing allowed to produce one is the tuner;
``test_no_module_ships_a_shape_keyed_tactic_table`` is the guard that keeps it
that way. The tests after it hold the manifest against the C++ producer
dispatcher in both directions, because admission and instantiation live in
different languages and drifted apart once already.
"""

import ast
import re
from pathlib import Path
from types import MappingProxyType

import pytest

from flashinfer.gemm.svdquant_sm120_routes import (
    sm120_producer_variants,
    sm120_variant_packs_l2t,
    SM120_FAMILY_M537,
    SM120_FAMILY_SMALL_M_PACKED,
    SM120_FUSED_LINEAR_MK,
    SM120_LINEAR_ROUTE_ABI_OVERRIDES,
    SM120_LINEAR_ROUTE_V6_MKR,
    sm120_linear_route_abi_version,
)


def test_route_manifest_containers_are_immutable() -> None:
    # Given: routing metadata is shared by dispatch, tests, and tuning tools.
    # When: its public container types are inspected.
    # Then: no consumer can mutate process-wide route decisions in place.
    assert isinstance(SM120_LINEAR_ROUTE_ABI_OVERRIDES, MappingProxyType)
    assert isinstance(SM120_FUSED_LINEAR_MK, frozenset)
    assert isinstance(SM120_LINEAR_ROUTE_V6_MKR, frozenset)


@pytest.mark.parametrize(
    "m,k,rank,expected_version",
    [
        (256, 12288, 32, 5),
        (73984, 14336, 32, 6),
        (1935, 14336, 32, 7),
        (537, 14336, 32, 9),
        (64, 3072, 32, 10),
        (64, 5120, 32, 10),
        (64, 5376, 32, 10),
        (64, 7168, 32, 10),
        (73984, 7168, 32, 10),
        (537, 5376, 32, 10),
        (82752, 5376, 32, 10),
        (17, 3072, 32, 10),
        (0, 3072, 32, 5),
        (64, 1536, 32, 5),
        (64, 3072, 16, 5),
        (64, 3072, 64, 11),
    ],
)
def test_route_abi_version_matches_manifest(
    m: int,
    k: int,
    rank: int,
    expected_version: int,
) -> None:
    # Given: affected and unaffected shapes, including an unlisted M and ranks.
    # When: its current cache namespace is selected.
    actual_version = sm120_linear_route_abi_version(m, k, rank)
    # Then: only runner-set changes advance beyond the retained v5 base.
    assert actual_version == expected_version


def test_route_abi_override_manifest_has_no_base_version_entries() -> None:
    # Given: the compact mapping contains only exceptions to the v5 base.
    # When: every override version is inspected.
    override_versions = set(SM120_LINEAR_ROUTE_ABI_OVERRIDES.values())
    # Then: redundant v5 entries cannot drift from the default independently.
    assert override_versions == {6, 7, 8, 9}


def test_the_two_formerly_pinned_problems_keep_their_fused_admission() -> None:
    # Given: 32760x1536x1536 and 32760x8960x1536 reached the fused runner only
    # because they carried a pinned tactic, and pinned prefixes were unioned into
    # admission separately from SM120_FUSED_LINEAR_MK.
    # When: the pins are deleted.
    # Then: their prefix has to survive in the admission set on its own, or the
    # tuner sees only Sm120CutlassLinearRunner for both -- measured slower on
    # 32760x1536x1536.
    assert (32760, 1536) in SM120_FUSED_LINEAR_MK


def _shape_keyed_tactic_tables(source: str) -> list[str]:
    """Names bound to a module-level mapping keyed by a four-int (m,n,k,rank)."""
    found: list[str] = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Assign):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            targets = [node.target.id]
        else:
            continue
        value = node.value
        if value is None:
            continue
        # MappingProxyType({...}) and frozenset({...}) both wrap the literal.
        while isinstance(value, ast.Call) and value.args:
            value = value.args[0]
        if not isinstance(value, ast.Dict):
            continue
        for key in value.keys:
            if not isinstance(key, ast.Tuple) or len(key.elts) != 4:
                break
            if not all(
                isinstance(e, ast.Constant) and isinstance(e.value, int)
                for e in key.elts
            ):
                break
        else:
            if value.keys:
                found.extend(targets)
    return found


def test_no_module_ships_a_shape_keyed_tactic_table() -> None:
    # Given: a tactic is a measurement of one kernel on one device, and the only
    # thing entitled to produce one is the tuner. A cache is fine; a source
    # literal is not, because nothing re-checks it and nothing dates it. One went
    # stale and was found only by measuring the whole candidate field.
    # When: every module of the gemm package is parsed.
    # Then: no module-level mapping is keyed by an exact (M, N, K, rank) problem,
    # which is the shape such a table has to have to be looked up at dispatch.
    root = Path(__file__).resolve().parents[2]
    package = root / "flashinfer" / "gemm"
    # Every SVDQuant module under flashinfer/gemm. The CuTeDSL kernels this
    # guard used to also scan are gone with that route; what replaced them as
    # the tempting place to write such a table is the SM120 CUTLASS backend,
    # which owns the route manifest and the runner set.
    offenders: dict[str, list[str]] = {}
    scanned = sorted(package.rglob("*svdquant*.py"))
    covered = {path.name for path in scanned}
    for expected in (
        "svdquant_sm120_routes.py",
        "svdquant_sm120_cutlass.py",
        "gemm_svdquant.py",
    ):
        assert expected in covered, (
            f"{expected} is no longer scanned; the guard passes vacuously for it"
        )
    for path in scanned:
        names = _shape_keyed_tactic_tables(path.read_text())
        if names:
            offenders[str(path.relative_to(root))] = names
    assert offenders == {}, (
        f"a shape-keyed table is back in source: {offenders}. Tactics come from "
        "the tuner and persist through AutoTuner.save_configs / load_configs."
    )


def _has_native_producer(m: int, k: int) -> bool:
    """Whether the shape has a legal producer geometry."""
    return bool(sm120_producer_variants(m, k))


def _specialized_producer_shapes(header: str) -> set[tuple[int, int]]:
    """Read the remaining fixed-M/K instantiations from the family dispatcher."""
    start = header.index(
        "inline cudaError_t nvfp4_smooth_quantize_lora_down_family_sm120("
    )
    end = header.index(
        "inline cudaError_t nvfp4_smooth_quantize_lora_down_sm120(", start
    )
    return {
        (537, int(match.group(1)))
        for match in re.finditer(
            r"launch_m537_mixed_kernel<\s*(\d+)", header[start:end]
        )
    }


def _producer_header() -> str:
    root = Path(__file__).resolve().parents[2]
    return (
        root
        / "include"
        / "flashinfer"
        / "gemm"
        / "nvfp4_smooth_quantize_lora_down_sm120.cuh"
    ).read_text()


def test_every_fused_admission_has_a_prefix_that_can_run() -> None:
    # Given: admitting (M, K) to SM120_FUSED_LINEAR_MK offers the fused runner to
    # the tuner, and the fused route reaches its prefix in one of two ways -- a
    # compiled native producer, or the cuBLASLt prefix the ladder can select.
    # When: the admission set is checked against both.
    # Then: nothing is admitted that has neither, which would make the C++
    # dispatcher return cudaErrorInvalidValue for a route the tuner can pick.
    # Reachable = the FFI lets it in AND the shape computes a geometry to launch.
    # The second term used to be the ladder's cuBLASLt column, for shapes that
    # had the prefix but no native producer. The prefix is a producer family now
    # and is offered exactly where the native families are, so a shape reachable
    # by one is reachable by the other and the ladder term has nothing to add.
    reachable = {s for s in SM120_FUSED_LINEAR_MK if _has_native_producer(*s)}
    orphans = sorted(set(SM120_FUSED_LINEAR_MK) - reachable)
    assert orphans == [], (
        f"{orphans} are admitted to the fused route but have no native producer "
        "and no cuBLASLt prefix rung"
    )


def test_every_specialized_producer_is_admitted() -> None:
    specialized = _specialized_producer_shapes(_producer_header())
    assert specialized, "the source scan must cover the fixed-M/K M537 family"
    assert specialized <= set(SM120_FUSED_LINEAR_MK)
    for m, k in specialized:
        assert any(
            family == SM120_FAMILY_M537
            for family, _, _ in sm120_producer_variants(m, k)
        ), f"the compiled M537 producer for {(m, k)} cannot be selected"


@pytest.mark.parametrize(
    "m,k",
    [(512, 1536), (7800, 1536), (256, 3072), (6889, 3072), (9216, 3072), (16384, 3072)],
)
def test_swept_producers_are_admitted_and_route_native(m: int, k: int) -> None:
    # Given: six prefixes gained a producer because a full sweep of the legal
    # launch geometries measured the fused producer beating the unfused prefix
    # by a measured margin.
    # When: each is resolved through the manifest.
    # Then: it has native candidates and no fixed-M537 candidate.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    # Was `(m, k) in _ffi_guard_shapes()`. The FFI no longer carries a shape
    # list; being runnable is being able to compute a producer geometry.
    assert sm120_producer_variants(m, k)
    # Packed small-M candidates may coexist with row-major native candidates;
    # the selected family determines the L2T layout.
    variants = sm120_producer_variants(m, k)
    assert variants, f"({m}, {k}) computes no producer geometry"
    assert all(family != SM120_FAMILY_M537 for family, _, _ in variants)
    for variant, (family, _, _) in enumerate(variants):
        assert sm120_variant_packs_l2t(m, k, variant) == (
            family == SM120_FAMILY_SMALL_M_PACKED
        )


@pytest.mark.parametrize("m,k", sorted(SM120_FUSED_LINEAR_MK | {(17, 3072)}))
def test_packed_family_is_appended_after_all_legacy_variants(m: int, k: int) -> None:
    # Given: old producer IDs are persisted as part of each packed tactic.
    # When: the producer list admits the new layout family.
    variants = sm120_producer_variants(m, k)
    packed_indices = [
        index
        for index, (family, _, _) in enumerate(variants)
        if family == SM120_FAMILY_SMALL_M_PACKED
    ]
    # Then: new entries form a suffix, preserving all previous indices.
    if packed_indices:
        assert packed_indices == list(range(packed_indices[0], len(variants)))


@pytest.mark.parametrize(
    "m,k,variant,expected",
    [
        (64, 3072, 17, (1, (768, 8, 8), 0)),
        (64, 3072, 23, (3, (0, 0, 0), 0)),
        (537, 5376, 19, (2, (1024, 24, 16), 0)),
        (537, 5376, 20, (3, (0, 0, 0), 0)),
    ],
)
def test_legacy_producer_ids_keep_their_meaning(
    m: int, k: int, variant: int, expected: tuple[int, tuple[int, int, int], int]
) -> None:
    # Given: representative persisted IDs from every affected legacy family.
    # When: the current producer list resolves each index.
    actual = sm120_producer_variants(m, k)[variant]
    # Then: appending candidates changes no existing tactic interpretation.
    assert actual == expected


@pytest.mark.parametrize(
    "m,k",
    [
        (4096, 3072),
        (1024, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
        (512, 5120),
        (537, 14336),
        (1935, 5376),
    ],
)
def test_prefixes_where_fusing_measured_slower_are_offered_anyway(
    m: int, k: int
) -> None:
    # Given: the sweep measured these prefixes too, and on every one the fused
    # producer was SLOWER than the prefix the shape pays today, against both
    # the unfused prefix and the cuBLASLt one.
    # When: the manifest is checked.
    # Then: they are present regardless. Whether a candidate is worth running is
    # a measurement, and the only thing entitled to make it is the tuner; a
    # prefix left out of this table has no fused candidate for the tuner to
    # reject, and the source-side judgement that put it there never re-checks
    # itself. The measured costs live beside each dispatcher entry.
    assert (m, k) in SM120_FUSED_LINEAR_MK
    assert _has_native_producer(m, k)
    # Was `(m, k) in _ffi_guard_shapes()`. The FFI no longer carries a shape
    # list; being runnable is being able to compute a producer geometry.
    assert sm120_producer_variants(m, k)


def test_every_production_prefix_has_a_fused_producer() -> None:
    # Given: a shape whose (M, K) has no producer cannot offer the fused route
    # to the tuner at all -- not as a losing candidate, not as anything. The
    # coverage question is therefore separate from the performance question.
    # When: every registered production shape's prefix is looked up.
    # Then: all of them have one. This is the invariant that says the fused
    # implementation exists everywhere; which route each shape actually runs is
    # decided by measurement, per shape, per card.
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    prefixes = {(case[0], case[2]) for case in MODEL_SHAPE_CASES}
    missing = sorted(p for p in prefixes if not _has_native_producer(*p))
    assert missing == [], f"{missing} have no fused producer"
    assert prefixes <= set(SM120_FUSED_LINEAR_MK)
    # Same: reachability is computed, not listed.
    assert all(sm120_producer_variants(m, k) for m, k in prefixes)


# --- the producer axis of a packed tactic ------------------------------------
#
# A tactic is now (consumer row, producer variant) packed into one integer. The
# tests below hold the two halves of that grammar together: the ordering
# contract the packing rests on, its inertness where the axis is degenerate, and
# the one failure mode that is silent rather than loud.


def test_fused_linear_mk_is_the_benchmark_projection() -> None:
    """The named shape list must stay the benchmark matrix, not drift into a gate.

    SM120_FUSED_LINEAR_MK used to decide fused admission. It does not any more --
    _sm120_fused_linear_supported computes that from the launch geometry -- so
    what is left is a name for the shapes this backend was measured on. If the
    two ever diverge, the list has started encoding a decision again.
    """
    from flashinfer.testing.svdq_model_shapes import MODEL_SHAPE_CASES

    assert set(SM120_FUSED_LINEAR_MK) == {(m, k) for m, _, k in MODEL_SHAPE_CASES}
