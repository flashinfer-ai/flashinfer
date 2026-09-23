# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import copy
import unittest
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    autotune as autotune_module,
    hopper_mxfp4,
    knob_cache,
    mxfp4_tuner,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    MXFP4_TUNING_ROUTING_PROFILES,
    MXFP4_TUNING_TOKEN_BUCKETS,
    _base_candidates,
    _ordered_base_candidates,
    hopper_mxfp4_cache_provenance_sha256,
    hopper_mxfp4_default_tactic,
    is_hopper_mxfp4_tactic_shape_compatible,
    normalize_hopper_mxfp4_routing_profile,
    validate_hopper_mxfp4_tactic,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
    expand_mxfp4_optimization_candidates,
    hopper_mxfp4_candidates,
    mxfp4_optimization_candidate_sha256,
    normalize_mxfp4_optimization_tactic,
    resolve_mxfp4_tactic_optimizations,
)

_OPTIMIZATION_SHAPE = dict(
    hidden=7168, intermediate=3072, num_experts=384, world_size=4
)

_DEFAULT_FIELDS = (
    "mma_tiler_mnk",
    "cluster_shape_mnk",
    "group_hint",
    "num_sched_stages",
    "pingpong",
    "load_balance_mode",
    "token_back_mode",
    "dedup_dispatch",
    "active_dispatch_warps",
    "fc1_store_offload",
    "fc1_early_done_publish",
    "fold_producer_warps",
)
_EXPECTED_DEFAULTS = {
    "block_permutation_v1": (
        (
            (256, 16, 256),
            (2, 1, 1),
            132,
            1,
            False,
            "static",
            "epi_warps",
            False,
            4,
            True,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            132,
            1,
            False,
            "static",
            "epi_warps",
            True,
            4,
            True,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            132,
            1,
            False,
            "static",
            "epi_warps",
            False,
            1,
            True,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            512,
            1,
            False,
            "static",
            "epi_warps",
            True,
            2,
            True,
            False,
            False,
        ),
        (
            (256, 32, 128),
            (1, 1, 1),
            330,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            2,
            True,
            False,
            False,
        ),
        (
            (256, 64, 256),
            (2, 1, 1),
            512,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            True,
            1,
            False,
            True,
            True,
        ),
        (
            (256, 64, 256),
            (2, 1, 1),
            528,
            2,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            1,
            False,
            True,
            True,
        ),
        (
            (256, 64, 256),
            (2, 1, 1),
            512,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            1,
            False,
            True,
            True,
        ),
    ),
    "published_exact_balanced_v1": (
        (
            (256, 16, 256),
            (2, 1, 1),
            396,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            512,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            512,
            1,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            512,
            2,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (256, 16, 256),
            (2, 1, 1),
            396,
            2,
            False,
            "static",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (256, 32, 128),
            (2, 1, 1),
            396,
            2,
            False,
            "atomic_counter",
            "epi_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (128, 64, 256),
            (2, 1, 1),
            330,
            2,
            False,
            "atomic_counter",
            "reuse_dispatch_warps",
            False,
            4,
            False,
            False,
            False,
        ),
        (
            (128, 64, 256),
            (1, 1, 1),
            528,
            2,
            False,
            "atomic_counter",
            "reuse_dispatch_warps",
            False,
            4,
            False,
            False,
            False,
        ),
    ),
}


@pytest.mark.parametrize("profile", MXFP4_TUNING_ROUTING_PROFILES)
def test_defaults_and_complete_candidate_order(profile):
    for token, expected in zip(
        MXFP4_TUNING_TOKEN_BUCKETS, _EXPECTED_DEFAULTS[profile], strict=True
    ):
        tactic = hopper_mxfp4_default_tactic(token, routing_profile=profile)
        assert tuple(tactic[key] for key in _DEFAULT_FIELDS) == expected
        assert tactic["swap_ab"] is True
        assert tactic["fp8_accum_mode"] == "1xacc"
        assert tactic["in_kernel_fc2_reduce"] is False
        assert tactic["grouped_token_back"] is False
        assert tactic["combine_format"] == "bf16"
        candidates = hopper_mxfp4_candidates(
            token, routing_profile=profile, **_OPTIMIZATION_SHAPE
        )
        assert candidates[0] == normalize_mxfp4_optimization_tactic(tactic)


def test_cache_identity_preserves_existing_winners_and_tracks_domain(monkeypatch):
    expected = {
        "block_permutation_v1": "0e34a82c58bf39fc04cd25c54a3d62ae88cc7d46f2449b8d5a36e388022d8682",
        "published_exact_balanced_v1": "ab16d80c9766ee9de4c0f57ffbfd93de7bd5aa4fe71947578aadf3fd2b7e2443",
    }
    for profile, identity in expected.items():
        assert hopper_mxfp4_cache_provenance_sha256(routing_profile=profile) == identity
    changed = (
        *mxfp4_tuner._BASE_TACTICS,
        dict(mxfp4_tuner._BASE_TACTICS[0], group_hint=999),
    )
    with monkeypatch.context() as patch:
        patch.setattr(mxfp4_tuner, "_BASE_TACTICS", changed)
        for profile, identity in expected.items():
            assert (
                hopper_mxfp4_cache_provenance_sha256(routing_profile=profile)
                != identity
            )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.src.moe_hopper_fp8 import (
        mxfp4_policy,
    )

    monkeypatch.setattr(
        mxfp4_policy, "MXFP4_OPTIMIZATION_VERSION", "changed-test-domain"
    )
    for profile, identity in expected.items():
        assert hopper_mxfp4_cache_provenance_sha256(routing_profile=profile) != identity


def test_h20_anchors_are_candidates_but_not_defaults():
    candidates = hopper_mxfp4_candidates(
        1, hidden=3072, intermediate=1280, num_experts=384, world_size=8
    )
    anchors = [
        t for t in candidates if t["group_hint"] == 78 and not t["tail_split_pairs"]
    ]
    assert len(anchors) == 2
    assert {t["num_sched_stages"] for t in anchors} == {1, 2}
    assert all(t["mma_tiler_mnk"] == (128, 16, 256) and t["pingpong"] for t in anchors)
    for profile in MXFP4_TUNING_ROUTING_PROFILES:
        for token in MXFP4_TUNING_TOKEN_BUCKETS:
            assert (
                hopper_mxfp4_default_tactic(token, routing_profile=profile)[
                    "group_hint"
                ]
                != 78
            )


def test_small_shape_uses_legal_candidates_from_both_profiles():
    candidates = hopper_mxfp4_candidates(
        512, hidden=128, intermediate=128, num_experts=8, world_size=1
    )
    assert {t["mma_tiler_mnk"][2] for t in candidates} == {128}
    assert {t["group_hint"] for t in candidates} >= {330, 396}
    assert (
        normalize_mxfp4_optimization_tactic(hopper_mxfp4_default_tactic(512))
        not in candidates
    )


@pytest.mark.parametrize("profile", MXFP4_TUNING_ROUTING_PROFILES)
def test_candidates_and_defaults_return_fresh_copies(profile):
    candidates = hopper_mxfp4_candidates(
        8, routing_profile=profile, **_OPTIMIZATION_SHAPE
    )
    expected = copy.deepcopy(candidates)
    candidates[0].clear()
    candidates.clear()
    default = hopper_mxfp4_default_tactic(8, routing_profile=profile)
    expected_default = dict(default)
    default.clear()
    assert (
        hopper_mxfp4_candidates(8, routing_profile=profile, **_OPTIMIZATION_SHAPE)
        == expected
    )
    assert hopper_mxfp4_default_tactic(8, routing_profile=profile) == expected_default


@pytest.mark.parametrize(
    "bad_profile",
    (
        None,
        True,
        "",
        "block_permutation",
        "published_exact_balanced",
        "BLOCK_PERMUTATION_V1",
    ),
)
def test_profile_aware_apis_reject_noncanonical_profile(bad_profile):
    calls = (
        lambda: normalize_hopper_mxfp4_routing_profile(bad_profile),
        lambda: normalize_sm90_routing_profile(bad_profile),
        lambda: hopper_mxfp4_cache_provenance_sha256(routing_profile=bad_profile),
        lambda: hopper_mxfp4_default_tactic(8, routing_profile=bad_profile),
        lambda: hopper_mxfp4_candidates(
            8, routing_profile=bad_profile, **_OPTIMIZATION_SHAPE
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="routing_profile"):
            call()


@pytest.mark.parametrize(
    ("max_tokens", "bucket"),
    ((1, 8), (8, 8), (9, 32), (31, 32), (33, 64), (2048, 2048), (4096, 2048)),
)
def test_default_uses_ceil_bucket_and_clamps_above_domain(
    max_tokens: int, bucket: int
) -> None:
    assert hopper_mxfp4_default_tactic(max_tokens) == hopper_mxfp4_default_tactic(
        bucket
    )


@pytest.mark.parametrize("bad_tokens", (True, False, 0, -1, 8.0, "8", None))
def test_default_rejects_invalid_token_count(bad_tokens: object) -> None:
    with pytest.raises(ValueError):
        hopper_mxfp4_default_tactic(bad_tokens)  # type: ignore[arg-type]


def test_fused_validator_rejects_fields_types_and_illegal_geometry() -> None:
    tactic = _base_candidates()[0]

    for key, value, message in (
        ("swap_ab", False, "swap_ab"),
        ("pingpong", 1, "pingpong"),
        ("mma_tiler_mnk", (64, 16, 128), "tile"),
        ("cluster_shape_mnk", (4, 1, 1), "cluster"),
        ("fp8_accum_mode", "2xacc", "1xacc"),
        ("load_balance_mode", "bad", "load balance"),
        ("token_back_mode", "bad", "token-back"),
        ("group_hint", True, "positive non-bool"),
        ("num_sched_stages", 0, "positive non-bool"),
        ("in_kernel_fc2_reduce", True, "in-kernel"),
        ("dedup_dispatch", 1, "dedup_dispatch must be bool"),
        ("grouped_token_back", True, "grouped_token_back=false"),
        ("combine_format", "32e4m3xe8m0", "combine_format='bf16'"),
        ("active_dispatch_warps", 3, "active_dispatch_warps"),
        ("fc1_store_offload", 1, "fc1_store_offload must be bool"),
        ("fc1_early_done_publish", 1, "fc1_early_done_publish must be bool"),
        ("fold_producer_warps", 1, "fold_producer_warps must be bool"),
    ):
        malformed = {**tactic, key: value}
        with pytest.raises(ValueError, match=message):
            validate_hopper_mxfp4_tactic(malformed)

    pingpong = {**tactic, "pingpong": True}
    with pytest.raises(ValueError, match="requires M128"):
        validate_hopper_mxfp4_tactic(pingpong)
    missing = dict(tactic)
    missing.pop("group_hint")
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(missing)
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic({**tactic, "unknown": 1})
    with pytest.raises(
        ValueError,
        match="fold_producer_warps=True requires active_dispatch_warps=1",
    ):
        validate_hopper_mxfp4_tactic(
            {
                **tactic,
                "active_dispatch_warps": 2,
                "fold_producer_warps": True,
            },
        )


def test_validator_requires_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_hopper_mxfp4_tactic([])  # type: ignore[arg-type]


def _optimization_anchor(**overrides):
    tactic = hopper_mxfp4_default_tactic(2048)
    tactic.update(overrides)
    return tactic


class TestMxfp4OptimizationCandidates(unittest.TestCase):
    def test_legacy_mapping_is_explicit_and_does_not_mutate(self):
        legacy = _optimization_anchor()
        before = copy.deepcopy(legacy)
        normalized = normalize_mxfp4_optimization_tactic(legacy)
        self.assertEqual(legacy, before)
        self.assertEqual(len(normalized), 20)
        self.assertIs(normalized["tail_split_pairs"], False)
        self.assertIs(normalized["fc2_tail_n8"], False)
        self.assertEqual(normalized["fc1_ready_mode"], "tile")
        self.assertEqual(normalize_mxfp4_optimization_tactic(normalized), normalized)
        for field in legacy:
            partial = dict(legacy)
            del partial[field]
            with self.assertRaises(ValueError, msg=field):
                normalize_mxfp4_optimization_tactic(partial)

    def test_partial_or_invalid_strategy_never_silently_defaults(self):
        for extra in (
            {"fc2_tail_n8": True},
            {"fc1_ready_mode": "k256"},
            {"fc2_tail_n8": 1, "fc1_ready_mode": "tile"},
            {"fc2_tail_n8": False, "fc1_ready_mode": "auto"},
            {"misspelled_strategy": True},
            {"tail_split_pairs": 1},
            {"tail_split_pairs": True},
        ):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                normalize_mxfp4_optimization_tactic(_optimization_anchor(**extra))

    def test_explicit_tail_pairs_have_distinct_identity_and_legacy_default(self):
        legacy = _optimization_anchor(cluster_shape_mnk=(1, 2, 1))
        disabled = normalize_mxfp4_optimization_tactic(legacy)
        enabled = normalize_mxfp4_optimization_tactic(
            dict(legacy, tail_split_pairs=True)
        )
        self.assertIs(enabled["tail_split_pairs"], True)
        self.assertEqual(validate_hopper_mxfp4_tactic(enabled), enabled)
        self.assertNotEqual(
            mxfp4_optimization_candidate_sha256([disabled]),
            mxfp4_optimization_candidate_sha256([enabled]),
        )
        legacy_19 = dict(disabled)
        legacy_19.pop("tail_split_pairs")
        self.assertEqual(normalize_mxfp4_optimization_tactic(legacy_19), disabled)
        with self.assertRaisesRegex(ValueError, "fc1_ready_mode='tile'"):
            normalize_mxfp4_optimization_tactic(dict(enabled, fc1_ready_mode="k256"))

    def test_expansion_is_bounded_and_keeps_originals_first(self):
        original = _base_candidates()
        frozen = copy.deepcopy(original)
        expanded = expand_mxfp4_optimization_candidates(original, **_OPTIMIZATION_SHAPE)
        self.assertEqual(original, frozen)
        self.assertEqual(
            expanded[: len(original)],
            [normalize_mxfp4_optimization_tactic(tactic) for tactic in original],
        )
        self.assertGreater(len(expanded), len(original))
        self.assertLessEqual(len(expanded), 4 * len(original))
        for candidate in expanded:
            self.assertIs(candidate["tail_split_pairs"], False)
            resolve_mxfp4_tactic_optimizations(candidate, **_OPTIMIZATION_SHAPE)
        self.assertEqual(
            expanded,
            expand_mxfp4_optimization_candidates(
                original + original, **_OPTIMIZATION_SHAPE
            ),
        )

    def test_effective_folding_is_used_for_protocol_eligibility(self):
        requested = _optimization_anchor(
            fc1_store_offload=True,
            fc1_early_done_publish=False,
            fold_producer_warps=True,
            active_dispatch_warps=1,
            fc2_tail_n8=False,
            fc1_ready_mode="k256",
        )
        resolved = resolve_mxfp4_tactic_optimizations(requested, **_OPTIMIZATION_SHAPE)
        self.assertEqual(
            (resolved.fc1_ready_segments, resolved.fc1_ready_bits), (12, 48)
        )
        self.assertTrue(requested["fc1_store_offload"])
        with self.assertRaises(ValueError):
            resolve_mxfp4_tactic_optimizations(
                dict(requested, fold_producer_warps=False), **_OPTIMIZATION_SHAPE
            )

    def test_unsupported_shapes_keep_base_but_not_protocol_variants(self):
        for override in (
            {"world_size": 8},
            {"intermediate": 4096},
            {"num_experts": 192},
        ):
            shape = dict(_OPTIMIZATION_SHAPE, **override)
            expanded = expand_mxfp4_optimization_candidates(
                [_optimization_anchor()], **shape
            )
            self.assertTrue(expanded)
            self.assertTrue(
                all(tactic["fc1_ready_mode"] == "tile" for tactic in expanded)
            )
        for override in ({"hidden": 3072},):
            expanded = expand_mxfp4_optimization_candidates(
                [_optimization_anchor()], **dict(_OPTIMIZATION_SHAPE, **override)
            )
            self.assertEqual(len(expanded), 1)
        for override in (
            {"dedup_dispatch": True},
            {"token_back_mode": "reuse_dispatch_warps"},
        ):
            expanded = expand_mxfp4_optimization_candidates(
                [_optimization_anchor(**override)], **_OPTIMIZATION_SHAPE
            )
            self.assertEqual(len(expanded), 1)

    def test_invalid_model_or_base_is_an_error_not_an_empty_union(self):
        for override in (
            {"hidden": True},
            {"world_size": 0},
            {"num_experts": 383},
            {"intermediate": 3000},
        ):
            with self.assertRaises(ValueError):
                expand_mxfp4_optimization_candidates(
                    [_optimization_anchor()], **dict(_OPTIMIZATION_SHAPE, **override)
                )

    def test_identity_covers_strategy_set_without_order_or_duplicate_aliases(self):
        base = [_optimization_anchor()]
        expanded = expand_mxfp4_optimization_candidates(base, **_OPTIMIZATION_SHAPE)
        digest = mxfp4_optimization_candidate_sha256
        self.assertNotEqual(digest(base), digest(expanded))
        self.assertEqual(digest(expanded), digest(list(reversed(expanded)) + expanded))
        self.assertEqual(
            digest(base),
            digest([normalize_mxfp4_optimization_tactic(base[0])]),
        )
        self.assertNotEqual(digest([expanded[0]]), digest([expanded[1]]))

    def test_one_domain_for_all_token_capacities_and_profiles(self):
        digests = set()
        for profile in MXFP4_TUNING_ROUTING_PROFILES:
            for token in (
                8,
                16,
                32,
                64,
                128,
                256,
                512,
                1024,
                2048,
                4096,
                8192,
                16384,
                32768,
            ):
                actual = hopper_mxfp4_candidates(
                    token, routing_profile=profile, **_OPTIMIZATION_SHAPE
                )
                original = _ordered_base_candidates(
                    token,
                    routing_profile=profile,
                    hidden=_OPTIMIZATION_SHAPE["hidden"],
                    intermediate=_OPTIMIZATION_SHAPE["intermediate"],
                )
                previous = expand_mxfp4_optimization_candidates(
                    original, **_OPTIMIZATION_SHAPE
                )
                self.assertEqual(actual[: len(previous)], previous)
                self.assertEqual(len(actual), 38)
                self.assertEqual(sum(c["tail_split_pairs"] for c in actual), 12)
                self.assertEqual(
                    actual[0],
                    normalize_mxfp4_optimization_tactic(original[0]),
                )
                digests.add(mxfp4_optimization_candidate_sha256(actual))
        self.assertEqual(len(digests), 1)

    def test_tail_catalog_covers_measured_neighbors_without_cross_product(self):
        candidates = hopper_mxfp4_candidates(2048, **_OPTIMIZATION_SHAPE)
        for token in MXFP4_TUNING_TOKEN_BUCKETS:
            base = normalize_mxfp4_optimization_tactic(
                hopper_mxfp4_default_tactic(token)
            )
            tail = dict(base, cluster_shape_mnk=(1, 2, 1), tail_split_pairs=True)
            self.assertIn(base, candidates)
            self.assertIn(tail, candidates)
        large_base = normalize_mxfp4_optimization_tactic(
            hopper_mxfp4_default_tactic(2048)
        )
        for group, ready in (
            (528, "tile"),
            (256, "k256"),
            (512, "k256"),
            (128, "k256"),
        ):
            base = dict(
                large_base,
                group_hint=group,
                fc2_tail_n8=True,
                fc1_ready_mode=ready,
            )
            self.assertIn(base, candidates)
            self.assertIn(
                dict(
                    base,
                    cluster_shape_mnk=(1, 2, 1),
                    fc1_ready_mode="tile",
                    tail_split_pairs=True,
                ),
                candidates,
            )
        self.assertNotIn(dict(large_base, group_hint=256), candidates)
        for candidate in candidates:
            resolve_mxfp4_tactic_optimizations(candidate, **_OPTIMIZATION_SHAPE)

    def test_tail_catalog_filters_candidates_for_each_model_shape(self):
        for override in (
            {"hidden": 4096},
            {"intermediate": 4096},
            {"num_experts": 192},
            {"world_size": 8},
            {"hidden": 384},
            {"intermediate": 384},
            {"hidden": 128, "intermediate": 128, "num_experts": 6, "world_size": 2},
        ):
            with self.subTest(override=override):
                shape = dict(_OPTIMIZATION_SHAPE, **override)
                base = _ordered_base_candidates(
                    2048, hidden=shape["hidden"], intermediate=shape["intermediate"]
                )
                previous = expand_mxfp4_optimization_candidates(base, **shape)
                actual = hopper_mxfp4_candidates(2048, **shape)
                self.assertEqual(actual[: len(previous)], previous)
                tails = [c for c in actual if c["tail_split_pairs"]]
                self.assertTrue(tails)
                self.assertTrue(all(c["fc1_ready_mode"] == "tile" for c in actual))
                if shape["hidden"] != 7168:
                    self.assertTrue(all(not c["fc2_tail_n8"] for c in actual))
                if shape["hidden"] % 256 or shape["intermediate"] % 256:
                    self.assertTrue(all(c["mma_tiler_mnk"][2] == 128 for c in actual))
                for token in MXFP4_TUNING_TOKEN_BUCKETS:
                    seed = normalize_mxfp4_optimization_tactic(
                        hopper_mxfp4_default_tactic(token)
                    )
                    tail = dict(
                        seed, cluster_shape_mnk=(1, 2, 1), tail_split_pairs=True
                    )
                    if is_hopper_mxfp4_tactic_shape_compatible(
                        tail, hidden=shape["hidden"], intermediate=shape["intermediate"]
                    ):
                        self.assertIn(tail, tails)
                    else:
                        self.assertNotIn(tail, actual)
                for candidate in actual:
                    resolve_mxfp4_tactic_optimizations(candidate, **shape)


@pytest.fixture
def allow_mock_tuning_device(monkeypatch):
    # Only wrapper tests use this bypass; real device-guard tests stay active.
    for module in (sm90_mega, mxfp4_tuner):
        monkeypatch.setattr(
            module, "require_hopper_mxfp4_fused_tuning_device", lambda: None
        )


def test_full_union_records_effective_winner_and_cache_identity(
    monkeypatch, allow_mock_tuning_device
):
    cfg = SimpleNamespace(
        rank=0,
        world_size=4,
        num_tokens_per_rank=64,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    candidates = hopper_mxfp4_candidates(
        cfg.num_tokens_per_rank,
        hidden=cfg.hidden,
        intermediate=cfg.intermediate,
        num_experts=cfg.num_total_experts,
        world_size=cfg.world_size,
        routing_profile=cfg.routing_profile,
    )
    candidate = next(c for c in candidates if c["tail_split_pairs"])
    effective = {
        **candidate,
        "active_dispatch_warps": 2,
        "fc1_store_offload": False,
        "fc1_early_done_publish": True,
        "fold_producer_warps": True,
    }
    preflight_order = []
    validated_launch = (object(), ("launch",))
    frontend = SimpleNamespace(
        config=cfg,
        prepare_launch=mock.Mock(),
        validate_launch=mock.Mock(
            side_effect=lambda *args, **kwargs: (
                preflight_order.append("validate"),
                validated_launch,
            )[1]
        ),
        release=mock.Mock(),
        set_gate_up_clamp=mock.Mock(
            side_effect=lambda *args, **kwargs: preflight_order.append("clamp")
        ),
        effective_tactic=mock.Mock(return_value=effective),
    )
    buffer = SimpleNamespace(
        _frontend=frontend,
        _destroyed=False,
        num_max_tokens=64,
        hidden=7168,
    )
    output = SimpleNamespace(shape=(37, 7168), dtype=torch.bfloat16)
    ep_group = object()
    record = mock.Mock(return_value="/tmp/cache.json")
    final_inputs = object()
    monkeypatch.setattr(knob_cache, "record_knobs", record)
    monkeypatch.setattr(
        hopper_mxfp4,
        "_build_mxfp4_inputs",
        mock.Mock(return_value=final_inputs),
    )

    def fake_autotune(frontend, launch, candidates, **kwargs):
        assert kwargs["process_group"] is ep_group
        assert kwargs["expected_world_size"] == 4
        assert candidates == candidates_expected
        kwargs["preflight"]()
        kwargs["prepare_candidate"]()
        kwargs["on_winner"](candidate, 0.00075)
        return candidate

    candidates_expected = candidates
    monkeypatch.setattr(autotune_module, "autotune_knobs", fake_autotune)
    winner = autotune_module.autotune_hopper_mxfp4_mega_moe(
        output,
        object(),
        object(),
        buffer,
        num_tokens=37,
        gate_up_clamp=10.0,
        process_group=ep_group,
        candidates=candidates,
    )

    assert winner == candidate
    assert preflight_order == ["clamp", "validate"]
    frontend.set_gate_up_clamp.assert_called_once_with(10.0)
    frontend.validate_launch.assert_called_once_with(final_inputs, num_tokens=None)
    frontend.prepare_launch.assert_called_once_with(
        final_inputs,
        num_tokens=None,
        validated=validated_launch,
    )
    frontend.effective_tactic.assert_called_once_with()
    assert record.call_args.args[0] == effective
    assert record.call_args.args[0] != candidate
    kwargs = record.call_args.kwargs
    assert kwargs["dtype"] == hopper_mxfp4._MXFP4_TUNING_DTYPE_ID
    assert "fused" in kwargs["dtype"]
    assert kwargs["p50_us"] == pytest.approx(750.0)
    assert kwargs["gate_up_clamp"] == 10.0
    assert kwargs["routing_profile"] == cfg.routing_profile
    assert kwargs["tuning_provenance_sha256"] == (
        hopper_mxfp4_cache_provenance_sha256(
            routing_profile=cfg.routing_profile,
        )
    )
    assert mxfp4_optimization_candidate_sha256(candidates) in kwargs["source"]


def test_fused_supplied_candidates_must_be_complete_strategy_union_subset(
    monkeypatch, allow_mock_tuning_device
):
    cfg = SimpleNamespace(
        rank=0,
        world_size=4,
        num_tokens_per_rank=64,
        num_topk=6,
        num_total_experts=384,
        hidden=7168,
        intermediate=3072,
        gate_up_clamp=10.0,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    buffer = SimpleNamespace(_frontend=SimpleNamespace(config=cfg, release=mock.Mock()))
    union = _base_candidates()
    subset = [union[2], union[0]]
    supplied = [
        {
            **subset[0],
            "mma_tiler_mnk": list(subset[0]["mma_tiler_mnk"]),
            "cluster_shape_mnk": list(subset[0]["cluster_shape_mnk"]),
        },
        subset[1],
    ]
    captured = {}
    record = mock.Mock()
    monkeypatch.setattr(knob_cache, "record_knobs", record)

    def fake_autotune(frontend, launch, candidates, **kwargs):
        captured["candidates"] = candidates
        kwargs["on_winner"](candidates[0], 0.0005)
        return candidates[0]

    monkeypatch.setattr(autotune_module, "autotune_knobs", fake_autotune)
    assert autotune_module.autotune_hopper_mxfp4_mega_moe(
        object(), object(), object(), buffer, candidates=supplied
    ) == normalize_mxfp4_optimization_tactic(subset[0])
    assert captured["candidates"] == [
        normalize_mxfp4_optimization_tactic(c) for c in subset
    ]
    record.assert_not_called()

    h20_anchor = next(
        candidate
        for candidate in union
        if candidate["pingpong"]
        and candidate["mma_tiler_mnk"] == (128, 16, 256)
        and candidate["group_hint"] == 78
    )
    assert autotune_module.autotune_hopper_mxfp4_mega_moe(
        object(), object(), object(), buffer, candidates=[h20_anchor]
    ) == normalize_mxfp4_optimization_tactic(h20_anchor)
    assert captured["candidates"] == [normalize_mxfp4_optimization_tactic(h20_anchor)]
    record.assert_not_called()

    full = hopper_mxfp4_candidates(
        cfg.num_tokens_per_rank,
        hidden=cfg.hidden,
        intermediate=cfg.intermediate,
        num_experts=cfg.num_total_experts,
        world_size=cfg.world_size,
        routing_profile=cfg.routing_profile,
    )
    tail = next(c for c in full if c["tail_split_pairs"])
    assert (
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(), object(), object(), buffer, candidates=[tail]
        )
        == tail
    )
    record.assert_not_called()
    # Admitting a neighbor's geometry does not admit its unlisted strategies.
    with pytest.raises(ValueError, match="model/layout/protocol candidate union"):
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(),
            object(),
            object(),
            buffer,
            candidates=[dict(tail, tail_split_pairs=False)],
        )

    outside = {**union[0], "group_hint": 999999}
    with pytest.raises(ValueError, match="outside the runtime candidate union"):
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(), object(), object(), buffer, candidates=[outside]
        )

    with pytest.raises(ValueError, match="candidates must be unique"):
        autotune_module.autotune_hopper_mxfp4_mega_moe(
            object(),
            object(),
            object(),
            buffer,
            candidates=[union[0], union[0]],
        )


def _mock_cuda_device(
    monkeypatch,
    *,
    name: str,
    capability: tuple[int, int],
    sm_count: int,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: name)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda device: capability,
    )
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(multi_processor_count=sm_count),
    )


def _resolver_kwargs() -> dict:
    return {
        "world_size": 4,
        "hidden": 7168,
        "intermediate": 3072,
        "num_total_experts": 384,
        "num_topk": 6,
        "num_max_tokens": 512,
        "gate_up_clamp": 10.0,
    }


def test_exact_standard_h200_is_accepted(monkeypatch) -> None:
    _mock_cuda_device(
        monkeypatch,
        name="NVIDIA H200",
        capability=(9, 0),
        sm_count=132,
    )
    mxfp4_tuner.require_hopper_mxfp4_fused_tuning_device()


@pytest.mark.parametrize(
    ("name", "sm_count"),
    [
        ("NVIDIA H20-3e", 78),
        ("NVIDIA H100 80GB HBM3", 132),
        ("NVIDIA H200 NVL", 114),
    ],
)
def test_any_sm90_product_is_accepted_for_fused(
    monkeypatch, name: str, sm_count: int
) -> None:
    _mock_cuda_device(
        monkeypatch,
        name=name,
        capability=(9, 0),
        sm_count=sm_count,
    )
    mxfp4_tuner.require_hopper_mxfp4_fused_tuning_device()


@pytest.mark.parametrize("capability", [(8, 9), (9, 1), (10, 0)])
def test_non_sm90_device_is_rejected_for_fused(
    monkeypatch, capability: tuple[int, int]
) -> None:
    _mock_cuda_device(
        monkeypatch,
        name="test GPU",
        capability=capability,
        sm_count=132,
    )
    with pytest.raises(RuntimeError, match="requires SM90"):
        mxfp4_tuner.require_hopper_mxfp4_fused_tuning_device()


def test_no_cuda_is_rejected(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires a CUDA device"):
        mxfp4_tuner.require_hopper_mxfp4_fused_tuning_device()


def test_cache_resolver_checks_sm90_device(monkeypatch) -> None:
    fused_guard = mock.Mock(side_effect=RuntimeError("fused device guard"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_fused_tuning_device",
        fused_guard,
    )
    with pytest.raises(RuntimeError, match="fused device guard"):
        hopper_mxfp4._resolve_mxfp4_knobs(None, **_resolver_kwargs())
    fused_guard.assert_called_once_with()


def test_h20_fused_none_cache_miss_uses_h200_derived_heuristic(monkeypatch) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
        normalize_mxfp4_optimization_tactic,
    )

    _mock_cuda_device(
        monkeypatch,
        name="NVIDIA H20-3e",
        capability=(9, 0),
        sm_count=78,
    )
    lookup = mock.Mock(return_value=None)
    monkeypatch.setattr(knob_cache, "lookup_knobs", lookup)
    kwargs = {
        "world_size": 8,
        "hidden": 3072,
        "intermediate": 1280,
        "num_total_experts": 384,
        "num_topk": 8,
        "num_max_tokens": 1,
        "gate_up_clamp": 10.0,
        "routing_profile": SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    }

    assert hopper_mxfp4._resolve_mxfp4_knobs(None, **kwargs) == (
        normalize_mxfp4_optimization_tactic(
            mxfp4_tuner.hopper_mxfp4_default_tactic(
                1,
                routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
            )
        )
    )
    lookup.assert_called_once()


def test_complete_explicit_tactics_bypass_device_guard(monkeypatch) -> None:
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
        normalize_mxfp4_optimization_tactic,
    )

    fused_guard = mock.Mock(side_effect=AssertionError("guard must not run"))
    monkeypatch.setattr(
        mxfp4_tuner,
        "require_hopper_mxfp4_fused_tuning_device",
        fused_guard,
    )
    fused = mxfp4_tuner._base_candidates()[0]
    assert hopper_mxfp4._resolve_mxfp4_knobs(
        fused,
        **_resolver_kwargs(),
    ) == normalize_mxfp4_optimization_tactic(fused)
    fused_guard.assert_not_called()
