# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega

import pytest

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    MXFP4_TUNING_PROVENANCE,
    autotune as autotune_module,
    hopper_mxfp4,
    knob_cache,
    mxfp4_tuner,
    MXFP4_TUNING_TOKEN_BUCKETS,
    hopper_mxfp4_candidate_records,
    hopper_mxfp4_candidates,
    hopper_mxfp4_candidates_for_shape,
    hopper_mxfp4_default_tactic,
    hopper_mxfp4_ordered_candidates,
    hopper_mxfp4_tuning_manifest,
    is_valid_hopper_mxfp4_tactic,
    validate_hopper_mxfp4_tactic,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_FUSED_RUNTIME_ANCHOR_PROVENANCE,
    MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE,
    MXFP4_TUNING_ROUTING_PROFILES,
    hopper_mxfp4_cache_provenance_sha256,
    hopper_mxfp4_tuning_provenance,
    hopper_mxfp4_runtime_candidates,
    hopper_mxfp4_runtime_candidates_for_shape,
    is_hopper_mxfp4_tactic_shape_compatible,
    normalize_hopper_mxfp4_routing_profile,
)

from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_optimization import (
    expand_mxfp4_optimization_candidates,
    hopper_mxfp4_optimization_candidates,
    mxfp4_optimization_candidate_sha256,
    normalize_mxfp4_optimization_tactic,
    resolve_mxfp4_tactic_optimizations,
)


_EXPECTED_BLOCK_PROVENANCE = {
    "artifact_manifest_sha256": "852cbb019a17a9e76991b4f82b5ae64fb3df3245f6e72ec43a20bb407cbbd477",
    "candidate_union_sha256": "dce1c6e5f22ac482928ebc2b07ba682665a41464ae44f367c306d2065c3b1926",
    "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
    "external_schema_version": 1,
    "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
    "policy_sha256": "d3349c7ca1976b02ab076dcd87617de7552164d640481d7d614256d70da9a4cf",
    "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
    "runtime_manifest_sha256": "94ae92677c24630f28192d5df57704ff3ca7d6d94e9a24ebf46b761c198d9bf8",
    "source_manifest_sha256": "1081ba172754b107a9ab2360a6fac8d2e854fd5c1fc45305b571c935885d5277",
    "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
}

_EXPECTED_BLOCK_RUNTIME_SHA256 = _EXPECTED_BLOCK_PROVENANCE["runtime_manifest_sha256"]
_EXPECTED_BLOCK_CANDIDATE_UNION_SHA256 = _EXPECTED_BLOCK_PROVENANCE[
    "candidate_union_sha256"
]

_EXPECTED_WINNER_IDS = {
    8: "e66dbeaf780401025f44e6543492740a0a0471a0edea7548ada45b7f42a06420",
    32: "f3f218e0009b41b3f0a2aef60cde244c1faf1d1988e6cd71e7d56020006ca50e",
    64: "c8ce6c465c70a34b5c1f5923319d07753efb199bdd164e352239bf53925c61a7",
    128: "0a635083f8a826f49f3bc5bbdcbcab108d93e880ffc86335d7d490668d01d8e8",
    256: "0fc1560a23803ffe743e1a6e3f0ecbe5dad93f544066cbe74e4a81b5d21e730a",
    512: "f573e32fe8c9c8d94754fd7002be6544925c063e77e461290d83505dad8d9c8b",
    1024: "851b0aa827a934c5e94fee613460b35b2ff19c38c0edf6884ef0f1ba9aa63eb7",
    2048: "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
}
_EXPECTED_BLOCK_CANDIDATE_IDS = tuple(sorted(_EXPECTED_WINNER_IDS.values()))

_EXPECTED_FUSED_RUNTIME_CANDIDATE_IDS = (
    "0a635083f8a826f49f3bc5bbdcbcab108d93e880ffc86335d7d490668d01d8e8",
    "0fc1560a23803ffe743e1a6e3f0ecbe5dad93f544066cbe74e4a81b5d21e730a",
    "1630f29e2346f2d817477c82aa520672961445712ec504cd8591581169e5c626",
    "170e811b97e04d5e5a2795335891fd647a404a6e8bc24f1cf4b5c701e293c8aa",
    "38265fb512a81b2b69d9ca7b156601a3e5363b2e15ba237615a9f4d3900813d6",
    "3df511dac7954f726e9002cb3cbbe86ce5b5479ca43e243a0d890cc48f6921d2",
    "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
    "78a5836d443416669b66535372b920ffe1dfe252903d350afdc12699f7b5046e",
    "851b0aa827a934c5e94fee613460b35b2ff19c38c0edf6884ef0f1ba9aa63eb7",
    "89927bf6af211d5047bc8a2e6be10ae892c65c85d2848ade50c742a47df3ee56",
    "b66df770384bc5c406d13b0d71d6f701aec6879498ed9d9e2edc209db6522c2f",
    "c8ce6c465c70a34b5c1f5923319d07753efb199bdd164e352239bf53925c61a7",
    "e2c39aa656b37beeb8dbc5a1533ede68a09fe7fb1101581bfab210a63c207b1f",
    "e66dbeaf780401025f44e6543492740a0a0471a0edea7548ada45b7f42a06420",
    "f3f218e0009b41b3f0a2aef60cde244c1faf1d1988e6cd71e7d56020006ca50e",
    "f573e32fe8c9c8d94754fd7002be6544925c063e77e461290d83505dad8d9c8b",
    "f9e97da647c04c0999bb38fe7a5c26c30e74060e856b037c90a1bd661136c341",
)
_FOLDED_FUSED_CANDIDATE_IDS = {
    "851b0aa827a934c5e94fee613460b35b2ff19c38c0edf6884ef0f1ba9aa63eb7",
    "f573e32fe8c9c8d94754fd7002be6544925c063e77e461290d83505dad8d9c8b",
    "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
}

_EXPECTED_EXACT_ARTIFACT_SHA256 = (
    "62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61"
)

_EXPECTED_EXACT_RUNTIME_SHA256 = (
    "f4112c7d0d7ead640239c1df3d7f4af74e2a1fb35cf5e821edd1beba9bba3e99"
)

_EXPECTED_EXACT_CANDIDATE_IDS = (
    "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
    "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
    "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
    "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
    "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
    "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
    "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
)

_EXPECTED_EXACT_WINNER_IDS = {
    8: "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
    32: "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
    64: "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
    128: "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
    256: "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
    512: "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
    1024: "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
    2048: "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
}

_H20_FUSED_RUNTIME_ANCHOR_IDS = {
    "1630f29e2346f2d817477c82aa520672961445712ec504cd8591581169e5c626",
    "89927bf6af211d5047bc8a2e6be10ae892c65c85d2848ade50c742a47df3ee56",
}
_EXPECTED_FUSED_RUNTIME_CANDIDATE_UNION_SHA256 = (
    "103cb31f7cbcc44ced8264689d66735386c7ee0df717898d514321d9fb735526"
)
_FUSED_LAYOUT_FIELDS = {
    "dedup_dispatch",
    "grouped_token_back",
    "combine_format",
    "active_dispatch_warps",
    "fc1_store_offload",
    "fc1_early_done_publish",
    "fold_producer_warps",
}


def _frozen_fused_projection(tactic: dict[str, object]) -> dict[str, object]:
    return {
        key: value for key, value in tactic.items() if key not in _FUSED_LAYOUT_FIELDS
    }


def _manifest_sha256(value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


def _fused_tactic_id(tactic: object) -> str:
    raw = json.dumps(
        {"implementation": "mxfp4_fused", "tactic": tactic},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _candidate_union_sha256(records: list[dict[str, object]]) -> str:
    payload = [
        {
            "candidate_id": record["candidate"]["candidate_id"],
            "tactic": record["candidate"]["effective_tactic"],
        }
        for record in records
    ]
    raw = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def test_embedded_manifest_is_byte_canonical() -> None:
    manifest = hopper_mxfp4_tuning_manifest()
    provenance = hopper_mxfp4_tuning_provenance()

    assert _manifest_sha256(manifest) == _EXPECTED_BLOCK_RUNTIME_SHA256
    assert dict(provenance) == _EXPECTED_BLOCK_PROVENANCE
    assert (
        manifest["artifact_manifest_sha256"] == provenance["artifact_manifest_sha256"]
    )
    assert manifest["external_schema_version"] == 1
    assert manifest["implementation"] == "mxfp4_fused"
    assert manifest["routing_profile"] == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE
    assert manifest["runtime_schema_version"] == 1
    assert _candidate_union_sha256(manifest["candidate_union"]) == (
        _EXPECTED_BLOCK_CANDIDATE_UNION_SHA256
    )
    assert set(map(int, manifest["per_token_winners"])) == set(
        MXFP4_TUNING_TOKEN_BUCKETS
    )


def test_default_profile_identity_and_public_provenance_are_current() -> None:
    assert MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE == (
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )
    assert MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE == (
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    )
    assert MXFP4_TUNING_ROUTING_PROFILES == (
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    assert dict(MXFP4_TUNING_PROVENANCE) == _EXPECTED_BLOCK_PROVENANCE
    assert (
        MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[
            MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE
        ]
        == MXFP4_TUNING_PROVENANCE
    )


def test_explicit_block_profile_is_identical_to_omitted_profile() -> None:
    profile = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE

    assert (
        hopper_mxfp4_candidate_records(routing_profile=profile)
        == hopper_mxfp4_candidate_records()
    )
    assert hopper_mxfp4_candidates(routing_profile=profile) == hopper_mxfp4_candidates()
    assert hopper_mxfp4_candidates_for_shape(
        hidden=7168,
        intermediate=3072,
        routing_profile=profile,
    ) == hopper_mxfp4_candidates_for_shape(hidden=7168, intermediate=3072)
    assert hopper_mxfp4_default_tactic(
        512, routing_profile=profile
    ) == hopper_mxfp4_default_tactic(512)
    assert hopper_mxfp4_ordered_candidates(
        512,
        hidden=7168,
        intermediate=3072,
        routing_profile=profile,
    ) == hopper_mxfp4_ordered_candidates(512, hidden=7168, intermediate=3072)
    assert (
        hopper_mxfp4_tuning_manifest(routing_profile=profile)
        == hopper_mxfp4_tuning_manifest()
    )
    assert (
        hopper_mxfp4_tuning_provenance(routing_profile=profile)
        == MXFP4_TUNING_PROVENANCE
    )


def test_published_exact_manifest_and_provenance_are_frozen() -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    manifest = hopper_mxfp4_tuning_manifest(routing_profile=profile)
    provenance = hopper_mxfp4_tuning_provenance(routing_profile=profile)

    assert manifest["artifact_manifest_sha256"] == (_EXPECTED_EXACT_ARTIFACT_SHA256)
    assert manifest["routing_profile"] == profile
    assert manifest["runtime_schema_version"] == 1
    assert manifest["external_schema_version"] == (1)
    assert _manifest_sha256(manifest) == _EXPECTED_EXACT_RUNTIME_SHA256
    assert provenance["artifact_manifest_sha256"] == (_EXPECTED_EXACT_ARTIFACT_SHA256)
    assert provenance["runtime_manifest_sha256"] == (_EXPECTED_EXACT_RUNTIME_SHA256)
    assert set(map(int, manifest["per_token_winners"])) == set(
        MXFP4_TUNING_TOKEN_BUCKETS
    )
    assert all(
        "requested_tactic_aliases" not in record["candidate"]
        for record in manifest["candidate_union"]
    )


def test_cache_provenance_is_deterministic_hex_and_domain_scoped() -> None:
    fingerprints = {
        profile: hopper_mxfp4_cache_provenance_sha256(
            routing_profile=profile,
        )
        for profile in MXFP4_TUNING_ROUTING_PROFILES
    }

    assert len(set(fingerprints.values())) == len(fingerprints)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in fingerprints.values()
    )
    assert fingerprints == {
        identity: hopper_mxfp4_cache_provenance_sha256(
            routing_profile=identity,
        )
        for identity in fingerprints
    }


def test_published_exact_union_is_legal_complete_and_profile_isolated() -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    records = hopper_mxfp4_candidate_records(routing_profile=profile)
    tactics = hopper_mxfp4_candidates(routing_profile=profile)
    candidate_ids = tuple(record["candidate"]["candidate_id"] for record in records)

    assert candidate_ids == _EXPECTED_EXACT_CANDIDATE_IDS
    assert candidate_ids == tuple(sorted(set(candidate_ids)))
    assert set(candidate_ids) != {
        record["candidate"]["candidate_id"]
        for record in hopper_mxfp4_candidate_records()
    }
    assert {
        token for record in records for token in record["winner_for_tokens"]
    } == set(MXFP4_TUNING_TOKEN_BUCKETS)
    for record, tactic in zip(records, tactics, strict=True):
        assert (
            _frozen_fused_projection(tactic) == record["candidate"]["effective_tactic"]
        )
        assert set(tactic) == (
            set(record["candidate"]["effective_tactic"]) | _FUSED_LAYOUT_FIELDS
        )
        assert is_valid_hopper_mxfp4_tactic(tactic)


def test_fused_runtime_union_is_routing_independent_and_provenance_tracked() -> None:
    block = hopper_mxfp4_runtime_candidates(
        routing_profile=MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    )
    exact = hopper_mxfp4_runtime_candidates(
        routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    )

    assert block == exact
    assert len(block) == 17
    assert len({json.dumps(tactic, sort_keys=True) for tactic in block}) == len(block)
    assert (
        MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256
        == _EXPECTED_FUSED_RUNTIME_CANDIDATE_UNION_SHA256
    )
    assert MXFP4_FUSED_RUNTIME_ANCHOR_PROVENANCE == {
        "device": "NVIDIA H20-3e",
        "compute_capability": (9, 0),
        "sm_count": 78,
        "world_size": 8,
        "tokens_per_rank": 1,
        "hidden": 3072,
        "intermediate": 1280,
        "num_experts": 384,
        "topk": 8,
        "routing_profile": MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
        "routing_seed": 1234,
        "route_ids_sha256": (
            "d95ea5e18e4bb5010dd9cdadf928c6a43e085537615c5dbd166027a64de844eb"
        ),
        "winner_manifest_sha256": (
            "ffb3f8df0edef5e6a07d8685b35e9759e4da691e6010d4251c0df9b37ca40ce7"
        ),
        "formal_manifest_sha256": (
            "fe5cca19a9bc8e30a74f28a47561fd08755ba3f5ab22f90e50b63ebaceabafd6"
        ),
        "artifact_files_sha256": (
            "ae23042d2d4505794867cbdd3523e5469be785506211d38a6266821fe45514f7"
        ),
    }

    runtime_ids = tuple(_fused_tactic_id(tactic) for tactic in block)
    assert runtime_ids == _EXPECTED_FUSED_RUNTIME_CANDIDATE_IDS
    assert set(runtime_ids) >= _H20_FUSED_RUNTIME_ANCHOR_IDS
    for profile in MXFP4_TUNING_ROUTING_PROFILES:
        assert all(
            tactic in block
            for tactic in hopper_mxfp4_candidates(
                routing_profile=profile,
            )
        )


def test_h20_runtime_anchors_are_shape_legal_but_not_heuristic_winners() -> None:
    legal = hopper_mxfp4_runtime_candidates_for_shape(
        hidden=3072,
        intermediate=1280,
        routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    )
    legal_ids = {_fused_tactic_id(tactic) for tactic in legal}
    assert legal_ids >= _H20_FUSED_RUNTIME_ANCHOR_IDS
    anchors = [
        tactic
        for tactic in legal
        if _fused_tactic_id(tactic) in _H20_FUSED_RUNTIME_ANCHOR_IDS
    ]
    assert len(anchors) == 2
    assert all(
        hopper_mxfp4_default_tactic(
            token,
            routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
        )
        not in anchors
        for token in MXFP4_TUNING_TOKEN_BUCKETS
    )


def test_published_exact_default_leads_runtime_ordering() -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    manifest = hopper_mxfp4_tuning_manifest(routing_profile=profile)

    for token, expected_id in _EXPECTED_EXACT_WINNER_IDS.items():
        default = hopper_mxfp4_default_tactic(token, routing_profile=profile)
        winner = manifest["per_token_winners"][str(token)]
        ordered = hopper_mxfp4_ordered_candidates(
            token,
            hidden=7168,
            intermediate=3072,
            routing_profile=profile,
        )
        assert winner["candidate_id"] == expected_id
        assert (
            _frozen_fused_projection(default) == winner["candidate"]["effective_tactic"]
        )
        assert ordered[0] == default

    assert (
        hopper_mxfp4_default_tactic(1024, routing_profile=profile)["token_back_mode"]
        == "reuse_dispatch_warps"
    )
    assert hopper_mxfp4_default_tactic(2048, routing_profile=profile)[
        "cluster_shape_mnk"
    ] == (1, 1, 1)


def test_candidate_union_is_legal_sorted_deduplicated_and_complete() -> None:
    records = hopper_mxfp4_candidate_records()
    tactics = hopper_mxfp4_candidates()
    candidate_ids = tuple(record["candidate"]["candidate_id"] for record in records)

    assert len(records) == len(tactics) == 8
    assert candidate_ids == _EXPECTED_BLOCK_CANDIDATE_IDS
    assert candidate_ids == tuple(sorted(set(candidate_ids)))
    assert {
        token for record in records for token in record["winner_for_tokens"]
    } == set(MXFP4_TUNING_TOKEN_BUCKETS)
    for record, tactic in zip(records, tactics, strict=True):
        candidate = record["candidate"]
        candidate_id = candidate["candidate_id"]
        requested = candidate["requested_tactic"]
        effective = candidate["effective_tactic"]
        assert candidate["implementation"] == "mxfp4_fused"
        assert candidate["requested_tactic_aliases"] == [requested]
        assert effective == tactic
        assert _fused_tactic_id(effective) == candidate_id
        if candidate_id in _FOLDED_FUSED_CANDIDATE_IDS:
            changed = {key for key in requested if requested[key] != effective[key]}
            assert changed == {"fc1_store_offload", "fc1_early_done_publish"}
            assert requested["fold_producer_warps"] is True
            assert requested["active_dispatch_warps"] == 1
            assert requested["fc1_store_offload"] is True
            assert requested["fc1_early_done_publish"] is False
            assert effective["fc1_store_offload"] is False
            assert effective["fc1_early_done_publish"] is True
        else:
            assert requested == effective
        assert validate_hopper_mxfp4_tactic(tactic) == tactic
        assert is_valid_hopper_mxfp4_tactic(tactic)


def test_h128_fused_uses_all_legal_cross_profile_candidates() -> None:
    profile_legal = hopper_mxfp4_candidates_for_shape(hidden=128, intermediate=128)
    assert len(profile_legal) == 1
    assert profile_legal[0]["mma_tiler_mnk"][2] == 128

    runtime_legal = hopper_mxfp4_runtime_candidates_for_shape(
        hidden=128, intermediate=128
    )
    assert len(runtime_legal) > len(profile_legal)
    assert profile_legal[0] in runtime_legal
    assert {tactic["mma_tiler_mnk"][2] for tactic in runtime_legal} == {128}
    default = hopper_mxfp4_default_tactic(512)
    assert default not in runtime_legal
    assert (
        hopper_mxfp4_ordered_candidates(
            512,
            hidden=128,
            intermediate=128,
        )
        == runtime_legal
    )


def test_per_token_defaults_are_the_exact_manifest_winners() -> None:
    manifest = hopper_mxfp4_tuning_manifest()

    for token, expected_id in _EXPECTED_WINNER_IDS.items():
        expected = manifest["per_token_winners"][str(token)]
        actual = hopper_mxfp4_default_tactic(token)
        ordered = hopper_mxfp4_ordered_candidates(
            token,
            hidden=7168,
            intermediate=3072,
        )
        assert expected["candidate_id"] == expected_id
        assert actual == expected["candidate"]["effective_tactic"]
        assert _fused_tactic_id(actual) == expected_id
        assert ordered[0] == actual


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


@pytest.mark.parametrize(
    "profile",
    (
        MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
        MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    ),
)
def test_candidate_apis_return_fresh_copies(profile: str) -> None:
    tactics = hopper_mxfp4_candidates(routing_profile=profile)
    records = hopper_mxfp4_candidate_records(routing_profile=profile)
    default = hopper_mxfp4_default_tactic(8, routing_profile=profile)

    tactics[0].clear()
    candidate = records[0]["candidate"]
    candidate["effective_tactic"].clear()
    candidate["requested_tactic"].clear()
    if "requested_tactic_aliases" in candidate:
        candidate["requested_tactic_aliases"][0].clear()
        candidate["requested_tactic_aliases"].clear()
    records[0]["winner_for_tokens"].clear()
    default.clear()

    fresh = hopper_mxfp4_candidate_records(routing_profile=profile)[0]
    assert hopper_mxfp4_candidates(routing_profile=profile)[0]
    assert fresh["candidate"]["effective_tactic"]
    assert fresh["candidate"]["requested_tactic"]
    if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
        assert fresh["candidate"]["requested_tactic_aliases"]
    assert fresh["winner_for_tokens"]
    assert hopper_mxfp4_default_tactic(8, routing_profile=profile)


def test_fused_validator_rejects_fields_types_and_illegal_geometry() -> None:
    tactic = hopper_mxfp4_candidates()[0]

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

    frozen = hopper_mxfp4_candidate_records(
        routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    )[0]["candidate"]
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(frozen["effective_tactic"])


@pytest.mark.parametrize(
    "bad_profile",
    (
        None,
        True,
        "",
        "block_permutation",
        "published_exact_balanced",
        "BLOCK_PERMUTATION_V1",
        "legacy",
    ),
)
def test_all_profile_aware_apis_reject_noncanonical_profile(
    bad_profile: object,
) -> None:
    with pytest.raises(ValueError, match="routing_profile"):
        normalize_hopper_mxfp4_routing_profile(bad_profile)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="routing_profile"):
        normalize_sm90_routing_profile(bad_profile)

    calls = (
        lambda: hopper_mxfp4_cache_provenance_sha256(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_candidate_records(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_candidates(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_candidates_for_shape(
            hidden=7168,
            intermediate=3072,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_runtime_candidates(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_runtime_candidates_for_shape(
            hidden=7168,
            intermediate=3072,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_default_tactic(
            8,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_ordered_candidates(
            8,
            hidden=7168,
            intermediate=3072,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_tuning_manifest(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_tuning_provenance(
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="routing_profile"):
            call()


def test_validator_requires_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_hopper_mxfp4_tactic([])  # type: ignore[arg-type]


_OPTIMIZATION_SHAPE = dict(
    hidden=7168, intermediate=3072, num_experts=384, world_size=4
)


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
        original = hopper_mxfp4_runtime_candidates_for_shape(
            hidden=_OPTIMIZATION_SHAPE["hidden"],
            intermediate=_OPTIMIZATION_SHAPE["intermediate"],
        )
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
                actual = hopper_mxfp4_optimization_candidates(
                    token, routing_profile=profile, **_OPTIMIZATION_SHAPE
                )
                original = hopper_mxfp4_ordered_candidates(
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
        candidates = hopper_mxfp4_optimization_candidates(2048, **_OPTIMIZATION_SHAPE)
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
                base = hopper_mxfp4_ordered_candidates(
                    2048, hidden=shape["hidden"], intermediate=shape["intermediate"]
                )
                previous = expand_mxfp4_optimization_candidates(base, **shape)
                actual = hopper_mxfp4_optimization_candidates(2048, **shape)
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

    def test_historical_fused_manifests_are_unchanged(self):
        for profile in MXFP4_TUNING_ROUTING_PROFILES:
            manifest = hopper_mxfp4_tuning_manifest(routing_profile=profile)
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            expected = hopper_mxfp4_tuning_provenance(routing_profile=profile)
            self.assertEqual(
                hashlib.sha256(raw).hexdigest(), expected["runtime_manifest_sha256"]
            )
            for token in MXFP4_TUNING_TOKEN_BUCKETS:
                tactic = hopper_mxfp4_default_tactic(token, routing_profile=profile)
                self.assertEqual(
                    tactic,
                    validate_hopper_mxfp4_tactic(tactic),
                )


@pytest.fixture
def allow_mock_tuning_device(monkeypatch):
    # Only wrapper tests use this bypass; real device-guard tests stay active.
    for module in (sm90_mega, mxfp4_tuner):
        monkeypatch.setattr(
            module, "require_hopper_mxfp4_fused_tuning_device", lambda: None
        )


def test_fused_full_union_records_only_fused_identity_and_manifest(
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
    candidates = hopper_mxfp4_optimization_candidates(
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
    provenance = hopper_mxfp4_tuning_provenance(
        routing_profile=cfg.routing_profile,
    )
    assert provenance["runtime_manifest_sha256"] in kwargs["source"]
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
    union = hopper_mxfp4_runtime_candidates(
        routing_profile=cfg.routing_profile,
    )
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

    full = hopper_mxfp4_optimization_candidates(
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
    fused = mxfp4_tuner.hopper_mxfp4_candidates()[0]
    assert hopper_mxfp4._resolve_mxfp4_knobs(
        fused,
        **_resolver_kwargs(),
    ) == normalize_mxfp4_optimization_tactic(fused)
    fused_guard.assert_not_called()
