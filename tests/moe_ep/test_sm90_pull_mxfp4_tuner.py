# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import hashlib
import json

import pytest

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
    MXFP4_TUNING_PROVENANCE,
    MXFP4_TUNING_TOKEN_BUCKETS,
    hopper_mxfp4_candidate_records,
    hopper_mxfp4_candidates,
    hopper_mxfp4_candidates_for_shape,
    hopper_mxfp4_default_tactic,
    hopper_mxfp4_ordered_candidates,
    hopper_mxfp4_tuning_manifest,
    is_hopper_mxfp4_tactic_shape_compatible,
    is_valid_hopper_mxfp4_tactic,
    validate_hopper_mxfp4_tactic,
)
from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.mxfp4_tuner import (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE,
    MXFP4_TUNING_ROUTING_PROFILES,
    hopper_mxfp4_tuning_provenance,
    normalize_hopper_mxfp4_routing_profile,
)


_EXPECTED_MANIFEST_SHA256 = {
    "fused": "455cf75bdd0c0011184ee5a3f48eab9ac80782b4824562cd796005887a19d1cf",
    "split": "1c350f333d365ef6284b23e1604faaa3388ba00f3cb82c63f686515778700f93",
}

_EXPECTED_WINNER_IDS = {
    "fused": {
        8: "104f2e89fd468598df2376c3d937728bfaa96b465676b5b8fafb53ee9a33878b",
        32: "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
        64: "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
        128: "c08058def16c3d732495cda3ad1ee2cd7244fb4d90469d1ca40cc3a196655563",
        256: "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
        512: "b18dae894b086cd7f0d4d22fb4cfdd28682015ad372c1cb2b3985bad50a12f03",
        1024: "331de4723cbac173f1786ad0974bd4233d4a996167e40cb4c559a0c772c4a63f",
        2048: "48a466d13815abe8e6d47885098a80e9cdbd58e057a5c38b67b233f733fc5226",
    },
    "split": {
        8: "b16e1cfffbd2796e4b4b8583276c6b9996b1a0e6461a7108e287e421b2353852",
        32: "2cd398370b552092dcaeeeb1ad79033cc1078b9cb234faa428c161708af3d60d",
        64: "2a98249053aa02c8c814db675d107323b73a420a6b6cf7bb8a6cfba195db1721",
        128: "31ddb01984acf3a18856e5e089af43974a36d8892399c50f6307f0f41f03b429",
        256: "5787644b0c7cf45f9c814a2eddb8ac951894db12d5335e0b4dcf854e245860e3",
        512: "7cd01fddb88a41bb1b5fe366cd775e8604424757b68df8e861cd2ef955ce7cce",
        1024: "e97d5814f9a03f7144d058014a4ad78c744dde50fb05a96ab02b1db643a0f905",
        2048: "6a0b4fd0b67832b45edebf656d670f57e0cc973c8b905de819e1ae546aff1ecb",
    },
}

_EXPECTED_EXACT_ARTIFACT_SHA256 = {
    "fused": "62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61",
    "split": "094d840c579a7331439d1acd50690909ad2c88e6085253326c6f2d98ddad248a",
}

_EXPECTED_EXACT_RUNTIME_SHA256 = {
    "fused": "f4112c7d0d7ead640239c1df3d7f4af74e2a1fb35cf5e821edd1beba9bba3e99",
    "split": "97a4a40bffeb062b9cc916959186e308bb8e3ff52150dab8a01b3273af22261f",
}

_EXPECTED_EXACT_CANDIDATE_IDS = {
    "fused": (
        "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
        "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
        "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
        "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
        "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
        "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
        "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
    ),
    "split": (
        "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
        "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
        "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
        "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
        "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
        "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
        "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
        "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
    ),
}

_EXPECTED_EXACT_WINNER_IDS = {
    "fused": {
        8: "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
        32: "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
        64: "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
        128: "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
        256: "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
        512: "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
        1024: "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
        2048: "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
    },
    "split": {
        8: "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
        32: "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
        64: "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
        128: "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
        256: "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
        512: "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
        1024: "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
        2048: "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
    },
}


def _manifest_sha256(value: object) -> str:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    return hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_embedded_manifest_is_byte_canonical(mode: str) -> None:
    manifest = hopper_mxfp4_tuning_manifest(execution_mode=mode)

    assert _manifest_sha256(manifest) == _EXPECTED_MANIFEST_SHA256[mode]
    assert (
        MXFP4_TUNING_PROVENANCE[mode]["manifest_sha256"]
        == (_EXPECTED_MANIFEST_SHA256[mode])
    )
    assert manifest["implementation"] == f"mxfp4_{mode}"
    assert tuple(int(token) for token in manifest["per_token_winners"]) != ()
    assert set(map(int, manifest["per_token_winners"])) == set(
        MXFP4_TUNING_TOKEN_BUCKETS
    )


def test_legacy_profile_identity_and_public_provenance_shape_are_unchanged() -> None:
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
    assert set(MXFP4_TUNING_PROVENANCE) == {"fused", "split"}
    assert {
        mode: MXFP4_TUNING_PROVENANCE[mode]["manifest_sha256"]
        for mode in ("fused", "split")
    } == _EXPECTED_MANIFEST_SHA256
    assert (
        MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[
            MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE
        ]
        == MXFP4_TUNING_PROVENANCE
    )


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_explicit_legacy_profile_is_identical_to_omitted_profile(mode: str) -> None:
    profile = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE

    assert hopper_mxfp4_candidate_records(
        execution_mode=mode, routing_profile=profile
    ) == hopper_mxfp4_candidate_records(execution_mode=mode)
    assert hopper_mxfp4_candidates(
        execution_mode=mode, routing_profile=profile
    ) == hopper_mxfp4_candidates(execution_mode=mode)
    assert hopper_mxfp4_candidates_for_shape(
        execution_mode=mode,
        hidden=7168,
        intermediate=3072,
        routing_profile=profile,
    ) == hopper_mxfp4_candidates_for_shape(
        execution_mode=mode, hidden=7168, intermediate=3072
    )
    assert hopper_mxfp4_default_tactic(
        512, execution_mode=mode, routing_profile=profile
    ) == hopper_mxfp4_default_tactic(512, execution_mode=mode)
    assert hopper_mxfp4_ordered_candidates(
        512,
        execution_mode=mode,
        hidden=7168,
        intermediate=3072,
        routing_profile=profile,
    ) == hopper_mxfp4_ordered_candidates(
        512, execution_mode=mode, hidden=7168, intermediate=3072
    )
    assert hopper_mxfp4_tuning_manifest(
        execution_mode=mode, routing_profile=profile
    ) == hopper_mxfp4_tuning_manifest(execution_mode=mode)
    assert (
        hopper_mxfp4_tuning_provenance(execution_mode=mode, routing_profile=profile)
        == MXFP4_TUNING_PROVENANCE[mode]
    )


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_published_exact_manifest_and_provenance_are_frozen(mode: str) -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    manifest = hopper_mxfp4_tuning_manifest(
        execution_mode=mode, routing_profile=profile
    )
    provenance = hopper_mxfp4_tuning_provenance(
        execution_mode=mode, routing_profile=profile
    )

    assert (
        manifest["artifact_manifest_sha256"] == (_EXPECTED_EXACT_ARTIFACT_SHA256[mode])
    )
    assert manifest["routing_profile"] == profile
    assert manifest["runtime_schema_version"] == 1
    assert manifest["external_schema_version"] == (1 if mode == "fused" else 2)
    assert _manifest_sha256(manifest) == _EXPECTED_EXACT_RUNTIME_SHA256[mode]
    assert (
        provenance["artifact_manifest_sha256"]
        == (_EXPECTED_EXACT_ARTIFACT_SHA256[mode])
    )
    assert (
        provenance["runtime_manifest_sha256"] == (_EXPECTED_EXACT_RUNTIME_SHA256[mode])
    )
    assert set(map(int, manifest["per_token_winners"])) == set(
        MXFP4_TUNING_TOKEN_BUCKETS
    )
    if mode == "split":
        assert provenance["candidate_union_sha256"] == (
            "210adb840e66c0f44949ea866a8bbfa9f5b2b3835ee8992e57b5edb75f8b9321"
        )
        assert "policy_sha256" not in provenance
        assert "per_token_provenance" not in manifest


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_published_exact_union_is_legal_complete_and_profile_isolated(
    mode: str,
) -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    records = hopper_mxfp4_candidate_records(
        execution_mode=mode, routing_profile=profile
    )
    tactics = hopper_mxfp4_candidates(execution_mode=mode, routing_profile=profile)
    candidate_ids = tuple(record["candidate"]["candidate_id"] for record in records)

    assert candidate_ids == _EXPECTED_EXACT_CANDIDATE_IDS[mode]
    assert candidate_ids == tuple(sorted(set(candidate_ids)))
    assert set(candidate_ids) != {
        record["candidate"]["candidate_id"]
        for record in hopper_mxfp4_candidate_records(execution_mode=mode)
    }
    assert {
        token for record in records for token in record["winner_for_tokens"]
    } == set(MXFP4_TUNING_TOKEN_BUCKETS)
    for record, tactic in zip(records, tactics, strict=True):
        assert record["candidate"]["effective_tactic"] == tactic
        assert is_valid_hopper_mxfp4_tactic(tactic, execution_mode=mode)


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_published_exact_defaults_and_ordering_use_only_exact_table(mode: str) -> None:
    profile = MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE
    manifest = hopper_mxfp4_tuning_manifest(
        execution_mode=mode, routing_profile=profile
    )

    for token, expected_id in _EXPECTED_EXACT_WINNER_IDS[mode].items():
        default = hopper_mxfp4_default_tactic(
            token, execution_mode=mode, routing_profile=profile
        )
        winner = manifest["per_token_winners"][str(token)]
        ordered = hopper_mxfp4_ordered_candidates(
            token,
            execution_mode=mode,
            hidden=7168,
            intermediate=3072,
            routing_profile=profile,
        )
        assert winner["candidate_id"] == expected_id
        assert default == winner["candidate"]["effective_tactic"]
        assert ordered[0] == default

    if mode == "fused":
        assert (
            hopper_mxfp4_default_tactic(
                1024, execution_mode=mode, routing_profile=profile
            )["token_back_mode"]
            == "reuse_dispatch_warps"
        )
        assert hopper_mxfp4_default_tactic(
            2048, execution_mode=mode, routing_profile=profile
        )["cluster_shape_mnk"] == (1, 1, 1)
    else:
        token8 = hopper_mxfp4_default_tactic(
            8, execution_mode=mode, routing_profile=profile
        )
        assert (token8["k1_sm_count"], token8["k2_sm_count"]) == (88, 44)


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_candidate_union_is_legal_sorted_deduplicated_and_complete(mode: str) -> None:
    records = hopper_mxfp4_candidate_records(execution_mode=mode)
    tactics = hopper_mxfp4_candidates(execution_mode=mode)
    candidate_ids = [record["candidate"]["candidate_id"] for record in records]

    assert len(records) == len(tactics) == 8
    assert candidate_ids == sorted(set(candidate_ids))
    assert {
        token for record in records for token in record["winner_for_tokens"]
    } == set(MXFP4_TUNING_TOKEN_BUCKETS)
    for record, tactic in zip(records, tactics, strict=True):
        candidate = record["candidate"]
        assert candidate["implementation"] == f"mxfp4_{mode}"
        assert candidate["requested_tactic"] == tactic
        assert candidate["effective_tactic"] == tactic
        assert validate_hopper_mxfp4_tactic(tactic, execution_mode=mode) == tactic
        assert is_valid_hopper_mxfp4_tactic(tactic, execution_mode=mode)


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_h128_shape_filters_out_k256_and_uses_stable_legal_fallback(
    mode: str,
) -> None:
    legal = hopper_mxfp4_candidates_for_shape(
        execution_mode=mode,
        hidden=128,
        intermediate=128,
    )
    assert legal
    assert all(
        is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode=mode,
            hidden=128,
            intermediate=128,
        )
        for tactic in legal
    )
    if mode == "fused":
        assert {tactic["mma_tiler_mnk"][2] for tactic in legal} == {128}
    else:
        assert {tactic["k1_mma_tiler_mnk"][2] for tactic in legal} == {128}
        assert {tactic["k2_mma_tiler_mnk"][2] for tactic in legal} == {128}

    default = hopper_mxfp4_default_tactic(512, execution_mode=mode)
    assert default not in legal
    assert (
        hopper_mxfp4_ordered_candidates(
            512,
            execution_mode=mode,
            hidden=128,
            intermediate=128,
        )
        == legal
    )


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_per_token_defaults_are_the_exact_manifest_winners(mode: str) -> None:
    manifest = hopper_mxfp4_tuning_manifest(execution_mode=mode)

    for token, expected_id in _EXPECTED_WINNER_IDS[mode].items():
        expected = manifest["per_token_winners"][str(token)]
        actual = hopper_mxfp4_default_tactic(token, execution_mode=mode)
        assert expected["candidate_id"] == expected_id
        assert actual == expected["candidate"]["effective_tactic"]


@pytest.mark.parametrize(
    ("max_tokens", "bucket"),
    ((1, 8), (8, 8), (9, 32), (31, 32), (33, 64), (2048, 2048), (4096, 2048)),
)
@pytest.mark.parametrize("mode", ("fused", "split"))
def test_default_uses_ceil_bucket_and_clamps_above_domain(
    mode: str, max_tokens: int, bucket: int
) -> None:
    assert hopper_mxfp4_default_tactic(
        max_tokens, execution_mode=mode
    ) == hopper_mxfp4_default_tactic(bucket, execution_mode=mode)


@pytest.mark.parametrize("bad_tokens", (True, False, 0, -1, 8.0, "8", None))
def test_default_rejects_invalid_token_count(bad_tokens: object) -> None:
    with pytest.raises(ValueError):
        hopper_mxfp4_default_tactic(bad_tokens, execution_mode="fused")  # type: ignore[arg-type]


def test_fused_and_split_candidate_types_are_isolated() -> None:
    fused = hopper_mxfp4_candidates(execution_mode="fused")[0]
    split = hopper_mxfp4_candidates(execution_mode="split")[0]

    assert not is_valid_hopper_mxfp4_tactic(fused, execution_mode="split")
    assert not is_valid_hopper_mxfp4_tactic(split, execution_mode="fused")
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(fused, execution_mode="split")
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(split, execution_mode="fused")


@pytest.mark.parametrize(
    "profile",
    (
        MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
        MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    ),
)
@pytest.mark.parametrize("mode", ("fused", "split"))
def test_candidate_apis_return_fresh_copies(mode: str, profile: str) -> None:
    tactics = hopper_mxfp4_candidates(execution_mode=mode, routing_profile=profile)
    records = hopper_mxfp4_candidate_records(
        execution_mode=mode, routing_profile=profile
    )
    default = hopper_mxfp4_default_tactic(
        8, execution_mode=mode, routing_profile=profile
    )

    tactics[0].clear()
    records[0]["candidate"]["effective_tactic"].clear()
    records[0]["winner_for_tokens"].clear()
    default.clear()

    assert hopper_mxfp4_candidates(execution_mode=mode, routing_profile=profile)[0]
    assert hopper_mxfp4_candidate_records(execution_mode=mode, routing_profile=profile)[
        0
    ]["winner_for_tokens"]
    assert hopper_mxfp4_default_tactic(8, execution_mode=mode, routing_profile=profile)


def test_fused_validator_rejects_fields_types_and_illegal_geometry() -> None:
    tactic = hopper_mxfp4_candidates(execution_mode="fused")[0]

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
    ):
        malformed = {**tactic, key: value}
        with pytest.raises(ValueError, match=message):
            validate_hopper_mxfp4_tactic(malformed, execution_mode="fused")

    pingpong = {**tactic, "pingpong": True}
    with pytest.raises(ValueError, match="requires M128"):
        validate_hopper_mxfp4_tactic(pingpong, execution_mode="fused")
    missing = dict(tactic)
    missing.pop("group_hint")
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(missing, execution_mode="fused")
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic({**tactic, "unknown": 1}, execution_mode="fused")


def test_split_validator_rejects_protocol_unsafe_tactics() -> None:
    tactic = hopper_mxfp4_candidates(execution_mode="split")[0]

    malformed_tactics = (
        ({**tactic, "k1_mma_tiler_mnk": (64, 16, 128)}, "split K1 tile"),
        ({**tactic, "k2_mma_tiler_mnk": (256, 8, 128)}, "split K2 tile"),
        ({**tactic, "k2_cluster_shape_mnk": (2, 1, 1)}, "clusters must match"),
        (
            {
                **tactic,
                "k1_cluster_shape_mnk": (2, 1, 1),
                "k2_cluster_shape_mnk": (2, 1, 1),
            },
            "quarantined",
        ),
        ({**tactic, "k1_sm_count": 79, "k2_sm_count": 53}, "partition"),
        ({**tactic, "k1_sm_count": 80, "k2_sm_count": 51}, "partition"),
        ({**tactic, "counter_epoch_banks": True}, "counter banks"),
        ({**tactic, "counter_epoch_banks": 3}, "counter banks"),
        ({**tactic, "graph_variant": "bad"}, "graph variant"),
        (
            {
                **tactic,
                "counter_epoch_banks": 2,
                "graph_variant": "cold_k0",
            },
            "require steady_k3_reset",
        ),
        ({**tactic, "enable_iket": True}, "IKET"),
    )
    for malformed, message in malformed_tactics:
        with pytest.raises(ValueError, match=message):
            validate_hopper_mxfp4_tactic(malformed, execution_mode="split")


def test_split_validator_accepts_an_explicit_78_sm_partition() -> None:
    tactic = hopper_mxfp4_candidates(execution_mode="split")[0]
    h20_tactic = {
        **tactic,
        "k1_sm_count": 48,
        "k2_sm_count": 30,
    }
    assert (
        validate_hopper_mxfp4_tactic(
            h20_tactic,
            execution_mode="split",
            total_sms=78,
        )
        == h20_tactic
    )
    with pytest.raises(ValueError, match="sum to 132 SMs"):
        validate_hopper_mxfp4_tactic(h20_tactic, execution_mode="split")


@pytest.mark.parametrize("bad_mode", ("mxfp4_fused", "mxfp4_split", "fp8", ""))
def test_public_apis_reject_noncanonical_execution_mode(bad_mode: str) -> None:
    with pytest.raises(ValueError, match="execution_mode"):
        hopper_mxfp4_candidates(execution_mode=bad_mode)
    with pytest.raises(ValueError, match="execution_mode"):
        hopper_mxfp4_default_tactic(8, execution_mode=bad_mode)


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
        lambda: hopper_mxfp4_candidate_records(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_candidates(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_candidates_for_shape(
            execution_mode="fused",
            hidden=7168,
            intermediate=3072,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_default_tactic(
            8,
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_ordered_candidates(
            8,
            execution_mode="fused",
            hidden=7168,
            intermediate=3072,
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_tuning_manifest(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_tuning_provenance(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
    )
    for call in calls:
        with pytest.raises(ValueError, match="routing_profile"):
            call()


def test_validator_requires_a_mapping() -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_hopper_mxfp4_tactic([], execution_mode="fused")  # type: ignore[arg-type]
