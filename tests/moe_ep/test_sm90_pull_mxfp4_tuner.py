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
    MXFP4_FUSED_RUNTIME_ANCHOR_PROVENANCE,
    MXFP4_FUSED_RUNTIME_CANDIDATE_UNION_SHA256,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE,
    MXFP4_TUNING_ROUTING_PROFILES,
    hopper_mxfp4_cache_provenance_sha256,
    hopper_mxfp4_tuning_provenance,
    hopper_mxfp4_runtime_candidates,
    hopper_mxfp4_runtime_candidates_for_shape,
    normalize_hopper_mxfp4_routing_profile,
)


_EXPECTED_BLOCK_PROVENANCE = {
    "fused": {
        "artifact_manifest_sha256": "afc2ce271c401815a11b7572ad3afb5b4c0306bce1b42c6bad84676c2bdfc9eb",
        "candidate_union_sha256": "61eeedd983d879fdb328f4a1356277df0899dbf59e1910f73d8a2f728ae7ab8a",
        "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
        "external_schema_version": 1,
        "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
        "policy_sha256": "a6c5b46d651a63b0b03369a3b681657260c67422574c140e82b93aebd5348a2b",
        "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
        "runtime_manifest_sha256": "d1b5c856569e192cfb98d1922630e8b045c196ac8c2afff334a9bf4086a59917",
        "source_manifest_sha256": "f3c23eb7fda299da1c27980a7d358e07df925377e5adc05252661387372b6ed7",
        "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
    },
    "split": {
        "artifact_manifest_sha256": "b1443f9e91c2d6b590634a32576793a3c72574bc051fc74a4e80350315e09152",
        "candidate_union_sha256": "0b21228410f04d07427e983ac3fbee0211cc16f202c97e0c4b1a5fb86cf4bba3",
        "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
        "external_schema_version": 1,
        "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
        "policy_sha256": "a6c5b46d651a63b0b03369a3b681657260c67422574c140e82b93aebd5348a2b",
        "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
        "runtime_manifest_sha256": "8eddd6c76f588164f85507d29b1502caba91b6985208eac75df6bcc82d08012b",
        "source_manifest_sha256": "f3c23eb7fda299da1c27980a7d358e07df925377e5adc05252661387372b6ed7",
        "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
    },
}

_EXPECTED_BLOCK_RUNTIME_SHA256 = {
    mode: values["runtime_manifest_sha256"]
    for mode, values in _EXPECTED_BLOCK_PROVENANCE.items()
}
_EXPECTED_BLOCK_CANDIDATE_UNION_SHA256 = {
    mode: values["candidate_union_sha256"]
    for mode, values in _EXPECTED_BLOCK_PROVENANCE.items()
}

_EXPECTED_WINNER_IDS = {
    "fused": {
        1024: "c099f07d617c03d86ed6f2ccb4600cebc7730c8778a2678571370156b817ea22",
        128: "4438af53f28959d895bd1f940ec3b4d4123ee3ec4992e82d584c7cdd7cda067a",
        2048: "d22b11b33850233223f4de0d5991a753da9642fe5b71c2f8987f9dfbef01c3cb",
        256: "2be24c69949d07d49969003065de721a414c6a9df04a56f7cc2da04b2e9956ac",
        32: "825d3b94a222ab92411face2a75a5f771b102c79afe9daba316342b4e7d2afb3",
        512: "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
        64: "ec7439eb60ab2f2a61684de25e27e3c1e99317d5a05cdaec35ab15bf98bdc724",
        8: "81bc1d7a297f413377d6c68139cca3443c0c51e43254c4fff84786fe9cdb9bb7",
    },
    "split": {
        1024: "7a5a4dd52fcd58302627f11c684805f31adc4919e79c41c7f0f1e0acad738b4b",
        128: "dd423a9514cd09ccfc5f2d654ecaff46461aff8cfd6e9154425dac2df8875eb8",
        2048: "334c802f8f46ae273e1f266589d8021e2d69099f033172e34ee32faf285cffbe",
        256: "27be572d1a7922957e20322422ff110dd8b325740315448c97bf9e70074da143",
        32: "5f63938822a1b32d8072b2c1ba5abf38f85362a27c789228b59191f663353f5e",
        512: "748ef68a8b715647dbf20cb8e9055bfde13170ef24981c4b22d0deca6e727efe",
        64: "ae61ce870a47537c25f29f360f40c4a97847af58db3b2b0bdf63c69eacdff0e0",
        8: "f2c1371a49be70076fb7b04007e3db0c6272e12344369af9146a77e99047de5e",
    },
}
_EXPECTED_BLOCK_CANDIDATE_IDS = {
    mode: tuple(sorted(token_winners.values()))
    for mode, token_winners in _EXPECTED_WINNER_IDS.items()
}

_EXPECTED_FUSED_RUNTIME_CANDIDATE_IDS = (
    "1630f29e2346f2d817477c82aa520672961445712ec504cd8591581169e5c626",
    "170e811b97e04d5e5a2795335891fd647a404a6e8bc24f1cf4b5c701e293c8aa",
    "2be24c69949d07d49969003065de721a414c6a9df04a56f7cc2da04b2e9956ac",
    "38265fb512a81b2b69d9ca7b156601a3e5363b2e15ba237615a9f4d3900813d6",
    "3df511dac7954f726e9002cb3cbbe86ce5b5479ca43e243a0d890cc48f6921d2",
    "4438af53f28959d895bd1f940ec3b4d4123ee3ec4992e82d584c7cdd7cda067a",
    "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
    "78a5836d443416669b66535372b920ffe1dfe252903d350afdc12699f7b5046e",
    "81bc1d7a297f413377d6c68139cca3443c0c51e43254c4fff84786fe9cdb9bb7",
    "825d3b94a222ab92411face2a75a5f771b102c79afe9daba316342b4e7d2afb3",
    "89927bf6af211d5047bc8a2e6be10ae892c65c85d2848ade50c742a47df3ee56",
    "b66df770384bc5c406d13b0d71d6f701aec6879498ed9d9e2edc209db6522c2f",
    "c099f07d617c03d86ed6f2ccb4600cebc7730c8778a2678571370156b817ea22",
    "d22b11b33850233223f4de0d5991a753da9642fe5b71c2f8987f9dfbef01c3cb",
    "e2c39aa656b37beeb8dbc5a1533ede68a09fe7fb1101581bfab210a63c207b1f",
    "ec7439eb60ab2f2a61684de25e27e3c1e99317d5a05cdaec35ab15bf98bdc724",
    "f9e97da647c04c0999bb38fe7a5c26c30e74060e856b037c90a1bd661136c341",
)
_FOLDED_FUSED_CANDIDATE_IDS = {
    "7140a7c4d125b7c36e1ce588f94f33af5de68d371baa072b5cc33f9386b2870e",
    "c099f07d617c03d86ed6f2ccb4600cebc7730c8778a2678571370156b817ea22",
    "d22b11b33850233223f4de0d5991a753da9642fe5b71c2f8987f9dfbef01c3cb",
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

_H20_FUSED_RUNTIME_ANCHOR_IDS = {
    "1630f29e2346f2d817477c82aa520672961445712ec504cd8591581169e5c626",
    "89927bf6af211d5047bc8a2e6be10ae892c65c85d2848ade50c742a47df3ee56",
}
_EXPECTED_FUSED_RUNTIME_CANDIDATE_UNION_SHA256 = (
    "3d458e8530fec250635bb3e9ead61ed3fa9b8243f85590d6827d5f22c1d809e5"
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


def _tactic_id(mode: str, tactic: object) -> str:
    raw = json.dumps(
        {"implementation": f"mxfp4_{mode}", "tactic": tactic},
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


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_embedded_manifest_is_byte_canonical(mode: str) -> None:
    manifest = hopper_mxfp4_tuning_manifest(execution_mode=mode)
    provenance = hopper_mxfp4_tuning_provenance(execution_mode=mode)

    assert _manifest_sha256(manifest) == _EXPECTED_BLOCK_RUNTIME_SHA256[mode]
    assert dict(provenance) == _EXPECTED_BLOCK_PROVENANCE[mode]
    assert (
        manifest["artifact_manifest_sha256"] == provenance["artifact_manifest_sha256"]
    )
    assert manifest["external_schema_version"] == 1
    assert manifest["implementation"] == f"mxfp4_{mode}"
    assert manifest["routing_profile"] == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE
    assert manifest["runtime_schema_version"] == 1
    assert (
        _candidate_union_sha256(manifest["candidate_union"])
        == (_EXPECTED_BLOCK_CANDIDATE_UNION_SHA256[mode])
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
    assert set(MXFP4_TUNING_PROVENANCE) == {"fused", "split"}
    assert {
        mode: dict(MXFP4_TUNING_PROVENANCE[mode]) for mode in ("fused", "split")
    } == _EXPECTED_BLOCK_PROVENANCE
    assert (
        MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[
            MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE
        ]
        == MXFP4_TUNING_PROVENANCE
    )


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_explicit_block_profile_is_identical_to_omitted_profile(mode: str) -> None:
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
    assert all(
        "requested_tactic_aliases" not in record["candidate"]
        for record in manifest["candidate_union"]
    )
    if mode == "split":
        assert provenance["candidate_union_sha256"] == (
            "210adb840e66c0f44949ea866a8bbfa9f5b2b3835ee8992e57b5edb75f8b9321"
        )
        assert "policy_sha256" not in provenance
        assert "per_token_provenance" not in manifest


def test_cache_provenance_is_deterministic_hex_and_domain_scoped() -> None:
    fingerprints = {
        (mode, profile): hopper_mxfp4_cache_provenance_sha256(
            execution_mode=mode,
            routing_profile=profile,
        )
        for mode in ("fused", "split")
        for profile in MXFP4_TUNING_ROUTING_PROFILES
    }

    assert len(set(fingerprints.values())) == len(fingerprints)
    assert all(
        len(value) == 64 and set(value) <= set("0123456789abcdef")
        for value in fingerprints.values()
    )
    assert fingerprints == {
        identity: hopper_mxfp4_cache_provenance_sha256(
            execution_mode=identity[0],
            routing_profile=identity[1],
        )
        for identity in fingerprints
    }


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
        if mode == "fused":
            assert (
                _frozen_fused_projection(tactic)
                == record["candidate"]["effective_tactic"]
            )
            assert set(tactic) == (
                set(record["candidate"]["effective_tactic"]) | _FUSED_LAYOUT_FIELDS
            )
        else:
            assert record["candidate"]["effective_tactic"] == tactic
        assert is_valid_hopper_mxfp4_tactic(tactic, execution_mode=mode)


def test_fused_runtime_union_is_routing_independent_and_provenance_tracked() -> None:
    block = hopper_mxfp4_runtime_candidates(
        execution_mode="fused",
        routing_profile=MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    )
    exact = hopper_mxfp4_runtime_candidates(
        execution_mode="fused",
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
                execution_mode="fused",
                routing_profile=profile,
            )
        )


def test_h20_runtime_anchors_are_shape_legal_but_not_heuristic_winners() -> None:
    legal = hopper_mxfp4_runtime_candidates_for_shape(
        execution_mode="fused",
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
            execution_mode="fused",
            routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
        )
        not in anchors
        for token in MXFP4_TUNING_TOKEN_BUCKETS
    )


@pytest.mark.parametrize("profile", MXFP4_TUNING_ROUTING_PROFILES)
def test_split_runtime_union_remains_profile_specific_and_frozen(profile: str) -> None:
    assert hopper_mxfp4_runtime_candidates(
        execution_mode="split", routing_profile=profile
    ) == hopper_mxfp4_candidates(execution_mode="split", routing_profile=profile)


@pytest.mark.parametrize("mode", ("fused", "split"))
def test_published_exact_default_leads_runtime_ordering(mode: str) -> None:
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
        if mode == "fused":
            assert (
                _frozen_fused_projection(default)
                == winner["candidate"]["effective_tactic"]
            )
        else:
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
    candidate_ids = tuple(record["candidate"]["candidate_id"] for record in records)

    assert len(records) == len(tactics) == 8
    assert candidate_ids == _EXPECTED_BLOCK_CANDIDATE_IDS[mode]
    assert candidate_ids == tuple(sorted(set(candidate_ids)))
    assert {
        token for record in records for token in record["winner_for_tokens"]
    } == set(MXFP4_TUNING_TOKEN_BUCKETS)
    for record, tactic in zip(records, tactics, strict=True):
        candidate = record["candidate"]
        candidate_id = candidate["candidate_id"]
        requested = candidate["requested_tactic"]
        effective = candidate["effective_tactic"]
        assert candidate["implementation"] == f"mxfp4_{mode}"
        assert candidate["requested_tactic_aliases"] == [requested]
        assert effective == tactic
        assert _tactic_id(mode, effective) == candidate_id
        if mode == "fused" and candidate_id in _FOLDED_FUSED_CANDIDATE_IDS:
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
        assert validate_hopper_mxfp4_tactic(tactic, execution_mode=mode) == tactic
        assert is_valid_hopper_mxfp4_tactic(tactic, execution_mode=mode)


def test_h128_fused_uses_cross_profile_runtime_fallback() -> None:
    with pytest.raises(ValueError, match="no manifest-derived MXFP4 fused tactic"):
        hopper_mxfp4_candidates_for_shape(
            execution_mode="fused", hidden=128, intermediate=128
        )

    legal = hopper_mxfp4_runtime_candidates_for_shape(
        execution_mode="fused", hidden=128, intermediate=128
    )
    assert legal
    assert {tactic["mma_tiler_mnk"][2] for tactic in legal} == {128}
    default = hopper_mxfp4_default_tactic(512, execution_mode="fused")
    assert default not in legal
    assert (
        hopper_mxfp4_ordered_candidates(
            512,
            execution_mode="fused",
            hidden=128,
            intermediate=128,
        )
        == legal
    )


def test_h128_split_profile_has_a_stable_legal_fallback() -> None:
    legal = hopper_mxfp4_candidates_for_shape(
        execution_mode="split", hidden=128, intermediate=128
    )
    assert legal
    assert {tactic["k1_mma_tiler_mnk"][2] for tactic in legal} == {128}
    assert {tactic["k2_mma_tiler_mnk"][2] for tactic in legal} == {128}
    assert all(
        is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode="split",
            hidden=128,
            intermediate=128,
        )
        for tactic in legal
    )
    default = hopper_mxfp4_default_tactic(512, execution_mode="split")
    assert default not in legal
    assert (
        hopper_mxfp4_ordered_candidates(
            512,
            execution_mode="split",
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
        ordered = hopper_mxfp4_ordered_candidates(
            token,
            execution_mode=mode,
            hidden=7168,
            intermediate=3072,
        )
        assert expected["candidate_id"] == expected_id
        assert actual == expected["candidate"]["effective_tactic"]
        assert _tactic_id(mode, actual) == expected_id
        assert ordered[0] == actual


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
    candidate = records[0]["candidate"]
    candidate["effective_tactic"].clear()
    candidate["requested_tactic"].clear()
    if "requested_tactic_aliases" in candidate:
        candidate["requested_tactic_aliases"][0].clear()
        candidate["requested_tactic_aliases"].clear()
    records[0]["winner_for_tokens"].clear()
    default.clear()

    fresh = hopper_mxfp4_candidate_records(
        execution_mode=mode, routing_profile=profile
    )[0]
    assert hopper_mxfp4_candidates(execution_mode=mode, routing_profile=profile)[0]
    assert fresh["candidate"]["effective_tactic"]
    assert fresh["candidate"]["requested_tactic"]
    if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
        assert fresh["candidate"]["requested_tactic_aliases"]
    assert fresh["winner_for_tokens"]
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
            execution_mode="fused",
        )

    frozen = hopper_mxfp4_candidate_records(
        execution_mode="fused",
        routing_profile=MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
    )[0]["candidate"]
    with pytest.raises(ValueError, match="fields differ"):
        validate_hopper_mxfp4_tactic(frozen["effective_tactic"], execution_mode="fused")


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
        hopper_mxfp4_cache_provenance_sha256(execution_mode=bad_mode)
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
        lambda: hopper_mxfp4_cache_provenance_sha256(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
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
        lambda: hopper_mxfp4_runtime_candidates(
            execution_mode="fused",
            routing_profile=bad_profile,  # type: ignore[arg-type]
        ),
        lambda: hopper_mxfp4_runtime_candidates_for_shape(
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
