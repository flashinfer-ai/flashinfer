# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""H200 MXFP4 offline winners and compact online-autotune candidates.

This module is deliberately data-only: it does not import torch, initialize a
process group, compile kernels, or touch the persistent knob cache.  The tables
are the exact result of the formal, fresh-process offline sweeps documented by
``MXFP4_TUNING_PROVENANCE``.  Backend integration can therefore share one
strict source of truth for both the no-autotune heuristic and the bounded
collective-autotune candidate set.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


Mxfp4ExecutionMode = Literal["fused", "split"]
Mxfp4RoutingProfile = Literal["block_permutation_v1", "published_exact_balanced_v1"]

MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
)
MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
)
MXFP4_TUNING_ROUTING_PROFILES = (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
)

MXFP4_TUNING_TOKEN_BUCKETS = (8, 32, 64, 128, 256, 512, 1024, 2048)


def require_hopper_mxfp4_tuning_device() -> None:
    """Fail closed unless the manifest H200 tuning domain is exact."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hopper MXFP4 cache/heuristic/autotune requires a CUDA device"
        )
    device = torch.cuda.current_device()
    name = str(torch.cuda.get_device_name(device))
    capability = tuple(int(v) for v in torch.cuda.get_device_capability(device))
    properties = torch.cuda.get_device_properties(device)
    sm_count = int(properties.multi_processor_count)
    expected = ("NVIDIA H200", (9, 0), 132)
    actual = (name, capability, sm_count)
    if actual != expected:
        raise RuntimeError(
            "Hopper MXFP4 cache/heuristic/autotune candidates are certified "
            f"only for standard NVIDIA H200, CC 9.0, 132 SM; got {actual!r}"
        )


_IMPLEMENTATION = {
    "fused": "mxfp4_fused",
    "split": "mxfp4_split",
}

_PROVENANCE = {
    "fused": {
        "manifest_sha256": (
            "455cf75bdd0c0011184ee5a3f48eab9ac80782b4824562cd796005887a19d1cf"
        ),
        "source_manifest_sha256": (
            "88f902a961a524c30f2fd950247d8b5d530f2a70cb840b6565d1298d94c71b34"
        ),
        "domain_sha256": (
            "188e59b1e2ebc5935cab21e1fec313823664e7c63f48b74235aae826f59478ef"
        ),
        "policy_sha256": (
            "c7244566d17512d8c651240cab97babb03132de9f98154b589ea130236f6b81b"
        ),
        "workload_recipe_sha256": (
            "813b6dd74140c4addc78845496e09a2a7025dc0c6ac4ce0ba5f64dc4cc80f667"
        ),
        "schema_version": 1,
    },
    "split": {
        "manifest_sha256": (
            "1c350f333d365ef6284b23e1604faaa3388ba00f3cb82c63f686515778700f93"
        ),
        "source_manifest_sha256": (
            "88f902a961a524c30f2fd950247d8b5d530f2a70cb840b6565d1298d94c71b34"
        ),
        "domain_sha256": (
            "188e59b1e2ebc5935cab21e1fec313823664e7c63f48b74235aae826f59478ef"
        ),
        "policy_sha256": (
            "585b6012416507d9af29293db93451c61dabb30723fd7f2264af95133c502910"
        ),
        "workload_recipe_sha256": (
            "813b6dd74140c4addc78845496e09a2a7025dc0c6ac4ce0ba5f64dc4cc80f667"
        ),
        "schema_version": 1,
    },
}

MXFP4_TUNING_PROVENANCE = MappingProxyType(
    {mode: MappingProxyType(values) for mode, values in _PROVENANCE.items()}
)

_FUSED_FIELDS = frozenset(
    {
        "swap_ab",
        "pingpong",
        "mma_tiler_mnk",
        "cluster_shape_mnk",
        "fp8_accum_mode",
        "load_balance_mode",
        "token_back_mode",
        "group_hint",
        "num_sched_stages",
        "in_kernel_fc2_reduce",
    }
)

_SPLIT_FIELDS = frozenset(
    {
        "k1_mma_tiler_mnk",
        "k2_mma_tiler_mnk",
        "k1_cluster_shape_mnk",
        "k2_cluster_shape_mnk",
        "k1_group_hint",
        "k2_group_hint",
        "k1_num_sched_stages",
        "k2_num_sched_stages",
        "k1_sm_count",
        "k2_sm_count",
        "counter_epoch_banks",
        "graph_variant",
        "enable_iket",
    }
)


def _fused_tactic(
    *,
    tile: tuple[int, int, int],
    cluster: tuple[int, int, int] = (2, 1, 1),
    group_hint: int,
    stages: int,
    load_balance: str = "atomic_counter",
    token_back: str = "epi_warps",
) -> dict[str, Any]:
    return {
        "cluster_shape_mnk": cluster,
        "fp8_accum_mode": "1xacc",
        "group_hint": group_hint,
        "in_kernel_fc2_reduce": False,
        "load_balance_mode": load_balance,
        "mma_tiler_mnk": tile,
        "num_sched_stages": stages,
        "pingpong": False,
        "swap_ab": True,
        "token_back_mode": token_back,
    }


def _split_tactic(
    *,
    banks: int,
    k1_group: int,
    k1_tile: tuple[int, int, int],
    k1_stages: int,
    k2_group: int,
    k2_tile: tuple[int, int, int],
    k2_stages: int,
    k1_sm_count: int = 80,
    k2_sm_count: int = 52,
) -> dict[str, Any]:
    return {
        "counter_epoch_banks": banks,
        "enable_iket": False,
        "graph_variant": "steady_k3_reset",
        "k1_cluster_shape_mnk": (1, 1, 1),
        "k1_group_hint": k1_group,
        "k1_mma_tiler_mnk": k1_tile,
        "k1_num_sched_stages": k1_stages,
        "k1_sm_count": k1_sm_count,
        "k2_cluster_shape_mnk": (1, 1, 1),
        "k2_group_hint": k2_group,
        "k2_mma_tiler_mnk": k2_tile,
        "k2_num_sched_stages": k2_stages,
        "k2_sm_count": k2_sm_count,
    }


# Candidate order is the lexicographically sorted candidate_id order from each
# formal manifest.  Keeping source and parent ids makes every online candidate
# traceable to its offline discovery stage.
_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "fused": (
        {
            "candidate_id": "104f2e89fd468598df2376c3d937728bfaa96b465676b5b8fafb53ee9a33878b",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=264, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "c84ec1b7d331167ffd05d3f76506c0fc688d7675a8cdf1a6768007e2f7ef2308",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "tactic": _fused_tactic(tile=(256, 32, 128), group_hint=396, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "61b9096beff96177d07bf68975d6a418189d52259a8a95ef55e6ef23b65c13c4",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "331de4723cbac173f1786ad0974bd4233d4a996167e40cb4c559a0c772c4a63f",
            "tactic": _fused_tactic(tile=(256, 32, 256), group_hint=396, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "cb3f76a06787bc6ad612ccd10dc8e107ab689c944f6b75a1a8aacf583af2b3e2",
            ),
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=512, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            ),
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "48a466d13815abe8e6d47885098a80e9cdbd58e057a5c38b67b233f733fc5226",
            "tactic": _fused_tactic(tile=(256, 32, 256), group_hint=512, stages=2),
            "source": "mandatory:t512_candidate_b_global",
            "parent_candidate_ids": (),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=396, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "f3bded653245824c8cd8a1b67605608757b5df569da3dca409302e94e7baf7a4",
            ),
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "b18dae894b086cd7f0d4d22fb4cfdd28682015ad372c1cb2b3985bad50a12f03",
            "tactic": _fused_tactic(tile=(256, 32, 256), group_hint=330, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "cb3f76a06787bc6ad612ccd10dc8e107ab689c944f6b75a1a8aacf583af2b3e2",
            ),
            "winner_for_tokens": (512,),
        },
        {
            "candidate_id": "c08058def16c3d732495cda3ad1ee2cd7244fb4d90469d1ca40cc3a196655563",
            "tactic": _fused_tactic(
                tile=(256, 16, 256),
                group_hint=512,
                stages=1,
                load_balance="static",
            ),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "68d2caf7a013e7c19a0644b5e93a19cf2bc05dc996d73e660198d42b16cd7624",
            ),
            "winner_for_tokens": (128,),
        },
    ),
    "split": (
        {
            "candidate_id": "2a98249053aa02c8c814db675d107323b73a420a6b6cf7bb8a6cfba195db1721",
            "tactic": _split_tactic(
                banks=1,
                k1_group=512,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=3,
            ),
            "source": "split_stages",
            "parent_candidate_ids": (
                "c265a07b8711cee3f3d7a4aa90464c5f0c095fd9efd18b27c0d1a783fbdb2f07",
            ),
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "2cd398370b552092dcaeeeb1ad79033cc1078b9cb234faa428c161708af3d60d",
            "tactic": _split_tactic(
                banks=2,
                k1_group=256,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=528,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "f1349db13dd317efe46d1a1271b652f652e4af36c86cdca4bf5e735372b49903",
            ),
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "31ddb01984acf3a18856e5e089af43974a36d8892399c50f6307f0f41f03b429",
            "tactic": _split_tactic(
                banks=2,
                k1_group=396,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=330,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "8a601b0816d82e10bb1d28baea0d83e67126f790da086c1bb748594fb9112fd8",
            ),
            "winner_for_tokens": (128,),
        },
        {
            "candidate_id": "5787644b0c7cf45f9c814a2eddb8ac951894db12d5335e0b4dcf854e245860e3",
            "tactic": _split_tactic(
                banks=1,
                k1_group=330,
                k1_tile=(256, 32, 128),
                k1_stages=2,
                k2_group=64,
                k2_tile=(256, 32, 256),
                k2_stages=2,
            ),
            "source": "split_k2_group",
            "parent_candidate_ids": (
                "136e40726a85019e72dce7279d4003f67fd15777559aa0ef006e00f5c552abb2",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "6a0b4fd0b67832b45edebf656d670f57e0cc973c8b905de819e1ae546aff1ecb",
            "tactic": _split_tactic(
                banks=1,
                k1_group=396,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=512,
                k2_tile=(256, 64, 256),
                k2_stages=2,
            ),
            "source": "split_k2_group",
            "parent_candidate_ids": (
                "f1dcd93a8a996ba2565b5d5a68db199bc52c86396db8688a5f5a4135a5a22988",
            ),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "7cd01fddb88a41bb1b5fe366cd775e8604424757b68df8e861cd2ef955ce7cce",
            "tactic": _split_tactic(
                banks=2,
                k1_group=396,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=264,
                k2_tile=(256, 32, 256),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "e4347ed5e2dca7f314dea1d7cb8ea0834c42e025a9778421773b9ef4fc0282ae",
            ),
            "winner_for_tokens": (512,),
        },
        {
            "candidate_id": "b16e1cfffbd2796e4b4b8583276c6b9996b1a0e6461a7108e287e421b2353852",
            "tactic": _split_tactic(
                banks=2,
                k1_group=128,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=3,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "6d4f9b98ad8e0310385f4ac835a0ce61514550ea9a3e8cb8329e300e2578b574",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "e97d5814f9a03f7144d058014a4ad78c744dde50fb05a96ab02b1db643a0f905",
            "tactic": _split_tactic(
                banks=1,
                k1_group=264,
                k1_tile=(256, 64, 128),
                k1_stages=2,
                k2_group=512,
                k2_tile=(256, 64, 256),
                k2_stages=2,
            ),
            "source": "split_k2_group",
            "parent_candidate_ids": (
                "33770ae15d6ce852aeab6887067d401556ee80b92d20fc281e3c5a8deb503d5f",
            ),
            "winner_for_tokens": (1024,),
        },
    ),
}

_WINNERS: dict[str, dict[int, dict[str, Any]]] = {
    "fused": {
        8: {
            "candidate_id": "104f2e89fd468598df2376c3d937728bfaa96b465676b5b8fafb53ee9a33878b",
            "median_score_us": 484.816,
            "relative_spread": 0.003564234678723503,
            "scores_us": (486.032009, 484.816, 484.304011),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "median_score_us": 832.672,
            "relative_spread": 0.0021136485915221573,
            "scores_us": (831.424028, 833.184004, 832.672),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "median_score_us": 928.240001,
            "relative_spread": 0.004188551447698365,
            "scores_us": (928.240001, 924.367994, 928.255975),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "c08058def16c3d732495cda3ad1ee2cd7244fb4d90469d1ca40cc3a196655563",
            "median_score_us": 972.608,
            "relative_spread": 0.006629598975126646,
            "scores_us": (979.039997, 972.608, 972.591996),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "median_score_us": 1051.679969,
            "relative_spread": 0.0028145691534037106,
            "scores_us": (1051.679969, 1054.240048, 1051.280022),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "b18dae894b086cd7f0d4d22fb4cfdd28682015ad372c1cb2b3985bad50a12f03",
            "median_score_us": 1422.320008,
            "relative_spread": 0.00046122391326152876,
            "scores_us": (1422.432005, 1421.775997, 1422.320008),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "331de4723cbac173f1786ad0974bd4233d4a996167e40cb4c559a0c772c4a63f",
            "median_score_us": 2228.751898,
            "relative_spread": 0.00035895112449166626,
            "scores_us": (2228.767991, 2227.967978, 2228.751898),
            "telemetry_warning": True,
        },
        2048: {
            "candidate_id": "48a466d13815abe8e6d47885098a80e9cdbd58e057a5c38b67b233f733fc5226",
            "median_score_us": 4079.504013,
            "relative_spread": 0.0032121458780876894,
            "scores_us": (4083.456039, 4079.504013, 4070.352077),
            "telemetry_warning": True,
        },
    },
    "split": {
        8: {
            "candidate_id": "b16e1cfffbd2796e4b4b8583276c6b9996b1a0e6461a7108e287e421b2353852",
            "median_score_us": 525.103986,
            "relative_spread": 0.0013407268251054785,
            "scores_us": (525.792003, 525.087982, 525.103986),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "2cd398370b552092dcaeeeb1ad79033cc1078b9cb234faa428c161708af3d60d",
            "median_score_us": 884.063989,
            "relative_spread": 0.004560773937371629,
            "scores_us": (883.872002, 887.904018, 884.063989),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "2a98249053aa02c8c814db675d107323b73a420a6b6cf7bb8a6cfba195db1721",
            "median_score_us": 982.944012,
            "relative_spread": 0.0009440873423826658,
            "scores_us": (982.944012, 983.024001, 982.096016),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "31ddb01984acf3a18856e5e089af43974a36d8892399c50f6307f0f41f03b429",
            "median_score_us": 1017.199993,
            "relative_spread": 0.0034290257805771832,
            "scores_us": (1017.199993, 1017.376006, 1013.888001),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "5787644b0c7cf45f9c814a2eddb8ac951894db12d5335e0b4dcf854e245860e3",
            "median_score_us": 1101.75997,
            "relative_spread": 0.001568398786534198,
            "scores_us": (1101.75997, 1101.056039, 1102.784038),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "7cd01fddb88a41bb1b5fe366cd775e8604424757b68df8e861cd2ef955ce7cce",
            "median_score_us": 1453.80801,
            "relative_spread": 0.0021900828569515465,
            "scores_us": (1451.312006, 1454.495966, 1453.80801),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "e97d5814f9a03f7144d058014a4ad78c744dde50fb05a96ab02b1db643a0f905",
            "median_score_us": 1961.488008,
            "relative_spread": 0.0016232370460661364,
            "scores_us": (1959.728003, 1961.488008, 1962.911963),
            "telemetry_warning": True,
        },
        2048: {
            "candidate_id": "6a0b4fd0b67832b45edebf656d670f57e0cc973c8b905de819e1ae546aff1ecb",
            "median_score_us": 3313.007951,
            "relative_spread": 0.00399395147723866,
            "scores_us": (3321.39194, 3313.007951, 3308.159947),
            "telemetry_warning": True,
        },
    },
}


# The published-exact sweep was intentionally frozen as a second, disjoint
# routing domain.  Its external artifacts are larger than the compact runtime
# tables below: in particular, the merged split schema-v2 manifest carries
# per-token shard lineage and telemetry records that are not embedded here.
_EXACT_PROVENANCE = {
    "fused": {
        "artifact_manifest_sha256": (
            "62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61"
        ),
        "domain_sha256": (
            "86b846926d510fb013132462cf88122e36479db5b374645cfa3951021becf1a6"
        ),
        "external_schema_version": 1,
        "policy_sha256": (
            "88503f1be444226eed1cf59d1083de5ffa92491d9bb5c626dcf7f96d22013137"
        ),
        "runtime_manifest_sha256": (
            "f4112c7d0d7ead640239c1df3d7f4af74e2a1fb35cf5e821edd1beba9bba3e99"
        ),
        "source_manifest_sha256": (
            "bcc0448df03348eb82addb5513152f7c7db42673769a40f9d67600d72c08a689"
        ),
        "workload_recipe_sha256": (
            "e53d38f2f6fc708fb93a6405ad2129b49a3ca5499e34083e1aa2c706c9eacbfd"
        ),
    },
    "split": {
        "artifact_manifest_sha256": (
            "094d840c579a7331439d1acd50690909ad2c88e6085253326c6f2d98ddad248a"
        ),
        "candidate_union_sha256": (
            "210adb840e66c0f44949ea866a8bbfa9f5b2b3835ee8992e57b5edb75f8b9321"
        ),
        "domain_sha256": (
            "86b846926d510fb013132462cf88122e36479db5b374645cfa3951021becf1a6"
        ),
        "external_schema_version": 2,
        "high_policy_sha256": (
            "1753d198bfaf7335a0998fe660642a3603b7448535f93435da2dcc2fd5d8f747"
        ),
        "low_policy_sha256": (
            "8069be3056f7a9235a1e67bcf3cd3da89fdec441e3735b8b6259f74895a8735d"
        ),
        "provenance_model": (
            "per-token source/policy/root bindings; no synthetic unified policy_sha256"
        ),
        "runtime_manifest_sha256": (
            "97a4a40bffeb062b9cc916959186e308bb8e3ff52150dab8a01b3273af22261f"
        ),
        "source_manifest_sha256": (
            "bcc0448df03348eb82addb5513152f7c7db42673769a40f9d67600d72c08a689"
        ),
        "workload_recipe_sha256": (
            "e53d38f2f6fc708fb93a6405ad2129b49a3ca5499e34083e1aa2c706c9eacbfd"
        ),
    },
}

_EXACT_CANDIDATES: dict[str, tuple[dict[str, Any], ...]] = {
    "fused": (
        {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "tactic": _fused_tactic(tile=(256, 32, 128), group_hint=396, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "8ef6a914cd598b8e1ed7753396d282020d907fc10a9ea9eaf46a56214eee73dd",
            ),
            "winner_for_tokens": (512,),
        },
        {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=512, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            ),
            "winner_for_tokens": (32, 64),
        },
        {
            "candidate_id": "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
            "tactic": _fused_tactic(
                tile=(128, 64, 256),
                group_hint=330,
                stages=2,
                token_back="reuse_dispatch_warps",
            ),
            "source": "fused_load",
            "parent_candidate_ids": (
                "f6cc04937e556078a2f039a414a4131da8beead85c37a13a84ea3b4769b2438d",
            ),
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
            "tactic": _fused_tactic(
                tile=(256, 16, 256),
                group_hint=396,
                stages=2,
                load_balance="static",
            ),
            "source": "fused_group",
            "parent_candidate_ids": (
                "08387b9c3c7e64639cf97a3fa92804fe5615850fa9d2105f733b97ad96321196",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=396, stages=1),
            "source": "fused_stages",
            "parent_candidate_ids": (
                "f3bded653245824c8cd8a1b67605608757b5df569da3dca409302e94e7baf7a4",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
            "tactic": _fused_tactic(
                tile=(128, 64, 256),
                cluster=(1, 1, 1),
                group_hint=528,
                stages=2,
                token_back="reuse_dispatch_warps",
            ),
            "source": "fused_load",
            "parent_candidate_ids": (
                "2b24fef280ba7168a0041f96990bfa93bee2570b31f3a0365a006b7f0f201813",
            ),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            "tactic": _fused_tactic(tile=(256, 16, 256), group_hint=512, stages=2),
            "source": "fused_group",
            "parent_candidate_ids": (
                "6764dee4fcd21556a13c148e2eca6bd30c642d3e7a44e9e5c0a1a66c8ea75f5f",
            ),
            "winner_for_tokens": (128,),
        },
    ),
    "split": (
        {
            "candidate_id": "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
            "tactic": _split_tactic(
                banks=2,
                k1_group=256,
                k1_tile=(256, 16, 256),
                k1_stages=3,
                k2_group=330,
                k2_tile=(256, 16, 256),
                k2_stages=3,
                k1_sm_count=88,
                k2_sm_count=44,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "4efd035fd9bbfacbef6787ccd8b311530b4e21245ff5f433721f94eb08234d47",
            ),
            "winner_for_tokens": (8,),
        },
        {
            "candidate_id": "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
            "tactic": _split_tactic(
                banks=2,
                k1_group=396,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "72e080ec57447dc33c70154a81e85105e13c1cb11760bfc2f5528653e97ff005",
            ),
            "winner_for_tokens": (256,),
        },
        {
            "candidate_id": "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
            "tactic": _split_tactic(
                banks=2,
                k1_group=528,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=528,
                k2_tile=(256, 64, 256),
                k2_stages=2,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "462d763bc23efc9777611f99a1b87d327aa8958e4e71bd406756ea315a9bd7f4",
            ),
            "winner_for_tokens": (2048,),
        },
        {
            "candidate_id": "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
            "tactic": _split_tactic(
                banks=2,
                k1_group=512,
                k1_tile=(256, 16, 128),
                k1_stages=1,
                k2_group=512,
                k2_tile=(256, 16, 128),
                k2_stages=1,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "dd27c6d35a943f035151c28964193336247bec4d576b16303cab9a937ac3d5c9",
            ),
            "winner_for_tokens": (128,),
        },
        {
            "candidate_id": "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
            "tactic": _split_tactic(
                banks=1,
                k1_group=512,
                k1_tile=(256, 64, 256),
                k1_stages=2,
                k2_group=528,
                k2_tile=(256, 64, 256),
                k2_stages=1,
            ),
            "source": "split_stages",
            "parent_candidate_ids": (
                "686c6f58c90e8ec5124794157c3e121f04a84cda560ff893dab0c09cd918fa8d",
            ),
            "winner_for_tokens": (1024,),
        },
        {
            "candidate_id": "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
            "tactic": _split_tactic(
                banks=1,
                k1_group=128,
                k1_tile=(256, 16, 128),
                k1_stages=2,
                k2_group=132,
                k2_tile=(256, 32, 256),
                k2_stages=2,
            ),
            "source": "split_k1_group",
            "parent_candidate_ids": (
                "7695f9dd75e66a74aaf91ed75224689b163dffcba43797321f66cced78e39871",
            ),
            "winner_for_tokens": (64,),
        },
        {
            "candidate_id": "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
            "tactic": _split_tactic(
                banks=2,
                k1_group=256,
                k1_tile=(256, 16, 128),
                k1_stages=2,
                k2_group=132,
                k2_tile=(256, 32, 256),
                k2_stages=2,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "5c2ec0a2c6828bfc0581a2b51fdc7774c019a4d3d63b00e7c2cf00ea7c09e5a9",
            ),
            "winner_for_tokens": (32,),
        },
        {
            "candidate_id": "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
            "tactic": _split_tactic(
                banks=2,
                k1_group=512,
                k1_tile=(256, 32, 128),
                k1_stages=2,
                k2_group=330,
                k2_tile=(256, 32, 256),
                k2_stages=3,
            ),
            "source": "split_graph_bank",
            "parent_candidate_ids": (
                "8533cb0e0b3ccfc144a498ab1c3a51031c00baa61b12ddbcd3026684febe9f6c",
            ),
            "winner_for_tokens": (512,),
        },
    ),
}

_EXACT_WINNERS: dict[str, dict[int, dict[str, Any]]] = {
    "fused": {
        8: {
            "candidate_id": "8f0b7bffc6d79127d296b90bafdd11c43509a156b304ef70f75d35352ec59676",
            "median_score_us": 519.807994,
            "relative_spread": 0.0015082684549864128,
            "scores_us": (520.352006, 519.807994, 519.567996),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "median_score_us": 937.472016,
            "relative_spread": 0.005359110367300778,
            "scores_us": (937.472016, 934.816003, 939.840019),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "489f2dde9c54076b5d1b8f040f7ad67416066cc379c339509d6b9e6746a398c9",
            "median_score_us": 948.416024,
            "relative_spread": 0.003525878849976127,
            "scores_us": (948.336005, 948.416024, 951.680005),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "de1e74bcd73b339753d2ad90ad2165b93d835ee50712fe87d7cdd610ea05b6d4",
            "median_score_us": 963.279992,
            "relative_spread": 0.006228731054137874,
            "scores_us": (963.279992, 963.904023, 957.904011),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "7e8b06c53cb13fb0cc356f05f927472f74d2ad1cf71ec3858b3f04088f538cd7",
            "median_score_us": 994.527996,
            "relative_spread": 0.0006917713757350173,
            "scores_us": (994.800001, 994.527996, 994.112015),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "1ab53d2740841966553b91b615f8e600b59afb1b9f4d5ab7c4ba5f6261daba16",
            "median_score_us": 1097.21601,
            "relative_spread": 0.002697760489295057,
            "scores_us": (1096.127987, 1097.21601, 1099.088013),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "79ed474eb1b459fd6e934edfc0fcddb81470ee36a7f356a2b291c63aa0e36e02",
            "median_score_us": 1783.792019,
            "relative_spread": 0.0023948890646987332,
            "scores_us": (1783.13601, 1783.792019, 1787.407994),
            "telemetry_warning": False,
        },
        2048: {
            "candidate_id": "d606939892f020b7e7527235737cb37adbbad233da8fd6a314c09876104ad114",
            "median_score_us": 3483.951926,
            "relative_spread": 0.001786487624456375,
            "scores_us": (3483.951926, 3488.576055, 3482.352018),
            "telemetry_warning": True,
        },
    },
    "split": {
        8: {
            "candidate_id": "302ce3733b64947c20ec514e6107d2c0b7e6f305b078d5fe3b9a12421a276aee",
            "median_score_us": 544.384003,
            "relative_spread": 0.0032036246296532397,
            "scores_us": (545.231998, 543.487996, 544.384003),
            "telemetry_warning": False,
        },
        32: {
            "candidate_id": "c052d631fb1e1c9b0a2a3890b789d9f48007955530aa38366e87860f25028e94",
            "median_score_us": 972.81599,
            "relative_spread": 0.009325479939942136,
            "scores_us": (972.81599, 970.080018, 979.151994),
            "telemetry_warning": False,
        },
        64: {
            "candidate_id": "9e7fd2c153bdcc2cf8477b913431e39be2040a6d49f31fffddcfafd94af0494f",
            "median_score_us": 978.991985,
            "relative_spread": 0.00400413799097653,
            "scores_us": (978.991985, 981.263995, 977.343976),
            "telemetry_warning": False,
        },
        128: {
            "candidate_id": "845359c757ce881e2d98077f209c02721dc95b66771dc7149b5f1e900631f355",
            "median_score_us": 985.95202,
            "relative_spread": 0.00149296920148304,
            "scores_us": (985.359997, 985.95202, 986.831993),
            "telemetry_warning": False,
        },
        256: {
            "candidate_id": "43ab071b691a2153a14e9173f5a99b4c167ad49f4e4160f25de5385cd4f5b634",
            "median_score_us": 1019.151986,
            "relative_spread": 0.003673648338453037,
            "scores_us": (1017.215967, 1020.959973, 1019.151986),
            "telemetry_warning": False,
        },
        512: {
            "candidate_id": "f6ad358a9b115bcaaf2c6222ec782ccfc98a8c6a9d393a78ed187713c621795a",
            "median_score_us": 1144.97602,
            "relative_spread": 0.0018725099587674724,
            "scores_us": (1144.97602, 1145.26397, 1143.119991),
            "telemetry_warning": False,
        },
        1024: {
            "candidate_id": "9a16b958e9f3acb42df1ba4ae66175a3a5021c0d05fffa606d9a113d403b6b1b",
            "median_score_us": 1569.82404,
            "relative_spread": 0.0009376821621358214,
            "scores_us": (1568.751991, 1570.223987, 1569.82404),
            "telemetry_warning": False,
        },
        2048: {
            "candidate_id": "47254cc9fcd2fae00d7ee5236e1b7e19fbe7c0b14da2b2ca5f6572af75f49a0c",
            "median_score_us": 2875.983953,
            "relative_spread": 0.0028539922802552553,
            "scores_us": (2874.992013, 2875.983953, 2883.200049),
            "telemetry_warning": True,
        },
    },
}

_PROFILE_PROVENANCE = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _PROVENANCE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_PROVENANCE,
}
_PROFILE_CANDIDATES = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _CANDIDATES,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_CANDIDATES,
}
_PROFILE_WINNERS = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: _WINNERS,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: _EXACT_WINNERS,
}

MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE = MappingProxyType(
    {
        profile: MappingProxyType(
            {mode: MappingProxyType(values) for mode, values in per_mode.items()}
        )
        for profile, per_mode in _PROFILE_PROVENANCE.items()
    }
)

# Filled from the canonical JSON serialization of the compact runtime
# projections returned by ``hopper_mxfp4_tuning_manifest``.  These hashes are
# independent of the larger external artifact hashes above.
_EXACT_RUNTIME_MANIFEST_SHA256 = {
    mode: values["runtime_manifest_sha256"]
    for mode, values in _EXACT_PROVENANCE.items()
}


def _mode(execution_mode: str) -> Mxfp4ExecutionMode:
    if execution_mode not in _IMPLEMENTATION:
        raise ValueError(
            f"execution_mode must be exactly 'fused' or 'split', got {execution_mode!r}"
        )
    return execution_mode  # type: ignore[return-value]


def normalize_hopper_mxfp4_routing_profile(
    routing_profile: str,
) -> Mxfp4RoutingProfile:
    """Compatibility wrapper around the canonical SM90 profile normalizer."""

    return normalize_sm90_routing_profile(routing_profile)  # type: ignore[return-value]


def hopper_mxfp4_tuning_provenance(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> Mapping[str, Any]:
    """Return immutable provenance for one execution/routing domain."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE[profile][mode]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-bool integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an M,N,K triple")
    return tuple(_positive_int(item, name) for item in value)  # type: ignore[return-value]


def _require_exact_fields(
    tactic: Mapping[str, Any], expected: frozenset[str], implementation: str
) -> None:
    actual = set(tactic)
    if actual != expected:
        raise ValueError(
            f"{implementation} tactic fields differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def validate_hopper_mxfp4_tactic(
    tactic: Mapping[str, Any], *, execution_mode: str, total_sms: int = 132
) -> dict[str, Any]:
    """Validate and normalize one tactic from the frozen H200 search domain.

    The returned mapping is a fresh copy whose tile and cluster fields are
    tuples.  No defaults are inserted: callers must preserve the complete
    fused or split tactic identity.
    """

    mode = _mode(execution_mode)
    if not isinstance(tactic, Mapping):
        raise TypeError(f"tactic must be a mapping, got {type(tactic).__name__}")

    if mode == "fused":
        _require_exact_fields(tactic, _FUSED_FIELDS, _IMPLEMENTATION[mode])
        if tactic["swap_ab"] is not True:
            raise ValueError("MXFP4 fused requires swap_ab=true")
        if not isinstance(tactic["pingpong"], bool):
            raise ValueError("MXFP4 fused pingpong must be bool")
        tile = _triple(tactic["mma_tiler_mnk"], "mma_tiler_mnk")
        m, n, k = tile
        if (
            m not in (128, 256)
            or n not in (16, 32, 64, 128)
            or k
            not in (
                128,
                256,
            )
        ):
            raise ValueError("illegal MXFP4 fused tile")
        if tactic["pingpong"] and m != 128:
            raise ValueError("MXFP4 fused pingpong requires M128")
        cluster = _triple(tactic["cluster_shape_mnk"], "cluster_shape_mnk")
        if cluster not in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)):
            raise ValueError("illegal MXFP4 fused cluster")
        if tactic["fp8_accum_mode"] != "1xacc":
            raise ValueError("MXFP4 fused fixes 1xacc")
        if tactic["load_balance_mode"] not in ("static", "atomic_counter"):
            raise ValueError("illegal MXFP4 fused load balance")
        if tactic["token_back_mode"] not in (
            "epi_warps",
            "reuse_dispatch_warps",
            "standalone_warps",
        ):
            raise ValueError("illegal MXFP4 fused token-back")
        group_hint = _optional_positive_int(tactic["group_hint"], "group_hint")
        stages = _optional_positive_int(tactic["num_sched_stages"], "num_sched_stages")
        if tactic["in_kernel_fc2_reduce"] is not False:
            raise ValueError("MXFP4 fused fixes in-kernel reduce false")
        return {
            "cluster_shape_mnk": cluster,
            "fp8_accum_mode": "1xacc",
            "group_hint": group_hint,
            "in_kernel_fc2_reduce": False,
            "load_balance_mode": tactic["load_balance_mode"],
            "mma_tiler_mnk": tile,
            "num_sched_stages": stages,
            "pingpong": tactic["pingpong"],
            "swap_ab": True,
            "token_back_mode": tactic["token_back_mode"],
        }

    _require_exact_fields(tactic, _SPLIT_FIELDS, _IMPLEMENTATION[mode])
    k1 = _triple(tactic["k1_mma_tiler_mnk"], "k1_mma_tiler_mnk")
    k2 = _triple(tactic["k2_mma_tiler_mnk"], "k2_mma_tiler_mnk")
    for role, tile in (("K1", k1), ("K2", k2)):
        if (
            tile[0] not in (128, 256)
            or tile[1] not in (16, 32, 64, 128)
            or tile[2] not in (128, 256)
        ):
            raise ValueError(f"illegal split {role} tile")
    c1 = _triple(tactic["k1_cluster_shape_mnk"], "k1_cluster_shape_mnk")
    c2 = _triple(tactic["k2_cluster_shape_mnk"], "k2_cluster_shape_mnk")
    if c1 != c2:
        raise ValueError("split K1/K2 clusters must match")
    if c1 != (1, 1, 1):
        raise ValueError(
            "non-1x1x1 split clusters are quarantined correctness failures"
        )
    k1_group = _optional_positive_int(tactic["k1_group_hint"], "k1_group_hint")
    k2_group = _optional_positive_int(tactic["k2_group_hint"], "k2_group_hint")
    k1_stages = _optional_positive_int(
        tactic["k1_num_sched_stages"], "k1_num_sched_stages"
    )
    k2_stages = _optional_positive_int(
        tactic["k2_num_sched_stages"], "k2_num_sched_stages"
    )
    k1_sms = _positive_int(tactic["k1_sm_count"], "k1_sm_count")
    k2_sms = _positive_int(tactic["k2_sm_count"], "k2_sm_count")
    expected_sms = _positive_int(total_sms, "total_sms")
    if k1_sms + k2_sms != expected_sms or k1_sms % 8:
        raise ValueError(
            f"split partition must sum to {expected_sms} SMs; "
            "only K1 is alignment-constrained"
        )
    banks = tactic["counter_epoch_banks"]
    if isinstance(banks, bool) or banks not in (1, 2):
        raise ValueError("split counter banks must be 1 or 2")
    graph = tactic["graph_variant"]
    if graph not in ("cold_k0", "steady_k3_reset"):
        raise ValueError("illegal split graph variant")
    if banks == 2 and graph != "steady_k3_reset":
        raise ValueError("two split counter banks require steady_k3_reset")
    if tactic["enable_iket"] is not False:
        raise ValueError("first formal split domain fixes IKET false")
    return {
        "counter_epoch_banks": banks,
        "enable_iket": False,
        "graph_variant": graph,
        "k1_cluster_shape_mnk": c1,
        "k1_group_hint": k1_group,
        "k1_mma_tiler_mnk": k1,
        "k1_num_sched_stages": k1_stages,
        "k1_sm_count": k1_sms,
        "k2_cluster_shape_mnk": c2,
        "k2_group_hint": k2_group,
        "k2_mma_tiler_mnk": k2,
        "k2_num_sched_stages": k2_stages,
        "k2_sm_count": k2_sms,
    }


def is_valid_hopper_mxfp4_tactic(
    tactic: Mapping[str, Any], *, execution_mode: str
) -> bool:
    """Return whether ``tactic`` is a complete legal fused/split tactic."""

    try:
        validate_hopper_mxfp4_tactic(tactic, execution_mode=execution_mode)
    except (TypeError, ValueError):
        return False
    return True


def _candidate_id(implementation: str, tactic: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"implementation": implementation, "tactic": tactic},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _candidate_by_id(
    mode: Mxfp4ExecutionMode, routing_profile: Mxfp4RoutingProfile
) -> dict[str, dict[str, Any]]:
    return {
        record["candidate_id"]: record
        for record in _PROFILE_CANDIDATES[routing_profile][mode]
    }


def _manifest_candidate(
    mode: Mxfp4ExecutionMode, record: Mapping[str, Any]
) -> dict[str, Any]:
    tactic = validate_hopper_mxfp4_tactic(record["tactic"], execution_mode=mode)
    return {
        "candidate_id": record["candidate_id"],
        "effective_tactic": copy.deepcopy(tactic),
        "implementation": _IMPLEMENTATION[mode],
        "parent_candidate_ids": list(record["parent_candidate_ids"]),
        "requested_tactic": copy.deepcopy(tactic),
        "source": record["source"],
    }


def hopper_mxfp4_candidate_records(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the deduplicated union with offline discovery provenance."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        {
            "candidate": _manifest_candidate(mode, record),
            "winner_for_tokens": list(record["winner_for_tokens"]),
        }
        for record in _PROFILE_CANDIDATES[profile][mode]
    ]


def hopper_mxfp4_candidates(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return fresh tactic dictionaries for bounded online autotuning."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    return [
        validate_hopper_mxfp4_tactic(record["tactic"], execution_mode=mode)
        for record in _PROFILE_CANDIDATES[profile][mode]
    ]


def is_hopper_mxfp4_tactic_shape_compatible(
    tactic: Mapping[str, Any],
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
) -> bool:
    """Whether a validated tactic's GEMM K axes divide the model shape."""

    mode = _mode(execution_mode)
    _positive_int(hidden, "hidden")
    _positive_int(intermediate, "intermediate")
    normalized = validate_hopper_mxfp4_tactic(tactic, execution_mode=mode)
    if mode == "fused":
        tile_k = int(normalized["mma_tiler_mnk"][2])
        return hidden % tile_k == 0 and intermediate % tile_k == 0
    k1_tile_k = int(normalized["k1_mma_tiler_mnk"][2])
    k2_tile_k = int(normalized["k2_mma_tiler_mnk"][2])
    return hidden % k1_tile_k == 0 and intermediate % k2_tile_k == 0


def hopper_mxfp4_candidates_for_shape(
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Return the stable manifest union filtered to tactics legal for a shape."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in hopper_mxfp4_candidates(
            execution_mode=mode, routing_profile=profile
        )
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic,
            execution_mode=mode,
            hidden=hidden,
            intermediate=intermediate,
        )
    ]
    if not legal:
        raise ValueError(
            f"no manifest-derived MXFP4 {mode} tactic supports "
            f"hidden={hidden}, intermediate={intermediate}"
        )
    return legal


def _token_bucket(max_tokens: int) -> int:
    _positive_int(max_tokens, "max_tokens")
    for bucket in MXFP4_TUNING_TOKEN_BUCKETS:
        if max_tokens <= bucket:
            return bucket
    return MXFP4_TUNING_TOKEN_BUCKETS[-1]


def hopper_mxfp4_default_tactic(
    max_tokens: int,
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the ceil-bucket offline winner, clamping above 2048 tokens."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    bucket = _token_bucket(max_tokens)
    winner_id = _PROFILE_WINNERS[profile][mode][bucket]["candidate_id"]
    record = _candidate_by_id(mode, profile)[winner_id]
    return validate_hopper_mxfp4_tactic(record["tactic"], execution_mode=mode)


def hopper_mxfp4_ordered_candidates(
    max_tokens: int,
    *,
    execution_mode: str,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Put a legal bucket winner first, otherwise keep stable union order."""

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = hopper_mxfp4_candidates_for_shape(
        execution_mode=mode,
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=profile,
    )
    default = hopper_mxfp4_default_tactic(
        max_tokens, execution_mode=mode, routing_profile=profile
    )
    if default not in legal:
        return legal
    return [default, *(candidate for candidate in legal if candidate != default)]


def hopper_mxfp4_tuning_manifest(
    *,
    execution_mode: str,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return the frozen manifest representation for one routing profile.

    The legacy profile reconstructs its schema-v1 artifact byte-for-byte.  The
    published-exact profile returns a compact runtime projection and records
    the external artifact SHA.  The external merged split schema-v2 artifact
    additionally contains per-token shard provenance and telemetry that are
    deliberately not embedded in this data-only runtime module.
    """

    mode = _mode(execution_mode)
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    provenance = _PROFILE_PROVENANCE[profile][mode]
    records = _candidate_by_id(mode, profile)
    per_token_winners: dict[str, dict[str, Any]] = {}
    for token in MXFP4_TUNING_TOKEN_BUCKETS:
        winner = _PROFILE_WINNERS[profile][mode][token]
        candidate_id = winner["candidate_id"]
        per_token_winners[str(token)] = {
            "candidate": _manifest_candidate(mode, records[candidate_id]),
            "candidate_id": candidate_id,
            "median_score_us": winner["median_score_us"],
            "relative_spread": winner["relative_spread"],
            "scores_us": list(winner["scores_us"]),
            "telemetry_warning": winner["telemetry_warning"],
        }
    candidate_union = hopper_mxfp4_candidate_records(
        execution_mode=mode, routing_profile=profile
    )
    if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
        return {
            "candidate_union": candidate_union,
            "domain_sha256": provenance["domain_sha256"],
            "implementation": _IMPLEMENTATION[mode],
            "per_token_winners": per_token_winners,
            "policy_sha256": provenance["policy_sha256"],
            "schema_version": provenance["schema_version"],
            "source_manifest_sha256": provenance["source_manifest_sha256"],
            "workload_recipe_sha256": provenance["workload_recipe_sha256"],
        }
    return {
        "artifact_manifest_sha256": provenance["artifact_manifest_sha256"],
        "candidate_union": candidate_union,
        "external_schema_version": provenance["external_schema_version"],
        "implementation": _IMPLEMENTATION[mode],
        "per_token_winners": per_token_winners,
        "routing_profile": profile,
        "runtime_schema_version": 1,
    }


def _validate_embedded_tables() -> None:
    for profile in MXFP4_TUNING_ROUTING_PROFILES:
        for mode in ("fused", "split"):
            candidate_ids = []
            winning_tokens = []
            for record in _PROFILE_CANDIDATES[profile][mode]:
                tactic = validate_hopper_mxfp4_tactic(
                    record["tactic"], execution_mode=mode
                )
                expected_id = _candidate_id(_IMPLEMENTATION[mode], tactic)
                if record["candidate_id"] != expected_id:
                    raise RuntimeError(
                        f"corrupt {profile}/{mode} MXFP4 candidate id "
                        f"{record['candidate_id']}"
                    )
                candidate_ids.append(record["candidate_id"])
                winning_tokens.extend(record["winner_for_tokens"])
            if candidate_ids != sorted(set(candidate_ids)):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 candidate union is not sorted/unique"
                )
            if sorted(winning_tokens) != list(MXFP4_TUNING_TOKEN_BUCKETS):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 winners do not cover every token bucket"
                )
            if set(_PROFILE_WINNERS[profile][mode]) != set(MXFP4_TUNING_TOKEN_BUCKETS):
                raise RuntimeError(
                    f"{profile}/{mode} MXFP4 heuristic table has wrong token buckets"
                )
            manifest = hopper_mxfp4_tuning_manifest(
                execution_mode=mode, routing_profile=profile
            )
            raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
            actual_sha256 = hashlib.sha256(raw).hexdigest()
            if profile == MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE:
                expected_sha256 = _PROVENANCE[mode]["manifest_sha256"]
            else:
                expected_sha256 = _EXACT_RUNTIME_MANIFEST_SHA256[mode]
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    f"embedded {profile}/{mode} MXFP4 runtime manifest "
                    f"sha256={actual_sha256} does not match {expected_sha256}"
                )


_validate_embedded_tables()


__all__ = [
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_PROVENANCE",
    "MXFP4_TUNING_PROVENANCE_BY_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "Mxfp4ExecutionMode",
    "Mxfp4RoutingProfile",
    "hopper_mxfp4_candidate_records",
    "hopper_mxfp4_candidates",
    "hopper_mxfp4_candidates_for_shape",
    "hopper_mxfp4_default_tactic",
    "hopper_mxfp4_ordered_candidates",
    "hopper_mxfp4_tuning_manifest",
    "hopper_mxfp4_tuning_provenance",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "require_hopper_mxfp4_tuning_device",
    "validate_hopper_mxfp4_tactic",
]
