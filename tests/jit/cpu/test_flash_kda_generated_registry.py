# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checks of the generated FlashKDA module table and its registry."""

import pytest

import json
import re

from flashinfer.jit import flash_kda

# One table row per generated module; a row lists every exact architecture that
# compiles its body, and the registry holds one entry per (architecture, module)
# pair. Rows of a folded body share the body file and differ in their knobs.
EXPECTED_ROWS = 59
EXPECTED_SHARED_ROWS = 45
EXPECTED_SINGLE_ARCH_ROWS = {"sm_100a": 2, "sm_103a": 12}
EXPECTED_REGISTRY_ENTRIES = 104
EXPECTED_TARGET_ENTRIES = {"sm100a": 47, "sm103a": 57}
ROW_FIELDS = (
    "module_ident",
    "body",
    "kernel",
    "threads",
    "smem_bytes",
    "use_pdl",
    "state_mode",
    "abi_family",
    "abi_variant",
    "tma_tile_tokens",
    "value_rows",
    "value_tma_rank",
    "pair_packed_beta",
    "affine_dependency",
)
# Folded bodies: one source serves the rows below. The loader passes each
# row's ``knobs`` to NVRTC as ``-DFLASHKDA_GENERATED_<FIELD>=<int>``; the
# body selects the row's code from them.
FOLDED_BODIES = {
    "flashkda_generated_bf16_fused_m64_dvsplit.cu": {
        "flashkda_bf16_fused_m64_2eb7864cd4": {
            "FLASHKDA_GENERATED_NUM_HEADS": 64,
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 0,
        },
        "flashkda_bf16_fused_m64_5eb5b5da39": {
            "FLASHKDA_GENERATED_NUM_HEADS": 6,
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 1,
        },
        "flashkda_bf16_fused_m64_9f7512076a": {
            "FLASHKDA_GENERATED_NUM_HEADS": 64,
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 1,
        },
        "flashkda_bf16_fused_m64_c2b2a382bb": {
            "FLASHKDA_GENERATED_NUM_HEADS": 6,
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 0,
        },
    },
    "flashkda_generated_bf16_fused_m128_scalar_lpt_bf16_state.cu": {
        "flashkda_bf16_fused_m128_79d0e71824": {"FLASHKDA_GENERATED_NUM_HEADS": 64},
        "flashkda_bf16_fused_m128_7c0223cea8": {"FLASHKDA_GENERATED_NUM_HEADS": 96},
    },
    "flashkda_generated_bf16_fused_m128_scalar_lpt_fp32_state.cu": {
        "flashkda_bf16_fused_m128_5ae1699f79": {"FLASHKDA_GENERATED_NUM_HEADS": 96},
        "flashkda_bf16_fused_m128_c8af151b6d": {"FLASHKDA_GENERATED_NUM_HEADS": 64},
    },
    "flashkda_generated_bf16_persistent_m128_piece_tasks.cu": {
        "flashkda_bf16_persistent_m128_47722a5d98": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 0,
        },
        "flashkda_bf16_persistent_m128_ddb5283bd6": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 1,
        },
    },
    "flashkda_generated_bf16_persistent_m128_head_grouped_lpt.cu": {
        "flashkda_bf16_persistent_m128_4b0769bf08": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 0,
        },
        "flashkda_bf16_persistent_m128_c258a6e650": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 1,
        },
    },
    "flashkda_generated_bf16_fused_m128_vtile_persistent_bf16_state.cu": {
        "flashkda_bf16_fused_m128_vtile_603e2d906d": {
            "FLASHKDA_GENERATED_NUM_HEADS": 64,
            "FLASHKDA_GENERATED_PERSISTENT_SIX_TASK_SCHEDULE": 0,
            "FLASHKDA_GENERATED_PERSISTENT_STRIDE_HEAD_ALIGNED": 1,
        },
        "flashkda_bf16_fused_m128_vtile_b784466ef6": {
            "FLASHKDA_GENERATED_NUM_HEADS": 96,
            "FLASHKDA_GENERATED_PERSISTENT_SIX_TASK_SCHEDULE": 1,
            "FLASHKDA_GENERATED_PERSISTENT_STRIDE_HEAD_ALIGNED": 0,
        },
    },
    "flashkda_generated_bf16_fused_m128_vtile_persistent_fp32_state.cu": {
        "flashkda_bf16_fused_m128_vtile_11bf731116": {
            "FLASHKDA_GENERATED_NUM_HEADS": 96,
            "FLASHKDA_GENERATED_PERSISTENT_SIX_TASK_SCHEDULE": 1,
            "FLASHKDA_GENERATED_PERSISTENT_STRIDE_HEAD_ALIGNED": 0,
        },
        "flashkda_bf16_fused_m128_vtile_a8f9b5a463": {
            "FLASHKDA_GENERATED_NUM_HEADS": 64,
            "FLASHKDA_GENERATED_PERSISTENT_SIX_TASK_SCHEDULE": 0,
            "FLASHKDA_GENERATED_PERSISTENT_STRIDE_HEAD_ALIGNED": 1,
        },
    },
    "flashkda_generated_bf16_bt16_chain_m64_s8.cu": {
        "flashkda_bf16_bt16_chain_m64_c68ffebac9": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 0,
            "FLASHKDA_GENERATED_SERVING_NATIVE_ABI": 0,
        },
        "flashkda_bf16_bt16_chain_m64_d3d65ffa34": {
            "FLASHKDA_GENERATED_STATE_DTYPE_IS_FP32": 1,
            "FLASHKDA_GENERATED_SERVING_NATIVE_ABI": 1,
        },
    },
    "flashkda_generated_bf16_fused_m128_affine_main_correction_fp32.cu": {
        "flashkda_bf16_fused_m128_4d394413ab": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_PDL_WAIT_INITIAL_STATE_F32": 1,
            "FLASHKDA_GENERATED_PDL_PUBLISH_FINAL_STATE": 0,
            "FLASHKDA_GENERATED_AFFINE_MAIN_INDEXED_INITIAL": 0,
        },
        "flashkda_bf16_fused_m128_7feb233c12": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 1,
            "FLASHKDA_GENERATED_SCALAR_BETA": 1,
            "FLASHKDA_GENERATED_PDL_WAIT_INITIAL_STATE_F32": 0,
            "FLASHKDA_GENERATED_PDL_PUBLISH_FINAL_STATE": 1,
            "FLASHKDA_GENERATED_AFFINE_MAIN_INDEXED_INITIAL": 1,
        },
        "flashkda_bf16_fused_m128_8838afb6eb": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 1,
            "FLASHKDA_GENERATED_SCALAR_BETA": 1,
            "FLASHKDA_GENERATED_PDL_WAIT_INITIAL_STATE_F32": 1,
            "FLASHKDA_GENERATED_PDL_PUBLISH_FINAL_STATE": 0,
            "FLASHKDA_GENERATED_AFFINE_MAIN_INDEXED_INITIAL": 0,
        },
        "flashkda_bf16_fused_m128_fca13c24e7": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_PDL_WAIT_INITIAL_STATE_F32": 0,
            "FLASHKDA_GENERATED_PDL_PUBLISH_FINAL_STATE": 1,
            "FLASHKDA_GENERATED_AFFINE_MAIN_INDEXED_INITIAL": 1,
        },
    },
    "flashkda_generated_bf16_fused_m128_direct_serving_bf16_state.cu": {
        "flashkda_bf16_fused_m128_97e6762e88": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 0,
        },
        "flashkda_bf16_fused_m128_b21b675208": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 1,
        },
        "flashkda_bf16_fused_m128_ea99273d0c": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 1,
            "FLASHKDA_GENERATED_SCALAR_BETA": 1,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 0,
        },
    },
    "flashkda_generated_bf16_fused_m128_direct_serving_fp32_state.cu": {
        "flashkda_bf16_fused_m128_388c6ad8eb": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 1,
        },
        "flashkda_bf16_fused_m128_7bb2a02766": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 0,
            "FLASHKDA_GENERATED_SCALAR_BETA": 0,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 0,
        },
        "flashkda_bf16_fused_m128_de729ccf63": {
            "FLASHKDA_GENERATED_PAIR_PACKED_BETA": 1,
            "FLASHKDA_GENERATED_SCALAR_BETA": 1,
            "FLASHKDA_GENERATED_N32_PREDICTION_FIRST": 0,
        },
    },
}
KNOB_RE = re.compile(r"\bFLASHKDA_GENERATED_[A-Z0-9_]+\b")


def _table_rows():
    path = (
        flash_kda._get_flash_kda_csrc_dir()
        / flash_kda._FLASH_KDA_GENERATED_METADATA_NAME
    )
    return json.loads(path.read_text())["modules"]


@pytest.mark.cpu_only
def test_generated_module_table_lists_each_module_once_with_its_archs():
    rows = _table_rows()
    assert len(rows) == EXPECTED_ROWS
    assert len({row["module_ident"] for row in rows}) == EXPECTED_ROWS
    shared = [row for row in rows if row["archs"] == ["sm_100a", "sm_103a"]]
    assert len(shared) == EXPECTED_SHARED_ROWS
    for arch, expected in EXPECTED_SINGLE_ARCH_ROWS.items():
        assert len([row for row in rows if row["archs"] == [arch]]) == expected
    assert sum(len(row["archs"]) for row in rows) == EXPECTED_REGISTRY_ENTRIES


@pytest.mark.cpu_only
def test_generated_registry_expands_each_table_row_per_arch():
    rows = _table_rows()
    registry = flash_kda.get_flash_kda_generated_registry()
    assert list(registry) == [
        f"{arch}:{row['module_ident']}" for row in rows for arch in row["archs"]
    ]
    assert len(registry) == EXPECTED_REGISTRY_ENTRIES
    for row in rows:
        expected_selectors = [
            (
                selector["route"],
                selector["route_role"],
                list(selector["specialization"].items()),
            )
            for selector in row["selectors"]
        ]
        for arch in row["archs"]:
            module = registry[f"{arch}:{row['module_ident']}"]
            assert module.arch == arch
            assert module.target == flash_kda._FLASH_KDA_GENERATED_ARCH_TARGETS[arch]
            for name in ROW_FIELDS:
                assert getattr(module, name) == row[name]
            assert [
                (selector.route, selector.route_role, list(selector.specialization))
                for selector in module.selectors
            ] == expected_selectors
    for target, expected in EXPECTED_TARGET_ENTRIES.items():
        variant_ids = flash_kda.get_flash_kda_generated_variant_ids(target)
        assert len(variant_ids) == expected
        assert all(registry[variant_id].target == target for variant_id in variant_ids)


@pytest.mark.cpu_only
def test_folded_generated_bodies_select_each_row_by_its_knobs():
    rows = {row["module_ident"]: row for row in _table_rows()}
    registry = flash_kda.get_flash_kda_generated_registry()
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    folded = set()
    for body, modules in FOLDED_BODIES.items():
        sharing = [row for row in rows.values() if row["body"] == body]
        assert {row["module_ident"] for row in sharing} == set(modules)
        # the knobs are exactly the fields that distinguish the rows sharing the body
        specializations = [row["selectors"][0]["specialization"] for row in sharing]
        varying = {
            name
            for name in specializations[0]
            if len({repr(s[name]) for s in specializations}) > 1
        }
        assert all(set(row["knobs"]) == varying for row in sharing)
        # every FLASHKDA_GENERATED_* macro the body uses is supplied by every row
        body_macros = set(KNOB_RE.findall((csrc_dir / body).read_text()))
        for module_ident, defines in modules.items():
            row = rows[module_ident]
            specialization = row["selectors"][0]["specialization"]
            assert list(defines) == [
                f"FLASHKDA_GENERATED_{knob.upper()}" for knob in row["knobs"]
            ]
            assert list(defines.values()) == [
                int(specialization[knob]) for knob in row["knobs"]
            ]
            assert body_macros <= set(defines)
            for arch in row["archs"]:
                module = registry[f"{arch}:{module_ident}"]
                assert module.body == body
                assert module.defines == tuple(defines.items())
            folded.add(module_ident)
    for module_ident, row in rows.items():
        if module_ident not in folded:
            assert "knobs" not in row
            for arch in row["archs"]:
                assert registry[f"{arch}:{module_ident}"].defines == ()


@pytest.mark.cpu_only
def test_every_generated_body_file_is_referenced_by_the_table():
    csrc_dir = flash_kda._get_flash_kda_csrc_dir()
    bodies = {row["body"] for row in _table_rows()}
    assert bodies == {path.name for path in csrc_dir.glob("flashkda_generated_*.cu")}
    assert all((csrc_dir / body).is_file() for body in bodies)
