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

import json

from flashinfer.jit import flash_kda

# One table row per generated body; a row lists every exact architecture that
# compiles it, and the registry holds one entry per (architecture, module) pair.
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


def _table_rows():
    path = (
        flash_kda._get_flash_kda_csrc_dir()
        / flash_kda._FLASH_KDA_GENERATED_METADATA_NAME
    )
    return json.loads(path.read_text())["modules"]


def test_generated_module_table_lists_each_body_once_with_its_archs():
    rows = _table_rows()
    assert len(rows) == EXPECTED_ROWS
    assert len({row["module_ident"] for row in rows}) == EXPECTED_ROWS
    shared = [row for row in rows if row["archs"] == ["sm_100a", "sm_103a"]]
    assert len(shared) == EXPECTED_SHARED_ROWS
    for arch, expected in EXPECTED_SINGLE_ARCH_ROWS.items():
        assert len([row for row in rows if row["archs"] == [arch]]) == expected
    assert sum(len(row["archs"]) for row in rows) == EXPECTED_REGISTRY_ENTRIES


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
