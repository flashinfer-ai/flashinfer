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

The KCP registry (``cake_kcp_jit``) and manifests of the Cake packed KDA training package: the records against the
frozen kernel ABIs, the program names against the identity rule, ``build.json`` (the shipped programs, never a
test-only one) and ``build.test.json`` (the test-only programs, loadable only after ``cake_kcp_plan.enable_test_knobs()``),
and the architecture the workspace byte query selects the local arena's records for.
No device is needed; nothing is built or launched.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

cake_build = cake_package("cake.cake_kda_det_train_kcp", "cake_build")
cake_jit = cake_package("cake.cake_kda_det_train", "cake_jit")
kcp_jit = cake_package("cake.cake_kda_det_train_kcp", "cake_kcp_jit")
kcp_plan = cake_package("cake.cake_kda_det_train_kcp", "cake_kcp_plan")

PACKAGE_DIR = Path(cake_build.__file__).parent
ARCHES = ("sm_100a", "sm_103a", "sm_107a")
GRID = [["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"]]

# The frozen kernel argument lists (name, kind) of the four KCP kernel packages and the two chunk stages of the route.
FWDH_PLAN = [
    ["buffer", "w_tma"], ["buffer", "kg_tma"], ["buffer", "u"], ["buffer", "gk"], ["parameter", "num_heads"],
    ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"], ["buffer", "walk_seg"], ["buffer", "h0"],
    ["buffer", "h0_index"], ["parameter", "h0_hg"], ["parameter", "h0_n"], ["buffer", "out"], ["buffer", "out_index"],
    ["parameter", "out_hg"], ["parameter", "out_n"], ["parameter", "tile_stride"], ["buffer", "h_tape"],
    ["buffer", "v_new"], *GRID,
]  # fmt: skip
DHU_PLAN = [
    ["buffer", "kg_tma"], ["buffer", "qg_tma"], ["buffer", "w_tma"], ["buffer", "do_tma"], ["buffer", "aqk_tma"],
    ["buffer", "gk"], ["parameter", "num_heads"], ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"],
    ["parameter", "scale"], ["buffer", "walk_seg"], ["buffer", "dht"], ["buffer", "dht_index"], ["parameter", "dht_hg"],
    ["parameter", "dht_n"], ["buffer", "out"], ["buffer", "out_index"], ["parameter", "out_hg"], ["parameter", "out_n"],
    ["parameter", "tile_stride"], ["buffer", "dh_tape"], ["buffer", "dv2"], *GRID,
]  # fmt: skip
TRANSFER_PLAN = [
    ["buffer", "w_tma"], ["buffer", "kg_tma"], ["buffer", "u"], ["buffer", "gk"], ["parameter", "num_heads"],
    ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"], ["buffer", "walk_seg"], ["buffer", "out"],
    ["buffer", "out_index"], ["parameter", "out_hg"], ["parameter", "out_n"], ["parameter", "tile_stride"],
    ["parameter", "init_mode"], ["parameter", "use_u"], ["parameter", "write_transposed"], *GRID,
]  # fmt: skip
MERGE_PLAN = [
    ["buffer", "m"], ["buffer", "c"], ["buffer", "x0"], ["buffer", "y"], ["buffer", "chain_start"],
    ["buffer", "elem_m"], ["buffer", "elem_c"], ["buffer", "elem_y"], ["buffer", "x0_index"], ["buffer", "y_final"],
    ["parameter", "ncols"], ["parameter", "transpose_m"], ["parameter", "write_states"], ["parameter", "cols_per_cta"],
    *GRID,
]  # fmt: skip
FWD_O_PLAN = [
    ["buffer", "qg_tma"], ["buffer", "h_tma"], ["buffer", "aqk_tma"], ["buffer", "vn_tma"], ["buffer", "o_out"],
    ["buffer", "chunk_bos"], ["buffer", "chunk_len"], ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"],
    ["parameter", "num_heads"], ["parameter", "scale"], *GRID,
]  # fmt: skip
FWDH_O_PLAN = [
    ["buffer", "w_tma"], ["buffer", "kg_tma"], ["buffer", "u"], ["buffer", "gk"], ["parameter", "num_heads"],
    ["buffer", "seq_chunk_start"], ["parameter", "num_seqs"], ["buffer", "walk_seg"], ["buffer", "h0"],
    ["buffer", "h0_index"], ["parameter", "h0_hg"], ["parameter", "h0_n"], ["parameter", "tile_stride"],
    ["buffer", "qg_tma"], ["buffer", "aqk_tma"], ["buffer", "o_out"], ["buffer", "chunk_bos"], ["buffer", "chunk_len"],
    ["parameter", "scale"], *GRID,
]  # fmt: skip
DAV_PLAN = [
    ["buffer", "do_tma"], ["buffer", "vnew_tma"], ["buffer", "aqk_tma"], ["buffer", "dAqk"], ["buffer", "dv1"],
    ["parameter", "num_heads"], ["parameter", "scale"], *GRID,
]  # fmt: skip
# The K4 merge records (the register-blocked v2 builds the rule names): record -> (build keywords, threads per CTA =
# 128 * cols_per_cta / the thread tile's outputs, architectures).  The pairs builds are scalar on sm_107a (the
# exporter's per-architecture keywords; the program name hashes the keywords above).
MERGE_V2_MODULE = "tart.kda_chunk_kcp_affine_merge.sm100_sm103_sm107.kda_chunk_det_affine_merge_fp32_v2_kcp"
MERGE_RECORDS = {
    "kcp_affine_merge_c8": (
        {"chunk_k": 16, "cols_per_cta": 8, "pairs": True, "ring": 4, "tile": "4x4"},
        64,
        ARCHES,
    ),
    "kcp_affine_merge_c16": (
        {"chunk_k": 16, "cols_per_cta": 16, "pairs": True, "ring": 4, "tile": "4x4"},
        128,
        ARCHES,
    ),
    "kcp_affine_merge_c32": (
        {"chunk_k": 32, "cols_per_cta": 32, "pairs": True, "ring": 4, "tile": "8x4"},
        128,
        ARCHES,
    ),
    "kcp_affine_merge_c64": (
        {"chunk_k": 32, "cols_per_cta": 64, "pairs": True, "ring": 4, "tile": "8x4"},
        256,
        ("sm_100a", "sm_103a"),
    ),
}
# The K4 rule at sample launches: (arch, ncols, n_chains) -> record; T = n_chains * ncols / 128 (the W4 merge launches
# are T = 64; a 256-column map chain counts twice).
MERGE_SELECT = {
    ("sm_103a", 128, 64): "kcp_affine_merge_c64",
    ("sm_100a", 128, 57): "kcp_affine_merge_c64",
    ("sm_107a", 128, 64): "kcp_affine_merge_c16",
    ("sm_107a", 128, 49): "kcp_affine_merge_c16",
    ("sm_107a", 128, 48): "kcp_affine_merge_c32",
    ("sm_103a", 128, 56): "kcp_affine_merge_c16",
    ("sm_103a", 256, 21): "kcp_affine_merge_c16",
    ("sm_103a", 128, 40): "kcp_affine_merge_c32",
    ("sm_107a", 256, 13): "kcp_affine_merge_c32",
    ("sm_107a", 128, 24): "kcp_affine_merge_c16",
    ("sm_107a", 128, 21): "kcp_affine_merge_c16",
    ("sm_103a", 256, 10): "kcp_affine_merge_c16",
    ("sm_103a", 128, 13): "kcp_affine_merge_c16",
    ("sm_103a", 128, 12): "kcp_affine_merge_c8",
    ("sm_107a", 256, 6): "kcp_affine_merge_c8",
    ("sm_100a", 128, 1): "kcp_affine_merge_c8",
}
PLANS = {
    "kcp_fwdh_offset": FWDH_PLAN,
    "kcp_fwdh_offset_vh64": FWDH_PLAN,
    "kcp_fwdh_tape": FWDH_PLAN,
    "kcp_dhu_offset": DHU_PLAN,
    "kcp_dhu_offset_overlap": DHU_PLAN,
    "kcp_dhu_offset_vh64": DHU_PLAN,
    "kcp_dhu_tape": DHU_PLAN,
    "kcp_transfer_walk": TRANSFER_PLAN,
    **dict.fromkeys(MERGE_RECORDS, MERGE_PLAN),
    "kcp_fwd_o": FWD_O_PLAN,
    "kcp_dav": DAV_PLAN,
    "kcp_fwdh_o": FWDH_O_PLAN,
    "kcp_fwdh_o_vh64": FWDH_O_PLAN,
    "kcp_transfer_walk_three_mma": TRANSFER_PLAN,
    "kcp_transfer_walk_plain_tf32": TRANSFER_PLAN,
    "kcp_transfer_walk_bf16x3": TRANSFER_PLAN,
    "kcp_transfer_walk_tf32_hilo": TRANSFER_PLAN,
    "kcp_fwdh_tape_full": FWDH_PLAN,
    "kcp_dhu_tape_full": DHU_PLAN,
}
# Launch block of the records that are not 192 threads wide (the merge records: MERGE_RECORDS); the full-width fwdh
# offset walk runs the split-role schedule (four state, four decay, one TMA and one MMA warp), the shipped transfer
# walk two compute warpgroups (eight compute, one load and one MMA warp), the fused forward walk ten warps in both
# forms.
BLOCKS = {
    "kcp_fwdh_offset": 320,
    "kcp_transfer_walk": 320,
    "kcp_fwdh_o": 320,
    "kcp_fwdh_o_vh64": 320,
}
# The identity rule: ``cake_kda_det_train_<20 hex of sha256("<source module>\\n<sorted JSON of the build keywords>")>``,
# then ``_kcp`` for a program only the context-parallel route runs (every one but the single-rank ``dav`` stage).
# The digest hashes the source module path the program shipped under, so the digests are those of the previous names
# (``kcp_fwd_o``'s KCP module is new: its path is its own).
_W = "sm100_sm103_sm107"
IDENTITIES = {
    "kcp_fwdh_offset": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_det_fwdh_v2_kcp",
        {"program": "offset", "tile_prefetch": 0},
    ),
    "kcp_fwdh_offset_vh64": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_kcp_fwdh_v1",
        {"program": "offset", "vh": 64, "prefetch": True, "tmem_batch": True},
    ),
    "kcp_fwdh_tape": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_kcp_fwdh_v1",
        {"program": "tape", "vh": 64, "tape_store": "tma"},
    ),
    "kcp_dhu_offset": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "offset", "fuse_dv1": 1},
    ),
    "kcp_dhu_offset_overlap": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_det_dhu_v2_kcp",
        {"program": "offset", "tile_prefetch": 0, "fuse_dv1": 1},
    ),
    "kcp_dhu_offset_vh64": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "offset", "vh": 64, "fuse_dv1": 1},
    ),
    "kcp_dhu_tape": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "tape", "vh": 64, "tape_store": "tma", "fuse_dv1": 1},
    ),
    "kcp_transfer_walk": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_det_transfer_walk_bf16x3_v2_kcp",
        {"vh": 64},
    ),
    **{name: (MERGE_V2_MODULE, row[0]) for name, row in MERGE_RECORDS.items()},
    "kcp_fwd_o": (f"tart.kda_chunk_kcp_fwd_o.{_W}.kda_chunk_det_fwd_o_v1_kcp", {}),
    "kcp_dav": (f"tart.kda_chunk_bwd_dav.{_W}.kda_chunk_train_bwd_dav_v1", {}),
    "kcp_fwdh_o": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_det_fwdh_o_v1_kcp",
        {"vh": 128},
    ),
    "kcp_fwdh_o_vh64": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_det_fwdh_o_v1_kcp",
        {"vh": 64},
    ),
    "kcp_transfer_walk_three_mma": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 1},
    ),
    "kcp_transfer_walk_plain_tf32": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 2},
    ),
    "kcp_transfer_walk_bf16x3": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_bf16x3_v1",
        {"vh": 128},
    ),
    "kcp_transfer_walk_tf32_hilo": (
        f"tart.kda_chunk_kcp_transfer_walk.{_W}.kda_chunk_kcp_transfer_walk_v1",
        {"precision_mode": 0},
    ),
    "kcp_fwdh_tape_full": (
        f"tart.kda_chunk_kcp_fwd_h.{_W}.kda_chunk_kcp_fwdh_v1",
        {"program": "tape"},
    ),
    "kcp_dhu_tape_full": (
        f"tart.kda_chunk_kcp_dhu.{_W}.kda_chunk_kcp_dhu_v1",
        {"program": "tape", "fuse_dv1": 1},
    ),
}


# The chunk stage the route runs at the rank's shape with the single-rank kernel: a det program without the ``_kcp``
# suffix (``kcp_fwd_o`` runs the KCP output program).
CHUNK_STAGES = ("kcp_dav",)


def _records() -> dict[str, dict]:
    return {**kcp_jit.MODULES, **kcp_jit.TEST_MODULES}


def test_the_tables_are_disjoint_and_complete() -> None:
    assert set(kcp_jit.MODULES) == set(PLANS) - set(kcp_jit.TEST_MODULES)
    assert set(kcp_jit.TEST_MODULES) == {
        "kcp_transfer_walk_three_mma",
        "kcp_transfer_walk_plain_tf32",
        "kcp_transfer_walk_bf16x3",
        "kcp_transfer_walk_tf32_hilo",
        "kcp_fwdh_tape_full",
        "kcp_dhu_tape_full",
    }
    assert kcp_jit.ARCHES == ARCHES == tuple(cake_jit.ARCHES)
    assert kcp_jit.PROGRAMS == (
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_dhu_offset",
        "kcp_dhu_tape",
        "kcp_transfer_walk",
        "kcp_affine_merge",
        "kcp_fwd_o",
        "kcp_dav",
        "kcp_fwdh_o",
    )
    assert (*kcp_jit.PROGRAMS[:5], "kcp_fwdh_o") == kcp_jit.WALK_PROGRAMS
    assert kcp_jit.PRECISION_MODES == (0, 1, 2)
    assert kcp_jit.TRANSFER_WALKS == ("bf16x3_hw64", "tf32_hilo", "bf16x3")


@pytest.mark.parametrize("name", sorted(PLANS))
def test_record_matches_the_frozen_abi(name: str) -> None:
    record = _records()[name]
    assert record["arg_plan"] == PLANS[name], name
    assert record["arches"] == list(
        MERGE_RECORDS[name][2] if name in MERGE_RECORDS else ARCHES
    )
    assert record["ffi_entry"] == "run"
    assert record["kernel_symbol"] == f"kernel_{record['library']}"
    assert (
        record["launch"]["cooperative"] is False
        and record["launch"]["use_pdl"] is False
    )
    assert record["launch"]["cluster"] == (1, 1, 1)
    assert record["launch"]["block"] == (
        (MERGE_RECORDS[name][1], 1, 1)
        if record["program"] == "kcp_affine_merge"
        else (BLOCKS.get(name, 192), 1, 1)
    )
    if record["program"] == "kcp_affine_merge":
        assert record["grid_rule"] == {
            "cols_per_cta": MERGE_RECORDS[name][0]["cols_per_cta"],
            "kind": "kcp_chains",
        }
    assert isinstance(record["launch"]["dynamic_smem_bytes"], int)
    expected_name = (
        record["program"]
        if record["variant"] is None
        else f"{record['program']}_{record['variant']}"
    )
    assert name == expected_name
    kept = {parameter for kind, parameter in record["arg_plan"] if kind != "grid"}
    assert set(record["logical"]) == kept
    for kind, parameter in record["arg_plan"]:
        if kind == "grid":
            continue
        vocabulary = (
            kcp_jit.LOGICAL_TENSORS if kind == "buffer" else kcp_jit.LOGICAL_SCALARS
        )
        assert record["logical"][parameter] in vocabulary, (name, parameter)


def test_program_names_follow_the_identity_rule() -> None:
    for name, (module, kwargs) in IDENTITIES.items():
        digest = hashlib.sha256(
            f"{module}\n{json.dumps(kwargs, sort_keys=True)}".encode()
        ).hexdigest()
        suffix = "" if name in CHUNK_STAGES else "_kcp"
        assert (
            _records()[name]["library"] == f"cake_kda_det_train_{digest[:20]}{suffix}"
        ), name
    libraries = [record["library"] for record in _records().values()]
    assert len(set(libraries)) == len(libraries), "every record is its own program"
    assert not set(libraries) & {r["library"] for r in cake_jit.MODULES.values()}, (
        "no KCP program shares a name with a single-rank one"
    )


def test_merge_selection_covers_every_launch_once() -> None:
    assert tuple(kcp_jit.MERGE_RULE) == ARCHES
    for (arch, ncols, n_chains), record in MERGE_SELECT.items():
        assert (
            kcp_jit.select_record(
                "kcp_affine_merge", ncols=ncols, n_chains=n_chains, arch=arch
            )
            == record
        ), (arch, ncols, n_chains)
        assert kcp_jit.merge_record(ncols, n_chains, arch) == record
    # every state / map launch is admitted by exactly one record built for its architecture (select_record raises
    # otherwise), and the rule names every merge record on every architecture it ships for
    named: dict[str, set[str]] = {}
    for arch in ARCHES:
        for ncols in (128, 256):
            for n_chains in range(1, 300):
                record = kcp_jit.merge_record(ncols, n_chains, arch)
                assert arch in kcp_jit.MODULES[record]["arches"], (arch, record)
                named.setdefault(record, set()).add(arch)
    assert named == {name: set(MERGE_RECORDS[name][2]) for name in MERGE_RECORDS}
    # without an architecture: the record every architecture names, refused where they differ (T 21-24, 41-48 and
    # above 56)
    assert kcp_jit.merge_record(128, 8) == "kcp_affine_merge_c8"
    with pytest.raises(ValueError):
        kcp_jit.merge_record(128, 64)
    with pytest.raises(ValueError):
        kcp_jit.merge_record(64, 4, "sm_103a")
    with pytest.raises(ValueError):
        kcp_jit.merge_record(128, 4, "sm_90a")
    # the rule needs the chain count
    with pytest.raises(ValueError):
        kcp_jit.select_record("kcp_affine_merge", ncols=128, arch="sm_103a")


def test_selection_and_grids() -> None:
    assert kcp_jit.select_record("kcp_fwdh_offset", form="full") == "kcp_fwdh_offset"
    assert (
        kcp_jit.select_record("kcp_fwdh_offset", form="split") == "kcp_fwdh_offset_vh64"
    )
    with pytest.raises(ValueError):
        kcp_jit.select_record("kcp_fwdh_offset")  # two records: the form must be given
    assert kcp_jit.select_record("kcp_fwdh_tape", form="split") == "kcp_fwdh_tape"
    assert kcp_jit.select_record("kcp_fwdh_o", form="full") == "kcp_fwdh_o"
    assert kcp_jit.select_record("kcp_fwdh_o", form="split") == "kcp_fwdh_o_vh64"
    with pytest.raises(ValueError):
        kcp_jit.select_record("kcp_fwdh_o")
    assert kcp_jit.launch_grid("kcp_fwdh_o", n_walk=3, num_heads=4) == (12, 1, 1)
    assert kcp_jit.launch_grid("kcp_fwdh_o_vh64", n_walk=3, num_heads=4) == (24, 1, 1)
    assert kcp_jit.launch_grid("kcp_fwdh_offset", n_walk=3, num_heads=4) == (12, 1, 1)
    assert kcp_jit.launch_grid("kcp_fwdh_offset_vh64", n_walk=3, num_heads=4) == (
        24,
        1,
        1,
    )
    assert kcp_jit.launch_grid("kcp_dhu_tape", n_walk=3, num_heads=4) == (24, 1, 1)
    assert kcp_jit.launch_grid("kcp_dhu_tape_full", n_walk=3, num_heads=4) == (12, 1, 1)
    assert kcp_jit.launch_grid("kcp_transfer_walk_bf16x3", n_walk=3, num_heads=4) == (
        12,
        1,
        1,
    )
    assert kcp_jit.launch_grid("kcp_transfer_walk", n_walk=3, num_heads=4) == (
        24,
        1,
        1,
    )
    assert kcp_jit.launch_grid(
        "kcp_transfer_walk_tf32_hilo", n_walk=3, num_heads=4
    ) == (12, 1, 1)
    assert kcp_jit.launch_grid("kcp_affine_merge_c16", n_chains=5, ncols=256) == (
        80,
        1,
        1,
    )
    assert kcp_jit.launch_grid("kcp_affine_merge_c64", n_chains=2, ncols=128) == (
        4,
        1,
        1,
    )
    assert kcp_jit.launch_grid("kcp_affine_merge_c32", n_chains=3, ncols=256) == (
        24,
        1,
        1,
    )
    with pytest.raises(ValueError):
        kcp_jit.launch_grid("kcp_fwdh_offset", n_walk=0, num_heads=4)
    # the chunk stages: one CTA per (chunk, head) tile whatever the SM count, one CTA per chunk pair and head
    assert kcp_jit.launch_grid(
        "kcp_fwd_o", num_chunks_pad=10, num_heads=3, sm_count=152
    ) == (30, 1, 1)
    assert kcp_jit.launch_grid(
        "kcp_fwd_o", num_chunks_pad=400, num_heads=4, sm_count=152
    ) == (1600, 1, 1)
    assert kcp_jit.launch_grid("kcp_fwd_o", num_chunks_pad=268, num_heads=64) == (
        17152,
        1,
        1,
    )
    with pytest.raises(ValueError):
        kcp_jit.launch_grid("kcp_fwd_o", num_chunks_pad=0, num_heads=4, sm_count=152)
    assert kcp_jit.launch_grid("kcp_dav", num_chunks_pad=10, num_heads=3) == (5, 3, 1)
    with pytest.raises(ValueError):
        kcp_jit.launch_grid("kcp_dav", num_chunks_pad=9, num_heads=3)


def test_test_record_selection_by_knobs() -> None:
    assert kcp_jit.test_record() is None
    assert kcp_jit.test_record(0, "bf16x3_hw64") is None
    assert kcp_jit.test_record(0, "tf32_hilo")[0] == "kcp_transfer_walk_tf32_hilo"
    assert kcp_jit.test_record(1)[0] == "kcp_transfer_walk_three_mma"
    assert kcp_jit.test_record(1, "tf32_hilo")[0] == "kcp_transfer_walk_three_mma"
    assert kcp_jit.test_record(2)[0] == "kcp_transfer_walk_plain_tf32"
    assert kcp_jit.test_record(0, "bf16x3")[0] == "kcp_transfer_walk_bf16x3"
    with pytest.raises(ValueError, match="name two test-only programs"):
        kcp_jit.test_record(1, "bf16x3")
    with pytest.raises(ValueError):
        kcp_jit.test_record(3)
    with pytest.raises(ValueError):
        kcp_jit.test_record(0, "fp32")


def test_walk_form_rules_and_records() -> None:
    # the offset programs: the fit rule (two shipped records); the tape programs: split on every launch, the full-width
    # record test-only under walk_form "full"; the fused forward walk: a rule per architecture (two shipped records)
    assert kcp_jit.WALK_FORM_RULES == {
        "kcp_fwdh_offset": "fit",
        "kcp_fwdh_tape": "split",
        "kcp_dhu_offset": "fit",
        "kcp_dhu_tape": "split",
        "kcp_fwdh_o": kcp_jit.FWDH_O_FORM_RULE,
    }
    assert kcp_jit.FWDH_O_FORM_RULE == {
        "sm_100a": ((32, "split"), (None, "full")),
        "sm_103a": ((32, "split"), (None, "full")),
        "sm_107a": ((48, "fit"), (None, "full")),
    }
    assert set(kcp_jit.FWDH_O_FORM_RULE) == set(ARCHES)
    sms = 152
    assert kcp_jit.walk_form("kcp_fwdh_offset", [222, 222], 64, sms) == "split"
    assert kcp_jit.walk_form("kcp_fwdh_offset", [111, 59, 52], 64, sms) == "split"
    assert kcp_jit.walk_form("kcp_fwdh_offset", [104, 104, 19, 18], 64, sms) == "split"
    assert kcp_jit.walk_form("kcp_dhu_offset", [124, 124, 124], 64, sms) == "full"
    assert (
        kcp_jit.walk_form("kcp_dhu_offset", [39] * 7 + [25] * 2 + [24] * 5, 64, sms)
        == "full"
    )
    assert kcp_jit.walk_form("kcp_dhu_offset", [60, 59, 50, 48], 64, sms) == "full"
    assert kcp_jit.walk_form("kcp_fwdh_offset", [64] * 7, 1, sms) == "split"
    assert kcp_jit.walk_form("kcp_fwdh_offset", [], 64, sms) == "full"
    assert kcp_jit.walk_form("kcp_fwdh_tape", [124, 124, 124], 64, sms) == "split"
    # the fused forward walk: the architecture's rule on every launch (one wave or several; on sm_100a / sm_103a the
    # two-CTA record up to 32 heads, the full-width one above; on sm_107a the fit rule up to 48 heads, the full-width
    # record above: the W4 list at 32 to 48 heads, 512 to 768 two-CTA CTAs on 304 slots, outlasts its longest chain),
    # the knob's form where given
    w4 = [48, 48, 32, 32, 28, 28, 20, 20]
    by_heads = {
        1: "split",
        4: "split",
        16: "split",
        32: "split",
        33: "full",
        64: "full",
    }
    fit_by_heads = {
        1: "split",
        4: "split",
        16: "split",
        32: "split",
        33: "split",
        48: "split",
        49: "full",
        64: "full",
    }
    for arch in ARCHES:
        for chains in (w4, [222, 222], [48]):
            for heads, form in (
                fit_by_heads if arch == "sm_107a" else by_heads
            ).items():
                if arch == "sm_107a" and chains is w4 and heads in (32, 33, 48):
                    form = "full"
                assert (
                    kcp_jit.walk_form("kcp_fwdh_o", chains, heads, sms, arch=arch)
                    == form
                ), (arch, chains, heads)
        assert (
            kcp_jit.walk_form("kcp_fwdh_o", [48], 4, sms, knob="full", arch=arch)
            == "full"
        )
        assert (
            kcp_jit.walk_form("kcp_fwdh_o", [48], 4, sms, knob="split", arch=arch)
            == "split"
        )
    with pytest.raises(ValueError, match="per architecture"):
        kcp_jit.walk_form("kcp_fwdh_o", w4, 64, sms)
    with pytest.raises(ValueError, match="per architecture"):
        kcp_jit.walk_form("kcp_fwdh_o", w4, 64, sms, arch="sm_90a")
    assert kcp_jit.walk_form("kcp_fwdh_o", w4, 64, sms, knob="split") == "split"
    # the other programs' rules take no architecture
    assert (
        kcp_jit.walk_form("kcp_fwdh_offset", [222, 222], 64, sms, arch="sm_107a")
        == "split"
    )
    assert kcp_jit.walk_form("kcp_dhu_tape", [39] * 14, 64, sms) == "split"
    assert (
        kcp_jit.walk_form("kcp_fwdh_offset", [222, 222], 64, sms, knob="full") == "full"
    )
    assert (
        kcp_jit.walk_form("kcp_fwdh_tape", [222, 222], 64, sms, knob="full") == "full"
    )
    with pytest.raises(ValueError):
        kcp_jit.walk_form("kcp_fwdh_tape", [222, 222], 64, sms, knob="half")
    assert kcp_jit.test_walk_record("kcp_fwdh_tape", "full")[0] == "kcp_fwdh_tape_full"
    assert kcp_jit.test_walk_record("kcp_dhu_tape", "full")[0] == "kcp_dhu_tape_full"
    with pytest.raises(ValueError):
        kcp_jit.test_walk_record("kcp_fwdh_offset", "full")
    for name in ("kcp_fwdh_tape_full", "kcp_dhu_tape_full"):
        record = kcp_jit.TEST_MODULES[name]
        assert record["test_knobs"] == {"walk_form": "full"}
        assert record["select_rule"] == {"kind": "always"}
        assert record["grid_rule"] == {"kind": "kcp_segments_heads"}
    with pytest.raises(ValueError):
        kcp_jit.test_walk_record("kcp_fwdh_o", "full")
    for name in (
        "kcp_fwdh_offset_vh64",
        "kcp_dhu_offset_vh64",
        "kcp_fwdh_tape",
        "kcp_dhu_tape",
        "kcp_fwdh_o_vh64",
    ):
        assert kcp_jit.MODULES[name]["grid_rule"] == {
            "ctas_per_head": 2,
            "kind": "kcp_segments_heads",
        }
    assert kcp_jit.MODULES["kcp_fwdh_offset_vh64"]["select_rule"] == {
        "form": "split",
        "kind": "walk_form",
    }
    assert kcp_jit.MODULES["kcp_fwdh_tape"]["select_rule"] == {"kind": "always"}
    assert kcp_jit.MODULES["kcp_fwdh_o"]["select_rule"] == {
        "form": "full",
        "kind": "walk_form",
    }
    assert kcp_jit.MODULES["kcp_fwdh_o_vh64"]["select_rule"] == {
        "form": "split",
        "kind": "walk_form",
    }
    assert kcp_jit.MODULES["kcp_fwdh_o"]["grid_rule"] == {"kind": "kcp_segments_heads"}


def test_walk_heads_selection() -> None:
    # the full-width adjoint offset walk: the plain build up to 63 heads, the overlap-schedule build from 64 (the W4
    # rows); its split form and the forward offset walk keep the walk_form rule
    assert kcp_jit.WALK_HEADS_RULE == {
        "kcp_dhu_offset": ((63, "kcp_dhu_offset"), (None, "kcp_dhu_offset_overlap"))
    }
    for name in ("kcp_dhu_offset", "kcp_dhu_offset_overlap"):
        assert kcp_jit.MODULES[name]["select_rule"] == {
            "form": "full",
            "kind": "walk_heads",
        }
        assert kcp_jit.MODULES[name]["grid_rule"] == {"kind": "kcp_segments_heads"}
    for heads, record in (
        (1, "kcp_dhu_offset"),
        (4, "kcp_dhu_offset"),
        (63, "kcp_dhu_offset"),
        (64, "kcp_dhu_offset_overlap"),
        (128, "kcp_dhu_offset_overlap"),
    ):
        assert kcp_jit.walk_heads_record("kcp_dhu_offset", heads) == record
        assert (
            kcp_jit.select_record("kcp_dhu_offset", form="full", num_heads=heads)
            == record
        )
        assert (
            kcp_jit.select_record("kcp_dhu_offset", form="split", num_heads=heads)
            == "kcp_dhu_offset_vh64"
        )
        assert (
            kcp_jit.select_record("kcp_fwdh_offset", form="full", num_heads=heads)
            == "kcp_fwdh_offset"
        )
    assert kcp_jit.launch_grid("kcp_dhu_offset_overlap", n_walk=3, num_heads=64) == (
        192,
        1,
        1,
    )
    with pytest.raises(ValueError):
        kcp_jit.select_record("kcp_dhu_offset", form="full")  # the head count is needed
    with pytest.raises(ValueError):
        kcp_jit.walk_heads_record("kcp_dhu_offset", 0)
    with pytest.raises(ValueError):
        kcp_jit.walk_heads_record("kcp_fwdh_offset", 64)


def test_the_shipped_transfer_walk_is_the_half_width_bf16x3_program() -> None:
    record = kcp_jit.MODULES["kcp_transfer_walk"]
    assert record["grid_rule"] == {"ctas_per_head": 2, "kind": "kcp_segments_heads"}
    assert record["library"] == "cake_kda_det_train_8ffd931ca0595948b608_kcp"
    previous = kcp_jit.TEST_MODULES["kcp_transfer_walk_tf32_hilo"]
    assert previous["grid_rule"] == {"kind": "kcp_segments_heads"}
    assert previous["library"] == "cake_kda_det_train_1ac972f0cb7cc7b13cf7_kcp"
    assert previous["test_knobs"] == {"transfer_walk": "tf32_hilo"}
    assert kcp_jit.TRANSFER_WALKS[0] == "bf16x3_hw64"


def test_shipped_manifest_lists_the_kcp_programs_and_no_test_only_one() -> None:
    manifest = cake_build.manifest()
    libraries = cake_build.libraries_by_name()
    for name, record in kcp_jit.MODULES.items():
        for arch in record["arches"]:
            library = kcp_jit.library_name(name, arch)
            assert library == f"{record['library']}_{arch}"
            entry = libraries[library]
            assert entry["arch"] == arch and entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert "--use_fast_math" in entry["nvcc_flags"]
            for source in entry["sources"]:
                path = PACKAGE_DIR / source
                assert path.is_file(), source
                assert (
                    hashlib.sha256(path.read_bytes()).hexdigest()
                    == entry["sha256"][source]
                ), source
    names = {entry["name"] for entry in manifest["libraries"]}
    test_only = {
        f"{record['library']}_{arch}"
        for record in kcp_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    assert not names & test_only


# The PTX ISA 8.8 instructions (CUDA 12.9; CUDA 12.8's ptxas refuses them): a 256-bit ld / st, a three-input
# min / max.
_PTX_ISA_8_8 = re.compile(
    rb"\b(?:min|max)(?:\.\w+)*\s+%[^;,\"]+(?:,[^;,\"]+){3};"
    rb"|\b(?:ld|st)(?:\.[\w:]+)*\.v8\.[bfsu]32\b"
)


@pytest.mark.parametrize("manifest", ["build.json", "build.test.json"])
def test_an_entry_whose_units_use_ptx_isa_8_8_needs_cuda_12_9(manifest: str) -> None:
    """An entry whose units use a PTX ISA 8.8 instruction (the 256-bit stores of the fused forward walk, the output
    stages and the full-width tape walks) states ``min_cuda`` 12.9 or newer: CUDA 12.8 cannot build it."""
    path = PACKAGE_DIR / manifest
    if not path.is_file():
        pytest.skip(f"{manifest} is not shipped here")
    wide = []
    for entry in json.loads(path.read_text())["libraries"]:
        if any(
            _PTX_ISA_8_8.search((PACKAGE_DIR / source).read_bytes())
            for source in entry["sources"]
        ):
            wide.append(entry["name"])
            assert tuple(map(int, entry["min_cuda"].split("."))) >= (12, 9), entry
    assert any(name.endswith("_sm_100a") for name in wide), wide


def test_test_manifest_is_the_test_only_programs_under_the_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = PACKAGE_DIR / kcp_jit.TEST_MANIFEST
    if not path.is_file():
        pytest.skip(f"{kcp_jit.TEST_MANIFEST} is not shipped here")
    kcp_jit._test_manifest.cache_clear()
    monkeypatch.setattr(kcp_plan, "_TEST_KNOBS_ENABLED", False)
    assert kcp_jit.test_knobs_enabled() is False
    with pytest.raises(RuntimeError, match="enable_test_knobs"):
        kcp_jit.test_manifest()
    monkeypatch.setattr(kcp_plan, "_TEST_KNOBS_ENABLED", True)
    kcp_jit._test_manifest.cache_clear()
    assert kcp_jit.test_knobs_enabled() is True
    document = kcp_jit.test_manifest()
    shipped = cake_build.manifest()
    assert document["schema"] == shipped["schema"] == "cake-build/1"
    assert (
        document["package"]
        == shipped["package"]
        == "flashinfer.experimental.cake.cake_kda_det_train_kcp"
    )
    # the shipped manifest is re-rendered by every kernel update of the package, the test-only manifest only when its
    # programs change: the two generator records share their form, not their stamp
    assert (
        set(document["generator"])
        == set(shipped["generator"])
        == {"cake_revision", "cuda", "rendered_at"}
    )
    names = {entry["name"] for entry in document["libraries"]}
    assert names == {
        f"{record['library']}_{arch}"
        for record in kcp_jit.TEST_MODULES.values()
        for arch in ARCHES
    }
    for entry in document["libraries"]:
        assert "--use_fast_math" in entry["nvcc_flags"]
        for source in entry["sources"]:
            source_path = PACKAGE_DIR / source
            assert source_path.is_file(), source
            assert (
                hashlib.sha256(source_path.read_bytes()).hexdigest()
                == entry["sha256"][source]
            ), source
    for name in kcp_jit.TEST_MODULES:
        for arch in ARCHES:
            assert kcp_jit.test_library_name(name, arch) in names
    # an opt-out after the parse is cached closes the test-only programs again (the gate is outside the caches)
    monkeypatch.setattr(kcp_plan, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(RuntimeError, match="enable_test_knobs"):
        kcp_jit.test_manifest()
    with pytest.raises(RuntimeError, match="enable_test_knobs"):
        kcp_jit.load_test(next(iter(kcp_jit.TEST_MODULES)), ARCHES[0])
    monkeypatch.setattr(kcp_plan, "_TEST_KNOBS_ENABLED", True)
    with pytest.raises(RuntimeError):
        kcp_jit.test_library_name("kcp_transfer_walk_three_mma", "sm_90a")
    kcp_jit._test_manifest.cache_clear()


REPO = Path(__file__).resolve().parents[2]
PACKAGE_NAME = "flashinfer.experimental.cake.cake_kda_det_train_kcp"
SINGLE_RANK_NAME = "flashinfer.experimental.cake.cake_kda_det_train"


def _manifest_checker():
    """``scripts/check_cake_build_manifests.py`` of this checkout, loaded as a module (it never imports flashinfer)."""
    path = REPO / "scripts" / "check_cake_build_manifests.py"
    if not path.is_file():
        pytest.skip(
            "scripts/check_cake_build_manifests.py is not beside this test (installed package)"
        )
    spec = importlib.util.spec_from_file_location("check_cake_build_manifests", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules.setdefault("check_cake_build_manifests", module)
    spec.loader.exec_module(module)
    return module


def test_manifest_checker_row_requires_the_single_rank_package() -> None:
    """The checker's row of this package names its directory and requires the single-rank sibling, and its
    manifest declares that sibling (``common.requires.packages``)."""
    checker = _manifest_checker()
    row = checker.package_named(PACKAGE_NAME)
    assert row is not None
    assert (row.path, row.loader) == (
        "flashinfer/experimental/cake/cake_kda_det_train_kcp",
        "cake_kcp_jit.py",
    )
    assert row.requires == (SINGLE_RANK_NAME,)
    assert checker.check_requires(row) == []
    assert checker.check_required_packages(row, cake_build.manifest()) == []


def test_manifest_checker_required_packages_rule() -> None:
    """``check_required_packages``: the manifest lists the directory names of ``Package.requires`` sorted and without
    duplicates (as #8928 specifies), whatever the order of ``Package.requires``, and lists none when the package
    requires none."""
    checker = _manifest_checker()

    def package(*requires: str):
        return checker.Package(
            "p.pkg", "p/pkg", "cake_jit.py", checker.refs_generic, requires=requires
        )

    def manifest(**requires):
        return {"common": {"requires": {"cuda_toolkit": {"min": "12.8"}, **requires}}}

    def messages(row, document):
        return [f.message for f in checker.check_required_packages(row, document)]

    two = package("p.sib", "p.other")
    assert messages(two, manifest(packages=["other", "sib"])) == []
    assert messages(package(), manifest()) == []
    (missing,) = messages(two, manifest())
    assert "common.requires.packages is missing; expected ['other', 'sib']" in missing
    (unsorted,) = messages(two, manifest(packages=["sib", "other"]))
    assert "!= ['other', 'sib']" in unsorted
    (duplicate,) = messages(two, manifest(packages=["other", "other", "sib"]))
    assert "!= ['other', 'sib']" in duplicate
    (dotted,) = messages(two, manifest(packages=["p.other", "p.sib"]))
    assert "!= ['other', 'sib']" in dotted
    (extra,) = messages(package(), manifest(packages=["sib"]))
    assert "Package.requires is empty" in extra


# --------------------------------------------------------------------------- the byte query's architecture

BYTE_QUERY_CALLS = ("plan", "workspace_bytes", "tables_bytes")


def _context_on(device: str):
    """Rank 0's context of a two-rank contiguous pack (148 SMs), seen as a context on ``device``: the byte query reads
    the context's host fields and its ``device``; a CPU host builds the context on the CPU."""
    part = kcp_plan.cake_kda_det_train_kcp_partition([0, 3000, 8192], 2)
    ctx = kcp_plan.cake_kda_det_train_kcp_build_context(
        part, None, 0, "cpu", 4, sm_count=148
    )
    assert ctx.T_r > 0

    class OnDevice:
        def __getattr__(self, name: str):
            return getattr(ctx, name)

    view = OnDevice()
    view.device = torch.device(device)
    return view


def _spy_byte_query(monkeypatch) -> list[tuple[str, object]]:
    """Record the ``arch`` each of :data:`BYTE_QUERY_CALLS` of the single-rank backend receives (the real function
    then runs); every context's intra record is taken to write its scratch table set, so all three are called."""
    single_rank = cake_package("cake.cake_kda_det_train_kcp", "_single_rank")
    backend = single_rank.module("cake_backend")
    seen: list[tuple[str, object]] = []
    for name in BYTE_QUERY_CALLS:
        real = getattr(backend, name)

        def spy(*args, _real=real, _name=name, **kwargs):
            seen.append((_name, kwargs.get("arch", "not passed")))
            return _real(*args, **kwargs)

        monkeypatch.setattr(backend, name, spy)
    monkeypatch.setattr(kcp_plan, "intra_writes_tables", lambda *a, **k: True)
    return seen


def test_workspace_bytes_plan_for_the_contexts_device_architecture(
    monkeypatch,
) -> None:
    """The byte query plans each context's local arena, its arena bytes and its scratch tables for the architecture
    of the context's device, as the forward plans its local stages, never for that of the current CUDA device: here
    the current device (``cuda:0``) is another model than the context's (``cuda:1``)."""
    current, own = cake_jit.ARCHES[0], cake_jit.ARCHES[1]
    by_index = {0: current, 1: own}
    capability = {arch: cap for cap, arch in cake_jit.ARCH_BY_CAPABILITY.items()}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_capability",
        lambda index=None: capability[by_index[0 if index is None else int(index)]],
    )
    monkeypatch.setattr(kcp_plan, "_cuda_arch", lambda index: by_index[int(index)])
    ctx = _context_on("cuda:1")
    seen = _spy_byte_query(monkeypatch)
    nbytes = kcp_plan.cake_kda_det_train_kcp_workspace_bytes(ctx, H=64)
    assert nbytes > 0
    assert seen == [(name, own) for name in BYTE_QUERY_CALLS]


@pytest.mark.parametrize("device", ["cuda:1", "cpu"])
def test_workspace_bytes_without_cuda_pass_no_architecture(
    monkeypatch, device: str
) -> None:
    """On a host without CUDA the byte query passes ``arch=None`` (the existing backend's default) and makes no CUDA
    call, whatever device the context names."""

    def no_cuda_call(*args, **kwargs):
        raise AssertionError("the byte query made a CUDA call on a host without CUDA")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for name in ("current_device", "get_device_capability", "get_device_properties"):
        monkeypatch.setattr(torch.cuda, name, no_cuda_call)
    monkeypatch.setattr(kcp_plan, "_cuda_arch", no_cuda_call)
    ctx = _context_on(device)
    seen = _spy_byte_query(monkeypatch)
    assert kcp_plan.cake_kda_det_train_kcp_workspace_bytes(ctx, H=64) > 0
    assert seen == [(name, None) for name in BYTE_QUERY_CALLS]
