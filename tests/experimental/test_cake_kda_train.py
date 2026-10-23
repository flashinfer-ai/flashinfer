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
"""

import hashlib
import itertools
import json
import re
import types
from pathlib import Path

import pytest
import torch

from tests.test_helpers.cake_kda_train_reference import (
    GRAD_NAMES,
    elementwise_gate,
    error_metrics,
    kda_exact_with_grads,
    make_inputs,
)
from tests.test_helpers.cake_package import cake_package

# The backend suite names its package once, through the shared alias helper, and imports nothing else from
# flashinfer: a build that vendors the package under another import path runs it unchanged after pointing the
# ``PACKAGES`` entry at its own package.  The public entry points of ``flashinfer.kda_train`` are covered by
# ``test_cake_kda_train_api.py``.
cake_backend = cake_package("cake_kda_train", "cake_backend")
cake_build = cake_package("cake_kda_train", "cake_build")
cake_jit = cake_package("cake_kda_train", "cake_jit")
cake_kcp_jit = cake_package("cake_kda_train", "cake_kcp_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent

ARCHES = tuple(cake_jit.ARCHES)
PATHS = ("fwd", "fwd_nosave", "light", "light_states", "tape", "replay")
BT, D, SCALE = cake_backend.BT, cake_backend.D, cake_backend.SCALE
KdaChunkSaved = cake_backend.KdaChunkSaved
KdaWorkspace = cake_backend.KdaWorkspace

# Accuracy gate, declared before the first run: every output elementwise within
# ``|cand - ref| <= ATOL + RTOL * |ref|`` of the exact FP64 per-document recurrence of the same BF16 / FP32 inputs
# (the BF16 outputs and the FP32 ``dg`` alike, compared in FP32).
ATOL = RTOL = 1e-2
OUTPUT_NAMES = ("out",) + GRAD_NAMES

# Dispatch constants the registry rules must reproduce (the kernels' validated regimes).
TABLE_V2_MAX_DOCS = 256  # the chunk-parallel layout table serves up to 256 documents
RECUR_CTAS_PER_SM = (
    4  # the 32-column serial forms fit when num_docs * H * 4 <= 4 * sm_count
)
RECUR_SLICES = {"v1": 2, "v2": 4}
FUSED_FWD_MIN_H = 32  # the serial forward forms the WY representation itself from 32 heads (sm_107a: 32 only)
FUSED_BWD_MIN_H = 64  # the 64-column serial recompute forms it from 64 heads
# The forward's carry || pass-2 overlap pair on the segmented route: the rows whose pass-2 grid is at least
# OV1_MIN_WAVES_BY_ARCH[arch] waves of the device and at most OV1_MAX_SEGMENT_HEADS segment-heads (the flag words of the
# eye buffer); the pass-2 CTA order is the carry's release order up to OV1_ORDER_MAX_H heads, the chain-major order
# above.  The saving forward launches the pair's token-row twins (the saved Aqk rows) and the saved-Akk WY program.
OV1_MIN_WAVES_BY_ARCH = {"sm_100a": 3.0, "sm_103a": 3.0, "sm_107a": 2.3}
OV1_ORDER_MAX_H = 2
OV1_MAX_SEGMENT_HEADS = 8192
PDL_RECORDS = (
    "fwdh_p3_o_ov",
    "fwdh_p3_o_ov_id",
    "fwdh_p3_o_ov_rows",
    "fwdh_p3_o_ov_id_rows",
)
# the saving forward's token-row records: where one runs, no program of the path reads the internal Aqk / Akk copies;
# the forward without a saved set launches their internal-row twins instead (the paired serial rows -- the 32-column
# form below the fused threshold -- launch wy_fwd_rows and fwdh_o_v2_rows; the 64-column form keeps the internal rows)
TOKEN_ROW_RECORDS = (
    "fwdh_wy_rows",
    "wy_fwd_rows",
    "fwdh_p3_o_ov_rows",
    "fwdh_p3_o_ov_id_rows",
    "fwdh_o_v2_rows",
)
INTERNAL_ROW_TWIN = {
    "fwdh_wy_rows": "fwdh_wy",
    "wy_fwd_rows": "wy_fwd",
    "fwdh_p3_o_ov_rows": "fwdh_p3_o_ov",
    "fwdh_p3_o_ov_id_rows": "fwdh_p3_o_ov_id",
    "fwdh_o_v2_rows": "fwdh_o_v2",
}
DAVQ_MIN_H = 16  # the fused gradient hosts dAqk on the serial route from 16 heads
# from this many heads the fused gradient runs with its L2 prefetch / evict-first hints off (the `_nohint` records:
# the same kernel built with the hint thresholds beyond any head count); the other architectures keep the hints
DQKG_NOHINT_MIN_H_BY_ARCH = {"sm_107a": 16}
FEW_CHAINS = 4  # at most four (document, head) chains: the few-chain segmented rows
# the saved-state pass 2 forms the WY representation itself (no WY launch) up to 32 (document, head) chains; on
# every segmented row the adjoint recurrence accumulates dv and the fused gradient hosts dAqk (no dav launch)
FUSED_P3_MAX_CHAINS = 32
DHU_TMA_WAVES = (
    4  # the segmented adjoint recurrence stores its tapes by TMA below four CTAs per SM
)
# on sm_103a the segmented adjoint recurrence of the rows below two waves of CTA pairs (two per SM: four of one CTA)
# prefetches the next chunk's tiles (dhu_v3p_f*); the plain TMA-tape form keeps the boundary rows there
DHU_PREFETCH_WAVES_BY_ARCH = {"sm_103a": 4.0}
# The segmented recurrences: at most SEG_MAX_H heads, clamp(ceil(SEG_SLOT_FACTOR * sm / (N * H)), 1, SEG_P_MAX)
# segments per document, at most SEG_P_SMALL of them with N * H <= SEG_P_SMALL_MAX_NH (document, head) chains.
SEG_MAX_H = 8
SEG_SLOT_FACTOR = 4
SEG_P_MAX = 64
SEG_P_SMALL = 38
SEG_P_SMALL_BY_ARCH = {
    "sm_100a": 37
}  # the few-chain count of the architectures with one of their own
SEG_P_SMALL_MAX_NH = 4
# the table stage is optional too: on the forward and the replay of the segmented route up to 256 documents the
# intra program writes the tables and no table record is selected
OPTIONAL_STAGES = (
    "table",
    "seg_table",
    "seg_carry",
    "fwdh_p3",
    "wy_fwd",
    "wy",
    "fwdh",
    "dav",
)
# The stages of each path, in the registry's order (the forward's; the backward's with the tables, the intra
# program on the replay path and seg_carry on the segmented route).
_BWD = (
    "table",
    "seg_table",
    "wy",
    "fwdh",
    "seg_carry",
    "fwdh_p3",
    "bwd_prep",
    "dav",
    "dhu",
    "dqkg",
    "glm_intra",
    "gate_epilogue",
)
PATH_STAGES = {
    "fwd": (
        "table",
        "seg_table",
        "fwd_intra",
        "wy_fwd",
        "fwdh",
        "seg_carry",
        "fwdh_p3",
    ),
    "light": _BWD,
    "light_states": _BWD,
    "tape": _BWD,
    "replay": tuple(s for s in cake_jit.STAGES if s in _BWD or s == "fwd_intra"),
}
# the forward without a saved set: the same stages, the internal-row records
PATH_STAGES["fwd_nosave"] = PATH_STAGES["fwd"]

# Small packings: single documents around the chunk length, packs with unaligned tails, a zero-length document
# (a no-op that still counts in the chunk bound) and pad chunks.
SHAPES = [
    ((1,), 1),
    ((63,), 2),
    ((64,), 1),
    ((65,), 4),
    ((127,), 1),
    ((128,), 2),
    ((129,), 1),
    ((1000,), 1),
    ((63, 1, 65), 4),
    ((64, 64), 1),
    ((100, 0, 300), 2),
    ((100, 640, 77, 1000, 129), 2),
    # the overlap pair's rows (148 SMs): four documents at 2 heads (256 segments: the release order), seven at 8
    # (77 segments: the chain-major order)
    ((63, 1, 65, 30), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
]


def _shape_id(shape) -> str:
    lengths, heads = shape
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _manifest_or_skip()


# --------------------------------------------------------------------------- registry and plan (no device)


def _rule_kinds(rule: dict) -> list[str]:
    kinds = [rule["kind"]]
    if rule["kind"] in ("all_of", "any_of"):
        for sub_rule in rule["rules"]:
            kinds.extend(_rule_kinds(sub_rule))
    return kinds


SELECT_KINDS = (
    "always",
    "otherwise",
    "all_of",
    "any_of",
    "ctas_at_least",
    "ctas_below",
    "ctas_between",
    "docs_at_most",
    "docs_above",
    "heads_at_most",
    "heads_above",
    "heads_at_least",
    "heads_below",
    "chains_at_most",
    "chains_above",
    "slots_at_most",
    "slots_above",
    "walk_ctas_below",
    "walk_ctas_at_most",
    "walk_ctas_at_least",
    "walk_ctas_above",
    "segments_heads_at_most",
    "segments_heads_above",
    "arch_in",
    "path_in",
)
GRID_KINDS = (
    "fixed",
    "chunks",
    "chunk_blocks",
    "pairs_bounded",
    "per_doc_head",
    "sm_bounded",
    "docs_heads",
    "segments_heads",
)
# The records of the segmented adjoint recurrence launch twice per stage (the carry between the passes); the
# forward recurrence's passes are two stages (fwdh, fwdh_p3).
TWO_PASS_RECORDS = (
    "dhu_v3_f",
    "dhu_v3_f_wt",
    "dhu_v3_f_replay",
    "dhu_v3p_f",
    "dhu_v3p_f_wt",
    "dhu_v3p_f_replay",
    "dhu_v3r_f",
    "dhu_v3r_f_wt",
    "dhu_v3r_f_replay",
    "dhu_v3s_f",
    "dhu_v3s_f_wt",
    "dhu_v3s_f_replay",
)


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.MODULES) == 61
    assert len({r["library"] for r in cake_jit.MODULES.values()}) == 59
    assert tuple(cake_jit.PATHS) == PATHS
    assert tuple(cake_jit.OPTIONAL_STAGES) == OPTIONAL_STAGES
    assert cake_jit.SEGMENTS == {
        "max_heads": SEG_MAX_H,
        "slot_factor": SEG_SLOT_FACTOR,
        "max_segments": SEG_P_MAX,
        "small_segments": SEG_P_SMALL,
        "small_segments_by_arch": SEG_P_SMALL_BY_ARCH,
        "small_max_chains": SEG_P_SMALL_MAX_NH,
        "prop_max_chains": FEW_CHAINS,
    }
    stages = set()
    tensors, scalars = set(cake_jit.LOGICAL_TENSORS), set(cake_jit.LOGICAL_SCALARS)
    for name, record in cake_jit.MODULES.items():
        assert record["stage"] in cake_jit.STAGES, name
        stages.add(record["stage"])
        expected_name = (
            record["stage"]
            if record["variant"] is None
            else f"{record['stage']}_{record['variant']}"
        )
        assert name == expected_name, name
        assert list(record["arches"]) == list(ARCHES), name
        assert record["ffi_entry"] == "run", name
        assert record["kernel_symbol"] == f"kernel_{record['library']}", name
        assert re.fullmatch(r"cake_kda_train_[0-9a-f]{20}", record["library"]), name
        launch = record["launch"]
        assert set(launch) >= {
            "block",
            "cluster",
            "cooperative",
            "dynamic_smem_bytes",
            "use_pdl",
        }, name
        assert len(tuple(launch["block"])) == 3 and tuple(launch["block"])[1:] == (
            1,
            1,
        ), name
        assert tuple(launch["cluster"]) == (1, 1, 1), name
        assert launch["cooperative"] is False, name
        # the overlap pair's pass 2 launches with the programmatic-stream-serialization attribute, nothing else does
        assert launch["use_pdl"] is (name in PDL_RECORDS), name
        assert (
            isinstance(launch["dynamic_smem_bytes"], int)
            and launch["dynamic_smem_bytes"] >= 0
        ), name
        plan = record["arg_plan"]
        assert [list(pair) for pair in plan[-3:]] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ], name
        for kind, key in plan[:-3]:
            assert kind in ("buffer", "parameter"), (name, kind, key)
            assert key in record["logical"], (name, key)
            logical = record["logical"][key]
            assert logical in (tensors if kind == "buffer" else scalars), (
                name,
                key,
                logical,
            )
        for kind in _rule_kinds(record["select_rule"]):
            assert kind in SELECT_KINDS, (name, kind)
        assert record["grid_rule"]["kind"] in GRID_KINDS, name
        # the records sharing one library: the segmented forward recurrence's two passes, and the WY program on
        # the saved Akk rows (the backward's, launched by the saving forward on its overlap rows too)
        twins = [
            other
            for other, rec in cake_jit.MODULES.items()
            if rec["library"] == record["library"] and other != name
        ]
        assert twins == [] or {name, *twins} in (
            {"fwdh_v3", "fwdh_p3"},
            {"wy_bwd", "wy_fwd_rows"},
        ), name
    assert stages == set(cake_jit.STAGES)
    for stage in cake_jit.STAGES:
        names = cake_jit.records_of_stage(stage)
        kinds = [cake_jit.MODULES[n]["select_rule"]["kind"] for n in names]
        if len(names) == 1 and stage in OPTIONAL_STAGES:
            assert kinds[0] not in ("always", "otherwise"), stage
        elif len(names) == 1:
            assert kinds == ["always"], stage
        else:
            assert "always" not in kinds and kinds.count("otherwise") <= 1, stage
    for name in cake_jit.MODULES:
        assert cake_jit.launches(name) == (2 if name in TWO_PASS_RECORDS else 1), name
    # the segmented records read the registry's one head bound; the carry's two records split the forward's
    # overlap rows (the pair: the two device families, at least the wave threshold, at most the flag words)
    # from their complement (every backward path, the third family, below the threshold, above the bound)
    seg_rule = {"kind": "heads_at_most", "max": SEG_MAX_H}
    # the table stage: on the segmented rows up to the document bound the intra program writes the tables on the
    # forward and the replay (no table record) and the fused table program rebuilds them for a saved-set path; the
    # chunk-parallel table on the serial rows up to the bound, the one-CTA table above; the block table is a launch
    # of its own above only
    docs_rule = {"kind": "docs_at_most", "max": TABLE_V2_MAX_DOCS}
    above_rule = {"kind": "docs_above", "min": TABLE_V2_MAX_DOCS}
    saved_paths = {"kind": "path_in", "paths": ["light", "light_states", "tape"]}
    assert cake_jit.MODULES["table_v3"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, docs_rule, saved_paths],
    }
    assert cake_jit.MODULES["table_v2"]["select_rule"] == {
        "kind": "all_of",
        "rules": [{"kind": "heads_above", "min": SEG_MAX_H}, docs_rule],
    }
    assert cake_jit.MODULES["table"]["select_rule"] == above_rule
    assert cake_jit.MODULES["fwd_intra_v14"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, docs_rule],
    }
    assert cake_jit.MODULES["fwd_intra"]["select_rule"] == {
        "kind": "any_of",
        "rules": [{"kind": "heads_above", "min": SEG_MAX_H}, above_rule],
    }
    assert cake_jit.MODULES["seg_table"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, {"kind": "docs_above", "min": TABLE_V2_MAX_DOCS}],
    }

    # the pair's wave threshold is a leaf per device family (3.0 on sm_100a / sm_103a, 2.3 on sm_107a)
    def waves(kind: str) -> dict:
        return {
            "kind": "any_of",
            "rules": [
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
                        {"kind": kind, "waves": OV1_MIN_WAVES_BY_ARCH["sm_100a"]},
                    ],
                },
                {
                    "kind": "all_of",
                    "rules": [
                        {"kind": "arch_in", "arches": ["sm_107a"]},
                        {"kind": kind, "waves": OV1_MIN_WAVES_BY_ARCH["sm_107a"]},
                    ],
                },
            ],
        }

    assert OV1_MIN_WAVES_BY_ARCH["sm_103a"] == OV1_MIN_WAVES_BY_ARCH["sm_100a"]
    assert cake_jit.MODULES["seg_carry_ov"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
            waves("walk_ctas_at_least"),
            {"kind": "segments_heads_at_most", "max": OV1_MAX_SEGMENT_HEADS},
        ],
    }
    assert cake_jit.MODULES["seg_carry"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            {
                "kind": "any_of",
                "rules": [
                    {
                        "kind": "path_in",
                        "paths": ["light", "light_states", "tape", "replay"],
                    },
                    waves("walk_ctas_below"),
                    {"kind": "segments_heads_above", "min": OV1_MAX_SEGMENT_HEADS},
                ],
            },
        ],
    }
    # the saving forward's token-row twins: the same stage, grid and rule leaves as the internal-row forms but the path
    for internal, rows in (
        ("fwdh_wy", "fwdh_wy_rows"),
        ("fwdh_p3_o_ov", "fwdh_p3_o_ov_rows"),
        ("fwdh_p3_o_ov_id", "fwdh_p3_o_ov_id_rows"),
        ("fwdh_o_v2", "fwdh_o_v2_rows"),
    ):
        a, b = cake_jit.MODULES[internal], cake_jit.MODULES[rows]
        assert (a["stage"], a["grid_rule"]) == (b["stage"], b["grid_rule"]), rows
        assert a["library"] != b["library"]
        assert str(a["select_rule"]).replace("'fwd_nosave'", "'fwd'") == str(
            b["select_rule"]
        ), rows
    assert (
        cake_jit.MODULES["wy_fwd_rows"]["library"]
        == cake_jit.MODULES["wy_bwd"]["library"]
    )


def _expected_segments(
    n: int, h: int, sm: int, arch: str | None = None
) -> tuple[str, int, int]:
    if h <= SEG_MAX_H:
        p = min(SEG_P_MAX, max(1, -(-SEG_SLOT_FACTOR * sm // (n * h))))
        p = max(1, min(p, sm // h))  # one walk CTA per SM per document at most
        if n * h <= SEG_P_SMALL_MAX_NH:
            p = min(p, SEG_P_SMALL_BY_ARCH.get(arch, SEG_P_SMALL))
        return "seg", p, n * p
    return "serial", 1, 0


def _query_arch() -> str | None:
    """The architecture the byte queries take without an explicit one: the current CUDA device's when it is an
    exported target, else None (the architecture-independent few-chain segment count)."""
    if not torch.cuda.is_available():
        return None
    index = torch.cuda.current_device()
    return cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))


def _expected_records(
    nc_pad: int, n: int, h: int, sm: int, arch: str, path: str
) -> dict[str, str | None]:
    """The record of every stage of ``path`` on (``nc_pad``, ``n``, ``h``, ``sm``, ``arch``): the dispatch of the
    kernels' composition, spelled independently of the registry's rules."""
    seg = h <= SEG_MAX_H
    _route, _p, nseg = _expected_segments(n, h, sm, arch)
    v2 = n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
    few = n * h <= FEW_CHAINS
    fused_p3 = seg and n * h <= FUSED_P3_MAX_CHAINS
    fused_fwd = (
        not seg and h >= FUSED_FWD_MIN_H and (arch != "sm_107a" or h <= FUSED_FWD_MIN_H)
    )
    fused_bwd = not seg and not v2 and h >= FUSED_BWD_MIN_H
    # the fused gradient hosts dAqk on the serial rows from 16 heads and on every segmented row (whose adjoint
    # recurrence accumulates dv): no dav launch there
    davq = (h >= DAVQ_MIN_H) if not seg else True
    replay = path == "replay"
    fwd = path in ("fwd", "fwd_nosave")
    # the saving forward reads the saved token rows where a program for them ships
    save = path == "fwd"
    # the paired serial rows: the 32-column form below the fused threshold reads the saved token rows on the saving
    # forward (wy_fwd_rows + fwdh_o_v2_rows); the 64-column form keeps the internal-row pair on both paths
    paired = save and not seg and v2 and not fused_fwd
    walk = nseg * h
    prefetch = walk < DHU_PREFETCH_WAVES_BY_ARCH.get(arch, 0.0) * sm
    tma = walk < DHU_TMA_WAVES * sm or (h <= 4 and walk <= DHU_TMA_WAVES * sm)
    ov1 = (
        seg
        and fwd
        and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
        and walk <= OV1_MAX_SEGMENT_HEADS
    )
    # the chunk-parallel tables up to the document bound (the segmented route's fused form writes the block table
    # too), the one-CTA table above, where the segmented route launches the block table on its own; on the segmented
    # route up to the bound the intra program writes the tables itself on the forward and the replay (no table
    # launch; the fused table program serves the saved-set paths whose set carries none)
    small = n <= TABLE_V2_MAX_DOCS
    intra_tables = seg and small
    if intra_tables:
        table = "table_v3" if path in ("light", "light_states", "tape") else None
    else:
        table = "table_v2" if small else "table"
    out: dict[str, str | None] = {
        "table": table,
        "seg_table": "seg_table" if seg and not small else None,
        "fwd_intra": "fwd_intra_v14" if intra_tables else "fwd_intra",
        "wy_fwd": None
        if fused_fwd
        else ("wy_fwd_rows" if (ov1 and save) or paired else "wy_fwd"),
        "seg_carry": None if not seg else ("seg_carry_ov" if ov1 else "seg_carry"),
        "glm_intra": "glm_intra",
        "gate_epilogue": "gate_epilogue",
    }
    if path == "tape":
        out["wy"] = "wy_bwd"
    elif path == "light":
        out["wy"] = None if fused_bwd else "wy_bwd"
    elif path == "light_states":
        out["wy"] = None if fused_p3 else "wy_bwd"
    elif replay:
        out["wy"] = None if fused_bwd else "wy_bwd_replay"
    else:
        out["wy"] = None
    if fwd:
        if seg:
            out["fwdh"] = "fwdh_v3"
        elif fused_fwd:
            out["fwdh"] = "fwdh_wy_rows" if save else "fwdh_wy"
        elif v2:
            out["fwdh"] = "fwdh_o_v2_rows" if paired else "fwdh_o_v2"
        else:
            out["fwdh"] = "fwdh_o_v1"
    elif path in ("light", "replay"):
        if seg:
            out["fwdh"] = "fwdh_v3"
        elif fused_bwd:
            out["fwdh"] = "fwdh_wy_bwd_replay" if replay else "fwdh_wy_bwd"
        elif v2:
            # the vh 32 build of the 32-column recompute from 16 heads; on sm_107a at 16 heads only
            narrow_wide = h >= 16 and (arch != "sm_107a" or h <= 16)
            out["fwdh"] = "fwdh_v2h" if narrow_wide else "fwdh_v2"
        else:
            out["fwdh"] = "fwdh_v1_sm107" if arch == "sm_107a" else "fwdh_v1"
    else:
        out["fwdh"] = None
    if not seg or path == "tape":
        out["fwdh_p3"] = None
    elif fwd:
        if ov1:
            out["fwdh_p3"] = (
                "fwdh_p3_o_ov" if h <= OV1_ORDER_MAX_H else "fwdh_p3_o_ov_id"
            ) + ("_rows" if save else "")
        else:
            out["fwdh_p3"] = "fwdh_p3_o"
    elif path == "light_states":
        out["fwdh_p3"] = "fwdh_p3_wy" if fused_p3 else "fwdh_p3"
    else:
        out["fwdh_p3"] = "fwdh_p3"
    out["bwd_prep"] = "bwd_prep_g8_replay" if replay else "bwd_prep_g8"
    if davq:
        out["dav"] = None
    else:  # the serial route below 16 heads and the few-chain segmented rows: dAqk alone
        out["dav"] = "dav_serial_replay" if replay else "dav_serial"
    if seg and few:
        if replay:
            out["dhu"] = "dhu_v3s_f_replay"
        else:
            out["dhu"] = "dhu_v3s_f_wt" if path == "light_states" else "dhu_v3s_f"
    elif seg:
        # the dv-accumulating adjoint in the row's tape build (the prefetching form below the sm_103a bound); its
        # w^T form behind the fused saved-state pass 2
        base = "dhu_v3p_f" if prefetch else ("dhu_v3_f" if tma else "dhu_v3r_f")
        if replay:
            out["dhu"] = base + "_replay"
        else:
            out["dhu"] = base + ("_wt" if path == "light_states" and fused_p3 else "")
    elif v2:
        out["dhu"] = "dhu_v2t_replay" if replay else "dhu_v2t"
    elif fused_bwd and path in ("light", "replay"):
        out["dhu"] = "dhu_v1_wt_replay" if replay else "dhu_v1_wt"
    else:
        out["dhu"] = "dhu_v1_replay" if replay else "dhu_v1"
    nohint = davq and not seg and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
    out["dqkg"] = (
        ("dqkg_int" if replay else "dqkg")
        + ("_davq" if davq else "")
        + ("_nohint" if nohint else "")
    )
    return {stage: out[stage] for stage in PATH_STAGES[path]}


def _expected_grid(
    name: str, nc_pad: int, n: int, h: int, sm: int, nseg: int = 0, launch: int = 0
) -> tuple[int, int, int]:
    ctas = nc_pad * h
    if name in ("table", "seg_table"):
        return (1, 1, 1)
    if name in ("table_v2", "table_v3"):
        return (max(1, -(-nc_pad // 256)), 1, 1)
    if name in (
        "fwd_intra",
        "fwd_intra_v14",
        "bwd_prep_g8",
        "bwd_prep_g8_replay",
        "glm_intra",
        "gate_epilogue",
    ):
        return (nc_pad, h, 1)
    if name.startswith(("wy_", "dav")):  # persistent over the (pair, head) items
        return (max(1, min(nc_pad // 2 * h, sm)), 1, 1)
    if name == "fwdh_v3":
        return (2 * nseg * h, 1, 1)
    if name.startswith("fwdh_p3"):
        return (nseg * h, 1, 1)
    if name.startswith("dhu_v3s_f"):
        return (2 * nseg * h, 1, 1)
    if name.startswith(("dhu_v3_f", "dhu_v3p_f", "dhu_v3r_f")):
        return (nseg * h, 1, 1)
    if name.startswith("seg_carry"):  # two CTAs per (document, head) chain
        return (2 * n * h, 1, 1)
    if name.startswith("fwdh_") or name.startswith("dhu_"):
        narrow = name.startswith(("fwdh_o_v2", "fwdh_v2", "dhu_v2t"))
        return (n * h * (4 if narrow else 2), 1, 1)
    if name.startswith("dqkg"):
        return (max(1, min(ctas, sm)), 1, 1)
    raise AssertionError(name)


PLAN_GRID = [
    (nc_pad, n, h, sm)
    for nc_pad in (2, 4, 6, 32, 64, 128, 256, 4096)
    for n in (1, 3, 4, 7, 300)
    for h in (1, 2, 4, 8, 16, 32, 64)
    for sm in (132, 148, 160, 212)
]


def test_every_stage_selects_the_expected_record_and_grid() -> None:
    for nc_pad, n, h, sm in PLAN_GRID:
        assert cake_jit.segment_plan(n, h, sm) == _expected_segments(n, h, sm)
        for arch in ARCHES:
            route, segments, nseg = _expected_segments(n, h, sm, arch)
            assert cake_jit.segment_plan(n, h, sm, arch) == (route, segments, nseg)
            for path in PATHS:
                if path == "light_states" and route == "serial":
                    continue
                expected = _expected_records(nc_pad, n, h, sm, arch, path)
                for stage, want in expected.items():
                    name = cake_jit.select_record(
                        stage,
                        num_chunks_pad=nc_pad,
                        num_seqs=n,
                        num_heads=h,
                        sm_count=sm,
                        arch=arch,
                        path=path,
                    )
                    assert name == want, (stage, nc_pad, n, h, sm, arch, path)
                    if name is None:
                        assert stage in OPTIONAL_STAGES
                        continue
                    for launch in range(cake_jit.launches(name)):
                        grid = cake_jit.launch_grid(
                            name,
                            num_chunks_pad=nc_pad,
                            num_seqs=n,
                            num_heads=h,
                            sm_count=sm,
                            num_segments=nseg,
                            launch=launch,
                        )
                        assert grid == _expected_grid(
                            name, nc_pad, n, h, sm, nseg, launch
                        ), (name, nc_pad, n, h, sm, launch)


def test_the_saving_forward_reads_the_token_rows_where_a_program_ships() -> None:
    # The intra program writes the internal Aqk / Akk copies (write_int) iff a later program of the path reads them:
    # never on the saving forward's fused serial rows, overlap rows and paired serial rows (the 32-column form below
    # the fused threshold: their token-row records), always on the forward without a saved set and on the replay.
    for n in (1, 3, 4, 7, 300):
        for h in (1, 2, 4, 8, 16, 32, 64):
            for sm in (132, 148, 160, 212):
                for arch in ARCHES:
                    p = cake_backend.plan(4096, n, h, sm, arch)
                    records = _expected_records(p.NC_pad, n, h, sm, arch, "fwd")
                    token = {
                        name for name in records.values() if name in TOKEN_ROW_RECORDS
                    }
                    seg = h <= SEG_MAX_H
                    walk = p.NSEG * h
                    fused = (
                        not seg
                        and h >= FUSED_FWD_MIN_H
                        and (arch != "sm_107a" or h <= FUSED_FWD_MIN_H)
                    )
                    ov1 = (
                        seg
                        and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                        and walk <= OV1_MAX_SEGMENT_HEADS
                    )
                    paired = (
                        not seg
                        and not fused
                        and n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
                    )
                    assert bool(token) == (fused or ov1 or paired), (
                        n,
                        h,
                        sm,
                        arch,
                        token,
                    )
                    if fused:
                        assert token == {"fwdh_wy_rows"}
                    if ov1:
                        assert "wy_fwd_rows" in token and len(token) == 2
                    if paired:
                        assert token == {"wy_fwd_rows", "fwdh_o_v2_rows"}
                    assert p.internal_rows("fwd") is (not token), (n, h, sm, arch)
                    assert p.internal_rows("fwd_nosave") is True
                    assert p.internal_rows("replay") is True
                    assert not set(p.records["fwd_nosave"].values()) & set(
                        TOKEN_ROW_RECORDS
                    )
                    assert p.launch_count("fwd") == p.launch_count("fwd_nosave")
                    for stage, name in p.records["fwd"].items():
                        other = p.records["fwd_nosave"][stage]
                        if name in TOKEN_ROW_RECORDS:
                            assert other == INTERNAL_ROW_TWIN[name], (stage, name)
                        else:
                            assert other == name, (stage, name)


def test_select_record_checks_its_inputs() -> None:
    kwargs = dict(num_chunks_pad=4, num_seqs=1, num_heads=4, sm_count=148)
    with pytest.raises(ValueError, match="arch"):
        cake_jit.select_record("table", **kwargs, arch="sm_90a", path="fwd")
    with pytest.raises(ValueError, match="path"):
        cake_jit.select_record("table", **kwargs, arch="sm_100a", path="bwd")
    with pytest.raises(ValueError, match="unknown stage"):
        cake_jit.select_record("fwd_o", **kwargs, arch="sm_100a", path="fwd")
    # a stage no record of the path admits is not an optional stage: the registry refuses the question
    with pytest.raises(ValueError, match="no record admits"):
        cake_jit.select_record("dqkg", **kwargs, arch="sm_100a", path="fwd")


def test_segment_args_follow_the_proportional_rule() -> None:
    # at most four chains: a proportional block table and seg_arg = -num_docs; otherwise the uniform count
    for sm in (132, 148, 152, 160):
        assert cake_jit.segment_args(1, 4, sm) == (1, -1)
        assert cake_jit.segment_args(4, 1, sm) == (1, -4)
        assert cake_jit.segment_args(2, 2, sm) == (1, -2)
        _route, segments, _nseg = cake_jit.segment_plan(3, 2, sm)
        assert cake_jit.segment_args(3, 2, sm) == (0, segments)
        assert cake_jit.segment_args(1, 16, sm) == (0, 1)  # the serial route
    p = cake_backend.plan(1024, 4, 1, 148)
    assert (p.proportional, p.seg_arg) == (1, -4)
    p = cake_backend.plan(1024, 7, 8, 148)
    assert (p.proportional, p.seg_arg) == (0, p.P)


def test_segmented_grids_need_segments_and_a_known_launch() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        cake_jit.launch_grid(
            "fwdh_v3", num_chunks_pad=4, num_seqs=1, num_heads=4, sm_count=148
        )
    with pytest.raises(ValueError, match="launch 1 of 1"):
        cake_jit.launch_grid(
            "fwdh_v3",
            num_chunks_pad=4,
            num_seqs=1,
            num_heads=4,
            sm_count=148,
            num_segments=64,
            launch=1,
        )
    with pytest.raises(ValueError, match="launch 2 of 2"):
        cake_jit.launch_grid(
            "dhu_v3_f",
            num_chunks_pad=4,
            num_seqs=1,
            num_heads=4,
            sm_count=148,
            num_segments=64,
            launch=2,
        )
    with pytest.raises(ValueError, match="must all be >= 1"):
        cake_jit.segment_plan(0, 4, 148)


def test_few_chains_cap_the_segment_count() -> None:
    # Four documents at H = 1 (four chains): 38 segments per document (NSEG = 152)
    # instead of the clamp's 64, a 37-step carry chain instead of 63; eight chains
    # keep the clamp.  A host-plan constant: no record changes with it.  sm_100a
    # has a few-chain count of its own, 37 (SEGMENTS["small_segments_by_arch"]);
    # arch=None is the architecture-independent 38.
    for sm in (132, 148, 152, 160):
        assert cake_jit.segment_plan(4, 1, sm) == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_103a") == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_107a") == ("seg", 38, 152)
        assert cake_jit.segment_plan(4, 1, sm, "sm_100a") == ("seg", 37, 148)
        # at H = 4 the per-SM cap sm // 4 (33, 37, 38, 40) applies before the few-chains cap of 38 (37 on sm_100a)
        cap = sm // 4
        assert cake_jit.segment_plan(1, 4, sm) == ("seg", min(38, cap), min(38, cap))
        assert cake_jit.segment_plan(1, 4, sm, "sm_100a") == (
            "seg",
            min(37, cap),
            min(37, cap),
        )
        assert cake_jit.segment_plan(8, 1, sm) == ("seg", 64, 512)
        assert cake_jit.segment_plan(8, 1, sm, "sm_100a") == ("seg", 64, 512)
        assert cake_jit.segment_plan(2, 4, sm) == ("seg", cap, 2 * cap)
        for arch, small in (("sm_100a", 37), ("sm_103a", 38), ("sm_107a", 38)):
            p = cake_backend.plan(1024, 4, 1, sm, arch)
            assert (p.recur, p.P, p.NSEG) == ("seg", small, 4 * small)
            assert cake_backend.plan(1024, 8, 1, sm, arch).P == 64
        # plan(arch=None) is the first exported architecture's plan, segment count included
        assert cake_backend.plan(1024, 4, 1, sm).P == 37
    # below the clamp the cap is inert: four chains on a 32-SM device give 32
    assert cake_jit.segment_plan(4, 1, 32) == ("seg", 32, 128)
    assert cake_jit.segment_plan(4, 1, 32, "sm_100a") == ("seg", 32, 128)


def test_segment_count_is_capped_at_one_walk_cta_per_sm() -> None:
    # P * H <= sm_count: one document's pass-2 (and adjoint) walk CTAs fit one wave of the device and its first
    # pass one round at two CTAs per SM.  152 SMs: 38 segments at H = 4 (four documents: the clamp gives the
    # same), 19 at H = 8; at H = 2 the clamp of 64 binds first.
    for n in (1, 2, 3, 4):
        for h in (1, 2, 4, 8):
            for sm in (132, 148, 152, 160):
                route, p, nseg = cake_jit.segment_plan(n, h, sm)
                assert route == "seg" and p >= 1 and p * h <= sm and nseg == n * p
    assert cake_jit.segment_plan(3, 4, 152) == ("seg", 38, 114)
    assert cake_jit.segment_plan(1, 8, 152) == ("seg", 19, 19)
    assert cake_jit.segment_plan(3, 2, 152) == ("seg", 64, 192)
    assert cake_jit.segment_plan(4, 8, 148) == ("seg", 18, 72)


def test_pairs_grid_needs_an_even_chunk_bound() -> None:
    with pytest.raises(ValueError, match="not even"):
        cake_jit.launch_grid(
            "wy_bwd", num_chunks_pad=3, num_seqs=1, num_heads=1, sm_count=148
        )


def test_pair_kernels_walk_the_pairs_on_a_persistent_grid() -> None:
    # The WY and dA / dv programs: one CTA per SM at most, walking the (chunk pair, head) items; the pair count is
    # a plan scalar.
    for name in (
        "wy_fwd",
        "wy_bwd",
        "wy_bwd_replay",
        "dav_serial",
        "dav_serial_replay",
    ):
        record = cake_jit.MODULES[name]
        assert record["grid_rule"] == {"kind": "pairs_bounded", "ctas_per_sm": 1}
        assert ["parameter", "num_pairs"] in [list(pair) for pair in record["arg_plan"]]
        assert record["logical"]["num_pairs"] == "num_pairs"
        for nc_pad, h, sm in (
            (2, 1, 148),
            (64, 4, 148),
            (4096, 64, 132),
            (4096, 1, 160),
        ):
            grid = cake_jit.launch_grid(
                name, num_chunks_pad=nc_pad, num_seqs=1, num_heads=h, sm_count=sm
            )
            assert grid == (max(1, min(nc_pad // 2 * h, sm)), 1, 1)


def test_the_backward_reads_the_token_rows_without_chunk_layout_copies() -> None:
    # The fused gradient and the GLM intra kernel read q / k / v where they are (TMA boxes on the token rows), so
    # the backward prep ships its copy-free forms alone and the backward layouts hold no slot for such copies.
    for name in (
        "dqkg",
        "dqkg_int",
        "dqkg_davq",
        "dqkg_int_davq",
        "dqkg_davq_nohint",
        "dqkg_int_davq_nohint",
    ):
        logical = cake_jit.MODULES[name]["logical"]
        assert (logical["v_tma"], logical["k_tma"], logical["q_tma"]) == ("v", "k", "q")
    logical = cake_jit.MODULES["glm_intra"]["logical"]
    assert (logical["k_e"], logical["q_e"]) == ("k", "q")
    assert cake_jit.records_of_stage("bwd_prep") == (
        "bwd_prep_g8",
        "bwd_prep_g8_replay",
    )
    for name in ("bwd_prep_g8", "bwd_prep_g8_replay"):
        prep = cake_jit.MODULES[name]
        assert prep["select_rule"]["kind"] == "path_in"
        assert not {"ke_out", "qe_out", "v_int", "vb_out"} & set(prep["logical"])
    # the saved-set paths' prep copies nothing of dout either; the replay's keeps the internal copy
    assert "do_int" in cake_jit.MODULES["bwd_prep_g8_replay"]["logical"].values()
    assert "do_int" not in cake_jit.MODULES["bwd_prep_g8"]["logical"].values()
    assert not {"ke", "qe", "v_int", "vb"} & set(cake_jit.LOGICAL_TENSORS)
    for mode, slots in cake_backend.SLOTS.items():
        names = {name for occupants in slots for name, *_ in occupants}
        assert not names & {"ke", "qe", "v_int", "vb"}, mode
        assert ("akk_int" in names) == (mode.startswith(("fwd", "replay"))), mode


def test_the_fused_gradient_turns_its_l2_hints_off_on_the_large_sm107a_rows() -> None:
    # The fused gradient's L2 prefetch and evict-first hints cost about 6 % on the sm_107a rows from 16 heads, so
    # its dAqk-hosting forms ship a second build there with the hints off (the `_nohint` records: the same kernel,
    # launch and operands; the hint thresholds beyond any head count).  The registry dispatches them on sm_107a
    # from 16 heads alone; the segmented rows and the other architectures keep the hinted builds.
    for name in ("dqkg_davq_nohint", "dqkg_int_davq_nohint"):
        record = cake_jit.MODULES[name]
        hinted = cake_jit.MODULES[name[: -len("_nohint")]]
        assert record["stage"] == hinted["stage"] == "dqkg"
        assert record["library"] != hinted["library"]
        assert record["arches"] == hinted["arches"] == list(ARCHES)
        assert record["logical"] == hinted["logical"]
        assert record["arg_plan"] == hinted["arg_plan"]
        assert record["launch"] == hinted["launch"]
        assert record["grid_rule"] == hinted["grid_rule"]
    for path, base in (
        ("light", "dqkg"),
        ("light_states", "dqkg"),
        ("tape", "dqkg"),
        ("replay", "dqkg_int"),
    ):
        for arch in ARCHES:
            for h in (1, 4, 8, 12, 16, 24, 32, 64, 128):
                seg = h <= SEG_MAX_H
                davq = seg or h >= DAVQ_MIN_H
                nohint = (
                    davq and not seg and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
                )
                want = base + ("_davq" if davq else "") + ("_nohint" if nohint else "")
                got = cake_jit.select_record(
                    "dqkg",
                    num_chunks_pad=64,
                    num_seqs=1,
                    num_heads=h,
                    sm_count=148,
                    arch=arch,
                    path=path,
                )
                assert got == want, (path, arch, h)


def test_the_saved_set_paths_read_the_token_rows_and_the_replay_the_internal_rows() -> (
    None
):
    # Operand sources: the records of the saved-set paths bind the saved Aqk / Akk token rows and the user's
    # dout; their replay twins bind the arena's internal rows (aqk_int / akk_int / do_int).
    pairs = {
        "wy_bwd": "wy_bwd_replay",
        "fwdh_wy_bwd": "fwdh_wy_bwd_replay",
        "dav_serial": "dav_serial_replay",
        "dhu_v3_f": "dhu_v3_f_replay",
        "dhu_v3p_f": "dhu_v3p_f_replay",
        "dhu_v3r_f": "dhu_v3r_f_replay",
        "dhu_v2t": "dhu_v2t_replay",
        "dhu_v1": "dhu_v1_replay",
        "dhu_v1_wt": "dhu_v1_wt_replay",
        "dhu_v3s_f": "dhu_v3s_f_replay",
        "dqkg": "dqkg_int",
        "dqkg_davq": "dqkg_int_davq",
        "dqkg_davq_nohint": "dqkg_int_davq_nohint",
    }
    token = {"aqk_src", "akk_src", "dout"}
    internal = {"aqk_int", "akk_int", "do_int"}
    for saved_name, replay_name in pairs.items():
        saved_values = set(cake_jit.MODULES[saved_name]["logical"].values())
        replay_values = set(cake_jit.MODULES[replay_name]["logical"].values())
        assert saved_values & token and not saved_values & internal, saved_name
        assert replay_values & internal and not replay_values & token, replay_name
        assert (
            cake_jit.MODULES[saved_name]["library"]
            != cake_jit.MODULES[replay_name]["library"]
        )
    # the segmented adjoint recurrences of the many-chain rows read dout on the saved-set paths (their w^T forms too)
    for name in (
        "dhu_v3_f",
        "dhu_v3_f_wt",
        "dhu_v3p_f",
        "dhu_v3p_f_wt",
        "dhu_v3r_f",
        "dhu_v3r_f_wt",
    ):
        assert "dout" in cake_jit.MODULES[name]["logical"].values(), name
    # the saving forward's token-row twins bind the saved Aqk / Akk where the internal-row forms bind the arena's copies
    saved_rows = {"aqk_ext", "akk_ext"}
    for rows_name, internal_name in INTERNAL_ROW_TWIN.items():
        internal_values = set(cake_jit.MODULES[internal_name]["logical"].values())
        rows_values = set(cake_jit.MODULES[rows_name]["logical"].values())
        assert internal_values & internal and not internal_values & saved_rows, (
            internal_name
        )
        assert rows_values & saved_rows and not rows_values & internal, rows_name
        assert (
            cake_jit.MODULES[internal_name]["library"]
            != cake_jit.MODULES[rows_name]["library"]
        )
        assert set(cake_jit.MODULES[internal_name]["logical"]) == set(
            cake_jit.MODULES[rows_name]["logical"]
        ), rows_name  # the same kernel parameters
    # the transposed w tile: the serial fused recompute writes w_t and its adjoint reads it, the segmented
    # saved-state pass 2 writes w_tk and its adjoint reads it
    assert cake_jit.MODULES["fwdh_wy_bwd"]["logical"]["w_out"] == "w_t"
    assert cake_jit.MODULES["dhu_v1_wt"]["logical"]["w_tma"] == "w_t"
    assert cake_jit.MODULES["fwdh_p3_wy"]["logical"]["w_t_out"] == "w_tk"
    assert cake_jit.MODULES["dhu_v3s_f_wt"]["logical"]["w_tma"] == "w_tk"
    for name in ("fwdh_v1", "dhu_v1", "dhu_v3s_f", "fwdh_p3"):
        assert cake_jit.MODULES[name]["logical"]["w_tma"] == "w", name


def test_the_forward_wy_forms_the_gated_values_from_the_token_rows() -> None:
    # The WY programs read the user's v and beta through the chunk table and form the gated values in shared
    # memory: no prep writes them and no layout holds a slot for them.
    fwd, bwd = cake_jit.MODULES["wy_fwd"], cake_jit.MODULES["wy_bwd"]
    assert fwd["library"] != bwd["library"]
    assert (fwd["stage"], bwd["stage"]) == ("wy_fwd", "wy")
    assert fwd["grid_rule"] == bwd["grid_rule"]
    for record in (fwd, bwd):
        assert (record["logical"]["v_tma"], record["logical"]["beta"]) == ("v", "beta")
        assert {"chunk_bos", "chunk_len"} <= set(record["logical"].values())
    assert (
        fwd["logical"]["akk_tma"] == "akk_int"
        and bwd["logical"]["akk_tma"] == "akk_src"
    )
    # the forward intra programs write the prep's outputs themselves (no prep launch) and read no v either
    for name in ("fwd_intra", "fwd_intra_v14"):
        intra = set(cake_jit.MODULES[name]["logical"].values())
        assert not intra & {"v", "vb"}, name
        assert {"gk", "kb", "kg", "qg", "v_new", "write_prep"} <= intra, name
    # the segmented route's intra program computes the chunk layout from cu_seqlens and writes the five tables
    v14 = set(cake_jit.MODULES["fwd_intra_v14"]["logical"].values())
    tables = set(cake_backend.TABLE_NAMES)
    assert {
        "cu_seqlens",
        "num_seqs",
        "segments",
        "proportional",
    } <= v14 and tables <= v14
    assert "cu_seqlens" not in cake_jit.MODULES["fwd_intra"]["logical"].values()
    assert "fwd_prep" not in cake_jit.STAGES
    assert "wy_fwd" in cake_jit.STAGES_FWD and "wy_fwd" not in cake_jit.STAGES_BWD
    assert "wy" in cake_jit.STAGES_BWD and "wy" not in cake_jit.STAGES_FWD
    for mode, slots in cake_backend.SLOTS.items():
        names = {name for occupants in slots for name, *_ in occupants}
        assert "vb" not in names and "kb" in names, mode
    # the fused forward recurrence takes the WY operands itself: no wy_fwd record on its rows (the saving forward's
    # form from the saved Akk / Aqk token rows)
    fused = cake_jit.MODULES["fwdh_wy"]["logical"]
    assert {"akk_int", "kb", "v", "beta", "out"} <= set(fused.values())
    rows = cake_jit.MODULES["fwdh_wy_rows"]["logical"]
    assert {"akk_ext", "aqk_ext", "kb", "v", "beta", "out"} <= set(rows.values())
    assert not {"akk_int", "aqk_int"} & set(rows.values())
    # the saving forward's overlap rows launch the backward's saved-Akk WY program under the forward stage
    rows_wy = cake_jit.MODULES["wy_fwd_rows"]
    assert rows_wy["stage"] == "wy_fwd" and rows_wy["library"] == bwd["library"]
    assert (
        rows_wy["logical"][next(k for k, v in bwd["logical"].items() if v == "akk_src")]
        == "akk_ext"
    )


def test_the_serial_walk_orders_the_documents_and_the_carry_splits_the_chain() -> None:
    # The serial recurrences take the layout table's longest-first document order (seq_order, a table of every
    # call); seg_carry runs two CTAs per (document, head) chain and reads the block table's chain geometry.
    for name in (
        "fwdh_v1",
        "fwdh_v1_sm107",
        "fwdh_v2",
        "fwdh_v2h",
        "fwdh_o_v1",
        "fwdh_o_v2",
        "fwdh_o_v2_rows",
        "fwdh_wy",
        "fwdh_wy_rows",
        "fwdh_wy_bwd",
        "dhu_v2t",
    ):
        record = cake_jit.MODULES[name]
        assert record["logical"]["seq_order"] == "seq_order", name
        assert ["buffer", "seq_order"] in [list(pair) for pair in record["arg_plan"]]
    assert "seq_order" in cake_backend.TABLE_NAMES
    carry = cake_jit.MODULES["seg_carry"]
    assert carry["grid_rule"] == {"kind": "docs_heads", "ctas_per_chain": 2}
    assert carry["logical"]["chain_seg_start"] == "seg_table"
    for n, h, sm in ((1, 1, 148), (4, 4, 152), (7, 8, 132)):
        grid = cake_jit.launch_grid(
            "seg_carry", num_chunks_pad=4, num_seqs=n, num_heads=h, sm_count=sm
        )
        assert grid == (2 * n * h, 1, 1)
    # the segmented programs read the block table and the walk programs' segment scalar
    for name in (
        "fwdh_v3",
        "fwdh_p3",
        "fwdh_p3_o",
        "fwdh_p3_wy",
        "fwdh_p3_o_ov_rows",
        "dhu_v3_f",
        "dhu_v3p_f",
        "dhu_v3r_f_wt",
        "dhu_v3s_f",
    ):
        logical = cake_jit.MODULES[name]["logical"]
        assert logical["seq_chunk_start"] == "seg_table", name
        assert logical["segments"] == "seg_arg", name
    assert cake_jit.MODULES["seg_table"]["logical"]["proportional"] == "proportional"


def _expected_launch_count(
    records: dict[str, str | None], seg: bool, path: str, tables: bool
) -> int:
    count = 0
    for stage, name in records.items():
        if name is None:
            continue
        if stage in ("table", "seg_table"):
            count += int((not tables) or path in ("fwd", "fwd_nosave", "replay"))
        elif stage == "seg_carry":
            count += int(records.get("fwdh") is not None) + int(
                path not in ("fwd", "fwd_nosave")
            )
        elif name in TWO_PASS_RECORDS:
            count += 2
        else:
            count += 1
    return count


def test_plan_follows_the_chunk_bound_formulas() -> None:
    for lengths, heads in SHAPES:
        S, N = sum(lengths), len(lengths)
        for sm in (132, 148):
            for arch in ARCHES:
                p = cake_backend.plan(S, N, heads, sm, arch)
                nc_max = -(-S // BT) + N
                assert (p.NC_max, p.NC_pad, p.R) == (
                    nc_max,
                    nc_max + nc_max % 2,
                    64 * (nc_max + nc_max % 2),
                )
                assert (p.recur, p.P, p.NSEG) == _expected_segments(N, heads, sm, arch)
                seg = p.recur == "seg"
                assert set(p.records) == {
                    path for path in PATHS if seg or path != "light_states"
                }
                for path in p.records:
                    expected = _expected_records(p.NC_pad, N, heads, sm, arch, path)
                    assert p.records[path] == expected, (path, lengths, heads, sm, arch)
                    for stage in cake_jit.STAGES:
                        if stage not in expected:
                            with pytest.raises(ValueError, match="not a stage of"):
                                p.record(stage, path)
                            continue
                        if not p.dispatched(stage, path):
                            assert stage in OPTIONAL_STAGES
                            with pytest.raises(ValueError, match="not launched"):
                                p.record(stage, path)
                            continue
                        for launch in range(p.launches(stage, path)):
                            assert p.grid(stage, path, launch) == _expected_grid(
                                p.record(stage, path),
                                p.NC_pad,
                                N,
                                heads,
                                sm,
                                p.NSEG,
                                launch,
                            )
                    assert set(p.stages(path)) == {s for s, r in expected.items() if r}
                    for tables in (True, False):
                        assert p.launch_count(path, tables=tables) == (
                            _expected_launch_count(expected, seg, path, tables)
                        ), (path, tables)
                # These packs are far below the size rule (S < 384 * NSEG): the light forward saves no segment
                # states and the backward recomputes them.
                assert p.seg_save is False and p.light_path == "light"
                assert p.launch_count("bwd") == p.launch_count("light")
                if not seg:
                    with pytest.raises(ValueError, match="segmented route only"):
                        p.record("dhu", "light_states")
    assert cake_backend.plan(1, 1, 1, 1).NC_pad == 2
    for bad in ((0, 1, 1, 148), (1, 0, 1, 148), (1, 1, 0, 148), (1, 1, 1, 0)):
        with pytest.raises(ValueError):
            cake_backend.plan(*bad)
    with pytest.raises(ValueError, match="int32"):
        cake_backend.plan(64, 1, 2**31 // (D * BT), 148)
    with pytest.raises(ValueError, match="arch"):
        cake_backend.plan(64, 1, 1, 148, "sm_90a")


def test_launch_counts_of_the_paths() -> None:
    # 148 SMs.  Serial rows: 16 heads in the 32-column form (no dav: the fused gradient hosts dAqk; on the saving
    # forward the token-row pair wy_fwd_rows + fwdh_o_v2_rows, the same count), 64 heads in
    # the 64-column form (the fused recompute: no WY launch on the light and replay paths).  Segmented rows: four
    # documents at 2 heads (eight chains: the dv-accumulating adjoint, the fused saved-state pass 2) and one
    # document at 4 heads (four chains: the two-CTA adjoint); no dav launch on either (the fused gradient hosts
    # dAqk), no table launch on the forward and the replay (the intra program writes the chunk tables and the block
    # table itself).  The forward has no prep launch.
    counts = {}
    for n, h in ((1, 16), (4, 64), (4, 2), (1, 4)):
        p = cake_backend.plan(4096, n, h, 148)
        counts[(n, h)] = {path: p.launch_count(path) for path in p.records}
    # the saving forward and the forward without a saved set launch the same number of programs (the token-row
    # records replace the internal-row ones)
    assert counts[(1, 16)] == {
        "fwd": 4,
        "fwd_nosave": 4,
        "light": 7,
        "tape": 6,
        "replay": 9,
    }
    assert counts[(4, 64)] == {
        "fwd": 3,
        "fwd_nosave": 3,
        "light": 6,
        "tape": 6,
        "replay": 8,
    }
    assert counts[(4, 2)] == {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 11,
        "light_states": 8,
        "tape": 8,
        "replay": 12,
    }
    assert counts[(1, 4)] == {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 11,
        "light_states": 8,
        "tape": 8,
        "replay": 12,
    }
    # a saved set whose tables the caller dropped adds the table launch to the saved-set paths (the fused table
    # program); the forward and the replay select no table record and the table-writing intra program
    p = cake_backend.plan(4096, 4, 2, 148)
    assert p.launch_count("light", tables=False) == 12
    assert p.launch_count("tape", tables=False) == 9
    assert p.launch_count("replay") == p.launch_count("replay", tables=False) == 12
    for path in ("fwd", "fwd_nosave", "replay"):
        assert p.records[path]["table"] is None and not p.dispatched("table", path)
        assert p.records[path]["fwd_intra"] == "fwd_intra_v14"
    for path in ("light", "light_states", "tape"):
        assert p.records[path]["table"] == "table_v3"
    assert (
        cake_backend.plan(4096, 1, 16, 148).records["fwd"]["fwd_intra"] == "fwd_intra"
    )
    # above the layout-table document bound the segmented route launches the block table on its own
    wide = cake_backend.plan(4096, 300, 2, 148)
    assert wide.records["fwd"]["table"] == "table" and wide.dispatched(
        "seg_table", "fwd"
    )
    assert wide.launch_count("fwd") == wide.launch_count("fwd_nosave") == 7
    assert wide.launch_count("replay") == 14
    # the backward from this plan's own light set: the states-saved path above the size rule
    assert p.launch_count("bwd") == p.launch_count("light") == 11
    tall = cake_backend.plan(384 * p.NSEG, 4, 2, 148)
    assert tall.light_path == "light_states" and tall.NSEG == p.NSEG
    assert tall.launch_count("bwd") == 8 and tall.launch_count("bwd", tables=False) == 9
    with pytest.raises(ValueError, match="path must be one of"):
        p.launch_count("sideways")


def test_seg_state_rule_is_a_pure_function_of_the_scalars() -> None:
    rule, k = cake_backend.seg_states_saved, cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG
    assert k == 384
    assert cake_backend.SAVED_VERSION == 3 and cake_backend.SAVED_VERSION_TAPE == 2
    for nseg in (1, 64, 153, 256):
        assert rule(k * nseg, nseg) is True and rule(k * nseg - 1, nseg) is False
    assert rule(10**9, 0) is False  # the serial route has no segment states
    for n, h, sm in ((1, 1, 148), (4, 4, 152), (3, 8, 148), (4, 1, 152)):
        nseg = cake_backend.plan(1, n, h, sm).NSEG  # independent of S
        assert (  # a plan without an architecture is the sm_100a plan
            nseg
            == cake_backend.plan(k * nseg, n, h, sm).NSEG
            == _expected_segments(n, h, sm, cake_jit.ARCHES[0])[2]
        )
        assert cake_backend.plan(k * nseg, n, h, sm).seg_save is True
        assert cake_backend.plan(k * nseg - 1, n, h, sm).seg_save is False
    assert cake_backend.plan(10**6, 1, 16, 148).seg_save is False  # the serial route


def test_byte_queries_follow_the_arena_layout() -> None:
    GiB = 2**30
    rows = SHAPES + [((239718 // 3,) * 3, 4), ((261393 // 4,) * 4, 64)]
    for (lengths, heads), arch in itertools.product(rows, ARCHES):
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148, arch)
        q = {"sm_count": 148, "arch": arch}  # the queries of this architecture
        assert tuple(cake_backend.MODES) == (
            "fwd",
            "fwd_tape",
            "bwd",
            "bwd_tape",
            "replay",
            "replay_tape",
        )
        for mode in cake_backend.MODES:
            regions = p.layout(mode)
            assert all(r.offset % cake_backend.ALIGN == 0 for r in regions), mode
            assert p.workspace_bytes(mode) == max(r.offset + r.nbytes for r in regions)
            names = {r.name for r in regions}
            seg_names = {"seg_b", "seg_m", "hseg"} | (
                {"dseg", "dhseg"} if not mode.startswith("fwd") else set()
            )
            assert (seg_names <= names) if p.recur == "seg" else not (seg_names & names)
            assert ("akk_int" in names) == (not mode.startswith("bwd")), mode
        # the saved-set paths' layouts drop the akk_int buffer: never larger than the replay's
        assert p.workspace_bytes("bwd") <= p.workspace_bytes("replay")
        assert p.workspace_bytes("bwd_tape") <= p.workspace_bytes("replay_tape")
        for backward in (False, True):
            for save_mode in cake_backend.SAVE_MODES:
                tape = save_mode == "tape"
                if backward:
                    want = max(
                        p.workspace_bytes("bwd_tape" if tape else "bwd"),
                        p.workspace_bytes("replay_tape" if tape else "replay"),
                    )
                else:
                    want = p.workspace_bytes("fwd_tape" if tape else "fwd")
                assert (
                    cake_backend.workspace_bytes(
                        S,
                        N,
                        heads,
                        backward=backward,
                        save_mode=save_mode,
                        **q,
                    )
                    == want
                    == p.arena_bytes(backward=backward, save_mode=save_mode)
                )
        assert cake_backend.workspace_bytes(
            S, N, heads, backward=False, **q
        ) <= cake_backend.workspace_bytes(S, N, heads, **q)
        assert (
            cake_backend.saved_bytes(S, heads)
            == 2 * S * heads * BT * 2
            == 256 * S * heads
        )
        tables = cake_backend.tables_bytes(S, N, heads, **q)
        assert tables == p.table_words() * 4 and tables % cake_backend.ALIGN == 0
        layout = p.table_layout()
        assert [name for name, _off, _words in layout] == list(
            cake_backend.TABLE_NAMES[: 5 if p.recur == "seg" else 4]
        )
        if p.recur == "seg":
            assert layout[-1][2] == p.NSEG + 2 * N + 1
        # light (version 3): the segment-boundary states (three FP32 state blocks) join the set when the size
        # rule holds: of these rows only the large segmented one (H = 4), not the small packs or the serial H = 64
        states = (
            3 * p.NSEG * heads * D * D * 4
            if cake_backend.seg_states_saved(S, p.NSEG)
            else 0
        )
        assert p.seg_save == (S >= 384 * p.NSEG > 0)
        assert (states > 0) == (S > 200_000 and heads == 4)
        assert (
            cake_backend.saved_bytes(S, heads, N, **q)
            == 256 * S * heads + tables + states
        )
        tape_bytes = cake_backend.saved_bytes(S, heads, N, save_mode="tape", **q)
        assert tape_bytes == (
            256 * S * heads
            + tables
            + p.NC_pad * heads * D * D * 2
            + p.R * heads * D * 2
            + 2 * p.NSEG * heads * D * D * 4
        )
        assert cake_backend.output_bytes(S, heads, backward=False) == S * heads * D * 2
        assert cake_backend.output_bytes(S, heads, backward=True) == S * heads * (
            3 * D * 2 + 2 + D * 4
        )
        report = cake_backend.memory_report(S, N, heads, **q)
        assert (report["recur"], report["segments"], report["num_segments"]) == (
            p.recur,
            p.P,
            p.NSEG,
        )
        assert report["arena_bwd"] == max(
            p.workspace_bytes("bwd"), p.workspace_bytes("replay")
        )
        assert report["arena_bwd_layout_saved_path"] == p.workspace_bytes("bwd")
        assert report["arena_bwd_layout_replay_path"] == p.workspace_bytes("replay")
        assert report["saved"] == 256 * S * heads + tables + states
        assert report["seg_states_saved"] == (states > 0)
        assert report["saved_seg_state_bytes"] == states
        tape_report = cake_backend.memory_report(S, N, heads, "tape", **q)
        assert tape_report["saved"] == tape_bytes
        assert tape_report["saved_seg_state_bytes"] == 2 * p.NSEG * heads * D * D * 4
        assert report["bwd_total"] == report["arena_bwd"] + report["out_bwd"]
        assert report["arch"] == arch
        # without an architecture the queries take the current device's (None without a device): the bytes of
        # the explicit query of that architecture
        host = _query_arch()
        assert cake_backend.workspace_bytes(S, N, heads, sm_count=148) == (
            cake_backend.workspace_bytes(S, N, heads, sm_count=148, arch=host)
        )
        assert cake_backend.memory_report(S, N, heads, sm_count=148)["arch"] == host
        if (
            S > 200_000
        ):  # the largest contract rows fit one 283 GiB device with room for two arenas (parity)
            assert report["arena_bwd"] < 85 * GiB
    with pytest.raises(ValueError, match="needs num_docs"):
        cake_backend.saved_bytes(64, 1, save_mode="tape")
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.saved_bytes(64, 1, save_mode="full")
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.workspace_bytes(64, 1, 1, save_mode="full", sm_count=148)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._layout(128, 1, 2, 0, "sideways")
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend.lifetime_table("sideways")


def test_lifetime_table_lists_every_occupant() -> None:
    for mode, slots in cake_backend.SLOTS.items():
        rows = cake_backend.lifetime_table(mode)
        assert len(rows) == sum(len(occupants) for occupants in slots)
        assert {(row["slot"], row["buffer"]) for row in rows} == {
            (slot, name)
            for slot, occupants in enumerate(slots)
            for name, *_ in occupants
        }
        for row in rows:
            assert row["first_write"] <= row["last_read"]
            assert row["unit"] in ("R*H", "seg_block")


# --------------------------------------------------------------------------- the lower-memory arena option (no device)

# bytes per internal token-head of the token kinds (a tape entry of 32768 B per chunk-head over the chunk's 64 rows)
KIND_BYTES_PER_TH = {
    "F32": 512,
    "B16": 256,
    "A16": 128,
    "A32": 256,
    "V32": 4,
    "TAPE": 512,
}
PASS_NAMES = frozenset(name for name, *_ in cake_backend._PASS_OCCUPANTS)
SEG_NAMES = frozenset(("seg_b", "seg_m", "hseg", "dseg", "dhseg"))
# the tail stages' operands the tail split hands over per range: offset views of the whole-sequence buffers, the
# range's table slices and count, the per-pass buffers; everything else they bind is addressed through the
# absolute chunk_bos (the token tensors) or is a scalar of the plan
TAIL_OFFSET_VIEWS = frozenset(("gk", "v_new", "dv2", "beta_int", "h2d", "dh2d"))
TAIL_TABLES = frozenset(
    ("chunk_bos", "chunk_len", "seq_chunk_start", "num_seqs", "num_chunks_pad")
)
TAIL_TOKEN_ROWS = frozenset(
    (
        "q",
        "k",
        "v",
        "g",
        "beta",
        "dout",
        "aqk_src",
        "akk_src",
        "dq",
        "dk",
        "dv",
        "dg",
        "dbeta",
        "cu_seqlens",
    )
)
TAIL_SCALARS = frozenset(
    (
        "scale",
        "num_heads",
        "group",
        "num_segments",
        "segments",
        "seg_arg",
        "proportional",
        "num_pairs",
    )
)
# buffers a tail kernel must not bind whole (their rows would be the whole sequence's, not the range's)
TAIL_FORBIDDEN = frozenset(
    (
        "h",
        "dh",
        "seq_order",
        "seg_table",
        "w",
        "w_t",
        "w_tk",
        "u",
        "kb",
        "qg",
        "kg",
        "akk_int",
        "aqk_int",
        "do_int",
        "dv1",
    )
)


# every arena buffer any layout table declares, plus the derived views the host hands over
ARENA_BUFFER_NAMES = frozenset(
    name for slots in cake_backend.SLOTS.values() for slot in slots for name, *_ in slot
) | frozenset(("seg_state", "w_t", "w_tk"))


def _slot_bytes(slots, R: int, H: int, NSEG: int, R_b: int) -> int:
    """An independent spelling of the layout's packing: per slot the farthest byte any occupant reaches (its
    sub-offset in the unit of its kind plus its extent), rounded up to the alignment, summed over the slots."""
    block = NSEG * H * D * D * 4
    total = 0
    for occupants in slots:
        end = 0
        for _name, kind, sub, _life in occupants:
            if kind in cake_backend.SEG_KINDS:
                if NSEG == 0:
                    continue
                unit = nbytes = block
            elif kind in cake_backend.PASS_KINDS:
                unit = R_b * H
                nbytes = unit * KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
            else:
                unit = R * H
                nbytes = unit * KIND_BYTES_PER_TH[kind]
            end = max(end, sub * unit + nbytes)
        total += -(-end // cake_backend.ALIGN) * cake_backend.ALIGN
    return total


def _expected_lean_bytes(tape: bool, t: int, R: int, H: int, NC: int, NSEG: int) -> int:
    """The lean layout's bytes from the documented rule: one pass is the saved-set layout without the replay's
    dout copy; from two passes on the whole-sequence set (2820 B per internal token-head, tape 2052) plus the
    per-pass set packed first-fit, in order, into the dead ``qg`` / ``kg`` (512) and ``w`` (256) slots, the rest
    into an overflow slot shared with the segmented route's three state blocks."""
    block = NSEG * H * D * D * 4
    if t == 1:
        return (3340 if tape else 4108) * R * H + max(0, 3 * block - 256 * R * H)
    R_b = BT * -(-NC // t)
    capacity = {"qg": 512 * R * H, "w": 256 * R * H}
    used = {"qg": 0, "w": 0}
    overflow = 0
    for _name, kind, _sub, _life in cake_backend._PASS_OCCUPANTS:
        nbytes = R_b * H * KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
        for host in ("qg", "w"):
            if used[host] + nbytes <= capacity[host]:
                used[host] += nbytes
                break
        else:
            overflow += nbytes
    extra = max(overflow, 3 * block) if (overflow or NSEG) else 0
    return (2052 if tape else 2820) * R * H + extra


def _lean_rows():
    return SHAPES + [
        ((239718 // 3,) * 3, 4),
        ((261393 // 4,) * 4, 64),
        ((2000,) * 300, 1),
    ]


def test_arena_option_validation() -> None:
    KdaArena = cake_backend.KdaArena
    assert KdaArena() == KdaArena(lean=False, tail_passes=1) and not KdaArena().lean
    assert KdaArena(tail_passes=2).lean  # the tail split lives in the lean layouts
    assert KdaArena(lean=True).tail_passes == 1
    for bad in (0, -1, 1.5, True, "2"):
        with pytest.raises(ValueError, match="tail_passes"):
            KdaArena(tail_passes=bad)
    off, lean = KdaArena(), KdaArena(lean=True, tail_passes=4)
    for base in cake_backend.MODES:
        assert off.mode(base) == base
    assert lean.mode("bwd") == "bwd_lean" and lean.mode("bwd_tape") == "bwd_tape_lean"
    assert lean.mode("fwd") == "fwd" and lean.mode("fwd_tape") == "fwd_tape"
    assert lean.mode("bwd_lean") == "bwd_lean"
    for base in ("replay", "replay_tape"):
        with pytest.raises(ValueError, match="lean=False"):
            lean.mode(base)
    assert tuple(cake_backend.LEAN_MODES) == ("bwd_lean", "bwd_tape_lean")
    assert not (set(cake_backend.LEAN_MODES) & set(cake_backend.MODES))
    assert cake_backend.PASS_KINDS == {"F32p": "F32", "A32p": "A32", "V32p": "V32"}
    with pytest.raises(ValueError, match="tail_passes"):
        cake_backend._slots_of("bwd", 2, 128, 1, 2, 0)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._slots_of("sideways", 1, 128, 1, 2, 0)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._layout(128, 1, 2, 0, "bwd_sideways", 2)
    for bad in ("full", "lean"):
        with pytest.raises(ValueError, match="save_mode"):
            cake_backend.workspace_bytes(
                64, 1, 1, save_mode=bad, sm_count=148, arena=lean
            )


def test_tail_ranges_cover_the_chunk_bound() -> None:
    ranges = cake_backend._tail_ranges
    assert ranges(10, 1) == ((0, 10),)
    assert ranges(10, 2) == ((0, 5), (5, 5))
    assert ranges(10, 3) == ((0, 4), (4, 4), (8, 2))
    assert ranges(10, 4) == ((0, 3), (3, 3), (6, 3), (9, 1))
    assert ranges(4, 3) == (
        (0, 2),
        (2, 2),
    )  # a clipped tail that is empty: fewer ranges than passes
    assert ranges(2, 8) == ((0, 1), (1, 1))
    for nc in (2, 4, 6, 8, 10, 64, 4086, 4088):
        for passes in (1, 2, 3, 4, 5, 8, 16):
            rs = ranges(nc, passes)
            length = -(-nc // passes)
            assert 1 <= len(rs) <= passes and all(0 < ln <= length for _c0, ln in rs)
            assert [c0 for c0, _ln in rs] == [i * length for i in range(len(rs))]
            assert sum(ln for _c0, ln in rs) == nc and rs[-1][0] + rs[-1][1] == nc


def test_lean_arena_refuses_the_rows_with_a_separate_dav_program() -> None:
    lean = cake_backend.KdaArena(lean=True)
    for arch in ARCHES:
        for H in range(1, 65):
            p = cake_backend.plan(2000, 3, H, 148, arch)
            dav = any(
                p.dispatched("dav", path)
                for path in p.records
                if path not in ("fwd", "fwd_nosave")
            )
            assert dav == (9 <= H <= 15), (arch, H)  # the serial rows below 16 heads
            if dav:
                with pytest.raises(ValueError, match="dav"):
                    p.check_arena(lean)
                with pytest.raises(ValueError, match="dav"):
                    p.arena_bytes(arena=lean)
                with pytest.raises(ValueError, match="dav"):
                    cake_backend.workspace_bytes(
                        2000, 3, H, sm_count=148, arch=arch, arena=lean
                    )
                with pytest.raises(ValueError, match="dav"):
                    cake_backend.memory_report(
                        2000, 3, H, sm_count=148, arch=arch, arena=lean
                    )
                # the forward layout is the same under every option: no refusal there
                assert cake_backend.workspace_bytes(
                    2000, 3, H, backward=False, sm_count=148, arch=arch, arena=lean
                ) == p.workspace_bytes("fwd")
            else:
                p.check_arena(lean)
                assert p.arena_bytes(arena=lean) == p.workspace_bytes("bwd_lean")
            p.check_arena(cake_backend.KdaArena())  # the default serves every row


def test_lean_layouts_follow_the_byte_formulas() -> None:
    for (lengths, heads), arch in itertools.product(_lean_rows(), ARCHES):
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148, arch)
        q = {"sm_count": 148, "arch": arch}
        R, H, NC, NSEG = p.R, p.H, p.NC_pad, p.NSEG
        block = NSEG * H * D * D * 4
        blocks_over = max(0, 3 * block - 256 * R * H)
        # the default layouts' constants: the saved-set layouts and the replay layouts (the arena of a default
        # backward workspace is the replay's)
        assert p.workspace_bytes("bwd") == 4360 * R * H + blocks_over
        assert p.workspace_bytes("replay") == 4488 * R * H + blocks_over
        assert p.workspace_bytes("bwd_tape") == 3592 * R * H + blocks_over
        assert p.workspace_bytes("replay_tape") == 3720 * R * H + blocks_over
        # the forward layouts (the README's constants); a lean backward workspace serves the forward: its layout
        # is the smaller one under every option
        assert p.workspace_bytes("fwd") == 2560 * R * H + max(
            0, 3 * block - 128 * R * H
        )
        assert p.workspace_bytes("fwd_tape") == 2048 * R * H + max(
            0, 3 * block - 128 * R * H
        )
        assert p.workspace_bytes("fwd") <= min(
            p.workspace_bytes("bwd_lean", t) for t in (1, 2, 3, 4, 8)
        )
        assert p.workspace_bytes("fwd_tape") <= min(
            p.workspace_bytes("bwd_tape_lean", t) for t in (1, 2, 3, 4, 8)
        )
        for tape in (False, True):
            mode = "bwd_tape_lean" if tape else "bwd_lean"
            save_mode = "tape" if tape else "light"
            whole = 2052 if tape else 2820
            default = cake_backend.memory_report(S, N, heads, save_mode, **q)
            assert default["lean"] is False and default["tail_passes"] == 1
            assert (
                default["arena_bwd_configured"]
                == default["arena_bwd_default"]
                == default["arena_bwd"]
                == p.workspace_bytes("replay_tape" if tape else "replay")
            )
            assert default["arena_fwd_configured"] == default["arena_fwd"]
            assert default["arena_floor"] == whole * R * H + 3 * block
            for t in (1, 2, 3, 4, 8):
                arena = cake_backend.KdaArena(lean=True, tail_passes=t)
                regions = p.layout(mode, t)
                R_b = BT * -(-NC // t)
                assert all(r.offset % cake_backend.ALIGN == 0 for r in regions), (
                    mode,
                    t,
                )
                want = p.workspace_bytes(mode, t)
                assert want == max(r.offset + r.nbytes for r in regions)
                slots = cake_backend._slots_of(mode, t, R, H, NC, NSEG)
                assert want == _slot_bytes(slots, R, H, NSEG, R_b), (mode, t)
                names = {r.name for r in regions}
                # no replay-only buffers (one pass keeps the saved-set layout's own aqk_int rows)
                assert not ({"do_int", "akk_int"} & names), (mode, t)
                assert (t == 1) or "aqk_int" not in names, (mode, t)
                for r in regions:
                    if (
                        r.name in PASS_NAMES and t > 1
                    ):  # one pass keeps the whole-sequence forms
                        assert r.shape[0] == R_b and r.lifetime in ((6, 7), (7, 8)), r
                    elif r.name in SEG_NAMES:
                        assert r.shape[0] == NSEG and r.nbytes == block, r
                    elif r.name in ("h", "dh"):
                        assert r.shape[0] == NC, r
                    else:
                        assert r.shape[0] == R, r
                seg = SEG_NAMES if NSEG else set()
                if t == 1:
                    # the saved-set layout without the replay's dout copy; kb under the dh tape
                    base = (
                        {
                            name
                            for slot in cake_backend.SLOTS[
                                "bwd_tape" if tape else "bwd"
                            ]
                            for name, *_ in slot
                        }
                        - {"do_int"}
                        - (set() if NSEG else SEG_NAMES)
                    )
                    assert names == base, (mode, t)
                    by_name = {r.name: r for r in regions}
                    assert by_name["kb"].slot == by_name["dh"].slot
                    assert by_name["kb"].offset == by_name["dh"].offset + 256 * R * H
                    assert by_name["kb"].lifetime == (1, 3)
                    assert want == (3340 if tape else 4108) * R * H + blocks_over
                else:
                    whole_names = {
                        "gk",
                        "dv2",
                        "dh",
                        "u",
                        "kb",
                        "qg",
                        "kg",
                        "w",
                        "beta_int",
                    } | ({"v_new_scratch"} if tape else {"v_new", "h"})
                    assert names == whole_names | PASS_NAMES | seg, (mode, t)
                    assert "dv1" not in names and "dAqk" in PASS_NAMES
                    # the documented packing rule, spelled independently: the per-pass set in order, first-fit
                    # into the dead qg / kg and w slots, the rest into the overflow slot shared with the blocks
                    assert want == _expected_lean_bytes(tape, t, R, H, NC, NSEG), (
                        mode,
                        t,
                    )
                    if t == 2:
                        assert want == whole * R * H + max(516 * R * H, 3 * block)
                    elif t >= 4 and NC % t == 0:  # from four passes on everything fits
                        assert want == whole * R * H + 3 * block
                    assert (
                        whole * R * H + 3 * block
                        <= want
                        <= whole * R * H + 3 * block + 2568 * R_b * H
                    )
                # the per-pass buffers never share bytes with one another or with a live whole-sequence buffer
                # at the tail (the layout raises on a lifetime clash; here: the packing keeps them in dead slots)
                for r in regions:
                    if r.name in PASS_NAMES:
                        for o in regions:
                            if (
                                o is not r
                                and o.slot == r.slot
                                and o.name not in PASS_NAMES
                            ):
                                assert (
                                    o.lifetime[1] < r.lifetime[0] or o.name in SEG_NAMES
                                ), (r, o)
                # the queries
                assert (
                    cake_backend.workspace_bytes(
                        S, N, heads, save_mode=save_mode, arena=arena, **q
                    )
                    == want
                    == p.arena_bytes(save_mode=save_mode, arena=arena)
                )
                assert cake_backend.workspace_bytes(
                    S, N, heads, backward=False, save_mode=save_mode, arena=arena, **q
                ) == p.workspace_bytes("fwd_tape" if tape else "fwd")
                report = cake_backend.memory_report(
                    S, N, heads, save_mode, arena=arena, **q
                )
                assert report["lean"] is True and report["tail_passes"] == t
                assert report["arena_bwd_configured"] == want
                assert (
                    report["arena_bwd_default"]
                    == report["arena_bwd"]
                    == default["arena_bwd"]
                )
                assert (
                    report["arena_fwd_configured"]
                    == report["arena_fwd"]
                    == default["arena_fwd"]
                )
                assert (
                    report["arena_floor"]
                    == whole * R * H + 3 * block
                    <= want
                    <= report["arena_bwd"]
                )
                assert report["bwd_total"] == want + report["out_bwd"]
                assert (
                    report["fwd_total_bwd_arena"]
                    == want + report["out_fwd"] + report["saved"]
                )
                assert report["replay_total"] == default["replay_total"]
                assert report["saved"] == default["saved"]
            assert p.workspace_bytes(mode, 4) <= p.workspace_bytes(mode, 1)
        # with the option off the queries are the default's
        off = cake_backend.KdaArena()
        assert cake_backend.workspace_bytes(
            S, N, heads, arena=off, **q
        ) == cake_backend.workspace_bytes(S, N, heads, **q)
        assert cake_backend.memory_report(
            S, N, heads, arena=off, **q
        ) == cake_backend.memory_report(S, N, heads, **q)


def test_lifetime_table_renders_the_lean_layouts() -> None:
    for mode in cake_backend.LEAN_MODES:
        tape = mode == "bwd_tape_lean"
        rows = cake_backend.lifetime_table(mode)
        single = cake_backend._lean_slots_single(tape)
        assert len(rows) == sum(len(occupants) for occupants in single)
        assert {(row["slot"], row["buffer"]) for row in rows} == {
            (slot, name)
            for slot, occupants in enumerate(single)
            for name, *_ in occupants
        }
        by_name = {row["buffer"]: row for row in rows}
        assert "do_int" not in by_name and by_name["db_int"]["unit"] == "R*H"
        assert by_name["kb"]["slot"] == by_name["dh"]["slot"]
        assert by_name["kb"]["sub_offset_per_th"] == 256
        assert (by_name["kb"]["first_write"], by_name["kb"]["last_read"]) == (1, 3)
        for row in rows:
            assert row["first_write"] <= row["last_read"]
            assert row["unit"] in ("R*H", "seg_block")
        with pytest.raises(ValueError, match="plan="):
            cake_backend.lifetime_table(mode, 2)
        for (lengths, heads), t in itertools.product(SHAPES[::4], (2, 3, 4)):
            p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, ARCHES[0])
            rows = cake_backend.lifetime_table(mode, t, plan=p)
            regions = {r.name: r for r in p.layout(mode, t)}
            assert {row["buffer"] for row in rows} == set(regions)
            for row in rows:
                r = regions[row["buffer"]]
                assert row["slot"] == r.slot
                assert (row["first_write"], row["last_read"]) == r.lifetime
                if row["buffer"] in PASS_NAMES:
                    assert row["unit"] == "R_b*H"
                    assert row["bytes_per_th"] * r.shape[0] * p.H == r.nbytes
                elif row["buffer"] in SEG_NAMES:
                    assert (
                        row["unit"] == "seg_block"
                        and row["bytes_per_seg_head"] * p.NSEG * p.H == r.nbytes
                    )
                else:
                    assert (
                        row["unit"] == "R*H"
                        and row["bytes_per_th"] * p.R * p.H == r.nbytes
                    )
    with pytest.raises(ValueError, match="tail_passes"):
        cake_backend.lifetime_table("bwd", 2)


def test_saved_set_path_records_bind_only_lean_layout_buffers() -> None:
    """Every record a plan selects on a saved-set path (``light``, ``light_states``, ``tape``) binds, of the arena
    buffers, only ones the lean layouts of two or more passes keep (``h`` / ``v_new`` come from the saved set or
    per-call buffers on the tape layouts; ``seg_state`` is the pair of state blocks): a lean workspace can hand
    every launch its operands."""
    for (lengths, heads), arch in itertools.product(_lean_rows(), ARCHES):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
        if any(
            p.dispatched("dav", path)
            for path in p.records
            if path not in ("fwd", "fwd_nosave")
        ):
            continue  # the rows a lean arena refuses
        for path in p.records:
            if path in ("fwd", "fwd_nosave", "replay"):
                continue
            mode = "bwd_tape_lean" if path == "tape" else "bwd_lean"
            kept = {r.name for r in p.layout(mode, 2)} | {"h", "v_new", "seg_state"}
            if "w" in kept:
                kept |= {"w_t", "w_tk"}
            for stage, name in p.records[path].items():
                if name is None:
                    continue
                logical = set(cake_jit.MODULES[name]["logical"].values())
                extra = (logical & ARENA_BUFFER_NAMES) - kept
                assert not extra, (
                    arch,
                    lengths,
                    heads,
                    path,
                    stage,
                    name,
                    sorted(extra),
                )


def test_tail_stages_bind_only_range_safe_operands() -> None:
    """The tail split hands every tail kernel the range's offset views, table slices and per-pass buffers; a
    kernel binding one of the whole-sequence buffers by its chunk index directly (``h`` instead of ``h2d``, the
    tables' ``seq_order``, the dead ``w`` / ``u`` / ``kb`` / ``qg`` / ``kg``) would read the wrong rows on a later
    range, so none of the records the plans select may bind one."""
    allowed = (
        TAIL_OFFSET_VIEWS | TAIL_TABLES | TAIL_TOKEN_ROWS | TAIL_SCALARS | PASS_NAMES
    )
    seen = set()
    for (lengths, heads), arch in itertools.product(_lean_rows(), ARCHES):
        p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
        for path in p.records:
            if path in ("fwd", "fwd_nosave", "replay"):
                continue
            for stage in ("dqkg", "glm_intra", "gate_epilogue"):
                name = p.record(stage, path)
                logical = set(cake_jit.MODULES[name]["logical"].values())
                assert not (logical & TAIL_FORBIDDEN), (
                    name,
                    sorted(logical & TAIL_FORBIDDEN),
                )
                assert logical <= allowed, (name, sorted(logical - allowed))
                seen.add(name)
    assert {"glm_intra", "gate_epilogue"} <= seen and any(
        n.startswith("dqkg") for n in seen
    )


def test_manifest_is_well_formed(manifest: dict) -> None:
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.experimental.cake_kda_train"
    arch_flags = manifest["arch_flags"]
    assert set(ARCHES) <= set(arch_flags)
    libraries = manifest["libraries"]
    names = [entry["name"] for entry in libraries]
    # The 59 programs of cake_jit.MODULES plus the KCP programs of cake_kcp_jit.MODULES (its test-only
    # TEST_MODULES live in build.test.json, never here).
    kcp_libraries = {record["library"] for record in cake_kcp_jit.MODULES.values()}
    assert len(names) == len(set(names)) == (59 + len(kcp_libraries)) * len(ARCHES)
    for entry in libraries:
        assert entry["arch"] in ARCHES
        assert entry["entry"] == "run"
        assert "--use_fast_math" in entry["nvcc_flags"]
        assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
        assert len(entry["sources"]) == 2
        for source in entry["sources"]:
            path = PACKAGE_DIR / source
            assert path.exists(), source
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"][source]
            ), source
            assert "by FlashInfer team" in "\n".join(
                path.read_text().splitlines()[:3]
            ), source
    expected = {
        cake_jit.library_name(name, arch)
        for name in cake_jit.MODULES
        for arch in ARCHES
    } | {
        cake_kcp_jit.library_name(name, arch)
        for name in cake_kcp_jit.MODULES
        for arch in ARCHES
    }
    assert set(names) == expected
    test_only = {
        cake_kcp_jit.test_library_name(name, arch)
        for name in cake_kcp_jit.TEST_MODULES
        for arch in ARCHES
    }
    assert not test_only & set(names), "test-only programs never enter build.json"


def test_library_name_refuses_an_undelivered_architecture() -> None:
    assert (
        cake_jit.library_name("table", "sm_100a")
        == f"{cake_jit.MODULES['table']['library']}_sm_100a"
    )
    with pytest.raises(RuntimeError, match="not delivered"):
        cake_jit.library_name("table", "sm_90a")


def test_run_module_names_a_missing_logical_argument() -> None:
    # The table kernel's first parameter is the packing's ``cu_seqlens``; an empty
    # value set is refused before any library is loaded.
    with pytest.raises(ValueError, match="needs logical argument 'cu_seqlens'"):
        cake_jit.run_module("table", {}, grid=(1, 1, 1), device_index=0)


# --------------------------------------------------------------------------- kernels (device)


def _expected_launches(
    p, path: str, *, saved_tables: bool = True, tail_passes: int = 1
) -> list[str]:
    """The launch order of ``path`` on plan ``p`` (the records by name), spelled from the composition's rules:
    the tables (on the forward and the replay where the plan dispatches a table program; on a saved-set path only
    when the set carries none, ``saved_tables=False``), the intra program (the forward: the prep folded in; the
    segmented route's writes the tables itself up to 256 documents), the backward prep (and the intra program on
    the replay: before the prep where it writes the tables), WY where
    a fused program does not form the representation, the recurrence (the segmented passes around the carry),
    dA / dv where the fused gradient does not host it, the adjoint recurrence, the fused gradient, GLM intra and
    the gate epilogue (the three tail stages once per chunk range of a lean workspace's ``tail_passes``)."""
    seg = p.recur == "seg"
    seq: list[str] = []
    # the table programs where the plan dispatches them (the segmented route's intra program writes the tables
    # itself up to 256 documents on the forward and the replay: no table launch there)
    tables_here = (not saved_tables) or path in ("fwd", "fwd_nosave", "replay")
    if tables_here and p.dispatched("table", path):
        seq.append(p.record("table", path))
        if seg and p.dispatched(
            "seg_table", path
        ):  # above 256 documents: the block table on its own
            seq.append("seg_table")
    if path in ("fwd", "fwd_nosave"):
        seq.append(p.record("fwd_intra", path))
        if p.dispatched("wy_fwd", path):
            seq.append(p.record("wy_fwd", path))
        seq.append(p.record("fwdh", path))
        if (
            seg
        ):  # the carry record of the row (the overlap pair's on its rows), then pass 2
            seq += [p.record("seg_carry", path), p.record("fwdh_p3", path)]
        return seq
    # the replay's intra program: before the prep where it writes the tables the prep reads, after it otherwise
    intra_first = path == "replay" and not p.dispatched("table", path)
    if intra_first:
        seq.append(p.record("fwd_intra", path))
    seq.append(p.record("bwd_prep", path))
    if path == "replay" and not intra_first:
        seq.append(p.record("fwd_intra", path))
    if p.dispatched("wy", path):
        seq.append(p.record("wy", path))
    if p.dispatched("fwdh", path):
        seq.append(p.record("fwdh", path))
        if seg:
            seq.append("seg_carry")
    if p.dispatched("fwdh_p3", path):
        seq.append(p.record("fwdh_p3", path))
    if p.dispatched("dav", path):
        seq.append(p.record("dav", path))
    dhu = p.record("dhu", path)
    seq += [dhu, "seg_carry", dhu] if seg else [dhu]
    tail = [p.record("dqkg", path), "glm_intra", "gate_epilogue"]
    seq += tail * len(cake_backend._tail_ranges(p.NC_pad, tail_passes))
    return seq


def _run_fwd_bwd(inputs, *, scale=SCALE, workspace=None, save=True, save_mode=None):
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, saved = cake_backend.chunk_kda_fwd(
        q,
        k,
        v,
        g,
        beta,
        cu,
        scale=scale,
        save_for_backward=save,
        workspace=workspace,
        save_mode=save_mode,
    )
    grads = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=saved, scale=scale, workspace=workspace
    )
    return out, dict(zip(("dq", "dk", "dv", "dbeta", "dg"), grads, strict=True)), saved


def _clone_all(out, grads):
    return out.clone(), {name: t.clone() for name, t in grads.items()}


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_forward_and_backward_match_the_exact_reference(
    device, manifest, shape
) -> None:
    lengths, heads = shape
    inputs = make_inputs(lengths, heads, device)
    out, grads, _ = _run_fwd_bwd(inputs)
    torch.cuda.synchronize(device)
    ref_out, ref_grads = kda_exact_with_grads(
        inputs["q"],
        inputs["k"],
        inputs["v"],
        inputs["g"],
        inputs["beta"],
        inputs["dout"],
        inputs["cu_seqlens"],
        SCALE,
    )
    candidates = {"out": out, **grads}
    references = {"out": ref_out, **ref_grads}
    failures = []
    for name in OUTPUT_NAMES:
        assert candidates[name].shape == references[name].shape, name
        gate = elementwise_gate(candidates[name], references[name], ATOL, RTOL)
        if not gate["pass"]:
            failures.append(
                (name, gate, error_metrics(candidates[name], references[name]))
            )
    assert not failures, failures


@pytest.mark.parametrize("shape", SHAPES[6:10], ids=_shape_id)
def test_replay_and_repeats_are_bitwise_identical(device, manifest, shape) -> None:
    lengths, heads = shape
    inputs = make_inputs(lengths, heads, device)
    out, grads, saved = _run_fwd_bwd(inputs)
    first_out, first_grads = _clone_all(out, grads)
    # The same inputs again (fixed-order reductions, no atomics).
    out, grads, _ = _run_fwd_bwd(inputs)
    assert torch.equal(out, first_out)
    for name, t in grads.items():
        assert torch.equal(t, first_grads[name]), name
    # Checkpoint replay (saved=None) recomputes the saved set and gives the same gradients.
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    replay = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=None, scale=SCALE
    )
    for name, t in zip(("dq", "dk", "dv", "dbeta", "dg"), replay, strict=True):
        assert torch.equal(t, first_grads[name]), name
    # A saved set without tables (the backward rebuilds them) and a saved set on another workspace.
    bare = KdaChunkSaved(
        saved.version,
        saved.Aqk,
        saved.Akk,
        None,
        saved.S,
        saved.H,
        saved.num_docs,
        saved.scale,
    )
    for other in (bare, saved):
        ws = KdaWorkspace.for_problem(sum(lengths), len(lengths), heads, device)
        grads2 = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=other, scale=SCALE, workspace=ws
        )
        for name, t in zip(("dq", "dk", "dv", "dbeta", "dg"), grads2, strict=True):
            assert torch.equal(t, first_grads[name]), name
    # The tape saved set (version 2: h, v_new and the segment transfers kept) gives the same gradients, from a
    # light workspace asked for the tape mode and from a tape workspace (its replay path too).
    out_t, grads_t, saved_t = _run_fwd_bwd(inputs, save_mode="tape")
    assert saved_t.version == cake_backend.SAVED_VERSION_TAPE and saved_t.h is not None
    assert (saved_t.seg_m is not None) == (heads <= SEG_MAX_H)
    assert torch.equal(out_t, first_out)
    for name, t in grads_t.items():
        assert torch.equal(t, first_grads[name]), name
    tape_ws = KdaWorkspace.for_problem(
        sum(lengths), len(lengths), heads, device, save_mode="tape"
    )
    assert tape_ws.bytes == cake_backend.workspace_bytes(
        sum(lengths),
        len(lengths),
        heads,
        save_mode="tape",
        sm_count=tape_ws.plan.sm_count,
    )
    out_w, grads_w, saved_w = _run_fwd_bwd(inputs, workspace=tape_ws)
    assert saved_w.version == cake_backend.SAVED_VERSION_TAPE and torch.equal(
        out_w, first_out
    )
    for name, t in grads_w.items():
        assert torch.equal(t, first_grads[name]), name
    replay_w = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=None, scale=SCALE, workspace=tape_ws
    )
    for name, t in zip(("dq", "dk", "dv", "dbeta", "dg"), replay_w, strict=True):
        assert torch.equal(t, first_grads[name]), name


def test_documents_do_not_leak_into_one_another(device, manifest) -> None:
    lengths, heads = (100, 640, 77), 2
    packed = make_inputs(lengths, heads, device)
    out, grads, _ = _run_fwd_bwd(packed)
    out, grads = _clone_all(out, grads)
    start = 0
    for length in lengths:
        stop = start + length
        single = {
            name: packed[name][start:stop].contiguous()
            for name in ("q", "k", "v", "g", "beta", "dout")
        }
        single["cu_seqlens"] = torch.tensor(
            [0, length], dtype=torch.int32, device=device
        )
        s_out, s_grads, _ = _run_fwd_bwd(single)
        assert elementwise_gate(out[start:stop], s_out, ATOL, RTOL)["pass"], (
            "out",
            start,
        )
        for name in s_grads:
            assert elementwise_gate(grads[name][start:stop], s_grads[name], ATOL, RTOL)[
                "pass"
            ], (name, start)
        start = stop


def test_autograd_and_checkpoint_give_the_explicit_gradients(device, manifest) -> None:
    lengths, heads = (63, 1, 65), 2
    inputs = make_inputs(lengths, heads, device)
    out, grads, _ = _run_fwd_bwd(inputs)
    out, grads = _clone_all(out, grads)
    cu, dout = inputs["cu_seqlens"], inputs["dout"]

    def leaves():
        return [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

    q, k, v, g, beta = leaves()
    o = cake_backend.chunk_kda(q, k, v, g, beta, cu)
    assert torch.equal(o, out)
    o.backward(dout)
    for leaf, name in zip(
        (q, k, v, g, beta), ("dq", "dk", "dv", "dg", "dbeta"), strict=True
    ):
        assert torch.equal(leaf.grad, grads[name]), name
    # torch.utils.checkpoint re-runs the forward in the backward; nothing is special-cased.
    from torch.utils.checkpoint import checkpoint

    q, k, v, g, beta = leaves()
    o = checkpoint(
        lambda *a: cake_backend.chunk_kda(*a, cu), q, k, v, g, beta, use_reentrant=False
    )
    o.backward(dout)
    for leaf, name in zip(
        (q, k, v, g, beta), ("dq", "dk", "dv", "dg", "dbeta"), strict=True
    ):
        assert torch.equal(leaf.grad, grads[name]), name
    # A non-contiguous upstream gradient is accepted by the autograd node (two heads: the transposed view of
    # a one-head tensor would still be contiguous).
    q, k, v, g, beta = leaves()
    o = cake_backend.chunk_kda(q, k, v, g, beta, cu)
    dout_nc = dout.transpose(0, 1).contiguous().transpose(0, 1)
    assert not dout_nc.is_contiguous()
    o.backward(dout_nc)
    assert torch.equal(v.grad, grads["dv"])


def test_batch_axis_is_optional(device, manifest) -> None:
    lengths, heads = (64, 64), 1
    flat = make_inputs(lengths, heads, device)
    batched = make_inputs(lengths, heads, device, batched=True)
    out, grads, _ = _run_fwd_bwd(flat)
    out, grads = _clone_all(out, grads)
    b_out, b_grads, _ = _run_fwd_bwd(batched)
    assert b_out.shape == (1, sum(lengths), heads, D)
    assert torch.equal(b_out[0], out)
    for name, t in b_grads.items():
        assert t.dim() == grads[name].dim() + 1 and torch.equal(t[0], grads[name]), name
    o = cake_backend.chunk_kda(
        *(batched[n] for n in ("q", "k", "v", "g", "beta")), batched["cu_seqlens"]
    )
    assert o.shape == b_out.shape


def test_workspace_reuse_and_byte_counts(device, manifest) -> None:
    lengths, heads = (129,), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    ws = KdaWorkspace.for_problem(S, N, heads, device)
    sm = ws.plan.sm_count
    # the byte queries without an architecture take the device's: with its SM count given or taken from it, the
    # same bytes as the workspace (a few-chain row: 37 segments per document on sm_100a, 38 elsewhere)
    expected_segments = 37 if ws.plan.arch == "sm_100a" else 38
    assert expected_segments == ws.plan.P
    assert (
        ws.bytes
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm)
        == cake_backend.workspace_bytes(S, N, heads)
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm, arch=ws.plan.arch)
        == ws.arena.numel()
    )
    assert cake_backend.saved_bytes(
        S, heads, N, sm_count=sm
    ) == cake_backend.saved_bytes(S, heads, N)
    assert cake_backend.memory_report(S, N, heads)["arch"] == ws.plan.arch
    fwd_only = KdaWorkspace.for_problem(S, N, heads, device, backward=False)
    assert fwd_only.bytes == cake_backend.workspace_bytes(
        S, N, heads, backward=False, sm_count=sm
    )
    out, grads, saved = _run_fwd_bwd(inputs, workspace=ws)
    # Outputs and the saved set are fresh per-call tensors; the arena holds the intermediates only.
    arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
    for t in (out, *grads.values(), saved.Aqk, saved.Akk, *saved.tables):
        assert not (arena_lo <= t.data_ptr() < arena_hi)
    assert len(saved.tables) == len(ws.plan.table_layout())
    assert saved.Aqk.numel() * saved.Aqk.element_size() * 2 == cake_backend.saved_bytes(
        S, heads
    )
    reference_out, reference_grads = _clone_all(out, grads)
    out2, grads2, _ = _run_fwd_bwd(inputs, workspace=ws)
    assert torch.equal(out2, reference_out)
    for name, t in grads2.items():
        assert torch.equal(t, reference_grads[name]), name
    out3, grads3, _ = _run_fwd_bwd(inputs)
    assert torch.equal(out3, reference_out)
    for name, t in grads3.items():
        assert torch.equal(t, reference_grads[name]), name
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    with pytest.raises(ValueError, match="backward=False"):
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=fwd_only
        )
    other = KdaWorkspace.for_problem(S + 1, N, heads, device)
    with pytest.raises(ValueError, match="allocated for"):
        cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=other)
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws, save_mode="full")
    assert set(ws.paths()) == {"fwd", "fwd_nosave", "light", "tape", "replay"} | (
        {"light_states"} if ws.plan.recur == "seg" else set()
    )
    assert ws.records() == {
        path: {stage: ws.plan.record(stage, path) for stage in ws.plan.stages(path)}
        for path in ws.paths()
    }
    assert fwd_only.paths() == ("fwd", "fwd_nosave")
    assert fwd_only.records() == {
        path: ws.records()[path] for path in ("fwd", "fwd_nosave")
    }
    ws.load()
    # A light workspace serves the tape forward (its layout is the smaller one); the tape backward needs a tape
    # workspace or a tape saved set.
    out_t, saved_t = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
    )
    assert saved_t.version == cake_backend.SAVED_VERSION_TAPE and torch.equal(
        out_t, reference_out
    )
    with pytest.raises(ValueError, match="with_plan"):
        ws.with_plan(other.plan)
    twin = ws.with_plan(ws.plan)
    assert (
        twin.arena.data_ptr() == ws.arena.data_ptr() and twin.records() == ws.records()
    )


def test_launch_sequences_follow_the_plan(device, manifest) -> None:
    # Seven rows: the segmented route with six chains, the 32-column serial form at 16 heads (the fused gradient
    # hosts dAqk), the 64-column serial form at 64 heads (the fused forward and the fused backward recompute), the
    # few-chain segmented rows (the two-CTA adjoint; the fused gradient hosts dAqk) and, on a 148-SM device, the
    # overlap pair's rows (the release-order pass 2 at 2 heads, the chain-major pass 2 at 8 heads, and three
    # documents at 4 heads: one walk CTA per SM per document, exactly the 3.0-wave threshold).
    for lengths, heads in (
        ((63, 1, 65), 2),
        ((129,), 16),
        ((63, 1, 65), 64),
        ((129,), 2),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((63, 1, 65), 4),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        launched: list[str] = []

        def recording(name, values, *, grid, device_index):
            launched.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = KdaWorkspace.for_problem(S, N, heads, device, launcher=recording)
        p = ws.plan
        assert p.recur == ("seg" if heads <= SEG_MAX_H else "serial")
        assert p.seg_save is False and p.light_path == "light"
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        out, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert launched == _expected_launches(p, "fwd")
        assert len(launched) == p.launch_count("fwd")
        launched.clear()
        # the forward without a saved set: the internal-row programs, the same launch count, the same output
        out_nosave, none = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_for_backward=False
        )
        assert none is None and torch.equal(out_nosave, out)
        assert launched == _expected_launches(p, "fwd_nosave")
        assert len(launched) == p.launch_count("fwd_nosave") == p.launch_count("fwd")
        assert (launched != _expected_launches(p, "fwd")) is (
            not p.internal_rows("fwd")
        )
        launched.clear()
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        assert launched == _expected_launches(p, "light")
        assert len(launched) == p.launch_count("light") == p.launch_count("bwd")
        launched.clear()
        bare = KdaChunkSaved(
            saved.version,
            saved.Aqk,
            saved.Akk,
            None,
            saved.S,
            saved.H,
            saved.num_docs,
            saved.scale,
        )
        cake_backend.chunk_kda_bwd(q, k, v, g, beta, cu, dout, saved=bare, workspace=ws)
        assert launched == _expected_launches(p, "light", saved_tables=False)
        assert len(launched) == p.launch_count("light", tables=False)
        launched.clear()
        cake_backend.chunk_kda_bwd(q, k, v, g, beta, cu, dout, saved=None, workspace=ws)
        assert launched == _expected_launches(p, "replay")
        assert len(launched) == p.launch_count("replay")
        launched.clear()
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        assert launched == _expected_launches(p, "fwd")
        launched.clear()
        tape_ws = KdaWorkspace.for_problem(
            S, N, heads, device, save_mode="tape", launcher=recording
        )
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_t, workspace=tape_ws
        )
        assert launched == _expected_launches(p, "tape")
        assert len(launched) == p.launch_count("tape")
        # the records of every path the workspace serves name the programs these sequences launched
        records = ws.records()
        assert set(records) == set(ws.paths())
        for path in ("fwd", "fwd_nosave", "light", "tape", "replay"):
            # the tables are the path's records too: launched when the saved set carries none
            expected = set(_expected_launches(p, path, saved_tables=False))
            assert expected == set(records[path].values()), path


def test_operand_checks(device, manifest) -> None:
    # Two heads: with one head the transposed view below would still be contiguous
    # (a size-1 axis has no stride of its own).
    lengths, heads = (65,), 2
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    with pytest.raises(ValueError, match="contiguous"):
        cake_backend.chunk_kda_fwd(
            q.transpose(0, 1).contiguous().transpose(0, 1), k, v, g, beta, cu
        )
    with pytest.raises(ValueError, match="expected torch.float32"):
        cake_backend.chunk_kda_fwd(q, k, v, g.to(torch.bfloat16), beta, cu)
    with pytest.raises(ValueError, match="expected shape"):
        cake_backend.chunk_kda_fwd(q, k, v, g, beta[:-1], cu)
    with pytest.raises(ValueError, match="batch axis"):
        cake_backend.chunk_kda_fwd(
            q.unsqueeze(0).expand(2, *q.shape), k, v, g, beta, cu
        )
    with pytest.raises(ValueError, match="cu_seqlens"):
        cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu.to(torch.int64))
    _, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu)
    assert saved.version == cake_backend.SAVED_VERSION == 3
    # version 1 (the light set without the segment states) is no longer accepted
    with pytest.raises(ValueError, match="version"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(1, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE),
        )
    # a version-3 set carries hseg and seg_state together, with the plan's shapes
    nseg = cake_backend.plan(
        65,
        1,
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    ).NSEG
    block = torch.zeros((nseg, heads, D, D), dtype=torch.float32, device=device)
    with pytest.raises(ValueError, match="together or neither"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(
                3, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE, hseg=block
            ),
        )
    with pytest.raises(ValueError, match="saved.seg_state"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(
                3,
                saved.Aqk,
                saved.Akk,
                None,
                65,
                heads,
                1,
                SCALE,
                hseg=block,
                seg_state=block,
            ),
        )
    with pytest.raises(ValueError, match="needs h and v_new"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(2, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE),
        )
    with pytest.raises(ValueError, match="saved.tables"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(
                3, saved.Aqk, saved.Akk, saved.tables[:2], 65, heads, 1, SCALE
            ),
        )
    with pytest.raises(ValueError, match="saved: for"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(3, saved.Aqk, saved.Akk, None, 64, heads, 1, SCALE),
        )
    out, _ = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, save_for_backward=False)
    assert out.shape == (65, heads, D)
    assert cake_backend.is_supported(device)


# The size rule's smallest instances: one document at H = 1 and two documents at H = 2 have at most four chains,
# so P = SEG_P_SMALL (37 on sm_100a) on any device with at least 38 SMs, NSEG = P * num_docs and the threshold is
# 384 * NSEG tokens (14592 and 29184; 14208 and 28416 on sm_100a).
STATE_SHAPES = [(1, 1), (2, 2)]


def _split(S: int, num_docs: int) -> tuple[int, ...]:
    base = S // num_docs
    return tuple(base + (1 if i < S % num_docs else 0) for i in range(num_docs))


def _retained_after_checkpointed_forward(fn, *leaves) -> int:
    """Bytes a forward of ``fn`` under non-reentrant ``torch.utils.checkpoint`` keeps allocated beyond its output:
    the measure of the activation-checkpointing report (0 expected: only the saved tensors go through the saved-
    tensor hooks, and anything else the node holds would be kept for every layer at once)."""
    from torch.utils.checkpoint import checkpoint

    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    out = checkpoint(fn, *leaves, use_reentrant=False)
    torch.cuda.synchronize()
    kept = torch.cuda.memory_allocated() - before - out.untyped_storage().nbytes()
    del out
    return kept


def test_a_checkpointed_forward_retains_no_workspace(device, manifest) -> None:
    """Three ways to run the autograd entry under activation checkpointing keep nothing beyond the output: the
    implicit workspace, a per-call workspace created inside the checkpointed function (held by the node only
    weakly) and a caller-owned workspace kept outside it.  The gradients through the checkpoint are the explicit
    ones in every case."""
    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    cu, dout = inputs["cu_seqlens"], inputs["dout"]
    out, grads, _ = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    assert plan.arena_bytes(backward=True) > 0
    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def implicit(q, k, v, g, beta):
        return cake_backend.chunk_kda(q, k, v, g, beta, cu)

    def per_call(q, k, v, g, beta):
        ws = KdaWorkspace(plan, device, backward=True)
        return cake_backend.chunk_kda(q, k, v, g, beta, cu, workspace=ws)

    shared = KdaWorkspace(plan, device, backward=True)

    def on_shared(q, k, v, g, beta):
        return cake_backend.chunk_kda(q, k, v, g, beta, cu, workspace=shared)

    for fn in (implicit, per_call, on_shared):
        warm = fn(
            *leaves
        )  # the programs are loaded and the allocator primed before the measurement
        assert torch.equal(warm, reference_out), fn.__name__
        del warm
        assert _retained_after_checkpointed_forward(fn, *leaves) == 0, fn.__name__
        for leaf in leaves:
            leaf.grad = None
        o = checkpoint(fn, *leaves, use_reentrant=False)
        o.backward(dout)
        for leaf, name in zip(leaves, ("dq", "dk", "dv", "dg", "dbeta"), strict=True):
            assert torch.equal(leaf.grad, reference_grads[name]), (fn.__name__, name)


def test_the_node_holds_the_workspace_weakly(device, manifest) -> None:
    """A caller-owned workspace serves the backward (its launcher records the backward's launches); a per-call
    workspace dropped after the forward is freed at once (the node holds no strong reference), and the backward
    then runs in a workspace of its own, with the same gradients."""
    import gc
    import weakref

    lengths, heads = (63, 1, 65), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    cu, dout = inputs["cu_seqlens"], inputs["dout"]
    out, grads, _ = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    launched: list[str] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    def leaves():
        return [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

    def check(leaf_list):
        for leaf, name in zip(
            leaf_list, ("dq", "dk", "dv", "dg", "dbeta"), strict=True
        ):
            assert torch.equal(leaf.grad, reference_grads[name]), name

    kept = KdaWorkspace(plan, device, backward=True, launcher=recording)
    q, k, v, g, beta = held = leaves()
    o = cake_backend.chunk_kda(q, k, v, g, beta, cu, workspace=kept)
    assert torch.equal(o, reference_out)
    assert launched == _expected_launches(plan, "fwd")
    launched.clear()
    o.backward(dout)
    assert launched == _expected_launches(plan, plan.light_path)
    check(held)
    launched.clear()
    dropped = KdaWorkspace(plan, device, backward=True, launcher=recording)
    alive = weakref.ref(dropped)
    q, k, v, g, beta = held = leaves()
    o = cake_backend.chunk_kda(q, k, v, g, beta, cu, workspace=dropped)
    assert torch.equal(o, reference_out)
    del dropped
    gc.collect()
    assert alive() is None  # the autograd node did not keep it
    launched.clear()
    o.backward(dout)
    assert (
        launched == []
    )  # the backward ran in a workspace of its own, not through the dropped one
    check(held)


def test_arena_content_never_reaches_a_result(device, manifest) -> None:
    """Write-before-read: every element a launch reads is written earlier in the same call, so an arena filled
    with NaN bytes gives the zero-filled arena's results bit for bit on every path (the forward with and without
    a saved set, the backward from the light set, the replay, and the tape layouts), and the arenas need no fill
    (``zero_init`` defaults to ``False``)."""
    cases = (
        ((129,), 2, "light"),
        ((100, 640, 77, 1000, 129), 2, "light"),
        ((63, 1, 65, 30), 2, "light"),
        ((20, 20, 20, 20, 20, 20, 20), 8, "light"),
        ((100, 640, 77, 1000, 129), 2, "tape"),
    )
    for lengths, heads, save_mode in cases:
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = KdaWorkspace(
            plan, device, backward=True, zero_init=True, save_mode=save_mode
        )
        assert zeroed.zero_init is True and not bool(zeroed.arena.any())
        poisoned = KdaWorkspace(plan, device, backward=True, save_mode=save_mode)
        assert poisoned.zero_init is False
        assert KdaWorkspace.for_problem(S, N, heads, device).zero_init is False

        def poison():
            poisoned.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        out, grads, saved = _run_fwd_bwd(inputs, workspace=zeroed)
        reference_out, reference_grads = _clone_all(out, grads)
        reference_aqk, reference_akk = saved.Aqk.clone(), saved.Akk.clone()
        replay = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=None, workspace=zeroed
        )
        reference_replay = [t.clone() for t in replay]
        nosave, _ = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, save_for_backward=False, workspace=zeroed
        )
        reference_nosave = nosave.clone()
        del out, grads, saved, replay, nosave
        case = (lengths, heads, save_mode)
        poison()
        out, saved = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=poisoned
        )
        assert torch.equal(out, reference_out), case
        assert torch.equal(saved.Aqk, reference_aqk) and torch.equal(
            saved.Akk, reference_akk
        ), case
        poison()
        grads = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=poisoned
        )
        for t, name in zip(grads, ("dq", "dk", "dv", "dbeta", "dg"), strict=True):
            assert torch.equal(t, reference_grads[name]), (case, name)
        poison()
        replay = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=None, workspace=poisoned
        )
        for t, ref in zip(replay, reference_replay, strict=True):
            assert torch.equal(t, ref), case
        poison()
        nosave, _ = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, save_for_backward=False, workspace=poisoned
        )
        assert torch.equal(nosave, reference_nosave), case


@pytest.mark.parametrize(("num_docs", "heads"), STATE_SHAPES, ids=["n1-h1", "n2-h2"])
def test_light_saved_set_carries_the_segment_states_above_the_size_rule(
    device, manifest, num_docs, heads
) -> None:
    from torch.utils.checkpoint import checkpoint

    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG  # independent of S
    assert nseg == SEG_P_SMALL_BY_ARCH.get(arch, SEG_P_SMALL) * num_docs
    threshold = cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    block = nseg * heads * D * D * 4

    def run(S: int, saves: bool) -> None:
        lengths = _split(S, num_docs)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        launched: list[str] = []

        def recording(name, values, *, grid, device_index):
            launched.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = KdaWorkspace.for_problem(S, num_docs, heads, device, launcher=recording)
        p = ws.plan
        assert p.recur == "seg" and nseg == p.NSEG and p.seg_save is saves
        assert cake_backend.seg_states_saved(S, nseg) is saves
        # (a) the saved set: the states present exactly at and above the threshold, the bytes as the query says
        out, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        # saving the states adds no launch
        assert launched == _expected_launches(p, "fwd")
        assert len(launched) == p.launch_count("fwd") == 5
        assert saved.version == cake_backend.SAVED_VERSION
        tables = p.table_words() * 4
        expected_bytes = 256 * S * heads + tables + (3 * block if saves else 0)
        assert (
            cake_backend.saved_bytes(S, heads, num_docs, sm_count=sm) == expected_bytes
        )
        report = cake_backend.memory_report(S, num_docs, heads, sm_count=sm)
        assert report["seg_states_saved"] is saves
        assert report["saved_seg_state_bytes"] == (3 * block if saves else 0)
        assert report["saved"] == expected_bytes
        if not saves:
            assert saved.hseg is None and saved.seg_state is None
        else:
            assert saved.hseg.shape == (nseg, heads, D, D)
            assert saved.seg_state.shape == (2, nseg, heads, D, D)
            assert saved.hseg.dtype == saved.seg_state.dtype == torch.float32
            arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
            for t in (saved.hseg, saved.seg_state):
                assert not (arena_lo <= t.data_ptr() < arena_hi)
            owned = (saved.Aqk, saved.Akk, saved.hseg, saved.seg_state)
            assert (
                sum(t.numel() * t.element_size() for t in owned) + tables
                == expected_bytes
            )
            states_before = (saved.hseg.clone(), saved.seg_state.clone())
        # the backward from the set: the recurrence as pass 2 alone with the states, the full recompute without
        launched.clear()
        grads = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        grads = [t.clone() for t in grads]
        # the few-chain rows: with the states the fused pass 2 forms the WY representation (no WY launch) and
        # the adjoint recurrence reads its transposed w tile; without them the full recompute
        path = "light_states" if saves else "light"
        assert p.light_path == path
        assert launched == _expected_launches(p, path)
        assert len(launched) == p.launch_count("bwd") == (8 if saves else 11)
        assert launched[1:3] == (
            ["fwdh_p3_wy", "dhu_v3s_f_wt"] if saves else ["wy_bwd", "fwdh_v3"]
        )
        assert ("dhu_v3s_f_wt" if saves else "dhu_v3s_f") in launched
        # (b) the gradients from the saved states equal the checkpoint replay's bitwise, and those from the same
        # set stripped of its states (the recompute path) and from a tape-mode workspace
        replay = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=None, workspace=ws
        )
        stripped = KdaChunkSaved(
            saved.version,
            saved.Aqk,
            saved.Akk,
            saved.tables,
            saved.S,
            saved.H,
            saved.num_docs,
            saved.scale,
        )
        from_stripped = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=stripped, workspace=ws
        )
        tape_ws = KdaWorkspace.for_problem(S, num_docs, heads, device, save_mode="tape")
        from_tape_ws = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
        )
        for other in (replay, from_stripped, from_tape_ws):
            for mine, theirs in zip(grads, other, strict=True):
                assert torch.equal(mine, theirs)
        if saves:  # the backward writes nothing into the saved set
            assert torch.equal(saved.hseg, states_before[0])
            assert torch.equal(saved.seg_state, states_before[1])
        # the tape set is unchanged by the rule: version 2 with seg_m, without the version-3 fields
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        assert (
            saved_t.seg_m is not None
            and saved_t.hseg is None
            and saved_t.seg_state is None
        )
        # (c) the autograd node packs and unpacks the states (or their absence); the checkpoint path re-runs the
        # forward and gives the same gradients
        expected = (grads[0], grads[1], grads[2], grads[4], grads[3])
        for wrap in (False, True):
            leaves = [
                inputs[n].detach().clone().requires_grad_(True)
                for n in ("q", "k", "v", "g", "beta")
            ]
            if wrap:
                o = checkpoint(
                    lambda *a: cake_backend.chunk_kda(*a, cu, workspace=ws),
                    *leaves,
                    use_reentrant=False,
                )
            else:
                o = cake_backend.chunk_kda(*leaves, cu, workspace=ws)
            assert torch.equal(o, out)
            o.backward(dout)
            for leaf, want in zip(leaves, expected, strict=True):
                assert torch.equal(leaf.grad, want)

    run(threshold - 1, False)  # the longest pack that recomputes the states
    run(threshold, True)  # the shortest pack that saves them


def test_plan_limits_cover_the_kernels_int32_offsets() -> None:
    # The [R, H, 64] intra-chunk buffers are addressed with 32-bit row-head offsets: R * H * 64 < 2**31 over the
    # padded rows R = 64 * NC_pad.
    assert cake_backend.plan(2**25 - 192, 1, 1, 148).R * 1 * BT < 2**31
    with pytest.raises(ValueError, match="row-head"):
        cake_backend.plan(2**25 - 128, 1, 1, 148)
    assert cake_backend.plan(2**19 - 192, 1, 64, 148).R * 64 * BT < 2**31
    with pytest.raises(ValueError, match="row-head"):
        cake_backend.plan(2**19 - 128, 1, 64, 148)
    # The layout table finds a chunk's document with a 16-step search: at most MAX_DOCS documents.
    assert (
        cake_backend.plan(70000, cake_backend.MAX_DOCS, 1, 148).N
        == cake_backend.MAX_DOCS
    )
    with pytest.raises(ValueError, match="document search"):
        cake_backend.plan(70000, cake_backend.MAX_DOCS + 1, 1, 148)


def test_tape_workspace_serves_light_calls(device, manifest) -> None:
    # A saved set can go to a backward on any workspace of the same problem: a tape-mode arena (sized for the tape
    # layouts) recomputes h / v_new into per-call buffers for a version-1 saved set, for the replay and for a
    # forward that saves nothing, as its own forward does; the results are those of a light workspace.
    lengths, heads = (63, 1, 65), 4
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, grads, _ = _run_fwd_bwd(inputs)
    out, grads = _clone_all(out, grads)
    S, N = sum(lengths), len(lengths)
    ws = KdaWorkspace.for_problem(S, N, heads, device, save_mode="tape")
    o, saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, workspace=ws, save_mode="light"
    )
    assert saved.version == cake_backend.SAVED_VERSION
    assert torch.equal(o, out)
    for saved_set in (saved, None):
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_set, workspace=ws
        )
        for name, t in zip(("dq", "dk", "dv", "dbeta", "dg"), got, strict=True):
            assert torch.equal(t, grads[name]), (name, saved_set is None)
    fwd_only = KdaWorkspace.for_problem(
        S, N, heads, device, backward=False, save_mode="tape"
    )
    o, saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, workspace=fwd_only, save_for_backward=False
    )
    assert saved is None
    assert torch.equal(o, out)


def test_workspace_accepts_an_unindexed_cuda_device(device, manifest) -> None:
    lengths, heads = (65,), 2
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
    )
    with torch.cuda.device(q.device):
        ws = KdaWorkspace.for_problem(sum(lengths), 1, heads, "cuda")
    assert ws.device == q.device
    o, _ = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
    assert o.shape == (sum(lengths), heads, D)


# --------------------------------------------------------------------------- the lower-memory arena option (device)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality: ``torch.equal`` on the integer views, so signed zeros and NaN payloads count."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = torch.int16 if a.element_size() == 2 else torch.int32
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _check_bits(
    got, reference: dict, names=("dq", "dk", "dv", "dbeta", "dg"), tag=None
) -> None:
    for name, t in zip(names, got, strict=True):
        assert _bits_equal(t, reference[name]), (tag, name)


def _check_tail_grids(launched: list, p, path: str, tail_passes: int) -> None:
    """The tail's launches: the three stages once per range, each with the grid of the plan's rule at the range's
    chunk count (one pass: the plan's grids)."""
    tail_names = tuple(
        p.record(stage, path) for stage in ("dqkg", "glm_intra", "gate_epilogue")
    )
    tail = [(name, grid) for name, grid in launched if name in tail_names]
    ranges = cake_backend._tail_ranges(p.NC_pad, tail_passes)
    assert len(tail) == 3 * len(ranges), (path, tail_passes)
    for i, (_c0, length) in enumerate(ranges):
        for j, (stage, name) in enumerate(
            zip(("dqkg", "glm_intra", "gate_epilogue"), tail_names, strict=True)
        ):
            got_name, grid = tail[3 * i + j]
            assert got_name == name
            want = cake_jit.launch_grid(
                name,
                num_chunks_pad=length,
                num_seqs=p.N,
                num_heads=p.H,
                sm_count=p.sm_count,
                num_segments=p.NSEG,
            )
            assert tuple(grid) == tuple(want), (name, i, grid, want)
            if tail_passes == 1:
                assert tuple(grid) == tuple(p.grid(stage, path))


@pytest.mark.parametrize("shape", SHAPES, ids=_shape_id)
def test_lean_arena_gives_the_default_gradients_bitwise(
    device, manifest, shape
) -> None:
    """The lower-memory option changes where the backward's intermediates live and how many launches the tail
    takes, nothing about what the kernels compute: for one, two and three tail passes the gradients from a lean
    workspace, from a per-call lean arena and through the autograd node carrying the option equal the default
    arena's bit for bit; the forward is untouched; a lean workspace refuses the replay and a differing option."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, grads, saved = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    default = KdaWorkspace.for_problem(S, N, heads, device)
    assert default.arena_option == cake_backend.KdaArena()
    p = default.plan
    for t in (1, 2, 3):
        arena = cake_backend.KdaArena(lean=True, tail_passes=t)
        ws = KdaWorkspace.for_problem(S, N, heads, device, arena=arena)
        assert ws.arena_option == arena and ws.arena.dtype == torch.uint8
        assert (
            ws.bytes
            == ws.arena.numel()
            == p.arena_bytes(arena=arena)
            == p.workspace_bytes("bwd_lean", t)
            == cake_backend.workspace_bytes(
                S, N, heads, sm_count=p.sm_count, arena=arena
            )
        )
        assert ws.bytes < default.bytes
        assert "replay" not in ws.paths() and set(ws.records()) == set(ws.paths())
        assert set(ws.paths()) == {"fwd", "fwd_nosave", "light", "tape"} | (
            {"light_states"} if p.recur == "seg" else set()
        )
        o, grads_l, saved_l = _run_fwd_bwd(inputs, workspace=ws)
        assert torch.equal(o, reference_out) and saved_l.version == saved.version
        _check_bits(list(grads_l.values()), reference_grads, tuple(grads_l), tag=t)
        # the default's saved set goes to a lean backward too; a per-call lean arena; both given
        for ws_arg, arena_arg in ((ws, None), (None, arena), (ws, arena)):
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved,
                workspace=ws_arg,
                arena=arena_arg,
            )
            _check_bits(
                got, reference_grads, tag=(t, ws_arg is None, arena_arg is None)
            )
        # a lean arena serves no replay; a differing option on a given workspace is refused
        with pytest.raises(ValueError, match="lean=False"):
            cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=None, workspace=ws
            )
        with pytest.raises(ValueError, match="lean=False"):
            cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=None, arena=arena
            )
        with pytest.raises(ValueError, match="lean=False"):
            ws.views("replay")
        with pytest.raises(ValueError, match="arena="):
            cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved,
                workspace=ws,
                arena=cake_backend.KdaArena(),
            )
        with pytest.raises(ValueError, match="arena="):
            cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=default, arena=arena
            )
        # the autograd node carries the option: the backward's own workspace is lean, the gradients the same
        leaves = [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        o = cake_backend.chunk_kda(*leaves, cu, arena=arena)
        assert torch.equal(o, reference_out)
        o.backward(dout)
        _check_bits(
            [leaf.grad for leaf in leaves],
            reference_grads,
            ("dq", "dk", "dv", "dg", "dbeta"),
            tag=t,
        )
        del ws


def test_lean_tail_passes_launch_the_tail_once_per_range(device, manifest) -> None:
    """The lean workspace's launch list is the default's with the three tail stages once per chunk range, each at
    the grid of the plan's rule for the range's chunk count; one pass is the default list with the default grids
    (and the lean layout of one pass moves nothing the kernels can tell apart)."""
    for lengths, heads in (
        ((1,), 1),  # NC_pad 2: three passes give two ranges of one chunk
        ((63, 1, 65), 2),
        ((129,), 16),
        ((63, 1, 65), 64),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((100, 640, 77, 1000, 129), 2),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        out, grads, saved = _run_fwd_bwd(inputs)
        reference_out, reference_grads = _clone_all(out, grads)
        del out, grads
        launched: list = []

        def recording(name, values, *, grid, device_index):
            launched.append((name, tuple(grid)))
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        for t in (1, 2, 3, 5):
            arena = cake_backend.KdaArena(lean=True, tail_passes=t)
            ws = KdaWorkspace.for_problem(
                S, N, heads, device, arena=arena, launcher=recording
            )
            p = ws.plan
            launched.clear()
            o, saved_l = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
            assert torch.equal(o, reference_out)
            assert [n for n, _ in launched] == _expected_launches(p, "fwd")
            launched.clear()
            got = cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=saved_l, workspace=ws
            )
            names = [n for n, _ in launched]
            assert names == _expected_launches(p, "light", tail_passes=t), (
                lengths,
                heads,
                t,
            )
            assert len(names) == p.launch_count("light") + 3 * (
                len(cake_backend._tail_ranges(p.NC_pad, t)) - 1
            )
            _check_tail_grids(launched, p, "light", t)
            _check_bits(got, reference_grads, tag=(lengths, heads, t))
            # the tail without the saved tables rebuilds them first, as the default does
            launched.clear()
            bare = KdaChunkSaved(
                saved.version,
                saved.Aqk,
                saved.Akk,
                None,
                saved.S,
                saved.H,
                saved.num_docs,
                saved.scale,
            )
            got = cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=bare, workspace=ws
            )
            assert [n for n, _ in launched] == _expected_launches(
                p, "light", saved_tables=False, tail_passes=t
            )
            _check_bits(got, reference_grads, tag=(lengths, heads, t, "bare"))


@pytest.mark.parametrize(("num_docs", "heads"), STATE_SHAPES, ids=["n1-h1", "n2-h2"])
def test_lean_arena_serves_the_saved_states_and_the_tape_paths(
    device, manifest, monkeypatch, num_docs, heads
) -> None:
    """On the segmented rows that save the segment-boundary states (``light_states``) and on the tape paths (a
    version-2 set on a lean tape workspace; a version-3 set on a lean tape workspace, which recomputes ``h`` /
    ``v_new`` into per-call buffers) the lean layouts give the default arena's gradients bit for bit, with the
    tail once per range; a per-call lean arena is sized by the saved set's version."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    nseg = cake_backend.plan(1, num_docs, heads, sm, arch).NSEG
    S = (
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    )  # the shortest pack that saves the states
    lengths = _split(S, num_docs)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, grads, saved = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    assert saved.version == cake_backend.SAVED_VERSION and saved.hseg is not None
    created: list = []
    original = cake_backend.KdaWorkspace.for_problem

    def spy(
        *args, **kwargs
    ):  # records the option, not the workspace: a strong reference would be "retained"
        ws_ = original(*args, **kwargs)
        created.append(
            types.SimpleNamespace(
                arena_option=ws_.arena_option,
                backward=ws_.backward,
                bytes=ws_.bytes,
                save_mode=ws_.save_mode,
            )
        )
        return ws_

    monkeypatch.setattr(cake_backend.KdaWorkspace, "for_problem", spy)
    launched: list = []

    def recording(name, values, *, grid, device_index):
        launched.append((name, tuple(grid)))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    for t in (1, 3):
        arena = cake_backend.KdaArena(lean=True, tail_passes=t)
        ws = KdaWorkspace.for_problem(
            S, num_docs, heads, device, arena=arena, launcher=recording
        )
        p = ws.plan
        assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
        launched.clear()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        assert [n for n, _ in launched] == _expected_launches(
            p, "light_states", tail_passes=t
        )
        _check_tail_grids(launched, p, "light_states", t)
        _check_bits(got, reference_grads, tag=("light_states", t))
        # the lean tape workspace: its own version-2 set (the tape path) and the version-3 set (per-call h / v_new)
        tape_ws = KdaWorkspace.for_problem(
            S,
            num_docs,
            heads,
            device,
            save_mode="tape",
            arena=arena,
            launcher=recording,
        )
        assert tape_ws.bytes == p.workspace_bytes("bwd_tape_lean", t) < ws.bytes
        launched.clear()
        o, saved_t = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=tape_ws)
        assert (
            torch.equal(o, reference_out)
            and saved_t.version == cake_backend.SAVED_VERSION_TAPE
        )
        assert [n for n, _ in launched] == _expected_launches(p, "fwd")
        launched.clear()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_t, workspace=tape_ws
        )
        assert [n for n, _ in launched] == _expected_launches(p, "tape", tail_passes=t)
        _check_tail_grids(launched, p, "tape", t)
        _check_bits(got, reference_grads, tag=("tape", t))
        launched.clear()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
        )
        assert [n for n, _ in launched] == _expected_launches(
            p, "light_states", tail_passes=t
        )
        _check_bits(got, reference_grads, tag=("light_states on tape", t))
        # per-call lean arenas: sized by the saved set's version
        created.clear()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_t, arena=arena
        )
        _check_bits(got, reference_grads, tag=("per-call tape", t))
        assert len(created) == 1 and created[0].save_mode == "tape"
        assert created[0].arena_option == arena and created[0].bytes == tape_ws.bytes
        created.clear()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, arena=arena
        )
        _check_bits(got, reference_grads, tag=("per-call light", t))
        assert (
            len(created) == 1
            and created[0].save_mode == "light"
            and created[0].bytes == ws.bytes
        )
        created.clear()


def test_lean_arena_under_checkpointing_and_a_dropped_workspace(
    device, manifest, monkeypatch
) -> None:
    """A checkpointed forward with the option retains nothing beyond its output, and the backward's own
    workspace (the forward's per-call one is gone, or the caller dropped theirs) takes the recorded option."""
    import gc

    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    cu, dout = inputs["cu_seqlens"], inputs["dout"]
    out, grads, _ = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    arena = cake_backend.KdaArena(lean=True, tail_passes=2)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    created: list = []
    original = cake_backend.KdaWorkspace.for_problem

    def spy(
        *args, **kwargs
    ):  # records the option, not the workspace: a strong reference would be "retained"
        ws_ = original(*args, **kwargs)
        created.append(
            types.SimpleNamespace(
                arena_option=ws_.arena_option,
                backward=ws_.backward,
                bytes=ws_.bytes,
                save_mode=ws_.save_mode,
            )
        )
        return ws_

    monkeypatch.setattr(cake_backend.KdaWorkspace, "for_problem", spy)

    def leaves():
        return [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

    def check(leaf_list, tag):
        _check_bits(
            [leaf.grad for leaf in leaf_list],
            reference_grads,
            ("dq", "dk", "dv", "dg", "dbeta"),
            tag=tag,
        )

    def implicit(q, k, v, g, beta):
        return cake_backend.chunk_kda(q, k, v, g, beta, cu, arena=arena)

    held = leaves()
    warm = implicit(*held)
    assert torch.equal(warm, reference_out)
    del warm
    gc.collect()  # garbage of earlier tests must not be collected inside the measurement
    assert _retained_after_checkpointed_forward(implicit, *held) == 0
    created.clear()
    o = checkpoint(implicit, *held, use_reentrant=False)
    o.backward(dout)
    check(held, "checkpoint")
    # every workspace of the checkpointed step took the option; the backward's is the lean backward layout
    assert created and all(w.arena_option == arena for w in created)
    backward_ws = [w for w in created if w.backward]
    assert backward_ws and all(
        w.bytes == plan.workspace_bytes("bwd_lean", 2) for w in backward_ws
    )
    # a caller's workspace dropped after the forward: the node recorded its option
    dropped = KdaWorkspace(plan, device, backward=True, arena=arena)
    created.clear()
    held = leaves()
    o = cake_backend.chunk_kda(*held, cu, workspace=dropped)
    assert torch.equal(o, reference_out)
    del dropped
    gc.collect()
    o.backward(dout)
    assert (
        len(created) == 1 and created[0].arena_option == arena and created[0].backward
    )
    check(held, "dropped")
    # the option given with a kept workspace of another option is refused by the node's forward
    kept = KdaWorkspace(plan, device, backward=True)
    with pytest.raises(ValueError, match="arena="):
        cake_backend.chunk_kda(*leaves(), cu, workspace=kept, arena=arena)


def test_lean_arena_content_never_reaches_a_result(device, manifest) -> None:
    """Write-before-read on the lean layouts too: the per-pass buffers and the offset views are written in the
    same call before the tail reads them, so an arena filled with NaN bytes gives the zero-filled default arena's
    results bit for bit (the forward, the backward from the light and the tape sets, one to three passes)."""
    cases = (
        ((129,), 2, "light", 2),
        ((100, 640, 77, 1000, 129), 2, "light", 3),
        ((63, 1, 65, 30), 2, "light", 1),
        ((20, 20, 20, 20, 20, 20, 20), 8, "light", 2),
        ((63, 1, 65), 64, "light", 3),
        ((100, 640, 77, 1000, 129), 2, "tape", 2),
    )
    for lengths, heads, save_mode, t in cases:
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = KdaWorkspace(
            plan, device, backward=True, zero_init=True, save_mode=save_mode
        )
        out, grads, saved = _run_fwd_bwd(inputs, workspace=zeroed)
        reference_out, reference_grads = _clone_all(out, grads)
        del out, grads
        arena = cake_backend.KdaArena(lean=True, tail_passes=t)
        poisoned = KdaWorkspace(
            plan, device, backward=True, save_mode=save_mode, arena=arena
        )
        assert poisoned.zero_init is False and poisoned.bytes < zeroed.bytes

        def poison():
            poisoned.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        case = (lengths, heads, save_mode, t)
        poison()
        o, saved_p = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=poisoned
        )
        assert torch.equal(o, reference_out), case
        assert torch.equal(saved_p.Aqk, saved.Aqk) and torch.equal(
            saved_p.Akk, saved.Akk
        ), case
        poison()
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_p, workspace=poisoned
        )
        _check_bits(got, reference_grads, tag=case)
        if (
            save_mode == "tape"
        ):  # a version-3 set on the poisoned tape workspace: per-call h / v_new
            _, saved_light = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=poisoned, save_mode="light"
            )
            poison()
            got = cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=saved_light, workspace=poisoned
            )
            _check_bits(got, reference_grads, tag=(*case, "light set"))
        poison()
        nosave, _ = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, save_for_backward=False, workspace=poisoned
        )
        assert torch.equal(nosave, reference_out), case


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option (document groups, KdaArena.batch_tokens): host checks


def _partitions(lengths):
    """``(label, KdaArena keywords)`` of the document groupings a test runs on a pack: two groups and three (the
    smallest token budget under which the greedy partition gives exactly that many groups, when one exists; the
    group count never grows with the budget, so a binary search finds it) and one document per group."""
    lengths = tuple(int(n) for n in lengths)
    S, longest = sum(lengths), max(lengths)
    out = []
    for k in (2, 3):
        lo, hi = longest, S
        while lo < hi:  # the smallest budget with at most k groups
            mid = (lo + hi) // 2
            if len(cake_backend.doc_batches(lengths, mid)) <= k:
                hi = mid
            else:
                lo = mid + 1
        if len(cake_backend.doc_batches(lengths, lo)) == k:
            out.append((f"g{k}", {"batch_tokens": lo}))
    out.append(("gdoc", {"batch_tokens": S, "max_docs_per_batch": 1}))
    return out


def test_doc_batches_partitions() -> None:
    db = cake_backend.doc_batches
    assert db((10, 20, 30), 60) == ((0, 3),)
    assert db((10, 20, 30), 30) == ((0, 2), (2, 3))
    assert db((10, 20, 30), 59) == ((0, 2), (2, 3))
    assert db((10, 20, 30), 60, 1) == ((0, 1), (1, 2), (2, 3))
    assert db((10, 20, 30), 60, 2) == ((0, 2), (2, 3))
    assert db((30, 30, 30, 30), 60) == ((0, 2), (2, 4))
    # a zero-length document joins the group before it (never a group of its own); leading ones open the first
    assert db((100, 0, 300), 300) == ((0, 2), (2, 3))
    assert db((100, 0, 300), 400, 1) == ((0, 2), (2, 3))
    assert db((0, 0, 100), 100, 1) == ((0, 3),)
    assert db((100, 0), 100, 1) == ((0, 2),)
    assert db((1,), 1) == ((0, 1),)
    assert db([10, 20], 25) == ((0, 1), (1, 2))
    with pytest.raises(ValueError, match="document 2 has 30 tokens"):
        db((10, 20, 30), 29)
    with pytest.raises(ValueError, match="no token"):
        db((0, 0), 5)
    with pytest.raises(ValueError, match="non-negative"):
        db((), 5)
    with pytest.raises(ValueError, match="non-negative"):
        db((1, -1), 5)
    # the search helper of these tests: exact group counts, one document per group
    parts = dict(_partitions((23496, 88510, 114974, 34413)))
    assert (
        len(cake_backend.doc_batches((23496, 88510, 114974, 34413), **parts["g2"])) == 2
    )
    assert (
        len(cake_backend.doc_batches((23496, 88510, 114974, 34413), **parts["g3"])) == 3
    )
    assert parts["gdoc"]["max_docs_per_batch"] == 1


def test_arena_option_validation_batched() -> None:
    KdaArena = cake_backend.KdaArena
    option = KdaArena(batch_tokens=1000)
    assert option.batched and not option.lean and option.max_docs_per_batch is None
    assert KdaArena(
        lean=True, tail_passes=4, batch_tokens=10, max_docs_per_batch=1
    ).batched
    assert not KdaArena().batched and not KdaArena(lean=True, tail_passes=3).batched
    for bad in (0, -1, True, 1.5, "3"):
        with pytest.raises(ValueError, match="batch_tokens"):
            KdaArena(batch_tokens=bad)
    with pytest.raises(ValueError, match="max_docs_per_batch"):
        KdaArena(batch_tokens=10, max_docs_per_batch=0)
    with pytest.raises(ValueError, match="batch_tokens too"):
        KdaArena(max_docs_per_batch=1)


def test_batch_plans_pin_the_whole_problems_records_and_segments() -> None:
    """The groups of a pack: contiguous, covering, each with its own geometry (at or above its real chunk count; the
    tape bound even and inside the whole problem's tape), a plan with the whole problem's records, ``P`` and route
    and the grids of the records' rules at the group's scalars; the arena of the exact groups never exceeds the
    bound of every group, which the byte query gives without ``doc_lengths``."""
    sm = 148
    for arch in ARCHES:
        for lengths, heads in (
            ((23496, 88510, 114974, 34413), 4),
            ((100, 0, 300), 2),
            ((63, 1, 65), 64),
            ((2000,) * 300, 1),
            ((20,) * 7, 8),
        ):
            S, N = sum(lengths), len(lengths)
            p = cake_backend.plan(S, N, heads, sm, arch)
            for _label, kw in _partitions(lengths):
                arena = cake_backend.KdaArena(lean=True, tail_passes=4, **kw)
                batches = p.batches(arena, lengths)
                assert [(b.d0, b.d1) for b in batches] == list(
                    cake_backend.doc_batches(
                        lengths, arena.batch_tokens, arena.max_docs_per_batch
                    )
                )
                assert batches[0].d0 == 0 and batches[-1].d1 == N
                assert all(
                    a.d1 == b.d0 for a, b in zip(batches, batches[1:], strict=False)
                )
                c0 = 0
                for b in batches:
                    assert (b.S, b.N, b.c0) == (
                        sum(lengths[b.d0 : b.d1]),
                        b.d1 - b.d0,
                        c0,
                    )
                    real = sum(-(-n // BT) for n in lengths[b.d0 : b.d1])
                    _nc_max, nc_pad, _rows = cake_backend._geometry(b.S, b.N)
                    assert b.NC_pad == nc_pad >= real
                    assert b.NC_pad_tape % 2 == 0 and real <= b.NC_pad_tape <= b.NC_pad
                    assert c0 + b.NC_pad_tape <= p.NC_pad
                    c0 += real
                    for tape in (False, True):
                        pb = p.batch_plan(b, tape=tape)
                        assert pb.records is p.records and (pb.P, pb.recur) == (
                            p.P,
                            p.recur,
                        )
                        assert pb.proportional == 0
                        assert pb.NC_pad == (b.NC_pad_tape if tape else b.NC_pad)
                        assert BT * pb.NC_pad == pb.R and pb.NC_max <= pb.NC_pad
                        assert (b.N * p.P if p.recur == "seg" else 0) == pb.NSEG
                        assert pb.seg_arg == (p.P if p.recur == "seg" else p.seg_arg)
                        assert set(pb.grids) == set(p.grids)
                        for name, grids in pb.grids.items():
                            assert grids == tuple(
                                cake_jit.launch_grid(
                                    name,
                                    num_chunks_pad=pb.NC_pad,
                                    num_seqs=pb.N,
                                    num_heads=heads,
                                    sm_count=sm,
                                    num_segments=pb.NSEG,
                                    launch=launch,
                                )
                                for launch in range(cake_jit.launches(name))
                            )
                exact = p.arena_bytes(arena=arena, doc_lengths=lengths)
                bound = p.arena_bytes(arena=arena)
                nc_bar, rows_bar, nseg_bar = p.batch_bound(arena)
                assert rows_bar == BT * nc_bar and nseg_bar == p.NSEG
                assert all(b.NC_pad <= nc_bar for b in batches)
                assert (
                    exact
                    <= bound
                    <= p.arena_bytes(
                        arena=cake_backend.KdaArena(lean=True, tail_passes=4)
                    )
                )
                assert exact == max(
                    cake_backend._arena_bytes(
                        b.NC_pad * BT,
                        heads,
                        b.NC_pad,
                        p.batch_segments(b),
                        backward=True,
                        save_mode="light",
                        arena=arena,
                    )
                    for b in batches
                )
                assert (
                    cake_backend.workspace_bytes(
                        S,
                        N,
                        heads,
                        sm_count=sm,
                        arch=arch,
                        arena=arena,
                        doc_lengths=lengths,
                    )
                    == exact
                )
                assert (
                    cake_backend.workspace_bytes(
                        S, N, heads, sm_count=sm, arch=arch, arena=arena
                    )
                    == bound
                )
                assert (
                    p.arena_bytes(backward=False, arena=arena, doc_lengths=lengths)
                    <= p.arena_bytes(backward=False, arena=arena)
                    <= p.arena_bytes(backward=False)
                )
                tape_exact = p.arena_bytes(
                    save_mode="tape", arena=arena, doc_lengths=lengths
                )
                assert tape_exact <= p.arena_bytes(save_mode="tape", arena=arena)
            with pytest.raises(ValueError, match="doc_lengths"):
                p.batches(cake_backend.KdaArena(batch_tokens=S), None)
            with pytest.raises(ValueError, match="doc_lengths"):
                p.batches(cake_backend.KdaArena(batch_tokens=S), lengths[:-1])
            with pytest.raises(ValueError, match="doc_lengths"):
                p.check_doc_lengths(tuple(lengths[:-1]) + (lengths[-1] + 1,))
            with pytest.raises(ValueError, match="above batch_tokens"):
                p.batches(cake_backend.KdaArena(batch_tokens=max(lengths) - 1), lengths)


def test_batched_arena_refuses_the_proportional_rows() -> None:
    for arch in ARCHES:
        p = cake_backend.plan(
            128, 2, 1, 148, arch
        )  # two chains: proportional segment counts
        assert p.recur == "seg" and p.proportional == 1
        arena = cake_backend.KdaArena(batch_tokens=64)
        with pytest.raises(ValueError, match="proportional"):
            p.check_arena(arena)
        with pytest.raises(ValueError, match="proportional"):
            p.check_arena(arena, backward=False)
        with pytest.raises(ValueError, match="proportional"):
            p.arena_bytes(arena=arena, doc_lengths=(64, 64))
        with pytest.raises(ValueError, match="proportional"):
            cake_backend.workspace_bytes(
                128, 2, 1, sm_count=148, arch=arch, arena=arena
            )
        with pytest.raises(ValueError, match="proportional"):
            cake_backend.memory_report(128, 2, 1, sm_count=148, arch=arch, arena=arena)
        p5 = cake_backend.plan(320, 5, 1, 148, arch)  # five chains are not
        assert p5.recur == "seg" and p5.proportional == 0
        p5.check_arena(arena)
        p64 = cake_backend.plan(128, 2, 64, 148, arch)  # the serial route never is
        assert p64.recur == "serial" and p64.proportional == 0
        p64.check_arena(arena)


def test_arena_for_budget_search_order() -> None:
    lengths, heads = (23496, 88510, 114974, 34413), 4
    S, N = sum(lengths), len(lengths)
    A = cake_backend.KdaArena
    for arch in ARCHES:
        p = cake_backend.plan(S, N, heads, 148, arch)
        steps = [
            A(),
            A(lean=True),
            A(lean=True, tail_passes=2),
            A(lean=True, tail_passes=4),
        ]
        sizes = [p.arena_bytes(arena=a) for a in steps]
        assert sizes == sorted(sizes, reverse=True) and len(set(sizes)) == 4
        for a, nbytes in zip(steps, sizes, strict=True):
            assert cake_backend.arena_for_budget(p, nbytes) == a
            assert cake_backend.arena_for_budget(p, nbytes, doc_lengths=lengths) == a
        below = sizes[-1] - 1
        with pytest.raises(ValueError, match="doc_lengths"):
            cake_backend.arena_for_budget(p, below)
        got = cake_backend.arena_for_budget(p, below, doc_lengths=lengths)
        assert got == A(
            lean=True, tail_passes=4, batch_tokens=max(max(lengths), S // 2)
        )
        assert p.arena_bytes(arena=got, doc_lengths=lengths) <= below
        single = A(
            lean=True, tail_passes=4, batch_tokens=max(lengths), max_docs_per_batch=1
        )
        floor = p.arena_bytes(arena=single, doc_lengths=lengths)
        got = cake_backend.arena_for_budget(p, floor, doc_lengths=lengths)
        assert got.batched and p.arena_bytes(arena=got, doc_lengths=lengths) <= floor
        with pytest.raises(ValueError, match="floor"):
            cake_backend.arena_for_budget(p, floor - 1, doc_lengths=lengths)
        # the dav rows skip the lean steps and batch the default layouts; the proportional rows refuse batching
        p12 = cake_backend.plan(S, N, 12, 148, arch)
        assert cake_backend.arena_for_budget(p12, p12.arena_bytes()) == A()
        got = cake_backend.arena_for_budget(
            p12, p12.arena_bytes() - 1, doc_lengths=lengths
        )
        assert got.batched and not got.lean and got.tail_passes == 1
        p2 = cake_backend.plan(128, 2, 1, 148, arch)
        with pytest.raises(ValueError, match="refuses the batched option"):
            cake_backend.arena_for_budget(p2, 1, doc_lengths=(64, 64))


def test_memory_report_batched_fields() -> None:
    lengths, heads = (23496, 88510, 114974, 34413), 4
    S, N = sum(lengths), len(lengths)
    for sm in (148, 152):
        for arch in ARCHES:
            base = cake_backend.memory_report(S, N, heads, sm_count=sm, arch=arch)
            assert base["batch_tokens"] is None and base["batches"] is None
            assert (
                base["arena_bwd_bound_batched"] is None
                and base["arena_floor_batched"] is None
            )
            arena = cake_backend.KdaArena(lean=True, tail_passes=4, batch_tokens=120000)
            p = cake_backend.plan(S, N, heads, sm, arch)
            r = cake_backend.memory_report(
                S, N, heads, sm_count=sm, arch=arch, arena=arena
            )
            assert r["batch_tokens"] == 120000 and r["max_docs_per_batch"] is None
            assert r["batches"] is None and r["batch_scratch"] is None
            assert (
                r["arena_bwd_configured"]
                == r["arena_bwd_bound_batched"]
                == p.arena_bytes(arena=arena)
            )
            assert r["largest_batch_rows"] == p.batch_bound(arena)[1]
            assert r["bwd_total"] == r["arena_bwd_configured"] + r["out_bwd"]
            r2 = cake_backend.memory_report(
                S, N, heads, sm_count=sm, arch=arch, arena=arena, doc_lengths=lengths
            )
            groups = p.batches(arena, lengths)
            assert r2["batches"] == [(b.d0, b.d1, b.S, b.N) for b in groups]
            assert r2["batches"] == [
                (0, 2, 112006, 2),
                (2, 3, 114974, 1),
                (3, 4, 34413, 1),
            ]
            assert r2["arena_bwd_configured"] == p.arena_bytes(
                arena=arena, doc_lengths=lengths
            )
            assert r2["arena_bwd_configured"] <= r2["arena_bwd_bound_batched"]
            assert r2["largest_batch_rows"] == max(b.NC_pad for b in groups) * BT
            assert r2["arena_fwd_configured"] == p.arena_bytes(
                backward=False, arena=arena, doc_lengths=lengths
            )
            big = max(groups, key=lambda b: b.NC_pad)
            blocks = 3 * p.batch_segments(big) * heads * D * D * 4
            assert (
                blocks
                <= r2["batch_scratch"]
                <= blocks
                + 4
                * (
                    2 * big.NC_pad
                    + 2 * big.N
                    + 2
                    + p.batch_segments(big)
                    + 2 * big.N
                    + 1
                    + 5 * 64
                )
            )
            nc_min = -(-max(lengths) // BT) + 1
            rows_min = BT * (nc_min + nc_min % 2)
            block_min = p.P * heads * D * D * 4 if p.recur == "seg" else 0
            assert r2["arena_floor_batched"] == 2820 * rows_min * heads + 3 * block_min
            single = cake_backend.KdaArena(
                lean=True,
                tail_passes=4,
                batch_tokens=max(lengths),
                max_docs_per_batch=1,
            )
            exact_floor = p.arena_bytes(arena=single, doc_lengths=lengths)
            # the exact one-document arena: the floor plus the ceil slack of the last tail range
            assert 0 <= exact_floor - r2["arena_floor_batched"] <= 2568 * 3 * BT * heads
            tape = cake_backend.memory_report(
                S,
                N,
                heads,
                "tape",
                sm_count=sm,
                arch=arch,
                arena=arena,
                doc_lengths=lengths,
            )
            assert (
                tape["arena_floor_batched"] == 2052 * rows_min * heads + 3 * block_min
            )
            assert tape["arena_bwd_configured"] == p.arena_bytes(
                save_mode="tape", arena=arena, doc_lengths=lengths
            )


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option: device checks

# the rows with at least two documents and more than four (document, head) chains (the proportional rows refuse the
# option), plus serial rows (16, 32 and 64 heads)
D_ROWS = [s for s in SHAPES if len(s[0]) >= 2 and len(s[0]) * s[1] > 4] + [
    ((63, 1, 65), 64),
    ((129, 300), 16),
    ((20, 20, 20, 20, 20, 20, 20), 32),
]


def _expected_batched_launches(p, arena, lengths, path, *, tail_passes=1):
    """The launch order of a batched call: on the saving forward the whole problem's tables by the saved-set
    paths' table program, then every group's forward list (its own tables where the row dispatches a table program
    on the forward); on the backward every group's list with the group's tables first (the saved set's describe the
    whole problem) and the tail once per range of the group's chunk bound."""
    seq = []
    batches = p.batches(arena, lengths)
    if path in ("fwd", "fwd_nosave"):
        if path == "fwd" and p.dispatched("table", "light"):
            seq.append(p.record("table", "light"))
            if p.recur == "seg" and p.dispatched("seg_table", "light"):
                seq.append("seg_table")
        for b in batches:
            seq += _expected_launches(p.batch_plan(b), path)
        return seq
    for b in batches:
        seq += _expected_launches(
            p.batch_plan(b, tape=path == "tape"),
            path,
            saved_tables=False,
            tail_passes=tail_passes,
        )
    return seq


def _check_group_launches(launched, p, arena, lengths, path, *, tail_passes=1) -> None:
    """Every launch of a group carries the group's scalars (``num_seqs``, ``num_chunks_pad``) and one of the
    group plan's grids; the tail passes carry ``num_seqs = 0`` and their range's chunk count."""
    batches = p.batches(arena, lengths)
    forward = path in ("fwd", "fwd_nosave")
    pos = 0
    if forward and path == "fwd" and p.dispatched("table", "light"):
        head = 2 if (p.recur == "seg" and p.dispatched("seg_table", "light")) else 1
        for name, grid, num_seqs, nc in launched[:head]:
            assert (num_seqs, nc) == (p.N, p.NC_pad) and grid in p.grids[name], name
        pos = head
    for b in batches:
        pb = p.batch_plan(b, tape=path == "tape")
        expected = (
            _expected_launches(pb, path)
            if forward
            else _expected_launches(
                pb, path, saved_tables=False, tail_passes=tail_passes
            )
        )
        group = launched[pos : pos + len(expected)]
        pos += len(expected)
        assert [n for n, *_ in group] == expected
        tail_names = (
            set()
            if forward
            else {pb.record(s, path) for s in ("dqkg", "glm_intra", "gate_epilogue")}
        )
        ranges = cake_backend._tail_ranges(pb.NC_pad, tail_passes)
        tails = []
        for name, grid, num_seqs, nc in group:
            if name in tail_names and tail_passes > 1:
                tails.append((name, grid, num_seqs, nc))
                continue
            assert (num_seqs, nc) == (pb.N, pb.NC_pad), (name, b)
            assert grid in pb.grids[name], (name, b, grid, pb.grids[name])
        if tails:
            assert len(tails) == 3 * len(ranges)
            for i, (_c0, length) in enumerate(ranges):
                for j in range(3):
                    name, grid, num_seqs, nc = tails[3 * i + j]
                    assert (num_seqs, nc) == (0, length), (name, b)
                    want = cake_jit.launch_grid(
                        name,
                        num_chunks_pad=length,
                        num_seqs=pb.N,
                        num_heads=pb.H,
                        sm_count=pb.sm_count,
                        num_segments=pb.NSEG,
                    )
                    assert tuple(grid) == tuple(want), (name, b, i)
    assert pos == len(launched)


def _real_chunks(lengths) -> int:
    return sum(-(-int(n) // BT) for n in lengths)


@pytest.mark.parametrize("shape", D_ROWS, ids=_shape_id)
def test_batched_arena_gives_the_default_results_bitwise(
    device, manifest, shape
) -> None:
    """Two groups, three and one document per group, with and without the lean layouts: the batched forward gives
    the default's ``out``, token rows, tables and saved states bit for bit, the batched backward the default's
    gradients from the default's saved set and from its own, the default backward reads the batched set, the
    per-call arena, the replay (a workspace that is not lean), the tape set (slices and transfers), a light set on
    the tape workspace and the autograd node agree; ``doc_lengths`` are required and checked."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, grads, saved = _run_fwd_bwd(inputs)
    ref_out, ref_grads = _clone_all(out, grads)
    del out, grads
    default = KdaWorkspace.for_problem(S, N, heads, device)
    p = default.plan
    real = _real_chunks(lengths)
    tape_default = KdaWorkspace.for_problem(S, N, heads, device, save_mode="tape")
    _, saved_t = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, workspace=tape_default
    )
    ref_h, ref_v_new = saved_t.h[:real].clone(), saved_t.v_new[: BT * real].clone()
    ref_seg_m = None if saved_t.seg_m is None else saved_t.seg_m.clone()
    del tape_default
    for label, kw in _partitions(lengths):
        for lean in (False, True):
            arena = cake_backend.KdaArena(lean=lean, tail_passes=3 if lean else 1, **kw)
            tag = (label, lean)
            ws = KdaWorkspace.for_problem(
                S, N, heads, device, arena=arena, doc_lengths=lengths
            )
            assert ws.arena_option == arena and ws.doc_lengths == tuple(lengths)
            assert ws.bytes == p.arena_bytes(arena=arena, doc_lengths=lengths)
            assert ws.bytes <= p.arena_bytes(arena=arena) and ws.bytes <= default.bytes
            batches = p.batches(arena, lengths)
            assert len(batches) >= 2, tag
            # the batched forward: the same out and saved set (tables, token rows, states)
            o, saved_b = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
            )
            assert _bits_equal(o, ref_out), tag
            assert _bits_equal(saved_b.Aqk, saved.Aqk) and _bits_equal(
                saved_b.Akk, saved.Akk
            ), tag
            assert len(saved_b.tables) == len(saved.tables), tag
            assert all(
                torch.equal(a, b)
                for a, b in zip(saved_b.tables, saved.tables, strict=True)
            ), tag
            assert (saved_b.hseg is None) == (saved.hseg is None), tag
            if saved.hseg is not None:
                assert _bits_equal(saved_b.hseg, saved.hseg), tag
                assert _bits_equal(saved_b.seg_state, saved.seg_state), tag
            # the batched backward from the default's set and from its own; the default backward from the batched set
            for name, source in (("default set", saved), ("batched set", saved_b)):
                got = cake_backend.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=source,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                _check_bits(got, ref_grads, tag=(*tag, name))
            got = cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=saved_b, workspace=default
            )
            _check_bits(
                got, ref_grads, tag=(*tag, "default backward from the batched set")
            )
            # a per-call batched arena (sized for the exact groups)
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved,
                arena=arena,
                doc_lengths=lengths,
            )
            _check_bits(got, ref_grads, tag=(*tag, "per-call"))
            # the replay group by group on a workspace that is not lean; a lean one refuses it naming both options
            if not lean:
                got = cake_backend.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=None,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                _check_bits(got, ref_grads, tag=(*tag, "replay"))
            else:
                with pytest.raises(ValueError, match="batch_tokens"):
                    cake_backend.chunk_kda_bwd(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=None,
                        workspace=ws,
                        doc_lengths=lengths,
                    )
            # the tape set: the batched forward's slices and transfers equal the default's; both backwards agree
            tws = KdaWorkspace.for_problem(
                S, N, heads, device, save_mode="tape", arena=arena, doc_lengths=lengths
            )
            assert tws.bytes == p.arena_bytes(
                save_mode="tape", arena=arena, doc_lengths=lengths
            )
            o, saved_tb = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=tws, doc_lengths=lengths
            )
            assert (
                _bits_equal(o, ref_out)
                and saved_tb.version == cake_backend.SAVED_VERSION_TAPE
            ), tag
            assert _bits_equal(saved_tb.h[:real], ref_h), tag
            assert _bits_equal(saved_tb.v_new[: BT * real], ref_v_new), tag
            assert (saved_tb.seg_m is None) == (ref_seg_m is None), tag
            if ref_seg_m is not None:
                assert _bits_equal(saved_tb.seg_m, ref_seg_m), tag
            for name, source in (
                ("own tape set", saved_tb),
                ("default tape set", saved_t),
            ):
                got = cake_backend.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=source,
                    workspace=tws,
                    doc_lengths=lengths,
                )
                _check_bits(got, ref_grads, tag=(*tag, name))
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved_b,
                workspace=tws,
                doc_lengths=lengths,
            )
            _check_bits(got, ref_grads, tag=(*tag, "light set on the tape workspace"))
            got = cake_backend.chunk_kda_bwd(
                q, k, v, g, beta, cu, dout, saved=saved_tb, workspace=default
            )
            _check_bits(
                got, ref_grads, tag=(*tag, "default backward from the batched tape set")
            )
            # the autograd node carries the option and doc_lengths; a forward without a saved set
            leaves = [
                inputs[n].detach().clone().requires_grad_(True)
                for n in ("q", "k", "v", "g", "beta")
            ]
            o = cake_backend.chunk_kda(*leaves, cu, arena=arena, doc_lengths=lengths)
            assert _bits_equal(o, ref_out), tag
            o.backward(dout)
            _check_bits(
                [leaf.grad for leaf in leaves],
                ref_grads,
                ("dq", "dk", "dv", "dg", "dbeta"),
                tag=(*tag, "autograd"),
            )
            nosave, none = cake_backend.chunk_kda_fwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                save_for_backward=False,
                workspace=ws,
                doc_lengths=lengths,
            )
            assert none is None and _bits_equal(nosave, ref_out), tag
            # doc_lengths are required and checked on the host
            with pytest.raises(ValueError, match="doc_lengths"):
                cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
            with pytest.raises(ValueError, match="doc_lengths"):
                cake_backend.chunk_kda_bwd(
                    q, k, v, g, beta, cu, dout, saved=saved, arena=arena
                )
            with pytest.raises(ValueError, match="doc_lengths"):
                cake_backend.chunk_kda_fwd(
                    q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths[:-1]
                )
            with pytest.raises(ValueError, match="doc_lengths"):
                cake_backend.chunk_kda(
                    *[inputs[n] for n in ("q", "k", "v", "g", "beta")], cu, arena=arena
                )
            del ws, tws


def test_batched_arena_launches_per_group(device, manifest) -> None:
    """The batched call's launch list is every group's list in turn (the whole problem's tables first on the saving
    forward; the group's tables first on the backward), each launch at the group's scalars and grids; the overlap
    pair's epoch advances once per group through the shared counter; the group tables are slices of the whole
    problem's."""
    for lengths, heads in (
        ((63, 1, 65), 4),
        ((100, 0, 300), 2),
        ((63, 1, 65, 30), 2),
        ((20, 20, 20, 20, 20, 20, 20), 8),
        ((100, 640, 77, 1000, 129), 2),
        ((63, 1, 65), 64),
        ((129, 300), 16),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        out, grads, saved = _run_fwd_bwd(inputs)
        ref_out, ref_grads = _clone_all(out, grads)
        del out, grads
        launched: list = []

        def recording(name, values, *, grid, device_index):
            launched.append(
                (
                    name,
                    tuple(grid),
                    int(values["num_seqs"]),
                    int(values["num_chunks_pad"]),
                )
            )
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        for label, kw in _partitions(lengths):
            for t in (1, 3):
                arena = cake_backend.KdaArena(lean=t > 1, tail_passes=t, **kw)
                ws = KdaWorkspace.for_problem(
                    S,
                    N,
                    heads,
                    device,
                    arena=arena,
                    launcher=recording,
                    doc_lengths=lengths,
                )
                p = ws.plan
                batches = p.batches(arena, lengths)
                case = (lengths, heads, label, t)
                epoch0 = ws._epoch[0]
                launched.clear()
                o, saved_b = cake_backend.chunk_kda_fwd(
                    q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
                )
                assert _bits_equal(o, ref_out), case
                names = [n for n, *_ in launched]
                assert names == _expected_batched_launches(p, arena, lengths, "fwd"), (
                    case
                )
                _check_group_launches(launched, p, arena, lengths, "fwd")
                overlap = (
                    p.recur == "seg" and p.record("seg_carry", "fwd") == "seg_carry_ov"
                )
                assert ws._epoch[0] - epoch0 == (len(batches) if overlap else 0), case
                for view in ws._batch_views.values():
                    assert view._epoch is ws._epoch
                    assert view.eye.data_ptr() == ws.eye.data_ptr()
                    assert view.arena.data_ptr() == ws.arena.data_ptr()
                launched.clear()
                nosave, _none = cake_backend.chunk_kda_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    save_for_backward=False,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                assert _bits_equal(nosave, ref_out), case
                assert [n for n, *_ in launched] == _expected_batched_launches(
                    p, arena, lengths, "fwd_nosave"
                ), case
                launched.clear()
                got = cake_backend.chunk_kda_bwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    cu,
                    dout,
                    saved=saved_b,
                    workspace=ws,
                    doc_lengths=lengths,
                )
                _check_bits(got, ref_grads, tag=case)
                path = p.light_path
                names = [n for n, *_ in launched]
                assert names == _expected_batched_launches(
                    p, arena, lengths, path, tail_passes=t
                ), case
                _check_group_launches(launched, p, arena, lengths, path, tail_passes=t)
                if t == 1:  # the replay group by group
                    launched.clear()
                    got = cake_backend.chunk_kda_bwd(
                        q,
                        k,
                        v,
                        g,
                        beta,
                        cu,
                        dout,
                        saved=None,
                        workspace=ws,
                        doc_lengths=lengths,
                    )
                    _check_bits(got, ref_grads, tag=(*case, "replay"))
                    assert [n for n, *_ in launched] == _expected_batched_launches(
                        p, arena, lengths, "replay"
                    ), case
                    _check_group_launches(launched, p, arena, lengths, "replay")
                # the group tables: absolute chunk offsets, the group's chunk index base, a permutation of its documents
                full = saved.tables
                for b in batches:
                    ws_b = ws.batch_view(b)
                    tb = cake_backend._tables(ws_b, cu[b.d0 : b.d1 + 1], "light")
                    real_b = _real_chunks(lengths[b.d0 : b.d1])
                    assert torch.equal(tb[0][:real_b], full[0][b.c0 : b.c0 + real_b]), (
                        case,
                        b,
                    )
                    assert torch.equal(tb[1][:real_b], full[1][b.c0 : b.c0 + real_b]), (
                        case,
                        b,
                    )
                    assert torch.equal(
                        tb[2][: b.N + 1] + b.c0, full[2][b.d0 : b.d1 + 1]
                    ), (case, b)
                    assert sorted(tb[3][: b.N].tolist()) == list(range(b.N)), (case, b)
                    assert (
                        not tb[0][real_b : b.NC_pad].any()
                        and not tb[1][real_b : b.NC_pad].any()
                    )
                del ws


def test_batched_arena_on_a_row_that_saves_the_states(device, manifest) -> None:
    """A pack above the size rule (its light set carries the states) whose last document alone would be below it:
    the groups follow the whole problem's rule, the forward scatters their blocks into the saved states by chain
    rank, the backward gathers them (``light_states`` as pass 2 alone), everything bit for bit."""
    lengths, heads = (40000, 20000, 20000, 1000), 4
    S, N = sum(lengths), len(lengths)
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    p = cake_backend.plan(S, N, heads, sm, arch)
    assert p.recur == "seg" and p.seg_save and p.light_path == "light_states"
    assert not cake_backend.plan(1000, 1, heads, sm, arch).seg_save
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    out, grads, saved = _run_fwd_bwd(inputs)
    assert saved.hseg is not None and saved.seg_state is not None
    ref_out, ref_grads = _clone_all(out, grads)
    del out, grads
    from torch.utils.checkpoint import checkpoint

    for label, kw in _partitions(lengths):
        arena = cake_backend.KdaArena(lean=True, tail_passes=4, **kw)
        ws = KdaWorkspace.for_problem(
            S, N, heads, device, arena=arena, doc_lengths=lengths
        )
        tag = (label,)
        o, saved_b = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        assert _bits_equal(o, ref_out), tag
        assert _bits_equal(saved_b.hseg, saved.hseg) and _bits_equal(
            saved_b.seg_state, saved.seg_state
        ), tag
        assert all(
            torch.equal(a, b) for a, b in zip(saved_b.tables, saved.tables, strict=True)
        ), tag
        for name, source, space in (
            ("batched backward, default set", saved, ws),
            ("batched backward, batched set", saved_b, ws),
            ("default backward, batched set", saved_b, None),
        ):
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=source,
                workspace=space,
                doc_lengths=lengths if space is not None else None,
            )
            _check_bits(got, ref_grads, tag=(*tag, name))
        leaves = [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        o = checkpoint(
            lambda *a: cake_backend.chunk_kda(*a, cu, arena=arena, doc_lengths=lengths),
            *leaves,
            use_reentrant=False,
        )
        assert _bits_equal(o, ref_out), tag
        o.backward(dout)
        _check_bits(
            [leaf.grad for leaf in leaves],
            ref_grads,
            ("dq", "dk", "dv", "dg", "dbeta"),
            tag=(*tag, "checkpoint"),
        )
        del ws


def test_batched_arena_with_exact_zero_values(device, manifest) -> None:
    """Rows with exact zeros in ``v`` and ``dout`` (a zero token row, a zero chunk at a group's first chunk, a whole
    zero document at an odd chunk): the paired WY programs accumulate ``0 * partner`` for the other chunk of a
    pair, so a changed partner could only change the sign of an exactly-zero entry; the bit-pattern comparison of
    every output, over every grouping, shows whether it does."""
    for lengths, heads in (
        ((63, 1, 65, 30), 2),
        ((63, 1, 65), 4),
        ((100, 640, 77, 1000, 129), 2),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        v, dout = inputs["v"], inputs["dout"]
        cu = [0]
        for n in lengths:
            cu.append(cu[-1] + n)
        v[cu[1] : cu[2]] = (
            0  # the second document (one token at an odd chunk, or a whole document)
        )
        v[cu[2] : cu[2] + 6] = (
            0  # the first rows of the third document (a group's first chunk)
        )
        dout[cu[2] : cu[2] + 6] = 0
        v[cu[-1] - 7 : cu[-1]] = 0  # the pack's last rows (a pad partner's neighbour)
        dout[:5] = 0
        q, k, g, beta, cu_t = (inputs[n] for n in ("q", "k", "g", "beta", "cu_seqlens"))
        out, grads, saved = _run_fwd_bwd(inputs)
        ref_out, ref_grads = _clone_all(out, grads)
        del out, grads
        real = _real_chunks(lengths)
        tape_default = KdaWorkspace.for_problem(S, N, heads, device, save_mode="tape")
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu_t, workspace=tape_default
        )
        ref_h, ref_v_new = saved_t.h[:real].clone(), saved_t.v_new[: BT * real].clone()
        del tape_default
        for label, kw in _partitions(lengths):
            arena = cake_backend.KdaArena(lean=True, tail_passes=2, **kw)
            tag = (lengths, heads, label)
            ws = KdaWorkspace.for_problem(
                S, N, heads, device, arena=arena, doc_lengths=lengths
            )
            o, saved_b = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu_t, workspace=ws, doc_lengths=lengths
            )
            assert _bits_equal(o, ref_out), tag
            assert _bits_equal(saved_b.Aqk, saved.Aqk) and _bits_equal(
                saved_b.Akk, saved.Akk
            ), tag
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu_t,
                dout,
                saved=saved_b,
                workspace=ws,
                doc_lengths=lengths,
            )
            _check_bits(got, ref_grads, tag=tag)
            tws = KdaWorkspace.for_problem(
                S, N, heads, device, save_mode="tape", arena=arena, doc_lengths=lengths
            )
            o, saved_tb = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu_t, workspace=tws, doc_lengths=lengths
            )
            assert _bits_equal(o, ref_out), tag
            assert _bits_equal(saved_tb.h[:real], ref_h), tag
            assert _bits_equal(saved_tb.v_new[: BT * real], ref_v_new), tag
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu_t,
                dout,
                saved=saved_tb,
                workspace=tws,
                doc_lengths=lengths,
            )
            _check_bits(got, ref_grads, tag=(*tag, "tape"))
            del ws, tws


def test_batched_arena_under_checkpointing_and_a_dropped_workspace(
    device, manifest, monkeypatch
) -> None:
    """A checkpointed forward with the batched option retains nothing beyond its output; the backward's own
    workspace takes the recorded option and ``doc_lengths`` (sized for the exact groups); a dropped workspace is
    replaced the same way; a kept workspace of another option is refused; ``doc_lengths`` are required."""
    import gc

    from torch.utils.checkpoint import checkpoint

    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    cu, dout = inputs["cu_seqlens"], inputs["dout"]
    out, grads, _ = _run_fwd_bwd(inputs)
    reference_out, reference_grads = _clone_all(out, grads)
    del out, grads
    arena = cake_backend.KdaArena(lean=True, tail_passes=2, batch_tokens=1000)
    plan = cake_backend.plan(
        S, N, heads, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )
    assert [(b.d0, b.d1) for b in plan.batches(arena, lengths)] == [
        (0, 3),
        (3, 4),
        (4, 5),
    ]
    exact = plan.arena_bytes(arena=arena, doc_lengths=lengths)
    assert exact < plan.arena_bytes(arena=arena) < plan.arena_bytes()
    created: list = []
    original = cake_backend.KdaWorkspace.for_problem

    def spy(
        *args, **kwargs
    ):  # records the option, not the workspace: a strong reference would be "retained"
        ws_ = original(*args, **kwargs)
        created.append(
            types.SimpleNamespace(
                arena_option=ws_.arena_option,
                backward=ws_.backward,
                bytes=ws_.bytes,
                doc_lengths=ws_.doc_lengths,
            )
        )
        return ws_

    monkeypatch.setattr(cake_backend.KdaWorkspace, "for_problem", spy)

    def leaves():
        return [
            inputs[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

    def check(leaf_list, tag):
        _check_bits(
            [leaf.grad for leaf in leaf_list],
            reference_grads,
            ("dq", "dk", "dv", "dg", "dbeta"),
            tag=tag,
        )

    def implicit(q, k, v, g, beta):
        return cake_backend.chunk_kda(
            q, k, v, g, beta, cu, arena=arena, doc_lengths=lengths
        )

    held = leaves()
    warm = implicit(*held)
    assert _bits_equal(warm, reference_out)
    del warm
    gc.collect()
    assert _retained_after_checkpointed_forward(implicit, *held) == 0
    created.clear()
    o = checkpoint(implicit, *held, use_reentrant=False)
    o.backward(dout)
    check(held, "checkpoint")
    assert created and all(w.arena_option == arena for w in created)
    assert all(
        w.doc_lengths == tuple(lengths) and w.bytes == exact
        for w in created
        if w.backward
    )
    assert any(w.backward for w in created)
    dropped = KdaWorkspace(
        plan, device, backward=True, arena=arena, doc_lengths=lengths
    )
    created.clear()
    held = leaves()
    o = cake_backend.chunk_kda(*held, cu, workspace=dropped, doc_lengths=lengths)
    assert _bits_equal(o, reference_out)
    del dropped
    gc.collect()
    o.backward(dout)
    assert (
        len(created) == 1 and created[0].arena_option == arena and created[0].backward
    )
    assert created[0].bytes == exact
    check(held, "dropped")
    kept = KdaWorkspace(plan, device, backward=True)
    with pytest.raises(ValueError, match="arena="):
        cake_backend.chunk_kda(
            *leaves(), cu, workspace=kept, arena=arena, doc_lengths=lengths
        )
    with pytest.raises(ValueError, match="doc_lengths"):
        cake_backend.chunk_kda(*leaves(), cu, arena=arena)


def test_batched_views_share_the_arena_eye_and_epoch(device, manifest) -> None:
    """The views of one arena (its groups, ``with_plan``) share the bytes, the identity / flag buffer and the epoch
    counter; every group's layout fits the exact arena, and a packing whose group does not fit is refused before
    any launch."""
    lengths, heads = (63, 1, 65, 30), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
    )
    arena = cake_backend.KdaArena(batch_tokens=95)
    ws = KdaWorkspace.for_problem(S, N, heads, device, arena=arena, doc_lengths=lengths)
    p = ws.plan
    batches = p.batches(arena, lengths)
    assert [(b.d0, b.d1) for b in batches] == [(0, 2), (2, 4)]
    views = [ws.batch_view(b) for b in batches]
    assert views[0] is ws.batch_view(batches[0]) and views[1] is not views[0]
    for view in views:
        assert view.arena.data_ptr() == ws.arena.data_ptr() and view.bytes == ws.bytes
        assert view.eye.data_ptr() == ws.eye.data_ptr()
        assert view.flags.data_ptr() == ws.flags.data_ptr()
        assert view._epoch is ws._epoch and view.arena_option == arena
        for mode in ("fwd", "bwd", "replay"):
            view.views(mode)
    e = ws.next_epoch()
    assert views[0].next_epoch() == e + 1 and views[1].next_epoch() == e + 2
    assert ws.next_epoch() == e + 3
    twin = ws.with_plan(p)
    assert twin._epoch is ws._epoch and twin.next_epoch() == e + 4
    assert twin.eye.data_ptr() == ws.eye.data_ptr()
    # a workspace sized for one document per group refuses a packing whose group is larger, before any launch
    single = cake_backend.KdaArena(batch_tokens=S, max_docs_per_batch=1)
    small = KdaWorkspace.for_problem(
        S, N, heads, device, arena=single, doc_lengths=lengths
    )
    assert small.bytes < ws.bytes
    other = (S, 0, 0, 0)  # the same S and num_docs, one group of every token
    with pytest.raises(ValueError, match="needs"):
        cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=small, doc_lengths=other
        )
    # the bound sizes a workspace without doc_lengths: every grouping of the budget fits it
    bound = KdaWorkspace.for_problem(S, N, heads, device, arena=arena)
    assert bound.doc_lengths is None and bound.bytes >= ws.bytes
    for b in batches:
        bound.batch_view(b).views("bwd")


def test_batched_arena_content_never_reaches_a_result(device, manifest) -> None:
    """Write-before-read on the batched option too: a batched arena filled with NaN bytes before every call gives
    the zero-filled default arena's results bit for bit (the forward, the backward from the light and the tape
    sets, a light set on the tape workspace, the forward without a saved set), including a row whose light set
    carries the states (the gathers read the saved tensors, not the arena)."""
    cases = (
        ((63, 1, 65, 30), 2, "light", 1, {"batch_tokens": 95}),
        ((100, 640, 77, 1000, 129), 2, "light", 3, {"batch_tokens": 1000}),
        (
            (20, 20, 20, 20, 20, 20, 20),
            8,
            "light",
            2,
            {"batch_tokens": 140, "max_docs_per_batch": 3},
        ),
        ((63, 1, 65), 64, "light", 3, {"batch_tokens": 129, "max_docs_per_batch": 1}),
        ((100, 640, 77, 1000, 129), 2, "tape", 2, {"batch_tokens": 1000}),
        ((40000, 20000, 20000, 1000), 4, "light", 4, {"batch_tokens": 40000}),
    )
    for lengths, heads, save_mode, t, kw in cases:
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        plan = cake_backend.plan(
            S,
            N,
            heads,
            cake_jit.sm_count(device.index),
            cake_jit.device_arch(device.index),
        )
        zeroed = KdaWorkspace(
            plan, device, backward=True, zero_init=True, save_mode=save_mode
        )
        out, grads, saved = _run_fwd_bwd(inputs, workspace=zeroed)
        reference_out, reference_grads = _clone_all(out, grads)
        del out, grads
        arena = cake_backend.KdaArena(lean=t > 1, tail_passes=t, **kw)
        poisoned = KdaWorkspace(
            plan,
            device,
            backward=True,
            save_mode=save_mode,
            arena=arena,
            doc_lengths=lengths,
        )
        assert poisoned.zero_init is False and poisoned.bytes < zeroed.bytes
        assert len(plan.batches(arena, lengths)) >= 2

        def poison():
            poisoned.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
            torch.cuda.synchronize()

        case = (lengths, heads, save_mode, t)
        poison()
        o, saved_p = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=poisoned, doc_lengths=lengths
        )
        assert _bits_equal(o, reference_out), case
        assert _bits_equal(saved_p.Aqk, saved.Aqk) and _bits_equal(
            saved_p.Akk, saved.Akk
        ), case
        if saved.hseg is not None:
            assert _bits_equal(saved_p.hseg, saved.hseg) and _bits_equal(
                saved_p.seg_state, saved.seg_state
            ), case
        poison()
        got = cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=saved_p,
            workspace=poisoned,
            doc_lengths=lengths,
        )
        _check_bits(got, reference_grads, tag=case)
        poison()
        got = cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=saved,
            workspace=poisoned,
            doc_lengths=lengths,
        )
        _check_bits(got, reference_grads, tag=(*case, "default set"))
        if (
            save_mode == "tape"
        ):  # a version-3 set on the poisoned tape workspace: per-call h / v_new per group
            _, saved_light = cake_backend.chunk_kda_fwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                workspace=poisoned,
                save_mode="light",
                doc_lengths=lengths,
            )
            poison()
            got = cake_backend.chunk_kda_bwd(
                q,
                k,
                v,
                g,
                beta,
                cu,
                dout,
                saved=saved_light,
                workspace=poisoned,
                doc_lengths=lengths,
            )
            _check_bits(got, reference_grads, tag=(*case, "light set"))
        poison()
        nosave, _ = cake_backend.chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            save_for_backward=False,
            workspace=poisoned,
            doc_lengths=lengths,
        )
        assert _bits_equal(nosave, reference_out), case
        zeroed = poisoned = None  # (no del: the poison closure refers to the name)
