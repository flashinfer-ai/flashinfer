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

import dataclasses
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
# form below the fused threshold, and on sm_107a the 64-column form too -- launch wy_fwd_rows and fwdh_o_v2_rows /
# fwdh_o_v1_rows; the 64-column form keeps the internal rows on the other architectures)
TOKEN_ROW_RECORDS = (
    "fwdh_wy_rows",
    "wy_fwd_rows",
    "fwdh_p3_o_ov_rows",
    "fwdh_p3_o_ov_id_rows",
    "fwdh_o_v2_rows",
    "fwdh_o_v1_rows",
)
INTERNAL_ROW_TWIN = {
    "fwdh_wy_rows": "fwdh_wy",
    "wy_fwd_rows": "wy_fwd",
    "fwdh_p3_o_ov_rows": "fwdh_p3_o_ov",
    "fwdh_p3_o_ov_id_rows": "fwdh_p3_o_ov_id",
    "fwdh_o_v2_rows": "fwdh_o_v2",
    "fwdh_o_v1_rows": "fwdh_o_v1",
}
# the architectures whose saving forward reads the saved token rows on the 64-column serial rows too (fwdh_o_v1_rows)
PAIRED_V1_ARCHES = ("sm_107a",)
DAVQ_MIN_H = 16  # the fused gradient hosts dAqk on the serial route from 16 heads
# from this many heads the fused gradient runs with its L2 prefetch / evict-first hints off (the `_nohint` records:
# the same kernel built with the hint thresholds beyond any head count) on either route (the 16-head rows of
# sm_107a are segmented ones); the other architectures keep the hints
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
# The segmented recurrences: at most SEG_MAX_H heads, or a head count of SEG_PLAN_HEADS_BY_ARCH on its architecture
# (the 16-head rows of sm_103a / sm_107a: their serial walk is the slower form there; the registry's rules read the
# route, not the head bound), clamp(ceil(SEG_SLOT_FACTOR * sm / (N * H)), 1, SEG_P_MAX) segments per document, at
# most SEG_P_SMALL of them with N * H <= SEG_P_SMALL_MAX_NH (document, head) chains.
SEG_MAX_H = 8
SEG_PLAN_HEADS_BY_ARCH = {"sm_103a": (16,), "sm_107a": (16,)}
SEG_SLOT_FACTOR = 4
SEG_P_MAX = 64
SEG_P_SMALL = 38
SEG_P_SMALL_BY_ARCH = {
    "sm_100a": 37
}  # the few-chain count of the architectures with one of their own
SEG_P_SMALL_MAX_NH = 4
# The two-level segment carry (seg_carry_v8): the segments of each document's chain in groups of g, the groups'
# products formed in parallel and chained, the CTAs handing over through CARRY_FLAG_WORDS_PER_CTA release-flag
# words each under a fresh epoch.  Dispatched on every architecture (CARRY_V8_ARCHES) at most CARRY_V8_MAX_H heads,
# on the segmented rows that are not the forward's overlap rows (those keep seg_carry_ov), where the host rule
# accepts: the shortest critical path that fits one CTA per SM gains more than CARRY_S_HOPS hops over the chain
# (_carry_choice).
CARRY_V8_ARCHES = ARCHES
CARRY_V8_MAX_H = 2
CARRY_S_HOPS = 14
CARRY_FLAG_WORDS_PER_CTA = 3
# the architectures whose GLM intra program (glm_intra_v14 on sm_103a, glm_intra_v18 on sm_107a: the same program
# with every named-barrier slot an immediate, two CTAs per SM there) writes the gate gradients itself: no gate_epilogue
# launch, no dg_int / db_int buffers in their layouts; sm_100a keeps glm_intra + gate_epilogue.  A byte query
# without an architecture takes the first exported one's answer.
GATE_FUSED_ARCHES = ("sm_103a", "sm_107a")
GATE_FUSED_RECORD = {"sm_103a": "glm_intra_v14", "sm_107a": "glm_intra_v18"}
# The light saved set (version 4) carries the fold's qg on every row (the light backward's prep, bwd_prep_g8_noqg,
# writes none) and the WY representation w / u on the paired serial WU_SAVED_H-head rows (the 32-column form) of
# WU_SAVED_ARCHES and on the segmented WU_SAVED_H-head rows of WU_SEG_SAVED_ARCHES above WU_SEG_MIN_CHAINS (document,
# head) chains (the rows above the fused saved-state pass 2's bound): their prep is bwd_prep_g8_noqg_nokb on both
# light paths and their light backward launches no wy program.
WU_SAVED_ARCHES = ("sm_100a",)
WU_SAVED_H = 16
WU_SEG_SAVED_ARCHES = ("sm_103a", "sm_107a")
WU_SEG_MIN_CHAINS = 32
# the table stage is optional too: on the forward and the replay of the segmented route up to 256 documents the
# intra program writes the tables and no table record is selected; the gate epilogue launches on sm_107a alone
OPTIONAL_STAGES = (
    "table",
    "seg_table",
    "seg_carry",
    "fwdh_p3",
    "wy_fwd",
    "wy",
    "fwdh",
    "dav",
    "gate_epilogue",
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
# the backward's chunk-parallel tail (once per range of a lean workspace's tail passes); the gate epilogue where
# the architecture dispatches it
TAIL_STAGES = ("dqkg", "glm_intra", "gate_epilogue")

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


def _current_arch() -> str:
    """The architecture :func:`cake_backend.plan` takes without ``arch``: the current CUDA device's where it is an
    exported target, else the first exported architecture."""
    if torch.cuda.is_available():
        arch = cake_jit.arch_for_capability(
            torch.cuda.get_device_capability(torch.cuda.current_device())
        )
        if arch is not None:
            return arch
    return ARCHES[0]


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
    "route_in",
    "carry_two_level",
    "carry_chain",
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
    "carry_groups",
    "segments_heads",
)
# the records delivered for a subset of the architectures (every other record ships on all three, the two-level
# carry and the no-kb prep among them): the gate epilogue and its unfused GLM intra program on sm_100a, the fused
# ones on sm_103a (glm_intra_v14) and sm_107a (glm_intra_v18)
RECORD_ARCHES = {
    "glm_intra": ("sm_100a",),
    "glm_intra_v14": ("sm_103a",),
    "glm_intra_v18": ("sm_107a",),
    "gate_epilogue": ("sm_100a",),
}
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


def _gate_fused(arch: str | None) -> bool:
    """The backend's rule, spelled independently: whether the GLM intra program of ``arch`` fuses the gate
    epilogue (``arch=None``: the first exported architecture's answer, as the byte queries take it)."""
    return (ARCHES[0] if arch is None else arch) in GATE_FUSED_ARCHES


def _wu_saved(n: int, h: int, sm: int, arch: str | None) -> bool:
    """Whether the light set of the row carries ``w`` / ``u``: the serial route at ``WU_SAVED_H`` heads in the
    32-column form on ``WU_SAVED_ARCHES``, the segmented route at ``WU_SAVED_H`` heads above ``WU_SEG_MIN_CHAINS``
    (document, head) chains on ``WU_SEG_SAVED_ARCHES`` (``arch=None``: the first exported architecture)."""
    arch = ARCHES[0] if arch is None else arch
    route = _expected_segments(n, h, sm, arch)[0]
    if h != WU_SAVED_H:
        return False
    if route == "serial":
        return (
            n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
            and arch in WU_SAVED_ARCHES
        )
    return n * h > WU_SEG_MIN_CHAINS and arch in WU_SEG_SAVED_ARCHES


def _carry_ctas(p: int, g: int) -> int:
    """CTAs per (document, head) chain of the two-level carry over ``p`` segments in groups of ``g``: two for a
    single group, else four per group less the four of the chain's ends."""
    groups = -(-p // g)
    return 2 if groups == 1 else 4 * groups - 4


def _carry_hops(p: int, g: int) -> int:
    """The critical path of the two-level carry in hop units: the chain's ``p - 1`` for a single group, else the
    longer of the group products' path and the last (shorter) group's."""
    groups = -(-p // g)
    if groups == 1:
        return p - 1
    last = p - (groups - 1) * g
    return max(2 * g + groups - 4, g + groups - 3 + last)


def _carry_choice(
    n: int, h: int, sm: int, nseg: int, s_hops: int = CARRY_S_HOPS
) -> tuple[int, int] | None:
    """The two-level carry's host rule, spelled independently of the registry: ``(g, grid)`` or ``None``.  Over
    the group sizes ``2..P`` the shortest critical path whose grid fits one CTA per SM (ties: the smaller grid);
    the grid is ``n * h * ctas`` CTAs, or on a proportional block table (at most ``FEW_CHAINS`` chains) the bound
    ``h * (4 * (nseg // g) + 2 n)``; taken only when it gains more than ``s_hops`` over the chain's ``P - 1``."""
    if n < 1 or nseg <= 0 or nseg % n:
        return None
    p = nseg // n
    if p < 2:
        return None
    proportional = n * h <= FEW_CHAINS
    best: tuple[int, int, int] | None = None
    for g in range(2, p + 1):
        if proportional:
            grid = h * (4 * (nseg // g) + 2 * n)
        else:
            grid = n * h * _carry_ctas(p, g)
        if grid > sm:
            continue
        hops = _carry_hops(p, g)
        if best is None or (hops, grid) < best[:2]:
            best = (hops, grid, g)
    if best is None or best[0] + s_hops >= p - 1:
        return None
    return best[2], best[1]


def _route_rule(route: str) -> dict:
    return {"kind": "route_in", "routes": [route]}


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.MODULES) == 67
    assert len({r["library"] for r in cake_jit.MODULES.values()}) == 65
    assert tuple(cake_jit.PATHS) == PATHS
    assert tuple(cake_jit.OPTIONAL_STAGES) == OPTIONAL_STAGES
    assert {
        "max_heads": SEG_MAX_H,
        "slot_factor": SEG_SLOT_FACTOR,
        "max_segments": SEG_P_MAX,
        "small_segments": SEG_P_SMALL,
        "small_segments_by_arch": SEG_P_SMALL_BY_ARCH,
        "small_max_chains": SEG_P_SMALL_MAX_NH,
        "prop_max_chains": FEW_CHAINS,
        "plan_heads_by_arch": {
            arch: list(heads) for arch, heads in SEG_PLAN_HEADS_BY_ARCH.items()
        },
    } == cake_jit.SEGMENTS
    assert cake_jit.CARRY_FLAG_WORDS_PER_CTA == CARRY_FLAG_WORDS_PER_CTA
    # the two-level carry's operands and group size extend the logical vocabulary next to the chain carry's
    tensor_names, scalar_names = (
        tuple(cake_jit.LOGICAL_TENSORS),
        tuple(cake_jit.LOGICAL_SCALARS),
    )
    assert tensor_names.index("carry_out_tiles") == tensor_names.index("carry_out") + 1
    assert scalar_names.index("gsize") == scalar_names.index("epoch") + 1
    stages = set()
    tensors, scalars = set(tensor_names), set(scalar_names)
    for name, record in cake_jit.MODULES.items():
        assert record["stage"] in cake_jit.STAGES, name
        stages.add(record["stage"])
        expected_name = (
            record["stage"]
            if record["variant"] is None
            else f"{record['stage']}_{record['variant']}"
        )
        assert name == expected_name, name
        # every record ships on all three architectures but the few of RECORD_ARCHES, each on an ordered subset
        assert list(record["arches"]) == list(RECORD_ARCHES.get(name, ARCHES)), name
        assert [a for a in ARCHES if a in record["arches"]] == list(record["arches"])
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
    # the segmented records read the route (segment_plan's: the head bound, or a plan head of the architecture);
    # the carry's records split the forward's overlap rows (the pair: the two device families, at least the wave
    # threshold, at most the flag words) from their complement (every backward path, the third family, below the
    # threshold, above the bound), which the two-level carry and the chain carry share by their own rule
    seg_rule, serial_rule = _route_rule("seg"), _route_rule("serial")
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
        "rules": [serial_rule, docs_rule],
    }
    assert cake_jit.MODULES["table"]["select_rule"] == above_rule
    assert cake_jit.MODULES["fwd_intra_v14"]["select_rule"] == {
        "kind": "all_of",
        "rules": [seg_rule, docs_rule],
    }
    assert cake_jit.MODULES["fwd_intra"]["select_rule"] == {
        "kind": "any_of",
        "rules": [serial_rule, above_rule],
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
    # the backward paths and the forward's non-overlap rows: the two-level carry on every architecture at most
    # CARRY_V8_MAX_H heads where its host rule accepts, the chain carry everywhere else (the two rules partition)
    not_overlap = {
        "kind": "any_of",
        "rules": [
            {
                "kind": "path_in",
                "paths": ["light", "light_states", "tape", "replay"],
            },
            waves("walk_ctas_below"),
            {"kind": "segments_heads_above", "min": OV1_MAX_SEGMENT_HEADS},
        ],
    }
    assert cake_jit.MODULES["seg_carry"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            not_overlap,
            {
                "kind": "any_of",
                "rules": [
                    {"kind": "heads_above", "min": CARRY_V8_MAX_H},
                    {"kind": "carry_chain", "s_hops": CARRY_S_HOPS},
                ],
            },
        ],
    }
    assert cake_jit.MODULES["seg_carry_v8"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            seg_rule,
            not_overlap,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_most", "max": CARRY_V8_MAX_H},
                    {"kind": "carry_two_level", "s_hops": CARRY_S_HOPS},
                ],
            },
        ],
    }
    assert cake_jit.MODULES["seg_carry_v8"]["grid_rule"] == {
        "kind": "carry_groups",
        "s_hops": CARRY_S_HOPS,
    }
    # the backward prep: the tape set's form (it writes qg), the replay's, and the light sets' no-qg forms (the
    # version-4 set carries the forward's qg): the no-kb one, on both light paths, on the rows whose set carries w / u
    # (the paired serial rows of WU_SAVED_ARCHES and the segmented rows of WU_SEG_SAVED_ARCHES above WU_SEG_MIN_CHAINS
    # chains), the other elsewhere
    light = {"kind": "path_in", "paths": ["light"]}
    light_states = {"kind": "path_in", "paths": ["light_states"]}
    slots_fit = {
        "kind": "slots_at_most",
        "slices": RECUR_SLICES["v2"],
        "ctas_per_sm": RECUR_CTAS_PER_SM,
    }
    slots_over = {
        "kind": "slots_above",
        "slices": RECUR_SLICES["v2"],
        "ctas_per_sm": RECUR_CTAS_PER_SM,
    }
    wu_rows = {
        "kind": "all_of",
        "rules": [
            serial_rule,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_least", "min": WU_SAVED_H},
                    {"kind": "heads_at_most", "max": WU_SAVED_H},
                ],
            },
            slots_fit,
            {"kind": "arch_in", "arches": list(WU_SAVED_ARCHES)},
        ],
    }
    not_wu_rows = {
        "kind": "any_of",
        "rules": [
            seg_rule,
            {"kind": "heads_below", "max": WU_SAVED_H},
            {"kind": "heads_above", "min": WU_SAVED_H},
            slots_over,
            {
                "kind": "arch_in",
                "arches": [a for a in ARCHES if a not in WU_SAVED_ARCHES],
            },
        ],
    }
    seg_wu_rows = {
        "kind": "all_of",
        "rules": [
            seg_rule,
            {
                "kind": "all_of",
                "rules": [
                    {"kind": "heads_at_least", "min": WU_SAVED_H},
                    {"kind": "heads_at_most", "max": WU_SAVED_H},
                ],
            },
            {"kind": "chains_above", "min": WU_SEG_MIN_CHAINS},
            {"kind": "arch_in", "arches": list(WU_SEG_SAVED_ARCHES)},
        ],
    }
    not_seg_wu_rows = {
        "kind": "any_of",
        "rules": [
            serial_rule,
            {"kind": "heads_below", "max": WU_SAVED_H},
            {"kind": "heads_above", "min": WU_SAVED_H},
            {"kind": "chains_at_most", "max": WU_SEG_MIN_CHAINS},
            {
                "kind": "arch_in",
                "arches": [a for a in ARCHES if a not in WU_SEG_SAVED_ARCHES],
            },
        ],
    }
    light_paths = {"kind": "path_in", "paths": ["light", "light_states"]}
    # the serial rows' form is gated to the light path (the serial route never saves the states); its complement
    # in the kb-writing form admits the states-saving set
    wu_rows_light = {"kind": "all_of", "rules": [light, *wu_rows["rules"]]}
    not_wu_rows_or_states = {
        "kind": "any_of",
        "rules": [light_states, *not_wu_rows["rules"]],
    }
    assert cake_jit.MODULES["bwd_prep_g8"]["select_rule"] == {
        "kind": "path_in",
        "paths": ["tape"],
    }
    assert cake_jit.MODULES["bwd_prep_g8_replay"]["select_rule"] == {
        "kind": "path_in",
        "paths": ["replay"],
    }
    assert cake_jit.MODULES["bwd_prep_g8_noqg"]["select_rule"] == {
        "kind": "all_of",
        "rules": [light_paths, not_wu_rows_or_states, not_seg_wu_rows],
    }
    assert cake_jit.MODULES["bwd_prep_g8_noqg_nokb"]["select_rule"] == {
        "kind": "all_of",
        "rules": [
            light_paths,
            {"kind": "any_of", "rules": [wu_rows_light, seg_wu_rows]},
        ],
    }
    # the backward's WY program: every tape set; the light sets but the fused 64-column recompute's rows and the
    # rows whose set carries w / u; the saved-state sets above 32 chains (the fused pass 2 forms it below) but,
    # again, the segmented rows whose set carries w / u
    assert cake_jit.MODULES["wy_bwd"]["select_rule"] == {
        "kind": "any_of",
        "rules": [
            {"kind": "path_in", "paths": ["tape"]},
            {
                "kind": "all_of",
                "rules": [
                    light,
                    {
                        "kind": "any_of",
                        "rules": [
                            {"kind": "heads_below", "max": FUSED_BWD_MIN_H},
                            slots_fit,
                        ],
                    },
                    not_wu_rows,
                    not_seg_wu_rows,
                ],
            },
            {
                "kind": "all_of",
                "rules": [
                    light_states,
                    {"kind": "chains_above", "min": FUSED_P3_MAX_CHAINS},
                    not_seg_wu_rows,
                ],
            },
        ],
    }
    # the GLM intra program and the gate epilogue: fused on sm_103a (glm_intra_v14) and sm_107a (glm_intra_v18), two
    # launches on sm_100a
    unfused = [a for a in ARCHES if a not in GATE_FUSED_ARCHES]
    for arch, fused_record in GATE_FUSED_RECORD.items():
        assert cake_jit.MODULES[fused_record]["select_rule"] == {
            "kind": "arch_in",
            "arches": [arch],
        }
    assert cake_jit.MODULES["glm_intra"]["select_rule"] == {
        "kind": "arch_in",
        "arches": unfused,
    }
    assert cake_jit.MODULES["gate_epilogue"]["select_rule"] == {
        "kind": "arch_in",
        "arches": unfused,
    }
    for name in ("glm_intra", "glm_intra_v14", "glm_intra_v18", "gate_epilogue"):
        assert cake_jit.MODULES[name]["grid_rule"] == {"kind": "chunks", "x_factor": 1}
    # the fused gradient's dAqk-hosting form: the serial route from 16 heads and every segmented row, with the
    # hints on sm_100a / sm_103a or below 16 heads; the sm_107a rows from 16 heads take the _nohint build on
    # either route
    nohint_h = DQKG_NOHINT_MIN_H_BY_ARCH["sm_107a"]
    hinted = {
        "kind": "any_of",
        "rules": [
            {"kind": "arch_in", "arches": ["sm_100a", "sm_103a"]},
            {"kind": "heads_below", "max": nohint_h},
        ],
    }
    for name, paths in (
        ("dqkg_davq", ["light", "light_states", "tape"]),
        ("dqkg_int_davq", ["replay"]),
    ):
        assert cake_jit.MODULES[name]["select_rule"] == {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": paths},
                {
                    "kind": "any_of",
                    "rules": [
                        {
                            "kind": "all_of",
                            "rules": [
                                serial_rule,
                                {"kind": "heads_at_least", "min": DAVQ_MIN_H},
                                hinted,
                            ],
                        },
                        {"kind": "all_of", "rules": [seg_rule, hinted]},
                    ],
                },
            ],
        }, name
        assert cake_jit.MODULES[name + "_nohint"]["select_rule"] == {
            "kind": "all_of",
            "rules": [
                {"kind": "path_in", "paths": paths},
                {"kind": "arch_in", "arches": ["sm_107a"]},
                {"kind": "heads_at_least", "min": nohint_h},
            ],
        }, name
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
    plan_heads = SEG_PLAN_HEADS_BY_ARCH.get(arch, ()) if arch is not None else ()
    if h <= SEG_MAX_H or h in plan_heads:
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
    route, _p, nseg = _expected_segments(n, h, sm, arch)
    seg = route == "seg"
    v2 = n * h * RECUR_SLICES["v2"] <= sm * RECUR_CTAS_PER_SM
    few = n * h <= FEW_CHAINS
    fused_p3 = seg and n * h <= FUSED_P3_MAX_CHAINS
    fused_fwd = (
        not seg and h >= FUSED_FWD_MIN_H and (arch != "sm_107a" or h <= FUSED_FWD_MIN_H)
    )
    fused_bwd = not seg and not v2 and h >= FUSED_BWD_MIN_H
    # the rows whose light set carries w / u: no wy launch on their light paths, a prep that writes no kb
    wu = _wu_saved(n, h, sm, arch)
    # the fused gradient hosts dAqk on the serial rows from 16 heads and on every segmented row (whose adjoint
    # recurrence accumulates dv): no dav launch there
    davq = (h >= DAVQ_MIN_H) if not seg else True
    replay = path == "replay"
    fwd = path in ("fwd", "fwd_nosave")
    # the saving forward reads the saved token rows where a program for them ships
    save = path == "fwd"
    # the paired serial rows: the 32-column form below the fused threshold reads the saved token rows on the saving
    # forward (wy_fwd_rows + fwdh_o_v2_rows); the 64-column form does so on sm_107a alone (wy_fwd_rows +
    # fwdh_o_v1_rows) and keeps the internal-row pair on both paths elsewhere
    paired = save and not seg and v2 and not fused_fwd
    paired_v1 = (
        save and not seg and not v2 and not fused_fwd and arch in PAIRED_V1_ARCHES
    )
    walk = nseg * h
    prefetch = walk < DHU_PREFETCH_WAVES_BY_ARCH.get(arch, 0.0) * sm
    tma = walk < DHU_TMA_WAVES * sm or (h <= 4 and walk <= DHU_TMA_WAVES * sm)
    ov1 = (
        seg
        and fwd
        and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
        and walk <= OV1_MAX_SEGMENT_HEADS
    )
    # the two-level carry on every architecture at most two heads where its host rule accepts (the forward's
    # overlap rows keep the pair's carry); the chain carry everywhere else
    two_level = (
        seg
        and arch in CARRY_V8_ARCHES
        and h <= CARRY_V8_MAX_H
        and _carry_choice(n, h, sm, nseg) is not None
    )
    gate_fused = _gate_fused(arch)
    if not seg:
        carry = None
    elif ov1:
        carry = "seg_carry_ov"
    else:
        carry = "seg_carry_v8" if two_level else "seg_carry"
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
        else ("wy_fwd_rows" if (ov1 and save) or paired or paired_v1 else "wy_fwd"),
        "seg_carry": carry,
        "glm_intra": GATE_FUSED_RECORD[arch] if gate_fused else "glm_intra",
        "gate_epilogue": None if gate_fused else "gate_epilogue",
    }
    if path == "tape":
        out["wy"] = "wy_bwd"
    elif path == "light":
        out["wy"] = None if (fused_bwd or wu) else "wy_bwd"
    elif path == "light_states":
        out["wy"] = None if (fused_p3 or wu) else "wy_bwd"
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
            out["fwdh"] = "fwdh_o_v1_rows" if paired_v1 else "fwdh_o_v1"
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
    # the backward prep: the tape set's form writes qg, the replay's the internal dout copy too; the light sets
    # carry the forward's qg (the no-qg forms) and, on the w / u rows, its WY representation (no kb either)
    if path == "tape":
        out["bwd_prep"] = "bwd_prep_g8"
    elif replay:
        out["bwd_prep"] = "bwd_prep_g8_replay"
    elif path in ("light", "light_states") and wu:
        out["bwd_prep"] = "bwd_prep_g8_noqg_nokb"
    elif path in ("light", "light_states"):
        out["bwd_prep"] = "bwd_prep_g8_noqg"
    else:
        out["bwd_prep"] = None
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
    nohint = davq and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
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
        "bwd_prep_g8_noqg",
        "bwd_prep_g8_noqg_nokb",
        "glm_intra",
        "glm_intra_v14",
        "glm_intra_v18",
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
    if name == "seg_carry_v8":  # the two-level carry's grid: its host rule's CTA count
        choice = _carry_choice(n, h, sm, nseg)
        assert choice is not None, (name, n, h, sm, nseg)
        return (choice[1], 1, 1)
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
                    seg = _expected_segments(n, h, sm, arch)[0] == "seg"
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
                    paired_v1 = (
                        not seg
                        and not fused
                        and n * h * RECUR_SLICES["v2"] > sm * RECUR_CTAS_PER_SM
                        and arch in PAIRED_V1_ARCHES
                    )
                    assert bool(token) == (fused or ov1 or paired or paired_v1), (
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
                    if paired_v1:
                        assert token == {"wy_fwd_rows", "fwdh_o_v1_rows"}
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
    p = cake_backend.plan(1024, 4, 1, 148, ARCHES[0])  # the sm_100a rows of record
    assert (p.proportional, p.seg_arg) == (1, -4)
    p = cake_backend.plan(1024, 7, 8, 148, ARCHES[0])
    assert (p.proportional, p.seg_arg) == (0, p.P)


def test_the_16_head_rows_of_two_architectures_take_the_segmented_route() -> None:
    # SEGMENTS["plan_heads_by_arch"]: on sm_103a / sm_107a the 16-head rows take the segmented route (their serial
    # walk is the slower form there); sm_100a and a query without an architecture keep the serial walk from nine
    # heads on, and every other head count above the bound stays serial everywhere.
    for sm in (132, 148, 160, 212):
        for n in (1, 2, 7, 300):
            assert cake_jit.segment_plan(n, 16, sm) == ("serial", 1, 0)
            assert cake_jit.segment_plan(n, 16, sm, "sm_100a") == ("serial", 1, 0)
            assert cake_jit.segment_args(n, 16, sm, "sm_100a") == (0, 1)
            for arch in ("sm_103a", "sm_107a"):
                p = min(SEG_P_MAX, max(1, -(-SEG_SLOT_FACTOR * sm // (n * 16))))
                p = min(p, sm // 16)  # one walk CTA per SM per document
                assert cake_jit.segment_plan(n, 16, sm, arch) == ("seg", p, n * p)
                assert cake_jit.segment_args(n, 16, sm, arch) == (0, p)
            for arch in ARCHES:
                for h in (9, 12, 24, 32, 64):
                    assert cake_jit.segment_plan(n, h, sm, arch) == ("serial", 1, 0)
    assert cake_jit.segment_plan(1, 16, 148, "sm_103a") == ("seg", 9, 9)
    assert cake_jit.segment_plan(2, 16, 148, "sm_107a") == ("seg", 9, 18)
    # the route_in rules follow: the same 16-head row runs the segmented programs on sm_103a and the serial
    # 32-column ones on sm_100a (the plan heads bind route_in, not the heads_* leaves)
    kw = dict(num_chunks_pad=64, num_seqs=1, num_heads=16, sm_count=148)
    want = {
        "sm_100a": ("table_v2", "fwdh_v2h", "dhu_v2t", "fwd_intra", "dqkg_davq"),
        "sm_103a": ("table_v3", "fwdh_v3", "dhu_v3p_f", "fwd_intra_v14", "dqkg_davq"),
        "sm_107a": (
            "table_v3",
            "fwdh_v3",
            "dhu_v3_f",
            "fwd_intra_v14",
            "dqkg_davq_nohint",
        ),
    }
    for arch, (table, fwdh, dhu, intra, dqkg) in want.items():
        assert cake_jit.select_record("table", **kw, arch=arch, path="light") == table
        assert cake_jit.select_record("fwdh", **kw, arch=arch, path="light") == fwdh
        assert cake_jit.select_record("dhu", **kw, arch=arch, path="light") == dhu
        assert cake_jit.select_record("fwd_intra", **kw, arch=arch, path="fwd") == intra
        assert cake_jit.select_record("dqkg", **kw, arch=arch, path="light") == dqkg
        p = cake_backend.plan(4096, 1, 16, 148, arch)
        assert (p.recur, p.P, p.NSEG) == (
            ("serial", 1, 0) if arch == "sm_100a" else ("seg", 9, 9)
        )
        assert ("light_states" in p.records) is (arch != "sm_100a")
    # arch=None is the current device's plan (the first exported architecture's without one)
    assert (
        cake_backend.plan(4096, 1, 16, 148).recur
        == cake_backend.plan(4096, 1, 16, 148, _current_arch()).recur
    )


def test_carry_choice_picks_the_shortest_critical_path_that_fits() -> None:
    choice = cake_jit.carry_choice
    # eight chains at 64 segments per document on 148 SMs (the uniform grid n * h * (4 G - 4)): groups of 13
    # (G = 5, 16 CTAs per chain, 128 CTAs) give the shortest critical path, 27 hops against the chain's 63
    assert choice(4, 2, 148, 256, 14) == (13, 128)
    assert (_carry_hops(64, 13), _carry_ctas(64, 13)) == (27, 16)
    # six chains at 64: groups of 10 (G = 7, 24 CTAs per chain: 144 CTAs fit 148 SMs, 23 hops); where they do not
    # fit (132 SMs) groups of 11 (G = 6, 120 CTAs, 24 hops); with 212 SMs groups of 8 (G = 8, 168 CTAs)
    assert choice(3, 2, 148, 192, 14) == (10, 144)
    assert choice(3, 2, 132, 192, 14) == (11, 120)
    assert choice(3, 2, 212, 192, 14) == (8, 168)
    # the proportional rows (at most four chains): the grid bound h * (4 * (NSEG // g) + 2 n); a tie on hops goes
    # to the smaller grid (one chain at 37 segments: g = 5 and g = 4 both take 14 hops, 30 against 38 CTAs)
    assert _carry_hops(37, 5) == _carry_hops(37, 4) == 14
    assert choice(1, 1, 148, 37, 14) == (5, 30)
    assert choice(1, 1, 148, 38, 14) == (5, 30)
    assert choice(1, 2, 148, 37, 14) == (5, 60)
    assert choice(2, 2, 148, 76, 14) == (5, 128)  # g = 4 would need 160 CTAs
    assert choice(2, 2, 148, 74, 14) == (5, 120)
    assert choice(4, 1, 148, 148, 14) == (5, 124)
    assert choice(4, 1, 148, 152, 14) == (5, 128)
    # the gain rule: 16 chains at 37 segments: the best two-level path (g = 13: 25 hops, 128 CTAs) gains 11 hops
    # over the chain's 36, less than the 14 level handoffs; with a cheaper handoff it is taken
    assert choice(8, 2, 148, 296, 14) is None
    assert choice(8, 2, 148, 296, 10) == (13, 128)
    # the fit guard: 80 chains at 8 segments: even the single group (two CTAs per chain) needs 160 CTAs; where it
    # fits, a single group never gains (its critical path is the chain's)
    assert choice(40, 2, 148, 320, 14) is None
    assert choice(40, 2, 160, 320, 0) is None
    # the rule accepts above two heads too (one document at 4 heads, 37 segments: g = 5, 120 CTAs); the records
    # bound the head count, not the rule
    assert choice(1, 4, 148, 37, 14) == (5, 120)
    # the serial route (no segments), fewer than two segments per document, a count that is no multiple
    assert choice(4, 2, 148, 0, 14) is None
    assert choice(4, 2, 148, 4, 14) is None
    assert choice(4, 2, 148, 6, 14) is None
    # the independent spelling agrees over a grid of problems and handoff costs
    for n, h, sm, p in itertools.product(
        (1, 2, 3, 4, 7, 8, 40),
        (1, 2, 4),
        (132, 148, 160, 212),
        (1, 2, 5, 9, 37, 38, 64),
    ):
        for s_hops in (0, 10, 14):
            assert choice(n, h, sm, n * p, s_hops) == _carry_choice(
                n, h, sm, n * p, s_hops
            ), (n, h, sm, p, s_hops)


def test_flag_words_count_three_per_cta() -> None:
    assert cake_jit.CARRY_FLAG_WORDS_PER_CTA == 3
    for grid in (1, 30, 128, 148, 216):
        assert cake_jit.flag_words(grid) == 3 * grid
    # every grid the rule can pick (one CTA per SM at most) fits the identity buffer's words a workspace views as
    # the flags
    for sm in (132, 148, 160, 212, 216):
        assert cake_jit.flag_words(sm) <= D * D


def test_the_two_level_carry_dispatches_on_every_architecture_at_most_two_heads() -> (
    None
):
    # The two-level carry takes the backward paths and the forward's non-overlap rows on every architecture at one
    # or two heads where its rule accepts; never above two heads, never on the forward's overlap rows
    # (seg_carry_ov); the chain carry where the rule rejects.
    sm = 148
    accepted = ((4, 2), (3, 2), (1, 1), (2, 2), (4, 1), (1, 2))
    for n, h in accepted:
        for arch in ARCHES:
            _route, _p, nseg = _expected_segments(n, h, sm, arch)
            assert _carry_choice(n, h, sm, nseg) is not None, (n, h, arch)
            for path in PATHS:
                walk = nseg * h
                ov1 = (
                    path in ("fwd", "fwd_nosave")
                    and walk >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                    and walk <= OV1_MAX_SEGMENT_HEADS
                )
                if ov1:
                    want = "seg_carry_ov"
                elif arch in CARRY_V8_ARCHES:
                    want = "seg_carry_v8"
                else:
                    want = "seg_carry"
                got = cake_jit.select_record(
                    "seg_carry",
                    num_chunks_pad=64,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                assert got == want, (n, h, arch, path)
    # above two heads the chain carry although the rule accepts; where the rule rejects (16 chains at 37
    # segments) the chain carry on every architecture; the overlap pair on the forward of the latter row
    assert _carry_choice(1, 4, sm, 37) is not None
    assert _carry_choice(8, 2, sm, 296) is None
    for n, h in ((1, 4), (8, 2)):
        for arch in ARCHES:
            for path in PATHS:
                nseg = _expected_segments(n, h, sm, arch)[2]
                ov1 = path in ("fwd", "fwd_nosave") and (
                    nseg * h >= OV1_MIN_WAVES_BY_ARCH[arch] * sm
                )
                got = cake_jit.select_record(
                    "seg_carry",
                    num_chunks_pad=64,
                    num_seqs=n,
                    num_heads=h,
                    sm_count=sm,
                    arch=arch,
                    path=path,
                )
                want = "seg_carry_ov" if ov1 else "seg_carry"
                assert got == want, (n, h, arch, path)
    # the plans: the group size where a path dispatches the two-level carry, zero otherwise
    p = cake_backend.plan(4096, 4, 2, sm, "sm_100a")
    assert p.gsize == 13 and p.grid("seg_carry", "light") == (128, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_ov"
    for path in ("light", "light_states", "tape", "replay"):
        assert p.records[path]["seg_carry"] == "seg_carry_v8"
        assert p.launches("seg_carry", path) == 1
    # sm_103a: the same 64 segments per document, the same choice
    p = cake_backend.plan(4096, 4, 2, sm, "sm_103a")
    assert p.gsize == 13 and p.grid("seg_carry", "light") == (128, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_ov"
    for path in ("light", "light_states", "tape", "replay"):
        assert p.records[path]["seg_carry"] == "seg_carry_v8"
    p = cake_backend.plan(4096, 1, 1, sm, "sm_107a")  # 38 segments: g = 5, 30 CTAs
    assert p.gsize == 5 and p.grid("seg_carry", "fwd") == (30, 1, 1)
    assert p.records["fwd"]["seg_carry"] == "seg_carry_v8"
    p = cake_backend.plan(
        4096, 1, 1, sm, "sm_100a"
    )  # 37 segments on sm_100a: the same choice
    assert p.gsize == 5 and p.grid("seg_carry", "light") == (30, 1, 1)
    assert cake_backend.plan(4096, 8, 2, sm, "sm_100a").gsize == 0
    assert cake_backend.plan(4096, 1, 4, sm, "sm_107a").gsize == 0
    assert cake_backend.plan(4096, 1, 16, sm, "sm_107a").gsize == 0


def test_carry_groups_grid_follows_the_rule_or_a_pinned_group_size() -> None:
    def grid(**kw):
        return cake_jit.launch_grid("seg_carry_v8", num_chunks_pad=64, **kw)

    whole = dict(num_seqs=4, num_heads=2, sm_count=148, num_segments=256)
    assert grid(**whole) == (128, 1, 1)
    assert grid(**whole, gsize=0) == (128, 1, 1)
    assert grid(**whole, gsize=13) == (128, 1, 1)
    # a pinned group size (a document group of a batched problem keeps the whole problem's): the uniform grid
    # num_seqs * num_heads * ctas(P, gsize) at the group's chain count, P segments per document
    for n in (1, 2, 3):
        assert grid(
            num_seqs=n, num_heads=2, sm_count=148, num_segments=64 * n, gsize=13
        ) == (n * 2 * _carry_ctas(64, 13), 1, 1)
    # the rule's rejection (16 chains at 37 segments) is the grid's
    with pytest.raises(ValueError, match="rejects"):
        grid(num_seqs=8, num_heads=2, sm_count=148, num_segments=296)
    # the other grid kinds ignore the group size
    assert cake_jit.launch_grid(
        "seg_carry",
        num_chunks_pad=4,
        num_seqs=4,
        num_heads=2,
        sm_count=148,
        num_segments=256,
        gsize=13,
    ) == (16, 1, 1)


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
        # plan(arch=None) is the current device's plan, segment count included
        assert (
            cake_backend.plan(1024, 4, 1, sm).P
            == cake_backend.plan(1024, 4, 1, sm, _current_arch()).P
        )
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
    assert {"k", "q"} <= set(cake_jit.MODULES["glm_intra_v14"]["logical"].values())
    assert {"k", "q"} <= set(cake_jit.MODULES["glm_intra_v18"]["logical"].values())
    preps = (
        "bwd_prep_g8",
        "bwd_prep_g8_replay",
        "bwd_prep_g8_noqg",
        "bwd_prep_g8_noqg_nokb",
    )
    assert cake_jit.records_of_stage("bwd_prep") == preps
    for name in preps:
        prep = cake_jit.MODULES[name]
        assert prep["grid_rule"] == {"kind": "chunks", "x_factor": 1}
        assert not {"ke_out", "qe_out", "v_int", "vb_out"} & set(prep["logical"])
    # the saved-set paths' prep copies nothing of dout either; the replay's keeps the internal copy
    for name in preps:
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert ("do_int" in values) == (name == "bwd_prep_g8_replay"), name
        assert ("dout" in values) == (name == "bwd_prep_g8_replay"), name
    # the tape set's and the replay's prep fold q into qg; the light sets' forms read no q and write no qg (the
    # version-4 set carries the forward's), and the w / u rows' form writes no kb either (no wy launch there)
    for name in ("bwd_prep_g8", "bwd_prep_g8_replay"):
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert {"q", "qg", "kb", "kg", "gk"} <= values, name
    for name in ("bwd_prep_g8_noqg", "bwd_prep_g8_noqg_nokb"):
        values = set(cake_jit.MODULES[name]["logical"].values())
        assert not {"q", "qg"} & values, name
        assert {"kg", "gk", "beta_int", "v_new", "dv2"} <= values, name
        assert ("kb" in values) == (name == "bwd_prep_g8_noqg"), name
    # the fused GLM intra program writes the token-row gate gradients itself and binds no epilogue buffer; the
    # unfused one hands them to the gate epilogue through dg_int / db_int
    v14 = set(cake_jit.MODULES["glm_intra_v14"]["logical"].values())
    v18 = set(cake_jit.MODULES["glm_intra_v18"]["logical"].values())
    unfused = set(cake_jit.MODULES["glm_intra"]["logical"].values())
    epilogue = set(cake_jit.MODULES["gate_epilogue"]["logical"].values())
    assert v18 == v14  # the same program: v18 differs in its named-barrier slots alone
    assert {"dg", "dbeta"} <= v14 and not {"dg_int", "db_int"} & v14
    assert {"dg_int", "db_int"} <= unfused and not {"dg", "dbeta"} & unfused
    assert {"dg_int", "db_int", "dg", "dbeta"} <= epilogue
    assert v14 - {"dg", "dbeta"} == unfused - {"dg_int", "db_int"}
    assert not {"ke", "qe", "v_int", "vb"} & set(cake_jit.LOGICAL_TENSORS)
    for mode, slots in cake_backend.SLOTS.items():
        names = {name for occupants in slots for name, *_ in occupants}
        assert not names & {"ke", "qe", "v_int", "vb"}, mode
        assert ("akk_int" in names) == (mode.startswith(("fwd", "replay"))), mode


def test_the_fused_gradient_turns_its_l2_hints_off_on_the_large_sm107a_rows() -> None:
    # The fused gradient's L2 prefetch and evict-first hints cost about 6 % on the sm_107a rows from 16 heads, so
    # its dAqk-hosting forms ship a second build there with the hints off (the `_nohint` records: the same kernel,
    # launch and operands; the hint thresholds beyond any head count).  The registry dispatches them on sm_107a
    # from 16 heads alone, on either route (its 16-head rows are segmented ones); the other architectures keep the
    # hinted builds.
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
                seg = _expected_segments(1, h, 148, arch)[0] == "seg"
                davq = seg or h >= DAVQ_MIN_H
                nohint = davq and h >= DQKG_NOHINT_MIN_H_BY_ARCH.get(arch, h + 1)
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
        "fwdh_o_v1_rows",
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
    # the two-level carry: the same stage and chain operands, plus the carried states viewed as tiles (the
    # groups' products), the group size and the release flags under an epoch
    v8 = cake_jit.MODULES["seg_carry_v8"]
    assert v8["stage"] == "seg_carry" and v8["library"] != carry["library"]
    chain_values = set(carry["logical"].values())
    v8_values = set(v8["logical"].values())
    assert {"carry_b", "carry_out", "seg_m", "seg_table", "reverse"} <= chain_values
    assert chain_values - {"num_seqs"} <= v8_values
    assert {"carry_out_tiles", "gsize", "flags", "epoch"} <= v8_values
    assert not {"carry_out_tiles", "gsize", "flags", "epoch"} & chain_values
    assert cake_jit.launch_grid(
        "seg_carry_v8",
        num_chunks_pad=4,
        num_seqs=4,
        num_heads=2,
        sm_count=148,
        num_segments=256,
    ) == (128, 1, 1)
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


def test_plan_is_cached_on_its_scalars(monkeypatch) -> None:
    """A repeated problem returns the same plan without selecting a record again (the default entry plans a
    repeated shape once); every scalar and the architecture are part of the key, and a refused problem raises
    on every call."""
    first = cake_backend.plan(261393, 4, 4, 152, "sm_100a")
    calls: list = []
    original = cake_jit.select_record

    def counting(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(cake_jit, "select_record", counting)
    assert cake_backend.plan(261393, 4, 4, 152, "sm_100a") is first
    assert cake_backend.plan(261393.0, 4, 4, 152, "sm_100a") is first
    assert calls == []
    for scalars in ((261392, 4, 4, 152), (261393, 3, 4, 152), (261393, 4, 2, 152)):
        assert cake_backend.plan(*scalars, "sm_100a") is not first
    assert cake_backend.plan(261393, 4, 4, 148, "sm_100a").sm_count == 148
    for arch in ARCHES[1:]:
        assert cake_backend.plan(261393, 4, 4, 152, arch).arch == arch
    for _ in range(2):
        with pytest.raises(ValueError, match="must all be >= 1"):
            cake_backend.plan(0, 1, 1, 148, "sm_100a")


def test_plan_without_arch_takes_the_current_device(monkeypatch) -> None:
    """``plan(..., arch=None)`` selects the records of the current CUDA device's architecture, so its plan serves a
    workspace there on every exported target; without a CUDA device, or on a device no record targets, it takes
    the first exported architecture (whose records the byte queries also take without an ``arch``)."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    for arch in ARCHES:
        monkeypatch.setattr(cake_backend, "_capability_arch", lambda index, a=arch: a)
        p = cake_backend.plan(4096, 1, 16, 148)
        assert p.arch == arch
        assert p.records == cake_backend.plan(4096, 1, 16, 148, arch).records
    monkeypatch.setattr(cake_backend, "_capability_arch", lambda index: None)
    assert cake_backend.plan(4096, 1, 16, 148).arch == ARCHES[0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert cake_backend.plan(4096, 1, 16, 148).arch == ARCHES[0]


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
                # the plan's round-3 fields: the gate fusion of the architecture, the light set's extra tensors
                # and the two-level carry's group size where a path dispatches it
                assert p.gate_fused is _gate_fused(arch)
                assert p.light_saves_qg is True
                assert p.light_saves_wu is _wu_saved(N, heads, sm, arch)
                two_level = any(
                    records.get("seg_carry") == "seg_carry_v8"
                    for records in p.records.values()
                )
                choice = _carry_choice(N, heads, sm, p.NSEG)
                assert p.gsize == (choice[0] if two_level else 0), (
                    lengths,
                    heads,
                    arch,
                )
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
    # 148 SMs, per architecture.  One document at 16 heads: on sm_100a the 32-column serial form (no dav: the fused
    # gradient hosts dAqk; on the saving forward the token-row pair wy_fwd_rows + fwdh_o_v2_rows, the same count)
    # whose light set carries w / u (no wy launch on the light path), on sm_103a / sm_107a a segmented row of 16
    # chains (nine segments per document); four documents at 16 heads: on sm_100a the same paired serial form, on
    # sm_103a / sm_107a a segmented row of 64 chains whose light set carries w / u (no wy launch on either light
    # path: one launch fewer there).  Four documents at 64 heads: the 64-column form (the fused recompute: no
    # WY launch on the light and replay paths; on sm_107a the saving forward's token-row pair wy_fwd_rows +
    # fwdh_o_v1_rows instead of the fused forward).  Segmented rows: four documents at 2 heads (eight chains: the
    # dv-accumulating adjoint, the fused saved-state pass 2; the overlap pair on the forward, the two-level carry
    # on the backward paths) and one document at 4 heads (four chains: the two-CTA adjoint);
    # no dav launch on either (the fused gradient hosts dAqk), no table launch on the forward and the replay (the
    # intra program writes the chunk tables and the block table itself).  The forward has no prep launch; the
    # backward of sm_103a has no gate epilogue launch (the fused GLM intra program); sm_100a / sm_107a launch it on
    # every backward path (the counts below are without it).
    serial_16 = {"fwd": 4, "fwd_nosave": 4, "light": 5, "tape": 5, "replay": 8}
    seg_16 = {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 10,
        "light_states": 7,
        "tape": 7,
        "replay": 11,
    }
    # four documents at 16 heads (64 chains) on sm_103a / sm_107a: the set carries w / u, no wy launch on the light
    # paths (one launch fewer than the one-document row there); on sm_100a the paired serial row again
    seg_16_wu = {
        "fwd": 5,
        "fwd_nosave": 5,
        "light": 9,
        "light_states": 7,
        "tape": 7,
        "replay": 11,
    }
    seg_small = dict(seg_16)
    counts_by_arch = {
        (1, 16): {"sm_100a": serial_16, "sm_103a": seg_16, "sm_107a": seg_16},
        (4, 16): {"sm_100a": serial_16, "sm_103a": seg_16_wu, "sm_107a": seg_16_wu},
        (4, 64): {
            "sm_100a": {"fwd": 3, "fwd_nosave": 3, "light": 5, "tape": 5, "replay": 7},
            "sm_103a": {"fwd": 3, "fwd_nosave": 3, "light": 5, "tape": 5, "replay": 7},
            "sm_107a": {"fwd": 4, "fwd_nosave": 4, "light": 5, "tape": 5, "replay": 7},
        },
        (4, 2): {arch: seg_small for arch in ARCHES},
        (1, 4): {arch: seg_small for arch in ARCHES},
    }
    for (n, h), by_arch in counts_by_arch.items():
        for arch, want in by_arch.items():
            p = cake_backend.plan(4096, n, h, 148, arch)
            # the gate epilogue is one more launch on every backward path of sm_100a / sm_107a
            gate = 0 if _gate_fused(arch) else 1
            expected = {
                path: count + (gate if path not in ("fwd", "fwd_nosave") else 0)
                for path, count in want.items()
            }
            # the saving forward and the forward without a saved set launch the same number of programs (the
            # token-row records replace the internal-row ones)
            assert {path: p.launch_count(path) for path in p.records} == expected, (
                n,
                h,
                arch,
            )
    # plan(arch=None) is the current device's (the first exported architecture's without one)
    for n, h in counts_by_arch:
        assert cake_backend.plan(4096, n, h, 148).records == (
            cake_backend.plan(4096, n, h, 148, _current_arch()).records
        )
    # the 16-head row of sm_100a: the light set carries w / u, so the light path launches no wy program (six launches:
    # the prep, the recompute, the adjoint, the fused gradient, GLM intra, the gate epilogue)
    p16 = cake_backend.plan(4096, 1, 16, 148, "sm_100a")
    assert p16.light_saves_wu and p16.records["light"]["wy"] is None
    assert p16.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg_nokb"
    assert p16.launch_count("light") == 6 and p16.launch_count("tape") == 6
    assert p16.records["tape"]["wy"] == "wy_bwd"
    # a saved set whose tables the caller dropped adds the table launch to the saved-set paths (the fused table
    # program); the forward and the replay select no table record and the table-writing intra program
    for arch in ARCHES:
        gate = 0 if _gate_fused(arch) else 1
        p = cake_backend.plan(4096, 4, 2, 148, arch)
        assert p.launch_count("light", tables=False) == 11 + gate
        assert p.launch_count("tape", tables=False) == 8 + gate
        assert (
            p.launch_count("replay")
            == p.launch_count("replay", tables=False)
            == 11 + gate
        )
        for path in ("fwd", "fwd_nosave", "replay"):
            assert p.records[path]["table"] is None and not p.dispatched("table", path)
            assert p.records[path]["fwd_intra"] == "fwd_intra_v14"
        for path in ("light", "light_states", "tape"):
            assert p.records[path]["table"] == "table_v3"
        # the two-level carry replaces the chain carry launch for launch, on every architecture
        assert p.records["light"]["seg_carry"] == "seg_carry_v8"
        assert cake_backend.plan(4096, 1, 16, 148, arch).records["fwd"][
            "fwd_intra"
        ] == ("fwd_intra" if arch == "sm_100a" else "fwd_intra_v14")
        # above the layout-table document bound the segmented route launches the block table on its own
        wide = cake_backend.plan(4096, 300, 2, 148, arch)
        assert wide.records["fwd"]["table"] == "table" and wide.dispatched(
            "seg_table", "fwd"
        )
        assert wide.launch_count("fwd") == wide.launch_count("fwd_nosave") == 7
        assert wide.launch_count("replay") == 13 + gate
        # the backward from this plan's own light set: the states-saved path above the size rule
        assert p.launch_count("bwd") == p.launch_count("light") == 10 + gate
        tall = cake_backend.plan(384 * p.NSEG, 4, 2, 148, arch)
        assert tall.light_path == "light_states" and tall.NSEG == p.NSEG
        assert tall.launch_count("bwd") == 7 + gate
        assert tall.launch_count("bwd", tables=False) == 8 + gate
    with pytest.raises(ValueError, match="path must be one of"):
        p.launch_count("sideways")


def test_saved_set_version_4_fields_and_the_autograd_extras() -> None:
    # The light set is version 4: the fold's qg always, hseg / seg_state under the size rule, w / u on their
    # rows; each optional on its own, so the autograd node carries the names of the present extras (the absent
    # ones are no longer a suffix of a fixed order).  The tape set stays version 2 and carries none of them.
    assert cake_backend.SAVED_VERSION == 4 and cake_backend.SAVED_VERSION_TAPE == 2
    assert [f.name for f in dataclasses.fields(KdaChunkSaved)] == [
        "version",
        "Aqk",
        "Akk",
        "tables",
        "S",
        "H",
        "num_docs",
        "scale",
        "h",
        "v_new",
        "seg_m",
        "hseg",
        "seg_state",
        "qg",
        "w",
        "u",
    ]
    for name in ("h", "v_new", "seg_m", "hseg", "seg_state", "qg", "w", "u"):
        assert KdaChunkSaved.__dataclass_fields__[name].default is None, name
    assert cake_backend._EXTRA_FIELDS == {
        4: ("qg", "hseg", "seg_state", "w", "u"),
        2: ("h", "v_new", "seg_m"),
    }
    t = torch.empty(0)
    light = KdaChunkSaved(4, t, t, None, 1, 1, 1, SCALE, qg=t)
    tensors, names = cake_backend._saved_extra(light)
    assert names == ("qg",) and len(tensors) == 1 and tensors[0] is t
    full = KdaChunkSaved(
        4, t, t, None, 1, 1, 1, SCALE, hseg=t, seg_state=t, qg=t, w=t, u=t
    )
    tensors, names = cake_backend._saved_extra(full)
    assert names == ("qg", "hseg", "seg_state", "w", "u") and len(tensors) == 5
    wu_only = KdaChunkSaved(4, t, t, None, 1, 1, 1, SCALE, qg=t, w=t, u=t)
    assert cake_backend._saved_extra(wu_only)[1] == ("qg", "w", "u")
    tape = KdaChunkSaved(2, t, t, None, 1, 1, 1, SCALE, h=t, v_new=t)
    assert cake_backend._saved_extra(tape)[1] == ("h", "v_new")
    tape_seg = KdaChunkSaved(2, t, t, None, 1, 1, 1, SCALE, h=t, v_new=t, seg_m=t)
    assert cake_backend._saved_extra(tape_seg)[1] == ("h", "v_new", "seg_m")


def test_the_light_set_saves_qg_everywhere_and_w_u_where_the_prep_writes_no_kb() -> (
    None
):
    # Plan.light_saves_qg / light_saves_wu read the light path's prep record: no-qg forms on every row (the set
    # carries the forward's qg), the no-kb form (w / u in the set, no wy launch on either light path) on the serial
    # 16-head rows of sm_100a whose 32-column form fits one wave and on the segmented 16-head rows of sm_103a /
    # sm_107a above 32 chains; the byte query adds the tensors accordingly.
    S = 4096
    seen_wu = seen_seg_wu = 0
    for arch, n, h, sm in itertools.product(
        ARCHES, (1, 2, 7, 300), (1, 2, 4, 8, 12, 16, 24, 32, 64), (132, 148, 212)
    ):
        p = cake_backend.plan(S, n, h, sm, arch)
        wu = _wu_saved(n, h, sm, arch)
        seen_wu += wu
        seen_seg_wu += wu and p.recur == "seg"
        assert p.light_saves_qg is True
        assert p.light_saves_wu is wu, (arch, n, h, sm)
        prep = p.records["light"]["bwd_prep"]
        assert prep == ("bwd_prep_g8_noqg_nokb" if wu else "bwd_prep_g8_noqg"), (
            arch,
            n,
            h,
            sm,
        )
        if "light_states" in p.records:
            assert p.records["light_states"]["bwd_prep"] == prep
        assert p.records["tape"]["bwd_prep"] == "bwd_prep_g8"
        assert p.records["replay"]["bwd_prep"] == "bwd_prep_g8_replay"
        if wu:
            assert h == WU_SAVED_H
            if p.recur == "serial":
                assert arch in WU_SAVED_ARCHES and n * h * 4 <= 4 * sm
            else:
                assert arch in WU_SEG_SAVED_ARCHES and n * h > WU_SEG_MIN_CHAINS
                assert p.records["light_states"]["wy"] is None
            assert p.records["light"]["wy"] is None
            assert p.records["tape"]["wy"] == "wy_bwd"
            assert p.records["replay"]["wy"] == "wy_bwd_replay"
        tables = cake_backend.tables_bytes(S, n, h, sm_count=sm, arch=arch)
        states = 0
        if cake_backend.seg_states_saved(S, p.NSEG):
            states = 3 * p.NSEG * h * D * D * 4
        want = 256 * S * h + tables + p.R * h * D * 2 + states
        if wu:
            want += 2 * p.R * h * D * 2
        assert cake_backend.saved_bytes(S, h, n, sm_count=sm, arch=arch) == want, (
            arch,
            n,
            h,
            sm,
        )
    assert seen_wu > seen_seg_wu > 0
    # 300 documents at 16 heads on sm_100a need 19200 slots, above 4 * 148: the 64-column form keeps the wy launch
    p = cake_backend.plan(S, 300, 16, 148, "sm_100a")
    assert p.light_saves_wu is False and p.records["light"]["wy"] == "wy_bwd"
    assert p.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg"
    for arch in WU_SEG_SAVED_ARCHES:
        # two documents at 16 heads are 32 chains, not above: the fused saved-state pass 2 keeps forming the WY
        # representation (its w^T adjoint behind it) and the light path keeps the wy launch
        p = cake_backend.plan(S, 2, 16, 148, arch)
        assert p.recur == "seg" and p.light_saves_wu is False
        assert p.records["light"]["wy"] == "wy_bwd"
        assert p.records["light"]["bwd_prep"] == "bwd_prep_g8_noqg"
        assert p.records["light_states"]["wy"] is None
        assert p.records["light_states"]["fwdh_p3"] == "fwdh_p3_wy"
        assert p.records["light_states"]["dhu"].endswith("_wt")
        # three documents are 48 chains: the set carries w / u, neither light path launches wy, pass 2 and the
        # adjoint take their plain forms (no w^T tile: the saved w is read as it is)
        p = cake_backend.plan(S, 3, 16, 148, arch)
        assert p.recur == "seg" and p.light_saves_wu is True
        for path in ("light", "light_states"):
            assert p.records[path]["bwd_prep"] == "bwd_prep_g8_noqg_nokb"
            assert p.records[path]["wy"] is None
            assert p.records[path]["fwdh_p3"] == "fwdh_p3"
            assert not p.records[path]["dhu"].endswith("_wt")
        assert p.records["light"]["fwdh"] == "fwdh_v3"
        assert p.records["light_states"]["fwdh"] is None


def test_gate_fused_architectures_lay_the_arena_out_without_the_epilogue_buffers() -> (
    None
):
    # glm_intra_v14 (sm_103a) / glm_intra_v18 (sm_107a) write the gate gradients themselves: no gate_epilogue record,
    # no dg_int / db_int in the layouts.  The default layouts keep their bytes (the two shared slots with h / do_int); the lean single-pass
    # layout loses the 4-byte db_int slot; the multi-pass per-pass set loses both buffers.  A query without an
    # architecture takes sm_100a's answer: the epilogue's buffers stay.
    assert cake_backend._gate_fused(None) is _gate_fused(None) is False
    lean = cake_backend.KdaArena(lean=True)
    for arch in ARCHES:
        fused = _gate_fused(arch)
        assert cake_backend._gate_fused(arch) is fused
        for lengths, heads in (((63, 1, 65), 2), ((129,), 16), ((2000,) * 3, 64)):
            S, N = sum(lengths), len(lengths)
            p = cake_backend.plan(S, N, heads, 148, arch)
            assert p.gate_fused is fused
            assert p.dispatched("gate_epilogue", "light") is not fused
            assert p.records["light"]["glm_intra"] == (
                GATE_FUSED_RECORD[arch] if fused else "glm_intra"
            )
            R, H, NC, NSEG = p.R, p.H, p.NC_pad, p.NSEG
            for mode in cake_backend.MODES:
                assert (
                    p.workspace_bytes(mode)
                    == cake_backend._mode_bytes(R, H, NC, NSEG, mode)
                    == cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, True)
                ), (arch, mode)
                names = {r.name for r in p.layout(mode)}
                assert bool(set(GATE_BUFFERS) & names) is (
                    not fused and not mode.startswith("fwd")
                ), (arch, mode)
            for mode in cake_backend.LEAN_MODES:
                plain = cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, False)
                dropped = cake_backend._mode_bytes(R, H, NC, NSEG, mode, 1, True)
                assert plain - dropped == 4 * R * H
                assert p.workspace_bytes(mode) == (dropped if fused else plain)
                for t in (2, 3, 4):
                    names = {r.name for r in p.layout(mode, t)}
                    assert names & PASS_NAMES == {
                        name for name, *_ in _pass_occupants(fused)
                    }, (arch, mode, t)
            q = {"sm_count": 148, "arch": arch}
            assert cake_backend.workspace_bytes(
                S, N, heads, arena=lean, **q
            ) == p.workspace_bytes("bwd_lean")
            assert cake_backend.memory_report(S, N, heads, **q)["gate_fused"] is fused


def test_seg_state_rule_is_a_pure_function_of_the_scalars() -> None:
    rule, k = cake_backend.seg_states_saved, cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG
    assert k == 384
    assert cake_backend.SAVED_VERSION == 4 and cake_backend.SAVED_VERSION_TAPE == 2
    for nseg in (1, 64, 153, 256):
        assert rule(k * nseg, nseg) is True and rule(k * nseg - 1, nseg) is False
    assert rule(10**9, 0) is False  # the serial route has no segment states
    # the sm_100a rule (a plan without an arch takes the current device's)
    a0 = cake_jit.ARCHES[0]
    for n, h, sm in ((1, 1, 148), (4, 4, 152), (3, 8, 148), (4, 1, 152)):
        nseg = cake_backend.plan(1, n, h, sm, a0).NSEG  # independent of S
        assert (
            nseg
            == cake_backend.plan(k * nseg, n, h, sm, a0).NSEG
            == _expected_segments(n, h, sm, a0)[2]
        )
        assert cake_backend.plan(k * nseg, n, h, sm, a0).seg_save is True
        assert cake_backend.plan(k * nseg - 1, n, h, sm, a0).seg_save is False
    # the serial route
    assert cake_backend.plan(10**6, 1, 16, 148, a0).seg_save is False


def test_byte_queries_follow_the_arena_layout() -> None:
    GiB = 2**30
    # three 16-head rows: the paired serial rows of sm_100a (their light set carries w / u), segmented elsewhere,
    # where the three-document row (48 chains) carries w / u too
    rows = SHAPES + [
        ((239718 // 3,) * 3, 4),
        ((261393 // 4,) * 4, 64),
        ((129,), 16),
        ((2000, 2000), 16),
        ((2000, 2000, 2000), 16),
    ]
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
        # light (version 4): the fold's qg BF16 [R, H, 128] on every row; the WY representation w / u (two more
        # such tensors) on the paired serial 16-head rows of sm_100a and on the segmented 16-head rows of sm_103a /
        # sm_107a above 32 chains; the segment-boundary states (three FP32
        # state blocks) when the size rule holds: of these rows only the large segmented one (H = 4), not the
        # small packs or the serial H = 64
        states = (
            3 * p.NSEG * heads * D * D * 4
            if cake_backend.seg_states_saved(S, p.NSEG)
            else 0
        )
        assert p.seg_save == (S >= 384 * p.NSEG > 0)
        assert (states > 0) == (S > 200_000 and heads == 4)
        qg = p.R * heads * D * 2
        wu = 2 * p.R * heads * D * 2 if _wu_saved(N, heads, 148, arch) else 0
        wu_row = heads == 16 and (
            arch in WU_SAVED_ARCHES or N * heads > WU_SEG_MIN_CHAINS
        )
        assert (wu > 0) == wu_row, (lengths, arch)
        assert p.light_saves_qg is True and p.light_saves_wu is (wu > 0)
        assert (
            cake_backend.saved_bytes(S, heads, N, **q)
            == 256 * S * heads + tables + qg + wu + states
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
        assert report["saved"] == 256 * S * heads + tables + qg + wu + states
        assert report["seg_states_saved"] == (states > 0)
        assert report["saved_seg_state_bytes"] == states
        assert report["gate_fused"] is _gate_fused(arch) is p.gate_fused
        tape_report = cake_backend.memory_report(S, N, heads, "tape", **q)
        assert tape_report["saved"] == tape_bytes
        assert tape_report["saved_seg_state_bytes"] == 2 * p.NSEG * heads * D * D * 4
        assert tape_report["gate_fused"] is p.gate_fused
        assert report["bwd_total"] == report["arena_bwd"] + report["out_bwd"]
        assert report["arch"] == arch
        # without an architecture the queries take the current device's (None without a device): the bytes of
        # the explicit query of that architecture
        host = _query_arch()
        assert cake_backend.workspace_bytes(S, N, heads, sm_count=148) == (
            cake_backend.workspace_bytes(S, N, heads, sm_count=148, arch=host)
        )
        assert cake_backend.saved_bytes(S, heads, N, sm_count=148) == (
            cake_backend.saved_bytes(S, heads, N, sm_count=148, arch=host)
        )
        host_report = cake_backend.memory_report(S, N, heads, sm_count=148)
        assert host_report["arch"] == host
        # a query without a device's architecture takes the first exported one's gate fusion and w / u rows
        assert host_report["gate_fused"] is _gate_fused(host)
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


def _gate_fused_slots(slots):
    """The layout's slots without the gate epilogue's buffers (an emptied slot leaves too), as the fused GLM
    intra program's architectures lay the arena out."""
    kept = (
        tuple(o for o in occupants if o[0] not in GATE_BUFFERS) for occupants in slots
    )
    return tuple(occupants for occupants in kept if occupants)


GATE_BUFFERS = ("dg_int", "db_int")


def test_lifetime_table_lists_every_occupant() -> None:
    assert cake_backend._GATE_BUFFERS == GATE_BUFFERS
    for mode, slots in cake_backend.SLOTS.items():
        # without a plan the table keeps the gate epilogue's buffers (gate_fused=None -> False)
        for rows, expected in (
            (cake_backend.lifetime_table(mode), slots),
            (cake_backend.lifetime_table(mode, gate_fused=False), slots),
            (
                cake_backend.lifetime_table(mode, gate_fused=True),
                _gate_fused_slots(slots),
            ),
        ):
            assert len(rows) == sum(len(occupants) for occupants in expected)
            assert {(row["slot"], row["buffer"]) for row in rows} == {
                (slot, name)
                for slot, occupants in enumerate(expected)
                for name, *_ in occupants
            }
            for row in rows:
                assert row["first_write"] <= row["last_read"]
                assert row["unit"] in ("R*H", "seg_block")
        names = {name for occupants in slots for name, *_ in occupants}
        assert (set(GATE_BUFFERS) <= names) is not mode.startswith("fwd"), mode
    # a plan's table takes its architecture's fusion
    for arch in ARCHES:
        p = cake_backend.plan(4096, 3, 2, 148, arch)
        for mode in ("bwd", "replay", "bwd_tape", "replay_tape"):
            names = {row["buffer"] for row in cake_backend.lifetime_table(mode, plan=p)}
            assert (set(GATE_BUFFERS) & names) == (
                set() if _gate_fused(arch) else set(GATE_BUFFERS)
            ), (arch, mode)
            assert names == {r.name for r in p.layout(mode)}


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


def _pass_occupants(gate_fused: bool):
    """The per-pass set of a lean layout: the fused GLM intra program's architectures drop the gate epilogue's
    ``dg_int`` (F32p) / ``db_int`` (V32p)."""
    return tuple(
        o
        for o in cake_backend._PASS_OCCUPANTS
        if not (gate_fused and o[0] in GATE_BUFFERS)
    )


def _pass_bytes_per_th(gate_fused: bool) -> int:
    """Bytes of the per-pass set per internal token-head of one tail range: 2568, fused 2052."""
    return sum(
        KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
        for _name, kind, _sub, _life in _pass_occupants(gate_fused)
    )


# the lean layouts' bytes per internal token-head: one pass (the saved-set layout without the replay's dout copy;
# the fused architectures drop the 4-byte db_int slot and the dg_int under the h tape), from two passes on the
# whole-sequence set
LEAN_SINGLE_BYTES = {False: {False: 4108, True: 3340}, True: {False: 4104, True: 3336}}
LEAN_WHOLE_BYTES = {False: 2820, True: 2052}
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


def _expected_lean_bytes(
    tape: bool, t: int, R: int, H: int, NC: int, NSEG: int, gate_fused: bool = False
) -> int:
    """The lean layout's bytes from the documented rule: one pass is the saved-set layout without the replay's
    dout copy (4108 B per internal token-head, tape 3340; the fused architectures 4104 / 3336: no db_int slot);
    from two passes on the whole-sequence set (2820 B per internal token-head, tape 2052) plus the per-pass set
    (without dg_int / db_int where the gate epilogue is fused) packed first-fit, in order, into the dead ``qg`` /
    ``kg`` (512) and ``w`` (256) slots, the rest into an overflow slot shared with the segmented route's three
    state blocks."""
    block = NSEG * H * D * D * 4
    if t == 1:
        single = LEAN_SINGLE_BYTES[gate_fused][tape]
        return single * R * H + max(0, 3 * block - 256 * R * H)
    R_b = BT * -(-NC // t)
    capacity = {"qg": 512 * R * H, "w": 256 * R * H}
    used = {"qg": 0, "w": 0}
    overflow = 0
    for _name, kind, _sub, _life in _pass_occupants(gate_fused):
        nbytes = R_b * H * KIND_BYTES_PER_TH[cake_backend.PASS_KINDS[kind]]
        for host in ("qg", "w"):
            if used[host] + nbytes <= capacity[host]:
                used[host] += nbytes
                break
        else:
            overflow += nbytes
    extra = max(overflow, 3 * block) if (overflow or NSEG) else 0
    return LEAN_WHOLE_BYTES[tape] * R * H + extra


def _lean_rows():
    return SHAPES + [
        ((239718 // 3,) * 3, 4),
        ((261393 // 4,) * 4, 64),
        ((2000,) * 300, 1),
        ((129,), 16),  # the paired serial row of sm_100a whose light set carries w / u
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
        fused = _gate_fused(arch)
        assert p.gate_fused is fused
        pass_names = frozenset(name for name, *_ in _pass_occupants(fused))
        # the default layouts' constants: the saved-set layouts and the replay layouts (the arena of a default
        # backward workspace is the replay's); the gate epilogue's buffers share their slots with h / do_int, so
        # the fused architectures' default layouts are no smaller
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
            whole = LEAN_WHOLE_BYTES[tape]
            default = cake_backend.memory_report(S, N, heads, save_mode, **q)
            assert default["gate_fused"] is fused
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
                if fused:
                    slots = _gate_fused_slots(slots)
                assert want == _slot_bytes(slots, R, H, NSEG, R_b), (mode, t)
                names = {r.name for r in regions}
                # no replay-only buffers (one pass keeps the saved-set layout's own aqk_int rows); no gate
                # epilogue buffers on the architectures that fuse it
                assert not ({"do_int", "akk_int"} & names), (mode, t)
                assert (t == 1) or "aqk_int" not in names, (mode, t)
                assert bool(set(GATE_BUFFERS) & names) is not fused, (mode, t)
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
                        - (set(GATE_BUFFERS) if fused else set())
                    )
                    assert names == base, (mode, t)
                    by_name = {r.name: r for r in regions}
                    assert by_name["kb"].slot == by_name["dh"].slot
                    assert by_name["kb"].offset == by_name["dh"].offset + 256 * R * H
                    assert by_name["kb"].lifetime == (1, 3)
                    # the fused architectures drop the 4-byte db_int slot (4104 / 3336 B per token-head)
                    assert want == LEAN_SINGLE_BYTES[fused][tape] * R * H + blocks_over
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
                    assert names == whole_names | pass_names | seg, (mode, t)
                    assert "dv1" not in names and "dAqk" in PASS_NAMES
                    # the documented packing rule, spelled independently: the per-pass set in order, first-fit
                    # into the dead qg / kg and w slots, the rest into the overflow slot shared with the blocks
                    assert want == _expected_lean_bytes(
                        tape, t, R, H, NC, NSEG, fused
                    ), (mode, t)
                    if t == 2:
                        # two passes: dq_f / dk_f fill the qg slot, dg_f the w slot; dAkk, dAqk and db_f
                        # overflow (258 B per token-head), with dg_int / db_int 516
                        over = 258 if fused else 516
                        assert want == whole * R * H + max(over * R * H, 3 * block)
                    elif t >= 4 and NC % t == 0:  # from four passes on everything fits
                        assert want == whole * R * H + 3 * block
                    floor = whole * R * H + 3 * block
                    assert floor <= want <= floor + _pass_bytes_per_th(fused) * R_b * H
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
        # the fused architectures' single-pass table: no db_int slot, the slots after it renumbered
        fused_rows = cake_backend.lifetime_table(mode, gate_fused=True)
        fused_single = _gate_fused_slots(single)
        assert {(row["slot"], row["buffer"]) for row in fused_rows} == {
            (slot, name)
            for slot, occupants in enumerate(fused_single)
            for name, *_ in occupants
        }
        assert len(fused_rows) == len(rows) - 2 and len(fused_single) == len(single) - 1
        with pytest.raises(ValueError, match="plan="):
            cake_backend.lifetime_table(mode, 2)
        for (lengths, heads), t, arch in itertools.product(
            SHAPES[::4], (2, 3, 4), ARCHES
        ):
            p = cake_backend.plan(sum(lengths), len(lengths), heads, 148, arch)
            rows = cake_backend.lifetime_table(mode, t, plan=p)
            regions = {r.name: r for r in p.layout(mode, t)}
            assert {row["buffer"] for row in rows} == set(regions)
            # gate_fused=None takes the plan's; an explicit value overrides it
            assert bool(set(GATE_BUFFERS) & set(regions)) is not p.gate_fused
            unfused = cake_backend.lifetime_table(mode, t, plan=p, gate_fused=False)
            names = {row["buffer"] for row in unfused}
            assert names == set(regions) | set(GATE_BUFFERS)
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
            assert p.dispatched("gate_epilogue", path) is not _gate_fused(arch)
            for stage in TAIL_STAGES:
                if not p.dispatched(stage, path):
                    continue  # the fused GLM intra program's architectures launch no gate epilogue
                name = p.record(stage, path)
                logical = set(cake_jit.MODULES[name]["logical"].values())
                assert not (logical & TAIL_FORBIDDEN), (
                    name,
                    sorted(logical & TAIL_FORBIDDEN),
                )
                assert logical <= allowed, (name, sorted(logical - allowed))
                seen.add(name)
    assert {
        "glm_intra",
        "glm_intra_v14",
        "glm_intra_v18",
        "gate_epilogue",
    } <= seen and any(n.startswith("dqkg") for n in seen)


def test_manifest_is_well_formed(manifest: dict) -> None:
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.experimental.cake_kda_train"
    arch_flags = manifest["arch_flags"]
    assert set(ARCHES) <= set(arch_flags)
    libraries = manifest["libraries"]
    names = [entry["name"] for entry in libraries]
    # The 65 programs of cake_jit.MODULES, each built for the architectures its record delivers (61 on all three,
    # the four of RECORD_ARCHES on one each: 187 libraries), plus the KCP programs of cake_kcp_jit.MODULES
    # on every architecture (its test-only TEST_MODULES live in build.test.json, never here).
    kcp_libraries = {record["library"] for record in cake_kcp_jit.MODULES.values()}
    train_libraries = {
        (record["library"], arch)
        for record in cake_jit.MODULES.values()
        for arch in record["arches"]
    }
    assert len(train_libraries) == 187
    assert len(names) == len(set(names)) == 187 + len(kcp_libraries) * len(ARCHES)
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
        for name, record in cake_jit.MODULES.items()
        for arch in record["arches"]
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
    # a record delivered for a subset of the architectures refuses the others
    for name, arches in RECORD_ARCHES.items():
        for arch in ARCHES:
            if arch in arches:
                assert cake_jit.library_name(name, arch).endswith(f"_{arch}")
            else:
                with pytest.raises(RuntimeError, match="not delivered"):
                    cake_jit.library_name(name, arch)


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
    dA / dv where the fused gradient does not host it, the adjoint recurrence, the fused gradient, GLM intra and,
    on sm_107a, the gate epilogue (the tail stages once per chunk range of a lean workspace's ``tail_passes``).
    The carry record is the row's: the overlap pair's on the forward's overlap rows, the two-level carry where its
    rule accepts, else the chain carry."""
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
            seq.append(p.record("seg_carry", path))
    if p.dispatched("fwdh_p3", path):
        seq.append(p.record("fwdh_p3", path))
    if p.dispatched("dav", path):
        seq.append(p.record("dav", path))
    dhu = p.record("dhu", path)
    seq += [dhu, p.record("seg_carry", path), dhu] if seg else [dhu]
    ranges = cake_backend._tail_ranges(p.NC_pad, tail_passes)
    seq += _tail_records(p, path) * len(ranges)
    return seq


def _tail_records(p, path: str) -> list[str]:
    """The tail's records on ``path`` in launch order: the fused gradient, GLM intra and, where the architecture
    dispatches it (sm_107a), the gate epilogue."""
    return [p.record(stage, path) for stage in TAIL_STAGES if p.dispatched(stage, path)]


def _device_route(device: torch.device, n: int, h: int) -> str:
    """The route of the row on the current device (its SM count and architecture)."""
    return _expected_segments(
        n, h, cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    )[0]


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


def _spy_call_workspaces(monkeypatch) -> list:
    """Record the workspace every call without one runs in (``_entry_acquire``): its option, direction, bytes, save
    mode and ``doc_lengths``, not the workspace itself (a strong reference would be "retained")."""
    created: list = []
    original = cake_backend._entry_acquire

    def spy(*args, **kwargs):
        ws_, key = original(*args, **kwargs)
        created.append(
            types.SimpleNamespace(
                arena_option=ws_.arena_option,
                backward=ws_.backward,
                bytes=ws_.bytes,
                save_mode=ws_.save_mode,
                doc_lengths=ws_.doc_lengths,
            )
        )
        return ws_, key

    monkeypatch.setattr(cake_backend, "_entry_acquire", spy)
    return created


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
    # A saved set without tables (the backward rebuilds them; the version-4 tensors stay) and a saved set on
    # another workspace.
    bare = dataclasses.replace(saved, tables=None)
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
    seg = _device_route(device, len(lengths), heads) == "seg"
    assert (saved_t.seg_m is not None) == seg
    assert saved_t.qg is None and saved_t.w is None and saved_t.u is None
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
    assert saved.qg is not None
    for t in (out, *grads.values(), saved.Aqk, saved.Akk, saved.qg, *saved.tables):
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
    # Seven rows: the segmented route with six chains (the two-level carry), 16 heads (on
    # sm_100a the 32-column serial form whose light set carries w / u: no wy launch on the light path; segmented
    # on sm_103a / sm_107a), the 64-column serial form at 64 heads (the fused forward and the fused backward
    # recompute), the few-chain segmented rows (the two-CTA adjoint; the fused gradient hosts dAqk) and, on a
    # 148-SM device, the overlap pair's rows (the release-order pass 2 at 2 heads, the chain-major pass 2 at 8
    # heads, and three documents at 4 heads: one walk CTA per SM per document, exactly the 3.0-wave threshold).
    # The gate epilogue launches on sm_100a / sm_107a (sm_103a's GLM intra program fuses it).
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
        assert p.recur == _device_route(device, N, heads)
        assert p.seg_save is False and p.light_path == "light"
        assert p.dispatched("gate_epilogue", "light") is not p.gate_fused
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
        assert not (p.light_saves_wu and "wy_bwd" in launched)
        launched.clear()
        bare = dataclasses.replace(saved, tables=None)
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
    assert saved.version == cake_backend.SAVED_VERSION == 4
    assert saved.qg is not None and saved.w is None and saved.u is None

    def refused(match: str, other: KdaChunkSaved) -> None:
        with pytest.raises(ValueError, match=match):
            cake_backend.chunk_kda_bwd(q, k, v, g, beta, cu, dout, saved=other)

    # versions 1 and 3 (the light sets without qg) are no longer accepted
    for version in (1, 3):
        refused(
            "version",
            KdaChunkSaved(
                version, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE, qg=saved.qg
            ),
        )
    # a version-4 light set carries the forward's qg (the light backward's prep writes none), with the plan's shape
    refused("qg", dataclasses.replace(saved, qg=None))
    refused("saved.qg", dataclasses.replace(saved, qg=saved.qg[:-BT]))
    # w and u come together, and only on the rows whose light backward reads them from the set (none at two heads)
    refused("w and u together", dataclasses.replace(saved, w=saved.qg))
    refused("w and u together", dataclasses.replace(saved, u=saved.qg))
    refused("carries w / u", dataclasses.replace(saved, w=saved.qg, u=saved.qg))
    # a version-4 set carries hseg and seg_state together, with the plan's shapes
    nseg = cake_backend.plan(
        65,
        1,
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    ).NSEG
    block = torch.zeros((nseg, heads, D, D), dtype=torch.float32, device=device)
    refused(
        "hseg and seg_state together",
        dataclasses.replace(saved, tables=None, hseg=block),
    )
    refused(
        "saved.seg_state",
        dataclasses.replace(saved, tables=None, hseg=block, seg_state=block),
    )
    refused(
        "needs h and v_new",
        KdaChunkSaved(2, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE),
    )
    refused("saved.tables", dataclasses.replace(saved, tables=saved.tables[:2]))
    refused("saved: for", dataclasses.replace(saved, tables=None, S=64))
    out, _ = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, save_for_backward=False)
    assert out.shape == (65, heads, D)
    assert cake_backend.is_supported(device)


def test_the_light_saved_set_carries_qg_and_w_u_where_the_plan_says(
    device, manifest
) -> None:
    """Version 4 on the device: every light set carries the fold's qg (BF16 [R, H, 128], a per-call tensor
    outside the arena); w / u exactly on the rows whose plan reads them from the set (the paired serial 16-head
    rows of sm_100a and the segmented 16-head rows of sm_103a / sm_107a above 32 chains: no wy launch there); the
    bytes are the query's; the backward refuses a set without qg, an unpaired w / u and w / u on the wrong row, and
    gives the same gradients from the set without its tables."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    seen_wu = 0
    # the 16-head rows: one, two and three documents (16, 32 and 48 chains: on sm_103a / sm_107a the third alone
    # carries w / u; on sm_100a all three are paired serial rows)
    cases = (
        ((129,), 16),
        ((65,), 2),
        ((63, 1, 65), 64),
        ((129, 300), 16),
        ((129, 300, 70), 16),
    )
    for lengths, heads in cases:
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        launched: list[str] = []

        def recording(name, values, *, grid, device_index):
            launched.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = KdaWorkspace.for_problem(S, N, heads, device, launcher=recording)
        p = ws.plan
        wu = _wu_saved(N, heads, sm, arch)
        seen_wu += wu
        assert p.light_saves_qg is True and p.light_saves_wu is wu
        out, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        assert saved.version == cake_backend.SAVED_VERSION
        rows = [saved.qg] + ([saved.w, saved.u] if wu else [])
        assert (saved.w is None) == (saved.u is None) == (not wu), (lengths, heads)
        arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
        for t in rows:
            assert t.shape == (p.R, heads, D) and t.dtype == torch.bfloat16
            assert t.device == ws.device
            assert not (arena_lo <= t.data_ptr() < arena_hi)
        tables = p.table_words() * 4
        states = [saved.hseg, saved.seg_state] if saved.hseg is not None else []
        tensors = (saved.Aqk, saved.Akk, *rows, *states)
        owned = sum(t.numel() * t.element_size() for t in tensors) + tables
        assert owned == cake_backend.saved_bytes(S, heads, N, sm_count=sm, arch=arch)
        assert owned == cake_backend.saved_bytes(S, heads, N, sm_count=sm)
        launched.clear()
        grads = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        reference = [t.clone() for t in grads]
        assert launched == _expected_launches(p, p.light_path)
        assert ("wy_bwd" in launched) is (p.records[p.light_path]["wy"] == "wy_bwd")
        assert not (wu and "wy_bwd" in launched)
        assert (launched[0] == "bwd_prep_g8_noqg_nokb") is wu
        # the set without its tables, and the tape set of the same problem (no version-4 tensors)
        without_tables = dataclasses.replace(saved, tables=None)
        bare = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=without_tables, workspace=ws
        )
        for mine, theirs in zip(reference, bare, strict=True):
            assert torch.equal(mine, theirs)
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        assert saved_t.qg is None and saved_t.w is None and saved_t.u is None

        def refused(match: str, other: KdaChunkSaved) -> None:
            with pytest.raises(ValueError, match=match):
                cake_backend.chunk_kda_bwd(
                    q, k, v, g, beta, cu, dout, saved=other, workspace=ws
                )

        refused("qg", dataclasses.replace(saved, qg=None))
        if wu:
            refused("carries none", dataclasses.replace(saved, w=None, u=None))
            refused("w and u together", dataclasses.replace(saved, u=None))
            refused("saved.w", dataclasses.replace(saved, w=saved.w[:-BT]))
        else:
            refused("w and u together", dataclasses.replace(saved, w=saved.qg))
            refused("carries w / u", dataclasses.replace(saved, w=saved.qg, u=saved.qg))
    assert seen_wu == (3 if arch in WU_SAVED_ARCHES else 1)


def test_the_states_saving_set_of_a_segmented_16_head_row_feeds_the_plain_passes(
    device, manifest
) -> None:
    """On the segmented 16-head rows whose light set carries ``w`` / ``u`` (above 32 (document, head) chains on
    sm_103a / sm_107a) a pack at the size rule's threshold saves the segment-boundary states too: the backward
    from that set launches the kb-free prep, no WY program, the plain pass 3 and the plain adjoint recurrence
    (the saved representation feeds both), and its gradients equal those of the checkpoint replay, of the same
    set stripped of its states and of the set on a tape-mode workspace bit for bit."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    num_docs, heads = 3, 16
    route, _, nseg = cake_jit.segment_plan(num_docs, heads, sm, arch)
    if route != "seg" or not _wu_saved(num_docs, heads, sm, arch):
        pytest.skip(f"{arch}: three documents at 16 heads are a {route} row")
    S = (
        cake_backend.SEG_SAVE_MIN_TOKENS_PER_SEG * nseg
    )  # the shortest pack that saves the states
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
    assert p.recur == "seg" and nseg == p.NSEG and p.seg_save is True
    assert p.light_path == "light_states"
    assert p.light_saves_qg is True and p.light_saves_wu is True
    records = p.records["light_states"]
    assert records["bwd_prep"] == "bwd_prep_g8_noqg_nokb" and records["wy"] is None
    assert records["fwdh_p3"] == "fwdh_p3" and not records["dhu"].endswith("_wt")
    _, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
    assert saved.hseg is not None and saved.seg_state is not None
    assert saved.qg is not None and saved.w is not None and saved.u is not None
    launched.clear()
    grads = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
    )
    grads = [t.clone() for t in grads]
    assert launched == _expected_launches(p, "light_states")
    assert launched[0] == "bwd_prep_g8_noqg_nokb" and "wy_bwd" not in launched
    assert "fwdh_p3" in launched and records["dhu"] in launched
    replay = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=None, workspace=ws
    )
    stripped = dataclasses.replace(saved, hseg=None, seg_state=None)
    launched.clear()
    from_stripped = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=stripped, workspace=ws
    )
    assert launched == _expected_launches(p, "light")
    assert launched[0] == "bwd_prep_g8_noqg_nokb" and "wy_bwd" not in launched
    tape_ws = KdaWorkspace.for_problem(S, num_docs, heads, device, save_mode="tape")
    from_tape_ws = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=saved, workspace=tape_ws
    )
    for other in (replay, from_stripped, from_tape_ws):
        for mine, theirs in zip(grads, other, strict=True):
            assert torch.equal(mine, theirs)


def test_the_two_level_carry_binds_its_group_operands(device, manifest) -> None:
    """Where a path dispatches the two-level carry (every architecture, at most two heads, the rule's rows) its
    launch binds the carried states viewed as [NSEG * H, 128, 128] tiles, the plan's group size, the flag buffer
    (enough words for its grid) and a fresh epoch per launch, at the rule's grid; the chain carry binds none of
    them."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    seen = 0
    for lengths, heads in (
        ((129,), 1),
        ((63, 1, 65), 2),
        ((100, 0, 300), 2),
        ((63, 1, 65, 30), 2),
    ):
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, sm, arch)
        choice = _carry_choice(N, heads, sm, p.NSEG)
        two_level = arch in CARRY_V8_ARCHES and choice is not None
        assert (p.records["light"]["seg_carry"] == "seg_carry_v8") is two_level
        assert (p.gsize > 0) is two_level
        if not two_level:
            assert p.records["light"]["seg_carry"] == "seg_carry"
            continue
        seen += 1
        assert p.gsize == choice[0]
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        captured: list = []

        def recording(name, values, *, grid, device_index):
            if name == "seg_carry_v8":
                captured.append((dict(values), tuple(grid)))
            elif name.startswith("seg_carry"):
                assert not {"gsize", "carry_out_tiles"} & set(values), name
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = KdaWorkspace.for_problem(S, N, heads, device, launcher=recording)
        epoch0 = ws._epoch[0]
        out, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        grads = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        fwd_carry = p.record("seg_carry", "fwd")
        expected = int(fwd_carry == "seg_carry_v8") + 2 * int(
            p.record("seg_carry", p.light_path) == "seg_carry_v8"
        )
        assert len(captured) == expected >= 2, (lengths, heads)
        # one fresh epoch per stamped carry launch (the overlap pair's forward carry stamps one too)
        epochs = [values["epoch"] for values, _grid in captured]
        assert epochs == sorted(epochs) and len(set(epochs)) == len(epochs)
        assert ws._epoch[0] - epoch0 == len(captured) + int(fwd_carry == "seg_carry_ov")
        for values, grid in captured:
            assert grid == (choice[1], 1, 1) == p.grid("seg_carry", p.light_path)
            assert values["gsize"] == p.gsize
            assert values["flags"].data_ptr() == ws.flags.data_ptr()
            assert values["flags"].numel() >= cake_jit.flag_words(grid[0])
            tiles = values["carry_out_tiles"]
            assert tiles.shape == (p.NSEG * heads, D, D)
            assert tiles.data_ptr() == values["carry_out"].data_ptr()
        # the results are the exact reference's
        ref_out, ref_grads = kda_exact_with_grads(q, k, v, g, beta, dout, cu, SCALE)
        names = ("dq", "dk", "dv", "dbeta", "dg")
        candidates = {"out": out, **dict(zip(names, grads, strict=True))}
        references = {"out": ref_out, **ref_grads}
        for name in OUTPUT_NAMES:
            gate = elementwise_gate(candidates[name], references[name], ATOL, RTOL)
            assert gate["pass"], (lengths, heads, name)
    assert (seen > 0) is (arch in CARRY_V8_ARCHES)


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


# The issue's pack: four documents of S = 261393 tokens, H = 4 (a segmented row whose light set saves the states).
ISSUE_LENGTHS, ISSUE_HEADS = (120001, 70707, 65537, 5148), 4


def _autograd_step(inputs, **kwargs):
    """One forward and backward through the autograd entry on fresh leaves: ``out`` (detached) and the five
    gradients in ``(dq, dk, dv, dbeta, dg)`` order.  The backward accumulates into the leaves: the graph task of
    ``torch.autograd.grad`` holds the gradients it returns until the autograd engine's device thread drops it,
    which can be after the call returned, so a memory check right after it would count them."""
    leaves = [
        inputs[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    out = cake_backend.chunk_kda(*leaves, inputs["cu_seqlens"], **kwargs)
    out.backward(inputs["dout"])
    dq, dk, dv, dg, dbeta = (leaf.grad for leaf in leaves)
    return out.detach(), (dq, dk, dv, dbeta, dg)


def test_the_default_entry_plans_a_repeated_problem_once_and_keeps_no_device_memory(
    device, manifest, monkeypatch
) -> None:
    """The default entry (no workspace) plans a repeated problem once and reuses its workspaces: no record is
    selected and no workspace is built again on a repeat, under every arena option.  Its arena, eye and
    ``doc_lengths`` copy live for the call alone, so the kept workspaces hold no device memory; a call on another
    stream runs in a workspace of its own; the results are bitwise those of a caller-owned workspace."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    plan = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
    assert plan.arch == cake_jit.device_arch(device.index)
    for arena, doc in (
        (None, None),
        (cake_backend.KdaArena(lean=True, tail_passes=4), None),
        (cake_backend.KdaArena(lean=True, tail_passes=2, batch_tokens=1000), lengths),
    ):
        ws = KdaWorkspace(plan, device, backward=True, arena=arena, doc_lengths=doc)
        ref_out, ref_grads = _autograd_step(inputs, workspace=ws, doc_lengths=doc)
        del ws
        _autograd_step(
            inputs, arena=arena, doc_lengths=doc
        )  # the first call builds the workspaces
        selected: list = []
        original = cake_jit.select_record

        def counting(*args, _sel=selected, _orig=original, **kwargs):
            _sel.append(args)
            return _orig(*args, **kwargs)

        monkeypatch.setattr(cake_jit, "select_record", counting)
        built: list = []
        original_init = KdaWorkspace.__init__

        def counting_init(self, *args, _built=built, _orig=original_init, **kwargs):
            _built.append(args)
            _orig(self, *args, **kwargs)

        monkeypatch.setattr(KdaWorkspace, "__init__", counting_init)
        out, grads = _autograd_step(inputs, arena=arena, doc_lengths=doc)
        assert selected == [] and built == [], arena
        assert _bits_equal(out, ref_out), arena
        _check_bits(
            grads,
            dict(zip(("dq", "dk", "dv", "dbeta", "dg"), ref_grads, strict=True)),
            tag=arena,
        )
        monkeypatch.setattr(cake_jit, "select_record", original)
        monkeypatch.setattr(KdaWorkspace, "__init__", original_init)
        del out, grads, ref_out, ref_grads
        # the baseline holds no kept workspace: building them again and reusing them leaves the device memory as
        # it was, so a kept workspace that held its arena, eye or doc_lengths copy would show here
        cake_backend._entry_pool.clear()
        torch.cuda.synchronize()
        before = torch.cuda.memory_allocated()
        for _ in range(2):
            _autograd_step(inputs, arena=arena, doc_lengths=doc)
        torch.cuda.synchronize()
        assert cake_backend._entry_pool, arena
        assert torch.cuda.memory_allocated() == before, arena
    # a call on another stream takes a workspace of its own (no two queued calls share an eye)
    created = _spy_call_workspaces(monkeypatch)
    keys = []
    original_acquire = cake_backend._entry_acquire

    def keyed(*args, **kwargs):
        ws_, key = original_acquire(*args, **kwargs)
        keys.append(key)
        return ws_, key

    monkeypatch.setattr(cake_backend, "_entry_acquire", keyed)
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        _autograd_step(inputs)
    torch.cuda.current_stream(device).wait_stream(side)
    _autograd_step(inputs)
    assert len(created) == 4 and len(set(keys)) == 4
    assert {key[1] for key in keys} == {
        side.cuda_stream,
        torch.cuda.current_stream(device).cuda_stream,
    }


def test_the_arena_options_never_synchronize_the_host(device, manifest) -> None:
    """The issue's pack through the default entry and through a caller-owned workspace: the lean arena with four
    tail passes and the batched option over two document groups (``doc_lengths``) run the forward and the backward
    without one host synchronization (PyTorch's sync debug mode raises on any it routes), on the first call of a
    problem as on a repeat, with the default arena's results bitwise.  Both groups of the issue's pack are one run
    of the saved state blocks (views and contiguous copies); a second pack above the size rule has a group that is
    not (its block indices ride with the ``doc_lengths`` copy), with the same guarantees."""
    split = _saving_pack(SPLIT_LENGTHS, 4, device)
    for lengths, heads, budget, runs in (
        (ISSUE_LENGTHS, ISSUE_HEADS, 200000, (True, True)),
        (split, 4, split[0] + split[1], (False, True)),
    ):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        ref_out, ref_grads = _autograd_step(inputs)
        reference = dict(zip(("dq", "dk", "dv", "dbeta", "dg"), ref_grads, strict=True))
        plan = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
        assert plan.light_path == "light_states"
        lean = cake_backend.KdaArena(lean=True, tail_passes=4)
        batched = cake_backend.KdaArena(lean=True, tail_passes=4, batch_tokens=budget)
        assert len(plan.batches(batched, lengths)) == 2
        packing = cake_backend._host_packing(
            plan,
            batched,
            tuple(lengths),
            nbytes=plan.arena_bytes(arena=batched, doc_lengths=lengths),
            save_mode="light",
        )
        assert tuple(b is not None for b in packing.blocks) == runs
        options = (
            ((lean, None), (batched, lengths)) if runs[0] else ((batched, lengths),)
        )
        for arena, doc in options:
            ws = KdaWorkspace(plan, device, backward=True, arena=arena, doc_lengths=doc)
            for kwargs in (
                {"arena": arena},
                {"arena": arena},
                {"workspace": ws},
                {"workspace": ws},
            ):
                torch.cuda.synchronize()
                torch.cuda.set_sync_debug_mode("error")
                try:
                    out, grads = _autograd_step(inputs, doc_lengths=doc, **kwargs)
                finally:
                    torch.cuda.set_sync_debug_mode("default")
                tag = (lengths, arena, tuple(kwargs))
                assert _bits_equal(out, ref_out), tag
                _check_bits(grads, reference, tag=tag)
            del ws


def test_plan_without_arch_serves_a_workspace_on_the_device(device, manifest) -> None:
    """A plan made without ``arch`` selects the device's records, so a workspace accepts it on every exported
    target (the plan of another architecture is refused, naming both)."""
    lengths, heads = (100, 640, 77), 4
    S, N = sum(lengths), len(lengths)
    p = cake_backend.plan(S, N, heads, cake_jit.sm_count(device.index))
    assert p.arch == cake_jit.device_arch(device.index)
    ws = KdaWorkspace(p, device)
    inputs = make_inputs(lengths, heads, device)
    out, grads, _ = _run_fwd_bwd(inputs, workspace=ws)
    ref_out, ref_grads, _ = _run_fwd_bwd(inputs)
    assert _bits_equal(out, ref_out)
    _check_bits(list(grads.values()), ref_grads)
    for arch in ARCHES:
        if arch != p.arch:
            other = cake_backend.plan(S, N, heads, p.sm_count, arch)
            with pytest.raises(ValueError, match=f"records of {arch}, the device is"):
                KdaWorkspace(other, device)


def test_a_batched_default_entry_under_capture_asks_for_a_warmed_workspace(
    device, manifest
) -> None:
    """Under CUDA-graph capture the default entry's workspace is the captured call's alone, so a batched arena
    would capture a copy of ``doc_lengths`` from a host buffer the graph does not own: it raises instead,
    naming the caller-owned workspace that ran these ``doc_lengths`` before the capture."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    args = [inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")]
    batched = cake_backend.KdaArena(lean=True, tail_passes=2, batch_tokens=1000)
    # eager first: the programs are loaded and the default entry's own copy is made outside the capture
    cake_backend.chunk_kda_fwd(*args, arena=batched, doc_lengths=lengths)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    refused = pytest.raises(RuntimeError, match="under CUDA-graph capture needs")
    with refused, torch.cuda.graph(graph):
        cake_backend.chunk_kda_fwd(*args, arena=batched, doc_lengths=lengths)


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
        ((129, 300), 16, "light"),  # the paired serial row of sm_100a: w / u saved
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
        reference_rows = {
            name: getattr(saved, name).clone()
            for name in ("qg", "w", "u")
            if getattr(saved, name) is not None
        }
        # the tape forward saves no version-4 rows (its backward recomputes the fold from the tape); the light
        # forward saves qg, and w / u on the rows whose light backward reads the WY representation from the set
        want_rows = (
            set()
            if save_mode == "tape"
            else ({"qg", "w", "u"} if plan.light_saves_wu else {"qg"})
        )
        assert set(reference_rows) == want_rows
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
        for name, reference in reference_rows.items():
            assert torch.equal(getattr(saved, name), reference), (case, name)
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
        # the fold's qg on every light set (no w / u: these rows are no paired 16-head rows)
        assert saved.qg.shape == (p.R, heads, D) and saved.qg.dtype == torch.bfloat16
        assert saved.w is None and saved.u is None and not p.light_saves_wu
        arena_lo, arena_hi = ws.arena.data_ptr(), ws.arena.data_ptr() + ws.bytes
        assert not (arena_lo <= saved.qg.data_ptr() < arena_hi)
        qg_bytes = p.R * heads * D * 2
        expected_bytes = 256 * S * heads + tables + qg_bytes
        if saves:
            expected_bytes += 3 * block
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
            for t in (saved.hseg, saved.seg_state):
                assert not (arena_lo <= t.data_ptr() < arena_hi)
            owned = (saved.Aqk, saved.Akk, saved.qg, saved.hseg, saved.seg_state)
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
        # (the gate epilogue is one more launch on sm_107a)
        gate = 0 if p.gate_fused else 1
        assert len(launched) == p.launch_count("bwd") == (7 if saves else 10) + gate
        assert launched[1:3] == (
            ["fwdh_p3_wy", "dhu_v3s_f_wt"] if saves else ["wy_bwd", "fwdh_v3"]
        )
        assert ("dhu_v3s_f_wt" if saves else "dhu_v3s_f") in launched
        # (b) the gradients from the saved states equal the checkpoint replay's bitwise, and those from the same
        # set stripped of its states (the recompute path) and from a tape-mode workspace
        replay = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=None, workspace=ws
        )
        stripped = dataclasses.replace(saved, hseg=None, seg_state=None)
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
        # the tape set is unchanged by the rule: version 2 with seg_m, without the version-4 fields
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        assert saved_t.version == cake_backend.SAVED_VERSION_TAPE
        assert (
            saved_t.seg_m is not None
            and saved_t.hseg is None
            and saved_t.seg_state is None
        )
        assert saved_t.qg is None and saved_t.w is None and saved_t.u is None
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
    # layouts) recomputes h / v_new into per-call buffers for a version-4 saved set, for the replay and for a
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
    """The tail's launches: the dispatched tail stages (two where the gate epilogue is fused, three on sm_107a)
    once per range, each with the grid of the plan's rule at the range's chunk count (one pass: the plan's
    grids)."""
    stages = [stage for stage in TAIL_STAGES if p.dispatched(stage, path)]
    tail_names = tuple(p.record(stage, path) for stage in stages)
    width = len(tail_names)
    assert width == 2 + int(not p.gate_fused)
    tail = [(name, grid) for name, grid in launched if name in tail_names]
    ranges = cake_backend._tail_ranges(p.NC_pad, tail_passes)
    assert len(tail) == width * len(ranges), (path, tail_passes)
    for i, (_c0, length) in enumerate(ranges):
        for j, (stage, name) in enumerate(zip(stages, tail_names, strict=True)):
            got_name, grid = tail[width * i + j]
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
    """The lean workspace's launch list is the default's with the tail stages once per chunk range, each at the
    grid of the plan's rule for the range's chunk count; one pass is the default list with the default grids
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
            assert len(names) == p.launch_count("light") + len(
                _tail_records(p, "light")
            ) * (len(cake_backend._tail_ranges(p.NC_pad, t)) - 1)
            _check_tail_grids(launched, p, "light", t)
            _check_bits(got, reference_grads, tag=(lengths, heads, t))
            # the tail without the saved tables rebuilds them first, as the default does
            launched.clear()
            bare = dataclasses.replace(saved, tables=None)
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
    version-2 set on a lean tape workspace; a version-4 set on a lean tape workspace, which recomputes ``h`` /
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
    assert saved.qg is not None
    created = _spy_call_workspaces(monkeypatch)
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
        # the lean tape workspace: its own version-2 set (the tape path) and the version-4 set (per-call h / v_new)
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
    created = _spy_call_workspaces(monkeypatch)

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
        ):  # a version-4 set on the poisoned tape workspace: per-call h / v_new
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
    tape bound even and inside the whole problem's tape), a plan with the whole problem's records, ``P``, route,
    gate fusion and two-level carry group size, and the grids of the records' rules at the group's scalars (the
    two-level carry's at the pinned group size: ``N_b * H * ctas(P, gsize)`` CTAs); the arena of the exact groups
    never exceeds the bound of every group, which the byte query gives without ``doc_lengths``."""
    sm = 148
    pinned = 0
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
                        assert (pb.gsize, pb.gate_fused) == (p.gsize, p.gate_fused)
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
                                    gsize=p.gsize,
                                )
                                for launch in range(cake_jit.launches(name))
                            )
                        if "seg_carry_v8" in pb.grids:
                            pinned += 1
                            assert p.gsize == _carry_choice(N, heads, sm, p.NSEG)[0]
                            ctas = _carry_ctas(p.P, p.gsize)
                            group_grid = pb.grids["seg_carry_v8"]
                            assert group_grid == ((b.N * heads * ctas, 1, 1),)
                            assert group_grid[0][0] <= p.grids["seg_carry_v8"][0][0]
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
    # the six-chain row at 2 heads pins the two-level carry's group size (on every architecture)
    assert pinned > 0


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
            # the largest table buffer of a group and three state blocks per segment of the group with the most
            # segments (here the two-document group, not the one with the most chunks)
            segments_b = max(p.batch_segments(b) for b in groups)
            if p.recur == "seg":
                assert (
                    segments_b
                    == p.batch_segments(groups[0])
                    > max(p.batch_segments(b) for b in groups[1:])
                )
            blocks = 3 * segments_b * heads * D * D * 4
            words = max(
                2 * b.NC_pad + 2 * b.N + 2 + p.batch_segments(b) + 2 * b.N + 1 + 5 * 64
                for b in groups
            )
            assert blocks <= r2["batch_scratch"] <= blocks + 4 * words
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


def _group_rows_sliced(path: str) -> bool:
    """Whether a batched call hands every group slices of whole-problem row tensors (the saving forward's qg / w /
    u or tape, the saved-set backward's saved rows): the groups then take the tape geometry
    (``batch_view(tape=True)``, the group's chunk bound inside the whole problem's rows).  The forward without a
    saved set and the replay slice nothing."""
    return path not in ("fwd_nosave", "replay")


def _fewest_tail_passes(pb, arena, nbytes, save_mode="light") -> int:
    """The tail passes of a group's backward (group plan ``pb``) on an arena of ``nbytes``, the package's rule
    restated: the fewest up to the option's ``tail_passes`` whose lean layouts fit (``bwd_tape_lean`` on a
    tape-mode workspace, also ``bwd_lean`` on a light one), else the option's."""
    modes = ("bwd_tape_lean",) if save_mode == "tape" else ("bwd_lean", "bwd_tape_lean")
    return next(
        (
            t
            for t in range(1, arena.tail_passes)
            if all(pb.workspace_bytes(mode, t) <= nbytes for mode in modes)
        ),
        arena.tail_passes,
    )


def _group_passes(p, arena, lengths, path, *, nbytes, save_mode="light") -> list[int]:
    """Per group of a batched call on ``path``, the tail passes of its backward (:func:`_fewest_tail_passes`)."""
    return [
        _fewest_tail_passes(
            p.batch_plan(b, tape=_group_rows_sliced(path)), arena, nbytes, save_mode
        )
        for b in p.batches(arena, lengths)
    ]


def _expected_batched_launches(p, arena, lengths, path, *, passes=None):
    """The launch order of a batched call: on the saving forward the whole problem's tables by the saved-set
    paths' table program, then every group's forward list (its own tables where the row dispatches a table program
    on the forward); on the backward every group's list with the group's tables first (the saved set's describe the
    whole problem) and the tail once per range of the group's chunk bound at the group's tail passes (``passes``,
    one per group: :func:`_group_passes`; ``None``: one pass each)."""
    seq = []
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    if path in ("fwd", "fwd_nosave"):
        if path == "fwd" and p.dispatched("table", "light"):
            seq.append(p.record("table", "light"))
            if p.recur == "seg" and p.dispatched("seg_table", "light"):
                seq.append("seg_table")
        for b in batches:
            pb = p.batch_plan(b, tape=_group_rows_sliced(path))
            seq += _expected_launches(pb, path)
        return seq
    for b, t in zip(batches, passes, strict=True):
        seq += _expected_launches(
            p.batch_plan(b, tape=_group_rows_sliced(path)),
            path,
            saved_tables=False,
            tail_passes=t,
        )
    return seq


def _check_group_launches(launched, p, arena, lengths, path, *, passes=None) -> None:
    """Every launch of a group carries the group's scalars (``num_seqs``, ``num_chunks_pad``) and one of the
    group plan's grids; the tail passes (``passes``, one per group, ``None``: one pass each) carry
    ``num_seqs = 0`` and their range's chunk count."""
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    forward = path in ("fwd", "fwd_nosave")
    pos = 0
    if forward and path == "fwd" and p.dispatched("table", "light"):
        head = 2 if (p.recur == "seg" and p.dispatched("seg_table", "light")) else 1
        for name, grid, num_seqs, nc in launched[:head]:
            assert (num_seqs, nc) == (p.N, p.NC_pad) and grid in p.grids[name], name
        pos = head
    for b, t in zip(batches, passes, strict=True):
        pb = p.batch_plan(b, tape=_group_rows_sliced(path))
        expected = (
            _expected_launches(pb, path)
            if forward
            else _expected_launches(pb, path, saved_tables=False, tail_passes=t)
        )
        group = launched[pos : pos + len(expected)]
        pos += len(expected)
        assert [n for n, *_ in group] == expected
        tail_names = set() if forward else set(_tail_records(pb, path))
        width = len(tail_names)
        ranges = cake_backend._tail_ranges(pb.NC_pad, t)
        tails = []
        for name, grid, num_seqs, nc in group:
            if name in tail_names and t > 1:
                tails.append((name, grid, num_seqs, nc))
                continue
            assert (num_seqs, nc) == (pb.N, pb.NC_pad), (name, b)
            assert grid in pb.grids[name], (name, b, grid, pb.grids[name])
        if tails:
            assert len(tails) == width * len(ranges)
            for i, (_c0, length) in enumerate(ranges):
                for j in range(width):
                    name, grid, num_seqs, nc = tails[width * i + j]
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


def _check_tail_counts(counts, p, arena, lengths, path, *, passes=None) -> None:
    """The one-entry chunk counts of a batched backward's tail launches (``seq_chunk_start`` at ``num_seqs = 0``, in
    launch order): every launch of a group's range ``c0`` reads ``clamp(real_b - c0, 0, ceil(NC_pad_b / b))`` at
    the group's tail passes ``b`` (``passes``, one per group, ``None``: one pass each), the device rule the host
    reproduces from ``doc_lengths``; none for a group in one pass."""
    batches = p.batches(arena, lengths)
    passes = [1] * len(batches) if passes is None else passes
    torch.cuda.synchronize()
    pos = 0
    for b, t in zip(batches, passes, strict=True):
        if t == 1:
            continue
        pb = p.batch_plan(b, tape=_group_rows_sliced(path))
        width = len(_tail_records(pb, path))
        real = _real_chunks(lengths[b.d0 : b.d1])
        ranges = cake_backend._tail_ranges(pb.NC_pad, t)
        for c0, _length in ranges:
            want = min(max(real - c0, 0), ranges[0][1])
            for _ in range(width):
                assert counts[pos].tolist() == [want], (b, c0)
                pos += 1
    assert pos == len(counts)
    assert (pos > 0) == any(t > 1 for t in passes)


def _real_chunks(lengths) -> int:
    return sum(-(-int(n) // BT) for n in lengths)


@pytest.mark.parametrize("shape", D_ROWS, ids=_shape_id)
def test_batched_arena_gives_the_default_results_bitwise(
    device, manifest, shape, monkeypatch
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
    ref_h, ref_v_new = saved_t.h[:real].clone(), saved_t.v_new.clone()
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
            # the version-4 rows the groups write as slices: the fold's qg, and w / u on the rows that save them
            # (the real token rows; the rows above BT * real are the chunk padding, which the whole-problem and the
            # per-group forwards fill differently and no backward reads)
            rows = slice(0, BT * real)
            assert _bits_equal(saved_b.qg[rows], saved.qg[rows]), tag
            assert (saved_b.w is None) == (saved.w is None), tag
            assert (saved.w is None) is not p.light_saves_wu, tag
            if saved.w is not None:
                assert _bits_equal(saved_b.w[rows], saved.w[rows]), tag
                assert _bits_equal(saved_b.u[rows], saved.u[rows]), tag
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
            # the saved v_new comes from its allocation filled with NaN: a row no group writes would show
            original_alloc, poisoned = cake_backend._alloc, []

            def nan_v_new(
                shape_,
                dtype,
                device_,
                _orig=original_alloc,
                _hit=poisoned,
                _ref=ref_v_new,
            ):
                t = _orig(shape_, dtype, device_)
                if dtype == _ref.dtype and tuple(shape_) == tuple(_ref.shape):
                    t.view(torch.uint8).fill_(0xFF)
                    _hit.append(t.data_ptr())
                return t

            monkeypatch.setattr(cake_backend, "_alloc", nan_v_new)
            o, saved_tb = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, workspace=tws, doc_lengths=lengths
            )
            monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
            assert poisoned == [saved_tb.v_new.data_ptr()], tag
            assert (
                _bits_equal(o, ref_out)
                and saved_tb.version == cake_backend.SAVED_VERSION_TAPE
            ), tag
            assert _bits_equal(saved_tb.h[:real], ref_h), tag
            # v_new whole, its pad rows included: zero in both (the rows after the last group's slice too)
            assert _bits_equal(saved_tb.v_new, ref_v_new), tag
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
    forward; the group's tables first on the backward), each launch at the group's scalars and grids (the saving
    forward's groups in the tape geometry: they write slices of the saved qg / w / u); the overlap pair's and the
    two-level carry's epoch advances once per group through the shared counter; every group runs its tail in the
    fewest passes that fit the arena (groups in several passes and in one, within a call too), whose launches read
    the host's chunk counts of their ranges; the group tables are slices of the whole problem's."""
    seen = {"multi": 0, "one_pass": 0, "mixed": 0}
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
        counts: list = []  # the tail passes' one-entry chunk counts (num_seqs = 0), in launch order

        def recording(name, values, *, grid, device_index):
            launched.append(
                (
                    name,
                    tuple(grid),
                    int(values["num_seqs"]),
                    int(values["num_chunks_pad"]),
                )
            )
            if int(values["num_seqs"]) == 0:
                counts.append(values["seq_chunk_start"])
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
                stamped = p.recur == "seg" and p.record("seg_carry", "fwd") in (
                    "seg_carry_ov",
                    "seg_carry_v8",
                )
                assert ws._epoch[0] - epoch0 == (len(batches) if stamped else 0), case
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
                counts.clear()
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
                # every group's tail in the fewest passes up to t that fit the arena, as its view has it
                passes = _group_passes(p, arena, lengths, path, nbytes=ws.bytes)
                assert [
                    ws.batch_view(b, tape=_group_rows_sliced(path)).tail_passes
                    for b in batches
                ] == passes, case
                if t > 1:
                    seen["multi"] += sum(x > 1 for x in passes)
                    seen["one_pass"] += passes.count(1)
                    seen["mixed"] += len(set(passes)) > 1
                names = [n for n, *_ in launched]
                assert names == _expected_batched_launches(
                    p, arena, lengths, path, passes=passes
                ), case
                _check_group_launches(launched, p, arena, lengths, path, passes=passes)
                _check_tail_counts(counts, p, arena, lengths, path, passes=passes)
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
    assert min(seen.values()) > 0, seen


def test_batched_arena_on_a_row_that_saves_the_states(device, manifest) -> None:
    """A pack above the size rule (its light set carries the states) whose last document alone would be below it:
    the groups follow the whole problem's rule, the forward writes their blocks into the saved states by chain
    rank, the backward reads them there (``light_states`` as pass 2 alone; the documents rank in pack order, so
    every group's blocks are one run of the saved ones: the carry writes the saved ``hseg``, the backward reads
    views), everything bit for bit."""
    lengths, heads = (40000, 20000, 20000, 1000), 4
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    # The size rule is S >= SEG_SAVE_MIN_TOKENS_PER_SEG * NSEG with NSEG = the SM count here, so on a device
    # with more SMs than this pack covers (216 on some sm_107a boards) the first document grows until the
    # plan saves the states; the last document alone stays below the rule.
    for _ in range(64):
        if cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch).seg_save:
            break
        lengths = (lengths[0] + 4096, *lengths[1:])
    S, N = sum(lengths), len(lengths)
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


def test_batched_arena_with_exact_zero_values(device, manifest, monkeypatch) -> None:
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
        ref_h, ref_v_new = saved_t.h[:real].clone(), saved_t.v_new.clone()
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
            # the saved v_new comes from its allocation filled with NaN: a row no group writes would show
            original_alloc, poisoned = cake_backend._alloc, []

            def nan_v_new(
                shape_,
                dtype,
                device_,
                _orig=original_alloc,
                _hit=poisoned,
                _ref=ref_v_new,
            ):
                t = _orig(shape_, dtype, device_)
                if dtype == _ref.dtype and tuple(shape_) == tuple(_ref.shape):
                    t.view(torch.uint8).fill_(0xFF)
                    _hit.append(t.data_ptr())
                return t

            monkeypatch.setattr(cake_backend, "_alloc", nan_v_new)
            o, saved_tb = cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu_t, workspace=tws, doc_lengths=lengths
            )
            monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
            assert poisoned == [saved_tb.v_new.data_ptr()], tag
            assert _bits_equal(o, ref_out), tag
            assert _bits_equal(saved_tb.h[:real], ref_h), tag
            # v_new whole, its pad rows included: zero in both (the rows after the last group's slice too)
            assert _bits_equal(saved_tb.v_new, ref_v_new), tag
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
    created = _spy_call_workspaces(monkeypatch)

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
    carries the states (the groups read the saved tensors, not the arena; pass 2 never touches the arena's
    ``seg_state`` it is bound to at ``write_ht = 0``)."""
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
        ):  # a version-4 set on the poisoned tape workspace: per-call h / v_new per group
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


# ------------------------------------------------------------------------------------------------------------------
# The batched arena option: the groups' state blocks by the host's chain ranks

# A pack above the size rule whose first group's state blocks are not one run of the whole problem's: document 1
# ranks last (16 chunks) and documents 2 and 3 between, so under ``batch_tokens = L0 + 1000`` the group (0, 2) holds
# ranks 0 and 3 (gathered and scattered by index) and the group (2, 4) ranks 1 and 2 (one run).  The first document
# grows with the device's segment count (_saving_pack).
SPLIT_LENGTHS = (40000, 1000, 20000, 20000)
# Packs of the host chain ranks: the issue's pack, tied chunk counts, zero-length documents, interleaved ties (no
# group of two or more documents is one run), one document, a small pack whose first group is not one run under
# ``batch_tokens = 704`` and 300 documents (the one-CTA ``table`` program; ties and two zero-length documents).
RANK_PACKS = (
    (ISSUE_LENGTHS, ISSUE_HEADS),
    ((128, 100, 128, 65, 1, 64), 2),
    ((100, 0, 300, 0, 64), 2),
    ((64, 640, 1, 600, 63, 577), 2),
    ((1000,), 8),
    ((640, 64, 320), 2),
    (tuple((37 * i + 11) % 130 for i in range(300)), 1),
)
# the groupings beyond _partitions a pack runs under (the issue's two groups of the lean four-pass budget, the
# groups that are not one run)
RANK_GROUPINGS = {
    ISSUE_LENGTHS: ({"batch_tokens": 200000},),
    (640, 64, 320): ({"batch_tokens": 704},),
    (1000,): ({"batch_tokens": 1000},),
}


def _saving_pack(lengths, heads, device) -> tuple[int, ...]:
    """``lengths`` with the first document grown (4096 tokens a step) until the plan on this device saves the
    segment states (``S >= 384 * NSEG``: on a device with more SMs the segment count grows)."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    lengths = tuple(lengths)
    for _ in range(64):
        if cake_backend.plan(sum(lengths), len(lengths), heads, sm, arch).seg_save:
            return lengths
        lengths = (lengths[0] + 4096, *lengths[1:])
    raise AssertionError(f"no pack from {lengths} saves the states on this device")


def _kernel_ranks(lengths) -> list[int]:
    """The chain rank every ``seq_order`` writer gives each document (``table``, ``table_v2``, ``table_v3`` and the
    segmented intra program ``fwd_intra_v14``, their loop spelled out): one per document with more chunks, one per
    earlier document with as many."""
    nc = [(int(n) + BT - 1) // BT for n in lengths]
    return [
        sum(1 for j, m in enumerate(nc) if m > nc[d] or (m == nc[d] and j < d))
        for d in range(len(nc))
    ]


def _expected_group_blocks(whole, lengths, batch, P) -> list[int]:
    """The whole problem's state blocks of one group in the group's own block order, by the kernels' layout: the
    group's tables rank its documents among themselves, the whole problem's among all (``whole``, the
    :func:`_kernel_ranks` of ``lengths``), ``P`` blocks per chain."""
    local = _kernel_ranks(lengths[batch.d0 : batch.d1])
    order = sorted(range(batch.N), key=lambda d: local[d])
    return [P * whole[batch.d0 + d] + s for d in order for s in range(P)]


def _one_run(blocks) -> bool:
    return blocks == list(range(blocks[0], blocks[0] + len(blocks)))


def test_host_chain_ranks_and_block_runs_follow_the_tables_rule() -> None:
    """The host's chain ranks (``_seq_order`` / ``_chain_ranks``) are the rank every ``seq_order`` writer computes,
    and a batched packing (``_Packing``) holds the kernels' block layout: per group the whole problem's first block
    where the group's blocks are one run of the whole problem's, else their indices, and the tail passes' chunk
    counts of the device rule at both of the group's chunk bounds.  On every architecture, over the rank packs and
    their groupings, with the counts at each group's own tail passes (:func:`_fewest_tail_passes`; none for a
    group in one pass); the issue's two groups of the lean four-pass budget are the runs [0, 76) and [76, 152) on
    a 152-SM ``sm_100a``, and the serial route has no blocks."""
    seen = {"run": 0, "index": 0, "tail": 0, "one_pass": 0}
    for lengths, heads in RANK_PACKS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        ranks = _kernel_ranks(lengths)
        order = cake_backend._seq_order(lengths)
        assert cake_backend._chain_ranks(lengths) == tuple(ranks), lengths
        assert sorted(order) == list(range(N)), lengths
        assert all(ranks[d] == r for r, d in enumerate(order)), lengths
        groupings = list(RANK_GROUPINGS.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        for arch in ARCHES:
            p = cake_backend.plan(S, N, heads, 148, arch)
            assert p.recur == "seg" and not p.proportional, (lengths, arch)
            for kw in groupings:
                for t in (1, 4):
                    arena = cake_backend.KdaArena(lean=t > 1, tail_passes=t, **kw)
                    nbytes = p.arena_bytes(arena=arena, doc_lengths=lengths)
                    packing = cake_backend._host_packing(
                        p, arena, lengths, nbytes=nbytes, save_mode="light"
                    )
                    case = (lengths, arch, kw, t)
                    assert packing.batches == p.batches(arena, lengths), case
                    assert packing.words[:N] == lengths and packing.P == p.P, case
                    assert packing.words_dev is None and packing.index_dev is None
                    for i, b in enumerate(packing.batches):
                        blocks = _expected_group_blocks(ranks, lengths, b, p.P)
                        assert len(blocks) == b.N * p.P, (case, b)
                        if _one_run(blocks):
                            assert packing.blocks[i] == blocks[0], (case, b)
                            assert i not in packing.index_at, (case, b)
                            seen["run"] += 1
                        else:
                            assert packing.blocks[i] is None, (case, b)
                            first, n = packing.index_at[i]
                            assert list(packing.index[first : first + n]) == blocks
                            seen["index"] += 1
                        real = _real_chunks(lengths[b.d0 : b.d1])
                        for nc_pad in (b.NC_pad, b.NC_pad_tape):
                            pb = p.batch_plan(b, tape=nc_pad != b.NC_pad)
                            passes = _fewest_tail_passes(pb, arena, nbytes)
                            if passes == 1:
                                assert (i, nc_pad) not in packing.tail_at, (case, b)
                                seen["one_pass"] += t > 1
                                continue
                            ranges = cake_backend._tail_ranges(nc_pad, passes)
                            want = [
                                min(max(real - c0, 0), ranges[0][1])
                                for c0, _n in ranges
                            ]
                            first, n = packing.tail_at[(i, nc_pad)]
                            got = list(packing.words[first : first + n])
                            assert got == want, (case, b, nc_pad)
                            seen["tail"] += 1
    assert min(seen.values()) > 0, seen
    # the issue's pack under the lean four-pass budget of two groups: the second starts at an odd chunk, both are
    # one run
    S = sum(ISSUE_LENGTHS)
    p = cake_backend.plan(S, len(ISSUE_LENGTHS), ISSUE_HEADS, 152, "sm_100a")
    arena = cake_backend.KdaArena(lean=True, tail_passes=4, batch_tokens=200000)
    packing = cake_backend._host_packing(
        p,
        arena,
        ISSUE_LENGTHS,
        nbytes=p.arena_bytes(arena=arena, doc_lengths=ISSUE_LENGTHS),
        save_mode="light",
    )
    assert [(b.d0, b.d1, b.c0) for b in packing.batches] == [(0, 2, 0), (2, 4, 2981)]
    assert (p.P, p.NSEG, packing.blocks, packing.index) == (38, 152, (0, 76), ())
    # the split pack: its first group by index, its second one run from the second chain
    for arch in ARCHES:
        p = cake_backend.plan(sum(SPLIT_LENGTHS), 4, 4, 148, arch)
        arena = cake_backend.KdaArena(batch_tokens=SPLIT_LENGTHS[0] + SPLIT_LENGTHS[1])
        packing = cake_backend._host_packing(
            p,
            arena,
            SPLIT_LENGTHS,
            nbytes=p.arena_bytes(arena=arena, doc_lengths=SPLIT_LENGTHS),
            save_mode="light",
        )
        assert [(b.d0, b.d1) for b in packing.batches] == [(0, 2), (2, 4)]
        assert packing.blocks == (None, p.P), arch
        assert packing.index == tuple(range(p.P)) + tuple(range(3 * p.P, 4 * p.P))
    # the serial route: no state blocks, no indices
    p = cake_backend.plan(129, 3, 64, 148, ARCHES[0])
    assert p.recur == "serial"
    arena = cake_backend.KdaArena(batch_tokens=65)
    packing = cake_backend._host_packing(
        p,
        arena,
        (63, 1, 65),
        nbytes=p.arena_bytes(arena=arena, doc_lengths=(63, 1, 65)),
        save_mode="light",
    )
    assert packing.blocks == (None, None)
    assert packing.index == () and not packing.index_at


def test_batched_groups_take_the_fewest_tail_passes_that_fit() -> None:
    """Under a batched option with ``tail_passes = B`` every group runs its tail in the fewest passes up to ``B``
    whose lean layouts at the group's geometry fit the arena (``_group_tail_passes``, which the group views and the
    host's chunk counts take): the restated rule's count, between one and ``B``, fitting the arena, no fewer
    fitting; the bound of every group (a larger arena) never takes more than the exact groups.  Over the rank packs
    and their groupings on every architecture, light and tape, groups in one pass and in several, within one call
    too.  The issue's pack on a 152-SM ``sm_100a`` under the lean four-pass budget of two groups: the first group,
    which sizes the arena, keeps four passes and the second fits one (three and one on the bound's arena), so the
    host's counts are the first group's alone and the tail launches five times per stage instead of eight."""
    seen = {"multi": 0, "one_pass": 0, "mixed": 0}
    for lengths, heads in RANK_PACKS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        groupings = list(RANK_GROUPINGS.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        for arch in ARCHES:
            p = cake_backend.plan(S, N, heads, 148, arch)
            for kw in groupings:
                for t in (1, 2, 3, 4):
                    arena = cake_backend.KdaArena(lean=t > 1, tail_passes=t, **kw)
                    batches = p.batches(arena, lengths)
                    for save_mode in ("light", "tape"):
                        modes = ("bwd_tape_lean",)
                        if save_mode == "light":
                            modes += ("bwd_lean",)
                        sized = {
                            sizing: p.arena_bytes(
                                save_mode=save_mode, arena=arena, doc_lengths=doc
                            )
                            for sizing, doc in (("exact", lengths), ("bound", None))
                        }
                        got = {}
                        for (sizing, nbytes), b in itertools.product(
                            sized.items(), batches
                        ):
                            for nc_pad in sorted({b.NC_pad, b.NC_pad_tape}):
                                pb = p.batch_plan(b, tape=nc_pad != b.NC_pad)
                                passes = cake_backend._group_tail_passes(
                                    p,
                                    b,
                                    nc_pad,
                                    arena,
                                    nbytes=nbytes,
                                    save_mode=save_mode,
                                )
                                case = (lengths, arch, kw, t, save_mode, sizing, b)
                                fits = [
                                    all(
                                        pb.workspace_bytes(m, x) <= nbytes
                                        for m in modes
                                    )
                                    for x in range(1, t + 1)
                                ]
                                assert passes == _fewest_tail_passes(
                                    pb, arena, nbytes, save_mode
                                ), case
                                assert 1 <= passes <= t and fits[passes - 1], case
                                assert not any(fits[: passes - 1]), case
                                got[(sizing, b, nc_pad)] = passes
                        for (sizing, b, nc_pad), passes in got.items():
                            assert passes <= got[("exact", b, nc_pad)], (lengths, t)
                            if t > 1 and sizing == "exact":
                                seen["multi" if passes > 1 else "one_pass"] += 1
                        if t > 1:
                            seen["mixed"] += (
                                len({got[("exact", b, b.NC_pad)] for b in batches}) > 1
                            )
    assert min(seen.values()) > 0, seen
    # the issue's pack: four passes for the group that sizes the arena, one for the other
    S = sum(ISSUE_LENGTHS)
    p = cake_backend.plan(S, len(ISSUE_LENGTHS), ISSUE_HEADS, 152, "sm_100a")
    arena = cake_backend.KdaArena(lean=True, tail_passes=4, batch_tokens=200000)
    batches = p.batches(arena, ISSUE_LENGTHS)
    assert [(b.NC_pad, b.NC_pad_tape) for b in batches] == [(2982, 2982), (1108, 1108)]
    light = p.arena_bytes(arena=arena, doc_lengths=ISSUE_LENGTHS)
    assert light == 2212534272
    assert p.batch_plan(batches[0]).workspace_bytes("bwd_lean", 3) > light
    for save_mode in ("light", "tape"):
        for doc, want in ((ISSUE_LENGTHS, [4, 1]), (None, [3, 1])):
            nbytes = p.arena_bytes(save_mode=save_mode, arena=arena, doc_lengths=doc)
            got = [
                cake_backend._group_tail_passes(
                    p, b, b.NC_pad, arena, nbytes=nbytes, save_mode=save_mode
                )
                for b in batches
            ]
            assert got == want, (save_mode, doc, got)
    packing = cake_backend._host_packing(
        p, arena, ISSUE_LENGTHS, nbytes=light, save_mode="light"
    )
    assert packing.tail_at == {(0, 2982): (4, 4)}
    assert packing.words[4:8] == (746, 746, 746, 743)


# the device rows of the host ranks: every seq_order writer (table_v3 and fwd_intra_v14 on the segmented route up to
# 256 documents, table_v2 on the serial route there, table above 256 documents)
RANK_ROWS = (*RANK_PACKS, ((128, 100, 128, 65, 1, 64), 32))


def test_host_chain_ranks_match_every_seq_order_writer(
    device, manifest, monkeypatch
) -> None:
    """The host's ``seq_order`` is the table every writer leaves on the device, and its chain ranks the ``argsort``
    of that table (the rule of the device glue it replaces): the saved-set paths' table program (``table_v3`` /
    ``table_v2`` / ``table``) and the forward's own tables (the segmented intra program ``fwd_intra_v14`` up to
    256 documents, ``table_v2`` on the serial route), over tied, zero-length, interleaved, one-document and
    300-document packs."""
    original = cake_backend._tables
    writers: set = set()
    compared = 0
    for lengths, heads in RANK_ROWS:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens")
        )
        ws = KdaWorkspace.for_problem(S, N, heads, device, backward=False)
        p = ws.plan
        written = [(p.record("table", "light"), original(ws, cu, "light"))]
        # the forward's own tables (above 256 documents the same table program writes them)
        if N <= 256:
            seen: list = []

            def spy(*args, _seen=seen, **kwargs):
                tables = original(*args, **kwargs)
                _seen.append(tables)
                return tables

            monkeypatch.setattr(cake_backend, "_tables", spy)
            cake_backend.chunk_kda_fwd(
                q, k, v, g, beta, cu, save_for_backward=False, workspace=ws
            )
            monkeypatch.setattr(cake_backend, "_tables", original)
            assert len(seen) == 1, lengths
            fwd = "fwd_nosave"
            writer = (
                p.record("table", fwd)
                if p.dispatched("table", fwd)
                else p.record("fwd_intra", fwd)
            )
            written.append((writer, seen[0]))
        torch.cuda.synchronize()
        want = list(cake_backend._seq_order(lengths))
        ranks = list(cake_backend._chain_ranks(lengths))
        assert ranks == _kernel_ranks(lengths), lengths
        for writer, tables in written:
            assert tables[3][:N].tolist() == want, (lengths, heads, writer)
            assert torch.argsort(tables[3][:N]).tolist() == ranks, (lengths, writer)
            writers.add(writer)
            compared += 1
        del ws
    assert writers == {"table", "table_v2", "table_v3", "fwd_intra_v14"}, writers
    assert compared == len(RANK_ROWS) + sum(
        1 for lengths, _heads in RANK_ROWS if len(lengths) <= 256
    )


def test_batched_block_runs_give_the_default_results_bitwise(
    device, manifest, monkeypatch
) -> None:
    """The groups' state blocks by the host's chain ranks give the default's results bit for bit on both sides of
    the run test: the saving forward's carry writes a one-run group's carried states straight into the saved
    ``hseg`` (its offsets / transfers two contiguous copies; the tape's transfers one) and scatters the other groups'
    by index; the backward reads views of a one-run group's saved ``hseg`` and transfers (``light_states``: its pass
    2 bound to the arena's ``seg_state``, poisoned here with the rest of the arena) or ``seg_m`` (``tape``) and
    gathers the others'; the lean tail passes take the host's chunk counts, a group whose one-pass lean layout fits
    the arena running its tail in one pass.  Over the issue's pack (two groups, the
    second at an odd chunk; one document per group), a pack above the size rule with a group that is not one run,
    tied chunk counts, zero-length documents, interleaved ties, one document and a small pack with a group that is
    not one run, under their groupings.  The saved state blocks start as NaN (every block must be written) and
    every comparison is counted."""
    sm, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    split = _saving_pack(SPLIT_LENGTHS, 4, device)
    cases = [(ISSUE_LENGTHS, ISSUE_HEADS, (1, 4)), (split, 4, (1, 4))] + [
        (lengths, heads, (1, 3))
        for lengths, heads in RANK_PACKS[1:6]  # the smaller packs (no 300-document one)
    ]
    groupings_of = {**RANK_GROUPINGS, split: ({"batch_tokens": split[0] + split[1]},)}
    original_alloc = cake_backend._alloc

    def poisoned_alloc(shape, dtype, device_):
        t = original_alloc(shape, dtype, device_)
        # the saved state blocks (hseg, seg_state, seg_m): NaN until a kernel or a copy writes them
        if dtype == torch.float32 and tuple(shape[-2:]) == (D, D):
            t.view(torch.uint8).fill_(0xFF)
        return t

    hits: dict = {}
    total = 0
    for lengths, heads, passes in cases:
        lengths = tuple(lengths)
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, sm, arch)
        assert p.recur == "seg" and not p.proportional, lengths
        inputs = make_inputs(lengths, heads, device)
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        out, grads, saved = _run_fwd_bwd(inputs)
        ref_out, ref_grads = _clone_all(out, grads)
        del out, grads
        tape_default = KdaWorkspace.for_problem(S, N, heads, device, save_mode="tape")
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=tape_default
        )
        del tape_default
        real = _real_chunks(lengths)
        rows = slice(0, BT * real)
        ranks = _kernel_ranks(lengths)
        groupings = list(groupings_of.get(lengths, ()))
        groupings += [kw for _label, kw in _partitions(lengths) if kw not in groupings]
        monkeypatch.setattr(cake_backend, "_alloc", poisoned_alloc)
        for kw in groupings:
            for t in passes:
                arena = cake_backend.KdaArena(lean=t > 1, tail_passes=t, **kw)
                for save_mode, reference in (("light", saved), ("tape", saved_t)):
                    path = "tape" if save_mode == "tape" else p.light_path
                    case = (lengths, heads, kw, t, save_mode)
                    ws = KdaWorkspace.for_problem(
                        S,
                        N,
                        heads,
                        device,
                        save_mode=save_mode,
                        arena=arena,
                        doc_lengths=lengths,
                    )
                    packing = ws._packing(lengths)
                    for i, b in enumerate(packing.batches):
                        blocks = _expected_group_blocks(ranks, lengths, b, p.P)
                        one_run = _one_run(blocks)
                        assert (packing.blocks[i] is not None) == one_run, (case, b)
                        key = (path, "run" if one_run else "index")
                        hits[key] = hits.get(key, 0) + 1
                        # the group's tail in its own passes, with the host's counts where it takes several
                        group_passes = ws.batch_view(b, tape=True).tail_passes
                        assert group_passes == _fewest_tail_passes(
                            ws.batch_view(b, tape=True).plan, arena, ws.bytes, save_mode
                        ), (case, b)
                        counted = packing.tail_counts(i, b.NC_pad_tape) is not None
                        assert counted == (group_passes > 1), (case, b)
                        if counted:
                            hits["tail"] = hits.get("tail", 0) + 1
                        elif t > 1:
                            hits["one_pass"] = hits.get("one_pass", 0) + 1
                    count = [0]

                    def same(a, b_, what, _case=case, _count=count):
                        assert _bits_equal(a, b_), (_case, what)
                        _count[0] += 1

                    ws.arena.fill_(0xFF)  # NaN in every BF16 / FP32 view
                    o, saved_b = cake_backend.chunk_kda_fwd(
                        q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
                    )
                    same(o, ref_out, "out")
                    same(saved_b.Aqk, reference.Aqk, "Aqk")
                    same(saved_b.Akk, reference.Akk, "Akk")
                    for j, (a, b_) in enumerate(
                        zip(saved_b.tables, reference.tables, strict=True)
                    ):
                        same(a, b_, f"tables[{j}]")
                    want = 3 + len(reference.tables)
                    if save_mode == "light":
                        assert saved_b.version == cake_backend.SAVED_VERSION
                        for name in ("hseg", "seg_state"):
                            assert (getattr(saved_b, name) is None) == (not p.seg_save)
                            if p.seg_save:
                                a, b_ = getattr(saved_b, name), getattr(reference, name)
                                same(a, b_, name)
                                want += 1
                        for name in ("qg", "w", "u"):  # the real token rows
                            a, b_ = getattr(saved_b, name), getattr(reference, name)
                            assert (a is None) == (b_ is None), (case, name)
                            if b_ is not None:
                                same(a[rows], b_[rows], name)
                                want += 1
                    else:
                        assert saved_b.version == cake_backend.SAVED_VERSION_TAPE
                        same(saved_b.h[:real], reference.h[:real], "h")
                        same(saved_b.v_new[rows], reference.v_new[rows], "v_new")
                        same(saved_b.seg_m, reference.seg_m, "seg_m")
                        want += 3
                    for source in (saved_b, reference):
                        ws.arena.fill_(0xFF)
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
                        for name, a in zip(
                            ("dq", "dk", "dv", "dbeta", "dg"), got, strict=True
                        ):
                            same(a, ref_grads[name], name)
                        want += 5
                    assert count[0] == want, case
                    total += count[0]
                    del ws, o, saved_b, got
        monkeypatch.setattr(cake_backend, "_alloc", original_alloc)
    for key in (
        ("light_states", "run"),
        ("light_states", "index"),
        ("tape", "run"),
        ("tape", "index"),
        "tail",
        "one_pass",
    ):
        assert hits.get(key, 0) > 0, (key, hits)
    assert total > 0


def test_a_warmed_batched_workspace_captures_its_calls(device, manifest) -> None:
    """A caller-owned batched workspace that ran its ``doc_lengths`` once holds their device copy (the lengths, the
    tail counts of its group in several passes and the block indices of its group that is not one run), so a
    forward and a backward captured in one CUDA graph make no copy, and the replay gives the eager results bit for
    bit; a workspace without the copy still refuses the capture."""
    lengths, heads = (640, 64, 320), 2
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    q, k, v, g, beta, cu, dout = (
        inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    )
    arena = cake_backend.KdaArena(lean=True, tail_passes=3, batch_tokens=704)
    ws = KdaWorkspace.for_problem(
        S, N, heads, device, save_mode="tape", arena=arena, doc_lengths=lengths
    )
    assert ws.plan.recur == "seg"
    o, saved = cake_backend.chunk_kda_fwd(
        q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
    )
    got = cake_backend.chunk_kda_bwd(
        q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, doc_lengths=lengths
    )
    reference = [o.clone(), saved.seg_m.clone(), *(x.clone() for x in got)]
    del o, saved, got
    packing = ws._doc_cache[lengths]
    assert packing.blocks[0] is None and packing.blocks[1] is not None
    assert packing.words_dev is not None and packing.index_dev is not None
    counted = [
        packing.tail_counts(i, b.NC_pad_tape) is not None
        for i, b in enumerate(packing.batches)
    ]
    assert counted == [
        ws.batch_view(b, tape=True).tail_passes > 1 for b in packing.batches
    ]
    assert any(counted)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        o, saved = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=lengths
        )
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, doc_lengths=lengths
        )
    after = ws._doc_cache[lengths]
    assert after.words_dev is packing.words_dev and after.index_dev is packing.index_dev
    graph.replay()
    torch.cuda.synchronize()
    results = [o, saved.seg_m, *got]
    assert len(results) == len(reference) == 7
    for i, (a, b) in enumerate(zip(results, reference, strict=True)):
        assert _bits_equal(a, b), i
    fresh = KdaWorkspace.for_problem(
        S, N, heads, device, save_mode="tape", arena=arena, doc_lengths=lengths
    )
    refused = pytest.raises(RuntimeError, match="under CUDA-graph capture needs")
    with refused, torch.cuda.graph(torch.cuda.CUDAGraph()):
        cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=fresh, doc_lengths=lengths
        )


def test_a_batched_workspace_keeps_the_device_copy_its_graph_reads(
    device, manifest
) -> None:
    """A captured graph reads the device copy of its ``doc_lengths`` (the lengths, the tail counts of the groups in
    several passes and the block indices of its group that is not one run) at each replay, so the workspace keeps
    that packing for its lifetime: nine other packs run eagerly after the capture (the cache's eight and one more)
    evict the oldest of them, never the captured one, and the replay still gives the eager results bit for bit."""
    lengths, heads = (640, 64, 320), 2
    others = [
        (640, 320, 64),
        (320, 640, 64),
        (64, 640, 320),
        (320, 64, 640),
        (64, 320, 640),
        (600, 360, 64),
        (600, 64, 360),
        (360, 600, 64),
        (64, 600, 360),
    ]
    S, N = sum(lengths), len(lengths)
    assert all(sum(o) == S and len(o) == N for o in others)
    names = ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
    arena = cake_backend.KdaArena(lean=True, tail_passes=3, batch_tokens=704)
    # sized for the groups of the captured pack, whose first group then runs two tail passes; the other packs' groups
    # (at most two documents of at most 704 tokens) fit that arena too
    ws = KdaWorkspace.for_problem(
        S, N, heads, device, save_mode="tape", arena=arena, doc_lengths=lengths
    )
    assert ws.plan.recur == "seg"

    def fwd_bwd(pack, inputs):
        q, k, v, g, beta, cu, dout = (inputs[n] for n in names)
        o, saved = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, doc_lengths=pack
        )
        got = cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws, doc_lengths=pack
        )
        return [o, saved.seg_m, *got]

    inputs = make_inputs(lengths, heads, device)
    reference = [x.clone() for x in fwd_bwd(lengths, inputs)]
    packing = ws._doc_cache[lengths]
    assert packing.blocks[0] is None and packing.index_dev is not None
    counted = [
        packing.tail_counts(i, b.NC_pad_tape) is not None
        for i, b in enumerate(packing.batches)
    ]
    assert counted == [
        ws.batch_view(b, tape=True).tail_passes > 1 for b in packing.batches
    ]
    assert any(counted)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = fwd_bwd(lengths, inputs)
    assert ws._doc_graphs == {lengths}
    for pack in others:
        fwd_bwd(pack, make_inputs(pack, heads, device))
    assert list(ws._doc_cache) == [lengths, *others[1:]]
    kept = ws._doc_cache[lengths]
    assert kept.words_dev is packing.words_dev and kept.index_dev is packing.index_dev
    graph.replay()
    torch.cuda.synchronize()
    assert len(results) == len(reference) == 7
    for i, (a, b) in enumerate(zip(results, reference, strict=True)):
        assert _bits_equal(a, b), i
