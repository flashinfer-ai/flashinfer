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
import json
import re
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
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent

ARCHES = tuple(cake_jit.ARCHES)
BT, D, SCALE = cake_backend.BT, cake_backend.D, cake_backend.SCALE
KdaChunkSaved = cake_backend.KdaChunkSaved
KdaWorkspace = cake_backend.KdaWorkspace

# Accuracy gate, declared before the first run: every output elementwise within
# ``|cand - ref| <= ATOL + RTOL * |ref|`` of the exact FP64 per-document recurrence of the same BF16 / FP32 inputs
# (the BF16 outputs and the FP32 ``dg`` alike, compared in FP32).
ATOL = RTOL = 1e-2
OUTPUT_NAMES = ("out",) + GRAD_NAMES

# Dispatch constants the registry rules must reproduce (the kernels' validated regimes).
FWD_BULK_MIN_CTAS = 256
G8_MIN_CTAS = 256
G8S_MIN_CTAS = 128
FWDH_V2_CTAS_PER_SM = 4
DHU_V2_CTAS_PER_SM = 1
RECUR_SLICES = {"v1": 2, "v2": 4}
# The segmented recurrences: at most SEG_MAX_H heads, clamp(ceil(SEG_SLOT_FACTOR * sm / (N * H)), 1, SEG_P_MAX)
# segments per document.
SEG_MAX_H = 8
SEG_SLOT_FACTOR = 4
SEG_P_MAX = 64
OPTIONAL_STAGES = ("seg_table", "seg_carry")

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
    if rule["kind"] == "all_of":
        for sub_rule in rule["rules"]:
            kinds.extend(_rule_kinds(sub_rule))
    return kinds


def test_registry_records_are_well_formed() -> None:
    assert len(cake_jit.MODULES) == 21
    assert tuple(cake_jit.OPTIONAL_STAGES) == OPTIONAL_STAGES
    assert cake_jit.SEGMENTS == {
        "max_heads": SEG_MAX_H,
        "slot_factor": SEG_SLOT_FACTOR,
        "max_segments": SEG_P_MAX,
    }
    stages = set()
    tensors, scalars = set(cake_jit.LOGICAL_TENSORS), set(cake_jit.LOGICAL_SCALARS)
    for name, record in cake_jit.MODULES.items():
        assert record["stage"] in cake_jit.STAGES, name
        stages.add(record["stage"])
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
        assert launch["cooperative"] is False and launch["use_pdl"] is False, name
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
            assert kind in (
                "always",
                "otherwise",
                "ctas_at_least",
                "ctas_below",
                "ctas_between",
                "slots_at_most",
                "heads_at_most",
                "heads_above",
                "all_of",
            ), name
        assert record["grid_rule"]["kind"] in (
            "fixed",
            "chunks",
            "pairs",
            "per_doc_head",
            "sm_bounded",
            "docs_heads",
            "segments_heads",
        ), name
        if record["select_rule"]["kind"] == "heads_at_most":
            assert record["select_rule"]["max"] == SEG_MAX_H, name
    assert stages == set(cake_jit.STAGES)
    for stage in cake_jit.STAGES:
        names = cake_jit.records_of_stage(stage)
        kinds = [cake_jit.MODULES[n]["select_rule"]["kind"] for n in names]
        if len(names) == 1 and stage in OPTIONAL_STAGES:
            assert kinds == ["heads_at_most"], stage
        elif len(names) == 1:
            assert kinds == ["always"], stage
        else:
            assert "always" not in kinds and kinds.count("otherwise") <= 1, stage
    # The segmented records launch twice per stage (the carry between the passes); every other record once.
    assert cake_jit.launches("fwdh_v3") == cake_jit.launches("dhu_v3") == 2
    assert all(
        cake_jit.launches(name) == 1
        for name in cake_jit.MODULES
        if name not in ("fwdh_v3", "dhu_v3")
    )


def _expected_segments(n: int, h: int, sm: int) -> tuple[str, int, int]:
    if h <= SEG_MAX_H:
        p = min(SEG_P_MAX, max(1, -(-SEG_SLOT_FACTOR * sm // (n * h))))
        return "seg", p, n * p
    return "serial", 1, 0


def _expected_records(nc_pad: int, n: int, h: int, sm: int) -> dict[str, str | None]:
    ctas = nc_pad * h
    seg = h <= SEG_MAX_H
    return {
        "table": "table",
        "seg_table": "seg_table" if seg else None,
        "fwd_prep": "fwd_prep_bulk" if ctas >= FWD_BULK_MIN_CTAS else "fwd_prep_split",
        "fwd_intra": "fwd_intra",
        "wy": "wy",
        "fwdh": "fwdh_v3"
        if seg
        else (
            "fwdh_v2"
            if n * h * RECUR_SLICES["v2"] <= sm * FWDH_V2_CTAS_PER_SM
            else "fwdh_v1"
        ),
        "seg_carry": "seg_carry" if seg else None,
        "fwd_o": "fwd_o",
        "bwd_prep": (
            "bwd_prep_g8"
            if ctas >= G8_MIN_CTAS
            else ("bwd_prep_g8s" if ctas >= G8S_MIN_CTAS else "bwd_prep_v2")
        ),
        "dav": "dav",
        "dhu": "dhu_v3"
        if seg
        else (
            "dhu_v2"
            if n * h * RECUR_SLICES["v2"] <= sm * DHU_V2_CTAS_PER_SM
            else "dhu_v1"
        ),
        "dqkg": "dqkg",
        "glm_intra": "glm_intra",
        "gate_epilogue": "gate_epilogue",
    }


def _expected_grid(
    name: str, nc_pad: int, n: int, h: int, sm: int, nseg: int = 0, launch: int = 0
) -> tuple[int, int, int]:
    ctas = nc_pad * h
    if name in ("table", "seg_table"):
        return (1, 1, 1)
    if name in ("fwd_prep_split", "bwd_prep_g8s"):
        return (2 * nc_pad, h, 1)
    if name in (
        "fwd_prep_bulk",
        "fwd_intra",
        "bwd_prep_v2",
        "bwd_prep_g8",
        "glm_intra",
        "gate_epilogue",
    ):
        return (nc_pad, h, 1)
    if name in ("wy", "dav"):
        return (nc_pad // 2, h, 1)
    if name == "fwdh_v3":
        return ((2 if launch == 0 else 1) * nseg * h, 1, 1)
    if name == "dhu_v3":
        return (nseg * h, 1, 1)
    if name == "seg_carry":
        return (n * h, 1, 1)
    if name.startswith(("fwdh_", "dhu_")):
        return (n * h * RECUR_SLICES[name[-2:]], 1, 1)
    if name in ("fwd_o", "dqkg"):
        return (max(1, min(ctas, sm)), 1, 1)
    raise AssertionError(name)


PLAN_GRID = [
    (nc_pad, n, h, sm)
    for nc_pad in (2, 4, 6, 32, 64, 128, 256, 4096)
    for n in (1, 3, 4, 7)
    for h in (1, 2, 4, 8, 16, 32, 64)
    for sm in (132, 148, 160)
]


def test_every_stage_selects_the_expected_record_and_grid() -> None:
    for nc_pad, n, h, sm in PLAN_GRID:
        expected = _expected_records(nc_pad, n, h, sm)
        route, segments, nseg = _expected_segments(n, h, sm)
        assert cake_jit.segment_plan(n, h, sm) == (route, segments, nseg), (n, h, sm)
        for stage in cake_jit.STAGES:
            name = cake_jit.select_record(
                stage, num_chunks_pad=nc_pad, num_seqs=n, num_heads=h, sm_count=sm
            )
            assert name == expected[stage], (stage, nc_pad, n, h, sm)
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
                assert grid == _expected_grid(name, nc_pad, n, h, sm, nseg, launch), (
                    name,
                    nc_pad,
                    n,
                    h,
                    sm,
                    launch,
                )


def test_segmented_grids_need_segments_and_a_known_launch() -> None:
    with pytest.raises(ValueError, match="at least one segment"):
        cake_jit.launch_grid(
            "fwdh_v3", num_chunks_pad=4, num_seqs=1, num_heads=4, sm_count=148
        )
    with pytest.raises(ValueError, match="launch 2 of 2"):
        cake_jit.launch_grid(
            "fwdh_v3",
            num_chunks_pad=4,
            num_seqs=1,
            num_heads=4,
            sm_count=148,
            num_segments=64,
            launch=2,
        )
    with pytest.raises(ValueError, match="must all be >= 1"):
        cake_jit.segment_plan(0, 4, 148)


def test_pairs_grid_needs_an_even_chunk_bound() -> None:
    with pytest.raises(ValueError, match="not even"):
        cake_jit.launch_grid(
            "wy", num_chunks_pad=3, num_seqs=1, num_heads=1, sm_count=148
        )


def test_plan_follows_the_chunk_bound_formulas() -> None:
    for lengths, heads in SHAPES:
        S, N = sum(lengths), len(lengths)
        for sm in (132, 148):
            p = cake_backend.plan(S, N, heads, sm)
            nc_max = -(-S // BT) + N
            assert (p.NC_max, p.NC_pad, p.R) == (
                nc_max,
                nc_max + nc_max % 2,
                64 * (nc_max + nc_max % 2),
            )
            assert p.records == _expected_records(p.NC_pad, N, heads, sm)
            assert (p.recur, p.P, p.NSEG) == _expected_segments(N, heads, sm)
            for stage in cake_jit.STAGES:
                if not p.dispatched(stage):
                    assert stage in OPTIONAL_STAGES and p.recur == "serial"
                    with pytest.raises(ValueError, match="not launched"):
                        p.record(stage)
                    continue
                for launch in range(p.launches(stage)):
                    assert p.grid(stage, launch) == _expected_grid(
                        p.record(stage), p.NC_pad, N, heads, sm, p.NSEG, launch
                    )
            seg = p.recur == "seg"
            assert p.launch_count("fwd") == (9 if seg else 6)
            assert p.launch_count("bwd") == (12 if seg else 8)
            assert p.launch_count("bwd", tables=False) == (14 if seg else 9)
            assert p.launch_count("bwd", tape=True) == (9 if seg else 7)
            assert p.launch_count("replay") == (15 if seg else 10)
            assert set(p.stages()) == {s for s, r in p.records.items() if r}
            assert set(p.stages(backward=False)) == {
                s
                for s in p.stages()
                if s not in cake_jit.STAGES_BWD or s in ("wy", "fwdh")
            }
    assert cake_backend.plan(1, 1, 1, 1).NC_pad == 2
    for bad in ((0, 1, 1, 148), (1, 0, 1, 148), (1, 1, 0, 148), (1, 1, 1, 0)):
        with pytest.raises(ValueError):
            cake_backend.plan(*bad)
    with pytest.raises(ValueError, match="int32"):
        cake_backend.plan(64, 1, 2**31 // (D * BT), 148)


def test_byte_queries_follow_the_arena_layout() -> None:
    GiB = 2**30
    for lengths, heads in SHAPES + [((239718 // 3,) * 3, 4), ((261393 // 4,) * 4, 64)]:
        S, N = sum(lengths), len(lengths)
        p = cake_backend.plan(S, N, heads, 148)
        for mode in cake_backend.MODES:
            regions = p.layout(mode)
            assert all(r.offset % cake_backend.ALIGN == 0 for r in regions), mode
            assert p.workspace_bytes(mode) == max(r.offset + r.nbytes for r in regions)
            names = {r.name for r in regions}
            seg_names = {"seg_b", "seg_m", "hseg"} | (
                {"dseg", "dhseg"}
                if mode.startswith("bwd") or mode == "replay"
                else set()
            )
            assert (seg_names <= names) if p.recur == "seg" else not (seg_names & names)
        for backward in (False, True):
            for save_mode in cake_backend.SAVE_MODES:
                tape = save_mode == "tape"
                mode = (
                    ("bwd_tape" if tape else "bwd")
                    if backward
                    else ("fwd_tape" if tape else "fwd")
                )
                assert cake_backend.workspace_bytes(
                    S, N, heads, backward=backward, save_mode=save_mode, sm_count=148
                ) == p.workspace_bytes(mode)
        assert cake_backend.workspace_bytes(
            S, N, heads, backward=False, sm_count=148
        ) <= cake_backend.workspace_bytes(S, N, heads, sm_count=148)
        assert (
            cake_backend.saved_bytes(S, heads)
            == 2 * S * heads * BT * 2
            == 256 * S * heads
        )
        tables = cake_backend.tables_bytes(S, N, heads, sm_count=148)
        assert tables == p.table_words() * 4 and tables % cake_backend.ALIGN == 0
        assert (
            cake_backend.saved_bytes(S, heads, N, sm_count=148)
            == 256 * S * heads + tables
        )
        tape_bytes = cake_backend.saved_bytes(
            S, heads, N, save_mode="tape", sm_count=148
        )
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
        report = cake_backend.memory_report(S, N, heads, sm_count=148)
        assert (report["recur"], report["segments"], report["num_segments"]) == (
            p.recur,
            p.P,
            p.NSEG,
        )
        assert (
            report["arena_bwd"] == p.workspace_bytes("bwd")
            and report["saved"] == 256 * S * heads + tables
        )
        assert report["bwd_total"] == report["arena_bwd"] + report["out_bwd"]
        if (
            S > 200_000
        ):  # the largest contract rows fit one 283 GiB device with room for two arenas (parity)
            assert report["arena_bwd"] < 85 * GiB
    with pytest.raises(ValueError, match="needs num_docs"):
        cake_backend.saved_bytes(64, 1, save_mode="tape")
    with pytest.raises(ValueError, match="save_mode"):
        cake_backend.workspace_bytes(64, 1, 1, save_mode="full", sm_count=148)
    with pytest.raises(ValueError, match="layout mode"):
        cake_backend._layout(128, 1, 2, 0, "sideways")


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


def test_manifest_is_well_formed(manifest: dict) -> None:
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == "flashinfer.experimental.cake_kda_train"
    arch_flags = manifest["arch_flags"]
    assert set(ARCHES) <= set(arch_flags)
    libraries = manifest["libraries"]
    names = [entry["name"] for entry in libraries]
    assert len(names) == len(set(names)) == 21 * len(ARCHES)
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
    }
    assert set(names) == expected


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
    assert (
        ws.bytes
        == cake_backend.workspace_bytes(S, N, heads, sm_count=sm)
        == ws.arena.numel()
    )
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
    assert ws.records() == {stage: ws.plan.record(stage) for stage in ws.plan.stages()}
    assert fwd_only.records() == {
        stage: ws.plan.record(stage) for stage in ws.plan.stages(backward=False)
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
    for lengths, heads in (((63, 1, 65), 2), ((129,), 16)):
        S, N = sum(lengths), len(lengths)
        inputs = make_inputs(lengths, heads, device)
        launched: list[str] = []

        def recording(name, values, *, grid, device_index):
            launched.append(name)
            cake_jit.run_module(name, values, grid=grid, device_index=device_index)

        ws = KdaWorkspace.for_problem(S, N, heads, device, launcher=recording)
        p = ws.plan
        assert p.recur == ("seg" if heads <= SEG_MAX_H else "serial")
        q, k, v, g, beta, cu, dout = (
            inputs[n] for n in ("q", "k", "v", "g", "beta", "cu_seqlens", "dout")
        )
        _, saved = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, workspace=ws)
        fwd = list(launched)
        recurrence = (
            ["fwdh_v3", "seg_carry", "fwdh_v3"]
            if p.recur == "seg"
            else [p.record("fwdh")]
        )
        adjoint = (
            ["dhu_v3", "seg_carry", "dhu_v3"] if p.recur == "seg" else [p.record("dhu")]
        )
        tables = ["table", "seg_table"] if p.recur == "seg" else ["table"]
        assert fwd == [
            *tables,
            p.record("fwd_prep"),
            "fwd_intra",
            "wy",
            *recurrence,
            "fwd_o",
        ]
        assert len(fwd) == p.launch_count("fwd")
        launched.clear()
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved, workspace=ws
        )
        tail = ["dav", *adjoint, "dqkg", "glm_intra", "gate_epilogue"]
        assert launched == [p.record("bwd_prep"), "wy", *recurrence, *tail]
        assert len(launched) == p.launch_count("bwd")
        launched.clear()
        cake_backend.chunk_kda_bwd(q, k, v, g, beta, cu, dout, saved=None, workspace=ws)
        assert launched == [
            *tables,
            p.record("bwd_prep"),
            "fwd_intra",
            "wy",
            *recurrence,
            *tail,
        ]
        assert len(launched) == p.launch_count("replay")
        launched.clear()
        _, saved_t = cake_backend.chunk_kda_fwd(
            q, k, v, g, beta, cu, workspace=ws, save_mode="tape"
        )
        launched.clear()
        tape_ws = KdaWorkspace.for_problem(
            S, N, heads, device, save_mode="tape", launcher=recording
        )
        cake_backend.chunk_kda_bwd(
            q, k, v, g, beta, cu, dout, saved=saved_t, workspace=tape_ws
        )
        assert launched == [p.record("bwd_prep"), "wy", *tail]
        assert len(launched) == p.launch_count("bwd", tape=True)


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
    with pytest.raises(ValueError, match="version"):
        cake_backend.chunk_kda_bwd(
            q,
            k,
            v,
            g,
            beta,
            cu,
            dout,
            saved=KdaChunkSaved(3, saved.Aqk, saved.Akk, None, 65, heads, 1, SCALE),
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
                1, saved.Aqk, saved.Akk, saved.tables[:2], 65, heads, 1, SCALE
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
            saved=KdaChunkSaved(1, saved.Aqk, saved.Akk, None, 64, heads, 1, SCALE),
        )
    out, _ = cake_backend.chunk_kda_fwd(q, k, v, g, beta, cu, save_for_backward=False)
    assert out.shape == (65, heads, D)
    assert cake_backend.is_supported(device)


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
