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

Host-side checks of the Cake NVFP4-transport tensor-parallel all-reduce (no GPU).

The program registry (``flashinfer.jit.cake_fp4_allreduce_programs``) against the
kernel contract: the program set, the kernel parameter orders, the launch
geometry and grid bound, the per-target build forms and the source files; the
receive-window plan re-derived from the kernels' window formulas at five capacities; the
phase-B selection rule; the program selection of the loader; the build-target
selection with ``FLASHINFER_CUDA_ARCH_LIST`` pinned; and unit checks of the
float32 reference in ``tests/comm/cake_fp4_allreduce_reference.py``.
"""

from __future__ import annotations

import math
from fractions import Fraction
from pathlib import Path

import pytest
import torch

from flashinfer.jit import cake_fp4_allreduce as loader
from flashinfer.jit.cake_fp4_allreduce_programs import (
    LIMITS,
    PROGRAMS,
    SELECT_BV,
    SELECT_PREFETCH,
    WINDOWS,
)
from tests.comm import cake_fp4_allreduce_reference as ref

TARGETS = ("sm_100a", "sm_103a", "sm_107a")
INPUT_FORMATS = ("bf16", "nvfp4")
OUTPUT_FORMATS = ("bf16", "nvfp4", "mxfp8")
BVS = (0, 1, 2)
# The two-shot (input format, bv) variants each target builds with the register-store re-arm
# ("_rst"), from its own measurement; the others use the bulk re-arm (sm_103a throughout).
REGISTER_STORE_FORMS = {
    "sm_100a": {("bf16", 0), ("bf16", 1), ("nvfp4", 1), ("nvfp4", 2)},
    "sm_103a": set(),
    "sm_107a": {("bf16", 0), ("bf16", 1), ("nvfp4", 0), ("nvfp4", 2)},
}
# The two-shot variants each target builds with the early first load ("_fe": the first phase-A load is
# issued ahead of the call-state barrier), from its own measurement: the BF16 variants of sm_103a.
FIRST_LOAD_EARLY_FORMS = {
    "sm_100a": set(),
    "sm_103a": {("bf16", 0), ("bf16", 1), ("bf16", 2)},
    "sm_107a": set(),
}
# The two-shot variants each target builds with the shared-memory pass-2 source ("_p2m": the owner keeps
# its pass-2 packets in shared memory instead of re-reading them from L2), from its own measurement.
PASS2_SMEM_FORMS = {
    "sm_100a": {("bf16", 0), ("bf16", 2), ("nvfp4", 0), ("nvfp4", 2)},
    "sm_103a": {("bf16", 0)},
    "sm_107a": {("nvfp4", 0), ("nvfp4", 2)},
}
# Two forms every target measured faster, so their tokens are uniform: every two-shot and fused-norm
# program keeps its call state per CTA ("_csp": one four-word slot per CTA of the control window, no exit
# quorum), and every two-shot program polls the bv-1 team half's peer row as one 12-tuple ("_b1r"; the
# token is spelled on every bv).
CALL_STATE_TOKEN = "_csp"
B1_POLL_TOKEN = "_b1r"
# The targets whose NVFP4-input fused-norm programs probe the peer row's six pairs at once ("_pa"), from
# their own measurement; the others (sm_107a, and every BF16-input program) poll in two-pair groups.
PROBE_ALL_TARGETS = ("sm_100a", "sm_103a")
# The one-shot BF16 kernel's form per target, from each target's own measurement: sm_103a reads the peer's
# row scale beside the data probe ("_rso") and triggers the dependent launch right after the grid
# dependency wait ("_el"), sm_107a re-arms with register stores ("_rst"), sm_100a ships the baseline's
# forms; the NVFP4 one-shot kernel ships one text for all three (its baseline triggers early), and sm_100a
# adds its bulk re-arm band (ONESHOT_BANDS below).
ONESHOT_BF16_FORMS = {"sm_100a": "", "sm_103a": "_rso_el", "sm_107a": "_rst"}
# The (input format, output format) fused-norm programs with NVFP4 / MXFP8 output each target builds with the
# packed quantizing pass 2 ("_qp"), from its own measurement: the BF16-input builds of sm_100a / sm_103a, the
# NVFP4-input builds of sm_103a and sm_100a's nvfp4 -> mxfp8; sm_107a and every other build use the FP32 tree.
PACKED_QUANT_FORMS = {
    "sm_100a": {("bf16", "nvfp4"), ("bf16", "mxfp8"), ("nvfp4", "mxfp8")},
    "sm_103a": {
        ("bf16", "nvfp4"),
        ("bf16", "mxfp8"),
        ("nvfp4", "nvfp4"),
        ("nvfp4", "mxfp8"),
    },
    "sm_107a": set(),
}
# The (input format, output format) fused-norm programs with NVFP4 / MXFP8 output each target builds with the
# scale store's swizzled address formed in 32-bit arithmetic once per row ("_sfu"), from its own measurement;
# every other build forms the whole 64-bit address per store.  Both forms store the same bytes to the same
# addresses, so the reference does not read the form.  A cell's band inherits the form.
SF_ADDR_U32_FORMS = {
    "sm_100a": {("nvfp4", "nvfp4")},
    "sm_103a": {("bf16", "nvfp4"), ("nvfp4", "mxfp8"), ("nvfp4", "nvfp4")},
    "sm_107a": {("bf16", "mxfp8"), ("nvfp4", "nvfp4")},
}
# The BF16-output fused-norm programs run the direct pass 2 ("_p2d") with the unroll each target measured
# ("_u<N>" for an unroll other than 1); the quantized outputs run the staged pass 2 (unroll 1, no token).
PASS2_DIRECT_UNROLL = {
    ("sm_100a", "bf16", "bf16"): 4,
    ("sm_103a", "bf16", "bf16"): 4,
    ("sm_107a", "bf16", "bf16"): 4,
    ("sm_100a", "nvfp4", "bf16"): 2,
    ("sm_103a", "nvfp4", "bf16"): 2,
    ("sm_107a", "nvfp4", "bf16"): 24,
}
# The fused-norm programs that trigger the dependent launch right after the grid dependency wait ("_el"),
# from each target's own measurement: sm_100a's bf16 -> nvfp4 build.
EARLY_LAUNCH_FORMS = {
    "sm_100a": {("bf16", "nvfp4")},
    "sm_103a": set(),
    "sm_107a": set(),
}
# The (input format, output format) fused-norm programs each target builds with the permuted pass-1 sum of
# squares ("_sop": the lane's loaded chunk order), from its own measurement: every build of sm_107a, every
# build but nvfp4 -> bf16 of sm_103a and sm_100a; the others accumulate in the row's chunk order
# ("natural").  The reference sums in the order of the selected program's record.
PERMUTED_SS_FORMS = {
    "sm_100a": {(i, o) for i in INPUT_FORMATS for o in OUTPUT_FORMATS}
    - {("nvfp4", "bf16")},
    "sm_103a": {(i, o) for i in INPUT_FORMATS for o in OUTPUT_FORMATS}
    - {("nvfp4", "bf16")},
    "sm_107a": {(i, o) for i in INPUT_FORMATS for o in OUTPUT_FORMATS},
}
# The rows-dispatched programs: beside the whole program of a cell, one program serves a band of row counts
# (a prefix band at or below the threshold, a suffix band at or above it; both inclusive) in the forms the
# band deviates in, from each target's own measurement.  The fused norm's bands switch at 6144 rows: the
# nvfp4 -> bf16 prefix sums the pass-1 squares in the loaded chunk order ("_sop"), the BF16-input suffixes run
# the direct pass 2 on eleven workers ("_p2d", "_w11"; sm_100a's bf16 -> nvfp4 band drops the whole program's
# early trigger).  The NVFP4 one-shot of sm_100a re-arms with bulk stores ("_rsb") at or above 16384 rows.
BAND_THRESHOLD = 6144
FUSED_NORM_BANDS = {
    ("sm_100a", "nvfp4", "bf16"): ("prefix", {"ss_order": "perm"}),
    ("sm_103a", "nvfp4", "bf16"): ("prefix", {"ss_order": "perm"}),
    ("sm_103a", "bf16", "bf16"): ("suffix", {"workers": 11}),
    ("sm_100a", "bf16", "nvfp4"): (
        "suffix",
        {"pass2": "direct", "workers": 11, "early_launch": False},
    ),
    ("sm_103a", "bf16", "nvfp4"): ("suffix", {"pass2": "direct", "workers": 11}),
    ("sm_100a", "bf16", "mxfp8"): ("suffix", {"pass2": "direct", "workers": 11}),
    ("sm_103a", "bf16", "mxfp8"): ("suffix", {"pass2": "direct", "workers": 11}),
}
ONESHOT_BAND_THRESHOLD = 16384
ONESHOT_BANDS = {("sm_100a", "nvfp4"): ("suffix", ONESHOT_BAND_THRESHOLD, "_rsb")}
CAPACITIES = (4, 1024, 4096, 16384, 100000)
SOURCE_DIR = "csrc/cake_fp4_allreduce"
MAX_DYNAMIC_SMEM_BYTES = 232448  # the opt-in limit of SM100 / SM103 / SM107
# Every program keeps four words of call state per CTA in the control window, which the host provisions
# for 4096 CTAs: the grid bound of every program and the control words of every window.
CTRL_WORDS_PER_CTA = 4
GRID_MAX = 4096
# The kernel contract: sentinel words, record geometry and the window formulas.
VAL, SF, RS = 0x88888888, 0xFFFFFFFF, 0x80000000
VALUE_BYTES, SCALE_BYTES, RECORD_BYTES, ROW_SCALE_BYTES = 3072, 384, 3456, 16
ROTATION = 3


def _two_shot_suffix(target: str, input_format: str, bv: int) -> str:
    """The name suffix of the two-shot program ``target`` builds for one (input format, bv): "_csp" for the
    per-CTA call state, "_fe" for the early first load, "_b1r" for the 12-tuple bv-1 poll, "_p2m" for the
    shared-memory pass-2 source, "_rst" for the register-store re-arm, in that order."""
    variant = (input_format, bv)
    return (
        CALL_STATE_TOKEN
        + ("_fe" if variant in FIRST_LOAD_EARLY_FORMS[target] else "")
        + B1_POLL_TOKEN
        + ("_p2m" if variant in PASS2_SMEM_FORMS[target] else "")
        + ("_rst" if variant in REGISTER_STORE_FORMS[target] else "")
    )


def _fused_norm_forms(target: str, input_format: str, output_format: str) -> dict:
    """The forms of the whole fused-norm program ``target`` builds for one (input format, output format), as
    the registry records them: the poll ("all" for the probe-all poll), the pass 2 and its unroll, the worker
    count (None for the kernel's own), the quantizing pass 2 and the scale-address form (None for BF16 output),
    the launch trigger and the pass-1 sum-of-squares order."""
    cell = (input_format, output_format)
    probe_all = input_format == "nvfp4" and target in PROBE_ALL_TARGETS
    unroll = PASS2_DIRECT_UNROLL.get((target, *cell))
    quant = sf_addr = None
    if output_format != "bf16":
        quant = "packed" if cell in PACKED_QUANT_FORMS[target] else "f32"
        sf_addr = "u32" if cell in SF_ADDR_U32_FORMS[target] else "i64"
    return {
        "poll": "all" if probe_all else "grouped",
        "pass2": "direct" if unroll else "staged",
        "unroll": unroll or 1,
        "workers": None,
        "quant": quant,
        "sf_addr": sf_addr,
        "early_launch": cell in EARLY_LAUNCH_FORMS[target],
        "ss_order": "perm" if cell in PERMUTED_SS_FORMS[target] else "natural",
    }


def _fused_norm_name(
    input_format: str, output_format: str, prefetch: bool, forms: dict
) -> str:
    """The name of the fused-norm program with ``forms``: "_pa" for the probe-all poll, "_p2d" for the direct
    pass 2 and "_u<N>" for its unroll other than 1, "_w<N>" for a worker count the build spells, "_qp" for the
    packed quantizing pass 2, "_sfu" for the 32-bit scale-address form, "_el" for the early launch trigger,
    "_csp" for the per-CTA call state, "_sop" for the permuted sum of squares, "_pf" for the residual prefetch,
    in that order."""
    return (
        f"tp2_rmsnorm_{input_format}_{output_format}"
        + ("_pa" if forms["poll"] == "all" else "")
        + ("_p2d" if forms["pass2"] == "direct" else "")
        + (f"_u{forms['unroll']}" if forms["unroll"] != 1 else "")
        + (f"_w{forms['workers']}" if forms["workers"] is not None else "")
        + ("_qp" if forms["quant"] == "packed" else "")
        + ("_sfu" if forms["sf_addr"] == "u32" else "")
        + ("_el" if forms["early_launch"] else "")
        + CALL_STATE_TOKEN
        + ("_sop" if forms["ss_order"] == "perm" else "")
        + ("_pf" if prefetch else "")
    )


def _fused_norm_band_forms(target: str, input_format: str, output_format: str) -> dict:
    """The forms of the band program ``target`` builds for the cell: the whole program's, the band's keys over
    them."""
    _, extra = FUSED_NORM_BANDS[(target, input_format, output_format)]
    return {**_fused_norm_forms(target, input_format, output_format), **extra}


def _fused_norm_band(target: str, input_format: str, output_format: str) -> str:
    """The name of the band program ``target`` builds for the cell (which must have a band); a band
    ships without the prefetch."""
    forms = _fused_norm_band_forms(target, input_format, output_format)
    return _fused_norm_name(input_format, output_format, False, forms)


def _fused_norm_whole_forms(
    target: str, input_format: str, output_format: str, prefetch: bool
) -> dict:
    """The forms of the whole program ``target`` builds for the cell's prefetch form.  The prefetch rows
    (18 to 28 per CTA: at most 4639 rows on 148 or 160 SMs) all fall inside a prefix band and outside a
    suffix band, so the prefetch form ships one program per cell: in the band's forms where the band is a
    prefix, in the forms the target measured everywhere else."""
    band = FUSED_NORM_BANDS.get((target, input_format, output_format))
    if prefetch and band is not None and band[0] == "prefix":
        return _fused_norm_band_forms(target, input_format, output_format)
    return _fused_norm_forms(target, input_format, output_format)


def _fused_norm_program(
    target: str,
    input_format: str,
    output_format: str,
    prefetch: bool,
    rows: int | None = None,
) -> str:
    """The fused-norm program ``target`` runs for the cell: the band program when the cell has a band, the
    call runs without the prefetch and ``rows`` falls inside the band, else the whole program."""
    band = FUSED_NORM_BANDS.get((target, input_format, output_format))
    if rows is not None and band is not None and not prefetch:
        kind, _ = band
        if loader.select_rows(rows, kind, BAND_THRESHOLD):
            return _fused_norm_band(target, input_format, output_format)
    forms = _fused_norm_whole_forms(target, input_format, output_format, prefetch)
    return _fused_norm_name(input_format, output_format, prefetch, forms)


def _one_shot_program(target: str, input_format: str, rows: int | None = None) -> str:
    """The one-shot program ``target`` runs for the input format: the band program when ``rows`` falls inside
    the target's band, else the whole program (the BF16 form the target measured, the NVFP4 single text)."""
    form = ONESHOT_BF16_FORMS[target] if input_format == "bf16" else ""
    band = ONESHOT_BANDS.get((target, input_format))
    if rows is not None and band is not None:
        kind, threshold, token = band
        if loader.select_rows(rows, kind, threshold):
            form += token
    return f"tp2_oneshot_{input_format}{form}"


def _expected_program_ids() -> set[str]:
    ids = {"tp2_oneshot_nvfp4"}
    ids |= {f"tp2_oneshot_bf16{form}" for form in ONESHOT_BF16_FORMS.values()}
    ids |= {
        f"tp2_oneshot_{i}{token}" for (_, i), (_, _, token) in ONESHOT_BANDS.items()
    }
    # One two-shot program per (target, input format, bv) and one fused-norm program per (target, input
    # format, output format, prefetch form), named by the forms that target measured; targets whose forms
    # agree share the name (and the program).  A cell with a band adds the band's program, which ships
    # without the prefetch.
    ids |= {
        f"tp4_twoshot_{i}_bv{bv}{_two_shot_suffix(t, i, bv)}"
        for t in TARGETS
        for i in INPUT_FORMATS
        for bv in BVS
    }
    ids |= {
        _fused_norm_program(t, i, o, pf)
        for t in TARGETS
        for i in INPUT_FORMATS
        for o in OUTPUT_FORMATS
        for pf in (False, True)
    }
    ids |= {_fused_norm_band(t, i, o) for (t, i, o) in FUSED_NORM_BANDS}
    return ids


def _expected_params(
    family: str, world_size: int, input_format: str, output_format
) -> tuple[str, ...]:
    """Kernel parameter order of the contract: pointers in kernel order, then rank, rows, cap[, eps, wbias]."""
    inputs = ("x",) if input_format == "bf16" else ("xq", "gs")
    if family == "allreduce":
        if world_size == 4:
            windows = ("ts_wire", "ts_rs")
        elif input_format == "bf16":
            windows = ("os_wire", "os_rs")
        else:
            windows = ("q_wire",)
        return (*inputs, "out", *windows, "peer_ptrs", "ctrl", "rank", "rows", "cap")
    outputs = ("out",) if output_format == "bf16" else ("outq", "out_sf", "gs_out")
    return (
        *inputs,
        "res_in",
        "w",
        *outputs,
        "res_out",
        "os_wire",
        "os_rs",
        "peer_ptrs",
        "ctrl",
        "rank",
        "rows",
        "cap",
        "eps",
        "wbias",
    )


def _binding_form(record: dict) -> str:
    """The shared binding form of a record: family, world size and formats."""
    if record["family"] == "allreduce":
        shot = "oneshot" if record["world_size"] == 2 else "twoshot"
        return f"tp{record['world_size']}_{shot}_{record['input_format']}"
    return f"tp2_rmsnorm_{record['input_format']}_{record['output_format']}"


# ---------------------------------------------------------------------------
# registry schema
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_registry_lists_every_program_with_the_contract_parameters() -> None:
    assert set(PROGRAMS) == _expected_program_ids()
    assert len(PROGRAMS) == 53
    keys = {
        "family",
        "world_size",
        "input_format",
        "output_format",
        "bv",
        "reset",
        "ss_order",
        "prefetch",
        "targets",
        "kernel_symbol",
        "sources",
        "block",
        "dynamic_smem_bytes",
        "params",
        "entry",
        "grid_max",
    }
    symbols = set()
    for program, record in PROGRAMS.items():
        assert keys <= set(record), (program, keys - set(record))
        family = record["family"]
        world_size = int(record["world_size"])
        input_format = record["input_format"]
        output_format = record["output_format"]
        assert family in ("allreduce", "rmsnorm")
        assert input_format in INPUT_FORMATS
        if family == "allreduce":
            assert output_format is None and world_size in (2, 4)
        else:
            assert output_format in OUTPUT_FORMATS and world_size == 2
        assert record["params"] == _expected_params(
            family, world_size, input_format, output_format
        )
        assert record["kernel_symbol"] == f"cake_fp4_allreduce_{program}"
        assert record["kernel_symbol"] not in symbols
        symbols.add(record["kernel_symbol"])
        # A prefetch program shares its base program's kernel source, and every
        # program shares the binding of its form; the kernel symbol, the block
        # size, the dynamic shared memory and the prefetch come through the
        # loader's -D flags.
        stem = program[: -len("_pf")] if record["prefetch"] else program
        assert tuple(record["sources"]) == (
            f"{SOURCE_DIR}/cake_fp4_allreduce_{stem}_kernel.cu",
            f"{SOURCE_DIR}/cake_fp4_allreduce_{_binding_form(record)}_binding.cu",
        )
        assert record["entry"] == "run"
        # One warp per worker and one more: ten workers for a BF16 input and eight for an NVFP4 input,
        # unless the build spells its own worker count ("_w11").
        workers = 11 if "_w11" in program else None
        workers = workers or (
            8 if (family == "rmsnorm" and input_format == "nvfp4") else 10
        )
        assert record["block"] == 32 * (workers + 1), program
        assert 0 < int(record["dynamic_smem_bytes"]) <= MAX_DYNAMIC_SMEM_BYTES, program
        assert tuple(record["targets"]) and set(record["targets"]) <= set(TARGETS)
        if world_size == 4:
            # The two-shot kernel's phase-B variant ("_bv<N>") and its re-arm ("_rst" for the
            # register-store re-arm, the bulk re-arm otherwise); the other build forms are name
            # tokens, pinned per target by the selection tests.
            assert record["bv"] in BVS
            assert record["reset"] in ("bulk", "st")
            assert program.startswith(f"tp4_twoshot_{input_format}_bv{record['bv']}")
            assert program.endswith("_rst") == (record["reset"] == "st")
        elif family == "allreduce":
            # The one-shot kernel's re-arm form off the input format's baseline (the BF16 kernel
            # re-arms with bulk stores, the NVFP4 kernel with register stores): "_rst" / "_rsb"
            # closes the name; the other lever forms ("_rso", "_el" / "_ll") are name tokens only.
            assert record["bv"] is None
            assert record["reset"] in ("bulk", "st")
            baseline_reset = {"bf16": "bulk", "nvfp4": "st"}[input_format]
            rearm = (
                ""
                if record["reset"] == baseline_reset
                else ("_rst" if record["reset"] == "st" else "_rsb")
            )
            assert program.startswith(f"tp2_oneshot_{input_format}"), program
            assert program.endswith(("_rst", "_rsb")) == bool(rearm), program
            assert program.endswith(rearm), program
        else:
            assert record["bv"] is None and record["reset"] is None
        if family == "rmsnorm":
            assert record["prefetch"] in (False, True)
            assert program.endswith("_pf") == record["prefetch"], program
            stem = program[: -len("_pf")] if record["prefetch"] else program
            # The pass-1 sum-of-squares order: "_sop" for the lane's loaded chunk order, before "_pf".
            assert record["ss_order"] in ("natural", "perm"), program
            assert stem.endswith("_sop") == (record["ss_order"] == "perm"), program
            stem = stem[: -len("_sop")] if record["ss_order"] == "perm" else stem
            # Every fused-norm build keeps its call state per CTA ("_csp", before "_sop"); the
            # other forms ("_pa", "_p2d", "_u<N>", "_w<N>", "_qp", "_sfu", "_el") are name tokens.
            assert stem.endswith(CALL_STATE_TOKEN), program
            assert stem.startswith(f"tp2_rmsnorm_{input_format}_{output_format}")
        else:
            assert record["prefetch"] is None and record["ss_order"] is None
    assert {
        target for record in PROGRAMS.values() for target in record["targets"]
    } == set(TARGETS)


def _checkout_source_dir() -> Path:
    return Path(__file__).resolve().parents[3] / SOURCE_DIR


@pytest.mark.cpu_only
def test_program_sources_exist_and_declare_their_kernels() -> None:
    source_dir = _checkout_source_dir()
    if not source_dir.is_dir():
        pytest.skip(f"{source_dir} is not part of this checkout")
    for program, record in PROGRAMS.items():
        kernel, binding = (
            Path(__file__).resolve().parents[3] / source for source in record["sources"]
        )
        assert kernel.is_file(), kernel
        assert binding.is_file(), binding
        kernel_text = kernel.read_text()
        assert "__global__" in kernel_text, program
        # A kernel whose prefetch twin ships serves both programs through the -D
        # macros: the symbol macro with its own symbol as the default, and the
        # prefetch switch.  Every other kernel keeps its plain symbol.
        stem = program[: -len("_pf")] if record["prefetch"] else program
        shared = f"{stem}_pf" in PROGRAMS
        if shared:
            assert "CAKE_FP4_ALLREDUCE_KERNEL(" in kernel_text, program
            default = f"#define CAKE_FP4_ALLREDUCE_KERNEL cake_fp4_allreduce_{stem}"
            assert default in kernel_text, program
            assert "#ifdef CAKE_FP4_ALLREDUCE_PREFETCH" in kernel_text, program
        else:
            assert f"{record['kernel_symbol']}(" in kernel_text, program
            assert "CAKE_FP4_ALLREDUCE_" not in kernel_text, program
        binding_text = binding.read_text()
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({record['entry']}," in binding_text, (
            program
        )
        # Every program of a (family, world size, input format, output format)
        # form shares one binding; its kernel symbol, block size and dynamic
        # shared memory come through the loader's -D flags, so the binding
        # names no program.
        form = _binding_form(record)
        assert binding.name == f"cake_fp4_allreduce_{form}_binding.cu", program
        for macro in (
            "CAKE_FP4_ALLREDUCE_KERNEL",
            "CAKE_FP4_ALLREDUCE_THREADS",
            "CAKE_FP4_ALLREDUCE_DYNAMIC_SMEM",
        ):
            assert f"#ifndef {macro}" in binding_text, (program, macro)
        assert "cake_fp4_allreduce_tp" not in binding_text, program
        assert loader.program_defines(program) == [
            f"-DCAKE_FP4_ALLREDUCE_KERNEL={record['kernel_symbol']}",
            f"-DCAKE_FP4_ALLREDUCE_THREADS={record['block']}",
            f"-DCAKE_FP4_ALLREDUCE_DYNAMIC_SMEM={record['dynamic_smem_bytes']}",
            *(["-DCAKE_FP4_ALLREDUCE_PREFETCH=1"] if record["prefetch"] else []),
        ], program
    # The loader resolves sources under the package data directory (``flashinfer/data/csrc``, a
    # symlink to the repository's ``csrc`` in an editable install): compare the real files.
    assert {path.resolve() for path in loader.program_sources("tp2_oneshot_bf16")} == {
        (source_dir / "cake_fp4_allreduce_tp2_oneshot_bf16_kernel.cu").resolve(),
        (source_dir / "cake_fp4_allreduce_tp2_oneshot_bf16_binding.cu").resolve(),
    }


@pytest.mark.cpu_only
def test_limits_and_selection_constants_match_the_contract() -> None:
    assert LIMITS == {
        "hidden": 6144,
        "record_bytes": 3456,
        "tp4_cap_max": 103000,
        "tp2_oneshot_grid_max": 4096,
        "tp4_row_multiple": 4,
        "nvfp4_value_bytes": 3072,
        "nvfp4_scales_per_row": 384,
        "mxfp8_value_bytes": 6144,
        "mxfp8_scales_per_row": 192,
        "scale_row_tile": 128,
    }
    assert LIMITS["nvfp4_value_bytes"] == ref.HIDDEN // 2
    assert LIMITS["nvfp4_scales_per_row"] == ref.HIDDEN // 16
    assert LIMITS["mxfp8_value_bytes"] == ref.HIDDEN
    assert LIMITS["mxfp8_scales_per_row"] == ref.HIDDEN // 32
    assert SELECT_BV == {"team_max": 40, "by_quad_min": 64}
    assert SELECT_PREFETCH == {"rows_per_cta_min": 18, "rows_per_cta_max": 28}
    assert loader.HIDDEN == ref.HIDDEN == 6144
    assert loader.RECORD_BYTES == ref.RECORD_BYTES == 3456
    assert LIMITS["hidden"] == ref.HIDDEN
    assert LIMITS["record_bytes"] == ref.RECORD_BYTES
    assert LIMITS["tp4_row_multiple"] == ref.TP4_ROW_MULTIPLE
    # One call-state slot per CTA bounds every program's grid at the one-shot bound of LIMITS.
    assert {record["grid_max"] for record in PROGRAMS.values()} == {GRID_MAX}
    assert LIMITS["tp2_oneshot_grid_max"] == GRID_MAX
    assert set(loader.FAMILIES) == {"allreduce", "rmsnorm"}
    assert loader.world_sizes("allreduce") == (2, 4)
    assert loader.world_sizes("rmsnorm") == (2,)
    assert set(loader.INPUT_FORMATS) == set(INPUT_FORMATS)
    assert set(loader.OUTPUT_FORMATS) == set(OUTPUT_FORMATS)
    with pytest.raises(ValueError):
        loader.world_sizes("layernorm")


# ---------------------------------------------------------------------------
# receive-window plan, re-derived from the kernels' window formulas
# ---------------------------------------------------------------------------


def _linear(form, cap: int):
    """``a * cap + b * capo + c`` of a registry linear form (``None`` stays ``None``)."""
    if form is None:
        return None
    assert set(form) <= {"cap", "capo", "const"}, form
    capo = -(-cap // 4)
    return form.get("cap", 0) * cap + form.get("capo", 0) * capo + form.get("const", 0)


def _expected_window(family: str, world_size: int, input_format: str, cap: int) -> dict:
    """The windows of the kernel contract at capacity ``cap``: bytes, fill period, fill runs, table and control
    words (every window: one four-word call-state slot per CTA of the grid bound)."""
    ctrl_words = CTRL_WORDS_PER_CTA * GRID_MAX
    nr = world_size
    if family == "allreduce" and world_size == 4:
        # Per rotation buffer a reduce-scatter and an all-gather slot array of nr * capo rows each.
        capo = -(-cap // 4)
        rows = nr * capo
        buffers = [
            (
                "wire",
                "ts_wire",
                ROTATION * 2 * rows * RECORD_BYTES,
                rows * RECORD_BYTES,
                ((VAL, rows * VALUE_BYTES), (SF, rows * SCALE_BYTES)),
            ),
            (
                "rsplane",
                "ts_rs",
                ROTATION * 2 * rows * ROW_SCALE_BYTES,
                None,
                ((RS, "rest"),),
            ),
        ]
        return {
            "ctrl_words": ctrl_words,
            "buffers": buffers,
            "table": ("wire", "rsplane"),
        }
    rows = nr * cap
    if family == "allreduce" and input_format == "nvfp4":
        # Three buffers of nr * cap records, each record its own run of sentinels; no row-scale plane.
        buffers = [
            (
                "wire",
                "q_wire",
                ROTATION * rows * RECORD_BYTES,
                RECORD_BYTES,
                ((VAL, VALUE_BYTES), (SF, SCALE_BYTES)),
            ),
        ]
        return {"ctrl_words": ctrl_words, "buffers": buffers, "table": ("wire",)}
    buffers = [
        (
            "wire",
            "os_wire",
            ROTATION * rows * RECORD_BYTES,
            rows * RECORD_BYTES,
            ((VAL, rows * VALUE_BYTES), (SF, rows * SCALE_BYTES)),
        ),
        ("rsplane", "os_rs", ROTATION * rows * ROW_SCALE_BYTES, None, ((RS, "rest"),)),
    ]
    return {"ctrl_words": ctrl_words, "buffers": buffers, "table": ("wire", "rsplane")}


def _registry_window(key: str, cap: int) -> dict:
    """The registry's window evaluated at ``cap`` in the shape of :func:`_expected_window`."""
    window = WINDOWS[key]
    buffers = []
    for buffer in window["buffers"]:
        nbytes = _linear(buffer["bytes"], cap)
        period = _linear(buffer["period"], cap)
        fills = tuple(
            (word, run if run == "rest" else _linear(run, cap))
            for word, run in buffer["fills"]
        )
        buffers.append((buffer["name"], buffer["param"], nbytes, period, fills))
    return {
        "ctrl_words": window["ctrl_words"],
        "buffers": buffers,
        "table": tuple(window["table"]),
    }


def _resolve_rest(
    nbytes: int, period, fills
) -> tuple[int, tuple[tuple[int, int], ...]]:
    """The period in bytes and the fill runs with ``"rest"`` resolved."""
    period = nbytes if period is None else period
    named = [run for _, run in fills if run != "rest"]
    rest = period - sum(named)
    return period, tuple((word, rest if run == "rest" else run) for word, run in fills)


@pytest.mark.cpu_only
@pytest.mark.parametrize("key", sorted(WINDOWS))
@pytest.mark.parametrize("cap", CAPACITIES)
def test_window_plan_matches_the_kernel_contract(key: str, cap: int) -> None:
    family, world_size, input_format = key.split("/")
    world_size = int(world_size)
    expected = _expected_window(family, world_size, input_format, cap)
    actual = _registry_window(key, cap)
    assert actual["ctrl_words"] == expected["ctrl_words"]
    assert actual["table"] == expected["table"]
    assert [buffer[0] for buffer in actual["buffers"]] == [
        buffer[0] for buffer in expected["buffers"]
    ]
    for got, want in zip(actual["buffers"], expected["buffers"], strict=True):
        name, param, nbytes, period, fills = got
        assert (name, param, nbytes) == want[:3], (key, cap, name)
        period, runs = _resolve_rest(nbytes, period, fills)
        want_period, want_runs = _resolve_rest(want[2], want[3], want[4])
        assert period == want_period, (key, cap, name)
        assert runs == want_runs, (key, cap, name)
        # Whole periods, whole words, every run positive, exactly one "rest" at the end.
        assert nbytes % period == 0 and nbytes % 4 == 0
        assert [run for _, run in fills].count("rest") == 1 and fills[-1][1] == "rest"
        assert all(run > 0 and run % 4 == 0 for _, run in runs)
        assert sum(run for _, run in runs) == period
        # Periods: the row-scale plane is one fill; the two-shot wire holds six slot arrays (a reduce-scatter
        # and an all-gather array per rotation buffer); the NVFP4 record window one period per record;
        # the planar one-shot wire one period per rotation buffer.
        periods = nbytes // period
        if name == "rsplane":
            assert periods == 1
        elif world_size == 4:
            assert periods == 6
        elif family == "allreduce" and input_format == "nvfp4":
            assert periods == 3 * world_size * cap
        else:
            assert periods == 3
        assert len(runs) == (2 if name == "wire" else 1)
        # Every window fits the 32-bit in-window offsets of the kernels at every capacity under the limit.
        if world_size == 4:
            assert cap > LIMITS["tp4_cap_max"] or nbytes < 2**31
    # Table entries: the copies of ranks 0..nr-1 of every buffer; the control window holds one four-word
    # call-state slot per CTA of the window's programs' grid bound (the one-shot bound of LIMITS, on every
    # window).
    assert len(actual["table"]) * world_size == (
        2 * world_size if expected["table"] == ("wire", "rsplane") else world_size
    )
    programs = loader.programs(
        family=family, world_size=world_size, input_format=input_format
    )
    assert actual["ctrl_words"] == CTRL_WORDS_PER_CTA * loader.grid_max(programs)
    assert actual["ctrl_words"] == CTRL_WORDS_PER_CTA * LIMITS["tp2_oneshot_grid_max"]
    # The loader's own evaluator agrees with the independent one.
    for buffer in WINDOWS[key]["buffers"]:
        assert loader.evaluate_linear_form(buffer["bytes"], cap) == _linear(
            buffer["bytes"], cap
        )
    assert loader.window_contract(family, world_size, input_format) is WINDOWS[key]


@pytest.mark.cpu_only
def test_window_keys_and_buffer_params_match_the_programs() -> None:
    assert set(WINDOWS) == {
        "allreduce/2/bf16",
        "allreduce/2/nvfp4",
        "allreduce/4/bf16",
        "allreduce/4/nvfp4",
        "rmsnorm/2/bf16",
        "rmsnorm/2/nvfp4",
    }
    for key, window in WINDOWS.items():
        family, world_size, input_format = key.split("/")
        programs = [
            record
            for record in PROGRAMS.values()
            if record["family"] == family
            and record["world_size"] == int(world_size)
            and record["input_format"] == input_format
        ]
        assert programs, key
        params = {buffer["param"] for buffer in window["buffers"]}
        for record in programs:
            assert params <= set(record["params"]), (key, record["kernel_symbol"])
            assert "peer_ptrs" in record["params"] and "ctrl" in record["params"]
            # The window provisions the control words of the program's grid bound.
            assert record["grid_max"] * CTRL_WORDS_PER_CTA == window["ctrl_words"], (
                record["kernel_symbol"]
            )
    with pytest.raises(ValueError):
        loader.window_contract("allreduce", 8, "bf16")
    with pytest.raises(ValueError):
        loader.window_contract("rmsnorm", 4, "bf16")


@pytest.mark.cpu_only
def test_window_bytes_at_the_default_capacity() -> None:
    """Hand-computed sizes at the default capacity of 16384 rows."""
    one_shot = _registry_window("allreduce/2/bf16", 16384)["buffers"]
    assert one_shot[0][2] == 3 * 2 * 16384 * 3456 == 339_738_624
    assert one_shot[1][2] == 3 * 2 * 16384 * 16 == 1_572_864
    records = _registry_window("allreduce/2/nvfp4", 16384)["buffers"]
    assert records[0][2] == 339_738_624 and records[0][3] == 3456
    two_shot = _registry_window("allreduce/4/bf16", 16384)["buffers"]
    assert two_shot[0][2] == 3 * 2 * 4 * 4096 * 3456 == 339_738_624
    assert two_shot[1][2] == 3 * 2 * 4 * 4096 * 16 == 1_572_864
    assert _registry_window("rmsnorm/2/nvfp4", 16384)["buffers"][0][2] == 339_738_624
    # capo = ceil(cap / 4): an odd capacity rounds up to whole owner rows.
    assert (
        _registry_window("allreduce/4/nvfp4", 5)["buffers"][0][2]
        == 3 * 2 * 4 * 2 * 3456
    )


# ---------------------------------------------------------------------------
# selection rules
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("rows", "grid", "expected"),
    [
        (64, 148, 1),
        (1024, 148, 1),
        (4096, 148, 1),
        (148 * 40, 148, 1),
        (148 * 41, 148, 0),
        (8192, 148, 0),
        (148 * 64, 148, 0),
        (148 * 65, 148, 2),
        (16384, 148, 2),
        (4096, 152, 1),
        (152 * 40, 152, 1),
        (152 * 41, 152, 0),
        (8192, 152, 0),
        (152 * 64, 152, 0),
        (152 * 65, 152, 2),
        (16384, 152, 2),
        (4, 148, 1),
    ],
)
def test_select_bv_rule(rows: int, grid: int, expected: int) -> None:
    assert loader.select_bv(rows, grid) == expected
    per_cta = rows // grid
    rule = (
        2
        if per_cta > SELECT_BV["by_quad_min"]
        else (0 if per_cta > SELECT_BV["team_max"] else 1)
    )
    assert rule == expected


@pytest.mark.cpu_only
def test_select_bv_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError):
        loader.select_bv(1024, 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("rows", "grid", "expected"),
    [
        (3, 148, False),
        (1024, 148, False),
        (2048, 148, False),
        (148 * 18 - 1, 148, False),
        (148 * 18, 148, True),
        (3072, 148, True),
        (4096, 148, True),
        (148 * 29 - 1, 148, True),
        (148 * 29, 148, False),
        (8192, 148, False),
        (16384, 148, False),
        (152 * 18 - 1, 152, False),
        (152 * 18, 152, True),
        (4096, 152, True),
        (152 * 29 - 1, 152, True),
        (152 * 29, 152, False),
        (4096, 160, True),
    ],
)
def test_select_prefetch_rule(rows: int, grid: int, expected: bool) -> None:
    assert loader.select_prefetch(rows, grid) is expected
    per_cta = rows // grid
    assert expected == (
        SELECT_PREFETCH["rows_per_cta_min"]
        <= per_cta
        <= SELECT_PREFETCH["rows_per_cta_max"]
    )


@pytest.mark.cpu_only
def test_select_prefetch_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError):
        loader.select_prefetch(4096, 0)


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("input_format", INPUT_FORMATS)
def test_program_for_two_shot_picks_the_targets_forms(
    target: str, input_format: str
) -> None:
    grid = 148
    for rows, bv in ((1024, 1), (8192, 0), (16384, 2)):
        assert loader.select_bv(rows, grid) == bv
        program = loader.program_for(
            family="allreduce",
            world_size=4,
            input_format=input_format,
            output_format=None,
            rows=rows,
            grid=grid,
            target=target,
        )
        suffix = _two_shot_suffix(target, input_format, bv)
        assert program == f"tp4_twoshot_{input_format}_bv{bv}{suffix}"
        record = PROGRAMS[program]
        assert target in record["targets"] and record["bv"] == bv
        variant = (input_format, bv)
        assert record["reset"] == (
            "st" if variant in REGISTER_STORE_FORMS[target] else "bulk"
        )


@pytest.mark.cpu_only
@pytest.mark.parametrize("target", TARGETS)
def test_program_for_one_shot_and_fused_norm(target: str) -> None:
    """The one-shot and fused-norm calls resolve the target's whole program, or its band program inside the
    band's row counts (the fused norm's prefetch form first, on rows per CTA; the band on the call's rows)."""
    for input_format in INPUT_FORMATS:
        for rows in (3, 1031, 2048, 2664, 4096, 4291, 4292, 6143, 6144, 6145, 16384):
            suffix = "_pf" if loader.select_prefetch(rows, 148) else ""
            assert (rows in (2664, 4096, 4291)) == bool(suffix)
            assert loader.program_for(
                family="allreduce",
                world_size=2,
                input_format=input_format,
                output_format=None,
                rows=rows,
                grid=148,
                target=target,
            ) == _one_shot_program(target, input_format, rows)
            for output_format in OUTPUT_FORMATS:
                program = loader.program_for(
                    family="rmsnorm",
                    world_size=2,
                    input_format=input_format,
                    output_format=output_format,
                    rows=rows,
                    grid=148,
                    target=target,
                )
                assert program == _fused_norm_program(
                    target, input_format, output_format, bool(suffix), rows
                ), (input_format, output_format, rows)
                # The selected record carries the sum-of-squares order the reference uses on this target.
                band = FUSED_NORM_BANDS.get((target, input_format, output_format))
                inside = bool(band) and loader.select_rows(
                    rows, band[0], BAND_THRESHOLD
                )
                forms = (
                    _fused_norm_band_forms(target, input_format, output_format)
                    if inside
                    else _fused_norm_forms(target, input_format, output_format)
                )
                assert PROGRAMS[program]["ss_order"] == forms["ss_order"]
                # The prefetch form has no band: inside a prefix band its whole program carries the band's
                # forms, and no prefetch row count reaches a suffix band.
                assert (program in loader.SELECT_ROWS) == (inside and not suffix)
    with pytest.raises(ValueError):
        loader.program_for(
            family="allreduce",
            world_size=8,
            input_format="bf16",
            output_format=None,
            rows=64,
            grid=148,
            target=target,
        )
    with pytest.raises(ValueError):
        loader.program_for(
            family="rmsnorm",
            world_size=2,
            input_format="bf16",
            output_format="fp8",
            rows=64,
            grid=148,
            target=target,
        )
    with pytest.raises(ValueError):
        loader.program_for(
            family="allreduce",
            world_size=2,
            input_format="bf16",
            output_format=None,
            rows=64,
            grid=148,
            target="sm_90a",
        )


# ---------------------------------------------------------------------------
# rows-dispatched programs: SELECT_ROWS and the rows filter of program_for
# ---------------------------------------------------------------------------

# The synthetic bands the dispatch tests install beside the registry's whole programs of one target: the
# fused-norm cell, the band kind, the name token and the record keys the band deviates in.  A prefix band
# serves rows <= threshold, a suffix band rows >= threshold (both inclusive); the whole program serves
# the other row counts.
# (input format, output format, band kind, name token, the band record's keys over the whole program's record,
# the whole program record's own keys): the third pair's whole program and band differ in more than one key --
# the band runs on eleven workers and sums the pass-1 squares in the row's chunk order where the whole
# program permutes them -- and the rows filter pairs them by cell all the same.
SYNTHETIC_BANDS = (
    ("nvfp4", "bf16", "prefix", "_sop", {"ss_order": "perm"}, {}),
    ("bf16", "bf16", "suffix", "_w11", {"block": 384}, {}),
    (
        "bf16",
        "nvfp4",
        "suffix",
        "_p2d_w11",
        {"block": 384, "ss_order": "natural"},
        {"ss_order": "perm"},
    ),
)


def _whole_program(
    target: str, input_format: str, output_format: str, prefetch: bool
) -> str:
    (program,) = [
        program
        for program, record in PROGRAMS.items()
        if record["family"] == "rmsnorm"
        and record["input_format"] == input_format
        and record["output_format"] == output_format
        and record["prefetch"] is prefetch
        and target in record["targets"]
        and program not in loader.SELECT_ROWS
    ]
    return program


@pytest.fixture
def synthetic_bands(monkeypatch):
    """A registry with rows-dispatched programs on sm_103a, installed over the registry's whole programs (its
    own bands set aside) the way the generator writes them: one PROGRAMS record per (band, prefetch form)
    carrying the whole program's record with the band's keys over it, and SELECT_ROWS naming each band's
    kind and threshold.  Yields ``{(input_format, output_format, prefetch): (band, whole)}``."""

    target = "sm_103a"
    programs = {
        name: dict(record)
        for name, record in loader.PROGRAMS.items()
        if name not in loader.SELECT_ROWS
    }
    bands: dict[str, dict] = {}
    cells: dict[tuple[str, str, bool], tuple[str, str]] = {}
    for input_format, output_format, kind, token, extra, whole_extra in SYNTHETIC_BANDS:
        for prefetch in (False, True):
            whole = _whole_program(target, input_format, output_format, prefetch)
            name = whole.replace("_pf", "") + token + ("_pf" if prefetch else "")
            assert name not in programs
            programs[whole] = {**programs[whole], **whole_extra}
            programs[name] = {
                **programs[whole],
                **extra,
                "targets": (target,),
                "kernel_symbol": f"cake_fp4_allreduce_{name}",
            }
            bands[name] = {"kind": kind, "threshold": BAND_THRESHOLD}
            cells[(input_format, output_format, prefetch)] = (name, whole)
    monkeypatch.setattr(loader, "PROGRAMS", programs)
    monkeypatch.setattr(loader, "SELECT_ROWS", bands)
    loader.program_for.cache_clear()
    yield cells
    loader.program_for.cache_clear()


@pytest.mark.cpu_only
def test_select_rows_rule_is_inclusive_at_the_threshold() -> None:
    probes = (1, BAND_THRESHOLD - 1, BAND_THRESHOLD, BAND_THRESHOLD + 1, 16384)
    assert [loader.select_rows(rows, "prefix", BAND_THRESHOLD) for rows in probes] == [
        True,
        True,
        True,
        False,
        False,
    ]
    assert [loader.select_rows(rows, "suffix", BAND_THRESHOLD) for rows in probes] == [
        False,
        False,
        True,
        True,
        True,
    ]
    with pytest.raises(ValueError):
        loader.select_rows(BAND_THRESHOLD, "middle", BAND_THRESHOLD)
    with pytest.raises(ValueError):
        loader.select_rows(0, "prefix", BAND_THRESHOLD)


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefetch", [False, True])
def test_program_for_dispatches_the_band_inside_its_rows(
    synthetic_bands, prefetch: bool
) -> None:
    """At threshold - 1 / threshold / threshold + 1 a prefix band's program is served / served / not served
    and a suffix band's not served / served / served, the whole program otherwise; the prefetch form is
    chosen first (rows per CTA), the band on the call's row count, however many keys the pair differs in."""
    target = "sm_103a"
    grid = (
        256 if prefetch else 148
    )  # 6144 // 256 = 24 rows per CTA (prefetch), 6144 // 148 = 41 (plain)
    probes = (BAND_THRESHOLD - 1, BAND_THRESHOLD, BAND_THRESHOLD + 1)
    for rows in probes:
        assert loader.select_prefetch(rows, grid) is prefetch
    for input_format, output_format, kind, _, extra, whole_extra in SYNTHETIC_BANDS:
        band, whole = synthetic_bands[(input_format, output_format, prefetch)]
        expected = (band, band, whole) if kind == "prefix" else (whole, band, band)
        for rows, program in zip(probes, expected, strict=True):
            assert (
                loader.program_for(
                    family="rmsnorm",
                    world_size=2,
                    input_format=input_format,
                    output_format=output_format,
                    rows=rows,
                    grid=grid,
                    target=target,
                )
                == program
            ), (input_format, output_format, rows)
        far = (
            (1 * grid + 1, 64 * grid) if kind == "prefix" else (64 * grid, 1 * grid + 1)
        )
        assert loader.select_prefetch(far[0], grid) is False
        assert loader.select_prefetch(far[1], grid) is False
        plain_band, plain_whole = synthetic_bands[(input_format, output_format, False)]
        inside, outside = far
        assert (
            loader.program_for(
                "rmsnorm", 2, input_format, output_format, inside, grid, target
            )
            == plain_band
        )
        assert (
            loader.program_for(
                "rmsnorm", 2, input_format, output_format, outside, grid, target
            )
            == plain_whole
        )
        # The band's record carries the deviating keys over the whole program's record.
        record = loader.PROGRAMS[band]
        for key, value in extra.items():
            assert record[key] == value
        # The whole program's own keys stay its own: a pair differing in several keys is still one cell.
        for key, value in whole_extra.items():
            assert loader.PROGRAMS[whole][key] == value, (whole, key)
            assert record[key] != value, (band, key)


@pytest.mark.cpu_only
@pytest.mark.parametrize("prefetch", [False, True])
def test_program_for_keeps_the_whole_program_without_a_band(
    synthetic_bands, prefetch: bool
) -> None:
    """A cell without a band, and the dispatched cells on the other targets, resolve the whole program at
    every probe (a registry without bands)."""
    grid = 256 if prefetch else 148
    probes = (BAND_THRESHOLD - 1, BAND_THRESHOLD, BAND_THRESHOLD + 1, 1024, 16384)
    for target in TARGETS:
        for input_format in INPUT_FORMATS:
            for output_format in OUTPUT_FORMATS:
                banded = target == "sm_103a" and any(
                    (i, o) == (input_format, output_format)
                    for i, o, *_ in SYNTHETIC_BANDS
                )
                if banded:
                    continue
                for rows in probes:
                    program = loader.program_for(
                        family="rmsnorm",
                        world_size=2,
                        input_format=input_format,
                        output_format=output_format,
                        rows=rows,
                        grid=grid,
                        target=target,
                    )
                    assert program not in loader.SELECT_ROWS, (
                        target,
                        input_format,
                        output_format,
                        rows,
                    )
                    assert loader.PROGRAMS[program][
                        "prefetch"
                    ] is loader.select_prefetch(rows, grid)
    # The all-reduce programs, with no band installed here, resolve as before.
    for target in TARGETS:
        assert loader.program_for(
            "allreduce", 2, "bf16", None, BAND_THRESHOLD, 148, target
        ) == (f"tp2_oneshot_bf16{ONESHOT_BF16_FORMS[target]}")


@pytest.mark.cpu_only
def test_program_for_refuses_two_bands_serving_one_call(
    synthetic_bands, monkeypatch
) -> None:
    """Exactly one candidate must remain: a second band of the same cell inside the row count is refused,
    as a cell left without a whole program is outside every band."""
    band, whole = synthetic_bands[("bf16", "bf16", False)]
    second = band + "_x"
    programs = dict(loader.PROGRAMS)
    programs[second] = dict(
        programs[band], kernel_symbol=f"cake_fp4_allreduce_{second}"
    )
    bands = dict(loader.SELECT_ROWS)
    bands[second] = dict(bands[band])
    monkeypatch.setattr(loader, "PROGRAMS", programs)
    monkeypatch.setattr(loader, "SELECT_ROWS", bands)
    loader.program_for.cache_clear()
    with pytest.raises(RuntimeError, match="resolves 2 programs"):
        loader.program_for("rmsnorm", 2, "bf16", "bf16", BAND_THRESHOLD, 148, "sm_103a")
    assert (
        loader.program_for(
            "rmsnorm", 2, "bf16", "bf16", BAND_THRESHOLD - 1, 148, "sm_103a"
        )
        == whole
    )
    programs.pop(whole)
    monkeypatch.setattr(loader, "PROGRAMS", programs)
    loader.program_for.cache_clear()
    with pytest.raises(RuntimeError, match="resolves 0 programs"):
        loader.program_for(
            "rmsnorm", 2, "bf16", "bf16", BAND_THRESHOLD - 1, 148, "sm_103a"
        )
    loader.program_for.cache_clear()


@pytest.fixture
def synthetic_oneshot_band(monkeypatch):
    """A registry with a rows-dispatched one-shot all-reduce program on sm_100a, installed over the registry's
    whole programs (its own bands set aside) the way the generator writes it: the whole NVFP4 one-shot
    program's record with the band's re-arm over it, and SELECT_ROWS naming the band's kind and threshold.
    Yields ``(band, whole)``."""

    target = "sm_100a"
    whole, band = "tp2_oneshot_nvfp4", "tp2_oneshot_nvfp4_rsb"
    programs = {
        name: dict(record)
        for name, record in loader.PROGRAMS.items()
        if name not in loader.SELECT_ROWS
    }
    assert programs[whole]["family"] == "allreduce"
    assert target in programs[whole]["targets"]
    assert band not in programs
    programs[band] = {
        **programs[whole],
        "reset": "bulk",
        "targets": (target,),
        "kernel_symbol": f"cake_fp4_allreduce_{band}",
    }
    bands = {band: {"kind": "suffix", "threshold": ONESHOT_BAND_THRESHOLD}}
    monkeypatch.setattr(loader, "PROGRAMS", programs)
    monkeypatch.setattr(loader, "SELECT_ROWS", bands)
    loader.program_for.cache_clear()
    yield band, whole
    loader.program_for.cache_clear()


@pytest.mark.cpu_only
def test_program_for_dispatches_the_one_shot_band_at_or_above_its_threshold(
    synthetic_oneshot_band,
) -> None:
    """The rows filter is per call, whatever the kernel: on sm_100a the NVFP4 one-shot call runs the whole
    program below the threshold and the band at and above it (the call's row count, inclusive; the one-shot
    kernel has no prefetch form to choose first, and the grid plays no part); the BF16 one-shot call and the
    other targets' NVFP4 calls keep their whole program at every probe, as do the fused-norm calls."""
    band, whole = synthetic_oneshot_band
    probes = (
        1,
        4096,
        ONESHOT_BAND_THRESHOLD - 1,
        ONESHOT_BAND_THRESHOLD,
        ONESHOT_BAND_THRESHOLD + 1,
        100000,
    )
    expected = (whole, whole, whole, band, band, band)

    def resolve(fmt: str, rows: int, grid: int, target: str) -> str:
        return loader.program_for("allreduce", 2, fmt, None, rows, grid, target)

    for grid in (148, 4096):
        for rows, program in zip(probes, expected, strict=True):
            assert resolve("nvfp4", rows, grid, "sm_100a") == program, (rows, grid)
    for target in ("sm_103a", "sm_107a"):
        for rows in probes:
            assert resolve("nvfp4", rows, 148, target) == whole, (target, rows)
    for target in TARGETS:
        bf16 = f"tp2_oneshot_bf16{ONESHOT_BF16_FORMS[target]}"
        for rows in probes:
            assert resolve("bf16", rows, 148, target) == bf16, (target, rows)
    # The band's record: the bulk re-arm over the whole program's register stores, one target; the
    # whole program keeps its three targets.
    record, plain = loader.PROGRAMS[band], loader.PROGRAMS[whole]
    assert (record["reset"], plain["reset"]) == ("bulk", "st")
    assert (record["output_format"], record["prefetch"], record["bv"]) == (
        None,
        None,
        None,
    )
    assert record["targets"] == ("sm_100a",)
    assert plain["targets"] == TARGETS
    assert loader.SELECT_ROWS[band] == {
        "kind": "suffix",
        "threshold": ONESHOT_BAND_THRESHOLD,
    }
    # The fused-norm calls of the same target and input format are untouched by the one-shot band.
    for output_format in OUTPUT_FORMATS:
        for rows in (ONESHOT_BAND_THRESHOLD - 1, ONESHOT_BAND_THRESHOLD):
            program = loader.program_for(
                "rmsnorm", 2, "nvfp4", output_format, rows, 148, "sm_100a"
            )
            assert program not in loader.SELECT_ROWS, (output_format, rows)


@pytest.mark.cpu_only
def test_grid_max_is_the_least_bound_of_the_programs(monkeypatch) -> None:
    """Every workspace's programs bound its grid at the CTAs its control window provisions (4096, on every
    path), and the bound of a selection is the least of its programs'."""
    for key, window in WINDOWS.items():
        family, world_size, input_format = key.split("/")
        for target in TARGETS:
            programs = loader.workspace_programs(
                family, int(world_size), input_format, target
            )
            bound = loader.grid_max(programs)
            assert bound == window["ctrl_words"] // CTRL_WORDS_PER_CTA == GRID_MAX
    assert loader.grid_max(tuple(PROGRAMS)) == GRID_MAX
    with pytest.raises(ValueError):
        loader.grid_max(())
    with pytest.raises(KeyError):
        loader.grid_max(("tp2_oneshot_fp8",))
    lowered = dict(PROGRAMS["tp2_oneshot_bf16"], grid_max=148)
    monkeypatch.setitem(PROGRAMS, "tp2_oneshot_bf16", lowered)
    assert loader.grid_max(("tp2_oneshot_nvfp4", "tp2_oneshot_bf16")) == 148
    assert loader.grid_max(("tp2_oneshot_nvfp4",)) == GRID_MAX


# ---------------------------------------------------------------------------
# build targets
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_build_targets_follow_flashinfer_cuda_arch_list(monkeypatch) -> None:
    def pin(arch_list: str) -> None:
        monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", arch_list)
        loader.target_capabilities.cache_clear()

    try:
        pin("9.0 10.0 10.3 12.0f")
        assert loader.target_capabilities() == ((10, 0), (10, 3))
        assert loader.supported_capability((10, 0)) == (10, 0)
        assert loader.supported_capability((10, 3)) == (10, 3)
        for capability in ((9, 0), (10, 7), (12, 0), (12, 1)):
            assert loader.supported_capability(capability) is None
        pin("10.0")
        assert loader.target_capabilities() == ((10, 0),)
        assert loader.supported_capability((10, 3)) is None
        pin("10.3")
        assert loader.target_capabilities() == ((10, 3),)
        assert loader.supported_capability((10, 0)) is None
        pin("10.7")
        assert loader.target_capabilities() == ((10, 7),)
        assert loader.supported_capability((10, 7)) == (10, 7)
        assert loader.supported_capability((10, 3)) is None
        pin("9.0 12.0f")
        assert loader.target_capabilities() == ()
    finally:
        loader.target_capabilities.cache_clear()


@pytest.mark.cpu_only
def test_exact_target_table() -> None:
    assert set(loader.TARGET_FLAGS) == set(TARGETS)
    assert (
        loader.TARGET_FLAGS["sm_100a"][0] == "-gencode=arch=compute_100a,code=sm_100a"
    )
    assert (
        loader.TARGET_FLAGS["sm_103a"][0] == "-gencode=arch=compute_103a,code=sm_103a"
    )
    assert (
        loader.TARGET_FLAGS["sm_107a"][0] == "-gencode=arch=compute_107a,code=sm_107a"
    )
    assert "--fmad=false" in loader.COMPILE_FLAGS


# ---------------------------------------------------------------------------
# reference unit checks
# ---------------------------------------------------------------------------


@pytest.mark.cpu_only
def test_e2m1_rounding_table() -> None:
    values = torch.tensor(
        [
            0.0,
            0.24,
            0.25,
            0.26,
            0.75,
            1.0,
            1.25,
            1.26,
            1.75,
            2.5,
            2.51,
            3.5,
            5.0,
            5.01,
            6.0,
            7.0,
            math.inf,
            -0.1,
            -0.3,
            -6.5,
            math.nan,
        ],
        dtype=torch.float32,
    )
    codes = ref.e2m1_rn_satfinite(values)
    assert codes.tolist() == [
        0,
        0,
        0,
        1,
        2,
        2,
        2,
        3,
        4,
        4,
        5,
        6,
        6,
        7,
        7,
        7,
        7,
        8,
        9,
        15,
        7,
    ]
    assert ref.canon_neg0(codes).tolist()[17] == 0
    assert ref.e2m1_to_f32(torch.arange(16, dtype=torch.uint8)).tolist() == list(
        ref.E2M1_VALUES
    )
    # Ties round to the even code: 0.25 -> 0, 0.75 -> 1.0, 1.25 -> 1.0, 1.75 -> 2, 2.5 -> 2, 3.5 -> 4, 5.0 -> 4.
    ties = ref.e2m1_to_f32(ref.e2m1_rn_satfinite(torch.tensor(ref.E2M1_TIES)))
    assert ties.tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0]


@pytest.mark.cpu_only
def test_e4m3_saturates_and_keeps_nan() -> None:
    values = torch.tensor(
        [448.0, 449.0, 1e9, -1e9, 0.0, 0.1, 1.0, math.nan], dtype=torch.float32
    )
    codes = ref.e4m3_rn_satfinite(values)
    decoded = ref.e4m3_to_f32(codes)
    assert decoded[:5].tolist() == [448.0, 448.0, 448.0, -448.0, 0.0]
    assert abs(decoded[5].item() - 0.1) <= 0.1 / 8
    assert codes[6].item() == 0x38 and decoded[6].item() == 1.0
    assert codes[7].item() == 0x7F


@pytest.mark.cpu_only
def test_float32_primitives_are_correctly_rounded() -> None:
    generator = torch.Generator().manual_seed(7)
    a = torch.randn(256, generator=generator) * 1e3
    b = torch.randn(256, generator=generator)
    c = torch.randn(256, generator=generator) * 1e-3
    results = {
        "fma": (
            ref.f32_fma_rn(a, b, c),
            lambda i: (
                Fraction(a[i].item()) * Fraction(b[i].item()) + Fraction(c[i].item())
            ),
        ),
        "mul": (
            ref.f32_mul_rn(a, b),
            lambda i: Fraction(a[i].item()) * Fraction(b[i].item()),
        ),
        "div": (
            ref.f32_div_rn(a, b),
            lambda i: Fraction(a[i].item()) / Fraction(b[i].item()),
        ),
        "add": (
            ref.f32_add_rn(a, c),
            lambda i: Fraction(a[i].item()) + Fraction(c[i].item()),
        ),
    }
    for name, (result, exact_of) in results.items():
        assert result.dtype == torch.float32, name
        up = torch.nextafter(result, torch.full_like(result, math.inf))
        down = torch.nextafter(result, torch.full_like(result, -math.inf))
        for i in range(a.numel()):
            exact = exact_of(i)
            error = abs(exact - Fraction(result[i].item()))
            assert error <= abs(exact - Fraction(up[i].item())), (name, i)
            assert error <= abs(exact - Fraction(down[i].item())), (name, i)


@pytest.mark.cpu_only
def test_pack_unpack_roundtrip() -> None:
    codes = (torch.arange(2 * ref.HIDDEN) % 16).to(torch.uint8).reshape(2, ref.HIDDEN)
    packed = ref.pack_e2m1(codes)
    assert packed.shape == (2, ref.VALUE_BYTES) and packed.dtype == torch.uint8
    assert (
        packed[0, 0].item() == 0x10
    )  # element 0 (code 0) in the low nibble, element 1 (code 1) in the high one
    assert torch.equal(ref.unpack_e2m1(packed), codes)


@pytest.mark.cpu_only
def test_codec_reproduces_exact_cases() -> None:
    x = torch.full((2, ref.HIDDEN), 2688.0, dtype=torch.bfloat16)
    # amax 2688: inv = rs = 1, every block scale 448, every value 6 -> 2688 per rank.
    q, sf, rs = ref.encode_bf16(x)
    assert torch.equal(sf, torch.full_like(sf, 0x7E)) and torch.equal(
        q, torch.full_like(q, 0x77)
    )
    assert rs.tolist() == [1.0, 1.0]
    assert torch.equal(ref.reference_allreduce([x, x]), torch.full_like(x, 5376.0))
    # Two-shot (a multiple of 4 rows): the owner re-encodes 10752 under rs 4 (inv 0.25) and decodes it exactly.
    x4 = torch.full((4, ref.HIDDEN), 2688.0, dtype=torch.bfloat16)
    assert torch.equal(
        ref.reference_allreduce([x4, x4, x4, x4]), torch.full_like(x4, 10752.0)
    )
    # A zero row takes the inv = 0 path: zero scales, zero codes, zero output.
    zero = torch.zeros((4, ref.HIDDEN), dtype=torch.bfloat16)
    q0, sf0, rs0 = ref.encode_bf16(zero)
    assert int(q0.sum()) == 0 and int(sf0.sum()) == 0 and rs0.tolist() == [0.0] * 4
    x4 = torch.full((4, ref.HIDDEN), 2688.0, dtype=torch.bfloat16)
    assert torch.equal(ref.reference_allreduce([zero, x4]), x4)
    assert torch.equal(
        ref.reference_allreduce([zero, zero, x4, x4]), torch.full_like(x4, 5376.0)
    )
    # A tiny negative value next to a large maximum rounds to E2M1 -0 and is canonicalised to +0 on the wire.
    mixed = torch.zeros((1, ref.HIDDEN), dtype=torch.bfloat16)
    mixed[0, 0] = 2688.0
    mixed[0, 1] = -1e-3
    q1 = ref.unpack_e2m1(ref.encode_bf16(mixed)[0])
    assert q1[0, 0].item() == 7 and q1[0, 1].item() == 0
    with pytest.raises(ValueError):
        ref.reference_allreduce([x, x, x])
    with pytest.raises(ValueError):
        ref.reference_allreduce(
            [x[:1], x[:1], x[:1], x[:1]]
        )  # the two-shot needs a multiple of 4 rows


@pytest.mark.cpu_only
def test_records_layout_and_static_scale() -> None:
    xs = ref.partials(4, 7, 2, "cpu")
    records, scales = ref.records_of(xs)
    assert scales.dtype == torch.float32 and tuple(scales.shape) == (2,)
    assert all(scale > 0 for scale in scales.tolist())
    for x, rec in zip(xs, records, strict=True):
        assert rec.shape == (4, ref.RECORD_BYTES) and rec.dtype == torch.uint8
        values, block_scales = ref.split_records(rec)
        assert values.shape == (4, ref.VALUE_BYTES) and block_scales.shape == (
            4,
            ref.SCALE_BYTES,
        )
        assert ref.global_scale(x).item() == pytest.approx(
            x.float().abs().amax().item() / 2688.0, rel=1e-6
        )
    out = ref.reference_allreduce_nvfp4(records, scales)
    assert out.shape == (4, ref.HIDDEN) and out.dtype == torch.bfloat16
    assert torch.isfinite(out.float()).all()
    # Under a static scale of 1 (amax 2688) the records equal the per-row encoding.
    x = torch.full((2, ref.HIDDEN), 2688.0, dtype=torch.bfloat16)
    rec = ref.encode_records(x, ref.global_scale(x))
    q, sf, _ = ref.encode_bf16(x)
    assert torch.equal(rec, torch.cat((q, sf), dim=1))
    assert torch.equal(
        ref.reference_allreduce_nvfp4([rec, rec], torch.ones(2)),
        torch.full_like(x, 5376.0),
    )
    with pytest.raises(ValueError):
        ref.split_records(rec[:, :-1])
    with pytest.raises(ValueError):
        ref.sources_records([rec, rec], torch.ones(1))


@pytest.mark.cpu_only
def test_partials_are_deterministic_with_outliers_zero_and_tiny_rows() -> None:
    rows = 200
    xs = ref.partials(rows, 123, 4, "cpu")
    assert len(xs) == 4
    for x in xs:
        assert x.shape == (rows, ref.HIDDEN) and x.dtype == torch.bfloat16
        assert int((x[96::97] != 0).sum()) == 0  # every 97th row is all zero
        tiny = x[88::89].float().abs()
        assert 0.0 < float(tiny.max()) < 2.0**-10  # every 89th row is scaled by 2^-20
        assert (
            int((x.float().abs() > 20.0).sum()) > rows
        )  # the outliers (one in a hundred, x 32)
    again = ref.partials(rows, 123, 4, "cpu")
    for x, y in zip(xs, again, strict=True):
        assert torch.equal(x, y)
    assert not torch.equal(xs[0], xs[1])
    assert not torch.equal(xs[0], ref.partials(rows, 124, 1, "cpu")[0])
    assert ref.launch_seed(1000, 1) == 1007
    residual = ref.residual(rows, 5, "cpu")
    assert residual.shape == (rows, ref.HIDDEN) and torch.equal(
        residual, ref.residual(rows, 5, "cpu")
    )
    weight = ref.weight(5, "cpu")
    assert weight.shape == (ref.HIDDEN,) and torch.equal(weight, ref.weight(5, "cpu"))
    assert float(weight.float().abs().max()) < 4.0


@pytest.mark.cpu_only
def test_swizzled_scale_offsets_match_the_direct_rule() -> None:
    rows, blocks = 200, ref.NVFP4_SCALES_PER_ROW
    assert (
        ref.sf_rows(rows) == 256 and ref.sf_rows(1024) == 1024 and ref.sf_rows(1) == 128
    )
    offsets = ref.sf_offsets(rows, blocks, "cpu")
    assert offsets.shape == (rows, blocks)
    ktiles = blocks // 4
    for row, block in (
        (0, 0),
        (1, 0),
        (31, 3),
        (32, 4),
        (37, 5),
        (127, 383),
        (128, 0),
        (rows - 1, blocks - 1),
    ):
        direct = (
            (row // 128) * ktiles * 512
            + (block // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + (block % 4)
        )
        assert int(offsets[row, block]) == direct
    # Over whole 128-row tiles the rule is a permutation of the scale bytes.
    full = ref.sf_offsets(256, blocks, "cpu").reshape(-1)
    assert torch.equal(torch.sort(full).values, torch.arange(256 * blocks))
    assert ref.scale_bytes(rows, blocks) == 256 * 384
    assert ref.scale_bytes(1000, ref.MXFP8_SCALES_PER_ROW) == 1024 * 192
    # unswizzle is the inverse of the scatter.
    marks = torch.zeros((ref.scale_bytes(rows, blocks),), dtype=torch.uint8)
    marks[int(offsets[37, 5])] = 22
    marks[int(offsets[rows - 1, blocks - 1])] = 33
    block_order = ref.unswizzle_scales(marks, rows, blocks)
    assert block_order.shape == (rows, blocks)
    assert (
        int(block_order[37, 5]) == 22 and int(block_order[rows - 1, blocks - 1]) == 33
    )
    assert int(block_order.sum()) == 55
    with pytest.raises(ValueError):
        ref.unswizzle_scales(marks[:-1], rows, blocks)


@pytest.mark.cpu_only
def test_fused_norm_reference_on_cpu() -> None:
    rows = 3
    xs = ref.partials(rows, 321, 2, "cpu")
    residual = ref.residual(rows, 321, "cpu")
    weight = ref.weight(321, "cpu")
    reference = ref.reference_fused_rmsnorm(xs, residual, weight)
    assert (
        reference.y.shape == (rows, ref.HIDDEN) and reference.y.dtype == torch.bfloat16
    )
    assert reference.residual_out.dtype == torch.bfloat16
    assert reference.eps == ref.EPS and reference.weight_bias == ref.WEIGHT_BIAS
    assert reference.ss_order == "natural"
    acc = ref.oneshot_sum(ref.sources_bf16(xs))
    z = ref.f32_add_rn(acc, residual.float())
    assert torch.equal(reference.residual_out, z.to(torch.bfloat16))
    # ss in the kernel's order is within a few float32 ulps of the exact sum of squares.
    exact = (z.double() ** 2).sum(dim=1)
    assert torch.allclose(reference.ss.double(), exact, rtol=1e-6, atol=0.0)
    rinv = 1.0 / torch.sqrt(exact / ref.HIDDEN + ref.EPS)
    assert torch.allclose(reference.rinv.double(), rinv, rtol=1e-6, atol=0.0)
    # The permuted sum-of-squares order (the ss_order of the programs that measured it faster) moves ss and
    # rinv by at most an ulp: residual_out is bitwise the same, y stays within the tolerance.
    permuted = ref.reference_fused_rmsnorm(xs, residual, weight, ss_order="perm")
    assert permuted.ss_order == "perm"
    assert torch.equal(permuted.residual_out, reference.residual_out)
    assert torch.allclose(permuted.ss.double(), exact, rtol=1e-6, atol=0.0)
    assert ref.max_bf16_ulp(permuted.y, reference.y) <= 1
    assert ref.tolerance_violations(permuted.y, reference.y) == 0
    with pytest.raises(ValueError):
        ref.reference_fused_rmsnorm(xs, residual, weight, ss_order="reversed")
    # y = bf16(z) * rinv * (1 + w): the plain formula agrees within the test tolerance.
    plain = (reference.residual_out.float() * rinv.float().reshape(rows, 1)) * (
        1.0 + weight.float()
    )
    assert ref.tolerance_violations(reference.y, plain.to(torch.bfloat16)) == 0
    assert ref.max_bf16_ulp(reference.y, plain.to(torch.bfloat16)) <= 1
    # A reference with a wrong weight bias keeps residual_out and moves y outside the tolerance.
    biased = ref.reference_fused_rmsnorm(
        xs, residual, weight, weight_bias=ref.NEGATIVE_CONTROL_WEIGHT_BIAS
    )
    assert torch.equal(biased.residual_out, reference.residual_out)
    assert ref.tolerance_violations(reference.y, biased.y) > 0
    # eps and weight_bias are honoured.
    # mean(z^2) is about 80 for these inputs (the outliers), so a visible eps has to be of that order.
    wide = ref.reference_fused_rmsnorm(xs, residual, weight, eps=100.0)
    assert ref.tolerance_violations(reference.y, wide.y) > 0
    unbiased = ref.reference_fused_rmsnorm(xs, residual, weight, weight_bias=0.0)
    expected = (
        reference.residual_out.float() * reference.rinv.reshape(rows, 1)
    ) * weight.float()
    assert ref.tolerance_violations(unbiased.y, expected.to(torch.bfloat16)) == 0
    # The records path shares the residual arithmetic.
    records, scales = ref.records_of(xs)
    from_records = ref.reference_fused_rmsnorm(
        residual_bf16=residual,
        weight_bf16=weight,
        records=records,
        global_scales=scales,
    )
    assert from_records.y.shape == reference.y.shape
    acc_records = ref.oneshot_sum(ref.sources_records(records, scales))
    assert torch.equal(
        from_records.residual_out,
        ref.f32_add_rn(acc_records, residual.float()).to(torch.bfloat16),
    )
    with pytest.raises(ValueError):
        ref.reference_fused_rmsnorm(xs + xs, residual, weight)
    with pytest.raises(ValueError):
        ref.reference_fused_rmsnorm(xs, residual, weight[:-1])
    with pytest.raises(ValueError):
        ref.reference_fused_rmsnorm(xs, residual[:-1], weight)
    with pytest.raises(ValueError):
        ref.reference_fused_rmsnorm(
            residual_bf16=residual, weight_bf16=weight, records=records
        )


@pytest.mark.cpu_only
def test_sum_of_squares_follows_the_lane_order() -> None:
    z = torch.randn((2, ref.HIDDEN), generator=torch.Generator().manual_seed(3))
    ss = ref.sum_of_squares(z)
    exact = (z.double() ** 2).sum(dim=1)
    assert ss.dtype == torch.float32 and tuple(ss.shape) == (2,)
    assert torch.allclose(ss.double(), exact, rtol=1e-6, atol=0.0)
    assert torch.equal(ref.sum_of_squares(z), ss)  # deterministic
    # Every lane's chain covers its 192 elements once: a lane-local permutation of z keeps the row sum bitwise.
    chains = (
        z.reshape(2, 6, 32, 4, 4, 2).permute(0, 2, 1, 3, 4, 5).reshape(2, 32, 96, 2)
    )
    assert chains.numel() == z.numel()
    assert torch.equal(ref.sum_of_squares(z * 1.0), ss)
    # The permuted order sums the same squares in the lane's loaded chunk order: as close to the exact sum,
    # deterministic, and bitwise equal to the natural order when no partial sum rounds.
    permuted = ref.sum_of_squares(z, order="perm")
    assert torch.allclose(permuted.double(), exact, rtol=1e-6, atol=0.0)
    assert torch.equal(ref.sum_of_squares(z, order="perm"), permuted)
    small = torch.randint(
        -8, 9, (2, ref.HIDDEN), generator=torch.Generator().manual_seed(4)
    ).float()
    assert torch.equal(
        ref.sum_of_squares(small), ref.sum_of_squares(small, order="perm")
    )
    # The orders differ where a rounding depends on the chunk order: lane 2 (f = 1) sums chunk 1 before
    # chunk 0 in the permuted order, so its four ones survive beside 2^24 in the even chain (the natural
    # order loses them: 2^24 + 1 rounds to 2^24); the odd chain adds four more in both orders.
    probe = torch.zeros((1, ref.HIDDEN))
    probe[0, 64] = 4096.0  # lane 2, pair 0, chunk 0, word 0, even element
    probe[0, 72:80] = 1.0  # lane 2, pair 0, chunk 1
    assert ref.sum_of_squares(probe).item() == 2.0**24 + 4
    assert ref.sum_of_squares(probe, order="perm").item() == 2.0**24 + 8
    # Lane 0 (f = 0) loads the chunks in the row's order: both orders agree on the same pattern there.
    probe0 = torch.zeros((1, ref.HIDDEN))
    probe0[0, 0] = 4096.0
    probe0[0, 8:16] = 1.0
    assert ref.sum_of_squares(probe0).item() == 2.0**24 + 4
    assert ref.sum_of_squares(probe0, order="perm").item() == 2.0**24 + 4
    with pytest.raises(ValueError):
        ref.sum_of_squares(z, order="reversed")


@pytest.mark.cpu_only
def test_nvfp4_quantiser_hand_case_and_tie_mask() -> None:
    rows = 1
    y = torch.zeros((rows, ref.HIDDEN), dtype=torch.bfloat16)
    y[0, 0] = (
        6.0  # block 0: vmax 6 -> sf = e4m3(1 * 6 * rcp(6)) = 1.0, output scale 1: the products are y itself
    )
    y[0, 1] = 0.25  # an E2M1 rounding midpoint
    y[0, 2] = 0.5
    y[0, 3] = -0.1  # rounds to -0: the quantiser keeps code 8
    y[0, 16] = 3.0  # block 1: vmax 3 -> sf = e4m3(0.5) = 0.5, scale 2: 3 * 2 = 6
    gs = torch.ones((1,), dtype=torch.float32)
    products, sf = ref.nvfp4_products(y, gs)
    assert sf[0, 0].item() == 0x38 and sf[0, 1].item() == 0x30 and sf[0, 2].item() == 0
    assert products[0, 0, :4].tolist() == [6.0, 0.25, 0.5, y[0, 3].float().item()]
    assert products[0, 1, 0].item() == 6.0
    values, scales = ref.quant_nvfp4(y, gs)
    codes = ref.unpack_e2m1(values)
    assert codes[0, :4].tolist() == [7, 0, 1, 8]
    assert codes[0, 16].item() == 7
    assert int(codes[0, 17:].sum()) == 0
    assert scales.shape == (ref.scale_bytes(rows, ref.NVFP4_SCALES_PER_ROW),)
    block_order = ref.unswizzle_scales(scales, rows, ref.NVFP4_SCALES_PER_ROW)
    assert block_order[0, 0].item() == 0x38 and block_order[0, 1].item() == 0x30
    mask = ref.nvfp4_tie_mask(y, gs)
    assert mask.shape == (rows, ref.VALUE_BYTES)
    assert (
        mask[0, 0].item() == 0xF0
    )  # element 1 (0.25) is a midpoint, element 0 (6.0) is not
    assert mask[0, 1].item() == 0x00
    assert int(mask.sum()) == 0xF0
    # canonical clears the tie nibbles on both sides.
    assert ref.canonical(values, mask)[0, 0].item() == 0x07
    other = values.clone()
    other[0, 0] = 0x17  # the midpoint element rounded the other way
    assert torch.equal(ref.canonical(values, mask), ref.canonical(other, mask))
    assert ref.count_mismatches(values, other) == 1
    # The global scale of the quantiser: 2688 / amax(y); 1 for an all-zero tensor.
    assert ref.global_scale_out(y).item() == 2688.0 / 6.0
    assert ref.global_scale_out(torch.zeros_like(y)).item() == 1.0
    # An all-zero block has a zero scale and zero codes.
    zero = torch.zeros((1, ref.HIDDEN), dtype=torch.bfloat16)
    zero_values, zero_scales = ref.quant_nvfp4(zero, gs)
    assert int(zero_values.sum()) == 0 and int(zero_scales.sum()) == 0


@pytest.mark.cpu_only
def test_ue8m0_round_up_and_mxfp8_hand_case() -> None:
    x = torch.tensor(
        [0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 2.0**-130, 2.0**100], dtype=torch.float32
    )
    assert ref.ue8m0_rp(x).tolist() == [0, 126, 127, 127, 128, 128, 0, 227]
    rows = 1
    y = torch.zeros((rows, ref.HIDDEN), dtype=torch.bfloat16)
    y[0, 0] = (
        448.0  # block 0: vmax 448 -> e = ue8m0(1.0) = 127, scale 1: the values are e4m3(y)
    )
    y[0, 1] = 1.0
    y[0, 2] = -0.1
    y[0, 32] = 896.0  # block 1: vmax 896 -> e = 128, scale 2: 896 / 2 = 448
    y[0, 33] = 3.0
    values, scales = ref.quant_mxfp8(y)
    assert values.shape == (rows, ref.MXFP8_VALUE_BYTES) and values.dtype == torch.uint8
    assert values[0, 0].item() == 0x7E and values[0, 1].item() == 0x38
    assert ref.e4m3_to_f32(values[0, 2:3]).item() == pytest.approx(-0.1, rel=0.07)
    assert values[0, 32].item() == 0x7E
    assert ref.e4m3_to_f32(values[0, 33:34]).item() == 1.5
    assert int(values[0, 3:32].sum()) == 0 and int(values[0, 34:].sum()) == 0
    block_order = ref.unswizzle_scales(scales, rows, ref.MXFP8_SCALES_PER_ROW)
    assert block_order[0, 0].item() == 127 and block_order[0, 1].item() == 128
    assert int(block_order[0, 2:].sum()) == 0
    assert scales.shape == (ref.scale_bytes(rows, ref.MXFP8_SCALES_PER_ROW),)


@pytest.mark.cpu_only
def test_comparison_helpers() -> None:
    a = torch.tensor([1.0, -1.0, 0.0, 2.0], dtype=torch.bfloat16)
    bits = a.view(torch.int16).to(torch.int32)
    neighbour = (bits + torch.where(a < 0, -1, 1)).to(torch.int16).view(torch.bfloat16)
    assert ref.bf16_ulp_distance(a, a).tolist() == [0, 0, 0, 0]
    assert ref.bf16_ulp_distance(a, neighbour).tolist() == [1, 1, 1, 1]
    assert ref.bf16_ulp_distance(
        torch.tensor([0.0], dtype=torch.bfloat16),
        torch.tensor([-0.0], dtype=torch.bfloat16),
    ).tolist() == [0]
    assert ref.max_bf16_ulp(a, neighbour) == 1
    assert ref.count_mismatches(a, neighbour) == 4 and ref.count_mismatches(a, a) == 0
    assert ref.mismatch_fraction(a, neighbour) == 1.0
    nan = a.clone()
    nan[0] = math.nan
    assert ref.count_mismatches(nan, a) == 1 and ref.count_mismatches(nan, nan) == 1
    assert ref.tolerance_violations(a, a) == 0
    assert (
        ref.tolerance_violations(a, neighbour) == 0
    )  # one bfloat16 step is inside 1e-2
    far = a.clone()
    far[3] = 2.1
    assert ref.tolerance_violations(far, a) == 1
    assert ref.tolerance_violations(nan, a) == 1
    with pytest.raises(ValueError):
        ref.count_mismatches(a, a.float())
    with pytest.raises(ValueError):
        ref.count_mismatches(a, a[:-1])
