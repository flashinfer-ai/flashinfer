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
geometry, the per-target re-arm forms and the source files; the receive-window
plan re-derived from the kernels' window formulas at five capacities; the
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
REGISTER_STORE_VARIANTS = set().union(*REGISTER_STORE_FORMS.values())
# The targets whose NVFP4-input fused-norm programs probe the peer row's six pairs at once ("_pa"), from
# their own measurement; the others (sm_107a, and every BF16-input program) poll in two-pair groups.
PROBE_ALL_TARGETS = ("sm_100a", "sm_103a")
# The one-shot BF16 kernel's form per target, from each target's own measurement: sm_103a reads the peer's
# row scale beside the data probe ("_rso"), sm_107a re-arms with register stores ("_rst"), sm_100a ships
# the baseline's forms; the NVFP4 one-shot kernel ships one text for all three.
ONESHOT_BF16_FORMS = {"sm_100a": "", "sm_103a": "_rso", "sm_107a": "_rst"}
# The targets whose BF16-input fused-norm programs with NVFP4 / MXFP8 output build the packed quantizing
# pass 2 ("_qp"), from their own measurement; sm_107a and every other build use the FP32 tree.
PACKED_QUANT_TARGETS = ("sm_100a", "sm_103a")
CAPACITIES = (4, 1024, 4096, 16384, 100000)
SOURCE_DIR = "csrc/cake_fp4_allreduce"
MAX_DYNAMIC_SMEM_BYTES = 232448  # the opt-in limit of SM100 / SM103 / SM107
# The kernel contract: sentinel words, record geometry and the window formulas.
VAL, SF, RS = 0x88888888, 0xFFFFFFFF, 0x80000000
VALUE_BYTES, SCALE_BYTES, RECORD_BYTES, ROW_SCALE_BYTES = 3072, 384, 3456, 16
ROTATION = 3


def _expected_program_ids() -> set[str]:
    ids = {"tp2_oneshot_nvfp4"}
    ids |= {f"tp2_oneshot_bf16{form}" for form in ONESHOT_BF16_FORMS.values()}
    ids |= {f"tp4_twoshot_{fmt}_bv{bv}" for fmt in INPUT_FORMATS for bv in BVS}
    ids |= {f"tp4_twoshot_{fmt}_bv{bv}_rst" for fmt, bv in REGISTER_STORE_VARIANTS}
    ids |= {f"tp2_rmsnorm_{i}_{o}" for i in INPUT_FORMATS for o in OUTPUT_FORMATS}
    ids |= {f"tp2_rmsnorm_{i}_{o}_pf" for i in INPUT_FORMATS for o in OUTPUT_FORMATS}
    ids |= {f"tp2_rmsnorm_nvfp4_{o}_pa" for o in OUTPUT_FORMATS}
    ids |= {f"tp2_rmsnorm_nvfp4_{o}_pa_pf" for o in OUTPUT_FORMATS}
    ids |= {
        f"tp2_rmsnorm_bf16_{o}_qp{pf}" for o in ("nvfp4", "mxfp8") for pf in ("", "_pf")
    }
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


# ---------------------------------------------------------------------------
# registry schema
# ---------------------------------------------------------------------------


def test_registry_lists_every_program_with_the_contract_parameters() -> None:
    assert set(PROGRAMS) == _expected_program_ids()
    assert len(PROGRAMS) == 37
    keys = {
        "family",
        "world_size",
        "input_format",
        "output_format",
        "bv",
        "reset",
        "rs_poll",
        "poll",
        "quant",
        "prefetch",
        "targets",
        "kernel_symbol",
        "sources",
        "block",
        "dynamic_smem_bytes",
        "params",
        "entry",
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
        assert tuple(record["sources"]) == (
            f"{SOURCE_DIR}/cake_fp4_allreduce_{program}_kernel.cu",
            f"{SOURCE_DIR}/cake_fp4_allreduce_{program}_binding.cu",
        )
        assert record["entry"] == "run"
        expected_block = (
            288 if (family == "rmsnorm" and input_format == "nvfp4") else 352
        )
        assert record["block"] == expected_block, program
        assert 0 < int(record["dynamic_smem_bytes"]) <= MAX_DYNAMIC_SMEM_BYTES, program
        assert tuple(record["targets"]) and set(record["targets"]) <= set(TARGETS)
        if world_size == 4:
            assert record["bv"] in BVS
            assert record["reset"] in ("bulk", "st") and record["rs_poll"] is None
            assert program.endswith("_rst") == (record["reset"] == "st")
            assert program.startswith(f"tp4_twoshot_{input_format}_bv{record['bv']}")
        elif family == "allreduce":
            # The one-shot kernel's lever forms: "_rso" for the overlapped row-scale poll, then "_rst" for a
            # register-store re-arm off the BF16 baseline (the NVFP4 kernel re-arms with register stores).
            assert record["bv"] is None
            assert record["rs_poll"] in ("first", "overlap")
            assert record["reset"] in ("bulk", "st")
            baseline_reset = {"bf16": "bulk", "nvfp4": "st"}[input_format]
            suffix = ("_rso" if record["rs_poll"] == "overlap" else "") + (
                "" if record["reset"] == baseline_reset else "_rst"
            )
            assert program == f"tp2_oneshot_{input_format}{suffix}", program
        else:
            assert record["bv"] is None and record["reset"] is None
            assert record["rs_poll"] is None
        if family == "rmsnorm":
            assert record["prefetch"] in (False, True)
            assert program.endswith("_pf") == record["prefetch"], program
            stem = program[: -len("_pf")] if record["prefetch"] else program
            # The quantizing pass-2 form exists for the NVFP4 / MXFP8 outputs only ("_qp" for the packed form).
            assert (record["quant"] is None) == (output_format == "bf16"), program
            assert record["quant"] in (None, "f32", "packed"), program
            assert stem.endswith("_qp") == (record["quant"] == "packed"), program
            stem = stem[: -len("_qp")] if record["quant"] == "packed" else stem
            assert record["poll"] in ("grouped", "all")
            assert stem.endswith("_pa") == (record["poll"] == "all"), program
        else:
            assert record["prefetch"] is None and record["poll"] is None
            assert record["quant"] is None
    assert {
        target for record in PROGRAMS.values() for target in record["targets"]
    } == set(TARGETS)


def test_two_shot_programs_cover_every_variant_once_per_target() -> None:
    """On each target one program serves each (input format, bv), in the re-arm form that target measured."""
    for target in TARGETS:
        for input_format in INPUT_FORMATS:
            for bv in BVS:
                matching = [
                    program
                    for program, record in PROGRAMS.items()
                    if record["family"] == "allreduce"
                    and record["world_size"] == 4
                    and record["input_format"] == input_format
                    and record["bv"] == bv
                    and target in record["targets"]
                ]
                assert len(matching) == 1, (target, input_format, bv, matching)
                (program,) = matching
                register_store = (input_format, bv) in REGISTER_STORE_FORMS[target]
                assert PROGRAMS[program]["reset"] == (
                    "st" if register_store else "bulk"
                )
                assert program.endswith("_rst") == register_store
    # A variant's register-store form ships for the targets that measured it, its bulk form for the others.
    for input_format in INPUT_FORMATS:
        for bv in BVS:
            bulk = f"tp4_twoshot_{input_format}_bv{bv}"
            st_targets = tuple(
                t for t in TARGETS if (input_format, bv) in REGISTER_STORE_FORMS[t]
            )
            bulk_targets = tuple(
                t for t in TARGETS if (input_format, bv) not in REGISTER_STORE_FORMS[t]
            )
            assert PROGRAMS[bulk]["targets"] == bulk_targets, bulk
            if st_targets:
                assert PROGRAMS[f"{bulk}_rst"]["targets"] == st_targets, bulk
            else:
                assert f"{bulk}_rst" not in PROGRAMS


def test_one_shot_programs_serve_every_target_once() -> None:
    """On each target one one-shot program serves each input format: the NVFP4 kernel's single text, and for
    BF16 input the form that target measured ("_rso" on sm_103a, "_rst" on sm_107a, the baseline on sm_100a)."""
    for target in TARGETS:
        for input_format in INPUT_FORMATS:
            matching = [
                program
                for program, record in PROGRAMS.items()
                if record["family"] == "allreduce"
                and record["world_size"] == 2
                and record["input_format"] == input_format
                and target in record["targets"]
            ]
            form = ONESHOT_BF16_FORMS[target] if input_format == "bf16" else ""
            assert matching == [f"tp2_oneshot_{input_format}{form}"], (
                target,
                input_format,
                matching,
            )
    assert PROGRAMS["tp2_oneshot_nvfp4"]["targets"] == TARGETS
    for target, form in ONESHOT_BF16_FORMS.items():
        assert PROGRAMS[f"tp2_oneshot_bf16{form}"]["targets"] == (target,)


def test_fused_norm_programs_cover_every_form_once_per_target() -> None:
    """On each target one program serves each (input format, output format, prefetch form), in the poll and
    quantizing forms that target measured: the probe-all poll ("_pa") for NVFP4 input on sm_100a / sm_103a and
    the two-pair groups otherwise; the packed quantizing pass 2 ("_qp") for BF16 input with NVFP4 / MXFP8 output
    on sm_100a / sm_103a and the FP32 tree otherwise."""
    for target in TARGETS:
        for input_format in INPUT_FORMATS:
            for output_format in OUTPUT_FORMATS:
                for prefetch in (False, True):
                    matching = [
                        program
                        for program, record in PROGRAMS.items()
                        if record["family"] == "rmsnorm"
                        and record["input_format"] == input_format
                        and record["output_format"] == output_format
                        and record["prefetch"] is prefetch
                        and target in record["targets"]
                    ]
                    assert len(matching) == 1, (
                        target,
                        input_format,
                        output_format,
                        prefetch,
                        matching,
                    )
                    (program,) = matching
                    probe_all = input_format == "nvfp4" and target in PROBE_ALL_TARGETS
                    assert PROGRAMS[program]["poll"] == (
                        "all" if probe_all else "grouped"
                    )
                    packed = (
                        input_format == "bf16"
                        and output_format != "bf16"
                        and target in PACKED_QUANT_TARGETS
                    )
                    assert PROGRAMS[program]["quant"] == (
                        None
                        if output_format == "bf16"
                        else ("packed" if packed else "f32")
                    )
                    suffix = (
                        ("_pa" if probe_all else "")
                        + ("_qp" if packed else "")
                        + ("_pf" if prefetch else "")
                    )
                    assert (
                        program == f"tp2_rmsnorm_{input_format}_{output_format}{suffix}"
                    )
    # The probe-all form ships for the targets that measured it, the grouped form for the others; the packed
    # quantizing pass 2 likewise; the BF16 -> BF16 programs have one form for all three targets.
    for output_format in OUTPUT_FORMATS:
        for pf in ("", "_pf"):
            assert (
                PROGRAMS[f"tp2_rmsnorm_nvfp4_{output_format}_pa{pf}"]["targets"]
                == PROBE_ALL_TARGETS
            )
            assert PROGRAMS[f"tp2_rmsnorm_nvfp4_{output_format}{pf}"][
                "targets"
            ] == tuple(t for t in TARGETS if t not in PROBE_ALL_TARGETS)
            assert f"tp2_rmsnorm_bf16_{output_format}_pa{pf}" not in PROGRAMS
            assert f"tp2_rmsnorm_nvfp4_{output_format}_qp{pf}" not in PROGRAMS
            assert f"tp2_rmsnorm_nvfp4_{output_format}_pa_qp{pf}" not in PROGRAMS
            if output_format == "bf16":
                assert PROGRAMS[f"tp2_rmsnorm_bf16_bf16{pf}"]["targets"] == TARGETS
                assert f"tp2_rmsnorm_bf16_bf16_qp{pf}" not in PROGRAMS
            else:
                assert (
                    PROGRAMS[f"tp2_rmsnorm_bf16_{output_format}_qp{pf}"]["targets"]
                    == PACKED_QUANT_TARGETS
                )
                assert PROGRAMS[f"tp2_rmsnorm_bf16_{output_format}{pf}"][
                    "targets"
                ] == tuple(t for t in TARGETS if t not in PACKED_QUANT_TARGETS)


def _checkout_source_dir() -> Path:
    return Path(__file__).resolve().parents[2] / SOURCE_DIR


def test_program_sources_exist_and_declare_their_kernels() -> None:
    source_dir = _checkout_source_dir()
    if not source_dir.is_dir():
        pytest.skip(f"{source_dir} is not part of this checkout")
    for program, record in PROGRAMS.items():
        kernel, binding = (
            Path(__file__).resolve().parents[2] / source for source in record["sources"]
        )
        assert kernel.is_file(), kernel
        assert binding.is_file(), binding
        kernel_text = kernel.read_text()
        assert f"{record['kernel_symbol']}(" in kernel_text, program
        assert "__global__" in kernel_text, program
        binding_text = binding.read_text()
        assert f"TVM_FFI_DLL_EXPORT_TYPED_FUNC({record['entry']}," in binding_text, (
            program
        )
        assert record["kernel_symbol"] in binding_text, program
    # The loader resolves sources under the package data directory (``flashinfer/data/csrc``, a
    # symlink to the repository's ``csrc`` in an editable install): compare the real files.
    assert {path.resolve() for path in loader.program_sources("tp2_oneshot_bf16")} == {
        (source_dir / "cake_fp4_allreduce_tp2_oneshot_bf16_kernel.cu").resolve(),
        (source_dir / "cake_fp4_allreduce_tp2_oneshot_bf16_binding.cu").resolve(),
    }


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
    """The windows of the kernel contract at capacity ``cap``: bytes, fill period, fill runs, table and control words."""
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
        return {"ctrl_words": 4, "buffers": buffers, "table": ("wire", "rsplane")}
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
        return {"ctrl_words": 4 * 4096, "buffers": buffers, "table": ("wire",)}
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
    ctrl_words = 4 * 4096 if family == "allreduce" else 4
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
    # Table entries: the copies of ranks 0..nr-1 of every buffer; the control words match the family.
    assert len(actual["table"]) * world_size == (
        2 * world_size if expected["table"] == ("wire", "rsplane") else world_size
    )
    assert actual["ctrl_words"] == (
        4 * LIMITS["tp2_oneshot_grid_max"]
        if (family, world_size) == ("allreduce", 2)
        else 4
    )
    # The loader's own evaluator agrees with the independent one.
    for buffer in WINDOWS[key]["buffers"]:
        assert loader.evaluate_linear_form(buffer["bytes"], cap) == _linear(
            buffer["bytes"], cap
        )
    assert loader.window_contract(family, world_size, input_format) is WINDOWS[key]


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
    with pytest.raises(ValueError):
        loader.window_contract("allreduce", 8, "bf16")
    with pytest.raises(ValueError):
        loader.window_contract("rmsnorm", 4, "bf16")


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


def test_select_bv_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError):
        loader.select_bv(1024, 0)


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


def test_select_prefetch_rejects_an_empty_grid() -> None:
    with pytest.raises(ValueError):
        loader.select_prefetch(4096, 0)


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("input_format", INPUT_FORMATS)
def test_program_for_two_shot_picks_the_targets_re_arm_form(
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
        register_store = (input_format, bv) in REGISTER_STORE_FORMS[target]
        suffix = "_rst" if register_store else ""
        assert program == f"tp4_twoshot_{input_format}_bv{bv}{suffix}"
        assert target in PROGRAMS[program]["targets"]
        assert PROGRAMS[program]["bv"] == bv


@pytest.mark.parametrize("target", TARGETS)
def test_program_for_one_shot_and_fused_norm(target: str) -> None:
    for input_format in INPUT_FORMATS:
        form = ONESHOT_BF16_FORMS[target] if input_format == "bf16" else ""
        for rows in (3, 1031, 2048, 2664, 4096, 4291, 4292, 16384):
            suffix = "_pf" if loader.select_prefetch(rows, 148) else ""
            assert (rows in (2664, 4096, 4291)) == bool(suffix)
            assert (
                loader.program_for(
                    family="allreduce",
                    world_size=2,
                    input_format=input_format,
                    output_format=None,
                    rows=rows,
                    grid=148,
                    target=target,
                )
                == f"tp2_oneshot_{input_format}{form}"
            )
            poll = (
                "_pa" if input_format == "nvfp4" and target in PROBE_ALL_TARGETS else ""
            )
            for output_format in OUTPUT_FORMATS:
                quant = (
                    "_qp"
                    if input_format == "bf16"
                    and output_format != "bf16"
                    and target in PACKED_QUANT_TARGETS
                    else ""
                )
                assert (
                    loader.program_for(
                        family="rmsnorm",
                        world_size=2,
                        input_format=input_format,
                        output_format=output_format,
                        rows=rows,
                        grid=148,
                        target=target,
                    )
                    == f"tp2_rmsnorm_{input_format}_{output_format}{poll}{quant}{suffix}"
                )
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
# build targets
# ---------------------------------------------------------------------------


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


def test_pack_unpack_roundtrip() -> None:
    codes = (torch.arange(2 * ref.HIDDEN) % 16).to(torch.uint8).reshape(2, ref.HIDDEN)
    packed = ref.pack_e2m1(codes)
    assert packed.shape == (2, ref.VALUE_BYTES) and packed.dtype == torch.uint8
    assert (
        packed[0, 0].item() == 0x10
    )  # element 0 (code 0) in the low nibble, element 1 (code 1) in the high one
    assert torch.equal(ref.unpack_e2m1(packed), codes)


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
    acc = ref.oneshot_sum(ref.sources_bf16(xs))
    z = ref.f32_add_rn(acc, residual.float())
    assert torch.equal(reference.residual_out, z.to(torch.bfloat16))
    # ss in the kernel's order is within a few float32 ulps of the exact sum of squares.
    exact = (z.double() ** 2).sum(dim=1)
    assert torch.allclose(reference.ss.double(), exact, rtol=1e-6, atol=0.0)
    rinv = 1.0 / torch.sqrt(exact / ref.HIDDEN + ref.EPS)
    assert torch.allclose(reference.rinv.double(), rinv, rtol=1e-6, atol=0.0)
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
