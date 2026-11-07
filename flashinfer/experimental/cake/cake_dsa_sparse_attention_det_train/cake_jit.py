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

from __future__ import annotations

import functools
from typing import Any

from . import cake_build

# Explicit target-owned registration of the generated training programs whose
# every output repeats bit for bit: one record per program (``PROGRAM``: the
# RoPE program, ``NOPE_PROGRAM``: the native zero-RoPE program) shared by every
# architecture it compiles for (``arches``), the kernel ``stages`` it registers
# in launch order, and one physical entry per stage: the ``library`` stem of its
# ``build.json`` entries (``<library>_<arch>``, see ``cake_build``), the FFI
# entry and the launch geometry the binding bakes in (block, cluster,
# cooperative launch, dynamic shared memory, programmatic dependent launch).
# The forward stages are the forward kernels of the non-deterministic package
# (fixed order: bitwise reproducible), compiled under program names of this
# package.  The backward stages run in the record's order, every one of them on
# every backward (``BACKWARD_STAGES``): no stage is selected by a tensor value
# or by the device's SM count.  What each library compiles -- translation
# units, nvcc flags, digests -- is recorded in ``build.json`` only; the
# positional argument order of every stage lives in the generated
# ``cake_launch`` module.  The records are populated by the generated-program
# export and only their stage names and policies are maintained by hand; the
# export rewrites the ``launch`` leaves.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_det_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": ["fwd"],
        "fwd": {
            "library": "cake_dsa_h64_det_train_c949a885132a772bdb53",
            "ffi_entry": "run",
            "launch": {
                "block": [384, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": True,
            },
        },
    },
    # The native zero-RoPE (NoPE) training program: shared latent K = V of 512
    # dimensions, no rope operands, one combined dK + dV gradient per key row.
    # ``index_prep`` (the fused index preparation: per-document rows to global
    # key rows, optionally compacted) runs ahead of ``nope_fwd`` on the packed
    # routes and serves the RoPE program's packed entry too.
    "cake_dsa_h64_det_nope_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": ["index_prep", "nope_fwd"],
        "index_prep": {
            "library": "cake_dsa_h64_det_nope_train_db791110b21a91be1388",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_fwd": {
            "library": "cake_dsa_h64_det_nope_train_46cdc056c891876ed545",
            "ffi_entry": "run",
            "launch": {
                "block": [320, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
    },
}

PROGRAM = "cake_dsa_h64_det_train"
NOPE_PROGRAM = "cake_dsa_h64_det_nope_train"
PROGRAMS = (PROGRAM, NOPE_PROGRAM)
FORWARD_STAGES = ("fwd",)
# The backward of the RoPE program, in launch order: the delta and the per-query
# bound terms, the zero fill of the integer accumulator region, the key norms,
# the per-row bounds, the per-row grids, the main stage (dQ and the integer dK /
# dV accumulation with its overflow counters), the preparation and the launch of
# the recomputation of flagged rows (always launched; it exits when no row is
# flagged), and the fold of the integers into the floating-point results.
BACKWARD_STAGES = (
    "bwd_delta",
    "bwd_zero",
    "bwd_norm",
    "bwd_bound",
    "bwd_grid",
    "bwd_main",
    "bwd_redo_prep",
    "bwd_redo",
    "bwd_fold",
)
STAGES = FORWARD_STAGES + BACKWARD_STAGES
# The NoPE program names its stages ``nope_<stage>`` (globally unique: the
# generated launch tables are keyed by stage name across both programs).
STAGE_PREFIX = {PROGRAM: "", NOPE_PROGRAM: "nope_"}
NOPE_STAGES = tuple(f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in STAGES)
NOPE_FORWARD_STAGES = NOPE_STAGES[:1]
NOPE_BACKWARD_STAGES = NOPE_STAGES[1:]
STAGES_OF = {PROGRAM: STAGES, NOPE_PROGRAM: NOPE_STAGES}
BACKWARD_STAGES_OF = {PROGRAM: BACKWARD_STAGES, NOPE_PROGRAM: NOPE_BACKWARD_STAGES}


def stage_name(stage: str, program: str = PROGRAM) -> str:
    """The name of the RoPE program's ``stage`` (``fwd``, ``bwd_main``, ...) in ``program``."""
    try:
        prefix = STAGE_PREFIX[program]
    except KeyError:
        raise ValueError(
            f"unknown program {program!r}; registered: {PROGRAMS}"
        ) from None
    return f"{prefix}{stage}"


def record(program: str = PROGRAM) -> dict[str, Any]:
    """The registered record of ``program`` (the RoPE program by default)."""
    if program not in PROGRAMS:
        raise ValueError(f"unknown program {program!r}; registered: {PROGRAMS}")
    return MODULES[program]


def registered_stages(program: str = PROGRAM) -> tuple[str, ...]:
    """Stages the record of ``program`` registers, in launch order."""
    rec = record(program)
    order = STAGES_OF[program]
    present = tuple(stage for stage in order if stage in rec)
    declared = tuple(rec.get("stages", present))
    if tuple(s for s in order if s in declared) != present:
        raise ValueError(
            f"registry record {program!r} declares stages {declared} but carries {present}"
        )
    return present


def library_name(stage: str, arch: str, program: str = PROGRAM) -> str:
    """The ``build.json`` entry of ``stage`` of ``program`` on ``arch``: ``<library>_<arch>``."""
    rec = record(program)
    if arch not in rec["arches"]:
        raise RuntimeError(
            f"the generated DSA training program {program!r} is registered for {rec['arches']}; "
            f"{arch!r} is not served by this checkout"
        )
    if stage not in rec:
        raise NotImplementedError(
            f"stage {stage!r} of the generated DSA training program {program!r} is not "
            f"registered in this checkout (registered: {registered_stages(program)})"
        )
    return f"{rec[stage]['library']}_{arch}"


@functools.cache
def load_cake_dsa_train_module(stage: str, arch: str, program: str = PROGRAM):
    """The loaded library of ``stage`` of ``program`` on ``arch``: prebuilt from ``aot/``
    when the package ships it, otherwise JIT-built by FlashInfer from ``build.json``."""
    return cake_build.load_library(library_name(stage, arch, program))
