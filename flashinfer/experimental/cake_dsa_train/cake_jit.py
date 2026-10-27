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

# Explicit target-owned registration of the generated training programs: one
# record per program (``PROGRAM``: the RoPE family, ``NOPE_PROGRAM``: the
# native zero-RoPE family) shared by every architecture it compiles for
# (``arches``), the kernel ``stages`` it registers in launch order, the host
# policies that select the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and the direct
# accumulation into the caller's packed rows through the natural-layout main
# stages (``dkv_direct``, see ``cake_backend.DkvDirectPolicy``), and one physical
# entry per stage: the ``library`` stem of its ``build.json`` entries
# (``<library>_<arch>``, see ``cake_build``), the FFI entry and the launch
# geometry the binding bakes in (block, cluster, cooperative launch, dynamic
# shared memory, programmatic dependent launch).  Two records may name one
# library (the rope-independent delta, compaction and cast stages serve both
# programs); their launch dicts then agree.  What each library compiles --
# translation units, nvcc flags, digests -- is recorded in ``build.json`` only;
# the positional argument order of every stage lives in the generated
# ``cake_launch`` module.  The records are populated by the generated-program
# export and only their stage names, policies and the pending stages of a
# record are maintained by hand; the export rewrites the ``launch`` leaves.
MODULES: dict[str, dict[str, Any]] = {
    "cake_dsa_h64_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "fwd",
            "bwd_delta",
            "bwd_main",
            "bwd_main_natural",
            "bwd_compact",
            "bwd_main_pass",
            "bwd_main_pass_natural",
            "bwd_cast",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2304,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 4,
            "rules": {
                "sm_100a": {
                    "tail_rows": 0,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 0,
                    "tail_cap": 0,
                },
                "sm_103a": {
                    "tail_rows": 1,
                    "fixed_passes": 0,
                    "min_kv": 0,
                    "tail_min_kv": 131072,
                    "tail_cap": 3,
                },
                "sm_107a": {
                    "tail_rows": 1,
                    "fixed_passes": 2,
                    "min_kv": 131072,
                    "tail_min_kv": 131072,
                    "tail_cap": 0,
                },
            },
        },
        "dkv_direct": {"min_keys_per_query": 4},
        "fwd": {
            "library": "cake_dsa_h64_train_c949a885132a772bdb53",
            "ffi_entry": "run",
            "launch": {
                "block": [384, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": True,
            },
        },
        "bwd_delta": {
            "library": "cake_dsa_h64_nope_train_ae9db52dff326ae8409b",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_main": {
            "library": "cake_dsa_h64_train_e29d165e3abca62c430b",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_main_natural": {
            "library": "cake_dsa_h64_train_e29d165e3abca62c430b_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_compact": {
            "library": "cake_dsa_h64_train_19c560118a407f5473fe",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "bwd_main_pass": {
            "library": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_main_pass_natural": {
            "library": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 231680,
                "use_pdl": False,
            },
        },
        "bwd_cast": {
            "library": "cake_dsa_h64_train_c96be96662a6d97e0b53",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
    },
    # The native zero-RoPE (NoPE) training program: shared latent K = V of 512
    # dimensions, no rope operands, one combined FP32 dK + dV gradient of 2048 B
    # per key.  Its key-range-pass policy is re-calibrated for that row size
    # (``key_bytes`` 2048; ``dq_partial_bytes_per_token`` = 64 heads x 512 x 4 B
    # per token of FP32 dQ partial).  Measured on the acceptance GPUs, one key
    # pass is the fastest plan on every row and build (``max_passes`` 1: the
    # policy never plans a second pass; ``key_passes`` still forces the pass
    # form, whose stages and scratch stay registered) and the natural-layout
    # main stage beats the permuted accumulator plus cast whenever the row has
    # at least one key per query (``min_keys_per_query`` 1).  The compaction and
    # cast stages are rope-independent kernels shared with the RoPE program
    # (same libraries, same launch geometry; the cast runs over the one 512-wide
    # accumulator, 16 groups per key); the delta stage is the strided delta
    # kernel, shared with the RoPE record like compact and cast (both records'
    # ``bwd_delta`` / ``nope_bwd_delta`` name one library; 256 threads: it
    # reads ``dout`` through its token and head strides, so a head-major
    # gradient view is consumed in place by both programs);
    # the forward (320 threads, 222,464 B dynamic shared memory) and the four
    # main backward forms (640 threads, 214,208 B; permuted / natural-layout
    # accumulation, single-pass / key-range-pass) are the NoPE kernels.  Stage
    # order = launch order.  ``build.json`` carries every library named here
    # once the export delivers it; ``scripts/check_cake_build_manifests.py``
    # reports a registry library the manifest lacks.
    "cake_dsa_h64_nope_train": {
        "arches": ["sm_100a", "sm_103a", "sm_107a"],
        "stages": [
            "index_prep",
            "nope_fwd",
            "nope_bwd_delta",
            "nope_bwd_main",
            "nope_bwd_main_natural",
            "nope_bwd_compact",
            "nope_bwd_main_pass",
            "nope_bwd_main_pass_natural",
            "nope_bwd_cast",
        ],
        "key_pass_policy": {
            "l2_budget_bytes": 104857600,
            "key_bytes": 2048,
            "workspace_budget_bytes": 671088640,
            "token_chunk_multiple": 128,
            "max_passes": 1,
            "dq_partial_bytes_per_token": 131072,
        },
        "dkv_direct": {"min_keys_per_query": 1},
        "index_prep": {
            "library": "cake_dsa_h64_nope_train_db791110b21a91be1388",
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
            "library": "cake_dsa_h64_nope_train_46cdc056c891876ed545",
            "ffi_entry": "run",
            "launch": {
                "block": [320, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 223488,
                "use_pdl": True,
            },
        },
        "nope_bwd_delta": {
            "library": "cake_dsa_h64_nope_train_ae9db52dff326ae8409b",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_main": {
            "library": "cake_dsa_h64_nope_train_d7b52eccaf718555e88f",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_natural": {
            "library": "cake_dsa_h64_nope_train_d7b52eccaf718555e88f_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_compact": {
            "library": "cake_dsa_h64_train_19c560118a407f5473fe",
            "ffi_entry": "run",
            "launch": {
                "block": [128, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_pass": {
            "library": "cake_dsa_h64_nope_train_90ab03044736858291f5",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_main_pass_natural": {
            "library": "cake_dsa_h64_nope_train_90ab03044736858291f5_dkv_natural1",
            "ffi_entry": "run",
            "launch": {
                "block": [640, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 215296,
                "use_pdl": False,
            },
        },
        "nope_bwd_cast": {
            "library": "cake_dsa_h64_train_c96be96662a6d97e0b53",
            "ffi_entry": "run",
            "launch": {
                "block": [256, 1, 1],
                "cluster": [1, 1, 1],
                "cooperative": False,
                "dynamic_smem_bytes": 0,
                "use_pdl": False,
            },
        },
    },
}

PROGRAM = "cake_dsa_h64_train"
NOPE_PROGRAM = "cake_dsa_h64_nope_train"
PROGRAMS = (PROGRAM, NOPE_PROGRAM)
STAGES = (
    "fwd",
    "bwd_delta",
    "bwd_main",
    "bwd_main_natural",
    "bwd_compact",
    "bwd_main_pass",
    "bwd_main_pass_natural",
    "bwd_cast",
)
FORWARD_STAGES = ("fwd",)
# The NoPE program names its stages ``nope_<stage>`` (globally unique: the
# generated launch tables are keyed by stage name across both programs).
STAGE_PREFIX = {PROGRAM: "", NOPE_PROGRAM: "nope_"}
NOPE_STAGES = tuple(f"{STAGE_PREFIX[NOPE_PROGRAM]}{stage}" for stage in STAGES)
NOPE_FORWARD_STAGES = NOPE_STAGES[:1]
STAGES_OF = {PROGRAM: STAGES, NOPE_PROGRAM: NOPE_STAGES}


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
