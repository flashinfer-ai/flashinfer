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

# Explicit target-owned registration of the generated training program: one
# record (``PROGRAM``) shared by every architecture it compiles for
# (``arches``), the kernel ``stages`` it registers in launch order, the host
# policies that select the key-range-pass form of the backward
# (``key_pass_policy``, see ``cake_backend.KeyPassPolicy``) and the direct
# accumulation into the caller's packed rows through the natural-layout main
# stages (``dkv_direct``, see ``cake_backend.DkvDirectPolicy``), and one physical
# entry per stage: the ``library`` stem of its ``build.json`` entries
# (``<library>_<arch>``, see ``cake_build``), the FFI entry and the launch block /
# cluster the binding bakes in.  What each library compiles -- translation units,
# nvcc flags, digests -- is recorded in ``build.json`` only; the positional
# argument order of every stage lives in the generated ``cake_launch`` module.
# Populated verbatim by the generated-program export; do not edit by hand.
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
            "module": "cake_dsa_h64_train_c949a885132a772bdb53",
            "library": "cake_dsa_h64_train_c949a885132a772bdb53",
            "ffi_entry": "run",
            "launch": {"block": [384, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_delta": {
            "module": "cake_dsa_h64_train_da16f2d6e1b0fa6a05fd",
            "library": "cake_dsa_h64_train_da16f2d6e1b0fa6a05fd",
            "ffi_entry": "run",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main": {
            "module": "cake_dsa_h64_train_e29d165e3abca62c430b",
            "library": "cake_dsa_h64_train_e29d165e3abca62c430b",
            "ffi_entry": "run",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_natural": {
            "module": "cake_dsa_h64_train_726ceb6d3890375e922d",
            "library": "cake_dsa_h64_train_726ceb6d3890375e922d",
            "ffi_entry": "run",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_compact": {
            "module": "cake_dsa_h64_train_19c560118a407f5473fe",
            "library": "cake_dsa_h64_train_19c560118a407f5473fe",
            "ffi_entry": "run",
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass": {
            "module": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667",
            "library": "cake_dsa_h64_train_d6a7bc0b3aa5b7084667",
            "ffi_entry": "run",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_main_pass_natural": {
            "module": "cake_dsa_h64_train_76061910b68bb0506c5e",
            "library": "cake_dsa_h64_train_76061910b68bb0506c5e",
            "ffi_entry": "run",
            "launch": {"block": [640, 1, 1], "cluster": [1, 1, 1]},
        },
        "bwd_cast": {
            "module": "cake_dsa_h64_train_c96be96662a6d97e0b53",
            "library": "cake_dsa_h64_train_c96be96662a6d97e0b53",
            "ffi_entry": "run",
            "launch": {"block": [256, 1, 1], "cluster": [1, 1, 1]},
        },
    },
}

PROGRAM = "cake_dsa_h64_train"
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
BACKWARD_STAGES = STAGES[1:]


def record() -> dict[str, Any]:
    """The registered program record."""
    try:
        return MODULES[PROGRAM]
    except KeyError:
        raise NotImplementedError(
            "The generated DSA sparse-attention training program is not registered "
            "in this checkout (see flashinfer-ai/flashinfer#5657)"
        ) from None


def registered_stages() -> tuple[str, ...]:
    """Stages the record registers, in launch order."""
    rec = record()
    present = tuple(stage for stage in STAGES if stage in rec)
    declared = tuple(rec.get("stages", present))
    if tuple(s for s in STAGES if s in declared) != present:
        raise ValueError(
            f"registry record declares stages {declared} but carries {present}"
        )
    return present


def library_name(stage: str, arch: str) -> str:
    """The ``build.json`` entry of ``stage`` on ``arch``: ``<library>_<arch>``."""
    rec = record()
    if arch not in rec["arches"]:
        raise RuntimeError(
            f"the generated DSA training program is registered for {rec['arches']}; "
            f"{arch!r} is not served by this checkout"
        )
    return f"{rec[stage]['library']}_{arch}"


@functools.cache
def load_cake_dsa_train_module(stage: str, arch: str):
    """The loaded library of ``stage`` on ``arch``: prebuilt from ``aot/`` when the
    package ships it, otherwise JIT-built by FlashInfer from ``build.json``."""
    return cake_build.load_library(library_name(stage, arch))


def specs_for_arch(arch: str) -> list[Any]:
    """JIT specs of every ``arch`` library without a prebuilt file, for
    ``flashinfer.jit.build_jit_specs``."""
    return cake_build.jit_specs(arch)
