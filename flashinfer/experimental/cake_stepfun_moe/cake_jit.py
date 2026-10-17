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

# Registry of the generated StepFun fused-MoE programs (filled verbatim by the
# Cake generated-program export ``exports/stepfun_moe``; do not edit the
# literals by hand).
#
# ``PROGRAMS``: one record per physical program -- one kernel source plus its
# host binding, shared by both architectures in ``arches`` -- with its
# architecture-free ``library`` stem, FFI entry, argument plan, launch shape
# (block, cluster) and the ``unit`` block: the stage metadata the host plan
# reads (stage, family, tile, block_k, routing window, split-K geometry,
# block-scale layout, the storage dtype the binding checks for every tensor
# argument, the rank of every tensor-map view).  What to compile (the
# translation units, the per-architecture closure digest) is recorded in
# ``build.json`` (see ``cake_build``), whose entry for a program is
# ``<library>_<arch>``; a program is built once per architecture, prebuilt
# under ``aot/`` or by FlashInfer's JIT.
#
# ``STAGE_TABLES``: the five dispatcher tables of the host (routing rows, FC1,
# requantization, FC2, finalize) as ordered lists of program names in the
# production dispatch order; every row fact lives once, in
# ``PROGRAMS[name]["unit"]``.
#
# ``ROUTES``: one record per exported route (standalone stage routes and the
# full-path routes per family and tile) naming its architectures and ordered
# stages; the manifest checker walks it to prove every program reachable.
#
# ``HOST_PLAN_CONSTANTS``: the geometry, family table and routing /
# requantization / finalize constants the host plan of ``cake_backend``
# reproduces from the production launchers.
PROGRAMS: dict[str, dict[str, Any]] = {}
STAGE_TABLES: dict[str, list[str]] = {}
ROUTES: dict[str, dict[str, Any]] = {}
HOST_PLAN_CONSTANTS: dict[str, Any] = {}

ARCHES = ("sm_100a", "sm_103a")
STAGE_ORDER = ("routing", "fc1", "requant", "fc2", "finalize")


def library_name(program: str, arch: str) -> str:
    """``build.json`` entry of ``program`` built for ``arch``: ``<library>_<arch>``."""
    record = PROGRAMS[program]
    if arch not in record["arches"]:
        raise ValueError(f"program {program!r} is not delivered for {arch}")
    return f"{record['library']}_{arch}"


@functools.cache
def _load_module(program: str, arch: str):
    return cake_build.load_library(library_name(program, arch))


def load_module(program: str, arch: str):
    """Load (once per program and exact target) one physical program: the prebuilt
    library under ``aot/`` when the package ships it, else FlashInfer's JIT build of
    its ``build.json`` entry."""
    return _load_module(program, arch)


def specs_for_arch(arch: str):
    """JIT specs of every ``build.json`` entry of ``arch`` without a prebuilt library (the
    ``flashinfer.jit.build_jit_specs`` warm-up)."""
    return cake_build.jit_specs(arch)


def programs_of(stage: str) -> list[str]:
    """The ordered program names of the dispatcher table ``stage`` (a ``STAGE_ORDER`` name)."""
    if stage not in STAGE_ORDER:
        raise KeyError(f"unknown stage {stage!r}; expected one of {STAGE_ORDER}")
    return STAGE_TABLES[stage]


def unit(program: str) -> dict[str, Any]:
    """The stage metadata block of ``program``."""
    return PROGRAMS[program]["unit"]
