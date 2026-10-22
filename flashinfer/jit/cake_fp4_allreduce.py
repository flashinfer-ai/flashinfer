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

JIT loader of the Cake NVFP4-transport tensor-parallel all-reduce (SM100 / SM103 / SM107).

The programs are generated into ``csrc/cake_fp4_allreduce`` and listed in
:mod:`flashinfer.jit.cake_fp4_allreduce_programs` (``PROGRAMS``: one physical
program per family, world size, input format, output format, phase-B variant,
re-arm form, poll form and residual-prefetch form; ``WINDOWS``: the receive-window
contract; ``SELECT_BV``, ``SELECT_PREFETCH`` and ``LIMITS``: the host rules).  Every program is an exact-architecture source:
it is compiled once per exact target (``sm_100a`` for B200, ``sm_103a`` for
B300, ``sm_107a`` for SM107 devices) with that target's flag set, without fast math and with
``--fmad=false``, into a module named after the program, the target and a
digest of its sources.  Build targets follow ``FLASHINFER_CUDA_ARCH_LIST``
when set, else the visible devices (:class:`CompilationContext`); a device
whose capability is not a build target is rejected by :func:`target_arch`.
"""

from __future__ import annotations

import functools
import hashlib
import re
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional

import torch

from ..compilation_context import CompilationContext
from . import env as jit_env
from .cake_fp4_allreduce_programs import (
    LIMITS,
    PROGRAMS,
    SELECT_BV,
    SELECT_PREFETCH,
    WINDOWS,
)
from .core import (
    JitSpec,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
    sm107a_nvcc_flags,
)

CakeFp4AllReduceTarget = Literal["sm_100a", "sm_103a", "sm_107a"]

SOURCE_PACKAGE = "cake_fp4_allreduce"
MODULE_PREFIX = "cake_fp4_allreduce"
# Exact-architecture flag sets: one module per (program, target).
TARGET_FLAGS: dict[str, list[str]] = {
    "sm_100a": sm100a_nvcc_flags,
    "sm_103a": sm103a_nvcc_flags,
    "sm_107a": sm107a_nvcc_flags,
}
TARGET_CAPABILITIES: dict[str, tuple[int, int]] = {
    "sm_100a": (10, 0),
    "sm_103a": (10, 3),
    "sm_107a": (10, 7),
}
SUPPORTED_CAPABILITIES: tuple[tuple[int, int], ...] = tuple(
    TARGET_CAPABILITIES.values()
)
# The source build's numerics: no fast math, fused multiply-add disabled.
COMPILE_FLAGS: list[str] = ["--fmad=false"]

HIDDEN = int(LIMITS["hidden"])
RECORD_BYTES = int(LIMITS["record_bytes"])
FAMILIES: tuple[str, ...] = tuple(
    dict.fromkeys(record["family"] for record in PROGRAMS.values())
)
INPUT_FORMATS: tuple[str, ...] = tuple(
    dict.fromkeys(record["input_format"] for record in PROGRAMS.values())
)
OUTPUT_FORMATS: tuple[str, ...] = tuple(
    dict.fromkeys(
        record["output_format"]
        for record in PROGRAMS.values()
        if record["output_format"] is not None
    )
)
# Scalar tail of every binding entry: the pointer parameters come first, in
# kernel order, then these scalars, then ``grid_x`` and ``launch_with_pdl``.
SCALAR_PARAMS: tuple[str, ...] = ("rank", "rows", "cap", "eps", "wbias")


class DeviceFacts(NamedTuple):
    capability: tuple[int, int]
    sm_count: int


def _capability_of(major: Any, minor: Any) -> tuple[int, int]:
    """``(major, minor)`` of a ``CompilationContext.TARGET_CUDA_ARCHS`` entry (``(10, "3a")`` -> ``(10, 3)``)."""

    return int(major), int(re.sub(r"[a-z]+$", "", str(minor)))


@functools.cache
def target_capabilities() -> tuple[tuple[int, int], ...]:
    """Supported capabilities this process builds for, ascending.

    Read once from ``CompilationContext`` (``FLASHINFER_CUDA_ARCH_LIST`` when
    set, else the visible devices) and restricted to
    :data:`SUPPORTED_CAPABILITIES`.
    """

    context = CompilationContext()
    targets = {
        _capability_of(major, minor) for major, minor in context.TARGET_CUDA_ARCHS
    }
    return tuple(sorted(targets & set(SUPPORTED_CAPABILITIES)))


def supported_capability(capability: tuple[int, int]) -> Optional[tuple[int, int]]:
    """Return ``(major, minor)`` when the programs are built for it, else ``None``."""

    key = (int(capability[0]), int(capability[1]))
    return key if key in target_capabilities() else None


def target_of_capability(capability: tuple[int, int]) -> Optional[str]:
    """Exact target (``"sm_100a"`` / ``"sm_103a"`` / ``"sm_107a"``) built for ``capability``, else ``None``."""

    key = supported_capability(capability)
    if key is None:
        return None
    for target, target_capability in TARGET_CAPABILITIES.items():
        if target_capability == key:
            return target
    return None


@functools.cache
def device_facts(device_index: int) -> DeviceFacts:
    """Compute capability and SM count of one device, queried once per process."""

    properties = torch.cuda.get_device_properties(int(device_index))
    return DeviceFacts(
        capability=(int(properties.major), int(properties.minor)),
        sm_count=int(properties.multi_processor_count),
    )


@functools.cache
def target_arch(device_index: int) -> str:
    """Exact target of the device at ``device_index`` (cached per index).

    Raises ``ValueError`` when the device's capability is not one of the
    supported capabilities this process builds for.
    """

    capability = device_facts(int(device_index)).capability
    target = target_of_capability(capability)
    if target is None:
        built = [f"{major}.{minor}" for major, minor in target_capabilities()]
        supported = [f"{major}.{minor}" for major, minor in SUPPORTED_CAPABILITIES]
        raise ValueError(
            "the Cake NVFP4-transport all-reduce is not built for compute capability "
            f"{capability[0]}.{capability[1]} of cuda:{int(device_index)} (build "
            f"targets: {built}; supported: {supported}; set FLASHINFER_CUDA_ARCH_LIST "
            "to include the device's capability)"
        )
    return target


def program_record(program: str) -> dict:
    """The registry row of ``program``."""

    try:
        return PROGRAMS[program]
    except KeyError:
        raise KeyError(
            f"unknown Cake NVFP4-transport all-reduce program {program!r}; "
            f"programs: {sorted(PROGRAMS)}"
        ) from None


def programs(
    *,
    family: Optional[str] = None,
    world_size: Optional[int] = None,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    bv: Optional[int] = None,
    poll: Optional[str] = None,
    prefetch: Optional[bool] = None,
    target: Optional[str] = None,
) -> tuple[str, ...]:
    """Program ids whose registry rows match every given key, in registry order.

    ``output_format=None``, ``bv=None``, ``poll=None`` and ``prefetch=None`` do
    not filter; pass ``output_format=""`` to select the rows without an output
    format (the ``allreduce`` family).  ``target`` keeps the rows compiled for
    that target.
    """

    selected = []
    for program, record in PROGRAMS.items():
        if family is not None and record["family"] != family:
            continue
        if world_size is not None and int(record["world_size"]) != int(world_size):
            continue
        if input_format is not None and record["input_format"] != input_format:
            continue
        if output_format is not None:
            wanted = None if output_format == "" else output_format
            if record["output_format"] != wanted:
                continue
        if bv is not None and record["bv"] != int(bv):
            continue
        if poll is not None and record["poll"] != poll:
            continue
        if prefetch is not None and record["prefetch"] != bool(prefetch):
            continue
        if target is not None and target not in record["targets"]:
            continue
        selected.append(program)
    return tuple(selected)


def world_sizes(family: str) -> tuple[int, ...]:
    """World sizes the registry delivers for ``family``, ascending."""

    if family not in FAMILIES:
        raise ValueError(
            f"unknown Cake NVFP4-transport all-reduce family {family!r}; families: {FAMILIES}"
        )
    return tuple(
        sorted(
            {
                int(record["world_size"])
                for record in PROGRAMS.values()
                if record["family"] == family
            }
        )
    )


def output_formats(family: str, world_size: int, input_format: str) -> tuple[str, ...]:
    """Output formats of one (family, world size, input format): ``("",)`` when the family has none."""

    formats = tuple(
        dict.fromkeys(
            "" if record["output_format"] is None else record["output_format"]
            for record in PROGRAMS.values()
            if record["family"] == family
            and int(record["world_size"]) == int(world_size)
            and record["input_format"] == input_format
        )
    )
    if not formats:
        raise ValueError(
            "no Cake NVFP4-transport all-reduce program for family "
            f"{family!r}, world size {world_size}, input format {input_format!r}"
        )
    return formats


def select_bv(rows: int, grid: int) -> int:
    """Phase-B schedule of the world-size-4 two-shot programs for ``rows`` on ``grid`` CTAs.

    ``per_cta = rows // grid``: 2 above ``SELECT_BV["by_quad_min"]`` rows per
    CTA, 0 above ``SELECT_BV["team_max"]``, else 1.  Every rank evaluates this
    on the same grid, so every rank runs the same program.
    """

    if int(grid) <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    per_cta = int(rows) // int(grid)
    if per_cta > int(SELECT_BV["by_quad_min"]):
        return 2
    if per_cta > int(SELECT_BV["team_max"]):
        return 0
    return 1


def select_prefetch(rows: int, grid: int) -> bool:
    """Whether the fused-norm programs with the residual prefetch serve ``rows`` on ``grid`` CTAs.

    The prefetch build is selected when ``rows // grid`` lies in
    ``[SELECT_PREFETCH["rows_per_cta_min"], SELECT_PREFETCH["rows_per_cta_max"]]``
    rows per CTA; outside that window the plain build is faster.  Every rank
    evaluates this on the same grid, so every rank runs the same program.
    """

    if int(grid) <= 0:
        raise ValueError(f"grid must be positive, got {grid}")
    per_cta = int(rows) // int(grid)
    return (
        int(SELECT_PREFETCH["rows_per_cta_min"])
        <= per_cta
        <= int(SELECT_PREFETCH["rows_per_cta_max"])
    )


@functools.cache
def program_for(
    family: str,
    world_size: int,
    input_format: str,
    output_format: Optional[str],
    rows: int,
    grid: int,
    target: str,
) -> str:
    """The one program a call runs: registry rows filtered by the structured key (cached per key).

    The candidates are the rows of (family, world size, input format, output
    format) compiled for ``target``; when they carry phase-B variants the one
    of :func:`select_bv` is kept, and when they carry a residual-prefetch form
    the one of :func:`select_prefetch`.  The per-target forms -- the one-shot
    kernel's row-scale poll and re-arm, the two-shot re-arm, the fused norm's
    poll, quantizing pass 2 and sum-of-squares order -- follow from the
    registry's rows (a target builds exactly one form of each), so this
    function never restates them.
    """

    candidates = programs(
        family=family,
        world_size=int(world_size),
        input_format=input_format,
        output_format="" if output_format is None else output_format,
        target=target,
    )
    if not candidates:
        raise ValueError(
            "no Cake NVFP4-transport all-reduce program for family "
            f"{family!r}, world size {world_size}, input format {input_format!r}, "
            f"output format {output_format!r} on target {target!r}"
        )
    if any(PROGRAMS[program]["bv"] is not None for program in candidates):
        bv = select_bv(int(rows), int(grid))
        candidates = tuple(
            program for program in candidates if PROGRAMS[program]["bv"] == bv
        )
    if any(PROGRAMS[program]["prefetch"] is not None for program in candidates):
        prefetch = select_prefetch(int(rows), int(grid))
        candidates = tuple(
            program
            for program in candidates
            if PROGRAMS[program]["prefetch"] == prefetch
        )
    if len(candidates) != 1:
        raise RuntimeError(
            "the Cake NVFP4-transport all-reduce registry resolves "
            f"{len(candidates)} programs for family {family!r}, world size "
            f"{world_size}, input format {input_format!r}, output format "
            f"{output_format!r}, rows {rows}, grid {grid}, target {target!r}: "
            f"{candidates}"
        )
    return candidates[0]


def workspace_programs(
    family: str, world_size: int, input_format: str, target: str
) -> tuple[str, ...]:
    """Every program a workspace of (family, world size, input format) can launch on ``target``."""

    selected = programs(
        family=family,
        world_size=int(world_size),
        input_format=input_format,
        target=target,
    )
    if not selected:
        raise ValueError(
            "no Cake NVFP4-transport all-reduce program for family "
            f"{family!r}, world size {world_size}, input format {input_format!r} "
            f"on target {target!r}"
        )
    return selected


def window_contract(family: str, world_size: int, input_format: str) -> dict:
    """The receive-window contract of one (family, world size, input format) (``WINDOWS`` row)."""

    key = f"{family}/{int(world_size)}/{input_format}"
    try:
        return WINDOWS[key]
    except KeyError:
        raise ValueError(
            f"no Cake NVFP4-transport all-reduce window contract for {key!r}; "
            f"contracts: {sorted(WINDOWS)}"
        ) from None


def evaluate_linear_form(form: dict, cap: int) -> int:
    """``a * cap + b * capo + c`` of a registry linear form ``{"cap": a, "capo": b, "const": c}``.

    ``capo = ceil(cap / 4)`` is the per-owner row capacity of the world-size-4
    two-shot windows.
    """

    unknown = set(form) - {"cap", "capo", "const"}
    if unknown:
        raise ValueError(f"unknown linear-form keys {sorted(unknown)} in {form!r}")
    cap = int(cap)
    capo = -(-cap // 4)
    return (
        int(form.get("cap", 0)) * cap
        + int(form.get("capo", 0)) * capo
        + int(form.get("const", 0))
    )


def _source_dir() -> Path:
    installed = jit_env.FLASHINFER_CSRC_DIR / SOURCE_PACKAGE
    if installed.is_dir():
        return installed
    checkout = Path(__file__).resolve().parents[2] / "csrc" / SOURCE_PACKAGE
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError(
        "Cake NVFP4-transport all-reduce CUDA sources were not found. Checked:\n"
        f"  - {installed}\n  - {checkout}"
    )


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    checkout = Path(__file__).resolve().parents[2] / "include"
    if checkout.is_dir():
        return checkout
    raise FileNotFoundError("FlashInfer headers were not found")


def _source_path(relative: str) -> Path:
    parts = Path(relative).parts
    if parts[:2] != ("csrc", SOURCE_PACKAGE) or len(parts) != 3:
        raise ValueError(
            f"exported source path is outside the {SOURCE_PACKAGE} package: {relative!r}"
        )
    return _source_dir() / parts[2]


def program_sources(program: str) -> list[Path]:
    """Device and binding translation units of ``program`` as resolved paths."""

    return [_source_path(relative) for relative in program_record(program)["sources"]]


def _check_target(target: str) -> None:
    if target not in TARGET_FLAGS:
        raise ValueError(
            f"unknown Cake NVFP4-transport all-reduce target {target!r}; "
            f"targets: {sorted(TARGET_FLAGS)}"
        )


@functools.cache
def module_name(program: str, target: str) -> str:
    """``cake_fp4_allreduce_<program>_<target>_<digest>``: the exact target and a SHA-256 over the sources (20 hex digits).

    FlashInfer resolves installed AOT artifacts by module name before ninja
    sees the build inputs, so the content-derived name keeps a changed program
    from reusing a stale library, and the target keeps the exact-architecture
    libraries of one program apart under one target set.
    """

    _check_target(target)
    record = program_record(program)
    if target not in record["targets"]:
        raise ValueError(
            f"program {program!r} is not compiled for target {target!r} "
            f"(targets: {record['targets']})"
        )
    digest = hashlib.sha256()
    for source in program_sources(program):
        digest.update(source.read_bytes())
    return f"{MODULE_PREFIX}_{program}_{target}_{digest.hexdigest()[:20]}"


@functools.cache
def gen_cake_fp4_allreduce_module(program: str, target: str) -> JitSpec:
    """JIT spec of one (program, exact target): the device and binding translation units, the target's flags, no fast math."""

    source_dir = _source_dir()
    return gen_jit_spec(
        name=module_name(program, target),
        sources=program_sources(program),
        extra_cuda_cflags=[*TARGET_FLAGS[target], *COMPILE_FLAGS],
        extra_include_paths=[source_dir, source_dir.parent, _include_dir()],
        # The source build carries ``--fmad=false``; FlashInfer's default
        # ``-use_fast_math`` would change the codec's rounding.
        use_fast_math=False,
    )


@functools.cache
def get_cake_fp4_allreduce_module(program: str, target: str):
    """The built module of one (program, exact target); its ``run`` entry launches the kernel."""

    return gen_cake_fp4_allreduce_module(program, target).build_and_load()


def gen_cake_fp4_allreduce_aot_modules(target: str) -> list[JitSpec]:
    """JIT specs of every program compiled for the exact ``target`` (one per program), for the AOT build."""

    _check_target(target)
    return [
        gen_cake_fp4_allreduce_module(program, target)
        for program in programs(target=target)
    ]


__all__ = [
    "COMPILE_FLAGS",
    "FAMILIES",
    "HIDDEN",
    "INPUT_FORMATS",
    "LIMITS",
    "MODULE_PREFIX",
    "OUTPUT_FORMATS",
    "PROGRAMS",
    "RECORD_BYTES",
    "SCALAR_PARAMS",
    "SELECT_BV",
    "SELECT_PREFETCH",
    "SOURCE_PACKAGE",
    "SUPPORTED_CAPABILITIES",
    "TARGET_CAPABILITIES",
    "TARGET_FLAGS",
    "WINDOWS",
    "CakeFp4AllReduceTarget",
    "DeviceFacts",
    "device_facts",
    "evaluate_linear_form",
    "gen_cake_fp4_allreduce_aot_modules",
    "gen_cake_fp4_allreduce_module",
    "get_cake_fp4_allreduce_module",
    "module_name",
    "output_formats",
    "program_for",
    "program_record",
    "program_sources",
    "programs",
    "select_bv",
    "select_prefetch",
    "supported_capability",
    "target_arch",
    "target_capabilities",
    "target_of_capability",
    "window_contract",
    "workspace_programs",
    "world_sizes",
]
