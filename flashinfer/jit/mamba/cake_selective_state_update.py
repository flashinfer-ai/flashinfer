# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cake selective-state-update programs for Blackwell (SM100 / SM103).

The architecture-neutral CUDA programs under ``csrc/cake_selective_state_update``
serve the Cake backend of :func:`flashinfer.mamba.selective_state_update`.
``MODULES`` records each generated program (sources, compile flags, FFI entry,
argument plan, the names of its compile-time defines and every delivered
instantiation keyed by the define-set digest that names its JIT module);
``PROGRAMS`` maps the logical program names to those records.  Routes are
chosen from host-known tensor metadata only: no device read-back, no device
allocation and no stream synchronization happen on the call path, so every
route can be captured into a CUDA graph.
"""

from __future__ import annotations

import functools
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any, NamedTuple, Optional, Sequence

import torch
from filelock import FileLock
from tvm_ffi import cpp

from .. import env as jit_env
from ..core import (
    JitSpec,
    MissingJITCacheError,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)
from ...utils import get_compute_capability, get_device_sm_count

# Filled by the generated-program export.
MODULES: dict[str, dict[str, Any]] = {
    "cake_selective_state_update_08b0852b0be22a7ed2d8": {
        "sources": [
            "cake_selective_state_update_08b0852b0be22a7ed2d8_kernel.cu",
            "cake_selective_state_update_08b0852b0be22a7ed2d8_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["tma_buffer", "state_tma"],
            ["tma_buffer", "x_tma"],
            ["tma_buffer", "b_tma"],
            ["tma_buffer", "c_tma"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "intermediate_state"],
            ["buffer", "intermediate_state_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "total_tiles"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "cache_intermediate"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "5d4c58c266249a6d8c12f0d11915a7f8c205474b4a0255cf19b3f349d5c99adc",
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_0aac35fc16c30d81a328": {
        "sources": [
            "cake_selective_state_update_0aac35fc16c30d81a328_kernel.cu",
            "cake_selective_state_update_0aac35fc16c30d81a328_binding.cu",
        ],
        "compile_flags": ["--fmad=false"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "77d68f4e98b735cb546711b906efa1c4ba757b1d1010b552868a7f2e3d4623a9",
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_4ececa3a791410c1bb3a": {
        "sources": [
            "cake_selective_state_update_4ececa3a791410c1bb3a_kernel.cu",
            "cake_selective_state_update_4ececa3a791410c1bb3a_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "output"],
            ["buffer", "intermediate_state"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["parameter", "d_addr"],
            ["parameter", "dt_bias_addr"],
            ["parameter", "state_batch_indices_addr"],
            ["parameter", "intermediate_state_indices_addr"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "state_stride_slot"],
            ["parameter", "intermediate_stride_slot"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "5f68e76d091d95d304d2e2119b501ff33b3f3720a737ee3e438f638d9432727b",
        "defines": [
            "COEFFICIENT_BF16",
            "INDEX_I32",
            "NHEADS_STATIC",
            "NGROUPS_STATIC",
            "X_BATCH_STRIDE",
            "X_STEP_STRIDE",
            "DT_BATCH_STRIDE",
            "DT_STEP_STRIDE",
            "B_BATCH_STRIDE",
            "B_STEP_STRIDE",
            "C_BATCH_STRIDE",
            "C_STEP_STRIDE",
        ],
        "instantiations": {
            "8cb7048fb7793f88": {
                "COEFFICIENT_BF16": 0,
                "INDEX_I32": 0,
                "NHEADS_STATIC": 0,
                "NGROUPS_STATIC": 0,
                "X_BATCH_STRIDE": 24576,
                "X_STEP_STRIDE": 4096,
                "DT_BATCH_STRIDE": 384,
                "DT_STEP_STRIDE": 64,
                "B_BATCH_STRIDE": 6144,
                "B_STEP_STRIDE": 1024,
                "C_BATCH_STRIDE": 6144,
                "C_STEP_STRIDE": 1024,
            },
            "2682678a821d9bed": {
                "COEFFICIENT_BF16": 0,
                "INDEX_I32": 0,
                "NHEADS_STATIC": 0,
                "NGROUPS_STATIC": 0,
                "X_BATCH_STRIDE": 24576,
                "X_STEP_STRIDE": 4096,
                "DT_BATCH_STRIDE": 384,
                "DT_STEP_STRIDE": 64,
                "B_BATCH_STRIDE": 768,
                "B_STEP_STRIDE": 128,
                "C_BATCH_STRIDE": 768,
                "C_STEP_STRIDE": 128,
            },
            "e5f56c56ab9d4db4": {
                "COEFFICIENT_BF16": 1,
                "INDEX_I32": 1,
                "NHEADS_STATIC": 64,
                "NGROUPS_STATIC": 1,
                "X_BATCH_STRIDE": 26112,
                "X_STEP_STRIDE": 4352,
                "DT_BATCH_STRIDE": 51072,
                "DT_STEP_STRIDE": 8512,
                "B_BATCH_STRIDE": 26112,
                "B_STEP_STRIDE": 4352,
                "C_BATCH_STRIDE": 26112,
                "C_STEP_STRIDE": 4352,
            },
        },
    },
    "cake_selective_state_update_a22e10ee14cc5c07acb7": {
        "sources": [
            "cake_selective_state_update_a22e10ee14cc5c07acb7_kernel.cu",
            "cake_selective_state_update_a22e10ee14cc5c07acb7_binding.cu",
        ],
        "compile_flags": [],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["parameter", "dt_addr"],
            ["parameter", "a_addr"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["parameter", "d_addr"],
            ["buffer", "z"],
            ["parameter", "dt_bias_addr"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["buffer", "dst_state_batch_indices"],
            ["parameter", "nheads"],
            ["parameter", "ngroups"],
            ["parameter", "dim_tiles"],
            ["parameter", "state_stride_slot"],
            ["parameter", "dt_batch_stride"],
            ["parameter", "dt_head_stride"],
            ["parameter", "a_head_stride"],
            ["parameter", "d_head_stride"],
            ["parameter", "dt_bias_head_stride"],
            ["parameter", "dt_softplus"],
            ["parameter", "has_z"],
            ["parameter", "disable_state_update"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "fbd3088b0ea10286305d02857d24c502031029b70a4f81275211230c05905657",
        "defines": [],
        "instantiations": {},
    },
    "cake_selective_state_update_a623bc5b6863d0333670": {
        "sources": [
            "cake_selective_state_update_a623bc5b6863d0333670_kernel.cu",
            "cake_selective_state_update_a623bc5b6863d0333670_binding.cu",
        ],
        "compile_flags": ["--use_fast_math"],
        "ffi_entry": "run",
        "arg_plan": [
            ["buffer", "state"],
            ["buffer", "x"],
            ["buffer", "dt"],
            ["buffer", "A"],
            ["buffer", "B"],
            ["buffer", "C"],
            ["buffer", "D"],
            ["buffer", "dt_bias"],
            ["buffer", "output"],
            ["buffer", "state_batch_indices"],
            ["parameter", "batch_size"],
            ["parameter", "nheads"],
            ["parameter", "dim"],
            ["parameter", "dstate"],
            ["parameter", "ngroups"],
            ["parameter", "token_steps"],
            ["parameter", "state_stride_slot"],
            ["parameter", "dt_softplus"],
            ["parameter", "pad_slot_id"],
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ],
        "closure_sha256": "acaf9008565a79c955c63733dac855699aa8a5e3f88d717a131920d40e801ea3",
        "defines": [],
        "instantiations": {},
    },
}
PROGRAMS: dict[str, str] = {
    "dynamic": "cake_selective_state_update_0aac35fc16c30d81a328",
    "mtp_cache_c4_t6": "cake_selective_state_update_4ececa3a791410c1bb3a",
    "mtp_horizontal": "cake_selective_state_update_08b0852b0be22a7ed2d8",
    "mtp_short": "cake_selective_state_update_a623bc5b6863d0333670",
    "stp_fp32_identity": "cake_selective_state_update_a22e10ee14cc5c07acb7",
}

_ARCH_FLAGS = {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}
_ARCH_OF_CAPABILITY = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
_GRID_AXES = ("grid_x", "grid_y", "grid_z")
_STP_HEADS_PER_CTA = 4
_STP_RESIDENT_CTAS_PER_SM = 3
_STP_DIRECT_MAX_RESIDENT_WAVES = 9
_STP_SATURATED_HEAD_TILES = 2048
_MTP_HEADS_PER_CTA = 2
_MTP_WORKER_CTAS_PER_SM = 2
_MTP_RESIDENT_CTAS_PER_SM = 4
_MTP_DIRECT_MAX_RESIDENT_WAVES = 2
_DYNAMIC_MAX_TOKEN_STEPS = 8


def _source_dir() -> Path:
    packaged = jit_env.FLASHINFER_CSRC_DIR / "cake_selective_state_update"
    if packaged.is_dir():
        return packaged
    return Path(__file__).resolve().parents[3] / "csrc" / "cake_selective_state_update"


def _include_dir() -> Path:
    if jit_env.FLASHINFER_INCLUDE_DIR.is_dir():
        return jit_env.FLASHINFER_INCLUDE_DIR
    return Path(__file__).resolve().parents[3] / "include"


@functools.cache
def define_digest(defines: Sequence[tuple[str, int]]) -> str:
    """Stable 16-hex digest of a define set (sorted ``NAME=value`` pairs)."""
    return hashlib.sha256(
        "|".join(f"{name}={value}" for name, value in sorted(defines)).encode()
    ).hexdigest()[:16]


def jit_spec(
    module: str, arch: str, defines: tuple[tuple[str, int], ...] = ()
) -> JitSpec:
    """Build specification of one delivered program for ``arch`` and its compile-time defines."""
    record = MODULES[module]
    source_dir = _source_dir()
    # The define set names the instantiation through its digest (spelling twelve defines out exceeds
    # the 255-byte file-name limit of the JIT cache); ``MODULES[module]["instantiations"]`` maps every
    # delivered digest back to its define values.
    suffix = f"_{define_digest(defines)}" if defines else ""
    return gen_jit_spec(
        name=f"{module}_{arch}{suffix}",
        sources=[source_dir / path for path in record["sources"]],
        extra_cuda_cflags=[
            *_ARCH_FLAGS[arch],
            *record["compile_flags"],
            *[f"-D{name}={value}" for name, value in defines],
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[source_dir, source_dir.parent, _include_dir()],
        use_fast_math=False,
    )


def gen_cake_selective_state_update_modules(arch: str) -> list[JitSpec]:
    """Build specifications of every delivered program and instantiation for ``arch`` (the AOT table).

    The shipped BF16 single-token programs are built from their embedded device sources at first
    use and have no build specification; a disabled JIT with a cache that lacks one of these
    specifications makes the loader hand the call to FlashInfer.
    """
    specs: list[JitSpec] = []
    for module, record in MODULES.items():
        instantiations = record["instantiations"] or {"": {}}
        for values in instantiations.values():
            defines = tuple((name, int(values[name])) for name in record["defines"])
            specs.append(jit_spec(module, arch, defines))
    return specs


class _Program:
    """A loaded FFI entry bound to its argument plan (``buffer`` / ``parameter`` bindings and grid axes)."""

    __slots__ = ("_call", "_names", "_grid_slots")

    def __init__(self, call: Any, arg_plan: Sequence[Sequence[str]]):
        self._call = call
        names: list[Optional[str]] = []
        grid_slots: list[tuple[int, int]] = []
        for slot, (kind, name) in enumerate(arg_plan):
            if kind == "grid":
                names.append(None)
                grid_slots.append((slot, _GRID_AXES.index(name)))
            else:
                names.append(name)
        self._names = tuple(names)
        self._grid_slots = tuple(grid_slots)

    def launch(self, grid: tuple[int, int, int], bindings: dict[str, Any]) -> None:
        args = [bindings[name] if name is not None else 0 for name in self._names]
        for slot, axis in self._grid_slots:
            args[slot] = grid[axis]
        self._call(*args)


def _generated_program(
    program: str, arch: str, defines: tuple[tuple[str, int], ...]
) -> _Program:
    record = MODULES[PROGRAMS[program]]
    call = getattr(
        jit_spec(PROGRAMS[program], arch, defines).build_and_load(), record["ffi_entry"]
    )
    return _Program(call, record["arg_plan"])


class _ShippedStp(NamedTuple):
    """One BF16 single-token program shipped as source under ``cuda/`` (device) and ``host/`` (binding)."""

    module_ident: str
    flags: tuple[str, ...]
    arg_plan: tuple[tuple[str, str], ...]


_SHIPPED_STP_HEAD = (
    ("buffer", "state_tma"),
    ("buffer", "x"),
    ("buffer", "dt"),
    ("buffer", "A"),
    ("buffer", "B"),
    ("buffer", "C"),
    ("buffer", "D"),
    ("buffer", "z"),
    ("buffer", "dt_bias"),
    ("buffer", "output"),
    ("buffer", "state_batch_indices"),
    ("buffer", "dst_state_batch_indices"),
    ("parameter", "nheads"),
    ("parameter", "ngroups"),
    ("parameter", "head_tiles"),
)
_SHIPPED_STP_FLAGS = (
    ("parameter", "dt_softplus"),
    ("parameter", "has_z"),
    ("parameter", "disable_state_update"),
)
_SHIPPED_STP_GRID = (("grid", "grid_x"), ("grid", "grid_y"), ("grid", "grid_z"))
_SHIPPED_STP_DIRECT_PLAN = _SHIPPED_STP_HEAD + _SHIPPED_STP_FLAGS + _SHIPPED_STP_GRID

# The BF16 single-token programs are kept as they shipped: one source per head ratio / grid
# class, compiled by nvcc into a cubin that its host binding embeds.  The regenerated direct
# program measured 3.5-4 % slower than this clone on B200 at one head per group, so the
# regeneration of these five programs is tracked separately.
_SHIPPED_STP: dict[str, _ShippedStp] = {
    "stp_bf16_direct": _ShippedStp(
        "cake_selective_state_update_stp_bf16_direct_c020352b2a",
        ("--fmad=false",),
        _SHIPPED_STP_DIRECT_PLAN,
    ),
    "stp_bf16_ratio8": _ShippedStp(
        "cake_selective_state_update_stp_bf16_ratio8_5bf1b03b5e",
        ("--fmad=false",),
        _SHIPPED_STP_DIRECT_PLAN,
    ),
    "stp_bf16_ratio8_saturated": _ShippedStp(
        "cake_selective_state_update_stp_bf16_ratio8_saturated_d48b00dba2",
        ("--fmad=false",),
        _SHIPPED_STP_DIRECT_PLAN,
    ),
    "stp_bf16_ratio16": _ShippedStp(
        "cake_selective_state_update_stp_bf16_ratio16_c67b335fc4",
        ("--fmad=false",),
        _SHIPPED_STP_DIRECT_PLAN,
    ),
    "stp_bf16_persistent": _ShippedStp(
        "cake_selective_state_update_stp_bf16_persistent_a098dbb568",
        ("--fmad=false",),
        _SHIPPED_STP_HEAD
        + (("parameter", "total_head_tiles"),)
        + _SHIPPED_STP_FLAGS
        + _SHIPPED_STP_GRID,
    ),
}


def _nvcc() -> Path:
    candidate = shutil.which("nvcc")
    if candidate is None:
        cuda_root = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
        if cuda_root:
            path = Path(cuda_root) / "bin" / "nvcc"
            if path.is_file():
                candidate = str(path)
    if candidate is None:
        raise RuntimeError("nvcc is required to build the Cake backend")
    return Path(candidate).resolve()


@functools.cache
def _load_shipped(name: str, arch: str) -> Any:
    """``run`` of one shipped BF16 STP program: its device source built to a cubin, embedded by its host binding."""
    program = _SHIPPED_STP[name]
    source_dir = _source_dir()
    device_source = source_dir / "cuda" / f"cake_selective_state_update_{name}.cu"
    host_source = source_dir / "host" / f"cake_selective_state_update_{name}.cc"
    nvcc = _nvcc()
    digest = hashlib.sha256()
    digest.update(device_source.read_bytes())
    digest.update(host_source.read_bytes())
    digest.update(arch.encode())
    digest.update(str(nvcc).encode())
    module_name = f"cake_selective_state_update_{name}_{arch}_{digest.hexdigest()[:16]}"
    build_dir = jit_env.FLASHINFER_JIT_DIR / module_name
    build_dir.mkdir(parents=True, exist_ok=True)
    cubin_path = build_dir / f"{program.module_ident}.cubin"
    with FileLock(build_dir / f"{program.module_ident}.lock", thread_local=False):
        if not cubin_path.is_file():
            temporary = build_dir / f"{program.module_ident}.{os.getpid()}.tmp.cubin"
            command = [
                str(nvcc),
                "-cubin",
                f"-arch={arch}",
                "--std=c++17",
                "-O3",
                "-I",
                str(nvcc.parent.parent / "include"),
                *program.flags,
                str(device_source),
                "-o",
                str(temporary),
            ]
            process = subprocess.run(command, text=True, capture_output=True)
            if process.returncode != 0:
                temporary.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Cake nvcc failed for {name} ({arch}):\n{process.stderr}"
                )
            os.replace(temporary, cubin_path)
        module = cpp.load_inline(
            module_name,
            cpp_sources=host_source.read_text(encoding="utf-8"),
            embed_cubin={program.module_ident: cubin_path.read_bytes()},
            extra_include_paths=[str(nvcc.parent.parent / "include")],
            extra_cflags=["-O3"],
            extra_ldflags=["-lcuda"],
            build_directory=str(build_dir),
        )
    return module.run


@functools.cache
def _program(program: str, arch: str, defines: tuple[tuple[str, int], ...]) -> _Program:
    shipped = _SHIPPED_STP.get(program)
    if shipped is not None:
        return _Program(_load_shipped(program, arch), shipped.arg_plan)
    return _generated_program(program, arch, defines)


@functools.cache
def _device_arch(device_index: int) -> Optional[str]:
    return _ARCH_OF_CAPABILITY.get(
        get_compute_capability(torch.device("cuda", device_index))
    )


@functools.cache
def _sm_count(device_index: int) -> int:
    return int(get_device_sm_count(torch.device("cuda", device_index)))


class RoutePlan(NamedTuple):
    """One resolved launch: program, compile-time defines, grid and kernel bindings."""

    program: str
    defines: tuple[tuple[str, int], ...]
    grid: tuple[int, int, int]
    bindings: dict[str, Any]
    arch: str
    device_index: int


def _balanced_workers(work: int, cap: int) -> int:
    if work <= cap:
        return work
    trips = (work + cap - 1) // cap
    return (work + trips - 1) // trips


def _compact_broadcasts(
    dt: torch.Tensor, A: torch.Tensor, D: torch.Tensor, dt_bias: torch.Tensor
) -> Optional[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Drop the zero-stride broadcast axes of the coefficient tensors (metadata only)."""
    if dt.stride(-1) != 0 or A.stride(-1) != 0 or A.stride(-2) != 0:
        return None
    if D.stride(-1) != 0 or dt_bias.stride(-1) != 0:
        return None
    return (
        dt.as_strided(dt.shape[:-1], dt.stride()[:-1]),
        A.as_strided((A.shape[0],), (A.stride(0),)),
        D.as_strided((D.shape[0],), (D.stride(0),)),
        dt_bias.as_strided((dt_bias.shape[0],), (dt_bias.stride(0),)),
    )


def _coefficients(dt, A, D, dt_bias, x_shape, nheads: int, dim: int, dstate: int):
    """Compact per-head views of the coefficient tensors, or ``None`` when a shape or stride is not the broadcast layout.

    The programs read one ``A`` / ``D`` / ``dt_bias`` value per head and one ``dt`` value per row (zero strides on
    the broadcast dims). One ``shape`` and one ``stride()`` read per tensor: the plan path is attribute traffic.
    """
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    if (
        dt.shape != x_shape
        or A.shape != (nheads, dim, dstate)
        or D.shape != (nheads, dim)
        or dt_bias.shape != (nheads, dim)
        or dt_stride[-1] != 0
        or a_stride[1:] != (0, 0)
        or d_stride[-1] != 0
        or bias_stride[-1] != 0
    ):
        return None
    return (
        dt.as_strided(x_shape[:-1], dt_stride[:-1]),
        A.as_strided((nheads,), (a_stride[0],)),
        D.as_strided((nheads,), (d_stride[0],)),
        dt_bias.as_strided((nheads,), (bias_stride[0],)),
    )


def _stp_bf16_launch(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    output,
    source,
    destination,
    ngroups,
    dt_softplus,
    disable_state_update,
) -> Optional[tuple[dict[str, Any], int, int]]:
    """``(bindings, heads_per_group, total_head_tiles)`` shared by every BF16 single-token program, or ``None``."""
    batch_size, nheads, _ = x.shape
    heads_per_group = nheads // ngroups
    head_tiles = (heads_per_group + _STP_HEADS_PER_CTA - 1) // _STP_HEADS_PER_CTA
    compact = _compact_broadcasts(dt, A, D, dt_bias)
    if compact is None:
        return None
    dt_compact, A_compact, D_compact, bias_compact = compact
    bindings = dict(
        state_tma=state,
        x=x,
        dt=dt_compact,
        A=A_compact,
        B=B,
        C=C,
        D=D_compact,
        z=x if z is None else z,
        dt_bias=bias_compact,
        output=output,
        state_batch_indices=source,
        dst_state_batch_indices=destination,
        nheads=nheads,
        ngroups=ngroups,
        head_tiles=head_tiles,
        dt_softplus=int(dt_softplus),
        has_z=int(z is not None),
        disable_state_update=int(disable_state_update),
    )
    return bindings, heads_per_group, batch_size * ngroups * head_tiles


def shipped_stp_program(
    heads_per_group: int, total_head_tiles: int, sm_count: int
) -> str:
    """Shipped BF16 single-token program of one launch.

    Grids above the direct resident-wave cap run the balanced persistent
    program; otherwise the head ratio selects the clone (16 heads per group,
    8 with a saturated or an unsaturated grid, any other ratio).
    """
    if (
        total_head_tiles
        > _STP_DIRECT_MAX_RESIDENT_WAVES * _STP_RESIDENT_CTAS_PER_SM * sm_count
    ):
        return "stp_bf16_persistent"
    if heads_per_group == 16:
        return "stp_bf16_ratio16"
    if heads_per_group == 8:
        return (
            "stp_bf16_ratio8_saturated"
            if total_head_tiles >= _STP_SATURATED_HEAD_TILES
            else "stp_bf16_ratio8"
        )
    return "stp_bf16_direct"


def _plan_stp_bf16(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    z,
    dt_bias,
    output,
    source,
    destination,
    ngroups,
    dt_softplus,
    disable_state_update,
    arch,
    device_index,
) -> Optional[RoutePlan]:
    """The BF16 single-token route on the shipped programs: same shapes and selection as before this loader."""
    launch = _stp_bf16_launch(
        state=state,
        x=x,
        dt=dt,
        A=A,
        B=B,
        C=C,
        D=D,
        z=z,
        dt_bias=dt_bias,
        output=output,
        source=source,
        destination=destination,
        ngroups=ngroups,
        dt_softplus=dt_softplus,
        disable_state_update=disable_state_update,
    )
    if launch is None:
        return None
    bindings, heads_per_group, total_head_tiles = launch
    sms = _sm_count(device_index)
    program = shipped_stp_program(heads_per_group, total_head_tiles, sms)
    if program == "stp_bf16_persistent":
        workers = _balanced_workers(total_head_tiles, _STP_RESIDENT_CTAS_PER_SM * sms)
        bindings["total_head_tiles"] = total_head_tiles
        return RoutePlan(program, (), (workers, 1, 1), bindings, arch, device_index)
    return RoutePlan(
        program, (), (total_head_tiles, 1, 1), bindings, arch, device_index
    )


def cache_defines(
    coefficient_dtype: torch.dtype,
    index_dtype: torch.dtype,
    nheads: int,
    ngroups: int,
    x_stride: Sequence[int],
    dt_stride: Sequence[int],
    b_stride: Sequence[int],
    c_stride: Sequence[int],
) -> tuple[tuple[str, int], ...]:
    """Compile-time defines of the six-token cache program for one launch.

    BF16 ``dt``/``D``/``dt_bias`` storage and int32 slot tables (the SGLang
    ABI) select the folded loads, and that ABI also folds its model constants
    ``nheads`` / ``ngroups``; the canonical FP32/int64 storage keeps runtime
    extents.  The projection (batch, step) strides are always folded: one
    instantiation per projection layout.
    """
    raw_sglang = coefficient_dtype == torch.bfloat16 and index_dtype == torch.int32
    return (
        ("COEFFICIENT_BF16", int(coefficient_dtype == torch.bfloat16)),
        ("INDEX_I32", int(index_dtype == torch.int32)),
        ("NHEADS_STATIC", int(nheads) if raw_sglang else 0),
        ("NGROUPS_STATIC", int(ngroups) if raw_sglang else 0),
        ("X_BATCH_STRIDE", int(x_stride[0])),
        ("X_STEP_STRIDE", int(x_stride[1])),
        ("DT_BATCH_STRIDE", int(dt_stride[0])),
        ("DT_STEP_STRIDE", int(dt_stride[1])),
        ("B_BATCH_STRIDE", int(b_stride[0])),
        ("B_STEP_STRIDE", int(b_stride[1])),
        ("C_BATCH_STRIDE", int(c_stride[0])),
        ("C_STEP_STRIDE", int(c_stride[1])),
    )


def _plan_mtp_cache(
    *,
    state,
    x,
    dt,
    A,
    B,
    C,
    D,
    dt_bias,
    output,
    source,
    intermediate,
    intermediate_indices,
    nheads,
    ngroups,
    pad_slot_id,
    arch,
    device_index,
) -> Optional[RoutePlan]:
    """Six-token D64/N128 cache program on any projection layout with dense head/group rows.

    One ``shape``/``stride()`` read per tensor: the per-call host path of this
    route is Python attribute traffic, so every metadata value is read once.
    """
    x_shape, x_stride = x.shape, x.stride()
    batch_size = x_shape[0]
    b_shape, b_stride, c_stride = B.shape, B.stride(), C.stride()
    dt_stride, a_stride, d_stride, bias_stride = (
        dt.stride(),
        A.stride(),
        D.stride(),
        dt_bias.stride(),
    )
    if (
        x_shape[1:] != (6, nheads, 64)
        or dt.shape != x_shape
        or b_shape != (batch_size, 6, ngroups, 128)
        or C.shape != b_shape
        or output.shape != x_shape
        or state.shape[1:] != (nheads, 64, 128)
        or A.shape != (nheads, 64, 128)
        or D.shape != (nheads, 64)
        or dt_bias.shape != (nheads, 64)
        or intermediate.shape[1:] != (6, nheads, 64, 128)
        or source.shape != (batch_size,)
        or intermediate_indices.shape != (batch_size,)
        or output.dtype != torch.bfloat16
        or intermediate.dtype != state.dtype
        or dt.dtype != D.dtype
        or dt.dtype != dt_bias.dtype
    ):
        return None
    # Broadcast coefficients (one element per head, unit head stride) and dense projection rows
    # (unit last stride, head/group stride = row length, 16-byte aligned batch/step strides).
    if (
        dt_stride[3] != 0
        or dt_stride[2] != 1
        or a_stride != (1, 0, 0)
        or d_stride != (1, 0)
        or bias_stride != (1, 0)
        or x_stride[3] != 1
        or x_stride[2] != 64
        or (x_stride[0] | x_stride[1]) & 7
        or b_stride[3] != 1
        or b_stride[2] != 128
        or (b_stride[0] | b_stride[1]) & 7
        or c_stride[3] != 1
        or c_stride[2] != 128
        or (c_stride[0] | c_stride[1]) & 7
        or (x.data_ptr() | B.data_ptr() | C.data_ptr()) & 15
    ):
        return None
    if not (
        state.is_contiguous()
        and output.is_contiguous()
        and intermediate.is_contiguous()
        and source.is_contiguous()
        and intermediate_indices.is_contiguous()
    ):
        return None
    bindings = dict(
        state=state,
        x=x,
        B=B,
        C=C,
        output=output,
        intermediate_state=intermediate,
        dt_addr=dt.data_ptr(),
        a_addr=A.data_ptr(),
        d_addr=D.data_ptr(),
        dt_bias_addr=dt_bias.data_ptr(),
        state_batch_indices_addr=source.data_ptr(),
        intermediate_state_indices_addr=intermediate_indices.data_ptr(),
        nheads=nheads,
        ngroups=ngroups,
        state_stride_slot=state.stride(0),
        intermediate_stride_slot=intermediate.stride(0),
        pad_slot_id=pad_slot_id,
    )
    defines = cache_defines(
        dt.dtype, source.dtype, nheads, ngroups, x_stride, dt_stride, b_stride, c_stride
    )
    return RoutePlan(
        "mtp_cache_c4_t6",
        defines,
        (batch_size, nheads, 4),
        bindings,
        arch,
        device_index,
    )


def plan_route(
    *,
    state: torch.Tensor,
    x: torch.Tensor,
    dt: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor,
    z: Optional[torch.Tensor],
    dt_bias: Optional[torch.Tensor],
    output: torch.Tensor,
    state_batch_indices: Optional[torch.Tensor],
    dst_state_batch_indices: Optional[torch.Tensor],
    pad_slot_id: int,
    disable_state_update: bool,
    intermediate_states_buffer: Optional[torch.Tensor],
    intermediate_state_indices: Optional[torch.Tensor],
    state_scale: Optional[torch.Tensor],
    intermediate_state_scales: Optional[torch.Tensor],
    rand_seed: Optional[torch.Tensor],
    cache_steps: int,
    cu_seqlens: Optional[torch.Tensor],
    num_accepted_tokens: Optional[torch.Tensor],
    algorithm: str,
    dt_softplus: bool,
) -> Optional[RoutePlan]:
    """Resolve the Cake route of one call from tensor metadata, or ``None`` when no program serves it."""
    if (
        dt_bias is None
        or state_batch_indices is None
        or state_scale is not None
        or intermediate_state_scales is not None
        or rand_seed is not None
        or cu_seqlens is not None
        or num_accepted_tokens is not None
        or state.ndim != 4
        or state.dtype not in (torch.bfloat16, torch.float32)
        or x.dtype != torch.bfloat16
        or B.dtype != torch.bfloat16
        or C.dtype != torch.bfloat16
        or A.dtype != torch.float32
        or state_batch_indices.ndim != 1
    ):
        return None
    # One dtype read per coefficient and index tensor; an absent index table takes the dtype of the required one.
    dt_dtype, d_dtype, bias_dtype = dt.dtype, D.dtype, dt_bias.dtype
    index_dtype = state_batch_indices.dtype
    dst_dtype = (
        index_dtype
        if dst_state_batch_indices is None
        else dst_state_batch_indices.dtype
    )
    intermediate_dtype = (
        index_dtype
        if intermediate_state_indices is None
        else intermediate_state_indices.dtype
    )
    canonical = (
        dt_dtype == torch.float32
        and d_dtype == torch.float32
        and bias_dtype == torch.float32
        and index_dtype == dst_dtype == intermediate_dtype == torch.int64
    )
    raw_coefficients = (
        dt_dtype == torch.bfloat16
        and d_dtype == torch.bfloat16
        and bias_dtype == torch.bfloat16
        and index_dtype == dst_dtype == intermediate_dtype == torch.int32
    )
    if not (canonical or raw_coefficients):
        return None
    device_index = state.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    arch = _device_arch(device_index)
    if arch is None:
        return None
    # One shape read per tensor (``torch.Size`` holds plain ints); the routes below compare against these
    # tuples. Layouts the shapes admit but the programs cannot read (non-contiguous buffers) are rejected by
    # each binding's argument validation before any launch, and ``try_cake_selective_state_update`` falls
    # back on that.
    x_shape, b_shape, state_shape = x.shape, B.shape, state.shape
    batch_size = x_shape[0]
    nheads, dim, dstate = state_shape[1], state_shape[2], state_shape[3]
    ngroups = b_shape[-2]
    if ngroups <= 0 or nheads % ngroups:
        return None

    if (
        len(x_shape) == 3
        and cache_steps == 0
        and dim == 128
        and dstate == 128
        and canonical
    ):
        destination = (
            state_batch_indices
            if dst_state_batch_indices is None
            else dst_state_batch_indices
        )
        if destination.ndim != 1:
            return None
        if state.dtype == torch.bfloat16:
            return _plan_stp_bf16(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                z=z,
                dt_bias=dt_bias,
                output=output,
                source=state_batch_indices,
                destination=destination,
                ngroups=ngroups,
                dt_softplus=dt_softplus,
                disable_state_update=disable_state_update,
                arch=arch,
                device_index=device_index,
            )
        if (
            z is None
            and not dt_softplus
            and not disable_state_update
            and destination.data_ptr() == state_batch_indices.data_ptr()
            and batch_size * nheads >= 8 * _sm_count(device_index)
        ):
            # The program reads one coefficient per head (broadcast dt / A / D / dt_bias, passed by address
            # and head stride) at the shapes below; any other shape or a dense coefficient tensor plans no
            # launch. Shapes are compared before the strides are indexed, and each stride tuple is read once
            # and reused by the bindings.
            dt_stride, a_stride, d_stride, bias_stride = (
                dt.stride(),
                A.stride(),
                D.stride(),
                dt_bias.stride(),
            )
            if (
                x_shape != (batch_size, nheads, dim)
                or b_shape != (batch_size, ngroups, dstate)
                or C.shape != b_shape
                or output.shape != x_shape
                or output.dtype != torch.bfloat16
                or state_batch_indices.shape != (batch_size,)
                or dt.shape != x_shape
                or A.shape != (nheads, dim, dstate)
                or D.shape != (nheads, dim)
                or dt_bias.shape != (nheads, dim)
                or dt_stride[2] != 0
                or a_stride[1] != 0
                or a_stride[2] != 0
                or d_stride[1] != 0
                or bias_stride[1] != 0
            ):
                return None
            bindings = dict(
                state=state,
                x=x,
                dt_addr=dt.data_ptr(),
                a_addr=A.data_ptr(),
                B=B,
                C=C,
                d_addr=D.data_ptr(),
                z=x,
                dt_bias_addr=dt_bias.data_ptr(),
                output=output,
                state_batch_indices=state_batch_indices,
                dst_state_batch_indices=destination,
                nheads=nheads,
                ngroups=ngroups,
                dim_tiles=1,
                state_stride_slot=state.stride(0),
                dt_batch_stride=dt_stride[0],
                dt_head_stride=dt_stride[1],
                a_head_stride=a_stride[0],
                d_head_stride=d_stride[0],
                dt_bias_head_stride=bias_stride[0],
                dt_softplus=0,
                has_z=0,
                disable_state_update=0,
                pad_slot_id=pad_slot_id,
            )
            return RoutePlan(
                "stp_fp32_identity",
                (),
                (batch_size * nheads, 1, 1),
                bindings,
                arch,
                device_index,
            )
        return None

    if len(x_shape) != 4:
        return None
    c_shape = C.shape
    token_steps = x_shape[1]

    if (
        canonical
        and state.dtype == torch.bfloat16
        and (dim, dstate) == (128, 128)
        and token_steps in (1, 2)
        and z is None
        and dst_state_batch_indices is None
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        # The shapes the program indexes; the compact coefficient views carry the broadcast check.
        if (
            x_shape != (batch_size, token_steps, nheads, dim)
            or b_shape != (batch_size, token_steps, ngroups, dstate)
            or c_shape != b_shape
            or output.shape != x_shape
            or output.dtype != torch.bfloat16
            or state_batch_indices.shape != (batch_size,)
        ):
            return None
        compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
        if compact is None:
            return None
        dt_compact, A_compact, D_compact, bias_compact = compact
        bindings = dict(
            state=state,
            x=x,
            dt=dt_compact,
            A=A_compact,
            B=B,
            C=C,
            D=D_compact,
            dt_bias=bias_compact,
            output=output,
            state_batch_indices=state_batch_indices,
            batch_size=batch_size,
            nheads=nheads,
            dim=dim,
            dstate=dstate,
            ngroups=ngroups,
            token_steps=token_steps,
            state_stride_slot=state.stride(0),
            dt_softplus=int(dt_softplus),
            pad_slot_id=pad_slot_id,
        )
        return RoutePlan(
            "mtp_short", (), (batch_size * nheads, 1, 1), bindings, arch, device_index
        )

    if (
        state.dtype == torch.bfloat16
        and (dim, dstate, token_steps) == (64, 128, 6)
        and z is None
        and dst_state_batch_indices is None
        and dt_softplus
        and disable_state_update
        and intermediate_states_buffer is not None
        and intermediate_state_indices is not None
    ):
        total_tiles = batch_size * nheads
        sms = _sm_count(device_index)
        if batch_size < 32 and (sms * 10) // total_tiles >= 4:
            return _plan_mtp_cache(
                state=state,
                x=x,
                dt=dt,
                A=A,
                B=B,
                C=C,
                D=D,
                dt_bias=dt_bias,
                output=output,
                source=state_batch_indices,
                intermediate=intermediate_states_buffer,
                intermediate_indices=intermediate_state_indices,
                nheads=nheads,
                ngroups=ngroups,
                pad_slot_id=pad_slot_id,
                arch=arch,
                device_index=device_index,
            )
        if canonical and batch_size >= 32 and algorithm == "horizontal":
            # The shapes the program indexes (state, x, B and C arrive through tensor maps encoded from
            # their strides); the compact coefficient views carry the broadcast check.
            if (
                x_shape != (batch_size, token_steps, nheads, dim)
                or b_shape != (batch_size, token_steps, ngroups, dstate)
                or c_shape != b_shape
                or output.shape != x_shape
                or output.dtype != torch.bfloat16
                or state_batch_indices.shape != (batch_size,)
                or intermediate_states_buffer.shape[1:]
                != (token_steps, nheads, dim, dstate)
                or intermediate_states_buffer.dtype != torch.bfloat16
                or intermediate_state_indices.shape != (batch_size,)
            ):
                return None
            compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
            if compact is None:
                return None
            dt_compact, A_compact, D_compact, bias_compact = compact
            work_groups = (total_tiles + _MTP_HEADS_PER_CTA - 1) // _MTP_HEADS_PER_CTA
            # One CTA per work group up to the direct tile cap; balanced
            # persistent workers above it.
            if (
                work_groups
                <= _MTP_DIRECT_MAX_RESIDENT_WAVES * _MTP_RESIDENT_CTAS_PER_SM * sms
            ):
                workers = work_groups
            else:
                workers = _balanced_workers(work_groups, _MTP_WORKER_CTAS_PER_SM * sms)
            bindings = dict(
                state_tma=state,
                x_tma=x,
                b_tma=B,
                c_tma=C,
                dt=dt_compact,
                A=A_compact,
                D=D_compact,
                dt_bias=bias_compact,
                output=output,
                state_batch_indices=state_batch_indices,
                intermediate_state=intermediate_states_buffer,
                intermediate_state_indices=intermediate_state_indices,
                nheads=nheads,
                ngroups=ngroups,
                total_tiles=total_tiles,
                intermediate_stride_slot=intermediate_states_buffer.stride(0),
                dt_softplus=int(dt_softplus),
                cache_intermediate=1,
            )
            return RoutePlan(
                "mtp_horizontal", (), (workers, 1, 1), bindings, arch, device_index
            )
        return None

    if (
        canonical
        and state.dtype == torch.float32
        and (nheads, dim, dstate, ngroups) == (16, 64, 128, 1)
        and 1 <= token_steps <= _DYNAMIC_MAX_TOKEN_STEPS
        and algorithm == "simple"
        and dt_softplus
        and z is None
        and dst_state_batch_indices is not None
        and dst_state_batch_indices.shape == (batch_size, token_steps)
        and intermediate_states_buffer is None
        and not disable_state_update
    ):
        # The fixed-index program indexes every buffer at the shapes below; the compact coefficient
        # views carry the broadcast check.
        if (
            x_shape != (batch_size, token_steps, 16, 64)
            or b_shape != (batch_size, token_steps, 1, 128)
            or c_shape != b_shape
            or output.shape != x_shape
            or output.dtype != torch.bfloat16
            or state_batch_indices.shape != (batch_size,)
        ):
            return None
        compact = _coefficients(dt, A, D, dt_bias, x_shape, nheads, dim, dstate)
        if compact is None:
            return None
        dt_compact, A_compact, D_compact, bias_compact = compact
        bindings = dict(
            state=state,
            x=x,
            dt=dt_compact,
            A=A_compact,
            B=B,
            C=C,
            D=D_compact,
            dt_bias=bias_compact,
            output=output,
            state_batch_indices=state_batch_indices,
            dst_state_batch_indices=dst_state_batch_indices,
            batch_size=batch_size,
            token_steps=token_steps,
            state_stride_slot=state.stride(0),
            pad_slot_id=pad_slot_id,
        )
        return RoutePlan(
            "dynamic", (), (batch_size, nheads, 4), bindings, arch, device_index
        )
    return None


def try_cake_selective_state_update(**kwargs: Any) -> bool:
    """Run the call through a Cake program when one serves it; ``False`` leaves it to FlashInfer."""
    plan = plan_route(**kwargs)
    if plan is None:
        return False
    with torch.cuda.device(plan.device_index):
        try:
            program = _program(plan.program, plan.arch, plan.defines)
        except MissingJITCacheError:
            # JIT builds are disabled and the cache holds no build of this program: the call runs on FlashInfer.
            return False
        try:
            program.launch(plan.grid, plan.bindings)
        except ValueError:
            # The binding validates every argument (device, dtype, contiguity, tensor-map strides) before
            # it launches, so nothing has run when it rejects a layout: the call goes to FlashInfer.
            return False
    return True


__all__ = [
    "MODULES",
    "PROGRAMS",
    "RoutePlan",
    "define_digest",
    "jit_spec",
    "plan_route",
    "shipped_stp_program",
    "try_cake_selective_state_update",
]
