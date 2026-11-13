"""SM90 role preparation and pointer ABI; compiler utilities are shared with GC MoE."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass, replace
from pathlib import Path
import re

import torch
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_ptr

from ...impls.sm90_fp8_fp4_gc_moe import SM90GCMoERoleKernel
from ...runtime.sm90_gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
)
from ..heuristics.sm90_gc_moe import (
    get_sm90_role_config,
    get_routed_role_config,
    get_shared_role_config,
)


@dataclass(frozen=True)
class _Prepared:
    binaries: tuple[RoleBinary, ...]


_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, SM90GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_sm90_role')


def _assemble_role_ptx(ptx, role, dump_dir, identity):
    ptx = _clean_cute_ptx(ptx)
    formals = _entry_formals(ptx)
    if len(formals) != 17:
        raise RuntimeError('SM90 CuTe launch ABI must have fourteen pointers and three scalars')
    for ordinal, formal in enumerate(formals):
        kind = (r'\.u64\s+\.ptr\s+\.global\s+\.align\s+16' if ordinal < 14
                else r'\.u32' if ordinal < 16 else r'\.f32')
        if re.search(kind + r'\s+\S+_param_' + str(ordinal) + r',?\s*$', formal) is None:
            raise RuntimeError(f'SM90 CuTe launch formal {ordinal} changed type or order')
    if role in (2, 4, 5) and ('wgmma.mma_async' not in ptx or 'cp.async.bulk' not in ptx):
        raise RuntimeError('SM90 CuTe compute lost its Hopper WGMMA/TMA operations')
    return assemble_ptx(ptx, dump_dir, f'sm90_role{role}', target='sm_90a',
                        ptxas_options=identity.ptxas_options)


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx

def _entry_formals(ptx: str) -> list[str]:
    """Read only the launch ABI, excluding function and call-local params."""
    entries = list(re.finditer(
        r"(?m)^[ \t]*\.visible[ \t]+\.entry[ \t]+[A-Za-z0-9_.$]+\([ \t]*$",
        ptx))
    if len(entries) != 1:
        raise RuntimeError('CuTe kernel entry count changed')
    start = entries[0].end()
    closing = re.search(r"(?m)^\)[ \t]*$", ptx[start:])
    if closing is None:
        raise RuntimeError('CuTe kernel entry signature is unterminated')
    lines = [line for line in ptx[start:start + closing.start()].splitlines()
             if line.strip()]
    if not lines or any(re.fullmatch(r"[ \t]*\.param\b[^\n]*", line) is None
                        for line in lines):
        raise RuntimeError('CuTe kernel entry formals changed')
    return lines

_POINTER_TYPES = (Uint8, Uint64, BFloat16, Uint8, Uint8, Float32,
                  Uint8, Uint8, Float32, Uint32, Uint8, Float32, Uint8, Float32)


def _role_specs(config, layout, schedule, has_stats):
    routed = get_routed_role_config(config, schedule)
    shared = get_shared_role_config(schedule) if config.shared_experts else None
    split = bool(config.shared_experts and schedule.split_shared)
    l2_grid = (schedule.shared_l2_grid or shared.grids[1]) if split else 0
    # Bound waiting L2 CTAs, and order their launch after every L1 CTA has
    # begun. L1 never waits for L2; original per-block releases/acquires
    # provide data visibility without relying on concurrent execution.
    stream_shared = split and 0 < l2_grid < getattr(schedule, 'compute_sms', 0)
    roles = ((1, 4, 5, 2, 3) if split else
             ((1, 4, 2, 3) if config.shared_experts else (1, 2, 3)))
    specs = []
    for role in roles:
        index = 1 if role in (4, 5) else role - 1
        # Workspace routed offsets are a strict prefix of the shared layout.
        role_config = replace(config, shared_experts=0) if role == 2 else config
        role_schedule = shared if role in (4, 5) else (routed.schedule if role == 2 else schedule)
        if role == 5 and schedule.shared_l2_grid:
            # A bounded L2 branch may overlap M64 routed on different SMs;
            # these two resource-heavy kernels cannot share one SM.
            role_schedule = replace(role_schedule,
                                    grids=(role_schedule.grids[0], schedule.shared_l2_grid,
                                           role_schedule.grids[2]))
        kernel = SM90GCMoERoleKernel(role_config, layout, role, role_schedule, has_stats,
                                    routed_n_factor=routed.n_factor if role == 2 else 1,
                                    routed_stage_count=routed.stage_count_override if role == 2 else 0,
                                    routed_k_pack=routed.k_pack if role == 2 else 1)
        wait_for = ((4,) if split and (role == 2 or (role == 5 and not stream_shared)) else
                    ((4,) if config.shared_experts and role == 2 and schedule.tile_m >= 32 else ()))
        launch = (role, role_schedule.smem[index], role_schedule.threads[index],
                  role_schedule.grids[index], wait_for)
        if role == 5 and stream_shared:
            # CUDA launch-order port 2, default input port 0, PROGRAMMATIC 1.
            # Unlike a full edge this may overlap, but only after all L1
            # blocks have started. Finalize still joins full completions.
            launch += (((4, 2, 0, 1),),)
        if role == 2 and stream_shared and schedule.tile_m in (32, 64):
            # Preserve all launch resources. Only allow the independent routed
            # worker to start after every bounded L2 CTA has begun; L1 is finite
            # and never depends on either consumer. Finalize joins completions.
            launch = (*launch[:4], (), ((5, 2, 0, 1),))
        specs.append((kernel, launch))
    return tuple(specs)


def _compile_roles(specs, identity):
    pointers = tuple(make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=16)
                     for dtype in _POINTER_TYPES)
    binaries = []
    for kernel, launch in specs:
        role = launch[0]
        cubin = compile_kernel(
            kernel, (*pointers, Uint32(0), Uint32(0), Float32(0)), identity,
            lambda ptx, dump_dir: _assemble_role_ptx(ptx, role, dump_dir, identity))
        binaries.append(RoleBinary(role, cubin, *launch[1:]))
    return tuple(binaries)


def _binary_key(identity, specs):
    # Pointer compile arguments have fixed types. Kernel attributes and launch
    # resources cover every constexpr/layout field, without CUDA addresses.
    return (identity, tuple((tuple(sorted(vars(kernel).items())), launch)
                            for kernel, launch in specs))


def _get_or_compile_binaries(key, identity, specs):
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = _compile_roles(specs, identity)
            expected = tuple(launch[0] for _, launch in specs)
            if (expected not in ((1, 2, 3), (1, 4, 2, 3), (1, 4, 5, 2, 3))
                    or tuple(item.role for item in binaries) != expected):
                raise RuntimeError('SM90 compilation did not produce all role binaries')
            if _compiler_identity('sm_90a') != identity:
                raise RuntimeError('SM90 compiler/source identity changed while compiling')
            _BINARY_CACHE[key] = binaries
        return binaries



def prepare_role_binaries(config, y, w1, w2, buffer, stats=None, context=None,
                          shared_l1_weights=None, shared_l2_weights=None):
    if torch.cuda.get_device_capability(y.device) != (9, 0):
        raise ValueError('SM90 CuTe GC requires compute capability 9.0')
    if context is None:
        context = get_context()
    # A prepared entry holds immutable configuration and cubins, never live
    # CUDA addresses. Resident JIT code does not need filesystem scans on each
    # call. CuTe resolves its compiler environment at initialization; identity
    # is audited on preparation/compilation misses, not by rescanning os.environ.
    key = (config, buffer.layout, stats is not None, context.dispatch_sms,
           context.compute_sms, context.schedule_policy)
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                schedule = get_sm90_role_config(config, context)
                identity = _compiler_identity('sm_90a')
                specs = _role_specs(config, buffer.layout, schedule, stats is not None)
                binaries = _get_or_compile_binaries(_binary_key(identity, specs), identity, specs)
                prepared = _Prepared(binaries)
                _CACHE[key] = prepared
    routed_addresses = []
    # Exponent pointers alias record storage; N512 gathers them separately.
    for record, secondary in (w1, w2):
        payload = record.data_ptr()
        routed_addresses.extend((payload, payload + 16384, secondary.data_ptr()))
    shared_addresses = ([t.data_ptr() for pair in (shared_l1_weights, shared_l2_weights)
                         for t in pair] if config.shared_experts else [0, 0, 0, 0])
    addresses = [buffer.buffer.data_ptr(), buffer.peer_table.data_ptr(), y.data_ptr(),
                 *routed_addresses, 0 if stats is None else stats.data_ptr(), *shared_addresses]
    arguments = [*(ctypes.c_void_p(p) for p in addresses),
                 ctypes.c_uint32(buffer.rank_idx), ctypes.c_uint32(config.tokens),
                 ctypes.c_float(config.clamp)]
    return prepared.binaries, arguments


def launch_sm90_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_l1_weights=None, shared_l2_weights=None):
    context = get_context()
    binaries, arguments = prepare_role_binaries(
        config, y, w1, w2, buffer, stats, context,
        shared_l1_weights, shared_l2_weights)
    context.launch(binaries, arguments)
