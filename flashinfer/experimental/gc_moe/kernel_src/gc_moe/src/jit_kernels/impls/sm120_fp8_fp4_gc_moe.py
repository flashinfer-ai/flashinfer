"""SM120 role preparation and by-value TensorMap launch arguments."""
from __future__ import annotations

import ctypes
from dataclasses import dataclass
from pathlib import Path
import re

import torch
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Uint8, Uint32, Uint64
from cutlass.cute.runtime import make_ptr

from ...impls.sm120_fp8_fp4_gc_moe import SM120GCMoERoleKernel
from ...runtime.sm120_gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
)
from ..heuristics.gc_moe import get_block_m_for_gc_moe
from ..heuristics.sm120_gc_moe import get_sm120_role_config
from .runtime_utils import TmaDescriptorSpec, make_tma_2d_desc, make_tma_weights_3d_desc


@dataclass(frozen=True)
class _Prepared:
    binaries: tuple[RoleBinary, ...]


_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, ...]] = {}
# Encoded TensorMaps plus every tensor whose address they embed.
_MAP_CACHE: dict[tuple, tuple] = {}


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, SM120GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_sm120_role')


def _assemble_sm120_ptx(ptx, dump_dir, basename, identity):
    # CuTe emits a 9.4 header for these kernels. CUDA 13.2 accepts the SM120
    # instruction stream under PTX 9.2 and checks every instruction itself.
    if re.search(r'\brelease[ \t]+13\.2(?=[,\s]|$)', identity.tool_version):
        ptx = re.sub(r'(?m)^\.version[ \t]+9\.4[ \t]*$', '.version 9.2', ptx, count=1)
    return assemble_ptx(ptx, dump_dir, basename, target='sm_120a',
                        ptxas_options=identity.ptxas_options)


def _assemble_role_ptx(ptx, role, dump_dir, identity, reg_cap=0, threads=0,
                       shared_maps=False):
    ptx = _clean_cute_ptx(ptx)
    formals = _entry_formals(ptx)
    pointer_count = len(_POINTER_TYPES)
    if len(formals) != pointer_count + 3:
        raise RuntimeError('SM120 CuTe launch ABI has an unexpected formal count')
    for ordinal, formal in enumerate(formals):
        align = 128 if ordinal in _MAP_FORMALS else 16
        kind = (r'\.u64\s+\.ptr\s+\.global\s+\.align\s+' + str(align) if ordinal < pointer_count
                else r'\.u32' if ordinal < pointer_count + 2 else r'\.f32')
        if re.search(kind + r'\s+\S+_param_' + str(ordinal) + r',?\s*$', formal) is None:
            raise RuntimeError(f'SM120 CuTe launch formal {ordinal} changed type or order')
    if role == 2:
        if 'mma.sync.aligned.kind::mxf8f6f4' not in ptx:
            raise RuntimeError('SM120 CuTe compute lost its block-scaled MMA')
        if 'cp.async.bulk.tensor' not in ptx:
            raise RuntimeError('SM120 CuTe compute lost its TensorMap loads')
        ordinals = _MAP_FORMALS if shared_maps else _MAP_FORMALS[:4]
        ptx = _pass_tensor_maps_by_value(ptx, formals, ordinals)
    if role == 3 and threads == 384:
        # Match the two-resident-CTA finalize grid used by the scheduler.
        ptx, rewritten = re.subn(
            r'(?m)^\.minnctapersm\s+4\s*$',
            '.minnctapersm 2', ptx, count=1)
        if rewritten != 1:
            raise RuntimeError('SM120 CuTe combine PTX occupancy directive changed')
    if reg_cap:
        # The warp-specialized producer adds enough live values to tip the
        # 160-thread role2 CTA past the 65,536-register file once the
        # allocator's 16-wide per-thread rounding lands (197 -> 208): both
        # the block limit and the driver's L1/shared carveout then collapse
        # to a single CTA per SM for a ~1.5x slower sweep. Rewrite the
        # CuTe-emitted minimum-occupancy directive to the co-residency the
        # SMEM budget intends and pin a 16-aligned register ceiling.
        resident = 65536 // (threads * reg_cap) if threads else 2
        ptx, rewritten = re.subn(
            r'(?m)^\.minnctapersm\s+1\s*$',
            f'.minnctapersm {resident}\n.maxnreg {reg_cap}', ptx, count=1)
        if rewritten != 1:
            raise RuntimeError('SM120 CuTe compute PTX lost its occupancy directive')
    return _assemble_sm120_ptx(ptx, dump_dir, f'sm120_role{role}', identity)


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


def _pass_tensor_maps_by_value(ptx, formals, ordinals):
    """Use the entry parameter's generic address at each TMA instruction."""
    for ordinal in ordinals:
        formal = formals[ordinal]
        name = formal.rstrip().rstrip(',').split()[-1]
        load = re.compile(r'(?m)^([ \t]*)ld\.param\.(?:u64|b64)[ \t]+(%rd[0-9]+),[ \t]*\['
                          + re.escape(name) + r'\];[ \t]*$')
        uses = re.findall(r'(?<![A-Za-z0-9_.$])' + re.escape(name) + r'(?![A-Za-z0-9_.$])', ptx)
        if len(uses) != 2 or len(list(load.finditer(ptx))) != 1:
            raise RuntimeError(f'SM120 CuTe TensorMap formal {ordinal} changed uses')
        comma = ',' if formal.rstrip().endswith(',') else ''
        ptx = ptx.replace(formal, f'    .param .align 128 .b8 {name}[128]{comma}', 1)
        ptx = load.sub(lambda m: f'{m[1]}cvta.param::entry.u64 {m[2]}, {name};', ptx, count=1)
    return ptx


_POINTER_TYPES = (Uint8, Uint64, BFloat16, Uint8, Uint32,
                  Uint8, Uint32, Uint8, Uint8, Uint8, Uint8, Uint32,
                  Uint8, Uint8, Uint8, Uint8, Uint8)
_MAP_FORMALS = (7, 8, 9, 10, 14, 15)


def _role_specs(config, layout, schedule, has_stats, primary_ctx, shared_layout):
    specs = []
    for role in (1, 2, 3):
        kernel = SM120GCMoERoleKernel(config, layout, role, schedule, has_stats,
                                    launch_dependents=primary_ctx and role == 1,
                                    shared_layout=shared_layout)
        specs.append((kernel, (role, schedule.smem[role - 1],
                               schedule.threads[role - 1],
                               schedule.grids[role - 1])))
    return tuple(specs)


def _compile_roles(specs, identity):
    pointers = []
    for ordinal, dtype in enumerate(_POINTER_TYPES):
        align = 128 if ordinal in _MAP_FORMALS else 16
        pointers.append(make_ptr(dtype, 0, cute.AddressSpace.gmem, assumed_align=align))
    pointers = tuple(pointers)
    binaries = []
    for kernel, launch in specs:
        role = launch[0]
        cubin = compile_kernel(
            kernel, (*pointers, Uint32(0), Uint32(0), Float32(0)), identity,
            lambda ptx, dump_dir, reg_cap=kernel.reg_cap, threads=kernel.threads,
                   shared_maps=kernel.shared_tile_n > 0:
                _assemble_role_ptx(ptx, role, dump_dir, identity,
                                   reg_cap, threads, shared_maps))
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
            expected = (1, 2, 3)
            if tuple(item.role for item in binaries) != expected:
                raise RuntimeError('SM120 compilation did not produce all role binaries')
            if _compiler_identity('sm_120a') != identity:
                raise RuntimeError('SM120 compiler/source identity changed while compiling')
            _BINARY_CACHE[key] = binaries
        return binaries


def _make_compact_weight_desc(payload):
    experts, k_tiles, rows, segment = payload.shape
    # The shared encoder maps E4M3FN to CUDA UINT8. Request byte transport
    # for the packed FP4 payload, which the SM120 kernel unpacks itself.
    return TmaDescriptorSpec(
        payload.data_ptr(), torch.float8_e4m3fn, (segment, rows, k_tiles, experts),
        (payload.stride(2), payload.stride(1), payload.stride(0)),
        (segment, 64, 1, 1), 64)


def _get_map_values(buffer, w1, w2, tile_n, compact_w, shared_weights):
    """Encode the role 2 TensorMaps once per live pointer set.

    The descriptors embed global base addresses, so the cache pins the
    workspace and weight tensors to keep those addresses stable.
    """
    layout = buffer.layout
    local, rows = layout.local, layout.rows
    key = (buffer.buffer.data_ptr(), w1[0].data_ptr(), w1[1].data_ptr(),
           w2[0].data_ptr(), w2[1].data_ptr(), tile_n, compact_w,
           tuple(t.data_ptr() for pair in shared_weights for t in pair))
    cached = _MAP_CACHE.get(key)
    if cached is not None:
        return cached[0]
    l1_payload, l2_payload = w1[0], w2[0]
    if compact_w:
        w1_desc = _make_compact_weight_desc(l1_payload)
        w2_desc = _make_compact_weight_desc(l2_payload)
    else:
        w1_desc = make_tma_weights_3d_desc(l1_payload, 128, 64, swizzle=128)
        w2_desc = make_tma_weights_3d_desc(l2_payload, 128, 64, swizzle=128)
    # The X box carries only the CTA token tile's rows, so the TMA moves
    # N x 128 E4M3 bytes and expect_tx matches the kernel's stage budget.
    specs = (
        w1_desc,
        make_tma_2d_desc(buffer.l1_acts.view(local * rows, layout.hidden),
                         layout.hidden, local * rows, 128, tile_n,
                         layout.hidden, 128),
        w2_desc,
        make_tma_2d_desc(buffer.l2_acts.view(local * rows, layout.intermediate),
                         layout.intermediate, local * rows, 128, tile_n,
                         layout.intermediate, 128),
    )
    shared_specs = []
    for payload, _ in shared_weights:
        blocks, weight_rows, _ = payload.shape
        shared_specs.append(make_tma_2d_desc(
            payload, 128, blocks * weight_rows, 128, 64, 128, 128))
    maps = []
    for spec in (*specs, *shared_specs):
        descriptor = spec.encode()
        payload = ctypes.string_at(descriptor.getPtr(), 128)
        if len(payload) != 128:
            raise RuntimeError('SM120 TensorMap is not 128 bytes')
        maps.append((ctypes.c_uint8 * 128).from_buffer_copy(payload))
    map_values = tuple(maps)
    _MAP_CACHE[key] = (map_values, buffer, w1, w2, shared_weights)
    return map_values


def prepare_role_binaries(config, y, w1, w2, buffer, stats=None, context=None,
                          shared_weights=()):
    if torch.cuda.get_device_capability(y.device) != (12, 0):
        raise ValueError('SM120 CuTe GC requires compute capability 12.0')
    if context is None:
        context = get_context()
    schedule = get_sm120_role_config(config, context, buffer.layout)
    shared_layout = None
    if schedule.shared_tile_n:
        k128 = tuple(sf.shape[1] != weight.shape[0] for weight, sf in shared_weights)
        block_m = get_block_m_for_gc_moe(
            config.ranks, config.experts, config.capacity, config.tokens, config.topk)
        shared_layout = (k128, block_m, buffer.shared_l1_acts_sf.shape[0])
    key = (config, buffer.layout, schedule, stats is not None,
           context.primary_ctx_roles, shared_layout)
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                identity = _compiler_identity('sm_120a')
                specs = _role_specs(config, buffer.layout, schedule, stats is not None,
                                    context.primary_ctx_roles, shared_layout)
                binaries = _get_or_compile_binaries(_binary_key(identity, specs), identity, specs)
                prepared = _Prepared(binaries)
                _CACHE[key] = prepared
    map_values = _get_map_values(buffer, w1, w2, schedule.tile_n,
                                 schedule.compact_w, shared_weights)
    addresses = [buffer.buffer.data_ptr(), buffer.peer_table.data_ptr(), y.data_ptr(),
                 w1[0].data_ptr(), w1[1].data_ptr(),
                 w2[0].data_ptr(), w2[1].data_ptr(),
                 0, 0, 0, 0,
                 0 if stats is None else stats.data_ptr()]
    if shared_weights:
        addresses.extend(sf.data_ptr() for _, sf in shared_weights)
        addresses.extend((0, 0, buffer.shared_l1_acts_sf.data_ptr()))
    else:
        addresses.extend((0,) * 5)
    common = tuple(ctypes.c_void_p(p) for p in addresses) + (
        ctypes.c_uint32(buffer.rank_idx), ctypes.c_uint32(config.tokens), ctypes.c_float(config.clamp))
    compute = list(common)
    ordinals = _MAP_FORMALS if schedule.shared_tile_n else _MAP_FORMALS[:4]
    for ordinal, value in zip(ordinals, map_values):
        compute[ordinal] = value
    return prepared.binaries, (common, compute, common)


def launch_sm120_gc_moe(config, y, w1, w2, buffer, stats,
                       shared_w1=None, shared_w2=None):
    context = get_context()
    shared_weights = (shared_w1, shared_w2) if config.shared_experts else ()
    binaries, arguments = prepare_role_binaries(
        config, y, w1, w2, buffer, stats, context, shared_weights)
    context.launch(binaries, arguments)


def release_workspace(buffer):
    from ...runtime.sm120_gc_moe import release_workspace_graphs

    address = buffer.buffer.data_ptr()
    release_workspace_graphs(address)
    for key in tuple(_MAP_CACHE):
        if key[0] == address:
            _MAP_CACHE.pop(key, None)
