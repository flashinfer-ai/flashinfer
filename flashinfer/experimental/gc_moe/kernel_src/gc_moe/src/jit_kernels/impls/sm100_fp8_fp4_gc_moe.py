"""Prepare the three CuTe GC role cubins and live TensorMaps for a call.

The supported H/I and token ranges are checked before graph launch. Each
role's PTX/cubin export is owned by its compile, including concurrent ranks.
"""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import inf
import os
from pathlib import Path
import re

from cutlass import Float32, Uint32
from cutlass.cute.runtime import from_dlpack
import torch
from cuda.bindings import driver as cu

from ...comm.gc_barrier import DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
from ...common.tma_copy import use_small_tile_path
from ...impls.sm100_fp8_fp4_gc_moe import GCMoERoleKernel
from ...layout.gc_moe import (
    MAX_RANKS,
    align,
    ceil_div,
)
from ...runtime.gc_moe import RoleBinary, get_context
from ...runtime.jit import (
    COMPILE_LOCK,
    PTXAS_OPTIONS,
    CompilerIdentity,
    assemble_ptx,
    compile_kernel,
    compiler_identity,
    device_compile_target,
    ptx_target,
)
from ..heuristics.gc_moe import (
    get_combine_geometry,
    get_role_config,
    num_max_live_pool_blocks,
)
from .runtime_utils import (
    TmaDescriptorSpec,
    make_tma_2d_desc,
    make_tma_sf_desc,
    make_tma_weights_2d_desc,
    make_tma_weights_3d_desc,
)


@dataclass(frozen=True)
class GCTmaPlan:
    """Original eighteen maps, optionally followed by four fused-SE prefix maps."""

    l1_acts: TmaDescriptorSpec
    l1_acts_sf: TmaDescriptorSpec
    l1_weights: TmaDescriptorSpec
    l1_weights_sf: TmaDescriptorSpec
    l1_output: TmaDescriptorSpec
    l2_acts: TmaDescriptorSpec
    l2_acts_sf: TmaDescriptorSpec
    l2_weights: TmaDescriptorSpec
    l2_weights_sf: TmaDescriptorSpec
    shared_l1_acts: TmaDescriptorSpec
    shared_l1_acts_sf: TmaDescriptorSpec
    shared_l1_weights: TmaDescriptorSpec
    shared_l1_weights_sf: TmaDescriptorSpec
    shared_l1_output: TmaDescriptorSpec
    shared_l2_acts: TmaDescriptorSpec
    shared_l2_acts_sf: TmaDescriptorSpec
    shared_l2_weights: TmaDescriptorSpec
    shared_l2_weights_sf: TmaDescriptorSpec

    l1_acts_16: TmaDescriptorSpec | None = None
    l2_acts_16: TmaDescriptorSpec | None = None
    l1_acts_24: TmaDescriptorSpec | None = None
    l2_acts_24: TmaDescriptorSpec | None = None

    def ordered(self) -> tuple[TmaDescriptorSpec, ...]:
        values = tuple(getattr(self, name) for name in self.__dataclass_fields__)
        optional = values[18:]
        if any(value is None for value in optional):
            if not all(value is None for value in optional):
                raise ValueError('prefix maps must be all absent or all present')
            return values[:18]
        return values


def build_gc_tma_plan(config, sym_buffer, l1_weights, l2_weights,
                      shared_l1_weights=None, shared_l2_weights=None,
                      shared_weight_gran_k: int = 32) -> GCTmaPlan:
    """Describe routed/shared tensors; encode with live pointers at launch.

    Each weight tuple's [0] tensor is the tile-repacked payload prepared by
    the caller; scale factors and activations use their original storage.
    """
    bk, bn = config.block_k, config.block_n
    h, i = sym_buffer.hidden, sym_buffer.intermediate_hidden
    sf_m = config.sf_block_m
    ring, sf_ring = sym_buffer.layout.ring_tokens, sym_buffer.layout.sf_ring_tokens
    shared_count = sym_buffer.num_shared_experts
    shared_i = shared_count * i
    if shared_weight_gran_k not in (32, 128):
        raise ValueError('shared UE8M0 K granularity must be 32 or 128')
    if bool(shared_l1_weights) != bool(shared_l2_weights) or \
            bool(shared_l1_weights) != bool(shared_count):
        raise ValueError('shared tensors and expert count disagree')
    def acts(t, inner, outer, block_inner, block_outer, swizzle):
        return make_tma_2d_desc(t, inner, outer, block_inner, block_outer,
                         t.stride(-2), swizzle)

    def scales(t, mn, k, block_mn, gran, groups):
        return make_tma_sf_desc(t, mn, k, block_mn, gran, groups,
                     (bk + gran * 4 - 1) // (gran * 4), t.stride(-1))

    routed = (
        acts(sym_buffer.l1_acts, h, ring, bk, config.load_m, config.swizzle_acts),
        scales(sym_buffer.l1_acts_sf, sf_ring, h, sf_m, 32, 1),
        make_tma_weights_3d_desc(l1_weights[0], bk, bn, swizzle=config.swizzle_weights),
        scales(l1_weights[1], 2 * i, h, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
        acts(sym_buffer.l2_acts, i, ring, bn // 2,
             config.store_m_l1, config.swizzle_output),
        acts(sym_buffer.l2_acts, i, ring, bk, config.load_m, config.swizzle_acts),
        scales(sym_buffer.l2_acts_sf, sf_ring, i, sf_m, 32, 1),
        make_tma_weights_3d_desc(l2_weights[0], bk, bn, swizzle=config.swizzle_weights),
        scales(l2_weights[1], h, i, bn, 32,
               sym_buffer.num_experts // sym_buffer.group.size()),
    )
    if shared_count:
        shared_sf_tokens = sym_buffer.shared_l1_acts_sf.shape[0]
        shared = (
            acts(sym_buffer.shared_l1_acts, h,
                 sym_buffer.num_max_tokens_per_rank, bk, config.load_m, config.swizzle_acts),
            scales(sym_buffer.shared_l1_acts_sf, shared_sf_tokens,
                   h, sf_m, 32, 1),
            make_tma_weights_2d_desc(shared_l1_weights[0], bk, bn, swizzle=config.swizzle_weights),
            scales(shared_l1_weights[1], 2 * shared_i, h, bn,
                   shared_weight_gran_k, 1),
            acts(sym_buffer.shared_l2_acts, shared_i,
                 sym_buffer.num_max_tokens_per_rank, bn // 2,
                 config.store_m_l1, config.swizzle_output),
            acts(sym_buffer.shared_l2_acts, shared_i,
                 sym_buffer.num_max_tokens_per_rank, bk, config.load_m, config.swizzle_acts),
            scales(sym_buffer.shared_l2_acts_sf,
                   sym_buffer.shared_l2_acts_sf.shape[0],
                   shared_i, sf_m, 32, 1),
            make_tma_weights_2d_desc(shared_l2_weights[0], bk, bn, swizzle=config.swizzle_weights),
            scales(shared_l2_weights[1], h, shared_i, bn,
                   shared_weight_gran_k, 1),
        )
    else:
        # Supply routed maps in unused shared positions.
        shared = routed
    narrow = tuple(acts(t, inner, ring, bk, min(rows, config.load_m),
                        config.swizzle_acts)
                   for rows in (16, 24)
                   for t, inner in ((sym_buffer.l1_acts, h),
                                    (sym_buffer.l2_acts, i))) \
        if use_small_tile_path(shared_count, config.load_m * 2) else ()
    return GCTmaPlan(*routed, *shared, *narrow)

def _get_gc_moe_nvlink_barrier_timeout_secs() -> int:
    """Read the positive signed-decimal int32 barrier timeout."""
    name = 'GC_MOE_NVLINK_BARRIER_TIMEOUT_SECS'
    value = os.getenv(name)
    if value is None or value == '':
        return DEFAULT_NVLINK_BARRIER_TIMEOUT_SECS
    if re.fullmatch(r"[ \t\n\v\f\r]*[+-]?[0-9]+", value, re.ASCII) is None:
        raise ValueError(f"Invalid {name}: {value}")
    try:
        seconds = int(value, 10)
    except ValueError as exc:
        raise ValueError(f"Invalid {name}: {value}") from exc
    if not 0 < seconds <= 2_147_483_647:
        raise ValueError(f"Invalid {name}: {value}")
    return seconds


def _routed_weight_linger(config, num_topk: int) -> bool:
    """Linger when the uniform-routing estimate gives an expert two or more m-blocks.

    The gate compares average tokens per expert against the tile, so skewed
    routing can leave some experts below one m-block and others above it.
    """
    tokens_per_expert = ceil_div(config.tokens * num_topk, config.local_experts)
    return tokens_per_expert >= 2 * config.block_m


def _compiler_identity(target: str) -> CompilerIdentity:
    return compiler_identity(target, GCMoERoleKernel.__call__,
                             Path(__file__).resolve().parents[2], 'gc_moe_role')


@dataclass
class _Prepared:
    maps: bytes
    binaries: tuple[RoleBinary, RoleBinary, RoleBinary]
    # Hold both payload and SF storage for every graph using these maps.
    keepalive: tuple = ()


# Lifetime owner of loaded weights used by TensorMaps and child/outer graphs.
# Entries live until release_workspace retires a workspace and its graphs.
_CACHE: dict[tuple, _Prepared] = {}
_BINARY_CACHE: dict[tuple, tuple[RoleBinary, RoleBinary, RoleBinary]] = {}


def _get_or_compile_binaries(key: tuple, identity: CompilerIdentity,
                             target: str, compile_roles) -> tuple[RoleBinary, RoleBinary, RoleBinary]:
    """Publish a complete cubin set once, after compiler identity is stable."""
    with COMPILE_LOCK:
        binaries = _BINARY_CACHE.get(key)
        if binaries is None:
            binaries = tuple(compile_roles())
            if len(binaries) != 3 or tuple(item.role for item in binaries) != (1, 2, 3):
                raise RuntimeError('GC role compilation did not produce all three roles')
            if _compiler_identity(target) != identity:
                raise RuntimeError('compiler identity changed while compiling role binaries')
            _BINARY_CACHE[key] = binaries
        return binaries


class _SymBufferValue(ctypes.Structure):
    """592-byte by-value symmetric peer-buffer launch argument."""
    _fields_ = (('base', ctypes.c_int64),
                ('offsets', ctypes.c_int64 * 72),
                ('rank_idx', ctypes.c_uint32))


def _sym_buffer_value(sym_buffer) -> _SymBufferValue:
    addresses = tuple(int(ptr) for ptr in sym_buffer.peer_addresses)
    rank = int(sym_buffer.rank_idx)
    if (not 1 <= len(addresses) <= 72 or not 0 <= rank < len(addresses) or
            any(ptr <= 0 or ptr >= 1 << 64 or ptr % 16 for ptr in addresses)):
        raise ValueError('invalid symmetric peer address set')
    base = int(sym_buffer.buffer.data_ptr())
    if addresses[rank] != base:
        raise ValueError('local symmetric buffer differs from rendezvous base')
    value = _SymBufferValue()
    value.base = base if base < 1 << 63 else base - (1 << 64)
    for i, address in enumerate(addresses):
        delta = address - base
        if not -(1 << 63) <= delta < (1 << 63):
            raise ValueError('symmetric peer offset exceeds signed int64')
        value.offsets[i] = delta
    value.rank_idx = rank
    if (ctypes.sizeof(value) != 592 or
            (_SymBufferValue.base.offset, _SymBufferValue.offsets.offset,
             _SymBufferValue.rank_idx.offset) != (0, 8, 584)):
        raise AssertionError('SymBuffer by-value ABI changed')
    return value


def _clean_cute_ptx(ptx: str) -> str:
    if ptx.endswith("\x00"):
        ptx = ptx[:-1]
    if "\x00" in ptx:
        raise RuntimeError('unexpected NUL inside CuTe PTX')
    return ptx


def _rewrite_value_formal(ptx: str, ordinal: int, value_bytes: int,
                          alignment: int, *, raw_param_address: bool = False) -> str:
    """Replace one audited 8B CuTe pointer formal with one value formal."""
    declaration = re.compile(
        r"(?m)^([ \t]*\.param[ \t]+\.align[ \t]+8[ \t]+\.b8[ \t]+)"
        rf"([A-Za-z0-9_.$]+_param_{ordinal})\[8\](,?)[ \t]*$")
    matches = list(declaration.finditer(ptx))
    if len(matches) != 1:
        raise RuntimeError(f"CuTe value formal {ordinal} changed")
    name = matches[0].group(2)
    uses = re.findall(r"(?<![A-Za-z0-9_.$])" + re.escape(name) +
                      r"(?![A-Za-z0-9_.$])", ptx)
    if len(uses) != 2:
        raise RuntimeError(f"CuTe value formal {ordinal} has unexpected uses")
    load = re.compile(
        r"(?m)^([ \t]*)ld\.param\.(?:u64|b64)[ \t]+(%rd[0-9]+),[ \t]*\[" +
        re.escape(name) + r"\];[ \t]*$")
    if len(list(load.finditer(ptx))) != 1:
        raise RuntimeError(f"CuTe value formal {ordinal} pointer load changed")
    ptx = declaration.sub(
        lambda m: re.sub(r"\.align[ \t]+8\b", f".align {alignment}",
                         m.group(1)) + f"{name}[{value_bytes}]{m.group(3)}",
        ptx, count=1)
    # The reference keeps by-value TensorMap addresses in param space across
    # task-phase selection. The peer value is accessed through a generic
    # address, so only the role2 maps use the raw form here.
    address_op = 'mov.b64' if raw_param_address else 'cvta.param::entry.u64'
    ptx = load.sub(
        lambda m: f"{m.group(1)}{address_op} {m.group(2)}, {name};",
        ptx, count=1)
    return ptx


def _convert_role2_map_uses(ptx: str, map_count: int) -> str:
    """Convert early prefetches and W1 output stores at PTX instruction uses.

    CuTe exposes map parameters as pointers. After the by-value formal rewrite
    those registers contain *param-space* addresses, as in the reference.
    Loader warps convert their two selected maps once per task before the K
    loop. This bridge converts only the earlier prefetches and the per-store
    W1 output map, which the reference also selects inside its elected store.
    Every conversion is scalar; no elected lane executes a warp collective.

    Fail closed on changes to CuTe's generated descriptor instruction forms.
    This bridge deliberately does not depend on PTX register IDs or labels.
    """
    map_ordinals = (4, *range(7, map_count + 6))
    raw_map_regs = []
    for ordinal in map_ordinals:
        raw = re.findall(
            r"(?m)^[ \t]*mov\.b64[ \t]+(%rd[0-9]+),[ \t]*"
            r"[A-Za-z0-9_.$]+_param_" + str(ordinal) + r";[ \t]*$", ptx)
        if len(raw) != 1:
            raise RuntimeError(f"role2 raw TensorMap {ordinal} changed")
        raw_map_regs.append(raw[0])
    if len(set(raw_map_regs)) != map_count:
        raise RuntimeError('role2 TensorMap address registers alias')

    register_decl = re.compile(
        r"(?m)^([ \t]*\.reg[ \t]+\.b64[ \t]+%rd<)([0-9]+)(>;[ \t]*)$")
    declarations = list(register_decl.finditer(ptx))
    if len(declarations) != 1:
        raise RuntimeError('role2 64-bit PTX register declaration changed')
    next_reg = int(declarations[0].group(2))
    prefetch = re.compile(
        r"(?m)^([ \t]*prefetch\.tensormap[ \t]+\[)(%rd[0-9]+)"
        r"(\];[ \t]*)$")
    g2s = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.(?:2d|4d)\.cta_group::2"
        r"[^\n]*?\[[^\n]*?\],[ \t]*\[)(%rd[0-9]+)"
        r"(,[ \t]*\{[^\n]+)$")
    s2g = re.compile(
        r"(?m)^([ \t]*cp\.async\.bulk\.tensor\.2d\.global\.shared"
        r"[^\n]*?\[)(%rd[0-9]+)(,[ \t]*\{[^\n]+)$")
    prefetch_regs = [m.group(2) for m in prefetch.finditer(ptx)]
    if len(prefetch_regs) != map_count or set(prefetch_regs) != set(raw_map_regs):
        raise RuntimeError('role2 TensorMap prefetch set changed')
    tensor_op_count = len(re.findall(
        r"(?m)^[ \t]*cp\.async\.bulk\.tensor\.", ptx))
    g2s_count = len(list(g2s.finditer(ptx)))
    s2g_count = len(list(s2g.finditer(ptx)))
    if g2s_count != 16 or s2g_count < 1 or tensor_op_count != \
            g2s_count + s2g_count:
        raise RuntimeError('role2 TensorMap load/store PTX forms changed')
    selected_conversions = re.findall(
        r"(?m)^[ \t]*cvta\.param\.u64[ \t]+(%rd[0-9]+),[ \t]*"
        r"(%rd[0-9]+);[ \t]*$", ptx)
    loader_regs = {m.group(2) for m in g2s.finditer(ptx)}
    if (len(selected_conversions) != 4 or
            len(loader_regs) != 4 or
            loader_regs != {dst for dst, _ in selected_conversions} or
            len({src for _, src in selected_conversions}) != 4 or
            any(src in raw_map_regs for _, src in selected_conversions)):
        raise RuntimeError('role2 loader selected-map conversions changed')
    source_cvta_count = len(selected_conversions)

    def converted_use(match: re.Match[str]) -> str:
        nonlocal next_reg
        prefix, raw_addr, suffix = match.groups()
        generic = f"%rd{next_reg}"
        next_reg += 1
        indent = re.match(r"[ \t]*", prefix).group(0)
        return (f"{indent}cvta.param.u64 {generic}, {raw_addr};\n"
                f"{prefix}{generic}{suffix}")

    ptx = prefetch.sub(converted_use, ptx)
    ptx = s2g.sub(converted_use, ptx)
    ptx = register_decl.sub(
        lambda m: m.group(1) + str(next_reg) + m.group(3), ptx, count=1)
    if len(re.findall(r"\bcvta\.param\.u64\b", ptx)) != \
            source_cvta_count + map_count + s2g_count:
        raise RuntimeError('role2 TensorMap conversion count changed')
    return ptx


def _device_compile_target(device: torch.device) -> str:
    major, minor = torch.cuda.get_device_capability(device)
    if (major, minor) not in ((10, 0), (10, 3)):
        raise NotImplementedError('current three-role device path requires compute capability 10.0 or 10.3')
    return device_compile_target((major, minor))


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


def _order_role2_value_formals(ptx: str, map_count: int) -> str:
    """Keep ordinary scalars ahead of the configuration-selected value maps."""
    formals = _entry_formals(ptx)
    if len(formals) != map_count + 6:
        raise RuntimeError('role2 value-formal count changed')
    for index, line in enumerate(formals):
        if re.search(r"\b[A-Za-z0-9_.$]+_param_" + str(index) +
                     r"(?:\[[0-9]+\])?,?[ \t]*$", line) is None:
            raise RuntimeError('role2 value-formal order changed')
    order = (0, 1, 2, 3, 5, 6, 4, *range(7, map_count + 6))
    ordered = [formals[i].rstrip().rstrip(',') for i in order]
    original = '\n'.join(formals)
    if ptx.count(original) != 1:
        raise RuntimeError('role2 value-formal signature changed')
    return ptx.replace(original, ',\n'.join(ordered), 1)


def _assemble_role_value_params(ptx: str, role: int,
                                dump_dir: str, *, target: str,
                                use_small_tile_path: bool,
                                ptxas_options: tuple[str, ...] = PTXAS_OPTIONS) -> tuple[str, bytes, str]:
    """Build each role with relative peer value, plus role2's value maps."""
    map_count = 22 if use_small_tile_path else 18
    ptx = _clean_cute_ptx(ptx)
    if ptx_target(ptx) != target:
        raise RuntimeError('CuTe generated a different device target')
    expected = {1: ('ptr', 'ptr', 'u32', 'u32', 'f32', 'f32'),
                2: (('ptr', 'ptr', 'u32', 'u32', 'ptr', 'f32', 'f32') +
                    ('ptr',) * (map_count - 1)),
                3: ('ptr', 'ptr', 'u32', 'u32', 'ptr', 'ptr', 'ptr',
                    'f32', 'f32')}[role]
    formals = _entry_formals(ptx)
    if len(formals) != len(expected):
        raise RuntimeError(f"role{role} CuTe parameter count changed")
    for index, (line, kind) in enumerate(zip(formals, expected)):
        form = (r"\.align[ \t]+8[ \t]+\.b8[ \t]+\S+_param_" +
                str(index) + r"\[8\]" if kind == 'ptr' else
                r"\." + kind + r"[ \t]+\S+_param_" + str(index))
        if re.search(form + r",?[ \t]*$", line) is None:
            raise RuntimeError(f"role{role} CuTe formal {index} changed")
    if 'cvta.to.param::entry.u64' not in ptx or 'ld.param.u64' not in ptx:
        raise RuntimeError(f"role{role} peer parameter load path is missing")
    ptx = _rewrite_value_formal(ptx, 1, 592, 8)
    if role == 2:
        for map_ordinal in (4, *range(7, map_count + 6)):
            ptx = _rewrite_value_formal(
                ptx, map_ordinal, 128, 128, raw_param_address=True)
        ptx = _convert_role2_map_uses(ptx, map_count)
        ptx = _order_role2_value_formals(ptx, map_count)
    return assemble_ptx(ptx, dump_dir, f"role{role}_value_params",
                                target=target, ptxas_options=ptxas_options)


def _capture_key(config, sym_buffer, green_context, y, l1_weights, l2_weights,
                 shared_l1_weights, shared_l2_weights, stats,
                 nvlink_timeout_secs: int, routed_weight_linger: bool,
                 role_resources: tuple,
                 combine_resources: tuple, compiler_identity: CompilerIdentity) -> tuple:
    status, cuda_context = cu.cuCtxGetCurrent()
    if status != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"cannot query CUDA context: {status}")
    layout = sym_buffer.layout
    weights = (*l1_weights, *l2_weights,
               *(shared_l1_weights or ()), *(shared_l2_weights or ()))
    return (int(cuda_context), compiler_identity, green_context.dispatch_sms,
            green_context.compute_sms,
            sym_buffer.buffer.data_ptr(),
            tuple(sym_buffer.peer_addresses), sym_buffer.rank_idx,
            sym_buffer.group.size(), sym_buffer.num_experts,
            sym_buffer.num_max_tokens_per_rank,
            config.tokens, config.block_m, config.hidden, config.intermediate,
            config.local_experts, config.shared_experts,
            config.shared_weight_gran_k, config.clamp != inf,
            config.fast_math, sym_buffer.num_topk,
            nvlink_timeout_secs,
            routed_weight_linger,
            stats is not None,
            role_resources, combine_resources,
            layout.nbytes, layout.ring_tokens, layout.sf_ring_tokens,
            _tensor_metadata(y),
            _tensor_metadata(stats) if stats is not None else None,
            *(item for tensor in weights
              for item in (tensor.data_ptr(), tuple(tensor.shape),
                           tuple(tensor.stride()), str(tensor.dtype))))


def _tensor_metadata(tensor: torch.Tensor) -> tuple:
    return (tuple(tensor.shape), tuple(tensor.stride()), str(tensor.dtype))


def _binary_key(compiler_identity: CompilerIdentity, role_kernels: tuple,
                role_resources: tuple, combine_resources: tuple,
                sym_buffer, y: torch.Tensor, weights: tuple,
                stats: torch.Tensor | None) -> tuple:
    """Identify code generation without retaining any live CUDA address.

    Every role kernel attribute is included so new constexpr or derived layout
    fields cannot silently reuse an older cubin. Tensor metadata covers the
    CuTe compile argument types; actual addresses remain in _capture_key.
    """
    key = (compiler_identity,
           tuple(tuple(sorted(vars(kernel).items())) for kernel in role_kernels),
           role_resources, combine_resources,
           sym_buffer.layout.nbytes,
           _tensor_metadata(sym_buffer.buffer), _tensor_metadata(y),
           tuple(_tensor_metadata(tensor) for tensor in weights),
           _tensor_metadata(stats) if stats is not None else None)
    return key


def _check_supported(config, sym_buffer) -> None:
    if (not 1 <= sym_buffer.group.size() <= MAX_RANKS or
            not 1 <= config.local_experts <= 512 or
            sym_buffer.group.size() * config.local_experts > 2048 or
            not 0 < sym_buffer.num_topk or
            sym_buffer.num_topk + bool(config.shared_experts) > 32 or
            config.shared_experts < 0):
        raise NotImplementedError('current CuTe device path needs 1..512 experts/rank, <=2048 total experts, topk>=1 and topk+shared slot<=32')
    if (config.hidden <= 0 or config.intermediate <= 0 or
            config.hidden % 256 != 0 or config.intermediate % 128 != 0 or
            (2 * config.intermediate) // 128 > 64 or
            config.block_m not in (16, 32, 64, 96, 128, 192, 240) or
            config.tokens < 0):
        raise NotImplementedError('current CuTe device math slice needs 2CTA H/I geometry, <=64 W1 N blocks, supported BM and nonnegative T')


def _check_ring_capacity(config, sym_buffer, compute_sms: int) -> None:
    """Recheck the full-device ring against the actual Green compute partition."""
    layout = sym_buffer.layout
    block_m = config.block_m
    if layout.ring_tokens % block_m:
        raise ValueError('GC ring capacity must contain whole M blocks')
    max_routed = (sym_buffer.num_max_tokens_per_rank * sym_buffer.group.size() *
                  min(sym_buffer.num_topk, config.local_experts))
    max_pool_blocks = ceil_div(max_routed, block_m) + config.local_experts
    live_blocks = num_max_live_pool_blocks(
        max_pool_blocks, compute_sms, config.hidden, config.intermediate)
    ring_blocks = layout.ring_tokens // block_m
    if live_blocks > ring_blocks:
        raise ValueError('GC ring capacity is insufficient for Green compute SMs')
    if layout.sf_ring_tokens < ring_blocks * align(block_m, 128):
        raise ValueError('GC SF ring capacity is insufficient for block M')


def prepare_role_binaries(config, y: torch.Tensor,
                          l1_weights: tuple[torch.Tensor, torch.Tensor],
                          l2_weights: tuple[torch.Tensor, torch.Tensor],
                          sym_buffer, shared_l1_weights=None,
                          shared_l2_weights=None, stats=None,
                          context=None):
    """Encode live maps once per pointer set and compile each role once/shape."""
    _check_supported(config, sym_buffer)
    small_tile_path = use_small_tile_path(config.shared_experts, config.block_m)
    map_count = 22 if small_tile_path else 18
    if config.shared_experts != sym_buffer.num_shared_experts:
        raise ValueError('configuration and symmetric-buffer shared experts disagree')
    if context is None:
        context = get_context(num_ranks=sym_buffer.group.size())
    device = sym_buffer.buffer.device
    weight_pairs = (l1_weights, l2_weights)
    if shared_l1_weights is not None:
        weight_pairs += (shared_l1_weights, shared_l2_weights)
    weights = tuple(tensor for pair in weight_pairs for tensor in pair)
    tensors = (y, *weights,
               *((stats,) if stats is not None else ()))
    if (device.type != 'cuda' or torch.cuda.current_device() != device.index or
            int(context.device) != device.index or any(tensor.device != device for tensor in tensors)):
        raise ValueError('role tensors and Green contexts must use the current CUDA device')
    _check_ring_capacity(config, sym_buffer, context.compute_sms)
    compiler_target = _device_compile_target(sym_buffer.buffer.device)
    compiler_identity = _compiler_identity(compiler_target)
    nvlink_timeout_secs = _get_gc_moe_nvlink_barrier_timeout_secs()
    routed_weight_linger = _routed_weight_linger(config, sym_buffer.num_topk)
    role_configs = tuple(get_role_config(role, config.block_m,
                                         sym_buffer.num_experts,
                                         config.hidden,
                                         local_experts=config.local_experts)
                         for role in (1, 2, 3))
    role_resources = tuple((item.num_stages, item.dynamic_smem, item.threads)
                           for item in role_configs)
    role3_config = role_configs[2]
    role3_layout = role3_config.shared_layout
    combine_reusable_bytes = role3_layout.reusable_bytes
    if role3_layout.field_end > role3_config.dynamic_smem:
        raise ValueError('role3 shared storage exceeds its launch allocation')
    combine_geometry = get_combine_geometry(
        config.hidden, sym_buffer.num_topk +
        (1 if sym_buffer.num_shared_experts else 0),
        combine_reusable_bytes, role3_config.dynamic_smem)
    combine_chunks, combine_chunk_bytes, _ = combine_geometry
    combine_resources = (combine_reusable_bytes, combine_chunks,
                         combine_chunk_bytes)
    key = _capture_key(config, sym_buffer, context, y, l1_weights, l2_weights,
                       shared_l1_weights, shared_l2_weights, stats,
                       nvlink_timeout_secs, routed_weight_linger,
                       role_resources, combine_resources,
                       compiler_identity)
    prepared = _CACHE.get(key)
    if prepared is None:
        with COMPILE_LOCK:
            prepared = _CACHE.get(key)
            if prepared is None:
                layout = sym_buffer.layout
                tma_plan = build_gc_tma_plan(role_configs[1], sym_buffer,
                                             l1_weights, l2_weights,
                                             shared_l1_weights, shared_l2_weights,
                                             config.shared_weight_gran_k)
                payloads = []
                for i, spec in enumerate(tma_plan.ordered()):
                    descriptor = spec.encode()
                    payload = ctypes.string_at(descriptor.getPtr(), 128)
                    if len(payload) != 128:
                        raise RuntimeError(f"TensorMap {i} is not 128 bytes")
                    payloads.append(payload)
                if len(payloads) != map_count:
                    raise RuntimeError('GC role2 TensorMap count disagrees with its configuration')
                maps = b"".join(payloads)
                role_kernels = []
                for index, role in enumerate((1, 2, 3)):
                    role_kernel = GCMoERoleKernel(
                        role, context.dispatch_sms, context.compute_sms,
                        config.block_m, config.hidden, config.intermediate,
                        config.tokens, sym_buffer.group.size(), config.local_experts,
                        config.shared_experts,
                        sym_buffer.num_max_tokens_per_rank,
                        finalize_output=role == 3,
                        ring_tokens=layout.ring_tokens,
                        sf_ring_tokens=layout.sf_ring_tokens,
                        topk=sym_buffer.num_topk,
                        shared_weight_gran_k=config.shared_weight_gran_k,
                        # Only the clamp-presence flag specializes the binary. The
                        # actual FP32 value is a kernel argument on every replay.
                        activation_clamp=0.0 if config.clamp != inf else inf,
                        fast_math=config.fast_math,
                        has_stats=stats is not None,
                        nvlink_timeout_secs=nvlink_timeout_secs,
                        routed_weight_linger=routed_weight_linger,
                        role_config=role_configs[index],
                        combine_geometry=combine_geometry if role == 3 else None)
                    role_kernels.append(role_kernel)
                binary_key = _binary_key(
                    compiler_identity, tuple(role_kernels), role_resources,
                    combine_resources, sym_buffer, y, weights, stats)
                def compile_roles():
                    signal_view = from_dlpack(sym_buffer.buffer.view(torch.int32))
                    peer_view = from_dlpack(sym_buffer.buffer.view(torch.int64).flatten()[:74])
                    # These views supply CuTe's pointer type/shape only. Every
                    # TensorMap pointer load becomes a host-supplied value param.
                    map_storage = sym_buffer.buffer.view(torch.uint8).flatten()
                    map_views = tuple(from_dlpack(map_storage[i * 128:(i + 1) * 128])
                                      for i in range(map_count))
                    role3_map_view = from_dlpack(map_storage[:2304])
                    y_view = from_dlpack(y)
                    stats_view = from_dlpack(stats) if stats is not None else signal_view
                    rank_view, token_view = Uint32(0), Uint32(0)
                    clamp_view, alpha_view = Float32(0.0), Float32(1.0)
                    compiled_binaries = []
                    for index, (role, role_kernel) in enumerate(zip((1, 2, 3), role_kernels)):
                        compile_args = ((signal_view, peer_view, rank_view, token_view,
                                         map_views[0], None, None,
                                         clamp_view, alpha_view, *map_views[1:])
                                        if role == 2 else
                                        (signal_view, peer_view, rank_view, token_view,
                                         role3_map_view, y_view, stats_view,
                                         clamp_view, alpha_view)
                                        if role == 3 else
                                        (signal_view, peer_view, rank_view, token_view,
                                         None, None, None,
                                         clamp_view, alpha_view))
                        cubin = compile_kernel(
                            role_kernel, compile_args, compiler_identity,
                            lambda ptx, dump_dir: _assemble_role_value_params(
                                ptx, role, dump_dir, target=compiler_target,
                                use_small_tile_path=small_tile_path,
                                ptxas_options=compiler_identity.ptxas_options))
                        role_config = role_configs[index]
                        compiled_binaries.append(RoleBinary(
                            role, cubin, role_config.dynamic_smem,
                            role_config.threads,
                            context.dispatch_sms if role == 1 else
                            context.compute_sms if role == 2 else
                            role_kernel.finalize_sms))
                    return compiled_binaries

                binaries = _get_or_compile_binaries(
                    binary_key, compiler_identity, compiler_target, compile_roles)
                prepared = _Prepared(maps, binaries, weights)
                if _compiler_identity(compiler_target) != compiler_identity:
                    raise RuntimeError('compiler identity changed while preparing role binaries')
                _CACHE[key] = prepared
    if len(prepared.maps) != map_count * 128:
        raise RuntimeError('cached TensorMap payload disagrees with its configuration')
    signal_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    peer_arg = _sym_buffer_value(sym_buffer)
    rank_arg = ctypes.c_uint32(sym_buffer.rank_idx)
    token_arg = ctypes.c_uint32(config.tokens)
    clamp_arg = ctypes.c_float(config.clamp)
    alpha_arg = ctypes.c_float(config.alpha)
    map_args = tuple((ctypes.c_uint8 * 128).from_buffer_copy(
        prepared.maps[i * 128:(i + 1) * 128]) for i in range(map_count))
    # Role3 retains its original 8-byte unused CuTe pointer formal. Role2
    # carries the configuration-selected 128-byte descriptor values.
    role3_map_arg = ctypes.c_void_p(sym_buffer.buffer.data_ptr())
    y_arg = ctypes.c_void_p(y.data_ptr())
    stats_arg = ctypes.c_void_p(stats.data_ptr() if stats is not None else
                               sym_buffer.buffer.data_ptr())
    arguments = ((signal_arg, peer_arg, rank_arg, token_arg,
                  clamp_arg, alpha_arg),
                 (signal_arg, peer_arg, rank_arg, token_arg,
                  clamp_arg, alpha_arg, *map_args),
                 (signal_arg, peer_arg, rank_arg, token_arg, role3_map_arg,
                  y_arg, stats_arg, clamp_arg, alpha_arg))
    return list(prepared.binaries), arguments

def launch_gc_moe(config, y, l1_weights, l2_weights, sym_buffer,
                  shared_l1_weights, shared_l2_weights, stats) -> None:
    context = get_context(num_ranks=sym_buffer.group.size())
    binaries, arguments = prepare_role_binaries(
        config, y, l1_weights, l2_weights, sym_buffer,
        shared_l1_weights, shared_l2_weights, stats, context)
    context.launch(binaries, arguments)


def release_workspace(buffer):
    """Release TensorMap and weight entries owned by a destroyed workspace."""
    address = buffer.buffer.data_ptr()
    for key in tuple(_CACHE):
        # _capture_key stores the local symmetric buffer pointer at index 4.
        if key[4] == address:
            _CACHE.pop(key, None)
