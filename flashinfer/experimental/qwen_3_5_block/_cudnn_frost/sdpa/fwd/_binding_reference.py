# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Frozen Python binding oracle, imported only by differential/contract tests.

Production forward execution owns only native binders. Pure geometry helpers
are still shared with the native dense binder; pointer framing lives here only.
"""

from __future__ import annotations
import math
from copy import copy
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import (
    BufferFacts,
    ThdLaunchSpec,
    DenseLaunchSpec,
    _native_quant_roles,
    _covering,
    _sf_byte_count,
    _paged_pool_layout,
    _paged_table_layout,
    _packed_role_layout,
    _dense_role_layout,
    _dense_lse_layout,
    _ragged_lse_layout,
    _DLPACK_CUDA,
    _ALIGN_TMA,
    _ALIGN_F32,
    _I32_MAX,
    _QUANT_ROLES,
    _buffers,
)


class ReferenceThdLaunchSpec(ThdLaunchSpec):
    """Keep the retired geometry cache out of production plan metadata."""

    __slots__ = ("_geometry_cache",)

    def __init__(self):
        self._geometry_cache = None


def _bind_block_output(spec, facts):
    block = spec.quant.block_output
    f = facts.get("sf_o")
    if block is None:
        if f is not None:
            raise ValueError("cudnn.sdpa: this specialization does not produce sf_o")
        return None
    if f is None:
        raise ValueError("cudnn.sdpa: block-scaled output requires sf_o")
    _on_plan_device(spec, "sf_o", f)
    if _buffers.DTYPE_ITEMSIZE.get(f.dtype) != 1:
        raise ValueError("cudnn.sdpa: sf_o requires byte storage")
    _sf_byte_count(f.shape, f.strides, f.dtype)
    # Token-major graph declarations omit the final 128-row atom padding.
    # Its capacity comes from the producer's observed storage, not logical
    # numel. Bare addresses retain the unknown-span caller contract.
    if 0 <= f.span < block.nbytes:
        raise ValueError("cudnn.sdpa: sf_o storage does not cover the compiled atom layout")
    if not f.ptr or f.ptr % 16:
        raise ValueError("cudnn.sdpa: sf_o must be 16-byte aligned")
    for name, other in facts.items():
        if name == "sf_o" or other is None or not other.numel:
            continue
        width = _buffers.DTYPE_ITEMSIZE[other.dtype]
        span = other.span if other.span >= 0 else 1 + sum((n - 1) * st for n, st in zip(other.shape, other.strides))
        if f.ptr < other.ptr + span * width and other.ptr < f.ptr + block.nbytes:
            raise ValueError(f"cudnn.sdpa: sf_o overlaps {name}")
    return f.ptr


def _bind_mxfp8_scales(spec, facts):
    patches, tiles = {}, []
    thd = isinstance(spec, ThdLaunchSpec)
    for name, heads, size in zip(("sf_q", "sf_k", "sf_v"), (spec.qh, spec.kh, spec.kh), spec.quant.sf_sizes):
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: MXFP8 requires {name}")
        _on_plan_device(spec, name, f)
        nbytes = _sf_byte_count(f.shape, f.strides, f.dtype)
        if 0 <= f.span * _buffers.DTYPE_ITEMSIZE[f.dtype] < nbytes:
            raise ValueError(f"cudnn.sdpa: {name} observed storage is too small")
        if nbytes and (not f.ptr or f.ptr % _ALIGN_TMA):
            raise ValueError(f"cudnn.sdpa: {name} must be 16-byte aligned")
        row = heads * size
        if thd:
            if nbytes % row:
                raise ValueError(f"cudnn.sdpa: {name} must hold whole packed SF tile rows")
            count = nbytes // row
            # Zero storage is legal only for a zero-capacity operand. Its SF
            # descriptor binds NULL with one dead tile, never an owned dummy.
            operand = facts[name[-1]]
            if count == 0 and operand.numel and operand.span != 0:
                raise ValueError(f"cudnn.sdpa: empty {name} requires zero-capacity {name[-1]}")
        elif getattr(spec, "paged", False) and name != "sf_q":
            # K/V SF pools page with K/V: page_size / 128 tiles per (page, head).
            count = spec.page_size // 128
            if nbytes != int(facts["k"].shape[0]) * row * count:
                raise ValueError(f"cudnn.sdpa: {name} size does not match the {int(facts['k'].shape[0])}-page pool")
        else:
            count = ((spec.s_q_max if name == "sf_q" else spec.s_k_max) + 127) // 128
            if nbytes != spec.b * row * count:
                raise ValueError(f"cudnn.sdpa: {name} size does not match the compiled dense geometry")
        if count > _I32_MAX:
            raise ValueError(f"cudnn.sdpa: {name} tile count exceeds Int32")
        patches[name + "_ptr"] = f.ptr if nbytes else 0
        tiles.append(max(1, count))
    if len(tiles) == 2:
        if facts.get("sf_v") is not None:
            raise ValueError("cudnn.sdpa: PV-BF16 does not consume sf_v")
        patches["sf_v_ptr"] = None
        tiles.append(0)
    elif tiles[1] != tiles[2]:
        raise ValueError("cudnn.sdpa: sf_k and sf_v must have the same packed tile count")
    patches["sf_tiles"] = tuple(tiles)
    return patches


def execute_quantized(spec, facts, workspace_ptr, stream, stream_int, *, scale_softmax_log2=None, stage_inputs=None):
    """Bind the shared geometry and runtime FP8 scalars before initializing or launching.

    Scales remain device pointers. The compiled host unscales a requested amax on the
    same stream; an unrequested amax uses the plan's declared scratch word. Standalone
    callers may omit scales (identity), while graph bindings require their declared UIDs.
    Return whether attention launched, for standalone empty-THD diagnostics.
    """
    quant = spec.quant
    if not workspace_ptr or workspace_ptr % _ALIGN_TMA:
        raise ValueError("cudnn.sdpa: prepared FP8 requires an aligned caller workspace")
    if quant.block_output is not None and quant.block_output.pack == 2:
        o = facts.get("o")
        if o is not None:
            if o.dtype not in ("float4_e2m1fn_x2", "uint8"):
                raise ValueError("cudnn.sdpa: NVFP4 O requires packed FP4 or byte storage")
            # Both native graph facts and the torch packed carrier already
            # report byte-slot geometry. Never divide that geometry again.
            facts = dict(facts, o=o._replace(dtype="uint8"))
    patches = _bind_mxfp8_scales(spec, facts) if quant.sf_sizes else {}
    if quant.block_output is not None or facts.get("sf_o") is not None:
        patches["sf_o_ptr"] = _bind_block_output(spec, facts)
    if quant.sf_sizes:
        patches["scale_o_ptr"] = None
        if quant.block_output is not None and (facts.get("scale_o") is not None) != quant.block_output.has_scale:
            raise ValueError("cudnn.sdpa: scale_o presence must match the block-output specialization")
    identity = workspace_ptr + quant.scratch_offset + 4
    needs_identity = False
    scalar_roles = (("amax_o",) + (("scale_o",) if quant.block_output is not None and quant.block_output.has_scale else ())) if quant.sf_sizes else _QUANT_ROLES
    for name in scalar_roles:
        f = facts.get(name)
        if f is None:
            if name == "amax_o":
                ptr = workspace_ptr + quant.scratch_offset
            else:
                ptr = identity
                needs_identity = True
        else:
            _on_plan_device(spec, name, f)
            if f.dtype != "float32" or f.numel != 1 or not f.contiguous or (f.span >= 0 and f.span < 1) or not f.ptr or f.ptr % 4:
                raise ValueError(f"cudnn.sdpa: {name} must be one aligned float32 device element with sufficient storage when observed")
            if name == "amax_o" and not quant.has_amax:
                raise ValueError("cudnn.sdpa: this specialization does not produce amax_o")
            ptr = f.ptr
        patches[name + "_ptr"] = ptr
    # Scalar output must not overwrite any input or caller output. Workspace alias
    # detection uses each operand's declared element width, including FP8 O.
    amax = patches["amax_o_ptr"]
    if facts.get("amax_o") is not None and workspace_ptr < amax + 4 and amax < workspace_ptr + quant.scratch_offset + 8:
        raise ValueError("cudnn.sdpa: prepared FP8 workspace overlaps amax_o")
    for name, f in facts.items():
        if f is None or name == "amax_o":
            continue
        # A declared raw address carries no observed allocation span. Its
        # declared footprint still detects aliases; capacity remains the
        # caller's contract, as for the shared dense metadata bindings.
        span = f.span if f.span >= 0 else (0 if f.numel == 0 else 1 + sum((n - 1) * st for n, st in zip(f.shape, f.strides)))
        if span <= 0:
            continue
        width = _buffers.DTYPE_ITEMSIZE[f.dtype]
        # Token-major declarations omit trailing atom padding. Even a bare
        # address promises that compiled extent, so it cannot alias scratch.
        if name == "sf_o" and quant.block_output is not None:
            span = max(span, quant.block_output.nbytes)
        scratch_end = workspace_ptr + quant.scratch_offset + 8
        if workspace_ptr < f.ptr + span * width and f.ptr < scratch_end:
            raise ValueError(f"cudnn.sdpa: prepared FP8 workspace overlaps {name}")
        if amax < f.ptr + span * width and f.ptr < amax + 4:
            raise ValueError(f"cudnn.sdpa: amax_o overlaps {name}")
    combine_args = None
    if isinstance(spec, ThdLaunchSpec):
        frame = bind_thd(spec, facts, workspace_ptr, stream, stream_int)
    elif spec.combine is not None:
        frame, combine_args = bind_dense_split(spec, facts, workspace_ptr, stream, stream_int)
        # The quantized combine appends its scalar pointers before the stream;
        # the half and ragged pointer ABIs remain unchanged.
        combine_args = (*combine_args[:-1], amax if quant.has_amax else None, patches.get("scale_o_ptr"), stream)
        if not quant.sf_sizes and spec.combine.output_dtype in ("float8_e4m3fn", "float8_e5m2"):
            # FP8 rounding and scale_o belong to the final combine, once.
            # The main host None-specializes its scale to one. Avoid an
            # identity memset (and its captured engine dependency) per call.
            patches["scale_o_ptr"] = None
    else:
        frame = bind_dense(spec, facts, stream, stream_int)
    if stage_inputs is not None:
        stage_inputs()  # All binding checks precede the existing input conversions.
    if isinstance(spec, ThdLaunchSpec):
        initialize_thd_stats(spec, facts, stream_int)
    if needs_identity:
        _buffers.fill_word_async(identity, 1, _buffers.init_word("fp32", 1.0), stream_int)
    if frame is not None:
        for name, ptr in patches.items():
            frame[spec.index[name]] = ptr
        if scale_softmax_log2 is not None:
            frame[spec.index["scale_softmax_log2"]] = scale_softmax_log2
        spec.fn(*frame)
        if combine_args is not None:
            spec.combine.fn(*combine_args)
    else:
        # No addressable Q token: there is no compiled host launch to reset
        # its reduction output, but Amax_O must still describe the empty O.
        _buffers.memset_zero_async(amax, 4, stream_int)
    return frame is not None


def _capacity(f: BufferFacts, geo: Tuple[int, int, int, int], name: str) -> int:
    """Token capacity of a packed operand under the resolved (ts, hs, es, row_span) geometry."""
    if f.span < 0:
        raise ValueError(f"cudnn.sdpa: " + (f"{name} was passed as a bare address; a ragged operand needs a sized buffer"))
    ts, row = geo[0], geo[3]
    return 0 if f.span < row else (f.span - row) // ts + 1


class ResolvedGeometry(NamedTuple):
    """The runtime geometry one THD call binds: the batch it runs, and per packed role the
    (token, head, elem) strides and the row span the capacities are computed from."""

    b: int
    roles: Mapping[str, Tuple[int, int, int, int]]  # name -> (ts, hs, es, row_span)


def resolve_thd_geometry(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]]) -> ResolvedGeometry:
    """Validate this call's effective geometry against the plan's fixed specialization and return
    what the binder writes; raise ValueError outside the supported domain. One implementation for
    admission and binding (no separate boolean query with its own rules).

    Fixed by the artifact: head counts (GQA specialization), head dims, layout kind, page size.
    Runtime per call: the batch (at most the declared one: workspace and unit envelope are sized
    for it), and each packed operand's token / head strides (TMA-expressible: elem stride 1,
    16-byte multiples, covering)."""
    cu_q, cu_kv = bool(spec.lens_form & 1), bool(spec.lens_form & 2)
    q_lens, kv_lens = facts.get("q_lens"), facts.get("kv_lens")
    if q_lens is None or kv_lens is None:
        raise ValueError(f"cudnn.sdpa: " + ("THD execute requires seq_q_lens and seq_kv_lens"))
    # Only the pure layout calculation is reusable. Addresses, observed storage spans,
    # producer devices, workspace and stream are still validated/bound on EVERY call
    # by bind_thd, before execution seeds padded Stats. Geometry changes (also
    # execute-time overrides) take the same admission rules below.
    names = ("q", "o") + (() if spec.paged else ("k", "v"))
    key = (q_lens.shape, kv_lens.shape, tuple((facts[name].shape, facts[name].strides) for name in names))
    cached = getattr(spec, "_geometry_cache", None)
    if cached is not None and cached[0] == key:
        return cached[1]
    b = q_lens.numel - (1 if cu_q else 0)
    if getattr(spec, "fixed_batch", False) and b != spec.b:
        raise ValueError(f"cudnn.sdpa: this artifact requires exactly {spec.b} sequences; got {b}")
    if b <= 0 or b > spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"seq_q_lens describes {b} sequences; this plan is prepared for 1..{spec.b}"))
    if kv_lens.numel != b + (1 if cu_kv else 0):
        raise ValueError(f"cudnn.sdpa: " + (f"seq_kv_lens must describe the same {b} sequences as seq_q_lens; got {kv_lens.numel} elements"))
    if spec.has_lse and spec.lse_padded and b != spec.b:
        raise ValueError(f"cudnn.sdpa: " + (f"a per-batch padded Stats buffer is declared for {spec.b} sequences; running {b} is not supported"))
    roles: Dict[str, Tuple[int, int, int, int]] = {}
    for name in names:
        f = facts[name]
        decl = spec.decl[name]
        h, d = decl[0], decl[1]
        st, sh = f.strides, f.shape
        if f.numel == 0:
            # A zero-numel operand has no geometry of its own, whatever its rank (issue #552: an all-KV-zero
            # call may pass an empty K/V; the packed-KV clamp below binds the dummy token). It takes the
            # plan's declared strides, as the pre-prepared path did.
            ts, hs, es = decl[2], decl[3], decl[4]
        elif len(st) == 4:  # the graph's (B, H, S, D) declaration, possibly overridden
            if int(sh[0]) != b or int(sh[1]) != h or int(sh[3]) != d:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: effective shape {tuple(sh)} must be ({b}, {h}, S, {d}) for this plan"))
            ts, hs, es = int(st[2]), int(st[1]), int(st[3])
        elif len(st) == 3:  # the caller's packed (T, H, D)
            if int(sh[1]) != h or int(sh[2]) != d:
                raise ValueError(f"cudnn.sdpa: " + (f"{name}: a packed THD buffer is (T, {h}, {d}); got {tuple(sh)}"))
            ts, hs, es = int(st[0]), int(st[1]), int(st[2])
        else:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: a THD operand is (T, H, D) or the graph's (B, H, S, D); got rank {len(st)}"))
        width = _buffers.DTYPE_ITEMSIZE[spec.expect[name]]
        if es != 1:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: the head dim must be contiguous (elem stride 1); got {es}"))
        if hs < d or (hs * width) % _ALIGN_TMA != 0:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: head stride {hs} must cover {d} elements and be a 16-byte multiple"))
        if ts < (h - 1) * hs + d or (ts * width) % _ALIGN_TMA != 0:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: token stride {ts} must cover the {h} heads and be a 16-byte multiple"))
        roles[name] = (ts, hs, es, (h - 1) * hs + (d - 1) * es + 1)
    geometry = ResolvedGeometry(b, MappingProxyType(roles))
    # Publish one immutable record, never a separately updated key/value pair: concurrent
    # calls with different geometry may replace the cache but keep their own geometry.
    # One entry bounds retained metadata independently of the number of batch shapes.
    if hasattr(spec, "_geometry_cache"):
        spec._geometry_cache = (key, geometry)
    return geometry


def _stats_layout_is_the_compiled_kind(spec: ThdLaunchSpec, lse: BufferFacts) -> int:
    """The host builds the Stats tensor from the compiled layout kind (token-major (T, H), head-major
    (1, H, ext), or the declared padded strides). Override-enabled half graph plans also bind the effective HN
    head stride; the layout kind remains fixed. Compact storage of
    rank <= 2 carries no layout that could contradict the kind (the kind is how that storage is written;
    this is what the tensor path bound too); a described rank-3 / rank-4 geometry must be the kind."""
    head_stride = spec.lse_head_stride
    if lse.numel == 0:
        return head_stride
    st, sh = lse.strides, lse.shape
    qh = spec.qh
    if spec.lse_padded:
        # The declared per-batch strides apply over the caller's STORAGE (the padded contract): any
        # contiguous allocation of the right size is that storage, whatever its own dim order; a
        # strided view is accepted only when it IS the declared (B, H, S) layout.
        if len(sh) in (3, 4) and (int(st[0]), int(st[1]), int(st[2])) == tuple(spec.lse_stride):
            return head_stride
        if not lse.contiguous:
            raise ValueError(f"cudnn.sdpa: padded lse_tensor strides {tuple(st)} must be the declared {tuple(spec.lse_stride)} or contiguous storage")
        return head_stride
    if len(sh) <= 2 and lse.contiguous:
        return head_stride  # flat (T*H) or (T, H) / (H, ext) storage: written in the compiled kind
    if len(sh) == 4:  # the graph's (B, H, S, 1)
        h_st, t_st = int(st[1]), int(st[2])
    elif len(sh) == 3 and spec.lse_head_major:  # (1, H, ext)
        h_st, t_st = int(st[1]), int(st[2])
    elif len(sh) == 3:  # (T, H, 1) token-major
        h_st, t_st = int(st[1]), int(st[0])
    elif len(sh) == 2:  # strided (T, H) / (H, ext)
        h_st, t_st = (int(st[0]), int(st[1])) if spec.lse_head_major else (int(st[1]), int(st[0]))
    else:
        raise ValueError(f"cudnn.sdpa: lse_tensor: unsupported Stats geometry {tuple(sh)} / {tuple(st)}")
    if spec.lse_head_major:
        if t_st != 1:
            raise ValueError(f"cudnn.sdpa: head-major lse_tensor must have the token axis contiguous; got stride {t_st}")
        if h_st <= 0:
            raise ValueError("cudnn.sdpa: head-major lse_tensor head stride must be positive")
        if getattr(spec, "lse_stride_override", False):
            head_stride = h_st
        elif spec.lse_head_stride and h_st != spec.lse_head_stride:
            raise ValueError(f"cudnn.sdpa: head-major lse_tensor head stride {h_st} must be the declared {spec.lse_head_stride}")
    elif h_st != 1 or t_st != qh:
        raise ValueError(f"cudnn.sdpa: token-major lse_tensor must be packed (T, H): head stride 1, token stride {qh}; got head {h_st}, token {t_st}")

    return head_stride


def _on_plan_device(spec, name: str, f: BufferFacts) -> None:
    # one device rule for every bound role: a KNOWN producer device must be the plan's CUDA device
    if f.device[0] != -1 and f.device != (_DLPACK_CUDA, spec.device_index):
        raise ValueError(f"cudnn.sdpa: " + (f"{name} must be on CUDA device {spec.device_index} (this plan's); got DLPack device {f.device}"))


def _bind_paged_kv(spec, frame: List[Any], ix: Dict[str, int], facts: Dict[str, Optional[BufferFacts]], k: BufferFacts, v: BufferFacts, b: int) -> int:
    """Bind the page pools and tables of a paged launch; returns SKV = max_pages * page_size. Shared by
    the THD and dense binders (one implementation of the paged domain)."""
    # K/V are page pools (n_pages, page_size, KH, D) in the kernel's order: a permutation of the
    # container's (n_pages, KH, page_size, D) strides; SKV = max_pages * page_size
    bt, btv = facts.get("block_table"), facts.get("block_table_v")
    if bt is None or btv is None:
        raise ValueError(f"cudnn.sdpa: " + ("paged KV requires paged_attention_k_table / paged_attention_v_table buffers"))
    if bt.dtype != "int32" or btv.dtype != "int32":
        raise ValueError(f"cudnn.sdpa: " + ("the page tables must be int32"))
    _on_plan_device(spec, "paged_attention_k_table", bt)
    _on_plan_device(spec, "paged_attention_v_table", btv)

    for name, table in (("paged_attention_k_table", bt), ("paged_attention_v_table", btv)):
        if not table.ptr or table.ptr % _ALIGN_F32:
            raise ValueError(f"cudnn.sdpa: {name} must have a non-null, 4-byte-aligned address")
    try:
        (tb, max_pages), table_strides = _paged_table_layout(bt.shape, bt.strides)
    except ValueError as error:
        raise ValueError(f"cudnn.sdpa: paged_attention_k_table {error}") from error
    try:
        (tbv, max_pages_v), table_strides_v = _paged_table_layout(btv.shape, btv.strides)
    except ValueError as error:
        raise ValueError(f"cudnn.sdpa: paged_attention_v_table {error}") from error
    if tb < b or tbv < b:
        raise ValueError(f"cudnn.sdpa: " + (f"the page tables describe {tb} / {tbv} sequences; this call runs {b}"))
    if max_pages_v != max_pages:
        raise ValueError("cudnn.sdpa: paged_attention_k_table and paged_attention_v_table must share max_pages")
    if "table_v_strides" not in ix and table_strides_v != table_strides:
        raise ValueError("cudnn.sdpa: this prepared host requires matching K/V table strides")
    # Each table keeps its own observed storage bound and physical geometry.
    for name, table, strides in (("k", bt, table_strides), ("v", btv, table_strides_v)):
        need = (b - 1) * strides[0] + (max_pages - 1) * strides[1] + 1
        if table.span >= 0 and table.span < need:
            raise ValueError(f"cudnn.sdpa: paged_attention_{name}_table spans {table.span} elements; ({b}, {max_pages}) with strides {strides} needs {need}")
    # the pools: (n_pages, KH, page_size, D) containers, head dim contiguous, one page count for K and V,
    # and the in-page layout kind the artifact was compiled for (its TMA descriptors order (row, head) by it)
    pool_strides = {}
    for name, f, d in (("k", k, spec.d_qk), ("v", v, spec.d_v)):
        try:
            strides, need = _paged_pool_layout(
                f.shape,
                f.strides,
                _buffers.DTYPE_ITEMSIZE[f.dtype],
                (getattr(spec, "paged_v_hnd", spec.paged_hnd) if name == "v" else spec.paged_hnd),
                spec.kh,
                spec.page_size,
                d,
            )
        except ValueError as error:
            raise ValueError(f"cudnn.sdpa: {name}: {error}") from error
        if f.span >= 0 and f.span < need:
            raise ValueError(f"cudnn.sdpa: {name}: page pool spans {f.span} elements; its effective geometry needs {need}")
        pool_strides[name] = strides
    if int(k.shape[0]) != int(v.shape[0]):
        raise ValueError(f"cudnn.sdpa: " + (f"K and V pools must hold the same number of pages; got {k.shape[0]} and {v.shape[0]}"))
    t_kv = max_pages * spec.page_size
    frame[ix["k_strides"]], frame[ix["v_strides"]] = pool_strides["k"], pool_strides["v"]
    frame[ix["block_table_ptr"]], frame[ix["block_table_v_ptr"]] = bt.ptr, btv.ptr
    frame[ix["table_strides"]] = (int(table_strides[0]), int(table_strides[1]))
    if "table_v_strides" in ix:
        frame[ix["table_v_strides"]] = (int(table_strides_v[0]), int(table_strides_v[1]))
    frame[ix["n_pages"]] = int(k.shape[0])
    return t_kv


def _bind_thd_python(spec: ThdLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int) -> Optional[List[Any]]:
    """Independent frame reference for the native THD binder.

    This call's argument frame for ``spec`` from the operands' facts (roles ``q k v o lse sinks
    q_lens kv_lens`` and, paged, ``block_table block_table_v``); None when no Q token is
    addressable. No buffer is written during binding."""
    ix = spec.index
    frame = list(spec.template)

    def on_plan_device(name: str, f: BufferFacts) -> None:
        _on_plan_device(spec, name, f)

    def operand(name: str) -> BufferFacts:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} is required"))
        if f.dtype != spec.expect[name]:
            raise ValueError(f"cudnn.sdpa: " + (f"{name}: runtime buffer dtype {f.dtype} does not match its declaration ({spec.expect[name]})"))
        on_plan_device(name, f)
        if f.ptr % _ALIGN_TMA != 0:
            raise ValueError(
                f"cudnn.sdpa: "
                + (f"{name}: runtime buffer base address must be 16-byte aligned (TMA global-address rule); got data_ptr() % 16 == {f.ptr % _ALIGN_TMA}")
            )
        return f

    def lens(name: str, n: int) -> int:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} is required"))
        on_plan_device(name, f)
        if f.dtype != "int32":
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be int32; got {f.dtype}"))
        if f.numel != n:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must have {n} elements; got {f.numel}"))
        if not f.contiguous:
            raise ValueError(f"cudnn.sdpa: " + (f"{name} must be contiguous (read as a flat ({n},) operand)"))
        if f.ptr % _ALIGN_F32:
            raise ValueError(f"cudnn.sdpa: {name} must be 4-byte aligned")
        if 0 <= f.span < n:
            raise ValueError(f"cudnn.sdpa: {name} observed storage is too small for its effective length")
        return f.ptr

    q, k, v, o = operand("q"), operand("k"), operand("v"), operand("o")
    geo = resolve_thd_geometry(spec, facts)
    if getattr(spec, "origin_specs", None) is not None:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.thd_origins import ROLES

        pointers = []
        for role, origin in zip(ROLES, spec.origin_specs):
            offset = facts.get(role)
            if origin is None:
                if offset is not None:
                    raise ValueError("unexpected THD physical-offset operand")
                pointers.append(None)
                continue
            if offset is None:
                raise ValueError("THD physical-offset operand is required")
            on_plan_device(role, offset)
            if (
                offset.dtype != f"int{origin[3]}"
                or not offset.contiguous
                or offset.numel != geo.b + 1
                or 0 <= offset.span < geo.b + 1
                or not offset.ptr
                or offset.ptr % (origin[3] // 8)
            ):
                raise ValueError("THD physical offsets require contiguous declared-width B+1 storage")
            pointers.append(offset.ptr)
        frame[ix["thd_origin_ptrs"]] = tuple(pointers)
    frame[ix["q_ptr"]], frame[ix["k_ptr"]], frame[ix["v_ptr"]], frame[ix["o_ptr"]] = q.ptr, k.ptr, v.ptr, o.ptr
    for name, slot in (("q", "q_strides"), ("o", "o_strides")) + (() if spec.paged else (("k", "k_strides"), ("v", "v_strides"))):
        ts, hs, _es, _row = geo.roles[name]
        frame[ix[slot]] = (ts, ts, hs)  # the extent-1 batch dim binds the token stride (never stepped)
    frame[ix["thd_q_lens_ptr"]] = lens("q_lens", geo.b + (1 if spec.lens_form & 1 else 0))
    frame[ix["thd_kv_lens_ptr"]] = lens("kv_lens", geo.b + (1 if spec.lens_form & 2 else 0))

    lse = facts.get("lse")
    lse_cap = None
    lse_head_stride = spec.lse_head_stride
    if spec.has_lse:
        if lse is None:
            raise ValueError(f"cudnn.sdpa: " + ("lse_tensor is required by this compiled specialization"))
        on_plan_device("lse_tensor", lse)
        if lse.dtype != "float32":
            raise ValueError(f"cudnn.sdpa: " + (f"lse_tensor must be float32; got {lse.dtype}"))
        if lse.ptr % _ALIGN_F32 != 0:
            raise ValueError(f"cudnn.sdpa: " + ("lse_tensor must be 4-byte aligned"))
        lse_head_stride = _stats_layout_is_the_compiled_kind(spec, lse)
        if spec.lse_padded:
            expected = spec.b * spec.qh * spec.s_q_max
            if lse.numel != expected:
                raise ValueError(f"cudnn.sdpa: " + (f"padded lse_tensor must have B*H_q*S_q_max = {expected} elements; got {lse.numel}"))
            need = 0 if expected == 0 else 1 + sum((n - 1) * st for n, st in zip((spec.b, spec.qh, spec.s_q_max), spec.lse_stride))
            if lse.ptr < 0 or need < 0 or lse.ptr + need * 4 > (1 << 63) - 1:
                raise ValueError("cudnn.sdpa: padded Stats address must fit in int64")
            if expected and not lse.ptr:
                raise ValueError("cudnn.sdpa: padded lse_tensor requires a non-null address")
            if lse.span >= 0 and lse.span < need:
                raise ValueError("cudnn.sdpa: padded lse_tensor observed storage must cover the declared strides")
        elif spec.lse_head_major and lse_head_stride:
            if 0 <= lse.span < spec.qh * lse_head_stride:
                raise ValueError("cudnn.sdpa: head-major lse_tensor observed storage must hold H_q*head_stride elements")
            if lse.span < 0 and lse.numel < spec.qh * lse_head_stride:
                raise ValueError(f"cudnn.sdpa: " + (f"head-major lse_tensor must hold H_q*head_stride = {spec.qh * lse_head_stride} elements; got {lse.numel}"))
        else:
            if lse.span < 0:
                raise ValueError(f"cudnn.sdpa: " + ("lse_tensor: a ragged Stats operand needs a sized buffer, not a bare address"))
            lse_cap = lse.span // spec.qh
        frame[ix["lse_ptr"]] = lse.ptr
    else:
        if lse is not None:
            raise ValueError(f"cudnn.sdpa: " + ("this specialization was compiled without a Stats output; construct the API without sample_lse"))

    t_q = min(_capacity(q, geo.roles["q"], "q"), _capacity(o, geo.roles["o"], "o"))
    if spec.total_q is not None:
        t_q = min(t_q, spec.total_q)
    if lse_cap is not None:
        t_q = min(t_q, lse_cap)
    # Spare backing storage does not enlarge the plan's live-Q bound. Use
    # the same bounded extent for descriptors, partials and the combine.
    split = getattr(spec, "split_workspace", None)
    if split is not None:
        t_q = min(t_q, split.capacity)
    # Strided HN padding is storage, not logical tokens. The span check above
    # still requires all head slots; the logical descriptor covers bounded Q.
    # Live rows are at most b * S_q, so a single-request view below the packed-Q capacity covers them (#1482).
    s_q_max = getattr(spec, "s_q_max", None)
    live_q = min(t_q, geo.b * s_q_max) if s_q_max else t_q
    if spec.has_lse and spec.lse_head_major and lse_head_stride and lse.numel < spec.qh * min(live_q, lse_head_stride):
        raise ValueError("cudnn.sdpa: head-major lse_tensor logical shape must cover bounded packed Q")
    # Empty Q still initializes padded Stats or quantized Amax/scalars. Its
    # sink, workspace and paged bindings must pass validation before any write.
    if t_q == 0 and not (spec.has_lse and spec.lse_padded) and getattr(spec, "quant", None) is None:
        return None

    if spec.paged:
        t_kv = _bind_paged_kv(spec, frame, ix, facts, k, v, geo.b)
    else:
        t_kv = min(_capacity(k, geo.roles["k"], "k"), _capacity(v, geo.roles["v"], "v"))
        if spec.total_kv is not None:
            t_kv = min(t_kv, spec.total_kv)
        if t_kv == 0:
            # All-KV-zero clamp: descriptor-only K/V rows alias live Q/O storage.
            # The setup kernel sees zero KV lengths and never reads either row;
            # QH >= KH guarantees that each allocation covers its one-row view.
            kh, d_qk, d_v = spec.kh, spec.d_qk, spec.d_v
            t_kv = 1
            frame[ix["k_ptr"]] = q.ptr
            frame[ix["k_strides"]] = (kh * d_qk, kh * d_qk, d_qk)
            frame[ix["v_ptr"]] = o.ptr
            frame[ix["v_strides"]] = (kh * d_v, kh * d_v, d_v)
    if spec.has_lse and spec.lse_head_major:
        frame[ix["lse_ext"]] = lse_head_stride or t_q
    frame[ix["problem_size"]] = (geo.b, spec.qh, spec.kh, t_q, t_kv, 0)
    # For B nonnegative lengths with sum <= T, sum(ceil(length / tile)) is
    # bounded by ceil(T / tile) + B - 1. Observe only host-known capacity;
    # device lengths may change during replay within this captured bound.
    # Keep a persistent kernel's smaller resident-cluster limit as well.
    wave_geometry = getattr(spec, "wave_geometry", None)
    unit_rows, unit_heads = (spec.cga_tile_m, spec.qh) if wave_geometry is None else wave_geometry[:2]
    units = ((t_q - 1) // unit_rows + geo.b) * unit_heads
    if split is not None:
        units *= split.splits
    frame[ix["n_thd_units"]] = min(frame[ix["n_thd_units"]], units)
    policy = getattr(spec, "wave_policy", None)
    if policy is not None:
        envelope, resident, initial_waves = policy
        bound = min(envelope, units)
        limit = resident * initial_waves if wave_geometry is None else wave_geometry[2]
        frame[ix["n_thd_units"]] = bound if bound <= limit else resident

    sinks = facts.get("sinks")
    if spec.has_sink:
        if sinks is None:
            raise ValueError(f"cudnn.sdpa: " + ("sinks is required by this compiled specialization"))
        on_plan_device("sinks", sinks)
        if sinks.dtype != "float32" or sinks.numel != spec.qh or not sinks.contiguous:
            raise ValueError(f"cudnn.sdpa: " + (f"sinks must be a contiguous ({spec.qh},) float32 tensor"))
        if sinks.ptr % _ALIGN_F32:
            raise ValueError("cudnn.sdpa: sinks must be 4-byte aligned")
        if sinks.span >= 0 and sinks.span < spec.qh:
            raise ValueError(f"cudnn.sdpa: sinks spans {sinks.span} elements; this launch reads {spec.qh}")
        frame[ix["sinks_ptr"]] = sinks.ptr
    else:
        if sinks is not None:
            raise ValueError(f"cudnn.sdpa: " + ("this specialization was compiled without a sink; construct the API with has_sink"))
        frame[ix["sinks_ptr"]] = 0  # HAS_SINK=False: dead slot; a null faults loudly if it is ever read (Rule 8)

    alignment = getattr(spec, "workspace_alignment", _ALIGN_TMA)
    if not workspace_ptr:
        raise ValueError("cudnn.sdpa: prepared THD requires a non-null workspace")
    if workspace_ptr % alignment != 0:
        raise ValueError(f"cudnn.sdpa: the workspace must be {alignment}-byte aligned; got 0x{workspace_ptr:x}")
    frame[ix["meta_ptr"]] = workspace_ptr
    frame[ix["o_desc_ptr"]] = workspace_ptr + spec.off_o_desc
    if split is not None:
        if workspace_ptr <= 0:
            raise ValueError("cudnn.sdpa: packed split requires a non-null workspace")
        frame[ix["o_partial_ptr"]] = workspace_ptr + split.off_o
        frame[ix["lse_partial_ptr"]] = workspace_ptr + split.off_lse
        frame[ix["partial_o_strides"]] = (t_q * spec.qh * spec.d_v, spec.qh * spec.d_v, spec.d_v)
    frame[ix["stream"]] = stream
    return frame if t_q else None


def initialize_thd_stats(spec, facts, stream_int):
    """Apply the plan's existing padded-Stats seed after successful binding."""
    if spec.has_lse and spec.lse_padded:
        _buffers.apply_fill_plan(facts["lse"].ptr, spec.lse_fill_plan, spec.neg_inf, stream_int)


def execute_thd(spec, facts, workspace_ptr, stream, stream_int, scale=None):
    """Execute the Python-bound half path: validate, initialize, then launch."""
    frame = bind_thd(spec, facts, workspace_ptr, stream, stream_int)
    initialize_thd_stats(spec, facts, stream_int)
    if frame is None:
        return False
    if scale is not None:
        frame[spec.index["scale_softmax_log2"]] = scale
    spec.fn(*frame)
    return True


def _ragged_offsets_role(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], name: str, *, required: bool = True) -> Optional[int]:
    """A (B+1,) int32 ragged-offset operand of the ragged-Q leg: device, dtype, alignment, extent."""
    f = facts.get(name)
    if f is None:
        if required:
            raise ValueError(f"cudnn.sdpa: {name} (ragged offsets) is required by the ragged-Q decode leg")
        return None
    _on_plan_device(spec, name, f)
    want = "int64" if spec.ragged_i64 else "int32"
    if f.dtype != want or not f.contiguous or f.numel < spec.b + 1:
        raise ValueError(f"cudnn.sdpa: {name} must be a contiguous {want} tensor of at least {spec.b + 1} ragged offsets; got {f.dtype} x {f.numel}")
    if f.ptr % (8 if spec.ragged_i64 else _ALIGN_F32):
        raise ValueError(f"cudnn.sdpa: {name} must be {'8' if spec.ragged_i64 else '4'}-byte aligned")
    if f.span >= 0 and f.span < spec.b + 1:
        raise ValueError(f"cudnn.sdpa: {name} spans {f.span} elements; this launch reads {spec.b + 1}")
    return f.ptr


def _packed_role(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], name: str, heads: int, d: int, expect: str, *, tma: bool):
    """One PACKED (THD) operand of the ragged-Q leg -- the graph's (B, H, S_max, D) declaration or a
    (T, H, D) buffer -- as ``(ptr, token_stride, head_stride, token_capacity)``.  The batch stride is
    never read (every row base is a ragged offset); the token stride must cover the heads and, for a
    TMA operand, be a 16-byte multiple; the capacity is the largest token whose row lies inside the
    buffer's span (the THD contract: rows below it are caller-provided, rows at or past it TMA-clip)."""
    f = facts.get(name)
    if f is None:
        raise ValueError(f"cudnn.sdpa: {name} is required")
    if f.dtype != expect:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
    _on_plan_device(spec, name, f)
    width = _buffers.DTYPE_ITEMSIZE[expect]
    if f.ptr % (_ALIGN_TMA if tma else width) != 0:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer base address must be {_ALIGN_TMA if tma else width}-byte aligned")
    ts, hs, row = _packed_role_layout(tuple(f.shape), tuple(f.strides), heads, d, width, tma, name)
    capacity = _capacity(f, (ts, hs, 1, row), name)
    return f.ptr, ts, hs, capacity


class _DenseRole(NamedTuple):
    ptr: int
    strides: Tuple[int, int, int]  # (batch, seq, head) element strides
    b: int
    s: int


def _dense_role(
    spec: DenseLaunchSpec,
    facts: Dict[str, Optional[BufferFacts]],
    name: str,
    heads: int,
    d: int,
    s_max: int,
    expect: str,
    b_mult: int = 1,
    *,
    tma: bool = True,
) -> _DenseRole:
    """One BHSD operand of a dense launch: dtype / device / alignment, the fixed head count and head
    dim, batch and sequence within the plan's envelope, and a layout TMA binds zero-copy — head dim
    contiguous, seq and head strides 16-byte multiples, token-major and covering (the shared rule of
    ``config_sm100.bshd_zero_copy_stride``); a smaller buffer than the geometry claims is rejected."""
    f = facts.get(name)
    if f is None:
        raise ValueError(f"cudnn.sdpa: {name} is required")
    if f.dtype != expect:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
    _on_plan_device(spec, name, f)
    align = _ALIGN_TMA if tma else _buffers.DTYPE_ITEMSIZE[expect]
    if f.ptr % align != 0:
        raise ValueError(f"cudnn.sdpa: {name}: runtime buffer base address must be {align}-byte aligned; got data_ptr() % {align} == {f.ptr % align}")
    bound, b, seq, need = _dense_role_layout(
        tuple(f.shape), tuple(f.strides), heads, d, s_max, spec.b * b_mult, _buffers.DTYPE_ITEMSIZE[expect], tma, name, getattr(spec, "dense_flex", False)
    )
    if f.span >= 0 and f.span < need:
        raise ValueError(f"cudnn.sdpa: {name} spans {f.span} elements; its geometry {tuple(f.shape)} / {tuple(f.strides)} needs {need}")
    return _DenseRole(f.ptr, bound, b, seq)


def _dense_lse(spec: DenseLaunchSpec, lse: Optional[BufferFacts], b: int, s_q: int, *, required: bool):
    """Validate main or final Stats with the same device, span and non-aliasing contract."""
    if required:
        if lse is None:
            raise ValueError("cudnn.sdpa: lse_tensor is required by this compiled specialization")
        _on_plan_device(spec, "lse_tensor", lse)
        if lse.dtype != "float32":
            raise ValueError(f"cudnn.sdpa: lse_tensor must be float32; got {lse.dtype}")
        if lse.ptr % _ALIGN_F32 != 0:
            raise ValueError("cudnn.sdpa: lse_tensor must be 4-byte aligned")
        strides, need = _dense_lse_layout(tuple(lse.shape), tuple(lse.strides), b, s_q, spec.qh)
        if lse.span >= 0 and lse.span < need:
            raise ValueError(f"cudnn.sdpa: lse_tensor spans {lse.span} elements; its geometry needs {need}")
        return lse.ptr, strides
    elif lse is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without a Stats output; construct the API without sample_lse")

    return None, (0, 0, 0)


def bind_dense(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], stream, stream_int: int) -> List[Any]:
    """This call's argument frame for a dense launch from the operands' facts (roles ``q k v o`` and,
    as compiled, ``lse sinks seq_kv seq_q gate`` and the paged ``block_table block_table_v``). Under a
    split the ``o`` / ``lse`` roles are the split-major partial slabs (batch extent split * B)."""
    ix = spec.index
    frame = list(spec.template)
    q_tokens = 0
    if getattr(spec, "ragged", False):
        # Ragged-Q leg: Q is the caller's PACKED buffer; the kernel binds it as the
        # [1, T, H, D] view (T = token capacity, tightened by the declared total) and
        # reads each batch's row base from the ragged offsets.  The launch runs the
        # DECLARED batch (the offsets describe B sequences) at S_q = 1.
        q_ptr, q_ts, q_hs, q_tokens = _packed_role(spec, facts, "q", spec.qh, spec.d_qk, spec.expect["q"], tma=True)
        if spec.total_q is not None:
            q_tokens = min(q_tokens, max(int(spec.total_q), 0))
        b, s_q = spec.b, spec.s_q_max
        frame[ix["q_ptr"]], frame[ix["q_strides"]] = q_ptr, (q_ts, q_ts, q_hs)
        frame[ix["ragged_q_addr"]] = _ragged_offsets_role(spec, facts, "ragged_q")
    else:
        q = _dense_role(spec, facts, "q", spec.qh, spec.d_qk, spec.s_q_max, spec.expect["q"])
        b, s_q = q.b, q.s
        frame[ix["q_ptr"]], frame[ix["q_strides"]] = q.ptr, q.strides
    o_d = spec.d_v // (spec.quant.block_output.pack if spec.quant is not None and spec.quant.block_output is not None else 1)
    o = _dense_role(spec, facts, "o", spec.qh, o_d, spec.s_q_max, spec.expect["o"], b_mult=spec.split)
    if o.b != b * spec.split or o.s != s_q:
        raise ValueError(
            f"cudnn.sdpa: o is ({o.b}, {spec.qh}, {o.s}, {spec.d_v}) but q runs batch {b} x seq {s_q}" + (f" ({spec.split} splits)" if spec.split > 1 else "")
        )
    frame[ix["o_ptr"]], frame[ix["o_strides"]] = o.ptr, o.strides
    if spec.paged:
        k, v = facts.get("k"), facts.get("v")
        if k is None or v is None:
            raise ValueError("cudnn.sdpa: k and v are required")
        for name, f, expect in (("k", k, spec.expect["k"]), ("v", v, spec.expect["v"])):
            if f.dtype != expect:
                raise ValueError(f"cudnn.sdpa: {name}: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
            _on_plan_device(spec, name, f)
            if f.ptr % _ALIGN_TMA != 0:
                raise ValueError(f"cudnn.sdpa: {name}: runtime buffer base address must be 16-byte aligned")
        frame[ix["k_ptr"]], frame[ix["v_ptr"]] = k.ptr, v.ptr
        s_kv = _bind_paged_kv(spec, frame, ix, facts, k, v, b)
    else:
        k = _dense_role(spec, facts, "k", spec.kh, spec.d_qk, spec.s_k_max, spec.expect["k"])
        v = _dense_role(spec, facts, "v", spec.kh, spec.d_v, spec.s_k_max, spec.expect["v"])
        if k.b != b or v.b != b or k.s != v.s:
            raise ValueError(f"cudnn.sdpa: k / v run batch {k.b} / {v.b} x seq {k.s} / {v.s}; q runs batch {b}")
        s_kv = k.s
        frame[ix["k_ptr"]], frame[ix["k_strides"]] = k.ptr, k.strides
        frame[ix["v_ptr"]], frame[ix["v_strides"]] = v.ptr, v.strides
    if spec.shape_fixed and (s_q, s_kv) != (spec.s_q_max, spec.s_k_max):
        raise ValueError(
            f"cudnn.sdpa: this artifact was lowered for exactly S_q={spec.s_q_max}, S_kv={spec.s_k_max} (a square-mask / schedule canonicalization "
            f"read the declared extents); it does not serve ({s_q}, {s_kv})"
        )
    if spec.lpt_grid_fixed and (b, s_q) != (spec.b, spec.s_q_max):
        raise ValueError(
            f"cudnn.sdpa: this artifact's grouped LPT schedule was compiled for exactly batch={spec.b}, S_q={spec.s_q_max}; "
            f"it does not serve batch={b}, S_q={s_q}"
        )
    if not _kv_tail_admitted(spec, s_q, s_kv):
        raise ValueError(
            f"cudnn.sdpa: S_kv ({s_kv}) must be a multiple of {spec.tile_n} for this artifact unless per-batch KV lengths are present or the "
            f"causal mask covers the KV tail (the compiled specialization does not mask a partial last tile)"
        )
    # The spare sixth slot carries the packed Q token capacity on the ragged-Q leg (0 otherwise).
    frame[ix["problem_size"]] = (b, spec.qh, spec.kh, s_q, s_kv, q_tokens)
    if spec.fp32_partial:
        frame[ix["o_partial_ptr"]] = o.ptr

    frame[ix["lse_ptr"]], frame[ix["lse_strides"]] = _dense_lse(spec, facts.get("lse"), b * spec.split, s_q, required=spec.has_lse)

    sinks = facts.get("sinks")
    if spec.has_sink:
        if sinks is None:
            raise ValueError("cudnn.sdpa: sinks is required by this compiled specialization")
        _on_plan_device(spec, "sinks", sinks)
        if sinks.dtype != "float32" or sinks.numel != spec.qh or not sinks.contiguous:
            raise ValueError(f"cudnn.sdpa: sinks must be float32 and contiguous with {spec.qh} elements")
        if sinks.ptr % _ALIGN_F32:
            raise ValueError("cudnn.sdpa: sinks must be 4-byte aligned")
        if sinks.span >= 0 and sinks.span < spec.qh:
            raise ValueError(f"cudnn.sdpa: sinks spans {sinks.span} elements; this launch reads {spec.qh}")
        frame[ix["sinks_ptr"]] = sinks.ptr
    else:
        if sinks is not None:
            raise ValueError("cudnn.sdpa: this specialization was compiled without a sink; construct the API with has_sink")
        frame[ix["sinks_ptr"]] = 0  # HAS_SINK=False: dead slot; a null faults loudly if it is ever read (Rule 8)

    def lens(name: str) -> int:
        f = facts.get(name)
        if f is None:
            raise ValueError(f"cudnn.sdpa: {name} is required by this compiled specialization")
        _on_plan_device(spec, name, f)
        if f.dtype != "int32" or not f.contiguous or f.numel < b:
            raise ValueError(f"cudnn.sdpa: {name} must be a contiguous int32 tensor of at least {b} elements; got {f.dtype} x {f.numel}")
        if f.ptr % _ALIGN_F32:
            raise ValueError(f"cudnn.sdpa: {name} must be 4-byte aligned")
        if f.span >= 0 and f.span < b:
            raise ValueError(f"cudnn.sdpa: {name} spans {f.span} elements; this launch reads {b}")
        return f.ptr

    # Dense hosts retain these pointer slots, but only SEQ_KV_PRESENT reads
    # meta and only THD reads o_desc (both compile-time flags): dead slots are 0.
    frame[ix["o_desc_ptr"]] = 0
    frame[ix["meta_ptr"]] = lens("seq_kv_lens") if spec.seq_kv_present else 0
    if spec.seq_q_present:
        frame[ix["seq_q_lens_addr"]] = lens("seq_q_lens")

    if spec.gate_expect is not None:
        g = _dense_role(spec, facts, "gate", spec.qh, spec.d_v, spec.s_q_max, spec.gate_expect)
        if g.b != b or g.s != s_q:
            raise ValueError(f"cudnn.sdpa: gate is ({g.b}, {spec.qh}, {g.s}, {spec.d_v}) but q runs batch {b} x seq {s_q}")
        frame[ix["gate_ptr"]], frame[ix["gate_strides"]] = g.ptr, g.strides
    elif facts.get("gate") is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without an epilogue gate")
    frame[ix["stream"]] = stream
    return frame


def bind_dense_split(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], workspace_ptr: int, stream, stream_int: int):
    """Bind partial workspace and final strided outputs before either split kernel launches."""
    combine = spec.combine
    if not workspace_ptr or workspace_ptr % _ALIGN_TMA:
        raise ValueError("cudnn.sdpa: split workspace must be non-null and 16-byte aligned")
    ragged_args = ()  # the dense pointer ABI is unchanged; the ragged entry appends offsets / divisors / capacities
    if getattr(spec, "ragged", False):
        # Ragged-Q leg: the final O / Stats are the caller's PACKED buffers; the
        # combine places row ``offset[b] / div + q_row`` with batch coord 0, so only
        # the token / head strides are bound (the batch stride is never stepped),
        # and every store is bounded by the buffer's packed token capacity.
        # The THD contract: a producer with NO addressable token (an empty Q, O
        # or Stats buffer) launches nothing -- return None and the caller skips
        # both kernels, as the prefill THD leg's binder does.
        _, _, _, q_cap = _packed_role(spec, facts, "q", spec.qh, spec.d_qk, spec.expect["q"], tma=True)
        o_ptr, o_ts, o_hs, o_cap = _packed_role(spec, facts, "o", spec.qh, spec.d_v, combine.output_dtype, tma=False)
        o_strides = (0, o_ts, o_hs, 1)
        lse_ptr, lse_strides, lse_cap = _ragged_lse(spec, facts.get("lse"), required=combine.has_stats)
        if spec.total_q is not None:
            if o_cap < int(spec.total_q):
                raise ValueError(f"cudnn.sdpa: o holds {o_cap} packed tokens; the declared packed Q total is {spec.total_q}")
            if combine.has_stats and lse_cap < int(spec.total_q):
                raise ValueError(f"cudnn.sdpa: lse_tensor holds {lse_cap} packed tokens; the declared packed Q total is {spec.total_q}")
        if q_cap == 0 or o_cap == 0 or not o_ptr or (combine.has_stats and (lse_cap == 0 or not lse_ptr)):
            return None
        ragged_args = (
            _ragged_offsets_role(spec, facts, "ragged_q"),
            _ragged_offsets_role(spec, facts, "ragged_o"),
            _ragged_offsets_role(spec, facts, "ragged_lse", required=combine.has_stats) if combine.has_stats else None,
            spec.ragged_divs,
            (min(o_cap, _I32_MAX), min(lse_cap, _I32_MAX)),
        )
    else:
        q = facts.get("q")
        if q is None or len(q.shape) != 4 or (q.shape[0], q.shape[2]) != (spec.b, spec.s_q_max):
            raise ValueError(f"cudnn.sdpa: a split launch runs the declared (B, S_q) = ({spec.b}, {spec.s_q_max})")
        o = _dense_role(spec, facts, "o", spec.qh, spec.d_v, spec.s_q_max, combine.output_dtype, tma=False)
        if (o.b, o.s) != (spec.b, spec.s_q_max):
            raise ValueError("cudnn.sdpa: split output must match the declared (B, S_q)")
        o_ptr, o_strides = o.ptr, (*o.strides, 1)
        lse_ptr, lse_strides = _dense_lse(spec, facts.get("lse"), spec.b, spec.s_q_max, required=combine.has_stats)
    lse_partial_ptr = workspace_ptr + combine.lse_offset
    partials = dict(facts, o=combine.o._replace(ptr=workspace_ptr), lse=combine.lse._replace(ptr=lse_partial_ptr))
    frame = bind_dense(spec, partials, stream, stream_int)
    combine_args = (
        workspace_ptr,
        lse_partial_ptr,
        o_ptr,
        lse_ptr,
        (spec.b, spec.qh, spec.s_q_max, spec.d_v),
        spec.split,
        o_strides,
        lse_strides,
        *ragged_args,
        stream,
    )
    return frame, combine_args


def _ragged_lse(spec: DenseLaunchSpec, lse: Optional[BufferFacts], *, required: bool):
    """The ragged-Q leg's final Stats: the caller's PACKED ragged Stats buffer (Rule S1: token-major
    ``(T, H)`` -- stride_h == 1, stride_s == H -- or head-major ``(H, head_stride)`` -- stride_s == 1),
    declared as the graph's (B, H, S_max[, 1]) or as its packed rank-2 form. Returns the pointer, the
    (batch, head, token) strides the combine indexes with batch coord 0, and the buffer's packed token
    CAPACITY (the largest token whose row lies inside the producer's span; a head-major buffer is also
    bounded by its head stride, which must cover the declared packed total -- Rule S1) -- the bound the
    combine's stores never cross."""
    if not required:
        if lse is not None:
            raise ValueError("cudnn.sdpa: this specialization was compiled without a Stats output; construct the API without sample_lse")
        return None, (0, 0, 0), 0
    if lse is None:
        raise ValueError("cudnn.sdpa: lse_tensor is required by this compiled specialization")
    _on_plan_device(spec, "lse_tensor", lse)
    if lse.dtype != "float32":
        raise ValueError(f"cudnn.sdpa: lse_tensor must be float32; got {lse.dtype}")
    if lse.ptr % _ALIGN_F32 != 0:
        raise ValueError("cudnn.sdpa: lse_tensor must be 4-byte aligned")
    stride_h, stride_s, head_span, head_major = _ragged_lse_layout(tuple(lse.shape), tuple(lse.strides), spec.qh, spec.ragged_lse_head_major)
    # Packed token capacity: the last token whose row still lies inside the span.
    if lse.span < 0:
        cap = _I32_MAX
    elif lse.span < head_span:
        cap = 0
    else:
        cap = (lse.span - head_span) // stride_s + 1
    if head_major:
        # Head-major (H, head_stride): heads are head_stride tokens apart, so the head
        # stride bounds the tokens too (a shorter one would let heads overlap).  The
        # CLASSIFIER decides the packing (Rule S1): at H == 1 a token-major (T, 1)
        # buffer also has stride_s == 1 and must keep its full token capacity.
        cap = min(cap, stride_h)
        if spec.total_q is not None and stride_h < int(spec.total_q):
            raise ValueError(f"cudnn.sdpa: head-major ragged Stats head stride {stride_h} does not cover the declared packed Q total {spec.total_q}")
    return lse.ptr, (0, stride_h, stride_s), int(cap)


bind_thd = _bind_thd_python


class _ReferenceBinder:
    """Test-only adapter into the frozen Python oracle; never a production fallback."""

    def __init__(self, spec):
        self.spec = copy(spec)
        self.spec.native = None

    def facts(self, pack, indices, roles):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import _DTYPE_BY_CODE

        present = [(role, index) for role, index in zip(roles, indices) if index >= 0 and pack.is_filled(index)]
        observed = pack._facts_as([index for _, index in present], BufferFacts, _DTYPE_BY_CODE)
        return dict(zip((role for role, _ in present), observed))


class ReferenceThdBinder(_ReferenceBinder):
    def execute(self, pack, indices, workspace, stream, scale=None):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import _NATIVE_THD_ROLES

        spec = self.spec
        roles = _NATIVE_THD_ROLES + _native_quant_roles(spec.quant)
        if getattr(spec, "origin_specs", None) is not None:
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.thd_origins import ROLES

            roles += ROLES
        facts = self.facts(pack, indices, roles)
        if spec.quant is not None:
            return execute_quantized(spec, facts, workspace, stream, int(stream), scale_softmax_log2=scale)
        return execute_thd(spec, facts, workspace, stream, int(stream), scale)


class ReferenceDenseBinder(_ReferenceBinder):
    def execute(self, pack, indices, stream, scale=None, workspace=0):
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.prepared import _NATIVE_DENSE_ROLES

        spec = self.spec
        facts = self.facts(pack, indices, _NATIVE_DENSE_ROLES + _native_quant_roles(spec.quant))
        if spec.quant is not None:
            return execute_quantized(spec, facts, workspace, stream, int(stream), scale_softmax_log2=scale)
        if spec.split > 1:
            bound = bind_dense_split(spec, facts, workspace, stream, int(stream))
            if bound is None:
                return False
            frame, combine_args = bound
        else:
            frame = bind_dense(spec, facts, stream, int(stream))
        if scale is not None:
            slot = "scale_softmax_log2" if "scale_softmax_log2" in spec.index else "scale_softmax"
            frame[spec.index[slot]] = scale
        spec.fn(*frame)
        if spec.split > 1:
            spec.combine.fn(*combine_args)
        return True


def use_reference(spec):
    """Explicitly install the independent oracle in a test's private launch spec."""
    spec.native = (ReferenceThdBinder if isinstance(spec, ThdLaunchSpec) else ReferenceDenseBinder)(spec)


def _kv_tail_admitted(spec, s_q: int, s_kv: int) -> bool:
    """The compiled artifact's KV-tail contract (the adapter's check_support rule, applied to the
    RUNTIME extents): a partial last KV tile is zero-filled by TMA but only MASKED on the padded /
    causal paths, so S_kv must be a tile multiple unless per-batch KV lengths carry the real lengths
    or the causal diagonal provably covers the tail. SM120 always masks its rightmost tile."""
    if getattr(spec, "kv_tail_native", False) or spec.paged or s_kv % spec.tile_n == 0 or spec.seq_kv_present:
        return True
    if not spec.causal:
        return False
    return (spec.causal_bottom_right and spec.window_right == 0) or (not spec.causal_bottom_right and s_q + spec.window_right <= s_kv)
