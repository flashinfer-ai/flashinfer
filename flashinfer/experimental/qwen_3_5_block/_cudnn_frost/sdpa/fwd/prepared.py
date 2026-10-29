# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
"""Immutable forward plans and native binding for SM90/SM100/SM107/SM120.

Graph, standalone and staged entry points share the native binder selected by
host ABI (dense or THD). Plan build retains immutable geometry and artifacts;
each execute observes current storage and binds a fresh frame. Device scratch
belongs to the caller. There is no production Python binder or fallback.

The dense native binder also uses the pure cached layout helpers below. They
retain geometry only, never addresses, streams or storage observations. The
retired Python frame implementations live solely in the differential tests.
"""

from __future__ import annotations

import inspect
import math
import threading
from copy import copy
from functools import lru_cache
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Tuple

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _DLPACK_FP4_CODE_BITS
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost import buffers as _buffers
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.compiled_cache import positional_entry

_ALIGN_TMA = 16
_ALIGN_F32 = 4
_DLPACK_CUDA = 2
_DLPACK_CPU = 1
_I32_MAX = 2**31 - 1
_DTYPE_BY_CODE = {(code, bits): name for name, (code, bits) in _buffers.DTYPES.items()}
_DTYPE_BY_CODE[_DLPACK_FP4_CODE_BITS] = "float4_e2m1fn_x2"


class BufferFacts(NamedTuple):
    """What a caller buffer is, observed once (see the module docstring)."""

    ptr: int
    dtype: str  # bare name ("bfloat16"); "" when unknown
    device: Tuple[int, int]  # DLPack (device_type, device_id); (-1, -1) unknown
    span: int  # element span the producer guarantees, in the DECLARED element width; -1 unknown (a bare address)
    shape: Tuple[int, ...]
    strides: Tuple[int, ...]

    @property
    def numel(self) -> int:
        n = 1
        for e in self.shape:
            n *= int(e)
        return n

    @property
    def contiguous(self) -> bool:
        return _buffers.is_contiguous(self.shape, self.strides)


def facts_of_tensor(t) -> Optional[BufferFacts]:
    """Facts of a torch tensor (the standalone adapter's operands); None for None."""
    if t is None:
        return None
    shape, strides = tuple(t.shape), tuple(t.stride())
    n = int(t.numel())
    span = n if (n == 0 or t.is_contiguous()) else int(1 + sum((s - 1) * st for s, st in zip(shape, strides)))
    dev = t.device
    device = (_DLPACK_CUDA, int(dev.index if dev.index is not None else 0)) if dev.type == "cuda" else (_DLPACK_CPU, 0)  # a known CPU tensor is not "unknown"
    return BufferFacts(t.data_ptr(), str(t.dtype).split(".")[-1], device, span, shape, strides)


def facts_of_roles(pack, indices: List[int]) -> List[BufferFacts]:
    """Project normalized operands into immutable facts in one native crossing.

    Effective dtype/geometry follow graph declarations and overrides; span and
    device retain the producer's observations. No operand objects are built.
    """
    return pack.native._facts_as(indices, BufferFacts, _DTYPE_BY_CODE)


_QUANT_ROLES = ("descale_q", "descale_k", "descale_v", "scale_o", "amax_o")
_QUANT_SLOTS = frozenset(name + "_ptr" for name in _QUANT_ROLES)
_NATIVE_MX_ROLES = _QUANT_ROLES + ("sf_q", "sf_k", "sf_v")
_NATIVE_BLOCK_ROLES = _NATIVE_MX_ROLES + ("sf_o",)


class BlockOutputSpec(NamedTuple):
    geometry: Tuple[int, int, int, int]
    nbytes: int
    pack: int
    has_scale: bool


class QuantizedLaunchSpec(NamedTuple):
    has_amax: bool
    scratch_offset: int  # unused amax and an identity scale, in caller-owned workspace
    sf_sizes: Tuple[int, ...] = ()  # opaque F8_128x4 tile byte sizes; empty for per-tensor FP8
    block_output: Optional[BlockOutputSpec] = None


def _quant_spec(api):
    if not (getattr(api, "_prepared_mxfp8", False) or getattr(api, "_prepared_fp8", False)):
        return None
    block = None
    if api.o_block_scale:
        plane, row_b, col_h, cols = api._sfo_geometry
        b, h, rows = int(api.batch_size), int(api.h_q), int(api.sf_o_desc.shape[2])
        needed = b * h * plane if plane else ((b * rows + 127) // 128 * 128) * cols
        block = BlockOutputSpec((plane, row_b, col_h, cols), needed, 2 if api.o_block_scale == 16 else 1, bool(api.has_scale_o))
    sizes = ()
    if getattr(api, "_prepared_mxfp8", False):
        km = api._k_mod
        # A template compiled per head geometry (SM120) sizes its scale tiles from
        # the head dims; fixed-geometry templates carry them as module constants.
        if hasattr(km, "mxfp8_sf_sizes"):
            sizes = km.mxfp8_sf_sizes(int(api.head_dim_qk), int(api.head_dim_v))
        else:
            sizes = (km.SF_SMEM_SIZE_Q, km.SF_SMEM_SIZE_K) + (() if getattr(api, "pv_bf16", False) else (km.SF_SMEM_SIZE_V,))
    return QuantizedLaunchSpec(bool(api.has_amax_o), api._prepared_quant_offset(), sizes, block)


def _native_quant_roles(quant):
    if quant is None:
        return ()
    # Keep the native quantized prefix stable: per-tensor block output leaves
    # the three input-SF roles unbound instead of shifting the output-SF slot.
    if quant.block_output is not None:
        return _NATIVE_BLOCK_ROLES
    return _NATIVE_MX_ROLES if quant.sf_sizes else _QUANT_ROLES


def _quant_roles(quant):
    roles = (("sf_q", "sf_k", "sf_v")[: len(quant.sf_sizes)] + ("amax_o",)) if quant.sf_sizes else _QUANT_ROLES
    if quant.block_output is not None:
        roles += ("sf_o",) + (("scale_o",) if quant.sf_sizes else ())
    return roles


def _quant_slots(quant):
    if quant is None:
        return frozenset()
    slots = frozenset(("sf_q_ptr", "sf_k_ptr", "sf_v_ptr", "sf_tiles", "amax_o_ptr", "scale_o_ptr")) if quant.sf_sizes else _QUANT_SLOTS
    return slots | {"sf_o_ptr"}


@lru_cache(maxsize=256)
def _sf_byte_count(shape, strides, dtype):
    """Opaque reordered SF must be gap-free storage, in any axis permutation."""
    if len(shape) != len(strides) or any(n < 0 or st < 0 for n, st in zip(shape, strides)):
        raise ValueError("cudnn.sdpa: MXFP8 SF geometry must have matching nonnegative extents and strides")
    width = _buffers.DTYPE_ITEMSIZE.get(dtype)
    if width is None:
        raise ValueError(f"cudnn.sdpa: unsupported MXFP8 SF storage dtype {dtype}")
    if 0 in shape:
        return 0
    extent = 1
    for stride, size in sorted((st, n) for n, st in zip(shape, strides) if n > 1):
        if stride != extent:
            raise ValueError("cudnn.sdpa: MXFP8 SF requires dense non-overlapping storage in physical order")
        extent *= size
    return extent * width


class ThdLaunchSpec:
    """Plan-time facts of one THD launch (built by :func:`build_thd_spec`); read-only after
    build with immutable declared geometry."""

    __slots__ = (
        "fn",
        "owner",
        "order",
        "index",
        "template",
        "native",
        "quant",
        "b",
        "qh",
        "kh",
        "d_qk",
        "d_v",
        "paged",
        "page_size",
        "paged_hnd",
        "paged_v_hnd",
        "decl",
        "expect",
        "gate_expect",
        "has_lse",
        "has_sink",
        "lse_padded",
        "lse_fill_plan",
        "lse_head_major",
        "lse_head_stride",
        "lse_stride_override",
        "lse_stride",
        "s_q_max",
        "cga_tile_m",
        "wave_policy",
        "wave_geometry",
        "total_q",
        "total_kv",
        "fixed_batch",
        "workspace_alignment",
        "n_q_lens",
        "n_kv_lens",
        "lens_form",
        "off_o_desc",
        "scratch_bytes",
        "split_workspace",
        "neg_inf",
        "device_index",
        "origin_specs",
    )


# The host slot vocabulary the prepared launch binds: constants written once at build, and slots the native THD binder writes per call.
_FILLED_AT_BUILD = frozenset(
    "q_strides o_strides k_strides v_strides lse_strides lse_ext scale_softmax_log2 scale_softmax thd_max_sq n_thd_units seq_q_lens_addr thd_lens_form thd_origin_ptrs o_partial_ptr "
    "block_table_ptr block_table_v_ptr table_strides table_v_strides n_pages gate_ptr gate_strides ragged_q_addr ragged_q_div "
    "score_max_ptr score_sum_exp_ptr score_max_strides score_sum_exp_strides "
    # The bias / kv_mask slots of the sm107 f16 d128-file host (dense-only arms; a THD plan never requests them).
    "bias_ptr bias_strides bias_extents kv_mask_ptr kv_mask_batch_stride inv_scale_softmax".split()
)
_FILLED_PER_CALL = frozenset(
    "q_ptr k_ptr v_ptr o_ptr lse_ptr sinks_ptr meta_ptr o_desc_ptr problem_size k_strides v_strides lse_ext n_pages thd_q_lens_ptr thd_kv_lens_ptr "
    "block_table_ptr block_table_v_ptr table_strides table_v_strides gate_ptr gate_strides stream".split()
)


def _declared_paged_hnd(api, which_v=False) -> bool:
    """The operand's declared in-page layout: row stride below head stride in the
    kernel's (page, row, head, d) order. K also chooses the shared TMA coordinate order;
    the two descriptors carry independent strides in that order."""
    ps = api._paged_pool_stride(api.v_desc if which_v else api.k_desc)
    return bool(int(ps[1]) < int(ps[2]))


def _pool_is_hnd(f: BufferFacts) -> bool:
    """A (n_pages, KH, page_size, D) pool container is HND when its row stride is below its head stride."""
    return int(f.strides[2]) < int(f.strides[1])


def _covering(shape: Tuple[int, ...], strides: Tuple[int, ...]) -> bool:
    """No two distinct indices of ``shape`` address the same element: sorted by stride, each stride covers
    the extents below it (extent-1 axes are free)."""
    axes = sorted((int(st), int(sz)) for sz, st in zip(shape, strides) if int(sz) > 1)
    need = 1
    for st, sz in axes:
        if st < need:
            return False
        need = st * sz
    return True


@lru_cache(maxsize=256)
def _host_parameter_order(host) -> Tuple[str, ...]:
    """Immutable template-host slots; compiled wrappers are checked separately."""
    return tuple(n for n, p in inspect.signature(host).parameters.items() if "Constexpr" not in str(p.annotation))


def _positional_order(api) -> Tuple[Any, Any, List[str]]:
    """``(raw positional entry, compiled artifact, host slot order)`` of the adapter's compiled kernel;
    NotImplementedError when the artifact has no positional entry or the host is not the expected shape."""
    km = api._k_mod
    compiled = api._compiled_kernel
    host = km._host_prepared if (getattr(api, "_prepared_fp8", False) or getattr(api, "_prepared_mxfp8", False)) else km._host
    if getattr(api, "packed_thd_split", False):
        host = km._host_thd_split
    raw = positional_entry(compiled)
    if raw is None:
        raise NotImplementedError("the compiled artifact exposes no positional tvm-ffi entry")
    order = list(_host_parameter_order(host))
    if order[-1] != "stream":
        raise NotImplementedError(f"{km.__name__}: the host entry does not end with the stream parameter: {order[-3:]}")
    wrapper_sig = getattr(compiled, "_kwargs_wrapper", None) or compiled
    try:
        seen = list(inspect.signature(wrapper_sig).parameters)
    except (TypeError, ValueError):
        seen = None
    if seen is not None and seen != order:
        raise NotImplementedError(f"{km.__name__}: positional ABI {order} disagrees with the compiled wrapper {seen}")
    return raw, compiled, order


def build_thd_spec(api, *, scale_softmax: Optional[float]) -> ThdLaunchSpec:
    """The adapter's plan-time facts as a :class:`ThdLaunchSpec`; NotImplementedError when the
    artifact has no positional entry or the host signature is not the expected shape."""
    km = api._k_mod
    raw, compiled, order = _positional_order(api)
    plan = api._thd_plan()
    s = ThdLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.origin_specs = getattr(api, "_thd_origin_specs", None)
    s.quant = _quant_spec(api)
    s.b, s.qh, s.kh, s.d_qk, s.d_v = api.batch_size, api.h_q, api.h_kv, api.head_dim_qk, api.head_dim_v
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
    s.fixed_batch = bool(getattr(km, "PREPARED_FIXED_BATCH", False))
    s.workspace_alignment = int(getattr(km, "PREPARED_WORKSPACE_ALIGNMENT", _ALIGN_TMA))
    s.paged_hnd = _declared_paged_hnd(api) if s.paged else False
    s.paged_v_hnd = _declared_paged_hnd(api, which_v=True) if s.paged else False
    s.decl = dict(q=plan.q, k=plan.k, v=plan.v, o=plan.o)  # (h, d, token_stride, head_stride, elem_stride, row_span)
    s.expect = {n: str(getattr(api, f"{n}_desc").dtype).split(".")[-1] for n in ("q", "k", "v", "o")}
    # The kernel's epilogue gate on a THD plan (the f16/bf16 d256 prefill is the only THD body with the gate seams; it
    # maps G through the same packed [1, T, H, D] form as Q / O): G is a PER-CALL operand the native binder validates and
    # binds like Q / O -- the template keeps gate_ptr / gate_strides at None and the binder fills them from the gate's
    # own pack index.  Every other THD plan has no gate (api.check_support declines the quantized THD bodies by name).
    s.gate_expect = str(api.gate_desc.dtype).split(".")[-1] if getattr(api, "gate_desc", None) is not None else None
    if s.gate_expect is not None:
        s.decl["gate"] = api._thd_decl(api.gate_desc)
    s.has_lse, s.has_sink = api.lse_desc is not None, bool(api.has_sink)
    s.lse_padded, s.lse_head_major = bool(api.thd_stats_padded), bool(api.thd_stats_head_major)
    s.lse_head_stride = int(api.thd_stats_head_stride or 0)
    s.lse_stride_override = False
    s.lse_stride = tuple(int(x) for x in api._lse_stride) if s.lse_padded else None
    s.s_q_max = int(api.s_q_max)
    s.lse_fill_plan = None
    if s.has_lse and s.lse_padded:
        s.lse_fill_plan = _buffers.strided_fill_plan((s.b, s.qh, s.s_q_max), s.lse_stride) if s.s_q_max else ()
        if s.lse_fill_plan is None:
            raise ValueError("cudnn.sdpa: padded Stats strides must not overlap")
        s.lse_fill_plan = tuple(s.lse_fill_plan)
    s.cga_tile_m = int(plan.cga_tile_m)
    s.wave_policy = getattr(plan, "wave_policy", None)
    s.wave_geometry = getattr(plan, "wave_geometry", None)
    s.total_q = None if plan.total_q is None else int(plan.total_q)
    s.total_kv = None if plan.total_kv is None else int(plan.total_kv)
    s.n_q_lens, s.n_kv_lens, s.lens_form = int(plan.n_q_lens), int(plan.n_kv_lens), int(plan.lens_form)
    s.off_o_desc, s.scratch_bytes = int(plan.off_o_desc), int(plan.scratch_bytes)
    s.split_workspace = getattr(plan, "split_workspace", None)
    s.neg_inf = _buffers.init_word("fp32", float("-inf"))
    s.device_index = int(api.q_desc.device.index or 0)
    scale = float(api.scale_softmax if scale_softmax is None else scale_softmax)

    t: List[Any] = [None] * len(order)

    def put(name, value):
        if name in s.index:
            t[s.index[name]] = value

    q, k, v, o = plan.q, plan.k, plan.v, plan.o
    put("q_strides", (q[2], q[2], q[3]))
    put("o_strides", (o[2], o[2], o[3]))
    if not s.paged:
        put("k_strides", (k[2], k[2], k[3]))
        put("v_strides", (v[2], v[2], v[3]))
    if s.lse_padded:
        put("lse_strides", s.lse_stride)
        put("lse_ext", s.s_q_max)
    else:
        put("lse_strides", (0, 0, 0))
        put("lse_ext", s.lse_head_stride)  # compact head-major: the token capacity, written per call
    # Plans with negate_scores (SM100/SM107/SM120) run at |scale| (#1435).
    put("scale_softmax_log2", (-scale if getattr(api, "_score_negated", False) else scale) * math.log2(math.e))
    put("scale_softmax", scale)  # SM90 retains natural units, including literal zero.
    put("thd_max_sq", int(api.s_q_max))
    put("n_thd_units", int(plan.units))
    put("ragged_q_addr", 0)
    put("ragged_q_div", 1)
    put("seq_q_lens_addr", 0)
    put("thd_lens_form", s.lens_form)
    put("o_partial_ptr", None)
    put("block_table_ptr", None)
    put("block_table_v_ptr", None)
    put("table_strides", (0, 0))
    put("table_v_strides", (0, 0) if s.paged else None)
    put("n_pages", 0)
    put("gate_ptr", None)  # gate-capable hosts declare the slots; a GATED plan's binder fills both per call (s.gate_expect)
    put("gate_strides", (0, 0, 0))
    # Score side outputs: the dense-only slots of the shared host entry; a THD build never binds them (compile() declines).
    put("score_max_ptr", None)
    put("score_sum_exp_ptr", None)
    put("score_max_strides", (0, 0, 0))
    put("score_sum_exp_strides", (0, 0, 0))
    # Bias / kv_mask: dense-only slots of the same host; a THD build never binds them (the config factory declines).
    put("bias_ptr", None)
    put("bias_strides", (0, 0, 0))
    put("bias_extents", (1, 1))
    put("kv_mask_ptr", None)
    put("kv_mask_batch_stride", 0)
    put("inv_scale_softmax", 1.0)
    split_slots = {"lse_partial_ptr", "partial_o_strides"} if s.split_workspace is not None else set()
    unfilled = sorted(set(order) - _FILLED_AT_BUILD - _FILLED_PER_CALL - _quant_slots(s.quant) - split_slots)
    if unfilled:
        raise NotImplementedError(f"{km.__name__}: host slots {unfilled} are not bound by the prepared THD launch")
    s.template = t
    # This ABI has one production binder. A new host slot or unsupported
    # contract must fail here, never silently choose a Python implementation.
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    if s.gate_expect is not None and not getattr(_pybind_module._SdpaThdBinder, "supports_epilogue_gate", False):
        # The binder of an older extension has no gate role: a gated THD plan would launch with a null gate pointer.
        raise NotImplementedError(
            "cudnn.sdpa: the prepared THD binder of this build has no epilogue-gate role (the pybind extension predates the THD gate "
            "descriptor); rebuild the extension, or run the gate as its own stage"
        )
    if s.wave_policy is not None and not getattr(_pybind_module._SdpaThdBinder, "supports_initial_waves", False):
        raise NotImplementedError("cudnn.sdpa: this prepared THD launch requires runtime initial-wave binding; rebuild the extension")
    if s.wave_geometry is not None and not getattr(_pybind_module._SdpaThdBinder, "supports_initial_wave_geometry", False):
        raise NotImplementedError("cudnn.sdpa: this prepared THD launch requires packed initial-wave geometry; rebuild the extension")
    if s.origin_specs is not None and not getattr(_pybind_module._SdpaThdBinder, "supports_physical_origins", False):
        raise NotImplementedError("physical THD origins require rebuilding the native extension")
    if s.origin_specs is not None and s.paged and not getattr(_pybind_module._SdpaThdBinder, "supports_paged_physical_origins", False):
        raise NotImplementedError("paged physical THD origins require rebuilding the native extension")
    if (
        s.origin_specs is not None
        and s.split_workspace is not None
        and not getattr(_pybind_module._SdpaThdBinder, "supports_paged_split_physical_origins", False)
    ):
        raise NotImplementedError("paged split physical THD origins require rebuilding the native extension")
    if s.quant is not None and s.paged and not getattr(_pybind_module._SdpaThdBinder, "supports_paged_fp8_physical_origins", False):
        raise NotImplementedError("paged FP8 THD requires rebuilding the native extension")
    s.native = _pybind_module._SdpaThdBinder(s)
    return s


class ThdSplitWorkspace(NamedTuple):
    splits: int
    capacity: int
    off_o: int
    off_lse: int


def thd_split_workspace(base_bytes: int, splits: int, capacity: int, heads: int, d_v: int):
    """Fixed caller-owned partial regions shared by graph and standalone plans."""
    if capacity <= 0 or capacity > _I32_MAX or splits <= 1:
        raise ValueError("packed split requires positive Int32 capacity and split_kv > 1")
    off_o = (base_bytes + 255) // 256 * 256
    off_lse = off_o + splits * capacity * heads * d_v * 4
    scratch_bytes = (off_lse + splits * capacity * heads * 4 + 255) // 256 * 256
    return ThdSplitWorkspace(splits, capacity, off_o, off_lse), scratch_bytes


_NATIVE_THD_ROLES = ("q", "k", "v", "o", "q_lens", "kv_lens", "lse", "sinks", "block_table", "block_table_v")
_NATIVE_THD_INDICES = tuple(range(len(_NATIVE_THD_ROLES)))


def _set_native_fact(pack, index, fact):
    """Observation fallback only; the native evaluator still owns admission."""
    if fact is None:
        return
    code, bits = _buffers.DTYPES.get(fact.dtype, (0, 0))
    width = _buffers.DTYPE_ITEMSIZE.get(fact.dtype, 1)
    pack.set_operand(index, fact.ptr, fact.shape, fact.strides, code, bits, 1, fact.span * width if fact.span >= 0 else -1, *fact.device)


def _native_pack_from_facts(facts, roles=_NATIVE_THD_ROLES):
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    return _pybind_module._native_pack_from_facts(facts, roles, _buffers.DTYPES, _buffers.DTYPE_ITEMSIZE)


def execute_native_thd_tensors(spec, buffers, workspace_ptr, stream, scale, *, lse_bhs_geometry=None, gate=None):
    """Standalone observation, shared native validation/launch with graph.execute.

    Each invocation owns its pack and frame. Frameworks without the DLPack
    exchange API use the existing tensor observer only for the unread buffers.
    ``gate`` (appended): the epilogue gate G of a gated THD plan (``spec.gate_expect``), observed as one more operand
    of the pack OUTSIDE the role tuple and handed to the binder by its index -- the binder validates it like Q / O
    (dtype, device, alignment, declared strides, a capacity covering the packed Q tokens) and refuses a gate on an
    ungated plan or a missing one on a gated plan.
    """
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    roles = _NATIVE_THD_ROLES + _native_quant_roles(getattr(spec, "quant", None))
    if getattr(spec, "origin_specs", None) is not None:
        from .thd_origins import ROLES

        roles += ROLES
    buffers = tuple(buffers) + (None,) * (len(roles) - len(buffers))
    gate_index = -1
    if gate is not None:
        gate_index = len(buffers)
        buffers = buffers + (gate,)
    pack, unread = _pybind_module._read_buffer_sequence(buffers)
    for index in unread:
        _set_native_fact(pack, index, facts_of_tensor(buffers[index]))
    if lse_bhs_geometry is not None and buffers[6] is not None:
        lse = pack._facts_as((6,), BufferFacts, _DTYPE_BY_CODE)[0]
        shape, strides = lse_bhs_geometry
        if len(lse.shape) == 3 and lse.shape == shape and lse.strides[1:] == strides[1:]:
            # Standalone BHS and packed TH1 can have identical shapes at S=1.
            # Disambiguate metadata using the declaration, without a tensor view.
            _set_native_fact(pack, 6, lse._replace(shape=(*lse.shape, 1), strides=(*lse.strides, 1)))
    if gate_index >= 0:
        return spec.native.execute(pack, tuple(range(len(roles))), workspace_ptr, stream, scale, gate_index)
    return spec.native.execute(pack, tuple(range(len(roles))), workspace_ptr, stream, scale)


@lru_cache(maxsize=256)
def _paged_pool_layout(shape, strides, elem_bytes, hnd, kh, page_size, d):
    """Cache pure geometry only; addresses and observed storage are checked per call."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import dense_bind_strides

    if len(shape) != 4 or any(n <= 0 for n in shape) or shape[1] != kh or shape[2] != page_size or shape[3] != d:
        raise ValueError(f"a page pool is (n_pages, {kh}, {page_size}, {d}); got {shape}")
    if strides[3] != 1:
        raise ValueError("the page pool's head dim must be contiguous")
    if (strides[1] > strides[2]) != hnd:
        raise ValueError(f"this artifact was compiled for {'HND' if hnd else 'NHD'} page pools; got strides {strides}")
    # Match dense TMA admission in storage order, including singleton axes.
    ordered_shape = (shape[0], shape[2], shape[1], shape[3]) if hnd else shape
    ordered_strides = (strides[0], strides[2], strides[1], strides[3]) if hnd else strides
    bound = dense_bind_strides(ordered_shape, ordered_strides, elem_bytes)
    if bound is None:
        raise ValueError(f"page-pool strides must be covering and 16-byte aligned; got {shape} / {strides}")
    bs, ss, hs = (bound[0], bound[2], bound[1]) if hnd else bound
    need = (shape[0] - 1) * bs + (page_size - 1) * ss + (kh - 1) * hs + d
    return (bs, ss, hs), need


@lru_cache(maxsize=256)
def _paged_table_layout(shape, strides):
    """Normalize immutable table geometry; pointer/span checks stay per call."""
    if len(shape) == 4:
        if shape[1] != 1 or shape[3] != 1:
            raise ValueError(f"must be (B, 1, max_pages, 1); got {shape}")
        shape, strides = (shape[0], shape[2]), (strides[0], strides[2])
    if len(shape) != 2 or any(n <= 0 for n in shape):
        raise ValueError(f"must be nonempty (B, max_pages); got {shape}")
    if any(st < 0 for st in strides):
        raise ValueError(f"requires nonnegative strides; got {strides}")
    return shape, strides


class PreparedThdLaunch:
    """The graph plan's THD f16 launch: the spec plus this graph's operand uids."""

    def __init__(self, spec: ThdLaunchSpec, binding, *, stats_stride_override=False):
        if stats_stride_override and spec.has_lse and spec.lse_head_major and not spec.lse_padded and spec.quant is None:
            # Keep the standalone adapter's declared-stride contract immutable.
            # Only override-enabled graph plans bind an effective HN stride.
            spec = copy(spec)
            spec.lse_stride_override = True
            from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

            spec.native = _pybind_module._SdpaThdBinder(spec)
        self.spec = spec
        uids = {
            "q": binding.q.get_uid(),
            "k": binding.k.get_uid(),
            "v": binding.v.get_uid(),
            "o": binding.o.get_uid(),
            "q_lens": (binding.cu_seq_len_q if spec.lens_form & 1 else binding.seq_len_q).get_uid(),
            "kv_lens": (binding.cu_seq_len_kv if spec.lens_form & 2 else binding.seq_len_kv).get_uid(),
        }
        if binding.stats is not None:
            uids["lse"] = binding.stats.get_uid()
        if spec.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        if spec.paged:
            uids["block_table"] = binding.paged_k_table.get_uid()
            uids["block_table_v"] = binding.paged_v_table.get_uid()
        if spec.quant is not None:
            for name in _quant_roles(spec.quant):
                tensor = getattr(binding, name)
                if tensor is not None:
                    uids[name] = tensor.get_uid()
        if spec.origin_specs is not None:
            from .thd_origins import ROLES

            for role, origin in zip(ROLES, spec.origin_specs):
                if origin is not None:
                    uids[role] = getattr(binding, role).get_uid()
        self._roles = list(uids)
        self._uids = [uids[r] for r in self._roles]
        self._indices: Optional[List[int]] = None
        self._native_indices = None

    def _prepare_indices(self, index_of):
        try:
            self._indices = [index_of(u) for u in self._uids]
        except KeyError as exc:
            raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        roles = dict(zip(self._roles, self._indices))
        native_roles = _NATIVE_THD_ROLES + _native_quant_roles(self.spec.quant)
        if self.spec.origin_specs is not None:
            from .thd_origins import ROLES

            native_roles += ROLES
        self._native_indices = tuple(roles.get(role, -1) for role in native_roles)

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        if self._native_indices is None:
            self._prepare_indices(pack.index_of)
        self.spec.native.execute(pack.native, self._native_indices, workspace_ptr, stream)


# ---------------------------------------------------------------------------------------------------
# Dense (padded) launch: the same explicit host, every extent and stride read from the operands.
# ---------------------------------------------------------------------------------------------------

# Constants written at build, and slots the native dense binder writes per call.
_FILLED_AT_BUILD_DENSE = frozenset(
    "lse_strides lse_ext scale_softmax_log2 scale_softmax thd_max_sq n_thd_units seq_q_lens_addr thd_q_lens_ptr thd_kv_lens_ptr thd_lens_form thd_origin_ptrs o_partial_ptr "
    "block_table_ptr block_table_v_ptr table_strides table_v_strides n_pages gate_ptr gate_strides ragged_q_addr ragged_q_div "
    "score_max_ptr score_sum_exp_ptr score_max_strides score_sum_exp_strides "
    "bias_ptr bias_strides bias_extents kv_mask_ptr kv_mask_batch_stride inv_scale_softmax".split()
)
_FILLED_PER_CALL_DENSE = frozenset(
    "q_ptr k_ptr v_ptr o_ptr q_strides k_strides v_strides o_strides lse_ptr lse_strides sinks_ptr meta_ptr o_desc_ptr problem_size seq_q_lens_addr "
    "o_partial_ptr block_table_ptr block_table_v_ptr table_strides table_v_strides n_pages gate_ptr gate_strides ragged_q_addr stream "
    "score_max_ptr score_sum_exp_ptr score_max_strides score_sum_exp_strides "
    "bias_ptr bias_strides bias_extents kv_mask_ptr kv_mask_batch_stride inv_scale_softmax".split()
)


class SplitCombineSpec(NamedTuple):
    """Immutable combine artifact and workspace geometry; addresses are bound per call.

    ``gate`` (appended; None on every plan but the gate-in-combine one) is the
    :class:`CombineGate` that ``fn`` IS on such a plan: the gate's address is bound
    per call before the native launch and appended to the combine frame."""

    fn: Any
    owner: Any
    o: BufferFacts
    lse: BufferFacts
    lse_offset: int
    output_dtype: str
    has_stats: bool
    gate: Any = None


class CombineGate:
    """The gate-in-combine binding of a split plan whose kernel has no epilogue-gate
    seams (the d256 decode tile): ``O *= sigmoid(G)`` is applied by the split combine
    on the fp32 merged value (``sm100/split_combine.compile_ptr(gate=True)``).

    The native dense binder refuses a gate on a split (its gate slot belongs to the
    kernel's epilogue) and calls ``SplitCombineSpec.fn`` with a FIXED positional frame
    (partials, O, Stats, geometry, splits, strides, stream), so the gate reaches the
    combine here: :meth:`bind` records G's address and BSHD strides for the next
    launch of the calling thread, and this object is the ``fn`` the binder calls -- it
    appends G to the frame and forwards to the gated combine artifact.  One bind per
    launch: the binding is consumed by exactly one call (a stale gate can never serve
    a later launch), and a launch without one is an error, never an un-gated O.
    Thread-local, like the binder's own per-call frame (two threads may execute one
    plan concurrently with different buffers).  Under CUDA-graph capture the launch
    is recorded with the bound address, exactly as every other pointer of the frame.
    The binding also keeps the plan's CUDA-device check the native binder applies to
    every operand it binds -- the gate is not one of them on this path -- with the
    binder's own policy: a KNOWN device other than the plan's (a CPU gate, another
    GPU) is refused before any launch, an unknown one (a bare address) is admitted.
    """

    __slots__ = ("entry", "dtype", "shape", "device", "_tls")

    def __init__(self, entry, dtype: str, shape: Tuple[int, int, int, int], device: Tuple[int, int]):
        self.entry = entry  # the gated combine's positional tvm-ffi entry
        self.dtype = dtype  # G's dtype as a bare name ("bfloat16")
        self.shape = tuple(int(x) for x in shape)  # (B, H_q, S_q, D_v): the logical BHSD gate = O's shape
        self.device = (int(device[0]), int(device[1]))  # the plan's DLPack (device_type, device_id)
        self._tls = threading.local()

    def bind(self, facts: Optional[BufferFacts]) -> None:
        """Record G (a logical BHSD ``(B, H_q, S_q, D_v)`` buffer's facts) for the next combine launch."""
        if facts is None or not facts.ptr:
            raise ValueError("cudnn.sdpa: the gate G is required by this gate-in-combine specialization and must have a non-null address")
        if int(facts.device[0]) != -1 and (int(facts.device[0]), int(facts.device[1])) != self.device:
            raise ValueError("cudnn.sdpa: gate must be on this plan's CUDA device")  # the native binder's operand rule, kept here
        if facts.dtype != self.dtype:
            raise ValueError(f"cudnn.sdpa: gate dtype {facts.dtype} does not match the compiled gate-in-combine dtype {self.dtype}")
        shape = tuple(int(x) for x in facts.shape)
        if shape != self.shape:
            raise ValueError(f"cudnn.sdpa: gate must have O's logical shape {self.shape} (B, H_q, S_q, D_v); got {shape}")
        strides = tuple(int(x) for x in facts.strides)
        if len(strides) != 4 or any(st < 0 for st in strides):
            raise ValueError(f"cudnn.sdpa: gate needs four non-negative BHSD strides; got {strides}")
        if facts.span >= 0:
            need = 1 + sum((n - 1) * st for n, st in zip(shape, strides) if n > 0)
            if need > facts.span:
                raise ValueError(f"cudnn.sdpa: gate buffer spans {facts.span} elements but its declaration addresses {need}")
        sb, sh, ss, sd = strides
        self._tls.frame = (int(facts.ptr), (sb, ss, sh, sd))  # the combine views G as [B, S_q, H, D]: BSHD stride order

    def __call__(self, o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides, stream):
        frame = getattr(self._tls, "frame", None)
        if frame is None:
            raise RuntimeError("cudnn.sdpa: the gate-in-combine launch has no gate bound for this thread (bind() precedes every native execute)")
        self._tls.frame = None
        return self.entry(o_partial_ptr, lse_partial_ptr, o_out_ptr, lse_out_ptr, problem_size, n_splits, o_strides, lse_strides, frame[0], frame[1], stream)


class DenseLaunchSpec:
    """Plan-time facts of one dense f16 launch (built by :func:`build_dense_spec`); read-only after build."""

    __slots__ = (
        "fn",
        "owner",
        "order",
        "index",
        "template",
        "native",
        "native_roles",
        "native_indices",
        "quant",
        "b",
        "qh",
        "kh",
        "s_q_max",
        "s_k_max",
        "d_qk",
        "d_v",
        "paged",
        "page_size",
        "paged_hnd",
        "paged_v_hnd",
        "expect",
        "elem_bytes",
        "has_lse",
        "has_sink",
        # Score side outputs (score_max / score_sum_exp): dense fp32 (B, QH, SQ) rows bound per call like the Stats.
        "has_score_max",
        "has_score_sum_exp",
        # Additive bias (the resident row-block arm): compiled in, its dtype string, bound per call like the gate; the
        # per-key boolean mask likewise.  The python frame is the only binder of either (no native slot).
        "has_bias",
        "bias_expect",
        "has_kv_mask",
        "kv_mask_expect",
        "seq_kv_present",
        "seq_q_present",
        "gate_expect",
        "split",
        "fp32_partial",
        "combine",
        # The split sink combine takes the per-batch Q lengths (int32 (B,)) after the sink pointer: a plan with a split sink AND the dense padded-Q
        # trim (``seq_q_present``).  The combine cannot tell a padded query row from a valid keyless one without them (the native binder appends the
        # bound Q-length address to the combine frame).
        "combine_q_lens",
        "tile_n",
        "kv_tail_native",
        "causal",
        "causal_bottom_right",
        "window_right",
        "shape_fixed",
        "lpt_grid_fixed",
        "dense_flex",
        "device_index",
        # The decode tile's ragged-Q leg: Q / O / Stats are the caller's PACKED
        # buffers, their rows placed by the bound (B+1,) int32 ragged offsets
        # (elements) divided by ``ragged_divs`` (Q, O, Stats elements per token);
        # ``total_q`` is the declared packed Q token total (None = buffer-derived).
        "ragged",
        "ragged_divs",
        "ragged_i64",
        "ragged_lse_head_major",
        "total_q",
    )

    def frame(self) -> List[Any]:
        return list(self.template)

    def kv_tail_admitted(self, s_q: int, s_kv: int) -> bool:
        """The compiled artifact's KV-tail contract (the adapter's check_support rule, applied to the
        RUNTIME extents): a partial last KV tile is zero-filled by TMA but only MASKED on the padded /
        causal paths, so S_kv must be a tile multiple unless per-batch KV lengths carry the real lengths
        or the causal diagonal provably covers the tail. SM120 always masks its rightmost tile."""
        if getattr(self, "kv_tail_native", False) or self.paged or s_kv % self.tile_n == 0 or self.seq_kv_present:
            return True
        if not self.causal:
            return False
        return (self.causal_bottom_right and self.window_right == 0) or (not self.causal_bottom_right and s_q + self.window_right <= s_kv)


def _plan_device_cc(api) -> Tuple[int, int]:
    """The compute capability a plan lowers for, as its adapter recorded it at ``check_support``: the SM100 and SM90
    adapters keep ``_device_cc``, the SM120 adapter ``compute_capability``; a plan that recorded neither reads its Q
    tensor's device once more (the read the adapters make).  Never None: the split combine's form pin keys on it."""
    cc = getattr(api, "_device_cc", None) or getattr(api, "compute_capability", None)
    if cc is None:
        import torch

        cc = torch.cuda.get_device_capability(api.q_desc.device)
    return (int(cc[0]), int(cc[1]))


def _split_combine_decode_grid(api, b: int, qh: int, s_q_max: int, split: int) -> int:
    """The split combine entry's kernel FORM for a plan (``split_combine.DECODE_GRID_AUTO`` / ``DECODE_GRID_OFF`` /
    ``DECODE_GRID_LEAN_WALK``): a plan-time specialization on the declared B / H_q / S_q, the split and the plan's DEVICE, never
    a runtime value.  On cc 10.7 a lean-grid-OVERFLOW launch (``heuristics.decode_combine_lean_walk_selected``: the lean grid
    past the decode-grid kernel's bound, where that kernel is never eligible, at a split of at most
    ``heuristics.DECODE_COMBINE_LEAN_WALK_MAX_SPLIT``) names the module's LEAN GROUP WALK, its former lean kernel, measured there
    against the module's own forms (B 32 / B 128 x 24 heads at split 2-3: 3.3-8.9 us against 5.3-12.4 per launch on the Rubin
    performance class); a deeper split past the grid bound keeps the AUTO entry (= the module's own forms there, as before the
    restore); inside the bound the AUTO entry takes the decode-grid kernel on every sink-free
    launch and the split-aware rule (``heuristics.decode_combine_grid_selected`` -- measured on the same class: that kernel
    loses to the module's forms on a mid-sized grid below 32 splits and on any grid at 4) pins the module's own forms where the
    two disagree.  Off cc 10.7 (the Blackwell line keeps the module's own three-way rule, measured on B200 / B300; a line the
    rules were never measured on likewise) a plan never names the walk and never compiles the AUTO entry inside the bound
    (outside it the AUTO entry is the module's own forms by construction)."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.heuristics import decode_combine_force_lean, decode_combine_lean_walk_selected
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100 import split_combine

    device_cc = _plan_device_cc(api)
    if decode_combine_lean_walk_selected(b, qh, s_q_max, split, device_cc=device_cc):
        return split_combine.DECODE_GRID_LEAN_WALK
    if decode_combine_force_lean(b, qh, s_q_max, split, device_cc=device_cc):
        return split_combine.DECODE_GRID_OFF
    return split_combine.DECODE_GRID_AUTO


def build_dense_spec(api, *, scale_softmax: Optional[float]) -> DenseLaunchSpec:
    """The adapter's plan-time facts for a dense (padded) launch: the positional entry, the argument
    template with every constant filled, the fixed specialization the binder checks each call against,
    with no owned device allocations."""
    km = api._k_mod
    cfg = getattr(km, "CFG", None)
    raw, compiled, order = _positional_order(api)
    s = DenseLaunchSpec()
    s.fn, s.owner, s.order = raw, compiled, order
    s.index = {n: i for i, n in enumerate(order)}
    s.quant = _quant_spec(api)
    s.native_roles = _NATIVE_DENSE_ROLES + _native_quant_roles(s.quant)
    s.native_indices = tuple(range(len(s.native_roles)))
    s.b, s.qh, s.kh, s.d_qk, s.d_v = int(api.batch_size), int(api.h_q), int(api.h_kv), int(api.head_dim_qk), int(api.head_dim_v)
    s.s_q_max, s.s_k_max = int(api.s_q_max), int(api.s_k_max)
    if getattr(cfg, "PACK_GQA", False) and s.qh != s.kh * cfg.QH_PER_KH:
        raise ValueError("cudnn.sdpa: runtime head counts must match the compiled PackGQA ratio")
    # A decode tile with the TOKEN-UNIT axis (Q_TOKEN_UNITS) covers any S_q: its host entry cuts the tokens into
    # ceil(S_q / Q_BOX_TOKENS) units per head group; a tile without it serves one Q box only.
    if hasattr(km, "N_Q") and not getattr(km, "Q_TOKEN_UNITS", False) and s.s_q_max * km.HEADS_PER_TILE > km.N_Q:
        raise ValueError(f"cudnn.sdpa: decode query rows exceed the compiled {km.N_Q}-row tile")
    s.paged, s.page_size = bool(api.paged), int(api.paged_page_size or 0)
    s.paged_hnd = _declared_paged_hnd(api) if s.paged else False
    s.paged_v_hnd = _declared_paged_hnd(api, which_v=True) if s.paged else False
    s.split = int(api.split_kv)
    s.fp32_partial = bool(api._fp32_partial_split()) if s.split > 1 else False
    s.expect = {n: str(getattr(api, f"{n}_desc").dtype).split(".")[-1] for n in ("q", "k", "v", "o")}
    if s.split > 1:  # the main kernel writes the split-major partial slabs, in the partial dtype
        s.expect["o"] = str(api._partial_torch_dtype()).split(".")[-1]
    if s.quant is not None and s.quant.block_output is not None and s.quant.block_output.pack == 2:
        s.expect["o"] = "uint8"
    s.elem_bytes = {n: _buffers.DTYPE_ITEMSIZE[t] for n, t in s.expect.items()}
    s.has_lse = (api.lse_desc is not None) or s.split > 1
    s.has_sink = bool(api.has_sink)
    s.has_score_max = getattr(api, "score_max_desc", None) is not None
    s.has_score_sum_exp = getattr(api, "score_sum_exp_desc", None) is not None
    _bias_desc = getattr(api, "bias_desc", None)
    s.has_bias = _bias_desc is not None
    s.bias_expect = str(_bias_desc.dtype).split(".")[-1] if _bias_desc is not None else None
    _kv_mask_desc = getattr(api, "kv_mask_desc", None)
    s.has_kv_mask = _kv_mask_desc is not None
    # The kv_mask's declared dtype: a byte mask (bool / int8 / uint8 -> the bit word) or the ADDITIVE vector's io / fp32 dtype.
    s.kv_mask_expect = str(_kv_mask_desc.dtype).split(".")[-1] if _kv_mask_desc is not None else None
    s.seq_kv_present, s.seq_q_present = bool(api.seq_kv_lens_present), bool(api.seq_q_lens_present)
    # The kernel's epilogue gate.  A gate that rides the split COMBINE instead
    # (api._gate_in_combine: the d256 decode tile, split > 1) is NOT the binder's
    # -- it refuses a gate on a split -- and binds through SplitCombineSpec.gate.
    gate_dtype = str(api.gate_desc.dtype).split(".")[-1] if getattr(api, "gate_desc", None) is not None else None
    gate_in_combine = bool(gate_dtype is not None and s.split > 1 and api._gate_in_combine())
    s.gate_expect = None if gate_in_combine else gate_dtype
    s.tile_n = int(getattr(cfg, "TILE_N", getattr(api, "kv_tile", 128)))
    # SM120 always masks its rightmost KV tile; an SM100/SM107 plan compiled with kv_tail_mask masks it too.
    s.kv_tail_native = bool(getattr(km, "PREPARED_KV_TAIL_NATIVE", False) or getattr(api, "_kv_tail_mask", False))
    s.dense_flex = bool(getattr(km, "PREPARED_DENSE_FLEX", False))
    # the COMPILED mask kind: the d192 lowering may have rewritten a square bottom-right mask as top-left
    s.causal = bool(api.is_causal)
    s.causal_bottom_right = bool(getattr(cfg, "BOTTOM_RIGHT", getattr(api, "causal_bottom_right", False)))
    s.window_right = int(api.window_size_right or 0)
    # Lowering canonicalizations that read the DECLARED (S_q, S_kv) pin the runtime extents to them: the square
    # bottom-right -> top-left rewrite (equal only for S_q == S_kv) and the 8K LPT-L2 head grouping of the d192 flavor.
    requested_br = bool(getattr(api, "causal_bottom_right", False))
    s.shape_fixed = (s.causal and requested_br and not s.causal_bottom_right) or (  # compiled top-left for a requested bottom-right: square only
        tuple(getattr(api, "flavor", ()) or ()) == (192, 128) and s.s_q_max == s.s_k_max == 8192  # the 8K LPT-L2 head grouping
    )
    # A grouped-LPT schedule decodes tile coordinates from a COMPILED Q-tile count and head group (derived from the
    # declared S_q and batch x heads): such an artifact serves exactly the declared (batch, S_q). The f16 kernels
    # compile neither (both fields are the FP8 flavors'), so this pins nothing today and guards their joining.
    params = getattr(km, "PARAMS", None)
    s.lpt_grid_fixed = int(getattr(params, "lpt_head_group", 1)) > 1 or int(getattr(params, "lpt_q_tiles", 0)) > 0
    if getattr(km, "PREPARED_FIXED_SHAPE", False):
        s.shape_fixed = s.lpt_grid_fixed = True
    if s.quant is not None and (s.quant.sf_sizes or s.quant.block_output is not None):
        s.shape_fixed = s.lpt_grid_fixed = True  # dense SF batch/head pitches are plan-fixed
        if s.paged:
            # K/V SF pools page with K/V, so S_kv follows the block table. Batch and S_q stay pinned.
            s.shape_fixed = False
    s.device_index = int(api.q_desc.device.index or 0)
    # The decode tile's ragged-Q leg (api.thd_decode_leg): a split launch by construction.
    s.ragged = bool(getattr(api, "thd_decode_leg", False))
    s.ragged_divs = tuple(int(d) for d in getattr(api, "ragged_divisors", (1, 1, 1))) if s.ragged else (1, 1, 1)
    s.ragged_i64 = bool(getattr(api, "ragged_offsets_int64", False)) if s.ragged else False
    s.ragged_lse_head_major = bool(getattr(api, "thd_stats_head_major", False)) if s.ragged else False
    s.total_q = getattr(api, "max_total_seq_len_q", None) if s.ragged else None
    if s.ragged and (s.split < 2 or not s.paged or not getattr(cfg, "RAGGED_Q", 0)):
        raise NotImplementedError("cudnn.sdpa: the ragged-Q decode leg needs a split, paged launch of a RAGGED_Q-compiled decode tile")
    s.combine = None
    s.combine_q_lens = False
    if s.split > 1:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.api_dsl import ws_align
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.kernels.sm100 import split_combine
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

        if s.has_sink and s.quant is not None and not getattr(_pybind_module._SdpaDenseBinder, "supports_quantized_split_sink", False):
            raise NotImplementedError("cudnn.sdpa: FP8 split attention with a sink needs the native quantized sink-combine binder; rebuild the extension")
        s.combine_q_lens = bool(s.has_sink) and s.seq_q_present and not s.ragged
        if s.combine_q_lens and not getattr(_pybind_module._SdpaDenseBinder, "supports_combine_q_lens", False):
            raise NotImplementedError(
                "cudnn.sdpa: a split attention sink over padded query lengths needs the native binder that passes them to the combine; rebuild the extension"
            )

        # The combine kernel's FORM (a plan-time specialization: the declared B / H_q / S_q, the split and the plan's device,
        # never a runtime value; _split_combine_decode_grid states the rule and reads the device the adapter recorded).
        decode_grid = _split_combine_decode_grid(api, s.b, s.qh, s.s_q_max, s.split)
        owner = split_combine.compile_ptr(
            dtype_o=api._combine_dtype_tag(),
            dtype_partial=api._partial_dtype_tag(),
            has_lse=api.lse_desc is not None,
            stats_log2=api.stats_log2,
            ragged=s.ragged,
            ragged_i64=s.ragged_i64,
            quantized=s.quant is not None,
            has_amax=s.quant.has_amax if s.quant is not None else False,
            has_scale_o=api._split_scale_o() if s.quant is not None else False,
            has_scale_o_input=not (s.quant is not None and s.quant.sf_sizes),
            gate=gate_in_combine,
            dtype_gate=({"bfloat16": "bf16", "float16": "f16"}[gate_dtype] if gate_in_combine else None),
            # The plan's programmatic-dependent-launch form reaches the combine too (the dense half entries; a split tile
            # compiled with its griddepcontrol sites releases this pass early, and this pass waits before its first read).
            **({"use_pdl": True} if getattr(api, "use_pdl", False) else {}),
            decode_grid=decode_grid,
            # Every admitted split sink is folded once over sink-free partials.
            has_sink=bool(s.has_sink),
            has_q_lens=s.combine_q_lens,
        )
        fn = positional_entry(owner)
        if fn is None:
            raise NotImplementedError("the split combine artifact exposes no positional tvm-ffi entry")
        rows, sq, h, d = s.split * s.b, s.s_q_max, s.qh, s.d_v
        device = (_DLPACK_CUDA, s.device_index)
        combine_gate = CombineGate(fn, gate_dtype, (s.b, h, sq, d), device) if gate_in_combine else None
        o_size, lse_size = rows * sq * h * d, rows * h * sq
        s.combine = SplitCombineSpec(
            combine_gate if combine_gate is not None else fn,
            owner,
            BufferFacts(0, s.expect["o"], device, o_size, (rows, h, sq, d), (sq * h * d, d, h * d, 1)),
            BufferFacts(0, "float32", device, lse_size, (rows, h, sq), (h * sq, sq, 1)),
            ws_align(o_size * s.elem_bytes["o"]),
            str(api.o_desc.dtype).split(".")[-1],
            api.lse_desc is not None,
            combine_gate,
        )
    scale = float(api.scale_softmax if scale_softmax is None else scale_softmax)

    t: List[Any] = [None] * len(order)

    def put(name, value):
        if name in s.index:
            t[s.index[name]] = value

    put("lse_strides", (0, 0, 0))
    put("lse_ext", 0)
    # Score side outputs: None / zero strides in the template; bind_dense fills the requested ones per call.
    # A host that predates the slots has none (its plans never request the outputs: check_support declines).
    put("score_max_ptr", None)
    put("score_sum_exp_ptr", None)
    put("score_max_strides", (0, 0, 0))
    put("score_sum_exp_strides", (0, 0, 0))
    # Bias / kv_mask: None / zero slots in the template; bind_dense fills the requested ones per call (the artifact was
    # compiled with has_bias / has_kv_mask, so a compiled-in slot without its buffer is a binder error, not a null read).
    put("bias_ptr", None)
    put("bias_strides", (0, 0, 0))
    put("bias_extents", (1, 1))
    put("kv_mask_ptr", None)
    put("kv_mask_batch_stride", 0)
    # Plans with negate_scores (SM100/SM107/SM120) run at |scale| (#1435): BMM1 negates S, and EVERY per-score constant of the kernel is
    # bound at that one effective positive magnitude -- the exp2 scale and the additive arms' 1 / scale alike.  The bias / additive kv_mask
    # arms fold their operand pre-scale as ``S_raw + bias * inv_scale``; with S_raw = -(Q.K) and inv_scale = 1 / |scale| that is
    # |scale| * (-(Q.K) + bias / |scale|) = scale * Q.K + bias, while the SIGNED 1 / scale would enter the bias with its sign flipped.
    effective_scale = -scale if getattr(api, "_score_negated", False) else scale
    put("inv_scale_softmax", 1.0 / effective_scale if effective_scale != 0.0 else 1.0)
    put("scale_softmax_log2", effective_scale * math.log2(math.e))
    put("scale_softmax", scale)  # SM90 retains natural units, including literal zero.
    put("thd_max_sq", int(api.s_q_max))
    put("n_thd_units", 0)
    put("seq_q_lens_addr", 0)
    put("thd_q_lens_ptr", None)
    put("thd_kv_lens_ptr", None)
    put("thd_lens_form", None)
    put("o_partial_ptr", None)
    put("block_table_ptr", None)
    put("block_table_v_ptr", None)
    put("table_strides", (0, 0))
    put("table_v_strides", (0, 0) if s.paged else None)
    put("n_pages", 0)
    put("gate_ptr", None)
    put("gate_strides", (0, 0, 0))
    put("ragged_q_addr", 0)
    put("ragged_q_div", s.ragged_divs[0] if s.ragged else 1)
    if s.ragged and "ragged_q_addr" not in s.index:
        raise NotImplementedError(f"{km.__name__}: the ragged-Q decode leg needs the ragged_q_addr host slot")
    unfilled = sorted(set(order) - _FILLED_AT_BUILD_DENSE - _FILLED_PER_CALL_DENSE - _quant_slots(s.quant))
    if unfilled:
        raise NotImplementedError(f"{km.__name__}: host slots {unfilled} are not bound by the prepared dense launch")
    s.template = t
    # Every admitted dense template, including staged compact cores, owns
    # the native contract. The constructor validates its complete host ABI.
    # One exception: the score side outputs and the resident-block arms (bias / kv_mask) bind through the python frame
    # (``bind_dense`` below) -- the native binder's slot vocabulary predates their host slots (it copies the template's
    # None / zero slots untouched), so a plan that requests them keeps ``native`` None and ``bind_dense`` fills the
    # pointer / stride slots per call.
    if s.has_score_max or s.has_score_sum_exp or s.has_bias or s.has_kv_mask:
        s.native = None
        return s
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    s.native = _pybind_module._SdpaDenseBinder(s)
    return s


# ---- The python-frame binding of the plans the native binder cannot bind (score side outputs, bias, kv_mask):
# the role helpers below, ``_dense_lse``, ``bind_dense`` and ``bind_dense_split`` are the pre-native dense frame, kept for
# exactly those plans (``build_dense_spec`` leaves their ``native`` None); every other dense plan binds natively.


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
    # and each operand's declared in-page layout kind (the descriptor carries that pool's own strides)
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


def _capacity(f: BufferFacts, geo: Tuple[int, int, int, int], name: str) -> int:
    """Token capacity of a packed operand under the resolved (ts, hs, es, row_span) geometry: the largest token whose row ends inside
    the producer's span (0 when the span holds no whole row); a bare address (span unknown) is a typed decline -- the ragged legs
    bound every row by the capacity, so an unsized buffer cannot be bound."""
    if f.span < 0:
        raise ValueError(f"cudnn.sdpa: {name} was passed as a bare address; a ragged operand needs a sized buffer")
    ts, row = geo[0], geo[3]
    return 0 if f.span < row else (f.span - row) // ts + 1


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


def _bias_role(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], b: int, s_q: int, s_kv: int):
    """The additive bias of a dense launch: ``(ptr, (batch, head, row) element strides, (B_b, H_b) extents)``.

    ``[B_b, H_b, S_q, S_kv]`` in the io dtype with ``B_b`` in {1, b} and ``H_b`` in {1, QH} (the broadcast forms the kernel
    serves by clamping its coordinates), the key dim contiguous, every other live stride a 16-byte multiple (the TMA
    global-stride rule at 2 B/elem -- ``S_kv % 8 == 0`` for a compact bias), 16-byte aligned base, on the plan's device;
    a buffer smaller than its geometry claims is rejected."""
    f = facts.get("bias")
    if f is None:
        raise ValueError("cudnn.sdpa: bias_tensor is required by this compiled specialization (constructed with sample_bias)")
    if f.dtype != spec.bias_expect:
        raise ValueError(f"cudnn.sdpa: bias: runtime buffer dtype {f.dtype} does not match its declaration ({spec.bias_expect})")
    _on_plan_device(spec, "bias", f)
    shape, st = tuple(int(x) for x in f.shape), tuple(int(x) for x in f.strides)
    if len(shape) != 4 or shape[2:] != (s_q, s_kv) or shape[0] not in (1, b) or shape[1] not in (1, spec.qh):
        raise ValueError(f"cudnn.sdpa: bias must be (1|{b}, 1|{spec.qh}, {s_q}, {s_kv}); got {shape}")
    if st[3] != 1:
        raise ValueError(f"cudnn.sdpa: bias must have its key dim contiguous (stride 1); got strides {st}")
    elem = _buffers.DTYPE_ITEMSIZE[spec.bias_expect]
    live = [(n, s) for n, s in zip(shape[:3], st[:3]) if n > 1]
    if any((s * elem) % _ALIGN_TMA for _, s in live) or f.ptr % _ALIGN_TMA:
        raise ValueError(f"cudnn.sdpa: bias strides {st} and base must be 16-byte multiples (TMA; S_kv % {_ALIGN_TMA // elem} == 0 for a compact bias)")
    need = 1 + sum((n - 1) * s for n, s in live) + (s_kv - 1)
    if f.span >= 0 and f.span < need:
        raise ValueError(f"cudnn.sdpa: bias spans {f.span} elements; its geometry {shape} / {st} needs {need}")
    strides = tuple(s if n > 1 else 0 for n, s in zip(shape[:3], st[:3]))
    return f.ptr, strides, (shape[0], shape[1])


def _kv_mask_role(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], b: int, s_kv: int):
    """The per-key mask of a dense launch: ``(ptr, batch element stride)``.  ``(1|b, 1, 1, S_kv)`` (or the squeezed
    ``(1|b, S_kv)``), the key dim contiguous; a batch extent of 1 binds stride 0 (one row broadcast over the batch).  A
    byte mask (torch.bool / int8 / uint8, the bit-word form) when the plan was compiled for one; otherwise the ADDITIVE
    vector in exactly its declared dtype (``spec.kv_mask_expect``: the io dtype or fp32), which the kernel reads as
    warp-uniform 16-byte granules -- its base and batch stride must be 16-byte aligned and its extent whole granules."""
    f = facts.get("kv_mask")
    if f is None:
        raise ValueError("cudnn.sdpa: kv_mask_tensor is required by this compiled specialization (constructed with sample_kv_mask)")
    expect = getattr(spec, "kv_mask_expect", None)
    additive = expect not in (None, "bool", "int8", "uint8")
    if additive:
        if f.dtype != expect:
            raise ValueError(f"cudnn.sdpa: kv_mask: runtime buffer dtype {f.dtype} does not match its declaration ({expect})")
    elif f.dtype not in ("bool", "int8", "uint8"):
        raise ValueError(f"cudnn.sdpa: kv_mask must be a byte mask (torch.bool / int8 / uint8); got {f.dtype}")
    _on_plan_device(spec, "kv_mask", f)
    shape, st = tuple(int(x) for x in f.shape), tuple(int(x) for x in f.strides)
    if len(shape) == 4 and shape[1:3] == (1, 1):
        shape, st = (shape[0], shape[3]), (st[0], st[3])
    if len(shape) != 2 or shape[1] != s_kv or shape[0] not in (1, b) or st[1] != 1:
        raise ValueError(f"cudnn.sdpa: kv_mask must be (1|{b}, 1, 1, {s_kv}) with a contiguous key dim; got shape {f.shape} strides {f.strides}")
    batch_stride = st[0] if shape[0] > 1 else 0
    need = (shape[0] - 1) * batch_stride + s_kv
    if f.span >= 0 and f.span < need:
        raise ValueError(f"cudnn.sdpa: kv_mask spans {f.span} elements; its geometry needs {need}")
    if additive:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm107 import KV_MASK_ADDITIVE_GRANULE_ELEMS

        elem = _buffers.DTYPE_ITEMSIZE[f.dtype]
        if f.ptr % _ALIGN_TMA or (batch_stride * elem) % _ALIGN_TMA or s_kv % KV_MASK_ADDITIVE_GRANULE_ELEMS:
            raise ValueError(
                f"cudnn.sdpa: an additive kv_mask vector is read as 16-byte granules: its base and batch stride must be 16-byte multiples and "
                f"S_kv % {KV_MASK_ADDITIVE_GRANULE_ELEMS} == 0; got base % 16 = {f.ptr % _ALIGN_TMA}, strides {st}, S_kv {s_kv}"
            )
    return f.ptr, batch_stride


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


@lru_cache(maxsize=256)
def _packed_role_layout(shape, strides, heads, d, width, tma, name):
    """Pure packed geometry shared by Python and native ragged decode binding."""
    sh, st = shape, strides
    if len(sh) != len(st):
        raise ValueError("cudnn.sdpa: operand shape and stride must have the same rank")
    if len(sh) == 4:
        if sh[1] != heads or sh[3] != d:
            raise ValueError(f"cudnn.sdpa: {name}: this plan was built for {heads} heads of dim {d}; got {sh}")
        ts, hs, es = st[2], st[1], st[3]
    elif len(sh) == 3:
        if sh[1] != heads or sh[2] != d:
            raise ValueError(f"cudnn.sdpa: {name}: a packed THD buffer is (T, {heads}, {d}); got {sh}")
        ts, hs, es = st[0], st[1], st[2]
    else:
        raise ValueError(f"cudnn.sdpa: {name}: a ragged operand is (T, H, D) or the graph's (B, H, S, D); got rank {len(sh)}")
    if es != 1:
        raise ValueError(f"cudnn.sdpa: {name}: the head dim must be contiguous (elem stride 1); got {es}")
    if hs < d or (tma and (hs * width) % _ALIGN_TMA != 0):
        raise ValueError(f"cudnn.sdpa: {name}: head stride {hs} must cover {d} elements" + (" and be a 16-byte multiple" if tma else ""))
    if ts < (heads - 1) * hs + d or (tma and (ts * width) % _ALIGN_TMA != 0):
        raise ValueError(f"cudnn.sdpa: {name}: token stride {ts} must cover the {heads} heads" + (" and be a 16-byte multiple" if tma else ""))
    row = (heads - 1) * hs + (d - 1) + 1
    return ts, hs, row


@lru_cache(maxsize=256)
def _dense_role_layout(shape, strides, heads, d, s_max, b_max, elem_bytes, tma, name, dense_flex=False):
    """Cache geometry only; current addresses, device, dtype and span stay per-call."""
    if len(shape) != 4:
        raise ValueError(f"cudnn.sdpa: {name}: a dense operand is (B, H, S, D); got {tuple(shape)}")
    b, h, seq, dd = (int(x) for x in shape)
    if h != heads or dd != d:
        raise ValueError(f"cudnn.sdpa: {name}: this plan was built for {heads} heads of dim {d}; got {tuple(shape)}")
    if b < 1 or b > b_max:
        raise ValueError(f"cudnn.sdpa: {name}: batch {b} is outside this plan's envelope (1..{b_max})")
    if seq < 1 or seq > s_max:
        raise ValueError(f"cudnn.sdpa: {name}: sequence length {seq} is outside this plan's envelope (1..{s_max})")
    # ONE layout predicate for admission and execution (the lowering's attach decision uses it too):
    # singleton axes canonicalized, then BSHD-compact or the zero-copy rule
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm100 import dense_bind_strides

    if dense_flex:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.fwd.config_sm90 import dense_bind_strides

    if tma:
        bound = dense_bind_strides((b, h, seq, dd), tuple(int(x) for x in strides), elem_bytes)
    else:
        from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import dense_layout_ok

        if not dense_layout_ok(shape, strides):
            raise ValueError(f"cudnn.sdpa: {name}: split output needs contiguous D and non-overlapping strides; got {shape} / {strides}")
        st = tuple(int(v) if int(n) > 1 else 0 for n, v in zip(shape, strides))
        bound = st[0], st[2], st[1]
    if bound is None:
        rule = (
            "contiguous, every live B/H/S byte stride a 16-byte multiple, and the layout a covering permutation (config_sm90.dense_bind_strides)"
            if dense_flex
            else "contiguous, seq and head strides 16-byte multiples, and the layout token-major and covering (config_sm100.dense_bind_strides)"
        )
        raise ValueError(
            f"cudnn.sdpa: {name}: shape {tuple(shape)} strides {tuple(strides)} is not a layout the kernel binds zero-copy: the head dim must be {rule}"
        )
    bs, ss, hs = bound
    need = (b - 1) * bs + (h - 1) * hs + (seq - 1) * ss + d
    return (bs, ss, hs), b, seq, need


@lru_cache(maxsize=256)
def _dense_lse_layout(shape, strides, b, s_q, qh, name: str = "lse_tensor"):
    """Stats layout/envelope validation contains no runtime storage observations."""
    sh, st = tuple(int(x) for x in shape), tuple(int(x) for x in strides)
    if len(sh) == 4 and sh[3] == 1:
        sh, st = sh[:3], st[:3]
    if len(sh) != 3 or sh[0] < b or sh[1] != qh or sh[2] < s_q:
        raise ValueError(f"cudnn.sdpa: {name} must be ({b}, {qh}, {s_q}[, 1]) or larger; got {tuple(shape)}")
    need = (b - 1) * st[0] + (qh - 1) * st[1] + (s_q - 1) * st[2] + 1
    if not _covering((b, qh, s_q), st):
        raise ValueError(f"cudnn.sdpa: {name} strides {st} alias distinct (batch, head, row) entries onto one address (a write race)")
    return (st[0], st[1], st[2]), need


def _dense_lse(spec: DenseLaunchSpec, lse: Optional[BufferFacts], b: int, s_q: int, *, required: bool, name: str = "lse_tensor", sample: str = "sample_lse"):
    """Validate main or final Stats -- or a score side output (``score_max`` / ``score_sum_exp``, the same dense fp32
    (B, QH, SQ) row geometry) -- with the same device, span and non-aliasing contract."""
    if required:
        if lse is None:
            raise ValueError(f"cudnn.sdpa: {name} is required by this compiled specialization")
        _on_plan_device(spec, name, lse)
        if lse.dtype != "float32":
            raise ValueError(f"cudnn.sdpa: {name} must be float32; got {lse.dtype}")
        if lse.ptr % _ALIGN_F32 != 0:
            raise ValueError(f"cudnn.sdpa: {name} must be 4-byte aligned")
        strides, need = _dense_lse_layout(tuple(lse.shape), tuple(lse.strides), b, s_q, spec.qh, name)
        if lse.span >= 0 and lse.span < need:
            raise ValueError(f"cudnn.sdpa: {name} spans {lse.span} elements; its geometry needs {need}")
        return lse.ptr, strides
    elif lse is not None:
        raise ValueError(f"cudnn.sdpa: this specialization was compiled without a {name} output; construct the API without {sample}")

    return None, (0, 0, 0)


def bind_dense(spec: DenseLaunchSpec, facts: Dict[str, Optional[BufferFacts]], stream, stream_int: int) -> List[Any]:
    """This call's argument frame for a dense launch from the operands' facts (roles ``q k v o`` and,
    as compiled, ``lse sinks seq_kv seq_q gate`` and the paged ``block_table block_table_v``). Under a
    split the ``o`` / ``lse`` roles are the split-major partial slabs (batch extent split * B)."""
    ix = spec.index
    frame = spec.frame()
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
    if not spec.kv_tail_admitted(s_q, s_kv):
        raise ValueError(
            f"cudnn.sdpa: S_kv ({s_kv}) must be a multiple of {spec.tile_n} for this artifact unless per-batch KV lengths are present or the "
            f"causal mask covers the KV tail (the compiled specialization does not mask a partial last tile)"
        )
    # The spare sixth slot carries the packed Q token capacity on the ragged-Q leg (0 otherwise).
    frame[ix["problem_size"]] = (b, spec.qh, spec.kh, s_q, s_kv, q_tokens)
    if spec.fp32_partial:
        frame[ix["o_partial_ptr"]] = o.ptr

    frame[ix["lse_ptr"]], frame[ix["lse_strides"]] = _dense_lse(spec, facts.get("lse"), b * spec.split, s_q, required=spec.has_lse)
    for name in ("score_max", "score_sum_exp"):
        # Score side outputs (dense, unsplit plans only -- the batch extent is b, never b * split): the Stats
        # contract, per REQUESTED output; an unrequested one keeps the template's absent value (the native
        # binder's frame is the template too, so the two frames stay identical slot for slot).  A host without
        # the slots never has a requesting plan (check_support declines before any launch).
        if getattr(spec, f"has_{name}", False):
            frame[ix[f"{name}_ptr"]], frame[ix[f"{name}_strides"]] = _dense_lse(
                spec, facts.get(name), b, s_q, required=True, name=f"{name}_tensor", sample=f"sample_{name}"
            )
        elif facts.get(name) is not None:
            raise ValueError(f"cudnn.sdpa: this specialization was compiled without a {name} output; construct the API without sample_{name}")

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
    # Additive bias / per-key mask (the resident-block arms): the strict presence contract of the LSE in both directions.
    if getattr(spec, "has_bias", False):
        frame[ix["bias_ptr"]], frame[ix["bias_strides"]], frame[ix["bias_extents"]] = _bias_role(spec, facts, b, s_q, s_kv)
    elif facts.get("bias") is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without a bias; construct the API with sample_bias")
    if getattr(spec, "has_kv_mask", False):
        frame[ix["kv_mask_ptr"]], frame[ix["kv_mask_batch_stride"]] = _kv_mask_role(spec, facts, b, s_kv)
    elif facts.get("kv_mask") is not None:
        raise ValueError("cudnn.sdpa: this specialization was compiled without a kv_mask; construct the API with sample_kv_mask")
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


@lru_cache(maxsize=256)
def _ragged_lse_layout(shape, strides, heads, head_major_decl):
    """Pure packed Stats geometry; runtime capacity never enters this cache."""
    sh, st = shape, strides
    if len(sh) != len(st):
        raise ValueError("cudnn.sdpa: operand shape and stride must have the same rank")
    if len(sh) == 4 and sh[3] == 1:
        sh, st = sh[:3], st[:3]
    if len(sh) == 3:  # the graph's (B, H, S_max) declaration
        if sh[1] != heads:
            raise ValueError(f"cudnn.sdpa: lse_tensor must carry {heads} heads; got {shape}")
        stride_h, stride_s = st[1], st[2]
    elif len(sh) == 2:  # packed (T, H) token-major or (H, T) head-major storage
        # A square rank-2 buffer has identical physical metadata under NH
        # and HN. The plan's declared packing resolves that ambiguity.
        if sh[1] == heads and st[1] == 1 and not (head_major_decl and sh[0] == heads):
            stride_h, stride_s = 1, st[0]
        elif sh[0] == heads and st[1] == 1:
            stride_h, stride_s = st[0], 1
        else:
            raise ValueError(f"cudnn.sdpa: a packed ragged Stats buffer is (T, {heads}) or ({heads}, T) with unit inner stride; got {shape} / {strides}")
    else:
        raise ValueError(f"cudnn.sdpa: ragged Stats is (B, H, S_max[, 1]) or packed rank-2; got {shape}")
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.sdpa.graph_analyzer import thd_stats_packing

    packing = thd_stats_packing(stride_h, stride_s, heads)
    if packing is None:
        raise ValueError(f"cudnn.sdpa: ragged Stats must be packed token-major (stride_h == 1, stride_s == H) or head-major (stride_s == 1); got strides {st}")
    return stride_h, stride_s, (heads - 1) * stride_h + 1, packing == "head_major"


class PreparedDenseLaunch:
    """The graph plan's dense f16 launch: the spec plus this graph's operand uids."""

    def __init__(self, spec: DenseLaunchSpec, binding, *, seq_kv_src=None, seq_q_src=None, gate_src=None):
        self.spec = spec
        uids = {"q": binding.q.get_uid(), "k": binding.k.get_uid(), "v": binding.v.get_uid(), "o": binding.o.get_uid()}
        if binding.stats is not None:
            uids["lse"] = binding.stats.get_uid()
        if getattr(spec, "has_score_max", False):
            uids["score_max"] = binding.score_max.get_uid()
        if getattr(spec, "has_score_sum_exp", False):
            uids["score_sum_exp"] = binding.score_sum_exp.get_uid()
        if spec.has_sink:
            uids["sinks"] = binding.sink_token.get_uid()
        if spec.seq_kv_present:
            uids["seq_kv_lens"] = seq_kv_src.get_uid()
        if spec.seq_q_present:
            uids["seq_q_lens"] = seq_q_src.get_uid()
        if spec.paged:
            uids["block_table"] = binding.paged_k_table.get_uid()
            uids["block_table_v"] = binding.paged_v_table.get_uid()
        if spec.gate_expect is not None:
            uids["gate"] = gate_src.get_uid()
        elif spec.combine is not None and spec.combine.gate is not None:
            # The gate rides the combine: NOT a native role (the binder refuses a gate on a
            # split) -- its facts are read off the pack per call and bound to the combine.
            if gate_src is None:
                raise ValueError("cudnn.sdpa: the gate-in-combine plan needs the graph's gate operand")
            uids["combine_gate"] = gate_src.get_uid()
        # The resident-block arms: the bias / kv_mask operands of the graph (bound like the gate; bind_dense validates them).
        if getattr(spec, "has_bias", False):
            uids["bias"] = binding.bias.get_uid()
        if getattr(spec, "has_kv_mask", False):
            uids["kv_mask"] = binding.kv_mask.get_uid()
        if spec.ragged:
            uids["ragged_q"] = binding.ragged_q.get_uid()
            uids["ragged_o"] = binding.ragged_o.get_uid()
            if spec.combine is not None and spec.combine.has_stats:
                uids["ragged_lse"] = binding.ragged_stats.get_uid()
        if spec.quant is not None:
            for name in _quant_roles(spec.quant):
                tensor = getattr(binding, name)
                if tensor is not None:
                    uids[name] = tensor.get_uid()
        self._roles = list(uids)
        self._uids = [uids[r] for r in self._roles]
        self._indices: Optional[List[int]] = None
        self._native_indices = None
        self._combine_gate_index: Optional[int] = None

    def _prepare_indices(self, index_of):
        try:
            self._indices = [index_of(u) for u in self._uids]
        except KeyError as exc:
            raise ValueError(f"cudnn.sdpa: tensor uid {exc} is bound by the plan but is not an operand of this graph") from exc
        roles = dict(zip(self._roles, self._indices))
        self._native_indices = tuple(roles.get(role, -1) for role in self.spec.native_roles)
        self._combine_gate_index = roles.get("combine_gate")
        return self._indices

    def execute(self, pack, workspace_ptr: int, stream, stream_int: int) -> None:
        indices = self._indices
        if indices is None:
            indices = self._prepare_indices(pack.index_of)
        if self._combine_gate_index is not None:
            # One native crossing for G's facts, bound to the gated combine for this launch (no sync, no allocation).
            self.spec.combine.gate.bind(facts_of_roles(pack, [self._combine_gate_index])[0])
        if self.spec.native is not None:
            self.spec.native.execute(pack.native, self._native_indices, stream, workspace=workspace_ptr)
            return
        # The plans without a native binder (the score side outputs, the bias / kv_mask arms -- build_dense_spec leaves
        # ``native`` None for exactly those; always dense half plans, never quantized) bind through the python frame from
        # the pack's facts, validated per call by bind_dense / bind_dense_split.
        facts = dict(zip(self._roles, facts_of_roles(pack, indices)))
        if self.spec.combine is not None:
            bound = bind_dense_split(self.spec, facts, workspace_ptr, stream, stream_int)
            if bound is None:
                return  # ragged-Q leg: no addressable token / empty producer -> no launch
            frame, combine_args = bound
            self.spec.fn(*frame)
            self.spec.combine.fn(*combine_args)
        else:
            self.spec.fn(*bind_dense(self.spec, facts, stream, stream_int))


_NATIVE_DENSE_ROLES = (
    "q",
    "k",
    "v",
    "o",
    "lse",
    "sinks",
    "seq_kv_lens",
    "seq_q_lens",
    "block_table",
    "block_table_v",
    "gate",
    "ragged_q",
    "ragged_o",
    "ragged_lse",
)
_NATIVE_DENSE_INDICES = tuple(range(len(_NATIVE_DENSE_ROLES)))


def execute_native_dense_tensors(spec, buffers, stream, scale, workspace_ptr=0):
    """Observe standalone buffers once; scale uses the selected host's units."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _pybind_module

    buffers = tuple(buffers) + (None,) * (len(_NATIVE_DENSE_ROLES) - len(buffers))
    if spec.combine is not None and spec.combine.gate is not None:
        # The gate rides the combine (the decode tile's split): bind it there and hand
        # the native binder an empty gate slot (it refuses a gate on a split).
        gi = _NATIVE_DENSE_ROLES.index("gate")
        spec.combine.gate.bind(facts_of_tensor(buffers[gi]))
        buffers = buffers[:gi] + (None,) + buffers[gi + 1 :]
    pack, unread = _pybind_module._read_buffer_sequence(buffers)
    for index in unread:
        _set_native_fact(pack, index, facts_of_tensor(buffers[index]))
    indices = tuple(range(len(buffers))) if spec.quant is not None else _NATIVE_DENSE_INDICES
    return spec.native.execute(pack, indices, stream, scale, workspace_ptr)
