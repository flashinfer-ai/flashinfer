# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Plan and launch of the sparse MLA classes on their own kernel sources (:mod:`.kernel`).

A plan is one compiled shape: device, head count, Q / KV dtype, query width and whether each scale is a host float or
a device tensor. The row count and the list width are launch-time extents, so one plan serves every batch size and
every ``top_k``. Nothing here reads ``flashinfer.attention.prims_ts``."""

from __future__ import annotations

import hashlib
import numbers
import os
import threading
from typing import Dict, Optional, Union

import torch

from ...api_logging import flashinfer_api
from ..support import check_device, unsupported
from . import spec

Scale = Union[float, torch.Tensor]
_JIT_MODULE = "batch_invariant_mla_sparse"
_DTYPE_KEYS = {torch.bfloat16: "bf16", torch.float8_e4m3fn: "e4m3"}
#: Host scales that round to a positive fp32 and stay finite when the kernel multiplies them by log2(e).
_MIN_SCALE, _MAX_SCALE = 1e-45, 2.35e38

#: Plans per compiled shape. Never evicted: a captured CUDA graph reads its plan's buffers.
_plans: Dict[tuple, "SparseMLAPlan"] = {}
#: Full-width list lengths: backing tensors per (device, top_k) and their views per row count.
_full: Dict[tuple, tuple] = {}
_full_views: Dict[tuple, torch.Tensor] = {}
_lock = threading.Lock()
_build_locks: Dict[tuple, threading.Lock] = {}
#: Plans and buffers set aside by :func:`clear_plans`: captured CUDA graphs still launch and read them.
_retired: list = []
#: The bf16 classes launch one grid slot per row.
_MAX_ROWS = {torch.bfloat16: 65535}


def clear_plans() -> None:
    """Forget every cached plan and buffer; the next call of a shape builds it again.

    The forgotten plans stay loaded and their buffers stay allocated for the life of the process, so a CUDA graph
    captured through one keeps replaying."""
    with _lock:
        _retired.append((dict(_plans), dict(_full), dict(_full_views)))
        _plans.clear()
        _full.clear()
        _full_views.clear()


def _settle(device) -> None:
    # A kept buffer is filled on the creating stream; other streams read it from the cache.
    torch.cuda.current_stream(device).synchronize()


def _capacity(rows: int) -> int:
    return max(64, 1 << (rows - 1).bit_length())


def capturing(device) -> bool:
    """Whether the current stream of ``device`` (not of the caller's current device) is capturing a CUDA graph."""
    if device.type != "cuda":
        return False
    if device.index is None or device.index == torch.cuda.current_device():
        return torch.cuda.is_current_stream_capturing()
    with torch.cuda.device(device):
        return torch.cuda.is_current_stream_capturing()


def _rows_share(first, last, step, width, rows, lo, hi) -> bool:
    """Whether bytes ``[lo, hi)`` meet a list row ``[first + r * step, + width)``; ``last`` ends the last row."""
    if not (first < hi and lo < last):
        return False
    if step <= width:
        return True
    row = max(0, (lo - first - width) // step + 1)
    return row < rows and first + row * step < hi


def _not_capturing(what: str, device) -> None:
    if capturing(device):
        raise RuntimeError(
            f"{what}: run one call of this shape before CUDA graph capture"
        )


def full_lengths(device, rows: int, top_k: int) -> torch.Tensor:
    """``[rows]`` int32 lengths equal to ``top_k``: a view of a kept buffer, so no row count allocates twice."""
    view = _full_views.get((device, rows, top_k))
    if view is None:
        _not_capturing(spec.FAMILY, device)
        with _lock:
            capacity, backing = _full.get((device, top_k), (0, None))
            if backing is None or rows > capacity:
                capacity = _capacity(rows)
                backing = torch.full(
                    (capacity,), top_k, dtype=torch.int32, device=device
                )
                _settle(device)
                # Earlier backings stay alive through the views that captured graphs hold.
                _full[(device, top_k)] = (capacity, backing)
            view = _full_views.setdefault((device, rows, top_k), backing[:rows])
    return view


class _RowBuffers:
    """Per row count: the unused second list, its zero lengths and the LSE scratch, as views of kept buffers."""

    def __init__(self, device, heads: int, name: str):
        self.device, self.heads, self.name = device, heads, name
        self.views: Dict[int, tuple] = {}
        self.capacity = 0
        self.backing: list = []

    def get(self, rows: int) -> tuple:
        found = self.views.get(rows)
        if found is None:
            _not_capturing(self.name, self.device)
            with _lock:
                if rows > self.capacity:
                    capacity, dev = _capacity(rows), self.device
                    self.backing.append(
                        (
                            torch.zeros((capacity, 1), device=dev, dtype=torch.int32),
                            torch.zeros((capacity,), device=dev, dtype=torch.int32),
                            torch.empty(
                                (capacity, 1, self.heads),
                                device=dev,
                                dtype=torch.float32,
                            ),
                        )
                    )
                    _settle(dev)
                    self.capacity = capacity
                no_list, no_lengths, scratch = self.backing[-1]
                found = self.views.setdefault(
                    rows,
                    (
                        _ffi(no_list[:rows]),
                        _ffi(no_lengths[:rows]),
                        _ffi(scratch[:rows].permute(2, 1, 0)),
                    ),
                )
        return found


def _stable(value) -> object:
    import dataclasses
    import enum

    if isinstance(value, enum.Enum):
        return (type(value).__qualname__, value.name)
    if value is None or type(value) in (bool, int, float, str):
        return value
    if isinstance(value, type):
        return (value.__module__, value.__qualname__)
    if dataclasses.is_dataclass(value):
        return (
            type(value).__qualname__,
            tuple(
                (f.name, _stable(getattr(value, f.name)))
                for f in dataclasses.fields(value)
                if f.compare
            ),
        )
    if isinstance(value, (tuple, list)):
        return tuple(_stable(v) for v in value)
    raise TypeError(f"no stable encoding for {type(value).__name__}")


def _ffi(tensor: torch.Tensor):
    """A plan-owned tensor converted once for the compiled call (it keeps the storage alive)."""
    try:
        import tvm_ffi

        return tvm_ffi.from_dlpack(tensor)
    except Exception:  # noqa: BLE001
        return tensor


def _fake_scale(is_tensor: bool):
    import cutlass
    import cutlass.cute as cute

    if is_tensor:
        return cute.runtime.make_fake_compact_tensor(
            cutlass.Float32, (1,), stride_order=(0,), assumed_align=4
        )
    return cutlass.Float32(1.0)


def build_kernel(device: torch.device, num_qo_heads: int, dtype: str, rope_dim: int):
    """The kernel object of one class on its pinned schedule (:data:`.spec.PROFILES`, :data:`.spec.KERNEL_STATE`)."""
    from .kernel.mla_decode.throughput_2cta.kernel import MlaDecodeTs

    profile = spec.PROFILES[dtype]
    sm_count = torch.cuda.get_device_properties(device).multi_processor_count
    kernel = MlaDecodeTs(
        page_size=1,
        max_active_clusters=sm_count // spec.CTAS_PER_ROW,
        is_persistent=True,
        is_var_seq=False,
        is_var_split_kv=False,
        static_split_kv=spec.SPLIT_KV,
        qkv_dtype=dtype,
        out_dtype="bf16",
        rope_dim=rope_dim,
        num_heads=num_qo_heads,
        seq_len_q=1,
        batch_size=1,
        mask_type="dense",
        softmax_rescale_threshold_log2=0.0,
        fixed_seqlen=True,
    )
    kernel.kv_cache_format = "native"
    kernel.extra_kv_cache_format = "native"
    kernel.sparse_gather_warps = profile["gather_issue_warps"]
    kernel.sparse_kv_stages = profile.get("kv_pipeline_stages", 0)
    kernel.sparse_uniform_pages = False
    kernel.sparse_balanced_registers = False
    kernel.lse_base_on_e = False
    kernel.runtime_kv_lengths = False
    kernel.use_kv_bounds = False
    kernel.zero_empty_rows = False
    kernel.output_quant = None
    kernel.assume_valid_prefix = False
    kernel.fuse_sparse_epilogue = True
    kernel.fuse_sparse_reduction = False
    kernel.direct_sparse = True
    kernel.direct_sparse_capacities = (1, 0)
    return kernel


def _compile(kernel, device, num_qo_heads, q_dtype, rope_dim, scale_is_tensor):
    import cutlass
    import cutlass.cute as cute

    from ...jit.cute_dsl_core import build_and_load_cute_dsl_kernel
    from .kernel.mla_decode.helpers.diagnostics import mla_build_diagnostics

    latent = spec.LATENT_DIM
    query_dim = latent + rope_dim
    rope_view = rope_dim or latent
    qkv = {torch.bfloat16: cutlass.BFloat16, torch.float8_e4m3fn: cutlass.Float8E4M3FN}[
        q_dtype
    ]
    pages = cute.sym_int()
    batch = cute.sym_int()
    q_stride = (query_dim, 1, num_qo_heads * query_dim, num_qo_heads * query_dim)
    q_latent = cute.runtime.make_fake_tensor(
        qkv, (num_qo_heads, latent, 1, batch), stride=q_stride, assumed_align=16
    )
    q_rope = cute.runtime.make_fake_tensor(
        qkv, (num_qo_heads, rope_view, 1, batch), stride=q_stride, assumed_align=16
    )
    c_latent = cute.runtime.make_fake_tensor(
        qkv, (1, latent, pages), stride=(query_dim, 1, query_dim), assumed_align=16
    )
    c_extra = cute.runtime.make_fake_tensor(
        qkv,
        (1, query_dim, cute.sym_int()),
        stride=(query_dim, 1, query_dim),
        assumed_align=16,
    )
    page_offsets = cute.runtime.make_fake_tensor(
        cutlass.Int32,
        (cute.sym_int(), batch),
        stride=(1, cute.sym_int64(divisibility=1)),
        assumed_align=4,
    )
    out_row = num_qo_heads * latent
    out = cute.runtime.make_fake_tensor(
        cutlass.BFloat16,
        (num_qo_heads, latent, 1, batch),
        stride=(latent, 1, out_row, out_row),
        assumed_align=16,
    )
    lse = cute.runtime.make_fake_tensor(
        cutlass.Float32,
        (num_qo_heads, 1, batch),
        stride=(1, num_qo_heads, num_qo_heads),
        assumed_align=4,
    )
    lengths = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32, (batch,), stride_order=(0,), assumed_align=4
    )

    def indices():
        return cute.runtime.make_fake_tensor(
            cutlass.Int32,
            (batch, cute.sym_int()),
            stride=(cute.sym_int64(), 1),
            assumed_align=4,
        )

    sinks = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32, (num_qo_heads,), stride_order=(0,), assumed_align=4
    )
    sparse_inputs = (indices(), indices(), lengths, lengths, sinks, _fake_scale(False))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    def compile_kernel():
        with mla_build_diagnostics():
            return cute.compile[cute.FrontendNext](
                kernel,
                q_latent,
                q_rope,
                c_latent,
                c_extra,
                page_offsets,
                out,
                lse,
                None,
                cutlass.Int32(spec.SPLIT_KV),
                lengths,
                None,
                sparse_inputs,
                _fake_scale(scale_is_tensor[0]),
                _fake_scale(scale_is_tensor[1]),
                stream,
                None,
                cutlass.Int32(0),
                None,
                options=spec.COMPILE_OPTIONS,
            )

    identity = (
        _stable(kernel.compile_signature()),
        num_qo_heads,
        _DTYPE_KEYS[q_dtype],
        rope_dim,
        tuple(scale_is_tensor),
        spec.COMPILE_OPTIONS,
    )
    key = hashlib.sha256(repr(identity).encode()).hexdigest()
    root = spec._kernel_dir()
    with torch.cuda.device(device):
        major, minor = torch.cuda.get_device_capability()
        return build_and_load_cute_dsl_kernel(
            _JIT_MODULE,
            f"sm{major}{minor}_{key}",
            compile_kernel,
            extra_key_files=[os.path.join(root, f) for f in spec.term_source_files()]
            + [os.path.abspath(__file__), os.path.abspath(spec.__file__)],
        )


class SparseMLAPlan:
    """One compiled shape of a sparse MLA class; :meth:`run` launches it for any row count and list width."""

    def __init__(
        self, device, num_qo_heads, q_dtype, rope_dim, softmax_scale, output_scale
    ):
        self.device = device
        self.num_qo_heads = num_qo_heads
        self.q_dtype = q_dtype
        self.rope_dim = rope_dim
        self.query_dim = spec.LATENT_DIM + rope_dim
        self._element_size = 2 if q_dtype == torch.bfloat16 else 1
        self._device_index = device.index
        self.class_name = spec.class_name(q_dtype, self.query_dim)
        self.softmax_scale = softmax_scale
        self.output_scale = output_scale
        self.scale_is_tensor = (
            isinstance(softmax_scale, torch.Tensor),
            isinstance(output_scale, torch.Tensor),
        )
        self._tensor_scales = self.scale_is_tensor != (False, False)
        self.kernel = build_kernel(device, num_qo_heads, _DTYPE_KEYS[q_dtype], rope_dim)
        self._compiled = _compile(
            self.kernel, device, num_qo_heads, q_dtype, rope_dim, self.scale_is_tensor
        )
        sinks = torch.full(
            (num_qo_heads,), -torch.inf, device=device, dtype=torch.float32
        )
        extra = torch.zeros(
            (1, 1, self.query_dim), device=device, dtype=q_dtype
        ).permute(1, 2, 0)
        _settle(device)
        self._sinks, self._extra = _ffi(sinks), _ffi(extra)
        self._rows = _RowBuffers(device, num_qo_heads, self.class_name)

    def check(
        self, query, kv_cache, indices, lengths, out, lse, contents: bool, scales=()
    ):
        """Layout checks of one launch (element counts, trailing dimensions, dtype, device, contiguity, alignment, and
        that ``out`` and ``lse`` share no memory with an input, a tensor scale in ``scales`` or each other);
        ``contents`` also reads the list on the host (bounds, and the packed-list contract)."""
        name, dev, heads = self.class_name, self.device, self.num_qo_heads
        index = self._device_index
        dim = self.query_dim
        if (
            lengths.ndim != 1
            or lengths.dtype != torch.int32
            or lengths.get_device() != index
            or not lengths.is_contiguous()
        ):
            raise unsupported(
                name,
                f"lengths of shape {tuple(lengths.shape)}, dtype {lengths.dtype}",
                f"contiguous int32 [rows] on {dev}",
            )
        rows = lengths.shape[0]
        if rows > _MAX_ROWS.get(self.q_dtype, rows):
            raise unsupported(
                name,
                f"{rows} query rows in one call",
                f"at most {_MAX_ROWS[self.q_dtype]} rows per call (split the call; rows do not depend on the split)",
            )
        shape = query.shape
        if (
            len(shape) < 2
            or shape[-1] != dim
            or shape[-2] != heads
            or query.numel() != rows * heads * dim
            or query.dtype != self.q_dtype
            or query.get_device() != index
            or not query.is_contiguous()
            or (q0 := query.data_ptr()) % 16
        ):
            raise unsupported(
                name,
                f"a query of shape {tuple(shape)}, dtype {query.dtype}",
                f"a contiguous 16-byte-aligned {self.q_dtype} [{rows}, {heads}, {dim}] on {dev}",
            )
        if (
            kv_cache.ndim < 2
            or kv_cache.shape[-1] != dim
            or not 0 < (kv_elements := kv_cache.numel()) < 0x7FFFFFFF * dim
            or kv_cache.dtype != self.q_dtype
            or kv_cache.get_device() != index
            or not kv_cache.is_contiguous()
            or (k0 := kv_cache.data_ptr()) % 16
        ):
            raise unsupported(
                name,
                f"a KV cache of shape {tuple(kv_cache.shape)}, dtype {kv_cache.dtype}",
                f"contiguous 16-byte-aligned {self.q_dtype} rows [storage_rows, {dim}] on {dev}",
            )
        shape = indices.shape
        if (
            len(shape) != 2
            or shape[0] != rows
            or shape[1] <= 0
            or indices.dtype != torch.int32
            or indices.get_device() != index
            or indices.stride(1) != 1
        ):
            raise unsupported(
                name,
                f"indices of shape {tuple(shape)}, dtype {indices.dtype}",
                f"int32 [{rows}, top_k] on {dev} with a contiguous last dimension",
            )
        if (
            out.ndim < 1
            or out.shape[-1] != spec.LATENT_DIM
            or out.numel() != rows * heads * spec.LATENT_DIM
            or out.dtype != torch.bfloat16
            or out.get_device() != index
            or not out.is_contiguous()
            or (o0 := out.data_ptr()) % 16
        ):
            raise unsupported(
                name,
                f"out of shape {tuple(out.shape)}, dtype {out.dtype}",
                f"a contiguous 16-byte-aligned bf16 [{rows}, {heads}, {spec.LATENT_DIM}] on {dev}",
            )
        if lse is not None and (
            lse.ndim < 1
            or lse.shape[-1] != heads
            or lse.numel() != rows * heads
            or lse.dtype != torch.float32
            or lse.get_device() != index
            or not lse.is_contiguous()
            or (s0 := lse.data_ptr()) % 4
        ):
            raise unsupported(
                name,
                f"lse of shape {tuple(lse.shape)}, dtype {lse.dtype}",
                f"a contiguous fp32 [{rows}, {heads}] on {dev}",
            )
        if rows:
            size = self._element_size
            q1 = q0 + rows * heads * dim * size
            k1 = k0 + kv_elements * size
            o1 = o0 + rows * heads * (2 * spec.LATENT_DIM)
            l0 = lengths.data_ptr()
            l1 = l0 + rows * 4
            i0 = indices.data_ptr()
            step, width = indices.stride(0) * 4, shape[1] * 4
            i1 = i0 + (rows - 1) * step + width
            shared = (
                (q0 < o1 and o0 < q1)
                or (k0 < o1 and o0 < k1)
                or (l0 < o1 and o0 < l1)
                or _rows_share(i0, i1, step, width, rows, o0, o1)
            )
            if lse is not None:
                s1 = s0 + rows * heads * 4
                shared = (
                    shared
                    or (q0 < s1 and s0 < q1)
                    or (k0 < s1 and s0 < k1)
                    or (l0 < s1 and s0 < l1)
                    or _rows_share(i0, i1, step, width, rows, s0, s1)
                    or (o0 < s1 and s0 < o1)
                )
            for scale in scales:
                if isinstance(scale, torch.Tensor):
                    a = scale.data_ptr()
                    shared = shared or (a < o1 and o0 < a + 4)
                    if lse is not None:
                        shared = shared or (a < s1 and s0 < a + 4)
            if shared:
                raise unsupported(
                    name,
                    "out or lse sharing memory with an input or with each other",
                    "out and lse in memory no other argument of the call occupies",
                )
        if contents and rows:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    f"{name}: list validation reads the list on the host and cannot run during CUDA graph capture "
                    "(validate=False, or FLASHINFER_VALIDATE_INPUTS unset)"
                )
            width = indices.shape[1]
            if ((lengths < 0) | (lengths > width)).any().item():
                raise ValueError(f"{name}: a list length is outside [0, {width}]")
            live = torch.arange(width, device=dev)[None, :] < lengths[:, None]
            storage_rows = kv_cache.numel() // dim
            if (live & ((indices < -1) | (indices >= storage_rows))).any().item():
                raise ValueError(f"{name}: a live index is outside the cache")
            empty = live & (indices < 0)
            if (empty.cummax(dim=1).values & live & ~empty).any().item():
                raise unsupported(
                    name,
                    "a -1 hole inside the live part of the list",
                    "live entries packed before any -1 (a hole moves every later key)",
                )

    def launch(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        out: torch.Tensor,
        lse: Optional[torch.Tensor] = None,
        softmax_scale: Optional[Scale] = None,
        output_scale: Optional[Scale] = None,
    ) -> None:
        """Launch on trusted arguments: ``query`` ``[rows, heads, 576|512]``, ``kv_cache`` ``[storage_rows, 576|512]``,
        ``indices`` int32 ``[rows, top_k]``, ``lengths`` int32 ``[rows]``, ``out`` bf16 ``[rows, heads, 512]``, ``lse``
        fp32 ``[rows, heads]`` or ``None``. ``query``, ``kv_cache``, ``out`` and ``lse`` may carry other leading
        dimensions as long as they are contiguous with these element counts. Nothing is checked and nothing is read
        on the host; :meth:`check` holds the layout rules. A scale override keeps the plan's kind (host float or
        device tensor)."""
        rows = lengths.shape[0]
        if rows == 0:
            return
        latent = spec.LATENT_DIM
        heads = self.num_qo_heads
        dim = self.query_dim
        no_list, no_lengths, scratch = self._rows.views.get(rows) or self._rows.get(
            rows
        )
        row = heads * dim
        start = query.storage_offset()
        q_latent = query.as_strided((heads, latent, 1, rows), (dim, 1, row, row), start)
        row = heads * latent
        self._compiled(
            q_latent,
            query.as_strided(
                (heads, self.rope_dim, 1, rows),
                (dim, 1, heads * dim, heads * dim),
                start + latent,
            )
            if self.rope_dim
            else q_latent,
            kv_cache.as_strided(
                (1, latent, kv_cache.numel() // dim),
                (dim, 1, dim),
                kv_cache.storage_offset(),
            ),
            self._extra,
            indices.transpose(0, 1),
            out.as_strided(
                (heads, latent, 1, rows), (latent, 1, row, row), out.storage_offset()
            ),
            scratch
            if lse is None
            else lse.as_strided(
                (heads, 1, rows), (1, heads, heads), lse.storage_offset()
            ),
            None,
            spec.SPLIT_KV,
            lengths,
            None,
            (indices, no_list, lengths, no_lengths, self._sinks, 1.0),
            self.softmax_scale if softmax_scale is None else softmax_scale,
            self.output_scale if output_scale is None else output_scale,
            None,
            1,
            None,
        )

    @flashinfer_api
    def run(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        indices: torch.Tensor,
        lengths: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        softmax_scale: Optional[Scale] = None,
        output_scale: Optional[Scale] = None,
        validate: bool = True,
    ):
        """Checked launch of the plan for any row count and list width.

        Parameters
        ----------
        query : torch.Tensor
            ``[rows, heads, 576 | 512]`` in the plan's dtype (bf16 or e4m3), contiguous and 16-byte aligned. Other
            leading dimensions are accepted when the element count is the same.
        kv_cache : torch.Tensor
            ``[storage_rows, 576 | 512]`` cache rows in the plan's dtype, contiguous and 16-byte aligned.
        indices : torch.Tensor
            int32 ``[rows, top_k]``: each row's selected cache rows, live entries first, ``-1`` after them.
        lengths : torch.Tensor
            int32 ``[rows]``: each row's live count, ``0 <= lengths <= top_k``. The bf16 classes take at most
            65,535 rows per call.
        out : Optional[torch.Tensor]
            bf16 ``[rows, heads, 512]`` to write; allocated when ``None``. Must not share memory with an input.
        lse : Optional[torch.Tensor]
            fp32 ``[rows, heads]`` to write the natural-log LSE; must not share memory with an input or ``out``.
        return_lse : bool
            Return ``(out, lse)``; ``lse`` is allocated when not given.
        softmax_scale, output_scale : Optional[Union[float, torch.Tensor]]
            Scales of this launch, of the kind the plan was made with (host float, or a one-element fp32 device
            tensor whose contents are not read on the host). ``None`` uses the plan's own.
        validate : bool
            Also read the list on the host: lengths in range, live indices inside the cache, no live entry after a
            ``-1``. This synchronizes and cannot run during CUDA graph capture. With ``validate=False`` those three
            rules are the caller's contract; see the README for what a row computes when they are broken.

        Returns
        -------
        torch.Tensor or Tuple[torch.Tensor, torch.Tensor]
            ``out``, or ``(out, lse)`` with ``return_lse``.

        Raises
        ------
        BatchInvariantUnsupportedError
            A layout, dtype, device, alignment, overlap or row count outside the class, or a ``-1`` hole.
        ValueError
            Under ``validate``: a length or a live index out of range.
        RuntimeError
            A row count's first call, or ``validate=True``, during CUDA graph capture.
        """
        rows = lengths.shape[0] if lengths.ndim == 1 else 0
        if out is None:
            out = torch.empty(
                (rows, self.num_qo_heads, spec.LATENT_DIM),
                device=self.device,
                dtype=torch.bfloat16,
            )
        if lse is None and return_lse:
            lse = torch.empty(
                (rows, self.num_qo_heads), device=self.device, dtype=torch.float32
            )
        softmax_scale = self.scale(softmax_scale, 0, "softmax_scale")
        output_scale = self.scale(output_scale, 1, "output_scale")
        self.check(
            query,
            kv_cache,
            indices,
            lengths,
            out,
            lse,
            validate,
            (
                self.softmax_scale if softmax_scale is None else softmax_scale,
                self.output_scale if output_scale is None else output_scale,
            )
            if self._tensor_scales
            else (),
        )
        self.launch(
            query, kv_cache, indices, lengths, out, lse, softmax_scale, output_scale
        )
        return (out, lse) if return_lse else out

    def scale(self, value, which: int, name: str):
        """A launch-time scale: ``None`` (the plan's own), or a checked value of the plan's kind. A host float equal to
        the plan's is passed through unchecked."""
        if value is None:
            return None
        planned = self.output_scale if which else self.softmax_scale
        if value is planned or (
            type(value) is float and type(planned) is float and value == planned
        ):
            return None
        return check_scale(
            value, name, self.device, self.class_name, self.scale_is_tensor[which]
        )


def check_scale(
    value, name: str, device, class_name: str, want_tensor: Optional[bool] = None
):
    """A positive host float, or a contiguous one-element fp32 tensor on ``device`` (its contents are not read)."""
    is_tensor = isinstance(value, torch.Tensor)
    if want_tensor is not None and is_tensor != want_tensor:
        raise unsupported(
            class_name,
            f"{name} as a {'tensor' if is_tensor else 'host float'}",
            "the kind the plan was made with (host float or device tensor)",
        )
    if is_tensor:
        if (
            value.device != device
            or value.dtype != torch.float32
            or value.numel() != 1
            or value.ndim > 1
            or not value.is_contiguous()
        ):
            raise unsupported(
                class_name,
                f"{name} of shape {tuple(value.shape)}, dtype {value.dtype}",
                f"a contiguous fp32 tensor of one element on {device}",
            )
        return value.view(1) if value.ndim == 0 else value
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise unsupported(class_name, f"{name}={value!r}", "a positive scale")
    try:
        value = float(value)
    except (OverflowError, TypeError, ValueError):
        value = float("inf")
    if not _MIN_SCALE <= value <= _MAX_SCALE:
        raise unsupported(
            class_name,
            f"{name}={value!r}",
            f"a positive scale of at most {_MAX_SCALE:g} (the kernel multiplies it by log2(e) in fp32)",
        )
    return value


@flashinfer_api
def plan(
    device,
    num_qo_heads: int,
    *,
    q_dtype: torch.dtype,
    qk_rope_head_dim: int,
    softmax_scale: Scale,
    output_scale: Scale = 1.0,
) -> SparseMLAPlan:
    """The plan of one compiled shape, built once per process and kept.

    Parameters
    ----------
    device : torch.device or str
        A CUDA device of architecture SM100, SM103 or SM107.
    num_qo_heads : int
        Query heads, 1 to 128.
    q_dtype : torch.dtype
        ``torch.float8_e4m3fn`` or ``torch.bfloat16``: the dtype of the query and of the cache rows.
    qk_rope_head_dim : int
        64 for 576-wide rows (512 latent + 64 RoPE), 0 for 512-wide rows.
    softmax_scale : Union[float, torch.Tensor]
        Required: the scale is an input of the row term. A positive host float of at most 2.35e38, or a one-element
        fp32 tensor on ``device`` (the same range is the caller's contract there: its contents are not read). It is the default of the plan's launches; only its kind is part of the compiled shape.
    output_scale : Union[float, torch.Tensor]
        Same rules; default ``1.0``.

    Returns
    -------
    SparseMLAPlan
        The plan; :meth:`SparseMLAPlan.run` checks every argument, :meth:`SparseMLAPlan.launch` checks nothing.

    Raises
    ------
    BatchInvariantUnsupportedError
        A device, dtype, width, head count or scale outside the classes.
    RuntimeError
        The first plan of a shape requested during CUDA graph capture.
    """
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    try:
        name = spec.class_name(q_dtype, spec.LATENT_DIM + qk_rope_head_dim)
    except (ValueError, TypeError):
        name = spec.FAMILY
    cc = check_device(device, name, spec.MLA_ARCHS)
    if q_dtype not in _DTYPE_KEYS:
        raise unsupported(name, f"q_dtype {q_dtype}", "bf16 or e4m3 Q / KV")
    if isinstance(qk_rope_head_dim, bool) or qk_rope_head_dim not in (0, spec.ROPE_DIM):
        raise unsupported(
            name,
            f"qk_rope_head_dim={qk_rope_head_dim!r}",
            "0 (512-wide) or 64 (576-wide)",
        )
    if (
        isinstance(num_qo_heads, bool)
        or not isinstance(num_qo_heads, int)
        or not 1 <= num_qo_heads <= spec.MAX_HEADS
    ):
        raise unsupported(name, f"{num_qo_heads!r} query heads", "1 to 128 heads")
    if softmax_scale is None:
        raise unsupported(
            name,
            "softmax_scale=None",
            "an explicit softmax_scale (the scale is an input of the row term)",
        )
    softmax_scale = check_scale(softmax_scale, "softmax_scale", device, name)
    output_scale = check_scale(output_scale, "output_scale", device, name)
    key = (
        device,
        num_qo_heads,
        q_dtype,
        qk_rope_head_dim,
        isinstance(softmax_scale, torch.Tensor),
        isinstance(output_scale, torch.Tensor),
    )
    found = _plans.get(key)
    if found is not None:
        if _same_scale(softmax_scale, found.softmax_scale) and _same_scale(
            output_scale, found.output_scale
        ):
            return found
        return _twin(found, softmax_scale, output_scale)
    _not_capturing(name, device)
    if cc == (10, 7):
        from ...cute_dsl.utils import require_cute_dsl_arch

        require_cute_dsl_arch(device.index)
    with _lock:
        building = _build_locks.setdefault(key, threading.Lock())
    with building:
        made = _plans.get(key)
        if made is None:
            made = SparseMLAPlan(
                device,
                num_qo_heads,
                q_dtype,
                qk_rope_head_dim,
                softmax_scale,
                output_scale,
            )
            with _lock:
                _plans[key] = made
    if _same_scale(softmax_scale, made.softmax_scale) and _same_scale(
        output_scale, made.output_scale
    ):
        return made
    return _twin(made, softmax_scale, output_scale)


def _twin(found, softmax_scale, output_scale):
    twin = SparseMLAPlan.__new__(SparseMLAPlan)
    twin.__dict__.update(found.__dict__)
    twin.softmax_scale, twin.output_scale = softmax_scale, output_scale
    return twin


def _same_scale(a, b) -> bool:
    return a is b or (not isinstance(a, torch.Tensor) and a == b)
