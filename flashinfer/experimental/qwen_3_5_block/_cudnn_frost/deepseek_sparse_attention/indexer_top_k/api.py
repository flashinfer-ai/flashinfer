# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""APIBase wrapper for the DSA indexer top-K CuTe DSL kernel."""

from __future__ import annotations

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import torch
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import contiguous_on_stream, device_context, launch_stream
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.api_base import APIBase, TupleDict, WorkspaceCarver, ws_align
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import device_capability
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import (
    torch_stream_context as _torch_stream_context,
)

from .compactify import compactify as _compactify
from .local_to_global_dsl import local_to_global as _local_to_global

_SUPPORTED_DTYPES = (torch.float32, torch.float16, torch.bfloat16)
# These limits keep the deterministic radix scratch, maximum selected-index
# array, bitmap, and synchronization state below SM120's 99 KiB CTA limit.
_SM120_DETERMINISTIC_MAX_NUM_COLS = {
    torch.float32: 16384,
    torch.float16: 32768,
    torch.bfloat16: 32768,
}


def _kernel_module():
    """Import the CuTe DSL implementation only when Top-K is used."""
    from . import indexer_top_k_decode_varlen

    return indexer_top_k_decode_varlen


def _check_execute_tensor(tensor, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> None:
    """Validate one strict execute-time tensor binding without staging it."""
    if tensor is None:
        raise ValueError(f"{name} is required")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} tensor shape mismatch: expected {tuple(shape)}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} dtype mismatch: expected {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous, got strides {tuple(tensor.stride())}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _tensor_byte_span(tensor: Optional[torch.Tensor]) -> Optional[Tuple[int, int]]:
    """Return the exact byte range touched by a validated contiguous tensor."""
    if tensor is None or tensor.numel() == 0:
        return None
    start = tensor.data_ptr()
    return start, start + tensor.numel() * tensor.element_size()


def _check_no_write_overlap(
    read_only: Tuple[Tuple[str, torch.Tensor], ...],
    writable: Tuple[Tuple[str, Optional[torch.Tensor]], ...],
) -> None:
    """Reject overlapping writable ranges while allowing disjoint storage views."""
    prior = list(read_only)
    for write_name, write_tensor in writable:
        write_span = _tensor_byte_span(write_tensor)
        if write_span is None:
            continue
        for other_name, other_tensor in prior:
            other_span = _tensor_byte_span(other_tensor)
            if other_span is not None and max(write_span[0], other_span[0]) < min(write_span[1], other_span[1]):
                raise ValueError(f"{write_name} overlaps {other_name}; writable outputs and scratch must have disjoint byte ranges")
        prior.append((write_name, write_tensor))


def _aligned_contiguous_on_stream(tensor: torch.Tensor, stream, alignment: int) -> torch.Tensor:
    """Stage layout first, then repair a contiguous view with a misaligned base."""
    tensor = contiguous_on_stream(tensor, stream, tensor.device)
    if tensor.data_ptr() % alignment == 0:
        return tensor
    tensor.record_stream(launch_stream(stream, tensor.device))
    with _torch_stream_context(stream, tensor.device):
        return tensor.clone(memory_format=torch.contiguous_format)


class IndexerTopK(APIBase):
    """Top-K filter using the SM90+ CuTe-DSL radix kernel.

    Selects the ``top_k`` largest entries from each row of ``input_values``.

    Supported domain
    ----------------
    ``input_values`` must be a contiguous, 32-byte-aligned CUDA tensor in FP32,
    FP16, or BF16;
    ``seq_lens`` must be contiguous CUDA INT32 on the same device. The kernel
    requires SM90+ and ``0 < top_k <= 2048``. The strict class API rejects
    strided tensors; :func:`indexer_top_k_wrapper` stages them contiguously.

    Parameter conventions (important — easy to misuse)
    --------------------------------------------------
    ``input_values`` has shape ``(n_rows, num_cols)``. The kernel treats the
    rows as belonging to ``batch_size = seq_lens.numel()`` groups, with
    ``n_rows == batch_size * next_n`` exactly. Within each group of
    ``next_n`` rows the kernel applies a **speculative-decoding stagger**:
    the effective valid length for row ``task_id`` inside batch ``b`` is
    ``seq_lens[b] - next_n + (task_id % next_n) + 1`` (i.e. the first row in
    a group sees the shortest prefix, the last row sees the full
    ``seq_lens[b]``).

    Use cases:

    * **Independent per-row top-K over equal-length rows** (the common
      case): set ``next_n = 1``, ``batch_size = n_rows``, ``seq_lens`` a
      length-``n_rows`` tensor. Each row then sees its own ``seq_lens[i]``
      columns.
    * **Speculative decoding / medusa-style drafts**: set ``next_n`` to the
      number of draft tokens per batch; ``seq_lens`` describes the cache
      length per batch. The kernel produces the staggered behaviour
      automatically.

    ``check_support()`` rejects zero extents and any shape that does not satisfy
    the exact row grouping invariant. Each ``seq_lens`` entry must satisfy
    ``0 <= seq_lens[i] <= num_cols``; this value contract is caller-owned and
    is not read back to the host. A value smaller than ``next_n`` can still
    make an early speculative row empty; empty rows are returned as ``-1``
    indices and ``-inf`` values.

    ``tie_break`` is 0 (arbitrary), 1 (smaller source-column index), or 2
    (larger source-column index) among equal radix keys. ``deterministic=True``
    makes cutoff selection repeatable (smaller indices win when ``tie_break=0``)
    and returns the valid selected local indices in ascending order. Padding
    for rows shorter than ``top_k`` remains ``-1``/``-inf``. Signed zeros
    remain distinct (+0 before -0); NaN ordering is unspecified.
    On SM120, deterministic FP32 supports at most 16,384 columns and
    deterministic FP16/BF16 at most 32,768 columns per row; ``check_support()``
    rejects larger shapes before compilation because of the 99 KiB per-block
    shared-memory limit. The default non-deterministic specialization is
    unchanged.

    Memory contract (Rule 8)
    ------------------------
    The class owns no device memory. ``execute()`` writes into the caller's
    ``out_indices`` (``(n_rows, top_k)`` int32) and, when ``return_val``,
    ``out_values`` (``(n_rows, top_k)`` in the input dtype), and carves the
    radix scratch from the caller's ``workspace`` of
    :meth:`scratch_workspace_bytes` bytes. All writable output and carved
    scratch byte ranges must be disjoint from the read-only inputs and from
    each other; disjoint views of one backing allocation are accepted.
    ``compile()`` builds the kernel from fake tensors; the module-level cache
    is keyed on the CUDA device, next-power-of-two of ``num_cols``, and the
    plan-time flags.
    """

    def __init__(
        self,
        sample_input_values: torch.Tensor,
        sample_seq_lens: torch.Tensor,
        top_k: int,
        next_n: int = 1,
        return_val: bool = True,
        num_copy_bits: int = 256,
        tie_break: int = 0,
        deterministic: bool = False,
    ):
        """Create a plan from sample tensor metadata and immutable options.

        The sample tensors provide shape, stride, dtype, and device metadata;
        the plan does not retain their storage. Call :meth:`check_support` and
        :meth:`compile` before graph capture, then bind matching caller-owned
        tensors and workspace through :meth:`execute`.
        """
        super().__init__()
        self.input_desc = self._make_tensor_desc(sample_input_values, name="input_values")
        self.seq_lens_desc = self._make_tensor_desc(sample_seq_lens, name="seq_lens")
        self.top_k = int(top_k)
        self.next_n = int(next_n)
        self.return_val = bool(return_val)
        self.num_copy_bits = int(num_copy_bits)
        self._buffer_dims = None
        self._scratch_bytes = None
        self.tie_break = tie_break
        self.deterministic = bool(deterministic)

    def check_support(self) -> bool:
        """Validate the plan-time tensor signature, shape domain, and device."""
        self._logger.debug("Entering check_support")
        self._check_dtype(self.input_desc, list(_SUPPORTED_DTYPES), name="input_values")
        self._check_dtype(self.seq_lens_desc, torch.int32, name="seq_lens")
        self._value_error_if(self.input_desc.device.type != "cuda", f"input_values must be a CUDA tensor, got {self.input_desc.device}")
        self._value_error_if(
            self.seq_lens_desc.device != self.input_desc.device,
            f"seq_lens must be on {self.input_desc.device}, got {self.seq_lens_desc.device}",
        )
        self._value_error_if(
            self.input_desc.ndim != 2,
            f"input_values must be 2-D (n_rows, num_cols), got {self.input_desc.shape}",
        )
        self._value_error_if(
            self.seq_lens_desc.ndim != 1,
            f"seq_lens must be 1-D, got {self.seq_lens_desc.shape}",
        )
        self._value_error_if(
            self.top_k <= 0 or self.top_k > 2048,
            f"top_k must be in (0, 2048], got {self.top_k}",
        )
        self._value_error_if(self.next_n <= 0, f"next_n must be positive, got {self.next_n}")

        self._value_error_if(self.tie_break not in (0, 1, 2), "tie_break must be 0 (none), 1 (small), or 2 (large)")

        # Enforce the kernel's n_rows == batch_size * next_n invariant
        # up-front so misuse surfaces here rather than as silently-empty
        # rows (the stagger formula produces non-positive lengths when
        # next_n exceeds n_rows per batch).
        n_rows, num_cols = self.input_desc.shape
        batch_size = self.seq_lens_desc.shape[0]
        self._value_error_if(n_rows <= 0, f"n_rows must be positive, got {n_rows}")
        self._value_error_if(num_cols <= 0, f"num_cols must be positive, got {num_cols}")
        self._value_error_if(batch_size <= 0, f"seq_lens must contain at least one batch, got {batch_size}")
        self._value_error_if(
            n_rows != batch_size * self.next_n,
            f"n_rows ({n_rows}) must equal seq_lens.numel() * next_n "
            f"({batch_size} * {self.next_n} = {batch_size * self.next_n}). "
            f"For independent top-K over equal-length rows use "
            f"next_n=1 and seq_lens of shape (n_rows,).",
        )

        self._check_tensor_stride(self.input_desc, stride=(self.input_desc.shape[1], 1), name="input_values")
        self._check_tensor_stride(self.seq_lens_desc, stride=(1,), name="seq_lens")

        major, minor = device_capability(self.input_desc.device)
        self._runtime_error_if(
            major < 9,
            f"IndexerTopK requires SM90+ compute capability, found SM{major}",
        )
        if self.deterministic and (major, minor) == (12, 0):
            max_num_cols = _SM120_DETERMINISTIC_MAX_NUM_COLS[self.input_desc.dtype]
            self._runtime_error_if(
                num_cols > max_num_cols,
                f"deterministic IndexerTopK on SM120 supports at most {max_num_cols} columns " f"for {self.input_desc.dtype}, got {num_cols}",
            )
        self._is_supported = True
        return True

    def _buffer_shape(self) -> Tuple[int, int, int]:
        if self._buffer_dims is None:
            n_rows, num_cols = self.input_desc.shape
            self._buffer_dims = (n_rows, _kernel_module().buffer_numbers(self.input_desc.dtype), num_cols)
        return self._buffer_dims

    def scratch_workspace_bytes(self) -> int:
        """Bytes of ``workspace`` ``execute()`` carves its int32 radix scratch from (R2)."""
        if self._scratch_bytes is None:
            n_rows, buffer_numbers, num_cols = self._buffer_shape()
            self._scratch_bytes = ws_align(n_rows * buffer_numbers * num_cols * torch.int32.itemsize)
        return self._scratch_bytes

    def compile(self) -> None:
        """Compile or reuse the specialization for this plan's CUDA device."""
        self._logger.debug("Entering compile")
        self._ensure_support_checked()
        n_rows, num_cols = self.input_desc.shape
        # Fake operands only (R11); the kernel is compiled for the current device.
        with torch.cuda.device(self.input_desc.device):
            self._compiled_kernel = _kernel_module().compile_topk_kernel(
                self.input_desc.dtype,
                n_rows,
                num_cols,
                self.top_k,
                self.next_n,
                return_val=self.return_val,
                num_copy_bits=self.num_copy_bits,
                tie_break=self.tie_break,
                deterministic=self.deterministic,
            )

    def execute(
        self,
        input_values: torch.Tensor,
        seq_lens: torch.Tensor,
        current_stream: Optional[cuda.CUstream] = None,
        *,
        out_indices: torch.Tensor,
        out_values: Optional[torch.Tensor] = None,
        workspace: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Launch into the caller's ``out_indices`` / ``out_values``; returns them.

        ``out_values`` is required when the object was built with ``return_val=True``
        and must be ``None`` otherwise. ``workspace`` is a CUDA uint8 tensor of at
        least :meth:`scratch_workspace_bytes` bytes on ``input_values``' device.
        After compilation/warmup, calls with stable tensor addresses are safe to
        capture and replay in a CUDA Graph; replay reads current input and length
        contents and writes the caller-owned outputs.
        """
        self._logger.debug("Entering execute")
        self._ensure_support_checked()

        device = self.input_desc.device
        n_rows, buffer_numbers, num_cols = self._buffer_shape()
        _check_execute_tensor(input_values, "input_values", self.input_desc.shape, self.input_desc.dtype, device)
        self._value_error_if(
            input_values.data_ptr() % _kernel_module().INPUT_ALIGN != 0,
            f"input_values must be {_kernel_module().INPUT_ALIGN}-byte aligned; got data_ptr=0x{input_values.data_ptr():x}",
        )
        _check_execute_tensor(seq_lens, "seq_lens", self.seq_lens_desc.shape, torch.int32, device)
        _check_execute_tensor(out_indices, "out_indices", (n_rows, self.top_k), torch.int32, device)
        if self.return_val:
            self._value_error_if(out_values is None, "out_values is required: this IndexerTopK was built with return_val=True")
            _check_execute_tensor(out_values, "out_values", (n_rows, self.top_k), self.input_desc.dtype, device)
        else:
            self._value_error_if(out_values is not None, "out_values must be None: this IndexerTopK was built with return_val=False")

        kernel_module = _kernel_module()
        carver = WorkspaceCarver(workspace, self.scratch_workspace_bytes(), "IndexerTopK")
        if workspace.device != device:
            raise ValueError(f"IndexerTopK workspace must be on input_values' device {device}, got {workspace.device}")
        buffer = carver.take(n_rows * buffer_numbers * num_cols, torch.int32).view(n_rows, buffer_numbers, num_cols)
        self._value_error_if(
            buffer.data_ptr() % kernel_module.BUFFER_ALIGN != 0,
            f"IndexerTopK workspace must be {kernel_module.BUFFER_ALIGN}-byte aligned; got data_ptr=0x{buffer.data_ptr():x}",
        )
        _check_no_write_overlap(
            (("input_values", input_values), ("seq_lens", seq_lens)),
            (("out_indices", out_indices), ("out_values", out_values), ("workspace scratch", buffer)),
        )

        if self._compiled_kernel is None:
            self.compile()

        # The compiled kernel launches on the TVM-FFI environment stream, i.e.
        # torch's current stream: enter the caller's stream so it is honored.
        with _torch_stream_context(current_stream, device):
            kernel_module.launch_topk_kernel(self._compiled_kernel, input_values, seq_lens, out_indices, out_values, buffer, self.next_n)

        # TVM-FFI launches are invisible to the caching allocator: record every
        # tensor the kernel touches on the launch stream (R1).
        consumer = launch_stream(current_stream, device)
        for tensor in (input_values, seq_lens, out_indices, out_values, workspace):
            if tensor is not None:
                tensor.record_stream(consumer)
        return out_indices, out_values


_cache_of_IndexerTopKObjects: dict = {}


def indexer_top_k_wrapper(
    input_values: torch.Tensor,
    seq_lens: torch.Tensor,
    top_k: int,
    next_n: int = 1,
    return_val: bool = True,
    num_copy_bits: int = 256,
    stream: Optional[cuda.CUstream] = None,
    tie_break: int = 0,
    deterministic: bool = False,
) -> TupleDict:
    """High-level wrapper returning ``{'indices', 'values'}``.

    ``input_values`` is ``(n_rows, num_cols)``; the kernel requires
    ``n_rows == seq_lens.numel() * next_n`` and applies a
    speculative-decoding length stagger within each ``next_n``-row group
    (first row sees the shortest prefix, last sees the full ``seq_lens``).
    For "independent top-K over every row" set ``next_n=1`` and make
    ``seq_lens`` a length-``n_rows`` tensor. See :class:`IndexerTopK` for
    full details.

    ``values`` is ``None`` when ``return_val=False``. The outputs and the
    :meth:`IndexerTopK.scratch_workspace_bytes` workspace are allocated here
    per call, on ``stream``; use the class API with caller-owned buffers to
    avoid the allocations. Strided ``input_values`` / ``seq_lens`` are copied
    contiguous here (the class declines them in ``check_support()``).

    ``tie_break`` chooses among equal radix keys at the cutoff: 0 (arbitrary,
    default), 1 (smaller source-column index), or 2 (larger source-column index).
    With ``deterministic=True``, mode 0 deterministically prefers smaller
    indices and the valid selected local indices are returned in ascending
    order; short-row padding remains ``-1``/``-inf``.
    Signed zeros retain their existing radix ordering (+0 before -0); NaN
    ordering is unspecified.

    """
    if not input_values.is_cuda:
        raise ValueError(f"input_values must be a CUDA tensor, got {input_values.device}")
    if not seq_lens.is_cuda:
        raise ValueError(f"seq_lens must be a CUDA tensor, got {seq_lens.device}")
    if input_values.device != seq_lens.device:
        raise ValueError(f"seq_lens must be on {input_values.device}, got {seq_lens.device}")

    # R1 staging: the originals are record_stream'ed on `stream` before they are rebound.
    input_values = _aligned_contiguous_on_stream(input_values, stream, _kernel_module().INPUT_ALIGN)
    seq_lens = contiguous_on_stream(seq_lens, stream, seq_lens.device)
    cache_key = (
        input_values.device,
        input_values.dtype,
        int(input_values.shape[0]),  # n_rows affects wrapper validation and output shape
        input_values.shape[-1],  # num_cols buckets internally
        int(seq_lens.shape[0]),  # batch_size / seq_lens.numel()
        int(top_k),
        int(next_n),
        bool(return_val),
        int(num_copy_bits),
        tie_break,
        bool(deterministic),
    )
    obj = _cache_of_IndexerTopKObjects.get(cache_key)
    if obj is None:
        obj = IndexerTopK(
            sample_input_values=input_values,
            sample_seq_lens=seq_lens,
            top_k=top_k,
            next_n=next_n,
            return_val=return_val,
            num_copy_bits=num_copy_bits,
            tie_break=tie_break,
            deterministic=deterministic,
        )
        assert obj.check_support()
        obj.compile()
        _cache_of_IndexerTopKObjects[cache_key] = obj

    n_rows = int(input_values.shape[0])
    device = input_values.device
    # One device/stream scope for the allocations and the launch; execute() re-enters it as a no-op.
    with device_context(device), _torch_stream_context(stream, device):
        indices = torch.empty(n_rows, top_k, dtype=torch.int32, device=device)
        values = torch.empty(n_rows, top_k, dtype=input_values.dtype, device=device) if return_val else None
        workspace = torch.empty(obj.scratch_workspace_bytes(), dtype=torch.uint8, device=device)
        obj.execute(input_values, seq_lens, out_indices=indices, out_values=values, current_stream=stream, workspace=workspace)
    return TupleDict(indices=indices, values=values)


def local_to_global_wrapper(
    local_indices: torch.Tensor,
    seqlen_k: int,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Convert local top-K indices to the global index space."""
    indices = _local_to_global(
        local_indices,
        seqlen_k,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        stream=stream,
    )
    return TupleDict(indices=indices)


def compactify_wrapper(
    indices: torch.Tensor,
    stream: Optional[cuda.CUstream] = None,
) -> TupleDict:
    """Pack valid indices row-wise and return ``indices`` plus ``topk_length``."""
    compact_indices, topk_length = _compactify(indices, stream=stream)
    return TupleDict(indices=compact_indices, topk_length=topk_length)
