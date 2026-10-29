# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Base classes for cuDNN API wrappers.

This module provides abstract base classes that define common interfaces
for cuDNN API wrapper classes, including validation, compilation, and execution patterns.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Tuple, Optional
import logging
import sys
import threading
import cuda.bindings.driver as cuda
import cutlass

import cutlass.cute as cute
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._experimental_warnings import warn_experimental_api_once

# ``TupleDict`` lives in the DSL-free ``cudnn._tupledict`` (the QSA front doors return it without importing this module, whose import
# executes the DSL) and is re-exported here, so every ``from cudnn.api_base import TupleDict`` keeps resolving.
from flashinfer.experimental.qwen_3_5_block._cudnn_frost import _tupledict as _tupledict_mod

TupleDict = _tupledict_mod.TupleDict  # the re-export: ``from cudnn.api_base import TupleDict`` keeps resolving
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.datatypes import _convert_to_cutlass_data_type, _convert_to_cutlass_data_type_or_none
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.tensor_adapter import (
    Device,
    canonicalize_unit_dim_strides,
    cuda_is_available,
    get_compute_capability,
    get_device,
    get_shape,
    get_strides,
)

# torch stays a lazy dependency: a torch tensor/dtype can only reach these APIs if torch
# is already imported, so probing sys.modules never imports torch on behalf of other
# frameworks (e.g. JAX).


def _torch():
    return sys.modules.get("torch")


_RAW_STREAM = None


def _raw_stream(torch):
    """``device_index -> raw CUstream int``, resolved once.

    ``torch._C._cuda_getCurrentRawStream`` is private, so fall back to the
    documented accessor if a torch build ever drops it.
    """
    global _RAW_STREAM
    if _RAW_STREAM is None:
        _RAW_STREAM = getattr(torch._C, "_cuda_getCurrentRawStream", None) or (lambda _dev: torch.cuda.current_stream().cuda_stream)
    return _RAW_STREAM


def _is_framework_tensor(obj: Any) -> bool:
    """True for framework tensors (torch/jax/numpy/...) as opposed to dtypes or shape/stride tuples."""
    return hasattr(obj, "__dlpack__")


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def is_power_of_2(n: int) -> bool:
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def tensor_byte_span(t) -> Tuple[int, int]:
    """The half-open byte range ``[lo, hi)`` the elements of a framework tensor occupy: its base address (``data_ptr()``, the storage
    offset included) plus the smallest and the largest element offset its shape and strides reach (a negative stride counted on the low
    side, a zero stride on neither), the last element's bytes included; an empty tensor spans nothing (``lo == hi``).  Host arithmetic on
    the metadata alone -- no device read, no synchronisation -- so it belongs on an execute path (Rule 1 / Rule 3)."""
    base = int(t.data_ptr())
    if t.numel() == 0:
        return base, base
    item = int(t.element_size())
    lo = hi = 0
    for size, stride in zip(t.shape, t.stride()):
        if size > 1:
            reach = (int(size) - 1) * int(stride)
            if reach < 0:
                lo += reach
            else:
                hi += reach
    return base + lo * item, base + (hi + 1) * item


def check_buffers_disjoint(writers, readers, *, what: str = "the kernels") -> None:
    """Rule 1's aliasing check of an execute path: every buffer in ``writers`` (``(name, tensor)`` pairs; a ``None`` tensor is skipped) must
    be disjoint, as a BYTE RANGE, from every buffer in ``readers`` and from the other writers.  Pointer equality is not that test: a caller's
    view that starts inside an input (two views over one allocation) has a distinct ``data_ptr()``, and the kernel overwrites rows it has
    still to read (measured on the triangle multiplicative update at N = 512: 16.7 M differing elements, max |diff| 2.56).  Inputs may
    overlap each other (read-only).  A typed ``ValueError`` naming the two operands and their ranges, before any launch."""
    spans = [(name, tensor_byte_span(t)) for name, t in writers if t is not None]
    read = [(name, tensor_byte_span(t)) for name, t in readers if t is not None]
    for i, (w_name, (w_lo, w_hi)) in enumerate(spans):
        for r_name, (r_lo, r_hi) in read + spans[i + 1 :]:
            if w_lo < r_hi and r_lo < w_hi:
                raise ValueError(
                    f"{w_name} must not alias {r_name}: their byte ranges intersect ({w_name} [{w_lo:#x}, {w_hi:#x}), {r_name} [{r_lo:#x}, {r_hi:#x})) "
                    f"and {what} write {w_name} while {r_name} is still read or written; pass buffers that do not overlap"
                )


def is_sm107_device() -> bool:
    """Return True when the current CUDA device is Rubin (SM107)."""
    return cuda_is_available() and get_compute_capability() == (10, 7)


def get_device_type() -> str:
    """Return the architecture family used by SM100 grouped GEMM wrappers."""
    return "rubin" if is_sm107_device() else "blackwell"


_experimental_api_warnings_emitted = set()
_experimental_api_warnings_lock = threading.Lock()


def warn_experimental_api_once(logger: logging.Logger, api_name: str) -> None:
    """Emit the experimental API warning once per API class per process."""
    with _experimental_api_warnings_lock:
        if api_name in _experimental_api_warnings_emitted:
            return
        _experimental_api_warnings_emitted.add(api_name)

    logger.warning("%s is an experimental API", api_name)


def _reset_experimental_api_warning_registry() -> None:
    """Reset experimental API warning state for tests."""
    with _experimental_api_warnings_lock:
        _experimental_api_warnings_emitted.clear()


@dataclass(frozen=True)
class TensorDesc:
    """Metadata needed to validate/compile tensor signatures without storage."""

    dtype: Any
    shape: Tuple[int, ...]
    stride: Tuple[int, ...]
    stride_order: Tuple[int, ...]
    device: Any
    interpret_uint8_as_fp4x2: bool = False
    ndim: int = field(init=False)
    name: str = ""

    def __post_init__(self):
        shape = tuple(self.shape)
        stride = tuple(self.stride)
        stride_order = tuple(self.stride_order)
        device = self.device
        torch = _torch()
        if torch is not None and not isinstance(device, (torch.device, Device)):
            try:
                device = torch.device(device)
            except (TypeError, ValueError, RuntimeError) as exc:
                raise TypeError(f"Invalid device for TensorDesc: {self.device!r}") from exc

        ndim = len(shape)
        if len(stride) != ndim:
            raise ValueError(f"Stride rank mismatch: expected {ndim}, got {len(stride)}")
        if len(stride_order) != ndim:
            raise ValueError(f"Stride order rank mismatch: expected {ndim}, got {len(stride_order)}")
        if tuple(sorted(stride_order)) != tuple(range(ndim)):
            raise ValueError(f"Stride order must be a permutation of [0, {ndim - 1}], got {stride_order}")

        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "stride", stride)
        object.__setattr__(self, "stride_order", stride_order)
        object.__setattr__(self, "device", device)
        object.__setattr__(self, "ndim", ndim)

    @staticmethod
    def _normalize_dim(dim: int, ndim: int, *, allow_new_dim: bool = False) -> int:
        min_dim = -ndim - (1 if allow_new_dim else 0)
        max_dim = ndim if allow_new_dim else ndim - 1
        if dim < min_dim or dim > max_dim:
            raise IndexError(f"Dimension out of range (expected to be in range of [{min_dim}, {max_dim}], but got {dim})")
        if dim < 0:
            dim += ndim + (1 if allow_new_dim else 0)
        return dim

    @staticmethod
    def _compute_stride_order(shape: Tuple[int, ...], stride: Tuple[int, ...]) -> Tuple[int, ...]:
        return tuple(i for i, _ in sorted(enumerate(stride), key=lambda x: (x[1], shape[x[0]])))

    @staticmethod
    def _numel(shape: Tuple[int, ...]) -> int:
        numel = 1
        for size in shape:
            numel *= size
        return numel

    @staticmethod
    def _compute_contiguous_stride(shape: Tuple[int, ...]) -> Tuple[int, ...]:
        if not shape:
            return ()
        strides = [0] * len(shape)
        running = 1
        for i in range(len(shape) - 1, -1, -1):
            strides[i] = running
            running *= max(shape[i], 1)
        return tuple(strides)

    @staticmethod
    def _is_contiguous_with_order(shape: Tuple[int, ...], stride: Tuple[int, ...], order: Tuple[int, ...]) -> bool:
        expected_stride = 1
        for dim in order:
            size = shape[dim]
            if size == 1:
                continue
            if stride[dim] != expected_stride:
                return False
            expected_stride *= size
        return True

    @staticmethod
    def _compute_view_stride(
        old_shape: Tuple[int, ...],
        old_stride: Tuple[int, ...],
        new_shape: Tuple[int, ...],
    ) -> Optional[Tuple[int, ...]]:
        old_numel = TensorDesc._numel(old_shape)
        if old_numel == 0:
            return TensorDesc._compute_contiguous_stride(new_shape)

        new_stride = [0] * len(new_shape)
        view_dim = len(new_shape) - 1
        tensor_numel = 1
        view_numel = 1

        for tensor_dim in range(len(old_shape) - 1, -1, -1):
            tensor_numel *= old_shape[tensor_dim]
            is_contiguous_chunk_end = tensor_dim == 0 or (
                old_shape[tensor_dim - 1] != 1 and old_stride[tensor_dim - 1] != tensor_numel * old_stride[tensor_dim]
            )
            if is_contiguous_chunk_end:
                while view_dim >= 0 and (view_numel < tensor_numel or new_shape[view_dim] == 1):
                    new_stride[view_dim] = view_numel * old_stride[tensor_dim]
                    view_numel *= new_shape[view_dim]
                    view_dim -= 1

                if view_numel != tensor_numel:
                    return None

                if tensor_dim > 0:
                    tensor_numel = 1
                    view_numel = 1

        if view_dim != -1:
            return None
        return tuple(new_stride)

    def _with_layout(self, shape: Tuple[int, ...], stride: Tuple[int, ...]) -> "TensorDesc":
        return TensorDesc(
            dtype=self.dtype,
            shape=shape,
            stride=stride,
            stride_order=self._compute_stride_order(shape, stride),
            device=self.device,
            interpret_uint8_as_fp4x2=self.interpret_uint8_as_fp4x2,
            name=self.name,
        )

    def __len__(self) -> int:
        if self.ndim == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    def size(self, dim: Optional[int] = None) -> Tuple[int, ...] | int:
        if dim is None:
            torch = _torch()
            return torch.Size(self.shape) if torch is not None else self.shape
        dim = self._normalize_dim(int(dim), self.ndim)
        return self.shape[dim]

    def permute(self, *dims: int | Tuple[int, ...] | List[int]) -> "TensorDesc":
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        dims = tuple(int(d) for d in dims)
        if len(dims) != self.ndim:
            raise RuntimeError(f"permute(): expected {self.ndim} dims, got {len(dims)}")
        dims = tuple(self._normalize_dim(d, self.ndim) for d in dims)
        if len(set(dims)) != self.ndim:
            raise RuntimeError(f"permute(): dims must be unique, got {dims}")

        new_shape = tuple(self.shape[d] for d in dims)
        new_stride = tuple(self.stride[d] for d in dims)
        return self._with_layout(new_shape, new_stride)

    def transpose(self, dim0: int, dim1: int) -> "TensorDesc":
        dim0 = self._normalize_dim(dim0, self.ndim)
        dim1 = self._normalize_dim(dim1, self.ndim)
        if dim0 == dim1:
            return self
        dims = list(range(self.ndim))
        dims[dim0], dims[dim1] = dims[dim1], dims[dim0]
        return self.permute(dims)

    def squeeze(self, dim: Optional[int | Tuple[int, ...] | List[int]] = None) -> "TensorDesc":
        if dim is None:
            keep_dims = [i for i, size in enumerate(self.shape) if size != 1]
        elif isinstance(dim, (tuple, list)):
            squeeze_dims = tuple(self._normalize_dim(int(d), self.ndim) for d in dim)
            if len(set(squeeze_dims)) != len(squeeze_dims):
                raise RuntimeError(f"squeeze(): dims must be unique, got {squeeze_dims}")
            squeeze_dims = {d for d in squeeze_dims if self.shape[d] == 1}
            keep_dims = [i for i in range(self.ndim) if i not in squeeze_dims]
        else:
            squeeze_dim = self._normalize_dim(int(dim), self.ndim)
            if self.shape[squeeze_dim] != 1:
                return self
            keep_dims = [i for i in range(self.ndim) if i != squeeze_dim]

        new_shape = tuple(self.shape[i] for i in keep_dims)
        new_stride = tuple(self.stride[i] for i in keep_dims)
        if new_shape == self.shape and new_stride == self.stride:
            return self
        return self._with_layout(new_shape, new_stride)

    def unsqueeze(self, dim: int) -> "TensorDesc":
        dim = self._normalize_dim(dim, self.ndim, allow_new_dim=True)

        if dim >= self.ndim:
            inserted_stride = 1
        else:
            inserted_stride = self.stride[dim] * self.shape[dim]

        new_shape = self.shape[:dim] + (1,) + self.shape[dim:]
        new_stride = self.stride[:dim] + (inserted_stride,) + self.stride[dim:]
        return self._with_layout(new_shape, new_stride)

    def is_contiguous(self, memory_format: Any = None) -> bool:
        """Row-major contiguity check; ``memory_format=None`` means torch.contiguous_format semantics."""
        torch = _torch()
        if memory_format is None or (torch is not None and memory_format in {torch.contiguous_format, torch.preserve_format}):
            if self._numel(self.shape) == 0:
                return True
            return self._is_contiguous_with_order(self.shape, self.stride, tuple(range(self.ndim - 1, -1, -1)))
        if torch is not None and memory_format == torch.channels_last:
            if self.ndim != 4:
                return False
            return self._is_contiguous_with_order(self.shape, self.stride, (1, 3, 2, 0))
        if torch is not None and memory_format == torch.channels_last_3d:
            if self.ndim != 5:
                return False
            return self._is_contiguous_with_order(self.shape, self.stride, (1, 4, 3, 2, 0))

        raise ValueError(f"Unsupported memory format: {memory_format}")

    def contiguous(self) -> "TensorDesc":
        if self.is_contiguous():
            return self
        contiguous_stride = self._compute_contiguous_stride(self.shape)
        return self._with_layout(self.shape, contiguous_stride)

    def view(self, *shape: int | Tuple[int, ...] | List[int]) -> "TensorDesc":
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = tuple(shape[0])
        new_shape = tuple(int(s) for s in shape)

        old_numel = self._numel(self.shape)
        infer_dim = None
        known_numel = 1
        for i, size in enumerate(new_shape):
            if size == -1:
                if infer_dim is not None:
                    raise RuntimeError("only one dimension can be inferred")
                infer_dim = i
                continue
            if size < 0:
                raise RuntimeError(f"invalid shape dimension {size}")
            known_numel *= size

        if infer_dim is not None:
            if known_numel == 0 or old_numel % known_numel != 0:
                raise RuntimeError(f"shape '{new_shape}' is invalid for input of size {old_numel}")
            inferred_size = old_numel // known_numel
            new_shape = new_shape[:infer_dim] + (inferred_size,) + new_shape[infer_dim + 1 :]
            known_numel *= inferred_size

        if known_numel != old_numel:
            raise RuntimeError(f"shape '{new_shape}' is invalid for input of size {old_numel}")

        new_stride = self._compute_view_stride(self.shape, self.stride, new_shape)
        if new_stride is None:
            raise RuntimeError(
                "view size is not compatible with input tensor's size and stride " "(at least one dimension spans across two contiguous subspaces)"
            )

        return self._with_layout(new_shape, new_stride)

    def as_strided(
        self,
        size: Tuple[int, ...] | List[int],
        stride: Tuple[int, ...] | List[int],
        storage_offset: int = 0,
    ) -> "TensorDesc":
        if storage_offset != 0:
            raise RuntimeError("TensorDesc.as_strided(): non-zero storage_offset is unsupported")
        if not isinstance(size, (tuple, list)) or not isinstance(stride, (tuple, list)):
            raise TypeError("TensorDesc.as_strided(): size and stride must be tuple/list")

        size = tuple(int(s) for s in size)
        stride = tuple(int(s) for s in stride)
        if len(size) != len(stride):
            raise RuntimeError(f"TensorDesc.as_strided(): mismatch in length of size ({len(size)}) and stride ({len(stride)})")
        if any(s < 0 for s in size):
            raise RuntimeError(f"TensorDesc.as_strided(): invalid size, got {size}")
        if any(s < 0 for s in stride):
            raise RuntimeError(f"TensorDesc.as_strided(): invalid stride, got {stride}")

        return self._with_layout(size, stride)


class APIBase(ABC):
    """Abstract base class for cuDNN API wrappers.

    This class defines the common interface that all API wrapper implementations
    should follow, including configuration validation, compilation, and execution.

    Provides common functionality:
    - Logging via self._logger
    - Support validation tracking via self._is_supported
    - Compiled kernel caching via self._compiled_kernel
    - Stream management helpers

    Subclasses should implement the abstract methods to provide
    API-specific validation logic and execution behavior.

    Example:
        >>> class MyKernelAPI(APIBase):
        ...     def __init__(self, sample_input, sample_output, config):
        ...         super().__init__()
        ...         self.sample_input = sample_input
        ...         self.sample_output = sample_output
        ...         self.config = config
        ...         self._kernel = MyKernel
        ...
        ...     def check_support(self) -> bool:
        ...         # Validate inputs and configuration
        ...         assert self.sample_input.dtype == torch.float32
        ...         self._is_supported = True
        ...         return True
        ...
        ...     def compile(self):
        ...         self._ensure_support_checked()
        ...         # Create and compile kernel
        ...         kernel = self._kernel(self.config)
        ...         self._compiled_kernel = cute.compile(kernel, ...)
        ...
        ...     def execute(self, input_tensor, output_tensor, current_stream=None):
        ...         current_stream = self._get_default_stream(current_stream)
        ...         self._compiled_kernel(input_tensor, output_tensor, current_stream)
    """

    def __init__(self):
        """Initialize the API base.

        Sets up:
        - self._is_supported: Flag indicating if configuration is validated
        - self._kernel: Kernel instance
        - self._compiled_kernel: Cache for compiled kernel
        - self._device_type: Architecture family used for dispatch/cache keys
        - self._is_rubin_kernel: True when running on Rubin (SM107)
        - self._logger: Logger instance for this class
        """
        self._is_supported = False
        self._kernel = None
        self._compiled_kernel = None
        self._interpret_uint8_as_fp4x2 = False
        self._device_type = get_device_type()
        self._is_rubin_kernel = self._device_type == "rubin"
        self._logger = logging.getLogger(self.__class__.__name__)

    def _warn_experimental_api(self) -> None:
        warn_experimental_api_once(self._logger, self.__class__.__name__)

    @abstractmethod
    def check_support(self) -> bool:
        """Check if the current configuration is supported by the kernel.

        This method should validate:
        - Input/output tensor shapes and strides
        - Data types compatibility
        - Hardware capabilities (compute capability, memory, etc.)
        - Configuration parameters (tile sizes, cluster shapes, etc.)

        Implementations should set self._is_supported = True if valid.

        :return: True if the configuration is supported
        :rtype: bool
        :raises AssertionError: If a configuration requirement is not met

        Example:
            >>> def check_support(self) -> bool:
            ...     self._logger.debug("Checking support")
            ...     assert self.input.dtype in {torch.float16, torch.float32}
            ...     assert self.input.shape[0] % 16 == 0, "Shape must be 16-aligned"
            ...     self._is_supported = True
            ...     return True
        """
        pass

    @abstractmethod
    def compile(self) -> None:
        """Compile the kernel with the current configuration.

        This method should:
        1. Ensure support has been checked (use self._ensure_support_checked())
        2. Create the underlying kernel implementation and fake cute tensors from the sample tensor descriptors
        3. Compile the kernel using cute.compile()
        4. Cache the compiled kernel in self._compiled_kernel

        :raises AssertionError: If the configuration is not supported

        Example:
            >>> def compile(self):
            ...     self._ensure_support_checked()
            ...
            ...     kernel = self._kernel(self.config)
            ...     sample_input_cute = self._make_fake_cute_tensor_from_desc(self.sample_input)
            ...     sample_output_cute = self._make_fake_cute_tensor_from_desc(self.sample_output)
            ...
            ...     self._compiled_kernel = cute.compile(
            ...         kernel,
            ...         sample_input_cute,
            ...         sample_output_cute
            ...     )
        """
        pass

    @abstractmethod
    def execute(
        self,
        *args,
        current_stream: Optional[cuda.CUstream] = None,
        **kwargs,
    ) -> Any:
        """Execute the kernel with the provided inputs.

        This method should execute using the cached compiled kernel.

        :param args: Positional arguments (typically input/output tensors)
        :param current_stream: CUDA stream for execution (optional)
        :type current_stream: cuda.CUstream or None
        :param kwargs: Additional keyword arguments for execution
        :return: Execution result (if any)
        :raises AssertionError: If compiled kernel is not available

        Example:
            >>> def execute(self, input_tensor, output_tensor, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     assert self._compiled_kernel is not None, "Kernel not compiled"
            ...     self._logger.debug("Executing with compiled kernel")
            ...     self._compiled_kernel(input_tensor, output_tensor, current_stream)
        """
        pass

    def __call__(self, *args, **kwargs) -> Any:
        """Convenience method to execute the kernel.

        This is a shorthand for compiling (if needed) and then executing.

        :param args: Positional arguments passed to execute()
        :param kwargs: Keyword arguments passed to execute()
        :return: Result from execute()

        Example:
            >>> api = MyKernelAPI(...)
            >>> api.check_support()
            >>> api.compile()
            >>> api(input_tensor, output_tensor)
        """
        if self._compiled_kernel is None:
            self.compile()
        return self.execute(*args, **kwargs)

    def _ensure_support_checked(self) -> None:
        """Helper to ensure check_support() was called before compilation.

        If check_support() has not been called yet (self._is_supported is False),
        this method will automatically call it. This prevents compilation
        with invalid configurations.

        :raises AssertionError: If check_support() returns False or raises

        Example:
            >>> def compile(self):
            ...     self._ensure_support_checked()  # Automatic validation
            ...     # ... rest of compilation
        """
        if not self._is_supported:
            self._logger.info(f"{self.__class__.__name__}: check_support not previously called, calling now")
            assert self.check_support(), "Unsupported configuration"

    def _get_default_stream(self, stream: Optional[cuda.CUstream]) -> cuda.CUstream:
        """Get default CUDA stream if none provided.

        This is a convenience helper to handle optional stream parameters.
        If a stream is provided, it is returned as-is. If None, the caller's
        current PyTorch stream is returned, so kernels stay ordered with
        surrounding torch ops under ``with torch.cuda.stream(s):``.

        :param stream: CUDA stream or None
        :type stream: cuda.CUstream or None
        :return: CUDA stream (either provided or default)
        :rtype: cuda.CUstream

        Example:
            >>> def execute(self, input_tensor, output_tensor, current_stream=None):
            ...     current_stream = self._get_default_stream(current_stream)
            ...     # Now current_stream is guaranteed to be a valid stream
        """
        if stream is not None:
            return stream
        torch = _torch()
        if torch is None:
            return cuda.CUstream(0)
        # _cuda_getCurrentRawStream is the same value torch.cuda.current_stream()
        # reports, without building the Stream wrapper: 0.1 us against 4.3 us,
        # and execute() calls this once per launch.
        return cuda.CUstream(_raw_stream(torch)(torch.cuda.current_device()))

    def _pad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor | TensorDesc]:
        """Pad a tensor/descriptor by unsqueezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Unsqueezes at dim -1 until tensor/descriptor rank == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor/descriptor to pad (or None)
        :param ndim: Target rank (pad trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The padded tensor/descriptor (or None)
        """
        if tensor is None:
            return None

        if tensor.ndim < ndim:
            self._logger.info(f"Padding {name} to {ndim}D from {tensor.shape}")
            if hasattr(tensor, "unsqueeze"):
                for _ in range(ndim - tensor.ndim):
                    tensor = tensor.unsqueeze(-1)
            else:
                # e.g. JAX arrays have no unsqueeze; appending unit dims via reshape is equivalent
                tensor = tensor.reshape(tuple(tensor.shape) + (1,) * (ndim - tensor.ndim))
        return tensor

    def _unpad_tensor_to_ndim(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        ndim: int,
        name: str,
    ) -> Optional[torch.Tensor | TensorDesc]:
        """Unpad a tensor/descriptor by squeezing at dim -1 until it reaches ndim rank.

        - If tensor is None, returns None.
        - Squeezes at dim -1 until tensor/descriptor rank == ndim.
        - Logs final reshape for traceability.

        :param tensor: The tensor/descriptor to unpad (or None)
        :param ndim: Target rank (squeeze trailing dims until reached)
        :param name: Logical tensor name for logging
        :return: The unpadded tensor/descriptor (or None)
        """
        if tensor is None:
            return None

        if tensor.ndim > ndim:
            self._logger.info(f"Unpadding {name} from {tensor.shape} to {ndim}D")
            for _ in range(tensor.ndim - ndim):
                if tensor.shape and tensor.shape[-1] == 1:
                    tensor = tensor.squeeze(-1)
                else:
                    break

            if tensor.ndim != ndim:
                self._logger.critical(f"Unpadding {name} resulted in shape {tensor.shape}, expected {ndim}D")
        return tensor

    def _is_fp4x2(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an FP4x2 packed datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP4x2 packed type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        if isinstance(tensor_or_dtype, TensorDesc):
            dtype = tensor_or_dtype.dtype
            interpret_uint8_as_fp4x2 = tensor_or_dtype.interpret_uint8_as_fp4x2
        elif _is_framework_tensor(tensor_or_dtype):
            dtype = tensor_or_dtype.dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        else:
            dtype = tensor_or_dtype
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        # Compare in canonical (cutlass) dtype space so torch/jax/numpy dtypes all resolve;
        # dtypes with no cutlass mapping compare False, as before.
        dtype = _convert_to_cutlass_data_type_or_none(dtype)
        return (dtype is cutlass.Float4E2M1FN) or (interpret_uint8_as_fp4x2 and dtype is cutlass.Uint8)

    def _is_fp8(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an FP8 datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an FP8 type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, TensorDesc) or _is_framework_tensor(tensor_or_dtype) else tensor_or_dtype
        return _convert_to_cutlass_data_type_or_none(dtype) in {cutlass.Float8E5M2, cutlass.Float8E4M3FN}

    def _is_f16(self, tensor_or_dtype: torch.Tensor | torch.dtype | TensorDesc) -> bool:
        """Check if tensor or dtype is an fp16 or bf16 datatype.

        :param tensor_or_dtype: The torch tensor or dtype to check
        :type tensor_or_dtype: torch.Tensor | torch.dtype
        :return: True if tensor/dtype is an fp16 or bf16 type
        :rtype: bool
        """
        if tensor_or_dtype is None:
            return False
        dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, TensorDesc) or _is_framework_tensor(tensor_or_dtype) else tensor_or_dtype
        return _convert_to_cutlass_data_type_or_none(dtype) in {cutlass.Float16, cutlass.BFloat16}

    def _get_innermost_stride_dim(self, tensor: torch.Tensor, name: str = "") -> int:
        """Return index of innermost contiguous dimension (stride == 1).

        :raises RuntimeError: If no dimension with stride 1 is found.
        """
        tensor_stride = get_strides(tensor)
        idx = next((i for i, s in enumerate(tensor_stride) if s == 1), None)
        if idx is None:
            self._logger.critical(f"tensor {name} has shape: {tensor.shape} stride {tensor_stride} – innermost contiguous (stride == 1) dimension not found. ")
            raise RuntimeError(f"tensor {name} has shape: {tensor.shape} stride {tensor_stride} – innermost contiguous (stride == 1) dimension not found. ")
        return idx

    def _tensor_shape(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Get the logical shape of a tensor, handling FP4x2 packed datatypes.

        For FP4x2 datatypes, two values are packed per byte. The innermost
        contiguous dimension (with stride 1) contains packed values, so the
        logical shape for that dimension is 2x the physical shape.

        :param tensor: The tensor to get shape from (or None)
        :type tensor: torch.Tensor or None
        :param name: Logical tensor name for logging
        :type name: str
        :return: The logical shape tuple (or None if tensor is None)
        :rtype: Tuple[int, ...] or None
        """
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.shape

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            shape = tuple(dim * 2 if i == innermost_dim_index else dim for i, dim in enumerate(get_shape(tensor)))
            self._logger.debug(f"FP4x2 tensor {name}: physical shape {tensor.shape} -> logical shape {shape}")
            return shape
        else:
            return get_shape(tensor)

    def _tensor_stride(
        self,
        tensor: Optional[torch.Tensor | TensorDesc],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Get the logical stride of a tensor, handling FP4x2 packed datatypes.

        For FP4x2 datatypes, two values are packed per byte. The strides must
        be adjusted to reflect logical element spacing. All strides are
        multiplied by 2 since each physical element contains 2 logical elements.

        :param tensor: The tensor to get stride from (or None)
        :type tensor: torch.Tensor or None
        :param name: Logical tensor name for logging
        :type name: str
        :return: The logical stride tuple (or None if tensor is None)
        :rtype: Tuple[int, ...] or None
        """
        if tensor is None:
            return None
        if isinstance(tensor, TensorDesc):
            return tensor.stride

        if self._is_fp4x2(tensor):
            innermost_dim_index = self._get_innermost_stride_dim(tensor, name=name)
            tensor_stride = get_strides(tensor)
            strides = tuple(s * 2 if i != innermost_dim_index else s for i, s in enumerate(tensor_stride))
            self._logger.debug(f"FP4x2 tensor {name}: physical stride {tensor_stride} -> logical stride {strides}")
            return strides
        else:
            return get_strides(tensor)

    def _check_tensor_shape(
        self,
        tensor_or_shape: torch.Tensor | TensorDesc | Tuple[int, ...],
        shape: Tuple[int, ...] | List[Tuple[int, ...]],
        name: str = "",
    ) -> Optional[Tuple[int, ...]]:
        """Check if the shape of a tensor matches the expected shape(s).

        :param tensor_or_shape: The tensor to get shape from or the shape to check
        :type tensor_or_shape: torch.Tensor | TensorDesc | Tuple[int, ...]
        :param shape: expected shape or list of expected shapes
        :type shape: Tuple[int, ...] | List[Tuple[int, ...]]
        :param name: Logical tensor name for logging
        :type name: str
        :raises ValueError: If the shape of the tensor does not match the expected shape(s)
        :return: The logical shape of the tensor
        :rtype: Optional[Tuple[int, ...]]
        """
        if tensor_or_shape is None:
            return None
        tensor_shape = (
            self._tensor_shape(tensor_or_shape, name=name)
            if isinstance(tensor_or_shape, TensorDesc) or _is_framework_tensor(tensor_or_shape)
            else tensor_or_shape
        )
        if isinstance(shape, tuple):
            if tensor_shape != shape:
                raise ValueError(f"{name} tensor shape mismatch: expected {shape}, got {tensor_shape}")
        elif isinstance(shape, list):
            if tensor_shape not in shape:
                raise ValueError(f"{name} tensor shape mismatch: expected one of {shape}, got {tensor_shape}")
        else:
            raise ValueError(f"Expected shape to be a tuple or list, got {type(shape)}")
        return tensor_shape

    def _check_tensor_stride(
        self,
        tensor_or_stride: torch.Tensor | TensorDesc | Tuple[int, ...],
        stride: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        stride_order: Optional[Tuple[int, ...] | List[Tuple[int, ...]]] = None,
        name: str = "",
        extra_error_msg: str = "",
    ) -> Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Check if the stride of a tensor matches the expected stride(s) or stride order(s).

        :param tensor_or_stride: The tensor to get stride from or the stride to check
        :type tensor_or_stride: torch.Tensor | TensorDesc | Tuple[int, ...]
        :param stride: The expected stride(s)
        :type stride: Tuple[int, ...] | List[Tuple[int, ...]]
        :param stride_order: The expected stride order(s)
        :type stride_order: Tuple[int, ...] | List[Tuple[int, ...]]
        :param name: Logical tensor name for logging
        :type name: str
        :param extra_error_msg: Extra error message to add to the error
        :type extra_error_msg: str
        :raises ValueError: If the stride of the tensor does not match the expected stride order
        :return: The stride and stride order of the tensor
        :rtype: Optional[Tuple[Tuple[int, ...], Tuple[int, ...]]]
        """
        if tensor_or_stride is None:
            return None, None
        if isinstance(tensor_or_stride, TensorDesc):
            tensor_stride = tensor_or_stride.stride
            tensor_stride_order = tensor_or_stride.stride_order
        elif _is_framework_tensor(tensor_or_stride):
            tensor_stride = self._tensor_stride(tensor_or_stride, name=name)
            tensor_stride_order = tuple(i for i, s in sorted(enumerate(tensor_stride), key=lambda x: x[1]))
        else:
            tensor_stride = tensor_or_stride
            tensor_stride_order = tuple(i for i, s in sorted(enumerate(tensor_stride), key=lambda x: x[1]))

        if stride is not None:
            if isinstance(stride, tuple):
                if tensor_stride != stride:
                    error_msg = f"{name} tensor stride mismatch: expected {stride}, got {tensor_stride}"
                    if extra_error_msg:
                        error_msg += f": {extra_error_msg}"
                    raise ValueError(error_msg)
            elif isinstance(stride, list):
                if tensor_stride not in stride:
                    error_msg = f"{name} tensor stride mismatch: expected one of {stride}, got {tensor_stride}"
                    if extra_error_msg:
                        error_msg += f": {extra_error_msg}"
                    raise ValueError(error_msg)
            else:
                error_msg = f"Expected stride to be a tuple or list, got {type(stride)}"
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        if stride_order is not None:
            if isinstance(stride_order, tuple):
                if tensor_stride_order != stride_order:
                    error_msg = f"{name} tensor stride order mismatch: expected {stride_order}, got {tensor_stride_order}"
                    if extra_error_msg:
                        error_msg += f": {extra_error_msg}"
                    raise ValueError(error_msg)
            elif isinstance(stride_order, list):
                if tensor_stride_order not in stride_order:
                    error_msg = f"{name} tensor stride order mismatch: expected one of {stride_order}, got {tensor_stride_order}"
                    if extra_error_msg:
                        error_msg += f": {extra_error_msg}"
                    raise ValueError(error_msg)
            else:
                error_msg = f"Expected stride order to be a tuple or list, got {type(stride_order)}"
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        return tensor_stride, tensor_stride_order

    def _check_dtype(
        self,
        tensor_or_dtype: torch.Tensor | TensorDesc | torch.dtype,
        dtype: torch.dtype | List[torch.dtype],
        name: str = "",
        extra_error_msg: str = "",
    ) -> Optional[torch.dtype]:
        """Check if the dtype of a tensor or dtype matches the expected dtype(s).

        :param tensor_or_dtype: The tensor to get dtype from or the dtype to check
        :type tensor_or_dtype: torch.Tensor | TensorDesc | torch.dtype
        :param dtype: The expected dtype(s)
        :type dtype: torch.dtype | List[torch.dtype]
        :param name: Logical tensor name for logging
        :type name: str
        :raises ValueError: If the dtype of the tensor does not match the expected dtype(s)
        :return: The dtype of the tensor
        :rtype: Optional[torch.dtype]
        """
        if tensor_or_dtype is None:
            return None
        tensor_dtype = tensor_or_dtype.dtype if isinstance(tensor_or_dtype, TensorDesc) or _is_framework_tensor(tensor_or_dtype) else tensor_or_dtype
        torch = _torch()
        is_scalar_dtype = (torch is not None and isinstance(dtype, torch.dtype)) or (isinstance(dtype, type) and issubclass(dtype, cutlass.Numeric))
        if is_scalar_dtype:
            if tensor_dtype != dtype:
                error_msg = f"{name} dtype mismatch: expected {dtype}, got {tensor_dtype}"
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        elif isinstance(dtype, list):
            if tensor_dtype not in dtype:
                error_msg = f"{name} dtype mismatch: expected one of {dtype}, got {tensor_dtype}"
                if extra_error_msg:
                    error_msg += f": {extra_error_msg}"
                raise ValueError(error_msg)
        else:
            raise ValueError(f"Expected dtype to be a torch.dtype or list, got {type(dtype)}")
        return tensor_dtype

    def _value_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a ValueError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises ValueError: If the condition is true
        """
        if condition:
            raise ValueError(error_msg)

    def _not_implemented_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a NotImplementedError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises NotImplementedError: If the condition is true
        """
        if condition:
            raise NotImplementedError(error_msg)

    def _runtime_error_if(self, condition: bool, error_msg: str) -> None:
        """Raise a RuntimeError if the condition is true.

        :param condition: The condition to check
        :type condition: bool
        :param error_msg: The error message to raise
        :type error_msg: str
        :raises RuntimeError: If the condition is true
        """
        if condition:
            raise RuntimeError(error_msg)

    def _make_fake_cute_tensor_like(
        self,
        tensor: torch.Tensor,
        assumed_align: int = 16,
        name: str = "",
    ) -> cute.Pointer:
        """Make a fake tensor like the provided tensor.
        :param tensor: The tensor to make a fake tensor like
        :type tensor: torch.Tensor
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :param name: Logical tensor name for logging
        :type name: str
        :return: A fake tensor like the provided tensor
        :rtype: cute.Pointer
        """
        return self._make_fake_cute_tensor_from_desc(
            self._make_tensor_desc(tensor, name=name),
            assumed_align=assumed_align,
        )

    def _make_tensor_desc(
        self,
        tensor: Optional[Any],
        name: str = "",
        interpret_uint8_as_fp4x2: Optional[bool] = None,
        canonical: bool = False,
    ) -> Optional[TensorDesc]:
        """Capture logical tensor metadata that is sufficient for validation/compile.

        With ``canonical=True`` the descriptor is framework-neutral: the dtype is converted
        to its cutlass type, the device is read via the tensor adapter (torch tensors keep
        torch.device), and extent-1 dims get canonicalized strides so layouts that differ
        only in unit-dim strides -- unobservable by the kernels -- compare and compile equal.
        """
        if tensor is None:
            return None
        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        prev_interpret = self._interpret_uint8_as_fp4x2
        self._interpret_uint8_as_fp4x2 = interpret_uint8_as_fp4x2
        try:
            tensor_shape = self._tensor_shape(tensor, name=name)
            tensor_stride = self._tensor_stride(tensor, name=name)
        finally:
            self._interpret_uint8_as_fp4x2 = prev_interpret
        if canonical:
            dtype = _convert_to_cutlass_data_type(tensor.dtype)
            device = get_device(tensor)
            tensor_stride = canonicalize_unit_dim_strides(tensor_shape, tensor_stride)
        else:
            dtype = tensor.dtype
            device = tensor.device
        tensor_stride_order = tuple(i for i, s in sorted(enumerate(tensor_stride), key=lambda x: (x[1], tensor_shape[x[0]])))
        return TensorDesc(
            dtype=dtype,
            shape=tensor_shape,
            stride=tensor_stride,
            stride_order=tensor_stride_order,
            device=device,
            interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2,
            name=name,
        )

    def _make_fake_cute_tensor_from_desc(
        self,
        tensor_desc: Optional[TensorDesc],
        assumed_align: int = 16,
    ) -> Optional[cute.Pointer]:
        """Build a fake cute tensor from a descriptor."""
        if tensor_desc is None:
            return None
        return self._make_fake_cute_tensor(
            dtype=tensor_desc.dtype,
            shape=tensor_desc.shape,
            stride=tensor_desc.stride,
            assumed_align=assumed_align,
            interpret_uint8_as_fp4x2=tensor_desc.interpret_uint8_as_fp4x2,
        )

    def _make_fake_cute_tensor(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stride: Tuple[int, ...],
        assumed_align: int = 16,
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> cute.Pointer:
        """Make a fake tensor.

        :param dtype: The dtype of the tensor
        :type dtype: torch.dtype
        :param shape: The shape of the tensor
        :type shape: Tuple[int, ...]
        :param stride: The stride of the tensor
        :type stride: Tuple[int, ...]
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :return: A fake tensor
        :rtype: cute.Pointer
        """
        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return cute.runtime.make_fake_tensor(
            dtype=_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2),
            shape=shape,
            stride=stride,
            assumed_align=assumed_align,
        )

    def _make_fake_cute_compact_tensor(
        self,
        dtype: torch.dtype,
        shape: Tuple[int, ...],
        stride_order: Tuple[int, ...],
        assumed_align: int = 16,
        dynamic_mode: Optional[int] = None,
        divisibility: int = 16,
        interpret_uint8_as_fp4x2: Optional[bool] = None,
    ) -> cute.Pointer:
        """Make a fake compact tensor.
        :param dtype: The dtype of the tensor
        :type dtype: torch.dtype
        :param shape: The shape of the tensor
        :type shape: Tuple[int, ...]
        :param stride_order: The stride order of the tensor
        :type stride_order: Tuple[int, ...]
        :param assumed_align: The assumed alignment of the tensor
        :type assumed_align: int
        :return: A fake compact tensor
        :rtype: cute.Pointer
        """
        if dynamic_mode is not None:
            dynamic_dim = cute.sym_int(divisibility=divisibility)
            shape = shape[:dynamic_mode] + (dynamic_dim,) + shape[dynamic_mode + 1 :]

        if interpret_uint8_as_fp4x2 is None:
            interpret_uint8_as_fp4x2 = self._interpret_uint8_as_fp4x2
        return cute.runtime.make_fake_compact_tensor(
            dtype=_convert_to_cutlass_data_type(dtype, interpret_uint8_as_fp4x2=interpret_uint8_as_fp4x2),
            shape=shape,
            stride_order=stride_order,
            assumed_align=assumed_align,
        )


# Workspace-carve chunk alignment. The contract minimum is 16 bytes; 128 is
# used so the per-sequence O TMA descriptors carved for the THD path satisfy
# the cuTensorMap GMEM alignment (64 B) with margin. torch storage bases are
# 512 B aligned, so 128 B-multiple offsets stay 128 B aligned absolutely.
_WS_ALIGN = 128


def ws_align(nbytes: int) -> int:
    """Round a scratch-chunk size up to the carve alignment (128 B)."""
    return -(-int(nbytes) // _WS_ALIGN) * _WS_ALIGN


class WorkspaceCarver:
    """Carves fixed-size, aligned scratch views out of the CALLER's workspace (R2).

    An APIBase adapter that records a non-zero workspace requirement
    (``scratch_workspace_bytes()`` / ``get_workspace_size()``) is handed the
    caller's buffer at execute and carves its per-execute scratch from it
    instead of allocating. Chunks are dealt sequentially at 128-byte relative
    alignment and never reach beyond the buffer; an absent, non-torch, or
    undersized buffer raises immediately with the required size in the message
    (never silent corruption). Constructing the carver without taking a chunk
    is the validation step for a backend graph that consumes the workspace
    whole (``pygraph.execute(variant_pack, workspace)``). ``align`` is the base
    alignment the consumer needs: 16 bytes by default; a path that carves TMA
    descriptors out of the buffer asks for ``_WS_ALIGN`` (cuTensorMap wants 64).
    """

    def __init__(self, workspace, required: int, owner: str, *, align: int = 16):
        if workspace is None:
            raise ValueError(
                f"cudnn: {owner} requires a {required}-byte workspace but execute() "
                f"received none; allocate graph.get_workspace_size() bytes (uint8, on the "
                f"graph's device) and pass the buffer to execute()"
            )
        if not (hasattr(workspace, "numel") and hasattr(workspace, "element_size") and hasattr(workspace, "view")):
            raise TypeError(f"cudnn: {owner} carves its scratch out of the caller's workspace and needs a torch.Tensor; got {type(workspace).__name__}")
        torch = _torch()
        if not workspace.is_contiguous():
            # reshape(-1) would copy a strided view: a hidden allocation carved into instead of the caller's buffer.
            raise ValueError(f"cudnn: {owner} workspace must be contiguous; got shape {tuple(workspace.shape)} strides {tuple(workspace.stride())}")
        flat = workspace if workspace.dtype == torch.uint8 else workspace.view(torch.uint8)
        if flat.dim() != 1:
            flat = flat.view(-1)
        if flat.numel() < required:
            raise ValueError(
                f"cudnn: {owner} requires a {required}-byte workspace; the provided "
                f"buffer has only {flat.numel()} bytes (size it with graph.get_workspace_size())"
            )
        if flat.data_ptr() % align != 0:
            raise ValueError(f"cudnn: {owner} workspace must be at least {align}-byte aligned; got data_ptr=0x{flat.data_ptr():x}")
        self._flat = flat
        self._off = 0
        self._owner = owner

    def take(self, numel: int, dtype) -> Any:
        """The next scratch chunk: a 1-D ``numel``-element view of ``dtype``."""
        nbytes = int(numel) * dtype.itemsize
        start, end = self._off, self._off + nbytes
        if end > self._flat.numel():
            raise ValueError(f"cudnn: {self._owner} workspace overrun: chunk [{start}, {end}) exceeds the {self._flat.numel()}-byte buffer (sizing bug)")
        self._off = start + ws_align(nbytes)
        chunk = self._flat if start == 0 and end == self._flat.numel() else self._flat[start:end]
        try:
            return chunk.view(dtype)
        except RuntimeError as exc:
            raise ValueError(f"cudnn: {self._owner} workspace is not sufficiently aligned for {dtype} scratch: {exc}") from None

    def remaining(self) -> Any:
        """The unconsumed tail (uint8) — handed down to a nested carver."""
        return self._flat[self._off :]
