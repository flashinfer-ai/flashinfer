"""
Copyright (c) 2023 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from enum import Enum
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.version
import pynvml
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version
import inspect

from .jit.spdlog import gen_spdlog_module


class PosEncodingMode(Enum):
    NONE = 0
    ROPE_LLAMA = 1
    ALIBI = 2


class MaskMode(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1
    CUSTOM = 2
    MULTIITEMSCORING = 3


class TensorLayout(Enum):
    NHD = 0
    HND = 1


log2e = 1.44269504088896340736


class GPUArchitectureError(Exception):
    """Custom exception for GPU architecture-related errors."""

    pass


class LibraryError(Exception):
    """Custom exception for library-related errors."""

    pass


class BackendSupportedError(Exception):
    """Custom exception for backend-related errors."""

    pass


def _expand_5d(x: torch.Tensor, kv_layout: str) -> torch.Tensor:
    if x.ndim not in [4, 5]:
        raise ValueError("x must be 4D or 5D")
    if x.ndim == 4:
        # page_size == 1
        if kv_layout == "NHD":
            # (num_pages, 2, num_heads, head_dim) -> (num_pages, 2, page_size=1, num_heads, head_dim)
            # expand to 5D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # (num_pages, 2, num_heads, head_dim) -> (num_pages, 2, num_heads, page_size=1, head_dim)
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def _expand_4d(x: torch.Tensor, kv_layout: str) -> torch.Tensor:
    if x.ndim not in [3, 4]:
        raise ValueError("x must be 3D or 4D")
    if x.ndim == 3:
        # page_size == 1
        if kv_layout == "NHD":
            # (num_pages, num_heads, head_dim) -> (num_pages, page_size=1, num_heads, head_dim)
            # expand to 4D on the 3nd last dimension
            return x.unsqueeze(-3)
        elif kv_layout == "HND":
            # (num_pages, num_heads, head_dim) -> (num_pages, num_heads, page_size=1, head_dim)
            # expand to 5D on the 2nd last dimension
            return x.unsqueeze(-2)
        else:
            raise KeyError("Invalid kv_layout {}".format(kv_layout))
    return x


def next_positive_power_of_2(x: int) -> int:
    if x < 1:
        return 1

    # Following code is equivalent to 1 << (x - 1).bit_length()
    # But this impl does not contain bit_length() so can be used by torch compile.
    # It can correctly handle 64bit number which should be enough for now.
    n = x - 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def calculate_tile_tokens_dim(
    num_tokens: int, num_experts: int, top_k: int, max_tile_tokens_dim: int = 128
) -> int:
    # Factor to account for the imbalance of the experts.
    # factor equals to the
    # max_real_num_tokens_per_expert / perfect_num_tokens_per_expert
    # - 1.0 means perfect expert distribution.
    # - > 1.0 means some experts have more
    #     tokens than the perfect distribution.
    # - < 1.0 does not make sense.
    imbalance_factor = 1.3
    # Calculate the number of tokens per expert
    # assuming perfect distribution.
    num_tokens_per_expert = (num_tokens * top_k) // num_experts
    # Apply the imbalance factor.
    num_tokens_per_expert = int(num_tokens_per_expert * imbalance_factor)
    # And pad the number to the next power of 2.
    tile_tokens_dim = next_positive_power_of_2(num_tokens_per_expert)
    # Cap to 8-max_tile_tokens_dim tokens per CTA tile
    # as it's the range supported by the kernel.
    tile_tokens_dim = min(max(tile_tokens_dim, 8), max_tile_tokens_dim)
    return tile_tokens_dim


def _check_pos_encoding_mode(pos_encoding_mode: str) -> None:
    if not hasattr(PosEncodingMode, pos_encoding_mode):
        raise KeyError("Invalid pos_encoding_mode {}".format(pos_encoding_mode))


def _check_kv_layout(kv_layout: str) -> None:
    if not hasattr(TensorLayout, kv_layout):
        raise KeyError("Invalid kv_layout {}".format(kv_layout))


def is_float8(x: torch.Tensor) -> bool:
    return x.dtype in [torch.float8_e4m3fn, torch.float8_e5m2]


def get_indptr(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.int64)
    ret = torch.zeros(x.shape[0] + 1, dtype=x.dtype, device=x.device)
    ret[1:] = x.cumsum(0)
    return ret


def _unpack_paged_kv_cache(
    paged_kv_cache: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
    kv_layout: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(paged_kv_cache, tuple):
        paged_k_cache, paged_v_cache = paged_kv_cache
        return (
            _expand_4d(paged_k_cache, kv_layout),
            _expand_4d(paged_v_cache, kv_layout),
        )
    elif torch.is_tensor(paged_kv_cache):
        # NOTE(Zihao): split on the second dimension
        paged_kv_cache = _expand_5d(paged_kv_cache, kv_layout)
        paged_k_cache, paged_v_cache = paged_kv_cache.unbind(dim=1)
        return paged_k_cache, paged_v_cache
    else:
        raise KeyError(
            "Unrecognized paged_kv_cache type {}, expect a single tensor or a tuple of tensor.".format(
                type(paged_kv_cache)
            )
        )


def get_alibi_slopes(n_heads: int) -> torch.Tensor:
    n = 2 ** math.floor(math.log2(n_heads))
    m_0 = 2.0 ** (-8.0 / n)
    m = torch.pow(m_0, torch.arange(1, 1 + n))
    if n < n_heads:
        m_hat_0 = 2.0 ** (-4.0 / n)
        m_hat = torch.pow(m_hat_0, torch.arange(1, 1 + 2 * (n_heads - n), 2))
        m = torch.cat([m, m_hat])
    return m.float()


_cache_buf: Dict[Tuple[str, torch.device], torch.Tensor] = {}


def _get_cache_buf(
    name: str, bytes: int, device: torch.device, zero_init: bool = False
) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None or buf.size(0) < bytes:
        if zero_init:
            buf = torch.zeros(bytes, dtype=torch.uint8, device=device)
        else:
            buf = torch.empty(bytes, dtype=torch.uint8, device=device)
        _cache_buf[key] = buf
    return buf


# find the least power of 2 that is greater than or equal to x
def _ceil_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


def _get_range_buf(seq_len: int, device: torch.device) -> torch.Tensor:
    seq_len_pow2 = _ceil_pow2(seq_len)
    key = (f"range_{seq_len_pow2}", device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = torch.arange(seq_len_pow2, device=device, dtype=torch.int32)
        _cache_buf[key] = buf
    return buf[:seq_len]


def _get_cache_alibi_slopes_buf(
    num_qo_heads: int, device: torch.device
) -> torch.Tensor:
    key = (f"alibi_slopes_{num_qo_heads}", device)
    buf = _cache_buf.get(key)
    if buf is None:
        buf = get_alibi_slopes(num_qo_heads).to(device)
        _cache_buf[key] = buf
    return buf


def canonicalize_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if isinstance(dtype, str):
        return getattr(torch, dtype)
    elif isinstance(dtype, torch.dtype):
        return dtype
    else:
        raise TypeError(
            "dtype must be a string or torch.dtype, got {}".format(type(dtype))
        )


@functools.cache
def get_compute_capability(device: torch.device) -> Tuple[int, int]:
    if device.type != "cuda":
        raise ValueError("device must be a cuda device")
    return torch.cuda.get_device_capability(device.index)


@functools.cache
def get_gpu_memory_bandwidth(device: torch.device) -> float:
    """
    Get GPU memory bandwidth in GB/s for the specified CUDA device.

    Args:
        device: torch.device object, e.g., torch.device('cuda:0')

    Returns:
        float: GPU memory bandwidth (GB/s)

    Raises:
        ValueError: If device is not a CUDA device
    """
    # Convert to torch.device object if string is passed
    if isinstance(device, str):
        device = torch.device(device)

    # Check if it's a CUDA device
    if device.type != "cuda":
        raise ValueError(f"Device must be a CUDA device, got {device}")

    # Get device index
    device_index = device.index if device.index is not None else 0

    # Use pynvml to get bandwidth
    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)

        # Calculate theoretical peak bandwidth (GB/s)
        bandwidth = (mem_clock * bus_width * 2) / 8 / 1000

        return bandwidth
    finally:
        pynvml.nvmlShutdown()


def _check_cached_qkv_data_type(
    q: torch.Tensor, k: torch.Tensor, dtype_q: torch.dtype, dtype_kv: torch.dtype
) -> None:
    if q.dtype != dtype_q:
        raise ValueError(
            f"The dtype of q {q.dtype} does not match the q_data_type {dtype_q} specified in plan function."
        )
    if k.dtype != dtype_kv:
        raise ValueError(
            f"The dtype of k {k.dtype} does not match the kv_data_type {dtype_kv} specified in plan function."
        )


if TorchVersion(torch_version) < TorchVersion("2.4"):

    def register_custom_op(
        name: str,
        fn: Optional[Callable] = None,
        /,
        *,
        mutates_args: Union[str, Iterable[str]],
        device_types: Optional[Union[str, Sequence[str]]] = None,
        schema: Optional[str] = None,
    ) -> Callable:
        return lambda x: x

    def register_fake_op(
        name: str,
        fn: Optional[Callable] = None,
    ) -> Callable:
        return lambda x: x

else:

    def register_custom_op(
        name: str,
        fn: Optional[Callable] = None,
        /,
        *,
        mutates_args: Union[str, Iterable[str]],
        device_types: Optional[Union[str, Sequence[str]]] = None,
        schema: Optional[str] = None,
    ) -> Callable:
        # NOTE(Zihao): torch.library.custom_op has significant overhead as mentioned in the following link
        # https://github.com/vllm-project/vllm/blob/36e76700453924c8d421db99af70a88a1df835cd/vllm/utils.py#L1660-L1674

        # return torch.library.custom_op(
        #     name,
        #     fn,
        #     mutates_args=mutates_args,
        #     device_types=device_types,
        #     schema=schema,
        # )
        return lambda x: x

    def register_fake_op(
        name: str,
        fn: Optional[Callable] = None,
    ) -> Callable:
        # return torch.library.register_fake(name, fn)
        return lambda x: x


def determine_gemm_backend(device: torch.device) -> str:
    major, _ = get_compute_capability(device)
    if major == 9 and torch.version.cuda >= "12.3":
        return "sm90"
    else:
        return "sm80"


def is_fa3_backend_supported(
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> bool:
    """
    Check if the FA3 backend is supported based on the given parameters.
    NOTE(Zihao): this function is a workaround for the lack of support for certain features in
    our FA3 backend, and will be removed once the backend is fully supported.

    Parameters
    ----------
    pos_encoding_mode : int
        The positional encoding mode.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reductions are allowed.
    use_custom_mask : bool
        Whether a custom mask is used.
    dtype_q : torch.dtype
        The data type of the query tensor.
    dtype_kv : torch.dtype
        The data type of the key-value tensor.

    Returns
    -------
    bool
        True if the FA3 backend is supported, False otherwise.
    """
    if use_custom_mask:
        return False
    if pos_encoding_mode != PosEncodingMode.NONE.value:
        return False
    if use_fp16_qk_reductions:
        return False
    return True


def is_cutlass_backend_supported(
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> bool:
    """
    Check if the cutlass backend is supported based on the given parameters.

    Parameters
    ----------
    pos_encoding_mode : int
        The positional encoding mode.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reductions are allowed.
    use_custom_mask : bool
        Whether a custom mask is used.
    dtype_q : torch.dtype
        The data type of the query tensor.
    dtype_kv : torch.dtype
        The data type of the key-value tensor.

    Returns
    -------
    bool
        True if the cutlass backend is supported, False otherwise.
    """
    if use_custom_mask:
        return False
    if pos_encoding_mode != PosEncodingMode.NONE.value:
        return False
    if use_fp16_qk_reductions:
        return False
    if dtype_q in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return False
    if dtype_kv in [torch.float8_e4m3fn, torch.float8_e5m2]:
        return False
    return True


def determine_attention_backend(
    device: torch.device,
    pos_encoding_mode: int,
    use_fp16_qk_reductions: bool,
    use_custom_mask: bool,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
) -> str:
    """
    Determine the appropriate attention backend based on the device and parameters.

    Parameters
    ----------
    device : torch.device
        The device to be used.
    mask_mode : int
        The mask mode.
    pos_encoding_mode : int
        The positional encoding mode.
    use_fp16_qk_reductions : bool
        Whether FP16 QK reductions are allowed.
    use_custom_mask : bool
        Whether a custom mask is used.
    dtype_q : torch.dtype
        The data type of the query tensor.
    dtype_kv : torch.dtype
        The data type of the key-value tensor.

    Returns
    -------
    str
        The name of the attention backend to be used.
    """
    if is_sm90a_supported(device) and is_fa3_backend_supported(
        pos_encoding_mode,
        use_fp16_qk_reductions,
        use_custom_mask,
        dtype_q,
        dtype_kv,
    ):
        return "fa3"
    else:
        return "fa2"


def version_at_least(version: str, base_version: str) -> bool:
    from packaging import version as pkg_version

    return pkg_version.parse(version) >= pkg_version.parse(base_version)


def has_cuda_cudart() -> bool:
    """
    Check if cuda.cudart module is available (cuda-python <= 12.9).

    Returns:
        True if cuda.cudart exists, False otherwise
    """
    import importlib.util

    return importlib.util.find_spec("cuda.cudart") is not None


# Re-export from jit.env to avoid circular dependency
from .jit.env import (
    has_flashinfer_jit_cache as has_flashinfer_jit_cache,
    has_flashinfer_cubin as has_flashinfer_cubin,
)


def get_cuda_python_version() -> str:
    import cuda

    return cuda.__version__


def is_sm90a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 9 and version_at_least(torch.version.cuda, "12.3")


def is_sm100a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 10 and version_at_least(torch.version.cuda, "12.8")


def is_sm100f_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 10 and version_at_least(torch.version.cuda, "12.9")


def is_sm110a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 11 and version_at_least(torch.version.cuda, "13.0")


def is_sm120a_supported(device: torch.device) -> bool:
    major, minor = get_compute_capability(device)
    return major == 12 and minor == 0 and version_at_least(torch.version.cuda, "12.8")


def is_sm121a_supported(device: torch.device) -> bool:
    major, minor = get_compute_capability(device)
    return major == 12 and minor == 1 and version_at_least(torch.version.cuda, "12.9")


def determine_mla_backend(device: torch.device) -> str:
    return "fa3" if is_sm90a_supported(device) else "fa2"


def check_shape_dtype_device(
    x: torch.Tensor,
    expected_shape: Optional[Sequence[int]],
    expected_dtype: Optional[torch.dtype],
    expected_device: Optional[torch.device],
    name: str,
) -> None:
    if expected_shape and x.shape != torch.Size(expected_shape):
        raise ValueError(
            f"Invalid shape of {name}: expected {expected_shape}, got {x.shape}"
        )
    if expected_dtype and x.dtype != expected_dtype:
        raise ValueError(
            f"Invalid dtype of {name}: expected {expected_dtype}, got {x.dtype}"
        )
    if expected_device and x.device != expected_device:
        raise ValueError(
            f"Invalid device of {name}: expected {expected_device}, got {x.device}"
        )


@functools.cache
def get_logging_module():
    return gen_spdlog_module().build_and_load()


class LogLevel(Enum):
    TRACE = 0
    DEBUG = 1
    INFO = 2
    WARN = 3
    ERROR = 4
    CRITICAL = 5


log_level_map = {
    "trace": LogLevel.TRACE,
    "debug": LogLevel.DEBUG,
    "info": LogLevel.INFO,
    "warn": LogLevel.WARN,
    "error": LogLevel.ERROR,
    "critical": LogLevel.CRITICAL,
}


def set_log_level(lvl_str: str) -> None:
    get_logging_module().set_log_level(log_level_map[lvl_str].value)


@functools.cache
def device_support_pdl(device: torch.device) -> bool:
    if device.type != "cuda":
        return False
    major, _ = get_compute_capability(device)
    return major >= 9


def ceil_div(x: int, y: int) -> int:
    """
    Perform ceiling division of two integers.

    Args:
        x: the dividend.
        y: the divisor.

    Returns:
        The result of the ceiling division.
    """
    return (x + y - 1) // y


def round_up(x: int, y: int) -> int:
    """Round up x to the nearest multiple of y"""
    return ceil_div(x, y) * y


@functools.cache
def get_device_sm_count(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


class FP4Tensor:
    """Wrapper class for FP4 tensors.

    Since PyTorch doesn't natively support FP4, this wrapper contains:
    - data: uint8 tensor storing the compressed FP4 data, the size of innermost dimension is ceil(original_dim / 2) since each uint8 stores 2 FP4 values
    - scale: float8_e4m3fn tensor storing the scale factors
    """

    def __init__(
        self,
        data: torch.Tensor,
        scale: torch.Tensor,
        scale_start_index: int = 0,
        original_shape: Optional[Tuple[int, ...]] = None,
    ):
        """Initialize FP4Tensor.

        Parameters
        ----------
        data : torch.Tensor
            uint8 tensor storing the compressed FP4 data
        scale : torch.Tensor
            float8_e4m3fn tensor storing the scale factors
        scale_start_index : int
            The start token index of the scale factors. This is needed when two kernels (like prefill and decode kernels) are reusing the same scale factor tensor with different offsets.
        original_shape : Optional[Tuple[int, ...]]
            The original shape before compression.
        """
        if data.dtype != torch.uint8:
            raise ValueError(f"data must be uint8 tensor, got {data.dtype}")

        # Validate scale factor tensor and scale start index
        if scale.dtype != torch.float8_e4m3fn:
            raise ValueError(f"scale must be float8_e4m3fn tensor, got {scale.dtype}")
        if scale.shape[0] % 128 != 0:
            raise ValueError(
                f"scale.shape[0] must be a multiple of 128, got {scale.shape[0]}"
            )
        if scale_start_index < 0 or scale_start_index >= scale.shape[0]:
            raise ValueError(
                f"scale start index must be in the range [0, scale.shape[0]). "
                f"scale_start_index={scale_start_index}, scale.shape[0]={scale.shape[0]}"
            )
        if scale_start_index + data.shape[0] > scale.shape[0]:
            raise ValueError(
                f"scale start index + data.shape[0] must not exceed scale.shape[0]. "
                f"scale_start_index={scale_start_index}, data.shape[0]={data.shape[0]}, scale.shape[0]={scale.shape[0]}"
            )

        # Validate shape relationship if original_shape is provided
        if original_shape is not None:
            if data.shape[:-1] != original_shape[:-1]:
                raise ValueError(
                    f"data and original_shape must have the same dimensions except the last one. "
                    f"data.shape={data.shape}, original_shape={original_shape}"
                )

            # Check the last dimension relationship: data_dim = ceil(original_dim / 2)
            expected_data_dim = math.ceil(original_shape[-1] / 2)
            if data.shape[-1] != expected_data_dim:
                raise ValueError(
                    f"data last dimension must be ceil(original_shape[-1] / 2). "
                    f"data.shape[-1]={data.shape[-1]}, original_shape[-1]={original_shape[-1]}, "
                    f"expected={expected_data_dim}"
                )

        self.data = data
        self.scale = scale
        self.scale_start_index = scale_start_index
        self.original_shape = original_shape
        self.dtype = "nvfp4"


# yapf: disable
srcToDstBlk16RowMap = [
    0,  8,
    1,  9,
    2, 10,
    3, 11,
    4, 12,
    5, 13,
    6, 14,
    7, 15
]

srcToDstBlk32RowMap = [
    0,  8, 16, 24,
    1,  9, 17, 25,
    2, 10, 18, 26,
    3, 11, 19, 27,
    4, 12, 20, 28,
    5, 13, 21, 29,
    6, 14, 22, 30,
    7, 15, 23, 31
]
# yapf: enable


def get_shuffle_block_size(epilogue_tile_m: int) -> int:
    shuffle_block_size = 16
    if epilogue_tile_m % 128 == 0:
        shuffle_block_size = 32
    return shuffle_block_size


def get_shuffle_matrix_a_row_indices(
    input_tensor: torch.Tensor, epilogue_tile_m: int
) -> torch.Tensor:
    """
    Higher-level PyTorch approach to reorder the rows in blocks of size 16 or 32.
    - We do NOT try to handle custom e2m1 memory usage (i.e. no 'K/2' bytes).
    - Instead, we purely reorder rows in a standard PyTorch shape [M, K].
    """
    assert input_tensor.dim() == 2, (
        f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"
    )

    # M, K from the input
    M, K = input_tensor.shape

    # Choose block size 16 or 32
    shuffle_block_size = get_shuffle_block_size(epilogue_tile_m)
    row_map = srcToDstBlk16RowMap if shuffle_block_size == 16 else srcToDstBlk32RowMap

    assert M % shuffle_block_size == 0, (
        f"input_tensor.shape[0] must be multiples of {shuffle_block_size}"
    )

    # row_indices[new_row] = old_row
    # so row_indices is an array of size M telling us from which old_row
    # the new_row should be taken.
    row_indices = torch.empty(M, dtype=torch.long)

    for old_row in range(M):
        block_idx = old_row // shuffle_block_size
        row_in_block = old_row % shuffle_block_size
        mapped_row_in_block = row_map[row_in_block]

        new_row = block_idx * shuffle_block_size + mapped_row_in_block

        row_indices[new_row] = old_row

    return row_indices


def get_shuffle_matrix_sf_a_row_indices(
    input_tensor: torch.Tensor, epilogue_tile_m: int, num_elts_per_sf: int = 16
) -> torch.Tensor:
    assert input_tensor.dtype == torch.uint8 or input_tensor.dtype == torch.bfloat16
    assert num_elts_per_sf == 16 or num_elts_per_sf == 32

    assert input_tensor.dim() == 2, (
        f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"
    )

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    row_indices = get_shuffle_matrix_a_row_indices(input_tensor, epilogue_tile_m)

    return row_indices


def get_native_fp4_dtype():
    """get native fp4 datatype if supported in Torch, otherwise return uint8."""
    if hasattr(torch, "float4_e2m1fn_x2"):
        return torch.float4_e2m1fn_x2
    else:
        return torch.uint8


def supported_compute_capability(supported_ccs: Iterable[int]) -> Callable:
    """
    Decorator to mark functions with their supported CUDA compute capabilities.

    This decorator annotates a function with metadata about which CUDA compute
    capabilities (CC) it supports. It adds a `_supported_ccs` attribute containing
    the set of supported compute capabilities and an `is_compute_capability_supported`
    method to check if a specific compute capability is supported.

    Parameters
    ----------
    supported_ccs : list or iterable of int
        A list of supported CUDA compute capability versions as integers
        (e.g., [75, 80, 86, 89, 90, 100, 103, 110, 120]).
        These are computed as major * 10 + minor (e.g., SM 8.0 = 80, SM 9.0 = 90).

    Returns
    -------
    decorator : callable
        A decorator function that adds compute capability metadata to the decorated function.

    Attributes Added to Decorated Function
    ---------------------------------------
    _supported_ccs : set of int
        A set of integers representing the supported compute capabilities.
    is_compute_capability_supported : callable
        A method that takes a compute capability (int) and returns True if it's
        supported, False otherwise.

    Examples
    --------
    >>> @supported_compute_capability([80, 86, 89, 90])
    ... def my_kernel_function():
    ...     pass
    ...
    >>> my_kernel_function._supported_ccs
    {80, 86, 89, 90}
    >>> my_kernel_function.is_compute_capability_supported(80)
    True
    >>> my_kernel_function.is_compute_capability_supported(75)
    False

    Notes
    -----
    This decorator is useful in conjunction with the backend_requirement decorator to mark functions with their supported CUDA compute capabilities.

    Raises
    ------
    TypeError
        If supported_ccs is not iterable or contains non-integer values.
    """
    # Validate that supported_ccs is iterable
    try:
        ccs_list = list(supported_ccs)
    except TypeError:
        raise TypeError(
            f"supported_ccs must be an iterable, got {type(supported_ccs).__name__}"
        ) from None

    # Validate and convert all elements to integers
    validated_ccs = []
    for i, cc in enumerate(ccs_list):
        if isinstance(cc, bool):
            # Reject booleans (which are technically ints in Python)
            raise TypeError(f"supported_ccs[{i}] must be an integer, got bool: {cc}")
        if not isinstance(cc, int):
            raise TypeError(
                f"supported_ccs[{i}] must be an integer, got {type(cc).__name__}: {cc}"
            )
        validated_ccs.append(cc)

    def decorator(func):
        func._supported_ccs = set(validated_ccs)

        def is_cc_supported(cc):
            return cc in func._supported_ccs

        func.is_compute_capability_supported = is_cc_supported
        return func

    return decorator


def backend_requirement(
    backend_checks: Dict[str, Callable],
    common_check: Optional[Callable] = None,
    heuristic_func: Optional[Callable] = None,
) -> Callable:
    """
    Decorator to enforce backend and problem size requirements for kernel functions.

    This decorator validates that a function is called with a supported backend and
    compute capability, and optionally validates problem size constraints. It performs
    runtime checks before executing the function and raises appropriate errors if
    requirements are not met. If checking overheads are a concern, you can pass a
    `skip_check` keyword argument to the function to bypass the validation.

    Parameters
    ----------
    backend_checks : dict
        A dictionary mapping backend names (str) to requirement checker functions.
        Each checker function should accept the same arguments as the decorated function
        and return True if the problem size is supported, False otherwise.
        Checkers can be decorated with @supported_compute_capability to specify
        which compute capabilities they support.
    common_check : callable, optional
        An optional function that performs additional validation checks common to all
        backends. Should accept the same arguments as the decorated function and return
        True if requirements are met, False otherwise.
        In the case where the kernel function does not have any specific backends, this can be decorated with @supported_compute_capability to specify the function's supported compute capabilities.
    heuristic_func : callable, optional
        A function that performs heuristic backend selection when backend is "auto".
        Must be provided if backend is "auto". Does not do anything if backend is not "auto".
        Should accept the same arguments as the decorated function.
        Should return an ordered list of runnable backends with the most preferred backend first.
        When decorated function is not autotuned, the first backend in the heuristic list will be run.
        When decorated function is autotuned, the backends in the heuristic list will be autotuned over to find the best backend.

    Returns
    -------
    decorator : callable
        A decorator function that wraps the target function with validation logic, and inserts
        the "skip_check" keyword argument to the function.

    Attributes Added to Decorated Function
    ---------------------------------------
    is_backend_supported : callable
        Method with signature `is_backend_supported(backend, cc=None)` that returns
        True if the specified backend is supported, optionally for a specific compute
        capability (cc).
    is_compute_capability_supported : callable
        Method with signature `is_compute_capability_supported(cc)` that returns True
        if any backend supports the given compute capability.

    Keyword Arguments Added to Decorated Function
    ---------------------------------------------
    skip_check : bool
        (Defaults to False)
        If True, the function will not be validated. This is useful for performance-critical code paths.

    Raises
    ------
    BackendSupportedError
        If the function is called with an unsupported backend or compute capability.
    ValueError
        If the problem size is not supported for the given backend.

    Examples
    --------
    >>> @supported_compute_capability([80, 86, 89, 90])
    ... def _cutlass_check(q, k, v, backend):
    ...     # Validate problem size constraints for CUTLASS backend
    ...     return q.shape[-1] <= 256
    ...
    >>> @supported_compute_capability([75, 80, 86, 89, 90])
    ... def _cudnn_check(q, k, v, backend):
    ...     # Validate problem size constraints for cuDNN backend
    ...     return True
    ...
    >>> @backend_requirement({
    ...     "cutlass": _cutlass_check,
    ...     "cudnn": _cudnn_check
    ... })
    ... def my_attention_kernel(q, k, v, backend="cutlass"):
    ...     # Backend invocation
    ...     pass
    ...
    >>> # Example with kernel function with no backend requirements
    >>> @supported_compute_capability([80, 86, 89, 90])
    ... def _common_size_check(q, k, v):
    ...     return True
    ...
    >>> @backend_requirement(
    ...     backend_checks={}, # Empty backend_checks
    ...     common_check=_common_size_check
    ... )
    ... def backend_agnostic_kernel(q, k, v):
    ...     pass

    Notes
    -----
    - The decorator automatically extracts compute capability from tensor arguments
      by finding the first torch.Tensor in args or kwargs.
    - A `skip_check=True` keyword argument can be passed to bypass validation for
      performance-critical code paths.
    - All validation is performed before the wrapped function executes.
    - Works in conjunction with the @supported_compute_capability decorator to
      provide fine-grained control over backend and architecture support.
    """

    def decorator(func):
        # Get the function signature once for reuse
        sig = inspect.signature(func)

        def is_backend_supported(backend, cc=None):
            # No backend-specific checks
            if not has_backend_choices():
                raise ValueError(
                    f"Invalid is_backend_supported call: no backend choices for {func.__name__}"
                )
            else:
                # Is this backend present?
                if backend not in backend_checks:
                    return False
                req_checker = backend_checks[backend]
                # If user just wants to check if the backend is supported (regardless of compute capability), return True
                if cc is None:
                    return True
                # Check compute capability support via attribute on requirement function
                elif hasattr(req_checker, "is_compute_capability_supported"):
                    return req_checker.is_compute_capability_supported(cc)
                return False

        def is_compute_capability_supported(cc):
            # In case there is only 1 implicit backend, the compute capability support needs to be added to the common check
            if not has_backend_choices():
                # No backend-specific checks, only check common_check
                if not hasattr(common_check, "is_compute_capability_supported"):
                    raise ValueError(
                        f"Invalid is_compute_capability_supported call: {common_check.__name__} does not have is_compute_capability_supported decorator"
                    )
                return common_check.is_compute_capability_supported(cc)
            else:
                # True if any backend requirement supports this cc
                return any(
                    hasattr(checker, "is_compute_capability_supported")
                    and checker.is_compute_capability_supported(cc)
                    for checker in backend_checks.values()
                )

        # @note: this function does not automatically apply defaults to the arguments.
        def _is_problem_size_supported(*args, **kwargs):
            # At this point, kwargs should have defaults applied, so backend should be present
            backend = kwargs.get("backend")

            # Handle empty backend_checks case
            if not has_backend_choices():
                return common_check(*args, **kwargs)

            if backend not in backend_checks:
                raise BackendSupportedError(
                    f"Backend '{backend}' is not supported for {func.__name__}"
                )
            req_checker = backend_checks[backend]
            if common_check is not None:
                return common_check(*args, **kwargs) and req_checker(*args, **kwargs)
            else:
                return req_checker(*args, **kwargs)

        def has_backend_choices() -> bool:
            # Whether there are any backend choices to make
            return bool(backend_checks)

        def has_backend(backend: str) -> bool:
            # Whether the given backend exists in the API
            return backend in backend_checks

        def suitable_auto_backends(cc, *args, **kwargs):
            if common_check is not None and not common_check(*args, **kwargs):
                return False
            suitable_backends = []
            # Check for each backend support
            for backend in backend_checks:
                req_checker = backend_checks[backend]
                try:
                    if req_checker(
                        *args, **kwargs
                    ) and req_checker.is_compute_capability_supported(cc):
                        suitable_backends.append(backend)
                except ValueError:
                    continue
            # If a heuristic function is provided, filter the suitable backends based on the heuristic function
            assert heuristic_func is not None, "Heuristic function must be provided"
            suitable_backends = heuristic_func(suitable_backends, *args, **kwargs)
            if not suitable_backends:
                return False
            wrapper.suitable_auto_backends = suitable_backends
            return True

        def _get_capability(*args, **kwargs):
            capability = None
            # Find the first tensor argument.
            # Assume all tensors are on the same device/capability.
            # We could consider check all tensors at a performance cost.
            tensor_arg = None
            all_args = args + tuple(kwargs.values())
            for value in all_args:
                if isinstance(value, torch.Tensor):
                    tensor_arg = value
                    break

            if tensor_arg is not None:
                # Get compute capability from the first tensor
                # Assume all tensors are on the same device/capability
                major, minor = get_compute_capability(tensor_arg.device)
                capability = major * 10 + minor
            return capability

        # @brief: Wrapper function that calls the orignal, decorated function, after applying a number of checks.
        # @note that here we manually apply defaults to the arguments in the wrapper function when doing validation.
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # skip_check is an optional argument that the decorator adds to any API function.
            # It prevents the performance overhead of checking.
            skip_check = kwargs.pop("skip_check", False)

            if not skip_check:
                # Apply defaults from the function signature for validation
                # This ensures that all parameters (including backend) have their default values
                # if not explicitly provided by the caller
                bound_args = sig.bind(*args, **kwargs)
                bound_args.apply_defaults()
                # Convert to kwargs for validation functions
                kwargs_with_defaults = dict(bound_args.arguments)
                backend = kwargs_with_defaults.get("backend")
                capability = _get_capability(*args, **kwargs)
                if not has_backend_choices() and common_check is None:
                    raise ValueError(
                        f"Invalid @backend_requirement decorator usage: no backend choices and no common_check for {func.__name__}"
                    )

                if has_backend_choices():
                    if backend == "auto":
                        if not suitable_auto_backends(
                            capability, **kwargs_with_defaults
                        ):
                            raise BackendSupportedError(
                                f"No suitable auto backends found for {func.__name__}"
                            )
                    else:
                        if not is_backend_supported(backend, capability):
                            extra = (
                                f" with capability {capability}" if capability else ""
                            )
                            raise BackendSupportedError(
                                f"{func.__name__} does not support backend '{backend}'{extra}"
                            )
                        if not _is_problem_size_supported(**kwargs_with_defaults):
                            raise ValueError(
                                f"Problem size is not supported for {func.__name__}"
                            )
                else:
                    # If the function doesnt have backends (i.e., there is only 1, implicit backend), run the following checks.
                    if not is_compute_capability_supported(capability):
                        raise BackendSupportedError(
                            f"{func.__name__} does not support compute capability {capability}"
                        )
                    if not _is_problem_size_supported(**kwargs_with_defaults):
                        raise ValueError(
                            f"Problem size is not supported for {func.__name__}"
                        )
            elif skip_check and heuristic_func is not None:
                if kwargs.get("backend") == "auto":
                    # This needs to be called for heuristic function
                    capability = _get_capability(*args, **kwargs)
                    suitable_auto_backends(capability, *args, **kwargs)

            return func(*args, **kwargs)

        wrapper.is_backend_supported = is_backend_supported
        wrapper.is_compute_capability_supported = is_compute_capability_supported
        wrapper.has_backend = has_backend
        wrapper.has_backend_choices = has_backend_choices
        return wrapper

    return decorator
