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
import os
from enum import Enum
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.version
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

from .jit import gen_jit_spec, env as jit_env

IS_BUILDING_DOCS = os.environ.get("FLASHINFER_BUILDING_DOCS") == "1"


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


def _get_cache_buf(name: str, bytes: int, device: torch.device) -> torch.Tensor:
    key = (name, device)
    buf = _cache_buf.get(key)
    if buf is None:
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


if IS_BUILDING_DOCS or TorchVersion(torch_version) < TorchVersion("2.4"):

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


def is_sm90a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 9 and version_at_least(torch.version.cuda, "12.3")


def is_sm100a_supported(device: torch.device) -> bool:
    major, _ = get_compute_capability(device)
    return major == 10 and version_at_least(torch.version.cuda, "12.8")


def determine_mla_backend(device: torch.device) -> str:
    return "fa3" if is_sm90a_supported(device) else "fa2"


def _check_shape_dtype_device(
    x: torch.Tensor,
    expected_shape: Sequence[int],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
    name: str,
) -> None:
    if x.shape != torch.Size(expected_shape):
        raise ValueError(
            f"Invalid shape of {name}: expected {expected_shape}, got {x.shape}"
        )
    if x.dtype != expected_dtype:
        raise ValueError(
            f"Invalid dtype of {name}: expected {expected_dtype}, got {x.dtype}"
        )
    if x.device != expected_device:
        raise ValueError(
            f"Invalid device of {name}: expected {expected_device}, got {x.device}"
        )


def get_logging_module():
    return gen_jit_spec(
        "logging",
        [
            jit_env.FLASHINFER_CSRC_DIR / "logging.cc",
        ],
        extra_include_paths=[
            jit_env.SPDLOG_INCLUDE_DIR,
            jit_env.FLASHINFER_INCLUDE_DIR,
        ],
    ).build_and_load()


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


def device_support_pdl(device: torch.device) -> bool:
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
    assert input_tensor.dtype == torch.uint8
    assert num_elts_per_sf == 16

    assert input_tensor.dim() == 2, (
        f"input_tensor should be a 2D tensor, not {input_tensor.dim()}"
    )

    # M, K from the input
    M, K = input_tensor.shape
    assert M % 128 == 0
    assert K % 4 == 0

    row_indices = get_shuffle_matrix_a_row_indices(input_tensor, epilogue_tile_m)

    return row_indices
