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

import math
import os
from enum import Enum
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import torch
import torch.version
from torch.torch_version import TorchVersion
from torch.torch_version import __version__ as torch_version

IS_BUILDING_DOCS = os.environ.get("FLASHINFER_BUILDING_DOCS") == "1"


class PosEncodingMode(Enum):
    NONE = 0
    ROPE_LLAMA = 1
    ALIBI = 2


class MaskMode(Enum):
    NON_CAUSAL = 0
    CAUSAL = 1
    CUSTOM = 2


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
        buf = (get_alibi_slopes(num_qo_heads) * log2e).to(device)
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


def get_cuda_stream(device: torch.device) -> int:
    return torch.cuda.current_stream(device).cuda_stream


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
    # NOTE: currently fp8 is not supported in our FA3 backend
    # will add support soon
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
    major, _ = get_compute_capability(device)

    if (
        major == 9
        and torch.version.cuda >= "12.3"
        and is_fa3_backend_supported(
            pos_encoding_mode,
            use_fp16_qk_reductions,
            use_custom_mask,
            dtype_q,
            dtype_kv,
        )
    ):
        return "fa3"
    else:
        return "fa2"
