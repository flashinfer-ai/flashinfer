# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Argument checks for the PrimTS decode frontends and their launch bounds."""

import numbers
import torch
from ...common.attention_args import validate_positive_int
from ...common.scales import host_scale_to_float32
from ...common.tensors import (
    MAX_INT32,
    attention_dtype_key,
    validate_16byte_alignment,
    validate_exact_compact_strides,
)
from typing import Optional
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .kernel.config import FmhaDecodeConfig


GENERIC_HEAD_DIMS = (64, 128, 192, 256)
SUPPORTED_HEAD_DIMS = (*GENERIC_HEAD_DIMS, 512)
# Decode K/V masks form an exclusive tile endpoint as
# ``tile_offset_k + tile_size_kv`` in signed Int32.  Public policies use at
# most a 256-token K/V tile, so reserve its full 255-token padded tail.
DECODE_MAX_KV_TILE_SIZE = 256
DECODE_MAX_KV_LEN = MAX_INT32 - (DECODE_MAX_KV_TILE_SIZE - 1)
SUPPORTED_INPUT_DTYPES = (
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
)
MAX_HEAD_RATIO = 128


def decode_output_shape(
    *,
    batch_size: int,
    num_qo_heads: int,
    seq_len_q: int,
    head_dim: int,
    total_q_tokens: Optional[int] = None,
) -> tuple[int, ...]:
    if total_q_tokens is not None:
        return (total_q_tokens, num_qo_heads, head_dim)
    if seq_len_q == 1:
        return (batch_size, num_qo_heads, head_dim)
    return (batch_size, seq_len_q, num_qo_heads, head_dim)


def read_packed_q_plan_metadata(
    qo_indptr: torch.Tensor,
) -> tuple[int, int, tuple[int, ...]]:
    """Validate Q offsets at plan time and return max, total, and row lengths."""

    offsets = tuple(int(value) for value in qo_indptr.tolist())
    if offsets[0] != 0:
        raise ValueError("qo_indptr must start at zero")
    q_lengths = tuple(
        end - begin for begin, end in zip(offsets[:-1], offsets[1:], strict=True)
    )
    if any(length <= 0 for length in q_lengths):
        raise ValueError("qo_indptr must be strictly increasing")
    exact_max_seq_len_q = max(q_lengths, default=0)
    if exact_max_seq_len_q <= 0:
        raise ValueError("a packed-Q plan must contain at least one query token")
    return exact_max_seq_len_q, offsets[-1], q_lengths


def resolve_q_mode(
    *,
    seq_len_q: int,
    qo_indptr: Optional[torch.Tensor],
    max_q_len: Optional[int],
    require_packed_max: bool,
    validate: bool = True,
) -> tuple[bool, Optional[int]]:
    """Resolve fixed versus packed Q without exposing an internal mode knob."""

    if validate:
        seq_len_q = validate_seq_len_q(seq_len_q)
    if qo_indptr is None:
        if max_q_len is not None:
            if validate:
                max_q_len = validate_seq_len_q(max_q_len)
            if max_q_len != seq_len_q:
                raise ValueError(
                    "fixed seq_len_q and max_q_len must agree: "
                    f"got {seq_len_q} and {max_q_len}"
                )
        return False, seq_len_q
    if max_q_len is None and seq_len_q != 1:
        # Preserve the legacy name as a packed static-bound alias.  The
        # nullable qo_indptr alone still selects fixed versus packed storage.
        return True, seq_len_q
    if max_q_len is None:
        if require_packed_max:
            raise ValueError(
                "max_q_len is required with qo_indptr for the standalone "
                "workspace/JIT interface"
            )
        return True, None
    if validate:
        max_q_len = validate_seq_len_q(max_q_len)
    if seq_len_q != 1 and seq_len_q != max_q_len:
        raise ValueError(
            "seq_len_q and max_q_len must agree when both provide the "
            f"packed static bound: got {seq_len_q} and {max_q_len}"
        )
    return True, max_q_len


def validate_decode_policy_kv_tile_size(config: "FmhaDecodeConfig") -> None:
    """Keep the public K/V bound coupled to generated decode policies."""

    tile_size_kv = int(config.tile_size_kv)
    if tile_size_kv > DECODE_MAX_KV_TILE_SIZE:
        raise RuntimeError(
            "FMHA decode Int32 extent safety assumes a K/V tile no larger "
            f"than {DECODE_MAX_KV_TILE_SIZE}, got {tile_size_kv}"
        )


def validate_decode_query_head_extent(
    *,
    batch_size: int,
    num_qo_heads: int,
    max_q_len: int,
) -> None:
    """Keep every fixed-capacity or packed Q/head coordinate in Int32."""

    batch_size = validate_positive_int(batch_size, "batch_size")
    num_qo_heads = validate_positive_int(num_qo_heads, "num_qo_heads")
    max_q_len = validate_seq_len_q(max_q_len)
    extent = batch_size * max_q_len * num_qo_heads
    if extent > MAX_INT32:
        raise NotImplementedError(
            "batch_size * max_q_len * num_qo_heads must fit in a signed int32"
        )


def validate_dtype_pair(
    q_dtype: torch.dtype,
    k_dtype: torch.dtype,
    v_dtype: torch.dtype,
    out_dtype: torch.dtype,
) -> None:
    attention_dtype_key(q_dtype)
    attention_dtype_key(k_dtype)
    attention_dtype_key(v_dtype)
    attention_dtype_key(out_dtype)
    matching_kv = k_dtype == v_dtype
    supported = (
        (
            q_dtype == torch.float16
            and k_dtype == v_dtype == torch.float16
            and out_dtype == torch.float16
        )
        or (
            q_dtype == torch.bfloat16
            and out_dtype == torch.bfloat16
            and (
                (
                    matching_kv
                    and k_dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.uint8)
                )
                or (k_dtype == torch.bfloat16 and v_dtype == torch.float8_e4m3fn)
            )
        )
        or (
            q_dtype == torch.float8_e4m3fn
            and matching_kv
            and k_dtype in (torch.float8_e4m3fn, torch.uint8)
            and out_dtype in (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
        )
    )
    if not supported:
        raise NotImplementedError(
            "attention-ts decode supports FP16->FP16, BF16->BF16, "
            "FP8-E4M3->FP16/BF16/FP8-E4M3; "
            "BF16 Q + FP8 K/V, BF16/FP8-E4M3 Q + NVFP4 K/V, "
            "and BF16 Q/K + FP8 V; got "
            f"Q_{q_dtype}_K_{k_dtype}_V_{v_dtype}->{out_dtype}"
        )


def validate_head_dim(head_dim: int) -> int:
    head_dim = validate_positive_int(head_dim, "head_dim")
    if head_dim not in SUPPORTED_HEAD_DIMS:
        raise ValueError(
            "attention-ts decode requires head_dim in "
            f"{SUPPORTED_HEAD_DIMS}, got {head_dim}"
        )
    return head_dim


def validate_head_geometry(num_qo_heads: int, num_kv_heads: int) -> None:
    num_qo_heads = validate_positive_int(num_qo_heads, "num_qo_heads")
    num_kv_heads = validate_positive_int(num_kv_heads, "num_kv_heads")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            "num_qo_heads must be divisible by num_kv_heads, got "
            f"{num_qo_heads} and {num_kv_heads}"
        )
    head_ratio = num_qo_heads // num_kv_heads
    if head_ratio > MAX_HEAD_RATIO:
        raise ValueError(
            "attention-ts decode requires "
            f"1 <= Hq/Hkv <= {MAX_HEAD_RATIO}, got {head_ratio}"
        )


def validate_max_kv_len(value: int, name: str) -> int:
    """Reserve the largest padded decode K/V tile in signed Int32."""

    value = validate_positive_int(value, name)
    if value > DECODE_MAX_KV_LEN:
        raise NotImplementedError(
            f"{name} must be <= {DECODE_MAX_KV_LEN} so padded FMHA decode "
            "K/V coordinates fit in a signed int32"
        )
    return value


def validate_out(
    out: torch.Tensor,
    *,
    q: torch.Tensor,
    expected_shape: tuple[int, ...],
    seq_len_q: int,
    packed_query: bool,
    out_dtype: torch.dtype,
) -> None:
    if not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor")
    if tuple(out.shape) != expected_shape:
        raise ValueError(
            f"out must have shape {expected_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != out_dtype:
        raise ValueError(f"out must have dtype {out_dtype}, got {out.dtype}")
    if out.device != q.device:
        raise ValueError(f"out must be on {q.device}, got {out.device}")
    layout = (
        "[total_q, Hq, D]"
        if packed_query
        else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
    )
    validate_exact_compact_strides(out, "out", layout)
    validate_16byte_alignment(out, "out")


def validate_packed_q_plan_values(
    qo_indptr: torch.Tensor,
    *,
    max_q_len: int,
    expected_total_q: Optional[int] = None,
) -> tuple[int, int]:
    """Synchronize once to validate the packed-Q values against a static bound."""

    derived_max_seq_len_q, total_q, _ = read_packed_q_plan_metadata(qo_indptr)
    if derived_max_seq_len_q > max_q_len:
        raise ValueError(
            "qo_indptr contains a per-request Q length larger than "
            f"max_q_len ({max_q_len}): got {derived_max_seq_len_q}"
        )
    if expected_total_q is not None and total_q != expected_total_q:
        raise ValueError(
            "the final qo_indptr offset must equal the packed q token count: "
            f"expected {expected_total_q}, got {total_q}"
        )
    return derived_max_seq_len_q, total_q


def validate_q(
    q: torch.Tensor,
    *,
    seq_len_q: int = 1,
    packed_query: bool = False,
    device: Optional[torch.device] = None,
    batch_size: Optional[int] = None,
    num_qo_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
    q_dtype: Optional[torch.dtype] = None,
) -> None:
    seq_len_q = validate_seq_len_q(seq_len_q)
    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    expected_rank = 3 if packed_query or seq_len_q == 1 else 4
    if q.ndim != expected_rank:
        expected_layout = (
            "[total_q, Hq, D]"
            if packed_query
            else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
        )
        raise ValueError(f"q must have shape {expected_layout} for this Q layout")
    if not packed_query and seq_len_q > 1 and q.shape[1] != seq_len_q:
        raise ValueError(
            f"q sequence length must match seq_len_q ({seq_len_q}), got {q.shape[1]}"
        )
    num_heads = int(q.shape[-2])
    if q.shape[0] <= 0 or num_heads <= 0:
        leading_name = "total Q token count" if packed_query else "batch size"
        raise ValueError(f"q {leading_name} and head count must be positive")
    validate_head_dim(int(q.shape[-1]))
    if q.dtype not in SUPPORTED_INPUT_DTYPES:
        raise NotImplementedError(f"unsupported attention-ts q dtype {q.dtype}")
    if q.device.type != "cuda":
        raise ValueError("q must be a CUDA tensor")
    if device is not None and q.device != device:
        raise ValueError(f"q must be on planned device {device}, got {q.device}")
    if not packed_query and batch_size is not None and q.shape[0] != batch_size:
        raise ValueError(
            f"q batch size must match the plan ({batch_size}), got {q.shape[0]}"
        )
    if packed_query and batch_size is not None:
        total_q = int(q.shape[0])
        max_total_q = batch_size * seq_len_q
        if total_q < batch_size or total_q > max_total_q:
            raise ValueError(
                "packed q token count must be within "
                f"[{batch_size}, {max_total_q}], got {total_q}"
            )
    if num_qo_heads is not None and num_heads != num_qo_heads:
        raise ValueError(
            f"q head count must match the plan ({num_qo_heads}), got {num_heads}"
        )
    if head_dim is not None and q.shape[-1] != head_dim:
        raise ValueError(
            f"q head dimension must match the plan ({head_dim}), got {q.shape[-1]}"
        )
    if q_dtype is not None and q.dtype != q_dtype:
        raise ValueError(f"q dtype must match the plan ({q_dtype}), got {q.dtype}")
    layout = (
        "[total_q, Hq, D]"
        if packed_query
        else ("[B, Hq, D]" if seq_len_q == 1 else "[B, SQ, Hq, D]")
    )
    validate_exact_compact_strides(q, "q", layout)
    validate_16byte_alignment(q, "q")


def validate_qo_indptr(
    qo_indptr: torch.Tensor,
    *,
    expected_device: torch.device,
    batch_size: int,
) -> None:
    """Validate packed-Q metadata without synchronizing device values."""

    if not isinstance(qo_indptr, torch.Tensor):
        raise TypeError("qo_indptr must be a torch.Tensor")
    if qo_indptr.ndim != 1:
        raise ValueError("qo_indptr must be one-dimensional")
    if qo_indptr.dtype != torch.int32:
        raise TypeError("qo_indptr must have dtype torch.int32")
    if qo_indptr.device != expected_device:
        raise ValueError(
            f"qo_indptr must be on {expected_device}, got {qo_indptr.device}"
        )
    if qo_indptr.numel() != batch_size + 1:
        raise ValueError(
            "qo_indptr must have B + 1 elements: expected "
            f"{batch_size + 1}, got {qo_indptr.numel()}"
        )
    if not qo_indptr.is_contiguous():
        raise ValueError("qo_indptr must be contiguous")
    if qo_indptr.data_ptr() % 4 != 0:
        raise ValueError("qo_indptr data pointer must be 4-byte aligned")


def validate_scale(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a positive Python scalar")
    return host_scale_to_float32(value, name)


def validate_seq_len_q(seq_len_q: int) -> int:
    return validate_positive_int(seq_len_q, "seq_len_q")


def validate_storage_page_size(page_size: int, storage_page_size: int) -> int:
    """Require physical cache pages to contain whole logical fragments."""
    storage_page_size = validate_positive_int(storage_page_size, "storage_page_size")
    if storage_page_size % page_size != 0:
        raise ValueError("storage_page_size must be divisible by page_size")
    return storage_page_size


def validate_window_left(window_left: int, mask_type: str) -> int:
    if isinstance(window_left, bool) or not isinstance(window_left, int):
        raise TypeError("window_left must be an integer")
    if window_left < -1:
        raise ValueError("window_left must be -1 (disabled) or non-negative")
    if window_left > 2**31 - 2:
        raise ValueError("window_left must be no larger than 2**31 - 2")
    if window_left >= 0 and mask_type != "causal":
        raise ValueError("window_left requires mask_type='causal'")
    return window_left
