"""
Copyright (c) 2026 by FlashInfer team.

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
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch

from flashinfer.api_logging import flashinfer_api


@functools.cache
def _load_flex_flash_attn_func():
    try:
        from magi_attention.api import flex_flash_attn_func
    except ImportError as exc:
        raise ImportError(
            "MagiAttention is required to use FlexFlashAttentionWrapper. "
            "Install MagiAttention in the active Python environment before "
            "calling run()."
        ) from exc
    return flex_flash_attn_func


def _check_range_tensor(name: str, value: Optional[torch.Tensor]) -> None:
    if value is None:
        return
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.dtype != torch.int32:
        raise TypeError(f"{name} must have dtype torch.int32")
    if value.ndim != 2 or value.shape[1] != 2:
        raise ValueError(f"{name} must have shape (num_ranges, 2)")


def _check_attn_type_map(
    attn_type_map: Optional[torch.Tensor],
    num_ranges: Optional[int],
) -> None:
    if attn_type_map is None:
        return
    if not isinstance(attn_type_map, torch.Tensor):
        raise TypeError("attn_type_map must be a torch.Tensor")
    if attn_type_map.dtype != torch.int32:
        raise TypeError("attn_type_map must have dtype torch.int32")
    if attn_type_map.ndim != 1:
        raise ValueError("attn_type_map must have shape (num_ranges,)")
    if num_ranges is not None and attn_type_map.numel() != num_ranges:
        raise ValueError("attn_type_map must have the same length as q_ranges")


def _copy_or_return(name: str, src: torch.Tensor, dst: Optional[torch.Tensor]):
    if dst is None:
        return src
    if not isinstance(dst, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if dst.shape != src.shape:
        raise ValueError(f"{name} must have shape {tuple(src.shape)}")
    if dst.dtype != src.dtype:
        raise TypeError(f"{name} must have dtype {src.dtype}")
    if dst.device != src.device:
        raise ValueError(f"{name} must be on device {src.device}")
    dst.copy_(src)
    return dst


class FlexFlashAttentionWrapper:
    r"""Wrapper for MagiAttention single-GPU Flex Flash Attention.

    This wrapper follows FlashInfer's ``plan``/``run`` style while keeping
    MagiAttention as an optional dependency. It accepts MagiAttention's range
    based mask representation directly and is intended for single-GPU
    prefill/full-forward attention, not paged KV decode.
    """

    @flashinfer_api
    def __init__(self) -> None:
        self._planned = False
        self._q_ranges: Optional[torch.Tensor] = None
        self._k_ranges: Optional[torch.Tensor] = None
        self._attn_type_map: Optional[torch.Tensor] = None
        self._sink: Optional[torch.Tensor] = None
        self._sink_layout = "sh"
        self._softmax_scale: Optional[float] = None
        self._softcap = 0.0
        self._deterministic = False
        self._sm_margin = 0
        self._auto_range_merge = False
        self._sparse_load = False
        self._ref_block_size: Optional[Tuple[int, int]] = None
        self._max_seqlen_q: Optional[int] = None
        self._magi_func: Optional[Callable[..., Tuple[torch.Tensor, Any]]] = None
        self._magi_run_kwargs: Dict[str, Any] = {}

    @flashinfer_api
    def plan(
        self,
        q_ranges: Optional[torch.Tensor] = None,
        k_ranges: Optional[torch.Tensor] = None,
        attn_type_map: Optional[torch.Tensor] = None,
        *,
        sink: Optional[torch.Tensor] = None,
        sink_layout: str = "sh",
        softmax_scale: Optional[float] = None,
        softcap: float = 0.0,
        deterministic: bool = False,
        sm_margin: int = 0,
        auto_range_merge: bool = False,
        sparse_load: bool = False,
        ref_block_size: Optional[Tuple[int, int]] = None,
        max_seqlen_q: Optional[int] = None,
    ) -> None:
        r"""Cache MagiAttention mask metadata and execution options."""
        if (q_ranges is None) != (k_ranges is None):
            raise ValueError("q_ranges and k_ranges must be provided together")
        _check_range_tensor("q_ranges", q_ranges)
        _check_range_tensor("k_ranges", k_ranges)
        if q_ranges is not None and k_ranges is not None:
            if q_ranges.shape[0] != k_ranges.shape[0]:
                raise ValueError("q_ranges and k_ranges must have the same length")
            if q_ranges.device != k_ranges.device:
                raise ValueError("q_ranges and k_ranges must be on the same device")
        num_ranges = q_ranges.shape[0] if q_ranges is not None else None
        _check_attn_type_map(attn_type_map, num_ranges)
        if attn_type_map is not None and q_ranges is not None:
            if attn_type_map.device != q_ranges.device:
                raise ValueError("attn_type_map must be on the same device as q_ranges")
        if sink_layout not in ("sh", "ssh"):
            raise ValueError("sink_layout must be either 'sh' or 'ssh'")
        if not isinstance(sm_margin, int):
            raise TypeError("sm_margin must be an integer")
        if sm_margin < 0:
            raise ValueError("sm_margin must be non-negative")

        self._q_ranges = q_ranges
        self._k_ranges = k_ranges
        self._attn_type_map = attn_type_map
        self._sink = sink
        self._sink_layout = sink_layout
        self._softmax_scale = softmax_scale
        self._softcap = softcap
        self._deterministic = deterministic
        self._sm_margin = sm_margin
        self._auto_range_merge = auto_range_merge
        self._sparse_load = sparse_load
        self._ref_block_size = ref_block_size
        self._max_seqlen_q = max_seqlen_q
        magi_run_kwargs = {
            "q_ranges": q_ranges,
            "k_ranges": k_ranges,
            "attn_type_map": attn_type_map,
        }
        if sink is not None:
            magi_run_kwargs["sink"] = sink
        if sink_layout != "sh":
            magi_run_kwargs["sink_layout"] = sink_layout
        if softmax_scale is not None:
            magi_run_kwargs["softmax_scale"] = softmax_scale
        if softcap != 0.0:
            magi_run_kwargs["softcap"] = softcap
        if deterministic:
            magi_run_kwargs["deterministic"] = deterministic
        if sm_margin != 0:
            magi_run_kwargs["sm_margin"] = sm_margin
        if auto_range_merge:
            magi_run_kwargs["auto_range_merge"] = auto_range_merge
        if sparse_load:
            magi_run_kwargs["sparse_load"] = sparse_load
        if ref_block_size is not None:
            magi_run_kwargs["ref_block_size"] = ref_block_size
        if max_seqlen_q is not None:
            magi_run_kwargs["max_seqlen_q"] = max_seqlen_q
        self._magi_run_kwargs = magi_run_kwargs
        self._planned = True

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run MagiAttention Flex Flash Attention on ``q``, ``k``, and ``v``."""
        if not self._planned:
            raise RuntimeError("plan() must be called before run()")
        if lse is not None and not return_lse:
            raise ValueError("return_lse must be True if lse is provided")
        magi_func = self._magi_func
        if magi_func is None:
            magi_func = _load_flex_flash_attn_func()
            self._magi_func = magi_func

        magi_out, meta = magi_func(
            q=q,
            k=k,
            v=v,
            **self._magi_run_kwargs,
        )

        if out is None and not return_lse:
            return magi_out

        result_out = _copy_or_return("out", magi_out, out)
        if return_lse:
            magi_lse = getattr(meta, "lse", None)
            if magi_lse is None:
                raise RuntimeError("MagiAttention did not return lse metadata")
            if lse is None:
                return result_out, magi_lse
            result_lse = _copy_or_return("lse", magi_lse, lse)
            return result_out, result_lse
        return result_out

    @flashinfer_api
    def run_return_lse(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""Run MagiAttention Flex Flash Attention and return LSE metadata."""
        return self.run(q, k, v, out=out, lse=lse, return_lse=True)


__all__ = ["FlexFlashAttentionWrapper"]
