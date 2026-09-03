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

from __future__ import annotations

import math
import threading
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional, Union

import torch

from ..jit.trtllm_mla_blackwell import get_domain_module
from ..jit.cpp_ext import is_cuda_version_at_least
from ..utils import get_compute_capability, get_device_sm_count, log2e

_ALIGNED_HEADS = 128
_ALIGNED_QK_DIM = 576
_ALIGNED_VALUE_DIM = 512
_CLUSTER_SIZE = 2
_MAX_GENERIC_TOKENS = 4096
_WORKSPACE_CACHE_CAPACITY = 8
_HOST_METADATA_CACHE_CAPACITY = 16

_VHALF_ROUTE = "bf16_q128_runtime_vhalf_underfill_sink_one_launch_v1"
_VQUARTER_ROUTE = "bf16_q128_runtime_vquarter_underfill_sink_one_launch_v1"
_UNSPLIT_ROUTE = "bf16_b64_q16_kv1024_unsplit_v1"
_CLC_ROUTE = "bf16_clc_packed_affine_full_tile_mask_bmm2_one_v1"
_GENERIC_BF16_ROUTE = "bf16_full_abi_runtime_tail_one_launch_v1"
_GENERIC_FP8_ROUTE = "fp8_full_abi_runtime_tail_one_launch_v1"
_FP8_P32_QK_L2_ROUTE = "fp8_p32_q2_kv1024_qk_l2_resident_v1"
_FP8_PAGE64_ROUTE = (
    "fp8_page64_native_pdl_sequence_unified_v_leader_consumer_v1"
)
_NATIVE_BF16_ROUTE = "v32_15stage_page_native_sink_pdl_runtime_v4_full_tmem_scrub"

ROUTE_TO_DOMAIN = {
    _VQUARTER_ROUTE: "mla_bf16_vquarter",
    _VHALF_ROUTE: "mla_bf16_vhalf",
    _UNSPLIT_ROUTE: "mla_bf16_unsplit",
    _CLC_ROUTE: "mla_bf16_clc",
    _GENERIC_BF16_ROUTE: "mla_bf16_tail",
    _GENERIC_FP8_ROUTE: "mla_fp8_tail",
    _FP8_P32_QK_L2_ROUTE: "mla_fp8_p32_qk_l2",
    _FP8_PAGE64_ROUTE: "mla_fp8_page64_pdl",
    _NATIVE_BF16_ROUTE: "mla_bf16_native_split8_pdl",
}

_WORKSPACE_CACHE: "OrderedDict[tuple[Any, ...], dict[str, Any]]" = OrderedDict()
_WORKSPACE_CACHE_LOCK = threading.Lock()
_HOST_METADATA_CACHE: "OrderedDict[tuple[Any, ...], tuple[int, ...]]" = OrderedDict()
_HOST_METADATA_CACHE_LOCK = threading.Lock()


@dataclass(frozen=True)
class _BlackwellDispatchMetadata:
    """Host-visible values which determine the exported semantic domain."""

    dtype: torch.dtype
    batch_size: int
    q_len: int
    total_q: int
    q_lens: tuple[int, ...]
    kv_lens: tuple[int, ...]
    num_heads: int
    qk_dim: int
    value_dim: int
    page_size: int
    max_seq_len: int
    topk: int
    table_ndim: int
    num_sms: int
    ragged_query: bool = False
    uses_shared_paged_kv_idx: bool = True
    enable_sink: bool = False
    skip_softmax: bool = False
    return_lse: bool = False
    provide_lse: bool = False
    device_scale: bool = False
    bmm2_scale: float = 1.0

    @property
    def variant(self) -> str:
        return "topk" if self.topk > 0 else "dense"


def _workspace_get_or_create(
    key: tuple[Any, ...], builder: Callable[[], dict[str, Any]]
) -> dict[str, Any]:
    with _WORKSPACE_CACHE_LOCK:
        state = _WORKSPACE_CACHE.get(key)
        if state is not None:
            _WORKSPACE_CACHE.move_to_end(key)
            return state
    state = builder()
    with _WORKSPACE_CACHE_LOCK:
        existing = _WORKSPACE_CACHE.get(key)
        if existing is not None:
            _WORKSPACE_CACHE.move_to_end(key)
            return existing
        _WORKSPACE_CACHE[key] = state
        while len(_WORKSPACE_CACHE) > _WORKSPACE_CACHE_CAPACITY:
            _WORKSPACE_CACHE.popitem(last=False)
    return state


def _tensor_ptr(value: Any) -> int:
    return 0 if value is None else int(value.data_ptr())


def _tensor_version(value: Any) -> int:
    return -1 if value is None else int(getattr(value, "_version", 0))


def _host_int_tuple(tensor: torch.Tensor) -> tuple[int, ...]:
    key = (
        _tensor_ptr(tensor),
        _tensor_version(tensor),
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
    )
    with _HOST_METADATA_CACHE_LOCK:
        values = _HOST_METADATA_CACHE.get(key)
        if values is not None:
            _HOST_METADATA_CACHE.move_to_end(key)
            return values
    values = tuple(int(value) for value in tensor.tolist())
    with _HOST_METADATA_CACHE_LOCK:
        existing = _HOST_METADATA_CACHE.get(key)
        if existing is not None:
            _HOST_METADATA_CACHE.move_to_end(key)
            return existing
        _HOST_METADATA_CACHE[key] = values
        while len(_HOST_METADATA_CACHE) > _HOST_METADATA_CACHE_CAPACITY:
            _HOST_METADATA_CACHE.popitem(last=False)
    return values


def _state_key(inputs: dict[str, Any], family: str) -> tuple[Any, ...]:
    tensors = (
        inputs.get("Q"),
        inputs.get("KV_cache"),
        inputs.get("page_table"),
        inputs.get("q_indptr"),
        inputs.get("seq_lens"),
        inputs.get("O"),
        inputs.get("LSE"),
        inputs.get("sinks"),
    )
    return (
        family,
        *((_tensor_ptr(value), _tensor_version(value)) for value in tensors),
        tuple(inputs["Q"].shape),
        tuple(inputs["KV_cache"].shape),
        str(inputs["dtype"]),
        int(inputs["page_size"]),
        int(inputs["q_len"]),
        int(inputs["total_q"]),
        tuple(inputs["q_lens"]),
        tuple(inputs["kv_lens"]),
        int(inputs["topk"]),
        int(inputs["stream"]),
    )


def _e4m3_decode_table() -> list[float]:
    values = []
    for bits in range(256):
        sign = -1.0 if bits & 0x80 else 1.0
        exponent = (bits >> 3) & 0xF
        mantissa = bits & 0x7
        if exponent == 0:
            magnitude = mantissa * (2.0**-9)
        elif exponent == 0xF and mantissa == 0x7:
            magnitude = 0.0
        else:
            magnitude = (1.0 + mantissa / 8.0) * (2.0 ** (exponent - 7))
        values.append(sign * magnitude)
    return values


def _runtime_rows(inputs: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    device = inputs["Q"].device
    row_batches, causal_lens = [], []
    for batch, (q_len, kv_len) in enumerate(
        zip(inputs["q_lens"], inputs["kv_lens"], strict=True)
    ):
        for query in range(q_len):
            row_batches.append(batch)
            causal_lens.append(kv_len - q_len + query + 1)
    row_batches = torch.tensor(row_batches, dtype=torch.int32, device=device)
    row_seq_lens = torch.tensor(causal_lens, dtype=torch.int32, device=device)
    return row_batches, row_seq_lens


def _base_state(inputs: dict[str, Any], family: str) -> dict[str, Any]:
    key = _state_key(inputs, family)

    def build() -> dict[str, Any]:
        row_batches, row_seq_lens = _runtime_rows(inputs)
        device = inputs["Q"].device
        row_batch_long = row_batches.to(torch.int64)
        use_sparse = int(inputs["topk"]) > 0
        sparse_width = int(inputs["topk"]) if use_sparse else 0
        source_table = inputs["page_table"]
        if source_table.ndim == 3:
            source_table = source_table[:, 0]
        source_table = source_table.to(dtype=torch.int32).contiguous()
        row_table = source_table.index_select(0, row_batch_long)
        if use_sparse:
            sparse_indices = inputs["page_table"].reshape(
                int(inputs["total_q"]), sparse_width
            )
            sparse_indices = sparse_indices.to(dtype=torch.int32).contiguous()
            row_seq_lens = torch.full_like(row_seq_lens, sparse_width)
        else:
            sparse_indices = row_batches
        sinks = inputs["sinks"]
        if sinks is None:
            sinks = torch.zeros(
                int(inputs["num_heads"]), dtype=torch.float32, device=device
            )
        lse = inputs["LSE"]
        if lse is None:
            lse = torch.empty(
                (int(inputs["total_q"]), int(inputs["num_heads"])),
                dtype=torch.float32,
                device=device,
            )
        fp8_lut = (
            torch.tensor(_e4m3_decode_table(), dtype=torch.float32, device=device)
            if inputs["dtype"] == torch.float8_e4m3fn
            else torch.zeros(1, dtype=torch.float32, device=device)
        )
        return {
            "num_rows": int(inputs["total_q"]),
            "row_batches": row_batches,
            "row_seq_lens": row_seq_lens,
            "source_page_table": source_table,
            "row_page_table": row_table,
            "sparse_indices": sparse_indices,
            "sparse_width": sparse_width,
            "use_sparse": use_sparse,
            "sinks": sinks,
            "lse": lse,
            "fp8_lut": fp8_lut,
        }

    return _workspace_get_or_create(key, build)


def _aligned_state(inputs: dict[str, Any]) -> dict[str, Any]:
    state = _base_state(inputs, "aligned_bf16")
    if "kv_half_pages" not in state:
        table = state["row_page_table"]
        if int(inputs["page_size"]) == 64:
            table = torch.stack((table * 2, table * 2 + 1), dim=-1)
            table = table.reshape(state["num_rows"], -1).contiguous()
        state.update(
            {
                "row_page_table": table,
                "kv_half_pages": inputs["KV_cache"].reshape(
                    -1, 32, _ALIGNED_QK_DIM
                ),
                "q_rows": inputs["Q"].reshape(-1, _ALIGNED_QK_DIM),
                "o_rows": inputs["O"].reshape(-1, _ALIGNED_VALUE_DIM),
                "num_sms": int(inputs["num_sms"]),
            }
        )
    return state


def _is_aligned(meta: _BlackwellDispatchMetadata) -> bool:
    return (
        meta.num_heads == _ALIGNED_HEADS
        and meta.qk_dim == _ALIGNED_QK_DIM
        and meta.value_dim == _ALIGNED_VALUE_DIM
        and meta.topk == 0
        and meta.uses_shared_paged_kv_idx
        and not meta.return_lse
        and not meta.provide_lse
    )


def _can_clc(meta: _BlackwellDispatchMetadata) -> bool:
    q_lens = meta.q_lens
    return (
        meta.dtype == torch.bfloat16
        and _is_aligned(meta)
        and meta.page_size == 32
        and meta.variant == "dense"
        and not meta.enable_sink
        and not meta.skip_softmax
        and meta.bmm2_scale == 1.0
        and bool(q_lens)
        and min(q_lens) > 0
        and len(set(q_lens)) == 1
    )


def _use_unsplit(meta: _BlackwellDispatchMetadata) -> bool:
    return (
        _can_clc(meta)
        and meta.q_lens == (16,) * 64
        and len(meta.kv_lens) == 64
        and max(meta.kv_lens) == 1024
        and meta.kv_lens[-1] == 1024
    )


def _use_native_bf16(meta: _BlackwellDispatchMetadata) -> bool:
    if (
        meta.dtype != torch.bfloat16
        or not _is_aligned(meta)
        or meta.variant != "dense"
        or meta.ragged_query
        or meta.skip_softmax
    ):
        return False
    return (
        meta.page_size == 32
        and not meta.enable_sink
        and meta.q_lens == (4,)
        and meta.kv_lens == (1024,)
    ) or (
        meta.page_size == 64
        and meta.enable_sink
        and meta.q_lens == (1,)
        and meta.kv_lens == (1024,)
    )


def _can_fp8_p32(meta: _BlackwellDispatchMetadata) -> bool:
    q_lens = meta.q_lens
    if not q_lens or len(q_lens) != len(meta.kv_lens) or len(set(q_lens)) != 1:
        return False
    q_len = q_lens[0]
    causal = [
        kv_len - q_len + query + 1
        for kv_len, query_len in zip(meta.kv_lens, q_lens, strict=True)
        for query in range(query_len)
    ]
    return (
        meta.dtype == torch.float8_e4m3fn
        and _is_aligned(meta)
        and q_len == 2
        and meta.kv_lens == (1024,)
        and not meta.ragged_query
        and meta.page_size == 32
        and meta.variant == "dense"
        and meta.table_ndim == 2
        and not meta.device_scale
        and not meta.enable_sink
        and not meta.skip_softmax
        and all(length > 256 for length in causal)
    )


def _can_page64(meta: _BlackwellDispatchMetadata) -> bool:
    if meta.ragged_query and len(set(meta.q_lens)) != 1:
        return False
    return (
        meta.dtype == torch.float8_e4m3fn
        and _is_aligned(meta)
        and meta.page_size == 64
        and meta.variant == "dense"
        and not meta.enable_sink
        and not meta.skip_softmax
    )


def _select_route(meta: _BlackwellDispatchMetadata) -> str:
    if meta.dtype == torch.bfloat16 and _is_aligned(meta):
        resident = max(1, meta.num_sms // _CLUSTER_SIZE)
        if _use_native_bf16(meta):
            return _NATIVE_BF16_ROUTE
        if _use_unsplit(meta):
            return _UNSPLIT_ROUTE
        if meta.enable_sink or meta.total_q * 4 <= resident:
            return _VQUARTER_ROUTE
        if meta.total_q * 2 <= resident:
            return _VHALF_ROUTE
        if _can_clc(meta):
            return _CLC_ROUTE
        raise ValueError(
            "TRT-LLM MLA Blackwell has no qualified aligned BF16 domain for this configuration"
        )
    if _can_fp8_p32(meta):
        return _FP8_P32_QK_L2_ROUTE
    if _can_page64(meta):
        return _FP8_PAGE64_ROUTE
    if max(meta.kv_lens) > _MAX_GENERIC_TOKENS:
        raise ValueError(
            "TRT-LLM MLA Blackwell generic tail supports max_seq_len <= "
            f"{_MAX_GENERIC_TOKENS}"
        )
    return (
        _GENERIC_BF16_ROUTE
        if meta.dtype == torch.bfloat16
        else _GENERIC_FP8_ROUTE
    )


def _aligned_launch(inputs: dict[str, Any], route: str):
    state = _aligned_state(inputs)
    resident = max(1, state["num_sms"] // _CLUSTER_SIZE)
    if route == _VQUARTER_ROUTE:
        split = 4
        work = state["num_rows"] * 4
        grid_x = min(work, resident) * 2
        grid_z = 1
    elif route == _VHALF_ROUTE:
        split = 2
        work = state["num_rows"] * 2
        grid_x = min(work, resident) * 2
        grid_z = 1
    elif route == _UNSPLIT_ROUTE:
        split = 1
        work = state["num_rows"]
        grid_x = min(work, resident) * 2
        grid_z = 1
    else:
        split = int(inputs["q_len"])
        work = state["num_rows"]
        grid_x = 2 * split
        grid_z = int(inputs["batch_size"])
    tensors = (
        state["q_rows"],
        state["kv_half_pages"],
        state["o_rows"],
        state["row_seq_lens"],
        state["row_page_table"],
        state["sinks"],
    )
    scalars = (
        float(inputs["bmm1_scale"]) * log2e,
        float(inputs["bmm2_scale"]),
        work,
        split,
        int(state["row_page_table"].shape[1]),
        int(inputs["sinks"] is not None),
        grid_x,
        grid_z,
    )
    return tensors, scalars


def _native_launch(inputs: dict[str, Any]):
    state = _aligned_state(inputs)
    num_split = 8
    total_work_items = state["num_rows"] * num_split
    grid_x = min(
        total_work_items, state["num_sms"] // _CLUSTER_SIZE
    ) * _CLUSTER_SIZE
    tensors = (
        state["q_rows"],
        state["kv_half_pages"],
        state["o_rows"],
        state["row_seq_lens"],
        state["row_page_table"],
        state["sinks"],
    )
    scalars = (
        float(inputs["bmm1_scale"]) * log2e,
        1.0,
        float(inputs["bmm2_scale"]),
        num_split,
        total_work_items,
        int(state["row_page_table"].shape[1]),
        int(inputs["sinks"] is not None),
        grid_x,
        state["num_rows"],
    )
    return tensors, scalars


def _generic_launch(inputs: dict[str, Any]):
    state = _base_state(inputs, "generic")
    num_heads = int(inputs["num_heads"])
    qk_dim = int(inputs["Q"].shape[-1])
    value_dim = int(inputs["O"].shape[-1])
    kv_stride = int(inputs["KV_cache"].shape[-1])
    query = inputs["Q"].reshape(-1, qk_dim)
    kv = inputs["KV_cache"].reshape(-1, kv_stride)
    if inputs["dtype"] == torch.float8_e4m3fn:
        query = query.view(torch.uint8)
        kv = kv.view(torch.uint8)
    tensors = (
        query,
        kv,
        state["fp8_lut"],
        state["source_page_table"],
        state["sparse_indices"],
        state["row_batches"],
        state["row_seq_lens"],
        inputs["O"].reshape(-1, value_dim),
        state["lse"].reshape(-1),
        state["sinks"],
    )
    scalars = (
        num_heads,
        qk_dim,
        value_dim,
        kv_stride,
        int(inputs["page_size"]),
        int(state["source_page_table"].shape[1]),
        state["sparse_width"],
        int(state["use_sparse"]),
        float(inputs["bmm1_scale"]),
        float(inputs["bmm2_scale"]),
        int(inputs["sinks"] is not None),
        int(inputs["return_lse"] or inputs["provide_lse"]),
        state["num_rows"],
    )
    return tensors, scalars


def _p32_launch(inputs: dict[str, Any]):
    reduction_groups = int(inputs["batch_size"]) * 32 * int(inputs["q_len"])
    key = _state_key(inputs, "fp8_p32")
    state = _workspace_get_or_create(
        key,
        lambda: {
            "completion": torch.zeros(
                reduction_groups,
                dtype=torch.uint32,
                device=inputs["Q"].device,
            )
        },
    )
    tensors = (
        inputs["Q"].reshape(-1, 576).view(torch.uint8),
        inputs["KV_cache"].reshape(-1, 32, 576).view(torch.uint8),
        inputs["page_table"].reshape(-1),
        inputs["seq_lens"],
        inputs["O"].reshape(-1),
        state["completion"],
    )
    scalars = (
        int(inputs["batch_size"]),
        int(inputs["q_len"]),
        int(inputs["page_table"].shape[-1]),
        int(inputs["q_len"]),
        2,
        float(inputs["bmm1_scale"]) * log2e,
        float(inputs["bmm2_scale"]),
    )
    return tensors, scalars


def _pick_num_split(work_items: int, tiles: list[int], num_sms: int) -> int:
    tiles = [int(value) for value in tiles if int(value) > 0]
    if not tiles:
        return 1
    min_tiles, max_tiles = min(tiles), max(tiles)
    source_cap = max(1, (max_tiles + 1) // 2)
    if min_tiles <= 2:
        return 1
    target = min(source_cap, max(1, num_sms // max(1, work_items * 2)))
    split = max(1, min(target, min_tiles))
    blocks = (min_tiles + split - 1) // split
    split = (min_tiles + blocks - 1) // blocks
    while split > 1:
        if all(
            (split - 1) * ((count + split - 1) // split) < count
            for count in tiles
        ):
            break
        split -= 1
    return split


def _page64_launch(inputs: dict[str, Any]):
    state = _base_state(inputs, "fp8_page64")
    q_lens = inputs["q_lens"]
    kv_lens = inputs["kv_lens"]
    work_items = sum(q_lens)
    causal_lengths = [
        kv_lens[batch] - q_len + query
        for batch, q_len in enumerate(q_lens)
        for query in range(q_len)
    ]
    tiles = [(length + 127) // 128 for length in causal_lengths]
    num_split = _pick_num_split(work_items, tiles, int(inputs["num_sms"]))
    if num_split <= 1:
        raise ValueError("TRT-LLM MLA Blackwell page-64 PDL domain requires split-K reduction")
    reduce_ctas = min(
        64,
        max(1, (int(inputs["num_sms"]) * 2) // max(1, work_items * 2)),
    )
    tensors = (
        inputs["Q"].reshape(-1, 576).view(torch.uint8),
        inputs["KV_cache"].reshape(-1, 576).view(torch.uint8),
        inputs["O"].reshape(-1, 512),
        state["lse"].reshape(-1, 128),
        torch.tensor(causal_lengths, dtype=torch.int32, device=inputs["Q"].device),
        state["row_batches"],
        inputs["page_table"].reshape(-1),
    )
    scalars = (
        float(inputs["bmm1_scale"]) * log2e,
        float(inputs["bmm2_scale"]),
        num_split,
        work_items,
        int(inputs["page_table"].shape[-1]),
        reduce_ctas,
    )
    return tensors, scalars


def _prepare(inputs: dict[str, Any], route: str):
    if route == _NATIVE_BF16_ROUTE:
        tensors, scalars = _native_launch(inputs)
    elif route in {_VQUARTER_ROUTE, _VHALF_ROUTE, _UNSPLIT_ROUTE, _CLC_ROUTE}:
        tensors, scalars = _aligned_launch(inputs, route)
    elif route in {_GENERIC_BF16_ROUTE, _GENERIC_FP8_ROUTE}:
        tensors, scalars = _generic_launch(inputs)
    elif route == _FP8_P32_QK_L2_ROUTE:
        tensors, scalars = _p32_launch(inputs)
    elif route == _FP8_PAGE64_ROUTE:
        tensors, scalars = _page64_launch(inputs)
    else:
        raise ValueError(f"route is outside the TRT-LLM MLA Blackwell export inventory: {route!r}")
    return ROUTE_TO_DOMAIN[route], tensors, scalars


def _check_tensor(
    tensor: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _normalize_sinks(
    sinks: Optional[Union[list[torch.Tensor], tuple[torch.Tensor, ...], torch.Tensor]],
    *,
    num_heads: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if sinks is None:
        return None
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError("TRT-LLM MLA Blackwell expects one sink tensor")
        sink = sinks[0]
    else:
        sink = sinks
    _check_tensor(sink, name="sinks", dtype=torch.float32, device=device)
    if tuple(sink.shape) != (num_heads,):
        raise ValueError(f"sinks must have shape ({num_heads},)")
    return sink


def _normalize_scale(value: float | torch.Tensor, name: str) -> float:
    if isinstance(value, torch.Tensor):
        raise TypeError(f"TRT-LLM MLA Blackwell requires scalar {name}")
    if not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a scalar float")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def trtllm_mla_blackwell_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: Optional[torch.Tensor],
    max_seq_len: int,
    *,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    out: Optional[torch.Tensor],
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: Optional[list[torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    enable_pdl: Optional[bool],
    uses_shared_paged_kv_idx: bool,
    lse: Optional[torch.Tensor],
    return_lse: bool,
    cum_seq_lens_q: Optional[torch.Tensor],
    max_q_len: Optional[int],
    multi_ctas_kv_counter_buffer: Optional[torch.Tensor],
    sparse_mla_top_k_lens: Optional[torch.Tensor],
    enable_dcp: bool,
    backend: Literal["trtllm-mla-blackwell"] = "trtllm-mla-blackwell",
) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
    """Dispatch the qualified SM100a/SM103a MLA semantic envelope."""

    if backend != "trtllm-mla-blackwell":
        raise ValueError(f"backend must be 'trtllm-mla-blackwell', got {backend!r}")
    if not isinstance(query, torch.Tensor):
        raise TypeError("query must be a torch.Tensor")
    if not query.is_cuda:
        raise ValueError("query must be a CUDA tensor")
    device = query.device
    capability = get_compute_capability(device)
    if capability not in {(10, 0), (10, 3)}:
        major, minor = capability
        raise RuntimeError(
            "TRT-LLM MLA Blackwell requires compute capability 10.0 or 10.3, "
            f"got {major}.{minor}"
        )
    if not is_cuda_version_at_least("12.9"):
        raise RuntimeError(
            "TRT-LLM MLA Blackwell on SM100a/SM103a requires CUDA 12.9 or newer"
        )
    if enable_dcp:
        raise ValueError("TRT-LLM MLA Blackwell does not support DCP")
    if multi_ctas_kv_counter_buffer is not None:
        raise ValueError("TRT-LLM MLA Blackwell does not use a multi-CTA counter buffer")
    if sparse_mla_top_k_lens is not None:
        raise ValueError("TRT-LLM MLA Blackwell does not accept sparse_mla_top_k_lens")
    if seq_lens is None:
        raise ValueError("seq_lens is required for TRT-LLM MLA Blackwell")

    if query.dtype not in {torch.bfloat16, torch.float8_e4m3fn}:
        raise TypeError("TRT-LLM MLA Blackwell query must use BF16 or FP8 E4M3")
    _check_tensor(query, name="query", dtype=query.dtype, device=device)
    _check_tensor(kv_cache, name="kv_cache", dtype=query.dtype, device=device)
    if query.ndim not in {3, 4}:
        raise ValueError(
            "query must be a fixed [B, Q, H, D] or compact [T, H, D] tensor"
        )
    if kv_cache.ndim not in {3, 4}:
        raise ValueError("kv_cache must be a 3D or 4D paged tensor")
    if kv_cache.ndim == 4 and kv_cache.shape[1] != 1:
        raise ValueError("4D kv_cache must have a singleton head axis")
    page_size = int(kv_cache.shape[-2])
    if page_size not in {32, 64}:
        raise ValueError("TRT-LLM MLA Blackwell requires page size 32 or 64")
    if int(query.shape[-1]) != int(kv_cache.shape[-1]):
        raise ValueError("query and kv_cache head dimensions differ")

    ragged_query = cum_seq_lens_q is not None
    if ragged_query != (query.ndim == 3):
        raise ValueError("compact query and cum_seq_lens_q must be provided together")
    if ragged_query:
        _check_tensor(
            cum_seq_lens_q,
            name="cum_seq_lens_q",
            dtype=torch.int32,
            device=device,
        )
        if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.numel() < 2:
            raise ValueError("cum_seq_lens_q must have shape [batch_size + 1]")
        if max_q_len is None or max_q_len <= 0:
            raise ValueError("max_q_len is required for compact TRT-LLM MLA Blackwell queries")
        batch_size = int(cum_seq_lens_q.numel() - 1)
        q_len = int(max_q_len)
        total_q = int(query.shape[0])
    else:
        batch_size = int(query.shape[0])
        q_len = int(query.shape[1])
        total_q = batch_size * q_len
    if batch_size <= 0 or q_len <= 0 or total_q <= 0:
        raise ValueError("TRT-LLM MLA Blackwell requires nonempty batch and query dimensions")

    _check_tensor(seq_lens, name="seq_lens", dtype=torch.int32, device=device)
    if tuple(seq_lens.shape) != (batch_size,):
        raise ValueError(f"seq_lens must have shape ({batch_size},)")
    if ragged_query:
        q_indptr_host = _host_int_tuple(cum_seq_lens_q)
        if q_indptr_host[0] != 0 or q_indptr_host[-1] != total_q:
            raise ValueError("cum_seq_lens_q must start at 0 and end at total_q")
        q_lens = tuple(
            right - left
            for left, right in zip(
                q_indptr_host[:-1], q_indptr_host[1:], strict=True
            )
        )
        if any(q <= 0 for q in q_lens) or max(q_lens) > q_len:
            raise ValueError("cum_seq_lens_q contains an invalid query length")
    else:
        q_lens = (q_len,) * batch_size
    kv_lens = _host_int_tuple(seq_lens)
    if len(q_lens) != len(kv_lens) or any(
        q <= 0 or kv <= 0 or q > kv
        for q, kv in zip(q_lens, kv_lens, strict=True)
    ):
        raise ValueError("every TRT-LLM MLA Blackwell row requires 0 < q_len <= kv_len")
    if max_seq_len <= 0 or max(kv_lens) > max_seq_len:
        raise ValueError("max_seq_len must cover every runtime KV length")
    _check_tensor(
        block_tables, name="block_tables", dtype=torch.int32, device=device
    )
    if sparse_mla_top_k > 0:
        expected_table_shape = (
            (total_q, sparse_mla_top_k)
            if ragged_query
            else (batch_size, q_len, sparse_mla_top_k)
        )
        if tuple(block_tables.shape) != expected_table_shape:
            raise ValueError(f"sparse block_tables must have shape {expected_table_shape}")
    else:
        expected_ndim = 2 if uses_shared_paged_kv_idx else 3
        if block_tables.ndim != expected_ndim or block_tables.shape[0] != batch_size:
            raise ValueError(
                "dense block_tables must match the batch and shared-index layout"
            )

    num_heads = int(query.shape[-2])
    qk_dim = int(query.shape[-1])
    value_dim = int(kv_lora_rank)
    valid_dense = sparse_mla_top_k == 0 and (
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        num_heads,
    ) in {
        (128, 512, 64, 128),
        (128, 512, 64, 64),
        (64, 256, 64, 32),
        (512, 512, 64, 128),
    }
    valid_topk = sparse_mla_top_k > 0 and (
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        num_heads,
    ) in {
        (128, 512, 64, 128),
        (128, 512, 64, 64),
        (192, 512, 64, 128),
        (192, 512, 64, 64),
    }
    if not valid_dense and not valid_topk:
        raise ValueError("unsupported TRT-LLM MLA Blackwell dimension tuple")
    if qk_dim != kv_lora_rank + qk_rope_head_dim:
        raise ValueError("query width must equal kv_lora_rank + qk_rope_head_dim")

    sink = _normalize_sinks(sinks, num_heads=num_heads, device=device)
    bmm1_scale_value = _normalize_scale(bmm1_scale, "bmm1_scale")
    bmm2_scale_value = _normalize_scale(bmm2_scale, "bmm2_scale")
    if skip_softmax_threshold_scale_factor is not None:
        threshold = float(skip_softmax_threshold_scale_factor)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("skip_softmax_threshold_scale_factor must be positive")
    expected_out_shape = (*query.shape[:-1], value_dim)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=device)
    else:
        _check_tensor(out, name="out", dtype=torch.bfloat16, device=device)
        if tuple(out.shape) != expected_out_shape:
            raise ValueError(f"out must have shape {expected_out_shape}")
    if lse is not None:
        _check_tensor(lse, name="lse", dtype=torch.float32, device=device)
        valid_lse_shapes = {(total_q, num_heads), (*query.shape[:-1],)}
        if tuple(lse.shape) not in valid_lse_shapes:
            raise ValueError("lse shape must match flattened or physical query rows")
    if return_lse and lse is None:
        lse = torch.empty((total_q, num_heads), dtype=torch.float32, device=device)

    num_sms = get_device_sm_count(device)
    metadata = _BlackwellDispatchMetadata(
        dtype=query.dtype,
        batch_size=batch_size,
        q_len=q_len,
        total_q=total_q,
        q_lens=q_lens,
        kv_lens=kv_lens,
        num_heads=num_heads,
        qk_dim=qk_dim,
        value_dim=value_dim,
        page_size=page_size,
        max_seq_len=int(max_seq_len),
        topk=int(sparse_mla_top_k),
        table_ndim=int(block_tables.ndim),
        num_sms=num_sms,
        ragged_query=ragged_query,
        uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
        enable_sink=sink is not None,
        skip_softmax=skip_softmax_threshold_scale_factor is not None,
        return_lse=return_lse,
        provide_lse=lse is not None and not return_lse,
        device_scale=False,
        bmm2_scale=bmm2_scale_value,
    )
    route = _select_route(metadata)
    stream = int(torch.cuda.current_stream(device).cuda_stream)
    inputs = {
        "Q": query,
        "KV_cache": kv_cache,
        "page_table": block_tables,
        "q_indptr": cum_seq_lens_q,
        "seq_lens": seq_lens,
        "O": out,
        "LSE": lse,
        "sinks": sink,
        "dtype": query.dtype,
        "batch_size": batch_size,
        "q_len": q_len,
        "total_q": total_q,
        "q_lens": q_lens,
        "kv_lens": kv_lens,
        "num_heads": num_heads,
        "page_size": page_size,
        "max_seq_len": int(max_seq_len),
        "topk": int(sparse_mla_top_k),
        "bmm1_scale": bmm1_scale_value,
        "bmm2_scale": bmm2_scale_value,
        "return_lse": return_lse,
        "provide_lse": lse is not None and not return_lse,
        "num_sms": num_sms,
        "stream": stream,
    }
    domain, tensors, scalars = _prepare(inputs, route)
    with torch.cuda.device(device):
        get_domain_module(domain, device).run(*tensors, *scalars, stream)
    if return_lse:
        assert lse is not None
        return out, lse
    return out


__all__ = ["ROUTE_TO_DOMAIN", "trtllm_mla_blackwell_decode"]
