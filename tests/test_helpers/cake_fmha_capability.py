"""Pinned, allocation-free Cake FMHA high-level selector replay.

The axes below are copied from the public TRT-LLM paged decode/context tests at
``PINNED_FLASHINFER_REVISION``.  Lightweight tensor-ABI objects preserve the
shapes, strides, dtypes, devices, alignment, scalar values, and metadata that
the Cake selectors inspect without allocating the very large attention
tensors in the original 80,768-cell parameter product.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from itertools import chain, product
from typing import Any, Iterable, Iterator

import torch

import flashinfer.cake_fmha as cake_api


PINNED_FLASHINFER_REVISION = "5b8da12050f80a5b5cb2bab9e87d9635a8872e5b"

DECODE_GEOMETRIES = (
    (4, 1, 16, 2, 1),
    (4, 1, 32, 2, 5),
    (4, 2, 64, 2, 5),
    (4, 3, 32, 2, 5),
    (4, 3, 64, 2, 1),
    (4, 4, 64, 4, 1),
    (4, 5, 64, 4, 8),
    (128, 1, 64, 2, 5),
    (128, 2, 32, 4, 1),
    (128, 3, 16, 4, 8),
    (128, 4, 16, 2, 5),
    (128, 5, 16, 2, 5),
    (256, 1, 64, 4, 8),
    (256, 2, 16, 2, 8),
    (256, 3, 64, 4, 5),
    (256, 4, 32, 2, 8),
    (256, 5, 32, 2, 1),
)

DECODE_DTYPES = (
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("bf16", "fp8", "bf16"),
    ("fp16", "fp8", "fp16"),
    ("bf16", "fp8", "fp8"),
    ("fp16", "fp8", "fp8"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
)

CONTEXT_GEOMETRIES = (
    (4, 16, 2, 1),
    (4, 32, 4, 5),
    (4, 64, 4, 8),
    (128, 16, 2, 5),
    (128, 32, 4, 1),
    (128, 64, 2, 8),
    (256, 16, 4, 8),
    (256, 32, 2, 8),
    (256, 64, 4, 1),
    (256, 64, 4, 5),
)

CONTEXT_DTYPES = (
    ("bf16", "bf16", "bf16"),
    ("fp16", "fp16", "fp16"),
    ("fp8", "fp8", "bf16"),
    ("fp8", "fp8", "fp16"),
    ("fp8", "fp8", "fp8"),
    ("fp8", "fp8", "nvfp4"),
    ("fp8", "nvfp4", "fp8"),
)

_TORCH_DTYPE = {
    "bf16": torch.bfloat16,
    "fp16": torch.float16,
    "fp8": torch.float8_e4m3fn,
    # The selector sees the packed payload, not a logical torch FP4 dtype.
    "nvfp4": torch.uint8,
}
_ELEMENT_BYTES = {
    torch.bfloat16: 2,
    torch.float16: 2,
    torch.float8_e4m3fn: 1,
    torch.uint8: 1,
    torch.int32: 4,
    torch.float32: 4,
}
_ROUTE_NAMES = {
    "context_bf16": "ctx_bf16_hnd_hd128_hgpack_03df_v2",
    "context_fp16_hd256": "ctx_fp16_nhd_hd256_stage16_v1",
    "context_fp8": "ctx_fp8_hnd_hd128_hgpack_48b5_v1",
    "context_fp8_hd256": "ctx_fp8_bf16_nhd_hd256_stage16_v1",
    "context_nvfp4": "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1",
    "decode_native_bf16": "decode_native_bf16_v1_bece",
    "decode_native_fp16_hd512": "decode_native_fp16_hd512_v1_66b1",
    "decode_native_fp16_nhd": "decode_native_fp16_nhd_v1_f32d",
    "decode_quant_bf16q": "decode_quantized_bf16q_9d8b_v1",
    "decode_quant_fp8": "decode_quantized_fp8_8e5b_v1",
    "decode_quant_nvfp4": "decode_quantized_nvfp4_8e5b_v1",
}
_CASE_FIELDS = (
    "mode",
    "kv_layout",
    "batch_size",
    "q_len",
    "kv_len",
    "page_size",
    "num_kv_heads",
    "num_qo_heads",
    "window_left",
    "q_dtype",
    "kv_dtype",
    "o_dtype",
    "enable_pdl",
    "enable_sink",
    "head_dim",
    "non_contiguous_query",
    "skip_softmax",
    "uses_shared_paged_kv_idx",
    "causal",
)
_DEVICE_SCALE = torch.ones((), dtype=torch.float32)
_SINKS_BY_HEADS: dict[int, torch.Tensor] = {}


@dataclass(frozen=True)
class TensorAbi:
    """The torch.Tensor protocol surface inspected by Cake route guards."""

    shape: tuple[int, ...]
    strides: tuple[int, ...]
    dtype: torch.dtype
    contiguous: bool = True
    values: tuple[int, ...] | None = None
    pointer: int = 0x1000
    numel_override: int | None = None
    device: torch.device = torch.device("cpu")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    def is_contiguous(self) -> bool:
        return self.contiguous

    def stride(self, dim: int | None = None):
        return self.strides if dim is None else self.strides[dim]

    def numel(self) -> int:
        return (
            self.numel_override
            if self.numel_override is not None
            else math.prod(self.shape)
        )

    def element_size(self) -> int:
        return _ELEMENT_BYTES[self.dtype]

    def data_ptr(self) -> int:
        return self.pointer

    def detach(self) -> TensorAbi:
        return self

    def cpu(self) -> TensorAbi:
        return self

    def tolist(self) -> list[int]:
        if self.values is None:
            raise TypeError("this ABI tensor has no host values")
        return list(self.values)


@dataclass(frozen=True)
class SelectorReplay:
    raw_cases: int
    valid_cases: int
    optimized_cases: int
    compat_cases: int
    route_counts: dict[str, int]
    digest: str


def iter_decode_cases() -> Iterator[dict[str, Any]]:
    axes: Iterable[tuple[Any, ...]] = product(
        ("HND", "NHD"),
        DECODE_GEOMETRIES,
        (-1, 127),
        DECODE_DTYPES,
        (True, False, None),
        (True, False),
        (128, 256),
        (False, True),
        (False, True),
        (True, False),
    )
    for (
        kv_layout,
        geometry,
        window_left,
        dtypes,
        enable_pdl,
        enable_sink,
        head_dim,
        non_contiguous_query,
        skip_softmax,
        uses_shared_paged_kv_idx,
    ) in axes:
        batch_size, q_len, page_size, num_kv_heads, head_group_size = geometry
        q_dtype, kv_dtype, o_dtype = dtypes
        yield {
            "mode": "decode",
            "batch_size": batch_size,
            "q_len": q_len,
            "kv_len": 110 + q_len,
            "page_size": page_size,
            "num_kv_heads": num_kv_heads,
            "num_qo_heads": num_kv_heads * head_group_size,
            "window_left": window_left,
            "q_dtype": q_dtype,
            "kv_dtype": kv_dtype,
            "o_dtype": o_dtype,
            "enable_pdl": enable_pdl,
            "enable_sink": enable_sink,
            "head_dim": head_dim,
            "non_contiguous_query": non_contiguous_query,
            "skip_softmax": skip_softmax,
            "uses_shared_paged_kv_idx": uses_shared_paged_kv_idx,
            "kv_layout": kv_layout,
            "causal": True,
        }


def iter_context_cases() -> Iterator[dict[str, Any]]:
    axes: Iterable[tuple[Any, ...]] = product(
        ("HND", "NHD"),
        CONTEXT_GEOMETRIES,
        CONTEXT_DTYPES,
        (True, False),
        (128, 256),
        (False, True),
        (False, True),
        (True, False),
        (True, False),
    )
    for (
        kv_layout,
        geometry,
        dtypes,
        enable_sink,
        head_dim,
        non_contiguous_query,
        skip_softmax,
        uses_shared_paged_kv_idx,
        causal,
    ) in axes:
        batch_size, page_size, num_kv_heads, head_group_size = geometry
        q_dtype, kv_dtype, o_dtype = dtypes
        yield {
            "mode": "context",
            "batch_size": batch_size,
            "q_len": 511,
            "kv_len": 2047,
            "page_size": page_size,
            "num_kv_heads": num_kv_heads,
            "num_qo_heads": num_kv_heads * head_group_size,
            "window_left": -1,
            "q_dtype": q_dtype,
            "kv_dtype": kv_dtype,
            "o_dtype": o_dtype,
            "enable_pdl": None,
            "enable_sink": enable_sink,
            "head_dim": head_dim,
            "non_contiguous_query": non_contiguous_query,
            "skip_softmax": skip_softmax,
            "uses_shared_paged_kv_idx": uses_shared_paged_kv_idx,
            "kv_layout": kv_layout,
            "causal": causal,
        }


def upstream_skip_reason(case: dict[str, Any]) -> str | None:
    if case["skip_softmax"] and case["q_dtype"] != case["kv_dtype"]:
        return "skip-softmax requires matching Q/KV dtypes"
    if case["mode"] == "decode" and case["o_dtype"] == "fp8" and case["q_dtype"] != "fp8":
        return "decode FP8 output requires FP8 query"
    if case["kv_dtype"] == "nvfp4" and (
        case["q_dtype"] != "fp8" or case["o_dtype"] != "fp8"
    ):
        return "NVFP4 KV requires FP8 Q/O"
    return None


def _contiguous_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride = 1
    result = []
    for extent in reversed(shape):
        result.append(stride)
        stride *= extent
    return tuple(reversed(result))


def _tensor(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    *,
    strides: tuple[int, ...] | None = None,
    contiguous: bool = True,
    values: tuple[int, ...] | None = None,
    numel_override: int | None = None,
) -> TensorAbi:
    return TensorAbi(
        shape,
        _contiguous_strides(shape) if strides is None else strides,
        dtype,
        contiguous=contiguous,
        values=values,
        numel_override=numel_override,
    )


def _normalized_kv_tensor(
    *,
    pages: int,
    num_kv_heads: int,
    page_size: int,
    stored_head_dim: int,
    kv_layout: str,
    shared: bool,
    stacked_shared: bool,
    dtype: torch.dtype,
) -> TensorAbi:
    """Mirror the public entrypoint's NHD->head-major normalization."""

    storage_pages = pages if shared else 2 * pages
    if kv_layout == "HND":
        shape = (storage_pages, num_kv_heads, page_size, stored_head_dim)
        strides = list(_contiguous_strides(shape))
        if stacked_shared:
            strides[0] *= 2
        return _tensor(
            shape,
            dtype,
            strides=tuple(strides),
            contiguous=not stacked_shared,
        )

    logical_shape = (storage_pages, page_size, num_kv_heads, stored_head_dim)
    logical_strides = list(_contiguous_strides(logical_shape))
    if stacked_shared:
        logical_strides[0] *= 2
    return _tensor(
        (storage_pages, num_kv_heads, page_size, stored_head_dim),
        dtype,
        strides=(
            logical_strides[0],
            logical_strides[2],
            logical_strides[1],
            logical_strides[3],
        ),
        contiguous=False,
    )


def _route_name(route: cake_api.CakeFmhaDecodeRoute | cake_api.CakeFmhaContextRoute | None) -> str:
    return "cake_fmha_compat_v1" if route is None else _ROUTE_NAMES[route.component]


def select_case(case: dict[str, Any]) -> str:
    batch_size = int(case["batch_size"])
    q_len = int(case["q_len"])
    kv_len = int(case["kv_len"])
    page_size = int(case["page_size"])
    num_q_heads = int(case["num_qo_heads"])
    num_kv_heads = int(case["num_kv_heads"])
    head_dim = int(case["head_dim"])
    shared = bool(case["uses_shared_paged_kv_idx"])
    kv_layout = str(case["kv_layout"])
    pages = batch_size * ((kv_len + page_size - 1) // page_size)

    q_shape = (batch_size * q_len, num_q_heads, head_dim)
    query = _tensor(
        q_shape,
        _TORCH_DTYPE[case["q_dtype"]],
        strides=(num_q_heads * 2 * head_dim, 2 * head_dim, 1)
        if case["non_contiguous_query"]
        else None,
        contiguous=not case["non_contiguous_query"],
    )
    out = _tensor(q_shape, _TORCH_DTYPE[case["o_dtype"]])
    stored_head_dim = head_dim // 2 if case["kv_dtype"] == "nvfp4" else head_dim
    key = _normalized_kv_tensor(
        pages=pages,
        num_kv_heads=num_kv_heads,
        page_size=page_size,
        stored_head_dim=stored_head_dim,
        kv_layout=kv_layout,
        shared=shared,
        stacked_shared=shared and case["kv_dtype"] != "nvfp4",
        dtype=_TORCH_DTYPE[case["kv_dtype"]],
    )
    value = TensorAbi(**{**key.__dict__, "pointer": 0x2000})
    max_pages = (kv_len + page_size - 1) // page_size
    block_shape = (batch_size, max_pages) if shared else (batch_size, 2, max_pages)
    block_tables = _tensor(block_shape, torch.int32)
    seq_lens = _tensor(
        (batch_size,),
        torch.int32,
        values=(kv_len,) * batch_size,
    )
    workspace = _tensor((1,), torch.uint8, numel_override=1 << 40)
    scale = _DEVICE_SCALE if case["kv_dtype"] in ("fp8", "nvfp4") else 1.0
    if case["enable_sink"]:
        sinks = _SINKS_BY_HEADS.get(num_q_heads)
        if sinks is None:
            sinks = torch.zeros((num_q_heads,), dtype=torch.float32)
            _SINKS_BY_HEADS[num_q_heads] = sinks
    else:
        sinks = None
    key_scales = value_scales = None
    if case["kv_dtype"] == "nvfp4":
        scale_pages = pages if shared else 2 * pages
        scale_shape = (scale_pages, num_kv_heads, page_size, head_dim // 16)
        key_scales = _tensor(scale_shape, torch.float8_e4m3fn)
        value_scales = TensorAbi(**{**key_scales.__dict__, "pointer": 0x3000})
    skip_threshold = 1e-30 if case["skip_softmax"] else None

    if case["mode"] == "decode":
        route = cake_api.select_cake_fmha_decode_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=value,
            out=out,
            workspace_buffer=workspace,
            block_tables=block_tables,
            seq_lens=seq_lens,
            batch_size=batch_size,
            q_len=q_len,
            max_seq_len=kv_len,
            window_left=case["window_left"],
            bmm1_scale=scale,
            bmm2_scale=scale,
            o_scale=1.0,
            sinks=sinks,
            kv_layout=kv_layout,
            uses_shared_paged_kv_idx=shared,
            cum_seq_lens_q=None,
            key_block_scales=key_scales,
            value_block_scales=value_scales,
            skip_softmax_threshold_scale_factor=skip_threshold,
            enable_block_sparse_attention=False,
            lse=None,
        )
    else:
        indptr = _tensor((batch_size + 1,), torch.int32)
        route = cake_api.select_cake_fmha_context_route(
            query.device,
            query=query,
            key_cache=key,
            value_cache=value,
            out=out,
            block_tables=block_tables,
            seq_lens=seq_lens,
            batch_size=batch_size,
            max_q_len=q_len,
            max_kv_len=kv_len,
            window_left=case["window_left"],
            bmm1_scale=scale,
            bmm2_scale=scale,
            sinks=sinks,
            uses_shared_paged_kv_idx=shared,
            cum_seq_lens_q=indptr,
            cum_seq_lens_kv=indptr,
            key_block_scales=key_scales,
            value_block_scales=value_scales,
            skip_softmax_threshold_scale_factor=skip_threshold,
            is_causal=case["causal"],
            lse=None,
            kv_layout=kv_layout,
            workspace_buffer=workspace,
        )
    return _route_name(route)


def replay_selectors() -> SelectorReplay:
    digest = hashlib.sha256()
    routes: Counter[str] = Counter()
    raw = valid = 0
    for case in chain(iter_decode_cases(), iter_context_cases()):
        raw += 1
        if upstream_skip_reason(case) is not None:
            continue
        valid += 1
        route = select_case(case)
        routes[route] += 1
        payload = [case[field] for field in _CASE_FIELDS]
        payload.append(route)
        digest.update(json.dumps(payload, separators=(",", ":")).encode())
        digest.update(b"\n")
    compat = routes["cake_fmha_compat_v1"]
    return SelectorReplay(
        raw_cases=raw,
        valid_cases=valid,
        optimized_cases=valid - compat,
        compat_cases=compat,
        route_counts=dict(sorted(routes.items())),
        digest=digest.hexdigest(),
    )


__all__ = [
    "PINNED_FLASHINFER_REVISION",
    "SelectorReplay",
    "iter_context_cases",
    "iter_decode_cases",
    "replay_selectors",
    "select_case",
    "upstream_skip_reason",
]
