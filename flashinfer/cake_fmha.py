"""Public Cake FMHA product entrypoints.

The conventional decode/context APIs are complete-domain Cake routes.  DCP
metadata selects an authenticated additive profile through the same
decode entrypoint and does not change ordinary-call behavior.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .jit.cake_fmha import (
    CakeFmhaTarget,
    get_cake_fmha_manifest,
    _is_cake_fmha_decode_native_bf16_available,
    load_cake_fmha_context_bf16_module,
    load_cake_fmha_context_fp8_module,
    load_cake_fmha_context_nvfp4_module,
    load_cake_fmha_context_fp16_hd256_module,
    load_cake_fmha_context_fp8_hd256_module,
    load_cake_fmha_compat_module,
    load_cake_fmha_decode_native_bf16_module,
    load_cake_fmha_decode_native_fp16_hd512_module,
    load_cake_fmha_decode_native_fp16_nhd_module,
    load_cake_fmha_decode_quant_bf16q_module,
    load_cake_fmha_decode_quant_fp8_module,
    load_cake_fmha_decode_quant_nvfp4_module,
)
from .jit.cake_fmha_request_ordered import (
    get_cake_fmha_request_ordered_manifest,
    load_cake_fmha_request_ordered_module,
)
from .utils import get_compute_capability


@dataclass(frozen=True)
class CakeFmhaRequestOrderedDecodePlan:
    """Immutable host-selected launch plan for graph-safe request ordering."""

    module_name: str
    batch_size: int
    q_len: int
    workspace_parts: int
    grid: tuple[int, int, int]
    total_tiles: int
    write_lse: bool
    num_q_heads: int = 8
    num_kv_heads: int = 1


def _request_ordered_plan_from_route(
    route: dict[str, Any], *, batch_size: int
) -> CakeFmhaRequestOrderedDecodePlan:
    plan = route["build_plan"]
    grid_x, grid_y, grid_z = (int(value) for value in plan["grid"])
    return CakeFmhaRequestOrderedDecodePlan(
        module_name=str(route["module_name"]),
        batch_size=batch_size,
        q_len=int(plan["q_len"]),
        workspace_parts=int(plan["workspace_parts"]),
        grid=(grid_x, grid_y, grid_z),
        total_tiles=int(plan["total_tiles"]),
        write_lse=bool(plan["write_lse"]),
        num_q_heads=int(plan.get("num_q_heads", 8)),
        num_kv_heads=int(plan.get("num_kv_heads", 1)),
    )


def _fallback_cake_fmha_request_ordered_plan(
    *,
    batch_size: int,
    q_len: int,
    write_lse: bool,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
) -> CakeFmhaRequestOrderedDecodePlan:
    if batch_size <= 0 or q_len not in (1, 6):
        raise ValueError(
            "request-ordered Cake FMHA requires a positive batch and q_len 1 or 6"
        )
    route_slug = (
        f"fallback_q{q_len}_ordered_s1_lse" if write_lse else f"fallback_q{q_len}"
    )
    matches = []
    for route in get_cake_fmha_request_ordered_manifest(num_q_heads, num_kv_heads)[
        "routes"
    ]:
        plan = route["build_plan"]
        if (
            plan["route_slug"] == route_slug
            and plan["q_len"] == q_len
            and plan["ordered"] is True
            and plan["num_split"] == 1
            and plan["workspace_parts"] == 1
            and plan["write_lse"] is write_lse
            and not plan["segmented_clc"]
            and not plan["static_one_tile"]
        ):
            matches.append(route)
    module_names = {route["module_name"] for route in matches}
    if len(module_names) != 1:
        raise RuntimeError(
            f"generated request-order fallback {route_slug!r} is not unique"
        )
    route = matches[0]
    return CakeFmhaRequestOrderedDecodePlan(
        module_name=str(route["module_name"]),
        batch_size=batch_size,
        q_len=q_len,
        workspace_parts=1,
        grid=(q_len, num_q_heads // 8, batch_size),
        total_tiles=batch_size * q_len * (num_q_heads // 8),
        write_lse=write_lse,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def plan_cake_fmha_request_ordered_paged_decode(
    kv_lens: Sequence[int],
    q_len: int,
    *,
    request_order_case: Literal["identity", "length_desc"] = "length_desc",
    real_batch_size: int | None = None,
    write_lse: bool = False,
    num_q_heads: int = 8,
    num_kv_heads: int = 1,
) -> CakeFmhaRequestOrderedDecodePlan:
    """Select an exported schedule from host-visible immutable metadata.

    The returned plan contains no device data.  Call this before CUDA Graph
    capture, then update the contents of the device ``request_order`` tensor
    in place between replays.
    """

    lengths = tuple(int(value) for value in kv_lens)
    if not lengths or any(value <= 0 for value in lengths):
        raise ValueError("kv_lens must contain one positive length per request")
    if q_len not in (1, 6):
        raise ValueError("request-ordered Cake FMHA requires q_len 1 or 6")
    batch_size = len(lengths)
    logical_batch = batch_size if real_batch_size is None else int(real_batch_size)
    if not 0 < logical_batch <= batch_size:
        raise ValueError("real_batch_size must be in [1, len(kv_lens)]")

    exact = []
    for route in get_cake_fmha_request_ordered_manifest(num_q_heads, num_kv_heads)[
        "routes"
    ]:
        args = route["args"]
        plan = route["build_plan"]
        if (
            tuple(args["q_lens"]) == (q_len,) * batch_size
            and tuple(args["kv_lens"]) == lengths
            and int(args["real_batch_size"]) == logical_batch
            and args["request_order_case"] == request_order_case
            and bool(args["provide_lse"]) is bool(write_lse)
            and plan["ordered"] is True
        ):
            exact.append(route)
    identities = {
        (
            route["module_name"],
            json.dumps(route["build_plan"], sort_keys=True, separators=(",", ":")),
        )
        for route in exact
    }
    if len(identities) == 1:
        return _request_ordered_plan_from_route(exact[0], batch_size=batch_size)
    return _fallback_cake_fmha_request_ordered_plan(
        batch_size=batch_size,
        q_len=q_len,
        write_lse=write_lse,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )


def _is_authenticated_request_ordered_plan(
    plan: CakeFmhaRequestOrderedDecodePlan,
) -> bool:
    """Return whether ``plan`` names an exported mutable-order route."""

    fallback = _fallback_cake_fmha_request_ordered_plan(
        batch_size=plan.batch_size,
        q_len=plan.q_len,
        write_lse=plan.write_lse,
        num_q_heads=plan.num_q_heads,
        num_kv_heads=plan.num_kv_heads,
    )
    if plan == fallback:
        return True
    for route in get_cake_fmha_request_ordered_manifest(
        plan.num_q_heads, plan.num_kv_heads
    )["routes"]:
        build_plan = route["build_plan"]
        if (
            build_plan["ordered"] is True
            and len(route["args"]["q_lens"]) == plan.batch_size
            and route["module_name"] == plan.module_name
            and int(build_plan["q_len"]) == plan.q_len
            and int(build_plan["workspace_parts"]) == plan.workspace_parts
            and tuple(int(value) for value in build_plan["grid"]) == plan.grid
            and int(build_plan["total_tiles"]) == plan.total_tiles
            and bool(build_plan["write_lse"]) is plan.write_lse
        ):
            return True
    return False


class CakeFmhaRequestOrderedCapture:
    """Explicit descriptor preparation for Q produced inside CUDA Graph capture.

    Construct with the selected plans outside capture, after warming each
    module's ordinary launch path. Allocate a distinct caller workspace for
    every live graph/layer binding before capture. Pass this object as
    ``request_order_capture`` when recording the public decode call, then call
    :meth:`finalize` after the graph context exits and before its first replay.
    Call :meth:`discard` if recording or finalization fails, and discard that
    graph. The object retains descriptor workspaces, not intermediate Q tensors;
    keep it alive with its graph. Ordinary prewarmed calls are unchanged.
    """

    def __init__(self, plans: Sequence[CakeFmhaRequestOrderedDecodePlan]) -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "create request-order capture preparation outside capture"
            )
        if not plans:
            raise ValueError(
                "request-order capture preparation requires selected plans"
            )
        self._plans = frozenset(plans)
        self._tokens: dict[str, tuple[Any, int]] = {}
        self._workspaces: dict[int, torch.Tensor] = {}
        self._state = "recording"
        try:
            for plan in plans:
                if not _is_authenticated_request_ordered_plan(plan):
                    raise ValueError(
                        "request-order capture plan is not an exported route"
                    )
                if plan.module_name not in self._tokens:
                    module = load_cake_fmha_request_ordered_module(plan.module_name)
                    self._tokens[plan.module_name] = (
                        module,
                        module.begin_tma_capture(),
                    )
        except BaseException:
            self.discard()
            raise

    @property
    def finalized(self) -> bool:
        return self._state == "finalized"

    def _record(
        self,
        plan: CakeFmhaRequestOrderedDecodePlan,
        workspace: torch.Tensor,
        arguments: tuple[Any, ...],
    ) -> None:
        if self._state != "recording" or plan not in self._plans:
            raise ValueError(
                "request-order capture is not recording this selected plan"
            )
        if not torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "request_order_capture is only used inside CUDA Graph capture"
            )
        module, token = self._tokens[plan.module_name]
        module.run_tma_capture(token, *arguments)
        self._workspaces[workspace.data_ptr()] = workspace

    def finalize(self) -> None:
        """Upload recorded tensor maps after capture, before publishing/replaying it."""
        if self._state != "recording":
            raise RuntimeError("request-order capture preparation has already finished")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("finalize request-order descriptors after capture ends")
        for name, (module, token) in tuple(self._tokens.items()):
            module.finalize_tma_capture(token)
            del self._tokens[name]
        self._state = "finalized"

    def discard(self) -> None:
        """Release unfinished host records; the associated graph must be discarded."""
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "discard request-order descriptor records after capture ends"
            )
        for name, (module, token) in tuple(self._tokens.items()):
            module.discard_tma_capture(token)
            del self._tokens[name]
        self._workspaces.clear()
        self._state = "discarded"


def _run_cake_fmha_request_ordered_paged_decode(
    *,
    backend: Literal["cake"],
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    lse: torch.Tensor | None,
    workspace_buffer: torch.Tensor,
    completion_buffer: torch.Tensor | None,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    request_order: torch.Tensor,
    max_seq_len: int,
    bmm1_scale_log2: torch.Tensor,
    bmm2_scale: torch.Tensor,
    uses_shared_paged_kv_idx: bool,
    plan: CakeFmhaRequestOrderedDecodePlan,
    capture: CakeFmhaRequestOrderedCapture | None = None,
) -> None:
    """Launch the explicitly selected Cake backend without auxiliary kernels.

    ``backend`` is the typed handoff from the public dispatcher; route admission
    has already rejected any other backend before entering this function.
    """

    if not query.is_cuda:
        raise ValueError("request-ordered Cake FMHA tensors must be on CUDA")
    if get_compute_capability(query.device) != (10, 3):
        raise RuntimeError("request-ordered Cake FMHA requires compute capability 10.3")
    sm_count = torch.cuda.get_device_properties(query.device).multi_processor_count
    if sm_count != 152:
        raise RuntimeError(
            f"request-ordered Cake FMHA requires a 152-SM device, got {sm_count}"
        )
    batch_size, q_len = plan.batch_size, plan.q_len
    if query.shape != (batch_size * q_len, plan.num_q_heads, 256):
        raise ValueError(
            "request-ordered Cake FMHA requires Q shape [batch*q_len, plan.num_q_heads, 256]"
        )
    if query.dtype != torch.bfloat16 or not query.is_contiguous():
        raise TypeError("request-ordered Cake FMHA requires contiguous BF16 Q")
    if (
        out.shape != query.shape
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
    ):
        raise TypeError(
            "request-ordered Cake FMHA requires contiguous BF16 O matching Q"
        )
    if (
        key_cache.ndim != 4
        or value_cache.ndim != 4
        or tuple(key_cache.shape[1:]) != (plan.num_kv_heads, 64, 256)
        or value_cache.shape != key_cache.shape
        or key_cache.dtype != torch.float8_e4m3fn
        or value_cache.dtype != torch.float8_e4m3fn
        or key_cache.stride(-1) != 1
        or value_cache.stride(-1) != 1
    ):
        raise TypeError(
            "request-ordered Cake FMHA requires FP8 E4M3 HND K/V with a "
            "contiguous head dimension and shape [pages, plan.num_kv_heads, 64, 256]"
        )
    if max_seq_len <= 0:
        raise ValueError("request-ordered Cake FMHA requires max_seq_len > 0")
    tensors = (
        key_cache,
        value_cache,
        out,
        workspace_buffer,
        block_tables,
        seq_lens,
        request_order,
        bmm1_scale_log2,
        bmm2_scale,
    )
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("all request-ordered Cake FMHA tensors must share Q's device")
    if (
        seq_lens.dtype != torch.int32
        or seq_lens.shape != (batch_size,)
        or not seq_lens.is_contiguous()
    ):
        raise TypeError("seq_lens must be contiguous CUDA int32 [batch]")
    if (
        request_order.dtype != torch.int32
        or request_order.shape != (batch_size,)
        or not request_order.is_contiguous()
    ):
        raise TypeError("request_order must be contiguous CUDA int32 [batch]")
    if block_tables.dtype != torch.int32 or not block_tables.is_contiguous():
        raise TypeError("block_tables must be contiguous CUDA int32")
    required_pages = 2 * (((max(1, (int(max_seq_len) + 127) // 128) + 1) // 2) * 2)
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            raise ValueError("shared block_tables must have shape [batch, pages]")
        pages = int(block_tables.shape[1])
        page_table_stride = pages
        page_table_v_offset = 0
        page_table_arg = block_tables
    else:
        if block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
            raise ValueError("separate block_tables must have shape [batch, 2, pages]")
        pages = int(block_tables.shape[2])
        page_table_stride = 2 * pages
        page_table_v_offset = pages
        page_table_arg = block_tables.view(batch_size, 2 * pages)
    if pages < required_pages:
        raise ValueError(
            f"block_tables needs at least {required_pages} page slots for max_seq_len"
        )
    if (
        workspace_buffer.dtype != torch.uint8
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.data_ptr() % 128
    ):
        raise TypeError(
            "workspace_buffer must be contiguous, 128-byte-aligned CUDA uint8"
        )
    for name, scale in (
        ("bmm1_scale_log2", bmm1_scale_log2),
        ("bmm2_scale", bmm2_scale),
    ):
        if (
            scale.dtype != torch.float32
            or scale.numel() < 1
            or not scale.is_contiguous()
        ):
            raise TypeError(f"{name} must be nonempty contiguous CUDA float32")
    if plan.write_lse:
        if (
            lse is None
            or lse.device != query.device
            or lse.dtype != torch.float32
            or lse.shape != query.shape[:-1]
            or not lse.is_contiguous()
        ):
            raise TypeError("the selected plan requires contiguous FP32 LSE")
        lse_arg = lse
    else:
        if lse is not None:
            raise ValueError("the selected plan does not write LSE")
        if workspace_buffer.numel() < 388:
            raise ValueError(
                "workspace_buffer is too small for the TMA and dummy LSE slots"
            )
        lse_arg = workspace_buffer[384:388].view(torch.float32)

    if workspace_buffer.numel() < 384:
        raise ValueError(
            "workspace_buffer needs at least 384 bytes for TMA descriptors"
        )
    if not _is_authenticated_request_ordered_plan(plan):
        raise ValueError("request_order_plan does not match an exported route")
    tma_workspace = workspace_buffer[:384]
    if plan.workspace_parts == 1:
        partial_o = out
        partial_lse = lse_arg
        completion = seq_lens.view(torch.uint32)
    else:
        completion_elems = batch_size * q_len * (plan.num_q_heads // 8)
        if (
            completion_buffer is None
            or completion_buffer.device != query.device
            or completion_buffer.dtype not in (torch.int32, torch.uint32)
            or not completion_buffer.is_contiguous()
            or completion_buffer.numel() < completion_elems
        ):
            raise ValueError(
                "split request-order plans require a zero-initialized contiguous "
                "int32/uint32 multi_ctas_kv_counter_buffer with batch*q_len*(num_q_heads/8) elements"
            )
        completion = completion_buffer[:completion_elems].view(torch.uint32)
        rows = batch_size * q_len * plan.num_q_heads * plan.workspace_parts
        partial_o_bytes = rows * 256 * 2
        partial_lse_bytes = rows * 4
        cursor = 32 * 1024 * 1024
        required = cursor + partial_o_bytes + partial_lse_bytes
        if workspace_buffer.numel() < required:
            raise ValueError(
                f"workspace_buffer needs at least {required} bytes for this plan"
            )
        partial_o = workspace_buffer[cursor : cursor + partial_o_bytes].view(
            torch.bfloat16
        )
        cursor += partial_o_bytes
        partial_lse = workspace_buffer[cursor : cursor + partial_lse_bytes].view(
            torch.float32
        )

    import tvm_ffi

    module = load_cake_fmha_request_ordered_module(plan.module_name)
    with tvm_ffi.use_torch_stream():
        arguments = (
            query,
            key_cache.view(torch.uint8),
            value_cache.view(torch.uint8),
            partial_o,
            partial_lse,
            completion,
            out,
            lse_arg,
            page_table_arg,
            seq_lens,
            request_order,
            page_table_stride,
            page_table_v_offset,
            0.0,
            0.0,
            bmm1_scale_log2,
            bmm2_scale,
            1,
            plan.num_q_heads,
            plan.num_kv_heads,
            batch_size,
            plan.total_tiles,
            tma_workspace,
            *plan.grid,
        )
        if capture is None:
            module.run(*arguments)
        else:
            capture._record(plan, workspace_buffer, arguments)


@dataclass(frozen=True)
class CakeFmhaDecodeRoute:
    """One exact manifest-backed optimized decode specialization."""

    target: CakeFmhaTarget
    batch_size: int
    q_len: int
    num_q_heads: int
    num_kv_heads: int
    has_sink: bool
    has_window: bool
    use_scale_ptr: bool
    retain_kv_l2: bool
    component: Literal[
        "decode_native_bf16",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8_reduce",
        "decode_quant_fp8",
        "decode_quant_nvfp4",
    ] = "decode_native_bf16"
    page_size: int = 16


CakeFmhaContextComponent = Literal[
    "context_bf16",
    "context_fp16_hd256",
    "context_fp8",
    "context_fp8_hd256",
    "context_nvfp4",
]


@dataclass(frozen=True)
class CakeFmhaContextRoute:
    """One exact manifest-backed optimized context specialization."""

    target: CakeFmhaTarget
    component: CakeFmhaContextComponent
    num_m_blocks: int
    num_q_heads: int
    num_kv_heads: int
    pack_g: int
    page_size: int
    l2_swizzle: int
    is_causal: bool
    return_lse: bool
    enable_sink: bool
    exact_profile: Literal["q511", "q257"] | None = None


# These are the exact optimized routes in the pinned 57,280-cell matrix.  Keep
# the component sequences explicit: hd256 needs its staging/scatter support,
# and split-KV NVFP4 decode may also need the shared reduction component.
_PRODUCT_ROUTE_COMPONENTS: dict[str, tuple[str, ...]] = {
    "ctx_bf16_hnd_hd128_hgpack_03df_v2": ("context_bf16",),
    "ctx_fp16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp16_hd256",
    ),
    "ctx_fp8_bf16_nhd_hd256_stage16_v1": (
        "context_hd256_support",
        "context_fp8_hd256",
    ),
    "ctx_fp8_hnd_hd128_hgpack_48b5_v1": ("context_fp8",),
    "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1": ("context_nvfp4",),
    "decode_native_bf16_v1_bece": ("decode_native_bf16",),
    "decode_native_fp16_hd512_v1_66b1": ("decode_native_fp16_hd512",),
    "decode_native_fp16_nhd_v1_f32d": ("decode_native_fp16_nhd",),
    "decode_quantized_bf16q_9d8b_v1": ("decode_quant_bf16q",),
    "decode_quantized_fp8_8e5b_v1": (
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
    ),
    "decode_quantized_nvfp4_8e5b_v1": (
        "decode_quant_nvfp4",
        "decode_quant_fp8_reduce",
    ),
}

# These components have an authenticated FlashInfer TVM-FFI adapter in the
# checked-in package.  A route remains fail-closed until every component in its
# declared chain is present here and covered by the adapter digest.
_AUTHENTICATED_JIT_COMPONENTS = frozenset(
    {
        "compat_v1",
        "context_bf16",
        "context_fp16_hd256",
        "context_fp8",
        "context_fp8_hd256",
        "context_hd256_support",
        "context_nvfp4",
        "decode_native_bf16",
        "decode_native_fp16_hd512",
        "decode_native_fp16_nhd",
        "decode_quant_bf16q",
        "decode_quant_fp8",
        "decode_quant_fp8_reduce",
        "decode_quant_nvfp4",
    }
)


def _route_components(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute,
) -> tuple[str, ...]:
    if isinstance(route, CakeFmhaDecodeRoute):
        route_name = {
            "decode_native_bf16": "decode_native_bf16_v1_bece",
            "decode_native_fp16_hd512": "decode_native_fp16_hd512_v1_66b1",
            "decode_native_fp16_nhd": "decode_native_fp16_nhd_v1_f32d",
            "decode_quant_bf16q": "decode_quantized_bf16q_9d8b_v1",
            "decode_quant_fp8": "decode_quantized_fp8_8e5b_v1",
            "decode_quant_nvfp4": "decode_quantized_nvfp4_8e5b_v1",
        }[route.component]
    else:
        route_name = {
            "context_bf16": "ctx_bf16_hnd_hd128_hgpack_03df_v2",
            "context_fp16_hd256": "ctx_fp16_nhd_hd256_stage16_v1",
            "context_fp8": "ctx_fp8_hnd_hd128_hgpack_48b5_v1",
            "context_fp8_hd256": "ctx_fp8_bf16_nhd_hd256_stage16_v1",
            "context_nvfp4": "ctx_nvfp4_hnd_hd128_dequant_fp8_hg_v1",
        }[route.component]
    return _PRODUCT_ROUTE_COMPONENTS[route_name]


def cake_fmha_route_is_optimized(
    route: CakeFmhaDecodeRoute | CakeFmhaContextRoute | None,
) -> bool:
    """Return whether ``route`` has a fully authenticated runnable adapter."""

    if route is None:
        return False
    if (
        isinstance(route, CakeFmhaDecodeRoute)
        and route.component == "decode_native_bf16"
        and not _is_cake_fmha_decode_native_bf16_available(
            route.target,
            route.batch_size,
            route.q_len,
            route.num_q_heads,
            route.num_kv_heads,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
        )
    ):
        return False
    return all(
        component in _AUTHENTICATED_JIT_COMPONENTS
        for component in _route_components(route)
    )


def _tma_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether the paged-KV view can be represented by our TMA maps."""

    if tensor.stride(3) != 1:
        return False
    element_size = tensor.element_size()
    return all(
        stride > 0 and stride * element_size % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_paged_kv_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether packed HND NVFP4 KV is exactly TMA-encodable."""

    return tensor.stride(3) == 1 and all(
        stride > 0 and stride % 16 == 0 for stride in tensor.stride()[:3]
    )


def _tma_nvfp4_scale_strides_supported(tensor: torch.Tensor) -> bool:
    """Return whether HND E4M3 block scales are exactly TMA-encodable."""

    return (
        tensor.stride(3) == 1
        and tensor.stride(2) == 8
        and tensor.stride(1) > 0
        and tensor.stride(0) > 0
        and tensor.stride(1) % 16 == 0
        and tensor.stride(0) % 16 == 0
    )


def _decode_native_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    max_seq_len: int,
    pages_per_block: int,
    page_table_rows: int,
    lse: torch.Tensor | None,
) -> bool:
    """Mirror one native binding's resolved metadata/LSE workspace layout."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    even_kv_blocks = max(4, even_kv_blocks)
    required_pages = even_kv_blocks * pages_per_block
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(required_pages, source_pages)
    if pages_per_block == 4:
        padded_pages = (padded_pages + 3) // 4 * 4
    needs_page_padding = source_pages != padded_pages

    cursor = (batch_size * 4 + 15) // 16 * 16
    if needs_page_padding:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    if lse is None:
        cursor += query.shape[0] * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Mirror the BF16Q adapter's padding, scale, and partial workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128 * 2
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    cursor += batch_size * query.shape[1] * 128 * 4
    cursor += 2 * batch_size * query.shape[1] * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_scale_supported(
    scale: float | torch.Tensor, query: torch.Tensor
) -> bool:
    return not isinstance(scale, torch.Tensor) or (
        scale.device == query.device
        and scale.dtype == torch.float32
        and scale.numel() == 1
        and scale.is_contiguous()
    )


def _pinned_noop_skip_softmax_supported(scale_factor: float | None) -> bool:
    """Accept the pinned matrix's numerically inert skip-softmax probe."""

    return scale_factor in (None, 0.0, 1e-30)


def _decode_quant_fp8_seq_lens_supported(
    seq_lens: torch.Tensor, max_seq_len: int
) -> bool:
    """Resolve the exact full-block/even-bucket route domain, or fail closed."""

    if seq_lens.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return False
    try:
        lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
    except (RuntimeError, TypeError, ValueError):
        return False
    if not lengths or max(lengths) != max_seq_len:
        return False
    if any(length < 512 or length % 128 for length in lengths):
        return False
    evened_buckets = {((length + 127) // 128 + 1) // 2 * 2 for length in lengths}
    return len(evened_buckets) == 1


def _decode_quant_fp8_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    page_size: int,
    max_seq_len: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Check a conservative upper bound for runtime split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    if not query.is_contiguous():
        cursor += batch_size * query.shape[1] * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _decode_quant_nvfp4_seq_lens_supported(
    seq_lens: torch.Tensor, max_seq_len: int
) -> bool:
    """Resolve NVFP4 runtime split inputs on the host, or fail closed."""

    if seq_lens.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return False
    try:
        lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
    except (RuntimeError, TypeError, ValueError):
        return False
    return bool(lengths) and min(lengths) > 0 and max(lengths) == max_seq_len


def _decode_quant_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor,
    query: torch.Tensor,
    block_tables: torch.Tensor,
    *,
    batch_size: int,
    num_kv_heads: int,
    page_size: int,
    max_seq_len: int,
    page_table_rows: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
) -> bool:
    """Conservatively bound NVFP4 GQA padding and split-KV workspace."""

    if not workspace_buffer.is_contiguous() or workspace_buffer.device != query.device:
        return False
    even_kv_blocks = (max_seq_len + 127) // 128
    even_kv_blocks += even_kv_blocks % 2
    max_splits = max(1, (even_kv_blocks + 3) // 4)
    required_pages = even_kv_blocks * (128 // page_size)
    source_pages = int(block_tables.shape[-1])
    padded_pages = max(source_pages, required_pages)

    cursor = 0
    if not isinstance(bmm1_scale, torch.Tensor):
        cursor += 4
    if not isinstance(bmm2_scale, torch.Tensor):
        cursor += 4
    cursor = (cursor + 15) // 16 * 16
    group_size = query.shape[1] // num_kv_heads
    if group_size != 8 or not query.is_contiguous():
        cursor += batch_size * num_kv_heads * 8 * 128
        cursor = (cursor + 15) // 16 * 16
    if source_pages < padded_pages:
        cursor += batch_size * page_table_rows * padded_pages * 4
        cursor = (cursor + 15) // 16 * 16
    partial_rows = batch_size * query.shape[1] * max_splits
    cursor += partial_rows * 128 * 4
    cursor += 2 * partial_rows * 4
    workspace_bytes = workspace_buffer.numel() * workspace_buffer.element_size()
    return workspace_bytes >= cursor


def _manifest_optimized_route_accounting() -> tuple[int, int]:
    """Return registered/total optimized counts recorded by the manifest.

    This is a route-name drift check, not selector parity.  Exact selector
    coverage requires replaying the independent pinned capability corpus.
    """

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    registered = sum(route_counts.get(name, 0) for name in _PRODUCT_ROUTE_COMPONENTS)
    return registered, total


def _manifest_authenticated_route_accounting() -> tuple[int, int]:
    """Return runnable/total optimized counts recorded by the manifest."""

    route_counts = get_cake_fmha_manifest()["capability"]["route_counts"]
    total = sum(
        count for name, count in route_counts.items() if name != "cake_fmha_compat_v1"
    )
    authenticated = sum(
        route_counts.get(route_name, 0)
        for route_name, components in _PRODUCT_ROUTE_COMPONENTS.items()
        if all(component in _AUTHENTICATED_JIT_COMPONENTS for component in components)
    )
    return authenticated, total


def _cake_fmha_target(device: torch.device) -> CakeFmhaTarget:
    """Resolve the exact Blackwell cubin target without cross-arch fallback."""

    from .jit.cpp_ext import is_cuda_version_at_least

    capability = get_compute_capability(device)
    if capability == (10, 0):
        if not is_cuda_version_at_least("12.8"):
            raise RuntimeError("Cake FMHA on B200 requires CUDA 12.8 or newer")
        return "sm100a"
    if capability == (10, 3):
        if not is_cuda_version_at_least("12.9"):
            raise RuntimeError("Cake FMHA on B300 requires CUDA 12.9 or newer")
        return "sm103a"
    raise RuntimeError(
        "Cake FMHA requires compute capability 10.0 (B200/GB200) or "
        f"10.3 (B300/GB300), got {capability[0]}.{capability[1]}"
    )


def get_cake_fmha_module(device: torch.device):
    """Load the authenticated complete-domain compatibility module."""

    target = _cake_fmha_target(device)
    return load_cake_fmha_compat_module(target)


def select_cake_fmha_decode_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    workspace_buffer: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    q_len: int | None,
    max_seq_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    o_scale: float | None,
    sinks: torch.Tensor | None,
    kv_layout: str,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor | None,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    enable_block_sparse_attention: bool,
    lse: torch.Tensor | None = None,
) -> CakeFmhaDecodeRoute | None:
    """Select an exact product decode route without broadening its contract."""

    if q_len is None or q_len <= 0 or batch_size <= 0 or max_seq_len <= 0:
        return None
    if cum_seq_lens_q is not None or enable_block_sparse_attention:
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if any(
        tensor.device != query.device
        for tensor in (
            key_cache,
            value_cache,
            out,
            workspace_buffer,
            block_tables,
            seq_lens,
        )
    ):
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    if query.shape[0] != batch_size * q_len:
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if (
        query.stride(0) <= 0
        or query.stride(1) <= 0
        or query.stride(0) != num_q_heads * query.stride(1)
    ):
        return None
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if not 1 <= num_q_heads // num_kv_heads <= 8:
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if not block_tables.is_contiguous():
        return None
    if uses_shared_paged_kv_idx:
        if block_tables.ndim != 2 or block_tables.shape[0] != batch_size:
            return None
    elif block_tables.ndim != 3 or block_tables.shape[:2] != (batch_size, 2):
        return None
    if (
        seq_lens.ndim != 1
        or seq_lens.shape[0] != batch_size
        or not seq_lens.is_contiguous()
    ):
        return None
    if seq_lens.dtype not in (torch.int32, torch.uint32):
        return None
    if sinks is not None and (
        not isinstance(sinks, torch.Tensor)
        or sinks.device != query.device
        or sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if lse is not None and (
        lse.device != query.device
        or lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
        or lse.stride() != (num_q_heads, 1)
    ):
        return None

    page_size = int(key_cache.shape[2])
    local_blocks = max(1, (max_seq_len + 127) // 128)

    def route(component, *, selected_page_size: int = page_size):
        candidate = CakeFmhaDecodeRoute(
            target=_cake_fmha_target(device),
            batch_size=batch_size,
            q_len=q_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            has_sink=sinks is not None,
            has_window=window_left >= 0,
            use_scale_ptr=isinstance(bmm1_scale, torch.Tensor),
            retain_kv_l2=local_blocks <= 9,
            component=component,
            page_size=selected_page_size,
        )
        exact_sink_no_lse = (
            component == "decode_native_bf16"
            and candidate.batch_size == 256
            and candidate.q_len == 1
            and candidate.num_q_heads == 32
            and candidate.num_kv_heads == 4
            and candidate.has_sink
            and not candidate.has_window
            and not candidate.use_scale_ptr
            and not candidate.retain_kv_l2
        )
        if exact_sink_no_lse and lse is not None:
            return None
        if component == "decode_native_bf16" and not cake_fmha_route_is_optimized(
            candidate
        ):
            return None
        return candidate

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None
    if dtypes == (torch.bfloat16,) * 4:
        if (
            query.is_contiguous()
            and query.shape[2] == 128
            and key_cache.shape[2:] == (16, 128)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=8,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_bf16", selected_page_size=16)
        return None

    if dtypes == (torch.float16,) * 4:
        if (
            query.shape[2] == 128
            and key_cache.shape[2:] == (32, 128)
            and kv_layout == "NHD"
            and not uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=4,
                page_table_rows=2,
                lse=lse,
            )
            and no_block_scales
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_nhd", selected_page_size=32)
        if (
            query.shape[2] == 512
            and query.is_contiguous()
            and key_cache.shape[2:] == (64, 512)
            and kv_layout == "HND"
            and uses_shared_paged_kv_idx
            and _tma_paged_kv_strides_supported(key_cache)
            and _tma_paged_kv_strides_supported(value_cache)
            and _decode_native_workspace_supported(
                workspace_buffer,
                query,
                block_tables,
                batch_size=batch_size,
                max_seq_len=max_seq_len,
                pages_per_block=2,
                page_table_rows=1,
                lse=lse,
            )
            and no_block_scales
            and sinks is None
            and not isinstance(bmm2_scale, torch.Tensor)
            and float(bmm2_scale) == 1.0
            and (o_scale is None or float(o_scale) == 1.0)
        ):
            return route("decode_native_fp16_hd512", selected_page_size=64)
        return None

    quant_extensions_absent = (
        sinks is None
        and lse is None
        and window_left == -1
        and skip_softmax_threshold_scale_factor in (None, 0.0)
    )
    if (
        dtypes == (torch.float8_e4m3fn,) * 4
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and num_q_heads // num_kv_heads == 8
        and no_block_scales
        and quant_extensions_absent
        and max_seq_len >= 512
        and max_seq_len % 128 == 0
        and _decode_quant_fp8_seq_lens_supported(seq_lens, max_seq_len)
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_fp8_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            page_size=page_size,
            max_seq_len=max_seq_len,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_fp8")

    if (
        dtypes
        == (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and q_len == 1
        and query.shape[2] == 128
        and page_size in (16, 32)
        and kv_layout in ("HND", "NHD")
        and num_q_heads % num_kv_heads == 0
        and 1 <= num_q_heads // num_kv_heads < 8
        and no_block_scales
        and quant_extensions_absent
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_bf16q")

    if (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and q_len == 1
        and query.shape[2] == 128
        and key_cache.shape[3] == 64
        and page_size in (16, 32)
        and kv_layout == "HND"
        and quant_extensions_absent
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.ndim == 4
        and value_block_scales.shape == key_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.stride(3) == 1
        and value_block_scales.stride(3) == 1
        and key_block_scales.device == query.device
        and value_block_scales.device == query.device
        and num_q_heads // num_kv_heads <= 8
        and _decode_quant_nvfp4_seq_lens_supported(seq_lens, max_seq_len)
        and _tma_nvfp4_paged_kv_strides_supported(key_cache)
        and _tma_nvfp4_paged_kv_strides_supported(value_cache)
        and _tma_nvfp4_scale_strides_supported(key_block_scales)
        and _tma_nvfp4_scale_strides_supported(value_block_scales)
        and query.data_ptr() % 16 == 0
        and key_cache.data_ptr() % 16 == 0
        and value_cache.data_ptr() % 16 == 0
        and key_block_scales.data_ptr() % 16 == 0
        and value_block_scales.data_ptr() % 16 == 0
        and out.data_ptr() % 16 == 0
        and _decode_quant_scale_supported(bmm1_scale, query)
        and _decode_quant_scale_supported(bmm2_scale, query)
        and _decode_quant_nvfp4_workspace_supported(
            workspace_buffer,
            query,
            block_tables,
            batch_size=batch_size,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            max_seq_len=max_seq_len,
            page_table_rows=1 if uses_shared_paged_kv_idx else 2,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
    ):
        return route("decode_quant_nvfp4")
    return None


def _resolve_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
) -> tuple[Any, bool]:
    """Resolve a decode module and whether its optimized ABI remains active."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device)), False
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA decode route target does not match the device")
    loader = {
        "decode_native_bf16": load_cake_fmha_decode_native_bf16_module,
        "decode_native_fp16_hd512": load_cake_fmha_decode_native_fp16_hd512_module,
        "decode_native_fp16_nhd": load_cake_fmha_decode_native_fp16_nhd_module,
        "decode_quant_bf16q": load_cake_fmha_decode_quant_bf16q_module,
        "decode_quant_fp8": load_cake_fmha_decode_quant_fp8_module,
        "decode_quant_nvfp4": load_cake_fmha_decode_quant_nvfp4_module,
    }.get(route.component)
    if loader is None:
        raise RuntimeError(
            f"Cake FMHA decode route has no authenticated loader: {route.component}"
        )
    common_args = (
        route.target,
        route.batch_size,
        route.q_len,
        route.num_q_heads,
        route.num_kv_heads,
    )
    if route.component == "decode_native_fp16_hd512":
        return (
            loader(
                *common_args,
                has_window=route.has_window,
                use_scale_ptr=route.use_scale_ptr,
                retain_kv_l2=route.retain_kv_l2,
            ),
            True,
        )
    if route.component == "decode_quant_bf16q":
        return loader(*common_args, route.page_size), True
    if route.component == "decode_quant_fp8":
        return loader(*common_args, route.page_size, full_blocks=True), True
    if route.component == "decode_quant_nvfp4":
        try:
            return loader(*common_args, route.page_size), True
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA portable NVFP4 loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target), False
    return (
        loader(
            *common_args,
            has_sink=route.has_sink,
            has_window=route.has_window,
            use_scale_ptr=route.use_scale_ptr,
            retain_kv_l2=route.retain_kv_l2,
        ),
        True,
    )


def get_cake_fmha_decode_module(
    device: torch.device, route: CakeFmhaDecodeRoute | None
):
    """Load an optimized decode module, or the authenticated portable fallback."""

    module, _ = _resolve_cake_fmha_decode_module(device, route)
    return module


def _context_tile_mma_work(q_len: int, kv_len: int, tokens_per_tile: int) -> int:
    """Mirror the standalone route's bottom-right causal tile-work model."""

    total = 0
    full_n_blocks = (kv_len + 127) // 128
    shift = kv_len - q_len
    for m_block in range((q_len + tokens_per_tile - 1) // tokens_per_tile):
        max_n = (m_block + 1) * tokens_per_tile + shift
        total += (max_n + 127) // 128 if max_n < kv_len else full_n_blocks
    return total


def _context_pack_g(
    max_q_len: int, max_kv_len: int, num_q_heads: int, num_kv_heads: int
) -> int:
    """Choose the canonical packed-GQA axis from host-visible maxima."""

    group = num_q_heads // num_kv_heads
    if group <= 1 or group > 128:
        return 1
    unpacked = num_q_heads * _context_tile_mma_work(max_q_len, max_kv_len, 256)
    packed = num_kv_heads * _context_tile_mma_work(
        max_q_len, max_kv_len, 2 * (128 // group)
    )
    return group if packed < unpacked else 1


def _context_hd256_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    query: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    fp8: bool,
) -> bool:
    """Mirror the HD256 binding's packed Q/K/V/O and metadata arena."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != query.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False

    padded_q = ((max_q_len + 127) // 128) * 128
    max_micro_pages = (max_kv_len + 15) // 16
    total_q_rows = batch_size * num_q_heads * padded_q
    total_micro_pages = batch_size * num_kv_heads * max_micro_pages
    input_bytes = 256 if fp8 else 512

    cursor = total_q_rows * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_micro_pages * 16 * input_bytes
    cursor = (cursor + 15) // 16 * 16
    cursor += total_q_rows * 512
    cursor = (cursor + 15) // 16 * 16
    cursor += 3 * batch_size * num_q_heads * 4
    cursor += total_micro_pages * 4
    return workspace_buffer.numel() * workspace_buffer.element_size() >= cursor


def _context_nvfp4_workspace_supported(
    workspace_buffer: torch.Tensor | None,
    key_cache: torch.Tensor,
    *,
    batch_size: int,
    num_q_heads: int,
    pack_g: int,
) -> bool:
    """Bound the fused kernel's expanded-metadata workspace exactly."""

    if (
        workspace_buffer is None
        or not workspace_buffer.is_contiguous()
        or workspace_buffer.device != key_cache.device
        or workspace_buffer.data_ptr() % 16
    ):
        return False
    total_bh = batch_size * (num_q_heads // pack_g)
    seq_kv_offset = ((total_bh * 4 + 15) // 16) * 16
    required = ((seq_kv_offset + 2 * total_bh * 4 + 15) // 16) * 16
    return workspace_buffer.numel() * workspace_buffer.element_size() >= required


def _context_bf16_exact_profile(
    query: torch.Tensor,
    seq_lens: torch.Tensor,
    cum_seq_lens_q: torch.Tensor,
    *,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    num_q_heads: int,
    num_kv_heads: int,
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    is_causal: bool,
    return_lse: bool,
    enable_sink: bool,
    kv_layout: str,
) -> Literal["q511", "q257"] | None:
    """Resolve the two measured mask-loop bodies from exact runtime lengths."""

    common = (
        batch_size == 4
        and num_q_heads == 10
        and num_kv_heads == 2
        and is_causal
        and not return_lse
        and not enable_sink
        and kv_layout == "HND"
        and seq_lens.device == query.device
        and cum_seq_lens_q.device == query.device
    )
    if not common:
        return None
    profile: Literal["q511", "q257"]
    if (
        max_q_len == 511
        and max_kv_len == 2047
        and page_size == 32
        and uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 511
    ):
        profile = "q511"
        expected_q_len = 511
        expected_kv_len = 2047
    elif (
        max_q_len == 257
        and max_kv_len == 1024
        and page_size == 1024
        and not uses_shared_paged_kv_idx
        and query.shape[0] == 4 * 257
    ):
        profile = "q257"
        expected_q_len = 257
        expected_kv_len = 1024
    else:
        return None
    if query.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return None
    try:
        kv_lengths = [int(value) for value in seq_lens.detach().cpu().tolist()]
        q_indptr = [int(value) for value in cum_seq_lens_q.detach().cpu().tolist()]
    except RuntimeError:
        return None
    if kv_lengths != [expected_kv_len] * batch_size:
        return None
    if q_indptr != [index * expected_q_len for index in range(batch_size + 1)]:
        return None
    return profile


def select_cake_fmha_context_route(
    device: torch.device,
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    out: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_size: int,
    max_q_len: int,
    max_kv_len: int,
    window_left: int,
    bmm1_scale: float | torch.Tensor,
    bmm2_scale: float | torch.Tensor,
    sinks: torch.Tensor | None,
    uses_shared_paged_kv_idx: bool,
    cum_seq_lens_q: torch.Tensor,
    cum_seq_lens_kv: torch.Tensor,
    key_block_scales: torch.Tensor | None,
    value_block_scales: torch.Tensor | None,
    skip_softmax_threshold_scale_factor: float | None,
    is_causal: bool,
    lse: torch.Tensor | None,
    kv_layout: str = "HND",
    workspace_buffer: torch.Tensor | None = None,
) -> CakeFmhaContextRoute | None:
    """Select an exact product context route without broadening its contract."""

    if batch_size <= 0 or max_q_len <= 0 or max_kv_len <= 0 or window_left != -1:
        return None
    if not _decode_quant_scale_supported(
        bmm1_scale, query
    ) or not _decode_quant_scale_supported(bmm2_scale, query):
        return None
    if not _pinned_noop_skip_softmax_supported(skip_softmax_threshold_scale_factor):
        return None
    if query.ndim != 3 or query.stride(2) != 1:
        return None
    if out.shape != query.shape or not out.is_contiguous():
        return None
    if key_cache.ndim != 4 or value_cache.ndim != 4:
        return None
    if key_cache.shape != value_cache.shape:
        return None
    if key_cache.stride(3) != 1 or value_cache.stride(3) != 1:
        return None
    page_size = int(key_cache.shape[2])
    if page_size not in (16, 32, 64, 128, 256, 512, 1024):
        return None
    num_q_heads = int(query.shape[1])
    num_kv_heads = int(key_cache.shape[1])
    if num_q_heads <= 0 or num_kv_heads <= 0 or num_q_heads % num_kv_heads:
        return None
    if seq_lens.ndim != 1 or seq_lens.shape[0] != batch_size:
        return None
    if seq_lens.dtype != torch.int32:
        return None
    if not seq_lens.is_contiguous():
        return None
    for indptr in (cum_seq_lens_q, cum_seq_lens_kv):
        if (
            indptr.ndim != 1
            or indptr.shape[0] != batch_size + 1
            or indptr.dtype != torch.int32
            or not indptr.is_contiguous()
        ):
            return None
    if sinks is not None and (
        sinks.dtype != torch.float32
        or sinks.numel() != num_q_heads
        or not sinks.is_contiguous()
    ):
        return None
    if sinks is not None and lse is not None:
        return None
    if lse is not None and (
        lse.dtype != torch.float32
        or lse.shape != (query.shape[0], num_q_heads)
        or not lse.is_contiguous()
    ):
        return None
    if block_tables.dtype not in (torch.int32, torch.uint32):
        return None
    if uses_shared_paged_kv_idx:
        if (
            block_tables.ndim != 2
            or block_tables.shape[0] != batch_size
            or block_tables.stride(1) != 1
        ):
            return None
    elif (
        block_tables.ndim != 3
        or block_tables.shape[:2] != (batch_size, 2)
        or block_tables.stride(2) != 1
    ):
        return None

    dtypes = (query.dtype, key_cache.dtype, value_cache.dtype, out.dtype)
    no_block_scales = key_block_scales is None and value_block_scales is None
    host_scalar_scales = not isinstance(bmm1_scale, torch.Tensor) and not isinstance(
        bmm2_scale, torch.Tensor
    )
    component: CakeFmhaContextComponent | None = None
    if (
        dtypes == (torch.bfloat16,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
        and host_scalar_scales
        and float(bmm2_scale) == 1.0
    ):
        component = "context_bf16"
    elif (
        dtypes == (torch.float8_e4m3fn,) * 4
        and query.shape[2] == 128
        and key_cache.shape[3] == 128
        and kv_layout in ("HND", "NHD")
        and _tma_paged_kv_strides_supported(key_cache)
        and _tma_paged_kv_strides_supported(value_cache)
        and no_block_scales
    ):
        component = "context_fp8"
    elif (
        dtypes == (torch.float16,) * 4
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and host_scalar_scales
        and not is_causal
        and sinks is None
        and lse is None
        and float(bmm2_scale) == 1.0
    ):
        component = "context_fp16_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.float8_e4m3fn,
            torch.bfloat16,
        )
        and query.shape[2] == 256
        and key_cache.shape[3] == 256
        and kv_layout == "NHD"
        and not uses_shared_paged_kv_idx
        and no_block_scales
        and is_causal
        and sinks is None
        and lse is None
    ):
        component = "context_fp8_hd256"
    elif (
        dtypes
        == (
            torch.float8_e4m3fn,
            torch.uint8,
            torch.uint8,
            torch.float8_e4m3fn,
        )
        and query.shape[2] == 128
        and key_cache.shape[2:] == (16, 64)
        and key_cache.is_contiguous()
        and value_cache.is_contiguous()
        and kv_layout == "HND"
        and uses_shared_paged_kv_idx
        and is_causal
        and sinks is None
        and lse is None
        and skip_softmax_threshold_scale_factor in (None, 0.0)
        and key_block_scales is not None
        and value_block_scales is not None
        and key_block_scales.dtype == torch.float8_e4m3fn
        and value_block_scales.dtype == torch.float8_e4m3fn
        and key_block_scales.shape == value_block_scales.shape
        and key_block_scales.shape[:3] == key_cache.shape[:3]
        and key_block_scales.shape[3] == 8
        and key_block_scales.is_contiguous()
        and value_block_scales.is_contiguous()
    ):
        component = "context_nvfp4"
    if component is None:
        return None

    exact_profile = None
    if component == "context_bf16":
        exact_profile = _context_bf16_exact_profile(
            query,
            seq_lens,
            cum_seq_lens_q,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            page_size=page_size,
            uses_shared_paged_kv_idx=uses_shared_paged_kv_idx,
            is_causal=is_causal,
            return_lse=lse is not None,
            enable_sink=sinks is not None,
            kv_layout=kv_layout,
        )
    if exact_profile == "q511":
        pack_g = 5
        num_m_blocks = 11
        l2_swizzle = 1
    elif exact_profile == "q257":
        pack_g = 5
        num_m_blocks = 6
        l2_swizzle = 8
    elif component in ("context_fp16_hd256", "context_fp8_hd256"):
        pack_g = 1
        num_m_blocks = (max_q_len + 127) // 128
        l2_swizzle = 1
    else:
        pack_g = _context_pack_g(max_q_len, max_kv_len, num_q_heads, num_kv_heads)
        tok_per_stage = 128 // pack_g
        num_m_blocks = (max_q_len + 2 * tok_per_stage - 1) // (2 * tok_per_stage)
        total_bh = batch_size * (num_q_heads // pack_g)
        l2_swizzle = 8 if total_bh % 8 == 0 else 1
    if component == "context_nvfp4" and not _context_nvfp4_workspace_supported(
        workspace_buffer,
        key_cache,
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        pack_g=pack_g,
    ):
        return None
    if component in ("context_fp16_hd256", "context_fp8_hd256") and not (
        _context_hd256_workspace_supported(
            workspace_buffer,
            query,
            batch_size=batch_size,
            max_q_len=max_q_len,
            max_kv_len=max_kv_len,
            num_q_heads=num_q_heads,
            num_kv_heads=num_kv_heads,
            fp8=component == "context_fp8_hd256",
        )
    ):
        return None
    return CakeFmhaContextRoute(
        target=_cake_fmha_target(device),
        component=component,
        num_m_blocks=num_m_blocks,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        pack_g=pack_g,
        page_size=page_size,
        l2_swizzle=l2_swizzle,
        is_causal=is_causal,
        return_lse=lse is not None,
        enable_sink=sinks is not None,
        exact_profile=exact_profile,
    )


def get_cake_fmha_context_module(
    device: torch.device, route: CakeFmhaContextRoute | None
):
    """Load an optimized context module, or the authenticated portable fallback."""

    if route is None or not cake_fmha_route_is_optimized(route):
        return load_cake_fmha_compat_module(_cake_fmha_target(device))
    if route.target != _cake_fmha_target(device):
        raise RuntimeError("Cake FMHA context route target does not match the device")
    if route.component == "context_fp16_hd256":
        return load_cake_fmha_context_fp16_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_fp8_hd256":
        return load_cake_fmha_context_fp8_hd256_module(
            route.target,
            route.num_m_blocks,
            route.num_q_heads,
            route.num_kv_heads,
            route.page_size,
        )
    if route.component == "context_nvfp4":
        try:
            return load_cake_fmha_context_nvfp4_module(
                route.target,
                route.num_m_blocks,
                route.num_q_heads,
                route.num_kv_heads,
                route.pack_g,
                route.page_size,
                route.l2_swizzle,
            )
        except (OSError, RuntimeError) as error:
            warnings.warn(
                f"Cake FMHA NVFP4 context loading failed closed to compat_v1: {error}",
                RuntimeWarning,
                stacklevel=2,
            )
            return load_cake_fmha_compat_module(route.target)
    common_args = (
        route.target,
        route.num_m_blocks,
        route.num_q_heads,
        route.num_kv_heads,
        route.pack_g,
        route.page_size,
        route.l2_swizzle,
    )
    common_kwargs = {
        "is_causal": route.is_causal,
        "return_lse": route.return_lse,
        "enable_sink": route.enable_sink,
    }
    if route.component == "context_bf16":
        return load_cake_fmha_context_bf16_module(
            *common_args,
            **common_kwargs,
            exact_profile=route.exact_profile,
        )
    return load_cake_fmha_context_fp8_module(*common_args, **common_kwargs)


def cake_fmha_manifest() -> dict[str, Any]:
    """Return a copy of the authenticated product/capability manifest."""

    return copy.deepcopy(get_cake_fmha_manifest())


def cake_batch_decode_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-decode ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_decode_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's default backend selection.
    """

    from .decode import trtllm_batch_decode_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_decode_with_kv_cache requires backend='cake'")
    return trtllm_batch_decode_with_kv_cache(*args, backend="cake", **kwargs)


def cake_batch_context_with_kv_cache(*args, **kwargs):
    """Run the FlashInfer TRTLLM paged-context ABI through Cake FMHA.

    Parameters and return values match
    :func:`flashinfer.trtllm_batch_context_with_kv_cache`.  The selection is
    explicit and never replaces FlashInfer's conventional default.
    """

    from .prefill import trtllm_batch_context_with_kv_cache

    requested_backend = kwargs.pop("backend", "cake")
    if requested_backend != "cake":
        raise ValueError("cake_batch_context_with_kv_cache requires backend='cake'")
    return trtllm_batch_context_with_kv_cache(*args, backend="cake", **kwargs)


__all__ = [
    "CakeFmhaContextRoute",
    "CakeFmhaDecodeRoute",
    "CakeFmhaRequestOrderedDecodePlan",
    "cake_batch_context_with_kv_cache",
    "cake_batch_decode_with_kv_cache",
    "cake_fmha_manifest",
    "cake_fmha_route_is_optimized",
    "get_cake_fmha_context_module",
    "get_cake_fmha_decode_module",
    "get_cake_fmha_module",
    "plan_cake_fmha_request_ordered_paged_decode",
    "select_cake_fmha_context_route",
    "select_cake_fmha_decode_route",
]
