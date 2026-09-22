"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import functools
import math
from typing import Callable, ClassVar, Optional, Protocol, Tuple, TypeVar, Union, cast

import torch

from ....jit import gen_batch_mla_module
from ....utils import (
    MaskMode,
    check_shape_dtype_device,
    get_compute_capability,
    get_device_properties,
    get_device_sm_count,
    is_sm90a_supported,
)
from ._capabilities import (
    MLAPlanCapabilities,
    _BackendPlanUnsupportedError,
    plan_capability_rejection_reason,
)
from .._planning import _MLAPlanArguments, _audit_plan_from_wrapper_arguments


class _GeneratedBatchMLAModule(Protocol):
    def plan(self, *args: object) -> object: ...

    def plan_with_staged_workspace_bytes(self, *args: object) -> tuple[object, int]: ...

    def run(self, *args: object) -> object: ...


@functools.lru_cache(maxsize=128)
def get_batch_mla_module(
    backend: str,
    dtype_q: torch.dtype,
    dtype_kv: torch.dtype,
    dtype_o: torch.dtype,
    dtype_idx: torch.dtype,
    head_dim_ckv: int,
    head_dim_kpe: int,
    use_profiler: bool,
) -> _GeneratedBatchMLAModule:
    return gen_batch_mla_module(
        backend,
        dtype_q,
        dtype_kv,
        dtype_o,
        dtype_idx,
        head_dim_ckv,
        head_dim_kpe,
        use_profiler,
    ).build_and_load()


def _validate_generated_fa_plan(
    *,
    backend: str,
    device: torch.device,
    qo_indptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_len_arr: torch.Tensor,
    head_dim_ckv: int,
    head_dim_kpe: int,
    q_data_type: torch.dtype,
    kv_data_type: torch.dtype,
    output_dtype: torch.dtype,
    scale_mode: str,
) -> None:
    for name, tensor in (
        ("qo_indptr", qo_indptr),
        ("kv_indptr", kv_indptr),
        ("kv_indices", kv_indices),
        ("kv_len_arr", kv_len_arr),
    ):
        if tensor.dtype != torch.int32:
            raise ValueError(f"{name} must have dtype torch.int32, got {tensor.dtype}.")
    if kv_indptr.numel() < kv_len_arr.numel() + 1:
        raise _BackendPlanUnsupportedError(
            "FA MLA requires KV offsets for every request."
        )
    for name, value, minimum in (
        ("head_dim_ckv", head_dim_ckv, 1),
        ("head_dim_kpe", head_dim_kpe, 0),
    ):
        if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}, got {value!r}.")
    for name, dtype in (("q_data_type", q_data_type), ("kv_data_type", kv_data_type)):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"{name} must be a torch.dtype, got {dtype!r}.")
    if q_data_type not in (torch.float16, torch.bfloat16):
        raise _BackendPlanUnsupportedError(
            f"MLA q_data_type {q_data_type} is not supported by the {backend} backend."
        )
    supported_kv_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    if kv_data_type not in supported_kv_dtypes:
        raise _BackendPlanUnsupportedError(
            f"MLA kv_data_type {kv_data_type} is not supported by the {backend} "
            f"backend. Supported dtypes: {list(supported_kv_dtypes)}."
        )
    if output_dtype != q_data_type:
        raise _BackendPlanUnsupportedError(
            f"{backend} backend output_dtype must match q_data_type, got "
            f"{output_dtype} and {q_data_type}."
        )
    if kv_data_type == torch.float8_e4m3fn:
        major, minor = get_compute_capability(device)
        if major != 9:
            raise _BackendPlanUnsupportedError(
                "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                f"got SM{major}{minor}."
            )
        if q_data_type != torch.bfloat16:
            raise _BackendPlanUnsupportedError(
                "FP8 kv_data_type for MLA currently only supports "
                f"q_data_type=torch.bfloat16, got {q_data_type}."
            )
        if head_dim_ckv != 512 or head_dim_kpe not in (0, 64):
            raise _BackendPlanUnsupportedError(
                "FP8 kv_data_type for MLA currently only supports "
                "head_dim_ckv=512 and head_dim_kpe in (0, 64), got "
                f"head_dim_ckv={head_dim_ckv}, head_dim_kpe={head_dim_kpe}."
            )
        if scale_mode != "kv-per-tensor":
            raise _BackendPlanUnsupportedError(
                "FP8 MLA plans require scale_mode='kv-per-tensor'."
            )
    elif kv_data_type != q_data_type:
        raise _BackendPlanUnsupportedError(
            f"{backend} non-FP8 KV dtype must match query dtype, got "
            f"{kv_data_type} and {q_data_type}."
        )
    elif scale_mode != "default":
        raise _BackendPlanUnsupportedError(
            "non-FP8 MLA plans require scale_mode='default'."
        )

    if head_dim_ckv % 128 or head_dim_kpe % 64:
        raise _BackendPlanUnsupportedError(
            f"{backend} requires head_dim_ckv to be a multiple of 128 and "
            "head_dim_kpe to be a multiple of 64 (including zero)."
        )
    major, minor = get_compute_capability(device)
    if backend == "fa3":
        if not is_sm90a_supported(device):
            raise _BackendPlanUnsupportedError(
                "fa3 MLA requires SM90 and CUDA >= 12.3."
            )
        # PV uses WGMMA N=CKV/2; hopper.cuh implements N=64,128,256
        # among the widths compatible with MLA's 128-wide output stores.
        if head_dim_ckv not in (128, 256, 512):
            raise _BackendPlanUnsupportedError(
                "fa3 MLA requires head_dim_ckv in (128, 256, 512)."
            )
    elif major < 8:
        raise _BackendPlanUnsupportedError(
            f"fa2 MLA requires SM80 or newer, got SM{major}{minor}."
        )
    properties = get_device_properties(device)
    shared_bytes = _generated_fa_shared_bytes(
        backend,
        head_dim_ckv,
        head_dim_kpe,
        kv_data_type == torch.float8_e4m3fn,
        properties.shared_memory_per_multiprocessor,
    )
    if shared_bytes > properties.shared_memory_per_block_optin:
        raise _BackendPlanUnsupportedError(
            f"{backend} MLA needs {shared_bytes} shared-memory bytes per block, "
            f"but this device supports {properties.shared_memory_per_block_optin}."
        )


def _generated_fa_shared_bytes(backend, ckv, kpe, fp8_kv, smem_per_sm):
    """Sizes of SharedStorageQKVO / HopperSharedStorageQKVO, including padding."""
    align16 = lambda size: (size + 15) // 16 * 16
    q_bytes = 64 * (ckv + kpe) * 2
    if backend == "fa3":
        stage = 64 * ckv * (1 if fp8_kv else 2) + max(
            64 * kpe * (1 if fp8_kv else 2), 64 * 64 * 2
        )
        repack = align16(64 * ckv * 2) + align16(max(64 * kpe, 1) * 2) if fp8_kv else 32
        # o_scale/m/d each have 64 float elements. Two PipelineAsync<2>
        # instances each contain four 8-byte ClusterBarrier objects.
        return q_bytes + max(2 * stage + repack, 64 * ckv * 2) + 3 * 64 * 4 + 64
    if smem_per_sm >= 221696:
        stages, tile_kv = 2, 64
    elif smem_per_sm >= 147968:
        stages, tile_kv = 2, 32
    elif smem_per_sm >= 92672:
        stages, tile_kv = 1, 16
    else:
        raise _BackendPlanUnsupportedError(
            f"fa2 MLA requires at least 92672 shared-memory bytes per SM, got {smem_per_sm}."
        )
    if fp8_kv:
        tile_kv = 32
    kv_bytes = stages * tile_kv * (ckv + max(kpe, 64)) * (1 if fp8_kv else 2)
    repack = (
        align16(stages * tile_kv * ckv * 2)
        + align16(stages * tile_kv * max(kpe, 1) * 2)
        if fp8_kv
        else 32
    )
    return max(q_bytes + kv_bytes + repack + 2 * 64 * 4, 64 * ckv * 2)


_FA_MAX_WORK_ITEMS = 16384


def _fa_plan_work_count(q_lens, kv_lens, num_heads, num_sm, causal):
    """Mirror scheduler.cuh MLAPlan, excluding its heap ordering (not its splits).

    Raise early when the unsplit tile count already exceeds native capacity;
    otherwise return the exact split-aware count. Never allocate O(totalQ).
    """
    if not q_lens:
        raise _BackendPlanUnsupportedError(
            "FA MLA native planning requires a nonempty batch."
        )
    if not isinstance(num_heads, int) or isinstance(num_heads, bool) or num_heads <= 0:
        raise ValueError(f"num_heads must be a positive integer, got {num_heads!r}.")
    cluster = 2 if sum(q_lens) * num_heads // len(q_lens) > 64 else 1
    clusters = num_sm // cluster
    if clusters <= 0:
        raise _BackendPlanUnsupportedError(
            "FA MLA device has insufficient SMs for its cluster size."
        )
    tile = 64 * cluster
    tile_counts = [(q * num_heads + tile - 1) // tile for q in q_lens]
    if sum(tile_counts) > _FA_MAX_WORK_ITEMS:
        raise _BackendPlanUnsupportedError(
            f"FA MLA needs at least {sum(tile_counts)} work items; native capacity is {_FA_MAX_WORK_ITEMS}."
        )
    effective = []
    for q, kv, count in zip(q_lens, kv_lens, tile_counts, strict=True):
        for index in range(count):
            effective.append(
                max(
                    min(kv - q + ((index + 1) * tile + num_heads - 1) // num_heads, kv),
                    0,
                )
                if causal and index + 1 != count
                else kv
            )
    average = max((sum(effective) + clusters - 1) // clusters, 1)
    if average > 2**31 - 1:
        raise _BackendPlanUnsupportedError(
            "FA MLA KV split limit exceeds the native int32 range."
        )
    limit = (
        32
        if average <= 8
        else 64
        if average <= 16
        else 128
        if average <= 32
        else 192
        if average <= 64
        else (average + 255) // 256 * 256
    )
    if average > 64 and average + 256 > 2**31 - 1:
        # Native ceil_div spells (x + y - 1) / y in signed int32: the
        # intermediate x + y must fit before the subtraction takes place.
        raise _BackendPlanUnsupportedError(
            "FA MLA rounded KV split limit exceeds the native int32 range."
        )
    if any(
        kv > limit and (kv * cluster > 2**31 - 1 or kv + limit > 2**31 - 1)
        for kv in effective
    ):
        # The native split branch computes both remaining_len * cluster_size
        # and ceil_div(remaining_len, kv_len_limit) with signed int32 operands.
        raise _BackendPlanUnsupportedError(
            "FA MLA KV split arithmetic exceeds the native int32 range."
        )
    return sum(max(1, (kv + limit - 1) // limit) for kv in effective)


def _validate_fa_causal_tile_bound(
    q_lens, kv_lens, num_heads, backend, kv_data_type, device
):
    """Reject the native floor/ceil mismatch, after the work-count bound.

    scheduler.cuh uses ceil(cluster_end / heads), while mla.cuh and
    mla_hopper.cuh use floor in their causal KV iteration bound. A partial
    query at a cluster end loses its last visible key exactly when that key
    starts a native KV tile. KV split starts are subtracted *after* absolute
    ceil_div, so this is an absolute key boundary, not a split-relative one.
    """
    tile_q = 64 * (2 if sum(q_lens) * num_heads // len(q_lens) > 64 else 1)
    if tile_q % num_heads == 0:
        return
    if backend == "fa3":
        # Hopper's tile is fixed at 64, including its FP8 repack path.
        tile_kv = 64
    elif kv_data_type == torch.float8_e4m3fn:
        tile_kv = 32
    else:
        smem = get_device_properties(device).shared_memory_per_multiprocessor
        # Match mla.cuh DISPATCH_SMEM_CONFIG (dimensions do not select tiles).
        tile_kv = 64 if smem >= 221696 else 32 if smem >= 147968 else 16
    for request, (q_len, kv_len) in enumerate(zip(q_lens, kv_lens, strict=True)):
        for end in range(tile_q, q_len * num_heads, tile_q):
            query, remainder = divmod(end, num_heads)
            key = kv_len - q_len + query
            if remainder and query < q_len and 0 <= key < kv_len and key % tile_kv == 0:
                raise _BackendPlanUnsupportedError(
                    f"{backend} MLA causal tile boundary is unsupported: request "
                    f"{request}, num_heads={num_heads}, query={query}, key={key}, "
                    f"native KV tile={tile_kv}; native execution would omit a visible key."
                )


def _validate_fa_plan_workload(
    qo_indptr, kv_len_arr, num_heads, causal, device, backend, kv_data_type
):
    offsets = qo_indptr.to(device="cpu", dtype=torch.int64).tolist()
    kv_lens = kv_len_arr.to(device="cpu", dtype=torch.int64).tolist()
    # Legacy flat CSR can have extra offsets; native MLAPlan uses the KV batch
    # length. Preserve that behavior, but refuse a short offset array before OOB.
    if len(offsets) < len(kv_lens) + 1:
        raise _BackendPlanUnsupportedError(
            "FA MLA requires query offsets for every KV request."
        )
    q_lens = [offsets[i + 1] - offsets[i] for i in range(len(kv_lens))]
    count = _fa_plan_work_count(
        q_lens, kv_lens, num_heads, get_device_sm_count(device), causal
    )
    if count > _FA_MAX_WORK_ITEMS:
        raise _BackendPlanUnsupportedError(
            f"FA MLA needs {count} work items including KV splits; native capacity is {_FA_MAX_WORK_ITEMS}."
        )
    if causal:
        _validate_fa_causal_tile_bound(
            q_lens, kv_lens, num_heads, backend, kv_data_type, device
        )


class _BatchMLAGeneratedFaMechanics:
    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool,
        qo_indptr_buf: Optional[torch.Tensor],
        kv_indptr_buf: Optional[torch.Tensor],
        kv_indices_buf: Optional[torch.Tensor],
        kv_len_arr_buf: Optional[torch.Tensor],
        int_workspace_buffer: Optional[torch.Tensor] = None,
        pin_memory_int_workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        self._backend = backend
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        if int_workspace_buffer is None:
            int_workspace_buffer = torch.empty(
                (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
            )
        if pin_memory_int_workspace_buffer is None:
            pin_memory_int_workspace_buffer = torch.empty(
                int_workspace_buffer.shape,
                dtype=int_workspace_buffer.dtype,
                pin_memory=True,
                device="cpu",
            )
        self._int_workspace_buffer = int_workspace_buffer
        self._pin_memory_int_workspace_buffer = pin_memory_int_workspace_buffer
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr_buf
        self._kv_indptr_buf = kv_indptr_buf
        self._kv_indices_buf = kv_indices_buf
        self._kv_len_arr_buf = kv_len_arr_buf

    @staticmethod
    def _storage_interval(tensor: torch.Tensor) -> tuple[int, int, int]:
        start = tensor.storage_offset() * tensor.element_size()
        return (
            tensor.untyped_storage().data_ptr(),
            start,
            start + tensor.numel() * tensor.element_size(),
        )

    def _preflight_graph_metadata_buffers(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
    ) -> None:
        named = (
            ("qo_indptr", self._qo_indptr_buf, qo_indptr, False),
            ("kv_indptr", self._kv_indptr_buf, kv_indptr, False),
            ("kv_indices", self._kv_indices_buf, kv_indices, True),
            ("kv_len_arr", self._kv_len_arr_buf, kv_len_arr, False),
        )
        for name, reserved, source, allow_larger in named:
            if reserved is None:
                raise ValueError(
                    "CUDA graph mode requires reserved qo_indptr, kv_indptr, "
                    "kv_indices, and kv_len_arr buffers."
                )
            if reserved.dtype != torch.int32:
                raise ValueError(
                    f"CUDA graph reserved {name} buffer must have dtype torch.int32."
                )
            if reserved.device != self.device:
                raise ValueError(
                    f"CUDA graph reserved {name} buffer must be on {self.device}."
                )
            if not reserved.is_contiguous():
                raise ValueError(
                    f"CUDA graph reserved {name} buffer must be contiguous."
                )
            if name == "kv_indices" and reserved.ndim != 1:
                raise ValueError(
                    "CUDA graph reserved kv_indices buffer must have rank 1."
                )
            if (allow_larger and reserved.shape[0] < source.shape[0]) or (
                not allow_larger and reserved.shape != source.shape
            ):
                raise ValueError(
                    f"CUDA graph reserved {name} buffer has insufficient or "
                    "incompatible capacity."
                )
        intervals = [
            (name, self._storage_interval(reserved))
            for name, reserved, _, _ in named
            if reserved is not None
        ]
        for index, (left_name, (left_ptr, left_start, left_end)) in enumerate(
            intervals
        ):
            for right_name, (right_ptr, right_start, right_end) in intervals[
                index + 1 :
            ]:
                if left_ptr == right_ptr and max(left_start, right_start) < min(
                    left_end, right_end
                ):
                    raise ValueError(
                        "CUDA graph reserved metadata buffers must not overlap: "
                        f"{left_name} overlaps {right_name}."
                    )
        copy_targets = {
            name: reserved[: source.shape[0]] if allow_larger else reserved
            for name, reserved, source, allow_larger in named
            if reserved is not None
        }
        for source_name, _, source, _ in named:
            source_interval = self._storage_interval(source)
            for target_name, target in copy_targets.items():
                target_interval = self._storage_interval(target)
                overlaps = source_interval[0] == target_interval[0] and max(
                    source_interval[1], target_interval[1]
                ) < min(source_interval[2], target_interval[2])
                exact_corresponding_reuse = (
                    source_name == target_name and source_interval == target_interval
                )
                if overlaps and not exact_corresponding_reuse:
                    raise ValueError(
                        f"CUDA graph {source_name} source overlaps reserved "
                        f"{target_name} target; only exact same-source/same-target "
                        "reuse is allowed."
                    )

    def _stage_metadata(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
    ) -> None:
        if self._use_cuda_graph:
            self._preflight_graph_metadata_buffers(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_len_arr,
            )
            assert self._qo_indptr_buf is not None
            assert self._kv_indptr_buf is not None
            assert self._kv_indices_buf is not None
            assert self._kv_len_arr_buf is not None
            targets = (
                (self._qo_indptr_buf, qo_indptr),
                (self._kv_indptr_buf, kv_indptr),
                (self._kv_indices_buf[: len(kv_indices)], kv_indices),
                (self._kv_len_arr_buf, kv_len_arr),
            )
            snapshots = tuple(target.clone() for target, _ in targets)
            try:
                for target, source in targets:
                    target.copy_(source, non_blocking=True)
            except Exception:
                for (target, _), snapshot in zip(targets, snapshots, strict=True):
                    target.copy_(snapshot)
                raise
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)

    def _plan_generated_fa(
        self,
        *,
        module_loader: Callable[[], _GeneratedBatchMLAModule],
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool,
    ) -> None:
        # ---------------------------------------------------------------------------
        # Build the generated backend plan
        # ---------------------------------------------------------------------------
        _validate_fa_plan_workload(
            qo_indptr,
            kv_len_arr,
            num_heads,
            causal,
            self.device,
            self._backend,
            kv_data_type,
        )
        cached_module = module_loader()
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")
        plan_args = (
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,
            causal,
        )
        if hasattr(cached_module, "plan_with_staged_workspace_bytes"):
            plan_info, staged_int_workspace_bytes = (
                cached_module.plan_with_staged_workspace_bytes(*plan_args)
            )
        else:
            # Compatibility for externally cached legacy modules; generated modules
            # are always adapted at load time.
            plan_info = cached_module.plan(*plan_args)
            staged_int_workspace_bytes = 0

        # ---------------------------------------------------------------------------
        # Stage metadata and publish backend state
        # ---------------------------------------------------------------------------
        self._stage_metadata(
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
        )
        self._cached_module = cached_module
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._head_dim_ckv = head_dim_ckv
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._use_profiler = use_profiler
        self._plan_info = plan_info
        self._staged_int_workspace_bytes = staged_int_workspace_bytes

    def _validate_run_input_dtypes(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
    ) -> None:
        if q_nope.dtype != self._q_data_type:
            raise ValueError(
                f"q_nope.dtype={q_nope.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if q_pe.dtype != self._q_data_type:
            raise ValueError(
                f"q_pe.dtype={q_pe.dtype} does not match the planned "
                f"q_data_type={self._q_data_type}."
            )
        if ckv_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"ckv_cache.dtype={ckv_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )
        if kpe_cache.dtype != self._kv_data_type:
            raise ValueError(
                f"kpe_cache.dtype={kpe_cache.dtype} does not match the planned "
                f"kv_data_type={self._kv_data_type}."
            )

    def _run_generated_fa(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        ckv_scale: Optional[float],
        ckv_scale_arr: Optional[torch.Tensor],
        kpe_scale: Optional[float],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        # ---------------------------------------------------------------------------
        # Validate inputs and scale arguments
        # ---------------------------------------------------------------------------
        self._validate_run_input_dtypes(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
        )
        kv_is_fp8 = self._kv_data_type == torch.float8_e4m3fn
        if kv_is_fp8:
            if (ckv_scale is None) == (ckv_scale_arr is None):
                raise ValueError(
                    "Exactly one of ckv_scale or ckv_scale_arr is required when "
                    "kv_data_type is FP8."
                )
            if kpe_scale is None:
                raise ValueError("kpe_scale is required when kv_data_type is FP8.")
            ckv_scale_f = 1.0 if ckv_scale is None else float(ckv_scale)
            kpe_scale_f = float(kpe_scale)
            if ckv_scale is not None and (
                not math.isfinite(ckv_scale_f) or ckv_scale_f <= 0.0
            ):
                raise ValueError(
                    f"ckv_scale must be a finite positive value, got {ckv_scale}"
                )
            if not math.isfinite(kpe_scale_f) or kpe_scale_f <= 0.0:
                raise ValueError(
                    f"kpe_scale must be a finite positive value, got {kpe_scale}"
                )
        else:
            if (
                ckv_scale is not None
                or ckv_scale_arr is not None
                or kpe_scale is not None
            ):
                raise ValueError(
                    "ckv_scale / ckv_scale_arr / kpe_scale are only valid when "
                    "kv_data_type is FP8."
                )
            ckv_scale_f = 1.0
            kpe_scale_f = 1.0

        # ---------------------------------------------------------------------------
        # Prepare output and auxiliary buffers
        # ---------------------------------------------------------------------------
        if profiler_buffer is None and self._use_profiler:
            raise ValueError("Profiler is enabled, profiler_buffer must be provided")
        if out is None:
            out = torch.empty_like(q_nope)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )
        if return_lse:
            if lse is None:
                lse = torch.empty(
                    q_nope.shape[:2], dtype=torch.float32, device=self.device
                )
            else:
                check_shape_dtype_device(
                    lse, q_nope.shape[:2], torch.float32, q_nope.device, "lse"
                )
        if ckv_scale_arr is not None:
            expected_scale_shape = (*ckv_cache.shape[:-1], self._head_dim_ckv // 128)
            check_shape_dtype_device(
                ckv_scale_arr,
                expected_scale_shape,
                torch.float32,
                ckv_cache.device,
                "ckv_scale_arr",
            )
            if not ckv_scale_arr.is_contiguous():
                raise ValueError("ckv_scale_arr must be contiguous.")

        # ---------------------------------------------------------------------------
        # Launch the generated backend
        # ---------------------------------------------------------------------------
        mask_mode = MaskMode.CAUSAL.value if self._causal else MaskMode.NON_CAUSAL.value
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            q_nope.shape[1],
            self._page_size,
            self._sm_scale,
            return_lse_base_on_e,
            ckv_scale_f,
            kpe_scale_f,
            ckv_scale_arr,
            *profiler_args,
        )
        return (out, lse) if return_lse else out


_FaBackendT = TypeVar("_FaBackendT", bound="_BatchMLAPagedAttentionFaBackendBase")


class _BatchMLAPagedAttentionFaBackendBase(_BatchMLAGeneratedFaMechanics):
    _plan_capabilities: ClassVar[Optional[MLAPlanCapabilities]] = None

    def __init__(
        self,
        *,
        backend: Optional[str] = None,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool,
        qo_indptr_buf: Optional[torch.Tensor],
        kv_indptr_buf: Optional[torch.Tensor],
        kv_indices_buf: Optional[torch.Tensor],
        kv_len_arr_buf: Optional[torch.Tensor],
        query_split_widths: tuple[int, int],
        kv_split_widths: tuple[int, int],
        int_workspace_buffer: Optional[torch.Tensor] = None,
        pin_memory_int_workspace_buffer: Optional[torch.Tensor] = None,
    ) -> None:
        if backend is None:
            if self._plan_capabilities is None:
                raise TypeError("generated-FA backend capabilities are required.")
            backend = self._plan_capabilities.backend_name
        super().__init__(
            backend=backend,
            float_workspace_buffer=float_workspace_buffer,
            use_cuda_graph=use_cuda_graph,
            qo_indptr_buf=qo_indptr_buf,
            kv_indptr_buf=kv_indptr_buf,
            kv_indices_buf=kv_indices_buf,
            kv_len_arr_buf=kv_len_arr_buf,
            int_workspace_buffer=int_workspace_buffer,
            pin_memory_int_workspace_buffer=pin_memory_int_workspace_buffer,
        )
        self._query_split_widths = query_split_widths
        self._kv_split_widths = kv_split_widths

    def plan(
        self,
        *,
        qo_indptr: torch.Tensor,
        kv_indptr: torch.Tensor,
        kv_indices: torch.Tensor,
        kv_len_arr: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        output_dtype: torch.dtype,
        scale_mode: str,
        use_profiler: bool,
    ) -> None:
        raise NotImplementedError

    @classmethod
    def preflight_plan_from_wrapper(cls, args: _MLAPlanArguments) -> None:
        assert cls._plan_capabilities is not None
        csr = args.csr()  # Malformed caller metadata is never a support refusal.
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise _BackendPlanUnsupportedError(reason)
        _validate_generated_fa_plan(
            backend=cls._plan_capabilities.backend_name,
            device=args._float_workspace_buffer.device,
            qo_indptr=csr.qo_indptr,
            kv_indptr=csr.kv_indptr,
            kv_indices=csr.kv_indices,
            kv_len_arr=csr.kv_len_arr,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            output_dtype=args.output_dtype,
            scale_mode=args.scale_mode,
        )
        _validate_fa_plan_workload(
            csr.qo_indptr,
            csr.kv_len_arr,
            args.num_heads,
            args.causal,
            args._float_workspace_buffer.device,
            cls._plan_capabilities.backend_name,
            args.kv_data_type,
        )

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(
        cls: type[_FaBackendT], args: _MLAPlanArguments
    ) -> _FaBackendT:
        assert cls._plan_capabilities is not None
        cls.preflight_plan_from_wrapper(args)
        csr = args.csr()
        backend = cls(
            float_workspace_buffer=args._float_workspace_buffer,
            use_cuda_graph=args._use_cuda_graph,
            qo_indptr_buf=args._qo_indptr_buf,
            kv_indptr_buf=args._kv_indptr_buf,
            kv_indices_buf=args._kv_indices_buf,
            kv_len_arr_buf=args._kv_len_arr_buf,
            query_split_widths=(args.head_dim_ckv, args.head_dim_kpe),
            kv_split_widths=(args.head_dim_ckv, args.head_dim_kpe),
            int_workspace_buffer=args._graph_plan_int_workspace_buffer,
        )
        if args._use_cuda_graph:
            backend._preflight_graph_metadata_buffers(
                qo_indptr=csr.qo_indptr,
                kv_indptr=csr.kv_indptr,
                kv_indices=csr.kv_indices,
                kv_len_arr=csr.kv_len_arr,
            )
        backend.plan(
            qo_indptr=csr.qo_indptr,
            kv_indptr=csr.kv_indptr,
            kv_indices=csr.kv_indices,
            kv_len_arr=csr.kv_len_arr,
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            page_size=args.page_size,
            causal=args.causal,
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            output_dtype=args.output_dtype,
            scale_mode=args.scale_mode,
            use_profiler=args.use_profiler,
        )
        return backend

    def run_from_wrapper(
        self,
        *,
        query: object,
        kv_cache: object,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        ckv_scale_arr: Optional[torch.Tensor],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
        if sinks is not None:
            raise ValueError("sinks are not supported with an fa2/fa3 backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with an "
                "fa2/fa3 backend."
            )
        if bmm1_scale is not None or bmm2_scale is not None:
            raise ValueError("BMM scales are not supported with an fa2/fa3 backend.")
        q_nope, q_pe = cast(tuple[torch.Tensor, torch.Tensor], query)
        ckv_cache, kpe_cache = cast(tuple[torch.Tensor, torch.Tensor], kv_cache)
        return self._run_generated_fa(
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            return_lse_base_on_e=return_lse_base_on_e,
            ckv_scale=ckv_scale,
            ckv_scale_arr=ckv_scale_arr,
            kpe_scale=kpe_scale,
        )
