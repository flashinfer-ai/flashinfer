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
from ....utils import MaskMode, check_shape_dtype_device, get_compute_capability
from ._capabilities import MLAPlanCapabilities, plan_capability_rejection_reason
from .._planning import _MLAPlanArguments


class _GeneratedBatchMLAModule(Protocol):
    def plan(self, *args: object) -> object: ...

    def run(self, *args: object) -> object: ...


@functools.cache
def get_batch_mla_module(backend: str, *args: object) -> _GeneratedBatchMLAModule:
    return gen_batch_mla_module(backend, *args).build_and_load()


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
    if q_data_type not in (torch.float16, torch.bfloat16):
        raise ValueError(
            f"MLA q_data_type {q_data_type} is not supported by the {backend} backend."
        )
    supported_kv_dtypes = (torch.float16, torch.bfloat16, torch.float8_e4m3fn)
    if kv_data_type not in supported_kv_dtypes:
        raise ValueError(
            f"MLA kv_data_type {kv_data_type} is not supported by the {backend} "
            f"backend. Supported dtypes: {list(supported_kv_dtypes)}."
        )
    if output_dtype != q_data_type:
        raise ValueError(
            f"{backend} backend output_dtype must match q_data_type, got "
            f"{output_dtype} and {q_data_type}."
        )
    if head_dim_kpe < 0:
        raise ValueError(f"head_dim_kpe must be >= 0, got {head_dim_kpe}.")
    if kv_data_type == torch.float8_e4m3fn:
        major, minor = get_compute_capability(device)
        if major != 9:
            raise ValueError(
                "FP8 kv_data_type for MLA requires an SM90 (Hopper) device, "
                f"got SM{major}{minor}."
            )
        if q_data_type != torch.bfloat16:
            raise ValueError(
                "FP8 kv_data_type for MLA currently only supports "
                f"q_data_type=torch.bfloat16, got {q_data_type}."
            )
        if head_dim_ckv != 512 or head_dim_kpe not in (0, 64):
            raise ValueError(
                "FP8 kv_data_type for MLA currently only supports "
                "head_dim_ckv=512 and head_dim_kpe in (0, 64), got "
                f"head_dim_ckv={head_dim_ckv}, head_dim_kpe={head_dim_kpe}."
            )
        if scale_mode != "kv-per-tensor":
            raise ValueError("FP8 MLA plans require scale_mode='kv-per-tensor'.")
    elif scale_mode != "default":
        raise ValueError("non-FP8 MLA plans require scale_mode='default'.")


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
        cached_module = module_loader()
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")
        plan_info = cached_module.plan(
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
    def plan_from_wrapper(
        cls: type[_FaBackendT], args: _MLAPlanArguments
    ) -> _FaBackendT:
        assert cls._plan_capabilities is not None
        if reason := plan_capability_rejection_reason(args, cls._plan_capabilities):
            raise ValueError(reason)
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
            pin_memory_int_workspace_buffer=args._graph_plan_pin_memory_int_workspace_buffer,
        )
        csr = args.csr()
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if kv_len is not None:
            raise ValueError("kv_len is only supported with cutlass backend.")
        if page_table is not None:
            raise ValueError("page_table is only supported with cutlass backend.")
        if o_scale is not None:
            raise ValueError(
                "o_scale is only supported with the cutlass backend for now."
            )
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
