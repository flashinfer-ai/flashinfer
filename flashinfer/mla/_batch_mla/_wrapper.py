"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
"""

import functools
import inspect
import math
import warnings
from dataclasses import replace
from typing import Any, ClassVar, Literal, Optional, Protocol, Tuple, Union, overload

import torch

from ...api_logging import flashinfer_api
from ...trace.templates.attention import mla_paged_decode_trace
from ...utils import determine_mla_backend, get_compute_capability
from ._backends._capabilities import MLAPlanCapabilities
from ._backends.cutlass_backend import _BatchMLAPagedAttentionCutlassBackend
from ._backends.cutile_backend import (
    _CUTILE_SUPPORTED_COMPUTE_CAPABILITIES,
    _BatchMLAPagedAttentionCutileBackend,
)
from ._backends.fa2_backend import _BatchMLAPagedAttentionFa2Backend
from ._backends.fa3_backend import _BatchMLAPagedAttentionFa3Backend
from ._contracts import (
    MLAInputContract,
    MLAPlanMetadata,
    _resolve_structural_mla_input,
    _structural_mla_input_facts,
)
from ._planning import _MLAPlanArguments


class _PlannedBackend(Protocol):
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...


class _WrapperBackendType(Protocol):
    _plan_capabilities: ClassVar[MLAPlanCapabilities]

    @classmethod
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> _PlannedBackend: ...


def _get_compute_capability(device: torch.device):
    return get_compute_capability(device)


_BACKEND_TYPES: dict[str, type[_WrapperBackendType]] = {
    "fa2": _BatchMLAPagedAttentionFa2Backend,
    "fa3": _BatchMLAPagedAttentionFa3Backend,
    "cutlass": _BatchMLAPagedAttentionCutlassBackend,
    "cutile": _BatchMLAPagedAttentionCutileBackend,
}

_MIRRORED_BACKEND_ATTRS = (
    "_cached_module",
    "_int_workspace_buffer",
    "_pin_memory_int_workspace_buffer",
)


def _warn_from_external_caller(message: str, category: type[Warning]) -> None:
    """Warn at the first caller outside the local and API-logging wrappers."""

    frame = inspect.currentframe()
    stacklevel = 1
    internal_modules = {__name__, flashinfer_api.__module__}
    try:
        while frame is not None and frame.f_globals.get("__name__") in internal_modules:
            stacklevel += 1
            frame = frame.f_back
    finally:
        del frame
    warnings.warn(message, category, stacklevel=stacklevel)


def _warn_on_positional_mla_arguments(method: Any) -> Any:
    """Warn once per wrapper while preserving positional-call compatibility."""

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            self._warn_deprecated_positional_arguments()
        return method(self, *args, **kwargs)

    return wrapped


class BatchMLAPagedAttentionWrapper:
    r"""Wrapper for MLA PagedAttention on DeepSeek models.

    This wrapper is intended for decode and incremental prefill with the
    Matrix Absorption formulation of MLA, where the query/key and value/output
    projections are absorbed before attention. For the non-absorbed MLA
    prefill path, use the appropriate prefill wrapper instead.

    The planned-wrapper surface owns FA2, FA3, CUTLASS, and cuTile planning
    and execution. Call :meth:`plan` once with canonical metadata before
    invoking :meth:`run`; the plan captures the supported input/output contract
    and the concrete backend's metadata representation.

    See :ref:`MLA Page Layout <mla-page-layout>` for the paged KV-cache layout
    and the `FlashInfer MLA blog post
    <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_ for the
    computation and Matrix Absorption background.
    """

    _blackwell_auto_fallback_warned: bool = False
    _legacy_plan_warned: bool = False

    @classmethod
    def _maybe_warn_blackwell_auto_fallback(
        cls, device: torch.device, selected_backend: str
    ) -> None:
        if cls._blackwell_auto_fallback_warned:
            return
        major, minor = _get_compute_capability(device)
        if major < 10:
            return
        cls._blackwell_auto_fallback_warned = True
        if (major, minor) in _CUTILE_SUPPORTED_COMPUTE_CAPABILITIES:
            in_wrapper_alternative = (
                "backend='cutile' is the native in-wrapper cuda.tile alternative."
            )
        else:
            in_wrapper_alternative = (
                "backend='cutlass' is the closest in-wrapper alternative but may be "
                "slower than this fallback for decode shapes."
            )
        warnings.warn(
            f"BatchMLAPagedAttentionWrapper: backend='auto' selected "
            f"'{selected_backend}' on SM{major}{minor}, which is not Blackwell-native "
            f"and gives poor MLA decode performance. For decode, use "
            f"flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla "
            f"(Blackwell-native trtllm-gen); {in_wrapper_alternative}",
            UserWarning,
            stacklevel=3,
        )

    @classmethod
    def _warn_legacy_plan_once(cls) -> None:
        if cls._legacy_plan_warned:
            return
        cls._legacy_plan_warned = True
        _warn_from_external_caller(
            "Passing flat BatchMLAPagedAttentionWrapper.plan metadata tensors is "
            "deprecated; pass metadata=flashinfer.mla.MLAPlanMetadata.csr(...) "
            "or metadata=flashinfer.mla.MLAPlanMetadata.dense(...).",
            DeprecationWarning,
        )

    def _warn_deprecated_positional_arguments(self) -> None:
        """Emit the historical positional-argument warning once per instance."""

        if getattr(self, "_warned_positional_arguments", False):
            return
        warnings.warn(
            "Positional MLA arguments are deprecated; pass plan() and run() "
            "arguments by keyword instead. Positional calling will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        self._warned_positional_arguments = True

    def _warn_deprecated_legacy_tensor_arguments(self) -> None:
        """Warn once when the legacy separate-tensor run form is accepted."""

        if getattr(self, "_warned_legacy_tensor_arguments", False):
            return
        _warn_from_external_caller(
            "Legacy MLA tensor arguments q_nope/q_pe and ckv_cache/kpe_cache "
            "are deprecated; pass query= and kv_cache= structural values instead. "
            "This compatibility path will be removed in a future release.",
            DeprecationWarning,
        )
        self._warned_legacy_tensor_arguments = True

    def _warn_legacy_dynamic_lse_once(self) -> None:
        """Warn when a legacy CSR plan relies on its historical LSE behavior."""

        if getattr(self, "_warned_legacy_dynamic_lse", False):
            return
        _warn_from_external_caller(
            "Legacy flat CSR MLA plans temporarily allow dynamic LSE at run time. "
            "Pass lse_mode= to plan() before migrating to canonical metadata; "
            "this compatibility path will be removed with flat-plan support.",
            DeprecationWarning,
        )
        self._warned_legacy_dynamic_lse = True

    def _publish_backend_mirrors(self, backend: object) -> None:
        """Expose backend fast-replay attrs without retaining stale mirrors."""

        for name in _MIRRORED_BACKEND_ATTRS:
            if hasattr(backend, name):
                setattr(self, name, getattr(backend, name))
            elif hasattr(self, name):
                delattr(self, name)

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        backend: str = "auto",
    ) -> None:
        r"""Construct a planned MLA wrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            Caller-owned workspace for intermediate attention results. A 128 MiB
            buffer is the usual starting point; it must be on the same device as
            the query and KV-cache tensors.
        use_cuda_graph : bool, optional
            Enable CUDA-graph-compatible planning. When enabled, the optional
            metadata buffers below are copied at :meth:`plan` time to preserve
            capture-time pointers. The captured batch shape cannot change.
        qo_indptr, kv_indptr : Optional[torch.Tensor]
            Caller-reserved ``int32`` buffers of shape ``[batch_size + 1]`` for
            CSR metadata. Used only with ``use_cuda_graph=True``.
        kv_indices : Optional[torch.Tensor]
            Caller-reserved ``int32`` CSR page-index buffer, sized for the
            maximum planned number of pages. Used only with CUDA graphs.
        kv_len_arr : Optional[torch.Tensor]
            Caller-reserved ``int32`` buffer of shape ``[batch_size]`` for CSR
            KV lengths. Used only with CUDA graphs.
        backend : {"auto", "fa2", "fa3", "cutlass", "cutile"}
            Requested concrete backend. ``"auto"`` selects the architecture
            default exposed by :func:`flashinfer.utils.determine_mla_backend`.
            Explicit CUTLASS callers should plan with canonical dense metadata;
            its historical planless ``run`` path remains deprecated.
            Explicit cuTile callers should plan packed or split FP16/BF16
            DeepSeek MLA decode inputs with canonical dense or CSR metadata.
            cuTile is not selected automatically.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        self._requested_backend = backend
        if backend == "auto":
            self._backend = determine_mla_backend(self.device)
            self._maybe_warn_blackwell_auto_fallback(self.device, self._backend)
        elif backend in _BACKEND_TYPES:
            self._backend = backend
        else:
            raise ValueError(
                "backend must be one of 'auto', 'fa2', 'fa3', 'cutlass', or "
                "'cutile', "
                f"got {backend!r}."
            )
        self._planned_backend: Optional[_PlannedBackend] = None
        self._input_contract: Optional[MLAInputContract] = None
        self._planned_query_layout: Optional[Literal["packed", "split"]] = None
        self._planned_kv_cache_layout: Optional[Literal["packed", "split"]] = None
        self._warned_positional_arguments = False
        self._warned_legacy_tensor_arguments = False
        self._legacy_flat_csr_plan = False
        self._warned_legacy_dynamic_lse = False

    # Preferred canonical metadata form.
    @overload
    def plan(
        self,
        *,
        metadata: MLAPlanMetadata,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool = False,
        query_layout: Literal["packed", "split"] = "packed",
        kv_cache_layout: Literal["packed", "split"] = "packed",
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor"] = "default",
        skip_softmax: bool = False,
    ) -> None: ...

    # Legacy flat-metadata compatibility: canonical CSR metadata, native for
    # FA2 and FA3. This form is deprecated; positional arguments also warn.
    @overload
    def plan(
        self,
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
        use_profiler: bool = False,
        *,
        query_layout: Literal["packed", "split"] = "split",
        kv_cache_layout: Literal["packed", "split"] = "split",
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor"] = "default",
        skip_softmax: bool = False,
    ) -> None: ...

    # Legacy flat-metadata compatibility: canonical dense page-table metadata,
    # native for CUTLASS and cuTile. This keyword-only form is deprecated.
    @overload
    def plan(
        self,
        *,
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        max_q_len: Optional[int] = None,
        use_profiler: bool = False,
        query_layout: Literal["packed", "split"] = "split",
        kv_cache_layout: Literal["packed", "split"] = "split",
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor"] = "default",
        skip_softmax: bool = False,
    ) -> None: ...

    @_warn_on_positional_mla_arguments
    @flashinfer_api
    def plan(
        self,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
        num_heads: Optional[int] = None,
        head_dim_ckv: Optional[int] = None,
        head_dim_kpe: Optional[int] = None,
        page_size: Optional[int] = None,
        causal: Optional[bool] = None,
        sm_scale: Optional[float] = None,
        q_data_type: Optional[torch.dtype] = None,
        kv_data_type: Optional[torch.dtype] = None,
        use_profiler: bool = False,
        *,
        metadata: Optional[MLAPlanMetadata] = None,
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        query_layout: Optional[Literal["packed", "split"]] = None,
        kv_cache_layout: Optional[Literal["packed", "split"]] = None,
        lse_mode: Literal["none", "base2", "basee"] = "none",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal["default", "kv-per-tensor"] = "default",
        skip_softmax: bool = False,
    ) -> None:
        r"""Plan a concrete MLA backend from canonical metadata.

        Prefer ``metadata=MLAPlanMetadata.csr(...)`` for the CSR form or
        ``metadata=MLAPlanMetadata.dense(...)`` for the dense page-table form.
        ``MLAPlanMetadata.dual(...)`` may be used when both representations
        already exist; the planner verifies that they describe the same
        requests and page mapping before publishing the plan. FA2 and FA3
        consume CSR metadata natively, while CUTLASS and cuTile consume dense
        metadata.

        Metadata tensors may be on CPU or the wrapper device. They are
        normalized to the device required by the selected backend; tensors on
        another accelerator device are rejected. Passing flat CSR or dense
        metadata fields remains supported for compatibility, but is deprecated
        in favor of the ``metadata=`` object form.

        The plan also declares the later :meth:`run` contract. In particular,
        ``query_layout``, ``kv_cache_layout``, ``lse_mode``, ``output_dtype``,
        ``output_scale``, and ``scale_mode`` must agree with the subsequent
        call. Canonical metadata defaults to packed inputs; the legacy flat
        forms retain their historical split-input defaults. Deprecated flat
        CSR plans also temporarily retain dynamic LSE behavior on FA2/FA3 and
        emit a warning when it is used.

        Parameters
        ----------
        metadata : Optional[MLAPlanMetadata]
            Preferred canonical CSR, dense, or dual metadata representation.
        qo_indptr, kv_indptr, kv_indices, kv_len_arr : Optional[torch.Tensor]
            Deprecated flat CSR metadata fields.
        cum_seq_lens_q, block_tables, seq_lens : Optional[torch.Tensor]
            Deprecated flat dense page-table metadata fields.
        max_q_len : Optional[int]
            Maximum dense query length; inferred from query metadata when
            omitted.
        num_heads : Optional[int]
            Number of query heads.
        head_dim_ckv, head_dim_kpe : Optional[int]
            Compressed-KV and RoPE feature widths.
        page_size : Optional[int]
            Number of KV tokens in a cache page.
        causal : Optional[bool]
            Whether the planned attention is causal.
        sm_scale : Optional[float]
            Softmax scale captured by the plan.
        q_data_type, kv_data_type : Optional[torch.dtype]
            Query and KV-cache dtypes accepted by the selected backend.
        use_profiler : bool
            Whether to enable backend profiler support.
        query_layout, kv_cache_layout : Optional[{"packed", "split"}]
            Tensor representations accepted by subsequent :meth:`run` calls.
        lse_mode : {"none", "base2", "basee"}
            Required log-sum-exp output mode.
        output_dtype : Optional[torch.dtype]
            Required output dtype; defaults to ``q_data_type``.
        output_scale : {"none", "per-tensor"}
            Required output scaling mode.
        scale_mode : {"default", "kv-per-tensor"}
            Required KV-scale mode for subsequent :meth:`run` calls.
        skip_softmax : bool
            Whether the plan must support the skip-softmax threshold feature.

        Notes
        -----
        Positional arguments are deprecated; use keyword arguments for all
        ``plan()`` parameters. Flat metadata arguments are also deprecated;
        use an ``MLAPlanMetadata`` object instead.
        """
        # ---------------------------------------------------------------------------
        # Normalize metadata and handle legacy forms
        # ---------------------------------------------------------------------------
        flat_values = (
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_len_arr,
            cum_seq_lens_q,
            block_tables,
            seq_lens,
            max_q_len,
        )
        if metadata is not None:
            if not isinstance(metadata, MLAPlanMetadata):
                raise TypeError("metadata must be an MLAPlanMetadata instance.")
            if any(value is not None for value in flat_values):
                raise ValueError(
                    "Both metadata object and flat metadata arguments were provided; "
                    "only one representation may be supplied."
                )
            plan_metadata = metadata
            legacy_flat = False
            legacy_flat_csr = False
        else:
            csr_present = any(
                value is not None
                for value in (qo_indptr, kv_indptr, kv_indices, kv_len_arr)
            )
            dense_present = any(
                value is not None
                for value in (cum_seq_lens_q, block_tables, seq_lens, max_q_len)
            )
            if csr_present and dense_present:
                raise ValueError("flat CSR and dense metadata forms cannot be mixed.")
            if not csr_present and not dense_present:
                raise TypeError("plan() requires metadata or flat metadata arguments.")
            plan_metadata = MLAPlanMetadata(
                qo_indptr=qo_indptr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
                kv_len_arr=kv_len_arr,
                cum_seq_lens_q=cum_seq_lens_q,
                block_tables=block_tables,
                seq_lens=seq_lens,
                max_q_len=max_q_len,
            )
            legacy_flat = True
            legacy_flat_csr = csr_present
            self._warn_legacy_plan_once()

        # ---------------------------------------------------------------------------
        # Normalize the declared run contract
        # ---------------------------------------------------------------------------
        required = {
            "num_heads": num_heads,
            "head_dim_ckv": head_dim_ckv,
            "head_dim_kpe": head_dim_kpe,
            "page_size": page_size,
            "causal": causal,
            "sm_scale": sm_scale,
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise TypeError("plan() missing required arguments: " + ", ".join(missing))
        if query_layout is None:
            query_layout = "split" if legacy_flat else "packed"
        if kv_cache_layout is None:
            kv_cache_layout = "split" if legacy_flat else "packed"
        output_dtype = q_data_type if output_dtype is None else output_dtype
        if kv_data_type == torch.float8_e4m3fn and scale_mode == "default":
            # Existing FP8 callers always supply CKV/KPE scales at run time.
            scale_mode = "kv-per-tensor"

        previous_backend = getattr(self, "_planned_backend", None)
        graph_plan_int_workspace_buffer = None
        if (
            self._use_cuda_graph
            and previous_backend is not None
            and self._backend in ("fa2", "fa3")
            and getattr(previous_backend, "_backend", None) in ("fa2", "fa3")
        ):
            graph_plan_int_workspace_buffer = previous_backend._int_workspace_buffer

        backend_type = _BACKEND_TYPES[self._backend]
        planned_capabilities = backend_type._plan_capabilities
        planned_query_layout: Literal["packed", "split"] = (
            "packed" if planned_capabilities.requires_packed_query else "split"
        )
        planned_kv_cache_layout: Literal["packed", "split"] = (
            "packed" if planned_capabilities.requires_packed_kv_cache else "split"
        )
        input_contract = MLAInputContract(
            lse_mode=lse_mode,
            output_dtype=output_dtype,
            output_scale=output_scale,
            scale_mode=scale_mode,
            query_layout=query_layout,
            kv_cache_layout=kv_cache_layout,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
        )

        # ---------------------------------------------------------------------------
        # Lower wrapper inputs to backend plan arguments
        # ---------------------------------------------------------------------------
        kv_layout: Literal["combined", "independent-split"] = (
            "combined" if kv_cache_layout == "packed" else "independent-split"
        )
        plan_args = _MLAPlanArguments(
            metadata=plan_metadata,
            num_heads=num_heads,
            head_dim_ckv=head_dim_ckv,
            head_dim_kpe=head_dim_kpe,
            page_size=page_size,
            causal=causal,
            sm_scale=sm_scale,
            q_data_type=q_data_type,
            kv_data_type=kv_data_type,
            query_kind=("packed" if query_layout == "packed" else "independent-split"),
            kv_kind=("packed" if kv_cache_layout == "packed" else "independent-split"),
            query_layout=query_layout,
            kv_cache_layout=kv_cache_layout,
            lse_mode=lse_mode,
            kv_layout=kv_layout,
            output_dtype=output_dtype,
            output_scale=output_scale,
            scale_mode=scale_mode,
            skip_softmax=skip_softmax,
            use_profiler=use_profiler,
            legacy_flat_csr=legacy_flat_csr,
            _float_workspace_buffer=self._float_workspace_buffer,
            _use_cuda_graph=self._use_cuda_graph,
            _qo_indptr_buf=self._qo_indptr_buf,
            _kv_indptr_buf=self._kv_indptr_buf,
            _kv_indices_buf=self._kv_indices_buf,
            _kv_len_arr_buf=self._kv_len_arr_buf,
            _graph_plan_int_workspace_buffer=graph_plan_int_workspace_buffer,
        )

        # ---------------------------------------------------------------------------
        # Enforce CUDA graph replanning constraints
        # ---------------------------------------------------------------------------
        planned_backend_name = getattr(
            getattr(self, "_planned_backend", None), "_backend", None
        )
        if (
            self._use_cuda_graph
            and getattr(self, "_planned_backend", None) is not None
            and planned_backend_name in ("cutlass", "cutile")
        ):
            graph_backend_name = (
                "CUTLASS" if planned_backend_name == "cutlass" else "cuTile"
            )
            raise RuntimeError(
                f"CUDA graph {graph_backend_name} plans cannot replan because "
                "dense metadata "
                "pointers must remain stable."
            )

        # ---------------------------------------------------------------------------
        # Plan with the selected backend
        # ---------------------------------------------------------------------------
        graph_workspace_snapshot = None
        if graph_plan_int_workspace_buffer is not None:
            prior_plan_workspace_bytes = int(
                getattr(previous_backend, "_staged_int_workspace_bytes", 0)
            )
            if not (
                0
                <= prior_plan_workspace_bytes
                <= graph_plan_int_workspace_buffer.numel()
            ):
                raise RuntimeError(
                    "previous CUDA graph plan has an invalid device int workspace "
                    "usage size."
                )
            graph_workspace_snapshot = graph_plan_int_workspace_buffer[
                :prior_plan_workspace_bytes
            ].clone()
        try:
            planned_backend = backend_type.plan_from_wrapper(plan_args)
        except Exception:
            if graph_workspace_snapshot is not None:
                graph_plan_int_workspace_buffer[
                    : graph_workspace_snapshot.numel()
                ].copy_(graph_workspace_snapshot)
            raise

        # ---------------------------------------------------------------------------
        # Publish the successful plan state
        # ---------------------------------------------------------------------------
        self._planned_backend = planned_backend
        self._input_contract = input_contract
        self._planned_query_layout = planned_query_layout
        self._planned_kv_cache_layout = planned_kv_cache_layout
        self._publish_backend_mirrors(planned_backend)
        self._legacy_flat_csr_plan = legacy_flat_csr
        self._qo_indptr_buf = getattr(
            planned_backend, "_qo_indptr_buf", self._qo_indptr_buf
        )
        self._kv_indptr_buf = getattr(
            planned_backend, "_kv_indptr_buf", self._kv_indptr_buf
        )
        self._kv_indices_buf = getattr(
            planned_backend, "_kv_indices_buf", self._kv_indices_buf
        )
        self._kv_len_arr_buf = getattr(
            planned_backend, "_kv_len_arr_buf", self._kv_len_arr_buf
        )
        self._sm_scale = sm_scale
        self._causal = causal
        self._page_size = page_size
        self._head_dim_ckv = head_dim_ckv
        self._q_data_type = q_data_type
        self._kv_data_type = kv_data_type
        self._use_profiler = use_profiler
        self._plan_info = getattr(planned_backend, "_plan_info", None)

    # Output-only form -- ``return_lse=False`` returns the output tensor.
    # Preferred structural-input form.
    @overload
    def run(
        self,
        *,
        query: object,
        kv_cache: object,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> torch.Tensor: ...

    # Output-and-LSE form -- ``return_lse=True`` returns ``(output, lse)``.
    # This form must be requested by an LSE-capable plan; CUTLASS rejects it.
    @overload
    def run(
        self,
        *,
        query: object,
        kv_cache: object,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True],
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    # Deprecated separate-tensor output-only form. Pass split pairs through
    # ``query=`` and ``kv_cache=`` instead; positional arguments also warn.
    @overload
    def run(
        self,
        q_nope: Optional[torch.Tensor] = None,
        q_pe: Optional[torch.Tensor] = None,
        ckv_cache: Optional[torch.Tensor] = None,
        kpe_cache: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> torch.Tensor: ...

    # Deprecated separate-tensor output-and-LSE form. Pass split pairs through
    # ``query=`` and ``kv_cache=`` instead; positional arguments also warn.
    @overload
    def run(
        self,
        q_nope: Optional[torch.Tensor] = None,
        q_pe: Optional[torch.Tensor] = None,
        ckv_cache: Optional[torch.Tensor] = None,
        kpe_cache: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True] = True,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @_warn_on_positional_mla_arguments
    @flashinfer_api(trace=mla_paged_decode_trace)
    def run(
        self,
        q_nope: Optional[torch.Tensor] = None,
        q_pe: Optional[torch.Tensor] = None,
        ckv_cache: Optional[torch.Tensor] = None,
        kpe_cache: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        *,
        query: object = None,
        kv_cache: object = None,
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run one planned MLA attention request.

        **Preferred structural input form**

        Pass ``query=`` and ``kv_cache=``. Each may be a packed tensor whose
        final dimension joins its no-PE and PE features, or a split pair of
        tensors. The representation must satisfy the layouts declared by
        :meth:`plan`. Adjacent split views can be reinterpreted as packed
        tensors without a copy; FA2 and FA3 also accept independent split
        inputs natively. Packed-native CUTLASS plans reject independent split
        KV caches instead of silently copying them.

        **Output forms**

        With ``return_lse=False`` (the default), the method returns the output
        tensor. With ``return_lse=True``, it returns ``(output, lse)`` when the
        canonical plan was created with a matching LSE mode.
        ``return_lse_base_on_e`` selects natural-log rather than base-2 LSE
        output. Deprecated flat CSR FA2/FA3 plans temporarily retain dynamic
        LSE behavior and emit a warning. Caller-provided ``out`` and ``lse``
        buffers are used directly and returned by identity.

        **Legacy separate-parameter form**

        The separate ``q_nope`` / ``q_pe`` and ``ckv_cache`` / ``kpe_cache``
        parameters remain a deprecated compatibility form. Pass the same split
        tensors through ``query=(q_nope, q_pe)`` and
        ``kv_cache=(ckv_cache, kpe_cache)`` instead. Structural split values
        are not deprecated. Positional arguments are deprecated independently;
        each warning is emitted once per wrapper instance.

        Parameters
        ----------
        query : object
            Preferred packed or split structural query value.
        kv_cache : object
            Preferred packed or split structural KV-cache value.
        q_nope, q_pe : Optional[torch.Tensor]
            Legacy split query tensors. Supply both together.
        ckv_cache, kpe_cache : Optional[torch.Tensor]
            Legacy split compressed-KV and PE cache tensors. Supply both
            together.
        out : Optional[torch.Tensor]
            Caller-owned output buffer. When ``o_scale`` is provided, it must
            be an FP8 tensor for CUTLASS output.
        lse : Optional[torch.Tensor]
            Caller-owned log-sum-exp output buffer for an LSE-enabled plan.
        return_lse : bool
            Return the LSE tensor in addition to the output.
        profiler_buffer : Optional[torch.Tensor]
            Backend profiler output buffer.
        kv_len, page_table : Optional[torch.Tensor]
            CUTLASS/cuTile metadata aliases. A planned request may omit them;
            the deprecated unplanned CUTLASS path requires both. Runtime
            metadata is a trusted hot-path input: every length must be
            nonnegative and fit within its page-table row, every live page ID
            must index ``kv_cache``, and callers must not mutate either tensor
            while a launch is in flight. These values are not synchronized to
            the host for validation so that CUDA-graph capture remains valid.
        return_lse_base_on_e : bool
            Return natural-log rather than base-2 LSE values.
        o_scale : Optional[float]
            Per-tensor FP8 output scale supported by CUTLASS.
        ckv_scale, ckv_scale_arr, kpe_scale : optional
            Per-tensor or per-token FP8 KV-cache scales required by plans that
            selected ``scale_mode="kv-per-tensor"``.

        Notes
        -----
        Positional arguments are deprecated; use keyword arguments for
        ``plan()`` and ``run()``. The separate ``q_nope`` / ``q_pe`` and
        ``ckv_cache`` / ``kpe_cache`` parameters are also deprecated; pass
        structural ``query=`` and ``kv_cache=`` values instead. An explicitly
        requested CUTLASS backend may still run without :meth:`plan` when both
        ``kv_len`` and ``page_table`` are supplied, but that compatibility path
        is also deprecated.
        """
        # ---------------------------------------------------------------------------
        # Normalize structural and legacy inputs
        # ---------------------------------------------------------------------------
        uses_legacy_query = query is None
        if query is None:
            if (q_nope is None) != (q_pe is None):
                raise ValueError("q_nope and q_pe must both be provided.")
            if q_nope is None:
                raise TypeError("run() requires query= or q_nope and q_pe.")
            query = (q_nope, q_pe)
        elif q_nope is not None or q_pe is not None:
            raise TypeError("pass either query= or q_nope/q_pe, not both.")
        uses_legacy_kv_cache = kv_cache is None
        if kv_cache is None:
            if (ckv_cache is None) != (kpe_cache is None):
                raise ValueError("ckv_cache and kpe_cache must both be provided.")
            if ckv_cache is None:
                raise TypeError("run() requires kv_cache= or ckv_cache and kpe_cache.")
            kv_cache = (ckv_cache, kpe_cache)
        elif ckv_cache is not None or kpe_cache is not None:
            raise TypeError("pass either kv_cache= or ckv_cache/kpe_cache, not both.")
        if uses_legacy_query or uses_legacy_kv_cache:
            self._warn_deprecated_legacy_tensor_arguments()

        # ---------------------------------------------------------------------------
        # Resolve plan state and handle legacy unplanned CUTLASS
        # ---------------------------------------------------------------------------
        planned_backend = getattr(self, "_planned_backend", None)
        is_unplanned_cutlass = planned_backend is None and self._backend == "cutlass"
        if planned_backend is None and not is_unplanned_cutlass:
            raise RuntimeError(
                "BatchMLAPagedAttentionWrapper.run() called before plan()."
            )

        if is_unplanned_cutlass:
            widths = (512, 64)
            _, query_dtype, query_shape = _structural_mla_input_facts(
                query, widths=widths, name="query"
            )
            _, kv_dtype, kv_shape = _structural_mla_input_facts(
                kv_cache, widths=widths, name="KV cache"
            )
            if return_lse:
                raise ValueError("return_lse is not supported with cutlass backend.")
            if lse is not None:
                raise ValueError("lse is not supported with cutlass backend.")
            if return_lse_base_on_e:
                raise ValueError(
                    "return_lse_base_on_e is not supported with cutlass backend."
                )
            if kv_len is None or page_table is None:
                raise ValueError(
                    "unplanned CUTLASS requires both kv_len and page_table metadata "
                    "when its dynamic batch or page size changes."
                )
            if o_scale is not None:
                if out is None:
                    raise ValueError(
                        "out tensor must be provided when o_scale is used for FP8 output."
                    )
                if out.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
                    raise ValueError(
                        "out must be an FP8 tensor when o_scale is provided, "
                        f"got {out.dtype}"
                    )
                output_scale = float(o_scale)
                if not math.isfinite(output_scale) or output_scale <= 0.0:
                    raise ValueError(
                        f"o_scale must be a finite positive value, got {o_scale}"
                    )
            _warn_from_external_caller(
                "Running an explicitly requested CUTLASS backend without first "
                "calling plan() is deprecated; call plan() with canonical dense "
                "metadata instead.",
                DeprecationWarning,
            )
            assert kv_len is not None and page_table is not None
            try:
                query_for_cutlass = _resolve_structural_mla_input(
                    query, desired="packed", widths=widths, name="query"
                )
            except ValueError:
                if isinstance(query, tuple) and len(query) == 2:
                    query_for_cutlass = torch.cat(query, dim=-1)
                else:
                    raise
            try:
                kv_for_cutlass = _resolve_structural_mla_input(
                    kv_cache, desired="packed", widths=widths, name="KV cache"
                )
            except ValueError:
                if isinstance(kv_cache, tuple) and len(kv_cache) == 2:
                    kv_for_cutlass = torch.cat(kv_cache, dim=-1)
                else:
                    raise
            return _BatchMLAPagedAttentionCutlassBackend.run_planless(
                float_workspace_buffer=self._float_workspace_buffer,
                query=query_for_cutlass,
                kv_cache=kv_for_cutlass,
                out=out,
                profiler_buffer=profiler_buffer,
                kv_len=kv_len,
                page_table=page_table,
                o_scale=o_scale,
                q_data_type=query_dtype,
                kv_data_type=kv_dtype,
                num_heads=query_shape[-2],
                page_size=kv_shape[-2],
            )

        # ---------------------------------------------------------------------------
        # Validate the declared run contract
        # ---------------------------------------------------------------------------
        contract = self._input_contract
        assert contract is not None
        if (
            getattr(self, "_legacy_flat_csr_plan", False)
            and self._planned_query_layout == "split"
            and self._planned_kv_cache_layout == "split"
            and contract.lse_mode == "none"
            and (return_lse or lse is not None or return_lse_base_on_e)
        ):
            self._warn_legacy_dynamic_lse_once()
            contract = replace(
                contract,
                lse_mode="basee" if return_lse_base_on_e else "base2",
            )
        contract.validate_run_options(
            out=out,
            lse=lse,
            return_lse=return_lse,
            return_lse_base_on_e=return_lse_base_on_e,
            o_scale=o_scale,
            ckv_scale=ckv_scale,
            ckv_scale_arr=ckv_scale_arr,
            kpe_scale=kpe_scale,
        )

        # ---------------------------------------------------------------------------
        # Lower structural inputs to backend layouts
        # ---------------------------------------------------------------------------
        widths = (contract.head_dim_ckv, contract.head_dim_kpe)
        assert widths[0] is not None and widths[1] is not None
        typed_widths = (widths[0], widths[1])
        backend_query_layout = self._planned_query_layout
        backend_kv_cache_layout = self._planned_kv_cache_layout
        assert backend_query_layout is not None
        assert backend_kv_cache_layout is not None
        query_for_backend = _resolve_structural_mla_input(
            query,
            desired=backend_query_layout,
            widths=typed_widths,
            name="query",
            accepted=contract.query_layout,
            expected_dtype=self._q_data_type,
            planned_dtype_name="q_data_type",
            split_leaf_names=("q_nope", "q_pe"),
        )
        kv_for_backend = _resolve_structural_mla_input(
            kv_cache,
            desired=backend_kv_cache_layout,
            widths=typed_widths,
            name="KV cache",
            accepted=contract.kv_cache_layout,
            expected_dtype=self._kv_data_type,
            planned_dtype_name="kv_data_type",
            split_leaf_names=("ckv_cache", "kpe_cache"),
        )

        # ---------------------------------------------------------------------------
        # Dispatch to the planned backend
        # ---------------------------------------------------------------------------
        planned_backend = self._planned_backend
        assert planned_backend is not None
        return planned_backend.run_from_wrapper(
            query=query_for_backend,
            kv_cache=kv_for_backend,
            out=out,
            lse=lse,
            return_lse=return_lse,
            profiler_buffer=profiler_buffer,
            kv_len=kv_len,
            page_table=page_table,
            return_lse_base_on_e=return_lse_base_on_e,
            o_scale=o_scale,
            ckv_scale=ckv_scale,
            ckv_scale_arr=ckv_scale_arr,
            kpe_scale=kpe_scale,
        )


__all__ = ["BatchMLAPagedAttentionWrapper", "MLAInputContract", "MLAPlanMetadata"]
