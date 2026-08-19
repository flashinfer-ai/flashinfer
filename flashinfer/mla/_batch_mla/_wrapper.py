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
import math
import warnings
from typing import (
    Any,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    Union,
    cast,
    overload,
)

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import (
    mla_paged_decode_trace,
)
from flashinfer.autotuner.autotuner import _get_autotune_context_mode
from flashinfer.utils import get_compute_capability
from ._backends.cute_dsl_modular_backend import (
    _BatchMLAPagedAttentionCuteDslModularBackend,
)
from ._backends.cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)

# Private imports preserve the Batch MLA core's compatibility surface.
from ._backends.cutlass_backend import _BatchMLAPagedAttentionCutlassBackend
from ._backends.fa2_backend import _BatchMLAPagedAttentionFa2Backend
from ._backends.fa3_backend import _BatchMLAPagedAttentionFa3Backend
from ._backends._fa_common import _BatchMLAGeneratedFaWorkspace
from ._backends.trtllm_gen_backend import _BatchMLAPagedAttentionTrtllmGenBackend
from ._backends.xqa_backend import (
    _BatchMLAPagedAttentionXqaBackend,
)
from ._planning import (
    _MLAPlanArguments,
)
from ._contracts import (
    MLAInputContract,
    MLAPlanMetadata,
    _structural_mla_input_facts,
)
from . import _auto_policy


class _PlannedWrapperBackend(Protocol):
    def run_from_wrapper(
        self, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...


_BATCH_MLA_BACKENDS: dict[str, Any] = {
    "fa2": _BatchMLAPagedAttentionFa2Backend,
    "fa3": _BatchMLAPagedAttentionFa3Backend,
    "cutlass": _BatchMLAPagedAttentionCutlassBackend,
    "trtllm-gen": _BatchMLAPagedAttentionTrtllmGenBackend,
    "cute-dsl-monolithic": _BatchMLAPagedAttentionCuteDslMonolithicBackend,
    "cute-dsl-modular": _BatchMLAPagedAttentionCuteDslModularBackend,
    "xqa": _BatchMLAPagedAttentionXqaBackend,
}


def _warn_on_positional_mla_arguments(method: Any) -> Any:
    """Warn once per wrapper while preserving positional-call compatibility."""

    @functools.wraps(method)
    def wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
        if args:
            self._warn_deprecated_positional_arguments()
            if method.__name__ == "run":
                legacy_names = (
                    "q_nope",
                    "q_pe",
                    "ckv_cache",
                    "kpe_cache",
                    "out",
                    "lse",
                    "return_lse",
                    "profiler_buffer",
                    "kv_len",
                    "page_table",
                    "return_lse_base_on_e",
                    "o_scale",
                )
                if len(args) > len(legacy_names):
                    raise TypeError(
                        f"run() takes at most {len(legacy_names)} positional "
                        f"arguments ({len(args)} given)"
                    )
                for name, value in zip(legacy_names, args, strict=False):
                    if name in kwargs:
                        raise TypeError(
                            f"run() got multiple values for argument {name!r}"
                        )
                    kwargs[name] = value
                return method(self, **kwargs)
        return method(self, *args, **kwargs)

    return wrapped


def _raise_planned_run_argument_mismatch(
    name: str, planned: object, actual: object
) -> None:
    raise ValueError(
        f"MLA planned run argument {name} mismatch: planned {planned!r}, got "
        f"{actual!r}; re-plan with the needed arguments."
    )


def _validate_planned_run_arguments(
    *,
    planned_lse_mode: str,
    planned_output_dtype: torch.dtype,
    planned_output_scale: str,
    planned_scale_mode: str,
    planned_skip_softmax: bool,
    out: Optional[torch.Tensor],
    return_lse: bool,
    lse: Optional[torch.Tensor],
    return_lse_base_on_e: bool,
    o_scale: Optional[float],
    ckv_scale: Optional[float],
    ckv_scale_arr: Optional[torch.Tensor],
    kpe_scale: Optional[float],
    bmm1_scale: Optional[Union[float, torch.Tensor]],
    bmm2_scale: Optional[Union[float, torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
    allow_default_kv_scale: bool = False,
) -> None:
    actual_lse_mode = "none"
    if return_lse or lse is not None:
        actual_lse_mode = "basee" if return_lse_base_on_e else "base2"
    if actual_lse_mode != planned_lse_mode:
        _raise_planned_run_argument_mismatch(
            "LSE mode", planned_lse_mode, actual_lse_mode
        )
    actual_output_dtype = planned_output_dtype if out is None else out.dtype
    if actual_output_dtype != planned_output_dtype:
        _raise_planned_run_argument_mismatch(
            "output dtype", planned_output_dtype, actual_output_dtype
        )
    actual_output_scale = "per-tensor" if o_scale is not None else "none"
    if actual_output_scale != planned_output_scale:
        _raise_planned_run_argument_mismatch(
            "o_scale", planned_output_scale, actual_output_scale
        )
    if (ckv_scale is not None or ckv_scale_arr is not None) and kpe_scale is not None:
        actual_scale_mode = "kv-per-tensor"
    elif ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None:
        actual_scale_mode = "incomplete-kv-per-tensor"
    elif bmm1_scale is not None and bmm2_scale is not None:
        bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
        bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
        if bmm1_is_tensor != bmm2_is_tensor:
            actual_scale_mode = "mixed-bmm"
        elif bmm1_is_tensor:
            actual_scale_mode = "bmm-tensor"
        else:
            actual_scale_mode = "bmm-scalar"
    elif bmm1_scale is not None or bmm2_scale is not None:
        actual_scale_mode = "incomplete-bmm"
    else:
        actual_scale_mode = "default"
    if actual_scale_mode != planned_scale_mode and not (
        allow_default_kv_scale
        and planned_scale_mode == "default"
        and actual_scale_mode == "kv-per-tensor"
    ):
        _raise_planned_run_argument_mismatch(
            "scale mode", planned_scale_mode, actual_scale_mode
        )

    actual_skip_softmax = skip_softmax_threshold_scale_factor is not None
    if actual_skip_softmax != planned_skip_softmax:
        _raise_planned_run_argument_mismatch(
            "skip-softmax", planned_skip_softmax, actual_skip_softmax
        )


class BatchMLAPagedAttentionWrapper:
    r"""Wrapper class for MLA (`Multi-head Latent Attention <https://arxiv.org/abs/2405.04434>`_)
    PagedAttention on DeepSeek models. This kernel can be used in decode, and incremental prefill
    and should be used together with `Matrix Absorption trick
    <https://github.com/madsys-dev/deepseekv2-profile/blob/main/workspace/blog/optimizing-mla.md>`_:
    where :math:`W_{UQ}` is absorbed with :math:`W_{UK}`, and :math:`W_{UV}` is
    absorbed with :math:`W_{O}`.
    For MLA attention without Matrix Absorption (``head_dim_qk=192`` and ``head_dim_vo=128``, which is
    used in prefilling self-attention stage), please use
    :class:`flashinfer.prefill.BatchPrefillWithRaggedKVCacheWrapper`.

    More information about The Paged KV-Cache layout in MLA is explained in our tutorial
    :ref:`MLA Page Layout <mla-page-layout>`.

    For more details about the MLA computation, Matrix Absorption and FlashInfer's MLA implementation,
    please refer to our `blog post <http://flashinfer.ai/2025/02/10/flashinfer-deepseek-mla.html>`_.

    ``backend="auto"`` deterministically promotes an architecture-preferred
    candidate: FA3 on SM90, TRTLLM-GEN on SM100/SM103, and XQA on SM120/SM121.
    SM80/SM89 and unrecognized architectures retain the conservative FA2-first
    order. Every ordering remains a complete candidate list, and backend
    planners may fall through only when they report the request unsupported.

    When run inside :func:`flashinfer.autotune(True)`, ``backend="auto"``
    profiles compatible concrete wrapper backends over synthetic batch
    buckets. Inside :func:`flashinfer.autotune(False)`, it performs cache-only
    selection. :meth:`run` always dispatches directly to the concrete backend
    selected by the successful plan.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_local_heads = 128
    >>> batch_size = 114
    >>> head_dim_ckv = 512
    >>> head_dim_kpe = 64
    >>> page_size = 1
    >>> mla_wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    ...     torch.empty(128 * 1024 * 1024, dtype=torch.int8).to(0),
    ...     backend="fa2"
    ... )
    >>> q_indptr = torch.arange(0, batch_size + 1).to(0).int() # for decode, each query length is 1
    >>> kv_lens = torch.full((batch_size,), 999, dtype=torch.int32).to(0)
    >>> kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * 999
    >>> kv_indices = torch.arange(0, batch_size * 999).to(0).int()
    >>> q_nope = torch.randn(
    ...     batch_size * 1, num_local_heads, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> q_pe = torch.zeros(
    ...     batch_size * 1, num_local_heads, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> ckv = torch.randn(
    ...     batch_size * 999, 1, head_dim_ckv, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> kpe = torch.zeros(
    ...     batch_size * 999, 1, head_dim_kpe, dtype=torch.bfloat16, device="cuda"
    ... )
    >>> sm_scale = 1.0 / ((128 + 64) ** 0.5)  # use head dimension before matrix absorption
    >>> mla_wrapper.plan(
    ...     q_indptr,
    ...     kv_indptr,
    ...     kv_indices,
    ...     kv_lens,
    ...     num_local_heads,
    ...     head_dim_ckv,
    ...     head_dim_kpe,
    ...     page_size,
    ...     False,  # causal
    ...     sm_scale,
    ...     q_nope.dtype,
    ...     ckv.dtype,
    ... )
    >>> o = mla_wrapper.run(q_nope, q_pe, ckv, kpe, return_lse=False)
    >>> o.shape
    torch.Size([114, 128, 512])
    """

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
        r"""Constructor for BatchMLAPagedAttentionWrapper.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved workspace buffer used to store intermediate attention results in
            split-k algorithm. The recommended size is 128MB, the device of the workspace buffer
            should be the same as the device of the input tensors. The XQA wrapper backend
            requires at least 128 MiB and initializes its live semaphore range during planning.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
            An initial ``cutlass``, ``trtllm-gen``, CuTe DSL, or ``xqa`` plan may
            be captured and replayed, but replanning those dense backends is rejected
            because they do not accept caller-reserved metadata buffers for
            pointer-stable replacement.
        qo_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``qo_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.  The wrapper
            copies into this buffer at :meth:`plan` time so capture-time pointers remain
            stable.
        kv_indptr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indptr`` array, shape ``[batch_size + 1]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        kv_indices : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_indices`` array, sized to the maximum
            expected number of pages, dtype ``int32``.  Only consulted when
            ``use_cuda_graph=True``.
        kv_len_arr : Optional[torch.Tensor]
            User-reserved buffer to back the ``kv_len_arr`` array, shape ``[batch_size]``,
            dtype ``int32``.  Only consulted when ``use_cuda_graph=True``.
        backend : str
            One of ``"auto"``, ``"fa2"``, ``"fa3"``, ``"cutlass"``,
            ``"trtllm-gen"``, ``"cute-dsl"``, ``"cute-dsl-monolithic"``,
            ``"cute-dsl-modular"``, or ``"xqa"``. Default ``"auto"``.

            ``"auto"`` normally ranks every wrapper backend (as defined in
            `_auto_policy.py`), then asks each backend planner in order until
            one accepts the request. Inside ``autotune(True)`` it profiles that
            candidate set; inside ``autotune(False)`` it performs cache-only
            selection. Neither mode adds work to ``run()``.

            ``"cutlass"`` uses the SM100/SM110 CUTLASS MLA decode kernel. Only
            ``float_workspace_buffer`` is required at construction. Pass
            packed ``kv_cache`` or adjacent split cache views to satisfy the
            zero-copy packed-native contract. Planned CUTLASS runs reject
            non-adjacent split caches. ``kv_len`` and
            ``page_table`` may be captured by ``plan()`` and omitted from ``run()``; planned
            metadata takes precedence over cheap-verified aliases supplied at
            run time. Deprecated: an explicitly requested CUTLASS backend may
            also run without a preceding ``plan()`` when both metadata tensors
            are supplied to ``run()``. This deprecated path preserves the
            historical behavior of concatenating independent split inputs on
            the GPU. Call ``plan()`` with canonical dense metadata before
            ``run()`` instead. This compatibility path will be removed in a
            future release.

            ``"trtllm-gen"`` uses the dense TRTLLM-GEN MLA decode path.

            Requesting ``backend="cute-dsl"`` selects between the distinct
            ``"cute-dsl-monolithic"`` and ``"cute-dsl-modular"`` implementations.
            It selects modular for ``use_sinks=True``; otherwise it tries
            monolithic first and falls back to modular only for an unsupported
            plan. Select either concrete name to require that implementation.

            ``"xqa"`` uses the SM120/SM121 XQA MLA decode path. Its contiguous
            workspace must contain at least 128 MiB.
        """
        if backend not in (
            "auto",
            "fa2",
            "fa3",
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "cute-dsl-monolithic",
            "cute-dsl-modular",
            "xqa",
        ):
            raise ValueError(
                "backend must be one of 'auto', 'fa2', 'fa3', 'cutlass', "
                "'trtllm-gen', 'cute-dsl', 'cute-dsl-monolithic', "
                f"'cute-dsl-modular', or 'xqa', got {backend!r}"
            )
        self._backend = backend
        self._selected_backend: Optional[str] = None
        self._backend_impl: Optional[object] = None
        self._input_contract: Optional[MLAInputContract] = None
        self._auto_selection_trace: Optional[_auto_policy.MLAAutoSelectionTrace] = None
        self._copy_legacy_cutlass_split_inputs = False
        self._warned_legacy_tensor_arguments = False
        self._warned_positional_arguments = False

        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer
        self._generated_fa_workspace = _BatchMLAGeneratedFaWorkspace(self.device)
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr

    @property
    def resolved_backend(self) -> Optional[str]:
        """Concrete backend resolved by the most recent successful plan.

        This is ``None`` before the first successful plan. It is populated for
        both explicit and automatic wrappers.
        """
        return self._selected_backend

    @property
    def auto_selection_trace(self) -> Optional[_auto_policy.MLAAutoSelectionTrace]:
        """Immutable result of the latest successful automatic plan.

        The record contains the ordered concrete ``candidates``,
        planner-declared unsupported ``rejections``, and the ``resolved_backend``.
        It is ``None`` for explicit wrappers. A failed
        replan does not replace the prior successful trace.
        """
        return self._auto_selection_trace

    @property
    def auto_backend_candidates(self) -> tuple[str, ...]:
        trace = self._auto_selection_trace
        return () if trace is None else trace.candidates

    @property
    def auto_backend_rejections(self) -> tuple[tuple[str, str], ...]:
        trace = self._auto_selection_trace
        return () if trace is None else trace.rejections

    def _generated_fa_backend(self) -> Any:
        """Return generated-FA state exposed by the legacy wrapper contract.

        SGLang's CUDA-graph replay planner reaches this state through the
        wrapper.  Keep those aliases while the backend redesign owns the
        underlying objects, without copying state or adding run-path work.
        """
        if self._selected_backend not in ("fa2", "fa3") or self._backend_impl is None:
            raise AttributeError(
                "generated-FA planning state is available only after a successful "
                "FA2 or FA3 plan"
            )
        return self._backend_impl

    @property
    def _cached_module(self) -> Any:
        return self._generated_fa_backend()._cached_module

    @property
    def _int_workspace_buffer(self) -> torch.Tensor:
        return self._generated_fa_backend()._int_workspace_buffer

    @property
    def _pin_memory_int_workspace_buffer(self) -> torch.Tensor:
        return self._generated_fa_backend()._pin_memory_int_workspace_buffer

    def _reject_unsafe_cuda_graph_replan(self) -> None:
        if self._use_cuda_graph and self._selected_backend in (
            "cutlass",
            "trtllm-gen",
            "cute-dsl-monolithic",
            "cute-dsl-modular",
            "xqa",
        ):
            raise ValueError(
                "CUDA graph dense backend replan is not supported for "
                f"{self._selected_backend!r}: "
                "the first plan remains valid for capture and replay, but replacing "
                "its metadata tensors would invalidate captured launch pointers."
            )

    def _warn_deprecated_legacy_tensor_arguments(self) -> None:
        if self._warned_legacy_tensor_arguments:
            return
        warnings.warn(
            "Legacy MLA tensor arguments are deprecated; pass structural "
            "query and kv_cache values instead. Legacy split tensors are "
            "ignored when their structural replacement is supplied. The "
            "compatibility path will be removed in a future release.",
            DeprecationWarning,
            stacklevel=4,
        )
        self._warned_legacy_tensor_arguments = True

    def _warn_deprecated_positional_arguments(self) -> None:
        if self._warned_positional_arguments:
            return
        warnings.warn(
            "Positional MLA arguments are deprecated; pass plan() and run() "
            "arguments by keyword instead. Positional calling will be removed "
            "in a future release.",
            DeprecationWarning,
            stacklevel=3,
        )
        self._warned_positional_arguments = True

    def _plan_backend(self, backend: str, args: _MLAPlanArguments) -> None:
        backend_type = _BATCH_MLA_BACKENDS[backend]
        self._backend_impl = backend_type.plan_from_wrapper(args)
        self._selected_backend = backend
        self._input_contract = args.input_contract
        self._copy_legacy_cutlass_split_inputs = False

    def _publish_auto_plan(
        self, args: _MLAPlanArguments, result: _auto_policy._MLAAutoPlanResult
    ) -> None:
        """Atomically publish a completed automatic selection result."""
        self._backend_impl = result.backend_impl
        self._selected_backend = result.backend_name
        self._input_contract = args.input_contract
        self._copy_legacy_cutlass_split_inputs = False
        self._auto_selection_trace = result.trace

    # Preferred form.
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
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        use_sinks: bool = False,
        lse_mode: Literal["none", "base2", "basee"] = "none",
        query_layout: Literal["packed", "split"] = "packed",
        kv_cache_layout: Literal["packed", "split"] = "packed",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal[
            "default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"
        ] = "default",
        skip_softmax: bool = False,
    ) -> None: ...

    # Legacy flat-metadata compatibility: Canonical CSR metadata form
    # native for FA2 and FA3
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
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        use_sinks: bool = False,
        lse_mode: Literal["none", "base2", "basee"] = "none",
        query_layout: Literal["packed", "split"] = "split",
        kv_cache_layout: Literal["packed", "split"] = "split",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal[
            "default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"
        ] = "default",
        skip_softmax: bool = False,
    ) -> None: ...

    # Legacy flat-metadata compatibility: Canonical dense page-table metadata form
    # native for CUTLASS, TRTLLM-GEN, CuTe DSL, and XQA
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
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        use_sinks: bool = False,
        lse_mode: Literal["none", "base2", "basee"] = "none",
        query_layout: Literal["packed", "split"] = "split",
        kv_cache_layout: Literal["packed", "split"] = "split",
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal[
            "default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"
        ] = "default",
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
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        use_sinks: bool = False,
        lse_mode: Literal["none", "base2", "basee"] = "none",
        query_layout: Optional[Literal["packed", "split"]] = None,
        kv_cache_layout: Optional[Literal["packed", "split"]] = None,
        output_dtype: Optional[torch.dtype] = None,
        output_scale: Literal["none", "per-tensor"] = "none",
        scale_mode: Literal[
            "default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"
        ] = "default",
        skip_softmax: bool = False,
    ) -> None:
        r"""Plan from one or two equivalent canonical metadata forms.

        Every explicit backend and ``backend="auto"`` accepts either canonical
        form. Planning validates the supplied form and lazily resolves the
        selected backend's native form: CSR for FA2/FA3 and dense page tables
        for CUTLASS, TRTLLM-GEN, CuTe DSL, and XQA. Supplying both forms is an
        assertion that they describe the same requests and page mapping; the
        planner validates that equivalence before committing a backend plan.
        Pass ``metadata=MLAPlanMetadata.csr(...)``, ``.dense(...)``, or
        ``.dual(...)``.

        **CSR form**

        Prefer ``metadata=MLAPlanMetadata.csr(qo_indptr, kv_indptr,
        kv_indices, kv_len_arr)``. CSR is native to FA2 and FA3. Passing
        ``qo_indptr``, ``kv_indptr``, ``kv_indices``, and ``kv_len_arr``
        directly remains a deprecated compatibility form.

        **Dense page-table form**

        Prefer ``metadata=MLAPlanMetadata.dense(cum_seq_lens_q, block_tables,
        seq_lens, max_q_len=max_q_len)``. Dense metadata is native to CUTLASS,
        TRTLLM-GEN, CuTe DSL, and XQA. The corresponding flat keyword
        arguments remain a deprecated compatibility form.

        If both forms already exist, use ``MLAPlanMetadata.dual(...)``.

        Metadata and required common arguments explicitly set to ``None`` are
        treated as omitted. In particular, ``max_q_len=None`` derives the value
        from the supplied query metadata.

        ``lse_mode``, ``query_layout``, ``kv_cache_layout``, ``output_dtype``, ``output_scale``,
        ``scale_mode``, and ``skip_softmax`` declare the later ``run()``
        behavior this plan must support. With ``backend="auto"``, these
        values participate in compatibility-based backend selection. A
        successful plan retains them, and later ``run()`` calls must use the
        same behavior.

        Requesting ``backend="cute-dsl"`` selects between the distinct
        ``"cute-dsl-monolithic"`` and ``"cute-dsl-modular"`` implementations.
        It selects modular for ``use_sinks=True``; otherwise it tries monolithic
        first and falls back to modular only for an unsupported plan. Select
        either concrete name to require that implementation.

        ``is_var_seq`` must match whether the planned KV ``seq_lens`` vary.
        Query lengths must remain uniform.

        ``query_layout`` and ``kv_cache_layout`` describe the representations
        supplied to ``run()``. A packed plan accepts packed tensors or adjacent
        split views that can be reinterpreted without a copy; a split plan also
        accepts packed tensors, which are sliced into zero-copy views. When
        omitted, canonical ``metadata`` defaults to packed inputs while the
        deprecated flat-metadata forms retain their historical split-input
        default.

        With ``use_cuda_graph=True``, the initial successful plan can be used
        for capture and replay. Dense backends (CUTLASS, TRTLLM-GEN, both CuTe
        DSL implementations, and XQA) reject replanning because their metadata
        pointers cannot be replaced safely. An automatic wrapper that first
        resolves to FA2 or FA3 restricts later graph-mode replans to that same
        concrete backend and does not fall through to another backend.

        Parameters
        ----------
        qo_indptr : Optional[torch.Tensor]
            Deprecated flat CSR query indptr.
        kv_indptr : Optional[torch.Tensor]
            Deprecated flat CSR KV indptr.
        kv_indices : Optional[torch.Tensor]
            Deprecated flat CSR page indices.
        kv_len_arr : Optional[torch.Tensor]
            Deprecated flat CSR per-request KV lengths.
        num_heads : Optional[int]
            Number of query heads.
        head_dim_ckv : Optional[int]
            Compressed-KV feature width.
        head_dim_kpe : Optional[int]
            RoPE feature width.
        page_size : Optional[int]
            Number of KV tokens in each cache page.
        causal : Optional[bool]
            Whether the planned attention is causal.
        sm_scale : Optional[float]
            Softmax scale captured by the plan.
        q_data_type : Optional[torch.dtype]
            Query dtype supported by the planned backend.
        kv_data_type : Optional[torch.dtype]
            KV-cache dtype supported by the planned backend.
        use_profiler : bool
            Whether to enable backend profiler support.
        metadata : Optional[MLAPlanMetadata]
            Preferred canonical CSR, dense, or dual planning metadata.
        cum_seq_lens_q : Optional[torch.Tensor]
            Deprecated flat dense query offsets.
        block_tables : Optional[torch.Tensor]
            Deprecated flat dense page table.
        seq_lens : Optional[torch.Tensor]
            Deprecated flat dense per-request KV lengths.
        max_q_len : Optional[int]
            Maximum query length for dense metadata; inferred when omitted.
        qk_nope_head_dim : Optional[int]
            Optional non-RoPE query-key width required by TRTLLM-GEN.
        enable_pdl : Optional[bool]
            Whether supported backends enable programmatic dependent launch.
        is_var_seq : Optional[bool]
            Whether planned KV sequence lengths vary across requests.
        use_sinks : bool
            Whether the plan must support attention sinks.
        lse_mode : {"none", "base2", "basee"}
            Required log-sum-exp output mode.
        query_layout : Optional[{"packed", "split"}]
            Query representation accepted by subsequent ``run()`` calls.
        kv_cache_layout : Optional[{"packed", "split"}]
            KV-cache representation accepted by subsequent ``run()`` calls.
        output_dtype : Optional[torch.dtype]
            Required output dtype; defaults to the query dtype.
        output_scale : {"none", "per-tensor"}
            Required output scaling mode.
        scale_mode : {"default", "kv-per-tensor", "bmm-scalar", "bmm-tensor"}
            Required runtime scaling contract.
        skip_softmax : bool
            Whether the plan must support the skip-softmax threshold feature.

        Deprecated
        ----------
        Flat CSR and dense metadata arguments are deprecated. New code should
        pass ``metadata=MLAPlanMetadata.csr(...)``, ``.dense(...)``, or
        ``.dual(...)``. Positional arguments are also deprecated; use keyword
        arguments for all ``plan()`` parameters.

        """
        self._reject_unsafe_cuda_graph_replan()

        # Legacy and deprecation handling: Metadata tensors
        uses_flat_metadata = metadata is None
        if metadata is not None:
            if not isinstance(metadata, MLAPlanMetadata):
                raise TypeError("metadata must be an MLAPlanMetadata instance.")
            if any(
                value is not None
                for value in (
                    qo_indptr,
                    kv_indptr,
                    kv_indices,
                    kv_len_arr,
                    cum_seq_lens_q,
                    block_tables,
                    seq_lens,
                    max_q_len,
                )
            ):
                raise ValueError(
                    "Both metadata object and flat metadata arguments were provided; only one representation may be supplied (object preferred)."
                )
            plan_metadata = metadata
        else:
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

        # Normalize the supported metadata-driven planning contract.
        deprecated_plan_reasons: list[str] = []
        if uses_flat_metadata:
            deprecated_plan_reasons.append(
                "flat metadata arguments should be replaced with MLAPlanMetadata"
            )
        if query_layout is None:
            query_layout = "split" if uses_flat_metadata else "packed"
        if kv_cache_layout is None:
            kv_cache_layout = "split" if uses_flat_metadata else "packed"
        if query_layout not in ("packed", "split"):
            raise ValueError(f"unsupported query layout {query_layout!r}")
        if kv_cache_layout not in ("packed", "split"):
            raise ValueError(f"unsupported KV-cache layout {kv_cache_layout!r}")
        missing_plan_arguments = [
            name
            for name, value in (
                ("num_heads", num_heads),
                ("head_dim_ckv", head_dim_ckv),
                ("head_dim_kpe", head_dim_kpe),
                ("page_size", page_size),
                ("causal", causal),
                ("sm_scale", sm_scale),
                ("q_data_type", q_data_type),
                ("kv_data_type", kv_data_type),
            )
            if value is None
        ]
        if missing_plan_arguments:
            raise TypeError(
                "plan() missing required arguments: "
                + ", ".join(missing_plan_arguments)
            )
        query_kind: Literal["packed", "independent-split"] = (
            "packed" if query_layout == "packed" else "independent-split"
        )
        kv_kind: Literal["packed", "independent-split"] = (
            "packed" if kv_cache_layout == "packed" else "independent-split"
        )
        kv_layout: Literal["combined", "independent-split"] = (
            "combined" if kv_cache_layout == "packed" else "independent-split"
        )

        if deprecated_plan_reasons:
            warnings.warn(
                "Deprecated MLA planning arguments: "
                + "; ".join(deprecated_plan_reasons)
                + ".",
                DeprecationWarning,
                stacklevel=4,
            )

        plan_args = _MLAPlanArguments(
            metadata=plan_metadata,
            num_heads=cast(int, num_heads),
            head_dim_ckv=cast(int, head_dim_ckv),
            head_dim_kpe=cast(int, head_dim_kpe),
            page_size=cast(int, page_size),
            causal=cast(bool, causal),
            sm_scale=cast(float, sm_scale),
            q_data_type=cast(torch.dtype, q_data_type),
            kv_data_type=cast(torch.dtype, kv_data_type),
            query_kind=query_kind,
            kv_kind=kv_kind,
            query_layout=query_layout,
            kv_cache_layout=kv_cache_layout,
            lse_mode=lse_mode,
            kv_layout=kv_layout,
            output_dtype=(
                cast(torch.dtype, q_data_type) if output_dtype is None else output_dtype
            ),
            output_scale=output_scale,
            scale_mode=scale_mode,
            skip_softmax=skip_softmax,
            use_profiler=use_profiler,
            qk_nope_head_dim=qk_nope_head_dim,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            use_sinks=use_sinks,
            _float_workspace_buffer=self._float_workspace_buffer,
            _generated_fa_workspace=self._generated_fa_workspace,
            _use_cuda_graph=self._use_cuda_graph,
            _qo_indptr_buf=self._qo_indptr_buf,
            _kv_indptr_buf=self._kv_indptr_buf,
            _kv_indices_buf=self._kv_indices_buf,
            _kv_len_arr_buf=self._kv_len_arr_buf,
        )

        # Special handling for `cute-dsl` backend
        if self._backend == "cute-dsl":
            candidates = (
                ["cute-dsl-modular"]
                if use_sinks
                else ["cute-dsl-monolithic", "cute-dsl-modular"]
            )
            rejections = []
            last_rejection = None
            for candidate in candidates:
                try:
                    self._plan_backend(candidate, plan_args)
                except _BackendPlanUnsupportedError as err:
                    last_rejection = err
                    rejections.append((candidate, str(err)))
                    continue
                return
            rejection_summary = "; ".join(
                f"{candidate}: {reason}" for candidate, reason in rejections
            )
            raise _BackendPlanUnsupportedError(
                "backend='cute-dsl' rejected all family candidates: "
                f"{rejection_summary}"
            ) from last_rejection

        if self._backend != "auto":
            self._plan_backend(self._backend, plan_args)
            return

        # auto backend selection
        ranked_candidates = _auto_policy.rank_auto_backend_candidates(
            get_compute_capability(self.device) if self.device.type == "cuda" else None
        )
        # CUDA-graph replan path: preserve the previously selected FA backend.
        auto_candidates: Sequence[str]
        if self._use_cuda_graph and self._selected_backend in ("fa2", "fa3"):
            auto_candidates = (self._selected_backend,)
        else:
            auto_candidates = ranked_candidates

        result = _auto_policy.plan_auto_backend(
            plan_args,
            candidates=tuple(auto_candidates),
            backend_types=_BATCH_MLA_BACKENDS,
            autotune_mode=_get_autotune_context_mode(),
        )
        # Publish only after selection and the real winner plan have both
        # completed successfully. A failed replan leaves prior state live.
        self._publish_auto_plan(plan_args, result)

    # Output-only form -- ``return_lse=False`` returns the output tensor.
    # Preferred form.
    @overload
    def run(
        self,
        *,
        query: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ),
        kv_cache: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ),
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
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor: ...

    # Output-and-LSE form -- ``return_lse=True`` returns ``(output, lse)``.
    # Unsupported by CUTLASS, XQA, and modular CuTe DSL.
    @overload
    def run(
        self,
        *,
        query: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ),
        kv_cache: (
            torch.Tensor
            | tuple[torch.Tensor, torch.Tensor]
            | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
            | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
        ),
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
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    # Legacy split-tensor compatibility forms.
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
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor: ...

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
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @_warn_on_positional_mla_arguments
    @flashinfer_api(trace=mla_paged_decode_trace)
    def run(
        self,
        query: Optional[
            (
                torch.Tensor
                | tuple[torch.Tensor, torch.Tensor]
                | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
                | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            )
        ] = None,
        kv_cache: Optional[
            (
                torch.Tensor
                | tuple[torch.Tensor, torch.Tensor]
                | tuple[tuple[torch.Tensor, torch.Tensor], torch.Tensor]
                | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]
            )
        ] = None,
        *,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        q_nope: Optional[torch.Tensor] = None,  # deprecated, use ``query`` instead
        q_pe: Optional[torch.Tensor] = None,  # deprecated, use ``query`` instead
        ckv_cache: Optional[
            torch.Tensor
        ] = None,  # deprecated, use ``kv_cache`` instead
        kpe_cache: Optional[
            torch.Tensor
        ] = None,  # deprecated, use ``kv_cache`` instead
        ckv_scale: Optional[float] = None,
        ckv_scale_arr: Optional[torch.Tensor] = None,
        kpe_scale: Optional[float] = None,
        sinks: Optional[torch.Tensor] = None,
        skip_softmax_threshold_scale_factor: Optional[float] = None,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

        **Output-only form**

        With ``return_lse=False``, returns the output tensor.

        **Output-and-LSE form**

        With ``return_lse=True``, returns a tuple containing the output tensor
        and the log-sum-exp tensor.

        Supply exactly one complete tensor form for each group: a packed
        ``query`` and ``kv_cache``, or the split ``q_nope`` / ``q_pe`` and
        ``ckv_cache`` / ``kpe_cache`` references. Supplying both complete
        forms is a trusted redundant form. The wrapper forwards all six raw
        references unchanged, and the selected backend resolves its native
        form while ignoring the redundant alternate.

        **Split tensors**

        Adjacent split cache views can be
        reinterpreted as the packed form without a copy; independently
        allocated split caches remain supported by FA2/FA3. Packed-native
        backends reject independent split KV caches rather than implicitly
        copying the full cache.

        **Packed form**

        Use ``query`` and ``kv_cache`` when each pair of features is
        packed along its last dimension in one tensor. The wrapper passes the
        six raw packed/split tensor references and the planned input contract,
        without representation tags, to the selected backend. Each backend
        owns native-form resolution and uses dimensions captured by ``plan()``
        for zero-copy no-PE/PE views when conversion is needed. "Packed" is
        more precise than "stacked" or "contiguous" for this representation.

        Parameters
        ----------
        query : Optional[torch.Tensor or tuple[torch.Tensor, torch.Tensor]]
            Packed query tensor or split no-PE/PE query tensor pair, matching
            the representation declared by ``plan()``.
        kv_cache : Optional[torch.Tensor or tuple[torch.Tensor, torch.Tensor]]
            Packed KV cache or split compressed-KV/RoPE cache tensor pair,
            matching the representation declared by ``plan()``.
        q_nope : Optional[torch.Tensor]
            The query tensor without rope, shape:
            ``[batch_size, num_heads, head_dim_ckv]``.
            Provide together with ``q_pe``.
        q_pe : Optional[torch.Tensor]
            The rope part of the query tensor, shape:
            ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : Optional[torch.Tensor]
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models. Provide together
            with ``kpe_cache``.
        kpe_cache : Optional[torch.Tensor]
            The rope part of the kv-cache tensor, shape: ``[num_pages, page_size, head_dim_kpe]``.
            ``head_dim_kpe`` can be zero for NoPE MLA.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
            When ``o_scale`` is provided, this should be an FP8 tensor.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The KV length of each request, shape: ``[batch_size]``. For CUTLASS,
            this may be omitted when captured by ``plan()``. If supplied after
            planning, it must alias the same planned tensor view.
        page_table : Optional[torch.Tensor]
            The CUTLASS page table, shape: ``[batch_size, num_pages]``. This may
            be omitted when captured by ``plan()``; run-time values must be
            supplied together with ``kv_len`` and alias planned metadata when
            both exist. Both ``kv_len`` and ``page_table`` are required when an
            explicitly requested CUTLASS backend runs without ``plan()``.
        return_lse_base_on_e : bool, optional
            Controls the base of the returned LSE values when ``return_lse=True``.
            If ``False`` (default), the LSE is returned in base-2
            (``log2(sum(exp2(...)))``) to match the kernel's internal log-base.
            If ``True``, the LSE is converted to natural-log base (``log(sum(exp(...)))``)
            for compatibility with cascade-merging APIs that expect base-e LSEs.
        o_scale : Optional[float]
            FP8 output dequantization scale (``real = quantized * o_scale``).
            When provided, ``out`` must be an FP8 tensor. Only supported with
            the ``cutlass`` backend.
        ckv_scale : Optional[float]
            Per-tensor dequantization scale for the compressed-KV cache when
            ``kv_data_type`` is FP8 (``real = quantized * ckv_scale``). Exactly
            one of ``ckv_scale`` or ``ckv_scale_arr`` is required for FP8 KV.
        ckv_scale_arr : Optional[torch.Tensor]
            Contiguous float32 per-token, per-128-channel CKV scales with shape
            ``ckv_cache.shape[:-1] + (head_dim_ckv // 128,)``. Exactly one of
            ``ckv_scale`` or ``ckv_scale_arr`` is required for FP8 KV.
        kpe_scale : Optional[float]
            Per-tensor dequantization scale for the rope-K cache when
            ``kv_data_type`` is FP8 (``real = quantized * kpe_scale``).
        sinks : Optional[torch.Tensor]
            Per-head float32 attention sinks. For the CuTe DSL family, sinks
            must be planned with ``use_sinks=True`` and use
            ``backend="cute-dsl"`` or ``backend="cute-dsl-modular"``.
        skip_softmax_threshold_scale_factor : Optional[float]
            Runtime threshold scale for plans created with
            ``skip_softmax=True``. Unsupported backends reject the option.
        bmm1_scale : Optional[float]
            Finite run-time attention-logit scale override for CuTe DSL or
            XQA. If omitted, the ``sm_scale`` captured by ``plan()`` is used.
            The XQA wrapper accepts Python floats only; its functional API also
            supports a paired FP8 device-tensor scale mode.
        bmm2_scale : Optional[float]
            Finite run-time output scale override for CuTe DSL or XQA.
            Defaults to ``1.0``. These wrapper backends accept scalar Python
            floats, not scale tensors.

        Deprecated
        ----------
        Positional arguments are deprecated; use keyword arguments instead. The positional compatibility warning is
        emitted at most once per wrapper instance.

        Running an explicitly requested CUTLASS backend without first calling
        :meth:`plan` is deprecated. Call ``plan()`` with canonical dense
        metadata before ``run()`` instead. This compatibility path will be
        removed in a future release. Independent split query and KV-cache
        tensors are concatenated on the GPU on each run through this path.

        Non-adjacent split cache tensors are rejected by packed-native
        backends outside the deprecated unplanned-CUTLASS compatibility path.
        FA2 and FA3 continue to accept split KV tensors natively.

        Notes
        -----
        The CuTe DSL monolithic implementation supports LSE output; the
        modular implementation does not. XQA does not support LSE output.
        """

        def adapt_legacy_pair(
            *,
            name: str,
            structured: object,
            left: object,
            right: object,
        ) -> object:
            if structured is not None:
                if left is not None or right is not None:
                    self._warn_deprecated_legacy_tensor_arguments()
                return structured

            if left is not None and right is not None:
                self._warn_deprecated_legacy_tensor_arguments()
                return (left, right)

            raise ValueError(
                f"{name} requires a structural value or both legacy split arguments."
            )

        query = adapt_legacy_pair(
            name="query", structured=query, left=q_nope, right=q_pe
        )
        kv_cache = adapt_legacy_pair(
            name="kv_cache", structured=kv_cache, left=ckv_cache, right=kpe_cache
        )
        contract = self._input_contract
        has_fused_scale = bmm1_scale is not None or bmm2_scale is not None
        has_kv_scale = (
            ckv_scale is not None or ckv_scale_arr is not None or kpe_scale is not None
        )
        if has_fused_scale and has_kv_scale:
            raise ValueError(
                "fused bmm scales and ckv_scale / ckv_scale_arr / kpe_scale "
                "are mutually exclusive."
            )
        bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
        bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
        if (
            bmm1_is_tensor != bmm2_is_tensor
            and contract is None
            and self._selected_backend != "xqa"
        ):
            raise ValueError(
                "bmm1_scale and bmm2_scale must be supplied together as a tensor pair."
            )

        is_unplanned_cutlass = (
            self._selected_backend is None and self._backend == "cutlass"
        )
        if is_unplanned_cutlass or self._copy_legacy_cutlass_split_inputs:
            if is_unplanned_cutlass and (kv_len is None or page_table is None):
                raise ValueError(
                    "unplanned CUTLASS requires both kv_len and page_table metadata."
                )

            widths = (512, 64)
            query_kind, query_dtype, query_shape = _structural_mla_input_facts(
                query, widths=widths, name="query"
            )
            kv_kind, kv_dtype, kv_shape = _structural_mla_input_facts(
                kv_cache, widths=widths, name="KV cache"
            )
            if is_unplanned_cutlass:
                warnings.warn(
                    "Running an explicitly requested CUTLASS backend without first "
                    "calling plan() is deprecated; call plan() with canonical dense "
                    "metadata before run() instead. Independent split query or "
                    "KV-cache tensors are concatenated on the GPU on every run "
                    "through this compatibility path, which will be removed in a "
                    "future release.",
                    DeprecationWarning,
                    stacklevel=4,
                )
                assert kv_len is not None and page_table is not None
                self.plan(
                    metadata=MLAPlanMetadata.dense(
                        torch.arange(
                            query_shape[0] + 1,
                            dtype=torch.int32,
                            device=self.device,
                        ),
                        page_table,
                        kv_len,
                        max_q_len=1,
                    ),
                    num_heads=query_shape[-2],
                    head_dim_ckv=widths[0],
                    head_dim_kpe=widths[1],
                    page_size=kv_shape[-2],
                    causal=False,
                    sm_scale=1.0 / math.sqrt(128 + widths[1]),
                    q_data_type=query_dtype,
                    kv_data_type=kv_dtype,
                    query_layout="packed",
                    kv_cache_layout="packed",
                    output_dtype=None if out is None else out.dtype,
                    output_scale="per-tensor" if o_scale is not None else "none",
                )
                self._copy_legacy_cutlass_split_inputs = True

            if query_kind == "independent-split":
                query = torch.cat(
                    cast(tuple[torch.Tensor, torch.Tensor], query), dim=-1
                )
            if kv_kind == "independent-split":
                kv_cache = torch.cat(
                    cast(tuple[torch.Tensor, torch.Tensor], kv_cache), dim=-1
                )

        contract = self._input_contract
        if contract is not None:
            widths = (contract.head_dim_ckv, contract.head_dim_kpe)
            if None not in widths:
                for name, value, layout in (
                    ("query", query, contract.query_layout),
                    ("KV cache", kv_cache, contract.kv_cache_layout),
                ):
                    if layout == "packed":
                        kind, _, _ = _structural_mla_input_facts(
                            value,
                            widths=cast(tuple[int, int], widths),
                            name=name,
                        )
                        if kind == "independent-split":
                            raise ValueError(
                                f"{name} cannot provide the planned packed "
                                "representation zero-copy; re-plan for split input."
                            )
            _validate_planned_run_arguments(
                planned_lse_mode=contract.lse_mode,
                planned_output_dtype=contract.output_dtype,
                planned_output_scale=contract.output_scale,
                planned_scale_mode=contract.scale_mode,
                planned_skip_softmax=contract.skip_softmax,
                out=out,
                return_lse=return_lse,
                lse=lse,
                return_lse_base_on_e=return_lse_base_on_e,
                o_scale=o_scale,
                ckv_scale=ckv_scale,
                ckv_scale_arr=ckv_scale_arr,
                kpe_scale=kpe_scale,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                allow_default_kv_scale=self._selected_backend in ("fa2", "fa3"),
            )

        if self._selected_backend in _BATCH_MLA_BACKENDS:
            assert self._backend_impl is not None
            backend_impl = cast(_PlannedWrapperBackend, self._backend_impl)
            if ckv_scale_arr is None:
                result = backend_impl.run_from_wrapper(
                    query=query,
                    kv_cache=kv_cache,
                    out=out,
                    lse=lse,
                    return_lse=return_lse,
                    profiler_buffer=profiler_buffer,
                    kv_len=kv_len,
                    page_table=page_table,
                    return_lse_base_on_e=return_lse_base_on_e,
                    o_scale=o_scale,
                    ckv_scale=ckv_scale,
                    kpe_scale=kpe_scale,
                    sinks=sinks,
                    skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                )
            else:
                result = backend_impl.run_from_wrapper(
                    query=query,
                    kv_cache=kv_cache,
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
                    sinks=sinks,
                    skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
                    bmm1_scale=bmm1_scale,
                    bmm2_scale=bmm2_scale,
                )
        else:
            raise RuntimeError(
                f"BatchMLAPagedAttentionWrapper.run() received unexpected selected backend {self._selected_backend!r}"
                "\nDid you forget to call BatchMLAPagedAttentionWrapper.plan()?"
            )
        return result
