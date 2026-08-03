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
from flashinfer.utils import get_compute_capability
from ._backends.cute_dsl_modular_backend import (
    _BatchMLAPagedAttentionCuteDslModularBackend,
)
from ._backends.cute_dsl_monolithic_backend import (
    _BatchMLAPagedAttentionCuteDslMonolithicBackend,
)

# Private imports plus assignments preserve the Batch MLA core's compatibility surface.
from ._backends.cutlass_backend import (
    _BatchMLAPagedAttentionCutlassBackend,
    get_mla_module as _get_mla_module,
)
from ._backends.fa2_backend import _BatchMLAPagedAttentionFa2Backend
from ._backends.fa3_backend import _BatchMLAPagedAttentionFa3Backend
from ._backends._fa_common import (
    _BatchMLAGeneratedFaWorkspace,
    get_batch_mla_module as _get_batch_mla_module,
)
from ._backends.trtllm_gen_backend import (
    _BatchMLAPagedAttentionTrtllmGenBackend,
    get_trtllm_gen_fmha_module as _get_trtllm_gen_fmha_module,
)
from ._backends.xqa_backend import (
    _BatchMLAPagedAttentionXqaBackend,
)
from ._planning import (
    _MLAPlanArguments,
)
from ._contracts import (
    MLAInputContract,
    MLAKVCache,
    MLAPlanMetadata,
    MLAQuery,
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
        return method(self, *args, **kwargs)

    return wrapped


get_mla_module = _get_mla_module
get_batch_mla_module = _get_batch_mla_module
get_trtllm_gen_fmha_module = _get_trtllm_gen_fmha_module


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
    planned_kv_layout: str,
    planned_output_dtype: torch.dtype,
    planned_output_scale: str,
    planned_scale_mode: str,
    planned_skip_softmax: bool,
    out: Optional[torch.Tensor],
    return_lse: bool,
    lse: Optional[torch.Tensor],
    return_lse_base_on_e: bool,
    kv_layout: str,
    o_scale: Optional[float],
    ckv_scale: Optional[float],
    kpe_scale: Optional[float],
    bmm1_scale: Optional[Union[float, torch.Tensor]],
    bmm2_scale: Optional[Union[float, torch.Tensor]],
    skip_softmax_threshold_scale_factor: Optional[float],
) -> None:
    actual_lse_mode = "none"
    if return_lse or lse is not None:
        actual_lse_mode = "basee" if return_lse_base_on_e else "base2"
    if actual_lse_mode != planned_lse_mode:
        _raise_planned_run_argument_mismatch(
            "LSE mode", planned_lse_mode, actual_lse_mode
        )
    if kv_layout != planned_kv_layout:
        _raise_planned_run_argument_mismatch("KV layout", planned_kv_layout, kv_layout)

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

    if ckv_scale is not None and kpe_scale is not None:
        actual_scale_mode = "kv-per-tensor"
    elif ckv_scale is not None or kpe_scale is not None:
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
    if actual_scale_mode != planned_scale_mode:
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
    SM80/SM89; unrecognized architectures retain the conservative FA2-first
    order. Every ordering remains a complete candidate list, and backend
    planners may fall through only when they report the request unsupported.

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

            ``"auto"`` ranks every wrapper backend (as defined in `_auto_policy.py`), then asks each backend planner in order until one
            accepts the request.

            ``"cutlass"`` uses the SM100/SM110 CUTLASS MLA decode kernel. Only
            ``float_workspace_buffer`` is required at construction. Pass
            ``kv=MLAKVCache.packed(...)`` or adjacent split views through
            ``MLAKVCache.split(...)`` to avoid a run-time copy. Non-adjacent
            split caches remain a
            deprecated compatibility path for CUTLASS only. ``kv_len`` and
            ``page_table`` may be captured by ``plan()`` and omitted from ``run()``; planned
            metadata takes precedence over cheap-verified aliases supplied at
            run time. Deprecated: an explicitly requested CUTLASS backend may
            also run without a preceding ``plan()`` when both metadata tensors
            are supplied to ``run()``. Call ``plan()`` with canonical dense
            metadata before ``run()`` instead. This compatibility path will be
            removed in a future release.

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
        self._warned_nonadjacent_cutlass_cache = False
        self._warned_legacy_arguments = False
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

    def _warn_deprecated_legacy_arguments(self) -> None:
        if self._warned_legacy_arguments:
            return
        warnings.warn(
            "Legacy MLA metadata and split tensor arguments are deprecated; pass "
            "MLAPlanMetadata to plan() and MLAQuery and MLAKVCache to run() "
            "instead. The compatibility path will be removed in a "
            "future release.",
            DeprecationWarning,
            stacklevel=5,
        )
        self._warned_legacy_arguments = True

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
        self._input_contract = MLAInputContract(
            query_split_widths=(args.head_dim_ckv, args.head_dim_kpe),
            kv_split_widths=(args.head_dim_ckv, args.head_dim_kpe),
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            kv_layout=args.kv_layout,
            lse_mode=args.lse_mode,
            output_dtype=args.output_dtype,
            output_scale=args.output_scale,
            scale_mode=args.scale_mode,
            skip_softmax=args.skip_softmax,
        )
        # Backend adapters consume the contract when resolving value-object
        # representations.  Lightweight test doubles may be immutable objects.
        if hasattr(self._backend_impl, "__dict__"):
            backend_impl = cast(Any, self._backend_impl)
            backend_impl._input_contract = self._input_contract

    # Preferred value-object form
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
        kv_layout: Literal[
            "combined", "adjacent-split", "independent-split"
        ] = "independent-split",
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
        kv_layout: Literal[
            "combined", "adjacent-split", "independent-split"
        ] = "independent-split",
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
        kv_layout: Literal[
            "combined", "adjacent-split", "independent-split"
        ] = "independent-split",
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
        kv_layout: Literal[
            "combined", "adjacent-split", "independent-split"
        ] = "independent-split",
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

        ``lse_mode``, ``kv_layout``, ``output_dtype``, ``output_scale``,
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

        With ``use_cuda_graph=True``, the initial successful plan can be used
        for capture and replay. Dense backends (CUTLASS, TRTLLM-GEN, both CuTe
        DSL implementations, and XQA) reject replanning because their metadata
        pointers cannot be replaced safely. An automatic wrapper that first
        resolves to FA2 or FA3 restricts later graph-mode replans to that same
        concrete backend and does not fall through to another backend.

        Deprecated
        ----------
        Flat CSR and dense metadata arguments are deprecated. New code should
        pass ``metadata=MLAPlanMetadata.csr(...)``, ``.dense(...)``, or
        ``.dual(...)``. Positional arguments are also deprecated; use keyword
        arguments for all ``plan()`` parameters.

        """
        self._reject_unsafe_cuda_graph_replan()
        if metadata is None:
            self._warn_deprecated_legacy_arguments()
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
        common_values = {
            "num_heads": num_heads,
            "head_dim_ckv": head_dim_ckv,
            "head_dim_kpe": head_dim_kpe,
            "page_size": page_size,
            "causal": causal,
            "sm_scale": sm_scale,
            "q_data_type": q_data_type,
            "kv_data_type": kv_data_type,
        }
        missing_common = [
            name for name, value in common_values.items() if value is None
        ]
        if missing_common:
            raise TypeError(
                "plan() missing required arguments: " + ", ".join(missing_common)
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

        rejections = []
        last_rejection = None
        for candidate in auto_candidates:
            try:
                self._plan_backend(candidate, plan_args)
            except _BackendPlanUnsupportedError as err:
                last_rejection = err
                reason = str(err)
                rejections.append((candidate, reason))
                continue

            self._auto_selection_trace = _auto_policy.MLAAutoSelectionTrace(
                candidates=tuple(auto_candidates),
                rejections=tuple(rejections),
                resolved_backend=candidate,
            )
            return

        candidate_names = ", ".join(auto_candidates)
        rejection_summary = "; ".join(
            f"{candidate}: {reason}" for candidate, reason in rejections
        )
        raise _BackendPlanUnsupportedError(
            f"backend='auto' rejected all candidates [{candidate_names}]: "
            f"{rejection_summary}"
        ) from last_rejection

    # Output-only form –– ``return_lse=False`` returns the output tensor.
    # Preferred value-object form.
    @overload
    def run(
        self,
        *,
        query: MLAQuery,
        kv: MLAKVCache,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[False] = False,
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        ckv_scale: Optional[float] = None,
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
        query: MLAQuery,
        kv: MLAKVCache,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: Literal[True],
        profiler_buffer: Optional[torch.Tensor] = None,
        kv_len: Optional[torch.Tensor] = None,
        page_table: Optional[torch.Tensor] = None,
        return_lse_base_on_e: bool = False,
        o_scale: Optional[float] = None,
        ckv_scale: Optional[float] = None,
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
        query: Optional[MLAQuery] = None,
        kv: Optional[MLAKVCache] = None,
        ckv_scale: Optional[float] = None,
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

        Pass ``query=MLAQuery.packed(...)`` or ``.split(...)`` and
        ``kv=MLAKVCache.packed(...)`` or ``.split(...)``. Value objects
        cannot be combined with their split tensor arguments.

        **Split form**

        Use ``query=MLAQuery.split(q_nope, q_pe)`` and
        ``kv=MLAKVCache.split(ckv_cache, kpe_cache)`` when the feature tensors
        are supplied separately. Adjacent split cache views can be
        reinterpreted as the packed form without a copy; independently
        allocated split caches remain supported by FA2/FA3, but may be
        unsupported or require a compatibility copy for other backends.

        **Packed form**

        Use ``query=MLAQuery.packed(q)`` and
        ``kv=MLAKVCache.packed(kv_cache)`` when each pair of features is
        packed along its last dimension in one tensor. The wrapper uses the
        dimensions captured by ``plan()`` to make zero-copy no-PE/PE query and
        KV-cache views. The wrapper passes the value objects and planned widths
        to the selected backend; each backend requests only the split or packed
        zero-copy views it consumes. "Packed" is more precise than "stacked"
        or "contiguous" for this representation.

        Parameters
        ----------
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
            ``head_dim_kpe`` is 64 in DeepSeek v2/v3 models.
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
            ``kv_data_type`` is FP8 (``real = quantized * ckv_scale``). Required
            (together with ``kpe_scale``) for the FP8 KV cache path on the
            ``fa3`` backend. Must be a finite positive value. Must not be
            provided when ``kv_data_type`` is BF16/FP16.
        kpe_scale : Optional[float]
            Per-tensor dequantization scale for the rope-K cache when
            ``kv_data_type`` is FP8 (``real = quantized * kpe_scale``). Same
            usage rules as ``ckv_scale``.
        sinks : Optional[torch.Tensor]
            Per-head float32 attention sinks. For the CuTe DSL family, sinks
            must be planned with ``use_sinks=True`` and use
            ``backend="cute-dsl"`` or ``backend="cute-dsl-modular"``.
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
        Split query and KV-cache arguments are deprecated. New code should pass
        ``query=MLAQuery.packed(...)`` or ``.split(...)`` and
        ``kv=MLAKVCache.packed(...)`` or ``.split(...)``. Positional
        arguments are also deprecated; use keyword arguments for all ``run()``
        parameters. Each compatibility warning is emitted at most once per
        wrapper instance and the compatibility paths will be removed in a
        future release.

        Running an explicitly requested CUTLASS backend without first calling
        :meth:`plan` is deprecated. Call ``plan()`` with canonical dense
        metadata before ``run()`` instead. This compatibility path will be
        removed in a future release.

        Non-adjacent split cache tensors are also deprecated for CUTLASS and
        trigger a warning because they require a full concatenation at every
        run. TRTLLM-GEN, CuTe DSL, and XQA reject them; FA2 and FA3 continue to
        accept independent split tensors.

        Non-adjacent split query tensors remain accepted. Their fallback copy
        is bounded by the active query size, unlike a paged KV-cache copy.

        Notes
        -----
        The CuTe DSL monolithic implementation supports LSE output; the
        modular implementation does not. XQA does not support LSE output.
        """
        if query is None or kv is None:
            self._warn_deprecated_legacy_arguments()
        if query is not None:
            if not isinstance(query, MLAQuery):
                raise TypeError("query must be an MLAQuery instance.")
            if q_nope is not None or q_pe is not None:
                raise ValueError(
                    "Both query object and flat q_nope/q_pe arguments were provided; only one representation may be supplied (object preferred)."
                )
        else:
            query = MLAQuery.split(q_nope, q_pe)

        if kv is not None:
            if not isinstance(kv, MLAKVCache):
                raise TypeError("kv must be an MLAKVCache instance.")
            if ckv_cache is not None or kpe_cache is not None:
                raise ValueError(
                    "Both kv object and flat ckv_cache/kpe_cache arguments were provided; only one representation may be supplied (object preferred)."
                )
        else:
            kv = MLAKVCache.split(ckv_cache, kpe_cache)

        selected_backend = self._selected_backend or self._backend
        contract = self._input_contract
        if contract is not None:
            contract.validate(query, kv)
        adjacent_kv_cache = kv.packed_or_adjacent()
        actual_kv_layout = kv.layout
        if (
            selected_backend == "cutlass"
            and not self._warned_nonadjacent_cutlass_cache
            and isinstance(kv.ckv_cache, torch.Tensor)
            and isinstance(kv.kpe_cache, torch.Tensor)
            and adjacent_kv_cache is None
        ):
            warnings.warn(
                "Non-adjacent ckv_cache and kpe_cache for CUTLASS are "
                "concatenated on every run. Pass MLAKVCache.packed(...) or "
                "adjacent MLAKVCache.split(...) views to avoid the copy. This "
                "compatibility path is deprecated and will be removed in a future "
                "release.",
                FutureWarning,
                stacklevel=2,
            )
            self._warned_nonadjacent_cutlass_cache = True

        has_fused_scale = bmm1_scale is not None or bmm2_scale is not None
        has_per_tensor_scale = ckv_scale is not None or kpe_scale is not None
        if has_fused_scale and has_per_tensor_scale:
            raise ValueError(
                "fused bmm scales and ckv_scale / kpe_scale are mutually exclusive."
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

        if contract is not None:
            _validate_planned_run_arguments(
                planned_lse_mode=contract.lse_mode,
                planned_kv_layout=contract.kv_layout,
                planned_output_dtype=contract.output_dtype,
                planned_output_scale=contract.output_scale,
                planned_scale_mode=contract.scale_mode,
                planned_skip_softmax=contract.skip_softmax,
                out=out,
                return_lse=return_lse,
                lse=lse,
                return_lse_base_on_e=return_lse_base_on_e,
                kv_layout=actual_kv_layout,
                o_scale=o_scale,
                ckv_scale=ckv_scale,
                kpe_scale=kpe_scale,
                bmm1_scale=bmm1_scale,
                bmm2_scale=bmm2_scale,
                skip_softmax_threshold_scale_factor=skip_softmax_threshold_scale_factor,
            )

        if self._selected_backend is None and self._backend == "cutlass":
            warnings.warn(
                "Running an explicitly requested CUTLASS backend without first "
                "calling plan() is deprecated; call plan() with canonical dense "
                "metadata before run() instead. This compatibility path will be "
                "removed in a future release.",
                DeprecationWarning,
                stacklevel=2,
            )
            result = _BatchMLAPagedAttentionCutlassBackend.run_unplanned_from_wrapper(
                self._float_workspace_buffer,
                query=query,
                kv=kv,
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
        elif self._selected_backend in _BATCH_MLA_BACKENDS:
            assert self._backend_impl is not None
            backend_impl = cast(_PlannedWrapperBackend, self._backend_impl)
            result = backend_impl.run_from_wrapper(
                query=query,
                kv=kv,
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
            raise RuntimeError(
                f"BatchMLAPagedAttentionWrapper.run() received unexpected selected backend {self._selected_backend!r}"
                "\nDid you forget to call BatchMLAPagedAttentionWrapper.plan()?"
            )
        return result
