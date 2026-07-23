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

import logging
import warnings
from typing import Any, Literal, Optional, Protocol, Tuple, Union, cast, overload

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.api_logging import flashinfer_api
from flashinfer.trace.templates.attention import (
    mla_paged_decode_trace,
)
from flashinfer.utils import (
    determine_mla_backend,
    get_compute_capability,
)

from ._backends.cute_dsl_backend import (
    _BatchMLAPagedAttentionCuteDslBackend,
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
    _MLAWrapperPlanResult,
)


_WRAPPER_BACKEND_TYPES = {
    "fa2": _BatchMLAPagedAttentionFa2Backend,
    "fa3": _BatchMLAPagedAttentionFa3Backend,
    "cutlass": _BatchMLAPagedAttentionCutlassBackend,
    "trtllm-gen": _BatchMLAPagedAttentionTrtllmGenBackend,
    "cute-dsl": _BatchMLAPagedAttentionCuteDslBackend,
    "xqa": _BatchMLAPagedAttentionXqaBackend,
}


class _PlannedWrapperBackend(Protocol):
    def run_from_wrapper(
        self, **kwargs: Any
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]: ...


class _WrapperBackendFactory(Protocol):
    @classmethod
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> _MLAWrapperPlanResult: ...


logger = logging.getLogger(__name__)


get_mla_module = _get_mla_module
get_batch_mla_module = _get_batch_mla_module
get_trtllm_gen_fmha_module = _get_trtllm_gen_fmha_module


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

    _blackwell_auto_fallback_warned: bool = False

    @classmethod
    def _maybe_warn_blackwell_auto_fallback(
        cls, device: torch.device, selected_backend: str
    ) -> None:
        if cls._blackwell_auto_fallback_warned:
            return
        try:
            major, minor = get_compute_capability(device)
        except ValueError:
            return
        if major < 10:
            return
        cls._blackwell_auto_fallback_warned = True
        warnings.warn(
            f"BatchMLAPagedAttentionWrapper: backend='auto' selected "
            f"'{selected_backend}' on SM{major}{minor}, which is not Blackwell-native "
            f"and gives poor MLA decode performance. "
            f"For decode, use "
            f"flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla "
            f"(Blackwell-native trtllm-gen); backend='cutlass' is the closest "
            f"in-wrapper alternative but may be slower than this fallback for "
            f"decode shapes.",
            UserWarning,
            stacklevel=3,
        )

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
            An initial ``cutlass``, ``trtllm-gen``, ``cute-dsl``, or ``xqa`` plan may
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
            ``"trtllm-gen"``, ``"cute-dsl"``, or ``"xqa"``. Default ``"auto"``.

            ``"auto"`` first tries ``"fa3"`` on SM90a, else ``"fa2"``, then
            falls back to ``"cutlass"`` when the preferred backend reports the
            configuration as unsupported. On SM>=100 neither FA backend is
            Blackwell-native; for MLA decode prefer
            :func:`trtllm_batch_decode_with_kv_cache_mla`. The ``"cutlass"`` option
            in this wrapper is the closest in-wrapper alternative but may be
            slower than the fa2 fallback for decode shapes.

            ``"cutlass"`` uses the SM100/SM110 CUTLASS MLA decode kernel. Only
            ``float_workspace_buffer`` is required at construction. Public
            ``q_nope``/``q_pe`` and ``ckv_cache``/``kpe_cache`` inputs stay split
            and are concatenated internally. ``kv_len`` and ``page_table`` may
            be captured by ``plan()`` and omitted from ``run()``; planned
            metadata takes precedence over cheap-verified aliases supplied at
            run time. Deprecated: an explicitly requested CUTLASS backend may
            also run without a preceding ``plan()`` when both metadata tensors
            are supplied to ``run()``. Call ``plan()`` with canonical dense
            metadata before ``run()`` instead. This compatibility path will be
            removed in a future release.

            ``"trtllm-gen"`` uses the dense TRTLLM-GEN MLA decode path. It is
            explicit-only initially and is not considered by ``backend="auto"``.

            ``"cute-dsl"`` uses the Blackwell CuTe DSL MLA decode path. It is
            explicit-only and is never considered by ``backend="auto"``.

            ``"xqa"`` uses the SM120/SM121 XQA MLA decode path. It is
            explicit-only and is never considered by ``backend="auto"``. Its
            contiguous workspace must contain at least 128 MiB.
        """
        if backend not in (
            "auto",
            "fa2",
            "fa3",
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "xqa",
        ):
            raise ValueError(
                "backend must be one of 'auto', 'fa2', 'fa3', 'cutlass', "
                f"'trtllm-gen', 'cute-dsl', or 'xqa', got {backend!r}"
            )
        self._backend = backend
        self._selected_backend: Optional[str] = None
        self._backend_impl: Optional[object] = None

        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer
        self._generated_fa_workspace = _BatchMLAGeneratedFaWorkspace(self.device)
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr

    def _reject_unsafe_cuda_graph_replan(self) -> None:
        if self._use_cuda_graph and self._selected_backend in (
            "cutlass",
            "trtllm-gen",
            "cute-dsl",
            "xqa",
        ):
            raise ValueError(
                "CUDA graph dense backend replan is not supported for "
                f"{self._selected_backend!r}: "
                "the first plan remains valid for capture and replay, but replacing "
                "its metadata tensors would invalidate captured launch pointers."
            )

    def _plan_backend(self, backend: str, args: _MLAPlanArguments) -> None:
        backend_type = cast(
            type[_WrapperBackendFactory], _WRAPPER_BACKEND_TYPES[backend]
        )
        result = backend_type.plan_from_wrapper(args)
        self._backend_impl, self._selected_backend = result.backend_impl, backend
        logger.info(
            "BatchMLAPagedAttentionWrapper requested backend '%s' selected backend '%s'",
            self._backend,
            backend,
        )

    # The canonical CSR and dense metadata forms below are public-equivalent:
    # planning validates either supplied form, checks equivalence when both are
    # supplied, and lazily derives the selected backend's native form.  This
    # applies to backend="auto" as well as explicit backends.

    # Canonical CSR metadata form -- native for FA2 and FA3; retains the
    # established positional calling convention.
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
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

    # Canonical dense page-table metadata form -- native for CUTLASS,
    # TRTLLM-GEN, CuTe DSL, and XQA; all metadata is keyword-only.
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
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

    # Both canonical metadata forms. plan() validates that they describe the
    # same requests and page mapping before committing the backend plan.
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
        cum_seq_lens_q: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None: ...

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
        cum_seq_lens_q: Optional[torch.Tensor] = None,
        block_tables: Optional[torch.Tensor] = None,
        seq_lens: Optional[torch.Tensor] = None,
        max_q_len: Optional[int] = None,
        qk_nope_head_dim: Optional[int] = None,
        enable_pdl: Optional[bool] = None,
        is_var_seq: Optional[bool] = None,
        cute_dsl_impl: str = "auto",
        use_sinks: bool = False,
    ) -> None:
        r"""Plan from one or two equivalent canonical metadata forms.

        Every explicit backend and ``backend="auto"`` accepts either canonical
        form. Planning validates the supplied form and lazily resolves the
        selected backend's native form: CSR for FA2/FA3 and dense page tables
        for CUTLASS, TRTLLM-GEN, CuTe DSL, and XQA. Supplying both forms is an
        assertion that they describe the same requests and page mapping; the
        planner validates that equivalence before committing a backend plan.

        **CSR form**

        Uses ``qo_indptr``, ``kv_indptr``, ``kv_indices``, and
        ``kv_len_arr``.  This form supports the existing positional call
        convention.

        **Dense page-table form**

        Uses the keyword-only ``cum_seq_lens_q``, ``block_tables``, and
        ``seq_lens`` arguments with optional ``max_q_len``.

        Metadata and required common arguments explicitly set to ``None`` are
        treated as omitted. In particular, ``max_q_len=None`` derives the value
        from the supplied query metadata.

        For ``backend="cute-dsl"``, ``cute_dsl_impl`` selects ``"auto"``,
        ``"monolithic"``, or ``"modular"``.  ``use_sinks=True`` plans the
        modular sink-capable kernel, and ``is_var_seq`` must match whether the
        planned KV ``seq_lens`` vary.  Query lengths must remain uniform.  CuTe
        DSL is explicit-only and is never an automatic-backend candidate.

        """
        self._reject_unsafe_cuda_graph_replan()
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
            qo_indptr=qo_indptr,
            kv_indptr=kv_indptr,
            kv_indices=kv_indices,
            kv_len_arr=kv_len_arr,
            num_heads=cast(int, num_heads),
            head_dim_ckv=cast(int, head_dim_ckv),
            head_dim_kpe=cast(int, head_dim_kpe),
            page_size=cast(int, page_size),
            causal=cast(bool, causal),
            sm_scale=cast(float, sm_scale),
            q_data_type=cast(torch.dtype, q_data_type),
            kv_data_type=cast(torch.dtype, kv_data_type),
            use_profiler=use_profiler,
            cum_seq_lens_q=cum_seq_lens_q,
            block_tables=block_tables,
            seq_lens=seq_lens,
            max_q_len=max_q_len,
            qk_nope_head_dim=qk_nope_head_dim,
            enable_pdl=enable_pdl,
            is_var_seq=is_var_seq,
            cute_dsl_impl=cute_dsl_impl,
            use_sinks=use_sinks,
            _float_workspace_buffer=self._float_workspace_buffer,
            _generated_fa_workspace=self._generated_fa_workspace,
            _use_cuda_graph=self._use_cuda_graph,
            _qo_indptr_buf=self._qo_indptr_buf,
            _kv_indptr_buf=self._kv_indptr_buf,
            _kv_indices_buf=self._kv_indices_buf,
            _kv_len_arr_buf=self._kv_len_arr_buf,
        )
        if self._backend != "auto":
            self._plan_backend(self._backend, plan_args)
            return

        # auto backend selection logic, TODO @mingyangw
        if self._use_cuda_graph and self._selected_backend in ("fa2", "fa3"):
            candidates = [self._selected_backend]
        else:
            preferred_backend = determine_mla_backend(self.device)
            candidates = [preferred_backend]
            if preferred_backend != "cutlass":
                candidates.append("cutlass")

        rejections = []
        last_rejection = None
        for candidate in candidates:
            try:
                if candidate not in ("fa2", "fa3", "cutlass"):
                    raise ValueError(f"unsupported automatic MLA backend {candidate!r}")
                if candidate in ("fa2", "fa3"):
                    self._maybe_warn_blackwell_auto_fallback(self.device, candidate)
                self._plan_backend(candidate, plan_args)
            except _BackendPlanUnsupportedError as err:
                last_rejection = err
                reason = str(err)
                rejections.append((candidate, reason))
                logger.debug(
                    "BatchMLAPagedAttentionWrapper automatically rejected backend '%s': %s",
                    candidate,
                    reason,
                )
                continue

            return

        candidate_names = ", ".join(candidates)
        rejection_summary = "; ".join(
            f"{candidate}: {reason}" for candidate, reason in rejections
        )
        raise _BackendPlanUnsupportedError(
            f"backend='auto' rejected all candidates [{candidate_names}]: "
            f"{rejection_summary}"
        ) from last_rejection

    # Output-only form -- ``return_lse=False`` returns the output tensor.
    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
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

    # Output-and-LSE form -- ``return_lse=True`` returns ``(output, lse)``.
    # Unsupported by CUTLASS and XQA, and by modular CuTe DSL.
    @overload
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
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

    @flashinfer_api(trace=mla_paged_decode_trace)
    def run(
        self,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

        **Output-only form**

        With ``return_lse=False``, returns the output tensor.

        **Output-and-LSE form**

        With ``return_lse=True``, returns a tuple containing the output tensor
        and the log-sum-exp tensor.

        Parameters
        ----------
        q_nope : torch.Tensor
            The query tensor without rope, shape: ``[batch_size, num_heads, head_dim_ckv]``.
        q_pe : torch.Tensor
            The rope part of the query tensor, shape: ``[batch_size, num_heads, head_dim_kpe]``.
        ckv_cache : torch.Tensor
            The compressed kv-cache tensor (without rope), shape: ``[num_pages, page_size, head_dim_ckv]``.
            ``head_dim_ckv`` is 512 in DeepSeek v2/v3 models.
        kpe_cache : torch.Tensor
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
            Per-head float32 attention sinks.  For ``backend="cute-dsl"``,
            sinks must be planned with ``use_sinks=True`` and use the modular
            implementation.
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
        Running an explicitly requested CUTLASS backend without first calling
        :meth:`plan` is deprecated. Call ``plan()`` with canonical dense
        metadata before ``run()`` instead. This compatibility path will be
        removed in a future release.

        Notes
        -----
        The CuTe DSL monolithic implementation supports LSE output; the
        modular implementation does not. XQA does not support LSE output.
        """
        has_fused_scale = bmm1_scale is not None or bmm2_scale is not None
        has_per_tensor_scale = ckv_scale is not None or kpe_scale is not None
        if has_fused_scale and has_per_tensor_scale:
            raise ValueError(
                "fused bmm scales and ckv_scale / kpe_scale are mutually exclusive."
            )
        bmm1_is_tensor = isinstance(bmm1_scale, torch.Tensor)
        bmm2_is_tensor = isinstance(bmm2_scale, torch.Tensor)
        if bmm1_is_tensor != bmm2_is_tensor and self._selected_backend != "xqa":
            raise ValueError(
                "bmm1_scale and bmm2_scale must be supplied together as a tensor pair."
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
                q_nope=q_nope,
                q_pe=q_pe,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
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
        elif self._selected_backend in _WRAPPER_BACKEND_TYPES:
            assert self._backend_impl is not None
            backend_impl = cast(_PlannedWrapperBackend, self._backend_impl)
            result = backend_impl.run_from_wrapper(
                q_nope=q_nope,
                q_pe=q_pe,
                ckv_cache=ckv_cache,
                kpe_cache=kpe_cache,
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
