"""Shared functional CuTe DSL preparation and runner mechanics."""

import math
from dataclasses import replace
from typing import Any, Callable, List, Optional, Tuple, Union

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import is_in_profile_measurement
from flashinfer.utils import (
    _check_block_tables_shape,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    next_positive_power_of_2,
)

from .._contracts import (
    _FunctionalBackendUnsupportedError,
    _FunctionalMLARequest,
    _FunctionalMLARunner,
)
from ._cute_dsl_common import (
    _CuteDslKernelUnsupportedError,
    _CuteDslMlaExecutionState,
    _prepare_cute_dsl_mla_execution_state,
    _run_cute_dsl_mla_execution_state,
)


class _CuteDslImplementationUnsupportedReason(str):
    """Marks a concrete capability rejection that may try the sibling backend."""


def _cute_dsl_max_supported_batch(
    workspace_bytes: int,
    q_len: int,
    num_heads: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    candidate_max: int,
) -> int:
    """Largest batch the caller's workspace can support for CuTe DSL MLA."""
    from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
        _get_split_kv_and_workspace_size,
    )

    lo, hi = 1, max(1, candidate_max)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, workspace_size = _get_split_kv_and_workspace_size(
            mid, q_len, num_heads, kv_lora_rank, max_active_blocks
        )
        if workspace_size <= workspace_bytes:
            lo = mid
        else:
            hi = mid - 1
    return lo


def _cute_dsl_incompatibility_reason(
    query: torch.Tensor,
    out_dtype: torch.dtype,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
    sinks: Optional[List[torch.Tensor]],
    sparse_mla_top_k: int,
    skip_softmax_threshold_scale_factor: Optional[float],
    uses_shared_paged_kv_idx: bool,
    qk_rope_head_dim: int,
    kv_lora_rank: int,
    page_size: int,
    is_var_seq: bool,
    return_lse: bool,
    lse: Optional[torch.Tensor],
    cute_dsl_impl: str = "auto",
) -> Optional[str]:
    """Return ``None`` when CuTe DSL can implement the functional request."""
    del return_lse, lse
    cc = get_compute_capability(query.device)
    if cc[0] < 10:
        return _CuteDslImplementationUnsupportedReason(
            "cute-dsl backend (MLA decode kernel) requires SM100+, "
            f"got SM{cc[0]}{cc[1]}"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor "
            "bmm1_scale, please pass a float value"
        )
    if isinstance(bmm2_scale, torch.Tensor):
        return (
            "cute-dsl backend (MLA decode kernel) does not support tensor "
            "bmm2_scale, please pass a float value"
        )
    if isinstance(sinks, (list, tuple)) and len(sinks) != 1:
        return (
            "cute-dsl backend (MLA decode kernel) expects sinks to be a "
            f"single tensor or a length-1 list/tuple; got len={len(sinks)}"
        )
    if sparse_mla_top_k > 0:
        return "cute-dsl backend (MLA decode kernel) does not support sparse_mla_top_k"
    if skip_softmax_threshold_scale_factor is not None:
        return (
            "cute-dsl backend (MLA decode kernel) does not support "
            "skip_softmax_threshold_scale_factor"
        )
    if not uses_shared_paged_kv_idx:
        return (
            "cute-dsl backend (MLA decode kernel) does not support separate KV "
            "page indices (uses_shared_paged_kv_idx=False)"
        )

    _, q_len, num_heads, _ = query.shape
    try:
        from flashinfer.cute_dsl.attention.mla_dispatch import _resolve_impl

        resolved_impl = _resolve_impl(requested=cute_dsl_impl, kwargs={"sinks": sinks})
    except (ValueError, ImportError) as error:
        return f"cute-dsl backend (MLA decode kernel): {error}"

    try:
        if resolved_impl == "monolithic":
            from flashinfer.cute_dsl.attention.monolithic.mla_decode import (
                _check_can_implement,
            )
        else:
            from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
                _check_can_implement,
            )

        _check_can_implement(
            torch_dtype=query.dtype,
            torch_out_dtype=out_dtype,
            page_size=page_size,
            num_heads=num_heads,
            seq_len_q=q_len,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            is_persistent=not is_var_seq,
            is_var_seq=is_var_seq,
            is_var_split_kv=False,
        )
    except (ValueError, ImportError) as error:
        return _CuteDslImplementationUnsupportedReason(
            "cute-dsl backend (MLA decode kernel) cannot implement this "
            f"configuration: {error}"
        )
    return None


class CuteDslMlaDecodeRunner(_FunctionalMLARunner):
    """Wrapper-free functional CuTe runner mechanics for concrete bindings."""

    native_query_representation = "packed"
    native_kv_representation = "packed"

    def __init__(
        self,
        request: _FunctionalMLARequest,
        *,
        implementation_name: str,
        supports_lse: bool,
        compile_kernel: Callable[..., Tuple[Any, Any, torch.Tensor, int, int]],
        launch_kernel: Callable[
            [_CuteDslMlaExecutionState, Tuple[Any, ...], Optional[torch.Tensor]],
            None,
        ],
    ) -> None:
        _FunctionalMLARunner.__init__(self, request)
        normalized, user_lse = self._normalize_request(
            request, implementation_name=implementation_name, supports_lse=supports_lse
        )
        self.request = normalized
        self.kv_cache = normalized.kv_cache
        self.workspace_buffer = normalized.workspace_buffer
        self.kv_lora_rank = normalized.kv_lora_rank
        self.qk_nope_head_dim = normalized.qk_nope_head_dim
        self.qk_rope_head_dim = normalized.qk_rope_head_dim
        self.page_size = normalized.kv_cache.shape[-2]
        self.max_seq_len = normalized.max_seq_len
        self.softmax_scale = normalized.bmm1_scale
        self.output_scale = normalized.bmm2_scale
        self.out_dtype = normalized.out.dtype
        self.enable_pdl = normalized.enable_pdl
        self.is_var_seq = normalized.is_var_seq
        self.uses_shared_paged_kv_idx = normalized.uses_shared_paged_kv_idx
        self.lse = normalized.lse
        self.return_lse = normalized.return_lse
        self.sinks = normalized.sinks
        self.cute_dsl_impl = normalized.cute_dsl_impl
        self._implementation_name = implementation_name
        self._supports_lse = supports_lse
        self._compile_kernel = compile_kernel
        self._launch_kernel = launch_kernel
        self._user_lse = user_lse
        self._inputs = [
            normalized.query,
            normalized.block_tables,
            normalized.seq_lens,
            normalized.out,
        ]
        self._prepared_execution_state: Optional[_CuteDslMlaExecutionState] = None
        self._prepared_execution_state_lse: Optional[torch.Tensor] = None
        self._dispatch_execution_state: Optional[_CuteDslMlaExecutionState] = None
        self._dispatch_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self._dispatch_request: Optional[_FunctionalMLARequest] = None

    @staticmethod
    def _normalize_request(
        request: _FunctionalMLARequest,
        *,
        implementation_name: str,
        supports_lse: bool,
    ) -> tuple[_FunctionalMLARequest, Optional[torch.Tensor]]:
        if request.cum_seq_lens_q is not None or request.max_q_len is not None:
            raise ValueError("cute-dsl MLA does not support cum_seq_lens_q / max_q_len")
        if request.sparse_mla_top_k > 0:
            raise ValueError("cute-dsl MLA does not support sparse_mla_top_k")
        if request.skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "cute-dsl MLA does not support skip_softmax_threshold_scale_factor"
            )
        if not request.uses_shared_paged_kv_idx:
            raise ValueError("cute-dsl MLA does not support separate KV page indices")
        if request.multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        if request.seq_lens is None:
            raise ValueError("seq_lens is required for cute-dsl MLA")
        if not isinstance(request.bmm1_scale, float) or not math.isfinite(
            request.bmm1_scale
        ):
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) requires bmm1_scale to be "
                f"a finite Python float, got {request.bmm1_scale!r}"
            )
        if not isinstance(request.bmm2_scale, float) or not math.isfinite(
            request.bmm2_scale
        ):
            raise ValueError(
                "cute-dsl backend (MLA decode kernel) requires bmm2_scale to be "
                f"a finite Python float, got {request.bmm2_scale!r}"
            )

        try:
            kv_cache = _check_cute_dsl_mla_shape(
                request.query,
                request.kv_cache,
                request.kv_lora_rank,
                request.qk_rope_head_dim,
                request.sparse_mla_top_k,
                request.block_tables,
                request.kv_cache.size(-2),
                request.uses_shared_paged_kv_idx,
                require_aligned_block_table=True,
            )
        except ValueError as error:
            if request.cute_dsl_impl == "auto" and str(error).startswith(
                "Expected block_num % (128 / block_size) == 0"
            ):
                raise _FunctionalBackendUnsupportedError(str(error)) from error
            raise
        out, lse, user_lse = _prepare_cute_dsl_functional_output_and_lse(
            request.query,
            request.kv_lora_rank,
            request.out,
            request.lse,
            request.return_lse,
        )
        sink = _normalize_cute_dsl_sinks(
            request.sinks,
            num_heads=request.query.shape[-2],
            device=request.query.device,
        )
        if sink is not None and implementation_name == "monolithic":
            message = (
                "cute-dsl-monolithic does not support sinks; use cute-dsl-modular."
            )
            if request.cute_dsl_impl == "auto":
                raise _FunctionalBackendUnsupportedError(message)
            raise ValueError(message)
        if (request.return_lse or request.lse is not None) and not supports_lse:
            message = f"cute-dsl-{implementation_name} does not support LSE."
            if request.cute_dsl_impl == "auto":
                raise _FunctionalBackendUnsupportedError(message)
            raise ValueError(message)

        reason = _cute_dsl_incompatibility_reason(
            request.query,
            out.dtype,
            request.bmm1_scale,
            request.bmm2_scale,
            sink,
            request.sparse_mla_top_k,
            request.skip_softmax_threshold_scale_factor,
            request.uses_shared_paged_kv_idx,
            request.qk_rope_head_dim,
            request.kv_lora_rank,
            kv_cache.shape[-2],
            request.is_var_seq,
            request.return_lse,
            lse,
            implementation_name,
        )
        if reason is not None:
            if isinstance(reason, _CuteDslImplementationUnsupportedReason):
                if request.cute_dsl_impl == "auto":
                    raise _FunctionalBackendUnsupportedError(str(reason))
            raise ValueError(str(reason))
        enable_pdl = (
            device_support_pdl(request.query.device)
            if request.enable_pdl is None
            else request.enable_pdl
        )
        return (
            replace(
                request,
                kv_cache=kv_cache,
                out=out,
                lse=lse,
                sinks=sink,
                enable_pdl=enable_pdl,
            ),
            user_lse,
        )

    @property
    def inputs(self) -> list[torch.Tensor]:
        return self._inputs

    def _prepare_execution_state(
        self, request: _FunctionalMLARequest
    ) -> _CuteDslMlaExecutionState:
        query = request.query
        batch_size, q_len, num_heads, _ = query.shape
        cum_seq_lens_q = torch.arange(
            batch_size + 1, device=query.device, dtype=torch.int32
        ).mul_(q_len)
        assert request.seq_lens is not None
        try:
            return _prepare_cute_dsl_mla_execution_state(
                workspace_buffer=request.workspace_buffer,
                cum_seq_lens_q=cum_seq_lens_q,
                block_tables=request.block_tables,
                seq_lens=request.seq_lens,
                max_q_len=q_len,
                num_heads=num_heads,
                head_dim_ckv=request.kv_lora_rank,
                head_dim_kpe=request.qk_rope_head_dim,
                page_size=request.kv_cache.shape[-2],
                causal=False,
                sm_scale=request.bmm1_scale,
                q_data_type=query.dtype,
                kv_data_type=request.kv_cache.dtype,
                use_profiler=False,
                is_var_seq=request.is_var_seq,
                use_sinks=request.sinks is not None,
                enable_pdl=request.enable_pdl,
                compile_kernel=self._compile_kernel,
            )
        except _BackendPlanUnsupportedError as error:
            if isinstance(error.__cause__, _CuteDslKernelUnsupportedError):
                raise _FunctionalBackendUnsupportedError(str(error)) from error
            raise

    def prepare_for_dispatch(self) -> None:
        """Prepare the caller shape before admitting this auto/family candidate."""
        self._dispatch_execution_state = self._prepare_execution_state(self.request)
        self._dispatch_inputs = tuple(self._inputs)
        self._dispatch_request = self.request

    def __hash__(self):
        # Keep the legacy CuTe family cache identity stable. The requested
        # family selector remains part of get_cache_key_extras().
        return hash(CuteDslMlaDecodeRunner)

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        del profile
        from flashinfer.cute_dsl.attention.wrappers.batch_mla import (
            _get_split_kv_and_workspace_size,
        )
        from flashinfer.cute_dsl.utils import get_num_sm

        query = inputs[0]
        batch_size, q_len, num_heads, _ = query.shape
        _, workspace_size = _get_split_kv_and_workspace_size(
            batch_size,
            q_len,
            num_heads,
            self.kv_lora_rank,
            get_num_sm(query.device),
        )
        workspace_bytes = (
            self.workspace_buffer.numel() * self.workspace_buffer.element_size()
        )
        if workspace_size > workspace_bytes:
            return []
        return [-1]

    def get_cache_key_extras(self, inputs):
        query, _, _, out = inputs
        sinks_key = (
            None if self.sinks is None else (tuple(self.sinks.shape), self.sinks.dtype)
        )
        workspace_bytes = (
            self.workspace_buffer.numel() * self.workspace_buffer.element_size()
        )
        return (
            query.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.qk_nope_head_dim,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            next_positive_power_of_2(self.max_seq_len),
            workspace_bytes,
            self.is_var_seq,
            self.uses_shared_paged_kv_idx,
            self.enable_pdl,
            sinks_key,
            self.cute_dsl_impl,
        )

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ):
        del kwargs
        if tactic != -1:
            raise ValueError(f"cute-dsl MLA only supports tactic -1, got {tactic!r}.")
        if len(inputs) != 4:
            raise ValueError("cute-dsl MLA runner expects four dynamic inputs.")
        query, block_tables, seq_lens, out = inputs
        profile_measurement = is_in_profile_measurement()
        use_dispatch_state = (
            not profile_measurement
            and not do_preparation
            and self._dispatch_execution_state is not None
            and self._dispatch_inputs is not None
            and all(
                actual is prepared
                for actual, prepared in zip(inputs, self._dispatch_inputs, strict=True)
            )
        )
        if use_dispatch_state:
            assert self._dispatch_request is not None
            request = self._dispatch_request
            state = self._dispatch_execution_state
        else:
            dynamic_request = replace(
                self.request,
                query=query,
                block_tables=block_tables,
                seq_lens=seq_lens,
                out=out,
            )
        if not use_dispatch_state and profile_measurement and not do_preparation:
            if self._prepared_execution_state is None:
                raise RuntimeError(
                    "CuTe DSL autotuner launch was not prepared before profiling."
                )
            state = self._prepared_execution_state
            request = replace(dynamic_request, lse=self._prepared_execution_state_lse)
        elif not use_dispatch_state:
            dynamic_lse = (
                self.request.lse if query.shape == self.request.query.shape else None
            )
            request, _ = self._normalize_request(
                replace(dynamic_request, lse=dynamic_lse),
                implementation_name=self._implementation_name,
                supports_lse=self._supports_lse,
            )
        if not use_dispatch_state and do_preparation:
            self._prepared_execution_state = self._prepare_execution_state(request)
            self._prepared_execution_state_lse = request.lse
            state = self._prepared_execution_state
        elif not use_dispatch_state and not profile_measurement:
            state = self._prepare_execution_state(request)
        _run_cute_dsl_mla_execution_state(
            state=state,
            launch_kernel=self._launch_kernel,
            query=request.query.flatten(0, 1),
            kv_cache=request.kv_cache.squeeze(1),
            out=request.out.flatten(0, 1),
            lse=request.lse,
            return_lse=request.return_lse,
            sinks=request.sinks,
            bmm1_scale=request.bmm1_scale,
            bmm2_scale=request.bmm2_scale,
            supports_lse=self._supports_lse,
            backend_name=f"cute-dsl-{self._implementation_name}",
        )
        assert request.out is not None
        if not request.return_lse:
            return request.out
        public_lse = (
            self._user_lse
            if request.query is self.request.query and self._user_lse is not None
            else request.lse
        )
        assert public_lse is not None
        return request.out, public_lse


def _check_cute_dsl_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    sparse_mla_top_k: int,
    page_table: torch.Tensor,
    page_size: int,
    uses_shared_paged_kv_idx: bool = True,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
    require_aligned_block_table: bool = True,
) -> torch.Tensor:
    is_flattened_query = False
    if query.ndim == 4:
        num_seqs, num_tokens, _, qk_head_dim = query.shape
    elif query.ndim == 3:
        is_flattened_query = True
        if batch_size is None or max_q_len is None:
            raise ValueError(
                "batch_size and max_q_len are required when query.ndim == 3"
            )
        num_seqs = batch_size
        num_tokens = max_q_len
        _, _, qk_head_dim = query.shape
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    if kv_cache.ndim == 3:
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    is_deepseek_dimensions = kv_lora_rank == 512 and qk_rope_head_dim == 64
    is_smaller_mla_dimensions = kv_lora_rank == 256 and qk_rope_head_dim == 64
    if not (is_deepseek_dimensions or is_smaller_mla_dimensions):
        raise ValueError(
            f"Unsupported MLA dimensions, got kv_lora_rank={kv_lora_rank} and "
            f"qk_rope_head_dim={qk_rope_head_dim}, supported dimensions are: "
            "[_MLAHeadDimensions(qk_nope_head_dim=128, qk_rope_head_dim=64, "
            "v_head_dim=128, kv_lora_rank=512), _MLAHeadDimensions("
            "qk_nope_head_dim=64, qk_rope_head_dim=64, v_head_dim=128, "
            "kv_lora_rank=256)]"
        )
    ckv_dim = kv_cache.shape[3]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or ckv_dim != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, "
            f"got {qk_head_dim} and {ckv_dim}"
        )
    if sparse_mla_top_k > 0:
        page_table_shape = page_table.shape
        expected_page_table_shape = (
            (query.size(0), sparse_mla_top_k)
            if is_flattened_query
            else (num_seqs, num_tokens, sparse_mla_top_k)
        )
        if page_table_shape != expected_page_table_shape:
            raise ValueError(
                "Expected page_table.shape == "
                f"{expected_page_table_shape}, got {page_table_shape}"
            )
    else:
        _check_block_tables_shape(page_table, uses_shared_paged_kv_idx)
        batch_size_from_table = page_table.shape[0]
        block_num = page_table.shape[-1]
        if num_seqs != batch_size_from_table:
            raise ValueError(
                "Expected batch size "
                f"{num_seqs} for query and block_table, got {num_seqs} and "
                f"{batch_size_from_table}"
            )
        if require_aligned_block_table and block_num % (128 / page_size) != 0:
            raise ValueError(
                "Expected block_num % (128 / block_size) == 0, "
                f"got block_num={block_num} and block_size={page_size}"
            )
    return kv_cache


def _prepare_cute_dsl_functional_output_and_lse(
    query: torch.Tensor,
    kv_lora_rank: int,
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    expected_out_shape = query.shape[:-1] + (kv_lora_rank,)
    if out is None:
        out = torch.empty(expected_out_shape, dtype=torch.bfloat16, device=query.device)
    else:
        check_shape_dtype_device(
            out, expected_out_shape, torch.bfloat16, query.device, "out"
        )

    user_lse = lse
    if return_lse:
        flat_lse_shape = (query.size(0) * query.size(1), query.size(2))
        nested_lse_shape = (query.size(0), query.size(1), query.size(2))
        if lse is None:
            lse = torch.empty(flat_lse_shape, dtype=torch.float32, device=query.device)
            user_lse = lse
        elif tuple(lse.shape) == flat_lse_shape:
            check_shape_dtype_device(
                lse, flat_lse_shape, torch.float32, query.device, "lse"
            )
        elif tuple(lse.shape) == nested_lse_shape:
            check_shape_dtype_device(
                lse, nested_lse_shape, torch.float32, query.device, "lse"
            )
            lse = lse.view(flat_lse_shape)
        else:
            raise ValueError(
                f"lse must have shape {flat_lse_shape} or {nested_lse_shape}; "
                f"got {tuple(lse.shape)}"
            )
    return out, lse, user_lse


def _normalize_cute_dsl_sinks(
    sinks: Optional[Union[List[torch.Tensor], torch.Tensor]],
    *,
    num_heads: int,
    device: torch.device,
) -> Optional[torch.Tensor]:
    if isinstance(sinks, (list, tuple)):
        if len(sinks) != 1:
            raise ValueError(
                "cute-dsl MLA expects sinks to be a single tensor or a length-1 list/tuple"
            )
        normalized_sinks: Optional[torch.Tensor] = sinks[0]
    else:
        normalized_sinks = sinks
    if normalized_sinks is None:
        return None
    if normalized_sinks.ndim != 1 or normalized_sinks.shape[0] != num_heads:
        raise ValueError(
            f"sinks tensor must have shape (num_qo_heads,) = ({num_heads},), "
            f"got shape {tuple(normalized_sinks.shape)}"
        )
    if not normalized_sinks.is_contiguous():
        raise ValueError(
            "sinks tensor must be contiguous, got strides "
            f"{normalized_sinks.stride()} for shape {normalized_sinks.shape}"
        )
    return normalized_sinks.to(dtype=torch.float32, device=device)
