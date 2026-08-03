"""XQA validation and concrete launch assembly for MLA decode."""

import math
from dataclasses import dataclass, replace
from typing import Any, List, Optional, Tuple, Union, cast

import torch

from flashinfer._backend import _BackendPlanUnsupportedError
from flashinfer.autotuner import is_in_profile_measurement
from flashinfer.utils import (
    _check_block_tables_shape,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    is_sm12x_supported,
    next_positive_power_of_2,
)
from flashinfer.xqa import get_xqa_module_mla

from ._capabilities import (
    BACKEND_OPERATIONAL_PLAN_FIELDS,
    MLAPlanCapabilities,
    validate_plan_capabilities,
)
from .._planning import (
    _MLAPlanArguments,
)
from .._contracts import (
    _FunctionalMLARequest,
    _FunctionalMLARunner,
    _concat_adjacent_views_or_cat,
    MLAKVCache,
    MLAQuery,
)


_SUPPORTED_MLA_DIMENSIONS = ((512, 64), (256, 64))
_SUPPORTED_XQA_PAGE_SIZES = (16, 32, 64, 128)
_XQA_MIN_WORKSPACE_BYTES = 128 * 1024 * 1024
_XQA_SEMAPHORE_BYTES = 8 * 1024 * 1024


def _is_xqa_wrapper_arch_supported(device: torch.device) -> bool:
    if not is_sm12x_supported(device):
        return False
    return get_compute_capability(device) in ((12, 0), (12, 1))


def _validate_xqa_mla_scales(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    bmm1_scale: Union[float, torch.Tensor],
    bmm2_scale: Union[float, torch.Tensor],
) -> None:
    """Validate the paired scalar-or-device-scalar contract of XQA MLA."""
    scales = (
        ("bmm1_scale", bmm1_scale),
        ("bmm2_scale", bmm2_scale),
    )
    modes = []
    for name, scale in scales:
        if isinstance(scale, torch.Tensor):
            modes.append("tensor")
        elif isinstance(scale, float):
            modes.append("float")
        else:
            raise TypeError(f"{name} must be a float or torch.Tensor")

    if modes[0] != modes[1]:
        raise TypeError(
            "bmm1_scale and bmm2_scale must use the same mode "
            "(both float or both tensor)"
        )
    if modes[0] == "float":
        return

    if not (
        query.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.float8_e4m3fn
    ):
        raise ValueError("XQA MLA tensor scale mode is supported for FP8 inputs only")

    tensor_scales = cast(Tuple[Tuple[str, torch.Tensor], ...], scales)
    for name, scale in tensor_scales:
        if scale.dtype != torch.float32:
            raise TypeError(f"{name} tensor must have dtype torch.float32")
        if scale.numel() != 1:
            raise ValueError(
                f"{name} must be a single-element tensor, got shape {tuple(scale.shape)}"
            )
        if scale.device != query.device:
            raise ValueError(
                f"{name} must be on the same device as query, "
                f"got {scale.device} and {query.device}"
            )


class _BatchMLAPagedAttentionXqaBackend:
    """Planned XQA MLA execution with a launch-only hot path."""

    _plan_capabilities = MLAPlanCapabilities(
        backend_name="XQA",
        lse_modes=frozenset({"none"}),
        kv_layouts=frozenset({"combined", "adjacent-split"}),
        output_scales=frozenset({"none"}),
        scale_modes=frozenset({"default", "bmm-scalar"}),
        supports_enable_pdl=True,
    )
    _backend_operational_plan_fields = BACKEND_OPERATIONAL_PLAN_FIELDS

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer

    @classmethod
    def plan_from_wrapper(
        cls, args: _MLAPlanArguments
    ) -> "_BatchMLAPagedAttentionXqaBackend":
        validate_plan_capabilities(args, cls._plan_capabilities)
        output_dtype = args.output_dtype
        if output_dtype != torch.bfloat16:
            raise _BackendPlanUnsupportedError(
                "XQA backend requires a bfloat16 output contract without o_scale."
            )
        enable_pdl = args.enable_pdl
        if args.use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the XQA wrapper backend."
            )
        if args.causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the XQA wrapper backend."
            )
        if (
            not isinstance(args.page_size, int)
            or isinstance(args.page_size, bool)
            or args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {args.page_size!r}."
            )
        if args.page_size > 128 or 128 % args.page_size != 0:
            raise _BackendPlanUnsupportedError(
                "xqa dense metadata requires page_size to divide 128, "
                f"got {args.page_size}."
            )
        dense = args.dense(
            table_width_alignment=128 // args.page_size,
        )
        backend = cls(args._float_workspace_buffer)
        backend.plan(
            cum_seq_lens_q=dense.cum_seq_lens_q,
            block_tables=dense.block_tables,
            seq_lens=dense.seq_lens,
            max_q_len=dense.max_q_len,
            num_heads=args.num_heads,
            head_dim_ckv=args.head_dim_ckv,
            head_dim_kpe=args.head_dim_kpe,
            page_size=args.page_size,
            causal=args.causal,
            sm_scale=args.sm_scale,
            q_data_type=args.q_data_type,
            kv_data_type=args.kv_data_type,
            use_profiler=args.use_profiler,
            enable_pdl=enable_pdl,
        )
        return backend

    def plan(
        self,
        *,
        cum_seq_lens_q: Optional[torch.Tensor],
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_q_len: int,
        num_heads: int,
        head_dim_ckv: int,
        head_dim_kpe: int,
        page_size: int,
        causal: bool,
        sm_scale: float,
        q_data_type: torch.dtype,
        kv_data_type: torch.dtype,
        use_profiler: bool,
        enable_pdl: Optional[bool],
        initialize_semaphore: bool = True,
    ) -> None:
        if not _is_xqa_wrapper_arch_supported(self.device):
            raise _BackendPlanUnsupportedError(
                "XQA MLA wrapper requires SM120a (CUDA >= 12.8) or "
                "SM121a (CUDA >= 12.9)"
            )
        if use_profiler:
            raise _BackendPlanUnsupportedError(
                "use_profiler is not supported by the XQA wrapper backend."
            )
        if causal:
            raise _BackendPlanUnsupportedError(
                "causal=True is not supported by the XQA wrapper backend."
            )
        if num_heads != 128:
            raise _BackendPlanUnsupportedError(
                f"XQA MLA only supports 128 query heads, got {num_heads}."
            )
        if (head_dim_ckv, head_dim_kpe) not in _SUPPORTED_MLA_DIMENSIONS:
            raise _BackendPlanUnsupportedError(
                "Unsupported MLA dimensions for XQA wrapper, got "
                f"head_dim_ckv={head_dim_ckv} and head_dim_kpe={head_dim_kpe}; "
                f"supported dimensions are {_SUPPORTED_MLA_DIMENSIONS}."
            )
        if page_size not in _SUPPORTED_XQA_PAGE_SIZES:
            raise _BackendPlanUnsupportedError(
                "XQA MLA page_size must be one of "
                f"{_SUPPORTED_XQA_PAGE_SIZES}, got {page_size}."
            )
        if q_data_type != kv_data_type:
            raise _BackendPlanUnsupportedError(
                "XQA MLA query and KV cache must use the same dtype, got "
                f"{q_data_type} and {kv_data_type}."
            )
        if q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
            raise _BackendPlanUnsupportedError(
                "XQA MLA wrapper supports BF16 or FP8 E4M3 inputs only, "
                f"got {q_data_type}."
            )
        if type(sm_scale) is not float or not math.isfinite(sm_scale):
            raise TypeError(
                "XQA MLA wrapper expects sm_scale to be a finite Python float, "
                f"got {sm_scale!r}."
            )
        if max_q_len != 1:
            raise _BackendPlanUnsupportedError(
                f"XQA MLA wrapper requires max_q_len/query length == 1, got {max_q_len}."
            )

        if cum_seq_lens_q is None:
            batch_size = block_tables.shape[0]
        else:
            check_shape_dtype_device(
                cum_seq_lens_q,
                None,
                torch.int32,
                self.device,
                "cum_seq_lens_q",
            )
            if cum_seq_lens_q.ndim != 1 or cum_seq_lens_q.numel() < 2:
                raise ValueError(
                    "XQA MLA wrapper expects one-dimensional cum_seq_lens_q with "
                    "at least two entries."
                )
            if not cum_seq_lens_q.is_contiguous():
                raise ValueError(
                    "cum_seq_lens_q must be contiguous for XQA MLA wrapper."
                )
            q_offsets = cum_seq_lens_q.to(device="cpu", dtype=torch.int64)
            q_lens = q_offsets[1:] - q_offsets[:-1]
            if int(q_offsets[0].item()) != 0:
                raise ValueError("cum_seq_lens_q must start at zero.")
            if torch.any(q_lens < 0).item():
                raise ValueError("cum_seq_lens_q must be nondecreasing.")
            if torch.any(q_lens != 1).item():
                raise _BackendPlanUnsupportedError(
                    "XQA MLA wrapper requires exactly one query token per request."
                )
            batch_size = cum_seq_lens_q.numel() - 1

        check_shape_dtype_device(
            block_tables,
            None,
            torch.int32,
            self.device,
            "block_tables",
        )
        _check_block_tables_shape(block_tables, True)
        if block_tables.shape[0] != batch_size:
            raise ValueError(
                "XQA MLA block_tables batch dimension must match "
                f"cum_seq_lens_q, got {block_tables.shape[0]} and {batch_size}."
            )
        if not block_tables.is_contiguous():
            raise ValueError("block_tables must be contiguous for XQA MLA wrapper.")
        alignment = 128 // page_size
        if block_tables.shape[1] == 0 or block_tables.shape[1] % alignment != 0:
            raise ValueError(
                "XQA MLA block_tables width must be a positive multiple of "
                f"{alignment} for page_size={page_size}."
            )
        check_shape_dtype_device(
            seq_lens,
            (batch_size,),
            torch.int32,
            self.device,
            "seq_lens",
        )
        if not seq_lens.is_contiguous():
            raise ValueError("seq_lens must be contiguous for XQA MLA wrapper.")
        max_seq_len = block_tables.shape[1] * page_size
        if cum_seq_lens_q is not None:
            seq_lens_host = seq_lens.to(device="cpu", dtype=torch.int64)
            if torch.any(seq_lens_host < 0).item():
                raise ValueError("seq_lens must be nonnegative for XQA MLA wrapper.")
            if torch.any(seq_lens_host > max_seq_len).item():
                raise ValueError(
                    f"seq_lens cannot exceed the XQA block-table capacity {max_seq_len}."
                )

        resolved_enable_pdl = (
            device_support_pdl(self.device) if enable_pdl is None else enable_pdl
        )
        if type(resolved_enable_pdl) is not bool:
            raise TypeError(
                "XQA MLA wrapper expects enable_pdl to be bool or None, got "
                f"{enable_pdl!r}."
            )
        workspace_u8 = self._float_workspace_buffer.view(torch.uint8).flatten()
        if initialize_semaphore and not self._float_workspace_buffer.is_contiguous():
            raise ValueError("workspace buffer must be contiguous for XQA MLA wrapper.")
        if initialize_semaphore and workspace_u8.numel() < _XQA_MIN_WORKSPACE_BYTES:
            raise _BackendPlanUnsupportedError(
                "XQA MLA wrapper workspace must contain at least 128 MiB, got "
                f"{workspace_u8.numel()} bytes."
            )
        sm_count = get_device_sm_count(self.device)
        module = get_xqa_module_mla(
            q_data_type,
            kv_data_type,
            page_size,
            head_dim_ckv + head_dim_kpe,
            num_heads,
            False,
        )
        semaphore = workspace_u8[:_XQA_SEMAPHORE_BYTES]
        scratch = workspace_u8[_XQA_SEMAPHORE_BYTES:]
        if initialize_semaphore:
            semaphore.zero_()

        self._module = module
        self._cum_seq_lens_q = cum_seq_lens_q
        self._block_tables = block_tables
        self._seq_lens = seq_lens
        self._seq_lens_2d = seq_lens.unsqueeze(1)
        self._batch_size = batch_size
        self._num_heads = num_heads
        self._kv_lora_rank = head_dim_ckv
        self._qk_rope_head_dim = head_dim_kpe
        self._page_size = page_size
        self._q_dtype = q_data_type
        self._kv_dtype = kv_data_type
        self._bmm1_scale = sm_scale
        self._bmm2_scale = 1.0
        self._enable_pdl = resolved_enable_pdl
        self._sm_count = sm_count
        self._max_seq_len = max_seq_len
        self._semaphore = semaphore
        self._scratch = scratch

    def run_from_wrapper(
        self,
        *,
        query: MLAQuery,
        kv: MLAKVCache,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        return_lse: bool,
        profiler_buffer: Optional[torch.Tensor],
        kv_len: Optional[torch.Tensor],
        page_table: Optional[torch.Tensor],
        return_lse_base_on_e: bool,
        o_scale: Optional[float],
        ckv_scale: Optional[float],
        kpe_scale: Optional[float],
        sinks: Optional[torch.Tensor],
        skip_softmax_threshold_scale_factor: Optional[float],
        bmm1_scale: Optional[Union[float, torch.Tensor]],
        bmm2_scale: Optional[Union[float, torch.Tensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        packed_query = query.require_packed()
        kv_cache = kv.require_packed_view()
        if return_lse or lse is not None:
            raise ValueError("XQA MLA wrapper does not support LSE output.")
        if profiler_buffer is not None:
            raise ValueError("profiler_buffer is not supported with XQA backend.")
        if kv_len is not None or page_table is not None:
            raise ValueError(
                "kv_len and page_table are not supported with XQA backend."
            )
        if return_lse_base_on_e:
            raise ValueError("return_lse_base_on_e is not supported with XQA backend.")
        if o_scale is not None:
            raise ValueError("o_scale is not supported with XQA backend.")
        if ckv_scale is not None or kpe_scale is not None:
            raise ValueError(
                "ckv_scale / kpe_scale are not supported with XQA backend."
            )
        if sinks is not None:
            raise ValueError("sinks are not supported with XQA backend.")
        if skip_softmax_threshold_scale_factor is not None:
            raise ValueError(
                "skip_softmax_threshold_scale_factor is not supported with XQA backend."
            )
        for name, scale in (("bmm1_scale", bmm1_scale), ("bmm2_scale", bmm2_scale)):
            if isinstance(scale, torch.Tensor):
                raise ValueError(
                    f"XQA MLA wrapper accepts {name} as a float only; "
                    "tensor scales are not supported."
                )
            if scale is not None and (
                type(scale) is not float or not math.isfinite(scale)
            ):
                raise ValueError(
                    f"XQA MLA wrapper expects {name} to be a finite Python float, "
                    f"got {scale!r}."
                )
        return self.run(
            query=packed_query,
            kv_cache=kv_cache,
            out=out,
            lse=None,
            return_lse=False,
            bmm1_scale=cast(Optional[float], bmm1_scale),
            bmm2_scale=cast(Optional[float], bmm2_scale),
        )

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        bmm1_scale: Optional[Union[float, torch.Tensor]] = None,
        bmm2_scale: Optional[Union[float, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionXqaBackend.run() called before plan()."
            )
        if return_lse or lse is not None:
            raise ValueError("XQA MLA wrapper does not support LSE output.")
        check_shape_dtype_device(
            query,
            (
                self._batch_size,
                self._num_heads,
                self._kv_lora_rank + self._qk_rope_head_dim,
            ),
            self._q_dtype,
            self.device,
            "query",
        )
        check_shape_dtype_device(
            kv_cache,
            (
                kv_cache.shape[0],
                self._page_size,
                self._kv_lora_rank + self._qk_rope_head_dim,
            ),
            self._kv_dtype,
            self.device,
            "kv_cache",
        )
        if out is None:
            out = torch.empty(
                (self._batch_size, self._num_heads, self._kv_lora_rank),
                dtype=torch.bfloat16,
                device=self.device,
            )
        else:
            check_shape_dtype_device(
                out,
                (self._batch_size, self._num_heads, self._kv_lora_rank),
                torch.bfloat16,
                self.device,
                "out",
            )
            if not out.is_contiguous():
                raise ValueError("out must be contiguous for XQA MLA wrapper.")

        query = query.reshape(
            self._batch_size,
            1,
            self._num_heads,
            self._kv_lora_rank + self._qk_rope_head_dim,
        )
        kv_cache = kv_cache.unsqueeze(2)
        resolved_bmm1_scale = self._bmm1_scale if bmm1_scale is None else bmm1_scale
        resolved_bmm2_scale = self._bmm2_scale if bmm2_scale is None else bmm2_scale
        _validate_xqa_mla_scales(
            query,
            kv_cache,
            bmm1_scale=resolved_bmm1_scale,
            bmm2_scale=resolved_bmm2_scale,
        )
        self._module.xqa_mla(
            self._sm_count,
            resolved_bmm1_scale,
            out,
            query,
            kv_cache,
            kv_cache,
            self._block_tables,
            self._max_seq_len,
            self._seq_lens_2d,
            self._batch_size,
            resolved_bmm2_scale,
            self._semaphore,
            self._scratch,
            self._enable_pdl,
        )
        return out


@dataclass
class _XqaMlaFunctionalState:
    """Concrete XQA launch state owned by the functional runner."""

    module: Any
    block_tables: torch.Tensor
    seq_lens_2d: torch.Tensor
    batch_size: int
    num_heads: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    page_size: int
    q_dtype: torch.dtype
    kv_dtype: torch.dtype
    bmm1_scale: float
    bmm2_scale: float
    enable_pdl: bool
    sm_count: int
    max_seq_len: int
    semaphore: torch.Tensor
    scratch: torch.Tensor


def _prepare_xqa_mla_functional_state(
    *,
    request: _FunctionalMLARequest,
    initialize_semaphore: bool,
) -> _XqaMlaFunctionalState:
    """Validate and reserve the launch state without entering wrapper lifecycle."""
    query = request.query
    seq_lens = request.seq_lens
    assert seq_lens is not None
    workspace_buffer = request.workspace_buffer
    device = workspace_buffer.device
    page_size = request.kv_cache.shape[-2]
    num_heads = query.size(2)
    max_q_len = query.size(1)
    sm_scale = request.bmm1_scale if isinstance(request.bmm1_scale, float) else 1.0

    if not _is_xqa_wrapper_arch_supported(device):
        raise _BackendPlanUnsupportedError(
            "XQA MLA wrapper requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 12.9)"
        )
    if num_heads != 128:
        raise _BackendPlanUnsupportedError(
            f"XQA MLA only supports 128 query heads, got {num_heads}."
        )
    if (
        request.kv_lora_rank,
        request.qk_rope_head_dim,
    ) not in _SUPPORTED_MLA_DIMENSIONS:
        raise _BackendPlanUnsupportedError(
            "Unsupported MLA dimensions for XQA wrapper, got "
            f"head_dim_ckv={request.kv_lora_rank} and "
            f"head_dim_kpe={request.qk_rope_head_dim}; "
            f"supported dimensions are {_SUPPORTED_MLA_DIMENSIONS}."
        )
    if page_size not in _SUPPORTED_XQA_PAGE_SIZES:
        raise _BackendPlanUnsupportedError(
            "XQA MLA page_size must be one of "
            f"{_SUPPORTED_XQA_PAGE_SIZES}, got {page_size}."
        )
    if query.dtype != request.kv_cache.dtype:
        raise _BackendPlanUnsupportedError(
            "XQA MLA query and KV cache must use the same dtype, got "
            f"{query.dtype} and {request.kv_cache.dtype}."
        )
    if query.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise _BackendPlanUnsupportedError(
            f"XQA MLA wrapper supports BF16 or FP8 E4M3 inputs only, got {query.dtype}."
        )
    if type(sm_scale) is not float or not math.isfinite(sm_scale):
        raise TypeError(
            "XQA MLA wrapper expects sm_scale to be a finite Python float, "
            f"got {sm_scale!r}."
        )
    if max_q_len != 1:
        raise _BackendPlanUnsupportedError(
            f"XQA MLA wrapper requires max_q_len/query length == 1, got {max_q_len}."
        )

    batch_size = request.block_tables.shape[0]
    check_shape_dtype_device(
        request.block_tables,
        None,
        torch.int32,
        device,
        "block_tables",
    )
    _check_block_tables_shape(request.block_tables, True)
    if not request.block_tables.is_contiguous():
        raise ValueError("block_tables must be contiguous for XQA MLA wrapper.")
    alignment = 128 // page_size
    if (
        request.block_tables.shape[1] == 0
        or request.block_tables.shape[1] % alignment != 0
    ):
        raise ValueError(
            "XQA MLA block_tables width must be a positive multiple of "
            f"{alignment} for page_size={page_size}."
        )
    check_shape_dtype_device(
        seq_lens,
        (batch_size,),
        torch.int32,
        device,
        "seq_lens",
    )
    if not seq_lens.is_contiguous():
        raise ValueError("seq_lens must be contiguous for XQA MLA wrapper.")

    resolved_enable_pdl = (
        device_support_pdl(device) if request.enable_pdl is None else request.enable_pdl
    )
    if type(resolved_enable_pdl) is not bool:
        raise TypeError(
            "XQA MLA wrapper expects enable_pdl to be bool or None, got "
            f"{request.enable_pdl!r}."
        )
    workspace_u8 = workspace_buffer.view(torch.uint8).flatten()
    if initialize_semaphore and not workspace_buffer.is_contiguous():
        raise ValueError("workspace buffer must be contiguous for XQA MLA wrapper.")
    if initialize_semaphore and workspace_u8.numel() < _XQA_MIN_WORKSPACE_BYTES:
        raise _BackendPlanUnsupportedError(
            "XQA MLA wrapper workspace must contain at least 128 MiB, got "
            f"{workspace_u8.numel()} bytes."
        )
    module = get_xqa_module_mla(
        query.dtype,
        request.kv_cache.dtype,
        page_size,
        request.kv_lora_rank + request.qk_rope_head_dim,
        num_heads,
        False,
    )
    semaphore = workspace_u8[:_XQA_SEMAPHORE_BYTES]
    scratch = workspace_u8[_XQA_SEMAPHORE_BYTES:]
    if initialize_semaphore:
        semaphore.zero_()

    return _XqaMlaFunctionalState(
        module=module,
        block_tables=request.block_tables,
        seq_lens_2d=seq_lens.unsqueeze(1),
        batch_size=batch_size,
        num_heads=num_heads,
        kv_lora_rank=request.kv_lora_rank,
        qk_rope_head_dim=request.qk_rope_head_dim,
        page_size=page_size,
        q_dtype=query.dtype,
        kv_dtype=request.kv_cache.dtype,
        bmm1_scale=sm_scale,
        bmm2_scale=1.0,
        enable_pdl=resolved_enable_pdl,
        sm_count=get_device_sm_count(device),
        max_seq_len=request.block_tables.shape[1] * page_size,
        semaphore=semaphore,
        scratch=scratch,
    )


class XqaMlaDecodeRunner(_FunctionalMLARunner):
    """Direct functional XQA runner with the single current tactic."""

    name = "xqa"

    def __init__(self, request: _FunctionalMLARequest) -> None:
        _FunctionalMLARunner.__init__(self, request)
        self.request = self._normalize_request(request)
        self.kv_cache = self.request.kv_cache
        self.workspace_buffer = self.request.workspace_buffer
        self.kv_lora_rank = self.request.kv_lora_rank
        self.qk_rope_head_dim = self.request.qk_rope_head_dim
        self.page_size = self.kv_cache.shape[-2]
        self.max_seq_len = self.request.max_seq_len
        self.bmm1_scale = self.request.bmm1_scale
        self.bmm2_scale = self.request.bmm2_scale
        self.enable_pdl = self.request.enable_pdl
        self._inputs = [
            self.request.query,
            self.request.block_tables,
            self.request.seq_lens,
            self.request.out,
        ]
        self._prepared_functional_state: Optional[_XqaMlaFunctionalState] = None

    @staticmethod
    def _normalize_request(request: _FunctionalMLARequest) -> _FunctionalMLARequest:
        if request.seq_lens is None:
            raise ValueError("seq_lens is required for XQA MLA")
        if request.sparse_mla_top_k > 0:
            raise ValueError("XQA MLA does not support sparse_mla_top_k")
        if request.cum_seq_lens_q is not None or request.max_q_len is not None:
            raise ValueError("XQA MLA does not support cum_seq_lens_q / max_q_len")
        if request.multi_ctas_kv_counter_buffer is not None:
            raise ValueError(
                "multi_ctas_kv_counter_buffer is only supported by the trtllm-gen backend"
            )
        if request.skip_softmax_threshold_scale_factor is not None:
            raise ValueError("skip_softmax is not supported for XQA backend")
        if not request.uses_shared_paged_kv_idx:
            raise ValueError(
                "XQA MLA does not support separate KV page indices "
                "(uses_shared_paged_kv_idx=False)"
            )
        if request.sinks is not None:
            raise ValueError("XQA MLA does not support sinks")
        if request.lse is not None or request.return_lse:
            raise ValueError("XQA MLA does not support LSE output.")

        kv_cache = request.kv_cache
        if kv_cache.ndim == 4:
            if kv_cache.size(1) != 1:
                raise ValueError(
                    "XQA MLA expects a single KV cache head, "
                    f"got kv_cache.shape[1] == {kv_cache.size(1)}"
                )
            kv_cache = kv_cache.squeeze(1)
        elif kv_cache.ndim != 3:
            raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")
        return replace(request, kv_cache=kv_cache)

    @property
    def inputs(self) -> list[torch.Tensor]:
        return self._inputs

    def __hash__(self) -> int:
        return hash(type(self))

    def get_valid_tactics(self, inputs, profile) -> List[int]:
        del inputs, profile
        return [-1]

    def get_cache_key_extras(self, inputs):
        query, _, _, out = inputs
        return (
            query.dtype,
            self.kv_cache.dtype,
            out.dtype,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            self.page_size,
            next_positive_power_of_2(self.max_seq_len),
            self.enable_pdl,
            "bmm1_tensor"
            if isinstance(self.bmm1_scale, torch.Tensor)
            else "bmm1_float",
            "bmm2_tensor"
            if isinstance(self.bmm2_scale, torch.Tensor)
            else "bmm2_float",
        )

    def _request_from_inputs(self, inputs) -> _FunctionalMLARequest:
        query, block_tables, seq_lens, out = inputs
        return replace(
            self.request,
            query=query,
            block_tables=block_tables,
            seq_lens=seq_lens,
            out=out,
        )

    def _prepare_output(
        self,
        *,
        query: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        native_shape = (query.size(0), query.size(2), self.kv_lora_rank)
        legacy_shape = (query.size(0), 1, query.size(2), self.kv_lora_rank)
        if out is None:
            native_out = torch.empty(
                native_shape, dtype=torch.bfloat16, device=query.device
            )
            return native_out, native_out.unsqueeze(1)
        if tuple(out.shape) == native_shape:
            check_shape_dtype_device(
                out, native_shape, torch.bfloat16, query.device, "out"
            )
            return out, out
        check_shape_dtype_device(out, legacy_shape, torch.bfloat16, query.device, "out")
        return out.squeeze(1), out

    def _launch_functional_state(
        self,
        *,
        request: _FunctionalMLARequest,
        state: _XqaMlaFunctionalState,
    ) -> torch.Tensor:
        native_out, result = self._prepare_output(
            query=request.query,
            out=request.out,
        )
        q_nope = request.query[..., : self.kv_lora_rank].squeeze(1)
        q_pe = request.query[..., self.kv_lora_rank :].squeeze(1)
        check_shape_dtype_device(
            q_nope,
            (state.batch_size, state.num_heads, state.kv_lora_rank),
            state.q_dtype,
            request.workspace_buffer.device,
            "q_nope",
        )
        check_shape_dtype_device(
            q_pe,
            (state.batch_size, state.num_heads, state.qk_rope_head_dim),
            state.q_dtype,
            request.workspace_buffer.device,
            "q_pe",
        )
        check_shape_dtype_device(
            request.kv_cache,
            (
                request.kv_cache.shape[0],
                state.page_size,
                state.kv_lora_rank + state.qk_rope_head_dim,
            ),
            state.kv_dtype,
            request.workspace_buffer.device,
            "kv_cache",
        )
        if not native_out.is_contiguous():
            raise ValueError("out must be contiguous for XQA MLA wrapper.")

        query = _concat_adjacent_views_or_cat(q_nope, q_pe).reshape(
            state.batch_size,
            1,
            state.num_heads,
            state.kv_lora_rank + state.qk_rope_head_dim,
        )
        kv_cache = request.kv_cache.unsqueeze(2)
        _validate_xqa_mla_scales(
            query,
            kv_cache,
            bmm1_scale=request.bmm1_scale,
            bmm2_scale=request.bmm2_scale,
        )
        state.module.xqa_mla(
            state.sm_count,
            request.bmm1_scale,
            native_out,
            query,
            kv_cache,
            kv_cache,
            state.block_tables,
            state.max_seq_len,
            state.seq_lens_2d,
            state.batch_size,
            request.bmm2_scale,
            state.semaphore,
            state.scratch,
            state.enable_pdl,
        )
        return result

    def forward(
        self,
        inputs,
        tactic: int = -1,
        do_preparation: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        del tactic, kwargs
        request = self._request_from_inputs(inputs)
        if do_preparation:
            self._prepared_functional_state = _prepare_xqa_mla_functional_state(
                request=request,
                initialize_semaphore=False,
            )
            state = self._prepared_functional_state
        elif is_in_profile_measurement():
            if self._prepared_functional_state is None:
                raise RuntimeError(
                    "XQA autotuner launch was not prepared before profiling."
                )
            state = self._prepared_functional_state
        else:
            state = _prepare_xqa_mla_functional_state(
                request=request,
                initialize_semaphore=True,
            )
        return self._launch_functional_state(request=request, state=state)
