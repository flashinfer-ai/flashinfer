"""XQA validation and concrete launch assembly for MLA decode."""

import math
from typing import Optional, Tuple, Union, cast

import torch

from flashinfer.utils import (
    _check_block_tables_shape,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    is_sm12x_supported,
)
from flashinfer.xqa import get_xqa_module_mla

from .._planning import (
    _audit_plan_from_wrapper_arguments,
    _MLAPlanArguments,
    _MLAWrapperPlanResult,
)
from ._layout import _concat_adjacent_views_or_cat


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

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer

    @classmethod
    @_audit_plan_from_wrapper_arguments
    def plan_from_wrapper(cls, args: _MLAPlanArguments) -> _MLAWrapperPlanResult:
        if args.use_profiler:
            raise ValueError(
                "use_profiler is not supported by the XQA wrapper backend."
            )
        if args.causal:
            raise ValueError("causal=True is not supported by the XQA wrapper backend.")
        if args.qk_nope_head_dim is not None:
            raise ValueError(
                "qk_nope_head_dim is not supported by the XQA wrapper backend."
            )
        if args.is_var_seq is not None:
            raise ValueError("is_var_seq is not supported by the XQA wrapper backend.")
        if args.cute_dsl_impl != "auto":
            raise ValueError(
                "cute_dsl_impl is not supported by the XQA wrapper backend."
            )
        if args.use_sinks:
            raise ValueError("use_sinks is not supported by the XQA wrapper backend.")
        if (
            not isinstance(args.page_size, int)
            or isinstance(args.page_size, bool)
            or args.page_size <= 0
        ):
            raise ValueError(
                f"page_size must be a positive int, got {args.page_size!r}."
            )
        if args.page_size > 128 or 128 % args.page_size != 0:
            raise ValueError(
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
            enable_pdl=args.enable_pdl,
        )
        return _MLAWrapperPlanResult(backend_impl=backend)

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
            raise ValueError(
                "XQA MLA wrapper requires SM120a (CUDA >= 12.8) or "
                "SM121a (CUDA >= 12.9)"
            )
        if use_profiler:
            raise ValueError(
                "use_profiler is not supported by the XQA wrapper backend."
            )
        if causal:
            raise ValueError("causal=True is not supported by the XQA wrapper backend.")
        if num_heads != 128:
            raise ValueError(f"XQA MLA only supports 128 query heads, got {num_heads}.")
        if (head_dim_ckv, head_dim_kpe) not in _SUPPORTED_MLA_DIMENSIONS:
            raise ValueError(
                "Unsupported MLA dimensions for XQA wrapper, got "
                f"head_dim_ckv={head_dim_ckv} and head_dim_kpe={head_dim_kpe}; "
                f"supported dimensions are {_SUPPORTED_MLA_DIMENSIONS}."
            )
        if page_size not in _SUPPORTED_XQA_PAGE_SIZES:
            raise ValueError(
                "XQA MLA page_size must be one of "
                f"{_SUPPORTED_XQA_PAGE_SIZES}, got {page_size}."
            )
        if q_data_type != kv_data_type:
            raise ValueError(
                "XQA MLA query and KV cache must use the same dtype, got "
                f"{q_data_type} and {kv_data_type}."
            )
        if q_data_type not in (torch.bfloat16, torch.float8_e4m3fn):
            raise ValueError(
                "XQA MLA wrapper supports BF16 or FP8 E4M3 inputs only, "
                f"got {q_data_type}."
            )
        if type(sm_scale) is not float or not math.isfinite(sm_scale):
            raise TypeError(
                "XQA MLA wrapper expects sm_scale to be a finite Python float, "
                f"got {sm_scale!r}."
            )
        if max_q_len != 1:
            raise ValueError(
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
            if int(q_offsets[0].item()) != 0 or torch.any(q_lens != 1).item():
                raise ValueError(
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
            raise ValueError(
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
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
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
            q_nope=q_nope,
            q_pe=q_pe,
            ckv_cache=ckv_cache,
            kpe_cache=kpe_cache,
            out=out,
            lse=None,
            return_lse=False,
            bmm1_scale=cast(Optional[float], bmm1_scale),
            bmm2_scale=cast(Optional[float], bmm2_scale),
        )

    def run(
        self,
        *,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        ckv_cache: torch.Tensor,
        kpe_cache: torch.Tensor,
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
            q_nope,
            (self._batch_size, self._num_heads, self._kv_lora_rank),
            self._q_dtype,
            self.device,
            "q_nope",
        )
        check_shape_dtype_device(
            q_pe,
            (self._batch_size, self._num_heads, self._qk_rope_head_dim),
            self._q_dtype,
            self.device,
            "q_pe",
        )
        check_shape_dtype_device(
            ckv_cache,
            (ckv_cache.shape[0], self._page_size, self._kv_lora_rank),
            self._kv_dtype,
            self.device,
            "ckv_cache",
        )
        check_shape_dtype_device(
            kpe_cache,
            (ckv_cache.shape[0], self._page_size, self._qk_rope_head_dim),
            self._kv_dtype,
            self.device,
            "kpe_cache",
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

        query = _concat_adjacent_views_or_cat(q_nope, q_pe).reshape(
            self._batch_size,
            1,
            self._num_heads,
            self._kv_lora_rank + self._qk_rope_head_dim,
        )
        kv_cache = _concat_adjacent_views_or_cat(ckv_cache, kpe_cache).unsqueeze(2)
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
