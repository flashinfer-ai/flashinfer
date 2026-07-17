"""XQA validation and concrete launch assembly for MLA decode."""

import math
from typing import List, Optional, Union

import torch

from flashinfer.utils import (
    _check_block_tables_shape,
    check_shape_dtype_device,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    is_sm12x_supported,
)
from flashinfer.xqa import get_xqa_module_mla, xqa_mla
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

    for name, scale in scales:
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


def _check_xqa_mla_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    *,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    page_size: int,
) -> torch.Tensor:
    """Validate XQA's combined query/cache and shared page-table layout."""
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
    batch_size, _, _, qk_head_dim = query.shape

    if kv_cache.ndim == 3:
        kv_cache = kv_cache.unsqueeze(1)
    elif kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")

    if (kv_lora_rank, qk_rope_head_dim) not in _SUPPORTED_MLA_DIMENSIONS:
        raise ValueError(
            "Unsupported MLA dimensions, got "
            f"kv_lora_rank={kv_lora_rank} and "
            f"qk_rope_head_dim={qk_rope_head_dim}, supported dimensions are: "
            f"{_SUPPORTED_MLA_DIMENSIONS}"
        )

    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or kv_cache.shape[3] != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, "
            f"got {qk_head_dim} and {kv_cache.shape[3]}"
        )

    _check_block_tables_shape(block_tables, True)
    if block_tables.shape[0] != batch_size:
        raise ValueError(
            f"Expected batch size {batch_size} for query and block_table, "
            f"got {batch_size} and {block_tables.shape[0]}"
        )
    block_num = block_tables.shape[-1]
    if block_num % (128 / page_size) != 0:
        raise ValueError(
            "Expected block_num % (128 / block_size) == 0, "
            f"got block_num={block_num} and block_size={page_size}"
        )
    return kv_cache


class _XqaMlaDecodeImplementation:
    """Own XQA-specific validation, preparation, and kernel launch assembly."""

    def run(
        self,
        *,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        workspace_buffer: torch.Tensor,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: Optional[torch.Tensor],
        bmm1_scale: Union[float, torch.Tensor],
        bmm2_scale: Union[float, torch.Tensor],
        sinks: Optional[List[torch.Tensor]],
        enable_pdl: Optional[bool],
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
    ) -> torch.Tensor:
        if return_lse or lse is not None:
            raise NotImplementedError(
                "XQA MLA backend does not support return_lse/lse output"
            )
        if not is_sm12x_supported(query.device):
            raise ValueError(
                "XQA MLA requires SM120a (CUDA >= 12.8) or SM121a (CUDA >= 12.9)"
            )
        if query.size(1) != 1:
            q_len_per_request = query.size(1)
            raise ValueError(
                f"XQA MLA only supports q_len_per_request == 1, got {q_len_per_request}"
            )
        fp8_ok = (
            query.dtype == torch.float8_e4m3fn and kv_cache.dtype == torch.float8_e4m3fn
        )
        bf16_ok = query.dtype == torch.bfloat16 and kv_cache.dtype == torch.bfloat16
        if not (fp8_ok or bf16_ok):
            raise ValueError(
                "XQA MLA supports (fp8, fp8) or (bfloat16, bfloat16) only, "
                f"got {query.dtype} and {kv_cache.dtype}"
            )
        _validate_xqa_mla_scales(
            query,
            kv_cache,
            bmm1_scale=bmm1_scale,
            bmm2_scale=bmm2_scale,
        )
        if sinks is not None:
            raise ValueError("XQA MLA does not support sinks")

        block_size = kv_cache.size(-2)
        kv_cache = _check_xqa_mla_shape(
            query,
            kv_cache,
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
            block_tables=block_tables,
            page_size=block_size,
        )

        if out is None:
            out_shape = query.shape[:-1] + (kv_lora_rank,)
            out = torch.empty(out_shape, dtype=torch.bfloat16, device=query.device)
        else:
            batch_size, _, num_q_heads, _ = query.shape
            check_shape_dtype_device(
                out,
                [batch_size, num_q_heads, kv_lora_rank],
                torch.bfloat16,
                query.device,
                "out",
            )

        enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        sm_count = get_device_sm_count(query.device)
        workspace_u8 = workspace_buffer.view(torch.uint8)
        semaphore = workspace_u8[: 8 * 1024 * 1024]
        scratch = workspace_u8[8 * 1024 * 1024 :]
        kv_cache_new = kv_cache.squeeze(1).unsqueeze(2)
        seq_lens_new = seq_lens.unsqueeze(1)

        xqa_mla(
            query,
            kv_cache_new,
            kv_cache_new,
            block_tables,
            seq_lens_new,
            out,
            scratch,
            semaphore,
            block_size,
            q_scale=bmm1_scale,
            kv_scale=bmm2_scale,
            sm_count=sm_count,
            enable_pdl=enable_pdl,
        )
        return out


class _BatchMLAPagedAttentionXqaBackend:
    """Planned XQA MLA execution with a launch-only hot path."""

    def __init__(self, float_workspace_buffer: torch.Tensor) -> None:
        self.device = float_workspace_buffer.device
        self._float_workspace_buffer = float_workspace_buffer

    def plan(
        self,
        *,
        cum_seq_lens_q: torch.Tensor,
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
            raise ValueError("cum_seq_lens_q must be contiguous for XQA MLA wrapper.")
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
        seq_lens_host = seq_lens.to(device="cpu", dtype=torch.int64)
        if torch.any(seq_lens_host < 0).item():
            raise ValueError("seq_lens must be nonnegative for XQA MLA wrapper.")
        max_seq_len = block_tables.shape[1] * page_size
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
        if not self._float_workspace_buffer.is_contiguous():
            raise ValueError("workspace buffer must be contiguous for XQA MLA wrapper.")
        workspace_u8 = self._float_workspace_buffer.view(torch.uint8).flatten()
        if workspace_u8.numel() < _XQA_MIN_WORKSPACE_BYTES:
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
        bmm1_scale: Optional[float] = None,
        bmm2_scale: Optional[float] = None,
    ) -> torch.Tensor:
        if not hasattr(self, "_module"):
            raise RuntimeError(
                "_BatchMLAPagedAttentionXqaBackend.run() called before plan()."
            )
        if return_lse or lse is not None:
            raise ValueError("XQA MLA wrapper does not support LSE output.")
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
        self._module.xqa_mla(
            self._sm_count,
            self._bmm1_scale if bmm1_scale is None else bmm1_scale,
            out,
            query,
            kv_cache,
            kv_cache,
            self._block_tables,
            self._max_seq_len,
            self._seq_lens_2d,
            self._batch_size,
            self._bmm2_scale if bmm2_scale is None else bmm2_scale,
            self._semaphore,
            self._scratch,
            self._enable_pdl,
        )
        return out
