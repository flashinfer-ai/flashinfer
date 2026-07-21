"""Context-parallel GDN prefill kernels for Blackwell SM100."""

from typing import Any

import torch
import cutlass
import cutlass.cute as cute

from ..delta_rule_dsl.custom_compile_cache import cached_compile
from ..delta_rule_dsl.delta_rule_cp_sm120 import (
    CPDeltaRuleFixupHmmaSm120,
    CPDeltaRuleFixupSimtSm120,
    CPDeltaRuleTPrecomputeSm120,
)
from ..delta_rule_dsl.varlen_helper import (
    max_num_chunks_host,
    workspace_num_chunks_host,
)
from .gated_delta_net_cp import (
    CPDeltaRuleMNPrecomputeUtcmma1Sm100,
    CPDeltaRuleMNPrecomputeUtcmma2Sm100,
)


def _blackwell_arch(device: torch.device) -> cute.GPUArch:
    major, minor = torch.cuda.get_device_capability(device)
    if major != 10:
        raise RuntimeError(
            f"SM100 CP delta rule requires a compute 10.x device, got {major}.{minor}"
        )
    return cute.GPUArch(f"sm_{major}{minor}a")


class CPDeltaRuleTPrecomputeSm100(CPDeltaRuleTPrecomputeSm120):
    """SM100 specialization of the architecture-portable T precompute body."""


def cp_delta_rule_t_precompute_dsl_sm100(
    k: torch.Tensor,
    beta: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
):
    """Precompute signed, beta-folded CP T tiles on SM100."""
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if beta.ndim != 2 or beta.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"beta must have shape (total_seqlen, num_sab_heads), got {tuple(beta.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_sab_heads = beta.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "beta heads must be a positive multiple of k heads, "
                f"got beta heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if d != 128:
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm100 only supports D=128, got {d}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleTPrecomputeSm100 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if beta.dtype != torch.float32:
            raise RuntimeError(f"beta must have dtype torch.float32, got {beta.dtype}")
        for name, tensor in (("k", k), ("beta", beta)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")

    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    max_t_blocks_per_seq = max_num_chunks_host(max_seqlen, 64)
    t = torch.empty(
        (total_t_blocks, num_sab_heads, 64, 64), dtype=k.dtype, device=k.device
    )
    if total_t_blocks == 0:
        return t
    k_tma = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))
    kernel_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
        k.dtype
    ]
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
        *args, **{**kwargs, "enable_tvm_ffi": True}
    )
    kernel = CPDeltaRuleTPrecomputeSm100(kernel_dtype)
    kernel_args = (
        from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(beta.reshape(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(t.view(-1), assumed_align=128).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(total_t_blocks),
        cutlass.Int32(max_t_blocks_per_seq),
        cutlass.Int32(num_seqs),
        stream,
    )
    compiled = cached_compile(
        kernel, *kernel_args, compile_options=(_blackwell_arch(k.device),)
    )
    compiled(*kernel_args)
    return t


def cp_delta_rule_mn_precompute_dsl_sm100(
    k: torch.Tensor,
    v: torch.Tensor,
    t: torch.Tensor,
    alpha: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    max_seqlen: int | None = None,
    *,
    _skip_check: bool = False,
    _kernel_kind: str = "utcmma_2sm",
):
    """Precompute local transfer and state workspaces on SM100."""
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if k.ndim != 3:
            raise RuntimeError(
                f"k must have shape (total_seqlen, num_k_heads, D), got {tuple(k.shape)}"
            )
        if v.ndim != 3 or v.shape[0] != k.shape[0] or v.shape[2] != k.shape[2]:
            raise RuntimeError(
                f"v must have shape (total_seqlen, num_v_heads, D), got {tuple(v.shape)}"
            )
        if alpha.ndim != 2 or alpha.shape[0] != k.shape[0]:
            raise RuntimeError(
                f"alpha must have shape (total_seqlen, num_sab_heads), got {tuple(alpha.shape)}"
            )
        if total_seqlen != k.shape[0]:
            raise RuntimeError(
                f"total_seqlen must match k.shape[0], got {total_seqlen} and {k.shape[0]}"
            )
        if cp_chunk_len % 64 != 0:
            raise RuntimeError(
                f"cp_chunk_len must be a multiple of 64, got {cp_chunk_len}"
            )
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1
    if max_seqlen is None:
        raise RuntimeError("max_seqlen must be provided")
    if not _skip_check and max_seqlen <= 0:
        raise RuntimeError(f"max_seqlen must be positive, got {max_seqlen}")
    _, num_k_heads, d = k.shape
    num_v_heads = v.shape[1]
    num_sab_heads = alpha.shape[1]
    if not _skip_check:
        if num_sab_heads < num_k_heads or num_sab_heads % num_k_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of k heads, "
                f"got alpha heads={num_sab_heads} and k heads={num_k_heads}"
            )
        if num_sab_heads < num_v_heads or num_sab_heads % num_v_heads != 0:
            raise RuntimeError(
                "alpha heads must be a positive multiple of v heads, "
                f"got alpha heads={num_sab_heads} and v heads={num_v_heads}"
            )
    total_t_blocks = workspace_num_chunks_host(cu_seqlens, 64, total_seqlen)
    if not _skip_check and t.shape != (total_t_blocks, num_sab_heads, 64, 64):
        raise RuntimeError(
            f"t must have shape {(total_t_blocks, num_sab_heads, 64, 64)}, got {tuple(t.shape)}"
        )
    total_cp_chunks = workspace_num_chunks_host(cu_seqlens, cp_chunk_len, total_seqlen)
    if not _skip_check:
        if d != 128:
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm100 only supports D=128, got {d}"
            )
        if k.dtype not in (torch.float16, torch.bfloat16):
            raise RuntimeError(
                f"CPDeltaRuleMNPrecomputeSm100 only supports fp16/bf16 inputs, got {k.dtype}"
            )
        if v.dtype != k.dtype or t.dtype != k.dtype:
            raise RuntimeError(
                f"v/t dtypes must match k dtype, got k={k.dtype}, v={v.dtype}, t={t.dtype}"
            )
        if alpha.dtype != torch.float32:
            raise RuntimeError(
                f"alpha must have dtype torch.float32, got {alpha.dtype}"
            )
        for name, tensor in (("k", k), ("v", v), ("t", t), ("alpha", alpha)):
            if not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")

    max_cp_chunks_per_seq = max_num_chunks_host(max_seqlen, cp_chunk_len)
    k_tma = k.as_strided((d, total_seqlen, num_k_heads), (1, num_k_heads * d, d))
    v_tma = v.as_strided((d, total_seqlen, num_v_heads), (1, num_v_heads * d, d))
    t_tma = t.as_strided(
        (64, 64, num_sab_heads, total_t_blocks),
        (64, 1, 64 * 64, num_sab_heads * 64 * 64),
    )
    transfer_t = torch.empty(
        (total_cp_chunks, num_sab_heads, d, d), dtype=torch.float32, device=k.device
    )
    state_t = torch.empty_like(transfer_t)
    if total_cp_chunks == 0:
        return transfer_t, state_t

    kernel_dtype = {torch.float16: cutlass.Float16, torch.bfloat16: cutlass.BFloat16}[
        k.dtype
    ]
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
        *args, **{**kwargs, "enable_tvm_ffi": True}
    )
    kernel: Any
    if _kernel_kind == "utcmma_2sm":
        kernel = CPDeltaRuleMNPrecomputeUtcmma2Sm100(kernel_dtype)
    elif _kernel_kind == "utcmma_1sm":
        kernel = CPDeltaRuleMNPrecomputeUtcmma1Sm100(kernel_dtype)
    else:
        raise ValueError(f"Unsupported MN precompute kernel kind: {_kernel_kind}")
    kernel_args = (
        from_dlpack(k_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(v_tma, assumed_align=16).mark_layout_dynamic(leading_dim=0),
        from_dlpack(t_tma, assumed_align=16).mark_layout_dynamic(leading_dim=1),
        from_dlpack(alpha.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(transfer_t.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(state_t.view(-1), assumed_align=16).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(cp_chunk_len),
        cutlass.Int32(num_k_heads),
        cutlass.Int32(num_v_heads),
        cutlass.Int32(num_sab_heads),
        cutlass.Int32(total_cp_chunks),
        cutlass.Int32(max_cp_chunks_per_seq),
        cutlass.Int32(num_seqs),
        stream,
    )
    compiled = cached_compile(
        kernel, *kernel_args, compile_options=(_blackwell_arch(k.device),)
    )
    compiled(*kernel_args)
    return transfer_t, state_t


class CPDeltaRuleFixupSm100(CPDeltaRuleFixupSimtSm120):
    """SM100 specialization of the register-prefetched SIMT fixup body."""


class CPDeltaRuleFixupHmmaSm100(CPDeltaRuleFixupHmmaSm120):
    """SM100 specialization of the HMMA fixup fallback."""


def cp_delta_rule_fixup_dsl_sm100(
    local_transfer: torch.Tensor,
    local_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
    total_seqlen: int,
    cp_chunk_len: int = 4096,
    initial_state: torch.Tensor | None = None,
    *,
    _skip_check: bool = False,
    _kernel_kind: str | None = None,
):
    """Fix SM100 CP chunk states into global chunk-boundary states."""
    import cuda.bindings.driver as cuda_driver

    if not _skip_check:
        if local_transfer.ndim != 4:
            raise RuntimeError(
                "local_transfer must have shape (num_chunks, num_heads, DimV, DimK), "
                f"got {tuple(local_transfer.shape)}"
            )
        if local_state.shape != local_transfer.shape:
            raise RuntimeError(
                "local_state shape must match local_transfer shape, "
                f"got {tuple(local_state.shape)} and {tuple(local_transfer.shape)}"
            )
        if local_transfer.shape[-2:] != (128, 128):
            raise RuntimeError(
                f"CPDeltaRuleFixupSm100 only supports D=128, got {tuple(local_transfer.shape[-2:])}"
            )
        if local_transfer.dtype != torch.float32 or local_state.dtype != torch.float32:
            raise RuntimeError(
                "CPDeltaRuleFixupSm100 only supports float32 inputs, "
                f"got {local_transfer.dtype} and {local_state.dtype}"
            )
        if initial_state is not None:
            expected_initial_state_shape = (
                cu_seqlens.shape[0] - 1,
                local_transfer.shape[1],
                128,
                128,
            )
            if initial_state.shape != expected_initial_state_shape:
                raise RuntimeError(
                    f"initial_state must have shape {expected_initial_state_shape}, got {tuple(initial_state.shape)}"
                )
            if initial_state.dtype != torch.float32:
                raise RuntimeError(
                    f"initial_state must have dtype torch.float32, got {initial_state.dtype}"
                )
        for name, tensor in (
            ("local_transfer", local_transfer),
            ("local_state", local_state),
            ("initial_state", initial_state),
        ):
            if tensor is not None and not tensor.is_contiguous():
                raise RuntimeError(f"{name} must be contiguous")

    total_cp_chunks, num_heads, _, d = local_transfer.shape
    if not _skip_check:
        if cp_chunk_len <= 0:
            raise RuntimeError(f"cp_chunk_len must be positive, got {cp_chunk_len}")
        if cu_seqlens.dtype != torch.int64:
            raise RuntimeError(
                f"cu_seqlens must have dtype torch.int64, got {cu_seqlens.dtype}"
            )
        expected_chunks = workspace_num_chunks_host(
            cu_seqlens, cp_chunk_len, total_seqlen
        )
        if expected_chunks != total_cp_chunks:
            raise RuntimeError(
                "local_transfer/local_state first dim must equal "
                f"chunk_bound(num_seqs, total_seqlen, cp_chunk_len)={expected_chunks}, got {total_cp_chunks}"
            )
        if not cu_seqlens.is_contiguous():
            raise RuntimeError("cu_seqlens must be contiguous")
    num_seqs = cu_seqlens.shape[0] - 1

    fixed_state = torch.empty_like(local_state)
    if total_cp_chunks == 0:
        return fixed_state

    local_transfer_tma = local_transfer.as_strided(
        (d, d, num_heads, total_cp_chunks), (d, 1, d * d, num_heads * d * d)
    )
    local_state_tma = local_state.as_strided(
        (d, d, num_heads, total_cp_chunks), (d, 1, d * d, num_heads * d * d)
    )
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    from_dlpack = lambda *args, **kwargs: cute.runtime.from_dlpack(
        *args, **{**kwargs, "enable_tvm_ffi": True}
    )
    needs_initial_state = initial_state is not None
    initial_state_cute = (
        from_dlpack(initial_state.reshape(-1), assumed_align=16).mark_layout_dynamic()
        if needs_initial_state
        else None
    )
    if _kernel_kind is None:
        if num_heads <= 8:
            _kernel_kind = "simt_row4"
        elif num_heads <= 16:
            _kernel_kind = "simt_row8"
        else:
            _kernel_kind = "hmma"
    kernel: Any
    if _kernel_kind == "simt_row4":
        kernel = CPDeltaRuleFixupSm100(needs_initial_state, 4)
    elif _kernel_kind == "simt_row8":
        kernel = CPDeltaRuleFixupSm100(needs_initial_state, 8)
    elif _kernel_kind == "hmma":
        kernel = CPDeltaRuleFixupHmmaSm100(needs_initial_state)
    else:
        raise ValueError(f"Unsupported fixup kernel kind: {_kernel_kind}")
    kernel_args = (
        from_dlpack(local_transfer_tma, assumed_align=128).mark_layout_dynamic(
            leading_dim=1
        ),
        from_dlpack(local_state_tma, assumed_align=128).mark_layout_dynamic(
            leading_dim=1
        ),
        initial_state_cute,
        from_dlpack(fixed_state.reshape(-1), assumed_align=128).mark_layout_dynamic(),
        from_dlpack(cu_seqlens, assumed_align=8).mark_layout_dynamic(),
        cutlass.Int32(cp_chunk_len),
        cutlass.Int32(total_cp_chunks),
        cutlass.Int32(num_seqs),
        cutlass.Int32(num_heads),
        stream,
    )
    compiled = cached_compile(
        kernel, *kernel_args, compile_options=(_blackwell_arch(local_transfer.device),)
    )
    compiled(*kernel_args)
    return fixed_state
