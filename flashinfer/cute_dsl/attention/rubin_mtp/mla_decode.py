# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepared interface to the Rubin paired-CGA absorbed MLA MTP kernel.

The launch split count is a static budget. Each request partitions its current
on-device KV length into at most that many nonempty contiguous 128-token tiles.
No device length is copied to the host, including on a prepared cache hit.
"""

import functools
import importlib
from typing import Optional

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.cute_dsl._mla_validation import _validate_mtp_scales
from flashinfer.cute_dsl.utils import (
    _as_cute_dsl_workspace_i8,
    cute_dsl_compile_arch,
    get_max_active_clusters,
    get_num_sm,
)
from flashinfer.utils import get_compute_capability

LOG2_E = 1.4426950408889634
MAX_SPLITS = 32
_INDEX_LIMIT = 1 << 31


@functools.cache
def _check_compiler_compatibility() -> None:
    """Refuse a compiler lacking the Rubin mixed-cluster pipeline, lazily."""
    rubin = importlib.import_module("cutlass.utils.rubin_helpers")
    utils = importlib.import_module("cutlass.utils")
    pipeline = importlib.import_module("cutlass.pipeline")
    dsl = importlib.import_module("cutlass.cutlass_dsl")
    cpasync = importlib.import_module("cutlass.cute.nvgpu.cpasync")
    base_dsl = importlib.import_module("cutlass.base_dsl.dsl")
    required = (
        (rubin, "make_trivial_tiled_mma"),
        (utils, "ClcDynamicPersistentTileSchedulerParams"),
        (pipeline, "PipelineClcFetchAsync"),
        (cpasync, "CopyBulkS2SOp"),
        (dsl, "if_generate"),
    )
    missing = [
        f"{module.__name__}.{name}"
        for module, name in required
        if not hasattr(module, name)
    ]
    launch_config = getattr(getattr(base_dsl, "BaseDSL", None), "LaunchConfig", None)
    launch_fields = getattr(launch_config, "__dataclass_fields__", {})
    missing.extend(
        f"BaseDSL.LaunchConfig.{name}"
        for name in ("fallback_cluster", "smem_merge_branch_allocs")
        if name not in launch_fields
    )
    if missing:
        raise ImportError(
            "Rubin MTP requires a mixed-CGA capable CuTe DSL compiler; missing "
            + ", ".join(missing)
        )


@functools.cache
def _check_can_implement(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    num_heads: int,
    seq_len_q: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    enable_dcp: bool = False,
    cp_world: int = 1,
) -> None:
    _check_compiler_compatibility()
    if (
        torch_dtype != torch.float8_e4m3fn
        or torch_out_dtype != torch.float8_e4m3fn
        or page_size not in (64, 128)
        or num_heads != 128
        or seq_len_q not in (2, 4)
        or kv_lora_rank != 512
        or qk_rope_head_dim != 64
        or is_var_split_kv
        or enable_dcp
        or cp_world != 1
    ):
        raise ValueError(
            "cute-dsl-rubin-mtp requires FP8 E4M3FN input/output, H=128, "
            "uniform Q=2/4, latent=512, RoPE=64, page=64/128, and no DCP"
        )


def _check_tensor_indexing(tensor: torch.Tensor, name: str) -> None:
    """Check address spans using host tensor metadata, never device contents.

    Non-KV layouts still construct offsets from signed 32-bit dimensions.
    KV TMA metadata is checked separately by _check_kv_tensor_indexing.
    """
    if tensor.numel() == 0:
        return
    span = 1 + sum(
        (size - 1) * stride
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True)
    )
    if span >= _INDEX_LIMIT or any(stride < 0 for stride in tensor.stride()):
        raise ValueError(f"{name} exceeds the Rubin MTP signed 32-bit indexing range")


def _check_kv_tensor_indexing(tensor: torch.Tensor, name: str) -> None:
    """Bound KV metadata passed through the signed32 fake-tensor ABI.

    The wrapper requires a contiguous packed pool. Its latent/RoPE views use
    TMA page coordinates and wide descriptor addressing in both kernel branches,
    so the whole pool span need not fit signed32. Each dimension and stride
    passed to the compiled callable must still fit.
    """
    if any(
        value < 0 or value >= _INDEX_LIMIT
        for value in (*tensor.shape, *tensor.stride())
    ):
        raise ValueError(f"{name} exceeds the Rubin MTP signed 32-bit indexing range")


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
    max_seq_len: Optional[int] = None,
    occupancy_q_tiles: Optional[int] = None,
    num_kv_splits: Optional[int] = None,
) -> tuple[int, int]:
    """Return an exact split budget and byte capacity, with zero bytes at one."""
    if B <= 0 or q_len not in (2, 4) or H != 128 or kv_lora_rank != 512:
        raise ValueError("Rubin MTP workspace requires B>0, Q=2/4, H=128, D=512")
    # Q is folded as a view; its combined 576-element row stride is retained.
    if B * q_len * H * (kv_lora_rank + 64) >= _INDEX_LIMIT:
        raise ValueError("query exceeds the Rubin MTP signed 32-bit indexing range")
    if max_seq_len is None or max_seq_len <= 0:
        raise ValueError("max_seq_len must be a positive prepared capacity")
    if num_kv_splits is not None and (
        isinstance(num_kv_splits, bool) or not isinstance(num_kv_splits, int)
    ):
        raise ValueError("num_kv_splits must be an integer or None")
    if num_kv_splits is None or num_kv_splits == -1:
        tiles = (max_seq_len + 127) // 128
        candidate = min(tiles, max(1, max_active_blocks // B // q_len))
        tiles_per_split = (tiles + candidate - 1) // candidate
        split_kv = min(MAX_SPLITS, (tiles + tiles_per_split - 1) // tiles_per_split)
    else:
        if (
            isinstance(num_kv_splits, bool)
            or not isinstance(num_kv_splits, int)
            or not 1 <= num_kv_splits <= MAX_SPLITS
        ):
            raise ValueError(
                f"num_kv_splits must be -1 or an integer in [1, {MAX_SPLITS}]"
            )
        split_kv = num_kv_splits
    scratch_elements = B * q_len * H * split_kv * (kv_lora_rank + 1)
    if split_kv > 1 and scratch_elements >= _INDEX_LIMIT:
        raise ValueError("workspace exceeds the Rubin MTP signed 32-bit indexing range")
    workspace = 0 if split_kv == 1 else scratch_elements * 4
    return split_kv, workspace


@functools.cache
def _get_compiled_mla_kernel(
    arch: str,
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    num_heads: int,
    seq_len_q: int,
    is_persistent: bool = True,
    is_var_seq: bool = True,
    is_var_q: bool = False,
    is_var_split_kv: bool = False,
    reducer_d_tiles: int = 1,
    reducer_max_splits: int = MAX_SPLITS,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    enable_pdl: bool = False,
    enable_dcp: bool = False,
    cp_world: int = 1,
):
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Rubin MTP must be prepared before CUDA graph capture")
    _check_can_implement(
        torch_dtype,
        torch_out_dtype,
        page_size,
        num_heads,
        seq_len_q,
        kv_lora_rank,
        qk_rope_head_dim,
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        enable_dcp,
        cp_world,
    )
    if (
        not arch.startswith("sm_107")
        or is_var_q
        or enable_pdl
        or skip_correction_threshold
    ):
        raise ValueError(
            "Rubin MTP requires SM107, uniform Q, default softmax and PDL disabled"
        )
    _check_compiler_compatibility()
    from .kernel import (
        RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo,
        _MixedFallbackPrep,
    )

    capacity = max(get_max_active_clusters(4), get_max_active_clusters(2) // 2)
    kernel = RubinMultiHeadLatentAttentionForwardFP8TwoPlusTwo(
        cutlass.Float32,
        cutlass.Float32,
        (256, 128),
        (256, 256),
        capacity,
        page_size,
        0.0,
        True,
        False,
        False,
        use_fp16_softmax=False,
        force_branch="auto",
    )
    kernel.causal_num_heads = 128
    kernel.causal_seq_len_q = seq_len_q
    kernel.causal_fold_ratio = 2
    kernel._fb = _MixedFallbackPrep(
        cutlass.Float32,
        cutlass.Float32,
        (128, 128),
        (128, 256),
        capacity,
        page_size,
        0.0,
        True,
        False,
        False,
        fold_sq=False,
        num_heads=128,
        seq_len_q=seq_len_q,
        use_fp16_softmax=False,
    )
    kernel._fb.causal_num_heads = 128
    kernel._fb.early_compute_clc = True
    kernel._fb.merge_softmax_loops = False

    sh, dl, dr, ss, sb, sp, sk, pc = (
        cute.sym_int(divisibility=16) if i in (1, 2) else cute.sym_int()
        for i in range(8)
    )

    def fake(dtype, shape, unit, align=16):
        return cute.runtime.make_fake_tensor(
            dtype,
            shape,
            stride=tuple(1 if j == unit else cute.sym_int() for j in range(len(shape))),
            assumed_align=align,
        )

    def compact(dtype, shape, order):
        return cute.runtime.make_fake_compact_tensor(
            dtype, shape, stride_order=order, assumed_align=16
        )

    compiled = cute.compile(
        kernel,
        fake(cutlass.Float8E4M3FN, (sh, dl, ss, sb), 1),
        fake(cutlass.Float8E4M3FN, (sh, dr, ss, sb), 1),
        fake(cutlass.Float8E4M3FN, (sk, dl, sp), 1),
        fake(cutlass.Float8E4M3FN, (sk, dr, sp), 1),
        compact(Int32, (pc, sb), (0, 1)),
        compact(cutlass.Float8E4M3FN, (sh, dl, ss, sb), (1, 0, 2, 3)),
        compact(Float32, (sh, ss, sb), (0, 1, 2)),
        (
            None
            if is_workspace_size_zero
            # The byte extent can exceed 2 GiB while FP32 element indices fit.
            else fake(cutlass.Int8, (cute.sym_int(64),), 0, 32)
        ),
        Int32(1),
        fake(Int32, (sb,), 0),
        None,
        Float32(1),
        Float32(1),
        Float32(1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--enable-tvm-ffi --opt-level 2 --gpu-arch {arch}",
    )

    def launch(
        q_nope,
        q_pe,
        c_nope,
        c_pe,
        block_tables,
        out,
        lse,
        workspace,
        splits,
        seq_lens,
        cum_seq_lens_q,
        causal_seqlens,
        cp_rank,
        block_splits,
        softmax_scale,
        output_scale,
        lse_scale,
    ):
        # Prepared callers can change storage while retaining a compiled kernel.
        # These checks inspect metadata only and also cover strided views whose
        # storage span is larger than their logical element count.
        for name, tensor in (
            ("query latent", q_nope),
            ("query rope", q_pe),
            ("block_tables", block_tables),
            ("out", out),
            ("lse", lse),
        ):
            _check_tensor_indexing(tensor, name)
        _check_kv_tensor_indexing(c_nope, "KV latent")
        _check_kv_tensor_indexing(c_pe, "KV rope")
        # Fold pairs of original query tokens into 256 rows without copying.
        batch = q_nope.shape[0]
        folded_q = seq_len_q // 2
        q_nope = q_nope.view(batch, folded_q, 256, 512).permute(2, 3, 1, 0)
        q_pe = q_pe.view(batch, folded_q, 256, 64).permute(2, 3, 1, 0)
        compiled(
            q_nope,
            q_pe,
            c_nope.permute(1, 2, 0),
            c_pe.permute(1, 2, 0),
            block_tables.T,
            out.view(batch, folded_q, 256, 512).permute(2, 3, 1, 0),
            lse.view(batch, folded_q, 256).permute(2, 1, 0),
            workspace,
            splits,
            seq_lens,
            None,
            softmax_scale,
            output_scale,
            lse_scale,
        )

    return launch


def prepare_cute_dsl_mla_decode(
    *,
    device,
    torch_dtype,
    torch_out_dtype,
    page_size,
    batch_size,
    num_heads,
    seq_len_q,
    kv_lora_rank,
    qk_rope_head_dim,
    max_seq_len,
    num_kv_splits=None,
):
    """Compile before capture; return a monolithic-compatible prepared launch."""
    splits, size = _get_split_kv_and_workspace_size(
        batch_size,
        seq_len_q,
        num_heads,
        kv_lora_rank,
        get_num_sm(device),
        max_seq_len,
        num_kv_splits=num_kv_splits,
    )
    return _get_compiled_mla_kernel(
        arch=cute_dsl_compile_arch(*get_compute_capability(device)),
        torch_dtype=torch_dtype,
        torch_out_dtype=torch_out_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        num_heads=num_heads,
        seq_len_q=seq_len_q,
        is_workspace_size_zero=size == 0,
    )


def cute_dsl_mla_decode(
    query,
    kv_cache,
    workspace_buffer,
    kv_lora_rank,
    qk_rope_head_dim,
    block_tables,
    seq_lens,
    max_seq_len,
    softmax_scale,
    output_scale=1.0,
    out=None,
    out_dtype=None,
    is_var_seq=True,
    enable_pdl=None,
    lse=None,
    return_lse=False,
    lse_scale=1 / LOG2_E,
    cum_seq_lens_q=None,
    max_q_len=None,
    enable_dcp=False,
    cp_world=1,
    cp_rank=0,
    causal_seqlens_kv_global=None,
    num_kv_splits=None,
):
    """Run absorbed MLA with uniform Q=2/4 and live per-request KV lengths."""
    _validate_mtp_scales(softmax_scale, output_scale)
    if query.ndim != 4:
        raise ValueError("Rubin MTP query must have shape [B,Q,128,576]")
    B, Q, H, D = query.shape
    dtype = out.dtype if out is not None else out_dtype or torch.bfloat16
    if kv_cache.ndim == 4 and kv_cache.shape[1] == 1:
        kv_cache = kv_cache.squeeze(1)
    if (
        kv_cache.ndim != 3
        or D != 576
        or kv_cache.shape[-1] != 576
        or kv_cache.dtype != query.dtype
    ):
        raise ValueError("Rubin MTP requires a combined 576-element FP8 paged cache")
    if (
        enable_pdl
        or cum_seq_lens_q is not None
        or enable_dcp
        or cp_world != 1
        or cp_rank != 0
        or causal_seqlens_kv_global is not None
    ):
        raise ValueError("Rubin MTP does not support PDL, ragged Q or DCP")
    _check_can_implement(
        query.dtype,
        dtype,
        kv_cache.shape[1],
        H,
        Q,
        kv_lora_rank,
        qk_rope_head_dim,
        True,
        is_var_seq,
        False,
    )
    if query.stride(-1) != 1 or query.stride(-3) != H * query.stride(-2):
        query = query.contiguous()
    if kv_cache.stride(-1) != 1:
        raise ValueError("Rubin MTP KV cache requires a contiguous last dimension")
    if (
        block_tables.dtype != torch.int32
        or block_tables.ndim != 2
        or block_tables.shape[0] != B
        or not block_tables.is_contiguous()
    ):
        raise ValueError("block_tables must be contiguous int32 [B,max_pages]")
    if (
        seq_lens.dtype != torch.int32
        or tuple(seq_lens.shape) != (B,)
        or not seq_lens.is_contiguous()
    ):
        raise ValueError("seq_lens must be contiguous int32 [B]")
    if max_seq_len > block_tables.shape[1] * kv_cache.shape[1]:
        raise ValueError("max_seq_len exceeds the page-table capacity")
    for tensor in (kv_cache, workspace_buffer, block_tables, seq_lens):
        if tensor.device != query.device:
            raise ValueError("all Rubin MTP tensors must be on the query device")
    shape = (B, Q, H, kv_lora_rank)
    if out is None:
        out = torch.empty(shape, dtype=dtype, device=query.device)
    elif (
        tuple(out.shape) != shape
        or not out.is_contiguous()
        or out.device != query.device
    ):
        raise ValueError(
            "out must be a contiguous [B,Q,H,512] tensor on the query device"
        )
    if lse is None:
        lse = torch.empty((B, Q, H), dtype=torch.float32, device=query.device)
    elif (
        tuple(lse.shape) not in ((B, Q, H), (B * Q, H))
        or lse.dtype != torch.float32
        or not lse.is_contiguous()
        or lse.device != query.device
    ):
        raise ValueError(
            "lse must be contiguous float32 [B,Q,H] or [B*Q,H] on the query device"
        )
    splits, size = _get_split_kv_and_workspace_size(
        B,
        Q,
        H,
        kv_lora_rank,
        get_num_sm(query.device),
        max_seq_len,
        num_kv_splits=num_kv_splits,
    )
    workspace = _as_cute_dsl_workspace_i8(workspace_buffer)
    if workspace.numel() < size:
        raise ValueError(
            f"workspace_buffer too small: {workspace.numel()} bytes, need {size} bytes"
        )
    launch = prepare_cute_dsl_mla_decode(
        device=query.device,
        torch_dtype=query.dtype,
        torch_out_dtype=dtype,
        page_size=kv_cache.shape[1],
        batch_size=B,
        num_heads=H,
        seq_len_q=Q,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        max_seq_len=max_seq_len,
        num_kv_splits=splits,
    )
    launch(
        query[..., :512],
        query[..., 512:],
        kv_cache[..., :512],
        kv_cache[..., 512:],
        block_tables,
        out,
        lse,
        workspace[:size] if size else None,
        Int32(splits),
        seq_lens,
        None,
        None,
        Int32(0),
        None,
        Float32(softmax_scale),
        Float32(output_scale),
        Float32(lse_scale),
    )
    return (out, lse) if return_lse else out
