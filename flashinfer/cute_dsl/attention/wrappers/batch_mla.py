# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchMLADecodeCuteDSLWrapper — PyTorch-facing API for MLA decode attention.

Constructs MLAConfig from user-facing parameters, compiles the modular
BlackwellMultiLatentAttentionForward kernel, and provides plan()/run().

Also re-exports a standalone `cute_dsl_mla_decode` function that mirrors the
original integration layer in flashinfer.mla.cute_dsl.mla_decode but uses the
modular kernel.
"""

import functools
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32

from flashinfer.api_logging import flashinfer_api
from flashinfer.utils import device_support_pdl
from flashinfer.cute_dsl.utils import (
    get_max_active_clusters,
    get_num_sm,
    torch_to_cutlass_dtype,
)

from ..config import AttentionFusion
from ..fusion.variant import AttentionVariant, StandardAttention
from ..mla_decode import BlackwellMultiLatentAttentionForward
from ..mla_decode_fp8 import BlackwellMultiLatentAttentionForwardFP8
from ..mla_config import MLAConfig


# ---------------------------------------------------------------------------
# Cached helpers (deterministic for the same args ⇒ safe to @functools.cache)
# ---------------------------------------------------------------------------


@functools.cache
def _get_split_kv_and_workspace_size(
    B: int,
    q_len: int,
    H: int,
    kv_lora_rank: int,
    max_active_blocks: int,
) -> Tuple[int, int]:
    """Cache split_kv and workspace_size since they are deterministic for the same params."""
    split_kv = BlackwellMultiLatentAttentionForward.get_split_kv_simplified(
        B, q_len, max_active_blocks
    )
    workspace_size = BlackwellMultiLatentAttentionForward.get_workspace_size(
        H, q_len, kv_lora_rank, B, split_kv, cutlass.Float32
    )
    return split_kv, workspace_size


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
) -> None:
    """Check if the kernel supports the given configuration (cached)."""
    mma_qk_tiler_mn = (128, 128)
    mma_pv_tiler_mn = (128, 256)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    KernelClass = (
        BlackwellMultiLatentAttentionForwardFP8
        if is_fp8
        else BlackwellMultiLatentAttentionForward
    )
    cutlass_in_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)
    if not KernelClass.can_implement(
        1,  # B (runtime, use placeholder)
        seq_len_q,
        1,  # K (runtime, use placeholder)
        num_heads,
        kv_lora_rank,
        qk_rope_head_dim,
        cutlass_in_dtype,
        cutlass_out_dtype,
        cutlass.Float32,
        cutlass.Float32,
        mma_qk_tiler_mn,
        mma_pv_tiler_mn,
        1,  # split_kv (runtime, use 1 to pass the H<128 check)
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        page_size,
    ):
        raise ValueError(
            f"cute_dsl_mla_decode: unsupported configuration "
            f"(q_len={seq_len_q}, num_heads={num_heads}, page_size={page_size}, "
            f"in_dtype={torch_dtype}, out_dtype={torch_out_dtype})"
        )


def _make_mla_fake_tensors(
    cutlass_dtype,
    cutlass_out_dtype,
    is_workspace_size_zero: bool,
    is_var_split_kv: bool,
):
    """Create fake tensors for MLA kernel compilation (shared by all paths)."""
    sym_heads = cute.sym_int()
    sym_latent = cute.sym_int(divisibility=16)
    sym_seq_q = cute.sym_int()
    sym_rope = cute.sym_int(divisibility=16)
    sym_batch = cute.sym_int()
    sym_kv_batch = cute.sym_int()
    sym_seq_kv = cute.sym_int()
    sym_page_count = cute.sym_int()
    sym_workspace_size = cute.sym_int()

    q_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    q_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    c_latent_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_latent),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    c_rope_fake = cute.runtime.make_fake_tensor(
        cutlass_dtype,
        (sym_kv_batch, sym_seq_kv, sym_rope),
        stride=(cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    page_table_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch, sym_page_count),
        stride_order=(1, 0),
        assumed_align=16,
    )
    o_fake = cute.runtime.make_fake_compact_tensor(
        cutlass_out_dtype,
        (sym_batch, sym_seq_q, sym_heads, sym_latent),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Float32,
        (sym_batch, sym_seq_q, sym_heads),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    if is_workspace_size_zero:
        workspace_fake = None
    else:
        workspace_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int8,
            (sym_workspace_size,),
            assumed_align=32,
        )
    cache_seqs_fake = cute.runtime.make_fake_compact_tensor(
        cutlass.Int32,
        (sym_batch,),
        assumed_align=16,
    )
    if is_var_split_kv:
        block_split_kvs_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Int32,
            (sym_batch,),
            assumed_align=16,
        )
    else:
        block_split_kvs_fake = None

    return (
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        cache_seqs_fake,
        block_split_kvs_fake,
    )


def _make_mla_config(
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    page_size: int,
    skip_correction_threshold: float,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    enable_pdl: bool,
    is_fp8: bool,
) -> MLAConfig:
    """Create an MLAConfig with standard tiler settings."""
    cluster_shape_mnk = (2, 1, 1)
    return MLAConfig(
        latent_dim=kv_lora_rank,
        rope_dim=qk_rope_head_dim,
        acc_dtype=cutlass.Float32,
        lse_dtype=cutlass.Float32,
        mma_qk_tiler_mn=(128, 128),
        mma_pv_tiler_mn=(128, 256),
        max_active_clusters=get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        ),
        page_size=page_size,
        skip_correction_threshold=skip_correction_threshold,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        enable_pdl=enable_pdl,
        is_fp8=is_fp8,
        mma_o_stage=2 if is_fp8 else 1,
    )


@functools.cache
def _compile_mla_kernel(
    torch_dtype: torch.dtype,
    torch_out_dtype: torch.dtype,
    page_size: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    is_persistent: bool,
    is_var_seq: bool,
    is_var_split_kv: bool,
    skip_correction_threshold: float = 0.0,
    is_workspace_size_zero: bool = False,
    enable_pdl: bool = False,
    variant: Optional[AttentionVariant] = None,
    params_shape: Optional[tuple] = None,
) -> Callable:
    """Compile and cache an MLA decode kernel (standard or variant).

    Uses ``@functools.cache`` so repeated calls with the same arguments
    return the previously compiled kernel in microseconds rather than
    recompiling (~3 s).  For standard attention pass ``variant=None``
    (the default); for custom variants pass the variant instance (hashable
    by identity).

    ``AttentionFusion`` is constructed *inside* this function so it never
    appears in the cache key (it is unhashable).
    """
    if variant is None:
        variant = StandardAttention()
    fusion = AttentionFusion(variant=variant)

    cutlass_dtype = torch_to_cutlass_dtype(torch_dtype)
    cutlass_out_dtype = torch_to_cutlass_dtype(torch_out_dtype)

    is_fp8 = torch_dtype == torch.float8_e4m3fn
    config = _make_mla_config(
        kv_lora_rank,
        qk_rope_head_dim,
        page_size,
        skip_correction_threshold,
        is_persistent,
        is_var_seq,
        is_var_split_kv,
        enable_pdl,
        is_fp8,
    )

    kernel_obj = (
        BlackwellMultiLatentAttentionForwardFP8(config, fusion=fusion)
        if is_fp8
        else BlackwellMultiLatentAttentionForward(config, fusion=fusion)
    )

    fakes = _make_mla_fake_tensors(
        cutlass_dtype,
        cutlass_out_dtype,
        is_workspace_size_zero,
        is_var_split_kv,
    )
    (
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        cache_seqs_fake,
        block_split_kvs_fake,
    ) = fakes

    params_fake = None
    if params_shape is not None:
        ndim = len(params_shape)
        stride_order = tuple(range(ndim - 1, -1, -1))
        params_fake = cute.runtime.make_fake_compact_tensor(
            cutlass.Float32,
            params_shape,
            stride_order=stride_order,
            assumed_align=16,
        )

    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    compiled_kernel = cute.compile(
        kernel_obj,
        q_latent_fake,
        q_rope_fake,
        c_latent_fake,
        c_rope_fake,
        page_table_fake,
        o_fake,
        lse_fake,
        workspace_fake,
        Int32(1),  # split_kv placeholder
        cache_seqs_fake,
        block_split_kvs_fake,
        Float32(1.0),  # softmax_scale placeholder
        Float32(1.0),  # output_scale placeholder
        params_fake,
        stream_fake,
        options="--enable-tvm-ffi --opt-level 2",
    )

    return compiled_kernel


# ---------------------------------------------------------------------------
# BatchMLADecodeCuteDSLWrapper — stateful plan()/run() interface
# ---------------------------------------------------------------------------


class BatchMLADecodeCuteDSLWrapper:
    """PyTorch-facing wrapper for the modular MLA decode kernel.

    Usage::

        wrapper = BatchMLADecodeCuteDSLWrapper(workspace_buffer)
        wrapper.plan(
            kv_lora_rank=512, qk_rope_head_dim=64, num_heads=128,
            page_size=64, q_dtype=torch.bfloat16,
        )
        out = wrapper.run(query, kv_cache, block_tables, seq_lens, max_seq_len,
                          softmax_scale=0.125)
    """

    @flashinfer_api
    def __init__(self, workspace_buffer: torch.Tensor) -> None:
        assert workspace_buffer.dtype == torch.int8, (
            f"workspace_buffer must be torch.int8, got {workspace_buffer.dtype}"
        )
        self._workspace_buffer = workspace_buffer
        self._device = workspace_buffer.device
        self._compiled_kernel: Optional[Callable] = None

    @flashinfer_api
    def plan(
        self,
        kv_lora_rank: int = 512,
        qk_rope_head_dim: int = 64,
        num_heads: int = 128,
        page_size: int = 1,
        q_dtype: torch.dtype = torch.bfloat16,
        out_dtype: Optional[torch.dtype] = None,
        is_var_seq: bool = True,
        enable_pdl: Optional[bool] = None,
        variant: Optional[AttentionVariant] = None,
    ) -> None:
        """Compile (or retrieve cached) MLA decode kernel for the given config.

        Parameters
        ----------
        kv_lora_rank : int
            Latent dimension (e.g. 512).
        qk_rope_head_dim : int
            RoPE dimension (e.g. 64).
        num_heads : int
            Number of attention heads (typically 128 for DeepSeek-V3).
        page_size : int
            KV cache page size.
        q_dtype : torch.dtype
            Query/KV data type (float16 or bfloat16).
        out_dtype : Optional[torch.dtype]
            Output data type. Defaults to same as q_dtype.
        is_var_seq : bool
            Whether sequence lengths vary across the batch.
        enable_pdl : Optional[bool]
            Whether to enable Programmatic Dependent Launch. Auto-detects if None.
        variant : Optional[AttentionVariant]
            Attention variant (ALiBi, SoftCapping, AttentionWithSink, etc.).
            None uses standard softmax attention.
        """
        self._kv_lora_rank = kv_lora_rank
        self._qk_rope_head_dim = qk_rope_head_dim
        self._num_heads = num_heads
        self._page_size = page_size
        self._q_dtype = q_dtype
        if out_dtype is not None:
            self._o_dtype = out_dtype
        elif q_dtype == torch.float8_e4m3fn:
            self._o_dtype = torch.bfloat16
        else:
            self._o_dtype = q_dtype
        self._is_var_seq = is_var_seq
        self._is_persistent = not is_var_seq
        self._is_var_split_kv = False
        self._skip_correction_threshold = 0.0

        self._enable_pdl = (
            device_support_pdl(self._device) if enable_pdl is None else enable_pdl
        )

        if variant is None:
            variant = StandardAttention()
        self._variant = variant

        if self._variant.has_logits_transform:
            raise ValueError(
                "MLA decode does not support logits_transform. "
                "Use score_mod, update_statistics, or transform_output instead."
            )

        self._has_params = self._variant.extra_params is not None
        if self._has_params:
            ep = self._variant.extra_params.to(torch.float32).to(self._device)
            if not ep.is_contiguous():
                raise ValueError(
                    f"AttentionVariant.extra_params must be contiguous, "
                    f"got strides {ep.stride()} for shape {ep.shape}. "
                    f"Call .contiguous() before returning from extra_params."
                )
            self._params_torch = ep
        else:
            self._params_torch = None

        _check_can_implement(
            torch_dtype=self._q_dtype,
            torch_out_dtype=self._o_dtype,
            page_size=self._page_size,
            num_heads=self._num_heads,
            seq_len_q=1,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            is_persistent=self._is_persistent,
            is_var_seq=self._is_var_seq,
            is_var_split_kv=self._is_var_split_kv,
        )

        self._cache_variant = (
            self._variant if not isinstance(self._variant, StandardAttention) else None
        )
        self._params_shape = (
            tuple(self._params_torch.shape) if self._has_params else None
        )

        self._compiled_kernel = _compile_mla_kernel(
            torch_dtype=self._q_dtype,
            torch_out_dtype=self._o_dtype,
            page_size=self._page_size,
            kv_lora_rank=self._kv_lora_rank,
            qk_rope_head_dim=self._qk_rope_head_dim,
            is_persistent=self._is_persistent,
            is_var_seq=self._is_var_seq,
            is_var_split_kv=self._is_var_split_kv,
            skip_correction_threshold=self._skip_correction_threshold,
            is_workspace_size_zero=False,
            enable_pdl=self._enable_pdl,
            variant=self._cache_variant,
            params_shape=self._params_shape,
        )

    def _validate_run_inputs(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        out: Optional[torch.Tensor],
    ) -> None:
        """Check that run() inputs are consistent with the plan() configuration."""
        expected_D = self._kv_lora_rank + self._qk_rope_head_dim
        if q.shape[-1] != expected_D:
            raise ValueError(
                f"q.shape[-1]={q.shape[-1]} does not match the planned "
                f"kv_lora_rank + qk_rope_head_dim = {expected_D}"
            )
        if q.dtype != self._q_dtype:
            raise ValueError(
                f"q.dtype={q.dtype} does not match the planned q_dtype={self._q_dtype}"
            )
        if kv_cache.dtype != self._q_dtype:
            raise ValueError(
                f"kv_cache.dtype={kv_cache.dtype} does not match the planned "
                f"q_dtype={self._q_dtype}"
            )
        if out is not None and out.dtype != self._o_dtype:
            raise ValueError(
                f"out.dtype={out.dtype} does not match the planned "
                f"out_dtype={self._o_dtype}"
            )

    @flashinfer_api
    def run(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        max_seq_len: int,
        softmax_scale: float,
        output_scale: float = 1.0,
        out: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run the MLA decode kernel.

        Parameters
        ----------
        q : torch.Tensor
            [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim.
        kv_cache : torch.Tensor
            [num_pages, page_size, D_total] (3D) or [num_pages, 1, page_size, D_total] (4D).
        block_tables : torch.Tensor
            [B, max_pages] page table indices.
        seq_lens : torch.Tensor
            [B] per-request KV sequence lengths.
        max_seq_len : int
            Maximum sequence length across the batch.
        softmax_scale : float
            Scale factor for QK^T before softmax.
        output_scale : float
            Scale factor applied to the output.
        out : Optional[torch.Tensor]
            Pre-allocated output [B, q_len, H, kv_lora_rank].

        Returns
        -------
        torch.Tensor
            Output tensor [B, q_len, H, kv_lora_rank].
        """
        if self._compiled_kernel is None:
            raise RuntimeError("Call plan() before run().")

        self._validate_run_inputs(q, kv_cache, block_tables, seq_lens, out)

        B, q_len, H, D_qk = q.shape

        # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
        if kv_cache.dim() == 4:
            if kv_cache.shape[1] != 1:
                raise ValueError(
                    f"Expected 4D kv_cache shape [num_pages, 1, page_size, D], "
                    f"got {tuple(kv_cache.shape)}"
                )
            kv_cache = kv_cache.squeeze(1)
        elif kv_cache.dim() != 3:
            raise ValueError(f"kv_cache must be 3D or 4D, got ndim={kv_cache.dim()}")

        # Split query into latent and rope components
        q_latent_k = q[..., : self._kv_lora_rank]
        q_rope_k = q[..., self._kv_lora_rank :]

        # KV cache slices
        c_latent_k = kv_cache[:, :, : self._kv_lora_rank]
        c_rope_k = kv_cache[:, :, self._kv_lora_rank :]

        page_table_k = block_tables

        # Compute split_kv and workspace size
        max_active_blocks = get_num_sm(q.device)
        split_kv, workspace_size = _get_split_kv_and_workspace_size(
            B, q_len, H, self._kv_lora_rank, max_active_blocks
        )

        if H < 128 and split_kv != 1:
            raise ValueError(
                f"num_heads={H} < 128 requires split_kv==1, got split_kv={split_kv}"
            )

        # Prepare workspace
        is_workspace_size_zero = workspace_size == 0
        if is_workspace_size_zero:
            workspace_bytes = None
        else:
            if self._workspace_buffer.numel() < workspace_size:
                raise ValueError(
                    f"workspace_buffer too small: {self._workspace_buffer.numel()} bytes, "
                    f"need {workspace_size} bytes"
                )
            workspace_bytes = self._workspace_buffer[:workspace_size]

        # Re-compile if workspace-zero-ness changed from what was planned
        compiled_kernel = self._compiled_kernel
        if is_workspace_size_zero:
            compiled_kernel = _compile_mla_kernel(
                torch_dtype=self._q_dtype,
                torch_out_dtype=self._o_dtype,
                page_size=self._page_size,
                kv_lora_rank=self._kv_lora_rank,
                qk_rope_head_dim=self._qk_rope_head_dim,
                is_persistent=self._is_persistent,
                is_var_seq=self._is_var_seq,
                is_var_split_kv=self._is_var_split_kv,
                skip_correction_threshold=self._skip_correction_threshold,
                is_workspace_size_zero=True,
                enable_pdl=self._enable_pdl,
                variant=self._cache_variant,
                params_shape=self._params_shape,
            )

        # Output buffer
        if out is None:
            out = torch.empty(
                (B, q_len, H, self._kv_lora_rank),
                dtype=self._o_dtype,
                device=q.device,
            )
        o_k = out

        # LSE buffer
        lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=q.device)

        # cache_seqs: per-batch sequence lengths
        cache_seqs = (
            seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)
        )

        block_split_kvs = None

        compiled_kernel(
            q_latent_k,
            q_rope_k,
            c_latent_k,
            c_rope_k,
            page_table_k,
            o_k,
            lse_k,
            workspace_bytes,
            Int32(split_kv),
            cache_seqs,
            block_split_kvs,
            Float32(softmax_scale),
            Float32(output_scale),
            self._params_torch if self._has_params else None,
        )

        return out


# ---------------------------------------------------------------------------
# Standalone function — drop-in replacement for the original integration layer
# ---------------------------------------------------------------------------


def cute_dsl_mla_decode(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    softmax_scale: float,
    output_scale: float = 1.0,
    out: Optional[torch.Tensor] = None,
    out_dtype: Optional[torch.dtype] = None,
    is_var_seq: bool = True,
    enable_pdl: Optional[bool] = None,
) -> torch.Tensor:
    """CuTe DSL MLA decode kernel for Blackwell SM100 (modular variant).

    Parameters
    ----------
    query : torch.Tensor
        [B, q_len, H, D_qk] where D_qk = kv_lora_rank + qk_rope_head_dim
    kv_cache : torch.Tensor
        [num_pages, page_size, D_ckv + D_kpe] (3D) or [num_pages, 1, page_size, D_ckv + D_kpe] (4D)
    workspace_buffer : torch.Tensor
        Pre-allocated workspace buffer (int8). Required size depends on batch size
        and split_kv (auto-computed from B, q_len, and number of SMs):

        - Formula: ``B * H * q_len * split_kv * (kv_lora_rank + 1) * 4`` bytes
          (0 when split_kv == 1, which happens when B >= num_SMs / 2)
        - Typical max: ~18 MB on a 148-SM GPU (e.g. B=4..8, H=128, D=512)
        - Safe default: 128 MB covers all realistic configurations
    kv_lora_rank : int
        Latent dimension (e.g. 512).
    qk_rope_head_dim : int
        RoPE dimension (e.g. 64).
    block_tables : torch.Tensor
        [B, max_pages] — page table indices.
    seq_lens : torch.Tensor
        [B] — per-request KV sequence lengths.
    max_seq_len : int
        Maximum sequence length across the batch.
    softmax_scale : float
        Scale factor for QK^T before softmax.
    output_scale : float
        Scale factor applied to the output.
    out : Optional[torch.Tensor]
        Pre-allocated output tensor [B, q_len, H, kv_lora_rank].
    out_dtype : Optional[torch.dtype]
        Output data type. If None, defaults to same as input dtype.
    is_var_seq : bool
        Whether the sequence length is variable across the batch.
    enable_pdl : Optional[bool], default=None
        Whether to enable Programmatic Dependent Launch (PDL).
        If None, auto-detects based on device capability.

    Returns
    -------
    torch.Tensor
        Output tensor [B, q_len, H, kv_lora_rank].
    """
    supported_dtypes = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}
    assert query.dtype in supported_dtypes, (
        f"cute_dsl_mla_decode only supports {supported_dtypes}, got {query.dtype}"
    )
    assert kv_cache.dtype == query.dtype, (
        f"kv_cache dtype {kv_cache.dtype} must match query dtype {query.dtype}"
    )
    B, q_len, H, D_qk = query.shape
    assert D_qk == kv_lora_rank + qk_rope_head_dim

    q_dtype = query.dtype
    if out is not None:
        o_dtype = out.dtype
    elif out_dtype is not None:
        o_dtype = out_dtype
    elif q_dtype == torch.float8_e4m3fn:
        o_dtype = torch.bfloat16
    else:
        o_dtype = q_dtype

    # Handle 3D vs 4D kv_cache: normalize to 3D [num_pages, page_size, D_total]
    if kv_cache.dim() == 4:
        if kv_cache.shape[1] != 1:
            raise ValueError(
                f"Expected 4D kv_cache shape [num_pages, 1, page_size, D], "
                f"got {tuple(kv_cache.shape)}"
            )
        kv_cache = kv_cache.squeeze(1)
    elif kv_cache.dim() != 3:
        raise ValueError(f"kv_cache must be 3D or 4D, got ndim={kv_cache.dim()}")
    page_size = kv_cache.shape[1]

    # Split query into latent and rope components
    q_latent_k = query[..., :kv_lora_rank]
    q_rope_k = query[..., kv_lora_rank:]

    # KV cache slices
    c_latent_k = kv_cache[:, :, :kv_lora_rank]
    c_rope_k = kv_cache[:, :, kv_lora_rank:]

    page_table_k = block_tables

    # Runtime validation
    if max_seq_len <= 0:
        raise ValueError(f"max_seq_len must be > 0, got {max_seq_len}")
    if H < 128 and H != 1:
        raise ValueError(
            f"cute_dsl_mla_decode requires num_heads >= 128 (or 1 for reduction), got {H}"
        )

    # Cached split_kv and workspace_size computation
    max_active_blocks = get_num_sm(query.device)
    split_kv, workspace_size = _get_split_kv_and_workspace_size(
        B, q_len, H, kv_lora_rank, max_active_blocks
    )

    if H < 128 and split_kv != 1:
        raise ValueError(
            f"cute_dsl_mla_decode: num_heads={H} < 128 requires split_kv==1, "
            f"got split_kv={split_kv}"
        )

    # Prepare workspace
    assert workspace_buffer.dtype == torch.int8, (
        f"workspace_buffer must be torch.int8, got {workspace_buffer.dtype}"
    )
    assert workspace_buffer.numel() >= workspace_size, (
        f"workspace_buffer too small: {workspace_buffer.numel()} bytes, "
        f"need {workspace_size} bytes"
    )
    is_workspace_size_zero = workspace_size == 0
    if is_workspace_size_zero:
        workspace_bytes = None
    else:
        workspace_bytes = workspace_buffer[:workspace_size]

    # Output buffer
    if out is not None:
        o_k = out
    else:
        o_k = torch.empty(
            (B, q_len, H, kv_lora_rank), dtype=o_dtype, device=query.device
        )

    # LSE buffer
    lse_k = torch.empty((B, q_len, H), dtype=torch.float32, device=query.device)

    # cache_seqs: per-batch sequence lengths
    cache_seqs = seq_lens if seq_lens.dtype == torch.int32 else seq_lens.to(torch.int32)

    is_var_split_kv = False
    block_split_kvs = None
    skip_correction_threshold = 0.0

    is_persistent = not is_var_seq

    # Validate configuration (cached, negligible overhead after first call)
    _check_can_implement(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        num_heads=H,
        seq_len_q=q_len,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
    )

    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl

    # Get compiled kernel (cached after first compile)
    compiled_kernel = _compile_mla_kernel(
        torch_dtype=q_dtype,
        torch_out_dtype=o_dtype,
        page_size=page_size,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        is_persistent=is_persistent,
        is_var_seq=is_var_seq,
        is_var_split_kv=is_var_split_kv,
        skip_correction_threshold=skip_correction_threshold,
        is_workspace_size_zero=is_workspace_size_zero,
        enable_pdl=enable_pdl,
    )

    # Call the kernel
    compiled_kernel(
        q_latent_k,
        q_rope_k,
        c_latent_k,
        c_rope_k,
        page_table_k,
        o_k,
        lse_k,
        workspace_bytes,
        Int32(split_kv),
        cache_seqs,
        block_split_kvs,
        Float32(softmax_scale),
        Float32(output_scale),
        None,  # params_in (no variant in standalone function)
    )

    if out is not None:
        return out

    return o_k
