"""
Copyright (c) 2024 by FlashInfer team.

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

import math
from typing import Optional, Tuple, Union

import torch

from .api_logging import flashinfer_api
from .trace.templates.attention import (
    block_sparse_attention_run_trace,
    variable_block_sparse_attention_run_trace,
)
from .decode import get_batch_decode_module
from .prefill import _compute_page_mask_indptr, get_batch_prefill_module
from .quantization import segment_packbits
from .utils import (
    MaskMode,
    PosEncodingMode,
    TensorLayout,
    _check_pos_encoding_mode,
    check_shape_dtype_device,
    _get_cache_alibi_slopes_buf,
    canonicalize_torch_dtype,
    determine_attention_backend,
    device_support_pdl,
    get_compute_capability,
    is_float8,
)


def _bsr_to_vsa_index(
    indptr: torch.Tensor,
    indices: torch.Tensor,
    MB: int,
    NB: int,
    num_heads: int,
    device: torch.device,
    non_blocking: bool = True,
    qhead_per_kvhead: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert head-independent BSR (indptr/indices) to VSA q2k_index / q2k_num tensors.

    Returns
    -------
    q2k_index : torch.Tensor  shape ``[1, num_heads, MB * qhead_per_kvhead, NB]``, dtype int32
        For each q_block, the list of attended KV-block indices, padded with -1.
        The same pattern is broadcast across all heads and tiled qhead_per_kvhead times
        in the m_block dimension for GQA pack_gqa mode.
    q2k_num : torch.Tensor  shape ``[1, num_heads, MB * qhead_per_kvhead]``, dtype int32
        Number of attended KV-blocks per Q-block.
    """
    indptr_cpu = indptr.cpu()
    indices_cpu = indices.cpu().to(torch.int32)

    if indices_cpu.numel() and (
        int(indices_cpu.min()) < 0 or int(indices_cpu.max()) >= NB
    ):
        raise ValueError(
            f"BSR indices out of range [0, {NB}): "
            f"got min={int(indices_cpu.min())}, max={int(indices_cpu.max())}"
        )

    q2k_index_flat = torch.full((MB, NB), -1, dtype=torch.int32)
    q2k_num_flat = (indptr_cpu[1:] - indptr_cpu[:-1]).to(torch.int32)

    for i in range(MB):
        s = int(indptr_cpu[i].item())
        e = int(indptr_cpu[i + 1].item())
        if e > s:
            q2k_index_flat[i, : e - s] = indices_cpu[s:e]

    # With pack_gqa, packed m_block b maps to original Q block b // qhead_per_kvhead.
    # repeat_interleave gives [blk0]*R, [blk1]*R, ... so that packed m_block b uses
    # the same KV list as original Q block b // qhead_per_kvhead.
    if qhead_per_kvhead > 1:
        q2k_index_flat = q2k_index_flat.repeat_interleave(
            qhead_per_kvhead, dim=0
        )  # [MB * qhead_per_kvhead, NB]
        q2k_num_flat = q2k_num_flat.repeat_interleave(
            qhead_per_kvhead
        )  # [MB * qhead_per_kvhead]

    # Broadcast the same pattern to every KV head: [1, H, MB_packed, NB]
    q2k_index = (
        q2k_index_flat.unsqueeze(0)
        .unsqueeze(0)
        .expand(1, num_heads, -1, -1)
        .contiguous()
    )
    q2k_num = (
        q2k_num_flat.unsqueeze(0).unsqueeze(0).expand(1, num_heads, -1).contiguous()
    )

    return (
        q2k_index.to(device, non_blocking=non_blocking),
        q2k_num.to(device, non_blocking=non_blocking),
    )


def _block_mask_to_vsa_index(
    block_mask: torch.Tensor,
    device: torch.device,
    non_blocking: bool = True,
    qhead_per_kvhead: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert a per-head boolean block mask to VSA q2k_index / q2k_num tensors.

    Parameters
    ----------
    block_mask : torch.Tensor  shape ``[H, MB, NB]``, dtype bool
        Per-KV-head block-level attention mask.
    qhead_per_kvhead : int
        Number of QO heads per KV head (for GQA pack_gqa mode).  The MB-block
        pattern is tiled qhead_per_kvhead times in the m_block dimension.

    Returns
    -------
    q2k_index : torch.Tensor  shape ``[1, H, MB * qhead_per_kvhead, max_nnz]``, dtype int32
        Per-head attended KV-block indices, padded with -1.
    q2k_num : torch.Tensor  shape ``[1, H, MB * qhead_per_kvhead]``, dtype int32
        Number of attended KV-blocks per (head, Q-block).
    """
    H, MB, NB = block_mask.shape
    block_mask_cpu = block_mask.cpu()

    q2k_num = block_mask_cpu.sum(dim=-1).to(torch.int32)  # [H, MB]
    max_nnz = int(q2k_num.max().item())
    if max_nnz == 0:
        max_nnz = 1  # avoid zero-size tensor

    # argsort with stable=True puts True (1) entries first along the NB dim
    sorted_idx = torch.argsort(~block_mask_cpu, dim=-1, stable=True)[:, :, :max_nnz]

    # mask out positions beyond each row's actual count
    valid = torch.arange(max_nnz).unsqueeze(0).unsqueeze(0) < q2k_num.unsqueeze(-1)
    q2k_index = torch.where(valid, sorted_idx, torch.full_like(sorted_idx, -1)).to(
        torch.int32
    )
    # q2k_index: [H, MB, max_nnz],  q2k_num: [H, MB]

    # Tile m_block dimension for pack_gqa: packed m_block b → original Q block b // qhead_per_kvhead.
    if qhead_per_kvhead > 1:
        q2k_index = q2k_index.repeat_interleave(
            qhead_per_kvhead, dim=1
        )  # [H, MB * qhead_per_kvhead, max_nnz]
        q2k_num = q2k_num.repeat_interleave(
            qhead_per_kvhead, dim=1
        )  # [H, MB * qhead_per_kvhead]

    return (
        q2k_index.unsqueeze(0).to(device, non_blocking=non_blocking),
        q2k_num.unsqueeze(0).to(device, non_blocking=non_blocking),
    )


def convert_bsr_mask_layout(mask: torch.Tensor, indptr: torch.Tensor) -> torch.Tensor:
    r"""Convert mask from BSR data layout to flashinfer's flattened mask layout.

    Parameters
    ----------
    mask : torch.Tensor
        A boolean mask tensor with shape ``(nnz, R, C)``.
    indptr : torch.Tensor
        The indptr tensor in BSR format.

    Returns
    -------
    flattened_mask : torch.Tensor
        A flattenedd mask tensor with shape ``(nnz * R * C,)``.
    """
    nnz, R, C = mask.shape
    MB = len(indptr) - 1
    mask_flashinfer = torch.empty((nnz * R * C,), dtype=mask.dtype, device=mask.device)
    for i in range(MB):
        mask_flashinfer[indptr[i] * R * C : indptr[i + 1] * R * C] = (
            mask[indptr[i] : indptr[i + 1]].transpose(0, 1).reshape(-1)
        )
    return mask_flashinfer


# Backward-compatible aliases: old marketing names → canonical arch-tagged names.
_BACKEND_ALIASES: dict = {
    "vsa_blackwell": "vsa_sm100_blk128",
    "vsa_blackwell_blk64": "vsa_sm100_blk64",
}


def _vsa_common_checks(
    backend: str,
    R: int,
    C: int,
    M: int,
    N: int,
    num_qo_heads: int,
    num_kv_heads: int,
    mask,
    packed_mask,
    causal: bool,
    pos_encoding_mode: str,
    logits_soft_cap,
) -> None:
    """Validate the arguments that are identical across all VSA backends."""
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be a multiple of num_kv_heads ({num_kv_heads})"
        )
    if M % R != 0:
        raise ValueError(f"M={M} must be divisible by block size R={R}")
    if N % C != 0:
        raise ValueError(f"N={N} must be divisible by block size C={C}")
    if mask is not None or packed_mask is not None:
        raise ValueError(
            f"{backend} backend does not support per-element block masks "
            "(mask / packed_mask).  Only block-level sparsity via indptr/indices "
            "or block_mask is supported."
        )
    if causal:
        raise ValueError(f"{backend} backend does not support causal masking.")
    if pos_encoding_mode != "NONE":
        raise ValueError(
            f"{backend} backend only supports pos_encoding_mode='NONE' "
            f"(got '{pos_encoding_mode}')."
        )
    if logits_soft_cap is not None and logits_soft_cap > 0:
        raise ValueError(f"{backend} backend does not support logits_soft_cap.")


def _vsa_run_core(
    fwd_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    vsa_q2k_index: torch.Tensor,
    vsa_q2k_num: torch.Tensor,
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
):
    """Shared NHD→BSHD dispatch, kernel call, and BSHD→NHD reshape for all VSA backends."""
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    q_b, k_b, v_b = _vsa_reshape_qkv(q, k, v)
    o_bsa, lse_bsa = fwd_fn(
        q_b,
        k_b,
        v_b,
        q2k_block_index=vsa_q2k_index,
        block_sparse_num=1,  # ignored when q2k_block_nums is provided
        block_sizes=None,
        q2k_block_nums=vsa_q2k_num,
        softmax_scale=sm_scale,
        return_lse=True,
    )

    return _vsa_finish_output(o_bsa, lse_bsa, out, lse, return_lse)


def _vsa_reshape_qkv(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    """NHD (no batch dim) -> BSHD with an implicit batch size of 1."""
    return (
        q.unsqueeze(0).contiguous(),
        k.unsqueeze(0).contiguous(),
        v.unsqueeze(0).contiguous(),
    )


def _vsa_finish_output(o_bsa, lse_bsa, out, lse, return_lse):
    output = o_bsa[0]  # [1, M, H, D] -> [M, H, D]
    if out is not None:
        check_shape_dtype_device(out, output.shape, output.dtype, output.device, "out")
        out.copy_(output)
        output = out

    if return_lse:
        lse_out = lse_bsa[0].permute(1, 0).contiguous()  # [1, H, M] -> [M, H]
        if lse is not None:
            check_shape_dtype_device(
                lse, lse_out.shape, lse_out.dtype, lse_out.device, "lse"
            )
            lse.copy_(lse_out)
            lse_out = lse
        return output, lse_out
    return output


def _vsa_run_core_blk64(
    fwd_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    vsa_q2k_index: torch.Tensor,
    vsa_q2k_num: torch.Tensor,
    sm_scale: Optional[float],
    out: Optional[torch.Tensor],
    lse: Optional[torch.Tensor],
    return_lse: bool,
    kv_splits: Optional[Union[int, str]],
    use_clc: Optional[bool],
    q_scale: Optional[torch.Tensor],
    k_scale: Optional[torch.Tensor],
    v_scale: Optional[torch.Tensor],
    sage_fp8_block_sparse_num: Optional[int],
):
    """blk64-specific variant of :func:`_vsa_run_core` with kv_splits/use_clc/Sage-FP8 passthrough.

    Kept separate from ``_vsa_run_core`` (used by the blk128/sm120_blk64
    backends) rather than adding blk64-only kwargs there, so those backends'
    call sites do not carry always-None dead parameters.
    """
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(q.size(-1))

    q_b, k_b, v_b = _vsa_reshape_qkv(q, k, v)

    is_sage_fp8 = q_scale is not None
    if is_sage_fp8:
        # Sage FP8 requires a uniform (dense) top-k: the underlying kernel
        # only accepts q2k_block_nums=None with a fixed block_sparse_num.
        # The uniform value is validated and cached in plan() (which already
        # syncs on vsa_q2k_num once) so run() never has to sync on it.
        block_nums_arg = None
        block_sparse_num_arg = sage_fp8_block_sparse_num
    else:
        block_nums_arg = vsa_q2k_num
        block_sparse_num_arg = 1  # ignored when q2k_block_nums is provided

    o_bsa, lse_bsa = fwd_fn(
        q_b,
        k_b,
        v_b,
        q2k_block_index=vsa_q2k_index,
        block_sparse_num=block_sparse_num_arg,
        block_sizes=None,
        q2k_block_nums=block_nums_arg,
        softmax_scale=sm_scale,
        return_lse=True,
        kv_splits=1 if kv_splits is None else kv_splits,
        use_clc=use_clc,
        q_scale=q_scale,
        k_scale=k_scale,
        v_scale=v_scale,
    )

    return _vsa_finish_output(o_bsa, lse_bsa, out, lse, return_lse)


class BlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    The definition of block sparse matrix can be found at
    `bsr_matrix <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.bsr_matrix.html>`_
    in SciPy.

    This API supports any block size ``(R, C)``.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 32
    >>> num_kv_heads = 8
    >>> head_dim = 128
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> bsr_wrapper = flashinfer.BlockSparseAttentionWrapper(workspace_buffer)
    >>> # sparse mask: [[0, 0, 1], [1, 0, 1], [0, 1, 1]]
    >>> M = 3
    >>> N = 3
    >>> indptr = torch.tensor([0, 1, 3, 5], dtype=torch.int32, device="cuda:0")
    >>> indices = torch.tensor([2, 0, 2, 1, 2], dtype=torch.int32, device="cuda:0")
    >>> bsr_wrapper.plan(
    ...     indptr,
    ...     indices,
    ...     M,
    ...     N,
    ...     1, # R(block_rows)=1
    ...     1, # C(block_columns)=1
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((M, num_qo_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((N, num_kv_heads, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = bsr_wrapper.run(q, k, v)
    >>> # use dense implementation with attention mask for comparison
    >>> mask = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.bool, device="cuda:0")
    >>> o_ref = flashinfer.single_prefill_with_kv_cache(q, k, v, custom_mask=mask)
    >>> torch.allclose(o, o_ref)
    True
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`BlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2``/``fa3`` or ``cake``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._backend = _BACKEND_ALIASES.get(backend, backend)
        if self._backend == "cake":
            # Cake consumes the caller's direct VSA metadata and never invokes
            # the generic sparse planner. Avoid allocating its per-wrapper 8 MiB
            # device/host workspaces: video diffusion creates one wrapper per
            # transformer layer.
            self._int_workspace_buffer = torch.empty(
                (0,), dtype=torch.uint8, device=self.device
            )
            self._kv_lens_buffer = torch.empty(
                (0,), dtype=torch.int32, device=self.device
            )
            self._pin_memory_int_workspace_buffer = torch.empty(
                (0,), dtype=torch.uint8, device="cpu"
            )
        else:
            self._int_workspace_buffer = torch.empty(
                (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
            )
            self._kv_lens_buffer = torch.empty(
                (32768,), dtype=torch.int32, device=self.device
            )
            self._pin_memory_int_workspace_buffer = torch.empty(
                self._int_workspace_buffer.shape,
                dtype=torch.uint8,
                pin_memory=True,
                device="cpu",
            )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._packed_mask_buf: Optional[torch.Tensor] = None
        self._mask_indptr_buf: Optional[torch.Tensor] = None
        self.R: Optional[int] = None
        self.C: Optional[int] = None
        self.M: Optional[int] = None
        self.N: Optional[int] = None
        self._cake_vsa_plan: Optional[dict] = None

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        indptr: Optional[torch.Tensor],
        indices: Optional[torch.Tensor],
        M: int,
        N: int,
        R: int,
        C: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        mask: Optional[torch.Tensor] = None,
        packed_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
        o_data_type: Union[str, torch.dtype] = "float16",
        non_blocking: bool = True,
        block_mask: Optional[torch.Tensor] = None,
        kv_block_lens: Optional[torch.Tensor] = None,
        q2k_indices: Optional[torch.Tensor] = None,
        q2k_num: Optional[torch.Tensor] = None,
        kv_splits: Optional[Union[int, str]] = None,
        use_clc: Optional[bool] = None,
        q_scale: Optional[torch.Tensor] = None,
        k_scale: Optional[torch.Tensor] = None,
        v_scale: Optional[torch.Tensor] = None,
    ) -> None:
        r"""Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        indptr : torch.Tensor, optional
            The block index pointer of the block-sparse matrix on row dimension, shape ``(MB + 1,)``,
            where ``MB`` is the number of blocks in the row dimension.
            Required for all backends except ``cake``, ``vsa_sm100_blk128``,
            ``vsa_sm100_blk64``, and ``vsa_sm120_blk64`` when ``block_mask`` is provided.
        indices: torch.Tensor, optional
            The block indices of the block-sparse matrix on column dimension, shape ``(nnz,)``, where
            ``nnz`` is the number of non-zero blocks. The elements in ``indices`` array should be less then ``NB``:
            the number of blocks in the column dimension.
            Required for all backends except ``cake``, ``vsa_sm100_blk128``,
            ``vsa_sm100_blk64``, and ``vsa_sm120_blk64`` when ``block_mask`` is provided.
        M : int
            The number of rows of the block-sparse matrix, ``MB = ceil_div(M, R)``.
        N : int
            The number of columns of the block-sparse matrix, ``NB = N // C``, ``N`` should be divisible by ``C``.
        R : int
            The number of rows in each block.
        C : int
            The number of columns in each block.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor.
        head_dim : int
            The dimension of each head.
        mask : torch.Tensor, optional
            The mask tensor with shape ``(nnz, R, C,)``, where nnz is the number of non-zero blocks.
            If every block is full, then we don't need to provide the mask tensor.
        packed_mask : torch.Tensor, optional
            The 1D packed mask tensor, if provided, the :attr:`custom_mask` will be ignored.
            The packed mask tensor is generated by :func:`flashinfer.quantization.packbits`.
        causal : bool
            Whether to apply causal mask to the attention matrix.
            This is only effective when :attr:`custom_mask` is not provided in
            :meth:`plan`.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        q_data_type : str, optional
            The data type of the query tensor.
        kv_data_type : Optional[Union[str, torch.dtype]]
            The data type of the key/value tensor. If None, will be set to :attr:`q_data_type`.
        o_data_type : str, optional
            The data type of the output tensor. Default is ``half``. As output dtype cannot
            be inferred by input dtype in quantization
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        block_mask : torch.Tensor, optional
            Per-head block-level attention mask, dtype bool.  Shape may be either
            ``(num_qo_heads, MB, NB)`` or ``(num_kv_heads, MB, NB)``.
            ``block_mask[h, i, j] = True`` means the Q-block ``i`` attends to KV-block ``j``
            for head ``h``.  For GQA (``num_qo_heads > num_kv_heads``), when providing
            ``(num_qo_heads, MB, NB)``, the first QO-head from each KV-head group is used
            (sparsity must be the same across QO-heads that share a KV-head).
            Supported by the ``cake``, ``vsa_sm100_blk128``, ``vsa_sm100_blk64``,
            and ``vsa_sm120_blk64`` backends.  When provided,
            ``indptr``/``indices`` are not required and will be ignored.
        kv_block_lens : torch.Tensor, optional
            Number of valid tokens in every KV block, shape ``(NB,)``. Entries
            must be in ``[1, C]``. Supported by the ``cake`` block-64 route;
            when omitted, every block is treated as having ``C`` valid tokens.
        q2k_indices : torch.Tensor, optional
            Direct per-head KV-block selections, contiguous int32 with shape
            ``(num_qo_heads, MB, topk)``. Supported by the ``cake`` block-64
            route and mutually exclusive with ``block_mask`` and BSR metadata.
        q2k_num : torch.Tensor, optional
            Number of valid entries in each direct selection row, contiguous
            int32 with shape ``(num_qo_heads, MB)``. When omitted, every direct
            row uses the full ``topk`` dimension.
        kv_splits : Optional[Union[int, str]]
            Number of KV splits for the split-KV combine path, or ``"auto"`` to pick a
            split count from the sparsity heuristics. Only supported for the
            ``vsa_sm100_blk64`` backend; must be ``None`` for all other backends.
            ``None`` (default) disables splitting, equivalent to passing ``1``
            explicitly. Pass ``"auto"`` to select the split count automatically via
            a sparsity heuristic.
        use_clc : Optional[bool]
            Override the SM100 blk64 scheduler: ``True`` forces the CLC persistent
            scheduler, ``False`` forces the static scheduler, ``None`` (default) uses
            the shape-based heuristic. Only supported for the ``vsa_sm100_blk64`` backend.
        q_scale : torch.Tensor, optional
            Sage FP8 quantization scale for ``q``, shape ``(1, num_qo_heads, seqlen_q)``,
            float32. Only supported for the ``vsa_sm100_blk64`` backend, and only when
            ``q``/``k``/``v`` are ``float8_e4m3fn``. Must be provided together with
            ``k_scale``/``v_scale``, or not at all. The Sage FP8 path additionally
            requires ``batch_size == 1``, ``num_qo_heads in (4, 8)``, and dense
            (non-variable) block sparsity -- see :func:`bsa_attn_sm100_blk64_fwd`.
        k_scale : torch.Tensor, optional
            Sage FP8 quantization scale for ``k``, shape
            ``(1, num_qo_heads, ceil(seqlen_k / 16))``, float32. See ``q_scale``.
        v_scale : torch.Tensor, optional
            Sage FP8 quantization scale for ``v``, shape ``(num_qo_heads, head_dim)``,
            float32. See ``q_scale``.

        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.

        .. note::
            The ``vsa_sm100_blk64`` backend does not support GQA/MQA: it has no
            KV-head mapping and requires ``num_kv_heads == num_qo_heads``.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = canonicalize_torch_dtype(o_data_type)

        if self._backend != "vsa_sm100_blk64" and (
            kv_splits is not None
            or use_clc is not None
            or q_scale is not None
            or k_scale is not None
            or v_scale is not None
        ):
            raise ValueError(
                "kv_splits/use_clc/q_scale/k_scale/v_scale are only supported "
                f"for backend='vsa_sm100_blk64', got backend={self._backend!r}"
            )

        if self._backend == "cake":
            from flashinfer.cake_vsa import plan_cake_vsa

            _vsa_common_checks(
                "cake",
                R,
                C,
                M,
                N,
                num_qo_heads,
                num_kv_heads,
                mask,
                packed_mask,
                causal,
                pos_encoding_mode,
                logits_soft_cap,
            )
            if kv_data_type != q_data_type:
                raise ValueError("cake backend requires matching Q/K/V dtypes")
            self._cake_vsa_plan = plan_cake_vsa(
                indptr,
                indices,
                block_mask,
                kv_block_lens,
                q2k_indices,
                q2k_num,
                M=M,
                N=N,
                R=R,
                C=C,
                num_qo_heads=num_qo_heads,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                q_data_type=q_data_type,
                sm_scale=sm_scale,
                device=self.device,
            )
            self.M = M
            self.N = N
            self.R = R
            self.C = C
            self._sm_scale = sm_scale
            return

        # ---- cuTile backend (pure cuda.tile Python kernel) ------------------------
        # No C++ module to JIT: just stash the BSR plan state. run() reconstructs a
        # dense block table from (indptr/indices) and treats each block-row as one
        # variable-length prefill "batch" with page_size == C.
        if self._backend == "cutile":
            if indptr is None or indices is None:
                raise ValueError(
                    "cuTile block-sparse backend requires indptr and indices."
                )
            if N % C != 0:
                raise ValueError(
                    f"cuTile block-sparse backend requires N % C == 0 (N={N}, C={C})."
                )
            if mask is not None or packed_mask is not None:
                raise NotImplementedError(
                    "cuTile block-sparse backend does not support per-element "
                    "intra-block masks (mask/packed_mask)."
                )
            if logits_soft_cap is not None and logits_soft_cap > 0:
                raise NotImplementedError(
                    "cuTile block-sparse backend does not support logits_soft_cap."
                )
            if pos_encoding_mode != "NONE":
                raise NotImplementedError(
                    "cuTile block-sparse backend does not apply position encoding "
                    "(pos_encoding_mode must be 'NONE')."
                )
            if M % R != 0:
                # run() forces each block-row batch to exactly R query rows; a
                # non-multiple M would slice past the query buffer -> OOB.
                raise ValueError(
                    f"cuTile block-sparse backend requires M % R == 0 (M={M}, R={R})."
                )
            if C < 16:
                # Prefill autotune's minimum BLOCK_N is 16 (32 on SM90); a smaller
                # C yields an empty search space -> opaque exhaustive_search error.
                raise ValueError(
                    "cuTile block-sparse backend requires C >= 16 (the minimum "
                    f"prefill BLOCK_N); got C={C}."
                )
            if causal:
                # The BSR->paged mapping gathers arbitrary column-blocks, so the
                # kernel's packed (gathered-block) position mask does NOT equal
                # global row/column causality. (The standalone prefill kernel
                # supports causal for contiguous paged/ragged inputs; only this
                # block-sparse mapping cannot express it correctly.)
                raise NotImplementedError(
                    "cuTile block-sparse backend does not support causal masking "
                    "under the block-sparse (gathered-block) mapping."
                )
            if num_qo_heads % num_kv_heads != 0:
                # run() maps each (block-row, kv-head) onto QUERY_GROUP_SIZE =
                # num_qo_heads // num_kv_heads query heads; a non-multiple would
                # silently drop the remainder heads.
                raise ValueError(
                    "cuTile block-sparse backend requires num_qo_heads % "
                    f"num_kv_heads == 0 (num_qo_heads={num_qo_heads}, "
                    f"num_kv_heads={num_kv_heads})."
                )
            num_col_blocks = N // C
            if indices.numel() > 0:
                # Each index selects a column-block gathered as a page; an
                # out-of-range value would issue an invalid page load.
                idx_min = int(indices.min().item())
                idx_max = int(indices.max().item())
                if idx_min < 0 or idx_max >= num_col_blocks:
                    raise ValueError(
                        "cuTile block-sparse backend requires all indices in "
                        f"[0, N // C) = [0, {num_col_blocks}); got "
                        f"[{idx_min}, {idx_max}]."
                    )
            self._R = R
            self._C = C
            self._M = M
            self._N = N
            self._num_qo_heads = num_qo_heads
            self._num_kv_heads = num_kv_heads
            self._head_dim = head_dim
            self._causal = causal
            self._sm_scale = sm_scale
            indptr = indptr.to(self.device, non_blocking=non_blocking)
            indices = indices.to(self.device, non_blocking=non_blocking)
            self._sparse_indptr = indptr
            self._sparse_indices = indices

            # Materialize the dense block table and the per-batch length/offset
            # arrays now (at plan time) so run() is CUDA-graph-capturable:
            # everything here derives only from (indptr, indices, R, C), which
            # are fixed at plan time. Reconstructing it in run() -- as an earlier
            # version did -- forced a host `.item()` sync on max_pages plus ~6
            # tensor allocations on every call, breaking graph capture (mirrors
            # the plan-time block-table build in decode.py's cuTile backend).
            num_block_rows = indptr.numel() - 1
            nnz_per_row = (indptr[1:] - indptr[:-1]).to(torch.int32)
            actual_seq_lens_q = torch.full(
                (num_block_rows,), R, dtype=torch.int32, device=self.device
            )
            max_pages = int(nnz_per_row.max().item()) if num_block_rows > 0 else 0
            block_tables = torch.zeros(
                (num_block_rows, max_pages), dtype=torch.int32, device=self.device
            )
            col = torch.arange(max_pages, device=self.device)
            valid = col[None, :] < nnz_per_row[:, None]
            block_tables[valid] = indices.to(torch.int32)
            actual_seq_offset = torch.nn.functional.pad(
                actual_seq_lens_q.cumsum(0), (1, 0)
            ).to(torch.int32)

            self._sparse_num_block_rows = num_block_rows
            self._sparse_actual_seq_lens_q = actual_seq_lens_q
            self._sparse_actual_seq_lens_kv = (nnz_per_row * C).to(torch.int32)
            self._sparse_block_tables = block_tables
            self._sparse_actual_seq_offset = actual_seq_offset
            return

        # ---- VSA blk128 backend (BSA CuTe-DSL kernel, SM100/SM103) ---------------
        if self._backend == "vsa_sm100_blk128":
            cc = get_compute_capability(self.device)
            arch = cc[0] * 10 + cc[1]
            if cc not in ((10, 0), (10, 3)):
                raise RuntimeError(
                    f"vsa_sm100_blk128 backend requires SM100/SM103, "
                    f"current device is SM{arch}"
                )
            # BSA blk128 kernel uses 128-token compute tiles; block index granularity = R = C = 128.
            if R != 128 or C != 128:
                raise ValueError(
                    f"vsa_sm100_blk128 backend requires R == C == 128 (got R={R}, C={C})"
                )
            if head_dim not in (64, 96, 128):
                raise ValueError(
                    f"vsa_sm100_blk128 backend requires head_dim in {{64, 96, 128}} (got {head_dim})"
                )
            _vsa_common_checks(
                "vsa_sm100_blk128",
                R,
                C,
                M,
                N,
                num_qo_heads,
                num_kv_heads,
                mask,
                packed_mask,
                causal,
                pos_encoding_mode,
                logits_soft_cap,
            )

            MB = M // R
            NB = N // C
            # blk128 uses pack_gqa when qhead_per_kvhead > 1: the tile scheduler
            # iterates over KV heads and qhead_per_kvhead * MB m_blocks, so the
            # block index head dimension is num_kv_heads and the m_block dimension
            # must be qhead_per_kvhead * MB.
            H_idx = num_kv_heads
            qhead_per_kvhead = num_qo_heads // num_kv_heads

            if block_mask is not None:
                # Per-head block_mask: accept both (num_qo_heads, MB, NB) and (num_kv_heads, MB, NB).
                # For GQA, reduce to num_kv_heads by taking the first QO head per KV-head group.
                # WARNING: sparsity must be identical for all QO heads within the same KV-head group.
                # Non-uniform masks across the group are silently ignored — only the first QO head's
                # mask is used. This is a kernel limitation of pack_gqa (blk128 schedules one block
                # index list per KV head); in true GQA VSA the compress stage produces per-QO-head
                # scores that may differ within a group.
                if (
                    block_mask.shape == (num_qo_heads, MB, NB)
                    and num_qo_heads != num_kv_heads
                ):
                    block_mask = block_mask[
                        ::qhead_per_kvhead
                    ]  # [num_kv_heads, MB, NB]
                if block_mask.shape != (H_idx, MB, NB):
                    raise ValueError(
                        f"block_mask must have shape (num_kv_heads={H_idx}, MB={MB}, NB={NB}) "
                        f"or (num_qo_heads={num_qo_heads}, MB={MB}, NB={NB}), "
                        f"got {tuple(block_mask.shape)}"
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _block_mask_to_vsa_index(
                    block_mask,
                    self.device,
                    non_blocking,
                    qhead_per_kvhead=qhead_per_kvhead,
                )
            else:
                # Head-independent BSR path: broadcast the same pattern across all KV heads.
                if indptr is None or indices is None:
                    raise ValueError(
                        "vsa_sm100_blk128 backend requires either block_mask or "
                        "(indptr, indices) to be provided."
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _bsr_to_vsa_index(
                    indptr,
                    indices,
                    MB,
                    NB,
                    H_idx,
                    self.device,
                    non_blocking,
                    qhead_per_kvhead=qhead_per_kvhead,
                )

            self.M = M
            self.N = N
            self.R = R
            self.C = C
            self._sm_scale = sm_scale
            return

        # ---- VSA blk64 backend (BSA CuTe-DSL kernel, SM100/SM103) -----------------
        if self._backend == "vsa_sm100_blk64":
            cc = get_compute_capability(self.device)
            arch = cc[0] * 10 + cc[1]
            if cc not in ((10, 0), (10, 3)):
                raise RuntimeError(
                    f"vsa_sm100_blk64 backend requires SM100/SM103, "
                    f"current device is SM{arch}"
                )
            # blk64 kernel uses 64-token compute tiles; block index granularity = R = C = 64.
            if R != 64 or C != 64:
                raise ValueError(
                    f"vsa_sm100_blk64 backend requires R == C == 64 (got R={R}, C={C})"
                )
            if head_dim != 128:
                raise ValueError(
                    f"vsa_sm100_blk64 backend requires head_dim=128 (got {head_dim})"
                )
            if kv_splits is not None and not isinstance(kv_splits, str):
                if not (1 <= int(kv_splits) <= 256):
                    raise ValueError(
                        f"vsa_sm100_blk64 kv_splits must be in [1, 256], got {kv_splits}"
                    )
            is_sage_fp8 = q_scale is not None
            if is_sage_fp8:
                if k_scale is None or v_scale is None:
                    raise ValueError(
                        "vsa_sm100_blk64 Sage FP8 requires q_scale/k_scale/v_scale "
                        "to all be provided together"
                    )
                if q_data_type != torch.float8_e4m3fn:
                    raise ValueError(
                        "vsa_sm100_blk64 backend requires q_data_type=float8_e4m3fn "
                        "when q_scale is provided"
                    )
                if num_qo_heads not in (4, 8):
                    raise ValueError(
                        "vsa_sm100_blk64 Sage FP8 requires num_qo_heads in (4, 8) "
                        f"(upstream kernel limit), got {num_qo_heads}"
                    )
            else:
                if k_scale is not None or v_scale is not None:
                    raise ValueError(
                        "vsa_sm100_blk64 Sage FP8 requires q_scale/k_scale/v_scale "
                        "to all be provided together"
                    )
                if q_data_type != torch.bfloat16:
                    raise ValueError(
                        "vsa_sm100_blk64 backend only supports bfloat16 inputs "
                        "unless q_scale/k_scale/v_scale (Sage FP8) are provided"
                    )
            # blk64 has no KV-head mapping: the launcher sizes K/V by the Q head
            # count, so num_qo_heads must equal num_kv_heads.
            _vsa_common_checks(
                "vsa_sm100_blk64",
                R,
                C,
                M,
                N,
                num_qo_heads,
                num_kv_heads,
                mask,
                packed_mask,
                causal,
                pos_encoding_mode,
                logits_soft_cap,
            )

            MB = M // R
            NB = N // C
            H = num_qo_heads

            if block_mask is not None:
                if block_mask.shape != (H, MB, NB):
                    raise ValueError(
                        f"block_mask must have shape (num_qo_heads={H}, MB={MB}, NB={NB}), "
                        f"got {tuple(block_mask.shape)}"
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _block_mask_to_vsa_index(
                    block_mask, self.device, non_blocking
                )
            else:
                if indptr is None or indices is None:
                    raise ValueError(
                        "vsa_sm100_blk64 backend requires either block_mask or "
                        "(indptr, indices) to be provided."
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _bsr_to_vsa_index(
                    indptr, indices, MB, NB, H, self.device, non_blocking
                )

            if self._vsa_q2k_num.min().item() == 0:
                raise ValueError(
                    "vsa_sm100_blk64 backend does not support empty sparse rows "
                    "(Q-blocks with zero KV blocks). All Q-blocks must attend to "
                    "at least one KV block."
                )
            # Note: the underlying blk64 CuTe-DSL kernel supports
            # allow_empty_block_nums=True, but FlashInfer keeps rejecting empty
            # sparse rows here to preserve existing behavior. Lifting this
            # restriction only requires removing the check above and passing
            # allow_empty_block_nums=True through to the kernel.

            # Sage FP8 requires a uniform (dense) top-k across the whole
            # sparsity pattern. Validate and cache the uniform value here
            # (plan()-time, already synchronizing above on the same tensor)
            # so run() doesn't need to sync on vsa_q2k_num on every call.
            self._vsa_sage_fp8_block_sparse_num = None
            if is_sage_fp8:
                uniform_num = int(self._vsa_q2k_num[0, 0, 0].item())
                if not torch.all(self._vsa_q2k_num == uniform_num):
                    raise ValueError(
                        "vsa_sm100_blk64 Sage FP8 requires a uniform number of "
                        "KV blocks per Q-block across the whole sparsity "
                        "pattern (upstream kernel limit: only a fixed "
                        "block_sparse_num is supported, not per-row "
                        "q2k_block_nums)."
                    )
                self._vsa_sage_fp8_block_sparse_num = uniform_num

            self.M = M
            self.N = N
            self.R = R
            self.C = C
            self._sm_scale = sm_scale
            self._vsa_kv_splits = kv_splits
            self._vsa_use_clc = use_clc
            self._vsa_q_scale = q_scale
            self._vsa_k_scale = k_scale
            self._vsa_v_scale = v_scale
            return

        # ---- VSA SM120 blk64 backend (sm120_blk64 CuTe-DSL kernel) ---------------
        if self._backend == "vsa_sm120_blk64":
            cc = get_compute_capability(self.device)
            arch = cc[0] * 10 + cc[1]
            if arch // 10 != 12:
                raise RuntimeError(
                    f"vsa_sm120_blk64 backend requires SM120/SM121, "
                    f"current device is SM{arch}"
                )
            if R != 64 or C != 64:
                raise ValueError(
                    f"vsa_sm120_blk64 backend requires R == C == 64 (got R={R}, C={C})"
                )
            if head_dim != 128:
                raise ValueError(
                    f"vsa_sm120_blk64 backend requires head_dim=128 (got {head_dim})"
                )
            if q_data_type not in (torch.float16, torch.bfloat16):
                raise ValueError(
                    "vsa_sm120_blk64 backend only supports float16 and bfloat16 inputs"
                )
            _vsa_common_checks(
                "vsa_sm120_blk64",
                R,
                C,
                M,
                N,
                num_qo_heads,
                num_kv_heads,
                mask,
                packed_mask,
                causal,
                pos_encoding_mode,
                logits_soft_cap,
            )

            MB = M // R
            NB = N // C
            # sm120 handles GQA natively via gqa_ratio; index is per QO head.
            H = num_qo_heads

            if block_mask is not None:
                if block_mask.shape != (H, MB, NB):
                    raise ValueError(
                        f"block_mask must have shape (num_qo_heads={H}, MB={MB}, NB={NB}), "
                        f"got {tuple(block_mask.shape)}"
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _block_mask_to_vsa_index(
                    block_mask, self.device, non_blocking
                )
            else:
                if indptr is None or indices is None:
                    raise ValueError(
                        "vsa_sm120_blk64 backend requires either block_mask or "
                        "(indptr, indices) to be provided."
                    )
                self._vsa_q2k_index, self._vsa_q2k_num = _bsr_to_vsa_index(
                    indptr, indices, MB, NB, H, self.device, non_blocking
                )

            self.M = M
            self.N = N
            self.R = R
            self.C = C
            self._sm_scale = sm_scale
            return

        if block_mask is not None:
            raise ValueError(
                "block_mask is only supported for the vsa_sm100_blk128, vsa_sm100_blk64, "
                "and vsa_sm120_blk64 backends."
            )
        if indptr is None or indices is None:
            raise ValueError("indptr and indices are required for non-VSA backends.")

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        num_blocks_row = len(indptr) - 1
        qo_indptr_host = R * torch.arange(num_blocks_row + 1, dtype=torch.int32)
        qo_indptr_host[-1] = M
        qo_indptr = qo_indptr_host.to(indptr.device, non_blocking=non_blocking)
        if indices.numel() > 0 and indices.max().item() * C > N:
            raise ValueError("indices out of bound")
        last_block_len = torch.full(
            (num_blocks_row,), C, dtype=torch.int32, device=indptr.device
        )

        if mask is not None or packed_mask is not None:
            mask_indptr = _compute_page_mask_indptr(
                qo_indptr,
                indptr,  # paged_kv_indptr
                last_block_len,  # paged_kv_last_page_len
                C,  # page_size
            )
        if packed_mask is None and mask is not None:
            # first convert BSR mask to flashinfer layout
            mask = convert_bsr_mask_layout(mask, indptr)
            # create packed mask from mask
            packed_mask, mask_indptr = segment_packbits(
                mask.contiguous().view(-1), mask_indptr, bitorder="little"
            )

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = indices.to(self.device, non_blocking=non_blocking)
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, non_blocking=non_blocking
        )
        if packed_mask is not None:
            self._packed_mask_buf = packed_mask.to(
                self.device, non_blocking=non_blocking
            )
            self._mask_indptr_buf = mask_indptr.to(
                self.device, non_blocking=non_blocking
            )
            mask_mode = MaskMode.CUSTOM.value
        else:
            self._packed_mask_buf = None
            self._mask_indptr_buf = None
            mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        self._mask_mode = mask_mode

        self.M = M
        self.N = N
        self.R = R
        self.C = C

        kv_indptr_host = indptr.to("cpu")

        # NOTE(Zihao): we haven't supported mask in cuda-core implementations but it should
        # be easy to add support for it if needed, leave it as a future work.
        # at this moment, when mask is provided, we use the tensor-core implementation
        if (
            R * (num_qo_heads // num_kv_heads) < 4
            and mask_mode != MaskMode.CUSTOM.value
            and q_data_type not in [torch.float8_e4m3fn, torch.float8_e5m2]
        ):
            # If the operation is not compute-bound, we use the cuda-core implementation
            self._use_tensor_cores = False
            self._cached_module = get_batch_decode_module(
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,
                head_dim,
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
            )

            self._plan_info = self._cached_module.plan(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                kv_indptr_host,
                num_blocks_row,
                num_qo_heads,
                num_kv_heads,
                C,
                False,  # is_cuda_graph_enabled
                -1,  # window_left
                logits_soft_cap,  # logits_soft_cap
                head_dim,
                head_dim,
                torch.empty(0, dtype=q_data_type),
                torch.empty(0, dtype=kv_data_type),
            )
        else:
            # if the operation is compute-bound, we use the tensor-core implementation
            self._use_tensor_cores = True

            if self._backend == "auto":
                self._backend = determine_attention_backend(
                    self.device,
                    PosEncodingMode[pos_encoding_mode].value,
                    use_fp16_qk_reduction,
                    mask_mode == MaskMode.CUSTOM.value,  # use_custom_mask
                    q_data_type,
                    kv_data_type,
                    head_dim_qk=head_dim,
                    head_dim_vo=head_dim,
                )

            get_module_args = (
                q_data_type,
                kv_data_type,
                self._o_dtype,
                indptr.dtype,
                head_dim,  # head_dim_qk
                head_dim,  # head_dim_vo
                PosEncodingMode[pos_encoding_mode].value,
                False,  # use_sliding_window
                logits_soft_cap > 0,  # use_logits_soft_cap
                use_fp16_qk_reduction,
            )
            self._cached_module = get_batch_prefill_module(
                self._backend, *get_module_args
            )

            kv_lens_arr_host = (kv_indptr_host[1:] - kv_indptr_host[:-1]) * self.C
            required_size = len(kv_lens_arr_host)
            if required_size > self._kv_lens_buffer.shape[0]:
                self._kv_lens_buffer = torch.empty(
                    (required_size,), dtype=torch.int32, device=self.device
                )
            self._kv_lens_buffer[:required_size].copy_(
                kv_lens_arr_host,
            )

            args = [
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._pin_memory_int_workspace_buffer,
                qo_indptr_host,
                kv_indptr_host,
                kv_lens_arr_host,
                M,  # total_num_rows
                num_blocks_row,  # batch_size
                num_qo_heads,
                num_kv_heads,
                self.C,  # page_size
                False,  # is_cuda_graph_enabled,
                head_dim,
                head_dim,
                causal,
                -1,  # window_left
            ]
            if self._backend == "fa2":
                args.append(-1)  # fixed_split_size
                args.append(False)  # disable_split_kv
                args.append(0)  # num_colocated_ctas
                args.append(0)  # uniform_q_len
            self._plan_info = self._cached_module.plan(
                *args,
            )

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta

    begin_forward = plan

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v, scale_q, scale_k, scale_v)

    @flashinfer_api(trace=block_sparse_attention_run_trace)
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        scale_q: Optional[torch.Tensor] = None,
        scale_k: Optional[torch.Tensor] = None,
        scale_v: Optional[torch.Tensor] = None,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(M, num_qo_heads, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(N, num_kv_heads, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(N, num_kv_heads, head_dim)``.
        scale_q : Optional[torch.Tensor]
            The scale tensor for query, per-head quantization with shape: ``[num_qo_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_k : Optional[torch.Tensor]
            The scale tensor for key, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        scale_v : Optional[torch.Tensor]
            The scale tensor for value, per-head quantization with shape: ``[num_kv_heads]``.
            Used with FP8 Quantization. If not provided, will be set to ``1.0``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        if self._backend == "cake":
            from flashinfer.cake_vsa import run_cake_vsa

            if scale_q is not None or scale_k is not None or scale_v is not None:
                raise ValueError("cake backend does not accept FP8 scale tensors")
            if self._cake_vsa_plan is None:
                raise RuntimeError("plan() must be called before run()")
            return run_cake_vsa(
                self._cake_vsa_plan,
                q,
                k,
                v,
                out=out,
                lse=lse,
                return_lse=return_lse,
                backend="cake",
            )

        # ---- cuTile backend (pure cuda.tile Python kernel) ------------------------
        # Map the BSR plan onto a paged prefill: each block-row (R query rows) is one
        # variable-length batch whose KV pages are the selected column-blocks (width C).
        if self._backend == "cutile":
            if return_lse:
                raise NotImplementedError(
                    "cuTile block-sparse backend does not support return_lse."
                )
            from .attention.kernels.cutile.fmha_prefill_bsr_cutile import (  # noqa: PLC0415
                prefill_attention_kv_paged_cutile,
            )

            R, C = self._R, self._C
            num_block_rows = self._sparse_num_block_rows

            # KV as page_size==C paged cache: [N, H_kv, D] -> [N // C, C, H_kv, D].
            k_cache = k.reshape(self._N // C, C, self._num_kv_heads, self._head_dim)
            v_cache = v.reshape(self._N // C, C, self._num_kv_heads, v.shape[-1])

            # Block table + per-batch length/offset arrays were materialized at
            # plan() time (they derive only from indptr/indices/R/C), so run()
            # does no host sync or allocation here and stays graph-capturable.
            actual_seq_lens_q = self._sparse_actual_seq_lens_q
            actual_seq_lens_kv = self._sparse_actual_seq_lens_kv
            block_tables = self._sparse_block_tables
            actual_seq_offset = self._sparse_actual_seq_offset

            # The kernel folds the softmax scale into k_scale (qk_scale =
            # k_scale * INV_LOG_2), so k_scale must carry the 1/sqrt(head_dim)
            # softmax scale, matching single_prefill_with_kv_cache's default.
            sm_scale = self._sm_scale
            if sm_scale is None:
                sm_scale = 1.0 / math.sqrt(self._head_dim)

            out, _ = prefill_attention_kv_paged_cutile(
                q=q,
                k_cache=k_cache,
                v_cache=v_cache,
                actual_seq_lens_q=actual_seq_lens_q,
                actual_seq_lens_kv=actual_seq_lens_kv,
                actual_seq_offset=actual_seq_offset,
                block_tables=block_tables,
                k_scale=sm_scale,
                v_scale=1.0,
                num_batch=num_block_rows,
                # Each block-row is one variable-length batch of exactly R query
                # rows, so the per-batch max query length is R (not the global M).
                # Passing R sizes the grid / LPT tile count to exactly ceil(R/
                # BLOCK_M) tiles instead of ceil(M/BLOCK_M), avoiding the ~M/R x
                # over-launch of CTAs that would otherwise early-return.
                max_seq_len=R,
                is_causal=self._causal,
                outputs=out,
            )
            return out

        # ---- VSA blk128 backend (BSA CuTe-DSL kernel, SM100/SM103) ---------------
        if self._backend == "vsa_sm100_blk128":
            from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk128 import (
                bsa_attn_sm100_blk128_fwd,
            )  # noqa: PLC0415

            return _vsa_run_core(
                bsa_attn_sm100_blk128_fwd,
                q,
                k,
                v,
                self._vsa_q2k_index,
                self._vsa_q2k_num,
                self._sm_scale,
                out,
                lse,
                return_lse,
            )

        # ---- VSA blk64 backend (BSA CuTe-DSL kernel, SM100/SM103) -----------------
        if self._backend == "vsa_sm100_blk64":
            from flashinfer.cute_dsl.sparse.bsa_attn_sm100_blk64 import (
                bsa_attn_sm100_blk64_fwd,
            )  # noqa: PLC0415

            return _vsa_run_core_blk64(
                bsa_attn_sm100_blk64_fwd,
                q,
                k,
                v,
                self._vsa_q2k_index,
                self._vsa_q2k_num,
                self._sm_scale,
                out,
                lse,
                return_lse,
                self._vsa_kv_splits,
                self._vsa_use_clc,
                self._vsa_q_scale,
                self._vsa_k_scale,
                self._vsa_v_scale,
                self._vsa_sage_fp8_block_sparse_num,
            )

        # ---- VSA SM120 blk64 backend (sm120_blk64 CuTe-DSL kernel) ---------------
        if self._backend == "vsa_sm120_blk64":
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
                bsa_attn_sm120_blk64_fwd,
            )  # noqa: PLC0415

            return _vsa_run_core(
                bsa_attn_sm120_blk64_fwd,
                q,
                k,
                v,
                self._vsa_q2k_index,
                self._vsa_q2k_num,
                self._sm_scale,
                out,
                lse,
                return_lse,
            )

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4
        k = k.reshape(-1, self.C, *k.shape[-2:])
        v = v.reshape(-1, self.C, *v.shape[-2:])

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        if is_float8(q):
            assert q.dtype == k.dtype == v.dtype
            assert q.shape[-1] == k.shape[-1] == v.shape[-1]
            assert self._backend == "fa3" and self._use_tensor_cores

            if scale_q is None:
                scale_q = torch.ones(q.shape[1], dtype=torch.float32, device=q.device)
            if scale_k is None:
                scale_k = torch.ones(k.shape[1], dtype=torch.float32, device=q.device)
            if scale_v is None:
                scale_v = torch.ones(v.shape[1], dtype=torch.float32, device=q.device)

        if self._use_tensor_cores:
            self._cached_module.paged_run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._qo_indptr,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                self._mask_mode,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                enable_pdl,
                # ADDITIONAL_FUNC_PARAMS
                self._packed_mask_buf,
                self._mask_indptr_buf,
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                None,  # maybe_prefix_len_ptr
                None,  # maybe_token_pos_in_items_ptr
                None,  # maybe_max_item_len_ptr
                logits_soft_cap,
                sm_scale,
                scale_q,
                scale_k,
                scale_v,
                rope_scale,
                rope_theta,
                0,  # token_pos_in_items_len
                self._workspace_size,  # workspace_size
            )
        else:
            self._cached_module.run(
                self._float_workspace_buffer,
                self._int_workspace_buffer,
                self._plan_info,
                q,
                k,
                v,
                self._paged_kv_indptr_buf,
                self._paged_kv_indices_buf,
                self._paged_kv_last_page_len,
                out,
                lse,
                TensorLayout[self._kv_layout].value,
                -1,  # window_left
                enable_pdl,
                # ADDITIONAL_FUNC_PARAMS
                _get_cache_alibi_slopes_buf(q.shape[1], self.device),
                logits_soft_cap,
                sm_scale,
                rope_scale,
                rope_theta,
            )

        return (out, lse) if return_lse else out

    def end_forward(self) -> None:
        r"""Warning: This method is deprecated and has no effect."""
        pass


class VariableBlockSparseAttentionWrapper:
    r"""Wrapper class for attention computation with a block-sparse matrix as attention mask.
    This API supports variable block sizes provided by ``block_row_sz`` and ``block_col_sz``.
    Besides, each ``kv_head_idx`` can specify its own sparse patterns without using the same mask.

    Example
    -------
    >>> import torch
    >>> import flashinfer
    >>> num_qo_heads = 1
    >>> num_kv_heads = 1
    >>> head_dim = 128
    >>> seq_len = 6 # This corresponds to the `block_row_sz` and `block_col_sz`
    >>> # allocate 128MB workspace buffer
    >>> workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device="cuda:0")
    >>> wrapper = flashinfer.VariableBlockSparseAttentionWrapper(workspace_buffer)
    >>> block_mask_map = torch.tensor([[[0, 0, 1], [1, 0, 1], [0, 1, 1]]], dtype=torch.bool, device="cuda:0")
    >>> block_row_sz = torch.tensor([[1, 2, 3]], dtype=torch.int32, device="cuda:0")
    >>> block_col_sz = torch.tensor([[3, 1, 2]], dtype=torch.int32, device="cuda:0")
    >>> wrapper.plan(
    ...     block_mask_map,
    ...     block_row_sz,
    ...     block_col_sz,
    ...     num_qo_heads,
    ...     num_kv_heads,
    ...     head_dim,
    ... )
    >>> q = torch.randn((num_qo_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> k = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> v = torch.randn((num_kv_heads, seq_len, head_dim), dtype=torch.float16, device="cuda:0")
    >>> o = wrapper.run(q, k, v)
    """

    @flashinfer_api
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        backend: str = "auto",
    ) -> None:
        r"""Constructs of :class:`VariableBlockSparseAttentionWrapper`.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The user reserved float workspace buffer used to store intermediate attention results
            in the split-k algorithm. The recommended size is 128MB, the device of the workspace
            buffer should be the same as the device of the input tensors.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )

        self._kv_lens_buffer = torch.empty(
            (32768,), dtype=torch.int32, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=torch.uint8,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = False
        self._kv_layout = "NHD"
        self._qo_indptr: Optional[torch.Tensor] = None
        self._paged_kv_indptr_buf: Optional[torch.Tensor] = None
        self._paged_kv_indices_buf: Optional[torch.Tensor] = None
        self._paged_kv_last_page_len: Optional[torch.Tensor] = None
        self._backend = backend

    def reset_workspace_buffer(
        self,
        float_workspace_buffer: torch.Tensor,
        int_workspace_buffer: torch.Tensor,
    ) -> None:
        r"""Reset the workspace buffer.

        Parameters
        ----------
        float_workspace_buffer : torch.Tensor
            The new float workspace buffer, the device of the new float workspace buffer should
            be the same as the device of the input tensors.

        int_workspace_buffer : torch.Tensor
            The new int workspace buffer, the device of the new int workspace buffer should
            be the same as the device of the input tensors.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self._int_workspace_buffer = int_workspace_buffer
        self._workspace_size = (
            float_workspace_buffer.numel() * float_workspace_buffer.element_size()
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
        )

    @flashinfer_api
    def plan(
        self,
        block_mask_map: torch.Tensor,
        block_row_sz: torch.Tensor,
        block_col_sz: torch.Tensor,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        causal: bool = False,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
        non_blocking: bool = True,
        q_data_type: Union[str, torch.dtype] = "float16",
        kv_data_type: Optional[Union[str, torch.dtype]] = None,
    ) -> None:
        r"""Create auxiliary data structures for block sparse attention.

        Parameters
        ----------
        block_mask_map : torch.Tensor
            The block mask map (boolean), shape ``(num_kv_heads, MB, NB)``, where ``MB`` is the number of blocks in the row dimension,
            ``NB`` is the number of blocks in the column dimension.
        block_row_sz : torch.Tensor
            The block row size, shape ``(num_kv_heads, MB,)``.
        block_col_sz : torch.Tensor
            The block column size, shape ``(num_kv_heads, NB,)``.
        num_qo_heads : int
            The number of heads in the query/output tensor.
        num_kv_heads : int
            The number of heads in the key/value tensor. Note that a group of ``qo_heads`` shares the same sparse pattern of ``kv_heads``.
        head_dim : int
            The dimension of each head.
        causal : bool
            Whether to apply causal mask to the attention matrix.
        pos_encoding_mode : str, optional
            The position encoding applied inside attention kernels, could be
            ``NONE``/``ROPE_LLAMA`` (LLAMA style rotary embedding) /``ALIBI``.
            Default is ``NONE``.
        use_fp16_qk_reduction : bool
            Whether to use f16 for qk reduction (faster at the cost of slight precision
            loss).
        logits_soft_cap : Optional[float]
            The attention logits soft capping value (used in Gemini, Grok and Gemma-2, etc.), if not
            provided, will be set to ``0``. If greater than 0, the logits will be capped according to
            formula:
            :math:`\texttt{logits_soft_cap} \times \mathrm{tanh}(x / \texttt{logits_soft_cap})`,
            where :math:`x` is the input logits.
        sm_scale : Optional[float]
            The scale used in softmax, if not provided, will be set to
            ``1.0 / sqrt(head_dim)``.
        rope_scale : Optional[float]
            The scale used in RoPE interpolation, if not provided, will be set to
            ``1.0``.
        rope_theta : Optional[float]
            The theta used in RoPE, if not provided, will be set to ``1e4``.
        non_blocking : bool
            Whether to copy the input tensors to the device asynchronously, defaults to ``True``.
        q_data_type : Union[str, torch.dtype]
            Dtype of the query tensor.  Used to specialize the JIT-compiled kernel.
            Defaults to ``"float16"``.
        kv_data_type : Optional[Union[str, torch.dtype]]
            Dtype of the key/value tensors.  When ``None``, defaults to
            ``q_data_type``.


        The :meth:`plan` method should be called before any :meth:`run` or
        :meth:`run_return_lse` calls, auxiliary data structures will be created
        during this call and cached for multiple kernel runs.

        The ``num_qo_heads`` must be a multiple of ``num_kv_heads``. If ``num_qo_heads``
        is not equal to ``num_kv_heads``, the function will use
        `grouped query attention <https://arxiv.org/abs/2305.13245>`_.
        """
        q_data_type = canonicalize_torch_dtype(q_data_type)
        if kv_data_type is None:
            kv_data_type = q_data_type
        kv_data_type = canonicalize_torch_dtype(kv_data_type)
        self._o_dtype = q_data_type

        if logits_soft_cap is None:
            logits_soft_cap = 0.0

        # num_blocks are constant across kv_heads
        num_blocks_row = block_row_sz.shape[-1]
        num_blocks_col = block_col_sz.shape[-1]

        # q layout: [seq_len, num_kv_heads, gqa_group_size, head_dim]
        # padded into: [seq_len * num_kv_heads, 1, gqa_group_size, head_dim]
        qo_indptr = torch.cat(
            [
                torch.zeros(1, dtype=torch.int32, device=block_row_sz.device),
                torch.cumsum(block_row_sz.flatten(), dim=0, dtype=torch.int32),
            ],
            dim=0,
        )
        qo_indptr_host = qo_indptr.to("cpu", non_blocking=non_blocking)
        last_block_len = torch.full(
            (num_blocks_row * num_kv_heads,),
            1,
            dtype=torch.int32,
            device=block_mask_map.device,
        )  # We use page_size == 1 for variable length support

        # HND kv layout: [num_kv_heads, num_blocks, block_size, head_dim]
        # padded into: [num_kv_heads * num_blocks, block_size, 1, head_dim]
        # for customized attention mask for each kv_head
        # NOTE(Yilong): This could be perf bottleneck. Consider Triton implementation.
        def _block_mask_map_to_expanded_indices(
            block_mask_map: torch.Tensor,  # [H, R, C] bool / {0,1}
            block_col_sz: torch.Tensor,  # [H, C]     int
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                block_mask_map:  bool/int  [num_kv_heads, num_blocks_row, num_blocks_col]
                block_col_sz:    int32/64  [num_kv_heads, num_blocks_col]
            Returns:
                kv_indptr:  [H*R + 1]  int32  —  CSR indptr
                kv_indices: [nnz]      int32  —  token indices per (head, row)
            """
            device = block_mask_map.device
            dtype_i = torch.int32

            # 1) Calculate the total length of each row (head, row)
            row_lengths = (block_mask_map * block_col_sz[:, None, :]).sum(-1)  # [H,R]
            kv_indptr = torch.cat(
                [
                    torch.zeros(1, dtype=dtype_i, device=device),
                    torch.cumsum(row_lengths.flatten(), 0),
                ],
                dim=0,
            )

            # 2) Calculate the offset of each column block within its head
            col_offset = (
                torch.cumsum(block_col_sz.to(dtype_i), 1) - block_col_sz
            )  # [H,C]
            head_len = block_col_sz.sum(1, dtype=dtype_i)
            head_offset = torch.cumsum(head_len, 0) - head_len

            # 3) Find all selected (h,r,c)
            h_idx, r_idx, c_idx = block_mask_map.nonzero(as_tuple=True)
            lengths = block_col_sz[h_idx, c_idx].to(dtype_i)  # [N]
            base = head_offset[h_idx] + col_offset[h_idx, c_idx]  # [N]

            # 4) Expand variable-length column blocks into token-level indices
            cum = torch.cumsum(lengths, 0)
            starts = torch.repeat_interleave(cum - lengths, lengths)  # [total]
            offsets_within = torch.arange(cum[-1], device=device) - starts
            kv_indices = torch.repeat_interleave(base, lengths) + offsets_within

            return kv_indptr.to(dtype=dtype_i, device=device), kv_indices.to(
                dtype=dtype_i, device=device
            )

        kv_indptr, kv_indices = _block_mask_map_to_expanded_indices(
            block_mask_map, block_col_sz
        )
        kv_indptr_host = kv_indptr.to("cpu", non_blocking=non_blocking)
        kv_indices_host = kv_indices.to("cpu", non_blocking=non_blocking)

        self._qo_indptr = qo_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indptr_buf = kv_indptr.to(self.device, non_blocking=non_blocking)
        self._paged_kv_indices_buf = kv_indices.to(
            self.device, non_blocking=non_blocking
        )
        self._paged_kv_last_page_len = last_block_len.to(
            self.device, non_blocking=non_blocking
        )
        torch.cuda.synchronize()  # for non-blocking copy
        self._mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value

        # Sanity check
        assert num_qo_heads % num_kv_heads == 0, (
            "num_qo_heads must be a multiple of num_kv_heads"
        )
        assert num_blocks_row * num_kv_heads + 1 == kv_indptr_host.shape[0]
        assert kv_indptr_host[-1].item() == kv_indices_host.shape[0], (
            f"{kv_indptr_host[-1].item()} != {kv_indices_host.shape[0]}"
        )
        assert num_kv_heads == block_mask_map.shape[0]
        assert num_kv_heads == block_row_sz.shape[0]
        assert num_kv_heads == block_col_sz.shape[0]
        assert num_blocks_row == block_mask_map.shape[1]
        assert num_blocks_col == block_mask_map.shape[2]

        if self._backend == "auto":
            self._backend = determine_attention_backend(
                self.device,
                PosEncodingMode[pos_encoding_mode].value,
                use_fp16_qk_reduction,
                self._mask_mode == MaskMode.CUSTOM.value,  # use_custom_mask
                q_data_type,
                kv_data_type,
                head_dim_qk=head_dim,
                head_dim_vo=head_dim,
            )

        get_module_args = (
            q_data_type,
            kv_data_type,
            self._o_dtype,
            kv_indptr_host.dtype,
            head_dim,  # head_dim_qk
            head_dim,  # head_dim_vo
            PosEncodingMode[pos_encoding_mode].value,
            False,  # use_sliding_window
            logits_soft_cap > 0,  # use_logits_soft_cap
            use_fp16_qk_reduction,
        )
        self._cached_module = get_batch_prefill_module(self._backend, *get_module_args)

        kv_lens_arr_host = kv_indptr_host[1:] - kv_indptr_host[:-1]  # page_size == 1
        required_size = len(kv_lens_arr_host)
        if required_size > self._kv_lens_buffer.shape[0]:
            self._kv_lens_buffer = torch.empty(
                (required_size,), dtype=torch.int32, device=self.device
            )
        self._kv_lens_buffer[:required_size].copy_(
            kv_lens_arr_host,
        )

        args = [
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_lens_arr_host,
            qo_indptr_host[-1].item(),  # total_num_rows
            num_blocks_row * num_kv_heads,  # batch_size
            num_qo_heads // num_kv_heads,  # num_qo_heads (gqa_group_size)
            1,  # num_kv_heads,
            1,  # page_size
            False,  # is_cuda_graph_enabled,
            head_dim,
            head_dim,
            causal,
            -1,  # window_left
        ]
        if self._backend == "fa2":
            args.append(-1)  # fixed_split_size
            args.append(False)  # disable_split_kv
            args.append(0)  # num_colocated_ctas
            args.append(0)  # uniform_q_len
        self._plan_info = self._cached_module.plan(
            *args,
        )

        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        self._num_kv_heads = num_kv_heads
        self._gqa_group_size = num_qo_heads // num_kv_heads

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        pos_encoding_mode: str = "NONE",
        use_fp16_qk_reduction: bool = False,
        logits_soft_cap: Optional[float] = None,
        sm_scale: Optional[float] = None,
        rope_scale: Optional[float] = None,
        rope_theta: Optional[float] = None,
    ) -> torch.Tensor:
        r"""Warning: This method is deprecated, please use :meth:`run` instead."""
        self._pos_encoding_mode = pos_encoding_mode
        self._use_fp16_qk_reduction = use_fp16_qk_reduction
        self._logits_soft_cap = logits_soft_cap
        self._sm_scale = sm_scale
        self._rope_scale = rope_scale
        self._rope_theta = rope_theta
        return self.run(q, k, v)

    @flashinfer_api(trace=variable_block_sparse_attention_run_trace)
    def run(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        out: Optional[torch.Tensor] = None,
        lse: Optional[torch.Tensor] = None,
        return_lse: bool = False,
        enable_pdl: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Compute block-sparse attention between Q/K/V tensors.

        Parameters
        ----------
        q : torch.Tensor
            The query tensor with shape ``(num_qo_heads, qo_len, head_dim)``.
        k : torch.Tensor
            The key tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        v : torch.Tensor
            The value tensor with shape ``(num_kv_heads, kv_len, head_dim)``.
        out : Optional[torch.Tensor]
            The output tensor, if not provided, will be allocated internally.
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool
            Whether to return the log-sum-exp of attention logits
        enable_pdl : bool
            Whether to enable Programmatic Dependent Launch (PDL). See https://docs.nvidia.com/cuda/cuda-c-programming-guide/#programmatic-dependent-launch-and-synchronization
            Only supported for >= sm90, and currently only for FA2 and CUDA core decode.

        Returns
        -------
        Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
            If :attr:`return_lse` is ``False``, the attention output, shape: ``[M, num_qo_heads, head_dim]``.
            If :attr:`return_lse` is ``True``, a tuple of two tensors:

            * The attention output, shape: ``[M, num_qo_heads, head_dim]``.
            * The logsumexp of attention output, shape: ``[M, num_qo_heads]``.
        """
        # NOTE(Zihao): defer import of einops
        import einops

        if enable_pdl is None:
            enable_pdl = device_support_pdl(q.device)

        pos_encoding_mode = self._pos_encoding_mode
        logits_soft_cap = self._logits_soft_cap
        sm_scale = self._sm_scale
        rope_scale = self._rope_scale
        rope_theta = self._rope_theta
        _check_pos_encoding_mode(pos_encoding_mode)
        if logits_soft_cap is None:
            logits_soft_cap = 0.0
        if sm_scale is None:
            sm_scale = 1.0 / math.sqrt(q.size(-1))
        if rope_scale is None:
            rope_scale = 1.0
        if rope_theta is None:
            rope_theta = 1e4

        # reshape to pad num_kv_heads into seq_len
        # input [num_qo_heads, qo_len, head_dim]
        # kernel layout is NHD -> [qo_len * num_kv_heads, gqa_group_size, head_dim]
        q = einops.rearrange(
            q,
            "(num_kv_heads gqa_group_size) qo_len head_dim -> (num_kv_heads qo_len) gqa_group_size head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()
        # HND -> [kv_len * num_kv_heads (num_pages), 1 (page_size), 1 (new_num_kv_heads), head_dim]
        k = einops.rearrange(
            k,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()
        v = einops.rearrange(
            v,
            "num_kv_heads kv_len head_dim -> (num_kv_heads kv_len) 1 1 head_dim",
        ).contiguous()

        if return_lse:
            if lse is None:
                lse = torch.empty(
                    (q.size(0), q.size(1)), dtype=torch.float32, device=q.device
                )
            else:
                check_shape_dtype_device(
                    lse, (q.size(0), q.size(1)), torch.float32, q.device, "lse"
                )

        if out is None:
            out = torch.empty_like(q, dtype=self._o_dtype)
        else:
            check_shape_dtype_device(out, q.shape, self._o_dtype, q.device, "out")

        self._cached_module.paged_run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q,
            k,
            v,
            self._qo_indptr,
            self._paged_kv_indptr_buf,
            self._paged_kv_indices_buf,
            self._paged_kv_last_page_len,
            out,
            lse,
            self._mask_mode,
            TensorLayout[self._kv_layout].value,
            -1,  # window_left
            enable_pdl,
            # ADDITIONAL_FUNC_PARAMS
            # Not supported yet
            None,  # packed_mask_buf
            None,  # mask_indptr_buf
            None,  # alibi_slopes_buf
            None,
            None,
            None,
            logits_soft_cap,
            sm_scale,
            None,  # scale_q
            None,  # scale_k
            None,  # scale_v
            rope_scale,
            rope_theta,
            0,  # token_pos_in_items_len
            self._workspace_size,
        )

        # [qo_len * num_kv_heads, gqa_group_size, head_dim] -> HND
        out = einops.rearrange(
            out,
            "(num_kv_heads qo_len) gqa_group_size head_dim -> (num_kv_heads gqa_group_size) qo_len head_dim",
            num_kv_heads=self._num_kv_heads,
        ).contiguous()

        if return_lse:
            lse = einops.rearrange(
                lse,
                "(num_kv_heads qo_len) gqa_group_size -> (num_kv_heads gqa_group_size) qo_len",
                num_kv_heads=self._num_kv_heads,
            ).contiguous()

        return (out, lse) if return_lse else out
