"""
Copyright (c) 2023 by FlashInfer team.

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
from typing import List, Literal, Optional, Tuple, Union, overload

import torch

from .api_logging import flashinfer_api
from .jit import gen_batch_mla_module, gen_trtllm_gen_fmha_module, setup_cubin_loader
from .jit.mla import gen_mla_module
from .utils import (
    MaskMode,
    check_shape_dtype_device,
    determine_mla_backend,
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
    log2e,
)
from .xqa import xqa_mla


def _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table):
    if q_nope_pe.ndim != 3:
        raise ValueError(f"Expected q_nope_pe.ndim == 3, got {q_nope_pe.ndim}")
    if ckv_kpe_cache.ndim != 3:
        raise ValueError(f"Expected ckv_kpe_cache.ndim == 3, got {ckv_kpe_cache.ndim}")
    if kv_len.ndim != 1:
        raise ValueError(f"Expected kv_len.ndim == 1, got {kv_len.ndim}")
    if page_table.ndim != 2:
        raise ValueError(f"Expected page_table.ndim == 2, got {page_table.ndim}")
    B_q, H, D_q = q_nope_pe.shape
    D_ckv = ckv_kpe_cache.shape[2]
    if H != 128:
        raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for q_nope_pe and ckv_kpe_cache, got {D_q} and {D_ckv}"
        )
    B_block_table, block_num = page_table.shape
    block_size = ckv_kpe_cache.shape[1]
    if B_q != B_block_table:
        raise ValueError(
            f"Expected batch size {B_q} for q_nope_pe and block_table, got {B_q} and {B_block_table}"
        )
    if block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
        )


def _check_trtllm_gen_mla_shape(
    query,
    kv_cache,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    sparse_mla_top_k,
    page_table,
    page_size,
):
    if query.ndim != 4:
        raise ValueError(f"Expected query.ndim == 4, got {query.ndim}")
    if kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 4, got {kv_cache.ndim}")
    if qk_nope_head_dim != 128:
        raise ValueError(f"Expected qk_nope_head_dim == 128, got {qk_nope_head_dim}")
    if kv_lora_rank != 512:
        raise ValueError(f"Expected kv_lora_rank == 512, got {kv_lora_rank}")
    if qk_rope_head_dim != 64:
        raise ValueError(f"Expected qk_rope_head_dim == 64, got {qk_rope_head_dim}")

    B_q, Q_len, H, D_q = query.shape
    D_ckv = kv_cache.shape[3]
    # if H != 128:
    #     raise ValueError(f"Expected 128 heads for query, got {H}")
    # todo(Yingyi): should we check num_heads == 128? Is this deepseek only?
    if D_q != D_ckv or D_q != 576:
        raise ValueError(
            f"Expected head dim 576 for query and kv_cache, got {D_q} and {D_ckv}"
        )

    if sparse_mla_top_k > 0:
        page_table_shape = page_table.shape
        if page_table_shape != (B_q, Q_len, sparse_mla_top_k):
            raise ValueError(
                f"Expected page_table.shape == (B_q, Q_len, sparse_mla_top_k), got {page_table_shape}"
            )
    else:
        B_block_table, block_num = page_table.shape
        block_size = page_size
        if B_q != B_block_table:
            raise ValueError(
                f"Expected batch size {B_q} for query and block_table, got {B_q} and {B_block_table}"
            )
        if block_num % (128 / block_size) != 0:
            raise ValueError(
                f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
            )


@functools.cache
def get_trtllm_gen_fmha_module():
    mod = gen_trtllm_gen_fmha_module()
    op = mod.build_and_load()
    setup_cubin_loader(mod.get_library_path())
    return op


@functools.cache
def get_mla_module():
    return gen_mla_module().build_and_load()


@functools.cache
def get_batch_mla_module(backend, *args):
    return gen_batch_mla_module(backend, *args).build_and_load()


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
            should be the same as the device of the input tensors.
        use_cuda_graph : bool, optional
            Whether to enable CUDA graph capture for the prefill kernels, if enabled, the
            auxiliary data structures will be stored in provided buffers. The ``batch_size``
            cannot change during the lifecycle of this wrapper when CUDAGraph is enabled.
        qo_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``qo_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indptr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indptr`` array, the size of the buffer
            should be ``[batch_size + 1]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_indices_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_indices`` array.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        kv_len_arr_buf : Optional[torch.Tensor]
            The user reserved buffer to store the ``kv_len_arr`` array, the size of the buffer
            should be ``[batch_size]``.
            This argument is only effective when ``use_cuda_graph`` is ``True``.
        backend : str
            The implementation backend, could be ``auto``/``fa2`` or ``fa3``. Defaults to ``auto``.
            If set to ``auto``, the function will automatically choose the backend based on the
            device architecture and kernel availability. If ``cutlass`` is provided, the MLA
            kernels will be generated by CUTLASS and only float_workspace_buffer is required and
            other arguments are ignored.
        """
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        if backend == "cutlass":
            self._backend = backend
            return

        self._int_workspace_buffer = torch.empty(
            (8 * 1024 * 1024,), dtype=torch.uint8, device=self.device
        )
        self._pin_memory_int_workspace_buffer = torch.empty(
            self._int_workspace_buffer.shape,
            dtype=self._int_workspace_buffer.dtype,
            pin_memory=True,
            device="cpu",
        )
        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr
        if backend == "auto":
            self._backend = determine_mla_backend(self.device)
        else:
            self._backend = backend

    @flashinfer_api
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
    ) -> None:
        r"""Plan the MLA attention computation.

        Parameters
        ----------
        qo_indptr : torch.IntTensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
            For decoding attention, the length of each query is 1, and the content
            of the tensor should be ``[0, 1, 2, ..., batch_size]``.
        kv_indptr : torch.IntTensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.IntTensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.IntTensor
            The query length of each request, shape: ``[batch_size]``.
        num_heads : int
            The number of heads in query/output tensor.
        head_dim_ckv : int
            The head dimension of compressed-kv.
        head_dim_kpe : int
            The head dimension for rope k-cache.
        page_size : int
            The page size of the paged kv-cache.
        causal : bool
            Whether to use causal attention.
        sm_scale : float
            The scale factor for softmax operation.
        q_data_type : torch.dtype
            The data type of the query tensor.
        kv_data_type : torch.dtype
            The data type of the kv-cache tensor.
        use_profiler : bool, optional
            Whether to enable intra-kernel profiler, default is False.
        """
        self._cached_module = get_batch_mla_module(
            self._backend,
            q_data_type,
            kv_data_type,
            q_data_type,
            qo_indptr.dtype,
            head_dim_ckv,
            head_dim_kpe,
            use_profiler,
        )
        qo_indptr_host = qo_indptr.to("cpu")
        kv_indptr_host = kv_indptr.to("cpu")
        kv_len_arr_host = kv_len_arr.to("cpu")

        if self._use_cuda_graph:
            self._qo_indptr_buf.copy_(qo_indptr, non_blocking=True)
            self._kv_indptr_buf.copy_(kv_indptr, non_blocking=True)
            self._kv_indices_buf[: len(kv_indices)].copy_(kv_indices, non_blocking=True)
            self._kv_len_arr_buf.copy_(kv_len_arr, non_blocking=True)
        else:
            self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
            self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
            self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
            self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)
        self._causal = causal
        self._page_size = page_size
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler

        self._plan_info = self._cached_module.plan(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._pin_memory_int_workspace_buffer,
            qo_indptr_host,
            kv_indptr_host,
            kv_len_arr_host,
            num_heads,
            head_dim_ckv,  # head_dim_o
            causal,
        )

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
    ) -> torch.Tensor: ...

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
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

    @flashinfer_api
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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation.

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
        lse : Optional[torch.Tensor]
            The log-sum-exp of attention logits, if not provided, will be allocated internally.
        return_lse : bool, optional
            Whether to return the log-sum-exp value, default is False.
        profiler_buffer : Optional[torch.Tensor]
            The buffer to store the profiler data.
        kv_len : Optional[torch.Tensor]
            The query length of each request, shape: ``[batch_size]``. Required when ``backend`` is ``cutlass``.
        page_table : Optional[torch.Tensor]
            The page table of the paged kv-cache, shape: ``[batch_size, num_pages]``. Required when ``backend`` is ``cutlass``.
        """
        if self._backend == "cutlass":
            if return_lse:
                raise ValueError("return_lse does not support cutlass backend for now.")
            if profiler_buffer is not None:
                raise ValueError(
                    "profiler_buffer does not support cutlass backend for now."
                )
            self._cached_module = get_mla_module()
            if out is None:
                out = torch.empty_like(q_nope)
            else:
                check_shape_dtype_device(
                    out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
                )
            q_nope_pe = torch.cat([q_nope, q_pe], dim=-1)
            ckv_kpe_cache = torch.cat([ckv_cache, kpe_cache], dim=-1)
            _check_cutlass_shape(q_nope_pe, ckv_kpe_cache, kv_len, page_table)
            lse = torch.empty(0, dtype=torch.float32, device=self.device)
            self._cached_module.cutlass_mla_paged_attention(
                self._float_workspace_buffer,
                out,
                lse,
                q_nope_pe,
                ckv_kpe_cache,
                kv_len,
                page_table,
            )
            return out

        if profiler_buffer is None:
            if self._use_profiler:
                raise ValueError(
                    "Profiler is enabled, profiler_buffer must be provided"
                )
        num_heads = q_nope.shape[1]
        page_size = self._page_size
        sm_scale = self._sm_scale
        causal = self._causal
        mask_mode = MaskMode.CAUSAL.value if causal else MaskMode.NON_CAUSAL.value
        device = self.device
        if out is None:
            out = torch.empty_like(q_nope)
        else:
            check_shape_dtype_device(
                out, q_nope.shape, q_nope.dtype, q_nope.device, "out"
            )

        if return_lse:
            if lse is None:
                lse = torch.empty(q_nope.shape[:2], dtype=torch.float32, device=device)
            else:
                check_shape_dtype_device(
                    lse, q_nope.shape[:2], torch.float32, q_nope.device, "lse"
                )
        profiler_args = (profiler_buffer,) if self._use_profiler else ()
        self._cached_module.run(
            self._float_workspace_buffer,
            self._int_workspace_buffer,
            self._plan_info,
            q_nope,
            q_pe,
            ckv_cache,
            kpe_cache,
            self._kv_indices_buf,
            out,
            lse,
            mask_mode,
            num_heads,
            page_size,
            sm_scale,
            return_lse_base_on_e,
            *profiler_args,
        )

        return (out, lse) if return_lse else out


@flashinfer_api
def trtllm_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    sparse_mla_top_k: int = 0,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    enable_pdl: bool = None,
    backend: str = "auto",
) -> torch.Tensor:
    """
    Parameters
    ----------
    query: [batch_size, q_len_per_request, num_heads, head_dim_qk], head_dim_qk = qk_nope_head_dim (kv_lora_rank) + qk_rope_head_dim, should be concated q_nope + q_rope; q_len_per_request is the MTP query length.
    kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe], should be concated ckv_cache + kpe_cache
    workspace_buffer: [num_semaphores, 4], used for multi_block mode. Must be initialized to 0 for its first use.
    qk_nope_head_dim: qk_nope_head_dim, must be 128
    kv_lora_rank: kv_lora_rank, must be 512
    qk_rope_head_dim: qk_rope_head_dim, must be 64
    sparse_mla_top_k: sparse MLA top k, must be 0 for non-sparse MLA.
    block_tables: page_table of kv cache, [batch_size, num_pages]
    seq_lens: query_len
    max_seq_len: max sequence length for kv_cache
    out: output tensor, if not provided, will be allocated internally
    bmm1_scale: fused scale for mla bmm1 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.
    bmm2_scale: fused scale for mla bmm2 input.
        when using trtllm-gen backend, it can be a torch.Tensor with dtype torch.float32.
    sinks: additional value per head in the denominator of the softmax.
    backend : str = "auto"
        The implementation backend, could be ``auto``/``xqa`` or ``trtllm-gen``. Defaults to ``auto``.
        When set to ``auto``, the backend will be chosen based on the device architecture and kernel availability.
        For sm_100 and sm_103 (blackwell architecture), ``auto`` will choose ``trtllm-gen`` backend.
        For sm_120 (blackwell architecture), ``auto`` will choose ``xqa`` backend.

    Note
    ----
    In MLA, the actual BMM1 and BMM2 scales applied would be fused as:
    bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
    bmm2_scale = v_scale * o_scale
    or,
    bmm1_scale = torch.Tensor([q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5))
    bmm2_scale = torch.Tensor([v_scale * o_scale])

    The two scale factors should be static constant for cuda graph capture.
    Either (bmm1_scale, bmm2_scale) or (bmm1_scale_log2_tensor, bmm2_scale_tensor) should be provided.

    For static constant scale factors, the scale factors should be provided as float.
        - (bmm1_scale, bmm2_scale)
    For on-device fused scale tensors, which could dynamically change, the scale factors should be provided as torch.Tensor.
        - (bmm1_scale_log2_tensor, bmm2_scale_tensor)
        - Currently, only fp8 tensor core operation supports this mode.
    When both are provided, the dynamic scale factor tensors will be used.
    """
    if backend == "auto":
        backend = (
            "trtllm-gen" if get_compute_capability(query.device)[0] == 10 else "xqa"
        )
    if isinstance(bmm1_scale, torch.Tensor):
        assert bmm1_scale.dtype == torch.float32
        bmm1_scale = bmm1_scale * log2e
    if isinstance(bmm2_scale, torch.Tensor):
        assert bmm2_scale.dtype == torch.float32
    if backend == "xqa":
        if (
            get_compute_capability(query.device)[0] != 12
            or query.dtype != torch.float8_e4m3fn
            or kv_cache.dtype != torch.float8_e4m3fn
        ):
            raise ValueError(
                f"XQA MLA only supports fp8 operation on SM120 GPUs, got {query.dtype} and {kv_cache.dtype}"
            )
        if sinks is not None:
            raise ValueError("XQA MLA does not support sinks")
        if query.size(1) != 1:
            raise ValueError(
                f"XQA MLA only supports q_len_per_request == 1, got {query.size(1)}"
            )
        return xqa_batch_decode_with_kv_cache_mla(
            query,
            kv_cache,
            workspace_buffer,
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            block_tables,
            seq_lens,
            max_seq_len,
            out,
            bmm1_scale,
            bmm2_scale,
            sinks,
            enable_pdl,
        )
    elif backend == "trtllm-gen":
        enable_pdl = (
            device_support_pdl(query.device) if enable_pdl is None else enable_pdl
        )
        run_func = get_trtllm_gen_fmha_module().trtllm_paged_attention_decode
        sm_count = get_device_sm_count(query.device)

        block_size = kv_cache.size(-2)
        if (
            block_size != 32 and block_size != 64
        ):  # todo(Yingyi): add support for more block sizes?
            raise ValueError(f"Supported block_size are 32 and 64, got {block_size}")

        _check_trtllm_gen_mla_shape(
            query,
            kv_cache,
            qk_nope_head_dim,
            kv_lora_rank,
            qk_rope_head_dim,
            sparse_mla_top_k,
            block_tables,
            block_size,
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

        batch_size = query.size(0)
        max_q_len = query.size(1)
        query = query.flatten(0, 1)  # [B*S, H, D]

        run_func(
            out,
            None,  # fp4 output not supported in wrapper api yet.
            query,
            kv_cache,
            kv_cache,
            workspace_buffer,
            block_tables,
            seq_lens,
            max_q_len,
            max_seq_len,
            bmm1_scale,
            bmm2_scale,
            -1,  # o_sf_scale
            -1,  # o_sf_vec_size
            0,  # o_sf_start_index
            batch_size,
            -1,  # window_left
            sparse_mla_top_k,
            sm_count,
            enable_pdl,
            workspace_buffer.numel() * workspace_buffer.element_size(),
            sinks,
            None,  # cum_seq_lens_q
        )

        return out
    else:
        raise ValueError(f"Backend {backend} not supported")


@flashinfer_api
def xqa_batch_decode_with_kv_cache_mla(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    workspace_buffer: torch.Tensor,
    qk_nope_head_dim: int,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int,
    out: Optional[torch.Tensor] = None,
    bmm1_scale: Union[float, torch.Tensor] = 1.0,
    bmm2_scale: Union[float, torch.Tensor] = 1.0,
    sinks: Optional[List[torch.Tensor]] = None,
    enable_pdl: bool = None,
) -> torch.Tensor:
    """
    Parameters:
    query: [batch_size, q_len_per_request, num_heads, head_dim_qk], head_dim_qk = qk_nope_head_dim (kv_lora_rank) + qk_rope_head_dim, should be concated q_nope + q_rope; q_len_per_request is the MTP query length.
    kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe], should be concated ckv_cache + kpe_cache
    workspace_buffer: torch.Tensor. Must be initialized to 0 for its first use.
    qk_nope_head_dim: qk_nope_head_dim, must be 128
    kv_lora_rank: kv_lora_rank, must be 512
    qk_rope_head_dim: qk_rope_head_dim, must be 64
    block_tables: page_table of kv cache, [batch_size, num_pages]
    seq_lens: query_len
    max_seq_len: max sequence length for kv_cache
    out: output tensor, if not provided, will be allocated internally
    bmm1_scale: fused scale for mla bmm1 input. Can be a float or a torch.Tensor.
    bmm2_scale: fused scale for mla bmm2 input. Can be a float or a torch.Tensor.
    sinks: additional value per head in the denominator of the softmax.

    Note:
    In MLA, the actual BMM1 and BMM2 scales applied would be fused as:
    bmm1_scale = q_scale * k_scale * sm_scale / (head_dim_qk ** 0.5)
    bmm2_scale = v_scale * o_scale

    The two scale factors should be static constant for cuda graph capture.
    Either (bmm1_scale, bmm2_scale) or (bmm1_scale_log2_tensor, bmm2_scale_tensor) should be provided.

    For static constant scale factors, the scale factors should be provided as float.
        - (bmm1_scale, bmm2_scale)
    For on-device fused scale tensors, which could dynamically change, the scale factors should be provided as torch.Tensor.
        - (bmm1_scale_log2_tensor, bmm2_scale_tensor)
        - Currently, only fp8 tensor core operation supports this mode.
    When both are provided, the dynamic scale factor tensors will be used.
    """
    enable_pdl = device_support_pdl(query.device) if enable_pdl is None else enable_pdl
    sm_count = get_device_sm_count(query.device)

    block_size = kv_cache.size(-2)
    q_len_per_request = query.size(1)
    if q_len_per_request != 1:
        raise ValueError(
            f"XQA MLA only supports q_len_per_request == 1, got {q_len_per_request}"
        )
    if query.dtype != torch.float8_e4m3fn or kv_cache.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"XQA MLA only supports fp8 tensor core operation, got {query.dtype} and {kv_cache.dtype}"
        )
    if sinks is not None:
        raise ValueError("XQA MLA does not support sinks")

    _check_trtllm_gen_mla_shape(
        query,
        kv_cache,
        qk_nope_head_dim,
        kv_lora_rank,
        qk_rope_head_dim,
        0,  # sparse_mla_top_k
        block_tables,
        block_size,
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

    workspace_u8 = workspace_buffer.view(torch.uint8)
    semaphore = workspace_u8[: 8 * 1024 * 1024]  # reserve 8MB for semaphore
    scratch = workspace_u8[8 * 1024 * 1024 :]
    # This can not be replaced by kv_cache.transpose(1, 2) because the stride is not the same
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
