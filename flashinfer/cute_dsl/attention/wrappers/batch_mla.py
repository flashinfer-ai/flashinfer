# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""BatchMLAPagedAttentionWrapperCuteDSL — PyTorch-facing API for MLA decode attention.

Constructs BlackwellMultiLatentAttentionForward from user-facing parameters,
compiles it, and provides the plan()/run() interface.
"""

import math
from typing import Type, Tuple, Optional, Union, overload, Literal

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
from cutlass.cute.runtime import from_dlpack

from ..mla_decode import BlackwellMultiLatentAttentionForward
from ..mla_config import mla_can_implement
from ..scheduler.mla_persistent import mla_get_split_kv


def ceil_div(a: int, b: int) -> int:
    return (a + b - 1) // b


def create_page_table(
    batch_size,
    seq_len,
    is_var_seq,
    use_page_table,
    page_size,
    cache_seqs_torch,
    kv_indptr=None,
    kv_indices=None,
):
    page_table_ref, page_table, page_table_gpu = None, None, None
    if use_page_table:
        max_seq_len = seq_len if not is_var_seq else torch.max(cache_seqs_torch)
        page_count = ceil_div(max_seq_len, page_size)
        page_table_ref = torch.empty([batch_size, page_count], dtype=torch.int32)
        if kv_indptr is not None and kv_indices is not None:
            kv_indptr_cpu = kv_indptr.cpu()
            kv_indices_cpu = kv_indices.cpu()
            for b in range(batch_size):
                start = kv_indptr_cpu[b].item()
                end = kv_indptr_cpu[b + 1].item()
                for j in range(page_count):
                    if start + j < end:
                        page_table_ref[b, j] = kv_indices_cpu[start + j].item()
                    else:
                        page_table_ref[b, j] = 0
        else:
            for b in range(batch_size):
                for j in range(page_count):
                    page_table_ref[b, j] = b + j * batch_size
        page_table_gpu = page_table_ref.permute(1, 0).cuda()
        page_table = from_dlpack(page_table_gpu, assumed_align=16).mark_layout_dynamic(
            leading_dim=0
        )
    return page_table_ref, page_table, page_table_gpu


def create_block_split_kvs(
    batch_size,
    split_kv,
    cache_seqs_ref,
    is_var_split_kv,
    mma_qk_tiler_mn,
    cluster_shape_mnk,
    max_active_clusters,
):
    block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu = None, None, None
    if is_var_split_kv:
        block_split_kvs_ref = torch.zeros([batch_size], dtype=torch.int32)
        for b in range(batch_size):
            block_split_kvs_ref[b] = mla_get_split_kv(
                batch_size,
                cache_seqs_ref[b].item(),
                mma_qk_tiler_mn,
                max_active_clusters * cluster_shape_mnk[0],
            )
        split_kv = torch.max(block_split_kvs_ref).item()
        block_split_kvs_gpu = block_split_kvs_ref.cuda()
        block_split_kvs = from_dlpack(
            block_split_kvs_gpu, assumed_align=16
        ).mark_layout_dynamic()
    elif split_kv <= 0:
        split_kv = mla_get_split_kv(
            batch_size,
            cache_seqs_ref[0].item(),
            mma_qk_tiler_mn,
            max_active_clusters * cluster_shape_mnk[0],
        )
    return split_kv, block_split_kvs_ref, block_split_kvs, block_split_kvs_gpu


def mla_get_workspace_size(H, D, B, split_kv, acc_dtype):
    """Get workspace size (bytes) for MLA split-KV intermediate buffers.

    :param H: Number of heads
    :param D: Latent dimension
    :param B: Batch size
    :param split_kv: Split-KV factor
    :param acc_dtype: Accumulator data type
    :return: Workspace size in bytes (0 if split_kv == 1)
    """
    if split_kv == 1:
        return 0
    return B * H * split_kv * (D + 1) * acc_dtype.width // 8


def create_workspace(num_heads, latent_dim, batch_size, split_kv, acc_dtype):
    workspace_size = mla_get_workspace_size(
        num_heads, latent_dim, batch_size, split_kv, acc_dtype
    )

    workspace, workspace_torch = None, None
    if workspace_size > 0:
        workspace_torch = torch.empty([workspace_size], dtype=torch.int8).cuda()
        workspace = from_dlpack(workspace_torch, assumed_align=16)
    return workspace, workspace_torch


def torch_to_cute(
    torch_tensor_gpu,
    dtype,
    is_dynamic_layout=True,
    page_table=None,
    page_size=None,
    cache_seqs=None,
    is_lse=False,
):
    if is_lse:
        shape = torch_tensor_gpu.shape
        B, HK = shape
        permute_order = (1, 0)
        stride_order = (1, 0)
        leading_dim = 0
    else:
        shape = torch_tensor_gpu.shape
        B, HK, D = shape
        permute_order = (1, 2, 0)
        stride_order = (2, 0, 1)
        leading_dim = 1
    if page_table is not None:
        if cache_seqs is not None:
            max_seq_len = torch.max(cache_seqs)
            shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
        else:
            shape = (B * ceil_div(HK, page_size), page_size, D)

    torch_tensor_gpu = torch_tensor_gpu.permute(permute_order)

    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype
    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        ).mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=(128 // dtype.width),
        )

    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )

    return cute_tensor, torch_tensor_gpu


def create_tensor(
    B,
    HK,
    D,
    dtype,
    is_dynamic_layout=True,
    page_table=None,
    cache_seqs=None,
    is_lse=False,
    page_size=None,
):
    shape = (B, HK, D)
    if page_table is not None:
        if cache_seqs is not None:
            max_seq_len = torch.max(cache_seqs)
            shape = (B * ceil_div(max_seq_len, page_size), page_size, D)
        else:
            shape = (B * ceil_div(HK, page_size), page_size, D)
    permute_order = (1, 2, 0)
    stride_order = (2, 0, 1)
    leading_dim = 1
    if is_lse:
        shape = (B, HK)
        permute_order = (1, 0)
        stride_order = (1, 0)
        leading_dim = 0
    init_config = cutlass.torch.RandomInitConfig(min_val=-2, max_val=2)
    torch_dtype = cutlass_torch.dtype(dtype)
    torch_tensor_cpu = cutlass_torch.create_and_permute_torch_tensor(
        shape,
        torch_dtype,
        permute_order=permute_order,
        init_type=cutlass.torch.TensorInitType.RANDOM,
        init_config=init_config,
    )
    torch_tensor_gpu = torch_tensor_cpu.cuda()

    cute_tensor = from_dlpack(torch_tensor_gpu, assumed_align=16)
    cute_tensor.element_type = dtype

    if is_dynamic_layout:
        cute_tensor = cute_tensor.mark_layout_dynamic(
            leading_dim=leading_dim
        ).mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=stride_order,
            divisibility=(128 // dtype.width),
        )
    cute_tensor = cutlass_torch.convert_cute_tensor(
        torch_tensor_gpu,
        cute_tensor,
        dtype,
        is_dynamic_layout=is_dynamic_layout,
    )
    return cute_tensor, torch_tensor_gpu


class BatchMLAPagedAttentionWrapperCuteDSL:
    def __init__(
        self,
        float_workspace_buffer: torch.Tensor,
        split_kv: int = -1,
        use_cuda_graph: bool = False,
        qo_indptr: Optional[torch.Tensor] = None,
        kv_indptr: Optional[torch.Tensor] = None,
        kv_indices: Optional[torch.Tensor] = None,
        kv_len_arr: Optional[torch.Tensor] = None,
    ) -> None:
        self._float_workspace_buffer = float_workspace_buffer
        self.device = float_workspace_buffer.device

        self._use_cuda_graph = use_cuda_graph
        self._qo_indptr_buf = qo_indptr
        self._kv_indptr_buf = kv_indptr
        self._kv_indices_buf = kv_indices
        self._kv_len_arr_buf = kv_len_arr

        self._is_persistent = True
        self._is_cpasync = False
        self._use_page_table = True
        self._in_dtype = None
        self._out_dtype = None
        self._acc_dtype = cutlass.Float32
        self._lse_dtype = cutlass.Float32
        self._split_kv = split_kv

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
        qo_indptr : torch.Tensor
            The indptr of the query/output tensor, shape: ``[batch_size + 1]``.
        kv_indptr : torch.Tensor
            The indptr of the paged kv-cache, shape: ``[batch_size + 1]``.
        kv_indices : torch.Tensor
            The page indices of the paged kv-cache, shape: ``[kv_indptr[-1]]`` or larger.
        kv_len_arr : torch.Tensor
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

        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required to run this example!")

        batch_size = qo_indptr.shape[0] - 1
        seq_len = kv_len_arr.max().item()
        pages_num = math.ceil(seq_len / page_size)

        if torch.all(kv_len_arr == kv_len_arr[0]):
            self._is_var_seq = False
            self._is_var_split_kv = False
        else:
            self._is_var_seq = True
            self._is_var_split_kv = self._split_kv > 0

        self._batch_size = batch_size
        self._seq_len = seq_len
        self._pages_num = math.ceil(seq_len / page_size)
        self._num_heads = num_heads
        self._head_dim_ckv = head_dim_ckv
        self._head_dim_kpe = head_dim_kpe
        self._page_size = page_size
        self._causal = causal
        self._sm_scale = sm_scale
        self._use_profiler = use_profiler
        self._use_2cta_instrs = num_heads == 128
        self._mma_qk_tiler_mn = (128, 128)
        self._mma_pv_tiler_mn = (128, 256)
        self._cluster_shape_mnk = (2, 1, 1)

        if q_data_type == torch.bfloat16:
            self._in_dtype = cutlass.BFloat16
            self._out_dtype = cutlass.BFloat16
        elif q_data_type == torch.half:
            self._in_dtype = cutlass.Float16
            self._out_dtype = cutlass.Float16
        elif q_data_type == torch.float8_e4m3fn:
            self._in_dtype = cutlass.Float8E4M3FN
            self._out_dtype = cutlass.Float16
        else:
            raise ValueError(f"Unsupported input data type: {q_data_type}")

        q_nope = torch.randn(
            batch_size * 1, num_heads, head_dim_ckv, dtype=q_data_type, device="cuda"
        )
        q_pe = torch.randn(
            batch_size * 1, num_heads, head_dim_kpe, dtype=q_data_type, device="cuda"
        )
        ckv = torch.randn(
            batch_size * pages_num,
            page_size,
            head_dim_ckv,
            dtype=q_data_type,
            device="cuda",
        )
        kpe = torch.randn(
            batch_size * pages_num,
            page_size,
            head_dim_kpe,
            dtype=q_data_type,
            device="cuda",
        )

        if not mla_can_implement(
            batch_size,
            seq_len,
            num_heads,
            head_dim_ckv,
            head_dim_kpe,
            self._in_dtype,
            self._out_dtype,
            self._acc_dtype,
            self._lse_dtype,
            self._mma_qk_tiler_mn,
            self._mma_pv_tiler_mn,
            self._split_kv,
            self._is_persistent,
            self._is_cpasync,
            self._is_var_seq,
            self._is_var_split_kv,
            self._use_page_table,
            page_size,
        ):
            raise TypeError(
                f"Unsupported testcase {self._in_dtype}, "
                f"{self._out_dtype}, "
                f"{self._acc_dtype}, "
                f"{self._lse_dtype}, {self._mma_qk_tiler_mn}, "
                f"{self._mma_pv_tiler_mn}, {self._split_kv}, {self._is_persistent}, "
                f"{self._is_cpasync}, {self._is_var_seq}, {self._is_var_split_kv}, "
                f"{self._use_page_table}, {page_size}"
            )

        self._qo_indptr_buf = qo_indptr.to(self.device, non_blocking=True)
        self._kv_indptr_buf = kv_indptr.to(self.device, non_blocking=True)
        self._kv_indices_buf = kv_indices.to(self.device, non_blocking=True)
        self._kv_len_arr_buf = kv_len_arr.to(self.device, non_blocking=True)

        self._cache_seqs_cute = from_dlpack(kv_len_arr, assumed_align=16)
        self._page_table_ref, self._page_table, self._page_table_gpu = (
            create_page_table(
                batch_size,
                seq_len,
                self._is_var_seq,
                self._use_page_table,
                page_size,
                kv_len_arr,
                kv_indptr=kv_indptr,
                kv_indices=kv_indices,
            )
        )
        cluster_shape_mnk = self._cluster_shape_mnk
        hardware_info = cutlass.utils.HardwareInfo()
        max_active_clusters = hardware_info.get_max_active_clusters(
            cluster_shape_mnk[0] * cluster_shape_mnk[1]
        )
        (
            self._split_kv,
            self._block_split_kvs_ref,
            self._block_split_kvs,
            self._block_split_kvs_gpu,
        ) = create_block_split_kvs(
            batch_size,
            self._split_kv,
            kv_len_arr,
            self._is_var_split_kv,
            self._mma_qk_tiler_mn,
            cluster_shape_mnk,
            max_active_clusters,
        )

        q_latent_cute, q_latent_torch = torch_to_cute(
            q_nope, self._in_dtype, is_dynamic_layout=True
        )

        q_rope_cute, q_rope_torch = torch_to_cute(
            q_pe, self._in_dtype, is_dynamic_layout=True
        )

        c_latent_cute, c_latent_torch = torch_to_cute(
            ckv,
            self._in_dtype,
            is_dynamic_layout=True,
            page_table=self._page_table,
            page_size=self._page_size,
            cache_seqs=kv_len_arr,
        )

        c_rope_cute, c_rope_torch = torch_to_cute(
            kpe,
            self._in_dtype,
            is_dynamic_layout=True,
            page_table=self._page_table,
            page_size=self._page_size,
            cache_seqs=kv_len_arr,
        )

        o_cute, o_torch = create_tensor(
            batch_size, num_heads, head_dim_ckv, self._out_dtype, is_dynamic_layout=True
        )
        lse_cute, lse_torch = create_tensor(
            batch_size,
            num_heads,
            1,
            self._lse_dtype,
            is_dynamic_layout=True,
            is_lse=True,
        )
        self._workspace, self._workspace_torch = create_workspace(
            num_heads, head_dim_ckv, batch_size, self._split_kv, self._acc_dtype
        )

        from ..mla_config import MLAConfig

        mla_config = MLAConfig(
            latent_dim=head_dim_ckv,
            rope_dim=head_dim_kpe,
            num_heads=num_heads,
            acc_dtype=self._acc_dtype,
            lse_dtype=self._lse_dtype,
            mma_qk_tiler_mn=self._mma_qk_tiler_mn,
            mma_pv_tiler_mn=self._mma_pv_tiler_mn,
            max_active_clusters=max_active_clusters,
            is_persistent=self._is_persistent,
            is_cpasync=self._is_cpasync,
            use_page_table=self._use_page_table,
            is_var_seq=self._is_var_seq,
            is_var_split_kv=self._is_var_split_kv,
            use_2cta_instrs=self._use_2cta_instrs,
            cluster_shape_mnk=self._cluster_shape_mnk,
        )
        mla = BlackwellMultiLatentAttentionForward(mla_config)

        torch_stream = torch.cuda.current_stream()
        stream = cuda.CUstream(torch_stream.cuda_stream)

        compiled_mla = cute.compile(
            mla,
            q_latent_cute,
            q_rope_cute,
            c_latent_cute,
            c_rope_cute,
            self._page_table,
            o_cute,
            lse_cute,
            self._workspace,
            self._split_kv,
            self._cache_seqs_cute,
            self._block_split_kvs,
            self._sm_scale,
            1.0,
            stream,
        )

        self._compiled_mla = compiled_mla

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
    ) -> Tuple[torch.Tensor, torch.Tensor]: ...

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
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        r"""Run the MLA attention computation."""

        if self._compiled_mla is None:
            raise RuntimeError("Plan the MLA attention computation first!")

        assert return_lse is True, "return_lse must be True for CuteDSL implementation"

        q_latent_cute, q_latent_torch = torch_to_cute(
            q_nope, self._in_dtype, is_dynamic_layout=True
        )

        q_rope_cute, q_rope_torch = torch_to_cute(
            q_pe, self._in_dtype, is_dynamic_layout=True
        )

        c_latent_cute, c_latent_torch = torch_to_cute(
            ckv_cache,
            self._in_dtype,
            is_dynamic_layout=True,
            page_table=self._page_table,
            page_size=self._page_size,
            cache_seqs=kv_len,
        )

        c_rope_cute, c_rope_torch = torch_to_cute(
            kpe_cache,
            self._in_dtype,
            is_dynamic_layout=True,
            page_table=self._page_table,
            page_size=self._page_size,
            cache_seqs=kv_len,
        )

        if out is None:
            out = torch.empty_like(q_nope, device=q_nope.device)
        o_cute, o_torch = torch_to_cute(out, self._out_dtype, is_dynamic_layout=True)

        if lse is None:
            lse = torch.empty(
                (self._batch_size, self._num_heads),
                dtype=torch.float32,
                device=q_nope.device,
            )
        lse_cute, lse_torch = torch_to_cute(
            lse, self._lse_dtype, is_dynamic_layout=True, is_lse=True
        )

        stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)

        self._compiled_mla(
            q_latent_cute,
            q_rope_cute,
            c_latent_cute,
            c_rope_cute,
            self._page_table,
            o_cute,
            lse_cute,
            self._workspace,
            self._split_kv,
            self._cache_seqs_cute,
            self._block_split_kvs,
            self._sm_scale,
            1.0,
            stream,
        )

        return o_torch, lse_torch
