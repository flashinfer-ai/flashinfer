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

import ctypes
import functools
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import register_custom_op

# NOTE(Zihao): we should use cuda-python instead of ctypes cuda runtime bindings.
# However, cuda-python's API is not stable yet, so we use ctypes bindings instead.
# which is copied from vllm codebase.


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


# for trtllm_custom_all_reduce
class AllReduceStrategyType:
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6


# for trtllm_custom_all_reduce
class AllReduceStrategyConfig:
    USE_MEMCPY = 1 << 0
    PUSH_MODE = 1 << 1


# for trtllm_custom_all_reduce
class AllReduceFusionOp:
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8


class cudaIpcMemHandle_t(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: List[Any]


def find_loaded_library(lib_name) -> Optional[str]:
    """
    According to according to https://man7.org/linux/man-pages/man5/proc_pid_maps.5.html,
    the file `/proc/self/maps` contains the memory maps of the process, which includes the
    shared libraries loaded by the process. We can use this file to find the path of the
    a loaded library.
    """  # noqa
    found = False
    with open("/proc/self/maps") as f:
        for line in f:
            if lib_name in line:
                found = True
                break
    if not found:
        # the library is not loaded in the current process
        return None
    # if lib_name is libcudart, we need to match a line with:
    # address /path/to/libcudart-hash.so.11.0
    start = line.index("/")
    path = line[start:].strip()
    filename = path.split("/")[-1]
    assert filename.rpartition(".so")[0].startswith(
        lib_name
    ), f"Unexpected filename: {filename} for library {lib_name}"
    return path


class CudaRTLibrary:
    exported_functions = [
        # ​cudaError_t cudaSetDevice ( int  device )
        Function("cudaSetDevice", cudaError_t, [ctypes.c_int]),
        # cudaError_t   cudaDeviceSynchronize ( void )
        Function("cudaDeviceSynchronize", cudaError_t, []),
        # ​cudaError_t cudaDeviceReset ( void )
        Function("cudaDeviceReset", cudaError_t, []),
        # const char*   cudaGetErrorString ( cudaError_t error )
        Function("cudaGetErrorString", ctypes.c_char_p, [cudaError_t]),
        # ​cudaError_t    cudaMalloc ( void** devPtr, size_t size )
        Function(
            "cudaMalloc",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t],
        ),
        # ​cudaError_t    cudaFree ( void* devPtr )
        Function("cudaFree", cudaError_t, [ctypes.c_void_p]),
        # ​cudaError_t cudaMemset ( void* devPtr, int  value, size_t count )
        Function(
            "cudaMemset", cudaError_t, [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        ),
        # ​cudaError_t cudaMemcpy ( void* dst, const void* src, size_t count, cudaMemcpyKind kind ) # noqa
        Function(
            "cudaMemcpy",
            cudaError_t,
            [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, cudaMemcpyKind],
        ),
        # cudaError_t cudaIpcGetMemHandle ( cudaIpcMemHandle_t* handle, void* devPtr ) # noqa
        Function(
            "cudaIpcGetMemHandle",
            cudaError_t,
            [ctypes.POINTER(cudaIpcMemHandle_t), ctypes.c_void_p],
        ),
        # ​cudaError_t cudaIpcOpenMemHandle ( void** devPtr, cudaIpcMemHandle_t handle, unsigned int  flags ) # noqa
        Function(
            "cudaIpcOpenMemHandle",
            cudaError_t,
            [ctypes.POINTER(ctypes.c_void_p), cudaIpcMemHandle_t, ctypes.c_uint],
        ),
    ]

    # class attribute to store the mapping from the path to the library
    # to avoid loading the same library multiple times
    path_to_library_cache: Dict[str, Any] = {}

    # class attribute to store the mapping from library path
    #  to the corresponding dictionary
    path_to_dict_mapping: Dict[str, Dict[str, Any]] = {}

    def __init__(self, so_file: Optional[str] = None):
        if so_file is None:
            so_file = find_loaded_library("libcudart")
            assert so_file is not None, "libcudart is not loaded in the current process"
        if so_file not in CudaRTLibrary.path_to_library_cache:
            lib = ctypes.CDLL(so_file)
            CudaRTLibrary.path_to_library_cache[so_file] = lib
        self.lib = CudaRTLibrary.path_to_library_cache[so_file]

        if so_file not in CudaRTLibrary.path_to_dict_mapping:
            _funcs = {}
            for func in CudaRTLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            CudaRTLibrary.path_to_dict_mapping[so_file] = _funcs
        self.funcs = CudaRTLibrary.path_to_dict_mapping[so_file]

    def CUDART_CHECK(self, result: cudaError_t) -> None:
        if result != 0:
            error_str = self.cudaGetErrorString(result)
            raise RuntimeError(f"CUDART error: {error_str}")

    def cudaGetErrorString(self, error: cudaError_t) -> str:
        return self.funcs["cudaGetErrorString"](error).decode("utf-8")

    def cudaSetDevice(self, device: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaSetDevice"](device))

    def cudaDeviceSynchronize(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceSynchronize"]())

    def cudaDeviceReset(self) -> None:
        self.CUDART_CHECK(self.funcs["cudaDeviceReset"]())

    def cudaMalloc(self, size: int) -> ctypes.c_void_p:
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(self.funcs["cudaMalloc"](ctypes.byref(devPtr), size))
        return devPtr

    def cudaFree(self, devPtr: ctypes.c_void_p) -> None:
        self.CUDART_CHECK(self.funcs["cudaFree"](devPtr))

    def cudaMemset(self, devPtr: ctypes.c_void_p, value: int, count: int) -> None:
        self.CUDART_CHECK(self.funcs["cudaMemset"](devPtr, value, count))

    def cudaMemcpy(
        self, dst: ctypes.c_void_p, src: ctypes.c_void_p, count: int
    ) -> None:
        cudaMemcpyDefault = 4
        kind = cudaMemcpyDefault
        self.CUDART_CHECK(self.funcs["cudaMemcpy"](dst, src, count, kind))

    def cudaIpcGetMemHandle(self, devPtr: ctypes.c_void_p) -> cudaIpcMemHandle_t:
        handle = cudaIpcMemHandle_t()
        self.CUDART_CHECK(
            self.funcs["cudaIpcGetMemHandle"](ctypes.byref(handle), devPtr)
        )
        return handle

    def cudaIpcOpenMemHandle(self, handle: cudaIpcMemHandle_t) -> ctypes.c_void_p:
        cudaIpcMemLazyEnablePeerAccess = 1
        devPtr = ctypes.c_void_p()
        self.CUDART_CHECK(
            self.funcs["cudaIpcOpenMemHandle"](
                ctypes.byref(devPtr), handle, cudaIpcMemLazyEnablePeerAccess
            )
        )
        return devPtr


cudart = CudaRTLibrary()


def gen_comm_module() -> JitSpec:
    return gen_jit_spec(
        "comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR / "custom_all_reduce.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_allreduce.cu",
        ],
        extra_cuda_cflags=sm100a_nvcc_flags,
    )


@functools.cache
def get_comm_module():
    module = gen_comm_module().build_and_load()

    # torch library for all
    @register_custom_op(
        "flashinfer::init_custom_ar",
        mutates_args=["ipc_ptrs", "rank_data", "rank", "full_nvlink"],
    )
    def init_custom_ar(
        ipc_ptrs: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
    ) -> int:
        return module.init_custom_ar(ipc_ptrs, rank_data, rank, full_nvlink)

    @register_custom_op("flashinfer::dispose", mutates_args=["fa"])
    def dispose(fa: int) -> None:
        module.dispose(fa)

    @register_custom_op("flashinfer::get_graph_buffer_ipc_meta", mutates_args=["fa"])
    def get_graph_buffer_ipc_meta(fa: int) -> Tuple[List[int], List[int]]:
        return module.get_graph_buffer_ipc_meta(fa)

    @register_custom_op(
        "flashinfer::register_buffer", mutates_args=["fa", "fake_ipc_ptrs"]
    )
    def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
        return module.register_buffer(fa, fake_ipc_ptrs)

    @register_custom_op(
        "flashinfer::register_graph_buffers",
        mutates_args=["fa", "handles", "offsets"],
    )
    def register_graph_buffers(
        fa: int, handles: List[List[int]], offsets: List[List[int]]
    ) -> None:
        module.register_graph_buffers(fa, handles, offsets)

    @register_custom_op("flashinfer::meta_size", mutates_args=[])
    def meta_size() -> int:
        return module.meta_size()

    @register_custom_op(
        "flashinfer::all_reduce",
        mutates_args=["out", "reg_buffer", "reg_buffer_sz_bytes"],
    )
    def all_reduce(
        fa: int,
        inp: torch.Tensor,
        out: torch.Tensor,
        reg_buffer: int,
        reg_buffer_sz_bytes: int,
        num_ctas: int,
    ) -> None:
        module.all_reduce(fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas)

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize", mutates_args=["buffer"]
    )
    def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None:
        module.trtllm_lamport_initialize(buffer_ptr, size, dtype)

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize_all",
        mutates_args=["buffer_0_ptr", "buffer_1_ptr", "buffer_2_ptr", "size", "dtype"],
    )
    def trtllm_lamport_initialize_all(
        buffer_0_ptr: int, buffer_1_ptr: int, buffer_2_ptr: int, size: int, dtype: torch.dtype
    ) -> None:
        module.trtllm_lamport_initialize_all(buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype)

    @register_custom_op(
        "flashinfer::trtllm_custom_all_reduce", mutates_args=["buffer", "inp", "out"]
    )
    def trtllm_custom_all_reduce(
        buffer: torch.Tensor,
        inp: torch.Tensor,
        out: torch.Tensor,
        tp_size: int,
        tp_rank: int,
        token_num: int,
        fusion_op_code: AllReduceFusionOp,
        strategy_code: AllReduceStrategyType,
        config_code: AllReduceStrategyConfig,
        launch_with_pdl: bool,
        flag_value: int,
        peer_comm_buffer_ptrs: torch.Tensor,
        peer_barrier_ptrs_in: torch.Tensor,
        peer_barrier_ptrs_out: torch.Tensor,
        bias: Optional[torch.Tensor],
        residual: Optional[torch.Tensor],
        weight: Optional[torch.Tensor],
        weight_pre_residual_norm: Optional[torch.Tensor],
        eps: Optional[float],
        intermediate_buffer: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_0: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_1: Optional[torch.Tensor],
        lamport_peer_comm_buffer_ptrs_2: Optional[torch.Tensor],
    ) -> None:
        module.trtllm_custom_all_reduce(
            buffer,
            inp,
            out,
            tp_size,
            tp_rank,
            token_num,
            fusion_op_code,
            strategy_code,
            config_code,
            launch_with_pdl,
            flag_value,
            peer_comm_buffer_ptrs,
            peer_barrier_ptrs_in,
            peer_barrier_ptrs_out,
            bias,
            residual,
            weight,
            weight_pre_residual_norm,
            eps,
            intermediate_buffer,
            lamport_peer_comm_buffer_ptrs_0,
            lamport_peer_comm_buffer_ptrs_1,
            lamport_peer_comm_buffer_ptrs_2,
        )

    return SimpleNamespace(
        init_custom_ar=init_custom_ar,
        dispose=dispose,
        get_graph_buffer_ipc_meta=get_graph_buffer_ipc_meta,
        register_buffer=register_buffer,
        register_graph_buffers=register_graph_buffers,
        meta_size=meta_size,
        all_reduce=all_reduce,
        trtllm_lamport_initialize=trtllm_lamport_initialize,
        trtllm_lamport_initialize_all=trtllm_lamport_initialize_all,
        trtllm_custom_all_reduce=trtllm_custom_all_reduce,
    )


def init_custom_ar(
    ipc_tensors: List[int], rank_data: torch.Tensor, rank: int, full_nvlink: bool
) -> int:
    return get_comm_module().init_custom_ar(ipc_tensors, rank_data, rank, full_nvlink)


def dispose(fa: int) -> None:
    get_comm_module().dispose(fa)


def all_reduce(
    fa: int,
    inp: torch.Tensor,
    out: torch.Tensor,
    reg_buffer: int,
    reg_buffer_sz_bytes: int,
    num_ctas: int,
) -> None:
    """Performs an out-of-place all reduce.

    Args:
        fa: The handle to the custom all reduce.
        inp: The input tensor to all reduce.
        out: The output tensor to all reduce.
        reg_buffer: The register buffer to all reduce.
        reg_buffer_sz_bytes: The size of the register buffer.
        num_ctas: The number of CTAs to use for the all reduce.
        CTA upper bounds: 36. Generally, we can saturate the bandwidth even with small amount the SMs.
    """
    get_comm_module().all_reduce(
        fa, inp, out, reg_buffer, reg_buffer_sz_bytes, num_ctas
    )


def get_graph_buffer_ipc_meta(fa) -> Tuple[List[int], List[int]]:
    return get_comm_module().get_graph_buffer_ipc_meta(fa)


def register_buffer(fa: int, fake_ipc_ptrs: List[int]) -> None:
    return get_comm_module().register_buffer(fa, fake_ipc_ptrs)


def register_graph_buffers(
    fa: int, handles: List[List[int]], offsets: List[List[int]]
) -> None:
    get_comm_module().register_graph_buffers(fa, handles, offsets)


def meta_size() -> int:
    return get_comm_module().meta_size()


def create_shared_buffer(
    size_in_bytes: int, group: Optional[ProcessGroup] = None
) -> List[int]:
    pointer = cudart.cudaMalloc(size_in_bytes)
    handle = cudart.cudaIpcGetMemHandle(pointer)
    if group is None:
        group = dist.group.WORLD
    world_size = dist.get_world_size(group=group)
    rank = dist.get_rank(group=group)

    handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
    input_tensor = torch.tensor(bytearray(handle_bytes), dtype=torch.uint8).to(
        f"cuda:{rank}"
    )
    gathered_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, input_tensor, group=group)

    handles = []
    handle_type = type(handle)
    for tensor in gathered_tensors:
        bytes_data = tensor.cpu().numpy().tobytes()
        handle_obj = handle_type()
        ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
        handles.append(handle_obj)

    pointers: List[int] = []
    for i, h in enumerate(handles):
        if i == rank:
            pointers.append(pointer.value)
        else:
            try:
                opened_ptr = cudart.cudaIpcOpenMemHandle(h)
                pointers.append(opened_ptr.value)
            except Exception as e:
                print(f"Rank {rank}: Failed to open IPC handle from rank {i}: {e}")
                raise

    dist.barrier(group=group)
    return pointers


def free_shared_buffer(
    pointers: List[int], group: Optional[ProcessGroup] = None
) -> None:
    if group is None:
        group = dist.group.WORLD
    rank = dist.get_rank(group=group)
    if pointers and len(pointers) > rank and pointers[rank] is not None:
        cudart.cudaFree(ctypes.c_void_p(pointers[rank]))
    dist.barrier(group=group)


# NOTE(Yingyi): The customAllReduce and allReduceFusion require different buffer size
# since allreduceFusion kernels are an improved implementation
# todo(review): align the ipc buffer size to tllm implementation
OneShotMaxToken = 128
MAX_ALL_REDUCE_BLOCKS = 24
LamportTokenNumThreshold = 16


def trtllm_create_ipc_buffer_for_all_reduce(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    '''
    Note:
    We would init 7 IPC buffers for trtllm_custom_all_reduce.
    They are sized as follows:
    [buffer_size, buffer_size, flag_size, flag_size, lamport_buffer_size, lamport_buffer_size, lamport_buffer_size]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * 4  # size of float
    - flag_size: (MAX_ALL_REDUCE_BLOCKS + 1) * 4  # size of uint32_t
    - lamport_buffer_size: tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * 2  # size of half

    They are for:
    ipcHandles[0] - peer_comm_buffer_ptrs
    ipcHandles[2] - peer_barrier_ptrs_in
    ipcHandles[3] - peer_barrier_ptrs_out
    ipcHandles[4] - lamport_peer_comm_buffer_ptrs[0:tp_size]
    ipcHandles[5] - lamport_peer_comm_buffer_ptrs[tp_size:tp_size * 2]
    ipcHandles[6] - lamport_peer_comm_buffer_ptrs[tp_size * 2:tp_size * 3]

    We use tp_size and world_size here interchangeably (customAllReduce).
    '''

    # NOTE(Yingyi): refer to tllm
    # - cpp/tests/unit_tests/kernels/allReduce/allReduceKernelTest.cu, Workspace init

    buffer_size = tp_size * max_token_num * hidden_dim * 4  # size of float
    FLAG_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * 4  # size of uint32_t
    flag_size = FLAG_SIZE * tp_size * 2
    lamport_buffer_size = (
        tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * 2
    )  # size of half

    ipc_handles = list()

    for size in [
        buffer_size,
        buffer_size,
        flag_size,
        flag_size,
        lamport_buffer_size,
        lamport_buffer_size,
        lamport_buffer_size,
    ]:
        ipc_handles.append(create_shared_buffer(size, group))

    # todo(yingyi): init flag to be 0 - move to c++ side
    # todo(yingyi): init lamport buffer to be negative zero - move to outer init flow
    # note(yingyi): maintain local buffer - unused in tllm, skip

    return ipc_handles

# todo(review): align the ipc buffer size to tllm implementation
BarrierFlagCount = 256


def trtllm_create_ipc_buffer_for_all_reduce_fusion(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    # NOTE(Yingyi): refer to tllm
    # - cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu, Workspace init

    buffer_size = tp_size * max_token_num * hidden_dim * 2  # size of half
    flag_size = tp_size * BarrierFlagCount * 4  # size of int
    lamport_comm_size = (
        tp_size * max(max_token_num, OneShotMaxToken) * hidden_dim * 2
    )  # size of half
    lamport_buffer_size = lamport_comm_size * 3

    # we should init 3 buffers for all reduce fusion:
    # [buffer_size, flag_size, lamport_buffer_size]

    ipc_handles = list()
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        ipc_handles.append(create_shared_buffer(size, group))

    return ipc_handles


# @dataclass
# class AllReduceFusionParams:
#     bias_buffer: int
#     residual_buffer: int
#     hidden_size: int
#     weight_buffer: int
#     weight_buffer_pre_residual_norm: int
#     eps: float
#     intermediate_buffer: int
#     lamport_peer_comm_buffer_ptrs: List[int]

# @dataclass
# class AllReduceParams:
#     elts_total: int
#     elts_per_rank: int
#     elts_per_block: int
#     rank_offset: int
#     ranks_per_node: int
#     local_rank: int
#     barrier_flag: int
#     peer_barrier_ptrs_in: List[int]
#     peer_barrier_ptrs_out: List[int]
#     peer_comm_buffer_ptrs: List[int]
#     local_output_buffer_ptr: int
#     local_input_buffer_ptr: int
#     fusion_params: AllReduceFusionParams

# def trtllm_create_all_reduce_params(
#     message_size: int,
#     world_size: int,
#     local_rank: int,
#     local_input_buffer_ptr: int,
#     local_output_buffer_ptr: int,
#     ipc_handles: List[List[int]],
#     fusion_bias_buffer_ptr: int = None,
#     fusion_residual_buffer_ptr: int = None,
#     fusion_weight_buffer_ptr: int = None,
#     fusion_weight_pre_residual_norm_buffer_ptr: int = None,
#     fusion_eps: float = None,
#     fusion_intermediate_buffer_ptr: int = None,
#     fusion_lamport_peer_comm_buffer_ptrs: List[int] = None,
# ) -> AllReduceParams:
#     all_reduce_params = AllReduceParams()
# all_reduce_params.elts_total = message_size
# all_reduce_params.ranks_per_node = world_size
# all_reduce_params.local_rank = local_rank
# all_reduce_params.local_output_buffer_ptr = local_output_buffer_ptr # in.data_ptr()
# all_reduce_params.local_input_buffer_ptr = local_input_buffer_ptr # out.data_ptr()

# fusion params
# all_reduce_params.fusion_params.bias_buffer = fusion_bias_buffer_ptr # bias.data_ptr()
# all_reduce_params.fusion_params.residual_buffer = fusion_residual_buffer_ptr # residual.data_ptr()
# all_reduce_params.fusion_params.weight_buffer = fusion_weight_buffer_ptr # weight.data_ptr()
# all_reduce_params.fusion_params.weight_buffer_pre_residual_norm = fusion_weight_pre_residual_norm_buffer_ptr # weight_pre_residual_norm.data_ptr()
# all_reduce_params.fusion_params.eps = fusion_eps # eps
# all_reduce_params.fusion_params.intermediate_buffer = fusion_intermediate_buffer_ptr # intermediate_buffer.data_ptr()

# ipc buffer pointers
# for i in range(world_size):
#     all_reduce_params.peer_comm_buffer_ptrs.append(ipc_handles[0][i])
#     all_reduce_params.peer_barrier_ptrs_in.append(ipc_handles[2][i])
#     all_reduce_params.peer_barrier_ptrs_out.append(ipc_handles[3][i])

#     # world size was MAX_RANKS_PER_NODE = 16
#     all_reduce_params.fusion_params.lamport_peer_comm_buffer_ptrs = [0] * (world_size * 3)
#     all_reduce_params.fusion_params.lamport_peer_comm_buffer_ptrs[i] = ipc_handles[4][i] # peer_comm_buffer_ptrs[i]
#     all_reduce_params.fusion_params.lamport_peer_comm_buffer_ptrs[i + world_size] = ipc_handles[5][i] # peer_comm_buffer_ptrs[i + world_size]
#     all_reduce_params.fusion_params.lamport_peer_comm_buffer_ptrs[i + world_size * 2] = ipc_handles[6][i] # peer_comm_buffer_ptrs[i + world_size * 2]

# return all_reduce_params


def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None:
    get_comm_module().trtllm_lamport_initialize(buffer_ptr, size, dtype)


def trtllm_lamport_initialize_all(
    buffer_0_ptr: int, buffer_1_ptr: int, buffer_2_ptr: int, size: int, dtype: torch.dtype
) -> None:
    get_comm_module().trtllm_lamport_initialize_all(buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype)


def trtllm_custom_all_reduce(
    inp: torch.Tensor,
    out: torch.Tensor,
    tp_size: int,
    tp_rank: int,
    token_num: int,
    fusion_op_code: AllReduceFusionOp,
    strategy_code: AllReduceStrategyType,
    config_code: AllReduceStrategyConfig,
    launch_with_pdl: bool,
    flag_value: int,
    peer_comm_buffer_ptrs: torch.Tensor,
    peer_barrier_ptrs_in: torch.Tensor,
    peer_barrier_ptrs_out: torch.Tensor,
    bias: Optional[torch.Tensor],
    residual: Optional[torch.Tensor],
    weight: Optional[torch.Tensor],
    weight_pre_residual_norm: Optional[torch.Tensor],
    eps: Optional[float],
    intermediate_buffer: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_0: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_1: Optional[torch.Tensor],
    lamport_peer_comm_buffer_ptrs_2: Optional[torch.Tensor],
) -> None:
    get_comm_module().trtllm_custom_all_reduce(
        inp,
        out,
        tp_size,
        tp_rank,
        token_num,
        fusion_op_code,
        strategy_code,
        config_code,
        launch_with_pdl,
        flag_value,
        peer_comm_buffer_ptrs,
        peer_barrier_ptrs_in,
        peer_barrier_ptrs_out,
        bias,
        residual,
        weight,
        weight_pre_residual_norm,
        eps,
        intermediate_buffer,
        lamport_peer_comm_buffer_ptrs_0,
        lamport_peer_comm_buffer_ptrs_1,
        lamport_peer_comm_buffer_ptrs_2,
    )
