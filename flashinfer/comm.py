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
from .jit import gen_jit_spec
from .utils import register_custom_op

# NOTE(Zihao): we should use cuda-python instead of ctypes cuda runtime bindings.
# However, cuda-python's API is not stable yet, so we use ctypes bindings instead.
# which is copied from vllm codebase.


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


class AllReduceStrategyType:
    # NOTE: for trtllm_custom_all_reduce
    NCCL = 0
    MIN_LATENCY = 1
    UB = 2
    AUTO = 3
    ONESHOT = 4
    TWOSHOT = 5
    LOWPRECISION = 6


class AllReduceStrategyConfig:
    # NOTE: for trtllm_custom_all_reduce
    USE_MEMCPY = 1 << 0
    PUSH_MODE = 1 << 1


class AllReduceFusionOp:
    # NOTE: for trtllm_custom_all_reduce
    NONE = 0
    RESIDUAL_RMS_NORM = 1
    LAST_PROCESS_FOR_UB = 2
    RESIDUAL_RMS_PREPOST_NORM = 3
    RESIDUAL_RMS_NORM_QUANT_FP8 = 4
    RESIDUAL_RMS_NORM_QUANT_NVFP4 = 5
    RESIDUAL_RMS_NORM_OUT_QUANT_FP8 = 6
    RESIDUAL_RMS_NORM_OUT_QUANT_NVFP4 = 7
    MOE_ALLREDUCE_RESIDUAL_RMS_NORM = 8


class FP4QuantizationSFLayout:
    # Block scale factors are stored in swizzled layout for cutlass FP4 kernel. Scale factor
    # blocks are organized in 512-byte blocks in global memory, with each block having 128x4 FP8
    # values. The SF matrix dimensions are therefore padded - rows to the nearest multiple of 128 and
    # columns to the nearest multiple of 4.
    #
    # The scale factor block rows map to data block rows in an interleaved pattern:
    # For a scale factor row 'i', it maps to data block row: (i % 4) * 32 + (i / 4)
    # Column 'j' in the scale factor block corresponds to scaling the j-th block in the data tensor.
    #
    # Please refer to https://nvbugs/4165523 for more details about the swizzled layout.
    SWIZZLED = 0
    # Block scale factors are stored in linear layout (row-major). This is used in some trtllm-gen
    # kernels standard.
    LINEAR = 1


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
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_moe_allreduce_fusion.cu",
        ],
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
    def trtllm_lamport_initialize(
        buffer_ptr: int, size: int, dtype: torch.dtype
    ) -> None:
        module.trtllm_lamport_initialize(buffer_ptr, size, dtype)

    @register_custom_op(
        "flashinfer::trtllm_lamport_initialize_all",
        mutates_args=["buffer_0_ptr", "buffer_1_ptr", "buffer_2_ptr", "size", "dtype"],
    )
    def trtllm_lamport_initialize_all(
        buffer_0_ptr: int,
        buffer_1_ptr: int,
        buffer_2_ptr: int,
        size: int,
        dtype: torch.dtype,
    ) -> None:
        module.trtllm_lamport_initialize_all(
            buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
        )

    @register_custom_op(
        "flashinfer::trtllm_custom_all_reduce",
        mutates_args=[
            "inp",
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "fusion_op_code",
            "strategy_code",
            "config_code",
            "launch_with_pdl",
            "flag_value",
            "peer_comm_buffer_ptrs",
            "peer_barrier_ptrs_in",
            "peer_barrier_ptrs_out",
            "bias",
            "residual",
            "weight",
            "weight_pre_residual_norm",
            "eps",
            "intermediate_buffer",
            "lamport_peer_comm_buffer_ptrs_0",
            "lamport_peer_comm_buffer_ptrs_1",
            "lamport_peer_comm_buffer_ptrs_2",
        ],
    )
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
        module.trtllm_custom_all_reduce(
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

    @register_custom_op(
        "flashinfer::trtllm_moe_allreduce_fusion",
        mutates_args=[
            "inp",
            "out",
            "tp_size",
            "tp_rank",
            "token_num",
            "hidden_dim",
            "workspace_ptr",
            "launch_with_pdl",
            "residual_in",
            "rms_gamma",
            "rms_eps",
            "scale_factor",
            "moe_reduction_device_num_experts",
            "moe_reduction_scale_input",
            "moe_reduction_active_experts_token_input",
            "moe_reduction_token_input",
            "layout_code",
            "residual_out",
            "norm_out",
            "quant_out",
            "scale_out",
        ],
    )
    def trtllm_moe_allreduce_fusion(
        inp: torch.Tensor,
        world_size: int,
        world_rank: int,
        token_num: int,
        hidden_dim: int,
        workspace_ptr: int,
        launch_with_pdl: bool,
        residual_in: torch.Tensor,
        rms_gamma: torch.Tensor,
        rms_eps: float,
        scale_factor: float,
        moe_reduction_device_num_experts: int,
        moe_reduction_scale_input: torch.Tensor,
        moe_reduction_active_experts_token_input: torch.Tensor,
        moe_reduction_token_input: torch.Tensor,
        layout_code: Optional[FP4QuantizationSFLayout],
        residual_out: Optional[torch.Tensor],
        norm_out: Optional[torch.Tensor],
        quant_out: Optional[torch.Tensor],
        scale_out: Optional[torch.Tensor],
    ) -> None:
        module.trtllm_moe_allreduce_fusion(
            inp,
            world_size,
            world_rank,
            token_num,
            hidden_dim,
            workspace_ptr,
            launch_with_pdl,
            residual_in,
            rms_gamma,
            rms_eps,
            scale_factor,
            moe_reduction_device_num_experts,
            moe_reduction_scale_input,
            moe_reduction_active_experts_token_input,
            moe_reduction_token_input,
            layout_code,
            residual_out,
            norm_out,
            quant_out,
            scale_out,
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
        trtllm_moe_allreduce_fusion=trtllm_moe_allreduce_fusion,
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
OneShotMaxToken = 128
MAX_ALL_REDUCE_BLOCKS = 24
LamportTokenNumThreshold = 16


def trtllm_create_ipc_workspace_for_all_reduce(
    rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    """
    Note:
    This function is used to create a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be initialized before calling trtllm_custom_all_reduce.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.

    We would init 7 IPC buffers for trtllm_custom_all_reduce.
    They are sized as follows:
    [buffer_size, buffer_size, flag_size, flag_size, lamport_buffer_size, lamport_buffer_size, lamport_buffer_size]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(float) * (maxBeamWidth)
    - flag_size: (MAX_ALL_REDUCE_BLOCKS + 1) * sizeof(uint32_t) * tp_size * 2
    - lamport_buffer_size: tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * sizeof(half)

    They are for:
    ipcHandles[0] - peer_comm_buffer_ptrs
    ipcHandles[2] - peer_barrier_ptrs_in
    ipcHandles[3] - peer_barrier_ptrs_out
    ipcHandles[4] - lamport_peer_comm_buffer_ptrs[0:tp_size]
    ipcHandles[5] - lamport_peer_comm_buffer_ptrs[tp_size:tp_size * 2]
    ipcHandles[6] - lamport_peer_comm_buffer_ptrs[tp_size * 2:tp_size * 3]

    We use tp_size and world_size here interchangeably (customAllReduce).

    Reference: trtllm, cpp/tests/unit_tests/kernels/allReduce/allReduceKernelTest.cu, Workspace init
    """

    buffer_size = tp_size * max_token_num * hidden_dim * 4
    FLAG_SIZE = (MAX_ALL_REDUCE_BLOCKS + 1) * 4
    flag_size = FLAG_SIZE * tp_size * 2
    lamport_buffer_size = tp_size * LamportTokenNumThreshold * tp_size * hidden_dim * 2

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

    print(
        f"rank {rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )

    return ipc_handles


def trtllm_destroy_ipc_workspace_for_all_reduce(
    workspace: List[int], group: Optional[ProcessGroup] = None
) -> None:
    """
    Note:
    This function is used to destroy a workspace for all reduce.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce.
    The workspace can be reused for multiple all reduce calls under the same configuration.
    """

    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


BarrierFlagCount = 256


def trtllm_create_ipc_workspace_for_all_reduce_fusion(
    tp_rank: int,
    tp_size: int,
    max_token_num: int,
    hidden_dim,
    group: Optional[ProcessGroup] = None,
) -> List[int]:
    """
    Note:
    We would init 3 IPC buffers for trtllm_custom_all_reduce_fusion.
    They are sized as follows:
    [buffer_size, flag_size, lamport_buffer_size * 3]
    where:
    - buffer_size: tp_size * max_token_num * hidden_dim * sizeof(half)
    - flag_size: tp_size * BarrierFlagCount * sizeof(int)
    - lamport_buffer_size: tp_size * max(max_token_num, OneShotMaxToken) * tp_size * hidden_dim * sizeof(half)

    The workspace is passed as workspace field in AllReduceFusionParams.

    We use tp_size and world_size here interchangeably (allReduceFusion).

    Reference: trtllm, cpp/tensorrt_llm/kernels/communicationKernels/allReduceWorkspace.cu, Workspace init
    """

    buffer_size = tp_size * max_token_num * hidden_dim * 2
    flag_size = tp_size * BarrierFlagCount * 4
    lamport_comm_size = tp_size * max(max_token_num, OneShotMaxToken) * hidden_dim * 2
    lamport_buffer_size = lamport_comm_size * 3

    # we should init 3 buffers for all reduce fusion:
    # [buffer_size, flag_size, lamport_buffer_size]

    ipc_handles = list()
    for size in [buffer_size, flag_size, lamport_buffer_size]:
        ipc_handles.append(create_shared_buffer(size, group))

    print(
        f"rank {tp_rank} allocated ipc_handles: {[[hex(handle) for handle in sublist] for sublist in ipc_handles]}"
    )

    # initialize workspace
    workspace = list()
    # add ipc handles to workspace
    for ipc_handle in ipc_handles:
        for rank in range(tp_size):
            workspace.append(ipc_handle[rank])

    # add flags to workspace
    """
    NOTE:
    The flags are for the lamport communication states.
    atomic flag read counter: kernel_flag_ptr[0] = 0;
    non-lamport flag: kernel_flag_ptr[1] = 0;
    lamport flag: kernel_flag_ptr[2] = 0;
    lamport triple buffer offset: kernel_flag_ptr[3] = lamport_comm_size;
    lamport clear size: kernel_flag_ptr[4] = 0;
    """
    # malloc cuda memory of int32_t * 5
    flag_ptr = cudart.cudaMalloc(5 * 4)
    # initialize flag values to 0
    cudart.cudaMemset(flag_ptr, 0, 5 * 4)
    # add flag_ptr to workspace
    workspace.append(flag_ptr)

    return workspace


def trtllm_destroy_ipc_workspace_for_all_reduce_fusion(
    workspace: List[int], group: Optional[ProcessGroup] = None
) -> None:
    """
    Note:
    This function is used to destroy a workspace for all reduce fusion.
    The workspace is a list of IPC handles.
    The workspace should be destroyed after calling trtllm_custom_all_reduce_fusion.
    The workspace can be reused for multiple all reduce fusion calls under the same configuration.
    """

    for ipc_handle in workspace:
        free_shared_buffer(ipc_handle, group)


def trtllm_lamport_initialize(buffer_ptr: int, size: int, dtype: torch.dtype) -> None:
    get_comm_module().trtllm_lamport_initialize(buffer_ptr, size, dtype)


def trtllm_lamport_initialize_all(
    buffer_0_ptr: int,
    buffer_1_ptr: int,
    buffer_2_ptr: int,
    size: int,
    dtype: torch.dtype,
) -> None:
    get_comm_module().trtllm_lamport_initialize_all(
        buffer_0_ptr, buffer_1_ptr, buffer_2_ptr, size, dtype
    )


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


def trtllm_moe_allreduce_fusion(
    inp: torch.Tensor,
    world_size: int,
    world_rank: int,
    token_num: int,
    hidden_dim: int,
    workspace_ptr: int,
    launch_with_pdl: bool,
    residual_in: torch.Tensor,
    rms_gamma: torch.Tensor,
    rms_eps: float,
    scale_factor: float,
    moe_reduction_device_num_experts: int,
    moe_reduction_scale_input: torch.Tensor,
    moe_reduction_active_experts_token_input: torch.Tensor,
    moe_reduction_token_input: torch.Tensor,
    layout_code: Optional[FP4QuantizationSFLayout],
    residual_out: Optional[torch.Tensor],
    norm_out: Optional[torch.Tensor],
    quant_out: Optional[torch.Tensor],
    scale_out: Optional[torch.Tensor],
) -> None:
    get_comm_module().trtllm_moe_allreduce_fusion(
        inp,
        world_size,
        world_rank,
        token_num,
        hidden_dim,
        workspace_ptr,
        launch_with_pdl,
        residual_in,
        rms_gamma,
        rms_eps,
        scale_factor,
        moe_reduction_device_num_experts,
        moe_reduction_scale_input,
        moe_reduction_active_experts_token_input,
        moe_reduction_token_input,
        layout_code,
        residual_out,
        norm_out,
        quant_out,
        scale_out,
    )
