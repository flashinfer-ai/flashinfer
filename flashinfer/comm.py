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
import threading
from dataclasses import dataclass
from enum import IntEnum
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from flashinfer.mapping import Mapping

from .jit import JitSpec
from .jit import env as jit_env
from .jit import gen_jit_spec, sm100a_nvcc_flags
from .utils import register_custom_op

# NOTE(Zihao): we should use cuda-python instead of ctypes cuda runtime bindings.
# However, cuda-python's API is not stable yet, so we use ctypes bindings instead.
# which is copied from vllm codebase.


cudaError_t = ctypes.c_int
cudaMemcpyKind = ctypes.c_int


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


def get_mpi_include_lib_path():
    import pathlib
    import shlex
    import subprocess

    cmd = ["mpicc", "-show"]
    output = subprocess.check_output(cmd, text=True)
    # Parse the output to extract include and library paths
    parts = shlex.split(output)
    include_dirs = []
    lib_dirs = []

    i = 0
    while i < len(parts):
        if parts[i] == "-I" and i + 1 < len(parts):
            include_dirs.append(pathlib.Path(parts[i + 1]))
            i += 2
        elif parts[i].startswith("-I"):
            include_dirs.append(pathlib.Path(parts[i][2:]))
            i += 1
        elif parts[i] == "-L" and i + 1 < len(parts):
            lib_dirs.append(pathlib.Path(parts[i + 1]))
            i += 2
        elif parts[i].startswith("-L"):
            lib_dirs.append(pathlib.Path(parts[i][2:]))
            i += 1
        else:
            i += 1

    # Return the first include directory found, or None if none found
    include_dir = include_dirs[0] if include_dirs else None

    return include_dir, lib_dirs


def gen_comm_module() -> JitSpec:
    mpi_include_path, mpi_lib_path = get_mpi_include_lib_path()
    print(mpi_include_path, mpi_lib_path)
    return gen_jit_spec(
        "comm",
        [
            jit_env.FLASHINFER_CSRC_DIR / "flashinfer_comm_ops.cu",
            jit_env.FLASHINFER_CSRC_DIR / "custom_all_reduce.cu",
            # jit_env.FLASHINFER_CSRC_DIR / "trtllm_comm/customAllReduceKernels.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_comm/allReduceFusionKernels.cu",
            jit_env.FLASHINFER_CSRC_DIR / "trtllm_comm/moeAllReduceFusionKernels.cu",
        ],
        extra_include_paths=[mpi_include_path],
        extra_ldflags=[f"-L{mpi_lib_path}", "-lmpi"],
        extra_cflags=["-DENABLE_MULTI_DEVICE"],
        extra_cuda_cflags=sm100a_nvcc_flags + ["-DENABLE_MULTI_DEVICE"],
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
        "flashinfer::allreduce",
        mutates_args=[
            "input",
            "residual",
            "norm_weight",
            "scale",
            "bias",
            "workspace",
            "group",
        ],
    )
    def allreduce(
        input: torch.Tensor,
        residual: Optional[torch.Tensor],
        norm_weight: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        bias: Optional[torch.Tensor],
        workspace: Optional[torch.Tensor],
        group: List[int],
        strategy: int,
        op: int,
        eps: float,
    ) -> List[torch.Tensor]:
        """Performs an all-reduce operation with optional fusion patterns.

        Args:
            input: Input tensor to all-reduce
            residual: Optional residual tensor for fusion
            norm_weight: Optional normalization weights
            scale: Optional scale factors for quantization
            bias: Optional bias tensor
            workspace: Optional workspace tensor
            group: List of ranks in the process group
            strategy: AllReduceStrategy type (0=NCCL, 1=MIN_LATENCY, etc)
            op: AllReduceFusionOp type (0=NONE, 1=RESIDUAL_RMS_NORM, etc)
            eps: Epsilon value for normalization

        Returns:
            List of output tensors depending on the fusion pattern
        """
        return module.allreduce(
            input,
            residual,
            norm_weight,
            scale,
            bias,
            workspace,
            group,
            strategy,
            op,
            eps,
        )

    @register_custom_op(
        "flashinfer::moe_allreduce",
        mutates_args=[
            "residual",
            "norm_weight",
            "device_num_experts",
            "scale_input",
            "active_experts_token_input",
            "token_input",
            "workspace",
        ],
    )
    def moe_allreduce(
        residual: torch.Tensor,
        norm_weight: torch.Tensor,
        device_num_experts: torch.Tensor,
        scale_input: torch.Tensor,
        active_experts_token_input: torch.Tensor,
        token_input: torch.Tensor,
        workspace: torch.Tensor,
        rank: int,
        nranks: int,
        eps: float,
    ) -> List[torch.Tensor]:
        """Performs MoE reduction + all-reduce operation with fusion.

        Args:
            residual: Residual tensor [m, hidden_dim]
            norm_weight: Normalization weights [hidden_dim]
            device_num_experts: Number of experts per device [1]
            scale_input: Scale factors [global_num_experts, m]
            active_experts_token_input: Active expert tokens [device_num_experts, m, hidden_dim]
            token_input: Input tokens [m, hidden_dim]
            workspace: Workspace tensor
            rank: Current process rank
            nranks: Total number of ranks
            eps: Epsilon for normalization

        Returns:
            List containing [norm_out, residual_out]
        """
        return module.moe_allreduce(
            residual,
            norm_weight,
            device_num_experts,
            scale_input,
            active_experts_token_input,
            token_input,
            workspace,
            rank,
            nranks,
            eps,
        )

    return SimpleNamespace(
        init_custom_ar=init_custom_ar,
        dispose=dispose,
        get_graph_buffer_ipc_meta=get_graph_buffer_ipc_meta,
        register_buffer=register_buffer,
        register_graph_buffers=register_graph_buffers,
        meta_size=meta_size,
        all_reduce=all_reduce,  # vllm
        allreduce=allreduce,  # trtllm
        moe_allreduce=moe_allreduce,  # trtllm
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


def allreduce(
    input: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    norm_weight: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    bias: Optional[torch.Tensor] = None,
    workspace: Optional[torch.Tensor] = None,
    group: Optional[List[int]] = None,
    strategy: int = 0,  # NCCL by default
    op: int = 0,  # NONE by default
    eps: float = 1e-6,
) -> List[torch.Tensor]:
    """Performs an all-reduce operation with optional fusion patterns. (from trtllm)

    Args:
        input: Input tensor to all-reduce
        residual: Optional residual tensor for fusion
        norm_weight: Optional normalization weights
        scale: Optional scale factors for quantization
        bias: Optional bias tensor
        workspace: Optional workspace tensor
        group: List of ranks in the process group (defaults to all ranks)
        strategy: AllReduceStrategy type (0=NCCL, 1=MIN_LATENCY, etc)
        op: AllReduceFusionOp type (0=NONE, 1=RESIDUAL_RMS_NORM, etc)
        eps: Epsilon value for normalization

    Returns:
        List of output tensors depending on the fusion pattern
    """
    if group is None:
        group = list(range(dist.get_world_size()))
    return get_comm_module().allreduce(
        input, residual, norm_weight, scale, bias, workspace, group, strategy, op, eps
    )


def moe_allreduce(
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    device_num_experts: torch.Tensor,
    scale_input: torch.Tensor,
    active_experts_token_input: torch.Tensor,
    token_input: torch.Tensor,
    workspace: torch.Tensor,
    rank: Optional[int] = None,
    nranks: Optional[int] = None,
    eps: float = 1e-6,
) -> List[torch.Tensor]:
    """Performs MoE reduction + all-reduce operation with fusion. (from trtllm)

    Args:
        residual: Residual tensor [m, hidden_dim]
        norm_weight: Normalization weights [hidden_dim]
        device_num_experts: Number of experts per device [1]
        scale_input: Scale factors [global_num_experts, m]
        active_experts_token_input: Active expert tokens [device_num_experts, m, hidden_dim]
        token_input: Input tokens [m, hidden_dim]
        workspace: Workspace tensor
        rank: Current process rank (defaults to current rank)
        nranks: Total number of ranks (defaults to world size)
        eps: Epsilon for normalization

    Returns:
        List containing [norm_out, residual_out]
    """
    if rank is None:
        rank = dist.get_rank()
    if nranks is None:
        nranks = dist.get_world_size()
    return get_comm_module().moe_allreduce(
        residual,
        norm_weight,
        device_num_experts,
        scale_input,
        active_experts_token_input,
        token_input,
        workspace,
        rank,
        nranks,
        eps,
    )
