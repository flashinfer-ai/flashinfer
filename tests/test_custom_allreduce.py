# flashinfer: adapted from sglang + vllm
# refer to sgl-kernel/tests/test_custom_allreduce.py from sglang

import ctypes
import multiprocessing as mp
import random
import socket
import unittest
from typing import Any, List, Optional

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

import flashinfer.custom_all_reduce as custom_ops

# Adapted from https://github.com/vllm-project/vllm/blob/v0.6.4.post1/vllm/distributed/device_communicators/cuda_wrapper.py

"""This file is a pure Python wrapper for the cudart library.
It avoids the need to compile a separate shared library, and is
convenient for use when we just need to call a few functions.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# this line makes it possible to directly load `libcudart.so` using `ctypes`
import torch  # noqa

logger = logging.getLogger(__name__)

# === export types and functions from cudart to Python ===
# for the original cudart definition, please check
# https://docs.nvidia.com/cuda/cuda-runtime-api/index.html

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


def _run_correctness_worker(
    world_size, rank, distributed_init_port, test_sizes, num_ctas_list
):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    distributed_init_method = f"tcp://localhost:{distributed_init_port}"
    dist.init_process_group(
        backend="nccl",
        init_method=distributed_init_method,
        rank=rank,
        world_size=world_size,
    )
    group = dist.group.WORLD

    try:
        device = torch.device(f"cuda:{rank}")
        max_size = 8192 * 1024
        meta_ptrs = TestCustomAllReduce.create_shared_buffer(
            custom_ops.meta_size() + max_size, group=group
        )

        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device=device)
        buffer_ptrs = TestCustomAllReduce.create_shared_buffer(max_size, group=group)

        custom_ptr = custom_ops.init_custom_ar(meta_ptrs, rank_data, rank, True)
        custom_ops.register_buffer(custom_ptr, buffer_ptrs)

        test_loop = 10
        for sz in test_sizes:
            for num_ctas in num_ctas_list:
                for dtype in [torch.float32, torch.float16, torch.bfloat16]:
                    for _ in range(test_loop):
                        inp1 = torch.randint(1, 16, (sz,), dtype=dtype, device=device)
                        inp1_ref = inp1.clone()
                        out1 = torch.empty_like(inp1)

                        custom_ops.all_reduce(
                            custom_ptr,
                            inp1,
                            out1,
                            buffer_ptrs[rank],
                            max_size,
                            num_ctas,
                        )

                        dist.all_reduce(inp1_ref, group=group)

                        torch.testing.assert_close(out1, inp1_ref)

    finally:
        dist.barrier(group=group)
        if custom_ptr is not None:
            custom_ops.dispose(custom_ptr)
        if buffer_ptrs:
            TestCustomAllReduce.free_shared_buffer(buffer_ptrs, group)
        if meta_ptrs:
            TestCustomAllReduce.free_shared_buffer(meta_ptrs, group)

        dist.destroy_process_group(group=group)


def get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
            s.bind(("::1", 0))
            return s.getsockname()[1]


def multi_process_parallel(
    world_size: int, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert (
            procs[i].exitcode == 0
        ), f"Process {i} failed with exit code {procs[i].exitcode}"


class TestCustomAllReduce(unittest.TestCase):
    test_sizes = [
        512,
        2560,
        4096,
        5120,
        7680,
        32768,
        262144,
        524288,
        1048576,
        2097152,
    ]
    world_sizes = [2, 4]
    num_ctas = [1, 2, 4, 8, 16, 32, 36]

    @staticmethod
    def create_shared_buffer(
        size_in_bytes: int, group: Optional[ProcessGroup] = None
    ) -> List[int]:
        lib = CudaRTLibrary()
        pointer = lib.cudaMalloc(size_in_bytes)
        handle = lib.cudaIpcGetMemHandle(pointer)
        if group is None:
            group = dist.group.WORLD
        world_size = dist.get_world_size(group=group)
        rank = dist.get_rank(group=group)

        handle_bytes = ctypes.string_at(ctypes.addressof(handle), ctypes.sizeof(handle))
        input_tensor = torch.ByteTensor(list(handle_bytes)).to(f"cuda:{rank}")
        gathered_tensors = [torch.empty_like(input_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, input_tensor, group=group)

        handles = []
        handle_type = type(handle)
        for tensor in gathered_tensors:
            bytes_list = tensor.cpu().tolist()
            bytes_data = bytes(bytes_list)
            handle_obj = handle_type()
            ctypes.memmove(ctypes.addressof(handle_obj), bytes_data, len(bytes_data))
            handles.append(handle_obj)

        pointers: List[int] = []
        for i, h in enumerate(handles):
            if i == rank:
                pointers.append(pointer.value)
            else:
                try:
                    opened_ptr = lib.cudaIpcOpenMemHandle(h)
                    pointers.append(opened_ptr.value)
                except Exception as e:
                    print(f"Rank {rank}: Failed to open IPC handle from rank {i}: {e}")
                    raise

        dist.barrier(group=group)
        return pointers

    @staticmethod
    def free_shared_buffer(
        pointers: List[int], group: Optional[ProcessGroup] = None
    ) -> None:
        if group is None:
            group = dist.group.WORLD
        rank = dist.get_rank(group=group)
        lib = CudaRTLibrary()
        if pointers and len(pointers) > rank and pointers[rank] is not None:
            lib.cudaFree(ctypes.c_void_p(pointers[rank]))
        dist.barrier(group=group)

    def test_correctness(self):
        for world_size in self.world_sizes:
            available_gpus = torch.cuda.device_count()
            if world_size > available_gpus:
                print(
                    f"Skipping world_size={world_size}, requires {world_size} GPUs, found {available_gpus}"
                )
                continue

            print(f"Running test for world_size={world_size}")
            multi_process_parallel(
                world_size,
                _run_correctness_worker,
                target_args=(self.test_sizes, self.num_ctas),
            )
            print(f"custom allreduce tp = {world_size}: OK")


if __name__ == "__main__":
    unittest.main()
