# Adapted from vllm/tests

import ctypes
import torch
import torch.distributed as dist

import flashinfer.comm as comm
from flashinfer.comm import CudaRTLibrary

# Usage: torchrun --nproc_per_node=2 test_create_ipc_buffer.py

# create a cpu process group for communicating metadata (ipc handle)
dist.init_process_group(backend="nccl")
rank = local_rank = dist.get_rank()
world_size = dist.get_world_size()

# every process sets its own device (differently)
cudart = CudaRTLibrary()
cudart.cudaSetDevice(rank)

buffer_size_in_bytes = 1024
byte_value = rank

pointers = comm.create_shared_buffer(buffer_size_in_bytes)

print(f"Rank {rank} init ipc buffer {pointers}")

dist.barrier()
torch.cuda.synchronize()

# tries to write to all buffers
for p in pointers:
    pointer = ctypes.c_void_p(p + rank * (buffer_size_in_bytes // world_size))
    cudart.cudaMemset(pointer, byte_value, buffer_size_in_bytes // world_size)

dist.barrier()
torch.cuda.synchronize()

host_data = (ctypes.c_char * buffer_size_in_bytes)()

# all ranks read from all buffers, and check if the data is correct
for p in pointers:
    pointer = ctypes.c_void_p(p)
    for cur_rank in range(world_size):
        # Fix pointer arithmetic by casting to int, adding offset, then back to c_void_p
        offset_pointer = ctypes.c_void_p(p + cur_rank * (buffer_size_in_bytes // world_size))
        cudart.cudaMemcpy(host_data, offset_pointer, buffer_size_in_bytes // world_size)
        for i in range(buffer_size_in_bytes // world_size):
            assert ord(host_data[i]) == cur_rank, (
                f"Rank {rank} failed"
                f" to verify buffer {p}. Expected {cur_rank}, "
                f"got {ord(host_data[i])}")

print(f"Rank {rank} verified all buffers.\n")

dist.barrier()
torch.cuda.synchronize()

comm.free_shared_buffer(pointers)