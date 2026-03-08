# Check torch version:
from typing import Any, Tuple

import multiprocessing as mp
import socket
import pytest
import torch
import torch.distributed as dist

import flashinfer.comm.trtllm_mnnvl_ar as trtllm_mnnvl_ar
from flashinfer.comm.mapping import Mapping
from flashinfer.comm.mnnvl import CommBackend as CommBackend

import pynvml

pynvml.nvmlInit()


class CustomCommunicator(CommBackend):
    def __init__(self, group):
        self._group = group

    def Get_rank(self) -> int:
        return dist.get_rank(self._group)

    def Get_size(self) -> int:
        return dist.get_world_size(self._group)

    def allgather(self, data: int | bytes):
        device = f"cuda:{torch.cuda.current_device()}"
        if isinstance(data, int):
            local_tensor = torch.tensor([data], device=device, dtype=torch.int32)
            world_size = self.Get_size()
            gathered = [torch.zeros_like(local_tensor) for _ in range(world_size)]

            dist.all_gather(gathered, local_tensor, group=self._group)
            return [int(x.item()) for x in gathered]

        elif isinstance(data, bytes):
            local_tensor = torch.ByteTensor(list(data)).unsqueeze(0).to(device)
            world_size = self.Get_size()
            gathered = [data] * self.Get_size()
            dist.all_gather_object(gathered, data, group=self._group)
            return gathered
        else:
            raise TypeError(f"Unsupported type for allgather: {type(data)}")

    def bcast(self, data, root: int = 0):
        """
        Broadcast a picklable Python object from `root` to all ranks.
        Uses torch.distributed.broadcast_object_list under the hood.

        Returns the broadcasted object on every rank.
        """
        obj_list = [data]
        # broadcast_object_list mutates obj_list in-place
        dist.broadcast_object_list(obj_list, src=root, group=self._group)
        return obj_list[0]

    def barrier(self):
        """
        Synchronize all ranks in this communicator.
        """
        dist.barrier(group=self._group)

    def Split(self, color: int, key: int) -> "CustomCommunicator":
        return self


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
    world_size: int, dtype: torch.dtype, test_target: Any, target_args: tuple = ()
) -> None:
    mp.set_start_method("spawn", force=True)

    procs = []
    distributed_init_port = get_open_port()
    for i in range(world_size):
        proc_args = (world_size, i, dtype, distributed_init_port) + target_args
        proc = mp.Process(target=test_target, args=proc_args, name=f"Worker-{i}")
        proc.start()
        procs.append(proc)

    for i in range(world_size):
        procs[i].join()
        assert procs[i].exitcode == 0, (
            f"Process {i} failed with exit code {procs[i].exitcode}"
        )


def _run_mnnvl_ar(world_size, rank, dtype, distributed_init_port, seq_len, hidden_size):
    # Set CUDA device based on rank
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
    torch.cuda.set_device(rank)
    comm = CustomCommunicator(group)
    mapping = Mapping(
        world_size=world_size,
        rank=rank,
        gpus_per_node=world_size,
        tp_size=world_size,
    )

    if mapping.local_rank == 0:
        print(
            f"[Node {mapping.node_rank}] Running MNNVL AllReduce test with {world_size} ranks"
        )
        print(
            f"[Node {mapping.node_rank}] Rank {rank} using GPU {torch.cuda.current_device()}"
        )

    tensor_parallel_size = world_size
    eps = 1e-5
    torch.manual_seed(42)

    # Track if this rank failed
    rank_failed = False
    failure_message = ""

    try:
        # Get workspace buffers using MPI rank - allocate once per seq_lens list and reuse within the list
        # This workspace is sized for the maximum expected sequence length and can be reused within each list
        # Each parameterized list gets its own fresh workspace allocation
        explicit_workspace_bytes = 3 * 2 * dtype.itemsize * hidden_size * seq_len
        mcast_buffer_mnnvl, buffer_flags_mnnvl, max_num_elements_mnnvl = (
            trtllm_mnnvl_ar.get_allreduce_mnnvl_workspace(
                mapping, dtype, comm, explicit_workspace_bytes
            )
        )

        multicast_ptr = mcast_buffer_mnnvl.get_multicast_ptr()
        buffer_ptrs_dev = mcast_buffer_mnnvl.get_buffer_ptrs_dev()
        unicast_ptr = mcast_buffer_mnnvl.mcast_device_memory.get_unicast_ptr(
            mapping.tp_rank
        )

        # Test each sequence length with the same workspace (reusing allocated buffers within this list)
        if rank == 0:
            print(
                f"Testing seq_len={seq_len}, hidden_size={hidden_size}, dtype={dtype}"
            )

        # Generate test data (same on all ranks due to same seed)
        x_full = torch.randn(
            (tensor_parallel_size, seq_len, hidden_size),
            dtype=dtype,
            device=torch.device("cuda"),
        )
        residual = torch.randn(
            (seq_len, hidden_size), dtype=dtype, device=torch.device("cuda")
        )
        norm_weight = torch.randn(
            (hidden_size,), dtype=dtype, device=torch.device("cuda")
        )

        # Each rank gets its slice of the input
        x = x_full[rank, :, :]

        # Compute reference output based on fusion mode
        reference_output: Tuple[torch.Tensor, ...] = None

        # Non-fused case: Only AllReduce
        allreduce_result = torch.sum(x_full, dim=0)  # AllReduce result
        reference_output = (allreduce_result,)

        # Run the test with the same workspace
        from .test_trtllm_mnnvl_allreduce import row_linear_residual_norm_fusion_forward

        row_linear_residual_norm_fusion_forward(
            x,
            residual,
            norm_weight,
            eps,
            hidden_size,
            dtype,
            mapping,
            False,
            reference_output,
            multicast_ptr,
            buffer_ptrs_dev,
            unicast_ptr,
            max_num_elements_mnnvl,
            buffer_flags_mnnvl,
            comm,
        )

        # Synchronize before next test
        comm.barrier()

        print(f"PASSED[rank={rank}]: seq_len={seq_len}, dtype={dtype}")

    except Exception as e:
        rank_failed = True
        failure_message = (
            f"FAILED[rank={rank}]: seq_lens={seq_len}, dtype={dtype} failed: {e}"
        )
        print(failure_message)
        # Gather failure status from all ranks
        all_failures = comm.allgather(rank_failed)

        # If any rank failed, fail the test
        if any(all_failures):
            failed_ranks = [i for i, failed in enumerate(all_failures) if failed]
            if rank == 0:
                print(f"Test failed on ranks: {failed_ranks}")

            # Fail the test on all ranks
            pytest.fail(f"Test failed on ranks {failed_ranks}")
            comm.barrier()

    finally:
        # Ensure cleanup happens for this list's workspace
        if "mcast_buffer_mnnvl" in locals():
            del mcast_buffer_mnnvl

    # Final synchronization and check for failures across all ranks
    comm.barrier()


"""Main test function that runs on each MPI rank"""


@pytest.mark.parametrize("world_size", [2, 4])
def test_mnnvl_allreduce_custom_communicator(
    monkeypatch,
    world_size,
):
    monkeypatch.setenv("TRTLLM_FORCE_MNNVL_AR", "1")  # force multi-node allreduce.
    seq_len = 24
    dtype = torch.bfloat16
    hidden_size = 2048

    available_gpus = torch.cuda.device_count()
    if world_size > available_gpus:
        pytest.skip(
            f"world_size {world_size} is greater than available_gpus {available_gpus}"
        )
    print(f"Running test for world_size={world_size}")
    multi_process_parallel(
        world_size,
        dtype,
        _run_mnnvl_ar,
        target_args=(seq_len, hidden_size),
    )
    print(f"custom mnnvl allreduce world_size = {world_size}: OK")
