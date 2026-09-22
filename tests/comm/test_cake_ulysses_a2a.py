"""Complete generated Ulysses portfolio through the public communicator."""

import pytest
import torch
import torch.distributed as dist

from examples.pytorch.ulysses_a2a_export.interface import prepare, shapes
from tests.comm.test_ulysses_a2a import multi_process_parallel


def _correctness_worker(world_size, rank, port):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
    )
    try:
        for shape in shapes(world_size):
            case = prepare(shape, backend="nvlink")
            try:
                case.check()
            finally:
                case.close()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 6, 8])
def test_generated_ulysses_portfolio(world_size):
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"requires {world_size} GPUs")
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        pytest.skip("generated portfolio targets Blackwell SM100/SM103")
    multi_process_parallel(world_size, _correctness_worker)
