"""Generated Ulysses portfolio through the public communicator, plus loader selection."""

import contextlib

import pytest
import torch
import torch.distributed as dist
from flashinfer.jit import cake_ulysses

from tests.comm.cake_ulysses_a2a_fixture import prepare, shapes
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


@contextlib.contextmanager
def _pinned_targets(arch_list):
    """Pin the process build targets to ``arch_list`` (``FLASHINFER_CUDA_ARCH_LIST`` syntax)."""
    from flashinfer.jit.core import current_compilation_context as context

    saved = set(context.TARGET_CUDA_ARCHS)
    context.TARGET_CUDA_ARCHS.clear()
    for entry in arch_list.split():
        major, minor = entry.split(".")
        context.TARGET_CUDA_ARCHS.add((int(major), minor))
    try:
        yield
    finally:
        context.TARGET_CUDA_ARCHS.clear()
        context.TARGET_CUDA_ARCHS.update(saved)


@pytest.mark.parametrize(
    "arch_list,capabilities,name",
    [
        ("10.0a", ((10, 0),), "ulysses_a2a_sm100a"),
        ("10.3a", ((10, 3),), "ulysses_a2a_sm103a"),
        ("9.0a 10.0a 10.3a 12.0f", ((10, 0), (10, 3)), "ulysses_a2a_sm100a_sm103a"),
        ("9.0a 12.0f", (), None),
    ],
)
def test_generated_module_follows_the_build_targets(arch_list, capabilities, name):
    with _pinned_targets(arch_list):
        assert cake_ulysses.supported_capabilities() == capabilities
        if name is None:
            assert cake_ulysses.generated_module_name() is None
            return
        assert cake_ulysses.generated_module_name() == name
        gencodes = [
            flag for flag in cake_ulysses.nvcc_flags() if flag.startswith("-gencode")
        ]
        assert gencodes == [
            f"-gencode=arch=compute_{entry.replace('.', '')},"
            f"code=sm_{entry.replace('.', '')}"
            for entry in arch_list.split()
            if entry.startswith("10.")
        ]
        assert "-DFLASHINFER_ULYSSES_GENERATED=1" in cake_ulysses.extra_cuda_cflags()
