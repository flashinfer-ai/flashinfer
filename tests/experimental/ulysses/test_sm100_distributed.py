# SPDX-License-Identifier: Apache-2.0
"""Run explicitly under torchrun on U2/U4/U8 SM100 after patch setup.

NVSHMEM_DISABLE_NVLS=1 torchrun --standalone --nproc_per_node=2 -m pytest \
  tests/experimental/ulysses/test_sm100_distributed.py -q

Skipped on normal single-process CI. This does not bootstrap dependencies.
"""

import os

import pytest
import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from flashinfer.comm import UlyssesCommunicator
from flashinfer.comm.ulysses_experimental import prepare_ulysses_distributed_fa4


def test_prepared_valid_prefix():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world not in (2, 4, 8) or not torch.cuda.is_available():
        pytest.skip("requires explicit multi-GPU SM100 torchrun")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("requires SM100")
    symm_mem.set_backend("NVSHMEM")
    dist.init_process_group("nccl")
    try:
        physical = 38912 if world == 8 else 37888
        used, local = 37807, physical // world
        runner = prepare_ulysses_distributed_fa4(
            group=dist.group.WORLD,
            local_seq=local,
            used_seqlen=used,
            full_nvlink=True,
            discard_tail_output=True,
        )
        comm = UlyssesCommunicator(
            group=dist.group.WORLD,
            max_elems=local * 56 * 128,
            dtype=torch.bfloat16,
            backend="nccl",
        )
        from flash_attn.cute.interface import _flash_attn_fwd

        for iteration in range(3):
            torch.manual_seed(97 + dist.get_rank() + 8 * iteration)
            qkv = [
                torch.randn(1, local, 56, 128, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            global_qkv = [comm.scatter_heads(t) for t in qkv]
            ref = _flash_attn_fwd(
                *global_qkv,
                seqused_k=torch.tensor([used], device="cuda", dtype=torch.int32),
                causal=False,
            )
            ref = ref[0] if isinstance(ref, tuple) else ref
            ref = comm.gather_heads(ref)
            got = runner.run(*qkv)
            rows = max(0, min(local, used - dist.get_rank() * local))
            torch.testing.assert_close(
                got[:, :rows], ref[:, :rows], atol=1e-3, rtol=1e-2
            )
        torch.cuda.synchronize()
        comm.close()
    finally:
        dist.destroy_process_group()
