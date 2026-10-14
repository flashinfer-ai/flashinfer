"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Tests of the Cake fused tensor-parallel GEMM + MNNVL Lamport all-reduce (``flashinfer.comm.cake_gemm_allreduce``).

The route table and workspace arithmetic are checked without a GPU.  The multi-rank test runs one process per GPU
(``torch.multiprocessing.spawn``) on an SM100 / SM103 node with MNNVL fabric memory, creates FlashInfer's own MNNVL
all-reduce fusion workspace, and compares the fused kernel against the NCCL fp32 all-reduce of the bf16-rounded local
partials (the production chain's semantics) at ``atol = rtol = 1e-2``, requiring the output to be bitwise identical on
every rank, across M = 1 ... 128, a relaunch, and the PDL launch.
"""

from __future__ import annotations

import os
import socket
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from flashinfer.jit import cake_gemm_allreduce as loader

HIDDEN = 7168
K_SHARD = 1536
TOKENS = (1, 8, 16, 32, 64, 128)


@pytest.mark.parametrize(
    "tokens,expected",
    [
        (1, (16, False)),
        (16, (16, False)),
        (17, (32, True)),
        (32, (32, True)),
        (33, (64, True)),
        (64, (64, True)),
        (65, (128, True)),
        (128, (128, True)),
        (129, (128, True)),
        (1024, (128, True)),
    ],
)
def test_route_is_one_shot_to_16_then_the_smallest_covering_two_shot_tile(
    tokens, expected
) -> None:
    assert loader.select_route(tokens) == expected
    from flashinfer.comm.cake_gemm_allreduce import cake_gemm_allreduce_select_route

    assert cake_gemm_allreduce_select_route(tokens) == expected


@pytest.mark.parametrize("tokens", [0, -1, True, 2.0])
def test_route_rejects_non_positive_counts(tokens) -> None:
    with pytest.raises(ValueError):
        loader.select_route(tokens)


def test_route_table_covers_every_world_size_and_tile() -> None:
    assert set(loader.ROUTES) == {
        (world, tile, tile > loader.ONESHOT_MAX_TOKENS)
        for world in loader.WORLD_SIZES
        for tile in loader.TOKEN_TILES
    }
    for (world, tile, twoshot), row in loader.ROUTES.items():
        kind = "twoshot" if twoshot else "oneshot"
        assert (
            row["symbol"]
            == f"kernel_cake_blackwell_gemm_allreduce_{kind}_n{tile}_bf16_ws{world}"
        )
        assert row["dynamic_smem_bytes"] > 0


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_workspace_bytes_follow_the_route(world_size) -> None:
    assert (
        loader.required_workspace_bytes(16, HIDDEN, world_size)
        == 16 * HIDDEN * world_size * 2
    )
    assert (
        loader.required_workspace_bytes(64, HIDDEN, world_size)
        == 2 * (-(-64 // world_size)) * world_size * HIDDEN * 2
    )
    with pytest.raises(ValueError):
        loader.required_workspace_bytes(16, HIDDEN, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_api_validates_operands_before_building() -> None:
    from flashinfer.comm.cake_gemm_allreduce import cake_gemm_allreduce

    workspace = SimpleNamespace(
        mc_ptr=1,
        uc_ptr_local=1,
        uc_ptrs_dev=1,
        buffer_flags=torch.zeros(9, dtype=torch.uint32, device="cuda"),
        rank=0,
        tp_size=2,
        buffer_size_bytes=1 << 30,
    )
    x = torch.zeros(4, K_SHARD, dtype=torch.bfloat16, device="cuda")
    w = torch.zeros(HIDDEN, K_SHARD, dtype=torch.bfloat16, device="cuda")
    out = torch.zeros(4, HIDDEN, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        cake_gemm_allreduce(
            x, w[:, : K_SHARD - 64].contiguous(), out, workspace
        )  # K mismatch
    with pytest.raises(ValueError):
        cake_gemm_allreduce(
            x[:, :1472].contiguous(), w[:, :1472].contiguous(), out, workspace
        )  # K % 128
    with pytest.raises(ValueError):
        cake_gemm_allreduce(
            x,
            w[: HIDDEN - 32].contiguous(),
            out[:, : HIDDEN - 32].contiguous(),
            workspace,
        )  # N % 64
    with pytest.raises(TypeError):
        cake_gemm_allreduce(x.float(), w, out, workspace)
    with pytest.raises(TypeError):
        cake_gemm_allreduce(x, w, out, SimpleNamespace(rank=0))
    with pytest.raises(ValueError):
        cake_gemm_allreduce(
            x, w, out, SimpleNamespace(**{**vars(workspace), "tp_size": 3})
        )
    with pytest.raises(ValueError):
        cake_gemm_allreduce(
            x, w, out, SimpleNamespace(**{**vars(workspace), "buffer_size_bytes": 16})
        )


def _supported_platform(world_size: int) -> str | None:
    if not torch.cuda.is_available():
        return "requires CUDA"
    if torch.cuda.device_count() < world_size:
        return f"requires {world_size} GPUs"
    if torch.cuda.get_device_capability(0) not in {(10, 0), (10, 3)}:
        return "the generated kernels target Blackwell SM100 / SM103"
    from flashinfer.comm.mnnvl import MnnvlMemory

    if not MnnvlMemory.supports_mnnvl():
        return "MNNVL fabric memory is not available on this platform"
    return None


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(rank: int, world_size: int, port: int) -> None:
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        from flashinfer.comm import (
            allreduce_fusion,
            cake_gemm_allreduce,
            create_allreduce_fusion_workspace,
        )
        from flashinfer.comm.cake_gemm_allreduce import (
            cake_gemm_allreduce_workspace_bytes,
        )
        from flashinfer.comm.comm_backend import TorchDistBackend

        group = dist.group.WORLD
        workspace = create_allreduce_fusion_workspace(
            backend="mnnvl",
            world_size=world_size,
            rank=rank,
            max_token_num=max(TOKENS),
            hidden_dim=HIDDEN,
            dtype=torch.bfloat16,
            comm_backend=TorchDistBackend(group),
            force_oneshot_support=True,
            group=group,
        )
        assert int(workspace.buffer_size_bytes) >= cake_gemm_allreduce_workspace_bytes(
            max(TOKENS), HIDDEN, world_size
        )
        generator = torch.Generator(device=device).manual_seed(780 + rank)
        for tokens in TOKENS:
            x = torch.randn(
                tokens,
                K_SHARD,
                dtype=torch.bfloat16,
                device=device,
                generator=generator,
            )
            w = (
                torch.randn(
                    HIDDEN,
                    K_SHARD,
                    dtype=torch.bfloat16,
                    device=device,
                    generator=generator,
                )
                * 0.05
            )
            reference = (
                x @ w.t()
            ).float()  # the bf16-rounded local partial, as the production chain reduces it
            dist.all_reduce(reference, group=group)
            for launch_with_pdl in (False, True, False):
                out = torch.empty(tokens, HIDDEN, dtype=torch.bfloat16, device=device)
                returned = cake_gemm_allreduce(
                    x, w, out, workspace, launch_with_pdl=launch_with_pdl
                )
                torch.cuda.synchronize(device)
                assert returned is out
                torch.testing.assert_close(out.float(), reference, atol=1e-2, rtol=1e-2)
                gathered = [torch.empty_like(out) for _ in range(world_size)]
                dist.all_gather(gathered, out, group=group)
                for peer, other in enumerate(gathered):
                    assert torch.equal(other, gathered[0]), (
                        f"rank {peer} differs bitwise from rank 0 (M={tokens})"
                    )
            # the workspace stays usable by FlashInfer's own all-reduce afterwards
            partial = x @ w.t()
            via_fusion = torch.empty_like(partial)
            allreduce_fusion(
                input=partial,
                workspace=workspace,
                pattern=0,
                launch_with_pdl=False,
                output=via_fusion,
                residual_out=None,
                norm_out=None,
                quant_out=None,
                scale_out=None,
                residual_in=None,
                rms_gamma=None,
                rms_eps=None,
                scale_factor=None,
                layout_code=None,
                use_oneshot=None,
                weight_bias=None,
            )
            torch.cuda.synchronize(device)
            torch.testing.assert_close(
                via_fusion.float(), reference, atol=1e-2, rtol=1e-2
            )
        dist.barrier(group=group)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_fused_gemm_allreduce_matches_the_all_reduced_partials(world_size: int) -> None:
    reason = _supported_platform(world_size)
    if reason is not None:
        pytest.skip(reason)
    os.environ.setdefault("FLASHINFER_CUDA_ARCH_LIST", "10.0a 10.3a")
    mp.spawn(_worker, args=(world_size, _free_port()), nprocs=world_size, join=True)
