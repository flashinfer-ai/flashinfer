# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import statistics
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from flashinfer.comm.dcp_direct_reduce import DCPDirectReduceWorkspace
from flashinfer.comm.torch_symmetric_memory import _enable_symm_mem_for_group

os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

TOTAL_HEADS = 64
HEAD_DIM = 512
DTYPE = torch.bfloat16
WARMUP = 50
ITERS = 500
SAMPLES = 7
TOKEN_ROWS = (1, 8, 32, 64, 128)
ACCEPT = {1: 0.80, 8: 0.80, 32: 0.95, 64: 1.03, 128: 1.03}


def _sanitize_lse(s: torch.Tensor) -> torch.Tensor:
    invalid = torch.isnan(s) | torch.isposinf(s)
    return torch.where(invalid, torch.full_like(s, -float("inf")), s)


def _merge(
    stacked_o: torch.Tensor, stacked_s: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    s = _sanitize_lse(stacked_s)
    m = s.max(dim=0).values
    m_math = torch.where(torch.isneginf(m), torch.zeros_like(m), m)
    weights = torch.exp(s - m_math)
    denom = weights.sum(dim=0)
    norm = torch.where(
        denom > 0, weights / denom.clamp_min(1e-30), torch.zeros_like(weights)
    )
    safe_o = torch.where(norm[..., None] == 0, torch.zeros_like(stacked_o), stacked_o)
    out = (safe_o.float() * norm[..., None]).sum(dim=0)
    lse = torch.where(
        denom > 0, torch.log(denom) + m_math, torch.full_like(denom, -float("inf"))
    )
    return out, lse


def _alloc_pair(shape, dtype, device, group, *, symmetric: bool):
    handles = []
    if not symmetric:
        return torch.empty(*shape, dtype=dtype, device=device), handles
    buf = symm_mem.empty(*shape, dtype=dtype, device=device)
    buf.zero_()
    torch.cuda.synchronize()
    handle = symm_mem.rendezvous(buf, group.group_name)
    assert handle is not None
    handle.barrier()
    handles.append(handle)
    return buf, handles


class NcclBaseline:
    """NCCL all_to_all + local merge.

    symmetric=False: ordinary torch.empty send/recv (portable NCCL).
    symmetric=True: same NCCL all_to_all, but send/recv live in
    PyTorch symmetric-memory allocations (NCCL user buffers).
    """

    def __init__(
        self,
        group: dist.ProcessGroup,
        tokens: int,
        h_local: int,
        head_dim: int,
        dtype,
        *,
        symmetric: bool,
    ):
        self.group = group
        self.world = group.size()
        self.tokens = tokens
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        self._handles = []
        self.send_o, h = _alloc_pair(
            (self.world, tokens, h_local, head_dim),
            dtype,
            device,
            group,
            symmetric=symmetric,
        )
        self._handles.extend(h)
        self.recv_o, h = _alloc_pair(
            (self.world, tokens, h_local, head_dim),
            dtype,
            device,
            group,
            symmetric=symmetric,
        )
        self._handles.extend(h)
        self.send_s, h = _alloc_pair(
            (self.world, tokens, h_local),
            torch.float32,
            device,
            group,
            symmetric=symmetric,
        )
        self._handles.extend(h)
        self.recv_s, h = _alloc_pair(
            (self.world, tokens, h_local),
            torch.float32,
            device,
            group,
            symmetric=symmetric,
        )
        self._handles.extend(h)
        self.out = torch.empty(tokens, h_local, head_dim, dtype=dtype, device=device)
        self.lse = torch.empty(tokens, h_local, dtype=torch.float32, device=device)
        self.h_local = h_local
        self.symmetric = symmetric

    def run(self, partial_o: torch.Tensor, partial_s: torch.Tensor) -> None:
        for dst in range(self.world):
            sl = slice(dst * self.h_local, (dst + 1) * self.h_local)
            self.send_o[dst].copy_(partial_o[:, sl])
            self.send_s[dst].copy_(partial_s[:, sl])
        dist.all_to_all_single(self.recv_o, self.send_o, group=self.group)
        dist.all_to_all_single(self.recv_s, self.send_s, group=self.group)
        out, lse = _merge(self.recv_o, self.recv_s)
        self.out.copy_(out.to(self.out.dtype))
        self.lse.copy_(lse)


def _time_graph(fn, po: torch.Tensor, ps: torch.Tensor) -> float:
    pool_o = torch.randn(8, *po.shape, dtype=po.dtype, device=po.device)
    pool_s = torch.randn(8, *ps.shape, dtype=ps.dtype, device=ps.device)
    for i in range(WARMUP):
        po.copy_(pool_o[i % 8])
        ps.copy_(pool_s[i % 8])
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    po.copy_(pool_o[0])
    ps.copy_(pool_s[0])
    with torch.cuda.graph(graph):
        fn()
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(SAMPLES):
        starter.record()
        for i in range(ITERS):
            po.copy_(pool_o[i % 8])
            ps.copy_(pool_s[i % 8])
            graph.replay()
        ender.record()
        ender.synchronize()
        samples.append(starter.elapsed_time(ender) / ITERS)
    return statistics.median(samples)


class DcpA2aBaseline:
    """Current FlashInfer decode_cp_a2a_alltoall + local merge."""

    def __init__(
        self, group: dist.ProcessGroup, tokens: int, h_local: int, head_dim: int, dtype
    ):
        from flashinfer.comm import (
            decode_cp_a2a_alltoall,
            decode_cp_a2a_allocate_mnnvl_workspace,
            decode_cp_a2a_init_workspace,
        )
        from flashinfer.comm.comm_backend import TorchDistBackend
        from flashinfer.comm.mapping import Mapping
        from flashinfer.comm.mnnvl import MnnvlConfig, MnnvlMemory
        import pynvml

        pynvml.nvmlInit()
        if not MnnvlMemory.supports_mnnvl():
            raise RuntimeError("MNNVL not supported")
        self.decode_cp_a2a_alltoall = decode_cp_a2a_alltoall
        self.group = group
        self.world = group.size()
        self.rank = group.rank()
        self.tokens = tokens
        self.h_local = h_local
        self.head_dim = head_dim
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        mapping = Mapping(
            world_size=self.world,
            rank=self.rank,
            cp_size=self.world,
            tp_size=1,
            pp_size=1,
        )
        MnnvlMemory.initialize()
        self.workspace = decode_cp_a2a_allocate_mnnvl_workspace(
            mapping, mnnvl_config=MnnvlConfig(comm_backend=TorchDistBackend(group))
        )
        decode_cp_a2a_init_workspace(self.workspace, self.rank, self.world)
        torch.cuda.synchronize()
        dist.barrier(group)
        rows = tokens * h_local
        self.send_o = torch.empty(
            rows, self.world, head_dim, dtype=dtype, device=device
        )
        self.send_s = torch.zeros(
            rows, self.world, 2, dtype=torch.float32, device=device
        )
        self.out = torch.empty(tokens, h_local, head_dim, dtype=dtype, device=device)
        self.lse = torch.empty(tokens, h_local, dtype=torch.float32, device=device)

    def run(self, partial_o: torch.Tensor, partial_s: torch.Tensor) -> None:
        t = partial_o.shape[0]
        send_o = self.send_o[: t * self.h_local]
        send_s = self.send_s[: t * self.h_local]
        view_o = send_o.view(t, self.h_local, self.world, self.head_dim)
        view_s = send_s.view(t, self.h_local, self.world, 2)
        for dst in range(self.world):
            sl = slice(dst * self.h_local, (dst + 1) * self.h_local)
            view_o[:, :, dst, :].copy_(partial_o[:, sl])
            view_s[:, :, dst, 0].copy_(partial_s[:, sl])
            view_s[:, :, dst, 1].zero_()
        recv_o, recv_s = self.decode_cp_a2a_alltoall(
            send_o, send_s, self.workspace, self.rank, self.world
        )
        if not isinstance(recv_o, torch.Tensor):
            recv_o = torch.from_dlpack(recv_o)
            recv_s = torch.from_dlpack(recv_s)
        stacked_o = recv_o.reshape(t, self.h_local, self.world, self.head_dim).permute(
            2, 0, 1, 3
        )
        stacked_s = (
            recv_s[:, :, 0].reshape(t, self.h_local, self.world).permute(2, 0, 1)
        )
        out, lse = _merge(stacked_o.contiguous(), stacked_s.contiguous())
        self.out[:t].copy_(out.to(self.out.dtype))
        self.lse[:t].copy_(lse)


def main() -> None:
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(backend="nccl", device_id=device)
    group = dist.group.WORLD
    _enable_symm_mem_for_group(group.group_name)
    if world != 4:
        raise SystemExit("this benchmark expects 4 ranks")

    max_tokens = max(TOKEN_ROWS)
    workspace = DCPDirectReduceWorkspace(
        group,
        max_tokens=max_tokens,
        total_heads=TOTAL_HEADS,
        head_dim=HEAD_DIM,
        dtype=DTYPE,
        num_slots=1,
    )
    h_local = workspace.local_heads
    caller_o = torch.empty(max_tokens, h_local, HEAD_DIM, dtype=DTYPE, device=device)
    caller_s = torch.empty(max_tokens, h_local, dtype=torch.float32, device=device)

    if rank == 0:
        print(f"{'T':>5} {'direct':>10} {'nccl':>10} {'nccl_symm':>10} {'d/nccl':>8}")
        print("us; direct=DCPDirectReduceWorkspace; nccl_symm=NCCL on symm_mem buffers")

    for t in TOKEN_ROWS:
        po = torch.randn(t, TOTAL_HEADS, HEAD_DIM, dtype=DTYPE, device=device)
        ps = torch.randn(t, TOTAL_HEADS, dtype=torch.float32, device=device)
        workspace.run(po, ps, slot=0)
        workspace.run(po, ps, slot=0, out=caller_o[:t], lse_out=caller_s[:t])
        nccl = NcclBaseline(group, t, h_local, HEAD_DIM, DTYPE, symmetric=False)
        nccl_symm = NcclBaseline(group, t, h_local, HEAD_DIM, DTYPE, symmetric=True)
        nccl.run(po, ps)
        nccl_symm.run(po, ps)
        dist.barrier()

        direct_ms = _time_graph(lambda: workspace.run(po, ps, slot=0), po, ps)
        nccl_ms = _time_graph(lambda: nccl.run(po, ps), po, ps)
        nccl_symm_ms = _time_graph(lambda: nccl_symm.run(po, ps), po, ps)

        if rank == 0:
            print(
                f"{t:5d} {direct_ms * 1e3:10.2f} {nccl_ms * 1e3:10.2f} "
                f"{nccl_symm_ms * 1e3:10.2f} {direct_ms / nccl_ms:8.3f}"
            )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
