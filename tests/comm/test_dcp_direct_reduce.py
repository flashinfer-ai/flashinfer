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
import socket
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

os.environ.setdefault("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "1800")
os.environ.setdefault("TORCH_NCCL_ENABLE_MONITORING", "0")

from flashinfer.comm.dcp_direct_reduce import DCPDirectReduceWorkspace
from flashinfer.comm.torch_symmetric_memory import _enable_symm_mem_for_group


def _sanitize_lse(s: torch.Tensor) -> torch.Tensor:
    invalid = torch.isnan(s) | torch.isposinf(s)
    return torch.where(invalid, torch.full_like(s, -float("inf")), s)


def reference_merge(
    stacked_output: torch.Tensor,
    stacked_lse: torch.Tensor,
    is_lse_base_on_e: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Independent PyTorch reference. stacked_*[source, T, H_local, ...]."""
    o = stacked_output
    s = _sanitize_lse(stacked_lse)
    m = s.max(dim=0).values
    all_invalid = torch.isneginf(m)
    m_math = torch.where(all_invalid, torch.zeros_like(m), m)
    if is_lse_base_on_e:
        weights = torch.exp(s - m_math)
    else:
        weights = torch.exp2(s - m_math)
    denom = weights.sum(dim=0)
    normalized = torch.where(
        denom > 0,
        weights / denom.clamp_min(1e-30),
        torch.zeros_like(weights),
    )
    safe_o = torch.where(normalized[..., None] == 0, torch.zeros_like(o), o)
    expected_o = (safe_o.float() * normalized[..., None]).sum(dim=0)
    log_fn = torch.log if is_lse_base_on_e else torch.log2
    expected_lse = torch.where(
        denom > 0,
        log_fn(denom) + m_math,
        torch.full_like(denom, -float("inf")),
    )
    return expected_o, expected_lse


def _assert_close(
    actual_o: torch.Tensor,
    actual_lse: torch.Tensor,
    expected_o: torch.Tensor,
    expected_lse: torch.Tensor,
) -> None:
    invalid = torch.isneginf(expected_lse)
    assert torch.equal(torch.isneginf(actual_lse), invalid)
    if (~invalid).any():
        torch.testing.assert_close(
            actual_lse[~invalid],
            expected_lse[~invalid],
            rtol=1e-4,
            atol=1e-4,
        )
    torch.testing.assert_close(
        actual_o.float(),
        expected_o.float(),
        rtol=1e-2,
        atol=1e-2,
    )


def _gather_stack(
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[torch.Tensor, torch.Tensor]:
    world = group.size()
    gathered_o = [torch.empty_like(partial_output) for _ in range(world)]
    gathered_s = [torch.empty_like(partial_lse) for _ in range(world)]
    dist.all_gather(gathered_o, partial_output.contiguous(), group=group)
    dist.all_gather(gathered_s, partial_lse.contiguous(), group=group)
    return torch.stack(gathered_o, dim=0), torch.stack(gathered_s, dim=0)


def _owned_stack(
    stacked_output: torch.Tensor,
    stacked_lse: torch.Tensor,
    rank: int,
    h_local: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    local = slice(rank * h_local, (rank + 1) * h_local)
    return stacked_output[:, :, local, :], stacked_lse[:, :, local]


def get_open_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _setup(rank: int, world_size: int, port: int):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    group = dist.group.WORLD
    _enable_symm_mem_for_group(group.group_name)
    torch.manual_seed(2026 + rank)
    return device, group


def _make_inputs(
    t: int,
    total_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
    padded: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if padded:
        base_o = torch.randn(t * 2, total_heads, head_dim, dtype=dtype, device=device)
        base_s = torch.randn(t * 2, total_heads, dtype=torch.float32, device=device)
        return base_o[::2], base_s[::2]
    return (
        torch.randn(t, total_heads, head_dim, dtype=dtype, device=device),
        torch.randn(t, total_heads, dtype=torch.float32, device=device),
    )


def _run_and_check(
    workspace: DCPDirectReduceWorkspace,
    partial_o: torch.Tensor,
    partial_s: torch.Tensor,
    group: dist.ProcessGroup,
    *,
    slot: int = 0,
    is_lse_base_on_e: bool = True,
    out: torch.Tensor | None = None,
    lse_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    output, lse = workspace.run(
        partial_o,
        partial_s,
        slot=slot,
        is_lse_base_on_e=is_lse_base_on_e,
        out=out,
        lse_out=lse_out,
    )
    stacked_o, stacked_s = _gather_stack(partial_o, partial_s, group)
    owned_o, owned_s = _owned_stack(
        stacked_o, stacked_s, workspace.rank, workspace.local_heads
    )
    expected_o, expected_s = reference_merge(owned_o, owned_s, is_lse_base_on_e)
    _assert_close(output, lse, expected_o, expected_s)
    return output, lse


def _worker(world_size: int, rank: int, port: int, suite: str) -> None:
    device, group = _setup(rank, world_size, port)
    try:
        if suite == "peer":
            _suite_peer(device, group)
        elif suite == "correctness":
            _suite_correctness(device, group)
        elif suite == "ownership":
            _suite_ownership(device, group)
        elif suite == "graph":
            _suite_graph(device, group)
        elif suite == "generations":
            _suite_generations(device, group)
        else:
            raise ValueError(suite)
    finally:
        dist.barrier(group)
        dist.destroy_process_group()


def _suite_peer(device: torch.device, group: dist.ProcessGroup) -> None:
    workspace = DCPDirectReduceWorkspace(
        group,
        max_tokens=8,
        total_heads=64,
        head_dim=128,
        dtype=torch.bfloat16,
        num_slots=2,
    )
    marker = torch.tensor(float(group.rank() + 1), dtype=torch.bfloat16, device=device)
    for dest in range(group.size()):
        workspace._peer_output_views[dest][0, 0, group.rank(), 0, 0, 0] = marker
    torch.cuda.synchronize()
    dist.barrier(group)
    for src in range(group.size()):
        got = workspace.received_output[0, 0, src, 0, 0, 0].item()
        assert got == float(src + 1), f"rank {group.rank()} src {src}: {got}"

    ptrs = workspace.peer_output_ptrs[0].tolist()
    assert len(ptrs) == group.size()
    assert all(p != 0 for p in ptrs)
    local_ptr = workspace.received_output[0].data_ptr()
    assert workspace.peer_output_ptrs[0, group.rank()].item() == local_ptr


def _suite_correctness(device: torch.device, group: dist.ProcessGroup) -> None:
    world = group.size()
    total_heads = 64
    cases = []
    for dtype in (torch.float16, torch.bfloat16):
        for head_dim in (128, 512):
            for base_e in (True, False):
                for t in (1, 8, 32, 64, 128):
                    cases.append((dtype, head_dim, base_e, t, False, "random"))
    cases.extend(
        [
            (torch.bfloat16, 128, True, 8, False, "one_neg_inf"),
            (torch.bfloat16, 128, True, 8, False, "multi_neg_inf"),
            (torch.bfloat16, 128, True, 8, False, "all_neg_inf"),
            (torch.bfloat16, 128, True, 8, False, "nan_lse"),
            (torch.bfloat16, 128, True, 8, False, "pos_inf_lse"),
            (torch.bfloat16, 128, False, 8, False, "nan_payload_zero_weight"),
            (torch.float16, 128, True, 8, True, "padded"),
        ]
    )

    cache: dict[tuple, DCPDirectReduceWorkspace] = {}

    def ws_for(dtype, head_dim):
        key = (dtype, head_dim)
        if key not in cache:
            cache[key] = DCPDirectReduceWorkspace(
                group,
                max_tokens=128,
                total_heads=total_heads,
                head_dim=head_dim,
                dtype=dtype,
                num_slots=2,
            )
        return cache[key]

    for dtype, head_dim, base_e, t, padded, kind in cases:
        workspace = ws_for(dtype, head_dim)
        partial_o, partial_s = _make_inputs(
            t, total_heads, head_dim, dtype, device, padded=padded
        )
        if kind == "one_neg_inf":
            partial_s[:, : workspace.local_heads] = -float("inf")
        elif kind == "multi_neg_inf":
            partial_s[:, :] = -float("inf")
            if group.rank() == 0:
                partial_s[:, workspace.local_heads :] = torch.randn(
                    t, total_heads - workspace.local_heads, device=device
                )
        elif kind == "all_neg_inf":
            partial_s.fill_(-float("inf"))
        elif kind == "nan_lse":
            if group.rank() == 0:
                partial_s[:, 0] = float("nan")
        elif kind == "pos_inf_lse":
            if group.rank() == 1 % world:
                partial_s[:, 1] = float("inf")
        elif kind == "nan_payload_zero_weight":
            partial_s.fill_(-float("inf"))
            partial_o.fill_(float("nan"))
        _run_and_check(
            workspace,
            partial_o,
            partial_s,
            group,
            slot=0,
            is_lse_base_on_e=base_e,
        )
        if kind == "all_neg_inf":
            out, lse = workspace.run(partial_o, partial_s, slot=0)
            assert torch.all(out == 0)
            assert torch.all(torch.isneginf(lse))


def _suite_ownership(device: torch.device, group: dist.ProcessGroup) -> None:
    workspace = DCPDirectReduceWorkspace(
        group,
        max_tokens=32,
        total_heads=64,
        head_dim=128,
        dtype=torch.bfloat16,
        num_slots=2,
    )
    t = 8
    po, ps = _make_inputs(t, 64, 128, torch.bfloat16, device)

    out, lse = workspace.run(po, ps, slot=0)
    assert out.data_ptr() == workspace.combined_output[0].data_ptr()
    assert lse.data_ptr() == workspace.combined_lse[0].data_ptr()
    _run_and_check(workspace, po, ps, group, slot=0)

    caller_o = torch.empty(
        t, workspace.local_heads, 128, dtype=torch.bfloat16, device=device
    )
    caller_s = torch.empty(t, workspace.local_heads, dtype=torch.float32, device=device)
    out_b, lse_b = workspace.run(po, ps, slot=0, out=caller_o, lse_out=caller_s)
    assert out_b.data_ptr() == caller_o.data_ptr()
    assert lse_b.data_ptr() == caller_s.data_ptr()
    _run_and_check(workspace, po, ps, group, slot=0, out=caller_o, lse_out=caller_s)

    saved_caller_o = caller_o.clone()
    saved_caller_s = caller_s.clone()

    first = workspace.run(po, ps, slot=0)
    po2, ps2 = _make_inputs(t, 64, 128, torch.bfloat16, device)
    second, _ = _run_and_check(workspace, po2, ps2, group, slot=0)
    assert second.data_ptr() == first[0].data_ptr()
    # same-slot reuse may overwrite the old workspace view; do not require first_o.

    a_o, a_s = _run_and_check(workspace, po, ps, group, slot=0)
    a_o_copy, a_s_copy = a_o.clone(), a_s.clone()
    _run_and_check(workspace, po2, ps2, group, slot=1)
    torch.testing.assert_close(a_o, a_o_copy)
    torch.testing.assert_close(a_s, a_s_copy)

    _run_and_check(workspace, po2, ps2, group, slot=0)
    torch.testing.assert_close(caller_o, saved_caller_o)
    torch.testing.assert_close(caller_s, saved_caller_s)

    with pytest.raises(ValueError):
        workspace.run(po, ps, out=caller_o, lse_out=None)
    with pytest.raises(ValueError):
        workspace.run(po, ps, out=None, lse_out=caller_s)

    # No post-merge aten::copy_ on the warmed-up hot path.
    workspace.run(po, ps, slot=0)
    torch.cuda.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        workspace.run(po, ps, slot=0)
        workspace.run(po, ps, slot=0, out=caller_o, lse_out=caller_s)
    copy_events = [ev for ev in prof.key_averages() if "aten::copy" in ev.key]
    assert copy_events == [], [ev.key for ev in copy_events]


def _suite_graph(device: torch.device, group: dist.ProcessGroup) -> None:
    workspace = DCPDirectReduceWorkspace(
        group,
        max_tokens=32,
        total_heads=64,
        head_dim=128,
        dtype=torch.bfloat16,
        num_slots=2,
    )
    t = 8
    po = torch.randn(t, 64, 128, dtype=torch.bfloat16, device=device)
    ps = torch.randn(t, 64, dtype=torch.float32, device=device)
    po1 = torch.randn_like(po)
    ps1 = torch.randn_like(ps)
    caller_o = torch.empty(
        t, workspace.local_heads, 128, dtype=torch.bfloat16, device=device
    )
    caller_s = torch.empty(t, workspace.local_heads, dtype=torch.float32, device=device)

    # Warmup / compile outside the graph.
    workspace.run(po, ps, slot=0)
    workspace.run(po1, ps1, slot=1)
    workspace.run(po, ps, slot=0, out=caller_o, lse_out=caller_s)
    dist.barrier(group)

    g_ws = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_ws):
        ws_out, ws_lse = workspace.run(po, ps, slot=0)
    g_caller = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_caller):
        c_out, c_lse = workspace.run(po, ps, slot=0, out=caller_o, lse_out=caller_s)
    g_slots = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_slots):
        s0_out, s0_lse = workspace.run(po, ps, slot=0)
        s1_out, s1_lse = workspace.run(po1, ps1, slot=1)

    for _ in range(100):
        po.copy_(torch.randn_like(po))
        ps.copy_(torch.randn_like(ps))
        g_ws.replay()
        stacked_o, stacked_s = _gather_stack(po, ps, group)
        owned_o, owned_s = _owned_stack(
            stacked_o, stacked_s, workspace.rank, workspace.local_heads
        )
        expected_o, expected_s = reference_merge(owned_o, owned_s, True)
        _assert_close(ws_out, ws_lse, expected_o, expected_s)

        po.copy_(torch.randn_like(po))
        ps.copy_(torch.randn_like(ps))
        g_caller.replay()
        assert c_out.data_ptr() == caller_o.data_ptr()
        assert c_lse.data_ptr() == caller_s.data_ptr()
        stacked_o, stacked_s = _gather_stack(po, ps, group)
        owned_o, owned_s = _owned_stack(
            stacked_o, stacked_s, workspace.rank, workspace.local_heads
        )
        expected_o, expected_s = reference_merge(owned_o, owned_s, True)
        _assert_close(c_out, c_lse, expected_o, expected_s)

        po.copy_(torch.randn_like(po))
        ps.copy_(torch.randn_like(ps))
        po1.copy_(torch.randn_like(po1))
        ps1.copy_(torch.randn_like(ps1))
        g_slots.replay()
        for buf_o, buf_s, slot_out, slot_lse in (
            (po, ps, s0_out, s0_lse),
            (po1, ps1, s1_out, s1_lse),
        ):
            stacked_o, stacked_s = _gather_stack(buf_o, buf_s, group)
            owned_o, owned_s = _owned_stack(
                stacked_o, stacked_s, workspace.rank, workspace.local_heads
            )
            expected_o, expected_s = reference_merge(owned_o, owned_s, True)
            _assert_close(slot_out, slot_lse, expected_o, expected_s)


def _suite_generations(device: torch.device, group: dist.ProcessGroup) -> None:
    workspace = DCPDirectReduceWorkspace(
        group,
        max_tokens=8,
        total_heads=64,
        head_dim=128,
        dtype=torch.bfloat16,
        num_slots=1,
    )
    last_epoch = 0
    for i in range(1000):
        po, ps = _make_inputs(8, 64, 128, torch.bfloat16, device)
        _run_and_check(workspace, po, ps, group, slot=0)
        epoch = int(workspace.epoch[0].item())
        assert epoch == last_epoch + 1, (epoch, last_epoch, i)
        last_epoch = epoch
        assert epoch == i + 1


def _spawn(world_size: int, suite: str) -> None:
    if torch.cuda.device_count() < world_size:
        pytest.skip(f"need {world_size} GPUs")
    port = get_open_port()
    mp.set_start_method("spawn", force=True)
    procs = []
    for rank in range(world_size):
        proc = mp.Process(target=_worker, args=(world_size, rank, port, suite))
        proc.start()
        procs.append(proc)
    for rank, proc in enumerate(procs):
        proc.join()
        assert proc.exitcode == 0, (
            f"rank {rank} failed with {proc.exitcode} suite={suite}"
        )


@pytest.mark.parametrize("world_size", [2, 4])
def test_peer_pointer_visibility(world_size):
    _spawn(world_size, "peer")


@pytest.mark.parametrize("world_size", [2, 4])
def test_correctness_matrix(world_size):
    _spawn(world_size, "correctness")


@pytest.mark.parametrize("world_size", [2, 4])
def test_output_ownership(world_size):
    _spawn(world_size, "ownership")


@pytest.mark.parametrize("world_size", [2, 4])
def test_cuda_graph_replay(world_size):
    _spawn(world_size, "graph")


@pytest.mark.parametrize("world_size", [2, 4])
def test_thousand_generations(world_size):
    _spawn(world_size, "generations")
