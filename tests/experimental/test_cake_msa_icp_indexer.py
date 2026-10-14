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

Tests for the experimental Cake ICP indexer chain (SM100 / SM103).

The CPU tests check the program registry, the host planning helpers, the
workspace budget and the argument validation.  The single-GPU test runs the
chain for every rank of a 2- or 4-rank group in one process on emulated
workspaces (plain device pools; the merge driven as its serialised push and
poll halves) against the CPU oracle, replays it, captures it into a CUDA
Graph and checks that a launch allocates nothing.  The multi-GPU test runs
the public API on two or four devices of one supported compute capability
with the real symmetric-memory workspace over NCCL and checks the merged
output of every rank bit for bit against the oracle, before and after a CUDA
Graph replay.
"""

from __future__ import annotations

import multiprocessing as mp
import socket
import traceback

import pytest
import torch

from flashinfer.experimental.cake_msa_icp_indexer import cake_backend, cake_jit
from flashinfer.experimental.cake_msa_icp_indexer.cake_workspace import (
    MSAIcpIndexerWorkspace,
    OUT_POISON,
    workspace_bytes,
)
from flashinfer.msa_ops import (
    create_msa_icp_indexer_workspace,
    destroy_msa_icp_indexer_workspace,
    msa_icp_indexer_workspace_bytes,
    prepare_msa_icp_indexer,
)
from tests.test_helpers.cake_msa_icp_inputs import (
    NUM_HEADS,
    SMALL_CASES,
    TOPK,
    build_case,
    build_rank_inputs,
    candidates_match,
    expected_out,
    local_topk_lists,
)

WORKER_TIMEOUT_SECONDS = 20 * 60
MULTI_GPU_CASES = ("tp2_q4_b8_ctx1500", "tp2_q1_b8_ctx1500", "tp4_q4_b8_ctx1500")


# ---------------------------------------------------------------------------
# CPU
# ---------------------------------------------------------------------------


def test_registry_records_are_consistent() -> None:
    assert cake_jit.PROGRAMS, "the checkout registers no generated ICP indexer program"
    assert set(cake_jit.ARG_PLANS) == {"scorer", "merge"}
    assert cake_jit.POLICY["topk"] == TOPK and cake_jit.POLICY["num_heads"] == NUM_HEADS
    for name, record in cake_jit.PROGRAMS.items():
        assert record["role"] in ("scorer", "merge"), name
        assert set(record["arches"]) <= set(cake_jit.ARCH_NVCC_FLAGS), name
        assert len(record["sources"]) == 2 and all(
            source.startswith(cake_jit.SOURCE_PACKAGE + "/")
            for source in record["sources"]
        ), name
        if record["role"] == "scorer":
            assert 1 <= int(record["query_len"]) <= cake_backend.MAX_QUERY_LEN
            assert int(record["rows_per_rank"]) in (32, 64, 128)
            assert int(record["page_size"]) in cake_backend.PAGE_SIZES
            assert int(record["ctas_per_sm"]) >= 1
            assert (
                record["pdl_trigger"]
                == cake_jit.POLICY["merge_production"]["scorer_pdl_trigger"]
            )
        else:
            assert int(record["world_size"]) in cake_backend.WORLD_SIZES
            assert 32 * int(record["warps"]) >= TOPK * int(record["world_size"])
            assert int(record["entries_per_thread"]) % 2 == 0
    for role, plan in cake_jit.ARG_PLANS.items():
        names = [name for _kind, name in plan]
        assert names[-3:] == ["grid_x", "grid_y", "grid_z"], role
        assert len(set(names)) == len(names), role
    merge_names = [name for _kind, name in cake_jit.ARG_PLANS["merge"]]
    for required in (
        "xpool",
        "xpool_peers",
        "pg_world",
        "pg_rank",
        "pg_flags",
        "out",
        "x_phase",
        "timeout_ns",
    ):
        assert required in merge_names, required


def test_pool_layout_and_merge_geometry_follow_the_policy() -> None:
    assert cake_backend.pool_layout(0, 4, 1, 4)["E"] == 0
    layout = cake_backend.pool_layout(13, 4, 1, 4)
    assert layout["E"] == layout["cA"] * layout["LA"] + layout["cB"] * layout["LB"]
    assert layout["LA"] + layout["LB"] <= 4 * 4  # at most one list per (split, warp)
    for columns in (1, 12, 64, 1600, 8192):
        for split in (1, 2, 37, 148, 296):
            for world in (2, 4):
                g = cake_backend.pool_geometry(columns, split, 1, 4, world_size=world)
                chunk = 32 * g["w"] * g["epl"]
                assert g["pool_stride"] % chunk == 0
                assert g["pool_stride"] >= g["E_row_max"]
                assert g["chunks"] * chunk == g["pool_stride"]
                assert 32 * g["w"] >= TOPK * world
                assert (
                    g["epl"] % 2 == 0
                    and 2 <= g["epl"] <= cake_jit.POLICY["merge_epl_max"]
                )
    assert cake_backend.merge_geometry(1)["w"] == max(
        1, cake_jit.POLICY["merge_w_policy"][0][1]
    )
    assert cake_backend.merge_geometry(1, min_w=2)["w"] >= 2


def test_default_split_count_fills_one_wave() -> None:
    assert cake_backend.default_split_count(1, 148, 2) == 296
    assert cake_backend.default_split_count(64, 148, 2) == 4
    assert cake_backend.default_split_count(1000, 148, 2) == 1
    assert cake_backend.default_split_count(8, 160, 1) == 20


def test_workspace_bytes_match_the_pool_layout() -> None:
    sizes = msa_icp_indexer_workspace_bytes(2, 256)
    assert sizes == workspace_bytes(2, 256)
    assert sizes["heads_local"] == 2
    assert sizes["par_stride"] == 256 * 2 * 2 * TOPK
    assert sizes["pool_bytes"] == 2 * sizes["par_stride"] * 8
    assert workspace_bytes(4, 64)["heads_local"] == 1
    with pytest.raises(ValueError):
        workspace_bytes(3, 64)
    with pytest.raises(ValueError):
        workspace_bytes(2, 0)


def test_public_api_rejects_other_backends_before_touching_cuda() -> None:
    t = torch.zeros((4, NUM_HEADS, 128), dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="backend"):
        prepare_msa_icp_indexer(
            t, t, t, t, t, t, t, t, t, t, workspace=None, query_len=1, backend="trtllm"
        )
    with pytest.raises(TypeError):
        destroy_msa_icp_indexer_workspace(object())


# ---------------------------------------------------------------------------
# GPU helpers
# ---------------------------------------------------------------------------


def _supported_device(index: int = 0) -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(index)
        in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program() -> torch.device:
    if not _supported_device():
        pytest.skip("requires a compute capability 10.0 or 10.3 device")
    device = torch.device("cuda", torch.cuda.current_device())
    if not cake_backend.generated_program_available(device):
        pytest.skip(
            "this checkout registers no generated ICP indexer program for the device"
        )
    return device


def _rank_inputs(case, device):
    first = build_rank_inputs(case, 0, device)
    inputs = [first]
    for rank in range(1, case.world_size):
        inputs.append(
            build_rank_inputs(
                case,
                rank,
                device,
                host_keys=first.full_index_keys_host,
                host_q=first.index_q.detach().cpu(),
            )
        )
    return inputs


def _prepare(inp, workspace, query_len):
    return prepare_msa_icp_indexer(
        inp.index_q,
        inp.index_k,
        inp.block_table,
        inp.query_start_loc,
        inp.seq_lens,
        inp.positions,
        inp.active_rows_u8,
        inp.forced_column,
        inp.forced_block,
        inp.n_ordinary,
        workspace=workspace,
        query_len=query_len,
    )


def _emulated_step(runners) -> None:
    """One decode step of every emulated rank: scorer + push per rank, then poll per rank."""
    for runner in runners:
        runner.launch_scorer()
        runner.launch_merge(cake_backend.PHASE_PUSH_ONLY)
    for runner in runners:
        runner.launch_merge(cake_backend.PHASE_POLL_ONLY)


# ---------------------------------------------------------------------------
# Single GPU: emulated ranks
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case_name", sorted(SMALL_CASES))
def test_emulated_chain_matches_the_oracle(case_name: str) -> None:
    device = _require_program()
    case = build_case(case_name, SMALL_CASES[case_name])
    inputs = _rank_inputs(case, device)
    workspaces = MSAIcpIndexerWorkspace.emulated(case.world_size, case.tokens, device)
    try:
        runners = [
            _prepare(inputs[r], workspaces[r], case.query_len)
            for r in range(case.world_size)
        ]
    except NotImplementedError as exc:
        pytest.skip(f"geometry not registered in this checkout: {exc}")
    for runner in runners:
        assert runner.plan.world_size == case.world_size
        assert runner.plan.split_count == cake_backend.default_split_count(
            case.batch,
            torch.cuda.get_device_properties(device).multi_processor_count,
            runner.plan.ctas_per_sm,
        )
    lists = [local_topk_lists(case, inputs[r]) for r in range(case.world_size)]
    expected = [expected_out(case, lists, inputs[r], r) for r in range(case.world_size)]

    def check(tag: str) -> None:
        torch.cuda.synchronize(device)
        for r, runner in enumerate(runners):
            assert workspaces[r].status_value() == 0, (
                f"{tag}: rank {r} status {workspaces[r].status_value()}"
            )
            assert torch.equal(runner.out.cpu(), expected[r]), (
                f"{tag}: rank {r} merged output differs from the oracle"
            )
            match = candidates_match(case, inputs[r], lists[r], runner.candidates)
            assert match["passed"], f"{tag}: rank {r} local lists differ: {match}"

    _emulated_step(runners)
    check("first step")
    # A second step (new generation, same inputs) reproduces the output bit for bit.
    for runner in runners:
        runner.out.fill_(OUT_POISON)
        runner.candidates.fill_(float("nan"))
    _emulated_step(runners)
    check("second step")

    # CUDA Graph: the serialised step captured once, replayed twice.
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    graph = torch.cuda.CUDAGraph()
    for runner in runners:
        runner.out.fill_(OUT_POISON)
    with torch.cuda.graph(graph, stream=stream):
        _emulated_step(runners)
    for replay in range(2):
        for runner in runners:
            runner.out.fill_(OUT_POISON)
        graph.replay()
        check(f"graph replay {replay}")

    # A launch allocates nothing.
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_stats(device)
    _emulated_step(runners)
    torch.cuda.synchronize(device)
    after = torch.cuda.memory_stats(device)
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert after["allocation.all.freed"] == before["allocation.all.freed"]


def test_validation_rejects_mismatched_inputs() -> None:
    device = _require_program()
    case = build_case("tp2_q4_b8_ctx1500", SMALL_CASES["tp2_q4_b8_ctx1500"])
    inp = build_rank_inputs(case, 0, device)
    workspaces = MSAIcpIndexerWorkspace.emulated(2, case.tokens, device)
    kwargs = dict(
        index_q=inp.index_q,
        index_k=inp.index_k,
        block_table=inp.block_table,
        query_start_loc=inp.query_start_loc,
        seq_lens=inp.seq_lens,
        positions=inp.positions,
        active_rows=inp.active_rows_u8,
        forced_column=inp.forced_column,
        forced_block=inp.forced_block,
        n_ordinary=inp.n_ordinary,
        workspace=workspaces[0],
        query_len=case.query_len,
        out=None,
        candidates=None,
    )
    cake_backend.validate_msa_icp_indexer_inputs(**kwargs)
    with pytest.raises(ValueError):
        cake_backend.validate_msa_icp_indexer_inputs(**{**kwargs, "query_len": 3})
    with pytest.raises(ValueError):
        # every per-token tensor is exactly [total_tokens] (the launch window is the whole batch)
        cake_backend.validate_msa_icp_indexer_inputs(
            **{
                **kwargs,
                "forced_block": torch.cat([inp.forced_block, inp.forced_block[:1]]),
            }
        )
    with pytest.raises(ValueError):
        cake_backend.validate_msa_icp_indexer_inputs(
            **{**kwargs, "positions": inp.positions.to(torch.int32)}
        )
    with pytest.raises(ValueError):
        cake_backend.validate_msa_icp_indexer_inputs(
            **{**kwargs, "active_rows": inp.active_rows_bool}
        )
    with pytest.raises(ValueError):
        cake_backend.validate_msa_icp_indexer_inputs(
            **{
                **kwargs,
                "out": torch.empty(
                    (case.tokens, 4, TOPK), dtype=torch.int32, device=device
                ),
            }
        )
    small = MSAIcpIndexerWorkspace.emulated(2, case.tokens - 1, device)[0]
    with pytest.raises(ValueError):
        cake_backend.validate_msa_icp_indexer_inputs(**{**kwargs, "workspace": small})
    four = MSAIcpIndexerWorkspace.emulated(4, case.tokens, device)[0]
    with pytest.raises(ValueError):
        # rows_per_rank 64 x world 4 = 256-token pages, admitted for two ranks only
        cake_backend.validate_msa_icp_indexer_inputs(**{**kwargs, "workspace": four})


# ---------------------------------------------------------------------------
# Multi GPU: the real symmetric-memory workspace over NCCL
# ---------------------------------------------------------------------------


def _get_open_port() -> int:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind(("127.0.0.1", 0))
            return int(sock.getsockname()[1])
    except OSError:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as sock:
            sock.bind(("::1", 0))
            return int(sock.getsockname()[1])


def _distributed_worker(
    rank: int, world: int, port: int, case_name: str, queue
) -> None:
    import torch.distributed as dist

    result: dict = {"rank": rank}
    try:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://127.0.0.1:{port}",
            rank=rank,
            world_size=world,
            device_id=device,
        )
        case = build_case(case_name, SMALL_CASES[case_name])
        inp = build_rank_inputs(case, rank, device)
        workspace = create_msa_icp_indexer_workspace(
            world_size=world, rank=rank, tokens_cap=case.tokens, device=device
        )
        runner = _prepare(inp, workspace, case.query_len)
        result["plan"] = runner.plan.as_dict()
        lists = local_topk_lists(case, inp)
        gathered: list = [None] * world
        dist.all_gather_object(gathered, lists)
        expected = expected_out(case, gathered, inp, rank)

        def step_ok() -> tuple[bool, int, dict]:
            torch.cuda.synchronize(device)
            status = workspace.status_value()
            same = bool(torch.equal(runner.out.cpu(), expected))
            match = candidates_match(case, inp, lists, runner.candidates)
            return same and status == 0 and bool(match["passed"]), status, match

        runner()
        ok, status, match = step_ok()
        result["first"] = dict(ok=ok, status=status, mismatches=match["mismatches"])
        runner.out.fill_(OUT_POISON)
        runner()
        ok2, status2, _ = step_ok()
        result["second"] = dict(ok=ok2, status=status2)

        stream = torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        graph = torch.cuda.CUDAGraph()
        runner.out.fill_(OUT_POISON)
        dist.barrier()
        with torch.cuda.graph(graph, stream=stream):
            runner()
        replays = []
        for _ in range(3):
            runner.out.fill_(OUT_POISON)
            torch.cuda.synchronize(device)
            dist.barrier()
            graph.replay()
            ok_r, status_r, _ = step_ok()
            replays.append(dict(ok=ok_r, status=status_r))
        result["replays"] = replays

        torch.cuda.synchronize(device)
        dist.barrier()
        before = torch.cuda.memory_stats(device)
        runner()
        torch.cuda.synchronize(device)
        after = torch.cuda.memory_stats(device)
        result["allocations"] = dict(
            allocated=after["allocation.all.allocated"]
            - before["allocation.all.allocated"],
            freed=after["allocation.all.freed"] - before["allocation.all.freed"],
        )
        dist.barrier()
        destroy_msa_icp_indexer_workspace(workspace)
        dist.destroy_process_group()
        result["passed"] = (
            result["first"]["ok"]
            and result["second"]["ok"]
            and all(r["ok"] for r in replays)
            and result["allocations"] == dict(allocated=0, freed=0)
        )
    except Exception as exc:  # pragma: no cover - reported to the parent
        result["error"] = repr(exc)
        result["traceback"] = traceback.format_exc()
        result["passed"] = False
    queue.put(result)


def _multi_gpu_ready(world: int) -> bool:
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        return False
    capabilities = {torch.cuda.get_device_capability(i) for i in range(world)}
    return (
        len(capabilities) == 1
        and next(iter(capabilities)) in cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
    )


@pytest.mark.parametrize("case_name", MULTI_GPU_CASES)
def test_distributed_chain_matches_the_oracle(case_name: str) -> None:
    case = build_case(case_name, SMALL_CASES[case_name])
    world = case.world_size
    if not _multi_gpu_ready(world):
        pytest.skip(
            f"requires {world} devices of one supported compute capability (10.0 or 10.3)"
        )
    if not cake_backend.generated_program_available(torch.device("cuda", 0)):
        pytest.skip(
            "this checkout registers no generated ICP indexer program for the device"
        )
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    port = _get_open_port()
    processes = [
        ctx.Process(
            target=_distributed_worker, args=(rank, world, port, case_name, queue)
        )
        for rank in range(world)
    ]
    for process in processes:
        process.start()
    results = {}
    for _ in processes:
        rank_result = queue.get(timeout=WORKER_TIMEOUT_SECONDS)
        results[rank_result["rank"]] = rank_result
    for process in processes:
        process.join(timeout=60)
    assert sorted(results) == list(range(world)), results
    for rank in range(world):
        result = results[rank]
        assert "error" not in result, (
            f"rank {rank}: {result.get('error')}\n{result.get('traceback')}"
        )
        assert result["passed"], f"rank {rank}: {result}"
