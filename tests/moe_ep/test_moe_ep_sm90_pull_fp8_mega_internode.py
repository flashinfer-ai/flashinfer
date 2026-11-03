"""Multi-node (NCCL GIN) SM90 FP8 pull MegaMoE tests.

The internode path (``enable_nccl_gin=True``, ``ranks_per_node=R``) splits the
EP world into NVLink domains of ``R`` ranks: tokens cross nodes once per
destination node through NCCL GIN puts between rail peers, which replay the
remote routing for their node's expert owners (``TokenInPullTokenBackPush``
in the SM90 pull kernel tree's ``src/src/token_comm.py``).

Launch on every node (one process per GPU)::

    torchrun --nnodes 2 --node_rank $NODE --nproc_per_node 4 \\
        --master_addr $MASTER --master_port 29560 \\
        -m pytest tests/moe_ep/test_moe_ep_sm90_pull_fp8_mega_internode.py -q

Requirements: nccl4py with the CuTe GIN device API and a matching NCCL
(``NCCL_HOME`` pointing at it), RDMA NICs with GPUDirect, and every node's
ranks on one NVLink domain.  The tests skip when the job is single-node.

Each launch is checked two ways:

* bitwise against the internode reduction tree (I11) rebuilt in torch from the
  per-(token, topk) FC2 rows the kernel itself produced -- the origin's local
  cells for its own node's experts and, for remote experts, the landing rows
  at the rail peer on the expert's node; this pins the rail-peer reduce, the
  RDMA return and the internode TopkReduce exactly;
* against the pure-torch global oracle (``compute_megamoe_reference_fp8``)
  within the single-GPU oracle tolerances (first launch).

Several launches with fresh inputs exercise the cross-launch protocol
(monotonic signals and payload flags, no rail barrier).
"""

from __future__ import annotations

import os

import pytest

from tests.moe_ep.test_moe_ep_sm90_pull_fp8_mega_multirank import (
    FC1_ACT_SCALE,
    _make_bf16_weights,
    _make_inputs,
    _preprocess_weights,
)


def _internode_ranks():
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    per_node = int(os.environ.get("LOCAL_WORLD_SIZE", str(world)))
    return rank, world, local, per_node


def _require_internode():
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    rank, world, local, per_node = _internode_ranks()
    if world <= per_node:
        pytest.skip("internode tests need a multi-node torchrun job")
    try:
        from nccl.core.device.cute import Gin  # noqa: F401
    except ImportError:
        pytest.skip("nccl4py with the CuTe GIN device API is not installed")
    major, _ = torch.cuda.get_device_capability()
    if major != 9:
        pytest.skip("SM90 only")
    return rank, world, local, per_node


def _init_dist(local_rank: int):
    import torch
    import torch.distributed as dist

    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        # gloo: the GIN arena brings its own NCCL communicator; keeping torch
        # off NCCL avoids two NCCL builds in one process.
        dist.init_process_group("gloo")


def _gather_cpu(t):
    """all_gather over gloo (CPU), stacked on a leading rank dim, back on GPU."""
    import torch
    import torch.distributed as dist

    tc = t.contiguous()
    byte_wire = tc.element_size() == 1 and tc.dtype != torch.uint8
    wire = (tc.view(torch.uint8) if byte_wire else tc).cpu()
    out = [torch.empty_like(wire) for _ in range(dist.get_world_size())]
    dist.all_gather(out, wire)
    stacked = torch.stack(out).to(t.device)
    return stacked.view(tc.dtype) if byte_wire else stacked


_DBG_PHASES = {
    1: "own prep",
    2: "D2 sync",
    3: "D3 puts",
    4: "D5 META",
    5: "D6 sync",
    6: "D7 proxy prep",
    7: "D8 sync",
    8: "D10 node barrier",
    9: "pull pass0",
    10: "PAY wait (CTA0)",
    11: "payload flags",
    12: "pull done",
    13: "ER node seg1 done",
    14: "ER reduce/puts done",
    15: "ER node odd phase done",
    20: "tail entry",
    21: "B1",
    22: "node reduce",
    23: "reduce sync",
    25: "RET wait (CTA0)",
    27: "tail done",
}


def _debug_buffer():
    """Developer-only stamp buffer (MEGA_GIN_DEBUG=1): pinned, device-visible."""
    import torch

    if os.environ.get("MEGA_GIN_DEBUG", "0") != "1":
        return None
    buf = torch.zeros(256 * 64, dtype=torch.int32, pin_memory=True)
    os.environ["MEGA_GIN_DEBUG_ADDR"] = str(buf.data_ptr())
    return buf


def _wait_with_stamps(dbg, rank, launch, timeout_s=120.0):
    import time

    import torch

    ev = torch.cuda.Event()
    ev.record()
    if dbg is None:
        torch.cuda.synchronize()
        return
    t0 = time.time()
    while not ev.query():
        if time.time() - t0 > timeout_s:
            stamps = dbg.view(256, 64)
            summary = ", ".join(
                f"{name}={int((stamps[:, p] != 0).sum())}"
                for p, name in sorted(_DBG_PHASES.items())
            )
            print(
                f"[internode rank {rank} launch {launch}] TIMEOUT stamps: {summary}",
                flush=True,
            )
            os._exit(3)
        time.sleep(0.05)


def _unreduced_plane(symm_buffer):
    """The rank's ((1+S)*T, K, H) bf16 combine plane (local cells + landing rows)."""
    import torch

    frontend = symm_buffer._frontend
    mega = frontend._mega
    kernel = mega.kernel
    spec = kernel._shared_region_by_name["unreduced_expert_output"]
    off = int(kernel._shared_offsets["unreduced_expert_output"])
    nbytes = int(spec.nbytes)
    return (
        mega.shared_workspace[off : off + nbytes].view(torch.bfloat16).view(*spec.shape)
    )


def _i11_reference(
    rank, world, per_node, *, n, t_max, topk_ids, planes, experts_per_rank
):
    """Rebuild this rank's output with the internode reduction tree (bitwise)."""
    import torch

    my_node = rank // per_node
    local = rank % per_node
    node_experts = per_node * experts_per_rank
    num_nodes = world // per_node
    hidden = planes.shape[-1]
    out = torch.zeros(n, hidden, dtype=torch.bfloat16, device=planes.device)
    for t in range(n):
        ids = topk_ids[t].tolist()
        nodes = [e // node_experts if e >= 0 else -1 for e in ids]
        acc = torch.zeros(hidden, dtype=torch.float32, device=planes.device)
        for k, d in enumerate(nodes):
            if d == my_node:
                acc = acc + planes[rank, t, k].float()
        for d in range(num_nodes):
            if d == my_node or d not in nodes:
                continue
            rail_peer = d * per_node + local
            slot = my_node if my_node < d else my_node - 1
            row = t_max + slot * t_max + t
            part = torch.zeros(hidden, dtype=torch.float32, device=planes.device)
            for k, dk in enumerate(nodes):
                if dk == d:
                    part = part + planes[rail_peer, row, k].float()
            acc = acc + part.to(torch.bfloat16).float()
        out[t] = acc.to(torch.bfloat16)
    return out


def _oracle_reference(
    rank, problem, x_local, x_sf_local, transformed_l1, transformed_l2
):
    import torch

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        Fp8BlockScaleK,
        compute_megamoe_reference_fp8,
    )

    fp8_scale_mode = problem["fp8_scale_mode"]
    hidden = problem["hidden"]
    x_g = _gather_cpu(x_local)
    idx_g = _gather_cpu(problem["topk_ids"])
    w_g = _gather_cpu(problem["topk_weights"])
    fc1_w_g = _gather_cpu(transformed_l1[0].mT).mT
    fc2_w_g = _gather_cpu(transformed_l2[0].mT).mT
    fc1_sf_g = _gather_cpu(transformed_l1[1])
    fc2_sf_g = _gather_cpu(transformed_l2[1])
    common = dict(
        input_activation=x_g,
        input_topk_idx=idx_g,
        input_topk_weights=w_g,
        fc1_weight=fc1_w_g,
        fc1_weight_sf=fc1_sf_g,
        fc2_weight=fc2_w_g,
        fc2_weight_sf=fc2_sf_g,
        ab_dtype=torch.float8_e4m3fn,
        ref_compute_graph="deepgemm",
        fp8_accum_mode="1xacc",
        mma_tiler_k=128,
        fc2_output_dtype=torch.bfloat16,
        gate_up_clamp=problem["gate_up_clamp"],
        fp8_scale_mode=fp8_scale_mode,
    )
    if fp8_scale_mode == "blockwise":
        sf_cols = hidden // Fp8BlockScaleK
        x_sf_g = _gather_cpu(x_sf_local[:, :sf_cols].clone())
        ref = compute_megamoe_reference_fp8(
            input_activation_sf=x_sf_g,
            fc1_activation_block_scale=x_sf_g,
            fc1_weight_block_scale=fc1_sf_g,
            fc2_weight_block_scale=fc2_sf_g,
            fc2_activation_block_scale=None,
            **common,
        )
    else:
        ref = compute_megamoe_reference_fp8(
            input_activation_sf=_gather_cpu(x_sf_local),
            fc1_activation_dequant_scale=transformed_l1[2],
            fc1_weight_dequant_scale=_gather_cpu(transformed_l1[3]),
            fc2_activation_dequant_scale=transformed_l2[2],
            fc2_weight_dequant_scale=_gather_cpu(transformed_l2[3]),
            **common,
        )
    return ref[rank].to(torch.float32).sum(dim=1)


def _run_internode(
    *,
    fp8_scale_mode: str,
    swap_ab: bool,
    token_back_mode=None,
    max_tokens: int = 64,
    token_counts=(64, 48, 64),
    experts_per_rank: int = 4,
    topk: int = 4,
    hidden: int = 2048,
):
    import torch
    import torch.distributed as dist

    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.staging import (
        stage_mega_moe_inputs,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        get_symm_buffer_for_hopper_fp8_mega_moe,
        hopper_fp8_mega_moe,
    )

    rank, world, local, per_node = _require_internode()
    _init_dist(local)
    dbg = _debug_buffer()
    num_experts = experts_per_rank * world
    intermediate = 1024
    gate_up_clamp = 10.0
    w13, w2 = _make_bf16_weights(
        rank,
        num_local_experts=experts_per_rank,
        hidden=hidden,
        intermediate=intermediate,
    )
    base_problem = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        topk=topk,
        gate_up_clamp=gate_up_clamp,
        kind="fp8_e4m3",
        fp8_scale_mode=fp8_scale_mode,
        w13=w13,
        w2=w2,
    )
    transformed_l1, transformed_l2 = _preprocess_weights(base_problem)
    symm = get_symm_buffer_for_hopper_fp8_mega_moe(
        num_experts,
        max_tokens,
        topk,
        hidden,
        intermediate,
        rank,
        world,
        kind="fp8_e4m3",
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        gate_up_clamp=gate_up_clamp,
        token_back_mode=token_back_mode,
        ranks_per_node=per_node,
        enable_nccl_gin=True,
    )
    try:
        for launch, n in enumerate(token_counts):
            hs, tw, ti = _make_inputs(
                rank + 1000 * launch,
                world,
                num_tokens=n,
                hidden=hidden,
                num_experts=num_experts,
                topk=topk,
            )
            stage_mega_moe_inputs(
                hs,
                tw,
                ti,
                symm.x,
                symm.x_sf,
                symm.topk_idx,
                symm.topk_weights,
                kind="fp8_e4m3",
                fp8_scale_mode=fp8_scale_mode,
                fc1_activation_dequant_scale=FC1_ACT_SCALE,
            )
            x_local = symm.x[:n].clone()
            x_sf_local = symm.x_sf[:n].clone()
            y = torch.empty(n, hidden, dtype=torch.bfloat16, device="cuda")
            if dbg is not None:
                dbg.zero_()
            hopper_fp8_mega_moe(
                y,
                transformed_l1,
                transformed_l2,
                symm,
                num_tokens=n,
                gate_up_clamp=gate_up_clamp,
                fast_math=True,
            )
            _wait_with_stamps(dbg, rank, launch)
            torch.cuda.synchronize()
            dist.barrier()

            planes = _gather_cpu(_unreduced_plane(symm))
            y_tree = _i11_reference(
                rank,
                world,
                per_node,
                n=n,
                t_max=max_tokens,
                topk_ids=ti,
                planes=planes,
                experts_per_rank=experts_per_rank,
            )
            diff = (y.view(torch.int16) != y_tree.view(torch.int16)).sum().item()
            remote_hits = sum(
                1
                for e in ti.flatten().tolist()
                if e >= 0 and e // (per_node * experts_per_rank) != rank // per_node
            )
            print(
                f"[internode rank {rank} launch {launch} n={n} {fp8_scale_mode} "
                f"swap_ab={swap_ab} tb={token_back_mode}] remote (token,k) hits="
                f"{remote_hits}, bitwise mismatches vs I11 tree={diff}",
                flush=True,
            )
            failure = None
            if diff != 0:
                failure = f"rank {rank} launch {launch}: {diff} mismatching elements"
            if launch == 0:
                problem = dict(base_problem, topk_ids=ti, topk_weights=tw)
                y_ref = _oracle_reference(
                    rank,
                    problem,
                    x_local,
                    x_sf_local,
                    transformed_l1,
                    transformed_l2,
                )
                yk = y.float()
                rel = ((yk - y_ref).norm() / y_ref.norm().clamp_min(1e-6)).item()
                print(f"[internode rank {rank}] oracle rel_l2={rel:.4g}", flush=True)
                # The fast-math FP8 scales flip a few FC1-output roundings
                # against the oracle (2048 tokens: max |d| 0.018 on one rank,
                # the same before the two-phase return); the band of the
                # single-node grouped-combine check.
                try:
                    torch.testing.assert_close(yk, y_ref, atol=2e-2, rtol=2e-2)
                    assert rel < 0.02, f"oracle rel_l2 {rel:.4g}"
                except AssertionError as e:
                    failure = failure or f"rank {rank} oracle: {e}"
            # Every rank learns of a failure before the next collective: a
            # rank that raised alone would pair its cleanup barrier with the
            # others' loop barrier and leave them inside the next launch.
            # Cross-node traffic is counted over the whole job: in a one-token
            # launch the forced routing of token 0 keeps node 0's ranks local.
            stats = torch.tensor(
                [remote_hits, int(failure is not None)], dtype=torch.int64
            )
            dist.all_reduce(stats)
            assert failure is None, failure
            assert stats[1].item() == 0, (
                f"{stats[1].item()} rank(s) failed launch {launch}"
            )
            assert stats[0].item() > 0, "routing produced no cross-node traffic"
            dist.barrier()
    finally:
        torch.cuda.synchronize()
        dist.barrier()
        symm.destroy()


@pytest.mark.parametrize(
    "fp8_scale_mode,swap_ab,token_back_mode,max_tokens",
    [
        ("per_tensor", True, None, 64),
        ("per_tensor", True, "reuse_dispatch_warps", 64),
        ("per_tensor", False, None, 64),
        ("blockwise", True, None, 64),
        ("blockwise", False, "reuse_dispatch_warps", 64),
        # The non-swap epi_warps early return: one return phase at 1024
        # tokens per rank, two (and the split local FC1) from 2048.
        ("blockwise", False, "epi_warps", 1024),
        ("blockwise", False, "epi_warps", 2048),
    ],
)
def test_moe_ep_sm90_pull_fp8_mega_internode(
    fp8_scale_mode, swap_ab, token_back_mode, max_tokens
):
    # Decode-size launches (an expert's rows all in the first remote segment,
    # a return phase without tokens) and an odd count (the phases differ by
    # one token).
    _run_internode(
        fp8_scale_mode=fp8_scale_mode,
        swap_ab=swap_ab,
        token_back_mode=token_back_mode,
        max_tokens=max_tokens,
        token_counts=(max_tokens, 16, 1, max_tokens * 3 // 4 - 1),
    )


@pytest.mark.parametrize(
    "swap_ab,mma_tiler_mnk,tokens",
    [
        (True, (256, 32, 128), 64),
        (False, (64, 128, 128), 64),
        (False, (64, 128, 128), 1024),
        (False, (64, 128, 128), 2048),
    ],
)
def test_moe_ep_sm90_pull_fp8_internode_kernel_layout(
    monkeypatch, swap_ab, mma_tiler_mnk, tokens
):
    """Host side, no launch: the internode kernel builds from the shim config,
    and the per-origin token counts sit in the counter prefix kernel_tail
    zeroes before the next launch.  From 1024 tokens per rank the non-swap
    epi_warps layout takes the early return, from 2048 in two phases."""
    from types import SimpleNamespace

    import cutlass.utils as cutlass_utils

    import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as pkg

    # The only device query on the build path (occupancy-aware group size).
    monkeypatch.setattr(
        cutlass_utils,
        "HardwareInfo",
        lambda: SimpleNamespace(get_max_active_clusters=lambda cluster_size: 132),
    )
    config = pkg.MegaMoEHopperFp8Config(
        rank=0,
        world_size=8,
        num_tokens_per_rank=tokens,
        num_topk=4,
        num_total_experts=16,
        hidden=1024,
        intermediate=512,
        swap_ab=swap_ab,
        mma_tiler_mnk=mma_tiler_mnk,
        ranks_per_node=4,
        enable_nccl_gin=True,
    )
    kernel, _ = pkg.MegaMoEHopperFp8Frontend(config)._build_kernel()
    names = [spec.name for spec in kernel._local_region_specs]
    # Data buffers start at l1_token_buffer; everything before it is reset.
    assert names.index("gin_token_count") < names.index("l1_token_buffer")
    # The epilogue warps run the node reduce from their own branch, or, with
    # the early return, the dispatch warps reduce during the deferred local FC2.
    comm = kernel.token_comm
    assert comm.num_tail_reduce_warps == len(kernel.epilogue_warp_id)
    assert comm.early_return == (not swap_ab and tokens >= 1024)
    assert comm.tail_reduce_in_epilogue == (not comm.early_return)
    two_phases = comm.early_return and tokens >= 2048
    assert comm.ret_phases == (2 if two_phases else 1)
    # Two return phases also split the local FC1: two thirds of the experts
    # stay ahead of the remote blocks.
    assert kernel.sched_seg0_fc1_experts == (
        round(kernel.num_experts_per_rank * 2 / 3) if two_phases else None
    )
    if comm.early_return:
        assert kernel.sched_defer_local_fc2
        assert kernel.sched_ret_phases == comm.ret_phases
        for region in ("gin_seg1_done", "gin_ret_chunk_done", "expert_send_count"):
            assert names.index(region) < names.index("l1_token_buffer")
        # One reservation set for the own tokens, then one per proxied
        # remote origin and return phase.
        send_count = kernel._local_region_by_name["expert_send_count"]
        assert send_count.shape == (
            config.num_total_experts * (1 + kernel.num_remote_slots * comm.ret_phases),
        )
