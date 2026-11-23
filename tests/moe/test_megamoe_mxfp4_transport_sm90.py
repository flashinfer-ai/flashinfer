"""NCCL-window transport of the SM90 MXFP4 MegaMoE kernel.

The same inputs go through a torch-symmetric-memory buffer and through an
NCCL-window buffer; inside one NVLink domain both must produce the same output,
with plain resident experts and with redundant expert slots.

Run with plain Python (workers are spawned here):
    python tests/moe/test_megamoe_mxfp4_transport_sm90.py --ep 8
"""

import argparse
import datetime
import functools
import os
import sys

import pytest
import torch
import torch.distributed as dist

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from tests.moe.megamoe_sm90_utils import make_input, make_weights, resolve_tokens  # noqa: E402
from tests.moe.test_megamoe_fp4_sm90 import _reference  # noqa: E402
from tests.moe.test_megamoe_mxfp4_eplb_sm90 import (  # noqa: E402
    BITWISE_REPEAT_MAX_TOKENS,
    _assert_close,
)


def _reference_weights(raw):
    """Decode Marlin nibbles independently of the storage transform under test."""
    packed, scales = raw
    e, n, half_k = packed.shape
    words = packed.reshape(e, n, -1, 4)
    codes = torch.cat((words >> 4, words & 15), dim=-1).reshape(e, n, 2 * half_k)
    lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=packed.device)
    values = lattice[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    values *= torch.exp2(scales.float() - 127).repeat_interleave(32, dim=-1)
    values = torch.where(values.abs() < 2**-6, 0.0, values)
    return values.to(torch.float8_e4m3fn).float()


def _check_graph(run, y, stats, expected, counts, m, label):
    """Check repeated graph calls with fresh statistics and the original inputs."""
    graph = torch.cuda.CUDAGraph()
    dist.barrier()
    torch.cuda.synchronize()
    with torch.cuda.graph(graph):
        stats.zero_()
        run()
    for _ in range(2):
        graph.replay()
        torch.cuda.synchronize()
        _same(y, expected, m, label)
        assert torch.equal(stats, counts), label


def _same(y, expected, m, label):
    """Both transports must agree: bitwise at small ``m``, within tolerance above."""
    torch.cuda.synchronize()
    if m <= BITWISE_REPEAT_MAX_TOKENS:
        assert torch.equal(y, expected), label
    else:
        _assert_close(y, expected, label)


def _check_route_cycle(
    buffer, inputs, weights, reference_weights, global_scales, rank, local_experts
):
    """Replay one graph as a persistent buffer changes local and remote routes."""
    from flashinfer.megamoe import mxfp4_mega_moe

    x, sf, routing = inputs
    columns = torch.arange(routing.shape[1], device=x.device)[None, :].expand_as(
        routing
    )
    local = rank * local_experts + columns
    remote = (rank ^ (dist.get_world_size() // 2)) * local_experts + columns
    routes = (
        ("remote", remote),
        ("local", local),
        ("masked", torch.full_like(local, -1)),
        ("mixed", torch.where((columns & 1).bool(), remote, local)),
        ("remote_again", remote),
    )
    y = torch.empty(x.shape, dtype=torch.bfloat16, device=x.device)
    stats = torch.zeros(local_experts, dtype=torch.int32, device=x.device)
    run = functools.partial(
        mxfp4_mega_moe,
        y,
        *weights,
        buffer,
        cumulative_local_expert_recv_stats=stats,
        l1_global_scales=global_scales[0],
        l2_global_scales=global_scales[1],
    )
    graph, first_remote = None, None
    for name, idx in routes:
        expected, counts = _reference(
            x.float() * sf.repeat_interleave(128, dim=-1),
            idx,
            routing,
            *reference_weights,
            *global_scales,
            rank,
            local_experts,
        )
        buffer.stage(x, sf, idx, routing)
        stats.zero_()
        run()
        torch.cuda.synchronize()
        _assert_close(y, expected, ("route cycle reference", name))
        assert torch.equal(stats.long(), counts), ("route cycle eager counts", name)
        eager = y.clone()
        if first_remote is None:
            first_remote = eager
        elif name == "remote_again":
            assert torch.equal(y.view(torch.uint8), first_remote.view(torch.uint8))
        if graph is None:
            graph = torch.cuda.CUDAGraph()
            dist.barrier()
            torch.cuda.synchronize()
            with torch.cuda.graph(graph):
                stats.zero_()
                run()
        for _ in range(2):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y.view(torch.uint8), eager.view(torch.uint8)), name
            assert torch.equal(stats.long(), counts), ("route cycle graph counts", name)
    graph.reset()


def _worker(rank, args):
    """Per-rank body: build both buffers, run identical inputs through each and compare."""
    local_rank = int(
        os.environ.get("LOCAL_RANK", rank % max(torch.cuda.device_count(), 1))
    )
    from flashinfer.megamoe import (
        MegaMoEExpertSlots,
        MegaMoENcclTransport,
        MegaMoESymmBuffer,
        mxfp4_mega_moe,
        transform_mxfp4_weights_for_mega_moe,
    )

    ep = args.ep
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=ep,
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", local_rank),
    )
    group = dist.group.WORLD
    h, ih, E, topk, S = (
        args.hidden,
        args.intermediate,
        args.experts,
        args.topk,
        args.slots,
    )
    H = E // ep
    raw = make_weights(rank * H, H, h, ih, args.seed, "cuda")
    ref1, ref2 = (_reference_weights(value) for value in raw)
    w1, w2 = transform_mxfp4_weights_for_mega_moe(*raw, storage=args.weight_storage)
    del raw
    gs1 = torch.linspace(0.73, 1.37, E, device="cuda")[rank * H : (rank + 1) * H]
    gs2 = torch.linspace(1.31, 0.67, E, device="cuda")[rank * H : (rank + 1) * H]
    hot = torch.randperm(E, generator=torch.Generator().manual_seed(31))[:S].tolist()

    transport = MegaMoENcclTransport(
        group,
        gin_contexts=args.gin_contexts,
        gin_signals=2 * (args.nodes - 1) if args.gin_contexts else 0,
    )
    plain, slotted, slots = [], [], []
    if not args.nodes_only:
        # Plain resident experts: one buffer per transport.
        plain = [
            MegaMoESymmBuffer(group, E, args.capacity, topk, h, ih),
            MegaMoESymmBuffer(
                group, E, args.capacity, topk, h, ih, transport=transport
            ),
        ]
        # Redundant expert slots: the buffer geometry counts the slots as experts.
        slotted = (
            [
                MegaMoESymmBuffer(group, E + ep * S, args.capacity, topk, h, ih),
                MegaMoESymmBuffer(
                    group, E + ep * S, args.capacity, topk, h, ih, transport=transport
                ),
            ]
            if S
            else []
        )
        slots = [MegaMoEExpertSlots(b, E, S) for b in slotted]
        for s in slots:
            s.set_resident_weights(w1, w2, gs1, gs2)
            assert s.set_replica_plan(torch.full((ep, S), -1)) == 0
        peers = transport.peer_pointers(plain[1].window)
        assert len(peers) == ep and all(p != 0 for p in peers), peers
        assert peers[rank] == plain[1].buffer.data_ptr()

    nodes_buffer = None
    if args.nodes > 1:
        # Ranks spread over `nodes` nodes (node-major): tokens with an expert on
        # another node travel through the mirror ranks' inboxes and outboxes.
        nodes_buffer = MegaMoESymmBuffer(
            group,
            E,
            args.capacity,
            topk,
            h,
            ih,
            transport=transport,
            num_nodes=args.nodes,
        )

    for m in args.tokens:
        x, sf, idx, routing = make_input(
            m, rank, h, E, topk, args.seed, "cuda", hot=hot
        )
        if getattr(args, "nonuniform_sf", False):
            rows = torch.arange(m, device=x.device, dtype=torch.float32)[:, None]
            chunks = torch.arange(h // 128, device=x.device, dtype=torch.float32)[
                None, :
            ]
            sf = (1 + rank / 16 + rows / 64 + chunks / 256) / 128
            owner = rank ^ (ep // 2)
            idx = (
                owner * H
                + (
                    torch.arange(m, device=x.device)[:, None] * topk
                    + torch.arange(topk, device=x.device)[None, :]
                )
                % H
            )
        if args.route_cycle:
            _check_route_cycle(
                nodes_buffer,
                (x, sf, routing),
                (w1, w2),
                (ref1, ref2),
                (gs1, gs2),
                rank,
                H,
            )
            if rank == 0:
                print(
                    f"Persistent-buffer route cycle passed at {m} tokens per rank",
                    flush=True,
                )
            continue
        expected, counts_logical = _reference(
            x.float() * sf.repeat_interleave(128, dim=-1),
            idx,
            routing,
            ref1,
            ref2,
            gs1,
            gs2,
            rank,
            H,
        )

        if nodes_buffer is not None:
            nodes_buffer.stage(x, sf, idx, routing)
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            stats = torch.zeros(H, dtype=torch.int32, device="cuda")
            run = functools.partial(
                mxfp4_mega_moe,
                y,
                w1,
                w2,
                nodes_buffer,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=gs1,
                l2_global_scales=gs2,
            )
            run()
            torch.cuda.synchronize()
            _assert_close(y, expected, (f"{args.nodes} nodes", m))
            assert torch.equal(stats.long(), counts_logical), (
                f"{args.nodes} nodes counts",
                m,
            )
            _check_graph(
                run, y, stats, y.clone(), counts_logical.int(), m, ("nodes graph", m)
            )
            if rank == 0:
                print(
                    f"{args.nodes}-node protocol check passed at {m} tokens per rank",
                    flush=True,
                )
            if args.nodes_only:
                continue

        outputs, counts = [], []
        for b in plain:
            b.stage(x, sf, idx, routing)
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            stats = torch.zeros(H, dtype=torch.int32, device="cuda")
            run = functools.partial(
                mxfp4_mega_moe,
                y,
                w1,
                w2,
                b,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=gs1,
                l2_global_scales=gs2,
            )
            run()
            _assert_close(y, expected, ("plain reference", m))
            assert torch.equal(stats.long(), counts_logical), ("plain counts", m)
            _check_graph(
                run, y, stats, y.clone(), counts_logical.int(), m, ("plain graph", m)
            )
            outputs.append(y)
            counts.append(stats)
        _same(outputs[1], outputs[0], m, ("plain", m))
        assert torch.equal(counts[0], counts[1]), ("plain counts", m)

        outputs, counts = [], []
        for b, s in zip(slotted, slots, strict=True):
            b.stage(x, sf, idx, routing)
            s.remap(m, s.block_m(m))
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            stats = torch.zeros(H + S, dtype=torch.int32, device="cuda")
            mxfp4_mega_moe(
                y,
                s.l1_weights,
                s.l2_weights,
                b,
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=s.l1_global_scales,
                l2_global_scales=s.l2_global_scales,
                expert_slots=s,
            )
            outputs.append(y)
            counts.append(stats)
        if slots:
            _same(outputs[1], outputs[0], m, ("slots", m))
            assert torch.equal(counts[0], counts[1]), ("slot counts", m)
        if rank == 0:
            print(f"transport check passed at {m} tokens per rank", flush=True)

    torch.cuda.synchronize()
    dist.barrier()
    for s in slots:
        s.destroy()
    for b in plain + slotted:
        b.destroy()
    if nodes_buffer is not None:
        nodes_buffer.destroy()
    transport.close()
    dist.barrier()
    dist.destroy_process_group()


def _run(args):
    """Spawn ``args.ep`` workers on as many GPUs, or run as one torchrun worker
    when RANK / WORLD_SIZE are set (ranks on several hosts)."""
    launched = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if launched:
        args.ep = int(os.environ["WORLD_SIZE"])
    args.ep = args.ep or 8
    local = int(os.environ.get("LOCAL_WORLD_SIZE", args.ep))
    if local < 1:
        raise ValueError("LOCAL_WORLD_SIZE must be positive")
    args.nodes = args.nodes if args.nodes is not None else args.ep // local
    if args.ep not in (4, 8, 16) or args.nodes < 1 or args.ep % args.nodes:
        raise ValueError("EP must be 4, 8 or 16 and divide evenly across nodes")
    if launched and local < args.ep and args.nodes != args.ep // local:
        raise ValueError("nodes must match the torchrun host count")
    if args.nodes > 1:
        args.nodes_only = True
    elif args.nodes_only:
        raise ValueError("nodes-only requires more than one logical node")
    if args.gin_contexts is None:
        args.gin_contexts = 1 if launched and local < args.ep else 0
    if launched and local < args.ep and args.gin_contexts < 1:
        raise ValueError("ranks on separate hosts need at least one GIN context")
    if args.seed < 0 or args.slots < 0 or args.gin_contexts < 0:
        raise ValueError("seed, slots and GIN context count must be nonnegative")
    if args.weight_storage != "mxfp4" and args.slots:
        raise ValueError("replica slots require canonical MXFP4 storage")
    if (
        args.hidden <= 0
        or args.hidden % 256
        or args.intermediate <= 0
        or args.intermediate % 128
    ):
        raise ValueError(
            "hidden must be a positive multiple of 256; intermediate of 128"
        )
    if (
        args.experts < 1
        or args.experts % args.ep
        or not 0 < args.topk <= min(32, args.experts)
    ):
        raise ValueError(
            "experts must divide by EP; topk must be in [1, min(32, experts)]"
        )
    if args.tokens is None:
        args.tokens = (
            [1, 4, 8, 16, 32, 64]
            if args.token_scope == "per-rank"
            else [16, 64, 128, 256, 512, 1024]
        )
    args.tokens = resolve_tokens(args.tokens, args.token_scope, args.ep, args.capacity)
    if args.route_cycle and (
        not launched
        or args.ep != 16
        or args.nodes != 2
        or local != 8
        or (args.hidden, args.intermediate, args.experts, args.topk)
        != (4096, 2048, 256, 6)
        or len(args.tokens) != 1
        or (args.tokens[0], args.capacity) not in ((1, 2), (4, 4), (8, 8))
        or args.weight_storage != "mxfp4_compact"
        or args.slots != 0
    ):
        raise ValueError(
            "route-cycle requires two-host torchrun EP16, E256/H4096/I2048/top6, "
            "compact weights, M1/cap2, M4/cap4 or M8/cap8, and no replica slots"
        )
    if launched:
        _worker(int(os.environ["RANK"]), args)
        return
    if torch.cuda.device_count() < args.ep:
        raise ValueError(f"needs {args.ep} local GPUs, or launch with torchrun")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29571")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)


def _namespace(ep, **overrides):
    """Default test arguments for ``ep`` ranks, with ``overrides`` applied."""
    args = dict(
        ep=ep,
        hidden=512,
        intermediate=256,
        experts=64,
        topk=4,
        slots=2,
        capacity=384,
        tokens=[1, 4, 8, 16, 32, 64],
        token_scope="per-rank",
        weight_storage="mxfp4",
        seed=7919,
        gin_contexts=0,
        nodes=1,
        nodes_only=False,
        route_cycle=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _worker_without_gin(rank, rendezvous):
    """Reject a communicator without GIN before creating its device communicator."""
    os.environ["NCCL_GIN_TYPE"] = "0"
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        init_method=rendezvous,
        rank=rank,
        world_size=1,
        timeout=datetime.timedelta(minutes=2),
        device_id=torch.device("cuda", rank),
    )
    from flashinfer.megamoe import MegaMoENcclTransport

    try:
        with pytest.raises(RuntimeError, match=r"NCCL_GIN_TYPE=2.*selected GIN type 0"):
            MegaMoENcclTransport(dist.group.WORLD, gin_contexts=1)
        transport = MegaMoENcclTransport(dist.group.WORLD, gin_contexts=0)
        transport.close()
    finally:
        dist.destroy_process_group()


def test_mxfp4_nccl_transport_without_gin(tmp_path):
    """Require the proxy backend only when GIN contexts are requested."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("needs an SM90 GPU")
    torch.multiprocessing.spawn(
        _worker_without_gin, args=((tmp_path / "nccl").as_uri(),), nprocs=1
    )


@pytest.mark.parametrize("ep", [4, 8])
def test_mxfp4_nccl_transport(ep):
    """Torch symmetric memory and NCCL windows give the same kernel output."""
    if torch.cuda.device_count() < ep:
        pytest.skip(f"needs {ep} GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    _run(_namespace(ep))


def test_mxfp4_two_virtual_nodes():
    """Eight ranks on one host run as two nodes of four: the inter-node protocol
    over the load/store carrier agrees with the logical reference."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    _run(_namespace(8, nodes=2, nodes_only=True))


def test_mxfp4_two_virtual_nodes_topk32():
    """Exercise all 32 route-mask bits through eager and captured calls."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    _run(
        _namespace(
            8, topk=32, slots=0, capacity=4, tokens=[1, 4], nodes=2, nodes_only=True
        )
    )


@pytest.mark.parametrize("hidden,intermediate", [(256, 128), (768, 256)])
def test_mxfp4_two_virtual_nodes_sf_tail(hidden, intermediate):
    """Copy nonuniform scale rows through odd remote packed slots without padding."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    _run(
        _namespace(
            8,
            hidden=hidden,
            intermediate=intermediate,
            slots=0,
            capacity=3,
            tokens=[3],
            nodes=2,
            nodes_only=True,
            nonuniform_sf=True,
        )
    )


@pytest.mark.parametrize(
    "storage,hidden,intermediate,experts,topk,capacity,tokens",
    [
        ("mxfp4_compact", 2048, 768, 128, 8, 2, [1]),
        ("mxfp4_compact", 2048, 768, 128, 8, 4, [1, 4]),
        ("fp8", 2048, 768, 128, 8, 4, [1, 4]),
        ("mxfp4_compact", 512, 256, 256, 32, 4, [4]),
    ],
)
def test_mxfp4_two_virtual_nodes_prepared_geometry(
    storage, hidden, intermediate, experts, topk, capacity, tokens
):
    """Check prepared records against the dense reference and graph replays."""
    if torch.cuda.device_count() < 8:
        pytest.skip("needs 8 GPUs")
    if torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("SM90 only")
    if torch.cuda.get_device_properties(0).multi_processor_count != 78:
        pytest.skip("prepared plans require 78 SMs")
    _run(
        _namespace(
            8,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            slots=0,
            capacity=capacity,
            tokens=tokens,
            weight_storage=storage,
            nodes=2,
            nodes_only=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--slots", type=int, default=0)
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument("--tokens", type=int, nargs="+")
    parser.add_argument(
        "--token-scope", choices=["per-rank", "total"], default="per-rank"
    )
    parser.add_argument("--seed", type=int, default=7919)
    parser.add_argument(
        "--weight-storage", choices=["mxfp4", "mxfp4_compact", "fp8"], default="mxfp4"
    )
    parser.add_argument("--gin-contexts", type=int)
    parser.add_argument("--nodes", type=int, help="defaults to the torchrun host count")
    parser.add_argument(
        "--nodes-only",
        action="store_true",
        help="skip the single-node pair (ranks on several hosts cannot share torch symmetric memory)",
    )
    parser.add_argument(
        "--route-cycle",
        action="store_true",
        help="check changing M1/M4/M8 routes through one EP16 compact buffer and captured graph",
    )
    _run(parser.parse_args())
