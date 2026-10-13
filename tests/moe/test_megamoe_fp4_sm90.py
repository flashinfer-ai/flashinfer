"""Four/eight-rank SM90 checks with an independent dense W4A8 reference.

Run with plain Python (workers are spawned here):
    python tests/moe/test_megamoe_fp4_sm90.py --tokens 1 32 96 384
No DeepGEMM extension or checkpoint is needed.
"""

import argparse
import datetime
import os

import pytest
import torch
import torch.distributed as dist


def _weights(e, n, k, group, generator):
    codes = torch.randint(
        0, 16, (e, n, k), device="cuda", generator=generator, dtype=torch.uint8
    )
    chunks = codes.reshape(e, n, k // 8, 8)
    packed = (chunks[..., 4:] | (chunks[..., :4] << 4)).reshape(e, n, k // 2)
    level = torch.randint(0, 3, (e, n, k // group), device="cuda", generator=generator)
    sf = (level + (16 if group == 16 else 122)).to(torch.uint8)
    scales = (
        sf.view(torch.float8_e4m3fn).float()
        if group == 16
        else torch.exp2(sf.float() - 127)
    )
    lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    values = lattice[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    values *= scales.repeat_interleave(group, dim=-1)
    if group == 32:
        values = torch.where(values.abs() < 2**-6, 0.0, values)
    return (packed, sf), values.to(torch.float8_e4m3fn).float()


def _gather(t):
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered)


def _reference(x, idx, weights, w1, w2, gs1, gs2, rank, local_experts):
    all_x, all_idx, all_weights = _gather(x), _gather(idx), _gather(weights)
    routes = torch.zeros(
        (*all_idx.shape, w2.shape[1]), device="cuda", dtype=torch.float32
    )
    for expert in range(local_experts):
        rows, slots = torch.where(all_idx == rank * local_experts + expert)
        if rows.numel() == 0:
            continue
        intermediate = all_x[rows] @ w1[expert].T * gs1[expert]
        gate, up = intermediate.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        hidden *= all_weights[rows, slots, None]
        grouped = hidden.reshape(hidden.shape[0], -1, 128)
        sf = torch.exp2(
            torch.ceil(torch.log2((grouped.abs().amax(-1) / 448).clamp_min(2**-126)))
        )
        quantized = (grouped / sf[..., None]).to(torch.float8_e4m3fn).float() * sf[
            ..., None
        ]
        down = quantized.reshape_as(hidden) @ w2[expert].T * gs2[expert]
        routes[rows, slots] = down.to(torch.bfloat16).float()
    dist.all_reduce(routes)
    result = routes.sum(1).to(torch.bfloat16)
    m = x.shape[0]
    counts = torch.stack(
        [(all_idx == rank * local_experts + e).sum() for e in range(local_experts)]
    )
    return result[rank * m : (rank + 1) * m], counts


def _worker(rank, args):
    from flashinfer.megamoe import (
        MegaMoESymmBuffer,
        nvfp4_mega_moe,
        mxfp4_mega_moe,
        transform_nvfp4_weights_for_mega_moe,
        transform_mxfp4_weights_for_mega_moe,
    )

    ep = getattr(args, "ep", 8)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=ep,
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    group = dist.group.WORLD
    h, ih = args.hidden, args.intermediate
    experts, topk = getattr(args, "experts", 256), getattr(args, "topk", 6)
    epr = experts // ep
    generator = torch.Generator(device="cuda").manual_seed(7919 + rank)
    buffer = MegaMoESymmBuffer(
        group, experts, args.capacity, topk, h, ih, fast_math=args.fast_math
    )
    gs1 = torch.linspace(0.73, 1.37, epr, device="cuda")
    gs2 = torch.linspace(1.31, 0.67, epr, device="cuda")
    for fmt in args.formats:
        gran = 16 if fmt == "nvfp4" else 32
        raw1, ref1 = _weights(epr, ih * 2, h, gran, generator)
        raw2, ref2 = _weights(epr, h, ih, gran, generator)
        transform = (
            transform_nvfp4_weights_for_mega_moe
            if fmt == "nvfp4"
            else transform_mxfp4_weights_for_mega_moe
        )
        w1, w2 = transform(raw1, raw2)
        for m in args.tokens:
            # Binary input SFs isolate the CUDA kernels from input quantizer differences.
            x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
                torch.float8_e4m3fn
            )
            sf = torch.full((m, h // 128), 1 / 128, device="cuda")
            scores = torch.randn(m, experts, device="cuda", generator=generator)
            idx = scores.topk(topk, dim=-1).indices
            routing = torch.randn(m, topk, device="cuda", generator=generator) * 0.25
            # Exercise masked slots, signed weights and uneven expert loading.
            idx[::3, -1] = -1
            if args.routing == "hot":
                idx[:] = torch.arange(topk, device="cuda")
                idx[::3, -1] = -1
            elif args.routing == "empty":
                idx.fill_(-1)
            expected, counts = _reference(
                x.float() / 128, idx, routing, ref1, ref2, gs1, gs2, rank, epr
            )
            stats = torch.zeros(epr, dtype=torch.int32, device="cuda")
            buffer.stage(x, sf, idx, routing)
            y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
            family = "split" if fmt == "nvfp4" and m > 256 else "fused"
            kwargs = dict(
                cumulative_local_expert_recv_stats=stats,
                l1_global_scales=gs1,
                l2_global_scales=gs2,
            )
            run = nvfp4_mega_moe if fmt == "nvfp4" else mxfp4_mega_moe
            if fmt == "nvfp4":
                kwargs["kernel_family"] = family
                kwargs["force_push_dispatch"] = getattr(
                    args, "force_push_dispatch", None
                )
                kwargs["force_no_clean_barrier"] = getattr(
                    args, "force_no_clean_barrier", None
                )
            if getattr(args, "check_no_swap", False):
                from flashinfer.megamoe import select_mxfp4_fused_plan

                plan = select_mxfp4_fused_plan(
                    buffer.num_sms, ep, experts, topk, h, ih, args.capacity, m
                )
                assert not plan.swap_ab, "regression must exercise the SS L2 path"
            direct_repeats = getattr(args, "repeats", 2)
            for repetition in range(direct_repeats):
                run(y, w1, w2, buffer, **kwargs)
                torch.cuda.synchronize()
                assert torch.isfinite(y).all()
                error = (y.float() - expected.float()).abs()
                relative_l2 = torch.linalg.vector_norm(
                    error
                ) / torch.linalg.vector_norm(expected.float()).clamp_min(1e-12)
                assert relative_l2 < 0.015, (
                    fmt,
                    family,
                    m,
                    relative_l2.item(),
                    error.max().item(),
                )
                assert torch.equal(stats.long(), counts * (repetition + 1)), (
                    stats,
                    counts,
                )
                if repetition == 0:
                    first = y.clone()
                else:
                    assert torch.equal(y, first), (
                        "workspace reuse changed the output",
                        fmt,
                        family,
                        m,
                        (y.float() - first.float()).abs().max().item(),
                        int((y != first).sum().item()),
                    )
            if getattr(args, "cuda_graph", False):
                dist.barrier()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    run(y, w1, w2, buffer, **kwargs)
                for replay in range(direct_repeats):
                    graph.replay()
                    torch.cuda.synchronize()
                    assert torch.equal(y, first), "CUDA graph replay changed the output"
                    assert torch.equal(
                        stats.long(), counts * (direct_repeats + replay + 1)
                    )
            metrics = torch.tensor(
                [relative_l2.item(), error.max().item()], device="cuda"
            )
            dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
            if rank == 0:
                print(
                    f"PASS {fmt} {family} M={m} fm={args.fast_math} routing={args.routing} "
                    f"relative_l2={metrics[0].item():.6g} max_abs={metrics[1].item():.6g} stats/reuse=exact",
                    flush=True,
                )
    buffer.destroy()
    dist.destroy_process_group()


def _run(args):
    if getattr(args, "repeats", 2) < 1:
        raise ValueError("repeats must be positive")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29648")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=getattr(args, "ep", 8))


@pytest.mark.parametrize("ep", [4, 8])
def test_megamoe_fp4_sm90(ep):
    if torch.cuda.device_count() < ep or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip(f"requires {ep} Hopper GPUs")
    _run(
        argparse.Namespace(
            ep=ep,
            hidden=512,
            intermediate=256,
            capacity=384,
            tokens=[1, 24, 32, 96, 384],
            formats=["nvfp4", "mxfp4"],
            fast_math=True,
            routing="random",
            cuda_graph=True,
        )
    )


def test_mxfp4_no_swap_decoded_b_reuse():
    if torch.cuda.device_count() < 8 or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires 8 Hopper GPUs")
    _run(
        argparse.Namespace(
            hidden=4096,
            intermediate=2048,
            capacity=384,
            tokens=[384],
            formats=["mxfp4"],
            fast_math=True,
            routing="random",
            cuda_graph=True,
            repeats=6,
            check_no_swap=True,
        )
    )


@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk",
    [(4096, 2048, 256, 6), (6144, 2048, 384, 8), (7168, 3072, 384, 6)],
)
def test_nvfp4_ep4_production_shape(hidden, intermediate, experts, topk):
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires 4 Hopper GPUs")
    _run(
        argparse.Namespace(
            ep=4,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            capacity=384,
            tokens=[1, 96, 384],
            formats=["nvfp4"],
            fast_math=True,
            routing="random",
            cuda_graph=True,
        )
    )


@pytest.mark.parametrize("no_clean", [False, True])
def test_nvfp4_ep4_counter_dispatch(no_clean):
    if torch.cuda.device_count() < 4 or torch.cuda.get_device_capability()[0] != 9:
        pytest.skip("requires 4 Hopper GPUs")
    _run(
        argparse.Namespace(
            ep=4,
            hidden=1024,
            intermediate=256,
            capacity=384,
            tokens=[1, 24],
            formats=["nvfp4"],
            fast_math=True,
            routing="random",
            cuda_graph=True,
            repeats=3,
            force_push_dispatch=True,
            force_no_clean_barrier=no_clean,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=[4, 8], default=8)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 24, 32, 96, 384])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=384)
    parser.add_argument(
        "--formats", nargs="+", choices=["nvfp4", "mxfp4"], default=["nvfp4", "mxfp4"]
    )
    parser.add_argument(
        "--routing", choices=["random", "hot", "empty"], default="random"
    )
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--check-no-swap", action="store_true")
    parser.add_argument(
        "--no-fast-math", dest="fast_math", action="store_false", default=True
    )
    _run(parser.parse_args())
