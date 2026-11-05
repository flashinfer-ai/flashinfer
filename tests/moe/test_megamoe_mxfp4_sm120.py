"""SM120 MXFP4 MegaMoE (PCIe peer-to-peer) against a dense reference with the SM120 kernel's numerics.

The SM120 kernel requantises the staged activations to MXFP8 (E4M3, one power-of-two scale per 32 channels,
2**ceil(log2(amax / 448))), quantises the clamped SwiGLU output the same way, and applies the routing weight to
each expert's BF16 output in the top-k combine. The reference follows those rules; it also reports the error of
both against an unquantised-intermediate reference.

Run with plain Python (workers are spawned here):
    python tests/moe/test_megamoe_mxfp4_sm120.py --ep 4 --tokens 1 8 20 64
"""

import argparse
import datetime
import os

import pytest
import torch
import torch.distributed as dist


def _weights(e, n, k, generator):
    """Random MXFP4 experts in the transform's input format (Marlin byte order, E8M0 per 32) and their dense
    dequantised values."""
    codes = torch.randint(
        0, 16, (e, n, k), device="cuda", generator=generator, dtype=torch.uint8
    )
    chunks = codes.reshape(e, n, k // 8, 8)
    packed = (chunks[..., 4:] | (chunks[..., :4] << 4)).reshape(e, n, k // 2)
    sf = (
        torch.randint(0, 3, (e, n, k // 32), device="cuda", generator=generator) + 122
    ).to(torch.uint8)
    lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    values = lattice[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    return (packed, sf), values * torch.exp2(sf.float() - 127).repeat_interleave(
        32, dim=-1
    )


def _gather(t):
    gathered = [torch.empty_like(t) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, t)
    return torch.cat(gathered)


def _mxfp8(t):
    groups = t.reshape(*t.shape[:-1], -1, 32)
    amax = groups.abs().amax(-1, keepdim=True)
    e = torch.where(
        amax > 0, torch.ceil(torch.log2(amax / 448)), torch.full_like(amax, -127)
    ).clamp(-127, 127)
    return (
        (groups * torch.exp2(-e)).to(torch.float8_e4m3fn).float() * torch.exp2(e)
    ).reshape_as(t)


def _effective_routes(idx, experts):
    """The kernel's view of top-k ids: out-of-range ids and repeats of an earlier slot of the same
    token are masked (-1), like an explicit -1."""
    idx = idx.clone()
    idx[(idx < 0) | (idx >= experts)] = -1
    for k in range(1, idx.shape[1]):
        repeat = (idx[:, :k] == idx[:, k : k + 1]).any(1) & (idx[:, k] >= 0)
        idx[repeat, k] = -1
    return idx


def _reference(x, idx, weights, w1, w2, gs1, gs2, rank, local_experts, quantize=True):
    all_x, all_idx, all_weights = _gather(x), _gather(idx), _gather(weights)
    all_x = _mxfp8(all_x) if quantize else all_x
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
        hidden = _mxfp8(hidden) if quantize else hidden
        down = (hidden @ w2[expert].T * gs2[expert]).to(torch.bfloat16).float()
        routes[rows, slots] = down * all_weights[rows, slots, None]
    dist.all_reduce(routes)
    result = routes.sum(1).to(torch.bfloat16)
    m = x.shape[0]
    counts = torch.stack(
        [(all_idx == rank * local_experts + e).sum() for e in range(local_experts)]
    )
    return result[rank * m : (rank + 1) * m], counts


def _rel(a, b):
    return (
        torch.linalg.vector_norm(a.float() - b.float())
        / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)
    ).item()


def _worker(rank, args):
    from flashinfer.megamoe import (
        MegaMoESymmBuffer,
        mxfp4_mega_moe,
        transform_mxfp4_weights_for_mega_moe,
    )

    ep = args.ep
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=ep,
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    h, ih, experts, topk = args.hidden, args.intermediate, args.experts, args.topk
    epr = experts // ep
    generator = torch.Generator(device="cuda").manual_seed(7919 + rank)
    buffer = MegaMoESymmBuffer(
        dist.group.WORLD, experts, args.capacity, topk, h, ih, fast_math=True
    )
    assert buffer.prewarm() == 1
    gs1 = torch.linspace(0.73, 1.37, epr, device="cuda")
    gs2 = torch.linspace(1.31, 0.67, epr, device="cuda")
    raw1, ref1 = _weights(epr, ih * 2, h, generator)
    raw2, ref2 = _weights(epr, h, ih, generator)
    w1, w2 = transform_mxfp4_weights_for_mega_moe(raw1, raw2)
    for m in args.tokens:
        x = (torch.randn(m, h, device="cuda", generator=generator) * 32).to(
            torch.float8_e4m3fn
        )
        sf = torch.full((m, h // 128), 1 / 128, device="cuda")
        idx = (
            torch.randn(m, experts, device="cuda", generator=generator)
            .topk(topk, dim=-1)
            .indices
        )
        routing = torch.randn(m, topk, device="cuda", generator=generator) * 0.25
        idx[::3, -1] = -1  # masked slots, signed weights, uneven expert loading
        if args.routing == "hot":
            idx[:] = torch.arange(topk, device="cuda")
            idx[::3, -1] = -1
        elif args.routing == "empty":
            idx.fill_(-1)
        elif args.routing == "invalid":
            # out-of-range ids (one above 32 bits) and ids repeated within a token
            idx[0::4, 1] = experts + 5
            idx[1::4, 2] = 1 << 40
            idx[2::4, 3] = idx[2::4, 0]
            idx[3::4, 4] = idx[3::4, 2]
        ref_idx = _effective_routes(idx, experts)
        x_in = x.float() / 128
        expected, counts = _reference(
            x_in, ref_idx, routing, ref1, ref2, gs1, gs2, rank, epr
        )
        exact, _ = _reference(
            x_in, ref_idx, routing, ref1, ref2, gs1, gs2, rank, epr, quantize=False
        )
        stats = torch.zeros(epr, dtype=torch.int32, device="cuda")
        buffer.stage(x, sf, idx, routing)
        y = torch.empty(m, h, device="cuda", dtype=torch.bfloat16)
        kwargs = dict(
            cumulative_local_expert_recv_stats=stats,
            l1_global_scales=gs1,
            l2_global_scales=gs2,
        )
        for repetition in range(args.repeats):
            mxfp4_mega_moe(y, w1, w2, buffer, **kwargs)
            torch.cuda.synchronize()
            assert buffer.error() == 0
            assert torch.isfinite(y).all()
            if args.routing == "empty":
                assert not y.any(), (
                    "a token with every route masked must come back zero"
                )
            else:
                relative_l2 = _rel(y, expected)
                assert relative_l2 < 5e-3, (m, relative_l2)
            assert torch.equal(stats.long(), counts * (repetition + 1)), (stats, counts)
            if repetition == 0:
                first = y.clone()
            else:
                assert torch.equal(y, first), "workspace reuse changed the output"
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mxfp4_mega_moe(y, w1, w2, buffer, **kwargs)
        for replay in range(args.repeats):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(y, first), "CUDA graph replay changed the output"
            assert torch.equal(stats.long(), counts * (args.repeats + replay + 1))
        rel = 0.0 if args.routing == "empty" else _rel(y, expected)
        rel_exact = 0.0 if args.routing == "empty" else _rel(y, exact)
        metrics = torch.tensor([rel, rel_exact], device="cuda")
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"PASS sm120 mxfp4 EP{ep} M={m} routing={args.routing} relative_l2={metrics[0].item():.3g} "
                f"vs_unquantised_intermediate={metrics[1].item():.3g} stats/reuse/graph=exact",
                flush=True,
            )
    buffer.destroy()
    dist.destroy_process_group()


def _run(args):
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")
    torch.multiprocessing.spawn(_worker, args=(args,), nprocs=args.ep)


def _args(**overrides):
    args = dict(
        ep=4,
        hidden=512,
        intermediate=256,
        experts=256,
        topk=6,
        capacity=128,
        tokens=[1, 8, 20, 64, 128],
        routing="random",
        repeats=2,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _sm120_gpus(n):
    return torch.cuda.device_count() >= n and all(
        torch.cuda.get_device_capability(i)[0] == 12 for i in range(n)
    )


@pytest.mark.parametrize("ep", [2, 4, 8])
def test_megamoe_mxfp4_sm120(ep):
    if not _sm120_gpus(ep):
        pytest.skip(f"requires {ep} SM120 GPUs")
    _run(_args(ep=ep))


@pytest.mark.parametrize("routing", ["hot", "empty", "invalid"])
def test_megamoe_mxfp4_sm120_skewed_routing(routing):
    """Every route on the same experts, every route masked, and out-of-range or repeated ids."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(routing=routing, repeats=3))


@pytest.mark.parametrize("hidden", [256, 768])
def test_megamoe_mxfp4_sm120_hidden_widths(hidden):
    """Hidden sizes whose MXFP8 scale row (hidden / 32 bytes) is not a multiple of 16 bytes."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(hidden=hidden, tokens=[1, 20, 64]))


def test_megamoe_mxfp4_sm120_production_shape():
    """DeepSeek-V4-Flash routed experts (H4096, I2048, 256 experts, top-6), EP4, up to the 256 tokens/rank cap."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(hidden=4096, intermediate=2048, capacity=256, tokens=[1, 16, 64, 256]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 20, 64, 128])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument(
        "--routing", choices=["random", "hot", "empty", "invalid"], default="random"
    )
    parser.add_argument("--repeats", type=int, default=2)
    _run(parser.parse_args())
