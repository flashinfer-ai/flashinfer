"""SM120 MXFP4 x MXFP8 nosync MegaMoE (moe_ep backend ``sm120_mxfp4_mxfp8_bf16_nosync``) through
``MoEEpLayer`` against a dense reference with the kernel's numerics.

``stage_inputs`` quantizes the BF16 activations to MXFP8 (E4M3, one power-of-two scale per 32
channels, 2**ceil(log2(amax / 448))); the kernel quantizes the clamped SwiGLU output the same way
and applies the routing weight to each expert's BF16 output in the top-k combine. The reference
follows those rules; it also reports the error of both against an unquantized-intermediate
reference.

Run with plain Python (workers are spawned here):
    python tests/moe_ep/test_moe_ep_sm120_w4a8_nosync_mega.py --ep 4 --tokens 1 8 20 64
"""

import argparse
import datetime
import os

import pytest
import torch
import torch.distributed as dist


def _weights(e, n, k, generator):
    """Random MXFP4 experts in the moe_ep layout (E2M1 packed low nibble first, UE8M0 per 32) and
    their dense dequantized values."""
    codes = torch.randint(
        0, 16, (e, n, k), device="cuda", generator=generator, dtype=torch.uint8
    )
    packed = codes[..., 0::2] | (codes[..., 1::2] << 4)
    sf = (
        torch.randint(0, 3, (e, n, k // 32), device="cuda", generator=generator) + 122
    ).to(torch.uint8)
    lattice = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device="cuda")
    values = lattice[(codes & 7).long()] * torch.where(codes < 8, 1.0, -1.0)
    return (packed, sf), values * torch.exp2(sf.float() - 127).repeat_interleave(
        32, dim=-1
    )


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


def _gather_padded(t, capacity):
    """[rows, ...] per rank -> [world * capacity, ...] (rank r's rows at r * capacity)."""
    padded = t.new_zeros((capacity, *t.shape[1:]))
    padded[: t.shape[0]] = t
    gathered = [torch.empty_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)
    return torch.cat(gathered)


def _reference(x, idx, weights, w1, w2, rank, local_experts, capacity, quantize=True):
    all_x = _gather_padded(x, capacity)
    all_idx = _gather_padded(idx, capacity)
    rows_per_rank = _gather_padded(
        torch.tensor([x.shape[0]], device="cuda"), 1
    ).tolist()
    for r, rows in enumerate(rows_per_rank):  # padding rows route nowhere
        all_idx[r * capacity + rows : (r + 1) * capacity] = -1
    all_weights = _gather_padded(weights, capacity)
    all_x = _mxfp8(all_x) if quantize else all_x
    routes = torch.zeros(
        (*all_idx.shape, w2.shape[1]), device="cuda", dtype=torch.float32
    )
    for expert in range(local_experts):
        rows, slots = torch.where(all_idx == rank * local_experts + expert)
        if rows.numel() == 0:
            continue
        intermediate = all_x[rows] @ w1[expert].T
        gate, up = intermediate.chunk(2, dim=-1)
        hidden = torch.nn.functional.silu(gate.clamp(max=10)) * up.clamp(-10, 10)
        hidden = _mxfp8(hidden) if quantize else hidden
        down = (hidden @ w2[expert].T).to(torch.bfloat16).float()
        routes[rows, slots] = down * all_weights[rows, slots, None]
    dist.all_reduce(routes)
    result = routes.sum(1).to(torch.bfloat16)
    return result[rank * capacity : rank * capacity + x.shape[0]]


def _rel(a, b):
    return (
        torch.linalg.vector_norm(a.float() - b.float())
        / torch.linalg.vector_norm(b.float()).clamp_min(1e-12)
    ).item()


def _layer(args, rank, experts, raw1, raw2, backend="nosync"):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        PrequantizedMoEWeights,
    )

    if backend == "nosync":
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig as Config,
        )

        config = Config(
            intermediate_size=args.intermediate,
            top_k=args.topk,
            transport=args.transport,
            tokens_per_item=args.tokens_per_item,
        )
    else:
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig as Config,
        )

        config = Config(
            intermediate_size=args.intermediate,
            top_k=args.topk,
            swiglu_limit=10,
            transport_table="all_lsa",
        )
    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=args.ep,
            rank=rank,
            device=torch.cuda.current_device(),
            process_group=dist.group.WORLD,
        ),
        fleet_params=FleetParams(
            num_experts=experts,
            max_tokens_per_rank=args.capacity,
            token_hidden_size=args.hidden,
        ),
        weights=PrequantizedMoEWeights(
            w13=raw1[0].view(torch.float4_e2m1fn_x2),
            w2=raw2[0].view(torch.float4_e2m1fn_x2),
            w13_scale=raw1[1].view(torch.float8_e8m0fnu),
            w2_scale=raw2[1].view(torch.float8_e8m0fnu),
        ),
        backend=MegaConfig(
            megakernel=config, quantize_input=True, preprocess_weights=True
        ),
    )


def _routing(args, m, experts, topk, generator):
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
        # out-of-range ids (one above 32 bits) and ids repeated within a token, each case only
        # where top-k has the slots it writes
        if topk > 1:
            idx[0::4, 1] = experts + 5
            idx[2::4, 1] = idx[2::4, 0]
        if topk > 2:
            idx[1::4, 2] = 1 << 40
        if topk > 4:
            idx[3::4, 4] = idx[3::4, 2]
    return idx, routing


def _worker(rank, args):
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake.cake_staging import (
        quantize_activations_k32,
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
    raw1, ref1 = _weights(epr, ih * 2, h, generator)
    raw2, ref2 = _weights(epr, h, ih, generator)
    layer = _layer(args, rank, experts, raw1, raw2)
    cake = _layer(args, rank, experts, raw1, raw2, "cake") if args.vs_cake else None
    for m_all in args.tokens:
        # uneven: ranks stage different row counts in the same call, one rank none
        m = (m_all, 0, max(m_all // 2, 1), 1)[rank % 4] if args.uneven else m_all
        x = torch.randn(m, h, device="cuda", generator=generator, dtype=torch.bfloat16)
        idx, routing = _routing(args, m, experts, topk, generator)
        ref_idx = _effective_routes(idx, experts)
        expected = _reference(
            x.float(), ref_idx, routing, ref1, ref2, rank, epr, args.capacity
        )
        exact = _reference(
            x.float(), ref_idx, routing, ref1, ref2, rank, epr, args.capacity, False
        )
        t = MoEEpTensors(hidden_states=x, topk_ids=idx, topk_weights=routing)
        for repetition in range(args.repeats):
            y = layer.forward(t)
            torch.cuda.synchronize()
            assert layer._kernel.protocol_error(layer._workspace) == 0
            assert y.shape == (m, h) and torch.isfinite(y).all()
            if args.routing == "empty":
                assert not y.any(), (
                    "a token with every route masked must come back zero"
                )
            elif m:
                relative_l2 = _rel(y, expected)
                assert relative_l2 < 5e-3, (m, relative_l2)
            if repetition == 0:
                first = y.clone()
            else:
                assert torch.equal(y, first), "workspace reuse changed the output"
        # pre-quantized activations (quantize_input=False contract): the same bits
        codes, exponents = quantize_activations_k32(x)
        layer._kernel.stage_inputs(
            MoEEpTensors(
                hidden_states=codes.view(torch.float8_e4m3fn),
                topk_ids=idx,
                topk_weights=routing,
                scales=exponents.view(torch.float8_e8m0fnu),
            ),
            layer._workspace,
            quantize_input=False,
        )
        y = layer._kernel.compute(
            layer._workspace, layer._get_transformed_weights(), output=None
        )
        torch.cuda.synchronize()
        assert torch.equal(y, first), "pre-quantized staging changed the output"
        # CUDA graph: staging (quantization included) and the kernel
        out = layer.output_buffer
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.stage_inputs(t, compile_tokens_per_rank=args.capacity)
            layer.compute_staged(output=out)
        for _ in range(args.repeats):
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:m], first), "CUDA graph replay changed the output"
        del graph
        rel = 0.0 if args.routing == "empty" or not m else _rel(first, expected)
        rel_exact = 0.0 if args.routing == "empty" or not m else _rel(first, exact)
        rel_cake = 0.0
        if cake is not None:
            y_cake = cake.forward(t)
            torch.cuda.synchronize()
            if m and args.routing != "empty":
                # cake applies the routing weight before quantizing the SwiGLU output, this backend
                # after FC2 (in the combine): the two differ by that rounding (4-5% here), and each
                # stays as close to the unquantized reference as the MXFP8 recipe allows
                rel_cake = _rel(first, y_cake)
                assert rel_cake < 6e-2, (m, rel_cake)
                assert _rel(y_cake, exact) < 8e-2 and _rel(first, exact) < 8e-2
        metrics = torch.tensor([rel, rel_exact, rel_cake], device="cuda")
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"PASS sm120 nosync EP{ep} M={m_all}{' uneven' if args.uneven else ''} "
                f"routing={args.routing} transport={args.transport} "
                f"relative_l2={metrics[0].item():.3g} "
                f"vs_unquantized_intermediate={metrics[1].item():.3g}"
                + (f" vs_cake={metrics[2].item():.3g}" if cake is not None else "")
                + " reuse/prequantized/graph=exact",
                flush=True,
            )
    layer.destroy()
    if cake is not None:
        cake.destroy()
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
        transport="p2p",
        tokens_per_item=None,
        uneven=False,
        vs_cake=False,
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _sm120_gpus(n):
    return torch.cuda.device_count() >= n and all(
        torch.cuda.get_device_capability(i) == (12, 0) for i in range(n)
    )


@pytest.mark.parametrize("ep", [2, 4, 8])
def test_sm120_nosync(ep):
    if not _sm120_gpus(ep):
        pytest.skip(f"requires {ep} SM120 GPUs")
    _run(_args(ep=ep))


@pytest.mark.parametrize("routing", ["hot", "empty", "invalid"])
def test_sm120_nosync_skewed_routing(routing):
    """Every route on the same experts, every route masked, and out-of-range or repeated ids."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(routing=routing, repeats=3))


def test_sm120_nosync_uneven_rows():
    """Ranks stage different row counts in the same call, one of them none."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(uneven=True, tokens=[1, 20, 128]))


@pytest.mark.parametrize("hidden", [256, 768])
def test_sm120_nosync_hidden_widths(hidden):
    """Hidden sizes whose MXFP8 scale row (hidden / 32 bytes) is not a multiple of 16 bytes."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(hidden=hidden, tokens=[1, 20, 64]))


@pytest.mark.parametrize("ep", [4, 8])
@pytest.mark.parametrize(
    "hidden,intermediate,experts,topk",
    [(4096, 2048, 256, 6), (5120, 2304, 384, 6), (4096, 2048, 288, 8)],
    ids=["dsv4_flash", "dsv41_flash", "glm53_flash"],
)
def test_sm120_nosync_model_shapes(ep, hidden, intermediate, experts, topk):
    """DeepSeek-V4-Flash, DeepSeek-V4.1-Flash and GLM-5.3-Flash routed experts at EP4 and EP8."""
    if not _sm120_gpus(ep):
        pytest.skip(f"requires {ep} SM120 GPUs")
    _run(
        _args(
            ep=ep,
            hidden=hidden,
            intermediate=intermediate,
            experts=experts,
            topk=topk,
            tokens=[1, 16, 64, 128],
        )
    )


@pytest.mark.parametrize("tokens_per_item", [8, 16, 24, 32])
def test_sm120_nosync_tokens_per_item(tokens_per_item):
    """Every work-item width, forced, at token counts below and above it."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(tokens=[1, 20, 64, 128], tokens_per_item=tokens_per_item))


def test_sm120_nosync_matches_cake():
    """Same weights and inputs through the sibling cake backend: the two agree up to their
    intermediate-quantization recipes."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(
        _args(
            hidden=4096,
            intermediate=2048,
            tokens=[1, 16, 128],
            repeats=1,
            vs_cake=True,
        )
    )


# The NIC transport needs NCCL 2.30.7 with GIN on the host's network; opt in where it exists.
_NIC = os.environ.get("FLASHINFER_TEST_MEGAMOE_SM120_NIC") == "1"


@pytest.mark.parametrize("ep", [2, 4, 8])
@pytest.mark.parametrize("routing", ["random", "hot", "empty", "invalid"])
def test_sm120_nosync_nic_transport(ep, routing):
    """Every pair over NCCL GIN (transport="nic"): same results as P2P."""
    if not _NIC:
        pytest.skip("set FLASHINFER_TEST_MEGAMOE_SM120_NIC=1 on a GIN-capable host")
    if not _sm120_gpus(ep):
        pytest.skip(f"requires {ep} SM120 GPUs")
    _run(_args(ep=ep, routing=routing, repeats=3, transport="nic"))


def test_sm120_nosync_auto_transport():
    """transport="auto": P2P where NCCL maps the pair, GIN elsewhere, with uneven rows."""
    if not _NIC:
        pytest.skip("set FLASHINFER_TEST_MEGAMOE_SM120_NIC=1 on a GIN-capable host")
    if not _sm120_gpus(8):
        pytest.skip("requires 8 SM120 GPUs")
    _run(_args(ep=8, hidden=4096, intermediate=2048, transport="auto", uneven=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=[1, 2, 4, 8], default=4)
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
    parser.add_argument("--transport", choices=["p2p", "nic", "auto"], default="p2p")
    parser.add_argument(
        "--tokens-per-item", type=int, choices=[8, 16, 24, 32], default=None
    )
    parser.add_argument("--uneven", action="store_true")
    parser.add_argument("--vs-cake", action="store_true")
    _run(parser.parse_args())
