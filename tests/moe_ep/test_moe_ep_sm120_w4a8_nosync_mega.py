"""SM120 MXFP4 x MXFP8 nosync MegaMoE (moe_ep backend ``sm120_mxfp4_mxfp8_bf16_nosync``) as the
small-batch option of ``MoEEpLayer``: the configured backend (cake, or cutedsl) is built as
usual, ``nosync=NosyncOptions(...)`` adds the nosync kernel.

Calls of up to ``NosyncOptions.max_tokens_per_rank`` tokens per rank run the nosync kernel and are checked
against a dense reference with its numerics: ``stage_inputs`` quantizes the BF16 activations to
MXFP8 (E4M3, one power-of-two scale per 32 channels, 2**ceil(log2(amax / 448))); the kernel
quantizes the clamped SwiGLU output the same way and applies the routing weight to each expert's
BF16 output in the top-k combine. Larger calls run the configured backend and must match a layer
built without the option, bit for bit.

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


def _fallback_config(args):
    if args.fallback == "cake":
        from flashinfer.moe_ep import (
            Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig as Config,
        )

        transport = os.environ.get(
            "FLASHINFER_TEST_SM120_CAKE_TRANSPORT",
            "all_gin" if args.ep > 4 else "all_lsa",
        )
        return Config(
            intermediate_size=args.intermediate,
            top_k=args.topk,
            swiglu_limit=10,
            transport_table=transport,
        )
    from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig as Config

    return Config(
        intermediate_size=args.intermediate,
        top_k=args.topk,
        gate_up_clamp=10.0,
        # the default dual-N8 tail reclaim hangs at small buckets on this GPU class
        knobs={"k2_tail_reclaim": False},
    )


def _layer(args, rank, experts, raw1, raw2, nosync=True):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        NosyncOptions,
        PrequantizedMoEWeights,
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
        # configured as without the option
        backend=MegaConfig(
            megakernel=_fallback_config(args),
            quantize_input=True,
            preprocess_weights=True,
        ),
        nosync=NosyncOptions(
            max_tokens_per_rank=args.threshold,
            transport=args.transport,
            tokens_per_item=args.tokens_per_item,
        )
        if nosync
        else None,
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
    # the configured backend alone, on the same weights: large calls must match it bit for bit,
    # small calls agree with it up to the intermediate-quantization recipe
    alone = _layer(args, rank, experts, raw1, raw2, nosync=False)
    assert layer.nosync is not None, "the nosync kernel must be on for these shapes"
    # the nosync kernel reads the configured backend's packed FP4 weights, not a copy
    transformed = layer.base._transformed
    base_w13 = transformed.w1_fp4 if args.fallback == "cake" else transformed[0][0]
    assert layer.nosync._get_transformed_weights().w13.data_ptr() == base_w13.data_ptr()
    for m_all in args.tokens:
        # uneven: ranks stage different row counts in the same call, one rank none
        m = (m_all, 0, max(m_all // 2, 1), 1)[rank % 4] if args.uneven else m_all
        nosync = m_all <= args.threshold  # m_all is the largest row count of any rank
        x = torch.randn(m, h, device="cuda", generator=generator, dtype=torch.bfloat16)
        idx, routing = _routing(args, m, experts, topk, generator)
        ref_idx = _effective_routes(idx, experts)
        t = MoEEpTensors(hidden_states=x, topk_ids=idx, topk_weights=routing)
        # cake alone hangs on hot routing from 96 tokens per rank (EP4); compare where it runs
        with_alone = not (args.routing == "hot" and m_all > 64)
        y_alone = alone.forward(t) if with_alone else None
        torch.cuda.synchronize()
        if nosync:
            expected = _reference(
                x.float(), ref_idx, routing, ref1, ref2, rank, epr, args.capacity
            )
            exact = _reference(
                x.float(), ref_idx, routing, ref1, ref2, rank, epr, args.capacity, False
            )
        for repetition in range(args.repeats):
            y = layer.forward(t)
            torch.cuda.synchronize()
            assert layer.last_call_nosync == nosync, (m_all, nosync)
            if nosync:
                assert layer.nosync._kernel.protocol_error(layer.nosync._workspace) == 0
            assert y.shape == (m, h) and torch.isfinite(y).all()
            if not nosync:
                assert torch.equal(y, y_alone), (
                    "large calls must match the configured backend"
                )
            elif args.routing == "empty":
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
        if nosync:
            # pre-quantized activations (quantize_input=False): the same bits
            codes, exponents = quantize_activations_k32(x)
            kernel, ws = layer.nosync._kernel, layer.nosync._workspace
            kernel.set_compile_tokens_per_rank(ws, m_all)
            kernel.stage_inputs(
                MoEEpTensors(
                    hidden_states=codes.view(torch.float8_e4m3fn),
                    topk_ids=idx,
                    topk_weights=routing,
                    scales=exponents.view(torch.float8_e8m0fnu),
                ),
                ws,
                quantize_input=False,
            )
            y = kernel.compute(ws, layer.nosync._get_transformed_weights(), output=None)
            torch.cuda.synchronize()
            assert torch.equal(y, first), "pre-quantized staging changed the output"
        # CUDA graph at this call's token bound: staging (quantization included) and the kernel
        out = layer.output_buffer
        # eager at the same bound first (cutedsl compiles per bucket before capture)
        layer.stage_inputs(t, compile_tokens_per_rank=m_all)
        layer.compute_staged(output=out)
        torch.cuda.synchronize()
        assert torch.equal(out[:m], first), "the bounded call changed the output"
        dist.barrier()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            layer.stage_inputs(t, compile_tokens_per_rank=m_all)
            layer.compute_staged(output=out)
        assert layer.last_call_nosync == nosync
        for _ in range(args.repeats):
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(out[:m], first), "CUDA graph replay changed the output"
        del graph
        live = m and args.routing != "empty"
        rel = _rel(first, expected) if nosync and live else 0.0
        rel_exact = _rel(first, exact) if nosync and live else 0.0
        rel_alone = _rel(first, y_alone) if live and with_alone else 0.0
        if nosync and live and with_alone and args.routing != "invalid":
            # (invalid ids: each backend masks them its own way; the reference check above holds)
            # cake applies the routing weight before quantizing the SwiGLU output, this kernel
            # after FC2: the two differ by that rounding (4-5% at a few tokens and more, up to ~7%
            # at one token)
            assert rel_alone < 1e-1, (m, rel_alone)
            assert _rel(y_alone, exact) < 1e-1 and rel_exact < 1e-1
        metrics = torch.tensor([rel, rel_exact, rel_alone], device="cuda")
        dist.all_reduce(metrics, op=dist.ReduceOp.MAX)
        if rank == 0:
            print(
                f"PASS sm120 nosync EP{ep} M={m_all}{' uneven' if args.uneven else ''} "
                f"routing={args.routing} kernel={'nosync' if nosync else args.fallback} "
                f"transport={args.transport} relative_l2={metrics[0].item():.3g} "
                f"vs_unquantized_intermediate={metrics[1].item():.3g} "
                f"vs_{args.fallback}={metrics[2].item():.3g} "
                "reuse/prequantized/graph=exact",
                flush=True,
            )
    layer.destroy()
    alone.destroy()
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
        threshold=128,
        tokens=[1, 8, 20, 64, 128],
        routing="random",
        repeats=2,
        transport="p2p",
        tokens_per_item=None,
        uneven=False,
        fallback="cake",
    )
    args.update(overrides)
    return argparse.Namespace(**args)


def _sm120_gpus(n):
    return torch.cuda.device_count() >= n and all(
        torch.cuda.get_device_capability(i) == (12, 0) for i in range(n)
    )


# cutedsl's FlashInfer shim needs NVSHMEM and does not capture on every SM120 SKU (RTX 6000D);
# opt in where it runs.
_CUTEDSL = os.environ.get("FLASHINFER_TEST_SM120_CUTEDSL") == "1"


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


@pytest.mark.parametrize("uneven", [False, True], ids=["even", "uneven"])
def test_sm120_nosync_switch(uneven):
    """Capacity 512: calls up to 128 tokens per rank run the nosync kernel, larger ones the
    configured backend (bit-identical to a layer without the option), eager (all-reduced token count) and under CUDA
    graphs (compile_tokens_per_rank); with uneven rows the largest rank decides."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(
        _args(
            hidden=4096,
            intermediate=2048,
            capacity=512,
            tokens=[1, 128, 129, 512],
            uneven=uneven,
        )
    )


def test_sm120_nosync_threshold():
    """A lower NosyncOptions.max_tokens_per_rank moves the switch point."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(threshold=16, tokens=[16, 17, 64], capacity=128))


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


@pytest.mark.parametrize("hidden", [256, 768])
def test_sm120_nosync_cutedsl_fallback(hidden):
    """cutedsl as the fallback, at hidden sizes whose MXFP8 scale row (hidden / 32 bytes) is not a
    multiple of 16 bytes (cake needs hidden % 512 == 0)."""
    if not _CUTEDSL:
        pytest.skip(
            "set FLASHINFER_TEST_SM120_CUTEDSL=1 where the cutedsl backend runs"
        )
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(
        _args(hidden=hidden, capacity=256, tokens=[1, 20, 128, 256], fallback="cutedsl")
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
    """transport="auto": P2P where NCCL maps the pair, GIN elsewhere, with uneven rows, across
    the switch point."""
    if not _NIC:
        pytest.skip("set FLASHINFER_TEST_MEGAMOE_SM120_NIC=1 on a GIN-capable host")
    if not _sm120_gpus(8):
        pytest.skip("requires 8 SM120 GPUs")
    _run(
        _args(
            ep=8,
            hidden=4096,
            intermediate=2048,
            capacity=256,
            tokens=[1, 20, 128, 256],
            transport="auto",
            uneven=True,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep", type=int, choices=[2, 4, 8], default=4)
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 8, 20, 64, 128])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=256)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--capacity", type=int, default=128)
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--fallback", choices=["cake", "cutedsl"], default="cake")
    parser.add_argument(
        "--routing", choices=["random", "hot", "empty", "invalid"], default="random"
    )
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--transport", choices=["p2p", "nic", "auto"], default="p2p")
    parser.add_argument(
        "--tokens-per-item", type=int, choices=[8, 16, 24, 32], default=None
    )
    parser.add_argument("--uneven", action="store_true")
    _run(parser.parse_args())
