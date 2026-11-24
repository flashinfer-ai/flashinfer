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
import dataclasses
import datetime
import os
import warnings

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

        transport = getattr(args, "cake_transport", None) or os.environ.get(
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


def _layer(args, rank, experts, raw1, raw2, nosync=True, quantize_input=True):
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
            quantize_input=quantize_input,
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
    if args.default_knobs:
        # the default profile is still configurable after construction (the cake table that
        # the nosync kernel follows is resolved at warmup)
        layer.create_workspace(args.capacity, knobs={})
    # the configured backend alone, on the same weights: large calls must match it bit for bit,
    # small calls agree with it up to the intermediate-quantization recipe
    alone = _layer(args, rank, experts, raw1, raw2, nosync=False)
    # pre-quantized activations (quantize_input=False) through the public wrapper (cake takes
    # them; cutedsl quantizes its own input)
    prequantized = (
        _layer(args, rank, experts, raw1, raw2, quantize_input=False)
        if args.fallback == "cake"
        else None
    )
    if layer.nosync is None:  # built on the cake table at warmup
        layer.warmup()
    assert layer.nosync is not None, "the nosync kernel must be on for these shapes"
    if args.epoch_start is not None:  # the next calls cross the epoch's 32-bit wrap
        layer.nosync._workspace.set_epoch(args.epoch_start)
        dist.barrier()
    # the nosync kernel reads the configured backend's packed FP4 weights, not a copy
    transformed = layer.base._transformed
    base_w13 = transformed.w1_fp4 if args.fallback == "cake" else transformed[0][0]
    assert layer.nosync.get_transformed_weights().w13.data_ptr() == base_w13.data_ptr()
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
        if prequantized is not None and (nosync or with_alone):
            # pre-quantized activations (quantize_input=False): the same bits, the same kernel
            codes, exponents = quantize_activations_k32(x)
            y = prequantized.forward(
                MoEEpTensors(
                    hidden_states=codes.view(torch.float8_e4m3fn),
                    topk_ids=idx,
                    topk_weights=routing,
                    scales=exponents.view(torch.float8_e8m0fnu),
                )
            )
            torch.cuda.synchronize()
            assert prequantized.last_call_nosync == nosync
            assert torch.equal(y, first), "pre-quantized activations changed the output"
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
    if prequantized is not None:
        prequantized.destroy()
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
        default_knobs=False,
        epoch_start=None,
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


def _check_borrowed_input_contract(pack, transformed, hidden, intermediate):
    from flashinfer.moe_ep import preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.weights import (
        validate_transformed_mega_weights,
    )

    kwargs = dict(
        hidden_size=hidden,
        intermediate_size=intermediate,
        interleave_w13=False,
        scale_layout="canonical",
    )
    validate_transformed_mega_weights(
        transformed,
        hidden_size=hidden,
        intermediate_size=intermediate,
        local_experts=pack.w13.shape[0],
    )

    def offset(tensor, num_bytes):
        raw = tensor.view(torch.uint8)
        padded = torch.empty(
            raw.numel() + num_bytes, dtype=raw.dtype, device=raw.device
        )
        view = padded[num_bytes:].view(tensor.dtype).view(tensor.shape)
        view.copy_(tensor)
        return view

    aligned = offset(pack.w13, 16)
    borrowed = preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
        dataclasses.replace(pack, w13=aligned), **kwargs
    )
    assert borrowed.w13.data_ptr() == aligned.data_ptr()
    with pytest.raises(ValueError, match="16-byte aligned"):
        preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
            dataclasses.replace(pack, w13=offset(pack.w13, 1)), **kwargs
        )
    with pytest.raises(ValueError, match="4-byte aligned"):
        preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
            dataclasses.replace(pack, w13_scale=offset(pack.w13_scale, 1)), **kwargs
        )
    for name, tensor in (("w13", pack.w13), ("w13_scale", pack.w13_scale)):
        raw = tensor.view(torch.uint8)
        expanded = raw.new_empty(raw.numel() * 2)
        noncontiguous = expanded.as_strided(
            tensor.shape,
            (tensor.stride(0), tensor.stride(1) * 2, tensor.stride(2)),
        ).view(tensor.dtype)
        noncontiguous.copy_(tensor)
        with pytest.raises(ValueError, match="contiguous"):
            preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
                dataclasses.replace(pack, **{name: noncontiguous}), **kwargs
            )
    with pytest.raises(ValueError, match="pre-interleaved fp4"):
        preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
            pack, fp4=(pack.w13, pack.w2), **kwargs
        )


def _weight_layout_worker(rank, duplicate_destinations=False, dedup_compact=False):
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        PrequantizedMoEWeights,
        Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig,
        preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.weights import (
        validate_transformed_mega_weights,
    )

    ep = 4
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")
    dist.init_process_group(
        "nccl", rank=rank, world_size=ep, device_id=torch.device("cuda", rank)
    )
    hidden, intermediate, experts, topk = 512, 256, 256, 6
    generator = torch.Generator(device="cuda").manual_seed(7919 + rank)
    raw1, ref1 = _weights(experts // ep, 2 * intermediate, hidden, generator)
    raw2, ref2 = _weights(experts // ep, hidden, intermediate, generator)
    args = _args(ep=ep)
    capacity = 16 if dedup_compact else 128
    output = torch.empty(capacity, hidden, dtype=torch.bfloat16, device="cuda")
    for interleave, scale_layout in (
        (True, "kernel"),
        (False, "deepgemm"),
        (False, "canonical"),
        (True, "deepgemm"),
        (True, "canonical"),
    ):
        raw_scales = [raw1[1], raw2[1]]
        scales = [
            (
                sf.view(torch.int32).transpose(1, 2).contiguous().transpose(1, 2)
                if scale_layout == "deepgemm"
                else sf.view(torch.float8_e8m0fnu)
            )
            for sf in raw_scales
        ]
        pack = PrequantizedMoEWeights(
            w13=raw1[0].view(torch.float4_e2m1fn_x2),
            w2=raw2[0].view(torch.float4_e2m1fn_x2),
            w13_scale=scales[0],
            w2_scale=scales[1],
        )
        transformed = preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
            pack,
            hidden_size=hidden,
            intermediate_size=intermediate,
            interleave_w13=interleave,
            scale_layout=scale_layout,
        )
        if not interleave:
            assert transformed.w13.data_ptr() == raw1[0].data_ptr()
        for name, source in zip(("w13_scale", "w2_scale"), scales, strict=True):
            pointer = getattr(transformed, name).data_ptr()
            if scale_layout == "kernel":
                assert pointer != source.data_ptr()
            else:
                assert pointer == source.data_ptr()
        if rank == 0:
            validate_transformed_mega_weights(
                transformed,
                hidden_size=hidden,
                intermediate_size=intermediate,
                local_experts=experts // ep,
            )
            if scale_layout == "deepgemm":
                uint8_scale = transformed.w13_scale
                checked = preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
                    dataclasses.replace(pack, w13_scale=uint8_scale),
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    interleave_w13=interleave,
                    scale_layout=scale_layout,
                )
                assert checked.w13_scale.data_ptr() == uint8_scale.data_ptr()
                invalid_scale = uint8_scale.as_strided(
                    uint8_scale.shape,
                    (
                        uint8_scale.stride(0),
                        1,
                        uint8_scale.stride(2),
                    ),
                )
                with pytest.raises(ValueError, match="DeepGEMM stride"):
                    preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
                        dataclasses.replace(pack, w13_scale=invalid_scale),
                        hidden_size=hidden,
                        intermediate_size=intermediate,
                        interleave_w13=interleave,
                        scale_layout=scale_layout,
                    )
                with pytest.raises(ValueError, match="stride"):
                    validate_transformed_mega_weights(
                        dataclasses.replace(
                            transformed,
                            w13_scale=transformed.w13_scale.as_strided(
                                transformed.w13_scale.shape,
                                (
                                    transformed.w13_scale.stride(0),
                                    1,
                                    transformed.w13_scale.stride(2),
                                ),
                            ),
                        ),
                        hidden_size=hidden,
                        intermediate_size=intermediate,
                        local_experts=experts // ep,
                    )
            if scale_layout == "canonical" and not interleave:
                _check_borrowed_input_contract(pack, transformed, hidden, intermediate)
            with pytest.raises(ValueError, match="gate/up-contiguous FP4"):
                preprocess_sm120_mxfp4_mxfp8_nosync_mega_weights(
                    pack,
                    hidden_size=hidden,
                    intermediate_size=intermediate,
                    interleave_w13=False,
                )
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                world_size=ep, rank=rank, device=rank, process_group=dist.group.WORLD
            ),
            fleet_params=FleetParams(
                num_experts=experts,
                max_tokens_per_rank=capacity,
                token_hidden_size=hidden,
            ),
            weights=pack,
            backend=MegaConfig(
                megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(
                    intermediate_size=intermediate, top_k=topk, transport="p2p"
                ),
                quantize_input=True,
                preprocess_weights=False,
                transformed_weights=transformed,
            ),
        )
        layer.warmup()
        dedup_layer = None
        if dedup_compact:
            dedup_layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    world_size=ep,
                    rank=rank,
                    device=rank,
                    process_group=dist.group.WORLD,
                ),
                fleet_params=FleetParams(
                    num_experts=experts,
                    max_tokens_per_rank=capacity,
                    token_hidden_size=hidden,
                ),
                weights=pack,
                backend=MegaConfig(
                    megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(
                        intermediate_size=intermediate,
                        top_k=topk,
                        transport="p2p",
                        dedup_replicated_inputs=True,
                    ),
                    quantize_input=True,
                    preprocess_weights=False,
                    transformed_weights=transformed,
                ),
            )
            dedup_layer.warmup()
        for bound in (1, 16) if dedup_compact else (1, 12, 128):
            rows = (
                bound if dedup_compact else (bound, 0, max(bound // 2, 1), 1)[rank % 4]
            )
            input_generator = (
                torch.Generator(device="cuda").manual_seed(7919 + bound)
                if dedup_compact
                else generator
            )
            x = torch.randn(
                rows,
                hidden,
                device="cuda",
                generator=input_generator,
                dtype=torch.bfloat16,
            )
            ids, weights = _routing(args, rows, experts, topk, input_generator)
            if dedup_compact:
                ids.copy_(torch.arange(topk, device=ids.device))
            if duplicate_destinations:
                ids.copy_(
                    rank * (experts // ep) + torch.arange(topk, device=ids.device)
                )
            t = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
            if dedup_compact:
                owned = (
                    torch.arange(rows, device=ids.device)
                    .remainder(ep)
                    .eq(rank)
                    .unsqueeze(1)
                )
                compact_weights = weights.masked_fill(~owned, 0)
                compact_t = MoEEpTensors(
                    hidden_states=x,
                    topk_ids=ids.masked_fill(~owned, -1),
                    topk_weights=compact_weights,
                )
                partial = dedup_layer.forward(compact_t).clone()
                expected = layer.forward(t).clone()
                leader = owned.squeeze(1)
                assert not partial[~leader].any()
                assert torch.equal(partial[leader], expected[leader])
                actual = partial.clone()
                dist.all_reduce(actual)
                assert torch.equal(actual, expected)

                compact_output = torch.empty_like(output)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    dedup_layer.stage_inputs(compact_t, compile_tokens_per_rank=bound)
                    dedup_layer.compute_staged(output=compact_output)
                x.copy_(
                    torch.randn(
                        rows,
                        hidden,
                        device="cuda",
                        generator=input_generator,
                        dtype=torch.bfloat16,
                    )
                )
                weights.copy_(
                    torch.rand(rows, topk, device="cuda", generator=input_generator)
                )
                compact_weights.copy_(weights.masked_fill(~owned, 0))
                expected = layer.forward(t).clone()
                compact_output.zero_()
                graph.replay()
                torch.cuda.synchronize()
                actual = compact_output[:rows].clone()
                dist.all_reduce(actual)
                assert torch.equal(actual, expected)
                assert layer._kernel.protocol_error(layer._workspace) == 0
                assert dedup_layer._kernel.protocol_error(dedup_layer._workspace) == 0
                del graph
                if rank == 0:
                    print(
                        f"PASS dedup compact layout={scale_layout} M={bound}: "
                        "original and reduced partial outputs, graph replay exact",
                        flush=True,
                    )
                continue
            layer.stage_inputs(t, compile_tokens_per_rank=bound)
            layer.compute_staged(output=output)
            torch.cuda.synchronize()
            if duplicate_destinations and scale_layout == "kernel" and bound == 128:
                from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.workspace import (
                    pool_row_bytes,
                )

                workspace = layer._workspace
                route_map = workspace._view(19, (ep, experts // ep, 128), torch.int32)
                source_ids = [None] * ep
                dist.all_gather_object(source_ids, ids.cpu())
                for source, routes in enumerate(source_ids):
                    expected_routes = {}
                    for token, expert_ids in enumerate(routes.tolist()):
                        for k, expert in enumerate(expert_ids):
                            if expert // (experts // ep) == rank:
                                expected_routes.setdefault(
                                    expert % (experts // ep), []
                                ).append(token * 8 + k)
                    for expert, expected_meta in expected_routes.items():
                        actual_meta = (
                            route_map[source, expert, : len(expected_meta)]
                            .cpu()
                            .tolist()
                        )
                        assert actual_meta == expected_meta, (
                            rank,
                            source,
                            expert,
                            actual_meta[:16],
                            expected_meta[:16],
                        )
                pool = workspace._view(
                    5, (ep, 128, pool_row_bytes(hidden)), torch.uint8
                )
                assert torch.equal(pool[rank, :rows, :hidden], workspace.x[:rows])
                assert torch.equal(
                    pool[rank, :rows, hidden : hidden + hidden // 32],
                    workspace.x_sf[:rows],
                )
            expected = _reference(
                x.float(),
                _effective_routes(ids, experts),
                weights,
                ref1,
                ref2,
                rank,
                experts // ep,
                128,
            )
            assert _rel(output[:rows], expected) < 5e-3, (
                interleave,
                scale_layout,
                bound,
                rank,
            )
            assert layer._kernel.protocol_error(layer._workspace) == 0
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                layer.stage_inputs(t, compile_tokens_per_rank=bound)
                layer.compute_staged(output=output)
            x.copy_(
                torch.randn(
                    rows,
                    hidden,
                    device="cuda",
                    generator=generator,
                    dtype=torch.bfloat16,
                )
            )
            if not duplicate_destinations:
                ids.copy_(
                    torch.randint(
                        0, experts, ids.shape, device="cuda", generator=generator
                    )
                )
            weights.copy_(torch.rand(rows, topk, device="cuda", generator=generator))
            expected = _reference(
                x.float(),
                _effective_routes(ids, experts),
                weights,
                ref1,
                ref2,
                rank,
                experts // ep,
                128,
            )
            output.zero_()
            graph.replay()
            torch.cuda.synchronize()
            assert _rel(output[:rows], expected) < 5e-3
            assert layer._kernel.protocol_error(layer._workspace) == 0
            del graph
        if rank == 0:
            print(
                f"PASS layout FP4 interleaved={interleave} scales={scale_layout}: "
                "storage contract, uneven DP and changed-input graph replay",
                flush=True,
            )
        layer.destroy()
        if dedup_layer is not None:
            dedup_layer.destroy()
    dist.destroy_process_group()


def test_sm120_nosync_p2p_workspace_layout():
    """P2P shares token rows and keeps route metadata in a bounded local map."""
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_nosync.workspace import (
        buffer_layout,
        pool_row_bytes,
    )

    ranks, experts, capacity, topk = 8, 384, 128, 6
    hidden, intermediate = 5120, 2304
    offsets = buffer_layout(ranks, experts, capacity, topk, hidden, intermediate)
    assert offsets[6] - offsets[5] == ranks * capacity * pool_row_bytes(hidden)
    route_map_bytes = experts * capacity * 4
    assert offsets[20] - offsets[19] == route_map_bytes
    gin_offsets = buffer_layout(
        ranks, experts, capacity, topk, hidden, intermediate, gin=True
    )
    assert gin_offsets[20] - gin_offsets[19] == (
        route_map_bytes + ranks * capacity * topk * 4
    )
    compact = buffer_layout(
        ranks,
        experts,
        capacity,
        topk,
        hidden,
        intermediate,
        dedup_replicated_inputs=True,
    )
    rows = experts // ranks * capacity
    assert compact[13] - compact[12] == rows * intermediate
    assert compact[14] - compact[13] == rows * (intermediate // 128) * 4
    assert compact[1:12] == offsets[1:12]
    assert compact[0] == 28_825_344


def test_sm120_nosync_dedup_layout_requires_p2p():
    from flashinfer.moe_ep import Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig

    with pytest.raises(ValueError, match="requires P2P"):
        Sm120_Mxfp4_Mxfp8_Bf16_Nosync_MegaMoeConfig(
            intermediate_size=256,
            top_k=2,
            transport="auto",
            dedup_replicated_inputs=True,
        )


def test_sm120_nosync_p2p_shared_token_rows():
    """Routes from one token to one rank share a row without losing route metadata."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    torch.multiprocessing.spawn(_weight_layout_worker, args=(True,), nprocs=4)


def test_sm120_nosync_dedup_compact_workspace():
    """Unique TP leaders match the original layout exactly, including a full expert pool."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    torch.multiprocessing.spawn(
        _weight_layout_worker, args=(False, True), nprocs=4, join=True
    )


def test_sm120_nosync_weight_layouts():
    """Default and borrowed weight layouts preserve numerics across DP and graph replay."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    torch.multiprocessing.spawn(_weight_layout_worker, nprocs=4, join=True)


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


@pytest.mark.parametrize("cake_transport", ["all_gin", "lsa_where_reachable"])
def test_sm120_nosync_follows_cake_table(cake_transport):
    """transport=None: the nosync kernel takes the per-pair table the cake backend resolved
    (GIN for every pair, or P2P where NCCL maps the pair and GIN elsewhere)."""
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
            transport=None,
            cake_transport=cake_transport,
            uneven=True,
        )
    )


def test_sm120_nosync_follows_cake_table_after_create_workspace():
    """transport=None on a P2P table (cake "all_lsa"): the default workspace can still be
    configured (create_workspace with knobs) after the layer is built; the nosync kernel takes
    the table at warmup."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(
        _args(
            tokens=[1, 20, 128],
            transport=None,
            cake_transport="all_lsa",
            default_knobs=True,
        )
    )


@pytest.mark.parametrize("epoch_start", [2**31 - 3, 2**32 - 3], ids=["int32", "uint32"])
def test_sm120_nosync_epoch_wrap(epoch_start):
    """The calls cross the epoch's signed (2^31) and unsigned (2^32) wrap: no stale flag passes a
    wait, every call matches the reference."""
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    _run(_args(tokens=[1, 20, 128], repeats=3, epoch_start=epoch_start))


def _bootstrap_one(rank, ep):
    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=ep,
        timeout=datetime.timedelta(minutes=10),
        device_id=torch.device("cuda", rank),
    )
    from flashinfer.moe_ep import BootstrapConfig

    return BootstrapConfig(
        world_size=ep,
        rank=rank,
        device=torch.cuda.current_device(),
        process_group=dist.group.WORLD,
    )


def _off_reason(build):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        layer = build()
    reasons = [
        str(w.message) for w in caught if "nosync kernel is off" in str(w.message)
    ]
    assert layer.nosync is None and len(reasons) == 1, reasons
    return layer, reasons[0]


def _shared_expert_worker(rank, ep):
    """cake with its in-kernel FP8 shared expert: the routed experts are zero, so the output is
    the shared expert alone; small calls must keep it (the nosync kernel stays off)."""
    from flashinfer.moe_ep import (
        CakePrequantizedMoEWeightsWithSharedExpert,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        NosyncOptions,
        Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig,
    )

    bootstrap = _bootstrap_one(rank, ep)
    h, ih, experts, topk = 512, 256, 64, 6
    epr = experts // ep
    generator = torch.Generator(device="cuda").manual_seed(31 + rank)

    def fp8(n, k):
        return (torch.randn(n, k, device="cuda", generator=generator) * 0.5).to(
            torch.float8_e4m3fn
        )

    def build(nosync):
        return MoEEpLayer(
            bootstrap=bootstrap,
            fleet_params=FleetParams(
                num_experts=experts, max_tokens_per_rank=128, token_hidden_size=h
            ),
            weights=CakePrequantizedMoEWeightsWithSharedExpert(
                w13=torch.zeros(
                    epr, 2 * ih, h // 2, dtype=torch.uint8, device="cuda"
                ).view(torch.float4_e2m1fn_x2),
                w2=torch.zeros(epr, h, ih // 2, dtype=torch.uint8, device="cuda").view(
                    torch.float4_e2m1fn_x2
                ),
                w13_scale=torch.full(
                    (epr, 2 * ih, h // 32), 127, dtype=torch.uint8, device="cuda"
                ).view(torch.float8_e8m0fnu),
                w2_scale=torch.full(
                    (epr, h, ih // 32), 127, dtype=torch.uint8, device="cuda"
                ).view(torch.float8_e8m0fnu),
                shared_w13=shared[0],
                shared_w2=shared[1],
                shared_w13_scale=shared[2],
                shared_w2_scale=shared[3],
            ),
            backend=MegaConfig(
                megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cake_MegaMoeConfig(
                    intermediate_size=ih,
                    top_k=topk,
                    shared_expert="fp8_in_kernel",
                    transport_table="all_lsa",
                )
            ),
            nosync=NosyncOptions() if nosync else None,
        )

    shared = (
        fp8(2 * ih, h),
        fp8(h, ih),
        torch.full((2 * ih, h // 32), 124, dtype=torch.uint8, device="cuda").view(
            torch.float8_e8m0fnu
        ),
        torch.full((h, ih // 32), 124, dtype=torch.uint8, device="cuda").view(
            torch.float8_e8m0fnu
        ),
    )
    layer, reason = _off_reason(lambda: build(True))
    assert "shared expert" in reason, reason
    alone = build(False)
    for m in (1, 20, 128):
        x = torch.randn(m, h, device="cuda", generator=generator, dtype=torch.bfloat16)
        idx = torch.randint(0, experts, (m, topk), device="cuda", generator=generator)
        routing = torch.rand(m, topk, device="cuda", generator=generator)
        t = MoEEpTensors(hidden_states=x, topk_ids=idx, topk_weights=routing)
        y = layer.forward(t)
        y_alone = alone.forward(t)
        torch.cuda.synchronize()
        assert not layer.last_call_nosync
        assert y.abs().amax() > 0, "the shared expert's contribution is missing"
        assert torch.equal(y, y_alone)
    if rank == 0:
        print("PASS sm120 nosync off with cake's shared expert", flush=True)
    layer.destroy()
    alone.destroy()
    dist.destroy_process_group()


def test_sm120_nosync_off_with_cake_shared_expert():
    if not _sm120_gpus(4):
        pytest.skip("requires 4 SM120 GPUs")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")
    torch.multiprocessing.spawn(_shared_expert_worker, args=(4,), nprocs=4)


def _geometry_worker(rank, ep):
    """A geometry the nosync kernel does not cover (intermediate 64) on a backend that does: the
    option turns off before transforming the weights, the configured layer stays usable."""
    from flashinfer.moe_ep import (
        FleetParams,
        MegaConfig,
        MoEEpNosyncLayer,
        NosyncOptions,
        PrequantizedMoEWeights,
        Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig,
    )

    bootstrap = _bootstrap_one(rank, ep)
    h, ih, experts = 256, 64, 8
    generator = torch.Generator(device="cuda").manual_seed(rank)
    raw1, _ = _weights(experts // ep, 2 * ih, h, generator)
    raw2, _ = _weights(experts // ep, h, ih, generator)
    base = torch.nn.Identity()  # the configured layer is not exercised here
    _, reason = _off_reason(
        lambda: MoEEpNosyncLayer(
            base,
            bootstrap,
            FleetParams(
                num_experts=experts, max_tokens_per_rank=128, token_hidden_size=h
            ),
            PrequantizedMoEWeights(
                w13=raw1[0].view(torch.float4_e2m1fn_x2),
                w2=raw2[0].view(torch.float4_e2m1fn_x2),
                w13_scale=raw1[1].view(torch.float8_e8m0fnu),
                w2_scale=raw2[1].view(torch.float8_e8m0fnu),
            ),
            MegaConfig(
                megakernel=Sm120_Mxfp4_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
                    intermediate_size=ih, top_k=2
                )
            ),
            NosyncOptions(),
        )
    )
    assert "intermediate_size" in reason, reason
    if rank == 0:
        print("PASS sm120 nosync off for an unsupported geometry", flush=True)
    dist.destroy_process_group()


def test_sm120_nosync_off_for_unsupported_geometry():
    if not _sm120_gpus(2):
        pytest.skip("requires 2 SM120 GPUs")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29649")
    torch.multiprocessing.spawn(_geometry_worker, args=(2,), nprocs=2)


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
    parser.add_argument("--default-knobs", action="store_true")
    parser.add_argument("--epoch-start", type=int, default=None)
    _run(parser.parse_args())
