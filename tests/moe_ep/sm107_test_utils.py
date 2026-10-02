"""MoE references using canonical weight layouts and the shim's quantizers.

The reference evaluates activation and scaling math without the production
weight transform. Quantization and unpacking helpers are shared with the shim.
"""

import torch

from flashinfer.moe_ep import PrequantizedMoEWeights
from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
    quantize_mxfp4_block32,
    quantize_mxfp8_block32,
    quantize_nvfp4_block16,
    scale_to_f32,
    unpack_fp4_to_f32,
)


def quantize(tensor, kind, norm=1.0):
    if tensor.numel() == 0:
        nvfp4 = kind == "nvfp4"
        dtype = (
            torch.uint8
            if nvfp4
            else torch.float8_e4m3fn
            if kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
            else torch.float8_e5m2
        )
        return (
            torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] // (2 if nvfp4 else 1)),
                device=tensor.device,
                dtype=dtype,
            ),
            torch.empty(
                (*tensor.shape[:-1], tensor.shape[-1] // (16 if nvfp4 else 32)),
                device=tensor.device,
                dtype=torch.float8_e4m3fn if nvfp4 else torch.float8_e8m0fnu,
            ),
        )
    if kind == "nvfp4":
        return quantize_nvfp4_block16(tensor.float(), norm)
    dtype = (
        torch.float8_e4m3fn
        if kind in ("mxfp8_e4m3", "mxfp4_mxfp8")
        else torch.float8_e5m2
    )
    return quantize_mxfp8_block32(tensor.float(), dtype)


def weight_pack(w13, w2, kind, *, scaled=False):
    """Return canonical packed weights and matching global-scale corrections."""
    experts = w13.shape[0]
    if scaled:
        n1 = torch.linspace(1.1, 2.3, experts, device=w13.device)
        n2 = torch.linspace(0.7, 1.9, experts, device=w13.device)
        intermediate_norm = torch.linspace(1.3, 2.7, experts, device=w13.device)
        input_norm = 1.7
        scalars = dict(
            fc1_alpha=(input_norm * n1).reciprocal(),
            fc2_alpha=(intermediate_norm * n2).reciprocal(),
            fc1_norm_const=intermediate_norm,
        )
    else:
        n1 = n2 = torch.ones(experts, device=w13.device)
        input_norm, scalars = 1.0, {}
    if kind == "mxfp4_mxfp8":
        q1, sf1 = quantize_mxfp4_block32(w13)
        q2, sf2 = quantize_mxfp4_block32(w2)
    else:
        q1, sf1 = quantize(w13, kind, n1[:, None, None])
        q2, sf2 = quantize(w2, kind, n2[:, None, None])
    return PrequantizedMoEWeights(q1, q2, sf1, sf2), input_norm, scalars


def canonical_reference(
    xq,
    xsf,
    ids,
    scores,
    pack,
    kind,
    *,
    situ=False,
    early=True,
    scalars=None,
    situ_beta=1.25,
    situ_linear_beta=0.75,
):
    """FP32 math on canonical weights, independent of the kernel's layout transform."""
    scalars = {} if scalars is None else scalars
    if xq.shape[0] == 0:
        return torch.empty(
            (0, pack.w2.shape[1]), device=xq.device, dtype=torch.bfloat16
        )
    vec = 16 if kind == "nvfp4" else 32

    def dequant(data, scale, *, weight=False):
        fp4 = kind == "nvfp4" or (weight and kind == "mxfp4_mxfp8")
        values = unpack_fp4_to_f32(data) if fp4 else data.float()
        return values * scale_to_f32(scale).repeat_interleave(vec, dim=-1)

    x = dequant(xq, xsf)
    result = torch.zeros(x.shape, device=x.device, dtype=torch.float32)
    for e in range(pack.w13.shape[0]):
        row, slot = (ids == e).nonzero(as_tuple=True)
        if not row.numel():
            continue
        w13 = dequant(pack.w13[e], pack.w13_scale[e], weight=True)
        w2 = dequant(pack.w2[e], pack.w2_scale[e], weight=True)
        fc1 = x[row] @ w13.T
        if "fc1_alpha" in scalars:
            fc1 *= scalars["fc1_alpha"][e]
        gate, up = fc1.chunk(2, dim=-1)
        if situ:
            act = (
                situ_beta
                * torch.tanh(gate / situ_beta)
                * torch.sigmoid(gate)
                * situ_linear_beta
                * torch.tanh(up / situ_linear_beta)
            )
        else:
            act = up * gate * torch.sigmoid(gate)
        if early:
            act *= scores[row, slot, None]
        norm = scalars["fc1_norm_const"][e] if "fc1_norm_const" in scalars else 1.0
        aq, sf = quantize(act, kind, norm)
        term = dequant(aq, sf) @ w2.T
        if "fc2_alpha" in scalars:
            term *= scalars["fc2_alpha"][e]
        term = term.bfloat16().float()
        if not early:
            term *= scores[row, slot, None]
        result.index_add_(0, row, term)
    return result.bfloat16()


def assert_reference(actual, expected, kind):
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-6)
    assert torch.isfinite(actual).all()
    assert float(error) < (0.06 if kind == "nvfp4" else 0.02), float(error)


def run_mxfp4_k3_geometry(rank, world):
    """Synthetic checkpoint-format weights at K3's routed-expert geometry."""
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
        bootstrap_moe_ep_runtime,
        ensure_moe_ep_cuda_device,
        finalize_moe_ep_runtime,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    hidden, intermediate, experts, topk, tokens = 3584, 3072, 896, 16, 4
    if world <= 0 or experts % world:
        raise ValueError(
            f"K3 geometry requires a positive world size that divides {experts}, "
            f"got {world}"
        )
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    cfg = Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=intermediate,
        top_k=topk,
        activation="situ",
        situ_beta=4.0,
        situ_linear_beta=25.0,
        apply_topk_in_fc1=False,
    )
    runtime = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(cfg).runtime_requirements(bootstrap)
    )
    layers = []
    try:
        generator = torch.Generator(device="cuda").manual_seed(9031)

        def payload(rows, cols):
            return torch.randint(
                256,
                (experts, rows, cols // 2),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            )

        def scales(rows, cols):
            return torch.randint(
                119,
                121,
                (experts, rows, cols // 32),
                dtype=torch.uint8,
                device="cuda",
                generator=generator,
            ).view(torch.float8_e8m0fnu)

        pack = PrequantizedMoEWeights(
            payload(2 * intermediate, hidden),
            payload(hidden, intermediate),
            scales(2 * intermediate, hidden),
            scales(hidden, intermediate),
        )
        count = experts // world
        local = slice(rank * count, (rank + 1) * count)
        local_pack = PrequantizedMoEWeights(
            *(
                getattr(pack, name)[local]
                for name in ("w13", "w2", "w13_scale", "w2_scale")
            )
        )
        generator.manual_seed(73 + rank)
        x = torch.randn(tokens, hidden, device="cuda", generator=generator).bfloat16()
        # Each token reaches every rank, including the final local expert.
        positions = torch.arange(tokens * topk, device="cuda").reshape(tokens, topk)
        ids = ((positions % world) * count + count - 1 - positions // world).int()
        scores = torch.softmax(
            torch.randn(tokens, topk, device="cuda", generator=generator), -1
        )
        xq, xsf = quantize(x, "mxfp4_mxfp8")
        expected = canonical_reference(
            xq,
            xsf,
            ids,
            scores,
            pack,
            "mxfp4_mxfp8",
            situ=True,
            early=False,
            situ_beta=4.0,
            situ_linear_beta=25.0,
        )
        for prestaged in (False, True):
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    rank=rank, world_size=world, auto_bootstrap=False
                ),
                fleet_params=FleetParams(
                    num_experts=experts,
                    max_tokens_per_rank=tokens,
                    token_hidden_size=hidden,
                ),
                weights=local_pack,
                backend=MegaConfig(megakernel=cfg, quantize_input=not prestaged),
            )
            layers.append(layer)
            inputs = MoEEpTensors(
                xq if prestaged else x, ids, scores, scales=xsf if prestaged else None
            )
            assert_reference(layer.forward(inputs), expected, "mxfp4_mxfp8")
        assert layers[0]._workspace is layers[1]._workspace
    finally:
        for layer in layers:
            layer.destroy()
        finalize_moe_ep_runtime(runtime)
