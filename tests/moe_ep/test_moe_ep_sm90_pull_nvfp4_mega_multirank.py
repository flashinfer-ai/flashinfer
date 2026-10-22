"""Public SM90 NVFP4 x FP8 EP4 execution against a test-owned FP32 oracle.

Run in its own process, separate from the SM100 vendor imports::

    torchrun --standalone --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_nvfp4_mega_multirank.py -v -rs

The caller supplies canonical packed E2M1/K16-E4M3 weights and natural BF16
inputs. The reference decodes and promotes those raw weights mathematically;
it does not read preprocessed weights or call a kernel/reference helper.
"""

from __future__ import annotations

import os

import pytest


pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_hopper, pytest.mark.node_exclusive]

WORLD_SIZE = 4
LOCAL_EXPERTS = 4
HIDDEN = 256
INTERMEDIATE = 256
TOKENS = 16
FP8_MAX = 448.0
CLAMP = 10.0


def _gather(tensor):
    import torch
    import torch.distributed as dist

    rows = [torch.empty_like(tensor) for _ in range(WORLD_SIZE)]
    dist.all_gather(rows, tensor.contiguous())
    return torch.stack(rows)


def _raw_weights(rank):
    import torch

    from flashinfer.moe_ep import Nvfp4WeightPack

    generator = torch.Generator(device="cuda").manual_seed(1701 + rank)

    def leg(rows, width):
        codes = torch.randint(
            0,
            16,
            (LOCAL_EXPERTS, rows, width),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        codes[codes == 8] = 0  # Canonicalize negative zero.
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
        scales = torch.randint(
            48,
            64,
            (LOCAL_EXPERTS, rows, width // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        return packed, scales

    w13, sf13 = leg(2 * INTERMEDIATE, HIDDEN)
    w2, sf2 = leg(HIDDEN, INTERMEDIATE)
    expert = torch.arange(LOCAL_EXPERTS, dtype=torch.float32, device="cuda")
    return Nvfp4WeightPack(
        w13=w13,
        w2=w2,
        w13_scale=sf13,
        w2_scale=sf2,
        w13_global_scale=(expert + rank + 2) / 128,
        w2_global_scale=(expert + rank + 3) / 128,
    )


def _inputs(rank, launch, *, tokens=TOKENS):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(2909 + 97 * launch + rank)
    hidden = (
        0.5 * torch.randn(tokens, HIDDEN, device="cuda", generator=generator)
        + (rank + 1) / 32
    ).to(torch.bfloat16)
    token = torch.arange(tokens, device="cuda").view(-1, 1)
    slot = torch.arange(WORLD_SIZE, device="cuda").view(1, -1)
    owner = (rank + slot + launch) % WORLD_SIZE
    expert = (token + slot + launch) % LOCAL_EXPERTS
    ids = (owner * LOCAL_EXPERTS + expert).to(torch.int64)
    weights = torch.tensor([0.125, 0.25, 0.25, 0.375], device="cuda")
    return hidden, ids.contiguous(), weights.expand(tokens, -1).contiguous()


def _promoted_weight(payload, scales):
    """RN_E4M3(q*s/8), without using the implementation's conversion LUT."""
    import torch

    codes = torch.stack((payload & 15, payload >> 4), dim=-1).flatten(-2)
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=payload.device)
    value = lut[(codes & 7).long()]
    value = torch.where((codes & 8) != 0, -value, value)
    scale = scales.contiguous().view(torch.float8_e4m3fn).float()
    return (value * scale.repeat_interleave(16, -1) / 8).to(torch.float8_e4m3fn)


def _k64_order(width, device):
    """pi64(32*i + 16*h + 4*c + j) = 16*c + 8*i + 4*h + j."""
    import torch

    index = torch.arange(width, device=device)
    p = index % 64
    return index // 64 * 64 + (p // 4 % 4) * 16 + p // 32 * 8 + p // 16 % 2 * 4 + p % 4


def _reference(hidden, ids, route_weights, raw):
    """Global FP32 GEMMs with the public FP8/BF16 data-format contract."""
    import torch
    import torch.nn.functional as functional

    def flatten_experts(tensor):
        return tensor.flatten(0, 1)

    ranks, tokens, width = hidden.shape
    w13, w2, sf13, sf2, g13, g2 = map(flatten_experts, raw)
    w13 = _promoted_weight(w13, sf13).float()
    w2 = _promoted_weight(w2, sf2).float()
    hidden = hidden.flatten(0, 1).float()
    ids = ids.flatten(0, 1)
    route_weights = route_weights.flatten(0, 1)
    input_scale = (hidden.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(1e-30)
    x = (hidden / input_scale).to(torch.float8_e4m3fn).float()
    order = _k64_order(HIDDEN, hidden.device)
    group_order = _k64_order(64, hidden.device)
    terms = torch.zeros(
        hidden.shape[0], WORLD_SIZE, HIDDEN, dtype=torch.bfloat16, device=hidden.device
    )
    for expert in range(WORLD_SIZE * LOCAL_EXPERTS):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        assert token.numel() > 0, f"global expert {expert} was not exercised"
        # Both GEMM operands follow pi64. No NV0/NV64 bitwise equality is assumed.
        fc1 = x[token].index_select(1, order) @ w13[expert].index_select(1, order).T
        fc1 *= input_scale[token] * (8 * g13[expert])
        gate = fc1[:, :INTERMEDIATE].clamp(max=CLAMP)
        up = fc1[:, INTERMEDIATE:].clamp(-CLAMP, CLAMP)
        activated = functional.silu(gate) * up * route_weights[token, slot, None]
        grouped = activated.reshape(-1, INTERMEDIATE // 64, 64)
        scale = (grouped.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(1e-30)
        quantized = (grouped * torch.reciprocal(scale)).to(torch.float8_e4m3fn).float()
        fc2 = torch.zeros(token.numel(), HIDDEN, device=hidden.device)
        for block in range(INTERMEDIATE // 64):
            a = quantized[:, block].index_select(1, group_order)
            b = w2[expert, :, block * 64 : (block + 1) * 64].index_select(
                1, group_order
            )
            fc2 += (a @ b.T) * scale[:, block]
        # The public BF16-combine path stores a BF16 term for each routed slot.
        terms[token, slot] = (fc2 * (8 * g2[expert])).to(torch.bfloat16)
    return terms.float().sum(1).reshape(ranks, tokens, width)


def test_sm90_nvfp4_public_ep4_matches_fp32_reference():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires four visible CUDA GPUs")
    if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(WORLD_SIZE)):
        pytest.skip("requires four SM90 GPUs")
    if int(os.environ.get("WORLD_SIZE", "1")) != WORLD_SIZE:
        pytest.skip("requires torchrun --nproc_per_node=4")
    pytest.importorskip(
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
    )

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
    )

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    bootstrap = BootstrapConfig(world_size=WORLD_SIZE, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    config = Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, top_k=WORLD_SIZE, gate_up_clamp=CLAMP
    )
    assert config.kperm == 64
    weights = _raw_weights(rank)
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=WORLD_SIZE * LOCAL_EXPERTS,
            max_tokens_per_rank=TOKENS,
            token_hidden_size=HIDDEN,
        ),
        weights=weights,
        backend=MegaConfig(megakernel=config),
    )
    try:
        raw = tuple(
            _gather(getattr(weights, name))
            for name in (
                "w13",
                "w2",
                "w13_scale",
                "w2_scale",
                "w13_global_scale",
                "w2_global_scale",
            )
        )
        previous_ids = None
        for launch in range(2):
            hidden, ids, route_weights = _inputs(rank, launch)
            gathered = tuple(_gather(t) for t in (hidden, ids, route_weights))
            owners = gathered[1] // LOCAL_EXPERTS
            for source in range(WORLD_SIZE):
                assert torch.equal(
                    owners[source].unique().sort().values,
                    torch.arange(WORLD_SIZE, device="cuda"),
                )
                assert (owners[source] != source).any()
            if previous_ids is not None:
                assert not torch.equal(gathered[1], previous_ids)
            previous_ids = gathered[1]
            tensors = MoEEpTensors(
                hidden_states=hidden, topk_ids=ids, topk_weights=route_weights
            )
            actual = layer.forward(tensors).clone()
            repeated = None
            if launch == 1:
                repeated = layer.forward(tensors).clone()
            actual = _gather(actual)
            if repeated is not None:
                repeated = _gather(repeated)
                torch.testing.assert_close(
                    repeated.view(torch.uint8), actual.view(torch.uint8), atol=0, rtol=0
                )
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                expected = _reference(*gathered, raw)
            finally:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
            assert not torch.equal(expected[0], expected[1])
            torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.02)
            for source in range(WORLD_SIZE):
                norm = expected[source].norm()
                assert norm > 1e-3
                rel_l2 = (actual[source].float() - expected[source]).norm() / norm
                assert rel_l2 < 0.025, (source, launch, rel_l2.item())
    finally:
        layer.destroy()
