"""Latency benchmark of the Cake SM120 NVFP4 x NVFP4 (W4A4) MegaMoE backend (``MoEEpLayer``).

The harness, columns and options of ``bench_moe_ep_sm120_w4a8_cake_mega.py`` (same shapes,
variants, return codecs and tables) with ``sm120_nvfp4_nvfp4_bf16_cake`` layers: random NVFP4
routed weights (E2M1 codes, E4M3 block-16 scales), unit global scales, an optional FP8 shared
expert, and BF16 inputs quantized by the backend (``quantize_input=True``).  Shapes outside the
NVFP4 region (``hidden_size % 1024 != 0``, or ``intermediate_size % 256 != 0`` with the shared
expert) are skipped.  Correctness is covered by ``tests/moe_ep/test_moe_ep_sm120_nvfp4_cake_mega.py``.

    torchrun --standalone --nproc_per_node=4 \\
        benchmarks/moe_ep/backends/mega/kernel/sm120/bench_moe_ep_sm120_nvfp4_cake_mega.py \\
        --return-codecs bf16 int8
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bench_moe_ep_sm120_w4a8_cake_mega as w4a8_bench  # noqa: E402


def _nvfp4_weights(rank, world, geometry, shared, seed):
    """Random NVFP4 routed weights (E2M1 codes, E4M3 block-16 scales 2^-8..2^-5) + optional FP8 shared."""
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        CakePrequantizedMoEWeightsWithSharedExpert,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm120.mxfp4_mxfp8_bf16_cake import (
        cake_synthetic as syn,
    )

    H, I = geometry.hidden, geometry.intermediate
    local = geometry.experts // world

    def plane(rows, k, g):
        codes = torch.randint(
            0, 16, (rows, k), dtype=torch.uint8, device="cuda", generator=g
        )
        packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
        sc = torch.randint(
            0x02, 0x10, (rows, k // 16), dtype=torch.uint8, device="cuda", generator=g
        )
        return packed, sc

    parts = []
    for e in range(rank * local, (rank + 1) * local):
        g = torch.Generator(device="cuda").manual_seed(seed * 4099 + e)
        parts.append((*plane(2 * I, H, g), *plane(H, I, g)))
    w13, w13s, w2, w2s = (torch.stack([p[i] for p in parts]) for i in range(4))
    routed = (w13, w2, w13s.view(torch.float8_e4m3fn), w2s.view(torch.float8_e4m3fn))
    if not shared:
        return MoEWeightPack(*routed)
    g = torch.Generator(device="cuda").manual_seed(seed + 7919)
    s13, s13s = syn._e4m3(2 * I, H, g)
    s2, s2s = syn._e4m3(H, I, g)
    return CakePrequantizedMoEWeightsWithSharedExpert(
        *routed,
        shared_w13=s13.view(torch.float8_e4m3fn),
        shared_w2=s2.view(torch.float8_e4m3fn),
        shared_w13_scale=s13s.view(torch.float8_e8m0fnu),
        shared_w2_scale=s2s.view(torch.float8_e8m0fnu),
    )


def _nvfp4_layer(
    rank,
    world,
    geometry,
    capacity,
    shared,
    table,
    limit,
    activation_fields,
    codec,
    seed,
):
    """The NVFP4 layer of one (shape, variant, codec); ``None`` outside the NVFP4 region."""
    if geometry.hidden % 1024 or (shared and geometry.intermediate % 256):
        return None
    weights = _nvfp4_weights(rank, world, geometry, shared, seed)
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    )

    return MoEEpLayer(
        bootstrap=BootstrapConfig(
            world_size=world, rank=rank, device=torch.cuda.current_device()
        ),
        fleet_params=FleetParams(
            num_experts=geometry.experts,
            max_tokens_per_rank=capacity,
            token_hidden_size=geometry.hidden,
        ),
        weights=weights,
        backend=MegaConfig(
            megakernel=Sm120_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
                intermediate_size=geometry.intermediate,
                top_k=geometry.top_k,
                gate_up_clamp=limit or None,
                **(activation_fields or {}),
                combine_dtype=codec,
                shared_expert="fp8_in_kernel" if shared else None,
                transport_table=table,
            ),
            quantize_input=True,
            preprocess_weights=True,
        ),
    )


if __name__ == "__main__":
    w4a8_bench.main(
        build=_nvfp4_layer, kernel="nvfp4", description=__doc__.splitlines()[0]
    )
