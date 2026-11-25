"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# ``sm107_nvfp4_nvfp4_bf16_cake`` at EP4: MoEEpLayer against the Torch
# reference of the sm107 block-scaled tests, on DeepSeek-V4-Pro's slice (the
# geometry the generated programs are built for). Runs under its own torchrun::
#
#     torchrun --nproc_per_node=4 -m pytest \
#         tests/moe_ep/test_moe_ep_sm107_nvfp4_cake_mega.py -v -m "gpu_4 and arch_rubin"
#
# Every rank builds the same global expert bank (fixed seed) and checks its
# own tokens' output against the reference over the full bank, for each token
# tier (1024 without the dedup combine, 4096 and 8192 with it), with BF16 and
# prequantized NVFP4 activations, dropped routes whose scores are NaN or inf,
# an empty rank, a rank no route reaches, unaligned and strided activation and
# output views, and repeated forwards over one workspace.

from __future__ import annotations

import os

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    FleetParams,
    MegaConfig,
    MoEEpLayer,
    MoEEpTensors,
    PrequantizedMoEWeights,
    Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig,
    bootstrap_moe_ep_runtime,
    ensure_moe_ep_cuda_device,
    finalize_moe_ep_runtime,
)
from flashinfer.moe_ep.backends.mega.kernel.sm107.nvfp4_nvfp4_bf16_cake import (
    cake_programs,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

HIDDEN, INTERMEDIATE, NUM_EXPERTS, TOP_K, WORLD = 7168, 3072, 48, 6, 4


def _ranks() -> tuple[int, int]:
    return int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))


def _require() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.get_device_capability() != (10, 7):
        pytest.skip("the Cake NVFP4 MegaMoE programs run on sm_107a")
    if _ranks()[1] != WORLD:
        pytest.skip(f"requires torchrun with {WORLD} ranks")
    if not cake_programs.programs_available():
        pytest.skip(f"{cake_programs.LOADER_MODULE} is not available")


@pytest.fixture(scope="module")
def runtime():
    """The EP runtime (NVSHMEM), bootstrapped once for the module's tests, as a
    process does that serves layers: a workspace's symmetric buffers live in one
    runtime."""
    _require()
    rank, world = _ranks()
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(intermediate_size=INTERMEDIATE)
    handle = bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(cfg).runtime_requirements(bootstrap)
    )
    yield rank, world
    finalize_moe_ep_runtime(handle)


def _problem(tokens: int):
    """Identical on every rank: the full expert bank and every rank's tokens."""
    gen = torch.Generator(device="cuda").manual_seed(20261114)
    w13 = (
        torch.randn(NUM_EXPERTS, 2 * INTERMEDIATE, HIDDEN, device="cuda", generator=gen)
        * HIDDEN**-0.5
    ).bfloat16()
    w2 = (
        torch.randn(NUM_EXPERTS, HIDDEN, INTERMEDIATE, device="cuda", generator=gen)
        * INTERMEDIATE**-0.5
    ).bfloat16()
    x = torch.randn(WORLD, tokens, HIDDEN, device="cuda", generator=gen).bfloat16()
    ids = torch.stack(
        [
            torch.stack(
                [
                    torch.randperm(NUM_EXPERTS, device="cuda", generator=gen)[:TOP_K]
                    for _ in range(tokens)
                ]
            )
            for _ in range(WORLD)
        ]
    ).to(torch.int32)
    scores = torch.softmax(
        torch.randn(WORLD, tokens, TOP_K, device="cuda", generator=gen), dim=-1
    )
    return w13, w2, x, ids, scores


@pytest.mark.gpu_4
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "max_tokens,tokens", [(1024, 1000), (4096, 4096), (8192, 6000)]
)
@pytest.mark.parametrize("scaled", [False, True])
def test_cake_nvfp4_mega_matches_the_reference(
    runtime, max_tokens, tokens, scaled
) -> None:
    from tests.moe_ep.sm107_test_utils import (
        assert_reference,
        canonical_reference,
        quantize,
        weight_pack,
    )

    rank, world = runtime
    w13, w2, x, ids, scores = _problem(tokens)
    pack, norm, scalars = weight_pack(w13, w2, "nvfp4", scaled=scaled, chunk=1)
    per_rank = NUM_EXPERTS // world
    local = slice(rank * per_rank, (rank + 1) * per_rank)
    local_pack = PrequantizedMoEWeights(
        *(getattr(pack, name)[local] for name in ("w13", "w2", "w13_scale", "w2_scale"))
    )
    cfg = Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
        intermediate_size=INTERMEDIATE, input_norm_const=norm
    )
    layers = []
    try:
        for prequantized in (False, True):
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(
                    rank=rank, world_size=world, auto_bootstrap=False
                ),
                fleet_params=FleetParams(
                    num_experts=NUM_EXPERTS,
                    max_tokens_per_rank=max_tokens,
                    token_hidden_size=HIDDEN,
                ),
                weights=local_pack,
                backend=MegaConfig(megakernel=cfg, quantize_input=not prequantized),
            )
            layers.append(layer)
            # Rank 0 is empty in the prequantized pass: the others still route
            # through it.
            live = 0 if prequantized and rank == 0 else tokens
            current_ids = ids[rank, :live].clone()
            current_ids[::3, 0] = -1
            current_ids[1::7] = -1
            current_scores = scores[rank, :live].clone()
            current_scores[::3, 0] = float("nan")
            current_scores[1::7] = float("inf")
            xq, xsf = quantize(x[rank, :live], "nvfp4", norm)
            t = MoEEpTensors(
                xq if prequantized else x[rank, :live],
                current_ids,
                current_scores,
                scales=xsf if prequantized else None,
                **{name: value[local] for name, value in scalars.items()},
            )
            reference = canonical_reference(
                xq, xsf, t.topk_ids, t.topk_weights, pack, "nvfp4", scalars=scalars
            )
            first = layer.forward(t).clone()
            assert_reference(first, reference, "nvfp4")
            # The programs leave every counter as they found it: a second
            # forward over the same workspace gives the same bits.
            assert torch.equal(layer.forward(t), first)
        assert layers[0]._workspace is layers[1]._workspace
    finally:
        # Destruction is an EP collective that frees the symmetric buffers:
        # every rank retires its forwards first. The empty rank's checks
        # never wait on the device, so without this it would enter the
        # collective free while its last forward is still in flight.
        torch.cuda.synchronize()
        for layer in layers:
            layer.destroy()


def _local_layer(rank, world, max_tokens, w13, w2, scaled):
    """This rank's experts of the bank ``w13``, ``w2`` in a layer of
    ``max_tokens`` tokens per rank, with the bank's weight pack and scalars
    (the reference's) and this rank's scalars (the layer's)."""
    from tests.moe_ep.sm107_test_utils import weight_pack

    pack, norm, scalars = weight_pack(w13, w2, "nvfp4", scaled=scaled, chunk=1)
    per_rank = NUM_EXPERTS // world
    local = slice(rank * per_rank, (rank + 1) * per_rank)
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(rank=rank, world_size=world, auto_bootstrap=False),
        fleet_params=FleetParams(
            num_experts=NUM_EXPERTS,
            max_tokens_per_rank=max_tokens,
            token_hidden_size=HIDDEN,
        ),
        weights=PrequantizedMoEWeights(
            *(
                getattr(pack, name)[local]
                for name in ("w13", "w2", "w13_scale", "w2_scale")
            )
        ),
        backend=MegaConfig(
            megakernel=Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
                intermediate_size=INTERMEDIATE, input_norm_const=norm
            )
        ),
    )
    local_scalars = {name: value[local] for name, value in scalars.items()}
    return layer, pack, norm, scalars, local_scalars


@pytest.mark.gpu_4
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "max_tokens,tokens", [(1024, 1000), (4096, 4096), (8192, 6000)]
)
def test_cake_nvfp4_mega_serves_a_rank_no_route_reaches(
    runtime, max_tokens, tokens
) -> None:
    """Every route picks one of the first three ranks' experts, so the last
    rank's mega program has no work and finishes ahead of its router: each
    forward still matches the reference, and repeats it bit for bit."""
    from tests.moe_ep.sm107_test_utils import (
        assert_reference,
        canonical_reference,
        quantize,
    )

    rank, world = runtime
    w13, w2 = _problem(1)[:2]
    layer, pack, norm, scalars, local_scalars = _local_layer(
        rank, world, max_tokens, w13, w2, scaled=True
    )
    try:
        gen = torch.Generator(device="cuda").manual_seed(20261117 + rank)
        reached = NUM_EXPERTS - NUM_EXPERTS // world
        x = torch.randn(tokens, HIDDEN, device="cuda", generator=gen).bfloat16()
        ids = (
            torch.rand(tokens, reached, device="cuda", generator=gen)
            .argsort(dim=-1)[:, :TOP_K]
            .to(torch.int32)
        )
        scores = torch.softmax(
            torch.randn(tokens, TOP_K, device="cuda", generator=gen), dim=-1
        )
        t = MoEEpTensors(x, ids, scores, **local_scalars)
        xq, xsf = quantize(x, "nvfp4", norm)
        reference = canonical_reference(
            xq, xsf, ids, scores, pack, "nvfp4", scalars=scalars
        )
        first = layer.forward(t).clone()
        assert_reference(first, reference, "nvfp4")
        for _ in range(20):
            assert torch.equal(layer.forward(t), first)
    finally:
        torch.cuda.synchronize()
        layer.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_rubin
def test_cake_nvfp4_mega_takes_unaligned_and_strided_views(runtime) -> None:
    """Activations that start one or two elements into their storage, and
    caller outputs that are strided or start off a 16-byte boundary, give the
    bits of the aligned forward."""
    rank, world = runtime
    tokens = 1000
    w13, w2, x, ids, scores = _problem(tokens)
    x, ids, scores = x[rank], ids[rank], scores[rank]
    layer, _, _, _, scalars = _local_layer(rank, world, 1024, w13, w2, scaled=False)
    try:
        aligned = layer.forward(MoEEpTensors(x, ids, scores, **scalars)).clone()
        for shift in (1, 2):
            storage = torch.empty(
                tokens * HIDDEN + shift, dtype=torch.bfloat16, device="cuda"
            )
            shifted = storage[shift:].view(tokens, HIDDEN)
            shifted.copy_(x)
            out = layer.forward(MoEEpTensors(shifted, ids, scores, **scalars))
            assert torch.equal(out, aligned)
        strided = torch.empty(tokens, 2 * HIDDEN, dtype=torch.bfloat16, device="cuda")[
            :, ::2
        ]
        offset = torch.empty(tokens * HIDDEN + 2, dtype=torch.bfloat16, device="cuda")[
            2:
        ].view(tokens, HIDDEN)
        for output in (strided, offset):
            layer.forward(MoEEpTensors(x, ids, scores, output=output, **scalars))
            assert torch.equal(output, aligned)
    finally:
        torch.cuda.synchronize()
        layer.destroy()


@pytest.mark.gpu_4
@pytest.mark.arch_rubin
def test_cake_nvfp4_mega_matches_the_cutedsl_route() -> None:
    """The same layer inputs through ``sm107_nvfp4_nvfp4_bf16_cutedsl``: the
    two routes agree within the reference's tolerance at 1024 tokens per rank
    (no dedup combine; their staging is the same instruction sequence)."""
    from flashinfer.moe_ep import Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
    from tests.moe_ep.sm107_test_utils import assert_reference, weight_pack

    _require()
    rank, world = _ranks()
    bootstrap = BootstrapConfig(rank=rank, world_size=world)
    ensure_moe_ep_cuda_device(bootstrap)
    tokens = 1024
    w13, w2, x, ids, scores = _problem(tokens)
    pack, norm, scalars = weight_pack(w13, w2, "nvfp4", scaled=True, chunk=1)
    per_rank = NUM_EXPERTS // world
    local = slice(rank * per_rank, (rank + 1) * per_rank)
    local_pack = PrequantizedMoEWeights(
        *(getattr(pack, name)[local] for name in ("w13", "w2", "w13_scale", "w2_scale"))
    )
    configs = (
        Sm107_Nvfp4_Nvfp4_Bf16_Cake_MegaMoeConfig(
            intermediate_size=INTERMEDIATE, input_norm_const=norm
        ),
        Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=INTERMEDIATE,
            top_k=TOP_K,
            gate_up_clamp=10.0,
            input_norm_const=norm,
        ),
    )
    t = MoEEpTensors(
        x[rank],
        ids[rank],
        scores[rank],
        **{name: value[local] for name, value in scalars.items()},
    )
    outputs = []
    for cfg in configs:
        runtime = bootstrap_moe_ep_runtime(
            bootstrap, create_mega_kernel(cfg).runtime_requirements(bootstrap)
        )
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(
                rank=rank, world_size=world, auto_bootstrap=False
            ),
            fleet_params=FleetParams(
                num_experts=NUM_EXPERTS,
                max_tokens_per_rank=tokens,
                token_hidden_size=HIDDEN,
            ),
            weights=local_pack,
            backend=MegaConfig(megakernel=cfg),
        )
        try:
            outputs.append(layer.forward(t).clone())
        finally:
            torch.cuda.synchronize()
            layer.destroy()
            finalize_moe_ep_runtime(runtime)
    assert_reference(outputs[0], outputs[1], "nvfp4")
