"""Public SM90 NVFP4 x FP8 EP4 execution with two independent references.

Run in its own process, separate from the SM100 vendor imports::

    torchrun --standalone --nproc_per_node=4 -m pytest \
        tests/moe_ep/test_moe_ep_sm90_pull_nvfp4_mega_multirank.py -v -rs

The caller supplies canonical packed E2M1/K16-E4M3 weights and natural BF16
inputs. The reference decodes and promotes those raw weights mathematically;
neither oracle reads preprocessed weights or calls the production converter.
The instruction-semantics oracle reuses the existing test-owned K32 WGMMA
and SM90 activation helpers; the plain FP32 oracle retains its accuracy gate.
"""

from __future__ import annotations

from dataclasses import replace
import os

import pytest


pytestmark = [pytest.mark.gpu_4, pytest.mark.arch_hopper]

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


def _raw_weights(rank, *, hidden_size=HIDDEN, intermediate_size=INTERMEDIATE):
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
        codes[:, 0, 0] = 8  # Retain the signed zero produced by NVFP4 quantizers.
        packed = (codes[..., 0::2] | (codes[..., 1::2] << 4)).contiguous()
        scales = torch.randint(
            48,
            64,
            (LOCAL_EXPERTS, rows, width // 16),
            dtype=torch.uint8,
            device="cuda",
            generator=generator,
        )
        # Cover the zero, minimum subnormal and maximum finite E4M3 scale
        # in every expert, while keeping most weights well conditioned.
        scales[:, 0, 0] = 1
        scales[:, 0, 1] = 0
        scales[:, 1, 0] = 126
        return packed, scales

    w13, sf13 = leg(2 * intermediate_size, hidden_size)
    w2, sf2 = leg(hidden_size, intermediate_size)
    expert = torch.arange(LOCAL_EXPERTS, dtype=torch.float32, device="cuda")
    return Nvfp4WeightPack(
        w13=w13,
        w2=w2,
        w13_scale=sf13,
        w2_scale=sf2,
        w13_global_scale=(expert + rank + 2) / 128,
        w2_global_scale=(expert + rank + 3) / 128,
    )


def _inputs(rank, launch, *, tokens=TOKENS, hidden_size=HIDDEN):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(2909 + 97 * launch + rank)
    hidden = (
        0.5 * torch.randn(tokens, hidden_size, device="cuda", generator=generator)
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


def _instruction_mm(a, b):
    """Independent RS K32 sequence, in bounded routed-row chunks."""
    import torch
    from tests.moe_ep._sm90_fp8_wgmma_reference import rs_k32_mm

    assert a.dtype == b.dtype == torch.float8_e4m3fn
    return torch.cat(
        [rs_k32_mm(a[begin : begin + 64], b) for begin in range(0, a.shape[0], 64)],
        dim=0,
    )


def _instruction_reference(
    hidden,
    ids,
    route_weights,
    raw,
    *,
    hidden_size=HIDDEN,
    intermediate_size=INTERMEDIATE,
):
    """Raw-weight oracle with SM90 FP8 instruction and epilogue semantics.

    PTX documents FP8 WGMMA accumulation below single precision even with F32
    outputs. Keep the separate ideal FP32 oracle as the accuracy L2 gate.
    See the wgmma.mma_async precision note in the PTX ISA:
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
    No production weight layout, converter, dispatch or MegaMoE is reused.
    """
    import torch
    from tests.moe_ep._sm90_fp8_wgmma_reference import fma_add

    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        _swiglu_pair_hw_match_cuda,
    )

    ranks, tokens, width = hidden.shape
    w13, w2, sf13, sf2, g13, g2 = (t.flatten(0, 1) for t in raw)
    w13 = _promoted_weight(w13, sf13)
    w2 = _promoted_weight(w2, sf2)
    hidden = hidden.flatten(0, 1).float()
    ids = ids.flatten(0, 1)
    route_weights = route_weights.flatten(0, 1)
    input_scale = (hidden.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(
        1e-30
    )
    x = (hidden / input_scale).to(torch.float8_e4m3fn)
    order = _k64_order(hidden_size, hidden.device)
    group_order = _k64_order(64, hidden.device)
    terms = torch.zeros(
        hidden.shape[0],
        WORLD_SIZE,
        hidden_size,
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    for expert in range(WORLD_SIZE * LOCAL_EXPERTS):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        assert token.numel() > 0, f"global expert {expert} was not exercised"
        fc1 = _instruction_mm(
            x[token].index_select(1, order), w13[expert].index_select(1, order).T
        )
        # Native mainloop applies token SF first; epilogue restores 8G next.
        fc1 *= input_scale[token]
        fc1 *= 8 * g13[expert]
        gate = fc1[:, :intermediate_size].clamp(max=CLAMP)
        up = fc1[:, intermediate_size:].clamp(-CLAMP, CLAMP)
        activated = _swiglu_pair_hw_match_cuda(gate, up)
        activated *= route_weights[token, slot, None]
        grouped = activated.reshape(-1, intermediate_size // 64, 64)
        scale = (grouped.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(
            1e-30
        )
        quantized = (grouped * torch.reciprocal(scale)).to(torch.float8_e4m3fn)
        fc2 = torch.zeros(token.numel(), hidden_size, device=hidden.device)
        for block in range(intermediate_size // 64):
            a = quantized[:, block].index_select(1, group_order)
            b = w2[expert, :, block * 64 : (block + 1) * 64].index_select(
                1, group_order
            )
            fc2 = fma_add(fc2, _instruction_mm(a, b.T), scale[:, block])
        terms[token, slot] = (fc2 * (8 * g2[expert])).to(torch.bfloat16)
    return terms.float().sum(1).reshape(ranks, tokens, width)


def _reference(
    hidden,
    ids,
    route_weights,
    raw,
    *,
    hidden_size=HIDDEN,
    intermediate_size=INTERMEDIATE,
    instruction_exact=False,
):
    """Global FP32 GEMMs with the public FP8/BF16 data-format contract."""
    if instruction_exact:
        return _instruction_reference(
            hidden,
            ids,
            route_weights,
            raw,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
        )
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
    input_scale = (hidden.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(
        1e-30
    )
    x = (hidden / input_scale).to(torch.float8_e4m3fn).float()
    order = _k64_order(hidden_size, hidden.device)
    group_order = _k64_order(64, hidden.device)
    terms = torch.zeros(
        hidden.shape[0],
        WORLD_SIZE,
        hidden_size,
        dtype=torch.bfloat16,
        device=hidden.device,
    )
    for expert in range(WORLD_SIZE * LOCAL_EXPERTS):
        token, slot = (ids == expert).nonzero(as_tuple=True)
        assert token.numel() > 0, f"global expert {expert} was not exercised"
        # Both GEMM operands follow pi64. No NV0/NV64 bitwise equality is assumed.
        fc1 = x[token].index_select(1, order) @ w13[expert].index_select(1, order).T
        fc1 *= input_scale[token] * (8 * g13[expert])
        gate = fc1[:, :intermediate_size].clamp(max=CLAMP)
        up = fc1[:, intermediate_size:].clamp(-CLAMP, CLAMP)
        activated = functional.silu(gate) * up * route_weights[token, slot, None]
        grouped = activated.reshape(-1, intermediate_size // 64, 64)
        scale = (grouped.abs().amax(-1, keepdim=True) * (1.0 / FP8_MAX)).clamp_min(
            1e-30
        )
        quantized = (grouped * torch.reciprocal(scale)).to(torch.float8_e4m3fn).float()
        fc2 = torch.zeros(token.numel(), hidden_size, device=hidden.device)
        for block in range(intermediate_size // 64):
            a = quantized[:, block].index_select(1, group_order)
            b = w2[expert, :, block * 64 : (block + 1) * 64].index_select(
                1, group_order
            )
            fc2 += (a @ b.T) * scale[:, block]
        # The public BF16-combine path stores a BF16 term for each routed slot.
        terms[token, slot] = (fc2 * (8 * g2[expert])).to(torch.bfloat16)
    return terms.float().sum(1).reshape(ranks, tokens, width)


def _require_sm90_ep4():
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        pytest.skip("requires four visible CUDA GPUs")
    if any(torch.cuda.get_device_capability(i) != (9, 0) for i in range(WORLD_SIZE)):
        pytest.skip("requires four SM90 GPUs")
    assert int(os.environ.get("WORLD_SIZE", "1")) == WORLD_SIZE, (
        "launch this test with torchrun --nproc_per_node=4"
    )
    pytest.importorskip(
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel"
    )


def _factory_conversion_knobs():
    # Factory conversion fields on a small K256 geometry: lane/R32 exercises
    # PRMT addressing; one dispatch warp makes compact storage effective.
    return {
        "mma_tiler_mnk": (256, 16, 256),
        "cluster_shape_mnk": (1, 1, 1),
        "group_hint": 132,
        "num_sched_stages": 1,
        "pingpong": False,
        "swap_ab": True,
        "fp8_accum_mode": "1xacc",
        "load_balance_mode": "atomic_counter",
        "token_back_mode": "epi_warps",
        "dedup_dispatch": False,
        "active_dispatch_warps": 1,
        "fc1_store_offload": False,
        "fc1_early_done_publish": False,
        "fold_producer_warps": False,
        "grouped_token_back": False,
        "combine_format": "bf16",
        "in_kernel_fc2_reduce": False,
        "fc2_tail_n8": False,
        "fc1_ready_mode": "tile",
        "tail_split_pairs": False,
        "compact_pull_buffer": True,
        "nvfp4_lut_replicas": 32,
        "nvfp4_lut_map": "lane",
    }


@pytest.mark.parametrize("conversion_layout", ["fallback", "factory", "sf2-r1"])
def test_sm90_nvfp4_public_ep4_matches_raw_weight_oracles(conversion_layout):
    import torch

    _require_sm90_ep4()

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
        preprocess_sm90_pull_nvfp4_mega_weights,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    bootstrap = BootstrapConfig(world_size=WORLD_SIZE, rank=rank, device=local_rank)
    ensure_moe_ep_cuda_device(bootstrap)
    scale_ring = conversion_layout == "sf2-r1"
    tokens = 1024 if scale_ring else TOKENS
    hidden_size = 1280 if scale_ring else HIDDEN
    intermediate_size = 1280 if scale_ring else INTERMEDIATE
    knobs = _factory_conversion_knobs() if conversion_layout == "factory" else None
    if scale_ring:
        # Five K256 tiles exercise steady-state scale production and ring
        # wrap. FC2's M=1280 also pads the final two-CTA cluster. Production
        # references remain independent of the production converter.
        knobs = {
            "mma_tiler_mnk": (256, 64, 256),
            "cluster_shape_mnk": (2, 1, 1),
            "group_hint": 528,
            "num_sched_stages": 1,
            "pingpong": False,
            "swap_ab": True,
            "fp8_accum_mode": "1xacc",
            "load_balance_mode": "atomic_counter",
            "token_back_mode": "epi_warps",
            "dedup_dispatch": False,
            "active_dispatch_warps": 1,
            "fc1_store_offload": False,
            "fc1_early_done_publish": True,
            "fold_producer_warps": True,
            "grouped_token_back": False,
            "combine_format": "bf16",
            "in_kernel_fc2_reduce": False,
            "fc2_tail_n8": False,
            "fc1_ready_mode": "tile",
            "tail_split_pairs": False,
            "compact_pull_buffer": True,
            "nvfp4_lut_replicas": 1,
            "nvfp4_lut_map": "lane",
            "nvfp4_sf_stages": 2,
        }
    config = Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=intermediate_size,
        top_k=WORLD_SIZE,
        gate_up_clamp=CLAMP,
        knobs=knobs,
        nvfp4_fc1_k256_overlap=True if scale_ring else None,
    )
    assert config.nvfp4_kperm == 64
    weights = _raw_weights(
        rank, hidden_size=hidden_size, intermediate_size=intermediate_size
    )
    fleet = FleetParams(
        num_experts=WORLD_SIZE * LOCAL_EXPERTS,
        max_tokens_per_rank=tokens,
        token_hidden_size=hidden_size,
    )
    prepared = preprocess_sm90_pull_nvfp4_mega_weights(
        weights,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        nvfp4_kperm=config.nvfp4_kperm,
    )
    # Equal tensor shapes cannot distinguish K32 from K64. Validation must
    # reject a layout-tag mismatch before a layer can launch either GEMM.
    mismatched = create_mega_kernel(replace(config, nvfp4_kperm=32))
    mismatched.bind_ep_bootstrap(bootstrap)
    with pytest.raises(ValueError, match="kperm"):
        mismatched.validate_transformed_weights(prepared, bootstrap, fleet)
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=fleet,
        weights=None,
        backend=MegaConfig(
            megakernel=config,
            preprocess_weights=False,
            transformed_weights=prepared,
        ),
    )
    baseline_layer = None
    try:
        if scale_ring:
            # Use a second public layer so constructing the fixed sf0 baseline
            # cannot replace the sf2 layer's default workspace. Only the ring
            # depth differs; K64 weights, geometry and overlap remain identical.
            baseline_layer = MoEEpLayer(
                bootstrap=bootstrap,
                fleet_params=FleetParams(
                    num_experts=WORLD_SIZE * LOCAL_EXPERTS,
                    max_tokens_per_rank=tokens,
                    token_hidden_size=hidden_size,
                ),
                weights=None,
                backend=MegaConfig(
                    megakernel=replace(config, knobs=dict(knobs, nvfp4_sf_stages=0)),
                    preprocess_weights=False,
                    transformed_weights=prepared,
                ),
            )
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
            hidden, ids, route_weights = _inputs(
                rank, launch, tokens=tokens, hidden_size=hidden_size
            )
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
            baseline = None
            if baseline_layer is not None:
                baseline = baseline_layer.forward(tensors).clone()
            actual = layer.forward(tensors).clone()
            repeated = None
            if launch == 1:
                repeated = layer.forward(tensors).clone()
            actual = _gather(actual)
            if baseline is not None:
                baseline = _gather(baseline)
                torch.testing.assert_close(
                    actual.view(torch.uint8), baseline.view(torch.uint8), atol=0, rtol=0
                )
            if repeated is not None:
                repeated = _gather(repeated)
                torch.testing.assert_close(
                    repeated.view(torch.uint8), actual.view(torch.uint8), atol=0, rtol=0
                )
            allow_tf32 = torch.backends.cuda.matmul.allow_tf32
            try:
                torch.backends.cuda.matmul.allow_tf32 = False
                expected = _reference(
                    *gathered,
                    raw,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                )
                instruction_expected = _reference(
                    *gathered,
                    raw,
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    instruction_exact=True,
                )
            finally:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            assert actual.dtype == torch.bfloat16 and torch.isfinite(actual).all()
            assert not torch.equal(expected[0], expected[1])
            # Elementwise correctness follows the documented FP8 instruction
            # semantics; the independent plain FP32 accuracy gate below is unchanged.
            torch.testing.assert_close(
                actual.float(), instruction_expected, atol=0.02, rtol=0.02
            )
            for source in range(WORLD_SIZE):
                norm = expected[source].norm()
                assert norm > 1e-3
                rel_l2 = (actual[source].float() - expected[source]).norm() / norm
                assert rel_l2 < 0.025, (source, launch, rel_l2.item())
    finally:
        if baseline_layer is not None:
            baseline_layer.destroy()
        layer.destroy()


def test_sm90_nvfp4_cached_fc1_schedule_switch_during_graph_capture(monkeypatch):
    import torch
    import torch.distributed as dist

    _require_sm90_ep4()
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MegaConfig,
        MoEEpLayer,
        MoEEpTensors,
        Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig,
        ensure_moe_ep_cuda_device,
        preprocess_sm90_pull_nvfp4_mega_weights,
    )
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
        hopper_nvfp4,
    )

    # Force a small automatic threshold; compiled schedules, workspace and
    # kernels remain real. The production threshold requires a large model.
    def select_overlap(*, num_tokens, override, **_kwargs):
        return num_tokens >= TOKENS if override is None else override

    monkeypatch.setattr(hopper_nvfp4, "resolve_nvfp4_fc1_overlap", select_overlap)
    rank = int(os.environ["RANK"])
    bootstrap = BootstrapConfig(
        world_size=WORLD_SIZE,
        rank=rank,
        device=int(os.environ.get("LOCAL_RANK", str(rank))),
    )
    ensure_moe_ep_cuda_device(bootstrap)
    config = Sm90_Fp8_Nvfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=INTERMEDIATE,
        top_k=WORLD_SIZE,
        gate_up_clamp=CLAMP,
        knobs=_factory_conversion_knobs(),
    )
    prepared = preprocess_sm90_pull_nvfp4_mega_weights(
        _raw_weights(rank),
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        nvfp4_kperm=config.nvfp4_kperm,
    )
    layer = MoEEpLayer(
        bootstrap=bootstrap,
        fleet_params=FleetParams(
            num_experts=WORLD_SIZE * LOCAL_EXPERTS,
            max_tokens_per_rank=TOKENS,
            token_hidden_size=HIDDEN,
        ),
        weights=None,
        backend=MegaConfig(
            megakernel=config,
            preprocess_weights=False,
            transformed_weights=prepared,
        ),
    )
    graphs = []
    captured = None
    cold_graph = None
    try:
        batches = {}
        for tokens in (TOKENS // 2, TOKENS):
            hidden, ids, weights = _inputs(rank, 0, tokens=tokens)
            batches[tokens] = MoEEpTensors(
                hidden_states=hidden, topk_ids=ids, topk_weights=weights
            )
        layer.warmup(batches[TOKENS // 2])
        frontend = layer._workspace._frontend
        assert set(frontend._nvfp4_schedule_cache) == {False}
        # A missing schedule still requires eager compilation on every rank.
        cold_graph = torch.cuda.CUDAGraph()
        with (
            pytest.raises(RuntimeError, match="NVFP4 FC1 schedule change.*cannot run"),
            torch.cuda.graph(cold_graph),
        ):
            layer.forward(batches[TOKENS])
        cold_graph = None
        dist.barrier()
        layer.warmup(batches[TOKENS])
        assert set(frontend._nvfp4_schedule_cache) == {False, True}
        expected = {
            tokens: layer.forward(batch).clone() for tokens, batch in batches.items()
        }
        for tokens in (TOKENS, TOKENS // 2):
            other = TOKENS // 2 if tokens == TOKENS else TOKENS
            layer.forward(batches[other])
            torch.cuda.synchronize()
            dist.barrier()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = layer.forward(batches[tokens])
            graphs.append(graph)
            assert frontend._nvfp4_fc1_k256_overlap is (tokens == TOKENS)
            dist.barrier()
            for _ in range(2):
                graph.replay()
                torch.cuda.synchronize()
                dist.barrier()
                torch.testing.assert_close(
                    captured.view(torch.uint8),
                    expected[tokens].view(torch.uint8),
                    atol=0,
                    rtol=0,
                )
    finally:
        torch.cuda.synchronize()
        captured = None
        cold_graph = None
        graph = None
        graphs.clear()
        layer.destroy()
