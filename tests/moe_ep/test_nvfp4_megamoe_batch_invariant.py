# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Native NVFP4 MegaMoE bitwise checks within each arithmetic class.

Run the single-rank cases with pytest, or launch the EP cases with torchrun.
Compute Sanitizer should also run the single-rank target: matching output bits
alone does not establish memory or synchronization correctness.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import os

import pytest

pytestmark = pytest.mark.gpu


def _device(provider):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("NVFP4 BI requires SM100/SM103/SM107")
    if cc == (10, 7) and provider == "legacy":
        pytest.skip("legacy NVFP4 BI is a Blackwell implementation")
    return cc


def _assert_bits(actual, expected, context):
    from ._megamoe_bi_helpers import assert_bits

    assert_bits(actual, expected, context)


def _problem(rank, world, hidden=384, intermediate=192, *, top_k=3, local_experts=4):
    import torch
    from flashinfer.moe_ep import MoEWeightPack

    gen = torch.Generator(device="cuda").manual_seed(732 + rank)
    x = torch.randn(257, hidden, device="cuda", dtype=torch.bfloat16, generator=gen)
    scores = torch.rand(257, top_k, device="cuda", generator=gen)
    scores /= scores.sum(-1, keepdim=True)
    scores[19].zero_()
    if top_k > 1:
        scores[20, 1] = 0
    route = (
        [local_experts * rank + i for i in range(top_k)]
        if top_k != 3
        else [
            local_experts * rank,
            local_experts * rank + 1,
            (
                local_experts * rank + 2
                if world == 1
                else local_experts * ((rank + 1) % world)
            ),
        ]
    )
    ids = (
        torch.tensor(
            route,
            device="cuda",
            dtype=torch.int64,
        )
        .expand(257, top_k)
        .contiguous()
    )
    w13 = (
        torch.randn(
            local_experts,
            2 * intermediate,
            hidden,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.05
    )
    w2 = (
        torch.randn(
            local_experts,
            hidden,
            intermediate,
            device="cuda",
            dtype=torch.bfloat16,
            generator=gen,
        )
        * 0.05
    )
    return x, scores, ids, MoEWeightPack(w13=w13, w2=w2)


def _config(
    provider,
    early,
    hidden,
    intermediate,
    *,
    persistent=False,
    batch_invariant=True,
    knobs=None,
    top_k=3,
    local_experts=4,
):
    import torch
    from flashinfer import moe_ep

    common = dict(
        intermediate_size=intermediate,
        top_k=top_k,
        batch_invariant=batch_invariant,
        knobs=knobs,
        gate_up_clamp=2.5,
        apply_routing_weights_before_fc2=early,
        input_norm_const=1.7,
        fc1_alpha=torch.tensor(
            [0.75, 1.0, 1.25, 2.0] * ((local_experts + 3) // 4), device="cuda"
        )[:local_experts],
        fc2_alpha=torch.tensor(
            [1.5, 0.75, 1.0, 1.25] * ((local_experts + 3) // 4), device="cuda"
        )[:local_experts],
        fc1_norm_const=torch.tensor(
            [1.0, 1.25, 0.75, 1.5] * ((local_experts + 3) // 4), device="cuda"
        )[:local_experts],
    )
    if torch.cuda.get_device_capability() == (10, 7):
        assert not persistent
        return moe_ep.Sm107_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
            **common,
            kernel_variant="genphase" if provider == "genphase" else "inference",
            cluster_shape_mn=(4, 1) if provider == "genphase" else (2, 1),
            fc2_use_bulk=provider == "genphase",
        )
    return moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        **common,
        kernel_name="sm100_nvfp4_nvfp4_bf16_cutedsl"
        + ("_next" if provider != "legacy" else ""),
        kernel_variant="genphase" if provider == "genphase" else "inference",
        finalize_mode=(
            moe_ep.FinalizeMode.PERSISTENT
            if persistent
            else moe_ep.FinalizeMode.DEFAULT
        ),
        num_valid_tokens_tensor=(
            torch.zeros(1, device="cuda", dtype=torch.int32) if persistent else None
        ),
    )


@contextmanager
def _layer(config, weights, rank, world, capacity, hidden, *, quantize=True):
    from flashinfer import moe_ep

    layer = moe_ep.MoEEpMegaLayer(
        bootstrap=moe_ep.BootstrapConfig(
            rank=rank, world_size=world, auto_bootstrap=False
        ),
        fleet_params=moe_ep.FleetParams(
            num_experts=weights.w13.shape[0] * world,
            max_tokens_per_rank=capacity,
            token_hidden_size=hidden,
        ),
        weights=weights,
        backend=moe_ep.MegaConfig(megakernel=config, quantize_input=quantize),
    )
    try:
        yield layer
    except BaseException as error:
        from ._megamoe_bi_helpers import fail_stop_distributed

        fail_stop_distributed(error, world)
        raise
    finally:
        layer.destroy()


def _run(provider, early, rank, world, full):
    import torch
    from flashinfer.moe_ep import MoEEpTensors
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
        nvfp4_batch_invariant_fingerprint,
    )

    hidden, intermediate = 384, 192
    x, scores, ids, weights = _problem(rank, world, hidden, intermediate)
    config = _config(provider, early, hidden, intermediate)
    receipt = nvfp4_batch_invariant_fingerprint(
        config, hidden=hidden, num_experts=4 * world
    )
    with _layer(config, weights, rank, world, 257, hidden) as layer:
        anchor = layer.forward(MoEEpTensors(x, ids, scores)).clone()
        torch.cuda.synchronize()
        assert layer.batch_invariant_fingerprint == receipt
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        assert torch.count_nonzero(anchor[19]) == 0

    if provider == "legacy":
        schedule_knobs = tuple(
            dict(load_balance_mode=mode) for mode in ("static", "atomic_counter")
        )
    else:
        schedule_knobs = tuple(
            dict(work_id_mode=mode) for mode in ("grid_stride", "atomic_counter")
        )
    if provider == "standard":
        schedule_knobs += (
            dict(
                schedule_policy=("phase_interleave", None),
                work_id_mode="atomic_counter",
            ),
        )
    carriers = (
        ("epi_warps",)
        if provider == "genphase"
        else (
            ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
            if full or provider == "standard"
            else ("epi_warps", "reuse_dispatch_warps")
        )
    )
    blackwell_genphase = (
        provider == "genphase" and torch.cuda.get_device_capability() != (10, 7)
    )
    if blackwell_genphase:
        schedule_knobs = ({},)
    capacities = (1, 129, 257) if full else (257,)
    for schedule in schedule_knobs:
        for carrier in carriers:
            knobs = dict(token_back_mode=carrier, **schedule)
            if blackwell_genphase:
                knobs = {}
            for capacity in capacities:
                selected = replace(config, knobs=knobs)
                with _layer(selected, weights, rank, world, capacity, hidden) as layer:
                    for n in (
                        0,
                        1,
                        4,
                        8,
                        16,
                        17,
                        31,
                        32,
                        63,
                        64,
                        65,
                        127,
                        128,
                        129,
                        257,
                    ):
                        if n > capacity:
                            continue
                        order = (torch.arange(n, device="cuda") * 37) % 257
                        for order in (order, order.flip(0), order.remainder(17)):
                            inputs = MoEEpTensors(x[order], ids[order], scores[order])
                            for _ in range(2):
                                _assert_bits(
                                    layer.forward(inputs),
                                    anchor[order],
                                    f"{provider}, {knobs}, T={n}",
                                )
                    if world > 1:
                        for n in (
                            0 if rank == 0 else 1,
                            1 if rank == 0 else min(capacity, 129),
                        ):
                            order = torch.arange(n, device="cuda")
                            _assert_bits(
                                layer.forward(
                                    MoEEpTensors(x[order], ids[order], scores[order])
                                ),
                                anchor[order],
                                f"{provider}, {knobs}, uneven/empty rank T={n}",
                            )
                    if capacity < 4:
                        continue
                    order = torch.tensor([19, 20, 1, 53], device="cuda")
                    inputs = MoEEpTensors(
                        x[order].clone(), ids[order].clone(), scores[order].clone()
                    )
                    # Capture a prepared forward with fixed input/output addresses.
                    output = torch.empty_like(anchor[order])
                    inputs.output = output
                    layer.forward(inputs)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        layer.forward(inputs)
                    for masked in (False, True, False):
                        prime = order.roll(2)
                        inputs.hidden_states.copy_(x[prime])
                        inputs.topk_ids.copy_(ids[prime])
                        inputs.topk_weights.copy_(scores[prime])
                        graph.replay()
                        _assert_bits(output, anchor[prime], "graph prime")
                        inputs.hidden_states.copy_(x[order])
                        inputs.topk_ids.copy_(ids[order])
                        inputs.topk_weights.copy_(scores[order])
                        if masked:
                            inputs.topk_ids[0].fill_(-1)
                            inputs.topk_weights[0].fill_(float("nan"))
                            inputs.topk_ids[1, 1] = -1
                            inputs.topk_weights[1, 1] = float("nan")
                        graph.replay()
                        _assert_bits(output, anchor[order], "graph masked routing")
                    del graph
                    assert layer.batch_invariant_fingerprint == receipt


@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
def test_nvfp4_batch_invariant_autotune_disk_reload(
    provider, early, monkeypatch, tmp_path
):
    from ._megamoe_bi_helpers import autotune_disk_reload

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank autotune test")
    _device(provider)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    hidden, intermediate = 384, 192
    x, scores, ids, weights = _problem(0, 1, hidden, intermediate)
    autotune_disk_reload(
        make_config=lambda knobs: _config(
            provider, early, hidden, intermediate, knobs=knobs
        ),
        weights=weights,
        x=x,
        ids=ids,
        scores=scores,
        layer_factory=_layer,
        provider=provider,
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize(
    "shape,top_k", (((2048, 2048), 3), ((128, 128), 1), ((128, 128), 32))
)
def test_nvfp4_batch_invariant_long_k_and_topk_boundaries(
    provider, early, shape, top_k, monkeypatch
):
    import torch
    from ._megamoe_bi_helpers import boundary_batches

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank boundary test")
    _device(provider)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    local_experts = max(4, top_k)
    x, scores, ids, weights = _problem(
        0, 1, *shape, top_k=top_k, local_experts=local_experts
    )
    scores.mul_(max(1, top_k / 3))
    scores[:, 0].neg_()
    config = _config(provider, early, *shape, top_k=top_k, local_experts=local_experts)
    if provider == "legacy":
        other = dict(
            load_balance_mode="atomic_counter", token_back_mode="reuse_dispatch_warps"
        )
    elif provider == "standard":
        other = dict(
            work_id_mode="atomic_counter", token_back_mode="reuse_dispatch_warps"
        )
    elif torch.cuda.get_device_capability() == (10, 7):
        other = dict(work_id_mode="atomic_counter")
    else:
        other = {}
    boundary_batches(config, weights, x, ids, scores, _layer, ({}, other))


@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
def test_nvfp4_batch_invariant_batches_plans_and_graphs(
    provider, early, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _device(provider)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    _run(provider, early, 0, 1, request.config.getoption("--full"))


@pytest.mark.parametrize("provider", ("legacy", "standard"))
@pytest.mark.parametrize("early", (False, True))
def test_nvfp4_batch_invariant_persistent_live_count(provider, early, monkeypatch):
    """DEFAULT and PERSISTENT share bits, including a captured device count."""
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    if _device(provider) == (10, 7):
        pytest.skip("SM107 does not expose persistent finalization")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    hidden, intermediate, capacity = 384, 192, 129
    x, scores, ids, weights = _problem(0, 1, hidden, intermediate)
    default = _config(provider, early, hidden, intermediate)
    with _layer(default, weights, 0, 1, capacity, hidden) as layer:
        anchor = layer.forward(
            MoEEpTensors(x[:capacity], ids[:capacity], scores[:capacity])
        ).clone()
        receipt = layer.batch_invariant_fingerprint

    persistent = _config(provider, early, hidden, intermediate, persistent=True)
    live_count = persistent.num_valid_tokens_tensor
    assert live_count is not None
    with _layer(persistent, weights, 0, 1, capacity, hidden) as layer:
        for n in (0, 1, 31, 32, 64, 129):
            order = (torch.arange(n, device="cuda") * 37) % capacity
            live_count.fill_(n)
            actual = layer.forward(MoEEpTensors(x[order], ids[order], scores[order]))
            _assert_bits(actual, anchor[order], f"{provider} persistent eager T={n}")
        assert layer.batch_invariant_fingerprint == receipt

        inputs = MoEEpTensors(
            x[:capacity].clone(), ids[:capacity].clone(), scores[:capacity].clone()
        )
        live_count.fill_(capacity)
        layer.forward(inputs, return_workspace_view=True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = layer.forward(inputs, return_workspace_view=True)
            # Input shapes, graph nodes and output addresses stay fixed. Only
            # this caller-owned CUDA scalar changes the reducer's live extent.
            # A borrowed view avoids a capacity-sized output copy hiding writes
            # past the live count (the legacy backend otherwise adds that copy).
            sentinel = torch.full_like(output, 123.0)
            for n in (0, 1, 31, 32, 64, 129, 1, 0, 129):
                output.copy_(sentinel)
                live_count.fill_(n)
                graph.replay()
                expected = sentinel.clone()
                expected[:n].copy_(anchor[:n])
                _assert_bits(output, expected, f"{provider} persistent graph live={n}")
        finally:
            graph.reset()


def _gather_packed_rows(packed, scales, order):
    """Gather bytes: indexing support for packed FP4/FP8 varies by Torch version."""
    import torch

    q = packed.view(torch.uint8)[order].contiguous().view(packed.dtype)
    sf = scales.view(torch.uint8)[order].contiguous().view(scales.dtype)
    return q, sf


@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
def test_nvfp4_batch_invariant_prequantized_batches(provider, early, monkeypatch):
    """Prequantized calls compare the same supplied bytes across batches."""
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _device(provider)
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    hidden, intermediate = 384, 192
    x, scores, ids, weights = _problem(0, 1, hidden, intermediate)
    config = _config(provider, early, hidden, intermediate)
    packed, scales = nvfp4_quantize_per_block_16(x.float(), config.input_norm_const)
    with _layer(config, weights, 0, 1, 257, hidden, quantize=False) as layer:
        anchor = layer.forward(MoEEpTensors(packed, ids, scores, scales=scales)).clone()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        receipt = layer.batch_invariant_fingerprint
    # This does not assert equality with BF16 staging: the packed boundary is
    # an operator input, and an independent quantizer may choose other FP4 bits.
    with _layer(config, weights, 0, 1, 129, hidden, quantize=False) as layer:
        for n in (0, 1, 8, 16, 17, 65, 129):
            order = ((torch.arange(n, device="cuda") * 37) % 257).flip(0)
            q, sf = _gather_packed_rows(packed, scales, order)
            actual = layer.forward(
                MoEEpTensors(q, ids[order], scores[order], scales=sf)
            )
            _assert_bits(actual, anchor[order], f"{provider} prequantized T={n}")
        assert layer.batch_invariant_fingerprint == receipt


def _plain_quantized_weights(weights, intermediate):
    """Independent plain quantization, before any provider's weight transform."""
    import torch
    from flashinfer.moe_ep import MoEWeightPack
    from tests.moe_ep.test_nvfp4_cutedsl_kernel_vs_reference import (
        _plain_nvfp4_from_bf16,
    )

    plain = _plain_nvfp4_from_bf16(
        dict(w13=weights.w13, w2=weights.w2, intermediate=intermediate)
    )

    def canonical(tensor):
        # The oracle consumes [gate16, up16] stripes, while the public pack is
        # [all gates, all ups]. Invert only that row permutation; packed K and
        # the plain scale columns remain untouched by the provider layout.
        experts, rows, cols = tensor.shape
        raw = tensor.view(torch.uint8)
        return (
            raw.view(experts, intermediate // 16, 2, 16, cols)
            .transpose(1, 2)
            .contiguous()
            .view(experts, rows, cols)
            .view(tensor.dtype)
        )

    q1, sf1, q2, sf2 = plain
    return (
        MoEWeightPack(w13=canonical(q1), w2=q2, w13_scale=canonical(sf1), w2_scale=sf2),
        plain,
    )


def _torch_reference(
    packed, scales, ids, scores, plain, config, hidden, intermediate, early
):
    import torch
    from tests.moe_ep.test_nvfp4_cutedsl_kernel_vs_reference import (
        _torch_nvfp4_mega_reference,
    )

    q1, sf1, q2, sf2 = plain
    kwargs = dict(
        act_packed=packed,
        act_sf=scales,
        fc1_weight=q1,
        fc1_sf=sf1,
        fc2_weight=q2,
        fc2_sf=sf2,
        hidden=hidden,
        intermediate=intermediate,
        gate_up_clamp=config.gate_up_clamp,
        fc1_alpha=config.fc1_alpha,
        fc2_alpha=config.fc2_alpha,
        fc1_norm_const=config.fc1_norm_const,
    )
    if early:
        return _torch_nvfp4_mega_reference(topk_idx=ids, topk_weights=scores, **kwargs)
    # The existing oracle applies weights before FC2. Run it with one unweighted
    # route at a time to obtain the BF16 wire terms, then apply late weights.
    # BF16 x FP32 products are exact in FP64; round after each ordered addition.
    acc = torch.zeros(ids.shape[0], hidden, device=packed.device, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        term = _torch_nvfp4_mega_reference(
            topk_idx=ids[:, slot : slot + 1],
            topk_weights=torch.ones_like(scores[:, slot : slot + 1]),
            **kwargs,
        )
        acc = (acc.double() + term.double() * scores[:, slot, None].double()).float()
    return acc.bfloat16()


def _distributed_oracle(provider, early, rank, world):
    from types import SimpleNamespace

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )
    from ._megamoe_bi_helpers import (
        all_gather_rows,
        distributed_reference,
        remote_routes,
    )

    hidden, intermediate = 384, 192
    x, scores, ids, weights = _problem(rank, world, hidden, intermediate)
    config = _config(provider, early, hidden, intermediate)
    packed_weights, plain = _plain_quantized_weights(weights, intermediate)
    global_plain = tuple(all_gather_rows(value) for value in plain)
    reference_config = SimpleNamespace(
        gate_up_clamp=config.gate_up_clamp,
        fc1_alpha=all_gather_rows(config.fc1_alpha),
        fc2_alpha=all_gather_rows(config.fc2_alpha),
        fc1_norm_const=all_gather_rows(config.fc1_norm_const),
    )
    packed, scales = nvfp4_quantize_per_block_16(
        x[:31].float(), config.input_norm_const
    )
    ids = remote_routes(ids[:31], rank, world, weights.w13.shape[0])

    def reference(q, sf, route_ids, route_scores):
        return _torch_reference(
            q,
            sf,
            route_ids,
            route_scores,
            global_plain,
            reference_config,
            hidden,
            intermediate,
            early,
        )

    distributed_reference(
        config=config,
        weights=packed_weights,
        packed=packed,
        scales=scales,
        ids=ids,
        scores=scores[:31],
        reference=reference,
        layer_factory=_layer,
        hidden=hidden,
        rank=rank,
        world=world,
        tolerance=0.02,
    )


@pytest.fixture
def _ieee_matmul():
    import torch

    # Use the public precision API: some Torch builds expose an allow_tf32
    # setter without a getter, which makes monkeypatch restoration invalid.
    previous = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(previous)


@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("batch_invariant", (False, True))
def test_nvfp4_batch_invariant_independent_reference(
    provider, early, batch_invariant, monkeypatch, _ieee_matmul
):
    """Whole-op numerical check, separate from the exact BI comparisons."""
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _device(provider)
    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
        nvfp4_quantize_per_block_16,
    )
    from tests.moe_ep.test_nvfp4_cutedsl_kernel_vs_reference import _oracle_atol
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.nvfp4_batch_invariant import (
        nvfp4_batch_invariant_default_knobs,
    )

    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    hidden, intermediate, n = 384, 192, 33
    x, scores, ids, weights = _problem(0, 1, hidden, intermediate)
    cc = torch.cuda.get_device_capability()
    knobs = nvfp4_batch_invariant_default_knobs(provider, arch=f"sm_{cc[0]}{cc[1]}")
    if provider == "genphase" and cc != (10, 7):
        knobs = {
            name: knobs[name]
            for name in ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        }
    # Both modes use the same explicit, legal geometry. The False branch still
    # exercises the provider's ordinary arithmetic and dynamic-N implementation.
    config = _config(
        provider,
        early,
        hidden,
        intermediate,
        batch_invariant=batch_invariant,
        knobs=knobs,
    )
    packed_weights, plain = _plain_quantized_weights(weights, intermediate)
    packed, scales = nvfp4_quantize_per_block_16(x[:n].float(), config.input_norm_const)
    expected = _torch_reference(
        packed, scales, ids[:n], scores[:n], plain, config, hidden, intermediate, early
    )
    with _layer(config, packed_weights, 0, 1, n, hidden, quantize=False) as layer:
        actual = layer.forward(MoEEpTensors(packed, ids[:n], scores[:n], scales=scales))
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-6)
        # Reuse the established NVFP4 oracle tolerance: FC1 requantization may
        # flip a code at an RTNE boundary, and each FC2 term is rounded to BF16.
        # This is an accuracy check, never evidence of bit-exact BI by itself.
        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=0.05,
            atol=_oracle_atol(expected.float().abs().max().item()),
        )
        assert error.item() < 0.02


@pytest.mark.gpu_2
@pytest.mark.parametrize("provider", ("legacy", "standard", "genphase"))
@pytest.mark.parametrize("early", (False, True))
def test_nvfp4_batch_invariant_expert_parallel(provider, early, request, _ieee_matmul):
    from flashinfer import moe_ep
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    world, rank = (
        int(os.environ.get("WORLD_SIZE", "1")),
        int(os.environ.get("RANK", "0")),
    )
    if world < 2:
        pytest.skip("requires torchrun with >=2 ranks")
    bootstrap = moe_ep.BootstrapConfig(rank=rank, world_size=world)
    moe_ep.ensure_moe_ep_cuda_device(bootstrap)
    _device(provider)
    config = _config(provider, early, 384, 192)
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        _distributed_oracle(provider, early, rank, world)
        _run(provider, early, rank, world, request.config.getoption("--full"))
    except BaseException as error:
        from ._megamoe_bi_helpers import fail_stop_distributed

        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)
