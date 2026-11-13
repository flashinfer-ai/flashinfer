# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Native MXFP4 BI checks, with accuracy separate from bitwise comparisons."""

from dataclasses import replace
import os

import pytest

from .test_nvfp4_megamoe_batch_invariant import (
    _assert_bits,
    _gather_packed_rows,
    _ieee_matmul as _ieee_precision_fixture,
    _layer,
    _problem,
)

pytestmark = pytest.mark.gpu
_ieee_matmul = _ieee_precision_fixture
KINDS = ("mxfp4_mxfp8", "mxfp4")
PROVIDERS = ("standard", "genphase")
HIDDEN = INTERMEDIATE = 384


def _device(kind):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cc = torch.cuda.get_device_capability()
    if cc not in ((10, 0), (10, 3), (10, 7)):
        pytest.skip("MXFP4 BI requires SM100/SM103/SM107")
    if kind == "mxfp4" and cc != (10, 7):
        pytest.skip("pure MXFP4 MegaMoE requires SM107")
    return cc


def _config(
    kind,
    provider,
    early,
    *,
    knobs=None,
    bi=True,
    persistent=False,
    intermediate=INTERMEDIATE,
    top_k=3,
):
    import torch
    from flashinfer import moe_ep

    common = dict(
        intermediate_size=intermediate,
        top_k=top_k,
        batch_invariant=bi,
        knobs=knobs,
        gate_up_clamp=2.5,
        apply_routing_weights_before_fc2=early,
    )
    if torch.cuda.get_device_capability() == (10, 7):
        assert not persistent
        cls = (
            moe_ep.Sm107_Mxfp4_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
            if kind == "mxfp4"
            else moe_ep.Sm107_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig
        )
        return cls(
            **common,
            kernel_variant="genphase" if provider == "genphase" else "inference",
            cluster_shape_mn=(4, 1) if provider == "genphase" else (2, 1),
            fc2_use_bulk=provider == "genphase",
        )
    common.update(
        finalize_mode=(
            moe_ep.FinalizeMode.PERSISTENT
            if persistent
            else moe_ep.FinalizeMode.DEFAULT
        ),
        num_valid_tokens_tensor=(
            torch.zeros(1, device="cuda", dtype=torch.int32) if persistent else None
        ),
    )
    if provider == "genphase":
        return moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            **common, kind=kind, kernel_variant="genphase"
        )
    return moe_ep.Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(**common)


def _plans(provider, cc):
    if provider == "genphase":
        return (
            [{"work_id_mode": mode} for mode in ("grid_stride", "atomic_counter")]
            if cc == (10, 7)
            else [{}]
        )
    return [
        dict(schedule_policy=schedule, work_id_mode=mode, token_back_mode=carrier)
        for schedule, mode in (
            (("grouped", None), "grid_stride"),
            (("grouped", None), "atomic_counter"),
            (("phase_interleave", None), "atomic_counter"),
        )
        for carrier in ("epi_warps", "standalone_warps", "reuse_dispatch_warps")
    ]


def _run(kind, provider, early, rank, world, full):
    import torch
    from flashinfer.batch_invariant import fingerprint
    from flashinfer.moe_ep import MoEEpTensors

    cc = _device(kind)
    x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
    config = _config(kind, provider, early)
    receipt = fingerprint(
        "megamoe_mxfp4",
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
        top_k=3,
        num_experts=4 * world,
        arch=f"sm_{cc[0]}{cc[1]}",
        quant_kind=kind,
        implementation=provider,
        apply_routing_weights_before_fc2=early,
        gate_up_clamp=2.5,
    )
    alternate_ids = ids.clone()
    alternate_ids[:, 0] = 4 * rank + 3
    with _layer(config, weights, rank, world, 257, HIDDEN) as layer:
        anchor = layer.forward(MoEEpTensors(x, ids, scores)).clone()
        alternate = layer.forward(MoEEpTensors(x, alternate_ids, scores)).clone()
        torch.cuda.synchronize()
        assert layer.batch_invariant_fingerprint == receipt
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        assert torch.count_nonzero(anchor[19]) == 0
    for knobs in _plans(provider, cc):
        for capacity in (1, 129, 257) if full else (257,):
            with _layer(
                replace(config, knobs=knobs), weights, rank, world, capacity, HIDDEN
            ) as layer:
                for n in (0, 1, 4, 8, 16, 17, 31, 32, 63, 64, 65, 127, 128, 129, 257):
                    if n > capacity:
                        continue
                    order = (torch.arange(n, device="cuda") * 37) % 257
                    for order in (order, order.flip(0), order.remainder(17)):
                        inputs = MoEEpTensors(x[order], ids[order], scores[order])
                        for _ in range(2):
                            _assert_bits(
                                layer.forward(inputs),
                                anchor[order],
                                f"{kind}/{provider}/{knobs}/T={n}",
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
                            f"{kind}/{provider}/{knobs}/uneven T={n}",
                        )
                if capacity >= 4:
                    order = torch.tensor([19, 20, 1, 53], device="cuda")
                    output = torch.empty_like(anchor[order])
                    inputs = MoEEpTensors(
                        x[order].clone(),
                        ids[order].clone(),
                        scores[order].clone(),
                        output=output,
                    )
                    layer.forward(inputs)
                    graph = torch.cuda.CUDAGraph()
                    try:
                        with torch.cuda.graph(graph):
                            layer.forward(inputs)
                        for changed, masked in (
                            (True, False),
                            (False, True),
                            (True, True),
                            (False, False),
                        ):
                            source_ids = alternate_ids if changed else ids
                            expected = alternate if changed else anchor
                            # The masked slots must contain nonzero old results.
                            # Their reference rows have zero scores, so merely
                            # replaying those rows would not catch a stale read
                            # in the early-routing path.
                            prime = order.roll(2)
                            inputs.hidden_states.copy_(x[prime])
                            inputs.topk_ids.copy_(source_ids[prime])
                            inputs.topk_weights.copy_(scores[prime])
                            graph.replay()
                            _assert_bits(output, expected[prime], "nonzero graph prime")
                            inputs.hidden_states.copy_(x[order])
                            inputs.topk_ids.copy_(source_ids[order])
                            inputs.topk_weights.copy_(scores[order])
                            if masked:
                                inputs.topk_ids[0].fill_(-1)
                                inputs.topk_weights[0].fill_(float("nan"))
                                inputs.topk_ids[1, 1] = -1
                                inputs.topk_weights[1, 1] = float("nan")
                            graph.replay()
                            _assert_bits(
                                output, expected[order], "changing/masked routes graph"
                            )
                    finally:
                        graph.reset()
                assert layer.batch_invariant_fingerprint == receipt


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
def test_mxfp4_batch_invariant_autotune_disk_reload(
    kind, provider, early, monkeypatch, tmp_path
):
    from ._megamoe_bi_helpers import autotune_disk_reload

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank autotune test")
    _device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    autotune_disk_reload(
        make_config=lambda knobs: _config(kind, provider, early, knobs=knobs),
        weights=weights,
        x=x,
        ids=ids,
        scores=scores,
        layer_factory=_layer,
        provider=provider,
        directory=tmp_path,
        monkeypatch=monkeypatch,
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize(
    "shape,top_k", (((2048, 2048), 3), ((128, 128), 1), ((128, 128), 32))
)
def test_mxfp4_batch_invariant_long_k_and_topk_boundaries(
    kind, provider, early, shape, top_k, monkeypatch
):
    from ._megamoe_bi_helpers import boundary_batches

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank boundary test")
    cc = _device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    x, scores, ids, weights = _problem(
        0, 1, *shape, top_k=top_k, local_experts=max(4, top_k)
    )
    scores.mul_(max(1, top_k / 3))
    scores[:, 0].neg_()
    config = _config(kind, provider, early, intermediate=shape[1], top_k=top_k)
    boundary_batches(
        config, weights, x, ids, scores, _layer, ({}, _plans(provider, cc)[-1])
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
def test_mxfp4_batch_invariant_batches_plans_and_graphs(
    kind, provider, early, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    monkeypatch.setenv("FLASHINFER_MEGA_FUSED_STAGE", "0")
    _run(kind, provider, early, 0, 1, request.config.getoption("--full"))


@pytest.mark.parametrize("early", (False, True))
def test_mxfp4_batch_invariant_persistent_live_count(early, monkeypatch):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    if _device("mxfp4_mxfp8") == (10, 7):
        pytest.skip("SM107 does not expose persistent finalization")
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    config = _config("mxfp4_mxfp8", "standard", early)
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    with _layer(config, weights, 0, 1, 129, HIDDEN) as layer:
        anchor = layer.forward(MoEEpTensors(x[:129], ids[:129], scores[:129])).clone()
        receipt = layer.batch_invariant_fingerprint
    config = _config("mxfp4_mxfp8", "standard", early, persistent=True)
    with _layer(config, weights, 0, 1, 129, HIDDEN) as layer:
        inputs = MoEEpTensors(x[:129].clone(), ids[:129].clone(), scores[:129].clone())
        config.num_valid_tokens_tensor.fill_(129)
        layer.forward(inputs, return_workspace_view=True)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                output = layer.forward(inputs, return_workspace_view=True)
            sentinel = torch.full_like(output, 123.0)
            for n in (0, 1, 31, 32, 64, 129, 1, 0, 129):
                output.copy_(sentinel)
                config.num_valid_tokens_tensor.fill_(n)
                graph.replay()
                expected = sentinel.clone()
                expected[:n].copy_(anchor[:n])
                _assert_bits(output, expected, f"persistent live={n}")
        finally:
            graph.reset()
        assert layer.batch_invariant_fingerprint == receipt


def _quantize(tensor, kind, *, weight=False):
    import torch
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        quantize_mxfp4_block32,
        quantize_mxfp8_block32,
    )

    return (
        quantize_mxfp4_block32(tensor.float())
        if kind == "mxfp4" or weight
        else quantize_mxfp8_block32(tensor.float(), torch.float8_e4m3fn)
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
def test_mxfp4_batch_invariant_prequantized_batches(kind, provider, early, monkeypatch):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    _device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    q, sf = _quantize(x, kind)
    config = _config(kind, provider, early)
    with _layer(config, weights, 0, 1, 257, HIDDEN, quantize=False) as layer:
        anchor = layer.forward(MoEEpTensors(q, ids, scores, scales=sf)).clone()
        assert torch.isfinite(anchor).all() and torch.count_nonzero(anchor) > 0
        receipt = layer.batch_invariant_fingerprint
    with _layer(config, weights, 0, 1, 129, HIDDEN, quantize=False) as layer:
        for n in (0, 1, 8, 16, 17, 65, 129):
            order = ((torch.arange(n, device="cuda") * 37) % 257).flip(0)
            data, scales = _gather_packed_rows(q, sf, order)
            _assert_bits(
                layer.forward(
                    MoEEpTensors(data, ids[order], scores[order], scales=scales)
                ),
                anchor[order],
                f"prequantized T={n}",
            )
        assert layer.batch_invariant_fingerprint == receipt


def _torch_reference(q, sf, ids, scores, packed, kind, early):
    """Canonical prepacked operands; no production weight transform or FC kernels."""
    import torch
    from flashinfer.moe_ep.kernel_src.sm107.next_cutedsl_megamoe import (
        scale_to_f32,
        unpack_fp4_to_f32,
    )

    def dequant(data, scale, *, weight=False):
        values = unpack_fp4_to_f32(data) if kind == "mxfp4" or weight else data.float()
        return values * scale_to_f32(scale).repeat_interleave(32, dim=-1)

    x = dequant(q, sf)
    acc = torch.zeros_like(x)
    for slot in range(ids.shape[1]):
        wire = torch.zeros_like(x, dtype=torch.bfloat16)
        for expert in range(packed.w13.shape[0]):
            rows = (ids[:, slot] == expert).nonzero(as_tuple=True)[0]
            if not rows.numel():
                continue
            w13 = dequant(packed.w13[expert], packed.w13_scale[expert], weight=True)
            w2 = dequant(packed.w2[expert], packed.w2_scale[expert], weight=True)
            gate, up = (x[rows] @ w13.T).chunk(2, dim=-1)
            gate, up = gate.clamp(max=2.5), up.clamp(-2.5, 2.5)
            act = (up * gate) * torch.sigmoid(gate)
            if early:
                act *= scores[rows, slot, None]
            aq, asf = _quantize(act, kind)
            wire[rows] = (dequant(aq, asf) @ w2.T).bfloat16()
        if early:
            acc = acc + wire.float()
        else:
            weight = torch.where(ids[:, slot] >= 0, scores[:, slot], 0.0)
            acc = (acc.double() + wire.double() * weight[:, None].double()).float()
    return acc.bfloat16()


def _distributed_oracle(kind, provider, early, rank, world):
    from flashinfer.moe_ep import MoEWeightPack
    from ._megamoe_bi_helpers import (
        all_gather_rows,
        distributed_reference,
        remote_routes,
    )

    x, scores, ids, weights = _problem(rank, world, HIDDEN, INTERMEDIATE)
    q1, sf1 = _quantize(weights.w13, kind, weight=True)
    q2, sf2 = _quantize(weights.w2, kind, weight=True)
    packed_weights = MoEWeightPack(w13=q1, w2=q2, w13_scale=sf1, w2_scale=sf2)
    global_weights = MoEWeightPack(
        w13=all_gather_rows(q1),
        w2=all_gather_rows(q2),
        w13_scale=all_gather_rows(sf1),
        w2_scale=all_gather_rows(sf2),
    )
    q, sf = _quantize(x[:31], kind)
    ids = remote_routes(ids[:31], rank, world, weights.w13.shape[0])

    def reference(q, sf, route_ids, route_scores):
        return _torch_reference(
            q, sf, route_ids, route_scores, global_weights, kind, early
        )

    distributed_reference(
        config=_config(kind, provider, early),
        weights=packed_weights,
        packed=q,
        scales=sf,
        ids=ids,
        scores=scores[:31],
        reference=reference,
        layer_factory=_layer,
        hidden=HIDDEN,
        rank=rank,
        world=world,
        tolerance=0.06 if kind == "mxfp4" else 0.02,
    )


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
@pytest.mark.parametrize("bi", (False, True))
def test_mxfp4_batch_invariant_independent_reference(
    kind, provider, early, bi, monkeypatch, _ieee_matmul
):
    import torch
    from flashinfer.moe_ep import MoEEpTensors, MoEWeightPack
    from flashinfer.moe_ep.backends.mega.kernel.sm100.common.mxfp4_batch_invariant import (
        mxfp4_batch_invariant_default_knobs,
    )

    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank test")
    cc = _device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    x, scores, ids, weights = _problem(0, 1, HIDDEN, INTERMEDIATE)
    q1, sf1 = _quantize(weights.w13, kind, weight=True)
    q2, sf2 = _quantize(weights.w2, kind, weight=True)
    packed = MoEWeightPack(w13=q1, w2=q2, w13_scale=sf1, w2_scale=sf2)
    q, sf = _quantize(x[:33], kind)
    expected = _torch_reference(q, sf, ids[:33], scores[:33], packed, kind, early)
    knobs = mxfp4_batch_invariant_default_knobs(
        provider, arch=f"sm_{cc[0]}{cc[1]}", quant_kind=kind
    )
    if provider == "genphase" and cc != (10, 7):
        knobs = {
            name: knobs[name]
            for name in ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
        }
    config = _config(kind, provider, early, knobs=knobs, bi=bi)
    with _layer(config, packed, 0, 1, 33, HIDDEN, quantize=False) as layer:
        actual = layer.forward(MoEEpTensors(q, ids[:33], scores[:33], scales=sf))
        assert torch.isfinite(actual).all() and torch.count_nonzero(actual) > 0
        error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm().clamp_min(1e-6)
        assert error.item() < (0.06 if kind == "mxfp4" else 0.02)


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("provider", PROVIDERS)
@pytest.mark.parametrize("early", (False, True))
def test_mxfp4_batch_invariant_expert_parallel(
    kind, provider, early, request, _ieee_matmul
):
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
    _device(kind)
    config = _config(kind, provider, early)
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    try:
        _distributed_oracle(kind, provider, early, rank, world)
        _run(kind, provider, early, rank, world, request.config.getoption("--full"))
    except BaseException as error:
        from ._megamoe_bi_helpers import fail_stop_distributed

        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)
