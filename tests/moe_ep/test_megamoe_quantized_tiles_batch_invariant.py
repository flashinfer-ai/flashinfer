# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""K-stage variants preserve the canonical native-MMA instruction sequence."""

from dataclasses import replace
import os

import pytest

from .test_mxfp4_megamoe_batch_invariant import _config as _mx4_config
from .test_mxfp4_megamoe_batch_invariant import _device as _mx4_device
from .test_mxfp4_megamoe_batch_invariant import _plans
from .test_mxfp8_megamoe_batch_invariant import _config as _mx8_config
from .test_nvfp4_megamoe_batch_invariant import (
    _assert_bits,
    _config as _nv4_config,
    _layer,
    _problem,
)

pytestmark = pytest.mark.gpu

CASES = (
    ("nvfp4", "aligned"),
    ("nvfp4", "padded"),
    ("mxfp4_mxfp8", "aligned"),
    ("mxfp4", "aligned"),
    ("mxfp4", "padded"),
    ("mxfp8_e4m3", "aligned"),
    ("mxfp8_e5m2", "aligned"),
)


def _shape_and_stage(kind, shape_kind, cc):
    rubin = cc == (10, 7)
    new_stage = 512 if rubin and kind in ("nvfp4", "mxfp4") else 256
    if shape_kind == "aligned":
        return (512, 512), new_stage
    if kind == "nvfp4":
        return ((384, 448) if rubin else (192, 448)), new_stage
    assert kind == "mxfp4" and rubin
    return (384, 384), new_stage


def _config(kind, early, hidden, intermediate):
    if kind == "nvfp4":
        return _nv4_config("standard", early, hidden, intermediate)
    if kind in ("mxfp4", "mxfp4_mxfp8"):
        return _mx4_config(kind, "standard", early, intermediate=intermediate)
    return _mx8_config(kind, "standard", early, intermediate=intermediate)


def _run_k_tiles(kind, shape_kind, early, rank, world, full):
    import torch
    from flashinfer.moe_ep import MoEEpTensors

    cc = _mx4_device(kind)
    (hidden, intermediate), stage = _shape_and_stage(kind, shape_kind, cc)
    x, scores, ids, weights = _problem(rank, world, hidden, intermediate)
    scores[:, 0].neg_()
    # Invalid slots never publish a wire cell. Poisoned scores make accidental
    # consumption visible, including after graph replay changes route identity.
    ids[19].fill_(-1)
    scores[19].fill_(float("nan"))
    ids[20, 1] = -1
    scores[20, 1] = float("nan")
    config = _config(kind, early, hidden, intermediate)
    with _layer(config, weights, rank, world, 257, hidden) as layer:
        anchor = layer.forward(MoEEpTensors(x, ids, scores)).clone()
        receipt = layer.batch_invariant_fingerprint
        assert receipt and torch.isfinite(anchor).all()
        assert torch.count_nonzero(anchor) > 0
        assert torch.count_nonzero(anchor[19]) == 0

    plans = _plans("standard", cc)
    if not full:
        plans = (plans[0], plans[-1])
    for plan in plans:
        knobs = {**plan, "mma_tiler_mnk": (256, 128, stage)}
        for capacity in (1, 33, 129) if full else (129,):
            with _layer(
                replace(config, knobs=knobs), weights, rank, world, capacity, hidden
            ) as layer:
                for n in (0, 1, 17, 32, 65, 129):
                    if n > capacity:
                        continue
                    order = (torch.arange(n, device="cuda") * 37) % 257
                    for selected in (order, order.flip(0), order.remainder(23)):
                        _assert_bits(
                            layer.forward(
                                MoEEpTensors(
                                    x[selected], ids[selected], scores[selected]
                                )
                            ),
                            anchor[selected],
                            f"{kind}/K={stage}/{knobs}/T={n}",
                        )
                        assert layer.batch_invariant_fingerprint == receipt
                if world > 1:
                    n = 0 if rank == 0 else min(capacity, 65)
                    _assert_bits(
                        layer.forward(MoEEpTensors(x[:n], ids[:n], scores[:n])),
                        anchor[:n],
                        f"{kind}/K={stage}/uneven T={n}",
                    )
                if capacity < 17:
                    continue
                order = torch.arange(17, 34, device="cuda")
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
                    for selected in (order.flip(0), order.remainder(5), order):
                        inputs.hidden_states.copy_(x[selected])
                        inputs.topk_ids.copy_(ids[selected])
                        inputs.topk_weights.copy_(scores[selected])
                        graph.replay()
                        _assert_bits(
                            output, anchor[selected], f"{kind}/K={stage}/graph"
                        )
                finally:
                    graph.reset()


@pytest.mark.parametrize("kind,shape_kind", CASES)
@pytest.mark.parametrize("early", (False, True))
def test_quantized_batch_invariant_across_k_stages(
    kind, shape_kind, early, monkeypatch, request
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-rank K-stage test")
    _mx4_device(kind)
    monkeypatch.setenv("MEGA_NO_DIST", "1")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    _run_k_tiles(kind, shape_kind, early, 0, 1, request.config.getoption("--full"))


@pytest.mark.gpu_2
@pytest.mark.parametrize("kind,shape_kind", CASES)
@pytest.mark.parametrize("early", (False, True))
def test_quantized_batch_invariant_k_stages_expert_parallel(
    kind, shape_kind, early, monkeypatch, request
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
    cc = _mx4_device(kind)
    (hidden, intermediate), _ = _shape_and_stage(kind, shape_kind, cc)
    config = _config(kind, early, hidden, intermediate)
    runtime = moe_ep.bootstrap_moe_ep_runtime(
        bootstrap, create_mega_kernel(config).runtime_requirements(bootstrap)
    )
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    try:
        _run_k_tiles(
            kind, shape_kind, early, rank, world, request.config.getoption("--full")
        )
    except BaseException as error:
        from ._megamoe_bi_helpers import fail_stop_distributed

        fail_stop_distributed(error, world)
        raise
    finally:
        moe_ep.finalize_moe_ep_runtime(runtime)
