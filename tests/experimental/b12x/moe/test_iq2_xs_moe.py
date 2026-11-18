"""Prepared IQ2_XS execution, mapped routes and graph reuse on CUDA."""

from pathlib import Path

import pytest

from b12x.moe.fused_moe import FC2Invocation, IQ2XSWeights, MoeDecodeConfig, plan_fc2
from b12x.preparation import PreparationSession
from benchmarks.experimental.b12x.iq2_xs_checkpoint import IQ2XSLayer
from benchmarks.experimental.b12x.benchmark_iq2_xs_moe import (
    prepare_experts,
    qualify_capacity,
)
from b12x.testing.reference.helpers import require_b12x
from .test_iq2_xs import blocks


@pytest.mark.parametrize("hidden_size,intermediate_size", [(256, 256), (2048, 512)])
@pytest.mark.parametrize(
    "activation,mapped,route,capacity,counts",
    [
        ("silu", False, "packed", 16, (1, 3, 8, 16)),
        ("silu", True, "packed", 65, (9, 17, 33, 65)),
        ("relu2", True, "packed", 65, (9, 17, 33, 65)),
        ("silu", True, "packed", 16, (1, 3, 8, 16)),
        ("silu", True, "direct", 8, (8,)),
        ("relu2", False, "packed", 16, (1, 3, 8, 16)),
        ("relu2", True, "packed", 16, (1, 3, 8, 16)),
        ("relu2", False, "direct", 6, (1, 3, 6)),
        ("relu2", True, "direct", 6, (1, 3, 6)),
        ("relu2", False, "direct", 8, (1, 3, 8)),
        ("relu2", True, "direct", 8, (1, 3, 8)),
    ],
)
@pytest.mark.parametrize("codec", ["iq2_xs", "iq2_xxs", "q8_0"])
def test_prepared_execution(
    activation, mapped, route, capacity, counts, hidden_size, intermediate_size, codec
):
    device = require_b12x()
    ids = (7, 0, 5) if mapped else tuple(range(8))
    w13 = blocks(
        e=len(ids),
        n=intermediate_size * (2 if activation == "silu" else 1),
        k=hidden_size,
        codec=codec,
    )
    w2 = blocks(e=len(ids), n=hidden_size, k=intermediate_size, codec=codec)
    layer = IQ2XSLayer(
        IQ2XSWeights(w13, w2, codec=codec),
        hidden_size,
        intermediate_size,
        8,
        2,
        ids,
        Path("synthetic-iq2-xs"),
        0,
        1,
        0,
    )
    experts, _ = prepare_experts(layer, device, activation=activation)
    with pytest.raises(NotImplementedError, match="standalone IQ2_XS FC2"):
        plan_fc2(experts=experts, invocation=FC2Invocation(max_routes=capacity * 2))
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        results = qualify_capacity(
            layer,
            experts,
            session,
            capacity=capacity,
            counts=counts,
            route_mode=route,
            activation=activation,
            patterns=("balanced", "hot"),
            repeats=1,
            launches=1,
        )
    assert len(results) == len(counts) * 2


@pytest.mark.parametrize("codec", ("iq2_xs", "iq2_xxs", "q8_0"))
@pytest.mark.parametrize(
    "tiles,block,stages,route",
    (
        ((128, 64, 64, 128), 8, 2, "direct"),
        ((64, 128, 128, 64), 8, 3, "direct"),
        ((128, 128, 64, 256), 8, 4, "direct"),
        ((64, 256, 128, 128), 8, 4, "direct"),
        ((128, 64, 128, 64), 8, 5, "direct"),
        ((128, 64, 128, 64), 16, 5, "packed"),
        ((128, 128, 128, 128), 32, 3, "packed"),
        ((64, 128, 64, 128), 48, 2, "packed"),
        ((64, 256, 64, 256), 64, 3, "packed"),
    ),
)
def test_tuned_tiles_pipeline_and_route_blocks_replay(
    codec, tiles, block, stages, route
):
    device = require_b12x()
    # Keep Super3's real projection geometry, with fewer resident experts for
    # the oracle. The mapped routes include absent experts and hot/balanced loads.
    h, i = 1024, 2048
    ids = (7, 0, 5)
    layer = IQ2XSLayer(
        IQ2XSWeights(
            blocks(e=3, n=i, k=h, codec=codec),
            blocks(e=3, n=h, k=i, codec=codec),
            codec=codec,
        ),
        h,
        i,
        8,
        2,
        ids,
        Path("synthetic-super3"),
        0,
        1,
        0,
        "relu2",
    )
    experts, _ = prepare_experts(layer, device, activation="relu2")
    capacity = 8 if route == "direct" else 65
    counts = (1, 3, 8) if route == "direct" else (1, 17, 65)
    config = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode=route,
        w4a16_tile_config=tiles,
        w4a16_block_size_m=block,
        w4a16_pipeline_stages=stages,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        results = qualify_capacity(
            layer,
            experts,
            session,
            capacity=capacity,
            counts=counts,
            route_mode=route,
            activation="relu2",
            config=config,
            patterns=("balanced", "hot"),
            repeats=1,
            launches=1,
        )
    assert len(results) == len(counts) * 2
    assert all(row["config"] == config.to_dict() for row in results)


@pytest.mark.parametrize("codec", ("iq2_xs", "iq2_xxs", "q8_0"))
@pytest.mark.parametrize("stages", (2, 3, 4))
@pytest.mark.parametrize("block", (8, 16, 32))
def test_tuned_wide_n_single_token_reduction_scratch(codec, stages, block):
    device = require_b12x()
    h, i, e, topk = 1024, 2048, 16, 9
    layer = IQ2XSLayer(
        IQ2XSWeights(
            blocks(e=e, n=i, k=h, codec=codec),
            blocks(e=e, n=h, k=i, codec=codec),
            codec=codec,
        ),
        h,
        i,
        e,
        topk,
        tuple(range(e)),
        Path("synthetic-super3"),
        0,
        1,
        0,
        "relu2",
    )
    experts, _ = prepare_experts(layer, device, activation="relu2")
    route = "direct" if block == 8 else "packed"
    config = MoeDecodeConfig(
        backend="w4a16",
        route_planner="internal",
        max_active_clusters=None,
        w4a16_route_mode=route,
        w4a16_tile_config=(64, 256, 64, 256),
        w4a16_block_size_m=block,
        w4a16_pipeline_stages=stages,
    )
    with PreparationSession(
        device=device, autotune=False, compile_workers=1
    ) as session:
        qualify_capacity(
            layer,
            experts,
            session,
            capacity=1,
            counts=(1,),
            route_mode=route,
            activation="relu2",
            config=config,
            patterns=("balanced", "hot"),
            repeats=1,
            launches=1,
        )
