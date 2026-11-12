"""Split-only selection does not require a Mega backend or Rubin hardware."""

import dataclasses
import os

import pytest
import torch

from flashinfer import autotune_v2, moe_ep
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe.api import TrtllmBf16Config
from flashinfer.fused_moe.layer import MoELayer
from flashinfer.moe_ep.core.selection import Collective
from tests.moe_ep import test_split_fused_moe_kernel_vs_reference as problem


@pytest.mark.arch_blackwell
def test_split_only_search_replay_and_capture(monkeypatch, tmp_path):
    if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("EP runtime wheels require CUDA 13+")
    problem._require_backend(TrtllmBf16Config)
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        pytest.skip("single-GPU test")
    if not moe_ep.have_nccl_ep():
        pytest.skip("requires NCCL-EP")
    for name, value in dict(
        NUM_EXPERTS=8, TOP_K=2, NUM_TOKENS=8, HIDDEN=4096, INTERMEDIATE=256
    ).items():
        monkeypatch.setattr(problem, name, value)
    x, w13, w2, ids, scores = problem._make_problem()
    cfg = problem._build_moe_config("bf16")
    cfg = dataclasses.replace(
        cfg, execution=dataclasses.replace(cfg.execution, tune_max_num_tokens=512)
    )
    config = moe_ep.AutoConfig(
        (moe_ep.SplitConfig(kernel=moe_ep.FusedMoeKernelConfig(cfg)),)
    )
    fleet = moe_ep.FleetParams(8, 16, 4096, layout=moe_ep.EpLayout.RANK_MAJOR)
    tensors = moe_ep.MoEEpTensors(x, ids, scores)
    expected = problem._dense_moe_reference(
        x.float(),
        w13.float(),
        w2.float(),
        ids,
        scores,
        act_roundtrip=lambda a: a.bfloat16().float(),
    )
    monkeypatch.setattr(
        MoELayer, "_select_winner", lambda *a, **k: pytest.fail("compute-only tuning")
    )
    layers, graph = [], None
    options = dict(cache_root=str(tmp_path / "managed"))
    try:
        layer = moe_ep.MoEEpLayer(
            moe_ep.BootstrapConfig(world_size=1, rank=0),
            fleet,
            moe_ep.MoEWeightPack(w13, w2),
            backend=config,
        )
        layers.append(layer)
        with autotune_v2(**options):
            layer.warmup(tensors)
        chosen = layer.get_selected_plan()
        assert chosen["candidate"] == 0
        assert chosen["selected"][0] == 0
        torch.testing.assert_close(
            layer(tensors).float(), expected, rtol=3e-2, atol=3e-2
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = layer(tensors)
        captured.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured.float(), expected, rtol=3e-2, atol=3e-2)
        AutoTuner.get().clear_cache()
        monkeypatch.setattr(
            Collective, "measure", lambda *a, **k: pytest.fail("replay measured")
        )
        with autotune_v2(mode="replay", **options):
            replay = moe_ep.MoEEpLayer(
                moe_ep.BootstrapConfig(world_size=1, rank=0),
                fleet,
                moe_ep.MoEWeightPack(w13, w2),
                backend=config,
            )
            layers.append(replay)
            replay.warmup(tensors)
            assert replay.get_selected_plan() == chosen
            torch.testing.assert_close(
                replay(tensors).float(), expected, rtol=3e-2, atol=3e-2
            )
    finally:
        if graph is not None:
            graph.reset()
        torch.cuda.synchronize()
        for layer in reversed(layers):
            layer.destroy()
