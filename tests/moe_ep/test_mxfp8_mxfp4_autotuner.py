# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host tests of the real general-autotuner selection/cache control flow."""

import importlib
import json
import datetime
import multiprocessing
from types import SimpleNamespace

import pytest
import torch

from flashinfer.autotuner import AutoTuner, autotune, get_autotune_process_group
from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
    resolve_tactic,
)
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4 import integration
from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.autotune import (
    CUSTOM_OP,
    Mxfp8Mxfp4MegaMoeRunner,
    select_tactic,
)


pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("cutlass") is None, reason="requires CuTe DSL helpers"
)


@pytest.fixture
def host_case(monkeypatch, tmp_path):
    return _host_case(monkeypatch, tmp_path)


def _host_case(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda: SimpleNamespace(cuda_stream=1)
    )
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    core = importlib.import_module("flashinfer.autotuner.autotuner")
    monkeypatch.setattr(core, "_collect_metadata", lambda: {"test": "mxfp8_mxfp4"})
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    resolve = lambda knobs: dict(
        resolve_tactic(16, knobs, hidden=256, intermediate=256)
    )
    ws = SimpleNamespace(
        rank=0,
        world_size=1,
        num_experts=4,
        num_topk=2,
        hidden=256,
        intermediate=256,
        max_tokens_per_rank=16,
        tactic=resolve({}),
        resolve=resolve,
        fits=lambda _: True,
        workspace_bytes=torch.zeros(4096, dtype=torch.uint8),
        shared_workspace_bytes=torch.zeros(4096, dtype=torch.uint8),
        x=torch.zeros(16, 256),
        x_sf=torch.zeros(16, 8),
        topk_idx=torch.zeros(16, 2, dtype=torch.int64),
        topk_weights=torch.zeros(16, 2),
        y=torch.zeros(16, 256),
        route_terms=torch.zeros(16, 2, 256),
        workspace_regions={},
        _autotune_selected=False,
    )
    prepared = []

    def apply(knobs):
        ws.tactic = resolve(knobs)

    def frontend(**kwargs):
        return SimpleNamespace(
            config=SimpleNamespace(tactic=tuple(sorted(ws.tactic.items()))),
            warmup=lambda spec: prepared.append((dict(ws.tactic), spec)),
        )

    ws.apply_knobs, ws.frontend = apply, frontend
    fc1, fc2 = (
        (torch.zeros(4, 128, 512), torch.zeros(4, 4096)),
        (torch.zeros(4, 128, 256), torch.zeros(4, 2048)),
    )
    candidates = [{"group_hint": 512}, {"group_hint": 4096}]
    profiled = []

    def profile(runner, inputs, tactic, config, **kwargs):
        tuner._prepare_tactic(runner, inputs, tactic)
        profiled.append(tactic)
        assert config.preserve_input_storage
        assert config.rank_reduction == "max"
        # The first tactic wins; the final bind must restore it after profiling
        # the slower second tactic.
        return 1.0 if ws.tactic["group_hint"] == 512 else 2.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    return SimpleNamespace(
        ws=ws,
        fc1=fc1,
        fc2=fc2,
        tuner=tuner,
        candidates=candidates,
        prepared=prepared,
        profiled=profiled,
        tmp=tmp_path,
    )


def tune(case, **kwargs):
    return integration.autotune_mxfp8_mxfp4_mega_moe(
        None,
        case.fc1,
        case.fc2,
        case.ws,
        num_tokens=7,
        candidates=case.candidates,
        **kwargs,
    )


def test_shared_search_applies_winner_and_reuses_cache(host_case):
    c = host_case
    winner = tune(c, warmup_iters=0, timed_iters=2)
    assert winner["group_hint"] == 512
    assert c.ws.tactic == winner and c.ws._autotune_selected
    assert len(c.profiled) == 2
    assert c.prepared[-1][0] == winner
    assert all(spec.output is c.ws.y for _, spec in c.prepared)
    assert get_autotune_process_group() is None
    assert not c.tuner.is_tuning_mode
    tune(c)
    assert len(c.profiled) == 2
    entry = json.loads((c.tmp / "knobs.json").read_text())["entries"][0]
    assert entry["source"] == "flashinfer.autotuner"
    assert entry["p50_us"] is None


def test_general_file_cache_roundtrip(host_case, monkeypatch):
    c = host_case
    winner = tune(c)
    path = str(c.tmp / "general.json")
    c.tuner.save_configs(path)
    c.tuner.clear_cache()
    assert c.tuner.load_configs(path)
    c.ws.apply_knobs({"group_hint": 4096})
    assert tune(c) == winner
    assert len(c.profiled) == 2


def test_managed_cache_default_winner_and_storage_policy(host_case, monkeypatch):
    from flashinfer.autotune_cache import autotune_v2, MeasurementPolicy

    c = host_case
    default = c.ws.resolve({})
    # The general v2 tuner can choose the default (-1),
    # which races before the explicit tactics. It must survive a workspace
    # tactic change and the compatibility-cache export.
    profile = c.tuner._profile_single_kernel

    def default_wins(runner, inputs, tactic, config, **kwargs):
        elapsed = profile(runner, inputs, tactic, config, **kwargs)
        return 0.5 if tactic == -1 else elapsed

    monkeypatch.setattr(c.tuner, "_profile_single_kernel", default_wins)
    with autotune_v2(
        persistent_cache=False, measurement_policy=MeasurementPolicy(cold_l2=True)
    ):
        winner = tune(c)
        assert winner == default
        c.ws.apply_knobs({"group_hint": 4096})
        assert tune(c) == default
    assert len(c.profiled) == 3
    assert c.profiled[0] == -1


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_tokens", 3),
        ("gate_up_clamp", 2.0),
        ("fast_math", False),
    ],
)
def test_workload_cache_identity(host_case, field, value):
    c = host_case
    first = Mxfp8Mxfp4MegaMoeRunner(c.ws, num_tokens=7, candidates=c.candidates)
    key = first.get_cache_key_extras([])
    kwargs = {"num_tokens": 7, "candidates": c.candidates}
    kwargs[field] = value
    second = Mxfp8Mxfp4MegaMoeRunner(c.ws, **kwargs)
    assert second.get_cache_key_extras([]) != key


def test_capacity_filter_and_candidate_order_roundtrip(host_case):
    c = host_case
    c.ws.fits = lambda knobs: knobs["group_hint"] == 512
    with pytest.warns(RuntimeWarning, match="dropped 1"):
        runner = Mxfp8Mxfp4MegaMoeRunner(c.ws, num_tokens=7, candidates=c.candidates)
    assert len(runner.tactics) == 1
    from flashinfer.autotuner import _json_to_tactic, _tactic_to_json

    tactic = _json_to_tactic(json.loads(json.dumps(_tactic_to_json(runner.tactics[0]))))
    assert runner.validate_tactic([], tactic)
    assert hash(runner) == hash(
        Mxfp8Mxfp4MegaMoeRunner(c.ws, num_tokens=7, candidates=c.candidates[:1])
    )


def test_lookup_preserves_legacy_tactic_and_skip_does_not_profile(host_case):
    c = host_case
    c.ws.apply_knobs({"group_hint": 4096})
    assert (
        select_tactic(c.fc1, c.fc2, c.ws, num_tokens=7, candidates=c.candidates)[
            "group_hint"
        ]
        == 4096
    )
    with autotune(skip_ops={CUSTOM_OP}):
        assert tune(c)["group_hint"] == 4096
    assert not c.profiled
    assert not (c.tmp / "knobs.json").exists()


def test_failure_restores_context_and_does_not_record_winner(host_case, monkeypatch):
    c = host_case
    monkeypatch.setattr(c.tuner, "_profile_single_kernel", lambda *a, **k: float("inf"))
    with pytest.raises(RuntimeError, match="every candidate failed"):
        tune(c)
    assert not c.ws._autotune_selected
    assert not c.tuner.is_tuning_mode
    assert get_autotune_process_group() is None
    assert not (c.tmp / "knobs.json").exists()


def test_disable_legacy_export_keeps_general_cache(host_case, monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
    tune(host_case)
    assert host_case.tuner.profiling_cache
    assert not (host_case.tmp / "knobs.json").exists()


def test_backend_tunes_each_workspace_and_pinned_knobs_bypass(host_case, monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.backend import (
        Mxfp8Mxfp4CutedslMegaKernelBackend,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm100.mxfp8_mxfp4_bf16_cutedsl.config import (
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig,
    )

    c = host_case
    calls = []

    def select(output, fc1, fc2, ws, **kwargs):
        calls.append(ws)
        ws._autotune_selected = True

    monkeypatch.setattr(integration, "autotune_mxfp8_mxfp4_mega_moe", select)
    monkeypatch.setattr(integration, "mxfp8_mxfp4_mega_moe", lambda *a, **k: c.ws.y[:7])
    backend = Mxfp8Mxfp4CutedslMegaKernelBackend(
        Sm100_Mxfp8_Mxfp4_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=256, top_k=2, knobs="auto"
        )
    )
    second = SimpleNamespace(**vars(c.ws))
    for ws in (c.ws, c.ws, second):
        backend.compute(ws, (c.fc1, c.fc2), output=c.ws.y[:7])
    assert len(calls) == 2 and calls[0] is c.ws and calls[1] is second
    backend._kernel_config.knobs = {}
    second._autotune_selected = False
    with autotune():
        backend.compute(second, (c.fc1, c.fc2), output=c.ws.y[:7])
    assert len(calls) == 2


def _collective_selection_rank(rank, root):
    from pathlib import Path
    import torch.distributed as dist

    root = Path(root)
    with pytest.MonkeyPatch.context() as monkeypatch:
        c = _host_case(monkeypatch, root)
        dist.init_process_group(
            "gloo",
            init_method=f"file://{root / 'rdzv'}",
            rank=rank,
            world_size=2,
            timeout=datetime.timedelta(seconds=30),
        )
        c.ws.rank, c.ws.world_size = rank, 2
        try:
            with autotune():
                select_tactic(
                    c.fc1, c.fc2, c.ws, num_tokens=7 - rank, candidates=c.candidates
                )
            assert len(c.profiled) == 2
            key = next(iter(c.tuner.profiling_cache))
            assert (7, 6) in key.extras

            # One rank's cache miss must be rejected before another rank can
            # skip the profiling launches. Both ranks see the same exception.
            if rank == 1:
                c.tuner.clear_cache()
            with autotune(), pytest.raises(RuntimeError, match="ranks disagree"):
                select_tactic(
                    c.fc1, c.fc2, c.ws, num_tokens=7 - rank, candidates=c.candidates
                )
            assert len(c.profiled) == 2

            c.tuner.clear_cache()
            candidates = c.candidates[:1] if rank == 0 else c.candidates
            with autotune(), pytest.raises(RuntimeError, match="ranks disagree"):
                select_tactic(
                    c.fc1, c.fc2, c.ws, num_tokens=7 - rank, candidates=candidates
                )
            assert len(c.profiled) == 2
        finally:
            dist.destroy_process_group()


def test_collective_workload_identity_and_mismatched_cache(tmp_path):
    ctx = multiprocessing.get_context("spawn")
    processes = [
        ctx.Process(target=_collective_selection_rank, args=(rank, str(tmp_path)))
        for rank in range(2)
    ]
    try:
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=50)
        assert [process.exitcode for process in processes] == [0, 0]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
