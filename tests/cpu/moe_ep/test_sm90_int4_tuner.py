# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Host contracts for the INT4 candidate domain, cache and tuning lifecycle."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest


pytestmark = pytest.mark.cpu_only
_SHIM = "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim"
_SHAPE = dict(hidden=3072, intermediate=1024, num_experts=256, world_size=2)


@pytest.mark.parametrize("check", ["candidates", "cache", "auto_lifecycle"])
def test_int4_tuning_contract(check, tmp_path):
    # SM90 and SM100 kernel drops have mutually exclusive top-level modules.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, pytest; "
            "from tests.cpu.moe_ep import test_sm90_int4_tuner as t; "
            "getattr(t, '_check_' + sys.argv[1])(pytest.MonkeyPatch(), sys.argv[2])",
            check,
            str(tmp_path / "knobs.json"),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_candidates(monkeypatch, cache_path):
    tuner = importlib.import_module(_SHIM + ".int4_tuner")
    frontend = importlib.import_module(_SHIM + ".hopper_int4")
    for hidden, intermediate in ((3072, 1024), (384, 384)):
        shape = dict(_SHAPE, hidden=hidden, intermediate=intermediate)
        candidates = tuner.hopper_int4_candidates(2048, **shape)
        assert candidates
        keys = [json.dumps(c, sort_keys=True) for c in candidates]
        assert len(keys) == len(set(keys))
        tile_ks = {c["mma_tiler_mnk"][2] for c in candidates}
        assert tile_ks == ({128, 256} if hidden == 3072 else {128})
        for candidate in candidates:
            # The actual launch config is an independent consumer of the tuner.
            config = frontend.MegaMoEHopperInt4Config(
                rank=0,
                world_size=shape["world_size"],
                num_tokens_per_rank=2048,
                num_topk=8,
                num_total_experts=shape["num_experts"],
                hidden=hidden,
                intermediate=intermediate,
                **candidate,
            )
            assert config.fc1_ready_mode == "tile" and not config.fc2_tail_n8
        for unsupported in (
            {"fc1_ready_mode": "k256"},
            {"fc2_tail_n8": True},
        ):
            with pytest.raises(ValueError):
                tuner.normalize_hopper_int4_tactic({**candidates[0], **unsupported})


def _check_cache(monkeypatch, cache_path):
    tuner = importlib.import_module(_SHIM + ".int4_tuner")
    cache = importlib.import_module(_SHIM + ".knob_cache")
    int4 = importlib.import_module(_SHIM + ".hopper_int4")
    mxfp4 = importlib.import_module(_SHIM + ".hopper_mxfp4")
    nvfp4 = importlib.import_module(_SHIM + ".hopper_nvfp4")
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", cache_path)
    provenance = tuner.hopper_int4_cache_provenance_sha256(max_tokens=2048, **_SHAPE)
    key = dict(
        _SHAPE,
        dtype=int4._INT4_TUNING_DTYPE_ID,
        fp8_scale_mode="mxfp4_hybrid",
        topk=8,
        max_tokens=2048,
        live_tokens=2048,
        device="test-h200",
        compute_capability=(9, 0),
        sm_count=132,
        gate_up_clamp=10.0,
        routing_profile="block_permutation_v1",
        tuning_provenance_sha256=provenance,
    )
    winner = tuner.hopper_int4_candidates(2048, **_SHAPE)[-1]
    for dtype in (mxfp4._MXFP4_TUNING_DTYPE_ID, nvfp4._NVFP4_TUNING_DTYPE_ID):
        assert cache.record_knobs(winner, **{**key, "dtype": dtype}) is not None
    assert cache.lookup_knobs(**key) is None
    assert cache.record_knobs(winner, **key) is not None
    assert cache.lookup_knobs(**key) == winner
    for changed in (
        {"tuning_provenance_sha256": "0" * 64},
        {"tuning_provenance_sha256": None},
        {"sm_count": 114},
        {"topk": 4},
    ):
        assert cache.lookup_knobs(**{**key, **changed}) is None


def _check_auto_lifecycle(monkeypatch, cache_path):
    import torch

    from flashinfer.moe_ep import (
        BootstrapConfig,
        Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel

    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = create_mega_kernel(
            Sm90_Fp8_Int4_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=128, top_k=2, knobs="auto"
            )
        )
    backend.bind_ep_bootstrap(
        BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    pkg = importlib.import_module(_SHIM.rsplit(".", 1)[0])
    winner = {"mma_tiler_mnk": (256, 32, 128), "group_hint": 132}
    tune = mock.Mock(side_effect=[RuntimeError("sweep failed"), winner])
    launch = mock.Mock()
    monkeypatch.setattr(pkg, "autotune_hopper_int4_mega_moe", tune)
    monkeypatch.setattr(pkg, "hopper_int4_mega_moe", launch)
    workspace = mock.Mock()
    output = torch.empty(2, 128, dtype=torch.bfloat16)
    weights = (object(), object())
    with pytest.raises(ValueError, match="output"):
        backend.compute(workspace, weights, output=None)
    tune.assert_not_called()
    with pytest.raises(RuntimeError, match="sweep failed"):
        backend.compute(workspace, weights, output=output)
    assert backend._autotune_pending
    launch.assert_not_called()
    assert backend.compute(workspace, weights, output=output) is output
    assert not backend._autotune_pending
    assert backend.get_tuned_knobs() == winner
    assert backend.compute(workspace, weights, output=output) is output
    assert tune.call_count == 2 and launch.call_count == 2
    assert tune.call_args.kwargs["num_tokens"] == 2


def test_int4_offline_tuner_dispatch(monkeypatch):
    from flashinfer.moe_ep import tune

    run = mock.Mock(return_value=17)
    load = mock.Mock(return_value=SimpleNamespace(run_tuning=run))
    monkeypatch.setattr(tune, "importlib", SimpleNamespace(import_module=load))
    argv = [
        "--dtype",
        "sm90_int4",
        "--hidden",
        "3072",
        "--intermediate",
        "1024",
        "--num-experts",
        "256",
        "--topk",
        "8",
        "--max-tokens",
        "128",
        "512",
    ]
    assert tune.main(argv) == 17
    load.assert_called_once_with(
        ".backends.mega.kernel.sm90.fp8_int4_bf16_pull_cutedsl.tuner",
        "flashinfer.moe_ep",
    )
    args = run.call_args.args[0]
    assert args.fp8_scale_mode == "mxfp4_hybrid"
    assert args.max_tokens == [128, 512]
