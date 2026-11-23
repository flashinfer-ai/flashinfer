# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
import sys

import pytest

pytestmark = pytest.mark.cpu_only


@pytest.fixture(autouse=True)
def _benchmark_import_path(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[3]))


def test_moe_benchmark_rejects_empty_tactic_selection_before_preparation(
    monkeypatch, capsys
):
    from benchmarks import bench_rubin_moe as bench

    monkeypatch.setattr(bench.torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--tokens",
            "4096",
            "32",
            "--tactics",
            "512_0",
            "--output",
            "unused.json",
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "All requested tactics are excluded for tokens=32" in capsys.readouterr().err


@pytest.mark.parametrize(
    "option,value",
    [
        ("--hidden", "0"),
        ("--intermediate", "-1"),
        ("--hidden", "10241"),
        ("--intermediate", "4097"),
    ],
)
def test_moe_benchmark_rejects_invalid_dimensions_before_preparation(
    monkeypatch, capsys, option, value
):
    from benchmarks import bench_rubin_moe as bench

    monkeypatch.setattr(bench.torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        bench.torch, "randn", lambda *a, **kw: pytest.fail("allocated invalid weights")
    )
    monkeypatch.setattr(
        sys, "argv", ["bench", option, value, "--output", "unused.json"]
    )
    with pytest.raises(SystemExit, match="2"):
        bench.main()
    assert "dimensions must be" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["default", "sweep", "replay"])
@pytest.mark.parametrize("tokens", [32, 4096])
def test_backend_report_distinguishes_measured_modes_and_tactic_provenance(
    monkeypatch, tmp_path, mode, tokens
):
    from collections import namedtuple
    from types import SimpleNamespace
    import torch
    from benchmarks import bench_rubin_moe_backends as bench

    # Exercise report generation without allocating the E32 benchmark's weights.
    class CpuTorch:
        cuda = SimpleNamespace(
            get_device_capability=lambda: (10, 7),
            get_device_name=lambda: "CPU test",
            empty_cache=lambda: None,
        )

        def __getattr__(self, name):
            return getattr(torch, name)

        def randn(self, *shape, **kwargs):
            return torch.ones(shape if len(shape) == 2 and shape[1] == 32 else (1,))

        def arange(self, *args, **kwargs):
            return torch.arange(*args)

        def linspace(self, *args, **kwargs):
            return torch.linspace(*args)

    monkeypatch.setattr(bench, "torch", CpuTorch())
    monkeypatch.setattr(bench.PrimsTsConfig, "prepare_weights", lambda *a, **kw: {})
    monkeypatch.setattr(
        bench.PrimsTsConfig,
        "prepare_activations",
        lambda *a, **kw: (torch.ones(1, dtype=torch.uint8), torch.ones(1)),
    )
    monkeypatch.setattr(
        bench.CuteDslConfig,
        "prepare_weights",
        lambda *a, **kw: dict.fromkeys(
            (
                "w1_weight",
                "w1_weight_sf",
                "w1_alpha",
                "fc2_input_scale",
                "w2_weight",
                "w2_weight_sf",
                "w2_alpha",
            ),
            torch.ones(1),
        )
        | {"gated_weight_group_size": 64},
    )
    tactic = bench.DEFAULT_RUBIN_MOE_TACTIC
    monkeypatch.setattr(
        bench,
        "CuteDslMoEWrapper",
        lambda **kw: SimpleNamespace(
            _runner=SimpleNamespace(get_valid_tactics=lambda *a: [tactic]),
            run=lambda **kw: torch.ones(4),
        ),
    )
    monkeypatch.setattr(
        bench, "prims_ts_fp4_block_scale_moe", lambda **kw: [torch.ones(4)]
    )
    Routed = namedtuple("Routed", "topk_ids topk_weights")
    monkeypatch.setattr(
        bench,
        "trtllm_gen_routing",
        lambda logits, bias, method, k, **kw: Routed(
            torch.topk(logits, k, dim=-1).indices, torch.full((tokens, k), 1 / k)
        ),
    )
    captures = []

    def capture(run, blocks=10):
        captures.append(blocks)
        return object(), [run()]

    monkeypatch.setattr(bench, "capture", capture)
    monkeypatch.setattr(bench, "measure", lambda *a, **kw: [1.0])
    output = tmp_path / "report.json"
    argv = ["bench", "--tokens", str(tokens), "--rounds", "1", "--output", str(output)]
    if mode == "default":
        argv.append("--default-only")
    elif mode == "replay":
        previous = tmp_path / "prior.json"
        previous.write_text(
            json.dumps(
                dict(
                    experts=32,
                    hidden=10240,
                    intermediate=4096,
                    top_k=8,
                    results=[
                        dict(
                            tokens=tokens,
                            backend="cute_best_from_logits",
                            tactic=tactic,
                            is_sweep_winner=False,
                            tactic_source="replay",
                            replayed_from="original.json",
                        )
                    ],
                )
            )
        )
        argv.extend(["--tactics-from", str(previous)])
    monkeypatch.setattr(sys, "argv", argv)
    bench.main()
    report = json.loads(output.read_text())
    rows = {row["backend"]: row for row in report["results"]}
    assert report["requested_prims_fused_finalize"] is True
    assert rows["prims_default"]["fused_finalize"] is False
    if tokens == 4096:
        assert rows["prims_1024"]["fused_finalize"] is True
        assert rows["prims_1024_unfused"]["fused_finalize"] is False
    assert (
        "2 blocks" in report["candidate_timer"]
        and "10 samples" in report["candidate_timer"]
    )
    assert "10 blocks" in report["timer"] and "30 samples" in report["timer"]
    assert len(captures) == (1 if tokens == 32 else 3) + 1 + (
        2 if mode == "default" else 4
    )
    if mode == "default":
        assert not any("best" in name for name in rows)
    else:
        assert rows["cute_best_from_logits"]["is_sweep_winner"] == (mode == "sweep")
        assert rows["cute_best_from_logits"]["tactic_source"] == mode
    if mode == "replay":
        assert report["sweeps"][0]["winner"] is None
        assert rows["cute_best_from_logits"]["replayed_from"] == str(previous)


@pytest.mark.parametrize(
    "default_only,winner", [(True, False), (False, False), (False, True), (False, None)]
)
def test_backend_benchmark_only_replays_sweep_winners(
    monkeypatch, tmp_path, capsys, default_only, winner
):
    from benchmarks import bench_rubin_moe_backends as bench

    row = dict(tokens=32, backend="cute_best_from_logits", tactic=[1, 2])
    if winner is not None:
        row["is_sweep_winner"] = winner
    report = dict(
        experts=32,
        hidden=10240,
        intermediate=4096,
        top_k=8,
        default_only=default_only,
        results=[row],
    )
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report))
    monkeypatch.setattr(bench.torch.cuda, "get_device_capability", lambda: (10, 7))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "bench",
            "--tokens",
            "32",
            "--tactics-from",
            str(path),
            "--output",
            str(tmp_path / "out.json"),
        ],
    )

    class PreparationReached(Exception):
        pass

    def stop_preparation(*args, **kwargs):
        raise PreparationReached

    monkeypatch.setattr(bench.torch, "randn", stop_preparation)
    if default_only or winner is False:
        with pytest.raises(SystemExit, match="2"):
            bench.main()
        assert "winner" in capsys.readouterr().err
    else:
        # Both new winners and genuine legacy sweep reports remain replayable.
        with pytest.raises(PreparationReached):
            bench.main()
