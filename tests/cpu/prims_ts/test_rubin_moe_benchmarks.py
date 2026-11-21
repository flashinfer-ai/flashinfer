# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

import json
import sys

import pytest

pytestmark = pytest.mark.cpu_only


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
