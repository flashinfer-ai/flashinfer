"""Host contracts specific to the formal SM90 Green-split benchmark mode."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from io import StringIO
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
)


_BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_moe_ep_sm90_mega.py"
)


@pytest.fixture(scope="module")
def bench():
    name = "_flashinfer_bench_moe_ep_sm90_mega_split_contract"
    spec = importlib.util.spec_from_file_location(name, _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _args(bench, *extra):
    return bench._parse_args(
        [
            "--backend",
            bench.MXFP4_BACKEND,
            "--execution-mode",
            "split",
            "--hidden",
            "128",
            "--intermediate",
            "128",
            "--num-experts",
            "8",
            "--top-k",
            "2",
            *extra,
        ]
    )


def _launcher_identity(value) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def test_split_cli_resolves_static_independent_k1_k2_baseline(bench) -> None:
    args = _args(bench)
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)
    k1_label, k2_label = bench._split_tactic_labels(args)

    assert modes == ("mxfp4_hybrid",)
    assert orders == ("swap_ab",)
    assert bench._resolved_load_balance_mode(args) == "static"
    assert config.execution_mode == "split"
    assert config.load_balance_mode == "static"
    assert config.swap_ab is None
    assert config.pingpong is None
    assert config.mma_tiler_mnk is None
    assert config.cluster_shape_mnk is None
    assert config.token_back_mode is None
    assert config.split_k1_mma_tiler_mnk == bench.MXFP4_SPLIT_K1_TILE
    assert config.split_k2_mma_tiler_mnk == bench.MXFP4_SPLIT_K2_TILE
    assert config.split_k1_cluster_shape_mnk == bench.MXFP4_SPLIT_K1_CLUSTER
    assert config.split_k2_cluster_shape_mnk == bench.MXFP4_SPLIT_K2_CLUSTER
    assert config.split_k1_sm_count == 80
    assert config.split_k2_sm_count == 52
    assert config.split_counter_epoch_banks == 1
    assert config.split_graph_variant == "steady_k3_reset"
    assert k1_label.startswith("k1_m256n64k128_sm80_")
    assert k2_label.startswith("k2_m128n64k128_sm52_")
    tactic = bench._tactic_label(args, operand_order=orders[0], tile=tile)
    assert "green_split_" in tactic
    assert "banks1_steady_k3_reset" in tactic


def test_split_cli_carries_explicit_tactics_banks_graph_and_iket(bench) -> None:
    args = _args(
        bench,
        "--split-k1-mma-tiler",
        "128,128,128",
        "--split-k2-mma-tiler",
        "256,128,128",
        "--split-k1-cluster",
        "2,1,1",
        "--split-k2-cluster",
        "2,1,1",
        "--split-k1-group-hint",
        "72",
        "--split-k2-group-hint",
        "44",
        "--split-k1-num-sched-stages",
        "1",
        "--split-k2-num-sched-stages",
        "3",
        "--split-k1-sm-count",
        "72",
        "--split-k2-sm-count",
        "60",
        "--split-counter-banks",
        "2",
        "--split-graph-variant",
        "steady_k3_reset",
        "--split-enable-iket",
    )
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)

    assert config.split_k1_mma_tiler_mnk == (128, 128, 128)
    assert config.split_k2_mma_tiler_mnk == (256, 128, 128)
    assert config.split_k1_cluster_shape_mnk == (2, 1, 1)
    assert config.split_k2_cluster_shape_mnk == (2, 1, 1)
    assert config.split_k1_group_hint == 72
    assert config.split_k2_group_hint == 44
    assert config.split_k1_num_sched_stages == 1
    assert config.split_k2_num_sched_stages == 3
    assert config.split_k1_sm_count == 72
    assert config.split_k2_sm_count == 60
    assert config.split_counter_epoch_banks == 2
    assert config.split_graph_variant == "steady_k3_reset"
    assert config.split_enable_iket is True
    k1_label, k2_label = bench._split_tactic_labels(args)
    assert k1_label == "k1_m128n128k128_sm72_s1_cga2x1x1_gh72"
    assert k2_label == "k2_m256n128k128_sm60_s3_cga2x1x1_gh44"
    assert (
        bench._tactic_label(args, operand_order=orders[0], tile=tile)
        == f"green_split_{k1_label}_{k2_label}_banks2_"
        "steady_k3_reset_iket1"
    )


def test_fp8_or_atomic_scheduler_cannot_masquerade_as_green_split(bench) -> None:
    fp8 = bench._parse_args(["--execution-mode", "split"])
    with pytest.raises(ValueError, match="only by the MXFP4 backend"):
        bench._resolve_sweep(fp8, world_size=4)

    atomic = _args(bench, "--load-balance-mode", "atomic_counter")
    with pytest.raises(ValueError, match="cannot share an atomic"):
        bench._resolve_sweep(atomic, world_size=4)


@pytest.mark.parametrize(
    "extra",
    [
        ("--mxfp4-mma-tiler", "128,32,128"),
        ("--mxfp4-cluster", "2,1,1"),
        ("--mxfp4-group-hint", "128"),
        ("--mxfp4-num-sched-stages", "2"),
        ("--mxfp4-pingpong",),
        ("--no-mxfp4-pingpong",),
    ],
)
def test_split_rejects_every_fused_mxfp4_tactic_flag(bench, extra) -> None:
    with pytest.raises(ValueError, match="fused tactic flags"):
        bench._resolve_sweep(_args(bench, *extra), world_size=4)


def test_split_rejects_legacy_fused_mma_tiler(bench) -> None:
    with pytest.raises(ValueError, match="legacy --mma-tiler is fused-only"):
        bench._resolve_sweep(_args(bench, "--mma-tiler", "128,32"), world_size=4)


def _session_config(**changes):
    values = dict(
        k1_mma_tiler_mnk=(256, 64, 128),
        k2_mma_tiler_mnk=(128, 64, 128),
        k1_cluster_shape_mnk=(1, 1, 1),
        k2_cluster_shape_mnk=(1, 1, 1),
        k1_group_hint=None,
        k2_group_hint=None,
        k1_num_sched_stages=2,
        k2_num_sched_stages=2,
        k1_sm_count=80,
        k2_sm_count=52,
        counter_epoch_banks=1,
        graph_variant="steady_k3_reset",
        enable_iket=False,
        handoff_token_n=64,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    values.update(changes)
    return SimpleNamespace(**values)


def _compiled_pair_kernel(config, role):
    cluster = getattr(config, f"{role}_cluster_shape_mnk")
    sm_count = getattr(config, f"{role}_sm_count")
    requested_group = getattr(config, f"{role}_group_hint")
    group_hint = (
        requested_group
        if requested_group is not None
        else 3 * sm_count // (cluster[0] * cluster[1])
    )
    requested_stages = getattr(config, f"{role}_num_sched_stages")
    return SimpleNamespace(
        mma_tiler_mnk=getattr(config, f"{role}_mma_tiler_mnk"),
        cluster_shape_mn=cluster[:2],
        group_hint=group_hint,
        num_sched_stages=2 if requested_stages is None else requested_stages,
    )


def _compiled_pairs(config):
    return tuple(
        SimpleNamespace(
            plan=SimpleNamespace(
                k1_sm_count=config.k1_sm_count,
                k2_sm_count=config.k2_sm_count,
            ),
            k1_kernel=_compiled_pair_kernel(config, "k1"),
            k2_kernel=_compiled_pair_kernel(config, "k2"),
            workspace=SimpleNamespace(counter_epoch_banks=config.counter_epoch_banks),
            counter_epoch_bank=bank,
        )
        for bank in range(config.counter_epoch_banks)
    )


def _session(**changes):
    config = changes.pop("config", _session_config())
    values = dict(
        captured=True,
        poisoned=False,
        destroyed=False,
        graph_variant="steady_k3_reset",
        green_sm_counts=(80, 52),
        max_active_clusters=(80, 52),
        generation=7,
        config=config,
        _pairs=_compiled_pairs(config),
    )
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "workspace,match",
    [
        (SimpleNamespace(), "no Green session"),
        (SimpleNamespace(_session=_session(captured=False)), "did not capture"),
        (SimpleNamespace(_session=_session(poisoned=True)), "poisoned"),
        (SimpleNamespace(_session=_session(destroyed=True)), "destroyed"),
        (
            SimpleNamespace(_session=_session(graph_variant="cold_k0")),
            "graph variant",
        ),
        (
            SimpleNamespace(_session=_session(green_sm_counts=(72, 60))),
            "Green SM partition",
        ),
        (
            SimpleNamespace(_session=_session(max_active_clusters=(79, 52))),
            "max_active_clusters",
        ),
        (
            SimpleNamespace(
                _session=_session(config=_session_config(handoff_token_n=128))
            ),
            "handoff_token_n",
        ),
    ],
)
def test_split_session_metadata_fails_closed_on_incomplete_or_wrong_session(
    bench, workspace, match: str
) -> None:
    with pytest.raises(RuntimeError, match=match):
        bench._split_session_metadata(_args(bench), workspace)


@pytest.mark.parametrize(
    ("field", "wrong_value"),
    [
        ("k1_mma_tiler_mnk", (128, 64, 128)),
        ("k2_mma_tiler_mnk", (256, 64, 128)),
        ("k1_cluster_shape_mnk", (2, 1, 1)),
        ("k2_cluster_shape_mnk", (1, 2, 1)),
        ("k1_group_hint", 396),
        ("k2_group_hint", 132),
        ("k1_num_sched_stages", 1),
        ("k2_num_sched_stages", 3),
        ("k1_sm_count", 79),
        ("k2_sm_count", 51),
        ("counter_epoch_banks", 2),
        ("graph_variant", "cold_k0"),
        ("enable_iket", True),
    ],
)
def test_split_session_metadata_fails_closed_on_every_config_identity_mismatch(
    bench, field: str, wrong_value
) -> None:
    config = _session_config(**{field: wrong_value})
    workspace = SimpleNamespace(_session=_session(config=config))
    with pytest.raises(RuntimeError, match=field):
        bench._split_session_metadata(_args(bench), workspace)


def test_split_session_metadata_records_actual_generation_partition_and_clusters(
    bench,
) -> None:
    metadata = bench._split_session_metadata(
        _args(bench), SimpleNamespace(_session=_session())
    )
    expected_session = {
        "generation": 7,
        "graph_variant": "steady_k3_reset",
        "green_sm_counts": (80, 52),
        "max_active_clusters": (80, 52),
        "handoff_token_n": 64,
        "counter_banks": 1,
    }
    assert {name: metadata[name] for name in expected_session} == expected_session
    assert metadata["runtime_implementation"] == "mxfp4_split"
    assert metadata["runtime_tactic"] == {
        "k1_mma_tiler_mnk": [256, 64, 128],
        "k2_mma_tiler_mnk": [128, 64, 128],
        "k1_cluster_shape_mnk": [1, 1, 1],
        "k2_cluster_shape_mnk": [1, 1, 1],
        "k1_group_hint": 240,
        "k2_group_hint": 156,
        "k1_num_sched_stages": 2,
        "k2_num_sched_stages": 2,
        "k1_sm_count": 80,
        "k2_sm_count": 52,
        "counter_epoch_banks": 1,
        "graph_variant": "steady_k3_reset",
        "enable_iket": False,
    }
    assert metadata["runtime_tactic_sha256"] == (
        bench._canonical_runtime_tactic_sha256(
            "mxfp4_split", metadata["runtime_tactic"]
        )
    )


def test_split_session_metadata_rejects_compiled_bank_pair_tactic_disagreement(
    bench,
) -> None:
    config = _session_config(counter_epoch_banks=2)
    pairs = list(_compiled_pairs(config))
    pairs[1].k2_kernel.group_hint += 1
    workspace = SimpleNamespace(_session=_session(config=config, _pairs=tuple(pairs)))
    args = _args(bench, "--split-counter-banks", "2")
    with pytest.raises(RuntimeError, match="compiled schedule"):
        bench._split_session_metadata(args, workspace)


def test_split_session_metadata_rejects_malformed_compiled_cluster_shape_mn(
    bench,
) -> None:
    config = _session_config()
    pairs = list(_compiled_pairs(config))
    pairs[0].k1_kernel.cluster_shape_mn = (1, 1, 1)
    workspace = SimpleNamespace(_session=_session(config=config, _pairs=tuple(pairs)))
    with pytest.raises(RuntimeError, match="malformed cluster_shape_mn"):
        bench._split_session_metadata(_args(bench), workspace)


def test_split_csv_reports_mode_pair_partition_graph_and_banks(bench, capsys) -> None:
    args = _args(bench)
    csv_file = StringIO()
    metadata = bench._split_session_metadata(args, SimpleNamespace(_session=_session()))
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0] * 4,
        e2e_us=[10.0] * 4,
        e2e_median_us=[9.0] * 4,
        compute_us=[7.0] * 4,
        compute_median_us=[6.0] * 4,
        runtime_metadata=[{**metadata, "generation": rank + 1} for rank in range(4)],
    )
    bench._emit_row(
        args,
        scale_mode="mxfp4_hybrid",
        operand_order="swap_ab",
        tile=(128, 32),
        tokens=64,
        world_size=4,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["execution_mode"] == "split"
    assert values["tactic"].startswith("green_split_")
    assert values["k1_tactic"].startswith("k1_m256n64k128_sm80_")
    assert values["k2_tactic"].startswith("k2_m128n64k128_sm52_")
    assert values["graph_variant"] == "steady_k3_reset"
    assert values["counter_banks"] == "1"
    assert values["k1_sm_count"] == "80"
    assert values["k2_sm_count"] == "52"
    assert values["k1_max_active_clusters"] == "80"
    assert values["k2_max_active_clusters"] == "52"
    assert values["handoff_token_n"] == "64"
    assert values["rank_session_generations"] == "r0:g1|r1:g2|r2:g3|r3:g4"
    assert values["runtime_k1_group_hint"] == "240"
    assert values["runtime_k2_group_hint"] == "156"
    assert values["runtime_k1_num_sched_stages"] == "2"
    assert values["runtime_k2_num_sched_stages"] == "2"
    assert values["runtime_tactic_sha256"] == _launcher_identity(
        {
            "implementation": "mxfp4_split",
            "tactic": metadata["runtime_tactic"],
        }
    )

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    legacy_file_fields = bench.CSV_FIELDS.split(",") + bench.HEUR_CSV_FIELDS.split(",")
    assert file_header[: len(legacy_file_fields)] == legacy_file_fields
    file_values = dict(zip(file_header, file_row, strict=True))
    assert file_values["k1_max_active_clusters"] == "80"
    assert file_values["k2_max_active_clusters"] == "52"
    assert file_values["handoff_token_n"] == "64"
    assert file_values["rank_session_generations"] == "r0:g1|r1:g2|r2:g3|r3:g4"


def test_split_csv_reports_complete_tactic_identity_and_formal_score(
    bench, capsys
) -> None:
    args = _args(
        bench,
        "--split-k1-mma-tiler",
        "128,128,128",
        "--split-k2-mma-tiler",
        "256,128,128",
        "--split-k1-cluster",
        "2,1,1",
        "--split-k2-cluster",
        "2,1,1",
        "--split-k1-group-hint",
        "72",
        "--split-k2-group-hint",
        "44",
        "--split-k1-num-sched-stages",
        "1",
        "--split-k2-num-sched-stages",
        "3",
        "--split-k1-sm-count",
        "72",
        "--split-k2-sm-count",
        "60",
        "--split-counter-banks",
        "2",
        "--split-enable-iket",
    )
    csv_file = StringIO()
    config = _session_config(
        **bench._expected_split_session_config(args),
        handoff_token_n=128,
    )
    metadata = bench._split_session_metadata(
        args,
        SimpleNamespace(
            _session=_session(
                config=config, green_sm_counts=(72, 60), max_active_clusters=(36, 30)
            )
        ),
    )
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0] * 4,
        e2e_us=[10.0] * 4,
        e2e_median_us=[9.0] * 4,
        compute_us=[7.0] * 4,
        compute_median_us=[6.0, 7.25, 5.5, 6.5],
        runtime_metadata=[{**metadata, "generation": rank + 1} for rank in range(4)],
    )
    bench._emit_row(
        args,
        scale_mode="mxfp4_hybrid",
        operand_order="swap_ab",
        tile=(128, 32),
        tokens=64,
        world_size=4,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    assert header == [
        "BENCH_CSV",
        *bench.CSV_FIELDS.split(","),
        *bench.BENCH_EXT_CSV_FIELDS.split(","),
        *bench.SPLIT_RUNTIME_CSV_FIELDS.split(","),
        *bench.FORMAL_TUNING_CSV_FIELDS.split(","),
        *bench.FP8_RUNTIME_CSV_FIELDS.split(","),
        *bench.RUNTIME_TACTIC_CSV_FIELDS.split(","),
        *bench.ROUTING_CSV_FIELDS.split(","),
    ]
    assert len(row) == len(header)
    values = dict(zip(header, row, strict=True))
    assert values["execution_mode"] == "split"
    assert values["tactic"].endswith("_banks2_steady_k3_reset_iket1")
    assert values["k1_tactic"] == "k1_m128n128k128_sm72_s1_cga2x1x1_gh72"
    assert values["k2_tactic"] == "k2_m256n128k128_sm60_s3_cga2x1x1_gh44"
    assert values["counter_banks"] == "2"
    assert values["k1_sm_count"] == "72"
    assert values["k2_sm_count"] == "60"
    assert values["k1_max_active_clusters"] == "36"
    assert values["k2_max_active_clusters"] == "30"
    assert values["handoff_token_n"] == "128"
    assert values["rank_session_generations"] == "r0:g1|r1:g2|r2:g3|r3:g4"
    assert values["runtime_k1_group_hint"] == "72"
    assert values["runtime_k2_group_hint"] == "44"
    assert values["runtime_k1_num_sched_stages"] == "1"
    assert values["runtime_k2_num_sched_stages"] == "3"
    assert values["runtime_tactic_sha256"] == _launcher_identity(
        {
            "implementation": "mxfp4_split",
            "tactic": metadata["runtime_tactic"],
        }
    )
    assert values["routing_mode"] == "block_permutation"
    assert values["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert values["routing_seed"] == "1234"
    assert len(values["route_ids_sha256"]) == 64
    expected_formal = {
        "compute_max_rank_median_us": "7.250000",
        "split_k1_tile_m": "128",
        "split_k1_tile_n": "128",
        "split_k1_tile_k": "128",
        "split_k2_tile_m": "256",
        "split_k2_tile_n": "128",
        "split_k2_tile_k": "128",
        "split_k1_cga_m": "2",
        "split_k1_cga_n": "1",
        "split_k1_cga_k": "1",
        "split_k2_cga_m": "2",
        "split_k2_cga_n": "1",
        "split_k2_cga_k": "1",
        "split_k1_group_hint": "72",
        "split_k2_group_hint": "44",
        "split_k1_num_sched_stages": "1",
        "split_k2_num_sched_stages": "3",
        "split_enable_iket": "1",
    }
    for field, expected in expected_formal.items():
        assert values[field] == expected
    for field in (
        "fused_pingpong",
        "fused_cga_m",
        "fused_cga_n",
        "fused_cga_k",
        "fused_group_hint",
        "fused_num_sched_stages",
        "fused_load_balance_mode",
        "fused_token_back_mode",
    ):
        assert values[field] == ""

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    assert file_header == [
        *bench.CSV_FIELDS.split(","),
        *bench.HEUR_CSV_FIELDS.split(","),
        *bench.BENCH_EXT_CSV_FIELDS.split(","),
        *bench.SPLIT_RUNTIME_CSV_FIELDS.split(","),
        *bench.FORMAL_TUNING_CSV_FIELDS.split(","),
        *bench.FP8_RUNTIME_CSV_FIELDS.split(","),
        *bench.RUNTIME_TACTIC_CSV_FIELDS.split(","),
        *bench.ROUTING_CSV_FIELDS.split(","),
    ]
    assert len(file_row) == len(file_header)
    file_values = dict(zip(file_header, file_row, strict=True))
    for field, expected in expected_formal.items():
        assert file_values[field] == expected


def test_split_csv_rejects_missing_or_cross_rank_partition_metadata(bench) -> None:
    args = _args(bench)
    base = dict(
        status="pass",
        cold_us=[100.0] * 4,
        e2e_us=[10.0] * 4,
        e2e_median_us=[9.0] * 4,
        compute_us=[7.0] * 4,
        compute_median_us=[6.0] * 4,
    )
    for metadata in (
        None,
        [{"green_sm_counts": (80, 52)}],
        [
            {"green_sm_counts": (80, 52)},
            {"green_sm_counts": (80, 52)},
            {"green_sm_counts": (72, 60)},
            {"green_sm_counts": (80, 52)},
        ],
    ):
        result = bench.PointResult(**base, runtime_metadata=metadata)
        with pytest.raises(RuntimeError, match="metadata|partition"):
            bench._emit_row(
                args,
                scale_mode="mxfp4_hybrid",
                operand_order="swap_ab",
                tile=(128, 32),
                tokens=64,
                world_size=4,
                result=result,
                header_done=True,
            )


def test_split_cache_mode_reports_actual_runtime_tactic_and_session(
    bench, capsys
) -> None:
    args = _args(bench, "--mxfp4-tactic-source", "cache_or_heuristic")
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    layer_config = bench._megakernel_config(args, modes[0], orders[0], tile)
    assert layer_config.knobs is None
    assert layer_config.split_k1_sm_count is None
    assert layer_config.split_k2_sm_count is None

    runtime_config = _session_config(
        k1_mma_tiler_mnk=(256, 32, 128),
        k2_mma_tiler_mnk=(256, 64, 128),
        k1_group_hint=396,
        k2_group_hint=264,
        k1_num_sched_stages=2,
        k2_num_sched_stages=1,
        counter_epoch_banks=2,
        handoff_token_n=64,
    )
    metadata = bench._split_session_metadata(
        args, SimpleNamespace(_session=_session(config=runtime_config))
    )
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0] * 4,
        e2e_us=[10.0] * 4,
        e2e_median_us=[9.0] * 4,
        compute_us=[7.0] * 4,
        compute_median_us=[6.0, 7.25, 5.5, 6.5],
        runtime_metadata=[{**metadata, "generation": rank + 1} for rank in range(4)],
    )
    csv_file = StringIO()
    bench._emit_row(
        args,
        scale_mode=modes[0],
        operand_order=orders[0],
        tile=tile,
        tokens=64,
        world_size=4,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["tactic"] == "mxfp4_split_cache_or_heuristic"
    assert values["k1_tactic"] == "k1_m256n32k128_sm80_s2_cga1x1x1_gh396"
    assert values["k2_tactic"] == "k2_m256n64k128_sm52_s1_cga1x1x1_gh264"
    assert values["graph_variant"] == "steady_k3_reset"
    assert values["counter_banks"] == "2"
    assert values["k1_sm_count"] == "80"
    assert values["k2_sm_count"] == "52"
    assert values["split_k1_tile_n"] == "32"
    assert values["split_k2_tile_n"] == "64"
    assert values["split_k1_group_hint"] == "396"
    assert values["split_k2_group_hint"] == "264"
    assert values["split_k1_num_sched_stages"] == "2"
    assert values["split_k2_num_sched_stages"] == "1"
    assert values["compute_max_rank_median_us"] == "7.250000"
    assert values["runtime_tactic_sha256"] == metadata["runtime_tactic_sha256"]

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    file_values = dict(zip(file_header, file_row, strict=True))
    for field in (
        "tactic",
        "k1_tactic",
        "k2_tactic",
        "graph_variant",
        "counter_banks",
        "runtime_tactic_sha256",
    ):
        assert file_values[field] == values[field]


def test_split_cache_failure_leaves_unresolved_identity_blank(bench, capsys) -> None:
    args = _args(bench, "--mxfp4-tactic-source", "cache_or_heuristic")
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    result = bench.PointResult(
        status="failed",
        cold_us=[],
        e2e_us=[],
        e2e_median_us=[],
        compute_us=[],
        compute_median_us=[],
        error="compile failure",
    )
    csv_file = StringIO()
    bench._emit_row(
        args,
        scale_mode=modes[0],
        operand_order=orders[0],
        tile=tile,
        tokens=64,
        world_size=4,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["tactic"] == "mxfp4_split_cache_or_heuristic"
    for field in (
        "tile_m",
        "tile_n",
        "tile_k",
        "k1_tactic",
        "k2_tactic",
        "graph_variant",
        "counter_banks",
        "k1_sm_count",
        "k2_sm_count",
        "split_k1_tile_m",
        "split_k2_tile_m",
        "runtime_tactic_sha256",
    ):
        assert values[field] == ""

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    file_values = dict(zip(file_header, file_row, strict=True))
    for field in ("tile_m", "tile_n", "tile_k", "k1_tactic", "k2_tactic"):
        assert file_values[field] == ""
