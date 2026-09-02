"""Host contracts for the formal SM90 FP8/MXFP4 MegaMoE benchmark."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from io import StringIO
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
)
from flashinfer.moe_ep.weights import PrequantizedMoEWeights


_BENCH_PATH = (
    Path(__file__).resolve().parents[2] / "benchmarks" / "bench_moe_ep_sm90_mega.py"
)
_HISTORICAL_FP8_CSV_FIELDS = (
    "kernel,scale_mode,operand_order,tile_m,tile_n,tile_k,"
    "tokens_per_rank,topk,world_size,total_experts,local_experts,hidden,"
    "intermediate_downproj,intermediate_gateup,warmup,iters,status,"
    "e2e_min_us,e2e_max_us,e2e_mean_us,e2e_median_us,"
    "compute_min_us,compute_max_us,compute_mean_us,compute_median_us,"
    "fc1_flops_per_rank,fc2_flops_per_rank,total_flops_per_rank,"
    "critical_tflops_compute,critical_tflops_e2e,tok_s_e2e,ref_csv"
)


def _mxfp4_args(bench, *extra: str):
    return bench._parse_args(["--backend", bench.MXFP4_BACKEND, *extra])


def _expected_fused_knobs(
    *,
    mma=(128, 32, 128),
    cluster=(1, 1, 1),
    pingpong=False,
    group_hint=None,
    num_sched_stages=None,
    load_balance_mode="atomic_counter",
):
    return {
        "swap_ab": True,
        "pingpong": pingpong,
        "mma_tiler_mnk": mma,
        "cluster_shape_mnk": cluster,
        "fp8_accum_mode": "1xacc",
        "load_balance_mode": load_balance_mode,
        "token_back_mode": "epi_warps",
        "in_kernel_fc2_reduce": False,
        "group_hint": group_hint,
        "num_sched_stages": num_sched_stages,
    }


def _fp8_candidate(**overrides):
    candidate = {
        "swap_ab": True,
        "pingpong": True,
        "mma_tiler_mnk": (128, 32, 128),
        "cluster_shape_mnk": (2, 1, 1),
        "fp8_accum_mode": "1xacc",
        "token_back_mode": "epi_warps",
    }
    candidate.update(overrides)
    return candidate


def _launcher_identity(value) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def _fused_runtime_metadata(bench, args, tile):
    tactic = bench._mxfp4_fused_tactic(args, tile)
    cluster = tactic["cluster_shape_mnk"]
    group_hint = tactic["group_hint"]
    if group_hint is None:
        group_hint = 132 // (cluster[0] * cluster[1])
    num_sched_stages = tactic["num_sched_stages"]
    if num_sched_stages is None:
        num_sched_stages = 2
    config = SimpleNamespace(
        swap_ab=tactic["swap_ab"],
        pingpong=tactic["pingpong"],
        mma_tiler_mnk=tactic["mma_tiler_mnk"],
        cluster_shape_mnk=cluster,
        fp8_accum_mode=tactic["fp8_accum_mode"],
        load_balance_mode=tactic["load_balance_mode"],
        resolved_token_back_mode=tactic["token_back_mode"],
        group_hint=tactic["group_hint"],
        num_sched_stages=tactic["num_sched_stages"],
        in_kernel_fc2_reduce=tactic["in_kernel_fc2_reduce"],
        routing_profile=bench.sm90_routing_profile_from_benchmark_mode(
            args.routing_mode
        ),
    )
    kernel = SimpleNamespace(group_hint=group_hint, num_sched_stages=num_sched_stages)
    frontend = SimpleNamespace(config=config, _mega=SimpleNamespace(kernel=kernel))
    return bench._mxfp4_fused_runtime_metadata(
        args, SimpleNamespace(_frontend=frontend)
    )


@pytest.fixture(scope="module")
def bench():
    name = "_flashinfer_bench_moe_ep_sm90_mega_contract"
    spec = importlib.util.spec_from_file_location(name, _BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_fp8_csv_fields_remain_exact_historical_prefix(bench):
    assert bench.CSV_FIELDS == _HISTORICAL_FP8_CSV_FIELDS
    historical = _HISTORICAL_FP8_CSV_FIELDS.split(",")
    stdout_header = bench.CSV_HEADER.split(",")
    assert stdout_header[0] == "BENCH_CSV"
    assert stdout_header[1 : 1 + len(historical)] == historical


def test_direct_launch_imports_this_source_tree_without_pythonpath(bench):
    repo_root = _BENCH_PATH.parents[1]
    assert Path(bench._repo_root).resolve() == repo_root
    assert Path(sys.path[0]).resolve() == repo_root

    code = f"""
import importlib.util
import pathlib
import sys

bench_path = pathlib.Path({str(_BENCH_PATH)!r})
name = "_flashinfer_benchmark_source_tree_probe"
spec = importlib.util.spec_from_file_location(name, bench_path)
module = importlib.util.module_from_spec(spec)
sys.modules[name] = module
spec.loader.exec_module(module)
import flashinfer
print(pathlib.Path(flashinfer.__file__).resolve())
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd="/tmp",
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    assert Path(result.stdout.strip()) == repo_root / "flashinfer" / "__init__.py"


def test_default_fp8_effective_cli_contract_is_unchanged(bench):
    args = bench._parse_args([])
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)

    assert args.backend == bench.FP8_BACKEND
    assert modes == ("per_tensor", "blockwise")
    assert orders == ("heuristic",)
    assert tile is None
    assert bench._resolved_token_back(args) is None
    assert args.routing_mode == "block_permutation"


def test_default_block_permutation_routing_identity_is_unchanged(bench, capsys):
    ids = bench._balanced_routing(
        512,
        6,
        384,
        rank=0,
        world_size=4,
        device=torch.device("cpu"),
    )
    assert tuple(ids.shape) == (512, 6)
    stdout = capsys.readouterr().out
    audit = json.loads(stdout.split("ROUTING_AUDIT,", 1)[1])
    assert audit["mode"] == "block_permutation"
    assert audit["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert (audit["expert_count_min"], audit["expert_count_max"]) == (26, 38)
    assert [owner["n32_tile_tasks"] for owner in audit["owners"]] == [
        133,
        132,
        141,
        134,
    ]


@pytest.mark.parametrize("tokens", [8, 32, 64, 128, 256, 512, 1024, 2048])
def test_published_exact_balanced_routing_is_deterministic(bench, tokens):
    kwargs = {
        "world_size": 4,
        "tokens": tokens,
        "topk": 6,
        "total_experts": 384,
        "seed": 1234,
    }
    routes = bench._published_exact_balanced_routes(**kwargs)
    replay = bench._published_exact_balanced_routes(**kwargs)
    routes_t = torch.from_numpy(routes.astype("int64"))
    replay_t = torch.from_numpy(replay.astype("int64"))

    assert torch.equal(routes_t, replay_t)
    assert tuple(routes_t.shape) == (4, tokens, 6)
    assert not torch.any(routes_t.sort(dim=2).values.diff(dim=2) == 0)
    counts = torch.bincount(routes_t.reshape(-1), minlength=384)
    assert int(counts.max() - counts.min()) <= 1

    rows_per_owner = tokens * 6 // 4
    for source_rank in range(4):
        owners = routes_t[source_rank].reshape(-1) // 96
        assert torch.bincount(owners, minlength=4).tolist() == [rows_per_owner] * 4

    if tokens == 512:
        assert torch.all(counts == 32)


@pytest.mark.parametrize(
    ("tokens", "expected_hash"),
    [
        (8, "1ba40a6fb0ab731b9085979a1968c60aa6b3a5fa3e13b444f2c0a55bcfb8aa00"),
        (32, "5499b7ae730372fb6ae53f29b852a07ec10aeeaa11d1e4e99424f3edd9be16ce"),
        (64, "d78a2b4df5bb769238a2528a76ccf80a980074bdb02c88152fbefbbd0d21e90e"),
        (128, "1209a05edefdc700fb8d45b54c2291b62d410cdc2934e801ca05f0a84a38b06f"),
        (256, "415b8f862a97ea9cc498bbc150e1e6d5d7b7111c27b387dcba95169386b1d7e2"),
        (512, "f5306ed4f8d1fd685fedf370c96e942f715b9481367be5932200e36d444379de"),
        (1024, "5999065601264efc000004684321ef46c4c1996b6531ecdbd985e8a617ec7dd5"),
        (2048, "85c6311af059960c02445ee051e3950991f34dc1979e537081d00c7f5da40b53"),
    ],
)
def test_published_exact_balanced_global_hash_is_frozen(bench, tokens, expected_hash):
    routes = bench._published_exact_balanced_routes(
        world_size=4,
        tokens=tokens,
        topk=6,
        total_experts=384,
        seed=1234,
    )
    audit = bench._routing_audit_payload(
        routes,
        mode="published_exact_balanced",
        seed=1234,
        num_experts=384,
        world_size=4,
    )
    assert audit["routing_profile"] == (SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED)
    assert audit["route_ids_sha256"] == expected_hash


def test_published_exact_balanced_routing_cli_is_explicit(bench):
    args = bench._parse_args(["--routing-mode", "published_exact_balanced"])
    assert args.routing_mode == "published_exact_balanced"


def test_benchmark_modes_map_strictly_to_canonical_profiles(bench):
    assert (
        bench.sm90_routing_profile_from_benchmark_mode("block_permutation")
        == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    )
    assert (
        bench.sm90_routing_profile_from_benchmark_mode("published_exact_balanced")
        == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    )
    with pytest.raises(ValueError, match="unsupported"):
        bench.sm90_routing_profile_from_benchmark_mode("block_permutation_v1")


@pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
def test_every_hopper_fp8_candidate_is_exactly_replayable(bench, scale_mode):
    from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel import (
        hopper_fp8_candidates,
    )

    candidates = hopper_fp8_candidates(fp8_scale_mode=scale_mode, max_tokens=512)
    labels = set()
    for candidate in candidates:
        args = bench._parse_args(
            ["--scale-mode", scale_mode, "--fp8-knobs-json", json.dumps(candidate)]
        )
        modes, orders, tile = bench._resolve_sweep(args, world_size=4)
        config = bench._megakernel_config(args, modes[0], orders[0], tile)
        assert modes == (scale_mode,)
        assert orders == ("swap_ab" if candidate["swap_ab"] else "non_swap_ab",)
        assert tile == tuple(candidate["mma_tiler_mnk"][:2])
        assert config.knobs == candidate
        assert config.swap_ab is None
        assert config.pingpong is None
        assert config.mma_tiler_mnk is None
        assert config.cluster_shape_mnk is None
        assert config.token_back_mode is None
        label = bench._tactic_label(args, operand_order=orders[0], tile=tile)
        assert "," not in label
        labels.add(label)
    assert len(labels) == len(candidates)


def test_fp8_explicit_schedule_knobs_are_normalized_and_identified(bench):
    candidate = _fp8_candidate(
        group_hint=512,
        num_sched_stages=3,
        flag_batch=4,
        epi_flag_batch=(4, 8),
        load_balance_mode="static",
        in_kernel_fc2_reduce=False,
    )
    args = bench._parse_args(["--fp8-knobs-json", json.dumps(candidate)])
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)
    assert config.knobs == candidate
    label = bench._tactic_label(args, operand_order=orders[0], tile=tile)
    assert "gh512_ns3_fb4_efb4x8_static_epi_warps_ikr0" in label


@pytest.mark.parametrize(
    ("candidate", "match"),
    [
        ([], "JSON object"),
        ({**_fp8_candidate(), "typo_flag_bach": 4}, "unsupported knob"),
        ({"swap_ab": True}, "fully specify"),
        ({**_fp8_candidate(), "swap_ab": 1}, "must be boolean"),
        ({**_fp8_candidate(), "mma_tiler_mnk": (256, 32, 128)}, "not a valid"),
        ({**_fp8_candidate(), "num_sched_stages": 0}, "null or positive"),
    ],
)
def test_fp8_explicit_knobs_reject_malformed_or_invalid_tactics(
    bench, candidate, match
):
    args = bench._parse_args(["--fp8-knobs-json", json.dumps(candidate)])
    with pytest.raises(ValueError, match=match):
        bench._resolve_sweep(args, world_size=4)


@pytest.mark.parametrize(
    "conflict",
    [["--swap-ab"], ["--mma-tiler", "128,32"], ["--token-back", "epi_warps"]],
)
def test_fp8_explicit_knobs_reject_ambiguous_legacy_flags(bench, conflict):
    args = bench._parse_args(
        ["--fp8-knobs-json", json.dumps(_fp8_candidate()), *conflict]
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        bench._resolve_sweep(args, world_size=4)


def test_mxfp4_rejects_fp8_explicit_knobs(bench):
    args = _mxfp4_args(bench, "--fp8-knobs-json", json.dumps(_fp8_candidate()))
    with pytest.raises(ValueError, match="requires --backend"):
        bench._resolve_sweep(args, world_size=4)


def test_fp8_csv_reports_actual_all_rank_resolved_identity(bench, capsys):
    candidate = _fp8_candidate(group_hint=None, num_sched_stages=2, flag_batch=4)
    args = bench._parse_args(["--fp8-knobs-json", json.dumps(candidate)])
    config = SimpleNamespace(
        swap_ab=True,
        pingpong=True,
        mma_tiler_mnk=(128, 32, 128),
        cluster_shape_mnk=(2, 1, 1),
        fp8_accum_mode="1xacc",
        fp8_scale_mode="per_tensor",
        group_hint=None,
        num_sched_stages=2,
        flag_batch=4,
        epi_flag_batch=(2, 4),
        load_balance_mode="atomic_counter",
        resolved_token_back_mode="epi_warps",
        in_kernel_fc2_reduce=False,
    )
    kernel = SimpleNamespace(group_hint=66, num_sched_stages=2)
    frontend = SimpleNamespace(config=config, _mega=SimpleNamespace(kernel=kernel))
    metadata = bench._fp8_runtime_metadata(args, SimpleNamespace(_frontend=frontend))
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0, 110.0],
        e2e_us=[10.0, 12.0],
        e2e_median_us=[9.0, 11.0],
        compute_us=[7.0, 8.0],
        compute_median_us=[6.0, 7.0],
        runtime_metadata=[metadata, dict(metadata)],
    )
    bench._emit_row(
        args,
        scale_mode="per_tensor",
        operand_order="swap_ab",
        tile=(128, 32),
        tokens=8,
        world_size=2,
        result=result,
        header_done=False,
    )
    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    values = dict(zip(header[1:], row[1:], strict=True))
    runtime_fields = bench.RUNTIME_TACTIC_CSV_FIELDS.split(",")
    fp8_fields = bench.FP8_RUNTIME_CSV_FIELDS.split(",")
    routing_fields = bench.ROUTING_CSV_FIELDS.split(",")
    assert header[-len(routing_fields) :] == routing_fields
    assert (
        header[-len(routing_fields) - len(runtime_fields) : -len(routing_fields)]
        == runtime_fields
    )
    assert (
        header[
            -len(routing_fields) - len(runtime_fields) - len(fp8_fields) : -len(
                routing_fields
            )
            - len(runtime_fields)
        ]
        == fp8_fields
    )
    assert values["fp8_tactic_mode"] == "explicit_knobs"
    assert values["fp8_tile_m"] == "128"
    assert values["fp8_cga_m"] == "2"
    assert values["fp8_group_hint"] == "66"
    assert values["fp8_num_sched_stages"] == "2"
    assert values["fp8_flag_batch"] == "4"
    assert values["fp8_token_back_mode"] == "epi_warps"
    assert values["runtime_group_hint"] == "66"
    assert values["runtime_num_sched_stages"] == "2"
    assert values["routing_mode"] == "block_permutation"
    assert values["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert values["routing_seed"] == "1234"
    assert len(values["route_ids_sha256"]) == 64
    assert values["runtime_tactic_sha256"] == _launcher_identity(
        {
            "implementation": "fp8_per_tensor",
            "tactic": metadata["runtime_tactic"],
        }
    )
    mismatched = dict(metadata)
    mismatched["flag_batch"] = 8
    result.runtime_metadata = [metadata, mismatched]
    with pytest.raises(RuntimeError, match="ranks disagree"):
        bench._fp8_runtime_cols(args, result)
    mismatched = dict(metadata)
    mismatched_tactic = dict(metadata["runtime_tactic"])
    mismatched_tactic["group_hint"] = 67
    mismatched["runtime_tactic"] = mismatched_tactic
    mismatched["runtime_tactic_sha256"] = _launcher_identity(
        {
            "implementation": "fp8_per_tensor",
            "tactic": mismatched_tactic,
        }
    )
    result.runtime_metadata = [metadata, mismatched]
    with pytest.raises(RuntimeError, match="ranks disagree"):
        bench._runtime_tactic_cols(args, "per_tensor", result, 2)


def test_mxfp4_defaults_resolve_to_explicit_fixed_tactic(bench):
    args = _mxfp4_args(bench, "--tokens", "64")
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)

    assert modes == ("mxfp4_hybrid",)
    assert orders == ("swap_ab",)
    assert tile == (128, 32)
    assert config.kernel_name == bench.MXFP4_BACKEND
    assert config.kind == "fp8_e4m3"
    assert config.fp8_scale_mode == "mxfp4_hybrid"
    assert config.fp8_accum_mode == "1xacc"
    assert config.routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert config.knobs == _expected_fused_knobs()
    assert config.swap_ab is None
    assert config.pingpong is None
    assert config.mma_tiler_mnk is None
    assert config.cluster_shape_mnk is None
    assert config.token_back_mode is None
    assert (
        bench._tactic_label(args, operand_order=orders[0], tile=tile)
        == "swapab_m128n32k128_cga1x1x1_pp0_ghauto_sauto_"
        "atomic_counter_epi_warps"
    )


def test_mxfp4_exact_mode_passes_canonical_profile_to_config(bench):
    args = _mxfp4_args(
        bench,
        "--routing-mode",
        "published_exact_balanced",
    )
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)
    assert config.routing_profile == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED


@pytest.mark.parametrize("mode", ["block_permutation", "published_exact_balanced"])
def test_fp8_and_mxfp4_share_global_route_identity_for_one_mode(bench, mode):
    fp8_args = bench._parse_args(["--routing-mode", mode])
    mxfp4_args = _mxfp4_args(bench, "--routing-mode", mode)
    failed = bench.PointResult(
        status="failed",
        cold_us=[],
        e2e_us=[],
        e2e_median_us=[],
        compute_us=[],
        compute_median_us=[],
    )
    assert bench._routing_csv_cols(fp8_args, 512, 4, failed) == (
        bench._routing_csv_cols(mxfp4_args, 512, 4, failed)
    )


def test_mxfp4_runtime_routing_profile_mismatch_fails_closed(bench):
    args = _mxfp4_args(
        bench,
        "--routing-mode",
        "published_exact_balanced",
    )
    with pytest.raises(RuntimeError, match="!= requested"):
        bench._verified_mxfp4_runtime_routing_profile(
            args,
            SimpleNamespace(routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION),
            "test runtime",
        )


def test_mxfp4_legacy_mn_override_preserves_k128(bench):
    args = _mxfp4_args(bench, "--mma-tiler", "256,32")
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)

    assert tile == (256, 32)
    assert config.knobs == _expected_fused_knobs(mma=(256, 32, 128))
    assert "m256n32k128" in bench._tactic_label(
        args, operand_order=orders[0], tile=tile
    )


def test_mxfp4_explicit_full_tactic_preserves_k256_and_all_knobs(bench):
    args = _mxfp4_args(
        bench,
        "--mxfp4-mma-tiler",
        "128,64,256",
        "--mxfp4-cluster",
        "2,1,1",
        "--mxfp4-group-hint",
        "37",
        "--mxfp4-num-sched-stages",
        "3",
        "--mxfp4-pingpong",
    )
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)

    assert tile == (128, 64)
    assert config.knobs == _expected_fused_knobs(
        mma=(128, 64, 256),
        cluster=(2, 1, 1),
        pingpong=True,
        group_hint=37,
        num_sched_stages=3,
    )
    assert (
        bench._tactic_label(args, operand_order=orders[0], tile=tile)
        == "swapab_m128n64k256_cga2x1x1_pp1_gh37_s3_"
        "atomic_counter_epi_warps"
    )


@pytest.mark.parametrize(
    ("cluster", "group_hint", "load_balance_mode", "expected_label"),
    [
        pytest.param(
            "1,1,1",
            528,
            "static",
            "swapab_m256n32k256_cga1x1x1_pp0_gh528_s2_static_epi_warps",
            id="accepted-anchor",
        ),
        pytest.param(
            "2,1,1",
            512,
            "atomic_counter",
            "swapab_m256n32k256_cga2x1x1_pp0_gh512_s2_atomic_counter_epi_warps",
            id="candidate-b",
        ),
    ],
)
def test_t512_cooldown_ab_tactics_are_exactly_representable(
    bench, cluster, group_hint, load_balance_mode, expected_label
):
    args = _mxfp4_args(
        bench,
        "--tokens",
        "512",
        "--mxfp4-mma-tiler",
        "256,32,256",
        "--mxfp4-cluster",
        cluster,
        "--mxfp4-group-hint",
        str(group_hint),
        "--mxfp4-num-sched-stages",
        "2",
        "--no-mxfp4-pingpong",
        "--load-balance-mode",
        load_balance_mode,
        "--token-back",
        "epi_warps",
    )
    modes, orders, tile = bench._resolve_sweep(args, world_size=4)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)

    assert args.tokens == "512"
    assert tile == (256, 32)
    assert config.knobs == _expected_fused_knobs(
        mma=(256, 32, 256),
        cluster=tuple(int(v) for v in cluster.split(",")),
        group_hint=group_hint,
        num_sched_stages=2,
        load_balance_mode=load_balance_mode,
    )
    assert (
        bench._tactic_label(args, operand_order=orders[0], tile=tile) == expected_label
    )


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_mxfp4_accepts_required_rank_counts(bench, world_size):
    args = bench._parse_args(["--backend", bench.MXFP4_BACKEND])
    assert bench._resolve_sweep(args, world_size)[0] == ("mxfp4_hybrid",)


@pytest.mark.parametrize(
    ("extra", "world_size", "match"),
    [
        (["--no-swap-ab"], 4, "requires --swap-ab"),
        (["--scale-mode", "per_tensor"], 4, "mxfp4_hybrid"),
        (["--kind", "fp8_e5m2"], 4, "fp8_e4m3"),
        (["--fp8-accum-mode", "2xacc"], 4, "1xacc"),
        (["--token-back", "heuristic"], 4, "must be fixed"),
        ([], 3, "exactly 1, 2, 4, or 8"),
    ],
)
def test_mxfp4_rejects_cross_format_or_nonfixed_runs(bench, extra, world_size, match):
    args = bench._parse_args(["--backend", bench.MXFP4_BACKEND, *extra])
    with pytest.raises(ValueError, match=match):
        bench._resolve_sweep(args, world_size)


def test_fp8_rejects_mxfp4_scale_mode(bench):
    args = bench._parse_args(["--scale-mode", "mxfp4_hybrid"])
    with pytest.raises(ValueError, match="no cross-format fallback"):
        bench._resolve_sweep(args, world_size=4)


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--mxfp4-group-hint", "1"], "require --backend"),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--execution-mode",
                "split",
                "--mxfp4-cluster",
                "1,1,1",
            ],
            "cannot be used with --execution-mode split",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mma-tiler",
                "128,32",
                "--mxfp4-mma-tiler",
                "128,32,128",
            ],
            "not both",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-mma-tiler",
                "128,32",
            ],
            "three positive integers",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-mma-tiler",
                "64,32,128",
            ],
            "M in",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-mma-tiler",
                "128,8,128",
            ],
            "N in",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-mma-tiler",
                "128,32,64",
            ],
            "K must be 128 or 256",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-mma-tiler",
                "256,32,256",
                "--mxfp4-pingpong",
            ],
            "ping-pong requires MMA tile M=128",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-cluster",
                "4,1,1",
            ],
            "unsupported MXFP4 fused cluster shape",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-group-hint",
                "0",
            ],
            "group-hint.*positive",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--mxfp4-num-sched-stages",
                "0",
            ],
            "num-sched-stages.*positive",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--hidden",
                "7169",
                "--mxfp4-mma-tiler",
                "128,32,256",
            ],
            "hidden.*divisible by tile K=256",
        ),
        (
            [
                "--backend",
                "sm90_fp8_mxfp4_bf16_pull_cutedsl",
                "--intermediate",
                "3073",
                "--mxfp4-mma-tiler",
                "128,32,256",
            ],
            "intermediate.*divisible by tile K=256",
        ),
    ],
)
def test_mxfp4_rejects_illegal_fused_flag_or_domain(bench, argv, match):
    args = bench._parse_args(argv)
    with pytest.raises(ValueError, match=match):
        bench._resolve_sweep(args, world_size=4)


def test_raw_mxfp4_pack_is_deterministic_canonical_e2m1_e8m0(bench):
    args = bench._parse_args(
        [
            "--backend",
            bench.MXFP4_BACKEND,
            "--hidden",
            "128",
            "--intermediate",
            "128",
            "--num-experts",
            "1",
        ]
    )
    first = bench._make_raw_mxfp4_weights(args, 1, 0, torch.device("cpu"))
    second = bench._make_raw_mxfp4_weights(args, 1, 0, torch.device("cpu"))

    assert isinstance(first, PrequantizedMoEWeights)
    assert first.w13.shape == (1, 256, 64)
    assert first.w13_scale.shape == (1, 256, 4)
    assert first.w2.shape == (1, 128, 64)
    assert first.w2_scale.shape == (1, 128, 4)

    for lhs, rhs in zip(
        (first.w13, first.w13_scale, first.w2, first.w2_scale),
        (second.w13, second.w13_scale, second.w2, second.w2_scale),
        strict=True,
    ):
        assert lhs.dtype == torch.uint8
        assert lhs.is_contiguous()
        torch.testing.assert_close(lhs, rhs, rtol=0, atol=0)

    nibbles = torch.cat(
        [
            first.w13.flatten() & 0xF,
            first.w13.flatten() >> 4,
            first.w2.flatten() & 0xF,
            first.w2.flatten() >> 4,
        ]
    )
    assert set(torch.unique(nibbles).tolist()) == set(range(16))
    for scale in (first.w13_scale, first.w2_scale):
        assert int(scale.min()) >= bench.MXFP4_E8M0_MIN
        assert int(scale.max()) < bench.MXFP4_E8M0_MAX_EXCLUSIVE


def test_registry_identity_check_forbids_fallback(bench):
    class WrongBackend:
        @classmethod
        def kernel_name(cls):
            return bench.FP8_BACKEND

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        bench._assert_backend_identity(WrongBackend(), bench.MXFP4_BACKEND)


def test_mxfp4_csv_reports_backend_tactic_cold_and_warm(bench, capsys):
    args = bench._parse_args(
        [
            "--backend",
            bench.MXFP4_BACKEND,
            "--hidden",
            "128",
            "--intermediate",
            "128",
            "--num-experts",
            "2",
            "--top-k",
            "2",
        ]
    )
    metadata = _fused_runtime_metadata(bench, args, (128, 32))
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0, 110.0],
        e2e_us=[10.0, 12.0],
        e2e_median_us=[9.0, 11.0],
        compute_us=[7.0, 8.0],
        compute_median_us=[6.0, 7.0],
        runtime_metadata=[metadata, dict(metadata)],
    )
    bench._emit_row(
        args,
        scale_mode="mxfp4_hybrid",
        operand_order="swap_ab",
        tile=(128, 32),
        tokens=8,
        world_size=2,
        result=result,
        header_done=False,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    assert header[0] == row[0] == "BENCH_CSV"
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["kernel"] == bench.MXFP4_BACKEND
    assert values["tactic"].startswith("swapab_m128n32k128_")
    assert values["scale_mode"] == "mxfp4_hybrid"
    assert values["cold_first_call_min_us"] == "100.00"
    assert values["cold_first_call_max_us"] == "110.00"
    assert values["cold_first_call_mean_us"] == "105.00"
    assert values["e2e_max_us"] == "12.00"
    assert values["compute_max_us"] == "8.00"
    assert values["ref_csv"] == "not_applicable(mxfp4)"
    assert values["runtime_group_hint"] == "132"
    assert values["runtime_num_sched_stages"] == "2"
    assert values["runtime_tactic_sha256"] == metadata["runtime_tactic_sha256"]


def test_mxfp4_csv_reports_strict_stdout_and_file_schema(bench, capsys):
    args = _mxfp4_args(
        bench,
        "--hidden",
        "256",
        "--intermediate",
        "256",
        "--num-experts",
        "2",
        "--top-k",
        "2",
        "--mxfp4-mma-tiler",
        "128,64,256",
        "--mxfp4-cluster",
        "2,1,1",
        "--mxfp4-group-hint",
        "512",
        "--mxfp4-num-sched-stages",
        "2",
    )
    csv_file = StringIO()
    metadata = _fused_runtime_metadata(bench, args, (128, 64))
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0, 110.0],
        e2e_us=[10.0, 12.0],
        e2e_median_us=[9.0, 11.0],
        compute_us=[1000.0, 8.0],
        compute_median_us=[6.1234564, 7.1234566],
        runtime_metadata=[metadata, dict(metadata)],
    )
    bench._emit_row(
        args,
        scale_mode="mxfp4_hybrid",
        operand_order="swap_ab",
        tile=(128, 64),
        tokens=8,
        world_size=2,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    expected_stdout_header = [
        "BENCH_CSV",
        *bench.CSV_FIELDS.split(","),
        *bench.BENCH_EXT_CSV_FIELDS.split(","),
        *bench.SPLIT_RUNTIME_CSV_FIELDS.split(","),
        *bench.FORMAL_TUNING_CSV_FIELDS.split(","),
        *bench.FP8_RUNTIME_CSV_FIELDS.split(","),
        *bench.RUNTIME_TACTIC_CSV_FIELDS.split(","),
        *bench.ROUTING_CSV_FIELDS.split(","),
    ]
    assert header == expected_stdout_header
    assert len(row) == len(header)
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["kernel"] == bench.MXFP4_BACKEND
    assert values["tactic"] == (
        "swapab_m128n64k256_cga2x1x1_pp0_gh512_s2_atomic_counter_epi_warps"
    )
    assert values["scale_mode"] == "mxfp4_hybrid"
    assert values["tile_k"] == "256"
    assert values["cold_first_call_min_us"] == "100.00"
    assert values["cold_first_call_max_us"] == "110.00"
    assert values["cold_first_call_mean_us"] == "105.00"
    assert values["e2e_max_us"] == "12.00"
    assert values["compute_max_us"] == "1000.00"
    assert values["compute_max_rank_median_us"] == "7.123457"
    assert values["fused_pingpong"] == "0"
    assert values["fused_cga_m"] == "2"
    assert values["fused_cga_n"] == "1"
    assert values["fused_cga_k"] == "1"
    assert values["fused_group_hint"] == "512"
    assert values["fused_num_sched_stages"] == "2"
    assert values["fused_load_balance_mode"] == "atomic_counter"
    assert values["fused_token_back_mode"] == "epi_warps"
    assert values["runtime_group_hint"] == "512"
    assert values["runtime_num_sched_stages"] == "2"
    assert values["runtime_tactic_sha256"] == metadata["runtime_tactic_sha256"]
    assert values["routing_mode"] == "block_permutation"
    assert values["routing_profile"] == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert values["routing_seed"] == "1234"
    assert len(values["route_ids_sha256"]) == 64
    assert values["ref_csv"] == "not_applicable(mxfp4)"

    file_lines = csv_file.getvalue().strip().splitlines()
    assert len(file_lines) == 2
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
    assert file_header[: len(bench.CSV_FIELDS.split(","))] == (
        _HISTORICAL_FP8_CSV_FIELDS.split(",")
    )
    file_values = dict(zip(file_header, file_row, strict=True))
    assert file_values["tactic"] == values["tactic"]
    assert file_values["compute_max_rank_median_us"] == "7.123457"
    assert file_values["fused_group_hint"] == "512"


@pytest.mark.parametrize("status", ["failed", "skip_oom"])
def test_mxfp4_failure_rows_preserve_tactic_and_nan_score(bench, capsys, status):
    args = _mxfp4_args(
        bench,
        "--tokens",
        "512",
        "--mxfp4-mma-tiler",
        "256,32,256",
        "--mxfp4-cluster",
        "1,1,1",
        "--mxfp4-group-hint",
        "528",
        "--mxfp4-num-sched-stages",
        "2",
        "--no-mxfp4-pingpong",
        "--load-balance-mode",
        "static",
    )
    csv_file = StringIO()
    result = bench.PointResult(
        status=status,
        cold_us=[],
        e2e_us=[],
        e2e_median_us=[],
        compute_us=[],
        compute_median_us=[],
    )
    bench._emit_row(
        args,
        scale_mode="mxfp4_hybrid",
        operand_order="swap_ab",
        tile=(256, 32),
        tokens=512,
        world_size=4,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 2
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    assert len(row) == len(header)
    values = dict(zip(header[1:], row[1:], strict=True))
    expected_tactic = "swapab_m256n32k256_cga1x1x1_pp0_gh528_s2_static_epi_warps"
    assert values["status"] == status
    assert values["tile_k"] == "256"
    assert values["tactic"] == expected_tactic
    assert values["compute_max_rank_median_us"] == "nan"
    for field in (
        "e2e_min_us",
        "e2e_max_us",
        "e2e_mean_us",
        "e2e_median_us",
        "compute_min_us",
        "compute_max_us",
        "compute_mean_us",
        "compute_median_us",
        "cold_first_call_min_us",
        "cold_first_call_max_us",
        "cold_first_call_mean_us",
    ):
        assert values[field] == "nan"
    assert values["fused_group_hint"] == "528"
    assert values["fused_num_sched_stages"] == "2"
    assert values["fused_load_balance_mode"] == "static"

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    assert len(file_row) == len(file_header)
    file_values = dict(zip(file_header, file_row, strict=True))
    assert file_values["status"] == status
    assert file_values["tactic"] == expected_tactic
    assert file_values["compute_max_rank_median_us"] == "nan"


@pytest.mark.parametrize(
    ("argv", "match"),
    [
        (["--mxfp4-tactic-source", "cache_or_heuristic"], "requires --backend"),
        (["--split-k1-mma-tiler", "128,32,128"], "split tactic flags"),
        (["--split-counter-banks", "2"], "split tactic flags"),
        (["--split-enable-iket"], "split tactic flags"),
    ],
)
def test_fp8_rejects_explicit_mxfp4_only_cli(bench, argv, match):
    args = bench._parse_args(argv)
    with pytest.raises(ValueError, match=match):
        bench._resolve_sweep(args, world_size=4)


@pytest.mark.parametrize(
    "argv",
    [
        ["--split-k1-mma-tiler", "128,32,128"],
        ["--split-k2-mma-tiler", "128,32,128"],
        ["--split-k1-cluster", "1,1,1"],
        ["--split-k2-cluster", "1,1,1"],
        ["--split-k1-group-hint", "80"],
        ["--split-k2-group-hint", "52"],
        ["--split-k1-num-sched-stages", "2"],
        ["--split-k2-num-sched-stages", "2"],
        ["--split-k1-sm-count", "80"],
        ["--split-k2-sm-count", "52"],
        ["--split-counter-banks", "2"],
        ["--split-graph-variant", "cold_k0"],
        ["--split-enable-iket"],
    ],
)
def test_mxfp4_fused_rejects_every_split_tactic_flag(bench, argv):
    args = _mxfp4_args(bench, *argv)
    with pytest.raises(ValueError, match="split tactic flags require"):
        bench._resolve_sweep(args, world_size=4)


@pytest.mark.parametrize("value", [(True, 1, 1), [1.0, 1, 1]])
def test_runtime_tactic_triplets_reject_bool_and_float(bench, value):
    with pytest.raises(RuntimeError, match="three positive integers"):
        bench._runtime_positive_triplet(value, "tile")


def test_mxfp4_fused_cache_mode_reports_actual_runtime_tactic(bench, capsys):
    args = _mxfp4_args(
        bench,
        "--hidden",
        "128",
        "--intermediate",
        "128",
        "--num-experts",
        "2",
        "--top-k",
        "2",
        "--mxfp4-tactic-source",
        "cache_or_heuristic",
    )
    modes, orders, tile = bench._resolve_sweep(args, world_size=2)
    config = bench._megakernel_config(args, modes[0], orders[0], tile)
    assert config.knobs is None
    assert config.mma_tiler_mnk is None
    assert config.cluster_shape_mnk is None

    tactic = json.loads(
        json.dumps(
            _expected_fused_knobs(
                mma=(256, 32, 128),
                cluster=(2, 1, 1),
                group_hint=330,
                num_sched_stages=2,
            )
        )
    )
    metadata = {
        "routing_profile": SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        **bench._runtime_tactic_envelope("mxfp4_fused", tactic),
    }
    result = bench.PointResult(
        status="pass",
        cold_us=[100.0, 110.0],
        e2e_us=[10.0, 12.0],
        e2e_median_us=[9.0, 11.0],
        compute_us=[7.0, 8.0],
        compute_median_us=[6.0, 7.25],
        runtime_metadata=[metadata, dict(metadata)],
    )
    csv_file = StringIO()
    bench._emit_row(
        args,
        scale_mode=modes[0],
        operand_order=orders[0],
        tile=tile,
        tokens=64,
        world_size=2,
        result=result,
        header_done=False,
        csv_file=csv_file,
    )

    lines = capsys.readouterr().out.strip().splitlines()
    header = next(csv.reader([lines[0]]))
    row = next(csv.reader([lines[1]]))
    values = dict(zip(header[1:], row[1:], strict=True))
    assert values["tactic"] == "mxfp4_fused_cache_or_heuristic"
    assert values["tile_m"] == "256"
    assert values["tile_n"] == "32"
    assert values["tile_k"] == "128"
    assert values["fused_cga_m"] == "2"
    assert values["fused_group_hint"] == "330"
    assert values["fused_num_sched_stages"] == "2"
    assert values["compute_max_rank_median_us"] == "7.250000"
    assert values["runtime_tactic_sha256"] == metadata["runtime_tactic_sha256"]

    file_lines = csv_file.getvalue().strip().splitlines()
    file_header = next(csv.reader([file_lines[0]]))
    file_row = next(csv.reader([file_lines[1]]))
    file_values = dict(zip(file_header, file_row, strict=True))
    for field in (
        "tactic",
        "tile_m",
        "tile_n",
        "tile_k",
        "fused_group_hint",
        "runtime_tactic_sha256",
    ):
        assert file_values[field] == values[field]


def test_mxfp4_fused_cache_failure_does_not_report_default_tactic(bench, capsys):
    args = _mxfp4_args(
        bench,
        "--mxfp4-tactic-source",
        "cache_or_heuristic",
    )
    _, orders, tile = bench._resolve_sweep(args, world_size=4)
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
        scale_mode="mxfp4_hybrid",
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
    assert values["tactic"] == "mxfp4_fused_cache_or_heuristic"
    for field in ("tile_m", "tile_n", "tile_k"):
        assert values[field] == ""
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
    file_values = dict(zip(file_header, file_row, strict=True))
    for field in ("tile_m", "tile_n", "tile_k"):
        assert file_values[field] == ""
