"""CPU-only contracts for the SM90 FP8/MXFP4 official benchmark."""

from __future__ import annotations

import contextlib
import importlib.util
import inspect
from io import StringIO
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import TestCase, main, mock
import sys


REPO = Path(__file__).resolve().parents[2]
BENCH_PATH = REPO / "benchmarks" / "bench_moe_ep_sm90_mega.py"
HISTORICAL_FP8_CSV_FIELDS = (
    "kernel,scale_mode,operand_order,tile_m,tile_n,tile_k,"
    "tokens_per_rank,topk,world_size,total_experts,local_experts,hidden,"
    "intermediate_downproj,intermediate_gateup,warmup,iters,status,"
    "e2e_min_us,e2e_max_us,e2e_mean_us,e2e_median_us,"
    "compute_min_us,compute_max_us,compute_mean_us,compute_median_us,"
    "fc1_flops_per_rank,fc2_flops_per_rank,total_flops_per_rank,"
    "critical_tflops_compute,critical_tflops_e2e,tok_s_e2e,ref_csv"
)


def _routing_modules(extra_moe_ep=None):
    flashinfer = ModuleType("flashinfer")
    flashinfer.__path__ = []
    moe_ep = ModuleType("flashinfer.moe_ep")
    moe_ep.__path__ = []
    routing = ModuleType("flashinfer.moe_ep.sm90_routing")

    def profile_from_mode(mode):
        return {
            "block_permutation": "block_permutation_v1",
            "published_exact_balanced": "published_exact_balanced_v1",
        }[mode]

    routing.sm90_routing_profile_from_benchmark_mode = profile_from_mode
    routing.normalize_sm90_routing_profile = lambda value: str(value)
    routing.generate_sm90_routing_numpy = lambda **kwargs: kwargs
    routing.generate_sm90_published_exact_balanced_routes_numpy = (
        lambda **kwargs: kwargs
    )
    routing.sm90_routing_audit_payload = lambda routes, **kwargs: {
        **kwargs,
        "route_ids_sha256": "0" * 64,
    }
    if extra_moe_ep:
        for name, value in extra_moe_ep.items():
            setattr(moe_ep, name, value)
    flashinfer.moe_ep = moe_ep
    moe_ep.sm90_routing = routing
    return {
        "flashinfer": flashinfer,
        "flashinfer.moe_ep": moe_ep,
        "flashinfer.moe_ep.sm90_routing": routing,
    }


def _load_benchmark():
    name = "_bench_moe_ep_sm90_mega_cpu_contract"
    spec = importlib.util.spec_from_file_location(name, BENCH_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _routing_modules()):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


bench = _load_benchmark()


class _ConfigCapture:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def _mxfp4_args(*extra):
    return bench._parse_args(["--backend", bench.MXFP4_BACKEND, *extra])


def _resolved_fused_metadata(args, *, store=False, early=True, fold=True):
    tactic = bench._mxfp4_fused_tactic(args, (256, 32))
    cluster = tactic["cluster_shape_mnk"]
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
        **{name: tactic[name] for name in bench._FUSED_EXECUTION_TACTIC_FIELDS},
    )
    kernel = SimpleNamespace(
        group_hint=tactic["group_hint"],
        num_sched_stages=tactic["num_sched_stages"],
        dedup_dispatch=tactic["dedup_dispatch"],
        grouped_token_back=tactic["grouped_token_back"],
        combine_format=SimpleNamespace(name=tactic["combine_format"]),
        token_comm=SimpleNamespace(
            active_dispatch_warps=tactic["active_dispatch_warps"]
        ),
        fc1_store_offload=store,
        fc1_early_done_publish=early,
        fold_producer_warps=fold,
    )
    effective_tactic = dict(tactic)
    effective_tactic.update(
        fc1_store_offload=store,
        fc1_early_done_publish=early,
        fold_producer_warps=fold,
    )
    frontend = SimpleNamespace(
        config=config,
        _mega=SimpleNamespace(kernel=kernel),
        requested_tactic=mock.Mock(return_value=tactic),
        effective_tactic=mock.Mock(return_value=effective_tactic),
    )
    return bench._mxfp4_fused_runtime_metadata(
        args, SimpleNamespace(_frontend=frontend)
    )


def _tensor_record(payload: bytes, **layout_overrides):
    layout = {
        "dtype": "uint8",
        "shape": (len(payload),),
        "stride": (1,),
        "storage_offset": 0,
        "numel": len(payload),
        "element_size": 1,
    }
    layout.update(layout_overrides)
    return bench._tensor_audit_from_chunks(chunks=[payload], **layout)


def _synthetic_rank_audit(rank, *, global_route="0" * 64, full=False):
    def record(section, name):
        return _tensor_record(f"r{rank}:{section}:{name}".encode())

    weights = None
    if full:
        weights = {
            "canonical": {
                "w13": record("canonical", "w13"),
                "w2": record("canonical", "w2"),
            },
            "transformed": {
                "fc1.weight": record("transformed", "fc1.weight"),
                "fc1.weight_sf": None,
                "fc2.weight": record("transformed", "fc2.weight"),
            },
        }
    return {
        "rank": rank,
        "route_ids": {
            "canonical_global_i64le_sha256": global_route,
            "canonical_local_sha256": f"canonical-{rank}",
            "input_local_sha256": f"canonical-{rank}",
            "staged_local_sha256": f"canonical-{rank}",
            "input_matches_canonical": True,
            "staged_matches_input": True,
        },
        "logical": {
            name: record("logical", name)
            for name in ("hidden_states", "topk_ids", "topk_weights")
        },
        "staged": {
            name: record("staged", name)
            for name in ("x", "x_sf", "topk_idx", "topk_weights")
        },
        "weights": weights,
    }


class BenchmarkContracts(TestCase):
    def test_historical_fp8_prefix_and_append_only_fields(self):
        self.assertEqual(bench.CSV_FIELDS, HISTORICAL_FP8_CSV_FIELDS)
        header = bench.CSV_HEADER.split(",")
        historical = HISTORICAL_FP8_CSV_FIELDS.split(",")
        self.assertEqual(header[0], "BENCH_CSV")
        self.assertEqual(header[1 : 1 + len(historical)], historical)
        self.assertIn("compute_max_rank_median_us", header)
        self.assertEqual(
            header[-10:],
            bench.FUSED_EXECUTION_RUNTIME_CSV_FIELDS.split(","),
        )
        self.assertNotIn("input_audit", header)

    def test_latest_fp8_defaults_and_cli_are_unchanged(self):
        args = bench._parse_args([])
        modes, orders, tile = bench._resolve_sweep(args, 4)
        self.assertEqual(args.scale_mode, "both")
        self.assertEqual(args.operand_order, "heuristic")
        self.assertEqual(modes, ("per_tensor", "blockwise"))
        self.assertEqual(orders, ("heuristic",))
        self.assertIsNone(tile)
        self.assertEqual(args.routing_mode, "block_permutation")
        self.assertEqual(args.compute_launch_mode, "direct")
        self.assertEqual(args.load_balance_mode, "atomic_counter")
        self.assertEqual(args.token_back, "heuristic")
        self.assertIsNone(bench._resolved_token_back(args))
        self.assertFalse(args.dedup_dispatch)
        self.assertFalse(args.grouped_token_back)
        self.assertEqual(args.combine_format, "bf16")
        self.assertEqual(args.active_dispatch_warps, 1)
        self.assertTrue(args.fc1_store_offload)
        self.assertFalse(args.fc1_early_done_publish)
        self.assertTrue(args.fold_producer_warps)
        self.assertEqual(args.input_audit, "none")
        self.assertEqual(
            bench._parse_args(["--input-audit", "point"]).input_audit,
            "point",
        )
        self.assertEqual(
            bench._parse_args(["--input-audit", "full"]).input_audit,
            "full",
        )

    def test_tensor_audit_streaming_and_descriptor_identity(self):
        payload = b"abcdefgh"
        contiguous = bench._tensor_audit_from_chunks(
            dtype="uint8",
            shape=(8,),
            stride=(1,),
            storage_offset=0,
            numel=8,
            element_size=1,
            chunks=[payload[:3], payload[3:6], payload[6:]],
        )
        one_chunk = _tensor_record(payload)
        self.assertEqual(contiguous, one_chunk)

        mutated = _tensor_record(payload[:-1] + b"Z")
        self.assertNotEqual(contiguous["content_sha256"], mutated["content_sha256"])
        relaid = _tensor_record(
            payload,
            shape=(2, 4),
            stride=(1, 2),
            storage_offset=7,
        )
        self.assertEqual(contiguous["content_sha256"], relaid["content_sha256"])
        self.assertNotEqual(contiguous["layout_sha256"], relaid["layout_sha256"])
        self.assertNotEqual(contiguous["tensor_sha256"], relaid["tensor_sha256"])
        with self.assertRaisesRegex(RuntimeError, "byte count mismatch"):
            bench._tensor_audit_from_chunks(
                dtype="uint8",
                shape=(8,),
                stride=(1,),
                storage_offset=0,
                numel=8,
                element_size=1,
                chunks=[b"short"],
            )

    def test_point_input_audit_is_rank_ordered_and_self_hashed(self):
        args = _mxfp4_args("--input-audit", "point", "--no-sparse-data")
        rank_audits = [_synthetic_rank_audit(1), _synthetic_rank_audit(0)]
        audit = bench._aggregate_input_audit(
            args,
            scale_mode="mxfp4_hybrid",
            tokens=8,
            world_size=2,
            rank_audits=rank_audits,
        )
        self.assertEqual(audit["implementation"], "mxfp4_fused")
        self.assertEqual(audit["level"], "point")
        self.assertIsNone(audit["weights"])
        self.assertEqual([record["rank"] for record in audit["ranks"]], [0, 1])
        payload = dict(audit)
        observed = payload.pop("audit_sha256")
        self.assertEqual(observed, bench._canonical_json_sha256(payload))
        self.assertEqual(
            audit["audit_sha256"],
            bench._aggregate_input_audit(
                args,
                scale_mode="mxfp4_hybrid",
                tokens=8,
                world_size=2,
                rank_audits=list(reversed(rank_audits)),
            )["audit_sha256"],
        )

    def test_full_input_audit_hashes_canonical_and_transformed_weights(self):
        args = _mxfp4_args("--input-audit", "full", "--no-sparse-data")
        audit = bench._aggregate_input_audit(
            args,
            scale_mode="mxfp4_hybrid",
            tokens=8,
            world_size=2,
            rank_audits=[
                _synthetic_rank_audit(0, full=True),
                _synthetic_rank_audit(1, full=True),
            ],
        )
        self.assertEqual(
            set(audit["weights"]),
            {"canonical_sha256", "transformed_sha256"},
        )
        self.assertTrue(all(audit["weights"].values()))

    def test_input_audit_fails_closed_on_route_or_output_mismatch(self):
        args = _mxfp4_args("--input-audit", "point")
        with self.assertRaisesRegex(RuntimeError, "disagree on canonical route"):
            bench._aggregate_input_audit(
                args,
                scale_mode="mxfp4_hybrid",
                tokens=8,
                world_size=2,
                rank_audits=[
                    _synthetic_rank_audit(0, global_route="a" * 64),
                    _synthetic_rank_audit(1, global_route="b" * 64),
                ],
            )
        with self.assertRaisesRegex(RuntimeError, "rank coverage"):
            bench._aggregate_input_audit(
                args,
                scale_mode="mxfp4_hybrid",
                tokens=8,
                world_size=2,
                rank_audits=[
                    _synthetic_rank_audit(0),
                    _synthetic_rank_audit(0),
                ],
            )

        audit = bench._aggregate_input_audit(
            args,
            scale_mode="mxfp4_hybrid",
            tokens=8,
            world_size=1,
            rank_audits=[_synthetic_rank_audit(0)],
        )
        result = bench.PointResult(
            "pass", [1.0], [2.0], [2.0], [3.0], [3.0], input_audit=audit
        )
        self.assertEqual(
            bench._validated_input_audit_json(args, result),
            bench._canonical_json(audit),
        )
        tampered = dict(audit)
        tampered["level"] = "full"
        result.input_audit = tampered
        with self.assertRaisesRegex(RuntimeError, "self-hash mismatch"):
            bench._validated_input_audit_json(args, result)

        result.status = "failed"
        with self.assertRaisesRegex(RuntimeError, "failed benchmark point"):
            bench._validated_input_audit_json(args, result)

    def test_input_audit_runs_only_after_timed_compute(self):
        source = inspect.getsource(bench._run_point)
        self.assertLess(
            source.index("compute = _time_calls"),
            source.index("rank_input_audit = _rank_input_audit"),
        )
        pre_timing = source[
            source.index("bench_backend.stage_inputs") : source.index(
                "compute = _time_calls"
            )
        ]
        self.assertNotIn("_rank_input_audit", pre_timing)

    def test_fp8_explicit_candidate_does_not_conflict_with_default_layout(self):
        candidate = {
            "swap_ab": True,
            "pingpong": False,
            "mma_tiler_mnk": (256, 32, 128),
            "cluster_shape_mnk": (1, 1, 1),
            "fp8_accum_mode": "1xacc",
            "token_back_mode": "epi_warps",
        }
        args = bench._parse_args(["--fp8-knobs-json", "{}"])
        with mock.patch.object(bench, "_parse_fp8_knobs_json", return_value=candidate):
            self.assertEqual(
                bench._resolve_sweep(args, 4),
                (("per_tensor", "blockwise"), ("swap_ab",), (256, 32)),
            )
        explicit_layout = bench._parse_args(["--fp8-knobs-json", "{}", "--swap-ab"])
        with (
            mock.patch.object(bench, "_parse_fp8_knobs_json", return_value=candidate),
            self.assertRaisesRegex(ValueError, "mutually exclusive"),
        ):
            bench._resolve_sweep(explicit_layout, 4)

    def test_neutral_and_hidden_legacy_aliases_are_identical(self):
        common = ("--tokens", "512", "--output-csv", "none")
        neutral = _mxfp4_args(
            *common,
            "--mma-tiler-mnk",
            "256,32,256",
            "--cluster-shape",
            "2,1,1",
            "--group-hint",
            "512",
            "--num-sched-stages",
            "2",
            "--pingpong",
            "off",
        )
        legacy = _mxfp4_args(
            *common,
            "--mxfp4-mma-tiler",
            "256,32,256",
            "--mxfp4-cluster",
            "2,1,1",
            "--mxfp4-group-hint",
            "512",
            "--mxfp4-num-sched-stages",
            "2",
            "--no-mxfp4-pingpong",
        )
        self.assertEqual(
            bench._resolve_sweep(neutral, 4), bench._resolve_sweep(legacy, 4)
        )
        self.assertEqual(
            bench._mxfp4_fused_tactic(neutral, (256, 32)),
            bench._mxfp4_fused_tactic(legacy, (256, 32)),
        )
        self.assertEqual(neutral.mma_tiler_mnk, legacy.mma_tiler_mnk)
        self.assertEqual(neutral.cluster_shape_mnk, legacy.cluster_shape_mnk)
        self.assertEqual(neutral.group_hint, legacy.group_hint)
        self.assertEqual(neutral.num_sched_stages, legacy.num_sched_stages)
        self.assertEqual(neutral.pingpong, legacy.pingpong)

    def test_alias_conflicts_fail_closed_and_are_hidden(self):
        args = _mxfp4_args(
            "--mma-tiler-mnk",
            "256,32,256",
            "--mxfp4-mma-tiler",
            "128,32,128",
        )
        with self.assertRaisesRegex(ValueError, "conflicting aliases"):
            bench._resolve_sweep(args, 4)
        output = StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(output):
            bench._parse_args(["--help"])
        help_text = output.getvalue()
        self.assertIn("--mma-tiler-mnk", help_text)
        self.assertNotIn("--mxfp4-mma-tiler", help_text)
        self.assertNotIn("--mxfp4-cluster", help_text)

    def test_fused_full_geometry_and_latest_seven_fields(self):
        args = _mxfp4_args(
            "--mma-tiler-mnk",
            "256,32,256",
            "--cluster-shape",
            "2,1,1",
            "--group-hint",
            "512",
            "--num-sched-stages",
            "2",
            "--pingpong",
            "off",
            "--dedup-dispatch",
            "--active-dispatch-warps",
            "2",
            "--no-fc1-store-offload",
            "--fc1-early-pub",
            "--no-fold-producer-warps",
        )
        self.assertEqual(
            bench._resolve_sweep(args, 4),
            (("mxfp4_hybrid",), ("swap_ab",), (256, 32)),
        )
        tactic = bench._mxfp4_fused_tactic(args, (256, 32))
        self.assertEqual(tactic["mma_tiler_mnk"], (256, 32, 256))
        self.assertEqual(tactic["cluster_shape_mnk"], (2, 1, 1))
        self.assertEqual(tactic["group_hint"], 512)
        self.assertEqual(tactic["num_sched_stages"], 2)
        self.assertFalse(tactic["pingpong"])
        self.assertTrue(tactic["dedup_dispatch"])
        self.assertFalse(tactic["grouped_token_back"])
        self.assertEqual(tactic["combine_format"], "bf16")
        self.assertEqual(tactic["active_dispatch_warps"], 2)
        self.assertFalse(tactic["fc1_store_offload"])
        self.assertTrue(tactic["fc1_early_done_publish"])
        self.assertFalse(tactic["fold_producer_warps"])
        self.assertEqual(
            set(tactic),
            bench.MXFP4_FUSED_RUNTIME_TACTIC_FIELDS,
        )
        modules = _routing_modules(
            {
                "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture,
            }
        )
        with mock.patch.dict(sys.modules, modules):
            config = bench._megakernel_config(
                args, "mxfp4_hybrid", "swap_ab", (256, 32), tokens=512
            )
        self.assertEqual(config.knobs, tactic)
        self.assertIsNone(config.load_balance_mode)
        self.assertEqual(config.execution_mode, "fused")
        self.assertEqual(config.routing_profile, "block_permutation_v1")

    def test_split_fields_enter_config_and_session_identity(self):
        args = _mxfp4_args(
            "--execution-mode",
            "split",
            "--split-k1-mma-tiler",
            "256,32,256",
            "--split-k2-mma-tiler",
            "128,64,128",
            "--split-k1-cluster",
            "2,1,1",
            "--split-k2-cluster",
            "1,1,1",
            "--split-k1-group-hint",
            "396",
            "--split-k2-group-hint",
            "132",
            "--split-k1-num-sched-stages",
            "3",
            "--split-k2-num-sched-stages",
            "2",
            "--split-k1-sm-count",
            "80",
            "--split-k2-sm-count",
            "52",
            "--split-counter-banks",
            "2",
            "--split-graph-variant",
            "cold_k0",
            "--split-enable-iket",
        )
        bench._resolve_sweep(args, 4)
        expected = bench._expected_split_session_config(args)
        self.assertEqual(set(expected), bench.MXFP4_SPLIT_RUNTIME_TACTIC_FIELDS)
        modules = _routing_modules(
            {
                "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture,
            }
        )
        with mock.patch.dict(sys.modules, modules):
            config = bench._megakernel_config(
                args, "mxfp4_hybrid", "swap_ab", (128, 32), tokens=512
            )
        self.assertEqual(config.split_k1_mma_tiler_mnk, (256, 32, 256))
        self.assertEqual(config.split_k2_mma_tiler_mnk, (128, 64, 128))
        self.assertEqual(config.split_k1_cluster_shape_mnk, (2, 1, 1))
        self.assertEqual(config.split_k2_cluster_shape_mnk, (1, 1, 1))
        self.assertEqual(config.split_k1_group_hint, 396)
        self.assertEqual(config.split_k2_group_hint, 132)
        self.assertEqual(config.split_k1_num_sched_stages, 3)
        self.assertEqual(config.split_k2_num_sched_stages, 2)
        self.assertEqual(config.split_k1_sm_count, 80)
        self.assertEqual(config.split_k2_sm_count, 52)
        self.assertEqual(config.split_counter_epoch_banks, 2)
        self.assertEqual(config.split_graph_variant, "cold_k0")
        self.assertTrue(config.split_enable_iket)
        self.assertIsNone(config.load_balance_mode)

    def test_mxfp4_cache_config_leaves_execution_selector_omitted(self):
        args = _mxfp4_args(
            "--mxfp4-tactic-source",
            "cache_or_heuristic",
        )
        bench._resolve_sweep(args, 4)
        modules = _routing_modules(
            {
                "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture,
            }
        )
        with mock.patch.dict(sys.modules, modules):
            config = bench._megakernel_config(
                args, "mxfp4_hybrid", "swap_ab", (128, 32), tokens=512
            )
        self.assertIsNone(config.knobs)
        self.assertIsNone(config.load_balance_mode)

    def test_compiled_effective_execution_fields_drive_runtime_sha(self):
        args = _mxfp4_args(
            "--mma-tiler-mnk",
            "256,32,256",
            "--cluster-shape",
            "2,1,1",
            "--group-hint",
            "512",
            "--num-sched-stages",
            "2",
        )
        metadata = _resolved_fused_metadata(
            args,
            store=False,
            early=True,
            fold=True,
        )
        requested = metadata["execution_knobs_requested"]
        effective = metadata["execution_knobs_effective"]
        self.assertTrue(requested["fc1_store_offload"])
        self.assertFalse(requested["fc1_early_done_publish"])
        self.assertFalse(effective["fc1_store_offload"])
        self.assertTrue(effective["fc1_early_done_publish"])
        self.assertEqual(metadata["runtime_tactic"]["fc1_store_offload"], False)
        self.assertEqual(
            metadata["runtime_tactic_sha256"],
            bench._canonical_runtime_tactic_sha256(
                "mxfp4_fused", metadata["runtime_tactic"]
            ),
        )
        result = bench.PointResult(
            "pass",
            [1.0] * 4,
            [2.0] * 4,
            [2.0] * 4,
            [3.0] * 4,
            [3.0] * 4,
            runtime_metadata=[metadata] * 4,
        )
        cols = bench._fused_execution_runtime_cols(args, result, 4)
        self.assertEqual(cols, ["0", "0", "bf16", "1", "1", "0", "0", "1", "1", "1"])

    def test_stdout_and_file_csv_rows_match_their_append_only_headers(self):
        args = bench._parse_args(["--output-csv", "none", "--swap-ab"])
        result = bench.PointResult("failed", [], [], [], [], [], error="")
        stdout = StringIO()
        csv_file = StringIO()
        with contextlib.redirect_stdout(stdout):
            bench._emit_row(
                args,
                scale_mode="per_tensor",
                operand_order="swap_ab",
                tile=(256, 32),
                tokens=8,
                world_size=4,
                result=result,
                header_done=False,
                csv_file=csv_file,
            )
        stdout_header, stdout_row = stdout.getvalue().strip().splitlines()
        self.assertEqual(len(stdout_header.split(",")), len(stdout_row.split(",")))
        file_header, file_row = csv_file.getvalue().strip().splitlines()
        self.assertEqual(len(file_header.split(",")), len(file_row.split(",")))

    def test_official_score_is_max_of_rank_local_medians(self):
        args = _mxfp4_args()
        result = bench.PointResult(
            "pass",
            [1.0] * 4,
            [2.0] * 4,
            [2.0] * 4,
            [3.0] * 4,
            [7.0, 10.0, 8.0, 9.0],
            runtime_metadata=None,
        )
        self.assertEqual(
            bench._formal_tuning_cols(args, (128, 32), result)[0], "10.000000"
        )

    def test_graph_is_explicit_fused_winner_replay_only(self):
        direct = _mxfp4_args()
        bench._resolve_sweep(direct, 4)
        self.assertEqual(bench._compute_launch_csv_cols(direct), ["direct", ""])

        graph = _mxfp4_args("--compute-launch-mode", "cuda_graph")
        bench._resolve_sweep(graph, 4)
        self.assertEqual(
            bench._compute_launch_csv_cols(graph),
            ["fused_external_cuda_graph", "3"],
        )

        cached = _mxfp4_args(
            "--compute-launch-mode",
            "cuda_graph",
            "--mxfp4-tactic-source",
            "cache_or_heuristic",
        )
        with self.assertRaisesRegex(ValueError, "explicit fused winner"):
            bench._resolve_sweep(cached, 4)

        split = _mxfp4_args(
            "--execution-mode",
            "split",
        )
        bench._resolve_sweep(split, 4)
        self.assertEqual(
            bench._compute_launch_csv_cols(split),
            ["split_internal_cuda_graph", ""],
        )

        split_graph = _mxfp4_args(
            "--execution-mode",
            "split",
            "--compute-launch-mode",
            "cuda_graph",
        )
        with self.assertRaisesRegex(ValueError, "split already owns"):
            bench._resolve_sweep(split_graph, 4)

    def test_split_rejects_fused_execution_axis(self):
        args = _mxfp4_args(
            "--execution-mode",
            "split",
            "--dedup-dispatch",
        )
        with self.assertRaisesRegex(ValueError, "fixed by split"):
            bench._resolve_sweep(args, 4)


if __name__ == "__main__":
    main()
