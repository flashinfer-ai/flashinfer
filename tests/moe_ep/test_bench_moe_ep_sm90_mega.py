"""CPU-only contracts for the SM90 FP8/MXFP4 official benchmark."""

from __future__ import annotations

import contextlib
import csv
import importlib.util
import json
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


class BenchmarkContracts(TestCase):
    def test_unambiguous_abbreviations_and_equals_resolve_to_full_names(self):
        for option, prefix, value in (
            ("--mma-tiler", "--mma", "128,64,256"),
            ("--scale-mode", "--scale-m", "blockwise"),
            ("--no-compact-pull-buffer", "--no-comp", None),
        ):
            full = [option] if value is None else [option, value]
            expected = vars(bench._parse_args(full))
            spellings = (
                [[prefix]]
                if value is None
                else [[prefix, value], [f"{prefix}={value}"]]
            )
            for argv in spellings:
                with self.subTest(argv=argv):
                    self.assertEqual(vars(bench._parse_args(argv)), expected)
        with contextlib.redirect_stderr(StringIO()), self.assertRaises(SystemExit):
            bench._parse_args(["--gr"])

    def test_mma_tiler_optional_k_preserves_backend_config(self):
        config_types = _routing_modules(
            {
                "Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture,
                "Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture,
            }
        )
        for backend in (bench.FP8_BACKEND, bench.MXFP4_BACKEND):
            values = (("256,32", 128), ("256,32,128", 128))
            if backend == bench.MXFP4_BACKEND:
                values += (("256,32,256", 256),)
            for value, tile_k in values:
                with self.subTest(backend=backend, value=value):
                    args = bench._parse_args(
                        ["--backend", backend, "--swap-ab", "--mma-tiler", value]
                    )
                    modes, orders, tile = bench._resolve_sweep(args, 4)
                    for mode in modes:
                        with mock.patch.dict(sys.modules, config_types):
                            config = bench._megakernel_config(
                                args, mode, orders[0], tile
                            )
                        mma = (
                            config.knobs["mma_tiler_mnk"]
                            if backend == bench.MXFP4_BACKEND
                            else config.mma_tiler_mnk
                        )
                        self.assertEqual(mma, (256, 32, tile_k))
        for mode in ("blockwise", "per_tensor"):
            args = bench._parse_args(
                ["--scale-mode", mode, "--swap-ab", "--mma-tiler", "256,32,256"]
            )
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "K=128"):
                bench._resolve_sweep(args, 4)
        for value in ("256", "256,32,128,1", "256,32,0", "256,32,k"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench._resolve_sweep(_mxfp4_args("--mma-tiler", value), 4)

    def test_abbreviated_mxfp4_options_cannot_bypass_applicability_checks(self):
        for flags, message in (
            (("--scale-m", "per_tensor"), "requires --scale-mode"),
            (("--scale-m=both",), "requires --scale-mode"),
            (("--generate",), "FP8-only"),
            (("--no-comp",), "FP8-only"),
            (
                (
                    "--mxfp4-tactic-source",
                    "cache_or_heuristic",
                    "--mma",
                    "128,64,256",
                ),
                "conflicts",
            ),
            (
                (
                    "--mxfp4-tactic-source",
                    "cache_or_heuristic",
                    "--mma=128,64,256",
                ),
                "conflicts",
            ),
            (
                (
                    "--mxfp4-tactic-source",
                    "cache_or_heuristic",
                    "--load-b=atomic_counter",
                ),
                "conflicts",
            ),
        ):
            with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, message):
                bench._resolve_sweep(_mxfp4_args(*flags), 4)

    def test_mxfp4_candidates_round_trip_through_json_and_config(self):
        from flashinfer.moe_ep.kernel_src.sm90 import (
            pull_style_cutedsl_megakernel as sm90,
        )

        candidates = sm90.hopper_mxfp4_optimization_candidates(
            2048,
            hidden=7168,
            intermediate=3072,
            num_experts=384,
            world_size=4,
        )
        self.assertTrue(any(c["tail_split_pairs"] for c in candidates))
        for candidate in candidates:
            with self.subTest(candidate=candidate):
                args = _mxfp4_args("--mxfp4-knobs-json", json.dumps(candidate))
                _, orders, tile = bench._resolve_sweep(args, 4)
                config = bench._megakernel_config(args, "mxfp4_hybrid", orders[0], tile)
                self.assertEqual(config.knobs, candidate)
                self.assertEqual(tile, candidate["mma_tiler_mnk"][:2])

    def test_fp8_runtime_record_uses_compiled_values_and_can_be_replayed(self):
        from flashinfer.moe_ep.kernel_src.sm90 import (
            pull_style_cutedsl_megakernel as sm90,
        )

        for scale in ("blockwise", "per_tensor"):
            tactic = next(
                c
                for c in sm90.hopper_fp8_candidates(
                    fp8_scale_mode=scale, max_tokens=2048
                )
                if c["tail_split_pairs"]
            )
            tactic = dict(bench._mxfp4_fused_tactic(_mxfp4_args()), **tactic)
            config = SimpleNamespace(
                **dict(
                    tactic,
                    flag_batch=1,
                    epi_flag_batch=(1, 1),
                    compact_pull_buffer=True,
                    generate_c=False,
                    resolved_token_back_mode=tactic["token_back_mode"],
                ),
            )
            effective = dict(
                tactic,
                group_hint=264,
                num_sched_stages=2,
                fc1_store_offload=False,
                fc1_early_done_publish=True,
            )
            kernel = SimpleNamespace(
                **effective, token_comm=SimpleNamespace(active_dispatch_warps=1)
            )
            workspace = SimpleNamespace(
                _frontend=SimpleNamespace(
                    config=config, _mega=SimpleNamespace(kernel=kernel)
                )
            )
            args = bench._parse_args(["--scale-mode", scale])
            record = bench._runtime_tactic(args, workspace)
            self.assertTrue(record["tail_split_pairs"])
            self.assertFalse(record["fc1_store_offload"])
            self.assertTrue(record["fc1_early_done_publish"])
            self.assertEqual(record["group_hint"], 264)
            replay = bench._parse_args(["--fp8-knobs-json", json.dumps(record)])
            bench._resolve_sweep(replay, 4)
            self.assertEqual(replay.knobs, record)

    def test_mxfp4_strategy_flags_rejected_on_fp8_and_cache(self):
        for prefix in (
            [],
            [
                "--backend",
                bench.MXFP4_BACKEND,
                "--mxfp4-tactic-source",
                "cache_or_heuristic",
            ],
        ):
            args = bench._parse_args([*prefix, "--mxfp4-fc2-tail-n8"])
            with self.assertRaises(ValueError):
                bench._resolve_sweep(args, 4)

    def test_historical_fp8_prefix_and_direct_fields(self):
        self.assertEqual(bench.CSV_FIELDS, HISTORICAL_FP8_CSV_FIELDS)
        header = bench.CSV_HEADER.split(",")
        historical = HISTORICAL_FP8_CSV_FIELDS.split(",")
        self.assertEqual(header[0], "BENCH_CSV")
        self.assertEqual(header[1 : 1 + len(historical)], historical)
        self.assertIn("compute_max_rank_median_us", header)
        self.assertEqual(header[-1], "runtime_tactic")
        self.assertNotIn("input_audit", header)
        self.assertIn("compute_launch_mode", header)
        self.assertFalse(any("graph" in field or "split" in field for field in header))

    def test_main_fp8_controls_are_forwarded(self):
        args = bench._parse_args(
            ["--swap-ab", "--cga", "2,1", "--generate-c", "--no-compact-pull-buffer"]
        )
        with mock.patch.dict(
            sys.modules,
            _routing_modules(
                {"Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig": _ConfigCapture}
            ),
        ):
            config = bench._megakernel_config(args, "per_tensor", "swap_ab", (256, 8))
        self.assertEqual(config.cluster_shape_mnk, (2, 1, 1))
        self.assertEqual(config.mma_tiler_mnk, (256, 8, 128))
        self.assertTrue(config.generate_c)
        self.assertFalse(config.compact_pull_buffer)
        self.assertFalse(config.enable_in_kernel_fc2_reduce)

    def test_main_fp8_only_controls_rejected_for_mxfp4(self):
        for flags in [
            ("--swap-token-tile", "8"),
            ("--generate-c",),
            ("--compact-pull-buffer",),
            ("--no-compact-pull-buffer",),
        ]:
            with (
                self.subTest(flags=flags),
                self.assertRaisesRegex(ValueError, "FP8-only"),
            ):
                bench._resolve_sweep(_mxfp4_args(*flags), 4)

    def test_latest_fp8_defaults_and_cli_are_unchanged(self):
        args = bench._parse_args([])
        modes, orders, tile = bench._resolve_sweep(args, 4)
        self.assertEqual(args.scale_mode, "both")
        self.assertEqual(args.operand_order, "heuristic")
        self.assertEqual(modes, ("per_tensor", "blockwise"))
        self.assertEqual(orders, ("heuristic",))
        self.assertIsNone(tile)
        self.assertEqual(args.routing_mode, "block_permutation")
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
        self.assertTrue(args.compact_pull_buffer)
        self.assertFalse(args.generate_c)
        self.assertIsNone(args.swap_token_tile)
        self.assertIsNone(args.cga)

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
        with mock.patch.object(bench, "_parse_knobs_json", return_value=candidate):
            self.assertEqual(
                bench._resolve_sweep(args, 4),
                (("per_tensor", "blockwise"), ("swap_ab",), (256, 32)),
            )
        explicit_layout = bench._parse_args(["--fp8-knobs-json", "{}", "--swap-ab"])
        with (
            mock.patch.object(bench, "_parse_knobs_json", return_value=candidate),
            self.assertRaisesRegex(ValueError, "mutually exclusive"),
        ):
            bench._resolve_sweep(explicit_layout, 4)

    def test_fp8_candidates_round_trip_through_json_and_config(self):
        from flashinfer.moe_ep.kernel_src.sm90 import (
            pull_style_cutedsl_megakernel as sm90,
        )

        for scale_mode in ("blockwise", "per_tensor"):
            candidates = sm90.hopper_fp8_candidates(
                fp8_scale_mode=scale_mode, max_tokens=2048
            )
            self.assertEqual({c["tail_split_pairs"] for c in candidates}, {False, True})
            for candidate in candidates:
                with self.subTest(scale_mode=scale_mode, candidate=candidate):
                    payload = json.dumps(candidate)
                    self.assertEqual(
                        bench._parse_knobs_json(payload, bench.FP8_BACKEND), candidate
                    )
                    args = bench._parse_args(["--fp8-knobs-json", payload])
                    _, orders, tile = bench._resolve_sweep(args, 4)
                    config = bench._megakernel_config(
                        args, scale_mode, orders[0], tile, tokens=2048
                    )
                    self.assertEqual(config.knobs, candidate)

    def test_fp8_json_tail_split_pairs_requires_boolean(self):
        from flashinfer.moe_ep.kernel_src.sm90 import (
            pull_style_cutedsl_megakernel as sm90,
        )

        candidate = sm90.hopper_fp8_candidates(max_tokens=2048)[0]
        for value in (None, 0, 1, "false", [], {}):
            with (
                self.subTest(value=value),
                self.assertRaisesRegex(ValueError, "tail_split_pairs must be boolean"),
            ):
                bench._parse_knobs_json(
                    json.dumps({**candidate, "tail_split_pairs": value}),
                    bench.FP8_BACKEND,
                )

    def test_mxfp4_cga_conflicts_with_cache_selection(self):
        args = _mxfp4_args(
            "--mxfp4-tactic-source", "cache_or_heuristic", "--cga", "1,2"
        )
        with self.assertRaisesRegex(ValueError, "conflicts with --cga"):
            bench._resolve_sweep(args, 4)

    def test_fused_full_geometry_and_latest_seven_fields(self):
        args = _mxfp4_args(
            "--mma-tiler",
            "256,32,256",
            "--cga",
            "2,1",
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
        tactic = bench._mxfp4_fused_tactic(args)
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
        self.assertEqual(config.routing_profile, "block_permutation_v1")

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

    def test_mxfp4_effective_tactic_is_used_for_replay(self):
        args = _mxfp4_args(
            "--mma-tiler",
            "256,32,256",
            "--group-hint",
            "512",
            "--num-sched-stages",
            "2",
        )
        bench._resolve_sweep(args, 4)
        effective = dict(
            args.knobs, fc1_store_offload=False, fc1_early_done_publish=True
        )
        frontend = SimpleNamespace(effective_tactic=mock.Mock(return_value=effective))
        record = bench._runtime_tactic(args, SimpleNamespace(_frontend=frontend))
        frontend.effective_tactic.assert_called_once_with()
        replay = _mxfp4_args("--mxfp4-knobs-json", json.dumps(record))
        bench._resolve_sweep(replay, 4)
        self.assertEqual(replay.knobs, effective)

    def test_json_replay_rejects_conflicting_manual_and_cache_settings(self):
        for backend, option in (
            (bench.FP8_BACKEND, "--fp8-knobs-json"),
            (bench.MXFP4_BACKEND, "--mxfp4-knobs-json"),
        ):
            for flags in (
                ("--pingpong", "on"),
                ("--load-balance-mode", "static"),
                ("--active-dispatch-warps", "2"),
                ("--fc1-early-pub",),
                ("--mxfp4-tactic-source", "cache_or_heuristic"),
            ):
                with (
                    self.subTest(backend=backend, flags=flags),
                    self.assertRaises(ValueError),
                ):
                    args = bench._parse_args(
                        ["--backend", backend, option, "{}", *flags]
                    )
                    bench._resolve_sweep(args, 4)

    def test_rank_agreement_compares_the_full_tactic(self):
        tactic = dict(
            bench._mxfp4_fused_tactic(_mxfp4_args()), group_hint=132, num_sched_stages=1
        )
        stats = [("pass", 2.0, 2.0, 3.0, 3.0, dict(tactic), "") for _ in range(4)]
        self.assertEqual(bench._summarize_ranks(stats).runtime_tactic, tactic)
        stats[-1][5]["tail_split_pairs"] = True
        failed = bench._summarize_ranks(stats)
        self.assertEqual(failed.status, "failed")
        self.assertIn("differs across ranks", failed.error)
        stats[-1] = ("skip_oom", 0, 0, 0, 0, {}, "out of memory")
        self.assertEqual(bench._summarize_ranks(stats).status, "skip_oom")

    def test_stdout_and_file_csv_round_trip_and_historical_statistics(self):
        args = _mxfp4_args()
        bench._resolve_sweep(args, 4)
        tactic = dict(args.knobs, group_hint=132, num_sched_stages=1)
        for status in ("pass", "failed", "skip_oom"):
            result = bench.PointResult(
                status,
                [10.0, 20.0, 30.0, 40.0],
                [8.0, 18.0, 28.0, 38.0],
                [2.0, 4.0, 6.0, 8.0],
                [7.0, 10.0, 8.0, 9.0],
                runtime_tactic=tactic if status == "pass" else None,
            )
            stdout, csv_file = StringIO(), StringIO()
            with contextlib.redirect_stdout(stdout):
                bench._emit_row(
                    args,
                    scale_mode="mxfp4_hybrid",
                    operand_order="swap_ab",
                    tile=(128, 32),
                    tokens=8,
                    world_size=4,
                    result=result,
                    header_done=False,
                    csv_file=csv_file,
                )
            for text in (stdout.getvalue(), csv_file.getvalue()):
                header, row = list(csv.reader(StringIO(text)))
                values = dict(zip(header, row, strict=True))
                self.assertEqual(values["compute_launch_mode"], "direct")
                self.assertEqual(values["routing_seed"], str(bench.ROUTING_SEED))
                if status == "pass":
                    self.assertEqual(
                        json.loads(values["runtime_tactic"]),
                        json.loads(json.dumps(tactic)),
                    )
                    self.assertEqual(values["e2e_mean_us"], "25.00")
                    self.assertEqual(values["e2e_median_us"], "23.00")
                    self.assertEqual(values["compute_max_us"], "8.00")
                    self.assertEqual(values["compute_median_us"], "8.50")
                    self.assertEqual(values["compute_max_rank_median_us"], "10.000000")
                else:
                    self.assertEqual(values["runtime_tactic"], "")
                    self.assertEqual(values["compute_mean_us"], "nan")

    def test_timing_invokes_each_sample_directly_with_cuda_events(self):
        calls = []
        start = SimpleNamespace(
            record=lambda: calls.append("start"),
            elapsed_time=lambda stop: 0.25,
        )
        stop = SimpleNamespace(record=lambda: calls.append("stop"))
        cuda = SimpleNamespace(
            Event=mock.Mock(side_effect=(start, stop)),
            synchronize=lambda: calls.append("sync"),
        )
        torch = ModuleType("torch")
        torch.cuda = cuda
        dist = ModuleType("torch.distributed")
        dist.barrier = lambda: calls.append("barrier")
        torch.distributed = dist
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.distributed": dist}):
            samples = bench._time_calls(lambda: calls.append("call"), warmup=2, iters=3)
        self.assertEqual(samples, [250.0] * 3)
        self.assertEqual(calls[:4], ["call", "call", "sync", "barrier"])
        self.assertEqual(
            calls[4:], ["barrier", "sync", "start", "call", "stop", "sync"] * 3
        )


if __name__ == "__main__":
    main()
