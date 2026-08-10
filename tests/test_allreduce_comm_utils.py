# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.routines.allreduce_comm_utils import (
    add_allreduce_control_args,
    aggregate_rank_times,
    append_jsonl,
    build_allreduce_control_kwargs,
    gather_process_group_initialization,
    gather_process_group_presence,
    gather_rank_errors,
    raise_if_rank0_error,
    select_rank_value,
    strategies_for_mode,
    strategy_request_name,
    summarize_times,
    timing_mode_request,
    validate_initialized_process_group,
)


class TestAllReduceCommUtils(unittest.TestCase):
    def test_control_defaults_preserve_behavior_except_intended_pdl_fix(self):
        parser = argparse.ArgumentParser()
        add_allreduce_control_args(parser)

        args = parser.parse_args([])

        self.assertEqual(args.strategy, "both")
        self.assertTrue(args.trigger_completion_at_end)
        self.assertFalse(args.fp32_acc)
        self.assertEqual(args.l2_cache, "cold")
        self.assertEqual(args.rank_aggregation, "max")
        self.assertIsNone(args.raw_jsonl_path)

    def test_control_arguments_expose_regression_inputs(self):
        parser = argparse.ArgumentParser()
        add_allreduce_control_args(parser)

        args = parser.parse_args(
            [
                "--strategy",
                "auto",
                "--no_trigger_completion_at_end",
                "--fp32_acc",
                "--l2_cache",
                "warm",
                "--rank_aggregation",
                "mean",
                "--raw_jsonl_path",
                "raw.jsonl",
            ]
        )

        self.assertEqual(args.strategy, "auto")
        self.assertFalse(args.trigger_completion_at_end)
        self.assertTrue(args.fp32_acc)
        self.assertEqual(args.l2_cache, "warm")
        self.assertEqual(args.rank_aggregation, "mean")
        self.assertEqual(args.raw_jsonl_path, "raw.jsonl")

    def test_strategy_modes_map_to_api_values(self):
        self.assertEqual(strategies_for_mode("oneshot"), [True])
        self.assertEqual(strategies_for_mode("twoshot"), [False])
        self.assertEqual(strategies_for_mode("both"), [True, False])
        self.assertEqual(strategies_for_mode("auto"), [None])
        self.assertEqual(strategy_request_name(True), "oneshot")
        self.assertEqual(strategy_request_name(False), "twoshot")
        self.assertEqual(strategy_request_name(None), "auto")

    def test_strategy_mode_rejects_unknown_value(self):
        with self.assertRaisesRegex(ValueError, "Unsupported AllReduce strategy"):
            strategies_for_mode("adaptive")

    def test_timing_mode_records_request_without_claiming_effective_mode(self):
        self.assertEqual(
            timing_mode_request(enable_cupti=True, use_cuda_graph=True), "cupti"
        )
        self.assertEqual(
            timing_mode_request(enable_cupti=True, use_cuda_graph=False), "cupti"
        )
        self.assertEqual(
            timing_mode_request(enable_cupti=False, use_cuda_graph=True),
            "cuda_graph",
        )
        self.assertEqual(
            timing_mode_request(enable_cupti=False, use_cuda_graph=False),
            "cuda_events",
        )

    def test_rank_aggregation_preserves_rank_local_samples(self):
        per_rank_times = [[1.0, 100.0, 4.0], [2.0, 3.0, 8.0]]
        self.assertEqual(aggregate_rank_times(per_rank_times, "max"), [2.0, 100.0, 8.0])
        self.assertEqual(
            aggregate_rank_times(per_rank_times, "rank0"), [1.0, 100.0, 4.0]
        )
        self.assertEqual(aggregate_rank_times(per_rank_times, "mean"), [1.5, 51.5, 6.0])

    def test_rank_aggregation_requires_equal_nonempty_vectors(self):
        for timings in ([], [[]], [[1.0], [2.0, 3.0]]):
            with self.subTest(timings=timings), self.assertRaises(ValueError):
                aggregate_rank_times(timings, "max")
        with self.assertRaisesRegex(ValueError, "Unsupported rank aggregation policy"):
            aggregate_rank_times([[1.0], [2.0]], "median")

    def test_timing_summary_reports_linear_p90(self):
        summary = summarize_times([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(summary["median_time"], 3.0)
        self.assertAlmostEqual(summary["p90_time"], 4.6)
        self.assertAlmostEqual(summary["std_time"], 2.0**0.5)
        with self.assertRaisesRegex(ValueError, "at least one sample"):
            summarize_times([])

    def test_timer_callback_selects_callers_rank(self):
        gathered = [0.125, 0.250, 0.500]
        self.assertEqual(select_rank_value(gathered, rank=1), 0.250)
        with self.assertRaisesRegex(ValueError, "outside a gathered world size"):
            select_rank_value(gathered, rank=3)

    def test_api_controls_propagate_pdl_and_precision(self):
        self.assertEqual(
            build_allreduce_control_kwargs(
                enable_pdl=False,
                trigger_completion_at_end=True,
                fp32_acc=False,
            ),
            {
                "launch_with_pdl": False,
                "trigger_completion_at_end": True,
                "fp32_acc": False,
            },
        )
        self.assertEqual(
            build_allreduce_control_kwargs(
                enable_pdl=True,
                trigger_completion_at_end=False,
                fp32_acc=True,
            ),
            {
                "launch_with_pdl": True,
                "trigger_completion_at_end": False,
                "fp32_acc": True,
            },
        )

    def test_initialized_process_group_must_match_mpi(self):
        validate_initialized_process_group(
            expected_rank=1,
            expected_world_size=2,
            actual_rank=1,
            actual_world_size=2,
        )

        with self.assertRaisesRegex(RuntimeError, "expected rank/world_size=1/2"):
            validate_initialized_process_group(
                expected_rank=1,
                expected_world_size=2,
                actual_rank=0,
                actual_world_size=2,
            )
        with self.assertRaisesRegex(RuntimeError, "got 1/4"):
            validate_initialized_process_group(
                expected_rank=1,
                expected_world_size=2,
                actual_rank=1,
                actual_world_size=4,
            )

    def test_process_group_initialization_is_collective_and_tracks_ownership(self):
        class FakeComm:
            def __init__(self, states):
                self.states = states

            def allgather(self, _value):
                return self.states

        created = {"error": None, "created": True, "initialized": True}
        status = gather_process_group_initialization(
            FakeComm([created, created]), None, True, True
        )
        self.assertTrue(status["ok"])
        self.assertTrue(status["created_by_benchmark"])

        caller_owned = {"error": None, "created": False, "initialized": True}
        status = gather_process_group_initialization(
            FakeComm([caller_owned, caller_owned]), None, False, True
        )
        self.assertTrue(status["ok"])
        self.assertFalse(status["created_by_benchmark"])

        mismatch = {
            "error": "RuntimeError: rank/world mismatch",
            "created": False,
            "initialized": True,
        }
        status = gather_process_group_initialization(
            FakeComm([caller_owned, mismatch]), None, False, True
        )
        self.assertFalse(status["ok"])
        self.assertIn("rank 1", status["error"])

        status = gather_process_group_initialization(
            FakeComm([created, caller_owned]), None, True, True
        )
        self.assertFalse(status["ok"])
        self.assertIn("ownership differs", status["error"])

    def test_process_group_presence_is_checked_before_nccl_initialization(self):
        class FakeComm:
            def __init__(self, initialized):
                self.initialized = initialized

            def allgather(self, _value):
                return self.initialized

        all_present = gather_process_group_presence(FakeComm([True, True]), True)
        self.assertTrue(all_present["ok"])
        self.assertTrue(all_present["all_initialized"])

        all_absent = gather_process_group_presence(FakeComm([False, False]), False)
        self.assertTrue(all_absent["ok"])
        self.assertFalse(all_absent["all_initialized"])

        mixed = gather_process_group_presence(FakeComm([True, False]), True)
        self.assertFalse(mixed["ok"])
        self.assertIn("rank(s) 0", mixed["error"])
        self.assertIn("rank(s) 1", mixed["error"])

    def test_rank_local_initialization_errors_are_collected(self):
        class FakeComm:
            def __init__(self, errors):
                self.errors = errors

            def allgather(self, _value):
                return self.errors

        self.assertIsNone(gather_rank_errors(FakeComm([None, None]), "MNNVL", None))
        error = gather_rank_errors(
            FakeComm([None, "RuntimeError: unavailable"]), "MNNVL", None
        )
        self.assertEqual(error, "MNNVL failed on rank 1: RuntimeError: unavailable")

    def test_append_jsonl_keeps_each_record(self):
        records = [
            {
                "strategy_request": "oneshot",
                "timing_mode_request": "cuda_events",
                "rank_aggregation": "max",
                "p90_time": 2.0,
                "per_rank_times": [[1.0], [2.0]],
            },
            {
                "strategy_request": "auto",
                "timing_mode_request": "cupti",
                "rank_aggregation": "rank0",
                "p90_time": 3.0,
                "per_rank_times": [[3.0], [4.0]],
            },
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "nested" / "raw.jsonl"
            append_jsonl(str(output_path), records[:1])
            append_jsonl(str(output_path), records[1:])
            actual = [
                json.loads(line)
                for line in output_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual(actual, records)

    def test_rank0_write_error_is_broadcast_before_raise(self):
        class FakeComm:
            def __init__(self, shared_error):
                self.shared_error = shared_error
                self.roots = []

            def bcast(self, value, root):
                self.roots.append(root)
                return self.shared_error

        failed_comm = FakeComm("raw timing write failed")
        with self.assertRaisesRegex(RuntimeError, "raw timing write failed"):
            raise_if_rank0_error(failed_comm, None)
        self.assertEqual(failed_comm.roots, [0])

        successful_comm = FakeComm(None)
        raise_if_rank0_error(successful_comm, None)
        self.assertEqual(successful_comm.roots, [0])


if __name__ == "__main__":
    unittest.main()
