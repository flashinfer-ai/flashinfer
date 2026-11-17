# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real Gloo processes must reject incompatible plans before timing reduces."""

import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


_WORKER = r"""
import datetime, os
import torch
import torch.distributed as dist
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune, set_autotune_process_group
from flashinfer.autotuner.autotuner import _TuningGroupError

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", init_method="file://" + os.environ["STORE"],
    rank=rank, world_size=world, timeout=datetime.timedelta(seconds=25))
torch.cuda.is_current_stream_capturing = lambda: False
torch.cuda.empty_cache = lambda: None
set_autotune_process_group(dist.group.WORLD)
config = TuningConfig(use_cold_l2_cache=False)
calls = []
case = ""

class BrokenIterable:
    def __iter__(self):
        raise RuntimeError("invalid tactic iterator")

class Inner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [] if case == "nested_empty" and rank == 1 else [0, 1]
    def forward(self, inputs, tactic=-1, **kwargs):
        return inputs[0]

class Runner(Inner):
    def get_cache_key_extras(self, inputs):
        return (rank,)
    def get_valid_tactics(self, inputs, profile):
        if case in ("nested_same", "nested_empty") or (case == "nested_different" and rank == 1):
            AutoTuner.get().rank_tactics("inner", [Inner()], config, inputs, k=2)
        if rank == 1:
            if case in ("candidates", "ranking_candidates"):
                return [0]
            if case == "candidate_order":
                return [1, 0]
            if case == "empty_candidates":
                return []
            if case == "candidate_error":
                raise ValueError("candidate discovery failed")
            if case == "candidate_encoding":
                return [object()]
            if case == "candidate_iterator":
                return [BrokenIterable()]
        return [0, 1]
    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        if rank == 1 and do_preparation and case in ("prepare_error", "ranking_prepare_error"):
            raise ValueError("runner preparation failed")
        return inputs[0]

class Other(Runner):
    pass

def profile(self, runner, inputs, tactic, config, **kwargs):
    calls.append((type(runner).__name__, tactic))
    elapsed = torch.tensor([1.0 + abs(tactic)], dtype=torch.float64)
    dist.all_reduce(elapsed)
    return elapsed.item() / world

AutoTuner._profile_single_kernel = profile
original_prepare = AutoTuner._prepare_input_tensors
original_profiles = AutoTuner._generate_optimization_profiles

def prepare(self, *args, **kwargs):
    if rank == 1 and case in ("input_error", "ranking_input_error"):
        raise ValueError("input preparation failed")
    return original_prepare(self, *args, **kwargs)

def profiles(self, *args, **kwargs):
    if rank == 1 and case == "profile_error":
        raise ValueError("profile generation failed")
    result = original_profiles(self, *args, **kwargs)
    if rank == 1 and case == "profiles":
        return result + result
    if rank == 1 and case == "ranking_profile_error":
        return []
    return result

AutoTuner._prepare_input_tensors = prepare
AutoTuner._generate_optimization_profiles = profiles
cases = ["op", "runner_order", "skip", "profiles", "profile_error", "policy",
         "candidates", "candidate_order", "empty_candidates", "candidate_error",
         "candidate_encoding", "input_error", "prepare_error", "nested_different",
         "ranking_candidates", "ranking_input_error", "ranking_prepare_error",
         "ranking_profile_error", "candidate_iterator", "nested_empty", "ranking_k",
         "method", "local_shapes", "nested_same", "all_skip"]
try:
    for case in cases:
        tuner = AutoTuner()
        AutoTuner._instance = tuner
        calls.clear()
        inputs = [torch.ones(2 + rank, 4)]
        runners = [Runner()]
        op = "op_other" if case == "op" and rank == 1 else "op"
        if case == "runner_order":
            runners = [Runner(), Other()] if rank == 0 else [Other(), Runner()]
        cfg = TuningConfig(use_cuda_graph=case == "policy" and rank == 1, use_cold_l2_cache=False)
        skipped = {op} if case == "all_skip" or (case == "skip" and rank == 1) else set()
        expected_failure = case not in ("local_shapes", "nested_same", "all_skip")
        failed = False
        try:
            with autotune(True, skip_ops=skipped):
                if case.startswith("ranking_") or (case == "method" and rank == 1):
                    tuner.rank_tactics(op, runners, cfg, inputs, k=1 if case == "ranking_k" and rank == 1 else 2)
                else:
                    _, tactic = tuner.choose_one(op, runners, cfg, inputs)
                    assert tactic == (-1 if case == "all_skip" else 0)
                    count = len(calls)
                    assert tuner.choose_one(op, runners, cfg, inputs)[1] == tactic
                    assert len(calls) == count, "warm hit profiled again"
        except _TuningGroupError as exc:
            failed = True
            assert "mismatch" in str(exc) or "failure" in str(exc), str(exc)
        assert failed == expected_failure, (rank, case, failed)
        if expected_failure or case == "all_skip":
            assert not calls, (rank, case, calls)
        else:
            assert calls
        # Every worker must reach the same next iteration after a rejected plan.
        dist.barrier()
        print("OK", rank, case, flush=True)
finally:
    set_autotune_process_group(None)
    dist.destroy_process_group()
"""


@pytest.mark.parametrize("world_size", [2, 4])
def test_distributed_phase_checks_and_rank_local_shapes(tmp_path, world_size):
    processes = []
    deadline = time.monotonic() + 120
    try:
        for rank in range(world_size):
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-c", _WORKER],
                    cwd=Path(__file__).resolve().parents[2],
                    env={
                        **os.environ,
                        "CUDA_VISIBLE_DEVICES": "",
                        "RANK": str(rank),
                        "WORLD_SIZE": str(world_size),
                        "STORE": str(tmp_path / "rendezvous"),
                        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": "0",
                    },
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            )
        for rank, process in enumerate(processes):
            output, _ = process.communicate(timeout=max(1, deadline - time.monotonic()))
            assert process.returncode == 0, f"rank {rank}:\n{output}"
            assert f"OK {rank} local_shapes" in output
            assert f"OK {rank} all_skip" in output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate()
