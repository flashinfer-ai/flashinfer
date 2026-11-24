# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full profiling under an autotune process group caches a winner only when
every rank confirmed the SM clock before every sample (real Gloo processes)."""

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
import flashinfer.autotuner.autotuner as autotuner_module
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune, set_autotune_process_group

rank = int(os.environ["RANK"])
world = int(os.environ["WORLD_SIZE"])
dist.init_process_group("gloo", init_method="file://" + os.environ["STORE"],
    rank=rank, world_size=world, timeout=datetime.timedelta(seconds=25))
torch.cuda.is_current_stream_capturing = lambda: False
torch.cuda.empty_cache = lambda: None
set_autotune_process_group(dist.group.WORLD)

RECOVERED = (2062, 2062, 0)
POWER_CAPPED = (1270, 2062, 0x4)
clock = {"state": RECOVERED}
# NVML readings of this rank; _wait_for_sm_clock itself is not patched.
autotuner_module._nvml_sm_clock_state = lambda device: clock["state"]
AutoTuner._FULL_PROFILING_CLOCK_TIMEOUT_S = 0.02
case = ""
calls = []


class Runner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def profile(self, runner, inputs, tactic, config, **kwargs):
    # Like _profile_single_kernel(full_profiling=True): one clock wait per
    # sample, then the group mean of the time, also when profiling fails.
    calls.append(tactic)
    for _ in range(3):
        if not autotuner_module._wait_for_sm_clock(
            torch.device("cpu"), self._FULL_PROFILING_CLOCK_TIMEOUT_S
        ):
            self._full_profiling_unconfirmed_samples += 1
    failed = case == "profile_error" and rank == 0 and tactic == 0
    elapsed = torch.tensor([float("inf") if failed else 1.0 + tactic], dtype=torch.float64)
    dist.all_reduce(elapsed)
    if failed:
        raise RuntimeError("profiling failed")
    return elapsed.item() / world


AutoTuner._profile_single_kernel = profile
config = TuningConfig(full_profiling=True)
inputs = [torch.empty((16, 32))]
runner = Runner()
tuner = AutoTuner.get()

# rank 0 has no NVML / a clock that stays capped / no NVML and a failed tactic;
# in "confirmed" every rank sees the clock recovered.
rank0_state = {"no_nvml": None, "timeout": POWER_CAPPED,
               "profile_error": None, "confirmed": RECOVERED}
for case, state in rank0_state.items():
    tuner.clear_cache()
    clock["state"] = state if rank == 0 else RECOVERED
    calls.clear()
    with autotune(True):
        chosen, tactic = tuner.choose_one("full_op", [runner], config, inputs)
    assert chosen is runner and calls == [0, 1, 2], (case, calls)
    expected = 1 if case == "profile_error" else 0
    assert tactic == expected, (case, tactic)
    hit = tuner.search_cache("full_op", [runner], ((16, 32),), config, inputs=inputs)[0]
    calls.clear()
    with autotune(True):
        again = tuner.choose_one("full_op", [runner], config, inputs)[1]
    assert again == expected, (case, again)
    if case == "confirmed":
        assert hit and calls == [], (case, hit, calls)
    else:
        # Not cached on any rank; the next full call measures again.
        assert not hit and calls == [0, 1, 2], (case, hit, calls)
    print(f"OK {rank} {case} tactic={tactic}", flush=True)

# Rejected full profiling fails on every rank at the same gate, before any
# measurement (no hang, nothing cached).
from flashinfer.autotune_cache import MeasurementPolicy
from flashinfer.autotuner.autotuner import _TuningGroupError

clock["state"] = RECOVERED
measure = AutoTuner._effective_measure_policy
rejected = {
    "cupti": lambda: tuner.choose_one("full_op", [runner], config, inputs),
    "rank_k2": lambda: tuner.rank_tactics("full_op", [runner], config, inputs, k=2),
}
for case, call in rejected.items():
    tuner.clear_cache()
    calls.clear()
    if case == "cupti":
        AutoTuner._effective_measure_policy = property(
            lambda self: MeasurementPolicy(_timer="cupti")
        )
    try:
        with autotune(True):
            call()
    except _TuningGroupError as error:
        assert "full_profiling" in str(error) or "another rank" in str(error), error
    else:
        raise AssertionError(f"{case}: full profiling was not rejected")
    finally:
        AutoTuner._effective_measure_policy = measure
    assert calls == [] and not tuner.profiling_cache, (case, calls)
    print(f"OK {rank} {case} rejected", flush=True)

set_autotune_process_group(None)
dist.destroy_process_group()
"""


def test_full_profiling_caches_only_when_every_rank_confirmed_the_clock(tmp_path):
    world_size = 2
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
            for case, tactic in (
                ("no_nvml", 0),
                ("timeout", 0),
                ("profile_error", 1),
                ("confirmed", 0),
            ):
                assert f"OK {rank} {case} tactic={tactic}" in output, output
            for case in ("cupti", "rank_k2"):
                assert f"OK {rank} {case} rejected" in output, output
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
        for process in processes:
            process.communicate()


if __name__ == "__main__":
    pytest.main([__file__])
