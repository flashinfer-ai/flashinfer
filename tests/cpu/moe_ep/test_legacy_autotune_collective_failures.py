# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Real two-rank CPU collectives around legacy tuning preparation failures."""

from datetime import timedelta
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


def _rank_worker(rank, directory):
    import torch.distributed as dist

    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe.shim.autotune import (
        _session_candidates,
        autotune_knobs,
    )

    directory = Path(directory)
    dist.init_process_group(
        "gloo",
        init_method=(directory / "rendezvous").as_uri(),
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=15),
    )
    results = []
    try:
        for failure in ("candidate_filter", "local_prepare", "compile", None):
            launches = 0

            class Frontend:
                active = None

                def prepare_tuning_local(self, knobs):
                    if failure == "local_prepare" and rank == 1:
                        raise RuntimeError("injected rank-local constructor failure")
                    return dict(knobs)

                def prepare_tuning_release(self, prepared):
                    dist.barrier()
                    self.active = None

                def prepare_tuning_allocate_local(self, prepared):
                    pass

                def prepare_tuning_collective(self, prepared):
                    # Model a symmetric-storage collective. Entering this on
                    # only one rank must fail this test instead of being hidden
                    # by a mock that returns immediately.
                    dist.barrier()
                    self.active = prepared

            frontend = Frontend()

            def launch():
                nonlocal launches
                launches += 1
                # Model a communication kernel's need for peer participation.
                dist.barrier()

            def prepare_launch():
                if failure == "compile" and rank == 1:
                    raise RuntimeError("injected rank-local compilation failure")
                return launch

            candidates = [{"plan": 0}, {"plan": 1}]

            def filter_candidates(proposed):
                return _session_candidates(
                    [] if failure == "candidate_filter" and rank == 1 else proposed,
                    None,
                    lambda _config, _knobs: True,
                    what="fault-injected MegaMoE",
                )

            winner, error = None, None
            try:
                winner = autotune_knobs(
                    frontend,
                    launch,
                    candidates,
                    label="cpu_fault_injection",
                    warmup_iters=1,
                    timed_iters=2,
                    prepare_launch=prepare_launch,
                    candidate_filter=filter_candidates,
                )
            except (RuntimeError, ValueError) as exc:
                error = str(exc)
            result = dict(
                failure=failure, launches=launches, winner=winner, error=error
            )
            peers = [None, None]
            dist.all_gather_object(peers, result)
            assert [p["error"] for p in peers] == [error, error]
            if failure:
                assert error is not None
                assert all(p["launches"] == 0 for p in peers)
            else:
                assert error is None
                assert all(p["winner"] == winner for p in peers)
                assert winner in candidates
                assert frontend.active == winner
                assert all(p["launches"] >= 6 for p in peers)
            results.append(result)
        (directory / f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


def test_two_rank_preparation_failures_do_not_enter_peer_launch(tmp_path):
    """A failed rank must stop its peer before storage collectives or launch."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires the CPU Gloo process-group backend")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", str(tmp_path)],
        cwd=Path(__file__).resolve().parents[3],
        env=environment,
        text=True,
        capture_output=True,
        timeout=65,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for rank in (0, 1):
        results = json.loads((tmp_path / f"rank-{rank}.json").read_text())
        assert len(results) == 4


if __name__ == "__main__":
    import torch.multiprocessing as mp

    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("use pytest or --worker DIRECTORY")
    mp.spawn(_rank_worker, args=(sys.argv[2],), nprocs=2, join=True)
