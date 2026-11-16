# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Real two-rank CPU collectives around legacy tuning preparation failures."""

from datetime import timedelta
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _next_allocation_failure(rank, architecture):
    import torch
    import torch.distributed as dist

    module = importlib.import_module(
        f"flashinfer.moe_ep.kernel_src.{architecture}.next_cutedsl_megamoe.shim.block_scaled"
    )
    rubin = architecture == "sm107"
    config_type = (
        module.Sm107BlockScaledMoeConfig if rubin else module.BlockScaledMoeConfig
    )
    buffer_type = (
        module.Sm107BlockScaledSymmBuffer if rubin else module.BlockScaledSymmBuffer
    )
    config = config_type(
        rank=rank,
        world_size=2,
        num_total_experts=2,
        max_tokens_per_rank=1,
        num_topk=1,
        hidden=512,
        intermediate=512,
        quant_kind="nvfp4",
        batch_invariant=True,
    )
    nbytes = config.padded_tokens_per_rank * config.num_topk * config.hidden * 2
    kernel = SimpleNamespace(
        _device_workspace=SimpleNamespace(
            total_bytes=lambda _region: nbytes,
            offset=lambda _region: 0,
            nbytes=lambda _region: nbytes,
        ),
        token_comm=SimpleNamespace(pre_reduced_activation_region="partials"),
    )
    workspace = object.__new__(buffer_type)
    workspace.config, workspace.kernel = config, kernel
    workspace._process_group = None
    workspace._destroyed = True
    zeros = torch.zeros
    old_local, old_shared = zeros(1), zeros(1)
    workspace.local_workspace, workspace.shared_workspace = old_local, old_shared
    workspace.output_activation = zeros(1)
    failing, symmetric_calls = True, 0

    def local_zeros(shape, **kwargs):
        if str(kwargs.get("device", "")).startswith("cuda"):
            if failing and rank == 1:
                raise RuntimeError("injected next-provider local allocation failure")
            kwargs["device"] = "cpu"
        # Preserve the actual allocator's alignment contract on CPU too.
        value = zeros(shape, **kwargs)
        extra = 128 // value.element_size()
        backing = zeros(value.numel() + extra, dtype=value.dtype)
        offset = (-backing.data_ptr() % 128) // value.element_size()
        return backing[offset : offset + value.numel()].view(value.shape)

    def symmetric_zeros(shape, dtype):
        nonlocal symmetric_calls
        symmetric_calls += 1
        dist.barrier()
        return local_zeros(shape, dtype=dtype)

    def allocate():
        if rubin:
            workspace._allocate_execution_state()
        else:
            workspace._allocate_kernel_workspace(kernel)

    with (
        patch.object(module.torch, "zeros", local_zeros),
        patch.object(module.comm, "sym_zeros", symmetric_zeros),
        patch.object(
            module.comm,
            "compute_peer_offsets",
            lambda tensor, world: (tensor.data_ptr(), (0,) * world),
        ),
    ):
        error = None
        try:
            allocate()
        except RuntimeError as exc:
            error = str(exc)
        peers = [None, None]
        dist.all_gather_object(peers, error)
        assert error and "injected next-provider local allocation failure" in error
        assert peers == [error, error]
        assert symmetric_calls == 0
        assert workspace.local_workspace is old_local
        assert workspace.shared_workspace is old_shared
        # The failed local phase must leave the group usable for another try.
        failing = False
        allocate()
        assert symmetric_calls == 1
        assert workspace.shared_workspace is not old_shared
    return dict(failure=f"{architecture}_local_allocation", error=error, retry=True)


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
        for failure in (
            "candidate_filter",
            "w4_candidate_filter",
            "local_prepare",
            "compile",
            None,
        ):
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
                if failure == "w4_candidate_filter":
                    from flashinfer.moe_ep.cute_dsl.megamoe.bf16_nvfp4 import (
                        MegaMoEBf16Nvfp4Config,
                        autotune_bf16_nvfp4_mega_moe,
                        bf16_nvfp4_candidates,
                    )
                    from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
                        tuner,
                    )

                    frontend.config = MegaMoEBf16Nvfp4Config(
                        rank=rank,
                        world_size=2,
                        num_tokens_per_rank=17,
                        num_topk=3,
                        num_total_experts=8,
                        hidden=384,
                        intermediate=192,
                        batch_invariant=True,
                    )
                    predicate = tuner.is_valid_bf16_nvfp4_for_config

                    def validate_w4(config, knobs):
                        if rank == 1:
                            raise RuntimeError(
                                "injected W4A16 candidate filter failure"
                            )
                        return predicate(config, knobs)

                    with patch.object(
                        tuner, "is_valid_bf16_nvfp4_for_config", validate_w4
                    ):
                        autotune_bf16_nvfp4_mega_moe(
                            None,
                            None,
                            None,
                            SimpleNamespace(_frontend=frontend),
                            candidates=bf16_nvfp4_candidates(batch_invariant=True)[:1],
                        )
                else:
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
        for architecture in ("sm100", "sm107"):
            results.append(_next_allocation_failure(rank, architecture))
        (directory / f"rank-{rank}.json").write_text(json.dumps(results, indent=2))
    finally:
        dist.destroy_process_group()


def test_two_rank_preparation_failures_do_not_enter_peer_launch(tmp_path):
    """A failed rank must stop its peer before storage collectives or launch."""
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("requires the CPU Gloo process-group backend")
    environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    environment.pop("MEGA_NO_DIST", None)
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
        assert len(results) == 7


if __name__ == "__main__":
    import torch.multiprocessing as mp

    if len(sys.argv) != 3 or sys.argv[1] != "--worker":
        raise SystemExit("use pytest or --worker DIRECTORY")
    mp.spawn(_rank_worker, args=(sys.argv[2],), nprocs=2, join=True)
