#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Minimal smoke test for :mod:`cutedsl_megamoe_front_end.megamoe_frontend`.

Reuses ``mega_runner`` for deterministic input + reference generation, then
launches via :class:`MegaMoENvfp4Frontend`.

Install once::

    cd cutedsl_megamoe/front_end && pip install -e .

Single-rank (no NVSHMEM)::

    cd cutedsl_megamoe
    MEGA_NO_DIST=1 CUDA_VISIBLE_DEVICES=0 python -u \\
        -m cutedsl_megamoe_front_end.megamoe_frontend.run \\
        --num_tokens_per_rank 128 --num_topk 4 --num_total_experts 32 \\
        --hidden 2048 --intermediate 1024

Multi-rank::

    cd cutedsl_megamoe
    PYTHONUNBUFFERED=1 torchrun --standalone --nproc_per_node=4 \\
        -m cutedsl_megamoe_front_end.megamoe_frontend.run \\
        --num_tokens_per_rank 256 --num_topk 4 --num_total_experts 32 \\
        --hidden 2048 --intermediate 1024 --route_distribution balanced
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

from . import (
    MegaMoENvfp4Config,
    MegaMoENvfp4Frontend,
    MegaMoENvfp4Inputs,
    bootstrap_dist,
)
from .common import free_sym_tensor
from moe_nvfp4_swapab.mega_runner import (
    _build_arg_parser,
    build_tester_from_args,
)


def _config_from_args(
    args: argparse.Namespace,
    *,
    rank: int,
    world_size: int,
) -> MegaMoENvfp4Config:
    def _parse_tuple(s: str) -> tuple[int, ...]:
        return tuple(int(x) for x in s.split(","))

    return MegaMoENvfp4Config(
        rank=rank,
        world_size=world_size,
        num_tokens_per_rank=args.num_tokens_per_rank,
        num_topk=args.num_topk,
        num_total_experts=args.num_total_experts,
        hidden=args.hidden,
        intermediate=args.intermediate,
        mma_tiler_mnk=_parse_tuple(args.mma_tiler_mnk),  # type: ignore[arg-type]
        cluster_shape_mnk=_parse_tuple(args.cluster_shape_mnk),  # type: ignore[arg-type]
        use_2cta_instrs=args.use_2cta_instrs,
        load_balance_mode=args.load_balance_mode,
        group_hint=args.group_hint,
        force_static_sched=not args.dynamic_sched,
        clc_bundle_size=args.clc_bundle_size,
        num_sched_stages=args.num_sched_stages,
        flag_batch=args.flag_batch,
        epi_flag_batch=_parse_tuple(args.epi_flag_batch),  # type: ignore[arg-type]
        non_ubulk_fc2_store=not args.use_bulk_fc2_store,
        in_kernel_fc2_reduce=args.in_kernel_fc2_reduce,
        token_back_mode=args.token_back_mode,
        combine_dtype=args.combine_dtype,
        apply_topk_in_fc1=args.ref_compute_graph == "deepgemm",
        gate_up_clamp=args.gate_up_clamp,
        enable_iket=args.enable_iket,
    )


def _inputs_from_tester(tester) -> MegaMoENvfp4Inputs:
    topk_score = (
        tester.my_topk_weights
        if tester.misc.ref_compute_graph == "transformers"
        else None
    )
    return MegaMoENvfp4Inputs(
        activation=tester.my_activation,
        activation_sf=tester.my_activation_sf,
        topk_idx=tester.my_topk_idx,
        topk_weights=tester.my_topk_weights,
        fc1_weight=tester.my_fc1_weight,
        fc1_weight_sf=tester.my_fc1_weight_sf,
        fc2_weight=tester.my_fc2_weight,
        fc2_weight_sf=tester.my_fc2_weight_sf,
        fc1_alpha=tester.my_fc1_alpha,
        fc2_alpha=tester.my_fc2_alpha,
        fc1_norm_const=tester.my_fc1_norm_const,
        combine_output=tester.combine_output,
        combine_reduced_output=tester.combine_reduced_output,
        combine_output_q=tester.combine_output_q,
        combine_sf_q=tester.combine_sf_q,
        combine_global_q=tester.combine_global_q,
        topk_score_for_reduce=topk_score,
    )


def _validate(tester, *, atol: float, rtol: float) -> bool:
    if tester.impl.fc2_reduces_topk:
        actual = tester.combine_output.squeeze(1).to(torch.float32)
        ref = tester.combine_output_ref.sum(dim=1).to(torch.float32)
    else:
        actual = tester.combine_reduced_output.to(torch.float32)
        ref = tester.combine_reduced_output_ref.to(torch.float32)
    ok = torch.allclose(actual, ref, atol=atol, rtol=rtol)
    if tester.rank == 0:
        diff = (actual - ref).abs()
        print(
            f"[validate] max_diff={diff.max().item():.4g} "
            f"mean_diff={diff.mean().item():.4g} pass={ok}"
        )
    return bool(ok)


def _free_tester_sym_tensors(tester) -> None:
    """Release NVSHMEM symmetric tensors owned by a ``MegaMoETester``."""
    for name in (
        "my_activation",
        "my_activation_sf",
        "my_topk_idx",
        "my_topk_weights",
        "combine_output",
        "shared_workspace",
    ):
        tensor = getattr(tester, name, None)
        free_sym_tensor(tensor)


def _finalize(runner: MegaMoENvfp4Frontend | None, tester) -> None:
    """Release frontend / tester resources and tear down NVSHMEM when active."""
    if runner is not None:
        runner.release()
    _free_tester_sym_tensors(tester)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    no_dist = bool(int(os.environ.get("MEGA_NO_DIST", "0")))
    if not no_dist and torch.distributed.is_initialized():
        torch.distributed.barrier()
        from src.bootstrap import finalize_dist_and_nvshmem

        finalize_dist_and_nvshmem()


def main(argv: list[str] | None = None) -> int:
    parser = _build_arg_parser()
    parser.description = "Minimal MegaMoENvfp4Frontend smoke test"
    parser.add_argument(
        "--val-atol",
        type=float,
        default=1e-2,
        help="Absolute tolerance for reference compare (default: 1e-2).",
    )
    parser.add_argument(
        "--val-rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for reference compare (default: 1e-2).",
    )
    args = parser.parse_args(argv)

    _, rank, world_size, _ = bootstrap_dist()
    tester = build_tester_from_args(args, rank=rank, world_size=world_size)
    cfg = _config_from_args(args, rank=rank, world_size=world_size)
    runner = MegaMoENvfp4Frontend(cfg)
    rc = 1

    try:
        if rank == 0:
            print(f"MegaMoENvfp4Frontend smoke: rank={rank} world_size={world_size}")
            print(cfg)

        tester.generate_inputs()
        if not args.skip_ref_check:
            tester.compute_reference()

        inputs = _inputs_from_tester(tester)
        runner.run(inputs)

        rc = 0
        if not args.skip_ref_check:
            if not _validate(tester, atol=args.val_atol, rtol=args.val_rtol):
                rc = 1
                if rank == 0:
                    print("FAIL: output mismatch vs reference", file=sys.stderr)
            elif rank == 0:
                print("PASS")

        if rank == 0:
            print("DONE")
    finally:
        _finalize(runner, tester)

    return rc


if __name__ == "__main__":
    sys.exit(main())
