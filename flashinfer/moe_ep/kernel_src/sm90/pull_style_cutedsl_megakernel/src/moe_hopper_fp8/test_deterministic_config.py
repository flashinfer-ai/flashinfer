# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""CPU tests of the Hopper FP8 MegaMoE ``deterministic`` switch: off by
default, rejected together with the in-kernel topk reduce (descriptor, kernel
and CLI), and wired to the deterministic dispatch region layout."""

import unittest

from common.megamoe_constants import SfPaddingBlock
from moe_hopper_fp8.heuristic_config import resolve_hopper_fp8_config
from moe_hopper_fp8.megamoe_kernel_fp8 import (
    Sm90MegaMoEFp8Kernel,
    Sm90MegaMoESwapABFp8Kernel,
)
from moe_hopper_fp8.runner_fc12 import ImplDesc, check_deterministic_config


def _kernel(scale_mode: str, tokens_per_rank: int, **kwargs):
    cfg = resolve_hopper_fp8_config(scale_mode, tokens_per_rank).config
    kernel_cls = Sm90MegaMoESwapABFp8Kernel if cfg.swap_ab else Sm90MegaMoEFp8Kernel
    m, n, _k = cfg.mma_tiler_mnk
    return kernel_cls(
        mma_tiler_mnk=cfg.mma_tiler_mnk,
        cluster_shape_mnk=cfg.cluster_shape_mnk,
        use_2cta_instrs=False,
        group_hint=1,
        token_padding_block=n if cfg.swap_ab else m,
        sf_padding_block=SfPaddingBlock,
        static_expert_shape=(8, 2048, 1024),
        fp8_scale_mode=scale_mode,
        fp8_accum_mode=cfg.accum_mode,
        pingpong=cfg.pingpong,
        world_size=4,
        local_rank=0,
        num_topk=4,
        max_tokens_per_rank=tokens_per_rank,
        hidden=1024,
        **kwargs,
    )


class CheckDeterministicConfigTest(unittest.TestCase):
    def test_combinations(self):
        check_deterministic_config(False, False)
        check_deterministic_config(False, True)
        check_deterministic_config(True, False)
        with self.assertRaises(ValueError):
            check_deterministic_config(True, True)


class ImplDescTest(unittest.TestCase):
    def _impl(self, **kwargs):
        cfg = resolve_hopper_fp8_config("per_tensor", 4096).config
        return ImplDesc(
            mma_tiler_mnk=cfg.mma_tiler_mnk,
            cluster_shape_mnk=cfg.cluster_shape_mnk,
            **kwargs,
        )

    def test_default_off(self):
        self.assertFalse(self._impl().deterministic)

    def test_deterministic_alone(self):
        self.assertTrue(self._impl(deterministic=True).deterministic)

    def test_rejects_in_kernel_reduce(self):
        self._impl(in_kernel_fc2_reduce=True)
        with self.assertRaises(ValueError):
            self._impl(deterministic=True, in_kernel_fc2_reduce=True)


class KernelTest(unittest.TestCase):
    CASES = (("per_tensor", 128), ("per_tensor", 4096), ("blockwise", 128), ("blockwise", 4096))

    def test_rejects_in_kernel_reduce(self):
        for mode, tokens in self.CASES:
            with self.subTest(mode=mode, tokens=tokens), self.assertRaises(ValueError):
                _kernel(mode, tokens, deterministic=True, fc2_in_kernel_topk_reduce=True)

    def test_dispatch_layout(self):
        for mode, tokens in self.CASES:
            with self.subTest(mode=mode, tokens=tokens):
                default = _kernel(mode, tokens)
                det = _kernel(mode, tokens, deterministic=True)
                self.assertFalse(default.deterministic)
                self.assertFalse(default.token_comm.deterministic_slots)
                self.assertNotIn("cta_expert_hist", default._local_region_by_name)
                self.assertTrue(det.token_comm.deterministic_slots)
                self.assertIn("cta_expert_hist", det._local_region_by_name)
                # One SMEM expert-count row per active dispatch warp.
                self.assertEqual(
                    det.token_comm.smem_expert_count_slots,
                    det.num_total_experts * det.token_comm.active_dispatch_warps,
                )
                self.assertEqual(
                    default.token_comm.smem_expert_count_slots, default.num_total_experts,
                )


class CliTest(unittest.TestCase):
    def test_parser_default_off(self):
        from moe_hopper_fp8.mega_runner import _build_arg_parser

        args = _build_arg_parser().parse_args([])
        self.assertFalse(args.deterministic)
        self.assertEqual(args.determinism_iters, 0)

    def test_rejects_in_kernel_reduce_before_launch(self):
        from moe_hopper_fp8.mega_runner import main

        with self.assertRaises(SystemExit) as ctx:
            main(["--deterministic", "--in_kernel_fc2_reduce"])
        self.assertEqual(ctx.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
