"""CPU reference/validator regression; GPU protocol tests are separate.

Run directly: python moe_sm120_nvfp4_split/test_combine_reference.py
No CUDA, CuTeDSL or package initialization is needed.
"""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import random
import struct
import unittest
from unittest.mock import patch

import torch

try:
    from .combine_reference import (
        byte_difference, check_staging_delivery, enforce_combine_reports,
        flat_topk_bf16_reference, make_combine_report, numerical_comparison,
        owner_bf16_partials_reference, owner_grouped_bf16_reference,
        reconstruct_owner_from_staging,
    )
except ImportError:
    from combine_reference import (
        byte_difference, check_staging_delivery, enforce_combine_reports,
        flat_topk_bf16_reference, make_combine_report, numerical_comparison,
        owner_bf16_partials_reference, owner_grouped_bf16_reference,
        reconstruct_owner_from_staging,
    )


def f32(x):
    return struct.unpack("f", struct.pack("f", x))[0]


def bf16(x):
    bits = struct.unpack("I", struct.pack("f", x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack("f", struct.pack("I", bits))[0]


def scalar_grouped(terms, routes, world=4, epr=8):
    answer = []
    for row, ids in zip(terms.tolist(), routes.tolist()):
        output = []
        for h in range(terms.shape[2]):
            acc = 0.0
            for owner in range(world):
                partial = 0.0
                for slot, expert in enumerate(ids):
                    if expert // epr == owner:
                        partial = f32(partial + row[slot][h])
                acc = f32(acc + bf16(partial))
            output.append(bf16(acc))
        answer.append(output)
    return torch.tensor(answer, dtype=torch.bfloat16).reshape(terms.shape[0], terms.shape[2])


class CombineReferenceTests(unittest.TestCase):
    def cancellation(self):
        terms = torch.tensor(
            [[[-75264], [45824], [10816], [-20736], [35328], [4032]]],
            dtype=torch.bfloat16,
        )
        route = torch.tensor([[89, 41, 182, 82, 31, 143]])
        grouped = owner_grouped_bf16_reference(
            terms, route, world_size=4, num_experts_per_rank=64,
        )
        return terms, route, grouped

    def test_cancellation_owner_rounding(self):
        terms, route, grouped = self.cancellation()
        self.assertEqual(flat_topk_bf16_reference(terms).item(), 0)
        self.assertEqual(grouped.item(), -512)
        self.assertEqual(byte_difference(
            grouped, scalar_grouped(terms, route, epr=64)), 0)

    def test_counts_zero_one_two_many_and_absent_owner(self):
        terms = torch.tensor([[[1], [2], [4], [8], [16], [32]]], dtype=torch.bfloat16)
        route = torch.tensor([[0, 8, 9, 16, 17, 18]])  # counts [1,2,3,0]
        partial = owner_bf16_partials_reference(
            terms, route, world_size=4, num_experts_per_rank=8,
        )
        self.assertEqual(partial.flatten().tolist(), [1, 6, 56, 0])

    def test_order_not_tree_reduction(self):
        terms = torch.tensor([[[2**26], [1], [-2**26], [1]]], dtype=torch.bfloat16)
        route = torch.tensor([[0, 1, 2, 3]])
        self.assertEqual(flat_topk_bf16_reference(terms).item(), 1)
        self.assertEqual(owner_grouped_bf16_reference(
            terms, route, world_size=4, num_experts_per_rank=1).item(), 1)

    def test_strict_on_and_off_and_advisory(self):
        terms, _, grouped = self.cancellation()
        for rlc in (False, True):
            report = make_combine_report(
                grouped, flat_topk_bf16_reference(terms),
                grouped_reference=grouped if rlc else None,
                transport={"status": "PASS"},
            )
            self.assertEqual(report["ordinary_topk"]["status"], "FAIL")
            enforce_combine_reports([report], strict=False)
            with self.assertRaisesRegex(AssertionError, "ordinary_topk"):
                enforce_combine_reports([report], strict=True)
            report["staging_transport"]["status"] = "FAIL"
            with self.assertRaisesRegex(AssertionError, "staging_transport"):
                enforce_combine_reports([report], strict=False)

    def test_grouped_errors_never_advisory(self):
        x = torch.ones((2, 3), dtype=torch.bfloat16)
        report = make_combine_report(x + 1, x, grouped_reference=x)
        with self.assertRaisesRegex(AssertionError, "grouped_implementation"):
            enforce_combine_reports([report], strict=False)

    def test_nonfinite_and_missing_checks_are_not_pass(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            x = torch.tensor([[value]], dtype=torch.bfloat16)
            report = make_combine_report(x, x, grouped_reference=x)
            self.assertEqual(report["ordinary_topk"]["status"], "FAIL")
            self.assertEqual(report["grouped_implementation"]["status"], "FAIL")
        report = make_combine_report(torch.ones(1), torch.ones(1))
        self.assertEqual(report["staging_transport"]["status"], "NOT_CHECKED")

    def independent_report(self, actual, expected, transport=None):
        if transport is None:
            transport = {"status": "PASS", "source": "actual_FC2_BF16_staging",
                         "errors": {"partial_bytes": 0, "output_bytes": 0, "ready": 0}}
        return make_combine_report(
            torch.tensor([[actual]], dtype=torch.bfloat16),
            torch.tensor([[expected]], dtype=torch.bfloat16),
            grouped_reference=torch.tensor([[expected]], dtype=torch.bfloat16),
            transport=transport, independent_grouped_arithmetic=True)

    def test_independent_gemm_rounding_separate_from_exact_staging(self):
        report = self.independent_report(-38400, -38144)
        self.assertEqual(report["independent_grouped_byte_equivalence"]["status"], "FAIL")
        self.assertEqual(report["grouped_arithmetic"]["status"], "PASS")
        self.assertEqual(report["grouped_implementation"]["status"], "PASS")
        enforce_combine_reports([report], strict=False)

    def test_independent_accuracy_failure_is_mandatory(self):
        report = self.independent_report(100, 1)
        with self.assertRaisesRegex(AssertionError, "grouped_arithmetic"):
            enforce_combine_reports([report], strict=False)

    def test_exact_staging_cannot_be_replaced_by_tolerance(self):
        for name in ("partial_bytes", "output_bytes", "ready"):
            report = self.independent_report(-38400, -38144)
            report["staging_transport"]["status"] = "FAIL"
            report["staging_transport"]["errors"][name] = 1
            with self.assertRaisesRegex(AssertionError, "staging_transport"):
                enforce_combine_reports([report], strict=False)
            rebuilt = self.independent_report(-38400, -38144, report["staging_transport"])
            self.assertEqual(rebuilt["grouped_implementation"]["status"], "FAIL")

    def test_independent_grouped_requires_real_staging_oracle(self):
        for transport in ({}, {"status": "PASS"}, {"status": "NOT_CHECKED"}):
            report = self.independent_report(-38400, -38144, transport)
            with self.assertRaisesRegex(AssertionError, "grouped_implementation"):
                enforce_combine_reports([report], strict=False)

    def test_independent_grouped_nonfinite_rejected(self):
        for x in (float("nan"), float("inf"), -float("inf")):
            report = self.independent_report(x, x)
            with self.assertRaisesRegex(AssertionError, "grouped_arithmetic"):
                enforce_combine_reports([report], strict=False)

    def test_signed_zero_byte_comparison(self):
        a = torch.tensor([0.0], dtype=torch.bfloat16)
        b = torch.tensor([-0.0], dtype=torch.bfloat16)
        self.assertEqual(numerical_comparison(a, b)["different_elements"], 0)
        self.assertGreater(byte_difference(a, b), 0)

    def test_shape_and_invalid_route(self):
        for route in (torch.tensor([[-1]]), torch.tensor([[4]])):
            with self.assertRaises(ValueError):
                owner_grouped_bf16_reference(
                    torch.ones((1, 1, 2), dtype=torch.bfloat16),
                    route, world_size=4, num_experts_per_rank=1,
                )
        with self.assertRaises(ValueError):
            flat_topk_bf16_reference(torch.ones(3))

    def test_empty_tokens(self):
        result = owner_grouped_bf16_reference(
            torch.empty((0, 6, 3), dtype=torch.bfloat16),
            torch.empty((0, 6), dtype=torch.int64),
            world_size=4, num_experts_per_rank=8,
        )
        self.assertEqual(result.shape, (0, 3))

    def test_dynamic_routes_100_staging_permutations_and_stale_state(self):
        rng = random.Random(1234)
        old_partial = None
        for epoch in range(1, 101):
            # Owner counts and absent owners change; no cached contributor plan.
            routes = torch.tensor([rng.sample(range(32), 6) for _ in range(5)])
            terms = torch.tensor(
                [[[rng.randrange(-100000, 100000) for _ in range(3)]
                  for _ in range(6)] for _ in range(5)], dtype=torch.bfloat16,
            )
            independent = scalar_grouped(terms, routes)
            output = owner_grouped_bf16_reference(
                terms, routes, world_size=4, num_experts_per_rank=8,
            )
            self.assertEqual(byte_difference(output, independent), 0)
            reconstructed = []
            for owner in range(4):
                entries = [(t, s) for t in range(5) for s in range(6)
                           if routes[t, s] // 8 == owner]
                rng.shuffle(entries)
                staged = torch.zeros((max(1, len(entries)), 3), dtype=torch.bfloat16)
                metadata = torch.zeros(max(1, len(entries)), dtype=torch.int64)
                mapping = torch.full((5, 6), -1, dtype=torch.int64)
                for i, (t, s) in enumerate(entries):
                    staged[i] = terms[t, s]
                    mapping[t, s] = i
                    metadata[i] = (s << 32) | t
                p, errors = reconstruct_owner_from_staging(
                    staged, mapping, metadata, routes, owner=owner,
                    tokens_per_rank=5, experts_per_rank=8,
                )
                self.assertFalse(any(errors.values()), errors)
                reconstructed.append(p)
                if entries:
                    bad = mapping.clone()
                    bad[entries[0]] = 10**9
                    _, errors = reconstruct_owner_from_staging(
                        staged, bad, metadata, routes, owner=owner,
                        tokens_per_rank=5, experts_per_rank=8,
                    )
                    self.assertGreater(errors["row_bounds"], 0)
            partial = torch.stack(reconstructed, dim=1)
            active = torch.stack([(routes // 8 == o).any(1) for o in range(4)], dim=1)
            ready = torch.where(active, epoch, epoch - 1)
            kwargs = dict(active=active, epoch=epoch)
            self.assertFalse(any(check_staging_delivery(
                partial, partial, output, ready, **kwargs).values()))
            self.assertGreater(check_staging_delivery(
                partial, partial, output, ready - 1, **kwargs)["ready"], 0)
            if old_partial is not None:
                self.assertGreater(check_staging_delivery(
                    partial, old_partial, output, ready, **kwargs)["partial_bytes"], 0)
            old_partial = partial

    def test_runner_entry_strict_and_collective_peer_failure(self):
        # Exercise the real runner methods without importing CUDA/CuTeDSL.
        tree = ast.parse(Path(__file__).with_name("mega_runner.py").read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MegaMoETester")
        methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name in ("validate", "_agree_combine_report")]
        namespace = dict(torch=torch, os=os, json=json,
                         make_combine_report=make_combine_report,
                         enforce_combine_reports=enforce_combine_reports,
                         rank_local_staging_report=lambda _: {"status": "PASS", "source": "actual_FC2_BF16_staging", "errors": {"partial_bytes": 0, "output_bytes": 0, "ready": 0}})
        exec(compile(ast.Module(body=methods, type_ignores=[]), "<runner methods>", "exec"), namespace)
        from types import SimpleNamespace
        class Tester:
            validate = namespace["validate"]
            _agree_combine_report = namespace["_agree_combine_report"]
            _validate_token_back_exact = lambda _: {"status": "NOT_CHECKED"}
            def _ep_all_gather(self, outputs, value):
                for out in outputs:
                    out.copy_(value)
                if self.peer_failure:
                    outputs[1][1] = 1
        t = Tester()
        t.misc = SimpleNamespace(skip_ref_check=False)
        t.rank, t.world_size, t.peer_failure = 0, 2, False
        for name in ("combine_output", "combine_output_ref",
                     "combine_reduced_output_ref", "output_tensor_ref"):
            setattr(t, name, torch.zeros((1, 1), dtype=torch.bfloat16))
        t.output_tensor = torch.full((1, 1), -512, dtype=torch.bfloat16)
        t.rank_local_output_ref = t.output_tensor.clone()
        for rlc in (False, True):
            t.rank_local_combine = rlc
            with contextlib.redirect_stdout(io.StringIO()):
                with patch.dict(os.environ, MEGA_STRICT_TORCH_REF="0"):
                    t.validate()
                with patch.dict(os.environ, MEGA_STRICT_TORCH_REF="1"):
                    with self.assertRaisesRegex(AssertionError, "ordinary_topk"):
                        t.validate()
        # Local rank passes but another EP rank fails: both must raise.
        t.output_tensor.zero_()
        t.rank_local_output_ref.zero_()
        t.peer_failure = True
        with patch.dict(os.environ, MEGA_STRICT_TORCH_REF="1"):
            with contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(AssertionError, "rank 1 ordinary_topk"):
                    t.validate()

    def test_workspace_byte_metadata_and_one_four_group_counters(self):
        from types import SimpleNamespace as NS
        tree = ast.parse(Path(__file__).with_name("combine_validation.py").read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef))
        ns = dict(torch=torch, check_staging_delivery=check_staging_delivery,
                  reconstruct_owner_from_staging=reconstruct_owner_from_staging,
                  RANK_COMBINE_MAX_GROUPS=4)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), "<transport>", "exec"), ns)
        for hidden, groups in ((128, 1), (2048, 4)):
            staged = torch.ones((2, 1, hidden), dtype=torch.bfloat16)
            done = torch.zeros(12, dtype=torch.int32)
            for g in range(groups):
                done[2 + g * 2:4 + g * 2] = 1
            if groups > 1:
                done[10:12] = groups
            arrays = {
                "rank_combine_route_map": torch.tensor([[0], [1]], dtype=torch.int32),
                "rank_combine_route_output": staged,
                "token_src_metadata": torch.tensor([0, 1], dtype=torch.int64).view(torch.uint8).reshape(2, 8),
                "rank_combine_tile_done": done,
            }
            offsets, specs, buffers, cursor = {}, {}, [], 0
            for name, value in arrays.items():
                buf = value.contiguous().view(torch.uint8).flatten()
                offsets[name], specs[name] = cursor, NS(nbytes=buf.numel(), shape=value.shape)
                buffers.append(buf)
                cursor += buf.numel()
            routes = torch.zeros((1, 2, 1), dtype=torch.int64)
            count = torch.tensor([1, 1, 1], dtype=torch.int32)
            t = NS(
                rank=0, world_size=1, k2_mma_tiler_mnk=(64, 128, 128),
                problem=NS(num_tokens_per_rank=2, hidden=hidden, num_topk=1, num_experts_per_rank=1),
                local_workspace=torch.cat(buffers), shared_workspace=count.view(torch.uint8),
                _global_topk_idx=routes, my_topk_idx=routes[0],
                combine_output=staged.clone(), output_tensor=staged[:, 0].clone(),
                rank_combine_ready=torch.ones((3, 1), dtype=torch.int32),
                _tp_reduce_enabled=lambda: False,
                _kernel=NS(
                    pool_token_capacity=2, _local_region_by_name=specs, _local_offsets=offsets,
                    _shared_region_by_name={"rank_combine_route_count": NS(nbytes=12)},
                    _shared_offsets={"rank_combine_route_count": 0},
                ),
            )
            report = ns["rank_local_staging_report"](t)
            self.assertEqual(report["status"], "PASS", report)
            t.local_workspace[offsets["rank_combine_tile_done"] + 8] += 1
            report = ns["rank_local_staging_report"](t)
            self.assertEqual(report["status"], "FAIL", report)
            self.assertGreater(report["errors"]["group_counters"], 0)


if __name__ == "__main__":
    unittest.main()
