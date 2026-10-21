"""CPU checks for benchmark routing and ragged-workspace capacity."""

import importlib.util
import math
import sys
from types import SimpleNamespace
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load("bench", ROOT / "benchmarks/bench_gdn_sm120_pipeline32.py")
layout = load(
    "layout", ROOT / "flashinfer/gdn_kernels/delta_rule_dsl/sm120_pipeline32_layout.py"
)


dispatch = load(
    "dispatch",
    ROOT / "flashinfer/gdn_kernels/delta_rule_dsl/sm120_pipeline32_dispatch.py",
)


class SubmissionTests(unittest.TestCase):
    def test_opt_in_is_disabled_by_default(self):
        with patch.dict("os.environ", {}, clear=True):
            self.assertFalse(dispatch.sm120_pipeline32_prefill_enabled())
        with patch.dict(
            "os.environ", {"FLASHINFER_GDN_SM120_PIPELINE32": "1"}, clear=True
        ):
            self.assertTrue(dispatch.sm120_pipeline32_prefill_enabled())
        with patch.dict(
            "os.environ",
            {
                "FLASHINFER_GDN_SM120_PIPELINE32": "0",
                "VLLM_GDN_FLASHINFER_PIPELINE32": "1",
            },
            clear=True,
        ):
            self.assertFalse(dispatch.sm120_pipeline32_prefill_enabled())

        with patch.dict(
            "os.environ", {"VLLM_GDN_FLASHINFER_PIPELINE32": "1"}, clear=True
        ):
            self.assertFalse(dispatch.sm120_pipeline32_prefill_enabled())

    def test_paths_are_distinct_and_p32_is_not_cp(self):
        self.assertEqual(bench.paths("all"), ["fuse", "cp", "p32"])
        self.assertEqual(bench.paths("cp32"), ["p32"])

        def unsupported(**kwargs):
            pass

        with self.assertRaises(RuntimeError):
            bench.require_use_cp(unsupported)

        def supported(use_cp="auto"):
            pass

        bench.require_use_cp(supported)

    def test_nonfinite_accuracy_fails(self):
        for value in (math.nan, math.inf, -math.inf, 0.03):
            self.assertFalse(bench.accepted([0.005, value], 0.02))
        self.assertTrue(bench.accepted([0.005, 0.006], 0.02))

    def test_cupti_timing_uses_cold_l2_independent_iterations(self):
        recorded = {}

        def timing(fn, **kwargs):
            recorded.update(kwargs)
            return [3.0, 1.0, 2.0]

        with (
            patch.dict(
                sys.modules,
                {
                    "flashinfer.testing": SimpleNamespace(bench_gpu_time=timing),
                    "cupti": SimpleNamespace(cupti=object()),
                },
            ),
            patch.object(bench.importlib.metadata, "version", return_value="13.0.0"),
        ):
            result = bench.elapsed_ms(
                lambda: None, SimpleNamespace(backend="cupti", warmup=2, repeats=3)
            )
        self.assertEqual(result, 2.0)
        self.assertTrue(recorded["enable_cupti"])
        self.assertTrue(recorded["cold_l2_cache"])
        self.assertFalse(recorded["use_cuda_graph"])

    def test_benchmark_restores_environment_on_failure(self):
        for original in (None, "1"):
            env = (
                {}
                if original is None
                else {"FLASHINFER_GDN_SM120_PIPELINE32": original}
            )
            with patch.dict("os.environ", env, clear=True):
                with self.assertRaises(RuntimeError), bench.official_routes():
                    self.assertEqual(
                        bench.os.environ["FLASHINFER_GDN_SM120_PIPELINE32"], "0"
                    )
                    raise RuntimeError("test failure")
                self.assertEqual(
                    bench.os.environ.get("FLASHINFER_GDN_SM120_PIPELINE32"), original
                )

    def test_cases(self):
        self.assertEqual(bench.cases("2x4096"), [[4096, 4096]])
        self.assertEqual(bench.cases("31+33"), [[31, 33]])
        for text in ("0x4096", "1x0", "-1"):
            with self.assertRaises(ValueError):
                bench.cases(text)

    def test_ragged_slots_do_not_overlap_or_exceed_capacity(self):
        # Include empty sequences and boundaries on both sides of 32.
        import itertools

        for lengths in itertools.product((0, 1, 31, 32, 33, 63, 64, 65), repeat=3):
            shape, _ = layout.workspace_shapes(sum(lengths), len(lengths), 48)
            occupied = set()
            start = 0
            for seq, length in enumerate(lengths):
                slots = set(
                    range(start // 32 + seq, start // 32 + seq + (length + 31) // 32)
                )
                self.assertFalse(occupied & slots)
                self.assertTrue(all(0 <= slot < shape[0] for slot in slots))
                occupied.update(slots)
                start += length


if __name__ == "__main__":
    unittest.main()
