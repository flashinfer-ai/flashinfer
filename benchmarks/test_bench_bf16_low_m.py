"""CPU-only tests; these do not validate CUDA execution or timings."""

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch


spec = importlib.util.spec_from_file_location(
    "bench_bf16_low_m", Path(__file__).with_name("bench_bf16_low_m.py")
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)


class TestLowMGemmReproduction(unittest.TestCase):
    """Exercise evidence-quality gates without importing the GPU backends."""

    def test_exact_and_zero_rows(self):
        """Accept exact matches, including zero-reference rows."""
        x = torch.tensor([[1.0, -2.0], [0.0, 0.0]])
        self.assertEqual(bench.check_rows(x, x)["max_nrmse"], 0)

    def test_one_bad_row_is_not_hidden_by_aggregation(self):
        """Guard against global averages concealing a failing row."""
        ref = torch.ones(8, 4)
        bad = ref.clone()
        bad[-1].neg_()
        with self.assertRaises(AssertionError):
            bench.check_rows(bad, ref)

    def test_nonfinite_is_rejected(self):
        """Reject non-finite values before computing error metrics."""
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                bench.check_rows(torch.tensor([[value]]), torch.ones(1, 1))

    def test_nonzero_against_zero_is_rejected(self):
        """Do not let zero-reference normalization hide output errors."""
        with self.assertRaises(AssertionError):
            bench.check_rows(torch.ones(1, 2), torch.zeros(1, 2))

    def test_gain_error_is_rejected_even_with_perfect_cosine(self):
        """Catch magnitude errors that cosine similarity cannot detect."""
        with self.assertRaises(AssertionError):
            bench.check_rows(torch.ones(1, 2) * 1.02, torch.ones(1, 2))

    def test_bracket(self):
        """Normalize speedup against the mean control median, not either endpoint."""
        result = bench.bracket([9, 10, 11], [4, 5, 6], [11, 12, 13])
        self.assertAlmostEqual(result["speedup"], 2.2)
        self.assertAlmostEqual(result["control_drift"], 2 / 11)

    def test_invalid_samples(self):
        """Reject timings that cannot yield a meaningful speedup."""
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.bracket([1], [value], [1])

    def test_failure_receipt_and_no_overwrite(self):
        """Retain useful gate failures without overwriting existing evidence."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["--m", "1", "--n", "16", "--k", "128", "--output", str(output)]
            failure = bench.BenchmarkGateError("requires SM100 or SM103")
            with (
                patch.object(bench, "run", side_effect=failure),
                self.assertRaises(RuntimeError) as raised,
            ):
                bench.main(argv)
            self.assertIs(raised.exception, failure)
            receipt = output.read_text()
            result = json.loads(receipt)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["error_type"], "BenchmarkGateError")
            self.assertEqual(result["error_message"], "requires SM100 or SM103")
            with self.assertRaises(FileExistsError):
                bench.main(argv)
            self.assertEqual(output.read_text(), receipt)

    def test_gate_failure_messages_are_distinct(self):
        """Distinguish benchmark gates even when their exception types match."""
        messages = (
            "disable inherited file-based tuning for a default comparison",
            "requires SM100 or SM103",
            "requires the current warp-capable CuTe DSL >=4.7 runtime",
            "requires cupti-python >=13",
            "CUPTI did not report every requested replay",
        )
        with tempfile.TemporaryDirectory() as directory:
            for index, message in enumerate(messages):
                output = Path(directory) / f"result-{index}.json"
                argv = ["--m", "1", "--n", "16", "--k", "128", "--output", str(output)]
                with (
                    self.subTest(message=message),
                    patch.object(
                        bench, "run", side_effect=bench.BenchmarkGateError(message)
                    ),
                    self.assertRaises(bench.BenchmarkGateError),
                ):
                    bench.main(argv)
                result = json.loads(output.read_text())
                self.assertEqual(result["error_type"], "BenchmarkGateError")
                self.assertEqual(result["error_message"], message)

    def test_unexpected_failure_does_not_export_private_details(self):
        """Preserve the original exception without copying private paths into JSON."""
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["--m", "1", "--n", "16", "--k", "128", "--output", str(output)]
            failure = RuntimeError(
                f"backend compilation failed at {directory}/kernel.py"
            )
            with (
                patch.object(bench, "run", side_effect=failure),
                self.assertRaises(RuntimeError) as raised,
            ):
                bench.main(argv)
            self.assertIs(raised.exception, failure)
            receipt = output.read_text()
            result = json.loads(receipt)
            self.assertEqual(result["status"], "incomplete")
            self.assertEqual(result["error_type"], "RuntimeError")
            self.assertEqual(
                result["error_message"], "Failure details are available in stderr."
            )
            self.assertNotIn(directory, receipt)
            self.assertNotIn(str(failure), receipt)


if __name__ == "__main__":
    unittest.main()
