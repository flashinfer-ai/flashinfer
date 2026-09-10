"""CPU-only tests; these do not validate CUDA execution or timings."""

import importlib.util
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
    def test_exact_and_zero_rows(self):
        x = torch.tensor([[1.0, -2.0], [0.0, 0.0]])
        self.assertEqual(bench.check_rows(x, x)["max_nrmse"], 0)

    def test_one_bad_row_is_not_hidden_by_aggregation(self):
        ref = torch.ones(8, 4)
        bad = ref.clone()
        bad[-1].neg_()
        with self.assertRaises(AssertionError):
            bench.check_rows(bad, ref)

    def test_nonfinite_is_rejected(self):
        for value in (math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(AssertionError):
                bench.check_rows(torch.tensor([[value]]), torch.ones(1, 1))

    def test_nonzero_against_zero_is_rejected(self):
        with self.assertRaises(AssertionError):
            bench.check_rows(torch.ones(1, 2), torch.zeros(1, 2))

    def test_gain_error_is_rejected_even_with_perfect_cosine(self):
        with self.assertRaises(AssertionError):
            bench.check_rows(torch.ones(1, 2) * 1.02, torch.ones(1, 2))

    def test_bracket(self):
        result = bench.bracket([9, 10, 11], [4, 5, 6], [11, 12, 13])
        self.assertAlmostEqual(result["speedup"], 2.2)
        self.assertAlmostEqual(result["control_drift"], 2 / 11)

    def test_invalid_samples(self):
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(value=value), self.assertRaises(ValueError):
                bench.bracket([1], [value], [1])

    def test_failure_receipt_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result.json"
            argv = ["--m", "1", "--n", "16", "--k", "128", "--output", str(output)]
            with (
                patch.object(bench, "run", side_effect=RuntimeError("test failure")),
                self.assertRaises(RuntimeError),
            ):
                bench.main(argv)
            receipt = output.read_text()
            self.assertIn('"status": "incomplete"', receipt)
            self.assertIn('"error_type": "RuntimeError"', receipt)
            with self.assertRaises(FileExistsError):
                bench.main(argv)
            self.assertEqual(output.read_text(), receipt)


if __name__ == "__main__":
    unittest.main()
