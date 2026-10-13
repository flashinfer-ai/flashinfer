# Copyright (c) 2026 by NVIDIA CORPORATION.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys

import pytest

from benchmarks import bench_union_gqa as benchmark


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--seq-lens", "16,17"], "positive and divisible by 8"),
        (["--seq-lens", "0"], "positive and divisible by 8"),
        (["--seq-lens", "-8"], "positive and divisible by 8"),
        (["--sharing-ratios", "1,0.1"], "exactly representable"),
        (
            ["--sharing-ratios", "0.5", "--topk-blocks", "16,3"],
            "exactly representable",
        ),
        (["--sharing-ratios", "nan"], "must be in [0, 1]"),
    ],
)
def test_invalid_grid_fails_before_any_case(monkeypatch, capsys, arguments, message):
    """Reject the whole grid even when an earlier combination would be valid."""
    monkeypatch.setattr(sys, "argv", ["bench_union_gqa.py", *arguments])
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda: (10, 0))

    def unexpected_case(*args):
        pytest.fail("invalid grid reached GPU execution")

    monkeypatch.setattr(benchmark, "_run_case", unexpected_case)
    with pytest.raises(SystemExit) as exc:
        benchmark.main()
    assert exc.value.code == 2
    assert message in capsys.readouterr().err


def test_representable_ratio_with_float_roundoff_reaches_grid(monkeypatch):
    """Use the generator's tolerance when the mathematical stride is integral."""
    monkeypatch.setattr(
        sys,
        "argv",
        ["bench_union_gqa.py", "--sharing-ratios", "0.8", "--topk-blocks", "5"],
    )
    monkeypatch.setattr(benchmark.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.torch.cuda, "get_device_capability", lambda: (10, 0))

    def reached_case(*args):
        raise RuntimeError("valid grid reached")

    monkeypatch.setattr(benchmark, "_run_case", reached_case)
    with pytest.raises(RuntimeError, match="valid grid reached"):
        benchmark.main()
