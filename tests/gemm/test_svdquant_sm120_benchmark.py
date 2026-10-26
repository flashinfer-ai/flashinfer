"""Backend selection must not prepare kernels outside the requested comparison."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    return importlib.import_module("svdquant_sm120_benchmark")


def test_cute_only_call_does_not_prepare_cutlass(
    helpers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Given: a CuTe-only benchmark with rank-32 weights.
    tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    case = helpers.Case(*([tensor] * 11), None)
    prepared: list[torch.Tensor] = []

    def prepare(source: torch.Tensor) -> torch.Tensor:
        prepared.append(source)
        return source

    monkeypatch.setattr(helpers, "prepare_svdquant_l2t", prepare)

    # When: constructing the benchmark call.
    call = helpers.make_call(case, "linear", "cute-dsl", False)

    # Then: it keeps the original operand and requires no CUTLASS preparation.
    assert not prepared
    assert call.operands[5] is case.l2t


@pytest.mark.parametrize("value", ["", "cute-dsl,", "invalid", "auto,invalid"])
def test_invalid_backend_selection_fails_early(helpers: ModuleType, value: str) -> None:
    with pytest.raises(ValueError, match="SVDQ_BENCH_BACKENDS"):
        helpers.parse_backends(value)


def test_backend_selection_strips_spaces_and_duplicates(helpers: ModuleType) -> None:
    assert helpers.parse_backends(" auto, cute-dsl,auto ") == ("auto", "cute-dsl")


def test_benchmark_rejects_rank_without_cutlass_support(helpers: ModuleType) -> None:
    cli = importlib.import_module("svdquant_sm120_cli")
    with pytest.raises(SystemExit):
        cli.parse_options(["--rank", "96"])
