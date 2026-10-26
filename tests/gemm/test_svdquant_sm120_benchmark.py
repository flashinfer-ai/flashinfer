"""Backend selection must not prepare kernels outside the requested comparison."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest
import torch


@pytest.fixture
def helpers(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import benchmark helpers through the same module path used by the CLI."""
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    return importlib.import_module("svdquant_sm120_benchmark")


def test_cute_only_call_does_not_prepare_cutlass(
    helpers: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    """CuTe-only setup must preserve L2T without requesting native preparation."""
    # Given: a CuTe-only benchmark with rank-32 weights.
    tensor = torch.ones((2, 32), dtype=torch.bfloat16)
    case = helpers.Case(*([tensor] * 11), None)
    prepared: list[torch.Tensor] = []

    def prepare(source: torch.Tensor) -> torch.Tensor:
        """Record preparation requests without compiling a CUDA module."""
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
    """Reject unknown or empty backend names before any benchmark work starts."""
    with pytest.raises(ValueError, match="SVDQ_BENCH_BACKENDS"):
        helpers.parse_backends(value)


def test_backend_selection_strips_spaces_and_duplicates(helpers: ModuleType) -> None:
    """Normalization must preserve the first backend as the accuracy reference."""
    assert helpers.parse_backends(" auto, cute-dsl,auto ") == ("auto", "cute-dsl")


def test_benchmark_rejects_rank_without_cutlass_support(helpers: ModuleType) -> None:
    """The CLI must reject ranks unsupported by the default backend comparison."""
    cli = importlib.import_module("svdquant_sm120_cli")
    with pytest.raises(SystemExit):
        cli.parse_options(["--rank", "96"])
