"""Benchmark precision options and real autotuner context restoration."""

import importlib
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("torch")

from flashinfer.autotuner import AutoTuner, TuningConfig


@pytest.fixture
def benchmark_modules(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, ModuleType]:
    monkeypatch.syspath_prepend(str(Path(__file__).parents[2] / "benchmarks"))
    return (
        importlib.import_module("svdquant_sm120_cli"),
        importlib.import_module("svdquant_sm120_benchmark"),
    )


def test_precision_defaults_preserve_operator_policy(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    cli, _ = benchmark_modules
    options = cli.parse_options([])
    assert options.tuning_repeat is None
    assert options.tuning_replays is None
    assert options.tuning_l2 == "operator"


def test_precision_options_do_not_change_cache_policy(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    cli, _ = benchmark_modules
    options = cli.parse_options(["--tuning-repeat", "64", "--tuning-replays", "5"])
    assert (options.tuning_repeat, options.tuning_replays) == (64, 5)
    assert options.tuning_l2 == "operator"


@pytest.mark.parametrize("flag", ["--tuning-repeat", "--tuning-replays"])
@pytest.mark.parametrize("value", ["0", "-1"])
def test_precision_counts_must_be_positive(
    benchmark_modules: tuple[ModuleType, ModuleType], flag: str, value: str
) -> None:
    cli, _ = benchmark_modules
    with pytest.raises(SystemExit) as failure:
        cli.parse_options([flag, value])
    assert failure.value.code == 2


@pytest.mark.parametrize("fail", [False, True])
@pytest.mark.parametrize("policy", ["operator", "warm", "cold"])
def test_precision_restores_repeat_and_replay_context(
    benchmark_modules: tuple[ModuleType, ModuleType], fail: bool, policy: str
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    original_repeat = tuner.repeat
    original_replays = tuner._override_cuda_graph_profile_replays
    original_tuning_mode = tuner.is_tuning_mode

    def run_context() -> None:
        with helpers.tuning_precision(repeat=64, replays=5):
            assert tuner.repeat == 64
            with helpers.tuning_context(helpers.tuning_policy(policy), tuning=True):
                assert tuner.is_tuning_mode
                config = tuner._apply_tuning_overrides(
                    TuningConfig(use_cuda_graph=True)
                )
                assert config.cuda_graph_profile_replays == 5
                assert tuner._get_profiling_repeat(config) == 64
            if fail:
                raise RuntimeError("benchmark failure")

    if fail:
        with pytest.raises(RuntimeError, match="benchmark failure"):
            run_context()
    else:
        run_context()
    assert tuner.repeat == original_repeat
    assert tuner._override_cuda_graph_profile_replays == original_replays
    assert tuner.is_tuning_mode == original_tuning_mode


def test_default_precision_preserves_an_outer_override(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    with helpers.tuning_precision(repeat=17, replays=3):
        with helpers.tuning_precision():
            assert tuner.repeat == 17
            assert tuner._override_cuda_graph_profile_replays == 3
        with helpers.tuning_precision(repeat=64, replays=5):
            assert tuner.repeat == 64
        assert tuner.repeat == 17
        assert tuner._override_cuda_graph_profile_replays == 3


def test_repeat_restored_if_replay_context_cannot_enter(
    benchmark_modules: tuple[ModuleType, ModuleType],
) -> None:
    _, helpers = benchmark_modules
    tuner = AutoTuner.get()
    original_repeat = tuner.repeat
    with (
        pytest.raises(ValueError, match="cuda_graph_profile_replays"),
        helpers.tuning_precision(repeat=64, replays=0),
    ):
        pytest.fail("invalid replay count entered the context")
    assert tuner.repeat == original_repeat
