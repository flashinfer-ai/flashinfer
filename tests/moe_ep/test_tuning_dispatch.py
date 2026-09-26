"""CLI dispatch across the SM90, SM100, and SM107 tuners."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.moe_ep import tune


_GEOMETRY = [
    "--hidden",
    "7168",
    "--intermediate",
    "2048",
    "--num-experts",
    "256",
    "--topk",
    "8",
    "--max-tokens",
    "64",
]


@pytest.fixture
def dispatched(monkeypatch):
    calls = []
    original = tune.importlib.import_module

    def load(name, package=None):
        if name.startswith(".backends.mega.kernel.") and name.endswith(".tuner"):
            calls.append(name)
            return SimpleNamespace(run_tuning=lambda args: 17)
        return original(name, package)

    monkeypatch.setattr(tune.importlib, "import_module", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    return calls


@pytest.mark.parametrize(
    "arch,dtype,backend",
    [
        ("auto", "nvfp4", "sm107.nvfp4_nvfp4_bf16_cutedsl"),
        ("auto", "mxfp8_e4m3", "sm107.mxfp8_mxfp8_bf16_cutedsl"),
        ("sm107", "mxfp8_e5m2", "sm107.mxfp8_mxfp8_bf16_cutedsl"),
        ("sm100", "nvfp4", "sm100.nvfp4_nvfp4_bf16_cutedsl"),
        ("sm100", "mxfp8_e5m2", "sm100.mxfp8_mxfp8_bf16_cutedsl"),
        ("sm100", "bf16", "sm100.bf16_bf16_bf16_cutedsl"),
        ("sm100", "bf16_mxfp8_e4m3", "sm100.bf16_mxfp8_bf16_cutedsl"),
        ("sm100", "bf16_mxfp8_e5m2", "sm100.bf16_mxfp8_bf16_cutedsl"),
        ("auto", "sm90_fp8_e4m3", "sm90.fp8_fp8_bf16_pull_cutedsl"),
        ("sm90", "sm90_fp8_e5m2", "sm90.fp8_fp8_bf16_pull_cutedsl"),
    ],
)
def test_tuner_dispatch(arch, dtype, backend, dispatched):
    assert tune.main([*_GEOMETRY, "--arch", arch, "--dtype", dtype]) == 17
    assert dispatched == [f".backends.mega.kernel.{backend}.tuner"]


@pytest.mark.parametrize(
    "options",
    [
        ["--arch", "sm107", "--dtype", "bf16"],
        ["--arch", "auto", "--dtype", "bf16_mxfp8_e4m3"],
        ["--arch", "sm107", "--dtype", "bf16_mxfp8_e5m2"],
        ["--arch", "sm107", "--dtype", "sm90_fp8_e4m3"],
        ["--arch", "sm100", "--dtype", "sm90_fp8_e5m2"],
        ["--arch", "sm90", "--dtype", "nvfp4"],
        ["--arch", "sm107", "--combine-dtype", "nvfp4"],
        ["--arch", "sm107", "--combine-dtype", "mxfp8"],
    ],
)
def test_unsupported_tuner_options_fail_before_import(options, dispatched, capsys):
    assert tune.main([*_GEOMETRY, *options]) == 2
    assert dispatched == []
    assert capsys.readouterr().err
