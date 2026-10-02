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
        ("auto", "mxfp4_mxfp8", "sm107.mxfp8_mxfp4_bf16_cutedsl"),
        ("sm100", "nvfp4", "sm100.nvfp4_nvfp4_bf16_cutedsl"),
        ("sm100", "mxfp8_e5m2", "sm100.mxfp8_mxfp8_bf16_cutedsl"),
        ("sm100", "bf16", "sm100.bf16_bf16_bf16_cutedsl"),
        ("sm100", "bf16_mxfp8_e4m3", "sm100.bf16_mxfp8_bf16_cutedsl"),
        ("sm100", "bf16_mxfp8_e5m2", "sm100.bf16_mxfp8_bf16_cutedsl"),
        ("auto", "sm90_fp8_e4m3", "sm90.fp8_fp8_bf16_pull_cutedsl"),
        ("sm90", "sm90_fp8_e5m2", "sm90.fp8_fp8_bf16_pull_cutedsl"),
        ("auto", "sm90_bf16", "sm90.bf16_bf16_bf16_pull_cutedsl"),
        ("sm90", "sm90_bf16", "sm90.bf16_bf16_bf16_pull_cutedsl"),
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
        ["--arch", "sm100", "--dtype", "sm90_bf16"],
        ["--arch", "sm90", "--dtype", "sm90_bf16", "--fp8-scale-mode", "blockwise"],
        ["--arch", "sm107", "--kernel-variant", "genphase", "--combine-dtype", "nvfp4"],
        ["--arch", "sm107", "--kernel-variant", "genphase", "--combine-dtype", "mxfp8"],
        ["--arch", "sm100", "--activation", "situ"],
        ["--arch", "sm100", "--situ-beta", "1.0"],
        ["--arch", "sm100", "--fc1-alpha", "0.5"],
        ["--arch", "sm107", "--dtype", "mxfp8_e4m3", "--input-norm-const", "2"],
    ],
)
def test_unsupported_tuner_options_fail_before_import(options, dispatched, capsys):
    assert tune.main([*_GEOMETRY, *options]) == 2
    assert dispatched == []
    assert capsys.readouterr().err


def test_situ_and_scaling_options_reach_sm107_tuner(monkeypatch):
    received = []
    monkeypatch.setattr(
        tune.importlib,
        "import_module",
        lambda *args: SimpleNamespace(run_tuning=lambda args: received.append(args)),
    )
    tune.main(
        [
            *_GEOMETRY,
            "--arch",
            "sm107",
            "--dtype",
            "nvfp4",
            "--activation",
            "situ",
            "--situ-beta",
            "1.25",
            "--situ-linear-beta",
            "0.75",
            "--input-norm-const",
            "2",
            "--fc1-alpha",
            "0.5",
            "--fc2-alpha",
            "0.25",
            "--fc1-norm-const",
            "4",
        ]
    )
    (args,) = received
    assert (args.activation, args.situ_beta, args.situ_linear_beta) == (
        "situ",
        1.25,
        0.75,
    )
    assert (
        args.input_norm_const,
        args.fc1_alpha,
        args.fc2_alpha,
        args.fc1_norm_const,
    ) == (2, 0.5, 0.25, 4)


@pytest.mark.parametrize(
    "field", ["input_norm_const", "fc1_alpha", "fc2_alpha", "fc1_norm_const"]
)
def test_invalid_tuning_scalar_identifies_argument(field):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.tuning import run_tuning

    args = tune._parse_args([*_GEOMETRY, "--" + field.replace("_", "-"), "0"])
    with pytest.raises(ValueError, match=field):
        run_tuning(args, "nvfp4")


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_mxfp4_tuner_rejects_unwired_architecture(arch, capsys):
    assert tune.main([*_GEOMETRY, "--arch", arch, "--dtype", "mxfp4_mxfp8"]) == 2
    assert capsys.readouterr().err


def test_genphase_reaches_sm107_tuner(dispatched):
    assert (
        tune.main([*_GEOMETRY, "--arch", "sm107", "--kernel-variant", "genphase"]) == 17
    )
    assert len(dispatched) == 1


@pytest.mark.parametrize("arch", ["sm90", "sm100"])
def test_genphase_rejects_other_architectures(arch, dispatched):
    dtype = "sm90_fp8_e4m3" if arch == "sm90" else "nvfp4"
    assert (
        tune.main(
            [
                *_GEOMETRY,
                "--arch",
                arch,
                "--dtype",
                dtype,
                "--kernel-variant",
                "genphase",
            ]
        )
        == 2
    )
    assert not dispatched


@pytest.mark.parametrize("dtype", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("combine_dtype", ["nvfp4", "mxfp8"])
def test_sm107_quantized_combine_reaches_tuner(dtype, combine_dtype, dispatched):
    assert (
        tune.main(
            [
                *_GEOMETRY,
                "--arch",
                "sm107",
                "--dtype",
                dtype,
                "--combine-dtype",
                combine_dtype,
            ]
        )
        == 17
    )
    assert len(dispatched) == 1
