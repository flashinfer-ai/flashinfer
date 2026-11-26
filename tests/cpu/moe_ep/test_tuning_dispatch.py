"""CLI dispatch across the SM90, SM100, and SM107 tuners."""

import json
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


@pytest.mark.cpu_only
def test_blackwell_instrumentation_cache_tag(monkeypatch):
    from flashinfer.jit import cute_dsl_core
    from flashinfer.moe_ep.shims.sm100.cutedsl_megamoe.sources import knob_cache

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(
            name="test-device", major=10, minor=0, multi_processor_count=148
        ),
    )
    monkeypatch.setattr(cute_dsl_core, "_get_cute_dsl_version", lambda: "test-dsl")
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_100a")
    monkeypatch.setenv("CUTE_DSL_COMPILER_OPT", "")
    monkeypatch.delenv("FLASHINFER_MEGAMOE_INSTRUMENTATION_TAG", raising=False)
    plain = json.loads(knob_cache.device_identity())
    assert plain == dict(
        name="test-device",
        cc=[10, 0],
        sms=148,
        cuda=torch.version.cuda,
        torch=torch.__version__,
        dsl_stack="test-dsl",
        compile_arch="sm_100a",
        compiler_options="",
        instrumentation="",
    )
    monkeypatch.setenv("FLASHINFER_MEGAMOE_INSTRUMENTATION_TAG", "profile-a")
    tagged = json.loads(knob_cache.device_identity())
    assert tagged == {**plain, "instrumentation": "profile-a"}
    monkeypatch.setenv("FLASHINFER_MEGAMOE_INSTRUMENTATION_TAG", "")
    assert json.loads(knob_cache.device_identity()) == plain


@pytest.fixture
def dispatched(monkeypatch):
    calls = []
    original = tune.importlib.import_module

    def load(name, package=None):
        if name.startswith(".backends.mega.kernel.") and name.endswith(
            (".tuner", ".next_tuning")
        ):
            calls.append(name)
            return SimpleNamespace(run_tuning=lambda args: 17)
        return original(name, package)

    monkeypatch.setattr(tune.importlib, "import_module", load)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 7))
    return calls


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch,dtype,backend",
    [
        ("auto", "nvfp4", "sm107.nvfp4_nvfp4_bf16_cutedsl"),
        ("auto", "mxfp8_e4m3", "sm107.mxfp8_mxfp8_bf16_cutedsl"),
        ("sm107", "mxfp8_e5m2", "sm107.mxfp8_mxfp8_bf16_cutedsl"),
        ("auto", "mxfp4_mxfp8", "sm107.mxfp8_mxfp4_bf16_cutedsl"),
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


@pytest.mark.cpu_only
def test_bare_sm100_command_tunes_serving_nvfp4(dispatched, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    assert tune.main(_GEOMETRY) == 17
    assert dispatched == [".backends.mega.kernel.sm100.next_tuning"]


@pytest.mark.cpu_only
def test_blackwell_nvfp4_legacy_tuner_is_explicit(dispatched):
    assert (
        tune.main([*_GEOMETRY, "--arch", "sm100", "--implementation", "legacy"]) == 17
    )
    assert dispatched == [".backends.mega.kernel.sm100.nvfp4_nvfp4_bf16_cutedsl.tuner"]


@pytest.mark.cpu_only
@pytest.mark.parametrize("dtype", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize("implementation", ["default", "next"])
def test_blackwell_genphase_dtype_dispatch(dtype, implementation, dispatched):
    assert (
        tune.main(
            [
                *_GEOMETRY,
                "--arch",
                "sm100",
                "--dtype",
                dtype,
                "--kernel-variant",
                "genphase",
                "--implementation",
                implementation,
                "--no-apply-routing-weights-before-fc2",
            ]
        )
        == 17
    )
    assert dispatched == [".backends.mega.kernel.sm100.next_tuning"]


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "options",
    [
        ["--arch", "sm107", "--dtype", "bf16"],
        ["--arch", "auto", "--dtype", "bf16_mxfp8_e4m3"],
        ["--arch", "sm107", "--dtype", "bf16_mxfp8_e5m2"],
        ["--arch", "sm107", "--dtype", "sm90_fp8_e4m3"],
        ["--arch", "sm100", "--dtype", "sm90_fp8_e5m2"],
        ["--arch", "sm90", "--dtype", "nvfp4"],
        ["--arch", "sm107", "--kernel-variant", "genphase", "--combine-dtype", "nvfp4"],
        ["--arch", "sm107", "--kernel-variant", "genphase", "--combine-dtype", "mxfp8"],
        ["--arch", "sm100", "--implementation", "legacy", "--activation", "situ"],
        ["--arch", "sm100", "--implementation", "legacy", "--situ-beta", "1.0"],
        ["--arch", "sm100", "--implementation", "legacy", "--fc1-alpha", "0.5"],
        ["--arch", "sm100", "--implementation", "legacy", "--dtype", "mxfp8_e4m3"],
        [
            "--arch",
            "sm100",
            "--implementation",
            "legacy",
            "--kernel-variant",
            "genphase",
        ],
        ["--arch", "sm107", "--implementation", "legacy"],
        ["--dtype", "sm90_mxfp4", "--implementation", "legacy"],
        ["--dtype", "sm120_mxfp4_mxfp8", "--implementation", "legacy"],
        ["--arch", "sm107", "--dtype", "mxfp8_e4m3", "--input-norm-const", "2"],
        ["--arch", "sm107", "--no-do-finalize"],
        ["--arch", "sm100", "--kernel-variant", "auto"],
        ["--dtype", "sm90_fp8_e4m3", "--kernel-variant", "auto"],
        ["--dtype", "sm120_mxfp4_mxfp8", "--kernel-variant", "auto"],
        ["--arch", "sm107", "--kernel-variant", "auto", "--max-candidates", "0"],
        ["--arch", "sm107", "--kernel-variant", "auto", "--timed-iters", "0"],
        ["--arch", "sm100", "--dtype", "bf16", "--no-do-finalize"],
        ["--arch", "sm100", "--no-do-finalize", "--allow-nondeterministic"],
        [
            "--arch",
            "sm100",
            "--no-apply-routing-weights-before-fc2",
            "--allow-nondeterministic",
        ],
        ["--arch", "sm100", "--no-do-finalize", "--combine-dtype", "nvfp4"],
    ],
)
def test_unsupported_tuner_options_fail_before_import(options, dispatched, capsys):
    assert tune.main([*_GEOMETRY, *options]) == 2
    assert dispatched == []
    assert capsys.readouterr().err


@pytest.mark.cpu_only
def test_v1_cache_option_is_rejected(dispatched, capsys):
    with pytest.raises(SystemExit) as exc:
        tune.main([*_GEOMETRY, "--autotune-cache", "cache.json"])
    assert exc.value.code == 2
    assert not dispatched
    assert "unrecognized arguments: --autotune-cache" in capsys.readouterr().err


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "options,do_finalize",
    [([], True), (["--do-finalize"], True), (["--no-do-finalize"], False)],
)
@pytest.mark.parametrize(
    "routing_option",
    ["--no-apply-routing-weights-before-fc2", "--no-apply-topk-in-fc1"],
)
def test_finalization_and_routing_options_reach_nvfp4_tuner(
    monkeypatch, options, do_finalize, routing_option
):
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
            "sm100",
            *options,
            routing_option,
        ]
    )
    (args,) = received
    assert args.do_finalize == do_finalize
    assert not args.apply_routing_weights_before_fc2


@pytest.mark.cpu_only
def test_post_fc2_routing_weights_reach_sm107_nvfp4_tuner(monkeypatch):
    received = []
    monkeypatch.setattr(
        tune.importlib,
        "import_module",
        lambda *args: SimpleNamespace(run_tuning=lambda args: received.append(args)),
    )
    tune.main([*_GEOMETRY, "--arch", "sm107", "--no-apply-routing-weights-before-fc2"])
    (args,) = received
    assert args.do_finalize
    assert not args.apply_routing_weights_before_fc2


@pytest.mark.cpu_only
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
            "--no-apply-routing-weights-before-fc2",
        ]
    )
    (args,) = received
    assert not args.apply_routing_weights_before_fc2
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


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "field", ["input_norm_const", "fc1_alpha", "fc2_alpha", "fc1_norm_const"]
)
def test_invalid_tuning_scalar_identifies_argument(field):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.tuning import run_tuning

    args = tune._parse_args([*_GEOMETRY, "--" + field.replace("_", "-"), "0"])
    with pytest.raises(ValueError, match=field):
        run_tuning(args, "nvfp4")


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", ["inference", "genphase"])
@pytest.mark.parametrize("allow_nondeterministic", [False, True])
def test_sm107_schedule_base_respects_nondeterminism(
    monkeypatch, tmp_path, variant, allow_nondeterministic
):
    from flashinfer.moe_ep.backends.mega.kernel.sm107.tuning import _candidates
    from flashinfer.moe_ep.shims.sm107.cutedsl_megamoe import knob_cache

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    args = tune._parse_args(
        [
            *_GEOMETRY,
            "--arch",
            "sm107",
            "--kernel-variant",
            variant,
            "--sweep",
            "schedule",
            *(["--allow-nondeterministic"] if allow_nondeterministic else []),
        ]
    )
    knob_cache.record_knobs(
        {"reduce_topk_in_kernel": True},
        dtype="nvfp4",
        kernel_variant=variant,
        world_size=1,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.num_experts,
        topk=args.topk,
        max_tokens=64,
        allow_nondeterministic=True,
    )
    choices = _candidates(args, 0, 1, 64, "nvfp4", variant)
    assert choices
    assert all(k["reduce_topk_in_kernel"] == allow_nondeterministic for k in choices)


@pytest.mark.cpu_only
@pytest.mark.parametrize("arch", ["sm90"])
def test_mxfp4_tuner_rejects_unwired_architecture(arch, capsys):
    assert tune.main([*_GEOMETRY, "--arch", arch, "--dtype", "mxfp4_mxfp8"]) == 2
    assert capsys.readouterr().err


@pytest.mark.cpu_only
@pytest.mark.parametrize("variant", ["genphase", "auto"])
@pytest.mark.parametrize(
    "dtype", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8", "mxfp4"]
)
def test_variant_reaches_sm107_tuner(dispatched, variant, dtype):
    assert (
        tune.main(
            [
                *_GEOMETRY,
                "--arch",
                "sm107",
                "--dtype",
                dtype,
                "--kernel-variant",
                variant,
            ]
        )
        == 17
    )
    assert len(dispatched) == 1


@pytest.mark.cpu_only
def test_auto_variant_rejects_detected_blackwell(dispatched, monkeypatch, capsys):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    assert tune.main([*_GEOMETRY, "--kernel-variant", "auto"]) == 2
    assert not dispatched
    assert "requires SM107" in capsys.readouterr().err


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    "arch,dtype",
    [("sm90", "sm90_fp8_e4m3"), ("sm100", "bf16"), ("sm100", "bf16_mxfp8_e4m3")],
)
def test_genphase_rejects_unsupported_dtypes(arch, dtype, dispatched):
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
@pytest.mark.parametrize(
    "options",
    [[], ["--no-do-finalize", "--no-apply-routing-weights-before-fc2"]],
)
@pytest.mark.parametrize(
    "implementation",
    [[], ["--implementation", "default"], ["--implementation", "next"]],
)
def test_standard_blackwell_dispatch(kind, options, implementation, dispatched):
    assert (
        tune.main(
            [
                *_GEOMETRY,
                "--arch",
                "sm100",
                "--dtype",
                kind,
                *implementation,
                *options,
            ]
        )
        == 17
    )
    assert dispatched == [".backends.mega.kernel.sm100.next_tuning"]
