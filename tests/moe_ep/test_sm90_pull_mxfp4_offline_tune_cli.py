"""CPU contracts for the SM90 MXFP4 backend-local offline tuner CLI."""

from __future__ import annotations

import inspect
from types import SimpleNamespace
from unittest import mock

import flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel as sm90_mega

import numpy as np
import pytest
import torch

from flashinfer.moe_ep import tune
from flashinfer.moe_ep.backends.mega.kernel import tuning as shared_tuning
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig,
    tuner,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


def _argv(
    dtype: str = "sm90_mxfp4",
    *extra: str,
    max_tokens: tuple[str, ...] = ("8", "512"),
) -> list[str]:
    return [
        "--dtype",
        dtype,
        "--hidden",
        "7168",
        "--intermediate",
        "3072",
        "--num-experts",
        "384",
        "--topk",
        "6",
        "--max-tokens",
        *max_tokens,
        *extra,
    ]


def test_shared_finish_sweep_forwards_optional_tune_kwargs() -> None:
    args = SimpleNamespace(
        skew=None,
        max_candidates=None,
        dtype="sm90_mxfp4",
        warmup_iters=3,
        timed_iters=10,
    )
    captured = {}

    def fake_tune(*positional, **kwargs):
        captured["positional"] = positional
        captured["kwargs"] = kwargs
        return {"winner": True}

    marker = object()
    assert shared_tuning.finish_sweep(
        args,
        1,
        32,
        8,
        marker,
        "y",
        "l1",
        "l2",
        [{"candidate": 1}],
        fake_tune,
        tune_kwargs={"gate_up_clamp": 10.0, "routing_profile": "profile"},
    ) == {"winner": True}
    assert captured["positional"] == ("y", "l1", "l2", marker)
    assert captured["kwargs"] == {
        "num_tokens": 8,
        "candidates": [{"candidate": 1}],
        "warmup_iters": 3,
        "timed_iters": 10,
        "gate_up_clamp": 10.0,
        "routing_profile": "profile",
    }


def test_shared_finish_sweep_forwards_max_candidates_subset() -> None:
    args = SimpleNamespace(
        skew=None,
        max_candidates=1,
        dtype="sm90_mxfp4",
        warmup_iters=3,
        timed_iters=10,
    )
    captured = {}

    def fake_tune(*positional, **kwargs):
        captured["candidates"] = kwargs["candidates"]
        return kwargs["candidates"][0]

    candidates = [{"candidate": 1}, {"candidate": 2}]
    assert (
        shared_tuning.finish_sweep(
            args,
            1,
            32,
            8,
            object(),
            "y",
            "l1",
            "l2",
            candidates,
            fake_tune,
        )
        == candidates[0]
    )
    assert captured["candidates"] == candidates[:1]


def test_mxfp4_defaults_select_fixed_format_and_fused_identity() -> None:
    args = tune._parse_args(_argv())
    runtime = Sm90_Fp8_Mxfp4_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=3072,
        top_k=6,
    )
    assert args.dtype == "sm90_mxfp4"
    assert args.fp8_scale_mode == "mxfp4_hybrid"
    assert args.gate_up_clamp == 10.0
    assert runtime.gate_up_clamp == args.gate_up_clamp
    assert args.routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    assert args.warmup_iters == 3
    assert args.timed_iters == 10
    assert tune._argument_error(args) is None


@pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
def test_fp8_cli_abbreviations_keep_existing_values(scale_mode):
    full = tune._parse_args(_argv("sm90_fp8_e4m3", "--fp8-scale-mode", scale_mode))
    short = tune._parse_args(_argv("sm90_fp8_e4m3", f"--fp8-s={scale_mode}"))
    assert vars(short) == vars(full)
    assert tune._argument_error(short) is None


@pytest.mark.parametrize(
    "profile",
    [
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    ],
)
@pytest.mark.parametrize("spelling", ["--routing-profile", "--routing-p"])
@pytest.mark.parametrize("equals", [False, True])
def test_explicit_routing_profile_is_detected_for_every_spelling(
    profile, spelling, equals
):
    flags = [f"{spelling}={profile}"] if equals else [spelling, profile]
    fp8 = tune._parse_args(_argv("sm90_fp8_e4m3", *flags))
    assert "only wired" in (tune._argument_error(fp8) or "")
    mxfp4 = tune._parse_args(_argv("sm90_mxfp4", *flags))
    assert mxfp4.routing_profile == profile
    assert tune._argument_error(mxfp4) is None


def test_existing_sm90_fp8_cli_default_remains_per_tensor() -> None:
    args = tune._parse_args(_argv("sm90_fp8_e4m3"))
    assert args.fp8_scale_mode == "per_tensor"
    assert args.gate_up_clamp is None


def test_mxfp4_explicit_gate_up_clamp_overrides_canonical_default() -> None:
    args = tune._parse_args(_argv("sm90_mxfp4", "--gate-up-clamp", "7.5"))
    assert args.gate_up_clamp == 7.5
    assert tune._argument_error(args) is None


def test_mxfp4_live_tokens_must_match_every_persistent_cache_bucket() -> None:
    mismatch = tune._parse_args(_argv("sm90_mxfp4", "--live-tokens", "8"))
    assert "not part of the persistent cache key" in (
        tune._argument_error(mismatch) or ""
    )

    matching = tune._parse_args(
        _argv("sm90_mxfp4", "--live-tokens", "8", max_tokens=("8",))
    )
    assert tune._argument_error(matching) is None


def test_existing_sm90_fp8_cli_retains_live_token_override() -> None:
    args = tune._parse_args(_argv("sm90_fp8_e4m3", "--live-tokens", "8"))
    assert args.live_tokens == 8
    assert args.max_tokens == [8, 512]
    assert tune._argument_error(args) is None


def test_mxfp4_main_dispatches_to_backend_local_tuner(monkeypatch) -> None:
    captured = {}

    def fake_run(args) -> int:
        captured["args"] = args
        return 17

    monkeypatch.setattr(tuner, "run_tuning", fake_run)
    assert tune.main(_argv()) == 17
    assert captured["args"].dtype == "sm90_mxfp4"
    assert captured["args"].fp8_scale_mode == "mxfp4_hybrid"
    assert captured["args"].routing_profile == SM90_ROUTING_PROFILE_BLOCK_PERMUTATION


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (("--fp8-scale-mode", "per_tensor"), "fixes --fp8-scale-mode"),
        (("--combine-dtype", "mxfp8"), "only wired for --dtype nvfp4"),
        (("--allow-nondeterministic",), "not applicable"),
        (("--sweep", "schedule"), "only --sweep default"),
        (("--base-knobs", "{}"), "no --base-knobs"),
        (("--skew", "2"), "balanced routing"),
        (("--max-candidates", "0"), "must be positive"),
        (("--seed", "1"), "requires --seed 0"),
    ],
)
def test_mxfp4_irrelevant_or_domain_expanding_flags_fail_closed(
    extra: tuple[str, ...], message: str
) -> None:
    args = tune._parse_args(_argv("sm90_mxfp4", *extra))
    assert message in (tune._argument_error(args) or "")


def test_non_mxfp4_rejects_hybrid_scale_and_routing_profile() -> None:
    hybrid = tune._parse_args(
        _argv("sm90_fp8_e4m3", "--fp8-scale-mode", "mxfp4_hybrid")
    )
    assert "requires --dtype sm90_mxfp4" in (tune._argument_error(hybrid) or "")
    routing = tune._parse_args(
        _argv(
            "sm90_fp8_e4m3",
            "--routing-profile",
            SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        )
    )
    assert "only wired" in (tune._argument_error(routing) or "")


@pytest.mark.parametrize(
    "profile",
    [
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    ],
)
def test_mxfp4_accepts_each_canonical_routing_profile(profile: str) -> None:
    args = tune._parse_args(_argv("sm90_mxfp4", "--routing-profile", profile))
    assert args.routing_profile == profile
    assert tune._argument_error(args) is None


@pytest.mark.parametrize(
    "invalid",
    [
        None,
        True,
        "block_permutation",
        "published_exact_balanced",
        " block_permutation_v1",
    ],
)
def test_routing_profile_normalizer_rejects_aliases_strictly(invalid) -> None:
    with pytest.raises(ValueError, match="exactly one of"):
        normalize_sm90_routing_profile(invalid)


def test_tune_one_uses_complete_fused_strategy_union(monkeypatch) -> None:
    wrapper_name = "autotune_hopper_mxfp4_mega_moe"
    from flashinfer.moe_ep.kernel_src.sm90 import (
        pull_style_cutedsl_megakernel as pkg,
    )

    args = tune._parse_args(_argv("sm90_mxfp4"))
    captured = {}

    class FakeBuffer:
        destroyed = False

        def destroy(self) -> None:
            self.destroyed = True

    buffer = FakeBuffer()

    def fake_create(
        actual_args,
        rank,
        world_size,
        max_tokens,
        live_tokens,
        initial_tactic,
    ):
        captured["create"] = (
            actual_args,
            rank,
            world_size,
            max_tokens,
            live_tokens,
            initial_tactic,
        )
        return "y", "l1", "l2", buffer

    def fake_finish(
        actual_args,
        rank,
        max_tokens,
        live_tokens,
        actual_buffer,
        y,
        l1,
        l2,
        candidates,
        tune_fn,
        *,
        tune_kwargs=None,
    ):
        captured["tune_kwargs"] = tune_kwargs
        captured["finish"] = (
            actual_args,
            rank,
            max_tokens,
            live_tokens,
            actual_buffer,
            y,
            l1,
            l2,
            candidates,
            tune_fn,
        )
        return {"winner": "fused"}

    monkeypatch.setattr(
        pkg,
        "require_hopper_mxfp4_fused_tuning_device",
        lambda: None,
    )
    monkeypatch.setattr(tuner, "_create_canonical_inputs", fake_create)
    monkeypatch.setattr(tuner, "finish_sweep", fake_finish)
    assert tuner.tune_one(args, rank=0, world_size=4, max_tokens=8) == {
        "winner": "fused"
    }

    expected = pkg.hopper_mxfp4_candidates(
        8,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.num_experts,
        world_size=4,
        routing_profile=args.routing_profile,
    )
    assert all(len(candidate) == 20 for candidate in expected)
    assert len(expected) == 38
    assert sum(candidate["tail_split_pairs"] for candidate in expected) == 12
    assert any(candidate["fc1_ready_mode"] == "k256" for candidate in expected)
    assert any(candidate["fc2_tail_n8"] for candidate in expected)
    assert captured["create"] == (args, 0, 4, 8, 8, expected[0])
    assert captured["finish"][-2] == expected
    assert captured["finish"][-1] is getattr(pkg, wrapper_name)
    assert captured["tune_kwargs"] == {"gate_up_clamp": 10.0}
    inspect.signature(getattr(pkg, wrapper_name)).bind_partial(
        **captured["tune_kwargs"]
    )
    assert buffer.destroyed


def test_tune_one_rejects_live_token_cache_alias_before_input_creation(
    monkeypatch,
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90 import (
        pull_style_cutedsl_megakernel as pkg,
    )

    args = tune._parse_args(_argv("sm90_mxfp4", "--live-tokens", "8"))
    monkeypatch.setattr(
        pkg,
        "require_hopper_mxfp4_fused_tuning_device",
        lambda: None,
    )
    monkeypatch.setattr(
        tuner,
        "_create_canonical_inputs",
        lambda *args, **kwargs: pytest.fail("input creation must not run"),
    )

    with pytest.raises(SystemExit, match="not part of the persistent cache key"):
        tuner.tune_one(args, rank=0, world_size=4, max_tokens=512)


def test_run_tuning_uses_sm90_shared_lifecycle(monkeypatch) -> None:
    args = tune._parse_args(_argv("sm90_mxfp4"))
    captured = {}

    def fake_run(actual_args, tune_one, *, pkg) -> int:
        captured.update(args=actual_args, tune_one=tune_one, pkg=pkg)
        return 23

    monkeypatch.setattr(tuner, "_run_tuning", fake_run)
    assert tuner.run_tuning(args) == 23
    assert captured["args"] is args
    assert captured["tune_one"] is tuner.tune_one
    assert callable(captured["pkg"].autotune_hopper_mxfp4_mega_moe)


def test_offline_tuner_default_routing_is_block_permutation() -> None:
    implicit = tuner._balanced_routing(
        512, 6, 384, 0, 4, torch.device("cpu"), seed=1234
    )
    explicit = tuner._balanced_routing(
        512,
        6,
        384,
        0,
        4,
        torch.device("cpu"),
        seed=1234,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    assert torch.equal(implicit, explicit)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"local_experts": 0, "hidden": 128, "intermediate": 128}, "positive"),
        (
            {"local_experts": 1, "hidden": 129, "intermediate": 128},
            "hidden",
        ),
        (
            {"local_experts": 1, "hidden": 128, "intermediate": 129},
            "intermediate",
        ),
    ],
)
def test_raw_mxfp4_shape_contract_fails_closed(kwargs, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        tuner._raw_mxfp4_shapes(**kwargs)


def test_raw_mxfp4_shapes_match_canonical_packed_and_k32_planes() -> None:
    assert tuner._raw_mxfp4_shapes(
        local_experts=96, hidden=7168, intermediate=3072
    ) == {
        "w13": (96, 6144, 3584),
        "w13_scale": (96, 6144, 224),
        "w2": (96, 7168, 1536),
        "w2_scale": (96, 7168, 96),
    }


def test_offline_cli_checks_device_before_input_creation(monkeypatch) -> None:
    guard_name = "require_hopper_mxfp4_fused_tuning_device"
    guard = mock.Mock(side_effect=RuntimeError("device guard"))
    monkeypatch.setattr(
        sm90_mega,
        guard_name,
        guard,
    )
    with pytest.raises(RuntimeError, match="device guard"):
        tuner.tune_one(
            SimpleNamespace(),
            rank=0,
            world_size=4,
            max_tokens=8,
        )
    guard.assert_called_once_with()


@pytest.mark.parametrize(
    "profile",
    [
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    ],
)
@pytest.mark.parametrize("rank", [0, 3])
def test_offline_routing_forwards_profile_seed_and_selects_rank(
    monkeypatch, profile, rank
):
    routes = np.arange(4 * 8 * 6, dtype=np.int32).reshape(4, 8, 6)
    generate = mock.Mock(return_value=routes)
    monkeypatch.setattr(tuner, "generate_sm90_routing_numpy", generate)
    actual = tuner._balanced_routing(
        8, 6, 384, rank, 4, torch.device("cpu"), seed=1234, routing_profile=profile
    )
    generate.assert_called_once_with(
        routing_profile=profile,
        world_size=4,
        tokens=8,
        topk=6,
        total_experts=384,
        seed=1234,
    )
    assert torch.equal(actual, torch.from_numpy(routes[rank].astype(np.int64)))
