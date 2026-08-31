"""CPU contracts for the SM90 MXFP4 backend-local offline tuner CLI."""

from __future__ import annotations

import ast
import hashlib
import inspect
import textwrap
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from flashinfer.moe_ep import tune
from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_mxfp4_bf16_pull_cutedsl import (
    tuner,
)
from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)


def _argv(dtype: str = "sm90_mxfp4", *extra: str) -> list[str]:
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
        "8",
        "512",
        *extra,
    ]


def test_mxfp4_defaults_select_fixed_format_and_fused_identity() -> None:
    args = tune._parse_args(_argv())
    assert args.dtype == "sm90_mxfp4"
    assert args.execution_mode == "fused"
    assert args.fp8_scale_mode == "mxfp4_hybrid"
    assert args.gate_up_clamp == 10.0
    assert args.routing_profile == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    assert args.warmup_iters == 3
    assert args.timed_iters == 10
    assert tune._argument_error(args) is None


def test_existing_sm90_fp8_cli_default_remains_per_tensor() -> None:
    args = tune._parse_args(_argv("sm90_fp8_e4m3"))
    assert args.execution_mode == "fused"
    assert args.fp8_scale_mode == "per_tensor"
    assert args.gate_up_clamp is None


def test_mxfp4_explicit_gate_up_clamp_overrides_canonical_default() -> None:
    args = tune._parse_args(_argv("sm90_mxfp4", "--gate-up-clamp", "7.5"))
    assert args.gate_up_clamp == 7.5
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
    assert (
        captured["args"].routing_profile
        == SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    )


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


def test_non_mxfp4_rejects_split_and_hybrid_scale() -> None:
    split = tune._parse_args(_argv("sm90_fp8_e4m3", "--execution-mode", "split"))
    assert "only wired" in (tune._argument_error(split) or "")
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


def test_candidate_union_uses_only_mode_specific_shim_api() -> None:
    calls = []

    class FakePackage:
        @staticmethod
        def hopper_mxfp4_candidates(*, execution_mode: str, routing_profile: str):
            calls.append((execution_mode, routing_profile))
            return [
                {
                    "execution_mode": execution_mode,
                    "routing_profile": routing_profile,
                }
            ]

    assert tuner._candidate_union(
        FakePackage,
        execution_mode="fused",
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    ) == [
        {
            "execution_mode": "fused",
            "routing_profile": SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        }
    ]
    assert tuner._candidate_union(
        FakePackage,
        execution_mode="split",
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    ) == [
        {
            "execution_mode": "split",
            "routing_profile": SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        }
    ]
    assert calls == [
        ("fused", SM90_ROUTING_PROFILE_BLOCK_PERMUTATION),
        ("split", SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED),
    ]


def test_empty_candidate_union_fails_closed() -> None:
    package = SimpleNamespace(
        hopper_mxfp4_candidates=lambda *, execution_mode, routing_profile: []
    )
    with pytest.raises(RuntimeError, match="empty manifest-derived"):
        tuner._candidate_union(
            package,
            execution_mode="fused",
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        )


def test_bucket_default_is_reordered_from_but_not_added_to_union() -> None:
    first = {"candidate": "first"}
    default = {"candidate": "default"}
    package = SimpleNamespace(
        hopper_mxfp4_candidates=lambda *, execution_mode, routing_profile: [
            first,
            default,
        ],
        hopper_mxfp4_default_tactic=(
            lambda max_tokens, *, execution_mode, routing_profile: default
        ),
        is_hopper_mxfp4_tactic_shape_compatible=lambda candidate, **kwargs: True,
    )
    assert tuner._ordered_candidates(
        package,
        execution_mode="split",
        max_tokens=512,
        hidden=128,
        intermediate=128,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    ) == [default, first]


def test_bucket_default_absent_from_union_fails_closed() -> None:
    package = SimpleNamespace(
        hopper_mxfp4_candidates=lambda *, execution_mode, routing_profile: [
            {"candidate": "union"}
        ],
        hopper_mxfp4_default_tactic=(
            lambda max_tokens, *, execution_mode, routing_profile: {
                "candidate": "missing"
            }
        ),
        is_hopper_mxfp4_tactic_shape_compatible=lambda candidate, **kwargs: True,
    )
    with pytest.raises(RuntimeError, match="default is absent"):
        tuner._ordered_candidates(
            package,
            execution_mode="fused",
            max_tokens=8,
            hidden=128,
            intermediate=128,
            routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
        )


@pytest.mark.parametrize(
    ("mode", "wrapper_name"),
    [
        ("fused", "autotune_hopper_mxfp4_mega_moe"),
        ("split", "autotune_hopper_mxfp4_split_mega_moe"),
    ],
)
def test_tune_one_routes_each_mode_to_its_dedicated_wrapper(
    monkeypatch, mode: str, wrapper_name: str
) -> None:
    from flashinfer.moe_ep.kernel_src.sm90 import (
        pull_style_cutedsl_megakernel as pkg,
    )

    args = tune._parse_args(_argv("sm90_mxfp4", "--execution-mode", mode))
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
        execution_mode,
        initial_tactic,
    ):
        captured["create"] = (
            actual_args,
            rank,
            world_size,
            max_tokens,
            live_tokens,
            execution_mode,
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
        return {"winner": mode}

    monkeypatch.setattr(
        "flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel."
        "shim.mxfp4_tuner.require_hopper_mxfp4_tuning_device",
        lambda: None,
    )
    monkeypatch.setattr(tuner, "_create_canonical_inputs", fake_create)
    monkeypatch.setattr(tuner, "finish_sweep", fake_finish)
    assert tuner.tune_one(args, rank=0, world_size=4, max_tokens=8) == {"winner": mode}

    expected = pkg.hopper_mxfp4_ordered_candidates(
        8,
        execution_mode=mode,
        routing_profile=args.routing_profile,
        hidden=args.hidden,
        intermediate=args.intermediate,
    )
    assert captured["create"][5:] == (mode, expected[0])
    assert captured["finish"][-2] == expected
    assert captured["finish"][-1] is getattr(pkg, wrapper_name)
    assert captured["tune_kwargs"] == {
        "gate_up_clamp": 10.0,
        "routing_profile": args.routing_profile,
    }
    assert buffer.destroyed


def test_tune_one_create_inputs_call_matches_function_contract() -> None:
    """Guard the real call site against duplicating max_tokens."""
    source = textwrap.dedent(inspect.getsource(tuner.tune_one))
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_create_canonical_inputs"
    ]
    assert len(calls) == 1
    assert [ast.unparse(arg) for arg in calls[0].args] == [
        "args",
        "rank",
        "world_size",
        "max_tokens",
        "live_tokens",
        "mode",
        "candidates[0]",
    ]
    inspect.signature(tuner._create_canonical_inputs).bind(
        object(), 0, 4, 8, 8, "fused", {}
    )


def test_run_tuning_accepts_split_and_uses_sm90_shared_lifecycle(monkeypatch) -> None:
    args = tune._parse_args(_argv("sm90_mxfp4", "--execution-mode", "split"))
    captured = {}

    def fake_run(actual_args, tune_one, *, pkg) -> int:
        captured.update(args=actual_args, tune_one=tune_one, pkg=pkg)
        return 23

    monkeypatch.setattr(tuner, "_run_tuning", fake_run)
    assert tuner.run_tuning(args) == 23
    assert captured["args"] is args
    assert captured["tune_one"] is tuner.tune_one
    assert callable(captured["pkg"].autotune_hopper_mxfp4_split_mega_moe)


@pytest.mark.parametrize("tokens", [8, 32, 64, 128, 256, 512, 1024, 2048])
def test_canonical_exact_balanced_routing_invariants(tokens: int) -> None:
    first = tuner._balanced_routing(
        tokens, 6, 384, 0, 4, torch.device("cpu"), seed=1234
    )
    second = tuner._balanced_routing(
        tokens, 6, 384, 0, 4, torch.device("cpu"), seed=1234
    )
    other_rank = tuner._balanced_routing(
        tokens, 6, 384, 1, 4, torch.device("cpu"), seed=1234
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, other_rank)
    assert first.shape == (tokens, 6)
    assert torch.all(first >= 0) and torch.all(first < 384)
    assert all(len(set(row.tolist())) == 6 for row in first)

    routes = np.stack(
        [
            tuner._balanced_routing(
                tokens, 6, 384, rank, 4, torch.device("cpu"), seed=1234
            ).numpy()
            for rank in range(4)
        ]
    )
    expert_counts = np.bincount(routes.reshape(-1), minlength=384)
    assert int(expert_counts.max()) - int(expert_counts.min()) <= 1
    for source_rank in range(4):
        owner_counts = np.bincount(routes[source_rank].reshape(-1) // 96, minlength=4)
        assert np.all(owner_counts == tokens * 6 // 4)


def test_offline_tuner_implements_both_distinct_routing_profiles() -> None:
    exact = tuner._balanced_routing(
        512,
        6,
        384,
        0,
        4,
        torch.device("cpu"),
        seed=1234,
        routing_profile=SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    )
    block = tuner._balanced_routing(
        512,
        6,
        384,
        0,
        4,
        torch.device("cpu"),
        seed=1234,
        routing_profile=SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    )
    assert not torch.equal(exact, block)
    assert torch.bincount(exact.reshape(-1), minlength=384).max() == 8
    block_counts = torch.bincount(block.reshape(-1), minlength=384)
    assert int(block_counts.max()) > int(block_counts.min())


@pytest.mark.parametrize(
    ("tokens", "expected_hash"),
    [
        (8, "1ba40a6fb0ab731b9085979a1968c60aa6b3a5fa3e13b444f2c0a55bcfb8aa00"),
        (32, "5499b7ae730372fb6ae53f29b852a07ec10aeeaa11d1e4e99424f3edd9be16ce"),
        (64, "d78a2b4df5bb769238a2528a76ccf80a980074bdb02c88152fbefbbd0d21e90e"),
        (128, "1209a05edefdc700fb8d45b54c2291b62d410cdc2934e801ca05f0a84a38b06f"),
        (256, "415b8f862a97ea9cc498bbc150e1e6d5d7b7111c27b387dcba95169386b1d7e2"),
        (512, "f5306ed4f8d1fd685fedf370c96e942f715b9481367be5932200e36d444379de"),
        (1024, "5999065601264efc000004684321ef46c4c1996b6531ecdbd985e8a617ec7dd5"),
        (2048, "85c6311af059960c02445ee051e3950991f34dc1979e537081d00c7f5da40b53"),
    ],
)
def test_canonical_routing_matches_published_donor_global_i64le_hash(
    tokens: int, expected_hash: str
) -> None:
    routes = np.stack(
        [
            tuner._balanced_routing(
                tokens, 6, 384, rank, 4, torch.device("cpu"), seed=1234
            ).numpy()
            for rank in range(4)
        ]
    )
    canonical = np.ascontiguousarray(routes, dtype="<i8")
    assert hashlib.sha256(canonical.tobytes()).hexdigest() == expected_hash
    counts = np.bincount(canonical.reshape(-1), minlength=384)
    assert int(counts.sum()) == 4 * tokens * 6
    assert int(counts.max()) - int(counts.min()) <= 1


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
