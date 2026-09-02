"""Host-side tests for the SM90 pull FP8 tuner / knob-cache / autotune stack.

No kernel compile or GPU launch: knob taxonomy, validity, heuristic-table
parity, cache round-trips, and the shim/backend knob wiring contracts.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from unittest import mock

import pytest
import torch


def _pkg():
    from flashinfer.moe_ep.kernel_src.sm90 import pull_style_cutedsl_megakernel

    return pull_style_cutedsl_megakernel


class TestDefaultKnobs:
    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    @pytest.mark.parametrize("tokens", [8, 100, 2048, 32768, 10**6])
    def test_matches_heuristic_table(self, scale_mode, tokens):
        pkg = _pkg()
        pkg.bootstrap_paths()
        from moe_hopper_fp8.heuristic_config import select_heuristic_config

        knobs = pkg.default_knobs(tokens, fp8_scale_mode=scale_mode)
        sel = select_heuristic_config(scale_mode, tokens)
        assert knobs["swap_ab"] == sel.config.swap_ab
        assert knobs["pingpong"] == sel.config.pingpong
        assert knobs["mma_tiler_mnk"] == tuple(sel.config.mma_tiler_mnk)
        assert knobs["cluster_shape_mnk"] == tuple(sel.config.cluster_shape_mnk)
        assert knobs["fp8_accum_mode"] == sel.config.accum_mode
        assert knobs["token_back_mode"] == sel.config.token_back_mode
        assert pkg.is_valid(knobs)


class TestIsValid:
    def test_geometry_rules(self):
        pkg = _pkg()
        ok = dict(swap_ab=False, pingpong=False, mma_tiler_mnk=(64, 128, 128))
        assert pkg.is_valid(ok)
        # native tile must be M=64.
        assert not pkg.is_valid({**ok, "mma_tiler_mnk": (128, 128, 128)})
        # swap-AB N=256 is illegal.
        assert not pkg.is_valid(dict(swap_ab=True, mma_tiler_mnk=(128, 256, 128)))
        # ping-pong tile coupling.
        assert not pkg.is_valid(
            dict(swap_ab=False, pingpong=True, mma_tiler_mnk=(64, 256, 128))
        )
        assert not pkg.is_valid(
            dict(swap_ab=True, pingpong=True, mma_tiler_mnk=(256, 32, 128))
        )
        assert pkg.is_valid(
            dict(swap_ab=True, pingpong=True, mma_tiler_mnk=(128, 32, 128))
        )
        # cluster domain.
        assert not pkg.is_valid({**ok, "cluster_shape_mnk": (4, 1, 1)})
        assert not pkg.is_valid({**ok, "cluster_shape_mnk": (1, 1, 2)})
        assert pkg.is_valid({**ok, "cluster_shape_mnk": (2, 2, 1)})
        # ikr requires apply_topk_in_fc1.
        assert not pkg.is_valid(
            {**ok, "in_kernel_fc2_reduce": True}, apply_topk_in_fc1=False
        )

    def test_iter_candidates_all_valid(self):
        pkg = _pkg()
        seen = 0
        for knobs in pkg.iter_candidates():
            assert pkg.is_valid(knobs)
            seen += 1
            if seen >= 500:
                break
        assert seen > 0

    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    def test_autotune_candidates(self, scale_mode):
        pkg = _pkg()
        cands = pkg.hopper_fp8_candidates(fp8_scale_mode=scale_mode, max_tokens=2048)
        # 16 sweep geometries x {epi_warps, reuse_dispatch_warps}.
        assert len(cands) >= 30
        assert all(pkg.is_valid(c) for c in cands)
        # standalone_warps is a correctness-supported explicit mode, but is
        # intentionally outside the perf sweep until measured winners exist.
        assert {c["token_back_mode"] for c in cands} == {
            "epi_warps",
            "reuse_dispatch_warps",
        }
        # heuristic winner leads (ties keep the established default).
        assert cands[0] == pkg.default_knobs(2048, fp8_scale_mode=scale_mode)
        # deduplicated.
        keys = [json.dumps(c, sort_keys=True, default=list) for c in cands]
        assert len(keys) == len(set(keys))


class TestWithKnobs:
    def _cfg(self, **overrides):
        pkg = _pkg()
        base = dict(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=4,
            num_total_experts=4,
            hidden=1024,
            intermediate=512,
        )
        base.update(overrides)
        return pkg.MegaMoEHopperFp8Config(**base)

    def test_applies_declared_fields(self):
        pkg = _pkg()
        cfg = self._cfg()
        knobs = dict(
            swap_ab=True,
            pingpong=True,
            mma_tiler_mnk=(128, 32, 128),
            cluster_shape_mnk=(1, 2, 1),
            flag_batch=4,
            token_back_mode="reuse_dispatch_warps",
            not_a_field=123,  # silently dropped
        )
        out = pkg.with_knobs(cfg, knobs)
        assert out.swap_ab and out.pingpong
        assert out.mma_tiler_mnk == (128, 32, 128)
        assert out.cluster_shape_mnk == (1, 2, 1)
        assert out.flag_batch == 4
        assert out.resolved_token_back_mode == "reuse_dispatch_warps"
        assert not hasattr(out, "not_a_field")

    def test_none_is_identity_and_invalid_raises(self):
        pkg = _pkg()
        cfg = self._cfg()
        assert pkg.with_knobs(cfg, None) is cfg
        with pytest.raises(ValueError):
            pkg.with_knobs(cfg, dict(swap_ab=True, mma_tiler_mnk=(64, 128, 128)))


class TestKnobCache:
    _KEY = dict(
        dtype="fp8_e4m3",
        fp8_scale_mode="per_tensor",
        world_size=4,
        hidden=7168,
        intermediate=3072,
        num_experts=384,
        topk=6,
    )

    def test_record_lookup_roundtrip_and_bucketing(self, tmp_path, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
        knobs = pkg.default_knobs(2048, fp8_scale_mode="per_tensor")
        assert (
            pkg.record_knobs(knobs, **self._KEY, max_tokens=2048, device="test-gpu")
            is not None
        )
        # Exact bucket.
        got = pkg.lookup_knobs(**self._KEY, max_tokens=2048, device="test-gpu")
        assert got == knobs
        # Smaller request rounds UP to the recorded bucket.
        assert pkg.lookup_knobs(**self._KEY, max_tokens=100, device="test-gpu") == knobs
        # Larger request falls back to the largest recorded bucket.
        assert (
            pkg.lookup_knobs(**self._KEY, max_tokens=8192, device="test-gpu") == knobs
        )
        # Key isolation: a different scale mode / topk never matches.
        assert (
            pkg.lookup_knobs(
                **{**self._KEY, "fp8_scale_mode": "blockwise"},
                max_tokens=2048,
                device="test-gpu",
            )
            is None
        )
        assert (
            pkg.lookup_knobs(
                **{**self._KEY, "topk": 8}, max_tokens=2048, device="test-gpu"
            )
            is None
        )

    def test_resolve_falls_back_to_heuristic(self, tmp_path, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
        knobs, source = pkg.resolve_knobs(**self._KEY, max_tokens=4096)
        assert source == "heuristic"
        assert knobs == pkg.default_knobs(4096, fp8_scale_mode="per_tensor")

    def test_resolve_returns_recorded_cache_hit(self, tmp_path, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
        knobs = pkg.default_knobs(2048, fp8_scale_mode="per_tensor")
        assert pkg.record_knobs(knobs, **self._KEY, max_tokens=2048) is not None
        got, source = pkg.resolve_knobs(**self._KEY, max_tokens=2048)
        assert source == "cache"
        assert got == knobs

    def test_every_session_axis_isolated(self, tmp_path, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
        knobs = pkg.default_knobs(2048, fp8_scale_mode="per_tensor")
        assert (
            pkg.record_knobs(knobs, **self._KEY, max_tokens=2048, device="test-gpu")
            is not None
        )
        variants = [
            {"dtype": "fp8_e5m2"},
            {"fp8_scale_mode": "blockwise"},
            {"world_size": 2},
            {"hidden": self._KEY["hidden"] + 128},
            {"intermediate": self._KEY["intermediate"] + 128},
            {"num_experts": self._KEY["num_experts"] + 4},
            {"topk": self._KEY["topk"] + 1},
        ]
        for changed in variants:
            key = {**self._KEY, **changed}
            assert pkg.lookup_knobs(**key, max_tokens=2048, device="test-gpu") is None
        assert (
            pkg.lookup_knobs(**self._KEY, max_tokens=2048, device="other-gpu") is None
        )

    def test_disabled_cache(self, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        assert pkg.knob_cache_path() is None
        assert pkg.lookup_knobs(**self._KEY, max_tokens=8) is None

    def test_sm100_entries_never_cross_match(self, tmp_path, monkeypatch):
        pkg = _pkg()
        path = tmp_path / "cache.json"
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
        # An SM100-shaped entry (no fp8_scale_mode field, nvfp4 dtype).
        path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "entries": [
                        dict(
                            device="test-gpu",
                            dtype="nvfp4",
                            world_size=4,
                            hidden=7168,
                            intermediate=3072,
                            num_experts=384,
                            topk=6,
                            combine_dtype="bf16",
                            max_tokens=2048,
                            knobs={"flag_batch": 8},
                        )
                    ],
                }
            )
        )
        assert pkg.lookup_knobs(**self._KEY, max_tokens=2048, device="test-gpu") is None


class TestShimAndBackendWiring:
    def test_symm_buffer_rejects_knobs_plus_manual_geometry(self):
        pkg = _pkg()
        with pytest.raises(ValueError, match="not both"):
            pkg.get_symm_buffer_for_hopper_fp8_mega_moe(
                384,
                64,
                6,
                7168,
                3072,
                0,
                1,
                knobs={"flag_batch": 4},
                swap_ab=False,
            )

    def test_backend_config_knobs_field(self):
        from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig

        cfg = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=1024, top_k=4
        )
        assert cfg.knobs is None
        auto = dataclasses.replace(cfg, knobs="auto")
        assert auto.knobs == "auto"

    def test_backend_rejects_bad_knobs_value_and_geometry_conflict(self):
        from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (  # noqa: E501
            Sm90PullFp8MegaKernelBackend,
        )

        base = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=1024, top_k=4
        )
        with pytest.raises(ValueError, match="knobs must be"):
            Sm90PullFp8MegaKernelBackend(dataclasses.replace(base, knobs="Auto"))
        for knobs in ("auto", {"flag_batch": 4}):
            with pytest.raises(ValueError, match="mutually exclusive"):
                Sm90PullFp8MegaKernelBackend(
                    dataclasses.replace(base, knobs=knobs, swap_ab=True)
                )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # knobs="auto" collective warning
            backend = Sm90PullFp8MegaKernelBackend(
                dataclasses.replace(base, knobs="auto")
            )
        assert backend._autotune_pending

    def test_frontend_compile_key_covers_regression_matrix_axes(self):
        pkg = _pkg()
        base = pkg.MegaMoEHopperFp8Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=4,
            num_total_experts=4,
            hidden=1024,
            intermediate=512,
        )
        variants = [
            dataclasses.replace(base, fp8_scale_mode="blockwise"),
            dataclasses.replace(base, fp8_accum_mode="2xacc"),
            dataclasses.replace(base, swap_ab=True, mma_tiler_mnk=(256, 32, 128)),
            dataclasses.replace(base, pingpong=True),
            dataclasses.replace(base, cluster_shape_mnk=(2, 2, 1)),
            dataclasses.replace(base, token_back_mode="standalone_warps"),
            dataclasses.replace(base, in_kernel_fc2_reduce=True),
            dataclasses.replace(base, load_balance_mode="atomic_counter"),
        ]
        baseline = pkg.MegaMoEHopperFp8Frontend(base)._mega_compile_key()
        changed = [
            pkg.MegaMoEHopperFp8Frontend(cfg)._mega_compile_key() for cfg in variants
        ]
        assert all(key != baseline for key in changed)
        assert len(set(changed)) == len(changed)

    def test_backend_workspace_pool_key_covers_regression_matrix_axes(self):
        from flashinfer.moe_ep import (
            FleetParams,
            Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (  # noqa: E501
            Sm90PullFp8MegaKernelBackend,
        )

        base = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=512,
            top_k=4,
        )
        fleet = FleetParams(
            num_experts=8,
            max_tokens_per_rank=64,
            token_hidden_size=1024,
        )
        group = object()

        def pool_key(config, problem=fleet):
            backend = Sm90PullFp8MegaKernelBackend(config)
            backend._ep_bootstrap = object()
            backend._ep_rank = 1
            backend._ep_world_size = 4
            backend._ep_comm_group = group
            return backend._workspace_pool_key(problem)

        config_variants = [
            dataclasses.replace(base, kind="fp8_e5m2"),
            dataclasses.replace(base, fp8_scale_mode="blockwise"),
            dataclasses.replace(base, fp8_accum_mode="2xacc"),
            dataclasses.replace(
                base,
                swap_ab=True,
                pingpong=False,
                mma_tiler_mnk=(256, 32, 128),
                cluster_shape_mnk=(1, 1, 1),
            ),
            dataclasses.replace(
                base,
                swap_ab=False,
                pingpong=True,
                mma_tiler_mnk=(64, 128, 128),
                cluster_shape_mnk=(2, 2, 1),
            ),
            dataclasses.replace(base, token_back_mode="standalone_warps"),
            dataclasses.replace(base, in_kernel_fc2_reduce=True),
            dataclasses.replace(base, load_balance_mode="atomic_counter"),
            dataclasses.replace(base, knobs={"flag_batch": 4}),
        ]
        problem_variants = [
            dataclasses.replace(fleet, num_experts=12),
            dataclasses.replace(fleet, max_tokens_per_rank=128),
            dataclasses.replace(fleet, token_hidden_size=1152),
        ]
        with mock.patch("torch.cuda.current_device", return_value=3):
            baseline = pool_key(base)
            changed = [pool_key(cfg) for cfg in config_variants]
            changed.extend(pool_key(base, problem) for problem in problem_variants)
        assert all(key != baseline for key in changed)
        assert len(set(changed)) == len(changed)


def test_backend_auto_uses_ep_singleton_group_inside_larger_global_job(
    monkeypatch,
):
    from flashinfer.moe_ep import (
        Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
    )
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (
        Sm90PullFp8MegaKernelBackend,
    )

    ep_singleton = object()
    autotune = mock.Mock(return_value={"winner": True})
    launch = mock.Mock(return_value=object())
    pkg = _pkg()
    monkeypatch.setattr(pkg, "autotune_hopper_fp8_mega_moe", autotune)
    monkeypatch.setattr(pkg, "hopper_fp8_mega_moe", launch)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda group=None: 4)
    config = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
        intermediate_size=128, top_k=2, knobs="auto"
    )
    with pytest.warns(UserWarning, match="COLLECTIVE"):
        backend = Sm90PullFp8MegaKernelBackend(config)
    backend._ep_bootstrap = object()
    backend._ep_rank = 0
    backend._ep_world_size = 1
    backend._ep_comm_group = ep_singleton

    output = torch.empty((3, 128), dtype=torch.bfloat16)
    transformed = (object(), object())
    assert backend.compute(object(), transformed, output=output) is output
    autotune.assert_called_once()
    assert autotune.call_args.kwargs["process_group"] is ep_singleton


def test_backend_local_tune_one_propagates_nondefault_clamp_to_cache_and_record(
    monkeypatch,
):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl import (
        tuner as tuner_module,
    )

    pkg = _pkg()
    symm_buffer = mock.Mock()
    create_inputs = mock.Mock(return_value=("y", "l1", "l2", symm_buffer))
    candidate = {"swap_ab": True, "mma_tiler_mnk": (128, 32, 128)}
    enumerate_candidates = mock.Mock(return_value=[candidate])
    resolve = mock.Mock(return_value=(candidate, "cache"))
    schedule = mock.Mock(return_value=[candidate])
    autotune = mock.Mock()
    finish = mock.Mock(return_value={"winner": candidate})
    monkeypatch.setattr(pkg, "create_dummy_hopper_fp8_inputs", create_inputs)
    monkeypatch.setattr(pkg, "hopper_fp8_candidates", enumerate_candidates)
    monkeypatch.setattr(pkg, "resolve_knobs", resolve)
    monkeypatch.setattr(pkg, "autotune_hopper_fp8_mega_moe", autotune)
    monkeypatch.setattr(tuner_module, "schedule_candidates", schedule)
    monkeypatch.setattr(tuner_module, "finish_sweep", finish)
    args = mock.Mock(
        dtype="sm90_fp8_e4m3",
        live_tokens=None,
        num_experts=8,
        topk=2,
        hidden=128,
        intermediate=128,
        fp8_scale_mode="per_tensor",
        gate_up_clamp=7.5,
        seed=123,
        sweep="schedule",
        base_knobs=None,
    )

    assert tuner_module.tune_one(args, rank=0, world_size=4, max_tokens=64) == {
        "winner": candidate
    }
    assert create_inputs.call_args.kwargs["gate_up_clamp"] == 7.5
    assert resolve.call_args.kwargs["gate_up_clamp"] == 7.5
    assert finish.call_args.kwargs["tune_kwargs"] == {"gate_up_clamp": 7.5}
    schedule.assert_called_once_with(candidate)
    symm_buffer.destroy.assert_called_once_with()
