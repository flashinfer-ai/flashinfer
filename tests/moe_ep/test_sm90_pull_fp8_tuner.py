"""Host-side tests for the SM90 pull FP8 tuner / knob-cache / autotune stack.

No kernel compile or GPU launch: knob taxonomy, validity, heuristic-table
parity, cache round-trips, and the shim/backend knob wiring contracts.
"""

from __future__ import annotations

import dataclasses
import json
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest


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
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.autotune import (  # noqa: E501
            _sweep_geometries,
        )

        cands = pkg.hopper_fp8_candidates(fp8_scale_mode=scale_mode, max_tokens=2048)
        # Every geometry that currently appears in the upstream heuristic
        # tables must retain both validated token-back variants. The number of
        # distinct geometries is table-derived and may change when PR4688's
        # blockwise table is refreshed.
        for geometry in _sweep_geometries():
            for token_back in ("epi_warps", "reuse_dispatch_warps"):
                assert {**geometry, "token_back_mode": token_back} in cands
        assert all(pkg.is_valid(c) for c in cands)
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

    def test_online_clamped_winner_replays_only_for_same_normalized_clamp(
        self, tmp_path, monkeypatch
    ):
        pkg = _pkg()
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (  # noqa: E501
            autotune as autotune_module,
        )

        cache_path = tmp_path / "cache.json"
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(cache_path))
        winner = {
            **pkg.default_knobs(64, fp8_scale_mode="per_tensor"),
            "flag_batch": 8,
        }
        frontend_config = SimpleNamespace(
            rank=0,
            kind="fp8_e4m3",
            fp8_scale_mode="per_tensor",
            world_size=1,
            hidden=1024,
            intermediate=512,
            num_total_experts=8,
            num_topk=4,
            num_tokens_per_rank=64,
            gate_up_clamp=10.0,
        )
        frontend = SimpleNamespace(config=frontend_config, release=mock.Mock())
        symm_buffer = SimpleNamespace(_frontend=frontend)

        def fake_autotune(actual_frontend, launch, candidates, *, on_winner, **kwargs):
            assert actual_frontend is frontend
            assert candidates == [winner]
            on_winner(winner, 123e-6)
            return winner

        monkeypatch.setattr(autotune_module, "autotune_knobs", fake_autotune)
        assert (
            pkg.autotune_hopper_fp8_mega_moe(
                object(),
                object(),
                object(),
                symm_buffer,
                candidates=[winner],
            )
            == winner
        )

        entry = json.loads(cache_path.read_text())["entries"][0]
        assert entry["source"] == "autotune"
        assert entry["gate_up_clamp"] == 10.0

        # Exercise alias normalization: activation_clamp must resolve the
        # cache entry written under the canonical gate_up_clamp key.
        with pytest.warns(DeprecationWarning, match="activation_clamp is deprecated"):
            replay = pkg.resolve_hopper_fp8_mega_moe_config(
                8,
                64,
                4,
                1024,
                512,
                0,
                1,
                activation_clamp=10.0,
            )
        assert replay.gate_up_clamp == 10.0
        assert replay.flag_batch == 8

        # Clamp is a cache identity axis: an unclamped call must not consume
        # the tuned clamped entry and therefore keeps the config default.
        unclamped = pkg.resolve_hopper_fp8_mega_moe_config(
            8,
            64,
            4,
            1024,
            512,
            0,
            1,
        )
        assert unclamped.gate_up_clamp is None
        assert unclamped.flag_batch == 1

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
    @pytest.mark.parametrize("knobs", [{"flag_batch": 4}, {}, "auto"])
    def test_symm_buffer_rejects_knobs_plus_manual_geometry(self, knobs):
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
                knobs=knobs,
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

    def test_backend_autotune_uses_ep_subgroup(self):
        pkg = _pkg()
        from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (  # noqa: E501
            Sm90PullFp8MegaKernelBackend,
        )

        cfg = Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
            intermediate_size=1024,
            top_k=4,
            knobs="auto",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = Sm90PullFp8MegaKernelBackend(cfg)

        # Model an EP=4 subgroup embedded in a larger global process group.
        # The autotuner must receive this exact subgroup rather than falling
        # back to torch.distributed's global group.
        ep_group = object()
        backend._ep_bootstrap = object()
        backend._ep_rank = 2
        backend._ep_world_size = 4
        backend._ep_comm_group = ep_group
        output = SimpleNamespace(shape=(8, 1024))
        transformed_weights = (object(), object())
        workspace = object()

        with (
            mock.patch.object(pkg, "autotune_hopper_fp8_mega_moe") as tune,
            mock.patch.object(
                pkg, "hopper_fp8_mega_moe", return_value=object()
            ) as launch,
        ):
            result = backend.compute(
                workspace,
                transformed_weights,
                output=output,
            )

        assert result is output
        assert tune.call_args.kwargs["process_group"] is ep_group
        assert launch.call_count == 1
        assert not backend._autotune_pending

    def test_compile_key_covers_latest_kernel_knobs(self):
        pkg = _pkg()
        cfg = pkg.MegaMoEHopperFp8Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=4,
            num_total_experts=4,
            hidden=1024,
            intermediate=512,
            token_back_mode="reuse_dispatch_warps",
            grouped_token_back=True,
        )
        baseline = pkg.MegaMoEHopperFp8Frontend(cfg)._mega_compile_key()
        variants = {
            "dedup_dispatch": True,
            "grouped_token_back": False,
            "combine_format": "32e4m3xe8m0",
            "active_dispatch_warps": 2,
            "fc1_store_offload": False,
            "fc1_early_done_publish": True,
            "fold_producer_warps": False,
        }
        for name, value in variants.items():
            candidate = dataclasses.replace(cfg, **{name: value})
            assert (
                pkg.MegaMoEHopperFp8Frontend(candidate)._mega_compile_key() != baseline
            ), name

    def test_pool_key_uses_resolved_tactic_not_selector(self, tmp_path, monkeypatch):
        pkg = _pkg()
        from flashinfer.moe_ep import Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (  # noqa: E501
            Sm90PullFp8MegaKernelBackend,
        )

        fleet = SimpleNamespace(
            num_experts=8,
            max_tokens_per_rank=64,
            token_hidden_size=1024,
        )
        group = object()

        def make_backend(knobs):
            backend = Sm90PullFp8MegaKernelBackend(
                Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
                    intermediate_size=512,
                    top_k=4,
                    knobs=knobs,
                )
            )
            backend._ep_bootstrap = object()
            backend._ep_rank = 0
            backend._ep_world_size = 1
            backend._ep_comm_group = group
            return backend

        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        explicit = pkg.default_knobs(64, fp8_scale_mode="per_tensor")
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.is_available", return_value=False),
        ):
            heuristic_key = make_backend(None)._workspace_pool_key(fleet)
            explicit_key = make_backend(explicit)._workspace_pool_key(fleet)
        assert heuristic_key == explicit_key
        assert isinstance(heuristic_key[-1], pkg.MegaMoEHopperFp8Config)

        cache_path = tmp_path / "cache.json"
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(cache_path))
        cache_key = dict(
            dtype="fp8_e4m3",
            fp8_scale_mode="per_tensor",
            world_size=1,
            hidden=1024,
            intermediate=512,
            num_experts=8,
            topk=4,
            max_tokens=64,
        )
        first = {**explicit, "active_dispatch_warps": 2}
        second = {**explicit, "active_dispatch_warps": 4}
        with (
            mock.patch("torch.cuda.current_device", return_value=0),
            mock.patch("torch.cuda.is_available", return_value=False),
            mock.patch.object(
                pkg.MegaMoEHopperFp8Frontend,
                "_ensure_mega_compiled",
                side_effect=AssertionError("pool-key resolution must not JIT"),
            ),
        ):
            pkg.record_knobs(first, **cache_key)
            first_key = make_backend(None)._workspace_pool_key(fleet)
            pkg.record_knobs(second, **cache_key)
            second_key = make_backend(None)._workspace_pool_key(fleet)

        assert first_key != second_key
        assert first_key[-1].active_dispatch_warps == 2
        assert second_key[-1].active_dispatch_warps == 4

    def test_prepare_workspace_binds_key_and_allocator_to_one_resolution(self):
        pkg = _pkg()
        from flashinfer.moe_ep import (
            BootstrapConfig,
            FleetParams,
            Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig,
        )
        from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl.backend import (  # noqa: E501
            Sm90PullFp8MegaKernelBackend,
        )
        from flashinfer.moe_ep.core.kernel import workspace_pool

        fleet = FleetParams(
            num_experts=8,
            max_tokens_per_rank=64,
            token_hidden_size=1024,
        )
        backend = Sm90PullFp8MegaKernelBackend(
            Sm90_Fp8_Fp8_Bf16_PullCutedsl_MegaMoeConfig(
                intermediate_size=512,
                top_k=4,
            )
        )
        first = pkg.MegaMoEHopperFp8Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=4,
            num_total_experts=8,
            hidden=1024,
            intermediate=512,
            active_dispatch_warps=1,
        )
        second = dataclasses.replace(first, active_dispatch_warps=2)
        workspace = object()
        captured = {}

        def acquire(key, factory):
            captured["key"] = key
            return factory()

        with (
            mock.patch.object(
                pkg,
                "resolve_hopper_fp8_mega_moe_config",
                side_effect=(first, second),
            ) as resolver,
            mock.patch.object(
                pkg,
                "_get_symm_buffer_for_hopper_fp8_mega_moe_from_resolved_config",
                return_value=workspace,
            ) as allocator,
            mock.patch.object(workspace_pool, "acquire_workspace", side_effect=acquire),
            mock.patch("torch.cuda.current_device", return_value=0),
        ):
            result = backend.prepare_workspace(
                BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
                fleet,
            )

        assert result is workspace
        resolver.assert_called_once()
        assert captured["key"][-1] is first
        assert allocator.call_args.args[0] is first


def test_offline_schedule_lookup_receives_gate_up_clamp(monkeypatch):
    from flashinfer.moe_ep.backends.mega.kernel.sm90.fp8_fp8_bf16_pull_cutedsl import (  # noqa: E501
        tuner as backend_tuner,
    )

    pkg = _pkg()
    captured = {}

    class FakeBuffer:
        destroyed = False

        def destroy(self):
            self.destroyed = True

    symm_buffer = FakeBuffer()
    base = pkg.default_knobs(64, fp8_scale_mode="per_tensor")

    def fake_resolve(**kwargs):
        captured["resolve"] = kwargs
        return base, "cache"

    def fake_finish(*args, **kwargs):
        captured["candidates"] = args[8]
        return {"winner": base}

    monkeypatch.setattr(
        pkg,
        "create_dummy_hopper_fp8_inputs",
        lambda *args, **kwargs: ("y", "l1", "l2", symm_buffer),
    )
    monkeypatch.setattr(pkg, "hopper_fp8_candidates", lambda **kwargs: [base])
    monkeypatch.setattr(pkg, "resolve_knobs", fake_resolve)
    monkeypatch.setattr(backend_tuner, "schedule_candidates", lambda value: [value])
    monkeypatch.setattr(backend_tuner, "finish_sweep", fake_finish)

    args = SimpleNamespace(
        dtype="sm90_fp8_e4m3",
        live_tokens=None,
        num_experts=8,
        topk=4,
        hidden=1024,
        intermediate=512,
        fp8_scale_mode="per_tensor",
        gate_up_clamp=10.0,
        seed=0,
        sweep="schedule",
        base_knobs=None,
    )
    assert backend_tuner.tune_one(args, rank=0, world_size=1, max_tokens=64) == {
        "winner": base
    }
    assert captured["resolve"]["gate_up_clamp"] == 10.0
    assert captured["candidates"] == [base]
    assert symm_buffer.destroyed
