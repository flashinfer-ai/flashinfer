"""Host-side tests for the SM90 pull FP8 tuner / knob-cache / autotune stack.

No kernel compile or GPU launch: knob taxonomy, validity, heuristic-table
parity, cache round-trips, and the shim/backend knob wiring contracts.
"""

from __future__ import annotations

import dataclasses
import json
import warnings

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
        cands = pkg.hopper_fp8_candidates(fp8_scale_mode=scale_mode, max_tokens=2048)
        # 16 sweep geometries x {epi_warps, reuse_dispatch_warps}.
        assert len(cands) >= 30
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
