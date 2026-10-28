"""Host-side tests for the SM90 pull FP8 tuner / knob-cache / autotune stack.

No kernel compile or GPU launch: knob taxonomy, validity, heuristic-table
parity, cache round-trips, and the shim/backend knob wiring contracts.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import warnings

import pytest


def _pkg():
    # The SM90 tree shares top-level kernel module names with the SM100
    # cutedsl_megamoe tree and the shim refuses to mix them in one process
    # (see shim/_paths.py).  tests/moe_ep/run_tests.sh runs this file in its
    # own interpreter; an ad-hoc whole-directory pytest run skips it instead.
    import os
    import sys

    common = sys.modules.get("common")
    common_file = getattr(common, "__file__", None) if common is not None else None
    if (
        common_file is not None
        and "pull_style_cutedsl_megakernel" not in os.path.abspath(common_file)
    ):
        pytest.skip(
            "SM100 cutedsl_megamoe kernel tree already imported in this process; "
            "run tests/moe_ep/test_sm90_pull_fp8_tuner.py in its own interpreter"
        )
    from flashinfer.moe_ep.kernel_src.sm90 import pull_style_cutedsl_megakernel

    return pull_style_cutedsl_megakernel


# Small single-rank problem for host-side config construction.
_CFG_BASE = dict(
    rank=0,
    world_size=1,
    num_tokens_per_rank=64,
    num_topk=4,
    num_total_experts=4,
    hidden=1024,
    intermediate=512,
)


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
        assert knobs["group_hint"] == sel.config.group_hint
        assert knobs["tail_split_pairs"] == sel.config.tail_split_pairs
        assert pkg.is_valid(knobs, fp8_scale_mode=scale_mode)


class TestEightRankOverrides:
    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    @pytest.mark.parametrize("tokens", [8, 256, 1024, 4096])
    def test_eight_rank_rows(self, scale_mode, tokens, monkeypatch):
        pkg = _pkg()
        pkg.bootstrap_paths()
        from moe_hopper_fp8.heuristic_config import (
            EP8_OVERRIDES,
            TAIL_SPLIT_ENV,
            select_heuristic_config,
            token_bucket,
        )

        # The env override would rewrite both rows; compare the table itself.
        monkeypatch.delenv(TAIL_SPLIT_ENV, raising=False)

        four = select_heuristic_config(scale_mode, tokens).config
        eight = select_heuristic_config(scale_mode, tokens, world_size=8).config
        assert eight == EP8_OVERRIDES[scale_mode].get(token_bucket(tokens), four)
        # Only the tuned model layout takes the eight-rank rows.
        assert (
            select_heuristic_config(
                scale_mode, tokens, world_size=8, num_experts=256
            ).config
            == four
        )
        knobs = pkg.default_knobs(tokens, fp8_scale_mode=scale_mode, world_size=8)
        assert knobs["mma_tiler_mnk"] == tuple(eight.mma_tiler_mnk)
        assert knobs["cluster_shape_mnk"] == tuple(eight.cluster_shape_mnk)
        assert knobs["group_hint"] == eight.group_hint
        assert knobs["tail_split_pairs"] == eight.tail_split_pairs
        assert knobs["token_back_mode"] == eight.token_back_mode
        assert pkg.is_valid(knobs, fp8_scale_mode=scale_mode)

    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    def test_eight_rank_default_leads_candidates(self, scale_mode, monkeypatch):
        pkg = _pkg()
        pkg.bootstrap_paths()
        from moe_hopper_fp8.heuristic_config import TAIL_SPLIT_ENV

        monkeypatch.delenv(TAIL_SPLIT_ENV, raising=False)
        cands = pkg.hopper_fp8_candidates(
            fp8_scale_mode=scale_mode, max_tokens=1024, world_size=8
        )
        assert cands[0] == pkg.default_knobs(
            1024, fp8_scale_mode=scale_mode, world_size=8
        )


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
        # tail-split pair tasks need exactly two CTAs along the tokens and
        # one along the weights: non-swap cga (2, 1, 1), swap-AB cga (1, 2, 1).
        assert pkg.is_valid(
            {**ok, "cluster_shape_mnk": (2, 1, 1), "tail_split_pairs": True}
        )
        assert not pkg.is_valid(
            {**ok, "cluster_shape_mnk": (1, 2, 1), "tail_split_pairs": True}
        )
        assert not pkg.is_valid(
            {**ok, "cluster_shape_mnk": (2, 2, 1), "tail_split_pairs": True}
        )
        swap = dict(swap_ab=True, pingpong=True, mma_tiler_mnk=(128, 128, 128))
        assert pkg.is_valid(
            {**swap, "cluster_shape_mnk": (1, 2, 1), "tail_split_pairs": True}
        )
        assert not pkg.is_valid(
            {**swap, "cluster_shape_mnk": (2, 1, 1), "tail_split_pairs": True}
        )
        # IKR requires routing weights before FC2.
        assert not pkg.is_valid(
            {**ok, "in_kernel_fc2_reduce": True},
            apply_routing_weights_before_fc2=False,
        )

    def test_blockwise_geometry_rules(self):
        pkg = _pkg()
        # Blockwise forms one 1x128 FC2-activation scale per task tile, so a
        # tile must produce 128 channels: native N=256, swap-AB M=256, and
        # never ping-pong (an N128 / M128 schedule).
        for knobs in (
            dict(swap_ab=False, mma_tiler_mnk=(64, 256, 128)),
            dict(swap_ab=True, mma_tiler_mnk=(256, 8, 128)),
        ):
            assert pkg.is_valid(knobs, fp8_scale_mode="blockwise")
        for knobs in (
            dict(swap_ab=False, mma_tiler_mnk=(64, 128, 128)),
            dict(swap_ab=False, pingpong=True, mma_tiler_mnk=(64, 128, 128)),
            dict(swap_ab=True, mma_tiler_mnk=(128, 32, 128)),
            dict(swap_ab=True, pingpong=True, mma_tiler_mnk=(128, 32, 128)),
        ):
            assert not pkg.is_valid(knobs, fp8_scale_mode="blockwise")
            # per_tensor keeps the 64-channel tiles.
            assert pkg.is_valid(knobs)
            # The config enforces the same rule.
            with pytest.raises(ValueError, match="128 output channels"):
                pkg.MegaMoEHopperFp8Config(
                    **_CFG_BASE, fp8_scale_mode="blockwise", **knobs
                )

    def test_hybrid_configs_keep_1x64_geometry(self):
        pkg = _pkg()
        # The fused MXFP4 / NVFP4 x FP8 configs validate through a blockwise
        # proxy; with 1x64 hybrid h blocks the swap-AB M=128 tiles
        # (cooperative and ping-pong) stay legal.  The default is 1x128.
        for config_cls in (pkg.MegaMoEHopperMxfp4Config, pkg.MegaMoEHopperNvfp4Config):
            assert (
                config_cls(**_CFG_BASE, mma_tiler_mnk=(256, 32, 128)).fc2_act_scale_k
                == 128
            )
            for pingpong in (False, True):
                config_cls(
                    **_CFG_BASE,
                    pingpong=pingpong,
                    mma_tiler_mnk=(128, 32, 128),
                    fc2_act_scale_k=64,
                )
            # Their kernels never received the deterministic slot path.
            with pytest.raises(ValueError, match="deterministic"):
                config_cls(**_CFG_BASE, deterministic=True)

    def test_hybrid_h128_needs_m256_cooperative(self):
        pkg = _pkg()
        # fc2_act_scale_k=128 spans both M=256 epilogue warpgroups, so it
        # takes the blockwise 1x128 geometry: no M=128, no ping-pong.
        for config_cls in (pkg.MegaMoEHopperMxfp4Config, pkg.MegaMoEHopperNvfp4Config):
            for tile_k in (128, 256):
                config = config_cls(
                    **_CFG_BASE, mma_tiler_mnk=(256, 32, tile_k), fc2_act_scale_k=128
                )
                assert config.fc2_act_scale_k == 128
            for pingpong in (False, True):
                with pytest.raises(ValueError, match="fc2_act_scale_k=128 needs"):
                    config_cls(
                        **_CFG_BASE,
                        pingpong=pingpong,
                        mma_tiler_mnk=(128, 32, 128),
                        fc2_act_scale_k=128,
                    )
            for scale_k in (32, 256, True):
                with pytest.raises(ValueError, match="fc2_act_scale_k"):
                    config_cls(**_CFG_BASE, fc2_act_scale_k=scale_k)

    def test_hybrid_h128_tuning_identity_and_fallback(self):
        pkg = _pkg()
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim import (
            mxfp4_tuner,
        )
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.nvfp4_tuner import (
            hopper_nvfp4_cache_provenance_sha256,
        )

        # 1x64 keeps the historical cache identity; 1x128 winners get their own.
        mx_prov = mxfp4_tuner.hopper_mxfp4_cache_provenance_sha256
        assert mx_prov() == mx_prov(fc2_act_scale_k=64) != mx_prov(fc2_act_scale_k=128)
        nv_shape = dict(
            routing_profile="block_permutation_v1",
            kperm=64,
            overlap=None,
            max_tokens=128,
            hidden=7168,
            intermediate=3072,
            num_experts=384,
            world_size=4,
        )
        assert hopper_nvfp4_cache_provenance_sha256(
            **nv_shape
        ) != hopper_nvfp4_cache_provenance_sha256(**nv_shape, fc2_act_scale_k=128)

        # Heuristic fallback (and autotune) keep only tactics that can form the
        # h block: M=256 cooperative for 1x128.
        shape = dict(hidden=7168, intermediate=3072)
        h128 = mxfp4_tuner._ordered_base_candidates(128, **shape, fc2_act_scale_k=128)
        assert h128
        assert all(c["mma_tiler_mnk"][0] == 256 and not c["pingpong"] for c in h128)
        h64 = mxfp4_tuner._ordered_base_candidates(128, **shape)
        assert any(c["mma_tiler_mnk"][0] == 128 for c in h64)

        # Manual mode picks an M=256 tile for the default 1x128 block.
        common = dict(
            num_total_experts=4,
            num_max_tokens=64,
            num_topk=4,
            hidden=1024,
            intermediate=512,
            rank=0,
            world_size=1,
            swap_ab=True,
        )
        for resolve in (
            pkg._resolve_hopper_mxfp4_mega_moe_config,
            pkg._resolve_hopper_nvfp4_mega_moe_config,
        ):
            config = resolve(**common)
            assert config.fc2_act_scale_k == 128
            assert config.mma_tiler_mnk[0] == 256
            config = resolve(**common, fc2_act_scale_k=64)
            assert config.mma_tiler_mnk[0] == 128

    def test_deterministic_rules(self):
        pkg = _pkg()
        ok = dict(swap_ab=True, mma_tiler_mnk=(256, 32, 128))
        assert pkg.is_valid({**ok, "deterministic": True})
        pkg.MegaMoEHopperFp8Config(**_CFG_BASE, **ok, deterministic=True)
        # Deterministic slots replace the atomic dispatch path dedup / grouped
        # token-back reserve on; the in-kernel reduce sums in arrival order.
        for extra in (
            {"dedup_dispatch": True},
            {"grouped_token_back": True, "token_back_mode": "reuse_dispatch_warps"},
            {"in_kernel_fc2_reduce": True},
        ):
            assert pkg.is_valid({**ok, **extra})
            assert not pkg.is_valid({**ok, **extra, "deterministic": True})
            with pytest.raises(ValueError, match="deterministic"):
                pkg.MegaMoEHopperFp8Config(
                    **_CFG_BASE, **ok, **extra, deterministic=True
                )

    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    def test_is_valid_matches_config_geometry(self, scale_mode):
        pkg = _pkg()
        from flashinfer.moe_ep.kernel_src.sm90.pull_style_cutedsl_megakernel.shim.tuner import (
            PERF_KNOBS,
        )

        # is_valid mirrors the config's geometry validation in both
        # directions over the whole geometry space.
        names = (
            "swap_ab",
            "pingpong",
            "mma_tiler_mnk",
            "cluster_shape_mnk",
            "fp8_accum_mode",
            "tail_split_pairs",
        )
        accepted_count = 0
        for values in itertools.product(*(PERF_KNOBS[n] for n in names)):
            knobs = dict(zip(names, values, strict=True))
            try:
                pkg.MegaMoEHopperFp8Config(
                    **_CFG_BASE, fp8_scale_mode=scale_mode, **knobs
                )
                accepted = True
            except ValueError:
                accepted = False
            assert accepted == pkg.is_valid(knobs, fp8_scale_mode=scale_mode), knobs
            accepted_count += accepted
        assert accepted_count > 0
        assert next(pkg.iter_candidates(fp8_scale_mode=scale_mode), None) is not None

    @pytest.mark.parametrize("scale_mode", ["per_tensor", "blockwise"])
    def test_autotune_candidates(self, scale_mode):
        pkg = _pkg()
        cands = pkg.hopper_fp8_candidates(fp8_scale_mode=scale_mode, max_tokens=2048)
        # The sweep geometries legal for the scale mode (blockwise keeps the
        # N256 / M256 tiles) x {epi_warps, reuse_dispatch_warps}.
        assert len(cands) >= (30 if scale_mode == "per_tensor" else 20)
        cfg = pkg.MegaMoEHopperFp8Config(
            **_CFG_BASE, fp8_scale_mode=scale_mode, mma_tiler_mnk=(64, 256, 128)
        )
        for c in cands:
            pkg.with_knobs(cfg, c)
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

    def test_resolve_skips_entry_the_kernel_rejects(self, tmp_path, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "cache.json"))
        # A blockwise winner recorded before the 1x128 FC2-activation scales:
        # falls back to the heuristic instead of failing at workspace creation.
        stale = dict(
            swap_ab=True,
            pingpong=True,
            mma_tiler_mnk=(128, 16, 128),
            cluster_shape_mnk=(1, 2, 1),
        )
        blockwise = {**self._KEY, "fp8_scale_mode": "blockwise"}
        assert pkg.record_knobs(stale, **blockwise, max_tokens=32) is not None
        with pytest.warns(UserWarning, match="ignoring cached knobs"):
            knobs, source = pkg.resolve_knobs(**blockwise, max_tokens=32)
        assert source == "heuristic"
        assert knobs == pkg.default_knobs(32, fp8_scale_mode="blockwise")
        # The same geometry stays a valid per_tensor entry.
        assert pkg.record_knobs(stale, **self._KEY, max_tokens=32) is not None
        assert pkg.resolve_knobs(**self._KEY, max_tokens=32) == (stale, "cache")
        # An entry without geometry keeps the table's, as the shim merges it.
        partial = {"group_hint": 128}
        assert pkg.record_knobs(partial, **blockwise, max_tokens=64) is not None
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert pkg.resolve_knobs(**blockwise, max_tokens=64) == (partial, "cache")

    def test_disabled_cache(self, monkeypatch):
        pkg = _pkg()
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", "off")
        assert pkg.knob_cache_path() is None
        assert pkg.lookup_knobs(**self._KEY, max_tokens=8) is None

    def test_sm100_entries_never_cross_match(self, tmp_path, monkeypatch):
        from flashinfer.moe_ep.core.tuning import KNOB_CACHE_VERSION

        pkg = _pkg()
        path = tmp_path / "cache.json"
        monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(path))
        # An SM100-shaped entry (no fp8_scale_mode field, nvfp4 dtype).
        path.write_text(
            json.dumps(
                {
                    "version": KNOB_CACHE_VERSION,
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
