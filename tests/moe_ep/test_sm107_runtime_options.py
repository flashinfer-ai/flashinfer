"""Rubin kernel selection, configuration limits, and cache isolation."""

import dataclasses

import pytest
import torch

from flashinfer.moe_ep import BootstrapConfig, FleetParams
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.sm107 import next_cutedsl_megamoe as pkg
from tests.moe_ep.test_sm107_block_scaled_config import _BACKENDS


def _config(**overrides):
    return pkg.Sm107BlockScaledMoeConfig(
        **(
            dict(
                num_total_experts=8,
                max_tokens_per_rank=3,
                num_topk=3,
                hidden=384,
                intermediate=256,
                rank=0,
                world_size=1,
                kernel_variant="genphase",
                cluster_shape_mn=(4, 1),
                fc2_use_bulk=True,
            )
            | overrides
        )
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"max_tokens_per_rank": 1025},
        {"cluster_shape_mn": (2, 1)},
        {"fallback_cluster_shape_mn": (2, 1)},
        {"fc2_use_bulk": False},
        {"mma_tiler_mnk": (256, 64, 128)},
        {"sf_padding_block": 256},
        {"token_in_flag_batch": 2},
        {"token_back_mode": "standalone_warps"},
        {"kernel_variant": "unknown"},
        {"combine_dtype": "nvfp4"},
        {"combine_dtype": "mxfp8"},
    ],
)
def test_genphase_rejects_unsupported_geometry(overrides):
    with pytest.raises(ValueError):
        _config(**overrides)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
def test_genphase_candidates_respect_fixed_geometry(kind):
    cfg = _config(quant_kind=kind)
    assert cfg.padded_tokens_per_rank == 4
    assert _config(max_tokens_per_rank=1024).padded_tokens_per_rank == 1024
    candidates = pkg.sm107_candidates(
        kind, kernel_variant="genphase", allow_in_kernel_fc2_reduce=True
    )
    assert candidates
    assert all(pkg.is_valid_sm107(candidate, cfg) for candidate in candidates)
    assert all(candidate["fc2_use_bulk"] for candidate in candidates)
    assert all(
        candidate["fallback_cluster_shape_mn"] is None for candidate in candidates
    )
    deep_k = {"mma_tiler_mnk": (256, 128, 4 * cfg.instruction_k)}
    with pytest.raises(ValueError, match="GenPhase .* requires tile K"):
        _config(quant_kind=kind, **deep_k)
    assert not pkg.is_valid_sm107(deep_k, cfg)


_MODES = [
    ("inference", "bf16"),
    ("genphase", "bf16"),
    ("inference", "nvfp4"),
    ("inference", "mxfp8"),
]


@pytest.mark.parametrize("config_cls,backend_cls,name", _BACKENDS)
def test_runtime_options_resolve_and_use_distinct_workspaces(
    config_cls, backend_cls, name, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    config = config_cls(256, 3)
    fleet = FleetParams(num_experts=8, max_tokens_per_rank=3, token_hidden_size=384)
    keys = []
    for variant, combine_dtype in _MODES:
        backend = create_mega_kernel(
            dataclasses.replace(
                config,
                kernel_variant=variant,
                combine_dtype=combine_dtype,
                cluster_shape_mn=(4, 1),
                fc2_use_bulk=variant == "genphase",
            )
        )
        backend.bind_ep_bootstrap(BootstrapConfig(rank=0, world_size=1))
        resolved = backend._resolved_config(fleet)
        assert (resolved.kernel_variant, resolved.combine_dtype) == (
            variant,
            combine_dtype,
        )
        keys.append(backend._workspace_pool_key(fleet))
    assert len(set(keys)) == len(_MODES)


def test_runtime_options_cannot_reuse_each_others_cached_knobs(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHINFER_MOE_EP_KNOB_CACHE", str(tmp_path / "knobs.json"))
    key = dict(
        dtype="mxfp8_e4m3",
        world_size=1,
        hidden=384,
        intermediate=256,
        num_experts=8,
        topk=3,
        max_tokens=3,
        device="test",
    )
    for i, (variant, combine_dtype) in enumerate(_MODES):
        assert (
            pkg.lookup_knobs(**key, kernel_variant=variant, combine_dtype=combine_dtype)
            is None
        )
        pkg.record_knobs(
            {"max_sm_count": 4 * (i + 1)},
            **key,
            kernel_variant=variant,
            combine_dtype=combine_dtype,
        )
        for j, (recorded_variant, recorded_combine) in enumerate(_MODES[: i + 1]):
            assert pkg.lookup_knobs(
                **key,
                kernel_variant=recorded_variant,
                combine_dtype=recorded_combine,
            ) == {"max_sm_count": 4 * (j + 1)}


@pytest.mark.parametrize("kernel_variant,combine_dtype", _MODES)
def test_cache_fallback_is_valid_for_each_runtime_option(kernel_variant, combine_dtype):
    knobs = pkg.default_knobs(
        3,
        quant_kind="mxfp8_e4m3",
        kernel_variant=kernel_variant,
        combine_dtype=combine_dtype,
    )
    _config(kernel_variant=kernel_variant, combine_dtype=combine_dtype, **knobs)
