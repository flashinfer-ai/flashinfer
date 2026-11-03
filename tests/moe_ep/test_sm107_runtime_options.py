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
        {"mma_tiler_mnk": (128, 64, 128)},
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


@pytest.mark.parametrize(
    "kind, cluster, supported",
    [
        ("mxfp8_e4m3", (2, 1), True),
        ("mxfp8_e4m3", (4, 1), True),
        ("mxfp4_mxfp8", (4, 1), True),
        ("mxfp4_mxfp8", (2, 1), False),
        ("mxfp8_e5m2", (4, 1), False),
        ("nvfp4", (4, 1), False),
    ],
)
def test_genphase_n64_dtype_and_cluster_admission(kind, cluster, supported):
    knobs = dict(
        quant_kind=kind,
        mma_tiler_mnk=(256, 64, 128),
        cluster_shape_mn=cluster,
        fc2_tma_stages=1,
        epi_flag_batches=(1, 4),
    )
    if supported:
        cfg = _config(**knobs)
        assert cfg.mma_tiler_mnk == (256, 64, 128)
    else:
        with pytest.raises(ValueError):
            _config(**knobs)


@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2", "mxfp4_mxfp8"])
def test_genphase_tuner_exposes_only_admitted_n64_candidates(kind):
    candidates = pkg.sm107_candidates(kind, kernel_variant="genphase")
    n64 = [candidate for candidate in candidates if candidate["mma_tiler_mnk"][1] == 64]
    assert len(n64) == (3 if kind in ("mxfp8_e4m3", "mxfp4_mxfp8") else 0)
    for candidate in n64:
        assert pkg.is_valid_sm107(candidate, _config(quant_kind=kind))
        assert candidate["fc2_tma_stages"] == 1
        assert candidate["reduce_topk_in_kernel"] is False
    assert {candidate["mma_tiler_mnk"][1] for candidate in candidates} >= {128, 256}


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize(
    "kind, cluster",
    [("mxfp8_e4m3", (2, 1)), ("mxfp8_e4m3", (4, 1)), ("mxfp4_mxfp8", (4, 1))],
)
def test_genphase_n64_multirank_schedule_admission(world, kind, cluster):
    cfg = _config(
        world_size=world,
        quant_kind=kind,
        mma_tiler_mnk=(256, 64, 128),
        cluster_shape_mn=cluster,
        fc2_tma_stages=1,
        epi_flag_batches=(1, 4),
        work_id_mode="atomic_counter",
    )
    assert pkg.is_valid_sm107({"schedule_policy": ("grouped", None)}, cfg)
    for hint in (None, 1):
        phase = {"schedule_policy": ("phase_interleave", hint)}
        if world == 1:
            assert pkg.is_valid_sm107(phase, cfg)
            assert (
                dataclasses.replace(cfg, **phase).schedule_policy
                == phase["schedule_policy"]
            )
        else:
            assert not pkg.is_valid_sm107(phase, cfg)
            with pytest.raises(ValueError, match="N64 .* requires grouped scheduling"):
                dataclasses.replace(cfg, **phase)


@pytest.mark.parametrize("world", [2, 4, 8])
def test_genphase_n128_multirank_phase_admission_is_unchanged(world):
    cfg = _config(world_size=world, quant_kind="mxfp8_e4m3")
    phase = {
        "schedule_policy": ("phase_interleave", None),
        "work_id_mode": "atomic_counter",
    }
    assert pkg.is_valid_sm107(phase, cfg)
    assert dataclasses.replace(cfg, **phase).resolved_mma_tiler_mnk[1] == 128


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
