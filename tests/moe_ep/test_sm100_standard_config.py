"""Standard Blackwell defaults and caller overrides without GPU allocation."""

import pytest
import torch

from flashinfer import moe_ep
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.core.validation.common import MoEEpConfigError


def make_config(capacity, *, kind="nvfp4", knobs=None, **options):
    if options.get("finalize_mode") is moe_ep.FinalizeMode.PERSISTENT:
        options.setdefault("num_valid_tokens_tensor", torch.tensor([capacity]))
    cls = (
        moe_ep.Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig
        if kind == "nvfp4"
        else moe_ep.Sm100_Mxfp8_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
    )
    kwargs = {} if kind == "nvfp4" else {"kind": kind}
    selector = "nvfp4" if kind == "nvfp4" else "mxfp8"
    backend = create_mega_kernel(
        cls(
            intermediate_size=256,
            top_k=2,
            kernel_name=f"sm100_{selector}_{selector}_bf16_cutedsl_next",
            knobs=knobs,
            **kwargs,
            **options,
        )
    )
    backend.bind_ep_bootstrap(
        moe_ep.BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False)
    )
    return backend._workspace_config(moe_ep.FleetParams(8, capacity, 512))


@pytest.mark.parametrize("capacity", [8, 511, 512, 1023, 1024, 2048, 8192])
@pytest.mark.parametrize("kind", ["nvfp4", "mxfp8_e4m3", "mxfp8_e5m2"])
@pytest.mark.parametrize("knobs", [None, "auto"])
def test_standard_defaults_construct_across_capacity_boundaries(capacity, kind, knobs):
    cfg = make_config(capacity, kind=kind, knobs=knobs)
    assert cfg.max_tokens_per_rank == capacity
    assert cfg.quant_kind == kind
    if kind == "nvfp4" and capacity >= 1024:
        assert cfg.resolved_mma_tiler_mnk == (256, 256, 128)


@pytest.mark.parametrize("capacity", [512, 1024, 2048, 8192])
@pytest.mark.parametrize("native", [False, True])
def test_standard_explicit_tile_takes_precedence(capacity, native):
    knobs = {"mma_tiler_mnk": (256, 128, 128)}
    if native:
        knobs["work_id_mode"] = "grid_stride"
    cfg = make_config(capacity, knobs=knobs)
    assert cfg.resolved_mma_tiler_mnk == (256, 128, 128)
    if native:
        assert cfg.work_id_mode == "grid_stride"


@pytest.mark.parametrize("native", [False, True])
def test_standard_explicit_invalid_tile_is_rejected(native):
    knobs = {"mma_tiler_mnk": (256, 256, 256)}
    if native:
        knobs["work_id_mode"] = "atomic_counter"
    with pytest.raises(ValueError):
        make_config(2048, knobs=knobs)


def test_native_in_kernel_reduction_resolves_persistent_selection_before_validation():
    cfg = make_config(
        2048,
        knobs={"reduce_topk_in_kernel": True},
        enable_in_kernel_fc2_reduce=True,
        finalize_mode=moe_ep.FinalizeMode.PERSISTENT,
    )
    assert cfg.reduce_topk_in_kernel
    assert not cfg.use_persistent_finalize_kernel


@pytest.mark.parametrize("mode", ["DEFAULT", "PERSISTENT", "NONE"])
def test_native_overrides_preserve_finalizer_contract(mode):
    cfg = make_config(
        8192,
        knobs={"mma_tiler_mnk": (256, 128, 128), "work_id_mode": "grid_stride"},
        finalize_mode=moe_ep.FinalizeMode[mode],
    )
    assert cfg.use_persistent_finalize_kernel == (mode == "PERSISTENT")
    assert cfg.use_custom_finalize == (mode == "NONE")


def test_native_in_kernel_reduction_requires_permission():
    with pytest.raises(MoEEpConfigError, match="enabled in the config"):
        make_config(2048, knobs={"reduce_topk_in_kernel": True})


def test_native_knobs_reject_unknown_keys():
    with pytest.raises(MoEEpConfigError, match="unknown Blackwell standard"):
        make_config(2048, knobs={"work_id_mode": "grid_stride", "typo": 1})


def test_native_custom_finalizer_rejects_in_kernel_reduction():
    with pytest.raises(ValueError, match="finalize_mode=NONE"):
        make_config(
            2048,
            knobs={"reduce_topk_in_kernel": True},
            enable_in_kernel_fc2_reduce=True,
            finalize_mode=moe_ep.FinalizeMode.NONE,
        )


@pytest.mark.parametrize("capacity", [32, 1024, 8192])
def test_native_nvfp4_n64_profile(capacity):
    cfg = make_config(
        capacity,
        knobs={
            "mma_tiler_mnk": (256, 64, 128),
            "fc2_use_bulk": True,
            "fc2_tma_stages": 1,
        },
    )
    assert cfg.resolved_mma_tiler_mnk == (256, 64, 128)
    assert cfg.token_padding_block == 64
