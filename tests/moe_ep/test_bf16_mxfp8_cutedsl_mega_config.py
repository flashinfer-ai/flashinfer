"""Host-only validation for mixed MXFP8-weight/BF16-activation MegaMoE."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashinfer.moe_ep import Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig
from flashinfer.moe_ep.backends.mega.kernel.sm100.common.bf16_config import (
    Sm100_Bf16_Cutedsl_MegaMoeConfigBase,
)
from flashinfer.moe_ep.core.kernel.registry import create_mega_kernel
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import MegaMoEBf16Mxfp8Config
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.autotune import (
    bf16_mxfp8_candidates,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16_mxfp8 import (
    MegaMoEBf16Mxfp8Frontend,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.tuner import (
    default_knobs,
    is_valid_bf16_mxfp8,
)

_LEGAL_IMPLS = {
    ((256, 128, 128), "tmem", False, 128),
    ((256, 256, 128), "smem", False, 128),
    ((256, 256, 128), "tmem", True, 64),
}


def _config(**kwargs):
    return MegaMoEBf16Mxfp8Config(
        rank=0,
        world_size=4,
        num_tokens_per_rank=256,
        num_topk=4,
        num_total_experts=32,
        hidden=1024,
        intermediate=1024,
        **kwargs,
    )


def _impl_key(knobs):
    return (
        knobs["mma_tiler_mnk"],
        knobs["transform_buffer"],
        knobs["accumulator_overlap"],
        knobs["transform_k_tile"],
    )


def test_mixed_knobs_and_candidates():
    knobs = default_knobs(256, dtype="bf16_mxfp8")
    assert is_valid_bf16_mxfp8(knobs)
    assert is_valid_bf16_mxfp8(
        {
            **knobs,
            "mma_tiler_mnk": (256, 256, 128),
            "transform_buffer": "smem",
            "accumulator_overlap": False,
            "transform_k_tile": 128,
        }
    )
    assert not is_valid_bf16_mxfp8({**knobs, "mma_tiler_mnk": (256, 256, 128)})
    assert not is_valid_bf16_mxfp8({**knobs, "token_back_mode": "standalone_warps"})
    assert not is_valid_bf16_mxfp8({**knobs, "clc_bundle_size": 1})
    # The mixed kernel carries ikr on either token-back mode (unlike pure
    # MXFP8); run_mega_tests.sh case M05 validates the dispatch-warp pair.
    for token_back in ("epi_warps", "reuse_dispatch_warps"):
        assert is_valid_bf16_mxfp8(
            {
                **knobs,
                "in_kernel_fc2_reduce": True,
                "token_back_mode": token_back,
            }
        )

    candidates = bf16_mxfp8_candidates()
    assert len(candidates) == 12
    assert all(is_valid_bf16_mxfp8(c) for c in candidates)
    assert {_impl_key(c) for c in candidates} == _LEGAL_IMPLS
    assert {c["token_back_mode"] for c in candidates} == {
        "epi_warps",
        "reuse_dispatch_warps",
    }

    # ikr must not shrink the token-back axis: a session that asked for
    # dispatch-warp token-back still gets it timed.
    ikr = bf16_mxfp8_candidates(in_kernel_fc2_reduce=True)
    assert len(ikr) == 12
    assert all(c["in_kernel_fc2_reduce"] for c in ikr)
    assert {c["token_back_mode"] for c in ikr} == {
        "epi_warps",
        "reuse_dispatch_warps",
    }

    frontend = MegaMoEBf16Mxfp8Frontend(_config())
    frontend.apply_knobs(knobs)
    assert frontend.config == _config()


@pytest.mark.parametrize(
    ("mma_tiler_mnk", "transform_buffer", "accumulator_overlap", "transform_k_tile"),
    sorted(_LEGAL_IMPLS),
)
def test_mixed_config_accepts_supported_implementation(
    mma_tiler_mnk, transform_buffer, accumulator_overlap, transform_k_tile
):
    config = _config(
        mma_tiler_mnk=mma_tiler_mnk,
        transform_buffer=transform_buffer,
        accumulator_overlap=accumulator_overlap,
        transform_k_tile=transform_k_tile,
    )
    assert config.num_experts_per_rank == 8


def test_mixed_config_rejects_unsupported_implementation_and_token_back():
    with pytest.raises(ValueError, match="implementation tuple"):
        _config(mma_tiler_mnk=(256, 256, 128))
    with pytest.raises(ValueError, match="standalone"):
        _config(token_back_mode="standalone_warps")  # type: ignore[arg-type]


def test_mixed_frontend_rejects_unsupported_knobs():
    frontend = MegaMoEBf16Mxfp8Frontend(_config())
    with pytest.raises(ValueError, match="unsupported mixed MegaMoE knobs"):
        frontend.apply_knobs({"mma_tiler_mnk": (256, 256, 128)})


def test_mixed_factory_accepts_session_compatible_pinned_knobs(monkeypatch):
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16_mxfp8

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16_mxfp8, "sym_zeros", fake_zeros)
    knobs = default_knobs(8, dtype="bf16_mxfp8")
    knobs["token_back_mode"] = "reuse_dispatch_warps"
    with pytest.warns(RuntimeWarning, match="token_back_mode: 'epi_warps'"):
        buf = bf16_mxfp8.get_symm_buffer_for_bf16_mxfp8_mega_moe(
            4,
            8,
            2,
            128,
            128,
            0,
            1,
            token_back_mode="epi_warps",
            knobs=knobs,
        )
    try:
        assert buf._frontend.config.token_back_mode == "reuse_dispatch_warps"
        assert buf._frontend.config.in_kernel_fc2_reduce is False
    finally:
        buf.destroy()


def test_mixed_factory_accepts_knobs_on_an_ikr_session(monkeypatch):
    """An ikr + dispatch-warp session must still accept pinned perf knobs."""
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16_mxfp8

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16_mxfp8, "sym_zeros", fake_zeros)
    buf = bf16_mxfp8.get_symm_buffer_for_bf16_mxfp8_mega_moe(
        4,
        8,
        2,
        128,
        128,
        0,
        1,
        in_kernel_fc2_reduce=True,
        token_back_mode="reuse_dispatch_warps",
        knobs={"flag_batch": 4},
    )
    try:
        config = buf._frontend.config
        assert config.in_kernel_fc2_reduce is True
        assert config.token_back_mode == "reuse_dispatch_warps"
        assert config.flag_batch == 4
    finally:
        buf.destroy()


def test_mixed_factory_rejects_pinned_ikr_mismatch(monkeypatch):
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16_mxfp8

    monkeypatch.setattr(
        bf16_mxfp8, "sym_zeros", lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    )
    with pytest.raises(ValueError, match="unsupported mixed MegaMoE knobs"):
        bf16_mxfp8.get_symm_buffer_for_bf16_mxfp8_mega_moe(
            4,
            8,
            2,
            128,
            128,
            0,
            1,
            knobs={
                **default_knobs(8, dtype="bf16_mxfp8"),
                "in_kernel_fc2_reduce": True,
            },
        )


def test_mixed_frontend_rejects_ikr_changing_knobs():
    frontend = MegaMoEBf16Mxfp8Frontend(_config())
    with pytest.raises(ValueError, match="unsupported mixed MegaMoE knobs"):
        frontend.apply_knobs({"in_kernel_fc2_reduce": True})


def test_mixed_autotune_filters_ikr_changing_candidates(monkeypatch):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import autotune

    frontend = MegaMoEBf16Mxfp8Frontend(_config())
    valid = default_knobs(8, dtype="bf16_mxfp8")
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_kwargs: candidates,
    )
    buffer = SimpleNamespace(_frontend=frontend)
    with pytest.warns(
        RuntimeWarning, match="in_kernel_fc2_reduce=True is caller-owned"
    ):
        assert autotune.autotune_bf16_mxfp8_mega_moe(
            None,
            None,
            None,
            buffer,
            candidates=[{**valid, "in_kernel_fc2_reduce": True}, valid],
        ) == [valid]
    with pytest.raises(ValueError, match="no valid mixed") as excinfo:
        autotune.autotune_bf16_mxfp8_mega_moe(
            None,
            None,
            None,
            buffer,
            candidates=[{**valid, "in_kernel_fc2_reduce": True}],
        )
    assert "in_kernel_fc2_reduce=True is caller-owned" in str(excinfo.value)


def test_mixed_backend_is_registered():
    backend = create_mega_kernel(
        Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=128,
            top_k=2,
        )
    )
    assert backend.kernel_name() == "sm100_bf16_mxfp8_bf16_cutedsl"


def test_mixed_config_inherits_bf16_options():
    config = Sm100_Bf16_Mxfp8_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=128,
        top_k=2,
        gate_up_clamp=1.5,
        in_kernel_fc2_reduce=True,
    )
    assert isinstance(config, Sm100_Bf16_Cutedsl_MegaMoeConfigBase)
    assert config.gate_up_clamp == 1.5
    assert config.in_kernel_fc2_reduce
