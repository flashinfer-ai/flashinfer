"""Host-only validation for the BF16 CuTeDSL MegaMoE integration."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from flashinfer.moe_ep.backends.mega.kernel.sm100.bf16_bf16_bf16_cutedsl.config import (
    Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import MegaMoEBf16Config
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.autotune import (
    bf16_candidates,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.tuner import (
    default_knobs,
    is_valid_bf16,
)


def test_bf16_knobs_only_expose_valid_fixed_geometry():
    knobs = default_knobs(256, dtype="bf16")
    assert knobs["mma_tiler_mnk"] == (256, 256, 64)
    assert knobs["cluster_shape_mnk"] == (2, 1, 1)
    assert is_valid_bf16(knobs)
    assert not is_valid_bf16({**knobs, "mma_tiler_mnk": (256, 256, 256)})
    assert not is_valid_bf16(
        {**knobs, "in_kernel_fc2_reduce": True, "token_back_mode": "epi_warps"}
    )
    assert not is_valid_bf16({**knobs, "force_static_sched": False})
    assert not is_valid_bf16({**knobs, "load_balance_mode": "invalid"})
    assert knobs["in_kernel_fc2_reduce"] is False
    assert bf16_candidates() == [knobs]
    assert bf16_candidates(enable_in_kernel_fc2_reduce=True) == [
        knobs,
        {
            **knobs,
            "in_kernel_fc2_reduce": True,
            "token_back_mode": "reuse_dispatch_warps",
        },
    ]


@pytest.mark.parametrize(
    ("hidden", "intermediate", "top_k", "message"),
    ((33, 64, 1, "hidden"), (32, 65, 1, "intermediate"), (32, 64, 33, "topk")),
)
def test_bf16_frontend_rejects_unsupported_shapes(
    hidden: int, intermediate: int, top_k: int, message: str
):
    with pytest.raises(ValueError, match=message):
        MegaMoEBf16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=64,
            num_topk=top_k,
            num_total_experts=1,
            hidden=hidden,
            intermediate=intermediate,
        )


def test_bf16_factory_accepts_session_compatible_pinned_knobs(monkeypatch):
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16, "sym_zeros", fake_zeros)
    knobs = default_knobs(8, dtype="bf16")
    knobs["token_back_mode"] = "reuse_dispatch_warps"
    buf = bf16.get_symm_buffer_for_bf16_mega_moe(
        4,
        8,
        2,
        128,
        128,
        0,
        1,
        knobs=knobs,
    )
    try:
        assert buf._frontend.config.token_back_mode == "reuse_dispatch_warps"
        assert buf._frontend.config.in_kernel_fc2_reduce is False
    finally:
        buf.destroy()


def test_bf16_factory_rejects_unpermitted_pinned_ikr(monkeypatch):
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16

    monkeypatch.setattr(
        bf16, "sym_zeros", lambda shape, dtype: torch.zeros(shape, dtype=dtype)
    )
    with pytest.raises(ValueError, match="unsupported BF16 MegaMoE knobs"):
        bf16.get_symm_buffer_for_bf16_mega_moe(
            4,
            8,
            2,
            128,
            128,
            0,
            1,
            knobs=default_knobs(8, dtype="bf16", enable_in_kernel_fc2_reduce=True),
        )


def test_bf16_factory_resolves_ikr_from_knobs_when_permitted(monkeypatch):
    """The permission is a ceiling: either knob value is now servable."""
    import torch

    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import bf16

    def fake_zeros(shape, dtype):
        tensor = torch.zeros(shape, dtype=dtype)
        tensor._mega_plain_alloc = True
        return tensor

    monkeypatch.setattr(bf16, "sym_zeros", fake_zeros)
    for ikr in (False, True):
        buf = bf16.get_symm_buffer_for_bf16_mega_moe(
            4,
            8,
            2,
            128,
            128,
            0,
            1,
            enable_in_kernel_fc2_reduce=True,
            knobs=default_knobs(8, dtype="bf16", enable_in_kernel_fc2_reduce=ikr),
        )
        try:
            assert buf._frontend.config.in_kernel_fc2_reduce is ikr
            # Both destinations exist either way, so the knob can flip later.
            assert buf.combine_output.shape == (8, 2, 128)
            assert buf.reduced_output.shape == (8, 1, 128)
            assert buf.kernel_combine_output.shape[1] == (1 if ikr else 2)
        finally:
            buf.destroy()


def test_bf16_frontend_rejects_unpermitted_ikr_knobs():
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import (
        MegaMoEBf16Frontend,
    )

    base = dict(
        rank=0,
        world_size=1,
        num_tokens_per_rank=8,
        num_topk=2,
        num_total_experts=4,
        hidden=128,
        intermediate=128,
    )
    ikr_knobs = {
        "in_kernel_fc2_reduce": True,
        "token_back_mode": "reuse_dispatch_warps",
    }
    with pytest.raises(ValueError, match="unsupported BF16 MegaMoE knobs"):
        MegaMoEBf16Frontend(MegaMoEBf16Config(**base)).apply_knobs(ikr_knobs)

    permitted = MegaMoEBf16Frontend(
        MegaMoEBf16Config(**base, enable_in_kernel_fc2_reduce=True)
    )
    permitted.apply_knobs(ikr_knobs)
    assert permitted.config.in_kernel_fc2_reduce is True


def test_bf16_autotune_filters_unpermitted_ikr_candidates(monkeypatch):
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import autotune
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import (
        MegaMoEBf16Frontend,
    )

    frontend = MegaMoEBf16Frontend(
        MegaMoEBf16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=8,
            num_topk=2,
            num_total_experts=4,
            hidden=128,
            intermediate=128,
        )
    )
    valid = default_knobs(8, dtype="bf16")
    ikr = default_knobs(8, dtype="bf16", enable_in_kernel_fc2_reduce=True)
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_kwargs: candidates,
    )
    buffer = SimpleNamespace(_frontend=frontend)
    with pytest.warns(
        RuntimeWarning, match="in_kernel_fc2_reduce=True is not permitted"
    ):
        assert autotune.autotune_bf16_mega_moe(
            None, None, None, buffer, candidates=[ikr, valid]
        ) == [valid]
    with pytest.raises(ValueError, match="no valid BF16") as excinfo:
        autotune.autotune_bf16_mega_moe(None, None, None, buffer, candidates=[ikr])
    assert "in_kernel_fc2_reduce=True is not permitted" in str(excinfo.value)


def test_bf16_autotune_sweeps_the_ikr_axis_when_permitted(monkeypatch):
    """A permitted session times both ikr modes; both must be runnable as-is."""
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim import autotune
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import (
        MegaMoEBf16Frontend,
    )

    frontend = MegaMoEBf16Frontend(
        MegaMoEBf16Config(
            rank=0,
            world_size=1,
            num_tokens_per_rank=8,
            num_topk=2,
            num_total_experts=4,
            hidden=128,
            intermediate=128,
            enable_in_kernel_fc2_reduce=True,
        )
    )
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_kwargs: candidates,
    )
    candidates = autotune.autotune_bf16_mega_moe(
        None, None, None, SimpleNamespace(_frontend=frontend)
    )
    assert candidates == [
        default_knobs(8, dtype="bf16"),
        default_knobs(8, dtype="bf16", enable_in_kernel_fc2_reduce=True),
    ]


def test_bf16_backend_defaults_to_scale_free_contract():
    config = Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(intermediate_size=64, top_k=1)
    assert config.kernel_name == "sm100_bf16_bf16_bf16_cutedsl"
    assert config.knobs is None


def test_bf16_backend_accepts_collective_autotune():
    assert (
        Sm100_Bf16_Bf16_Bf16_Cutedsl_MegaMoeConfig(
            intermediate_size=64, top_k=1, knobs="auto"
        ).knobs
        == "auto"
    )
