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
    assert bf16_candidates() == [{**knobs, "in_kernel_fc2_reduce": False}]
    assert bf16_candidates(
        in_kernel_fc2_reduce=True, token_back_mode="reuse_dispatch_warps"
    ) == [
        {
            **knobs,
            "in_kernel_fc2_reduce": True,
            "token_back_mode": "reuse_dispatch_warps",
        }
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
        token_back_mode="epi_warps",
        knobs=knobs,
    )
    try:
        assert buf._frontend.config.token_back_mode == "reuse_dispatch_warps"
        assert buf._frontend.config.in_kernel_fc2_reduce is False
    finally:
        buf.destroy()


def test_bf16_factory_rejects_pinned_ikr_mismatch(monkeypatch):
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
            knobs={**default_knobs(8, dtype="bf16"), "in_kernel_fc2_reduce": True},
        )


def test_bf16_frontend_rejects_ikr_changing_knobs():
    frontend = MegaMoEBf16Config(
        rank=0,
        world_size=1,
        num_tokens_per_rank=8,
        num_topk=2,
        num_total_experts=4,
        hidden=128,
        intermediate=128,
    )
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe.shim.bf16 import (
        MegaMoEBf16Frontend,
    )

    with pytest.raises(ValueError, match="unsupported BF16 MegaMoE knobs"):
        MegaMoEBf16Frontend(frontend).apply_knobs({"in_kernel_fc2_reduce": True})


def test_bf16_autotune_filters_ikr_changing_candidates(monkeypatch):
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
    monkeypatch.setattr(
        autotune,
        "autotune_knobs",
        lambda _frontend, _launch, candidates, **_kwargs: candidates,
    )
    buffer = SimpleNamespace(_frontend=frontend)
    assert autotune.autotune_bf16_mega_moe(
        None,
        None,
        None,
        buffer,
        candidates=[{**valid, "in_kernel_fc2_reduce": True}, valid],
    ) == [valid]
    with pytest.raises(ValueError, match="no valid BF16"):
        autotune.autotune_bf16_mega_moe(
            None,
            None,
            None,
            buffer,
            candidates=[{**valid, "in_kernel_fc2_reduce": True}],
        )


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
