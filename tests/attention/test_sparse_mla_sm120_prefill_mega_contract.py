"""Single-cache, empty-row and input-contract coverage for fused prefill."""

import pytest
import torch
from tests.attention.sparse_mla_prefill_utils import fixture


@pytest.mark.parametrize("cache_format", ["v41_fp8", "nvfp4"])
@pytest.mark.parametrize("tokens,heads", [(128, 16), (128, 64)])
@pytest.mark.parametrize("empty", [False, True])
def test_fused_single_cache_and_empty_prefill(cache_format, tokens, heads, empty):
    from tests.attention.sparse_mla_prefill_utils import epilogue

    f = fixture(tokens, heads, cache_format)
    if empty:
        f["lengths"][0].zero_()
    else:
        f["indices"][0][:, ::3] = -1
    f["prologue"]()
    opts = dict(topk_length=f["lengths"][0], return_lse=True)
    lse = (
        f["wrapper"]
        .run(
            f["rotated"],
            f["caches"][0],
            f["indices"][0],
            f["context"],
            512**-0.5,
            **opts,
        )
        .clone()
    )
    epilogue(f["context"], f["positions"], f["cs"], f["out"], f["scales"])
    expected, scales = f["out"].float().clone(), f["scales"].clone()
    actual_lse = f["wrapper"].run_mega(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["out"],
        512**-0.5,
        positions=f["positions"],
        cos_sin=f["cs"],
        output_scales=f["scales"],
        **opts,
    )
    torch.testing.assert_close(actual_lse, lse, atol=0, rtol=0)
    torch.testing.assert_close(f["out"].float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(f["scales"], scales, atol=0, rtol=0)
    if empty:
        assert torch.isneginf(actual_lse).all()
        assert not torch.count_nonzero(f["out"].float())


@pytest.mark.parametrize(
    "bad",
    ["positions_dtype", "positions_shape", "cs_width", "sf_shape", "output_dtype"],
)
def test_mega_binding_guards(bad):
    from tests.attention.sparse_mla_prefill_utils import fused_call

    f = fixture(65, 16, "v41_fp8")
    if bad == "positions_dtype":
        f["positions"] = f["positions"].long()
    elif bad == "positions_shape":
        f["positions"] = f["positions"].repeat(2)
    elif bad == "cs_width":
        f["cs"] = f["cs"][:, :32].contiguous()
    elif bad == "sf_shape":
        f["scales"] = f["scales"][..., :8].contiguous()
    else:
        f["out"] = f["out"].bfloat16()
    with pytest.raises((ValueError, RuntimeError)):
        fused_call(f)
