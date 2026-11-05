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


@pytest.mark.parametrize("inline", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("tokens", [128])
def test_mega_v41_cache_layouts(inline, mixed, tokens):
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        dsv41_fp8_quantize_pack_sparse_mla_cache,
        dsv41_fp4_quantize_pack_sparse_mla_cache,
    )
    from tests.attention.sparse_mla_prefill_utils import fused_call

    f = fixture(tokens, 64, "v41_fp8")
    packers = [
        dsv41_fp8_quantize_pack_sparse_mla_cache,
        dsv41_fp4_quantize_pack_sparse_mla_cache
        if mixed
        else dsv41_fp8_quantize_pack_sparse_mla_cache,
    ]
    for i, pack in enumerate(packers):
        f["caches"][i] = pack(
            f["decoded"][i].view(-1, 64, 512).bfloat16(),
            kv_layout="NHD",
            inline_scale=inline,
        )
    # Baseline closures own the original wrapper; use explicit calls here.
    f["wrapper"] = SparseMLASm120Wrapper(
        backend="sparse",
        kv_scale_format="ue8m0_g32_inline" if inline else "ue8m0_g32",
        extra_kv_fp4=mixed,
        compute_precision="fp8",
    )
    f["prologue"]()
    options = dict(
        topk_length=f["lengths"][0],
        attn_sink=f["sink"],
        extra_kv_cache=f["caches"][1],
        extra_indices=f["indices"][1],
        extra_topk_length=f["lengths"][1],
        return_lse=True,
        lse_scale=0.6931471805599453,
    )
    lse = (
        f["wrapper"]
        .run(
            f["rotated"],
            f["caches"][0],
            f["indices"][0],
            f["context"],
            512**-0.5,
            **options,
        )
        .clone()
    )
    f["post"]()
    expected, scales = f["out"].float().clone(), f["scales"].clone()
    actual_lse = fused_call(f)
    torch.testing.assert_close(actual_lse, lse, atol=2e-6, rtol=1e-6)
    assert (f["out"].float() != expected).float().mean().item() < 0.001
    torch.testing.assert_close(f["scales"], scales, atol=0, rtol=0)


@pytest.mark.parametrize("cache_format", ["v41_fp8", "nvfp4"])
@pytest.mark.parametrize("tokens", [128])
def test_mega_graph_kernel_boundaries(cache_format, tokens, tmp_path):
    import re
    from tests.attention.sparse_mla_prefill_utils import fused_call

    f = fixture(tokens, 64, cache_format)
    fused_call(f)
    plan = next(iter(f["wrapper"]._prepared_calls.values())).plan.inspect()
    graph = torch.cuda.CUDAGraph()
    graph.enable_debug_mode()
    with torch.cuda.graph(graph):
        fused_call(f)
    graph.replay()
    torch.cuda.synchronize()
    path = tmp_path / "mega.dot"
    graph.debug_dump(str(path))
    text = path.read_text()
    nodes = re.findall(r'"graph_\d+_node_\d+"\[.*?\];', text, re.DOTALL)
    kernels = [node for node in nodes if 'label="{KERNEL' in node]
    assert plan["variant"] != 0
    assert len(kernels) == 1, text
    assert all("sparse_mla" in node for node in kernels), text
    assert len(nodes) == len(kernels), text
