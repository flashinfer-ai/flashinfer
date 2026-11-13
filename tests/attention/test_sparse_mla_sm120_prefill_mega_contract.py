"""Single-cache, empty-row and input-contract coverage for fused prefill."""

import pytest
import torch
from tests.attention.sparse_mla_prefill_utils import (
    fixture,
    poison_mega_destinations,
    assert_mega_outputs,
    fused_call,
    numerical_metrics,
)


@pytest.mark.parametrize("cache_format", ["fp8", "v41_fp8"])
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
    actual_out = torch.empty_like(f["out"])
    actual_scales = torch.empty_like(scales)
    actual_lse = torch.empty_like(lse)
    poison_mega_destinations(actual_out, actual_scales, actual_lse, scales)
    f["wrapper"].run_mega(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        actual_out,
        512**-0.5,
        positions=f["positions"],
        cos_sin=f["cs"],
        output_scales=actual_scales,
        out_lse=actual_lse,
        **opts,
    )
    assert_mega_outputs(
        (actual_out, actual_scales, actual_lse), (expected, scales, lse)
    )
    if empty:
        assert torch.isneginf(actual_lse).all()
        assert not torch.count_nonzero(actual_out.float())


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
    f["out"] = torch.empty_like(f["out"])
    f["scales"] = torch.empty_like(scales)
    actual_lse = torch.empty_like(lse)
    fused_call(f, out_lse=actual_lse)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_call(f, out_lse=actual_lse)
    for _ in range(2):
        poison_mega_destinations(f["out"], f["scales"], actual_lse, scales)
        graph.replay()
        assert_mega_outputs(
            (f["out"], f["scales"], actual_lse), (expected, scales, lse)
        )


@pytest.mark.parametrize("cache_format", ["v41_fp8"])
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


@pytest.mark.parametrize(
    "cache_format,inline", [("fp8", False), ("v41_fp8", False), ("v41_fp8", True)]
)
@pytest.mark.parametrize("tokens", [1, 16, 64])
@pytest.mark.parametrize("topk", [128, 4096])
def test_default_mega_decode_merge(cache_format, inline, tokens, topk):
    """The public decode route must write all outputs in either merge branch."""
    f = fixture(
        tokens,
        32,
        cache_format,
        inline=inline,
        topk=topk,
        extra_topk=0,
        ragged=True,
        with_sink=False,
    )
    lengths = torch.tensor([topk, 0, 1, 63, 64, 65, topk - 1], device="cuda")
    f["lengths"][0].copy_(lengths[torch.arange(tokens, device="cuda") % len(lengths)])
    _check_default_mega(f, variant=0, large_merge=topk > 2048)


@pytest.mark.parametrize("heads", [16, 32, 64])
@pytest.mark.parametrize("route", ["single", "dual", "full_tile"])
def test_dsv4_mega_prefill_routes(heads, route):
    f = fixture(
        65,
        heads,
        "fp8",
        extra_topk=0 if route == "single" else 128,
        ragged=True,
        with_sink=False,
    )
    if route == "full_tile":
        # Omitting lengths selects the full-tile implementation. Negative IDs
        # still mask individual candidates, including an entirely empty row.
        for i, ids in enumerate(f["indices"]):
            if ids is not None:
                ids[0].fill_(-1)
                f["lengths"][i] = None
    _check_default_mega(
        f,
        variant=2 if route == "single" else 3,
        implementation="fulltile" if route == "full_tile" else "mg",
    )


def _check_default_mega(f, *, variant, implementation=None, large_merge=None):
    expected = tuple(x.clone() for x in f["baseline"]())
    f["out"] = torch.empty_like(f["out"])
    f["scales"] = torch.empty_like(f["scales"])
    lse = torch.empty_like(expected[2])
    fused_call(f, out_lse=lse)
    plan = next(iter(f["wrapper"]._prepared_calls.values())).plan.inspect()
    assert plan["variant"] == variant, plan
    if implementation is not None:
        assert plan["implementation"] == implementation, plan
    if large_merge is not None:
        assert (plan["scratch_split_stride"] > 32) == large_merge, plan
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_call(f, out_lse=lse)
    for _ in range(2):
        poison_mega_destinations(f["out"], f["scales"], lse, expected[1])
        graph.replay()
        assert_mega_outputs((f["out"], f["scales"], lse), expected)
    rows = torch.tensor([0, f["q"].shape[0] - 1], device="cuda").unique()
    metrics = numerical_metrics(f, lse, rows)
    assert metrics["finite_output"], metrics
    assert metrics["relative_l2"] < 0.07, metrics
    assert metrics["lse_max_abs"] < 0.06, metrics
