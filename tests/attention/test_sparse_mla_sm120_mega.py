"""Independent math, exact fusion equivalence, and graph replay contracts."""

import pytest
import torch
from tests.attention.mega_mla_utils import (
    fixture,
    reference,
    rope_reference,
    unpack_output,
)


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "tokens,heads,ragged", [(1, 64, False), (4, 16, True), (128, 64, False)]
)
def test_unfused_mega_contract(cache_format, tokens, heads, ragged):
    f = fixture(tokens, heads, cache_format, ragged=ragged)
    out, sf, lse = f["baseline"]()
    expected, ref_lse = reference(f)
    torch.testing.assert_close(
        f["rotated"],
        rope_reference(f["q"], f["positions"], f["cs"]).bfloat16(),
        atol=0.016,
        rtol=0.008,
    )
    actual = unpack_output(out, sf)
    # Full precision reference includes cache quantization, but not query/P quantization.
    # Track quantized-compute error separately from the stricter fused/unfused gate.
    relative_l2 = (
        (
            (actual.double() - expected).square().sum()
            / expected.square().sum().clamp_min(1.0e-30)
        )
        .sqrt()
        .item()
    )
    assert relative_l2 < (0.20 if cache_format == "nvfp4" else 0.07), relative_l2
    torch.testing.assert_close(
        lse.double(),
        ref_lse,
        atol=0.12 if cache_format == "nvfp4" else 0.025,
        rtol=0.025,
    )
    if ragged:
        assert torch.count_nonzero(actual[0]) == 0
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
def test_unfused_graph_reads_current_inputs(cache_format):
    f = fixture(4, 16, cache_format)
    f["baseline"]()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        f["baseline"]()
    old = unpack_output(f["out"], f["scales"]).clone()
    f["positions"].add_(13)
    f["q"].mul_(0.75)
    f["indices"][1].add_(1).remainder_(4096)
    f["lengths"][0][0] = 0
    f["lengths"][1][0] = 0
    graph.replay()
    replay = unpack_output(f["out"], f["scales"]).clone()
    f["baseline"]()
    torch.testing.assert_close(
        replay, unpack_output(f["out"], f["scales"]), atol=0, rtol=0
    )
    assert not torch.equal(old, replay)


@pytest.mark.parametrize("magnitude", [0.0, 1.0e-8, 1.0, 128.0])
def test_unfused_epilogue_quantization_contract(magnitude):
    from tests.attention.mega_mla_utils import epilogue

    f = fixture(2, 16)
    f["positions"][0] = 0
    f["positions"][1] = 100000
    f["context"].normal_().mul_(magnitude)
    epilogue(f["context"], f["positions"], f["cs"], f["out"], f["scales"])
    # Independent float32 inverse rotation and E4M3 per-32 scaling.
    x = f["context"].float()
    y = x.clone()
    c = f["cs"][f["positions"].long(), :32][:, None]
    s = f["cs"][f["positions"].long(), 32:][:, None]
    y[..., 448::2] = x[..., 448::2] * c + x[..., 449::2] * s
    y[..., 449::2] = x[..., 449::2] * c - x[..., 448::2] * s
    blocks = y.reshape(2, 16, 16, 32)
    sf = torch.exp2(
        torch.ceil(torch.log2(blocks.abs().amax(-1).clamp_min(1.0e-4) / 448.0))
    )
    expected = (blocks / sf[..., None]).to(torch.float8_e4m3fn).reshape_as(f["out"])
    # Signed zero encodings are equivalent; all nonzero FP8 values must match exactly.
    torch.testing.assert_close(f["out"].float(), expected.float(), atol=0, rtol=0)
    sf_bytes = ((sf.view(torch.int32) >> 23) & 255).byte()
    torch.testing.assert_close(f["scales"], sf_bytes, atol=0, rtol=0)


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize(
    "tokens,heads,ragged",
    [
        (1, 64, False),
        (4, 16, True),
        (64, 64, False),
        (128, 64, False),
        (1024, 64, False),
    ],
)
def test_fused_matches_unfused(cache_format, tokens, heads, ragged):
    from tests.attention.mega_mla_utils import fused_call

    f = fixture(tokens, heads, cache_format, ragged=ragged)
    f["positions"][0] = 100000
    out, scales, lse = f["baseline"]()
    expected, expected_scales, expected_lse = (
        out.float().clone(),
        scales.clone(),
        lse.clone(),
    )
    # Fused path must never write a global BF16 attention result.
    f["context"].fill_(123)
    original_q = f["q"].clone()
    actual_lse = fused_call(f)
    torch.testing.assert_close(actual_lse, expected_lse, atol=0, rtol=0)
    torch.testing.assert_close(f["scales"], expected_scales, atol=0, rtol=0)
    torch.testing.assert_close(f["out"].float(), expected, atol=0, rtol=0)
    assert (f["context"] == 123).all()
    torch.testing.assert_close(f["q"], original_q, atol=0, rtol=0)


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
def test_fused_graph_reads_current_inputs(cache_format):
    from tests.attention.mega_mla_utils import fused_call

    f = fixture(4, 16, cache_format)
    fused_call(f)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fused_call(f)
    old = f["out"].float().clone()
    f["positions"].zero_()
    f["q"].mul_(0.75)
    f["indices"][1].add_(1).remainder_(4096)
    for lengths in f["lengths"]:
        lengths[0] = 0
    graph.replay()
    actual, scales = f["out"].float().clone(), f["scales"].clone()
    f["baseline"]()
    torch.testing.assert_close(actual, f["out"].float(), atol=0, rtol=0)
    torch.testing.assert_close(scales, f["scales"], atol=0, rtol=0)
    assert not torch.equal(old, actual)


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize("tokens,heads", [(1, 16), (128, 16), (128, 64)])
@pytest.mark.parametrize("empty", [False, True])
def test_fused_single_cache_and_empty_prefill(cache_format, tokens, heads, empty):
    from tests.attention.mega_mla_utils import epilogue

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
    from tests.attention.mega_mla_utils import fused_call

    f = fixture(1, 16)
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
@pytest.mark.parametrize("tokens", [1, 128])
def test_mega_v41_cache_layouts(inline, mixed, tokens):
    from flashinfer.mla import (
        SparseMLASm120Wrapper,
        dsv41_fp8_quantize_pack_sparse_mla_cache,
        dsv41_fp4_quantize_pack_sparse_mla_cache,
    )
    from tests.attention.mega_mla_utils import fused_call

    f = fixture(tokens, 64)
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
    torch.testing.assert_close(actual_lse, lse, atol=0, rtol=0)
    torch.testing.assert_close(f["out"].float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(f["scales"], scales, atol=0, rtol=0)


@pytest.mark.parametrize("cache_format", ["fp8", "nvfp4"])
@pytest.mark.parametrize("tokens", [1, 128])
def test_mega_graph_kernel_boundaries(cache_format, tokens, tmp_path):
    import re
    from tests.attention.mega_mla_utils import fused_call

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
    expected = 2 if str(plan["merge"]) in ("general", "merge2") else 1
    assert len(kernels) == expected, text
    assert all("sparse_mla" in node for node in kernels), text
    assert len(nodes) == len(kernels), text


@pytest.mark.parametrize("heads", [16, 64])
@pytest.mark.parametrize("empty", [False, True])
def test_nvfp4_mega_forced_direct_decode(heads, empty):
    from flashinfer.mla._sparse_mla_sm120._prepared import (
        validate_metadata,
        device_caps,
    )
    from flashinfer.mla._sparse_mla_sm120._execution import get_sparse_mla_sm120_module

    f = fixture(1, heads, "nvfp4")
    if empty:
        for lengths in f["lengths"]:
            lengths.zero_()
    f["prologue"]()
    m = validate_metadata(
        f["q"],
        f["caches"][0],
        f["indices"][0],
        f["context"],
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        None,
        None,
        None,
        model=1,
        is_dsv4_nvfp4=True,
        extra_fp4=False,
        value_dim=512,
    )
    module = get_sparse_mla_sm120_module()
    sm, shared = device_caps(f["q"].device)
    plan = module.dsv4_nvfp4_resolve_attention(
        m._replace(variant=0), 3, 10, sm, shared, False
    )
    assert plan.inspect()["merge"] == "direct"
    lse = torch.empty(1, heads, device="cuda")
    common = (
        f["caches"][0],
        f["indices"][0],
        None,
        None,
        f["context"],
        lse,
        512**-0.5,
        f["lengths"][0],
        f["sink"],
        f["caches"][1],
        f["indices"][1],
        f["lengths"][1],
        1.0,
    )
    module.dsv4_nvfp4_execute_attention(plan, f["rotated"], *common)
    f["post"]()
    expected, scales, expected_lse = (
        f["out"].float().clone(),
        f["scales"].clone(),
        lse.clone(),
    )
    f["context"].fill_(123)
    module.dsv4_nvfp4_execute_mega_attention(
        plan,
        f["q"],
        *common,
        f["positions"],
        f["cs"],
        f["out"].view(torch.uint8),
        f["scales"],
    )
    torch.testing.assert_close(f["out"].float(), expected, atol=0, rtol=0)
    torch.testing.assert_close(f["scales"], scales, atol=0, rtol=0)
    torch.testing.assert_close(lse, expected_lse, atol=0, rtol=0)
    assert (f["context"] == 123).all()


@pytest.mark.parametrize("cache_format", ["fp8", "v41_mixed"])
@pytest.mark.parametrize("tokens", [4, 128])
@pytest.mark.parametrize("heads", [1, 12, 32, 128])
def test_mega_head_tiles(cache_format, tokens, heads):
    """Partial head tiles must preserve whole scale groups and mask stores."""
    from tests.attention.mega_mla_utils import fused_call

    f = fixture(tokens, heads, cache_format, ragged=True)
    if tokens > 64 and (
        heads in (1, 12) or (heads == 128 and cache_format == "v41_mixed")
    ):
        # These shapes are outside the existing prefill envelope. Mega must
        # reject them through the same planner rather than picking a new route.
        for call in (f["baseline"], lambda: fused_call(f)):
            with pytest.raises(
                ValueError, match="no decode kernel|FP8 prefill requires heads"
            ):
                call()
        return
    out, scales, lse = f["baseline"]()
    expected = (out.float().clone(), scales.clone(), lse.clone())
    actual_lse = fused_call(f)
    for actual, target in zip(
        (f["out"].float(), f["scales"], actual_lse), expected, strict=True
    ):
        torch.testing.assert_close(actual, target, atol=0, rtol=0)
