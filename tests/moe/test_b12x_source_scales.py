"""Static kernel on the caller's block-scale layout (no padded scale copies).

For gated NVFP4 shapes whose intermediate size is a multiple of 64 with I % 128 == 64 (I = 320 here) the static kernel
reads the source scale storage directly: the up branch tiles are the expert's first 128-row atoms, each gate tile is the
upper half of one atom followed by the lower half of the next (assembled by the DMA warp with plain 8-byte loads and
shared stores - TMA cannot address a half atom - and published through the stage's TMA full barrier, which also counts the
DMA warp's arrival after the stores), and the down scales
keep their true K extent.  A static-only wrapper therefore holds neither padded FP4 nor
padded block-scale copies; the dynamic / micro kernels build their tile-padded scales lazily on first use.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

from .utils import check_accuracy, compute_reference_moe_fp4, create_moe_tensors

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

HIDDEN, INTERMEDIATE, EXPERTS, TOPK = 256, 320, 64, 2
CAPACITY = 1024
EXTENTS = (192, 320, 448, 576, 704)
STATIC_M, DYNAMIC_M = 200, 700  # 400 pairs static (cutover 1024), 1400 pairs dynamic


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


sm120_only = pytest.mark.skipif(not _is_sm120(), reason="SM120 static kernel only")


def _wrapper(intermediate=INTERMEDIATE):
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        use_cuda_graph=True,
        max_num_tokens=CAPACITY,
    )


def _tensors(num_tokens: int, seed: int, intermediate=INTERMEDIATE):
    return create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        num_experts=EXPERTS,
        num_local_experts=EXPERTS,
        top_k=TOPK,
        seed=seed,
        interleave_gated_weights=False,
        use_nontrivial_alphas=False,
    )


def _kwargs(t):
    return {
        "x": t["x_bf16"],
        "w1_weight": t["w1_weight"],
        "w1_weight_sf": t["w1_weight_sf"],
        "w1_alpha": t["w1_alpha"],
        "fc2_input_scale": t["fc2_input_scale"],
        "w2_weight": t["w2_weight"],
        "w2_weight_sf": t["w2_weight_sf"],
        "w2_alpha": t["w2_alpha"],
        "token_selected_experts": t["token_selected_experts"],
        "token_final_scales": t["token_final_scales"],
    }


def _reference(t, num_tokens: int, intermediate=INTERMEDIATE):
    return compute_reference_moe_fp4(
        hidden_states=t["x_bf16"].float().cuda(),
        gemm1_weights=t["w1_weight_bf16"].float().cuda(),
        gemm2_weights=t["w2_weight_bf16"].float().cuda(),
        token_selected_experts=t["token_selected_experts"],
        token_final_scales=t["token_final_scales"],
        num_tokens=num_tokens,
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        fc2_input_scale=t["fc2_input_scale"],
    )


def _padded_scale_entries(t):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    ptrs = {t["w1_weight_sf"].data_ptr(), t["w2_weight_sf"].data_ptr()}
    return [k for k in md._PADDED_SCALE_CACHE if ptrs & set(k)]


def test_selection_rule_and_override(monkeypatch):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
    for intermediate in EXTENTS:
        assert md.static_source_scales(intermediate, True, "nvfp4")
        assert md.static_source_scales(intermediate, True, "w4a4")
        assert not md.static_source_scales(intermediate, False, "nvfp4")
    assert (
        md.static_source_scales(320, False) is False
    )  # non-gated: one branch, atom-aligned already
    assert md.static_source_scales(320, True, "mxfp4") is False
    for n in (
        128,
        256,
        512,
        384,
    ):  # atom-aligned branches: no half-atom gate offset to remap
        assert md.static_source_scales(n, True) is False
    for n in (80, 160, 96, 352):  # partial K atoms / single slice: padded copies stay
        assert md.static_source_scales(n, True) is False
    monkeypatch.setenv(md._STATIC_SOURCE_SCALES_ENV, "0")
    assert md.static_source_scales(320, True) is False


@sm120_only
class TestSourceScales:
    @pytest.mark.parametrize(
        "num_tokens,intermediate,strided_weight",
        [
            (1, 192, None),
            (1, 320, None),
            (2, 192, None),
            (2, 320, None),
            (2, 320, "w1_weight"),
            (2, 320, "w2_weight"),
            (3, 320, None),
            (3, 448, None),
            (3, 320, "w1_weight"),
            (3, 320, "w2_weight"),
        ],
    )
    def test_direct_micro_source_extent_matches_padded(
        self, monkeypatch, num_tokens, intermediate, strided_weight
    ):
        from flashinfer import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        monkeypatch.setattr(md, "_FORCED_BACKEND", "direct_micro")
        monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
        t = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=2560,
            intermediate_size=intermediate,
            num_experts=64,
            num_local_experts=64,
            top_k=10,
            seed=73,
            interleave_gated_weights=False,
            use_nontrivial_alphas=True,
        )
        t["fc2_input_scale"] = torch.linspace(0.7, 1.3, 64, device="cuda")
        if strided_weight is not None:
            # Keep logical weights identical while interleaving unrelated bytes.
            packed = t[strided_weight]
            t[strided_weight] = torch.stack((packed, torch.zeros_like(packed)), dim=-1)[
                ..., 0
            ]
            assert not t[strided_weight].is_contiguous()

        def wrapper():
            return B12xMoEWrapper(
                num_experts=64,
                top_k=10,
                hidden_size=2560,
                intermediate_size=intermediate,
                use_cuda_graph=True,
                max_num_tokens=8,
            )

        moe = wrapper()
        moe._static_workspace.dm_intermediate.fill_(float("nan"))
        actual = moe.run(**_kwargs(t)).clone()
        views = moe._weight_views
        assert views.source_scales
        if intermediate > 256 and strided_weight is None:
            assert not views.legacy_materialized
            assert views._padded_w13_sf_storage is None
            assert views._padded_down_sf_storage is None
        else:
            # Narrow FC2 lacks a tail mask; raw direct loads cannot use strides.
            assert views.legacy_materialized

        monkeypatch.setenv(md._STATIC_SOURCE_SCALES_ENV, "0")
        padded = wrapper()
        padded._static_workspace.dm_intermediate.fill_(float("nan"))
        expected = padded.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_dynamic_launch_pads_the_scales_lazily(self, monkeypatch):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        monkeypatch.delenv(md._STATIC_SOURCE_SCALES_ENV, raising=False)
        moe = _wrapper()
        t_static = _tensors(STATIC_M, 81)
        moe.run(**_kwargs(t_static))
        torch.cuda.synchronize()
        views = moe._weight_views
        assert views._padded_w13_sf_storage is None
        # a dynamic launch of the same weights pads on first use and is accurate
        t_dyn = _tensors(DYNAMIC_M, 82)
        for key in (
            "w1_weight",
            "w1_weight_sf",
            "w1_alpha",
            "fc2_input_scale",
            "w2_weight",
            "w2_weight_sf",
            "w2_alpha",
            "w1_weight_bf16",
            "w2_weight_bf16",
        ):
            t_dyn[key] = t_static[key]
        out = moe.run(**_kwargs(t_dyn))
        torch.cuda.synchronize()
        assert check_accuracy(out, _reference(t_dyn, DYNAMIC_M))[0]
        assert views._padded_w13_sf_storage is not None
        # ... and the static launch afterwards still reads the source layout
        out2 = moe.run(**_kwargs(t_static))
        torch.cuda.synchronize()
        assert check_accuracy(out2, _reference(t_static, STATIC_M))[0]


@sm120_only
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_static_only_extent_matches_the_padded_path_bitwise_and_holds_no_copies(
    intermediate, monkeypatch
):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    monkeypatch.delenv("FLASHINFER_B12X_STATIC_SOURCE_SCALES", raising=False)
    t = _tensors(STATIC_M, seed=100 + intermediate, intermediate=intermediate)
    scale_before = _padded_scale_entries(t)
    fp4_before = list(moe_dispatch._PADDED_FP4_CACHE)

    moe = _wrapper(intermediate)
    out = moe.run(**_kwargs(t)).clone()
    torch.cuda.synchronize()
    views = moe._weight_views
    assert views is not None and views.source_scales
    assert views.intermediate_size == intermediate
    assert _padded_scale_entries(t) == scale_before, (
        "static-only run padded the block scales"
    )
    assert list(moe_dispatch._PADDED_FP4_CACHE) == fp4_before, (
        "static-only run padded the FP4 weights"
    )
    assert views.w1_storage.data_ptr() == t["w1_weight"].data_ptr()
    assert views.w2_storage.data_ptr() == t["w2_weight"].data_ptr()
    assert views._w13_sf_storage.data_ptr() == t["w1_weight_sf"].data_ptr()
    assert views._down_sf_storage.data_ptr() == t["w2_weight_sf"].data_ptr()
    assert (
        views._padded_w13_sf_storage is None and views._padded_down_sf_storage is None
    )
    assert torch.isfinite(out).all()
    passed, pct, atol = check_accuracy(out, _reference(t, STATIC_M, intermediate))
    assert passed, f"I={intermediate}: {pct * 100:.2f}% within tol (atol={atol:.4f})"

    # Repeated launches on the same wrapper stay bitwise stable (every stage's gate scale tile
    # is published through the stage's TMA full barrier, which also counts the DMA warp's
    # arrival after its stores, before the MMA warps read it).
    for _ in range(8):
        again = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert again.data_ptr() != out.data_ptr()
        assert torch.equal(again, out), (
            f"I={intermediate}: repeated static launch differs"
        )

    # The padded-scale path on a second wrapper (environment override) is bitwise identical.
    monkeypatch.setenv(moe_dispatch._STATIC_SOURCE_SCALES_ENV, "0")
    padded = _wrapper(intermediate)
    ref = padded.run(**_kwargs(t))
    torch.cuda.synchronize()
    assert padded._weight_views is not None and not padded._weight_views.source_scales
    assert torch.equal(ref, out), (
        f"I={intermediate}: source layout differs from the padded path"
    )
    monkeypatch.delenv(moe_dispatch._STATIC_SOURCE_SCALES_ENV)


@sm120_only
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_graph_capture_after_static_warm_up_replays_bitwise(intermediate, monkeypatch):
    monkeypatch.delenv("FLASHINFER_B12X_STATIC_SOURCE_SCALES", raising=False)
    t = _tensors(STATIC_M, seed=300 + intermediate, intermediate=intermediate)
    moe = _wrapper(intermediate)
    eager = moe.run(**_kwargs(t)).clone()
    torch.cuda.synchronize()
    scale_before = _padded_scale_entries(t)
    graph = torch.cuda.CUDAGraph()
    kwargs = _kwargs(t)  # keep the captured inputs alive for the replays
    with torch.cuda.graph(graph):
        captured = moe.run(**kwargs)
    torch.cuda.synchronize()
    assert _padded_scale_entries(t) == scale_before, "capture prepared padded scales"
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), (
            f"I={intermediate}: graph replay differs from eager"
        )
