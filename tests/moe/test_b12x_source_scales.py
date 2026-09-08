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
STATIC_M, DYNAMIC_M = 200, 700  # 400 pairs static (cutover 1024), 1400 pairs dynamic


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


sm120_only = pytest.mark.skipif(not _is_sm120(), reason="SM120 static kernel only")


def _wrapper():
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        use_cuda_graph=True,
        max_num_tokens=CAPACITY,
    )


def _tensors(num_tokens: int, seed: int):
    return create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
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


def _reference(t, num_tokens: int):
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
        intermediate_size=INTERMEDIATE,
        fc2_input_scale=t["fc2_input_scale"],
    )


def _padded_scale_entries(t):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    ptrs = {t["w1_weight_sf"].data_ptr(), t["w2_weight_sf"].data_ptr()}
    return [k for k in md._PADDED_SCALE_CACHE if ptrs & set(k)]


def _weight_views(moe):
    views = getattr(moe, "_weight_views", None)
    if views is None:  # the wrapper keeps the views under another attribute name
        candidates = [
            v
            for v in vars(moe).values()
            if hasattr(v, "source_scales") and hasattr(v, "_w13_sf_storage")
        ]
        assert candidates, "wrapper carries no weight views"
        views = candidates[0]
    return views


def test_selection_rule():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    assert md.static_source_scales(320, True) is True
    assert md.static_source_scales(192, True) is True
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


@sm120_only
class TestSourceScales:
    def test_static_only_wrapper_holds_no_padded_scale_copies_and_matches_the_padded_path(
        self, monkeypatch
    ):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        t = _tensors(STATIC_M, 71)
        moe = _wrapper()
        out = moe.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        views = _weight_views(moe)
        assert views.source_scales is True
        assert (
            views._padded_w13_sf_storage is None
            and views._padded_down_sf_storage is None
        )
        assert _padded_scale_entries(t) == [], (
            "the static-only wrapper materialized padded scales"
        )
        # the static kernel's scale storages are the caller's tensors (views of the same storage)
        assert (
            views._w13_sf_storage.untyped_storage().data_ptr()
            == t["w1_weight_sf"].untyped_storage().data_ptr()
        )
        assert (
            views._down_sf_storage.untyped_storage().data_ptr()
            == t["w2_weight_sf"].untyped_storage().data_ptr()
        )
        assert check_accuracy(out, _reference(t, STATIC_M))[0]
        # the padded-scale path on a second wrapper computes bitwise the same output
        monkeypatch.setattr(md, "static_source_scales", lambda *args, **kwargs: False)
        moe_padded = _wrapper()
        out_padded = moe_padded.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        assert _weight_views(moe_padded).source_scales is False
        assert _padded_scale_entries(t), (
            "the padded path did not build its scale copies"
        )
        assert torch.equal(out, out_padded)

    def test_dynamic_launch_pads_the_scales_lazily(self):
        moe = _wrapper()
        t_static = _tensors(STATIC_M, 81)
        moe.run(**_kwargs(t_static))
        torch.cuda.synchronize()
        views = _weight_views(moe)
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

    def test_graph_capture_after_static_warmup_prepares_no_padded_scales(self):
        t = _tensors(STATIC_M, 91)
        kwargs = _kwargs(t)
        moe = _wrapper()
        eager = moe.run(**kwargs).clone()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = moe.run(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager)
        assert _weight_views(moe)._padded_w13_sf_storage is None
        assert _padded_scale_entries(t) == []
