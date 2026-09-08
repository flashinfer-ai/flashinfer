"""Source-scale static kernel over every enabled intermediate extent (schedule classes).

``static_source_scales`` enables the source-layout scale addressing for every gated NVFP4 extent with I % 128 == 64.
The gate branch of such an extent has ceil(I / 128) N128 slices; the static kernel publishes slices in pairs, so an odd
slice count carries a phantom slice whose scale loads must stay inside the caller's storage.  This module executes the
enabled set: I=192 (2 slices, even), 320 (3, odd + phantom), 448 (4, even), 576 (5, odd + phantom) and 704 (6, even).
For each extent the static-only wrapper must hold no padded scale copies (pointer / cache lifetime), agree bitwise with
the padded-scale path (environment override on a second wrapper), stay stable over repeated launches and capture into a
CUDA graph after warm-up without preparing anything.
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

HIDDEN, EXPERTS, TOPK = 256, 64, 2
CAPACITY = 1024
STATIC_M = 200  # 400 routed pairs: static below the 1024-pair cutover of this capacity
EXTENTS = (192, 320, 448, 576, 704)


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


sm120_only = pytest.mark.skipif(not _is_sm120(), reason="SM120 static kernel only")


def _wrapper(intermediate: int):
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        use_cuda_graph=True,
        max_num_tokens=CAPACITY,
    )


def _tensors(intermediate: int, num_tokens: int, seed: int):
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


def _reference(t, num_tokens: int, intermediate: int):
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


def _padded_scale_entries(intermediate: int):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return [key for key in moe_dispatch._PADDED_SCALE_CACHE if key[0] == intermediate]


def _padded_fp4_entries():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return list(moe_dispatch._PADDED_FP4_CACHE)


@pytest.mark.parametrize("intermediate", EXTENTS)
def test_predicate_enables_every_extent_of_the_matrix(intermediate):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        static_source_scales,
    )

    assert intermediate % 128 == 64
    assert static_source_scales(intermediate, True, "nvfp4")
    assert static_source_scales(intermediate, True, "w4a4")
    assert not static_source_scales(intermediate, False, "nvfp4")
    assert not static_source_scales(
        intermediate + 64, True, "nvfp4"
    )  # 128-aligned: no source layout needed


@sm120_only
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_static_only_extent_matches_the_padded_path_bitwise_and_holds_no_copies(
    intermediate, monkeypatch
):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    t = _tensors(intermediate, STATIC_M, seed=100 + intermediate)
    scale_before = _padded_scale_entries(intermediate)
    fp4_before = _padded_fp4_entries()

    moe = _wrapper(intermediate)
    out = moe.run(**_kwargs(t)).clone()
    torch.cuda.synchronize()
    views = moe._weight_views
    assert views is not None and views.source_scales
    assert views.intermediate_size == intermediate
    assert _padded_scale_entries(intermediate) == scale_before, (
        "static-only run padded the block scales"
    )
    assert _padded_fp4_entries() == fp4_before, "static-only run padded the FP4 weights"
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

    # Select the padded path only in this test; production uses shape eligibility.
    monkeypatch.setattr(
        moe_dispatch, "static_source_scales", lambda *args, **kwargs: False
    )
    padded = _wrapper(intermediate)
    ref = padded.run(**_kwargs(t))
    torch.cuda.synchronize()
    assert padded._weight_views is not None and not padded._weight_views.source_scales
    assert torch.equal(ref, out), (
        f"I={intermediate}: source layout differs from the padded path"
    )


@sm120_only
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_graph_capture_after_static_warm_up_replays_bitwise(intermediate):
    t = _tensors(intermediate, STATIC_M, seed=300 + intermediate)
    moe = _wrapper(intermediate)
    eager = moe.run(**_kwargs(t)).clone()
    torch.cuda.synchronize()
    scale_before = _padded_scale_entries(intermediate)
    graph = torch.cuda.CUDAGraph()
    kwargs = _kwargs(t)  # keep the captured inputs alive for the replays
    with torch.cuda.graph(graph):
        captured = moe.run(**kwargs)
    torch.cuda.synchronize()
    assert _padded_scale_entries(intermediate) == scale_before, (
        "capture prepared padded scales"
    )
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(captured, eager), (
            f"I={intermediate}: graph replay differs from eager"
        )
