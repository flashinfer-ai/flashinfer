"""True-extent TMA views share FP4 storage across static, micro, and dynamic.

Only the block scales need tile padding for MMA micro and generic dynamic.
Small per-call dynamic workspaces select generic; capacity workspaces with
M128 tiles select the optimized gated kernel. Both retain capture warm-up
requirements and must agree numerically without materializing FP4 copies.
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


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, minor = torch.cuda.get_device_capability()
    return major == 12


def _wrapper(max_num_tokens: int, use_cuda_graph: bool):
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
    )


def _tensors(num_tokens: int, seed: int = 2026):
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


def _cutover_tokens() -> int:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _get_static_compact_cutover_pairs,
    )

    pairs = _get_static_compact_cutover_pairs(
        "fp4", quant_mode="nvfp4", num_experts=EXPERTS, intermediate_size=INTERMEDIATE
    )
    return pairs // TOPK


def _padded_fp4_cache_entries():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return list(moe_dispatch._PADDED_FP4_CACHE)


def _capture(graph, moe, kwargs):
    with torch.cuda.graph(graph):
        return moe.run(**kwargs)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestLazyLegacyViews:
    def test_dynamic_first_shares_weights_and_is_accurate(self):
        num_tokens = _cutover_tokens() + 64
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        out = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        views = moe._weight_views
        assert not views.legacy_materialized
        assert (
            views.tma_w13_fp4.shape[0] == 2 * INTERMEDIATE
            and views.branch_major_down_fp4.shape[1] == INTERMEDIATE // 2
        ), "generic TMA views must carry the true extent per branch"
        assert views.tma_w13_fp4.data_ptr() == t["w1_weight"].data_ptr()
        assert views.branch_major_down_fp4.data_ptr() == t["w2_weight"].data_ptr()
        source_w1, source_w2 = views.w1_storage, views.w2_storage
        passed, pct, atol = check_accuracy(out, _reference(t, num_tokens))
        assert passed, f"dynamic output: {pct * 100:.2f}% within tol (atol={atol:.4f})"
        moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert (
            moe._weight_views.w1_storage is source_w1
            and moe._weight_views.w2_storage is source_w2
        )

    def test_static_then_dynamic_on_one_wrapper(self):
        big = _cutover_tokens() + 64
        t = _tensors(big)
        moe = _wrapper(max_num_tokens=big, use_cuda_graph=False)
        small = dict(_kwargs(t))
        small["x"] = t["x_bf16"][:64]
        small["token_selected_experts"] = t["token_selected_experts"][:64]
        small["token_final_scales"] = t["token_final_scales"][:64]
        out_small = moe.run(**small)
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        out_big = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        ref = _reference(t, big)
        assert check_accuracy(out_big, ref)[0]
        assert check_accuracy(out_small, ref[:64])[0]

    def test_micro_calls_share_true_extent_views(self):
        """Direct and MMA micro both reuse source FP4 storage at I=320."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens = 4  # 8 routed rows -> direct micro under auto dispatch
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        out_auto = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        assert check_accuracy(out_auto, _reference(t, num_tokens))[0]
        previous = moe_dispatch._FORCED_BACKEND
        moe_dispatch._FORCED_BACKEND = "micro"
        try:
            out = moe.run(**_kwargs(t))
            torch.cuda.synchronize()
            assert not moe._weight_views.legacy_materialized
            source_w1 = moe._weight_views.w1_storage
            assert source_w1.data_ptr() == t["w1_weight"].data_ptr()
            out = moe.run(**_kwargs(t))
            torch.cuda.synchronize()
        finally:
            moe_dispatch._FORCED_BACKEND = previous
        assert (
            moe._weight_views.w1_storage is source_w1
        )  # reused by the MMA micro kernel
        assert check_accuracy(out, _reference(t, num_tokens))[0]

    @pytest.mark.parametrize("backend", ["micro", "dynamic"])
    @pytest.mark.parametrize("weight", ["w1_weight", "w2_weight"])
    def test_strided_weights_copy_once_and_replay(self, backend, weight, monkeypatch):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        monkeypatch.setattr(moe_dispatch, "_FORCED_BACKEND", backend)
        num_tokens = 4 if backend == "micro" else _cutover_tokens() + 64
        t = _tensors(num_tokens)
        expected = _wrapper(num_tokens, True).run(**_kwargs(t)).clone()
        kwargs = _kwargs(t)
        packed = kwargs[weight]
        kwargs[weight] = torch.stack((packed, torch.zeros_like(packed)), -1)[..., 0]
        assert not kwargs[weight].is_contiguous()
        moe = _wrapper(num_tokens, True)
        actual = moe.run(**kwargs).clone()
        views = moe._weight_views
        assert not views.legacy_materialized
        pointers = (views.tma_w13_fp4.data_ptr(), views.tma_down_fp4.data_ptr())
        assert pointers[0 if weight == "w1_weight" else 1] != kwargs[weight].data_ptr()
        assert views.tma_w13_fp4.shape[0] == 2 * INTERMEDIATE
        assert views.tma_down_fp4.shape[1] == INTERMEDIATE // 2
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = moe.run(**kwargs)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, atol=2e-2, rtol=2e-2)
        assert pointers == (views.tma_w13_fp4.data_ptr(), views.tma_down_fp4.data_ptr())

    @pytest.mark.parametrize("backend", ["micro", "dynamic"])
    @pytest.mark.parametrize("weight", ["w1_weight", "w2_weight"])
    def test_functional_strided_weights_reuse_copies_during_capture(
        self, backend, weight, monkeypatch
    ):
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        monkeypatch.setattr(moe_dispatch, "_FORCED_BACKEND", backend)
        num_tokens = 4 if backend == "micro" else _cutover_tokens() + 64
        # I=160 exercises true extents with eagerly prepared block scales.
        t = create_moe_tensors(
            num_tokens=num_tokens,
            hidden_size=HIDDEN,
            intermediate_size=160,
            num_experts=EXPERTS,
            num_local_experts=EXPERTS,
            top_k=TOPK,
            interleave_gated_weights=False,
            use_nontrivial_alphas=False,
        )
        kwargs = dict(
            _kwargs(t),
            num_experts=EXPERTS,
            top_k=TOPK,
            output=torch.empty_like(t["x_bf16"]),
        )
        expected = b12x_fused_moe(**kwargs).clone()
        packed = kwargs[weight]
        kwargs[weight] = torch.stack((packed, torch.zeros_like(packed)), -1)[..., 0]
        actual = b12x_fused_moe(**kwargs).clone()
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
        # A fresh prepared view must find the eager copy instead of allocating
        # another one (or refusing capture) on every functional invocation.
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = b12x_fused_moe(**kwargs)
        for _ in range(3):
            graph.replay()
            torch.cuda.synchronize()
            torch.testing.assert_close(captured, expected, atol=2e-2, rtol=2e-2)

    def test_graph_capture_of_first_dynamic_use_is_refused_then_prewarm_works(self):
        num_tokens = _cutover_tokens() + 64
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=True)
        kwargs = _kwargs(t)
        # Warm the static path only (an eager static call must not materialize the legacy views) ...
        small = dict(kwargs)
        small["x"] = t["x_bf16"][:64]
        small["token_selected_experts"] = t["token_selected_experts"][:64]
        small["token_final_scales"] = t["token_final_scales"][:64]
        moe.run(**small)
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        # ... so capturing the first dynamic call is refused with a clear error, not a silent allocation.
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(
            (ValueError, RuntimeError), match="during CUDA graph capture"
        ):
            _capture(graph, moe, kwargs)
        assert not moe._weight_views.legacy_materialized
        # Eager warm-up prepares padded scales and kernels; replay reuses both.
        eager = moe.run(**kwargs).clone()
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = moe.run(**kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(captured, eager, atol=2e-2, rtol=2e-2)
        assert check_accuracy(captured, _reference(t, num_tokens))[0]

    def test_alternating_captured_graphs_share_one_wrapper(self):
        big = _cutover_tokens() + 64
        t = _tensors(big)
        moe = _wrapper(max_num_tokens=big, use_cuda_graph=True)
        kwargs_big = _kwargs(t)
        kwargs_small = dict(kwargs_big)
        kwargs_small["x"] = t["x_bf16"][:64]
        kwargs_small["token_selected_experts"] = t["token_selected_experts"][:64]
        kwargs_small["token_final_scales"] = t["token_final_scales"][:64]
        eager_small = moe.run(**kwargs_small).clone()
        eager_big = moe.run(
            **kwargs_big
        ).clone()  # pre-warm materializes the legacy views
        torch.cuda.synchronize()
        g_small, g_big = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(g_small):
            out_small = moe.run(**kwargs_small)
        with torch.cuda.graph(g_big):
            out_big = moe.run(**kwargs_big)
        for _ in range(3):
            g_small.replay()
            torch.cuda.synchronize()
            assert torch.allclose(out_small, eager_small, atol=2e-2, rtol=2e-2)
            g_big.replay()
            torch.cuda.synchronize()
            assert torch.allclose(out_big, eager_big, atol=2e-2, rtol=2e-2)

    @staticmethod
    def _functional_static_call(num_tokens: int, seed: int):
        from flashinfer import b12x_fused_moe

        t = _tensors(num_tokens, seed=seed)
        out = b12x_fused_moe(
            x=t["x_bf16"],
            w1_weight=t["w1_weight"],
            w1_weight_sf=t["w1_weight_sf"],
            w1_alpha=t["w1_alpha"],
            fc2_input_scale=t["fc2_input_scale"],
            w2_weight=t["w2_weight"],
            w2_weight_sf=t["w2_weight_sf"],
            w2_alpha=t["w2_alpha"],
            token_selected_experts=t["token_selected_experts"],
            token_final_scales=t["token_final_scales"],
            num_experts=EXPERTS,
            top_k=TOPK,
        )
        torch.cuda.synchronize()
        return t, out

    def test_functional_api_static_call_pads_nothing_for_source_scale_shape(self):
        """I=320 static call through the functional API: the kernel reads the callers' scale layout."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens = 64
        before_fp4 = len(_padded_fp4_cache_entries())
        before_scale = [
            key for key in moe_dispatch._PADDED_SCALE_CACHE if key[0] == INTERMEDIATE
        ]
        t, out = self._functional_static_call(num_tokens, seed=7)
        assert len(_padded_fp4_cache_entries()) == before_fp4
        after_scale = [
            key for key in moe_dispatch._PADDED_SCALE_CACHE if key[0] == INTERMEDIATE
        ]
        assert after_scale == before_scale, (
            "a static-only source-scale call must not pad the block scales"
        )
        assert check_accuracy(out, _reference(t, num_tokens))[0]

    def test_functional_api_static_call_pads_scales_only_when_source_scales_off(
        self, monkeypatch
    ):
        """With the source-scale mode disabled the static call pads the block scales (128-aligned layout) but never the FP4."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        monkeypatch.setenv(moe_dispatch._STATIC_SOURCE_SCALES_ENV, "0")
        num_tokens = 64
        before = len(_padded_fp4_cache_entries())
        t, out = self._functional_static_call(num_tokens, seed=7)
        assert len(_padded_fp4_cache_entries()) == before
        scale_only = [
            key for key in moe_dispatch._PADDED_SCALE_CACHE if key[0] == INTERMEDIATE
        ]
        assert scale_only, (
            "the 128-aligned block scales must still be padded for the static kernel"
        )
        assert check_accuracy(out, _reference(t, num_tokens))[0]


def _dynamic_key_extents():
    """branch_major_extent of every compiled dynamic kernel key for this module's shape (None = generic kernel)."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return sorted(
        {
            # branch_major_extent is the second-to-last key field; the last is
            # source_down_scales (see _dynamic_kernel_cache_key).
            key[-2]
            for key in moe_dispatch._DYNAMIC_KERNEL_CACHE
            if key[0] == "dynamic" and key[3] == EXPERTS and key[4] == HIDDEN
        },
        key=lambda v: (v is None, v or 0),
    )


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestBranchMajorDynamic:
    """The branch-paired gated NVFP4 dynamic kernel (tile M128) consumes the branch-major views."""

    # 96 routed rows per expert select the M128 dynamic tile (and with it the gated kernel); above the cutover.
    TOKENS = 96 * EXPERTS // TOPK + 64

    def _graph_wrapper(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            select_sm120_moe_backend,
        )

        assert (
            select_sm120_moe_backend(
                num_tokens=32 * EXPERTS // TOPK,
                num_topk=TOPK,
                activation_precision="fp4",
                quant_mode="nvfp4",
                num_experts=EXPERTS,
                intermediate_size=INTERMEDIATE,
            )
            == "dynamic"
        )
        return _wrapper(max_num_tokens=self.TOKENS, use_cuda_graph=True)

    def test_gated_dynamic_streams_branch_major_views_without_legacy_copies(self):
        t = _tensors(self.TOKENS)
        moe = self._graph_wrapper()
        before = len(_padded_fp4_cache_entries())
        out = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert moe._dynamic_workspace.tile_m == 128
        views = moe._weight_views
        assert not views.legacy_materialized, (
            "the gated dynamic kernel must not materialize the legacy views"
        )
        assert views.w13_fp4 is None and views.down_fp4 is None
        assert views.static_w13_fp4.shape[0] == INTERMEDIATE  # true extent, 2E batches
        assert views.static_w13_fp4.shape[2] == 2 * EXPERTS
        assert len(_padded_fp4_cache_entries()) == before
        assert INTERMEDIATE in _dynamic_key_extents(), (
            "the compiled dynamic key must carry the branch-major extent"
        )
        passed, pct, atol = check_accuracy(out, _reference(t, self.TOKENS))
        assert passed, (
            f"branch-major dynamic output: {pct * 100:.2f}% within tol (atol={atol:.4f})"
        )

    def test_gated_dynamic_agrees_with_generic_kernel_on_true_extent_views(self):
        """Two independent kernels, same FP4 numerics: the branch-major gated kernel (capacity workspace, tile M128)
        and the generic kernel on concatenated true-extent views (per-call workspace, 32 rows per expert -> tile M32) must
        agree at the FP4 noise floor on the same routes."""
        num_tokens = (
            32 * EXPERTS // TOPK
        )  # dynamic (above the cutover), below the M128 density
        t = _tensors(num_tokens, seed=11)
        moe_gated = self._graph_wrapper()
        out_gated = moe_gated.run(**_kwargs(t)).float()
        torch.cuda.synchronize()
        assert moe_gated._dynamic_workspace.tile_m == 128
        assert not moe_gated._weight_views.legacy_materialized
        moe_generic = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        out_generic = moe_generic.run(**_kwargs(t)).float()
        torch.cuda.synchronize()
        assert not moe_generic._weight_views.legacy_materialized
        assert None in _dynamic_key_extents()
        rel = ((out_gated - out_generic).norm() / out_generic.norm()).item()
        assert rel < 0.03, f"gated vs generic dynamic rel L2 {rel:.3e}"
        assert check_accuracy(out_generic, _reference(t, num_tokens))[0]
