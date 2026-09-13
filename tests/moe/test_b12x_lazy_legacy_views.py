"""SM12x operand preparation and routing-state reuse across eager/graph calls.

Small per-call workspaces exercise generic Dynamic on padded views; capacity
workspaces exercise gated Dynamic on true-extent views. Source-scale tails and
deferred routing initialization share the same tensor/reference fixtures.
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


def _wrapper(
    max_num_tokens: int = 1024,
    use_cuda_graph: bool = True,
    intermediate: int = INTERMEDIATE,
):
    from flashinfer import B12xMoEWrapper

    return B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        use_cuda_graph=use_cuda_graph,
        max_num_tokens=max_num_tokens,
    )


def _tensors(num_tokens: int, seed: int = 2026, intermediate: int = INTERMEDIATE):
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


def _reference(t, num_tokens: int, intermediate: int = INTERMEDIATE):
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
    def test_static_only_wrapper_never_materializes_padded_fp4(self):
        num_tokens = 64  # static: above the micro band, below the cutover
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        before = len(_padded_fp4_cache_entries())
        out = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        views = moe._weight_views
        assert views is not None and not views.legacy_materialized
        assert views.w13_fp4 is None and views.down_fp4 is None
        assert (
            views.static_w13_fp4 is not None and views.intermediate_size == INTERMEDIATE
        )
        assert len(_padded_fp4_cache_entries()) == before, (
            "static-only run padded the FP4 weights"
        )
        # I=320 is a source-scale shape: the static kernel reads the callers' scale
        # storage, so neither padded FP4 nor padded block scales exist and every
        # storage the views hold is the callers' tensor.
        assert views.source_scales
        assert views.w1_storage.data_ptr() == t["w1_weight"].data_ptr()
        assert views.w2_storage.data_ptr() == t["w2_weight"].data_ptr()
        assert views._w13_sf_storage.data_ptr() == t["w1_weight_sf"].data_ptr()
        assert views._down_sf_storage.data_ptr() == t["w2_weight_sf"].data_ptr()
        assert views._padded_w13_sf_storage is None
        assert views._padded_down_sf_storage is None
        passed, pct, atol = check_accuracy(out, _reference(t, num_tokens))
        assert passed, (
            f"static-only output: {pct * 100:.2f}% within tol (atol={atol:.4f})"
        )
        # A second static call reuses the views without materializing.
        moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized

    def test_dynamic_first_materializes_once_and_is_accurate(self):
        num_tokens = _cutover_tokens() + 64
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        out = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        views = moe._weight_views
        assert views.legacy_materialized
        assert (
            views.w13_fp4.shape[0] == 2 * 384 and views.down_fp4.shape[1] == 384 // 2
        ), (
            "legacy views must carry the 128-aligned extent (384 = align128(320)) per branch"
        )
        padded_w1, padded_w2 = views.w1_storage, views.w2_storage
        passed, pct, atol = check_accuracy(out, _reference(t, num_tokens))
        assert passed, f"dynamic output: {pct * 100:.2f}% within tol (atol={atol:.4f})"
        moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert (
            moe._weight_views.w1_storage is padded_w1
            and moe._weight_views.w2_storage is padded_w2
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
        assert moe._weight_views.legacy_materialized
        ref = _reference(t, big)
        assert check_accuracy(out_big, ref)[0]
        assert check_accuracy(out_small, ref[:64])[0]

    def test_micro_calls_materialize_legacy_views(self):
        """Every static-family sub-backend except the static kernel (direct CUDA-core micro at < 32 routed rows, the MMA
        micro kernel) indexes the tile-padded weights: their first call materializes the legacy copies once."""
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        num_tokens = 4  # 8 routed rows -> direct micro under auto dispatch
        t = _tensors(num_tokens)
        moe = _wrapper(max_num_tokens=num_tokens, use_cuda_graph=False)
        out_auto = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
        assert moe._weight_views.legacy_materialized
        assert check_accuracy(out_auto, _reference(t, num_tokens))[0]
        padded_w1 = moe._weight_views.w1_storage
        previous = moe_dispatch._FORCED_BACKEND
        moe_dispatch._FORCED_BACKEND = "micro"
        try:
            out = moe.run(**_kwargs(t))
            torch.cuda.synchronize()
        finally:
            moe_dispatch._FORCED_BACKEND = previous
        assert (
            moe._weight_views.w1_storage is padded_w1
        )  # materialized once, reused by the MMA micro kernel
        assert check_accuracy(out, _reference(t, num_tokens))[0]

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
            ValueError, match="cannot be created during CUDA graph capture"
        ):
            _capture(graph, moe, kwargs)
        assert not moe._weight_views.legacy_materialized
        # The natural pre-warm - one eager dynamic call - materializes; capture and replay then work and match eager.
        eager = moe.run(**kwargs).clone()
        torch.cuda.synchronize()
        assert moe._weight_views.legacy_materialized
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

        monkeypatch.setattr(
            moe_dispatch, "static_source_scales", lambda *args, **kwargs: False
        )
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
    """branch_major_extent of every compiled dynamic kernel key for this module's shape (None = legacy views)."""
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return sorted(
        {
            key[-1]
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

    def test_first_gated_dynamic_capture_is_refused_until_prewarmed(self):
        """No legacy views are needed on this path, but the first call still prepares the 128-aligned block scales,
        so a cold capture is refused (no silent allocation inside a capture); after one eager warm-up the
        capture succeeds without ever materializing the legacy views and the replay matches the eager output."""
        t = _tensors(self.TOKENS, seed=5)
        moe = self._graph_wrapper()
        kwargs = _kwargs(t)
        graph = torch.cuda.CUDAGraph()
        with pytest.raises(RuntimeError, match="warm-up"):
            _capture(graph, moe, kwargs)
        assert moe._weight_views is None or not moe._weight_views.legacy_materialized
        eager = moe.run(**kwargs).clone()  # the natural pre-warm
        torch.cuda.synchronize()
        assert not moe._weight_views.legacy_materialized
        graph = torch.cuda.CUDAGraph()
        captured = _capture(graph, moe, kwargs)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.allclose(captured, eager, atol=2e-2, rtol=2e-2)
        assert check_accuracy(eager, _reference(t, self.TOKENS))[0]
        assert not moe._weight_views.legacy_materialized

    def test_gated_dynamic_agrees_with_generic_kernel_on_legacy_views(self):
        """Two independent kernels, same FP4 numerics: the branch-major gated kernel (capacity workspace, tile M128)
        and the generic kernel on the padded legacy views (per-call workspace, 32 rows per expert -> tile M32) must
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
        assert moe_generic._weight_views.legacy_materialized
        assert None in _dynamic_key_extents()
        rel = ((out_gated - out_generic).norm() / out_generic.norm()).item()
        assert rel < 0.03, f"gated vs generic dynamic rel L2 {rel:.3e}"
        assert check_accuracy(out_generic, _reference(t, num_tokens))[0]


STATIC_M, DYNAMIC_M = 200, 700
EXTENTS = (192, 320, 448, 576, 704)


def _clean_marker() -> int:
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_static_kernel import (
        _ROUTE_STATE_CLEAN,
    )

    return int(_ROUTE_STATE_CLEAN)


def _assert_counters_clean(ws, *, marker: bool):
    """The exact state the static prologue's full clear produces (marker aside)."""
    torch.cuda.synchronize()
    weight_e = int(ws.global_to_local_expert.shape[0])
    scratch = ws.virt_route_scratch
    claim_slot = int(scratch.numel()) - 8
    assert int(ws.active_expert_count.item()) == 0
    assert int(ws.row_counts.abs().sum().item()) == 0
    assert int(scratch[:weight_e].abs().sum().item()) == 0, "row allocators"
    assert bool((scratch[weight_e:claim_slot] == -1).all().item()), "chunk map"
    assert bool((scratch[claim_slot:] == 0).all().item()), "work-claim counter + pad"
    assert bool((ws.global_to_local_expert == -1).all().item())
    assert int(ws.route_state[1:].abs().sum().item()) == 0, "reserved slots stay zero"
    assert (int(ws.route_state[0].item()) == _clean_marker()) is marker


def _run_static(moe, num_tokens: int, seed: int):
    t = _tensors(num_tokens, seed)
    out = moe.run(**_kwargs(t))
    torch.cuda.synchronize()
    passed, pct, atol = check_accuracy(out, _reference(t, num_tokens))
    assert passed, (num_tokens, seed, pct, atol)
    return out


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestDeferredInit:
    def test_same_workspace_launch_sequence_matches_the_reference(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        moe = _wrapper()
        # varying M and routes on one workspace: deferred launches (>= 256 routed pairs) restore the
        # counters and publish the marker; smaller static launches clear in their prologue and withdraw
        # the marker; tiny-decode launches (<= 8 tokens, micro path) leave the static state untouched
        state = None
        for num_tokens, seed in (
            (200, 1),
            (16, 2),
            (64, 3),
            (200, 4),
            (8, 5),
            (300, 6),
            (4, 7),
            (150, 8),
            (200, 9),
        ):
            _run_static(moe, num_tokens, seed)
            ws = moe._static_workspace
            torch.cuda.synchronize()
            if num_tokens <= md._MICRO_MAX_TOKENS:
                assert state is not None
                assert int(ws.route_state[0].item()) == state
            elif md._static_deferred_init(num_tokens * TOPK):
                _assert_counters_clean(ws, marker=True)
                state = _clean_marker()
            else:
                assert int(ws.route_state[0].item()) == 0
                state = 0

    def test_tiny_decode_launch_between_static_launches_keeps_the_clean_state(self):
        moe = _wrapper()
        _run_static(moe, 200, 11)
        ws = moe._static_workspace
        _assert_counters_clean(ws, marker=True)
        # tiny decode (micro / direct micro path) on the same wrapper
        for num_tokens in (1, 2, 4, 8):
            _run_static(moe, num_tokens, 20 + num_tokens)
            _assert_counters_clean(ws, marker=True)
        _run_static(moe, 200, 12)
        _assert_counters_clean(ws, marker=True)

    def test_forced_full_clear_and_deferred_off_agree_bitwise(self, monkeypatch):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        t = _tensors(200, 31)
        moe = _wrapper()
        out_deferred = moe.run(**_kwargs(t)).clone()
        ws = moe._static_workspace
        _assert_counters_clean(ws, marker=True)
        # a workspace without the marker takes the full clear and produces the same output
        ws.route_state.zero_()
        _assert_counters_clean(ws, marker=False)
        out_cleared = moe.run(**_kwargs(t)).clone()
        _assert_counters_clean(ws, marker=True)
        assert torch.equal(out_deferred, out_cleared)
        # the previous schedule (clear in every prologue) on a fresh wrapper
        monkeypatch.setattr(md, "_static_deferred_init", lambda routed_pairs: False)
        assert md._static_deferred_init(200 * TOPK) is False
        moe_off = _wrapper()
        out_off = moe_off.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        assert int(moe_off._static_workspace.route_state[0].item()) == 0, (
            "the non-deferred kernel never publishes the marker"
        )
        assert torch.equal(out_deferred, out_off)
        assert check_accuracy(out_deferred, _reference(t, 200))[0]

    def test_cuda_graph_replay_and_alternating_graphs(self):
        moe = _wrapper()
        cases = {}
        for num_tokens, seed in ((150, 41), (300, 42)):
            t = _tensors(num_tokens, seed)
            kwargs = _kwargs(t)
            eager = moe.run(
                **kwargs
            ).clone()  # eager warm-up of this M (compile / load)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                captured = moe.run(**kwargs)
            # the captured launch reads these input tensors on every replay: keep them alive
            cases[num_tokens] = (
                graph,
                captured,
                eager,
                _reference(t, num_tokens),
                t,
                kwargs,
            )
        ws = moe._static_workspace
        for _ in range(3):
            for num_tokens in (150, 300):
                graph, captured, eager, ref, _keep_t, _keep_kwargs = cases[num_tokens]
                graph.replay()
                torch.cuda.synchronize()
                assert torch.equal(captured, eager), num_tokens
                assert check_accuracy(captured, ref)[0]
                _assert_counters_clean(ws, marker=True)


@pytest.mark.parametrize("routed_pairs", [0, 1, 255, 256, 257, 1920, 81920])
def test_deferred_init_routing_density(routed_pairs):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    assert md._static_deferred_init(routed_pairs) is (routed_pairs >= 256)


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


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestSourceScaleLifecycle:
    def test_dynamic_launch_pads_the_scales_lazily(self):
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


def _padded_scale_entries(intermediate: int):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    return [key for key in moe_dispatch._PADDED_SCALE_CACHE if key[0] == intermediate]


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


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_static_only_extent_matches_the_padded_path_bitwise_and_holds_no_copies(
    intermediate, monkeypatch
):
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    t = _tensors(STATIC_M, seed=100 + intermediate, intermediate=intermediate)
    scale_before = _padded_scale_entries(intermediate)
    fp4_before = _padded_fp4_cache_entries()

    moe = _wrapper(intermediate=intermediate)
    out = moe.run(**_kwargs(t)).clone()
    torch.cuda.synchronize()
    views = moe._weight_views
    assert views is not None and views.source_scales
    assert views.intermediate_size == intermediate
    assert _padded_scale_entries(intermediate) == scale_before, (
        "static-only run padded the block scales"
    )
    assert _padded_fp4_cache_entries() == fp4_before, (
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

    # Select the padded path only in this test; production uses shape eligibility.
    monkeypatch.setattr(
        moe_dispatch, "static_source_scales", lambda *args, **kwargs: False
    )
    padded = _wrapper(intermediate=intermediate)
    ref = padded.run(**_kwargs(t))
    torch.cuda.synchronize()
    assert padded._weight_views is not None and not padded._weight_views.source_scales
    assert torch.equal(ref, out), (
        f"I={intermediate}: source layout differs from the padded path"
    )


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
@pytest.mark.parametrize("intermediate", EXTENTS)
def test_graph_capture_after_static_warm_up_replays_bitwise(intermediate):
    t = _tensors(STATIC_M, seed=300 + intermediate, intermediate=intermediate)
    moe = _wrapper(intermediate=intermediate)
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
