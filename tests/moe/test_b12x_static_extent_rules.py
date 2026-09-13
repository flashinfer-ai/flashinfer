"""Static-family operand extent rules of the SM12x W4A4 MoE dispatch.

Two shape classes cannot stream the caller's weights through TMA views at the true intermediate extent:

* a single N128 slice (intermediate size <= 128): the retained2 static kernel pairs two slices per group and produced
  wrong outputs when handed a one-slice extent, so the static kernel gets 256-aligned operands (and scale storages) of
  its own while dynamic and micro keep the 128-aligned ones;
* an intermediate size that is not a multiple of 32: the packed down row is not a 16-byte TMA stride, so the static and
  branch-major dynamic kernels stream the 128-padded copies at the aligned extent.

Every other shape keeps the true-extent views (no padded FP4 copies for static-only callers).  These tests pin the
selection and the accuracy of each class against the BF16 reference on small shapes.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

from .utils import check_accuracy, compute_reference_moe_fp4

pytestmark = pytest.mark.skipif(
    not (torch.cuda.is_available() and is_cute_dsl_available()),
    reason="CUDA + CuTe-DSL required",
)

HIDDEN, EXPERTS, TOPK = 256, 8, 2


def _is_sm120() -> bool:
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major == 12


def _tensors(num_tokens: int, intermediate: int, seed: int):
    """Per-expert quantized tensors (the shared utility quantizes the flattened stack and cannot pad 2*I rows that
    are not a 128 multiple per expert)."""
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize

    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(num_tokens, HIDDEN, dtype=torch.bfloat16, device=dev) / 10
    weights = torch.softmax(torch.randn(num_tokens, EXPERTS, device=dev), 1)
    weights, selected = torch.topk(weights, TOPK, -1)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    gs = torch.tensor([1.0], device=dev)
    w1 = (
        torch.randn(EXPERTS, 2 * intermediate, HIDDEN, dtype=torch.bfloat16, device=dev)
        / 10
    )
    w2 = (
        torch.randn(EXPERTS, HIDDEN, intermediate, dtype=torch.bfloat16, device=dev)
        / 10
    )
    q1, s1, q2, s2 = [], [], [], []
    for e in range(EXPERTS):
        q, s = fp4_quantize(
            w1[e], global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        q1.append(q)
        s1.append(s)
        q, s = fp4_quantize(
            w2[e], global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        q2.append(q)
        s2.append(s)
    ones = torch.ones(EXPERTS, device=dev)
    return {
        "x": x,
        "w1_weight": torch.stack(q1),
        "w1_weight_sf": convert_sf_to_mma_layout(
            torch.cat(s1), m=2 * intermediate, k=HIDDEN, num_groups=EXPERTS
        ),
        "w1_weight_bf16": w1,
        "w1_alpha": ones,
        "fc2_input_scale": torch.tensor([1.0], device=dev),
        "w2_weight": torch.stack(q2),
        "w2_weight_sf": convert_sf_to_mma_layout(
            torch.cat(s2), m=HIDDEN, k=intermediate, num_groups=EXPERTS
        ),
        "w2_weight_bf16": w2,
        "w2_alpha": ones.clone(),
        "token_selected_experts": selected.to(torch.int32),
        "token_final_scales": weights,
    }


def _tensors_shape(num_tokens: int, E: int, H: int, I: int, TOPK: int, seed: int):
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout
    from flashinfer.fp4_quantization import fp4_quantize

    torch.manual_seed(seed)
    dev = "cuda"
    x = torch.randn(num_tokens, H, dtype=torch.bfloat16, device=dev) / 10
    weights = torch.softmax(torch.randn(num_tokens, E, device=dev), 1)
    weights, selected = torch.topk(weights, TOPK, -1)
    weights = (weights / weights.sum(-1, keepdim=True)).float()
    gs = torch.tensor([1.0], device=dev)
    w1 = torch.randn(E, 2 * I, H, dtype=torch.bfloat16, device=dev) / 10
    w2 = torch.randn(E, H, I, dtype=torch.bfloat16, device=dev) / 10
    q1, s1, q2, s2 = [], [], [], []
    for e in range(E):
        q, s_ = fp4_quantize(
            w1[e], global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        q1.append(q)
        s1.append(s_)
        q, s_ = fp4_quantize(
            w2[e], global_scale=gs, sf_vec_size=16, is_sf_swizzled_layout=True
        )
        q2.append(q)
        s2.append(s_)
    ones = torch.ones(E, device=dev)
    return {
        "x": x,
        "w1_weight": torch.stack(q1),
        "w1_weight_sf": convert_sf_to_mma_layout(
            torch.cat(s1), m=2 * I, k=H, num_groups=E
        ),
        "w1_weight_bf16": w1,
        "w1_alpha": ones,
        "fc2_input_scale": torch.tensor([1.0], device=dev),
        "w2_weight": torch.stack(q2),
        "w2_weight_sf": convert_sf_to_mma_layout(torch.cat(s2), m=H, k=I, num_groups=E),
        "w2_weight_bf16": w2,
        "w2_alpha": ones.clone(),
        "token_selected_experts": selected.to(torch.int32),
        "token_final_scales": weights,
    }


def _reference_shape(t, num_tokens: int, E: int, H: int, I: int, TOPK: int):
    return compute_reference_moe_fp4(
        hidden_states=t["x"].float(),
        gemm1_weights=t["w1_weight_bf16"].float(),
        gemm2_weights=t["w2_weight_bf16"].float(),
        token_selected_experts=t["token_selected_experts"],
        token_final_scales=t["token_final_scales"],
        num_tokens=num_tokens,
        num_experts=E,
        top_k=TOPK,
        hidden_size=H,
        intermediate_size=I,
        fc2_input_scale=t["fc2_input_scale"],
    ).float()


def _kwargs(t):
    return {k: v for k, v in t.items() if k not in ("w1_weight_bf16", "w2_weight_bf16")}


def _reference(t, num_tokens: int, intermediate: int):
    return compute_reference_moe_fp4(
        hidden_states=t["x"].float(),
        gemm1_weights=t["w1_weight_bf16"].float(),
        gemm2_weights=t["w2_weight_bf16"].float(),
        token_selected_experts=t["token_selected_experts"],
        token_final_scales=t["token_final_scales"],
        num_tokens=num_tokens,
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        fc2_input_scale=t["fc2_input_scale"],
    ).float()


def _rel_l2(a, b):
    return ((a.float() - b).norm() / b.norm()).item()


def _run(intermediate: int, backend: str, num_tokens: int, seed: int = 5):
    """One forced-backend call on a capacity wrapper (tile M128 -> gated dynamic kernel); returns (rel L2, views)."""
    from flashinfer import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    t = _tensors(num_tokens, intermediate, seed)
    moe = B12xMoEWrapper(
        num_experts=EXPERTS,
        top_k=TOPK,
        hidden_size=HIDDEN,
        intermediate_size=intermediate,
        use_cuda_graph=True,
        max_num_tokens=max(num_tokens, DYNAMIC_TOKENS),
    )
    moe_dispatch._FORCED_BACKEND = backend  # None = auto dispatch
    try:
        out = moe.run(**_kwargs(t))
        torch.cuda.synchronize()
    finally:
        moe_dispatch._FORCED_BACKEND = None
    ref = _reference(t, num_tokens, intermediate)
    assert check_accuracy(out, ref)[0]
    return _rel_l2(out, ref), moe._weight_views


NOISE_FLOOR = 0.30  # FP4 quantization noise of these tensors is ~0.245; a mis-streamed operand gives > 1.0
DYNAMIC_TOKENS = 1024  # above the E8 cutover (512 tokens) and >= 96 routed rows per expert (tile M128 workspace)


@pytest.mark.parametrize("quant_mode,block", [("nvfp4", 16), ("mxfp4", 32)])
def test_w4a4_dimension_contract(quant_mode, block):
    from flashinfer import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    for hidden in (0, 144, 288):
        with pytest.raises(ValueError, match="hidden_size"):
            B12xMoEWrapper(
                num_experts=8,
                top_k=2,
                hidden_size=hidden,
                intermediate_size=320,
                quant_mode=quant_mode,
            )
    for intermediate in (0, block + 1):
        with pytest.raises(ValueError, match="intermediate_size"):
            md._validate_w4a4_dimensions(256, intermediate, quant_mode)
    for intermediate in (block, 96, 288, 320, 544):
        md._validate_w4a4_dimensions(256, intermediate, quant_mode)


def test_functional_hidden_tail_rejected_before_launch():
    from flashinfer import b12x_fused_moe

    t = _tensors_shape(17, 8, 288, 320, 2, seed=193)
    with pytest.raises(ValueError, match="hidden_size"):
        b12x_fused_moe(**_kwargs(t), num_experts=8, top_k=2)


@pytest.mark.parametrize("intermediate", [192, 320])
@pytest.mark.parametrize("backend", ["static", "dynamic"])
@pytest.mark.parametrize("inference_scale", [False, True])
def test_per_expert_scale_is_not_padded_and_replays_live(
    intermediate, backend, inference_scale, monkeypatch
):
    from flashinfer import B12xMoEWrapper, b12x_fused_moe
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

    E, H, M, topk = intermediate, 256, 17, 2
    t = _tensors_shape(M, E, H, intermediate, topk, seed=191)
    reference = _reference_shape(t, M, E, H, intermediate, topk)
    with torch.inference_mode(inference_scale):
        t["fc2_input_scale"] = torch.ones(E, device="cuda")
    monkeypatch.setattr(md, "_FORCED_BACKEND", backend)

    def make_wrapper():
        return B12xMoEWrapper(
            num_experts=E,
            top_k=topk,
            hidden_size=H,
            intermediate_size=intermediate,
            use_cuda_graph=True,
            max_num_tokens=128,
        )

    w = make_wrapper()
    before = w.run(**_kwargs(t)).clone()
    assert _rel_l2(before, reference) < 0.30
    functional = b12x_fused_moe(**_kwargs(t), num_experts=E, top_k=topk).clone()
    assert _rel_l2(functional, before.float()) < 0.01
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = w.run(**_kwargs(t))
    with torch.inference_mode():
        t["fc2_input_scale"].copy_(torch.linspace(0.25, 1.75, E, device="cuda"))
    fresh = make_wrapper()
    new_t = dict(t, fc2_input_scale=t["fc2_input_scale"].clone())
    expected = fresh.run(**_kwargs(new_t)).clone()
    again = w.run(**_kwargs(t)).clone()
    graph.replay()
    replay = captured.clone()
    functional = b12x_fused_moe(**_kwargs(t), num_experts=E, top_k=topk).clone()
    torch.cuda.synchronize()
    for out in (again, replay, functional):
        assert torch.isfinite(out).all()
        assert _rel_l2(out, expected.float()) < 0.01
    assert t["fc2_input_scale"].shape == (E,)


def test_scale_shape_validation():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
        _expand_to_experts,
    )

    for shape in ((7,), (8, 1)):
        with pytest.raises(ValueError, match="per-expert scale"):
            _expand_to_experts(torch.ones(shape, device="cuda"), 8)


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
class TestStaticExtentRules:
    @pytest.mark.parametrize("top_k", [1, 2])
    def test_three_slice_merge_requires_the_m64_path(self, top_k, monkeypatch):
        from flashinfer import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch as md

        num_tokens = 1920 // top_k
        t = _tensors_shape(num_tokens, 128, 256, 320, top_k, seed=81)
        modes = []
        get_kernel = md._get_static_kernel

        def record_kernel(*args, **kwargs):
            modes.append(kwargs["merged_groups"])
            return get_kernel(*args, **kwargs)

        monkeypatch.setattr(md, "_get_static_kernel", record_kernel)
        wrapper = B12xMoEWrapper(
            num_experts=128,
            top_k=top_k,
            hidden_size=256,
            intermediate_size=320,
            use_cuda_graph=True,
            max_num_tokens=num_tokens,
        )
        out = wrapper.run(**_kwargs(t)).clone()
        torch.cuda.synchronize()
        assert modes == [top_k > 1]
        assert check_accuracy(
            out, _reference_shape(t, num_tokens, 128, 256, 320, top_k)
        )[0]

    def test_rule_predicates(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            static_needs_256_extent,
            true_extent_views_supported,
        )

        assert [static_needs_256_extent(i) for i in (64, 80, 128, 160, 320, 512)] == [
            True,
            True,
            True,
            False,
            False,
            False,
        ]
        assert [
            true_extent_views_supported(i) for i in (64, 80, 272, 288, 320, 336)
        ] == [True, False, False, True, True, False]

    def test_single_slice_shapes_keep_the_flat_cutover(self):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_dispatch import (
            _get_static_compact_cutover_pairs,
        )

        # The static family streams a 256-aligned extent for one N128 slice (twice the true weights), so the
        # density-widened static band does not apply: the boundary stays at the flat 1024 routed pairs.
        for intermediate in (64, 80, 128):
            assert (
                _get_static_compact_cutover_pairs(
                    "fp4",
                    quant_mode="nvfp4",
                    num_experts=512,
                    intermediate_size=intermediate,
                )
                == 1024
            )
        assert (
            _get_static_compact_cutover_pairs(
                "fp4", quant_mode="nvfp4", num_experts=512, intermediate_size=320
            )
            == 16 * 512
        )
        assert (
            _get_static_compact_cutover_pairs(
                "fp4", quant_mode="nvfp4", num_experts=512, intermediate_size=512
            )
            == 8 * 512
        )

    @pytest.mark.parametrize("intermediate", [64, 128])
    def test_single_slice_static_uses_256_aligned_operands(self, intermediate):
        rel, views = _run(intermediate, "static", 64)
        assert rel < NOISE_FLOOR, f"static I={intermediate} rel L2 {rel:.3f}"
        assert views.static_intermediate_size == 256
        assert (
            views.static_w13_fp4.shape[0] == 256
            and views.static_down_fp4.shape[1] == 256 // 2
        )
        assert (
            views.static_w13_sf_storage is not None
            and views.static_down_sf_storage is not None
        )
        # dynamic and micro operands stay 128-aligned
        assert views.intermediate_size == intermediate

    @pytest.mark.parametrize("intermediate", [64, 128])
    def test_single_slice_dynamic_keeps_branch_major_true_extent(self, intermediate):
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        # Above the cutover (auto dispatch) on the capacity wrapper: tile M128 -> the gated kernel.
        rel, views = _run(intermediate, None, DYNAMIC_TOKENS)
        assert rel < NOISE_FLOOR, f"dynamic I={intermediate} rel L2 {rel:.3f}"
        assert views.intermediate_size == intermediate
        assert views.branch_major_w13_fp4.shape[0] == intermediate
        assert (128, intermediate) in {
            (k[8][0], k[-1])
            for k in moe_dispatch._DYNAMIC_KERNEL_CACHE
            if k[0] == "dynamic" and k[3] == EXPERTS
        }

    def test_single_slice_unaligned_stride_dynamic_uses_padded_extent(self):
        """I=80: one N128 slice (static family at 256) and a non-16-byte down stride (dynamic at the padded 128)."""
        rel_s, views = _run(80, "static", 64)
        assert rel_s < NOISE_FLOOR and views.static_intermediate_size == 256
        rel_d, views = _run(80, None, DYNAMIC_TOKENS)
        assert rel_d < NOISE_FLOOR, f"dynamic I=80 rel L2 {rel_d:.3f}"
        assert (
            views.intermediate_size == 128
            and views.branch_major_w13_fp4.shape[0] == 128
        )
        assert views.legacy_materialized

    @pytest.mark.parametrize(
        "shape", [(256, 4096, 128, 8), (8, 2560, 64, 2), (256, 4096, 64, 8)]
    )
    @pytest.mark.parametrize("num_tokens", [1, 2, 4])
    def test_single_slice_micro_band_uses_256_aligned_operands(self, shape, num_tokens):
        """Direct CUDA-core micro (< 32 routed rows) consumes the static family's 256-aligned copies for single-slice
        shapes (its 128-aligned operands produced NaN outputs at E>=256 / H4096); MMA micro (<= 8 tokens) stays on the
        128-aligned operands, which are correct and 25% faster than the 256 extent at E512/H4096/I128."""
        from flashinfer import B12xMoEWrapper
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        E, H, I, TOPK = shape
        t = _tensors_shape(num_tokens, E, H, I, TOPK, seed=21)
        moe = B12xMoEWrapper(
            num_experts=E,
            top_k=TOPK,
            hidden_size=H,
            intermediate_size=I,
            use_cuda_graph=True,
            max_num_tokens=8192,
        )
        getters = []
        names = ("_get_direct_micro_kernel", "_get_micro_kernel", "_get_static_kernel")
        originals = {name: getattr(moe_dispatch, name) for name in names}
        for name in names:

            def wrapped(*a, _o=originals[name], _n=name, **k):
                getters.append(_n)
                return _o(*a, **k)

            setattr(moe_dispatch, name, wrapped)
        try:
            out = moe.run(**_kwargs(t))
            torch.cuda.synchronize()
        finally:
            for name in names:
                setattr(moe_dispatch, name, originals[name])
        ref = _reference_shape(t, num_tokens, E, H, I, TOPK)
        assert torch.isfinite(out).all(), (
            f"{shape} M={num_tokens}: non-finite output from {getters}"
        )
        assert _rel_l2(out, ref) < NOISE_FLOOR, (
            f"{shape} M={num_tokens}: rel L2 {_rel_l2(out, ref):.3f} from {getters}"
        )
        views = moe._weight_views
        assert (
            views.static_family_w1_storage is not None
            and views.static_family_w1_storage.shape[1] == 2 * 256
        )
        expected = (
            "_get_direct_micro_kernel"
            if num_tokens * TOPK < 32
            else "_get_micro_kernel"
        )
        assert getters and getters[-1] == expected, getters
        n_aligned = (I + 127) // 128 * 128
        if expected == "_get_direct_micro_kernel":
            # direct micro: 256-aligned family operands (launch key: weight_E, m, k, n, ...)
            assert any(
                key[3] == 256
                for key in moe_dispatch._DIRECT_MICRO_LAUNCH_CACHE
                if key[0] == E and key[2] == H
            )
        else:
            # MMA micro keeps the 128-aligned operands (correct and faster than the 256 extent);
            # key: ("micro", quant_mode, state_E, weight_E, m, k, n, ...)
            assert any(
                key[6] == n_aligned
                for key in moe_dispatch._MICRO_KERNEL_CACHE
                if key[0] == "micro" and key[3] == E and key[5] == H
            )
            assert views.legacy_materialized

    @pytest.mark.parametrize("intermediate", [272, 336])
    def test_unaligned_stride_uses_padded_views_for_both_kernels(self, intermediate):
        n_aligned = (intermediate + 127) // 128 * 128
        rel_s, views = _run(intermediate, "static", 64)
        assert rel_s < NOISE_FLOOR, f"static I={intermediate} rel L2 {rel_s:.3f}"
        assert views.legacy_materialized and views.intermediate_size == n_aligned
        assert (
            views.static_intermediate_size == n_aligned
            and views.static_w13_fp4.shape[0] == n_aligned
        )
        assert (
            views.static_w13_sf_storage is None
        )  # the 128-aligned scales already match the padded extent
        rel_d, views = _run(intermediate, None, DYNAMIC_TOKENS)
        assert rel_d < NOISE_FLOOR, f"dynamic I={intermediate} rel L2 {rel_d:.3f}"
        assert views.legacy_materialized and views.intermediate_size == n_aligned
        assert views.branch_major_w13_fp4.shape[0] == n_aligned

    @pytest.mark.parametrize("intermediate", [160, 288, 352])
    def test_aligned_multi_slice_shapes_keep_true_extent_views(self, intermediate):
        rel_s, views = _run(intermediate, "static", 64)
        assert rel_s < NOISE_FLOOR
        assert (
            views.static_intermediate_size == intermediate
            and views.static_w13_fp4.shape[0] == intermediate
        )
        assert not views.legacy_materialized and views.static_w13_sf_storage is None
        rel_d, views = _run(intermediate, None, DYNAMIC_TOKENS)
        assert rel_d < NOISE_FLOOR
        assert not views.legacy_materialized and views.intermediate_size == intermediate
        assert views.branch_major_w13_fp4.shape[0] == intermediate

    def test_functional_path_single_slice_static_is_accurate(self):
        from flashinfer import b12x_fused_moe
        from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

        t = _tensors(64, 128, seed=9)
        moe_dispatch._FORCED_BACKEND = "static"
        try:
            out = b12x_fused_moe(
                **_kwargs(t), num_experts=EXPERTS, top_k=TOPK, num_local_experts=EXPERTS
            )
            torch.cuda.synchronize()
        finally:
            moe_dispatch._FORCED_BACKEND = None
        ref = _reference(t, 64, 128)
        assert check_accuracy(out, ref)[0] and _rel_l2(out, ref) < NOISE_FLOOR


@pytest.mark.parametrize("num_tokens,topk", [(4, 10), (8, 5)])
def test_micro_cold_graph_scatter_consistency(num_tokens, topk, monkeypatch):
    """Cold FC2 loads must not expose another warp's unfinished epilogue stores."""
    if not _is_sm120():
        pytest.skip("SM12x required")
    from flashinfer import B12xMoEWrapper
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    E, H, I, capacity = 512, 4096, 128, 8192
    tensors = _tensors_shape(capacity, E, H, I, topk, seed=2026)
    point = {
        key: value[:num_tokens].contiguous()
        if key in ("x", "token_selected_experts", "token_final_scales")
        else value
        for key, value in tensors.items()
    }
    moe = B12xMoEWrapper(
        num_experts=E,
        hidden_size=H,
        intermediate_size=I,
        top_k=topk,
        use_cuda_graph=True,
        max_num_tokens=capacity,
    )
    selected = []
    get_micro = moe_dispatch._get_micro_kernel

    def record_micro(*args, **kwargs):
        selected.append(True)
        return get_micro(*args, **kwargs)

    monkeypatch.setattr(moe_dispatch, "_get_micro_kernel", record_micro)
    kwargs = _kwargs(point)
    for _ in range(5):
        moe.run(**kwargs)
    assert selected, "This regression must exercise MMA Micro, not Static/DirectMicro"
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = moe.run(**kwargs)
    ref = _reference_shape(point, num_tokens, E, H, I, topk)
    flush = torch.empty(192 * 1024 * 1024, dtype=torch.uint8, device="cuda")
    cold, hot = [], []
    for _ in range(256):
        for _ in range(16):
            flush.zero_()
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
        cold.append(output.clone())
        hot.append(moe.run(**kwargs).clone())
    cold, hot = torch.stack(cold).float(), torch.stack(hot).float()
    assert torch.isfinite(cold).all() and torch.isfinite(hot).all()
    differential = (cold - hot).flatten(1).norm(dim=1) / hot.flatten(1).norm(dim=1)
    assert differential.max().item() < 0.015, differential.max().item()
    for outputs in (cold, hot):
        errors = (outputs - ref).flatten(1).norm(dim=1) / ref.norm()
        assert errors.max().item() < NOISE_FLOOR, errors.max().item()
        assert check_accuracy(
            outputs.reshape(-1, H), ref.expand_as(outputs).reshape(-1, H)
        )[0]


@pytest.mark.skipif(not _is_sm120(), reason="SM120 kernels")
def test_slot_sized_scratch_is_rejected_before_launch():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch

    routed_rows = 3 * 32 * 8  # 768 routed rows -> capacity well above the slot stride
    ws = moe_dispatch.allocate_sm120_moe_workspace(
        state_E=EXPERTS,
        weight_E=EXPERTS,
        routed_rows=routed_rows,
        k=HIDDEN,
        n=320,
        num_topk=TOPK,
        device=torch.device("cuda"),
        quant_mode="nvfp4",
        backend="static",
        activation="silu",
    )
    kwargs = dict(
        state_E=EXPERTS,
        weight_E=EXPERTS,
        routed_rows=routed_rows,
        k=HIDDEN,
        n=384,
        num_topk=TOPK,
        device=torch.device("cuda"),
        activation_precision="fp4",
        quant_mode="nvfp4",
    )
    moe_dispatch._validate_static_workspace_for_launch(
        ws, **kwargs
    )  # the capacity-sized workspace passes
    groups = moe_dispatch._static_retained_groups(384)
    good = ws.route_output_scratch
    ws.route_output_scratch = torch.empty(
        (moe_dispatch._STATIC_SLOT_ROWS, groups, HIDDEN),
        dtype=good.dtype,
        device=good.device,
    )
    with pytest.raises(ValueError, match="route_output_scratch"):
        moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
    ws.route_output_scratch = good
    good_virt = ws.virt_route_scratch
    ws.virt_route_scratch = torch.zeros(
        (EXPERTS * (1 + 1) + 8,), dtype=good_virt.dtype, device=good_virt.device
    )
    with pytest.raises(ValueError, match="virt_route_scratch"):
        moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
    ws.virt_route_scratch = good_virt
    moe_dispatch._validate_static_workspace_for_launch(ws, **kwargs)
