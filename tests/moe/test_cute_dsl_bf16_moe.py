"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

End-to-end tests for the SM90 CuTe-DSL fused MoE:
moe_sort + GEMM1 (gather+SwiGLU) + GEMM2 (fused finalize).
"""

import pytest
import torch

from flashinfer.cute_dsl.utils import is_cute_dsl_available
from flashinfer.utils import get_compute_capability

cute_dsl_available = pytest.mark.skipif(
    not is_cute_dsl_available(), reason="Requires cute-dsl (nvidia-cutlass-dsl)"
)


def is_sm90():
    if not torch.cuda.is_available():
        return False
    return get_compute_capability(torch.device("cuda"))[0] == 9


sm90_required = pytest.mark.skipif(not is_sm90(), reason="Requires SM90 (Hopper) GPU")


@cute_dsl_available
def test_sm90_moe_autotune_profile_contract():
    """SM90 tuning must cycle the dynamic profiling inputs."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import (
        CuteDslFusedMoESm90Runner,
    )

    config = CuteDslFusedMoESm90Runner(
        forward_impl=_moe_core_impl,
        num_experts=4,
        top_k=2,
        num_local_experts=4,
    ).tuning_config

    assert config.value_aware_input_indices == (1, 2)
    assert config.profile_arena_input_indices == (0, 1, 2, 5)
    assert config.use_cuda_graph


@cute_dsl_available
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_sm90_moe_autotune_uses_dynamic_profile_arena(monkeypatch):
    """Large shared weights must not collapse cold-L2 profiling to one batch."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import (
        CuteDslFusedMoESm90Runner,
    )

    tuner = AutoTuner(warmup=0, repeat=4)
    monkeypatch.setattr(tuner, "_get_l2_cache_size_in_bytes", lambda: 1024)
    config = CuteDslFusedMoESm90Runner(
        forward_impl=_moe_core_impl,
        num_experts=4,
        top_k=2,
        num_local_experts=4,
    ).tuning_config
    inputs = [
        torch.arange(32, dtype=torch.float32, device="cuda").view(4, 8),
        torch.arange(8, dtype=torch.int32, device="cuda").view(4, 2),
        torch.full((4, 2), 0.5, dtype=torch.float32, device="cuda"),
        torch.empty(4096, dtype=torch.uint8, device="cuda"),
        torch.empty(4096, dtype=torch.uint8, device="cuda"),
        torch.empty((4, 8), dtype=torch.float32, device="cuda"),
    ]

    batches = tuner._prepare_input_tensors_with_batches(inputs, config)

    assert len(batches) == tuner.repeat
    for input_index in (0, 1, 2, 5):
        assert len({batch[input_index].data_ptr() for batch in batches}) == tuner.repeat
    for batch in batches:
        assert torch.equal(batch[1], inputs[1])
        assert torch.equal(batch[2], inputs[2])
        assert batch[3] is inputs[3]
        assert batch[4] is inputs[4]


@cute_dsl_available
def test_sm90_moe_persistent_cache_key_separates_runtime_modes():
    """Persisted winners must not alias dtype, PDL, or finalize modes."""
    from flashinfer.autotuner import AutoTuner
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import (
        CuteDslFusedMoESm90Runner,
    )

    def make_inputs(dtype):
        return [
            torch.empty((4, 64), dtype=dtype),
            torch.empty((4, 2), dtype=torch.int32),
            torch.empty((4, 2), dtype=torch.float32),
            torch.empty((4, 64, 64), dtype=dtype),
            torch.empty((4, 64, 32), dtype=dtype),
            torch.empty((4, 64), dtype=dtype),
        ]

    def file_key(runner, inputs):
        input_shapes = tuple(tuple(tensor.shape) for tensor in inputs)
        return AutoTuner._get_cache_key(
            "CuteDslFusedMoE::run_moe_sm90::Swiglu",
            runner,
            input_shapes,
            runner.tuning_config,
            runner.get_cache_key_extras(inputs),
        ).file_key

    default_runner = CuteDslFusedMoESm90Runner(_moe_core_impl, 4, 2, 4)
    bf16_inputs = make_inputs(torch.bfloat16)
    default_key = file_key(default_runner, bf16_inputs)

    assert default_key != file_key(default_runner, make_inputs(torch.float16))
    assert default_key != file_key(
        CuteDslFusedMoESm90Runner(_moe_core_impl, 4, 2, 4, enable_pdl=False),
        bf16_inputs,
    )
    assert default_key != file_key(
        CuteDslFusedMoESm90Runner(_moe_core_impl, 4, 2, 4, use_fused_finalize=False),
        bf16_inputs,
    )


def make_random_topk(num_experts, num_tokens, top_k, device="cuda"):
    ids = torch.stack(
        [torch.randperm(num_experts, device=device)[:top_k] for _ in range(num_tokens)]
    ).int()
    scores = torch.rand(num_tokens, top_k, device=device, dtype=torch.float32)
    scales = scores / scores.sum(dim=1, keepdim=True)  # norm_topk_prob
    return ids, scales


def ref_moe(x, ids, scales, w_gate_up, w2):
    """Float32 reference MoE forward."""
    num_tokens, hidden = x.shape
    inter = w_gate_up.shape[1] // 2
    xf = x.float()
    w_gate = w_gate_up[:, :inter].float()
    w_up = w_gate_up[:, inter:].float()
    w2f = w2.float()
    out = torch.zeros(num_tokens, hidden, device=x.device, dtype=torch.float32)
    for kk in range(ids.shape[1]):
        e_ids = ids[:, kk].long()
        s = scales[:, kk].unsqueeze(1)
        for e in torch.unique(e_ids).tolist():
            m = e_ids == e
            xe = xf[m]
            act = torch.nn.functional.silu(xe @ w_gate[e].T) * (xe @ w_up[e].T)
            out[m] += s[m] * (act @ w2f[e].T)
    return out


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "hidden,inter,tile_m,g1_tile_n,g2_tile_n",
    [
        (2048, 768, 128, 128, 128),  # Qwen3-30B-A3B tp=1
        (2048, 768, 128, 256, 256),  # tp=1, 2-WG tiles both GEMMs
        (2048, 768, 64, 128, 128),  # tile_m=64
        (2048, 192, 128, 128, 128),  # tp=4 per-rank (2I=384)
        (2048, 96, 128, 64, 128),  # tp=8 per-rank (2I=192, N tile 64)
    ],
)
@pytest.mark.parametrize("num_tokens", [3, 777])
def test_cute_dsl_fused_moe_bf16(
    hidden, inter, tile_m, g1_tile_n, g2_tile_n, num_tokens
):
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(11)
    dtype = torch.bfloat16
    num_experts, top_k = 128, 8

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    w1 = interleave_up_gate_sm90(w_gate_up)
    out = cute_dsl_fused_moe_bf16(
        x,
        ids,
        scales,
        w1,
        w2,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=tile_m,
        gemm1_tile_n=g1_tile_n,
        gemm2_tile_n=g2_tile_n,
    )

    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    # bf16 through two GEMMs + bf16 atomic accumulation of top_k=8 partials.
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_process_cache_reuse(monkeypatch):
    """Both GEMMs reuse their compiled specialization within one process."""
    import importlib

    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import (
        cute_dsl_fused_moe_bf16,
    )
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    gemm1_module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion"
    )
    gemm2_module = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.sm90_contiguous_grouped_gemm_finalize_fusion"
    )

    torch.manual_seed(53)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter, num_tokens = 4, 2, 128, 64, 3
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w1 = interleave_up_gate_sm90(w_gate_up)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    def run():
        return cute_dsl_fused_moe_bf16(
            x,
            ids,
            scales,
            w1,
            w2,
            num_experts=num_experts,
            top_k=top_k,
            use_fused_finalize=False,
            tile_size=64,
            gemm1_tile_n=64,
            gemm2_tile_n=64,
            gemm2_tile_k=64,
        )

    saved_gemm1 = dict(gemm1_module._gather_kernel_cache)
    saved_gemm2 = dict(gemm2_module._finalize_kernel_cache)
    gemm1_module._gather_kernel_cache.clear()
    gemm2_module._finalize_kernel_cache.clear()
    try:
        cold_out = run()
        ref = ref_moe(x, ids, scales, w_gate_up, w2)
        torch.testing.assert_close(cold_out.float(), ref, atol=3e-1, rtol=5e-2)
        assert len(gemm1_module._gather_kernel_cache) == 1
        assert len(gemm2_module._finalize_kernel_cache) == 1

        def fail_compile(*args, **kwargs):
            del args, kwargs
            raise AssertionError("warm process-cache hit called cute.compile")

        monkeypatch.setattr(gemm1_module.cute, "compile", fail_compile)
        warm_out = run()
        assert torch.equal(cold_out, warm_out)
    finally:
        gemm1_module._gather_kernel_cache.clear()
        gemm1_module._gather_kernel_cache.update(saved_gemm1)
        gemm2_module._finalize_kernel_cache.clear()
        gemm2_module._finalize_kernel_cache.update(saved_gemm2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [3, 777])
def test_cute_dsl_bf16_moe_fp16(num_tokens):
    """FP16 e2e: fp16 GEMM1 output + fp16 fused-finalize scatter-reduce
    (``cp.reduce...add.noftz.f16``)."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(13)
    dtype = torch.float16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    w1 = interleave_up_gate_sm90(w_gate_up)
    out = cute_dsl_fused_moe_bf16(
        x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k, tile_size=128
    )
    assert out.dtype == dtype
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [64, 2048, 4096])
def test_cute_dsl_bf16_moe_auto_select(num_tokens):
    """Untuned fallback selection stays correct across decode and prefill."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(17)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    w1 = interleave_up_gate_sm90(w_gate_up)
    out = cute_dsl_fused_moe_bf16(
        x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
    )
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("gemm2_cluster_shape_mn", [(1, 1), (1, 2)])
@pytest.mark.parametrize("raster_along_m", [True, False])
def test_cute_dsl_bf16_moe_gemm2_tactic_overrides(
    gemm2_cluster_shape_mn, raster_along_m
):
    """Every explicit GEMM2 cluster/raster combination remains correct."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(23)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter, num_tokens = 128, 8, 2048, 96, 777

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    w1 = interleave_up_gate_sm90(w_gate_up)
    out = cute_dsl_fused_moe_bf16(
        x,
        ids,
        scales,
        w1,
        w2,
        num_experts=num_experts,
        top_k=top_k,
        tile_size=128,
        gemm2_cluster_shape_mn=gemm2_cluster_shape_mn,
        gemm2_raster_along_m=raster_along_m,
    )
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [0, 1])
def test_cute_dsl_bf16_moe_tiny_batch(num_tokens):
    """Empty batch returns an empty output without launching; a single token
    routes through the full pipeline."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(19)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, max(num_tokens, 1), top_k)
    ids, scales = ids[:num_tokens], scales[:num_tokens]

    w1 = interleave_up_gate_sm90(w_gate_up)
    out = cute_dsl_fused_moe_bf16(
        x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
    )
    assert out.shape == (num_tokens, hidden)
    if num_tokens > 0:
        ref = ref_moe(x, ids, scales, w_gate_up, w2)
        torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [3, 777])
def test_cute_dsl_bf16_moe_deterministic(num_tokens):
    """Deterministic (non-fused finalize) mode: GEMM2 scatters unscaled rows
    in expanded order, moe_unpermute reduces in a fixed order — the result
    must match the reference AND be bitwise-reproducible across runs."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(31)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    w1 = interleave_up_gate_sm90(w_gate_up)

    def run():
        return cute_dsl_fused_moe_bf16(
            x,
            ids,
            scales,
            w1,
            w2,
            num_experts=num_experts,
            top_k=top_k,
            use_fused_finalize=False,
        )

    out = run()
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)
    # The deterministic contract: bitwise-identical across reruns.
    out2 = run()
    assert torch.equal(out, out2), "deterministic mode is not bitwise-stable"


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_pdl_off_matches():
    """PDL only overlaps kernel launches — with the deterministic finalize
    and a pinned tile config, enable_pdl=False and True must be bitwise
    identical (distinct compile-cache entries, same math)."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16

    torch.manual_seed(47)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768
    num_tokens = 333

    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w1 = torch.randn(num_experts, 2 * inter, hidden, device="cuda", dtype=dtype) / (
        hidden**0.25
    )
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)

    def run(enable_pdl):
        return cute_dsl_fused_moe_bf16(
            x,
            ids,
            scales,
            w1,
            w2,
            num_experts=num_experts,
            top_k=top_k,
            use_fused_finalize=False,
            tile_size=128,
            gemm1_tile_n=128,
            gemm2_tile_n=128,
            enable_pdl=enable_pdl,
        )

    assert torch.equal(run(True), run(False)), "PDL changed numerics"


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [64, 777])
def test_cute_dsl_bf16_moe_cuda_graph(num_tokens):
    """CUDA-graph capture/replay: moe_sort + both GEMMs + the aux-stream
    fork-join zeroing must be capturable (vLLM piecewise graphs and
    MoELayer's winner timing capture this region), and a replay with fresh
    routing/activations must produce correct results."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(23)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    w1 = interleave_up_gate_sm90(w_gate_up)

    # Static input/output buffers (graph replays read/write these in place).
    x_st = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    ids_st, scales_st = make_random_topk(num_experts, num_tokens, top_k)
    out_st = torch.empty(num_tokens, hidden, device="cuda", dtype=dtype)

    def run():
        return cute_dsl_fused_moe_bf16(
            x_st,
            ids_st,
            scales_st,
            w1,
            w2,
            num_experts=num_experts,
            top_k=top_k,
            moe_output=out_st,
        )

    # Warmup: JIT compiles and the aux stream/events. The intermediate buffer
    # is allocated per call (from the graph's memory pool during capture).
    for _ in range(3):
        run()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        run()

    # Replay with fresh activations AND fresh routing: moe_sort re-derives the
    # index maps on-device inside the graph.
    torch.manual_seed(29)
    x_new = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    ids_new, scales_new = make_random_topk(num_experts, num_tokens, top_k)
    x_st.copy_(x_new)
    ids_st.copy_(ids_new)
    scales_st.copy_(scales_new)
    g.replay()
    torch.cuda.synchronize()

    ref = ref_moe(x_new, ids_new, scales_new, w_gate_up, w2)
    torch.testing.assert_close(out_st.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
def test_cute_dsl_fused_moe_bf16_autotune():
    """AutoTuner-integrated dispatch (SM100 pattern): under autotune(True)
    every tactic is profiled and the winner cached; the tuned call and the
    subsequent cached-winner call must both match the reference."""
    from flashinfer.autotuner import autotune
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(43)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    x = torch.randn(512, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, 512, top_k)
    w1 = interleave_up_gate_sm90(w_gate_up)

    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    with autotune(True):
        out = cute_dsl_fused_moe_bf16(
            x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
        )
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)
    # Cached-winner dispatch (autotune off).
    out2 = cute_dsl_fused_moe_bf16(
        x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
    )
    torch.testing.assert_close(out2.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_wrapper():
    """CuteDslBf16MoEWrapper: config held on the instance, run matches the
    reference (family convention of CuteDslMoEWrapper / B12xMoEWrapper)."""
    from flashinfer.fused_moe import CuteDslBf16MoEWrapper
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(41)
    dtype = torch.bfloat16
    num_experts, top_k, hidden, inter = 128, 8, 2048, 768

    moe = CuteDslBf16MoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
    )

    x = torch.randn(777, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, 777, top_k)
    w1 = interleave_up_gate_sm90(w_gate_up)

    out = moe.run(x, ids, scales, w1, w2)
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("offset", [8, 24])
def test_cute_dsl_bf16_moe_ep_shard(offset):
    """Expert-parallel shard through the direct API: 8 local experts at a
    nonzero global offset. Tokens routed entirely outside the shard must
    yield exactly-zero rows (not garbage)."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(47)
    dtype = torch.bfloat16
    num_experts, n_local, top_k, hidden, inter = 32, 8, 4, 512, 384

    x = torch.randn(128, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w_gate_up = torch.randn(n_local, 2 * inter, hidden, device="cuda", dtype=dtype) / (
        hidden**0.25
    )
    w2 = torch.randn(n_local, hidden, inter, device="cuda", dtype=dtype) / (inter**0.25)
    ids, scales = make_random_topk(num_experts, 128, top_k)
    w1 = interleave_up_gate_sm90(w_gate_up)

    out = cute_dsl_fused_moe_bf16(
        x,
        ids,
        scales,
        w1,
        w2,
        num_experts=num_experts,
        top_k=top_k,
        num_local_experts=n_local,
        local_expert_offset=offset,
    )

    # Reference over the LOCAL shard only (global id g -> local g - offset).
    xf = x.float()
    ref = torch.zeros_like(xf)
    for local_e in range(n_local):
        mask = ids == local_e + offset
        if not mask.any():
            continue
        tok, nth = torch.where(mask)
        a = xf[tok] @ w_gate_up[local_e].float().t()
        act = torch.nn.functional.silu(a[:, :inter]) * a[:, inter:]
        ref[tok] += scales[tok, nth, None] * (act @ w2[local_e].float().t())
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)
    # Tokens with no expert in [offset, offset + n_local) stay exactly zero.
    outside = ((ids < offset) | (ids >= offset + n_local)).all(dim=1)
    if outside.any():
        assert (out[outside] == 0).all(), "non-shard tokens must stay zero"


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize(
    "inter,dispatch_sizes",
    [
        # I=768 (tp=1): 16 tokens -> tile_m 64; 2048 -> tile_m 128 + GEMM2
        # (1,1); 4096 -> tile_m 128 + GEMM2 (1,2) (tokens*topk >= 256*E).
        (768, (16, 2048, 4096)),
        # I=96 (tp=8, tiny reduction): 16 -> tile_m 64; 2048 -> tile_m 128
        # (early flip) + GEMM2 (1,1); 8192 -> M-raster (output at the 32 MiB
        # heuristic boundary).
        (96, (16, 2048, 8192)),
    ],
)
def test_cute_dsl_bf16_moe_autotune_covers_dispatch(inter, dispatch_sizes):
    """One autotune pass profiles every legal tuned cluster/raster tactic and
    the fallback tactic. Subsequent real calls reuse those process-local
    specializations."""
    import importlib

    from flashinfer.autotuner import autotune

    # importlib: the package re-exports a FUNCTION named like the module,
    # shadowing the module attribute for `from ... import` forms.
    g1_mod = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion"
    )
    g2_mod = importlib.import_module(
        "flashinfer.fused_moe.cute_dsl.sm90_contiguous_grouped_gemm_finalize_fusion"
    )
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import (
        cute_dsl_fused_moe_bf16,
    )

    torch.manual_seed(37)
    dtype = torch.bfloat16
    num_experts, top_k, hidden = 128, 8, 2048

    w_gate_up = torch.randn(
        num_experts, 2 * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    w1 = interleave_up_gate_sm90(w_gate_up)
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )

    # The tuning pass mirrors vLLM's engine-init flashinfer_autotune: one
    # max-token-ish batch profiles every tactic after its compile warmup.
    x = torch.randn(4096, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    ids, scales = make_random_topk(num_experts, 4096, top_k)
    with autotune(True):
        cute_dsl_fused_moe_bf16(
            x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
        )
    g1_keys = set(g1_mod._gather_kernel_cache)
    g2_keys = set(g2_mod._finalize_kernel_cache)
    assert g1_keys and g2_keys, "autotune pass compiled nothing"

    for num_tokens in dispatch_sizes:
        x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
        ids, scales = make_random_topk(num_experts, num_tokens, top_k)
        cute_dsl_fused_moe_bf16(
            x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k
        )
        assert set(g1_mod._gather_kernel_cache) == g1_keys, (
            f"GEMM1 compiled in-dispatch at num_tokens={num_tokens}: "
            f"{set(g1_mod._gather_kernel_cache) - g1_keys}"
        )
        assert set(g2_mod._finalize_kernel_cache) == g2_keys, (
            f"GEMM2 compiled in-dispatch at num_tokens={num_tokens}: "
            f"{set(g2_mod._finalize_kernel_cache) - g2_keys}"
        )


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_bad_inputs():
    """Malformed inputs raise clean ValueErrors before any kernel launch."""
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
        sm90_contiguous_gather_grouped_gemm_act_fusion,
    )

    t, k, e = 8, 256, 4
    x = torch.randn(t, k, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(e, 128, k, device="cuda", dtype=torch.bfloat16)

    def i32(n):
        return torch.zeros(n, device="cuda", dtype=torch.int32)

    args = (i32(1), i32(1), i32(128), i32(1))
    kw = dict(topk=1, permuted_m=128, tile_shape_mn=(128, 128))

    # up/gate interleave requires I % 32 == 0.
    with pytest.raises(ValueError, match="multiple of 32"):
        interleave_up_gate_sm90(torch.randn(e, 2 * 48, k, device="cuda"))
    # A/B dtype mismatch.
    with pytest.raises(ValueError, match="mismatched dtypes"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x, w1.half(), *args, **kw)
    # Unsupported dtype (fp32).
    with pytest.raises(ValueError, match="mismatched dtypes"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(
            x.float(), w1.float(), *args, **kw
        )
    # Non-contiguous A.
    with pytest.raises(ValueError, match="contiguous"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x.t(), w1, *args, **kw)
    # K mismatch between x and w1.
    with pytest.raises(ValueError, match="k mismatch"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(
            x[:, : k // 2].contiguous(), w1, *args, **kw
        )
    # 2I not a multiple of 64 (gated tile constraint).
    w1_bad_n = torch.randn(e, 96, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of 64"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x, w1_bad_n, *args, **kw)
    # permuted_m not a multiple of tile_m.
    with pytest.raises(ValueError, match="multiple of tile_m"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(
            x,
            w1,
            i32(1),
            i32(1),
            i32(100),
            i32(1),
            topk=1,
            permuted_m=100,
            tile_shape_mn=(128, 128),
        )
    # token_id_mapping size mismatch.
    with pytest.raises(ValueError, match="token_id_mapping"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(
            x,
            w1,
            i32(1),
            i32(1),
            i32(64),
            i32(1),
            topk=1,
            permuted_m=128,
            tile_shape_mn=(128, 128),
        )
    # N not a multiple of tile_n (partial N tile would write out of bounds).
    w1_192 = torch.randn(e, 192, k, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of tile_n"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x, w1_192, *args, **kw)
    # K not a multiple of the 64-element K tile.
    x_k = torch.randn(t, 96, device="cuda", dtype=torch.bfloat16)
    w1_k = torch.randn(e, 128, 96, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="multiple of the K tile"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x_k, w1_k, *args, **kw)


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_gemm2_bad_inputs():
    """GEMM2 wrapper rejects partial N tiles before any kernel launch."""
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_grouped_gemm_finalize_fusion import (
        sm90_contiguous_grouped_gemm_finalize_fusion,
    )

    num_tokens, topk, k, e = 16, 1, 64, 2
    a = torch.randn(128, k, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(e, 192, k, device="cuda", dtype=torch.bfloat16)
    scales = torch.rand(num_tokens, topk, device="cuda", dtype=torch.float32)
    out = torch.zeros(num_tokens, 192, device="cuda", dtype=torch.bfloat16)

    def i32(n):
        return torch.zeros(n, device="cuda", dtype=torch.int32)

    # n=192 is not a multiple of tile_n=128: the finalize scatter copies a
    # full tile_n-wide row, so a partial N tile would write out of bounds.
    with pytest.raises(ValueError, match="multiple of tile_n"):
        sm90_contiguous_grouped_gemm_finalize_fusion(
            a,
            w2,
            i32(1),
            i32(1),
            i32(128),
            i32(1),
            scales,
            out,
            topk=topk,
            tile_shape_mn=(128, 128),
            tile_k=64,
        )
