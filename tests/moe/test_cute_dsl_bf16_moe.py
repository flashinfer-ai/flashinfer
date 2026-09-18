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
moe_sort + GEMM1 (gather + fused activation) + GEMM2 (fused finalize).
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


def ref_activation(xe, w1_e, activation=None):
    """Float32 GEMM1 + activation of one expert. ``w1_e`` is the model's
    ``[gate; up]`` pack for gated activations, ``[I, hidden]`` for ReLU2;
    ``activation`` is a typed ``ActivationConfig`` (default SwiGLU)."""
    from tests.moe.utils import compute_reference_activation
    from flashinfer.fused_moe import SwiGLU

    activation = activation or SwiGLU()
    if activation.is_gated:
        inter = w1_e.shape[0] // 2
        values = torch.cat((xe @ w1_e[inter:].T, xe @ w1_e[:inter].T), dim=-1)
    else:
        inter = w1_e.shape[0]
        values = xe @ w1_e.T
    return compute_reference_activation(values, activation, inter).float()


def ref_moe(x, ids, scales, w_gate_up, w2, activation=None):
    """Float32 reference MoE forward (``activation``: typed
    ``ActivationConfig``; default SwiGLU)."""
    num_tokens, hidden = x.shape
    xf = x.float()
    w1f = w_gate_up.float()
    w2f = w2.float()
    out = torch.zeros(num_tokens, hidden, device=x.device, dtype=torch.float32)
    for kk in range(ids.shape[1]):
        e_ids = ids[:, kk].long()
        s = scales[:, kk].unsqueeze(1)
        for e in torch.unique(e_ids).tolist():
            m = e_ids == e
            out[m] += s[m] * (ref_activation(xf[m], w1f[e], activation) @ w2f[e].T)
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
        tactic=(
            tile_m,
            ((tile_m, g1_tile_n), 1),
            ((tile_m, g2_tile_n), (1, 1), False),
        ),
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
            tactic=(64, ((64, 64), 1), ((64, 64), (1, 1), False)),
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
        x,
        ids,
        scales,
        w1,
        w2,
        num_experts=num_experts,
        top_k=top_k,
        tactic=(128, ((128, 128), 1), ((128, 128), (1, 1), False)),
    )
    assert out.dtype == dtype
    ref = ref_moe(x, ids, scales, w_gate_up, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("num_tokens", [4096])
def test_cute_dsl_bf16_moe_auto_select(num_tokens):
    """The fixed default tactic stays correct at a prefill batch size."""
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
        tactic=(
            128,
            ((128, 64), 1),
            ((128, 128), gemm2_cluster_shape_mn, raster_along_m),
        ),
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
            tactic=(128, ((128, 128), 1), ((128, 128), (1, 1), False)),
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
        # I=768 (tp=1): decode (16 tokens), prefill (2048) and the region
        # where the tuner picks the GEMM2 (1,2) cluster (4096).
        (768, (16, 2048, 4096)),
        # I=96 (tp=8, tiny reduction): decode, prefill, and the M-major raster
        # region (8192 tokens, 32 MiB output).
        (96, (16, 2048, 8192)),
    ],
)
def test_cute_dsl_bf16_moe_autotune_covers_dispatch(inter, dispatch_sizes):
    """One autotune pass at the largest batch profiles every candidate tactic
    (tile sizes, gated swizzles and GEMM2 clusters) and the fallback tactic.
    Real calls at any token count up to that batch reuse those process-local
    specializations, including the fixed default."""
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

    # The tuning pass mirrors an engine's init-time autotune: one batch at the
    # maximum token count profiles every tactic of every bucket below it.
    max_tokens = max(dispatch_sizes)
    x = torch.randn(max_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    ids, scales = make_random_topk(num_experts, max_tokens, top_k)
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
    with pytest.raises(ValueError, match="cannot implement"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x, w1_bad_n, *args, **kw)
    # permuted_m not a multiple of tile_m.
    with pytest.raises(ValueError, match="cannot implement"):
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
    with pytest.raises(ValueError, match="cannot implement"):
        sm90_contiguous_gather_grouped_gemm_act_fusion(x, w1_192, *args, **kw)
    # K not a multiple of the 64-element K tile.
    x_k = torch.randn(t, 96, device="cuda", dtype=torch.bfloat16)
    w1_k = torch.randn(e, 128, 96, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="cannot implement"):
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
    with pytest.raises(ValueError, match="cannot implement"):
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
        )


@cute_dsl_available
@pytest.mark.parametrize(
    "tactic",
    [
        [64, 128, 1, 128, 64, [1, 1], False],
        [64, [128, 8], [128, [1, 2], False]],
        [64, [[64, 128], 8], [[64, 128], [1, 2]]],
        7,
        None,
    ],
)
def test_sm90_moe_rejects_malformed_tactic_structure(tactic):
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import _extract_tactic_params

    with pytest.raises(ValueError, match="tactic must be"):
        _extract_tactic_params(tactic)


@cute_dsl_available
@pytest.mark.parametrize(
    "tactic",
    [
        (128, ((64, 64), 1), ((128, 64), (1, 1), False)),
        (64, ((64, 128), 1), ((128, 128), (1, 1), False)),
    ],
)
def test_sm90_moe_rejects_mismatched_row_tiles(tactic):
    """Both GEMM row tiles must be moe_sort's tile: the per-tile expert and
    row-limit maps are written for that tile."""
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import _extract_tactic_params

    with pytest.raises(ValueError, match="M tiles must equal tile_size"):
        _extract_tactic_params(tactic)


def _moe_activation_cases():
    """Non-default activation configs exercised end to end."""
    from flashinfer.fused_moe import GeGLUTanh, ReLU2, SiTU, SwiGLU

    return [
        pytest.param(SwiGLU(alpha=1.702, beta=1.0, limit=7.0), id="swiglu_oai"),
        pytest.param(SiTU(gate_scale=4.0, linear_scale=25.0), id="situ"),
        pytest.param(GeGLUTanh(), id="geglu_tanh"),
        pytest.param(ReLU2(), id="relu2"),
    ]


def _make_activation_case(
    activation, hidden, inter, num_experts, num_tokens, top_k, seed
):
    """Inputs for one activation: the kernel's w1 (interleaved for gated
    activations, plain for ReLU2) plus the model-layout pack for the reference."""
    from flashinfer.fused_moe.cute_dsl.sm90_contiguous_gather_grouped_gemm_act_fusion import (
        interleave_up_gate_sm90,
    )

    torch.manual_seed(seed)
    dtype = torch.bfloat16
    gated = activation.is_gated
    x = torch.randn(num_tokens, hidden, device="cuda", dtype=dtype) / (hidden**0.25)
    w1_model = torch.randn(
        num_experts, (2 if gated else 1) * inter, hidden, device="cuda", dtype=dtype
    ) / (hidden**0.25)
    w1 = interleave_up_gate_sm90(w1_model) if gated else w1_model
    w2 = torch.randn(num_experts, hidden, inter, device="cuda", dtype=dtype) / (
        inter**0.25
    )
    ids, scales = make_random_topk(num_experts, num_tokens, top_k)
    return x, ids, scales, w1, w1_model, w2


@cute_dsl_available
@sm90_required
@pytest.mark.parametrize("activation", _moe_activation_cases())
@pytest.mark.parametrize("num_tokens", [5, 640])
def test_cute_dsl_bf16_moe_activation_types(activation, num_tokens):
    """Each fused GEMM1 activation matches the shared float32 reference end
    to end on the fixed default tactic."""
    from flashinfer.fused_moe.runners import _cute_dsl_activation_kwargs
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import cute_dsl_fused_moe_bf16

    hidden, inter, num_experts, top_k = 1024, 256, 32, 4
    x, ids, scales, w1, w1_model, w2 = _make_activation_case(
        activation, hidden, inter, num_experts, num_tokens, top_k, seed=21
    )
    out = cute_dsl_fused_moe_bf16(
        x,
        ids,
        scales,
        w1,
        w2,
        num_experts=num_experts,
        top_k=top_k,
        **_cute_dsl_activation_kwargs(activation),
    )
    ref = ref_moe(x, ids, scales, w1_model, w2, activation)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)


@cute_dsl_available
@sm90_required
def test_cute_dsl_bf16_moe_relu2_autotune_and_wrapper():
    """The non-gated activation goes through the tuner (GEMM1 N = I, no
    interleave) and the wrapper class carries the activation configuration."""
    from flashinfer.fused_moe.runners import _cute_dsl_activation_kwargs
    from flashinfer import autotune
    from flashinfer.fused_moe import ReLU2
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import (
        CuteDslBf16MoEWrapper,
        cute_dsl_fused_moe_bf16,
    )

    activation = ReLU2()
    act = _cute_dsl_activation_kwargs(activation)
    hidden, inter, num_experts, top_k, num_tokens = 1024, 192, 16, 2, 300
    x, ids, scales, w1, w1_model, w2 = _make_activation_case(
        activation, hidden, inter, num_experts, num_tokens, top_k, seed=22
    )
    assert w1.shape == (num_experts, inter, hidden)
    ref = ref_moe(x, ids, scales, w1_model, w2, activation)

    with autotune(True):
        tuned = cute_dsl_fused_moe_bf16(
            x, ids, scales, w1, w2, num_experts=num_experts, top_k=top_k, **act
        )
    torch.testing.assert_close(tuned.float(), ref, atol=3e-1, rtol=5e-2)

    moe = CuteDslBf16MoEWrapper(
        num_experts=num_experts,
        top_k=top_k,
        hidden_size=hidden,
        intermediate_size=inter,
        **act,
    )
    out = moe.run(x, ids, scales, w1, w2)
    torch.testing.assert_close(out.float(), ref, atol=3e-1, rtol=5e-2)

    # A gated-shaped w1 ([E, 2I, H]) with the non-gated activation makes GEMM1's
    # output 2I wide, which GEMM2's [E, H, I] weights cannot consume.
    w1_gated_shape = torch.cat((w1, w1), dim=1)
    with pytest.raises(ValueError, match="does not match"):
        cute_dsl_fused_moe_bf16(
            x,
            ids,
            scales,
            w1_gated_shape,
            w2,
            num_experts=num_experts,
            top_k=top_k,
            tactic=(128, ((128, 64), 1), ((128, 64), (1, 1), False)),
            **act,
        )


@cute_dsl_available
def test_sm90_moe_runner_rejects_bad_activation_config():
    """Unsupported activation types and inconsistent SiTU parameters fail at
    runner construction (before any kernel work)."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import CuteDslFusedMoESm90Runner
    from flashinfer.tllm_enums import ActivationType

    def make(**kwargs):
        return CuteDslFusedMoESm90Runner(_moe_core_impl, 4, 2, 4, **kwargs)

    for bad in (ActivationType.Geglu, ActivationType.Silu, ActivationType.Gelu):
        with pytest.raises(ValueError, match="Unsupported activation_type"):
            make(activation_type=bad.value)
    with pytest.raises(ValueError, match="requires situ_beta"):
        make(situ_linear_beta=25.0)
    with pytest.raises(ValueError, match="require ActivationType.Swiglu"):
        make(activation_type=ActivationType.GegluTanh.value, situ_beta=4.0)
    with pytest.raises(ValueError, match="positive and finite"):
        make(situ_beta=-1.0)
    # Positive finite in f64 but 0.0 / inf after fp32 rounding.
    for bad_scale in (1e-50, 1e39):
        with pytest.raises(ValueError, match="situ_beta must be positive"):
            make(situ_beta=bad_scale)
        with pytest.raises(ValueError, match="situ_linear_beta must be positive"):
            make(situ_beta=4.0, situ_linear_beta=bad_scale)
    assert make(activation_type=ActivationType.Relu2.value).gated is False
    assert make().gated is True


@cute_dsl_available
def test_sm90_moe_activation_config_separates_cache_keys():
    """Activation type and constants are part of the tuner cache key and the
    runner hash: a persisted winner never serves a different activation."""
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _moe_core_impl
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import CuteDslFusedMoESm90Runner
    from flashinfer.tllm_enums import ActivationType

    inputs = [torch.empty((4, 64), dtype=torch.bfloat16)]
    configs = [
        {},
        dict(activation_type=ActivationType.GegluTanh.value),
        dict(activation_type=ActivationType.Relu2.value),
        dict(swiglu_alpha=1.702, swiglu_beta=1.0, swiglu_limit=7.0),
        dict(swiglu_limit=7.0),
        dict(situ_beta=4.0),
        dict(situ_beta=4.0, situ_linear_beta=25.0),
    ]
    runners = [
        CuteDslFusedMoESm90Runner(_moe_core_impl, 4, 2, 4, **cfg) for cfg in configs
    ]
    keys = {r.get_cache_key_extras(inputs) for r in runners}
    hashes = {hash(r) for r in runners}
    assert len(keys) == len(configs)
    assert len(hashes) == len(configs)

    # The autotuner op name carries the activation family (SiTU named as such).
    from flashinfer.fused_moe.cute_dsl.sm90_fused_moe import _sm90_moe_autotune_op_name

    assert _sm90_moe_autotune_op_name(ActivationType.Swiglu.value) == (
        "CuteDslFusedMoE::run_moe_sm90::Swiglu"
    )
    assert _sm90_moe_autotune_op_name(ActivationType.GegluTanh.value).endswith(
        "::GegluTanh"
    )
    assert _sm90_moe_autotune_op_name(ActivationType.Relu2.value).endswith("::Relu2")
    assert _sm90_moe_autotune_op_name(
        ActivationType.Swiglu.value, situ_beta=4.0
    ).endswith("::Situ")


@cute_dsl_available
def test_sm90_moe_non_gated_tactic_legality():
    """GEMM1 walks I (not 2I) for Relu2, so a tactic legal for a gated shape
    can be illegal for the same intermediate size without gating."""
    from flashinfer.fused_moe.cute_dsl.sm90_tuner import (
        DEFAULT_SM90_MOE_TACTIC,
        is_valid_tactic,
    )

    common = dict(
        dtype=torch.bfloat16,
        num_tokens=64,
        hidden_size=1024,
        top_k=2,
        num_local_experts=8,
    )
    # I = 96: 2I = 192 tiles by 64 (gated), but 96 does not (non-gated).
    assert is_valid_tactic(DEFAULT_SM90_MOE_TACTIC, intermediate_size=96, **common)
    assert not is_valid_tactic(
        DEFAULT_SM90_MOE_TACTIC, intermediate_size=96, gated=False, **common
    )
    assert is_valid_tactic(
        DEFAULT_SM90_MOE_TACTIC, intermediate_size=128, gated=False, **common
    )
    wide = (128, ((128, 256), 1), ((128, 64), (1, 1), False))
    assert is_valid_tactic(wide, intermediate_size=128, **common)
    assert not is_valid_tactic(wide, intermediate_size=128, gated=False, **common)
    assert is_valid_tactic(wide, intermediate_size=256, gated=False, **common)
