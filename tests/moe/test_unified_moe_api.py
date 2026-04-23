"""Tests for the unified MoE API (MoELayer + Packs).

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

Two test groups:

  1. TestUnifiedMoEAccuracy — correctness vs a shared bf16 reference:
       - MoELayer end-to-end output matches bf16 reference (FP4 tolerance).
       - Each candidate backend individually matches the same reference.
         Shared reference (not cross-backend agreement) catches cases where
         both backends are wrong in the same way.

  2. TestUnifiedMoEDispatch — plumbing:
       - Autotuner profiles every candidate backend (shape-robust —
         doesn't commit to a specific winner).
       - CUDA graph capture + replay produces outputs identical to eager.

Scope: NVFP4, SM100/SM103, pre-routed path.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    ActivationConfig,
    CuteDslConfig,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    RoutingConfig,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.api import BackendOptions

# Reuse the canonical reference implementation + accuracy helpers from the
# existing CuteDSL test — keeps tolerance bounds consistent across tests.
from tests.moe.test_cute_dsl_fused_moe import (  # noqa: E402
    check_accuracy,
    compute_reference_moe_fp4,
    create_moe_tensors,
    is_sm100_family,
)


sm100_required = pytest.mark.skipif(
    not is_sm100_family(),
    reason="Unified NVFP4 MoE requires SM100 family (Blackwell SM100/SM103)",
)


# Small-scale geometry for fast accuracy + dispatch tests.
SMALL = dict(hidden_size=1024, intermediate_size=512, num_experts=32, top_k=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_trtllm_view(
    w1_bf16: torch.Tensor,
    w2_bf16: torch.Tensor,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    device: torch.device,
    permute_cache: dict,
) -> dict:
    """Quantize + shuffle weights into TRTLLM NVFP4 Shuffled_MajorK layout.

    Mirrors benchmarks/routines/moe.py::_build_trtllm_nvfp4_view.  Inlined here
    to avoid test→benchmark import coupling.
    """
    from flashinfer.fp4_quantization import fp4_quantize
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )
    from flashinfer.quantization.fp4_quantization import block_scale_interleave

    sf_vec_size = 16
    epilogue_tile_m = 128

    w1_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w1_flat = w1_bf16.view(num_local_experts * 2 * intermediate_size, hidden_size)
    w1_q_flat, w1_sf_flat = fp4_quantize(
        w1_flat,
        global_scale=w1_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    g1_w = w1_q_flat.view(
        num_local_experts, 2 * intermediate_size, hidden_size // 2
    ).view(torch.uint8)
    g1_s = w1_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size
    )

    w2_gs = torch.tensor([1.0], device=device, dtype=torch.float32)
    w2_flat = w2_bf16.view(num_local_experts * hidden_size, intermediate_size)
    w2_q_flat, w2_sf_flat = fp4_quantize(
        w2_flat,
        global_scale=w2_gs,
        sf_vec_size=sf_vec_size,
        is_sf_swizzled_layout=False,
    )
    g2_w = w2_q_flat.view(num_local_experts, hidden_size, intermediate_size // 2).view(
        torch.uint8
    )
    g2_s = w2_sf_flat.view(torch.float8_e4m3fn).reshape(
        num_local_experts, hidden_size, intermediate_size // sf_vec_size
    )

    g1_w_sh, g1_s_sh, g2_w_sh, g2_s_sh = [], [], [], []
    for i in range(num_local_experts):
        p = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache, g1_w[i], epilogue_tile_m, is_gated_act_gemm=True
        )
        g1_w_sh.append(g1_w[i][p.to(device)].contiguous())

        p_sf = _maybe_get_cached_w3_w1_permute_indices(
            permute_cache,
            g1_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
            is_gated_act_gemm=True,
        )
        g1_s_sh.append(
            block_scale_interleave(
                g1_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

        p = get_w2_permute_indices_with_cache(permute_cache, g2_w[i], epilogue_tile_m)
        g2_w_sh.append(g2_w[i][p.to(device)].contiguous())

        p_sf = get_w2_permute_indices_with_cache(
            permute_cache,
            g2_s[i].view(torch.uint8),
            epilogue_tile_m,
            num_elts_per_sf=16,
        )
        g2_s_sh.append(
            block_scale_interleave(
                g2_s[i].view(torch.uint8)[p_sf.to(device)].contiguous()
            )
        )

    ones = torch.ones(num_local_experts, device=device, dtype=torch.float32)
    return {
        "gemm1_weights": torch.stack(g1_w_sh),
        "gemm1_weights_scale": torch.stack(g1_s_sh)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, 2 * intermediate_size, hidden_size // sf_vec_size),
        "gemm1_alpha": ones,
        "gemm2_weights": torch.stack(g2_w_sh),
        "gemm2_weights_scale": torch.stack(g2_s_sh)
        .view(torch.float8_e4m3fn)
        .reshape(num_local_experts, hidden_size, intermediate_size // sf_vec_size),
        "output1_scale_scalar": ones,
        "output1_scale_gate_scalar": ones,
        "output2_scale_scalar": ones,
    }


# Shared permute cache across tests — shape-keyed, safe to reuse.
_TEST_PERMUTE_CACHE: dict = {}


def _make_packs_and_config(
    num_tokens: int,
    *,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    top_k: int,
    local_num_experts: int | None = None,
    max_tokens: int | None = None,
):
    """Build (act_pack, weight_pack, config, tensors_dict) for a given shape.

    ``tensors_dict`` contains the original bf16 reference weights used to
    compute ground truth via ``compute_reference_moe_fp4``.
    """
    local_num_experts = local_num_experts or num_experts
    max_tokens = max_tokens or max(num_tokens, 8192)
    device = torch.device("cuda", torch.cuda.current_device())

    # CuteDSL view comes pre-built by create_moe_tensors + bf16 refs
    tensors = create_moe_tensors(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_experts=num_experts,
        num_local_experts=local_num_experts,
        top_k=top_k,
    )

    act_pack = MoEActivationPack(
        hidden_states_q=tensors["x"],
        hidden_states_scale=tensors["x_sf"].squeeze(-1),
        selected_experts=tensors["token_selected_experts"],
        final_scales=tensors["token_final_scales"],
    )

    weight_pack = MoEWeightPack()
    weight_pack.prepare_for(
        "cute_dsl_nvfp4",
        {
            "w1_weight": tensors["w1_weight"],
            "w1_weight_sf": tensors["w1_weight_sf"],
            "w1_alpha": tensors["w1_alpha"],
            "fc2_input_scale": tensors["fc2_input_scale"],
            "w2_weight": tensors["w2_weight"],
            "w2_weight_sf": tensors["w2_weight_sf"],
            "w2_alpha": tensors["w2_alpha"],
        },
    )
    weight_pack.prepare_for(
        "trtllm_fp4_routed",
        _build_trtllm_view(
            tensors["w1_weight_bf16"],
            tensors["w2_weight_bf16"],
            local_num_experts,
            hidden_size,
            intermediate_size,
            device,
            _TEST_PERMUTE_CACHE,
        ),
    )

    config = MoEConfig(
        routing=RoutingConfig(num_experts=num_experts, top_k=top_k),
        quant=QuantConfig(variant=QuantVariant.NVFP4),
        experts=ExpertConfig(
            intermediate_size=intermediate_size,
            local_num_experts=local_num_experts,
        ),
        activation=ActivationConfig(),
        backend=BackendOptions(candidates=(CuteDslConfig(), TrtllmFp4Config())),
        execution=ExecutionConfig(tune_max_num_tokens=max_tokens),
    )
    return act_pack, weight_pack, config, tensors


# ---------------------------------------------------------------------------
# 1. Accuracy
# ---------------------------------------------------------------------------


def _compute_ref(act_pack, tensors, shape):
    """bf16 ground-truth MoE output for the given pack + shape."""
    return compute_reference_moe_fp4(
        hidden_states=tensors["x_bf16"].float().cuda(),
        gemm1_weights=tensors["w1_weight_bf16"].float().cuda(),
        gemm2_weights=tensors["w2_weight_bf16"].float().cuda(),
        token_selected_experts=act_pack.selected_experts,
        token_final_scales=act_pack.final_scales,
        num_tokens=act_pack.num_tokens,
        num_experts=shape["num_experts"],
        top_k=shape["top_k"],
        hidden_size=shape["hidden_size"],
        intermediate_size=shape["intermediate_size"],
        fc2_input_scale=tensors["fc2_input_scale"],
    )


@sm100_required
class TestUnifiedMoEAccuracy:
    """Every path compares against the same bf16 reference.

    Catches cases where both backends are wrong in the same way, which a
    cross-backend agreement test would miss.
    """

    @pytest.mark.parametrize("num_tokens", [128, 512])
    def test_layer_output_matches_reference(self, num_tokens):
        """MoELayer end-to-end output matches bf16 reference."""
        act_pack, weight_pack, config, tensors = _make_packs_and_config(
            num_tokens, **SMALL
        )

        with autotune(True):
            layer = MoELayer(config)
            out = layer(act_pack, weight_pack)

        ref = _compute_ref(act_pack, tensors, SMALL)
        passed, pct, atol = check_accuracy(out, ref)
        assert passed, (
            f"MoELayer output {pct * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}) vs bf16 reference at num_tokens={num_tokens}"
        )

    @pytest.mark.parametrize("backend_key", ["cute_dsl_nvfp4", "trtllm_fp4_routed"])
    def test_each_backend_matches_reference(self, backend_key):
        """Each candidate backend individually matches the same bf16 reference.

        If either backend's weight view were semantically wrong, its output
        would diverge from the shared reference — even in cases where it
        might agree with the other backend.
        """
        act_pack, weight_pack, config, tensors = _make_packs_and_config(256, **SMALL)
        layer = MoELayer(config)
        runner = next(r for r in layer.runners if r.backend_key == backend_key)

        inputs = runner.pack_inputs(act_pack, weight_pack)
        out = runner.forward(inputs, tactic=-1)

        ref = _compute_ref(act_pack, tensors, SMALL)
        passed, pct, atol = check_accuracy(out, ref)
        assert passed, (
            f"{backend_key}: {pct * 100:.2f}% within tolerance "
            f"(atol={atol:.4f}) vs bf16 reference"
        )


# ---------------------------------------------------------------------------
# 2. Dispatch plumbing
# ---------------------------------------------------------------------------


@sm100_required
class TestUnifiedMoEDispatch:
    """Plumbing tests — invariants MoELayer must guarantee."""

    def test_autotune_visits_all_candidate_backends(self):
        """The autotuner actually profiles every candidate backend.

        Shape-robust: doesn't commit to a specific winner (those change with
        kernel updates), just asserts each backend's `forward` was invoked
        during _select_winner.
        """
        act_pack, weight_pack, config, _ = _make_packs_and_config(256, **SMALL)
        layer = MoELayer(config)

        # Wrap each runner's forward to count invocations.
        call_counts: dict = {}
        for runner in layer.runners:
            key = runner.backend_key
            call_counts[key] = 0
            original = runner.forward

            def counted(*args, __key=key, __orig=original, **kwargs):
                call_counts[__key] += 1
                return __orig(*args, **kwargs)

            runner.forward = counted  # type: ignore[assignment]

        with autotune(True):
            _ = layer(act_pack, weight_pack)

        assert len(call_counts) >= 2, (
            f"Expected ≥2 candidate backends, got {list(call_counts)}"
        )
        for key, count in call_counts.items():
            assert count > 0, (
                f"Backend {key!r} was never invoked — autotuner skipped it "
                f"(call counts: {call_counts})"
            )

    def test_graph_capture_replay(self):
        """CUDA-graph-captured replay matches eager output."""
        num_tokens = 256
        act_pack, weight_pack, config, _ = _make_packs_and_config(
            num_tokens, max_tokens=num_tokens, **SMALL
        )

        # Warm up: populate autotune cache + stabilize allocator
        with autotune(True):
            layer = MoELayer(config)
            for _ in range(3):
                _ = layer(act_pack, weight_pack)
        for _ in range(3):
            _ = layer(act_pack, weight_pack)

        eager_out = layer(act_pack, weight_pack).clone()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            captured = layer(act_pack, weight_pack)

        for _ in range(10):
            g.replay()
        torch.cuda.synchronize()

        passed, pct, atol = check_accuracy(captured, eager_out)
        assert passed, (
            f"Graph replay diverged from eager: {pct * 100:.2f}% within "
            f"tolerance (atol={atol:.4f})"
        )
