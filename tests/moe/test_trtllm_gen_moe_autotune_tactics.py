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
"""

import time
from typing import Literal

import pytest
import torch

from flashinfer import (
    RoutingMethodType,
    ActivationType,
    fp4_quantize,
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)
from flashinfer.autotuner import AutoTuner
from flashinfer.fused_moe import (
    trtllm_fp4_block_scale_routed_moe,
    trtllm_fp8_block_scale_routed_moe,
    WeightLayout,
)
from flashinfer.fused_moe.core import Fp8QuantizationType, MoEInputs
from flashinfer.jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module
from flashinfer.tllm_enums import DtypeTrtllmGen
from flashinfer.utils import device_support_pdl, get_compute_capability

from .test_trtllm_gen_fused_moe import (
    FP8BlockScaleMoe,
    QuantMode,
    routing_reference_renormalize,
    routing_reference_renormalize_naive,
    routing_reference_topk,
)


Fp4QuantMode = Literal["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"]
Fp8QuantMode = Literal["DeepSeekFp8", "MxFp8"]


def _last_positive_power_of_2(n: int) -> int:
    n = max(int(n), 1)
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _moe_profile_shapes(
    inputs: dict,
    num_tokens: int,
    bucket_m: int,
    scale_dim: int = 0,
) -> tuple:
    def _bucket(t: torch.Tensor | None, dim: int = 0) -> tuple:
        if t is None:
            return (0,)
        s = list(t.shape)
        s[dim] = bucket_m
        return tuple(s)

    hidden_size = inputs["hidden_size"]
    by_name = {
        "output": (bucket_m, hidden_size),
        "routing_logits": (0,),
        "topk_ids": _bucket(inputs["packed_topk"]),
        "expert_weights": _bucket(inputs["expert_weights"]),
        "hidden_states": _bucket(inputs["hidden_states"]),
        "hidden_states_scale": _bucket(inputs["hidden_states_scale"], dim=scale_dim),
        "per_token_scale": (0,),
    }
    return tuple(by_name[name] for name in MoEInputs._FIELDS)


_TEST_RUNNER = "MoERunner"
_TEST_OP_FP4 = "flashinfer::trtllm_fp4_block_scale_moe"
_TEST_OP_FP8 = "flashinfer::trtllm_fp8_block_scale_moe"
_TEST_LOG_KEY_FP4 = (_TEST_OP_FP4, _TEST_RUNNER)
_TEST_LOG_KEY_FP8 = (_TEST_OP_FP8, _TEST_RUNNER)


def _force_tactic_in_autotuner_cache(
    profile_shapes: tuple,
    tactic: list[int] | None,
    custom_op: str,
) -> None:
    file_key = str((custom_op, _TEST_RUNNER, profile_shapes))
    tuner = AutoTuner.get()
    tuner.profiling_cache.clear()
    tuner._file_configs.clear()
    if tactic is not None:
        tuner._file_configs[file_key] = (_TEST_RUNNER, list(tactic))


def _check_tactic(
    call_fn,
    tactic: list[int],
    reference: torch.Tensor,
    ref_max: float,
    n_iters: int,
) -> str | None:
    """Run a tactic ``n_iters`` times and return a one-line failure summary
    if any of the correctness invariants is violated; ``None`` on success."""
    worst_abs = 0.0
    n_nan_iters = n_inf_iters = n_huge_iters = n_nondet_iters = 0
    first_out: torch.Tensor | None = None
    inf_t = torch.tensor(float("inf"), device=reference.device)
    for it in range(n_iters):
        out = call_fn(tactic).float()
        if it == 0:
            first_out = out.clone()
        if torch.isnan(out).any():
            n_nan_iters += 1
        if torch.isinf(out).any():
            n_inf_iters += 1
        # max |out - ref| with NaN/Inf in either operand promoted to +inf.
        d = (out - reference).abs()
        d_clean = torch.where(torch.isnan(d) | torch.isinf(d), inf_t, d)
        d_max = d_clean.max().item()
        if d_max > worst_abs:
            worst_abs = d_max
        if d_max > 0.5 * max(ref_max, 1.0):
            n_huge_iters += 1
        if it > 0 and not torch.equal(out, first_out):
            n_nondet_iters += 1
    if max(n_nan_iters, n_inf_iters, n_huge_iters, n_nondet_iters) == 0:
        return None
    worst_rel = worst_abs / max(ref_max, 1e-12)
    return (
        f"tactic={list(tactic)}  "
        f"NaN={n_nan_iters}/{n_iters} Inf={n_inf_iters}/{n_iters} "
        f"huge={n_huge_iters}/{n_iters} nondet={n_nondet_iters}/{n_iters}  "
        f"worst_abs={worst_abs:.4g} worst_rel={worst_rel:.4g}"
    )


# ----------------------------------------------------------------------------
# FP4 routed MoE sweep.
# ----------------------------------------------------------------------------


def _quant_mode_config(quant_mode: Fp4QuantMode):
    if quant_mode == "NvFP4xNvFP4":
        return dict(
            sf_vec_size=16,
            sf_use_ue8m0=False,
            global_sf=1.0 / 448.0 / 6.0,
            dtype_act=DtypeTrtllmGen.E2m1,
            dtype_weights=DtypeTrtllmGen.E2m1,
        )
    if quant_mode == "MxFP4xMxFP8":
        return dict(
            sf_vec_size=32,
            sf_use_ue8m0=True,
            global_sf=1.0,
            dtype_act=DtypeTrtllmGen.MxE4m3,
            dtype_weights=DtypeTrtllmGen.MxE2m1,
        )
    if quant_mode == "MxFP4xBf16":
        return dict(
            sf_vec_size=32,
            sf_use_ue8m0=True,
            global_sf=1.0,
            dtype_act=DtypeTrtllmGen.Bfloat16,
            dtype_weights=DtypeTrtllmGen.MxE2m1,
        )
    raise ValueError(f"unknown quant_mode {quant_mode!r}")


def _build_fp4_routed_moe_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    quant_mode: Fp4QuantMode,
    routing_method_type: RoutingMethodType,
    device: torch.device,
) -> dict:
    """Build kernel-ready inputs for `trtllm_fp4_block_scale_routed_moe`."""
    cfg = _quant_mode_config(quant_mode)
    sf_vec = cfg["sf_vec_size"]
    use_ue8m0 = cfg["sf_use_ue8m0"]
    # Activation global SF inside the kernel; nvfp4 uses 1/(448*6), mxfp4 uses 1.
    global_sf_a = global_sf_w = cfg["global_sf"]

    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )
    hidden_states_bf16 = (
        torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
    )

    # Hidden-state quantization. mxfp4 + bf16-activation skips the quantize
    # and passes raw bf16 through; the other two modes quantize to fp4/fp8.
    if quant_mode == "NvFP4xNvFP4":
        hidden_states, hidden_states_scale = fp4_quantize(
            hidden_states_bf16,
            torch.tensor([1.0 / global_sf_a], device=device),
            sf_vec_size=sf_vec,
            sf_use_ue8m0=use_ue8m0,
            is_sf_swizzled_layout=False,
        )
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
    elif quant_mode == "MxFP4xMxFP8":
        hidden_states, hidden_states_scale = mxfp8_quantize(hidden_states_bf16, False)
        hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
            num_tokens, -1
        )
    else:  # MxFP4xBf16
        hidden_states = hidden_states_bf16
        hidden_states_scale = None

    # Weights are always fp4-quantized; the SF flavor (E4M3 vs UE8M0) and
    # the global scale factor differ between nvfp4 and mxfp4.
    def _quant_weight(
        t: torch.Tensor, last_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q, sf = fp4_quantize(
            t,
            torch.tensor([1.0 / global_sf_w], device=device),
            sf_vec_size=sf_vec,
            sf_use_ue8m0=use_ue8m0,
        )
        sf = sf.view(torch.float8_e4m3fn).reshape(num_experts, last_dim, -1)
        return q, sf

    w13_bf16 = (
        torch.randn(num_experts, intermediate_size * 2, hidden_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w2_bf16 = (
        torch.randn(num_experts, hidden_size, intermediate_size, device=device).to(
            torch.bfloat16
        )
        * 0.1
    )
    w13, w13_scale = _quant_weight(w13_bf16, intermediate_size * 2)
    w2, w2_scale = _quant_weight(w2_bf16, hidden_size)

    # Per-expert output dequantization scalars.
    output1_scale = torch.tensor(
        [global_sf_a * global_sf_w] * num_experts, device=device
    )
    output2_scale = torch.tensor(
        [global_sf_a * global_sf_w] * num_experts, device=device
    )

    # Routing: build packed (expert_id<<16 | bf16_weight) tensor expected by
    # the routed kernel. Reuse the existing python reference routing helpers.
    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    else:
        permute_info, expert_weights = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    packed_topk = (topk_ids.to(torch.int32) << 16) | expert_weights.to(
        torch.bfloat16
    ).view(torch.int16)

    return dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        w13=w13,
        w13_scale=w13_scale,
        w2=w2,
        w2_scale=w2_scale,
        output1_scale_scalar=output1_scale,
        output1_scale_gate_scalar=output1_scale,
        output2_scale_scalar=output2_scale,
        packed_topk=packed_topk,
        expert_weights=expert_weights,
        hidden_size=hidden_size,
    )


def _enumerate_valid_tactics(
    moe_op,
    quant_mode: Fp4QuantMode,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_tokens: int,
) -> list[list[int]]:
    """Enumerate every (tile_N, config) tactic the autotuner may select for
    the given problem shape."""
    from flashinfer.tllm_enums import Fp8QuantizationType

    cfg = _quant_mode_config(quant_mode)
    return list(
        moe_op.trtllm_get_valid_moe_configs(
            cfg["dtype_act"],
            cfg["dtype_weights"],
            Fp8QuantizationType.NoneFp8,
            top_k,
            hidden_size,
            intermediate_size,
            num_experts,  # num_local_experts
            ActivationType.Swiglu.value,
            True,  # use_shuffled_weight
            WeightLayout.MajorK.value,
            False,  # use_per_token_scaling
            num_tokens,
        )
    )


@pytest.mark.parametrize("quant_mode", ["NvFP4xNvFP4", "MxFP4xMxFP8", "MxFP4xBf16"])
@pytest.mark.parametrize("num_tokens", [16, 23, 128])
@pytest.mark.parametrize("hidden_size", [4096, 7168])
@pytest.mark.parametrize("intermediate_size", [3072])
@pytest.mark.parametrize("num_experts", [128, 384])
@pytest.mark.parametrize("top_k", [4, 6])
def test_trtllm_fp4_routed_moe_all_tactics_correctness(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    quant_mode: Fp4QuantMode,
):
    """Per-tactic correctness sweep of `trtllm_fp4_block_scale_routed_moe`.

    Forces every valid (tile_N, config) tactic into the autotuner cache,
    runs the kernel ``N_ITERS_PER_TACTIC`` times for each tactic with
    identical inputs, and asserts no NaN/Inf, magnitude bound, bitwise
    determinism, and approximate match to the heuristic-default tactic's
    output.
    """
    if get_compute_capability(torch.device(device="cuda"))[0] not in [10]:
        pytest.skip("Only work on SM100 / SM103.")

    AutoTuner.get()._logged_file_hits.discard(_TEST_LOG_KEY_FP4)

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_method_type = RoutingMethodType.Renormalize

    inputs = _build_fp4_routed_moe_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_experts=num_experts,
        quant_mode=quant_mode,
        routing_method_type=routing_method_type,
        device=device,
    )
    # Pin the autotuner bucket so cache-write and runtime cache-lookup match.
    tune_max_num_tokens = max(_last_positive_power_of_2(num_tokens), 16)
    bucket_m = min(_last_positive_power_of_2(num_tokens), tune_max_num_tokens)
    profile_shapes = _moe_profile_shapes(inputs, num_tokens, bucket_m)

    def _run_kernel_with_tactic(tactic: list[int] | None) -> torch.Tensor:
        _force_tactic_in_autotuner_cache(profile_shapes, tactic, custom_op=_TEST_OP_FP4)
        out = trtllm_fp4_block_scale_routed_moe(
            topk_ids=inputs["packed_topk"],
            routing_bias=None,
            hidden_states=inputs["hidden_states"],
            hidden_states_scale=inputs["hidden_states_scale"],
            gemm1_weights=inputs["w13"],
            gemm1_weights_scale=inputs["w13_scale"],
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=None,
            gemm2_weights=inputs["w2"],
            gemm2_weights_scale=inputs["w2_scale"],
            gemm2_bias=None,
            output1_scale_scalar=inputs["output1_scale_scalar"],
            output1_scale_gate_scalar=inputs["output1_scale_gate_scalar"],
            output2_scale_scalar=inputs["output2_scale_scalar"],
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type.value,
            do_finalize=True,
            enable_pdl=enable_pdl,
            activation_type=ActivationType.Swiglu.value,
            tune_max_num_tokens=tune_max_num_tokens,
        )[0]
        torch.cuda.synchronize()
        return out

    # Heuristic-default tactic acts as reference.
    reference = _run_kernel_with_tactic(None).float()
    ref_max = reference.abs().max().item()
    assert torch.isfinite(reference).all(), (
        f"[{quant_mode}] reference output is not finite — bad test setup"
    )

    moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
    valid_tactics = _enumerate_valid_tactics(
        moe_op,
        quant_mode,
        top_k,
        hidden_size,
        intermediate_size,
        num_experts,
        num_tokens,
    )
    assert len(valid_tactics) > 0, f"[{quant_mode}] no valid tactics returned"

    _run_kernel_with_tactic(list(valid_tactics[0]))
    assert _TEST_LOG_KEY_FP4 in AutoTuner.get()._logged_file_hits, (
        f"[{quant_mode}] forced tactic was not dispatched — autotuner did "
        f"not log a cache hit; check `_moe_profile_shapes` against the "
        f"actual MoEInputs layout."
    )
    print(
        f"\n[all_tactics_correctness] quant={quant_mode} "
        f"M={num_tokens} hidden={hidden_size} intermediate={intermediate_size} "
        f"experts={num_experts} top_k={top_k}: "
        f"sweeping {len(valid_tactics)} valid tactics",
        flush=True,
    )

    N_ITERS_PER_TACTIC = 10
    failures: list[str] = []
    t0 = time.time()
    for tactic in valid_tactics:
        line = _check_tactic(
            _run_kernel_with_tactic,
            list(tactic),
            reference,
            ref_max,
            N_ITERS_PER_TACTIC,
        )
        if line is not None:
            failures.append(line)
            print(f"  BAD: {line}", flush=True)
    print(
        f"  done in {time.time() - t0:.0f}s, "
        f"{len(failures)}/{len(valid_tactics)} tactics failed",
        flush=True,
    )
    if failures:
        raise AssertionError(
            f"[{quant_mode}] {len(failures)} of {len(valid_tactics)} tactics "
            f"failed correctness. First few:\n  " + "\n  ".join(failures[:20])
        )


# ----------------------------------------------------------------------------
# FP8 routed MoE sweep.
# ----------------------------------------------------------------------------


def _fp8_quant_mode_config(quant_mode: Fp8QuantMode) -> dict:
    if quant_mode == "DeepSeekFp8":
        return dict(
            scale_dim=1,
            dtype_act=DtypeTrtllmGen.E4m3,
            dtype_weights=DtypeTrtllmGen.E4m3,
            fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            use_shuffled_weight=False,
            weight_layout=WeightLayout.MajorK.value,
        )
    if quant_mode == "MxFp8":
        return dict(
            scale_dim=0,
            dtype_act=DtypeTrtllmGen.MxE4m3,
            dtype_weights=DtypeTrtllmGen.MxE4m3,
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            use_shuffled_weight=True,
            weight_layout=WeightLayout.MajorK.value,
        )
    raise ValueError(f"unknown fp8 quant_mode {quant_mode!r}")


def _build_fp8_routed_moe_inputs(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    quant_mode: Fp8QuantMode,
    routing_method_type: RoutingMethodType,
    device: torch.device,
) -> dict:
    cfg = _fp8_quant_mode_config(quant_mode)

    routing_logits = torch.rand(num_tokens, num_experts, device=device).to(
        torch.bfloat16
    )

    if quant_mode == "DeepSeekFp8":
        hidden_states_bf16 = (
            torch.randn(num_tokens, hidden_size, device=device).to(torch.bfloat16) * 0.1
        )
        hidden_states = hidden_states_bf16.to(torch.float8_e4m3fn)
        hidden_states_scale = torch.ones(
            hidden_size // 128, num_tokens, device=device, dtype=torch.float32
        )
        gemm1_weights = torch.randn(
            num_experts, 2 * intermediate_size, hidden_size, device=device
        ).to(torch.float8_e4m3fn)
        gemm2_weights = torch.randn(
            num_experts, hidden_size, intermediate_size, device=device
        ).to(torch.float8_e4m3fn)
        gemm1_weights_scale = torch.ones(
            num_experts,
            2 * intermediate_size // 128,
            hidden_size // 128,
            device=device,
            dtype=torch.float32,
        )
        gemm2_weights_scale = torch.ones(
            num_experts,
            hidden_size // 128,
            intermediate_size // 128,
            device=device,
            dtype=torch.float32,
        )
    else:  # MxFp8 — full quantize + shuffle pipeline
        hidden_states_bf16 = torch.randn(num_tokens, hidden_size, device=device).to(
            torch.bfloat16
        )
        gemm1_weights_bf16 = torch.randn(
            num_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
        gemm2_weights_bf16 = torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
        quant_impl = FP8BlockScaleMoe(
            fp8_quantization_type=QuantMode.FP8_BLOCK_SCALE_MXFP8
        )
        quant_w = quant_impl.quantize_weights(
            gemm1_weights_bf16, gemm2_weights_bf16, hidden_states_bf16
        )
        quant_in = quant_impl.quantize_inputs(hidden_states_bf16)
        hidden_states = quant_in["hidden_states"]
        hidden_states_scale = quant_in["hidden_states_scale"]

        epilogue_tile_m = 128
        w13_rows = 2 * intermediate_size
        g1_w, g1_s, g2_w, g2_s = [], [], [], []
        for i in range(num_experts):
            w1 = quant_w["gemm1_weights"][i].clone().reshape(w13_rows, -1)
            s1 = quant_w["gemm1_scales"][i].clone().reshape(w13_rows, -1)
            # Swiglu is gated → reorder rows for gated-act gemm.
            w1 = reorder_rows_for_gated_act_gemm(w1)
            s1 = reorder_rows_for_gated_act_gemm(s1)
            g1_w.append(
                shuffle_matrix_a(w1.view(torch.uint8), epilogue_tile_m)
                .contiguous()
                .view(quant_w["gemm1_weights"].dtype)
            )
            g1_s.append(
                shuffle_matrix_sf_a(
                    s1.view(torch.uint8).reshape(w13_rows, -1), epilogue_tile_m
                )
                .contiguous()
                .view(quant_w["gemm1_scales"].dtype)
            )
            g2_w.append(
                shuffle_matrix_a(
                    quant_w["gemm2_weights"][i].view(torch.uint8), epilogue_tile_m
                )
                .contiguous()
                .view(quant_w["gemm2_weights"].dtype)
            )
            g2_s.append(
                shuffle_matrix_sf_a(
                    quant_w["gemm2_scales"][i]
                    .view(torch.uint8)
                    .reshape(hidden_size, -1),
                    epilogue_tile_m,
                )
                .contiguous()
                .view(quant_w["gemm2_scales"].dtype)
            )
        gemm1_weights = torch.stack(g1_w)
        gemm1_weights_scale = torch.stack(g1_s)
        gemm2_weights = torch.stack(g2_w)
        gemm2_weights_scale = torch.stack(g2_s)

    if routing_method_type == RoutingMethodType.Renormalize:
        permute_info, expert_weights_full = routing_reference_renormalize(
            routing_logits, top_k, num_experts, 8
        )
    elif routing_method_type == RoutingMethodType.RenormalizeNaive:
        permute_info, expert_weights_full = routing_reference_renormalize_naive(
            routing_logits, top_k, num_experts, 8
        )
    else:
        permute_info, expert_weights_full = routing_reference_topk(
            routing_logits, top_k, num_experts, 8
        )
    topk_ids = permute_info["topKIndices"].to(torch.int32)
    expert_weights = expert_weights_full.view(num_tokens, num_experts)[
        torch.arange(num_tokens, device=device).unsqueeze(1), topk_ids
    ].to(torch.bfloat16)
    packed_topk = (topk_ids.to(torch.int32) << 16) | expert_weights.view(
        torch.int16
    ).to(torch.int32)

    expert_weights_placeholder = torch.empty(0, dtype=torch.bfloat16, device=device)

    return dict(
        hidden_states=hidden_states,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=gemm1_weights,
        gemm1_weights_scale=gemm1_weights_scale,
        gemm2_weights=gemm2_weights,
        gemm2_weights_scale=gemm2_weights_scale,
        packed_topk=packed_topk,
        expert_weights=expert_weights_placeholder,
        hidden_size=hidden_size,
        use_shuffled_weight=cfg["use_shuffled_weight"],
        weight_layout=cfg["weight_layout"],
        fp8_quantization_type=cfg["fp8_quantization_type"],
    )


def _enumerate_fp8_valid_tactics(
    moe_op,
    quant_mode: Fp8QuantMode,
    top_k: int,
    hidden_size: int,
    intermediate_size: int,
    num_experts: int,
    num_tokens: int,
) -> list[list[int]]:
    """Enumerate every (tile_N, config) tactic the autotuner may select for
    the given FP8 problem shape."""
    cfg = _fp8_quant_mode_config(quant_mode)
    return list(
        moe_op.trtllm_get_valid_moe_configs(
            cfg["dtype_act"],
            cfg["dtype_weights"],
            cfg["fp8_quantization_type"],
            top_k,
            hidden_size,
            intermediate_size,
            num_experts,  # num_local_experts
            ActivationType.Swiglu.value,
            cfg["use_shuffled_weight"],
            cfg["weight_layout"],
            False,  # use_per_token_scaling
            num_tokens,
        )
    )


@pytest.mark.parametrize("quant_mode", ["DeepSeekFp8", "MxFp8"])
@pytest.mark.parametrize("num_tokens", [16, 23, 128])
@pytest.mark.parametrize("hidden_size", [4096])
@pytest.mark.parametrize("intermediate_size", [3072])
@pytest.mark.parametrize("num_experts", [128])
@pytest.mark.parametrize("top_k", [4])
def test_trtllm_fp8_routed_moe_all_tactics_correctness(
    num_tokens: int,
    hidden_size: int,
    intermediate_size: int,
    top_k: int,
    num_experts: int,
    quant_mode: Fp8QuantMode,
):
    """Per-tactic correctness sweep of `trtllm_fp8_block_scale_routed_moe`."""
    if get_compute_capability(torch.device(device="cuda"))[0] not in [10]:
        pytest.skip("Only work on SM100 / SM103.")

    AutoTuner.get()._logged_file_hits.discard(_TEST_LOG_KEY_FP8)

    torch.manual_seed(42)
    device = torch.device("cuda:0")
    enable_pdl = device_support_pdl(device)
    routing_method_type = RoutingMethodType.Renormalize

    cfg = _fp8_quant_mode_config(quant_mode)
    inputs = _build_fp8_routed_moe_inputs(
        num_tokens=num_tokens,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        top_k=top_k,
        num_experts=num_experts,
        quant_mode=quant_mode,
        routing_method_type=routing_method_type,
        device=device,
    )
    tune_max_num_tokens = max(_last_positive_power_of_2(num_tokens), 16)
    bucket_m = min(_last_positive_power_of_2(num_tokens), tune_max_num_tokens)
    profile_shapes = _moe_profile_shapes(
        inputs, num_tokens, bucket_m, scale_dim=cfg["scale_dim"]
    )

    def _run_kernel_with_tactic(tactic: list[int] | None) -> torch.Tensor:
        _force_tactic_in_autotuner_cache(profile_shapes, tactic, custom_op=_TEST_OP_FP8)
        out = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)
        trtllm_fp8_block_scale_routed_moe(
            topk_ids=inputs["packed_topk"],
            routing_bias=None,
            hidden_states=inputs["hidden_states"],
            hidden_states_scale=inputs["hidden_states_scale"],
            gemm1_weights=inputs["gemm1_weights"],
            gemm1_weights_scale=inputs["gemm1_weights_scale"],
            gemm2_weights=inputs["gemm2_weights"],
            gemm2_weights_scale=inputs["gemm2_weights_scale"],
            num_experts=num_experts,
            top_k=top_k,
            n_group=None,
            topk_group=None,
            intermediate_size=intermediate_size,
            local_expert_offset=0,
            local_num_experts=num_experts,
            routed_scaling_factor=None,
            routing_method_type=routing_method_type.value,
            use_shuffled_weight=inputs["use_shuffled_weight"],
            weight_layout=inputs["weight_layout"],
            do_finalize=True,
            enable_pdl=enable_pdl,
            output=out,
            tune_max_num_tokens=tune_max_num_tokens,
            fp8_quantization_type=inputs["fp8_quantization_type"],
            activation_type=ActivationType.Swiglu.value,
        )
        torch.cuda.synchronize()
        return out

    reference = _run_kernel_with_tactic(None).float()
    ref_max = reference.abs().max().item()
    assert torch.isfinite(reference).all(), (
        f"[{quant_mode}] reference output is not finite — bad test setup"
    )

    moe_op = gen_trtllm_gen_fused_moe_sm100_module().build_and_load()
    valid_tactics = _enumerate_fp8_valid_tactics(
        moe_op,
        quant_mode,
        top_k,
        hidden_size,
        intermediate_size,
        num_experts,
        num_tokens,
    )
    assert len(valid_tactics) > 0, f"[{quant_mode}] no valid tactics returned"

    _run_kernel_with_tactic(list(valid_tactics[0]))
    assert _TEST_LOG_KEY_FP8 in AutoTuner.get()._logged_file_hits, (
        f"[{quant_mode}] forced tactic was not dispatched — autotuner did "
        f"not log a cache hit; check `_moe_profile_shapes` (scale_dim) "
        f"against the actual MoEInputs layout."
    )
    print(
        f"\n[all_tactics_correctness] op=fp8 quant={quant_mode} "
        f"M={num_tokens} hidden={hidden_size} intermediate={intermediate_size} "
        f"experts={num_experts} top_k={top_k}: "
        f"sweeping {len(valid_tactics)} valid tactics",
        flush=True,
    )

    N_ITERS_PER_TACTIC = 10
    failures: list[str] = []
    t0 = time.time()
    for tactic in valid_tactics:
        line = _check_tactic(
            _run_kernel_with_tactic,
            list(tactic),
            reference,
            ref_max,
            N_ITERS_PER_TACTIC,
        )
        if line is not None:
            failures.append(line)
            print(f"  BAD: {line}", flush=True)
    print(
        f"  done in {time.time() - t0:.0f}s, "
        f"{len(failures)}/{len(valid_tactics)} tactics failed",
        flush=True,
    )
    if failures:
        raise AssertionError(
            f"[{quant_mode}] {len(failures)} of {len(valid_tactics)} tactics "
            f"failed correctness. First few:\n  " + "\n  ".join(failures[:20])
        )
