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

Offline probe that generates a per-GPU tactics blocklist.

Runs the autotuner on the current GPU with representative MoE inputs. Any
tactic that fails during profiling is recorded in a JSON blocklist file that
the autotuner loads at runtime via the ``FLASHINFER_TACTICS_BLOCKLIST``
environment variable.

This module backs the ``flashinfer generate-tactics-blocklist`` CLI command;
its heavy imports (torch, CUTLASS/TRT-LLM MoE backends) are intentionally kept
out of the package's import path and only pulled in when the command runs.

Requires a GPU and the full FlashInfer + CUTLASS environment.
"""

import time
from typing import Dict, List, Optional, Sequence, Set

import torch

from . import ActivationType, RoutingMethodType, fp4_quantize
from .autotuner import AutoTuner, autotune
from .fused_moe import (
    Fp8QuantizationType,
    WeightLayout,
    cutlass_fused_moe,
    trtllm_fp4_block_scale_moe,
    trtllm_fp8_block_scale_moe,
)
from .tactics_blocklist import TacticsBlocklist
from .utils import device_support_pdl

FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max


def _fp8_quantize(x):
    mx = x.abs().max().float()
    scale = FLOAT8_E4M3_MAX / mx
    return (x * scale).to(torch.float8_e4m3fn), 1.0 / scale


def _probe_fp4(device, num_tokens, num_experts, hidden_size, intermediate_size, top_k):
    """Run FP4 MoE autotuning and return the autotuner's failed_tactics."""
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.bfloat16
    )
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )

    hidden_states_q, hidden_states_scale = fp4_quantize(
        hidden_states,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=False,
    )
    hidden_states_scale = hidden_states_scale.view(torch.float8_e4m3fn).reshape(
        num_tokens, -1
    )

    w13 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
    )
    w13_q, w13_scale = fp4_quantize(
        w13,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    w13_scale = w13_scale.view(torch.float8_e4m3fn).reshape(
        num_experts, intermediate_size * 2, -1
    )
    w2_q, w2_scale = fp4_quantize(
        w2,
        torch.tensor([448.0 * 6.0], device=device),
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    w2_scale = w2_scale.view(torch.float8_e4m3fn).reshape(num_experts, hidden_size, -1)

    global_scale = 1.0 / 448.0 / 6.0
    output1_scale = torch.full(
        (num_experts,), global_scale * global_scale, device=device
    )
    output1_scale_gate = torch.full(
        (num_experts,), global_scale * global_scale, device=device
    )
    output2_scale = torch.full(
        (num_experts,), global_scale * global_scale, device=device
    )
    bias13 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10
    bias2 = torch.randn(num_experts, intermediate_size * 2, device=device) * 10

    print("  Probing NvFP4xNvFP4 ...")
    try:
        with autotune(tune_mode=True):
            trtllm_fp4_block_scale_moe(
                hidden_states=hidden_states_q,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=w13_q,
                gemm1_weights_scale=w13_scale,
                gemm2_weights=w2_q,
                gemm2_weights_scale=w2_scale,
                gemm1_bias=bias13,
                gemm2_bias=bias2,
                routing_logits=routing_logits,
                routing_bias=None,
                gemm1_alpha=None,
                gemm1_beta=None,
                gemm1_clamp_limit=None,
                output1_scale_scalar=output1_scale,
                output1_scale_gate_scalar=output1_scale_gate,
                output2_scale_scalar=output2_scale,
                num_experts=num_experts,
                top_k=top_k,
                n_group=None,
                topk_group=None,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=None,
                routing_method_type=RoutingMethodType.Renormalize.value,
                do_finalize=True,
                enable_pdl=enable_pdl,
                activation_type=ActivationType.Swiglu,
                output=None,
                tune_max_num_tokens=num_tokens,
            )
    except Exception as e:
        print(f"    WARNING: FP4 probe raised: {e}")


def _probe_fp8_block(
    device, num_tokens, num_experts, hidden_size, intermediate_size, top_k
):
    """Run FP8 block-scale MoE autotuning and return the autotuner's failed_tactics."""
    enable_pdl = device_support_pdl(device)
    routing_logits = torch.rand(
        num_tokens, num_experts, device=device, dtype=torch.float32
    )
    routing_bias = torch.randn(num_experts, device=device, dtype=torch.bfloat16)
    hidden_states = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )

    scale_vec_size = 128
    hidden_states_q, hs_scale = _fp8_quantize(hidden_states)
    w13 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    w2 = torch.randn(
        num_experts, hidden_size, intermediate_size, device=device, dtype=torch.bfloat16
    )
    w13_q, w13_sc = _fp8_quantize(w13)
    w2_q, w2_sc = _fp8_quantize(w2)

    hidden_states_scale = torch.full(
        (hidden_size // scale_vec_size, num_tokens),
        hs_scale.item(),
        device=device,
    )
    w13_scale = torch.full(
        (
            num_experts,
            intermediate_size * 2 // scale_vec_size,
            hidden_size // scale_vec_size,
        ),
        w13_sc.item(),
        device=device,
    )
    w2_scale = torch.full(
        (
            num_experts,
            hidden_size // scale_vec_size,
            intermediate_size // scale_vec_size,
        ),
        w2_sc.item(),
        device=device,
    )

    print("  Probing Fp8-Block ...")
    try:
        with autotune(tune_mode=True):
            trtllm_fp8_block_scale_moe(
                hidden_states=hidden_states_q,
                hidden_states_scale=hidden_states_scale,
                gemm1_weights=w13_q,
                gemm1_weights_scale=w13_scale,
                gemm2_weights=w2_q,
                gemm2_weights_scale=w2_scale,
                routing_logits=routing_logits,
                routing_bias=routing_bias,
                num_experts=num_experts,
                top_k=top_k,
                n_group=8,
                topk_group=4,
                intermediate_size=intermediate_size,
                local_expert_offset=0,
                local_num_experts=num_experts,
                routed_scaling_factor=2.5,
                routing_method_type=RoutingMethodType.DeepSeekV3.value,
                use_shuffled_weight=False,
                weight_layout=WeightLayout.MajorK.value,
                enable_pdl=enable_pdl,
                tune_max_num_tokens=num_tokens,
                fp8_quantization_type=Fp8QuantizationType.DeepSeekFp8,
            )
    except Exception as e:
        print(f"    WARNING: FP8 probe raised: {e}")


def _run_cutlass_probe(
    label,
    device,
    num_tokens,
    num_experts,
    hidden_size,
    top_k,
    input_tensor,
    fc1,
    fc2,
    quant_scales,
    enable_pdl,
    input_sf=None,
    activation_type=ActivationType.Swiglu,
):
    """Run a single CUTLASS-path probe with autotune."""
    token_selected_experts = torch.randint(
        0, num_experts, (num_tokens, top_k), dtype=torch.int32, device=device
    )
    token_final_scales = torch.ones(
        num_tokens, top_k, dtype=torch.float32, device=device
    )
    output = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    print(f"  Probing {label} via CUTLASS path ...")
    try:
        with autotune(tune_mode=True):
            cutlass_fused_moe(
                input=input_tensor,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                fc1_expert_weights=fc1,
                fc2_expert_weights=fc2,
                output_dtype=torch.bfloat16,
                quant_scales=quant_scales,
                input_sf=input_sf,
                output=output,
                enable_pdl=enable_pdl,
                activation_type=activation_type,
                tune_max_num_tokens=num_tokens,
            )
    except Exception as e:
        print(f"    NOTE: Post-profiling execution raised (expected): {e}")


def _probe_fp4_cutlass(
    device, num_tokens, num_experts, hidden_size, intermediate_size, top_k
):
    """Probe NvFP4 MoE via the CUTLASS path (trtllm::fused_moe::gemm1/gemm2).

    Matches vLLM with moe_backend=flashinfer_cutlass + quantization=fp4.
    Tensor formats (weights as torch.long, scales as torch.int32) follow
    vLLM flashinfer_cutlass_moe.py.
    """
    enable_pdl = device_support_pdl(device)

    amax = torch.tensor([448.0 * 6.0], device=device)
    input_q, input_sf = fp4_quantize(
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16),
        amax,
        sf_vec_size=16,
        sf_use_ue8m0=False,
        is_sf_swizzled_layout=True,
    )
    input_sf = input_sf.view(torch.float8_e4m3fn).reshape(num_tokens, -1)

    fc1_weights, fc1_scales = fp4_quantize(
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        ),
        amax,
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    fc2_weights, fc2_scales = fp4_quantize(
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        ),
        amax,
        sf_vec_size=16,
        sf_use_ue8m0=False,
    )
    fc1_scales = fc1_scales.view(torch.int32)
    fc1_weights = fc1_weights.view(torch.long)
    fc2_scales = fc2_scales.view(torch.int32)
    fc2_weights = fc2_weights.view(torch.long)

    global_scale = 1.0 / (448.0 * 6.0)
    quant_scales = [
        torch.tensor([global_scale], device=device, dtype=torch.float32),
        fc1_scales,
        torch.tensor([global_scale * global_scale], device=device, dtype=torch.float32),
        torch.tensor([global_scale], device=device, dtype=torch.float32),
        fc2_scales,
        torch.tensor([global_scale * global_scale], device=device, dtype=torch.float32),
    ]

    _run_cutlass_probe(
        "NvFP4xNvFP4",
        device,
        num_tokens,
        num_experts,
        hidden_size,
        top_k,
        input_q,
        fc1_weights,
        fc2_weights,
        quant_scales,
        enable_pdl,
        input_sf=input_sf,
    )


def _probe_fp8_per_tensor_cutlass(
    device, num_tokens, num_experts, hidden_size, intermediate_size, top_k
):
    """Probe FP8 per-tensor MoE via the CUTLASS path."""
    enable_pdl = device_support_pdl(device)

    input_fp8, inv_input_scale = _fp8_quantize(
        torch.randn(num_tokens, hidden_size, device=device, dtype=torch.bfloat16)
    )
    fc1_fp8, inv_fc1_scale = _fp8_quantize(
        torch.randn(
            num_experts,
            intermediate_size * 2,
            hidden_size,
            device=device,
            dtype=torch.bfloat16,
        )
    )
    fc2_fp8, inv_fc2_scale = _fp8_quantize(
        torch.randn(
            num_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=torch.bfloat16,
        )
    )

    quant_scales = [
        torch.tensor(inv_fc1_scale * inv_input_scale, device=device),
        torch.tensor(1.0, device=device),
        torch.tensor(inv_fc2_scale, device=device),
        torch.tensor(inv_input_scale, device=device),
    ]

    _run_cutlass_probe(
        "Fp8-PerTensor",
        device,
        num_tokens,
        num_experts,
        hidden_size,
        top_k,
        input_fp8,
        fc1_fp8,
        fc2_fp8,
        quant_scales,
        enable_pdl,
    )


def _probe_bf16_cutlass(
    device, num_tokens, num_experts, hidden_size, intermediate_size, top_k
):
    """Probe unquantized BF16 MoE via the CUTLASS path.

    Covers drafter/MTP models that use the FlashInfer CUTLASS Unquantized
    MoE backend at runtime.
    """
    enable_pdl = device_support_pdl(device)

    input_bf16 = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    fc1 = torch.randn(
        num_experts,
        intermediate_size * 2,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    fc2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )

    _run_cutlass_probe(
        "BF16",
        device,
        num_tokens,
        num_experts,
        hidden_size,
        top_k,
        input_bf16,
        fc1,
        fc2,
        [],
        enable_pdl,
    )


def _probe_bf16_relu2_cutlass(
    device, num_tokens, num_experts, hidden_size, intermediate_size, top_k
):
    """Probe unquantized BF16 MoE with the non-gated Relu2 activation.

    Covers non-gated MoE models (e.g. Nemotron) so the blocklist is not limited
    to the gated Swiglu path. Relu2 has no gate, so GEMM1 produces a single
    ``intermediate_size`` projection rather than the ``2 * intermediate_size``
    used by gated activations.
    """
    enable_pdl = device_support_pdl(device)

    input_bf16 = torch.randn(
        num_tokens, hidden_size, device=device, dtype=torch.bfloat16
    )
    fc1 = torch.randn(
        num_experts,
        intermediate_size,
        hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    fc2 = torch.randn(
        num_experts,
        hidden_size,
        intermediate_size,
        device=device,
        dtype=torch.bfloat16,
    )

    _run_cutlass_probe(
        "BF16-Relu2",
        device,
        num_tokens,
        num_experts,
        hidden_size,
        top_k,
        input_bf16,
        fc1,
        fc2,
        [],
        enable_pdl,
        activation_type=ActivationType.Relu2,
    )


PROBE_FUNCTIONS = {
    "NvFP4xNvFP4": _probe_fp4,
    "Fp8-Block": _probe_fp8_block,
    # CUTLASS-path probes (for vLLM flashinfer_cutlass backend)
    "NvFP4-CUTLASS": _probe_fp4_cutlass,
    "Fp8-PerTensor-CUTLASS": _probe_fp8_per_tensor_cutlass,
    "BF16-CUTLASS": _probe_bf16_cutlass,
    # Non-gated activation coverage (e.g. Nemotron Relu2).
    "BF16-Relu2-CUTLASS": _probe_bf16_relu2_cutlass,
}


def generate(
    output: Optional[str] = None,
    quant_modes: Optional[Sequence[str]] = None,
    num_tokens: int = 64,
    num_experts: int = 256,
    hidden_size: int = 7168,
    intermediate_size: int = 2048,
    top_k: int = 8,
    device: str = "cuda:0",
) -> Optional[str]:
    """Probe the GPU and write a tactics blocklist JSON.

    Args:
        output: Output JSON path. Defaults to ``tactics_<gpu_name>.json``.
        quant_modes: Quant modes to probe. Defaults to all known modes
            (see ``PROBE_FUNCTIONS``).
        num_tokens, num_experts, hidden_size, intermediate_size, top_k:
            Representative MoE shape to profile.
        device: CUDA device to probe.

    Returns:
        The path the blocklist was written to, or ``None`` when every tactic
        passed and no file was needed.
    """
    if quant_modes is None:
        quant_modes = list(PROBE_FUNCTIONS.keys())
    unknown = [m for m in quant_modes if m not in PROBE_FUNCTIONS]
    if unknown:
        raise ValueError(
            f"Unknown quant mode(s): {unknown}. "
            f"Valid modes: {list(PROBE_FUNCTIONS.keys())}"
        )

    device = torch.device(device)
    gpu_name = torch.cuda.get_device_name(device)
    print(f"GPU: {gpu_name}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"Quant modes: {list(quant_modes)}")
    print()

    if output is None:
        safe_name = gpu_name.replace(" ", "_").replace("/", "_")
        output = f"tactics_{safe_name}.json"

    all_failed_tactics: Dict[str, Set] = {}
    start = time.time()

    for mode in quant_modes:
        print(f"[{mode}]")
        # Reset singleton so each probe starts clean.
        AutoTuner._instance = None
        tuner = AutoTuner.get()

        PROBE_FUNCTIONS[mode](
            device,
            num_tokens=num_tokens,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            top_k=top_k,
        )

        # stats.failed_tactics is Dict["custom_op::RunnerClassName", Set[tactic]].
        for key, tactic_set in tuner.stats.failed_tactics.items():
            all_failed_tactics.setdefault(key, set()).update(tactic_set)
            print(f"    {key}: {len(tactic_set)} failed tactic(s)")

        print()

    elapsed = time.time() - start
    total = sum(len(v) for v in all_failed_tactics.values())

    if total == 0:
        print(f"All tactics passed on {gpu_name}. No blocklist file needed.")
        print(f"Completed in {elapsed:.1f}s")
        return None

    serializable: Dict[str, List] = {k: list(v) for k, v in all_failed_tactics.items()}
    TacticsBlocklist.save(output, serializable)

    print(f"Saved {total} invalid tactic(s) to {output}")
    print(f"Completed in {elapsed:.1f}s")
    print()
    print("To use at runtime:")
    print(f"  export FLASHINFER_TACTICS_BLOCKLIST={output}")
    return output
