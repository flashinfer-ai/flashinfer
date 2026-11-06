"""The served kernel as the baseline: FlashInfer's CUTLASS fused MoE module entry.

cudagym calls run(*inputs, output); FlashInfer's trace_apply calls it by keyword
with the Definition's input names and output=. It writes `output` in place and
returns None. Every scalar is the served value, fixed: the served-scalar contract
registered beside this Solution says which calls may reach it. The module entry
is called rather than the public `cutlass_fused_moe`, so the call is neither
traced nor routed back to a Solution; the arguments mirror what the public API
passes, `enable_pdl` resolved the way it resolves it. The GEMM tactics are the
autotuner's, profiled once per token count the way serving profiles them.
"""

import torch
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.core import ActivationType, get_cutlass_fused_moe_module
from flashinfer.utils import device_support_pdl, get_compute_capability

NUM_LOCAL_EXPERTS = 128
HIDDEN_SIZE = 2688
INTERMEDIATE_SIZE = 1856
GEMM1_OUT_SIZE = 1856
#: Token counts whose two GEMM tactics this process has profiled, per device.
_TUNED = set()


def run(
    input,
    token_selected_experts,
    token_final_scales,
    fc1_expert_weights,
    fc2_expert_weights,
    output,
):
    fc1_shape = (NUM_LOCAL_EXPERTS, GEMM1_OUT_SIZE, HIDDEN_SIZE)
    fc2_shape = (NUM_LOCAL_EXPERTS, HIDDEN_SIZE, INTERMEDIATE_SIZE)
    if tuple(fc1_expert_weights.shape) != fc1_shape:
        raise ValueError(
            f"fc1_expert_weights {tuple(fc1_expert_weights.shape)} != {fc1_shape}"
        )
    if tuple(fc2_expert_weights.shape) != fc2_shape:
        raise ValueError(
            f"fc2_expert_weights {tuple(fc2_expert_weights.shape)} != {fc2_shape}"
        )
    major, minor = get_compute_capability(input.device)
    module = get_cutlass_fused_moe_module(f"{major * 10 + minor}")

    def _call():
        module.cutlass_fused_moe(
            output,
            input,
            token_selected_experts,
            token_final_scales,
            fc1_expert_weights,
            None,
            fc2_expert_weights,
            None,
            torch.bfloat16,
            [],
            input_sf=None,
            swiglu_alpha=None,
            swiglu_beta=None,
            swiglu_limit=None,
            tp_size=1,
            tp_rank=0,
            ep_size=1,
            ep_rank=0,
            cluster_size=1,
            cluster_rank=0,
            use_packed_weights=False,
            enable_alltoall=False,
            use_deepseek_fp8_block_scale=False,
            use_w4_group_scaling=False,
            use_mxfp8_act_scaling=False,
            min_latency_mode=False,
            tune_max_num_tokens=8192,
            enable_pdl=device_support_pdl(input.device),
            activation_type=ActivationType.Relu2,
            swizzled_input_sf=True,
        )

    # Serving profiles the two GEMM tactics at warm-up (vLLM runs FlashInfer's
    # autotune before CUDA-graph capture); a cold process would time the
    # fallback tactic instead. Profile once per token count, never inside a
    # capture, then run the call as the served path runs it.
    key = (input.shape[0], input.device.index)
    with torch.cuda.device(input.device):
        if key not in _TUNED and not torch.cuda.is_current_stream_capturing():
            with autotune(True):
                _call()
            _TUNED.add(key)
        _call()
