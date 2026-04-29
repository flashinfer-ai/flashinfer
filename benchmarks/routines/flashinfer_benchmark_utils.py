import argparse
import torch

from flashinfer.testing.utils import set_seed
from flashinfer.utils import get_compute_capability

# Output columns for the test results.
output_column_dict = {
    "perf": [
        "routine",
        "median_time",
        "std_time",
        "tflops",
        "tb_per_sec",
        "backend",
        "resolved_backend",
    ],
    "attention": [
        "s_qo",
        "s_kv",
        "head_dim_qk",
        "head_dim_vo",
        "head_dim_ckv",
        "head_dim_kpe",
        "causal",
        "q_dtype",
        "kv_dtype",
        "avg_actual_seq_len",
        "random_actual_seq_len",
    ],
    "gemm": [
        "n",
        "group_size",
        "tile_size",
        "scale_major_mode",
        "mma_sm",
        "use_128x4_sf_layout",
        "use_nvfp4",
        "bias",
    ],
    "moe": [
        "num_tokens",
        "intermediate_size",
        "num_experts",
        "top_k",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "local_expert_offset",
        "local_num_experts",
        "routing_method",
        "use_shuffled_weight",
        "weight_layout",
        "use_routing_bias",
        "use_routing_scales_on_input",
        "weight_dtype",
        "activation_type",
        "fp4_mode",
        # CUTLASS fused MoE specific
        "cutlass_variant",
        "quantized_input",
        "tp_size",
        "tp_rank",
        "ep_size",
        "ep_rank",
    ],
    "moe_comm": [
        "num_tokens",
        "num_experts",
        "top_k",
        "ep_size",
        "max_num_tokens",
    ],
    "allreduce_comm": [
        "num_tokens",
        "ar_backend",
        "pattern",
        "layout_code",
    ],
    "mixed_comm": [
        "local_bs",
        "op_name",
        "mode_name",
        "local_tp_size",
        "local_dp_size",
        "inter_tp_size",
        "inter_dp_size",
    ],
    "norm": [
        "num_heads",
        "scale",
        "eps",
        "use_global_scale",
    ],
    "quantization": [
        "alignment",
        "global_scale",
        "sf_layout",
        "do_shuffle",
        "sf_vec_size",
    ],
    "sampling": [
        "vocab_size",
        "top_k",
        "top_p",
        "min_p",
        "temperature",
        "num_speculate_tokens",
        "filter_apply_order",
        "max_len",
        "num_rows",
    ],
    "rope": [
        "seq_len",
        "head_dim",
        "rotary_dim",
        "no_rope_dim",
        "rope_theta",
        "rope_scale",
        "interleave",
        "kv_layout",
    ],
    "mamba": [
        "nheads",
        "dim",
        "dstate",
        "ngroups",
        "cache_steps",
        "state_dtype",
        "weight_dtype",
        "has_z",
        "dt_softplus",
    ],
    "general": [
        "batch_size",
        "hidden_size",
        "input_dtype",
        "out_dtype",
        "quant_dtype",
        "m",
        "k",
        "num_qo_heads",
        "num_kv_heads",
        "page_size",
        "enable_pdl",
        "is_sf_swizzled_layout",
        "refcheck",
        "no_cuda_graph",
        "use_cupti",
        "allow_output_mismatch",
        "random_seed",
        "case_tag",
        "generate_repro_command",
        "repro_command",
    ],
}

full_output_columns = (
    output_column_dict["perf"]
    + output_column_dict["attention"]
    + output_column_dict["gemm"]
    + output_column_dict["moe"]
    + output_column_dict["moe_comm"]
    + output_column_dict["allreduce_comm"]
    + output_column_dict["mixed_comm"]
    + output_column_dict["norm"]
    + output_column_dict["quantization"]
    + output_column_dict["sampling"]
    + output_column_dict["rope"]
    + output_column_dict["mamba"]
    + output_column_dict["general"]
)

benchmark_apis = {
    "attention": [
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
        "BatchMLAPagedAttentionWrapper",
    ],
    "gemm": [
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
        "bmm_fp8",
        "bmm_mxfp8",
        "mm_fp4",
        "mm_mxfp8",
        "mm_bf16",
        "bmm_bf16",
        "tinygemm_bf16",
    ],
    "moe": [
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp8_per_tensor_scale_moe",
        "cutlass_fused_moe",
        "cute_dsl_fp4_block_scale_moe",
        "b12x_fused_moe",
    ],
    "moe_comm": [
        "moe_a2a_dispatch_combine",
    ],
    "allreduce_comm": [
        "allreduce_fusion",
    ],
    "mixed_comm": [
        "mixed_comm",
    ],
    "norm": [
        "rmsnorm",
        "rmsnorm_quant",
        "fused_add_rmsnorm_quant",
        "rmsnorm_fp4quant",
        "add_rmsnorm_fp4quant",
        "fused_rmsnorm_silu",
    ],
    "quantization": [
        "mxfp8_quantize",
        "mxfp4_quantize",
        "nvfp4_quantize",
        "nvfp4_batched_quantize",
    ],
    "sampling": [
        "softmax",
        "sampling_from_probs",
        "sampling_from_logits",
        "top_k_sampling_from_probs",
        "top_p_sampling_from_probs",
        "top_k_top_p_sampling_from_probs",
        "top_k_top_p_sampling_from_logits",
        "min_p_sampling_from_probs",
        "top_k_renorm_probs",
        "top_p_renorm_probs",
        "top_k_mask_logits",
        "chain_speculative_sampling",
        "top_k",
        "top_k_page_table_transform",
        "top_k_ragged_transform",
    ],
    "rope": [
        "apply_rope",
        "apply_rope_pos_ids",
        "apply_llama31_rope",
        "apply_llama31_rope_pos_ids",
        "apply_rope_with_cos_sin_cache",
        "mla_rope_quantize_fp8",
        "rope_quantize_fp8",
        "rope_quantize_fp8_append_paged_kv_cache",
    ],
    "mamba": [
        "selective_state_update",
    ],
}


def print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec):
    output_backend_width = max(15, len(backend))
    print(
        f"[PERF] {backend.ljust(output_backend_width)}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
    )


def get_device(args):
    # Synchronize to ensure that the device is ready after previous tests
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    set_seed(args.random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(torch.cuda.current_device()).replace(" ", "_")
    if args.verbose >= 2:
        print(f"[VVERBOSE] {gpu_name = }")
    return device


def is_close_stats(input, other, rtol=1e-5, atol=1e-8):
    close_tensor = torch.isclose(input, other, rtol=rtol, atol=atol)
    num_elements = close_tensor.numel()
    num_different_elements = num_elements - close_tensor.sum().item()
    return (
        num_different_elements,  # number of different elements
        num_elements,  # total number of elements in tensor
        num_different_elements / num_elements * 100.0,
    )


def dtype_str_to_torch_dtype(dtype_str):
    if dtype_str == "bfloat16":
        return torch.bfloat16
    elif dtype_str == "float16":
        return torch.float16
    elif dtype_str == "float32":
        return torch.float32
    elif dtype_str == "float64":
        return torch.float64
    elif dtype_str == "fp8_e4m3":
        return torch.float8_e4m3fn
    elif dtype_str == "fp8_e5m2":
        return torch.float8_e5m2
    elif dtype_str == "nvfp4":
        return torch.uint8
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


routine_cc_to_supported_backends = {
    # ATTENTION
    "BatchDecodeWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache
        "7.5": ["fa2", "auto"],
        "8.0": ["fa2", "fa2_tc", "auto", "cudnn"],
        "8.6": ["fa2", "fa2_tc", "auto", "cudnn"],
        "8.9": ["fa2", "fa2_tc", "auto", "cudnn"],
        "9.0": ["fa2", "fa2_tc", "auto", "cudnn", "trtllm-native"],
        "10.0": ["fa2", "fa2_tc", "auto", "cudnn", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "fa2_tc", "auto", "cudnn", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "fa2_tc", "auto", "cudnn", "trtllm-native"],
        "12.1": ["fa2", "fa2_tc", "auto", "cudnn", "trtllm-native"],
    },
    "BatchPrefillWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_context_with_kv_cache
        # NOTE: cudnn-native calls cudnn_batch_prefill_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "auto", "cudnn", "cudnn-native"],
        "8.6": ["fa2", "auto", "cudnn", "cudnn-native"],
        "8.9": ["fa2", "auto", "cudnn", "cudnn-native"],
        "9.0": ["fa2", "fa3", "auto", "cudnn", "cudnn-native"],
        "10.0": ["fa2", "auto", "cudnn", "cudnn-native", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "auto", "cudnn", "cudnn-native", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "auto", "cudnn", "cudnn-native"],
        "12.1": ["fa2", "auto", "cudnn", "cudnn-native"],
    },
    "BatchPrefillWithRaggedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_ragged_attention_deepseek
        # NOTE: cudnn-native calls cudnn_batch_prefill_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "cudnn", "cudnn-native"],
        "8.6": ["fa2", "cudnn", "cudnn-native"],
        "8.9": ["fa2", "cudnn", "cudnn-native"],
        "9.0": ["fa2", "fa3", "cudnn", "cudnn-native"],
        "10.0": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "cute-dsl",
            "trtllm-native",
        ],
        "10.3": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "cute-dsl",
            "trtllm-native",
        ],
        "12.0": ["fa2", "cudnn", "cudnn-native"],
        "12.1": ["fa2", "cudnn", "cudnn-native"],
    },
    "BatchMLAPagedAttentionWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache_mla
        # NOTE: cute-dsl calls trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl")
        "7.5": [],
        "8.0": ["fa2"],
        "8.6": ["fa2"],
        "8.9": ["fa2"],
        "9.0": ["fa2", "fa3"],
        "10.0": ["fa2", "cutlass", "trtllm-native", "cute-dsl"],
        "10.3": ["fa2", "cutlass", "trtllm-native"],
        "12.0": ["fa2"],
        "12.1": ["fa2"],
    },
    # GEMM
    "gemm_fp8_nt_groupwise": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
        "12.1": [],
    },
    "group_gemm_fp8_nt_groupwise": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
        "12.1": [],
    },
    "bmm_fp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": ["cudnn", "cublas"],
        "9.0": ["cudnn", "cublas"],
        "10.0": ["cudnn", "cublas", "cutlass"],
        "10.3": ["cudnn", "cublas", "cutlass"],
        "12.0": ["cudnn", "cublas"],
        "12.1": ["cudnn", "cublas"],
    },
    "bmm_mxfp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cudnn"],
        "10.3": ["cudnn"],
        "12.0": [],
        "12.1": [],
    },
    "mm_mxfp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass", "cute-dsl", "trtllm"],
        "10.3": ["cutlass", "cute-dsl", "trtllm"],
        "11.0": ["cutlass"],
        "12.0": [],
        "12.1": [],
    },
    "tinygemm_bf16": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["tinygemm"],
        "10.0": ["tinygemm"],
        "10.3": ["tinygemm"],
        "11.0": ["tinygemm"],
        "12.0": ["tinygemm"],
        "12.1": ["tinygemm"],
    },
    # Note: mm_fp4, mm_bf16, and bmm_bf16 use support checkers to filter backends, so they are not listed here
    # MOE
    "trtllm_fp4_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
        "12.1": [],
    },
    "trtllm_fp8_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
        "12.1": [],
    },
    "trtllm_fp8_per_tensor_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "12.0": [],
        "12.1": [],
    },
    "cutlass_fused_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": ["cutlass"],
        "12.1": ["cutlass"],
    },
    "cute_dsl_fp4_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "12.0": [],
        "12.1": [],
    },
    "b12x_fused_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": [],
        "10.3": [],
        "12.0": ["b12x"],
        "12.1": ["b12x"],
    },
    # NORM
    "rmsnorm": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "rmsnorm_quant": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "fused_add_rmsnorm_quant": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # NORM - FP4 Quantization (Blackwell SM100+ only, CuTe-DSL kernels)
    "rmsnorm_fp4quant": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "add_rmsnorm_fp4quant": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    # QUANTIZATION
    "mxfp8_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl"],
        "10.3": ["cuda", "cute-dsl"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
    },
    "mxfp4_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl"],
        "10.3": ["cuda", "cute-dsl"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
    },
    "nvfp4_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl"],
        "10.3": ["cuda", "cute-dsl"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
    },
    "nvfp4_batched_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # SAMPLING
    "softmax": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "sampling_from_logits": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_p_sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_top_p_sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_top_p_sampling_from_logits": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "min_p_sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_renorm_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_p_renorm_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_mask_logits": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "chain_speculative_sampling": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_page_table_transform": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "top_k_ragged_transform": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # ROPE
    "apply_rope": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "apply_rope_pos_ids": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "apply_llama31_rope": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "apply_llama31_rope_pos_ids": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "apply_rope_with_cos_sin_cache": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "mla_rope_quantize_fp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "rope_quantize_fp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "rope_quantize_fp8_append_paged_kv_cache": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # MAMBA
    "selective_state_update": {
        "7.5": ["flashinfer", "triton"],
        "8.0": ["flashinfer", "triton"],
        "8.6": ["flashinfer", "triton"],
        "8.9": ["flashinfer", "triton"],
        "9.0": ["flashinfer", "triton"],
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
        "11.0": ["flashinfer", "triton"],
        "12.0": ["flashinfer", "triton"],
        "12.1": ["flashinfer", "triton"],
    },
}


def filter_backends_by_compute_capability(backends, routine, device):
    # FlashInfer currently does not have an isSupported() function that checks support.
    # WAR: Use helper function to check support.
    major, minor = get_compute_capability(device)
    compute_capability = f"{major}.{minor}"

    # If the compute capability is not supported, return an empty list.
    cc_to_supported_backends = routine_cc_to_supported_backends[routine]
    supported_backends = cc_to_supported_backends.get(compute_capability, [])
    backends_to_remove = []
    for backend in backends:
        if backend not in supported_backends:
            backends_to_remove.append(backend)
    for backend in backends_to_remove:
        backends.remove(backend)
        print(
            f"[WARNING] {backend} for routine {routine} is not supported on compute capability {compute_capability}. Skipping."
        )
    return backends


def enum_type(enum_class):
    """Generic factory for argparse enum types."""

    def converter(value):
        try:
            lower_name_to_member = {m.name.lower(): m for m in enum_class}
            return lower_name_to_member[value.lower()]
        except KeyError as e:
            raise argparse.ArgumentTypeError(
                f"Invalid value '{value}'. Must be one of: {', '.join([m.name for m in enum_class])}"
            ) from e

    return converter
