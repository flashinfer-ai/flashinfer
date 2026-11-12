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
        "v_dtype",
        "avg_actual_seq_len",
        "random_actual_seq_len",
        "is_var_seq",
        "cute_dsl_impl",
        "softmax_rescale_threshold_log2",
        "timing_metric",
        "row_activity_mode",
        "calls_per_sample",
    ],
    "dsv4_sparse_mla": [
        "swa_topk",
        "compressed_topk",
        "compressed_kv_len",
        "compressed_page_size",
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
        "block_m",
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
        "swiglu_limit",
        "quant_variant",
        "autotune",
        "tactic",
        "refcheck_passed",
        "relative_l2",
        "pairwise_relative_l2",
        "runner_backend_key",
        "fp4_mode",
        "cold_l2_cache",
        "prequantized_median_time",
        "prequantized_std_time",
        # CUTLASS fused MoE specific
        "cutlass_variant",
        "quantized_input",
        "tp_size",
        "tp_rank",
        "ep_size",
        "ep_rank",
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
        "dit_mode",
        "ppf",
        "pph",
        "ppw",
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
        "per_row_top_k",
        "top_k_max",
        "max_len",
        "num_rows",
    ],
    # top_k_varlen selects top-K KV positions per request; its row width is a
    # max sequence length, not a vocab size (see routines/topk_varlen.py).
    "topk_varlen": [
        "max_seq_len",
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
        "conv_tokens_per_sequence",
        "conv_seq_lens",
        "conv_bias",
        "conv_speculative",
        "conv_initial_state",
        "conv_layout",
        "conv_input_row_stride",
        "conv_graph_refcheck_passed",
        "conv_workspace_bytes",
        "conv_host_submit_us",
        "conv_cuda_ops",
    ],
    "gdn": [
        "num_q_heads",
        "num_k_heads",
        "num_v_heads",
        "head_size",
        "state_layout",
        "pool_mode",
        "update_state",
        "use_qk_l2norm",
    ],
    "kda": [
        # Which variant the policy chose, and whether this device's thresholds
        # were measured or inherited from another SM count. A time taken under
        # fallback thresholds is not a time taken under tuned ones, and no
        # other column distinguishes them.
        "kda_variant",
        "kda_variant_policy",
        "sm_count",
        "packed",
        "has_initial_state",
    ],
    "msa": [
        "topk",
        "max_k_tiles",
        "total_q",
        "total_kv",
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
        "capture_telemetry",
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
    + output_column_dict["dsv4_sparse_mla"]
    + output_column_dict["gemm"]
    + output_column_dict["moe"]
    + output_column_dict["allreduce_comm"]
    + output_column_dict["mixed_comm"]
    + output_column_dict["norm"]
    + output_column_dict["quantization"]
    + output_column_dict["sampling"]
    + output_column_dict["topk_varlen"]
    + output_column_dict["rope"]
    + output_column_dict["mamba"]
    + output_column_dict["gdn"]
    + output_column_dict["kda"]
    + output_column_dict["msa"]
    + output_column_dict["general"]
)

benchmark_apis = {
    "attention": [
        "BatchDecodeWithPagedKVCacheWrapper",
        "BatchPrefillWithPagedKVCacheWrapper",
        "BatchPrefillWithRaggedKVCacheWrapper",
        "BatchMLAPagedAttentionWrapper",
        "trtllm_batch_decode_sparse_mla_dsv4",
    ],
    "gemm": [
        "gemm_fp8_nt_groupwise",
        "group_gemm_fp8_nt_groupwise",
        "bmm_fp8",
        "mm_fp8",
        "bmm_mxfp8",
        "mm_fp4",
        "mm_bf16_fp4",
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
        "cute_dsl_bf16_moe",
        "b12x_fused_moe",
        "alphamoe_nvfp4_aligned_moe",
        "unified_nvfp4_moe",
        "bgmv_moe",
    ],
    # Uses each unified backend config's supported(arch) check followed by a
    # real runner construction/probe, like mm_fp4's runtime backend filtering.
    "unified_moe": [
        "unified_moe",
    ],
    "allreduce_comm": [
        "allreduce_fusion",
    ],
    "mixed_comm": [
        "mixed_comm",
    ],
    "norm": [
        "rmsnorm",
        "fused_add_rmsnorm",
        "gemma_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "rmsnorm_quant",
        "fused_add_rmsnorm_quant",
        "layernorm_quant",
        "rmsnorm_fp4quant",
        "add_rmsnorm_fp4quant",
        "fused_rmsnorm_silu",
        "fused_dit_layernorm",
        "fused_qk_rmsnorm_rope",
    ],
    "quantization": [
        "mxfp8_quantize",
        "mxfp4_quantize",
        "nvfp4_quantize",
        "nvfp4_batched_quantize",
    ],
    "sampling": [
        "softmax",
        "log_softmax",
        "log_probs",
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
    # top_k_varlen is a sparse-attention KV-selection primitive (not vocab
    # sampling), so it has its own category + routine module (routines/topk_varlen.py).
    "topk_varlen": [
        "top_k_varlen",
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
        "causal_conv1d_fn",
        "causal_conv1d_update",
    ],
    "gdn": [
        "gated_delta_rule_decode",
        "gated_delta_rule_mtp",
        "chunk_gated_delta_rule",
    ],
    "kda": [
        "recurrent_kda_prefill",
    ],
    "sparse_attention": [
        "MSAProxyScore",
        "MSASparseAttention",
        "MSASparseDecode",
        "MSAPipeline",
    ],
}


def print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec):
    output_backend_width = max(15, len(backend))
    print(
        f"[PERF] {backend.ljust(output_backend_width)}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
    )


def warn_if_pdl_unsupported(args, routine_name):
    """Emit a one-shot warning if --enable_pdl is set but the routine's API does
    not accept enable_pdl. Call from the top of test functions whose underlying
    flashinfer API takes no enable_pdl parameter, so users get clear feedback
    that the flag is a no-op instead of silently being ignored.
    """
    if getattr(args, "enable_pdl", False):
        print(
            f"[WARNING] --enable_pdl provided but routine {routine_name} does not support PDL; flag is ignored."
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


def to_float8(x, dtype=torch.float8_e4m3fn):
    """Quantize ``x`` to FP8 with a per-tensor scale and return the inverse scale.

    Matches the test_trtllm_gen_attention_decode.py approach: the scale keeps a
    10x headroom below the FP8 max so attention inputs do not saturate.
    """
    finfo = torch.finfo(dtype)
    min_val, max_val = x.aminmax()
    amax = torch.maximum(min_val.abs(), max_val.abs()).clamp(min=1e-12)
    scale = finfo.max / amax * 0.1
    x_scl_sat = (x * scale).clamp(min=finfo.min, max=finfo.max)
    return x_scl_sat.to(dtype), scale.float().reciprocal()


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
        # NOTE: cudnn-native calls cudnn_batch_decode_with_kv_cache
        "7.5": ["fa2", "auto"],
        "8.0": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native"],
        "8.6": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native"],
        "8.9": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native"],
        "9.0": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native", "trtllm-native"],
        "10.0": [
            "fa2",
            "fa2_tc",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "10.3": [
            "fa2",
            "fa2_tc",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "10.7": [
            "fa2",
            "fa2_tc",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "12.0": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native", "trtllm-native"],
        "12.1": ["fa2", "fa2_tc", "auto", "cudnn", "cudnn-native", "trtllm-native"],
    },
    "BatchPrefillWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_context_with_kv_cache
        # NOTE: trtllm-fmha-v2 calls trtllm_fmha_v2_prefill
        # NOTE: cudnn-native calls cudnn_batch_prefill_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "auto", "cudnn", "cudnn-native"],
        "8.6": ["fa2", "auto", "cudnn", "cudnn-native"],
        "8.9": ["fa2", "auto", "cudnn", "cudnn-native"],
        "9.0": ["fa2", "fa3", "auto", "cudnn", "cudnn-native", "trtllm-fmha-v2"],
        "10.0": [
            "fa2",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "10.3": [
            "fa2",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "10.7": [
            "fa2",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-gen",
            "trtllm-native",
            "prims-ts",
        ],
        "12.0": [
            "fa2",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-fmha-v2",
            "cute-dsl-prims",
        ],
        "12.1": ["fa2", "auto", "cudnn", "cudnn-native", "cute-dsl-prims"],
    },
    "BatchPrefillWithRaggedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_ragged_attention_deepseek
        # NOTE: trtllm-fmha-v2 calls trtllm_fmha_v2_prefill
        # NOTE: cudnn-native calls cudnn_batch_prefill_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "cudnn", "cudnn-native"],
        "8.6": ["fa2", "cudnn", "cudnn-native"],
        "8.9": ["fa2", "cudnn", "cudnn-native"],
        "9.0": ["fa2", "fa3", "cudnn", "cudnn-native", "trtllm-fmha-v2"],
        "10.0": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "cute-dsl",
            "trtllm-native",
            "prims-ts",
        ],
        "10.3": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "cute-dsl",
            "trtllm-native",
            "prims-ts",
        ],
        "10.7": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "cutlass",
            "cute-dsl",
            "trtllm-native",
            "prims-ts",
        ],
        "12.0": [
            "fa2",
            "cudnn",
            "cudnn-native",
            "trtllm-fmha-v2",
            "cute-dsl-prims",
        ],
        "12.1": ["fa2", "cudnn", "cudnn-native", "cute-dsl-prims"],
    },
    "BatchMLAPagedAttentionWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")
        # NOTE: cute-dsl calls trtllm_batch_decode_with_kv_cache_mla(backend="cute-dsl")
        # NOTE: auto calls trtllm_batch_decode_with_kv_cache_mla(backend="auto")
        #       and is the only backend that benefits from --autotune
        "7.5": [],
        "8.0": ["fa2"],
        "8.6": ["fa2"],
        "8.9": ["fa2"],
        "9.0": ["fa2", "fa3"],
        "10.0": ["fa2", "cutlass", "trtllm-native", "cute-dsl", "auto", "prims-ts"],
        "10.3": ["fa2", "cutlass", "trtllm-native", "cute-dsl", "auto", "prims-ts"],
        "10.7": ["fa2", "cutlass", "trtllm-native", "cute-dsl", "prims-ts"],
        "12.0": ["fa2"],
        "12.1": ["fa2"],
    },
    "trtllm_batch_decode_sparse_mla_dsv4": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm-gen"],
        "10.3": ["trtllm-gen"],
        "10.7": [],
        "12.0": [],
        "12.1": [],
    },
    # GEMM
    "gemm_fp8_nt_groupwise": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass", "auto", "batch_invariant"],
        "10.3": ["cutlass", "auto", "batch_invariant"],
        "10.7": ["cutlass", "cute-dsl", "auto", "batch_invariant"],
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
        "10.7": ["cutlass"],
        "12.0": [],
        "12.1": [],
    },
    "bmm_mxfp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cudnn"],
        "10.3": ["cudnn"],
        "10.7": ["cudnn", "cute-dsl"],
        "12.0": ["cudnn"],
        "12.1": ["cudnn"],
    },
    "mm_mxfp8": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass", "cute-dsl", "trtllm", "cudnn", "batch_invariant"],
        "10.3": ["cutlass", "cute-dsl", "trtllm", "cudnn", "batch_invariant"],
        "10.7": ["cutlass", "cute-dsl", "trtllm", "batch_invariant"],
        "11.0": ["cutlass", "cudnn"],
        "12.0": ["cutlass", "cudnn"],
        "12.1": ["cutlass", "cudnn"],
    },
    "tinygemm_bf16": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["tinygemm"],
        "10.0": ["tinygemm"],
        "10.3": ["tinygemm"],
        "10.7": ["tinygemm"],
        "11.0": ["tinygemm"],
        "12.0": ["tinygemm"],
        "12.1": ["tinygemm"],
    },
    # Note: bmm_fp8, mm_fp8, mm_fp4, mm_bf16, bmm_bf16, and top_k_varlen use support checkers to filter backends, so they are not listed here
    # MOE
    "trtllm_fp4_block_scale_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["trtllm"],
        "10.3": ["trtllm"],
        "10.7": ["trtllm"],
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
        "10.7": ["trtllm"],
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
        "10.7": ["trtllm"],
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
        "10.7": ["cutlass"],
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
        "10.7": ["cute-dsl"],
        "12.0": [],
        "12.1": [],
    },
    "cute_dsl_bf16_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["cute-dsl"],
        "10.0": [],
        "10.3": [],
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
    "alphamoe_nvfp4_aligned_moe": {
        "10.0": ["alphamoe"],
        "10.3": ["alphamoe"],
    },
    # MoELayer cross-backend NVFP4: intersection of CuteDSL + TRTLLM FP4 support.
    # Blackwell and Rubin; backend support checks enforce runtime requirements.
    "unified_nvfp4_moe": {
        "10.0": ["unified"],
        "10.3": ["unified"],
        "10.7": ["unified"],
    },
    # NORM
    "rmsnorm": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "fused_add_rmsnorm": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "gemma_rmsnorm": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "gemma_fused_add_rmsnorm": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "rmsnorm_quant": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "fused_add_rmsnorm_quant": {
        "7.5": ["cute-dsl"],
        "8.0": ["cute-dsl"],
        "8.6": ["cute-dsl"],
        "8.9": ["cute-dsl"],
        "9.0": ["cute-dsl"],
        "10.0": ["cute-dsl"],
        "10.3": ["cute-dsl"],
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    "layernorm_quant": {
        "7.5": ["cuda", "cute-dsl"],
        "8.0": ["cuda", "cute-dsl"],
        "8.6": ["cuda", "cute-dsl"],
        "8.9": ["cuda", "cute-dsl"],
        "9.0": ["cuda", "cute-dsl"],
        "10.0": ["cuda", "cute-dsl"],
        "10.3": ["cuda", "cute-dsl"],
        "10.7": ["cuda", "cute-dsl"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
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
        "10.7": ["cute-dsl"],
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
        "10.7": ["cute-dsl"],
        "12.0": ["cute-dsl"],
        "12.1": ["cute-dsl"],
    },
    # fused_qk_rmsnorm_rope: CC check done programmatically via
    # fused_qk_rmsnorm_rope.is_compute_capability_supported() in the benchmark.
    # QUANTIZATION
    "mxfp8_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl", "batch_invariant"],
        "10.3": ["cuda", "cute-dsl", "batch_invariant"],
        "10.7": ["cuda", "cute-dsl", "batch_invariant"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
    },
    "mxfp4_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl", "batch_invariant"],
        "10.3": ["cuda", "cute-dsl", "batch_invariant"],
        "10.7": ["cuda", "batch_invariant"],
        "12.0": ["cuda", "cute-dsl"],
        "12.1": ["cuda", "cute-dsl"],
    },
    "nvfp4_quantize": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cuda", "cute-dsl", "batch_invariant"],
        "10.3": ["cuda", "cute-dsl", "batch_invariant"],
        "10.7": ["cuda", "cute-dsl", "batch_invariant"],
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
        "10.7": ["cuda"],
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
        "10.0": ["cuda", "batch_invariant"],
        "10.3": ["cuda", "batch_invariant"],
        "10.7": ["cuda", "batch_invariant"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    "log_softmax": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["batch_invariant"],
        "10.3": ["batch_invariant"],
        "10.7": ["batch_invariant"],
        "12.0": [],
        "12.1": [],
    },
    "log_probs": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["batch_invariant"],
        "10.3": ["batch_invariant"],
        "10.7": ["batch_invariant"],
        "12.0": [],
        "12.1": [],
    },
    "sampling_from_probs": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # Note: top_k_varlen uses its @backend_requirement support checks
    # (top_k_varlen.is_backend_supported) to filter backends, so it is not listed here.
    # ROPE
    "apply_rope": {
        "7.5": ["cuda"],
        "8.0": ["cuda"],
        "8.6": ["cuda"],
        "8.9": ["cuda"],
        "9.0": ["cuda"],
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda"],
        "12.0": ["cuda"],
        "12.1": ["cuda"],
    },
    # MAMBA
    "causal_conv1d_fn": {
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
    },
    "causal_conv1d_update": {
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
    },
    "selective_state_update": {
        "7.5": ["flashinfer", "triton"],
        "8.0": ["flashinfer", "triton"],
        "8.6": ["flashinfer", "triton"],
        "8.9": ["flashinfer", "triton"],
        "9.0": ["flashinfer", "triton"],
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
        "10.7": ["flashinfer", "triton"],
        "11.0": ["flashinfer", "triton"],
        "12.0": ["flashinfer", "triton"],
        "12.1": ["flashinfer", "triton"],
    },
    # GDN (Gated Delta Net)
    "gated_delta_rule_decode": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["flashinfer", "cute", "triton"],
        "10.0": ["flashinfer", "cute", "cake_gdn", "triton"],
        "10.3": ["flashinfer", "cute", "cake_gdn", "triton"],
        "10.7": ["flashinfer", "cute", "cake_gdn", "triton", "rubin", "rubin_ugpu"],
        "11.0": ["triton"],
        "12.0": ["triton"],
        "12.1": ["triton"],
    },
    "gated_delta_rule_mtp": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["flashinfer", "triton"],
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
        "10.7": ["flashinfer", "triton"],
        "11.0": ["triton"],
        "12.0": ["triton"],
        "12.1": ["triton"],
    },
    "chunk_gated_delta_rule": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": ["flashinfer", "fla"],
        "10.0": ["flashinfer", "fla"],
        "10.3": ["flashinfer", "fla"],
        "10.7": ["flashinfer"],
        "11.0": [],
        "12.0": [],
        "12.1": [],
    },
    # KDA prefill on SM120a only. The SM100-family Cake prefill backend is a
    # different kernel with a different contract and is not benchmarked here;
    # listing it under 10.0/10.3 would put two unrelated implementations in one
    # column.
    "recurrent_kda_prefill": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["flashinfer"],
        "10.3": [],
        "10.7": [],
        "11.0": [],
        "12.0": [
            "flashinfer",
            "flashinfer-decomp",
            "flashinfer-fused",
            "cutekda",
            "flash-kda",
        ],
        "12.1": [],
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


def probe_backends(backends, run_backend, *inputs, only=None):
    """Run each backend once (those in ``only`` when given) and drop the ones that raise, printing the reason.

    The batch-invariant classes (``batch_invariant=True``) raise ``BatchInvariantUnsupportedError`` instead
    of falling back when the shape, dtype or device is outside their contract; the routines that filter
    their backends by the compute-capability table probe that backend here before timing it.
    """
    for backend in list(backends):
        if only is not None and backend not in only:
            continue
        try:
            run_backend(backend, *inputs)
        except Exception as e:
            print(
                f"[INFO] {backend} backend does not support this configuration: {type(e).__name__}: {e}"
            )
            backends.remove(backend)
    return backends


def _tensor_bits(t: torch.Tensor) -> torch.Tensor:
    """The tensor's bit pattern as an integer tensor of the same width (NaN payloads compare as bits)."""
    int_dtype = {
        1: torch.uint8,
        2: torch.int16,
        4: torch.int32,
        8: torch.int64,
    }[t.element_size()]
    return t.contiguous().view(int_dtype)


def check_batch_invariance(
    label,
    run_rows,
    num_rows,
    reference,
    row_dim=0,
    verbose=0,
):
    """Bitwise batch-invariance check of a ``batch_invariant`` backend's output.

    The batch-invariant mode (``batch_invariant=True``) promises that a row's output bits depend only on
    that row's inputs, never on the batch it is computed in. ``--refcheck`` therefore checks the bits, not a
    tolerance: ``run_rows(rows)`` runs the operator on the sub-batch made of the rows ``rows`` (an int64 index
    tensor on the device) and returns the output -- a tensor, or a tuple of tensors -- with the rows on
    ``row_dim``; ``reference`` is the output of the whole batch (``rows = arange(num_rows)``) in the same form.

    Checks: run-to-run (the whole batch again), a sub-batch (a contiguous slice of rows alone equals those
    rows of the whole batch), a row alone (one row by itself equals its row of the whole batch) and a
    permutation (the rows in another order equal the whole batch's rows in that order).

    Raises AssertionError on the first mismatch, whatever ``--allow_output_mismatch`` says: a differing
    bit is a broken contract, not a tolerance miss. Returns the names of the checks that ran.
    """

    def as_tuple(x):
        return tuple(x) if isinstance(x, (tuple, list)) else (x,)

    def select(ts, rows):
        return tuple(t.index_select(row_dim, rows) for t in ts)

    def compare(check, out, expected):
        out = as_tuple(out)
        if len(out) != len(expected):
            raise AssertionError(
                f"[{label}] batch-invariance check '{check}': {len(out)} outputs, expected {len(expected)}"
            )
        for i, (o, e) in enumerate(zip(out, expected, strict=True)):
            if tuple(o.shape) != tuple(e.shape) or o.dtype != e.dtype:
                raise AssertionError(
                    f"[{label}] batch-invariance check '{check}': output {i} is {tuple(o.shape)} {o.dtype}, "
                    f"the whole batch's rows are {tuple(e.shape)} {e.dtype}"
                )
            ob, eb = _tensor_bits(o), _tensor_bits(e)
            if not torch.equal(ob, eb):
                num_diff = int((ob != eb).sum().item())
                raise AssertionError(
                    f"[{label}] batch-invariance check '{check}' FAILED: output {i} differs from the whole "
                    f"batch's rows in {num_diff} / {ob.numel()} elements ({num_diff / ob.numel() * 100:.4f}%)"
                )

    ref = as_tuple(reference)
    if num_rows <= 0:
        return []
    device = ref[0].device
    all_rows = torch.arange(num_rows, device=device, dtype=torch.int64)
    checks = []

    compare("run-to-run", run_rows(all_rows), ref)
    checks.append("run-to-run")

    if num_rows >= 2:
        lo = num_rows // 4
        hi = min(num_rows, max(lo + 1, lo + (num_rows + 1) // 2))
        rows = all_rows[lo:hi]
        compare(f"sub-batch rows [{lo}, {hi})", run_rows(rows), select(ref, rows))
        checks.append(f"sub-batch [{lo}, {hi})")

        r = num_rows // 2
        rows = all_rows[r : r + 1]
        compare(f"row {r} alone", run_rows(rows), select(ref, rows))
        checks.append(f"row {r} alone")

        gen = torch.Generator(device="cpu").manual_seed(0)
        perm = torch.randperm(num_rows, generator=gen).to(device)
        compare("permutation", run_rows(perm), select(ref, perm))
        checks.append("permutation")

    if verbose >= 1:
        print(
            f"[REFCHECK] {label}: batch-invariance checks passed bit for bit ({', '.join(checks)})"
        )
    return checks
