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
    ],
    "attention": [
        "page_size",
        "batch_size",
        "s_qo",
        "s_kv",
        "num_qo_heads",
        "num_kv_heads",
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
        "m",
        "n",
        "k",
        "group_size",
        "tile_size",
        "scale_major_mode",
        "out_dtype",
        "mma_sm",
        "use_128x4_sf_layout",
        "use_nvfp4",
    ],
    "moe": [
        "num_tokens",
        "hidden_size",
        "intermediate_size",
        "num_experts",
        "top_k",
        "n_group",
        "topk_group",
        "routed_scaling_factor",
        "local_expert_offset",
        "local_num_experts",
        "tile_tokens_dim",
        "routing_method",
        "use_shuffled_weight",
        "weight_layout",
        "use_routing_bias",
        "use_routing_scales_on_input",
        "input_dtype",
        "weight_dtype",
        "gated_act",
        # CUTLASS fused MoE specific
        "cutlass_variant",
        "quantized_input",
        "tp_size",
        "tp_rank",
        "ep_size",
        "ep_rank",
    ],
    "general": [
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
        "mm_fp4",
    ],
    "moe": [
        "trtllm_fp4_block_scale_moe",
        "trtllm_fp8_block_scale_moe",
        "trtllm_fp8_per_tensor_scale_moe",
        "cutlass_fused_moe",
    ],
}


def print_perf_metrics(backend, median_time, std_time, tflops, tb_per_sec):
    output_backend_width = 15
    print(
        f"[PERF] {backend.ljust(output_backend_width)[:output_backend_width]}:: median time {median_time:.3f} ms; std {std_time:.3f} ms; achieved tflops {tflops:.3f} TFLOPs/sec; achieved tb_per_sec {tb_per_sec:.3f} TB/sec"
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
    else:
        raise ValueError(f"Unsupported dtype: {dtype_str}")


routine_cc_to_supported_backends = {
    # ATTENTION
    "BatchDecodeWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache
        "7.5": ["fa2"],
        "8.0": ["fa2", "fa2_tc", "cudnn"],
        "8.6": ["fa2", "fa2_tc", "cudnn"],
        "8.9": ["fa2", "fa2_tc", "cudnn"],
        "9.0": ["fa2", "fa2_tc", "cudnn", "trtllm-native"],
        "10.0": ["fa2", "fa2_tc", "cudnn", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "fa2_tc", "cudnn", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "fa2_tc", "cudnn", "trtllm-native"],
    },
    "BatchPrefillWithPagedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_context_with_kv_cache
        "7.5": [],
        "8.0": ["fa2", "cudnn"],
        "8.6": ["fa2", "cudnn"],
        "8.9": ["fa2", "cudnn"],
        "9.0": ["fa2", "fa3", "cudnn"],
        "10.0": ["fa2", "cudnn", "trtllm-gen", "trtllm-native"],
        "10.3": ["fa2", "cudnn", "trtllm-gen", "trtllm-native"],
        "12.0": ["fa2", "cudnn"],
    },
    "BatchPrefillWithRaggedKVCacheWrapper": {
        # NOTE: trtllm-native calls trtllm_ragged_attention_deepseek
        "7.5": [],
        "8.0": ["fa2", "cudnn"],
        "8.6": ["fa2", "cudnn"],
        "8.9": ["fa2", "cudnn"],
        "9.0": ["fa2", "fa3", "cudnn"],
        "10.0": ["fa2", "cudnn", "cutlass", "trtllm-native"],
        "10.3": ["fa2", "cudnn", "cutlass", "trtllm-native"],
        "12.0": ["fa2", "cudnn"],
    },
    "BatchMLAPagedAttentionWrapper": {
        # NOTE: trtllm-native calls trtllm_batch_decode_with_kv_cache_mla
        "7.5": [],
        "8.0": ["fa2"],
        "8.6": ["fa2"],
        "8.9": ["fa2"],
        "9.0": ["fa2", "fa3"],
        "10.0": ["fa2", "cutlass", "trtllm-native"],
        "10.3": ["fa2", "cutlass", "trtllm-native"],
        "12.0": ["fa2"],
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
    },
    "mm_fp4": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cudnn", "trtllm", "cutlass"],
        "10.3": ["cudnn", "trtllm", "cutlass"],
        "12.0": ["cudnn", "cutlass"],
        "12.1": ["cudnn", "cutlass"],
    },
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
    },
    "cutlass_fused_moe": {
        "7.5": [],
        "8.0": [],
        "8.6": [],
        "8.9": [],
        "9.0": [],
        "10.0": ["cutlass"],
        "10.3": ["cutlass"],
        "12.0": [],
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
