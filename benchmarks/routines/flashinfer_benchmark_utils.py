import argparse
import contextlib
import importlib

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
        "autotuned",
        "autotune_winner",
        "autotune_tactic",
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
        "quant_variant",
        "autotune",
        "tactic",
        "refcheck_passed",
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
    + output_column_dict["moe_comm"]
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


def warn_if_autotune_unsupported(args, routine_name):
    """Emit a one-shot warning if --autotune is set but the routine's library
    API has no autotuner. Call from the top of test functions whose underlying
    flashinfer API never consults the AutoTuner, so the flag is a no-op.
    """
    if getattr(args, "autotune", False):
        print(
            f"[WARNING] --autotune has no effect for routine {routine_name}: the library API has no autotuner; flag is ignored."
        )


def warn_if_outer_tuning_context():
    """Warn when an enclosing ``autotune(True)`` context is already open.

    The harness times outside its own tuning contexts. An outer tuning context
    (e.g. a process-wide ``autotune_v2(mode="tune")``) keeps the AutoTuner in
    tuning mode during CUDA-graph capture, so any op whose cache lookup misses
    for the timed call tries to profile inside the capture and raises.
    """
    try:
        from flashinfer.autotuner import AutoTuner

        tuning = AutoTuner.get().is_tuning_mode
    except Exception:
        return
    if tuning:
        print(
            "[WARNING] An enclosing autotune(True) context is active; timed runs execute in tuning mode, and CUDA-graph capture fails for any op whose tuned cache entries do not cover the timed call."
        )


@contextlib.contextmanager
def record_autotune_choices():
    """Record the (runner, tactic) pairs ``AutoTuner.choose_one`` returns.

    Yields a dict mapping each ``custom_op`` seen inside the block to the last
    ``(runner_class_name, tactic)`` returned for it. The dict stays empty when
    the autotuner cannot be instrumented.
    """
    choices = {}
    try:
        from flashinfer.autotuner import AutoTuner

        tuner = AutoTuner.get()
        original = tuner.choose_one
    except Exception:
        yield choices
        return

    def choose_one(custom_op, runners, *args, **kwargs):
        result = original(custom_op, runners, *args, **kwargs)
        try:
            runner, tactic = result
            choices[custom_op] = (type(runner).__name__, tactic)
        except Exception:
            pass
        return result

    previous = tuner.__dict__.get("choose_one")
    tuner.choose_one = choose_one
    try:
        yield choices
    finally:
        if previous is None:
            tuner.__dict__.pop("choose_one", None)
        else:
            tuner.choose_one = previous


def probe_autotune_choices(fn, *args, **kwargs):
    """Run ``fn`` once eagerly and return the autotuner choices it made.

    Call under the same autotune context as the timed run so the recorded
    choices are the ones the timed run uses. Returns an empty dict if ``fn``
    raises; the timed run surfaces the error.
    """
    with record_autotune_choices() as choices:
        try:
            fn(*args, **kwargs)
        except Exception:
            return {}
    return dict(choices)


def _format_tactic(tactic):
    return repr(tactic).replace(" ", "")


def format_autotune_choices(choices):
    """Return ``(autotune_winner, autotune_tactic)`` column strings.

    ``autotune_winner`` is ``op=RunnerClass`` and ``autotune_tactic`` is
    ``op=tactic``, one entry per op, ``;``-separated (op names may contain
    ``::``).
    """
    ops = sorted(choices)
    winner = ";".join(f"{op}={choices[op][0]}" for op in ops)
    tactic = ";".join(f"{op}={_format_tactic(choices[op][1])}" for op in ops)
    return winner, tactic


_RUNNER_BACKEND_KEYWORDS = (
    ("cutedsl", "cute-dsl"),
    ("cute_dsl", "cute-dsl"),
    ("cublaslt", "cublaslt"),
    ("cublas", "cublas"),
    ("cudnn", "cudnn"),
    ("cutlass", "cutlass"),
    ("tgv", "tgv"),
    ("tinygemm", "tinygemm"),
    ("trtllm", "trtllm"),
    ("b12x", "b12x"),
    ("cutile", "cutile"),
)


def backend_from_runner_name(runner_name):
    """Map an AutoTuner runner class name to a harness backend label, or ""."""
    lowered = runner_name.lower()
    for keyword, backend in _RUNNER_BACKEND_KEYWORDS:
        if keyword in lowered:
            return backend
    return ""


def resolve_backend_from_choices(choices):
    """Return the single backend all recorded runners map to, or ""."""
    backends = {backend_from_runner_name(runner) for runner, _ in choices.values()}
    if len(backends) == 1:
        return backends.pop()
    return ""


def set_autotune_columns(cur_res, autotuned, choices=None):
    """Fill the ``autotuned`` / ``autotune_winner`` / ``autotune_tactic`` columns.

    Winner and tactic are only reported for tuned rows.
    """
    cur_res["autotuned"] = bool(autotuned)
    if autotuned and choices:
        cur_res["autotune_winner"], cur_res["autotune_tactic"] = (
            format_autotune_choices(choices)
        )


def print_autotune_choices(backend, choices):
    for op in sorted(choices):
        runner, tactic = choices[op]
        print(
            f"[INFO] {backend} autotune choice: op={op} runner={runner} tactic={_format_tactic(tactic)}"
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


# Maps each benchmark backend name of a routine to the FlashInfer API that
# carries the support metadata for it (``@backend_requirement`` or
# ``@supported_compute_capability``) as (module, attribute, library backend).
# The library backend is the name passed to ``<api>.is_backend_supported``;
# ``None`` means the API has a single implicit backend and is checked with
# ``<api>.is_compute_capability_supported``, as is ``"auto"`` (supported when
# any of the API's backends supports the compute capability). A benchmark
# backend that is not
# listed for a routine here is not implemented by the benchmark harness.
routine_backend_to_library_api = {
    # GEMM
    "gemm_fp8_nt_groupwise": {
        "cutlass": ("flashinfer.gemm", "gemm_fp8_nt_groupwise", "cutlass"),
        "trtllm": ("flashinfer.gemm", "gemm_fp8_nt_groupwise", "trtllm"),
        "cutile": ("flashinfer.gemm", "gemm_fp8_nt_groupwise", "cutile"),
    },
    "group_gemm_fp8_nt_groupwise": {
        "cutlass": ("flashinfer.gemm", "group_gemm_fp8_nt_groupwise", None),
    },
    "bmm_mxfp8": {
        "cudnn": ("flashinfer.gemm", "bmm_mxfp8", "cudnn"),
    },
    "mm_mxfp8": {
        "cutlass": ("flashinfer.gemm", "mm_mxfp8", "cutlass"),
        "cute-dsl": ("flashinfer.gemm", "mm_mxfp8", "cute-dsl"),
        "trtllm": ("flashinfer.gemm", "mm_mxfp8", "trtllm"),
        "cudnn": ("flashinfer.gemm", "mm_mxfp8", "cudnn"),
        "auto": ("flashinfer.gemm", "mm_mxfp8", "auto"),
    },
    "tinygemm_bf16": {
        "tinygemm": ("flashinfer.gemm", "tinygemm_bf16", None),
    },
    # MOE
    "cute_dsl_fp4_block_scale_moe": {
        "cute-dsl": ("flashinfer", "cute_dsl_fused_moe", None),
    },
    "cute_dsl_bf16_moe": {
        "cute-dsl": ("flashinfer", "cute_dsl_fused_moe_bf16", None),
    },
    "b12x_fused_moe": {
        "b12x": ("flashinfer", "b12x_fused_moe", None),
    },
    "alphamoe_nvfp4_aligned_moe": {
        "alphamoe": ("flashinfer.fused_moe", "alphamoe_nvfp4_aligned_moe", None),
    },
}

# Fallback for routines whose FlashInfer API does not expose support metadata.
# Only routines missing from routine_backend_to_library_api belong here; once
# the library API gains ``@backend_requirement`` /
# ``@supported_compute_capability``, move the routine to
# routine_backend_to_library_api and delete its rows. Routines that query the
# library directly in their benchmark (bmm_fp8, mm_fp8, mm_fp4, mm_bf16,
# bmm_bf16, fused_qk_rmsnorm_rope, top_k_varlen) are listed in neither.
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
        "10.7": ["fa2", "auto", "cudnn", "cudnn-native", "trtllm-gen", "trtllm-native"],
        "12.0": [
            "fa2",
            "auto",
            "cudnn",
            "cudnn-native",
            "trtllm-fmha-v2",
            "cute-dsl-prims",
        ],
        "12.1": ["fa2", "auto", "cudnn", "cudnn-native"],
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
        "12.1": ["fa2", "cudnn", "cudnn-native"],
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
        "10.7": ["fa2", "cutlass", "trtllm-native"],
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
    # MoELayer cross-backend NVFP4: intersection of CuteDSL + TRTLLM FP4 support.
    # SM100 only (Blackwell); unlisted archs fall through to [] (skipped).
    "unified_nvfp4_moe": {
        "10.0": ["unified"],
        "10.3": ["unified"],
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
        "10.0": ["cuda", "cute-dsl"],
        "10.3": ["cuda", "cute-dsl"],
        "10.7": ["cuda", "cute-dsl"],
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
        "10.7": ["cuda"],
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
        "10.7": ["cuda", "cute-dsl"],
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
        "10.0": ["cuda"],
        "10.3": ["cuda"],
        "10.7": ["cuda"],
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
        "9.0": ["flashinfer", "triton"],
        "10.0": ["flashinfer", "triton"],
        "10.3": ["flashinfer", "triton"],
        "10.7": ["flashinfer", "triton"],
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


def _resolve_library_api(module_name, attr):
    api = importlib.import_module(module_name)
    for part in attr.split("."):
        api = getattr(api, part)
    return api


def get_backend_support(routine, backend, cc):
    """Return whether ``backend`` of ``routine`` can run on compute capability ``cc``.

    ``cc`` is an int encoded as ``major * 10 + minor`` (e.g. 107 for SM 10.7),
    matching what FlashInfer's support metadata expects. Returns a tuple
    ``(supported, message)``. ``message`` names the source of a negative
    answer (FlashInfer's support metadata, the benchmark harness, or the
    fallback table); for a positive answer it is ``None`` unless there is
    something to report.
    """
    cc_str = f"{cc // 10}.{cc % 10}"
    if routine in routine_backend_to_library_api:
        api_spec = routine_backend_to_library_api[routine].get(backend)
        if api_spec is None:
            return (
                False,
                f"{backend} for routine {routine} is not implemented by the benchmark harness",
            )
        module_name, attr, lib_backend = api_spec
        api_name = f"{module_name}.{attr}"
        try:
            api = _resolve_library_api(module_name, attr)
        except (ImportError, AttributeError) as e:
            return (
                False,
                f"{backend} for routine {routine} cannot be checked: {api_name} is unavailable ({type(e).__name__}: {e})",
            )
        if lib_backend is None or lib_backend == "auto":
            check = getattr(api, "is_compute_capability_supported", None)
            query = f"{api_name}.is_compute_capability_supported({cc})"
            args = (cc,)
        else:
            check = getattr(api, "is_backend_supported", None)
            query = f"{api_name}.is_backend_supported({lib_backend!r}, {cc})"
            args = (lib_backend, cc)
        if check is None:
            return (
                True,
                f"{api_name} exposes no support metadata; {backend} for routine {routine} is not filtered by compute capability",
            )
        if check(*args):
            return True, None
        return (
            False,
            f"FlashInfer reports {backend} for routine {routine} as unsupported on compute capability {cc_str} ({query} is False)",
        )

    cc_to_supported_backends = routine_cc_to_supported_backends[routine]
    if cc_str not in cc_to_supported_backends:
        return (
            False,
            f"{backend} for routine {routine} is not supported on compute capability {cc_str}: the benchmark's fallback support table has no entry for it",
        )
    if backend not in cc_to_supported_backends[cc_str]:
        return (
            False,
            f"{backend} for routine {routine} is not listed for compute capability {cc_str} in the benchmark's fallback support table",
        )
    return True, None


def filter_backends_by_compute_capability(backends, routine, device):
    major, minor = get_compute_capability(device)
    cc = major * 10 + minor
    supported_backends = []
    for backend in backends:
        supported, message = get_backend_support(routine, backend, cc)
        if supported:
            supported_backends.append(backend)
            if message:
                print(f"[INFO] {message}.")
        else:
            print(f"[WARNING] {message}. Skipping.")
    backends[:] = supported_backends
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
