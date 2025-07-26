# Output columns for the test results.
output_column_dict = {
    "perf": [
        "routine",
        "median_time",
        "std_time",
        "tflops",
        "tb_per_sec",
    ],
    "attention": [
        "backend",
        "page_size",
        "batch_size",
        "s_qo",
        "s_kv",
        "num_qo_heads",
        "num_kv_heads",
        "head_dim_qk",
        "head_dim_vo",
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
    ],
    "general": [
        "refcheck",
        "no_cuda_graph",
        "allow_output_mismatch",
        "random_seed",
    ],
}

full_output_columns = (
    output_column_dict["perf"]
    + output_column_dict["attention"]
    + output_column_dict["gemm"]
    + output_column_dict["general"]
)
