# Trace Reference Correctness Standards

This table records the unit-test source and the old
`tests/trace/test_reference_correctness.py` correctness standard preserved by
the split reference tests. Every listed file currently collects at least two
shape cases.

| Reference test file | Unit-test source | Old trace correctness standard |
| --- | --- | --- |
| `test_append_paged_kv_cache_reference_correctness.py` | `tests/attention/test_page.py` | Exact updated K/V cache comparison: `atol=0`, `rtol=0`. |
| `test_append_paged_mla_kv_cache_reference_correctness.py` | `tests/attention/test_mla_page.py` | Exact updated CKV/KPE cache comparison: `atol=0`, `rtol=0`. |
| `test_apply_llama31_rope_inplace_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_llama31_rope_pos_ids_inplace_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_llama31_rope_pos_ids_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_llama31_rope_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_inplace_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_pos_ids_inplace_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_pos_ids_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_with_cos_sin_cache_inplace_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_apply_rope_with_cos_sin_cache_reference_correctness.py` | `tests/attention/test_rope.py` | Q and K close with `_ROPE_TOL`: `atol=1e-2`, `rtol=1e-2`. |
| `test_batch_attention_run_reference_correctness.py` | `tests/attention/test_batch_attention.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_batch_pod_run_reference_correctness.py` | `tests/utils/test_pod_kernels.py` | Decode output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_block_sparse_run_reference_correctness.py` | `tests/attention/test_block_sparse.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_bmm_bf16_reference_correctness.py` | `tests/gemm/test_bmm_bf16.py` | Cosine similarity: `cos_sim > 0.99`. |
| `test_bmm_fp8_reference_correctness.py` | `tests/gemm/test_bmm_fp8.py` | Cosine similarity: `cos_sim > 0.99`. |
| `test_chain_speculative_sampling_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_concat_mla_k_reference_correctness.py` | `tests/utils/test_concat_mla.py` | Exact in-place concat output: `atol=0`, `rtol=0`. |
| `test_cudnn_batch_decode_reference_correctness.py` | `tests/attention/test_cudnn_decode.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_cudnn_batch_prefill_reference_correctness.py` | `tests/attention/test_cudnn_prefill.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_cutlass_fused_moe_reference_correctness.py` | `tests/moe/test_trtllm_cutlass_fused_moe.py` | Output close after reference dtype cast: `atol=1e-2`, `rtol=1e-2`. |
| `test_fp4_quantize_round_trip_reference_correctness.py` | `tests/utils/test_fp4_quantize.py` | Packed dtype/shape checks plus FP4 round-trip mean relative error `< 0.5`. |
| `test_fused_add_rmsnorm_quant_reference_correctness.py` | `tests/utils/test_norm.py` | Residual close `1e-3/1e-3`; dequantized quant output close `1.0/1.0`. |
| `test_fused_add_rmsnorm_reference_correctness.py` | `tests/utils/test_norm.py` | Normalized output and residual close: `atol=1e-3`, `rtol=1e-3`. |
| `test_gelu_and_mul_reference_correctness.py` | `tests/utils/test_activation.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_gelu_tanh_and_mul_reference_correctness.py` | `tests/utils/test_activation.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_gemma_fused_add_rmsnorm_reference_correctness.py` | `tests/utils/test_norm.py` | Normalized output and residual close: `atol=1e-3`, `rtol=1e-3`. |
| `test_gemma_rmsnorm_reference_correctness.py` | `tests/utils/test_norm.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_layernorm_reference_correctness.py` | `tests/utils/test_norm.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_merge_state_in_place_reference_correctness.py` | `tests/attention/test_shared_prefix_kernels.py` | V and S close: `atol=1e-3`, `rtol=1e-3`. |
| `test_merge_state_reference_correctness.py` | `tests/attention/test_shared_prefix_kernels.py` | V and S close: `atol=1e-3`, `rtol=1e-3`. |
| `test_merge_states_reference_correctness.py` | `tests/attention/test_shared_prefix_kernels.py` | V and S close: `atol=1e-3`, `rtol=1e-3`. |
| `test_min_p_sampling_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_mla_rope_quantize_fp8_reference_correctness.py` | `tests/attention/test_rope.py` | Q/K rope/nope FP8 outputs close: `atol=1e-2`, `rtol=2e-1`. |
| `test_mm_bf16_reference_correctness.py` | `tests/gemm/test_mm_bf16.py` | Cosine similarity: `cos_sim > 0.99`. |
| `test_multi_level_cascade_run_reference_correctness.py` | `tests/attention/test_batch_decode_kernels.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_mxfp4_quantize_reference_correctness.py` | `tests/moe/test_trtllm_cutlass_fused_moe.py` | Dequantized round-trip close: `atol=2.0`, `rtol=0.25`. |
| `test_mxfp8_quantize_reference_correctness.py` | `tests/gemm/test_mm_mxfp8.py` | Absolute-value mean close: `atol=2.0`, `rtol=0.5`. |
| `test_nvfp4_quantize_reference_correctness.py` | `tests/gemm/test_group_gemm_fp4.py` | Packed byte mismatch fraction `< 0.05`. |
| `test_pod_with_paged_kv_cache_run_reference_correctness.py` | `tests/utils/test_pod_kernels.py` | Prefill and decode outputs close: `atol=1e-3`, `rtol=1e-3`. |
| `test_rmsnorm_quant_reference_correctness.py` | `tests/utils/test_norm.py` | Dequantized output close: `atol=1.0`, `rtol=1.0`. |
| `test_rmsnorm_reference_correctness.py` | `tests/utils/test_norm.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_rope_quantize_fp8_append_paged_kv_cache_reference_correctness.py` | `tests/attention/test_rope.py` | Portable Q rope/nope FP8 outputs close: `atol=1e-2`, `rtol=2e-1`. |
| `test_rope_quantize_fp8_reference_correctness.py` | `tests/attention/test_rope.py` | Q/K rope/nope FP8 outputs close: `atol=1e-2`, `rtol=2e-1`. |
| `test_sampling_from_logits_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_sampling_from_probs_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_segment_gemm_run_reference_correctness.py` | `tests/gemm/test_group_gemm.py` | Output close: `atol=2e-3`, `rtol=1e-3`. |
| `test_silu_and_mul_reference_correctness.py` | `tests/utils/test_activation.py` | Output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_single_decode_reference_correctness.py` | `tests/attention/test_batch_decode_kernels.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_single_prefill_reference_correctness.py` | `tests/attention/test_single_prefill.py` | Output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_softmax_reference_correctness.py` | `tests/utils/test_sampling.py` | Probability output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_tgv_gemm_sm100_reference_correctness.py` | `tests/gemm/test_tgv_gemm.py` | Cosine similarity: `cos_sim > 0.99`. |
| `test_top_k_mask_logits_reference_correctness.py` | `tests/utils/test_sampling.py` | Finite mask positions exact, finite logits close: `atol=1e-3`, `rtol=1e-3`. |
| `test_top_k_renorm_probs_reference_correctness.py` | `tests/utils/test_sampling.py` | Probability output close: `atol=1e-3`, `rtol=1e-3`. |
| `test_top_k_sampling_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_top_k_top_p_sampling_from_logits_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_top_k_top_p_sampling_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_top_p_renorm_probs_reference_correctness.py` | `tests/utils/test_sampling.py` | Probability output close: `atol=1e-2`, `rtol=5e-2`. |
| `test_top_p_sampling_reference_correctness.py` | `tests/utils/test_sampling.py` | Deterministic token IDs exact: `atol=0`, `rtol=0`. |
| `test_trtllm_batch_context_reference_correctness.py` | `tests/attention/test_trtllm_gen_attention.py`, `tests/attention/test_cudnn_prefill.py` | BF16 paged prefill output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_trtllm_batch_decode_mla_reference_correctness.py` | `tests/attention/test_cute_dsl_mla_decode.py`, `tests/attention/test_trtllm_gen_mla.py` | MLA decode output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_trtllm_batch_decode_reference_correctness.py` | `tests/attention/test_trtllm_gen_attention.py`, `tests/attention/test_cudnn_decode.py` | BF16 paged decode output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_trtllm_fmha_v2_prefill_reference_correctness.py` | `tests/attention/test_fmha_v2_prefill.py` | BF16 PACKED_QKV output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_var_block_sparse_run_reference_correctness.py` | `tests/attention/test_block_sparse.py` | Transposed output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_xqa_batch_decode_mla_reference_correctness.py` | `tests/attention/test_xqa_mla_batch_decode.py` | FP8 MLA pass-ratio: at least `0.95` within `atol=0.05`, `rtol=0.05`. |
| `test_xqa_batch_decode_reference_correctness.py` | `tests/attention/test_xqa_batch_decode.py` | BF16 paged decode output close: `atol=1e-2`, `rtol=1e-2`. |
| `test_xqa_mla_reference_correctness.py` | `tests/attention/test_xqa.py`, `tests/attention/test_xqa_mla_batch_decode.py` | FP8 MLA pass-ratio: at least `0.95` within `atol=0.05`, `rtol=0.05`. |
| `test_xqa_reference_correctness.py` | `tests/attention/test_xqa.py` | XQA pass-ratio: at least `0.98` within `atol=0.05`, `rtol=0.05`. |
