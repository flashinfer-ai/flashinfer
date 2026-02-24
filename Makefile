bench-mla-decode:
	python benchmarks/flashinfer_benchmark.py \
	--routine BatchMLAPagedAttentionWrapper \
	--batch_size 1024 \
	--s_kv 8192 \
	--num_qo_heads 32 \
	--num_kv_heads 1 \
	--head_dim_ckv 256 \
	--head_dim_kpe 64 \
	--page_size 64 \
	--backends trtllm-native \
	--q_dtype bfloat16 \
	--kv_dtype bfloat16 \
	--s_qo 1 \
	--num_iters 500

bench-mla-prefill-bf16:
	python benchmarks/flashinfer_benchmark.py \
	--routine BatchPrefillWithRaggedKVCacheWrapper \
	--batch_size 2 \
	--s_kv 8192 \
	--s_qo 8192 \
	--num_qo_heads 128 \
	--num_kv_heads 128 \
	--head_dim_qk 128 \
	--head_dim_vo 128 \
	--page_size 64 \
	--backends trtllm-native \
	--q_dtype bfloat16 \
	--kv_dtype bfloat16 \
	--num_iters 100

bench-mla-prefill-fp8:
	python benchmarks/flashinfer_benchmark.py \
	--routine BatchPrefillWithRaggedKVCacheWrapper \
	--batch_size 2 \
	--s_kv 8192 \
	--s_qo 8192 \
	--num_qo_heads 128 \
	--num_kv_heads 128 \
	--head_dim_qk 128 \
	--head_dim_vo 128 \
	--page_size 64 \
	--backends trtllm-native \
	--q_dtype fp8_e4m3 \
	--kv_dtype fp8_e4m3 \
	--num_iters 100