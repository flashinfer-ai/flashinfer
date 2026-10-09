# CAKE Kimi-K3 MLA FP8 paged attention (SM100 / SM103)

Generated CUDA for `flashinfer.mla.cake_kimi_k3_mla` (`trtllm_batch_decode_with_kv_cache_mla(...,
backend="cake")`): Kimi-K3 multi-head latent attention over an FP8 (E4M3) paged latent cache
(latent 512 + rope 64, page size 64), FP8 query, BF16 output.

One source per program, compiled for the architecture the device runs; a kernel whose two
architecture lowerings do not fold keeps one program per architecture:

* `main_rt16` -> `cake_kimi_k3_mla_fp8_paged_attention_b0d0c47e81edc26da19e` (sm_100a, sm_103a)
* `main_rt32` -> `cake_kimi_k3_mla_fp8_paged_attention_2e4e7903daf6e75c836c` (sm_100a, sm_103a)
* `main_rt48` -> `cake_kimi_k3_mla_fp8_paged_attention_03fb1f7512d6581b0273` (sm_100a, sm_103a)
* `main_rt64` -> `cake_kimi_k3_mla_fp8_paged_attention_29b047f732fcfdfeb85b` (sm_100a, sm_103a)
* `main_rt96` -> `cake_kimi_k3_mla_fp8_paged_attention_07d8fd7941e0ab779c08` (sm_100a, sm_103a)
* `main_wide` -> `cake_kimi_k3_mla_fp8_paged_attention_ea17187e76fd47ab2653` (sm_100a, sm_103a)
* `reduce_cta` -> `cake_kimi_k3_mla_fp8_paged_attention_9c450b8e82598779f1b2` (sm_100a, sm_103a)
* `reduce_w1` -> `cake_kimi_k3_mla_fp8_paged_attention_ab133c42d484f9421945` (sm_100a, sm_103a)
* `reduce_w2` -> `cake_kimi_k3_mla_fp8_paged_attention_9f53b9e3e16d64b967c6` (sm_100a, sm_103a)
* `reduce_w4` -> `cake_kimi_k3_mla_fp8_paged_attention_582ec28a9ab46c704912` (sm_100a, sm_103a)

Per-architecture programs:

* none: every kernel's two architecture lowerings fold into one source.

* `main_rt16` .. `main_rt96`: swapped-AB tcgen05 kernel, one per row tile (packed (token, head)
  rows per CTA; the smallest tile holding a request's rows is taken).
* `main_wide`: two-CTA cluster kernel (cluster dims (2, 1, 1)), 128 packed rows per cluster,
  K tokens split across the pair; taken for requests with more than 64 packed rows
  whose longest KV is at least 8192 tokens.
* `reduce_w1` / `reduce_w2` / `reduce_w4`: warp-per-row split-KV merge for `num_split <= 32`;
  `reduce_cta`: CTA merge otherwise.

The registry (`MODULES`, `KERNELS`) lives in `flashinfer/jit/cake_kimi_k3_mla.py`; the host plan
(route, split count, grids) in `flashinfer/mla/cake_kimi_k3_mla.py`.  These files are written by
the Cake exporter: regenerate them, do not edit them by hand.
