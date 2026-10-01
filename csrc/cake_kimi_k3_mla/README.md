# CAKE Kimi-K3 MLA FP8 paged attention (SM100 / SM103)

Generated CUDA for `flashinfer.mla.cake_kimi_k3_mla` (`trtllm_batch_decode_with_kv_cache_mla(...,
backend="cake")`): Kimi-K3 multi-head latent attention over an FP8 (E4M3) paged latent cache
(latent 512 + rope 64, page size 64), FP8 query, BF16 output.

One source per program, compiled for the architecture the device runs; a kernel whose two
architecture lowerings do not fold keeps one program per architecture:

* `main_rt16` -> `cake_kimi_k3_mla_fp8_paged_attention_f2ed3823ac88848234c4` (sm_100a, sm_103a)
* `main_rt32` -> `cake_kimi_k3_mla_fp8_paged_attention_f6f8a85d05a12ec611f2` (sm_100a, sm_103a)
* `main_rt48` -> `cake_kimi_k3_mla_fp8_paged_attention_3ca2519f90933160bbcb` (sm_100a, sm_103a)
* `main_rt64` -> `cake_kimi_k3_mla_fp8_paged_attention_abc850f024ed156e98ca` (sm_100a, sm_103a)
* `main_rt96` -> `cake_kimi_k3_mla_fp8_paged_attention_a1945d8ab3d522beb089` (sm_100a, sm_103a)
* `main_wide` -> `cake_kimi_k3_mla_fp8_paged_attention_5e6f67c2c89c6eb24061` (sm_100a), `cake_kimi_k3_mla_fp8_paged_attention_e1bdfbc717f1844c99a8` (sm_103a)
* `reduce_cta` -> `cake_kimi_k3_mla_fp8_paged_attention_037deeabf96bd0d2d769` (sm_100a, sm_103a)
* `reduce_w1` -> `cake_kimi_k3_mla_fp8_paged_attention_977c0b337b3f4232a99f` (sm_100a, sm_103a)
* `reduce_w2` -> `cake_kimi_k3_mla_fp8_paged_attention_b1c5eb5cb28aa248c2a9` (sm_100a, sm_103a)
* `reduce_w4` -> `cake_kimi_k3_mla_fp8_paged_attention_213d59cceca7f0a817c0` (sm_100a, sm_103a)

Per-architecture programs:

* `main_wide`: its sm_103a variant computes the row max with the TMEM reduction load (`tcgen05.ld.red`) and the sm_100a variant with a three-input max tree; the two lowerings differ in too many statements to fold under architecture guards, so the wide decode kernel keeps one program per architecture.

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
