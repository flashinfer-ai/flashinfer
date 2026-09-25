# CAKE Kimi-K3 MLA FP8 paged attention (SM100 / SM103)

Generated CUDA for `flashinfer.mla.cake_kimi_k3_mla` (`trtllm_batch_decode_with_kv_cache_mla(...,
backend="cake")`): Kimi-K3 multi-head latent attention over an FP8 (E4M3) paged latent cache
(latent 512 + rope 64, page size 64), FP8 query, BF16 output.  One swapped-AB tcgen05 schedule
per row tile (`main_rt16` .. `main_rt96`: packed (token, head) rows per CTA) plus the split-KV
merge kernels (`reduce_w1` / `reduce_w2` / `reduce_w4` warp-per-row reducers for `num_split <= 32`,
`reduce_cta` otherwise).

* Source of truth: Cake `loom/examples/weave/kimi_k3_mla_fp8_paged_attention.py` (Linear CAKE-645).
* Exporter: Cake `exports/kimi_k3_mla_fp8_paged_attention/export.py` writes `sm_100a/` and
  `sm_103a/` (`*_kernel.cu` + `*_binding.cu` per physical module) and fills `MODULES` / `ROUTES` in
  `flashinfer/jit/cake_kimi_k3_mla.py`.
* Do not edit the generated files by hand; regenerate through the exporter.
