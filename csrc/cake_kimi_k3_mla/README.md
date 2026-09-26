# CAKE Kimi-K3 MLA FP8 paged attention (SM100 / SM103)

Generated CUDA for `flashinfer.mla.cake_kimi_k3_mla` (`trtllm_batch_decode_with_kv_cache_mla(...,
backend="cake")`): Kimi-K3 multi-head latent attention over an FP8 (E4M3) paged latent cache
(latent 512 + rope 64, page size 64), FP8 query, BF16 output.  Two attention schedules: one
swapped-AB tcgen05 kernel per row tile (`main_rt16` .. `main_rt96`: packed (token, head) rows per
CTA) and the two-CTA wide kernel (`main_wide`: 128 packed rows per cluster of two CTAs, K tokens
split across the pair; taken for requests with more than 64 packed rows whose longest KV is at
least 16384 tokens), plus the split-KV merge kernels shared by both (`reduce_w1` / `reduce_w2` /
`reduce_w4` warp-per-row reducers for `num_split <= 32`, `reduce_cta` otherwise).  The wide
kernel is a cluster kernel; its generated binding sets the cluster launch attribute (2, 1, 1).

* Source of truth: Cake `loom/examples/weave/kimi_k3_mla_fp8_paged_attention.py` (row tiles, host
  planner) and `loom/examples/weave/kimi_k3_mla_wide.py` (wide kernel) (Linear CAKE-645).
* Exporter: Cake `exports/kimi_k3_mla_fp8_paged_attention/export.py` writes `sm_100a/` and
  `sm_103a/` (`*_kernel.cu` + `*_binding.cu` per physical module) and fills `MODULES` / `ROUTES` in
  `flashinfer/jit/cake_kimi_k3_mla.py`.
* Do not edit the generated files by hand; regenerate through the exporter.
