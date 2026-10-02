# Experimental fused QK RMSNorm, RoPE, and paged KV append

This JIT-only backend fuses packed-QKV unpacking, optional per-head Q/K
RMSNorm, NeoX RoPE, Q output production, and paged K/V cache append. The FP8
variant additionally quantizes Q/K/V in the same launch and supports dynamic
per-token/per-Q-head or static Q scaling.

It fills a narrow gap in current FlashInfer: the stable tree has QK
RMSNorm+RoPE and RoPE+FP8 paged-append primitives, but does not combine the
complete serving preprocessing path into one GQA launch.

Limitations:

- SM90 and SM100 only;
- BF16 packed input;
- Q/K/V head dimension 128;
- `(num_q_heads, num_kv_heads)` is `(8, 1)` or `(64, 8)`;
- NHD cache layout `[page, page_size, kv_head, head_dim]`;
- FP8 cache type is E4M3 and K/V scales are scalar;
- JIT only, no automatic routing and no AOT registration.

The implementation is derived from Tencent hpc-ops commit
`2a2e26562433a8ba4b504858f1c938eb7612c901` and retains its MIT attribution.
The experimental owner is [`@slhslh`](https://github.com/slhslh). Lifecycle
and graduation work is tracked in
[flashinfer-ai/flashinfer#5368](https://github.com/flashinfer-ai/flashinfer/issues/5368).

See [RESULTS.md](RESULTS.md) for B200 correctness and latency measurements.
