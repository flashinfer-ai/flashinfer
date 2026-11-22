# Source provenance

The SM90 push-style FP8 implementation was introduced in
[FlashInfer #4069](https://github.com/flashinfer-ai/flashinfer/pull/4069),
merged as `f9b13ef11472d994ec02210e7a6fe62df3254636` from PR head
`301f8ce3dd42646bb12707251f50db619fb5c653`.

FlashInfer maintains this implementation. CUDA sources are under `a2a/` and
`fp8_gemm/`; adapters are under `moe_ep/shims/sm90/push_style_megamoe/`.
Backend dispatch remains at
`backends/mega/kernel/sm90/fp8_fp8_bf16_push_cuda/`. These files do not have an
independent provider export to refresh.
