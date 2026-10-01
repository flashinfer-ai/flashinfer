# Vendored from SGLang

Do not edit the files under `sgl_kernel/` in FlashInfer; re-vendor them from upstream.
The pre-commit formatting hooks exclude `csrc/sglang_dsv4/sgl_kernel/` so the files stay
byte-diffable against upstream.

- upstream: https://github.com/sgl-project/sglang (Apache-2.0, Copyright SGLang Team;
  license text in `licenses/LICENSE.sglang.txt`)
- commit: 7f27bf4708
- date: 2026-08
- files (upstream path under `python/sglang/kernels/jit/` -> here):
  - `include/sgl_kernel/deepseek_v4/topk_impl.cuh` -> `sgl_kernel/deepseek_v4/topk_impl.cuh`
    (the selection classes: TopKRegister, TopKStreaming, the transform helpers)
  - `include/sgl_kernel/{math,tile,type,utils,vec,warp}.cuh` -> `sgl_kernel/`
  - `include/sgl_kernel/{utils,source_location}.h` -> `sgl_kernel/`
  - `csrc/deepseek_v4/topk_v2.cuh` -> the verbatim device code (`topk_ragged_kernel` and its
    helpers) at the top of `sglang_dsv4_topk.cu`; the paged / cluster / plan machinery of
    that file is intentionally not vendored

FlashInfer-specific, not vendored: the host launchers and `topk_varlen_kernel` in
`sglang_dsv4_topk.cu`, the TVM-FFI binding `flashinfer_sglang_dsv4_topk_binding.cu`, and
`gen_sglang_dsv4_topk_module` in `flashinfer/jit/topk.py` (one single-architecture build per
device capability, as upstream's `-DSGL_CUDA_ARCH` requires).
