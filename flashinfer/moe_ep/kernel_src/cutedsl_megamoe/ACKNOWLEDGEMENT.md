# Acknowledgement

The CuTeDSL MegaMoE kernels vendored under `src/` — the fused
dispatch + FC1 + SwiGLU + FC2 + combine persistent kernels for NVFP4 and
MXFP8 expert-parallel MoE on Blackwell, together with their token-comm
primitives, epilogues, tester, and reference implementations — are the
work of the NVIDIA CuTeDSL MegaMoE kernel team:

- Bangyu Shen ([@shubaoyu2](https://github.com/shubaoyu2), bangyus@nvidia.com)
- Aragorn Guan ([@aragorn-guan](https://github.com/aragorn-guan), aragorng@nvidia.com)
- Yuhan Li ([@liyuhannnnn](https://github.com/liyuhannnnn), alel@nvidia.com)
- Hao Sheng ([@shdetect](https://github.com/shdetect), hsheng@nvidia.com)
- Jintao Peng ([@JintaoPengCS](https://github.com/JintaoPengCS), jintaop@nvidia.com)

FlashInfer's `moe_ep` integration (the `shim/` adapters, symmetric-buffer
API, tuner/autotuner, and tests in this tree) builds directly on their
kernel drops and on the tuning insights encoded in their tester and
solver sweeps — including the in-flight top-k reduce and the quantized
combine wire formats measured in `TUNING.md`.

Thank you to the kernel authors for the kernels themselves, the rigorous
validation harness that ships with them (the form-B K!-ordering check in
particular), and for keeping the drops reproducible enough to re-vendor
and audit on every update (see `SKILL.md` for that workflow).
