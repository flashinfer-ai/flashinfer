# CAKE training backend names

`det` identifies run-to-run bitwise determinism of the delivered forward and
backward for fixed inputs, hardware, execution plan and initial accumulation
buffers. It does not promise identical bits across shapes, devices, reduction
plans or parallel partitions. `kcp` is the suffix for a context-parallel route;
local kernels that can consume a CP window do not thereby implement KCP.

| Training family | Canonical backend / route name |
| --- | --- |
| GDN, single device | `cake_gdn_det_train` |
| GDN, KCP | `cake_gdn_det_train_kcp` |
| KDA, single device | `cake_kda_det_train` |
| KDA, KCP | `cake_kda_det_train_kcp` |
| Causal conv1d, single device | `cake_causal_conv1d_det_train` |
| Causal conv1d, KCP | `cake_causal_conv1d_det_train_kcp` |
| RMSNorm | `cake_rmsnorm_det_train` |
| RMSNorm + SiLU gate | `cake_rmsnorm_silu_gate_det_train` |
| mHC, including its head | `cake_mhc_det_train` |
| LM head + loss | `cake_lm_head_det_loss` |
| MoE grouped GEMM | `cake_moe_grouped_det_gemm` |
| Dense projection GEMM | `dense_projection_det_gemm` |
| GQA | `cake_gqa_train` (backward atomic reductions) |
| DSA, including NoPE | `cake_dsa_train` (backward atomic reductions) |
| MoK BF16 / MXFP8 | `cake_mok_bf16` (full distributed determinism not established) |

Backend implementation packages in core add `_backend` to their route name.
The explicit KCP entry modules are `flashinfer.cake_gdn_det_train_kcp`,
`flashinfer.cake_kda_det_train_kcp`, and
`flashinfer.cake_causal_conv1d_det_train_kcp`. They share the existing operations,
classes and implementation state. Operation signatures remain unchanged.

Legacy import paths forward to the canonical implementation. Generated CUDA
symbols and content-addressed library identifiers are build identities; they do
not establish the operator's determinism contract.
