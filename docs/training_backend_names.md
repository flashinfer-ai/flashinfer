# CAKE training backend names

`det` identifies run-to-run bitwise determinism of the delivered forward and
backward for fixed inputs, hardware, execution plan and initial accumulation
buffers. It does not promise identical bits across shapes, devices, reduction
plans or parallel partitions. `kcp` is the suffix for a context-parallel route.

| Training family | Canonical backend / route name |
| --- | --- |
| LM head + loss | `cake_lm_head_det_loss` |

## LM head + loss

| Canonical name | Name before the `det` naming (alias) |
| --- | --- |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_loss` | `chunked_lm_head_loss` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_logprob` | `chunked_lm_head_logprob` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_preload` | `chunked_lm_head_preload` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_logprob` | `chunked_lm_head_support_logprob` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_kl` | `chunked_lm_head_support_kl` |
| package `flashinfer.experimental.cake_lm_head_det_loss` | `flashinfer.experimental.cake_lm_head_loss` (each module is the same module object) |
| `cake_backend.prepare_lm_head_det_loss` | `prepare_lm_head_loss` |
| `cake_backend.validate_lm_head_det_inputs` | `validate_lm_head_inputs` |
| `cake_backend.lm_head_det_loss_workspace_size` | `lm_head_loss_workspace_size` |
| `cake_backend.LmHeadDetLossRunner` | `LmHeadLossRunner` |
| `cake_backend.ChunkedLmHeadDetLossFunction` | `ChunkedLmHeadLossFunction` |
| `cake_backend.ChunkedLmHeadDetLogprobFunction` | `ChunkedLmHeadLogprobFunction` |
| `cake_support.SupportDetLogprobFunction` | `SupportLogprobFunction` |
| `cake_jit.load_cake_lm_head_det_loss_module` | `load_cake_lm_head_loss_module` |

The generated sources, programs, kernel symbols and libraries are named
`cake_lm_head_det_loss_<20 hex>` (`csrc/cake_lm_head_det_loss/`,
`kernel_cake_lm_head_det_loss_<20 hex>`, `cake_lm_head_det_loss_<20 hex>_<arch>`);
the 20-hex program digests are unchanged. The `FLASHINFER_CAKE_LM_HEAD_LOSS_*`
environment variables and the registry's binding profile `abi = lm_head_loss_v1`
keep their names. Generated CUDA symbols and content-addressed library
identifiers are build identities; they do not establish the operator's
determinism contract.
