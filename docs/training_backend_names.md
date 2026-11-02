# CAKE training backend names

`det` identifies run-to-run bitwise determinism of the delivered forward and
backward for fixed inputs, hardware, execution plan and initial accumulation
buffers. It does not promise identical bits across shapes, devices, reduction
plans or parallel partitions. `kcp` is the suffix for a context-parallel route.

| Training family | Canonical backend / route name |
| --- | --- |
| LM head + loss | `cake_lm_head_det_loss` (deterministic); `cake.cake_lm_head_loss` (non-deterministic) |

## LM head + loss

| Canonical name | Name before the `det` naming (alias) |
| --- | --- |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_loss` | `chunked_lm_head_loss` (now a dispatcher, see below) |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_logprob` | `chunked_lm_head_logprob` (now a dispatcher) |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_preload` | `chunked_lm_head_preload` (now a dispatcher) |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_logprob` | `chunked_lm_head_support_logprob` (now a dispatcher) |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_kl` | `chunked_lm_head_support_kl` (now a dispatcher) |
| package `flashinfer.experimental.cake_lm_head_det_loss` | `flashinfer.experimental.cake_lm_head_loss` (each module is the same module object) |
| `cake_backend.prepare_lm_head_det_loss` | `prepare_lm_head_loss` |
| `cake_backend.validate_lm_head_det_inputs` | `validate_lm_head_inputs` |
| `cake_backend.lm_head_det_loss_workspace_size` | `lm_head_loss_workspace_size` |
| `cake_backend.LmHeadDetLossRunner` | `LmHeadLossRunner` |
| `cake_backend.ChunkedLmHeadDetLossFunction` | `ChunkedLmHeadLossFunction` |
| `cake_backend.ChunkedLmHeadDetLogprobFunction` | `ChunkedLmHeadLogprobFunction` |
| `cake_support.SupportDetLogprobFunction` | `SupportLogprobFunction` |
| `cake_jit.load_cake_lm_head_det_loss_module` | `load_cake_lm_head_loss_module` (in `cake_jit`, `cake_backend` and `cake_backend_opt_in`) |

The generated sources, programs, kernel symbols and libraries are named
`cake_lm_head_det_loss_<20 hex>` (`csrc/cake_lm_head_det_loss/`,
`kernel_cake_lm_head_det_loss_<20 hex>`, `cake_lm_head_det_loss_<20 hex>_<arch>`);
the 20-hex program digests are unchanged. The `FLASHINFER_CAKE_LM_HEAD_LOSS_*`
environment variables and the registry's binding profile `abi = lm_head_loss_v1`
keep their names. Generated CUDA symbols and content-addressed library
identifiers are build identities; they do not establish the operator's
determinism contract.

## LM head + loss: non-deterministic package

The non-deterministic package `flashinfer.experimental.cake.cake_lm_head_loss`
keeps every precision boundary of the deterministic package (BF16 logits and
`dlogits`, FP32 loss arithmetic, FP32 `dX` and cross-chunk `dW` accumulation,
one final cast each) and relaxes only run-to-run bitwise reproducibility: the
dX GEMM of a chunk of more than one K slice adds its partial tiles into the
FP32 `dX` accumulator in an unordered FP32 reduction (a one-slice chunk stores
its rows as the deterministic package does). `loss`, `logp` and `lse` keep
their fixed order. Its names are the deterministic names without `det`:

| Deterministic name | Non-deterministic name | Former alias |
| --- | --- | --- |
| package `flashinfer.experimental.cake_lm_head_det_loss` | package `flashinfer.experimental.cake.cake_lm_head_loss` | `flashinfer.experimental.cake_lm_head_loss` stays the deterministic package |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_loss` | `flashinfer.chunked_lm_head.chunked_lm_head_loss(..., deterministic=False)` | `chunked_lm_head_loss` is a dispatcher, default `deterministic=True` |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_logprob` | `chunked_lm_head_logprob(..., deterministic=False)` | dispatcher |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_preload` | `chunked_lm_head_preload(..., deterministic=False)` | dispatcher |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_logprob` | `chunked_lm_head_support_logprob(..., deterministic=False)` | dispatcher (the keyword is new) |
| `flashinfer.chunked_lm_head.chunked_lm_head_det_support_kl` | `chunked_lm_head_support_kl(..., deterministic=False)` | dispatcher (the keyword is new) |
| `cake_backend.prepare_lm_head_det_loss` | `cake_backend.prepare_lm_head_loss` | deterministic module alias, unchanged |
| `cake_backend.validate_lm_head_det_inputs` | `cake_backend.validate_lm_head_inputs` | deterministic module alias, unchanged |
| `cake_backend.lm_head_det_loss_workspace_size` | `cake_backend.lm_head_loss_workspace_size` | deterministic module alias, unchanged |
| `cake_backend.LmHeadDetLossRunner` | `cake_backend.LmHeadLossRunner` | deterministic module alias, unchanged |
| `cake_backend.ChunkedLmHeadDet{Loss,Logprob}Function` | `cake_backend.ChunkedLmHead{Loss,Logprob}Function` | deterministic module aliases, unchanged |
| `cake_support.SupportDetLogprobFunction` | `cake_support.SupportLogprobFunction` | deterministic module alias, unchanged |
| `cake_jit.load_cake_lm_head_det_loss_module` | `cake_jit.load_cake_lm_head_loss_module` | deterministic module alias, unchanged |
| programs `cake_lm_head_det_loss_<20 hex>` | programs `cake_lm_head_loss_<20 hex>`, libraries `cake_lm_head_loss_<20 hex>_<arch>` | none |

The two packages define the same backend names in different modules; the
deterministic modules' aliases of these names are the deterministic objects.

Mode and dispatch rules:

| Call | Runs |
| --- | --- |
| `chunked_lm_head_*(...)` or `deterministic=True` | the `chunked_lm_head_det_*` entry with the same arguments: the same bits, no added allocation or synchronization |
| `deterministic=False` | the non-deterministic package, after the same argument checks and with the same `FLASHINFER_CAKE_LM_HEAD_LOSS_*` fallbacks |
| `deterministic=False` while `torch.use_deterministic_algorithms(True)` is in effect (also `warn_only=True`) | the `chunked_lm_head_det_*` entry, with one `UserWarning` per entry point |
| `deterministic` not a `bool` | `TypeError` |
| `chunked_lm_head_det_*(..., deterministic=False)` | `NotImplementedError` (unchanged) |
| a non-deterministic package entry with `deterministic` other than `False` (its default `True` included) | `ValueError` naming the deterministic package |
| a non-deterministic package entry that launches gradient GEMMs while `torch.use_deterministic_algorithms(True)` is in effect | PyTorch's alert: `RuntimeError`, or a `UserWarning` with `warn_only=True` |

Zero-weight policy rows (`objective="policy"`, a valid label and
`loss_weights == 0`): the non-deterministic loss entry leaves them out of the
gradient path, as it does ignored rows.

| | `deterministic=True` | `deterministic=False` |
| --- | --- | --- |
| gradient rows of the chunk loop | every valid row | `(labels >= 0) & (loss_weights != 0)` (a NaN weight stays in) |
| `logp` of a zero-weight row (`return_logp=True`) | the loss path | a logp-only pass of the log-probability forward kernels |
| `loss`, `dX`, `dW` with finite inputs | the same mathematics | the same mathematics; `dW` and the loss sum group over other rows (bits may differ within the accuracy budgets) |
| a non-finite `X` row, logits or `infer_logp` in a zero-weight row | NaN through `0 * inf` | no effect on `loss` / `dX` / `dW` |
| host synchronizations | one count readback | one count readback (both counts in it) |
| not applied | -- | a call of fewer than `cake_backend.zero_weight_min_rows()` rows (default 4096), `compact_rows=False`, `objective="ce"`, the log-probability entry, the prepared runner, a caller-given `num_valid_rows`; `memory_mode="budget"` without `num_valid_rows` splits the rows (its bound still counts every valid row) |

Candidate-support entries (`chunked_lm_head_support_logprob` /
`chunked_lm_head_support_kl`):

| | `deterministic=True` | `deterministic=False` |
| --- | --- | --- |
| outputs, sparse `dX`, `dW` | the deterministic programs | the same programs: the same bits |
| dense `dX` | per vocabulary block an FP32 product block, added into the chunk sum in block order | each block's GEMM adds into the chunk sum from its epilogue: one K slice (the same bits) at every shape outside the measured cells of `support_dx_form`; two or three K slices inside them (B200 only: five classes of H, token chunk and vocabulary block, at H 2048 to 6144): an unordered FP32 sum within the accuracy budgets |
| global sparse `dW` zero fill | BF16 fill | FP32-word fill: the same bits |
| sparse `dW` long segments | parts summed in part order | the same |

The non-deterministic package adds no environment variable; the
`FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables apply to both modes through the
dispatchers, and its binding caches take the same two binding-cache settings.
