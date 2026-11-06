# CAKE training backend names

`det` identifies run-to-run bitwise determinism of the delivered forward and
backward for fixed inputs, hardware, execution plan and initial accumulation
buffers. It does not promise identical bits across shapes, devices, reduction
plans or parallel partitions. `kcp` is the suffix for a context-parallel route.

| Training family | Canonical backend / route name |
| --- | --- |
| LM head + loss | `cake_lm_head_det_train` (deterministic: module `flashinfer.cake_lm_head_det_train`, backend `flashinfer.experimental.cake.cake_lm_head_det_train`); `cake.cake_lm_head_loss` (non-deterministic) |

## LM head + loss

The deterministic module is `flashinfer.cake_lm_head_det_train`; its backend
package is `flashinfer.experimental.cake.cake_lm_head_det_train`. No entry takes
a `deterministic` keyword, and `torch.use_deterministic_algorithms(True)` changes
nothing. The former names are removed without aliases: importing one fails.

| Name | Former names (removed, no alias) |
| --- | --- |
| module `flashinfer.cake_lm_head_det_train` | the deterministic entries of `flashinfer.chunked_lm_head` (that module keeps only the non-deterministic dispatchers) |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss` | `flashinfer.chunked_lm_head.chunked_lm_head_det_loss`; before the `det` naming `chunked_lm_head_loss` (now a non-deterministic dispatcher) |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob` | `flashinfer.chunked_lm_head.chunked_lm_head_det_logprob`; before the `det` naming `chunked_lm_head_logprob` |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload` | `flashinfer.chunked_lm_head.chunked_lm_head_det_preload`; before the `det` naming `chunked_lm_head_preload` |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob` | `flashinfer.chunked_lm_head.chunked_lm_head_det_support_logprob`; before the `det` naming `chunked_lm_head_support_logprob` |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl` | `flashinfer.chunked_lm_head.chunked_lm_head_det_support_kl`; before the `det` naming `chunked_lm_head_support_kl` |
| package `flashinfer.experimental.cake.cake_lm_head_det_train` | `flashinfer.experimental.cake_lm_head_det_loss`; before the `det` naming `flashinfer.experimental.cake_lm_head_loss` |
| `cake_backend.cake_lm_head_det_train_{loss,logprob,logprob_temperature}` (also in `cake_backend_opt_in`) | `chunked_lm_head_det_{loss,logprob,logprob_temperature}`; before the `det` naming `chunked_lm_head_{loss,logprob}` |
| `cake_backend.cake_lm_head_det_train_logprob_keep` | `chunked_lm_head_det_logprob_keep` |
| `cake_backend_opt_in.cake_lm_head_det_train_preload` | `preload` |
| `cake_backend.CakeLMHeadDet{Loss,Logprob,LogprobTemperature}Function` (also in `cake_backend_opt_in`) | `ChunkedLmHeadDet{Loss,Logprob,LogprobTemperature}Function`; before the `det` naming `ChunkedLmHead{Loss,Logprob}Function` |
| `cake_backend.CakeLMHeadDetLogprobKeepFunction` | `ChunkedLmHeadDetLogprobKeepFunction` |
| `cake_support.cake_lm_head_det_train_support_{logprob,kl}` | `chunked_lm_head_det_support_{logprob,kl}`; before the `det` naming `chunked_lm_head_support_{logprob,kl}` |
| `cake_support.CakeLMHeadDetSupportFunction` (serves both support entries) | `SupportDetLogprobFunction`; before the `det` naming `SupportLogprobFunction` |
| `cake_backend.prepare_lm_head_det_loss` (internal, kept) | `prepare_lm_head_loss` (in `cake_backend` and `cake_backend_opt_in`) |
| `cake_backend.validate_lm_head_det_inputs` (internal, kept) | `validate_lm_head_inputs` (in `cake_backend` and `cake_backend_opt_in`) |
| `cake_backend.lm_head_det_loss_workspace_size` (internal, kept) | `lm_head_loss_workspace_size` |
| `cake_backend.LmHeadDetLossRunner` (internal, kept) | `LmHeadLossRunner` (in `cake_backend` and `cake_backend_opt_in`) |
| `cake_jit.load_cake_lm_head_det_loss_module` (internal, kept) | `load_cake_lm_head_loss_module` (in `cake_jit`, `cake_backend` and `cake_backend_opt_in`) |

`cake_support.__all__` lists the two support entries, their autograd class and
the constants of their contract; its helpers stay module attributes.

The generated sources, programs, kernel symbols and libraries are named
`cake_lm_head_det_loss_<20 hex>` (`csrc/cake_lm_head_det_loss/`,
`kernel_cake_lm_head_det_loss_<20 hex>`, `cake_lm_head_det_loss_<20 hex>_<arch>`);
the 20-hex program digests are unchanged. The registry's binding profile
`abi = lm_head_loss_v1` keeps its name. Generated CUDA symbols and
content-addressed library identifiers are build identities; they do not
establish the operator's determinism contract.

The `FLASHINFER_CAKE_LM_HEAD_LOSS_*` environment variables keep their names.
Both LM-head facades read them through the private module
`flashinfer/_cake_lm_head_train_knobs.py`.

## LM head + loss: non-deterministic package

The non-deterministic package `flashinfer.experimental.cake.cake_lm_head_loss`
keeps every precision boundary of the deterministic package (BF16 logits and
`dlogits`, FP32 loss arithmetic, FP32 `dX` and cross-chunk `dW` accumulation,
one final cast each) and relaxes only run-to-run bitwise reproducibility: the
dX GEMM of a chunk of more than one K slice adds its partial tiles into the
FP32 `dX` accumulator in an unordered FP32 reduction (a one-slice chunk stores
its rows as the deterministic package does). `loss`, `logp` and `lse` keep
their fixed order. Its names are the deterministic names without `det`:

| Deterministic name | Non-deterministic name | The same name in the deterministic package |
| --- | --- | --- |
| package `flashinfer.experimental.cake.cake_lm_head_det_train` | package `flashinfer.experimental.cake.cake_lm_head_loss` | none |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_loss` | `flashinfer.chunked_lm_head.chunked_lm_head_loss(..., deterministic=False)` | `chunked_lm_head_loss` is this package's dispatcher; it needs `deterministic=False` |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_logprob` | `chunked_lm_head_logprob(..., deterministic=False)` | dispatcher; `deterministic=False` required |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_preload` | `chunked_lm_head_preload(..., deterministic=False)` | dispatcher; `deterministic=False` required |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_logprob` | `chunked_lm_head_support_logprob(..., deterministic=False)` | dispatcher; `deterministic=False` required |
| `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_support_kl` | `chunked_lm_head_support_kl(..., deterministic=False)` | dispatcher; `deterministic=False` required |
| `cake_backend.prepare_lm_head_det_loss` | `cake_backend.prepare_lm_head_loss` | removed, no alias |
| `cake_backend.validate_lm_head_det_inputs` | `cake_backend.validate_lm_head_inputs` | removed, no alias |
| `cake_backend.lm_head_det_loss_workspace_size` | `cake_backend.lm_head_loss_workspace_size` | removed, no alias |
| `cake_backend.LmHeadDetLossRunner` | `cake_backend.LmHeadLossRunner` | removed, no alias |
| `cake_backend.CakeLMHeadDet{Loss,Logprob}Function` | `cake_backend.ChunkedLmHead{Loss,Logprob}Function` | removed, no aliases |
| `cake_support.CakeLMHeadDetSupportFunction` | `cake_support.SupportLogprobFunction` | removed, no alias |
| `cake_jit.load_cake_lm_head_det_loss_module` | `cake_jit.load_cake_lm_head_loss_module` | removed, no alias |
| programs `cake_lm_head_det_loss_<20 hex>` | programs `cake_lm_head_loss_<20 hex>`, libraries `cake_lm_head_loss_<20 hex>_<arch>` | none |

The non-deterministic package defines these names in its own modules; the
deterministic modules define only the `det` names.

Mode and dispatch rules:

| Call | Result |
| --- | --- |
| `chunked_lm_head_{loss,logprob,preload,support_logprob,support_kl}(...)` with the default or any `deterministic` other than `False` (`True`, `None`, `0`, `1`, `"False"`) | `ValueError`, raised before any package import or torch query; it names `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<same suffix>` and says "pass deterministic=False for the non-deterministic path" |
| the same with `deterministic=False` | the non-deterministic package, after the same argument checks and with the same `FLASHINFER_CAKE_LM_HEAD_LOSS_*` fallbacks |
| the same with `deterministic=False` while `torch.use_deterministic_algorithms(True)` is in effect (also `warn_only=True`) | the non-deterministic package; its alert decides (the row below); no fallback to the deterministic entry; a `k_slices` pin reaches it unchanged, and the dispatcher itself warns nothing |
| `cake_lm_head_det_train_{loss,logprob,preload,support_logprob,support_kl}(..., deterministic=<any>)`; likewise every function and autograd `Function` of the deterministic package | `TypeError` (none has a `deterministic` parameter); torch's deterministic mode changes nothing |
| a non-deterministic package entry (`flashinfer.experimental.cake.cake_lm_head_loss`) with `deterministic` other than `False` (its default `True` included) | `ValueError` before any launch, naming the deterministic entry `flashinfer.cake_lm_head_det_train.cake_lm_head_det_train_<suffix>` and saying "pass deterministic=False for the non-deterministic path"; `backward_loss` and `ForwardResult.backward` take no `deterministic` (they continue a `forward_loss` call) |
| a non-deterministic package entry while `torch.use_deterministic_algorithms(True)` is in effect | PyTorch's alert (`RuntimeError`, or a `UserWarning` with `warn_only=True`, after which the call runs) only right before a K-sliced dX launch: a `gemm_dx_red` chunk of more than one dX slice (the counted two-slice chunks of the zero-gradient row skip included) or the candidate-support backward's dense dX at more than one slice (`sm_100a` cells only); fixed-order launches do not alert (one-slice dX, the weight-gradient GEMMs in chunk order, the deferred last-chunk weight gradient, the budget recompute, the casts, the support sparse passes, forward-only calls) |

Zero-weight policy rows (`objective="policy"`, a valid label and
`loss_weights == 0`): the non-deterministic loss entry leaves them out of the
gradient path, as it does ignored rows.

| | `cake_lm_head_det_train_loss` | `chunked_lm_head_loss(..., deterministic=False)` |
| --- | --- | --- |
| gradient rows of the chunk loop | every valid row | `(labels >= 0) & (loss_weights != 0)` (a NaN weight stays in) |
| `logp` of a zero-weight row (`return_logp=True`) | the loss path | a logp-only pass of the log-probability forward kernels |
| `loss`, `dX`, `dW` with finite inputs | the same mathematics | the same mathematics; `dW` and the loss sum group over other rows (bits may differ within the accuracy budgets) |
| a non-finite `X` row, logits or `infer_logp` in a zero-weight row | NaN through `0 * inf` | no effect on `loss` / `dX` / `dW` |
| host synchronizations | one count readback | one count readback (both counts in it) |
| not applied | -- | a call of fewer than `cake_backend.zero_weight_min_rows()` rows (default 4096), `compact_rows=False`, `objective="ce"`, the log-probability entry, the prepared runner, a caller-given `num_valid_rows`; `memory_mode="budget"` without `num_valid_rows` splits the rows (its bound still counts every valid row) |

Candidate-support entries (`chunked_lm_head_support_logprob` /
`chunked_lm_head_support_kl`):

| | `cake_lm_head_det_train_support_*` | `chunked_lm_head_support_*(..., deterministic=False)` |
| --- | --- | --- |
| outputs, sparse `dX`, `dW` | the deterministic programs | the same programs: the same bits |
| dense `dX` | per vocabulary block an FP32 product block, added into the chunk sum in block order | each block's GEMM adds into the chunk sum from its epilogue: one K slice (the same bits) at every shape outside the measured cells of `support_dx_form`; two or three K slices inside them (B200 only: five classes of H, token chunk and vocabulary block, at H 2048 to 6144): an unordered FP32 sum within the accuracy budgets |
| global sparse `dW` zero fill | BF16 fill | FP32-word fill: the same bits |
| sparse `dW` long segments | parts summed in part order | the same |

The non-deterministic package adds no environment variable; the
`FLASHINFER_CAKE_LM_HEAD_LOSS_*` variables apply to the `det` entries and to
the dispatchers, and its binding caches take the same two binding-cache settings.
